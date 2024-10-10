import Mathlib

namespace solution_set_correct_l3658_365839

def solution_set : Set ℝ := {1, 2, 3, 4, 5}

def equation (x : ℝ) : Prop :=
  (x^2 - 5*x + 5)^(x^2 - 9*x + 20) = 1

theorem solution_set_correct :
  ∀ x : ℝ, equation x ↔ x ∈ solution_set := by sorry

end solution_set_correct_l3658_365839


namespace system_solution_l3658_365842

theorem system_solution :
  let solutions : List (ℝ × ℝ × ℝ) := [
    (1, -1, 1), (1, 3/2, -2/3), (-2, 1/2, 1), (-2, 3/2, 1/3), (3, -1, 1/3), (3, 1/2, -2/3)
  ]
  ∀ (x y z : ℝ),
    (x + 2*y + 3*z = 2 ∧
     1/x + 1/(2*y) + 1/(3*z) = 5/6 ∧
     x*y*z = -1) ↔
    (x, y, z) ∈ solutions := by
  sorry

end system_solution_l3658_365842


namespace neg_p_sufficient_not_necessary_l3658_365836

-- Define the conditions
def p (x : ℝ) : Prop := x ≤ 1
def q (x : ℝ) : Prop := 1 / x < 1

-- Statement to prove
theorem neg_p_sufficient_not_necessary :
  (∀ x : ℝ, ¬(p x) → q x) ∧ ¬(∀ x : ℝ, q x → ¬(p x)) :=
sorry

end neg_p_sufficient_not_necessary_l3658_365836


namespace garden_dimensions_l3658_365827

theorem garden_dimensions :
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 
      p.2 > p.1 ∧ 
      (p.1 - 6) * (p.2 - 6) = 12 ∧ 
      p.1 ≥ 7 ∧ p.2 ≥ 7)
    (Finset.product (Finset.range 100) (Finset.range 100))).card ∧ 
  n = 3 :=
by
  sorry

end garden_dimensions_l3658_365827


namespace possible_solutions_l3658_365892

theorem possible_solutions (a b : ℝ) (h1 : a + 1 > b) (h2 : b > 2/a) (h3 : 2/a > 0) :
  (∃ a₀, a₀ = 2 ∧ a₀ + 1 > 2/a₀ ∧ 2/a₀ > 0) ∧
  (∃ b₀, b₀ = 1 ∧ (∃ a₁, a₁ + 1 > b₀ ∧ b₀ > 2/a₁ ∧ 2/a₁ > 0)) :=
sorry

end possible_solutions_l3658_365892


namespace race_head_start_l3658_365845

/-- Represents the race scenario where A runs twice as fast as B -/
structure Race where
  speed_a : ℝ
  speed_b : ℝ
  course_length : ℝ
  head_start : ℝ
  speed_ratio : speed_a = 2 * speed_b
  course_length_value : course_length = 142

/-- Theorem stating that for the given race conditions, the head start must be 71 meters -/
theorem race_head_start (r : Race) : r.head_start = 71 := by
  sorry

#check race_head_start

end race_head_start_l3658_365845


namespace lcm_180_560_l3658_365829

theorem lcm_180_560 : Nat.lcm 180 560 = 5040 := by
  sorry

end lcm_180_560_l3658_365829


namespace circle_radius_from_area_l3658_365858

theorem circle_radius_from_area (A : ℝ) (r : ℝ) (h : A = 100 * Real.pi) :
  A = Real.pi * r^2 → r = 10 := by
  sorry

end circle_radius_from_area_l3658_365858


namespace number_ratio_problem_l3658_365895

theorem number_ratio_problem (x : ℝ) : 3 * (2 * x + 5) = 111 → x / (2 * x) = 1 / 2 := by
  sorry

end number_ratio_problem_l3658_365895


namespace snowball_distribution_l3658_365821

/-- Represents the number of snowballs each person has -/
structure Snowballs :=
  (charlie : ℕ)
  (lucy : ℕ)
  (linus : ℕ)

/-- The initial state of snowballs -/
def initial_state : Snowballs :=
  { charlie := 19 + 31,  -- Lucy's snowballs + difference
    lucy := 19,
    linus := 0 }

/-- The final state after Charlie gives half his snowballs to Linus -/
def final_state : Snowballs :=
  { charlie := (19 + 31) / 2,
    lucy := 19,
    linus := (19 + 31) / 2 }

/-- Theorem stating the correct distribution of snowballs after sharing -/
theorem snowball_distribution :
  final_state.charlie = 25 ∧
  final_state.lucy = 19 ∧
  final_state.linus = 25 :=
by sorry

end snowball_distribution_l3658_365821


namespace triangle_translation_l3658_365813

structure Point where
  x : ℝ
  y : ℝ

def translate (p : Point) (dx dy : ℝ) : Point :=
  ⟨p.x + dx, p.y + dy⟩

theorem triangle_translation :
  let A : Point := ⟨2, 1⟩
  let B : Point := ⟨4, 3⟩
  let C : Point := ⟨0, 2⟩
  let A' : Point := ⟨-1, 5⟩
  let dx : ℝ := A'.x - A.x
  let dy : ℝ := A'.y - A.y
  let C' : Point := translate C dx dy
  C'.x = -3 ∧ C'.y = 6 :=
by sorry

end triangle_translation_l3658_365813


namespace sum_vertices_penta_hexa_prism_is_22_l3658_365879

/-- The number of vertices in a polygon -/
def vertices_in_polygon (sides : ℕ) : ℕ := sides

/-- The number of vertices in a prism with polygonal bases -/
def vertices_in_prism (base_vertices : ℕ) : ℕ := 2 * base_vertices

/-- The sum of vertices in a pentagonal prism and a hexagonal prism -/
def sum_vertices_penta_hexa_prism : ℕ :=
  vertices_in_prism (vertices_in_polygon 5) + vertices_in_prism (vertices_in_polygon 6)

theorem sum_vertices_penta_hexa_prism_is_22 :
  sum_vertices_penta_hexa_prism = 22 := by
  sorry

end sum_vertices_penta_hexa_prism_is_22_l3658_365879


namespace prob_sum_five_twice_l3658_365870

/-- A die with 4 sides numbered 1 to 4. -/
def FourSidedDie : Finset ℕ := {1, 2, 3, 4}

/-- The set of all possible outcomes when rolling two 4-sided dice. -/
def TwoDiceOutcomes : Finset (ℕ × ℕ) :=
  FourSidedDie.product FourSidedDie

/-- The sum of two dice rolls. -/
def diceSum (roll : ℕ × ℕ) : ℕ := roll.1 + roll.2

/-- The set of all rolls that sum to 5. -/
def sumFiveOutcomes : Finset (ℕ × ℕ) :=
  TwoDiceOutcomes.filter (λ roll => diceSum roll = 5)

/-- The probability of rolling a sum of 5 with two 4-sided dice. -/
def probSumFive : ℚ :=
  (sumFiveOutcomes.card : ℚ) / (TwoDiceOutcomes.card : ℚ)

theorem prob_sum_five_twice :
  probSumFive * probSumFive = 1 / 16 := by
  sorry

end prob_sum_five_twice_l3658_365870


namespace complement_A_intersect_B_l3658_365826

-- Define the sets A and B
def A : Set ℝ := {x | x ≥ 3 ∨ x ≤ 1}
def B : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem complement_A_intersect_B :
  (Set.univ \ A) ∩ B = Set.Ioo 2 3 := by sorry

end complement_A_intersect_B_l3658_365826


namespace solve_sales_problem_l3658_365834

def sales_problem (m1 m2 m4 m5 m6 : ℕ) (average : ℚ) : Prop :=
  ∃ m3 : ℕ,
    (m1 + m2 + m3 + m4 + m5 + m6 : ℚ) / 6 = average ∧
    m3 = 5207

theorem solve_sales_problem :
  sales_problem 5124 5366 5399 6124 4579 (5400 : ℚ) :=
sorry

end solve_sales_problem_l3658_365834


namespace identity_polynomial_form_l3658_365828

/-- A polynomial that satisfies the given identity. -/
def IdentityPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x * P (x - 1) = (x - 2) * P x

/-- The theorem stating the form of polynomials satisfying the identity. -/
theorem identity_polynomial_form (P : ℝ → ℝ) (h : IdentityPolynomial P) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * (x^2 - x) :=
by
  sorry

end identity_polynomial_form_l3658_365828


namespace ben_win_probability_l3658_365833

theorem ben_win_probability (lose_prob : ℚ) (h1 : lose_prob = 5/8) (h2 : lose_prob + win_prob = 1) : win_prob = 3/8 := by
  sorry

end ben_win_probability_l3658_365833


namespace water_intersection_points_l3658_365815

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  edgeLength : ℝ

/-- Represents the water level in the cube -/
def waterLevel (c : Cube) (vol : ℝ) : ℝ :=
  vol * c.edgeLength

theorem water_intersection_points (c : Cube) (waterVol : ℝ) :
  c.edgeLength = 1 →
  waterVol = 5/6 →
  ∃ (x : ℝ), 
    0.26 < x ∧ x < 0.28 ∧ 
    0.72 < (1 - x) ∧ (1 - x) < 0.74 ∧
    (waterLevel c waterVol = x ∨ waterLevel c waterVol = 1 - x) := by
  sorry

#check water_intersection_points

end water_intersection_points_l3658_365815


namespace min_gumballs_for_four_same_color_l3658_365832

/-- Represents the number of gumballs of each color in the machine -/
structure GumballMachine :=
  (red : ℕ)
  (yellow : ℕ)
  (white : ℕ)
  (green : ℕ)

/-- The minimum number of gumballs needed to ensure obtaining four of the same color -/
def minGumballsForFourSameColor (machine : GumballMachine) : ℕ :=
  13

/-- Theorem stating that for the given gumball machine, 
    the minimum number of gumballs needed to ensure 
    obtaining four of the same color is 13 -/
theorem min_gumballs_for_four_same_color 
  (machine : GumballMachine) 
  (h1 : machine.red = 10)
  (h2 : machine.yellow = 6)
  (h3 : machine.white = 8)
  (h4 : machine.green = 9) :
  minGumballsForFourSameColor machine = 13 :=
by
  sorry


end min_gumballs_for_four_same_color_l3658_365832


namespace sum_P_equals_97335_l3658_365867

/-- P(n) represents the product of all non-zero digits of a positive integer n -/
def P (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n < 10 then n
  else let d := n % 10
       let r := n / 10
       if d = 0 then P r
       else d * P r

/-- The sum of P(n) for n from 1 to 999 -/
def sum_P : ℕ := (List.range 999).map (fun i => P (i + 1)) |>.sum

theorem sum_P_equals_97335 : sum_P = 97335 := by
  sorry

end sum_P_equals_97335_l3658_365867


namespace contractor_payment_result_l3658_365819

def contractor_payment (total_days : ℕ) (working_pay : ℚ) (absence_fine : ℚ) (absent_days : ℕ) : ℚ :=
  let working_days := total_days - absent_days
  let total_earnings := working_days * working_pay
  let total_fines := absent_days * absence_fine
  total_earnings - total_fines

theorem contractor_payment_result :
  contractor_payment 30 25 7.5 12 = 360 := by
  sorry

end contractor_payment_result_l3658_365819


namespace triangle_side_lengths_l3658_365887

/-- Given a triangle with sides in ratio 3:4:5 and perimeter 60, prove its side lengths are 15, 20, and 25 -/
theorem triangle_side_lengths (a b c : ℝ) (h_ratio : a / b = 3 / 4 ∧ b / c = 4 / 5) 
  (h_perimeter : a + b + c = 60) : a = 15 ∧ b = 20 ∧ c = 25 := by
  sorry

end triangle_side_lengths_l3658_365887


namespace product_of_imaginary_parts_l3658_365853

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_eq (z : ℂ) : Prop := z^2 + 3*z + (4 - 7*i) = 0

-- Define a function to get the imaginary part of a complex number
def im (z : ℂ) : ℝ := z.im

-- Theorem statement
theorem product_of_imaginary_parts :
  ∃ (z1 z2 : ℂ), quadratic_eq z1 ∧ quadratic_eq z2 ∧ z1 ≠ z2 ∧ (im z1 * im z2 = -14) :=
sorry

end product_of_imaginary_parts_l3658_365853


namespace valid_schedule_count_is_twelve_l3658_365830

/-- Represents the four subjects in the class schedule -/
inductive Subject
| Chinese
| Mathematics
| English
| PhysicalEducation

/-- Represents a schedule of four periods -/
def Schedule := Fin 4 → Subject

/-- Checks if a schedule is valid (PE is not in first or fourth period) -/
def isValidSchedule (s : Schedule) : Prop :=
  s 0 ≠ Subject.PhysicalEducation ∧ s 3 ≠ Subject.PhysicalEducation

/-- The number of valid schedules -/
def validScheduleCount : ℕ := sorry

/-- Theorem stating that the number of valid schedules is 12 -/
theorem valid_schedule_count_is_twelve : validScheduleCount = 12 := by sorry

end valid_schedule_count_is_twelve_l3658_365830


namespace triangle_count_l3658_365838

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of points on the circle -/
def num_points : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def vertices_per_triangle : ℕ := 3

theorem triangle_count :
  choose num_points vertices_per_triangle = 120 := by
  sorry

end triangle_count_l3658_365838


namespace orange_juice_mixture_fraction_l3658_365862

/-- Represents the fraction of orange juice in a mixture -/
def orange_juice_fraction (pitcher1_capacity pitcher2_capacity : ℚ)
  (pitcher1_oj_fraction pitcher2_oj_fraction : ℚ) : ℚ :=
  let total_oj := pitcher1_capacity * pitcher1_oj_fraction + pitcher2_capacity * pitcher2_oj_fraction
  let total_volume := pitcher1_capacity + pitcher2_capacity
  total_oj / total_volume

/-- Proves that the fraction of orange juice in the combined mixture is 29167/100000 -/
theorem orange_juice_mixture_fraction :
  orange_juice_fraction 800 800 (1/4) (1/3) = 29167/100000 := by
  sorry

end orange_juice_mixture_fraction_l3658_365862


namespace sum_a_b_is_one_third_l3658_365814

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The theorem stating that a + b = 1/3 given the conditions -/
theorem sum_a_b_is_one_third
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, f x = a * x^2 + b * x + 3 * a + b)
  (h2 : IsEven f)
  (h3 : Set.Icc (a - 1) (2 * a) = Set.range f) :
  a + b = 1 / 3 := by
  sorry

end sum_a_b_is_one_third_l3658_365814


namespace inequality_proof_l3658_365800

theorem inequality_proof (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) :
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) > 2 ∧
  ∀ m : ℝ, (∀ a b c d e : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → e > 0 →
    Real.sqrt (a / (b + c + d + e)) + 
    Real.sqrt (b / (a + c + d + e)) + 
    Real.sqrt (c / (a + b + d + e)) + 
    Real.sqrt (d / (a + b + c + e)) + 
    Real.sqrt (e / (a + b + c + d)) > m) → 
  m ≤ 2 :=
by sorry

end inequality_proof_l3658_365800


namespace sphere_in_cone_l3658_365891

theorem sphere_in_cone (b d : ℝ) : 
  let cone_base_radius : ℝ := 15
  let cone_height : ℝ := 30
  let sphere_radius : ℝ := b * Real.sqrt d - b
  sphere_radius = (cone_base_radius * cone_height) / (Real.sqrt (cone_base_radius^2 + cone_height^2) + cone_height) →
  b + d = 12.5 :=
by sorry

end sphere_in_cone_l3658_365891


namespace triangle_inequalities_l3658_365852

theorem triangle_inequalities (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a*(b-c)^2 + b*(c-a)^2 + c*(a-b)^2 + 4*a*b*c > a^3 + b^3 + c^3) ∧
  (2*a^2*b^2 + 2*b^2*c^2 + 2*c^2*a^2 > a^4 + b^4 + c^4) ∧
  (2*a*b + 2*b*c + 2*c*a > a^2 + b^2 + c^2) := by
  sorry

end triangle_inequalities_l3658_365852


namespace mahi_share_l3658_365806

structure Friend where
  name : String
  age : ℕ
  distance : ℕ
  removed_amount : ℚ
  ratio : ℕ

def total_amount : ℚ := 2200

def friends : List Friend := [
  ⟨"Neha", 25, 5, 5, 2⟩,
  ⟨"Sabi", 32, 8, 8, 8⟩,
  ⟨"Mahi", 30, 7, 4, 6⟩,
  ⟨"Ravi", 28, 10, 6, 4⟩,
  ⟨"Priya", 35, 4, 10, 10⟩
]

def distance_bonus : ℚ := 10

theorem mahi_share (mahi : Friend) 
  (h1 : mahi ∈ friends)
  (h2 : mahi.name = "Mahi")
  (h3 : ∀ f : Friend, f ∈ friends → 
    f.age * (mahi.ratio * (total_amount - (friends.map Friend.removed_amount).sum) / (friends.map Friend.ratio).sum + mahi.removed_amount + mahi.distance * distance_bonus) = 
    mahi.age * (f.ratio * (total_amount - (friends.map Friend.removed_amount).sum) / (friends.map Friend.ratio).sum + f.removed_amount + f.distance * distance_bonus)) :
  mahi.ratio * (total_amount - (friends.map Friend.removed_amount).sum) / (friends.map Friend.ratio).sum + mahi.removed_amount + mahi.distance * distance_bonus = 507.38 := by
  sorry

end mahi_share_l3658_365806


namespace sample_size_comparison_l3658_365893

theorem sample_size_comparison (n m : ℕ+) (x_bar y_bar z a : ℝ) :
  x_bar ≠ y_bar →
  0 < a →
  a < 1/2 →
  z = a * x_bar + (1 - a) * y_bar →
  n > m :=
sorry

end sample_size_comparison_l3658_365893


namespace expand_expression_l3658_365868

theorem expand_expression (x : ℝ) : 5 * (x + 3) * (2 * x - 4) = 10 * x^2 + 10 * x - 60 := by
  sorry

end expand_expression_l3658_365868


namespace solve_linear_equation_l3658_365854

theorem solve_linear_equation (x : ℝ) : 3 * x + 7 = -2 → x = -3 := by
  sorry

end solve_linear_equation_l3658_365854


namespace perpendicular_bisector_and_parallel_line_l3658_365817

-- Define points A, B, and P
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Define the perpendicular bisector equation
def perpendicular_bisector (x y : ℝ) : Prop :=
  3 * x - 4 * y - 23 = 0

-- Define the parallel line equation
def parallel_line (x y : ℝ) : Prop :=
  4 * x + 3 * y + 1 = 0

-- Theorem statement
theorem perpendicular_bisector_and_parallel_line :
  (∀ x y : ℝ, perpendicular_bisector x y ↔ 
    (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1) ∧
    (x - (A.1 + B.1) / 2) ^ 2 + (y - (A.2 + B.2) / 2) ^ 2 = 
    ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) / 4) ∧
  (∀ x y : ℝ, parallel_line x y ↔
    (y - P.2) * (B.1 - A.1) = (x - P.1) * (B.2 - A.2)) :=
by sorry

end perpendicular_bisector_and_parallel_line_l3658_365817


namespace number_of_rooms_l3658_365844

theorem number_of_rooms (total_paintings : ℕ) (paintings_per_room : ℕ) (h1 : total_paintings = 32) (h2 : paintings_per_room = 8) :
  total_paintings / paintings_per_room = 4 := by
sorry

end number_of_rooms_l3658_365844


namespace statement_is_proposition_l3658_365882

def is_proposition (statement : Prop) : Prop :=
  statement ∨ ¬statement

theorem statement_is_proposition : is_proposition (20 - 5 * 3 = 10) := by
  sorry

end statement_is_proposition_l3658_365882


namespace min_distance_line_circle_l3658_365843

/-- The minimum distance between a point on the line y = 2 and a point on the circle (x - 1)² + y² = 1 is 1 -/
theorem min_distance_line_circle : 
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (P Q : ℝ × ℝ), 
    (P.2 = 2) → 
    ((Q.1 - 1)^2 + Q.2^2 = 1) → 
    d ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) :=
by sorry

end min_distance_line_circle_l3658_365843


namespace sandwich_jam_cost_l3658_365835

theorem sandwich_jam_cost (N B J : ℕ) (h1 : N > 1) (h2 : N * (3 * B + 7 * J) = 276) : 
  (N * J * 7 : ℚ) / 100 = 0.14 * J := by
  sorry

end sandwich_jam_cost_l3658_365835


namespace smallest_of_four_consecutive_integers_product_2520_l3658_365877

theorem smallest_of_four_consecutive_integers_product_2520 :
  ∃ (n : ℕ), n > 0 ∧ n * (n + 1) * (n + 2) * (n + 3) = 2520 ∧
  ∀ (m : ℕ), m > 0 → m * (m + 1) * (m + 2) * (m + 3) = 2520 → n ≤ m :=
by sorry

end smallest_of_four_consecutive_integers_product_2520_l3658_365877


namespace f_has_one_min_no_max_l3658_365866

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3

-- State the theorem
theorem f_has_one_min_no_max :
  ∃! x : ℝ, IsLocalMin f x ∧ ∀ y : ℝ, ¬IsLocalMax f y :=
by sorry

end f_has_one_min_no_max_l3658_365866


namespace points_subtracted_per_wrong_answer_l3658_365840

theorem points_subtracted_per_wrong_answer
  (total_problems : ℕ)
  (total_score : ℕ)
  (points_per_correct : ℕ)
  (wrong_answers : ℕ)
  (h1 : total_problems = 25)
  (h2 : total_score = 85)
  (h3 : points_per_correct = 4)
  (h4 : wrong_answers = 3)
  : (total_problems * points_per_correct - total_score) / wrong_answers = 1 := by
  sorry

end points_subtracted_per_wrong_answer_l3658_365840


namespace square_of_geometric_is_geometric_l3658_365860

-- Define a geometric sequence
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

-- Statement to prove
theorem square_of_geometric_is_geometric (a : ℕ → ℝ) (h : IsGeometric a) :
  IsGeometric (fun n ↦ (a n)^2) := by
  sorry

end square_of_geometric_is_geometric_l3658_365860


namespace unique_solution_fractional_equation_l3658_365869

theorem unique_solution_fractional_equation :
  ∃! x : ℚ, (1 : ℚ) / (x - 3) = (3 : ℚ) / (x - 6) ∧ x = 3 / 2 := by
  sorry

end unique_solution_fractional_equation_l3658_365869


namespace power_of_two_equation_l3658_365874

theorem power_of_two_equation (m : ℤ) :
  2^2010 - 2^2009 - 2^2008 + 2^2007 - 2^2006 = m * 2^2006 →
  m = 5 := by
sorry

end power_of_two_equation_l3658_365874


namespace james_birthday_stickers_l3658_365849

/-- The number of stickers James gets for his birthday -/
def birthday_stickers (initial_stickers total_stickers : ℕ) : ℕ :=
  total_stickers - initial_stickers

/-- Theorem: James got 22 stickers for his birthday -/
theorem james_birthday_stickers :
  birthday_stickers 39 61 = 22 := by
  sorry

end james_birthday_stickers_l3658_365849


namespace largest_common_divisor_of_consecutive_odd_numbers_l3658_365850

theorem largest_common_divisor_of_consecutive_odd_numbers (n : ℕ) :
  (n % 2 = 0 ∧ n > 0) →
  ∃ (k : ℕ), k = 45 ∧ 
    (∀ (m : ℕ), m ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → m ≤ k) ∧
    k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
by sorry


end largest_common_divisor_of_consecutive_odd_numbers_l3658_365850


namespace rectangle_side_length_l3658_365847

theorem rectangle_side_length (a b c d : ℝ) : 
  a / c = 3 / 4 → 
  b / d = 3 / 4 → 
  c = 4 → 
  d = 8 → 
  b = 6 := by
sorry

end rectangle_side_length_l3658_365847


namespace investment_calculation_l3658_365889

/-- Calculates the investment amount given share details and dividend received -/
theorem investment_calculation (face_value premium dividend_rate total_dividend : ℚ) : 
  face_value = 100 →
  premium = 20 / 100 →
  dividend_rate = 5 / 100 →
  total_dividend = 600 →
  (total_dividend / (face_value * dividend_rate)) * (face_value * (1 + premium)) = 14400 := by
  sorry

end investment_calculation_l3658_365889


namespace total_symbol_count_is_62_l3658_365812

/-- The number of distinct symbols that can be represented by a sequence of dots and dashes of a given length. -/
def symbolCount (length : Nat) : Nat :=
  2^length

/-- The total number of distinct symbols that can be represented using sequences of 1 to 5 dots and/or dashes. -/
def totalSymbolCount : Nat :=
  (symbolCount 1) + (symbolCount 2) + (symbolCount 3) + (symbolCount 4) + (symbolCount 5)

/-- Theorem stating that the total number of distinct symbols is 62. -/
theorem total_symbol_count_is_62 : totalSymbolCount = 62 := by
  sorry

end total_symbol_count_is_62_l3658_365812


namespace negation_of_existential_proposition_l3658_365897

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) := by sorry

end negation_of_existential_proposition_l3658_365897


namespace percent_of_number_l3658_365884

theorem percent_of_number (x : ℝ) : 120 = 1.5 * x → x = 80 := by
  sorry

end percent_of_number_l3658_365884


namespace distance_from_origin_to_point_l3658_365807

-- Define the point
def point : ℝ × ℝ := (12, -5)

-- Theorem statement
theorem distance_from_origin_to_point :
  Real.sqrt ((point.1 - 0)^2 + (point.2 - 0)^2) = 13 := by
  sorry

end distance_from_origin_to_point_l3658_365807


namespace value_of_b_l3658_365875

theorem value_of_b (a b : ℝ) (h1 : 3 * a + 1 = 4) (h2 : b - a = 1) : b = 2 := by
  sorry

end value_of_b_l3658_365875


namespace q_contribution_l3658_365805

/-- Represents the contribution and time in the business for a partner -/
structure Partner where
  contribution : ℕ
  time : ℕ

/-- Calculates the weighted contribution of a partner -/
def weightedContribution (p : Partner) : ℕ := p.contribution * p.time

/-- Represents the business scenario -/
structure Business where
  p : Partner
  q : Partner
  profitRatio : Fraction

theorem q_contribution (b : Business) : b.q.contribution = 9000 :=
  sorry

end q_contribution_l3658_365805


namespace solve_final_grade_problem_l3658_365872

def final_grade_problem (total_students : ℕ) (fraction_A fraction_B fraction_C : ℚ) : Prop :=
  let fraction_D := 1 - (fraction_A + fraction_B + fraction_C)
  let num_D := total_students - (total_students * (fraction_A + fraction_B + fraction_C)).floor
  (total_students = 100) ∧
  (fraction_A = 1/5) ∧
  (fraction_B = 1/4) ∧
  (fraction_C = 1/2) ∧
  (num_D = 5)

theorem solve_final_grade_problem :
  ∃ (total_students : ℕ) (fraction_A fraction_B fraction_C : ℚ),
    final_grade_problem total_students fraction_A fraction_B fraction_C :=
by
  sorry

end solve_final_grade_problem_l3658_365872


namespace sqrt_eight_plus_sqrt_two_equals_three_sqrt_two_l3658_365824

theorem sqrt_eight_plus_sqrt_two_equals_three_sqrt_two : 
  Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 := by
  sorry

end sqrt_eight_plus_sqrt_two_equals_three_sqrt_two_l3658_365824


namespace two_Z_one_eq_one_l3658_365876

/-- The Z operation on two real numbers -/
def Z (a b : ℝ) : ℝ := a^3 - 3*a^2*b + 3*a*b^2 - b^3

/-- Theorem: 2 Z 1 = 1 -/
theorem two_Z_one_eq_one : Z 2 1 = 1 := by sorry

end two_Z_one_eq_one_l3658_365876


namespace probability_two_yellow_one_red_l3658_365873

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := 3

/-- The number of yellow marbles in the jar -/
def yellow_marbles : ℕ := 5

/-- The number of orange marbles in the jar -/
def orange_marbles : ℕ := 4

/-- The total number of marbles in the jar -/
def total_marbles : ℕ := red_marbles + yellow_marbles + orange_marbles

/-- The number of marbles to be chosen -/
def chosen_marbles : ℕ := 3

/-- Calculates the number of combinations of n items taken k at a time -/
def combination (n k : ℕ) : ℕ := (n.factorial) / ((k.factorial) * ((n - k).factorial))

/-- The probability of choosing 2 yellow and 1 red marble from the jar -/
theorem probability_two_yellow_one_red : 
  (combination yellow_marbles 2 * combination red_marbles 1) / 
  (combination total_marbles chosen_marbles) = 3 / 22 := by
  sorry

end probability_two_yellow_one_red_l3658_365873


namespace open_box_volume_l3658_365886

/-- Calculate the volume of an open box created from a rectangular sheet --/
theorem open_box_volume (sheet_length sheet_width cut_side : ℝ) :
  sheet_length = 48 ∧ 
  sheet_width = 36 ∧ 
  cut_side = 8 →
  (sheet_length - 2 * cut_side) * (sheet_width - 2 * cut_side) * cut_side = 5120 := by
  sorry

end open_box_volume_l3658_365886


namespace system_solution_l3658_365810

theorem system_solution (x y : ℝ) 
  (eq1 : 2019 * x + 2020 * y = 2018)
  (eq2 : 2020 * x + 2019 * y = 2021) :
  x + y = 1 ∧ x - y = 3 := by
  sorry

end system_solution_l3658_365810


namespace small_shape_placement_exists_l3658_365896

/-- Represents a shape with a given area -/
structure Shape where
  area : ℝ

/-- Represents an infinite grid with cells of a given area -/
structure Grid where
  cellArea : ℝ

/-- Represents a placement of a shape on a grid -/
structure Placement where
  shape : Shape
  grid : Grid

/-- Predicate to check if a placement covers any grid vertex -/
def coversVertex (p : Placement) : Prop :=
  sorry -- Definition omitted as it's not explicitly given in the problem

/-- Theorem stating that a shape smaller than a grid cell can be placed without covering vertices -/
theorem small_shape_placement_exists (s : Shape) (g : Grid) 
    (h : s.area < g.cellArea) : 
    ∃ (p : Placement), p.shape = s ∧ p.grid = g ∧ ¬coversVertex p :=
  sorry

end small_shape_placement_exists_l3658_365896


namespace largest_n_multiple_of_7_l3658_365863

def is_multiple_of_7 (n : ℕ) : Prop :=
  (5 * (n - 3)^6 - 2 * n^3 + 20 * n - 35) % 7 = 0

theorem largest_n_multiple_of_7 :
  ∀ n : ℕ, n < 100000 →
    (is_multiple_of_7 n → n ≤ 99998) ∧
    (n > 99998 → ¬is_multiple_of_7 n) ∧
    is_multiple_of_7 99998 :=
by sorry

end largest_n_multiple_of_7_l3658_365863


namespace log3_45_not_expressible_l3658_365822

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Given conditions
axiom log3_27 : log 3 27 = 3
axiom log3_81 : log 3 81 = 4

-- Define the property of being expressible without logarithmic tables
def expressible_without_tables (x : ℝ) : Prop :=
  ∃ (f : ℝ → ℝ → ℝ), f (log 3 27) (log 3 81) = log 3 x

-- Theorem statement
theorem log3_45_not_expressible :
  ¬ expressible_without_tables 45 :=
sorry

end log3_45_not_expressible_l3658_365822


namespace gift_items_solution_l3658_365841

theorem gift_items_solution :
  ∃ (x y z : ℕ) (x' y' z' : ℕ),
    x + y + z = 20 ∧
    60 * x + 50 * y + 10 * z = 720 ∧
    x' + y' + z' = 20 ∧
    60 * x' + 50 * y' + 10 * z' = 720 ∧
    ((x = 4 ∧ y = 8 ∧ z = 8) ∨ (x = 8 ∧ y = 3 ∧ z = 9)) ∧
    ((x' = 4 ∧ y' = 8 ∧ z' = 8) ∨ (x' = 8 ∧ y' = 3 ∧ z' = 9)) ∧
    ¬(x = x' ∧ y = y' ∧ z = z') :=
by
  sorry

#check gift_items_solution

end gift_items_solution_l3658_365841


namespace valid_numbers_l3658_365816

def isValid (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 : ℕ), d1 > d2 ∧ d2 > d3 ∧
    d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧
    d1 + d2 + d3 = 1457 ∧
    ∀ (d : ℕ), d ∣ n → d ≤ d1

theorem valid_numbers : ∀ (n : ℕ), isValid n ↔ n = 987 ∨ n = 1023 ∨ n = 1085 ∨ n = 1175 := by
  sorry

end valid_numbers_l3658_365816


namespace pet_store_profit_l3658_365809

-- Define Brandon's selling price
def brandon_price : ℕ := 100

-- Define the pet store's pricing strategy
def pet_store_price (brandon_price : ℕ) : ℕ := 3 * brandon_price + 5

-- Define the profit calculation
def profit (selling_price cost_price : ℕ) : ℕ := selling_price - cost_price

-- Theorem to prove
theorem pet_store_profit :
  profit (pet_store_price brandon_price) brandon_price = 205 := by
  sorry

end pet_store_profit_l3658_365809


namespace line_through_points_l3658_365855

/-- Given a line passing through points (-1, -4) and (x, k), where the slope
    of the line is equal to k and k = 1, prove that x = 4. -/
theorem line_through_points (x : ℝ) :
  let k : ℝ := 1
  let slope : ℝ := (k - (-4)) / (x - (-1))
  slope = k → x = 4 := by
  sorry

end line_through_points_l3658_365855


namespace max_a1_is_26_l3658_365802

/-- A sequence of positive integers satisfying the given conditions --/
def GoodSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (∀ n, a n ≤ a (n + 1)) ∧
  (∀ n, a (n + 1) ≤ a n + 5) ∧
  (∀ n, n ∣ a n)

/-- The maximum possible value of a₁ in a good sequence --/
def MaxA1 : ℕ := 26

/-- The theorem stating that the maximum possible value of a₁ in a good sequence is 26 --/
theorem max_a1_is_26 :
  (∃ a, GoodSequence a ∧ a 1 = MaxA1) ∧
  (∀ a, GoodSequence a → a 1 ≤ MaxA1) :=
sorry

end max_a1_is_26_l3658_365802


namespace mike_car_expenses_l3658_365890

def speakers : ℚ := 118.54
def tires : ℚ := 106.33
def windowTints : ℚ := 85.27
def seatCovers : ℚ := 79.99
def maintenance : ℚ := 199.75
def steeringWheelCover : ℚ := 15.63
def airFresheners : ℚ := 6.48 * 2  -- Assuming one set of two
def carWash : ℚ := 25

def totalExpenses : ℚ := speakers + tires + windowTints + seatCovers + maintenance + steeringWheelCover + airFresheners + carWash

theorem mike_car_expenses :
  totalExpenses = 643.47 := by sorry

end mike_car_expenses_l3658_365890


namespace sum_of_squares_divisible_by_seven_l3658_365803

theorem sum_of_squares_divisible_by_seven (a b : ℤ) : 
  (7 ∣ a^2 + b^2) → (7 ∣ a) ∧ (7 ∣ b) := by
  sorry

end sum_of_squares_divisible_by_seven_l3658_365803


namespace hunter_journey_l3658_365880

theorem hunter_journey (swamp_speed forest_speed highway_speed : ℝ)
  (total_time total_distance : ℝ) (swamp_time forest_time highway_time : ℝ) :
  swamp_speed = 2 →
  forest_speed = 4 →
  highway_speed = 6 →
  total_time = 4 →
  total_distance = 15 →
  swamp_time + forest_time + highway_time = total_time →
  swamp_speed * swamp_time + forest_speed * forest_time + highway_speed * highway_time = total_distance →
  swamp_time > highway_time := by
  sorry

end hunter_journey_l3658_365880


namespace area_outside_triangle_l3658_365898

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- The length of side PQ -/
  pq : ℝ
  /-- The length of the hypotenuse PR -/
  pr : ℝ
  /-- The circle is inscribed in the triangle -/
  inscribed : Bool
  /-- The triangle is right-angled at Q -/
  right_angle : Bool
  /-- The circle is tangent to PQ at M and to QR at N -/
  tangent_points : Bool
  /-- The points diametrically opposite M and N lie on PR -/
  diametric_points : Bool

/-- The theorem stating the area of the portion of the circle outside the triangle -/
theorem area_outside_triangle (t : RightTriangleWithInscribedCircle) 
  (h1 : t.pq = 9)
  (h2 : t.pr = 15)
  (h3 : t.inscribed)
  (h4 : t.right_angle)
  (h5 : t.tangent_points)
  (h6 : t.diametric_points) :
  ∃ (area : ℝ), area = 28.125 * Real.pi - 56.25 := by
  sorry

end area_outside_triangle_l3658_365898


namespace mary_vacuum_charges_l3658_365861

/-- The number of times Mary needs to charge her vacuum cleaner to clean her whole house -/
def charges_needed (battery_duration : ℕ) (time_per_room : ℕ) (num_bedrooms : ℕ) (num_kitchen : ℕ) (num_living_room : ℕ) : ℕ :=
  let total_rooms := num_bedrooms + num_kitchen + num_living_room
  let total_time := time_per_room * total_rooms
  (total_time + battery_duration - 1) / battery_duration

theorem mary_vacuum_charges :
  charges_needed 10 4 3 1 1 = 2 := by
  sorry

end mary_vacuum_charges_l3658_365861


namespace percentage_of_women_professors_l3658_365865

-- Define the percentage of professors who are women
variable (W : ℝ)

-- Define the percentage of professors who are tenured
def T : ℝ := 70

-- Define the principle of inclusion-exclusion
axiom inclusion_exclusion : W + T - (W * T / 100) = 90

-- Define the percentage of men who are tenured
axiom men_tenured : (100 - W) * 52 / 100 = T - (W * T / 100)

-- Theorem to prove
theorem percentage_of_women_professors : ∃ ε > 0, abs (W - 79.17) < ε := by
  sorry

end percentage_of_women_professors_l3658_365865


namespace dave_total_wage_l3658_365811

/-- Represents the daily wage information --/
structure DailyWage where
  hourly_rate : ℕ
  hours_worked : ℕ

/-- Calculates the total wage for a given day --/
def daily_total (dw : DailyWage) : ℕ :=
  dw.hourly_rate * dw.hours_worked

/-- Dave's wage information for Monday to Thursday --/
def dave_wages : List DailyWage := [
  ⟨6, 6⟩,  -- Monday
  ⟨7, 2⟩,  -- Tuesday
  ⟨9, 3⟩,  -- Wednesday
  ⟨8, 5⟩   -- Thursday
]

theorem dave_total_wage :
  (dave_wages.map daily_total).sum = 117 := by
  sorry

#eval (dave_wages.map daily_total).sum

end dave_total_wage_l3658_365811


namespace integer_fraction_characterization_l3658_365801

theorem integer_fraction_characterization (a b : ℕ+) :
  (∃ k : ℕ+, (a.val ^ 2 : ℤ) = k * (2 * a.val * b.val ^ 2 - b.val ^ 3 + 1)) ↔
  (∃ l : ℕ+, (a = 2 * l ∧ b = 1) ∨
             (a = l ∧ b = 2 * l) ∨
             (a = 8 * l.val ^ 4 - l ∧ b = 2 * l)) :=
by sorry

end integer_fraction_characterization_l3658_365801


namespace max_log_sum_l3658_365878

/-- Given that xyz + y + z = 12, the maximum value of log₄x + log₂y + log₂z is 3 -/
theorem max_log_sum (x y z : ℝ) (h : x * y * z + y + z = 12) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.log x / Real.log 4) + (Real.log y / Real.log 2) + (Real.log z / Real.log 2) ≤ 3 := by
  sorry

end max_log_sum_l3658_365878


namespace football_tournament_max_points_l3658_365837

theorem football_tournament_max_points (n : ℕ) : 
  (∃ (scores : Fin 15 → ℕ), 
    (∀ i j : Fin 15, i ≠ j → scores i + scores j ≤ 3) ∧ 
    (∃ (successful : Finset (Fin 15)), 
      successful.card = 6 ∧ 
      ∀ i ∈ successful, n ≤ scores i)) →
  n ≤ 34 :=
sorry

end football_tournament_max_points_l3658_365837


namespace farm_problem_l3658_365846

/-- The farm problem -/
theorem farm_problem (H C : ℕ) : 
  (H - 15) / (C + 15) = 3 →  -- After transaction, ratio is 3:1
  H - 15 = C + 15 + 70 →    -- After transaction, 70 more horses than cows
  H / C = 6                  -- Initial ratio is 6:1
:= by sorry

end farm_problem_l3658_365846


namespace triangle_dimensions_l3658_365883

theorem triangle_dimensions (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0)
  (eq1 : 2 * a / 3 = b) (eq2 : 2 * c = a) (eq3 : b - 2 = c) :
  a = 12 ∧ b = 8 ∧ c = 6 := by
sorry

end triangle_dimensions_l3658_365883


namespace min_value_theorem_l3658_365831

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) 
  (hm : m > 0) (hn : n > 0) (h_intersection : m + 4*n = 1) : 
  (1/m + 4/n) ≥ 25 := by
  sorry

end min_value_theorem_l3658_365831


namespace total_oranges_picked_l3658_365820

def monday_pick : ℕ := 100
def tuesday_pick : ℕ := 3 * monday_pick
def wednesday_pick : ℕ := 70

theorem total_oranges_picked : monday_pick + tuesday_pick + wednesday_pick = 470 := by
  sorry

end total_oranges_picked_l3658_365820


namespace division_problem_l3658_365825

theorem division_problem (x : ℝ) (h1 : 10 * x = 50) : 20 / x = 4 := by
  sorry

end division_problem_l3658_365825


namespace chocolate_manufacturer_min_price_l3658_365899

/-- Calculates the minimum selling price per unit for a chocolate manufacturer --/
theorem chocolate_manufacturer_min_price
  (units : ℕ)
  (cost_per_unit : ℝ)
  (min_profit : ℝ)
  (h1 : units = 400)
  (h2 : cost_per_unit = 40)
  (h3 : min_profit = 40000) :
  (min_profit + units * cost_per_unit) / units = 140 := by
  sorry

end chocolate_manufacturer_min_price_l3658_365899


namespace operator_value_l3658_365856

/-- The operator definition -/
def operator (a : ℝ) (x : ℝ) : ℝ := x * (a - x)

/-- Theorem stating the value of 'a' in the operator definition -/
theorem operator_value :
  ∃ a : ℝ, (∀ p : ℝ, p = 1 → p + 1 = operator a (p + 1)) → a = 2.5 := by
  sorry

end operator_value_l3658_365856


namespace arrangements_count_l3658_365871

/-- Represents the number of students -/
def num_students : ℕ := 6

/-- Represents the condition that B and C must be adjacent -/
def bc_adjacent : Prop := True

/-- Represents the condition that A cannot stand at either end -/
def a_not_at_ends : Prop := True

/-- The number of different arrangements satisfying the given conditions -/
def num_arrangements : ℕ := 144

/-- Theorem stating that the number of arrangements is 144 -/
theorem arrangements_count :
  (num_students = 6) →
  bc_adjacent →
  a_not_at_ends →
  num_arrangements = 144 := by
  sorry

end arrangements_count_l3658_365871


namespace complex_equation_sum_l3658_365851

/-- Given that (1+i)(x-yi) = 2, where x and y are real numbers and i is the imaginary unit, prove that x + y = 2 -/
theorem complex_equation_sum (x y : ℝ) : (Complex.I + 1) * (x - y * Complex.I) = 2 → x + y = 2 := by
  sorry

end complex_equation_sum_l3658_365851


namespace cannot_determine_dracula_state_l3658_365888

-- Define the possible states for the Transylvanian and Count Dracula
inductive State : Type
  | Human : State
  | Undead : State
  | Alive : State
  | Dead : State

-- Define the Transylvanian's statement
def transylvanianStatement (transylvanian : State) (dracula : State) : Prop :=
  (transylvanian = State.Human) → (dracula = State.Alive)

-- Define the theorem
theorem cannot_determine_dracula_state :
  ∀ (transylvanian : State) (dracula : State),
    transylvanianStatement transylvanian dracula →
    ¬(∀ (dracula' : State), dracula' = State.Alive ∨ dracula' = State.Dead) :=
by sorry

end cannot_determine_dracula_state_l3658_365888


namespace range_of_sum_of_squares_l3658_365818

theorem range_of_sum_of_squares (x y : ℝ) (h : x^2 - 2*x*y + 5*y^2 = 4) :
  3 - Real.sqrt 5 ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ 3 + Real.sqrt 5 := by
  sorry

end range_of_sum_of_squares_l3658_365818


namespace johns_age_l3658_365859

theorem johns_age (john dad brother : ℕ) 
  (h1 : john + 28 = dad)
  (h2 : john + dad = 76)
  (h3 : john + 5 = 2 * (brother + 5)) : 
  john = 24 := by
  sorry

end johns_age_l3658_365859


namespace second_divisor_is_nine_l3658_365885

theorem second_divisor_is_nine (least_number : Nat) (second_divisor : Nat) : 
  least_number = 282 →
  least_number % 31 = 3 →
  least_number % second_divisor = 3 →
  second_divisor ≠ 31 →
  second_divisor = 9 := by
sorry

end second_divisor_is_nine_l3658_365885


namespace CaO_weight_calculation_l3658_365804

/-- The atomic weight of calcium in g/mol -/
def calcium_weight : ℝ := 40.08

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of moles of CaO -/
def moles_CaO : ℝ := 7

/-- The molecular weight of CaO in g/mol -/
def molecular_weight_CaO : ℝ := calcium_weight + oxygen_weight

/-- The total weight of CaO in grams -/
def total_weight_CaO : ℝ := molecular_weight_CaO * moles_CaO

theorem CaO_weight_calculation : total_weight_CaO = 392.56 := by
  sorry

end CaO_weight_calculation_l3658_365804


namespace unique_n_exists_l3658_365881

theorem unique_n_exists : ∃! n : ℤ,
  0 ≤ n ∧ n < 17 ∧
  -150 ≡ n [ZMOD 17] ∧
  102 % n = 0 ∧
  n = 3 := by
  sorry

end unique_n_exists_l3658_365881


namespace paving_cost_l3658_365848

/-- The cost of paving a rectangular floor given its dimensions and the rate per square meter. -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 3.75) (h3 : rate = 600) :
  length * width * rate = 12375 := by sorry

end paving_cost_l3658_365848


namespace average_score_is_71_l3658_365857

def mathematics_score : ℕ := 76
def science_score : ℕ := 65
def social_studies_score : ℕ := 82
def english_score : ℕ := 47
def biology_score : ℕ := 85

def total_score : ℕ := mathematics_score + science_score + social_studies_score + english_score + biology_score
def number_of_subjects : ℕ := 5

theorem average_score_is_71 : (total_score : ℚ) / number_of_subjects = 71 := by
  sorry

end average_score_is_71_l3658_365857


namespace jerry_log_count_l3658_365864

/-- The number of logs produced by a pine tree -/
def logsPerPine : ℕ := 80

/-- The number of logs produced by a maple tree -/
def logsPerMaple : ℕ := 60

/-- The number of logs produced by a walnut tree -/
def logsPerWalnut : ℕ := 100

/-- The number of pine trees Jerry cuts -/
def pineTreesCut : ℕ := 8

/-- The number of maple trees Jerry cuts -/
def mapleTreesCut : ℕ := 3

/-- The number of walnut trees Jerry cuts -/
def walnutTreesCut : ℕ := 4

/-- The total number of logs Jerry gets -/
def totalLogs : ℕ := logsPerPine * pineTreesCut + logsPerMaple * mapleTreesCut + logsPerWalnut * walnutTreesCut

theorem jerry_log_count : totalLogs = 1220 := by
  sorry

end jerry_log_count_l3658_365864


namespace whole_number_between_constraints_l3658_365823

theorem whole_number_between_constraints (N : ℤ) : 
  (6 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 7.5) ↔ N ∈ ({25, 26, 27, 28, 29} : Set ℤ) :=
sorry

end whole_number_between_constraints_l3658_365823


namespace point_on_x_axis_l3658_365808

/-- Given a point A with coordinates (m+1, 2m-4) that is moved up by 2 units
    and lands on the x-axis, prove that m = 1. -/
theorem point_on_x_axis (m : ℝ) : 
  let initial_y := 2*m - 4
  let moved_y := initial_y + 2
  moved_y = 0 → m = 1 := by
  sorry

end point_on_x_axis_l3658_365808


namespace price_to_relatives_is_correct_l3658_365894

-- Define the given quantities
def total_peaches : ℕ := 15
def peaches_sold_to_friends : ℕ := 10
def peaches_sold_to_relatives : ℕ := 4
def peaches_kept : ℕ := 1
def price_per_peach_to_friends : ℚ := 2
def total_earnings : ℚ := 25

-- Define the function to calculate the price per peach sold to relatives
def price_per_peach_to_relatives : ℚ :=
  (total_earnings - price_per_peach_to_friends * peaches_sold_to_friends) / peaches_sold_to_relatives

-- Theorem statement
theorem price_to_relatives_is_correct : price_per_peach_to_relatives = 1.25 := by
  sorry

end price_to_relatives_is_correct_l3658_365894
