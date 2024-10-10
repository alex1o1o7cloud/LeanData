import Mathlib

namespace smallest_sports_team_size_l1954_195422

theorem smallest_sports_team_size : ∃ n : ℕ, 
  n > 0 ∧ 
  n % 3 = 1 ∧ 
  n % 4 = 2 ∧ 
  n % 6 = 4 ∧ 
  ∃ m : ℕ, n = m ^ 2 ∧
  ∀ k : ℕ, k > 0 → k % 3 = 1 → k % 4 = 2 → k % 6 = 4 → (∃ l : ℕ, k = l ^ 2) → k ≥ n :=
by sorry

end smallest_sports_team_size_l1954_195422


namespace dessert_preference_l1954_195427

structure Classroom where
  total : ℕ
  apple : ℕ
  chocolate : ℕ
  pumpkin : ℕ
  none : ℕ

def likes_apple_and_chocolate_not_pumpkin (c : Classroom) : ℕ :=
  c.apple + c.chocolate - (c.total - c.none) - 2

theorem dessert_preference (c : Classroom) 
  (h_total : c.total = 50)
  (h_apple : c.apple = 25)
  (h_chocolate : c.chocolate = 20)
  (h_pumpkin : c.pumpkin = 10)
  (h_none : c.none = 16) :
  likes_apple_and_chocolate_not_pumpkin c = 9 := by
  sorry

end dessert_preference_l1954_195427


namespace kindergarten_gifts_l1954_195457

theorem kindergarten_gifts :
  ∀ (n : ℕ) (total_gifts : ℕ),
    (2 * 4 + (n - 2) * 3 + 11 = total_gifts) →
    (4 * 3 + (n - 4) * 6 + 10 = total_gifts) →
    total_gifts = 28 :=
by
  sorry

end kindergarten_gifts_l1954_195457


namespace fraction_subtraction_addition_l1954_195401

theorem fraction_subtraction_addition : 
  (1 : ℚ) / 12 - 5 / 6 + 1 / 3 = -5 / 12 := by sorry

end fraction_subtraction_addition_l1954_195401


namespace point_coordinates_l1954_195496

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- The third quadrant of the Cartesian coordinate system -/
def third_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- The distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (A : Point) 
  (h1 : third_quadrant A)
  (h2 : distance_to_x_axis A = 2)
  (h3 : distance_to_y_axis A = 3) :
  A.x = -3 ∧ A.y = -2 :=
by sorry

end point_coordinates_l1954_195496


namespace david_widget_production_l1954_195419

theorem david_widget_production (w t : ℕ) (h : w = 3 * t) :
  w * t - (w + 5) * (t - 3) = 4 * t + 15 := by
  sorry

end david_widget_production_l1954_195419


namespace young_bonnets_theorem_l1954_195498

/-- Calculates the number of bonnets Mrs. Young sends to each orphanage --/
def bonnets_per_orphanage : ℕ :=
  let monday := 10
  let tuesday_wednesday := 2 * monday
  let thursday := monday + 5
  let friday := thursday - 5
  let saturday := friday - 8
  let sunday := 3 * saturday
  let total := monday + tuesday_wednesday + thursday + friday + saturday + sunday
  total / 10

/-- Theorem stating that Mrs. Young sends 6 bonnets to each orphanage --/
theorem young_bonnets_theorem : bonnets_per_orphanage = 6 := by
  sorry

end young_bonnets_theorem_l1954_195498


namespace smallest_number_l1954_195470

theorem smallest_number (a b c d : ℝ) (h1 : a = 1) (h2 : b = 0) (h3 : c = -2 * Real.sqrt 2) (h4 : d = -3) :
  min a (min b (min c d)) = -3 := by
  sorry

end smallest_number_l1954_195470


namespace three_color_plane_coloring_l1954_195445

-- Define a type for colors
inductive Color
| Red
| Green
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a type for lines in the plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define a coloring function
def Coloring := Point → Color

-- Define a predicate to check if a point is on a line
def IsOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define a predicate to check if a line contains at most two colors
def LineContainsAtMostTwoColors (coloring : Coloring) (l : Line) : Prop :=
  ∃ (c1 c2 : Color), ∀ (p : Point), IsOnLine p l → coloring p = c1 ∨ coloring p = c2

-- Define a predicate to check if all three colors are used
def AllColorsUsed (coloring : Coloring) : Prop :=
  (∃ (p : Point), coloring p = Color.Red) ∧
  (∃ (p : Point), coloring p = Color.Green) ∧
  (∃ (p : Point), coloring p = Color.Blue)

-- Theorem statement
theorem three_color_plane_coloring :
  ∃ (coloring : Coloring),
    (∀ (l : Line), LineContainsAtMostTwoColors coloring l) ∧
    AllColorsUsed coloring :=
by
  sorry

end three_color_plane_coloring_l1954_195445


namespace initial_eggs_count_l1954_195415

theorem initial_eggs_count (eggs_used : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (final_eggs : ℕ) : 
  eggs_used = 5 → chickens = 2 → eggs_per_chicken = 3 → final_eggs = 11 →
  ∃ initial_eggs : ℕ, initial_eggs = 10 ∧ initial_eggs - eggs_used + chickens * eggs_per_chicken = final_eggs :=
by
  sorry


end initial_eggs_count_l1954_195415


namespace length_of_24_l1954_195471

def length (k : ℕ) : ℕ :=
  (Nat.factors k).length

theorem length_of_24 : length 24 = 4 := by
  sorry

end length_of_24_l1954_195471


namespace storks_joined_l1954_195407

theorem storks_joined (initial_birds : ℕ) (initial_storks : ℕ) (final_total : ℕ) : 
  initial_birds = 3 → initial_storks = 4 → final_total = 13 →
  final_total - (initial_birds + initial_storks) = 6 :=
by
  sorry

end storks_joined_l1954_195407


namespace cubic_root_in_interval_l1954_195483

theorem cubic_root_in_interval (a b c : ℝ) 
  (h_roots : ∃ (r₁ r₂ r₃ : ℝ), ∀ x, x^3 + a*x^2 + b*x + c = (x - r₁) * (x - r₂) * (x - r₃))
  (h_sum : -2 ≤ a + b + c ∧ a + b + c ≤ 0) :
  ∃ r, (r^3 + a*r^2 + b*r + c = 0) ∧ (0 ≤ r ∧ r ≤ 2) := by
  sorry

end cubic_root_in_interval_l1954_195483


namespace hostel_accommodation_l1954_195435

/-- Proves that 20 additional students were accommodated in the hostel --/
theorem hostel_accommodation :
  ∀ (initial_students : ℕ) 
    (initial_avg_expenditure : ℚ)
    (avg_decrease : ℚ)
    (total_increase : ℕ)
    (new_total_expenditure : ℕ),
  initial_students = 100 →
  avg_decrease = 5 →
  total_increase = 400 →
  new_total_expenditure = 5400 →
  ∃ (additional_students : ℕ),
    additional_students = 20 ∧
    (initial_avg_expenditure - avg_decrease) * (initial_students + additional_students) = new_total_expenditure :=
by sorry

end hostel_accommodation_l1954_195435


namespace triangle_shift_area_ratio_l1954_195485

theorem triangle_shift_area_ratio (L α : ℝ) (h1 : 0 < α) (h2 : α < L) :
  let x := α / L
  (x * (2 * L^2 / 2) = (L^2 / 2 - (L - α)^2 / 2)) → x = (3 - Real.sqrt 5) / 2 := by
  sorry

end triangle_shift_area_ratio_l1954_195485


namespace problem_1_problem_2_l1954_195487

-- Problem 1
theorem problem_1 :
  2⁻¹ + |Real.sqrt 6 - 3| + 2 * Real.sqrt 3 * Real.sin (45 * π / 180) - (-2)^2023 * (1/2)^2023 = 9/2 := by
  sorry

-- Problem 2
theorem problem_2 (a : ℝ) (h : a = 3) :
  (3 / (a + 1) - a + 1) / ((a^2 - 4) / (a^2 + 2*a + 1)) = -4 := by
  sorry

end problem_1_problem_2_l1954_195487


namespace percentage_subtraction_l1954_195491

theorem percentage_subtraction (a : ℝ) : a - 0.02 * a = 0.98 * a := by
  sorry

end percentage_subtraction_l1954_195491


namespace angle_C_measure_l1954_195416

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem angle_C_measure (t : Triangle) : 
  t.A = 39 * π / 180 ∧ 
  (t.a^2 - t.b^2) * (t.a^2 + t.a * t.c - t.b^2) = t.b^2 * t.c^2 ∧
  t.A + t.B + t.C = π →
  t.C = 115 * π / 180 := by
  sorry

end angle_C_measure_l1954_195416


namespace inclined_line_properties_l1954_195431

/-- A line passing through a point with a given inclination angle -/
structure InclinedLine where
  point : ℝ × ℝ
  angle : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Theorem about the equation and triangle area of an inclined line -/
theorem inclined_line_properties (l : InclinedLine) 
  (h1 : l.point = (Real.sqrt 3, -2))
  (h2 : l.angle = π / 3) : 
  ∃ (eq : LineEquation) (area : ℝ),
    eq.a = Real.sqrt 3 ∧ 
    eq.b = -1 ∧ 
    eq.c = -5 ∧
    area = (25 * Real.sqrt 3) / 6 := by
  sorry

end inclined_line_properties_l1954_195431


namespace solve_for_q_l1954_195493

theorem solve_for_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p*q = 8) :
  q = 4 + 2 * Real.sqrt 2 := by
sorry

end solve_for_q_l1954_195493


namespace consecutive_product_not_25k_plus_1_l1954_195486

theorem consecutive_product_not_25k_plus_1 (k n : ℕ) : n * (n + 1) ≠ 25 * k + 1 := by
  sorry

end consecutive_product_not_25k_plus_1_l1954_195486


namespace triangle_side_sum_bound_l1954_195455

/-- Given a triangle ABC with side lengths a, b, and c, where c = 2 and the dot product 
    of vectors AC and AB is equal to b² - (1/2)ab, prove that 2 < a + b ≤ 4 -/
theorem triangle_side_sum_bound (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  let c : ℝ := 2
  let dot_product : ℝ := b^2 - (1/2) * a * b
  2 < a + b ∧ a + b ≤ 4 := by sorry

end triangle_side_sum_bound_l1954_195455


namespace negation_of_implication_l1954_195420

theorem negation_of_implication (P Q : Prop) :
  ¬(P → Q) ↔ (¬P → ¬Q) := by sorry

end negation_of_implication_l1954_195420


namespace thirteenth_root_unity_product_l1954_195476

theorem thirteenth_root_unity_product (w : ℂ) : w^13 = 1 → (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 := by
  sorry

end thirteenth_root_unity_product_l1954_195476


namespace abs_value_sum_l1954_195414

theorem abs_value_sum (a b c : ℝ) : 
  abs a = 1 → abs b = 2 → abs c = 3 → a > b → b > c → a + b - c = 2 := by
  sorry

end abs_value_sum_l1954_195414


namespace r_and_s_earnings_l1954_195437

/-- The daily earnings of individuals p, q, r, and s --/
structure DailyEarnings where
  p : ℚ
  q : ℚ
  r : ℚ
  s : ℚ

/-- The conditions given in the problem --/
def problem_conditions (e : DailyEarnings) : Prop :=
  e.p + e.q + e.r + e.s = 2380 / 9 ∧
  e.p + e.r = 600 / 5 ∧
  e.q + e.s = 800 / 6 ∧
  e.q + e.r = 910 / 7 ∧
  e.p = 150 / 3

/-- The theorem stating that r and s together earn 430/3 Rs per day --/
theorem r_and_s_earnings (e : DailyEarnings) :
  problem_conditions e → e.r + e.s = 430 / 3 := by
  sorry

#check r_and_s_earnings

end r_and_s_earnings_l1954_195437


namespace elinas_garden_area_l1954_195423

/-- The area of Elina's rectangular garden --/
def garden_area (length width : ℝ) : ℝ := length * width

/-- The perimeter of Elina's rectangular garden --/
def garden_perimeter (length width : ℝ) : ℝ := 2 * (length + width)

theorem elinas_garden_area :
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length * 30 = 1500 ∧
    garden_perimeter length width * 12 = 1500 ∧
    garden_area length width = 625 := by
  sorry

end elinas_garden_area_l1954_195423


namespace power_function_property_l1954_195465

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ a

-- State the theorem
theorem power_function_property (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 4 / f 2 = 3) :
  f (1/2) = 1/3 := by
  sorry

end power_function_property_l1954_195465


namespace percentage_of_long_term_employees_l1954_195448

/-- Represents the number of employees in each year range at the Pythagoras company -/
structure EmployeeDistribution where
  less_than_1_year : ℕ
  one_to_2_years : ℕ
  two_to_3_years : ℕ
  three_to_4_years : ℕ
  four_to_5_years : ℕ
  five_to_6_years : ℕ
  six_to_7_years : ℕ
  seven_to_8_years : ℕ
  eight_to_9_years : ℕ
  nine_to_10_years : ℕ
  ten_to_11_years : ℕ
  eleven_to_12_years : ℕ
  twelve_to_13_years : ℕ
  thirteen_to_14_years : ℕ
  fourteen_to_15_years : ℕ

/-- Calculates the total number of employees -/
def totalEmployees (d : EmployeeDistribution) : ℕ :=
  d.less_than_1_year + d.one_to_2_years + d.two_to_3_years + d.three_to_4_years +
  d.four_to_5_years + d.five_to_6_years + d.six_to_7_years + d.seven_to_8_years +
  d.eight_to_9_years + d.nine_to_10_years + d.ten_to_11_years + d.eleven_to_12_years +
  d.twelve_to_13_years + d.thirteen_to_14_years + d.fourteen_to_15_years

/-- Calculates the number of employees who have worked for 10 years or more -/
def employeesWithTenYearsOrMore (d : EmployeeDistribution) : ℕ :=
  d.ten_to_11_years + d.eleven_to_12_years + d.twelve_to_13_years +
  d.thirteen_to_14_years + d.fourteen_to_15_years

/-- Theorem: The percentage of employees who have worked at the Pythagoras company for 10 years or more is 15% -/
theorem percentage_of_long_term_employees (d : EmployeeDistribution)
  (h : d = { less_than_1_year := 4, one_to_2_years := 6, two_to_3_years := 7,
             three_to_4_years := 4, four_to_5_years := 3, five_to_6_years := 3,
             six_to_7_years := 2, seven_to_8_years := 2, eight_to_9_years := 1,
             nine_to_10_years := 1, ten_to_11_years := 2, eleven_to_12_years := 1,
             twelve_to_13_years := 1, thirteen_to_14_years := 1, fourteen_to_15_years := 1 }) :
  (employeesWithTenYearsOrMore d : ℚ) / (totalEmployees d : ℚ) = 15 / 100 := by
  sorry

end percentage_of_long_term_employees_l1954_195448


namespace circles_product_radii_equals_sum_squares_l1954_195447

/-- Given two circles passing through a point M(x₁, y₁) and tangent to both the x-axis and y-axis
    with radii r₁ and r₂, the product of their radii equals the sum of squares of the coordinates of M. -/
theorem circles_product_radii_equals_sum_squares (x₁ y₁ r₁ r₂ : ℝ) 
    (h1 : ∃ (a₁ b₁ : ℝ), (x₁ - a₁)^2 + (y₁ - b₁)^2 = r₁^2 ∧ |a₁| = r₁ ∧ |b₁| = r₁)
    (h2 : ∃ (a₂ b₂ : ℝ), (x₁ - a₂)^2 + (y₁ - b₂)^2 = r₂^2 ∧ |a₂| = r₂ ∧ |b₂| = r₂) :
  r₁ * r₂ = x₁^2 + y₁^2 := by
  sorry

end circles_product_radii_equals_sum_squares_l1954_195447


namespace arithmetic_sequence_problem_l1954_195429

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence where the 3rd term is 5 and the 7th term is 29, the 10th term is 47. -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h_arith : ArithmeticSequence a) 
  (h_3rd : a 3 = 5)
  (h_7th : a 7 = 29) : 
  a 10 = 47 := by
  sorry


end arithmetic_sequence_problem_l1954_195429


namespace set_intersection_equiv_a_range_l1954_195460

/-- Given real number a, define set A -/
def A (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}

/-- Define set B based on set A -/
def B (a : ℝ) : Set ℝ := {y | ∃ x ∈ A a, y = 2 * x + 3}

/-- Define set C based on set A -/
def C (a : ℝ) : Set ℝ := {z | ∃ x ∈ A a, z = x^2}

/-- Theorem stating the equivalence of the set intersection condition and the range of a -/
theorem set_intersection_equiv_a_range (a : ℝ) :
  (B a ∩ C a = C a) ↔ (a < -2 ∨ (1/2 ≤ a ∧ a ≤ 3)) := by
  sorry

end set_intersection_equiv_a_range_l1954_195460


namespace system_solution_l1954_195442

theorem system_solution : ∃! (u v : ℝ), 5 * u = -7 - 2 * v ∧ 3 * u = 4 * v - 25 := by
  sorry

end system_solution_l1954_195442


namespace expected_winning_percentage_approx_l1954_195464

/-- Represents the political parties --/
inductive Party
  | Republican
  | Democrat
  | Independent

/-- Represents a candidate in the election --/
inductive Candidate
  | X
  | Y

/-- The ratio of registered voters for each party --/
def partyRatio : Party → ℚ
  | Party.Republican => 3
  | Party.Democrat => 2
  | Party.Independent => 1

/-- The percentage of voters from each party expected to vote for Candidate X --/
def votePercentageForX : Party → ℚ
  | Party.Republican => 85 / 100
  | Party.Democrat => 60 / 100
  | Party.Independent => 40 / 100

/-- The total number of registered voters (assumed to be 6n for some positive integer n) --/
def totalVoters : ℚ := 6

/-- Calculate the expected winning percentage for Candidate X --/
def expectedWinningPercentage : ℚ :=
  let votesForX := (partyRatio Party.Republican * votePercentageForX Party.Republican +
                    partyRatio Party.Democrat * votePercentageForX Party.Democrat +
                    partyRatio Party.Independent * votePercentageForX Party.Independent)
  let votesForY := totalVoters - votesForX
  (votesForX - votesForY) / totalVoters * 100

/-- Theorem stating that the expected winning percentage for Candidate X is approximately 38.33% --/
theorem expected_winning_percentage_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 / 100 ∧ |expectedWinningPercentage - 3833 / 100| < ε :=
sorry

end expected_winning_percentage_approx_l1954_195464


namespace integer_solutions_for_equation_l1954_195461

theorem integer_solutions_for_equation : 
  {(x, y) : ℤ × ℤ | x^2 - y^4 = 2009} = {(45, 2), (45, -2), (-45, 2), (-45, -2)} :=
by sorry

end integer_solutions_for_equation_l1954_195461


namespace alpha_values_l1954_195489

theorem alpha_values (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^2 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^4 - 1) = 5 * Complex.abs (α - 1)) :
  ∃ (x y : ℝ), α = Complex.mk x y ∧ 
    ((x = (1 + 8*Real.sqrt 2/9)/2 ∨ x = (1 - 8*Real.sqrt 2/9)/2) ∧
     y^2 = 9 - ((x + 1)^2)) :=
by sorry

end alpha_values_l1954_195489


namespace vector_subtraction_l1954_195441

theorem vector_subtraction (a b : ℝ × ℝ) :
  a = (5, 3) → b = (1, -2) → a - 2 • b = (3, 7) := by
  sorry

end vector_subtraction_l1954_195441


namespace novelty_shop_costs_l1954_195404

/-- Represents the cost of items in dollars -/
structure ItemCost where
  magazine : ℝ
  chocolate : ℝ
  candy : ℝ
  toy : ℝ

/-- The conditions given in the problem -/
def shopConditions (cost : ItemCost) : Prop :=
  cost.magazine = 1 ∧
  4 * cost.chocolate = 8 * cost.magazine ∧
  2 * cost.candy + 3 * cost.toy = 5 * cost.magazine

/-- The theorem stating the cost of a dozen chocolate bars and the indeterminacy of candy and toy costs -/
theorem novelty_shop_costs (cost : ItemCost) (h : shopConditions cost) :
  12 * cost.chocolate = 24 ∧
  ∃ (c t : ℝ), c ≠ cost.candy ∧ t ≠ cost.toy ∧ shopConditions { magazine := cost.magazine, chocolate := cost.chocolate, candy := c, toy := t } :=
by sorry

end novelty_shop_costs_l1954_195404


namespace fraction_subtraction_l1954_195468

theorem fraction_subtraction : (18 : ℚ) / 42 - 2 / 9 = 13 / 63 := by sorry

end fraction_subtraction_l1954_195468


namespace four_propositions_l1954_195499

-- Definition of opposite numbers
def are_opposite (x y : ℝ) : Prop := x = -y

-- Definition of congruent triangles
def congruent_triangles (t1 t2 : Set ℝ × Set ℝ) : Prop := sorry

-- Definition of triangle area
def triangle_area (t : Set ℝ × Set ℝ) : ℝ := sorry

-- Statement of the theorem
theorem four_propositions :
  (∀ x y : ℝ, are_opposite x y → x + y = 0) ∧
  (∀ q : ℝ, (∃ x : ℝ, x^2 + 2*x + q = 0) → q ≤ 1) ∧
  (∃ t1 t2 : Set ℝ × Set ℝ, ¬(congruent_triangles t1 t2) ∧ triangle_area t1 = triangle_area t2) ∧
  (∃ a b c : ℝ, a > b ∧ ¬(a * c^2 > b * c^2)) :=
by sorry

end four_propositions_l1954_195499


namespace curve_scaling_transformation_l1954_195440

/-- Given a curve C that undergoes a scaling transformation,
    prove that the equation of the original curve is x^2/4 + 9y^2 = 1 -/
theorem curve_scaling_transformation (x y x' y' : ℝ) :
  (x' = 1/2 * x) →
  (y' = 3 * y) →
  (x'^2 + y'^2 = 1) →
  (x^2/4 + 9*y^2 = 1) :=
by sorry

end curve_scaling_transformation_l1954_195440


namespace height_prediction_age_10_l1954_195438

/-- Regression model for height prediction -/
def height_model (age : ℝ) : ℝ := 7.19 * age + 73.93

/-- The predicted height at age 10 is approximately 145.83 cm -/
theorem height_prediction_age_10 :
  ∃ ε > 0, abs (height_model 10 - 145.83) < ε :=
sorry

end height_prediction_age_10_l1954_195438


namespace sqrt_4_squared_times_5_to_6th_l1954_195432

theorem sqrt_4_squared_times_5_to_6th : Real.sqrt (4^2 * 5^6) = 500 := by
  sorry

end sqrt_4_squared_times_5_to_6th_l1954_195432


namespace min_value_theorem_l1954_195418

theorem min_value_theorem (x y a : ℝ) 
  (h1 : (x - 3)^3 + 2016 * (x - 3) = a) 
  (h2 : (2 * y - 3)^3 + 2016 * (2 * y - 3) = -a) : 
  ∃ (m : ℝ), m = 28 ∧ ∀ (x y : ℝ), x^2 + 4 * y^2 + 4 * x ≥ m := by
  sorry

end min_value_theorem_l1954_195418


namespace parabola_properties_l1954_195479

/-- Given a parabola y = x^2 + 2bx + b^2 - 2 where b > 0 and passing through point (0, -1) -/
theorem parabola_properties (b : ℝ) (h1 : b > 0) :
  let f (x : ℝ) := x^2 + 2*b*x + b^2 - 2
  ∃ (vertex_x vertex_y : ℝ),
    -- 1. The vertex coordinates are (-b, -2)
    (vertex_x = -b ∧ vertex_y = -2) ∧
    -- Parabola passes through (0, -1)
    (f 0 = -1) ∧
    -- 2. When -2 < x < 3, the range of y is -2 ≤ y < 14
    (∀ x, -2 < x → x < 3 → -2 ≤ f x ∧ f x < 14) ∧
    -- 3. When k ≤ x ≤ 2 and -2 ≤ y ≤ 7, the range of k is -4 ≤ k ≤ -1
    (∀ k, (∀ x, k ≤ x → x ≤ 2 → -2 ≤ f x → f x ≤ 7) → -4 ≤ k ∧ k ≤ -1) :=
by sorry

end parabola_properties_l1954_195479


namespace smallest_common_multiple_l1954_195411

theorem smallest_common_multiple (h : ℕ) (d : ℕ) : 
  (∀ k : ℕ, k > 0 ∧ 10 * k % 15 = 0 → k ≥ 3) ∧ 
  (10 * 3 % 15 = 0) := by
  sorry

end smallest_common_multiple_l1954_195411


namespace quarters_to_dollars_l1954_195403

/-- The number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- The total number of quarters -/
def total_quarters : ℕ := 8

/-- The dollar amount equivalent to the total number of quarters -/
def dollar_amount : ℚ := total_quarters / quarters_per_dollar

theorem quarters_to_dollars : dollar_amount = 2 := by
  sorry

end quarters_to_dollars_l1954_195403


namespace second_neighbor_brought_fewer_hotdog_difference_l1954_195480

/-- The number of hotdogs brought by the first neighbor -/
def first_neighbor_hotdogs : ℕ := 75

/-- The total number of hotdogs brought by both neighbors -/
def total_hotdogs : ℕ := 125

/-- The number of hotdogs brought by the second neighbor -/
def second_neighbor_hotdogs : ℕ := total_hotdogs - first_neighbor_hotdogs

/-- The second neighbor brought fewer hotdogs than the first -/
theorem second_neighbor_brought_fewer :
  second_neighbor_hotdogs < first_neighbor_hotdogs := by sorry

/-- The difference in hotdogs between the first and second neighbor is 25 -/
theorem hotdog_difference :
  first_neighbor_hotdogs - second_neighbor_hotdogs = 25 := by sorry

end second_neighbor_brought_fewer_hotdog_difference_l1954_195480


namespace order_of_numbers_l1954_195413

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def a : Nat := base_to_decimal [1,1,1,1,1,1] 2
def b : Nat := base_to_decimal [0,1,2] 6
def c : Nat := base_to_decimal [0,0,0,1] 4
def d : Nat := base_to_decimal [0,1,1] 8

theorem order_of_numbers : b > d ∧ d > a ∧ a > c := by
  sorry

end order_of_numbers_l1954_195413


namespace rationalize_and_sum_l1954_195406

theorem rationalize_and_sum (a b c d s f : ℚ) (p q r : ℕ) :
  let x := (1 : ℝ) / (Real.sqrt 5 + Real.sqrt 6 + Real.sqrt 8)
  let y := (a * Real.sqrt p + b * Real.sqrt q + c * Real.sqrt r + d * Real.sqrt s) / f
  (∃ (a b c d s : ℚ) (p q r : ℕ) (f : ℚ), 
    f > 0 ∧ 
    x = y ∧
    (p = 5 ∧ q = 6 ∧ r = 2 ∧ s = 1) ∧
    (a = 9 ∧ b = 7 ∧ c = -18 ∧ d = 0)) →
  a + b + c + d + s + f = 111 :=
by sorry

end rationalize_and_sum_l1954_195406


namespace gas_price_and_distance_l1954_195452

-- Define the problem parameters
def expected_gallons : ℝ := 12
def actual_gallons : ℝ := 10
def price_increase : ℝ := 0.3
def fuel_efficiency : ℝ := 25

-- Define the theorem
theorem gas_price_and_distance :
  ∃ (original_price : ℝ) (new_distance : ℝ),
    -- The total cost remains the same
    expected_gallons * original_price = actual_gallons * (original_price + price_increase) ∧
    -- Calculate the new distance
    new_distance = actual_gallons * fuel_efficiency ∧
    -- The original price is $1.50
    original_price = 1.5 ∧
    -- The new distance is 250 miles
    new_distance = 250 := by
  sorry

end gas_price_and_distance_l1954_195452


namespace non_intersecting_lines_parallel_or_skew_l1954_195462

/-- A line in 3D space represented by a point and a direction vector. -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Determines if two lines intersect in 3D space. -/
def intersect (l1 l2 : Line3D) : Prop :=
  ∃ t s : ℝ, l1.point + t • l1.direction = l2.point + s • l2.direction

/-- Two lines are parallel if their direction vectors are scalar multiples of each other. -/
def parallel (l1 l2 : Line3D) : Prop :=
  ∃ k : ℝ, l1.direction = k • l2.direction

/-- Two lines are skew if they are neither intersecting nor parallel. -/
def skew (l1 l2 : Line3D) : Prop :=
  ¬(intersect l1 l2) ∧ ¬(parallel l1 l2)

/-- Theorem: If two lines in 3D space do not intersect, then they are either parallel or skew. -/
theorem non_intersecting_lines_parallel_or_skew (l1 l2 : Line3D) :
  ¬(intersect l1 l2) → parallel l1 l2 ∨ skew l1 l2 := by
  sorry

end non_intersecting_lines_parallel_or_skew_l1954_195462


namespace fencing_problem_l1954_195451

theorem fencing_problem (length width : ℝ) : 
  width = 40 →
  length * width = 200 →
  2 * length + width = 50 :=
by sorry

end fencing_problem_l1954_195451


namespace doodads_for_thingamabobs_l1954_195434

/-- The number of doodads required to make one widget -/
def doodads_per_widget : ℚ := 18 / 5

/-- The number of widgets required to make one thingamabob -/
def widgets_per_thingamabob : ℚ := 11 / 4

/-- The number of thingamabobs we want to make -/
def target_thingamabobs : ℕ := 80

/-- Theorem stating that 792 doodads are required to make 80 thingamabobs -/
theorem doodads_for_thingamabobs : 
  ⌈(target_thingamabobs : ℚ) * widgets_per_thingamabob * doodads_per_widget⌉ = 792 := by
  sorry

end doodads_for_thingamabobs_l1954_195434


namespace sheetrock_area_is_30_l1954_195436

/-- Represents the area of a rectangular sheetrock given its length and width. -/
def sheetrockArea (length width : ℝ) : ℝ := length * width

/-- Theorem stating that the area of a rectangular sheetrock with length 6 feet and width 5 feet is 30 square feet. -/
theorem sheetrock_area_is_30 : sheetrockArea 6 5 = 30 := by
  sorry

end sheetrock_area_is_30_l1954_195436


namespace gcf_lcm_40_120_100_l1954_195475

theorem gcf_lcm_40_120_100 :
  (let a := 40
   let b := 120
   let c := 100
   (Nat.gcd a (Nat.gcd b c) = 20) ∧
   (Nat.lcm a (Nat.lcm b c) = 600)) := by
  sorry

end gcf_lcm_40_120_100_l1954_195475


namespace angle_with_same_terminal_side_l1954_195433

/-- The angle (in degrees) that has the same terminal side as 1303° -/
def equivalent_angle : ℝ := -137

/-- Theorem stating that the angle with the same terminal side as 1303° is -137° -/
theorem angle_with_same_terminal_side :
  ∃ (k : ℤ), 1303 = 360 * k + equivalent_angle := by
  sorry

end angle_with_same_terminal_side_l1954_195433


namespace supermarket_fruit_prices_l1954_195408

theorem supermarket_fruit_prices 
  (strawberry_pints : ℕ) 
  (strawberry_sale_revenue : ℕ) 
  (strawberry_revenue_difference : ℕ)
  (blueberry_pints : ℕ) 
  (blueberry_sale_revenue : ℕ) 
  (blueberry_revenue_difference : ℕ)
  (h1 : strawberry_pints = 54)
  (h2 : strawberry_sale_revenue = 216)
  (h3 : strawberry_revenue_difference = 108)
  (h4 : blueberry_pints = 36)
  (h5 : blueberry_sale_revenue = 144)
  (h6 : blueberry_revenue_difference = 72) :
  (strawberry_sale_revenue + strawberry_revenue_difference) / strawberry_pints = 
  (blueberry_sale_revenue + blueberry_revenue_difference) / blueberry_pints :=
by sorry

end supermarket_fruit_prices_l1954_195408


namespace a_range_when_A_union_B_is_R_A_union_B_is_R_when_a_in_range_l1954_195439

/-- The set A defined by the inequality (x - 1)(x - a) ≥ 0 -/
def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}

/-- The set B defined by the inequality x ≥ a - 1 -/
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

/-- Theorem stating that if A ∪ B = ℝ, then a ∈ (-∞, 2] -/
theorem a_range_when_A_union_B_is_R (a : ℝ) 
  (h : A a ∪ B a = Set.univ) : a ≤ 2 := by
  sorry

/-- Theorem stating that if a ∈ (-∞, 2], then A ∪ B = ℝ -/
theorem A_union_B_is_R_when_a_in_range (a : ℝ) 
  (h : a ≤ 2) : A a ∪ B a = Set.univ := by
  sorry

end a_range_when_A_union_B_is_R_A_union_B_is_R_when_a_in_range_l1954_195439


namespace jane_tom_sum_difference_l1954_195428

/-- The sum of numbers from 1 to 50 -/
def janeSum : ℕ := (List.range 50).map (· + 1) |>.sum

/-- Function to replace 3 with 2 in a number -/
def replace3With2 (n : ℕ) : ℕ :=
  let s := toString n
  (s.replace "3" "2").toNat!

/-- The sum of numbers from 1 to 50 with 3 replaced by 2 -/
def tomSum : ℕ := (List.range 50).map (· + 1) |>.map replace3With2 |>.sum

/-- Theorem stating the difference between Jane's and Tom's sums -/
theorem jane_tom_sum_difference : janeSum - tomSum = 105 := by
  sorry

end jane_tom_sum_difference_l1954_195428


namespace p_suff_not_nec_q_l1954_195410

-- Define propositions p, q, and r
variable (p q r : Prop)

-- Define the conditions
axiom p_suff_not_nec_r : (p → r) ∧ ¬(r → p)
axiom q_nec_r : r → q

-- Theorem to prove
theorem p_suff_not_nec_q : (p → q) ∧ ¬(q → p) := by
  sorry

end p_suff_not_nec_q_l1954_195410


namespace equation_solution_l1954_195466

theorem equation_solution : 
  ∃! x : ℝ, (x / (x + 2) + 3 / (x + 2) + 2 * x / (x + 2) = 4) ∧ x = -5 := by
  sorry

end equation_solution_l1954_195466


namespace arccos_less_than_arctan_in_interval_l1954_195459

theorem arccos_less_than_arctan_in_interval :
  ∀ x : ℝ, 0.5 < x ∧ x ≤ 1 → Real.arccos x < Real.arctan x := by
  sorry

end arccos_less_than_arctan_in_interval_l1954_195459


namespace smallest_solution_of_equation_l1954_195412

theorem smallest_solution_of_equation (x : ℚ) :
  (7 * (10 * x^2 + 10 * x + 11) = x * (10 * x - 45)) →
  x ≥ -7/5 :=
by sorry

end smallest_solution_of_equation_l1954_195412


namespace seventh_roots_of_unity_product_l1954_195400

theorem seventh_roots_of_unity_product (z : ℂ) (h : z = Complex.exp (2 * Real.pi * Complex.I / 7)) :
  (3 - z) * (3 - z^2) * (3 - z^3) * (3 - z^4) * (3 - z^5) * (3 - z^6) = 1093 := by
  sorry

end seventh_roots_of_unity_product_l1954_195400


namespace birthday_attendees_l1954_195469

theorem birthday_attendees : ∃ (n : ℕ), 
  (12 * (n + 2) = 16 * n) ∧ 
  (n = 6) := by
sorry

end birthday_attendees_l1954_195469


namespace eight_digit_integers_count_l1954_195482

theorem eight_digit_integers_count : 
  (Finset.range 8).card * (10 ^ 7) = 80000000 := by sorry

end eight_digit_integers_count_l1954_195482


namespace complement_of_M_l1954_195490

-- Define the universal set U as ℝ
def U := ℝ

-- Define the set M
def M : Set ℝ := {x | Real.log (1 - x) > 0}

-- State the theorem
theorem complement_of_M : 
  (Mᶜ : Set ℝ) = {x | x ≥ 0} := by sorry

end complement_of_M_l1954_195490


namespace average_tickets_sold_l1954_195481

/-- Proves that the average number of tickets sold per day is 80 -/
theorem average_tickets_sold (total_days : ℕ) (total_worth : ℕ) (ticket_cost : ℕ) :
  total_days = 3 →
  total_worth = 960 →
  ticket_cost = 4 →
  (total_worth / ticket_cost) / total_days = 80 := by
  sorry

end average_tickets_sold_l1954_195481


namespace fourth_intersection_point_l1954_195424

/-- Given a hyperbola xy = 2 and three points on this curve that also lie on a circle,
    prove that the fourth intersection point has specific coordinates. -/
theorem fourth_intersection_point (P₁ P₂ P₃ P₄ : ℝ × ℝ) : 
  P₁.1 * P₁.2 = 2 ∧ P₂.1 * P₂.2 = 2 ∧ P₃.1 * P₃.2 = 2 ∧ P₄.1 * P₄.2 = 2 →
  P₁ = (3, 2/3) ∧ P₂ = (-4, -1/2) ∧ P₃ = (1/2, 4) →
  ∃ (a b r : ℝ), 
    (P₁.1 - a)^2 + (P₁.2 - b)^2 = r^2 ∧
    (P₂.1 - a)^2 + (P₂.2 - b)^2 = r^2 ∧
    (P₃.1 - a)^2 + (P₃.2 - b)^2 = r^2 ∧
    (P₄.1 - a)^2 + (P₄.2 - b)^2 = r^2 →
  P₄ = (-2/3, -3) := by
sorry

end fourth_intersection_point_l1954_195424


namespace rectangle_area_l1954_195425

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) :
  square_area = 1225 →
  rectangle_breadth = 10 →
  let square_side := Real.sqrt square_area
  let circle_radius := square_side
  let rectangle_length := circle_radius / 4
  let rectangle_area := rectangle_length * rectangle_breadth
  rectangle_area = 87.5 := by
  sorry

end rectangle_area_l1954_195425


namespace sin_half_and_third_max_solutions_l1954_195456

open Real

theorem sin_half_and_third_max_solutions (α : ℝ) : 
  (∃ (s : Finset ℝ), (∀ x ∈ s, ∃ k : ℤ, (x = α/2 + k*π ∨ x = (π - α)/2 + k*π) ∧ sin x = sin α) ∧ s.card ≤ 4) ∧
  (∃ (t : Finset ℝ), (∀ x ∈ t, ∃ k : ℤ, x = α/3 + 2*k*π/3 ∧ sin x = sin α) ∧ t.card ≤ 3) :=
sorry

end sin_half_and_third_max_solutions_l1954_195456


namespace total_missed_pitches_l1954_195409

/-- The number of pitches per token -/
def pitches_per_token : ℕ := 15

/-- The number of tokens Macy used -/
def macy_tokens : ℕ := 11

/-- The number of tokens Piper used -/
def piper_tokens : ℕ := 17

/-- The number of times Macy hit the ball -/
def macy_hits : ℕ := 50

/-- The number of times Piper hit the ball -/
def piper_hits : ℕ := 55

/-- Theorem stating the total number of missed pitches -/
theorem total_missed_pitches :
  (macy_tokens * pitches_per_token - macy_hits) +
  (piper_tokens * pitches_per_token - piper_hits) = 315 := by
  sorry

end total_missed_pitches_l1954_195409


namespace cube_root_simplification_l1954_195453

theorem cube_root_simplification :
  (20^3 + 30^3 + 40^3 + 8000)^(1/3) = 8 * (1500)^(1/3) := by
  sorry

end cube_root_simplification_l1954_195453


namespace intersection_point_in_zero_one_l1954_195488

-- Define the function f(x) = x^3 - (1/2)^x
noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)^x

-- State the theorem
theorem intersection_point_in_zero_one :
  ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧ f x₀ = 0 :=
sorry

end intersection_point_in_zero_one_l1954_195488


namespace no_arithmetic_progression_40_terms_l1954_195458

theorem no_arithmetic_progression_40_terms : ¬ ∃ (a d : ℕ) (f : ℕ → ℕ × ℕ),
  (∀ i : ℕ, i < 40 → ∃ (m n : ℕ), f i = (m, n) ∧ a + i * d = 2^m + 3^n) :=
sorry

end no_arithmetic_progression_40_terms_l1954_195458


namespace parabola_hyperbola_tangent_l1954_195477

theorem parabola_hyperbola_tangent (a b p k : ℝ) : 
  a > 0 → 
  b > 0 → 
  p > 0 → 
  (2 * a = 4 * Real.sqrt 2) → 
  (b = p / 2) → 
  (k = p / (4 * Real.sqrt 2)) → 
  (∀ x y : ℝ, y = k * x - 1 → x^2 = 2 * p * y) →
  p = 4 :=
by sorry

end parabola_hyperbola_tangent_l1954_195477


namespace angle_bisector_shorter_than_median_l1954_195421

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the length of a line segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the angle bisector
def angle_bisector (t : Triangle) : ℝ × ℝ := sorry

-- Define the median
def median (t : Triangle) : ℝ × ℝ := sorry

-- Theorem statement
theorem angle_bisector_shorter_than_median (t : Triangle) :
  length t.A t.B ≤ length t.A t.C →
  length t.A (angle_bisector t) ≤ length t.A (median t) ∧
  (length t.A (angle_bisector t) = length t.A (median t) ↔ length t.A t.B = length t.A t.C) :=
sorry

end angle_bisector_shorter_than_median_l1954_195421


namespace margin_expression_l1954_195463

/-- Given a selling price S, a ratio m, and a cost C, prove that the margin M
    can be expressed as (1/m)S. -/
theorem margin_expression (S m : ℝ) (h_m : m ≠ 0) :
  let M := (1 / m) * S
  let C := S - M
  M = (1 / m) * S := by sorry

end margin_expression_l1954_195463


namespace geraldine_doll_count_l1954_195492

/-- The number of dolls Jazmin has -/
def jazmin_dolls : ℝ := 1209.0

/-- The number of additional dolls Geraldine has compared to Jazmin -/
def additional_dolls : ℕ := 977

/-- The total number of dolls Geraldine has -/
def geraldine_dolls : ℝ := jazmin_dolls + additional_dolls

theorem geraldine_doll_count : geraldine_dolls = 2186 := by
  sorry

end geraldine_doll_count_l1954_195492


namespace quadratic_inequality_solution_l1954_195430

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, (a * x^2 + 5 * x - 2 > 0) ↔ (1/2 < x ∧ x < b)) → 
  (a = -2 ∧ b = 2) := by sorry

end quadratic_inequality_solution_l1954_195430


namespace parabola_intersection_kite_coefficient_sum_l1954_195494

/-- Represents a parabola of the form y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents a kite formed by the intersection points of two parabolas with the coordinate axes -/
structure Kite where
  p1 : Parabola
  p2 : Parabola

/-- The area of a kite -/
def kiteArea (k : Kite) : ℝ := sorry

/-- The sum of the coefficients of the x^2 terms in the two parabolas forming the kite -/
def coefficientSum (k : Kite) : ℝ := k.p1.a + k.p2.a

theorem parabola_intersection_kite_coefficient_sum :
  ∀ k : Kite,
    k.p1 = Parabola.mk k.p1.a (-3) →
    k.p2 = Parabola.mk (-k.p2.a) 5 →
    kiteArea k = 15 →
    ∃ ε > 0, |coefficientSum k - 2.3| < ε :=
sorry

end parabola_intersection_kite_coefficient_sum_l1954_195494


namespace nancy_pearl_beads_difference_l1954_195473

/-- Prove that Nancy has 60 more pearl beads than metal beads -/
theorem nancy_pearl_beads_difference (beads_per_bracelet : ℕ) 
  (total_bracelets : ℕ) (nancy_metal_beads : ℕ) (rose_crystal_beads : ℕ) :
  beads_per_bracelet = 8 →
  total_bracelets = 20 →
  nancy_metal_beads = 40 →
  rose_crystal_beads = 20 →
  ∃ (nancy_pearl_beads : ℕ),
    nancy_pearl_beads = beads_per_bracelet * total_bracelets - 
      (nancy_metal_beads + rose_crystal_beads + 2 * rose_crystal_beads) ∧
    nancy_pearl_beads - nancy_metal_beads = 60 :=
by sorry

end nancy_pearl_beads_difference_l1954_195473


namespace photocopy_savings_theorem_l1954_195426

/-- Represents the cost structure for photocopies -/
structure CostStructure where
  base_cost : Real
  color_cost : Real
  double_sided_cost : Real
  discount_tier1 : Real
  discount_tier2 : Real
  discount_tier3 : Real

/-- Represents an order of photocopies -/
structure Order where
  bw_one_sided : Nat
  bw_double_sided : Nat
  color_one_sided : Nat
  color_double_sided : Nat

/-- Calculates the cost of an order without discount -/
def orderCost (cs : CostStructure) (o : Order) : Real := sorry

/-- Calculates the discount percentage based on the total number of copies -/
def discountPercentage (cs : CostStructure) (total_copies : Nat) : Real := sorry

/-- Calculates the total cost of combined orders with discount -/
def combinedOrderCost (cs : CostStructure) (o1 o2 : Order) : Real := sorry

/-- Calculates the savings when combining two orders -/
def savings (cs : CostStructure) (o1 o2 : Order) : Real := sorry

theorem photocopy_savings_theorem (cs : CostStructure) (steve_order dennison_order : Order) :
  cs.base_cost = 0.02 ∧
  cs.color_cost = 0.08 ∧
  cs.double_sided_cost = 0.03 ∧
  cs.discount_tier1 = 0.1 ∧
  cs.discount_tier2 = 0.2 ∧
  cs.discount_tier3 = 0.3 ∧
  steve_order.bw_one_sided = 35 ∧
  steve_order.bw_double_sided = 25 ∧
  steve_order.color_one_sided = 0 ∧
  steve_order.color_double_sided = 15 ∧
  dennison_order.bw_one_sided = 20 ∧
  dennison_order.bw_double_sided = 40 ∧
  dennison_order.color_one_sided = 12 ∧
  dennison_order.color_double_sided = 0 →
  savings cs steve_order dennison_order = 1.5 := by
  sorry

end photocopy_savings_theorem_l1954_195426


namespace function_value_at_negative_one_l1954_195444

theorem function_value_at_negative_one (f : ℝ → ℝ) 
  (h1 : Differentiable ℝ f) 
  (h2 : ∀ x, f x = x^2 + 2 * (deriv f 2) * x + 3) : 
  f (-1) = 12 := by
sorry

end function_value_at_negative_one_l1954_195444


namespace total_money_earned_l1954_195402

def clementine_cookies : ℕ := 72
def jake_cookies : ℕ := 2 * clementine_cookies
def combined_cookies : ℕ := jake_cookies + clementine_cookies
def tory_cookies : ℕ := combined_cookies / 2
def total_cookies : ℕ := clementine_cookies + jake_cookies + tory_cookies
def price_per_cookie : ℕ := 2

theorem total_money_earned : total_cookies * price_per_cookie = 648 := by
  sorry

end total_money_earned_l1954_195402


namespace complex_modulus_problem_l1954_195446

theorem complex_modulus_problem (i : ℂ) (a : ℝ) :
  i^2 = -1 →
  (∃ (b : ℝ), (2 - i) / (a + i) = b * i) →
  Complex.abs ((2 * a + 1) + Real.sqrt 2 * i) = Real.sqrt 6 :=
by sorry

end complex_modulus_problem_l1954_195446


namespace student_number_problem_l1954_195405

theorem student_number_problem (x : ℤ) : (8 * x - 138 = 102) → x = 30 := by
  sorry

end student_number_problem_l1954_195405


namespace dance_lesson_cost_l1954_195454

/-- Calculates the total cost of dance lessons given the number of lessons,
    cost per lesson, and number of free lessons. -/
def total_cost (total_lessons : ℕ) (cost_per_lesson : ℕ) (free_lessons : ℕ) : ℕ :=
  (total_lessons - free_lessons) * cost_per_lesson

/-- Theorem stating that given 10 dance lessons costing $10 each,
    with 2 lessons for free, the total cost is $80. -/
theorem dance_lesson_cost :
  total_cost 10 10 2 = 80 := by
  sorry

end dance_lesson_cost_l1954_195454


namespace change_in_math_preference_l1954_195472

theorem change_in_math_preference (initial_yes initial_no final_yes final_no absentee_rate : ℝ) :
  initial_yes = 0.4 →
  initial_no = 0.6 →
  final_yes = 0.8 →
  final_no = 0.2 →
  absentee_rate = 0.1 →
  ∃ (min_change max_change : ℝ),
    min_change ≥ 0 ∧
    max_change ≤ 1 ∧
    max_change - min_change = 0.4 :=
by sorry

end change_in_math_preference_l1954_195472


namespace min_value_a_l1954_195417

theorem min_value_a (x y z : ℝ) (h : x^2 + 4*y^2 + z^2 = 6) :
  (∃ (a : ℝ), ∀ (x y z : ℝ), x^2 + 4*y^2 + z^2 = 6 → x + 2*y + 3*z ≤ a) ∧
  (∀ (b : ℝ), (∀ (x y z : ℝ), x^2 + 4*y^2 + z^2 = 6 → x + 2*y + 3*z ≤ b) → Real.sqrt 66 ≤ b) :=
sorry

end min_value_a_l1954_195417


namespace cookies_removed_theorem_l1954_195497

/-- Calculates the number of cookies removed in 4 days given initial and final cookie counts over a week -/
def cookies_removed_in_four_days (initial_cookies : ℕ) (remaining_cookies : ℕ) : ℕ :=
  let total_removed : ℕ := initial_cookies - remaining_cookies
  let daily_removal : ℕ := total_removed / 7
  4 * daily_removal

/-- Theorem stating that given 70 initial cookies and 28 remaining after a week, 24 cookies are removed in 4 days -/
theorem cookies_removed_theorem :
  cookies_removed_in_four_days 70 28 = 24 := by
  sorry

end cookies_removed_theorem_l1954_195497


namespace height_range_is_75cm_l1954_195495

/-- The range of a set of values is the difference between the maximum and minimum values. -/
def range (max min : ℝ) : ℝ := max - min

/-- The heights of five students at Gleeson Middle School. -/
structure StudentHeights where
  num_students : ℕ
  max_height : ℝ
  min_height : ℝ

/-- The range of heights of the students is 75 cm. -/
theorem height_range_is_75cm (heights : StudentHeights) 
  (h1 : heights.num_students = 5)
  (h2 : heights.max_height = 175)
  (h3 : heights.min_height = 100) : 
  range heights.max_height heights.min_height = 75 := by
sorry

end height_range_is_75cm_l1954_195495


namespace cost_difference_l1954_195478

def rental_initial_cost : ℕ := 20
def rental_monthly_increase : ℕ := 5
def rental_insurance : ℕ := 15
def rental_maintenance : ℕ := 10

def new_car_monthly_payment : ℕ := 30
def new_car_down_payment : ℕ := 1500
def new_car_insurance : ℕ := 20
def new_car_maintenance_first_half : ℕ := 5
def new_car_maintenance_second_half : ℕ := 10

def months : ℕ := 12

def rental_total_cost : ℕ := 
  rental_initial_cost + rental_insurance + rental_maintenance + 
  (rental_initial_cost + rental_monthly_increase + rental_insurance + rental_maintenance) * (months - 1)

def new_car_total_cost : ℕ := 
  new_car_down_payment + 
  (new_car_monthly_payment + new_car_insurance + new_car_maintenance_first_half) * (months / 2) +
  (new_car_monthly_payment + new_car_insurance + new_car_maintenance_second_half) * (months / 2)

theorem cost_difference : new_car_total_cost - rental_total_cost = 1595 := by
  sorry

end cost_difference_l1954_195478


namespace inequality_proof_l1954_195484

theorem inequality_proof (a b : ℝ) (h : a * b > 0) : b / a + a / b ≥ 2 := by
  sorry

end inequality_proof_l1954_195484


namespace monotone_increasing_interval_minimum_m_for_inequality_l1954_195449

noncomputable section

def f (m : ℝ) (x : ℝ) := Real.log x - m * x^2
def g (m : ℝ) (x : ℝ) := (1/2) * m * x^2 + x

theorem monotone_increasing_interval (x : ℝ) :
  StrictMonoOn (f (1/2)) (Set.Ioo 0 1) := by sorry

theorem minimum_m_for_inequality :
  ∀ m : ℕ, (∀ x : ℝ, x > 0 → f m x + g m x ≤ m * x - 1) →
  m ≥ 2 := by sorry

end

end monotone_increasing_interval_minimum_m_for_inequality_l1954_195449


namespace square_of_binomial_form_l1954_195474

theorem square_of_binomial_form (x y : ℝ) :
  ∃ (a b : ℝ), (1/3 * x + y) * (y - 1/3 * x) = (a - b)^2 := by
  sorry

end square_of_binomial_form_l1954_195474


namespace probability_of_four_given_different_numbers_l1954_195467

/-- Two fair dice are rolled once each -/
def roll_two_dice : Type := Unit

/-- Event A: The two dice show different numbers -/
def event_A (roll : roll_two_dice) : Prop := sorry

/-- Event B: A 4 is rolled -/
def event_B (roll : roll_two_dice) : Prop := sorry

/-- P(B|A) is the conditional probability of event B given event A -/
def conditional_probability (A B : roll_two_dice → Prop) : ℝ := sorry

theorem probability_of_four_given_different_numbers :
  conditional_probability event_A event_B = 1/3 := by sorry

end probability_of_four_given_different_numbers_l1954_195467


namespace inequality_proof_l1954_195443

theorem inequality_proof (a b c d : ℝ) (h1 : d ≥ 0) (h2 : a + b = 2) (h3 : c + d = 2) :
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := by
  sorry

end inequality_proof_l1954_195443


namespace overtime_hours_is_eight_l1954_195450

/-- Calculates overtime hours given regular pay rate, regular hours, overtime rate multiplier, and total pay -/
def calculate_overtime_hours (regular_rate : ℚ) (regular_hours : ℚ) (overtime_multiplier : ℚ) (total_pay : ℚ) : ℚ :=
  let regular_pay := regular_rate * regular_hours
  let overtime_rate := regular_rate * overtime_multiplier
  let overtime_pay := total_pay - regular_pay
  overtime_pay / overtime_rate

/-- Proves that given the problem conditions, the number of overtime hours is 8 -/
theorem overtime_hours_is_eight :
  let regular_rate : ℚ := 3
  let regular_hours : ℚ := 40
  let overtime_multiplier : ℚ := 2
  let total_pay : ℚ := 168
  calculate_overtime_hours regular_rate regular_hours overtime_multiplier total_pay = 8 := by
  sorry

#eval calculate_overtime_hours 3 40 2 168

end overtime_hours_is_eight_l1954_195450
