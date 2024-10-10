import Mathlib

namespace midpoint_coordinate_sum_l155_15503

/-- Given that M(5,5) is the midpoint of line segment CD and C has coordinates (10,10),
    prove that the sum of the coordinates of point D is 0. -/
theorem midpoint_coordinate_sum (C D M : ℝ × ℝ) : 
  M = (5, 5) → 
  C = (10, 10) → 
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) → 
  D.1 + D.2 = 0 :=
by
  sorry

end midpoint_coordinate_sum_l155_15503


namespace algebraic_expression_value_l155_15572

theorem algebraic_expression_value :
  let a : ℝ := Real.sqrt 2 + 1
  let b : ℝ := Real.sqrt 2 - 1
  (a^2 - 2*a*b + b^2) / (a^2 - b^2) = Real.sqrt 2 / 2 := by
sorry

end algebraic_expression_value_l155_15572


namespace isosceles_triangle_legs_l155_15564

-- Define an isosceles triangle
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  is_isosceles : side1 = side2

-- Define the theorem
theorem isosceles_triangle_legs (t : IsoscelesTriangle) :
  t.side1 + t.side2 + t.base = 18 ∧ (t.side1 = 8 ∨ t.base = 8) →
  t.side1 = 8 ∨ t.side1 = 5 :=
sorry

end isosceles_triangle_legs_l155_15564


namespace five_million_times_eight_million_l155_15530

theorem five_million_times_eight_million :
  (5 * (10 : ℕ)^6) * (8 * (10 : ℕ)^6) = 40 * (10 : ℕ)^12 := by
  sorry

end five_million_times_eight_million_l155_15530


namespace smallest_result_l155_15595

def S : Set ℕ := {6, 8, 10, 12, 14, 16}

def process (a b c : ℕ) : ℕ := (a + b) * c - 10

def valid_choice (a b c : ℕ) : Prop :=
  a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_result :
  ∀ a b c : ℕ, valid_choice a b c →
    98 ≤ min (process a b c) (min (process a c b) (process b c a)) :=
by sorry

end smallest_result_l155_15595


namespace max_subjects_per_teacher_l155_15598

theorem max_subjects_per_teacher (maths_teachers physics_teachers chemistry_teachers min_teachers : ℕ)
  (h1 : maths_teachers = 11)
  (h2 : physics_teachers = 8)
  (h3 : chemistry_teachers = 5)
  (h4 : min_teachers = 8) :
  ∃ (max_subjects : ℕ), max_subjects = 3 ∧
    min_teachers * max_subjects ≥ maths_teachers + physics_teachers + chemistry_teachers ∧
    ∀ (x : ℕ), x > max_subjects → min_teachers * x > maths_teachers + physics_teachers + chemistry_teachers :=
by
  sorry

end max_subjects_per_teacher_l155_15598


namespace wendy_packaging_theorem_l155_15544

/-- Represents the number of chocolates Wendy can package in a given time -/
def chocolates_packaged (packaging_rate : ℕ) (packaging_time : ℕ) (work_time : ℕ) : ℕ :=
  (packaging_rate * 12 * (work_time * 60 / packaging_time))

/-- Proves that Wendy can package 1152 chocolates in 4 hours -/
theorem wendy_packaging_theorem :
  chocolates_packaged 2 5 240 = 1152 := by
  sorry

#eval chocolates_packaged 2 5 240

end wendy_packaging_theorem_l155_15544


namespace positive_root_k_values_negative_solution_k_range_l155_15581

-- Define the equation
def equation (x k : ℝ) : Prop :=
  4 / (x + 1) + 3 / (x - 1) = k / (x^2 - 1)

-- Part 1: Positive root case
theorem positive_root_k_values (k : ℝ) :
  (∃ x > 0, equation x k) → (k = 6 ∨ k = -8) := by sorry

-- Part 2: Negative solution case
theorem negative_solution_k_range (k : ℝ) :
  (∃ x < 0, equation x k) → (k < -1 ∧ k ≠ -8) := by sorry

end positive_root_k_values_negative_solution_k_range_l155_15581


namespace fifth_day_income_correct_l155_15543

/-- Calculates the income for the fifth day given the income for four days and the average income for five days. -/
def fifth_day_income (day1 day2 day3 day4 average : ℝ) : ℝ :=
  5 * average - (day1 + day2 + day3 + day4)

/-- Theorem stating that the calculated fifth day income is correct. -/
theorem fifth_day_income_correct (day1 day2 day3 day4 day5 average : ℝ) 
  (h_average : average = (day1 + day2 + day3 + day4 + day5) / 5) :
  fifth_day_income day1 day2 day3 day4 average = day5 := by
  sorry

#eval fifth_day_income 250 400 750 400 460

end fifth_day_income_correct_l155_15543


namespace probability_specific_arrangement_l155_15500

def total_tiles : ℕ := 7
def x_tiles : ℕ := 4
def o_tiles : ℕ := 3

theorem probability_specific_arrangement :
  (1 : ℚ) / (Nat.choose total_tiles x_tiles : ℚ) = 1 / 35 :=
by sorry

end probability_specific_arrangement_l155_15500


namespace max_principals_is_five_l155_15540

/-- Represents the duration of the entire period in years -/
def total_period : ℕ := 15

/-- Represents the length of each principal's term in years -/
def term_length : ℕ := 3

/-- Calculates the maximum number of principals that can serve in the given period -/
def max_principals : ℕ := total_period / term_length

theorem max_principals_is_five : max_principals = 5 := by
  sorry

end max_principals_is_five_l155_15540


namespace expand_expression_l155_15563

theorem expand_expression (x : ℝ) : (7*x + 5) * 3*x^2 = 21*x^3 + 15*x^2 := by
  sorry

end expand_expression_l155_15563


namespace count_integer_points_on_line_l155_15519

/-- A point with integer coordinates -/
structure IntPoint where
  x : Int
  y : Int

/-- Check if a point is strictly between two other points -/
def strictly_between (p q r : IntPoint) : Prop :=
  (p.x < q.x ∧ q.x < r.x) ∨ (r.x < q.x ∧ q.x < p.x)

/-- The line passing through two points -/
def line_through (p q : IntPoint) (r : IntPoint) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- The main theorem -/
theorem count_integer_points_on_line :
  let A : IntPoint := ⟨3, 3⟩
  let B : IntPoint := ⟨120, 150⟩
  ∃! (points : Finset IntPoint),
    (∀ p ∈ points, line_through A B p ∧ strictly_between A p B) ∧
    points.card = 3 := by
  sorry

end count_integer_points_on_line_l155_15519


namespace textbook_profit_l155_15597

/-- The profit of a textbook sale -/
def profit (cost_price selling_price : ℝ) : ℝ :=
  selling_price - cost_price

/-- Theorem: The profit of a textbook is $11 given that its cost price is $44 and its selling price is $55 -/
theorem textbook_profit :
  let cost_price : ℝ := 44
  let selling_price : ℝ := 55
  profit cost_price selling_price = 11 := by
  sorry

end textbook_profit_l155_15597


namespace probability_intersection_independent_events_l155_15562

theorem probability_intersection_independent_events 
  (a b : Set ℝ) 
  (p : Set ℝ → ℝ) 
  (h1 : p a = 5/7) 
  (h2 : p b = 2/5) 
  (h3 : p (a ∩ b) = p a * p b) : 
  p (a ∩ b) = 2/7 := by
sorry

end probability_intersection_independent_events_l155_15562


namespace min_PM_AB_implies_line_AB_l155_15573

/-- Circle M with equation x^2 + y^2 - 2x - 2y - 2 = 0 -/
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- Line l with equation 2x + y + 2 = 0 -/
def line_l (x y : ℝ) : Prop :=
  2*x + y + 2 = 0

/-- Point P on line l -/
structure Point_P where
  x : ℝ
  y : ℝ
  on_line_l : line_l x y

/-- Tangent line from P to circle M -/
def is_tangent (P : Point_P) (A : ℝ × ℝ) : Prop :=
  circle_M A.1 A.2 ∧ 
  ∃ (t : ℝ), A.1 = P.x + t * (A.1 - P.x) ∧ A.2 = P.y + t * (A.2 - P.y)

/-- The equation of line AB: 2x + y + 1 = 0 -/
def line_AB (x y : ℝ) : Prop :=
  2*x + y + 1 = 0

theorem min_PM_AB_implies_line_AB :
  ∀ (P : Point_P) (A B : ℝ × ℝ),
  is_tangent P A → is_tangent P B →
  (∀ (Q : Point_P) (C D : ℝ × ℝ),
    is_tangent Q C → is_tangent Q D →
    (P.x - 1)^2 + (P.y - 1)^2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤
    (Q.x - 1)^2 + (Q.y - 1)^2 * ((C.1 - D.1)^2 + (C.2 - D.2)^2)) →
  line_AB A.1 A.2 ∧ line_AB B.1 B.2 := by
  sorry

end min_PM_AB_implies_line_AB_l155_15573


namespace negation_of_proposition_l155_15599

theorem negation_of_proposition :
  (¬ ∀ n : ℕ, n^2 ≤ 2*n + 5) ↔ (∃ n : ℕ, n^2 > 2*n + 5) := by
  sorry

end negation_of_proposition_l155_15599


namespace pipe_length_difference_l155_15524

theorem pipe_length_difference (total_length shorter_length : ℕ) 
  (h1 : total_length = 68)
  (h2 : shorter_length = 28)
  (h3 : shorter_length < total_length - shorter_length) :
  total_length - shorter_length - shorter_length = 12 :=
by sorry

end pipe_length_difference_l155_15524


namespace min_value_sum_fractions_l155_15522

theorem min_value_sum_fractions (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a + b + c = 1) :
  let u := (3*a^2 - a)/(1 + a^2) + (3*b^2 - b)/(1 + b^2) + (3*c^2 - c)/(1 + c^2)
  u ≥ 0 ∧ (u = 0 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
by sorry

end min_value_sum_fractions_l155_15522


namespace locus_of_M_l155_15585

-- Define the constant k
variable (k : ℝ)

-- Define the coordinates of points A and B
variable (xA yA xB yB : ℝ)

-- Define the coordinates of point M
variable (xM yM : ℝ)

-- Axioms based on the problem conditions
axiom perpendicular_axes : xA * xB + yA * yB = 0
axiom A_on_x_axis : yA = 0
axiom B_on_y_axis : xB = 0
axiom sum_of_distances : Real.sqrt (xA^2 + yA^2) + Real.sqrt (xB^2 + yB^2) = k
axiom M_on_circumcircle : (xM - xA)^2 + (yM - yA)^2 = (xM - xB)^2 + (yM - yB)^2

-- Theorem statement
theorem locus_of_M : 
  (xM - k/2)^2 + (yM - k/2)^2 = k^2/2 := by sorry

end locus_of_M_l155_15585


namespace credit_card_balance_ratio_l155_15566

theorem credit_card_balance_ratio : 
  ∀ (gold_limit : ℝ) (gold_balance : ℝ) (platinum_balance : ℝ),
  gold_limit > 0 →
  platinum_balance = (1/8) * (2 * gold_limit) →
  0.7083333333333334 * (2 * gold_limit) = 2 * gold_limit - (platinum_balance + gold_balance) →
  gold_balance / gold_limit = 1/3 := by
  sorry

end credit_card_balance_ratio_l155_15566


namespace solve_equation_l155_15570

theorem solve_equation : ∃ x : ℝ, 8 * x - (5 * 0.85 / 2.5) = 5.5 ∧ x = 0.9 := by
  sorry

end solve_equation_l155_15570


namespace button_collection_value_l155_15535

theorem button_collection_value (total_buttons : ℕ) (sample_buttons : ℕ) (sample_value : ℚ) :
  total_buttons = 10 →
  sample_buttons = 2 →
  sample_value = 8 →
  (sample_value / sample_buttons) * total_buttons = 40 := by
sorry

end button_collection_value_l155_15535


namespace mikey_jelly_beans_mikey_jelly_beans_holds_l155_15508

/-- Proves that Mikey has 19 jelly beans given the conditions of the problem -/
theorem mikey_jelly_beans : ℕ → ℕ → ℕ → Prop :=
  fun napoleon sedrich mikey =>
    napoleon = 17 →
    sedrich = napoleon + 4 →
    2 * (napoleon + sedrich) = 4 * mikey →
    mikey = 19

/-- The theorem holds for the given values -/
theorem mikey_jelly_beans_holds : 
  ∃ (napoleon sedrich mikey : ℕ), mikey_jelly_beans napoleon sedrich mikey :=
by
  sorry

end mikey_jelly_beans_mikey_jelly_beans_holds_l155_15508


namespace camping_site_campers_l155_15532

theorem camping_site_campers (total : ℕ) (last_week : ℕ) : 
  total = 150 → last_week = 80 → ∃ (three_weeks_ago two_weeks_ago : ℕ), 
    two_weeks_ago = three_weeks_ago + 10 ∧ 
    total = three_weeks_ago + two_weeks_ago + last_week ∧
    two_weeks_ago = 40 := by sorry

end camping_site_campers_l155_15532


namespace slant_height_and_height_not_unique_l155_15549

/-- Represents a right triangular pyramid with a square base -/
structure RightTriangularPyramid where
  base_side : ℝ
  height : ℝ
  slant_height : ℝ

/-- Predicate to check if two pyramids are different -/
def different_pyramids (p1 p2 : RightTriangularPyramid) : Prop :=
  p1.base_side ≠ p2.base_side ∧ p1.height = p2.height ∧ p1.slant_height = p2.slant_height

/-- Theorem stating that slant height and height do not uniquely specify the pyramid -/
theorem slant_height_and_height_not_unique :
  ∃ (p1 p2 : RightTriangularPyramid), different_pyramids p1 p2 := by
  sorry

end slant_height_and_height_not_unique_l155_15549


namespace unique_triangle_function_l155_15501

def IsNonDegenerateTriangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def SatisfiesTriangleCondition (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x > 0 → y > 0 → IsNonDegenerateTriangle x (f y) (f (y + f x - 1))

theorem unique_triangle_function :
  ∃! f : ℕ → ℕ, (∀ x : ℕ, x > 0 → f x > 0) ∧ SatisfiesTriangleCondition f ∧ (∀ x : ℕ, x > 0 → f x = x) :=
sorry

end unique_triangle_function_l155_15501


namespace intersection_line_is_canonical_l155_15577

-- Define the two planes
def plane1 (x y z : ℝ) : Prop := 2*x + 3*y + z + 6 = 0
def plane2 (x y z : ℝ) : Prop := x - 3*y - 2*z + 3 = 0

-- Define the canonical form of the line
def canonical_line (x y z : ℝ) : Prop := (x + 3)/(-3) = y/5 ∧ y/5 = z/(-9)

-- Theorem statement
theorem intersection_line_is_canonical :
  ∀ x y z : ℝ, plane1 x y z → plane2 x y z → canonical_line x y z :=
sorry

end intersection_line_is_canonical_l155_15577


namespace unique_divisible_sum_l155_15517

theorem unique_divisible_sum (p : ℕ) (h_prime : Nat.Prime p) :
  ∃! n : ℕ, (p * n) % (p + n) = 0 :=
sorry

end unique_divisible_sum_l155_15517


namespace spiral_stripe_length_l155_15537

/-- The length of a spiral stripe on a right circular cylinder -/
theorem spiral_stripe_length (base_circumference height : ℝ) (h1 : base_circumference = 18) (h2 : height = 8) :
  Real.sqrt (height^2 + (2 * base_circumference)^2) = Real.sqrt 1360 := by
  sorry

end spiral_stripe_length_l155_15537


namespace book_exchange_ways_l155_15548

theorem book_exchange_ways (n₁ n₂ k : ℕ) (h₁ : n₁ = 6) (h₂ : n₂ = 8) (h₃ : k = 3) : 
  (n₁.choose k) * (n₂.choose k) = 1120 := by
  sorry

end book_exchange_ways_l155_15548


namespace triangle_side_and_angle_l155_15571

open Real

structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

def Triangle.perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

theorem triangle_side_and_angle (t : Triangle) :
  t.perimeter = Real.sqrt 3 + 1 →
  sin t.B + sin t.C = Real.sqrt 3 * sin t.A →
  t.c = 1 ∧
  (t.perimeter = Real.sqrt 3 + 1 →
   sin t.B + sin t.C = Real.sqrt 3 * sin t.A →
   (1/2) * t.a * t.b * sin t.A = (1/3) * sin t.A →
   t.A = π/3) := by
  sorry

end triangle_side_and_angle_l155_15571


namespace unique_root_of_increasing_function_l155_15545

theorem unique_root_of_increasing_function (f : ℝ → ℝ) (h : Monotone f) :
  ∃! x, f x = 0 ∨ (∀ x, f x ≠ 0) :=
by
  sorry

end unique_root_of_increasing_function_l155_15545


namespace average_temperature_l155_15551

def temperatures : List ℤ := [-36, 13, -15, -10]

theorem average_temperature (temps := temperatures) :
  (temps.sum : ℚ) / temps.length = -12 := by
  sorry

end average_temperature_l155_15551


namespace quadratic_roots_distance_l155_15589

/-- Given a quadratic function y = ax² + bx + c satisfying specific conditions,
    prove that the distance between its roots is √17/2 -/
theorem quadratic_roots_distance (a b c : ℝ) : 
  (a*(-1)^2 + b*(-1) + c = -1) →
  (a*0^2 + b*0 + c = -2) →
  (a*1^2 + b*1 + c = 1) →
  let f := fun x => a*x^2 + b*x + c
  let roots := {x : ℝ | f x = 0}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ roots ∧ x₂ ∈ roots ∧ x₁ ≠ x₂ ∧ |x₁ - x₂| = Real.sqrt 17 / 2 :=
by sorry


end quadratic_roots_distance_l155_15589


namespace tree_initial_height_l155_15507

/-- Represents the height of a tree over time -/
def TreeHeight (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (current_age : ℝ) : ℝ :=
  initial_height + growth_rate * (current_age - initial_age)

theorem tree_initial_height :
  ∀ (initial_height : ℝ) (growth_rate : ℝ) (initial_age : ℝ) (current_age : ℝ) (current_height : ℝ),
  growth_rate = 3 →
  initial_age = 1 →
  current_age = 7 →
  current_height = 23 →
  TreeHeight initial_height growth_rate initial_age current_age = current_height →
  initial_height = 5 := by
sorry

end tree_initial_height_l155_15507


namespace max_cut_length_30x30_225pieces_l155_15539

/-- Represents a square board with side length and number of pieces it's cut into -/
structure Board where
  side_length : ℕ
  num_pieces : ℕ

/-- Calculates the maximum possible total length of cuts for a given board -/
def max_cut_length (b : Board) : ℕ :=
  sorry

/-- The theorem stating the maximum cut length for a 30x30 board cut into 225 pieces -/
theorem max_cut_length_30x30_225pieces :
  let b : Board := { side_length := 30, num_pieces := 225 }
  max_cut_length b = 1065 := by
  sorry

end max_cut_length_30x30_225pieces_l155_15539


namespace lattice_points_bound_l155_15554

/-- A convex figure in a 2D plane -/
structure ConvexFigure where
  area : ℝ
  semiperimeter : ℝ
  lattice_points : ℕ

/-- Theorem: For any convex figure, the number of lattice points inside
    is greater than the difference between its area and semiperimeter -/
theorem lattice_points_bound (figure : ConvexFigure) :
  figure.lattice_points > figure.area - figure.semiperimeter := by
  sorry

end lattice_points_bound_l155_15554


namespace square_sum_reciprocal_l155_15550

theorem square_sum_reciprocal (x : ℝ) (h1 : x > 0) (h2 : x + 1/x = Real.sqrt 2020) :
  x^2 + 1/x^2 = 2018 := by
  sorry

end square_sum_reciprocal_l155_15550


namespace divisibility_implies_equality_l155_15574

theorem divisibility_implies_equality (a b : ℕ+) (h : (a + b) ∣ (5 * a + 3 * b)) : a = b := by
  sorry

end divisibility_implies_equality_l155_15574


namespace david_solo_completion_time_l155_15592

/-- The number of days it takes David to complete the job alone -/
def david_solo_days : ℝ := 12

/-- The number of days David works alone before Moore joins -/
def david_solo_work : ℝ := 6

/-- The number of days it takes David and Moore to complete the job together -/
def david_moore_total : ℝ := 6

/-- The number of days it takes David and Moore to complete the remaining job after David works alone -/
def david_moore_remaining : ℝ := 3

theorem david_solo_completion_time :
  (david_solo_work / david_solo_days) + 
  (david_moore_remaining / david_moore_total) = 1 :=
sorry

end david_solo_completion_time_l155_15592


namespace consecutive_odd_integers_sum_l155_15567

/-- Given three consecutive odd integers where the sum of the first and third is 150,
    prove that the second integer is 75. -/
theorem consecutive_odd_integers_sum (n : ℤ) : 
  (∃ (a b c : ℤ), a + 2 = b ∧ b + 2 = c ∧ Odd a ∧ Odd b ∧ Odd c ∧ a + c = 150) →
  b = 75 := by
sorry


end consecutive_odd_integers_sum_l155_15567


namespace f_monotonicity_and_equality_l155_15509

noncomputable def f (x : ℝ) : ℝ := (Real.exp 1) * x / Real.exp x

theorem f_monotonicity_and_equality (e : ℝ) (he : e = Real.exp 1) :
  (∀ x y, x < y → x < 1 → y < 1 → f x < f y) ∧
  (∀ x y, x < y → 1 < x → 1 < y → f y < f x) ∧
  (∀ x, x > 0 → f (1 - x) ≠ f (1 + x)) ∧
  (f (1 - 0) = f (1 + 0)) :=
by sorry

end f_monotonicity_and_equality_l155_15509


namespace green_ball_probability_l155_15511

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The containers in the problem -/
def containers : List Container := [
  ⟨10, 2⟩,  -- Container I
  ⟨3, 5⟩,   -- Container II
  ⟨2, 6⟩,   -- Container III
  ⟨5, 3⟩    -- Container IV
]

/-- The probability of selecting each container -/
def containerProbability : ℚ := 1 / containers.length

theorem green_ball_probability : 
  (containers.map (fun c => containerProbability * greenProbability c)).sum = 23 / 48 := by
  sorry

end green_ball_probability_l155_15511


namespace line_param_values_l155_15506

/-- The line equation y = (1/3)x + 3 parameterized as (x, y) = (-5, r) + t(m, -6) -/
def line_equation (x y : ℝ) : Prop := y = (1/3) * x + 3

/-- The parameterization of the line -/
def line_param (t r m : ℝ) (x y : ℝ) : Prop :=
  x = -5 + t * m ∧ y = r + t * (-6)

/-- Theorem stating that r = 4/3 and m = 0 for the given line and parameterization -/
theorem line_param_values :
  ∃ (r m : ℝ), (∀ (x y t : ℝ), line_equation x y ↔ line_param t r m x y) ∧ r = 4/3 ∧ m = 0 := by
  sorry

end line_param_values_l155_15506


namespace jiaqi_pe_grade_l155_15560

/-- Calculates the final grade based on component scores and weights -/
def calculate_grade (extracurricular_score : ℝ) (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  0.2 * extracurricular_score + 0.3 * midterm_score + 0.5 * final_score

/-- Proves that Jiaqi's physical education grade for the semester is 95.3 points -/
theorem jiaqi_pe_grade :
  let max_score : ℝ := 100
  let extracurricular_weight : ℝ := 0.2
  let midterm_weight : ℝ := 0.3
  let final_weight : ℝ := 0.5
  let jiaqi_extracurricular : ℝ := 96
  let jiaqi_midterm : ℝ := 92
  let jiaqi_final : ℝ := 97
  calculate_grade jiaqi_extracurricular jiaqi_midterm jiaqi_final = 95.3 := by
  sorry

end jiaqi_pe_grade_l155_15560


namespace imaginary_unit_power_l155_15502

theorem imaginary_unit_power (i : ℂ) : i^2 = -1 → i^2033 = i := by sorry

end imaginary_unit_power_l155_15502


namespace class_gender_ratio_l155_15552

theorem class_gender_ratio (female_count : ℕ) (male_count : ℕ) 
  (h1 : female_count = 28)
  (h2 : female_count = male_count + 6) :
  let total_count := female_count + male_count
  (female_count : ℚ) / (male_count : ℚ) = 14 / 11 ∧ 
  (male_count : ℚ) / (total_count : ℚ) = 11 / 25 := by
sorry

end class_gender_ratio_l155_15552


namespace cinema_visitors_l155_15542

theorem cinema_visitors (female_visitors : ℕ) (female_office_workers : ℕ) 
  (male_excess : ℕ) (male_non_workers : ℕ) 
  (h1 : female_visitors = 1518)
  (h2 : female_office_workers = 536)
  (h3 : male_excess = 525)
  (h4 : male_non_workers = 1257) :
  female_office_workers + (female_visitors + male_excess - male_non_workers) = 1322 := by
  sorry

end cinema_visitors_l155_15542


namespace euler_product_theorem_l155_15596

theorem euler_product_theorem : ∀ (z₁ z₂ : ℂ),
  (z₁ = Complex.exp (Complex.I * Real.pi / 3)) →
  (z₂ = Complex.exp (Complex.I * Real.pi / 6)) →
  z₁ * z₂ = Complex.I := by
  sorry

end euler_product_theorem_l155_15596


namespace conditional_probability_first_class_l155_15561

/-- A box containing products -/
structure Box where
  total : Nat
  firstClass : Nat
  secondClass : Nat
  h_sum : firstClass + secondClass = total

/-- The probability of selecting a first-class product on the second draw
    given that a first-class product was selected on the first draw -/
def conditionalProbability (b : Box) : ℚ :=
  (b.firstClass - 1 : ℚ) / (b.total - 1 : ℚ)

theorem conditional_probability_first_class
  (b : Box)
  (h_total : b.total = 4)
  (h_first : b.firstClass = 3)
  (h_second : b.secondClass = 1) :
  conditionalProbability b = 2/3 := by
  sorry

end conditional_probability_first_class_l155_15561


namespace diegos_stamp_collection_cost_l155_15591

def brazil_stamps : ℕ := 6 + 9
def peru_stamps : ℕ := 8 + 5
def colombia_stamps : ℕ := 7 + 6

def brazil_cost : ℚ := 0.07
def peru_cost : ℚ := 0.05
def colombia_cost : ℚ := 0.07

def total_cost : ℚ := 
  brazil_stamps * brazil_cost + 
  peru_stamps * peru_cost + 
  colombia_stamps * colombia_cost

theorem diegos_stamp_collection_cost : total_cost = 2.61 := by
  sorry

end diegos_stamp_collection_cost_l155_15591


namespace N_mod_52_l155_15594

/-- The number formed by concatenating integers from 1 to 51 -/
def N : ℕ := sorry

/-- The remainder when N is divided by 52 -/
def remainder : ℕ := N % 52

theorem N_mod_52 : remainder = 13 := by sorry

end N_mod_52_l155_15594


namespace rectangle_dimensions_l155_15580

theorem rectangle_dimensions : ∃ (x y : ℝ), 
  y = x + 3 ∧ 
  2 * (2 * (x + y)) = x * y ∧ 
  x = 8 ∧ 
  y = 11 := by
sorry

end rectangle_dimensions_l155_15580


namespace mckenna_start_time_l155_15565

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Mckenna's work schedule -/
structure WorkSchedule where
  officeEndTime : Time
  meetingEndTime : Time
  workDuration : Nat
  totalWorkDuration : Nat

/-- Calculate the difference between two times in hours -/
def timeDifference (t1 t2 : Time) : Nat :=
  sorry

/-- Calculate the time after adding hours to a given time -/
def addHours (t : Time) (hours : Nat) : Time :=
  sorry

theorem mckenna_start_time (schedule : WorkSchedule)
  (h1 : schedule.officeEndTime = ⟨11, 0⟩)
  (h2 : schedule.meetingEndTime = ⟨13, 0⟩)
  (h3 : schedule.workDuration = 2)
  (h4 : schedule.totalWorkDuration = 7) :
  timeDifference ⟨8, 0⟩ (addHours schedule.meetingEndTime schedule.workDuration) = schedule.totalWorkDuration :=
sorry

end mckenna_start_time_l155_15565


namespace third_file_size_l155_15538

theorem third_file_size 
  (internet_speed : ℝ) 
  (download_time : ℝ) 
  (file1_size : ℝ) 
  (file2_size : ℝ) 
  (h1 : internet_speed = 2) 
  (h2 : download_time = 2 * 60) 
  (h3 : file1_size = 80) 
  (h4 : file2_size = 90) : 
  ∃ (file3_size : ℝ), 
    file3_size = internet_speed * download_time - (file1_size + file2_size) ∧ 
    file3_size = 70 := by
  sorry

end third_file_size_l155_15538


namespace red_pepper_weight_l155_15533

theorem red_pepper_weight (total_weight green_weight : ℚ) 
  (h1 : total_weight = 0.66)
  (h2 : green_weight = 0.33) :
  total_weight - green_weight = 0.33 := by
sorry

end red_pepper_weight_l155_15533


namespace hyperbola_focus_to_asymptote_distance_l155_15531

/-- Given a hyperbola and a parabola with coinciding foci, prove the distance from the hyperbola's focus to its asymptote -/
theorem hyperbola_focus_to_asymptote_distance 
  (a : ℝ) 
  (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / 5 = 1 → (∃ c : ℝ, x^2 / c^2 - y^2 / 5 = 1 ∧ c^2 = 4)) 
  (h2 : ∀ x y : ℝ, y^2 = 12*x → (∃ p : ℝ × ℝ, p = (3, 0))) : 
  ∃ d : ℝ, d = Real.sqrt 5 := by
  sorry

end hyperbola_focus_to_asymptote_distance_l155_15531


namespace bug_probability_after_12_meters_l155_15586

/-- Probability of the bug being at vertex A after crawling n meters -/
def P (n : ℕ) : ℚ :=
  match n with
  | 0 => 1
  | n + 1 => (1 - P n) / 3

/-- Edge length of the tetrahedron in meters -/
def edgeLength : ℕ := 2

/-- Number of edges traversed after 12 meters -/
def edgesTraversed : ℕ := 12 / edgeLength

theorem bug_probability_after_12_meters :
  P edgesTraversed = 44287 / 177147 := by sorry

end bug_probability_after_12_meters_l155_15586


namespace expected_replacement_seeds_l155_15553

theorem expected_replacement_seeds :
  let germination_prob : ℝ := 0.9
  let initial_seeds : ℕ := 1000
  let replacement_per_failure : ℕ := 2
  let non_germination_prob : ℝ := 1 - germination_prob
  let expected_non_germinating : ℝ := initial_seeds * non_germination_prob
  let expected_replacements : ℝ := expected_non_germinating * replacement_per_failure
  expected_replacements = 200 := by sorry

end expected_replacement_seeds_l155_15553


namespace triangle_side_length_l155_15569

theorem triangle_side_length (a b c : ℝ) (A : ℝ) : 
  a = 7 → b = 8 → A = π/3 → (c = 3 ∨ c = 5) → 
  a^2 = b^2 + c^2 - 2*b*c*Real.cos A := by sorry

end triangle_side_length_l155_15569


namespace absolute_value_inequality_l155_15576

theorem absolute_value_inequality (x : ℝ) : 
  |x - 2| + |x + 3| < 8 ↔ -4.5 < x ∧ x < 3.5 := by
  sorry

end absolute_value_inequality_l155_15576


namespace consecutive_even_numbers_sum_18_l155_15512

theorem consecutive_even_numbers_sum_18 (n : ℤ) : 
  (n - 2) + n + (n + 2) = 18 → (n - 2 = 4 ∧ n = 6 ∧ n + 2 = 8) := by
sorry

end consecutive_even_numbers_sum_18_l155_15512


namespace track_completion_time_l155_15516

/-- Time to complete a circular track -/
def complete_track_time (half_track_time : ℝ) : ℝ :=
  2 * half_track_time

/-- Theorem: The time to complete the circular track is 6 minutes -/
theorem track_completion_time :
  let half_track_time : ℝ := 3
  complete_track_time half_track_time = 6 := by
  sorry


end track_completion_time_l155_15516


namespace log_inequality_l155_15590

theorem log_inequality : Real.log 2 / Real.log 3 < Real.log 3 / Real.log 2 ∧ 
                         Real.log 3 / Real.log 2 < Real.log 5 / Real.log 2 := by
  sorry

end log_inequality_l155_15590


namespace managers_salary_l155_15588

/-- Proves that the manager's salary is 14100 given the conditions -/
theorem managers_salary (num_employees : ℕ) (avg_salary : ℕ) (salary_increase : ℕ) :
  num_employees = 20 →
  avg_salary = 1500 →
  salary_increase = 600 →
  (num_employees * avg_salary + (num_employees + 1) * salary_increase) = 14100 := by
  sorry

#check managers_salary

end managers_salary_l155_15588


namespace johns_total_income_this_year_l155_15582

/-- Calculates the total income (salary + bonus) for the current year given the previous year's salary and bonus, and the current year's salary. -/
def totalIncomeCurrentYear (prevSalary prevBonus currSalary : ℕ) : ℕ :=
  let bonusRate := prevBonus / prevSalary
  let currBonus := currSalary * bonusRate
  currSalary + currBonus

/-- Theorem stating that John's total income this year is $220,000 -/
theorem johns_total_income_this_year :
  totalIncomeCurrentYear 100000 10000 200000 = 220000 := by
  sorry

end johns_total_income_this_year_l155_15582


namespace factorial_difference_l155_15558

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 8 = 3588480 := by
  sorry

end factorial_difference_l155_15558


namespace cell_phone_company_customers_l155_15536

theorem cell_phone_company_customers (us_customers other_customers : ℕ) 
  (h1 : us_customers = 723)
  (h2 : other_customers = 6699) :
  us_customers + other_customers = 7422 := by
  sorry

end cell_phone_company_customers_l155_15536


namespace chocolate_bars_distribution_l155_15525

theorem chocolate_bars_distribution (total_bars : ℕ) (num_small_boxes : ℕ) 
  (h1 : total_bars = 504) (h2 : num_small_boxes = 18) :
  total_bars / num_small_boxes = 28 := by
  sorry

end chocolate_bars_distribution_l155_15525


namespace second_fragment_speed_is_52_l155_15546

/-- Represents the motion of a firecracker that explodes into two fragments -/
structure Firecracker where
  initial_speed : ℝ
  explosion_time : ℝ
  gravity : ℝ
  first_fragment_horizontal_speed : ℝ

/-- Calculates the speed of the second fragment after explosion -/
def second_fragment_speed (f : Firecracker) : ℝ :=
  sorry

/-- Theorem stating that the speed of the second fragment is 52 m/s -/
theorem second_fragment_speed_is_52 (f : Firecracker) 
  (h1 : f.initial_speed = 20)
  (h2 : f.explosion_time = 3)
  (h3 : f.gravity = 10)
  (h4 : f.first_fragment_horizontal_speed = 48) :
  second_fragment_speed f = 52 :=
  sorry

end second_fragment_speed_is_52_l155_15546


namespace rowing_coach_votes_l155_15587

theorem rowing_coach_votes (num_coaches : ℕ) (votes_per_rower : ℕ) (votes_per_coach : ℕ) :
  num_coaches = 50 →
  votes_per_rower = 4 →
  votes_per_coach = 7 →
  ∃ (num_rowers : ℕ), num_rowers * votes_per_rower = num_coaches * votes_per_coach ∧ 
                       num_rowers = 88 := by
  sorry

end rowing_coach_votes_l155_15587


namespace stars_per_bottle_l155_15513

/-- Given Shiela's paper stars and number of classmates, prove the number of stars per bottle. -/
theorem stars_per_bottle (total_stars : ℕ) (num_classmates : ℕ) 
  (h1 : total_stars = 45) 
  (h2 : num_classmates = 9) : 
  total_stars / num_classmates = 5 := by
  sorry

#check stars_per_bottle

end stars_per_bottle_l155_15513


namespace mixture_weight_l155_15520

/-- Given a mixture of zinc, copper, and silver in the ratio 9 : 11 : 7,
    where 27 kg of zinc is used, the total weight of the mixture is 81 kg. -/
theorem mixture_weight (zinc copper silver : ℕ) (zinc_weight : ℝ) :
  zinc = 9 →
  copper = 11 →
  silver = 7 →
  zinc_weight = 27 →
  (zinc_weight / zinc) * (zinc + copper + silver) = 81 :=
by sorry

end mixture_weight_l155_15520


namespace bicycle_wheels_l155_15568

theorem bicycle_wheels (num_bicycles num_tricycles tricycle_wheels total_wheels : ℕ) 
  (h1 : num_bicycles = 24)
  (h2 : num_tricycles = 14)
  (h3 : tricycle_wheels = 3)
  (h4 : total_wheels = 90)
  : ∃ bicycle_wheels : ℕ, 
    bicycle_wheels * num_bicycles + tricycle_wheels * num_tricycles = total_wheels ∧ 
    bicycle_wheels = 2 := by
  sorry

end bicycle_wheels_l155_15568


namespace functional_equation_solution_l155_15555

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → f (x * f y) = f (x * y) + x) →
  (∀ x : ℝ, x > 0 → f x = x + 1) := by
  sorry

end functional_equation_solution_l155_15555


namespace f_eight_equals_zero_l155_15528

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem f_eight_equals_zero
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_period : ∀ x, f (x + 2) = -f x) :
  f 8 = 0 := by
  sorry

end f_eight_equals_zero_l155_15528


namespace yan_distance_ratio_l155_15559

/-- Yan's problem setup -/
structure YanProblem where
  w : ℝ  -- Yan's walking speed
  x : ℝ  -- Distance from Yan to home
  z : ℝ  -- Distance from Yan to school
  h_positive : w > 0 ∧ x > 0 ∧ z > 0  -- Positive distances and speed
  h_between : x + z > 0  -- Yan is between home and school
  h_equal_time : z / w = x / w + (x + z) / (5 * w)  -- Equal time condition

/-- The main theorem: ratio of distances is 2/3 -/
theorem yan_distance_ratio (p : YanProblem) : p.x / p.z = 2 / 3 := by
  sorry


end yan_distance_ratio_l155_15559


namespace least_number_with_remainder_seven_l155_15526

theorem least_number_with_remainder_seven (n : ℕ) : n = 1547 ↔ 
  (∀ d ∈ ({11, 17, 21, 29, 35} : Set ℕ), n % d = 7) ∧
  (∀ m < n, ∃ d ∈ ({11, 17, 21, 29, 35} : Set ℕ), m % d ≠ 7) :=
by sorry

end least_number_with_remainder_seven_l155_15526


namespace gcd_333_481_l155_15505

theorem gcd_333_481 : Nat.gcd 333 481 = 37 := by
  sorry

end gcd_333_481_l155_15505


namespace isosceles_trapezoid_side_length_l155_15541

/-- An isosceles trapezoid with given base lengths and area -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  area : ℝ

/-- The length of the side of an isosceles trapezoid -/
def side_length (t : IsoscelesTrapezoid) : ℝ :=
  sorry

theorem isosceles_trapezoid_side_length :
  ∀ t : IsoscelesTrapezoid,
    t.base1 = 11 ∧ t.base2 = 17 ∧ t.area = 56 →
    side_length t = 5 := by
  sorry

end isosceles_trapezoid_side_length_l155_15541


namespace b_work_time_l155_15504

-- Define the work rates for A, B, and C
def work_rate_A : ℚ := 1 / 5
def work_rate_BC : ℚ := 1 / 3
def work_rate_AC : ℚ := 1 / 2

-- Theorem to prove
theorem b_work_time (work_rate_B : ℚ) : 
  work_rate_A + (work_rate_BC - work_rate_B) = work_rate_AC → 
  (1 : ℚ) / work_rate_B = 30 := by
  sorry

end b_work_time_l155_15504


namespace sample_size_is_selected_size_l155_15527

/-- Represents the total number of first-year high school students -/
def population_size : ℕ := 1320

/-- Represents the number of students selected for measurement -/
def selected_size : ℕ := 220

/-- Theorem stating that the sample size is equal to the number of selected students -/
theorem sample_size_is_selected_size : 
  selected_size = 220 := by sorry

end sample_size_is_selected_size_l155_15527


namespace fib_F15_units_digit_l155_15578

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of F_{F₁₅} is 5 -/
theorem fib_F15_units_digit :
  unitsDigit (fib (fib 15)) = 5 := by
  sorry

end fib_F15_units_digit_l155_15578


namespace movie_ticket_ratio_l155_15534

def horror_tickets : ℕ := 93
def romance_tickets : ℕ := 25
def ticket_difference : ℕ := 18

theorem movie_ticket_ratio :
  (horror_tickets : ℚ) / romance_tickets = 93 / 25 ∧
  horror_tickets = romance_tickets + ticket_difference :=
sorry

end movie_ticket_ratio_l155_15534


namespace hyperbola_focal_length_l155_15557

/-- Given a hyperbola and a parabola with specific properties, 
    prove that the focal length of the hyperbola is 2√5 -/
theorem hyperbola_focal_length 
  (a b p : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hp : p > 0) 
  (h_vertex_focus : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2/a^2 - y₁^2/b^2 = 1 ∧ 
    y₂^2 = 2*p*x₂ ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) 
  (h_asymptote_directrix : ∃ (k : ℝ), 
    (-2)^2/a^2 - (-1)^2/b^2 = k^2 ∧ 
    -2 = -p/2) : 
  2 * (a^2 + b^2).sqrt = 2 * Real.sqrt 5 := by
  sorry

end hyperbola_focal_length_l155_15557


namespace lineup_ways_eq_choose_four_from_fourteen_l155_15510

/-- The number of ways to choose an 8-player lineup from 18 players,
    including two sets of twins that must be in the lineup. -/
def lineup_ways (total_players : ℕ) (lineup_size : ℕ) (twin_pairs : ℕ) : ℕ :=
  Nat.choose (total_players - 2 * twin_pairs) (lineup_size - 2 * twin_pairs)

/-- Theorem stating that the number of ways to choose the lineup
    is equal to choosing 4 from 14 players. -/
theorem lineup_ways_eq_choose_four_from_fourteen :
  lineup_ways 18 8 2 = Nat.choose 14 4 := by
  sorry

end lineup_ways_eq_choose_four_from_fourteen_l155_15510


namespace rational_equation_solution_l155_15579

theorem rational_equation_solution (x : ℚ) :
  x ≠ 3 →
  (x - 3) / (x + 2) + (3 * x - 6) / (x - 3) = 2 →
  x = 1 / 2 := by
sorry

end rational_equation_solution_l155_15579


namespace bucket_capacity_reduction_l155_15575

theorem bucket_capacity_reduction (original_buckets : ℕ) (capacity_ratio : ℚ) : 
  original_buckets = 10 →
  capacity_ratio = 2 / 5 →
  (original_buckets : ℚ) / capacity_ratio = 25 := by
sorry

end bucket_capacity_reduction_l155_15575


namespace jug_problem_l155_15593

theorem jug_problem (Cx Cy : ℝ) (h1 : Cx > 0) (h2 : Cy > 0) : 
  (1/6 : ℝ) * Cx = (2/3 : ℝ) * Cy → 
  (1/9 : ℝ) * Cx = (1/3 : ℝ) * Cy - (1/3 : ℝ) * Cy := by
sorry

end jug_problem_l155_15593


namespace opposite_numbers_cube_root_l155_15529

theorem opposite_numbers_cube_root (x y : ℝ) : 
  y = -x → 3 * x - 4 * y = 7 → (x * y) ^ (1/3 : ℝ) = -1 := by sorry

end opposite_numbers_cube_root_l155_15529


namespace min_digits_of_m_l155_15515

theorem min_digits_of_m (n : ℤ) : 
  let m := (n - 1001) * (n - 2001) * (n - 2002) * (n - 3001) * (n - 3002) * (n - 3003)
  m > 0 → m ≥ 10^10 :=
by sorry

end min_digits_of_m_l155_15515


namespace math_contest_grade11_score_l155_15584

theorem math_contest_grade11_score (n : ℕ) (grade11_score : ℝ) :
  let grade11_count : ℝ := 0.2 * n
  let grade12_count : ℝ := 0.8 * n
  let overall_average : ℝ := 78
  let grade12_average : ℝ := 75
  (grade11_count * grade11_score + grade12_count * grade12_average) / n = overall_average →
  grade11_score = 90 := by
sorry

end math_contest_grade11_score_l155_15584


namespace battle_station_staffing_l155_15583

/-- The number of job openings --/
def num_openings : ℕ := 6

/-- The total number of resumes received --/
def total_resumes : ℕ := 36

/-- The number of suitable candidates after removing one-third --/
def suitable_candidates : ℕ := total_resumes - (total_resumes / 3)

/-- The number of ways to staff the battle station --/
def staffing_ways : ℕ := 255024240

theorem battle_station_staffing :
  (suitable_candidates.factorial) / ((suitable_candidates - num_openings).factorial) = staffing_ways := by
  sorry

end battle_station_staffing_l155_15583


namespace unpainted_face_area_l155_15518

/-- A right circular cylinder with given dimensions -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- The unpainted face created by slicing the cylinder -/
def UnpaintedFace (c : Cylinder) (arcAngle : ℝ) : ℝ := sorry

/-- Theorem stating the area of the unpainted face for the given cylinder and arc angle -/
theorem unpainted_face_area (c : Cylinder) (h1 : c.radius = 6) (h2 : c.height = 8) (h3 : arcAngle = 2 * π / 3) :
  UnpaintedFace c arcAngle = 16 * π + 27 * Real.sqrt 3 := by
  sorry

end unpainted_face_area_l155_15518


namespace quadratic_equation_solution_l155_15514

theorem quadratic_equation_solution (m : ℝ) : 
  (m - 1 ≠ 0) → (m^2 - 3*m + 2 = 0) → (m = 2) := by
  sorry

#check quadratic_equation_solution

end quadratic_equation_solution_l155_15514


namespace problem_solution_l155_15556

def f (a : ℝ) (x : ℝ) : ℝ := |x - 2| - |x + a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x + x > 0 ↔ (-3 < x ∧ x < 1) ∨ x > 3) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x ≤ 3) ↔ -5 ≤ a ∧ a ≤ 1) :=
by sorry

end problem_solution_l155_15556


namespace oliver_fruit_consumption_l155_15523

/-- The number of fruits Oliver consumed -/
def fruits_consumed (initial_cherries initial_strawberries initial_blueberries
                     remaining_cherries remaining_strawberries remaining_blueberries : ℝ) : ℝ :=
  (initial_cherries - remaining_cherries) +
  (initial_strawberries - remaining_strawberries) +
  (initial_blueberries - remaining_blueberries)

/-- Theorem stating that Oliver consumed 17.2 fruits in total -/
theorem oliver_fruit_consumption :
  fruits_consumed 16.5 10.7 20.2 6.3 8.4 15.5 = 17.2 := by
  sorry

end oliver_fruit_consumption_l155_15523


namespace overlap_percentage_l155_15547

theorem overlap_percentage (square_side : ℝ) (rect_width rect_length : ℝ) 
  (overlap_rect_width overlap_rect_length : ℝ) :
  square_side = 12 →
  rect_width = 9 →
  rect_length = 12 →
  overlap_rect_width = 12 →
  overlap_rect_length = 18 →
  (((square_side + rect_width - overlap_rect_length) * rect_width) / 
   (overlap_rect_width * overlap_rect_length)) * 100 = 12.5 := by
  sorry

end overlap_percentage_l155_15547


namespace mower_blades_cost_is_47_l155_15521

/-- The amount Mike made mowing lawns -/
def total_earnings : ℕ := 101

/-- The number of games Mike could buy with the remaining money -/
def num_games : ℕ := 9

/-- The cost of each game -/
def game_cost : ℕ := 6

/-- The amount Mike spent on new mower blades -/
def mower_blades_cost : ℕ := total_earnings - (num_games * game_cost)

theorem mower_blades_cost_is_47 : mower_blades_cost = 47 := by
  sorry

end mower_blades_cost_is_47_l155_15521
