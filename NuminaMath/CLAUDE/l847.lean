import Mathlib

namespace NUMINAMATH_CALUDE_odd_cube_plus_multiple_l847_84785

theorem odd_cube_plus_multiple (p m : ℤ) (hp : Odd p) :
  Odd (p^3 + m*p) ↔ Even m :=
sorry

end NUMINAMATH_CALUDE_odd_cube_plus_multiple_l847_84785


namespace NUMINAMATH_CALUDE_product_102_105_l847_84775

theorem product_102_105 : 102 * 105 = 10710 := by
  sorry

end NUMINAMATH_CALUDE_product_102_105_l847_84775


namespace NUMINAMATH_CALUDE_min_value_theorem_l847_84781

theorem min_value_theorem (x : ℝ) (h : x > 4) :
  (x + 5) / Real.sqrt (x - 4) ≥ 6 ∧ ∃ y : ℝ, y > 4 ∧ (y + 5) / Real.sqrt (y - 4) = 6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l847_84781


namespace NUMINAMATH_CALUDE_problem_statements_l847_84723

theorem problem_statements (a b : ℝ) :
  (ab < 0 ∧ (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) → a / b = -1) ∧
  (a + b < 0 ∧ ab > 0 → |2*a + 3*b| = -(2*a + 3*b)) ∧
  ¬(∀ a b : ℝ, |a - b| + a - b = 0 → b > a) ∧
  ¬(∀ a b : ℝ, |a| > |b| → (a + b) * (a - b) < 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_statements_l847_84723


namespace NUMINAMATH_CALUDE_race_length_correct_l847_84792

/-- Represents the race scenario -/
structure Race where
  length : ℝ
  samTime : ℝ
  johnTime : ℝ
  headStart : ℝ

/-- The given race conditions -/
def givenRace : Race where
  length := 126
  samTime := 13
  johnTime := 18
  headStart := 35

/-- Theorem stating that the given race length satisfies all conditions -/
theorem race_length_correct (r : Race) : 
  r.samTime = 13 ∧ 
  r.johnTime = r.samTime + 5 ∧ 
  r.headStart = 35 ∧
  r.length / r.samTime * r.samTime = r.length / r.johnTime * r.samTime + r.headStart →
  r.length = 126 := by
  sorry

#check race_length_correct givenRace

end NUMINAMATH_CALUDE_race_length_correct_l847_84792


namespace NUMINAMATH_CALUDE_correct_average_l847_84750

theorem correct_average (n : ℕ) (initial_avg : ℚ) (increase : ℚ) : 
  n = 10 →
  initial_avg = 5 →
  increase = 10 →
  (n : ℚ) * initial_avg + increase = n * 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l847_84750


namespace NUMINAMATH_CALUDE_kyle_origami_stars_l847_84753

theorem kyle_origami_stars (initial_bottles : ℕ) (additional_bottles : ℕ) (stars_per_bottle : ℕ) :
  initial_bottles = 4 →
  additional_bottles = 5 →
  stars_per_bottle = 25 →
  (initial_bottles + additional_bottles) * stars_per_bottle = 225 := by
  sorry

end NUMINAMATH_CALUDE_kyle_origami_stars_l847_84753


namespace NUMINAMATH_CALUDE_square_divisible_by_six_between_30_and_150_l847_84715

theorem square_divisible_by_six_between_30_and_150 (x : ℕ) :
  (∃ n : ℕ, x = n^2) →  -- x is a square number
  x % 6 = 0 →           -- x is divisible by 6
  30 < x →              -- x is greater than 30
  x < 150 →             -- x is less than 150
  x = 36 ∨ x = 144 :=   -- x is either 36 or 144
by sorry

end NUMINAMATH_CALUDE_square_divisible_by_six_between_30_and_150_l847_84715


namespace NUMINAMATH_CALUDE_a_minus_b_values_l847_84745

theorem a_minus_b_values (a b : ℤ) (ha : |a| = 7) (hb : |b| = 5) (hab : a < b) :
  a - b = -12 ∨ a - b = -2 :=
sorry

end NUMINAMATH_CALUDE_a_minus_b_values_l847_84745


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l847_84739

theorem simplify_and_evaluate_expression (a : ℝ) (h : a = 3) :
  (1 - (a - 2) / (a^2 - 4)) / ((a^2 + a) / (a^2 + 4*a + 4)) = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l847_84739


namespace NUMINAMATH_CALUDE_eve_distance_difference_l847_84713

theorem eve_distance_difference : 
  let ran_distance : ℝ := 0.7
  let walked_distance : ℝ := 0.6
  ran_distance - walked_distance = 0.1 :=
by sorry

end NUMINAMATH_CALUDE_eve_distance_difference_l847_84713


namespace NUMINAMATH_CALUDE_min_value_expression_l847_84707

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 1) :
  ∃ m : ℝ, m = 3 ∧ ∀ x y : ℝ, x > 0 → y > 0 → x * y = 1 →
    x^2 + y^2 + 4 / (x + y)^2 ≥ m :=
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l847_84707


namespace NUMINAMATH_CALUDE_f_satisfies_data_points_l847_84768

/-- The function that relates x and y --/
def f (x : ℕ) : ℕ := x^2 + x

/-- The set of data points from the table --/
def data_points : List (ℕ × ℕ) := [(1, 2), (2, 6), (3, 12), (4, 20), (5, 30)]

/-- Theorem stating that the function f satisfies all data points --/
theorem f_satisfies_data_points : ∀ (point : ℕ × ℕ), point ∈ data_points → f point.1 = point.2 := by
  sorry

#check f_satisfies_data_points

end NUMINAMATH_CALUDE_f_satisfies_data_points_l847_84768


namespace NUMINAMATH_CALUDE_product_xy_is_eight_l847_84741

theorem product_xy_is_eight (x y : ℝ) 
  (h1 : (8:ℝ)^x / (4:ℝ)^(x+y) = 64)
  (h2 : (16:ℝ)^(x+y) / (4:ℝ)^(4*y) = 256) : 
  x * y = 8 := by
sorry

end NUMINAMATH_CALUDE_product_xy_is_eight_l847_84741


namespace NUMINAMATH_CALUDE_simplify_radical_product_l847_84749

theorem simplify_radical_product (x : ℝ) (h : x > 0) :
  2 * Real.sqrt (50 * x^3) * Real.sqrt (45 * x^5) * Real.sqrt (98 * x^7) = 420 * x^7 * Real.sqrt (5 * x) := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l847_84749


namespace NUMINAMATH_CALUDE_train_crossing_time_l847_84726

/-- The time taken for a train to cross a platform of equal length -/
theorem train_crossing_time (train_length platform_length : ℝ) (train_speed : ℝ) : 
  train_length = platform_length →
  train_length = 750 →
  train_speed = 90 * 1000 / 3600 →
  (train_length + platform_length) / train_speed = 60 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l847_84726


namespace NUMINAMATH_CALUDE_unique_solution_inequality_l847_84719

theorem unique_solution_inequality (x : ℝ) :
  (x > 0 ∧ x * Real.sqrt (16 - x) + Real.sqrt (16 * x - x^3) ≥ 16) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_l847_84719


namespace NUMINAMATH_CALUDE_inequality_property_l847_84786

theorem inequality_property (a b : ℝ) (h : a < 0 ∧ 0 < b) : 1 / a < 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inequality_property_l847_84786


namespace NUMINAMATH_CALUDE_trapezoid_shorter_diagonal_l847_84711

structure Trapezoid where
  EF : ℝ
  GH : ℝ
  side1 : ℝ
  side2 : ℝ
  acute_E : Bool
  acute_F : Bool

def shorter_diagonal (t : Trapezoid) : ℝ := sorry

theorem trapezoid_shorter_diagonal 
  (t : Trapezoid) 
  (h1 : t.EF = 20) 
  (h2 : t.GH = 26) 
  (h3 : t.side1 = 13) 
  (h4 : t.side2 = 15) 
  (h5 : t.acute_E = true) 
  (h6 : t.acute_F = true) : 
  shorter_diagonal t = Real.sqrt 1496 / 3 := by sorry

end NUMINAMATH_CALUDE_trapezoid_shorter_diagonal_l847_84711


namespace NUMINAMATH_CALUDE_three_times_root_equation_iff_roots_l847_84706

/-- A quadratic equation ax^2 + bx + c = 0 (a ≠ 0) with two distinct real roots -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- Definition of a "3 times root equation" -/
def is_three_times_root_equation (eq : QuadraticEquation) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    eq.a * r₁^2 + eq.b * r₁ + eq.c = 0 ∧
    eq.a * r₂^2 + eq.b * r₂ + eq.c = 0 ∧
    r₂ = 3 * r₁

/-- Theorem: A quadratic equation is a "3 times root equation" iff its roots satisfy r2 = 3r1 -/
theorem three_times_root_equation_iff_roots (eq : QuadraticEquation) :
  is_three_times_root_equation eq ↔
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
    eq.a * r₁^2 + eq.b * r₁ + eq.c = 0 ∧
    eq.a * r₂^2 + eq.b * r₂ + eq.c = 0 ∧
    r₂ = 3 * r₁ :=
by sorry


end NUMINAMATH_CALUDE_three_times_root_equation_iff_roots_l847_84706


namespace NUMINAMATH_CALUDE_circle_area_l847_84784

theorem circle_area (d : ℝ) (A : ℝ) (π : ℝ) (h1 : d = 10) (h2 : π = Real.pi) :
  A = π * 25 → A = (π * d^2) / 4 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_area_l847_84784


namespace NUMINAMATH_CALUDE_sallys_shopping_problem_l847_84799

/-- Sally's shopping problem -/
theorem sallys_shopping_problem 
  (peaches_price_after_coupon : ℝ) 
  (coupon_value : ℝ)
  (total_spent : ℝ)
  (h1 : peaches_price_after_coupon = 12.32)
  (h2 : coupon_value = 3)
  (h3 : total_spent = 23.86) :
  total_spent - (peaches_price_after_coupon + coupon_value) = 8.54 := by
sorry

end NUMINAMATH_CALUDE_sallys_shopping_problem_l847_84799


namespace NUMINAMATH_CALUDE_quadratic_properties_l847_84757

-- Define the quadratic equation
def quadratic_equation (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

-- Define the discriminant
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

theorem quadratic_properties (a b c : ℝ) (h : a ≠ 0) :
  (a - b + c = 0 → discriminant a b c ≥ 0) ∧
  (quadratic_equation a b c 1 ∧ quadratic_equation a b c 2 → 2*a - c = 0) ∧
  ((∃ x y : ℝ, x ≠ y ∧ a * x^2 + c = 0 ∧ a * y^2 + c = 0) →
    ∃ z : ℝ, quadratic_equation a b c z) ∧
  (b = 2*a + c → ∃ x y : ℝ, x ≠ y ∧ quadratic_equation a b c x ∧ quadratic_equation a b c y) :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l847_84757


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l847_84712

theorem largest_multiple_of_9_under_100 : ∃ (n : ℕ), n = 99 ∧ 
  (∀ m : ℕ, m < 100 ∧ 9 ∣ m → m ≤ n) := by
  sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l847_84712


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_nine_l847_84730

theorem sqrt_sum_equals_nine :
  Real.sqrt ((5 - 3 * Real.sqrt 2) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 2) ^ 2) - 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_nine_l847_84730


namespace NUMINAMATH_CALUDE_points_coplanar_iff_b_eq_neg_one_l847_84780

/-- Given four points in 3D space, prove they are coplanar iff b = -1 --/
theorem points_coplanar_iff_b_eq_neg_one (b : ℝ) :
  let p1 := (0 : ℝ × ℝ × ℝ)
  let p2 := (1, b, 0)
  let p3 := (0, 1, b^2)
  let p4 := (b^2, 0, 1)
  (∃ (a b c d : ℝ), a • p1 + b • p2 + c • p3 + d • p4 = 0 ∧ (a, b, c, d) ≠ 0) ↔ b = -1 := by
  sorry

#check points_coplanar_iff_b_eq_neg_one

end NUMINAMATH_CALUDE_points_coplanar_iff_b_eq_neg_one_l847_84780


namespace NUMINAMATH_CALUDE_time_saved_without_tide_change_l847_84705

/-- The time saved by a rower if the tide direction had not changed -/
theorem time_saved_without_tide_change 
  (speed_with_tide : ℝ) 
  (speed_against_tide : ℝ) 
  (distance_after_reversal : ℝ) 
  (h1 : speed_with_tide = 5)
  (h2 : speed_against_tide = 4)
  (h3 : distance_after_reversal = 40) : 
  distance_after_reversal / speed_with_tide - distance_after_reversal / speed_against_tide = 2 := by
  sorry

end NUMINAMATH_CALUDE_time_saved_without_tide_change_l847_84705


namespace NUMINAMATH_CALUDE_line_equation_and_x_intercept_l847_84737

-- Define the points A and B
def A : ℝ × ℝ := (-2, -3)
def B : ℝ × ℝ := (3, 0)

-- Define the line l
def l (x y : ℝ) : Prop := 5 * x + 3 * y + 2 = 0

-- Define symmetry about a line
def symmetric_about_line (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (M : ℝ × ℝ), l M.1 M.2 ∧ 
  (M.1 = (A.1 + B.1) / 2) ∧ 
  (M.2 = (A.2 + B.2) / 2)

-- Theorem statement
theorem line_equation_and_x_intercept :
  symmetric_about_line A B l →
  (∀ x y, l x y ↔ 5 * x + 3 * y + 2 = 0) ∧
  (∃ x, l x 0 ∧ x = -2/5) :=
sorry

end NUMINAMATH_CALUDE_line_equation_and_x_intercept_l847_84737


namespace NUMINAMATH_CALUDE_kekai_money_left_l847_84731

def garage_sale_problem (num_shirts num_pants : ℕ) (price_shirt price_pants : ℚ) : ℚ :=
  let total_earned := num_shirts * price_shirt + num_pants * price_pants
  let amount_to_parents := total_earned / 2
  total_earned - amount_to_parents

theorem kekai_money_left :
  garage_sale_problem 5 5 1 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_kekai_money_left_l847_84731


namespace NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l847_84770

theorem four_digit_number_with_specific_remainders :
  ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧
    n % 7 = 3 ∧
    n % 10 = 6 ∧
    n % 12 = 8 ∧
    n % 13 = 2 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_number_with_specific_remainders_l847_84770


namespace NUMINAMATH_CALUDE_sqrt8_same_type_as_sqrt2_l847_84702

-- Define a function to check if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- Define a function to check if a number is of the same type of quadratic root as √2
def same_type_as_sqrt2 (n : ℕ) : Prop :=
  ¬ (is_perfect_square n) ∧ ∃ k : ℕ, n = 2 * k ∧ ¬ (is_perfect_square k)

-- Theorem statement
theorem sqrt8_same_type_as_sqrt2 :
  same_type_as_sqrt2 8 ∧
  ¬ (same_type_as_sqrt2 4) ∧
  ¬ (same_type_as_sqrt2 12) ∧
  ¬ (same_type_as_sqrt2 24) :=
sorry

end NUMINAMATH_CALUDE_sqrt8_same_type_as_sqrt2_l847_84702


namespace NUMINAMATH_CALUDE_product_difference_bound_l847_84751

theorem product_difference_bound (x y a b m ε : ℝ) 
  (ε_pos : ε > 0) (m_pos : m > 0) 
  (h1 : |x - a| < ε / (2 * m))
  (h2 : |y - b| < ε / (2 * |a|))
  (h3 : 0 < y) (h4 : y < m) : 
  |x * y - a * b| < ε := by
sorry

end NUMINAMATH_CALUDE_product_difference_bound_l847_84751


namespace NUMINAMATH_CALUDE_triangle_ABC_point_C_l847_84704

-- Define the points
def A : ℝ × ℝ := (8, 5)
def B : ℝ × ℝ := (-1, -2)
def D : ℝ × ℝ := (2, 2)

-- Define the triangle ABC
def triangle_ABC (C : ℝ × ℝ) : Prop :=
  -- AB = AC (isosceles triangle)
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - C.1)^2 + (A.2 - C.2)^2 ∧
  -- D is on BC
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2) ∧
  -- AD is perpendicular to BC
  (A.1 - D.1) * (B.1 - C.1) + (A.2 - D.2) * (B.2 - C.2) = 0

-- Theorem statement
theorem triangle_ABC_point_C : 
  ∃ C : ℝ × ℝ, triangle_ABC C ∧ C = (5, 6) := by sorry

end NUMINAMATH_CALUDE_triangle_ABC_point_C_l847_84704


namespace NUMINAMATH_CALUDE_increasing_function_m_range_l847_84748

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + m

theorem increasing_function_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, -2 < x₁ ∧ x₁ < x₂ → f m x₁ < f m x₂) →
  m ∈ Set.Ici 4 :=
sorry

end NUMINAMATH_CALUDE_increasing_function_m_range_l847_84748


namespace NUMINAMATH_CALUDE_sum_of_s_and_t_l847_84729

theorem sum_of_s_and_t (s t : ℕ+) (h : s * (s - t) = 29) : s + t = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_s_and_t_l847_84729


namespace NUMINAMATH_CALUDE_circle_properties_l847_84771

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

-- Theorem statement
theorem circle_properties :
  ∃ (center_x center_y radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    center_x = 3 ∧ center_y = 0 ∧ radius = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_properties_l847_84771


namespace NUMINAMATH_CALUDE_catch_second_messenger_first_is_optimal_l847_84766

/-- Represents a person's position and movement --/
structure Person where
  position : ℝ
  speed : ℝ
  startTime : ℝ

/-- Represents the problem setup --/
structure ProblemSetup where
  messenger1 : Person
  messenger2 : Person
  cyclist : Person
  startPoint : ℝ

/-- Calculates the time needed for the cyclist to complete the task --/
def timeToComplete (setup : ProblemSetup) (catchSecondFirst : Bool) : ℝ :=
  sorry

/-- Theorem stating that catching the second messenger first is optimal --/
theorem catch_second_messenger_first_is_optimal (setup : ProblemSetup) :
  setup.messenger1.startTime + 0.25 = setup.messenger2.startTime →
  setup.messenger1.speed = setup.messenger2.speed →
  setup.cyclist.speed > setup.messenger1.speed →
  setup.messenger1.position < setup.startPoint →
  setup.messenger2.position > setup.startPoint →
  timeToComplete setup true ≤ timeToComplete setup false :=
sorry

end NUMINAMATH_CALUDE_catch_second_messenger_first_is_optimal_l847_84766


namespace NUMINAMATH_CALUDE_card_game_total_l847_84769

theorem card_game_total (total : ℕ) (ellis orion : ℕ) : 
  ellis = (11 : ℕ) * total / 20 →
  orion = (9 : ℕ) * total / 20 →
  ellis = orion + 50 →
  total = 500 := by
sorry

end NUMINAMATH_CALUDE_card_game_total_l847_84769


namespace NUMINAMATH_CALUDE_train_length_l847_84783

theorem train_length (tree_time : ℝ) (platform_time : ℝ) (platform_length : ℝ) :
  tree_time = 120 →
  platform_time = 220 →
  platform_length = 1000 →
  let train_length := (platform_time * platform_length) / (platform_time - tree_time)
  train_length = 1200 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l847_84783


namespace NUMINAMATH_CALUDE_quadratic_root_theorem_l847_84717

theorem quadratic_root_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hroot : a * 2^2 - (a + b + c) * 2 + (b + c) = 0) :
  ∃ x : ℝ, x ≠ 2 ∧ a * x^2 - (a + b + c) * x + (b + c) = 0 ∧ x = (b + c - a) / a :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_theorem_l847_84717


namespace NUMINAMATH_CALUDE_inequality_range_theorem_l847_84794

/-- The range of values for the real number a that satisfies the inequality
    2*ln(x) ≥ -x^2 + ax - 3 for all x ∈ (0, +∞) is (-∞, 4]. -/
theorem inequality_range_theorem (a : ℝ) : 
  (∀ x > 0, 2 * Real.log x ≥ -x^2 + a*x - 3) ↔ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_range_theorem_l847_84794


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l847_84782

-- Define the type for points in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the type for lines in 2D space
structure Line2D where
  f : ℝ → ℝ → ℝ

-- Define the property of a point being on a line
def PointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.f p.x p.y = 0

-- Define the property of two lines being parallel
def ParallelLines (l1 l2 : Line2D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), l1.f x y = k * l2.f x y

theorem parallel_line_through_point
  (l : Line2D) (p1 p2 : Point2D)
  (h1 : PointOnLine p1 l)
  (h2 : ¬PointOnLine p2 l) :
  let l2 : Line2D := { f := λ x y => l.f x y - l.f p2.x p2.y }
  ParallelLines l l2 ∧ PointOnLine p2 l2 :=
by
  sorry


end NUMINAMATH_CALUDE_parallel_line_through_point_l847_84782


namespace NUMINAMATH_CALUDE_player_two_wins_l847_84734

/-- Number of contacts in the microcircuit -/
def num_contacts : ℕ := 2000

/-- Total number of wires initially -/
def total_wires : ℕ := num_contacts * (num_contacts - 1) / 2

/-- Represents a player in the game -/
inductive Player
| One
| Two

/-- Represents the state of the game -/
structure GameState where
  remaining_wires : ℕ
  current_player : Player

/-- Defines a valid move in the game -/
def valid_move (state : GameState) (wires_cut : ℕ) : Prop :=
  match state.current_player with
  | Player.One => wires_cut = 1
  | Player.Two => wires_cut = 2 ∨ wires_cut = 3

/-- Defines the winning condition -/
def is_winning_state (state : GameState) : Prop :=
  state.remaining_wires = 0

/-- Theorem stating that Player Two has a winning strategy -/
theorem player_two_wins : 
  ∃ (strategy : GameState → ℕ), 
    ∀ (game : GameState), 
      game.remaining_wires > 0 → 
      game.current_player = Player.Two → 
      valid_move game (strategy game) ∧ 
      (∀ (opponent_move : ℕ), 
        valid_move (GameState.mk (game.remaining_wires - strategy game) Player.One) opponent_move → 
        is_winning_state (GameState.mk (game.remaining_wires - strategy game - opponent_move) Player.Two)) :=
sorry

end NUMINAMATH_CALUDE_player_two_wins_l847_84734


namespace NUMINAMATH_CALUDE_term_2020_2187_position_l847_84774

/-- Sequence term type -/
structure SeqTerm where
  k : Nat
  n : Nat
  h : k % 3 ≠ 0 ∧ 2 ≤ k ∧ k < 3^n

/-- Position of a term in the sequence -/
def termPosition (t : SeqTerm) : Nat :=
  t.k - (t.k / 3)

/-- The main theorem -/
theorem term_2020_2187_position :
  ∃ t : SeqTerm, t.k = 2020 ∧ t.n = 7 ∧ termPosition t = 1347 := by
  sorry

end NUMINAMATH_CALUDE_term_2020_2187_position_l847_84774


namespace NUMINAMATH_CALUDE_total_population_l847_84754

def wildlife_park (num_lions : ℕ) (num_leopards : ℕ) (num_adult_elephants : ℕ) (num_zebras : ℕ) : Prop :=
  (num_lions = 2 * num_leopards) ∧
  (num_adult_elephants = (num_lions * 3 / 4 + num_leopards * 3 / 5) / 2) ∧
  (num_zebras = num_adult_elephants + num_leopards) ∧
  (num_lions = 200)

theorem total_population (num_lions num_leopards num_adult_elephants num_zebras : ℕ) :
  wildlife_park num_lions num_leopards num_adult_elephants num_zebras →
  num_lions + num_leopards + (num_adult_elephants + 100) + num_zebras = 710 :=
by
  sorry

#check total_population

end NUMINAMATH_CALUDE_total_population_l847_84754


namespace NUMINAMATH_CALUDE_cubic_function_one_zero_l847_84772

/-- Given a cubic function f(x) = -x^3 - x on the interval [m, n] where f(m) * f(n) < 0,
    f(x) has exactly one zero in the open interval (m, n). -/
theorem cubic_function_one_zero (m n : ℝ) (hm : m < n)
  (f : ℝ → ℝ) (hf : ∀ x, f x = -x^3 - x)
  (h_neg : f m * f n < 0) :
  ∃! x, m < x ∧ x < n ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_one_zero_l847_84772


namespace NUMINAMATH_CALUDE_circle_area_ratio_after_tripling_diameter_l847_84714

theorem circle_area_ratio_after_tripling_diameter :
  ∀ (r : ℝ), r > 0 →
  (π * r^2) / (π * (3*r)^2) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_after_tripling_diameter_l847_84714


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l847_84742

/-- Two lines are parallel if their slopes are equal -/
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

theorem parallel_lines_m_value (m : ℝ) :
  let l₁ : ℝ → ℝ → Prop := λ x y => 2 * x + m * y + 1 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y => y = 3 * x - 1
  parallel (-1/2) (1/3) → m = -2/3 := by sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l847_84742


namespace NUMINAMATH_CALUDE_square_side_length_l847_84733

theorem square_side_length (rectangle_side1 rectangle_side2 : ℝ) 
  (h1 : rectangle_side1 = 9)
  (h2 : rectangle_side2 = 16) :
  ∃ (square_side : ℝ), 
    square_side * square_side = rectangle_side1 * rectangle_side2 ∧ 
    square_side = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l847_84733


namespace NUMINAMATH_CALUDE_club_members_neither_subject_l847_84724

theorem club_members_neither_subject (total : ℕ) (cs : ℕ) (bio : ℕ) (both : ℕ) 
  (h1 : total = 150)
  (h2 : cs = 80)
  (h3 : bio = 50)
  (h4 : both = 15) :
  total - (cs + bio - both) = 35 := by
  sorry

end NUMINAMATH_CALUDE_club_members_neither_subject_l847_84724


namespace NUMINAMATH_CALUDE_buckingham_palace_visitors_l847_84773

theorem buckingham_palace_visitors (current_day_visitors previous_day_visitors : ℕ) 
  (h1 : current_day_visitors = 661) 
  (h2 : previous_day_visitors = 600) : 
  current_day_visitors - previous_day_visitors = 61 := by
  sorry

end NUMINAMATH_CALUDE_buckingham_palace_visitors_l847_84773


namespace NUMINAMATH_CALUDE_equally_spaced_number_line_l847_84709

theorem equally_spaced_number_line (total_distance : ℝ) (num_steps : ℕ) (step_to_z : ℕ) : 
  total_distance = 16 → num_steps = 4 → step_to_z = 2 →
  let step_length := total_distance / num_steps
  let z := step_to_z * step_length
  z = 8 := by
  sorry

end NUMINAMATH_CALUDE_equally_spaced_number_line_l847_84709


namespace NUMINAMATH_CALUDE_probability_exact_scenario_verify_conditions_l847_84728

/-- The probability of drawing all red balls by the 4th draw in a specific scenario -/
def probability_all_red_by_fourth_draw (total_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) : ℚ :=
  let total_ways := (total_balls.choose 4)
  let favorable_ways := (red_balls.choose 2) * (white_balls.choose 2)
  (favorable_ways : ℚ) / total_ways

/-- The main theorem stating the probability for the given scenario -/
theorem probability_exact_scenario : 
  probability_all_red_by_fourth_draw 10 8 2 = 4338 / 91125 := by
  sorry

/-- Verifies that the conditions of the problem are met -/
theorem verify_conditions : 
  10 = 8 + 2 ∧ 
  8 ≥ 0 ∧ 
  2 ≥ 0 ∧
  10 > 0 := by
  sorry

end NUMINAMATH_CALUDE_probability_exact_scenario_verify_conditions_l847_84728


namespace NUMINAMATH_CALUDE_double_counted_page_l847_84738

/-- Given a book with 62 pages, prove that if the sum of all page numbers
    plus an additional count of one page number equals 1997,
    then the page number that was counted twice is 44. -/
theorem double_counted_page (n : ℕ) (x : ℕ) : 
  n = 62 → 
  (n * (n + 1)) / 2 + x = 1997 → 
  x = 44 := by
sorry

end NUMINAMATH_CALUDE_double_counted_page_l847_84738


namespace NUMINAMATH_CALUDE_range_of_m_l847_84708

/-- A function is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function has a minimum positive period of p if f(x + p) = f(x) for all x,
    and p is the smallest positive number with this property -/
def HasMinPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  (∀ x, f (x + p) = f x) ∧ ∀ q, 0 < q → q < p → ∃ x, f (x + q) ≠ f x

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
    (h_odd : IsOdd f)
    (h_period : HasMinPeriod f 3)
    (h_2015 : f 2015 > 1)
    (h_1 : f 1 = (2*m + 3)/(m - 1)) :
    -2/3 < m ∧ m < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l847_84708


namespace NUMINAMATH_CALUDE_xiaohong_total_score_l847_84759

/-- Calculates the total score based on midterm and final exam scores -/
def total_score (midterm_weight : ℝ) (final_weight : ℝ) (midterm_score : ℝ) (final_score : ℝ) : ℝ :=
  midterm_weight * midterm_score + final_weight * final_score

theorem xiaohong_total_score :
  let midterm_weight : ℝ := 0.4
  let final_weight : ℝ := 0.6
  let midterm_score : ℝ := 80
  let final_score : ℝ := 90
  total_score midterm_weight final_weight midterm_score final_score = 86 := by
  sorry

#eval total_score 0.4 0.6 80 90

end NUMINAMATH_CALUDE_xiaohong_total_score_l847_84759


namespace NUMINAMATH_CALUDE_at_least_one_real_root_l847_84764

theorem at_least_one_real_root (m : ℝ) : 
  ∃ x : ℝ, (x^2 - 5*x + m = 0) ∨ (2*x^2 + x + 6 - m = 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_real_root_l847_84764


namespace NUMINAMATH_CALUDE_profit_percentage_is_40_percent_l847_84791

/-- The percentage of puppies that can be sold for a greater profit -/
def profitable_puppies_percentage (total_puppies : ℕ) (puppies_with_more_than_4_spots : ℕ) : ℚ :=
  (puppies_with_more_than_4_spots : ℚ) / (total_puppies : ℚ) * 100

/-- Theorem stating that the percentage of puppies that can be sold for a greater profit is 40% -/
theorem profit_percentage_is_40_percent :
  profitable_puppies_percentage 20 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_profit_percentage_is_40_percent_l847_84791


namespace NUMINAMATH_CALUDE_soup_bins_calculation_l847_84746

/-- Given a canned food drive with different types of food, calculate the number of bins of soup. -/
theorem soup_bins_calculation (total_bins vegetables_bins pasta_bins : ℝ) 
  (h1 : vegetables_bins = 0.125)
  (h2 : pasta_bins = 0.5)
  (h3 : total_bins = 0.75) :
  total_bins - (vegetables_bins + pasta_bins) = 0.125 := by
  sorry

#check soup_bins_calculation

end NUMINAMATH_CALUDE_soup_bins_calculation_l847_84746


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l847_84779

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a parabola with parameter p -/
structure Parabola where
  p : ℝ
  h_pos : p > 0

/-- Theorem stating that under given conditions, the eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity_sqrt_two (h : Hyperbola) (p : Parabola)
  (h_focus : h.a * h.a - h.b * h.b = p.p * h.a) -- Right focus of hyperbola coincides with focus of parabola
  (h_intersection : ∃ A B : ℝ × ℝ, 
    A.1^2 / h.a^2 - A.2^2 / h.b^2 = 1 ∧
    B.1^2 / h.a^2 - B.2^2 / h.b^2 = 1 ∧
    A.1 = -p.p/2 ∧ B.1 = -p.p/2) -- Directrix of parabola intersects hyperbola at A and B
  (h_asymptotes : ∃ C D : ℝ × ℝ,
    C.2 = h.b / h.a * C.1 ∧
    D.2 = -h.b / h.a * D.1) -- Asymptotes of hyperbola intersect at C and D
  (h_distance : ∃ A B C D : ℝ × ℝ,
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = 2 * ((A.1 - B.1)^2 + (A.2 - B.2)^2)) -- |CD| = √2|AB|
  : Real.sqrt ((h.a^2 - h.b^2) / h.a^2) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_sqrt_two_l847_84779


namespace NUMINAMATH_CALUDE_g_50_equals_zero_l847_84736

-- Define φ(n) as the number of positive integers not exceeding n that are coprime to n
def phi (n : ℕ) : ℕ := sorry

-- Define g(n) that satisfies the given condition
def g (n : ℕ) : ℤ := sorry

-- Define the sum of g(d) over all positive divisors d of n
def sum_g_divisors (n : ℕ) : ℤ := sorry

-- State the condition that g(n) satisfies
axiom g_condition (n : ℕ) : sum_g_divisors n = phi n

-- Theorem to prove
theorem g_50_equals_zero : g 50 = 0 := by sorry

end NUMINAMATH_CALUDE_g_50_equals_zero_l847_84736


namespace NUMINAMATH_CALUDE_jose_alisson_difference_l847_84755

/-- Represents the scores of three students in a test -/
structure TestScores where
  jose : ℕ
  meghan : ℕ
  alisson : ℕ

/-- Properties of the test and scores -/
def valid_scores (s : TestScores) : Prop :=
  s.meghan = s.jose - 20 ∧
  s.jose > s.alisson ∧
  s.jose = 90 ∧
  s.jose + s.meghan + s.alisson = 210

/-- Theorem stating the difference between Jose's and Alisson's scores -/
theorem jose_alisson_difference (s : TestScores) 
  (h : valid_scores s) : s.jose - s.alisson = 40 := by
  sorry

end NUMINAMATH_CALUDE_jose_alisson_difference_l847_84755


namespace NUMINAMATH_CALUDE_product_of_square_roots_l847_84747

theorem product_of_square_roots (m : ℝ) :
  Real.sqrt (15 * m) * Real.sqrt (3 * m^2) * Real.sqrt (8 * m^3) = 6 * m^3 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_product_of_square_roots_l847_84747


namespace NUMINAMATH_CALUDE_alfonso_solution_l847_84797

/-- Alfonso's weekly earnings and financial goals --/
def alfonso_problem (weekday_earnings : ℕ) (weekend_earnings : ℕ) 
  (total_cost : ℕ) (current_savings : ℕ) (desired_remaining : ℕ) : Prop :=
  let weekly_earnings := 5 * weekday_earnings + 2 * weekend_earnings
  let total_needed := total_cost - current_savings + desired_remaining
  let weeks_needed := (total_needed + weekly_earnings - 1) / weekly_earnings
  weeks_needed = 10

/-- Theorem stating the solution to Alfonso's problem --/
theorem alfonso_solution : 
  alfonso_problem 6 8 460 40 20 := by sorry

end NUMINAMATH_CALUDE_alfonso_solution_l847_84797


namespace NUMINAMATH_CALUDE_multiple_of_x_l847_84796

theorem multiple_of_x (x y k : ℤ) 
  (eq1 : k * x + y = 34)
  (eq2 : 2 * x - y = 20)
  (y_sq : y^2 = 4) :
  k = 4 := by
sorry

end NUMINAMATH_CALUDE_multiple_of_x_l847_84796


namespace NUMINAMATH_CALUDE_percent_of_125_l847_84758

theorem percent_of_125 : ∃ p : ℚ, p * 125 / 100 = 70 ∧ p = 56 := by sorry

end NUMINAMATH_CALUDE_percent_of_125_l847_84758


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l847_84744

/-- An isosceles triangle with specific properties -/
structure IsoscelesTriangle where
  -- The length of the two equal sides
  a : ℝ
  -- The length of the base
  b : ℝ
  -- The height of the triangle
  h : ℝ
  -- The radius of the inscribed circle
  r : ℝ
  -- The perimeter is 56
  perimeter_eq : a + a + b = 56
  -- The radius is 2/7 of the height
  radius_height_ratio : r = 2/7 * h
  -- The height relates to sides by Pythagorean theorem
  height_pythagorean : h^2 + (b/2)^2 = a^2
  -- The area can be calculated using sides and radius
  area_eq : (a + a + b) * r / 2 = b * h / 2

/-- The sides of the isosceles triangle with the given properties are 16, 20, and 20 -/
theorem isosceles_triangle_sides (t : IsoscelesTriangle) : t.a = 20 ∧ t.b = 16 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l847_84744


namespace NUMINAMATH_CALUDE_batsman_average_batsman_average_proof_l847_84718

theorem batsman_average (total_innings : ℕ) (last_innings_score : ℕ) (average_increase : ℕ) : ℕ :=
  let prev_average := 30
  let new_average := prev_average + average_increase
  new_average

#check batsman_average 10 60 3 = 33

theorem batsman_average_proof 
  (total_innings : ℕ) 
  (last_innings_score : ℕ) 
  (average_increase : ℕ) 
  (h1 : total_innings = 10)
  (h2 : last_innings_score = 60)
  (h3 : average_increase = 3) :
  batsman_average total_innings last_innings_score average_increase = 33 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_batsman_average_proof_l847_84718


namespace NUMINAMATH_CALUDE_sin_negative_45_degrees_l847_84761

theorem sin_negative_45_degrees :
  Real.sin ((-45 : ℝ) * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_45_degrees_l847_84761


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_l847_84778

/-- The total cost of typing a manuscript with given revision rates and page counts. -/
theorem manuscript_typing_cost
  (initial_rate : ℕ)
  (revision_rate : ℕ)
  (total_pages : ℕ)
  (once_revised_pages : ℕ)
  (twice_revised_pages : ℕ)
  (h1 : initial_rate = 5)
  (h2 : revision_rate = 3)
  (h3 : total_pages = 100)
  (h4 : once_revised_pages = 30)
  (h5 : twice_revised_pages = 20) :
  initial_rate * total_pages +
  revision_rate * once_revised_pages +
  2 * revision_rate * twice_revised_pages = 710 := by
sorry


end NUMINAMATH_CALUDE_manuscript_typing_cost_l847_84778


namespace NUMINAMATH_CALUDE_sum_of_numbers_l847_84732

theorem sum_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y = 16) (h4 : 1 / x = 3 * (1 / y)) : 
  x + y = 16 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l847_84732


namespace NUMINAMATH_CALUDE_tax_rate_calculation_l847_84700

/-- Tax calculation problem -/
theorem tax_rate_calculation (total_value : ℝ) (tax_free_threshold : ℝ) (tax_paid : ℝ) :
  total_value = 1720 →
  tax_free_threshold = 600 →
  tax_paid = 112 →
  (tax_paid / (total_value - tax_free_threshold)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_calculation_l847_84700


namespace NUMINAMATH_CALUDE_tangent_and_inequality_conditions_l847_84798

noncomputable def f (x : ℝ) := Real.exp (2 * x)
def g (k : ℝ) (x : ℝ) := k * x + 1

theorem tangent_and_inequality_conditions (k : ℝ) :
  (∃ t : ℝ, (f t = g k t ∧ (deriv f) t = k)) ↔ k = 2 ∧
  (k > 0 → (∃ m : ℝ, m > 0 ∧ ∀ x : ℝ, 0 < x → x < m → |f x - g k x| > 2 * x) ↔ k > 4) :=
sorry

end NUMINAMATH_CALUDE_tangent_and_inequality_conditions_l847_84798


namespace NUMINAMATH_CALUDE_count_integers_squared_between_200_and_400_l847_84762

theorem count_integers_squared_between_200_and_400 :
  (Finset.filter (fun x => 200 ≤ x^2 ∧ x^2 ≤ 400) (Finset.range 401)).card = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_squared_between_200_and_400_l847_84762


namespace NUMINAMATH_CALUDE_newspaper_delivery_patterns_l847_84790

/-- Represents the number of valid newspaper delivery patterns for n houses -/
def D : ℕ → ℕ
| 0 => 1  -- Base case, one way to deliver to zero houses
| 1 => 2  -- Two ways to deliver to one house (deliver or not)
| 2 => 4  -- Four ways to deliver to two houses
| n + 3 => D (n + 2) + D (n + 1) + D n  -- Recurrence relation

/-- The condition that the last house must receive a newspaper -/
def lastHouseDelivery (n : ℕ) : ℕ := D (n - 1)

/-- The number of houses on the lane -/
def numHouses : ℕ := 12

/-- The theorem stating the number of valid delivery patterns for 12 houses -/
theorem newspaper_delivery_patterns :
  lastHouseDelivery numHouses = 927 := by sorry

end NUMINAMATH_CALUDE_newspaper_delivery_patterns_l847_84790


namespace NUMINAMATH_CALUDE_factor_expression_l847_84763

theorem factor_expression (b : ℝ) : 145 * b^2 + 29 * b = 29 * b * (5 * b + 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l847_84763


namespace NUMINAMATH_CALUDE_deposit_percentage_l847_84752

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) :
  deposit = 140 →
  remaining = 1260 →
  (deposit / (deposit + remaining)) * 100 = 10 := by
sorry

end NUMINAMATH_CALUDE_deposit_percentage_l847_84752


namespace NUMINAMATH_CALUDE_difference_of_two_numbers_l847_84760

theorem difference_of_two_numbers (x y : ℝ) 
  (sum_eq : x + y = 30) 
  (product_eq : x * y = 140) : 
  |x - y| = Real.sqrt 340 := by sorry

end NUMINAMATH_CALUDE_difference_of_two_numbers_l847_84760


namespace NUMINAMATH_CALUDE_an_is_arithmetic_sequence_l847_84721

/-- Definition of an arithmetic sequence's general term -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ n, a n = m * n + b

/-- The sequence defined by an = 3n - 1 -/
def a (n : ℕ) : ℝ := 3 * n - 1

/-- Theorem: The sequence defined by an = 3n - 1 is an arithmetic sequence -/
theorem an_is_arithmetic_sequence : is_arithmetic_sequence a := by
  sorry

end NUMINAMATH_CALUDE_an_is_arithmetic_sequence_l847_84721


namespace NUMINAMATH_CALUDE_truck_distance_before_meeting_l847_84740

/-- The distance between two trucks one minute before they meet, given their initial separation and speeds -/
theorem truck_distance_before_meeting
  (initial_distance : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (h1 : initial_distance = 4)
  (h2 : speed_A = 45)
  (h3 : speed_B = 60)
  : ∃ (d : ℝ), d = 250 / 1000 ∧ d = initial_distance + speed_A * (1 / 60) - speed_B * (1 / 60) :=
by sorry

end NUMINAMATH_CALUDE_truck_distance_before_meeting_l847_84740


namespace NUMINAMATH_CALUDE_recipe_total_cups_l847_84743

/-- Represents the ratio of ingredients in a recipe -/
structure RecipeRatio where
  butter : ℕ
  flour : ℕ
  sugar : ℕ

/-- Calculates the total cups of ingredients used in a recipe -/
def total_cups (ratio : RecipeRatio) (sugar_cups : ℕ) : ℕ :=
  let part_value := sugar_cups / ratio.sugar
  (ratio.butter + ratio.flour + ratio.sugar) * part_value

/-- Theorem: Given a recipe with ratio 2:7:5 and 10 cups of sugar, the total is 28 cups -/
theorem recipe_total_cups :
  let ratio := RecipeRatio.mk 2 7 5
  total_cups ratio 10 = 28 := by
  sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l847_84743


namespace NUMINAMATH_CALUDE_oranges_for_three_rubles_l847_84776

/-- Given that 25 oranges cost as many rubles as can be bought for 1 ruble,
    prove that 15 oranges can be bought for 3 rubles -/
theorem oranges_for_three_rubles : ∀ x : ℝ,
  (25 : ℝ) / x = x →  -- 25 oranges cost x rubles, and x oranges can be bought for 1 ruble
  (3 : ℝ) * x = 15 :=  -- 15 oranges can be bought for 3 rubles
by
  sorry

end NUMINAMATH_CALUDE_oranges_for_three_rubles_l847_84776


namespace NUMINAMATH_CALUDE_simplify_expression_l847_84765

theorem simplify_expression (a b c : ℝ) : a - (a - b + c) = b - c := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l847_84765


namespace NUMINAMATH_CALUDE_f_5_equals_neg_2_l847_84777

-- Define the inverse function f⁻¹
def f_inv (x : ℝ) : ℝ := 1 + x^2

-- State the theorem
theorem f_5_equals_neg_2 (f : ℝ → ℝ) (h1 : ∀ x < 0, f_inv (f x) = x) (h2 : ∀ y, y < 0 → f (f_inv y) = y) : 
  f 5 = -2 := by
  sorry

end NUMINAMATH_CALUDE_f_5_equals_neg_2_l847_84777


namespace NUMINAMATH_CALUDE_hybrid_rice_scientific_notation_l847_84735

theorem hybrid_rice_scientific_notation :
  ∃ n : ℕ, 250000000 = (2.5 : ℝ) * (10 : ℝ) ^ n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_hybrid_rice_scientific_notation_l847_84735


namespace NUMINAMATH_CALUDE_debt_calculation_l847_84727

theorem debt_calculation (initial_debt additional_borrowing : ℕ) :
  initial_debt = 20 →
  additional_borrowing = 15 →
  initial_debt + additional_borrowing = 35 :=
by sorry

end NUMINAMATH_CALUDE_debt_calculation_l847_84727


namespace NUMINAMATH_CALUDE_sam_cleaner_meet_twice_l847_84703

/-- Represents the movement of Sam and the street cleaner on a path with benches --/
structure PathMovement where
  sam_speed : ℝ
  cleaner_speed : ℝ
  bench_distance : ℝ
  cleaner_stop_time : ℝ

/-- Calculates the number of times Sam and the cleaner meet --/
def number_of_meetings (movement : PathMovement) : ℕ :=
  sorry

/-- The specific scenario described in the problem --/
def problem_scenario : PathMovement :=
  { sam_speed := 3
  , cleaner_speed := 9
  , bench_distance := 300
  , cleaner_stop_time := 40 }

/-- Theorem stating that Sam and the cleaner meet exactly twice --/
theorem sam_cleaner_meet_twice :
  number_of_meetings problem_scenario = 2 := by
  sorry

end NUMINAMATH_CALUDE_sam_cleaner_meet_twice_l847_84703


namespace NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l847_84722

theorem sphere_surface_area_rectangular_solid (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let diagonal := Real.sqrt (a^2 + b^2 + c^2)
  let radius := diagonal / 2
  let surface_area := 4 * Real.pi * radius^2
  surface_area = 200 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_rectangular_solid_l847_84722


namespace NUMINAMATH_CALUDE_sum_reciprocals_equal_two_point_five_l847_84725

theorem sum_reciprocals_equal_two_point_five
  (a b c d e : ℝ)
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1) (he : e ≠ -1)
  (ω : ℂ)
  (hω1 : ω^3 = 1)
  (hω2 : ω ≠ 1)
  (h : (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) + (1 / (e + ω)) = 5 / (2*ω)) :
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) + (1 / (e + 1)) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_equal_two_point_five_l847_84725


namespace NUMINAMATH_CALUDE_saplings_distribution_l847_84756

theorem saplings_distribution (total : ℕ) (a b c d : ℕ) : 
  total = 2126 →
  a = 2 * b + 20 →
  a = 3 * c + 24 →
  a = 5 * d - 45 →
  a + b + c + d = total →
  a = 1050 := by
  sorry

end NUMINAMATH_CALUDE_saplings_distribution_l847_84756


namespace NUMINAMATH_CALUDE_tangent_curve_sum_l847_84767

/-- The curve y = -2x^2 + bx + c is tangent to the line y = x - 3 at the point (2, -1).
    This theorem proves that b + c = -2. -/
theorem tangent_curve_sum (b c : ℝ) : 
  (∀ x, -2*x^2 + b*x + c = x - 3 → x = 2) →  -- Tangent condition
  -2*2^2 + b*2 + c = -1 →                    -- Point (2, -1) lies on the curve
  2 - 3 = -1 →                               -- Point (2, -1) lies on the line
  (-4*2 + b = 1) →                           -- Derivative equality at x = 2
  b + c = -2 := by
sorry

end NUMINAMATH_CALUDE_tangent_curve_sum_l847_84767


namespace NUMINAMATH_CALUDE_least_sum_of_bases_l847_84789

/-- Represents a number in a given base -/
def NumberInBase (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun digit acc => acc * base + digit) 0

theorem least_sum_of_bases :
  ∃ (c d : Nat),
    c > 0 ∧ d > 0 ∧
    NumberInBase [5, 8] c = NumberInBase [8, 5] d ∧
    c + d = 15 ∧
    ∀ (c' d' : Nat), c' > 0 → d' > 0 → NumberInBase [5, 8] c' = NumberInBase [8, 5] d' → c' + d' ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_least_sum_of_bases_l847_84789


namespace NUMINAMATH_CALUDE_negation_of_forall_geq_two_l847_84710

theorem negation_of_forall_geq_two :
  ¬(∀ x : ℝ, x > 0 → x + 1/x ≥ 2) ↔ ∃ x : ℝ, x > 0 ∧ x + 1/x < 2 := by sorry

end NUMINAMATH_CALUDE_negation_of_forall_geq_two_l847_84710


namespace NUMINAMATH_CALUDE_fourth_row_sum_spiral_l847_84720

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the spiral filling of the grid -/
def spiralFill (n : ℕ) : List (Position × ℕ) := sorry

/-- The sum of the smallest and largest numbers in a given row -/
def sumMinMaxInRow (row : ℕ) (filled : List (Position × ℕ)) : ℕ := sorry

theorem fourth_row_sum_spiral (n : ℕ) (h : n = 21) :
  let filled := spiralFill n
  sumMinMaxInRow 4 filled = 742 := by sorry

end NUMINAMATH_CALUDE_fourth_row_sum_spiral_l847_84720


namespace NUMINAMATH_CALUDE_circle_symmetry_l847_84716

def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

def line_of_symmetry (x y : ℝ) : Prop := y = x

theorem circle_symmetry :
  ∀ (x y : ℝ), circle1 x y ↔ circle2 y x ∧ line_of_symmetry x y :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l847_84716


namespace NUMINAMATH_CALUDE_exists_x_y_for_3k_l847_84795

theorem exists_x_y_for_3k (k : ℕ+) : 
  ∃ (x y : ℤ), (¬ 3 ∣ x) ∧ (¬ 3 ∣ y) ∧ (x^2 + 2*y^2 = 3^(k.val)) := by
  sorry

end NUMINAMATH_CALUDE_exists_x_y_for_3k_l847_84795


namespace NUMINAMATH_CALUDE_fifth_power_sum_equality_l847_84788

theorem fifth_power_sum_equality : ∃! m : ℕ+, m.val ^ 5 = 144 ^ 5 + 91 ^ 5 + 56 ^ 5 + 19 ^ 5 := by
  sorry

end NUMINAMATH_CALUDE_fifth_power_sum_equality_l847_84788


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_theorem_l847_84793

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem about the perimeter of a non-shaded region in a composite figure -/
theorem non_shaded_perimeter_theorem 
  (large_rect : Rectangle)
  (small_rect : Rectangle)
  (shaded_area : ℝ)
  (h1 : large_rect.length = 12)
  (h2 : large_rect.width = 10)
  (h3 : small_rect.length = 4)
  (h4 : small_rect.width = 3)
  (h5 : shaded_area = 104) : 
  ∃ (non_shaded_rect : Rectangle), 
    abs (perimeter non_shaded_rect - 21.34) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_theorem_l847_84793


namespace NUMINAMATH_CALUDE_solve_system_l847_84787

theorem solve_system (x y : ℤ) 
  (h1 : x + y = 260) 
  (h2 : x - y = 200) : 
  y = 30 := by
sorry

end NUMINAMATH_CALUDE_solve_system_l847_84787


namespace NUMINAMATH_CALUDE_prob_sum_three_dice_l847_84701

/-- The probability of rolling a specific number on a fair, standard six-sided die -/
def single_die_prob : ℚ := 1 / 6

/-- The number of dice rolled -/
def num_dice : ℕ := 3

/-- The desired sum on the top faces -/
def desired_sum : ℕ := 3

/-- The probability of rolling the desired sum on all dice -/
def prob_desired_sum : ℚ := single_die_prob ^ num_dice

theorem prob_sum_three_dice (h : desired_sum = 3 ∧ num_dice = 3) :
  prob_desired_sum = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_prob_sum_three_dice_l847_84701
