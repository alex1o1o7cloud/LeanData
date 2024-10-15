import Mathlib

namespace NUMINAMATH_CALUDE_parallelogram_fourth_vertex_x_sum_l1408_140823

/-- The sum of all possible x coordinates of the 4th vertex of a parallelogram 
    with three vertices at (1,2), (3,8), and (4,1) is equal to 8. -/
theorem parallelogram_fourth_vertex_x_sum : 
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (3, 8)
  let C : ℝ × ℝ := (4, 1)
  let D₁ : ℝ × ℝ := B + C - A
  let D₂ : ℝ × ℝ := A + C - B
  let D₃ : ℝ × ℝ := A + B - C
  (D₁.1 + D₂.1 + D₃.1 : ℝ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_fourth_vertex_x_sum_l1408_140823


namespace NUMINAMATH_CALUDE_weight_of_bart_and_cindy_l1408_140814

/-- Given the weights of pairs of people, prove the weight of a specific pair -/
theorem weight_of_bart_and_cindy 
  (abby bart cindy damon : ℝ) 
  (h1 : abby + bart = 280) 
  (h2 : cindy + damon = 290) 
  (h3 : abby + damon = 300) : 
  bart + cindy = 270 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_bart_and_cindy_l1408_140814


namespace NUMINAMATH_CALUDE_unique_solution_for_complex_equation_l1408_140878

theorem unique_solution_for_complex_equation (x : ℝ) :
  x - 8 ≥ 0 →
  (7 / (Real.sqrt (x - 8) - 10) + 2 / (Real.sqrt (x - 8) - 4) +
   9 / (Real.sqrt (x - 8) + 4) + 14 / (Real.sqrt (x - 8) + 10) = 0) ↔
  x = 55 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_complex_equation_l1408_140878


namespace NUMINAMATH_CALUDE_seven_digit_numbers_existence_l1408_140853

theorem seven_digit_numbers_existence :
  ∃ (x y : ℕ),
    (10^6 ≤ x ∧ x < 10^7) ∧
    (10^6 ≤ y ∧ y < 10^7) ∧
    (3 * x * y = 10^7 * x + y) ∧
    (x = 166667 ∧ y = 333334) := by
  sorry

end NUMINAMATH_CALUDE_seven_digit_numbers_existence_l1408_140853


namespace NUMINAMATH_CALUDE_largest_valid_sequence_length_l1408_140885

def isPrimePower (n : ℕ) : Prop :=
  ∃ p k, Prime p ∧ n = p ^ k

def validSequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i, i ≤ n → isPrimePower (a i)) ∧
  (∀ i, 3 ≤ i ∧ i ≤ n → a i = a (i - 1) + a (i - 2))

theorem largest_valid_sequence_length :
  (∃ a : ℕ → ℕ, validSequence a 7) ∧
  (∀ n : ℕ, n > 7 → ¬∃ a : ℕ → ℕ, validSequence a n) :=
sorry

end NUMINAMATH_CALUDE_largest_valid_sequence_length_l1408_140885


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1408_140860

/-- Calculates compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Theorem: Compound interest calculation for given problem -/
theorem compound_interest_problem :
  let principal := 1200
  let rate := 0.20
  let time := 4
  abs (compound_interest principal rate time - 1288.32) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1408_140860


namespace NUMINAMATH_CALUDE_a_13_value_l1408_140856

/-- An arithmetic sequence with specific terms -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_13_value (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a) 
  (h_a5 : a 5 = 6)
  (h_a8 : a 8 = 15) : 
  a 13 = 30 := by
sorry

end NUMINAMATH_CALUDE_a_13_value_l1408_140856


namespace NUMINAMATH_CALUDE_find_divisor_l1408_140829

theorem find_divisor (dividend quotient remainder : ℕ) : 
  dividend = 12401 → quotient = 76 → remainder = 13 →
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 163 := by
  sorry

end NUMINAMATH_CALUDE_find_divisor_l1408_140829


namespace NUMINAMATH_CALUDE_range_of_a_l1408_140871

-- Define the sets S and T
def S : Set ℝ := {x | |x - 1| + |x + 2| > 5}
def T (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}

-- State the theorem
theorem range_of_a (a : ℝ) : S ∪ T a = Set.univ → -2 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1408_140871


namespace NUMINAMATH_CALUDE_doll_difference_proof_l1408_140896

-- Define the number of dolls for each person
def geraldine_dolls : ℝ := 2186.0
def jazmin_dolls : ℝ := 1209.0

-- Define the difference in dolls
def doll_difference : ℝ := geraldine_dolls - jazmin_dolls

-- Theorem statement
theorem doll_difference_proof : doll_difference = 977.0 := by
  sorry

end NUMINAMATH_CALUDE_doll_difference_proof_l1408_140896


namespace NUMINAMATH_CALUDE_curve_equation_proof_l1408_140866

theorem curve_equation_proof :
  let x : ℝ → ℝ := λ t => 3 * Real.cos t - 2 * Real.sin t
  let y : ℝ → ℝ := λ t => 3 * Real.sin t
  let a : ℝ := 1 / 9
  let b : ℝ := -4 / 27
  let c : ℝ := 5 / 81
  let d : ℝ := 0
  let e : ℝ := 1 / 3
  ∀ t : ℝ, a * (x t)^2 + b * (x t) * (y t) + c * (y t)^2 + d * (x t) + e * (y t) = 1 :=
by sorry

end NUMINAMATH_CALUDE_curve_equation_proof_l1408_140866


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1408_140826

/-- The quadratic equation (k-1)x^2 + 2x - 2 = 0 has two distinct real roots if and only if k > 1/2 and k ≠ 1 -/
theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k - 1) * x^2 + 2 * x - 2 = 0 ∧ (k - 1) * y^2 + 2 * y - 2 = 0) ↔ 
  (k > 1/2 ∧ k ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1408_140826


namespace NUMINAMATH_CALUDE_smallest_sum_square_config_l1408_140842

/-- A configuration of four positive integers on a square's vertices. -/
structure SquareConfig where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+

/-- Predicate to check if one number is a multiple of another. -/
def isMultiple (x y : ℕ+) : Prop := ∃ k : ℕ+, x = k * y

/-- Predicate to check if the configuration satisfies the edge multiple condition. -/
def satisfiesEdgeCondition (config : SquareConfig) : Prop :=
  (isMultiple config.a config.b ∨ isMultiple config.b config.a) ∧
  (isMultiple config.b config.c ∨ isMultiple config.c config.b) ∧
  (isMultiple config.c config.d ∨ isMultiple config.d config.c) ∧
  (isMultiple config.d config.a ∨ isMultiple config.a config.d)

/-- Predicate to check if the configuration satisfies the diagonal non-multiple condition. -/
def satisfiesDiagonalCondition (config : SquareConfig) : Prop :=
  ¬(isMultiple config.a config.c ∨ isMultiple config.c config.a) ∧
  ¬(isMultiple config.b config.d ∨ isMultiple config.d config.b)

/-- Theorem stating the smallest possible sum of the four integers. -/
theorem smallest_sum_square_config :
  ∀ config : SquareConfig,
    satisfiesEdgeCondition config →
    satisfiesDiagonalCondition config →
    (config.a + config.b + config.c + config.d : ℕ) ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_square_config_l1408_140842


namespace NUMINAMATH_CALUDE_area_of_triangle_AGE_l1408_140847

/-- Square ABCD with side length 5 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 5) ∧ B = (0, 0) ∧ C = (5, 0) ∧ D = (5, 5))

/-- Point E on side BC -/
def E : ℝ × ℝ := (2, 0)

/-- Point G on diagonal BD -/
def G : ℝ × ℝ := sorry

/-- Circumscribed circle of triangle ABE -/
def circle_ABE (sq : Square) : Set (ℝ × ℝ) := sorry

/-- Area of a triangle given three points -/
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

theorem area_of_triangle_AGE (sq : Square) :
  G ∈ circle_ABE sq →
  G.1 + G.2 = 5 →
  triangle_area sq.A G E = 54.5 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_AGE_l1408_140847


namespace NUMINAMATH_CALUDE_distribute_three_items_five_people_l1408_140865

/-- The number of ways to distribute distinct items among distinct people -/
def distribute_items (num_items : ℕ) (num_people : ℕ) : ℕ :=
  num_people ^ num_items

/-- Theorem: Distributing 3 distinct items among 5 distinct people results in 125 ways -/
theorem distribute_three_items_five_people : 
  distribute_items 3 5 = 125 := by
  sorry

end NUMINAMATH_CALUDE_distribute_three_items_five_people_l1408_140865


namespace NUMINAMATH_CALUDE_trapezoid_diagonal_inequality_l1408_140820

/-- A trapezoid with non-parallel sides b and d, and diagonals e and f -/
structure Trapezoid where
  b : ℝ
  d : ℝ
  e : ℝ
  f : ℝ
  b_pos : 0 < b
  d_pos : 0 < d
  e_pos : 0 < e
  f_pos : 0 < f

/-- The inequality |e - f| > |b - d| holds for a trapezoid -/
theorem trapezoid_diagonal_inequality (t : Trapezoid) : |t.e - t.f| > |t.b - t.d| := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_diagonal_inequality_l1408_140820


namespace NUMINAMATH_CALUDE_work_completion_time_l1408_140840

/-- The number of days it takes B to complete the work alone -/
def b_days : ℝ := 20

/-- The fraction of work left after A and B work together for 3 days -/
def work_left : ℝ := 0.65

/-- The number of days A and B work together -/
def days_together : ℝ := 3

/-- The number of days it takes A to complete the work alone -/
def a_days : ℝ := 15

theorem work_completion_time :
  ∃ (x : ℝ), x > 0 ∧ 
  days_together * (1 / x + 1 / b_days) = 1 - work_left ∧
  x = a_days := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1408_140840


namespace NUMINAMATH_CALUDE_sum_10_terms_l1408_140895

/-- An arithmetic sequence with a₂ = 3 and a₉ = 17 -/
def arithmetic_seq (n : ℕ) : ℝ :=
  sorry

/-- The sum of the first n terms of the arithmetic sequence -/
def S (n : ℕ) : ℝ :=
  sorry

/-- Theorem: The sum of the first 10 terms of the arithmetic sequence is 100 -/
theorem sum_10_terms : S 10 = 100 :=
  sorry

end NUMINAMATH_CALUDE_sum_10_terms_l1408_140895


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_specific_roots_l1408_140825

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ 
  a * x₂^2 + b * x₂ + c = 0 →
  x₁^2 + x₂^2 = (b^2 / a^2) + 2 * (c / a) :=
by sorry

theorem sum_of_squares_of_specific_roots :
  let a : ℚ := 5
  let b : ℚ := -3
  let c : ℚ := -11
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x₁^2 + b * x₁ + c = 0 ∧ 
  a * x₂^2 + b * x₂ + c = 0 →
  x₁^2 + x₂^2 = 119 / 25 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_sum_of_squares_of_specific_roots_l1408_140825


namespace NUMINAMATH_CALUDE_ronald_banana_count_l1408_140897

/-- The number of times Ronald went to the store last month -/
def store_visits : ℕ := 2

/-- The number of bananas Ronald buys each time he goes to the store -/
def bananas_per_visit : ℕ := 10

/-- The total number of bananas Ronald bought last month -/
def total_bananas : ℕ := store_visits * bananas_per_visit

theorem ronald_banana_count : total_bananas = 20 := by
  sorry

end NUMINAMATH_CALUDE_ronald_banana_count_l1408_140897


namespace NUMINAMATH_CALUDE_binomial_square_constant_l1408_140811

theorem binomial_square_constant (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, 16 * x^2 + 40 * x + a = (4 * x + b)^2) → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_constant_l1408_140811


namespace NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1408_140821

def polynomial (x : ℝ) : ℝ := 6 * (x^5 + 2*x^3 + x + 3)

theorem sum_of_squared_coefficients : 
  (6^2 : ℝ) + (12^2 : ℝ) + (6^2 : ℝ) + (18^2 : ℝ) = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_coefficients_l1408_140821


namespace NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1408_140874

/-- Given a cylinder with volume 72π cm³ and a cone with the same height
    as the cylinder and half its radius, prove that the volume of the cone is 6π cm³. -/
theorem cone_volume_from_cylinder (r h : ℝ) : 
  (π * r^2 * h = 72 * π) →
  (1/3 * π * (r/2)^2 * h = 6 * π) :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_from_cylinder_l1408_140874


namespace NUMINAMATH_CALUDE_opposite_of_negative_negative_five_l1408_140830

theorem opposite_of_negative_negative_five :
  -(-(5 : ℤ)) = -5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_negative_five_l1408_140830


namespace NUMINAMATH_CALUDE_souvenir_sales_theorem_l1408_140868

/-- Represents the souvenir sales scenario -/
structure SouvenirSales where
  purchase_price : ℝ
  base_selling_price : ℝ
  base_daily_sales : ℝ
  price_sales_ratio : ℝ

/-- Calculates the daily sales quantity for a given selling price -/
def daily_sales (s : SouvenirSales) (selling_price : ℝ) : ℝ :=
  s.base_daily_sales - s.price_sales_ratio * (selling_price - s.base_selling_price)

/-- Calculates the daily profit for a given selling price -/
def daily_profit (s : SouvenirSales) (selling_price : ℝ) : ℝ :=
  (selling_price - s.purchase_price) * (daily_sales s selling_price)

/-- The main theorem about the souvenir sales scenario -/
theorem souvenir_sales_theorem (s : SouvenirSales) 
  (h1 : s.purchase_price = 40)
  (h2 : s.base_selling_price = 50)
  (h3 : s.base_daily_sales = 200)
  (h4 : s.price_sales_ratio = 10) :
  (daily_sales s 52 = 180) ∧ 
  (∃ x : ℝ, ∀ y : ℝ, daily_profit s x ≥ daily_profit s y) ∧
  (daily_profit s 55 = 2250) := by
  sorry

#check souvenir_sales_theorem

end NUMINAMATH_CALUDE_souvenir_sales_theorem_l1408_140868


namespace NUMINAMATH_CALUDE_smallest_b_for_scaled_property_l1408_140845

/-- A function with period 30 -/
def IsPeriodic30 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 30) = g x

/-- The property we want to prove for the scaled function -/
def HasScaledProperty (g : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, g ((x - b) / 10) = g (x / 10)

/-- The main theorem -/
theorem smallest_b_for_scaled_property (g : ℝ → ℝ) (h : IsPeriodic30 g) :
    (∃ b > 0, HasScaledProperty g b) →
    (∃ b > 0, HasScaledProperty g b ∧ ∀ b' > 0, HasScaledProperty g b' → b ≤ b') →
    (∃ b > 0, HasScaledProperty g b ∧ b = 300) :=
  sorry

end NUMINAMATH_CALUDE_smallest_b_for_scaled_property_l1408_140845


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l1408_140859

theorem algebraic_expression_value (a x : ℝ) : 
  (3 * a - x = x + 2) → (x = 2) → (a^2 - 2*a + 1 = 1) := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l1408_140859


namespace NUMINAMATH_CALUDE_equation_solution_l1408_140816

theorem equation_solution : 
  ∃! x : ℝ, 12 * (x - 3) - 1 = 2 * x + 3 :=
by
  use 4
  constructor
  · -- Prove that x = 4 satisfies the equation
    sorry
  · -- Prove uniqueness
    sorry

end NUMINAMATH_CALUDE_equation_solution_l1408_140816


namespace NUMINAMATH_CALUDE_josh_and_anna_marriage_problem_l1408_140832

/-- Josh and Anna's marriage problem -/
theorem josh_and_anna_marriage_problem 
  (josh_marriage_age : ℕ) 
  (marriage_duration : ℕ) 
  (combined_age_factor : ℕ) 
  (h1 : josh_marriage_age = 22)
  (h2 : marriage_duration = 30)
  (h3 : combined_age_factor = 5)
  (h4 : josh_marriage_age + marriage_duration + (josh_marriage_age + marriage_duration + anna_marriage_age) = combined_age_factor * josh_marriage_age) :
  anna_marriage_age = 28 :=
by sorry

end NUMINAMATH_CALUDE_josh_and_anna_marriage_problem_l1408_140832


namespace NUMINAMATH_CALUDE_simplify_fraction_l1408_140843

theorem simplify_fraction : (4^5 + 4^3) / (4^4 - 4^2 - 4) = 272 / 59 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1408_140843


namespace NUMINAMATH_CALUDE_three_numbers_sum_l1408_140800

theorem three_numbers_sum (a b c : ℝ) : 
  a ≤ b ∧ b ≤ c ∧  -- a, b, c are in ascending order
  b = 10 ∧  -- median is 10
  (a + b + c) / 3 = a + 20 ∧  -- mean is 20 more than least
  (a + b + c) / 3 = c - 10  -- mean is 10 less than greatest
  → a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_three_numbers_sum_l1408_140800


namespace NUMINAMATH_CALUDE_conditional_probability_rain_wind_l1408_140881

theorem conditional_probability_rain_wind (P_rain P_wind_and_rain : ℚ) 
  (h1 : P_rain = 4 / 15)
  (h2 : P_wind_and_rain = 1 / 10) :
  P_wind_and_rain / P_rain = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_rain_wind_l1408_140881


namespace NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l1408_140806

-- Part 1: System of Equations
theorem solve_system_equations :
  ∃! (x y : ℝ), 3 * x + 2 * y = 12 ∧ 2 * x - y = 1 ∧ x = 2 ∧ y = 3 := by sorry

-- Part 2: System of Inequalities
theorem solve_system_inequalities :
  ∀ x : ℝ, (x - 1 < 2 * x ∧ 2 * (x - 3) ≤ 3 - x) ↔ (-1 < x ∧ x ≤ 3) := by sorry

end NUMINAMATH_CALUDE_solve_system_equations_solve_system_inequalities_l1408_140806


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1408_140822

/-- Proves that for a hyperbola with equation x²/a² - y²/b² = 1, where a > b, 
    if the angle between the asymptotes is 45°, then a/b = 1/(-1 + √2). -/
theorem hyperbola_asymptote_angle (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (Real.pi / 4 = Real.arctan ((2 * b / a) / (1 - (b / a)^2))) →
  a / b = 1 / (-1 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_angle_l1408_140822


namespace NUMINAMATH_CALUDE_min_a_value_for_common_points_l1408_140828

/-- Given two curves C₁ and C₂, where C₁ is y = ax² (a > 0) and C₂ is y = eˣ, 
    if they have common points in (0, +∞), then the minimum value of a is e²/4 -/
theorem min_a_value_for_common_points (a : ℝ) (h1 : a > 0) :
  (∃ x : ℝ, x > 0 ∧ a * x^2 = Real.exp x) → a ≥ Real.exp 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_a_value_for_common_points_l1408_140828


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1408_140812

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : a*b + b*c + c*a = 72) : 
  a + b + c = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1408_140812


namespace NUMINAMATH_CALUDE_sand_heap_radius_l1408_140861

/-- The radius of a conical heap of sand formed from a cylindrical bucket -/
theorem sand_heap_radius (h_cylinder h_cone r_cylinder : ℝ) 
  (h_cylinder_pos : h_cylinder > 0)
  (h_cone_pos : h_cone > 0)
  (r_cylinder_pos : r_cylinder > 0)
  (h_cylinder_val : h_cylinder = 36)
  (h_cone_val : h_cone = 12)
  (r_cylinder_val : r_cylinder = 21) :
  ∃ r_cone : ℝ, r_cone > 0 ∧ r_cone^2 = 3 * r_cylinder^2 :=
by sorry

end NUMINAMATH_CALUDE_sand_heap_radius_l1408_140861


namespace NUMINAMATH_CALUDE_sum_six_consecutive_integers_l1408_140887

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_six_consecutive_integers_l1408_140887


namespace NUMINAMATH_CALUDE_system_solution_l1408_140883

theorem system_solution (x y : ℝ) : 
  x > 0 → y > 0 → 
  Real.log x / Real.log 4 + Real.log y / Real.log 4 = 1 + Real.log 9 / Real.log 4 →
  x + y = 20 →
  ((x = 2 ∧ y = 18) ∨ (x = 18 ∧ y = 2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1408_140883


namespace NUMINAMATH_CALUDE_flea_initial_position_l1408_140872

def electronic_flea (K : ℕ → ℤ) : Prop :=
  K 100 = 20 ∧ ∀ n : ℕ, K (n + 1) = K n + (-1)^n * (n + 1)

theorem flea_initial_position (K : ℕ → ℤ) (h : electronic_flea K) : K 0 = -30 := by
  sorry

end NUMINAMATH_CALUDE_flea_initial_position_l1408_140872


namespace NUMINAMATH_CALUDE_library_books_l1408_140854

theorem library_books (shelves : ℝ) (books_per_shelf : ℕ) : 
  shelves = 14240.0 → books_per_shelf = 8 → shelves * (books_per_shelf : ℝ) = 113920 := by
  sorry

end NUMINAMATH_CALUDE_library_books_l1408_140854


namespace NUMINAMATH_CALUDE_largest_number_with_digit_sum_20_l1408_140852

def digit_sum (n : Nat) : Nat :=
  Nat.digits 10 n |>.sum

def all_digits_different (n : Nat) : Prop :=
  (Nat.digits 10 n).Nodup

def no_zero_digit (n : Nat) : Prop :=
  0 ∉ Nat.digits 10 n

theorem largest_number_with_digit_sum_20 :
  ∀ n : Nat,
    (digit_sum n = 20 ∧
     all_digits_different n ∧
     no_zero_digit n) →
    n ≤ 9821 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_sum_20_l1408_140852


namespace NUMINAMATH_CALUDE_both_reunions_count_l1408_140876

/-- The number of people attending both the Oates and Yellow reunions -/
def both_reunions (total_guests oates_guests yellow_guests : ℕ) : ℕ :=
  oates_guests + yellow_guests - total_guests

theorem both_reunions_count :
  both_reunions 100 42 65 = 7 := by
  sorry

end NUMINAMATH_CALUDE_both_reunions_count_l1408_140876


namespace NUMINAMATH_CALUDE_vegetarian_eaters_l1408_140894

theorem vegetarian_eaters (only_veg : ℕ) (only_non_veg : ℕ) (both : ℕ) :
  only_veg = 13 →
  only_non_veg = 8 →
  both = 6 →
  only_veg + both = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_vegetarian_eaters_l1408_140894


namespace NUMINAMATH_CALUDE_exactly_three_props_true_l1408_140850

/-- Property P for a sequence -/
def has_property_P (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i → i < j → j ≤ n → (∃ k ≤ n, a k = a j + a i) ∨ (∃ k ≤ n, a k = a j - a i)

/-- The sequence is strictly increasing and starts with a non-negative number -/
def is_valid_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∀ i j, 1 ≤ i → i < j → j ≤ n → a i < a j) ∧ 0 ≤ a 1

/-- Proposition 1: The sequence 0, 2, 4, 6 has property P -/
def prop_1 : Prop :=
  let a : ℕ → ℕ := fun i => 2 * (i - 1)
  has_property_P a 4

/-- Proposition 2: If sequence A has property P, then a₁ = 0 -/
def prop_2 : Prop :=
  ∀ a : ℕ → ℕ, ∀ n ≥ 3, is_valid_sequence a n → has_property_P a n → a 1 = 0

/-- Proposition 3: If sequence A has property P and a₁ ≠ 0, then aₙ - aₙ₋ₖ = aₖ for k = 1, 2, ..., n-1 -/
def prop_3 : Prop :=
  ∀ a : ℕ → ℕ, ∀ n ≥ 3, is_valid_sequence a n → has_property_P a n → a 1 ≠ 0 →
    ∀ k, 1 ≤ k → k < n → a n - a (n - k) = a k

/-- Proposition 4: If the sequence a₁, a₂, a₃ (0 ≤ a₁ < a₂ < a₃) has property P, then a₃ = a₁ + a₂ -/
def prop_4 : Prop :=
  ∀ a : ℕ → ℕ, is_valid_sequence a 3 → has_property_P a 3 → a 3 = a 1 + a 2

theorem exactly_three_props_true : (prop_1 ∧ ¬prop_2 ∧ prop_3 ∧ prop_4) := by sorry

end NUMINAMATH_CALUDE_exactly_three_props_true_l1408_140850


namespace NUMINAMATH_CALUDE_sin_585_degrees_l1408_140834

theorem sin_585_degrees : Real.sin (585 * π / 180) = - Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_585_degrees_l1408_140834


namespace NUMINAMATH_CALUDE_every_nonzero_nat_is_product_of_primes_l1408_140801

theorem every_nonzero_nat_is_product_of_primes :
  ∀ n : ℕ, n > 0 → ∃ (primes : List ℕ), (∀ p ∈ primes, Nat.Prime p) ∧ n = primes.prod := by
  sorry

end NUMINAMATH_CALUDE_every_nonzero_nat_is_product_of_primes_l1408_140801


namespace NUMINAMATH_CALUDE_soccer_team_starters_l1408_140835

theorem soccer_team_starters (n : ℕ) (k : ℕ) (q : ℕ) (m : ℕ) 
  (h1 : n = 16) 
  (h2 : k = 7) 
  (h3 : q = 4) 
  (h4 : m = 1) :
  (q.choose m) * ((n - q).choose (k - m)) = 3696 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l1408_140835


namespace NUMINAMATH_CALUDE_first_triangle_isosceles_l1408_140873

theorem first_triangle_isosceles (α β γ : Real) (θ₁ θ₂ : Real) : 
  α + β + γ = π → 
  α + β = θ₁ → 
  α + γ = θ₂ → 
  θ₁ + θ₂ < π →
  β = γ := by
sorry

end NUMINAMATH_CALUDE_first_triangle_isosceles_l1408_140873


namespace NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l1408_140889

theorem polynomial_multiplication_simplification (x : ℝ) :
  (3*x - 2) * (5*x^12 - 3*x^11 + 4*x^9 - 2*x^8) =
  15*x^13 - 19*x^12 + 6*x^11 + 12*x^10 - 14*x^9 - 4*x^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_multiplication_simplification_l1408_140889


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l1408_140862

/-- Given a point A with coordinates (-1, 2) in the plane rectangular coordinate system xOy,
    prove that its coordinates with respect to the origin are (-1, 2). -/
theorem coordinates_wrt_origin (A : ℝ × ℝ) (h : A = (-1, 2)) :
  A = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l1408_140862


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1408_140877

theorem fraction_subtraction_simplification : 
  (9 : ℚ) / 19 - 5 / 57 - 2 / 38 = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l1408_140877


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1408_140827

theorem smaller_number_problem (a b : ℤ) : 
  a + b = 18 → a - b = 24 → min a b = -3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1408_140827


namespace NUMINAMATH_CALUDE_sum_of_squares_constant_l1408_140839

/-- A triangle with side lengths a, b, c and median length m from vertex A to the midpoint of side BC. -/
structure Triangle :=
  (a b c m : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (positive_m : 0 < m)
  (triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b)

/-- The sum of squares of two sides in a triangle given the length of the third side and the median to its midpoint. -/
def sumOfSquares (t : Triangle) : ℝ := t.b^2 + t.c^2

/-- The theorem stating that for a triangle with side length 10 and median length 7,
    the difference between the maximum and minimum possible values of the sum of squares
    of the other two sides is 0. -/
theorem sum_of_squares_constant (t : Triangle) 
  (side_length : t.a = 10)
  (median_length : t.m = 7) :
  ∀ (t' : Triangle), 
    t'.a = t.a → 
    t'.m = t.m → 
    sumOfSquares t = sumOfSquares t' :=
sorry

end NUMINAMATH_CALUDE_sum_of_squares_constant_l1408_140839


namespace NUMINAMATH_CALUDE_system_solution_l1408_140858

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x - 1 / ((x - y)^2) + y = -10
def equation2 (x y : ℝ) : Prop := x * y = 20

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) :=
  {(-4, -5), (-5, -4), (-2.7972, -7.15), (-7.15, -2.7972), (4.5884, 4.3588), (4.3588, 4.5884)}

-- Theorem statement
theorem system_solution :
  ∀ (x y : ℝ), (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_system_solution_l1408_140858


namespace NUMINAMATH_CALUDE_solution_is_two_l1408_140898

-- Define the base 10 logarithm
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  log10 (x^2 - 3) = log10 (3*x - 5) ∧ x^2 - 3 > 0 ∧ 3*x - 5 > 0

-- Theorem stating that 2 is the solution to the equation
theorem solution_is_two : equation 2 := by sorry

end NUMINAMATH_CALUDE_solution_is_two_l1408_140898


namespace NUMINAMATH_CALUDE_inequality_proof_l1408_140831

theorem inequality_proof (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_of_squares : a^2 + b^2 + c^2 = 1) :
  a*b/c + b*c/a + c*a/b ≥ Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1408_140831


namespace NUMINAMATH_CALUDE_root_squares_sum_l1408_140891

theorem root_squares_sum (a b : ℝ) (a_ne_b : a ≠ b) : 
  ∃ (s t : ℝ), (a * s^2 + b * s + b = 0) ∧ 
                (a * t^2 + a * t + b = 0) ∧ 
                (s * t = 1) → 
                (s^2 + t^2 = 3) := by
  sorry

end NUMINAMATH_CALUDE_root_squares_sum_l1408_140891


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l1408_140869

theorem sum_of_reciprocal_roots (a b : ℝ) : 
  (6 * a^2 + 5 * a + 7 = 0) →
  (6 * b^2 + 5 * b + 7 = 0) →
  a ≠ b →
  a ≠ 0 →
  b ≠ 0 →
  (1 / a) + (1 / b) = -5 / 7 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_roots_l1408_140869


namespace NUMINAMATH_CALUDE_product_of_numbers_l1408_140893

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 218) : x * y = 13 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1408_140893


namespace NUMINAMATH_CALUDE_calculate_birth_rate_l1408_140818

/-- Given a death rate and population increase rate, calculate the birth rate. -/
theorem calculate_birth_rate (death_rate : ℝ) (population_increase_rate : ℝ) : 
  death_rate = 11 → population_increase_rate = 2.1 → 
  ∃ (birth_rate : ℝ), birth_rate = 32 ∧ birth_rate - death_rate = population_increase_rate / 100 * 1000 := by
  sorry

#check calculate_birth_rate

end NUMINAMATH_CALUDE_calculate_birth_rate_l1408_140818


namespace NUMINAMATH_CALUDE_banquet_food_consumption_l1408_140833

/-- Represents the football banquet scenario -/
structure FootballBanquet where
  /-- The maximum amount of food (in pounds) consumed by any individual guest -/
  max_food_per_guest : ℝ
  /-- The minimum number of guests that attended the banquet -/
  min_guests : ℕ
  /-- The total amount of food (in pounds) consumed at the banquet -/
  total_food_consumed : ℝ

/-- Theorem stating that the total food consumed at the banquet is at least 326 pounds -/
theorem banquet_food_consumption (banquet : FootballBanquet)
  (h1 : banquet.max_food_per_guest ≤ 2)
  (h2 : banquet.min_guests ≥ 163)
  : banquet.total_food_consumed ≥ 326 := by
  sorry

#check banquet_food_consumption

end NUMINAMATH_CALUDE_banquet_food_consumption_l1408_140833


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1408_140851

theorem sufficient_not_necessary_condition :
  (∀ a b : ℝ, a > 2 ∧ b > 2 → a + b > 4 ∧ a * b > 4) ∧
  (∃ a b : ℝ, a + b > 4 ∧ a * b > 4 ∧ ¬(a > 2 ∧ b > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1408_140851


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_l1408_140841

/-- A triangle with integer side lengths, where one side is twice another, and the third side is 10 -/
structure SpecialTriangle where
  x : ℕ
  side1 : ℕ := x
  side2 : ℕ := 2 * x
  side3 : ℕ := 10

/-- The perimeter of a SpecialTriangle -/
def perimeter (t : SpecialTriangle) : ℕ := t.side1 + t.side2 + t.side3

/-- The triangle inequality for SpecialTriangle -/
def is_valid (t : SpecialTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The theorem stating the maximum perimeter of a valid SpecialTriangle -/
theorem max_perimeter_special_triangle :
  ∃ (t : SpecialTriangle), is_valid t ∧
  ∀ (t' : SpecialTriangle), is_valid t' → perimeter t' ≤ perimeter t ∧
  perimeter t = 37 := by sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_l1408_140841


namespace NUMINAMATH_CALUDE_three_fraction_equality_l1408_140844

theorem three_fraction_equality (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) 
  (hdiff : x ≠ y ∧ y ≠ z ∧ x ≠ z) 
  (heq : (y + 1) / (x - z + 1) = (x + y + 2) / (z + 2) ∧ 
         (x + y + 2) / (z + 2) = (x + 1) / (y + 1)) : 
  (x + 1) / (y + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_three_fraction_equality_l1408_140844


namespace NUMINAMATH_CALUDE_camel_cost_l1408_140899

/-- The cost of animals in rupees -/
structure AnimalCosts where
  camel : ℝ
  horse : ℝ
  ox : ℝ
  elephant : ℝ

/-- The conditions given in the problem -/
def problem_conditions (costs : AnimalCosts) : Prop :=
  10 * costs.camel = 24 * costs.horse ∧
  16 * costs.horse = 4 * costs.ox ∧
  6 * costs.ox = 4 * costs.elephant ∧
  10 * costs.elephant = 140000

/-- The theorem stating that under the given conditions, a camel costs 5600 rupees -/
theorem camel_cost (costs : AnimalCosts) : 
  problem_conditions costs → costs.camel = 5600 := by
  sorry

end NUMINAMATH_CALUDE_camel_cost_l1408_140899


namespace NUMINAMATH_CALUDE_sweetsies_leftover_l1408_140880

theorem sweetsies_leftover (m : ℕ) : 
  (∃ k : ℕ, m = 8 * k + 5) →  -- One bag leaves 5 when divided by 8
  (∃ l : ℕ, 4 * m = 8 * l + 4) -- Four bags leave 4 when divided by 8
  := by sorry

end NUMINAMATH_CALUDE_sweetsies_leftover_l1408_140880


namespace NUMINAMATH_CALUDE_basketball_spectators_l1408_140808

theorem basketball_spectators (total : Nat) (men : Nat) (women : Nat) (children : Nat) :
  total = 10000 →
  men = 7000 →
  total = men + women + children →
  children = 5 * women →
  children = 2500 := by
sorry

end NUMINAMATH_CALUDE_basketball_spectators_l1408_140808


namespace NUMINAMATH_CALUDE_robot_tracing_time_l1408_140810

/-- Represents a rectangular grid with width and height -/
structure Grid where
  width : ℕ
  height : ℕ

/-- Calculates the total length of lines in a grid -/
def totalLength (g : Grid) : ℕ :=
  (g.width + 1) * g.height + (g.height + 1) * g.width

/-- Represents the robot's tracing speed in grid units per minute -/
def robotSpeed (g : Grid) (time : ℚ) : ℚ :=
  (totalLength g : ℚ) / time

theorem robot_tracing_time 
  (g1 g2 : Grid) 
  (t1 : ℚ) 
  (hg1 : g1 = ⟨3, 7⟩) 
  (hg2 : g2 = ⟨5, 5⟩) 
  (ht1 : t1 = 26) :
  robotSpeed g1 t1 * (totalLength g2 : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_robot_tracing_time_l1408_140810


namespace NUMINAMATH_CALUDE_sides_when_k_is_two_k_values_l1408_140892

/-- Represents a regular pyramid -/
structure RegularPyramid where
  n : ℕ  -- number of sides of the base
  α : ℝ  -- dihedral angle at the base
  β : ℝ  -- angle formed by lateral edges with the base plane
  k : ℝ  -- relationship constant between α and β
  h1 : α > 0
  h2 : β > 0
  h3 : k > 0
  h4 : n ≥ 3
  h5 : Real.tan α = k * Real.tan β
  h6 : k = 1 / Real.cos (π / n)

/-- The number of sides of the base is 3 when k = 2 -/
theorem sides_when_k_is_two (p : RegularPyramid) : p.k = 2 → p.n = 3 := by sorry

/-- The possible values of k are given by 1 / cos(π/n) where n ≥ 3 -/
theorem k_values (p : RegularPyramid) : 
  ∃ (n : ℕ), n ≥ 3 ∧ p.k = 1 / Real.cos (π / n) := by sorry

end NUMINAMATH_CALUDE_sides_when_k_is_two_k_values_l1408_140892


namespace NUMINAMATH_CALUDE_factor_condition_l1408_140805

theorem factor_condition (n : ℕ) (hn : n ≥ 2) 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (k : ℕ), n < 2 * k + 1 ∧ 2 * k + 1 < 3 * n ∧
    a = (-2 * Real.cos ((2 * k + 1 : ℝ) * π / (2 * n : ℝ))) ^ (2 * n / (2 * n - 1 : ℝ)) ∧
    b = (2 * Real.cos ((2 * k + 1 : ℝ) * π / (2 * n : ℝ))) ^ (2 / (2 * n - 1 : ℝ))) ↔
  (∀ x : ℂ, (x ^ 2 + a * x + b = 0) → (a * x ^ (2 * n) + (a * x + b) ^ (2 * n) = 0)) :=
by sorry

end NUMINAMATH_CALUDE_factor_condition_l1408_140805


namespace NUMINAMATH_CALUDE_no_zero_roots_l1408_140807

theorem no_zero_roots : 
  (∀ x : ℝ, 5 * x^2 - 3 = 47 → x ≠ 0) ∧ 
  (∀ x : ℝ, (3 * x + 2)^2 = (x + 2)^2 → x ≠ 0) ∧ 
  (∀ x : ℝ, (2 * x^2 - 6 : ℝ) = (2 * x - 2 : ℝ) → x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_zero_roots_l1408_140807


namespace NUMINAMATH_CALUDE_sum_of_squares_of_rates_l1408_140815

theorem sum_of_squares_of_rates : ∀ (c j s : ℕ),
  3 * c + 2 * j + 2 * s = 80 →
  3 * j + 2 * s + 4 * c = 104 →
  c^2 + j^2 + s^2 = 592 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_rates_l1408_140815


namespace NUMINAMATH_CALUDE_second_box_weight_l1408_140857

/-- The weight of the second box in a set of three boxes -/
def weight_of_second_box (weight_first weight_last total_weight : ℕ) : ℕ :=
  total_weight - weight_first - weight_last

/-- Theorem: The weight of the second box is 11 pounds -/
theorem second_box_weight :
  weight_of_second_box 2 5 18 = 11 := by
  sorry

end NUMINAMATH_CALUDE_second_box_weight_l1408_140857


namespace NUMINAMATH_CALUDE_divisibility_problem_l1408_140875

theorem divisibility_problem (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 45)
  (h2 : Nat.gcd q r = 75)
  (h3 : Nat.gcd r s = 90)
  (h4 : 150 < Nat.gcd s p ∧ Nat.gcd s p < 200) :
  10 ∣ p.val := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1408_140875


namespace NUMINAMATH_CALUDE_divisibility_problem_l1408_140849

theorem divisibility_problem :
  {n : ℤ | (n - 2) ∣ (n^2 + 3*n + 27)} = {1, 3, 39, -35} := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l1408_140849


namespace NUMINAMATH_CALUDE_jerry_average_increase_l1408_140817

theorem jerry_average_increase :
  let initial_average : ℝ := 85
  let fourth_test_score : ℝ := 97
  let total_tests : ℕ := 4
  let sum_first_three : ℝ := initial_average * 3
  let sum_all_four : ℝ := sum_first_three + fourth_test_score
  let new_average : ℝ := sum_all_four / total_tests
  new_average - initial_average = 3 := by sorry

end NUMINAMATH_CALUDE_jerry_average_increase_l1408_140817


namespace NUMINAMATH_CALUDE_four_valid_orders_l1408_140888

/-- Represents a runner in the relay team -/
inductive Runner : Type
| Jordan : Runner
| Friend1 : Runner  -- The fastest friend
| Friend2 : Runner
| Friend3 : Runner

/-- Represents a lap in the relay race -/
inductive Lap : Type
| First : Lap
| Second : Lap
| Third : Lap
| Fourth : Lap

/-- A valid running order for the relay team -/
def RunningOrder : Type := Lap → Runner

/-- Checks if a running order is valid according to the given conditions -/
def isValidOrder (order : RunningOrder) : Prop :=
  (order Lap.First = Runner.Friend1) ∧  -- Fastest friend starts
  ((order Lap.Third = Runner.Jordan) ∨ (order Lap.Fourth = Runner.Jordan)) ∧  -- Jordan runs 3rd or 4th
  (∀ l : Lap, ∃! r : Runner, order l = r)  -- Each lap has exactly one runner

/-- The main theorem: there are exactly 4 valid running orders -/
theorem four_valid_orders :
  ∃ (orders : Finset RunningOrder),
    (∀ o ∈ orders, isValidOrder o) ∧
    (∀ o : RunningOrder, isValidOrder o → o ∈ orders) ∧
    (Finset.card orders = 4) :=
sorry

end NUMINAMATH_CALUDE_four_valid_orders_l1408_140888


namespace NUMINAMATH_CALUDE_diagonal_contains_all_numbers_l1408_140867

theorem diagonal_contains_all_numbers (n : ℕ) (h_odd : Odd n) 
  (grid : Fin n → Fin n → Fin n)
  (h_row : ∀ i j k, i ≠ k → grid i j ≠ grid k j)
  (h_col : ∀ i j k, j ≠ k → grid i j ≠ grid i k)
  (h_sym : ∀ i j, grid i j = grid j i) :
  ∀ k : Fin n, ∃ i : Fin n, grid i i = k := by
sorry

end NUMINAMATH_CALUDE_diagonal_contains_all_numbers_l1408_140867


namespace NUMINAMATH_CALUDE_product_equality_l1408_140863

theorem product_equality : 375680169467 * 4565579427629 = 1715110767607750737263 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l1408_140863


namespace NUMINAMATH_CALUDE_orange_purchase_price_l1408_140848

/-- The price of oranges per 3 pounds -/
def price_per_3_pounds : ℝ := 3

/-- The weight of oranges purchased in pounds -/
def weight_purchased : ℝ := 18

/-- The discount rate applied for purchases over 15 pounds -/
def discount_rate : ℝ := 0.05

/-- The minimum weight for discount eligibility in pounds -/
def discount_threshold : ℝ := 15

/-- The final price paid by the customer for the oranges -/
def final_price : ℝ := 17.10

theorem orange_purchase_price :
  weight_purchased > discount_threshold →
  final_price = (weight_purchased / 3 * price_per_3_pounds) * (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_orange_purchase_price_l1408_140848


namespace NUMINAMATH_CALUDE_power_six_mod_fifty_l1408_140864

theorem power_six_mod_fifty : 6^2040 ≡ 26 [ZMOD 50] := by sorry

end NUMINAMATH_CALUDE_power_six_mod_fifty_l1408_140864


namespace NUMINAMATH_CALUDE_point_not_on_graph_l1408_140809

/-- A linear function passing through (1, 2) -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x

/-- The theorem stating that (1, -2) is not on the graph of the function -/
theorem point_not_on_graph (k : ℝ) (h1 : k ≠ 0) (h2 : f k 1 = 2) :
  f k 1 ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l1408_140809


namespace NUMINAMATH_CALUDE_base_10_number_l1408_140846

-- Define the properties of the number
def is_valid_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a < 8 ∧ b < 8 ∧ c < 8 ∧ d < 8 ∧
  (8 * a + b + c / 8 + d / 64 : ℚ) = (12 * b + b + b / 12 + a / 144 : ℚ)

-- State the theorem
theorem base_10_number (a b c d : ℕ) :
  is_valid_number a b c d → a * 100 + b * 10 + c = 321 :=
by sorry

end NUMINAMATH_CALUDE_base_10_number_l1408_140846


namespace NUMINAMATH_CALUDE_min_sum_squares_l1408_140838

theorem min_sum_squares (y₁ y₂ y₃ : ℝ) (h_pos : y₁ > 0 ∧ y₂ > 0 ∧ y₃ > 0) 
  (h_sum : 2 * y₁ + 3 * y₂ + 4 * y₃ = 120) : 
  y₁^2 + y₂^2 + y₃^2 ≥ 6100 / 9 ∧ 
  ∃ (y₁' y₂' y₃' : ℝ), y₁'^2 + y₂'^2 + y₃'^2 = 6100 / 9 ∧ 
    y₁' > 0 ∧ y₂' > 0 ∧ y₃' > 0 ∧ 2 * y₁' + 3 * y₂' + 4 * y₃' = 120 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1408_140838


namespace NUMINAMATH_CALUDE_sprinkler_water_usage_5_days_l1408_140813

/-- A sprinkler system for a desert garden -/
structure SprinklerSystem where
  morning_usage : ℕ  -- Water usage in the morning in liters
  evening_usage : ℕ  -- Water usage in the evening in liters

/-- Calculates the total water usage for a given number of days -/
def total_water_usage (s : SprinklerSystem) (days : ℕ) : ℕ :=
  (s.morning_usage + s.evening_usage) * days

/-- Theorem: The sprinkler system uses 50 liters of water in 5 days -/
theorem sprinkler_water_usage_5_days :
  ∃ (s : SprinklerSystem), s.morning_usage = 4 ∧ s.evening_usage = 6 ∧ total_water_usage s 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_sprinkler_water_usage_5_days_l1408_140813


namespace NUMINAMATH_CALUDE_carls_garden_area_l1408_140837

/-- Represents a rectangular garden with fence posts -/
structure Garden where
  total_posts : Nat
  post_spacing : Nat
  longer_side_posts : Nat
  shorter_side_posts : Nat

/-- Calculates the area of the garden given the specifications -/
def calculate_area (g : Garden) : Nat :=
  (g.shorter_side_posts - 1) * g.post_spacing * 
  (g.longer_side_posts - 1) * g.post_spacing

/-- Theorem stating the area of Carl's garden -/
theorem carls_garden_area : 
  ∀ g : Garden, 
    g.total_posts = 24 ∧ 
    g.post_spacing = 5 ∧ 
    g.longer_side_posts = 2 * g.shorter_side_posts ∧
    g.longer_side_posts + g.shorter_side_posts = g.total_posts + 2 →
    calculate_area g = 900 := by
  sorry

end NUMINAMATH_CALUDE_carls_garden_area_l1408_140837


namespace NUMINAMATH_CALUDE_sin_theta_value_l1408_140882

theorem sin_theta_value (θ : Real) 
  (h1 : 10 * Real.tan θ = 5 * Real.cos θ) 
  (h2 : 0 < θ) 
  (h3 : θ < Real.pi / 2) : 
  Real.sin θ = 1 := by sorry

end NUMINAMATH_CALUDE_sin_theta_value_l1408_140882


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l1408_140855

/-- Represents the side lengths of an isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of the base or the equal side
  h : a > 0 ∧ b > 0  -- Lengths are positive

/-- Checks if the given side lengths form a valid triangle -/
def is_valid_triangle (t : IsoscelesTriangle) : Prop :=
  t.a + t.b > t.a ∧ t.a + t.a > t.b

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ := 2 * t.a + t.b

theorem isosceles_triangle_side_lengths :
  ∀ t : IsoscelesTriangle,
    is_valid_triangle t →
    perimeter t = 17 →
    (t.a = 4 ∨ t.b = 4) →
    ((t.a = 6 ∧ t.b = 5) ∨ (t.a = 5 ∧ t.b = 7)) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l1408_140855


namespace NUMINAMATH_CALUDE_polynomial_remainder_l1408_140803

def g (a b x : ℚ) : ℚ := a * x^3 - 8 * x^2 + b * x - 7

theorem polynomial_remainder (a b : ℚ) :
  (g a b 2 = 1) ∧ (g a b (-3) = -89) → a = -10/3 ∧ b = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l1408_140803


namespace NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_for_f_geq_x_minus_m_l1408_140804

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + 2 * |x + 1|

-- Theorem for part I
theorem solution_set_f_leq_5 :
  {x : ℝ | f x ≤ 5} = {x : ℝ | -2 ≤ x ∧ x ≤ 4/3} :=
sorry

-- Theorem for part II
theorem range_of_m_for_f_geq_x_minus_m :
  {m : ℝ | ∀ x, f x ≥ x - m} = {m : ℝ | m ≥ -3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_5_range_of_m_for_f_geq_x_minus_m_l1408_140804


namespace NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l1408_140819

/-- The cost ratio of muffins to bananas --/
def cost_ratio (muffin_cost banana_cost : ℚ) : ℚ := muffin_cost / banana_cost

/-- Susie's purchase --/
def susie_purchase (muffin_cost banana_cost : ℚ) : ℚ := 5 * muffin_cost + 4 * banana_cost

/-- Calvin's purchase --/
def calvin_purchase (muffin_cost banana_cost : ℚ) : ℚ := 3 * muffin_cost + 20 * banana_cost

theorem muffin_banana_cost_ratio :
  ∀ (muffin_cost banana_cost : ℚ),
    muffin_cost > 0 →
    banana_cost > 0 →
    calvin_purchase muffin_cost banana_cost = 3 * susie_purchase muffin_cost banana_cost →
    cost_ratio muffin_cost banana_cost = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_muffin_banana_cost_ratio_l1408_140819


namespace NUMINAMATH_CALUDE_benjamin_walks_95_miles_l1408_140890

/-- Represents the total miles Benjamin walks in a week -/
def total_miles_walked : ℕ :=
  let work_distance := 6
  let dog_walk_distance := 2
  let friend_distance := 1
  let store_distance := 3
  let work_days := 5
  let dog_walks_per_day := 2
  let days_in_week := 7
  let store_visits := 2
  let friend_visits := 1

  (2 * work_distance * work_days) + 
  (dog_walk_distance * dog_walks_per_day * days_in_week) + 
  (2 * store_distance * store_visits) + 
  (2 * friend_distance * friend_visits)

theorem benjamin_walks_95_miles : total_miles_walked = 95 := by
  sorry

end NUMINAMATH_CALUDE_benjamin_walks_95_miles_l1408_140890


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l1408_140870

/-- A quadratic function with vertex at (-3, 0) passing through (2, -64) has a = -64/25 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x y, y = a * x^2 + b * x + c) → -- quadratic function
  (0 = a * (-3)^2 + b * (-3) + c) → -- vertex at (-3, 0)
  (-64 = a * 2^2 + b * 2 + c) → -- passes through (2, -64)
  a = -64/25 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l1408_140870


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1408_140886

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h1 : ∀ x y : ℚ, f x * f y = f x + f y - f (x * y))
  (h2 : ∀ x y : ℚ, 1 + f (x + y) = f (x * y) + f x * f y) :
  (∀ x : ℚ, f x = 1) ∨ (∀ x : ℚ, f x = 1 - x) := by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1408_140886


namespace NUMINAMATH_CALUDE_train_length_l1408_140879

/-- The length of a train given specific crossing times and platform length -/
theorem train_length (platform_cross_time signal_cross_time : ℝ) (platform_length : ℝ) : 
  platform_cross_time = 54 →
  signal_cross_time = 18 →
  platform_length = 600.0000000000001 →
  ∃ (train_length : ℝ), train_length = 300.00000000000005 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l1408_140879


namespace NUMINAMATH_CALUDE_sequence_sum_comparison_l1408_140836

theorem sequence_sum_comparison (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h : ∀ k, k > 0 → S k = -a k - (1/2)^(k-1) + 2) : 
  (n ≥ 5 → S n > 2 - 1/(n-1)) ∧ 
  ((n = 3 ∨ n = 4) → S n < 2 - 1/(n-1)) := by
  sorry

end NUMINAMATH_CALUDE_sequence_sum_comparison_l1408_140836


namespace NUMINAMATH_CALUDE_strawberry_remainder_l1408_140884

/-- Given 3 kg and 300 g of strawberries, prove that after giving away 1 kg and 900 g, 
    the remaining amount is 1400 g. -/
theorem strawberry_remainder : 
  let total_kg : ℕ := 3
  let total_g : ℕ := 300
  let given_kg : ℕ := 1
  let given_g : ℕ := 900
  let g_per_kg : ℕ := 1000
  (total_kg * g_per_kg + total_g) - (given_kg * g_per_kg + given_g) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_remainder_l1408_140884


namespace NUMINAMATH_CALUDE_common_divisor_sequence_l1408_140824

theorem common_divisor_sequence (n : ℕ) : n = 4190 →
  ∀ k ∈ Finset.range 21, ∃ d > 1, d ∣ (n + k) ∧ d ∣ 30030 := by
  sorry

#check common_divisor_sequence

end NUMINAMATH_CALUDE_common_divisor_sequence_l1408_140824


namespace NUMINAMATH_CALUDE_area_at_stage_8_l1408_140802

/-- The area of a rectangle formed by adding squares in an arithmetic sequence -/
def rectangleArea (squareSize : ℕ) (stages : ℕ) : ℕ :=
  stages * (squareSize * squareSize)

/-- Theorem: The area of a rectangle formed by adding 4" by 4" squares
    in an arithmetic sequence for 8 stages is 128 square inches -/
theorem area_at_stage_8 : rectangleArea 4 8 = 128 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_8_l1408_140802
