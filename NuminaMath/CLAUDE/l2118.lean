import Mathlib

namespace NUMINAMATH_CALUDE_average_roots_quadratic_l2118_211865

theorem average_roots_quadratic (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a ≠ 0 → 3*x^2 + 4*x - 8 = 0 → (x₁ + x₂) / 2 = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_average_roots_quadratic_l2118_211865


namespace NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l2118_211873

/-- 
Given a quadratic equation ax^2 + 3bx + c = 0 with zero discriminant,
prove that the coefficients a, b, and c form a geometric progression.
-/
theorem quadratic_zero_discriminant_geometric_progression 
  (a b c : ℝ) (h_nonzero : a ≠ 0) 
  (h_discriminant : 9 * b^2 - 4 * a * c = 0) :
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r :=
sorry

end NUMINAMATH_CALUDE_quadratic_zero_discriminant_geometric_progression_l2118_211873


namespace NUMINAMATH_CALUDE_complement_of_union_equals_singleton_l2118_211811

def U : Finset Int := {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6}
def A : Finset Int := {-1, 0, 1, 2, 3}
def B : Finset Int := {-2, 3, 4, 5, 6}

theorem complement_of_union_equals_singleton : U \ (A ∪ B) = {-3} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_equals_singleton_l2118_211811


namespace NUMINAMATH_CALUDE_train_speed_calculation_l2118_211850

/-- 
Proves that the speed of a train is 72 km/hr, given its length, 
the platform length it crosses, and the time it takes to cross the platform.
-/
theorem train_speed_calculation (train_length platform_length crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : platform_length = 220)
  (h3 : crossing_time = 26) :
  (train_length + platform_length) / crossing_time * 3.6 = 72 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l2118_211850


namespace NUMINAMATH_CALUDE_circle_E_equation_line_circle_disjoint_l2118_211861

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define point D
def point_D : ℝ × ℝ := (-2, 0)

-- Define line l passing through D with slope k
def line_l (k x y : ℝ) : Prop := y = k * (x + 2)

-- Theorem for the equation of circle E
theorem circle_E_equation : ∀ x y : ℝ,
  (∃ k : ℝ, line_l k x y) →
  ((x + 1)^2 + (y - 2)^2 = 5) ↔ 
  (∃ t : ℝ, x = -2 * (1 - t) + 0 * t ∧ y = 0 * (1 - t) + 4 * t) :=
sorry

-- Theorem for the condition of line l and circle C being disjoint
theorem line_circle_disjoint : ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y → ¬circle_C x y) ↔ k < 3/4 :=
sorry

end NUMINAMATH_CALUDE_circle_E_equation_line_circle_disjoint_l2118_211861


namespace NUMINAMATH_CALUDE_system_solution_l2118_211816

theorem system_solution :
  ∃ (x y : ℚ), (7 * x = -5 - 3 * y) ∧ (4 * x = 5 * y - 36) ∧ (x = -41/11) ∧ (y = 232/33) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2118_211816


namespace NUMINAMATH_CALUDE_borrowed_sum_l2118_211832

/-- Proves that given the conditions of the problem, the principal amount is 1050 --/
theorem borrowed_sum (P : ℝ) : 
  (P * 0.06 * 6 = P - 672) → P = 1050 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_sum_l2118_211832


namespace NUMINAMATH_CALUDE_square_area_equals_perimeter_l2118_211820

theorem square_area_equals_perimeter (s : ℝ) (h : s > 0) : 
  s^2 = 4*s → 4*s = 16 := by
sorry

end NUMINAMATH_CALUDE_square_area_equals_perimeter_l2118_211820


namespace NUMINAMATH_CALUDE_f_increasing_iff_three_distinct_roots_iff_l2118_211866

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * |2 * a - x| + 2 * x

-- Theorem 1: f(x) is increasing on ℝ iff -1 ≤ a ≤ 1
theorem f_increasing_iff (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

-- Theorem 2: f(x) - t f(2a) = 0 has 3 distinct real roots iff 1 < t < 9/8
theorem three_distinct_roots_iff (a t : ℝ) :
  (a ∈ Set.Icc (-2) 2) →
  (∃ x y z : ℝ, x < y ∧ y < z ∧ f a x = t * f a (2 * a) ∧ f a y = t * f a (2 * a) ∧ f a z = t * f a (2 * a)) ↔
  (1 < t ∧ t < 9/8) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_three_distinct_roots_iff_l2118_211866


namespace NUMINAMATH_CALUDE_triangle_inequality_l2118_211856

theorem triangle_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
    (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : a + b + c = 1) :
    5 * (a^2 + b^2 + c^2) = 18 * a * b * c ∧ 18 * a * b * c ≥ 7/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2118_211856


namespace NUMINAMATH_CALUDE_f_derivatives_l2118_211890

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 3*x - 1

-- Theorem statement
theorem f_derivatives :
  (deriv f 2 = 0) ∧ (deriv f 1 = -1) := by
  sorry

end NUMINAMATH_CALUDE_f_derivatives_l2118_211890


namespace NUMINAMATH_CALUDE_modulo_eleven_residue_l2118_211838

theorem modulo_eleven_residue : (312 - 3 * 52 + 9 * 165 + 6 * 22) % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulo_eleven_residue_l2118_211838


namespace NUMINAMATH_CALUDE_marbles_collection_sum_l2118_211868

def total_marbles (adam mary greg john sarah : ℕ) : ℕ :=
  adam + mary + greg + john + sarah

theorem marbles_collection_sum :
  ∀ (adam mary greg john sarah : ℕ),
    adam = 29 →
    mary = adam - 11 →
    greg = adam + 14 →
    john = 2 * mary →
    sarah = greg - 7 →
    total_marbles adam mary greg john sarah = 162 :=
by
  sorry

end NUMINAMATH_CALUDE_marbles_collection_sum_l2118_211868


namespace NUMINAMATH_CALUDE_prime_divisor_of_2p_minus_1_l2118_211859

theorem prime_divisor_of_2p_minus_1 (p : ℕ) (hp : Prime p) :
  ∀ q : ℕ, Prime q → q ∣ (2^p - 1) → q > p :=
by sorry

end NUMINAMATH_CALUDE_prime_divisor_of_2p_minus_1_l2118_211859


namespace NUMINAMATH_CALUDE_parallelogram_area_l2118_211876

theorem parallelogram_area (a b : ℝ) (θ : ℝ) 
  (ha : a = 20) (hb : b = 10) (hθ : θ = 150 * π / 180) : 
  a * b * Real.sin θ = 100 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l2118_211876


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l2118_211893

theorem solve_exponential_equation :
  ∃ y : ℝ, (1000 : ℝ)^4 = 10^y ↔ y = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l2118_211893


namespace NUMINAMATH_CALUDE_english_only_enrollment_l2118_211825

theorem english_only_enrollment (total : ℕ) (both : ℕ) (german : ℕ) 
  (h1 : total = 32)
  (h2 : both = 12)
  (h3 : german = 22)
  (h4 : total ≥ german)
  (h5 : german ≥ both) :
  total - german + both = 10 := by
  sorry

end NUMINAMATH_CALUDE_english_only_enrollment_l2118_211825


namespace NUMINAMATH_CALUDE_set_contains_all_integers_l2118_211830

def is_closed_under_subtraction (A : Set ℤ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → (x - y) ∈ A

theorem set_contains_all_integers (A : Set ℤ) 
  (h_closed : is_closed_under_subtraction A) 
  (h_four : 4 ∈ A) 
  (h_nine : 9 ∈ A) : 
  A = Set.univ :=
sorry

end NUMINAMATH_CALUDE_set_contains_all_integers_l2118_211830


namespace NUMINAMATH_CALUDE_train_crossing_time_l2118_211857

/-- Proves that a train crosses a man in 18 seconds given its speed and time to cross a platform. -/
theorem train_crossing_time (train_speed : ℝ) (platform_length : ℝ) (platform_crossing_time : ℝ) :
  train_speed = 20 →
  platform_length = 340 →
  platform_crossing_time = 35 →
  (train_speed * platform_crossing_time - platform_length) / train_speed = 18 := by
  sorry

#check train_crossing_time

end NUMINAMATH_CALUDE_train_crossing_time_l2118_211857


namespace NUMINAMATH_CALUDE_median_on_hypotenuse_of_right_triangle_l2118_211844

theorem median_on_hypotenuse_of_right_triangle 
  (a b : ℝ) (ha : a = 6) (hb : b = 8) : 
  let c := Real.sqrt (a^2 + b^2)
  let m := c / 2
  m = 5 := by sorry

end NUMINAMATH_CALUDE_median_on_hypotenuse_of_right_triangle_l2118_211844


namespace NUMINAMATH_CALUDE_symmetric_curve_equation_l2118_211802

/-- Given a curve C defined by F(x, y) = 0 and a point of symmetry (a, b),
    the equation of the curve symmetric to C about (a, b) is F(2a-x, 2b-y) = 0 -/
theorem symmetric_curve_equation (F : ℝ → ℝ → ℝ) (a b : ℝ) :
  (∀ x y, F x y = 0 ↔ F (2*a - x) (2*b - y) = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_curve_equation_l2118_211802


namespace NUMINAMATH_CALUDE_sum_of_squares_204_l2118_211858

theorem sum_of_squares_204 (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℕ+) 
  (h : a₁^2 + (2*a₂)^2 + (3*a₃)^2 + (4*a₄)^2 + (5*a₅)^2 + (6*a₆)^2 + (7*a₇)^2 + (8*a₈)^2 = 204) :
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 8 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_204_l2118_211858


namespace NUMINAMATH_CALUDE_equilateral_triangle_vertices_product_l2118_211804

theorem equilateral_triangle_vertices_product (a b : ℝ) : 
  (∀ z : ℂ, z^3 = 1 ∧ z ≠ 1 → (a + 18 * I) * z = b + 42 * I) →
  a * b = -2652 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_vertices_product_l2118_211804


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2118_211840

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def digits_product (n : ℕ) : ℕ := (n / 10) * (n % 10)

def is_rearrangement (a b : ℕ) : Prop :=
  is_two_digit a ∧ is_two_digit b ∧
  (a / 10 = b % 10) ∧ (a % 10 = b / 10)

theorem unique_number_satisfying_conditions :
  ∃! n : ℕ, is_two_digit n ∧
    (n : ℚ) / (digits_product n : ℚ) = 16 / 3 ∧
    is_rearrangement n (n - 9) ∧
    n = 32 := by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l2118_211840


namespace NUMINAMATH_CALUDE_extreme_value_point_property_l2118_211855

/-- The function f(x) = x³ - x² + ax - a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - x^2 + a*x - a

/-- Theorem: If f(x) has an extreme value point x₀ and f(x₁) = f(x₀) where x₁ ≠ x₀, then x₁ + 2x₀ = 1 -/
theorem extreme_value_point_property (a : ℝ) (x₀ x₁ : ℝ) 
  (h_extreme : ∃ ε > 0, ∀ x, |x - x₀| < ε → f a x ≤ f a x₀ ∨ f a x ≥ f a x₀)
  (h_equal : f a x₁ = f a x₀)
  (h_distinct : x₁ ≠ x₀) : 
  x₁ + 2*x₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_point_property_l2118_211855


namespace NUMINAMATH_CALUDE_sequence_properties_l2118_211889

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := 3 * n - 1

-- Define the geometric sequence b_n
def b (n : ℕ) : ℚ := 2^n

-- Define the sum of the first n terms of a_n
def S (n : ℕ) : ℚ := n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

theorem sequence_properties :
  (a 1 = 2) ∧
  (b 1 = 2) ∧
  (a 4 + b 4 = 27) ∧
  (S 4 - b 4 = 10) ∧
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) ∧
  (∀ n : ℕ, b (n + 1) / b n = b 2 / b 1) ∧
  (∃ m : ℕ+, 
    (4 / m : ℚ) * a (7 - m) = (b 1)^2 ∧
    (4 / m : ℚ) * a 7 = (b 2)^2 ∧
    (4 / m : ℚ) * a (7 + 4 * m) = (b 3)^2) :=
by sorry

#check sequence_properties

end NUMINAMATH_CALUDE_sequence_properties_l2118_211889


namespace NUMINAMATH_CALUDE_initial_kola_solution_volume_l2118_211896

/-- Represents the composition and volume of a kola solution -/
structure KolaSolution where
  initialVolume : ℝ
  waterPercentage : ℝ
  concentratedKolaPercentage : ℝ
  sugarPercentage : ℝ

/-- Theorem stating the initial volume of the kola solution -/
theorem initial_kola_solution_volume
  (solution : KolaSolution)
  (h1 : solution.waterPercentage = 0.88)
  (h2 : solution.concentratedKolaPercentage = 0.05)
  (h3 : solution.sugarPercentage = 1 - solution.waterPercentage - solution.concentratedKolaPercentage)
  (h4 : let newVolume := solution.initialVolume + 3.2 + 10 + 6.8
        (solution.sugarPercentage * solution.initialVolume + 3.2) / newVolume = 0.075) :
  solution.initialVolume = 340 := by
  sorry

end NUMINAMATH_CALUDE_initial_kola_solution_volume_l2118_211896


namespace NUMINAMATH_CALUDE_f_properties_l2118_211824

noncomputable def f (x : ℝ) : ℝ := x * (Real.exp x + Real.exp (-x))

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2118_211824


namespace NUMINAMATH_CALUDE_money_division_l2118_211819

theorem money_division (a_share b_share c_share total : ℝ) : 
  (b_share = 0.65 * a_share) →
  (c_share = 0.4 * a_share) →
  (c_share = 32) →
  (total = a_share + b_share + c_share) →
  total = 164 := by
sorry

end NUMINAMATH_CALUDE_money_division_l2118_211819


namespace NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l2118_211899

theorem square_plus_abs_zero_implies_both_zero (a b : ℝ) :
  a^2 + |b| = 0 → a = 0 ∧ b = 0 := by
sorry

end NUMINAMATH_CALUDE_square_plus_abs_zero_implies_both_zero_l2118_211899


namespace NUMINAMATH_CALUDE_paco_cookies_l2118_211867

def cookie_problem (initial_cookies : ℕ) (given_to_friend1 : ℕ) (given_to_friend2 : ℕ) (eaten : ℕ) : Prop :=
  let total_given := given_to_friend1 + given_to_friend2
  eaten - total_given = 0

theorem paco_cookies : cookie_problem 100 15 25 40 := by
  sorry

end NUMINAMATH_CALUDE_paco_cookies_l2118_211867


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2118_211803

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 5) * (Real.sqrt 7 / Real.sqrt 11) * (Real.sqrt 15 / Real.sqrt 2) = 
  (3 * Real.sqrt 154) / 22 := by
sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2118_211803


namespace NUMINAMATH_CALUDE_tire_usage_proof_l2118_211817

/-- Represents the number of miles each tire is used when a car with 6 tires
    travels 40,000 miles, with each tire being used equally. -/
def miles_per_tire : ℕ := 26667

/-- The total number of tires. -/
def total_tires : ℕ := 6

/-- The number of tires used on the road at any given time. -/
def road_tires : ℕ := 4

/-- The total distance traveled by the car in miles. -/
def total_distance : ℕ := 40000

theorem tire_usage_proof :
  miles_per_tire * total_tires = total_distance * road_tires :=
sorry

end NUMINAMATH_CALUDE_tire_usage_proof_l2118_211817


namespace NUMINAMATH_CALUDE_expression_evaluation_l2118_211878

theorem expression_evaluation : 23 - 17 - (-7) + (-16) = -3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2118_211878


namespace NUMINAMATH_CALUDE_total_ice_cream_scoops_l2118_211806

def single_cone : ℕ := 1
def double_cone : ℕ := 3
def milkshake : ℕ := 2  -- Rounded up from 1.5
def banana_split : ℕ := 4 * single_cone
def waffle_bowl : ℕ := banana_split + 2
def ice_cream_sandwich : ℕ := waffle_bowl - 3

theorem total_ice_cream_scoops : 
  single_cone + double_cone + milkshake + banana_split + waffle_bowl + ice_cream_sandwich = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_ice_cream_scoops_l2118_211806


namespace NUMINAMATH_CALUDE_right_isosceles_not_scalene_l2118_211888

/-- A triangle in Euclidean space -/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- A right isosceles triangle -/
def RightIsosceles (t : Triangle) : Prop :=
  ∃ (a b : ℝ), t.A = (0, 0) ∧ t.B = (a, 0) ∧ t.C = (0, a) ∧ a > 0

/-- A scalene triangle -/
def Scalene (t : Triangle) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B ≠ d t.B t.C ∧ d t.B t.C ≠ d t.C t.A ∧ d t.C t.A ≠ d t.A t.B

theorem right_isosceles_not_scalene :
  ∀ t : Triangle, RightIsosceles t → ¬ Scalene t :=
sorry

end NUMINAMATH_CALUDE_right_isosceles_not_scalene_l2118_211888


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l2118_211892

theorem complex_power_magnitude : Complex.abs ((2 + 2 * Complex.I * Real.sqrt 3) ^ 6) = 4096 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l2118_211892


namespace NUMINAMATH_CALUDE_coefficient_of_x_4_in_expansion_l2118_211836

def binomial_expansion (n : ℕ) (x : ℝ) : ℝ → ℝ := 
  fun a => (1 + a * x)^n

def coefficient_of_x_power (f : ℝ → ℝ) (n : ℕ) : ℝ := sorry

theorem coefficient_of_x_4_in_expansion : 
  coefficient_of_x_power (fun x => (1 + x^2) * binomial_expansion 5 (-2) x) 4 = 120 := by sorry

end NUMINAMATH_CALUDE_coefficient_of_x_4_in_expansion_l2118_211836


namespace NUMINAMATH_CALUDE_arithmetic_progression_logarithm_l2118_211847

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the arithmetic progression property
def isArithmeticProgression (a b c : ℝ) : Prop := 2 * b = a + c

-- Theorem statement
theorem arithmetic_progression_logarithm :
  isArithmeticProgression (lg 3) (lg 6) (lg x) → x = 12 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_logarithm_l2118_211847


namespace NUMINAMATH_CALUDE_louise_pencils_l2118_211897

theorem louise_pencils (box_capacity : ℕ) (red_pencils : ℕ) (yellow_pencils : ℕ) (total_boxes : ℕ)
  (h1 : box_capacity = 20)
  (h2 : red_pencils = 20)
  (h3 : yellow_pencils = 40)
  (h4 : total_boxes = 8) :
  let blue_pencils := 2 * red_pencils
  let total_capacity := box_capacity * total_boxes
  let other_pencils := red_pencils + blue_pencils + yellow_pencils
  let green_pencils := total_capacity - other_pencils
  green_pencils = 60 ∧ green_pencils = red_pencils + blue_pencils :=
by sorry


end NUMINAMATH_CALUDE_louise_pencils_l2118_211897


namespace NUMINAMATH_CALUDE_range_of_m_l2118_211854

/-- The function f(x) = x^2 - 4x + 5 -/
def f (x : ℝ) := x^2 - 4*x + 5

/-- The maximum value of f on [0, m] is 5 -/
def max_value := 5

/-- The minimum value of f on [0, m] is 1 -/
def min_value := 1

/-- The range of m for which f has max_value and min_value on [0, m] -/
theorem range_of_m :
  ∃ (m : ℝ), m ∈ Set.Icc 2 4 ∧
  (∀ x ∈ Set.Icc 0 m, f x ≤ max_value) ∧
  (∃ x ∈ Set.Icc 0 m, f x = max_value) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ min_value) ∧
  (∃ x ∈ Set.Icc 0 m, f x = min_value) ∧
  (∀ m' > 4, ∃ x ∈ Set.Icc 0 m', f x > max_value) ∧
  (∀ m' < 2, ∀ x ∈ Set.Icc 0 m', f x > min_value) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2118_211854


namespace NUMINAMATH_CALUDE_student_multiplication_error_l2118_211879

theorem student_multiplication_error (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  (78 : ℚ) * ((1 + (100 * a + 10 * b + c : ℚ) / 999) - (1 + (a / 10 + b / 100 + c / 1000))) = (3 / 5) →
  100 * a + 10 * b + c = 765 := by
  sorry

end NUMINAMATH_CALUDE_student_multiplication_error_l2118_211879


namespace NUMINAMATH_CALUDE_inscribed_squares_inequality_l2118_211871

/-- Given a triangle ABC with sides a, b, and c, and inscribed squares with side lengths x, y, and z
    on sides BC, AC, and AB respectively, prove that (a/x) + (b/y) + (c/z) ≥ 3 + 2√3. -/
theorem inscribed_squares_inequality (a b c x y z : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ x > 0 ∧ y > 0 ∧ z > 0)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
    (h_square_x : x ≤ b ∧ x ≤ c)
    (h_square_y : y ≤ c ∧ y ≤ a)
    (h_square_z : z ≤ a ∧ z ≤ b) :
  a / x + b / y + c / z ≥ 3 + 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_inscribed_squares_inequality_l2118_211871


namespace NUMINAMATH_CALUDE_lidia_app_purchase_l2118_211810

/-- Proves that Lidia will be left with $15 after purchasing apps with a discount --/
theorem lidia_app_purchase (app_cost : ℝ) (num_apps : ℕ) (budget : ℝ) (discount_rate : ℝ) :
  app_cost = 4 →
  num_apps = 15 →
  budget = 66 →
  discount_rate = 0.15 →
  budget - (num_apps * app_cost * (1 - discount_rate)) = 15 :=
by
  sorry

#check lidia_app_purchase

end NUMINAMATH_CALUDE_lidia_app_purchase_l2118_211810


namespace NUMINAMATH_CALUDE_find_m_value_l2118_211827

theorem find_m_value (n : ℕ) (m : ℕ) (h1 : n = 9998) (h2 : 72517 * (n + 1) = m) : 
  m = 725092483 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l2118_211827


namespace NUMINAMATH_CALUDE_sports_participation_l2118_211808

theorem sports_participation (total_students : ℕ) (basketball cricket soccer : ℕ)
  (basketball_cricket basketball_soccer cricket_soccer : ℕ) (all_three : ℕ)
  (h1 : total_students = 50)
  (h2 : basketball = 16)
  (h3 : cricket = 11)
  (h4 : soccer = 10)
  (h5 : basketball_cricket = 5)
  (h6 : basketball_soccer = 4)
  (h7 : cricket_soccer = 3)
  (h8 : all_three = 2) :
  basketball + cricket + soccer - (basketball_cricket + basketball_soccer + cricket_soccer) + all_three = 27 := by
  sorry

end NUMINAMATH_CALUDE_sports_participation_l2118_211808


namespace NUMINAMATH_CALUDE_complex_inside_unit_circle_l2118_211823

theorem complex_inside_unit_circle (x : ℝ) :
  (∀ z : ℂ, z = x - (1/3 : ℝ) * Complex.I → Complex.abs z < 1) →
  -2 * Real.sqrt 2 / 3 < x ∧ x < 2 * Real.sqrt 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_inside_unit_circle_l2118_211823


namespace NUMINAMATH_CALUDE_russian_number_sequence_next_two_elements_l2118_211848

/-- Represents the first letter of a Russian number word -/
inductive RussianNumberLetter
| O  -- Один (One)
| D  -- Два (Two)
| T  -- Три (Three)
| C  -- Четыре (Four)
| P  -- Пять (Five)
| S  -- Шесть (Six)
| S' -- Семь (Seven)
| V  -- Восемь (Eight)

/-- Returns the RussianNumberLetter for a given natural number -/
def russianNumberLetter (n : ℕ) : RussianNumberLetter :=
  match n with
  | 1 => RussianNumberLetter.O
  | 2 => RussianNumberLetter.D
  | 3 => RussianNumberLetter.T
  | 4 => RussianNumberLetter.C
  | 5 => RussianNumberLetter.P
  | 6 => RussianNumberLetter.S
  | 7 => RussianNumberLetter.S'
  | 8 => RussianNumberLetter.V
  | _ => RussianNumberLetter.O  -- Default case, should not be reached for 1-8

theorem russian_number_sequence_next_two_elements :
  russianNumberLetter 7 = RussianNumberLetter.S' ∧
  russianNumberLetter 8 = RussianNumberLetter.V :=
by sorry

end NUMINAMATH_CALUDE_russian_number_sequence_next_two_elements_l2118_211848


namespace NUMINAMATH_CALUDE_lincoln_county_houses_l2118_211880

/-- The original number of houses in Lincoln County -/
def original_houses : ℕ := 20817

/-- The number of houses built during the housing boom -/
def houses_built : ℕ := 97741

/-- The current number of houses in Lincoln County -/
def current_houses : ℕ := 118558

/-- Theorem stating that the original number of houses plus the houses built
    during the boom equals the current number of houses -/
theorem lincoln_county_houses :
  original_houses + houses_built = current_houses := by
  sorry

end NUMINAMATH_CALUDE_lincoln_county_houses_l2118_211880


namespace NUMINAMATH_CALUDE_loan_problem_l2118_211809

/-- Proves that given the conditions of the loan problem, the time A lent money to C is 2/3 of a year. -/
theorem loan_problem (principal_B principal_C total_interest : ℚ) 
  (time_B : ℚ) (rate : ℚ) :
  principal_B = 5000 →
  principal_C = 3000 →
  time_B = 2 →
  rate = 9 / 100 →
  total_interest = 1980 →
  total_interest = principal_B * rate * time_B + principal_C * rate * (2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_loan_problem_l2118_211809


namespace NUMINAMATH_CALUDE_regular_polygon_160_degrees_has_18_sides_l2118_211818

/-- A regular polygon with interior angles measuring 160° has 18 sides. -/
theorem regular_polygon_160_degrees_has_18_sides :
  ∀ n : ℕ,
  n ≥ 3 →
  (180 * (n - 2) : ℝ) / n = 160 →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_160_degrees_has_18_sides_l2118_211818


namespace NUMINAMATH_CALUDE_unequal_outcome_probability_l2118_211807

def num_grandchildren : ℕ := 10

theorem unequal_outcome_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_outcomes := Nat.choose num_grandchildren (num_grandchildren / 2)
  (total_outcomes - equal_outcomes) / total_outcomes = 193 / 256 := by
  sorry

end NUMINAMATH_CALUDE_unequal_outcome_probability_l2118_211807


namespace NUMINAMATH_CALUDE_quadratic_root_zero_l2118_211833

theorem quadratic_root_zero (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m^2 - 1 = 0 ∧ x = 0) → m = 1 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_zero_l2118_211833


namespace NUMINAMATH_CALUDE_trigonometric_identities_l2118_211842

theorem trigonometric_identities (α β : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : 3 * Real.sin α = 4 * Real.cos α)
  (h4 : Real.cos (α + β) = -(2 * Real.sqrt 5) / 5) :
  Real.cos (2 * α) = -7 / 25 ∧ Real.sin β = (2 * Real.sqrt 5) / 5 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l2118_211842


namespace NUMINAMATH_CALUDE_unique_acute_prime_angled_triangle_l2118_211815

-- Define a structure for a triangle with three angles
structure Triangle where
  angle1 : ℕ
  angle2 : ℕ
  angle3 : ℕ

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define what it means for a triangle to be acute
def isAcute (t : Triangle) : Prop := t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

-- Define what it means for a triangle to have prime angles
def hasPrimeAngles (t : Triangle) : Prop := 
  isPrime t.angle1 ∧ isPrime t.angle2 ∧ isPrime t.angle3

-- Define what it means for a triangle to be valid (sum of angles is 180°)
def isValidTriangle (t : Triangle) : Prop := t.angle1 + t.angle2 + t.angle3 = 180

-- Define what it means for a triangle to be isosceles
def isIsosceles (t : Triangle) : Prop := 
  t.angle1 = t.angle2 ∨ t.angle2 = t.angle3 ∨ t.angle3 = t.angle1

-- Theorem statement
theorem unique_acute_prime_angled_triangle : 
  ∃! t : Triangle, isAcute t ∧ hasPrimeAngles t ∧ isValidTriangle t ∧
  t.angle1 = 2 ∧ t.angle2 = 89 ∧ t.angle3 = 89 ∧ isIsosceles t :=
sorry

end NUMINAMATH_CALUDE_unique_acute_prime_angled_triangle_l2118_211815


namespace NUMINAMATH_CALUDE_two_number_difference_l2118_211864

theorem two_number_difference (a b : ℕ) : 
  a + b = 22305 →
  a % 5 = 0 →
  b = (a / 10) + 3 →
  a - b = 14872 :=
by sorry

end NUMINAMATH_CALUDE_two_number_difference_l2118_211864


namespace NUMINAMATH_CALUDE_min_triangles_to_cover_l2118_211853

/-- The minimum number of unit equilateral triangles needed to cover a larger equilateral triangle and a square -/
theorem min_triangles_to_cover (small_side : ℝ) (large_side : ℝ) (square_side : ℝ) :
  small_side = 1 →
  large_side = 12 →
  square_side = 4 →
  ∃ (n : ℕ), n = ⌈145 * Real.sqrt 3 + 64⌉ ∧
    n * (Real.sqrt 3 / 4 * small_side^2) ≥
      (Real.sqrt 3 / 4 * large_side^2) + square_side^2 :=
by sorry

end NUMINAMATH_CALUDE_min_triangles_to_cover_l2118_211853


namespace NUMINAMATH_CALUDE_tangent_circles_bisector_l2118_211841

-- Define the basic geometric objects
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

def Point := ℝ × ℝ

-- Define the geometric relations
def tangentCircles (c1 c2 : Circle) (p : Point) : Prop := sorry

def tangentLineToCircle (l : Line) (c : Circle) (a : Point) : Prop := sorry

def lineIntersectsCircle (l : Line) (c : Circle) (b c : Point) : Prop := sorry

def angleBisector (l : Line) (a b c : Point) : Prop := sorry

-- State the theorem
theorem tangent_circles_bisector
  (c1 c2 : Circle) (p a b c : Point) (d : Line) :
  tangentCircles c1 c2 p →
  tangentLineToCircle d c1 a →
  lineIntersectsCircle d c2 b c →
  angleBisector (Line.mk p a) p b c := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_bisector_l2118_211841


namespace NUMINAMATH_CALUDE_simplify_expression_l2118_211845

theorem simplify_expression (m : ℝ) (hm : m > 0) :
  (m^(1/2) * 3*m * 4*m) / ((6*m)^5 * m^(1/4)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2118_211845


namespace NUMINAMATH_CALUDE_extremum_and_range_l2118_211814

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - a*x + b

-- Theorem statement
theorem extremum_and_range :
  ∀ a b : ℝ,
  (∃ ε > 0, ∀ x ∈ Set.Ioo (2 - ε) (2 + ε), f a b x ≥ f a b 2) ∧
  (f a b 2 = -8) →
  (a = 12 ∧ b = 8) ∧
  (∀ x ∈ Set.Icc (-3) 3, -8 ≤ f 12 8 x ∧ f 12 8 x ≤ 24) :=
by sorry

end NUMINAMATH_CALUDE_extremum_and_range_l2118_211814


namespace NUMINAMATH_CALUDE_initial_stuffed_animals_l2118_211829

def stuffed_animals (x : ℕ) : Prop :=
  let after_mom := x + 2
  let from_dad := 3 * after_mom
  x + 2 + from_dad = 48

theorem initial_stuffed_animals :
  ∃ (x : ℕ), stuffed_animals x ∧ x = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_stuffed_animals_l2118_211829


namespace NUMINAMATH_CALUDE_piecewise_continuity_l2118_211872

/-- A piecewise function f defined on real numbers -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 3 then x^2 + x + 2 else 2*x + a

/-- Theorem stating that the piecewise function f is continuous at x = 3 if and only if a = 8 -/
theorem piecewise_continuity (a : ℝ) :
  ContinuousAt (f a) 3 ↔ a = 8 := by sorry

end NUMINAMATH_CALUDE_piecewise_continuity_l2118_211872


namespace NUMINAMATH_CALUDE_no_unique_solution_l2118_211885

theorem no_unique_solution (x y z : ℕ+) : 
  ¬∃ (f : ℕ+ → ℕ+ → ℕ+ → Prop), 
    (∀ (a b c : ℕ+), f a b c ↔ Real.sqrt (a^2 + Real.sqrt ((b:ℝ)/(c:ℝ))) = (b:ℝ)^2 * Real.sqrt ((a:ℝ)/(c:ℝ))) ∧
    (∃! (g : ℕ+ → ℕ+ → ℕ+ → Prop), ∀ (a b c : ℕ+), g a b c ↔ f a b c) :=
sorry

end NUMINAMATH_CALUDE_no_unique_solution_l2118_211885


namespace NUMINAMATH_CALUDE_unknown_number_proof_l2118_211837

theorem unknown_number_proof :
  let N : ℕ := 15222392625570
  let a : ℕ := 1155
  let b : ℕ := 1845
  let product : ℕ := a * b
  let difference : ℕ := b - a
  let quotient : ℕ := 15 * (difference * difference)
  N / product = quotient ∧ N % product = 570 :=
by sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l2118_211837


namespace NUMINAMATH_CALUDE_early_arrival_l2118_211834

/-- A boy's journey to school -/
def school_journey (usual_time : ℕ) (speed_factor : ℚ) : Prop :=
  let new_time := (usual_time : ℚ) * (1 / speed_factor)
  (usual_time : ℚ) - new_time = 7

theorem early_arrival : school_journey 49 (7/6) := by
  sorry

end NUMINAMATH_CALUDE_early_arrival_l2118_211834


namespace NUMINAMATH_CALUDE_catering_weight_calculation_catering_weight_proof_l2118_211860

theorem catering_weight_calculation (silverware_weight : ℕ) (silverware_per_setting : ℕ)
  (plate_weight : ℕ) (plates_per_setting : ℕ) (num_tables : ℕ) (settings_per_table : ℕ)
  (backup_settings : ℕ) : ℕ :=
  let weight_per_setting := silverware_weight * silverware_per_setting + plate_weight * plates_per_setting
  let total_settings := num_tables * settings_per_table + backup_settings
  weight_per_setting * total_settings

theorem catering_weight_proof :
  catering_weight_calculation 4 3 12 2 15 8 20 = 5040 := by
  sorry

end NUMINAMATH_CALUDE_catering_weight_calculation_catering_weight_proof_l2118_211860


namespace NUMINAMATH_CALUDE_unique_function_property_l2118_211851

theorem unique_function_property (f : ℕ → ℕ) : 
  (∀ (a b c : ℕ), (f a + f b + f c - a * b - b * c - c * a) ∣ (a * f a + b * f b + c * f c - 3 * a * b * c)) → 
  (∀ (a : ℕ), f a = a ^ 2) := by
sorry

end NUMINAMATH_CALUDE_unique_function_property_l2118_211851


namespace NUMINAMATH_CALUDE_tenth_term_is_one_over_120_l2118_211894

def a (n : ℕ) : ℚ := 1 / (n * (n + 2))

theorem tenth_term_is_one_over_120 : a 10 = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_is_one_over_120_l2118_211894


namespace NUMINAMATH_CALUDE_virginia_adrienne_difference_l2118_211839

/-- The combined total years of teaching for Virginia, Adrienne, and Dennis -/
def total_years : ℕ := 93

/-- The number of years Dennis has taught -/
def dennis_years : ℕ := 40

/-- The number of years Virginia has taught -/
def virginia_years : ℕ := dennis_years - 9

/-- The number of years Adrienne has taught -/
def adrienne_years : ℕ := total_years - dennis_years - virginia_years

/-- Theorem stating the difference in teaching years between Virginia and Adrienne -/
theorem virginia_adrienne_difference : virginia_years - adrienne_years = 9 := by
  sorry

end NUMINAMATH_CALUDE_virginia_adrienne_difference_l2118_211839


namespace NUMINAMATH_CALUDE_part1_part2_part3_l2118_211898

-- Define the functions f and g
def f (x : ℝ) := x - 2
def g (m : ℝ) (x : ℝ) := x^2 - 2*m*x + 4

-- Part 1
theorem part1 (m : ℝ) :
  (∀ x, g m x > f x) ↔ m ∈ Set.Ioo (-Real.sqrt 6 - 1/2) (Real.sqrt 6 - 1/2) :=
sorry

-- Part 2
theorem part2 (m : ℝ) :
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 4 5, g m x₁ = f x₂) ↔ 
  m ∈ Set.Icc (5/4) (Real.sqrt 2) :=
sorry

-- Part 3
theorem part3 :
  (∀ n : ℝ, ∃ x₀ ∈ Set.Icc (-2) 2, |g (-1) x₀ - x₀^2 + n| ≥ k) ↔
  k ∈ Set.Iic 4 :=
sorry

end NUMINAMATH_CALUDE_part1_part2_part3_l2118_211898


namespace NUMINAMATH_CALUDE_equal_bill_time_l2118_211821

/-- United Telephone's base rate -/
def united_base : ℝ := 8

/-- United Telephone's per-minute rate -/
def united_per_minute : ℝ := 0.25

/-- Atlantic Call's base rate -/
def atlantic_base : ℝ := 12

/-- Atlantic Call's per-minute rate -/
def atlantic_per_minute : ℝ := 0.20

/-- The number of minutes at which the bills are equal -/
def equal_minutes : ℝ := 80

theorem equal_bill_time :
  united_base + united_per_minute * equal_minutes =
  atlantic_base + atlantic_per_minute * equal_minutes :=
by sorry

end NUMINAMATH_CALUDE_equal_bill_time_l2118_211821


namespace NUMINAMATH_CALUDE_nonzero_x_equality_l2118_211831

theorem nonzero_x_equality (x : ℝ) (hx : x ≠ 0) (h : (9 * x)^18 = (18 * x)^9) : x = 2/9 := by
  sorry

end NUMINAMATH_CALUDE_nonzero_x_equality_l2118_211831


namespace NUMINAMATH_CALUDE_rhombus_diagonal_length_l2118_211882

/-- Proves that in a rhombus with an area of 127.5 cm² and one diagonal of 15 cm, 
    the length of the other diagonal is 17 cm. -/
theorem rhombus_diagonal_length (area : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  area = 127.5 → d1 = 15 → area = (d1 * d2) / 2 → d2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_length_l2118_211882


namespace NUMINAMATH_CALUDE_product_of_special_integers_l2118_211869

theorem product_of_special_integers (p q r : ℤ) 
  (h1 : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h2 : p + q + r = 30)
  (h3 : 1 / p + 1 / q + 1 / r + 1000 / (p * q * r) = 1) :
  p * q * r = 1600 := by
  sorry

end NUMINAMATH_CALUDE_product_of_special_integers_l2118_211869


namespace NUMINAMATH_CALUDE_parabola_line_intersection_slope_l2118_211800

/-- The parabola y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- A line with slope k passing through the focus -/
def line (k x y : ℝ) : Prop := y = k * (x - 1)

/-- The distance between two points on the parabola -/
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := x₁ + x₂ + 2

theorem parabola_line_intersection_slope 
  (k : ℝ) 
  (h_k : k ≠ 0) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h_A : parabola x₁ y₁) 
  (h_B : parabola x₂ y₂) 
  (h_l₁ : line k x₁ y₁) 
  (h_l₂ : line k x₂ y₂) 
  (h_dist : distance x₁ y₁ x₂ y₂ = 5) : 
  k = 2 ∨ k = -2 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_slope_l2118_211800


namespace NUMINAMATH_CALUDE_harmonious_equations_have_real_roots_l2118_211826

/-- A harmonious equation is a quadratic equation ax² + bx + c = 0 where a ≠ 0 and b = a + c -/
def HarmoniousEquation (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b = a + c

/-- The discriminant of a quadratic equation ax² + bx + c = 0 -/
def Discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4*a*c

/-- A quadratic equation has real roots if and only if its discriminant is non-negative -/
def HasRealRoots (a b c : ℝ) : Prop :=
  Discriminant a b c ≥ 0

/-- Theorem: Harmonious equations always have real roots -/
theorem harmonious_equations_have_real_roots (a b c : ℝ) :
  HarmoniousEquation a b c → HasRealRoots a b c :=
by sorry

end NUMINAMATH_CALUDE_harmonious_equations_have_real_roots_l2118_211826


namespace NUMINAMATH_CALUDE_solution_set_equality_l2118_211875

def f (x : ℤ) : ℤ := 2 * x^2 + x - 6

def is_prime_power (n : ℤ) : Prop :=
  ∃ (p : ℕ) (k : ℕ+), Nat.Prime p ∧ n = (p : ℤ) ^ (k : ℕ)

theorem solution_set_equality : 
  {x : ℤ | ∃ (y : ℤ), y > 0 ∧ is_prime_power y ∧ f x = y} = {-3, 2, 5} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2118_211875


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2118_211884

theorem quadratic_inequality_solution (a : ℝ) (h : a ∈ Set.Icc (-1) 1) :
  ∀ x : ℝ, x^2 + (a - 4) * x + 4 - 2 * a > 0 ↔ x < 1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2118_211884


namespace NUMINAMATH_CALUDE_debt_average_payment_l2118_211828

theorem debt_average_payment 
  (total_installments : ℕ) 
  (first_payment_count : ℕ) 
  (first_payment_amount : ℚ) 
  (additional_amount : ℚ) : 
  total_installments = 100 → 
  first_payment_count = 30 → 
  first_payment_amount = 620 → 
  additional_amount = 110 → 
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + additional_amount
  let total_amount := 
    (first_payment_count * first_payment_amount) + 
    (remaining_payment_count * remaining_payment_amount)
  total_amount / total_installments = 697 := by
sorry

end NUMINAMATH_CALUDE_debt_average_payment_l2118_211828


namespace NUMINAMATH_CALUDE_prob_specific_draw_l2118_211887

def standard_deck : ℕ := 52

def prob_first_five (deck : ℕ) : ℚ := 4 / deck

def prob_second_diamond (deck : ℕ) : ℚ := 13 / (deck - 1)

def prob_third_three (deck : ℕ) : ℚ := 4 / (deck - 2)

theorem prob_specific_draw (deck : ℕ) (h : deck = standard_deck) :
  prob_first_five deck * prob_second_diamond deck * prob_third_three deck = 17 / 11050 := by
  sorry

end NUMINAMATH_CALUDE_prob_specific_draw_l2118_211887


namespace NUMINAMATH_CALUDE_triangle_third_side_l2118_211852

theorem triangle_third_side (a b x : ℝ) : 
  a = 3 ∧ b = 9 ∧ 
  (a + b > x) ∧ (a + x > b) ∧ (b + x > a) →
  x = 10 → True :=
by sorry

end NUMINAMATH_CALUDE_triangle_third_side_l2118_211852


namespace NUMINAMATH_CALUDE_circle_constraint_extrema_l2118_211849

theorem circle_constraint_extrema :
  ∀ x y : ℝ, x^2 + y^2 = 1 →
  (∀ a b : ℝ, a^2 + b^2 = 1 → (1 + x*y)*(1 - x*y) ≤ (1 + a*b)*(1 - a*b)) ∧
  (∃ a b : ℝ, a^2 + b^2 = 1 ∧ (1 + a*b)*(1 - a*b) = 1) ∧
  (∀ a b : ℝ, a^2 + b^2 = 1 → (1 + a*b)*(1 - a*b) ≥ 3/4) ∧
  (∃ a b : ℝ, a^2 + b^2 = 1 ∧ (1 + a*b)*(1 - a*b) = 3/4) :=
by sorry

end NUMINAMATH_CALUDE_circle_constraint_extrema_l2118_211849


namespace NUMINAMATH_CALUDE_value_of_y_l2118_211886

theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l2118_211886


namespace NUMINAMATH_CALUDE_sequence_zero_l2118_211891

/-- A sequence of real numbers indexed by positive integers. -/
def RealSequence := ℕ+ → ℝ

/-- The property that b_n ≤ c_n for all n. -/
def LessEqProperty (b c : RealSequence) : Prop :=
  ∀ n : ℕ+, b n ≤ c n

/-- The property that b_{n+1} and c_{n+1} are roots of x^2 + b_n*x + c_n = 0. -/
def RootProperty (b c : RealSequence) : Prop :=
  ∀ n : ℕ+, (b (n + 1))^2 + (b n) * (b (n + 1)) + (c n) = 0 ∧
            (c (n + 1))^2 + (b n) * (c (n + 1)) + (c n) = 0

theorem sequence_zero (b c : RealSequence) 
  (h1 : LessEqProperty b c) (h2 : RootProperty b c) :
  (∀ n : ℕ+, b n = 0 ∧ c n = 0) :=
sorry

end NUMINAMATH_CALUDE_sequence_zero_l2118_211891


namespace NUMINAMATH_CALUDE_right_triangle_division_l2118_211835

theorem right_triangle_division (n : ℝ) (h : n > 0) :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    ∃ (x y : ℝ),
      0 < x ∧ x < c ∧
      0 < y ∧ y < c ∧
      x * y = a * b ∧
      (1/2) * x * a = n * x * y ∧
      (1/2) * y * b = (1/(4*n)) * x * y :=
sorry

end NUMINAMATH_CALUDE_right_triangle_division_l2118_211835


namespace NUMINAMATH_CALUDE_correction_is_15x_l2118_211813

/-- Represents the value of a coin in cents -/
def coin_value (coin : String) : ℕ :=
  match coin with
  | "quarter" => 25
  | "dime" => 10
  | "half-dollar" => 50
  | "nickel" => 5
  | _ => 0

/-- Calculates the correction needed for miscounted coins -/
def correction_needed (x : ℕ) : ℤ :=
  let quarter_dime_diff := (coin_value "quarter" - coin_value "dime") * (2 * x)
  let half_dollar_nickel_diff := (coin_value "half-dollar" - coin_value "nickel") * x
  quarter_dime_diff - half_dollar_nickel_diff

theorem correction_is_15x (x : ℕ) : correction_needed x = 15 * x := by
  sorry

end NUMINAMATH_CALUDE_correction_is_15x_l2118_211813


namespace NUMINAMATH_CALUDE_base_three_20121_equals_178_l2118_211874

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_three_20121_equals_178 :
  base_three_to_decimal [2, 0, 1, 2, 1] = 178 := by
  sorry

end NUMINAMATH_CALUDE_base_three_20121_equals_178_l2118_211874


namespace NUMINAMATH_CALUDE_darla_electricity_bill_l2118_211883

-- Define the cost per watt in cents
def cost_per_watt : ℕ := 400

-- Define the amount of electricity used in watts
def electricity_used : ℕ := 300

-- Define the late fee in cents
def late_fee : ℕ := 15000

-- Define the total cost in cents
def total_cost : ℕ := cost_per_watt * electricity_used + late_fee

-- Theorem statement
theorem darla_electricity_bill : total_cost = 135000 := by
  sorry

end NUMINAMATH_CALUDE_darla_electricity_bill_l2118_211883


namespace NUMINAMATH_CALUDE_unique_triple_lcm_l2118_211812

theorem unique_triple_lcm : 
  ∃! (x y z : ℕ+), 
    Nat.lcm x y = 180 ∧ 
    Nat.lcm x z = 420 ∧ 
    Nat.lcm y z = 1260 := by
  sorry

end NUMINAMATH_CALUDE_unique_triple_lcm_l2118_211812


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2118_211805

theorem least_subtraction_for_divisibility (n : ℕ) : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(10 ∣ (724946 - y))) ∧ 
  (10 ∣ (724946 - x)) := by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_l2118_211805


namespace NUMINAMATH_CALUDE_probability_two_face_cards_total_20_l2118_211843

-- Define the deck
def deck_size : ℕ := 52

-- Define the number of face cards
def face_cards : ℕ := 12

-- Define the value of a face card
def face_card_value : ℕ := 10

-- Define the theorem
theorem probability_two_face_cards_total_20 :
  (face_cards : ℚ) * (face_cards - 1) / (deck_size * (deck_size - 1)) = 11 / 221 :=
sorry

end NUMINAMATH_CALUDE_probability_two_face_cards_total_20_l2118_211843


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2118_211801

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes forms a 30° angle with the x-axis,
    then its eccentricity is 2√3/3. -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_asymptote : b / a = Real.tan (π / 6)) :
  let e := Real.sqrt (1 + (b / a)^2)
  e = 2 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2118_211801


namespace NUMINAMATH_CALUDE_sqrt_sum_implies_product_l2118_211846

theorem sqrt_sum_implies_product (x : ℝ) :
  (Real.sqrt (10 + x) + Real.sqrt (30 - x) = 8) →
  ((10 + x) * (30 - x) = 144) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_implies_product_l2118_211846


namespace NUMINAMATH_CALUDE_inequality_proof_l2118_211863

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2118_211863


namespace NUMINAMATH_CALUDE_sin_double_alpha_l2118_211881

theorem sin_double_alpha (α : Real) 
  (h : Real.cos (α - Real.pi/4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = -3/4 := by
sorry

end NUMINAMATH_CALUDE_sin_double_alpha_l2118_211881


namespace NUMINAMATH_CALUDE_wave_number_probability_l2118_211877

/-- A permutation of the digits 1,2,3,4,5 --/
def Permutation := Fin 5 → Fin 5

/-- A permutation is valid if it's bijective --/
def is_valid_permutation (p : Permutation) : Prop :=
  Function.Bijective p

/-- A permutation represents a wave number if it satisfies the wave pattern --/
def is_wave_number (p : Permutation) : Prop :=
  p 0 < p 1 ∧ p 1 > p 2 ∧ p 2 < p 3 ∧ p 3 > p 4

/-- The total number of valid permutations --/
def total_permutations : ℕ := 120

/-- The number of wave numbers --/
def wave_numbers : ℕ := 16

/-- The main theorem: probability of selecting a wave number --/
theorem wave_number_probability :
  (wave_numbers : ℚ) / total_permutations = 2 / 15 := by sorry

end NUMINAMATH_CALUDE_wave_number_probability_l2118_211877


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2118_211862

theorem arithmetic_sequence_sum_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, S n = (n / 2) * (a 1 + a n)) →  -- sum formula
  (a 8 / a 7 = 13 / 5) →                -- given condition
  (S 15 / S 13 = 3) :=                  -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_ratio_l2118_211862


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l2118_211870

-- Define the triangle vertices
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (8, 4)
def C : ℝ × ℝ := (-2, 4)

-- Define vectors
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the line equation 4x + 3y + m = 0
def line_equation (x y m : ℝ) : Prop := 4 * x + 3 * y + m = 0

-- Define the circumcircle equation (x-3)^2 + (y-4)^2 = 25
def circumcircle_equation (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 25

-- Define the chord length
def chord_length : ℝ := 6

theorem triangle_abc_properties :
  (dot_product AB AC = 0) ∧ 
  (∃ m : ℝ, (m = -4 ∨ m = -44) ∧
    ∃ x y : ℝ, line_equation x y m ∧ circumcircle_equation x y) :=
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l2118_211870


namespace NUMINAMATH_CALUDE_expression_evaluation_l2118_211895

theorem expression_evaluation :
  ∀ x : ℕ, 
    x - 3 < 0 →
    x - 1 ≠ 0 →
    x - 2 ≠ 0 →
    (3 / (x - 1) - x - 1) / ((x - 2) / (x^2 - 2*x + 1)) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2118_211895


namespace NUMINAMATH_CALUDE_sum_70_is_negative_350_l2118_211822

/-- An arithmetic progression with specified properties -/
structure ArithmeticProgression where
  /-- First term of the progression -/
  a : ℚ
  /-- Common difference of the progression -/
  d : ℚ
  /-- Sum of first 20 terms is 200 -/
  sum_20 : (20 : ℚ) / 2 * (2 * a + (20 - 1) * d) = 200
  /-- Sum of first 50 terms is 50 -/
  sum_50 : (50 : ℚ) / 2 * (2 * a + (50 - 1) * d) = 50

/-- The sum of the first 70 terms of the arithmetic progression is -350 -/
theorem sum_70_is_negative_350 (ap : ArithmeticProgression) :
  (70 : ℚ) / 2 * (2 * ap.a + (70 - 1) * ap.d) = -350 := by
  sorry

end NUMINAMATH_CALUDE_sum_70_is_negative_350_l2118_211822
