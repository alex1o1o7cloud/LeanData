import Mathlib

namespace NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l801_80109

-- Define a line in 2D space
structure Line2D where
  slope : ℝ
  intercept : ℝ

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Function to check if a line passes through a point
def linePassesThroughPoint (l : Line2D) (p : Point2D) : Prop :=
  p.y = l.slope * p.x + l.intercept

-- Function to check if a line has equal intercepts on both axes
def hasEqualIntercepts (l : Line2D) : Prop :=
  l.intercept / l.slope = -l.intercept

-- Theorem statement
theorem line_through_point_with_equal_intercepts :
  ∀ (l : Line2D),
    linePassesThroughPoint l { x := 1, y := 2 } →
    hasEqualIntercepts l →
    (l.slope = -1 ∧ l.intercept = 3) ∨ (l.slope = 2 ∧ l.intercept = 0) :=
sorry

end NUMINAMATH_CALUDE_line_through_point_with_equal_intercepts_l801_80109


namespace NUMINAMATH_CALUDE_tub_fill_time_is_24_minutes_l801_80177

/-- Represents the tub filling problem -/
structure TubFilling where
  capacity : ℕ             -- Tub capacity in liters
  flow_rate : ℕ            -- Tap flow rate in liters per minute
  leak_rate : ℕ            -- Leak rate in liters per minute
  cycle_time : ℕ           -- Time for one on-off cycle in minutes

/-- Calculates the time needed to fill the tub -/
def fill_time (tf : TubFilling) : ℕ :=
  let net_gain_per_cycle := tf.flow_rate - tf.leak_rate * tf.cycle_time
  (tf.capacity + net_gain_per_cycle - 1) / net_gain_per_cycle * tf.cycle_time

/-- Theorem stating that the time to fill the tub is 24 minutes -/
theorem tub_fill_time_is_24_minutes :
  let tf : TubFilling := {
    capacity := 120,
    flow_rate := 12,
    leak_rate := 1,
    cycle_time := 2
  }
  fill_time tf = 24 := by sorry

end NUMINAMATH_CALUDE_tub_fill_time_is_24_minutes_l801_80177


namespace NUMINAMATH_CALUDE_initial_storks_count_l801_80129

theorem initial_storks_count (initial_birds : ℕ) (additional_birds : ℕ) (final_difference : ℕ) :
  initial_birds = 2 →
  additional_birds = 3 →
  final_difference = 1 →
  initial_birds + additional_birds + final_difference = 6 :=
by sorry

end NUMINAMATH_CALUDE_initial_storks_count_l801_80129


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_half_l801_80144

open Real

theorem trigonometric_expression_equals_half : 
  (cos (85 * π / 180) + sin (25 * π / 180) * cos (30 * π / 180)) / cos (25 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_half_l801_80144


namespace NUMINAMATH_CALUDE_fixed_point_theorem_l801_80135

/-- The line equation as a function of k, x, and y -/
def line_equation (k x y : ℝ) : ℝ := (2*k + 1)*x + (1 - k)*y + 7 - k

/-- The theorem stating that (-2, -5) is a fixed point of the line for all real k -/
theorem fixed_point_theorem :
  ∀ k : ℝ, line_equation k (-2) (-5) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_l801_80135


namespace NUMINAMATH_CALUDE_cake_portion_theorem_l801_80110

theorem cake_portion_theorem (tom_ate jenny_took : ℚ) : 
  tom_ate = 60 / 100 →
  jenny_took = 1 / 4 →
  (1 - tom_ate) * (1 - jenny_took) = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_cake_portion_theorem_l801_80110


namespace NUMINAMATH_CALUDE_option_a_false_option_b_true_option_c_false_option_d_true_l801_80163

-- Option A
theorem option_a_false (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ¬(1 / a > 1 / b) := by
sorry

-- Option B
theorem option_b_true (a b : ℝ) (h : a < b) (h2 : b < 0) : 
  a^2 > a * b := by
sorry

-- Option C
theorem option_c_false : 
  ∃ a b : ℝ, a > b ∧ ¬(|a| > |b|) := by
sorry

-- Option D
theorem option_d_true (a : ℝ) (h : a > 2) : 
  a + 4 / (a - 2) ≥ 6 := by
sorry

-- Final answer
def correct_options : List Char := ['B', 'D']

end NUMINAMATH_CALUDE_option_a_false_option_b_true_option_c_false_option_d_true_l801_80163


namespace NUMINAMATH_CALUDE_masons_grandmother_age_l801_80198

theorem masons_grandmother_age (mason_age sydney_age father_age grandmother_age : ℕ) :
  mason_age = 20 →
  sydney_age = 3 * mason_age →
  father_age = sydney_age + 6 →
  grandmother_age = 2 * father_age →
  grandmother_age = 132 := by
sorry

end NUMINAMATH_CALUDE_masons_grandmother_age_l801_80198


namespace NUMINAMATH_CALUDE_number_of_girls_in_school_l801_80178

/-- Proves the number of girls in a school with given conditions -/
theorem number_of_girls_in_school (total : ℕ) (girls : ℕ) (boys : ℕ) 
  (h_total : total = 300)
  (h_ratio : girls * 8 = boys * 5)
  (h_sum : girls + boys = total) : 
  girls = 116 := by sorry

end NUMINAMATH_CALUDE_number_of_girls_in_school_l801_80178


namespace NUMINAMATH_CALUDE_expand_binomials_l801_80153

theorem expand_binomials (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomials_l801_80153


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l801_80133

theorem arithmetic_calculation : 4 * 11 + 5 * 12 + 13 * 4 + 4 * 10 = 196 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l801_80133


namespace NUMINAMATH_CALUDE_stating_five_min_commercials_count_l801_80112

/-- Represents the duration of the commercial break in minutes -/
def total_time : ℕ := 37

/-- Represents the number of 2-minute commercials -/
def two_min_commercials : ℕ := 11

/-- Represents the duration of a short commercial in minutes -/
def short_commercial_duration : ℕ := 2

/-- Represents the duration of a long commercial in minutes -/
def long_commercial_duration : ℕ := 5

/-- 
Theorem stating that given the total time and number of 2-minute commercials,
the number of 5-minute commercials is 3
-/
theorem five_min_commercials_count : 
  ∃ (x : ℕ), x * long_commercial_duration + two_min_commercials * short_commercial_duration = total_time ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_stating_five_min_commercials_count_l801_80112


namespace NUMINAMATH_CALUDE_correct_scaling_l801_80139

/-- A cookie recipe with ingredients and scaling -/
structure CookieRecipe where
  originalCookies : ℕ
  originalFlour : ℚ
  originalSugar : ℚ
  desiredCookies : ℕ

/-- Calculate the required ingredients for a scaled cookie recipe -/
def scaleRecipe (recipe : CookieRecipe) : ℚ × ℚ :=
  let scaleFactor : ℚ := recipe.desiredCookies / recipe.originalCookies
  (recipe.originalFlour * scaleFactor, recipe.originalSugar * scaleFactor)

/-- Theorem: Scaling the recipe correctly produces the expected amounts of flour and sugar -/
theorem correct_scaling (recipe : CookieRecipe) 
    (h1 : recipe.originalCookies = 24)
    (h2 : recipe.originalFlour = 3/2)
    (h3 : recipe.originalSugar = 1/2)
    (h4 : recipe.desiredCookies = 120) :
    scaleRecipe recipe = (15/2, 5/2) := by
  sorry

#eval scaleRecipe { originalCookies := 24, originalFlour := 3/2, originalSugar := 1/2, desiredCookies := 120 }

end NUMINAMATH_CALUDE_correct_scaling_l801_80139


namespace NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l801_80132

theorem largest_whole_number_satisfying_inequality :
  ∀ x : ℕ, x ≤ 15 ↔ 9 * x - 8 < 130 :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_satisfying_inequality_l801_80132


namespace NUMINAMATH_CALUDE_inverse_function_problem_l801_80151

/-- Given a function g(x) = 4x - 6 and its relation to the inverse of f(x) = ax + b,
    prove that 4a + 3b = 4 -/
theorem inverse_function_problem (a b : ℝ) :
  (∀ x, (4 * x - 6 : ℝ) = (Function.invFun (fun x => a * x + b) x) - 2) →
  4 * a + 3 * b = 4 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_problem_l801_80151


namespace NUMINAMATH_CALUDE_fifth_term_of_sequence_l801_80176

/-- A geometric sequence with the given first four terms -/
def geometric_sequence (y : ℝ) : ℕ → ℝ
  | 0 => 3
  | 1 => 9 * y
  | 2 => 27 * y^2
  | 3 => 81 * y^3
  | n + 4 => geometric_sequence y n * 3 * y

/-- The fifth term of the geometric sequence is 243y^4 -/
theorem fifth_term_of_sequence (y : ℝ) :
  geometric_sequence y 4 = 243 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_sequence_l801_80176


namespace NUMINAMATH_CALUDE_nabla_ratio_equals_eight_l801_80101

-- Define the ∇ operation for positive integers m < n
def nabla (m n : ℕ) (h1 : 0 < m) (h2 : m < n) : ℕ :=
  (n - m + 1) * (m + n) / 2

-- Theorem statement
theorem nabla_ratio_equals_eight :
  nabla 22 26 (by norm_num) (by norm_num) / nabla 4 6 (by norm_num) (by norm_num) = 8 := by
  sorry

end NUMINAMATH_CALUDE_nabla_ratio_equals_eight_l801_80101


namespace NUMINAMATH_CALUDE_M_intersect_N_equals_one_two_open_l801_80120

def M : Set ℝ := {x | ∃ y, y = Real.log (-x^2 - x + 6)}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem M_intersect_N_equals_one_two_open :
  M ∩ N = {x | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_equals_one_two_open_l801_80120


namespace NUMINAMATH_CALUDE_winning_strategy_l801_80105

/-- Represents the player who has a winning strategy -/
inductive WinningPlayer
  | First
  | Second

/-- Defines the game on a grid board -/
def gridGame (m n : ℕ) : WinningPlayer :=
  if m % 2 = 0 ∧ n % 2 = 0 then
    WinningPlayer.Second
  else if m % 2 = 1 ∧ n % 2 = 1 then
    WinningPlayer.Second
  else
    WinningPlayer.First

/-- Theorem stating the winning strategy for different board sizes -/
theorem winning_strategy :
  (gridGame 10 12 = WinningPlayer.Second) ∧
  (gridGame 9 10 = WinningPlayer.First) ∧
  (gridGame 9 11 = WinningPlayer.Second) :=
sorry

end NUMINAMATH_CALUDE_winning_strategy_l801_80105


namespace NUMINAMATH_CALUDE_horner_method_a1_l801_80157

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 0.5x^5 + 4x^4 - 3x^2 + x - 1 -/
def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

/-- Coefficients of the polynomial in reverse order -/
def coeffs : List ℝ := [-1, 1, 0, -3, 4, 0.5]

theorem horner_method_a1 : 
  let x := 3
  let result := horner_eval coeffs x
  result = f x ∧ result = 1 := by
  sorry

#eval horner_eval coeffs 3

end NUMINAMATH_CALUDE_horner_method_a1_l801_80157


namespace NUMINAMATH_CALUDE_parabola_properties_l801_80155

-- Define the parabola and its properties
def parabola (a b c m n t x₀ : ℝ) : Prop :=
  a > 0 ∧
  m = a + b + c ∧
  n = 16*a + 4*b + c ∧
  t = -b / (2*a) ∧
  3*a + b = 0 ∧
  m < c ∧ c < n ∧
  x₀ ≠ 1 ∧
  m = a * x₀^2 + b * x₀ + c

-- State the theorem
theorem parabola_properties (a b c m n t x₀ : ℝ) 
  (h : parabola a b c m n t x₀) : 
  m < n ∧ 1/2 < t ∧ t < 2 ∧ 0 < x₀ ∧ x₀ < 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l801_80155


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_condition_l801_80187

theorem greatest_integer_with_gcd_condition :
  ∃ n : ℕ, n < 200 ∧ n.gcd 30 = 10 ∧ ∀ m : ℕ, m < 200 → m.gcd 30 = 10 → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_condition_l801_80187


namespace NUMINAMATH_CALUDE_weight_equivalence_l801_80131

-- Define the weights of shapes as real numbers
variable (triangle circle square : ℝ)

-- Define the conditions from the problem
axiom weight_relation1 : 5 * triangle = 3 * circle
axiom weight_relation2 : circle = triangle + 2 * square

-- Theorem to prove
theorem weight_equivalence : triangle + circle = 3 * square := by
  sorry

end NUMINAMATH_CALUDE_weight_equivalence_l801_80131


namespace NUMINAMATH_CALUDE_polynomial_identity_l801_80146

theorem polynomial_identity (x : ℝ) : 
  (x - 2)^4 + 5*(x - 2)^3 + 10*(x - 2)^2 + 10*(x - 2) + 5 = (x - 2 + Real.sqrt 2)^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l801_80146


namespace NUMINAMATH_CALUDE_odometer_puzzle_l801_80193

theorem odometer_puzzle (a b c d : ℕ) : 
  a ≥ 1 →
  a + b + c + d ≤ 10 →
  (1000 * d + 100 * c + 10 * b + a) - (1000 * a + 100 * b + 10 * c + d) % 60 = 0 →
  a^2 + b^2 + c^2 + d^2 = 83 := by
  sorry

end NUMINAMATH_CALUDE_odometer_puzzle_l801_80193


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_cosine_l801_80107

open Real

theorem min_max_abs_quadratic_cosine :
  (∃ y₀ : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → |x^2 + x*y₀ + cos y₀| ≤ 2)) ∧
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ |x^2 + x*y + cos y| ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_cosine_l801_80107


namespace NUMINAMATH_CALUDE_find_t_l801_80158

-- Define vectors in R²
def AB : Fin 2 → ℝ := ![2, 3]
def AC : ℝ → Fin 2 → ℝ := λ t => ![3, t]

-- Define the dot product of two vectors in R²
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define the perpendicular condition
def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  dot_product v w = 0

-- Theorem statement
theorem find_t : ∃ t : ℝ, 
  perpendicular AB (λ i => AC t i - AB i) ∧ t = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_find_t_l801_80158


namespace NUMINAMATH_CALUDE_solve_equation_l801_80111

theorem solve_equation (z : ℝ) :
  ∃ (n : ℝ), 14 * (-1 + z) + 18 = -14 * (1 - z) - n :=
by
  use -4
  sorry

#check solve_equation

end NUMINAMATH_CALUDE_solve_equation_l801_80111


namespace NUMINAMATH_CALUDE_f_is_quadratic_l801_80156

/-- Definition of a quadratic function -/
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function we want to prove is quadratic -/
def f (x : ℝ) : ℝ := x^2 + 1

/-- Theorem stating that f is a quadratic function -/
theorem f_is_quadratic : is_quadratic f := by
  sorry

end NUMINAMATH_CALUDE_f_is_quadratic_l801_80156


namespace NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l801_80188

theorem smallest_k_with_remainder_one (k : ℕ) : k = 103 ↔ 
  (k > 1) ∧ 
  (∀ n ∈ ({17, 6, 2} : Set ℕ), k % n = 1) ∧
  (∀ m : ℕ, m > 1 → (∀ n ∈ ({17, 6, 2} : Set ℕ), m % n = 1) → m ≥ k) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_with_remainder_one_l801_80188


namespace NUMINAMATH_CALUDE_shelby_heavy_rain_time_l801_80169

/-- Represents the speeds and durations of Shelby's scooter ride --/
structure ScooterRide where
  sunnySpeed : ℝ
  lightRainSpeed : ℝ
  heavyRainSpeed : ℝ
  totalDistance : ℝ
  totalTime : ℝ
  heavyRainTime : ℝ

/-- Theorem stating that given the conditions of Shelby's ride, she spent 20 minutes in heavy rain --/
theorem shelby_heavy_rain_time (ride : ScooterRide) 
  (h1 : ride.sunnySpeed = 35)
  (h2 : ride.lightRainSpeed = 25)
  (h3 : ride.heavyRainSpeed = 15)
  (h4 : ride.totalDistance = 50)
  (h5 : ride.totalTime = 100) :
  ride.heavyRainTime = 20 := by
  sorry

#check shelby_heavy_rain_time

end NUMINAMATH_CALUDE_shelby_heavy_rain_time_l801_80169


namespace NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l801_80168

theorem min_value_sqrt_plus_reciprocal (x : ℝ) (h : x > 0) :
  3 * Real.sqrt x + 2 / x ≥ 5 ∧
  ∃ y > 0, 3 * Real.sqrt y + 2 / y = 5 :=
sorry

end NUMINAMATH_CALUDE_min_value_sqrt_plus_reciprocal_l801_80168


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l801_80161

theorem gcd_of_three_numbers : Nat.gcd 13680 (Nat.gcd 20400 47600) = 80 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l801_80161


namespace NUMINAMATH_CALUDE_expression_value_l801_80117

theorem expression_value : 
  let x : ℝ := 3
  5 * 7 + 9 * 4 - 35 / 5 + x * 2 = 70 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l801_80117


namespace NUMINAMATH_CALUDE_base_4_divisibility_l801_80122

def base_4_to_decimal (a b c d : ℕ) : ℕ :=
  a * 4^3 + b * 4^2 + c * 4 + d

def is_divisible_by_13 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 13 * k

theorem base_4_divisibility :
  ∀ x : ℕ, x < 4 →
    is_divisible_by_13 (base_4_to_decimal 2 3 1 x) ↔ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_4_divisibility_l801_80122


namespace NUMINAMATH_CALUDE_fraction_equality_solution_l801_80167

theorem fraction_equality_solution :
  let f₁ (x : ℝ) := (5 + 2*x) / (7 + 3*x)
  let f₂ (x : ℝ) := (4 + 3*x) / (9 + 4*x)
  let x₁ := (-5 + Real.sqrt 93) / 2
  let x₂ := (-5 - Real.sqrt 93) / 2
  (f₁ x₁ = f₂ x₁) ∧ (f₁ x₂ = f₂ x₂) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_solution_l801_80167


namespace NUMINAMATH_CALUDE_triangle_ratio_theorem_l801_80185

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the area of a triangle
def area (t : Triangle) : ℝ := sorry

-- Define the length of a side
def sideLength (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two sides
def angle (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Define a point on a line segment
def pointOnSegment (p1 p2 : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := sorry

-- Define the intersection of two lines
def lineIntersection (l1p1 l1p2 l2p1 l2p2 : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the median of a triangle
def median (t : Triangle) (v : ℝ × ℝ) : ℝ × ℝ := sorry

theorem triangle_ratio_theorem (t : Triangle) :
  area t = 2 * Real.sqrt 3 →
  sideLength t.B t.C = 1 →
  angle t.B t.C t.A = π / 3 →
  let D := pointOnSegment t.A t.B 3
  let E := median t t.C
  let M := lineIntersection t.C D t.B E
  ∃ (r : ℝ), r = 3 / 5 ∧ sideLength t.B M = r * sideLength M E :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_theorem_l801_80185


namespace NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l801_80196

theorem integer_solutions_yk_eq_x2_plus_x (k : ℕ) (hk : k > 1) :
  (∃ (x y : ℤ), y^k = x^2 + x) ↔ (k = 2 ∧ (∃ x : ℤ, x = 0 ∨ x = -1)) :=
sorry

end NUMINAMATH_CALUDE_integer_solutions_yk_eq_x2_plus_x_l801_80196


namespace NUMINAMATH_CALUDE_probability_k_standard_parts_formula_l801_80104

/-- The probability of selecting exactly k standard parts when randomly choosing m parts from a batch of N parts containing n standard parts. -/
def probability_k_standard_parts (N n m k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.choose (N - n) (m - k)) / Nat.choose N m

/-- Theorem stating that the probability of selecting exactly k standard parts
    when randomly choosing m parts from a batch of N parts containing n standard parts
    is equal to (C_n^k * C_(N-n)^(m-k)) / C_N^m. -/
theorem probability_k_standard_parts_formula
  (N n m k : ℕ)
  (h1 : n ≤ N)
  (h2 : m ≤ N)
  (h3 : k ≤ m)
  (h4 : k ≤ n) :
  probability_k_standard_parts N n m k =
    (Nat.choose n k * Nat.choose (N - n) (m - k)) / Nat.choose N m :=
by
  sorry

#check probability_k_standard_parts_formula

end NUMINAMATH_CALUDE_probability_k_standard_parts_formula_l801_80104


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l801_80181

/-- A quadratic equation (a-1)x^2 - 2x + 1 = 0 has two distinct real roots if and only if a < 2 and a ≠ 1 -/
theorem quadratic_two_distinct_roots (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔ 
  (a < 2 ∧ a ≠ 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l801_80181


namespace NUMINAMATH_CALUDE_base7_multiplication_addition_l801_80154

/-- Converts a base 7 number represented as a list of digits to a natural number -/
def base7ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => 7 * acc + d) 0

/-- Converts a natural number to its base 7 representation as a list of digits -/
def natToBase7 (n : Nat) : List Nat :=
  if n < 7 then [n]
  else (n % 7) :: natToBase7 (n / 7)

theorem base7_multiplication_addition :
  (base7ToNat [5, 2]) * (base7ToNat [3]) + (base7ToNat [4, 4, 1]) =
  base7ToNat [3, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_base7_multiplication_addition_l801_80154


namespace NUMINAMATH_CALUDE_defective_shipped_percentage_l801_80186

theorem defective_shipped_percentage 
  (total_units : ℕ) 
  (defective_percentage : ℝ) 
  (shipped_percentage : ℝ) 
  (h1 : defective_percentage = 7) 
  (h2 : shipped_percentage = 5) : 
  (defective_percentage / 100) * (shipped_percentage / 100) * 100 = 0.35 := by
  sorry

end NUMINAMATH_CALUDE_defective_shipped_percentage_l801_80186


namespace NUMINAMATH_CALUDE_tv_discounted_price_l801_80170

def original_price : ℝ := 500.00
def discount1 : ℝ := 0.10
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.15

def final_price : ℝ := original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)

theorem tv_discounted_price : final_price = 306.00 := by
  sorry

end NUMINAMATH_CALUDE_tv_discounted_price_l801_80170


namespace NUMINAMATH_CALUDE_pond_draining_time_l801_80119

theorem pond_draining_time 
  (pump1_half_time : ℝ) 
  (pump2_full_time : ℝ) 
  (combined_half_time : ℝ) 
  (h1 : pump2_full_time = 1.25) 
  (h2 : combined_half_time = 0.5) :
  pump1_half_time = 5/12 := by
sorry

end NUMINAMATH_CALUDE_pond_draining_time_l801_80119


namespace NUMINAMATH_CALUDE_remaining_fence_is_48_feet_l801_80130

/-- The length of fence remaining to be whitewashed after three people have worked on it. -/
def remaining_fence (total_length : ℝ) (first_length : ℝ) (second_fraction : ℝ) (third_fraction : ℝ) : ℝ :=
  let remaining_after_first := total_length - first_length
  let remaining_after_second := remaining_after_first * (1 - second_fraction)
  remaining_after_second * (1 - third_fraction)

/-- Theorem stating that the remaining fence to be whitewashed is 48 feet. -/
theorem remaining_fence_is_48_feet :
  remaining_fence 100 10 (1/5) (1/3) = 48 := by
  sorry

end NUMINAMATH_CALUDE_remaining_fence_is_48_feet_l801_80130


namespace NUMINAMATH_CALUDE_inverse_proportion_l801_80191

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 15, 
    then x = -5/2 when y = -30 -/
theorem inverse_proportion (x y : ℝ) (h : ∃ k : ℝ, ∀ x y, x * y = k) 
    (h1 : 5 * 15 = x * y) : 
  x * (-30) = 5 * 15 → x = -5/2 := by sorry

end NUMINAMATH_CALUDE_inverse_proportion_l801_80191


namespace NUMINAMATH_CALUDE_real_y_condition_l801_80141

theorem real_y_condition (x y : ℝ) : 
  (4 * y^2 + 2 * x * y + |x| + 8 = 0) → 
  (∃ (y : ℝ), 4 * y^2 + 2 * x * y + |x| + 8 = 0) ↔ (x ≤ -10 ∨ x ≥ 10) := by
  sorry

end NUMINAMATH_CALUDE_real_y_condition_l801_80141


namespace NUMINAMATH_CALUDE_tom_payment_l801_80103

/-- The total amount Tom paid to the shopkeeper for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Proof that Tom paid 1190 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 70 = 1190 := by
  sorry

end NUMINAMATH_CALUDE_tom_payment_l801_80103


namespace NUMINAMATH_CALUDE_angle_measure_l801_80115

theorem angle_measure : ∃ (x : ℝ), 
  (180 - x = 7 * (90 - x)) ∧ 
  (0 < x) ∧ 
  (x < 180) ∧ 
  (x = 75) := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_l801_80115


namespace NUMINAMATH_CALUDE_mike_picked_limes_l801_80162

/-- The number of limes Alyssa ate -/
def limes_eaten : ℝ := 25.0

/-- The number of limes left -/
def limes_left : ℕ := 7

/-- The number of limes Mike picked -/
def mikes_limes : ℝ := limes_eaten + limes_left

theorem mike_picked_limes : mikes_limes = 32 := by
  sorry

end NUMINAMATH_CALUDE_mike_picked_limes_l801_80162


namespace NUMINAMATH_CALUDE_clothing_tax_rate_l801_80145

theorem clothing_tax_rate
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (other_tax_rate : ℝ)
  (total_tax_rate : ℝ)
  (h1 : clothing_percent = 0.5)
  (h2 : food_percent = 0.2)
  (h3 : other_percent = 0.3)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : other_tax_rate = 0.1)
  (h6 : total_tax_rate = 0.055) :
  ∃ (clothing_tax_rate : ℝ),
    clothing_tax_rate * clothing_percent + other_tax_rate * other_percent = total_tax_rate ∧
    clothing_tax_rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_clothing_tax_rate_l801_80145


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l801_80121

theorem hyperbola_eccentricity_range (a : ℝ) (h : a > 1) :
  let e := Real.sqrt (1 + 1 / a^2)
  1 < e ∧ e < Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l801_80121


namespace NUMINAMATH_CALUDE_function_is_linear_l801_80116

-- Define the function f and constants a and b
variable (f : ℝ → ℝ) (a b : ℝ)

-- State the theorem
theorem function_is_linear
  (h_continuous : Continuous f)
  (h_a : 0 < a ∧ a < 1/2)
  (h_b : 0 < b ∧ b < 1/2)
  (h_functional : ∀ x, f (f x) = a * f x + b * x) :
  ∃ k : ℝ, ∀ x, f x = k * x :=
by sorry

end NUMINAMATH_CALUDE_function_is_linear_l801_80116


namespace NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l801_80123

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def units_digit (n : ℕ) : ℕ := n % 10

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10

theorem unique_three_digit_divisible_by_11 :
  ∃! n : ℕ, is_three_digit n ∧ 
             units_digit n = 5 ∧ 
             hundreds_digit n = 6 ∧ 
             n % 11 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_three_digit_divisible_by_11_l801_80123


namespace NUMINAMATH_CALUDE_profit_range_l801_80134

/-- Price function for books -/
def C (n : ℕ) : ℕ :=
  if n ≤ 24 then 12 * n
  else if n ≤ 48 then 11 * n
  else 10 * n

/-- Total number of books -/
def total_books : ℕ := 60

/-- Cost per book to the company -/
def cost_per_book : ℕ := 5

/-- Profit function given two people buying books -/
def profit (a b : ℕ) : ℤ :=
  (C a + C b) - (cost_per_book * total_books)

/-- Theorem stating the range of profit -/
theorem profit_range :
  ∀ a b : ℕ,
  a + b = total_books →
  a ≥ 1 →
  b ≥ 1 →
  302 ≤ profit a b ∧ profit a b ≤ 384 :=
sorry

end NUMINAMATH_CALUDE_profit_range_l801_80134


namespace NUMINAMATH_CALUDE_base_subtraction_equality_l801_80108

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b ^ i) 0

/-- The problem statement -/
theorem base_subtraction_equality : 
  let base_6_num := [5, 2, 3]  -- 325 in base 6 (least significant digit first)
  let base_5_num := [1, 3, 2]  -- 231 in base 5 (least significant digit first)
  (to_base_10 base_6_num 6) - (to_base_10 base_5_num 5) = 59 := by
  sorry

end NUMINAMATH_CALUDE_base_subtraction_equality_l801_80108


namespace NUMINAMATH_CALUDE_square_difference_divided_by_three_l801_80102

theorem square_difference_divided_by_three : (121^2 - 112^2) / 3 = 699 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_by_three_l801_80102


namespace NUMINAMATH_CALUDE_count_integers_between_square_roots_l801_80124

theorem count_integers_between_square_roots : 
  (Finset.range 25 \ Finset.range 10).card = 15 := by sorry

end NUMINAMATH_CALUDE_count_integers_between_square_roots_l801_80124


namespace NUMINAMATH_CALUDE_remainder_equality_l801_80128

theorem remainder_equality (P P' K D R R' : ℕ) (r r' : ℕ) 
  (h1 : P > P') 
  (h2 : K ∣ P) 
  (h3 : K ∣ P') 
  (h4 : P % D = R) 
  (h5 : P' % D = R') 
  (h6 : (P * K - P') % D = r) 
  (h7 : (R * K - R') % D = r') : 
  r = r' := by
sorry

end NUMINAMATH_CALUDE_remainder_equality_l801_80128


namespace NUMINAMATH_CALUDE_max_missed_problems_l801_80171

/-- Given a test with 50 problems and a passing score of at least 85%,
    the maximum number of problems a student can miss and still pass is 7. -/
theorem max_missed_problems (total_problems : Nat) (passing_percentage : Rat) :
  total_problems = 50 →
  passing_percentage = 85 / 100 →
  (↑(total_problems - 7) : Rat) / total_problems ≥ passing_percentage ∧
  ∀ n : Nat, n > 7 → (↑(total_problems - n) : Rat) / total_problems < passing_percentage :=
by sorry

end NUMINAMATH_CALUDE_max_missed_problems_l801_80171


namespace NUMINAMATH_CALUDE_only_pi_smaller_than_neg_three_l801_80183

theorem only_pi_smaller_than_neg_three : 
  (-Real.sqrt 2 > -3) ∧ (1 > -3) ∧ (0 > -3) ∧ (-Real.pi < -3) := by
  sorry

end NUMINAMATH_CALUDE_only_pi_smaller_than_neg_three_l801_80183


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l801_80137

theorem initial_mean_calculation (n : ℕ) (initial_mean corrected_mean : ℚ) : 
  n = 50 → 
  corrected_mean = 36.02 → 
  n * initial_mean + 1 = n * corrected_mean → 
  initial_mean = 36 := by
sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l801_80137


namespace NUMINAMATH_CALUDE_total_messages_l801_80140

def messages_last_week : ℕ := 111

def messages_this_week : ℕ := 2 * messages_last_week - 50

theorem total_messages : messages_last_week + messages_this_week = 283 := by
  sorry

end NUMINAMATH_CALUDE_total_messages_l801_80140


namespace NUMINAMATH_CALUDE_perpendicular_vectors_condition_l801_80150

/-- Given two vectors m and n in ℝ², if m is perpendicular to n,
    then the second component of n is -2 times the first component of m. -/
theorem perpendicular_vectors_condition (m n : ℝ × ℝ) :
  m = (1, 2) →
  n.1 = a →
  n.2 = -1 →
  m.1 * n.1 + m.2 * n.2 = 0 →
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_condition_l801_80150


namespace NUMINAMATH_CALUDE_problem_statement_l801_80172

theorem problem_statement (a b : ℝ) (h : 4 * a^2 - a * b + b^2 = 1) :
  (abs a ≤ 2 * Real.sqrt 15 / 15) ∧
  (4 / 5 ≤ 4 * a^2 + b^2 ∧ 4 * a^2 + b^2 ≤ 4 / 3) ∧
  (abs (2 * a - b) ≤ 2 * Real.sqrt 10 / 5) := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l801_80172


namespace NUMINAMATH_CALUDE_ursula_change_l801_80184

/-- Calculates the change Ursula received after buying hot dogs and salads -/
theorem ursula_change (hot_dog_price : ℚ) (salad_price : ℚ) 
  (num_hot_dogs : ℕ) (num_salads : ℕ) (bill_value : ℚ) (num_bills : ℕ) :
  hot_dog_price = 3/2 →
  salad_price = 5/2 →
  num_hot_dogs = 5 →
  num_salads = 3 →
  bill_value = 10 →
  num_bills = 2 →
  (num_bills * bill_value) - (num_hot_dogs * hot_dog_price + num_salads * salad_price) = 5 :=
by sorry

end NUMINAMATH_CALUDE_ursula_change_l801_80184


namespace NUMINAMATH_CALUDE_eighteen_binary_l801_80180

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 2) ((m % 2) :: acc)
    aux n []

theorem eighteen_binary : decimal_to_binary 18 = [1, 0, 0, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_eighteen_binary_l801_80180


namespace NUMINAMATH_CALUDE_area_increase_when_perimeter_increased_l801_80199

/-- Represents a rectangle with integer dimensions. -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- Calculates the area of a rectangle. -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Represents the set of possible area increases. -/
def possibleAreaIncreases : Set ℕ := {2, 4, 21, 36, 38}

/-- Theorem stating the possible area increases when the perimeter is increased by 4 cm. -/
theorem area_increase_when_perimeter_increased
  (r : Rectangle)
  (h_perimeter : perimeter r = 40)
  (h_area : area r ≤ 40)
  (r_new : Rectangle)
  (h_new_perimeter : perimeter r_new = 44)
  : (area r_new - area r) ∈ possibleAreaIncreases := by
  sorry

end NUMINAMATH_CALUDE_area_increase_when_perimeter_increased_l801_80199


namespace NUMINAMATH_CALUDE_ninas_pet_insect_eyes_l801_80194

/-- The number of eyes among Nina's pet insects -/
def total_eyes (num_spiders : ℕ) (spider_eyes : ℕ) (num_ants : ℕ) (ant_eyes : ℕ) : ℕ :=
  num_spiders * spider_eyes + num_ants * ant_eyes

/-- Theorem stating the total number of eyes among Nina's pet insects -/
theorem ninas_pet_insect_eyes :
  total_eyes 3 8 50 2 = 124 := by
  sorry

end NUMINAMATH_CALUDE_ninas_pet_insect_eyes_l801_80194


namespace NUMINAMATH_CALUDE_function_transformation_l801_80113

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_transformation (h : f 4 = 2) : 
  ∃ x y, x = 4 ∧ y = -2 ∧ -f x = y :=
sorry

end NUMINAMATH_CALUDE_function_transformation_l801_80113


namespace NUMINAMATH_CALUDE_coconut_grove_problem_l801_80182

theorem coconut_grove_problem (x : ℝ) : 
  (60 * (x + 1) + 120 * x + 180 * (x - 1)) / (3 * x) = 100 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_coconut_grove_problem_l801_80182


namespace NUMINAMATH_CALUDE_course_selection_schemes_l801_80192

/-- The number of elective courses in each category (physical education and art) -/
def n : ℕ := 4

/-- The total number of different course selection schemes -/
def total_schemes : ℕ := 
  (n.choose 1 * n.choose 1) +  -- Selecting 2 courses (1 from each category)
  (n.choose 2 * n.choose 1) +  -- Selecting 3 courses (2 PE, 1 Art)
  (n.choose 1 * n.choose 2)    -- Selecting 3 courses (1 PE, 2 Art)

theorem course_selection_schemes : total_schemes = 64 := by
  sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l801_80192


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l801_80126

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  -- a, b, c are sides opposite to angles A, B, C respectively
  (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
  -- Given conditions
  ((2 * Real.cos A - 1) * Real.sin B + 2 * Real.cos A = 1) ∧
  (5 * b^2 = a^2 + 2 * c^2) →
  -- Conclusions
  (A = π / 3) ∧
  (Real.sin B / Real.sin C = 3 / 4) := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l801_80126


namespace NUMINAMATH_CALUDE_last_four_digits_of_2_to_15000_l801_80173

theorem last_four_digits_of_2_to_15000 (h : 2^500 ≡ 1 [ZMOD 1250]) :
  2^15000 ≡ 1 [ZMOD 1250] := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_2_to_15000_l801_80173


namespace NUMINAMATH_CALUDE_event_arrangements_eq_60_l801_80197

/-- The number of ways to select 4 students from 5 for a three-day event --/
def event_arrangements (total_students : ℕ) (selected_students : ℕ) (days : ℕ) 
  (first_day_attendees : ℕ) : ℕ :=
  Nat.choose total_students first_day_attendees * 
  (Nat.factorial (total_students - first_day_attendees) / 
   Nat.factorial (total_students - selected_students))

/-- Proof that the number of arrangements for the given conditions is 60 --/
theorem event_arrangements_eq_60 : 
  event_arrangements 5 4 3 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_event_arrangements_eq_60_l801_80197


namespace NUMINAMATH_CALUDE_x_fourth_coefficient_is_20th_term_l801_80195

def binomial_sum (n : ℕ) : ℕ := (n.choose 4) + ((n + 1).choose 4) + ((n + 2).choose 4)

def arithmetic_sequence (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

theorem x_fourth_coefficient_is_20th_term :
  ∃ n : ℕ, n = 5 ∧ 
  binomial_sum n = arithmetic_sequence (-2) 3 20 := by
  sorry

end NUMINAMATH_CALUDE_x_fourth_coefficient_is_20th_term_l801_80195


namespace NUMINAMATH_CALUDE_duck_problem_solution_l801_80100

/-- Represents the duck population problem --/
def duck_problem (initial_flock : ℕ) (killed_per_year : ℕ) (born_per_year : ℕ) 
                 (other_flock : ℕ) (combined_flock : ℕ) : Prop :=
  ∃ y : ℕ, 
    initial_flock + (born_per_year - killed_per_year) * y + other_flock = combined_flock

/-- Theorem stating the solution to the duck population problem --/
theorem duck_problem_solution : 
  duck_problem 100 20 30 150 300 → 
  ∃ y : ℕ, y = 5 ∧ duck_problem 100 20 30 150 300 := by
  sorry

#check duck_problem_solution

end NUMINAMATH_CALUDE_duck_problem_solution_l801_80100


namespace NUMINAMATH_CALUDE_log_50_bounds_sum_l801_80127

theorem log_50_bounds_sum : ∃ c d : ℤ, (1 : ℝ) ≤ c ∧ (c : ℝ) < Real.log 50 / Real.log 10 ∧ Real.log 50 / Real.log 10 < d ∧ (d : ℝ) ≤ 2 ∧ c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_log_50_bounds_sum_l801_80127


namespace NUMINAMATH_CALUDE_integer_solutions_of_system_l801_80143

theorem integer_solutions_of_system : 
  ∀ x y z : ℤ, 
  x + y + z = 2 ∧ 
  x^3 + y^3 + z^3 = -10 → 
  ((x = 3 ∧ y = 3 ∧ z = -4) ∨ 
   (x = 3 ∧ y = -4 ∧ z = 3) ∨ 
   (x = -4 ∧ y = 3 ∧ z = 3)) := by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_of_system_l801_80143


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l801_80142

theorem largest_divisor_of_n (n : ℕ+) 
  (h1 : (n : ℕ)^4 % 850 = 0)
  (h2 : ∀ p : ℕ, p > 20 → Nat.Prime p → (n : ℕ) % p ≠ 0) :
  ∃ k : ℕ, k ∣ (n : ℕ) ∧ k = 10 ∧ ∀ m : ℕ, m ∣ (n : ℕ) → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l801_80142


namespace NUMINAMATH_CALUDE_total_chips_is_90_l801_80136

/-- The total number of chips Viviana and Susana have together -/
def total_chips (viviana_vanilla : ℕ) (susana_chocolate : ℕ) : ℕ :=
  let viviana_chocolate := susana_chocolate + 5
  let susana_vanilla := (3 * viviana_vanilla) / 4
  viviana_vanilla + viviana_chocolate + susana_vanilla + susana_chocolate

/-- Theorem stating that the total number of chips is 90 -/
theorem total_chips_is_90 :
  total_chips 20 25 = 90 := by
  sorry

#eval total_chips 20 25

end NUMINAMATH_CALUDE_total_chips_is_90_l801_80136


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l801_80164

theorem solution_set_reciprocal_inequality (x : ℝ) :
  (1 / x > 3) ↔ (0 < x ∧ x < 1/3) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l801_80164


namespace NUMINAMATH_CALUDE_equation_roots_relation_l801_80138

theorem equation_roots_relation (k : ℝ) : 
  (∀ x y : ℝ, x^2 + k*x + 12 = 0 → y^2 - k*y + 12 = 0 → y = x + 6) →
  k = 6 := by
sorry

end NUMINAMATH_CALUDE_equation_roots_relation_l801_80138


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l801_80175

theorem greatest_multiple_of_5_and_6_less_than_1000 :
  (∃ n : ℕ, n < 1000 ∧ 5 ∣ n ∧ 6 ∣ n) →
  (∀ m : ℕ, m < 1000 ∧ 5 ∣ m ∧ 6 ∣ m → m ≤ 990) ∧
  990 < 1000 ∧ 5 ∣ 990 ∧ 6 ∣ 990 :=
by sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l801_80175


namespace NUMINAMATH_CALUDE_solution_added_mass_l801_80118

/-- Represents the composition and manipulation of a solution --/
structure Solution :=
  (total_mass : ℝ)
  (liquid_x_percentage : ℝ)

/-- Calculates the mass of liquid x in a solution --/
def liquid_x_mass (s : Solution) : ℝ :=
  s.total_mass * s.liquid_x_percentage

/-- Represents the problem scenario --/
def solution_problem (initial_solution : Solution) 
  (evaporated_water : ℝ) (added_solution : Solution) : Prop :=
  let remaining_solution : Solution := {
    total_mass := initial_solution.total_mass - evaporated_water,
    liquid_x_percentage := 
      liquid_x_mass initial_solution / (initial_solution.total_mass - evaporated_water)
  }
  let final_solution : Solution := {
    total_mass := remaining_solution.total_mass + added_solution.total_mass,
    liquid_x_percentage := 0.4
  }
  liquid_x_mass remaining_solution + liquid_x_mass added_solution = 
    liquid_x_mass final_solution

/-- The theorem to be proved --/
theorem solution_added_mass : 
  let initial_solution : Solution := { total_mass := 6, liquid_x_percentage := 0.3 }
  let evaporated_water : ℝ := 2
  let added_solution : Solution := { total_mass := 2, liquid_x_percentage := 0.3 }
  solution_problem initial_solution evaporated_water added_solution := by
  sorry

end NUMINAMATH_CALUDE_solution_added_mass_l801_80118


namespace NUMINAMATH_CALUDE_inequality_and_equality_conditions_l801_80190

theorem inequality_and_equality_conditions (a b : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 1) :
  (1/2 ≤ (a^3 + b^3) / (a^2 + b^2)) ∧ 
  ((a^3 + b^3) / (a^2 + b^2) ≤ 1) ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1/2 ↔ a = 1/2 ∧ b = 1/2) ∧
  ((a^3 + b^3) / (a^2 + b^2) = 1 ↔ (a = 0 ∧ b = 1) ∨ (a = 1 ∧ b = 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_and_equality_conditions_l801_80190


namespace NUMINAMATH_CALUDE_different_color_number_probability_l801_80160

/-- Represents the total number of balls -/
def total_balls : ℕ := 9

/-- Represents the number of balls to be drawn -/
def drawn_balls : ℕ := 3

/-- Represents the number of colors -/
def num_colors : ℕ := 3

/-- Represents the number of balls per color -/
def balls_per_color : ℕ := 3

/-- Represents the number of possible numbers on each ball -/
def num_numbers : ℕ := 3

/-- The probability of drawing 3 balls with different colors and numbers -/
def probability_different : ℚ := 1 / 14

theorem different_color_number_probability :
  (Nat.factorial num_colors) / (Nat.choose total_balls drawn_balls) = probability_different :=
sorry

end NUMINAMATH_CALUDE_different_color_number_probability_l801_80160


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_of_5400_l801_80179

/-- The number of perfect square factors of 5400 -/
def perfect_square_factors_of_5400 : ℕ :=
  let n := 5400
  let prime_factorization := (2, 2) :: (3, 3) :: (5, 2) :: []
  (prime_factorization.map (fun (p, e) => (e / 2 + 1))).prod

/-- Theorem stating that the number of perfect square factors of 5400 is 8 -/
theorem count_perfect_square_factors_of_5400 :
  perfect_square_factors_of_5400 = 8 := by sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_of_5400_l801_80179


namespace NUMINAMATH_CALUDE_pies_from_36_apples_l801_80174

/-- Given that 3 pies can be made from 12 apples, this function calculates
    the number of pies that can be made from a given number of apples. -/
def pies_from_apples (apples : ℕ) : ℕ :=
  (apples * 3) / 12

theorem pies_from_36_apples :
  pies_from_apples 36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_pies_from_36_apples_l801_80174


namespace NUMINAMATH_CALUDE_number_relationship_l801_80149

theorem number_relationship (a b c d : ℝ) : 
  a = Real.log (3/2) / Real.log (2/3) →
  b = Real.log 2 / Real.log 3 →
  c = 2 ^ (1/3 : ℝ) →
  d = 3 ^ (1/2 : ℝ) →
  a < b ∧ b < c ∧ c < d := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l801_80149


namespace NUMINAMATH_CALUDE_least_four_divisors_sum_of_squares_l801_80147

theorem least_four_divisors_sum_of_squares (n : ℕ+) 
  (h1 : ∃ (d1 d2 d3 d4 : ℕ+), d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ 
    (∀ m : ℕ+, m ∣ n → m = d1 ∨ m = d2 ∨ m = d3 ∨ m = d4 ∨ m > d4))
  (h2 : ∃ (d1 d2 d3 d4 : ℕ+), d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ 
    n = d1^2 + d2^2 + d3^2 + d4^2) : 
  n = 130 := by
sorry

end NUMINAMATH_CALUDE_least_four_divisors_sum_of_squares_l801_80147


namespace NUMINAMATH_CALUDE_product_units_digit_base_6_l801_80166

/-- The units digit of a positive integer in base-6 is the remainder when the integer is divided by 6 -/
def units_digit_base_6 (n : ℕ) : ℕ := n % 6

/-- The product of the given numbers -/
def product : ℕ := 123 * 57 * 29

theorem product_units_digit_base_6 :
  units_digit_base_6 product = 3 := by sorry

end NUMINAMATH_CALUDE_product_units_digit_base_6_l801_80166


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l801_80152

/-- Given ratios between w, x, y, and z, prove the ratio of w to y -/
theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 2)
  (hy : y / w = 3 / 5)
  (hz : z / x = 1 / 3) :
  w / y = 5 / 3 := by sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l801_80152


namespace NUMINAMATH_CALUDE_randy_gave_sally_l801_80114

theorem randy_gave_sally (initial_amount : ℕ) (received_amount : ℕ) (kept_amount : ℕ) :
  initial_amount = 3000 →
  received_amount = 200 →
  kept_amount = 2000 →
  initial_amount + received_amount - kept_amount = 1200 := by
sorry

end NUMINAMATH_CALUDE_randy_gave_sally_l801_80114


namespace NUMINAMATH_CALUDE_set_relations_l801_80125

theorem set_relations (A B : Set α) (h : ∃ x, x ∈ A ∧ x ∉ B) :
  (¬(A ⊆ B)) ∧
  (∃ A' B' : Set α, (∃ x, x ∈ A' ∧ x ∉ B') ∧ (A' ∩ B' ≠ ∅)) ∧
  (∃ A' B' : Set α, (∃ x, x ∈ A' ∧ x ∉ B') ∧ (B' ⊆ A')) ∧
  (∃ A' B' : Set α, (∃ x, x ∈ A' ∧ x ∉ B') ∧ (A' ∩ B' = ∅)) :=
by sorry

end NUMINAMATH_CALUDE_set_relations_l801_80125


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l801_80189

theorem decimal_sum_to_fraction :
  (0.1 : ℚ) + 0.03 + 0.004 + 0.0006 + 0.00007 = 13467 / 100000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l801_80189


namespace NUMINAMATH_CALUDE_diamond_value_l801_80165

/-- Given that ◇3 in base 5 equals ◇2 in base 6, where ◇ is a digit, prove that ◇ = 1 -/
theorem diamond_value (diamond : ℕ) (h1 : diamond < 10) :
  5 * diamond + 3 = 6 * diamond + 2 → diamond = 1 := by
  sorry

end NUMINAMATH_CALUDE_diamond_value_l801_80165


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l801_80148

theorem sqrt_x_div_sqrt_y (x y : ℝ) (h : (1/3)^2 + (1/4)^2 + (1/6)^2 = ((1/5)^2 + (1/7)^2 + (1/8)^2) * (54*x)/(115*y)) : 
  Real.sqrt x / Real.sqrt y = 49/29 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_l801_80148


namespace NUMINAMATH_CALUDE_box_height_proof_l801_80106

theorem box_height_proof (length width cube_volume num_cubes : ℝ) 
  (h1 : length = 12)
  (h2 : width = 16)
  (h3 : cube_volume = 3)
  (h4 : num_cubes = 384) :
  (num_cubes * cube_volume) / (length * width) = 6 := by
  sorry

end NUMINAMATH_CALUDE_box_height_proof_l801_80106


namespace NUMINAMATH_CALUDE_james_toy_ratio_l801_80159

/-- Given that James buys toy soldiers and toy cars, with 20 toy cars and a total of 60 toys,
    prove that the ratio of toy soldiers to toy cars is 2:1. -/
theorem james_toy_ratio :
  let total_toys : ℕ := 60
  let toy_cars : ℕ := 20
  let toy_soldiers : ℕ := total_toys - toy_cars
  (toy_soldiers : ℚ) / toy_cars = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_james_toy_ratio_l801_80159
