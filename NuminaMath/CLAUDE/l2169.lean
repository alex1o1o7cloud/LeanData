import Mathlib

namespace NUMINAMATH_CALUDE_cone_volume_l2169_216908

/-- The volume of a cone with slant height 15 cm and height 9 cm is 432π cubic centimeters. -/
theorem cone_volume (π : ℝ) (h : π > 0) : 
  let slant_height : ℝ := 15
  let height : ℝ := 9
  let radius : ℝ := Real.sqrt (slant_height^2 - height^2)
  let volume : ℝ := (1/3) * π * radius^2 * height
  volume = 432 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l2169_216908


namespace NUMINAMATH_CALUDE_max_area_triangle_l2169_216964

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x + Real.cos x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x - Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem max_area_triangle (C : ℝ) (a b c : ℝ) (hf : f C = 2) (hc : c = Real.sqrt 3) :
  ∃ S : ℝ, S ≤ (3 * Real.sqrt 3) / 4 ∧ 
  (∀ S' : ℝ, S' = 1/2 * a * b * Real.sin C → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_max_area_triangle_l2169_216964


namespace NUMINAMATH_CALUDE_tangent_line_inclination_angle_l2169_216998

/-- The curve y = x³ - 2x + 4 has a tangent line at (1, 3) with an inclination angle of 45° -/
theorem tangent_line_inclination_angle :
  let f (x : ℝ) := x^3 - 2*x + 4
  let f' (x : ℝ) := 3*x^2 - 2
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let θ : ℝ := Real.pi / 4  -- 45° in radians
  (f x₀ = y₀) ∧ 
  (Real.tan θ = f' x₀) →
  θ = Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_inclination_angle_l2169_216998


namespace NUMINAMATH_CALUDE_ammonia_molecular_weight_l2169_216948

/-- The atomic weight of nitrogen in atomic mass units (amu) -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of hydrogen in atomic mass units (amu) -/
def hydrogen_weight : ℝ := 1.008

/-- The number of nitrogen atoms in an ammonia molecule -/
def nitrogen_count : ℕ := 1

/-- The number of hydrogen atoms in an ammonia molecule -/
def hydrogen_count : ℕ := 3

/-- The molecular weight of ammonia in atomic mass units (amu) -/
def ammonia_weight : ℝ := nitrogen_weight * nitrogen_count + hydrogen_weight * hydrogen_count

theorem ammonia_molecular_weight :
  ammonia_weight = 17.034 := by sorry

end NUMINAMATH_CALUDE_ammonia_molecular_weight_l2169_216948


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l2169_216935

/-- An arithmetic sequence with first term -5 and positive terms starting from the 10th term
    has a common difference d in the range (5/9, 5/8] -/
theorem arithmetic_sequence_common_difference_range (a : ℕ → ℝ) (d : ℝ) :
  (∀ n : ℕ, a n = -5 + (n - 1) * d) →  -- Definition of arithmetic sequence
  (a 1 = -5) →                         -- First term is -5
  (∀ n ≥ 10, a n > 0) →                -- Terms from 10th onwards are positive
  5/9 < d ∧ d ≤ 5/8 :=                 -- Range of common difference
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_range_l2169_216935


namespace NUMINAMATH_CALUDE_sum_equals_square_l2169_216919

theorem sum_equals_square (k : ℕ) (N : ℕ) : N < 100 →
  (k * (k + 1)) / 2 = N^2 ↔ k = 1 ∨ k = 8 ∨ k = 49 := by sorry

end NUMINAMATH_CALUDE_sum_equals_square_l2169_216919


namespace NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l2169_216955

theorem one_and_two_thirds_of_x_is_45 (x : ℚ) : (5 / 3 : ℚ) * x = 45 → x = 27 := by
  sorry

end NUMINAMATH_CALUDE_one_and_two_thirds_of_x_is_45_l2169_216955


namespace NUMINAMATH_CALUDE_water_depth_conversion_l2169_216912

/-- Represents a right cylindrical water tank -/
structure WaterTank where
  height : Real
  baseDiameter : Real

/-- Calculates the volume of water in the tank when horizontal -/
def horizontalWaterVolume (tank : WaterTank) (depth : Real) : Real :=
  sorry

/-- Calculates the depth of water when the tank is vertical -/
def verticalWaterDepth (tank : WaterTank) (horizontalDepth : Real) : Real :=
  sorry

/-- Theorem stating the relationship between horizontal and vertical water depths -/
theorem water_depth_conversion (tank : WaterTank) (horizontalDepth : Real) :
  tank.height = 10 ∧ tank.baseDiameter = 6 ∧ horizontalDepth = 4 →
  verticalWaterDepth tank horizontalDepth = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_conversion_l2169_216912


namespace NUMINAMATH_CALUDE_strawberry_weight_difference_l2169_216902

/-- The weight difference between Marco's and his dad's strawberries -/
def weight_difference (marco_weight : ℕ) (total_weight : ℕ) : ℕ :=
  marco_weight - (total_weight - marco_weight)

/-- Theorem stating the weight difference given the problem conditions -/
theorem strawberry_weight_difference :
  weight_difference 30 47 = 13 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_weight_difference_l2169_216902


namespace NUMINAMATH_CALUDE_mangoes_rate_per_kg_l2169_216950

/-- Given Tom's fruit purchase details, prove the rate per kg for mangoes -/
theorem mangoes_rate_per_kg 
  (apple_quantity : ℕ) 
  (apple_rate : ℕ) 
  (mango_quantity : ℕ) 
  (total_paid : ℕ) 
  (h1 : apple_quantity = 8)
  (h2 : apple_rate = 70)
  (h3 : mango_quantity = 9)
  (h4 : total_paid = 965) :
  (total_paid - apple_quantity * apple_rate) / mango_quantity = 45 :=
by sorry

end NUMINAMATH_CALUDE_mangoes_rate_per_kg_l2169_216950


namespace NUMINAMATH_CALUDE_line_point_order_l2169_216900

theorem line_point_order (b : ℝ) (y₁ y₂ y₃ : ℝ) : 
  (y₁ = 3 * (-2.3) + b) → 
  (y₂ = 3 * (-1.3) + b) → 
  (y₃ = 3 * 2.7 + b) → 
  y₁ < y₂ ∧ y₂ < y₃ :=
by sorry

end NUMINAMATH_CALUDE_line_point_order_l2169_216900


namespace NUMINAMATH_CALUDE_odd_product_probability_l2169_216949

theorem odd_product_probability : 
  let n : ℕ := 25
  let odd_count : ℕ := (n + 1) / 2
  let total_combinations : ℕ := n * (n - 1) / 2
  let odd_combinations : ℕ := odd_count * (odd_count - 1) / 2
  (odd_combinations : ℚ) / total_combinations = 13 / 50 := by sorry

end NUMINAMATH_CALUDE_odd_product_probability_l2169_216949


namespace NUMINAMATH_CALUDE_floor_ceiling_identity_l2169_216987

theorem floor_ceiling_identity (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 1) :
  ⌊x⌋ + x - ⌈x⌉ = x - 1 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_identity_l2169_216987


namespace NUMINAMATH_CALUDE_f_inequality_l2169_216937

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Condition 1: Periodicity
axiom periodic (x : ℝ) : f (x + 4) = f x

-- Condition 2: Decreasing on [0, 2]
axiom decreasing (x₁ x₂ : ℝ) (h : 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 2) : f x₁ > f x₂

-- Condition 3: Symmetry about y-axis for f(x-2)
axiom symmetry (x : ℝ) : f ((-x) - 2) = f (x - 2)

-- Theorem to prove
theorem f_inequality : f (-1.5) < f 7 ∧ f 7 < f (-4.5) := by sorry

end NUMINAMATH_CALUDE_f_inequality_l2169_216937


namespace NUMINAMATH_CALUDE_ellipse_k_value_l2169_216982

/-- An ellipse with equation 4x² + ky² = 4 and a focus at (0, 1) has k = 2 -/
theorem ellipse_k_value (k : ℝ) : 
  (∀ x y : ℝ, 4 * x^2 + k * y^2 = 4) →  -- Ellipse equation
  (0, 1) ∈ {p : ℝ × ℝ | p.1^2 / 1^2 + p.2^2 / (4/k) = 1} →  -- Focus condition
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l2169_216982


namespace NUMINAMATH_CALUDE_polygon_perimeter_bounds_l2169_216962

theorem polygon_perimeter_bounds :
  ∃ (m₃ m₄ m₅ m₆ m₇ m₈ m₉ m₁₀ : ℝ),
    (abs m₃ ≤ 3) ∧
    (abs m₄ ≤ 5) ∧
    (abs m₅ ≤ 7) ∧
    (abs m₆ ≤ 9) ∧
    (abs m₇ ≤ 12) ∧
    (abs m₈ ≤ 14) ∧
    (abs m₉ ≤ 16) ∧
    (abs m₁₀ ≤ 19) ∧
    (m₃ ≤ m₄) ∧ (m₄ ≤ m₅) ∧ (m₅ ≤ m₆) ∧ (m₆ ≤ m₇) ∧
    (m₇ ≤ m₈) ∧ (m₈ ≤ m₉) ∧ (m₉ ≤ m₁₀) := by
  sorry


end NUMINAMATH_CALUDE_polygon_perimeter_bounds_l2169_216962


namespace NUMINAMATH_CALUDE_first_year_after_2020_with_sum_of_digits_10_l2169_216921

def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

def isFirstYearAfter2020WithSumOfDigits10 (year : ℕ) : Prop :=
  year > 2020 ∧ 
  sumOfDigits year = 10 ∧ 
  ∀ y, 2020 < y ∧ y < year → sumOfDigits y ≠ 10

theorem first_year_after_2020_with_sum_of_digits_10 :
  isFirstYearAfter2020WithSumOfDigits10 2026 := by
  sorry

#eval sumOfDigits 2026  -- Should output 10

end NUMINAMATH_CALUDE_first_year_after_2020_with_sum_of_digits_10_l2169_216921


namespace NUMINAMATH_CALUDE_largest_lcm_with_15_l2169_216925

theorem largest_lcm_with_15 : 
  let lcm_list := [Nat.lcm 15 3, Nat.lcm 15 5, Nat.lcm 15 6, Nat.lcm 15 9, Nat.lcm 15 10, Nat.lcm 15 15]
  List.maximum lcm_list = some 45 := by
sorry

end NUMINAMATH_CALUDE_largest_lcm_with_15_l2169_216925


namespace NUMINAMATH_CALUDE_dividend_calculation_l2169_216988

theorem dividend_calculation (remainder : ℕ) (divisor : ℕ) (quotient : ℕ) :
  remainder = 6 →
  divisor = 5 * quotient →
  divisor = 3 * remainder + 2 →
  divisor * quotient + remainder = 86 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2169_216988


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2169_216933

theorem inequality_solution_set (x : ℝ) :
  (|2*x + 1| - 2*|x - 1| > 0) ↔ (x > 0) := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2169_216933


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2169_216947

theorem sufficient_not_necessary_condition (x : ℝ) :
  (|x - 1/2| < 1/2 → x < 1) ∧ ¬(x < 1 → |x - 1/2| < 1/2) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l2169_216947


namespace NUMINAMATH_CALUDE_sine_inequality_l2169_216989

theorem sine_inequality (x y : ℝ) (h1 : 0 < x) (h2 : x < y) (h3 : y < 1) :
  0 < Real.sin x ∧ Real.sin x < Real.sin y := by sorry

end NUMINAMATH_CALUDE_sine_inequality_l2169_216989


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2169_216938

theorem geometric_sequence_sixth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 512) 
  (h2 : a * r^7 = 2) : 
  a * r^5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l2169_216938


namespace NUMINAMATH_CALUDE_estimate_N_l2169_216901

-- Define f(n) as the largest prime factor of n
def f (n : ℕ) : ℕ := sorry

-- Define the sum of f(n^2-1) for n from 2 to 10^6
def sum_f_nsquared_minus_one : ℕ := sorry

-- Define the sum of f(n) for n from 2 to 10^6
def sum_f_n : ℕ := sorry

-- Theorem statement
theorem estimate_N : 
  ⌊(10^4 : ℝ) * (sum_f_nsquared_minus_one : ℝ) / (sum_f_n : ℝ)⌋ = 18215 := by sorry

end NUMINAMATH_CALUDE_estimate_N_l2169_216901


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2169_216939

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
    (h : ArithmeticSequence a) 
    (eq : a 1 + 3 * a 8 + a 15 = 120) : 
  3 * a 9 - a 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2169_216939


namespace NUMINAMATH_CALUDE_ellen_painted_17_lilies_l2169_216960

/-- Time in minutes to paint each type of flower or vine -/
def lily_time : ℕ := 5
def rose_time : ℕ := 7
def orchid_time : ℕ := 3
def vine_time : ℕ := 2

/-- Total time spent painting -/
def total_time : ℕ := 213

/-- Number of roses, orchids, and vines painted -/
def roses : ℕ := 10
def orchids : ℕ := 6
def vines : ℕ := 20

/-- Function to calculate the number of lilies painted -/
def lilies_painted : ℕ := 
  (total_time - (roses * rose_time + orchids * orchid_time + vines * vine_time)) / lily_time

theorem ellen_painted_17_lilies : lilies_painted = 17 := by
  sorry

end NUMINAMATH_CALUDE_ellen_painted_17_lilies_l2169_216960


namespace NUMINAMATH_CALUDE_divisibility_condition_l2169_216956

theorem divisibility_condition (n : ℕ) : 
  n > 0 ∧ (n - 1) ∣ (n^3 + 4) ↔ n = 2 ∨ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l2169_216956


namespace NUMINAMATH_CALUDE_canvas_cost_decrease_canvas_cost_decrease_is_40_l2169_216920

theorem canvas_cost_decrease (paint_decrease : Real) (total_decrease : Real) 
  (paint_canvas_ratio : Real) (canvas_decrease : Real) : Real :=
  if paint_decrease = 60 ∧ 
     total_decrease = 55.99999999999999 ∧ 
     paint_canvas_ratio = 4 ∧ 
     ((1 - paint_decrease / 100) * paint_canvas_ratio + (1 - canvas_decrease / 100)) / 
     (paint_canvas_ratio + 1) = 1 - total_decrease / 100
  then canvas_decrease
  else 0

#check canvas_cost_decrease

theorem canvas_cost_decrease_is_40 :
  canvas_cost_decrease 60 55.99999999999999 4 40 = 40 := by
  sorry

end NUMINAMATH_CALUDE_canvas_cost_decrease_canvas_cost_decrease_is_40_l2169_216920


namespace NUMINAMATH_CALUDE_remaining_distance_is_546_point_5_l2169_216930

-- Define the total distance
def total_distance : ℝ := 1045

-- Define Amoli's driving
def amoli_speed : ℝ := 42
def amoli_time : ℝ := 3

-- Define Anayet's driving
def anayet_speed : ℝ := 61
def anayet_time : ℝ := 2.5

-- Define Bimal's driving
def bimal_speed : ℝ := 55
def bimal_time : ℝ := 4

-- Theorem statement
theorem remaining_distance_is_546_point_5 :
  total_distance - (amoli_speed * amoli_time + anayet_speed * anayet_time + bimal_speed * bimal_time) = 546.5 := by
  sorry

end NUMINAMATH_CALUDE_remaining_distance_is_546_point_5_l2169_216930


namespace NUMINAMATH_CALUDE_pedestrian_speeds_l2169_216985

theorem pedestrian_speeds (x y : ℝ) 
  (h1 : x + y = 14)
  (h2 : (3/2) * x + (1/2) * y = 13) :
  (x = 6 ∧ y = 8) ∨ (x = 8 ∧ y = 6) := by
  sorry

end NUMINAMATH_CALUDE_pedestrian_speeds_l2169_216985


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2169_216993

open Set
open Function
open Real

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Theorem statement
theorem solution_set_inequality 
  (h_domain : ∀ x, x > 0 → DifferentiableAt ℝ f x)
  (h_derivative : ∀ x, x > 0 → deriv f x = f' x)
  (h_inequality : ∀ x, x > 0 → f x > f' x) :
  {x : ℝ | Real.exp (x + 2) * f (x^2 - x) > Real.exp (x^2) * f 2} = 
  Ioo (-1) 0 ∪ Ioo 1 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2169_216993


namespace NUMINAMATH_CALUDE_vector_operation_proof_l2169_216924

def vector_operation : ℝ × ℝ := sorry

theorem vector_operation_proof :
  vector_operation = (5, 2) := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l2169_216924


namespace NUMINAMATH_CALUDE_train_length_proof_l2169_216957

def train_problem (distance_apart : ℝ) (train2_length : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time_to_meet : ℝ) : Prop :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let distance_covered := relative_speed * time_to_meet
  let train1_length := distance_covered - train2_length
  train1_length = 430

theorem train_length_proof :
  train_problem 630 200 90 72 13.998880089592832 :=
sorry

end NUMINAMATH_CALUDE_train_length_proof_l2169_216957


namespace NUMINAMATH_CALUDE_max_customers_interviewed_l2169_216905

theorem max_customers_interviewed (total : ℕ) (impulsive : ℕ) (ad_influence_percent : ℚ) (consultant_ratio : ℚ) : 
  total ≤ 50 ∧ 
  impulsive = 7 ∧ 
  ad_influence_percent = 3/4 ∧ 
  consultant_ratio = 1/3 →
  ∃ (max_customers : ℕ), 
    max_customers ≤ 50 ∧
    (∃ (ad_influenced : ℕ) (consultant_advised : ℕ),
      max_customers = impulsive + ad_influenced + consultant_advised ∧
      ad_influenced = ⌊(max_customers - impulsive) * ad_influence_percent⌋ ∧
      consultant_advised = ⌊ad_influenced * consultant_ratio⌋) ∧
    ∀ (n : ℕ), n > max_customers →
      ¬(∃ (ad_influenced : ℕ) (consultant_advised : ℕ),
        n = impulsive + ad_influenced + consultant_advised ∧
        ad_influenced = ⌊(n - impulsive) * ad_influence_percent⌋ ∧
        consultant_advised = ⌊ad_influenced * consultant_ratio⌋) ∧
    max_customers = 47 :=
by sorry

end NUMINAMATH_CALUDE_max_customers_interviewed_l2169_216905


namespace NUMINAMATH_CALUDE_expression_value_l2169_216944

theorem expression_value (a b c d m : ℝ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : m = 2 ∨ m = -2) : 
  (2 * a + 2 * b) / 3 - 5 * c * d + 8 * m = 11 ∨ 
  (2 * a + 2 * b) / 3 - 5 * c * d + 8 * m = -21 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2169_216944


namespace NUMINAMATH_CALUDE_sheep_with_only_fleas_l2169_216968

theorem sheep_with_only_fleas (total : ℕ) (lice : ℕ) (both : ℕ) (only_fleas : ℕ) : 
  total = 2 * lice →
  both = 84 →
  lice = 94 →
  total = only_fleas + (lice - both) + both →
  only_fleas = 94 := by
sorry

end NUMINAMATH_CALUDE_sheep_with_only_fleas_l2169_216968


namespace NUMINAMATH_CALUDE_prime_sum_of_squares_and_divisibility_l2169_216967

theorem prime_sum_of_squares_and_divisibility (p : ℕ) : 
  Prime p → 
  (∃ m n : ℤ, (p : ℤ) = m^2 + n^2 ∧ (m^3 + n^3 - 4) % p = 0) → 
  p = 2 ∨ p = 5 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_of_squares_and_divisibility_l2169_216967


namespace NUMINAMATH_CALUDE_binomial_expectation_and_variance_l2169_216915

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p
  h2 : p ≤ 1

/-- The expected value of a binomial random variable -/
def expected_value (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_expectation_and_variance :
  ∀ ξ : BinomialRV, ξ.n = 10 ∧ ξ.p = 0.6 → 
  expected_value ξ = 6 ∧ variance ξ = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expectation_and_variance_l2169_216915


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l2169_216979

theorem inscribed_circle_radius_right_triangle 
  (a b c r : ℝ) 
  (h_right : a^2 + b^2 = c^2) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_inscribed : r > 0 ∧ r * (a + b + c) = a * b) : 
  r = (a + b - c) / 2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_right_triangle_l2169_216979


namespace NUMINAMATH_CALUDE_boys_who_quit_l2169_216931

theorem boys_who_quit (initial_girls : ℕ) (initial_boys : ℕ) (girls_joined : ℕ) (final_total : ℕ) : 
  initial_girls = 18 → 
  initial_boys = 15 → 
  girls_joined = 7 → 
  final_total = 36 → 
  initial_boys - (final_total - (initial_girls + girls_joined)) = 4 := by
sorry

end NUMINAMATH_CALUDE_boys_who_quit_l2169_216931


namespace NUMINAMATH_CALUDE_females_with_advanced_degrees_only_l2169_216943

theorem females_with_advanced_degrees_only (total_employees : ℕ) 
  (female_employees : ℕ) (employees_with_advanced_degrees : ℕ)
  (employees_with_college_only : ℕ) (employees_with_multiple_degrees : ℕ)
  (males_with_college_only : ℕ) (males_with_multiple_degrees : ℕ)
  (females_with_multiple_degrees : ℕ)
  (h1 : total_employees = 148)
  (h2 : female_employees = 92)
  (h3 : employees_with_advanced_degrees = 78)
  (h4 : employees_with_college_only = 55)
  (h5 : employees_with_multiple_degrees = 15)
  (h6 : males_with_college_only = 31)
  (h7 : males_with_multiple_degrees = 8)
  (h8 : females_with_multiple_degrees = 10) :
  total_employees - female_employees - males_with_college_only - males_with_multiple_degrees +
  employees_with_advanced_degrees - females_with_multiple_degrees - males_with_multiple_degrees = 35 :=
by sorry

end NUMINAMATH_CALUDE_females_with_advanced_degrees_only_l2169_216943


namespace NUMINAMATH_CALUDE_hoseok_workbook_days_l2169_216981

/-- The number of days Hoseok solved the workbook -/
def days_solved : ℕ := 12

/-- The number of pages Hoseok solves per day -/
def pages_per_day : ℕ := 4

/-- The total number of pages Hoseok has solved -/
def total_pages : ℕ := 48

/-- Theorem stating that the number of days Hoseok solved the workbook is correct -/
theorem hoseok_workbook_days : 
  days_solved = total_pages / pages_per_day :=
by sorry

end NUMINAMATH_CALUDE_hoseok_workbook_days_l2169_216981


namespace NUMINAMATH_CALUDE_player_B_most_stable_l2169_216990

/-- Represents a player in the shooting test -/
inductive Player : Type
  | A
  | B
  | C
  | D

/-- Returns the variance of a given player -/
def variance (p : Player) : ℝ :=
  match p with
  | Player.A => 0.66
  | Player.B => 0.52
  | Player.C => 0.58
  | Player.D => 0.62

/-- Defines what it means for a player to have the most stable performance -/
def has_most_stable_performance (p : Player) : Prop :=
  ∀ q : Player, variance p ≤ variance q

/-- Theorem stating that Player B has the most stable performance -/
theorem player_B_most_stable :
  has_most_stable_performance Player.B := by
  sorry

end NUMINAMATH_CALUDE_player_B_most_stable_l2169_216990


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_twelve_l2169_216980

/-- The line equation in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The triangle formed by a line and the coordinate axes -/
structure Triangle where
  line : Line

def Triangle.perimeter (t : Triangle) : ℝ :=
  sorry

theorem triangle_perimeter_is_twelve (t : Triangle) :
  t.line = { a := 1/3, b := 1/4, c := 1 } →
  t.perimeter = 12 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_twelve_l2169_216980


namespace NUMINAMATH_CALUDE_initial_persons_count_l2169_216971

/-- The number of persons initially in the group. -/
def initial_persons : ℕ := sorry

/-- The average weight increase when a new person joins the group. -/
def avg_weight_increase : ℚ := 7/2

/-- The weight difference between the new person and the replaced person. -/
def weight_difference : ℚ := 28

theorem initial_persons_count : initial_persons = 8 := by
  have h1 : (initial_persons : ℚ) * avg_weight_increase = weight_difference := by sorry
  sorry

end NUMINAMATH_CALUDE_initial_persons_count_l2169_216971


namespace NUMINAMATH_CALUDE_total_pears_l2169_216916

def alyssa_pears : ℕ := 42
def nancy_pears : ℕ := 17

theorem total_pears : alyssa_pears + nancy_pears = 59 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_l2169_216916


namespace NUMINAMATH_CALUDE_k_value_at_4_l2169_216972

-- Define the polynomial h
def h (x : ℝ) : ℝ := x^3 - 2*x + 1

-- Define the properties of k
def k_properties (k : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, h a = 0 ∧ h b = 0 ∧ h c = 0 ∧
    ∀ x, k x = (x - a^2) * (x - b^2) * (x - c^2)) ∧
  k 0 = 1

-- Theorem statement
theorem k_value_at_4 (k : ℝ → ℝ) (hk : k_properties k) : k 4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_k_value_at_4_l2169_216972


namespace NUMINAMATH_CALUDE_projection_of_b_onto_a_l2169_216911

def a : Fin 3 → ℝ := ![2, -1, 2]
def b : Fin 3 → ℝ := ![1, -2, 1]

theorem projection_of_b_onto_a :
  let proj := (a • b) / (a • a) • a
  proj 0 = 4/3 ∧ proj 1 = -2/3 ∧ proj 2 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_projection_of_b_onto_a_l2169_216911


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2169_216904

/-- A polynomial P(x) with a real parameter r satisfies:
    1) P(x) has remainder 2 when divided by (x-r)
    2) P(x) has remainder (-2x^2 - 3x + 4) when divided by (2x^2 + 7x - 4)(x-r)
    This theorem states that r can only be 1/2 or -2. -/
theorem polynomial_remainder_theorem (P : ℝ → ℝ) (r : ℝ) :
  (∃ Q₁ : ℝ → ℝ, ∀ x, P x = (x - r) * Q₁ x + 2) ∧
  (∃ Q₂ : ℝ → ℝ, ∀ x, P x = (2*x^2 + 7*x - 4)*(x - r) * Q₂ x + (-2*x^2 - 3*x + 4)) →
  r = 1/2 ∨ r = -2 := by
sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_l2169_216904


namespace NUMINAMATH_CALUDE_no_solution_system_l2169_216941

theorem no_solution_system :
  ¬ ∃ x : ℝ, x > 2 ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_system_l2169_216941


namespace NUMINAMATH_CALUDE_system_solution_l2169_216954

theorem system_solution (a b : ℚ) : 
  (a/3 - 1) + 2*(b/5 + 2) = 4 ∧ 
  2*(a/3 - 1) + (b/5 + 2) = 5 → 
  a = 9 ∧ b = -5 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2169_216954


namespace NUMINAMATH_CALUDE_certain_number_is_eleven_l2169_216906

theorem certain_number_is_eleven : ∃ x : ℕ, 
  x + (3 * 13 + 3 * 14 + 3 * 17) = 143 ∧ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_eleven_l2169_216906


namespace NUMINAMATH_CALUDE_squares_not_always_congruent_l2169_216965

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define properties of squares
def Square.is_equiangular (s : Square) : Prop := True
def Square.is_rectangle (s : Square) : Prop := True
def Square.is_regular_polygon (s : Square) : Prop := True
def Square.is_similar_to (s1 s2 : Square) : Prop := True

-- Define congruence for squares
def Square.is_congruent_to (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem squares_not_always_congruent :
  ∃ (s1 s2 : Square),
    s1.is_equiangular ∧
    s1.is_rectangle ∧
    s1.is_regular_polygon ∧
    s2.is_equiangular ∧
    s2.is_rectangle ∧
    s2.is_regular_polygon ∧
    Square.is_similar_to s1 s2 ∧
    ¬ Square.is_congruent_to s1 s2 :=
by
  sorry

end NUMINAMATH_CALUDE_squares_not_always_congruent_l2169_216965


namespace NUMINAMATH_CALUDE_candy_calories_per_serving_l2169_216928

/-- Calculates the number of calories per serving in a package of candy. -/
def calories_per_serving (total_servings : ℕ) (half_package_calories : ℕ) : ℕ :=
  (2 * half_package_calories) / total_servings

/-- Proves that the number of calories per serving is 120, given the problem conditions. -/
theorem candy_calories_per_serving :
  calories_per_serving 3 180 = 120 := by
  sorry

end NUMINAMATH_CALUDE_candy_calories_per_serving_l2169_216928


namespace NUMINAMATH_CALUDE_company_theorem_l2169_216945

-- Define the type for people
variable {Person : Type}

-- Define the "knows" relation
variable (knows : Person → Person → Prop)

-- Define the company as a finite set of people
variable [Finite Person]

-- State the theorem
theorem company_theorem 
  (h : ∀ (S : Finset Person), S.card = 9 → ∃ (x y : Person), x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ knows x y) :
  ∃ (G : Finset Person), G.card = 8 ∧ 
    ∀ p, p ∉ G → ∃ q ∈ G, knows p q :=
sorry

end NUMINAMATH_CALUDE_company_theorem_l2169_216945


namespace NUMINAMATH_CALUDE_stratified_sampling_proportion_choose_two_from_six_prob_at_least_one_from_last_two_l2169_216984

/- Define the associations and their sizes -/
def associations : Fin 3 → ℕ
| 0 => 27  -- Association A
| 1 => 9   -- Association B
| 2 => 18  -- Association C

/- Total number of athletes -/
def total_athletes : ℕ := (associations 0) + (associations 1) + (associations 2)

/- Number of athletes to be selected -/
def selected_athletes : ℕ := 6

/- Theorem for stratified sampling -/
theorem stratified_sampling_proportion (i : Fin 3) :
  (associations i) * selected_athletes = (associations i) * total_athletes / total_athletes :=
sorry

/- Theorem for number of ways to choose 2 from 6 -/
theorem choose_two_from_six :
  Nat.choose selected_athletes 2 = 15 :=
sorry

/- Theorem for probability of selecting at least one from last two -/
theorem prob_at_least_one_from_last_two :
  (Nat.choose 4 1 * Nat.choose 2 1 + Nat.choose 2 2) / Nat.choose 6 2 = 3 / 5 :=
sorry

end NUMINAMATH_CALUDE_stratified_sampling_proportion_choose_two_from_six_prob_at_least_one_from_last_two_l2169_216984


namespace NUMINAMATH_CALUDE_root_implies_p_minus_q_l2169_216996

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (p q : ℝ) (x : ℂ) : Prop :=
  2 * x^2 + p * x + q = 0

-- State the theorem
theorem root_implies_p_minus_q (p q : ℝ) :
  equation p q (-2 * i - 3) → p - q = -14 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_p_minus_q_l2169_216996


namespace NUMINAMATH_CALUDE_star_difference_equals_28_l2169_216910

def star (a b : ℝ) : ℝ := a^2 + 2*a*b + b^2

theorem star_difference_equals_28 : (star 3 5) - (star 2 4) = 28 := by
  sorry

end NUMINAMATH_CALUDE_star_difference_equals_28_l2169_216910


namespace NUMINAMATH_CALUDE_commute_days_l2169_216976

theorem commute_days (bus_to_work bus_to_home train_days train_both : ℕ) : 
  bus_to_work = 12 → 
  bus_to_home = 20 → 
  train_days = 14 → 
  train_both = 2 → 
  ∃ x : ℕ, x = 23 ∧ 
    x = (bus_to_home - bus_to_work + train_both) + 
        (bus_to_work - train_both) + 
        (train_days - (bus_to_home - bus_to_work)) + 
        train_both :=
by sorry

end NUMINAMATH_CALUDE_commute_days_l2169_216976


namespace NUMINAMATH_CALUDE_elaine_rent_percentage_l2169_216975

/-- Represents Elaine's earnings and rent expenses over two years -/
structure ElaineFinances where
  lastYearEarnings : ℝ
  lastYearRentPercentage : ℝ
  earningsIncrease : ℝ
  rentIncrease : ℝ
  thisYearRentPercentage : ℝ

/-- The conditions of Elaine's finances -/
def elaineFinancesConditions (e : ElaineFinances) : Prop :=
  e.lastYearRentPercentage = 20 ∧
  e.earningsIncrease = 35 ∧
  e.rentIncrease = 202.5

/-- Theorem stating that given the conditions, Elaine's rent percentage this year is 30% -/
theorem elaine_rent_percentage (e : ElaineFinances) 
  (h : elaineFinancesConditions e) : e.thisYearRentPercentage = 30 := by
  sorry


end NUMINAMATH_CALUDE_elaine_rent_percentage_l2169_216975


namespace NUMINAMATH_CALUDE_pyramid_cube_tiling_exists_l2169_216969

/-- A shape constructed from a cube with a pyramid on one face -/
structure PyramidCube where
  -- The edge length of the base cube
  cube_edge : ℝ
  -- The height of the pyramid (assumed to be equal to cube_edge)
  pyramid_height : ℝ
  -- Assumption that the pyramid height equals the cube edge length
  height_eq_edge : pyramid_height = cube_edge

/-- A tiling of 3D space using congruent copies of a shape -/
structure Tiling (shape : PyramidCube) where
  -- The set of positions (as points in ℝ³) where shapes are placed
  positions : Set (Fin 3 → ℝ)
  -- Property ensuring the tiling is seamless (no gaps)
  seamless : sorry
  -- Property ensuring the tiling has no overlaps
  no_overlap : sorry

/-- Theorem stating that a space-filling tiling exists for the PyramidCube shape -/
theorem pyramid_cube_tiling_exists :
  ∃ (shape : PyramidCube) (tiling : Tiling shape), True :=
sorry

end NUMINAMATH_CALUDE_pyramid_cube_tiling_exists_l2169_216969


namespace NUMINAMATH_CALUDE_quadratic_roots_transformation_l2169_216983

theorem quadratic_roots_transformation (a b c x₁ x₂ : ℝ) (h₁ : a ≠ 0) :
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (∀ x, a^3 * x^2 - a * b^2 * x + 2 * c * (b^2 - 2 * a * c) = 0 ↔ x = x₁^2 + x₂^2 ∨ x = 2 * x₁ * x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_transformation_l2169_216983


namespace NUMINAMATH_CALUDE_expression_evaluation_l2169_216907

/-- Evaluates the expression (3x^3 - 7x^2 + 4x - 9) / (2x - 0.5) for x = 100 -/
theorem expression_evaluation :
  let x : ℝ := 100
  let numerator := 3 * x^3 - 7 * x^2 + 4 * x - 9
  let denominator := 2 * x - 0.5
  abs ((numerator / denominator) - 14684.73534) < 0.00001 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2169_216907


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l2169_216932

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

-- Define what it means for a quadratic radical to be in its simplest form
def isSimplestQuadraticRadical (n : ℚ) : Prop :=
  ∃ (a : ℕ), isPrime a ∧ n = (a : ℚ).sqrt

-- Theorem statement
theorem simplest_quadratic_radical :
  let options : List ℚ := [9, 7, 20, (1/3 : ℚ)]
  ∃ (x : ℚ), x ∈ options ∧ isSimplestQuadraticRadical x ∧
    ∀ (y : ℚ), y ∈ options → y ≠ x → ¬(isSimplestQuadraticRadical y) :=
by sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l2169_216932


namespace NUMINAMATH_CALUDE_kola_sugar_percentage_l2169_216995

/-- Calculates the percentage of sugar in a kola solution after adding ingredients -/
theorem kola_sugar_percentage
  (initial_volume : Real)
  (initial_water_percent : Real)
  (initial_kola_percent : Real)
  (added_sugar : Real)
  (added_water : Real)
  (added_kola : Real)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percent = 88)
  (h3 : initial_kola_percent = 5)
  (h4 : added_sugar = 3.2)
  (h5 : added_water = 10)
  (h6 : added_kola = 6.8) :
  let initial_sugar_percent := 100 - initial_water_percent - initial_kola_percent
  let initial_water := initial_volume * initial_water_percent / 100
  let initial_kola := initial_volume * initial_kola_percent / 100
  let initial_sugar := initial_volume * initial_sugar_percent / 100
  let final_water := initial_water + added_water
  let final_kola := initial_kola + added_kola
  let final_sugar := initial_sugar + added_sugar
  let final_volume := final_water + final_kola + final_sugar
  final_sugar / final_volume * 100 = 7.5 := by
  sorry


end NUMINAMATH_CALUDE_kola_sugar_percentage_l2169_216995


namespace NUMINAMATH_CALUDE_least_integer_square_36_more_than_triple_l2169_216974

theorem least_integer_square_36_more_than_triple (x : ℤ) :
  (x^2 = 3*x + 36) → (x ≥ -6) :=
by sorry

end NUMINAMATH_CALUDE_least_integer_square_36_more_than_triple_l2169_216974


namespace NUMINAMATH_CALUDE_original_denominator_problem_l2169_216936

theorem original_denominator_problem (d : ℤ) : 
  (3 : ℚ) / d ≠ 0 →
  (3 + 7 : ℚ) / (d + 7) = 1 / 3 →
  d = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l2169_216936


namespace NUMINAMATH_CALUDE_complex_norm_squared_l2169_216913

theorem complex_norm_squared (a b : ℝ) : 
  let z : ℂ := Complex.mk a (-b)
  Complex.normSq z = a^2 + b^2 := by sorry

end NUMINAMATH_CALUDE_complex_norm_squared_l2169_216913


namespace NUMINAMATH_CALUDE_remaining_savings_is_25_70_l2169_216973

/-- Calculates the remaining savings after jewelry purchases and tax --/
def remaining_savings (initial_savings earrings_cost necklace_cost bracelet_cost jewelry_set_cost discount_percent tax_percent : ℚ) : ℚ :=
  let individual_items_cost := earrings_cost + necklace_cost + bracelet_cost
  let discounted_jewelry_set_cost := jewelry_set_cost * (1 - discount_percent / 100)
  let total_before_tax := individual_items_cost + discounted_jewelry_set_cost
  let tax_amount := total_before_tax * (tax_percent / 100)
  let final_total_cost := total_before_tax + tax_amount
  initial_savings - final_total_cost

/-- Theorem stating that the remaining savings are $25.70 --/
theorem remaining_savings_is_25_70 :
  remaining_savings 200 23 48 35 80 25 5 = 25.70 := by
  sorry

end NUMINAMATH_CALUDE_remaining_savings_is_25_70_l2169_216973


namespace NUMINAMATH_CALUDE_greatest_common_divisor_450_90_under_60_l2169_216953

theorem greatest_common_divisor_450_90_under_60 : 
  ∃ (n : ℕ), n = 45 ∧ 
  n ∣ 450 ∧ 
  n < 60 ∧ 
  n ∣ 90 ∧ 
  ∀ (m : ℕ), m ∣ 450 ∧ m < 60 ∧ m ∣ 90 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_divisor_450_90_under_60_l2169_216953


namespace NUMINAMATH_CALUDE_circle_area_above_line_l2169_216917

/-- The circle equation -/
def circle_eq (x y : ℝ) : Prop := x^2 - 10*x + y^2 - 16*y + 56 = 0

/-- The line equation -/
def line_eq (y : ℝ) : Prop := y = 4

/-- The area of the circle portion above the line -/
noncomputable def area_above_line : ℝ := 99 * Real.pi / 4

/-- Theorem stating that the area of the circle portion above the line is approximately equal to 99π/4 -/
theorem circle_area_above_line :
  ∃ (ε : ℝ), ε > 0 ∧ 
  (∀ x y : ℝ, circle_eq x y → line_eq y → 
    abs (area_above_line - (Real.pi * 33 * 3 / 4)) < ε) :=
sorry

end NUMINAMATH_CALUDE_circle_area_above_line_l2169_216917


namespace NUMINAMATH_CALUDE_expression_evaluation_l2169_216959

theorem expression_evaluation (x : ℝ) (h : x = -2) : 
  (3 * x / (x - 1) - x / (x + 1)) * (x^2 - 1) / x = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2169_216959


namespace NUMINAMATH_CALUDE_most_economical_cost_l2169_216909

/-- Represents the problem of finding the most economical cost for purchasing warm reminder signs and garbage bins. -/
theorem most_economical_cost
  (price_difference : ℕ)
  (price_ratio : ℕ)
  (total_items : ℕ)
  (max_cost : ℕ)
  (bin_sign_ratio : ℚ)
  (h1 : price_difference = 350)
  (h2 : price_ratio = 3)
  (h3 : total_items = 3000)
  (h4 : max_cost = 350000)
  (h5 : bin_sign_ratio = 3/2)
  : ∃ (sign_price bin_price : ℕ) (num_signs num_bins : ℕ) (total_cost : ℕ),
    -- Price relationship between bins and signs
    4 * bin_price - 5 * sign_price = price_difference ∧
    bin_price = price_ratio * sign_price ∧
    -- Total number of items constraint
    num_signs + num_bins = total_items ∧
    -- Cost constraint
    num_signs * sign_price + num_bins * bin_price ≤ max_cost ∧
    -- Ratio constraint
    (num_bins : ℚ) ≥ bin_sign_ratio * (num_signs : ℚ) ∧
    -- Most economical solution
    num_signs = 1200 ∧
    total_cost = 330000 ∧
    -- No cheaper solution exists
    ∀ (other_signs : ℕ), 
      other_signs ≠ num_signs →
      other_signs + (total_items - other_signs) = total_items →
      (total_items - other_signs : ℚ) ≥ bin_sign_ratio * (other_signs : ℚ) →
      other_signs * sign_price + (total_items - other_signs) * bin_price ≥ total_cost :=
by sorry


end NUMINAMATH_CALUDE_most_economical_cost_l2169_216909


namespace NUMINAMATH_CALUDE_olivias_bags_l2169_216963

/-- The number of cans Olivia had in total -/
def total_cans : ℕ := 20

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 5

/-- The number of bags Olivia had -/
def number_of_bags : ℕ := total_cans / cans_per_bag

theorem olivias_bags : number_of_bags = 4 := by
  sorry

end NUMINAMATH_CALUDE_olivias_bags_l2169_216963


namespace NUMINAMATH_CALUDE_bakery_flour_calculation_l2169_216978

/-- Given a bakery that uses wheat flour and white flour, prove that the amount of white flour
    used is equal to the total amount of flour used minus the amount of wheat flour used. -/
theorem bakery_flour_calculation (total_flour white_flour wheat_flour : ℝ) 
    (h1 : total_flour = 0.3)
    (h2 : wheat_flour = 0.2) :
  white_flour = total_flour - wheat_flour := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_calculation_l2169_216978


namespace NUMINAMATH_CALUDE_money_problem_l2169_216926

theorem money_problem (a b : ℝ) 
  (h1 : 5 * a + b > 51)
  (h2 : 3 * a - b = 21) :
  a > 9 ∧ b > 6 := by
sorry

end NUMINAMATH_CALUDE_money_problem_l2169_216926


namespace NUMINAMATH_CALUDE_candied_apple_price_l2169_216958

/-- Given the conditions of candy production and sales, prove the price of each candied apple. -/
theorem candied_apple_price :
  ∀ (num_apples num_grapes : ℕ) (grape_price total_earnings : ℚ),
    num_apples = 15 →
    num_grapes = 12 →
    grape_price = 3/2 →
    total_earnings = 48 →
    ∃ (apple_price : ℚ),
      apple_price * num_apples + grape_price * num_grapes = total_earnings ∧
      apple_price = 2 := by
sorry

end NUMINAMATH_CALUDE_candied_apple_price_l2169_216958


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l2169_216934

theorem ice_cream_flavors (n : ℕ) (k : ℕ) : 
  n = 4 → k = 5 → (n + k - 1).choose k = 56 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l2169_216934


namespace NUMINAMATH_CALUDE_polynomial_coefficient_equality_l2169_216946

theorem polynomial_coefficient_equality (a b c : ℚ) : 
  (∀ x, (7*x^2 - 5*x + 9/4)*(a*x^2 + b*x + c) = 21*x^4 - 24*x^3 + 28*x^2 - 37/4*x + 21/4) →
  (a = 3 ∧ b = -9/7) := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_equality_l2169_216946


namespace NUMINAMATH_CALUDE_trigonometric_sum_equality_l2169_216923

theorem trigonometric_sum_equality : 
  Real.cos (π / 3) + Real.sin (π / 3) - Real.sqrt (3 / 4) + (Real.tan (π / 4))⁻¹ = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_sum_equality_l2169_216923


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_positive_l2169_216940

/-- Given an arithmetic sequence {a_n} where S_n denotes the sum of its first n terms,
    if S_(2k+1) > 0, then a_(k+1) > 0. -/
theorem arithmetic_sequence_middle_term_positive
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (k : ℕ)      -- An arbitrary natural number
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence condition
  (h_sum : ∀ n, S n = (n * (a 0 + a (n - 1))) / 2)  -- Sum formula for arithmetic sequence
  (h_positive : S (2 * k + 1) > 0)  -- Given condition
  : a (k + 1) > 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_positive_l2169_216940


namespace NUMINAMATH_CALUDE_balloon_rearrangements_eq_36_l2169_216903

/-- The number of distinguishable rearrangements of "BALLOON" with specific conditions -/
def balloon_rearrangements : ℕ :=
  let vowels := ['A', 'O', 'O']
  let consonants := ['B', 'L', 'L', 'N']
  let consonant_arrangements := Nat.factorial 4 / Nat.factorial 2
  let vowel_arrangements := Nat.factorial 3 / Nat.factorial 2
  consonant_arrangements * vowel_arrangements

/-- Theorem stating that the number of rearrangements is 36 -/
theorem balloon_rearrangements_eq_36 : balloon_rearrangements = 36 := by
  sorry

end NUMINAMATH_CALUDE_balloon_rearrangements_eq_36_l2169_216903


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2169_216927

/-- Represents a repeating decimal with a given numerator and denominator. -/
structure RepeatingDecimal where
  numerator : ℕ
  denominator : ℕ
  denom_nonzero : denominator ≠ 0

/-- Converts a repeating decimal to a rational number. -/
def RepeatingDecimal.toRational (r : RepeatingDecimal) : ℚ :=
  ↑r.numerator / ↑r.denominator

theorem repeating_decimal_subtraction :
  let a : RepeatingDecimal := ⟨845, 999, by norm_num⟩
  let b : RepeatingDecimal := ⟨267, 999, by norm_num⟩
  let c : RepeatingDecimal := ⟨159, 999, by norm_num⟩
  a.toRational - b.toRational - c.toRational = 419 / 999 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l2169_216927


namespace NUMINAMATH_CALUDE_grains_per_teaspoon_l2169_216994

/-- Represents the number of grains of rice in one cup -/
def grains_per_cup : ℕ := 480

/-- Represents the number of tablespoons in half a cup -/
def tablespoons_per_half_cup : ℕ := 8

/-- Represents the number of teaspoons in one tablespoon -/
def teaspoons_per_tablespoon : ℕ := 3

/-- Theorem stating that there are 10 grains of rice in a teaspoon -/
theorem grains_per_teaspoon :
  (grains_per_cup / (2 * tablespoons_per_half_cup * teaspoons_per_tablespoon)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_grains_per_teaspoon_l2169_216994


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l2169_216992

/-- The ratio of sheep to horses at Stewart farm -/
theorem stewart_farm_ratio : 
  ∀ (num_sheep num_horses : ℕ) (food_per_horse total_horse_food : ℕ),
  num_sheep = 24 →
  food_per_horse = 230 →
  total_horse_food = 12880 →
  num_horses * food_per_horse = total_horse_food →
  (num_sheep : ℚ) / (num_horses : ℚ) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l2169_216992


namespace NUMINAMATH_CALUDE_classroom_students_l2169_216918

/-- The number of pencils in a dozen -/
def pencils_per_dozen : ℕ := 12

/-- The number of dozens of pencils each student gets -/
def dozens_per_student : ℕ := 4

/-- The total number of pencils to be given out -/
def total_pencils : ℕ := 2208

/-- The number of students in the classroom -/
def num_students : ℕ := total_pencils / (dozens_per_student * pencils_per_dozen)

theorem classroom_students :
  num_students = 46 := by sorry

end NUMINAMATH_CALUDE_classroom_students_l2169_216918


namespace NUMINAMATH_CALUDE_division_problem_l2169_216951

theorem division_problem (d : ℕ) : d > 0 ∧ 23 = d * 7 + 2 → d = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2169_216951


namespace NUMINAMATH_CALUDE_circle_radius_calculation_l2169_216952

-- Define the circles and triangle
def circleA : ℝ := 13  -- radius of circle A
def circleB : ℝ := 4   -- radius of circle B
def circleC : ℝ := 3   -- radius of circle C

-- Define the theorem
theorem circle_radius_calculation (r : ℝ) : 
  -- Right triangle T inscribed in circle A
  -- Circle B internally tangent to A at one vertex of T
  -- Circle C internally tangent to A at another vertex of T
  -- Circles B and C externally tangent to circle E with radius r
  -- Angle between radii of A touching vertices related to B and C is 90°
  r = (Real.sqrt 181 - 7) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_calculation_l2169_216952


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2169_216999

theorem arithmetic_geometric_sequence : 
  ∃ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (b - a = c - b) ∧ 
    (a + b + c = 15) ∧ 
    ((a + 1) * (c + 9) = (b + 3)^2) ∧
    (a = 1 ∧ b = 5 ∧ c = 9) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l2169_216999


namespace NUMINAMATH_CALUDE_marble_count_l2169_216961

/-- The total number of marbles owned by Albert, Angela, Allison, Addison, and Alex -/
def total_marbles (allison angela albert addison alex : ℕ) : ℕ :=
  allison + angela + albert + addison + alex

/-- Theorem stating the total number of marbles given the conditions -/
theorem marble_count :
  ∀ (allison angela albert addison alex : ℕ),
    allison = 28 →
    angela = allison + 8 →
    albert = 3 * angela →
    addison = 2 * albert →
    alex = allison + 5 →
    alex = angela / 2 →
    total_marbles allison angela albert addison alex = 421 := by
  sorry


end NUMINAMATH_CALUDE_marble_count_l2169_216961


namespace NUMINAMATH_CALUDE_correct_operation_l2169_216942

theorem correct_operation (a b : ℝ) : 2 * a^2 * b - 3 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2169_216942


namespace NUMINAMATH_CALUDE_f_constant_on_interval_inequality_solution_condition_l2169_216977

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 3|

-- Theorem 1: f(x) is constant on the interval [-3, 1]
theorem f_constant_on_interval :
  ∀ x y : ℝ, x ∈ Set.Icc (-3) 1 → y ∈ Set.Icc (-3) 1 → f x = f y :=
sorry

-- Theorem 2: For f(x) - a ≤ 0 to have a solution, a must be ≥ 4
theorem inequality_solution_condition :
  ∀ a : ℝ, (∃ x : ℝ, f x - a ≤ 0) ↔ a ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_f_constant_on_interval_inequality_solution_condition_l2169_216977


namespace NUMINAMATH_CALUDE_sector_central_angle_l2169_216922

/-- Given a sector with radius 10 and area 50π/3, its central angle is π/3. -/
theorem sector_central_angle (r : ℝ) (S : ℝ) (h1 : r = 10) (h2 : S = 50 * Real.pi / 3) :
  S = 1/2 * r^2 * (Real.pi/3) := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l2169_216922


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l2169_216914

theorem no_solution_for_equation :
  ¬∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l2169_216914


namespace NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l2169_216991

theorem smallest_n_for_sqrt_inequality : 
  ∀ n : ℕ, n > 0 → (Real.sqrt n - Real.sqrt (n - 1) < 0.01 ↔ n ≥ 2501) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_sqrt_inequality_l2169_216991


namespace NUMINAMATH_CALUDE_seven_couples_handshakes_l2169_216929

/-- The number of handshakes in a gathering of couples -/
def handshakes (num_couples : ℕ) : ℕ :=
  let total_people := 2 * num_couples
  let handshakes_per_person := total_people - 3
  (total_people * handshakes_per_person) / 2

/-- Theorem: In a gathering of 7 couples, where each person shakes hands with everyone
    except their spouse and one other person, the total number of handshakes is 77. -/
theorem seven_couples_handshakes :
  handshakes 7 = 77 := by
  sorry

end NUMINAMATH_CALUDE_seven_couples_handshakes_l2169_216929


namespace NUMINAMATH_CALUDE_total_schedules_l2169_216970

/-- Represents the number of periods in a day -/
def total_periods : ℕ := 6

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 3

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 3

/-- Represents the total number of subjects -/
def total_subjects : ℕ := 6

/-- Represents the number of ways to schedule Mathematics in the morning and Art in the afternoon -/
def math_art_schedules : ℕ := morning_periods * afternoon_periods

/-- Represents the number of remaining subjects to be scheduled -/
def remaining_subjects : ℕ := total_subjects - 2

/-- Represents the number of remaining periods to schedule the remaining subjects -/
def remaining_periods : ℕ := total_periods - 2

/-- The main theorem stating the total number of possible schedules -/
theorem total_schedules : 
  math_art_schedules * (Nat.factorial remaining_subjects) = 216 :=
sorry

end NUMINAMATH_CALUDE_total_schedules_l2169_216970


namespace NUMINAMATH_CALUDE_binomial_17_9_l2169_216966

theorem binomial_17_9 (h1 : Nat.choose 15 6 = 5005) (h2 : Nat.choose 15 8 = 6435) :
  Nat.choose 17 9 = 24310 := by
  sorry

end NUMINAMATH_CALUDE_binomial_17_9_l2169_216966


namespace NUMINAMATH_CALUDE_parabola_intersection_l2169_216986

theorem parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 - 4 * y + 7) → k = 25/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_l2169_216986


namespace NUMINAMATH_CALUDE_printer_ratio_l2169_216997

theorem printer_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hx_time : x = 16) (hy_time : y = 12) (hz_time : z = 8) :
  x / ((1 / y + 1 / z)⁻¹) = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_printer_ratio_l2169_216997
