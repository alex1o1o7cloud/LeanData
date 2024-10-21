import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_tangent_ratio_l489_48909

open Real MeasureTheory

/-- The definite integral of (12 + tan(x)) / (3 * sin^2(x) + 12 * cos^2(x)) from 0 to arctan(2) is equal to π/2 + (1/6) * ln(2) -/
theorem integral_tangent_ratio : 
  ∫ x in (0)..(arctan 2), (12 + tan x) / (3 * (sin x)^2 + 12 * (cos x)^2) = π/2 + (1/6) * log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_tangent_ratio_l489_48909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_walked_miles_l489_48985

/-- Represents a pedometer with a maximum count before flipping --/
structure Pedometer where
  max_count : ℕ
  flips : ℕ
  final_reading : ℕ

/-- Calculates the total steps walked based on pedometer data --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_count * p.flips + p.final_reading

/-- Converts steps to miles --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℚ :=
  (steps : ℚ) / (steps_per_mile : ℚ)

/-- Rounds a rational number to the nearest integer --/
def round_to_nearest (q : ℚ) : ℤ :=
  ⌊q + 1/2⌋

/-- Theorem stating that Anne walked approximately 2950 miles --/
theorem anne_walked_miles : ∃ (p : Pedometer),
  p.max_count = 50000 ∧
  p.flips = 88 ∧
  p.final_reading = 25000 ∧
  round_to_nearest (steps_to_miles (total_steps p) 1500) = 2950 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_anne_walked_miles_l489_48985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_to_cos_l489_48984

theorem sin_shift_to_cos (φ : ℝ) : 
  (∀ x, Real.sin (2*x - 2*φ) = Real.cos (2*x + π/6)) ∧ φ > 0 → φ = 2*π/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_shift_to_cos_l489_48984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_product_l489_48925

theorem units_digit_product : ∃ (n : ℕ), n = 4 := by
  -- Define a function to calculate the units digit of a number
  let units_digit (n : ℕ) := n % 10

  -- Define the expressions
  let expr1 := 734^99 + 347^83
  let expr2 := 956^75 - 214^61

  -- Calculate the units digits
  let digit1 := units_digit expr1
  let digit2 := units_digit expr2

  -- Calculate the product of the units digits
  let result := digit1 * digit2

  -- State that there exists a natural number equal to 4
  -- which is the units digit of the product
  use units_digit result
  sorry

#eval 4 -- This will output 4, confirming our theorem statement

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_product_l489_48925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l489_48958

theorem sum_remainder (a b c : ℕ) 
  (ha : a % 15 = 11)
  (hb : b % 15 = 13)
  (hc : c % 15 = 14) :
  (a + b + c) % 15 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_remainder_l489_48958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_sale_earnings_l489_48986

/-- The price of a pair of mice -/
noncomputable def pair_price : ℚ := 534/100

/-- The number of mice sold -/
def mice_sold : ℕ := 7

/-- The total amount earned from selling the mice -/
noncomputable def total_earned : ℚ := (mice_sold : ℚ) * (pair_price / 2)

theorem mice_sale_earnings : total_earned = 1869/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_sale_earnings_l489_48986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_l489_48944

theorem cube_root_plus_abs : ((-27 : ℝ) ^ (1/3 : ℝ)) + |(-5 : ℝ)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_l489_48944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_invested_approx_l489_48907

/-- Calculates the amount invested in a stock given its parameters. -/
noncomputable def amount_invested (stock_rate : ℝ) (income : ℝ) (market_value : ℝ) (brokerage_rate : ℝ) : ℝ :=
  let face_value := income * 100 / stock_rate
  let actual_price := market_value * (1 + brokerage_rate / 100)
  face_value * actual_price / 100

/-- Theorem stating that the amount invested is approximately 8001.95 given the specified parameters. -/
theorem amount_invested_approx (stock_rate income market_value brokerage_rate : ℝ) 
  (h1 : stock_rate = 10.5)
  (h2 : income = 756)
  (h3 : market_value = 110.86111111111111)
  (h4 : brokerage_rate = 0.25) :
  ∃ ε > 0, |amount_invested stock_rate income market_value brokerage_rate - 8001.95| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amount_invested_approx_l489_48907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l489_48919

noncomputable def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

noncomputable def max_value (a : ℝ) : ℝ :=
  if a < 0 then 3 - 4*a
  else if a < 1 then 3 - 4*a
  else -1

noncomputable def min_value (a : ℝ) : ℝ :=
  if a < 0 then -1
  else if a < 2 then -1 - a^2
  else 3 - 4*a

theorem f_extrema (a : ℝ) :
  ∀ x ∈ Set.Icc 0 2, 
    min_value a ≤ f a x ∧ f a x ≤ max_value a := by
  sorry

#check f_extrema

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l489_48919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_c_value_l489_48918

/-- Represents the distribution of birds on trees -/
def BirdDistribution := Fin 120 → ℕ

/-- The total number of birds -/
def total_birds : ℕ := 2017

/-- The function c that sums the squares of birds on each tree -/
def c (dist : BirdDistribution) : ℕ :=
  Finset.sum Finset.univ (λ i => (dist i) ^ 2)

/-- The sum of birds across all trees equals the total number of birds -/
def valid_distribution (dist : BirdDistribution) : Prop :=
  Finset.sum Finset.univ dist = total_birds

theorem max_c_value (dist : BirdDistribution) (h : valid_distribution dist) :
  c dist ≤ total_birds ^ 2 ∧ ∃ (max_dist : BirdDistribution), valid_distribution max_dist ∧ c max_dist = total_birds ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_c_value_l489_48918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_recording_distribution_l489_48932

theorem novel_recording_distribution (total_time min_per_cd : ℕ) 
  (h1 : total_time = 528) 
  (h2 : min_per_cd = 70) : 
  (total_time / ((total_time + min_per_cd - 1) / min_per_cd) = 66) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_novel_recording_distribution_l489_48932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_modulo_l489_48902

theorem binomial_sum_modulo (n : ℕ) :
  (Finset.sum (Finset.range ((n + 2) / 3 + 1)) (λ i ↦ Nat.choose n (3 * i))) % 500 = 104 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_sum_modulo_l489_48902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_increasing_implies_decreasing_l489_48962

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def IncreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x < f y

def DecreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x < y → f x > f y

theorem even_increasing_implies_decreasing
  (f : ℝ → ℝ) (h1 : IsEven f) (h2 : IncreasingOn f (Set.Ici 0)) :
  DecreasingOn f (Set.Iic 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_increasing_implies_decreasing_l489_48962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_four_thirds_l489_48912

/-- Represents a cube with side length 2 -/
structure Cube :=
  (side_length : ℝ)
  (side_length_eq : side_length = 2)

/-- Represents the pyramid formed after cutting the cube -/
structure Pyramid :=
  (base_area : ℝ)
  (height : ℝ)
  (base_area_eq : base_area = 2)
  (height_eq : height = 2)

/-- The volume of a pyramid -/
noncomputable def pyramid_volume (p : Pyramid) : ℝ :=
  (1 / 3) * p.base_area * p.height

/-- Theorem stating that the volume of the specific pyramid is 4/3 -/
theorem pyramid_volume_is_four_thirds (c : Cube) (p : Pyramid) :
  pyramid_volume p = 4 / 3 := by
  sorry

#check pyramid_volume_is_four_thirds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_volume_is_four_thirds_l489_48912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_six_l489_48908

theorem cube_root_sum_equals_six : 
  (Real.rpow (27 - 18 * Real.sqrt 3) (1/3 : ℝ)) + (Real.rpow (27 + 18 * Real.sqrt 3) (1/3 : ℝ)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_sum_equals_six_l489_48908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_seven_l489_48931

def divisors_of_196 : List Nat := [2, 4, 7, 14, 28, 49, 98, 196]

def has_common_factor_greater_than_one (a b : Nat) : Prop :=
  ∃ (f : Nat), f > 1 ∧ f ∣ a ∧ f ∣ b

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.length = divisors_of_196.length ∧
  ∀ x ∈ arr, x ∈ divisors_of_196 ∧
  ∀ i, i < arr.length → has_common_factor_greater_than_one (arr[i]!) (arr[(i+1) % arr.length]!)

theorem sum_of_adjacent_to_seven (arr : List Nat) :
  is_valid_arrangement arr →
  let i := arr.indexOf 7
  (arr[(i-1) % arr.length]! + arr[(i+1) % arr.length]! = 147) :=
by
  intro h
  sorry

#eval divisors_of_196

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_adjacent_to_seven_l489_48931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_ratio_implies_n_sum_of_coefficients_is_4_pow_n_l489_48991

-- Define the expression
noncomputable def expr (x : ℝ) (n : ℕ) := (Real.sqrt x + 3 / x) ^ n

-- Define the sum of coefficients when x = 1
noncomputable def sum_of_coefficients (n : ℕ) := expr 1 n

-- Define the sum of binomial coefficients
def sum_of_binomial_coefficients (n : ℕ) := 2^n

-- Theorem statement
theorem expansion_ratio_implies_n (n : ℕ) :
  sum_of_coefficients n / sum_of_binomial_coefficients n = 64 → n = 6 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- Additional lemma to show that sum_of_coefficients n = 4^n when x = 1
theorem sum_of_coefficients_is_4_pow_n (n : ℕ) :
  sum_of_coefficients n = 4^n := by
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expansion_ratio_implies_n_sum_of_coefficients_is_4_pow_n_l489_48991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_to_athens_time_difference_l489_48926

/-- Represents a runner's journey with two halves of equal distance but different speeds -/
structure RunnerJourney where
  totalDistance : ℝ
  firstHalfSpeed : ℝ
  secondHalfTime : ℝ

/-- Calculates the time difference between the two halves of the journey -/
noncomputable def timeDifference (journey : RunnerJourney) : ℝ :=
  journey.secondHalfTime - (journey.totalDistance / (4 * journey.firstHalfSpeed))

/-- Theorem stating that for the specific journey described, the time difference is 8 hours -/
theorem marathon_to_athens_time_difference :
  ∃ (journey : RunnerJourney),
    journey.totalDistance = 40 ∧
    journey.secondHalfTime = 16 ∧
    timeDifference journey = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_to_athens_time_difference_l489_48926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_top_circle_number_l489_48924

theorem top_circle_number (p q r s t u v : ℕ) : 
  p ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
  q ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
  r ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
  s ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
  t ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
  u ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
  v ∈ ({1, 2, 3, 4, 5, 6, 7} : Set ℕ) ∧
  p + q + r + s + t + u + v = 28 ∧
  p + q + t = p + r + u ∧
  p + q + t = p + s + v ∧
  p + q + t = q + r + s ∧
  p + q + t = t + u + v →
  p = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_top_circle_number_l489_48924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_properties_l489_48922

/-- The ellipse C -/
noncomputable def C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- The circle O -/
noncomputable def O (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The line l with slope k -/
noncomputable def l (x y : ℝ) (k : ℝ) : Prop := y = k * x + Real.sqrt (4 * k^2 + 3)

/-- The point A -/
def A : ℝ × ℝ := (-2, 0)

/-- The point B -/
def B : ℝ × ℝ := (2, 0)

/-- M is to the left of N -/
def M_left_of_N (xM yM xN yN : ℝ) : Prop := xM < xN

/-- The slope of AM -/
noncomputable def k1 (xM yM : ℝ) : ℝ := yM / (xM + 2)

/-- The slope of BN -/
noncomputable def k2 (xN yN : ℝ) : ℝ := yN / (xN - 2)

/-- Main theorem -/
theorem ellipse_tangent_line_properties 
  (xM yM xN yN : ℝ) 
  (hC : C xM yM ∧ C xN yN) 
  (hO : O xM yM ∧ O xN yN) 
  (hl : l xM yM (1/2) ∧ l xN yN (1/2)) 
  (hMN : M_left_of_N xM yM xN yN) :
  (∃ d : ℝ, d = 4 * Real.sqrt 5 / 5 ∧ d = |yM - (1/2) * xM| / Real.sqrt (1 + (1/2)^2)) ∧ 
  k1 xM yM * k2 xN yN = -3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_tangent_line_properties_l489_48922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_train_crossing_time_l489_48955

/-- The time taken for a train to cross a platform -/
noncomputable def time_to_cross_platform (train_length platform_length : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  let train_speed := train_length / time_to_cross_pole
  let total_distance := train_length + platform_length
  total_distance / train_speed

/-- Theorem stating the time taken for a specific train to cross a specific platform -/
theorem specific_train_crossing_time :
  time_to_cross_platform 300 1162.5 8 = 39 := by
  -- Unfold the definition of time_to_cross_platform
  unfold time_to_cross_platform
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_train_crossing_time_l489_48955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l489_48972

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 3*x + 2)

-- Define the range of g
def range_g : Set ℝ := {y | ∃ x, g x = y}

-- Theorem stating the range of g
theorem range_of_g :
  range_g = {y | y ≤ -3 - Real.sqrt 8 ∨ y ≥ -3 + Real.sqrt 8} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_g_l489_48972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_frames_cost_l489_48959

-- Define the fixed sides of the triangle
def side1 : ℕ := 7
def side2 : ℕ := 3

-- Define the condition for the third side
def is_valid_third_side (x : ℕ) : Prop :=
  Odd x ∧ side2 < x ∧ x < side1 + side2

-- Define the cost per centimeter
def cost_per_cm : ℕ := 8

-- Theorem statement
theorem triangular_frames_cost :
  ∃! (valid_sides : Finset ℕ),
    (∀ x, x ∈ valid_sides ↔ is_valid_third_side x) ∧
    Finset.card valid_sides = 3 ∧
    cost_per_cm * (Finset.sum valid_sides (λ x ↦ x + side1 + side2)) = 408 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangular_frames_cost_l489_48959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_theorem_l489_48954

/-- Helper function to define the angle between median and angle bisector -/
noncomputable def angle_between_median_and_bisector (β : Real) : Real :=
  let tan_β := (2 * Real.tan (β/2)) / (1 - Real.tan (β/2)^2)
  let tan_median := tan_β / 2
  Real.arctan tan_median - β/2

/-- In a right triangle with an acute angle β satisfying tan(β/2) = 1/√2,
    the angle φ between the median and angle bisector drawn from this acute angle
    satisfies tan φ = √2/2 -/
theorem angle_between_median_and_bisector_theorem (β φ : Real) : 
  β > 0 → β < π/2 →  -- β is an acute angle
  Real.tan (β/2) = 1/Real.sqrt 2 →  -- given condition
  φ = angle_between_median_and_bisector β →  -- definition of φ
  Real.tan φ = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_median_and_bisector_theorem_l489_48954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_83_is_84_l489_48974

/-- Represents a circle with 100 equally spaced points, numbered from 1 to 100. -/
structure NumberedCircle where
  numbers : Fin 100 → Fin 100
  bijective : Function.Bijective numbers

/-- Checks if numbers less than k are evenly distributed across the diameter through k. -/
def evenlyDistributed (c : NumberedCircle) (k : Fin 100) : Prop :=
  ∀ m : Fin 100, m < k → 
    (c.numbers m < k ∧ (c.numbers m).val ≤ 50) ↔ 
    (c.numbers (Fin.add m 50) < k ∧ (c.numbers (Fin.add m 50)).val > 50)

/-- The main theorem stating that if numbers are evenly distributed for all k,
    then the number opposite to 83 is 84. -/
theorem opposite_of_83_is_84 (c : NumberedCircle) 
  (h : ∀ k : Fin 100, evenlyDistributed c k) :
  c.numbers (Fin.ofNat 33) = ⟨84, by norm_num⟩ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_of_83_is_84_l489_48974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_list_l489_48992

def median (l : List ℕ) : ℕ := sorry
def mean (l : List ℕ) : ℚ := sorry

theorem max_element_of_list (l : List ℕ) (h1 : l.length = 5) 
  (h2 : median l = 4) (h3 : mean l = 15) : 
  l.maximum ≤ 59 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_element_of_list_l489_48992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_arrives_12_minutes_before_jack_l489_48940

/-- The distance to the park in miles -/
noncomputable def distance_to_park : ℝ := 2

/-- Jill's speed in miles per hour -/
noncomputable def jill_speed : ℝ := 15

/-- Jack's speed in miles per hour -/
noncomputable def jack_speed : ℝ := 6

/-- Convert hours to minutes -/
noncomputable def hours_to_minutes (hours : ℝ) : ℝ := hours * 60

/-- Calculate travel time in hours given distance and speed -/
noncomputable def travel_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem jill_arrives_12_minutes_before_jack :
  hours_to_minutes (travel_time distance_to_park jack_speed - travel_time distance_to_park jill_speed) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jill_arrives_12_minutes_before_jack_l489_48940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_min_value_S_lambda_value_l489_48970

noncomputable section

-- Define the function f(x) = x ln x
def f (x : ℝ) : ℝ := x * Real.log x

-- Define the function g(x) = e^(λx ln x) - f(x)
def g (lambda : ℝ) (x : ℝ) : ℝ := Real.exp (lambda * x * Real.log x) - f x

-- Define the function S(x₀) = x₀²/(2(ln x₀ + 1))
def S (x₀ : ℝ) : ℝ := x₀^2 / (2 * (Real.log x₀ + 1))

theorem min_value_f :
  ∃ (x : ℝ), x > 0 ∧ f x = -1/Real.exp 1 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x :=
sorry

theorem min_value_S :
  ∀ (x₀ : ℝ), x₀ > 1/Real.exp 1 →
  S x₀ ≥ 1/Real.exp 1 :=
sorry

theorem lambda_value :
  ∃ (lambda : ℝ), (∀ (x : ℝ), x > 0 → g lambda x ≥ 1) ∧ lambda = 1 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_min_value_S_lambda_value_l489_48970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cubic_with_positive_roots_and_negative_derivative_roots_l489_48987

open Real

-- Define a cubic polynomial
def cubic_polynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + c * x + d

-- Define the derivative of a cubic polynomial
def cubic_derivative (a b c : ℝ) : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 2 * b * x + c

-- Theorem statement
theorem exists_cubic_with_positive_roots_and_negative_derivative_roots :
  ∃ (a b c d : ℝ), 
    (∀ x : ℝ, cubic_polynomial a b c d x = 0 → x > 0) ∧
    (∀ x : ℝ, cubic_derivative a b c x = 0 → x < 0) ∧
    (∃ x : ℝ, cubic_polynomial a b c d x = 0 ∧ 
      ∀ y : ℝ, y ≠ x → cubic_polynomial a b c d y ≠ 0) ∧
    (∃ x : ℝ, cubic_derivative a b c x = 0 ∧ 
      ∀ y : ℝ, y ≠ x → cubic_derivative a b c y ≠ 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_cubic_with_positive_roots_and_negative_derivative_roots_l489_48987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_correct_l489_48946

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 + 12 * x - 9 * y - 27 = 0

-- Define the area of the circle
noncomputable def circle_area : ℝ := 15.25 * Real.pi

-- Theorem statement
theorem circle_area_is_correct :
  ∃ (h k r : ℝ), (∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧
  circle_area = Real.pi * r^2 := by
  sorry

#check circle_area_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_is_correct_l489_48946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drains_in_14_hours_l489_48939

/-- Represents the time it takes for a leak to drain a full tank, given the fill times with and without the leak. -/
noncomputable def leak_drain_time (fill_time_no_leak : ℝ) (fill_time_with_leak : ℝ) : ℝ :=
  let pump_rate := 1 / fill_time_no_leak
  let combined_rate := 1 / fill_time_with_leak
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate

/-- Theorem stating that given the specific fill times, the leak will drain the tank in 14 hours. -/
theorem leak_drains_in_14_hours :
  leak_drain_time 2 (7/3) = 14 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leak_drains_in_14_hours_l489_48939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_l489_48929

/-- The time (in days) it takes for workers a and b together to complete a task -/
noncomputable def combined_time : ℝ := 20

/-- The time (in days) it takes for worker a alone to complete a task -/
noncomputable def a_alone_time : ℝ := 30

/-- The efficiency of worker a (portion of task completed per day) -/
noncomputable def efficiency_a : ℝ := 1 / a_alone_time

/-- The combined efficiency of workers a and b (portion of task completed per day) -/
noncomputable def combined_efficiency : ℝ := 1 / combined_time

/-- The efficiency of worker b (portion of task completed per day) -/
noncomputable def efficiency_b : ℝ := combined_efficiency - efficiency_a

/-- The theorem stating the ratio of efficiencies -/
theorem efficiency_ratio : efficiency_a / efficiency_b = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_efficiency_ratio_l489_48929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_formed_equals_calcium_carbonate_reacted_l489_48917

/-- Represents a chemical species in a reaction -/
inductive Species
| HCl
| CaCO3
| CaCl2
| CO2
| H2O

/-- Represents the number of moles of a species -/
def Moles := ℕ

instance : OfNat Moles n where
  ofNat := n

/-- Represents a chemical reaction with reactants and products -/
structure Reaction where
  reactants : Species → Moles
  products : Species → Moles

/-- The given balanced reaction -/
def balancedReaction : Reaction :=
  { reactants := λ s => match s with
    | Species.HCl => 2
    | Species.CaCO3 => 1
    | _ => 0
  , products := λ s => match s with
    | Species.CaCl2 => 1
    | Species.CO2 => 1
    | Species.H2O => 1
    | _ => 0
  }

/-- The actual reaction that occurred -/
def actualReaction : Reaction :=
  { reactants := λ s => match s with
    | Species.HCl => 6
    | Species.CaCO3 => 3
    | _ => 0
  , products := λ s => match s with
    | Species.CaCl2 => 3
    | Species.CO2 => 3
    | _ => 0  -- H2O is unknown
  }

theorem water_formed_equals_calcium_carbonate_reacted :
  actualReaction.products Species.H2O = actualReaction.reactants Species.CaCO3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_formed_equals_calcium_carbonate_reacted_l489_48917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_temperature_l489_48981

/-- Represents the temperatures for each day of the week --/
structure WeekTemperatures where
  monday : ℚ
  tuesday : ℚ
  wednesday : ℚ
  thursday : ℚ
  friday : ℚ

/-- The average temperature for a given set of days --/
def average (temps : List ℚ) : ℚ :=
  (temps.sum) / temps.length

theorem monday_temperature (w : WeekTemperatures) :
  average [w.monday, w.tuesday, w.wednesday, w.thursday] = 48 →
  average [w.tuesday, w.wednesday, w.thursday, w.friday] = 46 →
  (w.monday = 40 ∨ w.tuesday = 40 ∨ w.wednesday = 40 ∨ w.thursday = 40 ∨ w.friday = 40) →
  w.friday = 32 →
  w.monday = 40 := by
  sorry

#eval average [40, 50, 60, 70]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_temperature_l489_48981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_power_eq_identity_l489_48989

variable (p : ℕ) [Fact (Nat.Prime p)] [Fact (p ≥ 3)]
variable (A : Matrix (Fin p) (Fin p) ℂ)

theorem not_power_eq_identity
  (h_trace : Matrix.trace A = 0)
  (h_det : Matrix.det (A - 1) ≠ 0) :
  A ^ p ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_power_eq_identity_l489_48989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_bound_l489_48943

/-- A polygon inside a square -/
structure PolygonInSquare where
  area : ℝ
  area_nonneg : area ≥ 0

/-- A square with polygons inside -/
structure SquareWithPolygons where
  side_length : ℝ
  polygons : List PolygonInSquare
  side_length_positive : side_length > 0

/-- The theorem to be proved -/
theorem intersection_area_bound (s : SquareWithPolygons) 
  (h1 : s.side_length = 2)
  (h2 : s.polygons.length = 7)
  (h3 : ∀ p, p ∈ s.polygons → p.area ≥ 1) :
  ∃ p1 p2, p1 ∈ s.polygons ∧ p2 ∈ s.polygons ∧ p1 ≠ p2 ∧ 
  (∃ intersection_area : ℝ, intersection_area ≥ 1/7) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_bound_l489_48943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_and_tan_of_alpha_l489_48961

-- Define the angle α and its terminal point coordinates
noncomputable def α : ℝ := Real.arctan (12/5)
noncomputable def x : ℝ := -5/13
noncomputable def y : ℝ := 12/13

-- State the theorem
theorem sin_and_tan_of_alpha :
  (Real.sin α = 12/13) ∧ (Real.tan α = -12/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_and_tan_of_alpha_l489_48961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_max_l489_48969

/-- Given a triangle ABC where a, b, c are the lengths of the sides opposite to angles A, B, C
    respectively, and the height to side BC is a/2, the maximum value of c/b is √5. -/
theorem triangle_side_ratio_max (a b c : ℝ) (h : ℝ) : 
  a > 0 → b > 0 → c > 0 → h = a / 2 → h ≤ b → 
  (∃ (A B C : Real), 0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧ A + B + C = Real.pi ∧
    a = b * Real.sin A ∧ b = c * Real.sin B ∧ c = a * Real.sin C) →
  (∀ (c' b' : ℝ), c' > 0 → b' > 0 → 
    (∃ (A' B' C' : Real), 0 < A' ∧ A' < Real.pi ∧ 0 < B' ∧ B' < Real.pi ∧ 0 < C' ∧ C' < Real.pi ∧ A' + B' + C' = Real.pi ∧
      a = b' * Real.sin A' ∧ b' = c' * Real.sin B' ∧ c' = a * Real.sin C' ∧ h ≤ b') →
    c' / b' ≤ c / b) →
  c / b = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_ratio_max_l489_48969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_area_equals_triangle_area_specific_triangle_area_l489_48964

/-- The area of a triangle given its side lengths -/
noncomputable def triangleArea (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- The area of the union of a triangle and its 180° rotation around its centroid -/
noncomputable def unionAreaWithRotation (a b c : ℝ) : ℝ :=
  triangleArea a b c

theorem union_area_equals_triangle_area (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (htri : a + b > c ∧ b + c > a ∧ c + a > b) :
  unionAreaWithRotation a b c = triangleArea a b c := by
  -- The proof goes here
  sorry

/-- The specific case for the given triangle -/
theorem specific_triangle_area :
  unionAreaWithRotation 8 17 15 = 60 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_area_equals_triangle_area_specific_triangle_area_l489_48964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_adjacent_standing_probability_l489_48911

-- Define the number of people
def n : ℕ := 8

-- Define a function to calculate the number of valid arrangements
def valid_arrangements : ℕ → ℕ
| 0 => 1
| 1 => 2
| m+2 => valid_arrangements (m+1) + valid_arrangements m

-- Define the probability of no two adjacent people standing
def probability : ℚ := (valid_arrangements n : ℚ) / (2^n : ℚ)

-- State the theorem
theorem no_adjacent_standing_probability :
  probability = 47 / 256 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_adjacent_standing_probability_l489_48911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l489_48949

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 
  Real.sqrt ((h.a^2 + h.b^2) / h.a^2)

/-- Theorem: If one asymptote of the hyperbola is perpendicular to the line 3x - y + 5 = 0,
    then its eccentricity is √10/3 -/
theorem hyperbola_eccentricity (h : Hyperbola) 
    (asymptote_perp : h.b / h.a = 1 / 3) : 
    eccentricity h = Real.sqrt 10 / 3 := by
  sorry

#check hyperbola_eccentricity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l489_48949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l489_48953

/-- Given a circle, its tangent line, and a hyperbola, if the tangent line intersects
    the hyperbola at two points, then the eccentricity of the hyperbola is greater than 2. -/
theorem hyperbola_eccentricity_range 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h_intersect : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    (x₁ - 1)^2 + y₁^2 = 3/4 ∧
    (x₂ - 1)^2 + y₂^2 = 3/4 ∧
    ∃ (k : ℝ), y₁ = k * x₁ ∧ y₂ = k * x₂ ∧
    x₁^2 / a^2 - y₁^2 / b^2 = 1 ∧
    x₂^2 / a^2 - y₂^2 / b^2 = 1) :
  Real.sqrt (1 + b^2 / a^2) > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_range_l489_48953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l489_48990

-- Define the ellipse C
def C (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define the line that intersects C
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 1

-- Define the distance function
noncomputable def dist (x₁ y₁ x₂ y₂ : ℝ) : ℝ := Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

-- Main theorem
theorem ellipse_properties :
  -- C is the locus of points whose sum of distances from (0, √3) and (0, -√3) is 4
  (∀ x y : ℝ, C x y ↔ dist x y 0 (Real.sqrt 3) + dist x y 0 (-Real.sqrt 3) = 4) ∧
  -- When OA ⊥ OB, k = ±1/2
  (∀ k x₁ y₁ x₂ y₂ : ℝ, 
    C x₁ y₁ ∧ C x₂ y₂ ∧ Line k x₁ y₁ ∧ Line k x₂ y₂ ∧ 
    (x₁ * x₂ + y₁ * y₂ = 0) → (k = 1/2 ∨ k = -1/2)) ∧
  -- When OA ⊥ OB, |AB| = 4√65/17
  (∀ x₁ y₁ x₂ y₂ : ℝ,
    C x₁ y₁ ∧ C x₂ y₂ ∧ Line (1/2) x₁ y₁ ∧ Line (1/2) x₂ y₂ ∧
    (x₁ * x₂ + y₁ * y₂ = 0) → dist x₁ y₁ x₂ y₂ = 4 * Real.sqrt 65 / 17) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l489_48990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l489_48933

/-- The ellipse C defined by x^2/5 + y^2 = 1 -/
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

/-- The upper vertex of the ellipse C -/
def B : ℝ × ℝ := (0, 1)

/-- A point on the ellipse C -/
def P (p : C) : ℝ × ℝ := p

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- The theorem stating the maximum distance between P and B -/
theorem max_distance_PB :
  ∃ (max_dist : ℝ), max_dist = 5/2 ∧
  ∀ (p : C), distance (P p) B ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l489_48933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l489_48921

/-- A line in the form y = kx + 1 -/
structure Line where
  k : ℝ

/-- A circle with equation (x-1)² + (y+1)² = 12 -/
def Circle : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + (p.2 + 1)^2 = 12}

/-- The chord length of the intersection between a line and the circle -/
noncomputable def chordLength (l : Line) : ℝ :=
  2 * Real.sqrt ((8 - 4*l.k + 11*l.k^2) / (1 + l.k^2))

theorem shortest_chord_length :
  ∃ l₀ : Line, ∀ l' : Line, chordLength l₀ ≤ chordLength l' ∧ chordLength l₀ = 2 * Real.sqrt 7 := by
  sorry

#check shortest_chord_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_l489_48921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l489_48948

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  (Real.sqrt (2 * a + b / c) = 2 * a * Real.sqrt (b / c)) ↔ 
  (c = b * (4 * a^2 - 1) / (2 * a)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equality_condition_l489_48948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_same_direction_iff_magnitude_sum_eq_sum_magnitude_l489_48997

/-- Two non-zero vectors are parallel with the same direction if and only if 
    the magnitude of their sum equals the sum of their magnitudes -/
theorem parallel_same_direction_iff_magnitude_sum_eq_sum_magnitude 
  {V : Type*} [NormedAddCommGroup V] [Module ℝ V] {a b : V} (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ (k : ℝ), k > 0 ∧ b = k • a ↔ ‖a + b‖ = ‖a‖ + ‖b‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_same_direction_iff_magnitude_sum_eq_sum_magnitude_l489_48997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_y_l489_48928

/-- Given a triangle ABD where ∠ABC is a straight angle and ∠CBD = 103°, ∠BAD = 34°, 
    prove that the measure of ∠ABD is 69°. -/
theorem angle_measure_y (ABC CBD BAD ABD : Real) : 
  ABC = 180 → CBD = 103 → BAD = 34 → ABD = 69 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_measure_y_l489_48928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l489_48967

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if cos²A - cos²B + sin²C = sinB * sinC = 1/4 and the area is √3, then a = 2√3 -/
theorem triangle_side_length (a b c A B C : ℝ) : 
  (Real.cos A)^2 - (Real.cos B)^2 + (Real.sin C)^2 = (Real.sin B) * (Real.sin C) → 
  (Real.sin B) * (Real.sin C) = 1/4 →
  (1/2) * b * c * (Real.sin A) = Real.sqrt 3 →
  a = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l489_48967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_equation_l489_48982

/-- The slope of the regression line -/
def regression_slope : ℝ := 2.1

/-- The x-coordinate of the center point -/
def center_x : ℝ := 3

/-- The y-coordinate of the center point -/
def center_y : ℝ := 4

/-- The equation of a line passing through (center_x, center_y) with slope 'regression_slope' -/
def regression_line (x : ℝ) : ℝ := regression_slope * (x - center_x) + center_y

/-- Theorem stating that the regression line equation is equivalent to ŷ = 2.1x - 2.3 -/
theorem regression_line_equation : 
  ∀ x, regression_line x = 2.1 * x - 2.3 := by
  intro x
  unfold regression_line regression_slope center_x center_y
  ring
  -- The proof is completed by ring tactic, no need for sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_equation_l489_48982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l489_48976

theorem price_change_theorem (initial_price : ℝ) (initial_price_pos : initial_price > 0) :
  let price_after_increase := initial_price * (1 + 0.36)
  let price_after_first_discount := price_after_increase * (1 - 0.10)
  let final_price := price_after_first_discount * (1 - 0.15)
  let percentage_change := (final_price - initial_price) / initial_price * 100
  ∃ ε > 0, |percentage_change - 4.04| < ε := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_change_theorem_l489_48976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_inclination_angle_symmetric_point_theorem_l489_48942

-- Define a line by its direction vector
structure Line where
  direction : ℝ × ℝ

-- Define inclination angle
noncomputable def inclinationAngle (l : Line) : ℝ := Real.arctan (l.direction.2 / l.direction.1)

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a function to find the symmetric point
noncomputable def symmetricPoint (p : Point) (m : ℝ) (b : ℝ) : Point :=
  { x := sorry, y := sorry }

theorem direction_vector_inclination_angle :
  let l : Line := { direction := (3, Real.sqrt 3) }
  inclinationAngle l = π / 6 := by sorry

theorem symmetric_point_theorem :
  let p : Point := { x := 0, y := 2 }
  let symPoint := symmetricPoint p 1 1  -- y = x + 1 has slope 1 and y-intercept 1
  symPoint.x = 1 ∧ symPoint.y = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_inclination_angle_symmetric_point_theorem_l489_48942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_duckling_distance_l489_48963

theorem duckling_distance (pond : Set (ℝ × ℝ)) (ducklings : Finset (ℝ × ℝ)) : 
  (∃ c : ℝ × ℝ, ∃ r : ℝ, r = 5 ∧ pond = {p : ℝ × ℝ | (p.1 - c.1)^2 + (p.2 - c.2)^2 ≤ r^2}) →
  (ducklings.card = 6) →
  (∀ d : ℝ × ℝ, d ∈ ducklings → d ∈ pond) →
  (∃ d1 d2 : ℝ × ℝ, d1 ∈ ducklings ∧ d2 ∈ ducklings ∧ d1 ≠ d2 ∧ 
    (d1.1 - d2.1)^2 + (d1.2 - d2.2)^2 ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_duckling_distance_l489_48963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_solutions_l489_48975

/-- Definition of the sequence a_n -/
def a : ℕ → ℝ → ℝ
  | 0, s => s  -- Add case for n = 0
  | 1, s => s
  | (n+2), s => (a (n+1) s)^2 - 2

/-- Theorem stating that if solutions exist, they are distinct -/
theorem distinct_solutions (s : ℝ) :
  ∀ m n : ℕ, m ≠ n → (a m s = s ∧ a n s = s) → False := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_solutions_l489_48975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabola_and_line_l489_48999

-- Define the parabola function
def f (x : ℝ) : ℝ := -x^2

-- Define the line function
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the area enclosed by the two curves
noncomputable def enclosed_area : ℝ := ∫ x in Set.Icc (-3) 1, (g x - f x)

-- Theorem statement
theorem area_between_parabola_and_line :
  enclosed_area = 32/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_between_parabola_and_line_l489_48999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_cost_calculation_l489_48994

/-- Calculate the total cost in US dollars given the prices of items, discount, tax, and exchange rate --/
theorem shopping_cost_calculation 
  (high_heels_price : ℝ)
  (ballet_slippers_price : ℝ)
  (purse_price : ℝ)
  (scarf_price : ℝ)
  (high_heels_discount : ℝ)
  (sales_tax : ℝ)
  (exchange_rate : ℝ)
  (h1 : high_heels_price = 66)
  (h2 : ballet_slippers_price = 2/3 * high_heels_price)
  (h3 : purse_price = 49.5)
  (h4 : scarf_price = 27.5)
  (h5 : high_heels_discount = 0.1)
  (h6 : sales_tax = 0.075)
  (h7 : exchange_rate = 1 / 0.85) :
  ∃ (total_cost_usd : ℝ), 
    total_cost_usd = 
      (((high_heels_price * (1 - high_heels_discount) + ballet_slippers_price + purse_price + scarf_price) 
        * (1 + sales_tax)) * exchange_rate) ∧ 
    (abs (total_cost_usd - 228.11) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopping_cost_calculation_l489_48994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_digit_is_three_l489_48904

/-- Represents a base-n digit -/
def BaseDigit (n : ℕ) := {d : ℕ // d < n}

/-- Converts a number from its base-5 representation to its decimal value -/
def fromBase5 (a b c : BaseDigit 5) : ℕ := 25 * a.val + 5 * b.val + c.val

/-- Converts a number from its base-8 representation to its decimal value -/
def fromBase8 (a b c : BaseDigit 8) : ℕ := 64 * c.val + 8 * b.val + a.val

/-- The main theorem stating that the middle digit in base 5 is 3 -/
theorem middle_digit_is_three 
  (a b c : BaseDigit 5) 
  (ha : a.val ≠ 0)  -- Ensuring it's a 3-digit number in base 5
  (hequal : ∃ (a' b' c' : BaseDigit 8), fromBase5 a b c = fromBase8 c' b' a') :
  b.val = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_digit_is_three_l489_48904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_and_triangle_area_l489_48979

-- Define the equations
def eq1 (x y : ℝ) : Prop := y = x^2 + 6*x + 5
def eq2 (x y : ℝ) : Prop := y = x^2 - 4*x + 12

-- Define points A, B, and O
def A : ℝ × ℝ := (-3, -4)
def B : ℝ × ℝ := (2, 8)
def O : ℝ × ℝ := (0, 0)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the area function for a triangle given three points
noncomputable def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  (1/2) * abs (p1.1*(p2.2 - p3.2) + p2.1*(p3.2 - p1.2) + p3.1*(p1.2 - p2.2))

-- Theorem statement
theorem vertex_distance_and_triangle_area :
  (eq1 A.1 A.2) ∧ (eq2 B.1 B.2) →
  (distance A B = 13) ∧ (triangleArea A B O = 8) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertex_distance_and_triangle_area_l489_48979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_S_and_complement_l489_48966

def S : Set Nat :=
  {p | ∃ n : Nat, n > 0 ∧ Nat.Prime p ∧ p ∣ (2^(n^2 + 1) - 3^n)}

def P : Set Nat := {p | Nat.Prime p}

theorem infinite_S_and_complement :
  (Set.Infinite S) ∧ (Set.Infinite (P \ S)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_S_and_complement_l489_48966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lower_grades_is_three_l489_48968

/-- Given the following conditions:
  * There are 60 total tests in a semester
  * The goal is to achieve a "B" grade in at least 70% of the tests
  * 25 "B" grades were achieved in the first 40 tests
  Prove that the maximum number of remaining tests where a score lower than "B" can be obtained is 3 -/
def max_lower_grades (total_tests : ℕ) (goal_percentage : ℚ) 
  (completed_tests : ℕ) (completed_b_grades : ℕ) : ℕ :=
  let remaining_tests := total_tests - completed_tests
  let total_b_grades_needed := Int.ceil (goal_percentage * total_tests)
  let additional_b_grades_needed := total_b_grades_needed - completed_b_grades
  (remaining_tests - additional_b_grades_needed).toNat

#eval max_lower_grades 60 (70/100) 40 25

theorem max_lower_grades_is_three : 
  max_lower_grades 60 (70/100) 40 25 = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_lower_grades_is_three_l489_48968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_battle_station_staffing_l489_48956

theorem battle_station_staffing (total_candidates : ℕ) (unsuitable_candidates : ℕ) (positions : ℕ) :
  total_candidates = 30 →
  unsuitable_candidates = 15 →
  positions = 5 →
  (total_candidates - unsuitable_candidates).factorial / (total_candidates - unsuitable_candidates - positions).factorial = 360360 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_battle_station_staffing_l489_48956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_four_ones_in_five_rolls_l489_48920

/-- A fair, regular eight-sided die -/
def EightSidedDie : Type := Fin 8

/-- The probability of rolling a specific number on an eight-sided die -/
def prob_single_roll : ℚ := 1 / 8

/-- The number of rolls -/
def num_rolls : ℕ := 5

/-- The number of times we want to roll the number 1 -/
def target_rolls : ℕ := 4

/-- The probability of rolling the number 1 exactly four times in five rolls of a fair, regular eight-sided die -/
theorem prob_four_ones_in_five_rolls : 
  (Nat.choose num_rolls target_rolls : ℚ) * prob_single_roll^target_rolls * (1 - prob_single_roll)^(num_rolls - target_rolls) = 35 / 32768 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_four_ones_in_five_rolls_l489_48920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_eq_zero_l489_48930

/-- Given a function f such that f(x+1) = x^2 - 1 for all x, prove that f(2) = 0 -/
theorem f_of_two_eq_zero (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = x^2 - 1) : f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_two_eq_zero_l489_48930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_sets_l489_48995

def M (m : ℕ) : Set ℕ := {x | x = 5^m ∨ x = 2}
def N (m n : ℕ) : Set ℕ := {x | x = m ∨ x = n}

theorem union_of_sets (m n : ℕ) (h : M m ∩ N m n = {1}) : M m ∪ N m n = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_of_sets_l489_48995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_state_recurrence_l489_48934

/-- Represents a player in the card game -/
structure Player where
  id : Nat
  hand : Finset Nat

/-- Represents the state of the game at any given time -/
structure GameState (n : Nat) where
  players : Vector Player n
  h : n ≥ 3

/-- Simulates one exchange in the game -/
def exchange {n : Nat} (state : GameState n) : GameState n :=
  sorry

/-- Theorem stating that the game state after n-1 exchanges is the same as after 2n-1 exchanges -/
theorem game_state_recurrence (n : Nat) (h : n ≥ 3) :
  ∀ (initial_state : GameState n),
    (n - 1).iterate exchange initial_state = (2*n - 1).iterate exchange initial_state :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_state_recurrence_l489_48934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l489_48947

theorem order_of_expressions : 2^(1/2) > Real.log 3 / Real.log π ∧ Real.log 3 / Real.log π > Real.log (1/2) / Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_order_of_expressions_l489_48947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_1200_l489_48936

theorem divisors_of_1200 : 
  (Finset.filter (λ n : ℕ => n ∣ 1200 ∧ n ≤ 1200) (Finset.range 1201)).card = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_1200_l489_48936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangents_theorem_l489_48978

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Four circles touching externally in a cyclic manner -/
structure FourTouchingCircles where
  s₁ : Circle
  s₂ : Circle
  s₃ : Circle
  s₄ : Circle
  touch_externally : 
    (s₁.center.1 - s₂.center.1)^2 + (s₁.center.2 - s₂.center.2)^2 = (s₁.radius + s₂.radius)^2 ∧
    (s₂.center.1 - s₃.center.1)^2 + (s₂.center.2 - s₃.center.2)^2 = (s₂.radius + s₃.radius)^2 ∧
    (s₃.center.1 - s₄.center.1)^2 + (s₃.center.2 - s₄.center.2)^2 = (s₃.radius + s₄.radius)^2 ∧
    (s₄.center.1 - s₁.center.1)^2 + (s₄.center.2 - s₁.center.2)^2 = (s₄.radius + s₁.radius)^2

/-- A line in 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The common tangent of two circles -/
noncomputable def common_tangent (c₁ c₂ : Circle) : Line := sorry

/-- Predicate for four lines intersecting at one point -/
def intersect_at_one_point (l₁ l₂ l₃ l₄ : Line) : Prop := sorry

/-- Predicate for four lines being tangent to one circle -/
def tangent_to_one_circle (l₁ l₂ l₃ l₄ : Line) : Prop := sorry

/-- The main theorem -/
theorem four_tangents_theorem (circles : FourTouchingCircles) : 
  let t₁ := common_tangent circles.s₁ circles.s₂
  let t₂ := common_tangent circles.s₂ circles.s₃
  let t₃ := common_tangent circles.s₃ circles.s₄
  let t₄ := common_tangent circles.s₄ circles.s₁
  intersect_at_one_point t₁ t₂ t₃ t₄ ∨ tangent_to_one_circle t₁ t₂ t₃ t₄ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_tangents_theorem_l489_48978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_show_problem_l489_48914

theorem dog_show_problem (n : ℕ) (h1 : n = 24) :
  ∃ (x : ℕ), x ∈ ({1, n} : Set ℕ) ∧
  ∃ (k : ℕ), k ∈ Finset.range n ∧ k ≠ x ∧
  (↑((n * (n + 1)) / 2 - x) : ℚ) / 23 = k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_show_problem_l489_48914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_gathering_l489_48983

/-- Represents the state of chips on a circular board with 6 sectors -/
def BoardState := Fin 6 → ℕ

/-- The initial state of the board where each sector has one chip -/
def initial_state : BoardState := λ _ => 1

/-- Calculates the sum of sector numbers multiplied by the number of chips in each sector -/
def sector_sum (state : BoardState) : ℕ :=
  Finset.sum (Finset.univ : Finset (Fin 6)) (λ i => i.val.succ * state i)

/-- Represents a valid move: shifting two chips to adjacent sectors -/
def is_valid_move (before after : BoardState) : Prop :=
  ∃ (i j : Fin 6), 
    (i ≠ j) ∧
    (after i = before i - 1) ∧
    (after j = before j - 1) ∧
    (after ((i + 1) % 6) = before ((i + 1) % 6) + 1) ∧
    (after ((j + 1) % 6) = before ((j + 1) % 6) + 1) ∧
    (∀ k : Fin 6, k ≠ i ∧ k ≠ j ∧ k ≠ (i + 1) % 6 ∧ k ≠ (j + 1) % 6 → after k = before k)

/-- A state where all chips are in one sector -/
def all_in_one_sector (state : BoardState) : Prop :=
  ∃ i : Fin 6, state i = 6 ∧ ∀ j : Fin 6, j ≠ i → state j = 0

theorem impossibility_of_gathering : 
  ¬∃ (final : BoardState), 
    (∃ (moves : List (BoardState × BoardState)), 
      moves.head?.map (Prod.fst) = some initial_state ∧
      moves.getLast?.map (Prod.snd) = some final ∧
      ∀ (move : BoardState × BoardState), move ∈ moves → is_valid_move move.fst move.snd) ∧
    all_in_one_sector final := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossibility_of_gathering_l489_48983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l489_48971

-- Define birthday as a function from ℕ to ℕ
def birthday : ℕ → ℕ := sorry

theorem birthday_problem (people : ℕ) (days : ℕ) :
  people = 367 ∧ days = 366 → 
  ∃ (d : ℕ), d ≤ days ∧ 
  (∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ p1 ≤ people ∧ p2 ≤ people ∧ 
   birthday p1 = d ∧ birthday p2 = d) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_birthday_problem_l489_48971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_pyramid_l489_48916

-- Define the triangular pyramid SABC
structure TriangularPyramid where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  angleSAB : ℝ
  angleSCtoSAB : ℝ

-- Define the conditions of the problem
def validPyramid (p : TriangularPyramid) : Prop :=
  p.angleSAB = 30 * Real.pi / 180 ∧
  p.angleSCtoSAB = 45 * Real.pi / 180 ∧
  p.SA + p.SB + p.SC = 9

-- Define the volume function
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  (1 / 6) * (Real.sqrt 2 / 2) * p.SA * p.SB * p.SC

-- Theorem statement
theorem max_volume_of_pyramid (p : TriangularPyramid) 
  (h : validPyramid p) : 
  volume p ≤ 9 * Real.sqrt 2 / 4 := by
  sorry

#check max_volume_of_pyramid

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_of_pyramid_l489_48916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l489_48951

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a * Real.log x

noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := f a x + (2 + a) * Real.log x - 2 * (b - 1) * x

-- State the theorem
theorem problem_solution (a b : ℝ) (x₁ x₂ : ℝ) :
  (∀ x ∈ Set.Icc 3 5, ∀ y ∈ Set.Icc 3 5, x < y → f a x > f a y) →
  (x₁ < x₂) →
  (∀ x, g a b x ≤ g a b x₁ ∨ g a b x ≤ g a b x₂) →
  (b ≥ 7/2) →
  (a ≥ 50 ∧ g a b x₁ - g a b x₂ ≥ 15/4 - 4 * Real.log 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l489_48951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_vectors_l489_48998

def vector_a : ℝ × ℝ := (-1, 1)

theorem magnitude_of_sum_vectors (b : ℝ × ℝ) 
  (h1 : Real.cos (3 * Real.pi / 4) = -Real.sqrt 2 / 2)
  (h2 : ‖b‖ = 2) :
  ‖2 • vector_a + b‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_sum_vectors_l489_48998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_directrix_distance_l489_48905

/-- Given an ellipse with equation x²/3 + y²/2 = 1, if a point P on the ellipse
    has distance √3/2 to the left focus, then the distance from P to the right
    directrix is 9/2. -/
theorem ellipse_directrix_distance (x y : ℝ) :
  x^2 / 3 + y^2 / 2 = 1 →
  ∃ (a e c : ℝ),
    a > 0 ∧ 
    0 < e ∧ e < 1 ∧
    c > 0 ∧
    a^2 / c = 3 ∧
    ∃ (x₀ : ℝ),
      x₀^2 / 3 + y^2 / 2 = 1 ∧
      a + e * x₀ = Real.sqrt 3 / 2 ∧
      3 + 3 / 2 = 9 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_directrix_distance_l489_48905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l489_48901

/-- The length of the direct path from Uphill to Middleton -/
def x : ℝ := sorry

/-- The length of the direct path from Middleton to Downend -/
def y : ℝ := sorry

/-- The length of the direct path from Downend to Uphill -/
def z : ℝ := sorry

/-- The detour from Downend to Uphill via Middleton is 1 km longer than the direct path -/
axiom condition1 : x + y = z + 1

/-- The detour from Downend to Middleton via Uphill is 5 km longer than the direct path -/
axiom condition2 : x + z = y + 5

/-- The detour from Uphill to Middleton via Downend is 7 km longer than the direct path -/
axiom condition3 : y + z = x + 7

/-- The shortest direct path between any two of the three villages is 3 km long -/
theorem shortest_path : min x (min y z) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l489_48901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_shift_rhombus_l489_48937

/-- A line in the xy-plane defined by y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- A point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For a line y = kx + 3 intersecting the x-axis at point A and y-axis at point B,
    if the line is shifted 5 units to the right and the boundary of the region swept by
    line segment AB forms a rhombus, then k = ± 3/4 -/
theorem line_shift_rhombus (l : Line) (A B : Point) :
  l.b = 3 ∧
  A.y = 0 ∧
  B.x = 0 ∧
  A.x * l.k + l.b = A.y ∧
  B.x * l.k + l.b = B.y ∧
  distance A B = 5 →
  l.k = 3/4 ∨ l.k = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_shift_rhombus_l489_48937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_l489_48906

-- Define the parametric equations
noncomputable def x (θ : Real) : Real := 2 + (Real.sin θ) ^ 2
noncomputable def y (θ : Real) : Real := (Real.sin θ) ^ 2

-- State the theorem
theorem parametric_to_ordinary :
  ∀ θ : Real, ∃ x y : Real, 
    x = 2 + (Real.sin θ) ^ 2 ∧ 
    y = (Real.sin θ) ^ 2 ∧ 
    y = x - 2 ∧ 
    2 ≤ x ∧ x ≤ 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parametric_to_ordinary_l489_48906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_play_result_l489_48935

/-- Represents a player's strategy in the number removal game. -/
structure Strategy where
  remove : (turn : Nat) → (remaining : List Nat) → List Nat

/-- The game state after each turn. -/
structure GameState where
  remaining : List Nat
  turn : Nat

/-- Applies a strategy to the current game state. -/
def applyStrategy (s : Strategy) (g : GameState) : GameState :=
  { remaining := s.remove g.turn g.remaining,
    turn := g.turn + 1 }

/-- Represents the full game with both players' strategies. -/
def fullGame (s1 s2 : Strategy) : Nat → GameState → GameState
  | 0, g => g
  | n+1, g => fullGame s1 s2 n (applyStrategy (if n % 2 = 0 then s1 else s2) g)

/-- The initial game state with numbers from 0 to 1024. -/
def initialState : GameState :=
  { remaining := List.range 1025,
    turn := 0 }

/-- Optimal strategies for both players. -/
noncomputable def optimalStrategy1 : Strategy := sorry
noncomputable def optimalStrategy2 : Strategy := sorry

theorem optimal_play_result :
  let finalState := fullGame optimalStrategy1 optimalStrategy2 10 initialState
  let finalNumbers := finalState.remaining
  finalNumbers.length = 2 ∧ 
  (finalNumbers.maximum?.getD 0) - (finalNumbers.minimum?.getD 0) = 32 := by
  sorry

#check optimal_play_result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_play_result_l489_48935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_exists_l489_48915

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of five points in a plane -/
def FivePoints : Set Point := sorry

/-- Predicate to check if three points are collinear -/
def areCollinear (p q r : Point) : Prop := sorry

/-- Predicate to check if four points form a convex quadrilateral -/
def isConvexQuadrilateral (p q r s : Point) : Prop := sorry

/-- Theorem: Given five points in a plane where no three are collinear,
    four of these points form the vertices of a convex quadrilateral -/
theorem convex_quadrilateral_exists :
  (∀ (p q r : Point), p ∈ FivePoints → q ∈ FivePoints → r ∈ FivePoints →
    p ≠ q → q ≠ r → p ≠ r → ¬areCollinear p q r) →
  ∃ (p q r s : Point), p ∈ FivePoints ∧ q ∈ FivePoints ∧ r ∈ FivePoints ∧ s ∈ FivePoints ∧
    p ≠ q ∧ q ≠ r ∧ r ≠ s ∧ s ≠ p ∧ isConvexQuadrilateral p q r s :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_convex_quadrilateral_exists_l489_48915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l489_48980

/-- Calculates the final amount after compound interest is applied -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Rounds a real number to the nearest integer -/
noncomputable def round_to_nearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem investment_growth :
  let principal : ℝ := 8000
  let rate : ℝ := 0.05
  let time : ℕ := 3
  round_to_nearest (compound_interest principal rate time) = 9250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l489_48980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l489_48988

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := |x| / (Real.sqrt (1 + x^2) * Real.sqrt (4 + x^2))

-- State the theorem
theorem f_max_value : 
  (∀ x : ℝ, f x ≤ 1/3) ∧ (∃ x : ℝ, f x = 1/3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l489_48988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_product_252_l489_48996

theorem gcd_lcm_product_252 (a b : ℕ+) :
  (Nat.gcd a b) * (Nat.lcm a b) = 252 →
  ∃ S : Finset ℕ, (∀ x ∈ S, ∃ a b : ℕ+, (Nat.gcd a b) * (Nat.lcm a b) = 252 ∧ Nat.gcd a b = x) ∧
                  S.card = 4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_lcm_product_252_l489_48996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l489_48941

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = e^x + e^(-x) + ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + Real.exp (-x) + a * x

theorem even_function_implies_a_zero :
  ∀ a : ℝ, IsEven (f a) → a = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_implies_a_zero_l489_48941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_perimeter_and_shape_l489_48965

namespace TriangleProof

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

variable (t : Triangle)

-- Define the given condition
def condition (t : Triangle) : Prop :=
  (t.a^2 + t.c^2 - t.b^2) * Real.tan t.B = Real.sqrt 3 * (t.b^2 + t.c^2 - t.a^2)

-- Theorem 1: Measure of angle A
theorem angle_A_measure (h : condition t) : t.A = π / 3 := by
  sorry

-- Theorem 2: Maximum perimeter and shape
theorem max_perimeter_and_shape (h : condition t) (h2 : t.a = 2) :
  ∃ L : ℝ, L = t.a + t.b + t.c ∧ L ≤ 6 ∧
   (L = 6 → t.a = t.b ∧ t.b = t.c) := by
  sorry

end TriangleProof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_measure_max_perimeter_and_shape_l489_48965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l489_48923

-- Define the hyperbola
noncomputable def hyperbola (x y a b : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop := 2*x - y = 0

-- Define eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt ((a^2 + b^2) / a^2)

theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : ∃ x y, hyperbola x y a b ∧ asymptote x y) :
  eccentricity a b = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l489_48923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_value_l489_48913

theorem cos_difference_value (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1/2) 
  (h2 : Real.sin α + Real.sin β = 1/3) : 
  Real.cos (α - β) = -59/72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_value_l489_48913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_range_l489_48977

-- Define the function S
noncomputable def S (t : ℝ) : ℝ :=
  if t < 1 then 3 * t else 4 * t - t^2

-- State the theorem
theorem S_range :
  ∀ t ∈ Set.Ioo 0 3, S t ∈ Set.Ioc 0 4 ∧
  ∀ y ∈ Set.Ioc 0 4, ∃ t ∈ Set.Ioo 0 3, S t = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_range_l489_48977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_x_values_l489_48900

theorem circle_point_x_values (x : ℝ) :
  let circle_diameter_start : ℝ × ℝ := (-7, 0)
  let circle_diameter_end : ℝ × ℝ := (23, 0)
  let point_on_circle : ℝ × ℝ := (x, 10)
  let center : ℝ × ℝ := ((circle_diameter_start.1 + circle_diameter_end.1) / 2, 0)
  let radius : ℝ := (circle_diameter_end.1 - circle_diameter_start.1) / 2
  ((point_on_circle.1 - center.1)^2 + (point_on_circle.2 - center.2)^2 = radius^2) →
  (x = 8 + 5 * Real.sqrt 5 ∨ x = 8 - 5 * Real.sqrt 5) :=
by
  -- Introduce the hypotheses
  intro h
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_point_x_values_l489_48900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_theorem_l489_48903

def count_triangles (k : ℕ) (n : ℕ → ℕ) : ℕ :=
  (Finset.sum (Finset.range k) (λ p =>
    Finset.sum (Finset.range k) (λ q =>
      Finset.sum (Finset.range k) (λ r =>
        if p < q ∧ q < r then n p * n q * n r else 0)))) +
  (Finset.sum (Finset.range k) (λ p =>
    Finset.sum (Finset.range k) (λ q =>
      if p < q then n p * (Nat.choose (n q) 2) + n q * (Nat.choose (n p) 2) else 0)))

theorem triangle_count_theorem (k : ℕ) (n : ℕ → ℕ) :
  count_triangles k n =
    (Finset.sum (Finset.range k) (λ p =>
      Finset.sum (Finset.range k) (λ q =>
        Finset.sum (Finset.range k) (λ r =>
          if p < q ∧ q < r then n p * n q * n r else 0)))) +
    (Finset.sum (Finset.range k) (λ p =>
      Finset.sum (Finset.range k) (λ q =>
        if p < q then n p * (Nat.choose (n q) 2) + n q * (Nat.choose (n p) 2) else 0))) :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_count_theorem_l489_48903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l489_48952

-- Define the train lengths in meters
def train1_length : ℝ := 100
def train2_length : ℝ := 200

-- Define the initial distance between trains in meters
def initial_distance : ℝ := 100

-- Define the train speeds in kilometers per hour
def train1_speed_kmph : ℝ := 54
def train2_speed_kmph : ℝ := 72

-- Convert km/h to m/s
noncomputable def kmph_to_ms (speed : ℝ) : ℝ := speed * (1000 / 3600)

-- Calculate the relative speed in m/s
noncomputable def relative_speed : ℝ := kmph_to_ms train1_speed_kmph + kmph_to_ms train2_speed_kmph

-- Calculate the total distance to be covered
def total_distance : ℝ := train1_length + train2_length + initial_distance

-- Calculate the time until the trains meet
noncomputable def meeting_time : ℝ := total_distance / relative_speed

-- Theorem statement
theorem trains_meet_time :
  ∃ ε > 0, |meeting_time - 11.43| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_meet_time_l489_48952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_sum_squared_l489_48950

theorem min_trig_sum_squared (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  (Real.tan x + 1 / Real.tan x)^2 + (1 / Real.cos x + 1 / Real.sin x)^2 ≥ 12 ∧
  ∃ y, 0 < y ∧ y < Real.pi / 2 ∧ (Real.tan y + 1 / Real.tan y)^2 + (1 / Real.cos y + 1 / Real.sin y)^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_trig_sum_squared_l489_48950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_theorem_l489_48960

-- Define the circle C
noncomputable def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 10 = 0

-- Define the line l
def l (x y c : ℝ) : Prop := x - y + c = 0

-- Define the distance function between a point (x, y) and the line l
noncomputable def distance_to_line (x y c : ℝ) : ℝ :=
  |x - y + c| / Real.sqrt 2

-- Statement of the theorem
theorem circle_line_distance_theorem (c : ℝ) :
  (∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    C x₁ y₁ ∧ C x₂ y₂ ∧ C x₃ y₃ ∧
    (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (x₃, y₃) ∧ (x₂, y₂) ≠ (x₃, y₃) ∧
    distance_to_line x₁ y₁ c = 2 * Real.sqrt 2 ∧
    distance_to_line x₂ y₂ c = 2 * Real.sqrt 2 ∧
    distance_to_line x₃ y₃ c = 2 * Real.sqrt 2) →
  c ∈ Set.Icc (-2) 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_distance_theorem_l489_48960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_d_and_power_of_two_l489_48927

theorem divisibility_by_d_and_power_of_two (d : ℕ) :
  (∃ n : ℕ, (∀ digit : ℕ, digit ∈ n.digits 10 → digit = 1 ∨ digit = 2) ∧
    (d * 2^1996 ∣ n)) ↔ d > 0 ∧ ¬(5 ∣ d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_d_and_power_of_two_l489_48927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l489_48938

/-- Represents an ellipse with semi-major axis a, semi-minor axis b, and semi-focal distance c -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_c_eq : c^2 = a^2 - b^2

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := e.c / e.a

/-- The left focus of the ellipse -/
def left_focus (e : Ellipse) : ℝ × ℝ := (-e.c, 0)

/-- The right focus of the ellipse -/
def right_focus (e : Ellipse) : ℝ × ℝ := (e.c, 0)

/-- The point P on the ellipse -/
noncomputable def point_p (e : Ellipse) : ℝ × ℝ := (e.a^2 / e.c, Real.sqrt 3 * e.b)

/-- Theorem: If the perpendicular bisector of PF₁ passes through F₂, then the eccentricity is √2/2 -/
theorem ellipse_eccentricity_sqrt_two_over_two (e : Ellipse) :
  let p := point_p e
  let f1 := left_focus e
  let f2 := right_focus e
  (∃ m : ℝ, (m * (p.1 - f1.1) = f2.1 - (p.1 + f1.1)/2) ∧
            (m * (p.2 - f1.2) = f2.2 - (p.2 + f1.2)/2) ∧
            m ≠ 0) →
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l489_48938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fractions_differ_by_one_are_integers_l489_48957

theorem fractions_differ_by_one_are_integers (a b : ℤ) : 
  a > 1 → b > 1 → |a / b - (a - 1) / (b - 1)| = 1 → Int.fract (a / b) = 0 ∧ Int.fract ((a - 1) / (b - 1)) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fractions_differ_by_one_are_integers_l489_48957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_sequence_l489_48973

def mySequence (z : ℝ) : List ℝ := [0*z, z, 2*z, 4*z, 8*z, 32*z]

theorem average_of_sequence (z : ℝ) :
  (List.sum (mySequence z)) / (List.length (mySequence z) : ℝ) = 47 * z / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_of_sequence_l489_48973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_line_in_plane_not_necessarily_parallel_to_plane_l489_48910

-- Define the basic structures
structure Line where

structure Plane where

-- Define the relationships
def parallel_lines (l1 l2 : Line) : Prop := sorry

def line_in_plane (l : Line) (p : Plane) : Prop := sorry

def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Theorem statement
theorem line_parallel_to_line_in_plane_not_necessarily_parallel_to_plane 
  (L1 L2 : Line) (P : Plane) 
  (h1 : line_in_plane L2 P) 
  (h2 : parallel_lines L1 L2) : 
  ¬ (∀ (L1 : Line) (P : Plane), parallel_line_plane L1 P) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_parallel_to_line_in_plane_not_necessarily_parallel_to_plane_l489_48910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kayak_rental_cost_is_16_l489_48945

/-- Represents the rental business scenario --/
structure RentalBusiness where
  canoe_cost : ℕ
  canoe_kayak_ratio : Rat
  total_revenue : ℕ
  canoe_kayak_difference : ℕ

/-- Calculates the cost of a kayak rental per day --/
def kayak_rental_cost (rb : RentalBusiness) : ℚ :=
  let kayaks : ℕ := rb.canoe_kayak_difference
  let canoes : ℕ := kayaks + rb.canoe_kayak_difference
  let canoe_revenue : ℕ := canoes * rb.canoe_cost
  let kayak_revenue : ℕ := rb.total_revenue - canoe_revenue
  (kayak_revenue : ℚ) / (kayaks : ℚ)

/-- Theorem stating that the kayak rental cost is $16 given the specific conditions --/
theorem kayak_rental_cost_is_16 :
  kayak_rental_cost (RentalBusiness.mk 11 (4 / 3) 460 5) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kayak_rental_cost_is_16_l489_48945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l489_48993

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 3) :
  (∀ x y : ℝ, x + y = 3 → (2:ℝ)^x + (2:ℝ)^y ≥ (2:ℝ)^a + (2:ℝ)^b) → 
  (2:ℝ)^a + (2:ℝ)^b = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_exponential_sum_l489_48993
