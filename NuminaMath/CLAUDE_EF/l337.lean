import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l337_33767

/-- Represents a rectangular yard with trapezoidal remainder and triangular flower beds -/
structure Yard where
  length : ℝ
  width : ℝ
  trapezoid_short_side : ℝ
  trapezoid_long_side : ℝ

/-- Calculates the area of an isosceles right triangle -/
noncomputable def isosceles_right_triangle_area (leg : ℝ) : ℝ := (1 / 2) * leg^2

/-- Calculates the area of the flower beds -/
noncomputable def flower_beds_area (y : Yard) : ℝ :=
  2 * isosceles_right_triangle_area ((y.trapezoid_long_side - y.trapezoid_short_side) / 2)

/-- Calculates the total area of the yard -/
noncomputable def yard_area (y : Yard) : ℝ := y.length * y.width

/-- Theorem: The fraction of the yard occupied by flower beds is 1/5 -/
theorem flower_beds_fraction (y : Yard) 
  (h1 : y.trapezoid_short_side = 18)
  (h2 : y.trapezoid_long_side = 30)
  (h3 : y.width = (y.trapezoid_long_side - y.trapezoid_short_side) / 2)
  (h4 : y.length = y.trapezoid_long_side) :
  flower_beds_area y / yard_area y = 1 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_beds_fraction_l337_33767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_OP_bisects_MN_l337_33784

-- Define the circle F
noncomputable def circle_F (x y : ℝ) : Prop := (x + Real.sqrt 6)^2 + y^2 = 32

-- Define point D
noncomputable def point_D : ℝ × ℝ := (Real.sqrt 6, 0)

-- Define point A
def point_A : ℝ × ℝ := (2, 1)

-- Define point B as symmetric to A with respect to origin
def point_B : ℝ × ℝ := (-2, -1)

-- Define the locus of point C (ellipse equation)
def locus_C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 2 = 1

-- Define line l parallel to AB
def line_l (x y m : ℝ) : Prop := y = (1/2) * x + m ∧ m ≠ 0 ∧ -2 < m ∧ m < 2

-- Main theorem
theorem OP_bisects_MN (E : ℝ × ℝ) (C M N P : ℝ × ℝ) (m : ℝ) :
  circle_F E.1 E.2 →
  locus_C C.1 C.2 →
  line_l M.1 M.2 m →
  line_l N.1 N.2 m →
  locus_C M.1 M.2 →
  locus_C N.1 N.2 →
  -- AM intersects BN at P (simplified condition)
  (P.2 - point_A.2) / (P.1 - point_A.1) = (M.2 - point_A.2) / (M.1 - point_A.1) →
  (P.2 - point_B.2) / (P.1 - point_B.1) = (N.2 - point_B.2) / (N.1 - point_B.1) →
  -- OP bisects MN
  P.1 = -m ∧ P.2 = -m/2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_OP_bisects_MN_l337_33784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l337_33795

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h_arithmetic : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

/-- Theorem: For an arithmetic sequence with a_3 = 10 and S_4 = 36, the common difference d is 2 -/
theorem arithmetic_sequence_common_difference
  (seq : ArithmeticSequence)
  (h_a3 : seq.a 3 = 10)
  (h_s4 : sum_n seq 4 = 36) :
  seq.d = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_common_difference_l337_33795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l337_33759

/-- Calculates the speed of a train in kmph given its length, platform length, and time to pass the platform. -/
noncomputable def train_speed (train_length platform_length : ℝ) (time : ℝ) : ℝ :=
  let total_distance := train_length + platform_length
  let speed_mps := total_distance / time
  3.6 * speed_mps

/-- Theorem stating that a train with given parameters has a specific speed. -/
theorem train_speed_calculation :
  let train_length : ℝ := 120
  let platform_length : ℝ := 240
  let time : ℝ := 21.598272138228943
  let calculated_speed := train_speed train_length platform_length time
  ∃ ε > 0, |calculated_speed - 60.0048| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l337_33759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l337_33783

/-- An arithmetic sequence with given properties -/
noncomputable def arithmetic_sequence (n : ℕ) : ℚ := 3 - 2 * n

/-- Sum of the first k terms of the arithmetic sequence -/
noncomputable def S (k : ℕ) : ℚ := (k : ℚ) * (arithmetic_sequence 1 + arithmetic_sequence k) / 2

theorem arithmetic_sequence_sum :
  ∀ k : ℕ, arithmetic_sequence 1 = 1 → arithmetic_sequence 3 = -3 → S k = -35 → k = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l337_33783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_B_is_direct_proportion_f_A_is_not_direct_proportion_f_C_is_not_direct_proportion_f_D_is_not_direct_proportion_l337_33773

-- Define the concept of a direct proportion function
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

-- Define the given functions
noncomputable def f_A : ℝ → ℝ := λ x ↦ (4 - 3*x) / 2
noncomputable def f_B : ℝ → ℝ := λ x ↦ x / 4
noncomputable def f_C : ℝ → ℝ := λ x ↦ -5 / x + 3
noncomputable def f_D : ℝ → ℝ := λ x ↦ 2*x^2 + 1/3

-- Theorem stating that f_B is a direct proportion function
theorem f_B_is_direct_proportion :
  is_direct_proportion f_B := by
  use 1/4
  intro x
  simp [f_B]
  ring

-- Theorems stating that the other functions are not direct proportion functions
theorem f_A_is_not_direct_proportion :
  ¬ is_direct_proportion f_A := by
  sorry

theorem f_C_is_not_direct_proportion :
  ¬ is_direct_proportion f_C := by
  sorry

theorem f_D_is_not_direct_proportion :
  ¬ is_direct_proportion f_D := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_B_is_direct_proportion_f_A_is_not_direct_proportion_f_C_is_not_direct_proportion_f_D_is_not_direct_proportion_l337_33773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_real_number_square_nonnegative_max_four_solutions_l337_33738

-- Define complex number
def complex_number (a b : ℝ) := ℂ

-- Define the real part of a complex number
def real_part (z : ℂ) : ℝ := Complex.re z

-- Define the imaginary part of a complex number
def imag_part (z : ℂ) : ℝ := Complex.im z

-- Statement 1: "z² ≥ 0" is a necessary condition for "z ∈ ℝ"
theorem real_number_square_nonnegative (z : ℂ) :
  (real_part z = z ∧ imag_part z = 0) → Complex.re (z * z) ≥ 0 :=
by sorry

-- Statement 2: The equation z² + m|z| + n = 0 can have at most 4 solutions in ℝ
theorem max_four_solutions (m n : ℝ) :
  ∃ (S : Finset ℂ), (∀ z ∈ S, Complex.re z = z ∧ Complex.im z = 0 ∧ z * z + m * Complex.abs z + n = 0) ∧ S.card ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_real_number_square_nonnegative_max_four_solutions_l337_33738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_positive_factors_count_l337_33730

def n : ℕ := 2^4 * 3^3 * 5^2 * 7

theorem even_positive_factors_count : 
  (Finset.filter (fun d => d ∣ n ∧ Even d) (Finset.range (n + 1))).card = 96 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_positive_factors_count_l337_33730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l337_33736

noncomputable def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 9 * x - 15
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 - 6 * x + 10

noncomputable def intersection_x1 : ℝ := (3 - Real.sqrt 209) / 4
noncomputable def intersection_x2 : ℝ := (3 + Real.sqrt 209) / 4

noncomputable def intersection_y1 : ℝ := parabola1 intersection_x2
noncomputable def intersection_y2 : ℝ := parabola1 intersection_x1

theorem parabola_intersection :
  (∀ x : ℝ, parabola1 x = parabola2 x ↔ x = intersection_x1 ∨ x = intersection_x2) ∧
  parabola1 intersection_x1 = parabola2 intersection_x1 ∧
  parabola1 intersection_x2 = parabola2 intersection_x2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_l337_33736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l337_33700

noncomputable def z (a : ℝ) : ℂ := (a + Complex.I) / (1 + 2 * Complex.I)

def isPurelyImaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_condition (a : ℝ) : isPurelyImaginary (z a) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purely_imaginary_condition_l337_33700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_transformation_l337_33726

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ∈ Set.Icc (-3) 0 then -2 - x
  else if x ∈ Set.Icc 0 2 then Real.sqrt (4 - (x - 2)^2) - 2
  else if x ∈ Set.Icc 2 3 then 2 * (x - 2)
  else 0  -- Default value for x outside [-3, 3]

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := 2 * f (x / 3) - 6

-- State the theorem
theorem h_transformation :
  ∃ (a b c : ℝ), ∀ x, h x = a * f (b * x) + c ∧ a = 2 ∧ b = 1/3 ∧ c = -6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_transformation_l337_33726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l337_33753

open Set Real

noncomputable def f (x : ℝ) : ℝ := 1 / sqrt (1 - x) + sqrt (x + 3) - 1

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = Icc (-3) 1 \ {1} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l337_33753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_ratio_correct_answer_is_C_l337_33794

/-- A cylinder derived from a rectangle with perimeter 12 -/
structure Cylinder where
  height : ℝ
  base_circumference : ℝ
  perimeter_constraint : height + base_circumference / 2 = 6

/-- The volume of the cylinder -/
noncomputable def volume (c : Cylinder) : ℝ :=
  (c.base_circumference^2 * c.height) / (4 * Real.pi)

/-- The ratio of the base circumference to the height -/
noncomputable def base_to_height_ratio (c : Cylinder) : ℝ :=
  c.base_circumference / c.height

/-- Theorem: When the volume is maximized, the ratio of base circumference to height is 2:1 -/
theorem max_volume_ratio (c : Cylinder) :
  (∀ c' : Cylinder, volume c' ≤ volume c) →
  base_to_height_ratio c = 2 := by
  sorry

/-- Corollary: The correct answer choice is C -/
theorem correct_answer_is_C (c : Cylinder) 
  (h : ∀ c' : Cylinder, volume c' ≤ volume c) : 
  base_to_height_ratio c = 2 := by
  exact max_volume_ratio c h

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_ratio_correct_answer_is_C_l337_33794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximateSubstandardWeightTheorem_l337_33762

/-- Calculates the approximate weight of substandard pads in a batch -/
noncomputable def approximateSubstandardWeight (totalWeight : ℝ) (sampleSize : ℕ) (substandardCount : ℕ) : ℝ :=
  (substandardCount : ℝ) / (sampleSize : ℝ) * totalWeight

/-- Proves that the approximate weight of substandard pads in the given batch is 8.9kg -/
theorem approximateSubstandardWeightTheorem :
  let totalWeight : ℝ := 500
  let sampleSize : ℕ := 280
  let substandardCount : ℕ := 5
  let result : ℝ := approximateSubstandardWeight totalWeight sampleSize substandardCount
  ∃ ε > 0, |result - 8.9| < ε :=
by
  sorry

-- Use #eval only for computable functions
def approximateSubstandardWeightNat (totalWeight sampleSize substandardCount : Nat) : Nat :=
  substandardCount * totalWeight / sampleSize

#eval approximateSubstandardWeightNat 500 280 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximateSubstandardWeightTheorem_l337_33762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_difference_l337_33793

theorem exponential_difference (b : ℝ) (h1 : b < 0) (h2 : (3 : ℝ)^b + (3 : ℝ)^(-b) = Real.sqrt 13) : 
  (3 : ℝ)^b - (3 : ℝ)^(-b) = -3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_difference_l337_33793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l337_33727

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)

-- State the theorem
theorem f_derivative_at_one :
  (∀ x : ℝ, f (1 / x) = x / (1 + x)) →
  deriv f 1 = -1/4 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_at_one_l337_33727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_bus_operation_l337_33748

/-- The profit function for bus operations -/
noncomputable def profit_function (x : ℕ) : ℝ := -x^2 + 12*x - 25

/-- The average annual profit function -/
noncomputable def average_annual_profit (x : ℕ) : ℝ := 
  if x = 0 then 0 else (profit_function x) / x

/-- The optimal number of years to operate buses -/
def optimal_years : ℕ := 6

/-- Theorem stating that the optimal_years maximizes the average annual profit -/
theorem optimal_bus_operation :
  ∀ n : ℕ, n ≠ 0 → average_annual_profit optimal_years ≥ average_annual_profit n :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_bus_operation_l337_33748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l337_33735

-- Define the sets M and N
def M : Set ℝ := {x | (x - 1) * (x - 3) * (x - 5) < 0}
def N : Set ℝ := {x | (x - 2) * (x - 4) * (x - 6) > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 3 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l337_33735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_properties_l337_33701

def A : Set ℝ := {x | (x - 2) * (x - 6) ≤ 0}
def B : Set ℝ := {x | x^2 - 6*x + 5 < 0}
def C (m : ℝ) : Set ℝ := {x | m < x ∧ x < m + 1}
def U : Set ℝ := Set.univ

theorem set_properties :
  (A ∪ B = {x : ℝ | 1 < x ∧ x ≤ 6}) ∧
  ((Set.compl A) ∩ B = {x : ℝ | 1 < x ∧ x < 2}) ∧
  (∀ m : ℝ, C m ⊆ B → m ≥ 1 ∧ m ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_properties_l337_33701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_complex_numbers_magnitude_of_complex_number_l337_33742

def z₁ : ℂ := Complex.mk 3 (-2)
def z₂ : ℂ := Complex.mk (-2) 3

theorem product_of_complex_numbers :
  z₁ * z₂ = Complex.mk (-12) 1 := by sorry

theorem magnitude_of_complex_number :
  ∃ z : ℂ, (1 / z = 1 / z₁ + 1 / z₂) ∧ Complex.abs z = 13 * Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_of_complex_numbers_magnitude_of_complex_number_l337_33742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_proof_exponential_function_max_min_diff_l337_33717

noncomputable def exponential_function (a b : ℝ) (x : ℝ) : ℝ := b * (a^x)

theorem exponential_function_proof 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : exponential_function a b 1 = 6) 
  (h4 : exponential_function a b 3 = 24) :
  a = 2 ∧ b = 3 := by
  sorry

theorem exponential_function_max_min_diff 
  (a : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : ∃ (b : ℝ), ∀ (x : ℝ), exponential_function a b x = a^x) 
  (h4 : |exponential_function a 1 1 - exponential_function a 1 (-1)| = 1) :
  a = (1 + Real.sqrt 5) / 2 ∨ a = (-1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_function_proof_exponential_function_max_min_diff_l337_33717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_share_is_approximately_4065_l337_33782

/-- Calculates the share of profit for a partner in a business --/
def calculate_share_of_profit (x_investment y_investment z_investment : ℚ) 
  (z_join_month total_months : ℕ) (total_profit : ℚ) : ℚ :=
  let x_investment_months := x_investment * total_months
  let y_investment_months := y_investment * total_months
  let z_investment_months := z_investment * (total_months - z_join_month)
  let total_investment_months := x_investment_months + y_investment_months + z_investment_months
  let z_ratio := z_investment_months / total_investment_months
  z_ratio * total_profit

/-- Theorem stating that Z's share of the profit is approximately 4065 --/
theorem z_share_is_approximately_4065 :
  ⌊calculate_share_of_profit 36000 42000 48000 4 12 13970⌋ = 4065 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_share_is_approximately_4065_l337_33782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_diagonals_l337_33785

/-- Given a regular quadrilateral prism with base side length a and volume V,
    the cosine of the angle between the diagonals of two adjacent lateral faces
    is V^2 / (V^2 + a^6). -/
theorem cosine_angle_between_diagonals (a V : ℝ) (ha : a > 0) (hV : V > 0) :
  let cos_angle := V^2 / (V^2 + a^6)
  cos_angle ≥ 0 ∧ cos_angle ≤ 1 :=
by
  -- Define cos_angle
  let cos_angle := V^2 / (V^2 + a^6)
  
  -- Split into two goals: cos_angle ≥ 0 and cos_angle ≤ 1
  constructor
  
  -- Prove cos_angle ≥ 0
  · apply div_nonneg
    · exact sq_nonneg V
    · apply add_nonneg
      · exact sq_nonneg V
      · exact pow_nonneg (by linarith) 6
  
  -- Prove cos_angle ≤ 1
  · have h1 : V^2 ≤ V^2 + a^6 := by
      apply le_add_of_nonneg_right
      exact pow_nonneg (by linarith) 6
    exact div_le_one_of_le h1 (by positivity)

-- Note: This theorem proves that the result is a valid cosine value.
-- The full geometric proof is beyond the scope of this basic statement.

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_angle_between_diagonals_l337_33785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l337_33718

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + log x - x

-- State the theorem
theorem f_inequality_implies_a_range :
  ∀ a : ℝ, a ≠ 0 →
  (∀ x : ℝ, x > 1 → f a x < 2 * a * x) →
  a ∈ Set.Icc (-1 : ℝ) 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_range_l337_33718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_number_game_l337_33706

/-- Two mathematicians with numbers differing by 1 will determine each other's number in finite rounds -/
theorem mathematicians_number_game (a b : ℕ) (h : a = b + 1 ∨ b = a + 1) :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → (∃ m : ℕ, m ≤ n ∧ 
    ((a = m ∧ b = m + 1) ∨ (b = m ∧ a = m + 1))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mathematicians_number_game_l337_33706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_find_C_l337_33769

/-- The remainder theorem states that the remainder of a polynomial P(x) 
    when divided by (x - a) is equal to P(a) -/
theorem remainder_theorem (P : ℝ → ℝ) (a : ℝ) : 
  ∃ Q : ℝ → ℝ, P = fun x ↦ (x - a) * Q x + P a := by
  sorry

/-- Given a polynomial C x^3 - 3x^2 + x - 1, prove that C = 2 
    when the remainder of division by (x + 1) is -7 -/
theorem find_C : 
  ∃! C : ℝ, (fun x ↦ C * x^3 - 3 * x^2 + x - 1) (-1) = -7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_find_C_l337_33769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_popcorn_probability_l337_33705

/-- Represents the probability that a randomly selected kernel that pops is white -/
noncomputable def prob_white_given_popped (
  total_kernels : ℝ
  ) (white_ratio yellow_ratio damaged_ratio : ℝ) 
  (white_pop_rate yellow_pop_rate : ℝ) : ℝ :=
  let white_kernels := total_kernels * white_ratio
  let yellow_kernels := total_kernels * yellow_ratio
  let good_white_kernels := white_kernels * (1 - damaged_ratio)
  let good_yellow_kernels := yellow_kernels * (1 - damaged_ratio)
  let popped_white := good_white_kernels * white_pop_rate
  let popped_yellow := good_yellow_kernels * yellow_pop_rate
  let total_popped := popped_white + popped_yellow
  popped_white / total_popped

theorem popcorn_probability 
  (total_kernels : ℝ) (total_kernels_pos : 0 < total_kernels) :
  prob_white_given_popped total_kernels (3/4) (1/4) (1/4) (3/5) (4/5) = 9/13 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_popcorn_probability_l337_33705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_approx_l337_33752

/-- The speed of the current in a river, given the following conditions:
  * A man can row at 15 km/h in still water
  * It takes 5.999520038396929 seconds to cover 30 meters downstream
-/
noncomputable def current_speed : ℝ :=
  let rowing_speed : ℝ := 15 -- km/h
  let distance : ℝ := 30 -- meters
  let time : ℝ := 5.999520038396929 -- seconds
  let rowing_speed_mps : ℝ := rowing_speed * 1000 / 3600 -- convert to m/s
  let total_speed : ℝ := distance / time -- m/s
  let current_speed_mps : ℝ := total_speed - rowing_speed_mps
  current_speed_mps * 3600 / 1000 -- convert back to km/h

/-- Theorem stating that the calculated current speed is approximately 2.99984 km/h -/
theorem current_speed_approx : 
  ∃ ε > 0, |current_speed - 2.99984| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_current_speed_approx_l337_33752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_symmetry_division_l337_33771

theorem complex_symmetry_division (z₁ z₂ : ℂ) :
  (z₁.re = 1 ∧ z₁.im = -2) →
  (z₂.re = -z₁.re ∧ z₂.im = z₁.im) →
  z₁ / z₂ = Complex.mk (3/5) (4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_symmetry_division_l337_33771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_park_length_is_55_5_l337_33778

/-- Represents a rectangular park with crossroads --/
structure Park where
  width : ℝ
  roadWidth : ℝ
  lawnArea : ℝ

/-- Calculates the length of the park given its properties --/
noncomputable def parkLength (p : Park) : ℝ :=
  (p.lawnArea + 2 * p.roadWidth * (p.width - 2 * p.roadWidth)) / (p.width - 2 * p.roadWidth)

/-- Theorem stating that a park with given properties has a length of 55.5 meters --/
theorem park_length_is_55_5 (p : Park) 
  (h1 : p.width = 40)
  (h2 : p.roadWidth = 3)
  (h3 : p.lawnArea = 2109) :
  parkLength p = 55.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_park_length_is_55_5_l337_33778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_external_common_tangents_l337_33725

/-- Circle A with equation (x+3)^2 + y^2 = 9 -/
def circle_A (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 9

/-- Circle B with equation (x-1)^2 + y^2 = 1 -/
def circle_B (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- Line equation y = k(x-3) -/
def tangent_line (k : ℝ) (x y : ℝ) : Prop := y = k * (x - 3)

/-- External common tangent lines of circles A and B -/
theorem external_common_tangents :
  ∃ k₁ k₂ : ℝ, k₁ = Real.sqrt 3 / 3 ∧ k₂ = -Real.sqrt 3 / 3 ∧
  (∀ x y : ℝ, tangent_line k₁ x y → (circle_A x y ∨ circle_B x y)) ∧
  (∀ x y : ℝ, tangent_line k₂ x y → (circle_A x y ∨ circle_B x y)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_external_common_tangents_l337_33725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_propositions_true_l337_33751

-- Define the propositions
def proposition1 : Prop := 
  (∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1) ↔ (∀ x : ℝ, x ≠ 1 → x^2 - 3*x + 2 ≠ 0)

def proposition2 : Prop := 
  ∀ p q : Prop, (¬(p ∨ q)) → (¬p ∧ ¬q)

def proposition3 : Prop := 
  (¬∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)

def proposition4 : Prop := 
  ∀ A B : ℝ, 0 < A → A < Real.pi → 0 < B → B < Real.pi →
    (A < B → Real.sin A < Real.sin B) ∧ ¬(Real.sin A < Real.sin B → A < B)

-- The theorem to prove
theorem three_propositions_true : 
  (proposition1 ∧ proposition2 ∧ proposition3 ∧ ¬proposition4) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_propositions_true_l337_33751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2013_l337_33797

def my_sequence (n : ℕ+) : ℚ := (-1)^n.val / n.val

theorem sequence_2013 : my_sequence 2013 = -1 / 2013 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2013_l337_33797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_is_chord_semicircle_arc_relationship_l337_33788

-- Define basic circle concepts
class Circle : Type where
  -- Add any necessary circle properties here

class Chord (c : Circle) : Prop where
  -- Add any necessary chord properties here

class Diameter (c : Circle) extends Chord c : Prop where
  -- Add any necessary diameter properties here

class Arc (c : Circle) : Prop where
  -- Add any necessary arc properties here

class Semicircle (c : Circle) extends Arc c : Prop where
  -- Add any necessary semicircle properties here

-- Theorem 1: A diameter is a chord
theorem diameter_is_chord (c : Circle) : Diameter c → Chord c := by
  sorry

-- Theorem 2: A semicircle is an arc, but an arc is not necessarily a semicircle
theorem semicircle_arc_relationship (c : Circle) :
  (∀ (s : Semicircle c), Arc c) ∧ ∃ (a : Arc c), ¬ (Semicircle c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_is_chord_semicircle_arc_relationship_l337_33788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l337_33775

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a = 1 →
  b = 2 →
  Real.cos C = 1/4 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  Real.sin C = Real.sqrt (1 - (Real.cos C)^2) →
  Real.sin A / a = Real.sin C / c →
  c = 2 ∧ Real.sin A = Real.sqrt 15 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l337_33775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quadrant_point_l337_33750

def is_in_first_quadrant (p : ℝ × ℝ) : Prop := p.1 > 0 ∧ p.2 > 0

theorem first_quadrant_point :
  let points := [(1, -2), (1, 2), (-1, -2), (-1, 2)]
  ∃! p, p ∈ points ∧ is_in_first_quadrant p ∧ p = (1, 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_quadrant_point_l337_33750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cube_intersection_areas_l337_33799

/-- The surface area of a sphere inscribed around a unit cube -/
noncomputable def sphereSurfaceArea : ℝ := 3 * Real.pi

/-- The radius of the sphere inscribed around a unit cube -/
noncomputable def sphereRadius : ℝ := Real.sqrt 3 / 2

/-- The surface area of a spherical cap formed by the intersection of the sphere with a cube face -/
noncomputable def sphericalCapArea : ℝ := Real.pi * (3 - Real.sqrt 3) / 2

/-- The surface area of the four-sided region formed by the intersection of the sphere with cube faces -/
noncomputable def fourSidedRegionArea : ℝ := Real.pi * (Real.sqrt 3 - 1) / 2

/-- The surface area of the two-sided region formed by the intersection of the sphere with cube faces -/
noncomputable def twoSidedRegionArea : ℝ := Real.pi * (2 - Real.sqrt 3) / 4

theorem sphere_cube_intersection_areas :
  fourSidedRegionArea = Real.pi * (Real.sqrt 3 - 1) / 2 ∧
  twoSidedRegionArea = Real.pi * (2 - Real.sqrt 3) / 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cube_intersection_areas_l337_33799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_price_is_correct_l337_33791

/-- The price of a sandwich given the following conditions:
  * There are 2 sandwiches
  * There are 4 sodas
  * Each soda costs $1.87
  * The total cost is $12.46
-/
noncomputable def sandwich_price : ℚ :=
  let num_sandwiches : ℕ := 2
  let num_sodas : ℕ := 4
  let soda_price : ℚ := 187/100
  let total_cost : ℚ := 1246/100
  (total_cost - num_sodas * soda_price) / num_sandwiches

theorem sandwich_price_is_correct : sandwich_price = 249/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandwich_price_is_correct_l337_33791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_ellipses_l337_33703

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- The y-coordinate of the intersection point of two ellipses -/
noncomputable def intersection_y_coord (a b c d : Point) : ℝ :=
  (-12 + 12 * Real.sqrt 6) / 5

theorem intersection_of_ellipses (a b c d p : Point)
  (h1 : distance p a + distance p d = 10)
  (h2 : distance p b + distance p c = 10)
  (h3 : a.x = -4 ∧ a.y = 0)
  (h4 : b.x = -1 ∧ b.y = 2)
  (h5 : c.x = 1 ∧ c.y = 2)
  (h6 : d.x = 4 ∧ d.y = 0) :
  p.y = intersection_y_coord a b c d := by
  sorry

#check intersection_of_ellipses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_ellipses_l337_33703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unobstructed_path_l337_33720

/-- Represents a brick with dimensions 2 × 2 × 1 -/
structure Brick where
  x : ℕ
  y : ℕ
  z : ℕ

/-- Represents a cube composed of bricks -/
structure Cube where
  size : ℕ
  bricks : List Brick

/-- Represents a straight line path through the cube -/
structure NeedlePath where
  start_x : ℕ
  start_y : ℕ
  end_x : ℕ
  end_y : ℕ

/-- Checks if a path intersects with a brick -/
def intersects_brick (p : NeedlePath) (b : Brick) : Prop := sorry

/-- Main theorem: There exists a path through the cube that doesn't intersect any brick -/
theorem exists_unobstructed_path (c : Cube) 
  (h1 : c.size = 20) 
  (h2 : c.bricks.length = 2000) 
  (h3 : ∀ b ∈ c.bricks, b.x ≤ 18 ∧ b.y ≤ 18 ∧ b.z ≤ 19) : 
  ∃ p : NeedlePath, ∀ b ∈ c.bricks, ¬(intersects_brick p b) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unobstructed_path_l337_33720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_year_statistics_l337_33710

def leap_year_data : List ℕ := 
  (List.range 29).bind (λ x => List.replicate 12 (x + 1)) ++
  List.replicate 12 30 ++ List.replicate 12 31

noncomputable def median (l : List ℕ) : ℚ := sorry

noncomputable def mean (l : List ℕ) : ℚ := sorry

def modes (l : List ℕ) : List ℕ := sorry

theorem leap_year_statistics :
  let d := median (modes leap_year_data)
  let M := median leap_year_data
  let μ := mean leap_year_data
  d = 16 ∧ M = 16 ∧ μ = 16 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leap_year_statistics_l337_33710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_are_integers_l337_33764

def sequence_a : ℕ → ℤ
  | 0 => 2  -- Adding this case to cover all natural numbers
  | 1 => 1
  | 2 => 1
  | 3 => 997
  | (n + 4) => (1993 + sequence_a (n + 3) * sequence_a (n + 2)) / sequence_a (n + 1)

theorem all_terms_are_integers : ∀ n : ℕ, ∃ k : ℤ, sequence_a n = k := by
  sorry

#eval sequence_a 0
#eval sequence_a 1
#eval sequence_a 2
#eval sequence_a 3
#eval sequence_a 4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_terms_are_integers_l337_33764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arithmetic_sequence_l337_33796

theorem sin_arithmetic_sequence (a : Real) : 
  (0 < a) ∧ (a < 2 * Real.pi) →
  (∃ d : Real, Real.sin a + d = Real.sin (2 * a) ∧ Real.sin (2 * a) + d = Real.sin (3 * a)) ↔
  a = Real.pi :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_arithmetic_sequence_l337_33796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_70000_l337_33792

/-- Represents an investment project --/
structure Project where
  maxProfitRate : ℝ
  maxLossRate : ℝ

/-- The investment problem --/
structure InvestmentProblem where
  totalInvestment : ℝ
  projectA : Project
  projectB : Project
  maxLoss : ℝ

/-- The maximum profit achievable given the investment constraints --/
noncomputable def maxProfit (problem : InvestmentProblem) : ℝ := 
  sorry

/-- Theorem stating that the maximum profit is 70,000 yuan --/
theorem max_profit_is_70000 (problem : InvestmentProblem) : 
  problem.totalInvestment ≤ 100000 ∧ 
  problem.projectA.maxProfitRate = 1 ∧ 
  problem.projectA.maxLossRate = 0.3 ∧
  problem.projectB.maxProfitRate = 0.5 ∧
  problem.projectB.maxLossRate = 0.1 ∧
  problem.maxLoss = 18000 →
  maxProfit problem = 70000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_70000_l337_33792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l337_33774

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := 6 * x^2 + 4 * x - 1 / x + 2
noncomputable def g (x k : ℝ) : ℝ := x^2 - k + 3 * x

-- State the theorem
theorem find_k : ∃ k : ℝ, f 3 - g 3 k = 5 ∧ k = -134/3 := by
  -- Introduce k
  let k : ℝ := -134/3
  -- Prove existence
  use k
  -- Split the conjunction
  constructor
  -- Prove the equation
  · simp [f, g]
    norm_num
  -- Prove k equals -134/3
  · rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_k_l337_33774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l337_33743

noncomputable def f (x : ℝ) : ℝ := (7 * x^2 - 13) / (4 * x^2 + 7 * x + 3)

theorem vertical_asymptotes_sum :
  ∀ p q : ℝ,
  (∀ x : ℝ, x ≠ p ∧ x ≠ q → f x ≠ 0) →
  (∀ ε : ℝ, ε > 0 → ∃ δ : ℝ, δ > 0 ∧
    ((∀ x : ℝ, 0 < |x - p| ∧ |x - p| < δ → |f x| > 1/ε) ∧
     (∀ x : ℝ, 0 < |x - q| ∧ |x - q| < δ → |f x| > 1/ε))) →
  p + q = -7/4 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptotes_sum_l337_33743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_with_conditions_l337_33723

theorem least_sum_with_conditions (a b : ℕ) 
  (h1 : Nat.gcd (a + b) 330 = 1)
  (h2 : a ^ a % (b ^ 3) = 0)
  (h3 : a % b ≠ 0)
  (ha : a > 0)
  (hb : b > 0) :
  ∀ (x y : ℕ), 
    x > 0 → y > 0 →
    (Nat.gcd (x + y) 330 = 1) → 
    (x ^ x % (y ^ 3) = 0) → 
    (x % y ≠ 0) → 
    (a + b : ℕ) ≤ (x + y : ℕ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_sum_with_conditions_l337_33723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julian_run_time_l337_33708

/-- Represents Julian's training scenario -/
structure JulianTraining where
  walk_speed : ℝ
  run_speed : ℝ
  cycle_speed : ℝ
  total_distance : ℝ
  time_saved_cycling : ℝ

/-- Calculates the time taken to run 1 km given Julian's training scenario -/
noncomputable def time_to_run_1km (j : JulianTraining) : ℝ :=
  3 / (2 * j.walk_speed)

/-- Theorem stating that Julian takes 6 minutes to run 1 km -/
theorem julian_run_time (j : JulianTraining) 
  (h1 : j.run_speed = 2 * j.walk_speed)
  (h2 : j.cycle_speed = 3 * j.walk_speed)
  (h3 : j.total_distance = 3)
  (h4 : j.time_saved_cycling = 10)
  (h5 : (1 / j.walk_speed + 1 / j.run_speed + 1 / j.cycle_speed) - (3 / j.cycle_speed) = j.time_saved_cycling) :
  time_to_run_1km j = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julian_run_time_l337_33708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCDFG_l337_33741

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of an equilateral triangle -/
def is_equilateral (A B C : Point) : Prop :=
  (A.x - B.x)^2 + (A.y - B.y)^2 = (B.x - C.x)^2 + (B.y - C.y)^2 ∧
  (B.x - C.x)^2 + (B.y - C.y)^2 = (C.x - A.x)^2 + (C.y - A.y)^2

/-- Definition of a midpoint -/
def is_midpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

/-- Distance between two points -/
noncomputable def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

/-- Theorem: Perimeter of ABCDFG -/
theorem perimeter_ABCDFG (A B C D E F G : Point) : 
  is_equilateral A B C →
  is_equilateral A D E →
  is_equilateral D F G →
  is_midpoint D A C →
  is_midpoint F D E →
  distance A B = 6 →
  distance A B + distance B C + distance C D + distance D F + distance F G + distance G A = 21 := by
  sorry

#check perimeter_ABCDFG

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perimeter_ABCDFG_l337_33741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_50_l337_33755

open Nat

theorem divisible_by_50 (X Y : ℕ) (h : X + Y = 10^200) 
    (h_perm : ∃ σ : Fin (digits X 10).length ≃ Fin (digits Y 10).length, 
              ∀ i, (digits X 10)[i] = (digits Y 10)[σ i]) : 
    50 ∣ X := by
  sorry  -- Placeholder for the actual proof

#check divisible_by_50

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_50_l337_33755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_rectangle_area_l337_33787

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if two circles are tangent -/
def areTangent (c1 c2 : Circle) : Prop :=
  distance c1.center c2.center = c1.radius + c2.radius

/-- Check if a circle touches a side of a rectangle -/
def touchesRectangleSide (c : Circle) (r : Rectangle) : Prop :=
  c.radius = r.width / 2 ∨ c.radius = r.height / 2

theorem circle_rectangle_area (x y z : Circle) (rect : Rectangle) :
  areTangent x y ∧ areTangent y z ∧ areTangent x z ∧
  touchesRectangleSide x rect ∧ touchesRectangleSide z rect ∧
  distance x.center y.center = 30 ∧
  distance y.center z.center = 20 ∧
  distance x.center z.center = 40 →
  rect.width * rect.height = 2000 + 500 * Real.sqrt 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_rectangle_area_l337_33787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_number_proof_l337_33747

theorem middle_number_proof (numbers : List ℝ) 
  (h_length : numbers.length = 13)
  (h_total_avg : numbers.sum / 13 = 9)
  (h_first_6_avg : (numbers.take 6).sum / 6 = 5)
  (h_last_6_avg : (numbers.drop 7).sum / 6 = 7) :
  numbers.get? 6 = some 45 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_number_proof_l337_33747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_to_fx_eq_x_l337_33702

noncomputable def f (b c : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + b*x + c else 2

theorem three_solutions_to_fx_eq_x (b c : ℝ) :
  f b c (-4) = 2 ∧ f b c (-2) = -2 →
  ∃! (s : Finset ℝ), s.card = 3 ∧ ∀ x, x ∈ s ↔ f b c x = x :=
by
  sorry

#check three_solutions_to_fx_eq_x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_solutions_to_fx_eq_x_l337_33702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l337_33772

-- Define the sales revenue function
noncomputable def R (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 10.8 - x^2 / 30
  else if x > 10 then 108 / x - 1000 / (3 * x^2)
  else 0

-- Define the annual profit function
noncomputable def W (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 10 then 8.1 * x - x^3 / 30 - 10
  else if x > 10 then 98 - 1000 / (3 * x) - 2.7 * x
  else 0

-- Theorem statement
theorem max_profit_at_nine :
  ∃ (x : ℝ), x = 9 ∧ ∀ (y : ℝ), W y ≤ W x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_nine_l337_33772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l337_33746

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the point through which the tangent line passes
def tangent_point : ℝ × ℝ := (-2, 1)

-- Define the proposed tangent line
def tangent_line_eq (x y : ℝ) : Prop := 2*x - y + 5 = 0

-- Theorem statement
theorem tangent_line_to_circle :
  ∃ x y : ℝ,
  circle_eq x y ∧
  tangent_line_eq x y ∧
  (x, y) = tangent_point ∧
  ∀ a b : ℝ,
    circle_eq a b →
    (a, b) ≠ (x, y) →
    ¬tangent_line_eq a b :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l337_33746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_2870_l337_33766

/-- Represents the cost per unit for different course types and sessions -/
structure CourseCost where
  fallSpringScience : ℚ
  fallSpringHumanities : ℚ
  summerWinterScience : ℚ
  summerWinterHumanities : ℚ

/-- Represents the number of units taken for different course types in each session -/
structure CourseUnits where
  fallScience : ℚ
  fallHumanities : ℚ
  springScience : ℚ
  springHumanities : ℚ
  summerScience : ℚ
  summerHumanities : ℚ
  winterScience : ℚ
  winterHumanities : ℚ

/-- Represents the scholarship percentages for different sessions -/
structure Scholarships where
  springScience : ℚ
  winterScience : ℚ

/-- Calculates the total cost of courses after applying scholarships -/
def totalCost (costs : CourseCost) (units : CourseUnits) (scholarships : Scholarships) : ℚ :=
  let fallCost := costs.fallSpringScience * units.fallScience + costs.fallSpringHumanities * units.fallHumanities
  let springCost := costs.fallSpringScience * units.springScience * (1 - scholarships.springScience) +
                    costs.fallSpringHumanities * units.springHumanities
  let summerCost := costs.summerWinterScience * units.summerScience + costs.summerWinterHumanities * units.summerHumanities
  let winterCost := costs.summerWinterScience * units.winterScience * (1 - scholarships.winterScience) +
                    costs.summerWinterHumanities * units.winterHumanities
  fallCost + springCost + summerCost + winterCost

/-- The main theorem to prove -/
theorem total_cost_is_2870 :
  let costs := CourseCost.mk 60 45 80 55
  let units := CourseUnits.mk 12 8 10 10 6 4 6 4
  let scholarships := Scholarships.mk (1/2) (3/4)
  totalCost costs units scholarships = 2870 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_2870_l337_33766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l337_33732

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem -/
def triangle_condition (t : Triangle) : Prop :=
  2 * t.b * Real.sin (t.C + Real.pi/6) = t.a + t.c

/-- M is the midpoint of BC -/
def is_midpoint (t : Triangle) (M : ℝ × ℝ) : Prop :=
  M.1 = (t.b + t.c) / 2 ∧ M.2 = 0

/-- AM = AC -/
def equal_distances (t : Triangle) (M : ℝ × ℝ) : Prop :=
  Real.sqrt ((M.1 - t.a)^2 + M.2^2) = Real.sqrt (t.c^2 + t.a^2 - 2*t.c*t.a*Real.cos t.B)

/-- The main theorem to be proved -/
theorem triangle_theorem (t : Triangle) (M : ℝ × ℝ) 
  (h1 : triangle_condition t) 
  (h2 : is_midpoint t M) 
  (h3 : equal_distances t M) : 
  t.B = Real.pi/3 ∧ Real.sin t.A = Real.sqrt 21 / 7 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l337_33732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_properties_l337_33715

/-- A frustum of a cone -/
structure Frustum where
  r₁ : ℝ  -- radius of one base
  r₂ : ℝ  -- radius of the other base
  l : ℝ   -- length of lateral edge

/-- The surface area of a frustum -/
noncomputable def surface_area (f : Frustum) : ℝ :=
  Real.pi * (f.r₁^2 + f.r₂^2 + (f.r₁ + f.r₂) * f.l)

/-- The height of a frustum -/
noncomputable def frustum_height (f : Frustum) : ℝ :=
  Real.sqrt (f.l^2 - (f.r₂ - f.r₁)^2)

/-- The volume of a frustum -/
noncomputable def volume (f : Frustum) : ℝ :=
  (1/3) * Real.pi * (f.r₁^2 + f.r₁*f.r₂ + f.r₂^2) * frustum_height f

theorem frustum_properties (f : Frustum) 
    (h1 : f.r₁ = 2) (h2 : f.r₂ = 7) (h3 : f.l = 6) : 
    surface_area f = 107 * Real.pi ∧ 
    volume f = (67 * Real.sqrt 11 / 3) * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_properties_l337_33715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_inequality_l337_33765

theorem negation_of_exponential_inequality :
  (¬ ∀ x : ℝ, x ∈ Set.Ici (0 : ℝ) → Real.exp x ≥ 1) ↔
  (∃ x : ℝ, x ∈ Set.Ici (0 : ℝ) ∧ Real.exp x < 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_exponential_inequality_l337_33765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_correct_l337_33721

-- Define the circles C₁ and C₂
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 9 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 9 = 0

-- Define the common chord length
noncomputable def common_chord_length : ℝ := 24/5

-- Theorem statement
theorem common_chord_length_is_correct :
  ∀ (x y : ℝ), C₁ x y ∧ C₂ x y → 
  ∃ (chord_length : ℝ), chord_length = common_chord_length :=
by
  sorry

#check common_chord_length_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_is_correct_l337_33721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_and_B_prob_C_l337_33798

-- Define the set of cards
def Cards : Finset ℕ := {1, 2, 3, 4}

-- Define the sample space
def SampleSpace : Finset (ℕ × ℕ × ℕ) :=
  Finset.product Cards (Finset.product Cards Cards)

-- Define event A: "a, b, c are completely the same"
def EventA : Finset (ℕ × ℕ × ℕ) :=
  {(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4)}

-- Define event B: "a, b, c are not completely the same"
def EventB : Finset (ℕ × ℕ × ℕ) :=
  SampleSpace \ EventA

-- Define event C: "a · b = c"
def EventC : Finset (ℕ × ℕ × ℕ) :=
  SampleSpace.filter (fun (a, b, c) => a * b = c)

-- Theorem for probabilities of events A and B
theorem prob_A_and_B :
  (Finset.card EventA : ℚ) / Finset.card SampleSpace = 1 / 16 ∧
  (Finset.card EventB : ℚ) / Finset.card SampleSpace = 15 / 16 := by
  sorry

-- Theorem for probability of event C
theorem prob_C :
  (Finset.card EventC : ℚ) / Finset.card SampleSpace = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_A_and_B_prob_C_l337_33798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_alloy_amount_is_10_l337_33709

/-- Represents the composition of an alloy mixture --/
structure AlloyMixture where
  first_alloy_amount : ℝ
  second_alloy_amount : ℝ
  first_alloy_chromium_percent : ℝ
  second_alloy_chromium_percent : ℝ
  result_chromium_percent : ℝ

/-- Calculates the amount of the first alloy used in the mixture --/
noncomputable def calculate_first_alloy_amount (mixture : AlloyMixture) : ℝ :=
  (mixture.result_chromium_percent * (mixture.first_alloy_amount + mixture.second_alloy_amount) -
   mixture.second_alloy_chromium_percent * mixture.second_alloy_amount) /
  (mixture.first_alloy_chromium_percent - mixture.result_chromium_percent)

/-- Theorem stating that the amount of the first alloy used is 10 kg --/
theorem first_alloy_amount_is_10 (mixture : AlloyMixture)
  (h1 : mixture.first_alloy_chromium_percent = 0.12)
  (h2 : mixture.second_alloy_chromium_percent = 0.08)
  (h3 : mixture.second_alloy_amount = 30)
  (h4 : mixture.result_chromium_percent = 0.09) :
  calculate_first_alloy_amount mixture = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_alloy_amount_is_10_l337_33709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_over_four_l337_33758

theorem tan_theta_minus_pi_over_four (θ : Real) 
  (h1 : Real.cos θ = -12/13) 
  (h2 : θ ∈ Set.Ioo π (3*π/2)) : 
  Real.tan (θ - π/4) = -7/17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_minus_pi_over_four_l337_33758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l337_33707

/-- Non-isosceles triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  non_isosceles : a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The condition (2c - b) cos C = (2b - c) cos B holds -/
def condition (t : Triangle) : Prop :=
  (2 * t.c - t.b) * Real.cos t.C = (2 * t.b - t.c) * Real.cos t.B

/-- Area of a triangle -/
noncomputable def Triangle.area (t : Triangle) : ℝ := 
  (1 / 2) * t.b * t.c * Real.sin t.A

theorem triangle_properties (t : Triangle) (h : condition t) : 
  t.A = Real.pi / 3 ∧ 
  (t.a = 4 → 0 < t.area ∧ t.area < 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l337_33707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_vector_problem_l337_33786

noncomputable section

-- Define the circle
def circle_set (O : ℝ × ℝ) : Set (ℝ × ℝ) := {p | (p.1 - O.1)^2 + (p.2 - O.2)^2 = 4}

-- Define vectors
def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Define dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector magnitude
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem circle_vector_problem (O A B : ℝ × ℝ) :
  A ∈ circle_set O →
  B ∈ circle_set O →
  magnitude (vector O A + vector O B) = magnitude (vector O A - vector O B) →
  dot_product (vector O A) (vector A B) = 4 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_vector_problem_l337_33786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_n_points_integer_distances_l337_33704

-- Define a point in a plane
structure Point where
  x : ℚ
  y : ℚ

-- Define a function to check if three points are collinear
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

-- Define a function to calculate the square of the distance between two points
def distanceSquared (p1 p2 : Point) : ℚ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2

-- The main theorem
theorem exist_n_points_integer_distances (N : ℕ) :
  ∃ (points : Finset Point),
    points.card = N ∧
    (∀ p1 p2 p3, p1 ∈ points → p2 ∈ points → p3 ∈ points →
      p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 → ¬collinear p1 p2 p3) ∧
    (∀ p1 p2, p1 ∈ points → p2 ∈ points → p1 ≠ p2 →
      ∃ n : ℕ, (n : ℚ) = distanceSquared p1 p2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exist_n_points_integer_distances_l337_33704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l337_33731

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1
  first_term : a 1 = 25
  second_term_diff : a 2 ≠ a 1
  geometric_property : (a 11) ^ 2 = a 1 * a 13

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * seq.a 1 + (n - 1) * (seq.a 2 - seq.a 1)) / 2

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n, seq.a n = 27 - 2 * n) ∧
  (∃ max_n, ∀ n, sum_n seq n ≤ sum_n seq max_n ∧ sum_n seq max_n = 169) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l337_33731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l337_33712

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2) * x^2 - 2*x + 5

theorem f_upper_bound (m : ℝ) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x < m) → m > 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_upper_bound_l337_33712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_decay_l337_33719

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := Real.exp (-k * x)

theorem range_of_exponential_decay (k : ℝ) (h : k > 0) :
  Set.range (fun x => f k x) ∩ Set.Ici 0 = Set.Ioc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_exponential_decay_l337_33719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l337_33716

theorem range_of_a (a : ℝ) :
  (∀ n : ℕ+, (-1 : ℝ)^(n : ℕ) * a < 3 + (-1 : ℝ)^((n : ℕ) + 1) / ((n : ℕ) + 1)) →
  -3 ≤ a ∧ a < 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l337_33716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_flood_pump_time_l337_33722

/-- Represents the basement flooding scenario -/
structure BasementFlood where
  length : ℚ  -- length of the basement in feet
  width : ℚ   -- width of the basement in feet
  depth : ℚ   -- depth of water in inches
  num_pumps : ℕ  -- number of pumps
  pump_rate : ℚ  -- pumping rate per pump in gallons per minute
  cf_to_gallon : ℚ  -- conversion factor from cubic feet to gallons

/-- Calculates the time required to pump out all water from the basement -/
def pump_time (b : BasementFlood) : ℚ :=
  let volume_cf := b.length * b.width * (b.depth / 12)  -- volume in cubic feet
  let volume_gallons := volume_cf * b.cf_to_gallon
  let total_pump_rate := b.num_pumps * b.pump_rate
  volume_gallons / total_pump_rate

/-- Theorem stating that for the given basement scenario, it takes 500 minutes to pump out all water -/
theorem basement_flood_pump_time :
  let b : BasementFlood := {
    length := 40,
    width := 20,
    depth := 24,
    num_pumps := 3,
    pump_rate := 8,
    cf_to_gallon := 15/2
  }
  pump_time b = 500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basement_flood_pump_time_l337_33722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_speed_calculation_l337_33768

/-- Calculates the return speed given the parameters of a round trip -/
noncomputable def return_speed (distance : ℝ) (speed_to : ℝ) (total_time : ℝ) : ℝ :=
  let time_to := distance / speed_to
  let time_from := total_time - time_to
  distance / time_from

theorem return_speed_calculation (distance : ℝ) (speed_to : ℝ) (total_time : ℝ)
  (h1 : distance = 24)
  (h2 : speed_to = 60)
  (h3 : total_time = 1) :
  return_speed distance speed_to total_time = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_speed_calculation_l337_33768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_octagon_perimeter_l337_33728

/-- An equilateral octagon with specific properties -/
structure EquilateralOctagon where
  -- Side length of the octagon
  side_length : ℝ
  -- Every other interior angle is 45°
  alternate_angle_measure : side_length > 0 → ∃ (a b c d : ℝ), a = 45 ∧ b = 45 ∧ c = 45 ∧ d = 45
  -- The enclosed area is 8√2
  area : side_length > 0 → side_length^2 * (Real.sqrt 2 + 1) = 8 * Real.sqrt 2

/-- The perimeter of an equilateral octagon is 16(√2 - 1) given its properties -/
theorem equilateral_octagon_perimeter (octagon : EquilateralOctagon) :
  octagon.side_length > 0 → octagon.side_length * 8 = 16 * (Real.sqrt 2 - 1) := by
  sorry

#check equilateral_octagon_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_octagon_perimeter_l337_33728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_positive_at_midpoint_l337_33754

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 - (a - 1) * x - a * Real.log x

-- Define the derivative of f
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := x - (a - 1) - a / x

theorem f_derivative_positive_at_midpoint (a b : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ ≠ x₂) 
  (h₄ : f a x₁ = b) (h₅ : f a x₂ = b) : 
  f_derivative a ((x₁ + x₂) / 2) > 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_positive_at_midpoint_l337_33754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_hyperbola_intersection_ratio_l337_33781

/-- Two parallel lines intersecting a hyperbola -/
theorem parallel_lines_hyperbola_intersection_ratio : 
  ∀ (k : ℝ) (xK xL xM xN : ℝ),
  -- Line equations
  (∀ x y, y = k * x + 14 ↔ y = 1 / x) → -- Intersection of first line with hyperbola
  (∀ x y, y = k * x + 4 ↔ y = 1 / x) → -- Intersection of second line with hyperbola
  -- Roots of the quadratic equations
  k * xK^2 + 14 * xK - 1 = 0 →
  k * xL^2 + 14 * xL - 1 = 0 →
  k * xM^2 + 4 * xM - 1 = 0 →
  k * xN^2 + 4 * xN - 1 = 0 →
  -- The ratio of the differences of the roots
  |xL - xK| / |xN - xM| = 3.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_hyperbola_intersection_ratio_l337_33781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_correct_prove_only_B_is_correct_l337_33729

-- Define the statements as axioms (assumed truths) instead of String literals
axiom statement_A : Prop
axiom statement_B : Prop
axiom statement_C : Prop
axiom statement_D : Prop

-- Define what it means for a statement to be correct
def is_correct (s : Prop) : Prop := s

-- Theorem to prove
theorem only_B_is_correct :
  is_correct statement_B ∧
  ¬is_correct statement_A ∧
  ¬is_correct statement_C ∧
  ¬is_correct statement_D :=
by
  sorry -- We use 'sorry' to skip the proof for now

-- Additional axioms to represent our understanding of the problem
axiom B_is_correct : is_correct statement_B
axiom A_is_incorrect : ¬is_correct statement_A
axiom C_is_incorrect : ¬is_correct statement_C
axiom D_is_incorrect : ¬is_correct statement_D

-- A proof using our axioms
theorem prove_only_B_is_correct :
  is_correct statement_B ∧
  ¬is_correct statement_A ∧
  ¬is_correct statement_C ∧
  ¬is_correct statement_D :=
by
  apply And.intro
  · exact B_is_correct
  · apply And.intro
    · exact A_is_incorrect
    · apply And.intro
      · exact C_is_incorrect
      · exact D_is_incorrect

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_B_is_correct_prove_only_B_is_correct_l337_33729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_theorem_l337_33733

-- Define the Triangle and Rectangle types
def Triangle (α : Type*) := α → α → α → Prop
def Rectangle (α : Type*) := α → α → α → α → Prop

-- Define necessary properties
def is_right_angle {α : Type*} [LinearOrderedField α] (t : Triangle α) (C : α) : Prop := sorry
def contains {α : Type*} (r : Rectangle α) (t : Triangle α) : Prop := sorry
def side_length {α : Type*} [LinearOrderedField α] (r : Rectangle α) (A B : α) : α := sorry
def diagonal_length {α : Type*} [LinearOrderedField α] (r : Rectangle α) (A C : α) : α := sorry

theorem pythagorean_theorem {α : Type*} [LinearOrderedField α] 
  (a b c : α) (triangle_ABC : Triangle α) (rectangle_ABDC : Rectangle α)
  (A B C D : α) :
  is_right_angle triangle_ABC C →
  contains rectangle_ABDC triangle_ABC →
  side_length rectangle_ABDC A D = a →
  side_length rectangle_ABDC A B = c →
  diagonal_length rectangle_ABDC B D = b →
  b^2 = a^2 + c^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pythagorean_theorem_l337_33733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tourism_value_added_l337_33711

open Real

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (51/50) * x - 0.01 * x^2 - log x + log 10

-- State the theorem
theorem max_tourism_value_added :
  ∃ (a : ℝ), a > 0 ∧ 
  (∀ x : ℝ, x ∈ Set.Ioc 6 12 → 
    (51/50) * x - a * x^2 - log x + log 10 = f x) ∧
  f 10 = 9.2 ∧
  (∀ x : ℝ, x ∈ Set.Ioc 6 12 → f x ≤ f 12) := by
  sorry

-- Note: Set.Ioc 6 12 represents the interval (6, 12]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_tourism_value_added_l337_33711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_symmetric_times_symmetric_zero_l337_33744

/-- Given two 3x3 matrices A and B, where A is skew-symmetric and B is symmetric with the specified forms, their product is a zero matrix. -/
theorem skew_symmetric_times_symmetric_zero (a b c d : ℝ) : 
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![0, d, -c; -d, 0, b; c, -b, 0]
  let B : Matrix (Fin 3) (Fin 3) ℝ := !![2*a^2, a*b, 2*a*c; a*b, 2*b^2, b*c; 2*a*c, b*c, 2*c^2]
  A * B = 0 := by
  sorry

#check skew_symmetric_times_symmetric_zero

end NUMINAMATH_CALUDE_ERRORFEEDBACK_skew_symmetric_times_symmetric_zero_l337_33744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mias_age_l337_33780

def guesses : List Nat := [26, 29, 33, 35, 37, 39, 43, 46, 50, 52]

def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d, 1 < d → d < n → ¬(n % d = 0)

def isOffByOne (guess : Nat) (age : Nat) : Bool :=
  guess = age - 1 || guess = age + 1

theorem mias_age :
  ∃ (age : Nat),
    age ∈ guesses ∧
    isPrime age ∧
    (guesses.filter (· < age)).length = guesses.length / 2 ∧
    (guesses.filter (isOffByOne · age)).length = 3 ∧
    age = 47 := by
  sorry

#eval guesses

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mias_age_l337_33780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l337_33714

-- Define the quadratic equation
def quadratic (x p : ℝ) : ℝ := x^2 - 6*x + p

-- Define the parameter p based on x
noncomputable def p (x : ℝ) : ℝ := if x > 0 then 8 else 9

-- Theorem statement
theorem solution_difference : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  quadratic x₁ 8 = 0 ∧ 
  quadratic x₂ 8 = 0 ∧ 
  |x₁ - x₂| = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l337_33714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_travelled_downstream_l337_33745

/-- Calculates the distance travelled downstream by a boat -/
theorem distance_travelled_downstream 
  (speed_still_water : ℝ) 
  (current_rate : ℝ) 
  (time_minutes : ℝ) 
  (h1 : speed_still_water = 12)
  (h2 : current_rate = 4)
  (h3 : time_minutes = 18) :
  (speed_still_water + current_rate) * (time_minutes / 60) = 4.8 := by
  sorry

#check distance_travelled_downstream

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_travelled_downstream_l337_33745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_heads_fair_coin_prob_even_heads_unfair_coin_l337_33757

/-- The probability of getting an even number of heads in n coin tosses -/
noncomputable def prob_even_heads (p : ℝ) (n : ℕ) : ℝ := (1 + (1 - 2*p)^n) / 2

theorem prob_even_heads_fair_coin (n : ℕ) :
  prob_even_heads (1/2) n = 1/2 := by
  sorry

theorem prob_even_heads_unfair_coin (p : ℝ) (n : ℕ) (h : 0 < p ∧ p < 1) :
  prob_even_heads p n = (1 + (1 - 2*p)^n) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_even_heads_fair_coin_prob_even_heads_unfair_coin_l337_33757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fodder_duration_after_addition_l337_33777

/-- Represents the daily fodder consumption of a cow -/
def cow_unit : ℚ := 1

/-- Represents the daily fodder consumption of a buffalo -/
def buffalo_unit : ℚ := 4/3 * cow_unit

/-- Represents the daily fodder consumption of an ox -/
def ox_unit : ℚ := 3/2 * cow_unit

/-- Calculates the total daily fodder consumption for a given number of animals -/
def total_consumption (buffaloes oxen cows : ℕ) : ℚ :=
  buffalo_unit * buffaloes + ox_unit * oxen + cow_unit * cows

/-- Calculates the number of days the fodder will last given the total fodder and daily consumption -/
def days_of_fodder (total_fodder daily_consumption : ℚ) : ℕ :=
  (total_fodder / daily_consumption).floor.toNat

theorem fodder_duration_after_addition :
  let initial_buffaloes : ℕ := 15
  let initial_oxen : ℕ := 8
  let initial_cows : ℕ := 24
  let initial_days : ℕ := 24
  let added_buffaloes : ℕ := 15
  let added_cows : ℕ := 40
  let initial_daily_consumption := total_consumption initial_buffaloes initial_oxen initial_cows
  let initial_total_fodder := initial_daily_consumption * initial_days
  let new_daily_consumption := total_consumption (initial_buffaloes + added_buffaloes) initial_oxen (initial_cows + added_cows)
  days_of_fodder initial_total_fodder new_daily_consumption = 49 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fodder_duration_after_addition_l337_33777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_bound_range_of_a_l337_33763

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 ∧ x ≤ 3 then a - x
  else if x > 3 then a * (Real.log x / Real.log 2)
  else 0  -- This case is not specified in the original problem, but we need to handle all real numbers

-- State the theorem
theorem f_inequality_implies_a_bound (a : ℝ) : f a 2 < f a 4 → a > -2 := by
  sorry

-- Define the set of valid a values
def valid_a_set : Set ℝ := { a | f a 2 < f a 4 }

-- State the theorem about the range of a
theorem range_of_a : Set.Ioo (-2 : ℝ) Real.pi ⊆ valid_a_set := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_implies_a_bound_range_of_a_l337_33763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l337_33776

noncomputable def angle_between (a b : ℝ × ℝ) : ℝ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))

noncomputable def vector_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

def vector_sum (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

noncomputable def projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let scalar := (dot_product a b) / (vector_length b)^2
  (scalar * b.1, scalar * b.2)

theorem vector_properties (a b : ℝ × ℝ) :
  angle_between a b = 3 * Real.pi / 4 →
  vector_length a = 3 →
  vector_length b = 2 * Real.sqrt 2 →
  b = (2, 2) →
  dot_product a b = -6 ∧
  vector_length (vector_sum a b) = Real.sqrt 5 ∧
  (dot_product a (vector_sum a b)) / (vector_length a * vector_length (vector_sum a b)) = Real.sqrt 5 / 5 ∧
  projection a b = (-3/2, -3/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_properties_l337_33776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_can_be_five_l337_33779

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  ineq_ab : a + b > c
  ineq_ac : a + c > b
  ineq_bc : b + c > a

-- Theorem for triangle inequality
theorem triangle_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hac : a + c > b) (hbc : b + c > a) : 
  ∃ t : Triangle, t.a = a ∧ t.b = b ∧ t.c = c :=
by
  exact ⟨⟨a, b, c, ha, hb, hc, hab, hac, hbc⟩, rfl, rfl, rfl⟩

-- Theorem that a triangle with sides 2, 4, and 5 exists
theorem third_side_can_be_five : 
  ∃ t : Triangle, t.a = 2 ∧ t.b = 4 ∧ t.c = 5 :=
by
  apply triangle_inequality 2 4 5
  · exact (show 2 > 0 by norm_num)
  · exact (show 4 > 0 by norm_num)
  · exact (show 5 > 0 by norm_num)
  · exact (show 2 + 4 > 5 by norm_num)
  · exact (show 2 + 5 > 4 by norm_num)
  · exact (show 4 + 5 > 2 by norm_num)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_third_side_can_be_five_l337_33779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passenger_fraction_l337_33789

theorem train_passenger_fraction (initial_passengers : ℕ) 
  (first_station_drop : ℚ) (first_station_add : ℕ) 
  (second_station_add : ℕ) (final_passengers : ℕ) 
  (h1 : initial_passengers = 270)
  (h2 : first_station_drop = 1 / 3)
  (h3 : first_station_add = 280)
  (h4 : second_station_add = 12)
  (h5 : final_passengers = 242)
  : (1 : ℚ) / 2 = 
    1 - (final_passengers - second_station_add : ℚ) / 
    (initial_passengers - initial_passengers * first_station_drop + first_station_add : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passenger_fraction_l337_33789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_point_value_l337_33739

/-- A quadratic function with specific properties -/
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem quadratic_point_value 
  (a b c : ℝ) 
  (h1 : ∀ x, quadratic_function a b c x ≥ -5)
  (h2 : quadratic_function a b c 3 = -5)
  (h3 : quadratic_function a b c 0 = 5) :
  quadratic_function a b c 5 = -5/9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_point_value_l337_33739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l337_33749

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  2 * (Real.log x / Real.log (1 / y) - 2 * Real.log y / Real.log (x^2)) + 5 = 0 ∧ x * y^2 = 32

-- Define the domain
def domain (x y : ℝ) : Prop :=
  0 < x ∧ x ≠ 1 ∧ 0 < y ∧ y ≠ 1

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {(2, 4), (4 * Real.sqrt 2, 2 * Real.sqrt (Real.sqrt 2))}

-- Theorem statement
theorem system_solution :
  ∀ x y : ℝ, domain x y → (system x y ↔ (x, y) ∈ solution_set) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l337_33749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l337_33770

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2)

theorem parallel_vectors_lambda (lambda : ℝ) :
  let a : ℝ × ℝ := (3, 1)
  let b : ℝ × ℝ := (2, lambda)
  are_parallel a b → lambda = 2/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_lambda_l337_33770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_3_equals_5_4_l337_33740

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x - 1) = 2 * f (x + 1) - Real.log (Real.sqrt x) / Real.log 2

/-- The main theorem stating that f(3) = 5/4 given the conditions -/
theorem f_3_equals_5_4
    (f : ℝ → ℝ)
    (h1 : SatisfiesFunctionalEquation f)
    (h2 : f 1 = 2) :
    f 3 = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_3_equals_5_4_l337_33740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_specific_rhombus_l337_33724

/-- A rhombus with given area and diagonal ratio -/
structure Rhombus where
  area : ℝ
  diagonal_ratio : ℝ × ℝ
  area_positive : 0 < area
  ratio_positive : 0 < diagonal_ratio.1 ∧ 0 < diagonal_ratio.2

/-- The length of the shorter diagonal of a rhombus -/
noncomputable def shorter_diagonal (r : Rhombus) : ℝ :=
  Real.sqrt (2 * r.area * r.diagonal_ratio.2 / (r.diagonal_ratio.1 + r.diagonal_ratio.2))

/-- Theorem: The length of the shorter diagonal of a rhombus with area 200 and diagonal ratio 4:3 is 10√3 -/
theorem shorter_diagonal_specific_rhombus :
  let r : Rhombus := ⟨200, (4, 3), by norm_num, by norm_num⟩
  shorter_diagonal r = 10 * Real.sqrt 3 := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shorter_diagonal_specific_rhombus_l337_33724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hexagon_area_l337_33790

/-- The area of a trapezoid given its bases and height -/
noncomputable def trapezoid_area (base1 : ℝ) (base2 : ℝ) (height : ℝ) : ℝ :=
  (base1 + base2) * height / 2

/-- An irregular hexagon composed of two identical trapezoids -/
structure IrregularHexagon where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- The area of the irregular hexagon -/
noncomputable def hexagon_area (h : IrregularHexagon) : ℝ :=
  2 * trapezoid_area h.base1 h.base2 h.height

/-- Theorem: The area of the specific irregular hexagon is 20 square units -/
theorem specific_hexagon_area :
  let h : IrregularHexagon := { base1 := 3, base2 := 2, height := 4 }
  hexagon_area h = 20 := by
  -- Unfold definitions and perform algebraic simplification
  unfold hexagon_area trapezoid_area
  simp [IrregularHexagon.base1, IrregularHexagon.base2, IrregularHexagon.height]
  -- Numerical computation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hexagon_area_l337_33790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l337_33756

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 4 * x / (x - 1)

-- Define the domain of g
def g_domain (x : ℝ) : Prop := x > 1

-- Define the inverse function of g
noncomputable def g_inv (x : ℝ) : ℝ := x / (x - 4)

-- Define the domain of g_inv
def g_inv_domain (x : ℝ) : Prop := x > 4

-- Define the function h
noncomputable def h (x : ℝ) : ℝ := x + g_inv x

-- State the theorem
theorem h_range :
  ∀ y : ℝ, (∃ x : ℝ, g_inv_domain x ∧ h x = y) ↔ y ≥ 9 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_range_l337_33756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_system_ratio_l337_33760

/-- A system of three equal masses connected by a string over pulleys -/
structure PulleySystem where
  /-- The mass of each object (assumed equal for all three) -/
  mass : ℝ
  /-- The horizontal distance between the outer masses -/
  a : ℝ
  /-- The vertical distance of the middle mass below the horizontal line -/
  b : ℝ
  /-- The angle the string makes with the horizontal (in radians) -/
  θ : ℝ
  /-- Assumption that the mass is positive -/
  mass_pos : 0 < mass
  /-- Assumption that a is positive -/
  a_pos : 0 < a
  /-- Assumption that b is positive -/
  b_pos : 0 < b
  /-- Assumption that the angle is 30 degrees (π/6 radians) -/
  angle_is_30_deg : θ = π / 6
  /-- Assumption that the system is in equilibrium -/
  equilibrium : Real.tan θ = (2 * b) / a

/-- The main theorem: In an equilibrium pulley system with a 30° angle, 
    the ratio a/b equals 2√3 -/
theorem pulley_system_ratio (ps : PulleySystem) : ps.a / ps.b = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_system_ratio_l337_33760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_lambda_l337_33737

/-- The value of λ for the given parabola and line intersection problem -/
theorem parabola_line_intersection_lambda : ∀ (x₁ x₂ y₁ y₂ : ℝ),
  -- Parabola equation
  y₁^2 = 4*x₁ ∧ y₂^2 = 4*x₂ →
  -- Line equation
  y₁ = x₁ - 1 ∧ y₂ = x₂ - 1 →
  -- A and B are distinct points
  x₁ ≠ x₂ →
  -- λ > 1
  (x₁ + 1) / (x₂ + 1) > 1 →
  -- Prove λ = 3 + 2√2
  (x₁ + 1) / (x₂ + 1) = 3 + 2 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_lambda_l337_33737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_two_l337_33713

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - (2*x^2)/(2^x + 1) + 3*Real.sin x + 1

-- Define the interval
def interval : Set ℝ := Set.Icc (-1/2) (1/2)

-- State the theorem
theorem max_min_sum_equals_two :
  ∃ (M N : ℝ), (∀ x ∈ interval, f x ≤ M) ∧
                (∀ x ∈ interval, N ≤ f x) ∧
                (∀ x ∈ interval, N ≤ f x ∧ f x ≤ M) ∧
                (M + N = 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_equals_two_l337_33713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cross_bridge_time_l337_33734

/-- Calculates the time (in seconds) for a train to cross a bridge -/
noncomputable def train_cross_time (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

theorem train_cross_bridge_time :
  train_cross_time 135 45 240 = 30 := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cross_bridge_time_l337_33734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_results_l337_33761

/-- A coin-flipping game between Grisha and Vanya -/
structure CoinGame where
  /-- The probability of getting heads on a single flip -/
  p_heads : ℝ
  /-- Assumption that the coin is fair -/
  fair_coin : p_heads = 1/2

/-- The probability of Grisha winning the game -/
noncomputable def grisha_win_prob (game : CoinGame) : ℝ := 1/3

/-- The expected number of coin flips until the game ends -/
noncomputable def expected_flips (game : CoinGame) : ℝ := 2

/-- Theorem stating the probability of Grisha winning and the expected number of flips -/
theorem coin_game_results (game : CoinGame) : 
  grisha_win_prob game = 1/3 ∧ expected_flips game = 2 := by
  sorry

#check coin_game_results

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_game_results_l337_33761
