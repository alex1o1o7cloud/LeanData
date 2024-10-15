import Mathlib

namespace NUMINAMATH_CALUDE_simon_candy_count_l1391_139165

def candy_problem (initial_candies : ℕ) : Prop :=
  let day1_remaining := initial_candies - (initial_candies / 4) - 3
  let day2_remaining := day1_remaining - (day1_remaining / 2) - 5
  let day3_remaining := day2_remaining - ((3 * day2_remaining) / 4) - 6
  day3_remaining = 4

theorem simon_candy_count : 
  ∃ (x : ℕ), candy_problem x ∧ x = 124 :=
sorry

end NUMINAMATH_CALUDE_simon_candy_count_l1391_139165


namespace NUMINAMATH_CALUDE_derivative_at_one_l1391_139113

-- Define the function
def f (x : ℝ) : ℝ := (2 * x + 1) ^ 2

-- State the theorem
theorem derivative_at_one :
  deriv f 1 = 12 := by sorry

end NUMINAMATH_CALUDE_derivative_at_one_l1391_139113


namespace NUMINAMATH_CALUDE_biased_coin_probability_l1391_139132

theorem biased_coin_probability (p : ℝ) (h1 : 0 < p) (h2 : p < 1)
  (h3 : 5 * p * (1 - p)^4 = 10 * p^2 * (1 - p)^3)
  (h4 : 5 * p * (1 - p)^4 ≠ 0) :
  10 * p^3 * (1 - p)^2 = 40 / 243 := by
  sorry

end NUMINAMATH_CALUDE_biased_coin_probability_l1391_139132


namespace NUMINAMATH_CALUDE_shifted_parabola_equation_l1391_139110

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the vertical shift amount
def shift_amount : ℝ := 4

-- Define the shifted parabola function
def shifted_parabola (x : ℝ) : ℝ := original_parabola x - shift_amount

-- Theorem stating that the shifted parabola has the correct form
theorem shifted_parabola_equation : 
  ∀ x : ℝ, shifted_parabola x = 2 * x^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_shifted_parabola_equation_l1391_139110


namespace NUMINAMATH_CALUDE_initial_volumes_l1391_139192

/-- Represents the initial state and operations on three cubic containers --/
structure ContainerSystem where
  -- Capacities of the containers
  c₁ : ℝ
  c₂ : ℝ
  c₃ : ℝ
  -- Initial volumes of liquid
  v₁ : ℝ
  v₂ : ℝ
  v₃ : ℝ
  -- Ratio of capacities
  hCapRatio : c₂ = 8 * c₁ ∧ c₃ = 27 * c₁
  -- Ratio of initial volumes
  hVolRatio : v₂ = 2 * v₁ ∧ v₃ = 3 * v₁
  -- Total volume remains constant
  hTotalVol : ℝ
  hTotalVolDef : hTotalVol = v₁ + v₂ + v₃
  -- Volume transferred in final operation
  transferVol : ℝ
  hTransferVol : transferVol = 128 + 4/7
  -- Final state properties
  hFinalState : ∃ (h₁ h₂ : ℝ),
    h₁ > 0 ∧ h₂ > 0 ∧
    h₁ * c₁ + transferVol = v₁ - 100 ∧
    h₂ * c₂ - transferVol = v₂ ∧
    h₁ = 2 * h₂

/-- Theorem stating the initial volumes of liquid in the three containers --/
theorem initial_volumes (s : ContainerSystem) : 
  s.v₁ = 350 ∧ s.v₂ = 700 ∧ s.v₃ = 1050 := by
  sorry


end NUMINAMATH_CALUDE_initial_volumes_l1391_139192


namespace NUMINAMATH_CALUDE_vehicle_original_value_l1391_139134

/-- The original value of a vehicle given its insurance details -/
def original_value (insured_fraction : ℚ) (premium : ℚ) (premium_rate : ℚ) : ℚ :=
  premium / (premium_rate / 100) / insured_fraction

/-- Theorem stating the original value of the vehicle -/
theorem vehicle_original_value :
  original_value (4/5) 910 1.3 = 87500 := by
  sorry

end NUMINAMATH_CALUDE_vehicle_original_value_l1391_139134


namespace NUMINAMATH_CALUDE_diamond_set_eq_three_lines_l1391_139108

/-- Define the ⋄ operation -/
def diamond (a b : ℝ) : ℝ := a^2 * b - a * b^2

/-- The set of points (x, y) where x ⋄ y = y ⋄ x -/
def diamond_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- The union of three lines: x = 0, y = 0, and x = y -/
def three_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = 0 ∨ p.2 = 0 ∨ p.1 = p.2}

theorem diamond_set_eq_three_lines : diamond_set = three_lines := by
  sorry

end NUMINAMATH_CALUDE_diamond_set_eq_three_lines_l1391_139108


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l1391_139194

theorem triangle_inequality_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  a / (b + c) + b / (a + c) + c / (a + b) < 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l1391_139194


namespace NUMINAMATH_CALUDE_A_intersection_Z_l1391_139124

def A : Set ℝ := {x : ℝ | |x - 1| < 2}

theorem A_intersection_Z : A ∩ (Set.range (Int.cast : ℤ → ℝ)) = {0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_A_intersection_Z_l1391_139124


namespace NUMINAMATH_CALUDE_f_inequality_l1391_139156

noncomputable def f (x : ℝ) : ℝ := Real.exp (-(x - 1)^2)

theorem f_inequality : f (Real.sqrt 3 / 2) > f (Real.sqrt 6 / 2) ∧ f (Real.sqrt 6 / 2) > f (Real.sqrt 2 / 2) := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1391_139156


namespace NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l1391_139120

theorem abs_p_minus_q_equals_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_p_minus_q_equals_five_l1391_139120


namespace NUMINAMATH_CALUDE_donut_selection_equals_object_distribution_l1391_139172

/-- The number of ways to select n donuts from k types with at least one of a specific type -/
def donut_selections (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 2) (k - 1)

/-- The number of ways to distribute m objects into k distinct boxes -/
def object_distribution (m k : ℕ) : ℕ :=
  Nat.choose (m + k - 1) (k - 1)

theorem donut_selection_equals_object_distribution :
  donut_selections 5 4 = object_distribution 4 4 :=
by sorry

end NUMINAMATH_CALUDE_donut_selection_equals_object_distribution_l1391_139172


namespace NUMINAMATH_CALUDE_running_preference_related_to_gender_l1391_139197

/-- Represents the contingency table for students liking running --/
structure RunningPreference where
  total_students : Nat
  boys : Nat
  girls : Nat
  girls_like_running : Nat
  boys_dont_like_running : Nat

/-- Calculates the K^2 value for the contingency table --/
def calculate_k_squared (pref : RunningPreference) : Rat :=
  let boys_like_running := pref.boys - pref.boys_dont_like_running
  let girls_dont_like_running := pref.girls - pref.girls_like_running
  let N := pref.total_students
  let a := boys_like_running
  let b := pref.boys_dont_like_running
  let c := pref.girls_like_running
  let d := girls_dont_like_running
  (N * (a * d - b * c)^2 : Rat) / ((a + c) * (b + d) * (a + b) * (c + d))

/-- Theorem stating that the K^2 value exceeds the critical value --/
theorem running_preference_related_to_gender (pref : RunningPreference) 
  (h1 : pref.total_students = 200)
  (h2 : pref.boys = 120)
  (h3 : pref.girls = 80)
  (h4 : pref.girls_like_running = 30)
  (h5 : pref.boys_dont_like_running = 50)
  (critical_value : Rat := 6635 / 1000) :
  calculate_k_squared pref > critical_value := by
  sorry

end NUMINAMATH_CALUDE_running_preference_related_to_gender_l1391_139197


namespace NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1391_139118

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 10th term is 20 and the 11th term is 24,
    the 2nd term of the sequence is -12. -/
theorem arithmetic_sequence_second_term
  (a : ℕ → ℝ)
  (h_arithmetic : ArithmeticSequence a)
  (h_10th : a 10 = 20)
  (h_11th : a 11 = 24) :
  a 2 = -12 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_second_term_l1391_139118


namespace NUMINAMATH_CALUDE_gcd_1343_816_l1391_139160

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1343_816_l1391_139160


namespace NUMINAMATH_CALUDE_range_of_m_l1391_139137

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 4*x - 6

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ∈ Set.Icc (-10) (-6)) ∧
  (∀ y ∈ Set.Icc (-10) (-6), ∃ x ∈ Set.Icc 0 m, f x = y) →
  m ∈ Set.Icc 2 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1391_139137


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1391_139142

/-- The solution set of the quadratic inequality ax^2 + 3x - 2 < 0 --/
def SolutionSet (a b : ℝ) : Set ℝ := {x | x < 1 ∨ x > b}

/-- The quadratic function ax^2 + 3x - 2 --/
def QuadraticFunction (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 3*x - 2

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, QuadraticFunction a x < 0 ↔ x ∈ SolutionSet a b) →
  a = -1 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1391_139142


namespace NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1391_139185

theorem not_in_range_iff_b_in_interval (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x + 5 ≠ -2) ↔ b ∈ Set.Ioo (-Real.sqrt 28) (Real.sqrt 28) := by
  sorry

end NUMINAMATH_CALUDE_not_in_range_iff_b_in_interval_l1391_139185


namespace NUMINAMATH_CALUDE_miss_evans_class_contribution_l1391_139193

/-- Calculates the original contribution amount for a class given the number of students,
    individual contribution after using class funds, and the amount of class funds used. -/
def originalContribution (numStudents : ℕ) (individualContribution : ℕ) (classFunds : ℕ) : ℕ :=
  numStudents * individualContribution + classFunds

/-- Proves that for Miss Evans' class, the original contribution amount was $90. -/
theorem miss_evans_class_contribution :
  originalContribution 19 4 14 = 90 := by
  sorry

end NUMINAMATH_CALUDE_miss_evans_class_contribution_l1391_139193


namespace NUMINAMATH_CALUDE_increase_by_percentage_l1391_139163

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 450 → percentage = 75 → result = initial * (1 + percentage / 100) → result = 787.5 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l1391_139163


namespace NUMINAMATH_CALUDE_equation_solutions_l1391_139157

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 9 = 0 ↔ x = 3 ∨ x = -3) ∧
  (∀ x : ℝ, (-x)^3 = (-8)^2 ↔ x = -4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1391_139157


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1391_139136

theorem reciprocal_of_negative_three :
  (1 : ℚ) / (-3 : ℚ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l1391_139136


namespace NUMINAMATH_CALUDE_digit_reversal_difference_exists_198_difference_l1391_139188

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_units : units < 10

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses the hundreds and units digits of a three-digit number -/
def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ThreeDigitNumber :=
  { hundreds := n.units
    tens := n.tens
    units := n.hundreds
    h_hundreds := n.h_units
    h_tens := n.h_tens
    h_units := n.h_hundreds }

theorem digit_reversal_difference (n : ThreeDigitNumber) :
  ∃ k : ℕ, n.value - n.reverse.value = 99 * k ∨ n.reverse.value - n.value = 99 * k :=
sorry

theorem exists_198_difference :
  ∃ n : ThreeDigitNumber, n.value - n.reverse.value = 198 ∨ n.reverse.value - n.value = 198 :=
sorry

end NUMINAMATH_CALUDE_digit_reversal_difference_exists_198_difference_l1391_139188


namespace NUMINAMATH_CALUDE_probability_three_men_l1391_139138

/-- The probability of selecting 3 men out of 3 selections from a workshop with 7 men and 3 women -/
theorem probability_three_men (total : ℕ) (men : ℕ) (women : ℕ) (selections : ℕ) :
  total = men + women →
  total = 10 →
  men = 7 →
  women = 3 →
  selections = 3 →
  (men.choose selections : ℚ) / (total.choose selections) = 7 / 24 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_men_l1391_139138


namespace NUMINAMATH_CALUDE_barbecue_sauce_ketchup_amount_l1391_139145

theorem barbecue_sauce_ketchup_amount :
  let total_sauce := k + 1 + 1
  let burger_sauce := (1 : ℚ) / 4
  let sandwich_sauce := (1 : ℚ) / 6
  let num_burgers := 8
  let num_sandwiches := 18
  ∀ k : ℚ,
  (num_burgers * burger_sauce + num_sandwiches * sandwich_sauce = total_sauce) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_barbecue_sauce_ketchup_amount_l1391_139145


namespace NUMINAMATH_CALUDE_simplify_expression_l1391_139119

theorem simplify_expression (y : ℝ) : (3/2 - 5*y) - (5/2 + 7*y) = -1 - 12*y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1391_139119


namespace NUMINAMATH_CALUDE_h_zero_iff_b_eq_two_l1391_139175

def h (x : ℝ) : ℝ := 5 * x - 10

theorem h_zero_iff_b_eq_two : ∀ b : ℝ, h b = 0 ↔ b = 2 := by sorry

end NUMINAMATH_CALUDE_h_zero_iff_b_eq_two_l1391_139175


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1391_139147

theorem complex_equation_solution (z : ℂ) : 
  Complex.abs z + z = 2 + 4 * Complex.I → z = -3 + 4 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1391_139147


namespace NUMINAMATH_CALUDE_multiply_mixed_number_l1391_139169

theorem multiply_mixed_number : 8 * (11 + 1/4) = 90 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l1391_139169


namespace NUMINAMATH_CALUDE_negation_equivalence_l1391_139152

def exactly_one_even (a b c : ℕ) : Prop :=
  (Even a ∧ Odd b ∧ Odd c) ∨ (Odd a ∧ Even b ∧ Odd c) ∨ (Odd a ∧ Odd b ∧ Even c)

def all_odd_or_at_least_two_even (a b c : ℕ) : Prop :=
  (Odd a ∧ Odd b ∧ Odd c) ∨ (Even a ∧ Even b) ∨ (Even a ∧ Even c) ∨ (Even b ∧ Even c)

theorem negation_equivalence (a b c : ℕ) :
  ¬(exactly_one_even a b c) ↔ all_odd_or_at_least_two_even a b c := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1391_139152


namespace NUMINAMATH_CALUDE_juanita_daily_cost_l1391_139189

/-- The amount Juanita spends on a newspaper from Monday through Saturday -/
def daily_cost : ℝ := sorry

/-- Grant's yearly newspaper cost -/
def grant_yearly_cost : ℝ := 200

/-- Juanita's Sunday newspaper cost -/
def sunday_cost : ℝ := 2

/-- Number of weeks in a year -/
def weeks_per_year : ℕ := 52

/-- Difference between Juanita's and Grant's yearly newspaper costs -/
def cost_difference : ℝ := 60

theorem juanita_daily_cost :
  daily_cost * 6 * weeks_per_year + sunday_cost * weeks_per_year = 
  grant_yearly_cost + cost_difference :=
by sorry

end NUMINAMATH_CALUDE_juanita_daily_cost_l1391_139189


namespace NUMINAMATH_CALUDE_existence_of_larger_prime_factor_l1391_139100

theorem existence_of_larger_prime_factor (p : ℕ) (hp : Prime p) (hp_ge_3 : p ≥ 3) :
  ∃ N : ℕ, ∀ x ≥ N, ∃ i ∈ Finset.range ((p + 3) / 2), ∃ q : ℕ, Prime q ∧ q > p ∧ q ∣ (x + i + 1) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_larger_prime_factor_l1391_139100


namespace NUMINAMATH_CALUDE_abs_S_equals_512_l1391_139117

-- Define the complex number i
def i : ℂ := Complex.I

-- Define S
def S : ℂ := (1 + i)^17 - (1 - i)^17

-- Theorem statement
theorem abs_S_equals_512 : Complex.abs S = 512 := by sorry

end NUMINAMATH_CALUDE_abs_S_equals_512_l1391_139117


namespace NUMINAMATH_CALUDE_shifted_quadratic_sum_of_coefficients_l1391_139125

-- Define the original quadratic function
def f (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 5

-- Define the shifted function
def g (x : ℝ) : ℝ := f (x + 3)

-- Theorem statement
theorem shifted_quadratic_sum_of_coefficients :
  ∃ (a b c : ℝ), (∀ x, g x = a * x^2 + b * x + c) ∧ (a + b + c = 51) := by
sorry

end NUMINAMATH_CALUDE_shifted_quadratic_sum_of_coefficients_l1391_139125


namespace NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l1391_139144

/-- Arithmetic sequence with first term a and common difference d -/
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1 : ℤ) * d

/-- The 75th term of the arithmetic sequence starting with 3 and common difference 5 is 373 -/
theorem arithmetic_sequence_75th_term :
  arithmetic_sequence 3 5 75 = 373 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_75th_term_l1391_139144


namespace NUMINAMATH_CALUDE_original_population_multiple_of_three_l1391_139159

theorem original_population_multiple_of_three (x y z : ℕ) 
  (h1 : y * y = x * x + 121)
  (h2 : z * z = y * y + 121) : 
  ∃ k : ℕ, x * x = 3 * k :=
sorry

end NUMINAMATH_CALUDE_original_population_multiple_of_three_l1391_139159


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l1391_139123

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw_x : w / x = 4 / 3)
  (hy_z : y / z = 5 / 3)
  (hz_x : z / x = 1 / 6) :
  w / y = 24 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l1391_139123


namespace NUMINAMATH_CALUDE_simplify_expression_l1391_139111

theorem simplify_expression (a b : ℝ) :
  (33 * a + 75 * b + 12) + (15 * a + 44 * b + 7) - (12 * a + 65 * b + 5) = 36 * a + 54 * b + 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1391_139111


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1391_139191

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 + 6*x + 8 = -(x + 2)*(x + 6) ↔ x = -2 ∨ x = -5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1391_139191


namespace NUMINAMATH_CALUDE_divisibility_implies_k_value_l1391_139173

/-- 
If x^2 + kx - 3 is divisible by (x - 1), then k = 2.
-/
theorem divisibility_implies_k_value (k : ℤ) : 
  (∀ x : ℤ, (x - 1) ∣ (x^2 + k*x - 3)) → k = 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_implies_k_value_l1391_139173


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1391_139122

theorem solution_set_inequality (a b : ℝ) (h : |a - b| > 2) :
  ∀ x : ℝ, |x - a| + |x - b| > 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1391_139122


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1391_139162

/-- A quadratic polynomial with nonnegative coefficients -/
structure NonnegQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The value of a quadratic polynomial at a given point -/
def evalQuadratic (P : NonnegQuadratic) (x : ℝ) : ℝ :=
  P.a * x^2 + P.b * x + P.c

/-- Theorem: For any quadratic polynomial with nonnegative coefficients and any real numbers x and y,
    the inequality P(xy)^2 ≤ P(x^2)P(y^2) holds -/
theorem quadratic_inequality (P : NonnegQuadratic) (x y : ℝ) :
    (evalQuadratic P (x * y))^2 ≤ (evalQuadratic P (x^2)) * (evalQuadratic P (y^2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1391_139162


namespace NUMINAMATH_CALUDE_deepak_age_l1391_139105

/-- Given that the ratio of Rahul's age to Deepak's age is 4:3 and 
    Rahul's age after 6 years will be 34 years, 
    prove that Deepak's present age is 21 years. -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 34 →
  deepak_age = 21 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l1391_139105


namespace NUMINAMATH_CALUDE_percentage_relationship_l1391_139168

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 0.375)) :
  y = x * 1.6 :=
sorry

end NUMINAMATH_CALUDE_percentage_relationship_l1391_139168


namespace NUMINAMATH_CALUDE_cuboid_volume_l1391_139128

/-- The volume of a cuboid with given edge lengths -/
theorem cuboid_volume (l w h : ℝ) (hl : l = 3/2 + Real.sqrt (5/3)) 
  (hw : w = 2 + Real.sqrt (3/5)) (hh : h = π / 2) : 
  l * w * h = (3/2 + Real.sqrt (5/3)) * (2 + Real.sqrt (3/5)) * (π / 2) := by
  sorry

#check cuboid_volume

end NUMINAMATH_CALUDE_cuboid_volume_l1391_139128


namespace NUMINAMATH_CALUDE_bee_count_l1391_139101

theorem bee_count (initial_bees new_bees : ℕ) : 
  initial_bees = 16 → new_bees = 7 → initial_bees + new_bees = 23 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l1391_139101


namespace NUMINAMATH_CALUDE_f_properties_l1391_139158

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x) * Real.cos φ + 2 * Real.cos (ω * x) * Real.sin φ

theorem f_properties (ω φ : ℝ) (h_ω : ω > 0) (h_φ : abs φ < π / 2) (h_period : ∀ x, f ω φ (x + π) = f ω φ x) :
  ∃ φ',
    (∀ x, f ω φ x = 2 * Real.sin (2 * x + φ')) ∧
    (∀ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x ≤ 2) ∧
    (∀ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x ≥ 0) ∧
    (∃ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x = 2) ∧
    ((∃ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x = 0) ∨
     (∀ x ∈ Set.Icc (π / 6) (π / 2), f ω φ x ≥ 1)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1391_139158


namespace NUMINAMATH_CALUDE_int_tan_triangle_unique_l1391_139166

/-- A triangle with integer tangents for all angles -/
structure IntTanTriangle where
  α : Real
  β : Real
  γ : Real
  sum_180 : α + β + γ = Real.pi
  tan_int_α : ∃ m : Int, Real.tan α = m
  tan_int_β : ∃ n : Int, Real.tan β = n
  tan_int_γ : ∃ k : Int, Real.tan γ = k

/-- The only possible combination of integer tangents for a triangle is (1, 2, 3) -/
theorem int_tan_triangle_unique (t : IntTanTriangle) :
  (Real.tan t.α = 1 ∧ Real.tan t.β = 2 ∧ Real.tan t.γ = 3) ∨
  (Real.tan t.α = 1 ∧ Real.tan t.β = 3 ∧ Real.tan t.γ = 2) ∨
  (Real.tan t.α = 2 ∧ Real.tan t.β = 1 ∧ Real.tan t.γ = 3) ∨
  (Real.tan t.α = 2 ∧ Real.tan t.β = 3 ∧ Real.tan t.γ = 1) ∨
  (Real.tan t.α = 3 ∧ Real.tan t.β = 1 ∧ Real.tan t.γ = 2) ∨
  (Real.tan t.α = 3 ∧ Real.tan t.β = 2 ∧ Real.tan t.γ = 1) :=
by sorry

end NUMINAMATH_CALUDE_int_tan_triangle_unique_l1391_139166


namespace NUMINAMATH_CALUDE_female_students_count_l1391_139104

theorem female_students_count (total_average : ℚ) (male_count : ℕ) (male_average : ℚ) (female_average : ℚ) :
  total_average = 90 →
  male_count = 8 →
  male_average = 82 →
  female_average = 92 →
  ∃ (female_count : ℕ),
    (male_count * male_average + female_count * female_average) / (male_count + female_count) = total_average ∧
    female_count = 32 :=
by sorry

end NUMINAMATH_CALUDE_female_students_count_l1391_139104


namespace NUMINAMATH_CALUDE_similar_triangles_side_proportional_l1391_139112

/-- Two triangles are similar if their corresponding angles are equal -/
def similar_triangles (t1 t2 : Set (ℝ × ℝ)) : Prop := sorry

theorem similar_triangles_side_proportional 
  (G H I X Y Z : ℝ × ℝ) 
  (h_similar : similar_triangles {G, H, I} {X, Y, Z}) 
  (h_GH : dist G H = 8)
  (h_HI : dist H I = 20)
  (h_YZ : dist Y Z = 25) : 
  dist X Y = 80 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_side_proportional_l1391_139112


namespace NUMINAMATH_CALUDE_f_properties_l1391_139177

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then -x - 1
  else if -1 ≤ x ∧ x ≤ 1 then -x^2 + 1
  else x - 1

-- Theorem statement
theorem f_properties :
  (f 2 = 1 ∧ f (-2) = 1) ∧
  (∀ a : ℝ, f a = 1 ↔ a = -2 ∨ a = 0 ∨ a = 2) ∧
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, (-1 ≤ x ∧ x < y ∧ y ≤ 0) → f x < f y) ∧
  (∀ x y : ℝ, (1 ≤ x ∧ x < y) → f x < f y) ∧
  (∀ x y : ℝ, (x < y ∧ y ≤ -1) → f x > f y) ∧
  (∀ x y : ℝ, (0 < x ∧ x < y ∧ y ≤ 1) → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1391_139177


namespace NUMINAMATH_CALUDE_work_to_pump_liquid_l1391_139164

/-- Work required to pump liquid from a paraboloid cauldron -/
theorem work_to_pump_liquid (R H γ : ℝ) (h_R : R > 0) (h_H : H > 0) (h_γ : γ > 0) :
  ∃ (W : ℝ), W = 240 * π * H^3 * γ / 9810 ∧ W > 0 := by
  sorry

end NUMINAMATH_CALUDE_work_to_pump_liquid_l1391_139164


namespace NUMINAMATH_CALUDE_A_necessary_not_sufficient_l1391_139186

-- Define proposition A
def proposition_A (a : ℝ) : Prop :=
  ∀ x, a * x^2 + 2 * a * x + 1 > 0

-- Define proposition B
def proposition_B (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Theorem stating that A is necessary but not sufficient for B
theorem A_necessary_not_sufficient :
  (∀ a, proposition_B a → proposition_A a) ∧
  ¬(∀ a, proposition_A a → proposition_B a) :=
sorry

end NUMINAMATH_CALUDE_A_necessary_not_sufficient_l1391_139186


namespace NUMINAMATH_CALUDE_solve_fish_problem_l1391_139109

def fish_problem (initial_fish : ℕ) (yearly_increase : ℕ) (years : ℕ) (final_fish : ℕ) : ℕ → Prop :=
  λ yearly_deaths : ℕ =>
    initial_fish + years * yearly_increase - years * yearly_deaths = final_fish

theorem solve_fish_problem :
  ∃ yearly_deaths : ℕ, fish_problem 2 2 5 7 yearly_deaths :=
by
  sorry

end NUMINAMATH_CALUDE_solve_fish_problem_l1391_139109


namespace NUMINAMATH_CALUDE_algebraic_expression_equality_l1391_139198

theorem algebraic_expression_equality (x : ℝ) : 
  3 * x^2 - 4 * x = 6 → 6 * x^2 - 8 * x - 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_equality_l1391_139198


namespace NUMINAMATH_CALUDE_tan_22_5_deg_representation_l1391_139153

theorem tan_22_5_deg_representation :
  ∃ (a b c d : ℕ+), 
    (a ≥ b) ∧ (b ≥ c) ∧ (c ≥ d) ∧
    (Real.tan (22.5 * π / 180) = Real.sqrt a - Real.sqrt b + Real.sqrt c - d) ∧
    (a + b + c + d = 10) := by
  sorry

end NUMINAMATH_CALUDE_tan_22_5_deg_representation_l1391_139153


namespace NUMINAMATH_CALUDE_base7_3652_equals_base10_1360_l1391_139135

/-- Converts a base-7 number to base-10 --/
def base7ToBase10 (a b c d : ℕ) : ℕ :=
  a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0

/-- The theorem stating that 3652 in base 7 is equal to 1360 in base 10 --/
theorem base7_3652_equals_base10_1360 : base7ToBase10 3 6 5 2 = 1360 := by
  sorry

end NUMINAMATH_CALUDE_base7_3652_equals_base10_1360_l1391_139135


namespace NUMINAMATH_CALUDE_equation_rewrite_and_product_l1391_139127

theorem equation_rewrite_and_product (a b x y : ℝ) (m n p : ℤ) :
  a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1) →
  ∃ m n p : ℤ, (a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5 ∧ m*n*p = 8 :=
by sorry

end NUMINAMATH_CALUDE_equation_rewrite_and_product_l1391_139127


namespace NUMINAMATH_CALUDE_inequality_proof_l1391_139180

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1/(2*a) + 1/(2*b) + 1/(2*c) ≥ 1/(b+c) + 1/(c+a) + 1/(a+b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1391_139180


namespace NUMINAMATH_CALUDE_hex_addition_l1391_139133

/-- Represents a hexadecimal digit --/
inductive HexDigit
| D0 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | A | B | C | D | E | F

/-- Converts a HexDigit to its decimal value --/
def hexToDecimal (h : HexDigit) : Nat :=
  match h with
  | HexDigit.D0 => 0
  | HexDigit.D1 => 1
  | HexDigit.D2 => 2
  | HexDigit.D3 => 3
  | HexDigit.D4 => 4
  | HexDigit.D5 => 5
  | HexDigit.D6 => 6
  | HexDigit.D7 => 7
  | HexDigit.D8 => 8
  | HexDigit.D9 => 9
  | HexDigit.A => 10
  | HexDigit.B => 11
  | HexDigit.C => 12
  | HexDigit.D => 13
  | HexDigit.E => 14
  | HexDigit.F => 15

/-- Converts a list of HexDigits to its decimal value --/
def hexListToDecimal (l : List HexDigit) : Nat :=
  l.foldr (fun d acc => 16 * acc + hexToDecimal d) 0

/-- Theorem: The sum of 7A3₁₆ and 1F4₁₆ is equal to 997₁₆ --/
theorem hex_addition : 
  let a := [HexDigit.D7, HexDigit.A, HexDigit.D3]
  let b := [HexDigit.D1, HexDigit.F, HexDigit.D4]
  let result := [HexDigit.D9, HexDigit.D9, HexDigit.D7]
  hexListToDecimal a + hexListToDecimal b = hexListToDecimal result := by
  sorry


end NUMINAMATH_CALUDE_hex_addition_l1391_139133


namespace NUMINAMATH_CALUDE_triangle_table_height_l1391_139181

theorem triangle_table_height (DE EF FD : ℝ) (hDE : DE = 20) (hEF : EF = 21) (hFD : FD = 29) :
  let s := (DE + EF + FD) / 2
  let A := Real.sqrt (s * (s - DE) * (s - EF) * (s - FD))
  let h_d := 2 * A / FD
  let h_f := 2 * A / EF
  let k := (h_f * h_d) / (h_f + h_d)
  k = 7 * Real.sqrt 210 / 5 := by sorry

end NUMINAMATH_CALUDE_triangle_table_height_l1391_139181


namespace NUMINAMATH_CALUDE_quadratic_real_roots_imply_a_equals_negative_one_l1391_139155

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℂ :=
  a * (1 + i) * x^2 + (1 + a^2 * i) * x + a^2 + i

-- Theorem statement
theorem quadratic_real_roots_imply_a_equals_negative_one :
  ∀ a : ℝ, (∃ x : ℝ, quadratic_equation a x = 0) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_imply_a_equals_negative_one_l1391_139155


namespace NUMINAMATH_CALUDE_erroneous_multiplication_l1391_139154

/-- Given two positive integers where one is a two-digit number,
    if the product of the digit-reversed two-digit number and the other integer is 161,
    then the product of the original numbers is 224. -/
theorem erroneous_multiplication (a b : ℕ) : 
  a ≥ 10 ∧ a ≤ 99 →  -- a is a two-digit number
  b > 0 →  -- b is positive
  (10 * (a % 10) + (a / 10)) * b = 161 →  -- reversed a multiplied by b is 161
  a * b = 224 :=
by sorry

end NUMINAMATH_CALUDE_erroneous_multiplication_l1391_139154


namespace NUMINAMATH_CALUDE_inequality_always_holds_l1391_139149

theorem inequality_always_holds (a b c : ℝ) (h1 : a > b) (h2 : a * b ≠ 0) :
  a + c > b + c := by sorry

end NUMINAMATH_CALUDE_inequality_always_holds_l1391_139149


namespace NUMINAMATH_CALUDE_point_movement_l1391_139199

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Given conditions and proof goal -/
theorem point_movement :
  let A : Point := ⟨a - 5, 2 * b - 1⟩
  let B : Point := ⟨3 * a + 2, b + 3⟩
  let C : Point := ⟨a, b⟩
  A.x = 0 →  -- A lies on y-axis
  B.y = 0 →  -- B lies on x-axis
  (⟨C.x + 2, C.y - 3⟩ : Point) = ⟨7, -6⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l1391_139199


namespace NUMINAMATH_CALUDE_rental_income_calculation_l1391_139176

theorem rental_income_calculation (total_units : ℕ) (occupancy_rate : ℚ) (monthly_rent : ℕ) :
  total_units = 100 →
  occupancy_rate = 3/4 →
  monthly_rent = 400 →
  (total_units : ℚ) * occupancy_rate * (monthly_rent : ℚ) * 12 = 360000 := by
  sorry

end NUMINAMATH_CALUDE_rental_income_calculation_l1391_139176


namespace NUMINAMATH_CALUDE_sine_angle_equality_l1391_139195

theorem sine_angle_equality (n : ℤ) : 
  -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * π / 180) = Real.sin (604 * π / 180) → n = -64 := by
  sorry

end NUMINAMATH_CALUDE_sine_angle_equality_l1391_139195


namespace NUMINAMATH_CALUDE_foreign_language_score_l1391_139103

theorem foreign_language_score (chinese_score math_score foreign_score : ℕ) : 
  (chinese_score + math_score + foreign_score) / 3 = 95 →
  (chinese_score + math_score) / 2 = 93 →
  foreign_score = 99 := by
sorry

end NUMINAMATH_CALUDE_foreign_language_score_l1391_139103


namespace NUMINAMATH_CALUDE_namjoon_lowest_l1391_139129

def board_A : ℝ := 2.4
def board_B : ℝ := 3.2
def board_C : ℝ := 2.8

def eunji_height : ℝ := 8 * board_A
def namjoon_height : ℝ := 4 * board_B
def hoseok_height : ℝ := 5 * board_C

theorem namjoon_lowest : 
  namjoon_height < eunji_height ∧ namjoon_height < hoseok_height :=
by sorry

end NUMINAMATH_CALUDE_namjoon_lowest_l1391_139129


namespace NUMINAMATH_CALUDE_unique_plants_count_l1391_139174

/-- Represents a flower bed -/
structure FlowerBed where
  plants : ℕ

/-- Represents the overlap between two flower beds -/
structure Overlap where
  plants : ℕ

/-- Represents the overlap among three flower beds -/
structure TripleOverlap where
  plants : ℕ

/-- Calculates the total number of unique plants in three overlapping flower beds -/
def totalUniquePlants (x y z : FlowerBed) (xy yz xz : Overlap) (xyz : TripleOverlap) : ℕ :=
  x.plants + y.plants + z.plants - xy.plants - yz.plants - xz.plants + xyz.plants

/-- Theorem stating that the total number of unique plants is 1320 -/
theorem unique_plants_count :
  let x : FlowerBed := ⟨600⟩
  let y : FlowerBed := ⟨480⟩
  let z : FlowerBed := ⟨420⟩
  let xy : Overlap := ⟨60⟩
  let yz : Overlap := ⟨70⟩
  let xz : Overlap := ⟨80⟩
  let xyz : TripleOverlap := ⟨30⟩
  totalUniquePlants x y z xy yz xz xyz = 1320 := by
  sorry

end NUMINAMATH_CALUDE_unique_plants_count_l1391_139174


namespace NUMINAMATH_CALUDE_rope_percentage_theorem_l1391_139140

theorem rope_percentage_theorem (total_length used_length : ℝ) 
  (h1 : total_length = 20)
  (h2 : used_length = 15) :
  used_length / total_length = 0.75 ∧ (1 - used_length / total_length) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_rope_percentage_theorem_l1391_139140


namespace NUMINAMATH_CALUDE_stating_tour_days_correct_l1391_139184

/-- Represents the number of days Mr. Bhaskar is on tour -/
def tour_days : ℕ := 20

/-- Total budget for the tour -/
def total_budget : ℕ := 360

/-- Number of days the tour could be extended -/
def extension_days : ℕ := 4

/-- Amount by which daily expenses must be reduced if tour is extended -/
def expense_reduction : ℕ := 3

/-- 
Theorem stating that tour_days satisfies the given conditions:
1. The total budget divided by tour_days gives the daily expense
2. If the tour is extended by extension_days, the new daily expense is 
   reduced by expense_reduction
3. The total expenditure remains the same in both scenarios
-/
theorem tour_days_correct : 
  (total_budget / tour_days) * tour_days = 
  ((total_budget / tour_days) - expense_reduction) * (tour_days + extension_days) := by
  sorry

#check tour_days_correct

end NUMINAMATH_CALUDE_stating_tour_days_correct_l1391_139184


namespace NUMINAMATH_CALUDE_dihedral_angle_cosine_l1391_139182

/-- Given two spheres inscribed in a dihedral angle, this theorem proves that
    the cosine of the measure of the dihedral angle is 5/9 under specific conditions. -/
theorem dihedral_angle_cosine (α : Real) (R r : Real) (β : Real) :
  -- Two spheres are inscribed in a dihedral angle
  -- The spheres touch each other
  -- R is the radius of the larger sphere, r is the radius of the smaller sphere
  (R = 2 * r) →
  -- The line connecting the centers of the spheres forms a 45° angle with the edge of the dihedral angle
  (β = Real.pi / 4) →
  -- α is the measure of the dihedral angle
  -- The cosine of the measure of the dihedral angle is 5/9
  (Real.cos α = 5 / 9) :=
by sorry

end NUMINAMATH_CALUDE_dihedral_angle_cosine_l1391_139182


namespace NUMINAMATH_CALUDE_grinder_price_is_15000_l1391_139102

/-- The price of a grinder and a mobile phone transaction --/
def GrinderMobileTransaction (grinder_price : ℝ) : Prop :=
  let mobile_price : ℝ := 8000
  let grinder_sell_price : ℝ := grinder_price * 0.96
  let mobile_sell_price : ℝ := mobile_price * 1.10
  let total_buy_price : ℝ := grinder_price + mobile_price
  let total_sell_price : ℝ := grinder_sell_price + mobile_sell_price
  total_sell_price - total_buy_price = 200

/-- The grinder price is 15000 given the transaction conditions --/
theorem grinder_price_is_15000 : 
  ∃ (price : ℝ), GrinderMobileTransaction price ∧ price = 15000 := by
  sorry

end NUMINAMATH_CALUDE_grinder_price_is_15000_l1391_139102


namespace NUMINAMATH_CALUDE_loan_problem_l1391_139148

/-- Represents a simple interest loan -/
structure SimpleLoan where
  principal : ℝ
  rate : ℝ
  time : ℝ
  interest : ℝ

/-- Theorem stating the conditions and conclusion of the loan problem -/
theorem loan_problem (loan : SimpleLoan) 
  (h1 : loan.time = loan.rate)
  (h2 : loan.interest = 108)
  (h3 : loan.rate = 0.03)
  (h4 : loan.interest = loan.principal * loan.rate * loan.time) :
  loan.principal = 1200 := by
  sorry

#check loan_problem

end NUMINAMATH_CALUDE_loan_problem_l1391_139148


namespace NUMINAMATH_CALUDE_S_tiles_integers_not_naturals_l1391_139114

def S : Set ℤ := {1, 3, 4, 6}

def tiles_integers (S : Set ℤ) : Prop :=
  ∀ n : ℤ, ∃ s ∈ S, ∃ k : ℤ, n = s + 4 * k

def tiles_naturals (S : Set ℤ) : Prop :=
  ∀ n : ℕ, ∃ s ∈ S, ∃ k : ℤ, (n : ℤ) = s + 4 * k

theorem S_tiles_integers_not_naturals :
  tiles_integers S ∧ ¬tiles_naturals S := by sorry

end NUMINAMATH_CALUDE_S_tiles_integers_not_naturals_l1391_139114


namespace NUMINAMATH_CALUDE_pin_pierces_all_sheets_l1391_139139

/-- Represents a rectangular sheet of paper -/
structure Sheet :=
  (width : ℝ)
  (height : ℝ)
  (center : ℝ × ℝ)

/-- Represents a collection of sheets on a table -/
structure TableSetup :=
  (sheets : List Sheet)
  (top_sheet : Sheet)

/-- Predicate to check if a point is on a sheet -/
def point_on_sheet (p : ℝ × ℝ) (s : Sheet) : Prop :=
  let (x, y) := p
  let (cx, cy) := s.center
  |x - cx| ≤ s.width / 2 ∧ |y - cy| ≤ s.height / 2

/-- The main theorem -/
theorem pin_pierces_all_sheets (setup : TableSetup) 
  (h_identical : ∀ s ∈ setup.sheets, s = setup.top_sheet)
  (h_cover : ∀ s ∈ setup.sheets, s ≠ setup.top_sheet → 
    (Set.inter (Set.range (point_on_sheet · setup.top_sheet)) 
               (Set.range (point_on_sheet · s))).ncard > 
    (Set.range (point_on_sheet · s)).ncard / 2) :
  ∃ p : ℝ × ℝ, ∀ s ∈ setup.sheets, point_on_sheet p s :=
sorry

end NUMINAMATH_CALUDE_pin_pierces_all_sheets_l1391_139139


namespace NUMINAMATH_CALUDE_min_value_and_inequality_range_l1391_139161

theorem min_value_and_inequality_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 3) :
  (∃ (min : ℝ), min = 6 ∧ ∀ x y, x > 0 → y > 0 → x * y = 3 → x + 3 * y ≥ min) ∧
  (∀ m : ℝ, (∀ x y, x > 0 → y > 0 → x * y = 3 → m^2 - (x + 3 * y) * m + 5 ≤ 0) → 1 ≤ m ∧ m ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_inequality_range_l1391_139161


namespace NUMINAMATH_CALUDE_binomial_10_3_l1391_139115

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l1391_139115


namespace NUMINAMATH_CALUDE_fraction_value_l1391_139167

/-- Given that x is four times y, y is three times z, and z is five times w,
    prove that (x * z) / (y * w) = 20 -/
theorem fraction_value (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) : 
  (x * z) / (y * w) = 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1391_139167


namespace NUMINAMATH_CALUDE_max_value_of_function_l1391_139146

theorem max_value_of_function (x : ℝ) (h : x > 0) : 2 - x - 4 / x ≤ -2 ∧ (2 - x - 4 / x = -2 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1391_139146


namespace NUMINAMATH_CALUDE_estimate_red_balls_l1391_139107

theorem estimate_red_balls (black_balls : ℕ) (total_draws : ℕ) (black_draws : ℕ) : 
  black_balls = 4 → 
  total_draws = 100 → 
  black_draws = 40 → 
  ∃ red_balls : ℕ, (black_balls : ℚ) / (black_balls + red_balls : ℚ) = 2 / 5 ∧ red_balls = 6 :=
by sorry

end NUMINAMATH_CALUDE_estimate_red_balls_l1391_139107


namespace NUMINAMATH_CALUDE_tangent_point_and_zeros_l1391_139178

noncomputable def f (a x : ℝ) : ℝ := 2 * Real.exp x + 2 * a * x - x + 3 - a^2

theorem tangent_point_and_zeros (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ (∀ y : ℝ, f a y ≥ 0)) ↔ a = Real.log 3 - 3 ∧
  (∀ x : ℝ, x > 0 →
    (((a ≤ -Real.sqrt 5 ∨ a = Real.log 3 - 3 ∨ a > Real.sqrt 5) →
      (∃! y : ℝ, y > 0 ∧ f a y = 0)) ∧
    ((-Real.sqrt 5 < a ∧ a < Real.log 3 - 3) →
      (∃ y z : ℝ, 0 < y ∧ y < z ∧ f a y = 0 ∧ f a z = 0 ∧
        ∀ w : ℝ, 0 < w ∧ w ≠ y ∧ w ≠ z → f a w ≠ 0)) ∧
    ((Real.log 3 - 3 < a ∧ a ≤ Real.sqrt 5) →
      (∀ y : ℝ, y > 0 → f a y ≠ 0)))) :=
sorry

end NUMINAMATH_CALUDE_tangent_point_and_zeros_l1391_139178


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l1391_139187

theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ) :
  total_distance = 540 →
  total_time = 15 →
  second_half_speed = 30 →
  (total_distance / 2) / ((total_time - (total_distance / 2) / second_half_speed)) = 45 :=
by sorry

end NUMINAMATH_CALUDE_journey_speed_calculation_l1391_139187


namespace NUMINAMATH_CALUDE_plane_q_satisfies_conditions_l1391_139190

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ

/-- Represents a point in 3D space -/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if a plane contains a line defined by the intersection of two other planes -/
def containsIntersectionLine (p : Plane) (p1 p2 : Plane) : Prop := sorry

/-- Calculates the distance from a plane to a point -/
def distanceToPoint (p : Plane) (pt : Point) : ℝ := sorry

/-- Checks if two planes are different -/
def areDifferentPlanes (p1 p2 : Plane) : Prop := sorry

/-- Calculates the greatest common divisor of four integers -/
def gcd4 (a b c d : ℤ) : ℕ := sorry

theorem plane_q_satisfies_conditions : 
  let p1 : Plane := { a := 2, b := -1, c := 3, d := -4 }
  let p2 : Plane := { a := 3, b := 2, c := -1, d := -6 }
  let q : Plane := { a := 0, b := -7, c := 11, d := -6 }
  let pt : Point := { x := 2, y := -2, z := 1 }
  containsIntersectionLine q p1 p2 ∧ 
  areDifferentPlanes q p1 ∧
  areDifferentPlanes q p2 ∧
  distanceToPoint q pt = 3 / Real.sqrt 5 ∧
  q.a > 0 ∧
  gcd4 (Int.natAbs q.a) (Int.natAbs q.b) (Int.natAbs q.c) (Int.natAbs q.d) = 1 := by
  sorry

end NUMINAMATH_CALUDE_plane_q_satisfies_conditions_l1391_139190


namespace NUMINAMATH_CALUDE_minimal_cost_proof_l1391_139116

/-- Represents an entity that can clean -/
inductive Cleaner
| Janitor
| Student
| Company

/-- Represents a location to be cleaned -/
inductive Location
| Classes
| Gym

/-- Time (in hours) it takes for a cleaner to clean a location -/
def cleaning_time (c : Cleaner) (l : Location) : ℕ :=
  match c, l with
  | Cleaner.Janitor, Location.Classes => 8
  | Cleaner.Janitor, Location.Gym => 6
  | Cleaner.Student, Location.Classes => 20
  | Cleaner.Student, Location.Gym => 0  -- Student cannot clean the gym
  | Cleaner.Company, Location.Classes => 10
  | Cleaner.Company, Location.Gym => 5

/-- Hourly rate (in dollars) for each cleaner -/
def hourly_rate (c : Cleaner) : ℕ :=
  match c with
  | Cleaner.Janitor => 21
  | Cleaner.Student => 7
  | Cleaner.Company => 60

/-- Cost for a cleaner to clean a location -/
def cleaning_cost (c : Cleaner) (l : Location) : ℕ :=
  (cleaning_time c l) * (hourly_rate c)

/-- The minimal cost to clean both the classes and the gym -/
def minimal_cleaning_cost : ℕ := 266

theorem minimal_cost_proof :
  ∀ (c1 c2 : Cleaner) (l1 l2 : Location),
    l1 ≠ l2 →
    cleaning_cost c1 l1 + cleaning_cost c2 l2 ≥ minimal_cleaning_cost :=
by sorry

end NUMINAMATH_CALUDE_minimal_cost_proof_l1391_139116


namespace NUMINAMATH_CALUDE_final_orange_count_l1391_139130

def initial_oranges : ℕ := 150

def sold_to_peter (n : ℕ) : ℕ := n - n * 20 / 100

def sold_to_paula (n : ℕ) : ℕ := n - n * 30 / 100

def give_to_neighbor (n : ℕ) : ℕ := n - 10

def give_to_teacher (n : ℕ) : ℕ := n - 1

theorem final_orange_count :
  give_to_teacher (give_to_neighbor (sold_to_paula (sold_to_peter initial_oranges))) = 73 := by
  sorry

end NUMINAMATH_CALUDE_final_orange_count_l1391_139130


namespace NUMINAMATH_CALUDE_average_squares_first_11_even_numbers_l1391_139121

theorem average_squares_first_11_even_numbers :
  let first_11_even : List Nat := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
  let squares := first_11_even.map (λ x => x * x)
  let sum_squares := squares.sum
  let average := sum_squares / first_11_even.length
  average = 184 := by
sorry

end NUMINAMATH_CALUDE_average_squares_first_11_even_numbers_l1391_139121


namespace NUMINAMATH_CALUDE_power_function_above_identity_l1391_139131

theorem power_function_above_identity {α : ℝ} :
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1) → x^α > x) ↔ α < 1 :=
sorry

end NUMINAMATH_CALUDE_power_function_above_identity_l1391_139131


namespace NUMINAMATH_CALUDE_equal_values_l1391_139183

theorem equal_values (p q a b : ℝ) 
  (h1 : p + q = 1)
  (h2 : p * q ≠ 0)
  (h3 : (p / a) + (q / b) = 1 / (p * a + q * b)) :
  a = b := by sorry

end NUMINAMATH_CALUDE_equal_values_l1391_139183


namespace NUMINAMATH_CALUDE_average_speed_two_hours_l1391_139196

/-- The average speed of a car given its speeds in two consecutive hours -/
theorem average_speed_two_hours (speed1 speed2 : ℝ) : 
  speed1 = 95 → speed2 = 60 → (speed1 + speed2) / 2 = 77.5 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_two_hours_l1391_139196


namespace NUMINAMATH_CALUDE_set_operations_with_empty_l1391_139143

theorem set_operations_with_empty (A : Set α) : 
  (A ∩ ∅ = ∅) ∧ 
  (A ∪ ∅ = A) ∧ 
  ((A ∩ ∅ = ∅) ∧ (A ∪ ∅ = A)) ∧ 
  ((A ∩ ∅ = ∅) ∨ (A ∪ ∅ = A)) ∧ 
  ¬¬(A ∩ ∅ = ∅) ∧ 
  ¬¬(A ∪ ∅ = A) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_with_empty_l1391_139143


namespace NUMINAMATH_CALUDE_units_digit_sum_product_l1391_139179

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_product : units_digit ((13 * 41) + (27 * 34)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_product_l1391_139179


namespace NUMINAMATH_CALUDE_min_value_theorem_l1391_139126

theorem min_value_theorem (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_eq_one : x + y + z + w = 1) :
  (x + y + z) / (x * y * z * w) ≥ 144 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1391_139126


namespace NUMINAMATH_CALUDE_least_n_for_fraction_inequality_l1391_139150

theorem least_n_for_fraction_inequality : 
  (∃ n : ℕ, n > 0 ∧ (1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ 
  (∀ m : ℕ, m > 0 ∧ m < 4 → (1 : ℚ) / m - (1 : ℚ) / (m + 1) ≥ (1 : ℚ) / 15) ∧
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < (1 : ℚ) / 15) :=
by sorry

end NUMINAMATH_CALUDE_least_n_for_fraction_inequality_l1391_139150


namespace NUMINAMATH_CALUDE_democrat_ratio_l1391_139170

/-- Prove that the ratio of democrats to total participants is 1:3 -/
theorem democrat_ratio (total : ℕ) (female_democrats : ℕ) :
  total = 780 →
  female_democrats = 130 →
  (∃ (female male : ℕ),
    female + male = total ∧
    female = 2 * female_democrats ∧
    4 * (female_democrats + male / 4) = total / 3) :=
by sorry

end NUMINAMATH_CALUDE_democrat_ratio_l1391_139170


namespace NUMINAMATH_CALUDE_orange_picking_theorem_l1391_139171

/-- The total number of oranges picked over three days -/
def totalOranges (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) : ℕ :=
  day1 + day2 + day3

/-- Theorem stating the total number of oranges picked -/
theorem orange_picking_theorem :
  let day1 := 100
  let day2 := 3 * day1
  let day3 := 70
  totalOranges day1 day2 day3 = 470 := by
  sorry

end NUMINAMATH_CALUDE_orange_picking_theorem_l1391_139171


namespace NUMINAMATH_CALUDE_count_negative_numbers_l1391_139141

def numbers : List ℝ := [-2.5, 7, -3, 2, 0, 4, 5, -1]

theorem count_negative_numbers : 
  (numbers.filter (· < 0)).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_negative_numbers_l1391_139141


namespace NUMINAMATH_CALUDE_line_circle_relationship_l1391_139106

theorem line_circle_relationship (m : ℝ) (h : m > 0) :
  let line := {(x, y) : ℝ × ℝ | Real.sqrt 2 * (x + y) + 1 + m = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = m}
  (∃ p, p ∈ line ∩ circle ∧ (∀ q ∈ line ∩ circle, q = p)) ∨
  (line ∩ circle = ∅) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_relationship_l1391_139106


namespace NUMINAMATH_CALUDE_valid_arrangements_l1391_139151

/- Define the number of students and schools -/
def total_students : ℕ := 4
def num_schools : ℕ := 2
def students_per_school : ℕ := 2

/- Define a function to calculate the number of arrangements -/
def num_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  if n = total_students ∧ k = num_schools ∧ m = students_per_school ∧ n = k * m then
    2 * (Nat.factorial m) * (Nat.factorial m)
  else
    0

/- Theorem statement -/
theorem valid_arrangements :
  num_arrangements total_students num_schools students_per_school = 8 :=
by sorry

end NUMINAMATH_CALUDE_valid_arrangements_l1391_139151
