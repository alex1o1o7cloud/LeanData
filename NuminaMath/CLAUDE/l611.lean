import Mathlib

namespace conic_eccentricity_l611_61165

/-- Given that 4, m, 1 form a geometric sequence, 
    the eccentricity of x²/m + y² = 1 is √2/2 or √3 -/
theorem conic_eccentricity (m : ℝ) : 
  (4 * 1 = m^2) →  -- Geometric sequence condition
  (∃ (e : ℝ), (e = Real.sqrt 2 / 2 ∨ e = Real.sqrt 3) ∧
   ∀ (x y : ℝ), x^2 / m + y^2 = 1 → 
   (∃ (a b : ℝ), 
     (m > 0 → x^2 / a^2 + y^2 / b^2 = 1 ∧ e = Real.sqrt (1 - b^2 / a^2)) ∧
     (m < 0 → y^2 / a^2 - x^2 / b^2 = 1 ∧ e = Real.sqrt (1 + a^2 / b^2)))) :=
by sorry

end conic_eccentricity_l611_61165


namespace prop_a_false_prop_b_false_prop_c_true_prop_d_false_false_propositions_l611_61169

-- Proposition A
theorem prop_a_false : ¬(∀ x : ℝ, x^2 + 3 < 0) := by sorry

-- Proposition B
theorem prop_b_false : ¬(∀ x : ℕ, x^2 > 1) := by sorry

-- Proposition C
theorem prop_c_true : ∃ x : ℤ, x^5 < 1 := by sorry

-- Proposition D
theorem prop_d_false : ¬(∃ x : ℚ, x^2 = 3) := by sorry

-- Combined theorem
theorem false_propositions :
  (¬(∀ x : ℝ, x^2 + 3 < 0)) ∧
  (¬(∀ x : ℕ, x^2 > 1)) ∧
  (∃ x : ℤ, x^5 < 1) ∧
  (¬(∃ x : ℚ, x^2 = 3)) := by sorry

end prop_a_false_prop_b_false_prop_c_true_prop_d_false_false_propositions_l611_61169


namespace hyperbola_eccentricity_l611_61159

/-- The eccentricity of a hyperbola with equation x²/2 - y² = 1 is √6/2 -/
theorem hyperbola_eccentricity : 
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/2 - y^2 = 1}
  ∃ e : ℝ, e = (Real.sqrt 6) / 2 ∧ 
    ∀ (a b c : ℝ), 
      (a^2 = 2 ∧ b^2 = 1 ∧ c^2 = a^2 + b^2) → 
      e = c / a :=
by sorry

end hyperbola_eccentricity_l611_61159


namespace added_amount_proof_l611_61171

theorem added_amount_proof (n x : ℝ) : n = 20 → (1/2) * n + x = 15 → x = 5 := by
  sorry

end added_amount_proof_l611_61171


namespace stamp_ratio_problem_l611_61194

theorem stamp_ratio_problem (x : ℕ) 
  (h1 : x > 0)
  (h2 : 7 * x - 8 = (4 * x + 8) + 8) :
  (7 * x - 8) / (4 * x + 8) = 6 / 5 := by
  sorry

end stamp_ratio_problem_l611_61194


namespace solution_set_has_three_elements_l611_61162

/-- A pair of positive integers representing the sides of a rectangle. -/
structure RectangleSides where
  a : ℕ+
  b : ℕ+

/-- The condition that the perimeter of a rectangle equals its area. -/
def perimeterEqualsArea (sides : RectangleSides) : Prop :=
  2 * (sides.a.val + sides.b.val) = sides.a.val * sides.b.val

/-- The set of all rectangle sides satisfying the perimeter-area equality. -/
def solutionSet : Set RectangleSides :=
  {sides | perimeterEqualsArea sides}

/-- The theorem stating that the solution set contains exactly three elements. -/
theorem solution_set_has_three_elements :
    solutionSet = {⟨3, 6⟩, ⟨6, 3⟩, ⟨4, 4⟩} := by sorry

end solution_set_has_three_elements_l611_61162


namespace max_value_cos_sin_l611_61136

theorem max_value_cos_sin (θ : Real) (h : -π/2 < θ ∧ θ < π/2) :
  ∃ (M : Real), M = Real.sqrt 2 ∧ 
  ∀ θ', -π/2 < θ' ∧ θ' < π/2 → 
    Real.cos (θ'/2) * (1 + Real.sin θ') ≤ M :=
by sorry

end max_value_cos_sin_l611_61136


namespace system_solution_l611_61130

theorem system_solution :
  ∃ (x y : ℚ), 
    (7 * x - 50 * y = 2) ∧ 
    (3 * y - x = 4) ∧ 
    (x = -206/29) ∧ 
    (y = -30/29) :=
by sorry

end system_solution_l611_61130


namespace solution_set_part_i_range_of_a_l611_61183

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + a*|x - 1|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x ≥ 4} = {x : ℝ | x ≤ -2/3 ∨ x ≥ 2} :=
by sorry

-- Part II
theorem range_of_a :
  ∀ a : ℝ, (∀ x ∈ Set.Icc (1/2 : ℝ) 2, f a x ≥ |x - 2|) → a ≥ 3 :=
by sorry

end solution_set_part_i_range_of_a_l611_61183


namespace tan_sum_45_deg_l611_61189

theorem tan_sum_45_deg (A B : Real) (h : A + B = Real.pi / 4) :
  (1 + Real.tan A) * (1 + Real.tan B) = 2 := by
sorry

end tan_sum_45_deg_l611_61189


namespace factor_quadratic_l611_61107

theorem factor_quadratic (x : ℝ) : 3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 := by
  sorry

end factor_quadratic_l611_61107


namespace wrong_height_calculation_l611_61166

theorem wrong_height_calculation (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (correct_avg : ℝ) 
  (h1 : n = 35)
  (h2 : initial_avg = 185)
  (h3 : actual_height = 106)
  (h4 : correct_avg = 183) :
  ∃ wrong_height : ℝ, 
    wrong_height = n * initial_avg - (n * correct_avg - actual_height) := by
  sorry

end wrong_height_calculation_l611_61166


namespace parametric_to_standard_hyperbola_l611_61143

theorem parametric_to_standard_hyperbola 
  (a b t x y : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (ht : t ≠ 0) :
  x = (a / 2) * (t + 1 / t) ∧ y = (b / 2) * (t - 1 / t) → 
  x^2 / a^2 - y^2 / b^2 = 1 := by
  sorry

end parametric_to_standard_hyperbola_l611_61143


namespace unique_square_solution_l611_61100

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

def digits_match (abc adeff : ℕ) : Prop :=
  let abc_digits := [abc / 100, (abc / 10) % 10, abc % 10]
  let adeff_digits := [adeff / 10000, (adeff / 1000) % 10, (adeff / 100) % 10, (adeff / 10) % 10, adeff % 10]
  (abc_digits.head? = adeff_digits.head?) ∧
  (abc_digits.get? 2 = adeff_digits.get? 3) ∧
  (abc_digits.get? 2 = adeff_digits.get? 4)

theorem unique_square_solution :
  ∀ abc adeff : ℕ,
    is_three_digit abc →
    is_five_digit adeff →
    abc ^ 2 = adeff →
    digits_match abc adeff →
    abc = 138 ∧ adeff = 19044 := by
  sorry

end unique_square_solution_l611_61100


namespace sum_of_three_numbers_l611_61187

theorem sum_of_three_numbers (a b c : ℝ) : 
  a + b = 35 ∧ b + c = 50 ∧ c + a = 60 → a + b + c = 72.5 := by
  sorry

end sum_of_three_numbers_l611_61187


namespace hank_bake_sale_earnings_l611_61135

/-- Prove that Hank made $80 in the bake sale given the conditions of his fundraising activities. -/
theorem hank_bake_sale_earnings :
  let carwash_earnings : ℚ := 100
  let carwash_donation_rate : ℚ := 90 / 100
  let bake_sale_donation_rate : ℚ := 75 / 100
  let lawn_mowing_earnings : ℚ := 50
  let lawn_mowing_donation_rate : ℚ := 1
  let total_donation : ℚ := 200
  ∃ bake_sale_earnings : ℚ,
    bake_sale_earnings * bake_sale_donation_rate +
    carwash_earnings * carwash_donation_rate +
    lawn_mowing_earnings * lawn_mowing_donation_rate = total_donation ∧
    bake_sale_earnings = 80 :=
by sorry

end hank_bake_sale_earnings_l611_61135


namespace ad_arrangement_count_l611_61120

def num_commercial_ads : ℕ := 4
def num_public_service_ads : ℕ := 2
def total_ads : ℕ := 6

theorem ad_arrangement_count :
  (num_commercial_ads.factorial) * (num_public_service_ads.factorial) = 48 :=
sorry

end ad_arrangement_count_l611_61120


namespace inequality_solution_l611_61175

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 4 / (x + 4) ≥ 1) ↔ 
  (x ∈ Set.Ioc (-4) (-2) ∪ Set.Ioc (-2) (Real.sqrt 8)) :=
by sorry

end inequality_solution_l611_61175


namespace only_D_in_second_quadrant_l611_61147

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

def point_A : ℝ × ℝ := (2, 3)
def point_B : ℝ × ℝ := (2, -3)
def point_C : ℝ × ℝ := (-2, -3)
def point_D : ℝ × ℝ := (-2, 3)

theorem only_D_in_second_quadrant :
  ¬(second_quadrant point_A.1 point_A.2) ∧
  ¬(second_quadrant point_B.1 point_B.2) ∧
  ¬(second_quadrant point_C.1 point_C.2) ∧
  second_quadrant point_D.1 point_D.2 := by sorry

end only_D_in_second_quadrant_l611_61147


namespace towel_area_decrease_l611_61197

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let new_length := 0.7 * L
  let new_breadth := 0.85 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.405 := by
sorry

end towel_area_decrease_l611_61197


namespace cos_diff_symmetric_angles_l611_61190

/-- Two angles are symmetric with respect to the origin if their difference is an odd multiple of π -/
def symmetric_angles (α β : Real) : Prop :=
  ∃ k : Int, β = α + (2 * k - 1) * Real.pi

/-- 
If the terminal sides of angles α and β are symmetric with respect to the origin O,
then cos(α - β) = -1
-/
theorem cos_diff_symmetric_angles (α β : Real) 
  (h : symmetric_angles α β) : Real.cos (α - β) = -1 := by
  sorry

end cos_diff_symmetric_angles_l611_61190


namespace reverse_digit_integers_l611_61164

theorem reverse_digit_integers (q r : ℕ) : 
  (q ≥ 10 ∧ q < 100) →  -- q is a two-digit number
  (r ≥ 10 ∧ r < 100) →  -- r is a two-digit number
  (∃ (a b : ℕ), q = 10 * a + b ∧ r = 10 * b + a) →  -- q and r have reversed digits
  (q > r → q - r < 30) →  -- positive difference less than 30
  (∀ (q' r' : ℕ), (q' ≥ 10 ∧ q' < 100) → (r' ≥ 10 ∧ r' < 100) → 
    (∃ (a' b' : ℕ), q' = 10 * a' + b' ∧ r' = 10 * b' + a') → 
    (q' > r' → q' - r' ≤ q - r)) →  -- q - r is the greatest possible difference
  (q - r = 27) →  -- greatest difference is 27
  (∃ (a b : ℕ), q = 10 * a + b ∧ r = 10 * b + a ∧ a - b = 3 ∧ a = 9 ∧ b = 6) :=
by sorry

end reverse_digit_integers_l611_61164


namespace sum_n_value_l611_61117

/-- An arithmetic sequence {a_n} satisfying given conditions -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  condition1 : a 3 * a 7 = -16
  condition2 : a 4 + a 6 = 0

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n : ℤ) * (seq.a 1 + seq.a n) / 2

/-- Theorem stating the possible values for the sum of the first n terms -/
theorem sum_n_value (seq : ArithmeticSequence) (n : ℕ) :
  sum_n seq n = n * (n - 9) ∨ sum_n seq n = -n * (n - 9) := by
  sorry


end sum_n_value_l611_61117


namespace michelle_gas_usage_l611_61132

theorem michelle_gas_usage (start_gas end_gas : Real) 
  (h1 : start_gas = 0.5)
  (h2 : end_gas = 0.16666666666666666) :
  start_gas - end_gas = 0.33333333333333334 := by
  sorry

end michelle_gas_usage_l611_61132


namespace like_terms_exponent_sum_l611_61193

theorem like_terms_exponent_sum (m n : ℕ) : 
  (∃ (x y : ℝ), 3 * x^(2*m) * y^3 = -2 * x^2 * y^n) → m + n = 4 := by
  sorry

end like_terms_exponent_sum_l611_61193


namespace expression_not_simplifiable_l611_61103

theorem expression_not_simplifiable (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h : a + 2*b + 2*c = 0) : 
  ∃ (f : ℝ → ℝ → ℝ → ℝ), f a b c = 
    (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) ∧
    ∀ (g : ℝ → ℝ), (∀ x y z, f x y z = g (f x y z)) → g = id := by
  sorry

end expression_not_simplifiable_l611_61103


namespace range_of_b_minus_a_l611_61126

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem range_of_b_minus_a (a b : ℝ) :
  (∀ x ∈ Set.Icc a b, -1 ≤ f x ∧ f x ≤ 3) →
  (∃ x ∈ Set.Icc a b, f x = -1) →
  (∃ x ∈ Set.Icc a b, f x = 3) →
  2 ≤ b - a ∧ b - a ≤ 4 :=
sorry

end range_of_b_minus_a_l611_61126


namespace unique_valid_swap_l611_61137

/-- Represents a time between 6 and 7 o'clock -/
structure Time6To7 where
  hour : ℝ
  minute : ℝ
  h_range : 6 < hour ∧ hour < 7
  m_range : 0 ≤ minute ∧ minute < 60

/-- Checks if swapping hour and minute hands results in a valid time -/
def is_valid_swap (t : Time6To7) : Prop :=
  ∃ (t' : Time6To7), t.hour = t'.minute / 5 ∧ t.minute = t'.hour * 5

/-- The main theorem stating there's exactly one time where swapping hands is valid -/
theorem unique_valid_swap : ∃! (t : Time6To7), is_valid_swap t :=
sorry

end unique_valid_swap_l611_61137


namespace remainder_theorem_l611_61145

theorem remainder_theorem (r : ℤ) : (r^11 - 3) % (r - 2) = 2045 := by
  sorry

end remainder_theorem_l611_61145


namespace percentage_relationship_l611_61198

theorem percentage_relationship (x y : ℝ) (h : x = y * (1 - 0.4444444444444444)) :
  y = x * 1.8 := by
sorry

end percentage_relationship_l611_61198


namespace ferry_tourists_count_l611_61144

/-- Calculates the total number of tourists transported by a ferry -/
def total_tourists (trips : ℕ) (initial_tourists : ℕ) (decrease : ℕ) : ℕ :=
  trips * (2 * initial_tourists - (trips - 1) * decrease) / 2

/-- Proves that the total number of tourists transported is 798 -/
theorem ferry_tourists_count :
  total_tourists 7 120 2 = 798 := by
  sorry

end ferry_tourists_count_l611_61144


namespace log_problem_l611_61148

-- Define the logarithm base 10
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_problem (x : ℝ) (h1 : x < 1) (h2 : (log10 x)^3 - log10 (x^3) = 125) :
  (log10 x)^4 - log10 (x^4) = 645 := by
  sorry

end log_problem_l611_61148


namespace tangent_slope_at_one_l611_61150

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the limit condition
variable (h : ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((f 1 - f (1 + 2*x)) / (2*x)) - 1| < ε)

-- State the theorem
theorem tangent_slope_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((f 1 - f (1 + 2*x)) / (2*x)) - 1| < ε) : 
  deriv f 1 = -1 := by
  sorry

end tangent_slope_at_one_l611_61150


namespace gcd_of_B_is_two_l611_61125

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_B_is_two : 
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 :=
sorry

end gcd_of_B_is_two_l611_61125


namespace base6_to_base10_fraction_l611_61156

/-- Converts a base-6 number to base-10 --/
def base6To10 (n : ℕ) : ℕ := sorry

/-- Determines if a natural number is a valid 3-digit base-10 number --/
def isValidBase10 (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- Extracts the hundreds digit from a 3-digit base-10 number --/
def hundredsDigit (n : ℕ) : ℕ := n / 100

/-- Extracts the ones digit from a 3-digit base-10 number --/
def onesDigit (n : ℕ) : ℕ := n % 10

theorem base6_to_base10_fraction (c d e : ℕ) :
  base6To10 532 = 100 * c + 10 * d + e →
  isValidBase10 (100 * c + 10 * d + e) →
  (c * e : ℚ) / 10 = 0 := by sorry

end base6_to_base10_fraction_l611_61156


namespace oplus_two_one_l611_61163

def oplus (x y : ℝ) : ℝ := x^3 - 3*x*y^2 + y^3

theorem oplus_two_one : oplus 2 1 = 3 := by
  sorry

end oplus_two_one_l611_61163


namespace largest_integer_m_l611_61114

theorem largest_integer_m (x y m : ℝ) : 
  x + 2*y = 2*m + 1 →
  2*x + y = m + 2 →
  x - y > 2 →
  ∀ k : ℤ, k > m → k ≤ -2 :=
by sorry

end largest_integer_m_l611_61114


namespace x_value_when_y_is_two_l611_61105

theorem x_value_when_y_is_two (x y : ℚ) : 
  y = 1 / (5 * x + 2) → y = 2 → x = -3/10 := by
sorry

end x_value_when_y_is_two_l611_61105


namespace linear_function_comparison_inverse_proportion_comparison_l611_61153

-- Linear function
theorem linear_function_comparison (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = -2 * x₁ + 1) 
  (h2 : y₂ = -2 * x₂ + 1) 
  (h3 : x₁ < x₂) : 
  y₁ > y₂ := by sorry

-- Inverse proportion function
theorem inverse_proportion_comparison (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : y₁ = 2 / x₁) 
  (h2 : y₂ = 2 / x₂) 
  (h3 : x₁ < x₂) 
  (h4 : x₂ < 0) : 
  y₁ > y₂ := by sorry

end linear_function_comparison_inverse_proportion_comparison_l611_61153


namespace kyle_money_after_snowboarding_l611_61161

theorem kyle_money_after_snowboarding (dave_money : ℕ) (kyle_initial_money : ℕ) 
  (h1 : dave_money = 46) 
  (h2 : kyle_initial_money = 3 * dave_money - 12) 
  (h3 : kyle_initial_money ≥ 12) : 
  kyle_initial_money - (kyle_initial_money / 3) = 84 := by
  sorry

end kyle_money_after_snowboarding_l611_61161


namespace initial_boys_on_slide_l611_61179

theorem initial_boys_on_slide (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  additional = 13 → total = 35 → initial + additional = total → initial = 22 := by
  sorry

end initial_boys_on_slide_l611_61179


namespace determinant_of_roots_l611_61192

/-- Given a, b, c are roots of x^3 + px^2 + qx + r = 0, 
    the determinant of [[a, c, b], [c, b, a], [b, a, c]] is -c^3 + b^2c -/
theorem determinant_of_roots (p q r a b c : ℝ) : 
  a^3 + p*a^2 + q*a + r = 0 →
  b^3 + p*b^2 + q*b + r = 0 →
  c^3 + p*c^2 + q*c + r = 0 →
  Matrix.det !![a, c, b; c, b, a; b, a, c] = -c^3 + b^2*c := by
  sorry

end determinant_of_roots_l611_61192


namespace min_box_height_is_seven_l611_61176

/-- Represents the side length of the square base of the box -/
def base_side : ℝ → ℝ := λ x => x

/-- Represents the height of the box -/
def box_height : ℝ → ℝ := λ x => x + 5

/-- Calculates the surface area of the box -/
def surface_area : ℝ → ℝ := λ x => 2 * x^2 + 4 * x * (x + 5)

/-- Theorem stating that the minimum height of the box satisfying the conditions is 7 -/
theorem min_box_height_is_seven :
  ∃ x : ℝ, x > 0 ∧ 
    surface_area x ≥ 120 ∧
    box_height x = 7 ∧
    ∀ y : ℝ, y > 0 ∧ surface_area y ≥ 120 → box_height y ≥ box_height x :=
by
  sorry


end min_box_height_is_seven_l611_61176


namespace sum_of_repeating_decimals_l611_61149

-- Define the repeating decimals
def repeating_decimal_2 : ℚ := 2 / 9
def repeating_decimal_02 : ℚ := 2 / 99

-- Theorem statement
theorem sum_of_repeating_decimals :
  repeating_decimal_2 + repeating_decimal_02 = 8 / 33 := by
  sorry

end sum_of_repeating_decimals_l611_61149


namespace subway_speed_comparison_l611_61138

-- Define the speed function
def speed (s : ℝ) : ℝ := s^2 + 2*s

-- Define the theorem
theorem subway_speed_comparison :
  ∃! t : ℝ, 0 ≤ t ∧ t ≤ 7 ∧ speed 5 = speed t + 20 ∧ t = 3 := by
  sorry

end subway_speed_comparison_l611_61138


namespace dice_sum_pigeonhole_l611_61123

/-- Represents a fair six-sided die -/
def Die := Fin 6

/-- Represents the sum of four dice rolls -/
def DiceSum := Fin 21

/-- The minimum number of throws required to guarantee a repeated sum -/
def minThrows : Nat := 22

theorem dice_sum_pigeonhole :
  ∀ (rolls : Fin minThrows → DiceSum),
  ∃ (i j : Fin minThrows), i ≠ j ∧ rolls i = rolls j :=
sorry

end dice_sum_pigeonhole_l611_61123


namespace number_problem_l611_61101

theorem number_problem : ∃ x : ℚ, (35 / 100) * x = (40 / 100) * 50 ∧ x = 400 / 7 := by
  sorry

end number_problem_l611_61101


namespace a_sum_cube_minus_product_l611_61160

noncomputable def a (i : ℕ) (x : ℝ) : ℝ := ∑' n, (x ^ (3 * n + i)) / (Nat.factorial (3 * n + i))

theorem a_sum_cube_minus_product (x : ℝ) :
  (a 0 x) ^ 3 + (a 1 x) ^ 3 + (a 2 x) ^ 3 - 3 * (a 0 x) * (a 1 x) * (a 2 x) = 1 := by
  sorry

end a_sum_cube_minus_product_l611_61160


namespace marias_apple_sales_l611_61168

/-- Given Maria's apple sales, prove the amount sold in the second hour -/
theorem marias_apple_sales (first_hour_sales second_hour_sales : ℝ) 
  (h1 : first_hour_sales = 10)
  (h2 : (first_hour_sales + second_hour_sales) / 2 = 6) : 
  second_hour_sales = 2 := by
  sorry

end marias_apple_sales_l611_61168


namespace min_lateral_perimeter_is_six_l611_61124

/-- Represents a rectangular parallelepiped with a square base -/
structure Parallelepiped where
  base_side : ℝ
  height : ℝ

/-- The volume of a parallelepiped -/
def volume (p : Parallelepiped) : ℝ :=
  p.base_side^2 * p.height

/-- The perimeter of a lateral face of a parallelepiped -/
def lateral_perimeter (p : Parallelepiped) : ℝ :=
  2 * p.base_side + 2 * p.height

/-- Theorem: The minimum perimeter of a lateral face among all rectangular
    parallelepipeds with volume 4 and square bases is 6 -/
theorem min_lateral_perimeter_is_six :
  ∀ p : Parallelepiped, volume p = 4 → lateral_perimeter p ≥ 6 :=
by sorry

end min_lateral_perimeter_is_six_l611_61124


namespace diamond_equation_solution_l611_61185

def diamond (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 2

theorem diamond_equation_solution :
  ∀ X : ℝ, diamond X 6 = 35 → X = 51 / 4 := by
  sorry

end diamond_equation_solution_l611_61185


namespace trucks_left_l611_61118

-- Define the initial number of trucks Sarah had
def initial_trucks : ℕ := 51

-- Define the number of trucks Sarah gave away
def trucks_given_away : ℕ := 13

-- Theorem to prove
theorem trucks_left : initial_trucks - trucks_given_away = 38 := by
  sorry

end trucks_left_l611_61118


namespace range_of_a_l611_61109

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x - 4 < 0) ↔ -4 < a ∧ a ≤ 0 :=
sorry

end range_of_a_l611_61109


namespace five_mondays_in_march_after_five_sunday_february_l611_61113

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ
  isLeapYear : Bool

/-- Represents a month in a specific year -/
structure Month where
  year : Year
  monthNumber : ℕ
  days : ℕ
  firstDay : DayOfWeek

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to count occurrences of a specific day in a month -/
def countDayInMonth (m : Month) (d : DayOfWeek) : ℕ := sorry

theorem five_mondays_in_march_after_five_sunday_february 
  (y : Year) 
  (feb : Month) 
  (mar : Month) :
  y.isLeapYear = true →
  feb.year = y →
  feb.monthNumber = 2 →
  feb.days = 29 →
  mar.year = y →
  mar.monthNumber = 3 →
  mar.days = 31 →
  countDayInMonth feb DayOfWeek.Sunday = 5 →
  mar.firstDay = nextDay feb.firstDay →
  countDayInMonth mar DayOfWeek.Monday = 5 := by
  sorry


end five_mondays_in_march_after_five_sunday_february_l611_61113


namespace print_shop_Y_charge_l611_61174

/-- The charge per color copy at print shop X -/
def charge_X : ℚ := 1.25

/-- The number of copies being compared -/
def num_copies : ℕ := 80

/-- The additional charge at print shop Y for the given number of copies -/
def additional_charge : ℚ := 120

/-- The charge per color copy at print shop Y -/
def charge_Y : ℚ := (charge_X * num_copies + additional_charge) / num_copies

theorem print_shop_Y_charge : charge_Y = 2.75 := by
  sorry

end print_shop_Y_charge_l611_61174


namespace product_one_plus_minus_sqrt_three_l611_61154

theorem product_one_plus_minus_sqrt_three : (1 + Real.sqrt 3) * (1 - Real.sqrt 3) = -2 := by
  sorry

end product_one_plus_minus_sqrt_three_l611_61154


namespace max_segment_length_l611_61199

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4
def line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 2 = 0

-- Define the condition PB ≥ 2PA
def condition (x y : ℝ) : Prop :=
  ((x - 4)^2 + y^2 - 4) ≥ 4 * (x^2 + y^2 - 1)

-- Theorem statement
theorem max_segment_length :
  ∃ (E F : ℝ × ℝ),
    line E.1 E.2 ∧ line F.1 F.2 ∧
    (∀ (P : ℝ × ℝ), line P.1 P.2 →
      (E.1 ≤ P.1 ∧ P.1 ≤ F.1) → condition P.1 P.2) ∧
    Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = 2 * Real.sqrt 39 / 3 :=
sorry

end max_segment_length_l611_61199


namespace parallel_lines_equal_angles_plane_l611_61141

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the relation for a line forming equal angles with a plane
variable (forms_equal_angles : Line → Plane → Prop)

-- Theorem statement
theorem parallel_lines_equal_angles_plane (a b : Line) (M : Plane) :
  (parallel a b → forms_equal_angles b M) ∧
  ¬(forms_equal_angles b M → parallel a b) :=
sorry

end parallel_lines_equal_angles_plane_l611_61141


namespace total_people_count_l611_61106

theorem total_people_count (people_in_front : ℕ) (people_behind : ℕ) (total_lines : ℕ) :
  people_in_front = 2 →
  people_behind = 4 →
  total_lines = 8 →
  (people_in_front + 1 + people_behind) * total_lines = 56 :=
by sorry

end total_people_count_l611_61106


namespace yellow_two_days_ago_count_l611_61116

/-- Represents the count of dandelions for a specific day -/
structure DandelionCount where
  yellow : ℕ
  white : ℕ

/-- Represents the dandelion lifecycle and counts for three consecutive days -/
structure DandelionMeadow where
  twoDaysAgo : DandelionCount
  yesterday : DandelionCount
  today : DandelionCount

/-- Theorem stating the relationship between yellow dandelions two days ago and white dandelions on subsequent days -/
theorem yellow_two_days_ago_count (meadow : DandelionMeadow) 
  (h1 : meadow.yesterday.yellow = 20)
  (h2 : meadow.yesterday.white = 14)
  (h3 : meadow.today.yellow = 15)
  (h4 : meadow.today.white = 11) :
  meadow.twoDaysAgo.yellow = meadow.yesterday.white + meadow.today.white :=
sorry

end yellow_two_days_ago_count_l611_61116


namespace horner_method_v2_equals_6_l611_61122

def f (x : ℝ) : ℝ := 1 + 2*x + x^2 - 3*x^3 + 2*x^4

def horner_v2 (a₀ a₁ a₂ a₃ a₄ x : ℝ) : ℝ :=
  let v₁ := a₄ * x + a₃
  v₁ * x + a₂

theorem horner_method_v2_equals_6 :
  horner_v2 1 2 1 (-3) 2 (-1) = 6 := by
  sorry

end horner_method_v2_equals_6_l611_61122


namespace linear_equation_solution_l611_61131

theorem linear_equation_solution (x y m : ℝ) 
  (hx : x = -1)
  (hy : y = 2)
  (hm : 5 * x + 3 * y = m) : 
  m = 1 := by
sorry

end linear_equation_solution_l611_61131


namespace difference_of_squares_l611_61127

theorem difference_of_squares (a : ℝ) : a^2 - 9 = (a + 3) * (a - 3) := by
  sorry

end difference_of_squares_l611_61127


namespace division_and_addition_of_fractions_l611_61155

theorem division_and_addition_of_fractions : 
  (2 : ℚ) / 3 / ((4 : ℚ) / 5) + (1 : ℚ) / 2 = (4 : ℚ) / 3 := by
  sorry

end division_and_addition_of_fractions_l611_61155


namespace flower_baskets_count_l611_61139

/-- The number of baskets used to hold flowers --/
def num_baskets (initial_flowers_per_daughter : ℕ) (additional_flowers : ℕ) (dead_flowers : ℕ) (flowers_per_basket : ℕ) : ℕ :=
  ((2 * initial_flowers_per_daughter + additional_flowers - dead_flowers) / flowers_per_basket)

/-- Theorem stating the number of baskets in the given scenario --/
theorem flower_baskets_count : num_baskets 5 20 10 4 = 5 := by
  sorry

end flower_baskets_count_l611_61139


namespace angle_D_measure_l611_61111

/-- Prove that given the specified angle conditions, angle D measures 25 degrees. -/
theorem angle_D_measure (A B C D : ℝ) : 
  A + B = 180 →
  C = D →
  A = 50 →
  D = 25 := by
  sorry

end angle_D_measure_l611_61111


namespace watermelon_sharing_l611_61128

/-- The number of people that can share one watermelon -/
def people_per_watermelon : ℕ := 8

/-- The number of watermelons available -/
def num_watermelons : ℕ := 4

/-- The total number of people that can share the watermelons -/
def total_people : ℕ := people_per_watermelon * num_watermelons

theorem watermelon_sharing :
  total_people = 32 :=
by sorry

end watermelon_sharing_l611_61128


namespace room_tiles_proof_l611_61181

/-- Calculates the least number of square tiles required to pave a rectangular floor -/
def leastSquareTiles (length width : ℕ) : ℕ :=
  let gcd := Nat.gcd length width
  (length * width) / (gcd * gcd)

theorem room_tiles_proof (length width : ℕ) 
  (h_length : length = 5000)
  (h_width : width = 1125) :
  leastSquareTiles length width = 360 := by
  sorry

#eval leastSquareTiles 5000 1125

end room_tiles_proof_l611_61181


namespace average_speed_is_69_l611_61180

def speeds : List ℝ := [90, 30, 60, 120, 45]
def total_time : ℝ := 5

theorem average_speed_is_69 :
  (speeds.sum / total_time) = 69 := by sorry

end average_speed_is_69_l611_61180


namespace integer_solution_exists_l611_61129

theorem integer_solution_exists : ∃ (x₁ x₂ y₁ y₂ y₃ y₄ : ℤ),
  (x₁ + x₂ = y₁ + y₂ + y₃ + y₄) ∧
  (x₁^2 + x₂^2 = y₁^2 + y₂^2 + y₃^2 + y₄^2) ∧
  (x₁^3 + x₂^3 = y₁^3 + y₂^3 + y₃^3 + y₄^3) ∧
  (abs x₁ > 2020) ∧ (abs x₂ > 2020) ∧
  (abs y₁ > 2020) ∧ (abs y₂ > 2020) ∧
  (abs y₃ > 2020) ∧ (abs y₄ > 2020) := by
  sorry

#print integer_solution_exists

end integer_solution_exists_l611_61129


namespace point_on_transformed_plane_l611_61157

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Applies a similarity transformation to a plane -/
def transformPlane (p : Plane) (k : ℝ) : Plane :=
  { a := p.a, b := p.b, c := p.c, d := k * p.d }

/-- Checks if a point lies on a plane -/
def pointOnPlane (point : Point3D) (plane : Plane) : Prop :=
  plane.a * point.x + plane.b * point.y + plane.c * point.z + plane.d = 0

/-- The main theorem to be proved -/
theorem point_on_transformed_plane :
  let A : Point3D := { x := 4, y := 3, z := 1 }
  let a : Plane := { a := 3, b := -4, c := 5, d := -6 }
  let k : ℝ := 5/6
  pointOnPlane A (transformPlane a k) := by
  sorry

end point_on_transformed_plane_l611_61157


namespace sqrt_equation_solution_l611_61170

theorem sqrt_equation_solution : 
  Real.sqrt (2 + Real.sqrt (3 + Real.sqrt (81/256))) = (2 + Real.sqrt (81/256)) ^ (1/4) := by
  sorry

end sqrt_equation_solution_l611_61170


namespace field_division_l611_61178

theorem field_division (total_area smaller_area larger_area : ℝ) : 
  total_area = 900 ∧
  smaller_area + larger_area = total_area ∧
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 405 := by
  sorry

end field_division_l611_61178


namespace odd_natural_not_divisible_by_square_l611_61102

theorem odd_natural_not_divisible_by_square (n : ℕ) : 
  Odd n → (¬(Nat.factorial (n - 1) % (n^2) = 0) ↔ Nat.Prime n ∨ n = 9) :=
by sorry

end odd_natural_not_divisible_by_square_l611_61102


namespace gcd_factorial_problem_l611_61115

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem gcd_factorial_problem : Nat.gcd (factorial 7) ((factorial 10) / (factorial 5)) = 2520 := by
  sorry

end gcd_factorial_problem_l611_61115


namespace divisor_totient_sum_bound_l611_61151

/-- d(n) represents the number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- φ(n) represents Euler's totient function -/
def φ (n : ℕ+) : ℕ := sorry

/-- Theorem stating that c must be less than or equal to 1 -/
theorem divisor_totient_sum_bound (n : ℕ+) (c : ℕ) (h : d n + φ n = n + c) : c ≤ 1 := by
  sorry

end divisor_totient_sum_bound_l611_61151


namespace figure3_turns_l611_61182

/-- Represents a dot in the grid --/
inductive Dot
| Black : Dot
| White : Dot

/-- Represents a turn in the loop --/
inductive Turn
| Right : Turn

/-- Represents a grid with dots --/
structure Grid :=
(dots : List Dot)

/-- Represents a loop in the grid --/
structure Loop :=
(turns : List Turn)

/-- Function to check if a loop is valid for a given grid --/
def is_valid_loop (g : Grid) (l : Loop) : Prop := sorry

/-- Function to count the number of turns in a loop --/
def count_turns (l : Loop) : Nat := l.turns.length

/-- The specific grid configuration for Figure 3 --/
def figure3 : Grid := sorry

/-- Theorem stating that the valid loop for Figure 3 has 20 turns --/
theorem figure3_turns :
  ∃ (l : Loop), is_valid_loop figure3 l ∧ count_turns l = 20 := by sorry

end figure3_turns_l611_61182


namespace inequality_proof_l611_61191

theorem inequality_proof (a : ℝ) (h : a ≠ 1) : (1 + a + a^2)^2 < 3*(1 + a^2 + a^4) := by
  sorry

end inequality_proof_l611_61191


namespace graph6_triangle_or_independent_set_l611_61110

/-- A simple graph with 6 vertices -/
structure Graph6 where
  vertices : Finset (Fin 6)
  edges : Set (Fin 6 × Fin 6)
  symmetry : ∀ (a b : Fin 6), (a, b) ∈ edges → (b, a) ∈ edges
  irreflexive : ∀ (a : Fin 6), (a, a) ∉ edges

/-- A triangle in a graph -/
def HasTriangle (G : Graph6) : Prop :=
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∈ G.edges ∧ (b, c) ∈ G.edges ∧ (c, a) ∈ G.edges

/-- An independent set of size 3 in a graph -/
def HasIndependentSet3 (G : Graph6) : Prop :=
  ∃ (a b c : Fin 6), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a, b) ∉ G.edges ∧ (b, c) ∉ G.edges ∧ (c, a) ∉ G.edges

/-- The main theorem -/
theorem graph6_triangle_or_independent_set (G : Graph6) :
  HasTriangle G ∨ HasIndependentSet3 G :=
sorry

end graph6_triangle_or_independent_set_l611_61110


namespace tenth_root_of_unity_l611_61142

theorem tenth_root_of_unity (n : ℕ) (h : n = 3) :
  (Complex.tan (π / 4) + Complex.I) / (Complex.tan (π / 4) - Complex.I) =
  Complex.exp (Complex.I * (2 * ↑n * π / 10)) :=
by sorry

end tenth_root_of_unity_l611_61142


namespace consecutive_even_integers_sum_l611_61108

theorem consecutive_even_integers_sum (n : ℤ) : 
  n % 2 = 0 ∧ n * (n + 2) * (n + 4) = 3360 → n + (n + 2) + (n + 4) = 48 := by
  sorry

end consecutive_even_integers_sum_l611_61108


namespace equal_chore_time_l611_61188

/-- The time in minutes it takes to sweep one room -/
def sweep_time : ℕ := 3

/-- The time in minutes it takes to wash one dish -/
def dish_time : ℕ := 2

/-- The time in minutes it takes to do one load of laundry -/
def laundry_time : ℕ := 9

/-- The number of rooms Anna sweeps -/
def anna_rooms : ℕ := 10

/-- The number of laundry loads Billy does -/
def billy_laundry : ℕ := 2

/-- The number of dishes Billy should wash -/
def billy_dishes : ℕ := 6

theorem equal_chore_time : 
  anna_rooms * sweep_time = billy_laundry * laundry_time + billy_dishes * dish_time := by
  sorry

end equal_chore_time_l611_61188


namespace parabola_line_intersection_l611_61173

theorem parabola_line_intersection (k : ℝ) : 
  (∃! x : ℝ, -2 = x^2 + k*x - 1) → (k = 2 ∨ k = -2) := by
  sorry

end parabola_line_intersection_l611_61173


namespace expression_evaluation_l611_61186

theorem expression_evaluation :
  (5^500 + 6^501)^2 - (5^500 - 6^501)^2 = 24 * 30^500 :=
by sorry

end expression_evaluation_l611_61186


namespace largest_number_l611_61112

theorem largest_number (a b c d e : ℝ) : 
  a = 15679 + 1/3579 → 
  b = 15679 - 1/3579 → 
  c = 15679 * (1/3579) → 
  d = 15679 / (1/3579) → 
  e = 15679 * 1.03 → 
  d > a ∧ d > b ∧ d > c ∧ d > e := by
sorry

end largest_number_l611_61112


namespace min_correct_answers_quiz_problem_l611_61119

/-- The minimum number of correctly answered questions to exceed 81 points in a quiz -/
theorem min_correct_answers (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℕ) (target_score : ℕ) : ℕ :=
  let min_correct := ((target_score + 1 + incorrect_points * total_questions) + (correct_points + incorrect_points) - 1) / (correct_points + incorrect_points)
  min_correct

/-- The specific quiz problem -/
theorem quiz_problem : min_correct_answers 22 4 2 81 = 21 := by
  sorry

end min_correct_answers_quiz_problem_l611_61119


namespace total_people_needed_l611_61133

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := 2 * people_per_car

/-- The number of cars to be lifted -/
def num_cars : ℕ := 6

/-- The number of trucks to be lifted -/
def num_trucks : ℕ := 3

/-- Theorem stating the total number of people needed to lift the given vehicles -/
theorem total_people_needed : 
  num_cars * people_per_car + num_trucks * people_per_truck = 60 := by
  sorry

end total_people_needed_l611_61133


namespace team_points_l611_61152

/-- Calculates the total points earned by a sports team based on their performance. -/
def total_points (wins losses ties : ℕ) : ℕ :=
  2 * wins + 0 * losses + 1 * ties

/-- Theorem stating that a team with 9 wins, 3 losses, and 4 ties earns 22 points. -/
theorem team_points : total_points 9 3 4 = 22 := by
  sorry

end team_points_l611_61152


namespace solution_set_inequality_range_of_a_range_of_m_l611_61167

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2

-- Statement 1
theorem solution_set_inequality (a : ℝ) :
  (∀ x, f a x ≤ 0 ↔ x ∈ Set.Icc 1 2) →
  (∀ x, f a x ≥ 1 - x^2 ↔ x ∈ Set.Iic (1/2) ∪ Set.Ici 1) :=
sorry

-- Statement 2
theorem range_of_a :
  (∀ a, (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 2*a*(x-1) + 4) →
    a ∈ Set.Iic (1/3)) :=
sorry

-- Statement 3
def g (m : ℝ) (x : ℝ) : ℝ := -x + m

theorem range_of_m :
  (∀ m, (∀ x₁ ∈ Set.Icc 1 4, ∃ x₂ ∈ Set.Ioo 1 8, f (-3) x₁ = g m x₂) →
    m ∈ Set.Ioo 7 (31/4)) :=
sorry

end solution_set_inequality_range_of_a_range_of_m_l611_61167


namespace total_units_is_34_l611_61196

/-- The number of apartment units in two identical buildings with specific floor configurations -/
def total_apartment_units : ℕ := by
  -- Define the number of buildings
  let num_buildings : ℕ := 2

  -- Define the number of floors in each building
  let num_floors : ℕ := 4

  -- Define the number of units on the first floor
  let units_first_floor : ℕ := 2

  -- Define the number of units on each of the other floors
  let units_other_floors : ℕ := 5

  -- Calculate the total number of units in one building
  let units_per_building : ℕ := units_first_floor + (num_floors - 1) * units_other_floors

  -- Calculate the total number of units in all buildings
  exact num_buildings * units_per_building

/-- Theorem stating that the total number of apartment units is 34 -/
theorem total_units_is_34 : total_apartment_units = 34 := by
  sorry

end total_units_is_34_l611_61196


namespace rectangle_area_is_200_l611_61184

/-- A rectangular region with three fenced sides and one wall -/
structure FencedRectangle where
  short_side : ℝ
  long_side : ℝ
  fence_length : ℝ
  wall_side : ℝ := long_side
  fenced_sides : ℝ := 2 * short_side + long_side
  area : ℝ := short_side * long_side

/-- The fenced rectangular region satisfying the problem conditions -/
def problem_rectangle : FencedRectangle where
  short_side := 10
  long_side := 20
  fence_length := 40

theorem rectangle_area_is_200 (r : FencedRectangle) :
  r.long_side = 2 * r.short_side →
  r.fence_length = 40 →
  r.area = 200 := by
  sorry

#check rectangle_area_is_200 problem_rectangle

end rectangle_area_is_200_l611_61184


namespace average_monthly_balance_l611_61140

def monthly_balances : List ℕ := [200, 250, 300, 350, 400]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℚ) = 300 := by sorry

end average_monthly_balance_l611_61140


namespace f_at_two_equals_three_l611_61177

def f (x : ℝ) : ℝ := 5 * x - 7

theorem f_at_two_equals_three : f 2 = 3 := by
  sorry

end f_at_two_equals_three_l611_61177


namespace consecutive_negative_integers_sum_l611_61158

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2720 → n + (n + 1) = -103 := by
  sorry

end consecutive_negative_integers_sum_l611_61158


namespace spring_work_l611_61146

/-- Work done to stretch a spring -/
theorem spring_work (force : Real) (compression : Real) (stretch : Real) : 
  force = 10 →
  compression = 0.1 →
  stretch = 0.06 →
  (1/2) * (force / compression) * stretch^2 = 0.18 := by
  sorry

end spring_work_l611_61146


namespace distance_between_squares_l611_61121

theorem distance_between_squares (small_perimeter : ℝ) (large_area : ℝ) : 
  small_perimeter = 8 →
  large_area = 25 →
  let small_side := small_perimeter / 4
  let large_side := Real.sqrt large_area
  let horizontal := small_side + large_side
  let vertical := large_side - small_side
  Real.sqrt (horizontal ^ 2 + vertical ^ 2) = Real.sqrt 58 := by
  sorry

#check distance_between_squares

end distance_between_squares_l611_61121


namespace cubic_equation_special_case_l611_61134

/-- Given a cubic equation and parameters, prove it's a special case of a model equation. -/
theorem cubic_equation_special_case 
  (x a b : ℝ) 
  (h_b_nonneg : b ≥ 0) :
  6.266 * x^3 - 3 * a * x^2 + (3 * a^2 - b) * x - (a^3 - a * b) = 0 ↔ 
  ∃ (v u w : ℝ), 
    v = a ∧ 
    u = a ∧ 
    w^2 = b ∧
    6.266 * x^3 - 3 * v * x^2 + (3 * u^2 - w^2) * x - (v^3 - v * w^2) = 0 :=
sorry

end cubic_equation_special_case_l611_61134


namespace probability_even_and_less_equal_three_l611_61195

def dice_sides : ℕ := 6

def prob_even_first_die : ℚ := 1 / 2

def prob_less_equal_three_second_die : ℚ := 1 / 2

theorem probability_even_and_less_equal_three (independence : True) :
  prob_even_first_die * prob_less_equal_three_second_die = 1 / 4 := by
  sorry

end probability_even_and_less_equal_three_l611_61195


namespace books_checked_out_thursday_l611_61172

theorem books_checked_out_thursday (initial_books : ℕ) (wednesday_checkout : ℕ) 
  (thursday_return : ℕ) (friday_return : ℕ) (final_books : ℕ) :
  initial_books = 98 →
  wednesday_checkout = 43 →
  thursday_return = 23 →
  friday_return = 7 →
  final_books = 80 →
  ∃ (thursday_checkout : ℕ),
    final_books = initial_books - wednesday_checkout + thursday_return - thursday_checkout + friday_return ∧
    thursday_checkout = 5 :=
by sorry

end books_checked_out_thursday_l611_61172


namespace consecutive_integers_sum_of_squares_l611_61104

theorem consecutive_integers_sum_of_squares (a : ℕ) (h1 : a > 1) 
  (h2 : (a - 1) * a * (a + 1) = 10 * (3 * a)) : 
  (a - 1)^2 + a^2 + (a + 1)^2 = 110 := by
  sorry

end consecutive_integers_sum_of_squares_l611_61104
