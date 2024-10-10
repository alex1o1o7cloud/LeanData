import Mathlib

namespace solve_for_y_l2942_294203

theorem solve_for_y (x y : ℝ) (h1 : x - y = 20) (h2 : x + y = 10) : y = -5 := by
  sorry

end solve_for_y_l2942_294203


namespace simplest_fraction_of_decimal_l2942_294220

theorem simplest_fraction_of_decimal (a b : ℕ+) (h : (a : ℚ) / b = 0.478125) :
  (∀ d : ℕ+, d ∣ a → d ∣ b → d = 1) →
  (a : ℕ) = 153 ∧ b = 320 ∧ a + b = 473 := by
  sorry

end simplest_fraction_of_decimal_l2942_294220


namespace max_sum_under_constraints_l2942_294251

theorem max_sum_under_constraints (a b : ℝ) :
  (4 * a + 3 * b ≤ 10) →
  (3 * a + 6 * b ≤ 12) →
  a + b ≤ 14 / 5 := by
sorry

end max_sum_under_constraints_l2942_294251


namespace simplify_power_l2942_294212

theorem simplify_power (y : ℝ) : (3 * y^2)^4 = 81 * y^8 := by
  sorry

end simplify_power_l2942_294212


namespace complex_magnitude_problem_l2942_294209

theorem complex_magnitude_problem :
  let z : ℂ := ((1 - 4*I) * (1 + I) + 2 + 4*I) / (3 + 4*I)
  Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l2942_294209


namespace first_group_size_l2942_294216

/-- The amount of work done by one person in one day -/
def work_per_person_per_day : ℝ := 1

/-- The number of days to complete the work -/
def days : ℕ := 7

/-- The number of persons in the second group -/
def persons_second_group : ℕ := 9

/-- The amount of work completed by the first group -/
def work_first_group : ℕ := 7

/-- The amount of work completed by the second group -/
def work_second_group : ℕ := 9

/-- The number of persons in the first group -/
def persons_first_group : ℕ := 9

theorem first_group_size :
  persons_first_group * days * work_per_person_per_day = work_first_group ∧
  persons_second_group * days * work_per_person_per_day = work_second_group →
  persons_first_group = 9 := by
sorry

end first_group_size_l2942_294216


namespace sum_of_altitudes_for_specific_line_l2942_294258

/-- The line equation ax + by = c forming a triangle with coordinate axes -/
structure TriangleLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the sum of altitudes of the triangle formed by the given line and coordinate axes -/
def sumOfAltitudes (line : TriangleLine) : ℝ :=
  sorry

/-- The specific line 8x + 3y = 48 -/
def specificLine : TriangleLine :=
  { a := 8, b := 3, c := 48 }

theorem sum_of_altitudes_for_specific_line :
  sumOfAltitudes specificLine = 370 / Real.sqrt 292 := by
  sorry

end sum_of_altitudes_for_specific_line_l2942_294258


namespace min_value_quadratic_l2942_294224

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + 2*x*y + 2*y^2 + 3*x - 5*y ≥ -17/2 ∧ 
  ∃ x y : ℝ, x^2 + 2*x*y + 2*y^2 + 3*x - 5*y = -17/2 :=
by sorry

end min_value_quadratic_l2942_294224


namespace friend_bicycles_count_friend_owns_ten_bicycles_l2942_294297

theorem friend_bicycles_count (ignatius_bicycles : ℕ) (tires_per_bicycle : ℕ) 
  (friend_unicycles : ℕ) (friend_tricycles : ℕ) : ℕ :=
  let ignatius_total_tires := ignatius_bicycles * tires_per_bicycle
  let friend_total_tires := 3 * ignatius_total_tires
  let friend_other_tires := friend_unicycles * 1 + friend_tricycles * 3
  let friend_bicycle_tires := friend_total_tires - friend_other_tires
  friend_bicycle_tires / tires_per_bicycle

theorem friend_owns_ten_bicycles :
  friend_bicycles_count 4 2 1 1 = 10 := by
  sorry

end friend_bicycles_count_friend_owns_ten_bicycles_l2942_294297


namespace intersection_M_N_l2942_294271

def M : Set ℝ := {-3, 1, 3}
def N : Set ℝ := {x | x^2 - 3*x - 4 < 0}

theorem intersection_M_N : M ∩ N = {1, 3} := by
  sorry

end intersection_M_N_l2942_294271


namespace sum_18_probability_l2942_294294

/-- A fair coin with sides labeled 5 and 15 -/
inductive Coin
| Five : Coin
| Fifteen : Coin

/-- A standard six-sided die -/
inductive Die
| One : Die
| Two : Die
| Three : Die
| Four : Die
| Five : Die
| Six : Die

/-- The probability of getting a sum of 18 when flipping the coin and rolling the die -/
def prob_sum_18 : ℚ :=
  1 / 12

/-- Theorem stating that the probability of getting a sum of 18 is 1/12 -/
theorem sum_18_probability : prob_sum_18 = 1 / 12 := by
  sorry

end sum_18_probability_l2942_294294


namespace arithmetic_expression_equality_l2942_294200

theorem arithmetic_expression_equality : 54 + (42 / 14) + (27 * 17) - 200 - (360 / 6) + 2^4 = 272 := by
  sorry

end arithmetic_expression_equality_l2942_294200


namespace correct_selection_count_l2942_294292

/-- The number of ways to select 4 students from 7, including both boys and girls -/
def select_students (total : ℕ) (boys : ℕ) (girls : ℕ) (to_select : ℕ) : ℕ :=
  Nat.choose total to_select - Nat.choose boys to_select

/-- Theorem stating the correct number of selections -/
theorem correct_selection_count :
  select_students 7 4 3 4 = 34 := by
  sorry

#eval select_students 7 4 3 4

end correct_selection_count_l2942_294292


namespace square_not_always_positive_l2942_294260

theorem square_not_always_positive : ∃ (a : ℝ), ¬(a^2 > 0) := by sorry

end square_not_always_positive_l2942_294260


namespace max_value_of_function_l2942_294259

theorem max_value_of_function (x : ℝ) : 
  (Real.sin x * (2 - Real.cos x)) / (5 - 4 * Real.cos x) ≤ Real.sqrt 3 / 4 := by
  sorry

end max_value_of_function_l2942_294259


namespace problem_statement_l2942_294239

theorem problem_statement (a b : ℝ) (h : a + b - 1 = 0) : 3 * a^2 + 6 * a * b + 3 * b^2 = 3 := by
  sorry

end problem_statement_l2942_294239


namespace perfect_square_factors_450_l2942_294245

/-- The number of perfect square factors of 450 -/
def num_perfect_square_factors : ℕ := 4

/-- The prime factorization of 450 -/
def factorization_450 : List (ℕ × ℕ) := [(2, 1), (3, 2), (5, 2)]

/-- Theorem stating that the number of perfect square factors of 450 is 4 -/
theorem perfect_square_factors_450 :
  (List.prod (List.map (fun (p : ℕ × ℕ) => p.1 ^ p.2) factorization_450) = 450) →
  (∀ (n : ℕ), n * n ∣ 450 ↔ n ∈ [1, 3, 5, 15]) →
  num_perfect_square_factors = 4 :=
by sorry

end perfect_square_factors_450_l2942_294245


namespace prime_product_square_l2942_294207

theorem prime_product_square (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p ≠ q → p ≠ r → q ≠ r →
  (p * q * r) % (p + q + r) = 0 →
  ∃ n : ℕ, (p - 1) * (q - 1) * (r - 1) + 1 = n ^ 2 :=
by sorry

end prime_product_square_l2942_294207


namespace vegetable_ghee_mixture_weight_l2942_294201

/-- Calculates the weight of a mixture of two brands of vegetable ghee -/
theorem vegetable_ghee_mixture_weight
  (weight_a : ℝ)
  (weight_b : ℝ)
  (ratio_a : ℝ)
  (ratio_b : ℝ)
  (total_volume : ℝ)
  (h_weight_a : weight_a = 800)
  (h_weight_b : weight_b = 850)
  (h_ratio_a : ratio_a = 3)
  (h_ratio_b : ratio_b = 2)
  (h_total_volume : total_volume = 3) :
  let volume_a := (ratio_a / (ratio_a + ratio_b)) * total_volume
  let volume_b := (ratio_b / (ratio_a + ratio_b)) * total_volume
  let total_weight := (weight_a * volume_a + weight_b * volume_b) / 1000
  total_weight = 2.46 := by
  sorry


end vegetable_ghee_mixture_weight_l2942_294201


namespace sufficient_condition_range_l2942_294252

theorem sufficient_condition_range (a : ℝ) : 
  (a > 0) →
  (∀ x : ℝ, (|x - 4| > 6 → x^2 - 2*x + 1 - a^2 > 0)) →
  (∃ x : ℝ, x^2 - 2*x + 1 - a^2 > 0 ∧ |x - 4| ≤ 6) →
  (0 < a ∧ a ≤ 3) :=
by sorry

end sufficient_condition_range_l2942_294252


namespace odd_function_property_y_value_at_3_l2942_294223

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x

theorem odd_function_property (a b c : ℝ) :
  ∀ x, f a b c (-x) = -(f a b c x) :=
sorry

theorem y_value_at_3 (a b c : ℝ) :
  f a b c (-3) - 5 = 7 → f a b c 3 - 5 = -17 :=
sorry

end odd_function_property_y_value_at_3_l2942_294223


namespace largest_power_of_two_dividing_5_256_minus_1_l2942_294267

theorem largest_power_of_two_dividing_5_256_minus_1 :
  (∃ n : ℕ, 2^n ∣ (5^256 - 1) ∧ ∀ m : ℕ, 2^m ∣ (5^256 - 1) → m ≤ n) →
  (∃ n : ℕ, 2^n ∣ (5^256 - 1) ∧ ∀ m : ℕ, 2^m ∣ (5^256 - 1) → m ≤ n) ∧
  (∀ n : ℕ, 2^n ∣ (5^256 - 1) ∧ ∀ m : ℕ, 2^m ∣ (5^256 - 1) → m ≤ n → n = 10) :=
by sorry

end largest_power_of_two_dividing_5_256_minus_1_l2942_294267


namespace last_two_digits_product_l2942_294204

/-- Given an integer n that is divisible by 6 and whose last two digits sum to 15,
    the product of its last two digits is 54. -/
theorem last_two_digits_product (n : ℤ) : 
  (n % 100 ≥ 0) →  -- Ensure we're dealing with the last two positive digits
  (n % 6 = 0) →    -- n is divisible by 6
  ((n % 100) / 10 + n % 10 = 15) →  -- Sum of last two digits is 15
  ((n % 100) / 10) * (n % 10) = 54 :=
by sorry

end last_two_digits_product_l2942_294204


namespace blank_expression_proof_l2942_294296

theorem blank_expression_proof (x y : ℝ) : 2 * x * (-3 * x^2 * y) = -6 * x^3 * y := by
  sorry

end blank_expression_proof_l2942_294296


namespace all_two_digit_numbers_appear_l2942_294286

/-- Represents a sequence of numbers from 1 to 1,000,000 in arbitrary order -/
def ArbitrarySequence := Fin 1000000 → Fin 1000000

/-- Represents a two-digit number (from 10 to 99) -/
def TwoDigitNumber := Fin 90

/-- A function that checks if a given two-digit number appears in the sequence when cut into two-digit pieces -/
def appearsInSequence (seq : ArbitrarySequence) (n : TwoDigitNumber) : Prop :=
  ∃ i : Fin 999999, (seq i).val / 100 % 100 = n.val + 10 ∨ (seq i).val % 100 = n.val + 10

/-- The main theorem statement -/
theorem all_two_digit_numbers_appear (seq : ArbitrarySequence) :
  ∀ n : TwoDigitNumber, appearsInSequence seq n :=
sorry

end all_two_digit_numbers_appear_l2942_294286


namespace analects_reasoning_is_common_sense_l2942_294219

/-- Represents the types of reasoning --/
inductive ReasoningType
  | CommonSense
  | Inductive
  | Analogical
  | Deductive

/-- Represents a step in the logical progression --/
structure LogicalStep where
  premise : String
  consequence : String

/-- Represents the characteristics of the reasoning in the Analects passage --/
structure AnalectsReasoning where
  steps : List LogicalStep
  alignsWithCommonSense : Bool
  followsLogicalProgression : Bool

/-- Determines the type of reasoning based on its characteristics --/
def determineReasoningType (reasoning : AnalectsReasoning) : ReasoningType :=
  if reasoning.alignsWithCommonSense && reasoning.followsLogicalProgression then
    ReasoningType.CommonSense
  else
    ReasoningType.Inductive -- Default to another type if conditions are not met

/-- The main theorem stating that the reasoning in the Analects passage is Common Sense reasoning --/
theorem analects_reasoning_is_common_sense (analectsReasoning : AnalectsReasoning) 
    (h1 : analectsReasoning.steps.length > 0)
    (h2 : analectsReasoning.alignsWithCommonSense = true)
    (h3 : analectsReasoning.followsLogicalProgression = true) :
  determineReasoningType analectsReasoning = ReasoningType.CommonSense := by
  sorry

#check analects_reasoning_is_common_sense

end analects_reasoning_is_common_sense_l2942_294219


namespace max_ratio_squared_max_ratio_squared_achieved_l2942_294257

theorem max_ratio_squared (a b x y : ℝ) : 
  0 < a → 0 < b → a ≥ b → 
  0 ≤ x → x < a → 0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b + y)^2 →
  (a / b)^2 ≤ 2 :=
by sorry

theorem max_ratio_squared_achieved (a b : ℝ) :
  ∃ x y : ℝ, 0 < a → 0 < b → a ≥ b → 
  0 ≤ x → x < a → 0 ≤ y → y < b →
  a^2 + y^2 = b^2 + x^2 ∧ b^2 + x^2 = (a - x)^2 + (b + y)^2 →
  (a / b)^2 = 2 :=
by sorry

end max_ratio_squared_max_ratio_squared_achieved_l2942_294257


namespace left_of_origin_abs_value_l2942_294299

theorem left_of_origin_abs_value (a : ℝ) : 
  (a < 0) → (|a| = 4.5) → (a = -4.5) := by sorry

end left_of_origin_abs_value_l2942_294299


namespace mass_of_apples_left_correct_l2942_294287

/-- Calculates the mass of apples left after sales -/
def mass_of_apples_left (kidney_apples : ℕ) (golden_apples : ℕ) (canada_apples : ℕ) (apples_sold : ℕ) : ℕ :=
  (kidney_apples + golden_apples + canada_apples) - apples_sold

/-- Proves that the mass of apples left is correct given the initial masses and the mass of apples sold -/
theorem mass_of_apples_left_correct 
  (kidney_apples : ℕ) (golden_apples : ℕ) (canada_apples : ℕ) (apples_sold : ℕ) :
  mass_of_apples_left kidney_apples golden_apples canada_apples apples_sold =
  (kidney_apples + golden_apples + canada_apples) - apples_sold :=
by
  sorry

/-- Verifies the specific case in the problem -/
example : mass_of_apples_left 23 37 14 36 = 38 :=
by
  sorry

end mass_of_apples_left_correct_l2942_294287


namespace vector_equality_implies_norm_equality_l2942_294235

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem vector_equality_implies_norm_equality 
  (a b : E) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a = -2 • b) : 
  ‖a‖ - ‖b‖ = ‖a + b‖ := by
  sorry

end vector_equality_implies_norm_equality_l2942_294235


namespace sum_of_coefficients_l2942_294280

theorem sum_of_coefficients (a : ℝ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℝ) :
  (∀ x, (x - a)^8 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8) →
  a₅ = 56 →
  a + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 2^8 := by
sorry

end sum_of_coefficients_l2942_294280


namespace grants_apartment_rooms_l2942_294265

-- Define the number of rooms in Danielle's apartment
def danielles_rooms : ℕ := 6

-- Define the number of rooms in Heidi's apartment
def heidis_rooms : ℕ := 3 * danielles_rooms

-- Define the number of rooms in Grant's apartment
def grants_rooms : ℕ := heidis_rooms / 9

-- Theorem stating that Grant's apartment has 2 rooms
theorem grants_apartment_rooms : grants_rooms = 2 := by
  sorry

end grants_apartment_rooms_l2942_294265


namespace stock_price_decrease_l2942_294214

theorem stock_price_decrease (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_2006 := initial_price * 1.3
  let price_after_2007 := price_after_2006 * 1.2
  let decrease_percentage := (price_after_2007 - initial_price) / price_after_2007 * 100
  ∃ ε > 0, abs (decrease_percentage - 35.9) < ε :=
by
  sorry

end stock_price_decrease_l2942_294214


namespace number_equation_l2942_294242

theorem number_equation (x : ℝ) : 2500 - (x / 20.04) = 2450 ↔ x = 1002 := by
  sorry

end number_equation_l2942_294242


namespace min_product_abc_l2942_294241

theorem min_product_abc (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  a + b + c = 1 → 
  a ≤ 3*b → a ≤ 3*c → b ≤ 3*a → b ≤ 3*c → c ≤ 3*a → c ≤ 3*b → 
  a * b * c ≥ 9 / 343 := by
sorry

end min_product_abc_l2942_294241


namespace max_value_ab_l2942_294275

theorem max_value_ab (a b : ℝ) : 
  (∀ x : ℝ, Real.exp (x + 1) ≥ a * x + b) → 
  a * b ≤ (1/2) * Real.exp 3 := by
sorry

end max_value_ab_l2942_294275


namespace blood_type_sample_size_l2942_294285

/-- Given a population of students with known blood types, calculate the number of students
    with a specific blood type that should be drawn in a stratified sample. -/
theorem blood_type_sample_size (total_students sample_size blood_type_O : ℕ)
    (h1 : total_students = 500)
    (h2 : blood_type_O = 200)
    (h3 : sample_size = 40) :
    (blood_type_O : ℚ) / total_students * sample_size = 16 := by
  sorry


end blood_type_sample_size_l2942_294285


namespace solve_for_a_l2942_294278

theorem solve_for_a : ∃ a : ℝ, 
  (2 * 1 - a * (-1) = 3) ∧ (a = 1) := by sorry

end solve_for_a_l2942_294278


namespace team_selection_count_l2942_294272

/-- The number of ways to select a team of 4 boys from 10 boys and 4 girls from 12 girls -/
def select_team : ℕ :=
  Nat.choose 10 4 * Nat.choose 12 4

/-- Theorem stating that the number of ways to select the team is 103950 -/
theorem team_selection_count : select_team = 103950 := by
  sorry

end team_selection_count_l2942_294272


namespace largest_unique_solution_m_l2942_294249

theorem largest_unique_solution_m (x y : ℕ) (m : ℕ) : 
  (∃! (x y : ℕ), 2005 * x + 2007 * y = m) → m ≤ 2 * 2005 * 2007 ∧ 
  (∃! (x y : ℕ), 2005 * x + 2007 * y = 2 * 2005 * 2007) :=
sorry

end largest_unique_solution_m_l2942_294249


namespace union_equality_iff_m_range_l2942_294290

def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 ≤ x ∧ x ≤ 2*m + 1}

theorem union_equality_iff_m_range (m : ℝ) : 
  A ∪ B m = A ↔ -1/2 ≤ m ∧ m ≤ 2 := by sorry

end union_equality_iff_m_range_l2942_294290


namespace find_A_in_terms_of_B_and_C_l2942_294263

/-- Given two functions f and g, and constants A, B, and C, prove that A can be expressed in terms of B and C. -/
theorem find_A_in_terms_of_B_and_C
  (f g : ℝ → ℝ)
  (A B C : ℝ)
  (h₁ : ∀ x, f x = A * x^2 - 3 * B * C)
  (h₂ : ∀ x, g x = C * x^2)
  (h₃ : B ≠ 0)
  (h₄ : C ≠ 0)
  (h₅ : f (g 2) = A - 3 * C) :
  A = (3 * C * (B - 1)) / (16 * C^2 - 1) := by
sorry


end find_A_in_terms_of_B_and_C_l2942_294263


namespace quadratic_inequality_range_l2942_294279

theorem quadratic_inequality_range (m : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 - 4*x ≥ m) ↔ m ≤ -3 :=
sorry

end quadratic_inequality_range_l2942_294279


namespace necessary_not_sufficient_condition_quadratic_inequality_condition_l2942_294291

-- Statement 1
theorem necessary_not_sufficient_condition (x : ℝ) :
  (x + |x| > 0 → x ≠ 0) ∧ ¬(x ≠ 0 → x + |x| > 0) := by sorry

-- Statement 2
theorem quadratic_inequality_condition (a b c : ℝ) :
  (a > 0 ∧ b^2 - 4*a*c ≤ 0) ↔ 
  (∀ x : ℝ, a*x^2 + b*x + c ≥ 0) := by sorry

end necessary_not_sufficient_condition_quadratic_inequality_condition_l2942_294291


namespace parabola_point_coordinates_l2942_294295

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : p.equation x y

theorem parabola_point_coordinates (p : Parabola) (F : ℝ × ℝ) (A : PointOnParabola p) :
  p.equation = (fun x y => y^2 = 4*x) →
  F = (1, 0) →
  (A.x, A.y) • (1 - A.x, -A.y) = -4 →
  (A.x = 1 ∧ A.y = 2) ∨ (A.x = 1 ∧ A.y = -2) := by
  sorry

end parabola_point_coordinates_l2942_294295


namespace equation_solutions_l2942_294217

theorem equation_solutions : ∃ (x₁ x₂ x₃ : ℝ),
  (x₁ = 3 ∧ x₂ = (2 + Real.sqrt 1121) / 14 ∧ x₃ = (2 - Real.sqrt 1121) / 14) ∧
  (∀ x : ℝ, (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃)) :=
by sorry

end equation_solutions_l2942_294217


namespace closest_integer_to_sqrt_11_l2942_294268

theorem closest_integer_to_sqrt_11 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - Real.sqrt 11| ≤ |m - Real.sqrt 11| ∧ n = 3 := by
  sorry

end closest_integer_to_sqrt_11_l2942_294268


namespace faster_train_speed_l2942_294248

-- Define the lengths of the trains in meters
def train1_length : ℝ := 200
def train2_length : ℝ := 160

-- Define the time taken to cross in seconds
def crossing_time : ℝ := 11.999040076793857

-- Define the speed of the slower train in km/h
def slower_train_speed : ℝ := 40

-- Define the conversion factor from m/s to km/h
def ms_to_kmh : ℝ := 3.6

-- Theorem statement
theorem faster_train_speed : 
  ∃ (faster_speed : ℝ),
    faster_speed = 68 ∧ 
    (train1_length + train2_length) / crossing_time * ms_to_kmh = faster_speed + slower_train_speed :=
by sorry

end faster_train_speed_l2942_294248


namespace if_A_then_all_short_answer_correct_l2942_294261

/-- Represents the condition for receiving an A grade -/
def receivedA (allShortAnswerCorrect : Bool) (multipleChoicePercentage : ℝ) : Prop :=
  allShortAnswerCorrect ∧ multipleChoicePercentage ≥ 90

/-- Proves that if a student received an A, they must have answered all short-answer questions correctly -/
theorem if_A_then_all_short_answer_correct 
  (student : String) 
  (studentReceivedA : Bool) 
  (studentAllShortAnswerCorrect : Bool) 
  (studentMultipleChoicePercentage : ℝ) : 
  (receivedA studentAllShortAnswerCorrect studentMultipleChoicePercentage → studentReceivedA) →
  (studentReceivedA → studentAllShortAnswerCorrect) :=
by sorry

end if_A_then_all_short_answer_correct_l2942_294261


namespace truncated_cone_rope_theorem_l2942_294221

/-- Represents a truncated cone with given dimensions -/
structure TruncatedCone where
  r₁ : ℝ  -- Upper base radius
  r₂ : ℝ  -- Lower base radius
  h : ℝ   -- Slant height

/-- Calculates the minimum length of the rope for a given truncated cone -/
def min_rope_length (cone : TruncatedCone) : ℝ := sorry

/-- Calculates the minimum distance from the rope to the upper base circumference -/
def min_distance_to_upper_base (cone : TruncatedCone) : ℝ := sorry

theorem truncated_cone_rope_theorem (cone : TruncatedCone) 
  (h₁ : cone.r₁ = 5)
  (h₂ : cone.r₂ = 10)
  (h₃ : cone.h = 20) :
  (min_rope_length cone = 50) ∧ 
  (min_distance_to_upper_base cone = 4) := by sorry

end truncated_cone_rope_theorem_l2942_294221


namespace stratified_sample_middle_school_l2942_294255

/-- Represents the number of students in a school -/
structure School :=
  (students : ℕ)

/-- Represents a stratified sampling plan -/
structure StratifiedSample :=
  (schoolA : School)
  (schoolB : School)
  (schoolC : School)
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (isArithmeticSequence : schoolA.students + schoolC.students = 2 * schoolB.students)

/-- The theorem statement -/
theorem stratified_sample_middle_school 
  (sample : StratifiedSample)
  (h1 : sample.totalStudents = 1500)
  (h2 : sample.sampleSize = 120) :
  ∃ (d : ℕ), 
    sample.schoolA.students = 40 - d ∧ 
    sample.schoolB.students = 40 ∧ 
    sample.schoolC.students = 40 + d :=
sorry

end stratified_sample_middle_school_l2942_294255


namespace geometric_sequence_ratio_l2942_294215

theorem geometric_sequence_ratio (a₁ : ℝ) (q : ℝ) (h₁ : a₁ ≠ 0) :
  let S : ℕ → ℝ
    | 1 => a₁
    | 2 => a₁ + a₁ * q
    | 3 => a₁ + a₁ * q + a₁ * q^2
    | _ => 0  -- We only need S₁, S₂, and S₃ for this problem
  (S 3 - S 2 = S 2 - S 1) → q = -1/2 := by
  sorry

end geometric_sequence_ratio_l2942_294215


namespace sum_of_squares_of_quadratic_roots_l2942_294210

theorem sum_of_squares_of_quadratic_roots : ∀ (s₁ s₂ : ℝ), 
  s₁^2 - 20*s₁ + 32 = 0 → 
  s₂^2 - 20*s₂ + 32 = 0 → 
  s₁^2 + s₂^2 = 336 := by
  sorry

end sum_of_squares_of_quadratic_roots_l2942_294210


namespace optimal_rental_plan_minimum_transportation_cost_l2942_294247

/-- Represents the rental plan for trucks -/
structure RentalPlan where
  truckA : ℕ
  truckB : ℕ

/-- Checks if a rental plan is valid according to the problem constraints -/
def isValidPlan (plan : RentalPlan) : Prop :=
  plan.truckA + plan.truckB = 6 ∧
  45 * plan.truckA + 30 * plan.truckB ≥ 240 ∧
  400 * plan.truckA + 300 * plan.truckB ≤ 2300

/-- Calculates the total cost of a rental plan -/
def totalCost (plan : RentalPlan) : ℕ :=
  400 * plan.truckA + 300 * plan.truckB

/-- Theorem stating that the optimal plan is 4 Truck A and 2 Truck B -/
theorem optimal_rental_plan :
  ∀ (plan : RentalPlan),
    isValidPlan plan →
    totalCost plan ≥ totalCost { truckA := 4, truckB := 2 } :=
by sorry

/-- Corollary stating the minimum transportation cost -/
theorem minimum_transportation_cost :
  totalCost { truckA := 4, truckB := 2 } = 2200 :=
by sorry

end optimal_rental_plan_minimum_transportation_cost_l2942_294247


namespace winnie_kept_balloons_l2942_294237

/-- The number of balloons Winnie keeps for herself after distributing
    as many as possible equally among her friends -/
def balloons_kept (total_balloons : ℕ) (num_friends : ℕ) : ℕ :=
  total_balloons % num_friends

theorem winnie_kept_balloons :
  balloons_kept 200 12 = 8 := by
  sorry

end winnie_kept_balloons_l2942_294237


namespace gcd_7488_12467_l2942_294213

theorem gcd_7488_12467 : Nat.gcd 7488 12467 = 39 := by
  sorry

end gcd_7488_12467_l2942_294213


namespace max_cubic_sum_under_constraint_l2942_294211

theorem max_cubic_sum_under_constraint (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + a + b + c + d = 8) :
  a^3 + b^3 + c^3 + d^3 ≤ 15.625 := by
  sorry

end max_cubic_sum_under_constraint_l2942_294211


namespace challenge_probabilities_l2942_294277

/-- A challenge with 3 equally difficult questions -/
structure Challenge where
  num_questions : ℕ := 3
  num_chances : ℕ := 3
  correct_prob : ℝ := 0.7

/-- The probability of passing the challenge on the second attempt -/
def prob_pass_second_attempt (c : Challenge) : ℝ :=
  (1 - c.correct_prob) * c.correct_prob

/-- The overall probability of passing the challenge -/
def prob_pass_challenge (c : Challenge) : ℝ :=
  1 - (1 - c.correct_prob) ^ c.num_chances

/-- Theorem stating the probabilities for the given challenge -/
theorem challenge_probabilities (c : Challenge) :
  prob_pass_second_attempt c = 0.21 ∧ prob_pass_challenge c = 0.973 := by
  sorry

#check challenge_probabilities

end challenge_probabilities_l2942_294277


namespace range_of_a_l2942_294232

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x y : ℝ, x - y + a = 0 ∧ x^2 + y^2 - 2*x = 1

def q (a : ℝ) : Prop := ∀ x : ℝ, Real.exp x - a > 1

-- State the theorem
theorem range_of_a (a : ℝ) : p a ∧ ¬(q a) → -1 < a ∧ a < 1 := by
  sorry

end range_of_a_l2942_294232


namespace mork_tax_rate_l2942_294231

/-- Proves that Mork's tax rate is 10% given the specified conditions --/
theorem mork_tax_rate (mork_income : ℝ) (mork_tax_rate : ℝ) 
  (h1 : mork_income > 0)
  (h2 : mork_tax_rate > 0)
  (h3 : mork_tax_rate < 1)
  (h4 : (mork_tax_rate * mork_income + 3 * 0.2 * mork_income) / (4 * mork_income) = 0.175) :
  mork_tax_rate = 0.1 := by
sorry


end mork_tax_rate_l2942_294231


namespace odd_numbers_sum_product_equality_l2942_294227

/-- For a positive integer n, there exist n positive odd numbers whose sum equals 
    their product if and only if n is of the form 4k + 1, where k is a non-negative integer. -/
theorem odd_numbers_sum_product_equality (n : ℕ+) : 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ x ∈ S, Odd x ∧ x > 0) ∧ 
    (S.sum id = S.prod id)) ↔ 
  ∃ k : ℕ, n = 4 * k + 1 := by
  sorry

end odd_numbers_sum_product_equality_l2942_294227


namespace remainder_problem_l2942_294269

theorem remainder_problem (n : ℕ) (r₃ r₆ r₉ : ℕ) :
  r₃ < 3 ∧ r₆ < 6 ∧ r₉ < 9 →
  n % 3 = r₃ ∧ n % 6 = r₆ ∧ n % 9 = r₉ →
  r₃ + r₆ + r₉ = 15 →
  n % 18 = 17 := by sorry

end remainder_problem_l2942_294269


namespace pigeonhole_multiples_of_five_l2942_294274

theorem pigeonhole_multiples_of_five (n : ℕ) (h : n = 200) : 
  ∀ (S : Finset ℕ), S ⊆ Finset.range n → S.card ≥ 82 → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a + b) % 5 = 0 :=
by sorry

end pigeonhole_multiples_of_five_l2942_294274


namespace total_fruits_in_bowl_l2942_294205

/-- The total number of fruits in a bowl, given the number of bananas, 
    apples (twice the number of bananas), and oranges. -/
theorem total_fruits_in_bowl (bananas : ℕ) (oranges : ℕ) : 
  bananas = 2 → oranges = 6 → bananas + 2 * bananas + oranges = 12 := by
  sorry

end total_fruits_in_bowl_l2942_294205


namespace fruit_purchase_cost_l2942_294270

theorem fruit_purchase_cost (strawberry_price : ℝ) (cherry_price : ℝ) (blueberry_price : ℝ)
  (strawberry_amount : ℝ) (cherry_amount : ℝ) (blueberry_amount : ℝ)
  (blueberry_discount : ℝ) (bag_fee : ℝ) :
  strawberry_price = 2.20 →
  cherry_price = 6 * strawberry_price →
  blueberry_price = cherry_price / 2 →
  strawberry_amount = 3 →
  cherry_amount = 4.5 →
  blueberry_amount = 6.2 →
  blueberry_discount = 0.15 →
  bag_fee = 0.75 →
  strawberry_price * strawberry_amount +
  cherry_price * cherry_amount +
  blueberry_price * blueberry_amount * (1 - blueberry_discount) +
  bag_fee = 101.53 := by
sorry

end fruit_purchase_cost_l2942_294270


namespace mona_monday_miles_l2942_294230

/-- Represents Mona's biking schedule for a week --/
structure BikingWeek where
  total_miles : ℝ
  monday_miles : ℝ
  wednesday_miles : ℝ
  saturday_miles : ℝ
  steep_trail_speed : ℝ
  flat_road_speed : ℝ
  saturday_speed_reduction : ℝ

/-- Theorem stating that Mona biked 6 miles on Monday --/
theorem mona_monday_miles (week : BikingWeek) 
  (h1 : week.total_miles = 30)
  (h2 : week.wednesday_miles = 12)
  (h3 : week.saturday_miles = 2 * week.monday_miles)
  (h4 : week.steep_trail_speed = 6)
  (h5 : week.flat_road_speed = 15)
  (h6 : week.saturday_speed_reduction = 0.2)
  (h7 : week.total_miles = week.monday_miles + week.wednesday_miles + week.saturday_miles) :
  week.monday_miles = 6 := by
  sorry

#check mona_monday_miles

end mona_monday_miles_l2942_294230


namespace reactions_not_usable_in_primary_cell_l2942_294273

-- Define the types of reactions
inductive ReactionType
| Neutralization
| Redox
| Endothermic

-- Define a structure for chemical reactions
structure ChemicalReaction where
  id : Nat
  reactionType : ReactionType
  isExothermic : Bool

-- Define the condition for a reaction to be used in a primary cell
def canBeUsedInPrimaryCell (reaction : ChemicalReaction) : Prop :=
  reaction.reactionType = ReactionType.Redox ∧ reaction.isExothermic

-- Define the given reactions
def reaction1 : ChemicalReaction :=
  { id := 1, reactionType := ReactionType.Neutralization, isExothermic := true }

def reaction2 : ChemicalReaction :=
  { id := 2, reactionType := ReactionType.Redox, isExothermic := true }

def reaction3 : ChemicalReaction :=
  { id := 3, reactionType := ReactionType.Redox, isExothermic := true }

def reaction4 : ChemicalReaction :=
  { id := 4, reactionType := ReactionType.Endothermic, isExothermic := false }

-- Theorem to prove
theorem reactions_not_usable_in_primary_cell :
  ¬(canBeUsedInPrimaryCell reaction1) ∧ ¬(canBeUsedInPrimaryCell reaction4) :=
sorry

end reactions_not_usable_in_primary_cell_l2942_294273


namespace fraction_subtraction_l2942_294253

theorem fraction_subtraction : (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end fraction_subtraction_l2942_294253


namespace defeated_candidate_vote_percentage_l2942_294208

theorem defeated_candidate_vote_percentage 
  (total_polled_votes : ℕ) 
  (invalid_votes : ℕ) 
  (vote_difference : ℕ) 
  (h1 : total_polled_votes = 90083) 
  (h2 : invalid_votes = 83) 
  (h3 : vote_difference = 9000) : 
  let valid_votes := total_polled_votes - invalid_votes
  let defeated_votes := (valid_votes - vote_difference) / 2
  defeated_votes * 100 / valid_votes = 45 := by
sorry

end defeated_candidate_vote_percentage_l2942_294208


namespace other_solution_quadratic_l2942_294225

theorem other_solution_quadratic (h : 40 * (4/5)^2 - 69 * (4/5) + 24 = 0) :
  40 * (3/8)^2 - 69 * (3/8) + 24 = 0 := by
  sorry

end other_solution_quadratic_l2942_294225


namespace line_intersects_circle_l2942_294206

/-- The line l in polar coordinates -/
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = Real.sqrt 2

/-- The circle C in Cartesian coordinates -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The line l in Cartesian coordinates -/
def line_l_cartesian (x y : ℝ) : Prop := x + y = 2

/-- Theorem stating that the line l intersects the circle C -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), line_l_cartesian x y ∧ circle_C x y :=
sorry

end line_intersects_circle_l2942_294206


namespace max_sum_with_divisibility_conditions_l2942_294250

theorem max_sum_with_divisibility_conditions (a b c : ℕ) : 
  a > 2022 → b > 2022 → c > 2022 →
  (c - 2022) ∣ (a + b) →
  (b - 2022) ∣ (a + c) →
  (a - 2022) ∣ (b + c) →
  a + b + c ≤ 2022 * 85 := by
sorry

end max_sum_with_divisibility_conditions_l2942_294250


namespace smallest_possible_a_l2942_294254

theorem smallest_possible_a (P : ℤ → ℤ) (a : ℕ+) : 
  (∀ x, ∃ (c : ℤ), P x = c) →  -- P has integer coefficients
  (P 1 = a) →
  (P 3 = a) →
  (P 2 = -a) →
  (P 4 = -a) →
  (P 6 = -a) →
  (P 8 = -a) →
  (∀ b : ℕ+, b < 105 → 
    ¬(∃ Q : ℤ → ℤ, 
      (∀ x, ∃ (c : ℤ), Q x = c) ∧  -- Q has integer coefficients
      (Q 1 = b) ∧
      (Q 3 = b) ∧
      (Q 2 = -b) ∧
      (Q 4 = -b) ∧
      (Q 6 = -b) ∧
      (Q 8 = -b)
    )
  ) →
  a = 105 :=
by sorry

end smallest_possible_a_l2942_294254


namespace right_triangle_third_side_product_l2942_294298

theorem right_triangle_third_side_product (a b c : ℝ) : 
  (a = 6 ∧ b = 8 ∧ a^2 + b^2 = c^2) ∨ (a = 6 ∧ c = 8 ∧ a^2 + b^2 = c^2) → 
  c * b = 20 * Real.sqrt 7 := by
sorry

end right_triangle_third_side_product_l2942_294298


namespace samantha_bus_time_l2942_294282

/-- Calculates the time Samantha spends on the bus given her schedule --/
theorem samantha_bus_time :
  let leave_time : Nat := 7 * 60 + 15  -- 7:15 AM in minutes
  let return_time : Nat := 17 * 60 + 15  -- 5:15 PM in minutes
  let total_away_time : Nat := return_time - leave_time
  let class_time : Nat := 8 * 45  -- 8 classes of 45 minutes each
  let lunch_time : Nat := 40
  let extracurricular_time : Nat := 90
  let total_school_time : Nat := class_time + lunch_time + extracurricular_time
  let bus_time : Nat := total_away_time - total_school_time
  bus_time = 110 := by sorry

end samantha_bus_time_l2942_294282


namespace solution_values_l2942_294293

def has_twenty_solutions (n : ℕ+) : Prop :=
  (Finset.filter (fun (x, y, z) => 3 * x + 4 * y + z = n) 
    (Finset.product (Finset.range n) (Finset.product (Finset.range n) (Finset.range n)))).card = 20

theorem solution_values (n : ℕ+) (h : has_twenty_solutions n) : n = 21 ∨ n = 22 :=
sorry

end solution_values_l2942_294293


namespace product_of_real_and_imaginary_parts_l2942_294264

theorem product_of_real_and_imaginary_parts : ∃ (z : ℂ), z = (2 + Complex.I) * Complex.I ∧ (z.re * z.im = -2) := by
  sorry

end product_of_real_and_imaginary_parts_l2942_294264


namespace system_solution_1_l2942_294243

theorem system_solution_1 (x y : ℚ) : 
  (3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33) ↔ (x = 6 ∧ y = -1/2) :=
by sorry

end system_solution_1_l2942_294243


namespace max_popsicles_for_10_dollars_l2942_294289

/-- Represents the number of popsicles in a box -/
inductive BoxSize
  | Single : BoxSize
  | Three : BoxSize
  | Five : BoxSize
  | Seven : BoxSize

/-- Returns the cost of a box given its size -/
def boxCost (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 2
  | BoxSize.Five => 3
  | BoxSize.Seven => 4

/-- Returns the number of popsicles in a box given its size -/
def boxPopsicles (size : BoxSize) : ℕ :=
  match size with
  | BoxSize.Single => 1
  | BoxSize.Three => 3
  | BoxSize.Five => 5
  | BoxSize.Seven => 7

/-- Represents a purchase of popsicle boxes -/
structure Purchase where
  single : ℕ
  three : ℕ
  five : ℕ
  seven : ℕ

/-- Calculates the total cost of a purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.single * boxCost BoxSize.Single +
  p.three * boxCost BoxSize.Three +
  p.five * boxCost BoxSize.Five +
  p.seven * boxCost BoxSize.Seven

/-- Calculates the total number of popsicles in a purchase -/
def totalPopsicles (p : Purchase) : ℕ :=
  p.single * boxPopsicles BoxSize.Single +
  p.three * boxPopsicles BoxSize.Three +
  p.five * boxPopsicles BoxSize.Five +
  p.seven * boxPopsicles BoxSize.Seven

/-- Theorem: The maximum number of popsicles that can be bought with $10 is 17 -/
theorem max_popsicles_for_10_dollars :
  ∀ p : Purchase, totalCost p ≤ 10 → totalPopsicles p ≤ 17 ∧
  ∃ p : Purchase, totalCost p ≤ 10 ∧ totalPopsicles p = 17 :=
by sorry

end max_popsicles_for_10_dollars_l2942_294289


namespace two_numbers_problem_l2942_294226

theorem two_numbers_problem (x y : ℕ+) : 
  x + y = 667 →
  Nat.lcm x y / Nat.gcd x y = 120 →
  ((x = 232 ∧ y = 435) ∨ (x = 552 ∧ y = 115)) :=
by sorry

end two_numbers_problem_l2942_294226


namespace polynomial_factor_l2942_294288

theorem polynomial_factor (x : ℝ) :
  ∃ (k : ℝ), (29 * 37 * x^4 + 2 * x^2 + 9) = k * (x^2 - 2*x + 3) := by
  sorry

end polynomial_factor_l2942_294288


namespace keith_initial_cards_l2942_294262

/-- Represents the number of cards in Keith's collection --/
structure CardCollection where
  initial : ℕ
  added : ℕ
  remaining : ℕ

/-- Theorem stating the initial number of cards in Keith's collection --/
theorem keith_initial_cards (c : CardCollection) 
  (h1 : c.added = 8)
  (h2 : c.remaining = 46)
  (h3 : c.remaining * 2 = c.initial + c.added) :
  c.initial = 84 := by
  sorry

end keith_initial_cards_l2942_294262


namespace total_students_l2942_294222

theorem total_students (S : ℕ) (T : ℕ) : 
  T = 6 * S - 78 →
  T - S = 2222 →
  T = 2682 := by
sorry

end total_students_l2942_294222


namespace vector_perpendicular_l2942_294283

def problem1 (p q : ℝ × ℝ) : Prop :=
  p = (1, 2) ∧ 
  ∃ m : ℝ, q = (m, 1) ∧ 
  p.1 * q.1 + p.2 * q.2 = 0 →
  ‖q‖ = Real.sqrt 5

theorem vector_perpendicular : problem1 (1, 2) (-2, 1) := by sorry

end vector_perpendicular_l2942_294283


namespace units_digit_sum_base9_l2942_294266

/-- The units digit of a number in base 9 -/
def unitsDigitBase9 (n : ℕ) : ℕ := n % 9

/-- Addition in base 9 -/
def addBase9 (a b : ℕ) : ℕ := (a + b) % 9

theorem units_digit_sum_base9 :
  unitsDigitBase9 (addBase9 45 76) = 2 := by sorry

end units_digit_sum_base9_l2942_294266


namespace stratified_sampling_theorem_l2942_294233

/-- Represents a workshop with its production quantity -/
structure Workshop where
  production : ℕ

/-- Represents a stratified sampling scenario -/
structure StratifiedSampling where
  workshops : List Workshop
  sampleSizes : List ℕ

def StratifiedSampling.totalSampleSize (s : StratifiedSampling) : ℕ :=
  s.sampleSizes.sum

def StratifiedSampling.isValid (s : StratifiedSampling) : Prop :=
  s.workshops.length = s.sampleSizes.length ∧ 
  s.sampleSizes.all (· > 0)

theorem stratified_sampling_theorem (s : StratifiedSampling) 
  (h1 : s.workshops = [⟨120⟩, ⟨90⟩, ⟨60⟩])
  (h2 : s.sampleSizes.length = 3)
  (h3 : s.sampleSizes[2] = 2)
  (h4 : s.isValid)
  (h5 : ∀ s' : StratifiedSampling, s'.workshops = s.workshops → 
        s'.isValid → s'.totalSampleSize ≥ s.totalSampleSize) :
  s.totalSampleSize = 9 := by
  sorry

end stratified_sampling_theorem_l2942_294233


namespace arithmetic_mean_problem_l2942_294246

theorem arithmetic_mean_problem (x : ℚ) : 
  ((x + 10) + 17 + (3 * x) + 15 + (3 * x + 6)) / 5 = 26 → x = 82 / 7 := by
  sorry

end arithmetic_mean_problem_l2942_294246


namespace power_of_81_l2942_294276

theorem power_of_81 : 81^(8/3) = 59049 * (9^(1/3)) := by sorry

end power_of_81_l2942_294276


namespace distinct_collections_is_125_l2942_294228

/-- Represents the word "COMPUTATIONS" -/
def word : String := "COMPUTATIONS"

/-- The number of vowels in the word -/
def num_vowels : Nat := 5

/-- The number of consonants in the word, excluding T's -/
def num_consonants_without_t : Nat := 5

/-- The number of T's in the word -/
def num_t : Nat := 2

/-- The number of vowels to select -/
def vowels_to_select : Nat := 4

/-- The number of consonants to select -/
def consonants_to_select : Nat := 4

/-- Calculates the number of distinct collections of letters -/
def distinct_collections : Nat :=
  (Nat.choose num_vowels vowels_to_select) * 
  ((Nat.choose num_consonants_without_t consonants_to_select) + 
   (Nat.choose num_consonants_without_t (consonants_to_select - 1)) +
   (Nat.choose num_consonants_without_t (consonants_to_select - 2)))

/-- Theorem stating that the number of distinct collections is 125 -/
theorem distinct_collections_is_125 : distinct_collections = 125 := by
  sorry

end distinct_collections_is_125_l2942_294228


namespace safe_menu_fraction_l2942_294238

theorem safe_menu_fraction (total_dishes : ℕ) (vegetarian_dishes : ℕ) (gluten_free_vegetarian : ℕ) :
  vegetarian_dishes = total_dishes / 3 →
  gluten_free_vegetarian = vegetarian_dishes - 5 →
  (gluten_free_vegetarian : ℚ) / total_dishes = 1 / 18 := by
  sorry

end safe_menu_fraction_l2942_294238


namespace inverse_cube_root_relation_l2942_294281

/-- Given that z varies inversely as the cube root of x, and z = 2 when x = 8,
    prove that x = 1 when z = 4. -/
theorem inverse_cube_root_relation (z x : ℝ) (h1 : z * x^(1/3) = 2 * 8^(1/3)) :
  z = 4 → x = 1 := by
  sorry

end inverse_cube_root_relation_l2942_294281


namespace base_conversion_l2942_294244

theorem base_conversion (b : ℝ) : b > 0 → (3 * 5 + 2 = b^2 + 2) → b = Real.sqrt 15 := by
  sorry

end base_conversion_l2942_294244


namespace investment_calculation_l2942_294240

/-- Calculates the investment amount given dividend information -/
theorem investment_calculation (face_value premium dividend_rate dividend_received : ℚ) : 
  face_value = 100 →
  premium = 20 / 100 →
  dividend_rate = 5 / 100 →
  dividend_received = 600 →
  (dividend_received / (face_value * dividend_rate)) * (face_value * (1 + premium)) = 14400 := by
  sorry

end investment_calculation_l2942_294240


namespace bhanu_house_rent_expenditure_l2942_294236

/-- Calculates Bhanu's expenditure on house rent given his spending patterns and petrol expense -/
def house_rent_expenditure (total_income : ℝ) (petrol_percentage : ℝ) (rent_percentage : ℝ) (petrol_expense : ℝ) : ℝ :=
  let remaining_income := total_income - petrol_expense
  rent_percentage * remaining_income

/-- Proves that Bhanu's expenditure on house rent is 210 given his spending patterns and petrol expense -/
theorem bhanu_house_rent_expenditure :
  ∀ (total_income : ℝ),
    total_income > 0 →
    house_rent_expenditure total_income 0.3 0.3 300 = 210 :=
by
  sorry

#eval house_rent_expenditure 1000 0.3 0.3 300

end bhanu_house_rent_expenditure_l2942_294236


namespace max_cross_sectional_area_l2942_294234

-- Define the prism
def prism_base_side_length : ℝ := 8

-- Define the cutting plane
def cutting_plane (x y z : ℝ) : Prop := 3 * x - 5 * y + 2 * z = 20

-- Define the cross-sectional area function
noncomputable def cross_sectional_area (h : ℝ) : ℝ := 
  let diagonal := (2 * prism_base_side_length ^ 2 + h ^ 2) ^ (1/2 : ℝ)
  let area := h * diagonal / 2
  area

-- Theorem statement
theorem max_cross_sectional_area :
  ∃ h : ℝ, h > 0 ∧ 
    cross_sectional_area h = 9 * (38 : ℝ).sqrt ∧
    ∀ h' : ℝ, h' > 0 → cross_sectional_area h' ≤ cross_sectional_area h :=
by sorry

end max_cross_sectional_area_l2942_294234


namespace compute_expression_l2942_294218

theorem compute_expression : 9 + 4 * (5 - 2 * 3)^2 = 13 := by
  sorry

end compute_expression_l2942_294218


namespace order_of_7_wrt_g_l2942_294229

def g (x : ℕ) : ℕ := x^2 % 13

def g_iter (n : ℕ) (x : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n+1 => g (g_iter n x)

theorem order_of_7_wrt_g :
  (∀ k < 12, g_iter k 7 ≠ 7) ∧ g_iter 12 7 = 7 :=
sorry

end order_of_7_wrt_g_l2942_294229


namespace number_puzzle_l2942_294256

theorem number_puzzle : ∃ x : ℚ, (x / 5 + 6 = x / 4 - 6) ∧ x = 240 := by
  sorry

end number_puzzle_l2942_294256


namespace no_resident_claims_to_be_liar_l2942_294284

-- Define the types of residents on the island
inductive Resident
| Knight
| Liar

-- Define the statement made by a resident
def makes_statement (r : Resident) : Prop :=
  match r with
  | Resident.Knight => True   -- Knights always tell the truth
  | Resident.Liar => False    -- Liars always lie

-- Define the statement "I am a liar"
def claims_to_be_liar (r : Resident) : Prop :=
  makes_statement r = (r = Resident.Liar)

-- Theorem: No resident can claim to be a liar
theorem no_resident_claims_to_be_liar :
  ∀ r : Resident, ¬(claims_to_be_liar r) :=
by sorry

end no_resident_claims_to_be_liar_l2942_294284


namespace restaurant_tax_calculation_l2942_294202

-- Define the tax calculation function
def calculate_tax (turnover : ℕ) : ℕ :=
  if turnover ≤ 1000 then
    300
  else
    300 + (turnover - 1000) * 4 / 100

-- Theorem statement
theorem restaurant_tax_calculation :
  calculate_tax 35000 = 1660 :=
by sorry

end restaurant_tax_calculation_l2942_294202
