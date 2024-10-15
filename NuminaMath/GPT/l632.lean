import Mathlib

namespace NUMINAMATH_GPT_graph_of_abs_g_l632_63294

noncomputable def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 4 then x - 2
  else 0

noncomputable def abs_g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ -3 then -(x + 3)
  else if -3 < x ∧ x ≤ -1 then x + 3
  else if -1 < x ∧ x ≤ 1 then -x^2 + 2
  else if 1 < x ∧ x ≤ 2 then -(x - 2)
  else if 2 < x ∧ x ≤ 4 then x - 2
  else 0

theorem graph_of_abs_g :
  ∀ x : ℝ, abs_g x = |g x| :=
by
  sorry

end NUMINAMATH_GPT_graph_of_abs_g_l632_63294


namespace NUMINAMATH_GPT_sin_60_eq_sqrt_three_div_two_l632_63227

theorem sin_60_eq_sqrt_three_div_two :
  Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_60_eq_sqrt_three_div_two_l632_63227


namespace NUMINAMATH_GPT_system_soln_l632_63202

theorem system_soln (a1 b1 a2 b2 : ℚ)
  (h1 : a1 * 3 + b1 * 6 = 21)
  (h2 : a2 * 3 + b2 * 6 = 12) :
  (3 = 3 ∧ -3 = -3) ∧ (a1 * (2 * 3 + -3) + b1 * (3 - -3) = 21) ∧ (a2 * (2 * 3 + -3) + b2 * (3 - -3) = 12) :=
by
  sorry

end NUMINAMATH_GPT_system_soln_l632_63202


namespace NUMINAMATH_GPT_max_m_value_l632_63286

theorem max_m_value (a b m : ℝ) (ha : a > 0) (hb : b > 0) (H : (3/a + 1/b) ≥ m / (a + 3 * b)) : m ≤ 12 :=
sorry

end NUMINAMATH_GPT_max_m_value_l632_63286


namespace NUMINAMATH_GPT_binary_multiplication_l632_63226

theorem binary_multiplication : (0b1101 * 0b111 = 0b1001111) :=
by {
  -- placeholder for proof
  sorry
}

end NUMINAMATH_GPT_binary_multiplication_l632_63226


namespace NUMINAMATH_GPT_symmetric_codes_count_l632_63261

def isSymmetric (grid : List (List Bool)) : Prop :=
  -- condition for symmetry: rotational and reflectional symmetry
  sorry

def isValidCode (grid : List (List Bool)) : Prop :=
  -- condition for valid scanning code with at least one black and one white
  sorry

noncomputable def numberOfSymmetricCodes : Nat :=
  -- function to count the number of symmetric valid codes
  sorry

theorem symmetric_codes_count :
  numberOfSymmetricCodes = 62 := 
  sorry

end NUMINAMATH_GPT_symmetric_codes_count_l632_63261


namespace NUMINAMATH_GPT_smallest_missing_unit_digit_l632_63280

theorem smallest_missing_unit_digit :
  (∀ n, n ∈ [0, 1, 4, 5, 6, 9]) → ∃ smallest_digit, smallest_digit = 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_missing_unit_digit_l632_63280


namespace NUMINAMATH_GPT_moles_of_HCl_combined_l632_63287

/-- Prove the number of moles of Hydrochloric acid combined is 1, given that 
1 mole of Sodium hydroxide and some moles of Hydrochloric acid react to produce 
1 mole of Water, based on the balanced chemical equation: NaOH + HCl → NaCl + H2O -/
theorem moles_of_HCl_combined (moles_NaOH : ℕ) (moles_HCl : ℕ) (moles_H2O : ℕ)
  (h1 : moles_NaOH = 1) (h2 : moles_H2O = 1) 
  (balanced_eq : moles_NaOH = moles_HCl ∧ moles_HCl = moles_H2O) : 
  moles_HCl = 1 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_HCl_combined_l632_63287


namespace NUMINAMATH_GPT_true_propositions_identification_l632_63207

-- Definitions related to the propositions
def converse_prop1 (x y : ℝ) := (x + y = 0) → (x + y = 0)
-- Converse of additive inverses: If x and y are additive inverses, then x + y = 0
def converse_prop1_true (x y : ℝ) : Prop := (x + y = 0) → (x + y = 0)

def negation_prop2 : Prop := ¬(∀ (a b c d : ℝ), (a = b → c = d) → (a + b = c + d))
-- Negation of congruent triangles have equal areas: If two triangles are not congruent, areas not equal
def negation_prop2_false : Prop := ¬(∀ (a b c : ℝ), (a = b ∧ b ≠ c → a ≠ c))

def contrapositive_prop3 (q : ℝ) := (q ≤ 1) → (4 - 4 * q ≥ 0)
-- Contrapositive of real roots: If the equation x^2 + 2x + q = 0 does not have real roots then q > 1
def contrapositive_prop3_true (q : ℝ) : Prop := (4 - 4 * q < 0) → (q > 1)

def converse_prop4 (a b c : ℝ) := (a = b ∧ b = c ∧ c = a) → False
-- Converse of scalene triangle: If a triangle has three equal interior angles, it is a scalene triangle
def converse_prop4_false (a b c : ℝ) : Prop := (a = b ∧ b = c ∧ c = a) → False

theorem true_propositions_identification :
  (∀ x y : ℝ, converse_prop1_true x y) ∧
  ¬negation_prop2_false ∧
  (∀ q : ℝ, contrapositive_prop3_true q) ∧
  ¬(∀ a b c : ℝ, converse_prop4_false a b c) := by
  sorry

end NUMINAMATH_GPT_true_propositions_identification_l632_63207


namespace NUMINAMATH_GPT_expected_americans_with_allergies_l632_63219

theorem expected_americans_with_allergies (prob : ℚ) (sample_size : ℕ) (h_prob : prob = 1/5) (h_sample_size : sample_size = 250) :
  sample_size * prob = 50 := by
  rw [h_prob, h_sample_size]
  norm_num

#print expected_americans_with_allergies

end NUMINAMATH_GPT_expected_americans_with_allergies_l632_63219


namespace NUMINAMATH_GPT_rise_in_water_level_l632_63252

noncomputable def edge : ℝ := 15.0
noncomputable def base_length : ℝ := 20.0
noncomputable def base_width : ℝ := 15.0
noncomputable def volume_cube : ℝ := edge ^ 3
noncomputable def base_area : ℝ := base_length * base_width

theorem rise_in_water_level :
  (volume_cube / base_area) = 11.25 :=
by
  sorry

end NUMINAMATH_GPT_rise_in_water_level_l632_63252


namespace NUMINAMATH_GPT_proof_problem_l632_63230

noncomputable def A : Set ℝ := { x | x^2 - 4 = 0 }
noncomputable def B : Set ℝ := { y | ∃ x, y = x^2 - 4 }

theorem proof_problem :
  (A ∩ B = A) ∧ (A ∪ B = B) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l632_63230


namespace NUMINAMATH_GPT_prove_a_eq_neg2_solve_inequality_for_a_leq0_l632_63268

-- Problem 1: Proving that a = -2 given the solution set of the inequality
theorem prove_a_eq_neg2 (a : ℝ) (h : ∀ x : ℝ, (-1 < x ∧ x < -1/2) ↔ (ax - 1) * (x + 1) > 0) : a = -2 := sorry

-- Problem 2: Solving the inequality (ax-1)(x+1) > 0 for different conditions on a
theorem solve_inequality_for_a_leq0 (a x : ℝ) (h_a_le_0 : a ≤ 0) : 
  (ax - 1) * (x + 1) > 0 ↔ 
    if a < -1 then -1 < x ∧ x < 1/a
    else if a = -1 then false
    else if -1 < a ∧ a < 0 then 1/a < x ∧ x < -1
    else x < -1 := sorry

end NUMINAMATH_GPT_prove_a_eq_neg2_solve_inequality_for_a_leq0_l632_63268


namespace NUMINAMATH_GPT_range_M_l632_63242

theorem range_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b < 1) :
  1 < (1 / (1 + a)) + (1 / (1 + b)) ∧ (1 / (1 + a)) + (1 / (1 + b)) < 2 := by
  sorry

end NUMINAMATH_GPT_range_M_l632_63242


namespace NUMINAMATH_GPT_problem_statement_l632_63217

theorem problem_statement (x : ℝ) (h : 5 * x - 8 = 15 * x + 14) : 6 * (x + 3) = 4.8 :=
sorry

end NUMINAMATH_GPT_problem_statement_l632_63217


namespace NUMINAMATH_GPT_total_games_played_l632_63200

theorem total_games_played (n : ℕ) (h : n = 8) : (n.choose 2) = 28 := by
  sorry

end NUMINAMATH_GPT_total_games_played_l632_63200


namespace NUMINAMATH_GPT_smallest_e_value_l632_63247

noncomputable def poly := (1, -3, 7, -2/5)

theorem smallest_e_value (a b c d e : ℤ) 
  (h_poly_eq : a * (1)^4 + b * (1)^3 + c * (1)^2 + d * (1) + e = 0)
  (h_poly_eq_2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h_poly_eq_3 : a * 7^4 + b * 7^3 + c * 7^2 + d * 7 + e = 0)
  (h_poly_eq_4 : a * (-2/5)^4 + b * (-2/5)^3 + c * (-2/5)^2 + d * (-2/5) + e = 0)
  (h_e_positive : e > 0) :
  e = 42 :=
sorry

end NUMINAMATH_GPT_smallest_e_value_l632_63247


namespace NUMINAMATH_GPT_solution_correctness_l632_63208

noncomputable def solution_set : Set ℝ := { x : ℝ | (x + 1) * (x - 2) > 0 }

theorem solution_correctness (x : ℝ) :
  (x ∈ solution_set) ↔ (x < -1 ∨ x > 2) :=
by sorry

end NUMINAMATH_GPT_solution_correctness_l632_63208


namespace NUMINAMATH_GPT_number_of_action_figures_removed_l632_63285

-- Definitions for conditions
def initial : ℕ := 15
def added : ℕ := 2
def current : ℕ := 10

-- The proof statement
theorem number_of_action_figures_removed (initial added current : ℕ) : 
  (initial + added - current) = 7 := by
  sorry

end NUMINAMATH_GPT_number_of_action_figures_removed_l632_63285


namespace NUMINAMATH_GPT_combination_10_3_l632_63298

theorem combination_10_3 : Nat.choose 10 3 = 120 := by
  -- use the combination formula: \binom{n}{r} = n! / (r! * (n-r)!)
  sorry

end NUMINAMATH_GPT_combination_10_3_l632_63298


namespace NUMINAMATH_GPT_pie_shop_earnings_l632_63239

-- Define the conditions
def price_per_slice : ℕ := 3
def slices_per_pie : ℕ := 10
def number_of_pies : ℕ := 6

-- Calculate the total slices
def total_slices : ℕ := number_of_pies * slices_per_pie

-- Calculate the total earnings
def total_earnings : ℕ := total_slices * price_per_slice

-- State the theorem
theorem pie_shop_earnings : total_earnings = 180 :=
by
  -- Proof can be skipped with a sorry
  sorry

end NUMINAMATH_GPT_pie_shop_earnings_l632_63239


namespace NUMINAMATH_GPT_find_x_l632_63234

theorem find_x 
  (x : ℝ)
  (h : 0.4 * x + (0.6 * 0.8) = 0.56) : 
  x = 0.2 := sorry

end NUMINAMATH_GPT_find_x_l632_63234


namespace NUMINAMATH_GPT_salami_pizza_fraction_l632_63211

theorem salami_pizza_fraction 
    (d_pizza : ℝ) 
    (n_salami_diameter : ℕ) 
    (n_salami_total : ℕ) 
    (h1 : d_pizza = 16)
    (h2 : n_salami_diameter = 8) 
    (h3 : n_salami_total = 32) 
    : 
    (32 * (Real.pi * (d_pizza / (2 * n_salami_diameter / 2)) ^ 2)) / (Real.pi * (d_pizza / 2) ^ 2) = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_salami_pizza_fraction_l632_63211


namespace NUMINAMATH_GPT_abc_value_l632_63269

theorem abc_value (a b c : ℂ) (h1 : 2 * a * b + 3 * b = -21)
                   (h2 : 2 * b * c + 3 * c = -21)
                   (h3 : 2 * c * a + 3 * a = -21) :
                   a * b * c = 105.75 := 
sorry

end NUMINAMATH_GPT_abc_value_l632_63269


namespace NUMINAMATH_GPT_cube_edge_length_l632_63289

theorem cube_edge_length {e : ℝ} (h : 12 * e = 108) : e = 9 :=
by sorry

end NUMINAMATH_GPT_cube_edge_length_l632_63289


namespace NUMINAMATH_GPT_anton_has_more_cards_than_ann_l632_63250

-- Define Heike's number of cards
def heike_cards : ℕ := 60

-- Define Anton's number of cards in terms of Heike's cards
def anton_cards (H : ℕ) : ℕ := 3 * H

-- Define Ann's number of cards as equal to Heike's cards
def ann_cards (H : ℕ) : ℕ := H

-- Theorem statement
theorem anton_has_more_cards_than_ann 
  (H : ℕ) (H_equals : H = heike_cards) : 
  anton_cards H - ann_cards H = 120 :=
by
  -- At this point, the actual proof would be inserted.
  sorry

end NUMINAMATH_GPT_anton_has_more_cards_than_ann_l632_63250


namespace NUMINAMATH_GPT_circle_standard_equation_l632_63246

theorem circle_standard_equation (x y : ℝ) (center : ℝ × ℝ) (radius : ℝ) 
  (h_center : center = (2, -1)) (h_radius : radius = 2) :
  (x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2 ↔ (x - 2) ^ 2 + (y + 1) ^ 2 = 4 := by
  sorry

end NUMINAMATH_GPT_circle_standard_equation_l632_63246


namespace NUMINAMATH_GPT_starting_positions_P0_P1024_l632_63249

noncomputable def sequence_fn (x : ℝ) : ℝ := 4 * x / (x^2 + 1)

def find_starting_positions (n : ℕ) : ℕ := 2^n - 2

theorem starting_positions_P0_P1024 :
  ∃ P0 : ℝ, ∀ n : ℕ, P0 = sequence_fn^[n] P0 → P0 = sequence_fn^[1024] P0 ↔ find_starting_positions 1024 = 2^1024 - 2 :=
sorry

end NUMINAMATH_GPT_starting_positions_P0_P1024_l632_63249


namespace NUMINAMATH_GPT_find_a3_l632_63295

variable {α : Type} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) :=
  ∃ r : α, ∀ n, a (n + 1) = a n * r

theorem find_a3 (a : ℕ → α) (h : geometric_sequence a) (h1 : a 0 * a 4 = 16) :
  a 2 = 4 ∨ a 2 = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_l632_63295


namespace NUMINAMATH_GPT_darryl_books_l632_63253

variable (l m d : ℕ)

theorem darryl_books (h1 : l + m + d = 97) (h2 : l = m - 3) (h3 : m = 2 * d) : d = 20 := 
by
  sorry

end NUMINAMATH_GPT_darryl_books_l632_63253


namespace NUMINAMATH_GPT_average_attendance_l632_63262

def monday_attendance := 10
def tuesday_attendance := 15
def wednesday_attendance := 10
def thursday_attendance := 10
def friday_attendance := 10
def total_days := 5

theorem average_attendance :
  (monday_attendance + tuesday_attendance + wednesday_attendance + thursday_attendance + friday_attendance) / total_days = 11 :=
by
  sorry

end NUMINAMATH_GPT_average_attendance_l632_63262


namespace NUMINAMATH_GPT_hiker_walked_distance_first_day_l632_63288

theorem hiker_walked_distance_first_day (h d_1 d_2 d_3 : ℕ) (H₁ : d_1 = 3 * h)
    (H₂ : d_2 = 4 * (h - 1)) (H₃ : d_3 = 30) (H₄ : d_1 + d_2 + d_3 = 68) :
    d_1 = 18 := 
by 
  sorry

end NUMINAMATH_GPT_hiker_walked_distance_first_day_l632_63288


namespace NUMINAMATH_GPT_compute_f_1_g_3_l632_63276

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x + 2

theorem compute_f_1_g_3 : f (1 + g 3) = 7 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_compute_f_1_g_3_l632_63276


namespace NUMINAMATH_GPT_smallest_x_l632_63232

theorem smallest_x : ∃ x : ℕ, x + 6721 ≡ 3458 [MOD 12] ∧ x % 5 = 0 ∧ x = 45 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_l632_63232


namespace NUMINAMATH_GPT_nate_total_distance_l632_63254

def length_field : ℕ := 168
def distance_8s : ℕ := 4 * length_field
def additional_distance : ℕ := 500
def total_distance : ℕ := distance_8s + additional_distance

theorem nate_total_distance : total_distance = 1172 := by
  sorry

end NUMINAMATH_GPT_nate_total_distance_l632_63254


namespace NUMINAMATH_GPT_union_sets_l632_63245

noncomputable def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
noncomputable def C : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem union_sets (A : Set ℝ) (B : Set ℝ) : (A ∪ B = C) := by
  sorry

end NUMINAMATH_GPT_union_sets_l632_63245


namespace NUMINAMATH_GPT_lcm_of_two_numbers_hcf_and_product_l632_63223

theorem lcm_of_two_numbers_hcf_and_product (a b : ℕ) (h_hcf : Nat.gcd a b = 20) (h_prod : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_two_numbers_hcf_and_product_l632_63223


namespace NUMINAMATH_GPT_find_original_number_l632_63266

-- Let x be the original number
def maria_operations (x : ℤ) : Prop :=
  (3 * (x - 3) + 3) / 3 = 10

theorem find_original_number (x : ℤ) (h : maria_operations x) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l632_63266


namespace NUMINAMATH_GPT_determine_value_of_expression_l632_63228

theorem determine_value_of_expression (x y : ℤ) (h : y^2 + 4 * x^2 * y^2 = 40 * x^2 + 817) : 4 * x^2 * y^2 = 3484 :=
sorry

end NUMINAMATH_GPT_determine_value_of_expression_l632_63228


namespace NUMINAMATH_GPT_intersects_line_l632_63243

theorem intersects_line (x y : ℝ) : 
  (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) → ∃ x y : ℝ, (3 * x + 2 * y = 5) ∧ ((x / 3) + (y / 2) = 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_intersects_line_l632_63243


namespace NUMINAMATH_GPT_father_l632_63281

variable (S F : ℕ)

theorem father's_age (h1 : F = 3 * S + 3) (h2 : F + 3 = 2 * (S + 3) + 10) : F = 33 := by
  sorry

end NUMINAMATH_GPT_father_l632_63281


namespace NUMINAMATH_GPT_third_term_of_arithmetic_sequence_l632_63215

theorem third_term_of_arithmetic_sequence (a d : ℝ) (h : a + (a + 4 * d) = 10) : a + 2 * d = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_third_term_of_arithmetic_sequence_l632_63215


namespace NUMINAMATH_GPT_number_of_real_zeros_l632_63299

def f (x : ℝ) : ℝ := x^3 - x^2 - x + 2

theorem number_of_real_zeros : ∃! x : ℝ, f x = 0 := sorry

end NUMINAMATH_GPT_number_of_real_zeros_l632_63299


namespace NUMINAMATH_GPT_even_three_digit_numbers_less_than_600_l632_63274

def count_even_three_digit_numbers : ℕ :=
  let hundreds_choices := 5
  let tens_choices := 6
  let units_choices := 3
  hundreds_choices * tens_choices * units_choices

theorem even_three_digit_numbers_less_than_600 : count_even_three_digit_numbers = 90 := by
  -- sorry ensures that the statement type checks even without the proof.
  sorry

end NUMINAMATH_GPT_even_three_digit_numbers_less_than_600_l632_63274


namespace NUMINAMATH_GPT_vowel_soup_sequences_count_l632_63201

theorem vowel_soup_sequences_count :
  let vowels := 5
  let sequence_length := 6
  vowels ^ sequence_length = 15625 :=
by
  sorry

end NUMINAMATH_GPT_vowel_soup_sequences_count_l632_63201


namespace NUMINAMATH_GPT_five_times_number_equals_hundred_l632_63205

theorem five_times_number_equals_hundred (x : ℝ) (h : 5 * x = 100) : x = 20 :=
sorry

end NUMINAMATH_GPT_five_times_number_equals_hundred_l632_63205


namespace NUMINAMATH_GPT_book_pages_l632_63203

theorem book_pages (n days_n : ℕ) (first_day_pages break_days : ℕ) (common_difference total_pages_read : ℕ) (portion_of_book : ℚ) :
    n = 14 → days_n = 12 → first_day_pages = 10 → break_days = 2 → common_difference = 2 →
    total_pages_read = 252 → portion_of_book = 3/4 →
    (total_pages_read : ℚ) * (4/3) = 336 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_book_pages_l632_63203


namespace NUMINAMATH_GPT_circle_center_transformation_l632_63233

def original_center : ℤ × ℤ := (3, -4)

def reflect_x_axis (p : ℤ × ℤ) : ℤ × ℤ := (p.1, -p.2)

def translate_right (p : ℤ × ℤ) (d : ℤ) : ℤ × ℤ := (p.1 + d, p.2)

def final_center : ℤ × ℤ := (8, 4)

theorem circle_center_transformation :
  translate_right (reflect_x_axis original_center) 5 = final_center :=
by
  sorry

end NUMINAMATH_GPT_circle_center_transformation_l632_63233


namespace NUMINAMATH_GPT_minimum_value_exists_l632_63220

noncomputable def minimized_function (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 5 * y^2 - 8 * x - 6 * y + y^3

theorem minimum_value_exists :
  ∃ (x y : ℝ), minimized_function x y = minimized_function (4/3 - 2 * y/3) y :=
sorry

end NUMINAMATH_GPT_minimum_value_exists_l632_63220


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_in_rhombus_l632_63218

noncomputable def radius_of_inscribed_circle (d₁ d₂ : ℕ) : ℝ :=
  (d₁ * d₂) / (2 * Real.sqrt ((d₁ / 2) ^ 2 + (d₂ / 2) ^ 2))

theorem radius_of_inscribed_circle_in_rhombus :
  radius_of_inscribed_circle 8 18 = 36 / Real.sqrt 97 :=
by
  -- Skip the detailed proof steps
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_in_rhombus_l632_63218


namespace NUMINAMATH_GPT_darma_peanut_consumption_l632_63206

theorem darma_peanut_consumption :
  ∀ (t : ℕ) (rate : ℕ),
  (rate = 20 / 15) →  -- Given the rate of peanut consumption
  (t = 6 * 60) →     -- Given that the total time is 6 minutes
  (rate * t = 480) :=  -- Prove that the total number of peanuts eaten in 6 minutes is 480
by
  intros t rate h_rate h_time
  sorry

end NUMINAMATH_GPT_darma_peanut_consumption_l632_63206


namespace NUMINAMATH_GPT_sum_medians_is_64_l632_63275

noncomputable def median (l: List ℝ) : ℝ := sorry  -- Placeholder for median calculation

open List

/-- Define the scores for players A and B as lists of real numbers -/
def player_a_scores : List ℝ := sorry
def player_b_scores : List ℝ := sorry

/-- Prove that the sum of the medians of the scores lists is 64 -/
theorem sum_medians_is_64 : median player_a_scores + median player_b_scores = 64 := sorry

end NUMINAMATH_GPT_sum_medians_is_64_l632_63275


namespace NUMINAMATH_GPT_jill_total_tax_percentage_l632_63258

theorem jill_total_tax_percentage (spent_clothing_percent spent_food_percent spent_other_percent tax_clothing_percent tax_food_percent tax_other_percent : ℝ)
  (h1 : spent_clothing_percent = 0.5)
  (h2 : spent_food_percent = 0.25)
  (h3 : spent_other_percent = 0.25)
  (h4 : tax_clothing_percent = 0.1)
  (h5 : tax_food_percent = 0)
  (h6 : tax_other_percent = 0.2) :
  ((spent_clothing_percent * tax_clothing_percent + spent_food_percent * tax_food_percent + spent_other_percent * tax_other_percent) * 100) = 10 :=
by
  sorry

end NUMINAMATH_GPT_jill_total_tax_percentage_l632_63258


namespace NUMINAMATH_GPT_ear_muffs_total_l632_63283

theorem ear_muffs_total (a b : ℕ) (h1 : a = 1346) (h2 : b = 6444) : a + b = 7790 :=
by
  sorry

end NUMINAMATH_GPT_ear_muffs_total_l632_63283


namespace NUMINAMATH_GPT_bird_families_flew_to_Asia_l632_63273

-- Variables/Parameters
variable (A : ℕ) (X : ℕ)
axiom hA : A = 47
axiom hX : X = A + 47

-- Theorem Statement
theorem bird_families_flew_to_Asia : X = 94 :=
by
  sorry

end NUMINAMATH_GPT_bird_families_flew_to_Asia_l632_63273


namespace NUMINAMATH_GPT_expression_equivalence_l632_63240

theorem expression_equivalence (a b : ℝ) :
  let P := a + b
  let Q := a - b
  (P + Q)^2 / (P - Q)^2 - (P - Q)^2 / (P + Q)^2 = (a^2 + b^2) * (a^2 - b^2) / (a^2 * b^2) :=
by
  sorry

end NUMINAMATH_GPT_expression_equivalence_l632_63240


namespace NUMINAMATH_GPT_sufficient_not_necessary_implies_a_lt_1_l632_63229

theorem sufficient_not_necessary_implies_a_lt_1 {x a : ℝ} (h : ∀ x : ℝ, x > 1 → x > a ∧ ¬(x > a → x > 1)) : a < 1 :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_implies_a_lt_1_l632_63229


namespace NUMINAMATH_GPT_sums_remainders_equal_l632_63204

-- Definition and conditions
variables (A A' D S S' s s' : ℕ) 
variables (h1 : A > A') 
variables (h2 : A % D = S) 
variables (h3 : A' % D = S') 
variables (h4 : (A + A') % D = s) 
variables (h5 : (S + S') % D = s')

-- Proof statement
theorem sums_remainders_equal : s = s' := 
  sorry

end NUMINAMATH_GPT_sums_remainders_equal_l632_63204


namespace NUMINAMATH_GPT_leah_coins_worth_89_cents_l632_63260

variables (p n d : ℕ)

theorem leah_coins_worth_89_cents (h1 : p + n + d = 15) (h2 : d - 1 = n) : 
  1 * p + 5 * n + 10 * d = 89 := 
sorry

end NUMINAMATH_GPT_leah_coins_worth_89_cents_l632_63260


namespace NUMINAMATH_GPT_find_starting_number_l632_63248

theorem find_starting_number : 
  ∃ x : ℕ, (∃ n : ℕ, n = 21 ∧ (forall k, 1 ≤ k ∧ k ≤ n → x + k*19 ≤ 500) ∧ 
  (forall k, 1 ≤ k ∧ k < n → x + k*19 > 0)) ∧ x = 113 := by {
  sorry
}

end NUMINAMATH_GPT_find_starting_number_l632_63248


namespace NUMINAMATH_GPT_student_papers_count_l632_63210

theorem student_papers_count {F n k: ℝ}
  (h1 : 35 * k = 0.6 * n * F)
  (h2 : 5 * k > 0.5 * F)
  (h3 : 6 * k > 0.5 * F)
  (h4 : 7 * k > 0.5 * F)
  (h5 : 8 * k > 0.5 * F)
  (h6 : 9 * k > 0.5 * F) :
  n = 5 :=
by
  sorry

end NUMINAMATH_GPT_student_papers_count_l632_63210


namespace NUMINAMATH_GPT_lele_dongdong_meet_probability_l632_63265

-- Define the conditions: distances and speeds
def segment_length : ℕ := 500
def n : ℕ := sorry
def d : ℕ := segment_length * n
def lele_speed : ℕ := 18
def dongdong_speed : ℕ := 24

-- Define times to traverse distance d
def t_L : ℚ := d / lele_speed
def t_D : ℚ := d / dongdong_speed

-- Define the time t when they meet
def t : ℚ := d / (lele_speed + dongdong_speed)

-- Define the maximum of t_L and t_D
def max_t_L_t_D : ℚ := max t_L t_D

-- Define the probability they meet on their way
def P_meet : ℚ := t / max_t_L_t_D

-- The theorem to prove the probability of meeting is 97/245
theorem lele_dongdong_meet_probability : P_meet = 97 / 245 :=
sorry

end NUMINAMATH_GPT_lele_dongdong_meet_probability_l632_63265


namespace NUMINAMATH_GPT_blueberry_basket_count_l632_63212

noncomputable def number_of_blueberry_baskets 
    (plums_in_basket : ℕ) 
    (plum_baskets : ℕ) 
    (blueberries_in_basket : ℕ) 
    (total_fruits : ℕ) : ℕ := 
  let total_plums := plum_baskets * plums_in_basket
  let total_blueberries := total_fruits - total_plums
  total_blueberries / blueberries_in_basket

theorem blueberry_basket_count
  (plums_in_basket : ℕ) 
  (plum_baskets : ℕ) 
  (blueberries_in_basket : ℕ) 
  (total_fruits : ℕ)
  (h1 : plums_in_basket = 46)
  (h2 : plum_baskets = 19)
  (h3 : blueberries_in_basket = 170)
  (h4 : total_fruits = 1894) : 
  number_of_blueberry_baskets plums_in_basket plum_baskets blueberries_in_basket total_fruits = 6 := by
  sorry

end NUMINAMATH_GPT_blueberry_basket_count_l632_63212


namespace NUMINAMATH_GPT_general_term_less_than_zero_from_13_l632_63225

-- Define the arithmetic sequence and conditions
def an (n : ℕ) : ℝ := 12 - n

-- Condition: a_3 = 9
def a3_condition : Prop := an 3 = 9

-- Condition: a_9 = 3
def a9_condition : Prop := an 9 = 3

-- Prove the general term of the sequence is 12 - n
theorem general_term (n : ℕ) (h3 : a3_condition) (h9 : a9_condition) :
  an n = 12 - n := 
sorry

-- Prove that the sequence becomes less than 0 starting from the 13th term
theorem less_than_zero_from_13 (h3 : a3_condition) (h9 : a9_condition) :
  ∀ n, n ≥ 13 → an n < 0 :=
sorry

end NUMINAMATH_GPT_general_term_less_than_zero_from_13_l632_63225


namespace NUMINAMATH_GPT_simplify_and_evaluate_l632_63241

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  ((m ^ 2 - 9) / (m ^ 2 - 6 * m + 9) - 3 / (m - 3)) / (m ^ 2 / (m - 3)) = Real.sqrt 2 / 2 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_simplify_and_evaluate_l632_63241


namespace NUMINAMATH_GPT_sales_fifth_month_l632_63222

-- Definitions based on conditions
def sales1 : ℝ := 5420
def sales2 : ℝ := 5660
def sales3 : ℝ := 6200
def sales4 : ℝ := 6350
def sales6 : ℝ := 8270
def average_sale : ℝ := 6400

-- Lean proof problem statement
theorem sales_fifth_month :
  sales1 + sales2 + sales3 + sales4 + sales6 + s = 6 * average_sale  →
  s = 6500 :=
by
  sorry

end NUMINAMATH_GPT_sales_fifth_month_l632_63222


namespace NUMINAMATH_GPT_problem_solution_l632_63221

open Real

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (∃ (C₁ : ℝ), (2 : ℝ)^x + (4 : ℝ)^y = C₁ ∧ C₁ = 2 * sqrt 2) ∧
  (∃ (C₂ : ℝ), 1 / x + 2 / y = C₂ ∧ C₂ = 9) ∧
  (∃ (C₃ : ℝ), x^2 + 4 * y^2 = C₃ ∧ C₃ = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l632_63221


namespace NUMINAMATH_GPT_biff_hourly_earnings_l632_63237

theorem biff_hourly_earnings:
  let ticket_cost := 11
  let drinks_snacks_cost := 3
  let headphones_cost := 16
  let wifi_cost_per_hour := 2
  let bus_ride_hours := 3
  let total_non_wifi_expenses := ticket_cost + drinks_snacks_cost + headphones_cost
  let total_wifi_cost := bus_ride_hours * wifi_cost_per_hour
  let total_expenses := total_non_wifi_expenses + total_wifi_cost
  ∀ (x : ℝ), 3 * x = total_expenses → x = 12 :=
by sorry -- Proof skipped

end NUMINAMATH_GPT_biff_hourly_earnings_l632_63237


namespace NUMINAMATH_GPT_modulus_of_z_equals_two_l632_63271

namespace ComplexProblem

open Complex

-- Definition and conditions of the problem
def satisfies_condition (z : ℂ) : Prop :=
  (z + I) * (1 + I) = 1 - I

-- Statement that needs to be proven
theorem modulus_of_z_equals_two (z : ℂ) (h : satisfies_condition z) : abs z = 2 :=
sorry

end ComplexProblem

end NUMINAMATH_GPT_modulus_of_z_equals_two_l632_63271


namespace NUMINAMATH_GPT_area_percentage_l632_63277

theorem area_percentage (D_S D_R : ℝ) (h : D_R = 0.8 * D_S) : 
  let R_S := D_S / 2
  let R_R := D_R / 2
  let A_S := π * R_S^2
  let A_R := π * R_R^2
  (A_R / A_S) * 100 = 64 := 
by
  sorry

end NUMINAMATH_GPT_area_percentage_l632_63277


namespace NUMINAMATH_GPT_range_of_m_l632_63292

def p (m : ℝ) : Prop := ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0)
def q (m : ℝ) : Prop := ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1

theorem range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1 ∨ ∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0))
  ∧ (¬ (∀ x : ℝ, ¬ (x ^ 2 - 2 * m * x + 1 < 0) → ∃ x y : ℝ, (x ^ 2) / (m - 2) + (y ^ 2) / m = 1)) ↔
  (-1 ≤ m ∧ m ≤ 0) ∨ (1 < m ∧ m < 2) :=
  sorry

end NUMINAMATH_GPT_range_of_m_l632_63292


namespace NUMINAMATH_GPT_xy_square_diff_l632_63244

theorem xy_square_diff (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end NUMINAMATH_GPT_xy_square_diff_l632_63244


namespace NUMINAMATH_GPT_temp_pot_C_to_F_l632_63213

-- Definitions
def boiling_point_C : ℕ := 100
def boiling_point_F : ℕ := 212
def melting_point_C : ℕ := 0
def melting_point_F : ℕ := 32
def temp_pot_C : ℕ := 55
def celsius_to_fahrenheit (c : ℕ) : ℕ := (c * 9 / 5) + 32

-- Theorem to be proved
theorem temp_pot_C_to_F : celsius_to_fahrenheit temp_pot_C = 131 := by
  sorry

end NUMINAMATH_GPT_temp_pot_C_to_F_l632_63213


namespace NUMINAMATH_GPT_find_second_dimension_of_smaller_box_l632_63216

def volume_large_box : ℕ := 12 * 14 * 16
def volume_small_box (x : ℕ) : ℕ := 3 * x * 2
def max_small_boxes : ℕ := 64

theorem find_second_dimension_of_smaller_box (x : ℕ) : volume_large_box = max_small_boxes * volume_small_box x → x = 7 :=
by
  intros h
  unfold volume_large_box at h
  unfold volume_small_box at h
  sorry

end NUMINAMATH_GPT_find_second_dimension_of_smaller_box_l632_63216


namespace NUMINAMATH_GPT_football_daily_practice_hours_l632_63279

-- Define the total practice hours and the days missed.
def total_hours := 30
def days_missed := 1
def days_in_week := 7

-- Calculate the number of days practiced.
def days_practiced := days_in_week - days_missed

-- Define the daily practice hours.
def daily_practice_hours := total_hours / days_practiced

-- State the proposition.
theorem football_daily_practice_hours :
  daily_practice_hours = 5 := sorry

end NUMINAMATH_GPT_football_daily_practice_hours_l632_63279


namespace NUMINAMATH_GPT_geometric_first_term_l632_63263

-- Define the conditions
def is_geometric_series (first_term : ℝ) (r : ℝ) (sum : ℝ) : Prop :=
  sum = first_term / (1 - r)

-- Define the main theorem
theorem geometric_first_term (r : ℝ) (sum : ℝ) (first_term : ℝ) 
  (h_r : r = 1/4) (h_S : sum = 80) (h_sum_formula : is_geometric_series first_term r sum) : 
  first_term = 60 :=
by
  sorry

end NUMINAMATH_GPT_geometric_first_term_l632_63263


namespace NUMINAMATH_GPT_positive_integer_solutions_equation_l632_63236

theorem positive_integer_solutions_equation (x y : ℕ) (positive_x : x > 0) (positive_y : y > 0) :
  x^2 + 6 * x * y - 7 * y^2 = 2009 ↔ (x = 252 ∧ y = 251) ∨ (x = 42 ∧ y = 35) ∨ (x = 42 ∧ y = 1) :=
sorry

end NUMINAMATH_GPT_positive_integer_solutions_equation_l632_63236


namespace NUMINAMATH_GPT_polygon_sides_eq_2023_l632_63264

theorem polygon_sides_eq_2023 (n : ℕ) (h : n - 2 = 2021) : n = 2023 :=
sorry

end NUMINAMATH_GPT_polygon_sides_eq_2023_l632_63264


namespace NUMINAMATH_GPT_minimum_value_expression_l632_63251

theorem minimum_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  (∃ a b c : ℝ, (b > c ∧ c > a) ∧ b ≠ 0 ∧ (a + b) = b - c ∧ (b - c) = c - a ∧ (a - c) = 0 ∧
   ∀ x y z : ℝ, (x = a + b ∧ y = b - c ∧ z = c - a) → 
    (x^2 + y^2 + z^2) / b^2 = 4/3) :=
  sorry

end NUMINAMATH_GPT_minimum_value_expression_l632_63251


namespace NUMINAMATH_GPT_seller_loss_l632_63209

/--
Given:
1. The buyer took goods worth 10 rubles (v_goods : Real := 10).
2. The buyer gave 25 rubles (payment : Real := 25).
3. The seller exchanged 25 rubles of genuine currency with the neighbor (exchange : Real := 25).
4. The seller received 25 rubles in counterfeit currency from the neighbor (counterfeit : Real := 25).
5. The seller gave 15 rubles in genuine currency as change (change : Real := 15).
6. The neighbor discovered the counterfeit and the seller returned 25 rubles to the neighbor (returned : Real := 25).

Prove that the net loss incurred by the seller is 30 rubles.
-/
theorem seller_loss :
  let v_goods := 10
  let payment := 25
  let exchange := 25
  let counterfeit := 25
  let change := 15
  let returned := 25
  (exchange + change) - v_goods = 30 :=
by
  sorry

end NUMINAMATH_GPT_seller_loss_l632_63209


namespace NUMINAMATH_GPT_isosceles_triangle_leg_length_l632_63235

-- Define the necessary condition for the isosceles triangle
def isosceles_triangle (a b c : ℕ) : Prop :=
  b = c ∧ a + b + c = 16 ∧ a = 4

-- State the theorem we want to prove
theorem isosceles_triangle_leg_length :
  ∃ (b c : ℕ), isosceles_triangle 4 b c ∧ b = 6 :=
by
  -- Formal proof will be provided here
  sorry

end NUMINAMATH_GPT_isosceles_triangle_leg_length_l632_63235


namespace NUMINAMATH_GPT_cos_sin_fraction_l632_63282

theorem cos_sin_fraction (α β : ℝ) (h1 : Real.tan (α + β) = 2 / 5) 
                         (h2 : Real.tan (β - Real.pi / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 := 
  sorry

end NUMINAMATH_GPT_cos_sin_fraction_l632_63282


namespace NUMINAMATH_GPT_index_cards_per_pack_l632_63278

-- Definitions of the conditions
def students_per_period := 30
def periods_per_day := 6
def index_cards_per_student := 10
def total_spent := 108
def pack_cost := 3

-- Helper Definitions
def total_students := periods_per_day * students_per_period
def total_index_cards_needed := total_students * index_cards_per_student
def packs_bought := total_spent / pack_cost

-- Theorem to prove
theorem index_cards_per_pack :
  total_index_cards_needed / packs_bought = 50 := by
  sorry

end NUMINAMATH_GPT_index_cards_per_pack_l632_63278


namespace NUMINAMATH_GPT_jesse_gave_pencils_l632_63290

theorem jesse_gave_pencils (initial_pencils : ℕ) (final_pencils : ℕ) (pencils_given : ℕ) :
  initial_pencils = 78 → final_pencils = 34 → pencils_given = initial_pencils - final_pencils → pencils_given = 44 :=
by
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_jesse_gave_pencils_l632_63290


namespace NUMINAMATH_GPT_part1_part2_part3_l632_63297

noncomputable def quadratic_has_real_roots (k : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1^2 - 2*k*x1 + k^2 + k + 1 = 0 ∧ x2^2 - 2*k*x2 + k^2 + k + 1 = 0

theorem part1 (k : ℝ) :
  quadratic_has_real_roots k → k ≤ -1 :=
sorry

theorem part2 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ x1^2 + x2^2 = 10 → k = -2 :=
sorry

theorem part3 (k : ℝ) (x1 x2 : ℝ) :
  quadratic_has_real_roots k ∧ (|x1| + |x2| = 2) → k = -1 :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l632_63297


namespace NUMINAMATH_GPT_maximum_sequence_length_l632_63238

theorem maximum_sequence_length
  (seq : List ℚ) 
  (h1 : ∀ i : ℕ, i + 2 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2)) < 0)
  (h2 : ∀ i : ℕ, i + 3 < seq.length → (seq.get! i + seq.get! (i+1) + seq.get! (i+2) + seq.get! (i+3)) > 0) 
  : seq.length ≤ 5 := 
sorry

end NUMINAMATH_GPT_maximum_sequence_length_l632_63238


namespace NUMINAMATH_GPT_min_value_fraction_8_l632_63255

noncomputable def min_value_of_fraction (x y: ℝ) : Prop :=
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  let parallel := (3 * (y - 1)) = (-2) * x
  x > 0 ∧ y > 0 ∧ parallel → (∀ z, z = (3 / x) + (2 / y) → z ≥ 8)

theorem min_value_fraction_8 (x y : ℝ) (h_posx : x > 0) (h_posy : y > 0) :
  let a : ℝ × ℝ := (3, -2)
  let b : ℝ × ℝ := (x, y - 1)
  let parallel := (3 * (y - 1)) = (-2) * x
  parallel → (3 / x) + (2 / y) ≥ 8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_8_l632_63255


namespace NUMINAMATH_GPT_truncated_cone_sphere_radius_l632_63224

structure TruncatedCone :=
(base_radius_top : ℝ)
(base_radius_bottom : ℝ)

noncomputable def sphere_radius (c : TruncatedCone) : ℝ :=
  if c.base_radius_top = 24 ∧ c.base_radius_bottom = 6 then 12 else 0

theorem truncated_cone_sphere_radius (c : TruncatedCone) (h_radii : c.base_radius_top = 24 ∧ c.base_radius_bottom = 6) :
  sphere_radius c = 12 :=
by
  sorry

end NUMINAMATH_GPT_truncated_cone_sphere_radius_l632_63224


namespace NUMINAMATH_GPT_find_initial_number_l632_63270

theorem find_initial_number (N : ℕ) (k : ℤ) (h : N - 3 = 15 * k) : N = 18 := 
by
  sorry

end NUMINAMATH_GPT_find_initial_number_l632_63270


namespace NUMINAMATH_GPT_exists_four_distinct_natural_numbers_sum_any_three_prime_l632_63293

theorem exists_four_distinct_natural_numbers_sum_any_three_prime :
  ∃ a b c d : ℕ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧ 
  (Prime (a + b + c) ∧ Prime (a + b + d) ∧ Prime (a + c + d) ∧ Prime (b + c + d)) :=
sorry

end NUMINAMATH_GPT_exists_four_distinct_natural_numbers_sum_any_three_prime_l632_63293


namespace NUMINAMATH_GPT_pythagorean_triangle_exists_l632_63214

theorem pythagorean_triangle_exists (a : ℤ) (h : a ≥ 5) : 
  ∃ (b c : ℤ), c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_pythagorean_triangle_exists_l632_63214


namespace NUMINAMATH_GPT_total_reading_materials_l632_63284

theorem total_reading_materials 
  (magazines : ℕ) 
  (newspapers : ℕ) 
  (h_magazines : magazines = 425) 
  (h_newspapers : newspapers = 275) : 
  magazines + newspapers = 700 := 
by 
  sorry

end NUMINAMATH_GPT_total_reading_materials_l632_63284


namespace NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l632_63272

theorem simplify_expression_1 (x y : ℝ) :
  x^2 + 5*y - 4*x^2 - 3*y = -3*x^2 + 2*y :=
sorry

theorem simplify_expression_2 (a b : ℝ) :
  7*a + 3*(a - 3*b) - 2*(b - a) = 12*a - 11*b :=
sorry

end NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l632_63272


namespace NUMINAMATH_GPT_mariela_cards_l632_63231

theorem mariela_cards (cards_after_home : ℕ) (total_cards : ℕ) (cards_in_hospital : ℕ) : 
  cards_after_home = 287 → 
  total_cards = 690 → 
  cards_in_hospital = total_cards - cards_after_home → 
  cards_in_hospital = 403 := 
by 
  intros h1 h2 h3 
  rw [h1, h2] at h3 
  exact h3


end NUMINAMATH_GPT_mariela_cards_l632_63231


namespace NUMINAMATH_GPT_bob_total_profit_l632_63267

/-- Define the cost of each dog --/
def dog_cost : ℝ := 250.0

/-- Define the number of dogs Bob bought --/
def number_of_dogs : ℕ := 2

/-- Define the total cost of the dogs --/
def total_cost_for_dogs : ℝ := dog_cost * number_of_dogs

/-- Define the selling price of each puppy --/
def puppy_selling_price : ℝ := 350.0

/-- Define the number of puppies --/
def number_of_puppies : ℕ := 6

/-- Define the total revenue from selling the puppies --/
def total_revenue_from_puppies : ℝ := puppy_selling_price * number_of_puppies

/-- Define Bob's total profit from selling the puppies --/
def total_profit : ℝ := total_revenue_from_puppies - total_cost_for_dogs

/-- The theorem stating that Bob's total profit is $1600.00 --/
theorem bob_total_profit : total_profit = 1600.0 := 
by
  /- We leave the proof out as we just need the statement -/
  sorry

end NUMINAMATH_GPT_bob_total_profit_l632_63267


namespace NUMINAMATH_GPT_smallest_area_right_triangle_l632_63296

theorem smallest_area_right_triangle (a b : ℕ) (ha : a = 7) (hb : b = 10): 
  ∃ (A : ℕ), A = 35 :=
  by
    have hab := 1/2 * a * b
    sorry

-- Note: "sorry" is used as a placeholder for the proof.

end NUMINAMATH_GPT_smallest_area_right_triangle_l632_63296


namespace NUMINAMATH_GPT_Brazil_wins_10_l632_63256

/-- In the year 3000, the World Hockey Championship will follow new rules: 12 points will be awarded for a win, 
5 points will be deducted for a loss, and no points will be awarded for a draw. If the Brazilian team plays 
38 matches, scores 60 points, and loses at least once, then the number of wins they can achieve is 10. 
List all possible scenarios and justify why there cannot be any others. -/
theorem Brazil_wins_10 (x y z : ℕ) 
    (h1: x + y + z = 38) 
    (h2: 12 * x - 5 * y = 60) 
    (h3: y ≥ 1)
    (h4: z ≥ 0): 
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_Brazil_wins_10_l632_63256


namespace NUMINAMATH_GPT_find_number_l632_63291

theorem find_number (x a_3 a_4 : ℕ) (h1 : x + a_4 = 5574) (h2 : x + a_3 = 557) : x = 5567 :=
  sorry

end NUMINAMATH_GPT_find_number_l632_63291


namespace NUMINAMATH_GPT_first_quadrant_solution_l632_63259

theorem first_quadrant_solution (c : ℝ) :
  (∃ x y : ℝ, x - y = 2 ∧ c * x + y = 3 ∧ 0 < x ∧ 0 < y) ↔ -1 < c ∧ c < 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_first_quadrant_solution_l632_63259


namespace NUMINAMATH_GPT_can_divide_2007_triangles_can_divide_2008_triangles_l632_63257

theorem can_divide_2007_triangles :
  ∃ k : ℕ, 2007 = 9 + 3 * k :=
by
  sorry

theorem can_divide_2008_triangles :
  ∃ m : ℕ, 2008 = 4 + 3 * m :=
by
  sorry

end NUMINAMATH_GPT_can_divide_2007_triangles_can_divide_2008_triangles_l632_63257
