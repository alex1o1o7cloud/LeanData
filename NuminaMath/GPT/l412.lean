import Mathlib

namespace train_speed_correct_l412_41210

def train_length : ℝ := 100
def crossing_time : ℝ := 12
def expected_speed : ℝ := 8.33

theorem train_speed_correct : (train_length / crossing_time) = expected_speed :=
by
  -- Proof goes here
  sorry

end train_speed_correct_l412_41210


namespace ratio_AB_PQ_f_half_func_f_l412_41299

-- Define given conditions
variables {m n : ℝ} -- Lengths of AB and PQ
variables {h : ℝ} -- Height of triangle and rectangle (both are 1)
variables {x : ℝ} -- Variable in the range [0, 1]

-- Same area and height conditions
axiom areas_equal : m / 2 = n
axiom height_equal : h = 1

-- Given the areas are equal and height is 1
theorem ratio_AB_PQ : m / n = 2 :=
by sorry -- Proof of the ratio 

-- Given the specific calculation for x = 1/2
theorem f_half (hx : x = 1 / 2) (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  f (1 / 2) = 3 / 4 :=
by sorry -- Proof of function value at 1/2

-- Prove the expression of the function f(x)
theorem func_f (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  ∀ x, 0 ≤ x → x ≤ 1 → f x = 2 * x - x^2 :=
by sorry -- Proof of the function expression


end ratio_AB_PQ_f_half_func_f_l412_41299


namespace subtraction_of_negatives_l412_41222

theorem subtraction_of_negatives :
  -2 - (-3) = 1 := 
by
  sorry

end subtraction_of_negatives_l412_41222


namespace min_n_constant_term_exists_l412_41228

theorem min_n_constant_term_exists (n : ℕ) (h : 0 < n) :
  (∃ r : ℕ, (2 * n = 3 * r) ∧ n > 0) ↔ n = 3 :=
by
  sorry

end min_n_constant_term_exists_l412_41228


namespace abs_diff_51st_terms_correct_l412_41257

-- Definition of initial conditions for sequences A and C
def seqA_first_term : ℤ := 40
def seqA_common_difference : ℤ := 8

def seqC_first_term : ℤ := 40
def seqC_common_difference : ℤ := -5

-- Definition of the nth term function for an arithmetic sequence
def nth_term (a₁ d n : ℤ) : ℤ := a₁ + d * (n - 1)

-- 51st term of sequence A
def a_51 : ℤ := nth_term seqA_first_term seqA_common_difference 51

-- 51st term of sequence C
def c_51 : ℤ := nth_term seqC_first_term seqC_common_difference 51

-- Absolute value of the difference
def abs_diff_51st_terms : ℤ := Int.natAbs (a_51 - c_51)

-- The theorem to be proved
theorem abs_diff_51st_terms_correct : abs_diff_51st_terms = 650 := by
  sorry

end abs_diff_51st_terms_correct_l412_41257


namespace decimal_equivalent_of_one_quarter_l412_41275

theorem decimal_equivalent_of_one_quarter:
  ( (1:ℚ) / (4:ℚ) )^1 = 0.25 := 
sorry

end decimal_equivalent_of_one_quarter_l412_41275


namespace cos_17_pi_over_6_l412_41212

noncomputable def rad_to_deg (r : ℝ) : ℝ := r * 180 / Real.pi

theorem cos_17_pi_over_6 : Real.cos (17 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_17_pi_over_6_l412_41212


namespace additional_hours_q_l412_41225

variable (P Q : ℝ)

theorem additional_hours_q (h1 : P = 1.5 * Q) 
                           (h2 : P = Q + 8) 
                           (h3 : 480 / P = 20):
  (480 / Q) - (480 / P) = 10 :=
by
  sorry

end additional_hours_q_l412_41225


namespace three_digit_numbers_divisible_by_5_l412_41269

theorem three_digit_numbers_divisible_by_5 : ∃ n : ℕ, n = 181 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 ∧ x % 5 = 0) → ∃ k : ℕ, x = 100 + k * 5 ∧ k < n := sorry

end three_digit_numbers_divisible_by_5_l412_41269


namespace seeds_per_can_l412_41215

theorem seeds_per_can (total_seeds : ℕ) (num_cans : ℕ) (h1 : total_seeds = 54) (h2 : num_cans = 9) : total_seeds / num_cans = 6 :=
by {
  sorry
}

end seeds_per_can_l412_41215


namespace circles_intersect_twice_l412_41281

noncomputable def circle1 (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 9

noncomputable def circle2 (x y : ℝ) : Prop :=
  x^2 + (y - 1.5)^2 = 9 / 4

theorem circles_intersect_twice : 
  (∃ (p : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2) ∧ 
  (∀ (p q : ℝ × ℝ), circle1 p.1 p.2 ∧ circle2 p.1 p.2 ∧ circle1 q.1 q.2 ∧ circle2 q.1 q.2 → (p = q ∨ p ≠ q)) →
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧
    circle1 p1.1 p1.2 ∧ circle2 p1.1 p1.2 ∧
    circle1 p2.1 p2.2 ∧ circle2 p2.1 p2.2 := 
by {
  sorry
}

end circles_intersect_twice_l412_41281


namespace part_a_l412_41289

theorem part_a (n : ℕ) (hn : 0 < n) : 
  ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^(n-1) + y^n = z^(n+1) :=
sorry

end part_a_l412_41289


namespace find_principal_and_rate_l412_41245

variables (P R : ℝ)

theorem find_principal_and_rate
  (h1 : 20 = P * R * 2 / 100)
  (h2 : 22 = P * ((1 + R / 100) ^ 2 - 1)) :
  P = 50 ∧ R = 20 :=
by
  sorry

end find_principal_and_rate_l412_41245


namespace find_base_l412_41290
-- Import the necessary library

-- Define the conditions and the result
theorem find_base (x y b : ℕ) (h1 : x - y = 9) (h2 : x = 9) (h3 : b^x * 4^y = 19683) : b = 3 :=
by
  sorry

end find_base_l412_41290


namespace coefficients_balance_l412_41264

noncomputable def num_positive_coeffs (n : ℕ) : ℕ :=
  n + 1

noncomputable def num_negative_coeffs (n : ℕ) : ℕ :=
  n + 1

theorem coefficients_balance (n : ℕ) (h_odd: Odd n) (x : ℝ) :
  num_positive_coeffs n = num_negative_coeffs n :=
by
  sorry

end coefficients_balance_l412_41264


namespace total_people_at_fair_l412_41209

theorem total_people_at_fair (num_children : ℕ) (num_adults : ℕ) 
  (children_attended : num_children = 700) 
  (adults_attended : num_adults = 1500) : 
  num_children + num_adults = 2200 := by
  sorry

end total_people_at_fair_l412_41209


namespace number_of_relatively_prime_to_18_l412_41296

theorem number_of_relatively_prime_to_18 : 
  ∃ N : ℕ, N = 30 ∧ ∀ n : ℕ, 10 ≤ n ∧ n ≤ 99 → Nat.gcd n 18 = 1 ↔ false :=
by
  sorry

end number_of_relatively_prime_to_18_l412_41296


namespace alex_silver_tokens_l412_41258

theorem alex_silver_tokens :
  let R : Int -> Int -> Int := fun x y => 100 - 3 * x + 2 * y
  let B : Int -> Int -> Int := fun x y => 50 + 2 * x - 4 * y
  let x := 61
  let y := 42
  100 - 3 * x + 2 * y < 3 → 50 + 2 * x - 4 * y < 4 → x + y = 103 :=
by
  intro hR hB
  sorry

end alex_silver_tokens_l412_41258


namespace find_inverse_sum_l412_41235

theorem find_inverse_sum (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + a = b / a + b) : 1 / a + 1 / b = -1 :=
sorry

end find_inverse_sum_l412_41235


namespace angle_B_possible_values_l412_41286

theorem angle_B_possible_values
  (a b : ℝ) (A B : ℝ)
  (h_a : a = 2)
  (h_b : b = 2 * Real.sqrt 3)
  (h_A : A = Real.pi / 6) 
  (h_A_range : (0 : ℝ) < A ∧ A < Real.pi) :
  B = Real.pi / 3 ∨ B = 2 * Real.pi / 3 :=
  sorry

end angle_B_possible_values_l412_41286


namespace equal_distribution_l412_41229

theorem equal_distribution (total_cookies bags : ℕ) (h_total : total_cookies = 14) (h_bags : bags = 7) : total_cookies / bags = 2 := by
  sorry

end equal_distribution_l412_41229


namespace series_value_l412_41267

noncomputable def sum_series (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) : ℝ :=
∑' n : ℕ, (if h : n > 0 then
             1 / (((n - 1) * c - (n - 2) * b) * (n * c - (n - 1) * a))
           else 
             0)

theorem series_value (a b c : ℝ) (h_positivity : 0 < c ∧ 0 < b ∧ 0 < a) (h_order : a > b ∧ b > c) :
  sum_series a b c h_positivity h_order = 1 / ((c - a) * b) :=
by
  sorry

end series_value_l412_41267


namespace number_of_votes_for_winner_l412_41241

-- Define the conditions
def total_votes : ℝ := 1000
def winner_percentage : ℝ := 0.55
def margin_of_victory : ℝ := 100

-- The statement to prove
theorem number_of_votes_for_winner :
  0.55 * total_votes = 550 :=
by
  -- We are supposed to provide the proof but it's skipped here
  sorry

end number_of_votes_for_winner_l412_41241


namespace child_ticket_cost_l412_41252

def cost_of_adult_ticket : ℕ := 22
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 2
def total_family_cost : ℕ := 58
def cost_of_child_ticket : ℕ := 7

theorem child_ticket_cost :
  2 * cost_of_adult_ticket + number_of_children * cost_of_child_ticket = total_family_cost :=
by
  sorry

end child_ticket_cost_l412_41252


namespace range_of_a_l412_41255

open Set

variable (a : ℝ)

noncomputable def I := univ ℝ
noncomputable def A := {x : ℝ | x ≤ a + 1}
noncomputable def B := {x : ℝ | x ≥ 1}
noncomputable def complement_B := {x : ℝ | x < 1}

theorem range_of_a (h : A a ⊆ complement_B) : a < 0 := sorry

end range_of_a_l412_41255


namespace angle_D_measure_l412_41268

theorem angle_D_measure (B C E F D : ℝ) 
  (h₁ : B = 120)
  (h₂ : B + C = 180)
  (h₃ : E = 45)
  (h₄ : F = C) 
  (h₅ : D + E + F = 180) :
  D = 75 := sorry

end angle_D_measure_l412_41268


namespace minimum_value_of_fraction_l412_41237

variable {a b : ℝ}

theorem minimum_value_of_fraction (h1 : a > b) (h2 : a * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ ∀ x > b, a * x = 1 -> 
  (x - b + 2 / (x - b) ≥ c) :=
by
  sorry

end minimum_value_of_fraction_l412_41237


namespace solution_to_inequality_l412_41200

-- Define the combination function C(n, k)
def combination (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the permutation function A(n, k)
def permutation (n k : ℕ) : ℕ :=
  n.factorial / (n - k).factorial

-- State the final theorem
theorem solution_to_inequality : 
  ∀ x : ℕ, (combination 5 x + permutation x 3 < 30) ↔ (x = 3 ∨ x = 4) :=
by
  -- The actual proof is not required as per the instructions
  sorry

end solution_to_inequality_l412_41200


namespace a3_equals_1_div_12_l412_41292

-- Definition of the sequence
def seq (n : Nat) : Rat :=
  1 / (n * (n + 1))

-- Assertion to be proved
theorem a3_equals_1_div_12 : seq 3 = 1 / 12 := 
sorry

end a3_equals_1_div_12_l412_41292


namespace condition_a_gt_1_iff_a_gt_0_l412_41261

theorem condition_a_gt_1_iff_a_gt_0 : ∀ (a : ℝ), (a > 1) ↔ (a > 0) :=
by 
  sorry

end condition_a_gt_1_iff_a_gt_0_l412_41261


namespace circle_equation1_circle_equation2_l412_41242

-- Definitions for the first question
def center1 : (ℝ × ℝ) := (2, -2)
def pointP : (ℝ × ℝ) := (6, 3)

-- Definitions for the second question
def pointA : (ℝ × ℝ) := (-4, -5)
def pointB : (ℝ × ℝ) := (6, -1)

-- Theorems we need to prove
theorem circle_equation1 : (x - 2)^2 + (y + 2)^2 = 41 :=
sorry

theorem circle_equation2 : (x - 1)^2 + (y + 3)^2 = 29 :=
sorry

end circle_equation1_circle_equation2_l412_41242


namespace committee_probability_l412_41271

theorem committee_probability :
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  specific_committees / total_committees = 64 / 211 := 
by
  let total_committees := Nat.choose 30 6
  let boys_choose := Nat.choose 18 3
  let girls_choose := Nat.choose 12 3
  let specific_committees := boys_choose * girls_choose
  have h_total_committees : total_committees = 593775 := by sorry
  have h_boys_choose : boys_choose = 816 := by sorry
  have h_girls_choose : girls_choose = 220 := by sorry
  have h_specific_committees : specific_committees = 179520 := by sorry
  have h_probability : specific_committees / total_committees = 64 / 211 := by sorry
  exact h_probability

end committee_probability_l412_41271


namespace cherry_tomatoes_weight_l412_41250

def kilogram_to_grams (kg : ℕ) : ℕ := kg * 1000

theorem cherry_tomatoes_weight (kg_tomatoes : ℕ) (extra_tomatoes_g : ℕ) : kg_tomatoes = 2 → extra_tomatoes_g = 560 → kilogram_to_grams kg_tomatoes + extra_tomatoes_g = 2560 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end cherry_tomatoes_weight_l412_41250


namespace prime_factorization_of_expression_l412_41201

theorem prime_factorization_of_expression (p n : ℕ) (hp : Nat.Prime p) (hdiv : p^2 ∣ 2^(p-1) - 1) : 
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c) ∧ 
  a ∣ (p-1) ∧ b ∣ (p! + 2^n) ∧ c ∣ (p! + 2^n) := 
sorry

end prime_factorization_of_expression_l412_41201


namespace krishna_fraction_wins_l412_41273

theorem krishna_fraction_wins (matches_total : ℕ) (callum_points : ℕ) (points_per_win : ℕ) (callum_wins : ℕ) :
  matches_total = 8 → callum_points = 20 → points_per_win = 10 → callum_wins = callum_points / points_per_win →
  (matches_total - callum_wins) / matches_total = 3 / 4 :=
by
  intros h1 h2 h3 h4
  sorry

end krishna_fraction_wins_l412_41273


namespace A_neg10_3_eq_neg1320_l412_41288

noncomputable def A (x : ℝ) (m : ℕ) : ℝ :=
  if m = 0 then 1 else x * A (x - 1) (m - 1)

theorem A_neg10_3_eq_neg1320 : A (-10) 3 = -1320 := 
by
  sorry

end A_neg10_3_eq_neg1320_l412_41288


namespace translate_triangle_l412_41226

theorem translate_triangle (A B C A' : (ℝ × ℝ)) (hx_A : A = (2, 1)) (hx_B : B = (4, 3)) 
  (hx_C : C = (0, 2)) (hx_A' : A' = (-1, 5)) : 
  ∃ C' : (ℝ × ℝ), C' = (-3, 6) :=
by 
  sorry

end translate_triangle_l412_41226


namespace calculate_expression_l412_41265

theorem calculate_expression :
  ((2000000000000 - 1234567890123) * 3 = 2296296329631) :=
by 
  sorry

end calculate_expression_l412_41265


namespace find_seven_m_squared_minus_one_l412_41244

theorem find_seven_m_squared_minus_one (m : ℝ)
  (h1 : ∃ x₁, 5 * m + 3 * x₁ = 1 + x₁)
  (h2 : ∃ x₂, 2 * x₂ + m = 3 * m)
  (h3 : ∀ x₁ x₂, (5 * m + 3 * x₁ = 1 + x₁) → (2 * x₂ + m = 3 * m) → x₁ = x₂ + 2) :
  7 * m^2 - 1 = 2 / 7 :=
by
  let m := -3/7
  sorry

end find_seven_m_squared_minus_one_l412_41244


namespace ratio_of_democrats_l412_41202

variable (F M D_F D_M : ℕ)

theorem ratio_of_democrats (h1 : F + M = 750)
    (h2 : D_F = 1 / 2 * F)
    (h3 : D_F = 125)
    (h4 : D_M = 1 / 4 * M) :
    (D_F + D_M) / 750 = 1 / 3 :=
sorry

end ratio_of_democrats_l412_41202


namespace abs_neg_seven_l412_41294

theorem abs_neg_seven : |(-7 : ℤ)| = 7 := by
  sorry

end abs_neg_seven_l412_41294


namespace find_density_of_gold_l412_41223

theorem find_density_of_gold
  (side_length : ℝ)
  (gold_cost_per_gram : ℝ)
  (sale_factor : ℝ)
  (profit : ℝ)
  (density_of_gold : ℝ) :
  side_length = 6 →
  gold_cost_per_gram = 60 →
  sale_factor = 1.5 →
  profit = 123120 →
  density_of_gold = 19 :=
sorry

end find_density_of_gold_l412_41223


namespace subtraction_result_l412_41279

noncomputable def division_value : ℝ := 1002 / 20.04

theorem subtraction_result : 2500 - division_value = 2450.0499 :=
by
  have division_eq : division_value = 49.9501 := by sorry
  rw [division_eq]
  norm_num

end subtraction_result_l412_41279


namespace correct_result_l412_41227

theorem correct_result (x : ℕ) (h: (325 - x) * 5 = 1500) : 325 - x * 5 = 200 := 
by
  -- placeholder for proof
  sorry

end correct_result_l412_41227


namespace nat_perfect_square_l412_41205

theorem nat_perfect_square (a b : ℕ) (h : ∃ k : ℕ, a^2 + b^2 + a = k * a * b) : ∃ m : ℕ, a = m * m := by
  sorry

end nat_perfect_square_l412_41205


namespace alchemerion_age_problem_l412_41248

theorem alchemerion_age_problem 
  (A S F : ℕ)
  (h1 : A = 3 * S)
  (h2 : F = 2 * A + 40)
  (h3 : A = 360) :
  A + S + F = 1240 :=
by 
  sorry

end alchemerion_age_problem_l412_41248


namespace bronze_status_families_count_l412_41259

theorem bronze_status_families_count :
  ∃ B : ℕ, (B * 25) = (700 - (7 * 50 + 1 * 100)) ∧ B = 10 := 
sorry

end bronze_status_families_count_l412_41259


namespace largest_of_four_numbers_l412_41256

theorem largest_of_four_numbers 
  (a b c d : ℝ) 
  (h1 : a + 5 = b^2 - 1) 
  (h2 : a + 5 = c^2 + 3) 
  (h3 : a + 5 = d - 4) 
  : d > max (max a b) c :=
sorry

end largest_of_four_numbers_l412_41256


namespace remove_parentheses_l412_41251

variable (a b c : ℝ)

theorem remove_parentheses :
  -3 * a - (2 * b - c) = -3 * a - 2 * b + c :=
by
  sorry

end remove_parentheses_l412_41251


namespace box_height_correct_l412_41216

noncomputable def box_height : ℕ :=
  8

theorem box_height_correct (box_width box_length block_height block_width block_length : ℕ) (num_blocks : ℕ) :
  box_width = 10 ∧
  box_length = 12 ∧
  block_height = 3 ∧
  block_width = 2 ∧
  block_length = 4 ∧
  num_blocks = 40 →
  (num_blocks * block_height * block_width * block_length) /
  (box_width * box_length) = box_height :=
  by
  sorry

end box_height_correct_l412_41216


namespace eggs_left_in_jar_l412_41221

variable (initial_eggs : ℝ) (removed_eggs : ℝ)

theorem eggs_left_in_jar (h1 : initial_eggs = 35.3) (h2 : removed_eggs = 4.5) :
  initial_eggs - removed_eggs = 30.8 :=
by
  sorry

end eggs_left_in_jar_l412_41221


namespace scientific_notation_of_0_0000007_l412_41247

theorem scientific_notation_of_0_0000007 :
  0.0000007 = 7 * 10 ^ (-7) :=
  by
  sorry

end scientific_notation_of_0_0000007_l412_41247


namespace power_sum_l412_41211

theorem power_sum : 2^3 + 2^2 + 2^1 = 14 := by
  sorry

end power_sum_l412_41211


namespace employees_after_reduction_l412_41278

def reduction (original : Float) (percent : Float) : Float :=
  original - (percent * original)

theorem employees_after_reduction :
  reduction 243.75 0.20 = 195 := by
  sorry

end employees_after_reduction_l412_41278


namespace hours_buses_leave_each_day_l412_41266

theorem hours_buses_leave_each_day
  (num_buses : ℕ)
  (num_days : ℕ)
  (buses_per_half_hour : ℕ)
  (h1 : num_buses = 120)
  (h2 : num_days = 5)
  (h3 : buses_per_half_hour = 2) :
  (num_buses / num_days) / buses_per_half_hour = 12 :=
by
  sorry

end hours_buses_leave_each_day_l412_41266


namespace find_a_l412_41231

def F (a b c : ℤ) : ℤ := a * b^2 + c

theorem find_a (a : ℤ) (h : F a 3 (-1) = F a 5 (-3)) : a = 1 / 8 := by
  sorry

end find_a_l412_41231


namespace haley_lives_gained_l412_41263

-- Define the given conditions
def initial_lives : ℕ := 14
def lives_lost : ℕ := 4
def total_lives_after_gain : ℕ := 46

-- Define the goal: How many lives did Haley gain in the next level?
theorem haley_lives_gained : (total_lives_after_gain = initial_lives - lives_lost + lives_gained) → lives_gained = 36 :=
by
  intro h
  sorry

end haley_lives_gained_l412_41263


namespace prec_property_l412_41280

noncomputable def prec (a b : ℕ) : Prop :=
  sorry -- The construction of the relation from the problem

axiom prec_total : ∀ a b : ℕ, (prec a b ∨ prec b a ∨ a = b)
axiom prec_trans : ∀ a b c : ℕ, (prec a b ∧ prec b c) → prec a c

theorem prec_property : ∀ a b c : ℕ, (prec a b ∧ prec b c) → 2 * b ≠ a + c :=
by
  sorry

end prec_property_l412_41280


namespace find_a4_l412_41276

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (T_7 : ℝ)

-- Conditions
axiom geom_seq (n : ℕ) : a (n + 1) = q * a n
axiom common_ratio_ne_one : q ≠ 1
axiom product_first_seven_terms : (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) = 128

-- Goal
theorem find_a4 : a 4 = 2 :=
sorry

end find_a4_l412_41276


namespace evaluate_nested_operation_l412_41297

def operation (a b c : ℕ) : ℕ := (a + b) / c

theorem evaluate_nested_operation : operation (operation 72 36 108) (operation 4 2 6) (operation 12 6 18) = 2 := by
  -- Here we assume all operations are valid (c ≠ 0 for each case)
  sorry

end evaluate_nested_operation_l412_41297


namespace value_of_a_l412_41240

def P : Set ℝ := { x | x^2 ≤ 4 }
def M (a : ℝ) : Set ℝ := { a }

theorem value_of_a (a : ℝ) (h : P ∪ {a} = P) : a ∈ { x : ℝ | -2 ≤ x ∧ x ≤ 2 } := by
  sorry

end value_of_a_l412_41240


namespace find_x_value_l412_41238

-- Define the conditions and the proof problem as Lean 4 statement
theorem find_x_value 
  (k : ℚ)
  (h1 : ∀ (x y : ℚ), (2 * x - 3) / (2 * y + 10) = k)
  (h2 : (2 * 4 - 3) / (2 * 5 + 10) = k)
  : (∃ x : ℚ, (2 * x - 3) / (2 * 10 + 10) = k) ↔ x = 5.25 :=
by
  sorry

end find_x_value_l412_41238


namespace waiter_tables_l412_41272

theorem waiter_tables (init_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) (remaining_customers : ℕ) (num_tables : ℕ) :
  init_customers = 44 →
  left_customers = 12 →
  people_per_table = 8 →
  remaining_customers = init_customers - left_customers →
  num_tables = remaining_customers / people_per_table →
  num_tables = 4 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end waiter_tables_l412_41272


namespace abs_diff_of_solutions_l412_41220

theorem abs_diff_of_solutions (h : ∀ x : ℝ, x^2 - 7 * x + 10 = 0 → x = 2 ∨ x = 5) :
  |(2 - 5 : ℝ)| = 3 :=
by sorry

end abs_diff_of_solutions_l412_41220


namespace train_time_to_B_l412_41277

theorem train_time_to_B (T : ℝ) (M : ℝ) :
  (∃ (D : ℝ), (T + 5) * (D + M) / T = 6 * M ∧ 2 * D = 5 * M) → T = 7 :=
by
  sorry

end train_time_to_B_l412_41277


namespace result_when_decreased_by_5_and_divided_by_7_l412_41282

theorem result_when_decreased_by_5_and_divided_by_7 (x y : ℤ)
  (h1 : (x - 5) / 7 = y)
  (h2 : (x - 6) / 8 = 6) :
  y = 7 :=
by
  sorry

end result_when_decreased_by_5_and_divided_by_7_l412_41282


namespace percentage_reduction_in_women_l412_41291

theorem percentage_reduction_in_women
    (total_people : Nat) (men_in_office : Nat) (women_in_office : Nat)
    (men_in_meeting : Nat) (women_in_meeting : Nat)
    (even_men_women : men_in_office = women_in_office)
    (total_people_condition : total_people = men_in_office + women_in_office)
    (meeting_condition : total_people = 60)
    (men_meeting_condition : men_in_meeting = 4)
    (women_meeting_condition : women_in_meeting = 6) :
    ((women_in_meeting * 100) / women_in_office) = 20 :=
by
  sorry

end percentage_reduction_in_women_l412_41291


namespace constant_term_proof_l412_41239

noncomputable def constant_term_in_binomial_expansion (c : ℚ) (x : ℚ) : ℚ :=
  if h : (c = (2 : ℚ) - (1 / (8 * x^3))∧ x ≠ 0) then 
    28
  else 
    0

theorem constant_term_proof : 
  constant_term_in_binomial_expansion ((2 : ℚ) - (1 / (8 * (1 : ℚ)^3))) 1 = 28 := 
by
  sorry

end constant_term_proof_l412_41239


namespace complex_division_example_l412_41206

theorem complex_division_example : (2 : ℂ) / (I * (3 - I)) = (1 - 3 * I) / 5 := 
by {
  sorry
}

end complex_division_example_l412_41206


namespace trig_identity_l412_41230

noncomputable def trig_expr := 
  4.34 * (Real.cos (28 * Real.pi / 180) * Real.cos (56 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) + 
  (Real.cos (2 * Real.pi / 180) * Real.cos (4 * Real.pi / 180) / Real.sin (28 * Real.pi / 180))

theorem trig_identity : 
  trig_expr = (Real.sqrt 3 * Real.sin (38 * Real.pi / 180)) / (4 * Real.sin (2 * Real.pi / 180) * Real.sin (28 * Real.pi / 180)) :=
by 
  sorry

end trig_identity_l412_41230


namespace find_a10_l412_41295

-- Define the arithmetic sequence with its common difference and initial term
axiom a_seq : ℕ → ℝ
axiom a1 : ℝ
axiom d : ℝ

-- Conditions
axiom a3 : a_seq 3 = a1 + 2 * d
axiom a5_a8 : a_seq 5 + a_seq 8 = 15

-- Theorem statement
theorem find_a10 : a_seq 10 = 13 :=
by sorry

end find_a10_l412_41295


namespace horizontal_asymptote_l412_41213

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 7 * x^3 + 10 * x^2 + 6 * x + 4) / (4 * x^4 + 3 * x^3 + 9 * x^2 + 4 * x + 2)

theorem horizontal_asymptote :
  ∃ L : ℝ, (∀ ε > 0, ∃ M > 0, ∀ x > M, |rational_function x - L| < ε) → L = 15 / 4 :=
by
  sorry

end horizontal_asymptote_l412_41213


namespace factoring_expression_l412_41270

theorem factoring_expression (a b c x y : ℝ) :
  -a * (x - y) - b * (y - x) + c * (x - y) = -(x - y) * (a + b - c) :=
by
  sorry

end factoring_expression_l412_41270


namespace least_pos_int_satisfies_conditions_l412_41232

theorem least_pos_int_satisfies_conditions :
  ∃ x : ℕ, x > 0 ∧ 
  (x % 3 = 2) ∧ 
  (x % 4 = 3) ∧ 
  (x % 5 = 4) ∧ 
  (x % 7 = 6) ∧ 
  x = 419 :=
by
  sorry

end least_pos_int_satisfies_conditions_l412_41232


namespace sum_of_three_numbers_l412_41207

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 138) 
  (h2 : a * b + b * c + c * a = 131) : 
  a + b + c = 20 :=
sorry

end sum_of_three_numbers_l412_41207


namespace transformation_power_of_two_l412_41204

theorem transformation_power_of_two (n : ℕ) (h : n ≥ 3) :
  ∃ s : ℕ, 2 ^ s ≥ n :=
by sorry

end transformation_power_of_two_l412_41204


namespace circle_probability_l412_41274

noncomputable def problem_statement : Prop :=
  let outer_radius := 3
  let inner_radius := 1
  let pivotal_radius := 2
  let outer_area := Real.pi * outer_radius ^ 2
  let inner_area := Real.pi * pivotal_radius ^ 2
  let probability := inner_area / outer_area
  probability = 4 / 9

theorem circle_probability : problem_statement := sorry

end circle_probability_l412_41274


namespace find_x_l412_41217

theorem find_x (x : ℤ) (h : 3 * x + 36 = 48) : x = 4 :=
by
  -- proof is not required, so we insert sorry
  sorry

end find_x_l412_41217


namespace dave_winfield_home_runs_correct_l412_41208

def dave_winfield_home_runs (W : ℕ) : Prop :=
  755 = 2 * W - 175

theorem dave_winfield_home_runs_correct : dave_winfield_home_runs 465 :=
by
  -- The proof is omitted as requested
  sorry

end dave_winfield_home_runs_correct_l412_41208


namespace find_k_l412_41246

def triangle_sides (a b c : ℕ) : Prop :=
a < b + c ∧ b < a + c ∧ c < a + b

def is_right_triangle (a b c : ℕ) : Prop :=
a * a + b * b = c * c

def angle_bisector_length (a b c l : ℕ) : Prop :=
∃ k : ℚ, l = k * Real.sqrt 2 ∧ k = 5 / 2

theorem find_k :
  ∀ (AB BC AC BD : ℕ),
  triangle_sides AB BC AC ∧ is_right_triangle AB BC AC ∧
  AB = 5 ∧ BC = 12 ∧ AC = 13 ∧ angle_bisector_length 5 12 13 BD →
  ∃ k : ℚ, BD = k * Real.sqrt 2 ∧ k = 5 / 2 := by
  sorry

end find_k_l412_41246


namespace least_positive_t_l412_41236

theorem least_positive_t (t : ℕ) (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : π / 10 < α ∧ α ≤ π / 6) 
  (h3 : (3 * α)^2 = α * (π - 5 * α)) :
  t = 27 :=
by
  have hα : α = π / 14 := 
    by
      sorry
  sorry

end least_positive_t_l412_41236


namespace point_on_angle_bisector_l412_41234

theorem point_on_angle_bisector (a b : ℝ) (h : ∃ p : ℝ, (a, b) = (p, -p)) : a = -b :=
sorry

end point_on_angle_bisector_l412_41234


namespace ratio_AB_to_AD_l412_41243

/-
In rectangle ABCD, 30% of its area overlaps with square EFGH. Square EFGH shares 40% of its area with rectangle ABCD. If AD equals one-tenth of the side length of square EFGH, what is AB/AD?
-/

theorem ratio_AB_to_AD (s x y : ℝ)
  (h1 : 0.3 * (x * y) = 0.4 * s^2)
  (h2 : y = s / 10):
  (x / y) = 400 / 3 :=
by
  sorry

end ratio_AB_to_AD_l412_41243


namespace find_multiple_l412_41214

theorem find_multiple (n m : ℕ) (h_n : n = 5) (h_eq : m * n - 15 = 2 * n + 10) : m = 7 :=
by
  sorry

end find_multiple_l412_41214


namespace value_of_z_l412_41219

theorem value_of_z (x y z : ℝ) (h1 : x + y = 6) (h2 : z^2 = x * y - 9) : z = 0 :=
by
  sorry

end value_of_z_l412_41219


namespace proof_value_of_expression_l412_41203

theorem proof_value_of_expression (a b c d m : ℝ) 
  (h1: a + b = 0)
  (h2: c * d = 1)
  (h3: |m| = 4) : 
  m + c * d + (a + b) / m = 5 ∨ m + c * d + (a + b) / m = -3 := by
  sorry

end proof_value_of_expression_l412_41203


namespace calculate_non_defective_m3_percentage_l412_41298

def percentage_non_defective_m3 : ℝ := 93

theorem calculate_non_defective_m3_percentage 
  (P : ℝ) -- Total number of products
  (P_pos : 0 < P) -- Total number of products is positive
  (percentage_m1 : ℝ := 0.40)
  (percentage_m2 : ℝ := 0.30)
  (percentage_m3 : ℝ := 0.30)
  (defective_m1 : ℝ := 0.03)
  (defective_m2 : ℝ := 0.01)
  (total_defective : ℝ := 0.036) :
  percentage_non_defective_m3 = 93 :=
by sorry -- The actual proof is omitted

end calculate_non_defective_m3_percentage_l412_41298


namespace non_officers_count_l412_41224

theorem non_officers_count 
    (avg_salary_employees : ℝ) 
    (avg_salary_officers : ℝ) 
    (avg_salary_non_officers : ℝ) 
    (num_officers : ℕ) : 
    avg_salary_employees = 120 ∧ avg_salary_officers = 470 ∧ avg_salary_non_officers = 110 ∧ num_officers = 15 → 
    ∃ N : ℕ, N = 525 ∧ 
    (num_officers * avg_salary_officers + N * avg_salary_non_officers) / (num_officers + N) = avg_salary_employees := 
by 
    sorry

end non_officers_count_l412_41224


namespace correct_truth_values_l412_41254

open Real

def proposition_p : Prop := ∀ (a : ℝ), 0 < a → a^2 ≠ 0

def converse_p : Prop := ∀ (a : ℝ), a^2 ≠ 0 → 0 < a

def inverse_p : Prop := ∀ (a : ℝ), ¬(0 < a) → a^2 = 0

def contrapositive_p : Prop := ∀ (a : ℝ), a^2 = 0 → ¬(0 < a)

def negation_p : Prop := ∃ (a : ℝ), 0 < a ∧ a^2 = 0

theorem correct_truth_values : 
  (converse_p = False) ∧ 
  (inverse_p = False) ∧ 
  (contrapositive_p = True) ∧ 
  (negation_p = False) := by
  sorry

end correct_truth_values_l412_41254


namespace find_a_l412_41293

theorem find_a (a t : ℝ) 
    (h1 : (a + t) / 2 = 2020) 
    (h2 : t / 2 = 11) : 
    a = 4018 := 
by 
    sorry

end find_a_l412_41293


namespace number_from_first_group_is_6_l412_41249

-- Defining conditions
def num_students : Nat := 160
def sample_size : Nat := 20
def groups := List.range' 0 num_students (num_students / sample_size)

def num_from_group_16 (x : Nat) : Nat := 8 * 15 + x
def drawn_number_from_16 : Nat := 126

-- Main theorem
theorem number_from_first_group_is_6 : ∃ x : Nat, num_from_group_16 x = drawn_number_from_16 ∧ x = 6 := 
by
  sorry

end number_from_first_group_is_6_l412_41249


namespace find_r_values_l412_41233

theorem find_r_values (r : ℝ) (h1 : r ≥ 8) (h2 : r ≤ 20) :
  16 ≤ (r - 4) ^ (3/2) ∧ (r - 4) ^ (3/2) ≤ 128 :=
by {
  sorry
}

end find_r_values_l412_41233


namespace remainder_of_product_of_odd_primes_mod_32_l412_41218

def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

def product (l : List ℕ) : ℕ := l.foldl (· * ·) 1

theorem remainder_of_product_of_odd_primes_mod_32 :
  (product odd_primes_less_than_32) % 32 = 23 :=
by sorry

end remainder_of_product_of_odd_primes_mod_32_l412_41218


namespace sequence_existence_l412_41285

theorem sequence_existence (n : ℕ) : 
  (∃ (x : ℕ → ℤ), 
    (∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ n ∧ i + j ≤ n ∧ ((x i - x j) % 3 = 0) → (x (i + j) + x i + x j + 1) % 3 = 0)) ↔ (n = 8) := 
by 
  sorry

end sequence_existence_l412_41285


namespace unique_solution_linear_system_l412_41253

theorem unique_solution_linear_system
  (a11 a22 a33 : ℝ) (a12 a13 a21 a23 a31 a32 : ℝ) 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 < a11) (h2 : 0 < a22) (h3 : 0 < a33)
  (h4 : a12 < 0) (h5 : a13 < 0) (h6 : a21 < 0) (h7 : a23 < 0) (h8 : a31 < 0) (h9 : a32 < 0)
  (h10 : 0 < a11 + a12 + a13) (h11 : 0 < a21 + a22 + a23) (h12 : 0 < a31 + a32 + a33) :
  (a11 * x1 + a12 * x2 + a13 * x3 = 0) →
  (a21 * x1 + a22 * x2 + a23 * x3 = 0) →
  (a31 * x1 + a32 * x2 + a33 * x3 = 0) →
  x1 = 0 ∧ x2 = 0 ∧ x3 = 0 := by
  sorry

end unique_solution_linear_system_l412_41253


namespace art_performance_selection_l412_41284

-- Definitions from the conditions
def total_students := 6
def singers := 3
def dancers := 2
def both := 1

-- Mathematical expression in Lean
noncomputable def ways_to_select (n k : ℕ) : ℕ := Nat.choose n k

theorem art_performance_selection 
    (total_students singers dancers both: ℕ) 
    (h1 : total_students = 6)
    (h2 : singers = 3)
    (h3 : dancers = 2)
    (h4 : both = 1) :
  (ways_to_select 4 2 * 3 - 1) = (Nat.choose 4 2 * 3 - 1) := 
sorry

end art_performance_selection_l412_41284


namespace range_of_a_for_real_roots_l412_41287

theorem range_of_a_for_real_roots (a : ℝ) (h : a ≠ 0) :
  (∃ (x : ℝ), a*x^2 + 2*x + 1 = 0) ↔ a ≤ 1 :=
by
  sorry

end range_of_a_for_real_roots_l412_41287


namespace vertical_distance_l412_41262

variable (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ)

def totalVerticalDistance
  (storiesPerTrip tripsPerDay daysPerWeek feetPerStory : ℕ) : ℕ :=
  2 * storiesPerTrip * feetPerStory * tripsPerDay * daysPerWeek

theorem vertical_distance (h1 : storiesPerTrip = 5)
                          (h2 : tripsPerDay = 3)
                          (h3 : daysPerWeek = 7)
                          (h4 : feetPerStory = 10) :
  totalVerticalDistance storiesPerTrip tripsPerDay daysPerWeek feetPerStory = 2100 := by
  sorry

end vertical_distance_l412_41262


namespace maximum_value_l412_41260

theorem maximum_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) : 
  (x^2 - x * y + y^2) * (x^2 - x * z + z^2) * (y^2 - y * z + z^2) * (z^2 - z * x + x^2)  ≤ 1 :=
sorry

end maximum_value_l412_41260


namespace probability_of_bayonet_base_on_third_try_is_7_over_120_l412_41283

noncomputable def probability_picking_bayonet_base_bulb_on_third_try : ℚ :=
  (3 / 10) * (2 / 9) * (7 / 8)

/-- Given a box containing 3 screw base bulbs and 7 bayonet base bulbs, all with the
same shape and power and placed with their bases down. An electrician takes one bulb
at a time without returning it. The probability that he gets a bayonet base bulb on his
third try is 7/120. -/
theorem probability_of_bayonet_base_on_third_try_is_7_over_120 :
  probability_picking_bayonet_base_bulb_on_third_try = 7 / 120 :=
by 
  sorry

end probability_of_bayonet_base_on_third_try_is_7_over_120_l412_41283
