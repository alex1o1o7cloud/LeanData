import Mathlib

namespace rth_term_arithmetic_progression_l911_91188

-- Define the sum of the first n terms of the arithmetic progression
def S (n : ℕ) : ℕ := 5 * n + 4 * n^3

-- Define the r-th term of the arithmetic progression
def a (r : ℕ) : ℕ := S r - S (r - 1)

-- Theorem stating the r-th term of the arithmetic progression
theorem rth_term_arithmetic_progression (r : ℕ) : a r = 12 * r^2 - 12 * r + 9 := by
  sorry

end rth_term_arithmetic_progression_l911_91188


namespace average_annual_cost_reduction_l911_91157

theorem average_annual_cost_reduction (x : ℝ) (h : (1 - x) ^ 2 = 0.64) : x = 0.2 :=
sorry

end average_annual_cost_reduction_l911_91157


namespace smallest_possible_b_l911_91130

theorem smallest_possible_b 
  (a b : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : a - b = 8) 
  (h4 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : 
  b = 4 := 
sorry

end smallest_possible_b_l911_91130


namespace minimal_face_sum_of_larger_cube_l911_91184

-- Definitions
def num_small_cubes : ℕ := 27
def num_faces_per_cube : ℕ := 6

-- The goal: Prove the minimal sum of the integers shown on the faces of the larger cube
theorem minimal_face_sum_of_larger_cube (min_sum : ℤ) 
    (H : min_sum = 90) :
    min_sum = 90 :=
by {
  sorry
}

end minimal_face_sum_of_larger_cube_l911_91184


namespace problem1_problem2_l911_91175

-- First problem
theorem problem1 :
  2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) - (-1 / 3) ^ 0 + (-1) ^ 2023 = -2 :=
by
  sorry

-- Second problem
theorem problem2 :
  abs (1 - Real.sqrt 2) - Real.sqrt 12 + (1 / 3) ^ (-1 : ℤ) - 2 * Real.cos (Real.pi / 4) = 2 - 2 * Real.sqrt 3 :=
by
  sorry

end problem1_problem2_l911_91175


namespace find_number_l911_91166

-- Define the condition that k is a non-negative integer
def is_nonnegative_int (k : ℕ) : Prop := k ≥ 0

-- Define the condition that 18^k is a divisor of the number n
def is_divisor (n k : ℕ) : Prop := 18^k ∣ n

-- The main theorem statement
theorem find_number (n k : ℕ) (h_nonneg : is_nonnegative_int k) (h_eq : 6^k - k^6 = 1) (h_div : is_divisor n k) : n = 1 :=
  sorry

end find_number_l911_91166


namespace trapezoid_CD_length_l911_91124

theorem trapezoid_CD_length (AB CD AD BC : ℝ) (P : ℝ) 
  (h₁ : AB = 12) 
  (h₂ : AD = 5) 
  (h₃ : BC = 7) 
  (h₄ : P = 40) : CD = 16 :=
by
  sorry

end trapezoid_CD_length_l911_91124


namespace max_min_value_x_eq_1_l911_91196

noncomputable def f (x k : ℝ) : ℝ := x^2 - 2 * (2 * k - 1) * x + 3 * k^2 - 2 * k + 6

theorem max_min_value_x_eq_1 :
  ∀ (k : ℝ), (∀ x : ℝ, ∃ m : ℝ, f x k = m → k = 1 → m = 6) → (∃ x : ℝ, x = 1) :=
by
  sorry

end max_min_value_x_eq_1_l911_91196


namespace prime_quadruple_solution_l911_91138

-- Define the problem statement in Lean
theorem prime_quadruple_solution :
  ∀ (p q r : ℕ) (n : ℕ),
    Prime p → Prime q → Prime r → n > 0 →
    p^2 = q^2 + r^n →
    (p = 3 ∧ q = 2 ∧ r = 5 ∧ n = 1) ∨ (p = 5 ∧ q = 3 ∧ r = 2 ∧ n = 4) :=
by
  sorry -- Proof omitted

end prime_quadruple_solution_l911_91138


namespace geometric_sequence_a11_l911_91153

theorem geometric_sequence_a11
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h3 : a 3 = 4)
  (h7 : a 7 = 12) : 
  a 11 = 36 :=
by
  sorry

end geometric_sequence_a11_l911_91153


namespace minimum_value_proof_l911_91193

noncomputable def minimum_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : ℝ :=
  (x + y) / (x * y * z)

theorem minimum_value_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  minimum_value x y z hx hy hz h ≥ 4 :=
sorry

end minimum_value_proof_l911_91193


namespace solution_l911_91103

theorem solution (x : ℝ) (h : ¬ (x ^ 2 - 5 * x + 4 > 0)) : 1 ≤ x ∧ x ≤ 4 :=
by
  sorry

end solution_l911_91103


namespace travel_agency_choice_l911_91179

noncomputable def y₁ (x : ℝ) : ℝ := 350 * x + 1000

noncomputable def y₂ (x : ℝ) : ℝ := 400 * x + 800

theorem travel_agency_choice (x : ℝ) (h : 0 < x) :
  (x < 4 → y₁ x > y₂ x) ∧ 
  (x = 4 → y₁ x = y₂ x) ∧ 
  (x > 4 → y₁ x < y₂ x) :=
by {
  sorry
}

end travel_agency_choice_l911_91179


namespace common_chord_length_l911_91105

/-- Two circles intersect such that each passes through the other's center.
Prove that the length of their common chord is 8√3 cm. -/
theorem common_chord_length (r : ℝ) (h : r = 8) :
  let chord_length := 2 * (r * (Real.sqrt 3 / 2))
  chord_length = 8 * Real.sqrt 3 := by
  sorry

end common_chord_length_l911_91105


namespace Emilee_earns_25_l911_91185

variable (Terrence Jermaine Emilee : ℕ)
variable (h1 : Terrence = 30)
variable (h2 : Jermaine = Terrence + 5)
variable (h3 : Jermaine + Terrence + Emilee = 90)

theorem Emilee_earns_25 : Emilee = 25 := by
  -- Insert the proof here
  sorry

end Emilee_earns_25_l911_91185


namespace part_one_part_two_l911_91141

-- Definitions for the propositions
def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

def proposition_q (m : ℝ) : Prop :=
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ x y : ℝ, (x^2)/(a) + (y^2)/(b) = 1)

-- Theorems for the answers
theorem part_one (m : ℝ) : ¬ proposition_p m → m < 1 :=
by sorry

theorem part_two (m : ℝ) : ¬ (proposition_p m ∧ proposition_q m) ∧ (proposition_p m ∨ proposition_q m) → m < 1 ∨ (4 ≤ m ∧ m ≤ 6) :=
by sorry

end part_one_part_two_l911_91141


namespace simplified_expression_l911_91150

theorem simplified_expression (x : ℝ) : 
  x * (3 * x^2 - 2) - 5 * (x^2 - 2 * x + 7) = 3 * x^3 - 5 * x^2 + 8 * x - 35 := 
by
  sorry

end simplified_expression_l911_91150


namespace original_number_unique_l911_91162

theorem original_number_unique (x : ℝ) (h_pos : 0 < x) 
  (h_condition : 100 * x = 9 / x) : x = 3 / 10 :=
by
  sorry

end original_number_unique_l911_91162


namespace cube_root_sum_is_integer_iff_l911_91172

theorem cube_root_sum_is_integer_iff (n m : ℤ) (hn : n = m * (m^2 + 3) / 2) :
  ∃ (k : ℤ), (n + Real.sqrt (n^2 + 1))^(1/3) + (n - Real.sqrt (n^2 + 1))^(1/3) = k :=
by
  sorry

end cube_root_sum_is_integer_iff_l911_91172


namespace surface_area_of_rectangular_prism_l911_91144

theorem surface_area_of_rectangular_prism :
  ∀ (length width height : ℝ), length = 8 → width = 4 → height = 2 → 
    2 * (length * width + length * height + width * height) = 112 :=
by
  intros length width height h_length h_width h_height
  rw [h_length, h_width, h_height]
  sorry

end surface_area_of_rectangular_prism_l911_91144


namespace mark_owe_triple_amount_l911_91100

theorem mark_owe_triple_amount (P : ℝ) (r : ℝ) (t : ℕ) (hP : P = 2000) (hr : r = 0.04) :
  (1 + r)^t > 3 → t = 30 :=
by
  intro h
  norm_cast at h
  sorry

end mark_owe_triple_amount_l911_91100


namespace range_x2y2z_range_a_inequality_l911_91107

theorem range_x2y2z {x y z : ℝ} (h : x^2 + y^2 + z^2 = 1) : 
  -3 ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3 :=
by sorry

theorem range_a_inequality (a : ℝ) (h : ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |a - 3| + a / 2 ≥ x + 2*y + 2*z) :
  (4 ≤ a) ∨ (a ≤ 0) :=
by sorry

end range_x2y2z_range_a_inequality_l911_91107


namespace infinitely_many_n_gt_sqrt_two_l911_91115

/-- A sequence of positive integers indexed by natural numbers. -/
def a (n : ℕ) : ℕ := sorry

/-- Main theorem stating there are infinitely many n such that 1 + a_n > a_{n-1} * root n of 2. -/
theorem infinitely_many_n_gt_sqrt_two :
  ∀ (a : ℕ → ℕ), (∀ n, a n > 0) → ∃ᶠ n in at_top, 1 + a n > a (n - 1) * (2 : ℝ) ^ (1 / n : ℝ) :=
by {
  sorry
}

end infinitely_many_n_gt_sqrt_two_l911_91115


namespace partition_pos_integers_100_subsets_l911_91149

theorem partition_pos_integers_100_subsets :
  ∃ (P : (ℕ+ → Fin 100)), ∀ a b c : ℕ+, (a + 99 * b = c) → P a = P c ∨ P a = P b ∨ P b = P c :=
sorry

end partition_pos_integers_100_subsets_l911_91149


namespace weight_of_each_bag_l911_91117

theorem weight_of_each_bag (empty_weight loaded_weight : ℕ) (number_of_bags : ℕ) (weight_per_bag : ℕ)
    (h1 : empty_weight = 500)
    (h2 : loaded_weight = 1700)
    (h3 : number_of_bags = 20)
    (h4 : loaded_weight - empty_weight = number_of_bags * weight_per_bag) :
    weight_per_bag = 60 :=
by
  sorry

end weight_of_each_bag_l911_91117


namespace evaluate_expression_l911_91118

theorem evaluate_expression (x : ℤ) (h : x + 1 = 4) : 
  (-3)^3 + (-3)^2 + (-3 * x) + 3 * x + 3^2 + 3^3 = 18 :=
by
  -- Since we know the condition x + 1 = 4
  have hx : x = 3 := by linarith
  -- Substitution x = 3 into the expression
  rw [hx]
  -- The expression after substitution and simplification
  sorry

end evaluate_expression_l911_91118


namespace simon_legos_l911_91129

theorem simon_legos (Kent_legos : ℕ) (hk : Kent_legos = 40)
                    (Bruce_legos : ℕ) (hb : Bruce_legos = Kent_legos + 20)
                    (Simon_legos : ℕ) (hs : Simon_legos = Bruce_legos + Bruce_legos / 5) :
    Simon_legos = 72 := 
sorry

end simon_legos_l911_91129


namespace base_length_of_isosceles_l911_91198

-- Define the lengths of the sides and the perimeter of the triangle.
def side_length1 : ℝ := 10
def side_length2 : ℝ := 10
def perimeter : ℝ := 35

-- Define the problem statement to prove the length of the base.
theorem base_length_of_isosceles (b : ℝ) 
  (h1 : side_length1 = 10) 
  (h2 : side_length2 = 10) 
  (h3 : perimeter = 35) : b = 15 :=
by
  -- Skip the proof.
  sorry

end base_length_of_isosceles_l911_91198


namespace cube_volume_ratio_l911_91140

theorem cube_volume_ratio (edge1 edge2 : ℕ) (h1 : edge1 = 10) (h2 : edge2 = 36) :
  (edge1^3 : ℚ) / (edge2^3) = 125 / 5832 :=
by
  sorry

end cube_volume_ratio_l911_91140


namespace avg_choc_pieces_per_cookie_l911_91112

theorem avg_choc_pieces_per_cookie {cookies chips mms pieces : ℕ} 
  (h1 : cookies = 48) 
  (h2 : chips = 108) 
  (h3 : mms = chips / 3) 
  (h4 : pieces = chips + mms) : 
  pieces / cookies = 3 := 
by sorry

end avg_choc_pieces_per_cookie_l911_91112


namespace randy_quiz_score_l911_91173

theorem randy_quiz_score (q1 q2 q3 q5 : ℕ) (q4 : ℕ) :
  q1 = 90 → q2 = 98 → q3 = 94 → q5 = 96 → (q1 + q2 + q3 + q4 + q5) / 5 = 94 → q4 = 92 :=
by
  intros h1 h2 h3 h5 h_avg
  sorry

end randy_quiz_score_l911_91173


namespace sandy_saved_last_year_percentage_l911_91137

theorem sandy_saved_last_year_percentage (S : ℝ) (P : ℝ) :
  (this_year_salary: ℝ) → (this_year_savings: ℝ) → 
  (this_year_saved_percentage: ℝ) → (saved_last_year_percentage: ℝ) → 
  this_year_salary = 1.1 * S → 
  this_year_saved_percentage = 6 →
  this_year_savings = (this_year_saved_percentage / 100) * this_year_salary →
  (this_year_savings / ((P / 100) * S)) = 0.66 →
  P = 10 :=
by
  -- The proof is to be filled in here.
  sorry

end sandy_saved_last_year_percentage_l911_91137


namespace max_value_of_t_l911_91159

variable (n r t : ℕ)
variable (A : Finset (Finset (Fin n)))
variable (h₁ : n ≤ 2 * r)
variable (h₂ : ∀ s ∈ A, Finset.card s = r)
variable (h₃ : Finset.card A = t)

theorem max_value_of_t : 
  (n < 2 * r → t ≤ Nat.choose n r) ∧ 
  (n = 2 * r → t ≤ Nat.choose n r / 2) :=
by
  sorry

end max_value_of_t_l911_91159


namespace lucas_initial_money_l911_91160

theorem lucas_initial_money : (3 * 2 + 14 = 20) := by sorry

end lucas_initial_money_l911_91160


namespace groupA_forms_triangle_l911_91111

theorem groupA_forms_triangle (a b c : ℝ) (h1 : a = 13) (h2 : b = 12) (h3 : c = 20) : 
  a + b > c ∧ a + c > b ∧ b + c > a :=
by {
  sorry
}

end groupA_forms_triangle_l911_91111


namespace team_A_more_uniform_l911_91145

noncomputable def average_height : ℝ := 2.07

variables (S_A S_B : ℝ) (h_variance : S_A^2 < S_B^2)

theorem team_A_more_uniform : true ∧ false :=
by
  sorry

end team_A_more_uniform_l911_91145


namespace quadratic_eq_zero_l911_91120

theorem quadratic_eq_zero (x a b : ℝ) (h : x = a ∨ x = b) : x^2 - (a + b) * x + a * b = 0 :=
by sorry

end quadratic_eq_zero_l911_91120


namespace find_k_l911_91170

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h2 : k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l911_91170


namespace triangle_area_l911_91174

theorem triangle_area (a b c : ℝ) (h1 : a = 15) (h2 : b = 36) (h3 : c = 39) (h4 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 270 :=
by
  sorry

end triangle_area_l911_91174


namespace find_f_11_5_l911_91194

-- Definitions based on the conditions.
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def periodic_with_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

def functional_eqn (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -1 / f x

def f_defined_on_interval (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, -3 ≤ x ∧ x ≤ -2 → f x = 2 * x

-- The main theorem to prove.
theorem find_f_11_5 (f : ℝ → ℝ) :
  is_even_function f →
  functional_eqn f →
  f_defined_on_interval f →
  periodic_with_period f 6 →
  f 11.5 = 1 / 5 :=
  by
    intros h_even h_fun_eqn h_interval h_periodic
    sorry  -- proof goes here

end find_f_11_5_l911_91194


namespace largest_prime_inequality_l911_91192

def largest_prime_divisor (n : Nat) : Nat :=
  sorry  -- Placeholder to avoid distractions in problem statement

theorem largest_prime_inequality (q : Nat) (h_q_prime : Prime q) (hq_odd : q % 2 = 1) :
    ∃ k : Nat, k > 0 ∧ largest_prime_divisor (q^(2^k) - 1) < q ∧ q < largest_prime_divisor (q^(2^k) + 1) :=
sorry

end largest_prime_inequality_l911_91192


namespace area_of_triangle_formed_by_medians_l911_91171

variable {a b c m_a m_b m_c Δ Δ': ℝ}

-- Conditions from the problem
axiom rel_sum_of_squares : m_a^2 + m_b^2 + m_c^2 = (3 / 4) * (a^2 + b^2 + c^2)
axiom rel_fourth_powers : m_a^4 + m_b^4 + m_c^4 = (9 / 16) * (a^4 + b^4 + c^4)

-- Statement of the problem to prove
theorem area_of_triangle_formed_by_medians :
  Δ' = (3 / 4) * Δ := sorry

end area_of_triangle_formed_by_medians_l911_91171


namespace fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l911_91183

theorem fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes :
  (15 * 25 + 20 * 10 = 5 * 25 + n * 10) -> n = 45 :=
by
  sorry

end fifteen_quarters_twenty_dimes_equal_five_quarters_n_dimes_l911_91183


namespace abs_cube_root_neg_64_l911_91148

-- Definitions required for the problem
def cube_root (x : ℝ) : ℝ := x^(1/3)
def abs_value (x : ℝ) : ℝ := abs x

-- The statement of the problem
theorem abs_cube_root_neg_64 : abs_value (cube_root (-64)) = 4 :=
by sorry

end abs_cube_root_neg_64_l911_91148


namespace siblings_age_problem_l911_91116

variable {x y z : ℕ}

theorem siblings_age_problem
  (h1 : x - y = 3)
  (h2 : z - 1 = 2 * (x + y))
  (h3 : z + 20 = x + y + 40) :
  x = 11 ∧ y = 8 ∧ z = 39 :=
by
  sorry

end siblings_age_problem_l911_91116


namespace sum_of_decimals_l911_91127

theorem sum_of_decimals :
  let a := 0.3
  let b := 0.08
  let c := 0.007
  a + b + c = 0.387 :=
by
  sorry

end sum_of_decimals_l911_91127


namespace amount_of_loan_l911_91182

theorem amount_of_loan (P R T SI : ℝ) (hR : R = 6) (hT : T = 6) (hSI : SI = 432) :
  SI = (P * R * T) / 100 → P = 1200 :=
by
  intro h
  sorry

end amount_of_loan_l911_91182


namespace confetti_left_correct_l911_91121

-- Define the number of pieces of red and green confetti collected by Eunji
def red_confetti : ℕ := 1
def green_confetti : ℕ := 9

-- Define the total number of pieces of confetti collected by Eunji
def total_confetti : ℕ := red_confetti + green_confetti

-- Define the number of pieces of confetti given to Yuna
def given_to_Yuna : ℕ := 4

-- Define the number of pieces of confetti left with Eunji
def confetti_left : ℕ :=  red_confetti + green_confetti - given_to_Yuna

-- Goal to prove
theorem confetti_left_correct : confetti_left = 6 := by
  -- Here the steps proving the equality would go, but we add sorry to skip the proof
  sorry

end confetti_left_correct_l911_91121


namespace two_digit_number_l911_91113

theorem two_digit_number (x y : ℕ) (h1 : y = x + 4) (h2 : (10 * x + y) * (x + y) = 208) :
  10 * x + y = 26 :=
sorry

end two_digit_number_l911_91113


namespace digit_y_in_base_7_divisible_by_19_l911_91187

def base7_to_decimal (a b c d : ℕ) : ℕ := a * 7^3 + b * 7^2 + c * 7 + d

theorem digit_y_in_base_7_divisible_by_19 (y : ℕ) (hy : y < 7) :
  (∃ k : ℕ, base7_to_decimal 5 2 y 3 = 19 * k) ↔ y = 8 :=
by {
  sorry
}

end digit_y_in_base_7_divisible_by_19_l911_91187


namespace total_profit_is_64000_l911_91132

-- Definitions for investments and periods
variables (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ)

-- Conditions from the problem
def condition1 := IA = 5 * IB
def condition2 := TA = 3 * TB
def condition3 := Profit_B = 4000
def condition4 := Profit_A / Profit_B = (IA * TA) / (IB * TB)

-- Target statement to be proved
theorem total_profit_is_64000 (IB IA TB TA Profit_B Profit_A Total_Profit : ℕ) :
  condition1 IA IB → condition2 TA TB → condition3 Profit_B → condition4 IA TA IB TB Profit_A Profit_B → 
  Total_Profit = Profit_A + Profit_B → Total_Profit = 64000 :=
by {
  sorry
}

end total_profit_is_64000_l911_91132


namespace max_min_values_l911_91191

theorem max_min_values (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : 
  1 ≤ x^2 + 2*y^2 + 3*z^2 ∧ x^2 + 2*y^2 + 3*z^2 ≤ 3 := by
  sorry

end max_min_values_l911_91191


namespace determine_subtracted_number_l911_91186

theorem determine_subtracted_number (x y : ℤ) (h1 : x = 40) (h2 : 7 * x - y = 130) : y = 150 :=
by sorry

end determine_subtracted_number_l911_91186


namespace work_completion_time_l911_91163

theorem work_completion_time (A_works_in : ℕ) (A_works_days : ℕ) (B_works_remainder_in : ℕ) (total_days : ℕ) :
  (A_works_in = 60) → (A_works_days = 15) → (B_works_remainder_in = 30) → (total_days = 24) := 
by
  intros hA_work hA_days hB_work
  sorry

end work_completion_time_l911_91163


namespace fries_sold_l911_91136

theorem fries_sold (small_fries large_fries : ℕ) (h1 : small_fries = 4) (h2 : large_fries = 5 * small_fries) :
  small_fries + large_fries = 24 :=
  by
    sorry

end fries_sold_l911_91136


namespace least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l911_91104

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def product_of_digits (n : ℕ) : ℕ :=
  (digits n).foldl (λ x y => x * y) 1

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k, n = m * k

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of 45 n ∧ is_multiple_of 9 (product_of_digits n)

theorem least_positive_multiple_of_45_with_product_of_digits_multiple_of_9 : 
  ∀ n, satisfies_conditions n → 495 ≤ n :=
by
  sorry

end least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l911_91104


namespace bob_needs_additional_weeks_l911_91164

-- Definitions based on conditions
def weekly_prize : ℕ := 100
def initial_weeks_won : ℕ := 2
def total_prize_won : ℕ := initial_weeks_won * weekly_prize
def puppy_cost : ℕ := 1000
def additional_weeks_needed : ℕ := (puppy_cost - total_prize_won) / weekly_prize

-- Statement of the theorem
theorem bob_needs_additional_weeks : additional_weeks_needed = 8 := by
  -- Proof here
  sorry

end bob_needs_additional_weeks_l911_91164


namespace cube_number_sum_is_102_l911_91168

noncomputable def sum_of_cube_numbers (n1 n2 n3 n4 n5 n6 : ℕ) : ℕ := n1 + n2 + n3 + n4 + n5 + n6

theorem cube_number_sum_is_102 : 
  ∃ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 12 ∧ 
    n2 = n1 + 2 ∧ 
    n3 = n2 + 2 ∧ 
    n4 = n3 + 2 ∧ 
    n5 = n4 + 2 ∧ 
    n6 = n5 + 2 ∧ 
    ((n1 + n6 = n2 + n5) ∧ (n1 + n6 = n3 + n4)) ∧ 
    sum_of_cube_numbers n1 n2 n3 n4 n5 n6 = 102 :=
by
  sorry

end cube_number_sum_is_102_l911_91168


namespace B1F_base16_to_base10_is_2847_l911_91152

theorem B1F_base16_to_base10_is_2847 : 
  let B := 11
  let one := 1
  let F := 15
  let base := 16
  B * base^2 + one * base^1 + F * base^0 = 2847 := 
by
  sorry

end B1F_base16_to_base10_is_2847_l911_91152


namespace ramsey_example_l911_91180

theorem ramsey_example (P : Fin 10 → Fin 10 → Prop) :
  (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(¬P i j ∧ ¬P j k ∧ ¬P k i))
  ∨ (∀ (i j k : Fin 10), i ≠ j → i ≠ k → j ≠ k → ¬(P i j ∧ P j k ∧ P k i)) →
  (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (P i j ∧ P j k ∧ P k l ∧ P i k ∧ P j l ∧ P i l))
  ∨ (∃ (i j k l : Fin 10), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ (¬P i j ∧ ¬P j k ∧ ¬P k l ∧ ¬P i k ∧ ¬P j l ∧ ¬P i l)) :=
by
  sorry

end ramsey_example_l911_91180


namespace point_P_coordinates_l911_91125

-- Definitions based on conditions
def in_fourth_quadrant (P : ℝ × ℝ) : Prop :=
  P.1 > 0 ∧ P.2 < 0

def distance_to_x_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.2 = d

def distance_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop :=
  abs P.1 = d

-- The theorem statement based on the proof problem
theorem point_P_coordinates :
  ∃ P : ℝ × ℝ, 
    in_fourth_quadrant P ∧ 
    distance_to_x_axis P 2 ∧ 
    distance_to_y_axis P 3 ∧ 
    P = (3, -2) :=
by
  sorry

end point_P_coordinates_l911_91125


namespace peaches_total_l911_91143

def peaches_in_basket (a b : Nat) : Nat :=
  a + b 

theorem peaches_total (a b : Nat) (h1 : a = 20) (h2 : b = 25) : peaches_in_basket a b = 45 := 
by
  sorry

end peaches_total_l911_91143


namespace arithmetic_problem_l911_91154

theorem arithmetic_problem : 
  let x := 512.52 
  let y := 256.26 
  let diff := x - y 
  let result := diff * 3 
  result = 768.78 := 
by 
  sorry

end arithmetic_problem_l911_91154


namespace aaron_ate_more_apples_l911_91178

-- Define the number of apples eaten by Aaron and Zeb
def apples_eaten_by_aaron : ℕ := 6
def apples_eaten_by_zeb : ℕ := 1

-- Theorem to prove the difference in apples eaten
theorem aaron_ate_more_apples :
  apples_eaten_by_aaron - apples_eaten_by_zeb = 5 :=
by
  sorry

end aaron_ate_more_apples_l911_91178


namespace middle_rectangle_frequency_l911_91139

theorem middle_rectangle_frequency (S A : ℝ) (h1 : S + A = 100) (h2 : A = S / 3) : A = 25 :=
by
  sorry

end middle_rectangle_frequency_l911_91139


namespace compound_interest_amount_l911_91109

theorem compound_interest_amount (P r t SI : ℝ) (h1 : t = 3) (h2 : r = 0.10) (h3 : SI = 900) :
  SI = P * r * t → P = 900 / (0.10 * 3) → (P * (1 + r)^t - P = 993) :=
by
  intros hSI hP
  sorry

end compound_interest_amount_l911_91109


namespace number_of_players_in_tournament_l911_91134

theorem number_of_players_in_tournament (n : ℕ) (h : 2 * 30 = n * (n - 1)) : n = 10 :=
sorry

end number_of_players_in_tournament_l911_91134


namespace ranges_of_a_and_m_l911_91135

open Set Real

def A : Set Real := {x | x^2 - 3*x + 2 = 0}
def B (a : Real) : Set Real := {x | x^2 - a*x + a - 1 = 0}
def C (m : Real) : Set Real := {x | x^2 - m*x + 2 = 0}

theorem ranges_of_a_and_m (a m : Real) :
  A ∪ B a = A → A ∩ C m = C m → (a = 2 ∨ a = 3) ∧ (m = 3 ∨ (-2*sqrt 2 < m ∧ m < 2*sqrt 2)) :=
by
  have hA : A = {1, 2} := sorry
  sorry

end ranges_of_a_and_m_l911_91135


namespace min_xy_min_x_plus_y_l911_91110

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : xy ≥ 4 := sorry

theorem min_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 4 * y - 2 * x * y = 0) : x + y ≥ 9 / 2 := sorry

end min_xy_min_x_plus_y_l911_91110


namespace quadratic_single_root_pos_value_l911_91102

theorem quadratic_single_root_pos_value (m : ℝ) (h1 : (6 * m)^2 - 4 * 1 * 2 * m = 0) : m = 2 / 9 :=
sorry

end quadratic_single_root_pos_value_l911_91102


namespace largest_common_divisor_of_product_l911_91122

theorem largest_common_divisor_of_product (n : ℕ) (h_even : n % 2 = 0) (h_pos : 0 < n) :
  ∃ d : ℕ, d = 105 ∧ ∀ k : ℕ, k = (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13) → d ∣ k :=
by
  sorry

end largest_common_divisor_of_product_l911_91122


namespace quadratic_trinomial_positive_c_l911_91133

theorem quadratic_trinomial_positive_c
  (a b c : ℝ)
  (h1 : b^2 < 4 * a * c)
  (h2 : a + b + c > 0) :
  c > 0 :=
sorry

end quadratic_trinomial_positive_c_l911_91133


namespace calculate_nested_expression_l911_91156

theorem calculate_nested_expression :
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2 = 1457 :=
by
  sorry

end calculate_nested_expression_l911_91156


namespace pumpkins_at_other_orchard_l911_91197

-- Defining the initial conditions
def sunshine_pumpkins : ℕ := 54
def other_orchard_pumpkins : ℕ := 14

-- Equation provided in the problem
def condition_equation (P : ℕ) : Prop := 54 = 3 * P + 12

-- Proving the main statement using the conditions
theorem pumpkins_at_other_orchard : condition_equation other_orchard_pumpkins :=
by
  unfold condition_equation
  sorry -- To be completed with the proof

end pumpkins_at_other_orchard_l911_91197


namespace cars_produced_total_l911_91146

theorem cars_produced_total :
  3884 + 2871 = 6755 :=
by
  sorry

end cars_produced_total_l911_91146


namespace intersection_non_empty_l911_91199

open Set

def M (a : ℤ) : Set ℤ := {a, 0}
def N : Set ℤ := {x | 2 * x^2 - 5 * x < 0}

theorem intersection_non_empty (a : ℤ) (h : (M a) ∩ N ≠ ∅) : a = 1 ∨ a = 2 := 
sorry

end intersection_non_empty_l911_91199


namespace p_and_q_necessary_but_not_sufficient_l911_91195

theorem p_and_q_necessary_but_not_sufficient (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) := 
by 
  sorry

end p_and_q_necessary_but_not_sufficient_l911_91195


namespace correct_equation_by_moving_digit_l911_91169

theorem correct_equation_by_moving_digit :
  (10^2 - 1 = 99) → (101 = 102 - 1) :=
by
  intro h
  sorry

end correct_equation_by_moving_digit_l911_91169


namespace cleaner_for_dog_stain_l911_91161

theorem cleaner_for_dog_stain (D : ℝ) (H : 6 * D + 3 * 4 + 1 * 1 = 49) : D = 6 :=
by 
  -- Proof steps would go here, but we are skipping the proof.
  sorry

end cleaner_for_dog_stain_l911_91161


namespace appropriate_sampling_method_l911_91181

theorem appropriate_sampling_method
  (total_students : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (survey_size : ℕ)
  (diff_interests : Prop)
  (h1 : total_students = 1000)
  (h2 : male_students = 500)
  (h3 : female_students = 500)
  (h4 : survey_size = 100)
  (h5 : diff_interests) : 
  sampling_method = "stratified sampling" :=
by
  sorry

end appropriate_sampling_method_l911_91181


namespace lloyd_excess_rate_multiple_l911_91177

theorem lloyd_excess_rate_multiple :
  let h_regular := 7.5
  let r := 4.00
  let h_total := 10.5
  let e_total := 48
  let e_regular := h_regular * r
  let excess_hours := h_total - h_regular
  let e_excess := e_total - e_regular
  let m := e_excess / (excess_hours * r)
  m = 1.5 :=
by
  sorry

end lloyd_excess_rate_multiple_l911_91177


namespace isosceles_triangle_perimeter_l911_91190

/-- Given an isosceles triangle with one side length of 3 cm and another side length of 5 cm,
    its perimeter is either 11 cm or 13 cm. -/
theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 3) (h2 : b = 5) : 
  (∃ c : ℝ, (c = 3 ∨ c = 5) ∧ (2 * a + b = 11 ∨ 2 * b + a = 13)) :=
by
  sorry

end isosceles_triangle_perimeter_l911_91190


namespace total_fires_l911_91189

theorem total_fires (Doug_fires Kai_fires Eli_fires : ℕ)
  (h1 : Doug_fires = 20)
  (h2 : Kai_fires = 3 * Doug_fires)
  (h3 : Eli_fires = Kai_fires / 2) :
  Doug_fires + Kai_fires + Eli_fires = 110 :=
by
  sorry

end total_fires_l911_91189


namespace total_sum_lent_l911_91106

-- Conditions
def interest_equal (x y : ℕ) : Prop :=
  (x * 3 * 8) / 100 = (y * 5 * 3) / 100

def second_sum : ℕ := 1704

-- Assertion
theorem total_sum_lent : ∃ x : ℕ, interest_equal x second_sum ∧ (x + second_sum = 2769) :=
  by
  -- Placeholder proof
  sorry

end total_sum_lent_l911_91106


namespace three_pow_two_digits_count_l911_91142

theorem three_pow_two_digits_count : 
  ∃ n_set : Finset ℕ, (∀ n ∈ n_set, 10 ≤ 3^n ∧ 3^n < 100) ∧ n_set.card = 2 := 
sorry

end three_pow_two_digits_count_l911_91142


namespace sat_marking_problem_l911_91155

-- Define the recurrence relation for the number of ways to mark questions without consecutive markings of the same letter.
def f : ℕ → ℕ
| 0     => 1
| 1     => 2
| 2     => 3
| (n+2) => f (n+1) + f n

-- Define that each letter marking can be done in 32 different ways.
def markWays : ℕ := 32

-- Define the number of questions to be 10.
def numQuestions : ℕ := 10

-- Calculate the number of sequences of length numQuestions with no consecutive same markings.
def numWays := f numQuestions

-- Prove that the number of ways results in 2^20 * 3^10 and compute 100m + n + p where m = 20, n = 10, p = 3.
theorem sat_marking_problem :
  (numWays ^ 5 = 2 ^ 20 * 3 ^ 10) ∧ (100 * 20 + 10 + 3 = 2013) :=
by
  sorry

end sat_marking_problem_l911_91155


namespace circle_center_l911_91165

theorem circle_center (x y: ℝ) : 
  (x + 2)^2 + (y + 3)^2 = 29 ↔ (∃ c1 c2 : ℝ, c1 = -2 ∧ c2 = -3) :=
by sorry

end circle_center_l911_91165


namespace cost_function_segments_l911_91126

def C (n : ℕ) : ℕ :=
  if h : 1 ≤ n ∧ n ≤ 10 then 10 * n
  else if h : 10 < n then 8 * n - 40
  else 0

theorem cost_function_segments :
  (∀ n, 1 ≤ n ∧ n ≤ 10 → C n = 10 * n) ∧
  (∀ n, 10 < n → C n = 8 * n - 40) ∧
  (∀ n, C n = if (1 ≤ n ∧ n ≤ 10) then 10 * n else if (10 < n) then 8 * n - 40 else 0) ∧
  ∃ n₁ n₂, (1 ≤ n₁ ∧ n₁ ≤ 10) ∧ (10 < n₂ ∧ n₂ ≤ 20) ∧ C n₁ = 10 * n₁ ∧ C n₂ = 8 * n₂ - 40 :=
by
  sorry

end cost_function_segments_l911_91126


namespace teacher_students_and_ticket_cost_l911_91108

theorem teacher_students_and_ticket_cost 
    (C_s C_a : ℝ) 
    (n_k n_h : ℕ)
    (hk_total ht_total : ℝ) 
    (h_students : n_h = n_k + 3)
    (hk  : n_k * C_s + C_a = hk_total)
    (ht : n_h * C_s + C_a = ht_total)
    (hk_total_val : hk_total = 994)
    (ht_total_val : ht_total = 1120)
    (C_s_val : C_s = 42) : 
    (n_h = 25) ∧ (C_a = 70) := 
by
  -- Proof steps would be provided here
  sorry

end teacher_students_and_ticket_cost_l911_91108


namespace math_problem_l911_91147

variable (x b : ℝ)
variable (h1 : x < b)
variable (h2 : b < 0)
variable (h3 : b = -2)

theorem math_problem : x^2 > b * x ∧ b * x > b^2 :=
by
  sorry

end math_problem_l911_91147


namespace value_a6_l911_91128

noncomputable def seq (a : ℕ → ℕ) : Prop :=
a 1 = 1 ∧ ∀ n ≥ 2, a n - a (n - 1) = n - 1

theorem value_a6 : ∃ a : ℕ → ℕ, seq a ∧ a 6 = 16 := by
  sorry

end value_a6_l911_91128


namespace main_theorem_l911_91158

noncomputable def proof_problem (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  2 * Real.sin (2 * α) ≤ Real.cos (α / 2)

noncomputable def equality_case (α : ℝ) (h1 : 0 < α) (h2 : α < π) : Prop :=
  α = π / 3 → 2 * Real.sin (2 * α) = Real.cos (α / 2)

theorem main_theorem (α : ℝ) (h1 : 0 < α) (h2 : α < π) :
  proof_problem α h1 h2 ∧ equality_case α h1 h2 :=
by
  sorry

end main_theorem_l911_91158


namespace olivia_race_time_l911_91123

variable (O E : ℕ)

theorem olivia_race_time (h1 : O + E = 112) (h2 : E = O - 4) : O = 58 :=
sorry

end olivia_race_time_l911_91123


namespace solution_set_of_inequality_l911_91131

theorem solution_set_of_inequality (a x : ℝ) (h1 : a < 2) (h2 : a * x > 2 * x + a - 2) : x < 1 :=
sorry

end solution_set_of_inequality_l911_91131


namespace claire_speed_l911_91176

def distance := 2067
def time := 39

def speed (d : ℕ) (t : ℕ) : ℕ := d / t

theorem claire_speed : speed distance time = 53 := by
  sorry

end claire_speed_l911_91176


namespace total_items_8_l911_91119

def sandwiches_cost : ℝ := 5.0
def soft_drinks_cost : ℝ := 1.5
def total_money : ℝ := 40.0

noncomputable def total_items (s : ℕ) (d : ℕ) : ℕ := s + d

theorem total_items_8 :
  ∃ (s d : ℕ), 5 * (s : ℝ) + 1.5 * (d : ℝ) = 40 ∧ s + d = 8 := 
by
  sorry

end total_items_8_l911_91119


namespace counting_numbers_dividing_48_with_remainder_7_l911_91167

theorem counting_numbers_dividing_48_with_remainder_7 :
  ∃ (S : Finset ℕ), S.card = 5 ∧ ∀ n ∈ S, n > 7 ∧ 48 % n = 0 :=
by
  sorry

end counting_numbers_dividing_48_with_remainder_7_l911_91167


namespace total_pawns_left_l911_91114

  -- Definitions of initial conditions
  def initial_pawns_in_chess : Nat := 8
  def kennedy_pawns_lost : Nat := 4
  def riley_pawns_lost : Nat := 1

  -- Theorem statement to prove the total number of pawns left
  theorem total_pawns_left : (initial_pawns_in_chess - kennedy_pawns_lost) + (initial_pawns_in_chess - riley_pawns_lost) = 11 := by
    sorry
  
end total_pawns_left_l911_91114


namespace trigonometric_identity_proof_l911_91101

noncomputable def trigonometric_expression : ℝ := 
  (Real.sin (15 * Real.pi / 180) * Real.cos (25 * Real.pi / 180) 
  + Real.cos (165 * Real.pi / 180) * Real.cos (115 * Real.pi / 180)) /
  (Real.sin (35 * Real.pi / 180) * Real.cos (5 * Real.pi / 180) 
  + Real.cos (145 * Real.pi / 180) * Real.cos (85 * Real.pi / 180))

theorem trigonometric_identity_proof : trigonometric_expression = 1 :=
by
  sorry

end trigonometric_identity_proof_l911_91101


namespace hanging_spheres_ratio_l911_91151

theorem hanging_spheres_ratio (m1 m2 g T_B T_H : ℝ)
  (h1 : T_B = 3 * T_H)
  (h2 : T_H = m2 * g)
  (h3 : T_B = m1 * g + T_H)
  : m1 / m2 = 2 :=
by
  sorry

end hanging_spheres_ratio_l911_91151
