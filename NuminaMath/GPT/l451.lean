import Mathlib

namespace simplify_inverse_expression_l451_45101

theorem simplify_inverse_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ - y⁻¹ + z⁻¹)⁻¹ = (x * y * z) / (y * z - x * z + x * y) :=
by
  sorry

end simplify_inverse_expression_l451_45101


namespace cricket_bat_selling_price_l451_45105

theorem cricket_bat_selling_price (profit : ℝ) (profit_percentage : ℝ) (C : ℝ) (selling_price : ℝ) 
  (h1 : profit = 150) 
  (h2 : profit_percentage = 20) 
  (h3 : profit = (profit_percentage / 100) * C) 
  (h4 : selling_price = C + profit) : 
  selling_price = 900 := 
sorry

end cricket_bat_selling_price_l451_45105


namespace solve_fractional_equation_l451_45167

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -1) :
  1 / x = 2 / (x + 1) → x = 1 :=
by
  sorry

end solve_fractional_equation_l451_45167


namespace arithmetic_sequence_201_is_61_l451_45152

def is_arithmetic_sequence_term (a_5 a_45 : ℤ) (n : ℤ) (a_n : ℤ) : Prop :=
  ∃ d a_1, a_1 + 4 * d = a_5 ∧ a_1 + 44 * d = a_45 ∧ a_1 + (n - 1) * d = a_n

theorem arithmetic_sequence_201_is_61 : is_arithmetic_sequence_term 33 153 61 201 :=
sorry

end arithmetic_sequence_201_is_61_l451_45152


namespace solve_x_l451_45142

theorem solve_x : ∃ (x : ℚ), (3*x - 17) / 4 = (x + 9) / 6 ∧ x = 69 / 7 :=
by
  sorry

end solve_x_l451_45142


namespace total_students_l451_45135

theorem total_students (teams students_per_team : ℕ) (h1 : teams = 9) (h2 : students_per_team = 18) :
  teams * students_per_team = 162 := by
  sorry

end total_students_l451_45135


namespace chocolate_per_friend_l451_45159

-- Definitions according to the conditions
def total_chocolate : ℚ := 60 / 7
def piles := 5
def friends := 3

-- Proof statement for the equivalent problem
theorem chocolate_per_friend :
  (total_chocolate / piles) * (piles - 1) / friends = 16 / 7 := by
  sorry

end chocolate_per_friend_l451_45159


namespace rectangle_perimeter_l451_45137

theorem rectangle_perimeter
  (w l P : ℝ)
  (h₁ : l = 2 * w)
  (h₂ : l * w = 400) :
  P = 60 * Real.sqrt 2 :=
by
  sorry

end rectangle_perimeter_l451_45137


namespace balance_the_scale_l451_45189

theorem balance_the_scale (w1 : ℝ) (w2 : ℝ) (book_weight : ℝ) (h1 : w1 = 0.5) (h2 : w2 = 0.3) :
  book_weight = w1 + 2 * w2 :=
by
  sorry

end balance_the_scale_l451_45189


namespace part1_part2_l451_45166

noncomputable def A (x : ℝ) : Prop := x < 0 ∨ x > 2
noncomputable def B (a x : ℝ) : Prop := a ≤ x ∧ x ≤ 3 - 2 * a

-- Part (1)
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, A x ∨ B a x) ↔ (a ≤ 0) := 
sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, B a x → (0 ≤ x ∧ x ≤ 2)) ↔ (1 / 2 ≤ a) :=
sorry

end part1_part2_l451_45166


namespace min_value_of_f_l451_45174

def f (x : ℝ) (a : ℝ) := - x^3 + a * x^2 - 4

def f_deriv (x : ℝ) (a : ℝ) := - 3 * x^2 + 2 * a * x

theorem min_value_of_f (h : f_deriv (2) a = 0)
  (hm : ∀ m : ℝ, -1 ≤ m ∧ m ≤ 1 → f m a + f_deriv m a ≥ f 0 3 + f_deriv (-1) 3) :
  f 0 3 + f_deriv (-1) 3 = -13 :=
by sorry

end min_value_of_f_l451_45174


namespace product_sequence_equals_8_l451_45102

theorem product_sequence_equals_8 :
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := 
by
  sorry

end product_sequence_equals_8_l451_45102


namespace change_received_correct_l451_45131

-- Define the prices of items and the amount paid
def price_hamburger : ℕ := 4
def price_onion_rings : ℕ := 2
def price_smoothie : ℕ := 3
def amount_paid : ℕ := 20

-- Define the total cost and the change received
def total_cost : ℕ := price_hamburger + price_onion_rings + price_smoothie
def change_received : ℕ := amount_paid - total_cost

-- Theorem stating the change received
theorem change_received_correct : change_received = 11 := by
  sorry

end change_received_correct_l451_45131


namespace length_of_segment_GH_l451_45194

theorem length_of_segment_GH (a1 a2 a3 a4 : ℕ)
  (h1 : a1 = a2 + 11)
  (h2 : a2 = a3 + 5)
  (h3 : a3 = a4 + 13)
  : a1 - a4 = 29 :=
by
  sorry

end length_of_segment_GH_l451_45194


namespace polynomial_divisibility_l451_45111

theorem polynomial_divisibility (P : Polynomial ℂ) (n : ℕ) 
  (h : ∃ Q : Polynomial ℂ, P.comp (X ^ n) = (X - 1) * Q) : 
  ∃ R : Polynomial ℂ, P.comp (X ^ n) = (X ^ n - 1) * R :=
sorry

end polynomial_divisibility_l451_45111


namespace find_a_and_theta_find_sin_alpha_plus_pi_over_3_l451_45118

noncomputable def f (a θ x : ℝ) : ℝ :=
  (a + 2 * Real.cos x ^ 2) * Real.cos (2 * x + θ)

theorem find_a_and_theta (a θ : ℝ) (h1 : f a θ (Real.pi / 4) = 0)
  (h2 : ∀ x, f a θ (-x) = -f a θ x) :
  a = -1 ∧ θ = Real.pi / 2 :=
sorry

theorem find_sin_alpha_plus_pi_over_3 (α θ : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : f (-1) (Real.pi / 2) (α / 4) = -2 / 5) :
  Real.sin (α + Real.pi / 3) = (4 - 3 * Real.sqrt 3) / 10 :=
sorry

end find_a_and_theta_find_sin_alpha_plus_pi_over_3_l451_45118


namespace train_crossing_time_l451_45117

noncomputable def time_to_cross_bridge (length_train : ℝ) (length_bridge : ℝ) (speed_kmh : ℝ) : ℝ :=
  let total_distance := length_train + length_bridge
  let speed_ms := (speed_kmh * 1000) / 3600
  total_distance / speed_ms

theorem train_crossing_time :
  time_to_cross_bridge 100 145 65 = 13.57 :=
by
  sorry

end train_crossing_time_l451_45117


namespace nate_ratio_is_four_to_one_l451_45198

def nate_exercise : Prop :=
  ∃ (D T L : ℕ), 
    T = D + 500 ∧ 
    T = 1172 ∧ 
    L = 168 ∧ 
    D / L = 4

theorem nate_ratio_is_four_to_one : nate_exercise := 
  sorry

end nate_ratio_is_four_to_one_l451_45198


namespace triangle_area_ratio_l451_45191

theorem triangle_area_ratio 
  (AB BC CA : ℝ)
  (p q r : ℝ)
  (ABC_area DEF_area : ℝ)
  (hAB : AB = 12)
  (hBC : BC = 16)
  (hCA : CA = 20)
  (h1 : p + q + r = 3 / 4)
  (h2 : p^2 + q^2 + r^2 = 1 / 2)
  (area_DEF_to_ABC : DEF_area / ABC_area = 385 / 512)
  : 897 = 385 + 512 := 
by
  sorry

end triangle_area_ratio_l451_45191


namespace find_B_plus_C_l451_45148

-- Define the arithmetic translations for base 8 numbers
def base8_to_dec (a b c : ℕ) : ℕ := 8^2 * a + 8 * b + c

def condition1 (A B C : ℕ) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ 1 ≤ A ∧ A ≤ 7 ∧ 1 ≤ B ∧ B ≤ 7 ∧ 1 ≤ C ∧ C ≤ 7

-- Define the main condition in the problem
def condition2 (A B C : ℕ) : Prop :=
  base8_to_dec A B C + base8_to_dec B C A + base8_to_dec C A B = 8^3 * A + 8^2 * A + 8 * A

-- The main statement to be proven
theorem find_B_plus_C (A B C : ℕ) (h1 : condition1 A B C) (h2 : condition2 A B C) : B + C = 7 :=
sorry

end find_B_plus_C_l451_45148


namespace true_inverse_negation_l451_45164

theorem true_inverse_negation : ∀ (α β : ℝ),
  (α = β) ↔ (α = β) := 
sorry

end true_inverse_negation_l451_45164


namespace seating_arrangement_l451_45144

-- We define the conditions under which we will prove our theorem.
def chairs : ℕ := 7
def people : ℕ := 5

/-- Prove that there are exactly 1800 ways to seat five people in seven chairs such that the first person cannot sit in the first or last chair. -/
theorem seating_arrangement : (5 * 6 * 5 * 4 * 3) = 1800 :=
by
  sorry

end seating_arrangement_l451_45144


namespace Greg_and_Earl_together_l451_45151

-- Conditions
def Earl_initial : ℕ := 90
def Fred_initial : ℕ := 48
def Greg_initial : ℕ := 36

def Earl_to_Fred : ℕ := 28
def Fred_to_Greg : ℕ := 32
def Greg_to_Earl : ℕ := 40

def Earl_final : ℕ := Earl_initial - Earl_to_Fred + Greg_to_Earl
def Fred_final : ℕ := Fred_initial + Earl_to_Fred - Fred_to_Greg
def Greg_final : ℕ := Greg_initial + Fred_to_Greg - Greg_to_Earl

-- Theorem statement
theorem Greg_and_Earl_together : Greg_final + Earl_final = 130 := by
  sorry

end Greg_and_Earl_together_l451_45151


namespace no_such_ab_l451_45176

theorem no_such_ab (a b : ℤ) : ¬ (2006^2 ∣ a^2006 + b^2006 + 1) :=
sorry

end no_such_ab_l451_45176


namespace area_of_triangle_ABC_l451_45110

def point : Type := ℝ × ℝ

def A : point := (2, 1)
def B : point := (1, 4)
def on_line (C : point) : Prop := C.1 + C.2 = 9
def area_triangle (A B C : point) : ℝ := 0.5 * abs (A.1 * B.2 + B.1 * C.2 + C.1 * A.2 - B.1 * A.2 - C.1 * B.2 - A.1 * C.2)

theorem area_of_triangle_ABC :
  ∃ C : point, on_line C ∧ area_triangle A B C = 2 :=
sorry

end area_of_triangle_ABC_l451_45110


namespace product_ab_l451_45125

noncomputable def a : ℝ := 1           -- From the condition 1 = a * tan(π / 4)
noncomputable def b : ℝ := 2           -- From the condition π / b = π / 2

theorem product_ab (a b : ℝ)
  (ha : a > 0) (hb : b > 0)
  (period_condition : (π / b = π / 2))
  (point_condition : a * Real.tan ((π / 8) * b) = 1) :
  a * b = 2 := sorry

end product_ab_l451_45125


namespace speed_of_train_is_20_l451_45186

def length_of_train := 120 -- in meters
def time_to_cross := 6 -- in seconds

def speed_of_train := length_of_train / time_to_cross -- Speed formula

theorem speed_of_train_is_20 :
  speed_of_train = 20 := by
  sorry

end speed_of_train_is_20_l451_45186


namespace Willy_more_crayons_l451_45199

theorem Willy_more_crayons (Willy Lucy : ℕ) (h1 : Willy = 1400) (h2 : Lucy = 290) : (Willy - Lucy) = 1110 :=
by
  -- proof goes here
  sorry

end Willy_more_crayons_l451_45199


namespace route_one_speed_is_50_l451_45106

noncomputable def speed_route_one (x : ℝ) : Prop :=
  let time_route_one := 75 / x
  let time_route_two := 90 / (1.8 * x)
  time_route_one = time_route_two + 1/2

theorem route_one_speed_is_50 :
  ∃ x : ℝ, speed_route_one x ∧ x = 50 :=
by
  sorry

end route_one_speed_is_50_l451_45106


namespace quadratic_function_passes_through_origin_l451_45187

theorem quadratic_function_passes_through_origin (a : ℝ) :
  ((a - 1) * 0^2 - 0 + a^2 - 1 = 0) → a = -1 :=
by
  intros h
  sorry

end quadratic_function_passes_through_origin_l451_45187


namespace melanie_gave_8_dimes_l451_45185

theorem melanie_gave_8_dimes
  (initial_dimes : ℕ)
  (additional_dimes : ℕ)
  (current_dimes : ℕ)
  (given_away_dimes : ℕ) :
  initial_dimes = 7 →
  additional_dimes = 4 →
  current_dimes = 3 →
  given_away_dimes = (initial_dimes + additional_dimes - current_dimes) →
  given_away_dimes = 8 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end melanie_gave_8_dimes_l451_45185


namespace least_large_groups_l451_45145

theorem least_large_groups (total_members : ℕ) (members_large_group : ℕ) (members_small_group : ℕ) (L : ℕ) (S : ℕ)
  (H_total : total_members = 90)
  (H_large : members_large_group = 7)
  (H_small : members_small_group = 3)
  (H_eq : total_members = L * members_large_group + S * members_small_group) :
  L = 12 :=
by
  have h1 : total_members = 90 := by exact H_total
  have h2 : members_large_group = 7 := by exact H_large
  have h3 : members_small_group = 3 := by exact H_small
  rw [h1, h2, h3] at H_eq
  -- The proof is skipped here
  sorry

end least_large_groups_l451_45145


namespace product_even_permutation_l451_45179

theorem product_even_permutation (a : Fin 2015 → ℕ) (h : ∀ i j, i ≠ j → a i ≠ a j)
    (range_a : {x // 2015 ≤ x ∧ x ≤ 4029}): 
    ∃ i, (a i - (i + 1)) % 2 = 0 :=
by
  sorry

end product_even_permutation_l451_45179


namespace intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l451_45132

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + Real.log (2 - x) + a * x

theorem intervals_of_monotonicity_a_eq_1 : 
  ∀ x : ℝ, (0 < x ∧ x < Real.sqrt 2) → 
  f x 1 < f (Real.sqrt 2) 1 ∧ 
  ∀ x : ℝ, (Real.sqrt 2 < x ∧ x < 2) → 
  f x 1 > f (Real.sqrt 2) 1 := 
sorry

theorem max_value_implies_a_half : 
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ∧ f 1 a = 1/2 → a = 1/2 := 
sorry

end intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l451_45132


namespace equivalent_expression_l451_45123

theorem equivalent_expression (a : ℝ) (h1 : a ≠ -2) (h2 : a ≠ -1) :
  ( (a^2 + a - 2) / (a^2 + 3*a + 2) * 5 * (a + 1)^2 = 5*a^2 - 5 ) :=
by {
  sorry
}

end equivalent_expression_l451_45123


namespace percentage_seats_not_taken_l451_45155

theorem percentage_seats_not_taken
  (rows : ℕ) (seats_per_row : ℕ) 
  (ticket_price : ℕ)
  (earnings : ℕ)
  (H_rows : rows = 150)
  (H_seats_per_row : seats_per_row = 10) 
  (H_ticket_price : ticket_price = 10)
  (H_earnings : earnings = 12000) :
  (1500 - (12000 / 10)) / 1500 * 100 = 20 := 
by
  sorry

end percentage_seats_not_taken_l451_45155


namespace number_of_solutions_l451_45113

open Nat

-- Definitions arising from the conditions
def is_solution (x y : ℕ) : Prop := 3 * x + 5 * y = 501

-- Statement of the problem
theorem number_of_solutions :
  (∃ k : ℕ, k ≥ 0 ∧ k < 33 ∧ ∀ (x y : ℕ), x = 5 * k + 2 ∧ y = 99 - 3 * k → is_solution x y) :=
  sorry

end number_of_solutions_l451_45113


namespace cyclic_sum_ineq_l451_45122

theorem cyclic_sum_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 / (a^2 + a * b + b^2) + b^3 / (b^2 + b * c + c^2) + c^3 / (c^2 + c * a + a^2)) 
  ≥ (1 / 3) * (a + b + c) :=
by
  sorry

end cyclic_sum_ineq_l451_45122


namespace at_least_one_greater_l451_45181

theorem at_least_one_greater (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = a * b * c) :
  a > 17 / 10 ∨ b > 17 / 10 ∨ c > 17 / 10 :=
sorry

end at_least_one_greater_l451_45181


namespace money_left_l451_45109

noncomputable def initial_amount : ℝ := 10.10
noncomputable def spent_on_sweets : ℝ := 3.25
noncomputable def amount_per_friend : ℝ := 2.20
noncomputable def remaining_amount : ℝ := initial_amount - spent_on_sweets - 2 * amount_per_friend

theorem money_left : remaining_amount = 2.45 :=
by
  sorry

end money_left_l451_45109


namespace min_value_proof_l451_45190

noncomputable def min_value (m n : ℝ) : ℝ := 
  if 4 * m + n = 1 ∧ (m > 0 ∧ n > 0) then (4 / m + 1 / n) else 0

theorem min_value_proof : ∃ m n : ℝ, 4 * m + n = 1 ∧ m > 0 ∧ n > 0 ∧ min_value m n = 25 :=
by
  -- stating the theorem conditionally 
  -- and expressing that there exists values of m and n
  sorry

end min_value_proof_l451_45190


namespace bob_time_improvement_l451_45156

def time_improvement_percent (bob_time sister_time improvement_time : ℕ) : ℕ :=
  ((improvement_time * 100) / bob_time)

theorem bob_time_improvement : 
  ∀ (bob_time sister_time : ℕ), bob_time = 640 → sister_time = 608 → 
  time_improvement_percent bob_time sister_time (bob_time - sister_time) = 5 :=
by
  intros bob_time sister_time h_bob h_sister
  rw [h_bob, h_sister]
  sorry

end bob_time_improvement_l451_45156


namespace cubes_sum_eq_ten_squared_l451_45178

theorem cubes_sum_eq_ten_squared : 1^3 + 2^3 + 3^3 + 4^3 = 10^2 := by
  sorry

end cubes_sum_eq_ten_squared_l451_45178


namespace no_real_roots_for_polynomial_l451_45162

theorem no_real_roots_for_polynomial :
  (∀ x : ℝ, x^8 - x^7 + 2*x^6 - 2*x^5 + 3*x^4 - 3*x^3 + 4*x^2 - 4*x + (5/2) ≠ 0) :=
by
  sorry

end no_real_roots_for_polynomial_l451_45162


namespace opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l451_45127

theorem opposite_neg_abs_five_minus_six : -|5 - 6| = -1 := by
  sorry

theorem opposite_of_neg_abs (h : -|5 - 6| = -1) : -(-1) = 1 := by
  sorry

theorem math_problem_proof : -(-|5 - 6|) = 1 := by
  apply opposite_of_neg_abs
  apply opposite_neg_abs_five_minus_six

end opposite_neg_abs_five_minus_six_opposite_of_neg_abs_math_problem_proof_l451_45127


namespace find_root_of_polynomial_l451_45171

theorem find_root_of_polynomial (a c x : ℝ)
  (h1 : a + c = -3)
  (h2 : 64 * a + c = 60)
  (h3 : x = 2) :
  a * x^3 - 2 * x + c = 0 :=
by
  sorry

end find_root_of_polynomial_l451_45171


namespace smallest_k_l451_45157

theorem smallest_k (k : ℕ) 
  (h1 : 201 % 24 = 9 % 24) 
  (h2 : (201 + k) % (24 + k) = (9 + k) % (24 + k)) : 
  k = 8 :=
by 
  sorry

end smallest_k_l451_45157


namespace percent_decrease_in_square_area_l451_45128

theorem percent_decrease_in_square_area (A B C D : Type) 
  (side_length_AD side_length_AB side_length_CD : ℝ) 
  (area_square_original new_side_length new_area : ℝ) 
  (h1 : side_length_AD = side_length_AB) (h2 : side_length_AD = side_length_CD) 
  (h3 : area_square_original = side_length_AD^2)
  (h4 : new_side_length = side_length_AD * 0.8)
  (h5 : new_area = new_side_length^2)
  (h6 : side_length_AD = 9) : 
  (area_square_original - new_area) / area_square_original * 100 = 36 := 
  by 
    sorry

end percent_decrease_in_square_area_l451_45128


namespace max_closable_companies_l451_45133

def number_of_planets : ℕ := 10 ^ 2015
def number_of_companies : ℕ := 2015

theorem max_closable_companies (k : ℕ) : k = 1007 :=
sorry

end max_closable_companies_l451_45133


namespace six_digit_perfect_square_l451_45175

theorem six_digit_perfect_square :
  ∃ n : ℕ, ∃ x : ℕ, (n ^ 2 = 763876) ∧ (n ^ 2 >= 100000) ∧ (n ^ 2 < 1000000) ∧ (5 ≤ x) ∧ (x < 50) ∧ (76 * 10000 + 38 * 100 + 76 = 763876) ∧ (38 = 76 / 2) :=
by
  sorry

end six_digit_perfect_square_l451_45175


namespace winning_candidate_votes_percentage_l451_45126

theorem winning_candidate_votes_percentage (P : ℝ) 
    (majority : P/100 * 6000 - (6000 - P/100 * 6000) = 1200) : 
    P = 60 := 
by 
  sorry

end winning_candidate_votes_percentage_l451_45126


namespace problem_intersection_union_complement_l451_45173

open Set Real

noncomputable def A : Set ℝ := {x | x ≥ 2}
noncomputable def B : Set ℝ := {y | y ≤ 3}

theorem problem_intersection_union_complement :
  (A ∩ B = {x | 2 ≤ x ∧ x ≤ 3}) ∧ 
  (A ∪ B = univ) ∧ 
  (compl A ∩ compl B = ∅) :=
by
  sorry

end problem_intersection_union_complement_l451_45173


namespace arccos_cos_11_equals_4_717_l451_45163

noncomputable def arccos_cos_11 : Real :=
  let n : ℤ := Int.floor (11 / (2 * Real.pi))
  Real.arccos (Real.cos 11)

theorem arccos_cos_11_equals_4_717 :
  arccos_cos_11 = 4.717 := by
  sorry

end arccos_cos_11_equals_4_717_l451_45163


namespace problem_1_problem_2_l451_45103

variable (a : ℕ → ℝ)

variables (h1 : ∀ n, 0 < a n) (h2 : ∀ n, a (n + 1) + 1 / a n < 2)

-- Prove that: (1) a_{n+2} < a_{n+1} < 2 for n ∈ ℕ*
theorem problem_1 (n : ℕ) : a (n + 2) < a (n + 1) ∧ a (n + 1) < 2 := 
sorry

-- Prove that: (2) a_n > 1 for n ∈ ℕ*
theorem problem_2 (n : ℕ) : 1 < a n := 
sorry

end problem_1_problem_2_l451_45103


namespace highest_power_of_2_divides_n_highest_power_of_3_divides_n_l451_45193

noncomputable def n : ℕ := 15^4 - 11^4

theorem highest_power_of_2_divides_n : ∃ k : ℕ, 2^4 = 16 ∧ 2^(k) ∣ n :=
by
  sorry

theorem highest_power_of_3_divides_n : ∃ m : ℕ, 3^0 = 1 ∧ 3^(m) ∣ n :=
by
  sorry

end highest_power_of_2_divides_n_highest_power_of_3_divides_n_l451_45193


namespace simplify_expr_1_l451_45116

theorem simplify_expr_1 (a : ℝ) : (2 * a - 3) ^ 2 + (2 * a + 3) * (2 * a - 3) = 8 * a ^ 2 - 12 * a :=
by
  sorry

end simplify_expr_1_l451_45116


namespace ratio_volumes_tetrahedron_octahedron_l451_45104

theorem ratio_volumes_tetrahedron_octahedron (a b : ℝ) (h_eq_areas : a^2 * (Real.sqrt 3) = 2 * b^2 * (Real.sqrt 3)) :
  (a^3 * (Real.sqrt 2) / 12) / (b^3 * (Real.sqrt 2) / 3) = 1 / Real.sqrt 2 :=
by
  sorry

end ratio_volumes_tetrahedron_octahedron_l451_45104


namespace triangle_base_value_l451_45158

variable (L R B : ℕ)

theorem triangle_base_value
    (h1 : L = 12)
    (h2 : R = L + 2)
    (h3 : L + R + B = 50) :
    B = 24 := 
sorry

end triangle_base_value_l451_45158


namespace full_price_shoes_l451_45197

variable (P : ℝ)

def full_price (P : ℝ) : ℝ := P
def discount_1_year (P : ℝ) : ℝ := 0.80 * P
def discount_3_years (P : ℝ) : ℝ := 0.75 * discount_1_year P
def price_after_discounts (P : ℝ) : ℝ := 0.60 * P

theorem full_price_shoes : price_after_discounts P = 51 → full_price P = 85 :=
by
  -- Placeholder for proof steps,
  sorry

end full_price_shoes_l451_45197


namespace participants_initial_count_l451_45120

theorem participants_initial_count 
  (x : ℕ) 
  (p1 : x * (2 : ℚ) / 5 * 1 / 4 = 30) :
  x = 300 :=
by
  sorry

end participants_initial_count_l451_45120


namespace acid_base_mixture_ratio_l451_45136

theorem acid_base_mixture_ratio (r s t : ℝ) (hr : r ≥ 0) (hs : s ≥ 0) (ht : t ≥ 0) :
  (r ≠ -1) → (s ≠ -1) → (t ≠ -1) →
  let acid_volume := (r/(r+1) + s/(s+1) + t/(t+1))
  let base_volume := (1/(r+1) + 1/(s+1) + 1/(t+1))
  acid_volume / base_volume = (rst + rt + rs + st) / (rs + rt + st + r + s + t + 3) := 
by {
  sorry
}

end acid_base_mixture_ratio_l451_45136


namespace algebra_expression_value_l451_45141

theorem algebra_expression_value (a b c : ℝ) (h1 : a - b = 3) (h2 : b + c = -5) : 
  ac - bc + a^2 - ab = -6 := by
  sorry

end algebra_expression_value_l451_45141


namespace h_even_if_g_odd_l451_45108

structure odd_function (g : ℝ → ℝ) : Prop :=
(odd : ∀ x : ℝ, g (-x) = -g x)

def h (g : ℝ → ℝ) (x : ℝ) : ℝ := abs (g (x^5))

theorem h_even_if_g_odd (g : ℝ → ℝ) (hg : odd_function g) : ∀ x : ℝ, h g x = h g (-x) :=
by
  sorry

end h_even_if_g_odd_l451_45108


namespace ratio_EG_GD_l451_45160

theorem ratio_EG_GD (a EG GD : ℝ)
  (h1 : EG = 4 * GD)
  (gcd_1 : Int.gcd 4 1 = 1) :
  4 + 1 = 5 := by
  sorry

end ratio_EG_GD_l451_45160


namespace minimum_value_l451_45180

noncomputable def f (x : ℝ) : ℝ := -2 * (Real.cos x)^2 - 2 * (Real.sin x) + 9 / 2

theorem minimum_value :
  ∃ (x : ℝ) (hx : x ∈ Set.Icc (Real.pi / 6) (2 * Real.pi / 3)), f x = 2 :=
by
  use Real.pi / 6
  sorry

end minimum_value_l451_45180


namespace clock_rings_eight_times_in_a_day_l451_45112

theorem clock_rings_eight_times_in_a_day : 
  ∀ t : ℕ, t % 3 = 1 → 0 ≤ t ∧ t < 24 → ∃ n : ℕ, n = 8 := 
by 
  sorry

end clock_rings_eight_times_in_a_day_l451_45112


namespace scientific_notation_of_153000_l451_45147

theorem scientific_notation_of_153000 :
  ∃ (a : ℝ) (n : ℤ), 153000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.53 ∧ n = 5 := 
by
  sorry

end scientific_notation_of_153000_l451_45147


namespace abs_diff_squares_105_95_l451_45168

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l451_45168


namespace initial_oranges_correct_l451_45196

-- Define constants for the conditions
def oranges_shared : ℕ := 4
def oranges_left : ℕ := 42

-- Define the initial number of oranges
def initial_oranges : ℕ := oranges_left + oranges_shared

-- The theorem to prove
theorem initial_oranges_correct : initial_oranges = 46 :=
by 
  sorry  -- Proof to be provided

end initial_oranges_correct_l451_45196


namespace least_value_is_one_l451_45170

noncomputable def least_possible_value (x y : ℝ) : ℝ := (x^2 * y - 1)^2 + (x^2 + y)^2

theorem least_value_is_one : ∀ x y : ℝ, (least_possible_value x y) ≥ 1 :=
by
  sorry

end least_value_is_one_l451_45170


namespace stratified_sampling_height_group_selection_l451_45143

theorem stratified_sampling_height_group_selection :
  let total_students := 100
  let group1 := 20
  let group2 := 50
  let group3 := 30
  let total_selected := 18
  group1 + group2 + group3 = total_students →
  (group3 : ℝ) / total_students * total_selected = 5.4 →
  round ((group3 : ℝ) / total_students * total_selected) = 3 :=
by
  intros total_students group1 group2 group3 total_selected h1 h2
  sorry

end stratified_sampling_height_group_selection_l451_45143


namespace taehyung_mom_age_l451_45195

variables (taehyung_age_diff_mom : ℕ) (taehyung_age_diff_brother : ℕ) (brother_age : ℕ)

theorem taehyung_mom_age 
  (h1 : taehyung_age_diff_mom = 31) 
  (h2 : taehyung_age_diff_brother = 5) 
  (h3 : brother_age = 7) 
  : 43 = brother_age + taehyung_age_diff_brother + taehyung_age_diff_mom := 
by 
  -- Proof goes here
  sorry

end taehyung_mom_age_l451_45195


namespace curve_is_ellipse_with_foci_on_y_axis_l451_45161

theorem curve_is_ellipse_with_foci_on_y_axis (α : ℝ) (hα : 0 < α ∧ α < 90) :
  ∃ a b : ℝ, (0 < a) ∧ (0 < b) ∧ (a < b) ∧ 
  (∀ x y : ℝ, x^2 + y^2 * (Real.cos α) = 1 ↔ (x/a)^2 + (y/b)^2 = 1) :=
sorry

end curve_is_ellipse_with_foci_on_y_axis_l451_45161


namespace salary_net_change_l451_45140

variable {S : ℝ}

theorem salary_net_change (S : ℝ) : (1.4 * S - 0.4 * (1.4 * S)) - S = -0.16 * S :=
by
  sorry

end salary_net_change_l451_45140


namespace mandy_chocolate_l451_45100

theorem mandy_chocolate (total : ℕ) (h1 : total = 60)
  (michael : ℕ) (h2 : michael = total / 2)
  (paige : ℕ) (h3 : paige = (total - michael) / 2) :
  (total - michael - paige = 15) :=
by
  -- By hypothesis: total = 60, michael = 30, paige = 15
  sorry 

end mandy_chocolate_l451_45100


namespace arithmetic_sequence_property_l451_45149

-- Define the arithmetic sequence {an}
variable {α : Type*} [LinearOrderedField α]

def is_arith_seq (a : ℕ → α) := ∃ (d : α), ∀ (n : ℕ), a (n+1) = a n + d

-- Define the condition
def given_condition (a : ℕ → α) : Prop := a 5 / a 3 = 5 / 9

-- Main theorem statement
theorem arithmetic_sequence_property (a : ℕ → α) (h : is_arith_seq a) 
  (h_condition : given_condition a) : 1 = 1 :=
by
  sorry

end arithmetic_sequence_property_l451_45149


namespace volume_of_pyramid_l451_45165

-- Define conditions
variables (x h : ℝ)
axiom x_pos : x > 0
axiom h_pos : h > 0

-- Define the main theorem/problem statement
theorem volume_of_pyramid (x h : ℝ) (x_pos : x > 0) (h_pos : h > 0) : 
  ∃ (V : ℝ), V = (1 / 6) * x^2 * h :=
by sorry

end volume_of_pyramid_l451_45165


namespace inequality_D_no_solution_l451_45172

theorem inequality_D_no_solution :
  ¬ ∃ x : ℝ, 2 - 3 * x + 2 * x^2 ≤ 0 := 
sorry

end inequality_D_no_solution_l451_45172


namespace geometric_mean_of_4_and_9_l451_45138

theorem geometric_mean_of_4_and_9 :
  ∃ b : ℝ, (4 * 9 = b^2) ∧ (b = 6 ∨ b = -6) :=
by
  sorry

end geometric_mean_of_4_and_9_l451_45138


namespace perpendicular_line_eq_l451_45153

theorem perpendicular_line_eq (a b : ℝ) (ha : 2 * a - 5 * b + 3 = 0) (hpt : a = 2 ∧ b = -1) : 
    ∃ c : ℝ, c = 5 * a + 2 * b - 8 := 
sorry

end perpendicular_line_eq_l451_45153


namespace smallest_circle_covering_region_l451_45146

/-- 
Given the conditions describing the plane region:
1. x ≥ 0
2. y ≥ 0
3. x + 2y - 4 ≤ 0

Prove that the equation of the smallest circle covering this region is (x - 2)² + (y - 1)² = 5.
-/
theorem smallest_circle_covering_region :
  (∀ (x y : ℝ), (x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y - 4 ≤ 0) → (x - 2)^2 + (y - 1)^2 ≤ 5) :=
sorry

end smallest_circle_covering_region_l451_45146


namespace eggs_per_basket_l451_45169

-- Lucas places a total of 30 blue Easter eggs in several yellow baskets
-- Lucas places a total of 42 green Easter eggs in some purple baskets
-- Each basket contains the same number of eggs
-- There are at least 5 eggs in each basket

theorem eggs_per_basket (n : ℕ) (h1 : n ∣ 30) (h2 : n ∣ 42) (h3 : n ≥ 5) : n = 6 :=
by
  sorry

end eggs_per_basket_l451_45169


namespace smallest_integer_divisibility_l451_45192

def smallest_integer (a : ℕ) : Prop :=
  a > 0 ∧ ¬ ∀ b, a = b + 1

theorem smallest_integer_divisibility :
  ∃ a, smallest_integer a ∧ gcd a 63 > 1 ∧ gcd a 66 > 1 ∧ ∀ b, smallest_integer b → b < a → gcd b 63 ≤ 1 ∨ gcd b 66 ≤ 1 :=
sorry

end smallest_integer_divisibility_l451_45192


namespace combined_work_days_l451_45121

-- Definitions for the conditions
def work_rate (days : ℕ) : ℚ := 1 / days
def combined_work_rate (days_a days_b : ℕ) : ℚ :=
  work_rate days_a + work_rate days_b

-- Theorem to prove
theorem combined_work_days (days_a days_b : ℕ) (ha : days_a = 15) (hb : days_b = 30) :
  1 / (combined_work_rate days_a days_b) = 10 :=
by
  rw [ha, hb]
  sorry

end combined_work_days_l451_45121


namespace fraction_exponentiation_and_multiplication_l451_45124

theorem fraction_exponentiation_and_multiplication :
  ( (2 : ℚ) / 3 ) ^ 3 * (1 / 4) = 2 / 27 :=
by
  sorry

end fraction_exponentiation_and_multiplication_l451_45124


namespace minimum_occupied_seats_l451_45115

theorem minimum_occupied_seats (total_seats : ℕ) (min_empty_seats : ℕ) (occupied_seats : ℕ)
  (h1 : total_seats = 150)
  (h2 : min_empty_seats = 2)
  (h3 : occupied_seats = 2 * (total_seats / (occupied_seats + min_empty_seats + min_empty_seats)))
  : occupied_seats = 74 := by
  sorry

end minimum_occupied_seats_l451_45115


namespace length_AC_l451_45150

open Real

noncomputable def net_south_north (south north : ℝ) : ℝ := south - north
noncomputable def net_east_west (east west : ℝ) : ℝ := east - west
noncomputable def distance (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

theorem length_AC :
  let A : ℝ := 0
  let south := 30
  let north := 20
  let east := 40
  let west := 35
  let net_south := net_south_north south north
  let net_east := net_east_west east west
  distance net_south net_east = 5 * sqrt 5 :=
by
  sorry

end length_AC_l451_45150


namespace fermats_little_theorem_l451_45134

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (ha : ¬ p ∣ a) :
  a^(p-1) ≡ 1 [MOD p] :=
sorry

end fermats_little_theorem_l451_45134


namespace value_of_t5_l451_45114

noncomputable def t_5_value (t1 t2 : ℚ) (r : ℚ) (a : ℚ) : ℚ := a * r^4

theorem value_of_t5 
  (a r : ℚ)
  (h1 : a > 0)  -- condition: each term is positive
  (h2 : a + a * r = 15 / 2)  -- condition: sum of first two terms is 15/2
  (h3 : a^2 + (a * r)^2 = 153 / 4)  -- condition: sum of squares of first two terms is 153/4
  (h4 : r > 0)  -- ensuring positivity of r
  (h5 : r < 1)  -- ensuring t1 > t2
  : t_5_value a (a * r) r a = 3 / 128 :=
sorry

end value_of_t5_l451_45114


namespace segment_lengths_l451_45119

theorem segment_lengths (AB BC CD DE EF : ℕ) 
  (h1 : AB > BC)
  (h2 : BC > CD)
  (h3 : CD > DE)
  (h4 : DE > EF)
  (h5 : AB = 2 * EF)
  (h6 : AB + BC + CD + DE + EF = 53) :
  (AB, BC, CD, DE, EF) = (14, 12, 11, 9, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 11, 8, 7) ∨
  (AB, BC, CD, DE, EF) = (14, 13, 10, 9, 7) :=
sorry

end segment_lengths_l451_45119


namespace find_polynomial_value_l451_45107

theorem find_polynomial_value
  (x y : ℝ)
  (h1 : 3 * x + y = 5)
  (h2 : x + 3 * y = 6) :
  5 * x^2 + 8 * x * y + 5 * y^2 = 61 := 
by {
  -- The proof part is omitted here
  sorry
}

end find_polynomial_value_l451_45107


namespace length_of_bridge_l451_45188

theorem length_of_bridge
  (train_length : ℝ)
  (crossing_time : ℝ)
  (train_speed_kmph : ℝ)
  (conversion_factor : ℝ)
  (bridge_length : ℝ) :
  train_length = 100 →
  crossing_time = 12 →
  train_speed_kmph = 120 →
  conversion_factor = 1 / 3.6 →
  bridge_length = 299.96 :=
by
  sorry

end length_of_bridge_l451_45188


namespace part1_part2_l451_45139

theorem part1 (x p : ℝ) (h : abs p ≤ 2) : (x^2 + p * x + 1 > 2 * x + p) ↔ (x < -1 ∨ 3 < x) := 
by 
  sorry

theorem part2 (x p : ℝ) (h1 : 2 ≤ x) (h2 : x ≤ 4) : (x^2 + p * x + 1 > 2 * x + p) ↔ (-1 < p) := 
by 
  sorry

end part1_part2_l451_45139


namespace avg_bc_eq_28_l451_45182

variable (A B C : ℝ)

-- Conditions
def avg_abc_eq_30 : Prop := (A + B + C) / 3 = 30
def avg_ab_eq_25 : Prop := (A + B) / 2 = 25
def b_eq_16 : Prop := B = 16

-- The Proved Statement
theorem avg_bc_eq_28 (h1 : avg_abc_eq_30 A B C) (h2 : avg_ab_eq_25 A B) (h3 : b_eq_16 B) : (B + C) / 2 = 28 := 
by
  sorry

end avg_bc_eq_28_l451_45182


namespace lance_pennies_saved_l451_45177

theorem lance_pennies_saved :
  let a := 5
  let d := 2
  let n := 20
  let a_n := a + (n - 1) * d
  let S_n := n * (a + a_n) / 2
  S_n = 480 :=
by
  sorry

end lance_pennies_saved_l451_45177


namespace converse_of_x_eq_one_implies_x_squared_eq_one_l451_45183

theorem converse_of_x_eq_one_implies_x_squared_eq_one (x : ℝ) : x^2 = 1 → x = 1 := 
sorry

end converse_of_x_eq_one_implies_x_squared_eq_one_l451_45183


namespace find_fifth_day_sales_l451_45154

-- Define the variables and conditions
variables (x : ℝ)
variables (a : ℝ := 100) (b : ℝ := 92) (c : ℝ := 109) (d : ℝ := 96) (f : ℝ := 96) (g : ℝ := 105)
variables (mean : ℝ := 100.1)

-- Define the mean condition which leads to the proof of x
theorem find_fifth_day_sales : (a + b + c + d + x + f + g) / 7 = mean → x = 102.7 := by
  intro h
  -- Proof goes here
  sorry

end find_fifth_day_sales_l451_45154


namespace largest_possible_cupcakes_without_any_ingredients_is_zero_l451_45130

-- Definitions of properties of the cupcakes
def total_cupcakes : ℕ := 60
def blueberries (n : ℕ) : Prop := n = total_cupcakes / 3
def sprinkles (n : ℕ) : Prop := n = total_cupcakes / 4
def frosting (n : ℕ) : Prop := n = total_cupcakes / 2
def pecans (n : ℕ) : Prop := n = total_cupcakes / 5

-- Theorem statement
theorem largest_possible_cupcakes_without_any_ingredients_is_zero :
  ∃ n, blueberries n ∧ sprinkles n ∧ frosting n ∧ pecans n → n = 0 := 
sorry

end largest_possible_cupcakes_without_any_ingredients_is_zero_l451_45130


namespace tanner_savings_in_october_l451_45129

theorem tanner_savings_in_october 
    (sept_savings : ℕ := 17) 
    (nov_savings : ℕ := 25)
    (spent : ℕ := 49) 
    (left : ℕ := 41) 
    (X : ℕ) 
    (h : sept_savings + X + nov_savings - spent = left) 
    : X = 48 :=
by
  sorry

end tanner_savings_in_october_l451_45129


namespace number_of_green_balls_l451_45184

-- Define the problem statement and conditions
def total_balls : ℕ := 12
def probability_both_green (g : ℕ) : ℚ := (g / 12) * ((g - 1) / 11)

-- The main theorem statement
theorem number_of_green_balls (g : ℕ) (h : probability_both_green g = 1 / 22) : g = 3 :=
sorry

end number_of_green_balls_l451_45184
