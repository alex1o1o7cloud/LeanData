import Mathlib

namespace election_winning_percentage_l790_79068

def total_votes (a b c : ℕ) : ℕ := a + b + c

def winning_percentage (votes_winning : ℕ) (total : ℕ) : ℚ :=
(votes_winning * 100 : ℚ) / total

theorem election_winning_percentage (a b c : ℕ) (h_votes : a = 6136 ∧ b = 7636 ∧ c = 11628) :
  winning_percentage c (total_votes a b c) = 45.78 := by
  sorry

end election_winning_percentage_l790_79068


namespace product_pattern_l790_79064

theorem product_pattern (m n : ℝ) : 
  m * n = ( ( m + n ) / 2 ) ^ 2 - ( ( m - n ) / 2 ) ^ 2 := 
by 
  sorry

end product_pattern_l790_79064


namespace solution_set_ineq_l790_79056

theorem solution_set_ineq (x : ℝ) :
  x * (2 * x^2 - 3 * x + 1) ≤ 0 ↔ (x ≤ 0 ∨ (1/2 ≤ x ∧ x ≤ 1)) :=
sorry

end solution_set_ineq_l790_79056


namespace residue_n_mod_17_l790_79054

noncomputable def satisfies_conditions (m n k : ℕ) : Prop :=
  m^2 + 1 = 2 * n^2 ∧ 2 * m^2 + 1 = 11 * k^2 

theorem residue_n_mod_17 (m n k : ℕ) (h : satisfies_conditions m n k) : n % 17 = 5 :=
  sorry

end residue_n_mod_17_l790_79054


namespace problem_statement_l790_79029

theorem problem_statement (a b : ℝ) (h : a ≠ b) : (a - b) ^ 2 > 0 := sorry

end problem_statement_l790_79029


namespace inequality_solution_l790_79027

theorem inequality_solution : {x : ℝ | -2 < (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) ∧ (x^2 - 12 * x + 20) / (x^2 - 4 * x + 8) < 2} = {x : ℝ | 5 < x} := 
sorry

end inequality_solution_l790_79027


namespace find_a_l790_79089

-- Define the function f given a parameter a
def f (x a : ℝ) : ℝ := x^3 - 3*x^2 + a

-- Condition: f(x+1) is an odd function
theorem find_a (a : ℝ) (h : ∀ x : ℝ, f (-(x+1)) a = -f (x+1) a) : a = 2 := 
sorry

end find_a_l790_79089


namespace min_cost_correct_l790_79074

noncomputable def min_cost_to_feed_group : ℕ :=
  let main_courses := 50
  let salads := 30
  let soups := 15
  let price_salad := 200
  let price_soup_main := 350
  let price_salad_main := 350
  let price_all_three := 500
  17000

theorem min_cost_correct : min_cost_to_feed_group = 17000 :=
by
  sorry

end min_cost_correct_l790_79074


namespace residual_at_sample_point_l790_79050

theorem residual_at_sample_point :
  ∀ (x y : ℝ), (8 * x - 70 = 10) → (x = 10) → (y = 13) → (13 - (8 * x - 70) = 3) :=
by
  intros x y h1 h2 h3
  sorry

end residual_at_sample_point_l790_79050


namespace odot_subtraction_l790_79071

-- Define the new operation
def odot (a b : ℚ) : ℚ := (a^3) / (b^2)

-- State the theorem
theorem odot_subtraction :
  ((odot (odot 2 4) 6) - (odot 2 (odot 4 6)) = -81 / 32) :=
by
  sorry

end odot_subtraction_l790_79071


namespace area_ratio_proof_l790_79099

noncomputable def area_ratio (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) : ℝ := 
  (a * b) / (c * d)

theorem area_ratio_proof (a b c d : ℝ) (h1 : a / c = 2 / 3) (h2 : b / d = 2 / 3) :
  area_ratio a b c d h1 h2 = 4 / 9 := by
  sorry

end area_ratio_proof_l790_79099


namespace find_common_ratio_l790_79078

-- Define the variables and constants involved.
variables (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ)

-- Define the conditions of the problem.
def is_geometric_sequence := ∀ n, a (n + 1) = q * a n
def sum_of_first_n_terms := ∀ n, S n = a 0 * (1 - q^(n + 1)) / (1 - q)
def condition1 := a 5 = 4 * S 4 + 3
def condition2 := a 6 = 4 * S 5 + 3

-- The main statement that needs to be proved.
theorem find_common_ratio
  (h1: is_geometric_sequence a q)
  (h2: sum_of_first_n_terms a S q)
  (h3: condition1 a S)
  (h4: condition2 a S) : 
  q = 5 :=
sorry -- proof to be provided

end find_common_ratio_l790_79078


namespace goshawk_nature_reserve_l790_79053

-- Define the problem statement and conditions
def percent_hawks (H W K : ℝ) : Prop :=
  ∃ H W K : ℝ,
    -- Condition 1: 35% of the birds are neither hawks, paddyfield-warblers, nor kingfishers
    1 - (H + W + K) = 0.35 ∧
    -- Condition 2: 40% of the non-hawks are paddyfield-warblers
    W = 0.40 * (1 - H) ∧
    -- Condition 3: There are 25% as many kingfishers as paddyfield-warblers
    K = 0.25 * W ∧
    -- Given all conditions, calculate the percentage of hawks
    H = 0.65

theorem goshawk_nature_reserve :
  ∃ H W K : ℝ,
    1 - (H + W + K) = 0.35 ∧
    W = 0.40 * (1 - H) ∧
    K = 0.25 * W ∧
    H = 0.65 := by
    -- Proof is omitted
    sorry

end goshawk_nature_reserve_l790_79053


namespace total_packs_l790_79041

theorem total_packs (cards_bought : ℕ) (cards_per_pack : ℕ) (num_people : ℕ)
  (h1 : cards_bought = 540) (h2 : cards_per_pack = 20) (h3 : num_people = 4) :
  (cards_bought / cards_per_pack) * num_people = 108 :=
by
  sorry

end total_packs_l790_79041


namespace mark_height_feet_l790_79001

theorem mark_height_feet
  (mark_height_inches : ℕ)
  (mike_height_feet : ℕ)
  (mike_height_inches : ℕ)
  (mike_taller_than_mark : ℕ)
  (foot_in_inches : ℕ)
  (mark_height_eq : mark_height_inches = 3)
  (mike_height_eq : mike_height_feet * foot_in_inches + mike_height_inches = 73)
  (mike_taller_eq : mike_height_feet * foot_in_inches + mike_height_inches = mark_height_inches + mike_taller_than_mark)
  (foot_in_inches_eq : foot_in_inches = 12) :
  mark_height_inches = 63 ∧ mark_height_inches / foot_in_inches = 5 := by
sorry

end mark_height_feet_l790_79001


namespace tangent_line_to_circle_l790_79059

theorem tangent_line_to_circle : 
  ∀ (ρ θ : ℝ), (ρ = 4 * Real.sin θ) → (∃ ρ θ : ℝ, ρ * Real.cos θ = 2) :=
by
  sorry

end tangent_line_to_circle_l790_79059


namespace calculate_rate_l790_79011

-- Definitions corresponding to the conditions in the problem
def bankers_gain (td : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  td * rate * time

-- Given values according to the problem
def BG : ℝ := 7.8
def TD : ℝ := 65
def Time : ℝ := 1
def expected_rate_percentage : ℝ := 12

-- The mathematical proof problem statement in Lean 4
theorem calculate_rate : (BG = bankers_gain TD (expected_rate_percentage / 100) Time) :=
sorry

end calculate_rate_l790_79011


namespace inequality_range_l790_79006

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, |2 * x + 1| - |2 * x - 1| < a) → a > 2 :=
by
  sorry

end inequality_range_l790_79006


namespace prod_lcm_gcd_eq_216_l790_79015

theorem prod_lcm_gcd_eq_216 (a b : ℕ) (h1 : a = 12) (h2 : b = 18) :
  (Nat.gcd a b) * (Nat.lcm a b) = 216 := by
  sorry

end prod_lcm_gcd_eq_216_l790_79015


namespace train_passes_bridge_in_52_seconds_l790_79005

def length_of_train : ℕ := 510
def speed_of_train_kmh : ℕ := 45
def length_of_bridge : ℕ := 140
def total_distance := length_of_train + length_of_bridge
def speed_of_train_ms := speed_of_train_kmh * 1000 / 3600
def time_to_pass_bridge := total_distance / speed_of_train_ms

theorem train_passes_bridge_in_52_seconds :
  time_to_pass_bridge = 52 := sorry

end train_passes_bridge_in_52_seconds_l790_79005


namespace number_whose_square_is_64_l790_79093

theorem number_whose_square_is_64 (x : ℝ) (h : x^2 = 64) : x = 8 ∨ x = -8 :=
sorry

end number_whose_square_is_64_l790_79093


namespace quadratic_inequality_solution_l790_79036

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - 2 * k + 4 < 0) ↔ (-6 < k ∧ k < 2) :=
by
  sorry

end quadratic_inequality_solution_l790_79036


namespace instantaneous_velocity_at_t4_l790_79086

def position (t : ℝ) : ℝ := t^2 - t + 2

theorem instantaneous_velocity_at_t4 : 
  (deriv position 4) = 7 := 
by
  sorry

end instantaneous_velocity_at_t4_l790_79086


namespace employees_excluding_manager_l790_79000

theorem employees_excluding_manager (E : ℕ) (avg_salary_employee : ℕ) (manager_salary : ℕ) (new_avg_salary : ℕ) (total_employees_with_manager : ℕ) :
  avg_salary_employee = 1800 →
  manager_salary = 4200 →
  new_avg_salary = avg_salary_employee + 150 →
  total_employees_with_manager = E + 1 →
  (1800 * E + 4200) / total_employees_with_manager = new_avg_salary →
  E = 15 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end employees_excluding_manager_l790_79000


namespace train_length_l790_79035

theorem train_length (v_train_kmph : ℝ) (v_man_kmph : ℝ) (time_sec : ℝ) 
  (h1 : v_train_kmph = 25) 
  (h2 : v_man_kmph = 2) 
  (h3 : time_sec = 20) : 
  (150 : ℝ) = (v_train_kmph + v_man_kmph) * (1000 / 3600) * time_sec := 
by {
  -- sorry for the steps here
  sorry
}

end train_length_l790_79035


namespace find_x_satisfying_conditions_l790_79007

theorem find_x_satisfying_conditions :
  ∃ x : ℕ, (x % 2 = 1) ∧ (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) ∧ x = 59 :=
by
  sorry

end find_x_satisfying_conditions_l790_79007


namespace binomial_60_3_l790_79072

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end binomial_60_3_l790_79072


namespace problem_solution_l790_79049

theorem problem_solution
  (x : ℝ) (a b : ℕ) (hx_pos : 0 < x) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_eq : x ^ 2 + 5 * x + 5 / x + 1 / x ^ 2 = 40)
  (h_form : x = a + Real.sqrt b) :
  a + b = 11 :=
sorry

end problem_solution_l790_79049


namespace correct_random_variable_l790_79042

-- Define the given conditions
def total_white_balls := 5
def total_red_balls := 3
def total_balls := total_white_balls + total_red_balls
def balls_drawn := 3

-- Define the random variable
noncomputable def is_random_variable_correct (option : ℕ) :=
  option = 2

-- The theorem to be proved
theorem correct_random_variable: is_random_variable_correct 2 :=
by
  sorry

end correct_random_variable_l790_79042


namespace find_a_l790_79096

theorem find_a (a : ℝ) (h : (3 * a + 2) + (a + 14) = 0) : a = -4 :=
sorry

end find_a_l790_79096


namespace remaining_surface_area_correct_l790_79039

noncomputable def remaining_surface_area (a : ℕ) (c : ℕ) : ℕ :=
  let original_surface_area := 6 * a^2
  let corner_cube_area := 3 * c^2
  let net_change := corner_cube_area - corner_cube_area
  original_surface_area + 8 * net_change 

theorem remaining_surface_area_correct :
  remaining_surface_area 4 1 = 96 := by
  sorry

end remaining_surface_area_correct_l790_79039


namespace julias_change_l790_79060

theorem julias_change :
  let snickers := 2
  let mms := 3
  let cost_snickers := 1.5
  let cost_mms := 2 * cost_snickers
  let money_given := 2 * 10
  let total_cost := snickers * cost_snickers + mms * cost_mms
  let change := money_given - total_cost
  change = 8 :=
by
  sorry

end julias_change_l790_79060


namespace like_terms_mn_l790_79019

theorem like_terms_mn (m n : ℕ) (h1 : -2 * x^m * y^2 = 2 * x^3 * y^n) : m * n = 6 :=
by {
  -- Add the statements transforming the assumptions into intermediate steps
  sorry
}

end like_terms_mn_l790_79019


namespace find_x_l790_79079

variable (A B : Set ℕ)
variable (x : ℕ)

theorem find_x (hA : A = {1, 3}) (hB : B = {2, x}) (hUnion : A ∪ B = {1, 2, 3, 4}) : x = 4 := by
  sorry

end find_x_l790_79079


namespace rainfall_difference_l790_79051

theorem rainfall_difference :
  let day1 := 26
  let day2 := 34
  let day3 := day2 - 12
  let total_rainfall := day1 + day2 + day3
  let average_rainfall := 140
  (average_rainfall - total_rainfall = 58) :=
by
  sorry

end rainfall_difference_l790_79051


namespace xy_yz_zx_value_l790_79070

theorem xy_yz_zx_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^2 + x * y + y^2 = 9) (h2 : y^2 + y * z + z^2 = 16) (h3 : z^2 + z * x + x^2 = 25) :
  x * y + y * z + z * x = 8 * Real.sqrt 3 :=
by sorry

end xy_yz_zx_value_l790_79070


namespace meaningful_expression_range_l790_79075

theorem meaningful_expression_range (x : ℝ) (h1 : 3 * x + 2 ≥ 0) (h2 : x ≠ 0) : 
  x ∈ Set.Ico (-2 / 3) 0 ∪ Set.Ioi 0 := 
  sorry

end meaningful_expression_range_l790_79075


namespace find_m_l790_79095

theorem find_m (m : ℤ) (h₀ : 0 ≤ m) (h₁ : m < 31) (h₂ : 79453 % 31 = m) : m = 0 :=
by
  sorry

end find_m_l790_79095


namespace third_month_sale_l790_79057

theorem third_month_sale (s1 s2 s4 s5 s6 avg_sale: ℕ) (h1: s1 = 5420) (h2: s2 = 5660) (h3: s4 = 6350) (h4: s5 = 6500) (h5: s6 = 8270) (h6: avg_sale = 6400) :
  ∃ s3: ℕ, s3 = 6200 :=
by
  sorry

end third_month_sale_l790_79057


namespace line_through_point_hyperbola_l790_79003

theorem line_through_point_hyperbola {x y k : ℝ} : 
  (∃ k : ℝ, ∃ x y : ℝ, y = k * (x - 3) ∧ x^2 / 4 - y^2 = 1 ∧ (1 - 4 * k^2) = 0) → 
  (∃! k : ℝ, (k = 1 / 2) ∨ (k = -1 / 2)) := 
sorry

end line_through_point_hyperbola_l790_79003


namespace prove_inequality_l790_79094

theorem prove_inequality (x : ℝ) (h : x > 2) : x + 1 / (x - 2) ≥ 4 :=
  sorry

end prove_inequality_l790_79094


namespace sum_of_squares_l790_79067

theorem sum_of_squares (k₁ k₂ k₃ : ℝ)
  (h_sum : k₁ + k₂ + k₃ = 1) : k₁^2 + k₂^2 + k₃^2 ≥ 1/3 :=
by sorry

end sum_of_squares_l790_79067


namespace total_pages_in_book_l790_79013

theorem total_pages_in_book (pages_monday pages_tuesday total_pages_read total_pages_book : ℝ)
    (h1 : pages_monday = 15.5)
    (h2 : pages_tuesday = 1.5 * pages_monday + 16)
    (h3 : total_pages_read = pages_monday + pages_tuesday)
    (h4 : total_pages_book = 2 * total_pages_read) :
    total_pages_book = 109.5 :=
by
  sorry

end total_pages_in_book_l790_79013


namespace gnuff_tutoring_minutes_l790_79088

theorem gnuff_tutoring_minutes 
  (flat_rate : ℕ) 
  (rate_per_minute : ℕ) 
  (total_paid : ℕ) :
  flat_rate = 20 → 
  rate_per_minute = 7 →
  total_paid = 146 → 
  ∃ minutes : ℕ, minutes = 18 ∧ flat_rate + rate_per_minute * minutes = total_paid := 
by
  -- Assume the necessary steps here
  sorry

end gnuff_tutoring_minutes_l790_79088


namespace find_ab_average_l790_79010

variable (a b c k : ℝ)

-- Conditions
def sum_condition : Prop := (4 + 6 + 8 + 12 + a + b + c) / 7 = 20
def abc_condition : Prop := a + b + c = 3 * ((4 + 6 + 8) / 3)

-- Theorem
theorem find_ab_average 
  (sum_cond : sum_condition a b c) 
  (abc_cond : abc_condition a b c) 
  (c_eq_k : c = k) : 
  (a + b) / 2 = (18 - k) / 2 :=
sorry  -- Proof is omitted


end find_ab_average_l790_79010


namespace chess_team_selection_l790_79020

theorem chess_team_selection:
  let boys := 10
  let girls := 12
  let team_size := 8     -- total team size
  let boys_selected := 5 -- number of boys to select
  let girls_selected := 3 -- number of girls to select
  ∃ (w : ℕ), 
  (w = Nat.choose boys boys_selected * Nat.choose girls girls_selected) ∧ 
  w = 55440 :=
by
  sorry

end chess_team_selection_l790_79020


namespace inverse_square_relationship_l790_79043

theorem inverse_square_relationship (k : ℝ) (y : ℝ) (h1 : ∀ x y, x = k / y^2)
  (h2 : ∃ y, 1 = k / y^2) (h3 : 0.5625 = k / 4^2) :
  ∃ y, 1 = 9 / y^2 ∧ y = 3 :=
by
  sorry

end inverse_square_relationship_l790_79043


namespace calculate_value_l790_79083

theorem calculate_value : (2200 - 2090)^2 / (144 + 25) = 64 := 
by
  sorry

end calculate_value_l790_79083


namespace find_y_l790_79028

theorem find_y (x y : ℝ) (h1 : 3 * x + 2 = 2) (h2 : y - x = 2) : y = 2 :=
by
  sorry

end find_y_l790_79028


namespace determine_C_cards_l790_79058

-- Define the card numbers
def card_numbers : List ℕ := [1,2,3,4,5,6,7,8,9,10,11,12]

-- Define the card sum each person should have
def card_sum := 26

-- Define person's cards
def A_cards : List ℕ := [10, 12]
def B_cards : List ℕ := [6, 11]

-- Define sum constraints for A and B
def sum_A := A_cards.sum
def sum_B := B_cards.sum

-- Define C's complete set of numbers based on remaining cards and sum constraints
def remaining_cards := card_numbers.diff (A_cards ++ B_cards)
def sum_remaining := remaining_cards.sum

theorem determine_C_cards :
  (sum_A + (26 - sum_A)) = card_sum ∧
  (sum_B + (26 - sum_B)) = card_sum ∧
  (sum_remaining = card_sum) → 
  (remaining_cards = [8, 9]) :=
by
  sorry

end determine_C_cards_l790_79058


namespace find_n_l790_79063

theorem find_n (x : ℝ) (n : ℝ)
  (h1 : Real.log (Real.sin x) + Real.log (Real.cos x) = -2)
  (h2 : Real.log (Real.sin x + Real.cos x) = 1 / 2 * (Real.log n - 2)) :
  n = Real.exp 2 + 2 :=
by
  sorry

end find_n_l790_79063


namespace augmented_matrix_solution_l790_79038

theorem augmented_matrix_solution (c₁ c₂ : ℝ) (x y : ℝ) 
  (h1 : 2 * x + 3 * y = c₁) (h2 : 3 * x + 2 * y = c₂)
  (hx : x = 2) (hy : y = 1) : c₁ - c₂ = -1 := 
by
  sorry

end augmented_matrix_solution_l790_79038


namespace tel_aviv_rain_probability_l790_79069

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l790_79069


namespace greatest_value_product_l790_79098

def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def divisible_by (m n : ℕ) : Prop := ∃ k, m = k * n

theorem greatest_value_product (a b : ℕ) : 
    is_prime a → is_prime b → a < 10 → b < 10 → divisible_by (110 + 10 * a + b) 55 → a * b = 15 :=
by
    sorry

end greatest_value_product_l790_79098


namespace r_iterated_six_times_l790_79040

def r (θ : ℚ) : ℚ := 1 / (1 - 2 * θ)

theorem r_iterated_six_times (θ : ℚ) : r (r (r (r (r (r θ))))) = θ :=
by sorry

example : r (r (r (r (r (r 10))))) = 10 :=
by rw [r_iterated_six_times 10]

end r_iterated_six_times_l790_79040


namespace union_M_N_inter_complement_M_N_union_complement_M_N_l790_79023

open Set

variable (U : Set ℝ) (M : Set ℝ) (N : Set ℝ)

noncomputable def universal_set := U = univ

def set_M := M = {x : ℝ | x ≤ 3}
def set_N := N = {x : ℝ | x < 1}

theorem union_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    M ∪ N = {x : ℝ | x ≤ 3} :=
by sorry

theorem inter_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∩ N = ∅ :=
by sorry

theorem union_complement_M_N (hU : universal_set U) (hM : set_M M) (hN : set_N N) :
    (U \ M) ∪ (U \ N) = {x : ℝ | x ≥ 1} :=
by sorry

end union_M_N_inter_complement_M_N_union_complement_M_N_l790_79023


namespace problem1_problem2_l790_79085

-- Definition of the function f(x)
def f (x a : ℝ) : ℝ := |2 * x - a| + |2 * x - 1|

-- 1st problem: Prove the solution set for f(x) ≤ 2 when a = -1 is { x | x = ± 1/2 }
theorem problem1 : (∀ x : ℝ, f x (-1) ≤ 2 ↔ x = 1/2 ∨ x = -1/2) :=
by sorry

-- 2nd problem: Prove the range of real number a is [0, 3]
theorem problem2 : (∃ a : ℝ, (∀ x ∈ Set.Icc (1/2:ℝ) 1, f x a ≤ |2 * x + 1| ) ↔ 0 ≤ a ∧ a ≤ 3) :=
by sorry

end problem1_problem2_l790_79085


namespace all_non_positive_l790_79065

theorem all_non_positive (n : ℕ) (a : ℕ → ℤ) 
  (h₀ : a 0 = 0) 
  (hₙ : a n = 0) 
  (ineq : ∀ k, 1 ≤ k ∧ k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) : ∀ k, a k ≤ 0 :=
by 
  sorry

end all_non_positive_l790_79065


namespace subtraction_identity_l790_79014

theorem subtraction_identity : 4444444444444 - 2222222222222 - 444444444444 = 1777777777778 :=
  by norm_num

end subtraction_identity_l790_79014


namespace percentage_calculation_l790_79018

theorem percentage_calculation :
  let total_amt := 1600
  let pct_25 := 0.25 * total_amt
  let pct_5 := 0.05 * pct_25
  pct_5 = 20 := by
sorry

end percentage_calculation_l790_79018


namespace total_hair_cut_l790_79033

-- Define the amounts cut on two consecutive days
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- Statement: Prove that the total amount cut off is 0.875 inches
theorem total_hair_cut : first_cut + second_cut = 0.875 :=
by {
  -- The exact proof would go here
  sorry
}

end total_hair_cut_l790_79033


namespace dartboard_central_angle_l790_79037

theorem dartboard_central_angle (A : ℝ) (x : ℝ) (P : ℝ) (h1 : P = 1 / 4) 
    (h2 : A > 0) : (x / 360 = 1 / 4) -> x = 90 :=
by
  sorry

end dartboard_central_angle_l790_79037


namespace total_weight_proof_l790_79045

-- Definitions of the variables and conditions given in the problem
variable (M D C : ℕ)
variable (h1 : D + C = 60)  -- Daughter and grandchild together weigh 60 kg
variable (h2 : C = 1 / 5 * M)  -- Grandchild's weight is 1/5th of grandmother's weight
variable (h3 : D = 42)  -- Daughter's weight is 42 kg

-- The goal is to prove the total weight is 150 kg
theorem total_weight_proof (M D C : ℕ) (h1 : D + C = 60) (h2 : C = 1 / 5 * M) (h3 : D = 42) :
  M + D + C = 150 :=
by
  sorry

end total_weight_proof_l790_79045


namespace father_age_is_32_l790_79048

noncomputable def father_age (D F : ℕ) : Prop :=
  F = 4 * D ∧ (F + 5) + (D + 5) = 50

theorem father_age_is_32 (D F : ℕ) (h : father_age D F) : F = 32 :=
by
  sorry

end father_age_is_32_l790_79048


namespace probability_adjacent_A_before_B_l790_79092

theorem probability_adjacent_A_before_B 
  (total_students : ℕ)
  (A B C D : ℚ)
  (hA : total_students = 8)
  (hB : B = 1/3) : 
  (∃ prob : ℚ, prob = 1/3) :=
by
  sorry

end probability_adjacent_A_before_B_l790_79092


namespace area_of_shaded_region_l790_79012

def parallelogram_exists (EFGH : Type) : Prop :=
  ∃ (E F G H : EFGH) (EJ JH EH : ℝ) (height : ℝ), EJ + JH = EH ∧ EH = 12 ∧ JH = 8 ∧ height = 10

theorem area_of_shaded_region {EFGH : Type} (h : parallelogram_exists EFGH) : 
  ∃ (area_shaded : ℝ), area_shaded = 100 := 
by
  sorry

end area_of_shaded_region_l790_79012


namespace sum_on_simple_interest_is_1400_l790_79066

noncomputable def sum_placed_on_simple_interest : ℝ :=
  let P_c := 4000
  let r := 0.10
  let n := 1
  let t_c := 2
  let t_s := 3
  let A := P_c * (1 + r / n)^(n * t_c)
  let CI := A - P_c
  let SI := CI / 2
  100 * SI / (r * t_s)

theorem sum_on_simple_interest_is_1400 : sum_placed_on_simple_interest = 1400 := by
  sorry

end sum_on_simple_interest_is_1400_l790_79066


namespace squat_percentage_loss_l790_79008

variable (original_squat : ℕ)
variable (original_bench : ℕ)
variable (original_deadlift : ℕ)
variable (lost_deadlift : ℕ)
variable (new_total : ℕ)
variable (unchanged_bench : ℕ)

theorem squat_percentage_loss
  (h1 : original_squat = 700)
  (h2 : original_bench = 400)
  (h3 : original_deadlift = 800)
  (h4 : lost_deadlift = 200)
  (h5 : new_total = 1490)
  (h6 : unchanged_bench = 400) :
  (original_squat - (new_total - (unchanged_bench + (original_deadlift - lost_deadlift)))) * 100 / original_squat = 30 :=
by sorry

end squat_percentage_loss_l790_79008


namespace star_computation_l790_79076

-- Define the operation ☆
def star (m n : Int) := m^2 - m * n + n

-- Define the main proof problem
theorem star_computation :
  star 3 4 = 1 ∧ star (-1) (star 2 (-3)) = 15 := 
by
  sorry

end star_computation_l790_79076


namespace clara_biked_more_l790_79004

def clara_speed : ℕ := 18
def denise_speed : ℕ := 16
def race_duration : ℕ := 5

def clara_distance := clara_speed * race_duration
def denise_distance := denise_speed * race_duration
def distance_difference := clara_distance - denise_distance

theorem clara_biked_more : distance_difference = 10 := by
  sorry

end clara_biked_more_l790_79004


namespace matrix_product_is_correct_l790_79077

-- Define the matrices A and B
def A : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![3, 1, 1],
  ![2, 1, 2],
  ![1, 2, 3]
]

def B : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![1, 1, -1],
  ![2, -1, 1],
  ![1, 0, 1]
]

-- Define the expected product matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![6, 2, -1],
  ![6, 1, 1],
  ![8, -1, 4]
]

-- The statement of the problem
theorem matrix_product_is_correct : (A * B) = C := by
  sorry -- Proof is omitted as per instructions

end matrix_product_is_correct_l790_79077


namespace sum_lent_is_1000_l790_79026

theorem sum_lent_is_1000
    (P : ℝ)
    (r : ℝ)
    (t : ℝ)
    (I : ℝ)
    (h1 : r = 5)
    (h2 : t = 5)
    (h3 : I = P - 750)
    (h4 : I = P * r * t / 100) :
  P = 1000 :=
by sorry

end sum_lent_is_1000_l790_79026


namespace shooter_mean_hits_l790_79021

theorem shooter_mean_hits (p : ℝ) (n : ℕ) (h_prob : p = 0.9) (h_shots : n = 10) : n * p = 9 := by
  sorry

end shooter_mean_hits_l790_79021


namespace sphere_surface_area_l790_79034

theorem sphere_surface_area (V : ℝ) (hV : V = 72 * Real.pi) : 
  ∃ S : ℝ, S = 36 * 2^(2/3) * Real.pi :=
by
  sorry

end sphere_surface_area_l790_79034


namespace min_n_for_factorization_l790_79002

theorem min_n_for_factorization (n : ℤ) :
  (∃ A B : ℤ, 6 * A * B = 60 ∧ n = 6 * B + A) → n = 66 :=
sorry

end min_n_for_factorization_l790_79002


namespace intersection_A_B_l790_79062

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | 2 * x^2 - x - 1 < 0}
def B : Set ℝ := {x : ℝ | Real.log x / Real.log (1/2) < 3}

-- Define the intersection A ∩ B and state the theorem
theorem intersection_A_B : A ∩ B = {x : ℝ | 1/8 < x ∧ x < 1} := by
   sorry

end intersection_A_B_l790_79062


namespace smallest_x_for_max_f_l790_79097

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 4) + Real.sin (x / 12)

theorem smallest_x_for_max_f : ∃ x > 0, f x = 2 ∧ ∀ y > 0, (f y = 2 → y ≥ x) :=
sorry

end smallest_x_for_max_f_l790_79097


namespace number_of_flower_sets_l790_79084

theorem number_of_flower_sets (total_flowers : ℕ) (flowers_per_set : ℕ) (sets : ℕ) 
  (h1 : total_flowers = 270) 
  (h2 : flowers_per_set = 90) 
  (h3 : sets = total_flowers / flowers_per_set) : 
  sets = 3 := 
by 
  sorry

end number_of_flower_sets_l790_79084


namespace asymptotes_tangent_to_circle_l790_79073

theorem asymptotes_tangent_to_circle {m : ℝ} (hm : m > 0) 
  (hyp_eq : ∀ x y : ℝ, y^2 - (x^2 / m^2) = 1) 
  (circ_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * y + 3 = 0) : 
  m = (Real.sqrt 3) / 3 :=
sorry

end asymptotes_tangent_to_circle_l790_79073


namespace min_m_plus_n_l790_79044

theorem min_m_plus_n (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) (h3 : 32 * m = n^5) : m + n = 3 :=
  sorry

end min_m_plus_n_l790_79044


namespace pyramid_base_edge_length_l790_79087

theorem pyramid_base_edge_length (height : ℝ) (radius : ℝ) (side_len : ℝ) :
  height = 4 ∧ radius = 3 →
  side_len = (12 * Real.sqrt 14) / 7 :=
by
  intros h
  rcases h with ⟨h1, h2⟩
  sorry

end pyramid_base_edge_length_l790_79087


namespace car_speed_reduction_and_increase_l790_79032

theorem car_speed_reduction_and_increase (V x : ℝ)
  (h1 : V > 0) -- V is positive
  (h2 : V * (1 - x / 100) * (1 + 0.5 * x / 100) = V * (1 - 0.6 * x / 100)) :
  x = 20 :=
sorry

end car_speed_reduction_and_increase_l790_79032


namespace rectangle_width_l790_79024

theorem rectangle_width (length_rect : ℝ) (width_rect : ℝ) (side_square : ℝ)
  (h1 : side_square * side_square = 5 * (length_rect * width_rect))
  (h2 : length_rect = 125)
  (h3 : 4 * side_square = 800) : width_rect = 64 :=
by 
  sorry

end rectangle_width_l790_79024


namespace pies_per_day_l790_79031

theorem pies_per_day (daily_pies total_pies : ℕ) (h1 : daily_pies = 8) (h2 : total_pies = 56) :
  total_pies / daily_pies = 7 :=
by sorry

end pies_per_day_l790_79031


namespace geometric_sequence_product_l790_79047

theorem geometric_sequence_product (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = r * a n) (h_cond : a 7 * a 12 = 5) :
  a 8 * a 9 * a 10 * a 11 = 25 :=
by 
  sorry

end geometric_sequence_product_l790_79047


namespace minimum_value_of_quadratic_function_l790_79081

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 8 * x + 15

theorem minimum_value_of_quadratic_function :
  ∃ x : ℝ, quadratic_function x = -1 ∧ ∀ y : ℝ, quadratic_function y ≥ -1 :=
by
  sorry

end minimum_value_of_quadratic_function_l790_79081


namespace ratio_doubled_to_original_l790_79009

theorem ratio_doubled_to_original (x : ℝ) (h : 3 * (2 * x + 9) = 69) : (2 * x) / x = 2 :=
by
  -- We skip the proof here.
  sorry

end ratio_doubled_to_original_l790_79009


namespace number_of_male_students_l790_79052

variables (total_students sample_size female_sampled female_students male_students : ℕ)
variables (h_total : total_students = 1600)
variables (h_sample : sample_size = 200)
variables (h_female_sampled : female_sampled = 95)
variables (h_prob : (sample_size : ℚ) / total_students = (female_sampled : ℚ) / female_students)
variables (h_female_students : female_students = 760)

theorem number_of_male_students : male_students = total_students - female_students := by
  sorry

end number_of_male_students_l790_79052


namespace bears_on_each_shelf_l790_79082

theorem bears_on_each_shelf 
    (initial_bears : ℕ) (shipment_bears : ℕ) (shelves : ℕ)
    (h1 : initial_bears = 4) (h2 : shipment_bears = 10) (h3 : shelves = 2) :
    (initial_bears + shipment_bears) / shelves = 7 := by
  sorry

end bears_on_each_shelf_l790_79082


namespace Parabola_vertex_form_l790_79022

theorem Parabola_vertex_form (x : ℝ) (y : ℝ) : 
  (∃ h k : ℝ, (h = -2) ∧ (k = 1) ∧ (y = (x + h)^2 + k) ) ↔ (y = (x + 2)^2 + 1) :=
by
  sorry

end Parabola_vertex_form_l790_79022


namespace find_x_l790_79025

theorem find_x (x y : ℝ)
  (h1 : 2 * x + (x - 30) = 360)
  (h2 : y = x - 30)
  (h3 : 2 * x = 4 * y) :
  x = 130 := 
sorry

end find_x_l790_79025


namespace min_value_of_2gx_sq_minus_fx_l790_79061

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
noncomputable def g (a c : ℝ) (x : ℝ) : ℝ := a * x + c

theorem min_value_of_2gx_sq_minus_fx (a b c : ℝ) (h_a_nonzero : a ≠ 0)
  (h_min_fx : ∃ x : ℝ, 2 * (f a b x)^2 - g a c x = 7 / 2) :
  ∃ x : ℝ, 2 * (g a c x)^2 - f a b x = -15 / 4 :=
sorry

end min_value_of_2gx_sq_minus_fx_l790_79061


namespace triangle_side_inequality_l790_79016

theorem triangle_side_inequality (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : 1 = 1 / 2 * b * c) : b ≥ Real.sqrt 2 :=
sorry

end triangle_side_inequality_l790_79016


namespace monotonically_decreasing_range_l790_79090

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 - x + 1
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x - 1

theorem monotonically_decreasing_range (a : ℝ) :
  (∀ x : ℝ, f' a x ≤ 0) → a ≤ -3 := by
  sorry

end monotonically_decreasing_range_l790_79090


namespace part1_part2_l790_79055

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x / x) - Real.log x + x - a

theorem part1 (a : ℝ) : (∀ x > 0, f x a ≥ 0) ↔ a ≤ Real.exp 1 + 1 :=
  sorry

theorem part2 (a : ℝ) (x1 x2 : ℝ) (h1 : f x1 a = 0) (h2 : f x2 a = 0) : x1 * x2 < 1 :=
  sorry

end part1_part2_l790_79055


namespace carol_rectangle_width_l790_79030

theorem carol_rectangle_width 
  (area_jordan : ℕ) (length_jordan width_jordan : ℕ) (width_carol length_carol : ℕ)
  (h1 : length_jordan = 12)
  (h2 : width_jordan = 10)
  (h3 : width_carol = 24)
  (h4 : area_jordan = length_jordan * width_jordan)
  (h5 : area_jordan = length_carol * width_carol) :
  length_carol = 5 :=
by
  sorry

end carol_rectangle_width_l790_79030


namespace correct_option_b_l790_79017

theorem correct_option_b (a : ℝ) : (-2 * a ^ 4) ^ 3 = -8 * a ^ 12 :=
sorry

end correct_option_b_l790_79017


namespace nickels_left_l790_79080

theorem nickels_left (n b : ℕ) (h₁ : n = 31) (h₂ : b = 20) : n - b = 11 :=
by
  sorry

end nickels_left_l790_79080


namespace range_of_a_for_maximum_l790_79046

variable {f : ℝ → ℝ}
variable {a : ℝ}

theorem range_of_a_for_maximum (h : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_max : ∀ x, f x ≤ f a → x = a) : -1 < a ∧ a < 0 :=
sorry

end range_of_a_for_maximum_l790_79046


namespace minimum_routes_l790_79091

theorem minimum_routes (a b c : ℕ) (h1 : a + b ≥ 14) (h2 : b + c ≥ 14) (h3 : c + a ≥ 14) :
  a + b + c ≥ 21 :=
by sorry

end minimum_routes_l790_79091
