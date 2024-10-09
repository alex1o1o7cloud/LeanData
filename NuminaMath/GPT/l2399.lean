import Mathlib

namespace book_price_l2399_239962

theorem book_price (x : ℕ) : 
  9 * x ≤ 1100 ∧ 13 * x ≤ 1500 → x = 123 :=
sorry

end book_price_l2399_239962


namespace side_length_of_S2_l2399_239909

-- Define our context and the statements we need to work with
theorem side_length_of_S2
  (r s : ℕ)
  (h1 : 2 * r + s = 2450)
  (h2 : 2 * r + 3 * s = 4000) : 
  s = 775 :=
sorry

end side_length_of_S2_l2399_239909


namespace otimes_evaluation_l2399_239939

def otimes (a b : ℝ) : ℝ := a * b + a - b

theorem otimes_evaluation (a b : ℝ) : 
  otimes a b + otimes (b - a) b = b^2 - b := 
  by
  sorry

end otimes_evaluation_l2399_239939


namespace farm_area_l2399_239919

theorem farm_area
  (b : ℕ) (l : ℕ) (d : ℕ)
  (h_b : b = 30)
  (h_cost : 15 * (l + b + d) = 1800)
  (h_pythagorean : d^2 = l^2 + b^2) :
  l * b = 1200 :=
by
  sorry

end farm_area_l2399_239919


namespace temperature_difference_l2399_239979

variable (highest_temp : ℤ)
variable (lowest_temp : ℤ)

theorem temperature_difference : 
  highest_temp = 2 ∧ lowest_temp = -8 → (highest_temp - lowest_temp = 10) := by
  sorry

end temperature_difference_l2399_239979


namespace maximum_value_2a_plus_b_l2399_239922

variable (a b : ℝ)

theorem maximum_value_2a_plus_b (h : 4 * a^2 + b^2 + a * b = 1) : 2 * a + b ≤ 2 * Real.sqrt (10) / 5 :=
by sorry

end maximum_value_2a_plus_b_l2399_239922


namespace arithmetic_sequence_a4_eight_l2399_239936

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m : ℕ, a (n + m) = a n + m * (a 2 - a 1)

variable {a : ℕ → ℤ}

theorem arithmetic_sequence_a4_eight (h_arith_sequence : arithmetic_sequence a)
    (h_cond : a 3 + a 5 = 16) : a 4 = 8 :=
by
  sorry

end arithmetic_sequence_a4_eight_l2399_239936


namespace intersection_A_B_l2399_239903

def A (x : ℝ) : Prop := x^2 - 3 * x < 0
def B (x : ℝ) : Prop := x > 2

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l2399_239903


namespace f_f_n_plus_n_eq_n_plus_1_l2399_239963

-- Define the function f : ℕ+ → ℕ+ satisfying the given condition
axiom f : ℕ+ → ℕ+

-- Define that for all positive integers n, f satisfies the condition f(f(n)) + f(n+1) = n + 2
axiom f_condition : ∀ n : ℕ+, f (f n) + f (n + 1) = n + 2

-- State that we want to prove that f(f(n) + n) = n + 1 for all positive integers n
theorem f_f_n_plus_n_eq_n_plus_1 : ∀ n : ℕ+, f (f n + n) = n + 1 := 
by sorry

end f_f_n_plus_n_eq_n_plus_1_l2399_239963


namespace cassie_nails_l2399_239934

def num_dogs : ℕ := 4
def nails_per_dog_leg : ℕ := 4
def legs_per_dog : ℕ := 4
def num_parrots : ℕ := 8
def claws_per_parrot_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def extra_claws : ℕ := 1

def total_nails_to_cut : ℕ :=
  num_dogs * nails_per_dog_leg * legs_per_dog +
  num_parrots * claws_per_parrot_leg * legs_per_parrot + extra_claws

theorem cassie_nails : total_nails_to_cut = 113 :=
  by sorry

end cassie_nails_l2399_239934


namespace find_f_2_l2399_239964

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2 (h1 : ∀ x1 x2 : ℝ, f (x1 * x2) = f x1 + f x2) (h2 : f 8 = 3) : f 2 = 1 :=
by
  sorry

end find_f_2_l2399_239964


namespace hyperbola_center_l2399_239985

theorem hyperbola_center (x y : ℝ) :
  ( ∃ (h k : ℝ), ∀ (x y : ℝ), (4 * x - 8)^2 / 9^2 - (5 * y - 15)^2 / 7^2 = 1 → (h, k) = (2, 3) ) :=
by
  existsi 2
  existsi 3
  intros x y h
  sorry

end hyperbola_center_l2399_239985


namespace monotone_f_range_l2399_239921

noncomputable def f (a x : ℝ) : ℝ :=
  x - (1 / 3) * Real.sin (2 * x) + a * Real.sin x

theorem monotone_f_range (a : ℝ) :
  (∀ x : ℝ, (1 - (2 / 3) * Real.cos (2 * x) + a * Real.cos x) ≥ 0) ↔ (-1 / 3 ≤ a ∧ a ≤ 1 / 3) := 
sorry

end monotone_f_range_l2399_239921


namespace value_b_minus_a_l2399_239967

theorem value_b_minus_a (a b : ℝ) (h₁ : a + b = 507) (h₂ : (a - b) / b = 1 / 7) : b - a = -34.428571 :=
by
  sorry

end value_b_minus_a_l2399_239967


namespace calculate_value_l2399_239927

theorem calculate_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x = 1 / y) (hzy : z = 1 / y) : 
  (x + 1 / x) * (z - 1 / z) = 4 := 
by 
  -- Proof omitted, this is just the statement
  sorry

end calculate_value_l2399_239927


namespace tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l2399_239906

theorem tan_beta_of_tan_alpha_and_tan_alpha_plus_beta (α β : ℝ)
  (h1 : Real.tan α = 2)
  (h2 : Real.tan (α + β) = 1 / 5) :
  Real.tan β = -9 / 7 :=
sorry

end tan_beta_of_tan_alpha_and_tan_alpha_plus_beta_l2399_239906


namespace emily_small_gardens_l2399_239954

theorem emily_small_gardens (total_seeds planted_big_garden seeds_per_small_garden : ℕ) 
  (h1 : total_seeds = 41) 
  (h2 : planted_big_garden = 29) 
  (h3 : seeds_per_small_garden = 4) : 
  (total_seeds - planted_big_garden) / seeds_per_small_garden = 3 := 
by
  sorry

end emily_small_gardens_l2399_239954


namespace smallest_x_l2399_239960

theorem smallest_x (x : ℕ) :
  (x % 6 = 5) ∧ (x % 7 = 6) ∧ (x % 8 = 7) → x = 167 :=
by
  sorry

end smallest_x_l2399_239960


namespace will_3_point_shots_l2399_239902

theorem will_3_point_shots :
  ∃ x y : ℕ, 3 * x + 2 * y = 26 ∧ x + y = 11 ∧ x = 4 :=
by
  sorry

end will_3_point_shots_l2399_239902


namespace triangle_side_length_l2399_239920

theorem triangle_side_length 
  (a b c : ℝ) 
  (cosA : ℝ) 
  (h1: a = Real.sqrt 5) 
  (h2: c = 2) 
  (h3: cosA = 2 / 3) 
  (h4: a^2 = b^2 + c^2 - 2 * b * c * cosA) : 
  b = 3 := 
by 
  sorry

end triangle_side_length_l2399_239920


namespace center_cell_value_l2399_239990

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l2399_239990


namespace square_ratio_l2399_239961

theorem square_ratio (x y : ℝ) (hx : x = 60 / 17) (hy : y = 780 / 169) : 
  x / y = 169 / 220 :=
by
  sorry

end square_ratio_l2399_239961


namespace total_kids_in_lawrence_county_l2399_239993

theorem total_kids_in_lawrence_county :
  ∀ (h c T : ℕ), h = 274865 → c = 38608 → T = h + c → T = 313473 :=
by
  intros h c T h_eq c_eq T_eq
  rw [h_eq, c_eq] at T_eq
  exact T_eq

end total_kids_in_lawrence_county_l2399_239993


namespace thirty_five_power_identity_l2399_239912

theorem thirty_five_power_identity (m n : ℕ) : 
  let P := 5^m 
  let Q := 7^n 
  35^(m*n) = P^n * Q^m :=
by 
  sorry

end thirty_five_power_identity_l2399_239912


namespace ryan_distance_correct_l2399_239965

-- Definitions of the conditions
def billy_distance : ℝ := 30
def madison_distance : ℝ := billy_distance * 1.2
def ryan_distance : ℝ := madison_distance * 0.5

-- Statement to prove
theorem ryan_distance_correct : ryan_distance = 18 := by
  sorry

end ryan_distance_correct_l2399_239965


namespace intersection_of_complements_l2399_239978

theorem intersection_of_complements {U S T : Set ℕ}
  (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
  (hS : S = {1, 3, 5})
  (hT : T = {3, 6}) :
  (U \ S) ∩ (U \ T) = {2, 4, 7, 8} :=
by
  sorry

end intersection_of_complements_l2399_239978


namespace minimum_value_of_f_l2399_239930

noncomputable def f (x : ℝ) : ℝ := 2 * x + (3 * x) / (x^2 + 3) + (2 * x * (x + 5)) / (x^2 + 5) + (3 * (x + 3)) / (x * (x^2 + 5))

theorem minimum_value_of_f : ∃ a : ℝ, a > 0 ∧ (∀ x > 0, f x ≥ 7) ∧ (f a = 7) :=
by
  sorry

end minimum_value_of_f_l2399_239930


namespace snail_returns_l2399_239931

noncomputable def snail_path : Type := ℕ → ℝ × ℝ

def snail_condition (snail : snail_path) (speed : ℝ) : Prop :=
  ∀ n : ℕ, n % 4 = 0 → snail (n + 4) = snail n

theorem snail_returns (snail : snail_path) (speed : ℝ) (h1 : ∀ n m : ℕ, n ≠ m → snail n ≠ snail m)
    (h2 : snail_condition snail speed) :
  ∃ t : ℕ, t > 0 ∧ t % 4 = 0 ∧ snail t = snail 0 := 
sorry

end snail_returns_l2399_239931


namespace rowing_time_to_place_and_back_l2399_239995

def speed_man_still_water : ℝ := 8 -- km/h
def speed_river : ℝ := 2 -- km/h
def total_distance : ℝ := 7.5 -- km

theorem rowing_time_to_place_and_back :
  let V_m := speed_man_still_water
  let V_r := speed_river
  let D := total_distance / 2
  let V_up := V_m - V_r
  let V_down := V_m + V_r
  let T_up := D / V_up
  let T_down := D / V_down
  T_up + T_down = 1 :=
by
  sorry

end rowing_time_to_place_and_back_l2399_239995


namespace find_product_of_roots_l2399_239907

namespace ProductRoots

variables {k m : ℝ} {x1 x2 : ℝ}

theorem find_product_of_roots (h1 : x1 ≠ x2) 
    (hx1 : 5 * x1 ^ 2 - k * x1 = m) 
    (hx2 : 5 * x2 ^ 2 - k * x2 = m) : x1 * x2 = -m / 5 :=
sorry

end ProductRoots

end find_product_of_roots_l2399_239907


namespace factorization_correct_l2399_239900

-- Define the initial expression
def initial_expr (a x : ℝ) : ℝ := a * x^2 - 2 * a * x + a

-- Define the factorized form
def factorized_expr (a x : ℝ) : ℝ := a * (x - 1)^2

-- Create the theorem statement
theorem factorization_correct (a x : ℝ) : initial_expr a x = factorized_expr a x :=
by simp [initial_expr, factorized_expr, pow_two, mul_add, add_mul, add_assoc]; sorry

end factorization_correct_l2399_239900


namespace andrew_daily_work_hours_l2399_239970

theorem andrew_daily_work_hours (total_hours : ℝ) (days : ℝ) (h1 : total_hours = 7.5) (h2 : days = 3) : total_hours / days = 2.5 :=
by
  rw [h1, h2]
  norm_num

end andrew_daily_work_hours_l2399_239970


namespace can_capacity_l2399_239904

theorem can_capacity (x : ℝ) (milk water : ℝ) (full_capacity : ℝ) : 
  5 * x = milk ∧ 
  3 * x = water ∧ 
  full_capacity = milk + water + 8 ∧ 
  (milk + 8) / water = 2 → 
  full_capacity = 72 := 
sorry

end can_capacity_l2399_239904


namespace pasha_wins_9_games_l2399_239998

theorem pasha_wins_9_games :
  ∃ w l : ℕ, (w + l = 12) ∧ (2^w * (2^l - 1) - (2^l - 1) * 2^(w - 1) = 2023) ∧ (w = 9) :=
by
  sorry

end pasha_wins_9_games_l2399_239998


namespace basic_cable_cost_l2399_239971

variable (B M S : ℝ)

def CostOfMovieChannels (B : ℝ) : ℝ := B + 12
def CostOfSportsChannels (M : ℝ) : ℝ := M - 3

theorem basic_cable_cost :
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  B + M + S = 36 → B = 5 :=
by
  intro h
  let M := CostOfMovieChannels B
  let S := CostOfSportsChannels M
  sorry

end basic_cable_cost_l2399_239971


namespace recurrent_sequence_solution_l2399_239929

theorem recurrent_sequence_solution (a : ℕ → ℕ) : 
  (a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + 2^n) →
  (∀ n, n ≥ 1 → a n = (2 * n - 1) * 2^(n - 1)) :=
by
  sorry

end recurrent_sequence_solution_l2399_239929


namespace erica_pie_fraction_as_percentage_l2399_239972

theorem erica_pie_fraction_as_percentage (apple_pie_fraction : ℚ) (cherry_pie_fraction : ℚ) 
  (h1 : apple_pie_fraction = 1 / 5) 
  (h2 : cherry_pie_fraction = 3 / 4) 
  (common_denominator : ℚ := 20) : 
  (apple_pie_fraction + cherry_pie_fraction) * 100 = 95 :=
by
  sorry

end erica_pie_fraction_as_percentage_l2399_239972


namespace solution_set_I_range_of_m_l2399_239918

noncomputable def f (x : ℝ) : ℝ := |2 * x + 3| + |2 * x - 1|

theorem solution_set_I (x : ℝ) : f x < 8 ↔ -5 / 2 < x ∧ x < 3 / 2 :=
sorry

theorem range_of_m (m : ℝ) (h : ∃ x, f x ≤ |3 * m + 1|) : m ≤ -5 / 3 ∨ m ≥ 1 :=
sorry

end solution_set_I_range_of_m_l2399_239918


namespace min_PM_PN_l2399_239937

noncomputable def C1 (x y : ℝ) : Prop := (x + 6)^2 + (y - 5)^2 = 4
noncomputable def C2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

theorem min_PM_PN : ∀ (P M N : ℝ × ℝ),
  P.2 = 0 ∧ C1 M.1 M.2 ∧ C2 N.1 N.2 → (|P.1 - M.1| + (P.1 - N.1)^2 + (P.2 - N.2)^2).sqrt = 7 := by
  sorry

end min_PM_PN_l2399_239937


namespace solve_equation_l2399_239973

theorem solve_equation : ∀ (x : ℝ), x ≠ -3 → x ≠ 3 → 
  (x / (x + 3) + 6 / (x^2 - 9) = 1 / (x - 3)) → x = 1 :=
by
  intros x hx1 hx2 h
  sorry

end solve_equation_l2399_239973


namespace digit_B_for_divisibility_by_9_l2399_239924

theorem digit_B_for_divisibility_by_9 :
  ∃! (B : ℕ), B < 10 ∧ (5 + B + B + 3) % 9 = 0 :=
by
  sorry

end digit_B_for_divisibility_by_9_l2399_239924


namespace range_of_m_l2399_239974

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, mx^2 - 4*x + 1 = 0 ∧ ∀ y : ℝ, mx^2 - 4*x + 1 = 0 → y = x) → m ≤ 4 :=
sorry

end range_of_m_l2399_239974


namespace negation_of_proposition_l2399_239935

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x ≤ 0 ∧ x^2 ≥ 0) ↔ ∀ x : ℝ, x ≤ 0 → x^2 < 0 :=
by
  sorry

end negation_of_proposition_l2399_239935


namespace percentage_increase_visitors_l2399_239953

theorem percentage_increase_visitors
  (original_visitors : ℕ)
  (original_fee : ℝ := 1)
  (fee_reduction : ℝ := 0.25)
  (visitors_increase : ℝ := 0.20) :
  ((original_visitors + (visitors_increase * original_visitors)) / original_visitors - 1) * 100 = 20 := by
  sorry

end percentage_increase_visitors_l2399_239953


namespace jerry_average_increase_l2399_239959

-- Definitions of conditions
def first_three_tests_average (avg : ℕ) : Prop := avg = 85
def fourth_test_score (score : ℕ) : Prop := score = 97
def desired_average_increase (increase : ℕ) : Prop := increase = 3

-- The theorem to prove
theorem jerry_average_increase
  (first_avg first_avg_value : ℕ)
  (fourth_score fourth_score_value : ℕ)
  (increase_points : ℕ)
  (h1 : first_three_tests_average first_avg)
  (h2 : fourth_test_score fourth_score)
  (h3 : desired_average_increase increase_points) :
  fourth_score = 97 → (first_avg + fourth_score) / 4 = 88 → increase_points = 3 :=
by
  intros _ _
  sorry

end jerry_average_increase_l2399_239959


namespace total_spokes_in_garage_l2399_239957

theorem total_spokes_in_garage :
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114 :=
by
  let bicycle1_spokes := 12 + 10
  let bicycle2_spokes := 14 + 12
  let bicycle3_spokes := 10 + 14
  let tricycle_spokes := 14 + 12 + 16
  show bicycle1_spokes + bicycle2_spokes + bicycle3_spokes + tricycle_spokes = 114
  sorry

end total_spokes_in_garage_l2399_239957


namespace div_eq_of_scaled_div_eq_l2399_239925

theorem div_eq_of_scaled_div_eq (h : 29.94 / 1.45 = 17.7) : 2994 / 14.5 = 17.7 := 
by
  sorry

end div_eq_of_scaled_div_eq_l2399_239925


namespace number_of_red_balls_l2399_239911

theorem number_of_red_balls (x : ℕ) (h₀ : 4 > 0) (h₁ : (x : ℝ) / (x + 4) = 0.6) : x = 6 :=
sorry

end number_of_red_balls_l2399_239911


namespace james_proof_l2399_239949

def james_pages_per_hour 
  (writes_some_pages_an_hour : ℕ)
  (writes_5_pages_to_2_people_each_day : ℕ)
  (hours_spent_writing_per_week : ℕ) 
  (writes_total_pages_per_day : ℕ)
  (writes_total_pages_per_week : ℕ) 
  (pages_per_hour : ℕ) 
: Prop :=
  writes_some_pages_an_hour = writes_5_pages_to_2_people_each_day / hours_spent_writing_per_week

theorem james_proof
  (writes_some_pages_an_hour : ℕ := 10)
  (writes_5_pages_to_2_people_each_day : ℕ := 5 * 2)
  (hours_spent_writing_per_week : ℕ := 7)
  (writes_total_pages_per_day : ℕ := writes_5_pages_to_2_people_each_day)
  (writes_total_pages_per_week : ℕ := writes_total_pages_per_day * 7)
  (pages_per_hour : ℕ := writes_total_pages_per_week / hours_spent_writing_per_week)
: writes_some_pages_an_hour = pages_per_hour :=
by {
  sorry 
}

end james_proof_l2399_239949


namespace additional_savings_l2399_239983

def initial_price : Float := 30
def discount1 : Float := 5
def discount2_percent : Float := 0.25

def price_after_discount1_then_discount2 : Float := 
  (initial_price - discount1) * (1 - discount2_percent)

def price_after_discount2_then_discount1 : Float := 
  initial_price * (1 - discount2_percent) - discount1

theorem additional_savings :
  price_after_discount1_then_discount2 - price_after_discount2_then_discount1 = 1.25 := by
  sorry

end additional_savings_l2399_239983


namespace find_k_l2399_239947

-- Define the sum of even integers from 2 to 2k
def sum_even_integers (k : ℕ) : ℕ :=
  2 * (k * (k + 1)) / 2

-- Define the condition that this sum equals 132
def sum_condition (t : ℕ) (k : ℕ) : Prop :=
  sum_even_integers k = t

theorem find_k (k : ℕ) (t : ℕ) (h₁ : t = 132) (h₂ : sum_condition t k) : k = 11 := by
  sorry

end find_k_l2399_239947


namespace whisky_replacement_l2399_239994

variable (V : ℝ) (x : ℝ)

theorem whisky_replacement (h_condition : 0.40 * V - 0.40 * x + 0.19 * x = 0.26 * V) : 
  x = (2 / 3) * V := 
sorry

end whisky_replacement_l2399_239994


namespace common_chord_of_circles_is_x_eq_y_l2399_239999

theorem common_chord_of_circles_is_x_eq_y :
  ∀ x y : ℝ, (x^2 + y^2 - 4 * x - 3 = 0) ∧ (x^2 + y^2 - 4 * y - 3 = 0) → (x = y) :=
by
  sorry

end common_chord_of_circles_is_x_eq_y_l2399_239999


namespace find_abc_l2399_239989

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom condition1 : a * b = 45 * (3 : ℝ)^(1/3)
axiom condition2 : a * c = 75 * (3 : ℝ)^(1/3)
axiom condition3 : b * c = 30 * (3 : ℝ)^(1/3)

theorem find_abc : a * b * c = 75 * (2 : ℝ)^(1/2) := sorry

end find_abc_l2399_239989


namespace derivative_at_1_l2399_239996

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem derivative_at_1 : deriv f 1 = 0 :=
by
  -- Proof to be provided
  sorry

end derivative_at_1_l2399_239996


namespace LukaLemonadeSolution_l2399_239951

def LukaLemonadeProblem : Prop :=
  ∃ (L S W : ℕ), 
    (S = 3 * L) ∧
    (W = 3 * S) ∧
    (L = 4) ∧
    (W = 36)

theorem LukaLemonadeSolution : LukaLemonadeProblem :=
  by sorry

end LukaLemonadeSolution_l2399_239951


namespace find_number_l2399_239980

theorem find_number (x : ℤ) (h : x + 2 - 3 = 7) : x = 8 :=
sorry

end find_number_l2399_239980


namespace solve_system_equations_l2399_239941

-- Define the hypotheses of the problem
variables {a x y : ℝ}
variables (h1 : (0 < a) ∧ (a ≠ 1))
variables (h2 : (0 < x))
variables (h3 : (0 < y))
variables (eq1 : (log a x + log a y - 2) * log 18 a = 1)
variables (eq2 : 2 * x + y - 20 * a = 0)

-- State the theorem to be proved
theorem solve_system_equations :
  (x = a ∧ y = 18 * a) ∨ (x = 9 * a ∧ y = 2 * a) := by
  sorry

end solve_system_equations_l2399_239941


namespace inequality_proof_l2399_239950

-- Let x and y be real numbers such that x > y
variables {x y : ℝ} (hx : x > y)

-- We need to prove -2x < -2y
theorem inequality_proof (hx : x > y) : -2 * x < -2 * y :=
sorry

end inequality_proof_l2399_239950


namespace minimum_value_expression_l2399_239917

theorem minimum_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ( (3*a*b - 6*b + a*(1-a))^2 + (9*b^2 + 2*a + 3*b*(1-a))^2 ) / (a^2 + 9*b^2) ≥ 4 :=
sorry

end minimum_value_expression_l2399_239917


namespace tshirts_per_package_l2399_239955

-- Definitions based on the conditions
def total_tshirts : ℕ := 70
def num_packages : ℕ := 14

-- Theorem to prove the number of t-shirts per package
theorem tshirts_per_package : total_tshirts / num_packages = 5 := by
  -- The proof is omitted, only the statement is provided as required.
  sorry

end tshirts_per_package_l2399_239955


namespace fraction_product_is_one_l2399_239946

theorem fraction_product_is_one : 
  (1 / 4) * (1 / 5) * (1 / 6) * 120 = 1 :=
by 
  sorry

end fraction_product_is_one_l2399_239946


namespace probability_A_mc_and_B_tf_probability_at_least_one_mc_l2399_239977

-- Define the total number of questions
def total_questions : ℕ := 5

-- Define the number of multiple choice questions and true or false questions
def multiple_choice_questions : ℕ := 3
def true_false_questions : ℕ := 2

-- First proof problem: Probability that A draws a multiple-choice question and B draws a true or false question
theorem probability_A_mc_and_B_tf :
  (multiple_choice_questions * true_false_questions : ℚ) / (total_questions * (total_questions - 1)) = 3 / 10 :=
by
  sorry

-- Second proof problem: Probability that at least one of A and B draws a multiple-choice question
theorem probability_at_least_one_mc :
  1 - (true_false_questions * (true_false_questions - 1) : ℚ) / (total_questions * (total_questions - 1)) = 9 / 10 :=
by
  sorry

end probability_A_mc_and_B_tf_probability_at_least_one_mc_l2399_239977


namespace fraction_evaluation_l2399_239986

theorem fraction_evaluation :
  (2 + 3 * 6) / (23 + 6) = 20 / 29 := by
  -- Proof can be filled in here
  sorry

end fraction_evaluation_l2399_239986


namespace industrial_park_investment_l2399_239943

noncomputable def investment_in_projects : Prop :=
  ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500

theorem industrial_park_investment :
  investment_in_projects :=
by
  have h : ∃ (x : ℝ), 0.054 * x + 0.0828 * (2000 - x) = 122.4 ∧ x = 1500 ∧ (2000 - x) = 500 := 
    sorry
  exact h

end industrial_park_investment_l2399_239943


namespace find_number_with_10_questions_l2399_239952

theorem find_number_with_10_questions (n : ℕ) (h : n ≤ 1000) : n = 300 :=
by
  sorry

end find_number_with_10_questions_l2399_239952


namespace megan_folders_count_l2399_239958

theorem megan_folders_count (init_files deleted_files files_per_folder : ℕ) (h₁ : init_files = 93) (h₂ : deleted_files = 21) (h₃ : files_per_folder = 8) :
  (init_files - deleted_files) / files_per_folder = 9 :=
by
  sorry

end megan_folders_count_l2399_239958


namespace markup_is_correct_l2399_239933

def purchase_price : ℝ := 48
def overhead_percent : ℝ := 0.25
def net_profit : ℝ := 12

def overhead_cost := overhead_percent * purchase_price
def total_cost := purchase_price + overhead_cost
def selling_price := total_cost + net_profit
def markup := selling_price - purchase_price

theorem markup_is_correct : markup = 24 := by sorry

end markup_is_correct_l2399_239933


namespace max_area_of_backyard_l2399_239905

theorem max_area_of_backyard (fence_length : ℕ) (h1 : fence_length = 500) 
  (l w : ℕ) (h2 : l = 2 * w) (h3 : l + 2 * w = fence_length) : 
  l * w = 31250 := 
by
  sorry

end max_area_of_backyard_l2399_239905


namespace range_of_a_l2399_239984

def my_Op (a b : ℝ) : ℝ := a - 2 * b

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, my_Op x 3 > 0 → my_Op x a > a) ↔ (∀ x : ℝ, x > 6 → x > 3 * a) → a ≤ 2 :=
by sorry

end range_of_a_l2399_239984


namespace division_identity_l2399_239913

theorem division_identity
  (x y : ℕ)
  (h1 : x = 7)
  (h2 : y = 2)
  : (x^3 + y^3) / (x^2 - x * y + y^2) = 9 :=
by
  sorry

end division_identity_l2399_239913


namespace find_f_on_interval_l2399_239945

/-- Representation of periodic and even functions along with specific interval definition -/
noncomputable def f (x : ℝ) : ℝ := 
if 2 ≤ x ∧ x ≤ 3 then -2*(x-3)^2 + 4 else 0 -- Define f(x) on [2,3], otherwise undefined

/-- Main proof statement -/
theorem find_f_on_interval :
  (∀ x, f x = f (x + 2)) ∧  -- f(x) is periodic with period 2
  (∀ x, f x = f (-x)) ∧   -- f(x) is even
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x = -2*(x-3)^2 + 4) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = -2*(x-1)^2 + 4) :=
sorry

end find_f_on_interval_l2399_239945


namespace cone_height_l2399_239968

theorem cone_height 
  (sector_radius : ℝ) 
  (central_angle : ℝ) 
  (sector_radius_eq : sector_radius = 3) 
  (central_angle_eq : central_angle = 2 * π / 3) : 
  ∃ h : ℝ, h = 2 * Real.sqrt 2 :=
by
  -- Formalize conditions
  let r := 1
  let l := sector_radius
  let θ := central_angle

  -- Combine conditions
  have r_eq : r = 1 := by sorry

  -- Calculate height using Pythagorean theorem
  let h := (l^2 - r^2).sqrt

  use h
  have h_eq : h = 2 * Real.sqrt 2 := by sorry
  exact h_eq

end cone_height_l2399_239968


namespace monomial_sum_mn_l2399_239944

-- Define the conditions as Lean definitions
def is_monomial_sum (x y : ℕ) (m n : ℕ) : Prop :=
  ∃ k : ℕ, (x ^ 2) * (y ^ m) + (x ^ n) * (y ^ 3) = x ^ k

-- State our main theorem
theorem monomial_sum_mn (x y : ℕ) (m n : ℕ) (h : is_monomial_sum x y m n) : m + n = 5 :=
sorry  -- Completion of the proof is not required

end monomial_sum_mn_l2399_239944


namespace condition_sufficient_not_necessary_monotonicity_l2399_239916

theorem condition_sufficient_not_necessary_monotonicity
  (f : ℝ → ℝ) (a : ℝ) (h_def : ∀ x, f x = 2^(abs (x - a))) :
  (∀ x > 1, x - a ≥ 0) → (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y) ∧
  (∃ a, a ≤ 1 ∧ (∀ x > 1, x - a ≥ 0) ∧ (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y)) :=
by
  sorry

end condition_sufficient_not_necessary_monotonicity_l2399_239916


namespace largest_angle_is_75_l2399_239982

-- Let the measures of the angles be represented as 3x, 4x, and 5x for some value x
variable (x : ℝ)

-- Define the angles based on the given ratio
def angle1 := 3 * x
def angle2 := 4 * x
def angle3 := 5 * x

-- The sum of the angles in a triangle is 180 degrees
axiom angle_sum : angle1 + angle2 + angle3 = 180

-- Prove that the largest angle is 75 degrees
theorem largest_angle_is_75 : 5 * (180 / 12) = 75 :=
by
  -- Proof is not required as per the instructions
  sorry

end largest_angle_is_75_l2399_239982


namespace calculate_g3_l2399_239942

def g (x : ℚ) : ℚ := (2 * x - 3) / (5 * x + 2)

theorem calculate_g3 : g 3 = 3 / 17 :=
by {
    -- Here we add the proof steps if necessary, but for now we use sorry
    sorry
}

end calculate_g3_l2399_239942


namespace max_regions_divided_by_lines_l2399_239948

theorem max_regions_divided_by_lines (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) :
  ∃ r : ℕ, r = m * n + 2 * m + 2 * n - 1 :=
by
  sorry

end max_regions_divided_by_lines_l2399_239948


namespace repeating_decimal_to_fraction_l2399_239992

theorem repeating_decimal_to_fraction : (0.7 + 23 / 99 / 10) = (62519 / 66000) := by
  sorry

end repeating_decimal_to_fraction_l2399_239992


namespace find_a_l2399_239988

theorem find_a (a x : ℝ) (h1 : 3 * x + 5 = 11) (h2 : 6 * x + 3 * a = 22) : a = 10 / 3 :=
by
  -- the proof will go here
  sorry

end find_a_l2399_239988


namespace range_of_m_l2399_239969

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ 4) → 
  (∀ x : ℝ, |x + 1| + |x - 3| ≥ |m - 1|) → 
  -3 ≤ m ∧ m ≤ 5 := 
by 
  sorry

end range_of_m_l2399_239969


namespace total_cupcakes_baked_l2399_239915

-- Conditions
def morning_cupcakes : ℕ := 20
def afternoon_cupcakes : ℕ := morning_cupcakes + 15

-- Goal
theorem total_cupcakes_baked :
  (morning_cupcakes + afternoon_cupcakes) = 55 :=
by
  sorry

end total_cupcakes_baked_l2399_239915


namespace exists_segment_satisfying_condition_l2399_239908

theorem exists_segment_satisfying_condition :
  ∃ (x₁ x₂ x₃ : ℚ) (f : ℚ → ℤ), x₃ = (x₁ + x₂) / 2 ∧ f x₁ + f x₂ ≤ 2 * f x₃ :=
sorry

end exists_segment_satisfying_condition_l2399_239908


namespace sequence_general_formula_l2399_239938

theorem sequence_general_formula {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n : ℕ, a (n + 1) = 2 * a n + 3 * 5 ^ n) 
  : ∀ n : ℕ, a n = 5 ^ n - 3 * 2 ^ (n - 1) :=
sorry

end sequence_general_formula_l2399_239938


namespace container_volume_ratio_l2399_239991

theorem container_volume_ratio
  (A B C : ℚ)  -- A is the volume of the first container, B is the volume of the second container, C is the volume of the third container
  (h1 : (8 / 9) * A = (7 / 9) * B)  -- Condition: First container was 8/9 full and second container gets filled to 7/9 after transfer.
  (h2 : (7 / 9) * B + (1 / 2) * C = C)  -- Condition: Mixing contents from second and third containers completely fill third container.
  : A / C = 63 / 112 := sorry  -- We need to prove this.

end container_volume_ratio_l2399_239991


namespace line_through_longest_chord_l2399_239997

-- Define the point M and the circle equation
def M : ℝ × ℝ := (3, -1)
def circle_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + y - 2 = 0

-- Define the standard form of the circle equation
def standard_circle_eqn (x y : ℝ) : Prop := (x - 2)^2 + (y + 1/2)^2 = 25/4

-- Define the line equation
def line_eqn (x y : ℝ) : Prop := x + 2 * y - 2 = 0

-- Theorem: Equation of the line containing the longest chord passing through M
theorem line_through_longest_chord : 
  (circle_eqn 3 (-1)) → 
  ∀ (x y : ℝ), standard_circle_eqn x y → ∃ (k b : ℝ), line_eqn x y :=
by
  -- Proof goes here
  intro h1 x y h2
  sorry

end line_through_longest_chord_l2399_239997


namespace handshake_problem_l2399_239981

theorem handshake_problem :
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  unique_handshakes = 250 :=
by 
  let companies := 5
  let representatives_per_company := 5
  let total_people := companies * representatives_per_company
  let possible_handshakes_per_person := total_people - representatives_per_company - 1
  let total_possible_handshakes := total_people * possible_handshakes_per_person
  let unique_handshakes := total_possible_handshakes / 2
  sorry

end handshake_problem_l2399_239981


namespace haley_marbles_l2399_239910

theorem haley_marbles (boys marbles_per_boy : ℕ) (h1: boys = 5) (h2: marbles_per_boy = 7) : boys * marbles_per_boy = 35 := 
by 
  sorry

end haley_marbles_l2399_239910


namespace ratio_of_juice_to_bread_l2399_239914

variable (total_money : ℕ) (money_left : ℕ) (cost_bread : ℕ) (cost_butter : ℕ) (cost_juice : ℕ)

def compute_ratio (total_money money_left cost_bread cost_butter cost_juice : ℕ) : ℕ :=
  cost_juice / cost_bread

theorem ratio_of_juice_to_bread :
  total_money = 15 →
  money_left = 6 →
  cost_bread = 2 →
  cost_butter = 3 →
  total_money - money_left - (cost_bread + cost_butter) = cost_juice →
  compute_ratio total_money money_left cost_bread cost_butter cost_juice = 2 :=
by
  intros
  sorry

end ratio_of_juice_to_bread_l2399_239914


namespace systematic_sampling_interval_l2399_239987

def population_size : ℕ := 2000
def sample_size : ℕ := 50
def interval (N n : ℕ) : ℕ := N / n

theorem systematic_sampling_interval :
  interval population_size sample_size = 40 := by
  sorry

end systematic_sampling_interval_l2399_239987


namespace pool_one_quarter_capacity_at_6_l2399_239956

-- Variables and parameters
variables (volume : ℕ → ℝ) (T : ℕ)

-- Conditions
def doubles_every_hour : Prop :=
  ∀ t, volume (t + 1) = 2 * volume t

def full_capacity_at_8 : Prop :=
  volume 8 = T

def one_quarter_capacity (t : ℕ) : Prop :=
  volume t = T / 4

-- Theorem to prove
theorem pool_one_quarter_capacity_at_6 (h1 : doubles_every_hour volume) (h2 : full_capacity_at_8 volume T) : one_quarter_capacity volume T 6 :=
sorry

end pool_one_quarter_capacity_at_6_l2399_239956


namespace prove_inequality_l2399_239923

theorem prove_inequality
  (a : ℕ → ℕ) -- Define a sequence of natural numbers
  (h_initial : a 1 > a 0) -- Initial condition
  (h_recurrence : ∀ n ≥ 2, a n = 3 * a (n - 1) - 2 * a (n - 2)) -- Recurrence relation
  : a 100 > 2^99 := by
  sorry -- Proof placeholder

end prove_inequality_l2399_239923


namespace number_of_littering_citations_l2399_239940

variable (L D P : ℕ)
variable (h1 : L = D)
variable (h2 : P = 2 * (L + D))
variable (h3 : L + D + P = 24)

theorem number_of_littering_citations : L = 4 :=
by
  sorry

end number_of_littering_citations_l2399_239940


namespace A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l2399_239901

def prob_A_wins_B_one_throw : ℚ := 1 / 3
def prob_tie_one_throw : ℚ := 1 / 3
def prob_A_wins_B_no_more_2_throws : ℚ := 4 / 9

def prob_C_treats_two_throws : ℚ := 2 / 9

def prob_C_treats_exactly_2_days_out_of_3 : ℚ := 28 / 243

theorem A_wins_B_no_more_than_two_throws (P1 : ℚ := prob_A_wins_B_one_throw) (P2 : ℚ := prob_tie_one_throw) :
  P1 + P2 * P1 = prob_A_wins_B_no_more_2_throws := 
by
  sorry

theorem C_treats_after_two_throws : prob_tie_one_throw ^ 2 = prob_C_treats_two_throws :=
by
  sorry

theorem C_treats_exactly_two_days (n : ℕ := 3) (k : ℕ := 2) (p_success : ℚ := prob_C_treats_two_throws) :
  (n.choose k) * (p_success ^ k) * ((1 - p_success) ^ (n - k)) = prob_C_treats_exactly_2_days_out_of_3 :=
by
  sorry

end A_wins_B_no_more_than_two_throws_C_treats_after_two_throws_C_treats_exactly_two_days_l2399_239901


namespace line_eq_l2399_239928

theorem line_eq (m b : ℝ) 
  (h_slope : m = (4 + 2) / (3 - 1)) 
  (h_point : -2 = m * 1 + b) :
  m + b = -2 :=
by
  sorry

end line_eq_l2399_239928


namespace inequalities_in_quadrants_l2399_239966

theorem inequalities_in_quadrants (x y : ℝ) :
  (y > - (1 / 2) * x + 6) ∧ (y > 3 * x - 4) → (x > 0) ∧ (y > 0) :=
  sorry

end inequalities_in_quadrants_l2399_239966


namespace correct_option_l2399_239926

theorem correct_option : Real.sqrt 12 - Real.sqrt 3 = Real.sqrt 3 := sorry

end correct_option_l2399_239926


namespace distance_between_Stockholm_and_Malmoe_l2399_239932

noncomputable def actualDistanceGivenMapDistanceAndScale (mapDistance : ℕ) (scale : ℕ) : ℕ :=
  mapDistance * scale

theorem distance_between_Stockholm_and_Malmoe (mapDistance : ℕ) (scale : ℕ) :
  mapDistance = 150 → scale = 20 → actualDistanceGivenMapDistanceAndScale mapDistance scale = 3000 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end distance_between_Stockholm_and_Malmoe_l2399_239932


namespace circle_center_x_coordinate_eq_l2399_239976

theorem circle_center_x_coordinate_eq (a : ℝ) (h : (∃ k : ℝ, ∀ x y : ℝ, x^2 + y^2 - a * x = k) ∧ (1 = a / 2)) : a = 2 :=
sorry

end circle_center_x_coordinate_eq_l2399_239976


namespace geese_ratio_l2399_239975

/-- Define the problem conditions --/

def lily_ducks := 20
def lily_geese := 10

def rayden_ducks : ℕ := 3 * lily_ducks
def total_lily_animals := lily_ducks + lily_geese
def total_rayden_animals := total_lily_animals + 70
def rayden_geese := total_rayden_animals - rayden_ducks

/-- Prove the desired ratio of the number of geese Rayden bought to the number of geese Lily bought --/
theorem geese_ratio : rayden_geese / lily_geese = 4 :=
sorry

end geese_ratio_l2399_239975
