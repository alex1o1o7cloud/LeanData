import Mathlib

namespace wizard_answers_bal_l829_82995

-- Define the types for human and zombie as truth-tellers and liars respectively
inductive WizardType
| human : WizardType
| zombie : WizardType

-- Define the meaning of "bal"
inductive BalMeaning
| yes : BalMeaning
| no : BalMeaning

-- Question asked to the wizard
def question (w : WizardType) (b : BalMeaning) : Prop :=
  match w, b with
  | WizardType.human, BalMeaning.yes => true
  | WizardType.human, BalMeaning.no => false
  | WizardType.zombie, BalMeaning.yes => false
  | WizardType.zombie, BalMeaning.no => true

-- Theorem stating the wizard will answer "bal" to the given question
theorem wizard_answers_bal (w : WizardType) (b : BalMeaning) :
  question w b = true ↔ b = BalMeaning.yes :=
by
  sorry

end wizard_answers_bal_l829_82995


namespace coin_outcomes_equivalent_l829_82938

theorem coin_outcomes_equivalent :
  let outcomes_per_coin := 2
  let total_coins := 3
  (outcomes_per_coin ^ total_coins) = 8 :=
by
  sorry

end coin_outcomes_equivalent_l829_82938


namespace lower_upper_bound_f_l829_82990

-- definition of the function f(n, d) as given in the problem
def func_f (n : ℕ) (d : ℕ) : ℕ :=
  -- placeholder definition; actual definition would rely on the described properties
  sorry

theorem lower_upper_bound_f (n d : ℕ) (hn : 0 < n) (hd : 0 < d) :
  (n-1) * 2^d + 1 ≤ func_f n d ∧ func_f n d ≤ (n-1) * n^d + 1 :=
by
  sorry

end lower_upper_bound_f_l829_82990


namespace cheryl_material_need_l829_82981

-- Cheryl's conditions
def cheryl_material_used (x : ℚ) : Prop :=
  x + 2/3 - 4/9 = 2/3

-- The proof problem statement
theorem cheryl_material_need : ∃ x : ℚ, cheryl_material_used x ∧ x = 4/9 :=
  sorry

end cheryl_material_need_l829_82981


namespace max_homework_time_l829_82904

theorem max_homework_time :
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  biology + history + geography = 180 :=
by
  let biology := 20
  let history := biology * 2
  let geography := history * 3
  show biology + history + geography = 180
  sorry

end max_homework_time_l829_82904


namespace find_expression_value_l829_82943

theorem find_expression_value (x : ℝ) (h : x + 1/x = 3) : 
  x^10 - 5 * x^6 + x^2 = 8436*x - 338 := 
by {
  sorry
}

end find_expression_value_l829_82943


namespace min_value_ineq_l829_82913

noncomputable def minimum_value (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) : ℝ :=
  1 / a + 4 / b

theorem min_value_ineq (a b : ℝ) (ha : 1 < a) (hb : 1 < b) (h : 4 * a + b = 1) :
  minimum_value a b ha hb h ≥ 16 :=
sorry

end min_value_ineq_l829_82913


namespace find_unknown_blankets_rate_l829_82915

noncomputable def unknown_blankets_rate : ℝ :=
  let total_cost_3_blankets := 3 * 100
  let discount := 0.10 * total_cost_3_blankets
  let cost_3_blankets_after_discount := total_cost_3_blankets - discount
  let cost_1_blanket := 150
  let tax := 0.15 * cost_1_blanket
  let cost_1_blanket_after_tax := cost_1_blanket + tax
  let total_avg_price_per_blanket := 150
  let total_blankets := 6
  let total_cost := total_avg_price_per_blanket * total_blankets
  (total_cost - cost_3_blankets_after_discount - cost_1_blanket_after_tax) / 2

theorem find_unknown_blankets_rate : unknown_blankets_rate = 228.75 :=
  by
    sorry

end find_unknown_blankets_rate_l829_82915


namespace product_of_four_consecutive_integers_is_perfect_square_l829_82964

theorem product_of_four_consecutive_integers_is_perfect_square :
  ∃ k : ℤ, ∃ n : ℤ, k = (n-1) * n * (n+1) * (n+2) ∧
    k = 0 ∧
    ((n = 0) ∨ (n = -1) ∨ (n = 1) ∨ (n = -2)) :=
by
  sorry

end product_of_four_consecutive_integers_is_perfect_square_l829_82964


namespace largest_divisor_of_m_l829_82980

theorem largest_divisor_of_m (m : ℕ) (h1 : 0 < m) (h2 : 39 ∣ m^2) : 39 ∣ m := sorry

end largest_divisor_of_m_l829_82980


namespace base_length_of_parallelogram_l829_82926

-- Definitions and conditions
def parallelogram_area (base altitude : ℝ) : ℝ := base * altitude
def altitude (base : ℝ) : ℝ := 2 * base

-- Main theorem to prove
theorem base_length_of_parallelogram (A : ℝ) (base : ℝ)
  (hA : A = 200) 
  (h_altitude : altitude base = 2 * base) 
  (h_area : parallelogram_area base (altitude base) = A) : 
  base = 10 := 
sorry

end base_length_of_parallelogram_l829_82926


namespace simon_paid_amount_l829_82945

theorem simon_paid_amount:
  let pansy_price := 2.50
  let hydrangea_price := 12.50
  let petunia_price := 1.00
  let pansies_count := 5
  let hydrangeas_count := 1
  let petunias_count := 5
  let discount_rate := 0.10
  let change_received := 23.00

  let total_cost_before_discount := (pansies_count * pansy_price) + (hydrangeas_count * hydrangea_price) + (petunias_count * petunia_price)
  let discount := discount_rate * total_cost_before_discount
  let total_cost_after_discount := total_cost_before_discount - discount
  let amount_paid_with := total_cost_after_discount + change_received

  amount_paid_with = 50.00 :=
by
  sorry

end simon_paid_amount_l829_82945


namespace rectangular_floor_length_l829_82931

theorem rectangular_floor_length
    (cost_per_square : ℝ)
    (total_cost : ℝ)
    (carpet_length : ℝ)
    (carpet_width : ℝ)
    (floor_width : ℝ)
    (floor_area : ℝ) 
    (H1 : cost_per_square = 15)
    (H2 : total_cost = 225)
    (H3 : carpet_length = 2)
    (H4 : carpet_width = 2)
    (H5 : floor_width = 6)
    (H6 : floor_area = floor_width * carpet_length * carpet_width * 15): 
    floor_area / floor_width = 10 :=
by
  sorry

end rectangular_floor_length_l829_82931


namespace complete_square_form_l829_82930

noncomputable def quadratic_expr (x : ℝ) : ℝ := x^2 - 10 * x + 15

theorem complete_square_form (b c : ℤ) (h : ∀ x : ℝ, quadratic_expr x = 0 ↔ (x + b)^2 = c) :
  b + c = 5 :=
sorry

end complete_square_form_l829_82930


namespace initial_quarters_l829_82907

variable (q : ℕ)

theorem initial_quarters (h : q + 3 = 11) : q = 8 :=
by
  sorry

end initial_quarters_l829_82907


namespace seeds_in_fourth_pot_l829_82971

theorem seeds_in_fourth_pot (total_seeds : ℕ) (total_pots : ℕ) (seeds_per_pot : ℕ) (first_three_pots : ℕ)
  (h1 : total_seeds = 10) (h2 : total_pots = 4) (h3 : seeds_per_pot = 3) (h4 : first_three_pots = 3) : 
  (total_seeds - (seeds_per_pot * first_three_pots)) = 1 :=
by
  sorry

end seeds_in_fourth_pot_l829_82971


namespace find_m_value_l829_82999

-- Condition: P(-m^2, 3) lies on the axis of symmetry of the parabola y^2 = mx
def point_on_axis_of_symmetry (m : ℝ) : Prop :=
  let P := (-m^2, 3)
  let axis_of_symmetry := (-m / 4)
  P.1 = axis_of_symmetry

theorem find_m_value (m : ℝ) (h : point_on_axis_of_symmetry m) : m = 1 / 4 :=
  sorry

end find_m_value_l829_82999


namespace sufficient_not_necessary_l829_82994

namespace ProofExample

variable {x : ℝ}

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | x < 2}

-- Theorem: "1 < x < 2" is a sufficient but not necessary condition for "x < 2" to hold.
theorem sufficient_not_necessary : 
  (∀ x, 1 < x ∧ x < 2 → x < 2) ∧ ¬(∀ x, x < 2 → 1 < x ∧ x < 2) := 
by
  sorry

end ProofExample

end sufficient_not_necessary_l829_82994


namespace real_roots_exist_for_all_real_K_l829_82968

theorem real_roots_exist_for_all_real_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x-1) * (x-2) * (x-3) :=
by
  sorry

end real_roots_exist_for_all_real_K_l829_82968


namespace buffy_breath_time_l829_82976

theorem buffy_breath_time (k : ℕ) (b : ℕ) (f : ℕ) 
  (h1 : k = 3 * 60) 
  (h2 : b = k - 20) 
  (h3 : f = b - 40) :
  f = 120 :=
by {
  sorry
}

end buffy_breath_time_l829_82976


namespace prime_cubed_plus_seven_composite_l829_82986

theorem prime_cubed_plus_seven_composite (P : ℕ) (hP_prime : Nat.Prime P) (hP3_plus_5_prime : Nat.Prime (P ^ 3 + 5)) : ¬ Nat.Prime (P ^ 3 + 7) :=
by
  sorry

end prime_cubed_plus_seven_composite_l829_82986


namespace overall_gain_is_2_89_l829_82927

noncomputable def overall_gain_percentage : ℝ :=
  let cost1 := 500000
  let gain1 := 0.10
  let sell1 := cost1 * (1 + gain1)

  let cost2 := 600000
  let loss2 := 0.05
  let sell2 := cost2 * (1 - loss2)

  let cost3 := 700000
  let gain3 := 0.15
  let sell3 := cost3 * (1 + gain3)

  let cost4 := 800000
  let loss4 := 0.12
  let sell4 := cost4 * (1 - loss4)

  let cost5 := 900000
  let gain5 := 0.08
  let sell5 := cost5 * (1 + gain5)

  let total_cost := cost1 + cost2 + cost3 + cost4 + cost5
  let total_sell := sell1 + sell2 + sell3 + sell4 + sell5
  let overall_gain := total_sell - total_cost
  (overall_gain / total_cost) * 100

theorem overall_gain_is_2_89 :
  overall_gain_percentage = 2.89 :=
by
  -- Proof goes here
  sorry

end overall_gain_is_2_89_l829_82927


namespace die_face_never_lays_on_board_l829_82928

structure Chessboard :=
(rows : ℕ)
(cols : ℕ)
(h_size : rows = 8 ∧ cols = 8)

structure Die :=
(faces : Fin 6 → Nat)  -- a die has 6 faces

structure Position :=
(x : ℕ)
(y : ℕ)

structure State :=
(position : Position)
(bottom_face : Fin 6)
(visited : Fin 64 → Bool)

def initial_position : Position := ⟨0, 0⟩  -- top-left corner (a1)

def initial_state (d : Die) : State :=
  { position := initial_position,
    bottom_face := 0,
    visited := λ _ => false }

noncomputable def can_roll_over_entire_board_without_one_face_touching (board : Chessboard) (d : Die) : Prop :=
  ∃ f : Fin 6, ∀ s : State, -- for some face f of the die
    ((s.position.x < board.rows ∧ s.position.y < board.cols) → 
      s.visited (⟨s.position.x + board.rows * s.position.y, by sorry⟩) = true) → -- every cell visited
      ¬(s.bottom_face = f) -- face f is never the bottom face

theorem die_face_never_lays_on_board (board : Chessboard) (d : Die) :
  can_roll_over_entire_board_without_one_face_touching board d :=
  sorry

end die_face_never_lays_on_board_l829_82928


namespace part1_part2_l829_82997

-- Define properties for the first part of the problem
def condition1 (weightA weightB : ℕ) : Prop :=
  weightA + weightB = 7500 ∧ weightA = 3 * weightB / 2

def question1_answer : Prop :=
  ∃ weightA weightB : ℕ, condition1 weightA weightB ∧ weightA = 4500 ∧ weightB = 3000

-- Combined condition for the second part of the problem scenarios
def condition2a (y : ℕ) : Prop := y ≤ 1800 ∧ 18 * y - 10 * y = 17400
def condition2b (y : ℕ) : Prop := 1800 < y ∧ y ≤ 3000 ∧ 18 * y - (15 * y - 9000) = 17400
def condition2c (y : ℕ) : Prop := y > 3000 ∧ 18 * y - (20 * y - 24000) = 17400

def question2_answer : Prop :=
  (∃ y : ℕ, condition2b y ∧ y = 2800) ∨ (∃ y : ℕ, condition2c y ∧ y = 3300)

-- The Lean statements for both parts of the problem
theorem part1 : question1_answer := sorry

theorem part2 : question2_answer := sorry

end part1_part2_l829_82997


namespace tiling_2x12_l829_82935

def d : Nat → Nat
| 0     => 0  -- Unused but for safety in function definition
| 1     => 1
| 2     => 2
| (n+1) => d n + d (n-1)

theorem tiling_2x12 : d 12 = 233 := by
  sorry

end tiling_2x12_l829_82935


namespace smallest_five_digit_divisible_by_2_5_11_l829_82942

theorem smallest_five_digit_divisible_by_2_5_11 : ∃ n, n >= 10000 ∧ n % 2 = 0 ∧ n % 5 = 0 ∧ n % 11 = 0 ∧ n = 10010 :=
by
  sorry

end smallest_five_digit_divisible_by_2_5_11_l829_82942


namespace number_of_valid_numbers_l829_82974

def is_valid_number (N : ℕ) : Prop :=
  N ≥ 1000 ∧ N < 10000 ∧ ∃ a x : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ x < 1000 ∧ 
  N = 1000 * a + x ∧ x = N / 9

theorem number_of_valid_numbers : ∃ (n : ℕ), n = 7 ∧ ∀ N, is_valid_number N → N < 1000 * (n + 2) := 
sorry

end number_of_valid_numbers_l829_82974


namespace value_of_k_l829_82925

   noncomputable def k (a b : ℝ) : ℝ := 3 / 4

   theorem value_of_k (a b k : ℝ) 
     (h1: b = 4 * k + 1) 
     (h2: 5 = a * k + 1) 
     (h3: b + 1 = a * k + 1) : 
     k = 3 / 4 := 
   by 
     -- Proof goes here 
     sorry
   
end value_of_k_l829_82925


namespace average_production_last_5_days_l829_82985

theorem average_production_last_5_days
  (avg_first_25_days : ℕ → ℕ → ℕ → ℕ → Prop)
  (avg_monthly : ℕ)
  (total_days : ℕ)
  (days_first_period : ℕ)
  (avg_production_first_period : ℕ)
  (avg_total_monthly : ℕ)
  (days_second_period : ℕ)
  (total_production_five_days : ℕ):
  (days_first_period = 25) →
  (avg_production_first_period = 50) →
  (avg_total_monthly = 48) →
  (total_production_five_days = 190) →
  (days_second_period = 5) →
  avg_first_25_days days_first_period avg_production_first_period 
  (days_first_period * avg_production_first_period) avg_total_monthly ∧
  avg_monthly = avg_total_monthly →
  ((days_first_period + days_second_period) * avg_monthly - 
  days_first_period * avg_production_first_period = total_production_five_days) →
  (total_production_five_days / days_second_period = 38) := sorry

end average_production_last_5_days_l829_82985


namespace Faraway_not_possible_sum_l829_82969

theorem Faraway_not_possible_sum (h g : ℕ) : (74 ≠ 21 * h + 6 * g) ∧ (89 ≠ 21 * h + 6 * g) :=
by
  sorry

end Faraway_not_possible_sum_l829_82969


namespace solve_for_x_l829_82906

theorem solve_for_x (x : ℝ) (h : (x - 15) / 3 = (3 * x + 10) / 8) : x = -150 := 
by
  sorry

end solve_for_x_l829_82906


namespace dice_probability_four_less_than_five_l829_82992

noncomputable def probability_exactly_four_less_than_five (n : ℕ) : ℚ :=
  if n = 8 then (Nat.choose 8 4) * (1 / 2)^8 else 0

theorem dice_probability_four_less_than_five : probability_exactly_four_less_than_five 8 = 35 / 128 :=
by
  -- statement is correct, proof to be provided
  sorry

end dice_probability_four_less_than_five_l829_82992


namespace expression_value_l829_82909

theorem expression_value : 
  29^2 - 27^2 + 25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 389 :=
by
  sorry

end expression_value_l829_82909


namespace spadesuit_evaluation_l829_82919

def spadesuit (a b : ℝ) : ℝ := abs (a - b)

theorem spadesuit_evaluation : spadesuit 1.5 (spadesuit 2.5 (spadesuit 4.5 6)) = 0.5 :=
by
  sorry

end spadesuit_evaluation_l829_82919


namespace cubic_expression_l829_82970

theorem cubic_expression (a b c : ℝ) (h₁ : a + b + c = 12) (h₂ : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 1008 :=
sorry

end cubic_expression_l829_82970


namespace proportion_correct_l829_82944

theorem proportion_correct (x y : ℝ) (h1 : 2 * y = 5 * x) (h2 : x ≠ 0 ∧ y ≠ 0) : x / y = 2 / 5 := 
sorry

end proportion_correct_l829_82944


namespace gain_percent_l829_82951

theorem gain_percent (cost_price selling_price : ℝ) (h1 : cost_price = 900) (h2 : selling_price = 1440) : 
  ((selling_price - cost_price) / cost_price) * 100 = 60 :=
by
  sorry

end gain_percent_l829_82951


namespace arithmetic_sequence_a6_l829_82975

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h_root1 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 2 = x)
  (h_root2 : ∃ x : ℝ, x^2 + 12 * x - 8 = 0 ∧ a 10 = x) : 
  a 6 = -6 := 
by
  sorry

end arithmetic_sequence_a6_l829_82975


namespace cost_proof_l829_82950

-- Given conditions
def total_cost : Int := 190
def working_days : Int := 19
def trips_per_day : Int := 2
def total_trips : Int := working_days * trips_per_day

-- Define the problem to prove
def cost_per_trip : Int := 5

theorem cost_proof : (total_cost / total_trips = cost_per_trip) := 
by 
  -- This is a placeholder to indicate that we're skipping the proof
  sorry

end cost_proof_l829_82950


namespace min_polyline_distance_between_circle_and_line_l829_82937

def polyline_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

def on_circle (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

def on_line (Q : ℝ × ℝ) : Prop :=
  2 * Q.1 + Q.2 = 2 * Real.sqrt 5

theorem min_polyline_distance_between_circle_and_line :
  ∃ P Q, on_circle P ∧ on_line Q ∧ polyline_distance P Q = (Real.sqrt 5) / 2 :=
by
  sorry

end min_polyline_distance_between_circle_and_line_l829_82937


namespace find_x_of_total_area_l829_82960

theorem find_x_of_total_area 
  (x : Real)
  (h_triangle : (1/2) * (4 * x) * (3 * x) = 6 * x^2)
  (h_square1 : (3 * x)^2 = 9 * x^2)
  (h_square2 : (6 * x)^2 = 36 * x^2)
  (h_total : 6 * x^2 + 9 * x^2 + 36 * x^2 = 700) :
  x = Real.sqrt (700 / 51) :=
by {
  sorry
}

end find_x_of_total_area_l829_82960


namespace find_numbers_l829_82936

theorem find_numbers (x y : ℤ) (h1 : x + y = 18) (h2 : x - y = 24) : x = 21 ∧ y = -3 :=
by
  sorry

end find_numbers_l829_82936


namespace people_count_l829_82920

theorem people_count (wheels_per_person total_wheels : ℕ) (h1 : wheels_per_person = 4) (h2 : total_wheels = 320) :
  total_wheels / wheels_per_person = 80 :=
sorry

end people_count_l829_82920


namespace distance_between_A_and_B_l829_82957

theorem distance_between_A_and_B 
  (v1 v2: ℝ) (s: ℝ)
  (h1 : (s - 8) / v1 = s / v2)
  (h2 : s / (2 * v1) = (s - 15) / v2)
  (h3: s = 40) : 
  s = 40 := 
sorry

end distance_between_A_and_B_l829_82957


namespace farm_total_amount_90000_l829_82911

-- Defining the conditions
def apples_produce (mangoes: ℕ) : ℕ := 2 * mangoes
def oranges_produce (mangoes: ℕ) : ℕ := mangoes + 200

-- Defining the total produce of all fruits
def total_produce (mangoes: ℕ) : ℕ := apples_produce mangoes + mangoes + oranges_produce mangoes

-- Defining the price per kg
def price_per_kg : ℕ := 50

-- Defining the total amount from selling all fruits
noncomputable def total_amount (mangoes: ℕ) : ℕ := total_produce mangoes * price_per_kg

-- Proving that the total amount he got in that season is $90,000
theorem farm_total_amount_90000 : total_amount 400 = 90000 := by
  sorry

end farm_total_amount_90000_l829_82911


namespace optimal_tower_configuration_l829_82946

theorem optimal_tower_configuration (x y : ℕ) (h : x + 2 * y = 30) :
    x * y ≤ 112 := by
  sorry

end optimal_tower_configuration_l829_82946


namespace anthony_path_shortest_l829_82988

noncomputable def shortest_distance (A B C D M : ℝ) : ℝ :=
  4 + 2 * Real.sqrt 3

theorem anthony_path_shortest {A B C D : ℝ} (M : ℝ) (side_length : ℝ) (h : side_length = 4) : 
  shortest_distance A B C D M = 4 + 2 * Real.sqrt 3 :=
by 
  sorry

end anthony_path_shortest_l829_82988


namespace xiao_ming_proposition_false_l829_82934

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m * m ≤ n → m = 1 ∨ m = n → m ∣ n

def check_xiao_ming_proposition : Prop :=
  ∃ n : ℕ, ∃ (k : ℕ), k < n → ∃ (p q : ℕ), p = q → n^2 - n + 11 = p * q ∧ p > 1 ∧ q > 1

theorem xiao_ming_proposition_false : ¬ (∀ n: ℕ, is_prime (n^2 - n + 11)) :=
by
  sorry

end xiao_ming_proposition_false_l829_82934


namespace equilateral_triangle_area_in_circle_l829_82924

theorem equilateral_triangle_area_in_circle (r : ℝ) (h : r = 9) :
  let s := 2 * r * Real.sin (π / 3)
  let A := (Real.sqrt 3 / 4) * s^2
  A = (243 * Real.sqrt 3) / 4 := by
  sorry

end equilateral_triangle_area_in_circle_l829_82924


namespace factor_expression_l829_82955

theorem factor_expression (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) :=
by
  sorry

end factor_expression_l829_82955


namespace complement_M_in_U_l829_82905

-- Define the universal set U and set M
def U : Finset ℕ := {4, 5, 6, 8, 9}
def M : Finset ℕ := {5, 6, 8}

-- Define the complement of M in U
def complement (U M : Finset ℕ) : Finset ℕ := U \ M

-- Prove that the complement of M in U is {4, 9}
theorem complement_M_in_U : complement U M = {4, 9} := by
  sorry

end complement_M_in_U_l829_82905


namespace cylinder_volume_ratio_l829_82963

theorem cylinder_volume_ratio
  (S1 S2 : ℝ) (v1 v2 : ℝ)
  (lateral_area_equal : 2 * Real.pi * S1.sqrt = 2 * Real.pi * S2.sqrt)
  (base_area_ratio : S1 / S2 = 16 / 9) :
  v1 / v2 = 4 / 3 :=
by
  sorry

end cylinder_volume_ratio_l829_82963


namespace intersection_chord_line_eq_l829_82958

noncomputable def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 10
noncomputable def circle2 (x y : ℝ) : Prop := (x + 6)^2 + (y + 3)^2 = 50

theorem intersection_chord_line_eq (x y : ℝ) (h1 : circle1 x y) (h2 : circle2 x y) : 
  2 * x + y = 0 :=
sorry

end intersection_chord_line_eq_l829_82958


namespace marla_night_cost_is_correct_l829_82982

def lizard_value_bc := 8 -- 1 lizard is worth 8 bottle caps
def lizard_value_gw := 5 / 3 -- 3 lizards are worth 5 gallons of water
def horse_value_gw := 80 -- 1 horse is worth 80 gallons of water
def marla_daily_bc := 20 -- Marla can scavenge 20 bottle caps each day
def marla_days := 24 -- It takes Marla 24 days to collect the bottle caps

noncomputable def marla_night_cost_bc : ℕ :=
((marla_daily_bc * marla_days) - (horse_value_gw / lizard_value_gw * (3 * lizard_value_bc))) / marla_days

theorem marla_night_cost_is_correct :
  marla_night_cost_bc = 4 := by
  sorry

end marla_night_cost_is_correct_l829_82982


namespace mountaineers_arrangement_l829_82961
open BigOperators

-- Definition to state the number of combinations
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The main statement translating our problem
theorem mountaineers_arrangement :
  (choose 4 2) * (choose 6 2) = 120 := by
  sorry

end mountaineers_arrangement_l829_82961


namespace find_m_l829_82962

theorem find_m (m : ℝ) 
  (h : (1 : ℝ) * (-3 : ℝ) + (3 : ℝ) * ((3 : ℝ) + 2 * m) = 0) : 
  m = -1 :=
by sorry

end find_m_l829_82962


namespace cannot_be_external_diagonals_l829_82902

theorem cannot_be_external_diagonals (a b c : ℕ) : 
  ¬(3^2 + 4^2 = 6^2) :=
by
  sorry

end cannot_be_external_diagonals_l829_82902


namespace probability_of_union_l829_82984

def total_cards : ℕ := 52
def king_of_hearts : ℕ := 1
def spades : ℕ := 13

theorem probability_of_union :
  let P_A := king_of_hearts / total_cards
  let P_B := spades / total_cards
  (P_A + P_B) = (7 / 26) :=
by
  sorry

end probability_of_union_l829_82984


namespace number_of_oarsmen_l829_82923

-- Define the conditions
variables (n : ℕ)
variables (W : ℕ)
variables (h_avg_increase : (W + 40) / n = W / n + 2)

-- Lean 4 statement without the proof
theorem number_of_oarsmen : n = 20 :=
by
  sorry

end number_of_oarsmen_l829_82923


namespace perpendicular_lines_l829_82973

theorem perpendicular_lines (m : ℝ) : 
    (∀ x y : ℝ, x - 2 * y + 5 = 0) ∧ (∀ x y : ℝ, 2 * x + m * y - 6 = 0) → m = -1 :=
by
  sorry

end perpendicular_lines_l829_82973


namespace factorization_problem_1_factorization_problem_2_l829_82949

-- Problem 1: Factorize 2(m-n)^2 - m(n-m) and show it equals (n-m)(2n - 3m)
theorem factorization_problem_1 (m n : ℝ) :
  2 * (m - n)^2 - m * (n - m) = (n - m) * (2 * n - 3 * m) :=
by
  sorry

-- Problem 2: Factorize -4xy^2 + 4x^2y + y^3 and show it equals y(2x - y)^2
theorem factorization_problem_2 (x y : ℝ) :
  -4 * x * y^2 + 4 * x^2 * y + y^3 = y * (2 * x - y)^2 :=
by
  sorry

end factorization_problem_1_factorization_problem_2_l829_82949


namespace equal_heights_of_cylinder_and_cone_l829_82977

theorem equal_heights_of_cylinder_and_cone
  (r h : ℝ)
  (hc : h > 0)
  (hr : r > 0)
  (V_cylinder V_cone : ℝ)
  (V_cylinder_eq : V_cylinder = π * r ^ 2 * h)
  (V_cone_eq : V_cone = 1/3 * π * r ^ 2 * h)
  (volume_ratio : V_cylinder / V_cone = 3) :
h = h := -- Since we are given that the heights are initially the same
sorry

end equal_heights_of_cylinder_and_cone_l829_82977


namespace q_transformation_l829_82987

theorem q_transformation (w m z : ℝ) (q : ℝ) (h_q : q = 5 * w / (4 * m * z^2)) :
  let w' := 4 * w
  let m' := 2 * m
  let z' := 3 * z
  q = 5 * w / (4 * m * z^2) → (5 * w') / (4 * m' * (z'^2)) = (5 / 18) * q := by
  sorry

end q_transformation_l829_82987


namespace mother_older_than_twice_petra_l829_82941

def petra_age : ℕ := 11
def mother_age : ℕ := 36

def twice_petra_age : ℕ := 2 * petra_age

theorem mother_older_than_twice_petra : mother_age - twice_petra_age = 14 := by
  sorry

end mother_older_than_twice_petra_l829_82941


namespace common_ratio_geometric_series_l829_82901

-- Define the first three terms of the series
def first_term := (-3: ℚ) / 5
def second_term := (-5: ℚ) / 3
def third_term := (-125: ℚ) / 27

-- Prove that the common ratio = 25/9
theorem common_ratio_geometric_series :
  (second_term / first_term) = (25 : ℚ) / 9 :=
by
  sorry

end common_ratio_geometric_series_l829_82901


namespace min_max_diff_val_l829_82947

def find_min_max_diff (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : ℝ :=
  let m := 0
  let M := 1
  M - m

theorem min_max_diff_val (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : find_min_max_diff x y hx hy = 1 :=
by sorry

end min_max_diff_val_l829_82947


namespace white_marbles_count_l829_82989

section Marbles

variable (total_marbles black_marbles red_marbles green_marbles white_marbles : Nat)

theorem white_marbles_count
  (h_total: total_marbles = 60)
  (h_black: black_marbles = 32)
  (h_red: red_marbles = 10)
  (h_green: green_marbles = 5)
  (h_color: total_marbles = black_marbles + red_marbles + green_marbles + white_marbles) : 
  white_marbles = 13 := 
by
  sorry 

end Marbles

end white_marbles_count_l829_82989


namespace polar_to_cartesian_equiv_l829_82948

noncomputable def polar_to_cartesian (rho theta : ℝ) : Prop :=
  let x := rho * Real.cos theta
  let y := rho * Real.sin theta
  (Real.sqrt 3 * x + y = 2) ↔ (rho * Real.cos (theta - Real.pi / 6) = 1)

theorem polar_to_cartesian_equiv (rho theta : ℝ) : polar_to_cartesian rho theta :=
by
  sorry

end polar_to_cartesian_equiv_l829_82948


namespace measure_of_angle4_l829_82967

def angle1 := 62
def angle2 := 36
def angle3 := 24
def angle4 : ℕ := 122

theorem measure_of_angle4 (d e : ℕ) (h1 : angle1 + angle2 + angle3 + d + e = 180) (h2 : d + e = 58) :
  angle4 = 180 - (angle1 + angle2 + angle3 + d + e) :=
by
  sorry

end measure_of_angle4_l829_82967


namespace max_omega_value_l829_82940

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  Real.sin (ω * x + φ)

theorem max_omega_value 
  (ω : ℝ) 
  (φ : ℝ) 
  (hω : 0 < ω) 
  (hφ : |φ| ≤ Real.pi / 2)
  (h_zero : f ω φ (-Real.pi / 4) = 0)
  (h_sym : f ω φ (Real.pi / 4) = f ω φ (-Real.pi / 4))
  (h_monotonic : ∀ x₁ x₂, (Real.pi / 18) < x₁ → x₁ < x₂ → x₂ < (5 * Real.pi / 36) → f ω φ x₁ < f ω φ x₂) :
  ω = 9 :=
  sorry

end max_omega_value_l829_82940


namespace power_mod_lemma_l829_82914

theorem power_mod_lemma : (7^137 % 13) = 11 := by
  sorry

end power_mod_lemma_l829_82914


namespace find_m_l829_82954

theorem find_m (a b m : ℤ) (h1 : a - b = 6) (h2 : a + b = 0) : 2 * a + b = m → m = 3 :=
by
  sorry

end find_m_l829_82954


namespace infinite_n_exist_l829_82912

def S (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem infinite_n_exist (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ᶠ n in at_top, S n ≡ n [MOD p] :=
sorry

end infinite_n_exist_l829_82912


namespace total_weight_is_1kg_total_weight_in_kg_eq_1_l829_82966

theorem total_weight_is_1kg 
  (weight_msg : ℕ := 80)
  (weight_salt : ℕ := 500)
  (weight_detergent : ℕ := 420) :
  (weight_msg + weight_salt + weight_detergent) = 1000 := by
sorry

theorem total_weight_in_kg_eq_1 
  (total_weight_g : ℕ := weight_msg + weight_salt + weight_detergent) :
  (total_weight_g = 1000) → (total_weight_g / 1000 = 1) := by
sorry

end total_weight_is_1kg_total_weight_in_kg_eq_1_l829_82966


namespace river_width_l829_82910

theorem river_width (boat_max_speed : ℝ) (river_current_speed : ℝ) (time_to_cross : ℝ) (width : ℝ) :
  boat_max_speed = 4 ∧ river_current_speed = 3 ∧ time_to_cross = 2 ∧ width = 8 → 
  width = boat_max_speed * time_to_cross := by
  intros h
  cases h
  sorry

end river_width_l829_82910


namespace measure_diagonal_of_brick_l829_82956

def RectangularParallelepiped (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def DiagonalMeasurementPossible (a b c : ℝ) : Prop :=
  ∃ d : ℝ, d = (a^2 + b^2 + c^2)^(1/2)

theorem measure_diagonal_of_brick (a b c : ℝ) 
  (h : RectangularParallelepiped a b c) : DiagonalMeasurementPossible a b c :=
by
  sorry

end measure_diagonal_of_brick_l829_82956


namespace find_principal_amount_l829_82953

-- Define the given conditions
def interest_rate1 : ℝ := 0.08
def interest_rate2 : ℝ := 0.10
def interest_rate3 : ℝ := 0.12
def period1 : ℝ := 4
def period2 : ℝ := 6
def period3 : ℝ := 5
def total_interest_paid : ℝ := 12160

-- Goal is to find the principal amount P
theorem find_principal_amount (P : ℝ) :
  total_interest_paid = P * (interest_rate1 * period1 + interest_rate2 * period2 + interest_rate3 * period3) →
  P = 8000 :=
by
  sorry

end find_principal_amount_l829_82953


namespace correct_solutions_l829_82903

theorem correct_solutions (x y z t : ℕ) : 
  (x^2 + t^2) * (z^2 + y^2) = 50 → 
  (x = 1 ∧ y = 1 ∧ z = 2 ∧ t = 3) ∨ 
  (x = 3 ∧ y = 2 ∧ z = 1 ∧ t = 1) ∨ 
  (x = 4 ∧ y = 1 ∧ z = 3 ∧ t = 1) ∨ 
  (x = 1 ∧ y = 3 ∧ z = 4 ∧ t = 1) :=
sorry

end correct_solutions_l829_82903


namespace not_prime_n_quad_plus_n_sq_plus_one_l829_82939

theorem not_prime_n_quad_plus_n_sq_plus_one (n : ℕ) (h : n ≥ 2) : ¬Prime (n^4 + n^2 + 1) :=
by
  sorry

end not_prime_n_quad_plus_n_sq_plus_one_l829_82939


namespace subset_of_positive_reals_l829_82979

def M := { x : ℝ | x > -1 }

theorem subset_of_positive_reals : {0} ⊆ M :=
by
  sorry

end subset_of_positive_reals_l829_82979


namespace min_n_1014_dominoes_l829_82952

theorem min_n_1014_dominoes (n : ℕ) :
  (n + 1) ^ 2 ≥ 6084 → n ≥ 77 :=
sorry

end min_n_1014_dominoes_l829_82952


namespace simplify_expression_correct_l829_82932

noncomputable def simplify_expression : ℝ :=
  2 * Real.sqrt (3 + Real.sqrt (5 - Real.sqrt (13 + Real.sqrt (48))))

theorem simplify_expression_correct : simplify_expression = (Real.sqrt 6) + (Real.sqrt 2) :=
  sorry

end simplify_expression_correct_l829_82932


namespace pebbles_collected_by_tenth_day_l829_82900

-- Define the initial conditions
def a : ℕ := 2
def r : ℕ := 2
def n : ℕ := 10

-- Total pebbles collected by the end of the 10th day
def total_pebbles (a r n : ℕ) : ℕ :=
  a * (r ^ n - 1) / (r - 1)

-- Proof statement
theorem pebbles_collected_by_tenth_day : total_pebbles a r n = 2046 :=
  by sorry

end pebbles_collected_by_tenth_day_l829_82900


namespace no_solutions_a_l829_82929

theorem no_solutions_a (x y : ℤ) : x^2 + y^2 ≠ 2003 := 
sorry

end no_solutions_a_l829_82929


namespace james_beats_old_record_l829_82991

def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def two_point_conversions : ℕ := 6
def points_per_two_point_conversion : ℕ := 2
def field_goals : ℕ := 8
def points_per_field_goal : ℕ := 3
def extra_points : ℕ := 20
def points_per_extra_point : ℕ := 1
def old_record : ℕ := 300

theorem james_beats_old_record :
  touchdowns_per_game * points_per_touchdown * games_in_season +
  two_point_conversions * points_per_two_point_conversion +
  field_goals * points_per_field_goal +
  extra_points * points_per_extra_point - old_record = 116 := by
  sorry -- Proof is omitted.

end james_beats_old_record_l829_82991


namespace simplify_expression_l829_82978

variable (x y : ℝ)

theorem simplify_expression : (15 * x + 35 * y) + (20 * x + 45 * y) - (8 * x + 40 * y) = 27 * x + 40 * y :=
by
  sorry

end simplify_expression_l829_82978


namespace ratio_of_logs_l829_82998

noncomputable def log_base (b x : ℝ) := (Real.log x) / (Real.log b)

theorem ratio_of_logs (a b : ℝ) 
    (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : log_base 8 a = log_base 18 b)
    (h4 : log_base 18 b = log_base 32 (a + b)) : 
    b / a = (1 + Real.sqrt 5) / 2 :=
by sorry

end ratio_of_logs_l829_82998


namespace numeral_eq_7000_l829_82917

theorem numeral_eq_7000 
  (local_value face_value numeral : ℕ)
  (h1 : face_value = 7)
  (h2 : local_value - face_value = 6993) : 
  numeral = 7000 :=
by
  sorry

end numeral_eq_7000_l829_82917


namespace naomi_wash_time_l829_82921

theorem naomi_wash_time (C T S : ℕ) (h₁ : T = 2 * C) (h₂ : S = 2 * C - 15) (h₃ : C + T + S = 135) : C = 30 :=
by
  sorry

end naomi_wash_time_l829_82921


namespace number_of_sets_l829_82993

theorem number_of_sets (A : Set ℕ) : ∃ s : Finset (Set ℕ), 
  (∀ x ∈ s, ({1} ⊂ x ∧ x ⊆ {1, 2, 3, 4})) ∧ s.card = 7 :=
sorry

end number_of_sets_l829_82993


namespace focus_of_parabola_l829_82965

theorem focus_of_parabola (x y : ℝ) (h : y = 2 * x^2) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / 8) :=
by
  sorry

end focus_of_parabola_l829_82965


namespace x_intercept_is_7_0_l829_82972

-- Define the given line equation
def line_eq (x y : ℚ) : Prop := 4 * x + 7 * y = 28

-- State the theorem we want to prove
theorem x_intercept_is_7_0 :
  ∃ x : ℚ, ∃ y : ℚ, line_eq x y ∧ y = 0 ∧ x = 7 :=
by
  sorry

end x_intercept_is_7_0_l829_82972


namespace average_marks_of_a_b_c_d_l829_82922

theorem average_marks_of_a_b_c_d (A B C D E : ℕ)
  (h1 : (A + B + C) / 3 = 48)
  (h2 : A = 43)
  (h3 : (B + C + D + E) / 4 = 48)
  (h4 : E = D + 3) :
  (A + B + C + D) / 4 = 47 :=
by
  -- This theorem will be justified
  admit

end average_marks_of_a_b_c_d_l829_82922


namespace infinite_radical_solution_l829_82983

theorem infinite_radical_solution (x : ℝ) (hx : x = Real.sqrt (20 + x)) : x = 5 :=
by sorry

end infinite_radical_solution_l829_82983


namespace relationship_among_a_b_c_l829_82918

noncomputable def a := (1/2)^(2/3)
noncomputable def b := (1/5)^(2/3)
noncomputable def c := (1/2)^(1/3)

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l829_82918


namespace product_zero_when_b_is_3_l829_82916

theorem product_zero_when_b_is_3 (b : ℤ) (h : b = 3) :
  (b - 13) * (b - 12) * (b - 11) * (b - 10) * (b - 9) * (b - 8) * (b - 7) * (b - 6) *
  (b - 5) * (b - 4) * (b - 3) * (b - 2) * (b - 1) * b = 0 :=
by {
  sorry
}

end product_zero_when_b_is_3_l829_82916


namespace cube_iff_diagonal_perpendicular_l829_82996

-- Let's define the rectangular parallelepiped as a type
structure RectParallelepiped :=
-- Define the property of being a cube
(isCube : Prop)

-- Define the property q: any diagonal of the parallelepiped is perpendicular to the diagonal of its non-intersecting face
def diagonal_perpendicular (S : RectParallelepiped) : Prop := 
 sorry -- This depends on how you define diagonals and perpendicularity within the structure

-- Prove the biconditional relationship
theorem cube_iff_diagonal_perpendicular (S : RectParallelepiped) :
 S.isCube ↔ diagonal_perpendicular S :=
sorry

end cube_iff_diagonal_perpendicular_l829_82996


namespace smallest_value_of_n_l829_82933

theorem smallest_value_of_n (a b c m n : ℕ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : a + b + c = 2010) (h4 : (a! * b! * c!) = m * 10 ^ n) : ∃ n, n = 500 := 
sorry

end smallest_value_of_n_l829_82933


namespace acute_angle_l829_82959

variables (x : ℝ)

def a : ℝ × ℝ := (2, x)
def b : ℝ × ℝ := (1, 3)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem acute_angle (x : ℝ) : 
  (-2 / 3 < x) → x ≠ -2 / 3 → dot_product (2, x) (1, 3) > 0 :=
by
  intros h1 h2
  sorry

end acute_angle_l829_82959


namespace expected_revenue_day_14_plan_1_more_reasonable_plan_l829_82908

-- Define the initial conditions
def initial_valuation : ℕ := 60000
def rain_probability : ℚ := 0.4
def no_rain_probability : ℚ := 0.6
def hiring_cost : ℕ := 32000

-- Calculate the expected revenue if Plan ① is adopted
def expected_revenue_plan_1_day_14 : ℚ :=
  (initial_valuation / 10000) * (1/2 * rain_probability + no_rain_probability)

-- Calculate the total revenue for Plan ①
def total_revenue_plan_1 : ℚ :=
  (initial_valuation / 10000) + 2 * expected_revenue_plan_1_day_14

-- Calculate the total revenue for Plan ②
def total_revenue_plan_2 : ℚ :=
  3 * (initial_valuation / 10000) - (hiring_cost / 10000)

-- Define the lemmas to prove
theorem expected_revenue_day_14_plan_1 :
  expected_revenue_plan_1_day_14 = 4.8 := 
  by sorry

theorem more_reasonable_plan :
  total_revenue_plan_1 > total_revenue_plan_2 :=
  by sorry

end expected_revenue_day_14_plan_1_more_reasonable_plan_l829_82908
