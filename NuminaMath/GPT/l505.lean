import Mathlib

namespace nine_questions_insufficient_l505_50560

/--
We have 5 stones with distinct weights and we are allowed to ask nine questions of the form
"Is it true that A < B < C?". Prove that nine such questions are insufficient to always determine
the unique ordering of these stones.
-/
theorem nine_questions_insufficient (stones : Fin 5 → Nat) 
  (distinct_weights : ∀ i j : Fin 5, i ≠ j → stones i ≠ stones j) :
  ¬ (∃ f : { q : Fin 125 | q.1 ≤ 8 } → (Fin 5 → Fin 5 → Fin 5 → Bool),
    ∀ w1 w2 w3 w4 w5 : Fin 120,
      (f ⟨0, sorry⟩) = sorry  -- This line only represents the existence of 9 questions
      )
:=
sorry

end nine_questions_insufficient_l505_50560


namespace parabola_vertex_l505_50588

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ v : ℝ × ℝ, (v.1 = -1 ∧ v.2 = 4) ∧ ∀ (x : ℝ), (x^2 + 2*x + 5 = ((x + 1)^2 + 4))) :=
by
  sorry

end parabola_vertex_l505_50588


namespace fraction_to_decimal_l505_50524

theorem fraction_to_decimal : (7 : ℝ) / 16 = 0.4375 := 
by
  sorry

end fraction_to_decimal_l505_50524


namespace seating_possible_l505_50506

theorem seating_possible (n : ℕ) (guests : Fin (2 * n) → Finset (Fin (2 * n))) 
  (h1 : ∀ i, n ≤ (guests i).card)
  (h2 : ∀ i j, (i ≠ j) → i ∈ guests j → j ∈ guests i) : 
  ∃ (a b c d : Fin (2 * n)), 
    (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ d) ∧ (d ≠ a) ∧
    (a ∈ guests b) ∧ (b ∈ guests c) ∧ (c ∈ guests d) ∧ (d ∈ guests a) := 
sorry

end seating_possible_l505_50506


namespace problem_statement_l505_50562

theorem problem_statement (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 3 = 973 := 
by
  sorry

end problem_statement_l505_50562


namespace general_term_of_geometric_sequence_l505_50597

variable (a : ℕ → ℝ) (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem general_term_of_geometric_sequence
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4)
  (hq : is_geometric_sequence a q)
  (q := 1/2) :
  ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ * q^(n - 1) :=
sorry

end general_term_of_geometric_sequence_l505_50597


namespace determine_parabola_equation_l505_50553

-- Given conditions
variable (p : ℝ) (h_p : p > 0)
variable (x1 x2 : ℝ)
variable (AF BF : ℝ)
variable (h_AF : AF = x1 + p / 2)
variable (h_BF : BF = x2 + p / 2)
variable (h_AF_value : AF = 2)
variable (h_BF_value : BF = 3)

-- Prove the equation of the parabola
theorem determine_parabola_equation (h1 : x1 + x2 = 5 - p)
(h2 : x1 * x2 = p^2 / 4)
(h3 : AF * BF = 6) :
  y^2 = (24/5 : ℝ) * x := 
sorry

end determine_parabola_equation_l505_50553


namespace correct_option_l505_50574

variable (f : ℝ → ℝ)
variable (h_diff : ∀ x : ℝ, differentiable_at ℝ f x)
variable (h_cond : ∀ x : ℝ, f x > deriv f x)

theorem correct_option :
  e ^ 2016 * f (-2016) > f 0 ∧ f 2016 < e ^ 2016 * f 0 :=
sorry

end correct_option_l505_50574


namespace school_team_profit_is_333_l505_50507

noncomputable def candy_profit (total_bars : ℕ) (price_800_bars : ℕ) (price_400_bars : ℕ) (sold_600_bars_price : ℕ) (remaining_600_bars_price : ℕ) : ℚ :=
  let cost_800_bars := 800 / 3
  let cost_400_bars := 400 / 4
  let total_cost := cost_800_bars + cost_400_bars
  let revenue_sold_600_bars := 600 / 2
  let revenue_remaining_600_bars := (600 * 2) / 3
  let total_revenue := revenue_sold_600_bars + revenue_remaining_600_bars
  total_revenue - total_cost

theorem school_team_profit_is_333 :
  candy_profit 1200 3 4 2 2 = 333 := by
  sorry

end school_team_profit_is_333_l505_50507


namespace evaluate_expression_l505_50505

theorem evaluate_expression : (8^5 / 8^2) * 2^12 = 2^21 := by
  sorry

end evaluate_expression_l505_50505


namespace friends_in_group_l505_50526

theorem friends_in_group (n : ℕ) 
  (avg_before_increase : ℝ := 800) 
  (avg_after_increase : ℝ := 850) 
  (individual_rent_increase : ℝ := 800 * 0.25) 
  (original_rent : ℝ := 800) 
  (new_rent : ℝ := 1000)
  (original_total : ℝ := avg_before_increase * n) 
  (new_total : ℝ := original_total + individual_rent_increase):
  new_total = avg_after_increase * n → 
  n = 4 :=
by
  sorry

end friends_in_group_l505_50526


namespace angle_A_measure_in_triangle_l505_50591

theorem angle_A_measure_in_triangle (A B C : ℝ) 
  (h1 : B = 15)
  (h2 : C = 3 * B) 
  (angle_sum : A + B + C = 180) :
  A = 120 :=
by
  -- We'll fill in the proof steps later
  sorry

end angle_A_measure_in_triangle_l505_50591


namespace find_star_l505_50579

-- Define the problem conditions and statement
theorem find_star (x : ℤ) (star : ℤ) (h1 : x = 5) (h2 : -3 * (star - 9) = 5 * x - 1) : star = 1 :=
by
  sorry -- Proof to be filled in

end find_star_l505_50579


namespace perimeter_of_irregular_pentagonal_picture_frame_l505_50547

theorem perimeter_of_irregular_pentagonal_picture_frame 
  (base : ℕ) (left_side : ℕ) (right_side : ℕ) (top_left_diagonal_side : ℕ) (top_right_diagonal_side : ℕ)
  (h_base : base = 10) (h_left_side : left_side = 12) (h_right_side : right_side = 11)
  (h_top_left_diagonal_side : top_left_diagonal_side = 6) (h_top_right_diagonal_side : top_right_diagonal_side = 7) :
  base + left_side + right_side + top_left_diagonal_side + top_right_diagonal_side = 46 :=
by {
  sorry
}

end perimeter_of_irregular_pentagonal_picture_frame_l505_50547


namespace minimize_sum_of_cubes_l505_50530

theorem minimize_sum_of_cubes (x y : ℝ) (h : x + y = 8) : 
  (3 * x^2 - 3 * (8 - x)^2 = 0) → (x = 4) ∧ (y = 4) :=
by
  sorry

end minimize_sum_of_cubes_l505_50530


namespace minimum_n_l505_50595

noncomputable def a (n : ℕ) : ℕ := 2 ^ (n - 2)

noncomputable def b (n : ℕ) : ℕ := n - 6 + a n

noncomputable def S (n : ℕ) : ℕ := (n * (n - 11)) / 2 + (2 ^ n - 1) / 2

theorem minimum_n (n : ℕ) (hn : n ≥ 5) : S 5 > 0 := by
  sorry

end minimum_n_l505_50595


namespace calculation_101_squared_minus_99_squared_l505_50508

theorem calculation_101_squared_minus_99_squared : 101^2 - 99^2 = 400 :=
by
  sorry

end calculation_101_squared_minus_99_squared_l505_50508


namespace min_value_expression_l505_50583

theorem min_value_expression :
  ∃ x > 0, x^2 + 6 * x + 100 / x^3 = 3 * (50:ℝ)^(2/5) + 6 * (50:ℝ)^(1/5) :=
by
  sorry

end min_value_expression_l505_50583


namespace proof_problem_l505_50517

variable (x y : ℝ)

noncomputable def condition1 : Prop := x > y
noncomputable def condition2 : Prop := x * y = 1

theorem proof_problem (hx : condition1 x y) (hy : condition2 x y) : 
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := 
by
  sorry

end proof_problem_l505_50517


namespace sum_of_digits_T_l505_50598

-- Conditions:
def horse_lap_times := [1, 2, 3, 4, 5, 6, 7, 8]
def S := 840
def total_horses := 8
def min_horses_at_start := 4

-- Question:
def T := 12 -- Least time such that at least 4 horses meet

/-- Prove that the sum of the digits of T is 3 -/
theorem sum_of_digits_T : (1 + 2) = 3 := by
  sorry

end sum_of_digits_T_l505_50598


namespace ellipse_major_axis_min_length_l505_50534

theorem ellipse_major_axis_min_length (a b c : ℝ) 
  (h1 : b * c = 2)
  (h2 : a^2 = b^2 + c^2) 
  : 2 * a ≥ 4 :=
sorry

end ellipse_major_axis_min_length_l505_50534


namespace emma_age_l505_50561

variables (O N L E : ℕ)

def oliver_eq : Prop := O = N - 5
def nancy_eq : Prop := N = L + 6
def emma_eq : Prop := E = L + 4
def oliver_age : Prop := O = 16

theorem emma_age :
  oliver_eq O N ∧ nancy_eq N L ∧ emma_eq E L ∧ oliver_age O → E = 19 :=
by
  sorry

end emma_age_l505_50561


namespace initial_customers_l505_50599

theorem initial_customers (x : ℕ) (h : x - 3 + 39 = 50) : x = 14 :=
by
  sorry

end initial_customers_l505_50599


namespace xiao_wang_scores_problem_l505_50590

-- Defining the problem conditions and solution as a proof problem
theorem xiao_wang_scores_problem (x y : ℕ) (h1 : (x * y + 98) / (x + 1) = y + 1) 
                                 (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) :
  (x + 2 = 10) ∧ (y - 1 = 88) :=
by 
  sorry

end xiao_wang_scores_problem_l505_50590


namespace volume_of_second_cube_is_twosqrt2_l505_50525

noncomputable def side_length (volume : ℝ) : ℝ :=
  volume^(1/3)

noncomputable def surface_area (side : ℝ) : ℝ :=
  6 * side^2

theorem volume_of_second_cube_is_twosqrt2
  (v1 : ℝ)
  (h1 : v1 = 1)
  (A1 := surface_area (side_length v1))
  (A2 := 2 * A1)
  (s2 := (A2 / 6)^(1/2)) :
  (s2^3 = 2 * Real.sqrt 2) :=
by
  sorry

end volume_of_second_cube_is_twosqrt2_l505_50525


namespace calculate_first_worker_time_l505_50533

theorem calculate_first_worker_time
    (T : ℝ)
    (h : 1/T + 1/4 = 1/2.2222222222222223) :
    T = 5 := sorry

end calculate_first_worker_time_l505_50533


namespace max_bishops_correct_bishop_position_count_correct_l505_50532

-- Define the parameters and predicates
def chessboard_size : ℕ := 2015

def max_bishops (board_size : ℕ) : ℕ := 2 * board_size - 1 - 1

def bishop_position_count (board_size : ℕ) : ℕ := 2 ^ (board_size - 1) * 2 * 2

-- State the equalities to be proved
theorem max_bishops_correct : max_bishops chessboard_size = 4028 := by
  -- proof will be here
  sorry

theorem bishop_position_count_correct : bishop_position_count chessboard_size = 2 ^ 2016 := by
  -- proof will be here
  sorry

end max_bishops_correct_bishop_position_count_correct_l505_50532


namespace probability_of_observing_color_change_l505_50502

def cycle_duration := 100
def observation_interval := 4
def change_times := [45, 50, 100]

def probability_of_change : ℚ :=
  (observation_interval * change_times.length : ℚ) / cycle_duration

theorem probability_of_observing_color_change :
  probability_of_change = 0.12 := by
  -- Proof goes here
  sorry

end probability_of_observing_color_change_l505_50502


namespace toadon_population_percentage_l505_50512

theorem toadon_population_percentage {pop_total G L T : ℕ}
    (h_total : pop_total = 80000)
    (h_gordonia : G = pop_total / 2)
    (h_lakebright : L = 16000)
    (h_total_population : pop_total = G + T + L) :
    (T * 100 / G) = 60 :=
by sorry

end toadon_population_percentage_l505_50512


namespace carol_invitations_l505_50585

-- Definitions: each package has 3 invitations, Carol bought 2 packs, and Carol needs 3 extra invitations.
def invitations_per_pack : ℕ := 3
def packs_bought : ℕ := 2
def extra_invitations : ℕ := 3

-- Total number of invitations Carol will have
def total_invitations : ℕ := (packs_bought * invitations_per_pack) + extra_invitations

-- Statement to prove: Carol wants to invite 9 friends.
theorem carol_invitations : total_invitations = 9 := by
  sorry  -- Proof omitted

end carol_invitations_l505_50585


namespace middle_person_distance_l505_50569

noncomputable def Al_position (t : ℝ) : ℝ := 6 * t
noncomputable def Bob_position (t : ℝ) : ℝ := 10 * t - 12
noncomputable def Cy_position (t : ℝ) : ℝ := 8 * t - 32

theorem middle_person_distance (t : ℝ) (h₁ : t ≥ 0) (h₂ : t ≥ 2) (h₃ : t ≥ 4) :
  (Al_position t = 52) ∨ (Bob_position t = 52) ∨ (Cy_position t = 52) :=
sorry

end middle_person_distance_l505_50569


namespace hours_worked_each_day_l505_50537

-- Given conditions
def total_hours_worked : ℕ := 18
def number_of_days_worked : ℕ := 6

-- Statement to prove
theorem hours_worked_each_day : total_hours_worked / number_of_days_worked = 3 := by
  sorry

end hours_worked_each_day_l505_50537


namespace additional_time_due_to_leak_is_six_l505_50519

open Real

noncomputable def filling_time_with_leak (R L : ℝ) : ℝ := 1 / (R - L)
noncomputable def filling_time_without_leak (R : ℝ) : ℝ := 1 / R
noncomputable def additional_filling_time (R L : ℝ) : ℝ :=
  filling_time_with_leak R L - filling_time_without_leak R

theorem additional_time_due_to_leak_is_six :
  additional_filling_time 0.25 (3 / 20) = 6 := by
  sorry

end additional_time_due_to_leak_is_six_l505_50519


namespace evaluate_expression_l505_50593

variables (a b c : ℝ)

theorem evaluate_expression (h1 : c = b - 20) (h2 : b = a + 4) (h3 : a = 2)
  (h4 : a^2 + a ≠ 0) (h5 : b^2 - 6 * b + 8 ≠ 0) (h6 : c^2 + 12 * c + 36 ≠ 0):
  (a^2 + 2 * a) / (a^2 + a) * (b^2 - 4) / (b^2 - 6 * b + 8) * (c^2 + 16 * c + 64) / (c^2 + 12 * c + 36) = 3 / 4 :=
by sorry

end evaluate_expression_l505_50593


namespace black_cars_count_l505_50510

-- Conditions
def red_cars : ℕ := 28
def ratio_red_black : ℚ := 3 / 8

-- Theorem statement
theorem black_cars_count :
  ∃ (black_cars : ℕ), black_cars = 75 ∧ (red_cars : ℚ) / (black_cars) = ratio_red_black :=
sorry

end black_cars_count_l505_50510


namespace solution_valid_l505_50563

noncomputable def verify_solution (x : ℝ) : Prop :=
  (Real.arcsin (3 * x) + Real.arccos (2 * x) = Real.pi / 4) ∧
  (|2 * x| ≤ 1) ∧
  (|3 * x| ≤ 1)

theorem solution_valid (x : ℝ) :
  verify_solution x ↔ (x = 1 / Real.sqrt (11 - 2 * Real.sqrt 2) ∨ x = -(1 / Real.sqrt (11 - 2 * Real.sqrt 2))) :=
by {
  sorry
}

end solution_valid_l505_50563


namespace probability_of_draw_l505_50580

-- Define probabilities
def P_A_wins : ℝ := 0.4
def P_A_not_loses : ℝ := 0.9

-- Theorem statement
theorem probability_of_draw : P_A_not_loses = P_A_wins + 0.5 :=
by
  -- Proof is skipped
  sorry

end probability_of_draw_l505_50580


namespace other_toys_cost_1000_l505_50568

-- Definitions of the conditions
def cost_of_other_toys : ℕ := sorry
def cost_of_lightsaber (cost_of_other_toys : ℕ) : ℕ := 2 * cost_of_other_toys
def total_spent (cost_of_lightsaber cost_of_other_toys : ℕ) : ℕ := cost_of_lightsaber + cost_of_other_toys

-- The proof goal
theorem other_toys_cost_1000 (T : ℕ) (H1 : cost_of_lightsaber T = 2 * T) 
                            (H2 : total_spent (cost_of_lightsaber T) T = 3000) : T = 1000 := by
  sorry

end other_toys_cost_1000_l505_50568


namespace sparrows_initial_count_l505_50570

theorem sparrows_initial_count (a b c : ℕ) 
  (h1 : a + b + c = 24)
  (h2 : a - 4 = b + 1)
  (h3 : b + 1 = c + 3) : 
  a = 12 ∧ b = 7 ∧ c = 5 :=
by
  sorry

end sparrows_initial_count_l505_50570


namespace sacks_per_day_l505_50538

theorem sacks_per_day (total_sacks : ℕ) (total_days : ℕ) (harvest_per_day : ℕ) : 
  total_sacks = 56 → 
  total_days = 14 → 
  harvest_per_day = total_sacks / total_days → 
  harvest_per_day = 4 := 
by
  intros h_total_sacks h_total_days h_harvest_per_day
  rw [h_total_sacks, h_total_days] at h_harvest_per_day
  simp at h_harvest_per_day
  exact h_harvest_per_day

end sacks_per_day_l505_50538


namespace pen_ratio_l505_50541

theorem pen_ratio (R J D : ℕ) (pen_cost : ℚ) (total_spent : ℚ) (total_pens : ℕ) 
  (hR : R = 4)
  (hJ : J = 3 * R)
  (h_total_spent : total_spent = 33)
  (h_pen_cost : pen_cost = 1.5)
  (h_total_pens : total_pens = total_spent / pen_cost)
  (h_pens_expr : D + J + R = total_pens) :
  D / J = 1 / 2 :=
by
  sorry

end pen_ratio_l505_50541


namespace Q_current_age_l505_50528

-- Definitions for the current ages of P and Q
variable (P Q : ℕ)

-- Conditions
-- 1. P + Q = 100
-- 2. P = 3 * (Q - (P - Q))  (from P is thrice as old as Q was when P was as old as Q is now)

axiom age_sum : P + Q = 100
axiom age_relation : P = 3 * (Q - (P - Q))

theorem Q_current_age : Q = 40 :=
by
  sorry

end Q_current_age_l505_50528


namespace largest_among_a_b_c_l505_50516

theorem largest_among_a_b_c (x : ℝ) (h0 : 0 < x) (h1 : x < 1)
  (a : ℝ := 2 * Real.sqrt x) 
  (b : ℝ := 1 + x) 
  (c : ℝ := 1 / (1 - x)) : c > b ∧ b > a := by
  sorry

end largest_among_a_b_c_l505_50516


namespace additional_money_required_l505_50576

   theorem additional_money_required (patricia_money lisa_money charlotte_money total_card_cost : ℝ) 
       (h1 : patricia_money = 6)
       (h2 : lisa_money = 5 * patricia_money)
       (h3 : lisa_money = 2 * charlotte_money)
       (h4 : total_card_cost = 100) :
     (total_card_cost - (patricia_money + lisa_money + charlotte_money) = 49) := 
   by
     sorry
   
end additional_money_required_l505_50576


namespace integer_values_of_x_in_triangle_l505_50557

theorem integer_values_of_x_in_triangle (x : ℝ) :
  (x + 14 > 38 ∧ x + 38 > 14 ∧ 14 + 38 > x) → 
  ∃ (n : ℕ), n = 27 ∧ ∀ m : ℕ, (24 < m ∧ m < 52 ↔ (m : ℝ) > 24 ∧ (m : ℝ) < 52) :=
by {
  sorry
}

end integer_values_of_x_in_triangle_l505_50557


namespace graphs_intersection_l505_50584

theorem graphs_intersection 
  (a b c d x y : ℝ) 
  (h_a : a ≠ 0) (h_b : b ≠ 0) (h_c : c ≠ 0) (h_d : d ≠ 0) 
  (h1: y = ax^2 + bx + c) 
  (h2: y = ax^2 - bx + c + d) 
  : x = d / (2 * b) ∧ y = (a * d^2) / (4 * b^2) + d / 2 + c := 
sorry

end graphs_intersection_l505_50584


namespace solve_for_x_l505_50572

theorem solve_for_x (x : ℚ) (h : (x + 4) / (x - 3) = (x - 2) / (x + 2)) : x = -2 / 11 :=
by
  sorry

end solve_for_x_l505_50572


namespace monthly_compounding_greater_than_yearly_l505_50578

open Nat Real

theorem monthly_compounding_greater_than_yearly : 
  1 + 3 / 100 < (1 + 3 / (12 * 100)) ^ 12 :=
by
  -- This is the proof we need to write.
  sorry

end monthly_compounding_greater_than_yearly_l505_50578


namespace find_d_l505_50535

theorem find_d : ∃ d : ℝ, (∀ x : ℝ, 2 * x^2 + 9 * x + d = 0 ↔ x = (-9 + Real.sqrt 17) / 4 ∨ x = (-9 - Real.sqrt 17) / 4) ∧ d = 8 :=
by
  sorry

end find_d_l505_50535


namespace find_C_l505_50552

theorem find_C (A B C : ℕ) (h1 : (19 + A + B) % 3 = 0) (h2 : (15 + A + B + C) % 3 = 0) : C = 1 := by
  sorry

end find_C_l505_50552


namespace shells_picked_in_morning_l505_50536

-- Definitions based on conditions
def total_shells : ℕ := 616
def afternoon_shells : ℕ := 324

-- The goal is to prove that morning_shells = 292
theorem shells_picked_in_morning (morning_shells : ℕ) (h : total_shells = morning_shells + afternoon_shells) : morning_shells = 292 := 
by
  sorry

end shells_picked_in_morning_l505_50536


namespace simple_interest_rate_l505_50581

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (SI : ℝ) (h1 : T = 15) (h2 : SI = 3 * P) (h3 : SI = P * R * T / 100) : R = 20 :=
by 
  sorry

end simple_interest_rate_l505_50581


namespace sum_first_odd_numbers_not_prime_l505_50586

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem sum_first_odd_numbers_not_prime :
  ¬ (is_prime (1 + 3)) ∧
  ¬ (is_prime (1 + 3 + 5)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7)) ∧
  ¬ (is_prime (1 + 3 + 5 + 7 + 9)) :=
by
  sorry

end sum_first_odd_numbers_not_prime_l505_50586


namespace part1_part2_l505_50501

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4 * x + a + 3
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := b * x + 5 - 2 * b

theorem part1 (a : ℝ) : (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f x a = 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
sorry

theorem part2 (b : ℝ) : (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 4 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 4 ∧ f x2 3 = g x1 b) ↔ -1 ≤ b ∧ b ≤ 1/2 :=
sorry

end part1_part2_l505_50501


namespace Martha_points_l505_50596

def beef_cost := 3 * 11
def fv_cost := 8 * 4
def spice_cost := 3 * 6
def other_cost := 37

def total_spent := beef_cost + fv_cost + spice_cost + other_cost
def points_per_10 := 50
def bonus := 250

def increments := total_spent / 10
def points := increments * points_per_10
def total_points := points + bonus

theorem Martha_points : total_points = 850 :=
by
  sorry

end Martha_points_l505_50596


namespace village_population_l505_50545

theorem village_population (P : ℝ) (h : 0.8 * P = 64000) : P = 80000 := by
  sorry

end village_population_l505_50545


namespace kendra_sunday_shirts_l505_50521

def total_shirts := 22
def shirts_weekdays := 5 * 1
def shirts_after_school := 3
def shirts_saturday := 1

theorem kendra_sunday_shirts : 
  (total_shirts - 2 * (shirts_weekdays + shirts_after_school + shirts_saturday)) = 4 :=
by
  sorry

end kendra_sunday_shirts_l505_50521


namespace vitamin_d_supplements_per_pack_l505_50573

theorem vitamin_d_supplements_per_pack :
  ∃ (x : ℕ), (∀ (n m : ℕ), 7 * n = x * m → 119 <= 7 * n) ∧ (7 * n = 17 * m) :=
by
  -- definition of conditions
  let min_sold := 119
  let vitaminA_per_pack := 7
  -- let x be the number of Vitamin D supplements per pack
  -- the proof is yet to be completed
  sorry

end vitamin_d_supplements_per_pack_l505_50573


namespace system_of_equations_solution_l505_50527

theorem system_of_equations_solution (x y : ℝ) 
  (h1 : x + 3 * y = 7) 
  (h2 : y = 2 * x) : 
  x = 1 ∧ y = 2 :=
by
  sorry

end system_of_equations_solution_l505_50527


namespace tallest_player_height_correct_l505_50503

-- Define the height of the shortest player
def shortest_player_height : ℝ := 68.25

-- Define the height difference between the tallest and shortest player
def height_difference : ℝ := 9.5

-- Define the height of the tallest player based on the conditions
def tallest_player_height : ℝ :=
  shortest_player_height + height_difference

-- Theorem statement
theorem tallest_player_height_correct : tallest_player_height = 77.75 := by
  sorry

end tallest_player_height_correct_l505_50503


namespace A_union_B_l505_50564

noncomputable def A : Set ℝ := {x | ∃ y, y = 1 / Real.sqrt (1 - 2^x) ∧ x < 0}
noncomputable def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1) / Real.log 2 ∧ x > 0}
noncomputable def union_set : Set ℝ := {x | x < 0 ∨ x > 0}

theorem A_union_B :
  A ∪ B = union_set :=
by
  sorry

end A_union_B_l505_50564


namespace angle_measures_possible_l505_50587

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end angle_measures_possible_l505_50587


namespace carla_correct_questions_l505_50523

theorem carla_correct_questions :
  ∀ (Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct : ℕ), 
    Drew_correct = 20 →
    Drew_wrong = 6 →
    Carla_wrong = 2 * Drew_wrong →
    Total_questions = 52 →
    Carla_correct = Total_questions - Carla_wrong →
    Carla_correct = 40 :=
by
  intros Drew_correct Drew_wrong Carla_wrong Total_questions Carla_correct
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end carla_correct_questions_l505_50523


namespace square_perimeter_is_64_l505_50558

-- Given conditions
variables (s : ℕ)
def is_square_divided_into_four_congruent_rectangles : Prop :=
  ∀ (r : ℕ), r = 4 → (∀ (p : ℕ), p = (5 * s) / 2 → p = 40)

-- Lean 4 statement for the proof problem
theorem square_perimeter_is_64 
  (h : is_square_divided_into_four_congruent_rectangles s) 
  (hs : (5 * s) / 2 = 40) : 
  4 * s = 64 :=
by
  sorry

end square_perimeter_is_64_l505_50558


namespace MaximMethod_CorrectNumber_l505_50518

theorem MaximMethod_CorrectNumber (x y : ℕ) (N : ℕ) (h_digit_x : 0 ≤ x ∧ x ≤ 9) (h_digit_y : 1 ≤ y ∧ y ≤ 9)
  (h_N : N = 10 * x + y)
  (h_condition : 1 / (10 * x + y : ℚ) = 1 / (x + y : ℚ) - 1 / (x * y : ℚ)) :
  N = 24 :=
sorry

end MaximMethod_CorrectNumber_l505_50518


namespace line_through_point_with_equal_intercepts_l505_50592

theorem line_through_point_with_equal_intercepts :
  ∃ (m b : ℝ), ∀ (x y : ℝ), 
    ((y = m * x + b ∧ ((x = 0 ∨ y = 0) → (x = y))) ∧ 
    (1 = m * 1 + b ∧ 1 + 1 = b)) → 
    (m = 1 ∧ b = 0) ∨ (m = -1 ∧ b = 2) :=
by
  sorry

end line_through_point_with_equal_intercepts_l505_50592


namespace attendees_not_from_companies_l505_50542

theorem attendees_not_from_companies :
  let A := 30 
  let B := 2 * A
  let C := A + 10
  let D := C - 5
  let T := 185 
  T - (A + B + C + D) = 20 :=
by
  sorry

end attendees_not_from_companies_l505_50542


namespace maximize_expression_l505_50577

noncomputable def max_value_expression (x y z : ℝ) : ℝ :=
(x^2 + x * y + y^2) * (x^2 + x * z + z^2) * (y^2 + y * z + z^2)

theorem maximize_expression (x y z : ℝ) (h₀ : 0 ≤ x) (h₁ : 0 ≤ y) (h₂ : 0 ≤ z) (h₃ : x + y + z = 3) : 
    max_value_expression x y z ≤ 27 :=
sorry

end maximize_expression_l505_50577


namespace area_of_rectangle_l505_50567

-- Define the problem conditions in Lean
def circle_radius := 7
def circle_diameter := 2 * circle_radius
def width_of_rectangle := circle_diameter
def length_to_width_ratio := 3
def length_of_rectangle := length_to_width_ratio * width_of_rectangle

-- Define the statement to be proved (area of the rectangle)
theorem area_of_rectangle : 
  (length_of_rectangle * width_of_rectangle) = 588 := by
  sorry

end area_of_rectangle_l505_50567


namespace min_xy_min_x_plus_y_l505_50589

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x * y ≥ 9 :=
sorry

theorem min_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y - x - y = 3) : x + y ≥ 6 :=
sorry

end min_xy_min_x_plus_y_l505_50589


namespace playdough_cost_l505_50515

-- Definitions of the costs and quantities
def lego_cost := 250
def sword_cost := 120
def playdough_quantity := 10
def total_paid := 1940

-- Variables representing the quantities bought
def lego_quantity := 3
def sword_quantity := 7

-- Function to calculate the total cost for lego and sword
def total_lego_cost := lego_quantity * lego_cost
def total_sword_cost := sword_quantity * sword_cost

-- Variable representing the cost of playdough
variable (P : ℝ)

-- The main statement to prove
theorem playdough_cost :
  total_lego_cost + total_sword_cost + playdough_quantity * P = total_paid → P = 35 :=
by
  sorry

end playdough_cost_l505_50515


namespace leif_apples_l505_50500

-- Definitions based on conditions
def oranges : ℕ := 24
def apples (oranges apples_diff : ℕ) := oranges - apples_diff

-- Theorem stating the problem to prove
theorem leif_apples (oranges apples_diff : ℕ) (h1 : oranges = 24) (h2 : apples_diff = 10) : apples oranges apples_diff = 14 :=
by
  -- Using the definition of apples and given conditions, prove the number of apples
  rw [h1, h2]
  -- Calculating the number of apples
  show 24 - 10 = 14
  rfl

end leif_apples_l505_50500


namespace fixed_point_for_all_parabolas_l505_50565

theorem fixed_point_for_all_parabolas : ∃ (x y : ℝ), (∀ t : ℝ, y = 4 * x^2 + 2 * t * x - 3 * t) ∧ x = 1 ∧ y = 4 :=
by 
  sorry

end fixed_point_for_all_parabolas_l505_50565


namespace vertical_asymptote_singleton_l505_50513

theorem vertical_asymptote_singleton (c : ℝ) :
  (∃ x, (x^2 - 2 * x + c) = 0 ∧ ((x - 1) * (x + 3) = 0) ∧ (x ≠ 1 ∨ x ≠ -3)) 
  ↔ (c = 1 ∨ c = -15) :=
by
  sorry

end vertical_asymptote_singleton_l505_50513


namespace complex_exp_identity_l505_50540

theorem complex_exp_identity (i : ℂ) (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end complex_exp_identity_l505_50540


namespace log_product_solution_l505_50546

theorem log_product_solution (x : ℝ) (hx : 0 < x) : 
  (Real.log x / Real.log 2) * (Real.log x / Real.log 5) = Real.log 10 / Real.log 2 ↔ 
  x = 2 ^ Real.sqrt (6 * Real.log 2) :=
sorry

end log_product_solution_l505_50546


namespace geometric_progression_solution_l505_50550

noncomputable def first_term_of_geometric_progression (b2 b6 : ℚ) (q : ℚ) : ℚ := 
  b2 / q
  
theorem geometric_progression_solution 
  (b2 b6 : ℚ)
  (h1 : b2 = 37 + 1/3)
  (h2 : b6 = 2 + 1/3) :
  ∃ a q : ℚ, a = 224 / 3 ∧ q = 1/2 ∧ b2 = a * q ∧ b6 = a * q^5 :=
by
  sorry

end geometric_progression_solution_l505_50550


namespace shaded_area_of_three_circles_l505_50555

theorem shaded_area_of_three_circles :
  (∀ (r1 r2 : ℝ), (π * r1^2 = 100 * π) → (r2 = r1 / 2) → (shaded_area = (π * r1^2) / 2 + 2 * ((π * r2^2) / 2)) → (shaded_area = 75 * π)) :=
by
  sorry

end shaded_area_of_three_circles_l505_50555


namespace anthony_pencils_l505_50511

def initial_pencils : ℝ := 56.0  -- Condition 1
def pencils_left : ℝ := 47.0     -- Condition 2
def pencils_given : ℝ := 9.0     -- Correct Answer

theorem anthony_pencils :
  initial_pencils - pencils_left = pencils_given :=
by
  sorry

end anthony_pencils_l505_50511


namespace range_of_m_l505_50571

open Set

noncomputable def setA : Set ℝ := {y | ∃ x : ℝ, y = 2^x / (2^x + 1)}
noncomputable def setB (m : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ y = (1 / 3) * x + m}

theorem range_of_m {m : ℝ} (p q : Prop) :
  p ↔ ∃ x : ℝ, x ∈ setA →
  q ↔ ∃ x : ℝ, x ∈ setB m →
  ((p → q) ∧ ¬(q → p)) ↔ (1 / 3 < m ∧ m < 2 / 3) :=
by
  sorry

end range_of_m_l505_50571


namespace quadratic_inequality_solution_empty_l505_50554

theorem quadratic_inequality_solution_empty (m : ℝ) :
  (∀ x : ℝ, ((m + 1) * x^2 - m * x + m - 1 < 0) → false) →
  (m ≥ (2 * Real.sqrt 3) / 3 ∨ m ≤ -(2 * Real.sqrt 3) / 3) :=
by
  sorry

end quadratic_inequality_solution_empty_l505_50554


namespace find_quadruples_l505_50522

open Nat

theorem find_quadruples (a b p n : ℕ) (hp : Nat.Prime p) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n)
    (h : a^3 + b^3 = p^n) :
    (∃ k, a = 2^k ∧ b = 2^k ∧ p = 2 ∧ n = 3 * k + 1) ∨
    (∃ k, a = 3^k ∧ b = 2 * 3^k ∧ p = 3 ∧ n = 3 * k + 2) ∨
    (∃ k, a = 2 * 3^k ∧ b = 3^k ∧ p = 3 ∧ n = 3 * k + 2) :=
sorry

end find_quadruples_l505_50522


namespace sin_neg_4_div_3_pi_l505_50594

theorem sin_neg_4_div_3_pi : Real.sin (- (4 / 3) * Real.pi) = Real.sqrt 3 / 2 :=
by sorry

end sin_neg_4_div_3_pi_l505_50594


namespace curve_touch_all_Ca_l505_50504

theorem curve_touch_all_Ca (a : ℝ) (a_pos : a > 0) (x y : ℝ) :
  ( (y - a^2)^2 = x^2 * (a^2 - x^2) ) → (y = (3 / 4) * x^2) :=
by
  sorry

end curve_touch_all_Ca_l505_50504


namespace average_price_of_towels_l505_50520

theorem average_price_of_towels :
  let total_cost := 2350
  let total_towels := 10
  total_cost / total_towels = 235 :=
by
  sorry

end average_price_of_towels_l505_50520


namespace find_q_l505_50531

noncomputable def solution_condition (p q : ℝ) : Prop :=
  (p > 1) ∧ (q > 1) ∧ (1 / p + 1 / q = 1) ∧ (p * q = 9)

theorem find_q (p q : ℝ) (h : solution_condition p q) : 
  q = (9 + 3 * Real.sqrt 5) / 2 :=
sorry

end find_q_l505_50531


namespace fraction_B_A_plus_C_l505_50556

variable (A B C : ℝ)
variable (f : ℝ)
variable (hA : A = 1 / 3 * (B + C))
variable (hB : A = B + 30)
variable (hTotal : A + B + C = 1080)
variable (hf : B = f * (A + C))

theorem fraction_B_A_plus_C :
  f = 2 / 7 :=
sorry

end fraction_B_A_plus_C_l505_50556


namespace payment_for_150_books_equal_payment_number_of_books_l505_50575

/-- 
Xinhua Bookstore conditions:
- Both suppliers A and B price each book at 40 yuan. 
- Supplier A offers a 10% discount on all books.
- Supplier B offers a 20% discount on any books purchased exceeding 100 books.
-/

def price_per_book_supplier_A (n : ℕ) : ℝ := 40 * 0.9
def price_per_first_100_books_supplier_B : ℝ := 40
def price_per_excess_books_supplier_B (n : ℕ) : ℝ := 40 * 0.8

-- Prove that the payment amounts for 150 books from suppliers A and B are 5400 yuan and 5600 yuan respectively.
theorem payment_for_150_books :
  price_per_book_supplier_A 150 * 150 = 5400 ∧
  price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B 50 * (150 - 100) = 5600 :=
  sorry

-- Prove the equal payment equivalence theorem for supplier A and B.
theorem equal_payment_number_of_books (x : ℕ) :
  price_per_book_supplier_A x * x = price_per_first_100_books_supplier_B * 100 + price_per_excess_books_supplier_B (x - 100) * (x - 100) → x = 200 :=
  sorry

end payment_for_150_books_equal_payment_number_of_books_l505_50575


namespace karlson_wins_with_optimal_play_l505_50509

def game_win_optimal_play: Prop :=
  ∀ (total_moves: ℕ), 
  (total_moves % 2 = 1) 

theorem karlson_wins_with_optimal_play: game_win_optimal_play :=
by sorry

end karlson_wins_with_optimal_play_l505_50509


namespace correct_operation_l505_50548

theorem correct_operation : ∃ (a : ℝ), (3 + Real.sqrt 2 ≠ 3 * Real.sqrt 2) ∧ 
  ((a ^ 2) ^ 3 ≠ a ^ 5) ∧
  (Real.sqrt ((-7 : ℝ) ^ 2) ≠ -7) ∧
  (4 * a ^ 2 * a = 4 * a ^ 3) :=
by
  sorry

end correct_operation_l505_50548


namespace first_year_with_digit_sum_seven_l505_50544

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem first_year_with_digit_sum_seven : ∃ y, y > 2023 ∧ sum_of_digits y = 7 ∧ ∀ z, z > 2023 ∧ z < y → sum_of_digits z ≠ 7 :=
by
  use 2032
  sorry

end first_year_with_digit_sum_seven_l505_50544


namespace find_all_functions_l505_50529

theorem find_all_functions (f : ℕ → ℕ) : 
  (∀ a b : ℕ, 0 < a → 0 < b → f (a^2 + b^2) = f a * f b) →
  (∀ a : ℕ, 0 < a → f (a^2) = f a ^ 2) →
  (∀ n : ℕ, 0 < n → f n = 1) :=
by
  intros h1 h2 a ha
  sorry

end find_all_functions_l505_50529


namespace sum_consecutive_integers_product_1080_l505_50549

theorem sum_consecutive_integers_product_1080 :
  ∃ n : ℕ, n * (n + 1) = 1080 ∧ n + (n + 1) = 65 :=
by
  sorry

end sum_consecutive_integers_product_1080_l505_50549


namespace fraction_of_gasoline_used_l505_50543

-- Define the conditions
def gasoline_per_mile := 1 / 30  -- Gallons per mile
def full_tank := 12  -- Gallons
def speed := 60  -- Miles per hour
def travel_time := 5  -- Hours

-- Total distance traveled
def distance := speed * travel_time  -- Miles

-- Gasoline used
def gasoline_used := distance * gasoline_per_mile  -- Gallons

-- Fraction of the full tank used
def fraction_used := gasoline_used / full_tank

-- The theorem to be proved
theorem fraction_of_gasoline_used :
  fraction_used = 5 / 6 :=
by sorry

end fraction_of_gasoline_used_l505_50543


namespace arun_weight_upper_limit_l505_50539

theorem arun_weight_upper_limit (weight : ℝ) (avg_weight : ℝ) 
  (arun_opinion : 66 < weight ∧ weight < 72) 
  (brother_opinion : 60 < weight ∧ weight < 70) 
  (average_condition : avg_weight = 68) : weight ≤ 70 :=
by
  sorry

end arun_weight_upper_limit_l505_50539


namespace factorial_mod_10_l505_50566

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Define the problem statement
theorem factorial_mod_10 : factorial 10 % 13 = 7 :=
by sorry

end factorial_mod_10_l505_50566


namespace cryptarithm_no_solution_proof_l505_50559

def cryptarithm_no_solution : Prop :=
  ∀ (D O N K A L E V G R : ℕ),
    D ≠ O ∧ D ≠ N ∧ D ≠ K ∧ D ≠ A ∧ D ≠ L ∧ D ≠ E ∧ D ≠ V ∧ D ≠ G ∧ D ≠ R ∧
    O ≠ N ∧ O ≠ K ∧ O ≠ A ∧ O ≠ L ∧ O ≠ E ∧ O ≠ V ∧ O ≠ G ∧ O ≠ R ∧
    N ≠ K ∧ N ≠ A ∧ N ≠ L ∧ N ≠ E ∧ N ≠ V ∧ N ≠ G ∧ N ≠ R ∧
    K ≠ A ∧ K ≠ L ∧ K ≠ E ∧ K ≠ V ∧ K ≠ G ∧ K ≠ R ∧
    A ≠ L ∧ A ≠ E ∧ A ≠ V ∧ A ≠ G ∧ A ≠ R ∧
    L ≠ E ∧ L ≠ V ∧ L ≠ G ∧ L ≠ R ∧
    E ≠ V ∧ E ≠ G ∧ E ≠ R ∧
    V ≠ G ∧ V ≠ R ∧
    G ≠ R ∧
    (D * 100 + O * 10 + N) + (O * 100 + K * 10 + A) +
    (L * 1000 + E * 100 + N * 10 + A) + (V * 10000 + O * 1000 + L * 100 + G * 10 + A) =
    A * 100000 + N * 10000 + G * 1000 + A * 100 + R * 10 + A →
    false

theorem cryptarithm_no_solution_proof : cryptarithm_no_solution :=
by sorry

end cryptarithm_no_solution_proof_l505_50559


namespace brenda_num_cookies_per_box_l505_50582

def numCookiesPerBox (trays : ℕ) (cookiesPerTray : ℕ) (costPerBox : ℚ) (totalSpent : ℚ) : ℚ :=
  let totalCookies := trays * cookiesPerTray
  let numBoxes := totalSpent / costPerBox
  totalCookies / numBoxes

theorem brenda_num_cookies_per_box :
  numCookiesPerBox 3 80 3.5 14 = 60 := by
  sorry

end brenda_num_cookies_per_box_l505_50582


namespace remainder_when_divided_l505_50551

noncomputable def y : ℝ := 19.999999999999716
def quotient : ℝ := 76.4
def remainder : ℝ := 8

theorem remainder_when_divided (x : ℝ) (hx : x = y * 76 + y * 0.4) : x % y = 8 :=
by
  -- Proof is omitted
  sorry

end remainder_when_divided_l505_50551


namespace scientific_notation_of_virus_diameter_l505_50514

theorem scientific_notation_of_virus_diameter :
  0.000000102 = 1.02 * 10 ^ (-7) :=
  sorry

end scientific_notation_of_virus_diameter_l505_50514
