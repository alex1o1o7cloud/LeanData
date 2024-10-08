import Mathlib

namespace commission_percentage_l165_165570

theorem commission_percentage 
  (cost_price : ℝ) (profit_percentage : ℝ) (observed_price : ℝ) (C : ℝ) 
  (h1 : cost_price = 15)
  (h2 : profit_percentage = 0.10)
  (h3 : observed_price = 19.8) 
  (h4 : 1 + C / 100 = 19.8 / (cost_price * (1 + profit_percentage)))
  : C = 20 := 
by
  sorry

end commission_percentage_l165_165570


namespace initial_percentage_rise_l165_165201

-- Definition of the conditions
def final_price_gain (P : ℝ) (x : ℝ) : Prop :=
  P * (1 + x / 100) * 0.9 * 0.85 = P * 1.03275

-- The statement to be proven
theorem initial_percentage_rise (P : ℝ) (x : ℝ) : final_price_gain P x → x = 35.03 :=
by
  sorry -- Proof to be filled in

end initial_percentage_rise_l165_165201


namespace curtain_price_l165_165684

theorem curtain_price
  (C : ℝ)
  (h1 : 2 * C + 9 * 15 + 50 = 245) :
  C = 30 :=
sorry

end curtain_price_l165_165684


namespace exists_two_linear_functions_l165_165240

-- Define the quadratic trinomials and their general forms
variables (a b c d e f : ℝ)
-- Assuming coefficients a and d are non-zero
variable (ha : a ≠ 0)
variable (hd : d ≠ 0)

-- Define the linear function
def ell (m n x : ℝ) : ℝ := m * x + n

-- Define the quadratic trinomials P(x) and Q(x) 
def P (x : ℝ) := a * x^2 + b * x + c
def Q (x : ℝ) := d * x^2 + e * x + f

-- Prove that there exist exactly two linear functions ell(x) that satisfy the condition for all x
theorem exists_two_linear_functions : 
  ∃ (m1 m2 n1 n2 : ℝ), 
  (∀ x, P a b c x = Q d e f (ell m1 n1 x)) ∧ 
  (∀ x, P a b c x = Q d e f (ell m2 n2 x)) := 
sorry

end exists_two_linear_functions_l165_165240


namespace playground_area_l165_165783

theorem playground_area (B : ℕ) (L : ℕ) (playground_area : ℕ) 
  (h1 : L = 8 * B) 
  (h2 : L = 240) 
  (h3 : playground_area = (1 / 6) * (L * B)) : 
  playground_area = 1200 :=
by
  sorry

end playground_area_l165_165783


namespace diana_total_earnings_l165_165628

-- Define the earnings in each month
def july_earnings : ℕ := 150
def august_earnings : ℕ := 3 * july_earnings
def september_earnings : ℕ := 2 * august_earnings

-- State the theorem that the total earnings over the three months is $1500
theorem diana_total_earnings : july_earnings + august_earnings + september_earnings = 1500 :=
by
  have h1 : august_earnings = 3 * july_earnings := rfl
  have h2 : september_earnings = 2 * august_earnings := rfl
  sorry

end diana_total_earnings_l165_165628


namespace kho_kho_only_l165_165670

theorem kho_kho_only (kabaddi_total : ℕ) (both_games : ℕ) (total_players : ℕ) (kabaddi_only : ℕ) (kho_kho_only : ℕ) 
  (h1 : kabaddi_total = 10)
  (h2 : both_games = 5)
  (h3 : total_players = 50)
  (h4 : kabaddi_only = 10 - both_games)
  (h5 : kabaddi_only + kho_kho_only + both_games = total_players) :
  kho_kho_only = 40 :=
by
  -- Proof is not required
  sorry

end kho_kho_only_l165_165670


namespace find_b_value_l165_165795

theorem find_b_value (b : ℝ) : (∃ (x y : ℝ), (x, y) = ((2 + 4) / 2, (5 + 9) / 2) ∧ x + y = b) ↔ b = 10 :=
by
  sorry

end find_b_value_l165_165795


namespace general_term_seq_l165_165898

open Nat

-- Definition of the sequence given conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 2 = 2 ∧ ∀ n, n ≥ 1 → (n - 1) * a (n + 1) - n * a n + 1 = 0

-- To prove that the general term is a_n = n
theorem general_term_seq (a : ℕ → ℕ) (h : seq a) : ∀ n, n ≥ 1 → a n = n := 
by
  sorry

end general_term_seq_l165_165898


namespace max_value_of_sums_l165_165847

noncomputable def max_of_sums (a b c d : ℝ) : ℝ :=
  a^4 + b^4 + c^4 + d^4

theorem max_value_of_sums (a b c d : ℝ) (h : a^3 + b^3 + c^3 + d^3 = 4) :
  max_of_sums a b c d ≤ 16 :=
sorry

end max_value_of_sums_l165_165847


namespace positive_difference_sums_even_odd_l165_165612

theorem positive_difference_sums_even_odd:
  let sum_first_n_even (n : ℕ) := 2 * (n * (n + 1) / 2)
  let sum_first_n_odd (n : ℕ) := n * n
  sum_first_n_even 25 - sum_first_n_odd 20 = 250 :=
by
  sorry

end positive_difference_sums_even_odd_l165_165612


namespace find_start_number_l165_165360

def count_even_not_divisible_by_3 (start end_ : ℕ) : ℕ :=
  (end_ / 2 + 1) - (end_ / 6 + 1) - (if start = 0 then start / 2 else start / 2 + 1 - (start - 1) / 6 - 1)

theorem find_start_number (start end_ : ℕ) (h1 : end_ = 170) (h2 : count_even_not_divisible_by_3 start end_ = 54) : start = 8 :=
by 
  rw [h1] at h2
  sorry

end find_start_number_l165_165360


namespace find_b_over_a_find_angle_B_l165_165244

-- Definitions and main theorems
noncomputable def sides_in_triangle (A B C a b c : ℝ) : Prop :=
  a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = Real.sqrt 2 * a

noncomputable def cos_law_condition (a b c : ℝ) : Prop :=
  c^2 = b^2 + Real.sqrt 3 * a^2

theorem find_b_over_a {A B C a b c : ℝ} (h : sides_in_triangle A B C a b c) : b / a = Real.sqrt 2 :=
  sorry

theorem find_angle_B {A B C a b c : ℝ} (h1 : sides_in_triangle A B C a b c) (h2 : cos_law_condition a b c)
  (h3 : b / a = Real.sqrt 2) : B = Real.pi / 4 :=
  sorry

end find_b_over_a_find_angle_B_l165_165244


namespace difference_in_speed_l165_165937

theorem difference_in_speed (d : ℕ) (tA tE : ℕ) (vA vE : ℕ) (h1 : d = 300) (h2 : tA = tE - 3) 
    (h3 : vE = 20) (h4 : vE = d / tE) (h5 : vA = d / tA) : vA - vE = 5 := 
    sorry

end difference_in_speed_l165_165937


namespace john_writing_time_l165_165626

def pages_per_day : ℕ := 20
def pages_per_book : ℕ := 400
def number_of_books : ℕ := 3

theorem john_writing_time : (pages_per_book / pages_per_day) * number_of_books = 60 :=
by
  -- The proof should be placed here.
  sorry

end john_writing_time_l165_165626


namespace range_of_a_l165_165389

def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 - 2

noncomputable def f_prime (a x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a x = 0 → x < 0) → (∃ x : ℝ, f a x = 0) → a < -Real.sqrt 2 := by
  sorry

end range_of_a_l165_165389


namespace identify_letter_R_l165_165207

variable (x y : ℕ)

def date_A : ℕ := x + 2
def date_B : ℕ := x + 5
def date_E : ℕ := x

def y_plus_x := y + x
def combined_dates := date_A x + 2 * date_B x

theorem identify_letter_R (h1 : y_plus_x x y = combined_dates x) : 
  y = 2 * x + 12 ∧ ∃ (letter : String), letter = "R" := sorry

end identify_letter_R_l165_165207


namespace savings_calculation_l165_165755

noncomputable def calculate_savings (spent_price : ℝ) (saving_pct : ℝ) : ℝ :=
  let original_price := spent_price / (1 - (saving_pct / 100))
  original_price - spent_price

-- Define the spent price and saving percentage
def spent_price : ℝ := 20
def saving_pct : ℝ := 12.087912087912088

-- Statement to be proved
theorem savings_calculation : calculate_savings spent_price saving_pct = 2.75 :=
  sorry

end savings_calculation_l165_165755


namespace true_statement_l165_165481

theorem true_statement :
  -8 < -2 := 
sorry

end true_statement_l165_165481


namespace inequality_solution_l165_165931

noncomputable def inequality_proof (a b c : ℝ) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 2) : Prop :=
  (1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + c * a)) ≥ (27 / 13)

theorem inequality_solution (a b c : ℝ) 
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_sum : a + b + c = 2) : 
  inequality_proof a b c h_positive h_sum :=
sorry

end inequality_solution_l165_165931


namespace rachel_picked_apples_l165_165703

-- Defining the conditions
def original_apples : ℕ := 11
def grown_apples : ℕ := 2
def apples_left : ℕ := 6

-- Defining the equation
def equation (x : ℕ) : Prop :=
  original_apples - x + grown_apples = apples_left

-- Stating the theorem
theorem rachel_picked_apples : ∃ x : ℕ, equation x ∧ x = 7 :=
by 
  -- proof skipped 
  sorry

end rachel_picked_apples_l165_165703


namespace calculate_length_of_train_l165_165533

noncomputable def length_of_train (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) (time_seconds : ℝ) : ℝ :=
  let relative_speed_kmh := speed_train_kmh + speed_man_kmh
  let relative_speed_ms := (relative_speed_kmh : ℝ) * 1000 / 3600
  relative_speed_ms * time_seconds

theorem calculate_length_of_train :
  length_of_train 50 5 7.2 = 110 := by
  -- This is where the actual proof would go, but it's omitted for now as per instructions.
  sorry

end calculate_length_of_train_l165_165533


namespace arithmetic_operations_correct_l165_165562

theorem arithmetic_operations_correct :
  (3 + (3 / 3) = (77 / 7) - 7) :=
by
  sorry

end arithmetic_operations_correct_l165_165562


namespace impossible_event_D_l165_165980

-- Event definitions
def event_A : Prop := true -- This event is not impossible
def event_B : Prop := true -- This event is not impossible
def event_C : Prop := true -- This event is not impossible
def event_D (bag : Finset String) : Prop :=
  if "red" ∈ bag then false else true -- This event is impossible if there are no red balls

-- Bag condition
def bag : Finset String := {"white", "white", "white", "white", "white", "white", "white", "white"}

-- Proof statement
theorem impossible_event_D : event_D bag = true :=
by
  -- The bag contains only white balls, so drawing a red ball is impossible.
  rw [event_D, if_neg]
  sorry

end impossible_event_D_l165_165980


namespace machine_subtract_l165_165597

theorem machine_subtract (x : ℤ) (h1 : 26 + 15 - x = 35) : x = 6 :=
by
  sorry

end machine_subtract_l165_165597


namespace triangle_equilateral_if_arithmetic_sequences_l165_165747

theorem triangle_equilateral_if_arithmetic_sequences
  (A B C : ℝ) (a b c : ℝ)
  (h_angles : A + B + C = 180)
  (h_angle_seq : ∃ (N : ℝ), A = B - N ∧ C = B + N)
  (h_sides : ∃ (n : ℝ), a = b - n ∧ c = b + n) :
  A = B ∧ B = C ∧ a = b ∧ b = c :=
sorry

end triangle_equilateral_if_arithmetic_sequences_l165_165747


namespace terminating_decimal_of_7_div_200_l165_165754

theorem terminating_decimal_of_7_div_200 : (7 / 200 : ℝ) = 0.028 := sorry

end terminating_decimal_of_7_div_200_l165_165754


namespace number_of_solutions_l165_165575

-- Define the equation
def equation (x : ℝ) : Prop := (3 * x^2 - 15 * x) / (x^2 - 7 * x + 10) = x - 4

-- State the problem with conditions and conclusion
theorem number_of_solutions : (∀ x : ℝ, x ≠ 2 ∧ x ≠ 5 → equation x) ↔ (∃ x1 x2 : ℝ, x1 ≠ 2 ∧ x1 ≠ 5 ∧ x2 ≠ 2 ∧ x2 ≠ 5 ∧ equation x1 ∧ equation x2) :=
by
  sorry

end number_of_solutions_l165_165575


namespace cat_chase_rat_l165_165257

/--
Given:
- The cat chases a rat 6 hours after the rat runs.
- The cat takes 4 hours to reach the rat.
- The average speed of the rat is 36 km/h.
Prove that the average speed of the cat is 90 km/h.
-/
theorem cat_chase_rat
  (t_rat_start : ℕ)
  (t_cat_chase : ℕ)
  (v_rat : ℕ)
  (h1 : t_rat_start = 6)
  (h2 : t_cat_chase = 4)
  (h3 : v_rat = 36)
  (v_cat : ℕ)
  (h4 : 4 * v_cat = t_rat_start * v_rat + t_cat_chase * v_rat) :
  v_cat = 90 :=
by
  sorry

end cat_chase_rat_l165_165257


namespace width_of_river_l165_165687

def river_depth : ℝ := 7
def flow_rate_kmph : ℝ := 4
def volume_per_minute : ℝ := 35000

noncomputable def flow_rate_mpm : ℝ := (flow_rate_kmph * 1000) / 60

theorem width_of_river : 
  ∃ w : ℝ, 
    volume_per_minute = flow_rate_mpm * river_depth * w ∧
    w = 75 :=
by
  use 75
  field_simp [flow_rate_mpm, river_depth, volume_per_minute]
  norm_num
  sorry

end width_of_river_l165_165687


namespace speed_of_stream_l165_165669

theorem speed_of_stream
  (boat_speed : ℝ)
  (downstream_distance : ℝ)
  (upstream_distance : ℝ)
  (downstream_time_eq_upstream_time : downstream_distance / (boat_speed + v) = upstream_distance / (boat_speed - v)) :
  v = 8 :=
by
  let v := 8
  sorry

end speed_of_stream_l165_165669


namespace trigonometric_identity_l165_165886

theorem trigonometric_identity (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := 
by
  sorry

end trigonometric_identity_l165_165886


namespace sandy_red_marbles_l165_165499

theorem sandy_red_marbles (jessica_marbles : ℕ) (sandy_marbles : ℕ) 
  (h₀ : jessica_marbles = 3 * 12)
  (h₁ : sandy_marbles = 4 * jessica_marbles) : 
  sandy_marbles = 144 :=
by
  sorry

end sandy_red_marbles_l165_165499


namespace lcm_gcd_product_l165_165715

theorem lcm_gcd_product (n m : ℕ) (h1 : n = 9) (h2 : m = 10) : 
  Nat.lcm n m * Nat.gcd n m = 90 := by
  sorry

end lcm_gcd_product_l165_165715


namespace base_k_perfect_square_l165_165386

theorem base_k_perfect_square (k : ℤ) (h : k ≥ 6) : 
  (1 * k^8 + 2 * k^7 + 3 * k^6 + 4 * k^5 + 5 * k^4 + 4 * k^3 + 3 * k^2 + 2 * k + 1) = (k^4 + k^3 + k^2 + k + 1)^2 := 
by
  sorry

end base_k_perfect_square_l165_165386


namespace jacqueline_has_29_percent_more_soda_than_liliane_l165_165632

variable (A : ℝ) -- A is the amount of soda Alice has

-- Define the amount of soda Jacqueline has
def J (A : ℝ) : ℝ := 1.80 * A

-- Define the amount of soda Liliane has
def L (A : ℝ) : ℝ := 1.40 * A

-- The statement that needs to be proven
theorem jacqueline_has_29_percent_more_soda_than_liliane (A : ℝ) (hA : A > 0) : 
  ((J A - L A) / L A) * 100 = 29 :=
by
  sorry

end jacqueline_has_29_percent_more_soda_than_liliane_l165_165632


namespace winning_cards_at_least_one_l165_165860

def cyclicIndex (n : ℕ) (i : ℕ) : ℕ := (i % n + n) % n

theorem winning_cards_at_least_one (a : ℕ → ℕ) (h : ∀ i, (a (cyclicIndex 8 (i - 1)) + a i + a (cyclicIndex 8 (i + 1))) % 2 = 1) :
  ∀ i, 1 ≤ a i :=
by
  sorry

end winning_cards_at_least_one_l165_165860


namespace radius_of_base_of_cone_l165_165675

theorem radius_of_base_of_cone (S : ℝ) (hS : S = 9 * Real.pi)
  (H : ∃ (l r : ℝ), (Real.pi * l = 2 * Real.pi * r) ∧ S = Real.pi * r^2 + Real.pi * r * l) :
  ∃ (r : ℝ), r = Real.sqrt 3 :=
by
  sorry

end radius_of_base_of_cone_l165_165675


namespace find_x_for_g_l165_165695

noncomputable def g (x : ℝ) : ℝ := (↑((x + 5)/6))^(1/3)

theorem find_x_for_g :
  ∃ x : ℝ, g (3 * x) = 3 * g x ∧ x = -65 / 12 :=
by
  sorry

end find_x_for_g_l165_165695


namespace inequality_proof_l165_165538

noncomputable def x : ℝ := Real.exp (-1/2)
noncomputable def y : ℝ := Real.log 2 / Real.log 5
noncomputable def z : ℝ := Real.log 3

theorem inequality_proof : z > x ∧ x > y := by
  -- Conditions defined as follows:
  -- x = exp(-1/2)
  -- y = log(2) / log(5)
  -- z = log(3)
  -- To be proved:
  -- z > x > y
  sorry

end inequality_proof_l165_165538


namespace remainder_division_P_by_D_l165_165960

def P (x : ℝ) := 8 * x^4 - 20 * x^3 + 28 * x^2 - 32 * x + 15
def D (x : ℝ) := 4 * x - 8

theorem remainder_division_P_by_D :
  let remainder := P 2 % D 2
  remainder = 31 :=
by
  -- Proof will be inserted here, but currently skipped
  sorry

end remainder_division_P_by_D_l165_165960


namespace find_n_l165_165876

noncomputable
def equilateral_triangle_area_ratio (n : ℕ) (h : n > 4) : Prop :=
  let ratio := (2 : ℚ) / (n - 2 : ℚ)
  let area_PQR := (1 / 7 : ℚ)
  let menelaus_ap_pd := (n * (n - 2) : ℚ) / 4
  let area_triangle_ABP := (2 * (n - 2) : ℚ) / (n * (n - 2) + 4)
  let area_sum := 3 * area_triangle_ABP
  (area_sum * 7 = 6 * (n * (n - 2) + 4))

theorem find_n (n : ℕ) (h : n > 4) : 
  (equilateral_triangle_area_ratio n h) → n = 6 := sorry

end find_n_l165_165876


namespace no_perfect_square_after_swap_l165_165572

def is_consecutive_digits (a b c d : ℕ) : Prop := 
  (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1)

def swap_hundreds_tens (n : ℕ) : ℕ := 
  let d4 := n / 1000
  let d3 := (n % 1000) / 100
  let d2 := (n % 100) / 10
  let d1 := n % 10
  d4 * 1000 + d2 * 100 + d3 * 10 + d1

theorem no_perfect_square_after_swap : ¬ ∃ (n : ℕ), 
  1000 ≤ n ∧ n < 10000 ∧ 
  (let d4 := n / 1000
   let d3 := (n % 1000) / 100
   let d2 := (n % 100) / 10
   let d1 := n % 10
   is_consecutive_digits d4 d3 d2 d1) ∧ 
  let new_number := swap_hundreds_tens n
  (∃ m : ℕ, m * m = new_number) := 
sorry

end no_perfect_square_after_swap_l165_165572


namespace unique_positive_integer_satisfies_condition_l165_165181

def is_positive_integer (n : ℕ) : Prop := n > 0

def condition (n : ℕ) : Prop := 20 - 5 * n ≥ 15

theorem unique_positive_integer_satisfies_condition :
  ∃! n : ℕ, is_positive_integer n ∧ condition n :=
by
  sorry

end unique_positive_integer_satisfies_condition_l165_165181


namespace lottery_profit_l165_165166

-- Definitions

def Prob_A := (1:ℚ) / 5
def Prob_B := (4:ℚ) / 15
def Prob_C := (1:ℚ) / 5
def Prob_D := (2:ℚ) / 15
def Prob_E := (1:ℚ) / 5

def customers := 300

def first_prize_value := 9
def second_prize_value := 3
def third_prize_value := 1

-- Proof Problem Statement

theorem lottery_profit : 
  (first_prize_category == "D") ∧ 
  (second_prize_category == "B") ∧ 
  (300 * 3 - ((300 * Prob_D) * 9 + (300 * Prob_B) * 3 + (300 * (Prob_A + Prob_C + Prob_E)) * 1)) == 120 :=
by 
  -- Insert mathematical proof here using given probabilities and conditions
  sorry

end lottery_profit_l165_165166


namespace unique_solution_system_eqns_l165_165989

theorem unique_solution_system_eqns :
  ∃ (x y : ℝ), (2 * x - 3 * |y| = 1 ∧ |x| + 2 * y = 4 ∧ x = 2 ∧ y = 1) :=
sorry

end unique_solution_system_eqns_l165_165989


namespace solve_equation_l165_165086

theorem solve_equation (x : ℝ) : (3 * x - 2 * (10 - x) = 5) → x = 5 :=
by {
  sorry
}

end solve_equation_l165_165086


namespace school_seat_payment_l165_165768

def seat_cost (num_rows : ℕ) (seats_per_row : ℕ) (cost_per_seat : ℕ) (discount : ℕ → ℕ → ℕ) : ℕ :=
  let total_seats := num_rows * seats_per_row
  let total_cost := total_seats * cost_per_seat
  let groups_of_ten := total_seats / 10
  let total_discount := groups_of_ten * discount 10 cost_per_seat
  total_cost - total_discount

-- Define the discount function as 10% of the cost of a group of 10 seats
def discount (group_size : ℕ) (cost_per_seat : ℕ) : ℕ := (group_size * cost_per_seat) / 10

theorem school_seat_payment :
  seat_cost 5 8 30 discount = 1080 :=
sorry

end school_seat_payment_l165_165768


namespace min_correct_answers_l165_165321

theorem min_correct_answers (total_questions correct_points incorrect_points target_score : ℕ)
                            (h_total : total_questions = 22)
                            (h_correct_points : correct_points = 4)
                            (h_incorrect_points : incorrect_points = 2)
                            (h_target : target_score = 81) :
  ∃ x : ℕ, 4 * x - 2 * (22 - x) > 81 ∧ x ≥ 21 :=
by {
  sorry
}

end min_correct_answers_l165_165321


namespace union_A_B_l165_165292

-- Definitions based on the conditions
def A := { x : ℝ | x < -1 ∨ (2 ≤ x ∧ x < 3) }
def B := { x : ℝ | -2 ≤ x ∧ x < 4 }

-- The proof goal
theorem union_A_B : A ∪ B = { x : ℝ | x < 4 } :=
by
  sorry -- Proof placeholder

end union_A_B_l165_165292


namespace robin_initial_gum_l165_165274

theorem robin_initial_gum (x : ℕ) (h1 : x + 26 = 44) : x = 18 := 
by 
  sorry

end robin_initial_gum_l165_165274


namespace tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n_l165_165064

/-- Given the trigonometric identity and the ratio, we want to prove the relationship between the tangents of the angles. -/
theorem tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n
  (α β m n : ℝ)
  (h : (Real.sin (α + β)) / (Real.sin (α - β)) = m / n) :
  (Real.tan β) / (Real.tan α) = (m - n) / (m + n) :=
  sorry

end tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n_l165_165064


namespace cupcakes_gluten_nut_nonvegan_l165_165335

-- Definitions based on conditions
def total_cupcakes := 120
def gluten_free_cupcakes := total_cupcakes / 3
def vegan_cupcakes := total_cupcakes / 4
def nut_free_cupcakes := total_cupcakes / 5
def gluten_and_vegan_cupcakes := 15
def vegan_and_nut_free_cupcakes := 10

-- Defining the theorem to prove the main question
theorem cupcakes_gluten_nut_nonvegan : 
  total_cupcakes - ((gluten_free_cupcakes + (vegan_cupcakes - gluten_and_vegan_cupcakes)) - vegan_and_nut_free_cupcakes) = 65 :=
by sorry

end cupcakes_gluten_nut_nonvegan_l165_165335


namespace estimated_total_fish_population_l165_165441

-- Definitions of the initial conditions
def tagged_fish_in_first_catch : ℕ := 100
def total_fish_in_second_catch : ℕ := 300
def tagged_fish_in_second_catch : ℕ := 15

-- The theorem to prove the estimated number of total fish in the pond
theorem estimated_total_fish_population (tagged_fish_in_first_catch : ℕ) (total_fish_in_second_catch : ℕ) (tagged_fish_in_second_catch : ℕ) : ℕ :=
  2000

-- Assertion of the theorem with actual numbers
example : estimated_total_fish_population tagged_fish_in_first_catch total_fish_in_second_catch tagged_fish_in_second_catch = 2000 := by
  sorry

end estimated_total_fish_population_l165_165441


namespace weight_of_replaced_person_l165_165893

theorem weight_of_replaced_person
  (avg_increase : ∀ W : ℝ, W + 8 * 2.5 = W - X + 80)
  (new_person_weight : 80 = 80):
  X = 60 := by
  sorry

end weight_of_replaced_person_l165_165893


namespace find_g1_l165_165733

open Function

-- Definitions based on the conditions
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

theorem find_g1 (g : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f g (-x) + f g x = 0) 
  (h2 : g (-1) = 1) 
  : g 1 = -3 :=
sorry

end find_g1_l165_165733


namespace multiplication_addition_example_l165_165909

theorem multiplication_addition_example :
  469138 * 9999 + 876543 * 12345 = 15512230997 :=
by
  sorry

end multiplication_addition_example_l165_165909


namespace sixth_distance_l165_165812

theorem sixth_distance (A B C D : Point)
  (dist_AB dist_AC dist_BC dist_AD dist_BD dist_CD : ℝ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_lengths : (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BC = 1 ∧ dist_AD = 1) ∨
               (dist_AB = 1 ∧ dist_AC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AB = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_CD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1) ∨
               (dist_AC = 1 ∧ dist_AD = 1 ∧ dist_BD = 1 ∧ dist_CD = 1) ∨
               (dist_AD = 1 ∧ dist_BC = 1 ∧ dist_BD = 1 ∧ dist_CD = 1))
  (h_one_point_two : dist_AB = 1.2 ∨ dist_AC = 1.2 ∨ dist_BC = 1.2 ∨ dist_AD = 1.2 ∨ dist_BD = 1.2 ∨ dist_CD = 1.2) :
  dist_AB = 1.84 ∨ dist_AB = 0.24 ∨ dist_AB = 1.6 ∨
  dist_AC = 1.84 ∨ dist_AC = 0.24 ∨ dist_AC = 1.6 ∨
  dist_BC = 1.84 ∨ dist_BC = 0.24 ∨ dist_BC = 1.6 ∨
  dist_AD = 1.84 ∨ dist_AD = 0.24 ∨ dist_AD = 1.6 ∨
  dist_BD = 1.84 ∨ dist_BD = 0.24 ∨ dist_BD = 1.6 ∨
  dist_CD = 1.84 ∨ dist_CD = 0.24 ∨ dist_CD = 1.6 :=
sorry

end sixth_distance_l165_165812


namespace sum_of_abcd_is_1_l165_165102

theorem sum_of_abcd_is_1
  (a b c d : ℤ)
  (h1 : (x^2 + a*x + b)*(x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 8*x - 12) :
  a + b + c + d = 1 := by
  sorry

end sum_of_abcd_is_1_l165_165102


namespace sum_of_f_values_l165_165019

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f_values :
  (∀ x : ℝ, f x + f (-x) = 0) →
  (∀ x : ℝ, f x = f (x + 2)) →
  (∀ x : ℝ, 0 ≤ x → x < 1 → f x = 2^x - 1) →
  f (1/2) + f 1 + f (3/2) + f 2 + f (5/2) = Real.sqrt 2 - 1 :=
by
  intros h1 h2 h3
  sorry

end sum_of_f_values_l165_165019


namespace number_of_divisors_2310_l165_165419

theorem number_of_divisors_2310 : Nat.sqrt 2310 = 32 :=
by
  sorry

end number_of_divisors_2310_l165_165419


namespace opposite_of_neg_three_l165_165539

theorem opposite_of_neg_three : -(-3) = 3 :=
by 
  sorry

end opposite_of_neg_three_l165_165539


namespace cupcakes_frosted_in_10_minutes_l165_165824

def frosting_rate (time: ℕ) (cupcakes: ℕ) : ℚ := cupcakes / time

noncomputable def combined_frosting_rate : ℚ :=
  (frosting_rate 25 1) + (frosting_rate 35 1)

def effective_working_time (total_time: ℕ) (work_period: ℕ) (break_time: ℕ) : ℕ :=
  let break_intervals := total_time / work_period
  total_time - break_intervals * break_time

def total_cupcakes (working_time: ℕ) (rate: ℚ) : ℚ :=
  working_time * rate

theorem cupcakes_frosted_in_10_minutes :
  total_cupcakes (effective_working_time 600 240 30) combined_frosting_rate = 36 := by
  sorry

end cupcakes_frosted_in_10_minutes_l165_165824


namespace value_of_x_plus_y_l165_165661

theorem value_of_x_plus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x / 3 = y^2) (h2 : x / 9 = 9 * y) : 
  x + y = 2214 :=
sorry

end value_of_x_plus_y_l165_165661


namespace simplify_expression_l165_165149

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 4 * x^3 + 5 * x^2 + 2 * x + 8 - (3 * x^3 - 7 * x^2 + 4 * x - 6)

-- Define the simplified form
def simplified_expr (x : ℝ) : ℝ := x^3 + 12 * x^2 - 2 * x + 14

-- State the theorem
theorem simplify_expression (x : ℝ) : initial_expr x = simplified_expr x :=
by sorry

end simplify_expression_l165_165149


namespace largest_integer_solution_l165_165453

theorem largest_integer_solution (x : ℤ) (h : (x : ℚ) / 3 + 4 / 5 < 5 / 3) : x ≤ 2 :=
sorry

end largest_integer_solution_l165_165453


namespace prove_b_is_neg_two_l165_165319

-- Define the conditions
variables (b : ℝ)

-- Hypothesis: The real and imaginary parts of the complex number (2 - b * I) * I are opposites
def complex_opposite_parts (b : ℝ) : Prop :=
  b = -2

-- The theorem statement
theorem prove_b_is_neg_two : complex_opposite_parts b :=
sorry

end prove_b_is_neg_two_l165_165319


namespace seat_to_right_proof_l165_165392

def Xiaofang_seat : ℕ × ℕ := (3, 5)

def seat_to_right (seat : ℕ × ℕ) : ℕ × ℕ :=
  (seat.1 + 1, seat.2)

theorem seat_to_right_proof : seat_to_right Xiaofang_seat = (4, 5) := by
  unfold Xiaofang_seat
  unfold seat_to_right
  sorry

end seat_to_right_proof_l165_165392


namespace scientific_notation_of_170000_l165_165053

-- Define the concept of scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  (1 ≤ a) ∧ (a < 10) ∧ (x = a * 10^n)

-- The main statement to prove
theorem scientific_notation_of_170000 : is_scientific_notation 1.7 5 170000 :=
by sorry

end scientific_notation_of_170000_l165_165053


namespace range_of_a_l165_165683

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x
noncomputable def g (x a : ℝ) : ℝ := -(x + 1)^2 + a

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 : ℝ, f x2 ≤ g x1 a) ↔ a ≥ -1 / Real.exp 1 :=
by
  -- proof would go here
  sorry

end range_of_a_l165_165683


namespace geom_seq_sum_l165_165604

noncomputable def geom_seq (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
a₁ * r^(n-1)

theorem geom_seq_sum (a₁ r : ℝ) (h_pos : 0 < a₁) (h_pos_r : 0 < r)
  (h : a₁ * (geom_seq a₁ r 5) + 2 * (geom_seq a₁ r 3) * (geom_seq a₁ r 6) + a₁ * (geom_seq a₁ r 11) = 16) :
  (geom_seq a₁ r 3 + geom_seq a₁ r 6) = 4 :=
sorry

end geom_seq_sum_l165_165604


namespace total_students_is_correct_l165_165307

-- Define the number of students in each class based on the conditions
def number_of_students_finley := 24
def number_of_students_johnson := (number_of_students_finley / 2) + 10
def number_of_students_garcia := 2 * number_of_students_johnson
def number_of_students_smith := number_of_students_finley / 3
def number_of_students_patel := (3 / 4) * (number_of_students_finley + number_of_students_johnson + number_of_students_garcia)

-- Define the total number of students in all five classes combined
def total_number_of_students := 
  number_of_students_finley + 
  number_of_students_johnson + 
  number_of_students_garcia +
  number_of_students_smith + 
  number_of_students_patel

-- The theorem statement to prove
theorem total_students_is_correct : total_number_of_students = 166 := by
  sorry

end total_students_is_correct_l165_165307


namespace dice_probabilities_relationship_l165_165147

theorem dice_probabilities_relationship :
  let p1 := 5 / 18
  let p2 := 11 / 18
  let p3 := 1 / 2
  p1 < p3 ∧ p3 < p2
:= by
  sorry

end dice_probabilities_relationship_l165_165147


namespace EdProblem_l165_165269

/- Define the conditions -/
def EdConditions := 
  ∃ (m : ℕ) (N : ℕ), 
    m = 16 ∧ 
    N = Nat.choose 15 5 ∧
    N % 1000 = 3

/- The statement to be proven -/
theorem EdProblem : EdConditions :=
  sorry

end EdProblem_l165_165269


namespace geometric_sequence_terms_l165_165387

theorem geometric_sequence_terms
  (a : ℚ) (l : ℚ) (r : ℚ) (n : ℕ)
  (h_a : a = 9 / 8)
  (h_l : l = 1 / 3)
  (h_r : r = 2 / 3)
  (h_geo : l = a * r^(n - 1)) :
  n = 4 :=
by
  sorry

end geometric_sequence_terms_l165_165387


namespace first_number_in_set_l165_165006

theorem first_number_in_set (x : ℝ)
  (h : (x + 40 + 60) / 3 = (10 + 80 + 15) / 3 + 5) :
  x = 20 := by
  sorry

end first_number_in_set_l165_165006


namespace num_ways_to_select_officers_l165_165159

def ways_to_select_five_officers (n : ℕ) (k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldl (λ acc x => acc * x) 1

theorem num_ways_to_select_officers :
  ways_to_select_five_officers 12 5 = 95040 :=
by
  -- By definition of ways_to_select_five_officers, this is equivalent to 12 * 11 * 10 * 9 * 8.
  sorry

end num_ways_to_select_officers_l165_165159


namespace sum_of_products_eq_131_l165_165976

theorem sum_of_products_eq_131 (a b c : ℝ) 
    (h1 : a^2 + b^2 + c^2 = 222)
    (h2 : a + b + c = 22) : 
    a * b + b * c + c * a = 131 :=
by
  sorry

end sum_of_products_eq_131_l165_165976


namespace total_birds_in_marsh_l165_165333

def number_of_geese : Nat := 58
def number_of_ducks : Nat := 37

theorem total_birds_in_marsh :
  number_of_geese + number_of_ducks = 95 :=
sorry

end total_birds_in_marsh_l165_165333


namespace system_solutions_l165_165443

theorem system_solutions (x y z a b c : ℝ) :
  (a = 1 ∨ b = 1 ∨ c = 1 ∨ a + b + c + a * b * c = 0) → (¬(x = 1 ∨ y = 1 ∨ z = 1) → 
  ∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c) ∨
  (a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ a + b + c + a * b * c ≠ 0) → 
  ¬∃ (x y z : ℝ), (x - y) / (z - 1) = a ∧ (y - z) / (x - 1) = b ∧ (z - x) / (y - 1) = c :=
by
    sorry

end system_solutions_l165_165443


namespace fractional_equation_solution_l165_165941

theorem fractional_equation_solution (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) ↔ (m ≤ 2 ∧ m ≠ -2) := 
sorry

end fractional_equation_solution_l165_165941


namespace value_of_f_at_9_l165_165775

def f (n : ℕ) : ℕ := n^3 + n^2 + n + 17

theorem value_of_f_at_9 : f 9 = 836 := sorry

end value_of_f_at_9_l165_165775


namespace max_reached_at_2001_l165_165243

noncomputable def a (n : ℕ) : ℝ := n^2 / 1.001^n

theorem max_reached_at_2001 : ∀ n : ℕ, a 2001 ≥ a n := 
sorry

end max_reached_at_2001_l165_165243


namespace average_first_two_l165_165553

theorem average_first_two (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) = 16.8)
  (h2 : (c + d) = 4.6)
  (h3 : (e + f) = 7.4) : 
  (a + b) / 2 = 2.4 :=
by
  sorry

end average_first_two_l165_165553


namespace value_of_e_l165_165009

theorem value_of_e (a : ℕ) (e : ℕ) 
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * 45 * e) : 
  e = 49 := 
by 
  sorry

end value_of_e_l165_165009


namespace samantha_birth_year_l165_165740

theorem samantha_birth_year :
  ∀ (first_amc : ℕ) (amc9_year : ℕ) (samantha_age_in_amc9 : ℕ),
  (first_amc = 1983) →
  (amc9_year = first_amc + 8) →
  (samantha_age_in_amc9 = 13) →
  (amc9_year - samantha_age_in_amc9 = 1978) :=
by
  intros first_amc amc9_year samantha_age_in_amc9 h1 h2 h3
  sorry

end samantha_birth_year_l165_165740


namespace sequence_formula_l165_165377

theorem sequence_formula (a : ℕ → ℝ)
  (h1 : ∀ n : ℕ, a n ≠ 0)
  (h2 : a 1 = 1)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 1) = 1 / (n + 1 + 1 / (a n))) :
  ∀ n : ℕ, n > 0 → a n = 2 / ((n : ℝ) ^ 2 - n + 2) :=
by
  sorry

end sequence_formula_l165_165377


namespace trapezoid_median_l165_165723

noncomputable def median_trapezoid (base₁ base₂ height : ℝ) : ℝ :=
(base₁ + base₂) / 2

theorem trapezoid_median (b_t : ℝ) (a_t : ℝ) (h_t : ℝ) (a_tp : ℝ) 
  (h_eq : h_t = 16) (a_eq : a_t = 192) (area_tp_eq : a_tp = a_t) : median_trapezoid h_t h_t h_t = 12 :=
by
  have h_t_eq : h_t = 16 := by sorry
  have a_t_eq : a_t = 192 := by sorry
  have area_tp : a_tp = 192 := by sorry
  sorry

end trapezoid_median_l165_165723


namespace opposite_of_2023_is_neg2023_l165_165491

-- Given number and definition of the additive inverse
def given_number : ℤ := 2023

-- The definition of opposite (additive inverse)
def is_opposite (x y : ℤ) : Prop := x + y = 0

-- Statement of the theorem
theorem opposite_of_2023_is_neg2023 : is_opposite given_number (-2023) :=
sorry

end opposite_of_2023_is_neg2023_l165_165491


namespace opposite_of_three_l165_165554

theorem opposite_of_three :
  ∃ x : ℤ, 3 + x = 0 ∧ x = -3 :=
by
  sorry

end opposite_of_three_l165_165554


namespace hyperbola_eccentricity_l165_165977

theorem hyperbola_eccentricity (m : ℝ) (h1: ∃ x y : ℝ, (x^2 / 3) - (y^2 / m) = 1) (h2: ∀ a b : ℝ, a^2 = 3 ∧ b^2 = m ∧ (2 = Real.sqrt (1 + b^2 / a^2))) : m = -9 := 
sorry

end hyperbola_eccentricity_l165_165977


namespace no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l165_165410

theorem no_sequence_of_14_consecutive_divisible_by_some_prime_le_11 :
  ¬ ∃ n : ℕ, ∀ k : ℕ, k < 14 → ∃ p ∈ [2, 3, 5, 7, 11], (n + k) % p = 0 :=
by
  sorry

end no_sequence_of_14_consecutive_divisible_by_some_prime_le_11_l165_165410


namespace rectangle_area_l165_165408

theorem rectangle_area (b l: ℕ) (h1: l = 3 * b) (h2: 2 * (l + b) = 120) : l * b = 675 := 
by 
  sorry

end rectangle_area_l165_165408


namespace smallest_third_term_arith_seq_l165_165112

theorem smallest_third_term_arith_seq {a d : ℕ} 
  (h1 : a > 0) 
  (h2 : d > 0) 
  (sum_eq : 5 * a + 10 * d = 80) : 
  a + 2 * d = 16 := 
by {
  sorry
}

end smallest_third_term_arith_seq_l165_165112


namespace cos_alpha_minus_pi_six_l165_165300

theorem cos_alpha_minus_pi_six (α : ℝ) (h : Real.sin (α + Real.pi / 3) = 4 / 5) : 
  Real.cos (α - Real.pi / 6) = 4 / 5 :=
sorry

end cos_alpha_minus_pi_six_l165_165300


namespace sum_of_numbers_l165_165477

theorem sum_of_numbers : 
  (87 + 91 + 94 + 88 + 93 + 91 + 89 + 87 + 92 + 86 + 90 + 92 + 88 + 90 + 91 + 86 + 89 + 92 + 95 + 88) = 1799 := 
by 
  sorry

end sum_of_numbers_l165_165477


namespace similar_triangles_legs_l165_165802

theorem similar_triangles_legs (y : ℝ) (h : 12 / y = 9 / 7) : y = 84 / 9 := by
  sorry

end similar_triangles_legs_l165_165802


namespace factor_quadratic_l165_165796

theorem factor_quadratic (x : ℝ) : (16 * x^2 - 40 * x + 25) = (4 * x - 5)^2 :=
by 
  sorry

end factor_quadratic_l165_165796


namespace find_b_minus_c_l165_165855

noncomputable def a (n : ℕ) : ℝ :=
  if h : n > 1 then 1 / Real.log 1009 * Real.log n else 0

noncomputable def b : ℝ :=
  a 2 + a 3 + a 4 + a 5 + a 6

noncomputable def c : ℝ :=
  a 15 + a 16 + a 17 + a 18 + a 19

theorem find_b_minus_c : b - c = -Real.logb 1009 1938 := by
  sorry

end find_b_minus_c_l165_165855


namespace sum_of_common_ratios_l165_165136

variable {k a_2 a_3 b_2 b_3 p r : ℝ}
variable (hp : a_2 = k * p) (ha3 : a_3 = k * p^2)
variable (hr : b_2 = k * r) (hb3 : b_3 = k * r^2)
variable (hcond : a_3 - b_3 = 5 * (a_2 - b_2))

theorem sum_of_common_ratios (h_nonconst : k ≠ 0) (p_ne_r : p ≠ r) : p + r = 5 :=
by
  sorry

end sum_of_common_ratios_l165_165136


namespace couscous_problem_l165_165518

def total_couscous (S1 S2 S3 : ℕ) : ℕ :=
  S1 + S2 + S3

def couscous_per_dish (total : ℕ) (dishes : ℕ) : ℕ :=
  total / dishes

theorem couscous_problem 
  (S1 S2 S3 : ℕ) (dishes : ℕ) 
  (h1 : S1 = 7) (h2 : S2 = 13) (h3 : S3 = 45) (h4 : dishes = 13) :
  couscous_per_dish (total_couscous S1 S2 S3) dishes = 5 := by  
  sorry

end couscous_problem_l165_165518


namespace CH4_reaction_with_Cl2_l165_165061

def balanced_chemical_equation (CH4 Cl2 CH3Cl HCl : ℕ) : Prop :=
  CH4 + Cl2 = CH3Cl + HCl

theorem CH4_reaction_with_Cl2
  (CH4 Cl2 CH3Cl HCl : ℕ)
  (balanced_eq : balanced_chemical_equation 1 1 1 1)
  (reaction_cl2 : Cl2 = 2) :
  CH4 = 2 :=
by
  sorry

end CH4_reaction_with_Cl2_l165_165061


namespace smallest_value_of_c_l165_165605

def bound_a (a b : ℝ) : Prop := 1 + a ≤ b
def bound_inv (a b c : ℝ) : Prop := (1 / a) + (1 / b) ≤ (1 / c)

theorem smallest_value_of_c (a b c : ℝ) (ha : 1 < a) (hb : a < b) 
  (hc : b < c) (h_ab : bound_a a b) (h_inv : bound_inv a b c) : 
  c ≥ (3 + Real.sqrt 5) / 2 := 
sorry

end smallest_value_of_c_l165_165605


namespace second_die_sides_l165_165137

theorem second_die_sides (p : ℚ) (n : ℕ) (h1 : p = 0.023809523809523808) (h2 : n ≠ 0) :
  let first_die_sides := 6
  let probability := (1 : ℚ) / first_die_sides * (1 : ℚ) / n
  probability = p → n = 7 :=
by
  intro h
  sorry

end second_die_sides_l165_165137


namespace xy_sum_correct_l165_165276

theorem xy_sum_correct (x y : ℝ) 
  (h : (4 + 10 + 16 + 24) / 4 = (14 + x + y) / 3) : 
  x + y = 26.5 :=
by
  sorry

end xy_sum_correct_l165_165276


namespace Julio_spent_on_limes_l165_165793

theorem Julio_spent_on_limes
  (days : ℕ)
  (lime_cost_per_3 : ℕ)
  (mocktails_per_day : ℕ)
  (lime_juice_per_lime_tbsp : ℕ)
  (lime_juice_per_mocktail_tbsp : ℕ)
  (limes_per_set : ℕ)
  (days_eq_30 : days = 30)
  (lime_cost_per_3_eq_1 : lime_cost_per_3 = 1)
  (mocktails_per_day_eq_1 : mocktails_per_day = 1)
  (lime_juice_per_lime_tbsp_eq_2 : lime_juice_per_lime_tbsp = 2)
  (lime_juice_per_mocktail_tbsp_eq_1 : lime_juice_per_mocktail_tbsp = 1)
  (limes_per_set_eq_3 : limes_per_set = 3) :
  days * mocktails_per_day * lime_juice_per_mocktail_tbsp / lime_juice_per_lime_tbsp / limes_per_set * lime_cost_per_3 = 5 :=
sorry

end Julio_spent_on_limes_l165_165793


namespace balls_distribution_l165_165020

def balls_into_boxes : Nat := 6
def boxes : Nat := 3
def at_least_one_in_first (n m : Nat) : ℕ := sorry -- Use a function with appropriate constraints to ensure at least 1 ball is in the first box

theorem balls_distribution (n m : Nat) (h: n = 6) (h2: m = 3) :
  at_least_one_in_first n m = 665 :=
by
  sorry

end balls_distribution_l165_165020


namespace find_n_l165_165973

theorem find_n :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * Real.pi / 180) = Real.cos (942 * Real.pi / 180) := sorry

end find_n_l165_165973


namespace percentage_markup_on_cost_price_l165_165794

theorem percentage_markup_on_cost_price 
  (SP : ℝ) (CP : ℝ) (hSP : SP = 6400) (hCP : CP = 5565.217391304348) : 
  ((SP - CP) / CP) * 100 = 15 :=
by
  -- proof would go here
  sorry

end percentage_markup_on_cost_price_l165_165794


namespace solve_fraction_eq_zero_l165_165420

theorem solve_fraction_eq_zero (x : ℝ) (h : (x - 3) / (2 * x + 5) = 0) (h2 : 2 * x + 5 ≠ 0) : x = 3 :=
sorry

end solve_fraction_eq_zero_l165_165420


namespace probability_player_A_wins_first_B_wins_second_l165_165844

theorem probability_player_A_wins_first_B_wins_second :
  (1 / 2) * (4 / 5) * (2 / 3) + (1 / 2) * (1 / 3) * (2 / 3) = 17 / 45 :=
by
  sorry

end probability_player_A_wins_first_B_wins_second_l165_165844


namespace servant_service_duration_l165_165100

variables (x : ℕ) (total_compensation full_months received_compensation : ℕ)
variables (price_uniform compensation_cash : ℕ)

theorem servant_service_duration :
  total_compensation = 1000 →
  full_months = 12 →
  received_compensation = (compensation_cash + price_uniform) →
  received_compensation = 750 →
  total_compensation = (compensation_cash + price_uniform) →
  x / full_months = 750 / total_compensation →
  x = 9 :=
by sorry

end servant_service_duration_l165_165100


namespace largest_number_l165_165659

theorem largest_number (a b c : ℕ) (h1 : c = a + 6) (h2 : b = (a + c) / 2) (h3 : a * b * c = 46332) : 
  c = 39 := 
sorry

end largest_number_l165_165659


namespace possible_values_l165_165882

theorem possible_values (a b : ℕ → ℕ) (h1 : ∀ n, a n < (a (n + 1)))
  (h2 : ∀ n, b n < (b (n + 1)))
  (h3 : a 10 = b 10)
  (h4 : a 10 < 2017)
  (h5 : ∀ n, a (n + 2) = a (n + 1) + a n)
  (h6 : ∀ n, b (n + 1) = 2 * b n) :
  ∃ (a1 b1 : ℕ), (a 1 = a1) ∧ (b 1 = b1) ∧ (a1 + b1 = 13 ∨ a1 + b1 = 20) := sorry

end possible_values_l165_165882


namespace smallest_positive_m_l165_165119

theorem smallest_positive_m (m : ℕ) (h : ∃ n : ℤ, m^3 - 90 = n * (m + 9)) : m = 12 :=
by
  sorry

end smallest_positive_m_l165_165119


namespace fourth_power_square_prime_l165_165328

noncomputable def fourth_smallest_prime := 7

theorem fourth_power_square_prime :
  (fourth_smallest_prime ^ 2) ^ 4 = 5764801 :=
by
  -- This is a placeholder for the actual proof.
  sorry

end fourth_power_square_prime_l165_165328


namespace smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l165_165101

def is_palindrome (n : ℕ) (b : ℕ) : Prop :=
  let digits := n.digits b
  digits = digits.reverse

theorem smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5 :
  ∃ n : ℕ, n = 0b11011 ∧ is_palindrome n 2 ∧ is_palindrome n 5 :=
by
  existsi 0b11011
  sorry

end smallest_5digit_palindrome_base2_expressed_as_3digit_palindrome_base5_l165_165101


namespace pond_ratios_l165_165003

theorem pond_ratios (T A : ℕ) (h1 : T = 48) (h2 : A = 32) : A / (T - A) = 2 :=
by
  sorry

end pond_ratios_l165_165003


namespace maximum_value_of_x_minus_y_is_sqrt8_3_l165_165073

variable {x y z : ℝ}

noncomputable def maximum_value_of_x_minus_y (x y z : ℝ) : ℝ :=
  x - y

theorem maximum_value_of_x_minus_y_is_sqrt8_3 (h1 : x + y + z = 2) (h2 : x * y + y * z + z * x = 1) : 
  maximum_value_of_x_minus_y x y z = Real.sqrt (8 / 3) :=
sorry

end maximum_value_of_x_minus_y_is_sqrt8_3_l165_165073


namespace max_value_of_expr_l165_165270

theorem max_value_of_expr : ∃ t : ℝ, (∀ u : ℝ, (3^u - 2*u) * u / 9^u ≤ (3^t - 2*t) * t / 9^t) ∧ (3^t - 2*t) * t / 9^t = 1/8 :=
by sorry

end max_value_of_expr_l165_165270


namespace find_12th_term_l165_165728

noncomputable def geometric_sequence (a r : ℝ) : ℕ → ℝ
| 0 => a
| (n+1) => r * geometric_sequence a r n

theorem find_12th_term : ∃ a r, geometric_sequence a r 4 = 5 ∧ geometric_sequence a r 7 = 40 ∧ geometric_sequence a r 11 = 640 :=
by
  -- statement only, no proof provided
  sorry

end find_12th_term_l165_165728


namespace largest_angle_of_pentagon_l165_165337

theorem largest_angle_of_pentagon (x : ℝ) : 
  (2*x + 2) + 3*x + 4*x + 5*x + (6*x - 2) = 540 → 
  6*x - 2 = 160 :=
by
  intro h
  sorry

end largest_angle_of_pentagon_l165_165337


namespace average_brown_MnMs_l165_165172

theorem average_brown_MnMs 
  (a1 a2 a3 a4 a5 : ℕ)
  (h1 : a1 = 9)
  (h2 : a2 = 12)
  (h3 : a3 = 8)
  (h4 : a4 = 8)
  (h5 : a5 = 3) : 
  (a1 + a2 + a3 + a4 + a5) / 5 = 8 :=
by
  sorry

end average_brown_MnMs_l165_165172


namespace population_stable_at_K_l165_165398

-- Definitions based on conditions
def follows_S_curve (population : ℕ → ℝ) : Prop := sorry
def relatively_stable_at_K (population : ℕ → ℝ) (K : ℝ) : Prop := sorry
def ecological_factors_limit (population : ℕ → ℝ) : Prop := sorry

-- The main statement to be proved
theorem population_stable_at_K (population : ℕ → ℝ) (K : ℝ) :
  follows_S_curve population ∧ relatively_stable_at_K population K ∧ ecological_factors_limit population →
  relatively_stable_at_K population K :=
by sorry

end population_stable_at_K_l165_165398


namespace person_A_work_days_l165_165920

theorem person_A_work_days (x : ℝ) (h1 : 0 < x) 
                                 (h2 : ∃ b_work_rate, b_work_rate = 1 / 30) 
                                 (h3 : 5 * (1 / x + 1 / 30) = 0.5) : 
  x = 15 :=
by
-- Proof omitted
sorry

end person_A_work_days_l165_165920


namespace smallest_right_triangle_area_l165_165142

theorem smallest_right_triangle_area
  (a b : ℕ)
  (h₁ : a = 6)
  (h₂ : b = 8)
  (h₃ : ∃ c : ℕ, a * a + b * b = c * c) :
  (∃ A : ℕ, A = (1 / 2) * a * b) :=
by
  use 24
  sorry

end smallest_right_triangle_area_l165_165142


namespace proof_inequality_l165_165121

noncomputable def problem_statement (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) : Prop :=
  a + b + c ≤ (a ^ 4 + b ^ 4 + c ^ 4) / (a * b * c)

theorem proof_inequality (a b c : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) :
  problem_statement a b c h_a h_b h_c :=
by
  sorry

end proof_inequality_l165_165121


namespace half_sum_squares_ge_product_l165_165508

theorem half_sum_squares_ge_product (x y : ℝ) : 
  1 / 2 * (x^2 + y^2) ≥ x * y := 
by 
  sorry

end half_sum_squares_ge_product_l165_165508


namespace point_of_tangent_parallel_x_axis_l165_165432

theorem point_of_tangent_parallel_x_axis :
  ∃ M : ℝ × ℝ, (M.1 = -1 ∧ M.2 = -3) ∧
    (∃ y : ℝ, y = M.1^2 + 2 * M.1 - 2 ∧
    (∃ y' : ℝ, y' = 2 * M.1 + 2 ∧ y' = 0)) :=
sorry

end point_of_tangent_parallel_x_axis_l165_165432


namespace area_triangle_possible_values_l165_165787

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1 / 2 * a * c * Real.sin B

theorem area_triangle_possible_values (a b c : ℝ) (A B C : ℝ) (ha : a = 2) (hc : c = 2 * Real.sqrt 3) (hA : A = Real.pi / 6) :
  ∃ S, S = 2 * Real.sqrt 3 ∨ S = Real.sqrt 3 :=
by
  -- Define the area using the given values
  sorry

end area_triangle_possible_values_l165_165787


namespace monotonic_intervals_minimum_m_value_l165_165762

noncomputable def f (x : ℝ) (a : ℝ) := (2 * Real.exp 1 + 1) * Real.log x - (3 * a / 2) * x + 1

theorem monotonic_intervals (a : ℝ) : 
  if a ≤ 0 then ∀ x ∈ Set.Ioi 0, 0 < (2 * Real.exp 1 + 1) / x - (3 * a / 2) 
  else ∀ x ∈ Set.Ioc 0 ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) > 0 ∧
       ∀ x ∈ Set.Ioi ((2 * (2 * Real.exp 1 + 1)) / (3 * a)), (2 * Real.exp 1 + 1) / x - (3 * a / 2) < 0 := sorry

noncomputable def g (x : ℝ) (m : ℝ) := x * Real.exp x + m - ((2 * Real.exp 1 + 1) * Real.log x + x - 1)

theorem minimum_m_value :
  ∀ (m : ℝ), (∀ (x : ℝ), 0 < x → g x m ≥ 0) ↔ m ≥ - Real.exp 1 := sorry

end monotonic_intervals_minimum_m_value_l165_165762


namespace area_of_inscribed_triangle_l165_165565

noncomputable def calculate_triangle_area_inscribed_in_circle 
  (arc1 : ℝ) (arc2 : ℝ) (arc3 : ℝ) (total_circumference := arc1 + arc2 + arc3)
  (radius := total_circumference / (2 * Real.pi))
  (theta := (2 * Real.pi) / total_circumference)
  (angle1 := 5 * theta) (angle2 := 7 * theta) (angle3 := 8 * theta) : ℝ :=
  0.5 * (radius ^ 2) * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem area_of_inscribed_triangle : 
  calculate_triangle_area_inscribed_in_circle 5 7 8 = 119.85 / (Real.pi ^ 2) :=
by
  sorry

end area_of_inscribed_triangle_l165_165565


namespace constant_sums_l165_165133

theorem constant_sums (n : ℕ) 
  (x y z : ℝ) 
  (h₁ : x + y + z = 0) 
  (h₂ : x * y * z = 1) 
  : (x^n + y^n + z^n = 0 ∨ x^n + y^n + z^n = 3) ↔ (n = 1 ∨ n = 3) :=
by sorry

end constant_sums_l165_165133


namespace infinite_n_multiples_of_six_available_l165_165894

theorem infinite_n_multiples_of_six_available :
  ∃ (S : Set ℕ), (∀ n ∈ S, ∃ (A : Matrix (Fin 3) (Fin (n : ℕ)) Nat),
    (∀ (i : Fin n), (A 0 i + A 1 i + A 2 i) % 6 = 0) ∧ 
    (∀ (i : Fin 3), (Finset.univ.sum (λ j => A i j)) % 6 = 0)) ∧
  Set.Infinite S :=
sorry

end infinite_n_multiples_of_six_available_l165_165894


namespace find_nat_int_l165_165639

theorem find_nat_int (x y : ℕ) (h : x^2 = y^2 + 7 * y + 6) : x = 6 ∧ y = 3 := 
by
  sorry

end find_nat_int_l165_165639


namespace range_of_m_l165_165879

-- Definitions given in the problem
def p (x : ℝ) : Prop := x < -2 ∨ x > 10
def q (x m : ℝ) : Prop := x^2 - 2*x - (m^2 - 1) ≥ 0
def neg_q_sufficient_for_neg_p : Prop :=
  ∀ {x m : ℝ}, (1 - m < x ∧ x < 1 + m) → (-2 ≤ x ∧ x ≤ 10)

-- The statement to prove
theorem range_of_m (m : ℝ) (h1 : m > 0) (h2 : 1 - m ≥ -2) (h3 : 1 + m ≤ 10) :
  0 < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l165_165879


namespace monkey_climb_time_l165_165714

theorem monkey_climb_time : 
  ∀ (height hop slip : ℕ), 
    height = 22 ∧ hop = 3 ∧ slip = 2 → 
    ∃ (time : ℕ), time = 20 := 
by
  intros height hop slip h
  rcases h with ⟨h_height, ⟨h_hop, h_slip⟩⟩
  sorry

end monkey_climb_time_l165_165714


namespace solve_inequality_l165_165338

open Set

theorem solve_inequality (x : ℝ) (h : -3 * x^2 + 5 * x + 4 < 0 ∧ x > 0) : x ∈ Ioo 0 1 := by
  sorry

end solve_inequality_l165_165338


namespace smallest_x_abs_eq_29_l165_165396

theorem smallest_x_abs_eq_29 : ∃ x: ℝ, |4*x - 5| = 29 ∧ (∀ y: ℝ, |4*y - 5| = 29 → -6 ≤ y) :=
by
  sorry

end smallest_x_abs_eq_29_l165_165396


namespace no_primes_in_Q_plus_m_l165_165830

def Q : ℕ := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_primes_in_Q_plus_m (m : ℕ) (hm : 2 ≤ m ∧ m ≤ 32) : ¬is_prime (Q + m) :=
by
  sorry  -- Proof would be provided here

end no_primes_in_Q_plus_m_l165_165830


namespace solution_exists_l165_165433

theorem solution_exists (x : ℝ) :
  (|x - 10| + |x - 14| = |2 * x - 24|) ↔ (x = 12) :=
by
  sorry

end solution_exists_l165_165433


namespace ratio_of_q_to_r_l165_165177

theorem ratio_of_q_to_r
  (P Q R : ℕ)
  (h1 : R = 400)
  (h2 : P + Q + R = 1210)
  (h3 : 5 * Q = 4 * P) :
  Q * 10 = R * 9 :=
by
  sorry

end ratio_of_q_to_r_l165_165177


namespace percentage_of_copper_in_second_alloy_l165_165657

theorem percentage_of_copper_in_second_alloy
  (w₁ w₂ w_total : ℝ)
  (p₁ p_total : ℝ)
  (h₁ : w₁ = 66)
  (h₂ : p₁ = 0.10)
  (h₃ : w_total = 121)
  (h₄ : p_total = 0.15) :
  (w_total - w₁) * 0.21 = w_total * p_total - w₁ * p₁ := 
  sorry

end percentage_of_copper_in_second_alloy_l165_165657


namespace greatest_integer_gcd_l165_165075

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l165_165075


namespace fleas_initial_minus_final_l165_165591

theorem fleas_initial_minus_final (F : ℕ) (h : F / 16 = 14) :
  F - 14 = 210 :=
sorry

end fleas_initial_minus_final_l165_165591


namespace find_xy_l165_165641

theorem find_xy (x y : ℝ) (h1 : (x / 6) * 12 = 11) (h2 : 4 * (x - y) + 5 = 11) : 
  x = 5.5 ∧ y = 4 :=
sorry

end find_xy_l165_165641


namespace Roy_height_l165_165205

theorem Roy_height (Sara_height Joe_height Roy_height : ℕ) 
  (h1 : Sara_height = 45)
  (h2 : Sara_height = Joe_height + 6)
  (h3 : Joe_height = Roy_height + 3) :
  Roy_height = 36 :=
by
  sorry

end Roy_height_l165_165205


namespace value_in_box_l165_165478

theorem value_in_box (x : ℤ) (h : 5 + x = 10 + 20) : x = 25 := by
  sorry

end value_in_box_l165_165478


namespace arithmetic_sequence_a4_l165_165955

/-- Given an arithmetic sequence {a_n}, where S₁₀ = 60 and a₇ = 7, prove that a₄ = 5. -/
theorem arithmetic_sequence_a4 (a₁ d : ℝ) 
  (h1 : 10 * a₁ + 45 * d = 60) 
  (h2 : a₁ + 6 * d = 7) : 
  a₁ + 3 * d = 5 :=
  sorry

end arithmetic_sequence_a4_l165_165955


namespace remainder_x1002_div_x2_minus_1_mul_x_plus_1_l165_165424

noncomputable def polynomial_div_remainder (a b : Polynomial ℝ) : Polynomial ℝ := sorry

theorem remainder_x1002_div_x2_minus_1_mul_x_plus_1 :
  polynomial_div_remainder (Polynomial.X ^ 1002) ((Polynomial.X ^ 2 - 1) * (Polynomial.X + 1)) = 1 :=
by sorry

end remainder_x1002_div_x2_minus_1_mul_x_plus_1_l165_165424


namespace range_of_m_l165_165760

theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1 ∧ y = 0) ∧ 
  (∃ x y : ℝ, (x - m + 1)^2 + (y - m)^2 = 1 ∧ x = 0) ↔ 0 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l165_165760


namespace exists_zero_in_interval_l165_165356

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem exists_zero_in_interval : 
  (f 2) * (f 3) < 0 := by
  sorry

end exists_zero_in_interval_l165_165356


namespace total_clothes_donated_l165_165373

theorem total_clothes_donated
  (pants : ℕ) (jumpers : ℕ) (pajama_sets : ℕ) (tshirts : ℕ)
  (friends : ℕ)
  (adam_donation : ℕ)
  (half_adam_donated : ℕ)
  (friends_donation : ℕ)
  (total_donation : ℕ)
  (h1 : pants = 4) 
  (h2 : jumpers = 4) 
  (h3 : pajama_sets = 4 * 2) 
  (h4 : tshirts = 20) 
  (h5 : friends = 3)
  (h6 : adam_donation = pants + jumpers + pajama_sets + tshirts) 
  (h7 : half_adam_donated = adam_donation / 2) 
  (h8 : friends_donation = friends * adam_donation) 
  (h9 : total_donation = friends_donation + half_adam_donated) :
  total_donation = 126 :=
by
  sorry

end total_clothes_donated_l165_165373


namespace constant_term_expanded_eq_neg12_l165_165735

theorem constant_term_expanded_eq_neg12
  (a w c d : ℤ)
  (h_eq : (a * x + w) * (c * x + d) = 6 * x ^ 2 + x - 12)
  (h_abs_sum : abs a + abs w + abs c + abs d = 12) :
  w * d = -12 := by
  sorry

end constant_term_expanded_eq_neg12_l165_165735


namespace sequence_general_term_l165_165367

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 0 else 2 * n - 4

def S (n : ℕ) : ℤ :=
  n ^ 2 - 3 * n + 2

theorem sequence_general_term (n : ℕ) : a n = 
  if n = 1 then S n 
  else S n - S (n - 1) := by
  sorry

end sequence_general_term_l165_165367


namespace cook_remaining_potatoes_l165_165871

def total_time_to_cook_remaining_potatoes (total_potatoes cooked_potatoes time_per_potato : ℕ) : ℕ :=
  (total_potatoes - cooked_potatoes) * time_per_potato

theorem cook_remaining_potatoes 
  (total_potatoes cooked_potatoes time_per_potato : ℕ) 
  (h_total_potatoes : total_potatoes = 13)
  (h_cooked_potatoes : cooked_potatoes = 5)
  (h_time_per_potato : time_per_potato = 6) : 
  total_time_to_cook_remaining_potatoes total_potatoes cooked_potatoes time_per_potato = 48 :=
by
  -- Proof not required
  sorry

end cook_remaining_potatoes_l165_165871


namespace ratio_of_x_and_y_l165_165969

theorem ratio_of_x_and_y (x y : ℤ) (h : (3 * x - 2 * y) * 4 = 3 * (2 * x + y)) : (x : ℚ) / y = 11 / 6 :=
  sorry

end ratio_of_x_and_y_l165_165969


namespace Olivia_paint_area_l165_165114

theorem Olivia_paint_area
  (length width height : ℕ) (door_window_area : ℕ) (bedrooms : ℕ)
  (h_length : length = 14) 
  (h_width : width = 11) 
  (h_height : height = 9) 
  (h_door_window_area : door_window_area = 70) 
  (h_bedrooms : bedrooms = 4) :
  (2 * (length * height) + 2 * (width * height) - door_window_area) * bedrooms = 1520 :=
by
  sorry

end Olivia_paint_area_l165_165114


namespace sum_of_fourth_powers_eq_174_fourth_l165_165645

theorem sum_of_fourth_powers_eq_174_fourth :
  120 ^ 4 + 97 ^ 4 + 84 ^ 4 + 27 ^ 4 = 174 ^ 4 :=
by
  sorry

end sum_of_fourth_powers_eq_174_fourth_l165_165645


namespace distribute_a_eq_l165_165191

variable (a b c : ℝ)

theorem distribute_a_eq : a * (a + b - c) = a^2 + a * b - a * c := 
sorry

end distribute_a_eq_l165_165191


namespace divisibility_by_n_l165_165362

variable (a b c : ℤ) (n : ℕ)

theorem divisibility_by_n
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2 * n + 1) :
  ∃ k : ℤ, a^3 + b^2 - a^2 - b^3 = k * ↑n := 
sorry

end divisibility_by_n_l165_165362


namespace ratio_of_product_of_composites_l165_165024

theorem ratio_of_product_of_composites :
  let A := [4, 6, 8, 9, 10, 12]
  let B := [14, 15, 16, 18, 20, 21]
  (A.foldl (λ x y => x * y) 1) / (B.foldl (λ x y => x * y) 1) = 1 / 49 :=
by
  -- Proof will be filled here
  sorry

end ratio_of_product_of_composites_l165_165024


namespace metal_waste_l165_165952

theorem metal_waste (l w : ℝ) (h : l > w) :
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  wasted_metal = l * w - w ^ 2 / 2 :=
by
  let area_rectangle := l * w
  let area_circle := Real.pi * (w / 2) ^ 2
  let area_square := (w / Real.sqrt 2) ^ 2
  let wasted_metal := area_rectangle - area_circle + area_circle - area_square
  sorry

end metal_waste_l165_165952


namespace degree_of_d_l165_165613

theorem degree_of_d (f d q r : Polynomial ℝ) (f_deg : f.degree = 17)
  (q_deg : q.degree = 10) (r_deg : r.degree = 4) 
  (remainder : r = Polynomial.C 5 * X^4 - Polynomial.C 3 * X^3 + Polynomial.C 2 * X^2 - X + 15)
  (div_relation : f = d * q + r) (r_deg_lt_d_deg : r.degree < d.degree) :
  d.degree = 7 :=
sorry

end degree_of_d_l165_165613


namespace boy_travel_speed_l165_165444

theorem boy_travel_speed 
  (v : ℝ)
  (travel_distance : ℝ := 10) 
  (return_speed : ℝ := 2) 
  (total_time : ℝ := 5.8)
  (distance : ℝ := 9.999999999999998) :
  (v = 12.5) → (travel_distance = distance) →
  (total_time = (travel_distance / v) + (travel_distance / return_speed)) :=
by
  sorry

end boy_travel_speed_l165_165444


namespace percentage_difference_l165_165116

variables (G P R : ℝ)

-- Conditions
def condition1 : Prop := P = 0.9 * G
def condition2 : Prop := R = 3.0000000000000006 * G

-- Theorem to prove
theorem percentage_difference (h1 : condition1 P G) (h2 : condition2 R G) : 
  (R - P) / R * 100 = 70 :=
sorry

end percentage_difference_l165_165116


namespace shifted_line_does_not_pass_third_quadrant_l165_165789

def line_eq (x: ℝ) : ℝ := -2 * x - 1
def shifted_line_eq (x: ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_third_quadrant :
  ¬∃ x y : ℝ, shifted_line_eq x = y ∧ x < 0 ∧ y < 0 :=
sorry

end shifted_line_does_not_pass_third_quadrant_l165_165789


namespace range_of_m_l165_165560

theorem range_of_m (m : ℝ) : 
    (∀ x : ℝ, mx^2 - 6 * m * x + m + 8 ≥ 0) ↔ (0 ≤ m ∧ m ≤ 1) :=
sorry

end range_of_m_l165_165560


namespace sin_half_angle_l165_165532

theorem sin_half_angle 
  (θ : ℝ) 
  (h_cos : |Real.cos θ| = 1 / 5) 
  (h_theta : 5 * Real.pi / 2 < θ ∧ θ < 3 * Real.pi)
  : Real.sin (θ / 2) = - (Real.sqrt 15) / 5 := 
by
  sorry

end sin_half_angle_l165_165532


namespace value_of_1_plus_i_cubed_l165_165376

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- The main statement to verify
theorem value_of_1_plus_i_cubed : (1 + i ^ 3) = (1 - i) :=
by {  
  -- Use given conditions here if needed
  sorry
}

end value_of_1_plus_i_cubed_l165_165376


namespace find_sum_x1_x2_l165_165978

-- Define sets A and B with given properties
def set_A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}
def set_B (x1 x2 : ℝ) : Set ℝ := {x | x1 ≤ x ∧ x ≤ x2}

-- Conditions of union and intersection
def union_condition (x1 x2 : ℝ) : Prop := set_A ∪ set_B x1 x2 = {x | x > -2}
def intersection_condition (x1 x2 : ℝ) : Prop := set_A ∩ set_B x1 x2 = {x | 1 < x ∧ x ≤ 3}

-- Main theorem to prove
theorem find_sum_x1_x2 (x1 x2 : ℝ) (h_union : union_condition x1 x2) (h_intersect : intersection_condition x1 x2) :
  x1 + x2 = 2 :=
sorry

end find_sum_x1_x2_l165_165978


namespace mixing_ratios_l165_165320

theorem mixing_ratios (V : ℝ) (hV : 0 < V) :
  (4 * V / 5 + 7 * V / 10) / (V / 5 + 3 * V / 10) = 3 :=
by
  sorry

end mixing_ratios_l165_165320


namespace remaining_laps_l165_165691

theorem remaining_laps (total_laps_friday : ℕ)
                       (total_laps_saturday : ℕ)
                       (laps_sunday_morning : ℕ)
                       (total_required_laps : ℕ)
                       (total_laps_weekend : ℕ)
                       (remaining_laps : ℕ) :
  total_laps_friday = 63 →
  total_laps_saturday = 62 →
  laps_sunday_morning = 15 →
  total_required_laps = 198 →
  total_laps_weekend = total_laps_friday + total_laps_saturday + laps_sunday_morning →
  remaining_laps = total_required_laps - total_laps_weekend →
  remaining_laps = 58 := by
  intros
  sorry

end remaining_laps_l165_165691


namespace range_of_y_given_x_l165_165012

theorem range_of_y_given_x (x : ℝ) (h₁ : x > 3) : 0 < (6 / x) ∧ (6 / x) < 2 :=
by 
  sorry

end range_of_y_given_x_l165_165012


namespace count_possible_values_l165_165359

open Nat

def distinct_digits (A B C D : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

def is_valid_addition (A B C D : ℕ) : Prop :=
  ∀ x y z w v u : ℕ, 
  (x = A) ∧ (y = B) ∧ (z = C) ∧ (w = D) ∧ (v = B) ∧ (u = D) →
  (A + C = D) ∧ (A + D = B) ∧ (B + B = D) ∧ (D + D = C)

theorem count_possible_values : ∀ (A B C D : ℕ), 
  distinct_digits A B C D → is_valid_addition A B C D → num_of_possible_D = 4 :=
by
  intro A B C D hd hv
  sorry

end count_possible_values_l165_165359


namespace friends_count_l165_165238

variables (F : ℕ)
def cindy_initial_marbles : ℕ := 500
def marbles_per_friend : ℕ := 80
def marbles_given : ℕ := F * marbles_per_friend
def marbles_remaining := cindy_initial_marbles - marbles_given

theorem friends_count (h : 4 * marbles_remaining = 720) : F = 4 :=
by sorry

end friends_count_l165_165238


namespace sophie_total_spend_l165_165023

-- Definitions based on conditions
def cost_cupcakes : ℕ := 5 * 2
def cost_doughnuts : ℕ := 6 * 1
def cost_apple_pie : ℕ := 4 * 2
def cost_cookies : ℕ := 15 * 6 / 10 -- since 0.60 = 6/10

-- Total cost
def total_cost : ℕ := cost_cupcakes + cost_doughnuts + cost_apple_pie + cost_cookies

-- Prove the total cost
theorem sophie_total_spend : total_cost = 33 := by
  sorry

end sophie_total_spend_l165_165023


namespace solve_for_x_l165_165600

theorem solve_for_x (x : ℤ) (h : 45 - (5 * 3) = x + 7) : x = 23 := 
by
  sorry

end solve_for_x_l165_165600


namespace express_set_l165_165336

open Set

/-- Define the set of natural numbers for which an expression is also a natural number. -/
theorem express_set : {x : ℕ | ∃ y : ℕ, 6 = y * (5 - x)} = {2, 3, 4} :=
by
  sorry

end express_set_l165_165336


namespace problem_series_sum_l165_165135

noncomputable def series_sum : ℝ := ∑' n : ℕ, (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem problem_series_sum :
  series_sum = 1 / 200 :=
sorry

end problem_series_sum_l165_165135


namespace min_value_expression_l165_165021

theorem min_value_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) * (1 / x + 4 / y) ≥ 9 :=
sorry

end min_value_expression_l165_165021


namespace solve_for_nabla_l165_165029

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 :=
by
  sorry

end solve_for_nabla_l165_165029


namespace max_magnitude_vector_sub_l165_165306

open Real

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
sqrt (v.1^2 + v.2^2)

noncomputable def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem max_magnitude_vector_sub (a b : ℝ × ℝ)
  (ha : vector_magnitude a = 2)
  (hb : vector_magnitude b = 1) :
  ∃ θ : ℝ, |vector_magnitude (vector_sub a b)| = 3 :=
by
  use π  -- θ = π to minimize cos θ to be -1
  sorry

end max_magnitude_vector_sub_l165_165306


namespace perfect_square_representation_l165_165065

theorem perfect_square_representation :
  29 - 12*Real.sqrt 5 = (2*Real.sqrt 5 - 3*Real.sqrt 5 / 5)^2 :=
sorry

end perfect_square_representation_l165_165065


namespace boiling_temperature_l165_165374

-- Definitions according to conditions
def initial_temperature : ℕ := 41

def temperature_increase_per_minute : ℕ := 3

def pasta_cooking_time : ℕ := 12

def mixing_and_salad_time : ℕ := pasta_cooking_time / 3

def total_evening_time : ℕ := 73

-- Conditions and the problem statement in Lean
theorem boiling_temperature :
  initial_temperature + (total_evening_time - (pasta_cooking_time + mixing_and_salad_time)) * temperature_increase_per_minute = 212 :=
by
  -- Here would be the proof, skipped with sorry
  sorry

end boiling_temperature_l165_165374


namespace remainder_3_pow_2023_mod_5_l165_165682

theorem remainder_3_pow_2023_mod_5 : (3 ^ 2023) % 5 = 2 := by
  sorry

end remainder_3_pow_2023_mod_5_l165_165682


namespace problem_a4_inv_a4_l165_165601

theorem problem_a4_inv_a4 (a : ℝ) (h : (a + 1/a)^4 = 16) : (a^4 + 1/a^4) = 2 := 
by 
  sorry

end problem_a4_inv_a4_l165_165601


namespace sum_of_roots_is_zero_l165_165720

-- Definitions
def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- Problem Statement
theorem sum_of_roots_is_zero (f : ℝ → ℝ) (h_even : is_even f) (h_intersects : ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) : 
  x1 + x2 + x3 + x4 = 0 :=
by 
  sorry -- Proof can be provided here

end sum_of_roots_is_zero_l165_165720


namespace length_of_segment_l165_165195

theorem length_of_segment : ∃ (a b : ℝ), (|a - (16 : ℝ)^(1/5)| = 3) ∧ (|b - (16 : ℝ)^(1/5)| = 3) ∧ abs (a - b) = 6 :=
by
  sorry

end length_of_segment_l165_165195


namespace greatest_valid_number_l165_165252

-- Define the conditions
def is_valid_number (n : ℕ) : Prop :=
  n < 200 ∧ Nat.gcd n 30 = 5

-- Formulate the proof problem
theorem greatest_valid_number : ∃ n, is_valid_number n ∧ (∀ m, is_valid_number m → m ≤ n) ∧ n = 185 := 
by
  sorry

end greatest_valid_number_l165_165252


namespace tan_alpha_eq_one_then_expr_value_l165_165038

theorem tan_alpha_eq_one_then_expr_value (α : ℝ) (h : Real.tan α = 1) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 2 / 3 :=
by
  sorry

end tan_alpha_eq_one_then_expr_value_l165_165038


namespace pounds_of_fudge_sold_l165_165462

variable (F : ℝ)
variable (price_fudge price_truffles price_pretzels total_revenue : ℝ)

def conditions := 
  price_fudge = 2.50 ∧
  price_truffles = 60 * 1.50 ∧
  price_pretzels = 36 * 2.00 ∧
  total_revenue = 212 ∧
  total_revenue = (price_fudge * F) + price_truffles + price_pretzels

theorem pounds_of_fudge_sold (F : ℝ) (price_fudge price_truffles price_pretzels total_revenue : ℝ) 
  (h : conditions F price_fudge price_truffles price_pretzels total_revenue ) :
  F = 20 :=
by
  sorry

end pounds_of_fudge_sold_l165_165462


namespace batch_production_equation_l165_165625

theorem batch_production_equation (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 20) :
  (500 / x) = (300 / (x - 20)) :=
sorry

end batch_production_equation_l165_165625


namespace number_of_tables_l165_165890

/-- Problem Statement
  In a hall used for a conference, each table is surrounded by 8 stools and 4 chairs. Each stool has 3 legs,
  each chair has 4 legs, and each table has 4 legs. If the total number of legs for all tables, stools, and chairs is 704,
  the number of tables in the hall is 16. -/
theorem number_of_tables (legs_per_stool legs_per_chair legs_per_table total_legs t : ℕ) 
  (Hstools : ∀ tables, stools = 8 * tables)
  (Hchairs : ∀ tables, chairs = 4 * tables)
  (Hlegs : 3 * stools + 4 * chairs + 4 * t = total_legs)
  (Hleg_values : legs_per_stool = 3 ∧ legs_per_chair = 4 ∧ legs_per_table = 4)
  (Htotal_legs : total_legs = 704) :
  t = 16 := by
  sorry

end number_of_tables_l165_165890


namespace martin_boxes_l165_165120

theorem martin_boxes (total_crayons : ℕ) (crayons_per_box : ℕ) (number_of_boxes : ℕ) 
  (h1 : total_crayons = 56) (h2 : crayons_per_box = 7) 
  (h3 : total_crayons = crayons_per_box * number_of_boxes) : 
  number_of_boxes = 8 :=
by 
  sorry

end martin_boxes_l165_165120


namespace find_prime_pair_l165_165621

-- Definition of the problem
def is_integral_expression (p q : ℕ) : Prop :=
  (p + q)^(p + q) * (p - q)^(p - q) - 1 ≠ 0 ∧
  (p + q)^(p - q) * (p - q)^(p + q) - 1 ≠ 0 ∧
  ((p + q)^(p + q) * (p - q)^(p - q) - 1) % ((p + q)^(p - q) * (p - q)^(p + q) - 1) = 0

-- Mathematical theorem to be proved
theorem find_prime_pair (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (h : p > q) :
  is_integral_expression p q → (p, q) = (3, 2) :=
by 
  sorry

end find_prime_pair_l165_165621


namespace carnival_candies_l165_165380

theorem carnival_candies :
  ∃ (c : ℕ), c % 5 = 4 ∧ c % 6 = 3 ∧ c % 8 = 5 ∧ c < 150 ∧ c = 69 :=
by
  sorry

end carnival_candies_l165_165380


namespace number_of_keyboards_l165_165411

-- Definitions based on conditions
def keyboard_cost : ℕ := 20
def printer_cost : ℕ := 70
def printers_bought : ℕ := 25
def total_cost : ℕ := 2050

-- The variable we want to prove
variable (K : ℕ)

-- The main theorem statement
theorem number_of_keyboards (K : ℕ) (keyboard_cost printer_cost printers_bought total_cost : ℕ) :
  keyboard_cost * K + printer_cost * printers_bought = total_cost → K = 15 :=
by
  -- Placeholder for the proof
  sorry

end number_of_keyboards_l165_165411


namespace arithmetic_progression_root_difference_l165_165908

theorem arithmetic_progression_root_difference (a b c : ℚ) (h : 81 * a * a * a - 225 * a * a + 164 * a - 30 = 0)
  (hb : b = 5/3) (hprog : ∃ d : ℚ, a = b - d ∧ c = b + d) :
  c - a = 5 / 9 :=
sorry

end arithmetic_progression_root_difference_l165_165908


namespace arithmetic_sequence_problem_l165_165667

variable {a b : ℕ → ℕ}
variable (S T : ℕ → ℕ)

-- Conditions
def condition (n : ℕ) : Prop :=
  S n / T n = (2 * n + 1) / (3 * n + 2)

-- Conjecture to prove
theorem arithmetic_sequence_problem (h : ∀ n, condition S T n) :
  (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
by
  sorry

end arithmetic_sequence_problem_l165_165667


namespace ordered_triple_solution_l165_165122

theorem ordered_triple_solution (a b c : ℝ) (h1 : a > 5) (h2 : b > 5) (h3 : c > 5)
  (h4 : (a + 3) * (a + 3) / (b + c - 5) + (b + 5) * (b + 5) / (c + a - 7) + (c + 7) * (c + 7) / (a + b - 9) = 49) :
  (a, b, c) = (13, 9, 6) :=
sorry

end ordered_triple_solution_l165_165122


namespace add_complex_eq_required_complex_addition_l165_165942

theorem add_complex_eq (a b c d : ℝ) (i : ℂ) (h : i ^ 2 = -1) :
  (a + b * i) + (c + d * i) = (a + c) + (b + d) * i :=
by sorry

theorem required_complex_addition :
  let a : ℂ := 5 - 3 * i
  let b : ℂ := 2 + 12 * i
  a + b = 7 + 9 * i := 
by sorry

end add_complex_eq_required_complex_addition_l165_165942


namespace sum_of_possible_values_of_N_l165_165887

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l165_165887


namespace proof_problem_l165_165271

-- Definitions of the conditions
def domain_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, true

def symmetric_graph_pt (f : ℝ → ℝ) (a : ℝ) (b : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = 2 * b - f (a + x)

def symmetric (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = -f (x)

def symmetric_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (2*a - x) = f (x)

-- Definitions of the statements to prove
def statement_1 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (y = f (x - 1) → y = f (1 - x) → x = 1)

def statement_2 (f : ℝ → ℝ) : Prop :=
  symmetric_line f (3 / 2)

def statement_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -f (x)

-- Main proof problem
theorem proof_problem (f : ℝ → ℝ) 
  (h_domain : domain_R f)
  (h_symmetric_pt : symmetric_graph_pt f (-3 / 4) 0)
  (h_symmetric : ∀ x : ℝ, f (x + 3 / 2) = -f (x))
  (h_property : ∀ x : ℝ, f (x + 2) = -f (-x + 4)) :
  statement_1 f ∧ statement_2 f ∧ statement_3 f :=
sorry

end proof_problem_l165_165271


namespace vehicle_wax_initial_amount_l165_165157

theorem vehicle_wax_initial_amount
  (wax_car wax_suv wax_spilled wax_left original_amount : ℕ)
  (h_wax_car : wax_car = 3)
  (h_wax_suv : wax_suv = 4)
  (h_wax_spilled : wax_spilled = 2)
  (h_wax_left : wax_left = 2)
  (h_total_wax_used : wax_car + wax_suv = 7)
  (h_wax_before_waxing : wax_car + wax_suv + wax_spilled = 9) :
  original_amount = 11 := by
  sorry

end vehicle_wax_initial_amount_l165_165157


namespace solve_quadratic_equation_l165_165778

theorem solve_quadratic_equation :
  ∀ (x : ℝ), ((x - 2) * (x + 3) = 0) ↔ (x = 2 ∨ x = -3) :=
by
  intro x
  sorry

end solve_quadratic_equation_l165_165778


namespace charles_cleaning_time_l165_165218

theorem charles_cleaning_time :
  let Alice_time := 20
  let Bob_time := (3/4) * Alice_time
  let Charles_time := (2/3) * Bob_time
  Charles_time = 10 :=
by
  sorry

end charles_cleaning_time_l165_165218


namespace greatest_integer_radius_l165_165015

theorem greatest_integer_radius (r : ℕ) :
  (π * (r: ℝ)^2 < 30 * π) ∧ (2 * π * (r: ℝ) > 10 * π) → r = 5 :=
by
  sorry

end greatest_integer_radius_l165_165015


namespace income_in_scientific_notation_l165_165322

theorem income_in_scientific_notation :
  10870 = 1.087 * 10^4 := 
sorry

end income_in_scientific_notation_l165_165322


namespace intersection_of_asymptotes_l165_165906

theorem intersection_of_asymptotes :
  ∃ x y : ℝ, (y = 1) ∧ (x = 3) ∧ (y = (x^2 - 6*x + 8) / (x^2 - 6*x + 9)) := 
by {
  sorry
}

end intersection_of_asymptotes_l165_165906


namespace arithmetic_sequence_a1_geometric_sequence_sum_l165_165264

-- Definition of the arithmetic sequence problem
theorem arithmetic_sequence_a1 (a_n s_n : ℕ) (d : ℕ) (h1 : a_n = 32) (h2 : s_n = 63) (h3 : d = 11) :
  ∃ a_1 : ℕ, a_1 = 10 :=
by
  sorry

-- Definition of the geometric sequence problem
theorem geometric_sequence_sum (a_1 q : ℕ) (h1 : a_1 = 1) (h2 : q = 2) (m : ℕ) :
  let a_m := a_1 * (q ^ (m - 1))
  let a_m_sq := a_m * a_m
  let sm'_sum := (1 - 4^m) / (1 - 4)
  sm'_sum = (4^m - 1) / 3 :=
by
  sorry

end arithmetic_sequence_a1_geometric_sequence_sum_l165_165264


namespace count_integers_with_block_178_l165_165104

theorem count_integers_with_block_178 (a b : ℕ) : 10000 ≤ a ∧ a < 100000 → 10000 ≤ b ∧ b < 100000 → a = b → b - a = 99999 → ∃ n, n = 280 ∧ (n = a + b) := sorry

end count_integers_with_block_178_l165_165104


namespace expression_eq_one_if_and_only_if_k_eq_one_l165_165529

noncomputable def expression (a b c k : ℝ) :=
  (k * a^2 * b^2 + a^2 * c^2 + b^2 * c^2) /
  ((a^2 - b * c) * (b^2 - a * c) + (a^2 - b * c) * (c^2 - a * b) + (b^2 - a * c) * (c^2 - a * b))

theorem expression_eq_one_if_and_only_if_k_eq_one
  (a b c k : ℝ) (h : a + b + c = 0) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0) :
  expression a b c k = 1 ↔ k = 1 :=
by
  sorry

end expression_eq_one_if_and_only_if_k_eq_one_l165_165529


namespace smallest_N_sum_of_digits_eq_six_l165_165463

def bernardo_wins (N : ℕ) : Prop :=
  let b1 := 3 * N
  let s1 := b1 - 30
  let b2 := 3 * s1
  let s2 := b2 - 30
  let b3 := 3 * s2
  let s3 := b3 - 30
  let b4 := 3 * s3
  b4 < 800

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else sum_of_digits (n / 10) + (n % 10)

theorem smallest_N_sum_of_digits_eq_six :
  ∃ N : ℕ, bernardo_wins N ∧ sum_of_digits N = 6 :=
by
  sorry

end smallest_N_sum_of_digits_eq_six_l165_165463


namespace evaluate_expression_l165_165284

theorem evaluate_expression : 15^2 + 2 * 15 * 3 + 3^2 = 324 := by
  sorry

end evaluate_expression_l165_165284


namespace remainder_2011_2015_mod_17_l165_165394

theorem remainder_2011_2015_mod_17 :
  ((2011 * 2012 * 2013 * 2014 * 2015) % 17) = 7 :=
by
  have h1 : 2011 % 17 = 5 := by sorry
  have h2 : 2012 % 17 = 6 := by sorry
  have h3 : 2013 % 17 = 7 := by sorry
  have h4 : 2014 % 17 = 8 := by sorry
  have h5 : 2015 % 17 = 9 := by sorry
  sorry

end remainder_2011_2015_mod_17_l165_165394


namespace tan_double_angle_l165_165080

theorem tan_double_angle (α : ℝ) (x y : ℝ) (hxy : y / x = -2) : 
  2 * y / (1 - (y / x)^2) = (4 : ℝ) / 3 :=
by sorry

end tan_double_angle_l165_165080


namespace find_other_number_l165_165623

-- Define the conditions and the theorem
theorem find_other_number (hcf lcm a b : ℕ) (hcf_def : hcf = 20) (lcm_def : lcm = 396) (a_def : a = 36) (rel : hcf * lcm = a * b) : b = 220 :=
by 
  sorry -- Proof to be provided

end find_other_number_l165_165623


namespace solve_for_x_l165_165366

theorem solve_for_x (x : ℝ) (h_pos : 0 < x) (h : (x / 100) * (x ^ 2) = 9) : x = 10 * (3 ^ (1 / 3)) :=
by
  sorry

end solve_for_x_l165_165366


namespace sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l165_165124

variable (m : ℝ)

def vector_a : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (4, -2)

def perp_vectors (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

theorem sufficient_not_necessary_condition :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) ↔ (m = 1 ∨ m = -3) :=
by
  sorry

theorem m_eq_1_sufficient :
  (m = 1) → perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) :=
by
  sorry

theorem m_eq_1_not_necessary :
  perp_vectors (vector_a m) ((vector_a m).1 - (vector_b).1, (vector_a m).2 - (vector_b).2) → (m = 1 ∨ m = -3) :=
by
  sorry

end sufficient_not_necessary_condition_m_eq_1_sufficient_m_eq_1_not_necessary_l165_165124


namespace tangent_parallel_to_line_l165_165342

theorem tangent_parallel_to_line (x y : ℝ) :
  (y = x^3 + x - 1) ∧ (3 * x^2 + 1 = 4) → (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -3) := by
  sorry

end tangent_parallel_to_line_l165_165342


namespace max_mondays_in_59_days_l165_165184

theorem max_mondays_in_59_days (start_day : ℕ) : ∃ d : ℕ, d ≤ 6 ∧ 
  start_day = d → (d = 0 → ∃ m : ℕ, m = 9) :=
by 
  sorry

end max_mondays_in_59_days_l165_165184


namespace contractor_days_l165_165620

def days_engaged (days_worked days_absent : ℕ) (earnings_per_day : ℝ) (fine_per_absent_day : ℝ) : ℝ :=
  earnings_per_day * days_worked - fine_per_absent_day * days_absent

theorem contractor_days
  (days_absent : ℕ)
  (earnings_per_day : ℝ)
  (fine_per_absent_day : ℝ)
  (total_amount : ℝ)
  (days_worked : ℕ)
  (h1 : days_absent = 12)
  (h2 : earnings_per_day = 25)
  (h3 : fine_per_absent_day = 7.50)
  (h4 : total_amount = 360)
  (h5 : days_engaged days_worked days_absent earnings_per_day fine_per_absent_day = total_amount) :
  days_worked = 18 :=
by sorry

end contractor_days_l165_165620


namespace gcd_number_between_75_and_90_is_5_l165_165032

theorem gcd_number_between_75_and_90_is_5 :
  ∃ n : ℕ, 75 ≤ n ∧ n ≤ 90 ∧ Nat.gcd 15 n = 5 :=
sorry

end gcd_number_between_75_and_90_is_5_l165_165032


namespace anthony_pencils_total_l165_165689

def pencils_initial : Nat := 9
def pencils_kathryn : Nat := 56
def pencils_greg : Nat := 84
def pencils_maria : Nat := 138

theorem anthony_pencils_total : 
  pencils_initial + pencils_kathryn + pencils_greg + pencils_maria = 287 := 
by
  sorry

end anthony_pencils_total_l165_165689


namespace sum_of_xi_l165_165324

theorem sum_of_xi {x1 x2 x3 x4 : ℝ} (h1: (x1 - 3) * Real.sin (π * x1) = 1)
  (h2: (x2 - 3) * Real.sin (π * x2) = 1)
  (h3: (x3 - 3) * Real.sin (π * x3) = 1)
  (h4: (x4 - 3) * Real.sin (π * x4) = 1)
  (hx1 : x1 > 0) (hx2: x2 > 0) (hx3 : x3 > 0) (hx4: x4 > 0) :
  x1 + x2 + x3 + x4 = 12 :=
by
  sorry

end sum_of_xi_l165_165324


namespace solve_cubic_equation_l165_165679

variable (t : ℝ)

theorem solve_cubic_equation (x : ℝ) :
  x^3 - 2 * t * x^2 + t^3 = 0 ↔ 
  x = t ∨ x = t * (1 + Real.sqrt 5) / 2 ∨ x = t * (1 - Real.sqrt 5) / 2 :=
sorry

end solve_cubic_equation_l165_165679


namespace math_problem_l165_165629

-- Define the main variables a and b
def a : ℕ := 312
def b : ℕ := 288

-- State the main theorem to be proved
theorem math_problem : (a^2 - b^2) / 24 + 50 = 650 := 
by 
  sorry

end math_problem_l165_165629


namespace expr_value_l165_165097

-- Define the constants
def w : ℤ := 3
def x : ℤ := -2
def y : ℤ := 1
def z : ℤ := 4

-- Define the expression
def expr : ℤ := (w^2 * x^2 * y * z) - (w * x^2 * y * z^2) + (w * y^3 * z^2) - (w * y^2 * x * z^4)

-- Statement to be proved
theorem expr_value : expr = 1536 :=
by
  -- Proof is omitted, so we use sorry.
  sorry

end expr_value_l165_165097


namespace equilateral_triangle_of_altitude_sum_l165_165780

def triangle (a b c : ℝ) : Prop := 
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def altitude (a b c : ℝ) (S : ℝ) : ℝ := 
  2 * S / a

noncomputable def inradius (S : ℝ) (s : ℝ) : ℝ := 
  S / s

def shape_equilateral (a b c : ℝ) : Prop := 
  a = b ∧ b = c

theorem equilateral_triangle_of_altitude_sum (a b c h_a h_b h_c r S s : ℝ) 
  (habc : triangle a b c)
  (ha : h_a = altitude a b c S)
  (hb : h_b = altitude b a c S)
  (hc : h_c = altitude c a b S)
  (hr : r = inradius S s)
  (h_sum : h_a + h_b + h_c = 9 * r)
  (h_area : S = s * r)
  (h_semi : s = (a + b + c) / 2) : 
  shape_equilateral a b c := 
sorry

end equilateral_triangle_of_altitude_sum_l165_165780


namespace positive_difference_of_fraction_results_l165_165125

theorem positive_difference_of_fraction_results :
  let a := 8
  let expr1 := (a ^ 2 - a ^ 2) / a
  let expr2 := (a ^ 2 * a ^ 2) / a
  expr1 = 0 ∧ expr2 = 512 ∧ (expr2 - expr1) = 512 := 
by
  sorry

end positive_difference_of_fraction_results_l165_165125


namespace cos_alpha_minus_pi_l165_165829

theorem cos_alpha_minus_pi (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (α - Real.pi) = -1/6 := 
by
  sorry

end cos_alpha_minus_pi_l165_165829


namespace range_of_a_l165_165211

theorem range_of_a (a : ℝ) : (4 - a < 0) → (a > 4) :=
by
  intros h
  sorry

end range_of_a_l165_165211


namespace shopkeeper_percentage_profit_l165_165514

variable {x : ℝ} -- cost price per kg of apples

theorem shopkeeper_percentage_profit 
  (total_weight : ℝ)
  (first_half_sold_at : ℝ)
  (second_half_sold_at : ℝ)
  (first_half_profit : ℝ)
  (second_half_profit : ℝ)
  (total_cost_price : ℝ)
  (total_selling_price : ℝ)
  (total_profit : ℝ)
  (percentage_profit : ℝ) :
  total_weight = 100 →
  first_half_sold_at = 0.5 * total_weight →
  second_half_sold_at = 0.5 * total_weight →
  first_half_profit = 25 →
  second_half_profit = 30 →
  total_cost_price = x * total_weight →
  total_selling_price = (first_half_sold_at * (1 + first_half_profit / 100) * x) + (second_half_sold_at * (1 + second_half_profit / 100) * x) →
  total_profit = total_selling_price - total_cost_price →
  percentage_profit = (total_profit / total_cost_price) * 100 →
  percentage_profit = 27.5 := by
  sorry

end shopkeeper_percentage_profit_l165_165514


namespace bankers_gain_is_60_l165_165327

def banker's_gain (BD F PV R T : ℝ) : ℝ :=
  let TD := F - PV
  BD - TD

theorem bankers_gain_is_60 (BD F PV R T BG : ℝ) (h₁ : BD = 260) (h₂ : R = 0.10) (h₃ : T = 3)
  (h₄ : F = 260 / 0.3) (h₅ : PV = F / (1 + (R * T))) :
  banker's_gain BD F PV R T = 60 :=
by
  rw [banker's_gain, h₄, h₅]
  -- Further simplifications and exact equality steps would be added here with actual proof steps
  sorry

end bankers_gain_is_60_l165_165327


namespace solve_rational_equation_l165_165749

theorem solve_rational_equation (x : ℝ) (h : x ≠ (2/3)) : 
  (6*x + 4) / (3*x^2 + 6*x - 8) = 3*x / (3*x - 2) ↔ x = -4/3 ∨ x = 3 :=
sorry

end solve_rational_equation_l165_165749


namespace Darren_paints_432_feet_l165_165895

theorem Darren_paints_432_feet (t : ℝ) (h : t = 792) (paint_ratio : ℝ) 
  (h_ratio : paint_ratio = 1.20) : 
  let d := t / (1 + paint_ratio)
  let D := d * paint_ratio
  D = 432 :=
by
  sorry

end Darren_paints_432_feet_l165_165895


namespace intersection_A_B_l165_165634

def A (x : ℝ) : Prop := (x ≥ 2 ∧ x ≠ 3)
def B (x : ℝ) : Prop := (3 ≤ x ∧ x ≤ 5)
def C := {x : ℝ | 3 < x ∧ x ≤ 5}

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = C :=
  by sorry

end intersection_A_B_l165_165634


namespace solution_set_abs_le_one_inteval_l165_165256

theorem solution_set_abs_le_one_inteval (x : ℝ) : |x| ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 :=
by sorry

end solution_set_abs_le_one_inteval_l165_165256


namespace find_k_l165_165107

theorem find_k (k : ℝ) (h : 64 / k = 8) : k = 8 := 
sorry

end find_k_l165_165107


namespace sum_of_squares_of_rates_l165_165822

theorem sum_of_squares_of_rates (c j s : ℕ) (cond1 : 3 * c + 2 * j + 2 * s = 80) (cond2 : 2 * j + 2 * s + 4 * c = 104) : 
  c^2 + j^2 + s^2 = 592 :=
sorry

end sum_of_squares_of_rates_l165_165822


namespace quadratic_root_value_k_l165_165035

theorem quadratic_root_value_k (k : ℝ) :
  (
    ∃ x₁ x₂ : ℝ, x₁ = 3 ∧ x₂ = -4 / 3 ∧
    (∀ x : ℝ, x^2 * k - 8 * x - 18 = 0 ↔ (x = x₁ ∨ x = x₂))
  ) → k = 4.5 :=
by
  sorry

end quadratic_root_value_k_l165_165035


namespace John_more_marbles_than_Ben_l165_165832

theorem John_more_marbles_than_Ben :
  let ben_initial := 18
  let john_initial := 17
  let ben_gave := ben_initial / 2
  let ben_final := ben_initial - ben_gave
  let john_final := john_initial + ben_gave
  john_final - ben_final = 17 :=
by
  sorry

end John_more_marbles_than_Ben_l165_165832


namespace arithmetic_mean_common_difference_l165_165225

theorem arithmetic_mean_common_difference (a : ℕ → ℝ) (d : ℝ) 
    (h1 : ∀ n, a (n + 1) = a n + d) 
    (h2 : a 1 + a 4 = 2 * (a 2 + 1))
    : d = 2 := 
by 
  -- Proof is omitted as it is not required.
  sorry

end arithmetic_mean_common_difference_l165_165225


namespace train_crossing_time_l165_165435

/-- Prove the time it takes for a train of length 50 meters running at 60 km/hr to cross a pole is 3 seconds. -/
theorem train_crossing_time
  (speed_kmh : ℝ)
  (length_m : ℝ)
  (conversion_factor : ℝ)
  (time_seconds : ℝ) :
  speed_kmh = 60 →
  length_m = 50 →
  conversion_factor = 1000 / 3600 →
  time_seconds = 3 →
  time_seconds = length_m / (speed_kmh * conversion_factor) := 
by
  intros
  sorry

end train_crossing_time_l165_165435


namespace systematic_sampling_eighth_group_number_l165_165698

theorem systematic_sampling_eighth_group_number (total_students groups students_per_group draw_lots_first : ℕ) 
  (h_total : total_students = 480)
  (h_groups : groups = 30)
  (h_students_per_group : students_per_group = 16)
  (h_draw_lots_first : draw_lots_first = 5) : 
  (8 - 1) * students_per_group + draw_lots_first = 117 :=
by
  sorry

end systematic_sampling_eighth_group_number_l165_165698


namespace product_of_solutions_eq_zero_l165_165002

theorem product_of_solutions_eq_zero :
  (∀ x : ℝ, (3 * x + 5) / (6 * x + 5) = (5 * x + 4) / (9 * x + 4) → (x = 0 ∨ x = 8 / 3)) →
  0 * (8 / 3) = 0 :=
by
  intro h
  sorry

end product_of_solutions_eq_zero_l165_165002


namespace average_ABC_eq_2A_plus_3_l165_165981

theorem average_ABC_eq_2A_plus_3 (A B C : ℝ) 
  (h1 : 2023 * C - 4046 * A = 8092) 
  (h2 : 2023 * B - 6069 * A = 10115) : 
  (A + B + C) / 3 = 2 * A + 3 :=
sorry

end average_ABC_eq_2A_plus_3_l165_165981


namespace positive_integer_condition_l165_165344

theorem positive_integer_condition (n : ℕ) (h : 15 * n = n^2 + 56) : n = 8 :=
sorry

end positive_integer_condition_l165_165344


namespace total_notebooks_correct_l165_165776

-- Definitions based on conditions
def total_students : ℕ := 28
def half_students : ℕ := total_students / 2
def notebooks_per_student_group1 : ℕ := 5
def notebooks_per_student_group2 : ℕ := 3

-- Total notebooks calculation
def total_notebooks : ℕ :=
  (half_students * notebooks_per_student_group1) + (half_students * notebooks_per_student_group2)

-- Theorem to be proved
theorem total_notebooks_correct : total_notebooks = 112 := by
  sorry

end total_notebooks_correct_l165_165776


namespace farm_entrance_fee_for_students_is_five_l165_165258

theorem farm_entrance_fee_for_students_is_five
  (students : ℕ) (adults : ℕ) (adult_fee : ℕ) (total_cost : ℕ) (student_fee : ℕ)
  (h_students : students = 35)
  (h_adults : adults = 4)
  (h_adult_fee : adult_fee = 6)
  (h_total_cost : total_cost = 199)
  (h_equation : students * student_fee + adults * adult_fee = total_cost) :
  student_fee = 5 :=
by
  sorry

end farm_entrance_fee_for_students_is_five_l165_165258


namespace problem_l165_165283

noncomputable def f (ω x : ℝ) : ℝ := (Real.sin (ω * x / 2))^2 + (1 / 2) * Real.sin (ω * x) - 1 / 2

theorem problem (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, x ∈ Set.Ioo (Real.pi : ℝ) (2 * Real.pi) → f ω x ≠ 0) →
  ω ∈ Set.Icc 0 (1 / 8) ∪ Set.Icc (1 / 4) (5 / 8) :=
by
  sorry

end problem_l165_165283


namespace gcd_9011_2147_l165_165110

theorem gcd_9011_2147 : Int.gcd 9011 2147 = 1 := sorry

end gcd_9011_2147_l165_165110


namespace polynomial_roots_cubed_l165_165556

noncomputable def f (x : ℝ) : ℝ := x^3 - 2*x^2 + 5*x - 3
noncomputable def g (x : ℝ) : ℝ := x^3 - 2*x^2 - 5*x + 3

theorem polynomial_roots_cubed {r : ℝ} (h : f r = 0) :
  g (r^3) = 0 := by
  sorry

end polynomial_roots_cubed_l165_165556


namespace cost_price_bicycle_A_l165_165369

variable {CP_A CP_B SP_C : ℝ}

theorem cost_price_bicycle_A (h1 : CP_B = 1.25 * CP_A) (h2 : SP_C = 1.25 * CP_B) (h3 : SP_C = 225) :
  CP_A = 144 :=
by
  sorry

end cost_price_bicycle_A_l165_165369


namespace machine_does_not_print_13824_l165_165573

-- Definitions corresponding to the conditions:
def machine_property (S : Set ℕ) : Prop :=
  ∀ n ∈ S, (2 * n) ∉ S ∧ (3 * n) ∉ S

def machine_prints_2 (S : Set ℕ) : Prop :=
  2 ∈ S

-- Statement to be proved
theorem machine_does_not_print_13824 (S : Set ℕ) 
  (H1 : machine_property S) 
  (H2 : machine_prints_2 S) : 
  13824 ∉ S :=
sorry

end machine_does_not_print_13824_l165_165573


namespace quadratic_real_roots_a_leq_2_l165_165786

theorem quadratic_real_roots_a_leq_2
    (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - 4*x1 + 2*a = 0) ∧ (x2^2 - 4*x2 + 2*a = 0)) →
    a ≤ 2 :=
by sorry

end quadratic_real_roots_a_leq_2_l165_165786


namespace number_of_ways_to_sign_up_probability_student_A_online_journalists_l165_165275

-- Definitions for the conditions
def students : Finset String := {"A", "B", "C", "D", "E"}
def projects : Finset String := {"Online Journalists", "Robot Action", "Sounds of Music"}

-- Function to calculate combinations (nCr)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate arrangements
def arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Proof opportunity for part 1
theorem number_of_ways_to_sign_up : 
  (combinations 5 3 * arrangements 3) + ((combinations 5 2 * combinations 3 2) / arrangements 2 * arrangements 3) = 150 :=
sorry

-- Proof opportunity for part 2
theorem probability_student_A_online_journalists
  (h : (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 243) : 
  ((combinations 4 3 * arrangements 2) * projects.card ^ 3) / 
  (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 1 / 15 :=
sorry

end number_of_ways_to_sign_up_probability_student_A_online_journalists_l165_165275


namespace inequality_for_any_x_l165_165185

theorem inequality_for_any_x (a : ℝ) (h : ∀ x : ℝ, |3 * x + 2 * a| + |2 - 3 * x| - |a + 1| > 2) :
  a < -1/3 ∨ a > 5 := 
sorry

end inequality_for_any_x_l165_165185


namespace smallest_d_l165_165726

noncomputable def abc_identity_conditions (a b c d e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  ∀ x : ℝ, (x + a) * (x + b) * (x + c) = x^3 + 3 * d * x^2 + 3 * x + e^3

theorem smallest_d (a b c d e : ℝ) (h : abc_identity_conditions a b c d e) : d = 1 := 
sorry

end smallest_d_l165_165726


namespace infinite_primes_of_form_m2_mn_n2_l165_165880

theorem infinite_primes_of_form_m2_mn_n2 : ∀ m n : ℤ, ∃ p : ℕ, ∃ k : ℕ, (p = k^2 + k * m + n^2) ∧ Prime k :=
sorry

end infinite_primes_of_form_m2_mn_n2_l165_165880


namespace problem_solution_l165_165510

theorem problem_solution : ∃ n : ℕ, (n > 0) ∧ (21 - 3 * n > 15) ∧ (∀ m : ℕ, (m > 0) ∧ (21 - 3 * m > 15) → m = n) :=
by
  sorry

end problem_solution_l165_165510


namespace correct_transformation_l165_165757

theorem correct_transformation (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
by
  sorry

end correct_transformation_l165_165757


namespace hyperbola_eccentricity_range_l165_165055

theorem hyperbola_eccentricity_range {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a > b) :
  ∃ e : ℝ, e = (Real.sqrt (a^2 + b^2)) / a ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  -- Proof would go here
  sorry

end hyperbola_eccentricity_range_l165_165055


namespace M_empty_iff_k_range_M_interval_iff_k_range_l165_165018

-- Part 1
theorem M_empty_iff_k_range (k : ℝ) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 ≤ 0) ↔ -3 ≤ k ∧ k ≤ 1 / 5 := sorry

-- Part 2
theorem M_interval_iff_k_range (k a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_ab : a < b) :
  (∀ x : ℝ, (k^2 + 2 * k - 3) * x^2 + (k + 3) * x - 1 > 0 ↔ a < x ∧ x < b) ↔ 1 / 5 < k ∧ k < 1 := sorry

end M_empty_iff_k_range_M_interval_iff_k_range_l165_165018


namespace parallel_lines_count_l165_165614

theorem parallel_lines_count (n : ℕ) (h : 7 * (n - 1) = 588) : n = 85 :=
sorry

end parallel_lines_count_l165_165614


namespace original_contribution_amount_l165_165656

theorem original_contribution_amount (F : ℕ) (N : ℕ) (C : ℕ) (A : ℕ) 
  (hF : F = 14) (hN : N = 19) (hC : C = 4) : A = 90 :=
by 
  sorry

end original_contribution_amount_l165_165656


namespace solve_basketball_court_dimensions_l165_165088

theorem solve_basketball_court_dimensions 
  (A B C D E F : ℕ) 
  (h1 : A - B = C) 
  (h2 : D = 2 * (A + B)) 
  (h3 : E = A * B) 
  (h4 : F = 3) : 
  A = 28 ∧ B = 15 ∧ C = 13 ∧ D = 86 ∧ E = 420 ∧ F = 3 := 
by 
  sorry

end solve_basketball_court_dimensions_l165_165088


namespace father_walk_time_l165_165809

-- Xiaoming's cycling speed is 4 times his father's walking speed.
-- Xiaoming continues for another 18 minutes to reach B after meeting his father.
-- Prove that Xiaoming's father needs 288 minutes to walk from the meeting point to A.
theorem father_walk_time {V : ℝ} (h₁ : V > 0) (h₂ : ∀ t : ℝ, t > 0 → 18 * V = (V / 4) * t) :
  288 = 4 * 72 :=
by
  sorry

end father_walk_time_l165_165809


namespace parking_spots_first_level_l165_165232

theorem parking_spots_first_level (x : ℕ) 
    (h1 : ∃ x, x + (x + 7) + (x + 13) + 14 = 46) : x = 4 :=
by
  sorry

end parking_spots_first_level_l165_165232


namespace michael_regular_hours_l165_165801

-- Define the constants and conditions
def regular_rate : ℝ := 7
def overtime_rate : ℝ := 14
def total_earnings : ℝ := 320
def total_hours : ℝ := 42.857142857142854

-- Declare the proof problem
theorem michael_regular_hours :
  ∃ R O : ℝ, (regular_rate * R + overtime_rate * O = total_earnings) ∧ (R + O = total_hours) ∧ (R = 40) :=
by
  sorry

end michael_regular_hours_l165_165801


namespace tangent_line_circle_m_values_l165_165241

theorem tangent_line_circle_m_values {m : ℝ} :
  (∀ (x y: ℝ), 3 * x + 4 * y + m = 0 → (x - 1)^2 + (y + 2)^2 = 4) →
  (m = 15 ∨ m = -5) :=
by
  sorry

end tangent_line_circle_m_values_l165_165241


namespace number_of_real_roots_of_cubic_l165_165417

-- Define the real number coefficients
variables (a b c d : ℝ)

-- Non-zero condition on coefficients
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)

-- Statement of the problem: The cubic polynomial typically has 3 real roots
theorem number_of_real_roots_of_cubic :
  ∃ (x : ℝ), (x ^ 3 + x * (c ^ 2 - d ^ 2 - b * d) - (b ^ 2) * c = 0) := by
  sorry

end number_of_real_roots_of_cubic_l165_165417


namespace min_shots_for_probability_at_least_075_l165_165460

theorem min_shots_for_probability_at_least_075 (hit_rate : ℝ) (target_probability : ℝ) :
  hit_rate = 0.25 → target_probability = 0.75 → ∃ n : ℕ, n = 4 ∧ (1 - hit_rate)^n ≤ 1 - target_probability := by
  intros h_hit_rate h_target_probability
  sorry

end min_shots_for_probability_at_least_075_l165_165460


namespace find_interest_rate_l165_165905

-- Define the conditions
def total_amount : ℝ := 2500
def second_part_rate : ℝ := 0.06
def annual_income : ℝ := 145
def first_part_amount : ℝ := 500.0000000000002
noncomputable def interest_rate (r : ℝ) : Prop :=
  first_part_amount * r + (total_amount - first_part_amount) * second_part_rate = annual_income

-- State the theorem
theorem find_interest_rate : interest_rate 0.05 :=
by
  sorry

end find_interest_rate_l165_165905


namespace number_in_marked_square_is_10_l165_165914

theorem number_in_marked_square_is_10 : 
  ∃ f : ℕ × ℕ → ℕ, 
    (f (0,0) = 5 ∧ f (0,1) = 6 ∧ f (0,2) = 7) ∧ 
    (∀ r c, r > 0 → 
      f (r,c) = f (r-1,c) + f (r-1,c+1)) 
    ∧ f (1, 1) = 13 
    ∧ f (2, 1) = 10 :=
    sorry

end number_in_marked_square_is_10_l165_165914


namespace distance_between_two_cars_l165_165285

theorem distance_between_two_cars 
    (initial_distance : ℝ) 
    (first_car_distance1 : ℝ) 
    (first_car_distance2 : ℝ)
    (second_car_distance : ℝ) 
    (final_distance : ℝ) :
    initial_distance = 150 →
    first_car_distance1 = 25 →
    first_car_distance2 = 25 →
    second_car_distance = 35 →
    final_distance = initial_distance - (first_car_distance1 + first_car_distance2 + second_car_distance) →
    final_distance = 65 :=
by
  intros h_initial h_first1 h_first2 h_second h_final
  sorry

end distance_between_two_cars_l165_165285


namespace triangle_inequality_l165_165445

theorem triangle_inequality
  (a b c x y z : ℝ)
  (h_order : a < b ∧ b < c ∧ 0 < x)
  (h_area_eq : c * x = a * y + b * z) :
  x < y + z :=
by
  sorry

end triangle_inequality_l165_165445


namespace profit_percentage_mobile_l165_165542

-- Definitions derived from conditions
def cost_price_grinder : ℝ := 15000
def cost_price_mobile : ℝ := 8000
def loss_percentage_grinder : ℝ := 0.05
def total_profit : ℝ := 50
def selling_price_grinder := cost_price_grinder * (1 - loss_percentage_grinder)
def total_cost_price := cost_price_grinder + cost_price_mobile
def total_selling_price := total_cost_price + total_profit
def selling_price_mobile := total_selling_price - selling_price_grinder
def profit_mobile := selling_price_mobile - cost_price_mobile

-- The theorem to prove the profit percentage on the mobile phone is 10%
theorem profit_percentage_mobile : (profit_mobile / cost_price_mobile) * 100 = 10 :=
by
  sorry

end profit_percentage_mobile_l165_165542


namespace honor_students_count_l165_165332

noncomputable def number_of_honor_students (G B Eg Eb : ℕ) (p_girl p_boy : ℚ) : ℕ :=
  if G < 30 ∧ B < 30 ∧ Eg = (3 / 13) * G ∧ Eb = (4 / 11) * B ∧ G + B < 30 then
    Eg + Eb
  else
    0

theorem honor_students_count :
  ∃ (G B Eg Eb : ℕ), (G < 30 ∧ B < 30 ∧ G % 13 = 0 ∧ B % 11 = 0 ∧ Eg = (3 * G / 13) ∧ Eb = (4 * B / 11) ∧ G + B < 30 ∧ number_of_honor_students G B Eg Eb (3 / 13) (4 / 11) = 7) :=
by {
  sorry
}

end honor_students_count_l165_165332


namespace surface_area_of_sphere_with_diameter_two_l165_165835

theorem surface_area_of_sphere_with_diameter_two :
  let diameter := 2
  let radius := diameter / 2
  4 * Real.pi * radius ^ 2 = 4 * Real.pi :=
by
  sorry

end surface_area_of_sphere_with_diameter_two_l165_165835


namespace find_abc_unique_solution_l165_165474

theorem find_abc_unique_solution (N a b c : ℕ) 
  (hN : N > 3 ∧ N % 2 = 1)
  (h_eq : a^N = b^N + 2^N + a * b * c)
  (h_c : c ≤ 5 * 2^(N-1)) : 
  N = 5 ∧ a = 3 ∧ b = 1 ∧ c = 70 := 
sorry

end find_abc_unique_solution_l165_165474


namespace fifty_three_days_from_friday_is_tuesday_l165_165875

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l165_165875


namespace max_value_of_expression_l165_165700

variables (a x1 x2 : ℝ)

theorem max_value_of_expression :
  (x1 < 0) → (0 < x2) → (∀ x, x^2 - a * x + a - 2 > 0 ↔ (x < x1) ∨ (x > x2)) →
  (x1 * x2 = a - 2) → 
  x1 + x2 + 2 / x1 + 2 / x2 ≤ 0 :=
by
  intros h1 h2 h3 h4
  -- Proof goes here
  sorry

end max_value_of_expression_l165_165700


namespace problem_statement_l165_165837

theorem problem_statement 
  (x y z : ℝ)
  (h1 : 5 = 0.25 * x)
  (h2 : 5 = 0.10 * y)
  (h3 : z = 2 * y) :
  x - z = -80 :=
sorry

end problem_statement_l165_165837


namespace train_length_l165_165624

theorem train_length (speed_kmph : ℕ) (time_s : ℕ) (platform_length_m : ℕ) (h1 : speed_kmph = 72) (h2 : time_s = 26) (h3 : platform_length_m = 260) :
  ∃ train_length_m : ℕ, train_length_m = 260 := by
  sorry

end train_length_l165_165624


namespace continuous_stripe_probability_l165_165745

noncomputable def probability_continuous_stripe : ℚ :=
  let total_configurations := 4^6
  let favorable_configurations := 48
  favorable_configurations / total_configurations

theorem continuous_stripe_probability : probability_continuous_stripe = 3 / 256 :=
  by
  sorry

end continuous_stripe_probability_l165_165745


namespace max_shortest_side_decagon_inscribed_circle_l165_165008

noncomputable def shortest_side_decagon : ℝ :=
  2 * Real.sin (36 * Real.pi / 180 / 2)

theorem max_shortest_side_decagon_inscribed_circle :
  shortest_side_decagon = (Real.sqrt 5 - 1) / 2 :=
by {
  -- Proof details here
  sorry
}

end max_shortest_side_decagon_inscribed_circle_l165_165008


namespace zero_point_interval_l165_165234

noncomputable def f (x : ℝ) : ℝ := (4 / x) - (2^x)

theorem zero_point_interval : ∃ x : ℝ, (1 < x ∧ x < 1.5) ∧ f x = 0 :=
sorry

end zero_point_interval_l165_165234


namespace kim_paid_with_amount_l165_165438

-- Define the conditions
def meal_cost : ℝ := 10
def drink_cost : ℝ := 2.5
def tip_rate : ℝ := 0.20
def change_received : ℝ := 5

-- Define the total amount paid formula
def total_cost_before_tip := meal_cost + drink_cost
def tip_amount := tip_rate * total_cost_before_tip
def total_cost_after_tip := total_cost_before_tip + tip_amount
def amount_paid := total_cost_after_tip + change_received

-- Statement of the theorem
theorem kim_paid_with_amount : amount_paid = 20 := by
  sorry

end kim_paid_with_amount_l165_165438


namespace width_of_box_is_correct_l165_165523

noncomputable def length_of_box : ℝ := 62
noncomputable def height_lowered : ℝ := 0.5
noncomputable def volume_removed_in_gallons : ℝ := 5812.5
noncomputable def gallons_to_cubic_feet : ℝ := 1 / 7.48052

theorem width_of_box_is_correct :
  let volume_removed_in_cubic_feet := volume_removed_in_gallons * gallons_to_cubic_feet
  let area_of_base := length_of_box * W
  let needed_volume := area_of_base * height_lowered
  volume_removed_in_cubic_feet = needed_volume →
  W = 25.057 :=
by
  sorry

end width_of_box_is_correct_l165_165523


namespace find_f_of_2_l165_165816

variable (f : ℝ → ℝ)

def functional_equation_condition :=
  ∀ x : ℝ, f (f (f x)) + 3 * f (f x) + 9 * f x + 27 * x = 0

theorem find_f_of_2
  (h : functional_equation_condition f) :
  f (f (f (f 2))) = 162 :=
sorry

end find_f_of_2_l165_165816


namespace P_72_l165_165815

def P (n : ℕ) : ℕ :=
  -- The definition of P(n) should enumerate the ways of expressing n as a product
  -- of integers greater than 1, considering the order of factors.
  sorry

theorem P_72 : P 72 = 17 :=
by
  sorry

end P_72_l165_165815


namespace cost_of_pencil_l165_165484

theorem cost_of_pencil (x y : ℕ) (h1 : 4 * x + 3 * y = 224) (h2 : 2 * x + 5 * y = 154) : y = 12 := 
by
  sorry

end cost_of_pencil_l165_165484


namespace distribute_problems_l165_165416

theorem distribute_problems :
  let n_problems := 7
  let n_friends := 12
  (n_friends ^ n_problems) = 35831808 :=
by 
  sorry

end distribute_problems_l165_165416


namespace birches_planted_l165_165305

variable 
  (G B X : ℕ) -- G: number of girls, B: number of boys, X: number of birches

-- Conditions:
variable
  (h1 : G + B = 24) -- Total number of students
  (h2 : 3 * G + X = 24) -- Total number of plants
  (h3 : X = B / 3) -- Birches planted by boys

-- Proof statement:
theorem birches_planted : X = 6 :=
by 
  sorry

end birches_planted_l165_165305


namespace strictly_positive_integers_equal_l165_165761

theorem strictly_positive_integers_equal 
  (a b : ℤ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (h : (4 * a * b - 1) ∣ (4 * a^2 - 1)^2) : 
  a = b :=
sorry

end strictly_positive_integers_equal_l165_165761


namespace inequality_solution_set_l165_165001

theorem inequality_solution_set (a c x : ℝ) 
  (h1 : -1/3 < x ∧ x < 1/2 → 0 < a * x^2 + 2 * x + c) :
  -2 < x ∧ x < 3 ↔ -c * x^2 + 2 * x - a > 0 :=
by sorry

end inequality_solution_set_l165_165001


namespace area_of_border_l165_165493

theorem area_of_border
  (h_photo : Nat := 9)
  (w_photo : Nat := 12)
  (border_width : Nat := 3) :
  (let area_photo := h_photo * w_photo
    let h_frame := h_photo + 2 * border_width
    let w_frame := w_photo + 2 * border_width
    let area_frame := h_frame * w_frame
    let area_border := area_frame - area_photo
    area_border = 162) := 
  sorry

end area_of_border_l165_165493


namespace pairs_satisfy_condition_l165_165699

theorem pairs_satisfy_condition (a b : ℝ) :
  (∀ n : ℕ, n > 0 → a * (⌊b * n⌋) = b * (⌊a * n⌋)) →
  (a = 0 ∨ b = 0 ∨ a = b ∨ (∃ a_int b_int : ℤ, a = a_int ∧ b = b_int)) :=
by
  sorry

end pairs_satisfy_condition_l165_165699


namespace arithmetic_sequence_sum_l165_165248

variable {a : ℕ → ℤ} 
variable {a_3 a_4 a_5 : ℤ}

-- Hypothesis: arithmetic sequence and given condition
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n, a (n+1) - a n = a 2 - a 1

theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a) (h_sum : a_3 + a_4 + a_5 = 12) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry

end arithmetic_sequence_sum_l165_165248


namespace sum_six_terms_l165_165456

variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (S_2 S_4 S_6 : ℝ)

-- Given conditions
axiom sum_two_terms : S 2 = 4
axiom sum_four_terms : S 4 = 16

-- Problem statement
theorem sum_six_terms : S 6 = 52 :=
by
  -- Insert the proof here
  sorry

end sum_six_terms_l165_165456


namespace incorrect_options_l165_165063

variable (a b : ℚ) (h : a / b = 5 / 6)

theorem incorrect_options :
  (2 * a - b ≠ b * 6 / 4) ∧
  (a + 3 * b ≠ 2 * a * 19 / 10) :=
by
  sorry

end incorrect_options_l165_165063


namespace angle_y_value_l165_165154

theorem angle_y_value (ABC ABD ABE BAE y : ℝ) (h1 : ABC = 180) (h2 : ABD = 66) 
  (h3 : ABE = 114) (h4 : BAE = 31) (h5 : 31 + 114 + y = 180) : y = 35 :=
  sorry

end angle_y_value_l165_165154


namespace range_of_m_l165_165784

theorem range_of_m 
  (m : ℝ)
  (hM : -4 ≤ m ∧ m ≤ 4)
  (ellipse : ∀ (x y : ℝ), x^2 / 16 + y^2 / 12 = 1 → y = 0) :
  1 ≤ m ∧ m ≤ 4 := sorry

end range_of_m_l165_165784


namespace people_per_apartment_l165_165255

/-- A 25 story building has 4 apartments on each floor. 
There are 200 people in the building. 
Prove that each apartment houses 2 people. -/
theorem people_per_apartment (stories : ℕ) (apartments_per_floor : ℕ) (total_people : ℕ)
    (h_stories : stories = 25)
    (h_apartments_per_floor : apartments_per_floor = 4)
    (h_total_people : total_people = 200) :
  (total_people / (stories * apartments_per_floor)) = 2 :=
by
  sorry

end people_per_apartment_l165_165255


namespace find_A_l165_165917

theorem find_A (
  A B C A' r : ℕ
) (hA : A = 312) (hB : B = 270) (hC : C = 211)
  (hremA : A % A' = 4 * r)
  (hremB : B % A' = 2 * r)
  (hremC : C % A' = r) :
  A' = 19 :=
by
  sorry

end find_A_l165_165917


namespace money_left_after_shopping_l165_165527

def initial_budget : ℝ := 999.00
def shoes_price : ℝ := 165.00
def yoga_mat_price : ℝ := 85.00
def sports_watch_price : ℝ := 215.00
def hand_weights_price : ℝ := 60.00
def sales_tax_rate : ℝ := 0.07
def discount_rate : ℝ := 0.10

def total_cost_before_discount : ℝ :=
  shoes_price + yoga_mat_price + sports_watch_price + hand_weights_price

def discount_on_watch : ℝ := sports_watch_price * discount_rate

def discounted_watch_price : ℝ := sports_watch_price - discount_on_watch

def total_cost_after_discount : ℝ :=
  shoes_price + yoga_mat_price + discounted_watch_price + hand_weights_price

def sales_tax : ℝ := total_cost_after_discount * sales_tax_rate

def total_cost_including_tax : ℝ := total_cost_after_discount + sales_tax

def money_left : ℝ := initial_budget - total_cost_including_tax

theorem money_left_after_shopping : 
  money_left = 460.25 :=
by
  sorry

end money_left_after_shopping_l165_165527


namespace ratio_saturday_friday_l165_165617

variable (S : ℕ)
variable (soldOnFriday : ℕ := 30)
variable (soldOnSunday : ℕ := S - 15)
variable (totalSold : ℕ := 135)

theorem ratio_saturday_friday (h1 : soldOnFriday = 30)
                              (h2 : totalSold = 135)
                              (h3 : soldOnSunday = S - 15)
                              (h4 : soldOnFriday + S + soldOnSunday = totalSold) :
  (S / soldOnFriday) = 2 :=
by
  -- Prove the theorem here...
  sorry

end ratio_saturday_friday_l165_165617


namespace exists_triangle_sides_l165_165972

theorem exists_triangle_sides (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h1 : a * b * c ≤ 1 / 4)
  (h2 : 1 / (a^2) + 1 / (b^2) + 1 / (c^2) < 9) : 
  a + b > c ∧ b + c > a ∧ c + a > b := 
by
  sorry

end exists_triangle_sides_l165_165972


namespace smallest_k_for_repeating_representation_l165_165464

theorem smallest_k_for_repeating_representation:
  ∃ k : ℕ, (k > 0) ∧ (∀ m : ℕ, m > 0 → m < k → ¬(97*(5*m + 6) = 11*(m^2 - 1))) ∧ 97*(5*k + 6) = 11*(k^2 - 1) := by
  sorry

end smallest_k_for_repeating_representation_l165_165464


namespace no_valid_pairs_l165_165399

/-- 
Statement: There are no pairs of positive integers (a, b) such that
a * b + 100 = 25 * lcm(a, b) + 15 * gcd(a, b).
-/
theorem no_valid_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) : 
  a * b + 100 ≠ 25 * Nat.lcm a b + 15 * Nat.gcd a b :=
sorry

end no_valid_pairs_l165_165399


namespace area_transformation_l165_165138

variable {g : ℝ → ℝ}

theorem area_transformation (h : ∫ x, g x = 20) : ∫ x, -4 * g (x + 3) = 80 := by
  sorry

end area_transformation_l165_165138


namespace exists_equal_subinterval_l165_165534

open Set Metric Function

variable {a b : ℝ}
variable {f : ℕ → ℝ → ℝ}
variable {n m : ℕ}

-- Define the conditions
def continuous_on_interval (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ n, ContinuousOn (f n) (Icc a b)

def root_cond (f : ℕ → ℝ → ℝ) (a b : ℝ) :=
  ∀ x ∈ Icc a b, ∃ m n, m ≠ n ∧ f m x = f n x

-- The main theorem statement
theorem exists_equal_subinterval (f : ℕ → ℝ → ℝ) (a b : ℝ) 
  (h_cont : continuous_on_interval f a b) 
  (h_root : root_cond f a b) : 
  ∃ (c d : ℝ), c < d ∧ Icc c d ⊆ Icc a b ∧ ∃ m n, m ≠ n ∧ ∀ x ∈ Icc c d, f m x = f n x := 
sorry

end exists_equal_subinterval_l165_165534


namespace expected_greetings_l165_165849

theorem expected_greetings :
  let p1 := 1       -- Probability 1
  let p2 := 0.8     -- Probability 0.8
  let p3 := 0.5     -- Probability 0.5
  let p4 := 0       -- Probability 0
  let n1 := 8       -- Number of colleagues with probability 1
  let n2 := 15      -- Number of colleagues with probability 0.8
  let n3 := 14      -- Number of colleagues with probability 0.5
  let n4 := 3       -- Number of colleagues with probability 0
  p1 * n1 + p2 * n2 + p3 * n3 + p4 * n4 = 27 :=
by
  sorry

end expected_greetings_l165_165849


namespace remainder_when_2n_divided_by_4_l165_165743

theorem remainder_when_2n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (2 * n) % 4 = 2 :=
by
  sorry

end remainder_when_2n_divided_by_4_l165_165743


namespace inradius_of_triangle_l165_165326

theorem inradius_of_triangle (p A r : ℝ) (h1 : p = 20) (h2 : A = 25) : r = 2.5 :=
sorry

end inradius_of_triangle_l165_165326


namespace smallest_integer_for_perfect_square_l165_165127

-- Given condition: y = 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11
def y : ℕ := 2^3 * 3^2 * 4^6 * 5^5 * 7^8 * 8^3 * 9^10 * 11^11

-- The statement to prove
theorem smallest_integer_for_perfect_square (y : ℕ) : ∃ n : ℕ, n = 110 ∧ ∃ m : ℕ, (y * n) = m^2 := 
by {
  sorry
}

end smallest_integer_for_perfect_square_l165_165127


namespace min_value_16_l165_165466

noncomputable def min_value_expr (a b : ℝ) : ℝ :=
  1 / a + 3 / b

theorem min_value_16 (a b : ℝ) (h : a > 0 ∧ b > 0) (h_constraint : a + 3 * b = 1) :
  min_value_expr a b ≥ 16 :=
sorry

end min_value_16_l165_165466


namespace find_c_find_cos_2B_minus_pi_over_4_l165_165077

variable (A B C : Real) (a b c : Real)

-- Given conditions
def conditions (a b c : Real) (A : Real) : Prop :=
  a = 4 * Real.sqrt 3 ∧
  b = 6 ∧
  Real.cos A = -1 / 3

-- Proof of question 1
theorem find_c (h : conditions a b c A) : c = 2 :=
sorry

-- Proof of question 2
theorem find_cos_2B_minus_pi_over_4 (h : conditions a b c A) (B : Real) :
  (angle_opp_b : b = Real.sin B) → -- This is to ensure B is the angle opposite to side b
  Real.cos (2 * B - Real.pi / 4) = (4 - Real.sqrt 2) / 6 :=
sorry

end find_c_find_cos_2B_minus_pi_over_4_l165_165077


namespace ribbons_at_start_l165_165016

theorem ribbons_at_start (morning_ribbons : ℕ) (afternoon_ribbons : ℕ) (left_ribbons : ℕ)
  (h_morning : morning_ribbons = 14) (h_afternoon : afternoon_ribbons = 16) (h_left : left_ribbons = 8) :
  morning_ribbons + afternoon_ribbons + left_ribbons = 38 :=
by
  sorry

end ribbons_at_start_l165_165016


namespace average_other_students_l165_165215

theorem average_other_students (total_students other_students : ℕ) (mean_score_first : ℕ) 
 (mean_score_class : ℕ) (mean_score_other : ℕ) (h1 : total_students = 20) (h2 : other_students = 10)
 (h3 : mean_score_first = 80) (h4 : mean_score_class = 70) :
 mean_score_other = 60 :=
by
  sorry

end average_other_students_l165_165215


namespace jenny_improvements_value_l165_165311

-- Definitions based on the conditions provided
def property_tax_rate : ℝ := 0.02
def initial_house_value : ℝ := 400000
def rail_project_increase : ℝ := 0.25
def affordable_property_tax : ℝ := 15000

-- Statement of the theorem
theorem jenny_improvements_value :
  let new_house_value := initial_house_value * (1 + rail_project_increase)
  let max_affordable_house_value := affordable_property_tax / property_tax_rate
  let value_of_improvements := max_affordable_house_value - new_house_value
  value_of_improvements = 250000 := 
by
  sorry

end jenny_improvements_value_l165_165311


namespace largest_lcm_l165_165049

theorem largest_lcm :
  ∀ (a b c d e f : ℕ),
  a = Nat.lcm 18 2 →
  b = Nat.lcm 18 4 →
  c = Nat.lcm 18 6 →
  d = Nat.lcm 18 9 →
  e = Nat.lcm 18 12 →
  f = Nat.lcm 18 16 →
  max (max (max (max (max a b) c) d) e) f = 144 :=
by
  intros a b c d e f ha hb hc hd he hf
  sorry

end largest_lcm_l165_165049


namespace union_example_l165_165516

theorem union_example (P Q : Set ℕ) (hP : P = {1, 2, 3, 4}) (hQ : Q = {2, 4}) :
  P ∪ Q = {1, 2, 3, 4} :=
by
  sorry

end union_example_l165_165516


namespace calculate_unoccupied_volume_l165_165173

def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def water_volume : ℕ := tank_volume / 3
def ice_cube_volume : ℕ := 1
def ice_cubes_count : ℕ := 12
def total_ice_volume : ℕ := ice_cubes_count * ice_cube_volume
def occupied_volume : ℕ := water_volume + total_ice_volume

def unoccupied_volume : ℕ := tank_volume - occupied_volume

theorem calculate_unoccupied_volume : unoccupied_volume = 628 := by
  sorry

end calculate_unoccupied_volume_l165_165173


namespace total_money_l165_165865

def JamesPocketBills : Nat := 3
def BillValue : Nat := 20
def WalletMoney : Nat := 75

theorem total_money (JamesPocketBills BillValue WalletMoney : Nat) : 
  (JamesPocketBills * BillValue + WalletMoney) = 135 :=
by
  sorry

end total_money_l165_165865


namespace even_fn_a_eq_zero_l165_165805

def f (x a : ℝ) : ℝ := x^2 - |x + a|

theorem even_fn_a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end even_fn_a_eq_zero_l165_165805


namespace relationship_of_y_values_l165_165653

theorem relationship_of_y_values 
  (k : ℝ) (x1 x2 x3 y1 y2 y3 : ℝ)
  (h_pos : k > 0) 
  (hA : y1 = k / x1) 
  (hB : y2 = k / x2) 
  (hC : y3 = k / x3) 
  (h_order : x1 < 0 ∧ 0 < x2 ∧ x2 < x3) : y1 < y3 ∧ y3 < y2 := 
by
  sorry

end relationship_of_y_values_l165_165653


namespace find_ab_l165_165884

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, (3 * x - a) * (2 * x + 5) - x = 6 * x^2 + 2 * (5 * x - b)) → a = 2 ∧ b = 5 :=
by
  intro h
  -- We assume the condition holds for all x
  sorry -- Proof not needed as per instructions

end find_ab_l165_165884


namespace consecutive_integers_greatest_l165_165450

theorem consecutive_integers_greatest (n : ℤ) (h : n + 2 = 8) : 
  (n + 2 = 8) → (max n (max (n + 1) (n + 2)) = 8) :=
by {
  sorry
}

end consecutive_integers_greatest_l165_165450


namespace blister_slowdown_l165_165014

theorem blister_slowdown
    (old_speed new_speed time : ℕ) (new_speed_initial : ℕ) (blister_freq : ℕ)
    (distance_old : ℕ) (blister_per_hour_slowdown : ℝ):
    -- Given conditions
    old_speed = 6 →
    new_speed = 11 →
    new_speed_initial = 11 →
    time = 4 →
    blister_freq = 2 →
    distance_old = old_speed * time →
    -- Prove that each blister slows Candace down by 10 miles per hour
    blister_per_hour_slowdown = 10 :=
  by
    sorry

end blister_slowdown_l165_165014


namespace problem_statement_l165_165938

def a : ℝ × ℝ := (0, 2)
def b : ℝ × ℝ := (2, 2)

def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem problem_statement : dot_product (vector_sub a b) a = 0 := 
by 
  -- The proof would go here
  sorry

end problem_statement_l165_165938


namespace ellipse_condition_l165_165497

theorem ellipse_condition (k : ℝ) :
  (4 < k ∧ k < 9) ↔ (9 - k > 0 ∧ k - 4 > 0 ∧ 9 - k ≠ k - 4) :=
by sorry

end ellipse_condition_l165_165497


namespace find_f_zero_l165_165654

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem find_f_zero (a b : ℝ)
  (h1 : f 3 a b = 7)
  (h2 : f 5 a b = -1) : f 0 a b = 19 :=
by
  sorry

end find_f_zero_l165_165654


namespace selling_price_range_l165_165227

theorem selling_price_range
  (unit_purchase_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_sales_volume : ℝ)
  (price_increase_effect : ℝ)
  (daily_profit_threshold : ℝ)
  (x : ℝ) :
  unit_purchase_price = 8 →
  initial_selling_price = 10 →
  initial_sales_volume = 100 →
  price_increase_effect = 10 →
  daily_profit_threshold = 320 →
  (initial_selling_price - unit_purchase_price) * initial_sales_volume > daily_profit_threshold →
  12 < x → x < 16 →
  (x - unit_purchase_price) * (initial_sales_volume - price_increase_effect * (x - initial_selling_price)) > daily_profit_threshold :=
sorry

end selling_price_range_l165_165227


namespace common_terms_count_l165_165500

theorem common_terms_count (β : ℕ) (h1 : β = 55) (h2 : β + 1 = 56) : 
  ∃ γ : ℕ, γ = 6 :=
by
  sorry

end common_terms_count_l165_165500


namespace dumpling_probability_l165_165618

theorem dumpling_probability :
  let total_dumplings := 15
  let choose4 := Nat.choose total_dumplings 4
  let choose1 := Nat.choose 3 1
  let choose5_2 := Nat.choose 5 2
  let choose5_1 := Nat.choose 5 1
  (choose1 * choose5_2 * choose5_1 * choose5_1) / choose4 = 50 / 91 := by
  sorry

end dumpling_probability_l165_165618


namespace B_work_days_l165_165881

/-- 
  A and B undertake to do a piece of work for $500.
  A alone can do it in 5 days while B alone can do it in a certain number of days.
  With the help of C, they finish it in 2 days. C's share is $200.
  Prove B alone can do the work in 10 days.
-/
theorem B_work_days (x : ℕ) (h1 : (1/5 : ℝ) + (1/x : ℝ) = 3/10) : x = 10 := 
  sorry

end B_work_days_l165_165881


namespace positive_difference_balances_l165_165105

noncomputable def cedric_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

noncomputable def daniel_balance (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r) ^ t

theorem positive_difference_balances :
  let P : ℝ := 15000
  let r_cedric : ℝ := 0.06
  let r_daniel : ℝ := 0.08
  let t : ℕ := 15
  let A_cedric := cedric_balance P r_cedric t
  let A_daniel := daniel_balance P r_daniel t
  (A_daniel - A_cedric) = 11632.65 :=
by
  sorry

end positive_difference_balances_l165_165105


namespace trapezoid_area_l165_165627

-- Definitions based on the problem conditions
def Vertex := (Real × Real)

structure Triangle :=
(A : Vertex)
(B : Vertex)
(C : Vertex)
(area : Real)

structure Trapezoid :=
(AB : Real)
(CD : Real)
(M : Vertex)
(area_triangle_ABM : Real)
(area_triangle_CDM : Real)

-- The main theorem we want to prove
theorem trapezoid_area (T : Trapezoid)
  (parallel_sides : T.AB < T.CD)
  (intersect_at_M : ∃ M : Vertex, M = T.M)
  (area_ABM : T.area_triangle_ABM = 2)
  (area_CDM : T.area_triangle_CDM = 8) :
  T.AB * T.CD / (T.CD - T.AB) + T.CD * T.AB / (T.CD - T.AB) = 18 :=
sorry

end trapezoid_area_l165_165627


namespace mark_donates_cans_l165_165447

-- Definitions coming directly from the conditions
def num_shelters : ℕ := 6
def people_per_shelter : ℕ := 30
def cans_per_person : ℕ := 10

-- The final statement to be proven
theorem mark_donates_cans : (num_shelters * people_per_shelter * cans_per_person) = 1800 :=
by sorry

end mark_donates_cans_l165_165447


namespace cubic_polynomial_root_sum_cube_value_l165_165948

noncomputable def α : ℝ := (17 : ℝ)^(1 / 3)
noncomputable def β : ℝ := (67 : ℝ)^(1 / 3)
noncomputable def γ : ℝ := (137 : ℝ)^(1 / 3)

theorem cubic_polynomial_root_sum_cube_value
    (p q r : ℝ)
    (h1 : (p - α) * (p - β) * (p - γ) = 1)
    (h2 : (q - α) * (q - β) * (q - γ) = 1)
    (h3 : (r - α) * (r - β) * (r - γ) = 1) :
    p^3 + q^3 + r^3 = 218 := 
by
  sorry

end cubic_polynomial_root_sum_cube_value_l165_165948


namespace even_function_value_l165_165489

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 3 * a + b

theorem even_function_value (h_even : ∀ x, f a b x = f a b (-x))
    (h_domain : a - 1 = -2 * a) :
    f a (0 : ℝ) (1 / 2) = 13 / 12 :=
by
  sorry

end even_function_value_l165_165489


namespace calculate_two_squared_l165_165581

theorem calculate_two_squared : 2^2 = 4 :=
by
  sorry

end calculate_two_squared_l165_165581


namespace real_part_of_product_l165_165202

open Complex

theorem real_part_of_product (α β : ℝ) :
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  Complex.re (z1 * z2) = Real.cos (α + β) :=
by
  let z1 := Complex.mk (Real.cos α) (Real.sin α)
  let z2 := Complex.mk (Real.cos β) (Real.sin β)
  sorry

end real_part_of_product_l165_165202


namespace order_abc_l165_165777

noncomputable def a : ℝ := (3 * (2 - Real.log 3)) / Real.exp 2
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.sqrt (Real.exp 1)) / (2 * Real.exp 1)

theorem order_abc : c < a ∧ a < b := by
  sorry

end order_abc_l165_165777


namespace probability_of_spade_then_king_l165_165176

theorem probability_of_spade_then_king :
  ( (24 / 104) * (8 / 103) + (2 / 104) * (7 / 103) ) = 103 / 5356 :=
sorry

end probability_of_spade_then_king_l165_165176


namespace factorization_of_polynomial_l165_165901

theorem factorization_of_polynomial :
  ∀ x : ℝ, (x^4 - 4*x^3 + 6*x^2 - 4*x + 1) = (x - 1)^4 :=
by
  intro x
  sorry

end factorization_of_polynomial_l165_165901


namespace count_multiples_of_5_not_10_or_15_l165_165144

theorem count_multiples_of_5_not_10_or_15 : 
  ∃ n : ℕ, n = 33 ∧ (∀ x : ℕ, x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0) → x < 500 ∧ (x % 5 = 0) ∧ (x % 10 ≠ 0) ∧ (x % 15 ≠ 0)) :=
by
  sorry

end count_multiples_of_5_not_10_or_15_l165_165144


namespace area_of_region_l165_165838

theorem area_of_region :
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  (∫ t in (Real.pi / 3)..(Real.pi / 2), (x t) * (deriv y t)) * 2 = 2 * Real.pi - 3 * Real.sqrt 3 := by
  let x := fun t : ℝ => 6 * Real.cos t
  let y := fun t : ℝ => 2 * Real.sin t
  have h1 : ∫ t in (Real.pi / 3)..(Real.pi / 2), x t * deriv y t = 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2*t)) / 2 := sorry
  have h2 : 12 * ∫ t in (Real.pi / 3)..(Real.pi / 2), (1 + Real.cos (2 * t)) / 2 = 2 * Real.pi - 3 * Real.sqrt 3 := sorry
  sorry

end area_of_region_l165_165838


namespace log9_log11_lt_one_l165_165922

theorem log9_log11_lt_one (log9_pos : 0 < Real.log 9) (log11_pos : 0 < Real.log 11) : 
  Real.log 9 * Real.log 11 < 1 :=
by
  sorry

end log9_log11_lt_one_l165_165922


namespace max_distinct_prime_factors_of_a_l165_165403

noncomputable def distinct_prime_factors (n : ℕ) : ℕ := sorry -- placeholder for the number of distinct prime factors

theorem max_distinct_prime_factors_of_a (a b : ℕ)
  (ha_pos : a > 0) (hb_pos : b > 0)
  (gcd_ab_primes : distinct_prime_factors (gcd a b) = 5)
  (lcm_ab_primes : distinct_prime_factors (lcm a b) = 18)
  (a_less_than_b : distinct_prime_factors a < distinct_prime_factors b) :
  distinct_prime_factors a = 11 :=
sorry

end max_distinct_prime_factors_of_a_l165_165403


namespace baseball_cards_per_friend_l165_165353

theorem baseball_cards_per_friend (total_cards friends : ℕ) (h_total : total_cards = 24) (h_friends : friends = 4) : total_cards / friends = 6 :=
by
  sorry

end baseball_cards_per_friend_l165_165353


namespace train_speed_l165_165945

theorem train_speed (d t : ℝ) (h1 : d = 500) (h2 : t = 3) : d / t = 166.67 := by
  sorry

end train_speed_l165_165945


namespace ratio_of_areas_l165_165940

def side_length_S : ℝ := sorry
def longer_side_R : ℝ := 1.2 * side_length_S
def shorter_side_R : ℝ := 0.8 * side_length_S
def area_S : ℝ := side_length_S ^ 2
def area_R : ℝ := longer_side_R * shorter_side_R

theorem ratio_of_areas (side_length_S : ℝ) :
  (area_R / area_S) = (24 / 25) :=
by
  sorry

end ratio_of_areas_l165_165940


namespace matrix_cube_l165_165741

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_cube : A^3 = !![3, -6; 6, -3] := by
  sorry

end matrix_cube_l165_165741


namespace smallest_integer_solution_l165_165220

theorem smallest_integer_solution (x : ℤ) :
  (7 - 5 * x < 12) → ∃ (n : ℤ), x = n ∧ n = 0 :=
by
  intro h
  sorry

end smallest_integer_solution_l165_165220


namespace restore_price_by_percentage_l165_165371

theorem restore_price_by_percentage 
  (p : ℝ) -- original price
  (h₀ : p > 0) -- condition that price is positive
  (r₁ : ℝ := 0.25) -- reduction of 25%
  (r₁_applied : ℝ := p * (1 - r₁)) -- first reduction
  (r₂ : ℝ := 0.20) -- additional reduction of 20%
  (r₂_applied : ℝ := r₁_applied * (1 - r₂)) -- second reduction
  (final_price : ℝ := r₂_applied) -- final price after two reductions
  (increase_needed : ℝ := p - final_price) -- amount to increase to restore the price
  (percent_increase : ℝ := (increase_needed / final_price) * 100) -- percentage increase needed
  : abs (percent_increase - 66.67) < 0.01 := -- proof that percentage increase is approximately 66.67%
sorry

end restore_price_by_percentage_l165_165371


namespace number_of_polynomials_l165_165593

-- Define conditions
def is_positive_integer (n : ℤ) : Prop :=
  5 * 151 * n > 0

-- Define the main theorem
theorem number_of_polynomials (n : ℤ) (h : is_positive_integer n) : 
  ∃ k : ℤ, k = ⌊n / 2⌋ + 1 :=
by
  sorry

end number_of_polynomials_l165_165593


namespace smallest_number_diminished_by_2_divisible_12_16_18_21_28_l165_165630

def conditions_holds (n : ℕ) : Prop :=
  (n - 2) % 12 = 0 ∧ (n - 2) % 16 = 0 ∧ (n - 2) % 18 = 0 ∧ (n - 2) % 21 = 0 ∧ (n - 2) % 28 = 0

theorem smallest_number_diminished_by_2_divisible_12_16_18_21_28 :
  ∃ (n : ℕ), conditions_holds n ∧ (∀ m, conditions_holds m → n ≤ m) ∧ n = 1009 :=
by
  sorry

end smallest_number_diminished_by_2_divisible_12_16_18_21_28_l165_165630


namespace grey_eyes_black_hair_l165_165853

-- Definitions based on conditions
def num_students := 60
def num_black_hair := 36
def num_green_eyes_red_hair := 20
def num_grey_eyes := 24

-- Calculate number of students with red hair
def num_red_hair := num_students - num_black_hair

-- Calculate number of grey-eyed students with red hair
def num_grey_eyes_red_hair := num_red_hair - num_green_eyes_red_hair

-- Prove the number of grey-eyed students with black hair
theorem grey_eyes_black_hair:
  ∃ n, n = num_grey_eyes - num_grey_eyes_red_hair ∧ n = 20 :=
by
  sorry

end grey_eyes_black_hair_l165_165853


namespace min_fraction_value_l165_165764

-- Define the conditions: geometric sequence, specific term relationship, product of terms

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, ∀ n, a (n + 1) = a n * q

def specific_term_relationship (a : ℕ → ℝ) : Prop :=
  a 3 = a 2 + 2 * a 1

def product_of_terms (a : ℕ → ℝ) (m n : ℕ) : Prop :=
  a m * a n = 64 * (a 1)^2

def min_value_fraction (m n : ℕ) : Prop :=
  1 / m + 9 / n = 2

theorem min_fraction_value (a : ℕ → ℝ) (m n : ℕ)
  (h1 : geometric_sequence a)
  (h2 : specific_term_relationship a)
  (h3 : product_of_terms a m n)
  : min_value_fraction m n := by
  sorry

end min_fraction_value_l165_165764


namespace melanie_turnips_l165_165108

theorem melanie_turnips (b : ℕ) (d : ℕ) (h_b : b = 113) (h_d : d = 26) : b + d = 139 :=
by
  sorry

end melanie_turnips_l165_165108


namespace families_seating_arrangements_l165_165150

theorem families_seating_arrangements : 
  let factorial := Nat.factorial
  let family_ways := factorial 3
  let bundles := family_ways * family_ways * family_ways
  let bundle_ways := factorial 3
  bundles * bundle_ways = (factorial 3) ^ 4 := by
  sorry

end families_seating_arrangements_l165_165150


namespace duration_of_period_l165_165103

noncomputable def birth_rate : ℕ := 7
noncomputable def death_rate : ℕ := 3
noncomputable def net_increase : ℕ := 172800

theorem duration_of_period : (net_increase / ((birth_rate - death_rate) / 2)) / 3600 = 12 := by
  sorry

end duration_of_period_l165_165103


namespace regular_octagon_exterior_angle_l165_165509

theorem regular_octagon_exterior_angle : 
  ∀ (n : ℕ), n = 8 → (180 * (n - 2) / n) + (180 - (180 * (n - 2) / n)) = 180 := by
  sorry

end regular_octagon_exterior_angle_l165_165509


namespace number_of_arrangements_l165_165429

noncomputable def arrangements_nonadjacent_teachers (A : ℕ → ℕ → ℕ) : ℕ :=
  let students_arrangements := A 8 8
  let gaps_count := 9
  let teachers_arrangements := A gaps_count 2
  students_arrangements * teachers_arrangements

theorem number_of_arrangements (A : ℕ → ℕ → ℕ) :
  arrangements_nonadjacent_teachers A = A 8 8 * A 9 2 := 
  sorry

end number_of_arrangements_l165_165429


namespace rainfall_ratio_l165_165903

theorem rainfall_ratio (S M T : ℝ) (h1 : M = S + 3) (h2 : S = 4) (h3 : S + M + T = 25) : T / M = 2 :=
by
  sorry

end rainfall_ratio_l165_165903


namespace sheilas_family_contribution_l165_165968

theorem sheilas_family_contribution :
  let initial_amount := 3000
  let monthly_savings := 276
  let duration_years := 4
  let total_after_duration := 23248
  let months_in_year := 12
  let total_months := duration_years * months_in_year
  let savings_over_duration := monthly_savings * total_months
  let sheilas_total_savings := initial_amount + savings_over_duration
  let family_contribution := total_after_duration - sheilas_total_savings
  family_contribution = 7000 :=
by
  sorry

end sheilas_family_contribution_l165_165968


namespace xiaohua_apples_l165_165262

theorem xiaohua_apples (x : ℕ) (h1 : ∃ n, (n = 4 * x + 20)) 
                       (h2 : (4 * x + 20 - 8 * (x - 1) > 0) ∧ (4 * x + 20 - 8 * (x - 1) < 8)) : 
                       4 * x + 20 = 44 := by
  sorry

end xiaohua_apples_l165_165262


namespace N_perfect_square_l165_165910

theorem N_perfect_square (N : ℕ) (hN_pos : N > 0) 
  (h_pairs : ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 2005 ∧ 
  ∀ p ∈ pairs, (1 : ℚ) / (p.1 : ℚ) + (1 : ℚ) / (p.2 : ℚ) = (1 : ℚ) / N ∧ p.1 > 0 ∧ p.2 > 0) : 
  ∃ k : ℕ, N = k^2 := 
sorry

end N_perfect_square_l165_165910


namespace range_of_a_l165_165964

variable (a x : ℝ)

-- Condition p: ∀ x ∈ [1, 2], x^2 - a ≥ 0
def p : Prop := ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

-- Condition q: ∃ x ∈ ℝ, x^2 + 2 * a * x + 2 - a = 0
def q : Prop := ∃ x, x^2 + 2 * a * x + 2 - a = 0

-- The proof goal given p ∧ q: a ≤ -2 or a = 1
theorem range_of_a (h : p a ∧ q a) : a ≤ -2 ∨ a = 1 := sorry

end range_of_a_l165_165964


namespace smallest_consecutive_integer_l165_165404

theorem smallest_consecutive_integer (n : ℤ) (h : 7 * n + 21 = 112) : n = 13 :=
sorry

end smallest_consecutive_integer_l165_165404


namespace turnip_difference_l165_165289

theorem turnip_difference (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : melanie_turnips - benny_turnips = 26 := by
  sorry

end turnip_difference_l165_165289


namespace quadratic_min_value_max_l165_165987

theorem quadratic_min_value_max (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : b^2 - 4 * a * c ≥ 0) :
    (min (min ((b + c) / a) ((c + a) / b)) ((a + b) / c)) ≤ (5 / 4) :=
sorry

end quadratic_min_value_max_l165_165987


namespace value_range_abs_function_l165_165694

theorem value_range_abs_function : 
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 9 → 1 ≤ (abs (x - 3) + 1) ∧ (abs (x - 3) + 1) ≤ 7 :=
by
  intro x hx
  sorry

end value_range_abs_function_l165_165694


namespace student_correct_answers_l165_165204

theorem student_correct_answers (C I : ℕ) (h1 : C + I = 100) (h2 : C - 2 * I = 64) : C = 88 :=
by
  sorry

end student_correct_answers_l165_165204


namespace sin_13pi_over_4_l165_165409

theorem sin_13pi_over_4 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 :=
by
  sorry

end sin_13pi_over_4_l165_165409


namespace computers_built_per_month_l165_165819

theorem computers_built_per_month (days_in_month : ℕ) (hours_per_day : ℕ) (computers_per_interval : ℚ) (intervals_per_hour : ℕ)
    (h_days : days_in_month = 28) (h_hours : hours_per_day = 24) (h_computers : computers_per_interval = 2.25) (h_intervals : intervals_per_hour = 2) :
    days_in_month * hours_per_day * intervals_per_hour * computers_per_interval = 3024 :=
by
  -- We would give the proof here, but it's omitted as per instructions.
  sorry

end computers_built_per_month_l165_165819


namespace alpha_beta_range_l165_165203

theorem alpha_beta_range (α β : ℝ) (P : ℝ × ℝ)
  (h1 : α > 0) 
  (h2 : β > 0) 
  (h3 : P = (α, 3 * β))
  (circle_eq : (α - 1)^2 + 9 * (β^2) = 1) :
  1 < α + β ∧ α + β < 5 / 3 :=
sorry

end alpha_beta_range_l165_165203


namespace jane_total_worth_l165_165439

open Nat

theorem jane_total_worth (q d : ℕ) (h1 : q + d = 30)
  (h2 : 25 * q + 10 * d + 150 = 10 * q + 25 * d) :
  25 * q + 10 * d = 450 :=
by
  sorry

end jane_total_worth_l165_165439


namespace Stuart_reward_points_l165_165753

theorem Stuart_reward_points (reward_points_per_unit : ℝ) (spending : ℝ) (unit_amount : ℝ) : 
  reward_points_per_unit = 5 → 
  spending = 200 → 
  unit_amount = 25 → 
  (spending / unit_amount) * reward_points_per_unit = 40 :=
by 
  intros h_points h_spending h_unit
  sorry

end Stuart_reward_points_l165_165753


namespace mixed_sum_in_range_l165_165246

def mixed_to_improper (a : ℕ) (b c : ℕ) : ℚ := a + b / c

def mixed_sum (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℕ) : ℚ :=
  (mixed_to_improper a1 b1 c1) + (mixed_to_improper a2 b2 c2) + (mixed_to_improper a3 b3 c3)

theorem mixed_sum_in_range :
  11 < mixed_sum 1 4 6 3 1 2 8 3 21 ∧ mixed_sum 1 4 6 3 1 2 8 3 21 < 12 :=
by { sorry }

end mixed_sum_in_range_l165_165246


namespace division_identity_l165_165022

theorem division_identity (h : 6 / 3 = 2) : 72 / (6 / 3) = 36 := by
  sorry

end division_identity_l165_165022


namespace assignment_schemes_correct_l165_165928

-- Define the total number of students
def total_students : ℕ := 6

-- Define the total number of tasks
def total_tasks : ℕ := 4

-- Define a predicate that checks if a student can be assigned to task A
def can_assign_to_task_A (student : ℕ) : Prop := student ≠ 1 ∧ student ≠ 2

-- Calculate the total number of unrestricted assignments
def total_unrestricted_assignments : ℕ := 6 * 5 * 4 * 3

-- Calculate the restricted number of assignments if student A or B is assigned to task A
def restricted_assignments : ℕ := 2 * 5 * 4 * 3

-- Define the problem statement
def number_of_assignment_schemes : ℕ :=
  total_unrestricted_assignments - restricted_assignments

-- The theorem to prove
theorem assignment_schemes_correct :
  number_of_assignment_schemes = 240 :=
by
  -- We acknowledge the problem statement is correct
  sorry

end assignment_schemes_correct_l165_165928


namespace Jake_has_8_peaches_l165_165988

variable (Jake Steven Jill : ℕ)

theorem Jake_has_8_peaches
  (h_steven_peaches : Steven = 15)
  (h_steven_jill : Steven = Jill + 14)
  (h_jake_steven : Jake = Steven - 7) :
  Jake = 8 := by
  sorry

end Jake_has_8_peaches_l165_165988


namespace four_at_three_equals_thirty_l165_165123

def custom_operation (a b : ℕ) : ℕ :=
  3 * a^2 - 2 * b^2

theorem four_at_three_equals_thirty : custom_operation 4 3 = 30 :=
by
  sorry

end four_at_three_equals_thirty_l165_165123


namespace proof_problem_l165_165487

noncomputable def a : ℝ := (11 + Real.sqrt 337) ^ (1 / 3)
noncomputable def b : ℝ := (11 - Real.sqrt 337) ^ (1 / 3)
noncomputable def x : ℝ := a + b

theorem proof_problem : x^3 + 18 * x = 22 := by
  sorry

end proof_problem_l165_165487


namespace no_half_probability_socks_l165_165526

theorem no_half_probability_socks (n m : ℕ) (h_sum : n + m = 2009) : 
  (n - m)^2 ≠ 2009 :=
sorry

end no_half_probability_socks_l165_165526


namespace acute_triangle_angle_A_is_60_degrees_l165_165090

open Real

variables {A B C : ℝ} -- Assume A, B, C are reals representing the angles of the triangle

theorem acute_triangle_angle_A_is_60_degrees
  (h_acute : A < 90 ∧ B < 90 ∧ C < 90)
  (h_eq_dist : dist A O = dist A H) : A = 60 :=
  sorry

end acute_triangle_angle_A_is_60_degrees_l165_165090


namespace find_f_20_l165_165799

theorem find_f_20 (f : ℝ → ℝ)
  (h1 : ∀ x : ℝ, f x = (1/2) * f (x + 2))
  (h2 : f 2 = 1) :
  f 20 = 512 :=
sorry

end find_f_20_l165_165799


namespace volume_ratio_of_rotated_solids_l165_165828

theorem volume_ratio_of_rotated_solids (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let V1 := π * b^2 * a
  let V2 := π * a^2 * b
  V1 / V2 = b / a :=
by
  intros
  -- Proof omitted
  sorry

end volume_ratio_of_rotated_solids_l165_165828


namespace total_instruments_correct_l165_165219

def fingers : Nat := 10
def hands : Nat := 2
def heads : Nat := 1

def trumpets := fingers - 3
def guitars := hands + 2
def trombones := heads + 2
def french_horns := guitars - 1
def violins := trumpets / 2
def saxophones := trombones / 3

theorem total_instruments_correct : 
  (trumpets + guitars = trombones + violins + saxophones) →
  trumpets + guitars + trombones + french_horns + violins + saxophones = 21 := by
  sorry

end total_instruments_correct_l165_165219


namespace find_a_and_b_l165_165599

theorem find_a_and_b (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = x^3 - a * x^2 - b * x + a^2) →
  f 1 = 10 →
  deriv f 1 = 0 →
  (a = -4 ∧ b = 11) :=
by
  intros hf hf1 hderiv
  sorry

end find_a_and_b_l165_165599


namespace abs_inequality_proof_by_contradiction_l165_165718

theorem abs_inequality_proof_by_contradiction (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  |a| > |b| :=
by
  let h := |a| ≤ |b|
  sorry

end abs_inequality_proof_by_contradiction_l165_165718


namespace find_a_minus_b_l165_165422

variable {a b : ℤ}

theorem find_a_minus_b (h1 : a^2 = 9) (h2 : |b| = 4) (h3 : a > b) : a - b = 7 :=
  sorry

end find_a_minus_b_l165_165422


namespace radius_range_of_sector_l165_165650

theorem radius_range_of_sector (a : ℝ) (h : a > 0) :
  ∃ (R : ℝ), (a / (2 * (1 + π)) < R ∧ R < a / 2) :=
sorry

end radius_range_of_sector_l165_165650


namespace diane_bakes_gingerbreads_l165_165662

open Nat

theorem diane_bakes_gingerbreads :
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  total_gingerbreads1 + total_gingerbreads2 = 160 := 
by
  let trays1 := 4
  let gingerbreads_per_tray1 := 25
  let trays2 := 3
  let gingerbreads_per_tray2 := 20
  let total_gingerbreads1 := trays1 * gingerbreads_per_tray1
  let total_gingerbreads2 := trays2 * gingerbreads_per_tray2
  exact Eq.refl (total_gingerbreads1 + total_gingerbreads2)

end diane_bakes_gingerbreads_l165_165662


namespace find_sixth_term_l165_165037

noncomputable def first_term : ℝ := Real.sqrt 3
noncomputable def fifth_term : ℝ := Real.sqrt 243
noncomputable def common_ratio (q : ℝ) : Prop := fifth_term = first_term * q^4
noncomputable def sixth_term (b6 : ℝ) (q : ℝ) : Prop := b6 = fifth_term * q

theorem find_sixth_term (q : ℝ) (b6 : ℝ) : 
  first_term = Real.sqrt 3 ∧
  fifth_term = Real.sqrt 243 ∧
  common_ratio q ∧ 
  sixth_term b6 q → 
  b6 = 27 ∨ b6 = -27 := 
by
  intros
  sorry

end find_sixth_term_l165_165037


namespace slope_intercept_product_l165_165170

theorem slope_intercept_product (b m : ℤ) (h1 : b = -3) (h2 : m = 3) : m * b = -9 := by
  sorry

end slope_intercept_product_l165_165170


namespace expected_total_rain_correct_l165_165259

-- Define the probabilities and rain amounts for one day.
def prob_sun : ℝ := 0.30
def prob_rain3 : ℝ := 0.40
def prob_rain8 : ℝ := 0.30
def rain_sun : ℝ := 0
def rain_three : ℝ := 3
def rain_eight : ℝ := 8
def days : ℕ := 7

-- Define the expected value of daily rain.
def E_daily_rain : ℝ :=
  prob_sun * rain_sun + prob_rain3 * rain_three + prob_rain8 * rain_eight

-- Define the expected total rain over seven days.
def E_total_rain : ℝ :=
  days * E_daily_rain

-- Statement of the proof problem.
theorem expected_total_rain_correct : E_total_rain = 25.2 := by
  -- Proof goes here
  sorry

end expected_total_rain_correct_l165_165259


namespace star_operation_possible_l165_165649

noncomputable def star_operation_exists : Prop := 
  ∃ (star : ℤ → ℤ → ℤ), 
  (∀ (a b c : ℤ), star (star a b) c = star a (star b c)) ∧ 
  (∀ (x y : ℤ), star (star x x) y = y ∧ star y (star x x) = y)

theorem star_operation_possible : star_operation_exists :=
sorry

end star_operation_possible_l165_165649


namespace problem_statement_l165_165949

variables {A B C O D : Type}
variables [AddCommGroup A] [Module ℝ A]
variables (a b c o d : A)

-- Define the geometric conditions
axiom condition1 : a + 2 • b + 3 • c = 0
axiom condition2 : ∃ (D: A), (∃ (k : ℝ), a = k • d ∧ k ≠ 0) ∧ (∃ (u v : ℝ),  u • b + v • c = d ∧ u + v = 1)

-- Define points
def OA : A := a - o
def OB : A := b - o
def OC : A := c - o
def OD : A := d - o

-- The main statement to prove
theorem problem_statement : 2 • (b - d) + 3 • (c - d) = (0 : A) :=
by
  sorry

end problem_statement_l165_165949


namespace point_side_opposite_l165_165011

def equation_lhs (x y : ℝ) : ℝ := 2 * y - 6 * x + 1

theorem point_side_opposite : 
  (equation_lhs 0 0 * equation_lhs 2 1 < 0) := 
by 
   sorry

end point_side_opposite_l165_165011


namespace solve_for_m_l165_165750

noncomputable def operation (a b c x y : ℝ) := a * x + b * y + c * x * y

theorem solve_for_m (a b c : ℝ) (h1 : operation a b c 1 2 = 3)
                              (h2 : operation a b c 2 3 = 4) 
                              (h3 : ∃ (m : ℝ), m ≠ 0 ∧ ∀ (x : ℝ), operation a b c x m = x) :
  ∃ (m : ℝ), m = 4 :=
sorry

end solve_for_m_l165_165750


namespace number_of_soccer_campers_l165_165206

-- Conditions as definitions in Lean
def total_campers : ℕ := 88
def basketball_campers : ℕ := 24
def football_campers : ℕ := 32
def soccer_campers : ℕ := total_campers - (basketball_campers + football_campers)

-- Theorem statement to prove
theorem number_of_soccer_campers : soccer_campers = 32 := by
  sorry

end number_of_soccer_campers_l165_165206


namespace range_of_m_l165_165339

theorem range_of_m (k : ℝ) (m : ℝ) (y x : ℝ)
  (h1 : ∀ x, y = k * (x - 1) + m)
  (h2 : y = 3 ∧ x = -2)
  (h3 : (∃ x, x < 0 ∧ y > 0) ∧ (∃ x, x < 0 ∧ y < 0) ∧ (∃ x, x > 0 ∧ y < 0)) :
  m < - (3 / 2) :=
sorry

end range_of_m_l165_165339


namespace total_cars_produced_l165_165622

def CarCompanyA_NorthAmerica := 3884
def CarCompanyA_Europe := 2871
def CarCompanyA_Asia := 1529

def CarCompanyB_NorthAmerica := 4357
def CarCompanyB_Europe := 3690
def CarCompanyB_Asia := 1835

def CarCompanyC_NorthAmerica := 2937
def CarCompanyC_Europe := 4210
def CarCompanyC_Asia := 977

def TotalNorthAmerica :=
  CarCompanyA_NorthAmerica + CarCompanyB_NorthAmerica + CarCompanyC_NorthAmerica

def TotalEurope :=
  CarCompanyA_Europe + CarCompanyB_Europe + CarCompanyC_Europe

def TotalAsia :=
  CarCompanyA_Asia + CarCompanyB_Asia + CarCompanyC_Asia

def TotalProduction := TotalNorthAmerica + TotalEurope + TotalAsia

theorem total_cars_produced : TotalProduction = 26290 := 
by sorry

end total_cars_produced_l165_165622


namespace alex_lost_fish_l165_165678

theorem alex_lost_fish (jacob_initial : ℕ) (alex_catch_ratio : ℕ) (jacob_additional : ℕ) (alex_initial : ℕ) (alex_final : ℕ) : 
  (jacob_initial = 8) → 
  (alex_catch_ratio = 7) → 
  (jacob_additional = 26) →
  (alex_initial = alex_catch_ratio * jacob_initial) →
  (alex_final = (jacob_initial + jacob_additional) - 1) → 
  alex_initial - alex_final = 23 :=
by
  intros
  sorry

end alex_lost_fish_l165_165678


namespace unicorn_journey_length_l165_165935

theorem unicorn_journey_length (num_unicorns : ℕ) (flowers_per_step : ℕ) (total_flowers : ℕ) (step_length_meters : ℕ) : (num_unicorns = 6) → (flowers_per_step = 4) → (total_flowers = 72000) → (step_length_meters = 3) → 
(total_flowers / flowers_per_step / num_unicorns * step_length_meters / 1000 = 9) :=
by
  intros h1 h2 h3 h4
  sorry

end unicorn_journey_length_l165_165935


namespace johns_improvement_l165_165839

-- Declare the variables for the initial and later lap times.
def initial_minutes : ℕ := 50
def initial_laps : ℕ := 25
def later_minutes : ℕ := 54
def later_laps : ℕ := 30

-- Calculate the initial and later lap times in seconds, and the improvement.
def initial_lap_time_seconds := (initial_minutes * 60) / initial_laps 
def later_lap_time_seconds := (later_minutes * 60) / later_laps
def improvement := initial_lap_time_seconds - later_lap_time_seconds

-- State the theorem to prove the improvement is 12 seconds per lap.
theorem johns_improvement : improvement = 12 := by
  sorry

end johns_improvement_l165_165839


namespace pow_two_div_factorial_iff_exists_l165_165583

theorem pow_two_div_factorial_iff_exists (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, k > 0 ∧ n = 2^(k-1)) ↔ 2^(n-1) ∣ n! := 
by {
  sorry
}

end pow_two_div_factorial_iff_exists_l165_165583


namespace net_calorie_deficit_l165_165140

-- Define the conditions as constants.
def total_distance : ℕ := 3
def calories_burned_per_mile : ℕ := 150
def calories_in_candy_bar : ℕ := 200

-- Prove the net calorie deficit.
theorem net_calorie_deficit : total_distance * calories_burned_per_mile - calories_in_candy_bar = 250 := by
  sorry

end net_calorie_deficit_l165_165140


namespace number_of_cows_l165_165346

theorem number_of_cows (C H : ℕ) (hcnd : 4 * C + 2 * H = 2 * (C + H) + 18) :
  C = 9 :=
sorry

end number_of_cows_l165_165346


namespace initial_red_marbles_l165_165791

theorem initial_red_marbles (r g : ℕ) 
  (h1 : r = 5 * g / 3) 
  (h2 : (r - 20) * 5 = g + 40) : 
  r = 317 :=
by
  sorry

end initial_red_marbles_l165_165791


namespace calc_result_l165_165537

theorem calc_result (a : ℤ) : 3 * a - 5 * a + a = -a := by
  sorry

end calc_result_l165_165537


namespace cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l165_165084

noncomputable def cos_sum : ℝ :=
  (Real.cos (2 * Real.pi / 17) +
   Real.cos (6 * Real.pi / 17) +
   Real.cos (8 * Real.pi / 17))

theorem cos_sum_equals_fraction_sqrt_13_minus_1_div_4 :
  cos_sum = (Real.sqrt 13 - 1) / 4 := 
sorry

end cos_sum_equals_fraction_sqrt_13_minus_1_div_4_l165_165084


namespace option_C_true_l165_165582

theorem option_C_true (a b : ℝ):
    (a^2 + b^2 ≥ 2 * a * b) ↔ ((a^2 + b^2 > 2 * a * b) ∨ (a^2 + b^2 = 2 * a * b)) :=
by
  sorry

end option_C_true_l165_165582


namespace children_absent_on_independence_day_l165_165468

theorem children_absent_on_independence_day
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (extra_bananas : ℕ)
  (total_possible_children : total_children = 780)
  (bananas_distributed : bananas_per_child = 2)
  (additional_bananas : extra_bananas = 2) :
  ∃ (A : ℕ), A = 390 := 
sorry

end children_absent_on_independence_day_l165_165468


namespace jessica_total_spent_l165_165017

noncomputable def catToyCost : ℝ := 10.22
noncomputable def cageCost : ℝ := 11.73
noncomputable def totalCost : ℝ := 21.95

theorem jessica_total_spent :
  catToyCost + cageCost = totalCost :=
sorry

end jessica_total_spent_l165_165017


namespace lena_more_than_nicole_l165_165932

theorem lena_more_than_nicole :
  ∀ (L K N : ℝ),
    L = 37.5 →
    (L + 9.5) = 5 * K →
    K = N - 8.5 →
    (L - N) = 19.6 :=
by
  intros L K N hL hLK hK
  sorry

end lena_more_than_nicole_l165_165932


namespace problem_f1_l165_165425

noncomputable def f : ℝ → ℝ := sorry

theorem problem_f1 (h : ∀ x y : ℝ, f x + f (2 * x + y) + 7 * x * y = f (3 * x - y) + 3 * x^2 + 2) : f 10 = -48 :=
sorry

end problem_f1_l165_165425


namespace otimes_identity_l165_165870

def otimes (x y : ℝ) : ℝ := x^2 - y^2

theorem otimes_identity (h : ℝ) : otimes h (otimes h h) = h^2 :=
by
  sorry

end otimes_identity_l165_165870


namespace range_of_ab_l165_165273

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |2 - a^2| = |2 - b^2|) : 0 < a * b ∧ a * b < 2 := by
  sorry

end range_of_ab_l165_165273


namespace max_value_of_x_plus_y_l165_165223

variable (x y : ℝ)

-- Define the condition
def condition : Prop := x^2 + y + 3 * x - 3 = 0

-- Define the proof statement
theorem max_value_of_x_plus_y (hx : condition x y) : x + y ≤ 4 :=
sorry

end max_value_of_x_plus_y_l165_165223


namespace jennifer_money_left_over_l165_165586

theorem jennifer_money_left_over :
  let original_amount := 120
  let sandwich_cost := original_amount / 5
  let museum_ticket_cost := original_amount / 6
  let book_cost := original_amount / 2
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  let money_left := original_amount - total_spent
  money_left = 16 :=
by
  let original_amount := 120
  let sandwich_cost := original_amount / 5
  let museum_ticket_cost := original_amount / 6
  let book_cost := original_amount / 2
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  let money_left := original_amount - total_spent
  exact sorry

end jennifer_money_left_over_l165_165586


namespace floor_double_l165_165109

theorem floor_double (a : ℝ) (h : 0 < a) : 
  ⌊2 * a⌋ = ⌊a⌋ + ⌊a + 1/2⌋ :=
sorry

end floor_double_l165_165109


namespace problem1_problem2_l165_165688

def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b - 1
def f (a b : ℝ) : ℝ := 3 * A a b + 6 * B a b

theorem problem1 (a b : ℝ) : f a b = 15 * a * b - 6 * a - 9 :=
by 
  sorry

theorem problem2 (b : ℝ) : (∀ a : ℝ, f a b = -9) → b = 2 / 5 :=
by 
  sorry

end problem1_problem2_l165_165688


namespace omega_range_l165_165994

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - 1

theorem omega_range (ω : ℝ) 
  (h_pos : 0 < ω) 
  (h_zeros : ∀ x ∈ Set.Icc (0 : ℝ) (2 * Real.pi), 
    Real.cos (ω * x) - 1 = 0 ↔ 
    (∃ k : ℤ, x = (2 * k * Real.pi / ω) ∧ 0 ≤ x ∧ x ≤ 2 * Real.pi)) :
  (2 ≤ ω ∧ ω < 3) :=
by
  sorry

end omega_range_l165_165994


namespace polynomial_expansion_l165_165315

variable (t : ℝ)

theorem polynomial_expansion :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-4 * t^3 + 3 * t - 5) = -12 * t^6 - 8 * t^5 + 25 * t^4 - 21 * t^3 - 22 * t^2 + 29 * t - 15 :=
by {
  sorry
}

end polynomial_expansion_l165_165315


namespace number_of_hydrogen_atoms_l165_165384

theorem number_of_hydrogen_atoms (C_atoms : ℕ) (O_atoms : ℕ) (molecular_weight : ℕ) 
    (C_weight : ℕ) (O_weight : ℕ) (H_weight : ℕ) : C_atoms = 3 → O_atoms = 1 → 
    molecular_weight = 58 → C_weight = 12 → O_weight = 16 → H_weight = 1 → 
    (molecular_weight - (C_atoms * C_weight + O_atoms * O_weight)) / H_weight = 6 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end number_of_hydrogen_atoms_l165_165384


namespace man_rate_still_water_l165_165564

def speed_with_stream : ℝ := 6
def speed_against_stream : ℝ := 2

theorem man_rate_still_water : (speed_with_stream + speed_against_stream) / 2 = 4 := by
  sorry

end man_rate_still_water_l165_165564


namespace algebra_problem_l165_165609

theorem algebra_problem
  (x : ℝ)
  (h : 59 = x^4 + 1 / x^4) :
  x^2 + 1 / x^2 = Real.sqrt 61 :=
sorry

end algebra_problem_l165_165609


namespace sum_of_three_numbers_l165_165866

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end sum_of_three_numbers_l165_165866


namespace john_and_lisa_meet_at_midpoint_l165_165919

-- Define the conditions
def john_position : ℝ × ℝ := (2, 9)
def lisa_position : ℝ × ℝ := (-6, 1)

-- Assertion for their meeting point
theorem john_and_lisa_meet_at_midpoint :
  ∃ (x y : ℝ), (x, y) = ((john_position.1 + lisa_position.1) / 2,
                         (john_position.2 + lisa_position.2) / 2) :=
sorry

end john_and_lisa_meet_at_midpoint_l165_165919


namespace age_difference_is_36_l165_165393

open Nat

theorem age_difference_is_36 (a b : ℕ) (h1 : a < 10) (h2 : b < 10) 
    (h_eq : (10 * a + b) + 8 = 3 * ((10 * b + a) + 8)) :
    (10 * a + b) - (10 * b + a) = 36 :=
by
  sorry

end age_difference_is_36_l165_165393


namespace fraction_pow_four_result_l165_165294

theorem fraction_pow_four_result (x : ℚ) (h : x = 1 / 4) : x ^ 4 = 390625 / 100000000 :=
by sorry

end fraction_pow_four_result_l165_165294


namespace repeating_decimal_sum_l165_165423

noncomputable def repeating_decimal_four : ℚ := 0.44444 -- 0.\overline{4}
noncomputable def repeating_decimal_seven : ℚ := 0.77777 -- 0.\overline{7}

-- Proving that the sum of these repeating decimals is equivalent to the fraction 11/9.
theorem repeating_decimal_sum : repeating_decimal_four + repeating_decimal_seven = 11/9 := by
  -- Placeholder to skip the actual proof
  sorry

end repeating_decimal_sum_l165_165423


namespace age_difference_is_eight_l165_165638

theorem age_difference_is_eight (A B k : ℕ)
  (h1 : A = B + k)
  (h2 : A - 1 = 3 * (B - 1))
  (h3 : A = 2 * B + 3) :
  k = 8 :=
by sorry

end age_difference_is_eight_l165_165638


namespace find_n_l165_165442

-- Defining necessary conditions and declarations
def isThreeDigit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def sumOfDigits (n : ℕ) : ℕ :=
  n / 100 + (n / 10) % 10 + n % 10

def productOfDigits (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

theorem find_n (n : ℕ) (s : ℕ) (p : ℕ) 
  (h1 : isThreeDigit n) 
  (h2 : isPerfectSquare n) 
  (h3 : sumOfDigits n = s) 
  (h4 : productOfDigits n = p) 
  (h5 : 10 ≤ s ∧ s < 100)
  (h6 : ∀ m : ℕ, isThreeDigit m → isPerfectSquare m → sumOfDigits m = s → productOfDigits m = p → (m = n → false))
  (h7 : ∃ m : ℕ, isThreeDigit m ∧ isPerfectSquare m ∧ sumOfDigits m = s ∧ productOfDigits m = p ∧ (∃ k : ℕ, k ≠ m → true)) :
  n = 841 :=
sorry

end find_n_l165_165442


namespace bus_problem_l165_165674

theorem bus_problem : ∀ before_stop after_stop : ℕ, before_stop = 41 → after_stop = 18 → before_stop - after_stop = 23 :=
by
  intros before_stop after_stop h_before h_after
  sorry

end bus_problem_l165_165674


namespace Uncle_Bradley_bills_l165_165098

theorem Uncle_Bradley_bills :
  let total_money := 1000
  let fifty_bills_portion := 3 / 10
  let fifty_bill_value := 50
  let hundred_bill_value := 100
  -- Calculate the number of $50 bills
  let fifty_bills_count := (total_money * fifty_bills_portion) / fifty_bill_value
  -- Calculate the number of $100 bills
  let hundred_bills_count := (total_money * (1 - fifty_bills_portion)) / hundred_bill_value
  -- Calculate the total number of bills
  fifty_bills_count + hundred_bills_count = 13 :=
by 
  -- Note: Proof omitted, as it is not required 
  sorry

end Uncle_Bradley_bills_l165_165098


namespace first_rocket_height_l165_165568

theorem first_rocket_height (h : ℝ) (combined_height : ℝ) (second_rocket_height : ℝ) 
  (H1 : second_rocket_height = 2 * h) 
  (H2 : combined_height = h + second_rocket_height) 
  (H3 : combined_height = 1500) : h = 500 := 
by 
  -- The proof would go here but is not required as per the instruction.
  sorry

end first_rocket_height_l165_165568


namespace marks_lost_per_incorrect_sum_l165_165904

variables (marks_per_correct : ℕ) (total_attempts total_marks correct_sums : ℕ)
variable (marks_per_incorrect : ℕ)
variable (incorrect_sums : ℕ)

def calc_marks_per_incorrect_sum : Prop :=
  marks_per_correct = 3 ∧ 
  total_attempts = 30 ∧ 
  total_marks = 50 ∧ 
  correct_sums = 22 ∧ 
  incorrect_sums = total_attempts - correct_sums ∧ 
  (marks_per_correct * correct_sums) - (marks_per_incorrect * incorrect_sums) = total_marks ∧ 
  marks_per_incorrect = 2

theorem marks_lost_per_incorrect_sum : calc_marks_per_incorrect_sum 3 30 50 22 2 (30 - 22) :=
sorry

end marks_lost_per_incorrect_sum_l165_165904


namespace solve_equation_l165_165707

theorem solve_equation : ∀ x : ℝ, -2 * x + 11 = 0 → x = 11 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l165_165707


namespace true_proposition_l165_165350

open Real

-- Proposition p
def p : Prop := ∀ x > 0, log x + 4 * x ≥ 3

-- Proposition q
def q : Prop := ∃ x > 0, 8 * x + 1 / (2 * x) ≤ 4

theorem true_proposition : ¬ p ∧ q := by
  sorry

end true_proposition_l165_165350


namespace union_of_M_N_l165_165547

-- Define the sets M and N
def M : Set ℕ := {0, 2, 3}
def N : Set ℕ := {1, 3}

-- State the theorem to prove that M ∪ N = {0, 1, 2, 3}
theorem union_of_M_N : M ∪ N = {0, 1, 2, 3} :=
by
  sorry -- Proof goes here

end union_of_M_N_l165_165547


namespace Lagrange_interpol_equiv_x_squared_l165_165045

theorem Lagrange_interpol_equiv_x_squared (a b c x : ℝ)
    (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
    c^2 * ((x - a) * (x - b)) / ((c - a) * (c - b)) +
    b^2 * ((x - a) * (x - c)) / ((b - a) * (b - c)) +
    a^2 * ((x - b) * (x - c)) / ((a - b) * (a - c)) = x^2 := 
    sorry

end Lagrange_interpol_equiv_x_squared_l165_165045


namespace min_sum_a_b2_l165_165692

theorem min_sum_a_b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 1) : a + b ≥ 2 + 2 * Real.sqrt 2 :=
by
  sorry

end min_sum_a_b2_l165_165692


namespace ratio_of_55_to_11_l165_165557

theorem ratio_of_55_to_11 : (55 / 11) = 5 := 
by
  sorry

end ratio_of_55_to_11_l165_165557


namespace largest_y_coordinate_ellipse_l165_165946

theorem largest_y_coordinate_ellipse (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 := 
by
  -- proof to be filled in
  sorry

end largest_y_coordinate_ellipse_l165_165946


namespace july14_2030_is_sunday_l165_165027

-- Define the given condition that July 3, 2030 is a Wednesday. 
def july3_2030_is_wednesday : Prop := true -- Assume the existence and correctness of this statement.

-- Define the proof problem that July 14, 2030 is a Sunday given the above condition.
theorem july14_2030_is_sunday : july3_2030_is_wednesday → (14 % 7 = 0) := 
sorry

end july14_2030_is_sunday_l165_165027


namespace kostya_table_prime_l165_165385

theorem kostya_table_prime {n : ℕ} (hn : n > 3)
  (h : ∀ r s : ℕ, r ≥ 3 → s ≥ 3 → rs - (r + s) ≠ n) : Prime (n + 1) := 
sorry

end kostya_table_prime_l165_165385


namespace smallest_four_digit_palindrome_divisible_by_8_l165_165282

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem smallest_four_digit_palindrome_divisible_by_8 : ∃ (n : ℕ), is_palindrome n ∧ is_four_digit n ∧ divisible_by_8 n ∧ n = 4004 := by
  sorry

end smallest_four_digit_palindrome_divisible_by_8_l165_165282


namespace finish_remaining_work_l165_165117

theorem finish_remaining_work (x y : ℕ) (hx : x = 30) (hy : y = 15) (hy_work_days : y_work_days = 10) :
  x = 10 :=
by
  sorry

end finish_remaining_work_l165_165117


namespace expected_balls_in_original_pos_after_two_transpositions_l165_165025

theorem expected_balls_in_original_pos_after_two_transpositions :
  ∃ (n : ℚ), n = 3.2 := 
sorry

end expected_balls_in_original_pos_after_two_transpositions_l165_165025


namespace bread_products_wasted_l165_165078

theorem bread_products_wasted :
  (50 * 8 - (20 * 5 + 15 * 4 + 10 * 10 * 1.5)) / 1.5 = 60 := by
  -- The proof steps are omitted here
  sorry

end bread_products_wasted_l165_165078


namespace cooler1_water_left_l165_165194

noncomputable def waterLeftInFirstCooler (gallons1 gallons2 : ℝ) (chairs rows : ℕ) (ozSmall ozLarge ozPerGallon : ℝ) : ℝ :=
  let totalChairs := chairs * rows
  let totalSmallOunces := totalChairs * ozSmall
  let initialOunces1 := gallons1 * ozPerGallon
  initialOunces1 - totalSmallOunces

theorem cooler1_water_left :
  waterLeftInFirstCooler 4.5 3.25 12 7 4 8 128 = 240 :=
by
  sorry

end cooler1_water_left_l165_165194


namespace lcm_150_414_l165_165804

theorem lcm_150_414 : Nat.lcm 150 414 = 10350 :=
by
  sorry

end lcm_150_414_l165_165804


namespace whole_process_time_is_6_hours_l165_165229

def folding_time_per_fold : ℕ := 5
def number_of_folds : ℕ := 4
def resting_time_per_rest : ℕ := 75
def number_of_rests : ℕ := 4
def mixing_time : ℕ := 10
def baking_time : ℕ := 30

def total_time_process_in_minutes : ℕ :=
  mixing_time + 
  (folding_time_per_fold * number_of_folds) + 
  (resting_time_per_rest * number_of_rests) + 
  baking_time

def total_time_process_in_hours : ℕ := total_time_process_in_minutes / 60

theorem whole_process_time_is_6_hours :
  total_time_process_in_hours = 6 :=
by sorry

end whole_process_time_is_6_hours_l165_165229


namespace total_bill_l165_165217

def num_adults := 2
def num_children := 5
def cost_per_meal := 3

theorem total_bill : (num_adults + num_children) * cost_per_meal = 21 := 
by 
  sorry

end total_bill_l165_165217


namespace second_year_growth_rate_l165_165031

variable (initial_investment : ℝ) (first_year_growth : ℝ) (additional_investment : ℝ) (final_value : ℝ) (second_year_growth : ℝ)

def calculate_portfolio_value_after_first_year (initial_investment first_year_growth : ℝ) : ℝ :=
  initial_investment * (1 + first_year_growth)

def calculate_new_value_after_addition (value_after_first_year additional_investment : ℝ) : ℝ :=
  value_after_first_year + additional_investment

def calculate_final_value_after_second_year (new_value second_year_growth : ℝ) : ℝ :=
  new_value * (1 + second_year_growth)

theorem second_year_growth_rate 
  (h1 : initial_investment = 80) 
  (h2 : first_year_growth = 0.15) 
  (h3 : additional_investment = 28) 
  (h4 : final_value = 132) : 
  calculate_final_value_after_second_year
    (calculate_new_value_after_addition
      (calculate_portfolio_value_after_first_year initial_investment first_year_growth)
      additional_investment)
    0.1 = final_value := 
  by
  sorry

end second_year_growth_rate_l165_165031


namespace sin_x_eq_x_has_unique_root_in_interval_l165_165885

theorem sin_x_eq_x_has_unique_root_in_interval :
  ∃! x : ℝ, x ∈ Set.Icc (-Real.pi) Real.pi ∧ x = Real.sin x :=
sorry

end sin_x_eq_x_has_unique_root_in_interval_l165_165885


namespace find_ordered_pair_l165_165174

theorem find_ordered_pair (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x = 2 ∧ y = 4) :=
by {
  sorry
}

end find_ordered_pair_l165_165174


namespace units_digit_of_153_base_3_l165_165169

theorem units_digit_of_153_base_3 :
  (153 % 3 ^ 1) = 2 := by
sorry

end units_digit_of_153_base_3_l165_165169


namespace part1_part2_l165_165473

def A (t : ℝ) : Prop :=
  ∀ x : ℝ, (t+2)*x^2 + 2*x + 1 > 0

def B (a x : ℝ) : Prop :=
  (a*x - 1)*(x + a) > 0

theorem part1 (t : ℝ) : A t ↔ t < -1 :=
sorry

theorem part2 (a : ℝ) : (∀ t : ℝ, t < -1 → ∀ x : ℝ, B a x) → (0 ≤ a ∧ a ≤ 1) :=
sorry

end part1_part2_l165_165473


namespace find_integer_l165_165859

theorem find_integer (n : ℕ) (hn1 : n % 20 = 0) (hn2 : 8.2 < (n : ℝ)^(1/3)) (hn3 : (n : ℝ)^(1/3) < 8.3) : n = 560 := sorry

end find_integer_l165_165859


namespace f_seven_point_five_l165_165199

noncomputable def f (x : ℝ) : ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom f_in_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem f_seven_point_five : f 7.5 = -0.5 := by
  sorry

end f_seven_point_five_l165_165199


namespace function_properties_l165_165697

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_properties :
  (∀ x y : ℝ, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → f (x + y) = f x + f y) ∧
  (∀ x : ℝ, x ∈ Set.Ioo 0 2 → f x > 0) →
  (∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f (-x) = -f x) ∧
  f 1 = 3 →
  Set.range f = Set.Icc (-6) 6 :=
sorry

end function_properties_l165_165697


namespace overlapping_segments_length_l165_165521

theorem overlapping_segments_length 
    (total_length : ℝ) 
    (actual_distance : ℝ) 
    (num_overlaps : ℕ) 
    (h1 : total_length = 98) 
    (h2 : actual_distance = 83)
    (h3 : num_overlaps = 6) :
    (total_length - actual_distance) / num_overlaps = 2.5 :=
by
  sorry

end overlapping_segments_length_l165_165521


namespace beach_ball_properties_l165_165158

theorem beach_ball_properties :
  let d : ℝ := 18
  let r : ℝ := d / 2
  let surface_area : ℝ := 4 * π * r^2
  let volume : ℝ := (4 / 3) * π * r^3
  surface_area = 324 * π ∧ volume = 972 * π :=
by
  sorry

end beach_ball_properties_l165_165158


namespace solve_for_x_l165_165235

theorem solve_for_x (x : ℝ) (h : 3*x - 4*x + 5*x = 140) : x = 35 :=
by 
  sorry

end solve_for_x_l165_165235


namespace solve_for_x_l165_165268

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 6 = 13) : x = 35.5 :=
by
  -- Proof steps would go here
  sorry

end solve_for_x_l165_165268


namespace smallest_sum_B_c_l165_165060

theorem smallest_sum_B_c : 
  ∃ (B : ℕ) (c : ℕ), (0 ≤ B ∧ B ≤ 4) ∧ (c ≥ 6) ∧ 31 * B = 4 * (c + 1) ∧ B + c = 8 := 
sorry

end smallest_sum_B_c_l165_165060


namespace total_matches_in_group_l165_165426

theorem total_matches_in_group (n : ℕ) (hn : n = 6) : 2 * (n * (n - 1) / 2) = 30 :=
by
  sorry

end total_matches_in_group_l165_165426


namespace four_times_num_mod_nine_l165_165902

theorem four_times_num_mod_nine (n : ℤ) (h : n % 9 = 4) : (4 * n - 3) % 9 = 4 :=
sorry

end four_times_num_mod_nine_l165_165902


namespace percent_of_class_received_50_to_59_l165_165790

-- Define the frequencies for each score range
def freq_90_to_100 := 5
def freq_80_to_89 := 7
def freq_70_to_79 := 9
def freq_60_to_69 := 8
def freq_50_to_59 := 4
def freq_below_50 := 3

-- Define the total number of students
def total_students := freq_90_to_100 + freq_80_to_89 + freq_70_to_79 + freq_60_to_69 + freq_50_to_59 + freq_below_50

-- Define the frequency of students scoring in the 50%-59% range
def freq_50_to_59_ratio := (freq_50_to_59 : ℚ) / total_students

-- Define the percentage calculation
def percent_50_to_59 := freq_50_to_59_ratio * 100

theorem percent_of_class_received_50_to_59 :
  percent_50_to_59 = 100 / 9 := 
by {
  sorry
}

end percent_of_class_received_50_to_59_l165_165790


namespace max_unsealed_windows_l165_165351

-- Definitions of conditions for the problem
def windows : Nat := 15
def panes : Nat := 15

-- Definition of the matching and selection process conditions
def matched_panes (window pane : Nat) : Prop :=
  pane >= window

-- Proof problem statement
theorem max_unsealed_windows 
  (glazier_approaches_window : ∀ (current_window : Nat), ∃ pane : Nat, pane >= current_window) :
  ∃ (max_unsealed : Nat), max_unsealed = 7 :=
by
  sorry

end max_unsealed_windows_l165_165351


namespace least_possible_value_l165_165504

theorem least_possible_value (x y : ℝ) : 
  ∃ (x y : ℝ), (xy + 1)^2 + (x + y + 1)^2 = 0 := 
sorry

end least_possible_value_l165_165504


namespace perpendicular_transfer_l165_165515

variables {Line Plane : Type} 
variables (a b : Line) (α β : Plane)

def perpendicular (l : Line) (p : Plane) : Prop := sorry
def parallel (l : Line) (p : Plane) : Prop := sorry
def parallel_planes (p1 p2 : Plane) : Prop := sorry

theorem perpendicular_transfer
  (h1 : perpendicular a α)
  (h2 : parallel_planes α β) :
  perpendicular a β := 
sorry

end perpendicular_transfer_l165_165515


namespace min_sum_of_dimensions_l165_165000

theorem min_sum_of_dimensions 
  (a b c : ℕ) 
  (h_pos : a > 0) 
  (h_pos_2 : b > 0) 
  (h_pos_3 : c > 0) 
  (h_even : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) 
  (h_vol : a * b * c = 1806) 
  : a + b + c = 56 :=
sorry

end min_sum_of_dimensions_l165_165000


namespace gcd_min_b_c_l165_165296

theorem gcd_min_b_c (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h1 : Nat.gcd a b = 294) (h2 : Nat.gcd a c = 1155) :
  Nat.gcd b c = 21 :=
sorry

end gcd_min_b_c_l165_165296


namespace exists_prime_seq_satisfying_condition_l165_165818

theorem exists_prime_seq_satisfying_condition :
  ∃ (a : ℕ → ℕ), (∀ n, a n > 0) ∧ (∀ m n, m < n → a m < a n) ∧ 
  (∀ i j, i ≠ j → (i * a j, j * a i) = (i, j)) :=
sorry

end exists_prime_seq_satisfying_condition_l165_165818


namespace x_sq_plus_3x_eq_1_l165_165383

theorem x_sq_plus_3x_eq_1 (x : ℝ) (h : (x^2 + 3*x)^2 + 2*(x^2 + 3*x) - 3 = 0) : x^2 + 3*x = 1 :=
sorry

end x_sq_plus_3x_eq_1_l165_165383


namespace solve_quadratic1_solve_quadratic2_l165_165826

open Real

-- Equation 1
theorem solve_quadratic1 (x : ℝ) : x^2 - 6 * x + 8 = 0 → x = 2 ∨ x = 4 := 
by sorry

-- Equation 2
theorem solve_quadratic2 (x : ℝ) : x^2 - 8 * x + 1 = 0 → x = 4 + sqrt 15 ∨ x = 4 - sqrt 15 := 
by sorry

end solve_quadratic1_solve_quadratic2_l165_165826


namespace finite_parabolas_do_not_cover_plane_l165_165545

theorem finite_parabolas_do_not_cover_plane (parabolas : Finset (ℝ → ℝ)) :
  ¬ (∀ x y : ℝ, ∃ p ∈ parabolas, y < p x) :=
by sorry

end finite_parabolas_do_not_cover_plane_l165_165545


namespace carrie_fourth_day_miles_l165_165541

theorem carrie_fourth_day_miles (d1 d2 d3 d4: ℕ) (charge_interval charges: ℕ) 
  (h1: d1 = 135) 
  (h2: d2 = d1 + 124) 
  (h3: d3 = 159) 
  (h4: charge_interval = 106) 
  (h5: charges = 7):
  d4 = 742 - (d1 + d2 + d3) :=
by
  sorry

end carrie_fourth_day_miles_l165_165541


namespace triangle_internal_angle_60_l165_165868

theorem triangle_internal_angle_60 (A B C : ℝ) (h_sum : A + B + C = 180) : A >= 60 ∨ B >= 60 ∨ C >= 60 :=
sorry

end triangle_internal_angle_60_l165_165868


namespace percent_increase_l165_165637

variable (E : ℝ)

-- Given conditions
def enrollment_1992 := 1.20 * E
def enrollment_1993 := 1.26 * E

-- Theorem to prove
theorem percent_increase :
  ((enrollment_1993 E - enrollment_1992 E) / enrollment_1992 E) * 100 = 5 := by
  sorry

end percent_increase_l165_165637


namespace find_n_mod_60_l165_165115

theorem find_n_mod_60 {x y : ℤ} (hx : x ≡ 45 [ZMOD 60]) (hy : y ≡ 98 [ZMOD 60]) :
  ∃ n, 150 ≤ n ∧ n ≤ 210 ∧ (x - y ≡ n [ZMOD 60]) ∧ n = 187 := by
  sorry

end find_n_mod_60_l165_165115


namespace pump_A_time_l165_165052

theorem pump_A_time (B C A : ℝ) (hB : B = 1/3) (hC : C = 1/6)
(h : (A + B - C) * 0.75 = 0.5) : 1 / A = 2 :=
by
sorry

end pump_A_time_l165_165052


namespace total_students_l165_165911

-- Definitions extracted from the conditions 
def ratio_boys_girls := 8 / 5
def number_of_boys := 128

-- Theorem to prove the total number of students
theorem total_students : 
  (128 + (5 / 8) * 128 = 208) ∧ ((128 : ℝ) * (13 / 8) = 208) :=
by
  sorry

end total_students_l165_165911


namespace total_rainbow_nerds_is_36_l165_165970

def purple_candies : ℕ := 10
def yellow_candies : ℕ := purple_candies + 4
def green_candies : ℕ := yellow_candies - 2
def total_candies : ℕ := purple_candies + yellow_candies + green_candies

theorem total_rainbow_nerds_is_36 : total_candies = 36 := by
  sorry

end total_rainbow_nerds_is_36_l165_165970


namespace option_C_incorrect_l165_165899

variable (a b : ℝ)

theorem option_C_incorrect : ((-a^3)^2 * (-b^2)^3) ≠ (a^6 * b^6) :=
by {
  sorry
}

end option_C_incorrect_l165_165899


namespace solve_for_x_l165_165998

theorem solve_for_x (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : 6 * x^3 + 18 * x^2 * y * z = 3 * x^4 + 6 * x^3 * y * z) : 
  x = 2 := 
by sorry

end solve_for_x_l165_165998


namespace sin_double_angle_value_l165_165758

theorem sin_double_angle_value 
  (α : ℝ) 
  (hα1 : π / 2 < α) 
  (hα2 : α < π)
  (h : 3 * Real.cos (2 * α) = Real.cos (π / 4 + α)) : 
  Real.sin (2 * α) = - 17 / 18 := 
by
  sorry

end sin_double_angle_value_l165_165758


namespace bird_families_flew_away_to_Asia_l165_165681

-- Defining the given conditions
def Total_bird_families_flew_away_for_winter : ℕ := 118
def Bird_families_flew_away_to_Africa : ℕ := 38

-- Proving the main statement
theorem bird_families_flew_away_to_Asia : 
  (Total_bird_families_flew_away_for_winter - Bird_families_flew_away_to_Africa) = 80 :=
by
  sorry

end bird_families_flew_away_to_Asia_l165_165681


namespace remainder_when_divided_by_5_l165_165717

theorem remainder_when_divided_by_5 (n : ℕ) (h1 : n^2 % 5 = 1) (h2 : n^3 % 5 = 4) : n % 5 = 4 :=
sorry

end remainder_when_divided_by_5_l165_165717


namespace welders_that_left_first_day_l165_165716

-- Definitions of conditions
def welders := 12
def days_to_complete_order := 3
def days_remaining_work_after_first_day := 8
def work_done_first_day (r : ℝ) := welders * r * 1
def total_work (r : ℝ) := welders * r * days_to_complete_order

-- Theorem statement
theorem welders_that_left_first_day (r : ℝ) : 
  ∃ x : ℝ, 
    (welders - x) * r * days_remaining_work_after_first_day = total_work r - work_done_first_day r 
    ∧ x = 9 :=
by
  sorry

end welders_that_left_first_day_l165_165716


namespace range_of_f_l165_165196

-- Define the function f
def f (x : ℕ) : ℤ := x^2 - 2 * x

-- Define the domain
def domain : Finset ℕ := {0, 1, 2, 3}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 0, 3}

-- State the theorem
theorem range_of_f : (domain.image f) = expected_range := by
  sorry

end range_of_f_l165_165196


namespace inequality_proof_l165_165498

theorem inequality_proof 
  (a b : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (h1 : a ≤ 2 * b) 
  (h2 : 2 * b ≤ 4 * a) :
  4 * a * b ≤ 2 * (a ^ 2 + b ^ 2) ∧ 2 * (a ^ 2 + b ^ 2) ≤ 5 * a * b := 
by
  sorry

end inequality_proof_l165_165498


namespace p_sufficient_not_necessary_for_q_l165_165452

-- Define the propositions p and q
def is_ellipse (m : ℝ) : Prop := (1 / 4 < m) ∧ (m < 1)
def is_hyperbola (m : ℝ) : Prop := (0 < m) ∧ (m < 1)

-- Define the theorem to prove the relationship between p and q
theorem p_sufficient_not_necessary_for_q (m : ℝ) :
  (is_ellipse m → is_hyperbola m) ∧ ¬(is_hyperbola m → is_ellipse m) :=
sorry

end p_sufficient_not_necessary_for_q_l165_165452


namespace builder_total_amount_paid_l165_165091

theorem builder_total_amount_paid :
  let cost_drill_bits := 5 * 6
  let tax_drill_bits := 0.10 * cost_drill_bits
  let total_cost_drill_bits := cost_drill_bits + tax_drill_bits

  let cost_hammers := 3 * 8
  let discount_hammers := 0.05 * cost_hammers
  let total_cost_hammers := cost_hammers - discount_hammers

  let cost_toolbox := 25
  let tax_toolbox := 0.15 * cost_toolbox
  let total_cost_toolbox := cost_toolbox + tax_toolbox

  let total_amount_paid := total_cost_drill_bits + total_cost_hammers + total_cost_toolbox

  total_amount_paid = 84.55 :=
by
  sorry

end builder_total_amount_paid_l165_165091


namespace num_good_triples_at_least_l165_165388

noncomputable def num_good_triples (S : Finset (ℕ × ℕ)) (n m : ℕ) : ℕ :=
  4 * m * (m - n^2 / 4) / (3 * n)

theorem num_good_triples_at_least
  (S : Finset (ℕ × ℕ))
  (n m : ℕ)
  (h_S : ∀ (x : ℕ × ℕ), x ∈ S → 1 ≤ x.1 ∧ x.1 < x.2 ∧ x.2 ≤ n)
  (h_m : S.card = m)
  : ∃ t ≤ num_good_triples S n m, True := 
sorry

end num_good_triples_at_least_l165_165388


namespace ratio_of_sums_l165_165221

theorem ratio_of_sums (a b c d : ℚ) (h1 : b / a = 3) (h2 : d / b = 4) (h3 : c = (a + b) / 2) :
  (a + b + c) / (b + c + d) = 8 / 17 :=
by
  sorry

end ratio_of_sums_l165_165221


namespace geometric_sequence_problem_l165_165397

theorem geometric_sequence_problem
  (a : ℕ → ℝ) (q : ℝ)
  (h1 : q ≠ 1)
  (h_sum : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 6)
  (h_sum_squares : a 1 ^ 2 + a 2 ^ 2 + a 3 ^ 2 + a 4 ^ 2 + a 5 ^ 2 + a 6 ^ 2 + a 7 ^ 2 = 18)
  (h_geom_seq : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
  a 1 - a 2 + a 3 - a 4 + a 5 - a 6 + a 7 = 3 :=
by sorry

end geometric_sequence_problem_l165_165397


namespace center_of_tangent_circle_l165_165093

theorem center_of_tangent_circle (x y : ℝ) 
    (h1 : 3 * x - 4 * y = 20) 
    (h2 : 3 * x - 4 * y = -40) 
    (h3 : x - 3 * y = 0) : 
    (x, y) = (-6, -2) := 
by
    sorry

end center_of_tangent_circle_l165_165093


namespace min_value_l165_165528

open Real

-- Definitions
variables (a b : ℝ)
axiom a_gt_zero : a > 0
axiom b_gt_one : b > 1
axiom sum_eq : a + b = 3 / 2

-- The theorem to be proved.
theorem min_value (a : ℝ) (b : ℝ) (a_gt_zero : a > 0) (b_gt_one : b > 1) (sum_eq : a + b = 3 / 2) :
  ∃ (m : ℝ), m = 6 + 4 * sqrt 2 ∧ ∀ (x y : ℝ), (x > 0) → (y > 1) → (x + y = 3 / 2) → (∃ (z : ℝ), z = 2 / x + 1 / (y - 1) ∧ z ≥ m) :=
sorry

end min_value_l165_165528


namespace find_teacher_age_l165_165585

/-- Given conditions: 
1. The class initially has 30 students with an average age of 10.
2. One student aged 11 leaves the class.
3. The average age of the remaining 29 students plus the teacher is 11.
Prove that the age of the teacher is 30 years.
-/
theorem find_teacher_age (total_students : ℕ) (avg_age : ℕ) (left_student_age : ℕ) 
  (remaining_avg_age : ℕ) (teacher_age : ℕ) :
  total_students = 30 →
  avg_age = 10 →
  left_student_age = 11 →
  remaining_avg_age = 11 →
  289 + teacher_age = 29 * remaining_avg_age + teacher_age →
  teacher_age = 30 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end find_teacher_age_l165_165585


namespace f_at_2_f_shifted_range_f_shifted_l165_165179

def f (x : ℝ) := x^2 - 2*x + 7

-- 1) Prove that f(2) = 7
theorem f_at_2 : f 2 = 7 := sorry

-- 2) Prove the expressions for f(x-1) and f(x+1)
theorem f_shifted (x : ℝ) : f (x-1) = x^2 - 4*x + 10 ∧ f (x+1) = x^2 + 6 := sorry

-- 3) Prove the range of f(x+1) is [6, +∞)
theorem range_f_shifted : ∀ x, f (x+1) ≥ 6 := sorry

end f_at_2_f_shifted_range_f_shifted_l165_165179


namespace smallest_integer_proof_l165_165724

noncomputable def smallestInteger (s : ℝ) (h : s < 1 / 2000) : ℤ :=
  Nat.ceil (Real.sqrt (1999 / 3))

theorem smallest_integer_proof (s : ℝ) (h : s < 1 / 2000) (m : ℤ) (hm : m = (smallestInteger s h + s)^3) : smallestInteger s h = 26 :=
by 
  sorry

end smallest_integer_proof_l165_165724


namespace problem_statement_l165_165710

theorem problem_statement
  (b1 b2 b3 c1 c2 c3 : ℝ)
  (h : ∀ x : ℝ, x^8 - 3*x^6 + 3*x^4 - x^2 + 2 = 
                 (x^2 + b1*x + c1) * (x^2 + b2*x + c2) * (x^2 + 2*b3*x + c3)) :
  b1 * c1 + b2 * c2 + 2 * b3 * c3 = 0 := 
sorry

end problem_statement_l165_165710


namespace arrangements_7_people_no_A_at_head_no_B_in_middle_l165_165087

theorem arrangements_7_people_no_A_at_head_no_B_in_middle :
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  total_arrangements - 2 * A_at_head + overlap = 3720 :=
by
  let n := 7
  let total_arrangements := Nat.factorial n
  let A_at_head := Nat.factorial (n - 1)
  let B_in_middle := A_at_head
  let overlap := Nat.factorial (n - 2)
  show total_arrangements - 2 * A_at_head + overlap = 3720
  sorry

end arrangements_7_people_no_A_at_head_no_B_in_middle_l165_165087


namespace profit_without_discount_l165_165331

theorem profit_without_discount (CP SP_with_discount SP_without_discount : ℝ) (h1 : CP = 100) (h2 : SP_with_discount = CP + 0.235 * CP) (h3 : SP_with_discount = 0.95 * SP_without_discount) : (SP_without_discount - CP) / CP * 100 = 30 :=
by
  sorry

end profit_without_discount_l165_165331


namespace evaluate_expression_l165_165141

theorem evaluate_expression
  (p q r s : ℚ)
  (h1 : p / q = 4 / 5)
  (h2 : r / s = 3 / 7) :
  (18 / 7) + ((2 * q - p) / (2 * q + p)) - ((3 * s + r) / (3 * s - r)) = 5 / 3 := by
  sorry

end evaluate_expression_l165_165141


namespace contrapositive_statement_l165_165983

-- Definitions derived from conditions
def Triangle (ABC : Type) : Prop := 
  ∃ a b c : ABC, true

def IsIsosceles (ABC : Type) : Prop :=
  ∃ a b c : ABC, a = b ∨ b = c ∨ a = c

def InteriorAnglesNotEqual (ABC : Type) : Prop :=
  ∀ a b : ABC, a ≠ b

-- The contrapositive implication we need to prove
theorem contrapositive_statement (ABC : Type) (h : Triangle ABC) 
  (h_not_isosceles_implies_not_equal : ¬IsIsosceles ABC → InteriorAnglesNotEqual ABC) :
  (∃ a b c : ABC, a = b → IsIsosceles ABC) := 
sorry

end contrapositive_statement_l165_165983


namespace no_real_solution_l165_165971

noncomputable def quadratic_eq (x : ℝ) : ℝ := (2*x^2 - 3*x + 5)

theorem no_real_solution : 
  ∀ x : ℝ, quadratic_eq x ^ 2 + 1 ≠ 1 :=
by
  intro x
  sorry

end no_real_solution_l165_165971


namespace estimate_3_sqrt_2_range_l165_165722

theorem estimate_3_sqrt_2_range :
  4 < 3 * Real.sqrt 2 ∧ 3 * Real.sqrt 2 < 5 :=
by
  sorry

end estimate_3_sqrt_2_range_l165_165722


namespace third_team_cups_l165_165584

theorem third_team_cups (required_cups : ℕ) (first_team : ℕ) (second_team : ℕ) (third_team : ℕ) :
  required_cups = 280 ∧ first_team = 90 ∧ second_team = 120 →
  third_team = required_cups - (first_team + second_team) :=
by
  intro h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end third_team_cups_l165_165584


namespace find_S2_side_length_l165_165165

theorem find_S2_side_length 
    (x r : ℝ)
    (h1 : 2 * r + x = 2100)
    (h2 : 3 * x + 300 = 3500)
    : x = 1066.67 := 
sorry

end find_S2_side_length_l165_165165


namespace divisibility_of_difference_by_9_l165_165536

theorem divisibility_of_difference_by_9 (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) :
  9 ∣ ((10 * a + b) - (10 * b + a)) :=
by {
  -- The problem statement
  sorry
}

end divisibility_of_difference_by_9_l165_165536


namespace part_1_prob_excellent_part_2_rounds_pvalues_l165_165375

-- Definition of the probability of an excellent pair
def prob_excellent (p1 p2 : ℚ) : ℚ :=
  2 * p1 * (1 - p1) * p2 * p2 + p1 * p1 * 2 * p2 * (1 - p2) + p1 * p1 * p2 * p2

-- Part (1) statement: Prove the probability that they achieve "excellent pair" status in the first round
theorem part_1_prob_excellent (p1 p2 : ℚ) (hp1 : p1 = 3/4) (hp2 : p2 = 2/3) :
  prob_excellent p1 p2 = 2/3 := by
  rw [hp1, hp2]
  sorry

-- Part (2) statement: Prove the minimum number of rounds and values of p1 and p2
theorem part_2_rounds_pvalues (n : ℕ) (p1 p2 : ℚ) (h_sum : p1 + p2 = 4/3)
  (h_goal : n * prob_excellent p1 p2 ≥ 16) :
  (n = 27) ∧ (p1 = 2/3) ∧ (p2 = 2/3) := by
  sorry

end part_1_prob_excellent_part_2_rounds_pvalues_l165_165375


namespace minimum_value_l165_165676

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  (Real.sqrt ((x^2 + y^2) * (4 * x^2 + y^2))) / (x * y)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  3 ≤ min_value_expr x y :=
  sorry

end minimum_value_l165_165676


namespace diamond_problem_l165_165295

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_problem : (diamond (diamond 1 2) 3) - (diamond 1 (diamond 2 3)) = -7 / 30 := by
  sorry

end diamond_problem_l165_165295


namespace tangent_line_parabola_l165_165407

theorem tangent_line_parabola (d : ℝ) : 
    (∀ y : ℝ, (-4)^2 - 4 * (y^2 - 4 * y + 4 * d) = 0) ↔ d = 1 :=
by
    sorry

end tangent_line_parabola_l165_165407


namespace cos_of_angle_sum_l165_165042

variable (θ : ℝ)

-- Given condition
axiom sin_theta : Real.sin θ = 1 / 4

-- To prove
theorem cos_of_angle_sum : Real.cos (3 * Real.pi / 2 + θ) = -1 / 4 :=
by
  sorry

end cos_of_angle_sum_l165_165042


namespace flowers_per_bouquet_l165_165358

theorem flowers_per_bouquet (narcissus chrysanthemums bouquets : ℕ) 
  (h1: narcissus = 75) 
  (h2: chrysanthemums = 90) 
  (h3: bouquets = 33) 
  : (narcissus + chrysanthemums) / bouquets = 5 := 
by 
  sorry

end flowers_per_bouquet_l165_165358


namespace evaluate_expression_l165_165831

theorem evaluate_expression:
  let a := 11
  let b := 13
  let c := 17
  (121 * (1/b - 1/c) + 169 * (1/c - 1/a) + 289 * (1/a - 1/b)) / 
  (11 * (1/b - 1/c) + 13 * (1/c - 1/a) + 17 * (1/a - 1/b)) = 41 :=
by
  let a := 11
  let b := 13
  let c := 17
  sorry

end evaluate_expression_l165_165831


namespace probability_product_even_gt_one_fourth_l165_165926

def n := 100
def is_even (x : ℕ) : Prop := x % 2 = 0
def is_odd (x : ℕ) : Prop := ¬ is_even x

theorem probability_product_even_gt_one_fourth :
  (∃ (p : ℝ), p > 0 ∧ p = 1 - (50 * 49 * 48 : ℝ) / (100 * 99 * 98) ∧ p > 1 / 4) :=
sorry

end probability_product_even_gt_one_fourth_l165_165926


namespace a_is_perfect_square_l165_165857

variable (a b : ℕ)
variable (h1 : 0 < a) 
variable (h2 : 0 < b)
variable (h3 : b % 2 = 1)
variable (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b)

theorem a_is_perfect_square (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b % 2 = 1) 
  (h4 : ∃ k : ℕ, (a + b) ^ 2 + 4 * a = k * a * b) : ∃ n : ℕ, a = n ^ 2 :=
sorry

end a_is_perfect_square_l165_165857


namespace problem_part1_problem_part2_problem_part3_l165_165231

noncomputable def find_ab (a b : ℝ) : Prop :=
  (5 * a + b = 40) ∧ (30 * a + b = 140)

noncomputable def production_cost (x : ℕ) : Prop :=
  (4 * x + 20 + 7 * (100 - x) = 660)

noncomputable def transport_cost (m : ℝ) : Prop :=
  ∃ n : ℝ, 10 ≤ n ∧ n ≤ 20 ∧ (m - 2) * n + 130 = 150

theorem problem_part1 : ∃ (a b : ℝ), find_ab a b ∧ a = 4 ∧ b = 20 := 
  sorry

theorem problem_part2 : ∃ (x : ℕ), production_cost x ∧ x = 20 := 
  sorry

theorem problem_part3 : ∃ (m : ℝ), transport_cost m ∧ m = 4 := 
  sorry

end problem_part1_problem_part2_problem_part3_l165_165231


namespace no_valid_n_l165_165010

theorem no_valid_n (n : ℕ) : (100 ≤ n / 4 ∧ n / 4 ≤ 999) → (100 ≤ 4 * n ∧ 4 * n ≤ 999) → false :=
by
  intro h1 h2
  sorry

end no_valid_n_l165_165010


namespace tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l165_165111

theorem tangent_triangle_perimeter_acute (a b c: ℝ) (h1: a^2 + b^2 > c^2) (h2: b^2 + c^2 > a^2) (h3: c^2 + a^2 > b^2) :
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) = 
  2 * a * b * c * (1 / (b^2 + c^2 - a^2) + 1 / (c^2 + a^2 - b^2) + 1 / (a^2 + b^2 - c^2)) := 
by sorry -- proof goes here

theorem tangent_triangle_perimeter_obtuse (a b c: ℝ) (h1: a^2 > b^2 + c^2) :
  2 * a * b * c / (a^2 - b^2 - c^2) = 2 * a * b * c / (a^2 - b^2 - c^2) := 
by sorry -- proof goes here

end tangent_triangle_perimeter_acute_tangent_triangle_perimeter_obtuse_l165_165111


namespace orange_juice_fraction_l165_165999

def capacity_small_pitcher := 500 -- mL
def orange_juice_fraction_small := 1 / 4
def capacity_large_pitcher := 800 -- mL
def orange_juice_fraction_large := 1 / 2

def total_orange_juice_volume := 
  (capacity_small_pitcher * orange_juice_fraction_small) + 
  (capacity_large_pitcher * orange_juice_fraction_large)
def total_volume := capacity_small_pitcher + capacity_large_pitcher

theorem orange_juice_fraction :
  (total_orange_juice_volume / total_volume) = (21 / 52) := 
by 
  sorry

end orange_juice_fraction_l165_165999


namespace pow_mod_remainder_l165_165451

theorem pow_mod_remainder (x : ℕ) (h : x = 3) : x^1988 % 8 = 1 := by
  sorry

end pow_mod_remainder_l165_165451


namespace car_speeds_l165_165472

theorem car_speeds (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v) / 2) :=
sorry

end car_speeds_l165_165472


namespace required_fencing_l165_165982

-- Define constants given in the problem
def L : ℕ := 20
def A : ℕ := 720

-- Define the width W based on the area and the given length L
def W : ℕ := A / L

-- Define the total amount of fencing required
def F : ℕ := 2 * W + L

-- State the theorem that this amount of fencing is equal to 92
theorem required_fencing : F = 92 := by
  sorry

end required_fencing_l165_165982


namespace part1_part2_l165_165713

def setA (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def setB : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem part1 (m : ℝ) (h : m = 1) : 
  {x | x ∈ setA m} ∩ {x | x ∈ setB} = {x | 3 ≤ x ∧ x < 4} :=
by {
  sorry
}

theorem part2 (m : ℝ): 
  ({x | x ∈ setA m} ∪ {x | x ∈ setB} = {x | x ∈ setB}) ↔ (m ≥ 3 ∨ m ≤ -3) :=
by {
  sorry
}

end part1_part2_l165_165713


namespace samantha_born_in_1979_l165_165934

-- Condition definitions
def first_AMC8_year := 1985
def annual_event (n : ℕ) : ℕ := first_AMC8_year + n
def seventh_AMC8_year := annual_event 6

variable (Samantha_age_in_seventh_AMC8 : ℕ)
def Samantha_age_when_seventh_AMC8 := 12
def Samantha_birth_year := seventh_AMC8_year - Samantha_age_when_seventh_AMC8

-- Proof statement
theorem samantha_born_in_1979 : Samantha_birth_year = 1979 :=
by
  sorry

end samantha_born_in_1979_l165_165934


namespace floor_sqrt_80_l165_165561

theorem floor_sqrt_80 : ∀ (x : ℝ), 8 ^ 2 < 80 ∧ 80 < 9 ^ 2 → x = 8 :=
by
  intros x h
  sorry

end floor_sqrt_80_l165_165561


namespace initial_pieces_of_gum_l165_165874

theorem initial_pieces_of_gum (additional_pieces given_pieces leftover_pieces initial_pieces : ℕ)
  (h_additional : additional_pieces = 3)
  (h_given : given_pieces = 11)
  (h_leftover : leftover_pieces = 2)
  (h_initial : initial_pieces + additional_pieces = given_pieces + leftover_pieces) :
  initial_pieces = 10 :=
by
  sorry

end initial_pieces_of_gum_l165_165874


namespace find_abc_l165_165555

noncomputable def a_b_c_exist : Prop :=
  ∃ (a b c : ℝ), 
    (a + b + c = 21/4) ∧ 
    (1/a + 1/b + 1/c = 21/4) ∧ 
    (a * b * c = 1) ∧ 
    (a < b) ∧ (b < c) ∧ 
    (a = 1/4) ∧ (b = 1) ∧ (c = 4)

theorem find_abc : a_b_c_exist :=
sorry

end find_abc_l165_165555


namespace coefficients_equal_l165_165827

theorem coefficients_equal (n : ℕ) (h : n ≥ 6) : 
  (n = 7) ↔ 
  (Nat.choose n 5 * 3 ^ 5 = Nat.choose n 6 * 3 ^ 6) := by
  sorry

end coefficients_equal_l165_165827


namespace age_of_youngest_child_l165_165658

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 50) : x = 4 :=
by
  sorry

end age_of_youngest_child_l165_165658


namespace cosine_value_l165_165736

theorem cosine_value (α : ℝ) (h : Real.sin (α - Real.pi / 3) = 1 / 3) :
  Real.cos (α + Real.pi / 6) = -1 / 3 :=
by
  sorry

end cosine_value_l165_165736


namespace inequality_division_l165_165212

variable {a b c : ℝ}

theorem inequality_division (h1 : a > b) (h2 : b > 0) (h3 : c < 0) : 
  (a / (a - c)) > (b / (b - c)) := 
sorry

end inequality_division_l165_165212


namespace cannot_form_right_triangle_l165_165026

theorem cannot_form_right_triangle (a b c : ℝ) (h₁ : a = 2) (h₂ : b = 2) (h₃ : c = 3) :
  a^2 + b^2 ≠ c^2 :=
by
  rw [h₁, h₂, h₃]
  -- Next step would be to simplify and show the inequality, but we skip the proof
  -- 2^2 + 2^2 = 4 + 4 = 8 
  -- 3^2 = 9 
  -- 8 ≠ 9
  sorry

end cannot_form_right_triangle_l165_165026


namespace inequality_solution_l165_165415

noncomputable def inequality (x : ℝ) : Prop :=
  ((x - 1) * (x - 4) * (x - 5)) / ((x - 2) * (x - 6) * (x - 7)) > 0

theorem inequality_solution :
  {x : ℝ | inequality x} = {x : ℝ | x < 1} ∪ {x : ℝ | 2 < x ∧ x < 4} ∪ {x : ℝ | 5 < x ∧ x < 6} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end inequality_solution_l165_165415


namespace proof_problem_l165_165685

theorem proof_problem (x y : ℚ) : 
  (x ^ 2 - 9 * y ^ 2 = 0) ∧ 
  (x + y = 1) ↔ 
  ((x = 3/4 ∧ y = 1/4) ∨ (x = 3/2 ∧ y = -1/2)) :=
by
  sorry

end proof_problem_l165_165685


namespace relationship_between_a_and_b_l165_165737

theorem relationship_between_a_and_b 
  (a b : ℝ) 
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : Real.exp a + 2 * a = Real.exp b + 3 * b) : 
  a > b :=
sorry

end relationship_between_a_and_b_l165_165737


namespace solve_fractional_equation_l165_165286

theorem solve_fractional_equation (x : ℝ) (h₀ : x ≠ 2 / 3) :
  (3 * x + 2) / (3 * x^2 + 4 * x - 4) = (3 * x) / (3 * x - 2) ↔ x = 1 / 3 ∨ x = -2 := by
  sorry

end solve_fractional_equation_l165_165286


namespace solve_for_w_l165_165729

theorem solve_for_w (w : ℂ) (i : ℂ) (i_squared : i^2 = -1) 
  (h : 3 - i * w = 1 + 2 * i * w) : 
  w = -2 * i / 3 := 
sorry

end solve_for_w_l165_165729


namespace identity_x_squared_minus_y_squared_l165_165602

theorem identity_x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : 
  x^2 - y^2 = 80 := by sorry

end identity_x_squared_minus_y_squared_l165_165602


namespace greatest_integer_not_exceeding_a_l165_165156

theorem greatest_integer_not_exceeding_a (a : ℝ) (h : 3^a + a^3 = 123) : ⌊a⌋ = 4 :=
sorry

end greatest_integer_not_exceeding_a_l165_165156


namespace solution_to_system_of_eqns_l165_165153

theorem solution_to_system_of_eqns (x y z : ℝ) :
  (x = (2 * z ^ 2) / (1 + z ^ 2) ∧ y = (2 * x ^ 2) / (1 + x ^ 2) ∧ z = (2 * y ^ 2) / (1 + y ^ 2)) →
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 1 ∧ z = 1) :=
by sorry

end solution_to_system_of_eqns_l165_165153


namespace michael_total_cost_l165_165925

def rental_fee : ℝ := 20.99
def charge_per_mile : ℝ := 0.25
def miles_driven : ℕ := 299

def total_cost (rental_fee : ℝ) (charge_per_mile : ℝ) (miles_driven : ℕ) : ℝ :=
  rental_fee + (charge_per_mile * miles_driven)

theorem michael_total_cost :
  total_cost rental_fee charge_per_mile miles_driven = 95.74 :=
by
  sorry

end michael_total_cost_l165_165925


namespace shaniqua_style_income_correct_l165_165854

def shaniqua_income_per_style (haircut_income : ℕ) (total_income : ℕ) (number_of_haircuts : ℕ) (number_of_styles : ℕ) : ℕ :=
  (total_income - (number_of_haircuts * haircut_income)) / number_of_styles

theorem shaniqua_style_income_correct :
  shaniqua_income_per_style 12 221 8 5 = 25 :=
by
  sorry

end shaniqua_style_income_correct_l165_165854


namespace simplify_fraction_addition_l165_165807

theorem simplify_fraction_addition (a b : ℚ) (h1 : a = 4 / 252) (h2 : b = 17 / 36) :
  a + b = 41 / 84 := 
by
  sorry

end simplify_fraction_addition_l165_165807


namespace total_spending_eq_total_is_19_l165_165265

variable (friend_spending your_spending total_spending : ℕ)

-- Conditions
def friend_spending_eq : friend_spending = 11 := by sorry
def friend_spent_more : friend_spending = your_spending + 3 := by sorry

-- Proof that total_spending is 19
theorem total_spending_eq : total_spending = friend_spending + your_spending :=
  by sorry

theorem total_is_19 : total_spending = 19 :=
  by sorry

end total_spending_eq_total_is_19_l165_165265


namespace initial_amounts_l165_165089

theorem initial_amounts (x y z : ℕ) (h1 : x + y + z = 24)
  (h2 : z = 24 - x - y)
  (h3 : x - (y + z) = 8)
  (h4 : y - (x + z) = 12) :
  x = 13 ∧ y = 7 ∧ z = 4 :=
by
  sorry

end initial_amounts_l165_165089


namespace time_to_pass_platform_l165_165851

-- Conditions of the problem
def length_of_train : ℕ := 1500
def time_to_cross_tree : ℕ := 100
def length_of_platform : ℕ := 500

-- Derived values according to solution steps
def speed_of_train : ℚ := length_of_train / time_to_cross_tree
def total_distance_to_pass_platform : ℕ := length_of_train + length_of_platform

-- The theorem to be proved
theorem time_to_pass_platform :
  (total_distance_to_pass_platform / speed_of_train : ℚ) = 133.33 := sorry

end time_to_pass_platform_l165_165851


namespace domain_of_sqrt_expression_l165_165833

def isDomain (x : ℝ) : Prop := x ≥ -3 ∧ x < 7

theorem domain_of_sqrt_expression : 
  { x : ℝ | isDomain x } = { x | x ≥ -3 ∧ x < 7 } :=
by
  sorry

end domain_of_sqrt_expression_l165_165833


namespace arithmetic_sequence_a6_l165_165665

theorem arithmetic_sequence_a6 (a : ℕ → ℝ) 
  (h₀ : ∀ n : ℕ, a n = a 0 + n * (a 1 - a 0))
  (h₁ : ∃ x y : ℝ, x = a 4 ∧ y = a 8 ∧ (x^2 - 4 * x - 1 = 0) ∧ (y^2 - 4 * y - 1 = 0) ∧ (x + y = 4)) :
  a 6 = 2 := 
sorry

end arithmetic_sequence_a6_l165_165665


namespace average_birds_monday_l165_165660

variable (M : ℕ)

def avg_birds_monday (M : ℕ) : Prop :=
  let total_sites := 5 + 5 + 10
  let total_birds := 5 * M + 5 * 5 + 10 * 8
  (total_birds = total_sites * 7)

theorem average_birds_monday (M : ℕ) (h : avg_birds_monday M) : M = 7 := by
  sorry

end average_birds_monday_l165_165660


namespace kiki_scarves_count_l165_165631

variable (money : ℝ) (scarf_cost : ℝ) (hat_spending_ratio : ℝ) (scarves : ℕ) (hats : ℕ)

-- Condition: Kiki has $90.
axiom kiki_money : money = 90

-- Condition: Kiki spends 60% of her money on hats.
axiom kiki_hat_spending_ratio : hat_spending_ratio = 0.60

-- Condition: Each scarf costs $2.
axiom scarf_price : scarf_cost = 2

-- Condition: Kiki buys twice as many hats as scarves.
axiom hat_scarf_relationship : hats = 2 * scarves

theorem kiki_scarves_count 
  (kiki_money : money = 90)
  (kiki_hat_spending_ratio : hat_spending_ratio = 0.60)
  (scarf_price : scarf_cost = 2)
  (hat_scarf_relationship : hats = 2 * scarves)
  : scarves = 18 := 
sorry

end kiki_scarves_count_l165_165631


namespace values_of_a_l165_165558

noncomputable def M : Set ℝ := {x | x^2 = 1}

noncomputable def N (a : ℝ) : Set ℝ := 
  if a = 0 then ∅ else {x | a * x = 1}

theorem values_of_a (a : ℝ) : (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := by
  sorry

end values_of_a_l165_165558


namespace minimum_value_of_a_squared_plus_b_squared_l165_165034

def quadratic (a b x : ℝ) : ℝ := a * x^2 + (2 * b + 1) * x - a - 2

theorem minimum_value_of_a_squared_plus_b_squared (a b : ℝ) (hab : a ≠ 0)
  (hroot : ∃ (x : ℝ), 3 ≤ x ∧ x ≤ 4 ∧ quadratic a b x = 0) :
  a^2 + b^2 = 1 / 100 :=
sorry

end minimum_value_of_a_squared_plus_b_squared_l165_165034


namespace simple_interest_correct_l165_165633

-- Define the parameters
def principal : ℝ := 10000
def rate_decimal : ℝ := 0.04
def time_years : ℝ := 1

-- Define the simple interest calculation function
noncomputable def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Prove that the simple interest is equal to $400
theorem simple_interest_correct : simple_interest principal rate_decimal time_years = 400 :=
by
  -- Placeholder for the proof
  sorry

end simple_interest_correct_l165_165633


namespace gcd_1337_382_l165_165485

theorem gcd_1337_382 : Nat.gcd 1337 382 = 191 := by
  sorry

end gcd_1337_382_l165_165485


namespace cafeteria_orders_green_apples_l165_165475

theorem cafeteria_orders_green_apples (G : ℕ) (h1 : 6 + G = 5 + 16) : G = 15 :=
by
  sorry

end cafeteria_orders_green_apples_l165_165475


namespace book_price_increase_percentage_l165_165544

theorem book_price_increase_percentage :
  let P_original := 300
  let P_new := 480
  (P_new - P_original : ℝ) / P_original * 100 = 60 :=
by
  sorry

end book_price_increase_percentage_l165_165544


namespace integer_value_of_fraction_l165_165845

theorem integer_value_of_fraction (m n p : ℕ) (hm_diff: m ≠ n) (hn_diff: n ≠ p) (hp_diff: m ≠ p) 
  (hm_range: 2 ≤ m ∧ m ≤ 9) (hn_range: 2 ≤ n ∧ n ≤ 9) (hp_range: 2 ≤ p ∧ p ≤ 9) :
  (m + n + p) / (m + n) = 2 :=
by
  sorry

end integer_value_of_fraction_l165_165845


namespace gcd_7429_13356_l165_165503

theorem gcd_7429_13356 : Nat.gcd 7429 13356 = 1 := by
  sorry

end gcd_7429_13356_l165_165503


namespace emily_final_score_l165_165996

theorem emily_final_score :
  16 + 33 - 48 = 1 :=
by
  -- proof skipped
  sorry

end emily_final_score_l165_165996


namespace not_kth_power_l165_165950

theorem not_kth_power (m k : ℕ) (hk : k > 1) : ¬ ∃ a : ℤ, m * (m + 1) = a^k :=
by
  sorry

end not_kth_power_l165_165950


namespace weekend_price_of_coat_l165_165279

-- Definitions based on conditions
def original_price : ℝ := 250
def sale_price_discount : ℝ := 0.4
def weekend_additional_discount : ℝ := 0.3

-- To prove the final weekend price
theorem weekend_price_of_coat :
  (original_price * (1 - sale_price_discount) * (1 - weekend_additional_discount)) = 105 := by
  sorry

end weekend_price_of_coat_l165_165279


namespace textopolis_word_count_l165_165068

theorem textopolis_word_count :
  let alphabet_size := 26
  let total_one_letter := 2 -- only "A" and "B"
  let total_two_letter := alphabet_size^2
  let excl_two_letter := (alphabet_size - 2)^2
  let total_three_letter := alphabet_size^3
  let excl_three_letter := (alphabet_size - 2)^3
  let total_four_letter := alphabet_size^4
  let excl_four_letter := (alphabet_size - 2)^4
  let valid_two_letter := total_two_letter - excl_two_letter
  let valid_three_letter := total_three_letter - excl_three_letter
  let valid_four_letter := total_four_letter - excl_four_letter
  2 + valid_two_letter + valid_three_letter + valid_four_letter = 129054 := by
  -- To be proved
  sorry

end textopolis_word_count_l165_165068


namespace remaining_lawn_area_l165_165967

theorem remaining_lawn_area (lawn_length lawn_width path_width : ℕ) 
  (h_lawn_length : lawn_length = 10) 
  (h_lawn_width : lawn_width = 5) 
  (h_path_width : path_width = 1) : 
  (lawn_length * lawn_width - lawn_length * path_width) = 40 := 
by 
  sorry

end remaining_lawn_area_l165_165967


namespace square_sum_inverse_eq_23_l165_165792

theorem square_sum_inverse_eq_23 {x : ℝ} (h : x + 1/x = 5) : x^2 + (1/x)^2 = 23 :=
by
  sorry

end square_sum_inverse_eq_23_l165_165792


namespace find_p_l165_165889

theorem find_p (m n p : ℝ)
  (h1 : m = 4 * n + 5)
  (h2 : m + 2 = 4 * (n + p) + 5) : 
  p = 1 / 2 :=
sorry

end find_p_l165_165889


namespace ted_gathered_10_blue_mushrooms_l165_165924

noncomputable def blue_mushrooms_ted_gathered : ℕ :=
  let bill_red_mushrooms := 12
  let bill_brown_mushrooms := 6
  let ted_green_mushrooms := 14
  let total_white_spotted_mushrooms := 17
  
  let bill_white_spotted_red_mushrooms := bill_red_mushrooms / 2
  let bill_white_spotted_brown_mushrooms := bill_brown_mushrooms

  let total_bill_white_spotted_mushrooms := bill_white_spotted_red_mushrooms + bill_white_spotted_brown_mushrooms
  let ted_white_spotted_mushrooms := total_white_spotted_mushrooms - total_bill_white_spotted_mushrooms

  ted_white_spotted_mushrooms * 2

theorem ted_gathered_10_blue_mushrooms :
  blue_mushrooms_ted_gathered = 10 :=
by
  sorry

end ted_gathered_10_blue_mushrooms_l165_165924


namespace arithmetic_progression_sum_l165_165680

-- Define the sum of the first 15 terms of the arithmetic progression
theorem arithmetic_progression_sum (a d : ℝ) 
  (h : (a + 3 * d) + (a + 11 * d) = 16) :
  (15 / 2) * (2 * a + 14 * d) = 120 := by
  sorry

end arithmetic_progression_sum_l165_165680


namespace parabola_focus_eq_l165_165594

theorem parabola_focus_eq (focus : ℝ × ℝ) (hfocus : focus = (0, 1)) :
  ∃ (p : ℝ), p = 1 ∧ ∀ (x y : ℝ), x^2 = 4 * p * y → x^2 = 4 * y :=
by { sorry }

end parabola_focus_eq_l165_165594


namespace hydrogen_moles_formed_l165_165751

open Function

-- Define types for the substances involved in the reaction
structure Substance :=
  (name : String)
  (moles : ℕ)

-- Define the reaction
def reaction (NaH H2O NaOH H2 : Substance) : Prop :=
  NaH.moles = H2O.moles ∧ NaOH.moles = H2.moles

-- Given conditions
def NaH_initial : Substance := ⟨"NaH", 2⟩
def H2O_initial : Substance := ⟨"H2O", 2⟩
def NaOH_final : Substance := ⟨"NaOH", 2⟩
def H2_final : Substance := ⟨"H2", 2⟩

-- Problem statement in Lean
theorem hydrogen_moles_formed :
  reaction NaH_initial H2O_initial NaOH_final H2_final → H2_final.moles = 2 :=
by
  -- Skip proof
  sorry

end hydrogen_moles_formed_l165_165751


namespace pupils_in_class_l165_165043

theorem pupils_in_class (n : ℕ) (wrong_entry_increase : n * (1/2) = 13) : n = 26 :=
sorry

end pupils_in_class_l165_165043


namespace curves_intersect_at_three_points_l165_165619

theorem curves_intersect_at_three_points :
  (∀ x y a : ℝ, (x^2 + y^2 = 4 * a^2) ∧ (y = x^2 - 2 * a) → a = 1) := sorry

end curves_intersect_at_three_points_l165_165619


namespace tetrahedron_three_edges_form_triangle_l165_165873

-- Defining a tetrahedron
structure Tetrahedron := (A B C D : ℝ)
-- length of edges - since it's a geometry problem using the absolute value
def edge_length (x y : ℝ) := abs (x - y)

theorem tetrahedron_three_edges_form_triangle (T : Tetrahedron) :
  ∃ v : ℕ, ∃ e1 e2 e3 : ℝ, 
    (edge_length T.A T.B = e1 ∨ edge_length T.A T.C = e1 ∨ edge_length T.A T.D = e1) ∧ 
    (edge_length T.B T.C = e2 ∨ edge_length T.B T.D = e2 ∨ edge_length T.C T.D = e2) ∧
    (edge_length T.A T.B < e2 + e3 ∧ edge_length T.B T.C < e1 + e3 ∧ edge_length T.C T.D < e1 + e2) := 
sorry

end tetrahedron_three_edges_form_triangle_l165_165873


namespace half_angle_in_second_quadrant_l165_165168

theorem half_angle_in_second_quadrant (α : ℝ) (h : 180 < α ∧ α < 270) : 90 < α / 2 ∧ α / 2 < 135 := 
by
  sorry

end half_angle_in_second_quadrant_l165_165168


namespace factor_polynomial_int_l165_165772

theorem factor_polynomial_int : 
  ∀ x : ℤ, 5 * (x + 3) * (x + 7) * (x + 9) * (x + 11) - 4 * x^2 = 
           (5 * x^2 + 81 * x + 315) * (x^2 + 16 * x + 213) := 
by
  intros
  norm_num
  sorry

end factor_polynomial_int_l165_165772


namespace green_eyed_snack_min_l165_165774

variable {total_count green_eyes_count snack_bringers_count : ℕ}

def least_green_eyed_snack_bringers (total_count green_eyes_count snack_bringers_count : ℕ) : ℕ :=
  green_eyes_count - (total_count - snack_bringers_count)

theorem green_eyed_snack_min 
  (h_total : total_count = 35)
  (h_green_eyes : green_eyes_count = 18)
  (h_snack_bringers : snack_bringers_count = 24)
  : least_green_eyed_snack_bringers total_count green_eyes_count snack_bringers_count = 7 :=
by
  rw [h_total, h_green_eyes, h_snack_bringers]
  unfold least_green_eyed_snack_bringers
  norm_num

end green_eyed_snack_min_l165_165774


namespace k_plus_alpha_is_one_l165_165979

variable (f : ℝ → ℝ) (k α : ℝ)

-- Conditions from part a)
def power_function := ∀ x : ℝ, f x = k * x ^ α
def passes_through_point := f (1 / 2) = 2

-- Statement to be proven
theorem k_plus_alpha_is_one (h1 : power_function f k α) (h2 : passes_through_point f) : k + α = 1 :=
sorry

end k_plus_alpha_is_one_l165_165979


namespace isosceles_triangle_base_length_l165_165771

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 7) (h2 : b + 2 * a = 25) : b = 11 := by
  sorry

end isosceles_triangle_base_length_l165_165771


namespace distance_and_speed_l165_165986

-- Define the conditions given in the problem
def first_car_speed (y : ℕ) := y + 4
def second_car_speed (y : ℕ) := y
def third_car_speed (y : ℕ) := y - 6

def time_relation1 (x : ℕ) (y : ℕ) :=
  x / (first_car_speed y) = x / (second_car_speed y) - 3 / 60

def time_relation2 (x : ℕ) (y : ℕ) :=
  x / (second_car_speed y) = x / (third_car_speed y) - 5 / 60 

-- State the theorem to prove both the distance and the speed of the second car
theorem distance_and_speed : ∃ (x y : ℕ), 
  time_relation1 x y ∧ 
  time_relation2 x y ∧ 
  x = 120 ∧ 
  y = 96 :=
by
  sorry

end distance_and_speed_l165_165986


namespace residue_of_neg_1001_mod_37_l165_165913

theorem residue_of_neg_1001_mod_37 : (-1001 : ℤ) % 37 = 35 :=
by
  sorry

end residue_of_neg_1001_mod_37_l165_165913


namespace horse_food_needed_l165_165318

theorem horse_food_needed
  (ratio_sheep_horses : ℕ := 6)
  (ratio_horses_sheep : ℕ := 7)
  (horse_food_per_day : ℕ := 230)
  (sheep_on_farm : ℕ := 48)
  (units : ℕ := sheep_on_farm / ratio_sheep_horses)
  (horses_on_farm : ℕ := units * ratio_horses_sheep) :
  horses_on_farm * horse_food_per_day = 12880 := by
  sorry

end horse_food_needed_l165_165318


namespace sum_of_squares_of_rates_l165_165072

variable (b j s : ℕ)

theorem sum_of_squares_of_rates
  (h1 : 3 * b + 2 * j + 3 * s = 82)
  (h2 : 5 * b + 3 * j + 2 * s = 99) :
  b^2 + j^2 + s^2 = 314 := by
  sorry

end sum_of_squares_of_rates_l165_165072


namespace gardening_project_total_cost_l165_165891

noncomputable def cost_gardening_project : ℕ := 
  let number_rose_bushes := 20
  let cost_per_rose_bush := 150
  let cost_fertilizer_per_bush := 25
  let gardener_work_hours := [6, 5, 4, 7]
  let gardener_hourly_rate := 30
  let soil_amount := 100
  let cost_per_cubic_foot := 5

  let cost_roses := number_rose_bushes * cost_per_rose_bush
  let cost_fertilizer := number_rose_bushes * cost_fertilizer_per_bush
  let total_work_hours := List.sum gardener_work_hours
  let cost_labor := total_work_hours * gardener_hourly_rate
  let cost_soil := soil_amount * cost_per_cubic_foot

  cost_roses + cost_fertilizer + cost_labor + cost_soil

theorem gardening_project_total_cost : cost_gardening_project = 4660 := by
  sorry

end gardening_project_total_cost_l165_165891


namespace tens_digit_of_13_pow_2021_l165_165047

theorem tens_digit_of_13_pow_2021 :
  let p := 2021
  let base := 13
  let mod_val := 100
  let digit := (base^p % mod_val) / 10
  digit = 1 := by
  sorry

end tens_digit_of_13_pow_2021_l165_165047


namespace cost_of_later_purchase_l165_165461

-- Define the costs of bats and balls as constants.
def cost_of_bat : ℕ := 500
def cost_of_ball : ℕ := 100

-- Define the quantities involved in the later purchase.
def bats_purchased_later : ℕ := 3
def balls_purchased_later : ℕ := 5

-- Define the expected total cost for the later purchase.
def expected_total_cost_later : ℕ := 2000

-- The theorem to be proved: the cost of the later purchase of bats and balls is $2000.
theorem cost_of_later_purchase :
  bats_purchased_later * cost_of_bat + balls_purchased_later * cost_of_ball = expected_total_cost_later :=
sorry

end cost_of_later_purchase_l165_165461


namespace find_c_l165_165355

theorem find_c 
  (a b c : ℝ) 
  (h_vertex : ∀ x y, y = a * x^2 + b * x + c → 
    (∃ k l, l = b / (2 * a) ∧ k = a * l^2 + b * l + c ∧ k = 3 ∧ l = -2))
  (h_pass : ∀ x y, y = a * x^2 + b * x + c → 
    (x = 2 ∧ y = 7)) : c = 4 :=
by sorry

end find_c_l165_165355


namespace cupcakes_left_l165_165364

def num_packages : ℝ := 3.5
def cupcakes_per_package : ℝ := 7
def cupcakes_eaten : ℝ := 5.75

theorem cupcakes_left :
  num_packages * cupcakes_per_package - cupcakes_eaten = 18.75 :=
by
  sorry

end cupcakes_left_l165_165364


namespace charlie_paints_140_square_feet_l165_165579

-- Define the conditions
def total_area : ℕ := 320
def ratio_allen : ℕ := 4
def ratio_ben : ℕ := 5
def ratio_charlie : ℕ := 7
def total_parts : ℕ := ratio_allen + ratio_ben + ratio_charlie
def area_per_part := total_area / total_parts
def charlie_parts := 7

-- Prove the main statement
theorem charlie_paints_140_square_feet : charlie_parts * area_per_part = 140 := by
  sorry

end charlie_paints_140_square_feet_l165_165579


namespace chord_length_cube_l165_165958

noncomputable def diameter : ℝ := 1
noncomputable def AC (a : ℝ) : ℝ := a
noncomputable def AD (b : ℝ) : ℝ := b
noncomputable def AE (a b : ℝ) : ℝ := (a^2 + b^2).sqrt / 2
noncomputable def AF (b : ℝ) : ℝ := b^2

theorem chord_length_cube (a b : ℝ) (h : AE a b = b^2) : a = b^3 :=
by
  sorry

end chord_length_cube_l165_165958


namespace find_interest_rate_l165_165702

noncomputable def annual_interest_rate (P A : ℝ) (n : ℕ) (t r : ℝ) : Prop :=
  A = P * (1 + r / n)^(n * t)

theorem find_interest_rate :
  annual_interest_rate 5000 5100.50 4 0.5 0.04 :=
by
  sorry

end find_interest_rate_l165_165702


namespace two_digit_number_is_42_l165_165188

theorem two_digit_number_is_42 (a b : ℕ) (ha : a < 10) (hb : b < 10) (h : 10 * a + b = 42) :
  ((10 * a + b) : ℚ) / (10 * b + a) = 7 / 4 := by
  sorry

end two_digit_number_is_42_l165_165188


namespace simplify_expression_l165_165548

theorem simplify_expression :
  (-2 : ℝ) ^ 2005 + (-2) ^ 2006 + (3 : ℝ) ^ 2007 - (2 : ℝ) ^ 2008 =
  -7 * (2 : ℝ) ^ 2005 + (3 : ℝ) ^ 2007 := 
by
    sorry

end simplify_expression_l165_165548


namespace transformed_parabola_equation_l165_165316

-- Conditions
def original_parabola (x : ℝ) : ℝ := 3 * x^2
def translate_downwards (y : ℝ) : ℝ := y - 3

-- Translations
def translate_to_right (x : ℝ) : ℝ := x - 2
def transformed_parabola (x : ℝ) : ℝ := 3 * (x - 2)^2 - 3

-- Assertion
theorem transformed_parabola_equation :
  (∀ x : ℝ, translate_downwards (original_parabola x) = 3 * (translate_to_right x)^2 - 3) := by
  sorry

end transformed_parabola_equation_l165_165316


namespace new_oranges_added_l165_165455

-- Defining the initial conditions
def initial_oranges : Nat := 40
def thrown_away_oranges : Nat := 37
def total_oranges_now : Nat := 10
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges := total_oranges_now - remaining_oranges

-- The theorem we want to prove
theorem new_oranges_added : new_oranges = 7 := by
  sorry

end new_oranges_added_l165_165455


namespace inequality_solution_l165_165414

theorem inequality_solution (a x : ℝ) (h₁ : 0 < a) : 
  (0 < a ∧ a < 1 → 2 < x ∧ x < (a-2)/(a-1) → (a * (x - 1)) / (x-2) > 1) ∧ 
  (a = 1 → 2 < x → (a * (x - 1)) / (x-2) > 1 ∧ true) ∧ 
  (a > 1 → (2 < x ∨ x < (a-2)/(a-1)) → (a * (x - 1)) / (x-2) > 1) := 
sorry

end inequality_solution_l165_165414


namespace ratio_of_lateral_edges_l165_165118

theorem ratio_of_lateral_edges (A B : ℝ) (hA : A > 0) (hB : B > 0) (h : A / B = 4 / 9) : 
  let upper_length_ratio := 2
  let lower_length_ratio := 3
  upper_length_ratio / lower_length_ratio = 2 / 3 :=
by 
  sorry

end ratio_of_lateral_edges_l165_165118


namespace fraction_meaningful_l165_165580

theorem fraction_meaningful (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) :=
by
  sorry

end fraction_meaningful_l165_165580


namespace solve_sqrt_eq_l165_165056

theorem solve_sqrt_eq (x : ℝ) :
  (Real.sqrt ((3 + 2 * Real.sqrt 2)^x) + Real.sqrt ((3 - 2 * Real.sqrt 2)^x) = 5) ↔ (x = 2 ∨ x = -2) := by
  sorry

end solve_sqrt_eq_l165_165056


namespace problem1_l165_165051

variable (x y : ℝ)
variable (h1 : x = Real.sqrt 3 + Real.sqrt 5)
variable (h2 : y = Real.sqrt 3 - Real.sqrt 5)

theorem problem1 : 2 * x^2 - 4 * x * y + 2 * y^2 = 40 :=
by sorry

end problem1_l165_165051


namespace decimal_to_base7_l165_165459

-- Define the decimal number
def decimal_number : ℕ := 2011

-- Define the base-7 conversion function
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else to_base7 (n / 7) ++ [n % 7]

-- Calculate the base-7 representation of 2011
def base7_representation : List ℕ := to_base7 decimal_number

-- Prove that the base-7 representation of 2011 is [5, 6, 0, 2]
theorem decimal_to_base7 : base7_representation = [5, 6, 0, 2] :=
  by sorry

end decimal_to_base7_l165_165459


namespace repeating_decimal_to_fraction_l165_165519

noncomputable def x : ℚ := 0.6 + 41 / 990  

theorem repeating_decimal_to_fraction (h : x = 0.6 + 41 / 990) : x = 127 / 198 :=
by sorry

end repeating_decimal_to_fraction_l165_165519


namespace common_ratio_of_geometric_series_l165_165401

noncomputable def geometric_series_common_ratio (a S : ℝ) : ℝ := 1 - (a / S)

theorem common_ratio_of_geometric_series :
  geometric_series_common_ratio 520 3250 = 273 / 325 :=
by
  sorry

end common_ratio_of_geometric_series_l165_165401


namespace johnny_age_multiple_l165_165953

theorem johnny_age_multiple
  (current_age : ℕ)
  (age_in_2_years : ℕ)
  (age_3_years_ago : ℕ)
  (k : ℕ)
  (h1 : current_age = 8)
  (h2 : age_in_2_years = current_age + 2)
  (h3 : age_3_years_ago = current_age - 3)
  (h4 : age_in_2_years = k * age_3_years_ago) :
  k = 2 :=
by
  sorry

end johnny_age_multiple_l165_165953


namespace range_of_a_l165_165495

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x^3 - 3 * a^2 * x + 1 ≠ 3)) 
  → (-1 < a ∧ a < 1) := 
by
  sorry

end range_of_a_l165_165495


namespace int_squares_l165_165596

theorem int_squares (n : ℕ) (h : ∃ k : ℕ, n^4 - n^3 + 3 * n^2 + 5 = k^2) : n = 2 := by
  sorry

end int_squares_l165_165596


namespace sin_half_angle_l165_165494

theorem sin_half_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_angle_l165_165494


namespace parametric_function_f_l165_165301

theorem parametric_function_f (f : ℚ → ℚ)
  (x y : ℝ) (t : ℚ) :
  y = 20 * t - 10 →
  y = (3 / 4 : ℝ) * x - 15 →
  x = f t →
  f t = (80 / 3) * t + 20 / 3 :=
by
  sorry

end parametric_function_f_l165_165301


namespace Barbara_spent_46_22_on_different_goods_l165_165059

theorem Barbara_spent_46_22_on_different_goods :
  let tuna_cost := (5 * 2) -- Total cost of tuna
  let water_cost := (4 * 1.5) -- Total cost of water
  let total_before_discount := 56 / 0.9 -- Total before discount, derived from the final amount paid after discount
  let total_tuna_water_cost := 10 + 6 -- Total cost of tuna and water together
  let different_goods_cost := total_before_discount - total_tuna_water_cost
  different_goods_cost = 46.22 := 
sorry

end Barbara_spent_46_22_on_different_goods_l165_165059


namespace division_and_subtraction_l165_165588

theorem division_and_subtraction : (23 ^ 11 / 23 ^ 8) - 15 = 12152 := by
  sorry

end division_and_subtraction_l165_165588


namespace sales_tax_paid_l165_165470

theorem sales_tax_paid 
  (total_spent : ℝ) 
  (tax_free_cost : ℝ) 
  (tax_rate : ℝ) 
  (cost_of_taxable_items : ℝ) 
  (sales_tax : ℝ) 
  (h1 : total_spent = 40) 
  (h2 : tax_free_cost = 34.7) 
  (h3 : tax_rate = 0.06) 
  (h4 : cost_of_taxable_items = 5) 
  (h5 : sales_tax = 0.3) 
  (h6 : 1.06 * cost_of_taxable_items + tax_free_cost = total_spent) : 
  sales_tax = tax_rate * cost_of_taxable_items :=
sorry

end sales_tax_paid_l165_165470


namespace man_rate_in_still_water_l165_165155

theorem man_rate_in_still_water (V_m V_s: ℝ) 
(h1 : V_m + V_s = 19) 
(h2 : V_m - V_s = 11) : 
V_m = 15 := 
by
  sorry

end man_rate_in_still_water_l165_165155


namespace like_terms_exponents_l165_165961

theorem like_terms_exponents (m n : ℤ) (h1 : 2 * n - 1 = m) (h2 : m = 3) : m = 3 ∧ n = 2 :=
by
  sorry

end like_terms_exponents_l165_165961


namespace nearby_island_banana_production_l165_165615

theorem nearby_island_banana_production
  (x : ℕ)
  (h_prod: 10 * x + x = 99000) :
  x = 9000 :=
sorry

end nearby_island_banana_production_l165_165615


namespace total_amount_paid_l165_165028

theorem total_amount_paid (cost_lunch : ℝ) (sales_tax_rate : ℝ) (tip_rate : ℝ) (sales_tax : ℝ) (tip : ℝ) 
  (h1 : cost_lunch = 100) 
  (h2 : sales_tax_rate = 0.04) 
  (h3 : tip_rate = 0.06) 
  (h4 : sales_tax = cost_lunch * sales_tax_rate) 
  (h5 : tip = cost_lunch * tip_rate) :
  cost_lunch + sales_tax + tip = 110 :=
by
  sorry

end total_amount_paid_l165_165028


namespace probability_not_red_light_l165_165739

theorem probability_not_red_light :
  ∀ (red_light yellow_light green_light : ℕ),
    red_light = 30 →
    yellow_light = 5 →
    green_light = 40 →
    (yellow_light + green_light) / (red_light + yellow_light + green_light) = (3 : ℚ) / 5 :=
by intros red_light yellow_light green_light h_red h_yellow h_green
   sorry

end probability_not_red_light_l165_165739


namespace pi_minus_five_floor_value_l165_165817

noncomputable def greatest_integer_function (x : ℝ) : ℤ := Int.floor x

theorem pi_minus_five_floor_value :
  greatest_integer_function (Real.pi - 5) = -2 :=
by
  -- The proof is omitted
  sorry

end pi_minus_five_floor_value_l165_165817


namespace avg_rate_of_change_l165_165427

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 + 5

theorem avg_rate_of_change :
  (f 0.2 - f 0.1) / (0.2 - 0.1) = 0.9 := by
  sorry

end avg_rate_of_change_l165_165427


namespace smallest_c_is_52_l165_165693

def seq (n : ℕ) : ℤ := -103 + (n:ℤ) * 2

theorem smallest_c_is_52 :
  ∃ c : ℕ, 
  (∀ n : ℕ, n < c → (∀ m : ℕ, m < n → seq m < 0) ∧ seq n = 0) ∧
  seq c > 0 ∧
  c = 52 :=
by
  sorry

end smallest_c_is_52_l165_165693


namespace reunion_handshakes_l165_165505

/-- 
Given 15 boys at a reunion:
- 5 are left-handed and will only shake hands with other left-handed boys.
- Each boy shakes hands exactly once with each of the others unless they forget.
- Three boys each forget to shake hands with two others.

Prove that the total number of handshakes is 49. 
-/
theorem reunion_handshakes : 
  let total_boys := 15
  let left_handed := 5
  let forgetful_boys := 3
  let forgotten_handshakes_per_boy := 2

  let total_handshakes := total_boys * (total_boys - 1) / 2
  let left_left_handshakes := left_handed * (left_handed - 1) / 2
  let left_right_handshakes := left_handed * (total_boys - left_handed)
  let distinct_forgotten_handshakes := forgetful_boys * forgotten_handshakes_per_boy / 2

  total_handshakes 
    - left_right_handshakes 
    - distinct_forgotten_handshakes
    - left_left_handshakes
  = 49 := 
sorry

end reunion_handshakes_l165_165505


namespace find_p_l165_165668

-- Definitions
variables {n : ℕ} {p : ℝ}
def X : Type := ℕ -- Assume X is ℕ-valued

-- Conditions
axiom binomial_expectation : n * p = 6
axiom binomial_variance : n * p * (1 - p) = 3

-- Question to prove
theorem find_p : p = 1 / 2 :=
by
  sorry

end find_p_l165_165668


namespace inequality_proof_l165_165057

theorem inequality_proof (a : ℝ) (h1 : 0 < a) (h2 : a < 1) : 
  (1 / a + 4 / (1 - a) ≥ 9) := 
sorry

end inequality_proof_l165_165057


namespace remainder_7n_mod_4_l165_165079

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by sorry

end remainder_7n_mod_4_l165_165079


namespace max_expression_value_l165_165094

theorem max_expression_value :
  ∀ (a b : ℝ), (100 ≤ a ∧ a ≤ 500) → (500 ≤ b ∧ b ≤ 1500) → 
  (∃ x, x = (b - 100) / (a + 50) ∧ ∀ y, y = (b - 100) / (a + 50) → y ≤ (28 / 3)) :=
by
  sorry

end max_expression_value_l165_165094


namespace count_correct_propositions_l165_165552

def line_parallel_plane (a : Line) (M : Plane) : Prop := sorry
def line_perpendicular_plane (a : Line) (M : Plane) : Prop := sorry
def line_parallel_line (a b : Line) : Prop := sorry
def line_perpendicular_line (a b : Line) : Prop := sorry
def plane_perpendicular_plane (M N : Plane) : Prop := sorry

theorem count_correct_propositions 
  (a b c : Line) 
  (M N : Plane) 
  (h1 : ¬ (line_parallel_plane a M ∧ line_parallel_plane b M → line_parallel_line a b)) 
  (h2 : line_parallel_plane a M ∧ line_perpendicular_plane b M → line_perpendicular_line b a) 
  (h3 : ¬ ((line_parallel_plane a M ∧ line_perpendicular_plane b M ∧ line_perpendicular_line c a ∧ line_perpendicular_line c b) → line_perpendicular_plane c M))
  (h4 : line_perpendicular_plane a M ∧ line_parallel_plane a N → plane_perpendicular_plane M N) :
  (0 + 1 + 0 + 1) = 2 :=
sorry

end count_correct_propositions_l165_165552


namespace total_players_on_ground_l165_165965

theorem total_players_on_ground :
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  cricket_players + hockey_players + football_players + softball_players +
  basketball_players + volleyball_players + netball_players + rugby_players = 263 := 
by 
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  sorry

end total_players_on_ground_l165_165965


namespace parabolas_equation_l165_165272

theorem parabolas_equation (vertex_origin : (0, 0) ∈ {(x, y) | y = x^2} ∨ (0, 0) ∈ {(x, y) | x = -y^2})
  (focus_on_axis : ∀ F : ℝ × ℝ, (F ∈ {(x, y) | y = x^2} ∨ F ∈ {(x, y) | x = -y^2}) → (F.1 = 0 ∨ F.2 = 0))
  (through_point : (-2, 4) ∈ {(x, y) | y = x^2} ∨ (-2, 4) ∈ {(x, y) | x = -y^2}) :
  {(x, y) | y = x^2} ∪ {(x, y) | x = -y^2} ≠ ∅ :=
by
  sorry

end parabolas_equation_l165_165272


namespace original_number_is_twenty_l165_165808

theorem original_number_is_twenty (x : ℕ) (h : 100 * x = x + 1980) : x = 20 :=
sorry

end original_number_is_twenty_l165_165808


namespace sin2alpha_plus_cosalpha_l165_165725

theorem sin2alpha_plus_cosalpha (α : ℝ) (h1 : Real.tan α = 2) (h2 : 0 < α ∧ α < Real.pi / 2) : 
  Real.sin (2 * α) + Real.cos α = (4 + Real.sqrt 5) / 5 :=
by
  sorry

end sin2alpha_plus_cosalpha_l165_165725


namespace prime_power_divides_power_of_integer_l165_165406

theorem prime_power_divides_power_of_integer 
    {p a n : ℕ} 
    (hp : Nat.Prime p)
    (ha_pos : 0 < a) 
    (hn_pos : 0 < n) 
    (h : p ∣ a^n) :
    p^n ∣ a^n := 
by 
  sorry

end prime_power_divides_power_of_integer_l165_165406


namespace correct_comparison_l165_165197

-- Definitions of conditions based on the problem 
def hormones_participate : Prop := false 
def enzymes_produced_by_living_cells : Prop := true 
def hormones_produced_by_endocrine : Prop := true 
def endocrine_can_produce_both : Prop := true 
def synthesize_enzymes_not_nec_hormones : Prop := true 
def not_all_proteins : Prop := true 

-- Statement of the equivalence between the correct answer and its proof
theorem correct_comparison :  (¬hormones_participate ∧ enzymes_produced_by_living_cells ∧ hormones_produced_by_endocrine ∧ endocrine_can_produce_both ∧ synthesize_enzymes_not_nec_hormones ∧ not_all_proteins) → (endocrine_can_produce_both) :=
by
  sorry

end correct_comparison_l165_165197


namespace percentage_of_burpees_is_10_l165_165099

-- Definitions for each exercise count
def jumping_jacks : ℕ := 25
def pushups : ℕ := 15
def situps : ℕ := 30
def burpees : ℕ := 10
def lunges : ℕ := 20

-- Total number of exercises
def total_exercises : ℕ := jumping_jacks + pushups + situps + burpees + lunges

-- The proof statement
theorem percentage_of_burpees_is_10 :
  (burpees * 100) / total_exercises = 10 :=
by
  sorry

end percentage_of_burpees_is_10_l165_165099


namespace rectangle_area_l165_165507

theorem rectangle_area (c h x : ℝ) (h_pos : 0 < h) (c_pos : 0 < c) : 
  (A : ℝ) = (x * (c * x / h)) :=
by
  sorry

end rectangle_area_l165_165507


namespace height_of_pole_l165_165773

noncomputable section
open Real

theorem height_of_pole (α β γ : ℝ) (h xA xB xC : ℝ) 
  (hA : tan α = h / xA) (hB : tan β = h / xB) (hC : tan γ = h / xC) 
  (sum_angles : α + β + γ = π / 2) : h = 10 :=
by
  sorry

end height_of_pole_l165_165773


namespace negation_of_universal_prop_l165_165900

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ (∃ x : ℝ, x^2 - 2*x + 4 > 0) :=
by
  sorry

end negation_of_universal_prop_l165_165900


namespace distance_travelled_is_960_l165_165299

-- Definitions based on conditions
def speed_slower := 60 -- Speed of slower bike in km/h
def speed_faster := 64 -- Speed of faster bike in km/h
def time_diff := 1 -- Time difference in hours

-- Problem statement: Prove that the distance covered by both bikes is 960 km.
theorem distance_travelled_is_960 (T : ℝ) (D : ℝ) 
  (h1 : D = speed_slower * T)
  (h2 : D = speed_faster * (T - time_diff)) :
  D = 960 := 
sorry

end distance_travelled_is_960_l165_165299


namespace factor_expression_l165_165277

variable (x y : ℝ)

theorem factor_expression : 3 * x^3 - 6 * x^2 * y + 3 * x * y^2 = 3 * x * (x - y)^2 := 
by 
  sorry

end factor_expression_l165_165277


namespace smaller_cube_volume_l165_165918

theorem smaller_cube_volume
  (V_L : ℝ) (N : ℝ) (SA_diff : ℝ) 
  (h1 : V_L = 8)
  (h2 : N = 8)
  (h3 : SA_diff = 24) :
  (∀ V_S : ℝ, V_L = N * V_S → V_S = 1) :=
by
  sorry

end smaller_cube_volume_l165_165918


namespace sum_of_ages_l165_165673

theorem sum_of_ages (Petra_age : ℕ) (Mother_age : ℕ)
  (h_petra : Petra_age = 11)
  (h_mother : Mother_age = 36) :
  Petra_age + Mother_age = 47 :=
by
  -- Using the given conditions:
  -- Petra_age = 11
  -- Mother_age = 36
  sorry

end sum_of_ages_l165_165673


namespace final_surface_area_l165_165400

theorem final_surface_area 
  (original_cube_volume : ℕ)
  (small_cube_volume : ℕ)
  (remaining_cubes : ℕ)
  (removed_cubes : ℕ)
  (per_face_expose_area : ℕ)
  (initial_surface_area_per_cube : ℕ)
  (total_cubes : ℕ)
  (shared_internal_faces_area : ℕ)
  (final_surface_area : ℕ) :
  original_cube_volume = 12 * 12 * 12 →
  small_cube_volume = 3 * 3 * 3 →
  total_cubes = 64 →
  removed_cubes = 14 →
  remaining_cubes = total_cubes - removed_cubes →
  initial_surface_area_per_cube = 6 * 3 * 3 →
  per_face_expose_area = 6 * 4 →
  final_surface_area = remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area →
  (remaining_cubes * (initial_surface_area_per_cube + per_face_expose_area) - shared_internal_faces_area) = 2820 :=
sorry

end final_surface_area_l165_165400


namespace find_percentage_find_percentage_as_a_percentage_l165_165382

variable (P : ℝ)

theorem find_percentage (h : P / 2 = 0.02) : P = 0.04 :=
by
  sorry

theorem find_percentage_as_a_percentage (h : P / 2 = 0.02) : P = 4 :=
by
  sorry

end find_percentage_find_percentage_as_a_percentage_l165_165382


namespace find_a_l165_165263

-- Define the domains of the functions f and g
def A : Set ℝ :=
  {x | x < -1 ∨ x ≥ 1}

def B (a : ℝ) : Set ℝ :=
  {x | 2 * a < x ∧ x < a + 1}

-- Restate the problem as a Lean proposition
theorem find_a (a : ℝ) (h : a < 1) (hb : B a ⊆ A) :
  a ∈ {x | x ≤ -2 ∨ (1 / 2 ≤ x ∧ x < 1)} :=
sorry

end find_a_l165_165263


namespace recreation_spending_l165_165877

theorem recreation_spending : 
  ∀ (W : ℝ), 
  (last_week_spent : ℝ) -> last_week_spent = 0.20 * W →
  (this_week_wages : ℝ) -> this_week_wages = 0.80 * W →
  (this_week_spent : ℝ) -> this_week_spent = 0.40 * this_week_wages →
  this_week_spent / last_week_spent * 100 = 160 :=
by
  sorry

end recreation_spending_l165_165877


namespace disk_max_areas_l165_165247

-- Conditions Definition
def disk_divided (n : ℕ) : ℕ :=
  let radii := 3 * n
  let secant_lines := 2
  let total_areas := 9 * n
  total_areas

theorem disk_max_areas (n : ℕ) : disk_divided n = 9 * n :=
by
  sorry

end disk_max_areas_l165_165247


namespace mutual_fund_share_increase_l165_165611

theorem mutual_fund_share_increase (P : ℝ) (h1 : (P * 1.20) = 1.20 * P) (h2 : (1.20 * P) * (1 / 3) = 0.40 * P) :
  ((1.60 * P) = (P * 1.60)) :=
by
  sorry

end mutual_fund_share_increase_l165_165611


namespace inequality_holds_l165_165846

theorem inequality_holds (x : ℝ) : (∀ y : ℝ, y > 0 → (4 * (x^2 * y^2 + 4 * x * y^2 + 4 * x^2 * y + 16 * y^2 + 12 * x^2 * y) / (x + y) > 3 * x^2 * y)) ↔ x > 0 := 
sorry

end inequality_holds_l165_165846


namespace complex_mul_example_l165_165391

theorem complex_mul_example (i : ℝ) (h : i^2 = -1) : (⟨2, 2 * i⟩ : ℂ) * (⟨1, -2 * i⟩) = ⟨6, -2 * i⟩ :=
by
  sorry

end complex_mul_example_l165_165391


namespace lauren_total_money_made_is_correct_l165_165610

-- Define the rate per commercial view
def rate_per_commercial_view : ℝ := 0.50
-- Define the rate per subscriber
def rate_per_subscriber : ℝ := 1.00
-- Define the number of commercial views on Tuesday
def commercial_views : ℕ := 100
-- Define the number of new subscribers on Tuesday
def subscribers : ℕ := 27
-- Calculate the total money Lauren made on Tuesday
def total_money_made (rate_com_view : ℝ) (rate_sub : ℝ) (com_views : ℕ) (subs : ℕ) : ℝ :=
  (rate_com_view * com_views) + (rate_sub * subs)

-- Theorem stating that the total money Lauren made on Tuesday is $77.00
theorem lauren_total_money_made_is_correct : total_money_made rate_per_commercial_view rate_per_subscriber commercial_views subscribers = 77.00 :=
by
  sorry

end lauren_total_money_made_is_correct_l165_165610


namespace martha_meeting_distance_l165_165139

theorem martha_meeting_distance (t : ℝ) (d : ℝ)
  (h1 : 0 < t)
  (h2 : d = 45 * (t + 0.75))
  (h3 : d - 45 = 55 * (t - 1)) :
  d = 230.625 := 
  sorry

end martha_meeting_distance_l165_165139


namespace transform_equation_l165_165765

theorem transform_equation (x : ℝ) : x^2 - 2 * x - 2 = 0 ↔ (x - 1)^2 = 3 :=
sorry

end transform_equation_l165_165765


namespace diamonds_G20_l165_165372

def diamonds_in_figure (n : ℕ) : ℕ :=
if n = 1 then 1 else 4 * n^2 + 4 * n - 7

theorem diamonds_G20 : diamonds_in_figure 20 = 1673 :=
by sorry

end diamonds_G20_l165_165372


namespace symmetrical_line_range_l165_165482

theorem symmetrical_line_range {k : ℝ} :
  (∀ x y : ℝ, (y = k * x - 1) ∧ (x + y - 1 = 0) → y ≠ -x + 1) → k > 1 ↔ k > 1 :=
by
  sorry

end symmetrical_line_range_l165_165482


namespace monica_usd_start_amount_l165_165993

theorem monica_usd_start_amount (x : ℕ) (H : ∃ (y : ℕ), y = 40 ∧ (8 : ℚ) / 5 * x - y = x) :
  (x / 100) + (x % 100 / 10) + (x % 10) = 2 := 
by
  sorry

end monica_usd_start_amount_l165_165993


namespace second_chapter_pages_is_80_l165_165421

def first_chapter_pages : ℕ := 37
def second_chapter_pages : ℕ := first_chapter_pages + 43

theorem second_chapter_pages_is_80 : second_chapter_pages = 80 :=
by
  sorry

end second_chapter_pages_is_80_l165_165421


namespace line_and_circle_condition_l165_165076

theorem line_and_circle_condition (P Q : ℝ × ℝ) (radius : ℝ) 
  (x y m : ℝ) (n : ℝ) (l : ℝ × ℝ → Prop)
  (hPQ : P = (4, -2)) 
  (hPQ' : Q = (-1, 3)) 
  (hC : ∀ (x y : ℝ), (x - 1)^2 + y^2 = radius) 
  (hr : radius < 5) 
  (h_y_segment : ∃ (k : ℝ), |k - 0| = 4 * Real.sqrt 3) 
  : (∀ (x y : ℝ), x + y = 2) ∧ 
    ((∀ (x y : ℝ), l (x, y) ↔ x + y + m = 0 ∨ x + y = 0) 
    ∧ (m = 3 ∨ m = -4) 
    ∧ (∀ A B : ℝ × ℝ, l A → l B → (A.1 - B.1)^2 + (A.2 - B.2)^2 = radius)) := 
  by
  sorry

end line_and_circle_condition_l165_165076


namespace square_floor_tile_count_l165_165502

theorem square_floor_tile_count (n : ℕ) (h : 2 * n - 1 = 49) : n^2 = 625 := by
  sorry

end square_floor_tile_count_l165_165502


namespace quadratic_two_distinct_real_roots_l165_165128

def quadratic_function_has_two_distinct_real_roots (k : ℝ) : Prop :=
  let a := k
  let b := -4
  let c := -2
  b * b - 4 * a * c > 0 ∧ a ≠ 0

theorem quadratic_two_distinct_real_roots (k : ℝ) :
  quadratic_function_has_two_distinct_real_roots k ↔ (k > -2 ∧ k ≠ 0) :=
by
  sorry

end quadratic_two_distinct_real_roots_l165_165128


namespace simplify_and_evaluate_expression_l165_165721

variables (a b : ℚ)

theorem simplify_and_evaluate_expression : 
  (4 * (a^2 - 2 * a * b) - (3 * a^2 - 5 * a * b + 1)) = 5 :=
by
  let a := -2
  let b := (1 : ℚ) / 3
  sorry

end simplify_and_evaluate_expression_l165_165721


namespace find_digits_l165_165428

variable (M N : ℕ)
def x := 10 * N + M
def y := 10 * M + N

theorem find_digits (h₁ : x > y) (h₂ : x + y = 11 * (x - y)) : M = 4 ∧ N = 5 :=
sorry

end find_digits_l165_165428


namespace problem1_problem2_l165_165214

open Set

-- Part (1)
theorem problem1 (a : ℝ) :
  (∀ x, x ∉ Icc (0 : ℝ) (2 : ℝ) → x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ)) ∨ (∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∉ Icc (0 : ℝ) (2 : ℝ)) → a ≤ 0 := 
sorry

-- Part (2)
theorem problem2 (a : ℝ) :
  (¬ ∀ x, x ∈ Icc (a : ℝ) (3 - 2 * a : ℝ) → x ∈ Icc (0 : ℝ) (2 : ℝ)) → (a < 0.5 ∨ a > 1) :=
sorry

end problem1_problem2_l165_165214


namespace triangle_inequality_inequality_l165_165132

variable {a b c : ℝ}
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (triangle_ineq : a + b > c)

theorem triangle_inequality_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0) (triangle_ineq : a + b > c) :
  a^3 + b^3 + 3 * a * b * c > c^3 :=
sorry

end triangle_inequality_inequality_l165_165132


namespace compounding_frequency_l165_165927

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem compounding_frequency (P A r t n : ℝ) 
  (principal : P = 6000) 
  (amount : A = 6615)
  (rate : r = 0.10)
  (time : t = 1) 
  (comp_freq : n = 2) :
  compound_interest P r n t = A := 
by 
  simp [compound_interest, principal, rate, time, comp_freq, amount]
  -- calculations and proof omitted
  sorry

end compounding_frequency_l165_165927


namespace unique_intersection_l165_165213

open Real

-- Defining the functions f and g as per the conditions
def f (b : ℝ) (x : ℝ) : ℝ := b * x^2 + 5 * x + 3
def g (x : ℝ) : ℝ := -2 * x - 2

-- The condition that the intersection occurs at one point translates to a specific b satisfying the discriminant condition.
theorem unique_intersection (b : ℝ) : (∃ x : ℝ, f b x = g x) ∧ (f b x = g x → ∀ y : ℝ, y ≠ x → f b y ≠ g y) ↔ b = 49 / 20 :=
by {
  sorry
}

end unique_intersection_l165_165213


namespace Jason_more_blue_marbles_l165_165531

theorem Jason_more_blue_marbles (Jason_blue_marbles Tom_blue_marbles : ℕ) 
  (hJ : Jason_blue_marbles = 44) (hT : Tom_blue_marbles = 24) :
  Jason_blue_marbles - Tom_blue_marbles = 20 :=
by
  sorry

end Jason_more_blue_marbles_l165_165531


namespace jane_vases_per_day_l165_165254

theorem jane_vases_per_day : 
  ∀ (total_vases : ℝ) (days : ℝ), 
  total_vases = 248 → days = 16 → 
  (total_vases / days) = 15.5 :=
by
  intros total_vases days h_total_vases h_days
  rw [h_total_vases, h_days]
  norm_num

end jane_vases_per_day_l165_165254


namespace bonnie_egg_count_indeterminable_l165_165933

theorem bonnie_egg_count_indeterminable
    (eggs_Kevin : ℕ)
    (eggs_George : ℕ)
    (eggs_Cheryl : ℕ)
    (diff_Cheryl_combined : ℕ)
    (c1 : eggs_Kevin = 5)
    (c2 : eggs_George = 9)
    (c3 : eggs_Cheryl = 56)
    (c4 : diff_Cheryl_combined = 29)
    (h₁ : eggs_Cheryl = diff_Cheryl_combined + (eggs_Kevin + eggs_George + some_children)) :
    ∀ (eggs_Bonnie : ℕ), ∃ some_children : ℕ, eggs_Bonnie = eggs_Bonnie :=
by
  -- The proof is omitted here
  sorry

end bonnie_egg_count_indeterminable_l165_165933


namespace P_zero_eq_zero_l165_165897

open Polynomial

noncomputable def P (x : ℝ) : ℝ := sorry

axiom distinct_roots : ∃ y : Fin 17 → ℝ, Function.Injective y ∧ ∀ i, P (y i ^ 2) = 0

theorem P_zero_eq_zero : P 0 = 0 :=
by
  sorry

end P_zero_eq_zero_l165_165897


namespace arc_length_EF_l165_165334

-- Definitions based on the conditions
def angle_DEF_degrees : ℝ := 45
def circumference_D : ℝ := 80
def total_circle_degrees : ℝ := 360

-- Theorems/lemmata needed to prove the required statement
theorem arc_length_EF :
  let proportion := angle_DEF_degrees / total_circle_degrees
  let arc_length := proportion * circumference_D
  arc_length = 10 :=
by
  -- Placeholder for the proof
  sorry

end arc_length_EF_l165_165334


namespace lettuce_price_1_l165_165823

theorem lettuce_price_1 (customers_per_month : ℕ) (lettuce_per_customer : ℕ) (tomatoes_per_customer : ℕ) 
(price_per_tomato : ℝ) (total_sales : ℝ)
  (h_customers : customers_per_month = 500)
  (h_lettuce_per_customer : lettuce_per_customer = 2)
  (h_tomatoes_per_customer : tomatoes_per_customer = 4)
  (h_price_per_tomato : price_per_tomato = 0.5)
  (h_total_sales : total_sales = 2000) :
  let heads_of_lettuce_sold := customers_per_month * lettuce_per_customer
  let tomato_sales := customers_per_month * tomatoes_per_customer * price_per_tomato
  let lettuce_sales := total_sales - tomato_sales
  let price_per_lettuce := lettuce_sales / heads_of_lettuce_sold
  price_per_lettuce = 1 := by
{
  sorry
}

end lettuce_price_1_l165_165823


namespace total_snacks_l165_165228

variable (peanuts : ℝ) (raisins : ℝ)

theorem total_snacks (h1 : peanuts = 0.1) (h2 : raisins = 0.4) : peanuts + raisins = 0.5 :=
by
  sorry

end total_snacks_l165_165228


namespace product_gcd_lcm_l165_165821

theorem product_gcd_lcm (a b : ℕ) (ha : a = 90) (hb : b = 150) :
  Nat.gcd a b * Nat.lcm a b = 13500 := by
  sorry

end product_gcd_lcm_l165_165821


namespace find_g_inv_84_l165_165742

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem find_g_inv_84 : g 3 = 84 → ∃ x, g x = 84 ∧ x = 3 :=
by
  sorry

end find_g_inv_84_l165_165742


namespace necessary_but_not_sufficient_condition_l165_165883
-- Import the required Mathlib library in Lean 4

-- State the equivalent proof problem
theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (|a| ≤ 1 → a ≤ 1) ∧ ¬ (a ≤ 1 → |a| ≤ 1) :=
by
  sorry

end necessary_but_not_sufficient_condition_l165_165883


namespace slower_whale_length_is_101_25_l165_165297

def length_of_slower_whale (v_i_f v_i_s a_f a_s t : ℝ) : ℝ :=
  let D_f := v_i_f * t + 0.5 * a_f * t^2
  let D_s := v_i_s * t + 0.5 * a_s * t^2
  D_f - D_s

theorem slower_whale_length_is_101_25
  (v_i_f v_i_s a_f a_s t L : ℝ)
  (h1 : v_i_f = 18)
  (h2 : v_i_s = 15)
  (h3 : a_f = 1)
  (h4 : a_s = 0.5)
  (h5 : t = 15)
  (h6 : length_of_slower_whale v_i_f v_i_s a_f a_s t = L) :
  L = 101.25 :=
by
  sorry

end slower_whale_length_is_101_25_l165_165297


namespace find_k_l165_165546

variable (x y k : ℝ)

-- Definition: the line equations and the intersection condition
def line1_eq (x y k : ℝ) : Prop := 3 * x - 2 * y = k
def line2_eq (x y : ℝ) : Prop := x - 0.5 * y = 10
def intersect_at_x (x : ℝ) : Prop := x = -6

-- The theorem we need to prove
theorem find_k (h1 : line1_eq x y k)
               (h2 : line2_eq x y)
               (h3 : intersect_at_x x) :
               k = 46 :=
sorry

end find_k_l165_165546


namespace daniel_utility_equation_solution_l165_165278

theorem daniel_utility_equation_solution (t : ℚ) :
  t * (10 - t) = (4 - t) * (t + 4) → t = 8 / 5 := by
  sorry

end daniel_utility_equation_solution_l165_165278


namespace sequence_solution_l165_165708

-- Defining the sequence and the condition
def sequence_condition (a S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → S n = 2 * a n - 1

-- Defining the sequence formula we need to prove
def sequence_formula (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n = 2 ^ (n - 1)

theorem sequence_solution (a S : ℕ → ℝ) (h : sequence_condition a S) :
  sequence_formula a :=
by 
  sorry

end sequence_solution_l165_165708


namespace part_a_part_b_l165_165148

-- Conditions
def has_three_classmates_in_any_group_of_ten (students : Fin 60 → Type) : Prop :=
  ∀ (g : Finset (Fin 60)), g.card = 10 → ∃ (a b c : Fin 60), a ∈ g ∧ b ∈ g ∧ c ∈ g ∧ students a = students b ∧ students b = students c

-- Part (a)
theorem part_a (students : Fin 60 → Type) (h : has_three_classmates_in_any_group_of_ten students) : ∃ g : Finset (Fin 60), g.card ≥ 15 ∧ ∀ a b : Fin 60, a ∈ g → b ∈ g → students a = students b :=
sorry

-- Part (b)
theorem part_b (students : Fin 60 → Type) (h : has_three_classmates_in_any_group_of_ten students) : ¬ ∃ g : Finset (Fin 60), g.card ≥ 16 ∧ ∀ a b : Fin 60, a ∈ g → b ∈ g → students a = students b :=
sorry

end part_a_part_b_l165_165148


namespace log_abs_is_even_l165_165696

open Real

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

noncomputable def f (x : ℝ) : ℝ := log (abs x)

theorem log_abs_is_even : is_even_function f :=
by
  sorry

end log_abs_is_even_l165_165696


namespace div_m_by_18_equals_500_l165_165013

-- Define the conditions
noncomputable def m : ℕ := 9000 -- 'm' is given as 9000 since it fulfills all conditions described
def is_multiple_of_18 (n : ℕ) : Prop := n % 18 = 0
def all_digits_9_or_0 (n : ℕ) : Prop := ∀ (d : ℕ), (∃ (k : ℕ), n = 10^k * d) → (d = 0 ∨ d = 9)

-- Define the proof problem statement
theorem div_m_by_18_equals_500 
  (h1 : is_multiple_of_18 m) 
  (h2 : all_digits_9_or_0 m) 
  (h3 : ∀ n, is_multiple_of_18 n ∧ all_digits_9_or_0 n → n ≤ m) : 
  m / 18 = 500 :=
sorry

end div_m_by_18_equals_500_l165_165013


namespace complex_square_l165_165033

theorem complex_square (z : ℂ) (i : ℂ) (h1 : z = 2 - 3 * i) (h2 : i^2 = -1) : z^2 = -5 - 12 * i :=
sorry

end complex_square_l165_165033


namespace cube_greater_than_quadratic_minus_linear_plus_one_l165_165261

variable (x : ℝ)

theorem cube_greater_than_quadratic_minus_linear_plus_one (h : x > 1) :
  x^3 > x^2 - x + 1 := by
  sorry

end cube_greater_than_quadratic_minus_linear_plus_one_l165_165261


namespace rhombus_area_l165_165190

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 15) (h2 : d2 = 12) : (d1 * d2) / 2 = 90 := by
  sorry

end rhombus_area_l165_165190


namespace exactly_two_overlap_l165_165308

-- Define the concept of rectangles
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

-- Define the given rectangles
def rect1 : Rectangle := ⟨4, 6⟩
def rect2 : Rectangle := ⟨4, 6⟩
def rect3 : Rectangle := ⟨4, 6⟩

-- Hypothesis defining the overlapping areas
def overlap1_2 : ℕ := 4 * 2 -- first and second rectangles overlap in 8 cells
def overlap2_3 : ℕ := 2 * 6 -- second and third rectangles overlap in 12 cells
def overlap1_3 : ℕ := 0    -- first and third rectangles do not directly overlap

-- Total overlap calculation
def total_exactly_two_overlap : ℕ := (overlap1_2 + overlap2_3)

-- The theorem we need to prove
theorem exactly_two_overlap (rect1 rect2 rect3 : Rectangle) : total_exactly_two_overlap = 14 := sorry

end exactly_two_overlap_l165_165308


namespace cos_sum_identity_l165_165744

theorem cos_sum_identity (θ : ℝ) (h1 : Real.tan θ = -5 / 12) (h2 : θ ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi)) :
  Real.cos (θ + Real.pi / 4) = 17 * Real.sqrt 2 / 26 :=
sorry

end cos_sum_identity_l165_165744


namespace problem_simplify_l165_165852

variable (a : ℝ)

theorem problem_simplify (h1 : a ≠ 3) (h2 : a ≠ -3) :
  (1 / (a - 3) - 6 / (a^2 - 9) = 1 / (a + 3)) :=
sorry

end problem_simplify_l165_165852


namespace discount_percentage_l165_165178

theorem discount_percentage (original_price sale_price : ℝ) (h₁ : original_price = 128) (h₂ : sale_price = 83.2) :
  (original_price - sale_price) / original_price * 100 = 35 :=
by
  sorry

end discount_percentage_l165_165178


namespace equation1_solution_equation2_solution_l165_165126

theorem equation1_solution (x : ℝ) : x^2 - 10*x + 16 = 0 ↔ x = 2 ∨ x = 8 :=
by sorry

theorem equation2_solution (x : ℝ) : 2*x*(x-1) = x-1 ↔ x = 1 ∨ x = 1/2 :=
by sorry

end equation1_solution_equation2_solution_l165_165126


namespace matrix_cube_computation_l165_165598

-- Define the original matrix
def matrix1 : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2], ![2, 0]]

-- Define the expected result matrix
def expected_matrix : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-8, 0], ![0, -8]]

-- State the theorem to be proved
theorem matrix_cube_computation : matrix1 ^ 3 = expected_matrix :=
  by sorry

end matrix_cube_computation_l165_165598


namespace number_of_ways_to_write_2024_l165_165070

theorem number_of_ways_to_write_2024 :
  (∃ a b c : ℕ, 2 * a + 3 * b + 4 * c = 2024) -> 
  (∃ n m p : ℕ, a = 3 * n + 2 * m + p ∧ n + m + p = 337) ->
  (∃ n m p : ℕ, n + m + p = 337 ∧ 2 * n * 3 + m * 2 + p * 6 = 2 * (57231 + 498)) :=
sorry

end number_of_ways_to_write_2024_l165_165070


namespace range_of_c_l165_165643

-- Definitions of the propositions p and q
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := ∃ x : ℝ, x^2 - c^2 ≤ - (1 / 16)

-- Main theorem
theorem range_of_c (c : ℝ) (h1 : c > 0) (h2 : p c) (h3 : q c) : c ≥ 1 / 4 ∧ c < 1 :=
  sorry

end range_of_c_l165_165643


namespace determine_vector_p_l165_165004

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def vector_operation (m p : Vector2D) : Vector2D :=
  Vector2D.mk (m.x * p.x + m.y * p.y) (m.x * p.y + m.y * p.x)

theorem determine_vector_p (p : Vector2D) : 
  (∀ (m : Vector2D), vector_operation m p = m) → p = Vector2D.mk 1 0 :=
by
  sorry

end determine_vector_p_l165_165004


namespace x_coordinate_second_point_l165_165361

theorem x_coordinate_second_point (m n : ℝ) 
(h₁ : m = 2 * n + 5)
(h₂ : m + 2 = 2 * (n + 1) + 5) : 
  (m + 2) = 2 * n + 7 :=
by sorry

end x_coordinate_second_point_l165_165361


namespace circle_equation_l165_165146

-- Define conditions
def on_parabola (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x^2 = 4 * y

def tangent_to_y_axis (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (x, _) := M
  abs x = r

def tangent_to_axis_of_symmetry (M : ℝ × ℝ) (r : ℝ) : Prop :=
  let (_, y) := M
  abs (1 + y) = r

-- Main theorem statement
theorem circle_equation (M : ℝ × ℝ) (r : ℝ) (x y : ℝ)
  (h1 : on_parabola M)
  (h2 : tangent_to_y_axis M r)
  (h3 : tangent_to_axis_of_symmetry M r) :
  (x - M.1)^2 + (y - M.2)^2 = r^2 ↔
  x^2 + y^2 + 4 * M.1 * x - 2 * M.2 * y + 1 = 0 := 
sorry

end circle_equation_l165_165146


namespace firstGradeMuffins_l165_165501

-- Define the conditions as the number of muffins baked by each class
def mrsBrierMuffins : ℕ := 18
def mrsMacAdamsMuffins : ℕ := 20
def mrsFlanneryMuffins : ℕ := 17

-- Define the total number of muffins baked
def totalMuffins : ℕ := mrsBrierMuffins + mrsMacAdamsMuffins + mrsFlanneryMuffins

-- Prove that the total number of muffins baked is 55
theorem firstGradeMuffins : totalMuffins = 55 := by
  sorry

end firstGradeMuffins_l165_165501


namespace intersection_sets_l165_165250

-- Define set A as all x such that x >= -2
def setA : Set ℝ := {x | x >= -2}

-- Define set B as all x such that x < 1
def setB : Set ℝ := {x | x < 1}

-- The statement to prove in Lean 4
theorem intersection_sets : (setA ∩ setB) = {x | -2 <= x ∧ x < 1} :=
by
  sorry

end intersection_sets_l165_165250


namespace specialPermutationCount_l165_165916

def countSpecialPerms (n : ℕ) : ℕ := 2 ^ (n - 1)

theorem specialPermutationCount (n : ℕ) : 
  (countSpecialPerms n = 2 ^ (n - 1)) := 
by 
  sorry

end specialPermutationCount_l165_165916


namespace fewest_handshakes_organizer_l165_165200

theorem fewest_handshakes_organizer (n k : ℕ) (h : k < n) 
  (total_handshakes: n*(n-1)/2 + k = 406) :
  k = 0 :=
sorry

end fewest_handshakes_organizer_l165_165200


namespace clock_strikes_twelve_l165_165325

def clock_strike_interval (strikes : Nat) (time : Nat) : Nat :=
  if strikes > 1 then time / (strikes - 1) else 0

def total_time_for_strikes (strikes : Nat) (interval : Nat) : Nat :=
  if strikes > 1 then (strikes - 1) * interval else 0

theorem clock_strikes_twelve (interval_six : Nat) (time_six : Nat) (time_twelve : Nat) :
  interval_six = clock_strike_interval 6 time_six →
  time_twelve = total_time_for_strikes 12 interval_six →
  time_six = 30 →
  time_twelve = 66 :=
by
  -- The proof will go here
  sorry

end clock_strikes_twelve_l165_165325


namespace neither_coffee_tea_juice_l165_165163

open Set

theorem neither_coffee_tea_juice (total : ℕ) (coffee : ℕ) (tea : ℕ) (both_coffee_tea : ℕ)
  (juice : ℕ) (juice_and_tea_not_coffee : ℕ) :
  total = 35 → 
  coffee = 18 → 
  tea = 15 → 
  both_coffee_tea = 7 → 
  juice = 6 → 
  juice_and_tea_not_coffee = 3 →
  (total - ((coffee + tea - both_coffee_tea) + (juice - juice_and_tea_not_coffee))) = 6 :=
sorry

end neither_coffee_tea_juice_l165_165163


namespace xsq_plus_ysq_l165_165365

theorem xsq_plus_ysq (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 12) : x^2 + y^2 = 25 :=
by
  sorry

end xsq_plus_ysq_l165_165365


namespace tank_capacity_l165_165413

theorem tank_capacity (C : ℝ) (rate_leak : ℝ) (rate_inlet : ℝ) (combined_rate_empty : ℝ) :
  rate_leak = C / 3 ∧ rate_inlet = 6 * 60 ∧ combined_rate_empty = C / 12 →
  C = 864 :=
by
  intros h
  sorry

end tank_capacity_l165_165413


namespace books_arrangement_l165_165951

/-
  Theorem:
  If there are 4 distinct math books, 6 distinct English books, and 3 distinct science books,
  and each category of books must stay together, then the number of ways to arrange
  these books on a shelf is 622080.
-/

def num_math_books : ℕ := 4
def num_english_books : ℕ := 6
def num_science_books : ℕ := 3

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def num_arrangements :=
  factorial 3 * factorial num_math_books * factorial num_english_books * factorial num_science_books

theorem books_arrangement : num_arrangements = 622080 := by
  sorry

end books_arrangement_l165_165951


namespace emily_trip_duration_same_l165_165242

theorem emily_trip_duration_same (s : ℝ) (h_s_pos : 0 < s) : 
  let t1 := (90 : ℝ) / s
  let t2 := (360 : ℝ) / (4 * s)
  t2 = t1 := sorry

end emily_trip_duration_same_l165_165242


namespace number_of_flowers_alissa_picked_l165_165251

-- Define the conditions
variable (A : ℕ) -- Number of flowers Alissa picked
variable (M : ℕ) -- Number of flowers Melissa picked
variable (flowers_gifted : ℕ := 18) -- Flowers given to mother
variable (flowers_left : ℕ := 14) -- Flowers left after gifting

-- Define that Melissa picked the same number of flowers as Alissa
axiom pick_equal : M = A

-- Define the total number of flowers they had initially
axiom total_flowers : 2 * A = flowers_gifted + flowers_left

-- Prove that Alissa picked 16 flowers
theorem number_of_flowers_alissa_picked : A = 16 := by
  -- Use placeholders for proof steps
  sorry

end number_of_flowers_alissa_picked_l165_165251


namespace triangle_inequality_from_inequality_l165_165440

theorem triangle_inequality_from_inequality
  (a b c : ℝ)
  (h : 0 < a ∧ 0 < b ∧ 0 < c)
  (ineq : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end triangle_inequality_from_inequality_l165_165440


namespace increase_percent_exceeds_l165_165209

theorem increase_percent_exceeds (p q M : ℝ) (M_positive : 0 < M) (p_positive : 0 < p) (q_positive : 0 < q) (q_less_p : q < p) :
  (M * (1 + p / 100) * (1 + q / 100) > M) ↔ (0 < p ∧ 0 < q) :=
by
  sorry

end increase_percent_exceeds_l165_165209


namespace angle_A_is_60_degrees_l165_165245

theorem angle_A_is_60_degrees
  (a b c : ℝ) (A : ℝ) 
  (h1 : (a + b + c) * (b + c - a) = 3 * b * c) 
  (h2 : 0 < A) (h3 : A < 180) : 
  A = 60 := 
  sorry

end angle_A_is_60_degrees_l165_165245


namespace proof_one_third_of_seven_times_nine_subtract_three_l165_165648

def one_third_of_seven_times_nine_subtract_three : ℕ :=
  let product := 7 * 9
  let one_third := product / 3
  one_third - 3

theorem proof_one_third_of_seven_times_nine_subtract_three : one_third_of_seven_times_nine_subtract_three = 18 := by
  sorry

end proof_one_third_of_seven_times_nine_subtract_three_l165_165648


namespace negation_of_proposition_l165_165574

noncomputable def original_proposition :=
  ∀ a b : ℝ, (a * b = 0) → (a = 0)

theorem negation_of_proposition :
  ¬ original_proposition ↔ ∃ a b : ℝ, (a * b = 0) ∧ (a ≠ 0) :=
by
  sorry

end negation_of_proposition_l165_165574


namespace inequality_proof_l165_165143

theorem inequality_proof (x y z : ℝ) (hx : x ≥ y) (hy : y ≥ z) (hz : z > 0) :
  (x^2 * y / z + y^2 * z / x + z^2 * x / y) ≥ (x^2 + y^2 + z^2) := 
  sorry

end inequality_proof_l165_165143


namespace not_every_constant_is_geometric_l165_165145

def is_constant_sequence (s : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, s n = s m

def is_geometric_sequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, s (n + 1) = r * s n

theorem not_every_constant_is_geometric :
  (¬ ∀ s : ℕ → ℝ, is_constant_sequence s → is_geometric_sequence s) ↔
  ∃ s : ℕ → ℝ, is_constant_sequence s ∧ ¬ is_geometric_sequence s := 
by
  sorry

end not_every_constant_is_geometric_l165_165145


namespace complex_in_second_quadrant_l165_165549

-- Define the complex number z based on the problem conditions.
def z : ℂ := Complex.I + (Complex.I^6)

-- State the condition to check whether z is in the second quadrant.
def is_in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- Formulate the theorem stating that the complex number z is in the second quadrant.
theorem complex_in_second_quadrant : is_in_second_quadrant z :=
by
  sorry

end complex_in_second_quadrant_l165_165549


namespace hyperbola_standard_equation_l165_165701

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_standard_equation
  (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b)
  (focus_distance_condition : ∃ (F1 F2 : ℝ), |F1 - F2| = 2 * (c a b))
  (circle_intersects_asymptote : ∃ (x y : ℝ), (x, y) = (1, 2) ∧ y = (b/a) * x + 2): 
  (a = 1) ∧ (b = 2) → (x^2 - (y^2 / 4) = 1) := 
sorry

end hyperbola_standard_equation_l165_165701


namespace Owen_spending_on_burgers_in_June_l165_165469

theorem Owen_spending_on_burgers_in_June (daily_burgers : ℕ) (cost_per_burger : ℕ) (days_in_June : ℕ) :
  daily_burgers = 2 → 
  cost_per_burger = 12 → 
  days_in_June = 30 → 
  daily_burgers * cost_per_burger * days_in_June = 720 :=
by
  intros
  sorry

end Owen_spending_on_burgers_in_June_l165_165469


namespace fold_paper_crease_length_l165_165644

theorem fold_paper_crease_length 
    (w l : ℝ) (w_pos : w = 12) (l_pos : l = 16) 
    (F G : ℝ × ℝ) (F_on_AD : F = (0, 12))
    (G_on_BC : G = (16, 12)) :
    dist F G = 20 := 
by
  sorry

end fold_paper_crease_length_l165_165644


namespace c_geq_one_l165_165171

open Real

theorem c_geq_one (a b : ℕ) (c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h_eqn : (a + 1) / (b + c) = b / a) : c ≥ 1 :=
by
  sorry

end c_geq_one_l165_165171


namespace waste_scientific_notation_correct_l165_165671

def total_waste_in_scientific : ℕ := 500000000000

theorem waste_scientific_notation_correct :
  total_waste_in_scientific = 5 * 10^10 :=
by
  sorry

end waste_scientific_notation_correct_l165_165671


namespace sarah_age_is_26_l165_165434

theorem sarah_age_is_26 (mark_age billy_age ana_age : ℕ) (sarah_age : ℕ) 
  (h1 : sarah_age = 3 * mark_age - 4)
  (h2 : mark_age = billy_age + 4)
  (h3 : billy_age = ana_age / 2)
  (h4 : ana_age = 15 - 3) :
  sarah_age = 26 := 
sorry

end sarah_age_is_26_l165_165434


namespace trains_meet_in_16_67_seconds_l165_165402

noncomputable def TrainsMeetTime (length1 length2 distance initial_speed1 initial_speed2 : ℝ) : ℝ := 
  let speed1 := initial_speed1 * 1000 / 3600
  let speed2 := initial_speed2 * 1000 / 3600
  let relativeSpeed := speed1 + speed2
  let totalDistance := distance + length1 + length2
  totalDistance / relativeSpeed

theorem trains_meet_in_16_67_seconds : 
  TrainsMeetTime 100 200 450 90 72 = 16.67 := 
by 
  sorry

end trains_meet_in_16_67_seconds_l165_165402


namespace harkamal_total_amount_l165_165578

-- Define the conditions as constants
def quantity_grapes : ℕ := 10
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost of grapes and mangoes based on the given conditions
def cost_grapes : ℕ := quantity_grapes * rate_grapes
def cost_mangoes : ℕ := quantity_mangoes * rate_mangoes

-- Define the total amount paid
def total_amount_paid : ℕ := cost_grapes + cost_mangoes

-- The theorem stating the problem and the solution
theorem harkamal_total_amount : total_amount_paid = 1195 := by
  -- Proof goes here (omitted)
  sorry

end harkamal_total_amount_l165_165578


namespace complementary_angle_decrease_l165_165559

theorem complementary_angle_decrease :
  (ratio : ℚ := 3 / 7) →
  let total_angle := 90
  let small_angle := (ratio * total_angle) / (1+ratio)
  let large_angle := total_angle - small_angle
  let new_small_angle := small_angle * 1.2
  let new_large_angle := total_angle - new_small_angle
  let decrease_percent := (large_angle - new_large_angle) / large_angle * 100
  decrease_percent = 8.57 :=
by
  sorry

end complementary_angle_decrease_l165_165559


namespace largest_divisor_product_of_consecutive_odds_l165_165892

theorem largest_divisor_product_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) : 
  15 ∣ (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) :=
sorry

end largest_divisor_product_of_consecutive_odds_l165_165892


namespace find_sum_of_relatively_prime_integers_l165_165370

theorem find_sum_of_relatively_prime_integers :
  ∃ (x y : ℕ), x * y + x + y = 119 ∧ x < 25 ∧ y < 25 ∧ Nat.gcd x y = 1 ∧ x + y = 20 :=
by
  sorry

end find_sum_of_relatively_prime_integers_l165_165370


namespace hose_removal_rate_l165_165587

theorem hose_removal_rate (w l d : ℝ) (capacity_fraction : ℝ) (drain_time : ℝ) 
  (h_w : w = 60) 
  (h_l : l = 150) 
  (h_d : d = 10) 
  (h_capacity_fraction : capacity_fraction = 0.80) 
  (h_drain_time : drain_time = 1200) : 
  ((w * l * d * capacity_fraction) / drain_time) = 60 :=
by
  -- the proof is omitted here
  sorry

end hose_removal_rate_l165_165587


namespace find_constants_l165_165640

-- Given definitions based on the conditions and conjecture
def S (n : ℕ) : ℕ := 
  match n with
  | 1 => 1
  | 2 => 5
  | 3 => 15
  | 4 => 34
  | 5 => 65
  | _ => 0

noncomputable def conjecture_S (n a b c : ℤ) := (2 * n - 1) * (a * n^2 + b * n + c)

theorem find_constants (a b c : ℤ) (h1 : conjecture_S 1 a b c = 1) (h2 : conjecture_S 2 a b c = 5) (h3 : conjecture_S 3 a b c = 15) : 3 * a + b = 4 :=
by
  -- Proof omitted
  sorry

end find_constants_l165_165640


namespace expression_value_l165_165293

theorem expression_value (x : ℝ) (h : x = 3 + 5 / (2 + 5 / x)) : x = 5 :=
sorry

end expression_value_l165_165293


namespace simplify_and_evaluate_l165_165430

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) : (x + 5)^2 - (x - 2) * (x + 2) = 9 :=
by
  rw [h]
  -- Continue with standard proof techniques here
  sorry

end simplify_and_evaluate_l165_165430


namespace square_area_l165_165005

noncomputable def line_lies_on_square_side (a b : ℝ) : Prop :=
  ∃ (A B : ℝ × ℝ), A = (a, a + 4) ∧ B = (b, b + 4)

noncomputable def points_on_parabola (x y : ℝ) : Prop :=
  ∃ (C D : ℝ × ℝ), C = (y^2, y) ∧ D = (x^2, x)

theorem square_area (a b : ℝ) (x y : ℝ)
  (h1 : line_lies_on_square_side a b)
  (h2 : points_on_parabola x y) :
  ∃ (s : ℝ), s^2 = (boxed_solution) :=
sorry

end square_area_l165_165005


namespace Yan_ratio_distance_l165_165280

theorem Yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq : y/w = x/w + (x + y)/(5 * w)) : x/y = 2/3 := by
  sorry

end Yan_ratio_distance_l165_165280


namespace ordered_pair_solution_l165_165577

theorem ordered_pair_solution :
  ∃ x y : ℚ, 7 * x - 50 * y = 3 ∧ 3 * y - x = 5 ∧ x = -259 / 29 ∧ y = -38 / 29 :=
by sorry

end ordered_pair_solution_l165_165577


namespace find_fraction_l165_165655

theorem find_fraction
  (x : ℝ)
  (h : (x)^35 * (1/4)^18 = 1 / (2 * 10^35)) : x = 1/5 :=
by 
  sorry

end find_fraction_l165_165655


namespace monthly_cost_per_person_is_1000_l165_165962

noncomputable def john_pays : ℝ := 32000
noncomputable def initial_fee_per_person : ℝ := 4000
noncomputable def total_people : ℝ := 4
noncomputable def john_pays_half : Prop := true

theorem monthly_cost_per_person_is_1000 :
  john_pays_half →
  (john_pays * 2 - (initial_fee_per_person * total_people)) / (total_people * 12) = 1000 :=
by
  intro h
  sorry

end monthly_cost_per_person_is_1000_l165_165962


namespace coordinates_of_B_l165_165635

/--
Given point A with coordinates (2, -3) and line segment AB parallel to the x-axis,
and the length of AB being 4, prove that the coordinates of point B are either (-2, -3)
or (6, -3).
-/
theorem coordinates_of_B (x1 y1 : ℝ) (d : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : d = 4) (hx : 0 ≤ d) :
  ∃ x2 : ℝ, ∃ y2 : ℝ, (y2 = y1) ∧ ((x2 = x1 + d) ∨ (x2 = x1 - d)) :=
by
  sorry

end coordinates_of_B_l165_165635


namespace distinct_nonzero_reals_xy_six_l165_165260

theorem distinct_nonzero_reals_xy_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 6/x = y + 6/y) (h_distinct : x ≠ y) : x * y = 6 := 
sorry

end distinct_nonzero_reals_xy_six_l165_165260


namespace tamtam_blue_shells_l165_165390

theorem tamtam_blue_shells 
  (total_shells : ℕ)
  (purple_shells : ℕ)
  (pink_shells : ℕ)
  (yellow_shells : ℕ)
  (orange_shells : ℕ)
  (H_total : total_shells = 65)
  (H_purple : purple_shells = 13)
  (H_pink : pink_shells = 8)
  (H_yellow : yellow_shells = 18)
  (H_orange : orange_shells = 14) :
  ∃ blue_shells : ℕ, blue_shells = 12 :=
by
  sorry

end tamtam_blue_shells_l165_165390


namespace tan_A_minus_B_l165_165727

theorem tan_A_minus_B (A B : ℝ) (h1: Real.cos A = -Real.sqrt 2 / 2) (h2 : Real.tan B = 1 / 3) : 
  Real.tan (A - B) = -2 := by
  sorry

end tan_A_minus_B_l165_165727


namespace smallest_nat_number_l165_165947

theorem smallest_nat_number (n : ℕ) (h1 : ∃ a, 0 ≤ a ∧ a < 20 ∧ n % 20 = a ∧ n % 21 = a + 1) (h2 : n % 22 = 2) : n = 838 := by 
  sorry

end smallest_nat_number_l165_165947


namespace find_digit_B_l165_165811

theorem find_digit_B (A B : ℕ) (h : 1 ≤ A ∧ A ≤ 9) (h' : 0 ≤ B ∧ B ≤ 9) (eqn : 10 * A + 22 = 9 * B) : B = 8 := 
  sorry

end find_digit_B_l165_165811


namespace total_minutes_exercised_l165_165302

-- Defining the conditions
def Javier_minutes_per_day : Nat := 50
def Javier_days : Nat := 10

def Sanda_minutes_day_90 : Nat := 90
def Sanda_days_90 : Nat := 3

def Sanda_minutes_day_75 : Nat := 75
def Sanda_days_75 : Nat := 2

def Sanda_minutes_day_45 : Nat := 45
def Sanda_days_45 : Nat := 4

-- Main statement to prove
theorem total_minutes_exercised : 
  (Javier_minutes_per_day * Javier_days) + 
  (Sanda_minutes_day_90 * Sanda_days_90) +
  (Sanda_minutes_day_75 * Sanda_days_75) +
  (Sanda_minutes_day_45 * Sanda_days_45) = 1100 := by
  sorry

end total_minutes_exercised_l165_165302


namespace find_original_revenue_l165_165540

variable (currentRevenue : ℝ) (percentageDecrease : ℝ)
noncomputable def originalRevenue (currentRevenue : ℝ) (percentageDecrease : ℝ) : ℝ :=
  currentRevenue / (1 - percentageDecrease)

theorem find_original_revenue (h1 : currentRevenue = 48.0) (h2 : percentageDecrease = 0.3333333333333333) :
  originalRevenue currentRevenue percentageDecrease = 72.0 := by
  rw [h1, h2]
  unfold originalRevenue
  norm_num
  sorry

end find_original_revenue_l165_165540


namespace prob_one_boy_one_girl_l165_165781

-- Defining the probabilities of birth
def prob_boy := 2 / 3
def prob_girl := 1 / 3

-- Calculating the probability of all boys
def prob_all_boys := prob_boy ^ 4

-- Calculating the probability of all girls
def prob_all_girls := prob_girl ^ 4

-- Calculating the probability of having at least one boy and one girl
def prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)

-- Proof statement
theorem prob_one_boy_one_girl : prob_at_least_one_boy_and_one_girl = 64 / 81 :=
by sorry

end prob_one_boy_one_girl_l165_165781


namespace midpoint_trajectory_l165_165454

   -- Defining the given conditions
   def P_moves_on_circle (x1 y1 : ℝ) : Prop :=
     (x1 + 1)^2 + y1^2 = 4

   def Q_coordinates : (ℝ × ℝ) := (4, 3)

   -- Defining the midpoint relationship
   def midpoint_relation (x y x1 y1 : ℝ) : Prop :=
     x1 + Q_coordinates.1 = 2 * x ∧ y1 + Q_coordinates.2 = 2 * y

   -- Proving the trajectory equation of the midpoint M
   theorem midpoint_trajectory (x y : ℝ) : 
     (∃ x1 y1 : ℝ, midpoint_relation x y x1 y1 ∧ P_moves_on_circle x1 y1) →
     (x - 3/2)^2 + (y - 3/2)^2 = 1 :=
   by
     intros h
     sorry
   
end midpoint_trajectory_l165_165454


namespace smaller_angle_linear_pair_l165_165820

theorem smaller_angle_linear_pair (a b : ℝ) (h1 : a + b = 180) (h2 : a = 5 * b) : b = 30 := by
  sorry

end smaller_angle_linear_pair_l165_165820


namespace inequality_check_l165_165957

theorem inequality_check : (-1 : ℝ) / 3 < -1 / 5 := 
by 
  sorry

end inequality_check_l165_165957


namespace john_total_distance_l165_165567

-- Define the given conditions
def initial_speed : ℝ := 45 -- mph
def first_leg_time : ℝ := 2 -- hours
def second_leg_time : ℝ := 3 -- hours
def distance_before_lunch : ℝ := initial_speed * first_leg_time
def distance_after_lunch : ℝ := initial_speed * second_leg_time

-- Define the total distance
def total_distance : ℝ := distance_before_lunch + distance_after_lunch

-- Prove the total distance is 225 miles
theorem john_total_distance : total_distance = 225 := by
  sorry

end john_total_distance_l165_165567


namespace roy_age_product_l165_165690

theorem roy_age_product (R J K : ℕ) 
  (h1 : R = J + 8)
  (h2 : R = K + (R - J) / 2)
  (h3 : R + 2 = 3 * (J + 2)) :
  (R + 2) * (K + 2) = 96 :=
by
  sorry

end roy_age_product_l165_165690


namespace range_of_a_l165_165233

variable (a : ℝ) (x : ℝ)

theorem range_of_a
  (h1 : 2 * x < 3 * (x - 3) + 1)
  (h2 : (3 * x + 2) / 4 > x + a) :
  -11 / 4 ≤ a ∧ a < -5 / 2 :=
sorry

end range_of_a_l165_165233


namespace negation_of_p_negation_of_q_l165_165412

def p (x : ℝ) : Prop := x > 0 → x^2 - 5 * x ≥ -25 / 4

def even (n : ℕ) : Prop := ∃ k, n = 2 * k

def q : Prop := ∃ n, even n ∧ ∃ m, n = 3 * m

theorem negation_of_p : ¬(∀ x : ℝ, x > 0 → x^2 - 5 * x ≥ - 25 / 4) → ∃ x : ℝ, x > 0 ∧ x^2 - 5 * x < - 25 / 4 := 
by sorry

theorem negation_of_q : ¬ (∃ n : ℕ, even n ∧ ∃ m : ℕ, n = 3 * m) → ∀ n : ℕ, even n → ¬ (∃ m : ℕ, n = 3 * m) := 
by sorry

end negation_of_p_negation_of_q_l165_165412


namespace convert_rect_to_polar_l165_165734

theorem convert_rect_to_polar (y x : ℝ) (h : y = x) : ∃ θ : ℝ, θ = π / 4 :=
by
  sorry

end convert_rect_to_polar_l165_165734


namespace remainder_division_by_8_is_6_l165_165312

theorem remainder_division_by_8_is_6 (N Q2 R1 : ℤ) (h1 : N = 64 + R1) (h2 : N % 5 = 4) : R1 = 6 :=
by
  sorry

end remainder_division_by_8_is_6_l165_165312


namespace hattie_jumps_l165_165867

theorem hattie_jumps (H : ℝ) (h1 : Lorelei_jumps1 = (3/4) * H)
  (h2 : Hattie_jumps2 = (2/3) * H)
  (h3 : Lorelei_jumps2 = (2/3) * H + 50)
  (h4 : H + Lorelei_jumps1 + Hattie_jumps2 + Lorelei_jumps2 = 605) : H = 180 :=
by
  sorry

noncomputable def Lorelei_jumps1 (H : ℝ) := (3/4) * H
noncomputable def Hattie_jumps2 (H : ℝ) := (2/3) * H
noncomputable def Lorelei_jumps2 (H : ℝ) := (2/3) * H + 50

end hattie_jumps_l165_165867


namespace cooler_capacity_l165_165167

theorem cooler_capacity (linemen: ℕ) (linemen_drink: ℕ) 
                        (skill_position: ℕ) (skill_position_drink: ℕ) 
                        (linemen_count: ℕ) (skill_position_count: ℕ) 
                        (skill_wait: ℕ) 
                        (h1: linemen_count = 12) 
                        (h2: linemen_drink = 8) 
                        (h3: skill_position_count = 10) 
                        (h4: skill_position_drink = 6) 
                        (h5: skill_wait = 5):
 linemen_count * linemen_drink + skill_wait * skill_position_drink = 126 :=
by
  sorry

end cooler_capacity_l165_165167


namespace least_prime_factor_five_power_difference_l165_165959

theorem least_prime_factor_five_power_difference : 
  ∃ p : ℕ, (Nat.Prime p ∧ p ∣ (5^4 - 5^3)) ∧ ∀ q : ℕ, (Nat.Prime q ∧ q ∣ (5^4 - 5^3) → p ≤ q) := 
sorry

end least_prime_factor_five_power_difference_l165_165959


namespace card_draw_probability_l165_165291

-- Define a function to compute the probability of a sequence of draws
noncomputable def probability_of_event : Rat :=
  (4 / 52) * (4 / 51) * (1 / 50)

theorem card_draw_probability :
  probability_of_event = 4 / 33150 :=
by
  -- Proof goes here
  sorry

end card_draw_probability_l165_165291


namespace identifyNewEnergySources_l165_165467

-- Definitions of energy types as elements of a set.
inductive EnergySource 
| NaturalGas
| Coal
| OceanEnergy
| Petroleum
| SolarEnergy
| BiomassEnergy
| WindEnergy
| HydrogenEnergy

open EnergySource

-- Set definition for types of new energy sources
def newEnergySources : Set EnergySource := 
  { OceanEnergy, SolarEnergy, BiomassEnergy, WindEnergy, HydrogenEnergy }

-- Set definition for the correct answer set of new energy sources identified by Option B
def optionB : Set EnergySource := 
  { OceanEnergy, SolarEnergy, BiomassEnergy, WindEnergy, HydrogenEnergy }

-- The theorem asserting the equivalence between the identified new energy sources and the set option B
theorem identifyNewEnergySources : newEnergySources = optionB :=
  sorry

end identifyNewEnergySources_l165_165467


namespace dan_money_left_l165_165800

def initial_money : ℝ := 50.00
def candy_bar_price : ℝ := 1.75
def candy_bar_count : ℕ := 3
def gum_price : ℝ := 0.85
def soda_price : ℝ := 2.25
def sales_tax_rate : ℝ := 0.08

theorem dan_money_left : 
  initial_money - (candy_bar_count * candy_bar_price + gum_price + soda_price) * (1 + sales_tax_rate) = 40.98 :=
by
  sorry

end dan_money_left_l165_165800


namespace rearrange_expression_l165_165357

theorem rearrange_expression :
  1 - 2 - 3 - 4 - (5 - 6 - 7) = 0 :=
by
  sorry

end rearrange_expression_l165_165357


namespace seeds_germination_l165_165054

theorem seeds_germination (seed_plot1 seed_plot2 : ℕ) (germ_rate2 total_germ_rate : ℝ) (germinated_total_pct : ℝ)
  (h1 : seed_plot1 = 300)
  (h2 : seed_plot2 = 200)
  (h3 : germ_rate2 = 0.35)
  (h4 : germinated_total_pct = 28.999999999999996 / 100) :
  (germinated_total_pct * (seed_plot1 + seed_plot2) - germ_rate2 * seed_plot2) / seed_plot1 * 100 = 25 :=
by sorry  -- Proof not required

end seeds_germination_l165_165054


namespace cody_initial_marbles_l165_165208

theorem cody_initial_marbles (x : ℕ) (h1 : x - 5 = 7) : x = 12 := by
  sorry

end cody_initial_marbles_l165_165208


namespace sqrt_1_plus_inv_squares_4_5_sqrt_1_plus_inv_squares_general_sqrt_101_100_plus_1_121_l165_165705

open Real

theorem sqrt_1_plus_inv_squares_4_5 :
  sqrt (1 + 1/4^2 + 1/5^2) = 1 + 1/20 :=
by
  sorry

theorem sqrt_1_plus_inv_squares_general (n : ℕ) (h : 0 < n) :
  sqrt (1 + 1/n^2 + 1/(n+1)^2) = 1 + 1/(n * (n + 1)) :=
by
  sorry

theorem sqrt_101_100_plus_1_121 :
  sqrt (101/100 + 1/121) = 1 + 1/110 :=
by
  sorry

end sqrt_1_plus_inv_squares_4_5_sqrt_1_plus_inv_squares_general_sqrt_101_100_plus_1_121_l165_165705


namespace prove_equivalence_l165_165431

variable (x : ℝ)

def operation1 (x : ℝ) : ℝ := 8 - x

def operation2 (x : ℝ) : ℝ := x - 8

theorem prove_equivalence : operation2 (operation1 14) = -14 := by
  sorry

end prove_equivalence_l165_165431


namespace total_new_cans_l165_165096

-- Define the condition
def initial_cans : ℕ := 256
def first_term : ℕ := 64
def ratio : ℚ := 1 / 4
def terms : ℕ := 4

-- Define the sum of the geometric series
noncomputable def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r ^ n) / (1 - r))

-- Problem statement in Lean 4
theorem total_new_cans : geometric_series_sum first_term ratio terms = 85 := by
  sorry

end total_new_cans_l165_165096


namespace train_speed_is_260_kmph_l165_165912

-- Define the conditions: length of the train and time to cross the pole
def length_of_train : ℝ := 130
def time_to_cross_pole : ℝ := 9

-- Define the conversion factor from meters per second to kilometers per hour
def conversion_factor : ℝ := 3.6

-- Define the expected speed in kilometers per hour
def expected_speed_kmph : ℝ := 260

-- The theorem statement
theorem train_speed_is_260_kmph :
  (length_of_train / time_to_cross_pole) * conversion_factor = expected_speed_kmph :=
sorry

end train_speed_is_260_kmph_l165_165912


namespace find_x_l165_165287

def seq : ℕ → ℕ
| 0 => 2
| 1 => 5
| 2 => 11
| 3 => 20
| 4 => 32
| 5 => 47
| (n+6) => seq (n+5) + 3 * (n + 1)

theorem find_x : seq 6 = 65 := by
  sorry

end find_x_l165_165287


namespace final_price_on_monday_l165_165525

-- Definitions based on the conditions
def saturday_price : ℝ := 50
def sunday_increase : ℝ := 1.2
def monday_discount : ℝ := 0.2

-- The statement to prove
theorem final_price_on_monday : 
  let sunday_price := saturday_price * sunday_increase
  let monday_price := sunday_price * (1 - monday_discount)
  monday_price = 48 :=
by
  sorry

end final_price_on_monday_l165_165525


namespace Brad_age_l165_165457

theorem Brad_age (shara_age : ℕ) (h_shara : shara_age = 10)
  (jaymee_age : ℕ) (h_jaymee : jaymee_age = 2 * shara_age + 2)
  (brad_age : ℕ) (h_brad : brad_age = (shara_age + jaymee_age) / 2 - 3) : brad_age = 13 := by
  sorry

end Brad_age_l165_165457


namespace bricks_needed_for_room_floor_l165_165341

-- Conditions
def length : ℕ := 4
def breadth : ℕ := 5
def bricks_per_square_meter : ℕ := 17

-- Question and Answer (Proof Problem)
theorem bricks_needed_for_room_floor : 
  (length * breadth) * bricks_per_square_meter = 340 := by
  sorry

end bricks_needed_for_room_floor_l165_165341


namespace remainder_1235678_div_127_l165_165465

theorem remainder_1235678_div_127 : 1235678 % 127 = 69 := by
  sorry

end remainder_1235678_div_127_l165_165465


namespace problem_statement_l165_165856

theorem problem_statement (a b : ℝ) (h1 : a + b = 3) (h2 : a - b = 5) : b^2 - a^2 = -15 := by
  sorry

end problem_statement_l165_165856


namespace plywood_long_side_length_l165_165677

theorem plywood_long_side_length (L : ℕ) (h1 : 2 * (L + 5) = 22) : L = 6 :=
by
  sorry

end plywood_long_side_length_l165_165677


namespace train_crossing_time_l165_165944

variable (length_train : ℝ) (time_pole : ℝ) (length_platform : ℝ) (time_platform : ℝ)

-- Given conditions
def train_conditions := 
  length_train = 300 ∧
  time_pole = 14 ∧
  length_platform = 535.7142857142857

-- Theorem statement
theorem train_crossing_time (h : train_conditions length_train time_pole length_platform) :
  time_platform = 39 := sorry

end train_crossing_time_l165_165944


namespace solve_for_x_l165_165963

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 17) : x = 4 :=
by {
  sorry
}

end solve_for_x_l165_165963


namespace area_change_correct_l165_165134

theorem area_change_correct (L B : ℝ) (A : ℝ) (x : ℝ) (hx1 : A = L * B)
  (hx2 : ((L + (x / 100) * L) * (B - (x / 100) * B)) = A - (1 / 100) * A) :
  x = 10 := by
  sorry

end area_change_correct_l165_165134


namespace log2_75_in_terms_of_a_b_l165_165719

noncomputable def log_base2 (x : ℝ) : ℝ := Real.log x / Real.log 2

variables (a b : ℝ)
variables (log2_9_eq_a : log_base2 9 = a)
variables (log2_5_eq_b : log_base2 5 = b)

theorem log2_75_in_terms_of_a_b : log_base2 75 = (1 / 2) * a + 2 * b :=
by sorry

end log2_75_in_terms_of_a_b_l165_165719


namespace worm_in_apple_l165_165329

theorem worm_in_apple (radius : ℝ) (travel_distance : ℝ) (h_radius : radius = 31) (h_travel_distance : travel_distance = 61) :
  ∃ S : Set ℝ, ∀ point_on_path : ℝ, (point_on_path ∈ S) → false :=
by
  sorry

end worm_in_apple_l165_165329


namespace max_number_of_different_ages_l165_165974

theorem max_number_of_different_ages
  (a : ℤ) (s : ℤ)
  (h1 : a = 31)
  (h2 : s = 5) :
  ∃ n : ℕ, n = (36 - 26 + 1) :=
by sorry

end max_number_of_different_ages_l165_165974


namespace pizza_pieces_per_person_l165_165706

theorem pizza_pieces_per_person (total_people : ℕ) (fraction_eat : ℚ) (total_pizza : ℕ) (remaining_pizza : ℕ)
  (H1 : total_people = 15) (H2 : fraction_eat = 3/5) (H3 : total_pizza = 50) (H4 : remaining_pizza = 14) :
  (total_pizza - remaining_pizza) / (fraction_eat * total_people) = 4 :=
by
  -- proof goes here
  sorry

end pizza_pieces_per_person_l165_165706


namespace calculate_si_l165_165323

section SimpleInterest

def Principal : ℝ := 10000
def Rate : ℝ := 0.04
def Time : ℝ := 1
def SimpleInterest : ℝ := Principal * Rate * Time

theorem calculate_si : SimpleInterest = 400 := by
  -- Proof goes here.
  sorry

end SimpleInterest

end calculate_si_l165_165323


namespace single_elimination_games_l165_165069

theorem single_elimination_games (n : ℕ) (h : n = 23) : 
  ∃ games : ℕ, games = n - 1 :=
by
  use 22
  sorry

end single_elimination_games_l165_165069


namespace variation_of_variables_l165_165230

variables (k j : ℝ) (x y z : ℝ)

theorem variation_of_variables (h1 : x = k * y^2) (h2 : y = j * z^3) : ∃ m : ℝ, x = m * z^6 :=
by
  -- Placeholder for the proof
  sorry

end variation_of_variables_l165_165230


namespace minimum_room_size_for_table_l165_165782

theorem minimum_room_size_for_table (S : ℕ) :
  (∃ S, S ≥ 13) := sorry

end minimum_room_size_for_table_l165_165782


namespace factorize_expr_l165_165192

-- Define the expression
def expr (a : ℝ) := -3 * a + 12 * a^2 - 12 * a^3

-- State the theorem
theorem factorize_expr (a : ℝ) : expr a = -3 * a * (1 - 2 * a)^2 :=
by
  sorry

end factorize_expr_l165_165192


namespace solution_set_l165_165193

noncomputable def satisfies_inequality (x : ℝ) : Prop :=
  (x - 2) / (x - 4) ≥ 3

theorem solution_set :
  {x : ℝ | satisfies_inequality x} = {x : ℝ | 4 < x ∧ x ≤ 5} :=
by
  sorry

end solution_set_l165_165193


namespace angle_QRS_determination_l165_165608

theorem angle_QRS_determination (PQ_parallel_RS : ∀ (P Q R S T : Type) 
  (angle_PTQ : ℝ) (angle_SRT : ℝ), 
  PQ_parallel_RS → (angle_PTQ = angle_SRT) → (angle_PTQ = 4 * angle_SRT - 120)) 
  (angle_SRT : ℝ) (angle_QRS : ℝ) 
  (h : angle_SRT = 4 * angle_SRT - 120) : angle_QRS = 40 :=
by 
  sorry

end angle_QRS_determination_l165_165608


namespace reciprocal_of_neg3_l165_165448

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l165_165448


namespace radius_ratio_eq_inv_sqrt_5_l165_165304

noncomputable def ratio_of_radii (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
  a / b

theorem radius_ratio_eq_inv_sqrt_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : 
  ratio_of_radii a b h = 1 / Real.sqrt 5 :=
sorry

end radius_ratio_eq_inv_sqrt_5_l165_165304


namespace fraction_subtraction_l165_165991

theorem fraction_subtraction : (5 / 6 + 1 / 4 - 2 / 3) = (5 / 12) := by
  sorry

end fraction_subtraction_l165_165991


namespace greatest_possible_remainder_l165_165767

theorem greatest_possible_remainder {x : ℤ} (h : ∃ (k : ℤ), x = 11 * k + 10) : 
  ∃ y, y = 10 := sorry

end greatest_possible_remainder_l165_165767


namespace mr_kishore_savings_l165_165803

theorem mr_kishore_savings :
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let misc := 3940
  let total_expenses := rent + milk + groceries + education + petrol + misc
  let savings_percentage := 0.10
  let salary := total_expenses / (1 - savings_percentage)
  let savings := savings_percentage * salary
  savings = 1937.78 := by
  sorry

end mr_kishore_savings_l165_165803


namespace value_of_a_l165_165160

theorem value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - x - 2 < 0 ↔ -2 < x ∧ x < a) → (a = 2 ∨ a = 3 ∨ a = 4) :=
by sorry

end value_of_a_l165_165160


namespace way_to_cut_grid_l165_165313

def grid_ways : ℕ := 17

def rectangles (size : ℕ × ℕ) (count : ℕ) := 
  size = (1, 2) ∧ count = 8

def square (size : ℕ × ℕ) (count : ℕ) := 
  size = (1, 1) ∧ count = 1

theorem way_to_cut_grid :
  (∃ ways : ℕ, ways = 10) ↔ 
  ∀ g ways, g = grid_ways → 
  (rectangles (1, 2) 8 ∧ square (1, 1) 1 → ways = 10) :=
by 
  sorry

end way_to_cut_grid_l165_165313


namespace compute_ratio_d_e_l165_165712

open Polynomial

noncomputable def quartic_polynomial (a b c d e : ℚ) : Polynomial ℚ := 
  C a * X^4 + C b * X^3 + C c * X^2 + C d * X + C e

def roots_of_quartic (a b c d e: ℚ) : Prop :=
  (quartic_polynomial a b c d e).roots = {1, 2, 3, 5}

theorem compute_ratio_d_e (a b c d e : ℚ) 
    (h : roots_of_quartic a b c d e) :
    d / e = -61 / 30 :=
  sorry

end compute_ratio_d_e_l165_165712


namespace problem1_problem2_l165_165348

-- Problem 1
theorem problem1 : 
  (-2.8) - (-3.6) + (-1.5) - (3.6) = -4.3 := 
by 
  sorry

-- Problem 2
theorem problem2 :
  (- (5 / 6 : ℚ) + (1 / 3 : ℚ) - (3 / 4 : ℚ)) * (-24) = 30 := 
by 
  sorry

end problem1_problem2_l165_165348


namespace angle_solution_exists_l165_165788

theorem angle_solution_exists :
  ∃ (x : ℝ), 0 < x ∧ x < 180 ∧ 9 * (Real.sin x) * (Real.cos x)^4 - 9 * (Real.sin x)^4 * (Real.cos x) = 1 / 2 ∧ x = 30 :=
by
  sorry

end angle_solution_exists_l165_165788


namespace mary_investment_amount_l165_165907

theorem mary_investment_amount
  (A : ℝ := 100000) -- Future value in dollars
  (r : ℝ := 0.08) -- Annual interest rate
  (n : ℕ := 12) -- Compounded monthly
  (t : ℝ := 10) -- Time in years
  : (⌈A / (1 + r / n) ^ (n * t)⌉₊ = 45045) :=
by
  sorry

end mary_investment_amount_l165_165907


namespace shooting_competition_l165_165686

variable (x y : ℕ)

theorem shooting_competition (H1 : 20 * x - 12 * (10 - x) + 20 * y - 12 * (10 - y) = 208)
                             (H2 : 20 * x - 12 * (10 - x) = 20 * y - 12 * (10 - y) + 64) :
  x = 8 ∧ y = 6 := 
by 
  sorry

end shooting_competition_l165_165686


namespace find_f_2006_l165_165436

-- Assuming an odd periodic function f with period 3(3x+1), defining the conditions.
def f : ℤ → ℤ := sorry -- Definition of f is not provided.

-- Conditions
axiom odd_function : ∀ x : ℤ, f (-x) = -f x
axiom period_3_function : ∀ x : ℤ, f (3 * x + 1) = f (3 * (x + 1) + 1)
axiom value_at_1 : f 1 = -1

-- Question: What is f(2006)?
theorem find_f_2006 : f 2006 = 1 := sorry

end find_f_2006_l165_165436


namespace find_a_l165_165164

def M : Set ℝ := {-1, 0, 1}

def N (a : ℝ) : Set ℝ := {a, a^2}

theorem find_a (a : ℝ) : N a ⊆ M → a = -1 :=
by
  sorry

end find_a_l165_165164


namespace slope_of_line_l165_165349

theorem slope_of_line (x y : ℝ) (h : 4 * y = 5 * x + 20) : y = (5/4) * x + 5 :=
by {
  sorry
}

end slope_of_line_l165_165349


namespace min_value_expression_l165_165862

theorem min_value_expression (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 = 4 :=
sorry

end min_value_expression_l165_165862


namespace average_infection_rate_infected_computers_exceed_700_l165_165298

theorem average_infection_rate (h : (1 + x) ^ 2 = 81) : x = 8 := by
  sorry

theorem infected_computers_exceed_700 (h_infection_rate : 8 = 8) : (1 + 8) ^ 3 > 700 := by
  sorry

end average_infection_rate_infected_computers_exceed_700_l165_165298


namespace book_page_count_l165_165048

def total_pages_in_book (pages_three_nights_ago pages_two_nights_ago pages_last_night pages_tonight total_pages : ℕ) : Prop :=
  pages_three_nights_ago = 15 ∧
  pages_two_nights_ago = 2 * pages_three_nights_ago ∧
  pages_last_night = pages_two_nights_ago + 5 ∧
  pages_tonight = 20 ∧
  total_pages = pages_three_nights_ago + pages_two_nights_ago + pages_last_night + pages_tonight

theorem book_page_count : total_pages_in_book 15 30 35 20 100 :=
by {
  sorry
}

end book_page_count_l165_165048


namespace max_term_in_sequence_l165_165709

theorem max_term_in_sequence (a : ℕ → ℝ)
  (h : ∀ n, a n = (n+1) * (7/8)^n) :
  (∀ n, a n ≤ a 6 ∨ a n ≤ a 7) ∧ (a 6 = max (a 6) (a 7)) ∧ (a 7 = max (a 6) (a 7)) :=
sorry

end max_term_in_sequence_l165_165709


namespace percent_not_participating_music_sports_l165_165106

theorem percent_not_participating_music_sports
  (total_students : ℕ) 
  (both : ℕ) 
  (music_only : ℕ) 
  (sports_only : ℕ) 
  (not_participating : ℕ)
  (percentage_not_participating : ℝ) :
  total_students = 50 →
  both = 5 →
  music_only = 15 →
  sports_only = 20 →
  not_participating = total_students - (both + music_only + sports_only) →
  percentage_not_participating = (not_participating : ℝ) / (total_students : ℝ) * 100 →
  percentage_not_participating = 20 :=
by
  sorry

end percent_not_participating_music_sports_l165_165106


namespace sequence_equality_l165_165975

theorem sequence_equality (a : Fin 1973 → ℝ) (hpos : ∀ n, a n > 0)
  (heq : a 0 ^ a 0 = a 1 ^ a 2 ∧ a 1 ^ a 2 = a 2 ^ a 3 ∧ 
         a 2 ^ a 3 = a 3 ^ a 4 ∧ 
         -- etc., continued for all indices, 
         -- ensuring last index correctly refers back to a 0
         a 1971 ^ a 1972 = a 1972 ^ a 0) :
  a 0 = a 1972 :=
sorry

end sequence_equality_l165_165975


namespace find_m_plus_n_l165_165317

theorem find_m_plus_n (a m n : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a^m = n) (h4 : a^0 = 1) : m + n = 1 :=
sorry

end find_m_plus_n_l165_165317


namespace speed_of_stream_l165_165236

-- Define the conditions as premises
def boat_speed_in_still_water : ℝ := 24
def travel_time_downstream : ℝ := 3
def distance_downstream : ℝ := 84

-- The effective speed downstream is the sum of the boat's speed and the speed of the stream
def effective_speed_downstream (stream_speed : ℝ) : ℝ :=
  boat_speed_in_still_water + stream_speed

-- The speed of the stream
theorem speed_of_stream (stream_speed : ℝ) :
  84 = effective_speed_downstream stream_speed * travel_time_downstream →
  stream_speed = 4 :=
by
  sorry

end speed_of_stream_l165_165236


namespace circle_center_radius_l165_165267

theorem circle_center_radius {x y : ℝ} :
  (∃ r : ℝ, (x - 1)^2 + y^2 = r^2) ↔ (x^2 + y^2 - 2*x - 5 = 0) :=
by sorry

end circle_center_radius_l165_165267


namespace sum_of_integer_pair_l165_165956

theorem sum_of_integer_pair (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 10) (h3 : 1 ≤ b) (h4 : b ≤ 10) (h5 : a * b = 14) : a + b = 9 := 
sorry

end sum_of_integer_pair_l165_165956


namespace like_terms_powers_eq_l165_165222

theorem like_terms_powers_eq (m n : ℕ) :
  (-2 : ℝ) * (x : ℝ) * (y : ℝ) ^ m = (1 / 3 : ℝ) * (x : ℝ) ^ n * (y : ℝ) ^ 3 → m = 3 ∧ n = 1 :=
by
  sorry

end like_terms_powers_eq_l165_165222


namespace area_of_complex_polygon_l165_165651

-- Defining the problem
def area_of_polygon (side1 side2 side3 : ℝ) (rot1 rot2 : ℝ) : ℝ :=
  -- This is a placeholder definition.
  -- In a complete proof, here we would calculate the area based on the input conditions.
  sorry

-- Main theorem statement
theorem area_of_complex_polygon :
  area_of_polygon 4 5 6 (π / 4) (-π / 6) = 72 :=
by sorry

end area_of_complex_polygon_l165_165651


namespace arithmetic_geometric_fraction_l165_165129

theorem arithmetic_geometric_fraction (a x₁ x₂ b y₁ y₂ : ℝ) 
  (h₁ : x₁ + x₂ = a + b) 
  (h₂ : y₁ * y₂ = ab) : 
  (x₁ + x₂) / (y₁ * y₂) = (a + b) / (ab) := 
by
  sorry

end arithmetic_geometric_fraction_l165_165129


namespace geometric_seq_general_formula_sum_c_seq_terms_l165_165825

noncomputable def a_seq (n : ℕ) : ℕ := 2 * 3 ^ (n - 1)

noncomputable def S_seq (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (a_seq n - 2) / 2

theorem geometric_seq_general_formula (n : ℕ) (h : n > 0) : 
  a_seq n = 2 * 3 ^ (n - 1) := 
by {
  sorry
}

noncomputable def d_n (n : ℕ) : ℕ :=
  (a_seq (n + 1) - a_seq n) / (n + 1)

noncomputable def c_seq (n : ℕ) : ℕ :=
  d_n n / (n * a_seq n)

noncomputable def T_n (n : ℕ) : ℕ :=
  2 * (1 - 1 / (n + 1)) * n

theorem sum_c_seq_terms (n : ℕ) (h : n > 0) : 
  T_n n = 2 * n / (n + 1) :=
by {
  sorry
}

end geometric_seq_general_formula_sum_c_seq_terms_l165_165825


namespace interval_of_monotonic_increase_l165_165995

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem interval_of_monotonic_increase : {x : ℝ | -1 ≤ x} ⊆ {x : ℝ | 0 < deriv f x} :=
by
  sorry

end interval_of_monotonic_increase_l165_165995


namespace violet_ticket_cost_l165_165437

theorem violet_ticket_cost :
  (2 * 35 + 5 * 20 = 170) ∧
  (((35 - 17.50) + 35 + 5 * 20) = 152.50) ∧
  ((152.50 - 150) = 2.50) :=
by
  sorry

end violet_ticket_cost_l165_165437


namespace abs_inequality_solution_l165_165590

theorem abs_inequality_solution (x : ℝ) :
  abs (2 * x - 5) ≤ 7 ↔ -1 ≤ x ∧ x ≤ 6 :=
sorry

end abs_inequality_solution_l165_165590


namespace no_nondegenerate_triangle_l165_165310

def distinct_positive_integers (a b c : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def nondegenerate_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem no_nondegenerate_triangle (a b c : ℕ)
  (h_distinct : distinct_positive_integers a b c)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1)
  (h1 : a ∣ (b - c) ^ 2)
  (h2 : b ∣ (c - a) ^ 2)
  (h3 : c ∣ (a - b) ^ 2) :
  ¬nondegenerate_triangle a b c :=
sorry

end no_nondegenerate_triangle_l165_165310


namespace prove_travel_cost_l165_165806

noncomputable def least_expensive_travel_cost
  (a_cost_per_km : ℝ) (a_booking_fee : ℝ) (b_cost_per_km : ℝ)
  (DE DF EF : ℝ) :
  ℝ := by
  let a_cost_DE := DE * a_cost_per_km + a_booking_fee
  let b_cost_DE := DE * b_cost_per_km
  let cheaper_cost_DE := min a_cost_DE b_cost_DE

  let a_cost_EF := EF * a_cost_per_km + a_booking_fee
  let b_cost_EF := EF * b_cost_per_km
  let cheaper_cost_EF := min a_cost_EF b_cost_EF

  let a_cost_DF := DF * a_cost_per_km + a_booking_fee
  let b_cost_DF := DF * b_cost_per_km
  let cheaper_cost_DF := min a_cost_DF b_cost_DF

  exact cheaper_cost_DE + cheaper_cost_EF + cheaper_cost_DF

def travel_problem : Prop :=
  let DE := 5000
  let DF := 4000
  let EF := 2500 -- derived from the Pythagorean theorem
  least_expensive_travel_cost 0.12 120 0.20 DE DF EF = 1740

theorem prove_travel_cost : travel_problem := sorry

end prove_travel_cost_l165_165806


namespace problem_solution_l165_165309

/-- Let f be an even function on ℝ such that f(x + 2) = f(x) and f(x) = x - 2 for x ∈ [3, 4]. 
    Then f(sin 1) < f(cos 1). -/
theorem problem_solution (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = f x)
  (h2 : ∀ x, f (x + 2) = f x)
  (h3 : ∀ x, 3 ≤ x ∧ x ≤ 4 → f x = x - 2) :
  f (Real.sin 1) < f (Real.cos 1) :=
sorry

end problem_solution_l165_165309


namespace one_fourth_of_eight_times_x_plus_two_l165_165569

theorem one_fourth_of_eight_times_x_plus_two (x : ℝ) : 
  (1 / 4) * (8 * x + 2) = 2 * x + 1 / 2 :=
by
  sorry

end one_fourth_of_eight_times_x_plus_two_l165_165569


namespace monthly_growth_rate_selling_price_april_l165_165843

-- First problem: Proving the monthly average growth rate
theorem monthly_growth_rate (sales_jan sales_mar : ℝ) (x : ℝ) 
    (h1 : sales_jan = 256)
    (h2 : sales_mar = 400)
    (h3 : sales_mar = sales_jan * (1 + x)^2) :
  x = 0.25 := 
sorry

-- Second problem: Proving the selling price in April
theorem selling_price_april (unit_profit desired_profit current_sales sales_increase_per_yuan_change current_price new_price : ℝ)
    (h1 : unit_profit = new_price - 25)
    (h2 : desired_profit = 4200)
    (h3 : current_sales = 400)
    (h4 : sales_increase_per_yuan_change = 4)
    (h5 : current_price = 40)
    (h6 : desired_profit = unit_profit * (current_sales + sales_increase_per_yuan_change * (current_price - new_price))) :
  new_price = 35 := 
sorry

end monthly_growth_rate_selling_price_april_l165_165843


namespace calculate_total_cups_l165_165210

variable (butter : ℕ) (flour : ℕ) (sugar : ℕ) (total_cups : ℕ)

def ratio_condition : Prop :=
  3 * butter = 2 * sugar ∧ 3 * flour = 5 * sugar

def sugar_condition : Prop :=
  sugar = 9

def total_cups_calculation : Prop :=
  total_cups = butter + flour + sugar

theorem calculate_total_cups (h1 : ratio_condition butter flour sugar) (h2 : sugar_condition sugar) :
  total_cups_calculation butter flour sugar total_cups -> total_cups = 30 := by
  sorry

end calculate_total_cups_l165_165210


namespace total_number_of_cards_l165_165182

theorem total_number_of_cards (groups : ℕ) (cards_per_group : ℕ) (h_groups : groups = 9) (h_cards_per_group : cards_per_group = 8) : groups * cards_per_group = 72 := by
  sorry

end total_number_of_cards_l165_165182


namespace b3_b8_product_l165_165030

-- Definitions based on conditions
def is_arithmetic_seq (b : ℕ → ℤ) := ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

-- The problem statement
theorem b3_b8_product (b : ℕ → ℤ) (h_seq : is_arithmetic_seq b) (h4_7 : b 4 * b 7 = 24) : 
  b 3 * b 8 = 200 / 9 :=
sorry

end b3_b8_product_l165_165030


namespace positive_root_of_quadratic_eqn_l165_165513

theorem positive_root_of_quadratic_eqn 
  (b : ℝ)
  (h1 : ∃ x0 : ℝ, x0^2 - 4 * x0 + b = 0 ∧ (-x0)^2 + 4 * (-x0) - b = 0) 
  : ∃ x : ℝ, (x^2 + b * x - 4 = 0) ∧ x = 2 := 
by
  sorry

end positive_root_of_quadratic_eqn_l165_165513


namespace contrapositive_proposition_l165_165041

theorem contrapositive_proposition (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
sorry

end contrapositive_proposition_l165_165041


namespace comic_books_stacking_order_l165_165074

-- Definitions of the conditions
def num_spiderman_books : ℕ := 6
def num_archie_books : ℕ := 5
def num_garfield_books : ℕ := 4

-- Calculations of factorials
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Grouping and order calculation
def ways_to_arrange_group_books : ℕ :=
  factorial num_spiderman_books *
  factorial num_archie_books *
  factorial num_garfield_books

def num_groups : ℕ := 3

def ways_to_arrange_groups : ℕ :=
  factorial num_groups

def total_ways_to_stack_books : ℕ :=
  ways_to_arrange_group_books * ways_to_arrange_groups

-- Theorem stating the total number of different orders
theorem comic_books_stacking_order :
  total_ways_to_stack_books = 12441600 :=
by
  sorry

end comic_books_stacking_order_l165_165074


namespace second_percentage_increase_l165_165888

theorem second_percentage_increase 
  (P : ℝ) 
  (x : ℝ) 
  (h1: 1.20 * P * (1 + x / 100) = 1.38 * P) : 
  x = 15 := 
  sorry

end second_percentage_increase_l165_165888


namespace correct_statements_l165_165237

-- Definitions
def p_A : ℚ := 1 / 2
def p_B : ℚ := 1 / 3

-- Statements to be verified
def statement1 := (p_A * (1 - p_B) + (1 - p_A) * p_B) = (1 / 2 + 1 / 3)
def statement2 := (p_A * p_B) = (1 / 2 * 1 / 3)
def statement3 := (p_A * (1 - p_B) + p_A * p_B) = (1 / 2 * 2 / 3 + 1 / 2 * 1 / 3)
def statement4 := (1 - (1 - p_A) * (1 - p_B)) = (1 - 1 / 2 * 2 / 3)

-- Theorem stating the correct sequence of statements
theorem correct_statements : (statement2 ∧ statement4) ∧ ¬(statement1 ∨ statement3) :=
by
  sorry

end correct_statements_l165_165237


namespace min_re_z4_re_z4_l165_165535

theorem min_re_z4_re_z4 (z : ℂ) (h : z.re ≠ 0) : 
  ∃ t : ℝ, (t = (z.im / z.re)) ∧ ((1 - 6 * (t^2) + (t^4)) = -8) := sorry

end min_re_z4_re_z4_l165_165535


namespace pos_difference_between_highest_and_second_smallest_enrollment_l165_165381

def varsity_enrollment : ℕ := 1520
def northwest_enrollment : ℕ := 1430
def central_enrollment : ℕ := 1900
def greenbriar_enrollment : ℕ := 1850

theorem pos_difference_between_highest_and_second_smallest_enrollment :
  (central_enrollment - varsity_enrollment) = 380 := 
by 
  sorry

end pos_difference_between_highest_and_second_smallest_enrollment_l165_165381


namespace simplify_expression_l165_165603

theorem simplify_expression :
  ((2 + 3 + 4 + 5) / 2) + ((2 * 5 + 8) / 3) = 13 :=
by
  sorry

end simplify_expression_l165_165603


namespace students_in_class_l165_165303

theorem students_in_class
  (B : ℕ) (E : ℕ) (G : ℕ)
  (h1 : B = 12)
  (h2 : G + B = 22)
  (h3 : E = 10) :
  G + E + B = 32 :=
by
  sorry

end students_in_class_l165_165303


namespace find_A_l165_165363

theorem find_A :
  ∃ A B C D : ℕ, A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
               A * B = 72 ∧ C * D = 72 ∧
               A + B = C - D ∧ A = 4 :=
by
  sorry

end find_A_l165_165363


namespace arithmetic_sequence_l165_165131

theorem arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) (h1 : a 2 = 3) (h2 : a (n - 1) = 17) (h3 : n ≥ 2) (h4 : (n * (3 + 17)) / 2 = 100) : n = 10 :=
sorry

end arithmetic_sequence_l165_165131


namespace smaller_side_of_new_rectangle_is_10_l165_165592

/-- We have a 10x25 rectangle that is divided into two congruent polygons and rearranged 
to form another rectangle. We need to prove that the length of the smaller side of the 
resulting rectangle is 10. -/
theorem smaller_side_of_new_rectangle_is_10 :
  ∃ (y x : ℕ), (y * x = 10 * 25) ∧ (y ≤ x) ∧ y = 10 := 
sorry

end smaller_side_of_new_rectangle_is_10_l165_165592


namespace inequality_solution_l165_165405

theorem inequality_solution (x : ℝ) : (3 < x ∧ x < 5) → (x - 5) / ((x - 3)^2) < 0 := 
by 
  intro h
  sorry

end inequality_solution_l165_165405


namespace second_smallest_packs_of_hot_dogs_l165_165224

theorem second_smallest_packs_of_hot_dogs (n m : ℕ) (k : ℕ) :
  (12 * n ≡ 5 [MOD 10]) ∧ (10 * m ≡ 3 [MOD 12]) → n = 15 :=
by
  sorry

end second_smallest_packs_of_hot_dogs_l165_165224


namespace rope_segment_length_l165_165368

theorem rope_segment_length (L : ℕ) (half_fold_times : ℕ) (dm_to_cm : ℕ → ℕ) 
  (hL : L = 8) (h_half_fold_times : half_fold_times = 2) (h_dm_to_cm : dm_to_cm 1 = 10)
  : dm_to_cm (L / 2 ^ half_fold_times) = 20 := 
by 
  sorry

end rope_segment_length_l165_165368


namespace compare_log_exp_l165_165050

theorem compare_log_exp (x y z : ℝ) 
  (hx : x = Real.log 2 / Real.log 5) 
  (hy : y = Real.log 2) 
  (hz : z = Real.sqrt 2) : 
  x < y ∧ y < z := 
sorry

end compare_log_exp_l165_165050


namespace express_2011_with_digit_1_l165_165095

theorem express_2011_with_digit_1 :
  ∃ (a b c d e: ℕ), 2011 = a * b - c * d + e - f + g ∧
  (a = 1111 ∧ b = 1111) ∧ (c = 111 ∧ d = 11111) ∧ (e = 1111) ∧ (f = 111) ∧ (g = 11) ∧
  (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ g) :=
sorry

end express_2011_with_digit_1_l165_165095


namespace original_price_is_975_l165_165711

variable (x : ℝ)
variable (discounted_price : ℝ := 780)
variable (discount : ℝ := 0.20)

-- The condition that Smith bought the shirt for Rs. 780 after a 20% discount
def original_price_calculation (x : ℝ) (discounted_price : ℝ) (discount : ℝ) : Prop :=
  (1 - discount) * x = discounted_price

theorem original_price_is_975 : ∃ x : ℝ, original_price_calculation x 780 0.20 ∧ x = 975 := 
by
  -- Proof will be provided here
  sorry

end original_price_is_975_l165_165711


namespace total_pots_needed_l165_165985

theorem total_pots_needed
    (p : ℕ) (s : ℕ) (h : ℕ)
    (hp : p = 5)
    (hs : s = 3)
    (hh : h = 4) :
    p * s * h = 60 := by
  sorry

end total_pots_needed_l165_165985


namespace num_ordered_pairs_l165_165216

theorem num_ordered_pairs (N : ℕ) :
  (N = 20) ↔ ∃ (a b : ℕ), 
  (a < b) ∧ (100 ≤ a ∧ a ≤ 1000)
  ∧ (100 ≤ b ∧ b ≤ 1000)
  ∧ (gcd a b * lcm a b = 495 * gcd a b)
  := 
sorry

end num_ordered_pairs_l165_165216


namespace julie_reads_tomorrow_l165_165858

theorem julie_reads_tomorrow :
  let total_pages := 120
  let pages_read_yesterday := 12
  let pages_read_today := 2 * pages_read_yesterday
  let pages_read_so_far := pages_read_yesterday + pages_read_today
  let remaining_pages := total_pages - pages_read_so_far
  remaining_pages / 2 = 42 :=
by
  sorry

end julie_reads_tomorrow_l165_165858


namespace rental_cost_equal_mileage_l165_165496

theorem rental_cost_equal_mileage:
  ∃ x : ℝ, (17.99 + 0.18 * x = 18.95 + 0.16 * x) ∧ x = 48 := 
by
  sorry

end rental_cost_equal_mileage_l165_165496


namespace john_speed_first_part_l165_165046

theorem john_speed_first_part (S : ℝ) (h1 : 2 * S + 3 * 55 = 255) : S = 45 :=
by
  sorry

end john_speed_first_part_l165_165046


namespace pool_students_count_l165_165589

noncomputable def total_students (total_women : ℕ) (female_students : ℕ) (extra_men : ℕ) (non_student_men : ℕ) : ℕ := 
  let total_men := total_women + extra_men
  let male_students := total_men - non_student_men
  female_students + male_students

theorem pool_students_count
  (total_women : ℕ := 1518)
  (female_students : ℕ := 536)
  (extra_men : ℕ := 525)
  (non_student_men : ℕ := 1257) :
  total_students total_women female_students extra_men non_student_men = 1322 := 
by
  sorry

end pool_students_count_l165_165589


namespace sector_area_l165_165666

theorem sector_area (α : ℝ) (r : ℝ) (hα : α = π / 3) (hr : r = 6) : 
  1/2 * r^2 * α = 6 * π :=
by {
  sorry
}

end sector_area_l165_165666


namespace find_x_l165_165785

variables (x : ℝ)
axiom h1 : (180 / x) + (5 * 12 / x) + 80 = 81

theorem find_x : x = 240 :=
by {
  sorry
}

end find_x_l165_165785


namespace integer_modulo_problem_l165_165092

theorem integer_modulo_problem : ∃ n : ℤ, 0 ≤ n ∧ n < 23 ∧ (-250 % 23 = n) := 
  sorry

end integer_modulo_problem_l165_165092


namespace painters_workdays_l165_165085

theorem painters_workdays (d₁ d₂ : ℚ) (p₁ p₂ : ℕ)
  (h1 : p₁ = 5) (h2 : p₂ = 4) (rate: 5 * d₁ = 7.5) :
  (p₂:ℚ) * d₂ = 7.5 → d₂ = 1 + 7 / 8 :=
by
  sorry

end painters_workdays_l165_165085


namespace division_of_fractions_l165_165748

theorem division_of_fractions : (4 : ℚ) / (5 / 7) = 28 / 5 := sorry

end division_of_fractions_l165_165748


namespace at_least_one_woman_selected_l165_165071

noncomputable def probability_at_least_one_woman_selected (men women : ℕ) (total_selected : ℕ) : ℚ :=
  let total_people := men + women
  let prob_no_woman := (men / total_people) * ((men - 1) / (total_people - 1)) * ((men - 2) / (total_people - 2))
  1 - prob_no_woman

theorem at_least_one_woman_selected (men women : ℕ) (total_selected : ℕ) :
  men = 5 → women = 5 → total_selected = 3 → 
  probability_at_least_one_woman_selected men women total_selected = 11 / 12 := by
  intros hmen hwomen hselected
  rw [hmen, hwomen, hselected]
  unfold probability_at_least_one_woman_selected
  sorry

end at_least_one_woman_selected_l165_165071


namespace interest_rate_second_share_l165_165966

variable (T : ℝ) (r1 : ℝ) (I2 : ℝ) (T_i : ℝ)

theorem interest_rate_second_share 
  (h1 : T = 100000)
  (h2 : r1 = 0.09)
  (h3 : I2 = 24999.999999999996)
  (h4 : T_i = 0.095 * T) : 
  (2750 / I2) * 100 = 11 :=
by {
  sorry
}

end interest_rate_second_share_l165_165966


namespace angle_terminal_side_l165_165062

def angle_on_line (β : ℝ) : Prop :=
  ∃ n : ℤ, β = 135 + n * 180

def angle_in_range (β : ℝ) : Prop :=
  -360 < β ∧ β < 360

theorem angle_terminal_side :
  ∀ β, angle_on_line β → angle_in_range β → β = -225 ∨ β = -45 ∨ β = 135 ∨ β = 315 :=
by
  intros β h_line h_range
  sorry

end angle_terminal_side_l165_165062


namespace solve_inequality_l165_165984

theorem solve_inequality :
  {x : ℝ | x^2 - 9 * x + 14 < 0} = {x : ℝ | 2 < x ∧ x < 7} := sorry

end solve_inequality_l165_165984


namespace lcm_25_35_50_l165_165543

theorem lcm_25_35_50 : Nat.lcm (Nat.lcm 25 35) 50 = 350 := by
  sorry

end lcm_25_35_50_l165_165543


namespace number_of_buildings_l165_165044

theorem number_of_buildings (studio_apartments : ℕ) (two_person_apartments : ℕ) (four_person_apartments : ℕ)
    (occupancy_percentage : ℝ) (current_occupancy : ℕ)
    (max_occupancy_building : ℕ) (max_occupancy_complex : ℕ) (num_buildings : ℕ)
    (h_studio : studio_apartments = 10)
    (h_two_person : two_person_apartments = 20)
    (h_four_person : four_person_apartments = 5)
    (h_occupancy_percentage : occupancy_percentage = 0.75)
    (h_current_occupancy : current_occupancy = 210)
    (h_max_occupancy_building : max_occupancy_building = 10 * 1 + 20 * 2 + 5 * 4)
    (h_max_occupancy_complex : max_occupancy_complex = current_occupancy / occupancy_percentage)
    (h_num_buildings : num_buildings = max_occupancy_complex / max_occupancy_building) :
    num_buildings = 4 :=
by
  sorry

end number_of_buildings_l165_165044


namespace trigonometric_identity_l165_165571

noncomputable def trigonometric_identity_proof : Prop :=
  let cos_30 := Real.sqrt 3 / 2;
  let sin_60 := Real.sqrt 3 / 2;
  let sin_30 := 1 / 2;
  let cos_60 := 1 / 2;
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1

theorem trigonometric_identity : trigonometric_identity_proof :=
  sorry

end trigonometric_identity_l165_165571


namespace building_height_l165_165842

theorem building_height (H : ℝ) 
                        (bounced_height : ℕ → ℝ) 
                        (h_bounce : ∀ n, bounced_height n = H / 2 ^ (n + 1)) 
                        (h_fifth : bounced_height 5 = 3) : 
    H = 96 := 
by {
  sorry
}

end building_height_l165_165842


namespace suitable_survey_l165_165861

inductive Survey
| FavoriteTVPrograms : Survey
| PrintingErrors : Survey
| BatteryServiceLife : Survey
| InternetUsage : Survey

def is_suitable_for_census (s : Survey) : Prop :=
  match s with
  | Survey.PrintingErrors => True
  | _ => False

theorem suitable_survey : is_suitable_for_census Survey.PrintingErrors = True :=
by
  sorry

end suitable_survey_l165_165861


namespace value_of_other_bills_is_40_l165_165506

-- Define the conditions using Lean definitions
def class_fund_contains_only_10_and_other_bills (total_amount : ℕ) (num_other_bills num_10_bills : ℕ) : Prop :=
  total_amount = 120 ∧ num_other_bills = 3 ∧ num_10_bills = 2 * num_other_bills

def value_of_each_other_bill (total_amount num_other_bills : ℕ) : ℕ :=
  total_amount / num_other_bills

-- The theorem we want to prove
theorem value_of_other_bills_is_40 (total_amount num_other_bills : ℕ) 
  (h : class_fund_contains_only_10_and_other_bills total_amount num_other_bills (2 * num_other_bills)) :
  value_of_each_other_bill total_amount num_other_bills = 40 := 
by 
  -- We use the conditions here to ensure they are part of the proof even if we skip the actual proof with sorry
  have h1 : total_amount = 120 := by sorry
  have h2 : num_other_bills = 3 := by sorry
  -- Skipping the proof
  sorry

end value_of_other_bills_is_40_l165_165506


namespace volume_of_rectangular_prism_l165_165652

theorem volume_of_rectangular_prism (a b c : ℝ)
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : a * c = Real.sqrt 6) :
  a * b * c = Real.sqrt 6 := by
sorry

end volume_of_rectangular_prism_l165_165652


namespace find_least_positive_x_l165_165198

theorem find_least_positive_x :
  ∃ x : ℕ, 0 < x ∧ (x + 5713) % 15 = 1847 % 15 ∧ x = 4 :=
by
  sorry

end find_least_positive_x_l165_165198


namespace pentagon_interior_angles_l165_165007

theorem pentagon_interior_angles
  (x y : ℝ)
  (H_eq_triangle : ∀ (angle : ℝ), angle = 60)
  (H_rect_QT : ∀ (angle : ℝ), angle = 90)
  (sum_interior_angles_pentagon : ∀ (n : ℕ), (n - 2) * 180 = 3 * 180) :
  x + y = 60 :=
by
  sorry

end pentagon_interior_angles_l165_165007


namespace simplify_fraction_l165_165992

theorem simplify_fraction : (4^4 + 4^2) / (4^3 - 4) = 17 / 3 := by
  sorry

end simplify_fraction_l165_165992


namespace right_angled_triangle_l165_165752

-- Define the lengths of the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- State the theorem using the Pythagorean theorem
theorem right_angled_triangle : a^2 + b^2 = c^2 :=
by
  sorry

end right_angled_triangle_l165_165752


namespace steve_ate_bags_l165_165354

-- Given conditions
def total_macaroons : Nat := 12
def weight_per_macaroon : Nat := 5
def num_bags : Nat := 4
def total_weight_remaining : Nat := 45

-- Derived conditions
def total_weight_macaroons : Nat := total_macaroons * weight_per_macaroon
def macaroons_per_bag : Nat := total_macaroons / num_bags
def weight_per_bag : Nat := macaroons_per_bag * weight_per_macaroon
def bags_remaining : Nat := total_weight_remaining / weight_per_bag

-- Proof statement
theorem steve_ate_bags : num_bags - bags_remaining = 1 := by
  sorry

end steve_ate_bags_l165_165354


namespace product_mod_25_l165_165642

def remainder_when_divided_by_25 (n : ℕ) : ℕ := n % 25

theorem product_mod_25 (a b c d : ℕ) 
  (h1 : a = 1523) (h2 : b = 1857) (h3 : c = 1919) (h4 : d = 2012) :
  remainder_when_divided_by_25 (a * b * c * d) = 8 :=
by
  sorry

end product_mod_25_l165_165642


namespace sum_of_integers_mod_59_l165_165647

theorem sum_of_integers_mod_59 (a b c : ℕ) (h1 : a % 59 = 29) (h2 : b % 59 = 31) (h3 : c % 59 = 7)
  (h4 : a^2 % 59 = 29) (h5 : b^2 % 59 = 31) (h6 : c^2 % 59 = 7) :
  (a + b + c) % 59 = 8 :=
by
  sorry

end sum_of_integers_mod_59_l165_165647


namespace sum_is_seventeen_l165_165566

variable (x y : ℕ)

def conditions (x y : ℕ) : Prop :=
  x > y ∧ x - y = 3 ∧ x * y = 56

theorem sum_is_seventeen (x y : ℕ) (h: conditions x y) : x + y = 17 :=
by
  sorry

end sum_is_seventeen_l165_165566


namespace gcd_cube_sum_condition_l165_165814

theorem gcd_cube_sum_condition (n : ℕ) (hn : n > 32) : Nat.gcd (n^3 + 125) (n + 5) = 1 := 
  by 
  sorry

end gcd_cube_sum_condition_l165_165814


namespace aitana_jayda_total_spending_l165_165249

theorem aitana_jayda_total_spending (jayda_spent : ℤ) (more_fraction : ℚ) (jayda_spent_400 : jayda_spent = 400) (more_fraction_2_5 : more_fraction = 2 / 5) :
  jayda_spent + (jayda_spent + (more_fraction * jayda_spent)) = 960 :=
by
  sorry

end aitana_jayda_total_spending_l165_165249


namespace calcium_iodide_weight_l165_165738

theorem calcium_iodide_weight
  (atomic_weight_Ca : ℝ)
  (atomic_weight_I : ℝ)
  (moles : ℝ) :
  atomic_weight_Ca = 40.08 →
  atomic_weight_I = 126.90 →
  moles = 5 →
  (atomic_weight_Ca + 2 * atomic_weight_I) * moles = 1469.4 :=
by
  intros
  sorry

end calcium_iodide_weight_l165_165738


namespace unique_three_digit_numbers_l165_165446

noncomputable def three_digit_numbers_no_repeats : Nat :=
  let total_digits := 10
  let permutations := total_digits * (total_digits - 1) * (total_digits - 2)
  let invalid_start_with_zero := (total_digits - 1) * (total_digits - 2)
  permutations - invalid_start_with_zero

theorem unique_three_digit_numbers : three_digit_numbers_no_repeats = 648 := by
  sorry

end unique_three_digit_numbers_l165_165446


namespace max_value_function_l165_165731

theorem max_value_function (x : ℝ) (h : x < 0) : 
  ∃ y_max, (∀ x', x' < 0 → (x' + 4 / x') ≤ y_max) ∧ y_max = -4 := 
sorry

end max_value_function_l165_165731


namespace distance_between_foci_of_hyperbola_l165_165990

theorem distance_between_foci_of_hyperbola (a b c : ℝ) : (x^2 - y^2 = 4) → (a = 2) → (b = 0) → (c = Real.sqrt (4 + 0)) → 
    dist (2, 0) (-2, 0) = 4 :=
by
  sorry

end distance_between_foci_of_hyperbola_l165_165990


namespace joan_number_of_games_l165_165864

open Nat

theorem joan_number_of_games (a b c d e : ℕ) (h_a : a = 10) (h_b : b = 12) (h_c : c = 6) (h_d : d = 9) (h_e : e = 4) :
  a + b + c + d + e = 41 :=
by
  sorry

end joan_number_of_games_l165_165864


namespace jacket_cost_l165_165458

noncomputable def cost_of_shorts : ℝ := 13.99
noncomputable def cost_of_shirt : ℝ := 12.14
noncomputable def total_spent : ℝ := 33.56
noncomputable def cost_of_jacket : ℝ := total_spent - (cost_of_shorts + cost_of_shirt)

theorem jacket_cost : cost_of_jacket = 7.43 := by
  sorry

end jacket_cost_l165_165458


namespace least_number_to_add_l165_165921

theorem least_number_to_add (n : ℕ) (h₁ : n = 1054) :
  ∃ k : ℕ, (n + k) % 23 = 0 ∧ k = 4 :=
by
  use 4
  have h₂ : n % 23 = 19 := by sorry
  have h₃ : (n + 4) % 23 = 0 := by sorry
  exact ⟨h₃, rfl⟩

end least_number_to_add_l165_165921


namespace max_students_distributing_pens_and_pencils_l165_165923

theorem max_students_distributing_pens_and_pencils :
  Nat.gcd 1001 910 = 91 :=
by
  -- remaining proof required
  sorry

end max_students_distributing_pens_and_pencils_l165_165923


namespace tutors_next_together_in_360_days_l165_165936

open Nat

-- Define the intervals for each tutor
def evan_interval := 5
def fiona_interval := 6
def george_interval := 9
def hannah_interval := 8
def ian_interval := 10

-- Statement to prove
theorem tutors_next_together_in_360_days :
  Nat.lcm (Nat.lcm evan_interval fiona_interval) (Nat.lcm george_interval (Nat.lcm hannah_interval ian_interval)) = 360 :=
by
  sorry

end tutors_next_together_in_360_days_l165_165936


namespace john_unanswered_questions_l165_165576

theorem john_unanswered_questions (c w u : ℕ) 
  (h1 : 25 + 5 * c - 2 * w = 95) 
  (h2 : 6 * c - w + 3 * u = 105) 
  (h3 : c + w + u = 30) : 
  u = 2 := 
sorry

end john_unanswered_questions_l165_165576


namespace scientific_notation_to_standard_form_l165_165490

theorem scientific_notation_to_standard_form :
  - 3.96 * 10^5 = -396000 :=
sorry

end scientific_notation_to_standard_form_l165_165490


namespace count_divisible_by_25_l165_165550

-- Define the conditions
def is_positive_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def ends_in_25 (n : ℕ) : Prop := n % 100 = 25

-- Define the main statement to prove
theorem count_divisible_by_25 : 
  (∃ (count : ℕ), count = 90 ∧
  ∀ n, is_positive_four_digit n ∧ ends_in_25 n → count = 90) :=
by {
  -- Outline the proof
  sorry
}

end count_divisible_by_25_l165_165550


namespace rectangle_area_problem_l165_165520

/--
Given a rectangle with dimensions \(3x - 4\) and \(4x + 6\),
show that the area of the rectangle equals \(12x^2 + 2x - 24\) if and only if \(x \in \left(\frac{4}{3}, \infty\right)\).
-/
theorem rectangle_area_problem 
  (x : ℝ) 
  (h1 : 3 * x - 4 > 0)
  (h2 : 4 * x + 6 > 0) :
  (3 * x - 4) * (4 * x + 6) = 12 * x^2 + 2 * x - 24 ↔ x > 4 / 3 :=
sorry

end rectangle_area_problem_l165_165520


namespace perimeter_of_triangle_l165_165850

theorem perimeter_of_triangle (x y : ℝ) (h : 0 < x) (h1 : 0 < y) (h2 : x < y) :
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  (2 * leg_length + hypotenuse = (y - x) * (1 + 1 / Real.sqrt 2)) :=
by
  let leg_length := (y - x) / 2
  let hypotenuse := (y - x) / (Real.sqrt 2)
  sorry

end perimeter_of_triangle_l165_165850


namespace probability_of_one_or_two_l165_165646

/-- Represents the number of elements in the first 20 rows of Pascal's Triangle. -/
noncomputable def total_elements : ℕ := 210

/-- Represents the number of ones in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_ones : ℕ := 39

/-- Represents the number of twos in the first 20 rows of Pascal's Triangle. -/
noncomputable def number_of_twos : ℕ :=18

/-- Prove that the probability of randomly choosing an element which is either 1 or 2
from the first 20 rows of Pascal's Triangle is 57/210. -/
theorem probability_of_one_or_two (h1 : total_elements = 210)
                                  (h2 : number_of_ones = 39)
                                  (h3 : number_of_twos = 18) :
    39 + 18 = 57 ∧ (57 : ℚ) / 210 = 57 / 210 :=
by {
    sorry
}

end probability_of_one_or_two_l165_165646


namespace evaluate_expression_l165_165239

theorem evaluate_expression : (Real.sqrt ((Real.sqrt 2)^4))^6 = 64 := by
  sorry

end evaluate_expression_l165_165239


namespace gcd_2814_1806_l165_165175

def a := 2814
def b := 1806

theorem gcd_2814_1806 : Nat.gcd a b = 42 :=
by
  sorry

end gcd_2814_1806_l165_165175


namespace line_equation_through_origin_and_circle_chord_length_l165_165180

theorem line_equation_through_origin_and_circle_chord_length 
  (x y : ℝ) 
  (h : x^2 + y^2 - 2 * x - 4 * y + 4 = 0) 
  (chord_length : ℝ) 
  (h_chord : chord_length = 2) 
  : 2 * x - y = 0 := 
sorry

end line_equation_through_origin_and_circle_chord_length_l165_165180


namespace solve_for_x_l165_165161

theorem solve_for_x (x : ℚ) (h : (7 * x) / (x - 2) + 4 / (x - 2) = 6 / (x - 2)) : x = 2 / 7 :=
sorry

end solve_for_x_l165_165161


namespace base_video_card_cost_l165_165746

theorem base_video_card_cost
    (cost_computer : ℕ)
    (fraction_monitor_peripherals : ℕ → ℕ → ℕ)
    (twice : ℕ → ℕ)
    (total_spent : ℕ)
    (cost_monitor_peripherals_eq : fraction_monitor_peripherals cost_computer 5 = 300)
    (twice_eq : ∀ x, twice x = 2 * x)
    (eq_total : ∀ (base_video_card : ℕ), cost_computer + fraction_monitor_peripherals cost_computer 5 + twice base_video_card = total_spent)
    : ∃ x, total_spent = 2100 ∧ cost_computer = 1500 ∧ x = 150 :=
by
  sorry

end base_video_card_cost_l165_165746


namespace distance_between_cities_l165_165187

theorem distance_between_cities 
  (t : ℝ)
  (h1 : 60 * t = 70 * (t - 1 / 4)) 
  (d : ℝ) : 
  d = 105 := by
sorry

end distance_between_cities_l165_165187


namespace sacks_per_day_l165_165352

theorem sacks_per_day (total_sacks : ℕ) (days : ℕ) (h1 : total_sacks = 56) (h2 : days = 4) : total_sacks / days = 14 := by
  sorry

end sacks_per_day_l165_165352


namespace rope_in_two_months_period_l165_165663

theorem rope_in_two_months_period :
  let week1 := 6
  let week2 := 3 * week1
  let week3 := week2 - 4
  let week4 := - (week2 / 2)
  let week5 := week1 + 2
  let week6 := - (2 / 2)
  let week7 := 3 * (2 / 2)
  let week8 := - 10
  let total_length := (week1 + week2 + week3 + week4 + week5 + week6 + week7 + week8)
  total_length * 12 = 348
:= sorry

end rope_in_two_months_period_l165_165663


namespace smallest_positive_integer_a_l165_165186

theorem smallest_positive_integer_a (a : ℕ) (hpos : a > 0) :
  (∃ k, 5880 * a = k ^ 2) → a = 15 := 
by
  sorry

end smallest_positive_integer_a_l165_165186


namespace problem_solution_l165_165769

theorem problem_solution (x : ℝ) (h : 3 * x + 2 = 11) : 6 * x + 3 = 21 :=
by 
  sorry

end problem_solution_l165_165769


namespace total_telephone_bill_second_month_l165_165152

theorem total_telephone_bill_second_month
  (F C1 : ℝ) 
  (h1 : F + C1 = 46)
  (h2 : F + 2 * C1 = 76) :
  F + 2 * C1 = 76 :=
by
  sorry

end total_telephone_bill_second_month_l165_165152


namespace cone_base_diameter_l165_165253

theorem cone_base_diameter (l r : ℝ) 
  (h1 : (1/2) * π * l^2 + π * r^2 = 3 * π) 
  (h2 : π * l = 2 * π * r) : 2 * r = 2 :=
by
  sorry

end cone_base_diameter_l165_165253


namespace geometric_sequence_sum_l165_165766

theorem geometric_sequence_sum (q : ℝ) (h_pos : q > 0) (h_ratio_ne_one : q ≠ 1)
  (S : ℕ → ℝ) (h_a1 : S 1 = 1) (h_S4_eq_5S2 : S 4 - 5 * S 2 = 0) :
  S 5 = 31 :=
sorry

end geometric_sequence_sum_l165_165766


namespace last_two_digits_condition_l165_165732

-- Define the function to get last two digits of a number
def last_two_digits (n : ℕ) : ℕ :=
  n % 100

-- Given numbers
def n1 := 122
def n2 := 123
def n3 := 125
def n4 := 129

-- The missing number
variable (x : ℕ)

theorem last_two_digits_condition : 
  last_two_digits (last_two_digits n1 * last_two_digits n2 * last_two_digits n3 * last_two_digits n4 * last_two_digits x) = 50 ↔ last_two_digits x = 1 :=
by 
  sorry

end last_two_digits_condition_l165_165732


namespace part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l165_165797

-- Part 1: Is x^2 - 3x + 2 = 0 a "double root equation"?
theorem part1_double_root_equation :
    ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂ ∧ x₁ * 2 = x₂) 
              ∧ (x^2 - 3 * x + 2 = 0) :=
sorry

-- Part 2: Given (x - 2)(x - m) = 0 is a "double root equation", find value of m^2 + 2m + 2.
theorem part2_value_m_squared_2m_2 (m : ℝ) :
    ∃ (v : ℝ), v = m^2 + 2 * m + 2 ∧ 
          (m = 1 ∨ m = 4) ∧
          (v = 5 ∨ v = 26) :=
sorry

-- Part 3: Determine m such that x^2 - (m-1)x + 32 = 0 is a "double root equation".
theorem part3_value_m (m : ℝ) :
    x^2 - (m - 1) * x + 32 = 0 ∧ 
    (m = 13 ∨ m = -11) :=
sorry

end part1_double_root_equation_part2_value_m_squared_2m_2_part3_value_m_l165_165797


namespace irreducible_fraction_l165_165511

-- Statement of the theorem
theorem irreducible_fraction (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry -- Proof would be placed here

end irreducible_fraction_l165_165511


namespace total_number_of_animals_l165_165704

-- Define the problem conditions
def number_of_cats : ℕ := 645
def number_of_dogs : ℕ := 567

-- State the theorem to be proved
theorem total_number_of_animals : number_of_cats + number_of_dogs = 1212 := by
  sorry

end total_number_of_animals_l165_165704


namespace leak_empty_time_l165_165836

theorem leak_empty_time (A L : ℝ) (h1 : A = 1 / 8) (h2 : A - L = 1 / 12) : 1 / L = 24 :=
by
  -- The proof will be provided here
  sorry

end leak_empty_time_l165_165836


namespace Jose_Raju_Work_Together_l165_165664

-- Definitions for the conditions
def JoseWorkRate : ℚ := 1 / 10
def RajuWorkRate : ℚ := 1 / 40
def CombinedWorkRate : ℚ := JoseWorkRate + RajuWorkRate

-- Theorem statement
theorem Jose_Raju_Work_Together :
  1 / CombinedWorkRate = 8 := by
    sorry

end Jose_Raju_Work_Together_l165_165664


namespace difference_is_cube_sum_1996_impossible_l165_165488

theorem difference_is_cube (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  M - m = (n - 1)^3 := 
by {
  sorry
}

theorem sum_1996_impossible (n : ℕ) (M m : ℕ) 
  (M_eq : M = (3 * n^3 - 4 * n^2 + 5 * n - 2) / 2)
  (m_eq : m = (n^3 + 2 * n^2 - n) / 2) :
  ¬(1996 ∈ {x | m ≤ x ∧ x ≤ M}) := 
by {
  sorry
}

end difference_is_cube_sum_1996_impossible_l165_165488


namespace probability_of_selecting_specific_letters_l165_165810

theorem probability_of_selecting_specific_letters :
  let total_cards := 15
  let amanda_cards := 6
  let chloe_or_ethan_cards := 9
  let prob_amanda_then_chloe_or_ethan := (amanda_cards / total_cards) * (chloe_or_ethan_cards / (total_cards - 1))
  let prob_chloe_or_ethan_then_amanda := (chloe_or_ethan_cards / total_cards) * (amanda_cards / (total_cards - 1))
  let total_prob := prob_amanda_then_chloe_or_ethan + prob_chloe_or_ethan_then_amanda
  total_prob = 18 / 35 :=
by
  sorry

end probability_of_selecting_specific_letters_l165_165810


namespace b_coordinates_bc_equation_l165_165379

section GeometryProof

-- Define point A
def A : ℝ × ℝ := (1, 1)

-- Altitude CD has the equation: 3x + y - 12 = 0
def altitude_CD (x y : ℝ) : Prop := 3 * x + y - 12 = 0

-- Angle bisector BE has the equation: x - 2y + 4 = 0
def angle_bisector_BE (x y : ℝ) : Prop := x - 2 * y + 4 = 0

-- Coordinates of point B
def B : ℝ × ℝ := (-8, -2)

-- Equation of line BC
def line_BC (x y : ℝ) : Prop := 9 * x - 13 * y + 46 = 0

-- Proof statement for the coordinates of point B
theorem b_coordinates : ∃ x y : ℝ, (x, y) = B :=
by sorry

-- Proof statement for the equation of line BC
theorem bc_equation : ∃ (f : ℝ → ℝ → Prop), f = line_BC :=
by sorry

end GeometryProof

end b_coordinates_bc_equation_l165_165379


namespace probability_f_ge1_l165_165759

noncomputable def f (x: ℝ) : ℝ := 3*x^2 - x - 1

def domain : Set ℝ := { x | -1 ≤ x ∧ x ≤ 2 }

def valid_intervals : Set ℝ := { x | -1 ≤ x ∧ x ≤ -2/3 } ∪ { x | 1 ≤ x ∧ x ≤ 2 }

def interval_length (a b : ℝ) : ℝ := b - a

theorem probability_f_ge1 : 
  (interval_length (-2/3) (-1) + interval_length 1 2) / interval_length (-1) 2 = 4 / 9 := 
by
  sorry

end probability_f_ge1_l165_165759


namespace jane_crayons_l165_165730

theorem jane_crayons :
  let start := 87
  let eaten := 7
  start - eaten = 80 :=
by
  sorry

end jane_crayons_l165_165730


namespace equal_volume_cubes_l165_165183

noncomputable def volume_box : ℝ := 1 -- volume of the cubical box in cubic meters

noncomputable def edge_length_small_cube : ℝ := 0.04 -- edge length of small cubes in meters

noncomputable def number_of_cubes : ℝ := 15624.999999999998 -- number of small cubes

noncomputable def volume_small_cube : ℝ := edge_length_small_cube^3 -- volume of one small cube

theorem equal_volume_cubes : volume_box = volume_small_cube * number_of_cubes :=
  by
  -- Proof goes here
  sorry

end equal_volume_cubes_l165_165183


namespace apple_cost_price_orange_cost_price_banana_cost_price_l165_165954

theorem apple_cost_price (A : ℚ) : 15 = A - (1/6 * A) → A = 18 := by
  intro h
  sorry

theorem orange_cost_price (O : ℚ) : 20 = O + (1/5 * O) → O = 100/6 := by
  intro h
  sorry

theorem banana_cost_price (B : ℚ) : 10 = B → B = 10 := by
  intro h
  sorry

end apple_cost_price_orange_cost_price_banana_cost_price_l165_165954


namespace increasing_decreasing_intervals_l165_165082

noncomputable def f (x : ℝ) : ℝ := Real.sin (-2 * x + 3 * Real.pi / 4)

theorem increasing_decreasing_intervals : (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + 5 * Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 9 * Real.pi / 8) 
      → 0 < f x ∧ f x < 1) 
  ∧ 
    (∀ k : ℤ, 
    ∀ x, 
      ((k : ℝ) * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + 5 * Real.pi / 8) 
      → -1 < f x ∧ f x < 0) :=
by
  sorry

end increasing_decreasing_intervals_l165_165082


namespace max_value_neg7s_squared_plus_56s_plus_20_l165_165036

theorem max_value_neg7s_squared_plus_56s_plus_20 :
  ∃ s : ℝ, s = 4 ∧ ∀ t : ℝ, -7 * t^2 + 56 * t + 20 ≤ 132 := 
by
  sorry

end max_value_neg7s_squared_plus_56s_plus_20_l165_165036


namespace mouse_jump_distance_l165_165551

theorem mouse_jump_distance
  (g f m : ℕ)
  (hg : g = 25)
  (hf : f = g + 32)
  (hm : m = f - 26) :
  m = 31 := by
  sorry

end mouse_jump_distance_l165_165551


namespace sin_sum_leq_3_sqrt3_over_2_l165_165040

theorem sin_sum_leq_3_sqrt3_over_2 
  (A B C : ℝ) 
  (h₁ : A + B + C = Real.pi) 
  (h₂ : 0 < A ∧ A < Real.pi)
  (h₃ : 0 < B ∧ B < Real.pi)
  (h₄ : 0 < C ∧ C < Real.pi) :
  Real.sin A + Real.sin B + Real.sin C ≤ 3 * Real.sqrt 3 / 2 :=
sorry

end sin_sum_leq_3_sqrt3_over_2_l165_165040


namespace rhombus_diagonal_sum_maximum_l165_165471

theorem rhombus_diagonal_sum_maximum 
    (x y : ℝ) 
    (h1 : x^2 + y^2 = 100) 
    (h2 : x ≥ 6) 
    (h3 : y ≤ 6) : 
    x + y = 14 :=
sorry

end rhombus_diagonal_sum_maximum_l165_165471


namespace find_rate_of_interest_l165_165872

theorem find_rate_of_interest (P SI : ℝ) (r : ℝ) (hP : P = 1200) (hSI : SI = 108) (ht : r = r) :
  SI = P * r * r / 100 → r = 3 := by
  intros
  sorry

end find_rate_of_interest_l165_165872


namespace geometric_sequence_a5_l165_165798

-- Definitions from the conditions
def a1 : ℕ := 2
def a9 : ℕ := 8

-- The statement we need to prove
theorem geometric_sequence_a5 (q : ℝ) (h1 : a1 = 2) (h2 : a9 = a1 * q ^ 8) : a1 * q ^ 4 = 4 := by
  have h_q4 : q ^ 4 = 2 := sorry
  -- Proof continues...
  sorry

end geometric_sequence_a5_l165_165798


namespace sufficient_drivers_and_completion_time_l165_165636

noncomputable def one_way_trip_minutes : ℕ := 2 * 60 + 40
noncomputable def round_trip_minutes : ℕ := 2 * one_way_trip_minutes
noncomputable def rest_period_minutes : ℕ := 60
noncomputable def twelve_forty_pm : ℕ := 12 * 60 + 40 -- in minutes from midnight
noncomputable def one_forty_pm : ℕ := twelve_forty_pm + rest_period_minutes
noncomputable def thirteen_five_pm : ℕ := 13 * 60 + 5 -- 1:05 PM
noncomputable def sixteen_ten_pm : ℕ := 16 * 60 + 10 -- 4:10 PM
noncomputable def sixteen_pm : ℕ := 16 * 60 -- 4:00 PM
noncomputable def seventeen_thirty_pm : ℕ := 17 * 60 + 30 -- 5:30 PM
noncomputable def twenty_one_thirty_pm : ℕ := sixteen_ten_pm + round_trip_minutes -- 9:30 PM (21:30)

theorem sufficient_drivers_and_completion_time :
  4 = 4 ∧ twenty_one_thirty_pm = 21 * 60 + 30 := by
  sorry 

end sufficient_drivers_and_completion_time_l165_165636


namespace total_flower_petals_l165_165522

def num_lilies := 8
def petals_per_lily := 6
def num_tulips := 5
def petals_per_tulip := 3

theorem total_flower_petals :
  (num_lilies * petals_per_lily) + (num_tulips * petals_per_tulip) = 63 :=
by
  sorry

end total_flower_petals_l165_165522


namespace min_sum_six_l165_165770

theorem min_sum_six (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h : a * b = a + b + 3) :
  a + b ≥ 6 :=
sorry

end min_sum_six_l165_165770


namespace eval_expression_l165_165943

theorem eval_expression (h : (Real.pi / 2) < 2 ∧ 2 < Real.pi) :
  Real.sqrt (1 - 2 * Real.sin (Real.pi + 2) * Real.cos (Real.pi + 2)) = Real.sin 2 - Real.cos 2 :=
sorry

end eval_expression_l165_165943


namespace initial_bananas_tree_l165_165347

-- Definitions for the conditions
def bananas_left_on_tree : ℕ := 100
def bananas_eaten_by_raj : ℕ := 70
def bananas_in_basket_of_raj := 2 * bananas_eaten_by_raj
def bananas_cut_from_tree := bananas_eaten_by_raj + bananas_in_basket_of_raj
def initial_bananas_on_tree := bananas_cut_from_tree + bananas_left_on_tree

-- The theorem to be proven
theorem initial_bananas_tree : initial_bananas_on_tree = 310 :=
by sorry

end initial_bananas_tree_l165_165347


namespace hybrids_with_full_headlights_l165_165863

theorem hybrids_with_full_headlights (total_cars hybrids_percentage one_headlight_percentage : ℝ) 
  (hc : total_cars = 600) (hp : hybrids_percentage = 0.60) (ho : one_headlight_percentage = 0.40) : 
  total_cars * hybrids_percentage - total_cars * hybrids_percentage * one_headlight_percentage = 216 := by
  sorry

end hybrids_with_full_headlights_l165_165863


namespace balance_blue_balls_l165_165162

variables (G Y W R B : ℕ)

axiom green_balance : 3 * G = 6 * B
axiom yellow_balance : 2 * Y = 5 * B
axiom white_balance : 6 * B = 4 * W
axiom red_balance : 4 * R = 10 * B

theorem balance_blue_balls : 5 * G + 3 * Y + 3 * W + 2 * R = 27 * B :=
  by
  sorry

end balance_blue_balls_l165_165162


namespace prove_A_annual_savings_l165_165672

noncomputable def employee_A_annual_savings
  (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02) : ℝ :=
  let total_deductions := tax_rate + pension_rate + healthcare_rate
  let Income_after_deductions := A_income * (1 - total_deductions)
  let annual_savings := 12 * Income_after_deductions
  annual_savings

theorem prove_A_annual_savings : 
  ∀ (A_income B_income C_income D_income : ℝ)
  (C_income_val : C_income = 14000)
  (income_ratio : A_income / C_income = 5 / 3 ∧ B_income / C_income = 2 / 3 ∧ C_income / D_income = 3 / 4 ∧ B_income = 1.12 * C_income ∧ C_income = 0.85 * D_income)
  (tax_rate pension_rate healthcare_rate : ℝ)
  (tax_rate_val : tax_rate = 0.10)
  (pension_rate_val : pension_rate = 0.05)
  (healthcare_rate_val : healthcare_rate = 0.02),
  employee_A_annual_savings A_income B_income C_income D_income C_income_val income_ratio tax_rate pension_rate healthcare_rate tax_rate_val pension_rate_val healthcare_rate_val = 232400.16 :=
by
  sorry

end prove_A_annual_savings_l165_165672


namespace heartsuit_ratio_l165_165067

def k : ℝ := 3

def heartsuit (n m : ℕ) : ℝ := k * n^3 * m^2

theorem heartsuit_ratio : (heartsuit 3 5) / (heartsuit 5 3) = 3 / 5 := 
by
  sorry

end heartsuit_ratio_l165_165067


namespace sum_of_selected_terms_l165_165083

variable {a : ℕ → ℚ} -- Define the arithmetic sequence as a function from natural numbers to rational numbers

noncomputable def sum_first_n_terms (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))

theorem sum_of_selected_terms (h₁ : sum_first_n_terms a 13 = 39) : a 6 + a 7 + a 8 = 13 :=
sorry

end sum_of_selected_terms_l165_165083


namespace correct_sampling_method_is_D_l165_165779

def is_simple_random_sample (method : String) : Prop :=
  method = "drawing lots method to select 3 out of 10 products for quality inspection"

theorem correct_sampling_method_is_D : 
  is_simple_random_sample "drawing lots method to select 3 out of 10 products for quality inspection" :=
sorry

end correct_sampling_method_is_D_l165_165779


namespace fair_die_proba_l165_165606
noncomputable def probability_of_six : ℚ := 1 / 6

theorem fair_die_proba : 
  (1 / 6 : ℚ) = probability_of_six :=
by
  sorry

end fair_die_proba_l165_165606


namespace symmetric_axis_and_vertex_l165_165486

theorem symmetric_axis_and_vertex (x : ℝ) : 
  (∀ x y, y = (1 / 2) * (x - 1)^2 + 6 → x = 1) 
  ∧ (1, 6) = (1, 6) :=
by 
  sorry

end symmetric_axis_and_vertex_l165_165486


namespace minimum_value_l165_165266

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 20) :
  (∃ (m : ℝ), m = (1 / x ^ 2 + 1 / y ^ 2) ∧ m ≥ 2 / 25) :=
by
  sorry

end minimum_value_l165_165266


namespace journey_duration_l165_165492

theorem journey_duration
  (distance : ℕ) (speed : ℕ) (h1 : distance = 48) (h2 : speed = 8) :
  distance / speed = 6 := 
by
  sorry

end journey_duration_l165_165492


namespace solve_for_k_l165_165480

theorem solve_for_k (k : ℕ) (h : 16 / k = 4) : k = 4 :=
sorry

end solve_for_k_l165_165480


namespace alcohol_percentage_l165_165066

theorem alcohol_percentage (x : ℝ)
  (h1 : 8 * x / 100 + 2 * 12 / 100 = 22.4 * 10 / 100) : x = 25 :=
by
  -- skip the proof
  sorry

end alcohol_percentage_l165_165066


namespace find_y_l165_165226

theorem find_y (y : ℝ) (hy : 0 < y) 
  (h : (Real.sqrt (12 * y)) * (Real.sqrt (6 * y)) * (Real.sqrt (18 * y)) * (Real.sqrt (9 * y)) = 27) : 
  y = 1 / 2 := 
sorry

end find_y_l165_165226


namespace terry_daily_driving_time_l165_165840

theorem terry_daily_driving_time 
  (d1: ℝ) (s1: ℝ)
  (d2: ℝ) (s2: ℝ)
  (d3: ℝ) (s3: ℝ)
  (h1 : d1 = 15) (h2 : s1 = 30)
  (h3 : d2 = 35) (h4 : s2 = 50)
  (h5 : d3 = 10) (h6 : s3 = 40) : 
  2 * ((d1 / s1) + (d2 / s2) + (d3 / s3)) = 2.9 := 
by
  sorry

end terry_daily_driving_time_l165_165840


namespace tangent_line_at_1_l165_165113

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x
def tangent_line_eq : ℝ × ℝ → ℝ := fun ⟨x, y⟩ => x - y - 1

theorem tangent_line_at_1 : tangent_line_eq (1, f 1) = 0 := by
  -- Proof would go here
  sorry

end tangent_line_at_1_l165_165113


namespace angle_is_10_l165_165288

theorem angle_is_10 (x : ℕ) (h1 : 180 - x = 2 * (90 - x) + 10) : x = 10 := 
by sorry

end angle_is_10_l165_165288


namespace allison_greater_prob_l165_165878

noncomputable def prob_allison_greater (p_brian : ℝ) (p_noah : ℝ) : ℝ :=
  p_brian * p_noah

theorem allison_greater_prob : prob_allison_greater (2/3) (1/2) = 1/3 :=
by {
  -- Calculate the combined probability
  sorry
}

end allison_greater_prob_l165_165878


namespace parabola_x_intercepts_count_l165_165345

theorem parabola_x_intercepts_count :
  let a := -3
  let b := 4
  let c := -1
  let discriminant := b ^ 2 - 4 * a * c
  discriminant ≥ 0 →
  let num_roots := if discriminant > 0 then 2 else if discriminant = 0 then 1 else 0
  num_roots = 2 := 
by {
  sorry
}

end parabola_x_intercepts_count_l165_165345


namespace type_b_quantity_l165_165524

theorem type_b_quantity 
  (x : ℕ)
  (hx : x + 2 * x + 4 * x = 140) : 
  2 * x = 40 := 
sorry

end type_b_quantity_l165_165524


namespace find_integer_values_of_m_l165_165395

theorem find_integer_values_of_m (m : ℤ) (x : ℚ) 
  (h₁ : 5 * x - 2 * m = 3 * x - 6 * m + 1)
  (h₂ : -3 < x ∧ x ≤ 2) : m = 0 ∨ m = 1 := 
by 
  sorry

end find_integer_values_of_m_l165_165395


namespace find_x_l165_165930

theorem find_x (x y z : ℚ) (h1 : (x * y) / (x + y) = 4) (h2 : (x * z) / (x + z) = 5) (h3 : (y * z) / (y + z) = 6) : x = 40 / 9 :=
by
  -- Structure the proof here
  sorry

end find_x_l165_165930


namespace find_triangle_sides_l165_165418

-- Define the conditions and translate them into Lean 4
theorem find_triangle_sides :
  (∃ a b c: ℝ, a + b + c = 40 ∧ a^2 + b^2 = c^2 ∧ 
   (a + 4)^2 + (b + 1)^2 = (c + 3)^2 ∧ 
   a = 8 ∧ b = 15 ∧ c = 17) :=
by 
  sorry

end find_triangle_sides_l165_165418


namespace therapy_hours_l165_165130

theorem therapy_hours (x n : ℕ) : 
  (x + 30) + 2 * x = 252 → 
  104 + (n - 1) * x = 400 → 
  x = 74 → 
  n = 5 := 
by
  sorry

end therapy_hours_l165_165130


namespace largest_angle_in_convex_pentagon_l165_165896

theorem largest_angle_in_convex_pentagon (x : ℕ) (h : (x - 2) + (x - 1) + x + (x + 1) + (x + 2) = 540) : 
  x + 2 = 110 :=
by
  sorry

end largest_angle_in_convex_pentagon_l165_165896


namespace johnny_words_l165_165848

def words_johnny (J : ℕ) :=
  let words_madeline := 2 * J
  let words_timothy := 2 * J + 30
  let total_words := J + words_madeline + words_timothy
  total_words = 3 * 260 → J = 150

-- Statement of the main theorem (no proof provided, hence sorry is used)
theorem johnny_words (J : ℕ) : words_johnny J :=
by sorry

end johnny_words_l165_165848


namespace stormi_lawns_mowed_l165_165841

def num_lawns_mowed (cars_washed : ℕ) (money_per_car : ℕ) 
                    (lawns_mowed : ℕ) (money_per_lawn : ℕ) 
                    (bike_cost : ℕ) (money_needed : ℕ) : Prop :=
  (cars_washed * money_per_car + lawns_mowed * money_per_lawn) = (bike_cost - money_needed)

theorem stormi_lawns_mowed : num_lawns_mowed 3 10 2 13 80 24 :=
by
  sorry

end stormi_lawns_mowed_l165_165841


namespace determine_even_condition_l165_165869

theorem determine_even_condition (x : ℤ) (m : ℤ) (h : m = x % 2) : m = 0 ↔ x % 2 = 0 :=
by sorry

end determine_even_condition_l165_165869


namespace sum_of_three_numbers_l165_165763

theorem sum_of_three_numbers :
  1.35 + 0.123 + 0.321 = 1.794 :=
sorry

end sum_of_three_numbers_l165_165763


namespace profit_percentage_l165_165378

theorem profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) 
  (h1 : cost_price = 66.5) (h2 : marked_price = 87.5) (h3 : discount_rate = 0.05) : 
  (100 * ((marked_price * (1 - discount_rate) - cost_price) / cost_price)) = 25 :=
by
  sorry

end profit_percentage_l165_165378


namespace sum_of_squares_l165_165039

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 72) : x^2 + y^2 = 180 :=
sorry

end sum_of_squares_l165_165039


namespace company_fund_initial_amount_l165_165449

theorem company_fund_initial_amount (n : ℕ) (fund_initial : ℤ) 
  (h1 : ∃ n, fund_initial = 60 * n - 10)
  (h2 : ∃ n, 55 * n + 120 = fund_initial + 130)
  : fund_initial = 1550 := 
sorry

end company_fund_initial_amount_l165_165449


namespace evaporation_period_l165_165607

theorem evaporation_period
  (total_water : ℕ)
  (daily_evaporation_rate : ℝ)
  (percentage_evaporated : ℝ)
  (evaporation_period_days : ℕ)
  (h_total_water : total_water = 10)
  (h_daily_evaporation_rate : daily_evaporation_rate = 0.006)
  (h_percentage_evaporated : percentage_evaporated = 0.03)
  (h_evaporation_period_days : evaporation_period_days = 50):
  (percentage_evaporated * total_water) / daily_evaporation_rate = evaporation_period_days := by
  sorry

end evaporation_period_l165_165607


namespace sqrt_9_minus_1_eq_2_l165_165517

theorem sqrt_9_minus_1_eq_2 : Real.sqrt 9 - 1 = 2 := by
  sorry

end sqrt_9_minus_1_eq_2_l165_165517


namespace lowest_two_digit_number_whose_digits_product_is_12_l165_165997

def is_valid_two_digit_number (n : ℕ) : Prop :=
  10 <= n ∧ n < 100 ∧ ∃ d1 d2 : ℕ, 1 ≤ d1 ∧ d1 < 10 ∧ 1 ≤ d2 ∧ d2 < 10 ∧ n = 10 * d1 + d2 ∧ d1 * d2 = 12

theorem lowest_two_digit_number_whose_digits_product_is_12 :
  ∃ n : ℕ, is_valid_two_digit_number n ∧ ∀ m : ℕ, is_valid_two_digit_number m → n ≤ m ∧ n = 26 :=
sorry

end lowest_two_digit_number_whose_digits_product_is_12_l165_165997


namespace exists_nonneg_integers_l165_165281

theorem exists_nonneg_integers (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (x y z t : ℕ), (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ t ≠ 0) ∧ t < p ∧ x^2 + y^2 + z^2 = t * p :=
sorry

end exists_nonneg_integers_l165_165281


namespace max_interval_length_l165_165058

def m (x : ℝ) : ℝ := x^2 - 3 * x + 4
def n (x : ℝ) : ℝ := 2 * x - 3

def are_close_functions (m n : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |m x - n x| ≤ 1

theorem max_interval_length
  (h : are_close_functions m n 2 3) :
  3 - 2 = 1 :=
sorry

end max_interval_length_l165_165058


namespace train_speed_l165_165915

theorem train_speed (length_bridge : ℕ) (time_total : ℕ) (time_on_bridge : ℕ) (speed_of_train : ℕ) 
  (h1 : length_bridge = 800)
  (h2 : time_total = 60)
  (h3 : time_on_bridge = 40)
  (h4 : length_bridge + (time_total - time_on_bridge) * speed_of_train = time_total * speed_of_train) :
  speed_of_train = 20 := sorry

end train_speed_l165_165915


namespace cos_alpha_minus_beta_cos_alpha_plus_beta_l165_165151

variables (α β : Real) (h1 : 0 < α ∧ α < π/2 ∧ 0 < β ∧ β < π/2)
           (h2 : Real.tan α * Real.tan β = 13/7)
           (h3 : Real.sin (α - β) = sqrt 5 / 3)

-- Part (1): Prove that cos (α - β) = 2/3
theorem cos_alpha_minus_beta : Real.cos (α - β) = 2 / 3 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

-- Part (2): Prove that cos (α + β) = -1/5
theorem cos_alpha_plus_beta : Real.cos (α + β) = -1 / 5 := by
  have h := h1
  have h := h2
  have h := h3
  sorry

end cos_alpha_minus_beta_cos_alpha_plus_beta_l165_165151


namespace regression_total_sum_of_squares_l165_165595

variables (y : Fin 10 → ℝ) (y_hat : Fin 10 → ℝ)
variables (residual_sum_of_squares : ℝ) 

-- Given conditions
def R_squared := 0.95
def RSS := 120.53

-- The total sum of squares is what we need to prove
noncomputable def total_sum_of_squares := 2410.6

-- Statement to prove
theorem regression_total_sum_of_squares :
  1 - RSS / total_sum_of_squares = R_squared := by
sorry

end regression_total_sum_of_squares_l165_165595


namespace cyclist_overtake_points_l165_165483

theorem cyclist_overtake_points (p c : ℝ) (track_length : ℝ) (h1 : c = 1.55 * p) (h2 : track_length = 55) : 
  ∃ n, n = 11 :=
by
  -- we'll add the proof steps later
  sorry

end cyclist_overtake_points_l165_165483


namespace correct_operation_l165_165939

theorem correct_operation (a b : ℝ) : (-a * b^2)^2 = a^2 * b^4 :=
  sorry

end correct_operation_l165_165939


namespace solution_set_inequality_l165_165314

theorem solution_set_inequality : {x : ℝ | (x-1)*(x-2) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end solution_set_inequality_l165_165314


namespace equal_distribution_l165_165834

theorem equal_distribution 
  (total_profit : ℕ) 
  (num_employees : ℕ) 
  (profit_kept_percent : ℕ) 
  (remaining_to_distribute : ℕ)
  (each_employee_gets : ℕ) :
  total_profit = 50 →
  num_employees = 9 →
  profit_kept_percent = 10 →
  remaining_to_distribute = total_profit - (total_profit * profit_kept_percent / 100) →
  each_employee_gets = remaining_to_distribute / num_employees →
  each_employee_gets = 5 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end equal_distribution_l165_165834


namespace parametric_equation_correct_max_min_x_plus_y_l165_165290

noncomputable def parametric_equation (φ : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ)

theorem parametric_equation_correct (ρ θ : ℝ) (h : ρ^2 - 4 * Real.sqrt 2 * Real.cos (θ - π/4) + 6 = 0) :
  ∃ (φ : ℝ), parametric_equation φ = ( 2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ) := 
sorry

theorem max_min_x_plus_y (P : ℝ × ℝ) (hP : ∃ (φ : ℝ), P = parametric_equation φ) :
  ∃ f : ℝ, (P.fst + P.snd) = f ∧ (f = 6 ∨ f = 2) :=
sorry

end parametric_equation_correct_max_min_x_plus_y_l165_165290


namespace part1_part2_l165_165330

-- Define set A and set B for m = 3
def setA : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def setB_m3 : Set ℝ := {x | x^2 - 2 * x - 3 < 0}

-- Define the complement of B in ℝ and the intersection of complements
def complB_m3 : Set ℝ := {x | x ≤ -1 ∨ x ≥ 3}
def intersection_complB_A : Set ℝ := complB_m3 ∩ setA

-- Verify that the intersection of the complement of B and A equals the given set
theorem part1 : intersection_complB_A = {x | 3 ≤ x ∧ x ≤ 5} :=
by
  sorry

-- Define set A and the intersection of A and B
def setA' : Set ℝ := {x | (x - 5) / (x + 1) ≤ 0}
def setAB : Set ℝ := {x | -1 < x ∧ x < 4}

-- Given A ∩ B = {x | -1 < x < 4}, determine m such that B = {x | -1 < x < 4}
theorem part2 : ∃ m : ℝ, (setA' ∩ {x | x^2 - 2 * x - m < 0} = setAB) ∧ m = 8 :=
by
  sorry

end part1_part2_l165_165330


namespace num_ways_award_medals_l165_165340

-- There are 8 sprinters in total
def num_sprinters : ℕ := 8

-- Three of the sprinters are Americans
def num_americans : ℕ := 3

-- The number of non-American sprinters
def num_non_americans : ℕ := num_sprinters - num_americans

-- The question to prove: the number of ways the medals can be awarded if at most one American gets a medal
theorem num_ways_award_medals 
  (n : ℕ) (m : ℕ) (k : ℕ) (h1 : n = num_sprinters) (h2 : m = num_americans) 
  (h3 : k = num_non_americans) 
  (no_american : ℕ := k * (k - 1) * (k - 2)) 
  (one_american : ℕ := m * 3 * k * (k - 1)) 
  : no_american + one_american = 240 :=
sorry

end num_ways_award_medals_l165_165340


namespace man_and_son_work_together_l165_165563

theorem man_and_son_work_together (man_days son_days : ℕ) (h_man : man_days = 15) (h_son : son_days = 10) :
  (1 / (1 / man_days + 1 / son_days) = 6) :=
by
  rw [h_man, h_son]
  sorry

end man_and_son_work_together_l165_165563


namespace hyperbola_asymptotes_identical_l165_165813

theorem hyperbola_asymptotes_identical (x y M : ℝ) :
  (∃ (a b : ℝ), a = 3 ∧ b = 4 ∧ (y = (b/a) * x ∨ y = -(b/a) * x)) ∧
  (∃ (c d : ℝ), c = 5 ∧ y = (c / d) * x ∨ y = -(c / d) * x) →
  M = (225 / 16) :=
by sorry

end hyperbola_asymptotes_identical_l165_165813


namespace division_criterion_based_on_stroke_l165_165616

-- Definition of a drawable figure with a single stroke
def drawable_in_one_stroke (figure : Type) : Prop := sorry -- exact conditions can be detailed with figure representation

-- Example figures for the groups (types can be extended based on actual representation)
def Group1 := {fig1 : Type // drawable_in_one_stroke fig1}
def Group2 := {fig2 : Type // ¬drawable_in_one_stroke fig2}

-- Problem Statement:
theorem division_criterion_based_on_stroke (fig : Type) :
  (drawable_in_one_stroke fig ∨ ¬drawable_in_one_stroke fig) := by
  -- We state that every figure belongs to either Group1 or Group2
  sorry

end division_criterion_based_on_stroke_l165_165616


namespace line_shift_up_l165_165081

theorem line_shift_up (x y : ℝ) (k : ℝ) (h : y = -2 * x - 4) : 
    y + k = -2 * x - 1 := by
  sorry

end line_shift_up_l165_165081


namespace roots_distinct_and_real_l165_165756

variables (b d : ℝ)
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem roots_distinct_and_real (h₁ : discriminant b (-3 * Real.sqrt 5) d = 25) :
    ∃ x1 x2 : ℝ, x1 ≠ x2 :=
by 
  sorry

end roots_distinct_and_real_l165_165756


namespace wall_length_proof_l165_165929

noncomputable def volume_of_brick (length width height : ℝ) : ℝ := length * width * height

noncomputable def total_volume (brick_volume num_of_bricks : ℝ) : ℝ := brick_volume * num_of_bricks

theorem wall_length_proof
  (height_of_wall : ℝ) (width_of_walls : ℝ) (num_of_bricks : ℝ)
  (length_of_brick width_of_brick height_of_brick : ℝ)
  (total_volume_of_bricks : ℝ) :
  total_volume (volume_of_brick length_of_brick width_of_brick height_of_brick) num_of_bricks = total_volume_of_bricks →
  volume_of_brick length_of_wall height_of_wall width_of_walls = total_volume_of_bricks →
  height_of_wall = 600 →
  width_of_walls = 2 →
  num_of_bricks = 2909.090909090909 →
  length_of_brick = 5 →
  width_of_brick = 11 →
  height_of_brick = 6 →
  total_volume_of_bricks = 960000 →
  length_of_wall = 800 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end wall_length_proof_l165_165929


namespace ellipse_equation_l165_165189

theorem ellipse_equation
  (x y t : ℝ)
  (h1 : x = (3 * (Real.sin t - 2)) / (3 - Real.cos t))
  (h2 : y = (4 * (Real.cos t - 6)) / (3 - Real.cos t))
  (h3 : ∀ t : ℝ, (Real.cos t)^2 + (Real.sin t)^2 = 1) :
  ∃ (A B C D E F : ℤ), (9 * x^2 + 36 * x * y + 9 * y^2 + 216 * x + 432 * y + 1440 = 0) ∧ 
  (Int.gcd (Int.gcd (Int.gcd (Int.gcd (Int.gcd A B) C) D) E) F = 1) ∧
  (|A| + |B| + |C| + |D| + |E| + |F| = 2142) :=
sorry

end ellipse_equation_l165_165189


namespace b_minus_a_equals_two_l165_165479

open Set

variables {a b : ℝ}

theorem b_minus_a_equals_two (h₀ : {1, a + b, a} = ({0, b / a, b} : Finset ℝ)) (h₁ : a ≠ 0) : b - a = 2 :=
sorry

end b_minus_a_equals_two_l165_165479


namespace rectangle_area_l165_165530

theorem rectangle_area
  (width : ℕ) (length : ℕ)
  (h1 : width = 7)
  (h2 : length = 4 * width) :
  length * width = 196 := by
  sorry

end rectangle_area_l165_165530


namespace ratio_sheep_to_horses_l165_165343

theorem ratio_sheep_to_horses (sheep horses : ℕ) (total_horse_food daily_food_per_horse : ℕ)
  (h1 : sheep = 16)
  (h2 : total_horse_food = 12880)
  (h3 : daily_food_per_horse = 230)
  (h4 : horses = total_horse_food / daily_food_per_horse) :
  (sheep / gcd sheep horses) / (horses / gcd sheep horses) = 2 / 7 := by
  sorry

end ratio_sheep_to_horses_l165_165343


namespace linear_function_implies_m_value_l165_165476

variable (x m : ℝ)

theorem linear_function_implies_m_value :
  (∃ y : ℝ, y = (m-3)*x^(m^2-8) + m + 1 ∧ ∀ x1 x2 : ℝ, y = y * (x2 - x1) + y * x1) → m = -3 :=
by
  sorry

end linear_function_implies_m_value_l165_165476


namespace derivative_at_pi_l165_165512

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_at_pi :
  deriv f π = -1 / (π^2) :=
sorry

end derivative_at_pi_l165_165512
