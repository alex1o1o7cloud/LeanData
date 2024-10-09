import Mathlib

namespace unique_intersection_point_l1210_121074

def f (x : ℝ) : ℝ := x^3 + 3 * x^2 + 9 * x + 15

theorem unique_intersection_point : ∃ a : ℝ, f a = a ∧ f a = -1 ∧ f a = f⁻¹ a :=
by 
  sorry

end unique_intersection_point_l1210_121074


namespace determine_coefficients_l1210_121046

theorem determine_coefficients (p q : ℝ) :
  (∃ x : ℝ, x^2 + p * x + q = 0 ∧ x = p) ∧ (∃ y : ℝ, y^2 + p * y + q = 0 ∧ y = q)
  ↔ (p = 0 ∧ q = 0) ∨ (p = 1 ∧ q = -2) := by
sorry

end determine_coefficients_l1210_121046


namespace find_sum_of_p_q_r_s_l1210_121057

theorem find_sum_of_p_q_r_s 
    (p q r s : ℝ)
    (h1 : r + s = 12 * p)
    (h2 : r * s = -13 * q)
    (h3 : p + q = 12 * r)
    (h4 : p * q = -13 * s)
    (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s) :
    p + q + r + s = 2028 := 
sorry

end find_sum_of_p_q_r_s_l1210_121057


namespace beckett_younger_than_olaf_l1210_121005

-- Define variables for ages
variables (O B S J : ℕ) (x : ℕ)

-- Express conditions as Lean hypotheses
def conditions :=
  B = O - x ∧  -- Beckett's age
  B = 12 ∧    -- Beckett is 12 years old
  S = O - 2 ∧ -- Shannen's age
  J = 2 * S + 5 ∧ -- Jack's age
  O + B + S + J = 71 -- Sum of ages
  
-- The theorem stating that Beckett is 8 years younger than Olaf
theorem beckett_younger_than_olaf (h : conditions O B S J x) : x = 8 :=
by
  -- The proof is omitted (using sorry)
  sorry

end beckett_younger_than_olaf_l1210_121005


namespace problems_completed_l1210_121025

theorem problems_completed (p t : ℕ) (hp : p > 10) (eqn : p * t = (2 * p - 2) * (t - 1)) :
  p * t = 48 := 
sorry

end problems_completed_l1210_121025


namespace bill_score_l1210_121047

theorem bill_score (B J S E : ℕ)
                   (h1 : B = J + 20)
                   (h2 : B = S / 2)
                   (h3 : E = B + J - 10)
                   (h4 : B + J + S + E = 250) :
                   B = 50 := 
by sorry

end bill_score_l1210_121047


namespace initial_logs_l1210_121097

theorem initial_logs (x : ℕ) (h1 : x - 3 - 3 - 3 + 2 + 2 + 2 = 3) : x = 6 := by
  sorry

end initial_logs_l1210_121097


namespace mikes_lower_rate_l1210_121007

theorem mikes_lower_rate (x : ℕ) (high_rate : ℕ) (total_paid : ℕ) (lower_payments : ℕ) (higher_payments : ℕ)
  (h1 : high_rate = 310)
  (h2 : total_paid = 3615)
  (h3 : lower_payments = 5)
  (h4 : higher_payments = 7)
  (h5 : lower_payments * x + higher_payments * high_rate = total_paid) :
  x = 289 :=
sorry

end mikes_lower_rate_l1210_121007


namespace percentage_gain_on_powerlifting_total_l1210_121011

def initialTotal : ℝ := 2200
def initialWeight : ℝ := 245
def weightIncrease : ℝ := 8
def finalWeight : ℝ := initialWeight + weightIncrease
def liftingRatio : ℝ := 10
def finalTotal : ℝ := finalWeight * liftingRatio

theorem percentage_gain_on_powerlifting_total :
  ∃ (P : ℝ), initialTotal * (1 + P / 100) = finalTotal :=
by
  sorry

end percentage_gain_on_powerlifting_total_l1210_121011


namespace days_in_month_l1210_121089

theorem days_in_month 
  (S : ℕ) (D : ℕ) (h1 : 150 * S + 120 * D = (S + D) * 125) (h2 : S = 5) :
  S + D = 30 :=
by
  sorry

end days_in_month_l1210_121089


namespace namjoon_used_pencils_l1210_121030

variable (taehyungUsed : ℕ) (namjoonUsed : ℕ)

/-- 
Statement:
Taehyung and Namjoon each initially have 10 pencils.
Taehyung gives 3 of his remaining pencils to Namjoon.
After this, Taehyung ends up with 6 pencils and Namjoon ends up with 6 pencils.
We need to prove that Namjoon used 7 pencils.
-/
theorem namjoon_used_pencils (H1 : 10 - taehyungUsed = 9 - 3)
  (H2 : 13 - namjoonUsed = 6) : namjoonUsed = 7 :=
sorry

end namjoon_used_pencils_l1210_121030


namespace decreasing_interval_b_l1210_121014

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := - (1 / 2) * x ^ 2 + b * Real.log x

theorem decreasing_interval_b (b : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici (Real.sqrt 2) → ∀ x1 x2 : ℝ, x1 ∈ Set.Ici (Real.sqrt 2) → x2 ∈ Set.Ici (Real.sqrt 2) → 
   x1 ≤ x2 → f x1 b ≥ f x2 b) ↔ b ≤ 2 :=
by
  sorry

end decreasing_interval_b_l1210_121014


namespace intersection_A_B_l1210_121067

def A := { x : ℝ | -1 < x ∧ x ≤ 3 }
def B := { x : ℝ | 0 < x ∧ x < 10 }

theorem intersection_A_B : A ∩ B = { x : ℝ | 0 < x ∧ x ≤ 3 } :=
  by sorry

end intersection_A_B_l1210_121067


namespace difference_brothers_l1210_121037

def aaron_brothers : ℕ := 4
def bennett_brothers : ℕ := 6

theorem difference_brothers : 2 * aaron_brothers - bennett_brothers = 2 := by
  sorry

end difference_brothers_l1210_121037


namespace transactions_Mabel_l1210_121049

variable {M A C J : ℝ}

theorem transactions_Mabel (h1 : A = 1.10 * M)
                          (h2 : C = 2 / 3 * A)
                          (h3 : J = C + 18)
                          (h4 : J = 84) :
  M = 90 :=
by
  sorry

end transactions_Mabel_l1210_121049


namespace sum_of_center_coordinates_l1210_121017

theorem sum_of_center_coordinates 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : (x1, y1) = (4, 3)) 
  (h2 : (x2, y2) = (-6, 5)) : 
  (x1 + x2) / 2 + (y1 + y2) / 2 = 3 := by
  sorry

end sum_of_center_coordinates_l1210_121017


namespace problem_I_problem_II_problem_III_problem_IV_l1210_121031

/-- Problem I: Given: (2x - y)^2 = 1, Prove: y = 2x - 1 ∨ y = 2x + 1 --/
theorem problem_I (x y : ℝ) : (2 * x - y) ^ 2 = 1 → (y = 2 * x - 1) ∨ (y = 2 * x + 1) := 
sorry

/-- Problem II: Given: 16x^4 - 8x^2y^2 + y^4 - 8x^2 - 2y^2 + 1 = 0, Prove: y = 2x - 1 ∨ y = -2x - 1 ∨ y = 2x + 1 ∨ y = -2x + 1 --/
theorem problem_II (x y : ℝ) : 16 * x^4 - 8 * x^2 * y^2 + y^4 - 8 * x^2 - 2 * y^2 + 1 = 0 ↔ 
    (y = 2 * x - 1) ∨ (y = -2 * x - 1) ∨ (y = 2 * x + 1) ∨ (y = -2 * x + 1) := 
sorry

/-- Problem III: Given: x^2 * (1 - |y| / y) + y^2 + y * |y| = 8, Prove: (y = 2 ∧ y > 0) ∨ ((x = 2 ∨ x = -2) ∧ y < 0) --/
theorem problem_III (x y : ℝ) (hy : y ≠ 0) : x^2 * (1 - abs y / y) + y^2 + y * abs y = 8 →
    (y = 2 ∧ y > 0) ∨ ((x = 2 ∨ x = -2) ∧ y < 0) := 
sorry

/-- Problem IV: Given: x^2 + x * |x| + y^2 + (|x| * y^2 / x) = 8, Prove: x^2 + y^2 = 4 ∧ x > 0 --/
theorem problem_IV (x y : ℝ) (hx : x ≠ 0) : x^2 + x * abs x + y^2 + (abs x * y^2 / x) = 8 →
    (x^2 + y^2 = 4 ∧ x > 0) := 
sorry

end problem_I_problem_II_problem_III_problem_IV_l1210_121031


namespace julia_download_songs_l1210_121048

-- Basic definitions based on conditions
def internet_speed_MBps : ℕ := 20
def song_size_MB : ℕ := 5
def half_hour_seconds : ℕ := 30 * 60

-- Statement of the proof problem
theorem julia_download_songs : 
  (internet_speed_MBps * half_hour_seconds) / song_size_MB = 7200 :=
by
  sorry

end julia_download_songs_l1210_121048


namespace fraction_sum_l1210_121078

theorem fraction_sum :
  (1 / 3 + 1 / 2 - 5 / 6 + 1 / 5 + 1 / 4 - 9 / 20 - 5 / 6 : ℚ) = -5 / 6 :=
by sorry

end fraction_sum_l1210_121078


namespace vectors_parallel_y_eq_minus_one_l1210_121024

theorem vectors_parallel_y_eq_minus_one (y : ℝ) :
  let a := (1, 2)
  let b := (1, -2 * y)
  b.1 * a.2 - a.1 * b.2 = 0 → y = -1 :=
by
  intros a b h
  simp at h
  sorry

end vectors_parallel_y_eq_minus_one_l1210_121024


namespace meaningful_expression_l1210_121018

-- Definition stating the meaningfulness of the expression (condition)
def is_meaningful (a : ℝ) : Prop := (a - 1) ≠ 0

-- Theorem stating that for the expression to be meaningful, a ≠ 1
theorem meaningful_expression (a : ℝ) : is_meaningful a ↔ a ≠ 1 :=
by sorry

end meaningful_expression_l1210_121018


namespace steve_nickels_dimes_l1210_121034

theorem steve_nickels_dimes (n d : ℕ) (h1 : d = n + 4) (h2 : 5 * n + 10 * d = 70) : n = 2 :=
by
  -- The proof goes here
  sorry

end steve_nickels_dimes_l1210_121034


namespace area_of_circle_with_radius_2_is_4pi_l1210_121016

theorem area_of_circle_with_radius_2_is_4pi :
  ∀ (π : ℝ), ∀ (r : ℝ), r = 2 → π > 0 → π * r^2 = 4 * π := 
by
  intros π r hr hπ
  sorry

end area_of_circle_with_radius_2_is_4pi_l1210_121016


namespace sum_of_first_15_terms_is_largest_l1210_121021

theorem sum_of_first_15_terms_is_largest
  (a : ℕ → ℝ)
  (s : ℕ → ℝ)
  (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, s n = n * a 1 + (n * (n - 1) * d) / 2)
  (h1: 13 * a 6 = 19 * (a 6 + 3 * d))
  (h2: a 1 > 0) : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≠ 15 → s 15 > s n :=
by
  sorry

end sum_of_first_15_terms_is_largest_l1210_121021


namespace solution_set_of_inequality_l1210_121044

theorem solution_set_of_inequality :
  {x : ℝ | x^2 < x + 6} = {x : ℝ | -2 < x ∧ x < 3} := 
sorry

end solution_set_of_inequality_l1210_121044


namespace previous_salary_is_40_l1210_121069

-- Define the conditions
def new_salary : ℕ := 80
def percentage_increase : ℕ := 100

-- Proven goal: John's previous salary before the raise
def previous_salary : ℕ := new_salary / 2

theorem previous_salary_is_40 : previous_salary = 40 := 
by
  -- Proof steps would go here
  sorry

end previous_salary_is_40_l1210_121069


namespace common_factor_l1210_121087

theorem common_factor (x y a b : ℤ) : 
  3 * x * (a - b) - 9 * y * (b - a) = 3 * (a - b) * (x + 3 * y) :=
by {
  sorry
}

end common_factor_l1210_121087


namespace percent_of_value_and_divide_l1210_121038

theorem percent_of_value_and_divide (x : ℝ) (y : ℝ) (z : ℝ) (h : x = 1/300 * 180) (h1 : y = x / 6) : 
  y = 0.1 := 
by
  sorry

end percent_of_value_and_divide_l1210_121038


namespace find_ax5_by5_l1210_121095

theorem find_ax5_by5 (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 9)
  (h3 : a * x^3 + b * y^3 = 21)
  (h4 : a * x^4 + b * y^4 = 55) :
  a * x^5 + b * y^5 = -131 :=
sorry

end find_ax5_by5_l1210_121095


namespace fraction_of_earth_surface_habitable_for_humans_l1210_121041

theorem fraction_of_earth_surface_habitable_for_humans
  (total_land_fraction : ℚ) (habitable_land_fraction : ℚ)
  (h1 : total_land_fraction = 1/3)
  (h2 : habitable_land_fraction = 3/4) :
  (total_land_fraction * habitable_land_fraction) = 1/4 :=
by
  sorry

end fraction_of_earth_surface_habitable_for_humans_l1210_121041


namespace evaluate_m_l1210_121039

theorem evaluate_m :
  ∀ m : ℝ, (243:ℝ)^(1/5) = 3^m → m = 1 :=
by
  intro m
  sorry

end evaluate_m_l1210_121039


namespace find_angles_and_area_l1210_121012

noncomputable def angles_in_arithmetic_progression (A B C : ℝ) : Prop :=
  A + C = 2 * B ∧ A + B + C = 180

noncomputable def side_ratios (a b : ℝ) : Prop :=
  a / b = Real.sqrt 2 / Real.sqrt 3

noncomputable def triangle_area (a b c A B C : ℝ) : ℝ :=
  (1/2) * a * c * Real.sin B

theorem find_angles_and_area :
  ∃ (A B C a b c : ℝ), 
    angles_in_arithmetic_progression A B C ∧ 
    side_ratios a b ∧ 
    c = 2 ∧ 
    A = 45 ∧ 
    B = 60 ∧ 
    C = 75 ∧ 
    triangle_area a b c A B C = 3 - Real.sqrt 3 :=
sorry

end find_angles_and_area_l1210_121012


namespace paco_ate_sweet_cookies_l1210_121059

noncomputable def PacoCookies (sweet: Nat) (salty: Nat) (salty_eaten: Nat) (extra_sweet: Nat) : Nat :=
  let corrected_salty_eaten := if salty_eaten > salty then salty else salty_eaten
  corrected_salty_eaten + extra_sweet

theorem paco_ate_sweet_cookies : PacoCookies 39 6 23 9 = 15 := by
  sorry

end paco_ate_sweet_cookies_l1210_121059


namespace parts_processed_per_hour_before_innovation_l1210_121090

variable (x : ℝ) (h : 1500 / x - 1500 / (2.5 * x) = 18)

theorem parts_processed_per_hour_before_innovation : x = 50 :=
by
  sorry

end parts_processed_per_hour_before_innovation_l1210_121090


namespace matrix_not_invertible_x_l1210_121015

theorem matrix_not_invertible_x (x : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := ![![2 + x, 9], ![4 - x, 10]]
  A.det = 0 ↔ x = 16 / 19 := sorry

end matrix_not_invertible_x_l1210_121015


namespace find_S25_l1210_121062

variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Definitions based on conditions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a 1 - a 0
def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop := ∀ n, S n = (n / 2) * (2 * a 0 + (n - 1) * (a 1 - a 0))

-- Condition that given S_{15} - S_{10} = 1
axiom sum_difference : S 15 - S 10 = 1

-- Theorem we need to prove
theorem find_S25 (h_arith : is_arithmetic_sequence a) (h_sum : sum_of_first_n_terms a S) : S 25 = 5 :=
by
-- Placeholder for the actual proof
sorry

end find_S25_l1210_121062


namespace jason_attended_games_l1210_121086

-- Define the conditions as given in the problem
def games_planned_this_month : ℕ := 11
def games_planned_last_month : ℕ := 17
def games_missed : ℕ := 16

-- Define the total number of games planned
def games_planned_total : ℕ := games_planned_this_month + games_planned_last_month

-- Define the number of games attended
def games_attended : ℕ := games_planned_total - games_missed

-- Prove that Jason attended 12 games
theorem jason_attended_games : games_attended = 12 := by
  -- The proof is omitted, but the theorem statement is required
  sorry

end jason_attended_games_l1210_121086


namespace acute_angle_inequality_l1210_121063

theorem acute_angle_inequality (α : ℝ) (h₀ : 0 < α) (h₁ : α < π / 2) :
  α < (Real.sin α + Real.tan α) / 2 := 
sorry

end acute_angle_inequality_l1210_121063


namespace initial_oranges_is_sum_l1210_121004

-- Define the number of oranges taken by Jonathan
def oranges_taken : ℕ := 45

-- Define the number of oranges left in the box
def oranges_left : ℕ := 51

-- The theorem states that the initial number of oranges is the sum of the oranges taken and those left
theorem initial_oranges_is_sum : oranges_taken + oranges_left = 96 := 
by 
  -- This is where the proof would go
  sorry

end initial_oranges_is_sum_l1210_121004


namespace dice_probability_l1210_121026

theorem dice_probability :
  let prob_one_digit := (9:ℚ) / 20
  let prob_two_digit := (11:ℚ) / 20
  let prob := 10 * (prob_two_digit^2) * (prob_one_digit^3)
  prob = 1062889 / 128000000 := 
by 
  sorry

end dice_probability_l1210_121026


namespace shirt_cost_is_43_l1210_121000

def pantsCost : ℕ := 140
def tieCost : ℕ := 15
def totalPaid : ℕ := 200
def changeReceived : ℕ := 2

def totalCostWithoutShirt := totalPaid - changeReceived
def totalCostWithPantsAndTie := pantsCost + tieCost
def shirtCost := totalCostWithoutShirt - totalCostWithPantsAndTie

theorem shirt_cost_is_43 : shirtCost = 43 := by
  have h1 : totalCostWithoutShirt = 198 := by rfl
  have h2 : totalCostWithPantsAndTie = 155 := by rfl
  have h3 : shirtCost = totalCostWithoutShirt - totalCostWithPantsAndTie := by rfl
  rw [h1, h2] at h3
  exact h3

end shirt_cost_is_43_l1210_121000


namespace sum_prime_factors_1170_l1210_121029

theorem sum_prime_factors_1170 : 
  let smallest_prime_factor := 2
  let largest_prime_factor := 13
  (smallest_prime_factor + largest_prime_factor) = 15 :=
by
  sorry

end sum_prime_factors_1170_l1210_121029


namespace prob_shooting_A_first_l1210_121019

-- Define the probabilities
def prob_A_hits : ℝ := 0.4
def prob_A_misses : ℝ := 0.6
def prob_B_hits : ℝ := 0.6
def prob_B_misses : ℝ := 0.4

-- Define the overall problem
theorem prob_shooting_A_first (k : ℕ) (ξ : ℕ) (hξ : ξ = k) :
  ((prob_A_misses * prob_B_misses)^(k-1)) * (1 - (prob_A_misses * prob_B_misses)) = 0.24^(k-1) * 0.76 :=
by
  -- Placeholder for proof
  sorry

end prob_shooting_A_first_l1210_121019


namespace hexagon_side_squares_sum_l1210_121056

variables {P Q R P' Q' R' A B C D E F : Type}
variables (a1 a2 a3 b1 b2 b3 : ℝ)
variables (h_eq_triangles : congruent (triangle P Q R) (triangle P' Q' R'))
variables (h_sides : 
  AB = a1 ∧ BC = b1 ∧ CD = a2 ∧ 
  DE = b2 ∧ EF = a3 ∧ FA = b3)
  
theorem hexagon_side_squares_sum :
  a1^2 + a2^2 + a3^2 = b1^2 + b2^2 + b3^2 :=
sorry

end hexagon_side_squares_sum_l1210_121056


namespace find_p_q_sum_l1210_121088

theorem find_p_q_sum (p q : ℝ) 
  (sum_condition : p / 3 = 8) 
  (product_condition : q / 3 = 12) : 
  p + q = 60 :=
by
  sorry

end find_p_q_sum_l1210_121088


namespace price_reduction_to_achieve_profit_l1210_121032

/-- 
A certain store sells clothing that cost $45$ yuan each to purchase for $65$ yuan each.
On average, they can sell $30$ pieces per day. For each $1$ yuan price reduction, 
an additional $5$ pieces can be sold per day. Given these conditions, 
prove that to achieve a daily profit of $800$ yuan, 
the price must be reduced by $10$ yuan per piece.
-/
theorem price_reduction_to_achieve_profit :
  ∃ x : ℝ, x = 10 ∧
    let original_cost := 45
    let original_price := 65
    let original_pieces_sold := 30
    let additional_pieces_per_yuan := 5
    let target_profit := 800
    let new_profit_per_piece := (original_price - original_cost) - x
    let new_pieces_sold := original_pieces_sold + additional_pieces_per_yuan * x
    new_profit_per_piece * new_pieces_sold = target_profit :=
by {
  sorry
}

end price_reduction_to_achieve_profit_l1210_121032


namespace tv_horizontal_length_l1210_121084

-- Conditions
def is_rectangular_tv (width height : ℝ) : Prop :=
width / height = 9 / 12

def diagonal_is (d : ℝ) : Prop :=
d = 32

-- Theorem to prove
theorem tv_horizontal_length (width height diagonal : ℝ) 
(h1 : is_rectangular_tv width height) 
(h2 : diagonal_is diagonal) : 
width = 25.6 := by 
sorry

end tv_horizontal_length_l1210_121084


namespace borrowing_period_l1210_121035

theorem borrowing_period 
  (principal : ℕ) (rate_1 : ℕ) (rate_2 : ℕ) (gain : ℕ)
  (h1 : principal = 5000)
  (h2 : rate_1 = 4)
  (h3 : rate_2 = 8)
  (h4 : gain = 200)
  : ∃ n : ℕ, n = 1 :=
by
  sorry

end borrowing_period_l1210_121035


namespace arithmetic_sequence_sum_l1210_121098

variable {a : ℕ → ℕ}

theorem arithmetic_sequence_sum
  (h1 : a 1 = 2)
  (h2 : a 2 + a 3 = 13) :
  a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l1210_121098


namespace categorize_numbers_l1210_121075

def numbers : List ℚ := [-16/10, -5/6, 89/10, -7, 1/12, 0, 25]

def is_positive (x : ℚ) : Prop := x > 0
def is_negative_fraction (x : ℚ) : Prop := x < 0 ∧ x.den ≠ 1
def is_negative_integer (x : ℚ) : Prop := x < 0 ∧ x.den = 1

theorem categorize_numbers :
  { x | x ∈ numbers ∧ is_positive x } = { 89 / 10, 1 / 12, 25 } ∧
  { x | x ∈ numbers ∧ is_negative_fraction x } = { -5 / 6 } ∧
  { x | x ∈ numbers ∧ is_negative_integer x } = { -7 } := by
  sorry

end categorize_numbers_l1210_121075


namespace factorial_div_l1210_121040

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l1210_121040


namespace probability_of_three_primes_out_of_six_l1210_121070

-- Define the conditions
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11

-- Given six 12-sided fair dice
def total_dice : ℕ := 6
def sides : ℕ := 12

-- Probability of rolling a prime number on one die
def prime_probability : ℚ := 5 / 12

-- Probability of rolling a non-prime number on one die
def non_prime_probability : ℚ := 7 / 12

-- Number of ways to choose 3 dice from 6 to show a prime number
def combination (n k : ℕ) : ℕ := n.choose k
def choose_3_out_of_6 : ℕ := combination total_dice 3

-- Combined probability for exactly 3 primes and 3 non-primes
def combined_probability : ℚ :=
  (prime_probability ^ 3) * (non_prime_probability ^ 3)

-- Total probability
def total_probability : ℚ :=
  choose_3_out_of_6 * combined_probability

-- Main theorem statement
theorem probability_of_three_primes_out_of_six :
  total_probability = 857500 / 5177712 :=
by
  sorry

end probability_of_three_primes_out_of_six_l1210_121070


namespace value_of_expression_l1210_121027

-- Conditions
def isOdd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
def isIncreasingOn (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def hasMaxOn (f : ℝ → ℝ) (a b : ℝ) (M : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = M
def hasMinOn (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) := ∃ x, a ≤ x ∧ x ≤ b ∧ f x = m

-- Proof statement
theorem value_of_expression (f : ℝ → ℝ) 
  (hf1 : isOdd f)
  (hf2 : isIncreasingOn f 3 7)
  (hf3 : hasMaxOn f 3 6 8)
  (hf4 : hasMinOn f 3 6 (-1)) :
  2 * f (-6) + f (-3) = -15 :=
sorry

end value_of_expression_l1210_121027


namespace maximum_consecutive_positive_integers_sum_500_l1210_121091

theorem maximum_consecutive_positive_integers_sum_500 : 
  ∃ n : ℕ, (n * (n + 1) / 2 < 500) ∧ (∀ m : ℕ, (m * (m + 1) / 2 < 500) → m ≤ n) :=
sorry

end maximum_consecutive_positive_integers_sum_500_l1210_121091


namespace students_called_back_l1210_121061

theorem students_called_back (girls boys not_called_back called_back : ℕ) 
  (h1 : girls = 17)
  (h2 : boys = 32)
  (h3 : not_called_back = 39)
  (h4 : called_back = (girls + boys) - not_called_back):
  called_back = 10 := by
  sorry

end students_called_back_l1210_121061


namespace area_of_square_land_l1210_121099

-- Define the problem conditions
variable (A P : ℕ)

-- Define the main theorem statement: proving area A given the conditions
theorem area_of_square_land (h₁ : 5 * A = 10 * P + 45) (h₂ : P = 36) : A = 81 := by
  sorry

end area_of_square_land_l1210_121099


namespace describe_T_correctly_l1210_121082

def T (x y : ℝ) : Prop :=
(x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2)

theorem describe_T_correctly :
  (∀ x y : ℝ, T x y ↔
    ((x = 2 ∧ y < 7) ∨ (y = 7 ∧ x < 2) ∨ (y = x + 5 ∧ x > 2))) :=
by
  sorry

end describe_T_correctly_l1210_121082


namespace supplement_complement_diff_l1210_121010

theorem supplement_complement_diff (α : ℝ) : (180 - α) - (90 - α) = 90 := 
by
  sorry

end supplement_complement_diff_l1210_121010


namespace basement_pump_time_l1210_121020

/-- A basement has a 30-foot by 36-foot rectangular floor, flooded to a depth of 24 inches.
Using three pumps, each pumping 10 gallons per minute, and knowing that a cubic foot of water
contains 7.5 gallons, this theorem asserts it will take 540 minutes to pump out all the water. -/
theorem basement_pump_time :
  let length := 30 -- in feet
  let width := 36 -- in feet
  let depth_inch := 24 -- in inches
  let depth := depth_inch / 12 -- converting depth to feet
  let volume_ft3 := length * width * depth -- volume in cubic feet
  let gallons_per_ft3 := 7.5 -- gallons per cubic foot
  let total_gallons := volume_ft3 * gallons_per_ft3 -- total volume in gallons
  let pump_capacity_gpm := 10 -- gallons per minute per pump
  let total_pumps := 3 -- number of pumps
  let total_pump_gpm := pump_capacity_gpm * total_pumps -- total gallons per minute for all pumps
  let pump_time := total_gallons / total_pump_gpm -- time in minutes to pump all the water
  pump_time = 540 := sorry

end basement_pump_time_l1210_121020


namespace find_exponent_M_l1210_121042

theorem find_exponent_M (M : ℕ) : (32^4) * (4^6) = 2^M → M = 32 := by
  sorry

end find_exponent_M_l1210_121042


namespace largest_angle_smallest_angle_middle_angle_l1210_121077

-- Definitions for angles of a triangle in degrees
variable (α β γ : ℝ)
variable (h_sum : α + β + γ = 180)

-- Largest angle condition
theorem largest_angle (h1 : α ≥ β) (h2 : α ≥ γ) : (60 ≤ α ∧ α < 180) :=
  sorry

-- Smallest angle condition
theorem smallest_angle (h1 : α ≤ β) (h2 : α ≤ γ) : (0 < α ∧ α ≤ 60) :=
  sorry

-- Middle angle condition
theorem middle_angle (h1 : α > β ∧ α < γ ∨ α < β ∧ α > γ) : (0 < α ∧ α < 90) :=
  sorry

end largest_angle_smallest_angle_middle_angle_l1210_121077


namespace cylinder_radius_and_volume_l1210_121081

theorem cylinder_radius_and_volume
  (h : ℝ) (surface_area : ℝ) :
  h = 8 ∧ surface_area = 130 * Real.pi →
  ∃ (r : ℝ) (V : ℝ), r = 5 ∧ V = 200 * Real.pi := by
  sorry

end cylinder_radius_and_volume_l1210_121081


namespace sum_series_eq_four_l1210_121009

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n = 0 then 0 else (3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3))

theorem sum_series_eq_four :
  series_sum = 4 :=
by
  sorry

end sum_series_eq_four_l1210_121009


namespace union_complement_equals_set_l1210_121083

universe u

variable {I A B : Set ℕ}

def universal_set : Set ℕ := {0, 1, 2, 3, 4}
def set_A : Set ℕ := {1, 2}
def set_B : Set ℕ := {2, 3, 4}
def complement_B : Set ℕ := { x ∈ universal_set | x ∉ set_B }

theorem union_complement_equals_set :
  set_A ∪ complement_B = {0, 1, 2} := by
  sorry

end union_complement_equals_set_l1210_121083


namespace train_cross_time_proof_l1210_121013

noncomputable def train_cross_time_opposite (L : ℝ) (v1 v2 : ℝ) (t_same : ℝ) : ℝ :=
  let speed_same := (v1 - v2) * (5/18)
  let dist_same := speed_same * t_same
  let speed_opposite := (v1 + v2) * (5/18)
  dist_same / speed_opposite

theorem train_cross_time_proof : 
  train_cross_time_opposite 69.444 50 40 50 = 5.56 :=
by
  sorry

end train_cross_time_proof_l1210_121013


namespace x_squared_minus_y_squared_l1210_121079

theorem x_squared_minus_y_squared
  (x y : ℚ)
  (h1 : x + y = 9 / 16)
  (h2 : x - y = 5 / 16) :
  x^2 - y^2 = 45 / 256 :=
by
  sorry

end x_squared_minus_y_squared_l1210_121079


namespace revenue_fraction_l1210_121066

variable (N D J : ℝ)
variable (h1 : J = 1 / 5 * N)
variable (h2 : D = 4.166666666666666 * (N + J) / 2)

theorem revenue_fraction (h1 : J = 1 / 5 * N) (h2 : D = 4.166666666666666 * (N + J) / 2) : N / D = 2 / 5 :=
by
  sorry

end revenue_fraction_l1210_121066


namespace part_time_employees_l1210_121028

theorem part_time_employees (total_employees : ℕ) (full_time_employees : ℕ) (h1 : total_employees = 65134) (h2 : full_time_employees = 63093) :
  total_employees - full_time_employees = 2041 :=
by
  -- Suppose that total_employees - full_time_employees = 2041
  sorry

end part_time_employees_l1210_121028


namespace mother_to_father_age_ratio_l1210_121085

def DarcieAge : ℕ := 4
def FatherAge : ℕ := 30
def MotherAge : ℕ := DarcieAge * 6

theorem mother_to_father_age_ratio :
  (MotherAge : ℚ) / (FatherAge : ℚ) = (4 / 5) := by
  sorry

end mother_to_father_age_ratio_l1210_121085


namespace greatest_power_of_two_factor_l1210_121071

theorem greatest_power_of_two_factor (n : ℕ) (h : n = 1000) :
  ∃ k, 2^k ∣ 10^n + 4^(n/2) ∧ k = 1003 :=
by {
  sorry
}

end greatest_power_of_two_factor_l1210_121071


namespace solution_set_of_linear_inequalities_l1210_121052

theorem solution_set_of_linear_inequalities (x : ℝ) : (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_of_linear_inequalities_l1210_121052


namespace number_of_handshakes_l1210_121003

-- Define the context of the problem
def total_women := 8
def teams (n : Nat) := 4

-- Define the number of people each woman will shake hands with (excluding her partner)
def handshakes_per_woman := total_women - 2

-- Define the total number of handshakes
def total_handshakes := (total_women * handshakes_per_woman) / 2

-- The theorem that we're to prove
theorem number_of_handshakes : total_handshakes = 24 :=
by
  sorry

end number_of_handshakes_l1210_121003


namespace power_of_power_eq_512_l1210_121096

theorem power_of_power_eq_512 : (2^3)^3 = 512 := by
  sorry

end power_of_power_eq_512_l1210_121096


namespace sam_initial_watermelons_l1210_121055

theorem sam_initial_watermelons (x : ℕ) (h : x + 3 = 7) : x = 4 :=
by
  -- proof steps would go here
  sorry

end sam_initial_watermelons_l1210_121055


namespace envelopes_left_l1210_121006

theorem envelopes_left (initial_envelopes : ℕ) (envelopes_per_friend : ℕ) (number_of_friends : ℕ) (remaining_envelopes : ℕ) :
  initial_envelopes = 37 → envelopes_per_friend = 3 → number_of_friends = 5 → remaining_envelopes = initial_envelopes - (envelopes_per_friend * number_of_friends) → remaining_envelopes = 22 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  norm_num at h4
  exact h4

end envelopes_left_l1210_121006


namespace additional_pots_last_hour_l1210_121065

theorem additional_pots_last_hour (h1 : 60 / 6 = 10) (h2 : 60 / 5 = 12) : 12 - 10 = 2 :=
by
  sorry

end additional_pots_last_hour_l1210_121065


namespace price_per_cup_l1210_121050

theorem price_per_cup
  (num_trees : ℕ)
  (oranges_per_tree_g : ℕ)
  (oranges_per_tree_a : ℕ)
  (oranges_per_tree_m : ℕ)
  (oranges_per_cup : ℕ)
  (total_income : ℕ)
  (h_g : num_trees = 110)
  (h_a : oranges_per_tree_g = 600)
  (h_al : oranges_per_tree_a = 400)
  (h_m : oranges_per_tree_m = 500)
  (h_o : oranges_per_cup = 3)
  (h_income : total_income = 220000) :
  total_income / (((num_trees * oranges_per_tree_g) + (num_trees * oranges_per_tree_a) + (num_trees * oranges_per_tree_m)) / oranges_per_cup) = 4 :=
by
  repeat {sorry}

end price_per_cup_l1210_121050


namespace LeahsCoinsValueIs68_l1210_121051

def LeahsCoinsWorthInCents (p n d : Nat) : Nat :=
  p * 1 + n * 5 + d * 10

theorem LeahsCoinsValueIs68 {p n d : Nat} (h1 : p + n + d = 17) (h2 : n + 2 = p) :
  LeahsCoinsWorthInCents p n d = 68 := by
  sorry

end LeahsCoinsValueIs68_l1210_121051


namespace total_number_of_books_ways_to_select_books_l1210_121068

def first_layer_books : ℕ := 6
def second_layer_books : ℕ := 5
def third_layer_books : ℕ := 4

theorem total_number_of_books : first_layer_books + second_layer_books + third_layer_books = 15 := by
  sorry

theorem ways_to_select_books : first_layer_books * second_layer_books * third_layer_books = 120 := by
  sorry

end total_number_of_books_ways_to_select_books_l1210_121068


namespace factor_expression_l1210_121076

theorem factor_expression (x : ℝ) : 
  (21 * x ^ 4 + 90 * x ^ 3 + 40 * x - 10) - (7 * x ^ 4 + 6 * x ^ 3 + 8 * x - 6) = 
  2 * x * (7 * x ^ 3 + 42 * x ^ 2 + 16) - 4 :=
by sorry

end factor_expression_l1210_121076


namespace sqrt_expression_non_negative_l1210_121080

theorem sqrt_expression_non_negative (x : ℝ) : 4 + 2 * x ≥ 0 ↔ x ≥ -2 :=
by sorry

end sqrt_expression_non_negative_l1210_121080


namespace alice_number_l1210_121060

theorem alice_number (n : ℕ) 
  (h1 : 180 ∣ n) 
  (h2 : 75 ∣ n) 
  (h3 : 900 ≤ n) 
  (h4 : n ≤ 3000) : 
  n = 900 ∨ n = 1800 ∨ n = 2700 := 
by
  sorry

end alice_number_l1210_121060


namespace unique_solution_pair_l1210_121043

theorem unique_solution_pair (x y : ℝ) :
  (4 * x ^ 2 + 6 * x + 4) * (4 * y ^ 2 - 12 * y + 25) = 28 →
  (x, y) = (-3 / 4, 3 / 2) := by
  intro h
  sorry

end unique_solution_pair_l1210_121043


namespace count_multiples_of_12_between_25_and_200_l1210_121002

theorem count_multiples_of_12_between_25_and_200 :
  ∃ n, (∀ i, 25 < i ∧ i < 200 → (∃ k, i = 12 * k)) ↔ n = 14 :=
by
  sorry

end count_multiples_of_12_between_25_and_200_l1210_121002


namespace estevan_initial_blankets_l1210_121008

theorem estevan_initial_blankets (B : ℕ) 
  (polka_dot_initial : ℕ) 
  (polka_dot_total : ℕ) 
  (h1 : (1 / 3 : ℚ) * B = polka_dot_initial) 
  (h2 : polka_dot_initial + 2 = polka_dot_total) 
  (h3 : polka_dot_total = 10) : 
  B = 24 := 
by 
  sorry

end estevan_initial_blankets_l1210_121008


namespace person_A_arrives_before_B_l1210_121036

variable {a b S : ℝ}

theorem person_A_arrives_before_B (h : a ≠ b) (a_pos : 0 < a) (b_pos : 0 < b) (S_pos : 0 < S) :
  (2 * S / (a + b)) < ((a + b) * S / (2 * a * b)) :=
by
  sorry

end person_A_arrives_before_B_l1210_121036


namespace binomial_coeff_arithmetic_seq_l1210_121022

theorem binomial_coeff_arithmetic_seq (n : ℕ) (x : ℝ) (h : ∀ (a b c : ℝ), a = 1 ∧ b = n/2 ∧ c = n*(n-1)/8 → (b - a) = (c - b)) : n = 8 :=
sorry

end binomial_coeff_arithmetic_seq_l1210_121022


namespace measure_of_angle_C_maximum_area_of_triangle_l1210_121094

noncomputable def triangle (A B C a b c : ℝ) : Prop :=
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C ∧
  2 * (Real.sin A ^ 2 - Real.sin C ^ 2) = (Real.sqrt 2 * a - b) * Real.sin B

theorem measure_of_angle_C :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 :=
by
  intros A B C a b c h
  sorry

theorem maximum_area_of_triangle :
  ∀ (A B C a b c : ℝ),
  triangle A B C a b c →
  C = π / 4 →
  1 / 2 * a * b * Real.sin C = (Real.sqrt 2 / 2 + 1 / 2) :=
by
  intros A B C a b c h hC
  sorry

end measure_of_angle_C_maximum_area_of_triangle_l1210_121094


namespace range_of_k_l1210_121073

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x = 4 ∧ x < 2) → (k < 1 ∨ k > 3) := 
by 
  sorry

end range_of_k_l1210_121073


namespace klinker_twice_as_old_l1210_121053

theorem klinker_twice_as_old :
  ∃ x : ℕ, (∀ (m k d : ℕ), m = 35 → d = 10 → m + x = 2 * (d + x)) → x = 15 :=
by
  sorry

end klinker_twice_as_old_l1210_121053


namespace no_real_solution_implies_a_range_l1210_121092

noncomputable def quadratic (a x : ℝ) : ℝ := x^2 - 4 * x + a^2

theorem no_real_solution_implies_a_range (a : ℝ) :
  (∀ x : ℝ, quadratic a x ≤ 0 → false) ↔ a < -2 ∨ a > 2 := 
sorry

end no_real_solution_implies_a_range_l1210_121092


namespace find_difference_l1210_121064

theorem find_difference (x y : ℚ) (h₁ : x + y = 520) (h₂ : x / y = 3 / 4) : y - x = 520 / 7 :=
by
  sorry

end find_difference_l1210_121064


namespace shortest_distance_to_left_focus_l1210_121023

def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 16 = 1

def left_focus : ℝ × ℝ := (-5, 0)

theorem shortest_distance_to_left_focus : 
  ∃ P : ℝ × ℝ, 
  hyperbola P.1 P.2 ∧ 
  (∀ Q : ℝ × ℝ, hyperbola Q.1 Q.2 → dist Q left_focus ≥ dist P left_focus) ∧ 
  dist P left_focus = 2 :=
sorry

end shortest_distance_to_left_focus_l1210_121023


namespace gravel_cost_l1210_121093

-- Definitions of conditions
def lawn_length : ℝ := 70
def lawn_breadth : ℝ := 30
def road_width : ℝ := 5
def gravel_cost_per_sqm : ℝ := 4

-- Theorem statement
theorem gravel_cost : (lawn_length * road_width + lawn_breadth * road_width - road_width * road_width) * gravel_cost_per_sqm = 1900 :=
by
  -- Definitions used in the problem
  let area_first_road := lawn_length * road_width
  let area_second_road := lawn_breadth * road_width
  let area_intersection := road_width * road_width

  -- Total area to be graveled
  let total_area_to_be_graveled := area_first_road + area_second_road - area_intersection

  -- Calculate the cost
  let cost := total_area_to_be_graveled * gravel_cost_per_sqm

  show cost = 1900
  sorry

end gravel_cost_l1210_121093


namespace derivative_at_one_max_value_l1210_121072

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x

-- Prove that f'(1) = 0
theorem derivative_at_one : deriv f 1 = 0 :=
by sorry

-- Prove that the maximum value of f(x) is 2
theorem max_value : ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = 2 :=
by sorry

end derivative_at_one_max_value_l1210_121072


namespace students_in_diligence_before_transfer_l1210_121001

theorem students_in_diligence_before_transfer (D I : ℕ) 
  (h1 : D + 2 = I - 2) 
  (h2 : D + I = 50) : 
  D = 23 := 
by
  sorry

end students_in_diligence_before_transfer_l1210_121001


namespace parallelogram_area_l1210_121033

variable (d : ℕ) (h : ℕ)

theorem parallelogram_area (h_d : d = 30) (h_h : h = 20) : 
  ∃ a : ℕ, a = 600 := 
by
  sorry

end parallelogram_area_l1210_121033


namespace find_x_l1210_121045

theorem find_x (x : ℝ) (h₁ : x > 0) (h₂ : x^4 = 390625) : x = 25 := 
by sorry

end find_x_l1210_121045


namespace calculate_exponent_product_l1210_121058

theorem calculate_exponent_product : (2^2021) * (-1/2)^2022 = (1/2) :=
by
  sorry

end calculate_exponent_product_l1210_121058


namespace find_m_l1210_121054

-- Define the vectors a and b and the condition for parallelicity
def a : ℝ × ℝ := (2, 1)
def b (m : ℝ) : ℝ × ℝ := (m, 2)
def parallel (u v : ℝ × ℝ) := u.1 * v.2 = u.2 * v.1

-- State the theorem with the given conditions and required proof goal
theorem find_m (m : ℝ) (h : parallel a (b m)) : m = 4 :=
by sorry  -- skipping proof

end find_m_l1210_121054
