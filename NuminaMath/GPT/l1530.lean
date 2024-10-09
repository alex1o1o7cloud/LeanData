import Mathlib

namespace steve_pencils_left_l1530_153061

def initial_pencils : ℕ := 2 * 12
def pencils_given_to_lauren : ℕ := 6
def pencils_given_to_matt : ℕ := pencils_given_to_lauren + 3

def pencils_left (initial_pencils given_lauren given_matt : ℕ) : ℕ :=
  initial_pencils - given_lauren - given_matt

theorem steve_pencils_left : pencils_left initial_pencils pencils_given_to_lauren pencils_given_to_matt = 9 := by
  sorry

end steve_pencils_left_l1530_153061


namespace problem1_problem2_l1530_153034

-- Problem (1)
theorem problem1 : (Real.sqrt 12 + (-1 / 3)⁻¹ + (-2)^2 = 2 * Real.sqrt 3 + 1) :=
  sorry

-- Problem (2)
theorem problem2 (a : Real) (h : a ≠ 2) :
  (2 * a / (a^2 - 4) / (1 + (a - 2) / (a + 2)) = 1 / (a - 2)) :=
  sorry

end problem1_problem2_l1530_153034


namespace greatest_monthly_drop_is_march_l1530_153082

-- Define the price changes for each month
def price_change_january : ℝ := -0.75
def price_change_february : ℝ := 1.50
def price_change_march : ℝ := -3.00
def price_change_april : ℝ := 2.50
def price_change_may : ℝ := -1.00
def price_change_june : ℝ := 0.50
def price_change_july : ℝ := -2.50

-- Prove that the month with the greatest drop in price is March
theorem greatest_monthly_drop_is_march :
  (price_change_march = -3.00) →
  (∀ m, m ≠ price_change_march → m ≥ price_change_march) :=
by
  intros h1 h2
  sorry

end greatest_monthly_drop_is_march_l1530_153082


namespace kennedy_lost_pawns_l1530_153052

-- Definitions based on conditions
def initial_pawns_per_player := 8
def total_pawns := 2 * initial_pawns_per_player -- Total pawns in the game initially
def pawns_lost_by_Riley := 1 -- Riley lost 1 pawn
def pawns_remaining := 11 -- 11 pawns left in the game

-- Translations of conditions to Lean
theorem kennedy_lost_pawns : 
  initial_pawns_per_player - (pawns_remaining - (initial_pawns_per_player - pawns_lost_by_Riley)) = 4 := 
by 
  sorry

end kennedy_lost_pawns_l1530_153052


namespace units_digit_n_l1530_153006

theorem units_digit_n (m n : ℕ) (h1 : m * n = 31^8) (h2 : m % 10 = 7) : n % 10 = 3 := 
sorry

end units_digit_n_l1530_153006


namespace cost_price_of_article_l1530_153056

-- Define the conditions and goal as a Lean 4 statement
theorem cost_price_of_article (M C : ℝ) (h1 : 0.95 * M = 75) (h2 : 1.25 * C = 75) : 
  C = 60 := 
by 
  sorry

end cost_price_of_article_l1530_153056


namespace weeks_to_save_l1530_153067

-- Define the conditions as given in the problem
def cost_of_bike : ℕ := 600
def gift_from_parents : ℕ := 60
def gift_from_uncle : ℕ := 40
def gift_from_sister : ℕ := 20
def gift_from_friend : ℕ := 30
def weekly_earnings : ℕ := 18

-- Total gift money
def total_gift_money : ℕ := gift_from_parents + gift_from_uncle + gift_from_sister + gift_from_friend

-- Total money after x weeks
def total_money_after_weeks (x : ℕ) : ℕ := total_gift_money + weekly_earnings * x

-- Main theorem statement
theorem weeks_to_save (x : ℕ) : total_money_after_weeks x = cost_of_bike → x = 25 := by
  sorry

end weeks_to_save_l1530_153067


namespace math_problem_l1530_153024

theorem math_problem (x : ℝ) (h : x + (1 / x) = 5) : x^2 + (1 / x^2) = 23 :=
sorry

end math_problem_l1530_153024


namespace twenty_eight_is_seventy_percent_of_what_number_l1530_153063

theorem twenty_eight_is_seventy_percent_of_what_number (x : ℝ) (h : 28 / x = 70 / 100) : x = 40 :=
by
  sorry

end twenty_eight_is_seventy_percent_of_what_number_l1530_153063


namespace find_building_block_width_l1530_153080

noncomputable def building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40)
: ℕ :=
(8 * 10 * 12) / 40 / (3 * 4)

theorem find_building_block_width
  (box_height box_width box_length building_block_height building_block_length : ℕ)
  (num_building_blocks : ℕ)
  (box_height_eq : box_height = 8)
  (box_width_eq : box_width = 10)
  (box_length_eq : box_length = 12)
  (building_block_height_eq : building_block_height = 3)
  (building_block_length_eq : building_block_length = 4)
  (num_building_blocks_eq : num_building_blocks = 40) :
  building_block_width box_height box_width box_length building_block_height building_block_length num_building_blocks box_height_eq box_width_eq box_length_eq building_block_height_eq building_block_length_eq num_building_blocks_eq = 2 := 
sorry

end find_building_block_width_l1530_153080


namespace sum_of_digits_inequality_l1530_153008

def sum_of_digits (n : ℕ) : ℕ := -- Definition of the sum of digits function
  -- This should be defined, for demonstration we use a placeholder
  sorry

theorem sum_of_digits_inequality (n : ℕ) (h : n > 0) :
  sum_of_digits n ≤ 8 * sum_of_digits (8 * n) :=
sorry

end sum_of_digits_inequality_l1530_153008


namespace ratio_copper_zinc_l1530_153053

theorem ratio_copper_zinc (total_mass zinc_mass : ℕ) (h1 : total_mass = 100) (h2 : zinc_mass = 35) : 
  ∃ (copper_mass : ℕ), 
    copper_mass = total_mass - zinc_mass ∧ (copper_mass / 5, zinc_mass / 5) = (13, 7) :=
by {
  sorry
}

end ratio_copper_zinc_l1530_153053


namespace vitya_catches_up_in_5_minutes_l1530_153015

noncomputable def catch_up_time (s : ℝ) : ℝ :=
  let initial_distance := 20 * s
  let vitya_speed := 5 * s
  let mom_speed := s
  let relative_speed := vitya_speed - mom_speed
  initial_distance / relative_speed

theorem vitya_catches_up_in_5_minutes (s : ℝ) (h : s > 0) :
  catch_up_time s = 5 :=
by
  -- Proof is here.
  sorry

end vitya_catches_up_in_5_minutes_l1530_153015


namespace trapezoid_lower_side_length_l1530_153049

variable (U L : ℝ) (height area : ℝ)

theorem trapezoid_lower_side_length
  (h1 : L = U - 3.4)
  (h2 : height = 5.2)
  (h3 : area = 100.62)
  (h4 : area = (1 / 2) * (U + L) * height) :
  L = 17.65 :=
by
  sorry

end trapezoid_lower_side_length_l1530_153049


namespace total_distance_crawled_l1530_153057

theorem total_distance_crawled :
  let pos1 := 3
  let pos2 := -5
  let pos3 := 8
  let pos4 := 0
  abs (pos2 - pos1) + abs (pos3 - pos2) + abs (pos4 - pos3) = 29 :=
by
  sorry

end total_distance_crawled_l1530_153057


namespace quadrilateral_inequality_l1530_153050

theorem quadrilateral_inequality
  (AB AC BD CD: ℝ)
  (h1 : AB + BD ≤ AC + CD)
  (h2 : AB + CD < AC + BD) :
  AB < AC := by
  sorry

end quadrilateral_inequality_l1530_153050


namespace taller_cycle_shadow_length_l1530_153035

theorem taller_cycle_shadow_length 
  (h_taller : ℝ) (h_shorter : ℝ) (shadow_shorter : ℝ) (shadow_taller : ℝ) 
  (h_taller_val : h_taller = 2.5) 
  (h_shorter_val : h_shorter = 2) 
  (shadow_shorter_val : shadow_shorter = 4)
  (similar_triangles : h_taller / shadow_taller = h_shorter / shadow_shorter) :
  shadow_taller = 5 := 
by 
  sorry

end taller_cycle_shadow_length_l1530_153035


namespace table_tennis_matches_l1530_153085

def num_players : ℕ := 8

def total_matches (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem table_tennis_matches : total_matches num_players = 28 := by
  sorry

end table_tennis_matches_l1530_153085


namespace net_marble_change_l1530_153027

/-- Josh's initial number of marbles. -/
def initial_marbles : ℕ := 20

/-- Number of marbles Josh lost. -/
def lost_marbles : ℕ := 16

/-- Number of marbles Josh found. -/
def found_marbles : ℕ := 8

/-- Number of marbles Josh traded away. -/
def traded_away_marbles : ℕ := 5

/-- Number of marbles Josh received in a trade. -/
def received_in_trade_marbles : ℕ := 9

/-- Number of marbles Josh gave away. -/
def gave_away_marbles : ℕ := 3

/-- Number of marbles Josh received from his cousin. -/
def received_from_cousin_marbles : ℕ := 4

/-- Final number of marbles Josh has after all transactions. -/
def final_marbles : ℕ :=
  initial_marbles - lost_marbles + found_marbles - traded_away_marbles + received_in_trade_marbles
  - gave_away_marbles + received_from_cousin_marbles

theorem net_marble_change : (final_marbles : ℤ) - (initial_marbles : ℤ) = -3 := 
by
  sorry

end net_marble_change_l1530_153027


namespace right_rectangular_prism_volume_l1530_153030

theorem right_rectangular_prism_volume (x y z : ℝ) 
  (h1 : x * y = 72) (h2 : y * z = 75) (h3 : x * z = 80) : 
  x * y * z = 657 :=
sorry

end right_rectangular_prism_volume_l1530_153030


namespace railway_original_stations_l1530_153011

theorem railway_original_stations (m n : ℕ) (hn : n > 1) (h : n * (2 * m - 1 + n) = 58) : m = 14 :=
by
  sorry

end railway_original_stations_l1530_153011


namespace mark_deposit_amount_l1530_153065

-- Define the conditions
def bryans_deposit (M : ℝ) : ℝ := 5 * M - 40
def total_deposit (M : ℝ) : ℝ := M + bryans_deposit M

-- State the theorem
theorem mark_deposit_amount (M : ℝ) (h1: total_deposit M = 400) : M = 73.33 :=
by
  sorry

end mark_deposit_amount_l1530_153065


namespace evaluate_expression_l1530_153096

theorem evaluate_expression : 8^3 + 4 * 8^2 + 6 * 8 + 3 = 1000 := by
  sorry

end evaluate_expression_l1530_153096


namespace valid_interval_for_k_l1530_153003

theorem valid_interval_for_k :
  ∀ k : ℝ, (∀ x : ℝ, x^2 - 8*x + k < 0 → 0 < k ∧ k < 16) :=
by
  sorry

end valid_interval_for_k_l1530_153003


namespace sequence_periodic_l1530_153060

def sequence_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 2) = |a (n + 1)| - a n

theorem sequence_periodic (a : ℕ → ℝ) (m_0 : ℕ) (h : sequence_condition a) :
  ∀ m ≥ m_0, a (m + 9) = a m := 
sorry

end sequence_periodic_l1530_153060


namespace number_subtracted_from_15n_l1530_153026

theorem number_subtracted_from_15n (m n : ℕ) (h_pos_n : 0 < n) (h_pos_m : 0 < m) (h_eq : m = 15 * n - 1) (h_remainder : m % 5 = 4) : 1 = 1 :=
by
  sorry

end number_subtracted_from_15n_l1530_153026


namespace solve_for_a_l1530_153023

noncomputable def parabola (a b c : ℚ) (x : ℚ) := a * x^2 + b * x + c

theorem solve_for_a (a b c : ℚ) (h1 : parabola a b c 2 = 5) (h2 : parabola a b c 1 = 2) : 
  a = -3 :=
by
  -- Given: y = ax^2 + bx + c with vertex (2,5) and point (1,2)
  have eq1 : a * (2:ℚ)^2 + b * (2:ℚ) + c = 5 := h1
  have eq2 : a * (1:ℚ)^2 + b * (1:ℚ) + c = 2 := h2

  -- Combine information to find a
  sorry

end solve_for_a_l1530_153023


namespace trapezoid_AD_BC_ratio_l1530_153055

variables {A B C D M N K : Type} {AD BC CM MD NA CN : ℝ}

-- Definition of the trapezoid and the ratio conditions
def is_trapezoid (A B C D : Type) : Prop := sorry -- Assume existence of a trapezoid for lean to accept the statement
def ratio_CM_MD (CM MD : ℝ) : Prop := CM / MD = 4 / 3
def ratio_NA_CN (NA CN : ℝ) : Prop := NA / CN = 4 / 3

-- Proof statement for the given problem
theorem trapezoid_AD_BC_ratio 
  (h_trapezoid: is_trapezoid A B C D)
  (h_CM_MD: ratio_CM_MD CM MD)
  (h_NA_CN: ratio_NA_CN NA CN) :
  AD / BC = 7 / 12 :=
sorry

end trapezoid_AD_BC_ratio_l1530_153055


namespace tire_mileage_problem_l1530_153005

/- Definitions -/
def total_miles : ℕ := 45000
def enhancement_ratio : ℚ := 1.2
def total_tire_miles : ℚ := 180000

/- Question as theorem -/
theorem tire_mileage_problem
  (x y : ℚ)
  (h1 : y = enhancement_ratio * x)
  (h2 : 4 * x + y = total_tire_miles) :
  (x = 34615 ∧ y = 41538) :=
sorry

end tire_mileage_problem_l1530_153005


namespace problem_solution_set_l1530_153040

variable {a b c : ℝ}

theorem problem_solution_set (h_condition : ∀ x, 1 ≤ x → x ≤ 2 → a * x^2 - b * x + c ≥ 0) : 
  { x : ℝ | c * x^2 + b * x + a ≤ 0 } = { x : ℝ | x ≤ -1 } ∪ { x | -1/2 ≤ x } :=
by 
  sorry

end problem_solution_set_l1530_153040


namespace stationery_difference_l1530_153017

theorem stationery_difference :
  let georgia := 25
  let lorene := 3 * georgia
  lorene - georgia = 50 :=
by
  let georgia := 25
  let lorene := 3 * georgia
  show lorene - georgia = 50
  sorry

end stationery_difference_l1530_153017


namespace tank_capacity_l1530_153078

variable (c : ℕ) -- Total capacity of the tank in liters.
variable (w_0 : ℕ := c / 3) -- Initial volume of water in the tank in liters.

theorem tank_capacity (h1 : w_0 = c / 3) (h2 : (w_0 + 5) / c = 2 / 5) : c = 75 :=
by
  -- Proof steps would be here.
  sorry

end tank_capacity_l1530_153078


namespace Jerome_money_left_l1530_153037

-- Definitions based on conditions
def J_half := 43              -- Half of Jerome's money
def to_Meg := 8               -- Amount Jerome gave to Meg
def to_Bianca := to_Meg * 3   -- Amount Jerome gave to Bianca

-- Total initial amount of Jerome's money
def J_initial : ℕ := J_half * 2

-- Amount left after giving money to Meg
def after_Meg : ℕ := J_initial - to_Meg

-- Amount left after giving money to Bianca
def after_Bianca : ℕ := after_Meg - to_Bianca

-- Statement to be proved
theorem Jerome_money_left : after_Bianca = 54 :=
by
  sorry

end Jerome_money_left_l1530_153037


namespace negation_of_proposition_l1530_153013

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 = 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≠ 0) :=
by sorry

end negation_of_proposition_l1530_153013


namespace number_of_nickels_is_3_l1530_153019

-- Defining the problem conditions
def total_coins := 8
def total_value := 53 -- in cents
def at_least_one_penny := 1
def at_least_one_nickel := 1
def at_least_one_dime := 1

-- Stating the proof problem
theorem number_of_nickels_is_3 : ∃ (pennies nickels dimes : Nat), 
  pennies + nickels + dimes = total_coins ∧ 
  pennies ≥ at_least_one_penny ∧ 
  nickels ≥ at_least_one_nickel ∧ 
  dimes ≥ at_least_one_dime ∧ 
  pennies + 5 * nickels + 10 * dimes = total_value ∧ 
  nickels = 3 := sorry

end number_of_nickels_is_3_l1530_153019


namespace least_consecutive_odd_integers_l1530_153007

theorem least_consecutive_odd_integers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = 8 * 414)) :
  x = 407 :=
by
  sorry

end least_consecutive_odd_integers_l1530_153007


namespace option_C_is_quadratic_l1530_153001

-- Definitions based on conditions
def option_A (x : ℝ) : Prop := x^2 + (1/x^2) = 0
def option_B (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def option_C (x : ℝ) : Prop := (x - 1) * (x + 2) = 1
def option_D (x y : ℝ) : Prop := 3 * x^2 - 2 * x * y - 5 * y^2 = 0

-- Statement to prove option C is a quadratic equation in one variable.
theorem option_C_is_quadratic : ∀ x : ℝ, (option_C x) → (∃ a b c : ℝ, a ≠ 0 ∧ a * x^2 + b * x + c = 0) :=
by
  intros x hx
  -- To be proven
  sorry

end option_C_is_quadratic_l1530_153001


namespace A_doubles_after_6_months_l1530_153070

variable (x : ℕ)

def A_investment_share (x : ℕ) := (3000 * x) + (6000 * (12 - x))
def B_investment_share := 4500 * 12

theorem A_doubles_after_6_months (h : A_investment_share x = B_investment_share) : x = 6 :=
by
  sorry

end A_doubles_after_6_months_l1530_153070


namespace alice_bob_not_both_l1530_153039

-- Define the group of 8 students
def total_students : ℕ := 8

-- Define the committee size
def committee_size : ℕ := 5

-- Calculate the total number of unrestricted committees
def total_committees : ℕ := Nat.choose total_students committee_size

-- Calculate the number of committees where both Alice and Bob are included
def alice_bob_committees : ℕ := Nat.choose (total_students - 2) (committee_size - 2)

-- Calculate the number of committees where Alice and Bob are not both included
def not_both_alice_bob : ℕ := total_committees - alice_bob_committees

-- Now state the theorem we want to prove
theorem alice_bob_not_both : not_both_alice_bob = 36 :=
by
  sorry

end alice_bob_not_both_l1530_153039


namespace fifth_term_is_67_l1530_153059

noncomputable def satisfies_sequence (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) (e : ℕ) :=
  (a = 3) ∧ (d = 27) ∧ 
  (a = (1/3 : ℚ) * (3 + b)) ∧
  (b = (1/3 : ℚ) * (a + 27)) ∧
  (27 = (1/3 : ℚ) * (b + e))

theorem fifth_term_is_67 :
  ∃ (e : ℕ), satisfies_sequence 3 a b 27 e ∧ e = 67 :=
sorry

end fifth_term_is_67_l1530_153059


namespace avg_waiting_time_l1530_153000

theorem avg_waiting_time : 
  let P_G := 1 / 3      -- Probability of green light
  let P_red := 2 / 3    -- Probability of red light
  let E_T_given_G := 0  -- Expected time given green light
  let E_T_given_red := 1 -- Expected time given red light
  (E_T_given_G * P_G) + (E_T_given_red * P_red) = 2 / 3
:= by
  sorry

end avg_waiting_time_l1530_153000


namespace minimum_at_neg_one_l1530_153041

noncomputable def f (x : Real) : Real := x * Real.exp x

theorem minimum_at_neg_one : 
  ∃ c : Real, c = -1 ∧ ∀ x : Real, f c ≤ f x := sorry

end minimum_at_neg_one_l1530_153041


namespace difference_between_m_and_n_l1530_153032

theorem difference_between_m_and_n (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : 10 * 2^m = 2^n + 2^(n + 2)) :
  n - m = 1 :=
sorry

end difference_between_m_and_n_l1530_153032


namespace ratio_of_crates_l1530_153028

/-
  Gabrielle sells eggs. On Monday she sells 5 crates of eggs. On Tuesday she sells 2 times as many
  crates of eggs as Monday. On Wednesday she sells 2 fewer crates than Tuesday. On Thursday she sells
  some crates of eggs. She sells a total of 28 crates of eggs for the 4 days. Prove the ratio of the 
  number of crates she sells on Thursday to the number she sells on Tuesday is 1/2.
-/

theorem ratio_of_crates 
    (mon_crates : ℕ) 
    (tue_crates : ℕ) 
    (wed_crates : ℕ) 
    (thu_crates : ℕ) 
    (total_crates : ℕ) 
    (h_mon : mon_crates = 5) 
    (h_tue : tue_crates = 2 * mon_crates) 
    (h_wed : wed_crates = tue_crates - 2) 
    (h_total : total_crates = mon_crates + tue_crates + wed_crates + thu_crates) 
    (h_total_val : total_crates = 28): 
  (thu_crates / tue_crates : ℚ) = 1 / 2 := 
by 
  sorry

end ratio_of_crates_l1530_153028


namespace intersection_M_N_l1530_153033

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x^2 - 3*x ≤ 0}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l1530_153033


namespace fraction_increases_by_3_l1530_153073

-- Define initial fraction
def initial_fraction (x y : ℕ) : ℕ :=
  2 * x * y / (3 * x - y)

-- Define modified fraction
def modified_fraction (x y : ℕ) (m : ℕ) : ℕ :=
  2 * (m * x) * (m * y) / (m * (3 * x) - (m * y))

-- State the theorem to prove the value of modified fraction is 3 times the initial fraction
theorem fraction_increases_by_3 (x y : ℕ) : modified_fraction x y 3 = 3 * initial_fraction x y :=
by sorry

end fraction_increases_by_3_l1530_153073


namespace charge_difference_l1530_153002

theorem charge_difference (cost_x cost_y : ℝ) (num_copies : ℕ) (hx : cost_x = 1.25) (hy : cost_y = 2.75) (hn : num_copies = 40) : 
  num_copies * cost_y - num_copies * cost_x = 60 := by
  sorry

end charge_difference_l1530_153002


namespace rod_length_is_38_point_25_l1530_153094

noncomputable def length_of_rod (n : ℕ) (l : ℕ) (conversion_factor : ℕ) : ℝ :=
  (n * l : ℝ) / conversion_factor

theorem rod_length_is_38_point_25 :
  length_of_rod 45 85 100 = 38.25 :=
by
  sorry

end rod_length_is_38_point_25_l1530_153094


namespace smallest_fourth_number_l1530_153076

theorem smallest_fourth_number :
  ∃ (a b : ℕ), 145 + 10 * a + b = 4 * (28 + a + b) ∧ 10 * a + b = 35 :=
by
  sorry

end smallest_fourth_number_l1530_153076


namespace percentage_of_music_students_l1530_153066

theorem percentage_of_music_students 
  (total_students : ℕ) 
  (dance_students : ℕ) 
  (art_students : ℕ) 
  (drama_students : ℕ)
  (h_total : total_students = 2000) 
  (h_dance : dance_students = 450) 
  (h_art : art_students = 680) 
  (h_drama : drama_students = 370) 
  : (total_students - (dance_students + art_students + drama_students)) / total_students * 100 = 25 
:= by 
  sorry

end percentage_of_music_students_l1530_153066


namespace photos_in_gallery_l1530_153046

theorem photos_in_gallery (P : ℕ) 
  (h1 : P / 2 + (P / 2 + 120) + P = 920) : P = 400 :=
by
  sorry

end photos_in_gallery_l1530_153046


namespace min_moves_to_emit_all_colors_l1530_153058

theorem min_moves_to_emit_all_colors :
  ∀ (colors : Fin 7 → Prop) (room : Fin 4 → Fin 7)
  (h : ∀ i j, i ≠ j → room i ≠ room j) (moves : ℕ),
  (∀ (n : ℕ) (i : Fin 4), n < moves → ∃ c : Fin 7, colors c ∧ room i = c ∧
    (∀ j, j ≠ i → room j ≠ c)) →
  (∃ n, n = 8) :=
by
  sorry

end min_moves_to_emit_all_colors_l1530_153058


namespace emily_chairs_count_l1530_153054

theorem emily_chairs_count 
  (C : ℕ) 
  (T : ℕ) 
  (time_per_furniture : ℕ)
  (total_time : ℕ) 
  (hT : T = 2) 
  (h_time : time_per_furniture = 8) 
  (h_total : 8 * C + 8 * T = 48) : 
  C = 4 := by
    sorry

end emily_chairs_count_l1530_153054


namespace mr_bodhi_adds_twenty_sheep_l1530_153036

def cows : ℕ := 20
def foxes : ℕ := 15
def zebras : ℕ := 3 * foxes
def required_total : ℕ := 100

def sheep := required_total - (cows + foxes + zebras)

theorem mr_bodhi_adds_twenty_sheep : sheep = 20 :=
by
  -- Proof for the theorem is not required and is thus replaced with sorry.
  sorry

end mr_bodhi_adds_twenty_sheep_l1530_153036


namespace major_axis_length_l1530_153029

theorem major_axis_length {r : ℝ} (h_r : r = 1) (h_major : ∃ (minor_axis : ℝ), minor_axis = 2 * r ∧ 1.5 * minor_axis = major_axis) : major_axis = 3 :=
by
  sorry

end major_axis_length_l1530_153029


namespace line_intersects_circle_l1530_153042

-- Definitions
def radius : ℝ := 5
def distance_to_center : ℝ := 3

-- Theorem statement
theorem line_intersects_circle (r : ℝ) (d : ℝ) (h_r : r = radius) (h_d : d = distance_to_center) : d < r :=
by
  rw [h_r, h_d]
  exact sorry

end line_intersects_circle_l1530_153042


namespace smallest_value_of_Q_l1530_153084

def Q (x : ℝ) : ℝ := x^4 - 2*x^3 + 3*x^2 - 4*x + 5

noncomputable def A := Q (-1)
noncomputable def B := Q (0)
noncomputable def C := (2 : ℝ)^2
def D := 1 - 2 + 3 - 4 + 5
def E := 2 -- assuming all zeros are real

theorem smallest_value_of_Q :
  min (min (min (min A B) C) D) E = 2 :=
by sorry

end smallest_value_of_Q_l1530_153084


namespace angle_BCM_in_pentagon_l1530_153044

-- Definitions of the conditions
structure Pentagon (A B C D E : Type) :=
  (is_regular : ∀ (x y : Type), ∃ (angle : ℝ), angle = 108)

structure EquilateralTriangle (A B M : Type) :=
  (is_equilateral : ∀ (x y : Type), ∃ (angle : ℝ), angle = 60)

-- Problem statement
theorem angle_BCM_in_pentagon (A B C D E M : Type) (P : Pentagon A B C D E) (T : EquilateralTriangle A B M) :
  ∃ (angle : ℝ), angle = 66 :=
by
  sorry

end angle_BCM_in_pentagon_l1530_153044


namespace solve_inequality_l1530_153031

theorem solve_inequality :
  {x : ℝ | (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2)) < 1 / 2)} =
  {x : ℝ | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)} :=
by
  sorry

end solve_inequality_l1530_153031


namespace cheenu_time_difference_l1530_153081

theorem cheenu_time_difference :
  let boy_distance : ℝ := 18
  let boy_time_hours : ℝ := 4
  let old_man_distance : ℝ := 12
  let old_man_time_hours : ℝ := 5
  let hour_to_minute : ℝ := 60
  
  let boy_time_minutes := boy_time_hours * hour_to_minute
  let old_man_time_minutes := old_man_time_hours * hour_to_minute

  let boy_time_per_mile := boy_time_minutes / boy_distance
  let old_man_time_per_mile := old_man_time_minutes / old_man_distance
  
  old_man_time_per_mile - boy_time_per_mile = 12 :=
by sorry

end cheenu_time_difference_l1530_153081


namespace class_8_1_total_score_l1530_153095

noncomputable def total_score (spirit neatness standard_of_movements : ℝ) 
(weights_spirit weights_neatness weights_standard : ℝ) : ℝ :=
  (spirit * weights_spirit + neatness * weights_neatness + standard_of_movements * weights_standard) / 
  (weights_spirit + weights_neatness + weights_standard)

theorem class_8_1_total_score :
  total_score 8 9 10 2 3 5 = 9.3 :=
by
  sorry

end class_8_1_total_score_l1530_153095


namespace power_of_i_l1530_153098

theorem power_of_i (i : ℂ) 
  (h1: i^1 = i) 
  (h2: i^2 = -1) 
  (h3: i^3 = -i) 
  (h4: i^4 = 1)
  (h5: i^5 = i) 
  : i^2016 = 1 :=
by {
  sorry
}

end power_of_i_l1530_153098


namespace coffee_cost_per_week_l1530_153077

def num_people: ℕ := 4
def cups_per_person_per_day: ℕ := 2
def ounces_per_cup: ℝ := 0.5
def cost_per_ounce: ℝ := 1.25

theorem coffee_cost_per_week : 
  (num_people * cups_per_person_per_day * ounces_per_cup * 7 * cost_per_ounce) = 35 :=
by
  sorry

end coffee_cost_per_week_l1530_153077


namespace add_three_to_both_sides_l1530_153086

variable {a b : ℝ}

theorem add_three_to_both_sides (h : a < b) : 3 + a < 3 + b :=
by
  sorry

end add_three_to_both_sides_l1530_153086


namespace find_blue_balls_l1530_153090

theorem find_blue_balls 
  (B : ℕ)
  (red_balls : ℕ := 7)
  (green_balls : ℕ := 4)
  (prob_red_red : ℚ := 7 / 40) -- 0.175 represented as a rational number
  (h : (21 / ((11 + B) * (10 + B) / 2 : ℚ)) = prob_red_red) :
  B = 5 :=
sorry

end find_blue_balls_l1530_153090


namespace gcd_triples_l1530_153075

theorem gcd_triples (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  gcd a 20 = b ∧ gcd b 15 = c ∧ gcd a c = 5 ↔
  ∃ t : ℕ, t > 0 ∧ 
    ((a = 20 * t ∧ b = 20 ∧ c = 5) ∨ 
     (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨ 
     (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by
  sorry

end gcd_triples_l1530_153075


namespace max_chord_length_of_parabola_l1530_153091

-- Definitions based on the problem conditions
def parabola (x y : ℝ) : Prop := x^2 = 8 * y
def y_midpoint_condition (y1 y2 : ℝ) : Prop := (y1 + y2) / 2 = 4

-- The theorem to prove that the maximum length of the chord AB is 12
theorem max_chord_length_of_parabola (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h_mid : y_midpoint_condition y1 y2) : 
  abs ((y1 + y2) + 2 * 2) = 12 :=
sorry

end max_chord_length_of_parabola_l1530_153091


namespace hexagon_perimeter_of_intersecting_triangles_l1530_153045

/-- Given two equilateral triangles with parallel sides, where the perimeter of the blue triangle 
    is 4 and the perimeter of the green triangle is 5, prove that the perimeter of the hexagon 
    formed by their intersection is 3. -/
theorem hexagon_perimeter_of_intersecting_triangles 
    (P_blue P_green P_hexagon : ℝ)
    (h_blue : P_blue = 4)
    (h_green : P_green = 5) :
    P_hexagon = 3 := 
sorry

end hexagon_perimeter_of_intersecting_triangles_l1530_153045


namespace quadratic_root_form_l1530_153014

theorem quadratic_root_form {a b : ℂ} (h : 6 * a ^ 2 - 5 * a + 18 = 0 ∧ a.im = 0 ∧ b.im = 0) : 
  a + b^2 = (467:ℚ) / 144 :=
by
  sorry

end quadratic_root_form_l1530_153014


namespace jason_commute_with_detour_l1530_153062

theorem jason_commute_with_detour (d1 d2 d3 d4 d5 : ℝ) 
  (h1 : d1 = 4)     -- Distance from house to first store
  (h2 : d2 = 6)     -- Distance between first and second store
  (h3 : d3 = d2 + (2/3) * d2) -- Distance between second and third store without detour
  (h4 : d4 = 3)     -- Additional distance due to detour
  (h5 : d5 = d1)    -- Distance from third store to work
  : d1 + d2 + (d3 + d4) + d5 = 27 :=
by
  sorry

end jason_commute_with_detour_l1530_153062


namespace hyperbola_intersection_l1530_153064

variable (a b c : ℝ) -- positive constants
variables (F1 F2 : (ℝ × ℝ)) -- foci of the hyperbola

-- The positive constants a and b
axiom a_pos : a > 0
axiom b_pos : b > 0

-- The foci are at (-c, 0) and (c, 0)
axiom F1_def : F1 = (-c, 0)
axiom F2_def : F2 = (c, 0)

-- We want to prove that the points (-c, b^2 / a) and (-c, -b^2 / a) are on the hyperbola
theorem hyperbola_intersection :
  (F1 = (-c, 0) ∧ F2 = (c, 0) ∧ a > 0 ∧ b > 0) →
  ∀ y : ℝ, ∃ y1 y2 : ℝ, (y1 = b^2 / a ∧ y2 = -b^2 / a ∧ 
  ( ( (-c)^2 / a^2) - (y1^2 / b^2) = 1 ∧  (-c)^2 / a^2 - y2^2 / b^2 = 1 ) ) :=
by
  intros h
  sorry

end hyperbola_intersection_l1530_153064


namespace volume_of_larger_prism_is_correct_l1530_153021

noncomputable def volume_of_larger_solid : ℝ :=
  let A := (0, 0, 0)
  let B := (2, 0, 0)
  let C := (2, 2, 0)
  let D := (0, 2, 0)
  let E := (0, 0, 2)
  let F := (2, 0, 2)
  let G := (2, 2, 2)
  let H := (0, 2, 2)
  let P := (1, 1, 1)
  let Q := (1, 0, 1)
  
  -- Assume the plane equation here divides the cube into equal halves
  -- Calculate the volume of one half of the cube
  let volume := 2 -- This represents the volume of the larger solid

  volume

theorem volume_of_larger_prism_is_correct :
  volume_of_larger_solid = 2 :=
sorry

end volume_of_larger_prism_is_correct_l1530_153021


namespace total_handshakes_at_convention_l1530_153068

def number_of_gremlins := 30
def number_of_imps := 20
def disagreeing_imps := 5
def specific_gremlins := 10

theorem total_handshakes_at_convention : 
  (number_of_gremlins * (number_of_gremlins - 1) / 2) +
  ((number_of_imps - disagreeing_imps) * number_of_gremlins) + 
  (disagreeing_imps * (number_of_gremlins - specific_gremlins)) = 985 :=
by 
  sorry

end total_handshakes_at_convention_l1530_153068


namespace max_value_of_N_l1530_153079

def I_k (k : Nat) : Nat :=
  10^(k + 1) + 32

def N (k : Nat) : Nat :=
  (Nat.factors (I_k k)).count 2

theorem max_value_of_N :
  ∃ k : Nat, N k = 6 ∧ (∀ m : Nat, N m ≤ 6) :=
by
  sorry

end max_value_of_N_l1530_153079


namespace max_value_inequality_am_gm_inequality_l1530_153038

-- Given conditions and goals as Lean statements
theorem max_value_inequality (x : ℝ) : (|x - 1| + |x - 2| ≥ 1) := sorry

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (1/a) + (1/(2*b)) + (1/(3*c)) = 1) : (a + 2*b + 3*c) ≥ 9 := sorry

end max_value_inequality_am_gm_inequality_l1530_153038


namespace price_alloy_per_kg_l1530_153004

-- Defining the costs of the two metals.
def cost_metal1 : ℝ := 68
def cost_metal2 : ℝ := 96

-- Defining the mixture ratio.
def ratio : ℝ := 1

-- The proposition that the price per kg of the alloy is 82 Rs.
theorem price_alloy_per_kg (C1 C2 r : ℝ) (hC1 : C1 = 68) (hC2 : C2 = 96) (hr : r = 1) :
  (C1 + C2) / (r + r) = 82 :=
by
  sorry

end price_alloy_per_kg_l1530_153004


namespace general_rule_equation_l1530_153071

theorem general_rule_equation (n : ℕ) (hn : n > 0) : (n + 1) / n + (n + 1) = (n + 2) + 1 / n :=
by
  sorry

end general_rule_equation_l1530_153071


namespace calculation_A_B_l1530_153093

theorem calculation_A_B :
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  A - B = 4397 :=
by
  let A := 19 * 10 + 55 * 100
  let B := 173 + 224 * 5
  sorry

end calculation_A_B_l1530_153093


namespace sum_of_xy_is_1289_l1530_153083

-- Define the variables and conditions
def internal_angle1 (x y : ℕ) : ℕ := 5 * x + 3 * y
def internal_angle2 (x y : ℕ) : ℕ := 3 * x + 20
def internal_angle3 (x y : ℕ) : ℕ := 10 * y + 30

-- Definition of the sum of angles of a triangle
def sum_of_angles (x y : ℕ) : ℕ := internal_angle1 x y + internal_angle2 x y + internal_angle3 x y

-- Define the theorem statement
theorem sum_of_xy_is_1289 (x y : ℕ) (hx : x > 0) (hy : y > 0) 
  (h : sum_of_angles x y = 180) : x + y = 1289 :=
by sorry

end sum_of_xy_is_1289_l1530_153083


namespace dan_initial_money_l1530_153074

theorem dan_initial_money (cost_candy : ℕ) (cost_chocolate : ℕ) (total_spent: ℕ) (hc : cost_candy = 7) (hch : cost_chocolate = 6) (hs : total_spent = 13) 
  (h : total_spent = cost_candy + cost_chocolate) : total_spent = 13 := by
  sorry

end dan_initial_money_l1530_153074


namespace length_real_axis_l1530_153022

theorem length_real_axis (x y : ℝ) : 
  (x^2 / 4 - y^2 / 12 = 1) → 4 = 4 :=
by
  intro h
  sorry

end length_real_axis_l1530_153022


namespace josephine_total_milk_l1530_153012

-- Define the number of containers and the amount of milk they hold
def cnt_1 : ℕ := 3
def qty_1 : ℚ := 2

def cnt_2 : ℕ := 2
def qty_2 : ℚ := 0.75

def cnt_3 : ℕ := 5
def qty_3 : ℚ := 0.5

-- Define the total amount of milk sold
def total_milk_sold : ℚ := cnt_1 * qty_1 + cnt_2 * qty_2 + cnt_3 * qty_3

theorem josephine_total_milk : total_milk_sold = 10 := by
  -- This is the proof placeholder
  sorry

end josephine_total_milk_l1530_153012


namespace determine_asymptotes_l1530_153043

noncomputable def asymptotes_of_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  (∀ x y : ℝ, (y = x * (Real.sqrt 2 / 2) ∨ y = -x * (Real.sqrt 2 / 2)))

theorem determine_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  asymptotes_of_hyperbola a b ha hb :=
by
  intros h
  sorry

end determine_asymptotes_l1530_153043


namespace average_marks_five_subjects_l1530_153089

theorem average_marks_five_subjects 
  (P total_marks : ℕ)
  (h1 : total_marks = P + 350) :
  (total_marks - P) / 5 = 70 :=
by
  sorry

end average_marks_five_subjects_l1530_153089


namespace least_possible_value_of_m_plus_n_l1530_153047

noncomputable def least_possible_sum (m n : ℕ) : ℕ :=
m + n

theorem least_possible_value_of_m_plus_n (m n : ℕ) 
  (h1 : m > 0) 
  (h2 : n > 0)
  (h3 : Nat.gcd (m + n) 330 = 1)
  (h4 : m^m % n^n = 0)
  (h5 : m % n ≠ 0) : 
  least_possible_sum m n = 98 := 
sorry

end least_possible_value_of_m_plus_n_l1530_153047


namespace jinsu_third_attempt_kicks_l1530_153016

theorem jinsu_third_attempt_kicks
  (hoseok_kicks : ℕ) (jinsu_first_attempt : ℕ) (jinsu_second_attempt : ℕ) (required_kicks : ℕ) :
  hoseok_kicks = 48 →
  jinsu_first_attempt = 15 →
  jinsu_second_attempt = 15 →
  required_kicks = 19 →
  jinsu_first_attempt + jinsu_second_attempt + required_kicks > hoseok_kicks :=
by
  sorry

end jinsu_third_attempt_kicks_l1530_153016


namespace triangle_incircle_ratio_l1530_153087

theorem triangle_incircle_ratio (r s q : ℝ) (h1 : r + s = 8) (h2 : r < s) (h3 : r + q = 13) (h4 : s + q = 17) (h5 : 8 + 13 > 17 ∧ 8 + 17 > 13 ∧ 13 + 17 > 8):
  r / s = 1 / 3 := by sorry

end triangle_incircle_ratio_l1530_153087


namespace age_of_other_replaced_man_l1530_153092

theorem age_of_other_replaced_man (A B C D : ℕ) (h1 : A = 23) (h2 : ((52 + C + D) / 4 > (A + B + C + D) / 4)) :
  B < 29 := 
by
  sorry

end age_of_other_replaced_man_l1530_153092


namespace fraction_value_l1530_153088

theorem fraction_value : (5 - Real.sqrt 4) / (5 + Real.sqrt 4) = 3 / 7 := by
  sorry

end fraction_value_l1530_153088


namespace cats_in_village_l1530_153020

theorem cats_in_village (C : ℕ) (h1 : 1 / 3 * C = (1 / 4) * (1 / 3) * C)
  (h2 : (1 / 12) * C = 10) : C = 120 :=
sorry

end cats_in_village_l1530_153020


namespace conic_is_parabola_l1530_153010

-- Define the main equation
def main_equation (x y : ℝ) : Prop :=
  y^4 - 6 * x^2 = 3 * y^2 - 2

-- Definition of parabola condition
def is_parabola (x y : ℝ) : Prop :=
  ∃ a b c : ℝ, y^2 = a * x + b ∧ a ≠ 0

-- The theorem statement.
theorem conic_is_parabola :
  ∀ x y : ℝ, main_equation x y → is_parabola x y :=
by
  intros x y h
  sorry

end conic_is_parabola_l1530_153010


namespace seats_per_bus_correct_l1530_153048

-- Define the conditions given in the problem
def students : ℕ := 28
def buses : ℕ := 4

-- Define the number of seats per bus
def seats_per_bus : ℕ := students / buses

-- State the theorem that proves the number of seats per bus
theorem seats_per_bus_correct : seats_per_bus = 7 := by
  -- conditions are used as definitions, the goal is to prove seats_per_bus == 7
  sorry

end seats_per_bus_correct_l1530_153048


namespace chickens_cheaper_than_buying_eggs_l1530_153018

theorem chickens_cheaper_than_buying_eggs :
  ∃ W, W ≥ 80 ∧ 80 + W ≤ 2 * W :=
by
  sorry

end chickens_cheaper_than_buying_eggs_l1530_153018


namespace equal_serving_weight_l1530_153099

theorem equal_serving_weight (total_weight : ℝ) (num_family_members : ℕ)
  (h1 : total_weight = 13) (h2 : num_family_members = 5) :
  total_weight / num_family_members = 2.6 :=
by
  sorry

end equal_serving_weight_l1530_153099


namespace xiangming_payment_methods_count_l1530_153009

def xiangming_payment_methods : Prop :=
  ∃ x y z : ℕ, 
    x + y + z ≤ 10 ∧ 
    x + 2 * y + 5 * z = 18 ∧ 
    ((x > 0 ∧ y > 0) ∨ (x > 0 ∧ z > 0) ∨ (y > 0 ∧ z > 0))

theorem xiangming_payment_methods_count : 
  xiangming_payment_methods → ∃! n, n = 11 :=
by sorry

end xiangming_payment_methods_count_l1530_153009


namespace quadratic_real_roots_condition_l1530_153051

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (m-1) * x₁^2 - 4 * x₁ + 1 = 0 ∧ (m-1) * x₂^2 - 4 * x₂ + 1 = 0) ↔ (m < 5 ∧ m ≠ 1) :=
by
  sorry

end quadratic_real_roots_condition_l1530_153051


namespace find_a_l1530_153025

theorem find_a (a : ℝ) 
  (line_through : ∃ (p1 p2 : ℝ × ℝ), p1 = (a-2, -1) ∧ p2 = (-a-2, 1)) 
  (perpendicular : ∀ (l1 l2 : ℝ × ℝ), l1 = (2, 3) → l2 = (-1/a, 1) → false) : 
  a = -2/3 :=
by 
  sorry

end find_a_l1530_153025


namespace find_m_l1530_153069

-- Definitions of the given vectors and their properties
def a : ℝ × ℝ := (1, -3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

-- Condition that vectors a and b are parallel
def are_parallel (v₁ v₂ : ℝ × ℝ) : Prop :=
  v₁.1 * v₂.2 - v₁.2 * v₂.1 = 0

-- Goal: Find the value of m such that vectors a and b are parallel
theorem find_m (m : ℝ) : 
  are_parallel a (b m) → m = 6 :=
by
  sorry

end find_m_l1530_153069


namespace rational_root_theorem_l1530_153097

theorem rational_root_theorem :
  (∃ x : ℚ, 3 * x^4 - 4 * x^3 - 10 * x^2 + 8 * x + 3 = 0)
  → (x = 1 ∨ x = 1/3) := by
  sorry

end rational_root_theorem_l1530_153097


namespace prime_pairs_satisfying_conditions_l1530_153072

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q : ℕ) : Prop :=
  (7 * p + 1) % q = 0 ∧ (7 * q + 1) % p = 0

theorem prime_pairs_satisfying_conditions :
  { (p, q) | is_prime p ∧ is_prime q ∧ satisfies_conditions p q } = {(2, 3), (2, 5), (3, 11)} := 
sorry

end prime_pairs_satisfying_conditions_l1530_153072
