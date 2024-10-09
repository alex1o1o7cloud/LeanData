import Mathlib

namespace monotonic_intervals_range_of_a_l441_44170

noncomputable def f (x a : ℝ) := Real.log x + (a / 2) * x^2 - (a + 1) * x
noncomputable def f' (x a : ℝ) := 1 / x + a * x - (a + 1)

theorem monotonic_intervals (a : ℝ) (ha : f 1 a = -2 ∧ f' 1 a = 0):
  (∀ x : ℝ, 0 < x ∧ x < (1 / 2) → f' x a > 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a > 0) ∧ 
  (∀ x : ℝ, (1 / 2) < x ∧ x < 1 → f' x a < 0) := sorry

theorem range_of_a (a : ℝ) 
  (h : ∀ x : ℕ, x > 0 → (f x a) / x < (f' x a) / 2):
  a > 2 * Real.exp (- (3 / 2)) - 1 := sorry

end monotonic_intervals_range_of_a_l441_44170


namespace opposite_of_2023_is_neg_2023_l441_44142

-- Definitions based on conditions
def is_additive_inverse (x y : Int) : Prop := x + y = 0

-- The proof statement
theorem opposite_of_2023_is_neg_2023 : is_additive_inverse 2023 (-2023) :=
by
  -- This is where the proof would go, but it is marked as sorry for now
  sorry

end opposite_of_2023_is_neg_2023_l441_44142


namespace max_sector_area_l441_44192

theorem max_sector_area (r θ : ℝ) (h₁ : 2 * r + r * θ = 16) : 
  (∃ A : ℝ, A = 1/2 * r^2 * θ ∧ A ≤ 16) ∧ (∃ r θ, r = 4 ∧ θ = 2 ∧ 1/2 * r^2 * θ = 16) := 
by
  sorry

end max_sector_area_l441_44192


namespace line_intersects_extension_of_segment_l441_44155

theorem line_intersects_extension_of_segment
  (A B C x1 y1 x2 y2 : ℝ)
  (hnz : A ≠ 0 ∨ B ≠ 0)
  (h1 : (A * x1 + B * y1 + C) * (A * x2 + B * y2 + C) > 0)
  (h2 : |A * x1 + B * y1 + C| > |A * x2 + B * y2 + C|) :
  ∃ t : ℝ, t ≥ 0 ∧ l * (t * (x2 - x1) + x1) + m * (t * (y2 - y1) + y1) = 0 :=
sorry

end line_intersects_extension_of_segment_l441_44155


namespace range_of_x_l441_44197

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 * x else 2 * -x

theorem range_of_x {x : ℝ} :
  f (1 - 2 * x) < f 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end range_of_x_l441_44197


namespace counting_five_digit_numbers_l441_44129

theorem counting_five_digit_numbers :
  ∃ (M : ℕ), 
    (∃ (b : ℕ), (∃ (y : ℕ), 10000 * b + y = 8 * y ∧ 10000 * b = 7 * y ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1429 ≤ y ∧ y ≤ 9996)) ∧ 
    (M = 1224) := 
by
  sorry

end counting_five_digit_numbers_l441_44129


namespace sheep_per_herd_l441_44123

theorem sheep_per_herd (herds : ℕ) (total_sheep : ℕ) (h_herds : herds = 3) (h_total_sheep : total_sheep = 60) : 
  (total_sheep / herds) = 20 :=
by
  sorry

end sheep_per_herd_l441_44123


namespace mike_games_l441_44122

theorem mike_games (initial_money spent_money game_cost remaining_games : ℕ)
  (h1 : initial_money = 101)
  (h2 : spent_money = 47)
  (h3 : game_cost = 6)
  (h4 : remaining_games = (initial_money - spent_money) / game_cost) :
  remaining_games = 9 := by
  sorry

end mike_games_l441_44122


namespace latest_leave_time_correct_l441_44184

-- Define the conditions
def flight_time := 20 -- 8:00 pm in 24-hour format
def check_in_early := 2 -- 2 hours early
def drive_time := 45 -- 45 minutes
def park_time := 15 -- 15 minutes

-- Define the target time to be at the airport
def at_airport_time := flight_time - check_in_early -- 18:00 or 6:00 pm

-- Total travel time required (minutes)
def total_travel_time := drive_time + park_time -- 60 minutes

-- Convert total travel time to hours
def travel_time_in_hours : ℕ := total_travel_time / 60

-- Define the latest time to leave the house
def latest_leave_time := at_airport_time - travel_time_in_hours

-- Theorem to state the equivalence of the latest time they can leave their house
theorem latest_leave_time_correct : latest_leave_time = 17 :=
    by
    sorry

end latest_leave_time_correct_l441_44184


namespace total_money_spent_l441_44147

-- Definitions based on conditions
def num_bars_of_soap : Nat := 20
def weight_per_bar_of_soap : Float := 1.5
def cost_per_pound_of_soap : Float := 0.5

def num_bottles_of_shampoo : Nat := 15
def weight_per_bottle_of_shampoo : Float := 2.2
def cost_per_pound_of_shampoo : Float := 0.8

-- The theorem to prove
theorem total_money_spent :
  let cost_per_bar_of_soap := weight_per_bar_of_soap * cost_per_pound_of_soap
  let total_cost_of_soap := Float.ofNat num_bars_of_soap * cost_per_bar_of_soap
  let cost_per_bottle_of_shampoo := weight_per_bottle_of_shampoo * cost_per_pound_of_shampoo
  let total_cost_of_shampoo := Float.ofNat num_bottles_of_shampoo * cost_per_bottle_of_shampoo
  total_cost_of_soap + total_cost_of_shampoo = 41.40 := 
by
  -- proof goes here
  sorry

end total_money_spent_l441_44147


namespace ferris_wheel_capacity_l441_44116

theorem ferris_wheel_capacity :
  let num_seats := 4
  let people_per_seat := 4
  num_seats * people_per_seat = 16 := 
by
  let num_seats := 4
  let people_per_seat := 4
  sorry

end ferris_wheel_capacity_l441_44116


namespace cuboid_surface_area_l441_44135

-- Definitions
def Length := 12  -- meters
def Breadth := 14  -- meters
def Height := 7  -- meters

-- Surface area of a cuboid formula
def surfaceAreaOfCuboid (l b h : Nat) : Nat :=
  2 * (l * b + l * h + b * h)

-- Proof statement
theorem cuboid_surface_area : surfaceAreaOfCuboid Length Breadth Height = 700 := by
  sorry

end cuboid_surface_area_l441_44135


namespace jackson_weekly_mileage_increase_l441_44181

theorem jackson_weekly_mileage_increase :
  ∃ (weeks : ℕ), weeks = (7 - 3) / 1 := by
  sorry

end jackson_weekly_mileage_increase_l441_44181


namespace baseball_cards_per_pack_l441_44169

theorem baseball_cards_per_pack (cards_each : ℕ) (packs_total : ℕ) (total_cards : ℕ) (cards_per_pack : ℕ) :
    (cards_each = 540) →
    (packs_total = 108) →
    (total_cards = cards_each * 4) →
    (cards_per_pack = total_cards / packs_total) →
    cards_per_pack = 20 :=
by
  intros h1 h2 h3 h4
  sorry

end baseball_cards_per_pack_l441_44169


namespace complete_the_square_d_l441_44124

theorem complete_the_square_d (x : ℝ) : (∃ c d : ℝ, x^2 + 6 * x - 4 = 0 → (x + c)^2 = d) ∧ d = 13 :=
by
  sorry

end complete_the_square_d_l441_44124


namespace passengers_taken_at_second_station_l441_44105

noncomputable def initial_passengers : ℕ := 270
noncomputable def passengers_dropped_first_station := initial_passengers / 3
noncomputable def passengers_after_first_station := initial_passengers - passengers_dropped_first_station + 280
noncomputable def passengers_dropped_second_station := passengers_after_first_station / 2
noncomputable def passengers_after_second_station (x : ℕ) := passengers_after_first_station - passengers_dropped_second_station + x
noncomputable def passengers_at_third_station := 242

theorem passengers_taken_at_second_station : ∃ x : ℕ,
  passengers_after_second_station x = passengers_at_third_station ∧ x = 12 :=
by
  sorry

end passengers_taken_at_second_station_l441_44105


namespace evaluate_expression_l441_44171

theorem evaluate_expression : (831 * 831) - (830 * 832) = 1 :=
by
  sorry

end evaluate_expression_l441_44171


namespace multiply_polynomials_l441_44137

theorem multiply_polynomials (x : ℝ) : 
  (x^6 + 64 * x^3 + 4096) * (x^3 - 64) = x^9 - 262144 :=
by
  sorry

end multiply_polynomials_l441_44137


namespace hide_and_seek_problem_l441_44185

variable (A B V G D : Prop)

theorem hide_and_seek_problem :
  (A → (B ∧ ¬V)) →
  (B → (G ∨ D)) →
  (¬V → (¬B ∧ ¬D)) →
  (¬A → (B ∧ ¬G)) →
  ¬A ∧ B ∧ ¬V ∧ ¬G ∧ D :=
by
  intros h1 h2 h3 h4
  sorry

end hide_and_seek_problem_l441_44185


namespace double_exceeds_one_fifth_by_nine_l441_44154

theorem double_exceeds_one_fifth_by_nine (x : ℝ) (h : 2 * x = (1 / 5) * x + 9) : x^2 = 25 :=
sorry

end double_exceeds_one_fifth_by_nine_l441_44154


namespace door_height_eight_l441_44125

theorem door_height_eight (x : ℝ) (h w : ℝ) (H1 : w = x - 4) (H2 : h = x - 2) (H3 : x^2 = (x - 4)^2 + (x - 2)^2) : h = 8 :=
by
  sorry

end door_height_eight_l441_44125


namespace minimum_value_of_ex_4e_negx_l441_44194

theorem minimum_value_of_ex_4e_negx : 
  ∃ (x : ℝ), (∀ (y : ℝ), y = Real.exp x + 4 * Real.exp (-x) → y ≥ 4) ∧ (Real.exp x + 4 * Real.exp (-x) = 4) :=
sorry

end minimum_value_of_ex_4e_negx_l441_44194


namespace intersection_of_A_and_B_l441_44108

namespace IntersectionProblem

def setA : Set ℝ := {0, 1, 2}
def setB : Set ℝ := {x | x^2 - x ≤ 0}
def intersection : Set ℝ := {0, 1}

theorem intersection_of_A_and_B : A ∩ B = intersection := sorry

end IntersectionProblem

end intersection_of_A_and_B_l441_44108


namespace average_speed_correct_l441_44136

-- Definitions for the conditions
def distance1 : ℚ := 40
def speed1 : ℚ := 8
def time1 : ℚ := distance1 / speed1

def distance2 : ℚ := 20
def speed2 : ℚ := 40
def time2 : ℚ := distance2 / speed2

def total_distance : ℚ := distance1 + distance2
def total_time : ℚ := time1 + time2

-- Definition of average speed
def average_speed : ℚ := total_distance / total_time

-- Proof statement that needs to be proven
theorem average_speed_correct : average_speed = 120 / 11 :=
by 
  -- The details for the proof will be filled here
  sorry

end average_speed_correct_l441_44136


namespace price_of_fruit_l441_44165

theorem price_of_fruit
  (price_milk_per_liter : ℝ)
  (milk_per_batch : ℝ)
  (fruit_per_batch : ℝ)
  (cost_for_three_batches : ℝ)
  (F : ℝ)
  (h1 : price_milk_per_liter = 1.5)
  (h2 : milk_per_batch = 10)
  (h3 : fruit_per_batch = 3)
  (h4 : cost_for_three_batches = 63)
  (h5 : 3 * (milk_per_batch * price_milk_per_liter + fruit_per_batch * F) = cost_for_three_batches) :
  F = 2 :=
by sorry

end price_of_fruit_l441_44165


namespace trigonometric_identity_l441_44150

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -2) :
  (Real.sin θ * (1 + Real.sin (2 * θ))) / (Real.sin θ + Real.cos θ) = 2 / 5 := 
sorry

end trigonometric_identity_l441_44150


namespace sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l441_44113

def recurrence_relation (x : ℕ → ℝ) (n : ℕ) : Prop :=
  x (n + 1) = (2 * x n ^ 2 - x n) / (3 * (x n - 2))

-- For the first problem
theorem sequence_increasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : 4 < x 0 ∧ x 0 < 5) : ∀ n, x n < x (n + 1) ∧ x (n + 1) < 5 :=
by
  sorry

-- For the second problem
theorem sequence_decreasing_and_bounded (x : ℕ → ℝ) (h_rec : ∀ n, recurrence_relation x n)
  (h_initial : x 0 > 5) : ∀ n, 5 < x (n + 1) ∧ x (n + 1) < x n :=
by
  sorry

end sequence_increasing_and_bounded_sequence_decreasing_and_bounded_l441_44113


namespace cabin_charges_per_night_l441_44115

theorem cabin_charges_per_night 
  (total_lodging_cost : ℕ)
  (hostel_cost_per_night : ℕ)
  (hostel_days : ℕ)
  (total_cabin_days : ℕ)
  (friends_sharing_expenses : ℕ)
  (jimmy_lodging_expense : ℕ) 
  (total_cost_paid_by_jimmy : ℕ) :
  total_lodging_cost = total_cost_paid_by_jimmy →
  hostel_cost_per_night = 15 →
  hostel_days = 3 →
  total_cabin_days = 2 →
  friends_sharing_expenses = 3 →
  jimmy_lodging_expense = 75 →
  ∃ cabin_cost_per_night, cabin_cost_per_night = 45 :=
by
  sorry

end cabin_charges_per_night_l441_44115


namespace k_value_range_l441_44198

noncomputable def f (x : ℝ) : ℝ := x - 1 - Real.log x

theorem k_value_range {k : ℝ} (h : ∀ x : ℝ, 0 < x → f x ≥ k * x - 2) : 
  k ≤ 1 - 1 / Real.exp 2 := 
sorry

end k_value_range_l441_44198


namespace solution_set_of_inequality_l441_44106

theorem solution_set_of_inequality (x : ℝ) : x * (x + 2) ≥ 0 ↔ x ≤ -2 ∨ x ≥ 0 := 
sorry

end solution_set_of_inequality_l441_44106


namespace intersection_A_CRB_l441_44199

-- Definition of sets A and C_{R}B
def is_in_A (x: ℝ) := 0 < x ∧ x < 2

def is_in_CRB (x: ℝ) := x ≤ 1 ∨ x ≥ Real.exp 2

-- Proof that the intersection of A and C_{R}B is (0, 1]
theorem intersection_A_CRB : {x : ℝ | is_in_A x} ∩ {x : ℝ | is_in_CRB x} = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by
  sorry

end intersection_A_CRB_l441_44199


namespace sqrt_product_eq_sixty_sqrt_two_l441_44119

theorem sqrt_product_eq_sixty_sqrt_two : (Real.sqrt 50) * (Real.sqrt 18) * (Real.sqrt 8) = 60 * (Real.sqrt 2) := 
by 
  sorry

end sqrt_product_eq_sixty_sqrt_two_l441_44119


namespace value_of_y_l441_44191

noncomputable def k : ℝ := 168.75

theorem value_of_y (x y : ℝ) (h1 : x * y = k) (h2 : x + y = 30) (h3 : x = 3 * y) : y = -16.875 :=
by 
  sorry

end value_of_y_l441_44191


namespace expected_value_is_90_l441_44121

noncomputable def expected_value_coins_heads : ℕ :=
  let nickel := 5
  let quarter := 25
  let half_dollar := 50
  let dollar := 100
  1/2 * (nickel + quarter + half_dollar + dollar)

theorem expected_value_is_90 : expected_value_coins_heads = 90 := by
  sorry

end expected_value_is_90_l441_44121


namespace henry_needs_30_dollars_l441_44101

def henry_action_figures_completion (current_figures total_figures cost_per_figure : ℕ) : ℕ :=
  (total_figures - current_figures) * cost_per_figure

theorem henry_needs_30_dollars : henry_action_figures_completion 3 8 6 = 30 := by
  sorry

end henry_needs_30_dollars_l441_44101


namespace correct_option_l441_44152

theorem correct_option :
  (3 * a^2 + 5 * a^2 ≠ 8 * a^4) ∧
  (5 * a^2 * b - 6 * a * b^2 ≠ -a * b^2) ∧
  (2 * x + 3 * y ≠ 5 * x * y) ∧
  (9 * x * y - 6 * x * y = 3 * x * y) :=
by
  sorry

end correct_option_l441_44152


namespace investment_of_q_is_correct_l441_44162

-- Define investments and the profit ratio
def p_investment : ℝ := 30000
def profit_ratio_p : ℝ := 2
def profit_ratio_q : ℝ := 3

-- Define q's investment as x
def q_investment : ℝ := 45000

-- The goal is to prove that q_investment is indeed 45000 given the above conditions
theorem investment_of_q_is_correct :
  (p_investment / q_investment) = (profit_ratio_p / profit_ratio_q) :=
sorry

end investment_of_q_is_correct_l441_44162


namespace sqrt_meaningful_range_l441_44176

theorem sqrt_meaningful_range (a : ℝ) : (∃ x : ℝ, x = Real.sqrt (a + 2)) ↔ a ≥ -2 := 
sorry

end sqrt_meaningful_range_l441_44176


namespace compartments_count_l441_44161

-- Definition of initial pennies per compartment
def initial_pennies_per_compartment : ℕ := 2

-- Definition of additional pennies added to each compartment
def additional_pennies_per_compartment : ℕ := 6

-- Definition of total pennies is 96
def total_pennies : ℕ := 96

-- Prove the number of compartments is 12
theorem compartments_count (c : ℕ) 
  (h1 : initial_pennies_per_compartment + additional_pennies_per_compartment = 8)
  (h2 : 8 * c = total_pennies) : 
  c = 12 :=
by
  sorry

end compartments_count_l441_44161


namespace mean_of_three_is_90_l441_44144

-- Given conditions as Lean definitions
def mean_twelve (s : ℕ) : Prop := s = 12 * 40
def added_sum (x y z : ℕ) (s : ℕ) : Prop := s + x + y + z = 15 * 50
def z_value (x z : ℕ) : Prop := z = x + 10

-- Theorem statement to prove the mean of x, y, and z is 90
theorem mean_of_three_is_90 (x y z s : ℕ) : 
  (mean_twelve s) → (z_value x z) → (added_sum x y z s) → 
  (x + y + z) / 3 = 90 := 
by 
  intros h1 h2 h3 
  sorry

end mean_of_three_is_90_l441_44144


namespace new_container_volume_l441_44196

theorem new_container_volume (original_volume : ℕ) (factor : ℕ) (new_volume : ℕ) 
    (h1 : original_volume = 5) (h2 : factor = 4 * 4 * 4) : new_volume = 320 :=
by
  sorry

end new_container_volume_l441_44196


namespace sum_of_products_l441_44182

theorem sum_of_products (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 390) 
  (h2 : a + b + c = 20) : 
  ab + bc + ca = 5 :=
by 
  sorry

end sum_of_products_l441_44182


namespace problem_l441_44109

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

theorem problem :
  (∀ x, f (-x) = -f x) → -- f is odd
  (∀ x, f (x + 2) = -1 / f x) → -- Functional equation
  (∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) → -- Definition on interval (0,1)
  f (Real.log (54) / Real.log 3) = -3 / 2 := sorry

end problem_l441_44109


namespace cube_face_sum_l441_44172

theorem cube_face_sum (a b c d e f : ℕ) (h1 : e = b) (h2 : 2 * (a * b * c + a * b * f + d * b * c + d * b * f) = 1332) :
  a + b + c + d + e + f = 47 :=
sorry

end cube_face_sum_l441_44172


namespace multiple_of_kids_finishing_early_l441_44127

-- Definitions based on conditions
def num_10_percent_kids (total_kids : ℕ) : ℕ := (total_kids * 10) / 100

def num_remaining_kids (total_kids kids_less_6 kids_more_14 : ℕ) : ℕ := total_kids - kids_less_6 - kids_more_14

def num_multiple_finishing_less_8 (total_kids : ℕ) (multiple : ℕ) : ℕ := multiple * num_10_percent_kids total_kids

-- Main theorem statement
theorem multiple_of_kids_finishing_early 
  (total_kids : ℕ)
  (h_total_kids : total_kids = 40)
  (kids_more_14 : ℕ)
  (h_kids_more_14 : kids_more_14 = 4)
  (h_1_6_remaining : kids_more_14 = num_remaining_kids total_kids (num_10_percent_kids total_kids) kids_more_14 / 6)
  : (num_multiple_finishing_less_8 total_kids 3) = (total_kids - num_10_percent_kids total_kids - kids_more_14) := 
by 
  sorry

end multiple_of_kids_finishing_early_l441_44127


namespace inequality_holds_l441_44132

theorem inequality_holds (a : ℝ) : (∀ x : ℝ, a * x ^ 2 - 2 * a * x + 1 > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end inequality_holds_l441_44132


namespace find_number_l441_44174

theorem find_number (N : ℤ) (h1 : ∃ k : ℤ, N - 3 = 5 * k) (h2 : ∃ l : ℤ, N - 2 = 7 * l) (h3 : 50 < N ∧ N < 70) : N = 58 :=
by
  sorry

end find_number_l441_44174


namespace tan_half_angle_l441_44138

theorem tan_half_angle {α β : ℝ} (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) (h1 : Real.tan α = 2) (h2 : Real.tan β = 3) :
  Real.tan ((α + β) / 2) = 1 + Real.sqrt 2 := 
sorry

end tan_half_angle_l441_44138


namespace first_valve_time_l441_44103

noncomputable def first_valve_filling_time (V1 V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ) : ℝ :=
  pool_capacity / V1

theorem first_valve_time :
  ∀ (V1 : ℝ) (V2 : ℝ) (pool_capacity : ℝ) (combined_time : ℝ),
    V2 = V1 + 50 →
    V1 + V2 = pool_capacity / combined_time →
    combined_time = 48 →
    pool_capacity = 12000 →
    first_valve_filling_time V1 V2 pool_capacity combined_time / 60 = 2 :=
  
by
  intros V1 V2 pool_capacity combined_time h1 h2 h3 h4
  sorry

end first_valve_time_l441_44103


namespace trajectory_of_P_eqn_l441_44149

theorem trajectory_of_P_eqn :
  ∀ {x y : ℝ}, -- For all real numbers x and y
  (-(x + 2)^2 + (x - 1)^2 + y^2 = 3*((x - 1)^2 + y^2)) → -- Condition |PA| = 2|PB|
  (x^2 + y^2 - 4*x = 0) := -- Prove the trajectory equation
by
  intros x y h
  sorry -- Proof to be completed

end trajectory_of_P_eqn_l441_44149


namespace original_price_of_article_l441_44130

theorem original_price_of_article (P : ℝ) : 
  (P - 0.30 * P) * (1 - 0.20) = 1120 → P = 2000 :=
by
  intro h
  -- h represents the given condition for the problem
  sorry  -- proof will go here

end original_price_of_article_l441_44130


namespace aqua_park_earnings_l441_44134

def admission_cost : ℕ := 12
def tour_cost : ℕ := 6
def group1_size : ℕ := 10
def group2_size : ℕ := 5

theorem aqua_park_earnings :
  (group1_size * admission_cost + group1_size * tour_cost) + (group2_size * admission_cost) = 240 :=
by
  sorry

end aqua_park_earnings_l441_44134


namespace find_larger_number_l441_44145

theorem find_larger_number (L S : ℕ)
  (h1 : L - S = 1370)
  (h2 : L = 6 * S + 15) :
  L = 1641 := sorry

end find_larger_number_l441_44145


namespace total_cost_l441_44118

noncomputable def cost_sandwich : ℝ := 2.44
noncomputable def quantity_sandwich : ℕ := 2
noncomputable def cost_soda : ℝ := 0.87
noncomputable def quantity_soda : ℕ := 4

noncomputable def total_cost_sandwiches : ℝ := cost_sandwich * quantity_sandwich
noncomputable def total_cost_sodas : ℝ := cost_soda * quantity_soda

theorem total_cost (total_cost_sandwiches total_cost_sodas : ℝ) : (total_cost_sandwiches + total_cost_sodas = 8.36) :=
by
  sorry

end total_cost_l441_44118


namespace james_ride_time_l441_44178

theorem james_ride_time (distance speed : ℝ) (h_distance : distance = 200) (h_speed : speed = 25) : distance / speed = 8 :=
by
  rw [h_distance, h_speed]
  norm_num

end james_ride_time_l441_44178


namespace wendy_pictures_l441_44107

theorem wendy_pictures (album1_pics rest_albums albums each_album_pics : ℕ)
    (h1 : album1_pics = 44)
    (h2 : rest_albums = 5)
    (h3 : each_album_pics = 7)
    (h4 : albums = rest_albums * each_album_pics)
    (h5 : albums = 5 * 7):
  album1_pics + albums = 79 :=
by
  -- We leave the proof as an exercise
  sorry

end wendy_pictures_l441_44107


namespace sixth_term_geometric_sequence_l441_44156

theorem sixth_term_geometric_sequence (a r : ℚ) (h_a : a = 16) (h_r : r = 1/2) : 
  a * r^(5) = 1/2 :=
by 
  rw [h_a, h_r]
  sorry

end sixth_term_geometric_sequence_l441_44156


namespace find_d_l441_44179

theorem find_d (a₁: ℤ) (d : ℤ) (Sn : ℤ → ℤ) : 
  a₁ = 190 → 
  (Sn 20 > 0) → 
  (Sn 24 < 0) → 
  (Sn n = n * a₁ + (n * (n - 1)) / 2 * d) →
  d = -17 :=
by
  intros
  sorry

end find_d_l441_44179


namespace speed_of_each_train_l441_44186

theorem speed_of_each_train (v : ℝ) (train_length time_cross : ℝ) (km_pr_s : ℝ) 
  (h_train_length : train_length = 120)
  (h_time_cross : time_cross = 8)
  (h_km_pr_s : km_pr_s = 3.6)
  (h_relative_speed : 2 * v = (2 * train_length) / time_cross) :
  v * km_pr_s = 54 := 
by sorry

end speed_of_each_train_l441_44186


namespace largest_fraction_of_three_l441_44133

theorem largest_fraction_of_three (a b c : Nat) (h1 : Nat.gcd a 6 = 1)
  (h2 : Nat.gcd b 15 = 1) (h3 : Nat.gcd c 20 = 1)
  (h4 : (a * b * c) = 60) :
  max (a / 6) (max (b / 15) (c / 20)) = 5 / 6 :=
by
  sorry

end largest_fraction_of_three_l441_44133


namespace total_sheets_of_paper_l441_44193

theorem total_sheets_of_paper (classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ) 
  (h1 : classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) : 
  (classes * students_per_class * sheets_per_student) = 400 := 
by {
  sorry
}

end total_sheets_of_paper_l441_44193


namespace gcd_18_30_is_6_gcd_18_30_is_even_l441_44146

def gcd_18_30 : ℕ := Nat.gcd 18 30

theorem gcd_18_30_is_6 : gcd_18_30 = 6 := by
  sorry

theorem gcd_18_30_is_even : Even gcd_18_30 := by
  sorry

end gcd_18_30_is_6_gcd_18_30_is_even_l441_44146


namespace value_of_expression_l441_44188

theorem value_of_expression (x : ℝ) (h : (3 / (x - 3)) + (5 / (2 * x - 6)) = 11 / 2) : 2 * x - 6 = 2 :=
sorry

end value_of_expression_l441_44188


namespace find_m_plus_n_l441_44160

def operation (m n : ℕ) : ℕ := m^n + m * n

theorem find_m_plus_n :
  ∃ (m n : ℕ), (2 ≤ m) ∧ (2 ≤ n) ∧ (operation m n = 64) ∧ (m + n = 6) :=
by {
  -- Begin the proof context
  sorry
}

end find_m_plus_n_l441_44160


namespace hyperbola_vertices_distance_l441_44167

noncomputable def distance_between_vertices : ℝ :=
  2 * Real.sqrt 7.5

theorem hyperbola_vertices_distance :
  ∀ (x y : ℝ), 4 * x^2 - 24 * x - y^2 + 6 * y - 3 = 0 →
  distance_between_vertices = 2 * Real.sqrt 7.5 :=
by sorry

end hyperbola_vertices_distance_l441_44167


namespace trapezium_other_side_l441_44187

theorem trapezium_other_side (x : ℝ) :
  1/2 * (20 + x) * 10 = 150 → x = 10 :=
by
  sorry

end trapezium_other_side_l441_44187


namespace kayla_scores_on_sixth_level_l441_44102

-- Define the sequence of points scored in each level
def points (n : ℕ) : ℕ :=
  match n with
  | 0 => 2
  | 1 => 3
  | 2 => 5
  | 3 => 8
  | 4 => 12
  | n + 5 => points (n + 4) + (n + 1) + 1

-- Statement to prove that Kayla scores 17 points on the sixth level
theorem kayla_scores_on_sixth_level : points 5 = 17 :=
by
  sorry

end kayla_scores_on_sixth_level_l441_44102


namespace sum_of_first_four_terms_l441_44180

def arithmetic_sequence_sum (a1 a2 : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * (a2 - a1))) / 2

theorem sum_of_first_four_terms : arithmetic_sequence_sum 4 6 4 = 28 :=
by
  sorry

end sum_of_first_four_terms_l441_44180


namespace probability_truth_or_lies_l441_44166

def probability_truth := 0.30
def probability_lies := 0.20
def probability_both := 0.10

theorem probability_truth_or_lies :
  (probability_truth + probability_lies - probability_both) = 0.40 :=
by
  sorry

end probability_truth_or_lies_l441_44166


namespace interest_rate_increase_l441_44159

theorem interest_rate_increase (P : ℝ) (A1 A2 : ℝ) (T : ℝ) (R1 R2 : ℝ) (percentage_increase : ℝ) :
  P = 500 → A1 = 600 → A2 = 700 → T = 2 → 
  (A1 - P) = P * R1 * T →
  (A2 - P) = P * R2 * T →
  percentage_increase = (R2 - R1) / R1 * 100 →
  percentage_increase = 100 :=
by sorry

end interest_rate_increase_l441_44159


namespace initial_oranges_per_rupee_l441_44175

theorem initial_oranges_per_rupee (loss_rate_gain_rate cost_rate : ℝ) (initial_oranges : ℤ) : 
  loss_rate_gain_rate = 0.92 ∧ cost_rate = 18.4 ∧ 1.25 * cost_rate = 1.25 * 0.92 * (initial_oranges : ℝ) →
  initial_oranges = 14 := by
  sorry

end initial_oranges_per_rupee_l441_44175


namespace inequality_solution_set_l441_44140

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + (a-1)*x^2

theorem inequality_solution_set (a : ℝ) (ha : ∀ x : ℝ, f x a = -f (-x) a) :
  {x : ℝ | f (a*x) a > f (a-x) a} = {x : ℝ | x > 1/2} :=
by
  sorry

end inequality_solution_set_l441_44140


namespace upstream_travel_time_l441_44110

-- Define the given conditions
def downstream_time := 1 -- 1 hour
def stream_speed := 3 -- 3 kmph
def boat_speed_still_water := 15 -- 15 kmph

-- Compute the downstream speed
def downstream_speed : Nat := boat_speed_still_water + stream_speed

-- Compute the distance covered downstream
def distance_downstream : Nat := downstream_speed * downstream_time

-- Compute the upstream speed
def upstream_speed : Nat := boat_speed_still_water - stream_speed

-- The goal is to prove the time it takes to cover the distance upstream is 1.5 hours
theorem upstream_travel_time : (distance_downstream : Real) / upstream_speed = 1.5 := by
  sorry

end upstream_travel_time_l441_44110


namespace initial_shirts_count_l441_44183

theorem initial_shirts_count 
  (S T x : ℝ)
  (h1 : 2 * S + x * T = 1600)
  (h2 : S + 6 * T = 1600)
  (h3 : 12 * T = 2400) :
  x = 4 :=
by
  sorry

end initial_shirts_count_l441_44183


namespace difference_of_smallest_integers_l441_44158

theorem difference_of_smallest_integers (n_1 n_2: ℕ) (h1 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_1 > 1 ∧ n_1 % k = 1)) (h2 : ∀ k, 2 ≤ k ∧ k ≤ 6 → (n_2 > 1 ∧ n_2 % k = 1)) (h_smallest : n_1 = 61) (h_second_smallest : n_2 = 121) : n_2 - n_1 = 60 :=
by
  sorry

end difference_of_smallest_integers_l441_44158


namespace box_volume_correct_l441_44131

def volume_of_box (x : ℝ) : ℝ := (16 - 2 * x) * (12 - 2 * x) * x

theorem box_volume_correct {x : ℝ} (h1 : 1 ≤ x) (h2 : x ≤ 3) : 
  volume_of_box x = 4 * x^3 - 56 * x^2 + 192 * x := 
by 
  unfold volume_of_box 
  sorry

end box_volume_correct_l441_44131


namespace manufacturing_cost_eq_210_l441_44100

theorem manufacturing_cost_eq_210 (transport_cost : ℝ) (shoecount : ℕ) (selling_price : ℝ) (gain : ℝ) (M : ℝ) :
  transport_cost = 500 / 100 →
  shoecount = 100 →
  selling_price = 258 →
  gain = 0.20 →
  M = (selling_price / (1 + gain)) - (transport_cost) :=
by
  intros
  sorry

end manufacturing_cost_eq_210_l441_44100


namespace find_first_term_l441_44157

open Int

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_first_term
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 2 = 1)
  (h_a4_a10 : a 3 + a 9 = 18) :
  a 0 = -3 :=
by
  sorry

end find_first_term_l441_44157


namespace find_triangle_value_l441_44164

theorem find_triangle_value 
  (triangle : ℕ)
  (h_units : (triangle + 3) % 7 = 2)
  (h_tens : (1 + 4 + triangle) % 7 = 4)
  (h_hundreds : (2 + triangle + 1) % 7 = 2)
  (h_thousands : 3 + 0 + 1 = 4) :
  triangle = 6 :=
sorry

end find_triangle_value_l441_44164


namespace find_x_l441_44104

theorem find_x (x : ℝ) (h : 49 / x = 700) : x = 0.07 :=
sorry

end find_x_l441_44104


namespace incorrect_conclusion_l441_44114

noncomputable def quadratic (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

theorem incorrect_conclusion (m : ℝ) (hx : m - 2 = 0) :
  ¬(∀ x : ℝ, quadratic m x = 2 ↔ x = 2) :=
by
  sorry

end incorrect_conclusion_l441_44114


namespace problem_statement_l441_44163

def S : ℤ := (-2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19)

theorem problem_statement (hS : S = -2^20 + 4) : 2 - 2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19 + 2^20 = 6 :=
by
  sorry

end problem_statement_l441_44163


namespace problem_m_n_sum_l441_44195

theorem problem_m_n_sum (m n : ℕ) 
  (h1 : m^2 + n^2 = 3789) 
  (h2 : Nat.gcd m n + Nat.lcm m n = 633) : 
  m + n = 87 :=
sorry

end problem_m_n_sum_l441_44195


namespace area_of_triangle_l441_44143

open Real

theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (h1 : sin A = sqrt 3 * sin C)
                        (h2 : B = π / 6) (h3 : b = 2) :
    1 / 2 * a * c * sin B = sqrt 3 :=
by
  sorry

end area_of_triangle_l441_44143


namespace calculate_a3_l441_44173

theorem calculate_a3 (S : ℕ → ℕ) (a : ℕ → ℕ) (h1 : ∀ n, S n = 2^n - 1) (h2 : ∀ n, a n = S n - S (n-1)) : 
  a 3 = 4 :=
by
  sorry

end calculate_a3_l441_44173


namespace point_in_which_quadrant_l441_44168

noncomputable def quadrant_of_point (x y : ℝ) : String :=
if (x > 0) ∧ (y > 0) then
    "First"
else if (x < 0) ∧ (y > 0) then
    "Second"
else if (x < 0) ∧ (y < 0) then
    "Third"
else if (x > 0) ∧ (y < 0) then
    "Fourth"
else
    "On Axis"

theorem point_in_which_quadrant (α : ℝ) (h1 : π / 2 < α) (h2 : α < π) : quadrant_of_point (Real.sin α) (Real.cos α) = "Fourth" :=
by {
    sorry
}

end point_in_which_quadrant_l441_44168


namespace bus_driver_total_hours_l441_44139

variables (R OT : ℕ)

-- Conditions
def regular_rate := 16
def overtime_rate := 28
def max_regular_hours := 40
def total_compensation := 864

-- Proof goal: total hours worked is 48
theorem bus_driver_total_hours :
  (regular_rate * R + overtime_rate * OT = total_compensation) →
  (R ≤ max_regular_hours) →
  (R + OT = 48) :=
by
  sorry

end bus_driver_total_hours_l441_44139


namespace smallest_possible_a_l441_44190

theorem smallest_possible_a (a b c : ℤ) (h1 : a < b) (h2 : b < c)
  (h3 : 2 * b = a + c) (h4 : a^2 = c * b) : a = 1 :=
by
  sorry

end smallest_possible_a_l441_44190


namespace find_second_half_profit_l441_44153

variable (P : ℝ)
variable (profit_difference total_annual_profit : ℝ)
variable (h_difference : profit_difference = 2750000)
variable (h_total : total_annual_profit = 3635000)

theorem find_second_half_profit (h_eq : P + (P + profit_difference) = total_annual_profit) : 
  P = 442500 :=
by
  rw [h_difference, h_total] at h_eq
  sorry

end find_second_half_profit_l441_44153


namespace domain_of_composite_l441_44128

theorem domain_of_composite (f : ℝ → ℝ) (x : ℝ) (hf : ∀ y, (0 ≤ y ∧ y ≤ 1) → f y = f y) :
  (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) → (0 ≤ x ∧ x ≤ 1) →
  0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ 2*x ∧ 2*x ≤ 1 ∧ 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 →
  0 ≤ x ∧ x ≤ 1/2 :=
by
  intro h1 h2 h3 h4
  have h5: 0 ≤ 2*x ∧ 2*x ≤ 1 := sorry
  have h6: 0 ≤ x + 1/3 ∧ x + 1/3 ≤ 1 := sorry
  sorry

end domain_of_composite_l441_44128


namespace max_value_of_expression_l441_44141

theorem max_value_of_expression :
  ∀ r : ℝ, -3 * r^2 + 30 * r + 8 ≤ 83 :=
by
  -- Proof needed
  sorry

end max_value_of_expression_l441_44141


namespace train_passes_jogger_time_l441_44177

noncomputable def jogger_speed_km_per_hr : ℝ := 9
noncomputable def train_speed_km_per_hr : ℝ := 75
noncomputable def jogger_head_start_m : ℝ := 500
noncomputable def train_length_m : ℝ := 300

noncomputable def km_per_hr_to_m_per_s (speed: ℝ) : ℝ := speed * (1000 / 3600)

noncomputable def jogger_speed_m_per_s := km_per_hr_to_m_per_s jogger_speed_km_per_hr
noncomputable def train_speed_m_per_s := km_per_hr_to_m_per_s train_speed_km_per_hr

noncomputable def relative_speed_m_per_s := train_speed_m_per_s - jogger_speed_m_per_s

noncomputable def total_distance_to_cover_m := jogger_head_start_m + train_length_m

theorem train_passes_jogger_time :
  let time_to_pass := total_distance_to_cover_m / relative_speed_m_per_s
  abs (time_to_pass - 43.64) < 0.01 :=
by
  sorry

end train_passes_jogger_time_l441_44177


namespace negation_of_P_l441_44117

variable (x : ℝ)

def P : Prop := ∀ x : ℝ, x^2 + 2*x + 3 ≥ 0

theorem negation_of_P : ¬P ↔ ∃ x : ℝ, x^2 + 2*x + 3 < 0 :=
by sorry

end negation_of_P_l441_44117


namespace pills_in_a_week_l441_44120

def insulin_pills_per_day : Nat := 2
def blood_pressure_pills_per_day : Nat := 3
def anticonvulsant_pills_per_day : Nat := 2 * blood_pressure_pills_per_day

def total_pills_per_day : Nat := insulin_pills_per_day + blood_pressure_pills_per_day + anticonvulsant_pills_per_day

theorem pills_in_a_week : total_pills_per_day * 7 = 77 := by
  sorry

end pills_in_a_week_l441_44120


namespace percentage_error_in_side_measurement_l441_44148

theorem percentage_error_in_side_measurement :
  (forall (S S' : ℝ) (A A' : ℝ), 
    A = S^2 ∧ A' = S'^2 ∧ (A' - A) / A * 100 = 25.44 -> 
    (S' - S) / S * 100 = 12.72) :=
by
  intros S S' A A' h
  sorry

end percentage_error_in_side_measurement_l441_44148


namespace system1_solution_l441_44111

theorem system1_solution (x y : ℝ) (h1 : 4 * x - 3 * y = 1) (h2 : 3 * x - 2 * y = -1) : x = -5 ∧ y = 7 :=
sorry

end system1_solution_l441_44111


namespace inequality_l441_44151

theorem inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (a / (b.sqrt) + b / (a.sqrt)) ≥ (a.sqrt + b.sqrt) :=
by
  sorry

end inequality_l441_44151


namespace min_degree_g_l441_44189

theorem min_degree_g (f g h : Polynomial ℝ) (hf : f.degree = 8) (hh : h.degree = 9) (h_eq : 3 * f + 4 * g = h) : g.degree ≥ 9 :=
sorry

end min_degree_g_l441_44189


namespace selection_methods_correct_l441_44126

-- Define the number of students in each year
def first_year_students : ℕ := 3
def second_year_students : ℕ := 5
def third_year_students : ℕ := 4

-- Define the total number of different selection methods
def total_selection_methods : ℕ := first_year_students + second_year_students + third_year_students

-- Lean statement to prove the question is equivalent to the answer
theorem selection_methods_correct :
  total_selection_methods = 12 := by
  sorry

end selection_methods_correct_l441_44126


namespace investment_simple_compound_l441_44112

theorem investment_simple_compound (P y : ℝ) 
    (h1 : 600 = P * y * 2 / 100)
    (h2 : 615 = P * (1 + y/100)^2 - P) : 
    P = 285.71 :=
by
    sorry

end investment_simple_compound_l441_44112
