import Mathlib

namespace terminal_side_in_second_quadrant_l312_31206

theorem terminal_side_in_second_quadrant (α : ℝ) 
  (hcos : Real.cos α = -1/5) 
  (hsin : Real.sin α = 2 * Real.sqrt 6 / 5) : 
  (π / 2 < α ∧ α < π) :=
by
  sorry

end terminal_side_in_second_quadrant_l312_31206


namespace total_pencils_proof_l312_31290

noncomputable def total_pencils (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ) : ℕ :=
  Asaf_pencils + Alexander_pencils

theorem total_pencils_proof :
  ∀ (Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff : ℕ),
  Asaf_age = 50 →
  Alexander_age = 140 - Asaf_age →
  total_age_diff = Alexander_age - Asaf_age →
  Asaf_pencils = 2 * total_age_diff →
  Alexander_pencils = Asaf_pencils + 60 →
  total_pencils Asaf_age Alexander_age Asaf_pencils Alexander_pencils total_age_diff = 220 :=
by
  intros
  sorry

end total_pencils_proof_l312_31290


namespace fraction_of_orange_juice_is_correct_l312_31293

noncomputable def fraction_of_orange_juice_in_mixture (V1 V2 juice1_ratio juice2_ratio : ℚ) : ℚ :=
  let juice1 := V1 * juice1_ratio
  let juice2 := V2 * juice2_ratio
  let total_juice := juice1 + juice2
  let total_volume := V1 + V2
  total_juice / total_volume

theorem fraction_of_orange_juice_is_correct :
  fraction_of_orange_juice_in_mixture 800 500 (1/4) (1/3) = 7 / 25 :=
by sorry

end fraction_of_orange_juice_is_correct_l312_31293


namespace find_a_l312_31257

theorem find_a (a : ℝ) (α : ℝ) (h1 : ∃ (y : ℝ), (a, y) = (a, -2))
(h2 : Real.tan (π + α) = 1 / 3) : a = -6 :=
sorry

end find_a_l312_31257


namespace compound_proposition_truth_l312_31264

theorem compound_proposition_truth (p q : Prop) (h1 : ¬p ∨ ¬q = False) : (p ∧ q) ∧ (p ∨ q) :=
by
  sorry

end compound_proposition_truth_l312_31264


namespace plane_tiled_squares_triangles_percentage_l312_31268

theorem plane_tiled_squares_triangles_percentage :
    (percent_triangle_area : ℚ) = 625 / 10000 := sorry

end plane_tiled_squares_triangles_percentage_l312_31268


namespace straight_flush_probability_l312_31295

open Classical

noncomputable def number_of_possible_hands : ℕ := Nat.choose 52 5

noncomputable def number_of_straight_flushes : ℕ := 40 

noncomputable def probability_of_straight_flush : ℚ := number_of_straight_flushes / number_of_possible_hands

theorem straight_flush_probability :
  probability_of_straight_flush = 1 / 64974 := by
  sorry

end straight_flush_probability_l312_31295


namespace remainder_sum_mod_l312_31265

theorem remainder_sum_mod (a b c d e : ℕ)
  (h₁ : a = 17145)
  (h₂ : b = 17146)
  (h₃ : c = 17147)
  (h₄ : d = 17148)
  (h₅ : e = 17149)
  : (a + b + c + d + e) % 10 = 5 := by
  sorry

end remainder_sum_mod_l312_31265


namespace matrix_multiplication_problem_l312_31259

variable {A B : Matrix (Fin 2) (Fin 2) ℝ}

theorem matrix_multiplication_problem 
  (h1 : A + B = A * B)
  (h2 : A * B = ![![5, 2], ![-2, 4]]) :
  B * A = ![![5, 2], ![-2, 4]] :=
sorry

end matrix_multiplication_problem_l312_31259


namespace carmen_candle_usage_l312_31285

-- Define the duration a candle lasts when burned for 1 hour every night.
def candle_duration_1_hour_per_night : ℕ := 8

-- Define the number of hours Carmen burns a candle each night.
def hours_burned_per_night : ℕ := 2

-- Define the number of nights over which we want to calculate the number of candles needed.
def number_of_nights : ℕ := 24

-- We want to show that given these conditions, Carmen will use 6 candles.
theorem carmen_candle_usage :
  (number_of_nights / (candle_duration_1_hour_per_night / hours_burned_per_night)) = 6 :=
by
  sorry

end carmen_candle_usage_l312_31285


namespace machine_x_widgets_per_hour_l312_31273

-- Definitions of the variables and conditions
variable (Wx Wy Tx Ty: ℝ)
variable (h1: Tx = Ty + 60)
variable (h2: Wy = 1.20 * Wx)
variable (h3: Wx * Tx = 1080)
variable (h4: Wy * Ty = 1080)

-- Statement of the problem to prove
theorem machine_x_widgets_per_hour : Wx = 3 := by
  sorry

end machine_x_widgets_per_hour_l312_31273


namespace box_dimensions_l312_31249

theorem box_dimensions (a b c : ℕ) (h1 : a + c = 17) (h2 : a + b = 13) (h3 : b + c = 20) :
  a = 5 ∧ b = 8 ∧ c = 12 :=
by
  -- We assume the proof is correct based on given conditions
  sorry

end box_dimensions_l312_31249


namespace intersection_sets_m_n_l312_31229

theorem intersection_sets_m_n :
  let M := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
  let N := { x : ℝ | x > 0 }
  M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_sets_m_n_l312_31229


namespace area_of_intersection_is_zero_l312_31260

-- Define the circles
def circle1 (x y : ℝ) := x^2 + y^2 = 16
def circle2 (x y : ℝ) := (x - 3)^2 + y^2 = 9

-- Define the theorem to prove
theorem area_of_intersection_is_zero : 
  ∃ x1 y1 x2 y2 : ℝ,
    circle1 x1 y1 ∧ circle2 x1 y1 ∧
    circle1 x2 y2 ∧ circle2 x2 y2 ∧
    x1 = x2 ∧ y1 = -y2 → 
    0 = 0 :=
by
  sorry -- proof goes here

end area_of_intersection_is_zero_l312_31260


namespace find_common_remainder_l312_31271

theorem find_common_remainder :
  ∃ (d : ℕ), 100 ≤ d ∧ d ≤ 999 ∧ (312837 % d = 96) ∧ (310650 % d = 96) :=
sorry

end find_common_remainder_l312_31271


namespace find_other_outlet_rate_l312_31255

open Real

-- Definitions based on conditions
def V : ℝ := 20 * 1728   -- volume of the tank in cubic inches
def r1 : ℝ := 5          -- rate of inlet pipe in cubic inches/min
def r2 : ℝ := 8          -- rate of one outlet pipe in cubic inches/min
def t : ℝ := 2880        -- time in minutes required to empty the tank
 
-- Mathematically equivalent proof statement
theorem find_other_outlet_rate (x : ℝ) : 
  -- Given conditions
  V = 34560 →
  r1 = 5 →
  r2 = 8 →
  t = 2880 →
  -- Statement to prove
  V = (r2 + x - r1) * t → x = 9 :=
by
  intro hV hr1 hr2 ht hEq
  sorry

end find_other_outlet_rate_l312_31255


namespace positive_solution_x_l312_31209

theorem positive_solution_x (x y z : ℝ) (h1 : x * y = 8 - x - 4 * y) (h2 : y * z = 12 - 3 * y - 6 * z) (h3 : x * z = 40 - 5 * x - 2 * z) (hy : y = 3) (hz : z = -1) : x = 6 :=
by
  sorry

end positive_solution_x_l312_31209


namespace time_after_1876_minutes_l312_31238

-- Define the structure for Time
structure Time where
  hour : Nat
  minute : Nat

-- Define a function to add minutes to a time
noncomputable def add_minutes (t : Time) (m : Nat) : Time :=
  let total_minutes := t.minute + m
  let additional_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let new_hour := (t.hour + additional_hours) % 24
  { hour := new_hour, minute := remaining_minutes }

-- Definition of the starting time
def three_pm : Time := { hour := 15, minute := 0 }

-- The main theorem statement
theorem time_after_1876_minutes : add_minutes three_pm 1876 = { hour := 10, minute := 16 } :=
  sorry

end time_after_1876_minutes_l312_31238


namespace min_dot_product_on_hyperbola_l312_31251

open Real

theorem min_dot_product_on_hyperbola :
  ∀ (P : ℝ × ℝ), (P.1 ≥ 1 ∧ P.1^2 - (P.2^2) / 3 = 1) →
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  ∃ m : ℝ, m = -2 ∧ PA1.1 * PF2.1 + PA1.2 * PF2.2 = m :=
by
  intros P h
  let PA1 := (P.1 + 1, P.2)
  let PF2 := (P.1 - 2, P.2)
  use -2
  sorry

end min_dot_product_on_hyperbola_l312_31251


namespace mul_exponents_l312_31288

theorem mul_exponents (a : ℝ) : ((-2 * a) ^ 2) * (a ^ 4) = 4 * a ^ 6 := by
  sorry

end mul_exponents_l312_31288


namespace average_cost_of_testing_l312_31252

theorem average_cost_of_testing (total_machines : Nat) (faulty_machines : Nat) (cost_per_test : Nat) 
  (h_total : total_machines = 5) (h_faulty : faulty_machines = 2) (h_cost : cost_per_test = 1000) :
  (2000 * (2 / 5 * 1 / 4) + 3000 * (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3) + 
  4000 * (1 - (2 / 5 * 1 / 4) - (2 / 5 * 3 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3 + 3 / 5 * 2 / 4 * 1 / 3))) = 3500 :=
  by
  sorry

end average_cost_of_testing_l312_31252


namespace sin_alpha_value_l312_31289

theorem sin_alpha_value
  (α : ℝ)
  (h₀ : 0 < α)
  (h₁ : α < Real.pi)
  (h₂ : Real.sin (α / 2) = Real.sqrt 3 / 3) :
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

end sin_alpha_value_l312_31289


namespace negation_equivalence_l312_31256

theorem negation_equivalence (x : ℝ) : ¬(∀ x, x^2 - x + 2 ≥ 0) ↔ ∃ x, x^2 - x + 2 < 0 :=
sorry

end negation_equivalence_l312_31256


namespace circle_intersection_range_l312_31298

noncomputable def circleIntersectionRange (r : ℝ) : Prop :=
  1 < r ∧ r < 11

theorem circle_intersection_range (r : ℝ) (h1 : r > 0) :
  (∃ x y : ℝ, x^2 + y^2 = r^2 ∧ (x + 3)^2 + (y - 4)^2 = 36) ↔ circleIntersectionRange r :=
by
  sorry

end circle_intersection_range_l312_31298


namespace identify_counterfeit_coin_correct_l312_31267

noncomputable def identify_counterfeit_coin (coins : Fin 8 → ℝ) : ℕ :=
  sorry

theorem identify_counterfeit_coin_correct (coins : Fin 8 → ℝ) (h_fake : 
  ∃ i : Fin 8, ∀ j : Fin 8, j ≠ i → coins i > coins j) : 
  ∃ i : Fin 8, identify_counterfeit_coin coins = i ∧ ∀ j : Fin 8, j ≠ i → coins i > coins j :=
by
  sorry

end identify_counterfeit_coin_correct_l312_31267


namespace stamps_in_album_l312_31236

theorem stamps_in_album (n : ℕ) : 
  n % 2 = 1 ∧ n % 3 = 2 ∧ n % 4 = 3 ∧ n % 5 = 4 ∧ 
  n % 6 = 5 ∧ n % 7 = 6 ∧ n % 8 = 7 ∧ n % 9 = 8 ∧ 
  n % 10 = 9 ∧ n < 3000 → n = 2519 :=
by
  sorry

end stamps_in_album_l312_31236


namespace find_a_l312_31270

theorem find_a (a : ℤ) (h1 : 0 ≤ a ∧ a ≤ 20) (h2 : (56831742 - a) % 17 = 0) : a = 2 :=
by
  sorry

end find_a_l312_31270


namespace matrix_characteristic_eq_l312_31241

noncomputable def B : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![1, 2, 2], ![2, 1, 2], ![2, 2, 1]]

theorem matrix_characteristic_eq :
  ∃ (a b c : ℚ), a = -6 ∧ b = -12 ∧ c = -18 ∧ 
  (B ^ 3 + a • (B ^ 2) + b • B + c • (1 : Matrix (Fin 3) (Fin 3) ℚ) = 0) :=
by
  sorry

end matrix_characteristic_eq_l312_31241


namespace problem_A_inter_complement_B_l312_31242

noncomputable def A : Set ℝ := {x : ℝ | Real.log x / Real.log 2 < 2}
noncomputable def B : Set ℝ := {x : ℝ | (x - 2) / (x - 1) ≥ 0}
noncomputable def complement_B : Set ℝ := {x : ℝ | ¬((x - 2) / (x - 1) ≥ 0)}

theorem problem_A_inter_complement_B : 
  (A ∩ complement_B) = {x : ℝ | 1 < x ∧ x < 2} := by
  sorry

end problem_A_inter_complement_B_l312_31242


namespace dylan_speed_constant_l312_31269

theorem dylan_speed_constant (d t s : ℝ) (h1 : d = 1250) (h2 : t = 25) (h3 : s = d / t) : s = 50 := 
by 
  -- Proof steps will go here
  sorry

end dylan_speed_constant_l312_31269


namespace find_number_l312_31213

theorem find_number (x : ℝ) (h : x = (1 / 3) * x + 120) : x = 180 :=
by
  sorry

end find_number_l312_31213


namespace football_team_total_players_l312_31216

/-- The conditions are:
1. There are some players on a football team.
2. 46 are throwers.
3. All throwers are right-handed.
4. One third of the rest of the team are left-handed.
5. There are 62 right-handed players in total.
And we need to prove that the total number of players on the football team is 70. 
--/

theorem football_team_total_players (P : ℕ) 
  (h_throwers : P >= 46) 
  (h_total_right_handed : 62 = 46 + 2 * (P - 46) / 3)
  (h_remainder_left_handed : 1 * (P - 46) / 3 = (P - 46) / 3) :
  P = 70 :=
by
  sorry

end football_team_total_players_l312_31216


namespace complex_expression_evaluation_l312_31217

-- Definition of the imaginary unit i with property i^2 = -1
def i : ℂ := Complex.I

-- Theorem stating that the given expression equals i
theorem complex_expression_evaluation : i * (1 - i) - 1 = i := by
  -- Proof omitted
  sorry

end complex_expression_evaluation_l312_31217


namespace donation_amount_l312_31292

theorem donation_amount 
  (total_needed : ℕ) (bronze_amount : ℕ) (silver_amount : ℕ) (raised_so_far : ℕ)
  (bronze_families : ℕ) (silver_families : ℕ) (other_family_donation : ℕ)
  (final_push_needed : ℕ) 
  (h1 : total_needed = 750) 
  (h2 : bronze_amount = 25)
  (h3 : silver_amount = 50)
  (h4 : bronze_families = 10)
  (h5 : silver_families = 7)
  (h6 : raised_so_far = 600)
  (h7 : final_push_needed = 50)
  (h8 : raised_so_far = bronze_families * bronze_amount + silver_families * silver_amount)
  (h9 : total_needed - raised_so_far - other_family_donation = final_push_needed) : 
  other_family_donation = 100 :=
by
  sorry

end donation_amount_l312_31292


namespace calculate_inverse_y3_minus_y_l312_31279

theorem calculate_inverse_y3_minus_y
  (i : ℂ) (y : ℂ)
  (h_i : i = Complex.I)
  (h_y : y = (1 + i * Real.sqrt 3) / 2) :
  (1 / (y^3 - y)) = -1/2 + i * (Real.sqrt 3) / 6 :=
by
  sorry

end calculate_inverse_y3_minus_y_l312_31279


namespace box_height_l312_31246

theorem box_height (volume length width : ℝ) (h : ℝ) (h_volume : volume = 315) (h_length : length = 7) (h_width : width = 9) :
  h = 5 :=
by
  -- Proof would go here
  sorry

end box_height_l312_31246


namespace best_fit_slope_eq_l312_31224

theorem best_fit_slope_eq :
  let x1 := 150
  let y1 := 2
  let x2 := 160
  let y2 := 3
  let x3 := 170
  let y3 := 4
  (x2 - x1 = 10 ∧ x3 - x2 = 10) →
  let slope := (x1 - x2) * (y1 - y2) + (x3 - x2) * (y3 - y2) / (x1 - x2)^2 + (x3 - x2)^2
  slope = 1 / 10 :=
sorry

end best_fit_slope_eq_l312_31224


namespace slope_of_line_l312_31284

theorem slope_of_line : ∀ x y : ℝ, 3 * y + 2 * x = 6 * x - 9 → ∃ m b : ℝ, y = m * x + b ∧ m = -4 / 3 :=
by
  -- Sorry to skip proof
  sorry

end slope_of_line_l312_31284


namespace additional_cards_l312_31218

theorem additional_cards (total_cards : ℕ) (complete_decks : ℕ) (cards_per_deck : ℕ) (num_decks : ℕ) 
  (h1 : total_cards = 160) (h2 : num_decks = 3) (h3 : cards_per_deck = 52) :
  total_cards - (num_decks * cards_per_deck) = 4 :=
by
  sorry

end additional_cards_l312_31218


namespace not_divisible_by_1980_divisible_by_1981_l312_31276

open Nat

theorem not_divisible_by_1980 (x : ℕ) : ¬ (2^100 * x - 1) % 1980 = 0 := by
sorry

theorem divisible_by_1981 : ∃ x : ℕ, (2^100 * x - 1) % 1981 = 0 := by
sorry

end not_divisible_by_1980_divisible_by_1981_l312_31276


namespace calculate_ray_grocery_bill_l312_31258

noncomputable def ray_grocery_total_cost : ℝ :=
let hamburger_meat_price := 5.0
let crackers_price := 3.5
let frozen_vegetables_price := 2.0 * 4
let cheese_price := 3.5
let chicken_price := 6.5
let cereal_price := 4.0
let wine_price := 10.0
let cookies_price := 3.0

let discount_hamburger_meat := hamburger_meat_price * 0.10
let discount_crackers := crackers_price * 0.10
let discount_frozen_vegetables := frozen_vegetables_price * 0.10
let discount_cheese := cheese_price * 0.05
let discount_chicken := chicken_price * 0.05
let discount_wine := wine_price * 0.15

let discounted_hamburger_meat_price := hamburger_meat_price - discount_hamburger_meat
let discounted_crackers_price := crackers_price - discount_crackers
let discounted_frozen_vegetables_price := frozen_vegetables_price - discount_frozen_vegetables
let discounted_cheese_price := cheese_price - discount_cheese
let discounted_chicken_price := chicken_price - discount_chicken
let discounted_wine_price := wine_price - discount_wine

let total_discounted_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  discounted_wine_price +
  cookies_price

let food_items_total_price :=
  discounted_hamburger_meat_price +
  discounted_crackers_price +
  discounted_frozen_vegetables_price +
  discounted_cheese_price +
  discounted_chicken_price +
  cereal_price +
  cookies_price

let food_sales_tax := food_items_total_price * 0.06
let wine_sales_tax := discounted_wine_price * 0.09

let total_with_tax := total_discounted_price + food_sales_tax + wine_sales_tax

total_with_tax

theorem calculate_ray_grocery_bill :
  ray_grocery_total_cost = 42.51 :=
sorry

end calculate_ray_grocery_bill_l312_31258


namespace sector_area_correct_l312_31231

-- Definitions based on the conditions
def sector_perimeter := 16 -- cm
def central_angle := 2 -- radians
def radius := 4 -- The radius computed from perimeter condition

-- Lean 4 statement to prove the equivalent math problem
theorem sector_area_correct : ∃ (s : ℝ), 
  (∀ (r : ℝ), (2 * r + r * central_angle = sector_perimeter → r = 4) → 
  (s = (1 / 2) * central_angle * (radius) ^ 2) → 
  s = 16) :=
by 
  sorry

end sector_area_correct_l312_31231


namespace dog_total_distance_l312_31228

-- Define the conditions
def distance_between_A_and_B : ℝ := 100
def speed_A : ℝ := 6
def speed_B : ℝ := 4
def speed_dog : ℝ := 10

-- Define the statement we want to prove
theorem dog_total_distance : ∀ t : ℝ, (speed_A + speed_B) * t = distance_between_A_and_B → speed_dog * t = 100 :=
by
  intro t
  intro h
  sorry

end dog_total_distance_l312_31228


namespace meeting_time_final_time_statement_l312_31215

-- Define the speeds and distance as given conditions
def brodie_speed : ℝ := 50
def ryan_speed : ℝ := 40
def initial_distance : ℝ := 120

-- Define what we know about their meeting time and validate it mathematically
theorem meeting_time :
  (initial_distance / (brodie_speed + ryan_speed)) = 4 / 3 := sorry

-- Assert the time in minutes for completeness
noncomputable def time_in_minutes : ℝ := ((4 / 3) * 60)

-- Assert final statement matching the answer in hours and minutes
theorem final_time_statement :
  time_in_minutes = 80 := sorry

end meeting_time_final_time_statement_l312_31215


namespace certain_number_is_1_l312_31296

theorem certain_number_is_1 (z : ℕ) (hz : z % 4 = 0) :
  ∃ n : ℕ, (z * (6 + z) + n) % 2 = 1 ∧ n = 1 :=
by
  sorry

end certain_number_is_1_l312_31296


namespace total_equipment_cost_l312_31272

-- Define the cost of each piece of equipment
def jersey_cost : ℝ := 25
def shorts_cost : ℝ := 15.2
def socks_cost : ℝ := 6.8

-- Define the number of players
def players : ℕ := 16

-- Define the total cost of equipment for one player
def equipment_cost_per_player : ℝ := jersey_cost + shorts_cost + socks_cost

-- Define the total cost for all players
def total_cost : ℝ := players * equipment_cost_per_player

-- The proof problem to be stated:
theorem total_equipment_cost (jc sc k p : ℝ) (n : ℕ) :
  jc = 25 ∧ sc = 15.2 ∧ k = 6.8 ∧ p = 16 →
  total_cost = 752 :=
by
  intro h
  rcases h with ⟨hc1, hc2, hc3, hc4⟩
  simp [total_cost, equipment_cost_per_player, hc1, hc2, hc3, hc4]
  exact sorry

end total_equipment_cost_l312_31272


namespace max_large_sculptures_l312_31208

theorem max_large_sculptures (x y : ℕ) (h1 : 1 * x = x) 
  (h2 : 3 * y = y + y + y) 
  (h3 : ∃ n, n = (x + y) / 2) 
  (h4 : x + 3 * y + (x + y) / 2 ≤ 30) 
  (h5 : x > y) : 
  y ≤ 4 := 
sorry

end max_large_sculptures_l312_31208


namespace sequence_term_306_l312_31286

theorem sequence_term_306 (a1 a2 : ℤ) (r : ℤ) (n : ℕ) (h1 : a1 = 7) (h2 : a2 = -7) (h3 : r = -1) (h4 : a2 = r * a1) : 
  ∃ a306 : ℤ, a306 = -7 ∧ a306 = a1 * r^305 :=
by
  use -7
  sorry

end sequence_term_306_l312_31286


namespace part1_part2_l312_31247

noncomputable def f (x k : ℝ) : ℝ := (x - 1) * Real.exp x - k * x^2 + 2

theorem part1 {x : ℝ} (hx : x = 0) : 
    f x 0 = 1 :=
by
  sorry

theorem part2 {x k : ℝ} (hx : 0 ≤ x) (hxf : f x k ≥ 1) : 
    k ≤ 1 / 2 :=
by
  sorry

end part1_part2_l312_31247


namespace recolor_possible_l312_31240

theorem recolor_possible (cell_color : Fin 50 → Fin 50 → Fin 100)
  (H1 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color i (j+1) = l ∧ k ≠ l ∧ j < 49)
  (H2 : ∀ i j, ∃ k l, cell_color i j = k ∧ cell_color (i+1) j = l ∧ k ≠ l ∧ i < 49) :
  ∃ c1 c2, (c1 ≠ c2) ∧
  ∀ i j, (cell_color i j = c1 → cell_color i j = c2 ∨ ∀ k l, (cell_color k l = c1 → cell_color k l ≠ c2)) :=
  by
  sorry

end recolor_possible_l312_31240


namespace coprime_divisors_imply_product_divisor_l312_31222

theorem coprime_divisors_imply_product_divisor 
  (a b n : ℕ) (h_coprime : Nat.gcd a b = 1)
  (h_a_div_n : a ∣ n) (h_b_div_n : b ∣ n) : a * b ∣ n :=
by
  sorry

end coprime_divisors_imply_product_divisor_l312_31222


namespace find_set_A_find_range_a_l312_31274

-- Define the universal set and the complement condition for A
def universal_set : Set ℝ := {x | true}
def complement_A : Set ℝ := {x | 2 * x^2 - 3 * x - 2 > 0}

-- Define the set A
def set_A : Set ℝ := {x | -1/2 ≤ x ∧ x ≤ 2}

-- Define the set C
def set_C (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + a ≤ 0}

-- Define the proof problem for part (1)
theorem find_set_A : { x | -1 / 2 ≤ x ∧ x ≤ 2 } = { x | ¬ (2 * x^2 - 3 * x - 2 > 0) } :=
by
  sorry

-- Define the proof problem for part (2)
theorem find_range_a (a : ℝ) (C_ne_empty : (set_C a).Nonempty) (sufficient_not_necessary : ∀ x, x ∈ set_C a → x ∈ set_A → x ∈ set_A) :
  a ∈ Set.Icc (-1/8 : ℝ) 0 ∪ Set.Icc 1 (4/3 : ℝ) :=
by
  sorry

end find_set_A_find_range_a_l312_31274


namespace gcd_is_13_eval_at_neg1_l312_31275

-- Define the GCD problem
def gcd_117_182 : ℕ := gcd 117 182

-- Define the polynomial evaluation problem
def f (x : ℝ) : ℝ := 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

-- Formalize the statements to be proved
theorem gcd_is_13 : gcd_117_182 = 13 := 
by sorry

theorem eval_at_neg1 : f (-1) = 12 := 
by sorry

end gcd_is_13_eval_at_neg1_l312_31275


namespace parabola_equation_l312_31207

theorem parabola_equation (a b c : ℝ) (h1 : a^2 = 3) (h2 : b^2 = 1) (h3 : c^2 = a^2 + b^2) : 
  (c = 2) → (vertex = 0) → (focus = 2) → ∀ x y, y^2 = 16 * x := 
by 
  sorry

end parabola_equation_l312_31207


namespace min_value_expression_is_4_l312_31262

noncomputable def min_value_expression (x : ℝ) : ℝ :=
(3 * x^2 + 6 * x + 5) / (0.5 * x^2 + x + 1)

theorem min_value_expression_is_4 : ∃ x : ℝ, min_value_expression x = 4 :=
sorry

end min_value_expression_is_4_l312_31262


namespace system_of_equations_correct_l312_31297

def question_statement (x y : ℕ) : Prop :=
  x + y = 12 ∧ 6 * x = 3 * 4 * y

theorem system_of_equations_correct
  (x y : ℕ)
  (h1 : x + y = 12)
  (h2 : 6 * x = 3 * 4 * y)
: question_statement x y :=
by
  unfold question_statement
  exact ⟨h1, h2⟩

end system_of_equations_correct_l312_31297


namespace tan_theta_cos_sin_id_l312_31277

theorem tan_theta_cos_sin_id (θ : ℝ) (h : Real.tan θ = 3) :
  (1 + Real.cos θ) / Real.sin θ + Real.sin θ / (1 - Real.cos θ) =
  (17 * (Real.sqrt 10 + 1)) / 24 :=
by
  sorry

end tan_theta_cos_sin_id_l312_31277


namespace bread_slices_l312_31237

theorem bread_slices (c : ℕ) (cost_each_slice_in_cents : ℕ)
  (total_paid_in_cents : ℕ) (change_in_cents : ℕ) (n : ℕ) (slices_per_loaf : ℕ) :
  c = 3 →
  cost_each_slice_in_cents = 40 →
  total_paid_in_cents = 2 * 2000 →
  change_in_cents = 1600 →
  total_paid_in_cents - change_in_cents = n * cost_each_slice_in_cents →
  n = c * slices_per_loaf →
  slices_per_loaf = 20 :=
by sorry

end bread_slices_l312_31237


namespace sum_midpoints_x_coordinates_is_15_l312_31203

theorem sum_midpoints_x_coordinates_is_15 :
  ∀ (a b : ℝ), a + 2 * b = 15 → 
  (a + 2 * b) = 15 :=
by
  intros a b h
  sorry

end sum_midpoints_x_coordinates_is_15_l312_31203


namespace old_manufacturing_cost_l312_31214

theorem old_manufacturing_cost (P : ℝ) :
  (50 : ℝ) = P * 0.50 →
  (0.65 : ℝ) * P = 65 :=
by
  intros hp₁
  -- Proof omitted
  sorry

end old_manufacturing_cost_l312_31214


namespace find_number_of_students_l312_31226

theorem find_number_of_students 
    (N T : ℕ) 
    (h1 : T = 80 * N)
    (h2 : (T - 350) / (N - 5) = 90) : 
    N = 10 :=
sorry

end find_number_of_students_l312_31226


namespace max_principals_l312_31245

theorem max_principals (n_years term_length max_principals: ℕ) 
  (h1 : n_years = 12) 
  (h2 : term_length = 4)
  (h3 : max_principals = 4): 
  (∃ p : ℕ, p = max_principals) :=
by
  sorry

end max_principals_l312_31245


namespace parabola_c_value_l312_31278

theorem parabola_c_value (b c : ℝ) 
  (h1 : 20 = 2*(-2)^2 + b*(-2) + c) 
  (h2 : 28 = 2*2^2 + b*2 + c) : 
  c = 16 :=
by
  sorry

end parabola_c_value_l312_31278


namespace twenty_five_percent_M_eq_thirty_five_percent_1504_l312_31204

theorem twenty_five_percent_M_eq_thirty_five_percent_1504 (M : ℝ) : 
  0.25 * M = 0.35 * 1504 → M = 2105.6 :=
by
  sorry

end twenty_five_percent_M_eq_thirty_five_percent_1504_l312_31204


namespace largest_common_value_less_than_1000_l312_31283

theorem largest_common_value_less_than_1000 :
  ∃ a : ℕ, 
    (∃ n : ℕ, a = 4 + 5 * n) ∧
    (∃ m : ℕ, a = 5 + 10 * m) ∧
    a % 4 = 1 ∧
    a < 1000 ∧
    (∀ b : ℕ, 
      (∃ n : ℕ, b = 4 + 5 * n) ∧
      (∃ m : ℕ, b = 5 + 10 * m) ∧
      b % 4 = 1 ∧
      b < 1000 → 
      b ≤ a) ∧ 
    a = 989 :=
by
  sorry

end largest_common_value_less_than_1000_l312_31283


namespace distance_to_cheaper_gas_station_l312_31235

-- Define the conditions
def miles_per_gallon : ℕ := 3
def initial_gallons : ℕ := 12
def additional_gallons : ℕ := 18

-- Define the question and proof statement
theorem distance_to_cheaper_gas_station : 
  (initial_gallons + additional_gallons) * miles_per_gallon = 90 := by
  sorry

end distance_to_cheaper_gas_station_l312_31235


namespace parabola_distance_l312_31202

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l312_31202


namespace custom_op_evaluation_l312_31232

def custom_op (x y : ℤ) : ℤ := x * y - 3 * x + y

theorem custom_op_evaluation : custom_op 6 5 - custom_op 5 6 = -4 := by
  sorry

end custom_op_evaluation_l312_31232


namespace first_discount_percentage_l312_31299

theorem first_discount_percentage
  (P : ℝ)
  (initial_price final_price : ℝ)
  (second_discount : ℕ)
  (h1 : initial_price = 200)
  (h2 : final_price = 144)
  (h3 : second_discount = 10)
  (h4 : final_price = (P - (second_discount / 100) * P)) :
  (∃ x : ℝ, P = initial_price - (x / 100) * initial_price ∧ x = 20) :=
sorry

end first_discount_percentage_l312_31299


namespace not_right_triangle_l312_31287

/-- In a triangle ABC, with angles A, B, C, the condition A = B = 2 * C does not form a right-angled triangle. -/
theorem not_right_triangle (A B C : ℝ) (h1 : A = B) (h2 : A = 2 * C) (h3 : A + B + C = 180) : 
    ¬(A = 90 ∨ B = 90 ∨ C = 90) := 
by
  sorry

end not_right_triangle_l312_31287


namespace correct_operation_l312_31205

theorem correct_operation {a : ℝ} : (a ^ 6 / a ^ 2 = a ^ 4) :=
by sorry

end correct_operation_l312_31205


namespace Teresa_age_at_Michiko_birth_l312_31250

-- Definitions of the conditions
def Teresa_age_now : ℕ := 59
def Morio_age_now : ℕ := 71
def Morio_age_at_Michiko_birth : ℕ := 38

-- Prove that Teresa was 26 years old when she gave birth to Michiko.
theorem Teresa_age_at_Michiko_birth : 38 - (71 - 59) = 26 := by
  -- Provide the proof here
  sorry

end Teresa_age_at_Michiko_birth_l312_31250


namespace expand_product_l312_31200

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 6) = 2 * x^2 + 6 * x - 36 :=
by sorry

end expand_product_l312_31200


namespace range_of_a_for_two_unequal_roots_l312_31230

theorem range_of_a_for_two_unequal_roots (a : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * Real.log x₁ = x₁ ∧ a * Real.log x₂ = x₂) ↔ a > Real.exp 1 :=
sorry

end range_of_a_for_two_unequal_roots_l312_31230


namespace abs_twice_sub_pi_l312_31291

theorem abs_twice_sub_pi (h : Real.pi < 10) : abs (Real.pi - abs (Real.pi - 10)) = 10 - 2 * Real.pi :=
by
  sorry

end abs_twice_sub_pi_l312_31291


namespace find_three_power_l312_31225

theorem find_three_power (m n : ℕ) (h₁: 3^m = 4) (h₂: 3^n = 5) : 3^(2*m + n) = 80 := by
  sorry

end find_three_power_l312_31225


namespace range_of_decreasing_function_l312_31221

noncomputable def f (a x : ℝ) : ℝ := 2 * a * x^2 + 4 * (a - 3) * x + 5

theorem range_of_decreasing_function (a : ℝ) :
  (∀ x : ℝ, x < 3 → (deriv (f a) x) ≤ 0) ↔ 0 ≤ a ∧ a ≤ 3/4 := 
sorry

end range_of_decreasing_function_l312_31221


namespace find_x_for_prime_square_l312_31253

theorem find_x_for_prime_square (x p : ℤ) (hp : Prime p) (h : 2 * x^2 - x - 36 = p^2) : x = 13 ∧ p = 17 :=
by
  sorry

end find_x_for_prime_square_l312_31253


namespace problem1_problem2_l312_31239

theorem problem1 :
  Real.sqrt 27 - (Real.sqrt 2 * Real.sqrt 6) + 3 * Real.sqrt (1/3) = 2 * Real.sqrt 3 := 
  by sorry

theorem problem2 :
  (Real.sqrt 5 + Real.sqrt 2) * (Real.sqrt 5 - Real.sqrt 2) + (Real.sqrt 3 - 1)^2 = 7 - 2 * Real.sqrt 3 := 
  by sorry

end problem1_problem2_l312_31239


namespace value_of_A_l312_31223

theorem value_of_A
  (A B C D E F G H I J : ℕ)
  (h_diff : ∀ x y : ℕ, x ≠ y → x ≠ y)
  (h_decreasing_ABC : A > B ∧ B > C)
  (h_decreasing_DEF : D > E ∧ E > F)
  (h_decreasing_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_consecutive_odd_DEF : D % 2 = 1 ∧ E % 2 = 1 ∧ F % 2 = 1 ∧ E = D - 2 ∧ F = E - 2)
  (h_consecutive_even_GHIJ : G % 2 = 0 ∧ H % 2 = 0 ∧ I % 2 = 0 ∧ J % 2 = 0 ∧ H = G - 2 ∧ I = H - 2 ∧ J = I - 2)
  (h_sum : A + B + C = 9) : 
  A = 8 :=
sorry

end value_of_A_l312_31223


namespace bottles_produced_l312_31243

/-- 
14 machines produce 2520 bottles in 4 minutes, given that 6 machines produce 270 bottles per minute. 
-/
theorem bottles_produced (rate_6_machines : Nat) (bottles_per_minute : Nat) 
  (rate_one_machine : Nat) (rate_14_machines : Nat) (total_production : Nat) : 
  rate_6_machines = 6 ∧ bottles_per_minute = 270 ∧ rate_one_machine = bottles_per_minute / rate_6_machines 
  ∧ rate_14_machines = 14 * rate_one_machine ∧ total_production = rate_14_machines * 4 → 
  total_production = 2520 :=
sorry

end bottles_produced_l312_31243


namespace fit_small_boxes_l312_31280

def larger_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

def small_box_volume (length width height : ℕ) : ℕ :=
  length * width * height

theorem fit_small_boxes (L W H l w h : ℕ)
  (larger_box_dim : L = 12 ∧ W = 14 ∧ H = 16)
  (small_box_dim : l = 3 ∧ w = 7 ∧ h = 2)
  (min_boxes : larger_box_volume L W H / small_box_volume l w h = 64) :
  ∃ n, n ≥ 64 :=
by
  sorry

end fit_small_boxes_l312_31280


namespace scores_are_sample_l312_31281

-- Define the total number of students
def total_students : ℕ := 5000

-- Define the number of selected students for sampling
def selected_students : ℕ := 200

-- Define a predicate that checks if a selection is a sample
def is_sample (total selected : ℕ) : Prop :=
  selected < total

-- The proposition that needs to be proven
theorem scores_are_sample : is_sample total_students selected_students := 
by 
  -- Proof of the theorem is omitted.
  sorry

end scores_are_sample_l312_31281


namespace harry_sandy_meet_point_l312_31244

theorem harry_sandy_meet_point :
  let H : ℝ × ℝ := (10, -3)
  let S : ℝ × ℝ := (2, 7)
  let t : ℝ := 2 / 3
  let meet_point : ℝ × ℝ := (H.1 + t * (S.1 - H.1), H.2 + t * (S.2 - H.2))
  meet_point = (14 / 3, 11 / 3) := 
by
  sorry

end harry_sandy_meet_point_l312_31244


namespace inequality_holds_l312_31294

theorem inequality_holds (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by 
  sorry

end inequality_holds_l312_31294


namespace coffee_is_32_3_percent_decaf_l312_31282

def percent_decaf_coffee_stock (total_weight initial_weight : ℕ) (initial_A_rate initial_B_rate initial_C_rate additional_weight additional_A_rate additional_D_rate : ℚ) 
(initial_A_decaf initial_B_decaf initial_C_decaf additional_D_decaf : ℚ) : ℚ :=
  let initial_A_weight := initial_A_rate * initial_weight
  let initial_B_weight := initial_B_rate * initial_weight
  let initial_C_weight := initial_C_rate * initial_weight
  let additional_A_weight := additional_A_rate * additional_weight
  let additional_D_weight := additional_D_rate * additional_weight

  let initial_A_decaf_weight := initial_A_decaf * initial_A_weight
  let initial_B_decaf_weight := initial_B_decaf * initial_B_weight
  let initial_C_decaf_weight := initial_C_decaf * initial_C_weight
  let additional_A_decaf_weight := initial_A_decaf * additional_A_weight
  let additional_D_decaf_weight := additional_D_decaf * additional_D_weight

  let total_decaf_weight := initial_A_decaf_weight + initial_B_decaf_weight + initial_C_decaf_weight + additional_A_decaf_weight + additional_D_decaf_weight

  (total_decaf_weight / total_weight) * 100

theorem coffee_is_32_3_percent_decaf : 
  percent_decaf_coffee_stock 1000 800 (40/100) (35/100) (25/100) 200 (50/100) (50/100) (20/100) (30/100) (45/100) (65/100) = 32.3 := 
  by 
    sorry

end coffee_is_32_3_percent_decaf_l312_31282


namespace non_basalt_rocks_total_eq_l312_31219

def total_rocks_in_box_A : ℕ := 57
def basalt_rocks_in_box_A : ℕ := 25

def total_rocks_in_box_B : ℕ := 49
def basalt_rocks_in_box_B : ℕ := 19

def non_basalt_rocks_in_box_A : ℕ := total_rocks_in_box_A - basalt_rocks_in_box_A
def non_basalt_rocks_in_box_B : ℕ := total_rocks_in_box_B - basalt_rocks_in_box_B

def total_non_basalt_rocks : ℕ := non_basalt_rocks_in_box_A + non_basalt_rocks_in_box_B

theorem non_basalt_rocks_total_eq : total_non_basalt_rocks = 62 := by
  -- proof goes here
  sorry

end non_basalt_rocks_total_eq_l312_31219


namespace cubic_eq_one_real_root_l312_31212

/-- The equation x^3 - 4x^2 + 9x + c = 0 has exactly one real root for any real number c. -/
theorem cubic_eq_one_real_root (c : ℝ) : 
  ∃! x : ℝ, x^3 - 4 * x^2 + 9 * x + c = 0 :=
sorry

end cubic_eq_one_real_root_l312_31212


namespace find_x_in_plane_figure_l312_31261

theorem find_x_in_plane_figure (x : ℝ) 
  (h1 : 0 < x) 
  (h2 : x < 360) 
  (h3 : 2 * x + 160 = 360) : 
  x = 100 :=
by
  sorry

end find_x_in_plane_figure_l312_31261


namespace Keiko_speed_is_pi_div_3_l312_31234

noncomputable def Keiko_avg_speed {r : ℝ} (v : ℝ → ℝ) (pi : ℝ) : ℝ :=
let C1 := 2 * pi * (r + 6) - 2 * pi * r
let t1 := 36
let v1 := C1 / t1

let C2 := 2 * pi * (r + 8) - 2 * pi * r
let t2 := 48
let v2 := C2 / t2

if v r = v1 ∧ v r = v2 then (v1 + v2) / 2 else 0

theorem Keiko_speed_is_pi_div_3 (pi : ℝ) (r : ℝ) (v : ℝ → ℝ) :
  v r = π / 3 ∧ (forall t1 t2 C1 C2, C1 / t1 = π / 3 ∧ C2 / t2 = π / 3 → 
  (C1/t1 + C2/t2)/2 = π / 3) :=
sorry

end Keiko_speed_is_pi_div_3_l312_31234


namespace find_square_l312_31201

theorem find_square (s : ℕ) : 
    (7863 / 13 = 604 + (s / 13)) → s = 11 :=
by
  sorry

end find_square_l312_31201


namespace simplify_expression_l312_31220

theorem simplify_expression : (27 * 10^9) / (9 * 10^2) = 3000000 := 
by sorry

end simplify_expression_l312_31220


namespace total_number_of_players_l312_31211

theorem total_number_of_players (n : ℕ) (h1 : n > 7) 
  (h2 : (4 * (n * (n - 1)) / 3 + 56 = (n + 8) * (n + 7) / 2)) : n + 8 = 50 :=
by
  sorry

end total_number_of_players_l312_31211


namespace diff_reading_math_homework_l312_31210

-- Define the conditions as given in the problem
def pages_math_homework : ℕ := 3
def pages_reading_homework : ℕ := 4

-- The statement to prove that Rachel had 1 more page of reading homework than math homework
theorem diff_reading_math_homework : pages_reading_homework - pages_math_homework = 1 := by
  sorry

end diff_reading_math_homework_l312_31210


namespace sophie_saves_money_l312_31263

-- Definitions based on the conditions
def loads_per_week : ℕ := 4
def sheets_per_load : ℕ := 1
def cost_per_box : ℝ := 5.50
def sheets_per_box : ℕ := 104
def weeks_per_year : ℕ := 52

-- Main theorem statement
theorem sophie_saves_money :
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  annual_saving = 11.00 := 
by {
  -- Calculation steps
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  -- Proving the final statement
  sorry
}

end sophie_saves_money_l312_31263


namespace relationship_S_T_l312_31227

-- Definitions based on the given conditions
def seq_a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 2 * n

def seq_b (n : ℕ) : ℕ :=
  2 ^ (n - 1) + 1

def S (n : ℕ) : ℕ :=
  (n * (n + 1))

def T (n : ℕ) : ℕ :=
  (2^n) + n - 1

-- The conjecture and proofs
theorem relationship_S_T (n : ℕ) : 
  if n = 1 then T n = S n
  else if (2 ≤ n ∧ n < 5) then T n < S n
  else n ≥ 5 → T n > S n :=
by sorry

end relationship_S_T_l312_31227


namespace ramesh_transport_cost_l312_31266

-- Definitions for conditions
def labelled_price (P : ℝ) : Prop := P = 13500 / 0.80
def selling_price (P : ℝ) : Prop := P * 1.10 = 18975
def transport_cost (T : ℝ) (extra_amount : ℝ) (installation_cost : ℝ) : Prop := T = extra_amount - installation_cost

-- The theorem statement to be proved
theorem ramesh_transport_cost (P T extra_amount installation_cost: ℝ) 
  (h1 : labelled_price P) 
  (h2 : selling_price P) 
  (h3 : extra_amount = 18975 - P)
  (h4 : installation_cost = 250) : 
  transport_cost T extra_amount installation_cost :=
by
  sorry

end ramesh_transport_cost_l312_31266


namespace number_of_students_playing_soccer_l312_31233

-- Definitions of the conditions
def total_students : ℕ := 500
def total_boys : ℕ := 350
def percent_boys_playing_soccer : ℚ := 0.86
def girls_not_playing_soccer : ℕ := 115

-- To be proved
theorem number_of_students_playing_soccer :
  ∃ (S : ℕ), S = 250 ∧ 0.14 * (S : ℚ) = 35 :=
sorry

end number_of_students_playing_soccer_l312_31233


namespace reciprocal_relationship_l312_31248

theorem reciprocal_relationship (a b : ℝ) (h₁ : a = 2 - Real.sqrt 3) (h₂ : b = Real.sqrt 3 + 2) : 
  a * b = 1 :=
by
  rw [h₁, h₂]
  sorry

end reciprocal_relationship_l312_31248


namespace election_total_votes_l312_31254

theorem election_total_votes (V: ℝ) (valid_votes: ℝ) (candidate_votes: ℝ) (invalid_rate: ℝ) (candidate_rate: ℝ) :
  candidate_rate = 0.75 →
  invalid_rate = 0.15 →
  candidate_votes = 357000 →
  valid_votes = (1 - invalid_rate) * V →
  candidate_votes = candidate_rate * valid_votes →
  V = 560000 :=
by
  intros candidate_rate_eq invalid_rate_eq candidate_votes_eq valid_votes_eq equation
  sorry

end election_total_votes_l312_31254
