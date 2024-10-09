import Mathlib

namespace tetradecagon_edge_length_correct_l524_52402

-- Define the parameters of the problem
def regular_tetradecagon_perimeter (n : ℕ := 14) : ℕ := 154

-- Define the length of one edge
def edge_length (P : ℕ) (n : ℕ) : ℕ := P / n

-- State the theorem
theorem tetradecagon_edge_length_correct :
  edge_length (regular_tetradecagon_perimeter 14) 14 = 11 := by
  sorry

end tetradecagon_edge_length_correct_l524_52402


namespace abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l524_52432

theorem abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one (x : ℝ) :
  |x| < 1 → x^3 < 1 ∧ (x^3 < 1 → |x| < 1 → False) :=
by
  sorry

end abs_x_lt_one_sufficient_not_necessary_for_x_cubed_lt_one_l524_52432


namespace original_pencils_l524_52481

-- Definition of the conditions
def pencils_initial := 115
def pencils_added := 100
def pencils_total := 215

-- Theorem stating the problem to be proved
theorem original_pencils :
  pencils_initial + pencils_added = pencils_total :=
by
  sorry

end original_pencils_l524_52481


namespace largest_circle_area_rounded_to_nearest_int_l524_52439

theorem largest_circle_area_rounded_to_nearest_int
  (x : Real)
  (hx : 3 * x^2 = 180) :
  let r := (16 * Real.sqrt 15) / (2 * Real.pi)
  let area_of_circle := Real.pi * r^2
  round (area_of_circle) = 306 :=
by
  sorry

end largest_circle_area_rounded_to_nearest_int_l524_52439


namespace problem_statement_l524_52435

variable {x y : ℝ}

theorem problem_statement (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : y - 2 / x ≠ 0) :
  (2 * x - 3 / y) / (3 * y - 2 / x) = (2 * x * y - 3) / (3 * x * y - 2) :=
sorry

end problem_statement_l524_52435


namespace find_machines_l524_52418

theorem find_machines (R : ℝ) : 
  (N : ℕ) -> 
  (H1 : N * R * 6 = 1) -> 
  (H2 : 4 * R * 12 = 1) -> 
  N = 8 :=
by
  sorry

end find_machines_l524_52418


namespace distance_of_canteen_from_each_camp_l524_52454

noncomputable def distanceFromCanteen (distGtoRoad distBtoG : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (distGtoRoad ^ 2 + distBtoG ^ 2)
  hypotenuse / 2

theorem distance_of_canteen_from_each_camp :
  distanceFromCanteen 360 800 = 438.6 :=
by
  sorry -- The proof is omitted but must show that this statement is valid.

end distance_of_canteen_from_each_camp_l524_52454


namespace john_friends_count_l524_52433

-- Define the initial conditions
def initial_amount : ℚ := 7.10
def cost_of_sweets : ℚ := 1.05
def amount_per_friend : ℚ := 1.00
def remaining_amount : ℚ := 4.05

-- Define the intermediate values
def after_sweets : ℚ := initial_amount - cost_of_sweets
def given_away : ℚ := after_sweets - remaining_amount

-- Define the final proof statement
theorem john_friends_count : given_away / amount_per_friend = 2 :=
by
  sorry

end john_friends_count_l524_52433


namespace elephant_weight_l524_52447

theorem elephant_weight :
  ∃ (w : ℕ), ∀ i : Fin 15, (i.val ≤ 13 → w + 2 * w = 15000) ∧ ((0:ℕ) < w → w = 5000) :=
by
  sorry

end elephant_weight_l524_52447


namespace sufficient_but_not_necessary_condition_l524_52436

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 1 → 1 / a < 1) ∧ ((1 / a < 1) → (a > 1 ∨ a < 0)) → 
  (∀ (P Q : Prop), (P → Q) → (Q → P ∨ False) → P ∧ ¬Q → False) :=
by
  sorry

end sufficient_but_not_necessary_condition_l524_52436


namespace circle_through_origin_and_point_l524_52465

theorem circle_through_origin_and_point (a r : ℝ) :
  (∃ a r : ℝ, (a^2 + (5 - 3 * a)^2 = r^2) ∧ ((a - 3)^2 + (3 * a - 6)^2 = r^2)) →
  a = 5/3 ∧ r^2 = 25/9 :=
sorry

end circle_through_origin_and_point_l524_52465


namespace cost_prices_sum_l524_52485

theorem cost_prices_sum
  (W B : ℝ)
  (h1 : 0.9 * W + 196 = 1.04 * W)
  (h2 : 1.08 * B - 150 = 1.02 * B) :
  W + B = 3900 := 
sorry

end cost_prices_sum_l524_52485


namespace alyssas_weekly_allowance_l524_52426

-- Define the constants and parameters
def spent_on_movies (A : ℝ) := 0.5 * A
def spent_on_snacks (A : ℝ) := 0.2 * A
def saved_for_future (A : ℝ) := 0.25 * A

-- Define the remaining allowance after expenses
def remaining_allowance_after_expenses (A : ℝ) := A - spent_on_movies A - spent_on_snacks A - saved_for_future A

-- Define Alyssa's allowance given the conditions
theorem alyssas_weekly_allowance : ∀ (A : ℝ), 
  remaining_allowance_after_expenses A = 12 → 
  A = 240 :=
by
  -- Proof omitted
  sorry

end alyssas_weekly_allowance_l524_52426


namespace nested_fraction_expression_l524_52479

theorem nested_fraction_expression : 
  1 + (1 / (1 - (1 / (1 + (1 / 2))))) = 4 := 
by sorry

end nested_fraction_expression_l524_52479


namespace sara_total_cents_l524_52407

-- Define the conditions as constants
def quarters : ℕ := 11
def value_per_quarter : ℕ := 25

-- Define the total amount formula based on the conditions
def total_cents (q : ℕ) (v : ℕ) : ℕ := q * v

-- The theorem to be proven
theorem sara_total_cents : total_cents quarters value_per_quarter = 275 :=
by
  -- Proof goes here
  sorry

end sara_total_cents_l524_52407


namespace distance_BF_l524_52453

-- Given the focus F of the parabola y^2 = 4x
def focus_of_parabola : (ℝ × ℝ) := (1, 0)

-- Points A and B lie on the parabola y^2 = 4x
def point_A (x y : ℝ) := y^2 = 4 * x
def point_B (x y : ℝ) := y^2 = 4 * x

-- The line through F intersects the parabola at points A and B, and |AF| = 2
def distance_AF : ℝ := 2

-- Prove that |BF| = 2
theorem distance_BF : ∀ (A B F : ℝ × ℝ), 
  A = (1, F.2) → 
  B = (1, -F.2) → 
  F = (1, 0) → 
  |A.1 - F.1| + |A.2 - F.2| = distance_AF → 
  |B.1 - F.1| + |B.2 - F.2| = 2 :=
by
  intros A B F hA hB hF d_AF
  sorry

end distance_BF_l524_52453


namespace cistern_wet_surface_area_l524_52427

def cistern (length : ℕ) (width : ℕ) (water_height : ℝ) : ℝ :=
  (length * width : ℝ) + 2 * (water_height * length) + 2 * (water_height * width)

theorem cistern_wet_surface_area :
  cistern 7 5 1.40 = 68.6 :=
by
  sorry

end cistern_wet_surface_area_l524_52427


namespace sequence_1_formula_sequence_2_formula_sequence_3_formula_l524_52442

theorem sequence_1_formula (n : ℕ) (hn : n > 0) : 
  (∃ a : ℕ → ℚ, (a 1 = 1/2) ∧ (a 2 = 1/6) ∧ (a 3 = 1/12) ∧ (a 4 = 1/20) ∧ (∀ n, a n = 1/(n*(n+1)))) :=
by
  sorry

theorem sequence_2_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℕ, (a 1 = 1) ∧ (a 2 = 2) ∧ (a 3 = 4) ∧ (a 4 = 8) ∧ (∀ n, a n = 2^(n-1))) :=
by
  sorry

theorem sequence_3_formula (n : ℕ) (hn : n > 0) :
  (∃ a : ℕ → ℚ, (a 1 = 4/5) ∧ (a 2 = 1/2) ∧ (a 3 = 4/11) ∧ (a 4 = 2/7) ∧ (∀ n, a n = 4/(3*n + 2))) :=
by
  sorry

end sequence_1_formula_sequence_2_formula_sequence_3_formula_l524_52442


namespace emery_family_first_hour_distance_l524_52499

noncomputable def total_time : ℝ := 4
noncomputable def remaining_distance : ℝ := 300
noncomputable def first_hour_distance : ℝ := 100

theorem emery_family_first_hour_distance :
  (remaining_distance / (total_time - 1)) = first_hour_distance :=
sorry

end emery_family_first_hour_distance_l524_52499


namespace total_precious_stones_is_305_l524_52428

theorem total_precious_stones_is_305 :
  let agate := 25
  let olivine := agate + 5
  let sapphire := 2 * olivine
  let diamond := olivine + 11
  let amethyst := sapphire + diamond
  let ruby := diamond + 7
  agate + olivine + sapphire + diamond + amethyst + ruby = 305 :=
by
  sorry

end total_precious_stones_is_305_l524_52428


namespace num_distinct_combinations_l524_52456

-- Define the conditions
def num_dials : Nat := 4
def digits : List Nat := List.range 10  -- Digits from 0 to 9

-- Define what it means for a combination to have distinct digits
def distinct_digits (comb : List Nat) : Prop :=
  comb.length = num_dials ∧ comb.Nodup

-- The main statement for the theorem
theorem num_distinct_combinations : 
  ∃ (n : Nat), n = 5040 ∧ ∀ comb : List Nat, distinct_digits comb → comb.length = num_dials →
  (List.permutations digits).length = n :=
by
  sorry

end num_distinct_combinations_l524_52456


namespace man_can_lift_one_box_each_hand_l524_52496

theorem man_can_lift_one_box_each_hand : 
  ∀ (people boxes : ℕ), people = 7 → boxes = 14 → (boxes / people) / 2 = 1 :=
by
  intros people boxes h_people h_boxes
  sorry

end man_can_lift_one_box_each_hand_l524_52496


namespace solve_y_l524_52440

theorem solve_y (y : ℝ) (hyp : (5 - (2 / y))^(1 / 3) = -3) : y = 1 / 16 :=
by
  sorry

end solve_y_l524_52440


namespace ethanol_in_full_tank_l524_52470

theorem ethanol_in_full_tank:
  ∀ (capacity : ℕ) (vol_A : ℕ) (vol_B : ℕ) (eth_A_perc : ℝ) (eth_B_perc : ℝ) (eth_A : ℝ) (eth_B : ℝ),
  capacity = 208 →
  vol_A = 82 →
  vol_B = (capacity - vol_A) →
  eth_A_perc = 0.12 →
  eth_B_perc = 0.16 →
  eth_A = vol_A * eth_A_perc →
  eth_B = vol_B * eth_B_perc →
  eth_A + eth_B = 30 :=
by
  intros capacity vol_A vol_B eth_A_perc eth_B_perc eth_A eth_B h1 h2 h3 h4 h5 h6 h7
  sorry

end ethanol_in_full_tank_l524_52470


namespace num_lighting_methods_l524_52486

-- Definitions of the problem's conditions
def total_lights : ℕ := 15
def lights_off : ℕ := 6
def lights_on : ℕ := total_lights - lights_off
def available_spaces : ℕ := lights_on - 1

-- Statement of the mathematically equivalent proof problem
theorem num_lighting_methods : Nat.choose available_spaces lights_off = 28 := by
  sorry

end num_lighting_methods_l524_52486


namespace joska_has_higher_probability_l524_52404

open Nat

def num_4_digit_with_all_diff_digits := 10 * 9 * 8 * 7
def total_4_digit_combinations := 10^4
def num_4_digit_with_repeated_digits := total_4_digit_combinations - num_4_digit_with_all_diff_digits

-- Calculate probabilities
noncomputable def prob_joska := (num_4_digit_with_all_diff_digits : ℝ) / (total_4_digit_combinations : ℝ)
noncomputable def prob_gabor := (num_4_digit_with_repeated_digits : ℝ) / (total_4_digit_combinations : ℝ)

theorem joska_has_higher_probability :
  prob_joska > prob_gabor :=
  by
    sorry

end joska_has_higher_probability_l524_52404


namespace necessary_but_not_sufficient_l524_52441

theorem necessary_but_not_sufficient (x : ℝ) : 
  (0 < x ∧ x < 2) → (x^2 - x - 6 < 0) ∧ ¬ ((x^2 - x - 6 < 0) → (0 < x ∧ x < 2)) :=
by
  sorry

end necessary_but_not_sufficient_l524_52441


namespace simplify_expression_l524_52431

variable (x : ℝ) (h : x ≠ 0)

theorem simplify_expression : (2 * x)⁻¹ + 2 = (1 + 4 * x) / (2 * x) :=
by
  sorry

end simplify_expression_l524_52431


namespace time_for_grid_5x5_l524_52422

-- Definition for the 3x7 grid conditions
def grid_3x7_minutes := 26
def grid_3x7_total_length := 4 * 7 + 8 * 3
def time_per_unit_length := grid_3x7_minutes / grid_3x7_total_length

-- Definition for the 5x5 grid total length
def grid_5x5_total_length := 6 * 5 + 6 * 5

-- Theorem stating that the time it takes to trace all lines of a 5x5 grid is 30 minutes
theorem time_for_grid_5x5 : (time_per_unit_length * grid_5x5_total_length) = 30 := by
  sorry

end time_for_grid_5x5_l524_52422


namespace sum_of_remainders_l524_52414

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 15 = 11) (h2 : b % 15 = 13) (h3 : c % 15 = 14) : (a + b + c) % 15 = 8 :=
by
  sorry

end sum_of_remainders_l524_52414


namespace relationship_among_abc_l524_52467

noncomputable def a : ℝ := 4^(1/3 : ℝ)
noncomputable def b : ℝ := Real.log 1/7 / Real.log 3
noncomputable def c : ℝ := (1/3 : ℝ)^(1/5 : ℝ)

theorem relationship_among_abc : a > c ∧ c > b := 
by 
  sorry

end relationship_among_abc_l524_52467


namespace sum_of_c_d_l524_52408

theorem sum_of_c_d (c d : ℝ) (g : ℝ → ℝ) 
(hg : ∀ x, g x = (x + 5) / (x^2 + c * x + d)) 
(hasymp : ∀ x, (x = 2 ∨ x = -3) → x^2 + c * x + d = 0) : 
c + d = -5 := 
by 
  sorry

end sum_of_c_d_l524_52408


namespace edward_total_money_l524_52444

-- define the amounts made and spent
def money_made_spring : ℕ := 2
def money_made_summer : ℕ := 27
def money_spent_supplies : ℕ := 5

-- total money left is calculated by adding what he made and subtracting the expenses
def total_money_end (m_spring m_summer m_supplies : ℕ) : ℕ :=
  m_spring + m_summer - m_supplies

-- the theorem to prove
theorem edward_total_money :
  total_money_end money_made_spring money_made_summer money_spent_supplies = 24 :=
by
  sorry

end edward_total_money_l524_52444


namespace original_speed_correct_l524_52430

variables (t m s : ℝ)

noncomputable def original_speed (t m s : ℝ) : ℝ :=
  ((t * m + Real.sqrt (t^2 * m^2 + 4 * t * m * s)) / (2 * t))

theorem original_speed_correct (t m s : ℝ) (ht : 0 < t) : 
  original_speed t m s = (Real.sqrt (t * m * (4 * s + t * m)) - t * m) / (2 * t) :=
by
  sorry

end original_speed_correct_l524_52430


namespace circle_cut_by_parabolas_l524_52455

theorem circle_cut_by_parabolas (n : ℕ) (h : n = 10) : 
  2 * n ^ 2 + 1 = 201 :=
by
  sorry

end circle_cut_by_parabolas_l524_52455


namespace polygon_sides_l524_52421

theorem polygon_sides (n : ℕ) (h : n * (n - 3) / 2 = 20) : n = 8 :=
by
  sorry

end polygon_sides_l524_52421


namespace split_coins_l524_52449

theorem split_coins (p n d q : ℕ) (hp : p % 5 = 0) 
  (h_total : p + 5 * n + 10 * d + 25 * q = 10000) :
  ∃ (p1 n1 d1 q1 p2 n2 d2 q2 : ℕ),
    (p1 + 5 * n1 + 10 * d1 + 25 * q1 = 5000) ∧
    (p2 + 5 * n2 + 10 * d2 + 25 * q2 = 5000) ∧
    (p = p1 + p2) ∧ (n = n1 + n2) ∧ (d = d1 + d2) ∧ (q = q1 + q2) :=
sorry

end split_coins_l524_52449


namespace joe_sold_50_cookies_l524_52450

theorem joe_sold_50_cookies :
  ∀ (x : ℝ), (1.20 = 1 + 0.20 * 1) → (60 = 1.20 * x) → x = 50 :=
by
  intros x h1 h2
  sorry

end joe_sold_50_cookies_l524_52450


namespace original_decimal_number_l524_52498

theorem original_decimal_number (x : ℝ) (h : 10 * x - x / 10 = 23.76) : x = 2.4 :=
sorry

end original_decimal_number_l524_52498


namespace instantaneous_velocity_at_3s_l524_52413

theorem instantaneous_velocity_at_3s (t s v : ℝ) (hs : s = t^3) (hts : t = 3*s) : v = 27 :=
by
  sorry

end instantaneous_velocity_at_3s_l524_52413


namespace rainfall_on_thursday_l524_52464

theorem rainfall_on_thursday
  (monday_am : ℝ := 2)
  (monday_pm : ℝ := 1)
  (tuesday_factor : ℝ := 2)
  (wednesday : ℝ := 0)
  (thursday : ℝ)
  (weekly_avg : ℝ := 4)
  (days_in_week : ℕ := 7)
  (total_weekly_rain : ℝ := days_in_week * weekly_avg) :
  2 * (monday_am + monday_pm + tuesday_factor * (monday_am + monday_pm) + thursday) 
    = total_weekly_rain
  → thursday = 5 :=
by
  sorry

end rainfall_on_thursday_l524_52464


namespace calculate_total_selling_price_l524_52473

noncomputable def total_selling_price (cost_price1 cost_price2 cost_price3 profit_percent1 profit_percent2 profit_percent3 : ℝ) : ℝ :=
  let sp1 := cost_price1 + (profit_percent1 / 100 * cost_price1)
  let sp2 := cost_price2 + (profit_percent2 / 100 * cost_price2)
  let sp3 := cost_price3 + (profit_percent3 / 100 * cost_price3)
  sp1 + sp2 + sp3

theorem calculate_total_selling_price :
  total_selling_price 550 750 1000 30 25 20 = 2852.5 :=
by
  -- proof omitted
  sorry

end calculate_total_selling_price_l524_52473


namespace catch_up_distance_l524_52457

def v_a : ℝ := 10 -- A's speed in kmph
def v_b : ℝ := 20 -- B's speed in kmph
def t : ℝ := 10 -- Time in hours when B starts after A

theorem catch_up_distance : v_b * t + v_a * t = 200 :=
by sorry

end catch_up_distance_l524_52457


namespace value_of_f_at_log_l524_52448

noncomputable def f : ℝ → ℝ := sorry -- We will define this below

-- Conditions as hypotheses
axiom odd_f : ∀ x : ℝ, f (-x) = - f (x)
axiom periodic_f : ∀ x : ℝ, f (x + 2) + f (x) = 0
axiom f_on_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (x) = 2^x - 1

-- Theorem statement
theorem value_of_f_at_log : f (Real.logb (1/8) 125) = 1 / 4 :=
sorry

end value_of_f_at_log_l524_52448


namespace min_x_y_l524_52452

theorem min_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 2) :
  x + y ≥ 9 / 2 := 
by 
  sorry

end min_x_y_l524_52452


namespace const_sequence_l524_52417

theorem const_sequence (x y : ℝ) (n : ℕ) (a : ℕ → ℝ)
  (h1 : ∀ n, a n - a (n + 1) = (a n ^ 2 - 1) / (a n + a (n - 1)))
  (h2 : ∀ n, a n = a (n + 1) → a n ^ 2 = 1 ∧ a n ≠ -a (n - 1))
  (h_init : a 1 = y ∧ a 0 = x)
  (hx : |x| = 1 ∧ y ≠ -x) :
  (∃ n0, ∀ n ≥ n0, a n = 1 ∨ a n = -1) := sorry

end const_sequence_l524_52417


namespace circle_represents_circle_iff_a_nonzero_l524_52423

-- Define the equation given in the problem
def circleEquation (a x y : ℝ) : Prop :=
  a*x^2 + a*y^2 - 4*(a-1)*x + 4*y = 0

-- State the required theorem
theorem circle_represents_circle_iff_a_nonzero (a : ℝ) :
  (∃ c : ℝ, ∃ h k : ℝ, ∀ x y : ℝ, circleEquation a x y ↔ (x - h)^2 + (y - k)^2 = c)
  ↔ a ≠ 0 :=
by
  sorry

end circle_represents_circle_iff_a_nonzero_l524_52423


namespace amount_after_two_years_l524_52468

/-- Defining given conditions. -/
def initial_value : ℤ := 65000
def first_year_increase : ℚ := 12 / 100
def second_year_increase : ℚ := 8 / 100

/-- The main statement that needs to be proved. -/
theorem amount_after_two_years : 
  let first_year_amount := initial_value + (initial_value * first_year_increase)
  let second_year_amount := first_year_amount + (first_year_amount * second_year_increase)
  second_year_amount = 78624 := 
by 
  sorry

end amount_after_two_years_l524_52468


namespace impossible_condition_l524_52405

noncomputable def f (x : ℝ) : ℝ := |2^x - 1|

theorem impossible_condition (a b c : ℝ) (h : f a > f b ∧ f b > f c) : ¬ (b < a ∧ a < c) :=
by
  sorry

end impossible_condition_l524_52405


namespace number_of_people_l524_52489

def totalCups : ℕ := 10
def cupsPerPerson : ℕ := 2

theorem number_of_people {n : ℕ} (h : n = totalCups / cupsPerPerson) : n = 5 := by
  sorry

end number_of_people_l524_52489


namespace grapes_total_sum_l524_52494

theorem grapes_total_sum (R A N : ℕ) 
  (h1 : A = R + 2) 
  (h2 : N = A + 4) 
  (h3 : R = 25) : 
  R + A + N = 83 := by
  sorry

end grapes_total_sum_l524_52494


namespace weekly_sales_correct_l524_52488

open Real

noncomputable def cost_left_handed_mouse (cost_normal_mouse : ℝ) : ℝ :=
  cost_normal_mouse * 1.3

noncomputable def cost_left_handed_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  cost_normal_keyboard * 1.2

noncomputable def cost_left_handed_scissors (cost_normal_scissors : ℝ) : ℝ :=
  cost_normal_scissors * 1.5

noncomputable def daily_sales_mouse (cost_normal_mouse : ℝ) : ℝ :=
  25 * cost_left_handed_mouse cost_normal_mouse

noncomputable def daily_sales_keyboard (cost_normal_keyboard : ℝ) : ℝ :=
  10 * cost_left_handed_keyboard cost_normal_keyboard

noncomputable def daily_sales_scissors (cost_normal_scissors : ℝ) : ℝ :=
  15 * cost_left_handed_scissors cost_normal_scissors

noncomputable def bundle_price (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  (cost_left_handed_mouse cost_normal_mouse + cost_left_handed_keyboard cost_normal_keyboard + cost_left_handed_scissors cost_normal_scissors) * 0.9

noncomputable def daily_sales_bundle (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  5 * bundle_price cost_normal_mouse cost_normal_keyboard cost_normal_scissors

noncomputable def weekly_sales (cost_normal_mouse : ℝ) (cost_normal_keyboard : ℝ) (cost_normal_scissors : ℝ) : ℝ :=
  3 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors) +
  1.5 * (daily_sales_mouse cost_normal_mouse + daily_sales_keyboard cost_normal_keyboard + daily_sales_scissors cost_normal_scissors + daily_sales_bundle cost_normal_mouse cost_normal_keyboard cost_normal_scissors)

theorem weekly_sales_correct :
  weekly_sales 120 80 30 = 29922.25 := sorry

end weekly_sales_correct_l524_52488


namespace probability_of_four_twos_in_five_rolls_l524_52460

theorem probability_of_four_twos_in_five_rolls :
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  total_probability = 3125 / 7776 :=
by
  let p_2 := 1 / 6
  let p_not_2 := 5 / 6
  let total_probability := 5 * (p_2 ^ 4) * p_not_2
  show total_probability = 3125 / 7776
  sorry

end probability_of_four_twos_in_five_rolls_l524_52460


namespace square_area_l524_52480

theorem square_area (p : ℝ) (h : p = 20) : (p / 4) ^ 2 = 25 :=
by
  sorry

end square_area_l524_52480


namespace simplify_expression_l524_52401

theorem simplify_expression : |(-5^2 - 6 * 2)| = 37 := by
  sorry

end simplify_expression_l524_52401


namespace largest_base_conversion_l524_52462

theorem largest_base_conversion :
  let a := (3: ℕ)
  let b := (1 * 2^1 + 1 * 2^0: ℕ)
  let c := (3 * 8^0: ℕ)
  let d := (1 * 3^1 + 1 * 3^0: ℕ)
  max a (max b (max c d)) = d :=
by
  sorry

end largest_base_conversion_l524_52462


namespace original_classes_l524_52459

theorem original_classes (x : ℕ) (h1 : 280 % x = 0) (h2 : 585 % (x + 6) = 0) : x = 7 :=
sorry

end original_classes_l524_52459


namespace line_tangent_to_parabola_j_eq_98_l524_52475

theorem line_tangent_to_parabola_j_eq_98 (j : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 7 * y + j = 0 → x ≠ 0) →
  j = 98 :=
by
  sorry

end line_tangent_to_parabola_j_eq_98_l524_52475


namespace sounds_meet_at_x_l524_52493

theorem sounds_meet_at_x (d c s : ℝ) (h1 : 0 < d) (h2 : 0 < c) (h3 : 0 < s) :
  ∃ x : ℝ, x = d / 2 * (1 + s / c) ∧ x <= d ∧ x > 0 :=
by
  sorry

end sounds_meet_at_x_l524_52493


namespace simplify_and_evaluate_l524_52476

theorem simplify_and_evaluate (a b : ℝ) (h₁ : a = -1) (h₂ : b = 1/4) : 
  (a + 2 * b) ^ 2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_and_evaluate_l524_52476


namespace find_smaller_number_l524_52458

def one_number_is_11_more_than_3times_another (x y : ℕ) : Prop :=
  y = 3 * x + 11

def their_sum_is_55 (x y : ℕ) : Prop :=
  x + y = 55

theorem find_smaller_number (x y : ℕ) (h1 : one_number_is_11_more_than_3times_another x y) (h2 : their_sum_is_55 x y) :
  x = 11 :=
by
  -- The proof will be inserted here
  sorry

end find_smaller_number_l524_52458


namespace solve_for_x_l524_52490

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 14.7 -> x = 105 := by
  sorry

end solve_for_x_l524_52490


namespace compare_neg_fractions_l524_52482

theorem compare_neg_fractions : - (3 / 4 : ℚ) > - (4 / 5 : ℚ) :=
sorry

end compare_neg_fractions_l524_52482


namespace inverse_proportion_range_l524_52445

theorem inverse_proportion_range (k : ℝ) (x : ℝ) :
  (∀ x : ℝ, (x < 0 -> (k - 1) / x > 0) ∧ (x > 0 -> (k - 1) / x < 0)) -> k < 1 :=
by
  sorry

end inverse_proportion_range_l524_52445


namespace fraction_spent_on_fruits_l524_52424

theorem fraction_spent_on_fruits (M : ℕ) (hM : M = 24) :
  (M - (M / 3 + M / 6) - 6) / M = 1 / 4 :=
by
  sorry

end fraction_spent_on_fruits_l524_52424


namespace percentage_half_day_students_l524_52429

theorem percentage_half_day_students
  (total_students : ℕ)
  (full_day_students : ℕ)
  (h_total : total_students = 80)
  (h_full_day : full_day_students = 60) :
  ((total_students - full_day_students) / total_students : ℚ) * 100 = 25 := 
by
  sorry

end percentage_half_day_students_l524_52429


namespace telepathic_connection_correct_l524_52472

def telepathic_connection_probability : ℚ := sorry

theorem telepathic_connection_correct :
  telepathic_connection_probability = 7 / 25 := sorry

end telepathic_connection_correct_l524_52472


namespace largest_number_among_list_l524_52446

theorem largest_number_among_list :
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  let a := 0.989
  let b := 0.997
  let c := 0.991
  let d := 0.999
  let e := 0.990
  sorry

end largest_number_among_list_l524_52446


namespace spadesuit_calculation_l524_52416

def spadesuit (x y : ℝ) : ℝ := (x + 2 * y) ^ 2 * (x - y)

theorem spadesuit_calculation :
  spadesuit 3 (spadesuit 2 3) = 1046875 :=
by
  sorry

end spadesuit_calculation_l524_52416


namespace fixed_point_line_l524_52403

theorem fixed_point_line (k : ℝ) :
  ∃ A : ℝ × ℝ, (3 + k) * A.1 - 2 * A.2 + 1 - k = 0 ∧ (A = (1, 2)) :=
by
  let A : ℝ × ℝ := (1, 2)
  use A
  sorry

end fixed_point_line_l524_52403


namespace find_A_l524_52438

theorem find_A (A B : ℝ) 
  (h1 : A - 3 * B = 303.1)
  (h2 : 10 * B = A) : 
  A = 433 :=
by
  sorry

end find_A_l524_52438


namespace nine_chapters_coins_l524_52497

theorem nine_chapters_coins (a d : ℚ)
  (h1 : (a - 2 * d) + (a - d) = a + (a + d) + (a + 2 * d))
  (h2 : (a - 2 * d) + (a - d) + a + (a + d) + (a + 2 * d) = 5) :
  a - d = 7 / 6 :=
by 
  sorry

end nine_chapters_coins_l524_52497


namespace range_of_a_for_local_maximum_l524_52471

noncomputable def f' (a x : ℝ) := a * (x + 1) * (x - a)

theorem range_of_a_for_local_maximum {a : ℝ} (hf_max : ∀ x : ℝ, f' a x = 0 → ∀ y : ℝ, y ≠ x → f' a y ≤ f' a x) :
  -1 < a ∧ a < 0 :=
sorry

end range_of_a_for_local_maximum_l524_52471


namespace distance_closer_to_R_after_meeting_l524_52406

def distance_between_R_and_S : ℕ := 80
def rate_of_man_from_R : ℕ := 5
def initial_rate_of_man_from_S : ℕ := 4

theorem distance_closer_to_R_after_meeting 
  (t : ℕ) 
  (x : ℕ) 
  (h1 : t ≠ 0) 
  (h2 : distance_between_R_and_S = 80) 
  (h3 : rate_of_man_from_R = 5) 
  (h4 : initial_rate_of_man_from_S = 4) 
  (h5 : (rate_of_man_from_R * t) 
        + (t * initial_rate_of_man_from_S 
        + ((t - 1) * t / 2)) = distance_between_R_and_S) :
  x = 20 :=
sorry

end distance_closer_to_R_after_meeting_l524_52406


namespace polygon_encloses_250_square_units_l524_52420

def vertices : List (ℕ × ℕ) := [(0, 0), (20, 0), (20, 20), (10, 20), (10, 10), (0, 10)]

def polygon_area (vertices : List (ℕ × ℕ)) : ℕ :=
  -- Function to calculate the area of the given polygon
  sorry

theorem polygon_encloses_250_square_units : polygon_area vertices = 250 := by
  -- Proof that the area of the polygon is 250 square units
  sorry

end polygon_encloses_250_square_units_l524_52420


namespace auntie_em_parking_l524_52419

theorem auntie_em_parking (total_spaces cars : ℕ) (probability_can_park : ℚ) :
  total_spaces = 20 →
  cars = 15 →
  probability_can_park = 232/323 :=
by
  sorry

end auntie_em_parking_l524_52419


namespace problem_solution_l524_52437

theorem problem_solution (x y : ℝ) (h1 : x + 2 * y = 1) (h2 : 2 * x + y = 2) : x + y = 1 :=
by
  sorry

end problem_solution_l524_52437


namespace park_available_spaces_l524_52400

theorem park_available_spaces :
  let section_A_benches := 30
  let section_A_capacity_per_bench := 4
  let section_B_benches := 20
  let section_B_capacity_per_bench := 5
  let section_C_benches := 15
  let section_C_capacity_per_bench := 6
  let section_A_people := 50
  let section_B_people := 40
  let section_C_people := 45
  let section_A_total_capacity := section_A_benches * section_A_capacity_per_bench
  let section_B_total_capacity := section_B_benches * section_B_capacity_per_bench
  let section_C_total_capacity := section_C_benches * section_C_capacity_per_bench
  let section_A_available := section_A_total_capacity - section_A_people
  let section_B_available := section_B_total_capacity - section_B_people
  let section_C_available := section_C_total_capacity - section_C_people
  let total_available_spaces := section_A_available + section_B_available + section_C_available
  total_available_spaces = 175 := 
by
  let section_A_benches := 30
  let section_A_capacity_per_bench := 4
  let section_B_benches := 20
  let section_B_capacity_per_bench := 5
  let section_C_benches := 15
  let section_C_capacity_per_bench := 6
  let section_A_people := 50
  let section_B_people := 40
  let section_C_people := 45
  let section_A_total_capacity := section_A_benches * section_A_capacity_per_bench
  let section_B_total_capacity := section_B_benches * section_B_capacity_per_bench
  let section_C_total_capacity := section_C_benches * section_C_capacity_per_bench
  let section_A_available := section_A_total_capacity - section_A_people
  let section_B_available := section_B_total_capacity - section_B_people
  let section_C_available := section_C_total_capacity - section_C_people
  let total_available_spaces := section_A_available + section_B_available + section_C_available
  sorry

end park_available_spaces_l524_52400


namespace new_sum_after_decrease_l524_52474

theorem new_sum_after_decrease (a b : ℕ) (h₁ : a + b = 100) (h₂ : a' = a - 48) : a' + b = 52 := by
  sorry

end new_sum_after_decrease_l524_52474


namespace find_a_and_b_l524_52425

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := Real.log (abs (a + 1 / (1 - x))) + b

theorem find_a_and_b (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  a = -1 / 2 ∧ b = Real.log 2 :=
by
  sorry

end find_a_and_b_l524_52425


namespace inequality_solution_set_is_correct_l524_52409

noncomputable def inequality_solution_set (x : ℝ) : Prop :=
  (3 * x - 1) / (2 - x) ≥ 1

theorem inequality_solution_set_is_correct :
  { x : ℝ | inequality_solution_set x } = { x : ℝ | 3 / 4 ≤ x ∧ x < 2 } :=
by sorry

end inequality_solution_set_is_correct_l524_52409


namespace find_fraction_l524_52411

variable (x y z : ℂ) -- All complex numbers
variable (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) -- Non-zero conditions
variable (h2 : x + y + z = 10) -- Sum condition
variable (h3 : 2 * ((x - y)^2 + (x - z)^2 + (y - z)^2) = x * y * z) -- Given equation condition

theorem find_fraction 
    (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0) 
    (h2 : x + y + z = 10)
    (h3 : 2 * ((x - y) ^ 2 + (x - z) ^ 2 + (y - z) ^ 2) = x * y * z) :
    (x^3 + y^3 + z^3) / (x * y * z) = 11 / 2 := 
sorry -- Proof yet to be completed

end find_fraction_l524_52411


namespace todd_ate_cupcakes_l524_52484

theorem todd_ate_cupcakes :
  let C := 38   -- Total cupcakes baked by Sarah
  let P := 3    -- Number of packages made
  let c := 8    -- Number of cupcakes per package
  let L := P * c  -- Total cupcakes left after packaging
  C - L = 14 :=  -- Cupcakes Todd ate is 14
by
  sorry

end todd_ate_cupcakes_l524_52484


namespace inverse_proportion_symmetric_l524_52461

theorem inverse_proportion_symmetric (a b : ℝ) (h : a ≠ 0) (h_ab : b = -6 / -a) : (-b) = -6 / a :=
by
  -- the proof goes here
  sorry

end inverse_proportion_symmetric_l524_52461


namespace Dalton_saved_amount_l524_52415

theorem Dalton_saved_amount (total_cost uncle_contribution additional_needed saved_from_allowance : ℕ) 
  (h_total_cost : total_cost = 7 + 12 + 4)
  (h_uncle_contribution : uncle_contribution = 13)
  (h_additional_needed : additional_needed = 4)
  (h_current_amount : total_cost - additional_needed = 19)
  (h_saved_amount : 19 - uncle_contribution = saved_from_allowance) :
  saved_from_allowance = 6 :=
sorry

end Dalton_saved_amount_l524_52415


namespace units_digit_x_pow_75_plus_6_eq_9_l524_52478

theorem units_digit_x_pow_75_plus_6_eq_9 (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9)
  (h3 : (x ^ 75 + 6) % 10 = 9) : x = 3 :=
sorry

end units_digit_x_pow_75_plus_6_eq_9_l524_52478


namespace inequality_solution_set_l524_52443

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) : (x - 1) / x > 1 ↔ x < 0 :=
by
  sorry

end inequality_solution_set_l524_52443


namespace find_number_l524_52495

theorem find_number (N : ℕ) (h1 : ∃ k : ℤ, N = 13 * k + 11) (h2 : ∃ m : ℤ, N = 17 * m + 9) : N = 89 := 
sorry

end find_number_l524_52495


namespace mixing_ratios_indeterminacy_l524_52434

theorem mixing_ratios_indeterminacy (x : ℝ) (a b : ℝ) (h1 : a + b = 50) (h2 : 0.40 * a + (x / 100) * b = 25) : False :=
sorry

end mixing_ratios_indeterminacy_l524_52434


namespace tax_percentage_l524_52469

theorem tax_percentage (total_pay take_home_pay: ℕ) (h1 : total_pay = 650) (h2 : take_home_pay = 585) :
  ((total_pay - take_home_pay) * 100 / total_pay) = 10 :=
by
  -- Assumptions
  have hp1 : total_pay = 650 := h1
  have hp2 : take_home_pay = 585 := h2
  -- Calculate tax paid
  let tax_paid := total_pay - take_home_pay
  -- Calculate tax percentage
  let tax_percentage := (tax_paid * 100) / total_pay
  -- Prove the tax percentage is 10%
  sorry

end tax_percentage_l524_52469


namespace product_greater_than_constant_l524_52410

noncomputable def f (x m : ℝ) := Real.log x - (m + 1) * x + (1 / 2) * m * x ^ 2
noncomputable def g (x m : ℝ) := Real.log x - (m + 1) * x

variables {x1 x2 m : ℝ} 
  (h1 : g x1 m = 0)
  (h2 : g x2 m = 0)
  (h3 : x2 > Real.exp 1 * x1)

theorem product_greater_than_constant :
  x1 * x2 > 2 / (Real.exp 1 - 1) :=
sorry

end product_greater_than_constant_l524_52410


namespace original_price_correct_percentage_growth_rate_l524_52477

-- Definitions and conditions
def original_price := 45
def sale_discount := 15
def price_after_discount := original_price - sale_discount

def initial_cost_before_event := 90
def final_cost_during_event := 120
def ratio_of_chickens := 2

def initial_buyers := 50
def increase_percentage := 20
def total_sales := 5460
def time_slots := 2  -- 1 hour = 2 slots of 30 minutes each

-- The problem: Prove the original price and growth rate
theorem original_price_correct (x : ℕ) : (120 / (x - 15) = 2 * (90 / x) → x = original_price) :=
by
  sorry

theorem percentage_growth_rate (m : ℕ) :
  (50 + 50 * (1 + m / 100) + 50 * (1 + m / 100)^2 = total_sales / (original_price - sale_discount) →
  m = increase_percentage) :=
by
  sorry

end original_price_correct_percentage_growth_rate_l524_52477


namespace mod_product_example_l524_52463

theorem mod_product_example :
  ∃ m : ℤ, 256 * 738 ≡ m [ZMOD 75] ∧ 0 ≤ m ∧ m < 75 ∧ m = 53 :=
by
  use 53
  sorry

end mod_product_example_l524_52463


namespace artworks_per_student_in_first_half_l524_52483

theorem artworks_per_student_in_first_half (x : ℕ) (h1 : 10 = 10) (h2 : 20 = 20) (h3 : 5 * x + 5 * 4 = 35) : x = 3 := by
  sorry

end artworks_per_student_in_first_half_l524_52483


namespace cube_faces_sum_39_l524_52491

theorem cube_faces_sum_39 (a b c d e f g h : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧ g > 0 ∧ h > 0)
    (vertex_sum : (a*e*b*h + a*e*c*h + a*f*b*h + a*f*c*h + d*e*b*h + d*e*c*h + d*f*b*h + d*f*c*h) = 2002) :
    (a + b + c + d + e + f + g + h) = 39 := 
sorry

end cube_faces_sum_39_l524_52491


namespace sin_theta_value_l524_52487

theorem sin_theta_value (θ : ℝ) (h₁ : 8 * (Real.tan θ) = 3 * (Real.cos θ)) (h₂ : 0 < θ ∧ θ < Real.pi) : 
  Real.sin θ = 1 / 3 := 
by sorry

end sin_theta_value_l524_52487


namespace distance_from_point_to_x_axis_l524_52466

theorem distance_from_point_to_x_axis (x y : ℤ) (h : (x, y) = (5, -12)) : |y| = 12 :=
by
  -- sorry serves as a placeholder for the proof
  sorry

end distance_from_point_to_x_axis_l524_52466


namespace numbers_in_ratio_l524_52492

theorem numbers_in_ratio (a b c : ℤ) :
  (∃ x : ℤ, a = 2 * x ∧ b = 3 * x ∧ c = 4 * x) ∧ (a * a + b * b + c * c = 725) →
  (a = 10 ∧ b = 15 ∧ c = 20 ∨ a = -10 ∧ b = -15 ∧ c = -20) :=
by
  sorry

end numbers_in_ratio_l524_52492


namespace tangent_lines_ln_l524_52451

theorem tangent_lines_ln (x y: ℝ) : 
    (y = Real.log (abs x)) → 
    (x = 0 ∧ y = 0) ∨ ((x = yup ∨ x = ydown) ∧ (∀ (ey : ℝ), x = ey ∨ x = -ey)) :=
by 
    intro h
    sorry

end tangent_lines_ln_l524_52451


namespace compute_expression_l524_52412

theorem compute_expression :
  (1 / 36) / ((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) + 
  (((1 / 4) + (1 / 12) - (7 / 18) - (1 / 36)) / (1 / 36)) = -10 / 3 :=
by
  sorry

end compute_expression_l524_52412
