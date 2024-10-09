import Mathlib

namespace average_of_four_numbers_l2233_223331

theorem average_of_four_numbers (a b c d : ℝ) 
  (h1 : b + c + d = 24) (h2 : a + c + d = 36)
  (h3 : a + b + d = 28) (h4 : a + b + c = 32) :
  (a + b + c + d) / 4 = 10 := 
sorry

end average_of_four_numbers_l2233_223331


namespace walked_8_miles_if_pace_4_miles_per_hour_l2233_223387

-- Define the conditions
def walked_some_miles_in_2_hours (d : ℝ) : Prop :=
  d = 2

def pace_same_4_miles_per_hour (p : ℝ) : Prop :=
  p = 4

-- Define the proof problem
theorem walked_8_miles_if_pace_4_miles_per_hour :
  ∀ (d p : ℝ), walked_some_miles_in_2_hours d → pace_same_4_miles_per_hour p → (p * d = 8) :=
by
  intros d p h1 h2
  rw [h1, h2]
  exact sorry

end walked_8_miles_if_pace_4_miles_per_hour_l2233_223387


namespace average_xy_l2233_223355

theorem average_xy (x y : ℝ) 
  (h : (4 + 6 + 9 + x + y) / 5 = 20) : (x + y) / 2 = 40.5 :=
sorry

end average_xy_l2233_223355


namespace price_of_whole_pizza_l2233_223353

theorem price_of_whole_pizza
    (price_per_slice : ℕ)
    (num_slices_sold : ℕ)
    (num_whole_pizzas_sold : ℕ)
    (total_revenue : ℕ) 
    (H : price_per_slice * num_slices_sold + num_whole_pizzas_sold * P = total_revenue) : 
    P = 15 :=
by
  let price_per_slice := 3
  let num_slices_sold := 24
  let num_whole_pizzas_sold := 3
  let total_revenue := 117
  sorry

end price_of_whole_pizza_l2233_223353


namespace customer_saves_7_906304_percent_l2233_223361

variable {P : ℝ} -- Define the base retail price as a variable

-- Define the percentage reductions and additions
def reduced_price (P : ℝ) : ℝ := 0.88 * P
def further_discount_price (P : ℝ) : ℝ := reduced_price P * 0.95
def price_with_dealers_fee (P : ℝ) : ℝ := further_discount_price P * 1.02
def final_price (P : ℝ) : ℝ := price_with_dealers_fee P * 1.08

-- Define the final price factor
def final_price_factor : ℝ := 0.88 * 0.95 * 1.02 * 1.08

noncomputable def total_savings (P : ℝ) : ℝ :=
  P - (final_price_factor * P)

theorem customer_saves_7_906304_percent (P : ℝ) :
  total_savings P = P * 0.07906304 := by
  sorry -- Proof to be added

end customer_saves_7_906304_percent_l2233_223361


namespace yellow_marbles_at_least_zero_l2233_223348

noncomputable def total_marbles := 30
def blue_marbles (n : ℕ) := n / 3
def red_marbles (n : ℕ) := n / 3
def green_marbles := 10
def yellow_marbles (n : ℕ) := n - ((2 * n) / 3 + 10)

-- Conditions
axiom h1 : total_marbles % 3 = 0
axiom h2 : total_marbles = 30

-- Prove the smallest number of yellow marbles is 0
theorem yellow_marbles_at_least_zero : yellow_marbles total_marbles = 0 := by
  sorry

end yellow_marbles_at_least_zero_l2233_223348


namespace prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l2233_223392

noncomputable def prob_win_single_game : ℚ := 7 / 10
noncomputable def prob_lose_single_game : ℚ := 3 / 10

theorem prob_win_all_6_games : (prob_win_single_game ^ 6) = 117649 / 1000000 :=
by
  sorry

theorem prob_win_exactly_5_out_of_6_games : (6 * (prob_win_single_game ^ 5) * prob_lose_single_game) = 302526 / 1000000 :=
by
  sorry

end prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l2233_223392


namespace contrapositive_of_lt_l2233_223385

theorem contrapositive_of_lt (a b c : ℝ) :
  (a < b → a + c < b + c) → (a + c ≥ b + c → a ≥ b) :=
by
  intro h₀ h₁
  sorry

end contrapositive_of_lt_l2233_223385


namespace largest_side_of_rectangle_l2233_223336

theorem largest_side_of_rectangle :
  ∃ (l w : ℝ), (2 * l + 2 * w = 240) ∧ (l * w = 12 * 240) ∧ (l = 86.835 ∨ w = 86.835) :=
by
  -- Actual proof would be here
  sorry

end largest_side_of_rectangle_l2233_223336


namespace average_score_is_correct_l2233_223351

-- Define the given conditions
def numbers_of_students : List ℕ := [12, 28, 40, 35, 20, 10, 5]
def scores : List ℕ := [95, 85, 75, 65, 55, 45, 35]

-- Function to calculate the total score
def total_score (scores numbers : List ℕ) : ℕ :=
  List.sum (List.zipWith (λ a b => a * b) scores numbers)

-- Calculate the average percent score
def average_percent_score (total number_of_students : ℕ) : ℕ :=
  total / number_of_students

-- Prove that the average percentage score is 70
theorem average_score_is_correct :
  average_percent_score (total_score scores numbers_of_students) 150 = 70 :=
by
  sorry

end average_score_is_correct_l2233_223351


namespace part_a_part_b_part_c_l2233_223350

-- Part (a)
theorem part_a : ∃ a b, a * b = 80 ∧ (a = 8 ∨ a = 4) ∧ (b = 10 ∨ b = 5) :=
by sorry

-- Part (b)
theorem part_b : ∃ a b c, (a * b) / c = 50 ∧ (a = 10 ∨ a = 5) ∧ (b = 10 ∨ b = 5) ∧ (c = 2 ∨ c = 1) :=
by sorry

-- Part (c)
theorem part_c : ∃ n, n = 4 ∧ ∀ a b c, (a + b) / c = 23 :=
by sorry

end part_a_part_b_part_c_l2233_223350


namespace possible_values_x_l2233_223315

variable (a b x : ℕ)

theorem possible_values_x (h1 : a + b = 20)
                          (h2 : a * x + b * 3 = 109) :
    x = 10 ∨ x = 52 :=
sorry

end possible_values_x_l2233_223315


namespace find_f3_l2233_223372

theorem find_f3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, x * f y = y * f x) (h2 : f 15 = 20) : f 3 = 4 := 
  sorry

end find_f3_l2233_223372


namespace sum_of_squares_eq_l2233_223399

theorem sum_of_squares_eq :
  (1000^2 + 1001^2 + 1002^2 + 1003^2 + 1004^2 + 1005^2 + 1006^2) = 7042091 :=
by {
  sorry
}

end sum_of_squares_eq_l2233_223399


namespace perpendicular_vectors_x_value_l2233_223396

theorem perpendicular_vectors_x_value:
  ∀ (x : ℝ), let a : ℝ × ℝ := (1, 2)
             let b : ℝ × ℝ := (x, 1)
             (a.1 * b.1 + a.2 * b.2 = 0) → x = -2 :=
by
  intro x
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 1)
  intro h
  sorry

end perpendicular_vectors_x_value_l2233_223396


namespace log_inequality_solution_l2233_223333

variable {a x : ℝ}

theorem log_inequality_solution (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  (1 + Real.log (a ^ x - 1) / Real.log 2 ≤ Real.log (4 - a ^ x) / Real.log 2) →
  ((1 < a ∧ x ≤ Real.log (7 / 4) / Real.log a) ∨ (0 < a ∧ a < 1 ∧ x ≥ Real.log (7 / 4) / Real.log a)) :=
sorry

end log_inequality_solution_l2233_223333


namespace negation_p_equiv_l2233_223328

noncomputable def negation_of_proposition_p : Prop :=
∀ m : ℝ, ¬ ∃ x : ℝ, x^2 + m * x + 1 = 0

theorem negation_p_equiv (p : Prop) (h : p = ∃ m : ℝ, ∃ x : ℝ, x^2 + m * x + 1 = 0) :
  ¬ p ↔ negation_of_proposition_p :=
by {
  sorry
}

end negation_p_equiv_l2233_223328


namespace problem_i_problem_ii_l2233_223339

noncomputable def f (m x : ℝ) := (Real.log x / Real.log m) ^ 2 + 2 * (Real.log x / Real.log m) - 3

theorem problem_i (x : ℝ) : f 2 x < 0 ↔ (1 / 8) < x ∧ x < 2 :=
by sorry

theorem problem_ii (m : ℝ) (H : ∀ x, 2 ≤ x ∧ x ≤ 4 → f m x < 0) : 
  (0 < m ∧ m < 4^(1/3)) ∨ (4 < m) :=
by sorry

end problem_i_problem_ii_l2233_223339


namespace cone_surface_area_l2233_223305

theorem cone_surface_area {h : ℝ} {A_base : ℝ} (h_eq : h = 4) (A_base_eq : A_base = 9 * Real.pi) :
  let r := Real.sqrt (A_base / Real.pi)
  let l := Real.sqrt (r^2 + h^2)
  let lateral_area := Real.pi * r * l
  let total_surface_area := lateral_area + A_base
  total_surface_area = 24 * Real.pi :=
by
  sorry

end cone_surface_area_l2233_223305


namespace gallons_left_l2233_223386

theorem gallons_left (initial_gallons : ℚ) (gallons_given : ℚ) (gallons_left : ℚ) : 
  initial_gallons = 4 ∧ gallons_given = 16/3 → gallons_left = -4/3 :=
by
  sorry

end gallons_left_l2233_223386


namespace square_area_with_circles_l2233_223346

theorem square_area_with_circles (r : ℝ) (h : r = 8) : (2 * (2 * r))^2 = 1024 := 
by 
  sorry

end square_area_with_circles_l2233_223346


namespace mutually_exclusive_probability_zero_l2233_223376

theorem mutually_exclusive_probability_zero {A B : Prop} (p1 p2 : ℝ) 
  (hA : 0 ≤ p1 ∧ p1 ≤ 1) 
  (hB : 0 ≤ p2 ∧ p2 ≤ 1) 
  (hAB : A ∧ B → False) : 
  (A ∧ B) = False :=
by
  sorry

end mutually_exclusive_probability_zero_l2233_223376


namespace hotel_r_charge_percentage_l2233_223325

-- Let P, R, and G be the charges for a single room at Hotels P, R, and G respectively
variables (P R G : ℝ)

-- Given conditions:
-- 1. The charge for a single room at Hotel P is 55% less than the charge for a single room at Hotel R.
-- 2. The charge for a single room at Hotel P is 10% less than the charge for a single room at Hotel G.
axiom h1 : P = 0.45 * R
axiom h2 : P = 0.90 * G

-- The charge for a single room at Hotel R is what percent greater than the charge for a single room at Hotel G.
theorem hotel_r_charge_percentage : (R - G) / G * 100 = 100 :=
sorry

end hotel_r_charge_percentage_l2233_223325


namespace B_lap_time_l2233_223383

-- Definitions based on given conditions.
def time_to_complete_lap_A := 40
def meeting_interval := 15

-- The theorem states that given the conditions, B takes 24 seconds to complete the track.
theorem B_lap_time (l : ℝ) (t : ℝ) (h1 : t = 24)
                    (h2 : l / time_to_complete_lap_A + l / t = l / meeting_interval):
  t = 24 := by sorry

end B_lap_time_l2233_223383


namespace solve_abs_equation_l2233_223390

theorem solve_abs_equation (x : ℝ) : 2 * |x - 5| = 6 ↔ x = 2 ∨ x = 8 :=
by
  sorry

end solve_abs_equation_l2233_223390


namespace piles_3_stones_impossible_l2233_223342

theorem piles_3_stones_impossible :
  ∀ n : ℕ, ∀ piles : ℕ → ℕ,
  (piles 0 = 1001) →
  (∀ k : ℕ, k > 0 → ∃ i j : ℕ, piles (k-1) > 1 → piles k = i + j ∧ i > 0 ∧ j > 0) →
  ¬ (∀ m : ℕ, piles m ≠ 3) :=
by
  sorry

end piles_3_stones_impossible_l2233_223342


namespace abc_plus_2_gt_a_plus_b_plus_c_l2233_223352

theorem abc_plus_2_gt_a_plus_b_plus_c (a b c : ℝ) (ha : -1 < a) (ha' : a < 1) (hb : -1 < b) (hb' : b < 1) (hc : -1 < c) (hc' : c < 1) :
  a * b * c + 2 > a + b + c :=
sorry

end abc_plus_2_gt_a_plus_b_plus_c_l2233_223352


namespace travel_time_l2233_223395

theorem travel_time (distance speed : ℕ) (h_distance : distance = 810) (h_speed : speed = 162) :
  distance / speed = 5 :=
by
  sorry

end travel_time_l2233_223395


namespace ratio_thursday_to_wednesday_l2233_223310

variables (T : ℕ)

def time_studied_wednesday : ℕ := 2
def time_studied_thursday : ℕ := T
def time_studied_friday : ℕ := T / 2
def time_studied_weekend : ℕ := 2 + T + T / 2
def total_time_studied : ℕ := 22

theorem ratio_thursday_to_wednesday (h : 
  time_studied_wednesday + time_studied_thursday + time_studied_friday + time_studied_weekend = total_time_studied
) : (T : ℚ) / time_studied_wednesday = 3 := by
  sorry

end ratio_thursday_to_wednesday_l2233_223310


namespace minimum_value_of_2m_plus_n_solution_set_for_inequality_l2233_223370

namespace MathProof

-- Definitions and conditions
def f (x m n : ℝ) : ℝ := |x + m| + |2 * x - n|

-- Part (I)
theorem minimum_value_of_2m_plus_n
  (m n : ℝ)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_f_nonneg : ∀ x : ℝ, f x m n ≥ 1) :
  2 * m + n ≥ 2 :=
sorry

-- Part (II)
theorem solution_set_for_inequality
  (x : ℝ) :
  (f x 2 3 > 5 ↔ (x < 0 ∨ x > 2)) :=
sorry

end MathProof

end minimum_value_of_2m_plus_n_solution_set_for_inequality_l2233_223370


namespace min_bounces_for_height_less_than_two_l2233_223317

theorem min_bounces_for_height_less_than_two : 
  ∃ (k : ℕ), (20 * (3 / 4 : ℝ)^k < 2 ∧ ∀ n < k, ¬(20 * (3 / 4 : ℝ)^n < 2)) :=
sorry

end min_bounces_for_height_less_than_two_l2233_223317


namespace chord_ratio_l2233_223347

theorem chord_ratio (EQ GQ HQ FQ : ℝ) (h1 : EQ = 5) (h2 : GQ = 12) (h3 : HQ = 3) (h4 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 12 / 5 := by
  sorry

end chord_ratio_l2233_223347


namespace negation_of_universal_proposition_l2233_223380

def P (x : ℝ) : Prop := x^3 + 2 * x ≥ 0

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 0 ≤ x → P x) ↔ (∃ x : ℝ, 0 ≤ x ∧ ¬ P x) :=
by
  sorry

end negation_of_universal_proposition_l2233_223380


namespace exists_integer_cube_ends_with_2007_ones_l2233_223375

theorem exists_integer_cube_ends_with_2007_ones :
  ∃ x : ℕ, x^3 % 10^2007 = 10^2007 - 1 :=
sorry

end exists_integer_cube_ends_with_2007_ones_l2233_223375


namespace Mrs_Martin_pays_32_l2233_223377

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def num_regular_scoops : ℕ := 2
def num_kiddie_scoops : ℕ := 2
def num_double_scoops : ℕ := 3

def total_cost : ℕ := 
  (num_regular_scoops * regular_scoop_cost) + 
  (num_kiddie_scoops * kiddie_scoop_cost) + 
  (num_double_scoops * double_scoop_cost)

theorem Mrs_Martin_pays_32 :
  total_cost = 32 :=
by
  sorry

end Mrs_Martin_pays_32_l2233_223377


namespace extremum_at_one_and_value_at_two_l2233_223369

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_one_and_value_at_two (a b : ℝ) (h_deriv : 3 + 2*a + b = 0) (h_value : 1 + a + b + a^2 = 10) : 
  f 2 a b = 18 := 
by 
  sorry

end extremum_at_one_and_value_at_two_l2233_223369


namespace distance_between_places_l2233_223393

theorem distance_between_places
  (d : ℝ) -- let d be the distance between A and B
  (v : ℝ) -- let v be the original speed
  (h1 : v * 4 = d) -- initially, speed * time = distance
  (h2 : (v + 20) * 3 = d) -- after speed increase, speed * new time = distance
  : d = 240 :=
sorry

end distance_between_places_l2233_223393


namespace three_brothers_pizza_slices_l2233_223363

theorem three_brothers_pizza_slices :
  let large_pizza_slices := 14
  let small_pizza_slices := 8
  let num_brothers := 3
  let total_slices := small_pizza_slices + 2 * large_pizza_slices
  total_slices / num_brothers = 12 := by
  sorry

end three_brothers_pizza_slices_l2233_223363


namespace snowball_total_distance_l2233_223311

noncomputable def total_distance (a1 d n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

theorem snowball_total_distance :
  total_distance 6 5 25 = 1650 := by
  sorry

end snowball_total_distance_l2233_223311


namespace tan_of_angle_l2233_223344

noncomputable def tan_val (α : ℝ) : ℝ := Real.tan α

theorem tan_of_angle (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (h2 : Real.cos (2 * α) = -3 / 5) :
  tan_val α = -2 := by
  sorry

end tan_of_angle_l2233_223344


namespace harkamal_total_amount_l2233_223391

-- Conditions
def cost_grapes : ℝ := 8 * 80
def cost_mangoes : ℝ := 9 * 55
def cost_apples_before_discount : ℝ := 6 * 120
def cost_oranges : ℝ := 4 * 75
def discount_apples : ℝ := 0.10 * cost_apples_before_discount
def cost_apples_after_discount : ℝ := cost_apples_before_discount - discount_apples

def total_cost_before_tax : ℝ :=
  cost_grapes + cost_mangoes + cost_apples_after_discount + cost_oranges

def sales_tax : ℝ := 0.05 * total_cost_before_tax

def total_amount_paid : ℝ := total_cost_before_tax + sales_tax

-- Question translated into a Lean statement
theorem harkamal_total_amount:
  total_amount_paid = 2187.15 := 
sorry

end harkamal_total_amount_l2233_223391


namespace sufficient_but_not_necessary_condition_l2233_223364

def P (x : ℝ) : Prop := 0 < x ∧ x < 5
def Q (x : ℝ) : Prop := |x - 2| < 3

theorem sufficient_but_not_necessary_condition
  (x : ℝ) : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end sufficient_but_not_necessary_condition_l2233_223364


namespace perfect_square_quadratic_l2233_223314

theorem perfect_square_quadratic (a : ℝ) :
  ∃ (b : ℝ), (x : ℝ) → (x^2 - ax + 16) = (x + b)^2 ∨ (x^2 - ax + 16) = (x - b)^2 → a = 8 ∨ a = -8 :=
by
  sorry

end perfect_square_quadratic_l2233_223314


namespace fabulous_integers_l2233_223330

def is_fabulous (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ a : ℕ, 2 ≤ a ∧ a ≤ n - 1 ∧ (a^n - a) % n = 0

theorem fabulous_integers (n : ℕ) : is_fabulous n ↔ ¬(∃ k : ℕ, n = 2^k ∧ k ≥ 1) := 
sorry

end fabulous_integers_l2233_223330


namespace ivan_running_distance_l2233_223366

theorem ivan_running_distance (x MondayDistance TuesdayDistance WednesdayDistance ThursdayDistance FridayDistance : ℝ) 
  (h1 : MondayDistance = x)
  (h2 : TuesdayDistance = 2 * x)
  (h3 : WednesdayDistance = x)
  (h4 : ThursdayDistance = (1 / 2) * x)
  (h5 : FridayDistance = x)
  (hShortest : ThursdayDistance = 5) :
  MondayDistance + TuesdayDistance + WednesdayDistance + ThursdayDistance + FridayDistance = 55 :=
by
  sorry

end ivan_running_distance_l2233_223366


namespace total_selling_price_correct_l2233_223304

def meters_sold : ℕ := 85
def cost_price_per_meter : ℕ := 80
def profit_per_meter : ℕ := 25

def selling_price_per_meter : ℕ :=
  cost_price_per_meter + profit_per_meter

def total_selling_price : ℕ :=
  selling_price_per_meter * meters_sold

theorem total_selling_price_correct :
  total_selling_price = 8925 := by
  sorry

end total_selling_price_correct_l2233_223304


namespace network_connections_l2233_223319

theorem network_connections (n m : ℕ) (hn : n = 30) (hm : m = 5) 
(h_total_conn : (n * 4) / 2 = 60) : 
60 + m = 65 :=
by
  sorry

end network_connections_l2233_223319


namespace pi_irrational_l2233_223356

theorem pi_irrational :
  ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (π = a / b) :=
by
  sorry

end pi_irrational_l2233_223356


namespace katie_candy_l2233_223398

theorem katie_candy (K : ℕ) (H1 : K + 6 - 9 = 7) : K = 10 :=
by
  sorry

end katie_candy_l2233_223398


namespace postage_stamp_problem_l2233_223320

theorem postage_stamp_problem
  (x y z : ℕ) (h1: y = 10 * x) (h2: x + 2 * y + 5 * z = 100) :
  x = 5 ∧ y = 50 ∧ z = 0 :=
by
  sorry

end postage_stamp_problem_l2233_223320


namespace max_value_expression_l2233_223343

theorem max_value_expression (x : ℝ) : 
  (∃ y : ℝ, y = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16) ∧ 
                ∀ z : ℝ, 
                (∃ x : ℝ, z = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16)) → 
                y ≥ z) → 
  ∃ y : ℝ, y = 1 / 16 := 
sorry

end max_value_expression_l2233_223343


namespace no_solution_condition_l2233_223327

theorem no_solution_condition (n : ℝ) : ¬(∃ x y z : ℝ, n^2 * x + y = 1 ∧ n * y + z = 1 ∧ x + n^2 * z = 1) ↔ n = -1 := 
by {
    sorry
}

end no_solution_condition_l2233_223327


namespace orange_face_probability_correct_l2233_223307

-- Define the number of faces
def total_faces : ℕ := 12
def green_faces : ℕ := 5
def orange_faces : ℕ := 4
def purple_faces : ℕ := 3

-- Define the probability of rolling an orange face
def probability_of_orange_face : ℚ := orange_faces / total_faces

-- Statement of the theorem
theorem orange_face_probability_correct :
  probability_of_orange_face = 1 / 3 :=
by
  sorry

end orange_face_probability_correct_l2233_223307


namespace bob_shucks_240_oysters_in_2_hours_l2233_223308

-- Definitions based on conditions provided:
def oysters_per_minute (oysters : ℕ) (minutes : ℕ) : ℕ :=
  oysters / minutes

def minutes_in_hour : ℕ :=
  60

def oysters_in_two_hours (oysters_per_minute : ℕ) (hours : ℕ) : ℕ :=
  oysters_per_minute * (hours * minutes_in_hour)

-- Parameters given in the problem:
def initial_oysters : ℕ := 10
def initial_minutes : ℕ := 5
def hours : ℕ := 2

-- The main theorem we need to prove:
theorem bob_shucks_240_oysters_in_2_hours :
  oysters_in_two_hours (oysters_per_minute initial_oysters initial_minutes) hours = 240 :=
by
  sorry

end bob_shucks_240_oysters_in_2_hours_l2233_223308


namespace Jason_reroll_probability_optimal_l2233_223323

/-- Represents the action of rerolling dice to achieve a sum of 9 when
    the player optimizes their strategy. The probability 
    that the player chooses to reroll exactly two dice.
 -/
noncomputable def probability_reroll_two_dice : ℚ :=
  13 / 72

/-- Prove that the probability Jason chooses to reroll exactly two
    dice to achieve a sum of 9, given the optimal strategy, is 13/72.
 -/
theorem Jason_reroll_probability_optimal :
  probability_reroll_two_dice = 13 / 72 :=
sorry

end Jason_reroll_probability_optimal_l2233_223323


namespace meaningful_range_l2233_223334

noncomputable def isMeaningful (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x - 2 ≠ 0)

theorem meaningful_range (x : ℝ) : isMeaningful x ↔ (x ≥ -1) ∧ (x ≠ 2) :=
by
  sorry

end meaningful_range_l2233_223334


namespace find_two_digit_number_l2233_223349

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end find_two_digit_number_l2233_223349


namespace interest_equality_l2233_223303

-- Definitions based on the conditions
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

-- Constants for the problem
def P1 : ℝ := 200 -- 200 Rs is the principal of the first case
def r1 : ℝ := 0.1 -- 10% converted to a decimal
def t1 : ℝ := 12 -- 12 years

def P2 : ℝ := 1000 -- Correct answer for the other amount
def r2 : ℝ := 0.12 -- 12% converted to a decimal
def t2 : ℝ := 2 -- 2 years

-- Theorem stating that the interest generated is the same
theorem interest_equality : 
  simple_interest P1 r1 t1 = simple_interest P2 r2 t2 :=
by 
  -- Skip the proof since it is not required
  sorry

end interest_equality_l2233_223303


namespace purely_imaginary_complex_is_two_l2233_223382

theorem purely_imaginary_complex_is_two
  (a : ℝ)
  (h_imag : (a^2 - 3 * a + 2) + (a - 1) * I = (a - 1) * I) :
  a = 2 := by
  sorry

end purely_imaginary_complex_is_two_l2233_223382


namespace power_function_const_coeff_l2233_223384

theorem power_function_const_coeff (m : ℝ) (h1 : m^2 + 2 * m - 2 = 1) (h2 : m ≠ 1) : m = -3 :=
  sorry

end power_function_const_coeff_l2233_223384


namespace expression_value_l2233_223300

   theorem expression_value :
     (20 - (2010 - 201)) + (2010 - (201 - 20)) = 40 := 
   by
     sorry
   
end expression_value_l2233_223300


namespace sin_585_eq_neg_sqrt_two_div_two_l2233_223302

theorem sin_585_eq_neg_sqrt_two_div_two : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end sin_585_eq_neg_sqrt_two_div_two_l2233_223302


namespace part1_part2_l2233_223388

variables (α β : Real)

theorem part1 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos α * Real.cos β = 7 / 12 := 
sorry

theorem part2 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos (2 * α - 2 * β) = 7 / 18 := 
sorry

end part1_part2_l2233_223388


namespace graph_is_two_lines_l2233_223378

theorem graph_is_two_lines : ∀ (x y : ℝ), (x ^ 2 - 25 * y ^ 2 - 20 * x + 100 = 0) ↔ (x = 10 + 5 * y ∨ x = 10 - 5 * y) := 
by 
  intro x y
  sorry

end graph_is_two_lines_l2233_223378


namespace range_of_a_l2233_223365

noncomputable def p (x : ℝ) : Prop := (3*x - 1)/(x - 2) ≤ 1
noncomputable def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) < 0

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, ¬ q x a) → (¬ ∃ x : ℝ, ¬ p x) → -1/2 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l2233_223365


namespace baker_sales_difference_l2233_223345

/-!
  Prove that the difference in dollars between the baker's daily average sales and total sales for today is 48 dollars.
-/

theorem baker_sales_difference :
  let price_pastry := 2
  let price_bread := 4
  let avg_pastries := 20
  let avg_bread := 10
  let today_pastries := 14
  let today_bread := 25
  let daily_avg_sales := avg_pastries * price_pastry + avg_bread * price_bread
  let today_sales := today_pastries * price_pastry + today_bread * price_bread
  daily_avg_sales - today_sales = 48 :=
sorry

end baker_sales_difference_l2233_223345


namespace total_goats_l2233_223341

theorem total_goats (W: ℕ) (H_W: W = 180) (H_P: W + 70 = 250) : W + (W + 70) = 430 :=
by
  -- proof goes here
  sorry

end total_goats_l2233_223341


namespace percentage_greater_l2233_223312

theorem percentage_greater (x : ℝ) (h1 : x = 96) (h2 : x > 80) : ((x - 80) / 80) * 100 = 20 :=
by
  sorry

end percentage_greater_l2233_223312


namespace six_letter_words_no_substring_amc_l2233_223397

theorem six_letter_words_no_substring_amc : 
  let alphabet := ['A', 'M', 'C']
  let totalNumberOfWords := 3^6
  let numberOfWordsContainingAMC := 4 * 3^3 - 1
  let numberOfWordsNotContainingAMC := totalNumberOfWords - numberOfWordsContainingAMC
  numberOfWordsNotContainingAMC = 622 :=
by
  sorry

end six_letter_words_no_substring_amc_l2233_223397


namespace greatest_servings_l2233_223337

def servings (ingredient_amount recipe_amount: ℚ) (recipe_servings: ℕ) : ℚ :=
  (ingredient_amount / recipe_amount) * recipe_servings

theorem greatest_servings (chocolate_new_recipe sugar_new_recipe water_new_recipe milk_new_recipe : ℚ)
                         (servings_new_recipe : ℕ)
                         (chocolate_jordan sugar_jordan milk_jordan : ℚ)
                         (lots_of_water : Prop) :
  chocolate_new_recipe = 3 ∧ sugar_new_recipe = 1/3 ∧ water_new_recipe = 1.5 ∧ milk_new_recipe = 5 ∧
  servings_new_recipe = 6 ∧ chocolate_jordan = 8 ∧ sugar_jordan = 3 ∧ milk_jordan = 12 ∧ lots_of_water →
  max (servings chocolate_jordan chocolate_new_recipe servings_new_recipe)
      (max (servings sugar_jordan sugar_new_recipe servings_new_recipe)
           (servings milk_jordan milk_new_recipe servings_new_recipe)) = 16 :=
by
  sorry

end greatest_servings_l2233_223337


namespace parallel_lines_slope_l2233_223394

theorem parallel_lines_slope (b : ℝ) 
  (h₁ : ∀ x y : ℝ, 3 * y - 3 * b = 9 * x → (b = 3 - 9)) 
  (h₂ : ∀ x y : ℝ, y + 2 = (b + 9) * x → (b = 3 - 9)) : b = -6 :=
by
  sorry

end parallel_lines_slope_l2233_223394


namespace distance_from_A_to_y_axis_l2233_223389

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the distance function from a point to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- State the theorem
theorem distance_from_A_to_y_axis :
  distance_to_y_axis point_A = 3 :=
  by
    -- This part will contain the proof, but we omit it with 'sorry' for now.
    sorry

end distance_from_A_to_y_axis_l2233_223389


namespace solve_for_y_l2233_223373

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 40

theorem solve_for_y 
  (h₁ : 1.5 * x = 0.75 * y) 
  (h₂ : x = 20) : 
  y = 40 :=
by
  sorry

end solve_for_y_l2233_223373


namespace rows_needed_correct_l2233_223335

variable (pencils rows_needed : Nat)

def total_pencils : Nat := 35
def pencils_per_row : Nat := 5
def rows_expected : Nat := 7

theorem rows_needed_correct : rows_needed = total_pencils / pencils_per_row →
  rows_needed = rows_expected := by
  sorry

end rows_needed_correct_l2233_223335


namespace top_square_is_9_l2233_223358

def initial_grid : List (List ℕ) := 
  [[1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]]

def fold_step_1 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col3 := grid.map (fun row => row.get! 2)
  let col2 := grid.map (fun row => row.get! 1)
  [[col1.get! 0, col3.get! 0, col2.get! 0],
   [col1.get! 1, col3.get! 1, col2.get! 1],
   [col1.get! 2, col3.get! 2, col2.get! 2]]

def fold_step_2 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col2 := grid.map (fun row => row.get! 1)
  let col3 := grid.map (fun row => row.get! 2)
  [[col2.get! 0, col1.get! 0, col3.get! 0],
   [col2.get! 1, col1.get! 1, col3.get! 1],
   [col2.get! 2, col1.get! 2, col3.get! 2]]

def fold_step_3 (grid : List (List ℕ)) : List (List ℕ) :=
  let row1 := grid.get! 0
  let row2 := grid.get! 1
  let row3 := grid.get! 2
  [row3, row2, row1]

def folded_grid : List (List ℕ) :=
  fold_step_3 (fold_step_2 (fold_step_1 initial_grid))

theorem top_square_is_9 : folded_grid.get! 0 = [9, 7, 8] :=
  sorry

end top_square_is_9_l2233_223358


namespace fresh_fruit_water_content_l2233_223367

theorem fresh_fruit_water_content (W N : ℝ) 
  (fresh_weight_dried: W + N = 50) 
  (dried_weight: (0.80 * 5) = N) : 
  ((W / (W + N)) * 100 = 92) :=
by
  sorry

end fresh_fruit_water_content_l2233_223367


namespace yoongi_age_l2233_223374

theorem yoongi_age (H Yoongi : ℕ) : H = Yoongi + 2 ∧ H + Yoongi = 18 → Yoongi = 8 :=
by
  sorry

end yoongi_age_l2233_223374


namespace repeating_decimal_fraction_l2233_223360

theorem repeating_decimal_fraction :
  (5 + 341 / 999) = (5336 / 999) :=
by
  sorry

end repeating_decimal_fraction_l2233_223360


namespace students_taking_either_geometry_or_history_but_not_both_l2233_223316

theorem students_taking_either_geometry_or_history_but_not_both
    (students_in_both : ℕ)
    (students_in_geometry : ℕ)
    (students_only_in_history : ℕ)
    (students_in_both_cond : students_in_both = 15)
    (students_in_geometry_cond : students_in_geometry = 35)
    (students_only_in_history_cond : students_only_in_history = 18) :
    (students_in_geometry - students_in_both + students_only_in_history = 38) :=
by
  sorry

end students_taking_either_geometry_or_history_but_not_both_l2233_223316


namespace unoccupied_garden_area_is_correct_l2233_223332

noncomputable def area_unoccupied_by_pond_trees_bench (π : ℝ) : ℝ :=
  let garden_area := 144
  let pond_area_rectangle := 6
  let pond_area_semi_circle := 2 * π
  let trees_area := 3
  let bench_area := 3
  garden_area - (pond_area_rectangle + pond_area_semi_circle + trees_area + bench_area)

theorem unoccupied_garden_area_is_correct : 
  area_unoccupied_by_pond_trees_bench Real.pi = 132 - 2 * Real.pi :=
by
  sorry

end unoccupied_garden_area_is_correct_l2233_223332


namespace sin_thirty_deg_l2233_223340

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end sin_thirty_deg_l2233_223340


namespace rowing_upstream_speed_l2233_223318

theorem rowing_upstream_speed (Vm Vdown : ℝ) (H1 : Vm = 20) (H2 : Vdown = 33) :
  ∃ Vup Vs : ℝ, Vup = Vm - Vs ∧ Vs = Vdown - Vm ∧ Vup = 7 := 
by {
  sorry
}

end rowing_upstream_speed_l2233_223318


namespace how_fast_is_a_l2233_223381

variable (a b : ℝ) (k : ℝ)

theorem how_fast_is_a (h1 : a = k * b) (h2 : a + b = 1 / 30) (h3 : a = 1 / 40) : k = 3 := sorry

end how_fast_is_a_l2233_223381


namespace maximum_students_l2233_223362

-- Definitions for conditions
def students (n : ℕ) := Fin n → Prop

-- Condition: Among any six students, there are two who are not friends
def not_friend_in_six (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 6 → ∃ (a b : Fin n), a ∈ s ∧ b ∈ s ∧ ¬ friend a b

-- Condition: For any pair of students not friends, there is a student who is friends with both
def friend_of_two_not_friends (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b : Fin n), ¬ friend a b → ∃ (c : Fin n), c ≠ a ∧ c ≠ b ∧ friend c a ∧ friend c b

-- Theorem stating the main result
theorem maximum_students (n : ℕ) (friend : Fin n → Fin n → Prop) :
  not_friend_in_six n friend ∧ friend_of_two_not_friends n friend → n ≤ 25 := 
sorry

end maximum_students_l2233_223362


namespace quadratic_function_properties_l2233_223359

-- We define the primary conditions
def axis_of_symmetry (f : ℝ → ℝ) (x_sym : ℝ) : Prop := 
  ∀ x, f x = f (2 * x_sym - x)

def minimum_value (f : ℝ → ℝ) (y_min : ℝ) (x_min : ℝ) : Prop := 
  ∀ x, f x_min ≤ f x

def passes_through (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop := 
  f pt.1 = pt.2

-- We need to prove that a quadratic function exists with the given properties and find intersections
theorem quadratic_function_properties :
  ∃ f : ℝ → ℝ,
    axis_of_symmetry f (-1) ∧
    minimum_value f (-4) (-1) ∧
    passes_through f (-2, 5) ∧
    (∀ y : ℝ, f 0 = y → y = 5) ∧
    (∀ x : ℝ, f x = 0 → (x = -5/3 ∨ x = -1/3)) :=
sorry

end quadratic_function_properties_l2233_223359


namespace kola_age_l2233_223379

variables (x y : ℕ)

-- Condition 1: Kolya is twice as old as Olya was when Kolya was as old as Olya is now
def condition1 : Prop := x = 2 * (2 * y - x)

-- Condition 2: When Olya is as old as Kolya is now, their combined age will be 36 years.
def condition2 : Prop := (3 * x - y = 36)

theorem kola_age : condition1 x y → condition2 x y → x = 16 :=
by { sorry }

end kola_age_l2233_223379


namespace length_of_second_train_is_229_95_l2233_223326

noncomputable def length_of_second_train (length_first_train : ℝ) 
                                          (speed_first_train : ℝ) 
                                          (speed_second_train : ℝ) 
                                          (time_to_cross : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train * (1000 / 3600)
  let speed_second_train_mps := speed_second_train * (1000 / 3600)
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance_covered := relative_speed * time_to_cross
  total_distance_covered - length_first_train

theorem length_of_second_train_is_229_95 :
  length_of_second_train 270 120 80 9 = 229.95 :=
by
  sorry

end length_of_second_train_is_229_95_l2233_223326


namespace inequality_solution_l2233_223309

theorem inequality_solution (x : ℝ) :
  (0 < x ∧ x ≤ 5 / 6 ∨ 2 < x) ↔ 
  ((2 * x) / (x - 2) + (x - 3) / (3 * x) ≥ 2) :=
by
  sorry

end inequality_solution_l2233_223309


namespace corvette_trip_average_rate_l2233_223322

theorem corvette_trip_average_rate (total_distance : ℕ) (first_half_distance : ℕ)
  (first_half_rate : ℕ) (second_half_time_multiplier : ℕ) (total_time : ℕ) :
  total_distance = 640 →
  first_half_distance = total_distance / 2 →
  first_half_rate = 80 →
  second_half_time_multiplier = 3 →
  total_time = (first_half_distance / first_half_rate) + (second_half_time_multiplier * (first_half_distance / first_half_rate)) →
  (total_distance / total_time) = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end corvette_trip_average_rate_l2233_223322


namespace ratio_a_to_c_l2233_223321

variable (a b c : ℕ)

theorem ratio_a_to_c (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
sorry

end ratio_a_to_c_l2233_223321


namespace middle_part_l2233_223306

theorem middle_part (x : ℝ) (h : 2 * x + (2 / 3) * x + (2 / 9) * x = 120) : 
  (2 / 3) * x = 27.6 :=
by
  -- Assuming the given conditions
  sorry

end middle_part_l2233_223306


namespace fraction_of_sum_l2233_223329

theorem fraction_of_sum (P : ℝ) (R : ℝ) (T : ℝ) (H_R : R = 8.333333333333337) (H_T : T = 2) : 
  let SI := (P * R * T) / 100
  let A := P + SI
  A / P = 1.1666666666666667 :=
by
  sorry

end fraction_of_sum_l2233_223329


namespace total_rooms_to_paint_l2233_223371

theorem total_rooms_to_paint :
  ∀ (hours_per_room hours_remaining rooms_painted : ℕ),
    hours_per_room = 7 →
    hours_remaining = 63 →
    rooms_painted = 2 →
    rooms_painted + hours_remaining / hours_per_room = 11 :=
by
  intros
  sorry

end total_rooms_to_paint_l2233_223371


namespace problem1_l2233_223368

theorem problem1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) : 
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end problem1_l2233_223368


namespace coins_difference_l2233_223357

theorem coins_difference (p n d : ℕ) (h1 : p + n + d = 3030)
  (h2 : 1 ≤ p) (h3 : 1 ≤ n) (h4 : 1 ≤ d) (h5 : p ≤ 3029) (h6 : n ≤ 3029) (h7 : d ≤ 3029) :
  (max (p + 5 * n + 10 * d) (max (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) - 
  (min (p + 5 * n + 10 * d) (min (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) = 27243 := 
sorry

end coins_difference_l2233_223357


namespace possible_values_of_a_l2233_223324

theorem possible_values_of_a :
  ∃ a b c : ℤ, 
    (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ↔ 
    (a = 3 ∨ a = 7) :=
by
  sorry

end possible_values_of_a_l2233_223324


namespace find_ordered_pairs_l2233_223313

theorem find_ordered_pairs (a b x : ℕ) (h1 : b > a) (h2 : a + b = 15) (h3 : (a - 2 * x) * (b - 2 * x) = 2 * a * b / 3) :
  (a, b) = (8, 7) :=
by
  sorry

end find_ordered_pairs_l2233_223313


namespace solve_for_A_l2233_223338

def spadesuit (A B : ℝ) : ℝ := 4*A + 3*B + 6

theorem solve_for_A (A : ℝ) : spadesuit A 5 = 79 → A = 14.5 :=
by
  intros h
  sorry

end solve_for_A_l2233_223338


namespace find_integer_a_l2233_223301

-- Definitions based on the conditions
def in_ratio (x y z : ℕ) := ∃ k : ℕ, x = 3 * k ∧ y = 4 * k ∧ z = 7 * k
def satisfies_equation (z : ℕ) (a : ℕ) := z = 30 * a - 15

-- The proof problem statement
theorem find_integer_a (x y z : ℕ) (a : ℕ) :
  in_ratio x y z →
  satisfies_equation z a →
  (∃ a : ℕ, a = 4) :=
by
  intros h1 h2
  sorry

end find_integer_a_l2233_223301


namespace equation_of_line_AB_l2233_223354

-- Definition of the given circle
def circle1 : Type := { p : ℝ × ℝ // p.1^2 + (p.2 - 2)^2 = 4 }

-- Definition of the center and point on the second circle
def center : ℝ × ℝ := (0, 2)
def point : ℝ × ℝ := (-2, 6)

-- Definition of the second circle with diameter endpoints
def circle2_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 4)^2 = 5

-- Statement to be proved
theorem equation_of_line_AB :
  ∃ x y : ℝ, (x^2 + (y - 2)^2 = 4) ∧ ((x + 1)^2 + (y - 4)^2 = 5) ∧ (x - 2*y + 6 = 0) := 
sorry

end equation_of_line_AB_l2233_223354
