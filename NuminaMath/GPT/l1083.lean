import Mathlib

namespace length_of_crease_correct_l1083_108385

noncomputable def length_of_crease (theta : ℝ) : ℝ := Real.sqrt (40 + 24 * Real.cos theta)

theorem length_of_crease_correct (theta : ℝ) : 
  length_of_crease theta = Real.sqrt (40 + 24 * Real.cos theta) := 
by 
  sorry

end length_of_crease_correct_l1083_108385


namespace partial_fraction_sum_l1083_108311

theorem partial_fraction_sum :
  (∃ A B C D E : ℝ, 
    (∀ x : ℝ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -5 → 
    (1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5))) ∧
    (A + B + C + D + E = 1 / 30)) :=
sorry

end partial_fraction_sum_l1083_108311


namespace joanna_reading_rate_l1083_108339

variable (P : ℝ)

theorem joanna_reading_rate (h : 3 * P + 6.5 * P + 6 * P = 248) : P = 16 := by
  sorry

end joanna_reading_rate_l1083_108339


namespace molecular_weight_of_NH4Cl_l1083_108343

theorem molecular_weight_of_NH4Cl (weight_8_moles : ℕ) (weight_per_mole : ℕ) :
  weight_8_moles = 424 →
  weight_per_mole = 53 →
  weight_8_moles / 8 = weight_per_mole :=
by
  intro h1 h2
  sorry

end molecular_weight_of_NH4Cl_l1083_108343


namespace floor_eq_solution_l1083_108388

theorem floor_eq_solution (x : ℝ) : 2.5 ≤ x ∧ x < 3.5 → (⌊2 * x + 0.5⌋ = ⌊x + 3⌋) :=
by
  sorry

end floor_eq_solution_l1083_108388


namespace max_profit_price_l1083_108300

-- Define the initial conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 90
def initial_sales_volume : ℝ := 400
def price_increase_effect : ℝ := 1
def sales_volume_decrease : ℝ := 20

-- Define the profit function
def profit (x : ℝ) : ℝ :=
  let selling_price := initial_selling_price + x
  let sales_volume := initial_sales_volume - x * sales_volume_decrease
  let profit_per_item := selling_price - purchase_price
  profit_per_item * sales_volume

-- The statement that needs to be proved
theorem max_profit_price : ∃ x : ℝ, x = 10 ∧ (initial_selling_price + x = 100) := by
  sorry

end max_profit_price_l1083_108300


namespace guarantee_min_points_l1083_108376

-- Define points for positions
def points_for_position (pos : ℕ) : ℕ :=
  if pos = 1 then 6
  else if pos = 2 then 4
  else if pos = 3 then 2
  else 0

-- Define the maximum points
def max_points_per_race := 6
def races := 4
def max_points := max_points_per_race * races

-- Define the condition of no ties
def no_ties := true

-- Define the problem statement
theorem guarantee_min_points (no_ties: true) (h1: points_for_position 1 = 6)
  (h2: points_for_position 2 = 4) (h3: points_for_position 3 = 2)
  (h4: max_points = 24) : 
  ∃ min_points, (min_points = 22) ∧ (∀ points, (points < min_points) → (∃ another_points, (another_points > points))) :=
  sorry

end guarantee_min_points_l1083_108376


namespace slices_per_pizza_l1083_108346

theorem slices_per_pizza (total_slices number_of_pizzas slices_per_pizza : ℕ) 
  (h_total_slices : total_slices = 168) 
  (h_number_of_pizzas : number_of_pizzas = 21) 
  (h_division : total_slices / number_of_pizzas = slices_per_pizza) : 
  slices_per_pizza = 8 :=
sorry

end slices_per_pizza_l1083_108346


namespace problem1_problem2_l1083_108338

variable {m x : ℝ}

-- Definition of the function f
def f (x m : ℝ) : ℝ := |x - m| + |x|

-- Statement for Problem (1)
theorem problem1 (h : f 1 m = 1) : 
  ∀ x, f x 1 < 2 ↔ (-1 / 2) < x ∧ x < (3 / 2) := 
sorry

-- Statement for Problem (2)
theorem problem2 (h : ∀ x, f x m ≥ m^2) : 
  -1 ≤ m ∧ m ≤ 1 := 
sorry

end problem1_problem2_l1083_108338


namespace profit_function_expression_l1083_108361

def dailySalesVolume (x : ℝ) : ℝ := 300 + 3 * (99 - x)

def profitPerItem (x : ℝ) : ℝ := x - 50

def dailyProfit (x : ℝ) : ℝ := (x - 50) * (300 + 3 * (99 - x))

theorem profit_function_expression (x : ℝ) :
  dailyProfit x = (x - 50) * dailySalesVolume x :=
by sorry

end profit_function_expression_l1083_108361


namespace find_interest_rate_l1083_108352

theorem find_interest_rate
  (P A : ℝ) (n t : ℕ) (r : ℝ)
  (hP : P = 100)
  (hA : A = 121.00000000000001)
  (hn : n = 2)
  (ht : t = 1)
  (compound_interest : A = P * (1 + r / n) ^ (n * t)) :
  r = 0.2 :=
by
  sorry

end find_interest_rate_l1083_108352


namespace speed_at_perigee_l1083_108380

-- Define the conditions
def semi_major_axis (a : ℝ) := a > 0
def perigee_distance (a : ℝ) := 0.5 * a
def point_P_distance (a : ℝ) := 0.75 * a
def speed_at_P (v1 : ℝ) := v1 > 0

-- Define what we need to prove
theorem speed_at_perigee (a v1 v2 : ℝ) (h1 : semi_major_axis a) (h2 : speed_at_P v1) :
  v2 = (3 / Real.sqrt 5) * v1 :=
sorry

end speed_at_perigee_l1083_108380


namespace gain_percentage_is_twenty_l1083_108308

theorem gain_percentage_is_twenty (SP CP Gain : ℝ) (h0 : SP = 90) (h1 : Gain = 15) (h2 : SP = CP + Gain) : (Gain / CP) * 100 = 20 :=
by
  sorry

end gain_percentage_is_twenty_l1083_108308


namespace tax_collection_amount_l1083_108362

theorem tax_collection_amount (paid_tax : ℝ) (willam_percentage : ℝ) (total_collected : ℝ) (h_paid: paid_tax = 480) (h_percentage: willam_percentage = 0.3125) :
    total_collected = 1536 :=
by
  sorry

end tax_collection_amount_l1083_108362


namespace xyz_sum_l1083_108349

theorem xyz_sum (x y z : ℕ) (h1 : xyz = 240) (h2 : xy + z = 46) (h3 : x + yz = 64) : x + y + z = 20 :=
sorry

end xyz_sum_l1083_108349


namespace expression_value_l1083_108367

theorem expression_value (x : ℤ) (h : x = -2) : x ^ 2 + 6 * x - 8 = -16 := 
by 
  rw [h]
  sorry

end expression_value_l1083_108367


namespace angle_complement_supplement_l1083_108337

theorem angle_complement_supplement (θ : ℝ) (h1 : 90 - θ = (1/3) * (180 - θ)) : θ = 45 :=
by
  sorry

end angle_complement_supplement_l1083_108337


namespace operation_preserves_remainder_l1083_108372

theorem operation_preserves_remainder (N : ℤ) (k : ℤ) (m : ℤ) 
(f : ℤ → ℤ) (hN : N = 6 * k + 3) (hf : f N = 6 * m + 3) : f N % 6 = 3 :=
by
  sorry

end operation_preserves_remainder_l1083_108372


namespace solve_inner_parentheses_l1083_108370

theorem solve_inner_parentheses (x : ℝ) : 
  45 - (28 - (37 - (15 - x))) = 57 ↔ x = 18 := by
  sorry

end solve_inner_parentheses_l1083_108370


namespace proof_of_expression_value_l1083_108306

theorem proof_of_expression_value (m n : ℝ) 
  (h1 : m^2 - 2019 * m = 1) 
  (h2 : n^2 - 2019 * n = 1) : 
  (m^2 - 2019 * m + 3) * (n^2 - 2019 * n + 4) = 20 := 
by 
  sorry

end proof_of_expression_value_l1083_108306


namespace find_a_extremum_and_min_value_find_max_k_l1083_108396

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x + 1

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a

theorem find_a_extremum_and_min_value :
  (∀ a : ℝ, f' a 0 = 0 → a = -1) ∧
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → f (-1) x ≥ 2) :=
by sorry

theorem find_max_k (k : ℤ) :
  (∀ x : ℝ, 0 < x → k * (Real.exp x - 1) < x * Real.exp x + 1) →
  k ≤ 2 :=
by sorry

end find_a_extremum_and_min_value_find_max_k_l1083_108396


namespace ribbons_count_l1083_108316

theorem ribbons_count (ribbons : ℕ) 
  (yellow_frac purple_frac orange_frac : ℚ)
  (black_ribbons : ℕ)
  (h1 : yellow_frac = 1/4)
  (h2 : purple_frac = 1/3)
  (h3 : orange_frac = 1/6)
  (h4 : ribbons - (yellow_frac * ribbons + purple_frac * ribbons + orange_frac * ribbons) = black_ribbons) :
  ribbons * orange_frac = 160 / 6 :=
by {
  sorry
}

end ribbons_count_l1083_108316


namespace average_speed_l1083_108364

-- Defining conditions
def speed_first_hour : ℕ := 100  -- The car travels 100 km in the first hour
def speed_second_hour : ℕ := 60  -- The car travels 60 km in the second hour
def total_distance : ℕ := speed_first_hour + speed_second_hour  -- Total distance traveled

def total_time : ℕ := 2  -- Total time taken in hours

-- Stating the theorem
theorem average_speed : total_distance / total_time = 80 := 
by
  sorry

end average_speed_l1083_108364


namespace _l1083_108304

open Nat

/-- Function to check the triangle inequality theorem -/
def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : canFormTriangle 6 4 5 := by
  /- Proof omitted -/
  sorry

end _l1083_108304


namespace largest_of_four_consecutive_odd_numbers_l1083_108371

theorem largest_of_four_consecutive_odd_numbers (x : ℤ) : 
  (x % 2 = 1) → 
  ((x + (x + 2) + (x + 4) + (x + 6)) / 4 = 24) →
  (x + 6 = 27) :=
by
  sorry

end largest_of_four_consecutive_odd_numbers_l1083_108371


namespace part1_l1083_108389

theorem part1 (n : ℕ) (m : ℕ) (h_form : m = 2 ^ (n - 2) * 5 ^ n) (h : 6 * 10 ^ n + m = 25 * m) :
  ∃ k : ℕ, 6 * 10 ^ n + m = 625 * 10 ^ (n - 2) :=
by
  sorry

end part1_l1083_108389


namespace toms_weekly_revenue_l1083_108305

def crabs_per_bucket : Nat := 12
def number_of_buckets : Nat := 8
def price_per_crab : Nat := 5
def days_per_week : Nat := 7

theorem toms_weekly_revenue :
  (crabs_per_bucket * number_of_buckets * price_per_crab * days_per_week) = 3360 :=
by
  sorry

end toms_weekly_revenue_l1083_108305


namespace diana_age_is_8_l1083_108301

noncomputable def age_of_grace_last_year : ℕ := 3
noncomputable def age_of_grace_today : ℕ := age_of_grace_last_year + 1
noncomputable def age_of_diana_today : ℕ := 2 * age_of_grace_today

theorem diana_age_is_8 : age_of_diana_today = 8 :=
by
  -- The proof would go here
  sorry

end diana_age_is_8_l1083_108301


namespace pizzeria_provolone_shred_l1083_108321

theorem pizzeria_provolone_shred 
    (cost_blend : ℝ) 
    (cost_mozzarella : ℝ) 
    (cost_romano : ℝ) 
    (cost_provolone : ℝ) 
    (prop_mozzarella : ℝ) 
    (prop_romano : ℝ) 
    (prop_provolone : ℝ) 
    (shredded_mozzarella : ℕ) 
    (shredded_romano : ℕ) 
    (shredded_provolone_needed : ℕ) :
  cost_blend = 696.05 ∧ 
  cost_mozzarella = 504.35 ∧ 
  cost_romano = 887.75 ∧ 
  cost_provolone = 735.25 ∧ 
  prop_mozzarella = 2 ∧ 
  prop_romano = 1 ∧ 
  prop_provolone = 2 ∧ 
  shredded_mozzarella = 20 ∧ 
  shredded_romano = 10 → 
  shredded_provolone_needed = 20 :=
by {
  sorry -- proof to be provided
}

end pizzeria_provolone_shred_l1083_108321


namespace max_product_l1083_108359

theorem max_product (a b : ℝ) (h1 : 9 * a ^ 2 + 16 * b ^ 2 = 25) (h2 : a > 0) (h3 : b > 0) :
  a * b ≤ 25 / 24 :=
sorry

end max_product_l1083_108359


namespace train_speed_in_m_per_s_l1083_108318

theorem train_speed_in_m_per_s (speed_kmph : ℕ) (h : speed_kmph = 162) :
  (speed_kmph * 1000) / 3600 = 45 :=
by {
  sorry
}

end train_speed_in_m_per_s_l1083_108318


namespace candidate_percentage_l1083_108333

theorem candidate_percentage (P : ℝ) (l : ℝ) (V : ℝ) : 
  l = 5000.000000000007 ∧ 
  V = 25000.000000000007 ∧ 
  V - 2 * (P / 100) * V = l →
  P = 40 :=
by
  sorry

end candidate_percentage_l1083_108333


namespace mike_laptop_row_division_impossible_l1083_108358

theorem mike_laptop_row_division_impossible (total_laptops : ℕ) (num_rows : ℕ) 
(types_ratios : List ℕ)
(H_total : total_laptops = 44)
(H_rows : num_rows = 5) 
(H_ratio : types_ratios = [2, 3, 4]) :
  ¬ (∃ (n : ℕ), (total_laptops = n * num_rows) 
  ∧ (n % (types_ratios.sum) = 0)
  ∧ (∀ (t : ℕ), t ∈ types_ratios → t ≤ n)) := sorry

end mike_laptop_row_division_impossible_l1083_108358


namespace janet_faster_playtime_l1083_108382

theorem janet_faster_playtime 
  (initial_minutes : ℕ)
  (initial_seconds : ℕ)
  (faster_rate : ℝ)
  (initial_time_in_seconds := initial_minutes * 60 + initial_seconds)
  (target_time_in_seconds := initial_time_in_seconds / faster_rate) :
  initial_minutes = 3 →
  initial_seconds = 20 →
  faster_rate = 1.25 →
  target_time_in_seconds = 160 :=
by
  intros h1 h2 h3
  sorry

end janet_faster_playtime_l1083_108382


namespace tom_speed_first_part_l1083_108391

-- Definitions of conditions in Lean
def total_distance : ℕ := 20
def distance_first_part : ℕ := 10
def speed_second_part : ℕ := 10
def average_speed : ℚ := 10.909090909090908
def distance_second_part := total_distance - distance_first_part

-- Lean statement to prove the speed during the first part of the trip
theorem tom_speed_first_part (v : ℚ) :
  (distance_first_part / v + distance_second_part / speed_second_part) = total_distance / average_speed → v = 12 :=
by
  intro h
  sorry

end tom_speed_first_part_l1083_108391


namespace combined_mean_correct_l1083_108374

section MeanScore

variables (score_first_section mean_first_section : ℝ)
variables (score_second_section mean_second_section : ℝ)
variables (num_first_section num_second_section : ℝ)

axiom mean_first : mean_first_section = 92
axiom mean_second : mean_second_section = 78
axiom ratio_students : num_first_section / num_second_section = 5 / 7

noncomputable def combined_mean_score : ℝ := 
  let total_score := (mean_first_section * num_first_section + mean_second_section * num_second_section)
  let total_students := (num_first_section + num_second_section)
  total_score / total_students

theorem combined_mean_correct : combined_mean_score 92 78 (5 / 7 * num_second_section) num_second_section = 83.8 := by
  sorry

end MeanScore

end combined_mean_correct_l1083_108374


namespace combined_length_of_trains_is_correct_l1083_108340

noncomputable def combined_length_of_trains : ℕ :=
  let speed_A := 120 * 1000 / 3600 -- speed of train A in m/s
  let speed_B := 100 * 1000 / 3600 -- speed of train B in m/s
  let speed_motorbike := 64 * 1000 / 3600 -- speed of motorbike in m/s
  let relative_speed_A := (120 - 64) * 1000 / 3600 -- relative speed of train A with respect to motorbike in m/s
  let relative_speed_B := (100 - 64) * 1000 / 3600 -- relative speed of train B with respect to motorbike in m/s
  let length_A := relative_speed_A * 75 -- length of train A in meters
  let length_B := relative_speed_B * 90 -- length of train B in meters
  length_A + length_B

theorem combined_length_of_trains_is_correct :
  combined_length_of_trains = 2067 :=
  by
  sorry

end combined_length_of_trains_is_correct_l1083_108340


namespace taehyung_walks_more_than_minyoung_l1083_108363

def taehyung_distance_per_minute : ℕ := 114
def minyoung_distance_per_minute : ℕ := 79
def minutes_per_hour : ℕ := 60

theorem taehyung_walks_more_than_minyoung :
  (taehyung_distance_per_minute * minutes_per_hour) -
  (minyoung_distance_per_minute * minutes_per_hour) = 2100 := by
  sorry

end taehyung_walks_more_than_minyoung_l1083_108363


namespace problem_statement_l1083_108383

variable {a : ℕ → ℝ} 
variable {a1 d : ℝ}
variable (h_arith : ∀ n, a (n + 1) = a n + d)  -- Arithmetic sequence condition
variable (h_d_nonzero : d ≠ 0)  -- d ≠ 0
variable (h_a1_nonzero : a1 ≠ 0)  -- a1 ≠ 0
variable (h_geom : (a 1) * (a 7) = (a 3) ^ 2)  -- Geometric sequence condition a2 = a 1, a4 = a 3, a8 = a 7

theorem problem_statement :
  (a 0 + a 4 + a 8) / (a 1 + a 2) = 3 :=
by
  sorry

end problem_statement_l1083_108383


namespace coefficient_of_x_in_expression_l1083_108373

theorem coefficient_of_x_in_expression : 
  let expr := 2 * (x - 5) + 5 * (8 - 3 * x^2 + 6 * x) - 9 * (3 * x - 2) + 3 * (x + 4)
  ∃ k : ℤ, (expr = k * x + term) ∧ 
  (∃ coefficient_x : ℤ, coefficient_x = 8) := 
sorry

end coefficient_of_x_in_expression_l1083_108373


namespace discount_percentage_l1083_108330

theorem discount_percentage (wm_cost dryer_cost after_discount before_discount discount_amount : ℝ)
    (h0 : wm_cost = 100) 
    (h1 : dryer_cost = wm_cost - 30) 
    (h2 : after_discount = 153) 
    (h3 : before_discount = wm_cost + dryer_cost) 
    (h4 : discount_amount = before_discount - after_discount) 
    (h5 : (discount_amount / before_discount) * 100 = 10) : 
    True := sorry

end discount_percentage_l1083_108330


namespace number_of_red_balls_l1083_108329

theorem number_of_red_balls (total_balls : ℕ) (probability : ℚ) (num_red_balls : ℕ) 
  (h1 : total_balls = 12) 
  (h2 : probability = 1 / 22) 
  (h3 : (num_red_balls * (num_red_balls - 1) : ℚ) / (total_balls * (total_balls - 1)) = probability) :
  num_red_balls = 3 := 
by
  sorry

end number_of_red_balls_l1083_108329


namespace similar_triangle_perimeter_l1083_108354

noncomputable def is_similar_triangles (a b c a' b' c' : ℝ) := 
  ∃ (k : ℝ), k > 0 ∧ (a = k * a') ∧ (b = k * b') ∧ (c = k * c')

noncomputable def is_isosceles (a b c : ℝ) := (a = b) ∨ (a = c) ∨ (b = c)

theorem similar_triangle_perimeter :
  ∀ (a b c a' b' c' : ℝ),
    is_isosceles a b c → 
    is_similar_triangles a b c a' b' c' →
    c' = 42 →
    (a = 12) → 
    (b = 12) → 
    (c = 14) →
    (b' = 36) →
    (a' = 36) →
    a' + b' + c' = 114 :=
by
  intros
  sorry

end similar_triangle_perimeter_l1083_108354


namespace a_minus_b_is_neg_seven_l1083_108398

-- Definitions for sets
def setA : Set ℝ := {x | -2 < x ∧ x < 3}
def setB : Set ℝ := {x | 1 < x ∧ x < 4}
def setC : Set ℝ := {x | 1 < x ∧ x < 3}

-- Proving the statement
theorem a_minus_b_is_neg_seven :
  ∀ (a b : ℝ), (∀ x, (x ∈ setC) ↔ (x^2 + a*x + b < 0)) → a - b = -7 :=
by
  intros a b h
  sorry

end a_minus_b_is_neg_seven_l1083_108398


namespace range_of_a_l1083_108314

def satisfies_p (x : ℝ) : Prop := (2 * x - 1) / (x - 1) ≤ 0

def satisfies_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) < 0

def sufficient_but_not_necessary (p q : ℝ → Prop) : Prop :=
  ∀ x, p x → q x ∧ ∃ x, q x ∧ ¬(p x)

theorem range_of_a :
  (∀ (x a : ℝ), satisfies_p x → satisfies_q x a → 0 ≤ a ∧ a < 1 / 2) ↔ (∀ a, 0 ≤ a ∧ a < 1 / 2) := by sorry

end range_of_a_l1083_108314


namespace john_saved_120_dollars_l1083_108303

-- Defining the conditions
def num_machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := num_machines * ball_bearings_per_machine
def regular_price_per_bearing : ℝ := 1
def sale_price_per_bearing : ℝ := 0.75
def bulk_discount : ℝ := 0.20
def discounted_price_per_bearing : ℝ := sale_price_per_bearing - (bulk_discount * sale_price_per_bearing)

-- Calculate total costs
def total_cost_without_sale : ℝ := total_ball_bearings * regular_price_per_bearing
def total_cost_with_sale : ℝ := total_ball_bearings * discounted_price_per_bearing

-- Calculate the savings
def savings : ℝ := total_cost_without_sale - total_cost_with_sale

-- The theorem we want to prove
theorem john_saved_120_dollars : savings = 120 := by
  sorry

end john_saved_120_dollars_l1083_108303


namespace sum_div_mult_sub_result_l1083_108313

-- Define the problem with conditions and expected answer
theorem sum_div_mult_sub_result :
  3521 + 480 / 60 * 3 - 521 = 3024 :=
by 
  sorry

end sum_div_mult_sub_result_l1083_108313


namespace initial_number_of_girls_l1083_108336

theorem initial_number_of_girls (n A : ℕ) (new_girl_weight : ℕ := 80) (original_girl_weight : ℕ := 40)
  (avg_increase : ℕ := 2)
  (condition : n * (A + avg_increase) - n * A = 40) :
  n = 20 :=
by
  sorry

end initial_number_of_girls_l1083_108336


namespace porch_width_l1083_108328

theorem porch_width (L_house W_house total_area porch_length W : ℝ)
  (h1 : L_house = 20.5) (h2 : W_house = 10) (h3 : total_area = 232) (h4 : porch_length = 6) (h5 : total_area = (L_house * W_house) + (porch_length * W)) :
  W = 4.5 :=
by 
  sorry

end porch_width_l1083_108328


namespace charity_event_revenue_l1083_108377

theorem charity_event_revenue :
  ∃ (f t p : ℕ), f + t = 190 ∧ f * p + t * (p / 3) = 2871 ∧ f * p = 1900 :=
by
  sorry

end charity_event_revenue_l1083_108377


namespace compute_expression_l1083_108365

theorem compute_expression : 1013^2 - 987^2 - 1007^2 + 993^2 = 24000 := by
  sorry

end compute_expression_l1083_108365


namespace Alyssa_total_spent_l1083_108356

-- Declare the costs of grapes and cherries.
def costOfGrapes : ℝ := 12.08
def costOfCherries : ℝ := 9.85

-- Total amount spent by Alyssa.
def totalSpent : ℝ := 21.93

-- Statement to prove that the sum of the costs is equal to the total spent.
theorem Alyssa_total_spent (g : ℝ) (c : ℝ) (t : ℝ) 
  (hg : g = costOfGrapes) 
  (hc : c = costOfCherries) 
  (ht : t = totalSpent) :
  g + c = t := by
  sorry

end Alyssa_total_spent_l1083_108356


namespace mariela_cards_total_l1083_108375

theorem mariela_cards_total : 
  let a := 287.0
  let b := 116
  a + b = 403 := 
by
  sorry

end mariela_cards_total_l1083_108375


namespace sequence_expression_l1083_108317

noncomputable def seq (n : ℕ) : ℝ := 
  match n with
  | 0 => 1  -- note: indexing from 1 means a_1 corresponds to seq 0 in Lean
  | m+1 => seq m / (3 * seq m + 1)

theorem sequence_expression (n : ℕ) : 
  ∀ n, seq (n + 1) = 1 / (3 * (n + 1) - 2) := 
sorry

end sequence_expression_l1083_108317


namespace additional_bags_at_max_weight_l1083_108366

/-
Constants representing the problem conditions.
-/
def num_people : Nat := 6
def bags_per_person : Nat := 5
def max_weight_per_bag : Nat := 50
def total_weight_capacity : Nat := 6000

/-
Calculate the total existing luggage weight.
-/
def total_existing_bags : Nat := num_people * bags_per_person
def total_existing_weight : Nat := total_existing_bags * max_weight_per_bag
def remaining_weight_capacity : Nat := total_weight_capacity - total_existing_weight

/-
The proof statement asserting that given the conditions, 
the airplane can hold 90 more bags at maximum weight.
-/
theorem additional_bags_at_max_weight : remaining_weight_capacity / max_weight_per_bag = 90 := by
  sorry

end additional_bags_at_max_weight_l1083_108366


namespace download_speeds_l1083_108332

theorem download_speeds (x : ℕ) (s4 : ℕ := 4) (s5 : ℕ := 60) :
  (600 / x - 600 / (15 * x) = 140) → (x = s4 ∧ 15 * x = s5) := by
  sorry

end download_speeds_l1083_108332


namespace proof_problem_l1083_108351

open Classical

variable (x y z : ℝ)

theorem proof_problem
  (cond1 : 0 < x ∧ x < 1)
  (cond2 : 0 < y ∧ y < 1)
  (cond3 : 0 < z ∧ z < 1)
  (cond4 : x * y * z = (1 - x) * (1 - y) * (1 - z)) :
  ((1 - x) * y ≥ 1/4) ∨ ((1 - y) * z ≥ 1/4) ∨ ((1 - z) * x ≥ 1/4) := by
  sorry

end proof_problem_l1083_108351


namespace sum_of_reciprocals_of_squares_l1083_108348

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 3) : (1 : ℚ)/a^2 + (1 : ℚ)/b^2 = 10/9 := 
sorry

end sum_of_reciprocals_of_squares_l1083_108348


namespace disproves_proposition_l1083_108390

theorem disproves_proposition (a b : ℤ) (h₁ : a = -4) (h₂ : b = 3) : (a^2 > b^2) ∧ ¬ (a > b) :=
by
  sorry

end disproves_proposition_l1083_108390


namespace sum_of_integers_l1083_108357

/-- Given two positive integers x and y such that the sum of their squares equals 181 
    and their product equals 90, prove that the sum of these two integers is 19. -/
theorem sum_of_integers (x y : ℤ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_integers_l1083_108357


namespace cost_per_pound_peanuts_l1083_108369

-- Defining the conditions as needed for our problem
def one_dollar_bills := 7
def five_dollar_bills := 4
def ten_dollar_bills := 2
def twenty_dollar_bills := 1
def change := 4
def pounds_per_day := 3
def days_in_week := 7

-- Calculating the total initial amount of money Frank has
def total_initial_money := (one_dollar_bills * 1) + (five_dollar_bills * 5) + (ten_dollar_bills * 10) + (twenty_dollar_bills * 20)

-- Calculating the total amount spent on peanuts
def total_spent := total_initial_money - change

-- Calculating the total pounds of peanuts
def total_pounds := pounds_per_day * days_in_week

-- The proof statement
theorem cost_per_pound_peanuts : total_spent / total_pounds = 3 := sorry

end cost_per_pound_peanuts_l1083_108369


namespace remainder_of_sum_is_12_l1083_108319

theorem remainder_of_sum_is_12 (D k1 k2 : ℤ) (h1 : 242 = k1 * D + 4) (h2 : 698 = k2 * D + 8) : (242 + 698) % D = 12 :=
by
  sorry

end remainder_of_sum_is_12_l1083_108319


namespace people_sharing_bill_l1083_108341

theorem people_sharing_bill (total_bill : ℝ) (tip_percent : ℝ) (share_per_person : ℝ) (n : ℝ) :
  total_bill = 211.00 →
  tip_percent = 0.15 →
  share_per_person = 26.96 →
  abs (n - 9) < 1 :=
by
  intros h1 h2 h3
  sorry

end people_sharing_bill_l1083_108341


namespace minimum_value_l1083_108323

open Real

-- Statement of the conditions
def conditions (a b c : ℝ) : Prop :=
  -0.5 < a ∧ a < 0.5 ∧ -0.5 < b ∧ b < 0.5 ∧ -0.5 < c ∧ c < 0.5

-- Expression to be minimized
noncomputable def expression (a b c : ℝ) : ℝ :=
  1 / ((1 - a) * (1 - b) * (1 - c)) + 1 / ((1 + a) * (1 + b) * (1 + c))

-- Minimum value to prove
theorem minimum_value (a b c : ℝ) (h : conditions a b c) : expression a b c ≥ 4.74 :=
sorry

end minimum_value_l1083_108323


namespace mean_of_remaining_four_numbers_l1083_108355

theorem mean_of_remaining_four_numbers (a b c d : ℝ) (h: (a + b + c + d + 105) / 5 = 90) :
  (a + b + c + d) / 4 = 86.25 :=
by
  sorry

end mean_of_remaining_four_numbers_l1083_108355


namespace volleyball_shotput_cost_l1083_108334

theorem volleyball_shotput_cost (x y : ℝ) :
  (2*x + 3*y = 95) ∧ (5*x + 7*y = 230) :=
  sorry

end volleyball_shotput_cost_l1083_108334


namespace range_m_l1083_108393

-- Definitions for propositions p and q
def p (m : ℝ) : Prop :=
  (m^2 - 4 > 0) ∧ (-m < 0)

def q (m : ℝ) : Prop :=
  (m - 1 > 0)

-- Given conditions:
-- 1. p ∨ q is true
-- 2. p ∧ q is false

theorem range_m (m : ℝ) (h1: p m ∨ q m) (h2: ¬ (p m ∧ q m)) : 1 < m ∧ m ≤ 2 :=
by
  sorry

end range_m_l1083_108393


namespace find_a_equiv_l1083_108302

theorem find_a_equiv (a x : ℝ) (h : ∀ x, (a * x^2 + 20 * x + 25) = (2 * x + 5) * (2 * x + 5)) : a = 4 :=
by
  sorry

end find_a_equiv_l1083_108302


namespace arithmetic_sequence_sum_l1083_108324

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℤ) (S_n : ℕ → ℤ),
  (∀ n : ℕ, S_n n = (n * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) / 2) →
  S_n 17 = 170 →
  a_n 7 + a_n 8 + a_n 12 = 30 := 
by
  sorry

end arithmetic_sequence_sum_l1083_108324


namespace total_vehicles_l1083_108342

theorem total_vehicles (morn_minivans afternoon_minivans evening_minivans night_minivans : Nat)
                       (morn_sedans afternoon_sedans evening_sedans night_sedans : Nat)
                       (morn_SUVs afternoon_SUVs evening_SUVs night_SUVs : Nat)
                       (morn_trucks afternoon_trucks evening_trucks night_trucks : Nat)
                       (morn_motorcycles afternoon_motorcycles evening_motorcycles night_motorcycles : Nat) :
                       morn_minivans = 20 → afternoon_minivans = 22 → evening_minivans = 15 → night_minivans = 10 →
                       morn_sedans = 17 → afternoon_sedans = 13 → evening_sedans = 19 → night_sedans = 12 →
                       morn_SUVs = 12 → afternoon_SUVs = 15 → evening_SUVs = 18 → night_SUVs = 20 →
                       morn_trucks = 8 → afternoon_trucks = 10 → evening_trucks = 14 → night_trucks = 20 →
                       morn_motorcycles = 5 → afternoon_motorcycles = 7 → evening_motorcycles = 10 → night_motorcycles = 15 →
                       morn_minivans + afternoon_minivans + evening_minivans + night_minivans +
                       morn_sedans + afternoon_sedans + evening_sedans + night_sedans +
                       morn_SUVs + afternoon_SUVs + evening_SUVs + night_SUVs +
                       morn_trucks + afternoon_trucks + evening_trucks + night_trucks +
                       morn_motorcycles + afternoon_motorcycles + evening_motorcycles + night_motorcycles = 282 :=
by
  intros
  sorry

end total_vehicles_l1083_108342


namespace man_rowing_upstream_speed_l1083_108307

theorem man_rowing_upstream_speed (V_down V_m V_up V_s : ℕ) 
  (h1 : V_down = 41)
  (h2 : V_m = 33)
  (h3 : V_down = V_m + V_s)
  (h4 : V_up = V_m - V_s) 
  : V_up = 25 := 
by
  sorry

end man_rowing_upstream_speed_l1083_108307


namespace distance_between_points_l1083_108335

open Real

theorem distance_between_points :
  ∀ (x1 y1 x2 y2 : ℝ),
  (x1, y1) = (-3, 1) →
  (x2, y2) = (5, -5) →
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 10 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end distance_between_points_l1083_108335


namespace stella_doll_price_l1083_108344

theorem stella_doll_price 
  (dolls_count clocks_count glasses_count : ℕ)
  (price_per_clock price_per_glass cost profit : ℕ)
  (D : ℕ)
  (h1 : dolls_count = 3)
  (h2 : clocks_count = 2)
  (h3 : glasses_count = 5)
  (h4 : price_per_clock = 15)
  (h5 : price_per_glass = 4)
  (h6 : cost = 40)
  (h7 : profit = 25)
  (h8 : 3 * D + 2 * price_per_clock + 5 * price_per_glass = cost + profit) :
  D = 5 :=
by
  sorry

end stella_doll_price_l1083_108344


namespace butterfly_probability_l1083_108315

-- Define the vertices of the cube
inductive Vertex
| A | B | C | D | E | F | G | H

open Vertex

-- Define the edges of the cube
def edges : Vertex → List Vertex
| A => [B, D, E]
| B => [A, C, F]
| C => [B, D, G]
| D => [A, C, H]
| E => [A, F, H]
| F => [B, E, G]
| G => [C, F, H]
| H => [D, E, G]

-- Define a function to simulate the butterfly's movement
noncomputable def move : Vertex → ℕ → List (Vertex × ℕ)
| v, 0 => [(v, 0)]
| v, n + 1 =>
  let nextMoves := edges v
  nextMoves.bind (λ v' => move v' n)

-- Define the probability calculation part
noncomputable def probability_of_visiting_all_vertices (n_moves : ℕ) : ℚ :=
  let total_paths := (3 ^ n_moves : ℕ)
  let valid_paths := 27 -- Based on given final solution step
  valid_paths / total_paths

-- Statement of the problem in Lean 4
theorem butterfly_probability :
  probability_of_visiting_all_vertices 11 = 27 / 177147 :=
by
  sorry

end butterfly_probability_l1083_108315


namespace sum_of_areas_of_circles_l1083_108384

theorem sum_of_areas_of_circles (r s t : ℝ) (h1 : r + s = 6) (h2 : r + t = 8) (h3 : s + t = 10) :
  ∃ (π : ℝ), π * (r^2 + s^2 + t^2) = 56 * π :=
by {
  sorry
}

end sum_of_areas_of_circles_l1083_108384


namespace primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l1083_108309

-- Define Part (a)
theorem primitive_root_coprime_distinct_residues (m k : ℕ) (h: Nat.gcd m k = 1) :
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∀ i j s t, (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k) :=
sorry

-- Define Part (b)
theorem noncoprime_non_distinct_residues (m k : ℕ) (h: Nat.gcd m k > 1) :
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∃ i j x t, (i ≠ x ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a x * b t) % (m * k) :=
sorry

end primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l1083_108309


namespace hyperbola_condition_l1083_108312

theorem hyperbola_condition (m : ℝ) : ((m - 2) * (m + 3) < 0) ↔ (-3 < m ∧ m < 0) := by
  sorry

end hyperbola_condition_l1083_108312


namespace fourth_term_is_fifteen_l1083_108326

-- Define the problem parameters
variables (a d : ℕ)

-- Define the conditions
def sum_first_third_term : Prop := (a + (a + 2 * d) = 10)
def fourth_term_def : ℕ := a + 3 * d

-- Declare the theorem to be proved
theorem fourth_term_is_fifteen (h1 : sum_first_third_term a d) : fourth_term_def a d = 15 :=
sorry

end fourth_term_is_fifteen_l1083_108326


namespace koschei_coin_count_l1083_108325

theorem koschei_coin_count (a : ℕ) :
  (a % 10 = 7) ∧
  (a % 12 = 9) ∧
  (300 ≤ a ∧ a ≤ 400) →
  a = 357 :=
sorry

end koschei_coin_count_l1083_108325


namespace plane_centroid_l1083_108345

theorem plane_centroid (a b : ℝ) (h : 1 / a ^ 2 + 1 / b ^ 2 + 1 / 25 = 1 / 4) :
  let p := a / 3
  let q := b / 3
  let r := 5 / 3
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 369 / 400 :=
by
  sorry

end plane_centroid_l1083_108345


namespace final_selling_price_l1083_108387

variable (a : ℝ)

theorem final_selling_price (h : a > 0) : 0.9 * (1.25 * a) = 1.125 * a := 
by
  sorry

end final_selling_price_l1083_108387


namespace probability_diff_color_ball_l1083_108368

variable (boxA : List String) (boxB : List String)
def P_A (boxA := ["white", "white", "red", "red", "black"]) (boxB := ["white", "white", "white", "white", "red", "red", "red", "black", "black"]) : ℚ := sorry

theorem probability_diff_color_ball :
  P_A boxA boxB = 29 / 50 :=
sorry

end probability_diff_color_ball_l1083_108368


namespace tan_sum_angles_l1083_108381

theorem tan_sum_angles : (Real.tan (17 * Real.pi / 180) + Real.tan (28 * Real.pi / 180)) / (1 - Real.tan (17 * Real.pi / 180) * Real.tan (28 * Real.pi / 180)) = 1 := 
by sorry

end tan_sum_angles_l1083_108381


namespace tent_ratio_l1083_108322

-- Define variables for tents in different parts of the camp
variables (N E C S T : ℕ)

-- Given conditions
def northernmost (N : ℕ) := N = 100
def center (C N : ℕ) := C = 4 * N
def southern (S : ℕ) := S = 200
def total (T N C E S : ℕ) := T = N + C + E + S

-- Main theorem statement for the proof
theorem tent_ratio (N E C S T : ℕ) 
  (hn : northernmost N)
  (hc : center C N) 
  (hs : southern S)
  (ht : total T N C E S) :
  E / N = 2 :=
by sorry

end tent_ratio_l1083_108322


namespace winning_percentage_l1083_108353

theorem winning_percentage (total_votes winner_votes : ℕ) 
  (h1 : winner_votes = 1344) 
  (h2 : winner_votes - 288 = total_votes - winner_votes) : 
  (winner_votes * 100 / total_votes = 56) :=
sorry

end winning_percentage_l1083_108353


namespace peanuts_remaining_l1083_108395

def initial_peanuts := 220
def brock_fraction := 1 / 4
def bonita_fraction := 2 / 5
def carlos_peanuts := 17

noncomputable def peanuts_left := initial_peanuts - (initial_peanuts * brock_fraction + ((initial_peanuts - initial_peanuts * brock_fraction) * bonita_fraction)) - carlos_peanuts

theorem peanuts_remaining : peanuts_left = 82 :=
by
  sorry

end peanuts_remaining_l1083_108395


namespace water_required_for_reaction_l1083_108392

noncomputable def sodium_hydride_reaction (NaH H₂O NaOH H₂ : Type) : Nat :=
  1

theorem water_required_for_reaction :
  let NaH := 2
  let required_H₂O := 2 -- Derived from balanced chemical equation and given condition
  sodium_hydride_reaction Nat Nat Nat Nat = required_H₂O :=
by
  sorry

end water_required_for_reaction_l1083_108392


namespace ages_of_Linda_and_Jane_l1083_108331

theorem ages_of_Linda_and_Jane : 
  ∃ (J L : ℕ), 
    (L = 2 * J + 3) ∧ 
    (∃ (p : ℕ), Nat.Prime p ∧ p = L - J) ∧ 
    (L + J = 4 * J - 5) ∧ 
    (L = 19 ∧ J = 8) :=
by
  sorry

end ages_of_Linda_and_Jane_l1083_108331


namespace multiplication_result_l1083_108397

theorem multiplication_result :
  (500 ^ 50) * (2 ^ 100) = 10 ^ 75 :=
by
  sorry

end multiplication_result_l1083_108397


namespace find_positive_n_l1083_108350

def consecutive_product (k : ℕ) : ℕ := k * (k + 1) * (k + 2)

theorem find_positive_n (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  n^6 + 5*n^3 + 4*n + 116 = consecutive_product k ↔ n = 3 := 
by 
  sorry

end find_positive_n_l1083_108350


namespace D_300_l1083_108310

def D (n : ℕ) : ℕ :=
sorry

theorem D_300 : D 300 = 73 := 
by 
sorry

end D_300_l1083_108310


namespace simplify_expr_l1083_108386

variable (x y : ℝ)

def expr (x y : ℝ) := (x + y) * (x - y) - y * (2 * x - y)

theorem simplify_expr :
  let x := Real.sqrt 2
  let y := Real.sqrt 3
  expr x y = 2 - 2 * Real.sqrt 6 := by
  sorry

end simplify_expr_l1083_108386


namespace not_prime_5n_plus_3_l1083_108327

theorem not_prime_5n_plus_3 (n a b : ℕ) (hn_pos : n > 0) (ha_pos : a > 0) (hb_pos : b > 0)
  (ha : 2 * n + 1 = a^2) (hb : 3 * n + 1 = b^2) : ¬Prime (5 * n + 3) :=
by
  sorry

end not_prime_5n_plus_3_l1083_108327


namespace system1_solution_l1083_108399

theorem system1_solution (x y : ℝ) 
  (h1 : x + y = 10^20) 
  (h2 : x - y = 10^19) :
  x = 55 * 10^18 ∧ y = 45 * 10^18 := 
by
  sorry

end system1_solution_l1083_108399


namespace max_n_is_11_l1083_108360

noncomputable def max_n (a1 d : ℝ) : ℕ :=
if h : d < 0 then
  11
else
  sorry

theorem max_n_is_11 (d : ℝ) (a1 : ℝ) (c : ℝ) :
  (d / 2) * (22 ^ 2) + (a1 - (d / 2)) * 22 + c ≥ 0 →
  22 = (a1 - (d / 2)) / (- (d / 2)) →
  max_n a1 d = 11 :=
by
  intros h1 h2
  rw [max_n]
  split_ifs
  · exact rfl
  · exact sorry

end max_n_is_11_l1083_108360


namespace remainder_when_x150_divided_by_x1_4_l1083_108394

noncomputable def remainder_div_x150_by_x1_4 (x : ℝ) : ℝ :=
  x^150 % (x-1)^4

theorem remainder_when_x150_divided_by_x1_4 (x : ℝ) :
  remainder_div_x150_by_x1_4 x = -551300 * x^3 + 1665075 * x^2 - 1667400 * x + 562626 :=
by
  sorry

end remainder_when_x150_divided_by_x1_4_l1083_108394


namespace ratio_length_breadth_l1083_108347

noncomputable def b : ℝ := 18
noncomputable def l : ℝ := 972 / b

theorem ratio_length_breadth
  (A : ℝ)
  (h1 : b = 18)
  (h2 : l * b = 972) :
  (l / b) = 3 :=
by
  sorry

end ratio_length_breadth_l1083_108347


namespace C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l1083_108320

noncomputable def parametric_C1 (α : ℝ) : ℝ × ℝ := (2 + Real.cos α, 4 + Real.sin α)

noncomputable def polar_C2 (ρ θ m : ℝ) : ℝ := ρ * (Real.cos θ - m * Real.sin θ) + 1

theorem C1_Cartesian_equation :
  ∀ (x y : ℝ), (∃ α : ℝ, parametric_C1 α = (x, y)) ↔ (x - 2)^2 + (y - 4)^2 = 1 := sorry

theorem C2_Cartesian_equation :
  ∀ (x y m : ℝ), (∃ ρ θ : ℝ, polar_C2 ρ θ m = 0 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)
  ↔ x - m * y + 1 = 0 := sorry

def closest_point_on_C1_to_x_axis : ℝ × ℝ := (2, 3)

theorem m_value_when_C2_passes_through_P :
  ∃ (m : ℝ), x - m * y + 1 = 0 ∧ x = 2 ∧ y = 3 ∧ m = 1 := sorry

end C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l1083_108320


namespace janelle_gave_green_marbles_l1083_108379

def initial_green_marbles : ℕ := 26
def bags_blue_marbles : ℕ := 6
def marbles_per_bag : ℕ := 10
def total_blue_marbles : ℕ := bags_blue_marbles * marbles_per_bag
def total_marbles_after_gift : ℕ := 72
def blue_marbles_in_gift : ℕ := 8
def final_blue_marbles : ℕ := total_blue_marbles - blue_marbles_in_gift
def final_green_marbles : ℕ := total_marbles_after_gift - final_blue_marbles
def initial_green_marbles_after_gift : ℕ := final_green_marbles
def green_marbles_given : ℕ := initial_green_marbles - initial_green_marbles_after_gift

theorem janelle_gave_green_marbles :
  green_marbles_given = 6 :=
by {
  sorry
}

end janelle_gave_green_marbles_l1083_108379


namespace six_smallest_distinct_integers_l1083_108378

theorem six_smallest_distinct_integers:
  ∃ (a b c d e f : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    a * b * c * d * e = 999999 ∧ a = 3 ∧
    f = 37 ∨
    a * b * c * d * f = 999999 ∧ a = 3 ∧ e = 13 ∨ 
    a * b * d * f * e = 999999 ∧ c = 9 ∧ 
    a * c * d * e * f = 999999 ∧ b = 7 ∧ 
    b * c * d * e * f = 999999 ∧ a = 3 := 
sorry

end six_smallest_distinct_integers_l1083_108378
