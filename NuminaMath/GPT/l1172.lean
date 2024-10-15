import Mathlib

namespace NUMINAMATH_GPT_bananas_per_friend_l1172_117295

theorem bananas_per_friend (total_bananas : ℤ) (total_friends : ℤ) (H1 : total_bananas = 21) (H2 : total_friends = 3) : 
  total_bananas / total_friends = 7 :=
by
  sorry

end NUMINAMATH_GPT_bananas_per_friend_l1172_117295


namespace NUMINAMATH_GPT_cheryl_found_more_eggs_l1172_117281

theorem cheryl_found_more_eggs :
  let kevin_eggs := 5
  let bonnie_eggs := 13
  let george_eggs := 9
  let cheryl_eggs := 56
  cheryl_eggs - (kevin_eggs + bonnie_eggs + george_eggs) = 29 := by
  sorry

end NUMINAMATH_GPT_cheryl_found_more_eggs_l1172_117281


namespace NUMINAMATH_GPT_train_crossing_time_l1172_117292

-- Condition definitions
def length_train : ℝ := 100
def length_bridge : ℝ := 150
def speed_kmph : ℝ := 54
def speed_mps : ℝ := 15

-- Given the conditions, prove the time to cross the bridge is 16.67 seconds
theorem train_crossing_time :
  (100 + 150) / (54 * 1000 / 3600) = 16.67 := by sorry

end NUMINAMATH_GPT_train_crossing_time_l1172_117292


namespace NUMINAMATH_GPT_rectangle_area_l1172_117241

theorem rectangle_area
  (b : ℝ)
  (l : ℝ)
  (P : ℝ)
  (h1 : l = 3 * b)
  (h2 : P = 2 * (l + b))
  (h3 : P = 112) :
  l * b = 588 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1172_117241


namespace NUMINAMATH_GPT_neighbors_receive_28_mangoes_l1172_117264

/-- 
  Mr. Wong harvested 560 mangoes. He sold half, gave 50 to his family,
  and divided the remaining mangoes equally among 8 neighbors.
  Each neighbor should receive 28 mangoes.
-/
theorem neighbors_receive_28_mangoes : 
  ∀ (initial : ℕ) (sold : ℕ) (given : ℕ) (neighbors : ℕ), 
  initial = 560 → 
  sold = initial / 2 → 
  given = 50 → 
  neighbors = 8 → 
  (initial - sold - given) / neighbors = 28 := 
by 
  intros initial sold given neighbors
  sorry

end NUMINAMATH_GPT_neighbors_receive_28_mangoes_l1172_117264


namespace NUMINAMATH_GPT_basic_computer_price_l1172_117272

theorem basic_computer_price (C P : ℝ) 
  (h1 : C + P = 2500)
  (h2 : P = 1 / 8 * ((C + 500) + P)) :
  C = 2125 :=
by
  sorry

end NUMINAMATH_GPT_basic_computer_price_l1172_117272


namespace NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l1172_117216

theorem quadratic_has_distinct_real_roots (a : ℝ) (h : a = -2) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 3 = 0 ∧ a * x2^2 + 2 * x2 + 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_distinct_real_roots_l1172_117216


namespace NUMINAMATH_GPT_susan_homework_start_time_l1172_117283

def start_time_homework (finish_time : ℕ) (homework_duration : ℕ) (interval_duration : ℕ) : ℕ :=
  finish_time - homework_duration - interval_duration

theorem susan_homework_start_time :
  let finish_time : ℕ := 16 * 60 -- 4:00 p.m. in minutes
  let homework_duration : ℕ := 96 -- Homework duration in minutes
  let interval_duration : ℕ := 25 -- Interval between homework finish and practice in minutes
  start_time_homework finish_time homework_duration interval_duration = 13 * 60 + 59 := -- 13:59 in minutes
by
  sorry

end NUMINAMATH_GPT_susan_homework_start_time_l1172_117283


namespace NUMINAMATH_GPT_function_value_l1172_117236

theorem function_value (f : ℝ → ℝ) (h : ∀ x, x + 17 = 60 * f x) : f 3 = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_function_value_l1172_117236


namespace NUMINAMATH_GPT_one_number_greater_than_one_l1172_117212

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos: 0 < a ∧ 0 < b ∧ 0 < c)
  (h_prod: a * b * c = 1)
  (h_sum: a + b + c > 1 / a + 1 / b + 1 / c) :
  (1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (1 < b ∧ a ≤ 1 ∧ c ≤ 1) ∨ (1 < c ∧ a ≤ 1 ∧ b ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_one_number_greater_than_one_l1172_117212


namespace NUMINAMATH_GPT_remainder_2345678901_div_101_l1172_117284

theorem remainder_2345678901_div_101 : 2345678901 % 101 = 12 :=
sorry

end NUMINAMATH_GPT_remainder_2345678901_div_101_l1172_117284


namespace NUMINAMATH_GPT_max_value_f_l1172_117230

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem max_value_f : ∃ x ∈ (Set.Icc 0 (Real.pi / 2)), f x = Real.pi / 12 + Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_max_value_f_l1172_117230


namespace NUMINAMATH_GPT_solve_for_real_a_l1172_117202

theorem solve_for_real_a (a : ℝ) (i : ℂ) (h : i^2 = -1) (h1 : (a - i)^2 = 2 * i) : a = -1 :=
by sorry

end NUMINAMATH_GPT_solve_for_real_a_l1172_117202


namespace NUMINAMATH_GPT_area_of_triangle_PQR_l1172_117263

def Point := (ℝ × ℝ)
def area_of_triangle (P Q R : Point) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

def P : Point := (1, 1)
def Q : Point := (4, 5)
def R : Point := (7, 2)

theorem area_of_triangle_PQR :
  area_of_triangle P Q R = 10.5 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_PQR_l1172_117263


namespace NUMINAMATH_GPT_num_factors_of_M_l1172_117240

theorem num_factors_of_M (M : ℕ) 
  (hM : M = (2^5) * (3^4) * (5^3) * (11^2)) : ∃ n : ℕ, n = 360 ∧ M = (2^5) * (3^4) * (5^3) * (11^2) := 
by
  sorry

end NUMINAMATH_GPT_num_factors_of_M_l1172_117240


namespace NUMINAMATH_GPT_angle_CBD_is_10_degrees_l1172_117218

theorem angle_CBD_is_10_degrees (angle_ABC angle_ABD : ℝ) (h1 : angle_ABC = 40) (h2 : angle_ABD = 30) :
  angle_ABC - angle_ABD = 10 :=
by
  sorry

end NUMINAMATH_GPT_angle_CBD_is_10_degrees_l1172_117218


namespace NUMINAMATH_GPT_smallest_natural_number_l1172_117215

theorem smallest_natural_number (n : ℕ) (h : 2006 ^ 1003 < n ^ 2006) : n ≥ 45 := 
by {
    sorry
}

end NUMINAMATH_GPT_smallest_natural_number_l1172_117215


namespace NUMINAMATH_GPT_weight_of_b_l1172_117269

theorem weight_of_b (A B C : ℝ) 
  (h1 : A + B + C = 129)
  (h2 : A + B = 96)
  (h3 : B + C = 84) : 
  B = 51 :=
sorry

end NUMINAMATH_GPT_weight_of_b_l1172_117269


namespace NUMINAMATH_GPT_area_of_rectangle_l1172_117206

theorem area_of_rectangle (AB AC : ℝ) (angle_ABC : ℝ) (h_AB : AB = 15) (h_AC : AC = 17) (h_angle_ABC : angle_ABC = 90) :
  ∃ BC : ℝ, (BC = 8) ∧ (AB * BC = 120) :=
by
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l1172_117206


namespace NUMINAMATH_GPT_circle_area_with_radius_three_is_9pi_l1172_117261

theorem circle_area_with_radius_three_is_9pi (r : ℝ) (h : r = 3) : Real.pi * r^2 = 9 * Real.pi := by
  sorry

end NUMINAMATH_GPT_circle_area_with_radius_three_is_9pi_l1172_117261


namespace NUMINAMATH_GPT_determine_g_two_l1172_117245

variables (a b c d p q r s : ℝ) -- Define variables a, b, c, d, p, q, r, s as real numbers
variables (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) -- The conditions a < b < c < d

noncomputable def f (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d
noncomputable def g (x : ℝ) : ℝ := (x - 1/p) * (x - 1/q) * (x - 1/r) * (x - 1/s)

noncomputable def g_two := g 2
noncomputable def f_two := f 2

theorem determine_g_two :
  g_two a b c d = (16 + 8*a + 4*b + 2*c + d) / (p*q*r*s) :=
sorry

end NUMINAMATH_GPT_determine_g_two_l1172_117245


namespace NUMINAMATH_GPT_relationship_among_abc_l1172_117232

noncomputable def a : ℝ := 2 ^ (3 / 2)
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2
noncomputable def c : ℝ := 0.8 ^ 2

theorem relationship_among_abc : b < c ∧ c < a := 
by
  -- these are conditions directly derived from the problem
  let h1 : a = 2 ^ (3 / 2) := rfl
  let h2 : b = Real.log 0.3 / Real.log 2 := rfl
  let h3 : c = 0.8 ^ 2 := rfl
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l1172_117232


namespace NUMINAMATH_GPT_max_value_10x_plus_3y_plus_12z_l1172_117270

theorem max_value_10x_plus_3y_plus_12z (x y z : ℝ) 
  (h1 : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) 
  (h2 : z = 2 * y) : 
  10 * x + 3 * y + 12 * z ≤ Real.sqrt 253 :=
sorry

end NUMINAMATH_GPT_max_value_10x_plus_3y_plus_12z_l1172_117270


namespace NUMINAMATH_GPT_bruce_initial_money_l1172_117208

-- Definitions of the conditions
def cost_crayons : ℕ := 5 * 5
def cost_books : ℕ := 10 * 5
def cost_calculators : ℕ := 3 * 5
def total_spent : ℕ := cost_crayons + cost_books + cost_calculators
def cost_bags : ℕ := 11 * 10
def initial_money : ℕ := total_spent + cost_bags

-- Theorem statement
theorem bruce_initial_money :
  initial_money = 200 := by
  sorry

end NUMINAMATH_GPT_bruce_initial_money_l1172_117208


namespace NUMINAMATH_GPT_fruit_basket_count_l1172_117252

theorem fruit_basket_count :
  let pears := 8
  let bananas := 12
  let total_baskets := (pears + 1) * (bananas + 1) - 1
  total_baskets = 116 :=
by
  sorry

end NUMINAMATH_GPT_fruit_basket_count_l1172_117252


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1172_117211

variable (S : ℕ → ℝ)

def arithmetic_seq_property (S : ℕ → ℝ) : Prop :=
  S 4 = 4 ∧ S 8 = 12

theorem sum_of_arithmetic_sequence (h : arithmetic_seq_property S) : S 12 = 24 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l1172_117211


namespace NUMINAMATH_GPT_stockings_total_cost_l1172_117266

-- Define a function to compute the total cost given the conditions
def total_cost (n_grandchildren n_children : Nat) 
               (stocking_price discount monogram_cost : Nat) : Nat :=
  let total_stockings := n_grandchildren + n_children
  let discounted_price := stocking_price - (stocking_price * discount / 100)
  let total_stockings_cost := discounted_price * total_stockings
  let total_monogram_cost := monogram_cost * total_stockings
  total_stockings_cost + total_monogram_cost

-- Prove that the total cost calculation is correct given the conditions
theorem stockings_total_cost :
  total_cost 5 4 20 10 5 = 207 :=
by
  -- A placeholder for the proof
  sorry

end NUMINAMATH_GPT_stockings_total_cost_l1172_117266


namespace NUMINAMATH_GPT_ariel_years_fencing_l1172_117200

-- Definitions based on given conditions
def fencing_start_year := 2006
def birth_year := 1992
def current_age := 30

-- To find: The number of years Ariel has been fencing
def current_year : ℕ := birth_year + current_age
def years_fencing : ℕ := current_year - fencing_start_year

-- Proof statement
theorem ariel_years_fencing : years_fencing = 16 := by
  sorry

end NUMINAMATH_GPT_ariel_years_fencing_l1172_117200


namespace NUMINAMATH_GPT_min_speed_to_arrive_before_cara_l1172_117210

theorem min_speed_to_arrive_before_cara (d : ℕ) (sc : ℕ) (tc : ℕ) (sd : ℕ) (td : ℕ) (hd : ℕ) :
  d = 180 ∧ sc = 30 ∧ tc = d / sc ∧ hd = 1 ∧ td = tc - hd ∧ sd = d / td ∧ (36 < sd) :=
sorry

end NUMINAMATH_GPT_min_speed_to_arrive_before_cara_l1172_117210


namespace NUMINAMATH_GPT_trail_length_proof_l1172_117260

theorem trail_length_proof (x1 x2 x3 x4 x5 : ℝ)
  (h1 : x1 + x2 = 28)
  (h2 : x2 + x3 = 30)
  (h3 : x3 + x4 + x5 = 42)
  (h4 : x1 + x4 = 30) :
  x1 + x2 + x3 + x4 + x5 = 70 := by
  sorry

end NUMINAMATH_GPT_trail_length_proof_l1172_117260


namespace NUMINAMATH_GPT_number_div_0_04_eq_200_9_l1172_117267

theorem number_div_0_04_eq_200_9 (n : ℝ) (h : n / 0.04 = 200.9) : n = 8.036 :=
sorry

end NUMINAMATH_GPT_number_div_0_04_eq_200_9_l1172_117267


namespace NUMINAMATH_GPT_solve_quadratic_l1172_117213

theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
sorry

end NUMINAMATH_GPT_solve_quadratic_l1172_117213


namespace NUMINAMATH_GPT_range_of_x_l1172_117297

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := |x - 4| + |x - a|

theorem range_of_x (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, f x a ≥ |a - 4|) (h3 : |a - 4| = 3) :
  { x : ℝ | f x a ≤ 5 } = { x : ℝ | 3 ≤ x ∧ x ≤ 8 } := 
sorry

end NUMINAMATH_GPT_range_of_x_l1172_117297


namespace NUMINAMATH_GPT_max_hardcover_books_l1172_117225

-- Define the conditions as provided in the problem
def total_books : ℕ := 36
def is_composite (n : ℕ) : Prop := 
  ∃ a b : ℕ, 2 ≤ a ∧ 2 ≤ b ∧ a * b = n

-- The logical statement we need to prove
theorem max_hardcover_books :
  ∃ h : ℕ, (∃ c : ℕ, is_composite c ∧ 2 * h + c = total_books) ∧ 
  ∀ h' c', is_composite c' ∧ 2 * h' + c' = total_books → h' ≤ h :=
sorry

end NUMINAMATH_GPT_max_hardcover_books_l1172_117225


namespace NUMINAMATH_GPT_minimum_value_f_l1172_117256

noncomputable def f (x y : ℝ) : ℝ :=
  (x^2 / (y - 2)^2) + (y^2 / (x - 2)^2)

theorem minimum_value_f (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  ∃ (a : ℝ), (∀ (b : ℝ), f x y >= b) ∧ a = 10 := sorry

end NUMINAMATH_GPT_minimum_value_f_l1172_117256


namespace NUMINAMATH_GPT_initial_typists_count_l1172_117221

theorem initial_typists_count
  (letters_per_20_min : Nat)
  (letters_total_1_hour : Nat)
  (letters_typists_count : Nat)
  (n_typists_init : Nat)
  (h1 : letters_per_20_min = 46)
  (h2 : letters_typists_count = 30)
  (h3 : letters_total_1_hour = 207) :
  n_typists_init = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_typists_count_l1172_117221


namespace NUMINAMATH_GPT_john_paintball_times_l1172_117299

theorem john_paintball_times (x : ℕ) (cost_per_box : ℕ) (boxes_per_play : ℕ) (monthly_spending : ℕ) :
  (cost_per_box = 25) → (boxes_per_play = 3) → (monthly_spending = 225) → (boxes_per_play * cost_per_box * x = monthly_spending) → x = 3 :=
by
  intros h1 h2 h3 h4
  -- proof would go here
  sorry

end NUMINAMATH_GPT_john_paintball_times_l1172_117299


namespace NUMINAMATH_GPT_solve_for_x_l1172_117209

theorem solve_for_x (x : ℝ) (h : 9 / x^2 = x / 81) : x = 9 := 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1172_117209


namespace NUMINAMATH_GPT_range_of_m_l1172_117255

noncomputable def setA := {x : ℝ | -2 ≤ x ∧ x ≤ 7}
noncomputable def setB (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem range_of_m (m : ℝ) : (setA ∪ setB m = setA) → m ≤ 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_m_l1172_117255


namespace NUMINAMATH_GPT_maximize_wind_power_l1172_117258

variable {C S ρ v_0 : ℝ}

theorem maximize_wind_power : 
  ∃ v : ℝ, (∀ (v' : ℝ),
           let F := (C * S * ρ * (v_0 - v)^2) / 2;
           let N := F * v;
           let N' := (C * S * ρ / 2) * (v_0^2 - 4 * v_0 * v + 3 * v^2);
           N' = 0
         → N ≤ (C * S * ρ / 2) * (v_0^2 * (v_0/3) - 2 * v_0 * (v_0/3)^2 + (v_0/3)^3)) ∧ v = v_0 / 3 :=
by sorry

end NUMINAMATH_GPT_maximize_wind_power_l1172_117258


namespace NUMINAMATH_GPT_C_investment_l1172_117286

theorem C_investment (A B total_profit A_share : ℝ) (x : ℝ) :
  A = 6300 → B = 4200 → total_profit = 12600 → A_share = 3780 →
  (A / (A + B + x) = A_share / total_profit) → x = 10500 :=
by
  intros hA hB h_total_profit h_A_share h_ratio
  sorry

end NUMINAMATH_GPT_C_investment_l1172_117286


namespace NUMINAMATH_GPT_weaving_sequence_l1172_117235

-- Define the arithmetic sequence conditions
def day1_weaving := 5
def total_cloth := 390
def days := 30

-- Mathematical statement to be proved
theorem weaving_sequence : 
    ∃ d : ℚ, 30 * day1_weaving + (days * (days - 1) / 2) * d = total_cloth ∧ d = 16 / 29 :=
by 
  sorry

end NUMINAMATH_GPT_weaving_sequence_l1172_117235


namespace NUMINAMATH_GPT_sum_ages_l1172_117271

theorem sum_ages (A_years B_years C_years : ℕ) (h1 : B_years = 30)
  (h2 : 10 * (B_years - 10) = (A_years - 10) * 2)
  (h3 : 10 * (B_years - 10) = (C_years - 10) * 3) :
  A_years + B_years + C_years = 90 :=
sorry

end NUMINAMATH_GPT_sum_ages_l1172_117271


namespace NUMINAMATH_GPT_proportion_solution_l1172_117237

-- Define the given proportion condition as a hypothesis
variable (x : ℝ)

-- The definition is derived directly from the given problem
def proportion_condition : Prop := x / 5 = 1.2 / 8

-- State the theorem using the given proportion condition to prove x = 0.75
theorem proportion_solution (h : proportion_condition x) : x = 0.75 :=
  by
    sorry

end NUMINAMATH_GPT_proportion_solution_l1172_117237


namespace NUMINAMATH_GPT_bus_seating_capacity_l1172_117250

-- Conditions
def left_side_seats : ℕ := 15
def right_side_seats : ℕ := left_side_seats - 3
def seat_capacity : ℕ := 3
def back_seat_capacity : ℕ := 9
def total_seats : ℕ := left_side_seats + right_side_seats

-- Proof problem statement
theorem bus_seating_capacity :
  (total_seats * seat_capacity) + back_seat_capacity = 90 := by
  sorry

end NUMINAMATH_GPT_bus_seating_capacity_l1172_117250


namespace NUMINAMATH_GPT_unique_solution_eq_condition_l1172_117223

theorem unique_solution_eq_condition (p q : ℝ) :
  (∃! x : ℝ, (2 * x - 2 * p + q) / (2 * x - 2 * p - q) = (2 * q + p + x) / (2 * q - p - x)) ↔ (p = 3 * q / 4 ∧ q ≠ 0) :=
  sorry

end NUMINAMATH_GPT_unique_solution_eq_condition_l1172_117223


namespace NUMINAMATH_GPT_initial_girls_count_l1172_117203

variable (p : ℕ) -- total number of people initially in the group
variable (initial_girls : ℕ) -- number of girls initially

-- Condition 1: Initially, 50% of the group are girls
def initially_fifty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop := initial_girls = p / 2

-- Condition 2: Three girls leave and three boys arrive
def after_girls_leave_and_boys_arrive (initial_girls : ℕ) : ℕ := initial_girls - 3

-- Condition 3: After the change, 40% of the group are girls
def after_the_change_forty_percent_girls (p : ℕ) (initial_girls : ℕ) : Prop :=
  (after_girls_leave_and_boys_arrive initial_girls) = 2 * (p / 5)

theorem initial_girls_count (p : ℕ) (initial_girls : ℕ) :
  initially_fifty_percent_girls p initial_girls →
  after_the_change_forty_percent_girls p initial_girls →
  initial_girls = 15 := by
  sorry

end NUMINAMATH_GPT_initial_girls_count_l1172_117203


namespace NUMINAMATH_GPT_thomas_annual_insurance_cost_l1172_117273

theorem thomas_annual_insurance_cost (total_cost : ℕ) (number_of_years : ℕ) 
  (h1 : total_cost = 40000) (h2 : number_of_years = 10) : 
  total_cost / number_of_years = 4000 := 
by 
  sorry

end NUMINAMATH_GPT_thomas_annual_insurance_cost_l1172_117273


namespace NUMINAMATH_GPT_time_to_school_gate_l1172_117248

theorem time_to_school_gate (total_time gate_to_building building_to_room time_to_gate : ℕ) 
                            (h1 : total_time = 30)
                            (h2 : gate_to_building = 6)
                            (h3 : building_to_room = 9)
                            (h4 : total_time = time_to_gate + gate_to_building + building_to_room) :
  time_to_gate = 15 :=
  sorry

end NUMINAMATH_GPT_time_to_school_gate_l1172_117248


namespace NUMINAMATH_GPT_a0_a1_consecutive_l1172_117279

variable (a : ℕ → ℤ)
variable (cond : ∀ i ≥ 2, a i = 2 * a (i - 1) - a (i - 2) ∨ a i = 2 * a (i - 2) - a (i - 1))
variable (consec : |a 2024 - a 2023| = 1)

theorem a0_a1_consecutive :
  |a 1 - a 0| = 1 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_a0_a1_consecutive_l1172_117279


namespace NUMINAMATH_GPT_employees_count_l1172_117289

theorem employees_count (n : ℕ) (avg_salary : ℝ) (manager_salary : ℝ)
  (new_avg_salary : ℝ) (total_employees_with_manager : ℝ) : 
  avg_salary = 1500 → 
  manager_salary = 3600 → 
  new_avg_salary = avg_salary + 100 → 
  total_employees_with_manager = (n + 1) * 1600 → 
  (n * avg_salary + manager_salary) / total_employees_with_manager = new_avg_salary →
  n = 20 := by
  intros
  sorry

end NUMINAMATH_GPT_employees_count_l1172_117289


namespace NUMINAMATH_GPT_negation_of_exists_lt_zero_l1172_117226

theorem negation_of_exists_lt_zero (m : ℝ) :
  ¬ (∃ x : ℝ, x < 0 ∧ x^2 + 2 * x - m > 0) ↔ ∀ x : ℝ, x < 0 → x^2 + 2 * x - m ≤ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_exists_lt_zero_l1172_117226


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1172_117249

theorem arithmetic_sequence_ninth_term 
  (a d : ℤ)
  (h1 : a + 2 * d = 23)
  (h2 : a + 5 * d = 29)
  : a + 8 * d = 35 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1172_117249


namespace NUMINAMATH_GPT_customers_tried_sample_l1172_117214

theorem customers_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left_over : ℕ)
  (samples_per_customer : ℕ := 1)
  (h_samples_per_box : samples_per_box = 20)
  (h_boxes_opened : boxes_opened = 12)
  (h_samples_left_over : samples_left_over = 5) :
  (samples_per_box * boxes_opened - samples_left_over) / samples_per_customer = 235 :=
by
  sorry

end NUMINAMATH_GPT_customers_tried_sample_l1172_117214


namespace NUMINAMATH_GPT_david_marks_in_physics_l1172_117265

theorem david_marks_in_physics
  (marks_english : ℤ)
  (marks_math : ℤ)
  (marks_chemistry : ℤ)
  (marks_biology : ℤ)
  (average_marks : ℚ)
  (number_of_subjects : ℤ)
  (h_english : marks_english = 96)
  (h_math : marks_math = 98)
  (h_chemistry : marks_chemistry = 100)
  (h_biology : marks_biology = 98)
  (h_average : average_marks = 98.2)
  (h_subjects : number_of_subjects = 5) : 
  ∃ (marks_physics : ℤ), marks_physics = 99 := 
by {
  sorry
}

end NUMINAMATH_GPT_david_marks_in_physics_l1172_117265


namespace NUMINAMATH_GPT_sum_of_numbers_eq_8140_l1172_117293

def numbers : List ℤ := [1200, 1300, 1400, 1510, 1530, 1200]

theorem sum_of_numbers_eq_8140 : (numbers.sum = 8140) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_eq_8140_l1172_117293


namespace NUMINAMATH_GPT_sin_double_angle_plus_pi_over_six_l1172_117277

variable (θ : ℝ)
variable (h : 7 * Real.sqrt 3 * Real.sin θ = 1 + 7 * Real.cos θ)

theorem sin_double_angle_plus_pi_over_six :
  Real.sin (2 * θ + π / 6) = 97 / 98 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_plus_pi_over_six_l1172_117277


namespace NUMINAMATH_GPT_inequality_B_l1172_117288

variable {x y : ℝ}

theorem inequality_B (hx : 0 < x) (hy : 0 < y) (hxy : x > y) : x + 1 / (2 * y) > y + 1 / x :=
sorry

end NUMINAMATH_GPT_inequality_B_l1172_117288


namespace NUMINAMATH_GPT_shaded_area_correct_l1172_117294

-- Define points as vectors in the 2D plane.
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define points K, L, M, J based on the given coordinates.
def K : Point := {x := 0, y := 0}
def L : Point := {x := 5, y := 0}
def M : Point := {x := 5, y := 6}
def J : Point := {x := 0, y := 6}

-- Define intersection point N based on the equations of lines.
def N : Point := {x := 2.5, y := 3}

-- Define the function to calculate area of a trapezoid.
def trapezoid_area (b1 b2 h : ℝ) : ℝ :=
  0.5 * (b1 + b2) * h

-- Define the function to calculate area of a triangle.
def triangle_area (b h : ℝ) : ℝ :=
  0.5 * b * h

-- Compute total shaded area according to the problem statement.
def shaded_area (K L M J N : Point) : ℝ :=
  trapezoid_area 5 2.5 3 + triangle_area 2.5 1

theorem shaded_area_correct : shaded_area K L M J N = 12.5 := by
  sorry

end NUMINAMATH_GPT_shaded_area_correct_l1172_117294


namespace NUMINAMATH_GPT_solve_inequality_l1172_117238

theorem solve_inequality (x : ℝ) : x + 1 > 3 → x > 2 := 
sorry

end NUMINAMATH_GPT_solve_inequality_l1172_117238


namespace NUMINAMATH_GPT_cos_sq_minus_sin_sq_l1172_117234

noncomputable def alpha : ℝ := sorry

axiom tan_alpha_eq_two : Real.tan alpha = 2

theorem cos_sq_minus_sin_sq : Real.cos alpha ^ 2 - Real.sin alpha ^ 2 = -3/5 := by
  sorry

end NUMINAMATH_GPT_cos_sq_minus_sin_sq_l1172_117234


namespace NUMINAMATH_GPT_find_radius_squared_l1172_117291

theorem find_radius_squared (r : ℝ) (AB_len CD_len BP : ℝ) (angle_APD : ℝ) (h1 : AB_len = 12)
    (h2 : CD_len = 9) (h3 : BP = 10) (h4 : angle_APD = 60) : r^2 = 111 := by
  have AB_len := h1
  have CD_len := h2
  have BP := h3
  have angle_APD := h4
  sorry

end NUMINAMATH_GPT_find_radius_squared_l1172_117291


namespace NUMINAMATH_GPT_nina_math_homework_l1172_117204

theorem nina_math_homework (x : ℕ) :
  let ruby_math := 6
  let ruby_read := 2
  let nina_math := x * ruby_math
  let nina_read := 8 * ruby_read
  let nina_total := nina_math + nina_read
  nina_total = 48 → x = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_nina_math_homework_l1172_117204


namespace NUMINAMATH_GPT_decimal_to_fraction_l1172_117296

theorem decimal_to_fraction (h : 0.36 = 36 / 100): (36 / 100 = 9 / 25) := by
    sorry

end NUMINAMATH_GPT_decimal_to_fraction_l1172_117296


namespace NUMINAMATH_GPT_j_at_4_l1172_117228

noncomputable def h (x : ℚ) : ℚ := 5 / (3 - x)

noncomputable def h_inv (x : ℚ) : ℚ := (3 * x - 5) / x

noncomputable def j (x : ℚ) : ℚ := (1 / h_inv x) + 7

theorem j_at_4 : j 4 = 53 / 7 :=
by
  -- Proof steps would be inserted here.
  sorry

end NUMINAMATH_GPT_j_at_4_l1172_117228


namespace NUMINAMATH_GPT_g_g_3_eq_3606651_l1172_117280

def g (x: ℤ) : ℤ := 3 * x^3 + 3 * x^2 - x + 1

theorem g_g_3_eq_3606651 : g (g 3) = 3606651 := 
by {
  sorry
}

end NUMINAMATH_GPT_g_g_3_eq_3606651_l1172_117280


namespace NUMINAMATH_GPT_expression_value_l1172_117247

theorem expression_value (a b c d m : ℚ) (h1 : a + b = 0) (h2 : a ≠ 0) (h3 : c * d = 1) (h4 : m = -5 ∨ m = 1) :
  |m| - (a / b) + ((a + b) / 2020) - (c * d) = 1 ∨ |m| - (a / b) + ((a + b) / 2020) - (c * d) = 5 :=
by sorry

end NUMINAMATH_GPT_expression_value_l1172_117247


namespace NUMINAMATH_GPT_no_ordered_triples_exist_l1172_117242

theorem no_ordered_triples_exist :
  ¬ ∃ (x y z : ℤ), 
    (x^2 - 3 * x * y + 2 * y^2 - z^2 = 39) ∧
    (-x^2 + 6 * y * z + 2 * z^2 = 40) ∧
    (x^2 + x * y + 8 * z^2 = 96) :=
sorry

end NUMINAMATH_GPT_no_ordered_triples_exist_l1172_117242


namespace NUMINAMATH_GPT_cristine_final_lemons_l1172_117259

def cristine_lemons_initial : ℕ := 12
def cristine_lemons_given_to_neighbor : ℕ := 1 / 4 * cristine_lemons_initial
def cristine_lemons_left_after_giving : ℕ := cristine_lemons_initial - cristine_lemons_given_to_neighbor
def cristine_lemons_exchanged_for_oranges : ℕ := 1 / 3 * cristine_lemons_left_after_giving
def cristine_lemons_left_after_exchange : ℕ := cristine_lemons_left_after_giving - cristine_lemons_exchanged_for_oranges

theorem cristine_final_lemons : cristine_lemons_left_after_exchange = 6 :=
by
  sorry

end NUMINAMATH_GPT_cristine_final_lemons_l1172_117259


namespace NUMINAMATH_GPT_remaining_money_after_shopping_l1172_117262

theorem remaining_money_after_shopping (initial_money : ℝ) (percentage_spent : ℝ) (final_amount : ℝ) :
  initial_money = 1200 → percentage_spent = 0.30 → final_amount = initial_money - (percentage_spent * initial_money) → final_amount = 840 :=
by
  intros h_initial h_percentage h_final
  sorry

end NUMINAMATH_GPT_remaining_money_after_shopping_l1172_117262


namespace NUMINAMATH_GPT_fireworks_number_l1172_117244

variable (x : ℕ)
variable (fireworks_total : ℕ := 484)
variable (happy_new_year_fireworks : ℕ := 12 * 5)
variable (boxes_of_fireworks : ℕ := 50 * 8)
variable (year_fireworks : ℕ := 4 * x)

theorem fireworks_number :
    4 * x + happy_new_year_fireworks + boxes_of_fireworks = fireworks_total →
    x = 6 := 
by
  sorry

end NUMINAMATH_GPT_fireworks_number_l1172_117244


namespace NUMINAMATH_GPT_cubes_squares_problem_l1172_117229

theorem cubes_squares_problem (h1 : 2^3 - 7^2 = 1) (h2 : 3^3 - 6^2 = 9) (h3 : 5^3 - 9^2 = 16) : 4^3 - 8^2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_cubes_squares_problem_l1172_117229


namespace NUMINAMATH_GPT_average_of_numbers_is_correct_l1172_117233

theorem average_of_numbers_is_correct :
  let nums := [12, 13, 14, 510, 520, 530, 1120, 1, 1252140, 2345]
  let sum_nums := 1253205
  let count_nums := 10
  (sum_nums / count_nums.toFloat) = 125320.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_average_of_numbers_is_correct_l1172_117233


namespace NUMINAMATH_GPT_find_percentage_l1172_117207

variable (P : ℝ)

def percentage_condition (P : ℝ) : Prop :=
  P * 30 = (0.25 * 16) + 2

theorem find_percentage : percentage_condition P → P = 0.2 :=
by
  intro h
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_find_percentage_l1172_117207


namespace NUMINAMATH_GPT_four_machines_save_11_hours_l1172_117257

-- Define the conditions
def three_machines_complete_order_in_44_hours := 3 * (1 / (3 * 44)) * 44 = 1

def additional_machine_reduces_time (T : ℝ) := 4 * (1 / (3 * 44)) * T = 1

-- Define the theorem to prove the number of hours saved
theorem four_machines_save_11_hours : 
  (∃ T : ℝ, additional_machine_reduces_time T ∧ three_machines_complete_order_in_44_hours) → 
  44 - 33 = 11 :=
by
  sorry

end NUMINAMATH_GPT_four_machines_save_11_hours_l1172_117257


namespace NUMINAMATH_GPT_factor_polynomial_l1172_117282

theorem factor_polynomial :
  (x : ℝ) → (x^2 - 6*x + 9 - 64*x^4) = (-8*x^2 + x - 3) * (8*x^2 + x - 3) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1172_117282


namespace NUMINAMATH_GPT_find_k_for_perpendicular_lines_l1172_117246

theorem find_k_for_perpendicular_lines (k : ℝ) :
  (∀ x y : ℝ, (k-3) * x + (5 - k) * y + 1 = 0) →
  (∀ x y : ℝ, 2 * (k-3) * x - 2 * y + 3 = 0) →
  (k = 1 ∨ k = 4) :=
by
  sorry

end NUMINAMATH_GPT_find_k_for_perpendicular_lines_l1172_117246


namespace NUMINAMATH_GPT_original_cookies_l1172_117227

noncomputable def initial_cookies (final_cookies : ℝ) (ratio : ℝ) (days : ℕ) : ℝ :=
  final_cookies / ratio^days

theorem original_cookies :
  ∀ (final_cookies : ℝ) (ratio : ℝ) (days : ℕ),
  final_cookies = 28 →
  ratio = 0.7 →
  days = 3 →
  initial_cookies final_cookies ratio days = 82 :=
by
  intros final_cookies ratio days h_final h_ratio h_days
  rw [initial_cookies, h_final, h_ratio, h_days]
  norm_num
  sorry

end NUMINAMATH_GPT_original_cookies_l1172_117227


namespace NUMINAMATH_GPT_bathroom_cleaning_time_ratio_l1172_117285

noncomputable def hourlyRate : ℝ := 5
noncomputable def vacuumingHours : ℝ := 2 -- per session
noncomputable def vacuumingSessions : ℕ := 2
noncomputable def washingDishesTime : ℝ := 0.5
noncomputable def totalEarnings : ℝ := 30

theorem bathroom_cleaning_time_ratio :
  let vacuumingEarnings := vacuumingHours * vacuumingSessions * hourlyRate
  let washingDishesEarnings := washingDishesTime * hourlyRate
  let knownEarnings := vacuumingEarnings + washingDishesEarnings
  let bathroomEarnings := totalEarnings - knownEarnings
  let bathroomCleaningTime := bathroomEarnings / hourlyRate
  bathroomCleaningTime / washingDishesTime = 3 := 
by
  sorry

end NUMINAMATH_GPT_bathroom_cleaning_time_ratio_l1172_117285


namespace NUMINAMATH_GPT_long_furred_brown_dogs_l1172_117290

-- Definitions based on given conditions
def T : ℕ := 45
def L : ℕ := 36
def B : ℕ := 27
def N : ℕ := 8

-- The number of long-furred brown dogs (LB) that needs to be proved
def LB : ℕ := 26

-- Lean 4 statement to prove LB
theorem long_furred_brown_dogs :
  L + B - LB = T - N :=
by 
  unfold T L B N LB -- we unfold definitions to simplify the theorem
  sorry

end NUMINAMATH_GPT_long_furred_brown_dogs_l1172_117290


namespace NUMINAMATH_GPT_donna_has_40_bananas_l1172_117222

-- Define the number of bananas each person has
variables (dawn lydia donna total : ℕ)

-- State the conditions
axiom h1 : dawn + lydia + donna = total
axiom h2 : dawn = lydia + 40
axiom h3 : lydia = 60
axiom h4 : total = 200

-- State the theorem to be proved
theorem donna_has_40_bananas : donna = 40 :=
by {
  sorry -- Placeholder for the proof
}

end NUMINAMATH_GPT_donna_has_40_bananas_l1172_117222


namespace NUMINAMATH_GPT_regular_polygon_sides_l1172_117253

theorem regular_polygon_sides (h : ∀ n : ℕ, n ≥ 3 → (total_internal_angle_sum / n) = 150) :
    n = 12 := by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1172_117253


namespace NUMINAMATH_GPT_find_f_neg_five_half_l1172_117275

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
else if 0 ≤ x + 2 ∧ x + 2 ≤ 1 then 2 * (x + 2) * (1 - (x + 2))
     else -2 * abs x * (1 - abs x)

theorem find_f_neg_five_half (x : ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 2) = f x)
  (h_def : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x * (1 - x)) : 
  f (-5 / 2) = -1 / 2 :=
  by sorry

end NUMINAMATH_GPT_find_f_neg_five_half_l1172_117275


namespace NUMINAMATH_GPT_range_a_l1172_117243

theorem range_a (a : ℝ) :
  (∀ x : ℝ, a ≤ x ∧ x ≤ 3 → -1 ≤ -x^2 + 2 * x + 2 ∧ -x^2 + 2 * x + 2 ≤ 3) →
  -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_a_l1172_117243


namespace NUMINAMATH_GPT_only_solution_2_pow_eq_y_sq_plus_y_plus_1_l1172_117219

theorem only_solution_2_pow_eq_y_sq_plus_y_plus_1 {x y : ℕ} (h1 : 2^x = y^2 + y + 1) : x = 0 ∧ y = 0 := 
by {
  sorry -- proof goes here
}

end NUMINAMATH_GPT_only_solution_2_pow_eq_y_sq_plus_y_plus_1_l1172_117219


namespace NUMINAMATH_GPT_selection_methods_count_l1172_117274

/-- Consider a school with 16 teachers, divided into four departments (First grade, Second grade, Third grade, and Administrative department), with 4 teachers each. 
We need to select 3 leaders such that not all leaders are from the same department and at least one leader is from the Administrative department. 
Prove that the number of different selection methods that satisfy these conditions is 336. -/
theorem selection_methods_count :
  let num_teachers := 16
  let teachers_per_department := 4
  ∃ (choose : ℕ → ℕ → ℕ), 
  choose num_teachers 3 = 336 :=
  sorry

end NUMINAMATH_GPT_selection_methods_count_l1172_117274


namespace NUMINAMATH_GPT_otimes_square_neq_l1172_117220

noncomputable def otimes (a b : ℝ) : ℝ :=
  if a > b then a else b

theorem otimes_square_neq (a b : ℝ) (h : a ≠ b) : (otimes a b) ^ 2 ≠ otimes (a ^ 2) (b ^ 2) := by
  sorry

end NUMINAMATH_GPT_otimes_square_neq_l1172_117220


namespace NUMINAMATH_GPT_line_y_axis_intersect_l1172_117276

theorem line_y_axis_intersect (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3 ∧ y1 = 27) (h2 : x2 = -7 ∧ y2 = -1) :
  ∃ y : ℝ, (∀ x : ℝ, y = (y2 - y1) / (x2 - x1) * (x - x1) + y1) ∧ y = 18.6 :=
by
  sorry

end NUMINAMATH_GPT_line_y_axis_intersect_l1172_117276


namespace NUMINAMATH_GPT_real_ratio_sum_values_l1172_117205

variables (a b c d : ℝ)

theorem real_ratio_sum_values :
  (a / b + b / c + c / d + d / a = 6) ∧
  (a / c + b / d + c / a + d / b = 8) →
  (a / b + c / d = 2 ∨ a / b + c / d = 4) :=
by
  sorry

end NUMINAMATH_GPT_real_ratio_sum_values_l1172_117205


namespace NUMINAMATH_GPT_feet_count_l1172_117231

-- We define the basic quantities
def total_heads : ℕ := 50
def num_hens : ℕ := 30
def num_cows : ℕ := total_heads - num_hens
def hens_feet : ℕ := num_hens * 2
def cows_feet : ℕ := num_cows * 4
def total_feet : ℕ := hens_feet + cows_feet

-- The theorem we want to prove
theorem feet_count : total_feet = 140 :=
  by
  sorry

end NUMINAMATH_GPT_feet_count_l1172_117231


namespace NUMINAMATH_GPT_quadratic_inequality_empty_solution_range_l1172_117217

theorem quadratic_inequality_empty_solution_range (b : ℝ) :
  (∀ x : ℝ, ¬ (x^2 + b * x + 1 ≤ 0)) ↔ -2 < b ∧ b < 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_empty_solution_range_l1172_117217


namespace NUMINAMATH_GPT_wrapping_paper_area_l1172_117298

variable (a b h w : ℝ) (a_gt_b : a > b)

theorem wrapping_paper_area : 
  ∃ total_area, total_area = 4 * (a * b + a * w + b * w + w ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_wrapping_paper_area_l1172_117298


namespace NUMINAMATH_GPT_triangle_inequality_sqrt_equality_condition_l1172_117268

theorem triangle_inequality_sqrt 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c) := 
sorry

theorem equality_condition 
  {a b c : ℝ} 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (Real.sqrt (a + b - c) + Real.sqrt (c + a - b) + Real.sqrt (b + c - a) 
  = Real.sqrt a + Real.sqrt b + Real.sqrt c) → 
  (a = b ∧ b = c) := 
sorry

end NUMINAMATH_GPT_triangle_inequality_sqrt_equality_condition_l1172_117268


namespace NUMINAMATH_GPT_original_price_l1172_117287

theorem original_price (x : ℝ) (h : x * (1 / 8) = 8) : x = 64 := by
  -- To be proved
  sorry

end NUMINAMATH_GPT_original_price_l1172_117287


namespace NUMINAMATH_GPT_solve_for_y_l1172_117254

theorem solve_for_y (y : ℝ) : 4 * y + 6 * y = 450 - 10 * (y - 5) → y = 25 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1172_117254


namespace NUMINAMATH_GPT_ratio_of_sums_l1172_117239

theorem ratio_of_sums (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49) 
  (h2 : x^2 + y^2 + z^2 = 64) 
  (h3 : a * x + b * y + c * z = 56) : 
  (a + b + c) / (x + y + z) = 7/8 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_sums_l1172_117239


namespace NUMINAMATH_GPT_megan_bottles_l1172_117201

theorem megan_bottles (initial_bottles drank gave_away remaining_bottles : ℕ) 
  (h1 : initial_bottles = 45)
  (h2 : drank = 8)
  (h3 : gave_away = 12) :
  remaining_bottles = initial_bottles - (drank + gave_away) :=
by 
  sorry

end NUMINAMATH_GPT_megan_bottles_l1172_117201


namespace NUMINAMATH_GPT_minimize_abs_a_n_l1172_117251

noncomputable def a_n (n : ℕ) : ℝ :=
  14 - (3 / 4) * (n - 1)

theorem minimize_abs_a_n : ∃ n : ℕ, n = 20 ∧ ∀ m : ℕ, |a_n n| ≤ |a_n m| := by
  sorry

end NUMINAMATH_GPT_minimize_abs_a_n_l1172_117251


namespace NUMINAMATH_GPT_savannah_wraps_4_with_third_roll_l1172_117224

variable (gifts total_rolls : ℕ)
variable (wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ)
variable (no_leftover : Prop)

def savannah_wrapping_presents (gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ) (no_leftover : Prop) : Prop :=
  gifts = 12 ∧
  total_rolls = 3 ∧
  wrap_with_roll1 = 3 ∧
  wrap_with_roll2 = 5 ∧
  remaining_wrap_with_roll3 = gifts - (wrap_with_roll1 + wrap_with_roll2) ∧
  no_leftover = (total_rolls = 3) ∧ (wrap_with_roll1 + wrap_with_roll2 + remaining_wrap_with_roll3 = gifts)

theorem savannah_wraps_4_with_third_roll
  (h : savannah_wrapping_presents gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 no_leftover) :
  remaining_wrap_with_roll3 = 4 :=
by
  sorry

end NUMINAMATH_GPT_savannah_wraps_4_with_third_roll_l1172_117224


namespace NUMINAMATH_GPT_rate_calculation_l1172_117278

noncomputable def rate_per_sq_meter
  (lawn_length : ℝ) (lawn_breadth : ℝ)
  (road_width : ℝ) (total_cost : ℝ) : ℝ :=
  let area_road_1 := road_width * lawn_breadth
  let area_road_2 := road_width * lawn_length
  let area_intersection := road_width * road_width
  let total_area_roads := (area_road_1 + area_road_2) - area_intersection
  total_cost / total_area_roads

theorem rate_calculation :
  rate_per_sq_meter 100 60 10 4500 = 3 := by
  sorry

end NUMINAMATH_GPT_rate_calculation_l1172_117278
