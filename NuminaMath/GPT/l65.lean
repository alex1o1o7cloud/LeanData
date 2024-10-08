import Mathlib

namespace PQRS_product_l65_65757

theorem PQRS_product :
  let P := (Real.sqrt 2012 + Real.sqrt 2013)
  let Q := (-Real.sqrt 2012 - Real.sqrt 2013)
  let R := (Real.sqrt 2012 - Real.sqrt 2013)
  let S := (Real.sqrt 2013 - Real.sqrt 2012)
  P * Q * R * S = 1 :=
by
  sorry

end PQRS_product_l65_65757


namespace solve_for_x_l65_65906

theorem solve_for_x (x : ℝ) (h : (1 / 2) * (1 / 7) * x = 14) : x = 196 :=
by
  sorry

end solve_for_x_l65_65906


namespace together_time_l65_65530

theorem together_time (P_time Q_time : ℝ) (hP : P_time = 4) (hQ : Q_time = 6) : (1 / ((1 / P_time) + (1 / Q_time))) = 2.4 :=
by
  sorry

end together_time_l65_65530


namespace james_distance_ridden_l65_65802

theorem james_distance_ridden : 
  let speed := 16 
  let time := 5 
  speed * time = 80 := 
by
  sorry

end james_distance_ridden_l65_65802


namespace correct_remainder_l65_65997

-- Define the problem
def count_valid_tilings (n k : Nat) : Nat :=
  Nat.factorial (n + k) / (Nat.factorial n * Nat.factorial k) * (3 ^ (n + k) - 3 * 2 ^ (n + k) + 3)

noncomputable def tiles_mod_1000 : Nat :=
  let pairs := [(8, 0), (6, 1), (4, 2), (2, 3), (0, 4)]
  let M := pairs.foldl (λ acc (nk : Nat × Nat) => acc + count_valid_tilings nk.1 nk.2) 0
  M % 1000

theorem correct_remainder : tiles_mod_1000 = 328 :=
  by sorry

end correct_remainder_l65_65997


namespace find_a_l65_65627

-- Define the conditions of the problem
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a + 3, 1, -3) -- Coefficients of line1: (a+3)x + y - 3 = 0
def line2 (a : ℝ) : ℝ × ℝ × ℝ := (5, a - 3, 4)  -- Coefficients of line2: 5x + (a-3)y + 4 = 0

-- Definition of direction vector and normal vector
def direction_vector (a : ℝ) : ℝ × ℝ := (1, -(a + 3))
def normal_vector (a : ℝ) : ℝ × ℝ := (5, a - 3)

-- Proof statement
theorem find_a (a : ℝ) : (direction_vector a = normal_vector a) → a = -2 :=
by {
  -- Insert proof here
  sorry
}

end find_a_l65_65627


namespace tangent_line_is_tangent_l65_65077

noncomputable def func1 (x : ℝ) : ℝ := x + 1 + Real.log x
noncomputable def func2 (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

theorem tangent_line_is_tangent
  (a : ℝ) (h_tangent : ∃ x₀ : ℝ, func2 a x₀ = 2 * x₀ ∧ (deriv (func2 a) x₀ = 2))
  (deriv_eq : deriv func1 1 = 2)
  : a = 4 :=
by
  sorry

end tangent_line_is_tangent_l65_65077


namespace determinant_in_terms_of_roots_l65_65388

theorem determinant_in_terms_of_roots 
  (r s t a b c : ℝ)
  (h1 : a^3 - r*a^2 + s*a - t = 0)
  (h2 : b^3 - r*b^2 + s*b - t = 0)
  (h3 : c^3 - r*c^2 + s*c - t = 0) :
  (2 + a) * ((2 + b) * (2 + c) - 4) - 2 * (2 * (2 + c) - 4) + 2 * (2 * 2 - (2 + b) * 2) = t - 2 * s :=
by
  sorry

end determinant_in_terms_of_roots_l65_65388


namespace num_children_attended_show_l65_65425

def ticket_price_adult : ℕ := 26
def ticket_price_child : ℕ := 13
def num_adults : ℕ := 183
def total_revenue : ℕ := 5122

theorem num_children_attended_show : ∃ C : ℕ, (num_adults * ticket_price_adult + C * ticket_price_child = total_revenue) ∧ C = 28 :=
by
  sorry

end num_children_attended_show_l65_65425


namespace maximize_revenue_l65_65235

theorem maximize_revenue (p : ℝ) (h : p ≤ 30) : 
  (∀ q : ℝ, q ≤ 30 → (150 * 18.75 - 4 * (18.75:ℝ)^2) ≥ (150 * q - 4 * q^2)) ↔ p = 18.75 := 
sorry

end maximize_revenue_l65_65235


namespace negation_of_proposition_l65_65127

theorem negation_of_proposition {c : ℝ} (h : ∃ (c : ℝ), c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) :
  ∀ (c : ℝ), c > 0 → ¬ ∃ x : ℝ, x^2 - x + c = 0 :=
by
  sorry

end negation_of_proposition_l65_65127


namespace breadth_of_water_tank_l65_65581

theorem breadth_of_water_tank (L H V : ℝ) (n : ℕ) (avg_displacement : ℝ) (total_displacement : ℝ)
  (h_len : L = 40)
  (h_height : H = 0.25)
  (h_avg_disp : avg_displacement = 4)
  (h_number : n = 50)
  (h_total_disp : total_displacement = avg_displacement * n)
  (h_displacement_value : total_displacement = 200) :
  (40 * B * 0.25 = 200) → B = 20 :=
by
  intro h_eq
  sorry

end breadth_of_water_tank_l65_65581


namespace jose_share_is_correct_l65_65616

noncomputable def total_profit : ℝ := 
  5000 - 2000 + 7000 + 1000 - 3000 + 10000 + 500 + 4000 - 2500 + 6000 + 8000 - 1000

noncomputable def tom_investment_ratio : ℝ := 30000 * 12
noncomputable def jose_investment_ratio : ℝ := 45000 * 10
noncomputable def maria_investment_ratio : ℝ := 60000 * 8

noncomputable def total_investment_ratio : ℝ := tom_investment_ratio + jose_investment_ratio + maria_investment_ratio

noncomputable def jose_share : ℝ := (jose_investment_ratio / total_investment_ratio) * total_profit

theorem jose_share_is_correct : jose_share = 14658 := 
by 
  sorry

end jose_share_is_correct_l65_65616


namespace A_intersection_B_eq_intersection_set_l65_65985

def A : Set ℝ := {x : ℝ | x * (x - 2) < 0}
def B : Set ℝ := {x : ℝ | x > 1}
def intersection_set := {x : ℝ | 1 < x ∧ x < 2}

theorem A_intersection_B_eq_intersection_set : A ∩ B = intersection_set := by
  sorry

end A_intersection_B_eq_intersection_set_l65_65985


namespace evaluate_powers_of_i_l65_65829

-- Define complex number "i"
def i := Complex.I

-- Define the theorem to prove
theorem evaluate_powers_of_i : i^44 + i^444 + 3 = 5 := by
  -- use the cyclic property of i to simplify expressions
  sorry

end evaluate_powers_of_i_l65_65829


namespace Morse_code_distinct_symbols_l65_65396

-- Morse code sequences conditions
def MorseCodeSequence (n : ℕ) := n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

-- Total number of distinct symbols calculation
def total_distinct_symbols : ℕ :=
  2 + 4 + 8 + 16

-- The theorem to prove
theorem Morse_code_distinct_symbols : total_distinct_symbols = 30 := by
  sorry

end Morse_code_distinct_symbols_l65_65396


namespace extremum_f_range_a_for_no_zeros_l65_65082

noncomputable def f (a b x : ℝ) : ℝ :=
  (a * (x - 1) + b * Real.exp x) / Real.exp x

theorem extremum_f (a b : ℝ) (h_a_ne_zero : a ≠ 0) :
  (∃ (x : ℝ), a = -1 ∧ b = 0 ∧ f a b x = -1 / Real.exp 2) := sorry

theorem range_a_for_no_zeros (a : ℝ) :
  (∀ x : ℝ, a * x - a + Real.exp x ≠ 0) ↔ (-Real.exp 2 < a ∧ a < 0) := sorry

end extremum_f_range_a_for_no_zeros_l65_65082


namespace dimes_total_l65_65489

def initial_dimes : ℕ := 9
def added_dimes : ℕ := 7

theorem dimes_total : initial_dimes + added_dimes = 16 := by
  sorry

end dimes_total_l65_65489


namespace part_one_part_two_part_three_l65_65537

-- Define the sequence and the sum of its first n terms
def S (a : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a n - 2 ^ n

-- Prove that a_1 = 2 and a_4 = 40
theorem part_one (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  a 1 = 2 ∧ a 4 = 40 := by
  sorry
  
-- Prove that the sequence {a_{n+1} - 2a_n} is a geometric sequence
theorem part_two (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∃ r : ℕ, (r = 2) ∧ (∀ n, (a (n + 1) - 2 * a n) = r ^ n) := by
  sorry

-- Prove the general term formula for the sequence {a_n}
theorem part_three (a : ℕ → ℕ) (h : ∀ n, S a n = 2 * a n - 2 ^ n) : 
  ∀ n, a n = 2 ^ (n + 1) - 2 := by
  sorry

end part_one_part_two_part_three_l65_65537


namespace incorrect_statement_l65_65493

def angles_on_x_axis := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi}
def angles_on_y_axis := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 2 + k * Real.pi}
def angles_on_axes := {α : ℝ | ∃ (k : ℤ), α = k * Real.pi / 2}
def angles_on_y_eq_neg_x := {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}

theorem incorrect_statement : ¬ (angles_on_y_eq_neg_x = {α : ℝ | ∃ (k : ℤ), α = Real.pi / 4 + 2 * k * Real.pi}) :=
sorry

end incorrect_statement_l65_65493


namespace solution_one_solution_two_solution_three_l65_65359

open Real

noncomputable def problem_one (a b : ℝ) (cosA : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 then 1 else 0

theorem solution_one (a b : ℝ) (cosA : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → problem_one a b cosA = 1 := by
  intros ha hb hcos
  unfold problem_one
  simp [ha, hb, hcos]

noncomputable def problem_two (a b : ℝ) (cosA sinB : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 ∧ sinB = sqrt 10 / 4 then sqrt 10 / 4 else 0

theorem solution_two (a b : ℝ) (cosA sinB : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → sinB = sqrt 10 / 4 → problem_two a b cosA sinB = sqrt 10 / 4 := by
  intros ha hb hcos hsinB
  unfold problem_two
  simp [ha, hb, hcos, hsinB]

noncomputable def problem_three (a b : ℝ) (cosA sinB sin2AminusB : ℝ) : ℝ :=
if a = sqrt 6 ∧ b = 2 * 1 ∧ cosA = -1/4 ∧ sinB = sqrt 10 / 4 ∧ sin2AminusB = sqrt 10 / 8 then sqrt 10 / 8 else 0

theorem solution_three (a b : ℝ) (cosA sinB sin2AminusB : ℝ) :
  a = sqrt 6 → b = 2 * 1 → cosA = -1/4 → sinB = sqrt 10 / 4 → sin2AminusB = sqrt 10 / 8 → problem_three a b cosA sinB sin2AminusB = sqrt 10 / 8 := by
  intros ha hb hcos hsinB hsin2AminusB
  unfold problem_three
  simp [ha, hb, hcos, hsinB, hsin2AminusB]

end solution_one_solution_two_solution_three_l65_65359


namespace river_current_speed_l65_65299

noncomputable section

variables {d r w : ℝ}

def time_equation_normal_speed (d r w : ℝ) : Prop :=
  (d / (r + w)) + 4 = (d / (r - w))

def time_equation_tripled_speed (d r w : ℝ) : Prop :=
  (d / (3 * r + w)) + 2 = (d / (3 * r - w))

theorem river_current_speed (d r : ℝ) (h1 : time_equation_normal_speed d r w) (h2 : time_equation_tripled_speed d r w) : w = 2 :=
sorry

end river_current_speed_l65_65299


namespace find_weight_of_b_l65_65890

theorem find_weight_of_b (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) : B = 31 :=
sorry

end find_weight_of_b_l65_65890


namespace battery_life_in_standby_l65_65589

noncomputable def remaining_battery_life (b_s : ℝ) (b_a : ℝ) (t_total : ℝ) (t_active : ℝ) : ℝ :=
  let standby_rate := 1 / b_s
  let active_rate := 1 / b_a
  let standby_time := t_total - t_active
  let consumption_active := t_active * active_rate
  let consumption_standby := standby_time * standby_rate
  let total_consumption := consumption_active + consumption_standby
  let remaining_battery := 1 - total_consumption
  remaining_battery * b_s

theorem battery_life_in_standby :
  remaining_battery_life 30 4 10 1.5 = 10.25 := sorry

end battery_life_in_standby_l65_65589


namespace abs_neg_three_l65_65783

theorem abs_neg_three : abs (-3) = 3 := by
  sorry

end abs_neg_three_l65_65783


namespace inequality_transitive_l65_65090

theorem inequality_transitive (a b c : ℝ) (h : a < b) (h' : b < c) : a - c < b - c :=
by
  sorry

end inequality_transitive_l65_65090


namespace impossible_fractions_l65_65068

theorem impossible_fractions (a b c r s t : ℕ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_r : 0 < r) (h_pos_s : 0 < s) (h_pos_t : 0 < t)
  (h1 : a * b + 1 = r ^ 2) (h2 : a * c + 1 = s ^ 2) (h3 : b * c + 1 = t ^ 2) :
  ¬ (∃ (k1 k2 k3 : ℕ), rt / s = k1 ∧ rs / t = k2 ∧ st / r = k3) :=
by
  sorry

end impossible_fractions_l65_65068


namespace simultaneous_equations_solution_l65_65290

theorem simultaneous_equations_solution :
  ∃ x y : ℚ, (3 * x + 4 * y = 20) ∧ (9 * x - 8 * y = 36) ∧ (x = 76 / 15) ∧ (y = 18 / 15) :=
by
  sorry

end simultaneous_equations_solution_l65_65290


namespace Lee_charge_per_lawn_l65_65089

theorem Lee_charge_per_lawn
  (x : ℝ)
  (mowed_lawns : ℕ)
  (total_earned : ℝ)
  (tips : ℝ)
  (tip_amount : ℝ)
  (num_customers_tipped : ℕ)
  (earnings_from_mowing : ℝ)
  (total_earning_with_tips : ℝ) :
  mowed_lawns = 16 →
  total_earned = 558 →
  num_customers_tipped = 3 →
  tip_amount = 10 →
  tips = num_customers_tipped * tip_amount →
  earnings_from_mowing = mowed_lawns * x →
  total_earning_with_tips = earnings_from_mowing + tips →
  total_earning_with_tips = total_earned →
  x = 33 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end Lee_charge_per_lawn_l65_65089


namespace profit_calculation_l65_65334

def actors_cost : ℕ := 1200
def people_count : ℕ := 50
def cost_per_person : ℕ := 3
def food_cost : ℕ := people_count * cost_per_person
def total_cost_actors_food : ℕ := actors_cost + food_cost
def equipment_rental_cost : ℕ := 2 * total_cost_actors_food
def total_movie_cost : ℕ := total_cost_actors_food + equipment_rental_cost
def movie_sale_price : ℕ := 10000
def profit : ℕ := movie_sale_price - total_movie_cost

theorem profit_calculation : profit = 5950 := by
  sorry

end profit_calculation_l65_65334


namespace monotonic_range_of_a_l65_65501

noncomputable def f (a x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1
noncomputable def f' (a x : ℝ) : ℝ := -3*x^2 + 2*a*x - 1

theorem monotonic_range_of_a (a : ℝ) : 
  (∀ x : ℝ, f' a x ≤ 0) ↔ (-Real.sqrt 3 ≤ a ∧ a ≤ Real.sqrt 3) :=
by 
  sorry

end monotonic_range_of_a_l65_65501


namespace distance_at_1_5_l65_65046

def total_distance : ℝ := 174
def speed : ℝ := 60
def travel_time (x : ℝ) : ℝ := total_distance - speed * x

theorem distance_at_1_5 :
  travel_time 1.5 = 84 := by
  sorry

end distance_at_1_5_l65_65046


namespace least_number_to_multiply_l65_65441

theorem least_number_to_multiply (x : ℕ) :
  (72 * x) % 112 = 0 → x = 14 :=
by 
  sorry

end least_number_to_multiply_l65_65441


namespace total_gas_consumed_l65_65496

def highway_consumption_rate : ℕ := 3
def city_consumption_rate : ℕ := 5

-- Distances driven each day
def day_1_highway_miles : ℕ := 200
def day_1_city_miles : ℕ := 300

def day_2_highway_miles : ℕ := 300
def day_2_city_miles : ℕ := 500

def day_3_highway_miles : ℕ := 150
def day_3_city_miles : ℕ := 350

-- Function to calculate the total consumption for a given day
def daily_consumption (highway_miles city_miles : ℕ) : ℕ :=
  (highway_miles * highway_consumption_rate) + (city_miles * city_consumption_rate)

-- Total consumption over three days
def total_consumption : ℕ :=
  (daily_consumption day_1_highway_miles day_1_city_miles) +
  (daily_consumption day_2_highway_miles day_2_city_miles) +
  (daily_consumption day_3_highway_miles day_3_city_miles)

-- Theorem stating the total consumption over the three days
theorem total_gas_consumed : total_consumption = 7700 := by
  sorry

end total_gas_consumed_l65_65496


namespace time_to_meet_in_minutes_l65_65357

def distance_between_projectiles : ℕ := 1998
def speed_projectile_1 : ℕ := 444
def speed_projectile_2 : ℕ := 555

theorem time_to_meet_in_minutes : 
  (distance_between_projectiles / (speed_projectile_1 + speed_projectile_2)) * 60 = 120 := 
by
  sorry

end time_to_meet_in_minutes_l65_65357


namespace num_integers_D_l65_65653

theorem num_integers_D :
  ∃ (D : ℝ) (n : ℕ), 
    (∀ (a b : ℝ), -1/4 < a → a < 1/4 → -1/4 < b → b < 1/4 → abs (a^2 - D * b^2) < 1) → n = 32 :=
sorry

end num_integers_D_l65_65653


namespace jeep_initial_distance_l65_65371

theorem jeep_initial_distance (D : ℝ) (h1 : ∀ t : ℝ, t = 4 → D / t = 103.33 * (3 / 8)) :
  D = 275.55 :=
sorry

end jeep_initial_distance_l65_65371


namespace summation_indices_equal_l65_65461

theorem summation_indices_equal
  (a : ℕ → ℕ)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (h_bound : ∀ i, a i ≤ 100)
  (h_length : ∀ i, i < 16) :
  ∃ i j k l, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ a i + a j = a k + a l := 
by {
  sorry
}

end summation_indices_equal_l65_65461


namespace carrie_spent_l65_65368

-- Define the cost of one t-shirt
def cost_per_tshirt : ℝ := 9.65

-- Define the number of t-shirts bought
def num_tshirts : ℝ := 12

-- Define the total cost function
def total_cost (cost_per_tshirt : ℝ) (num_tshirts : ℝ) : ℝ := cost_per_tshirt * num_tshirts

-- State the theorem which we need to prove
theorem carrie_spent :
  total_cost cost_per_tshirt num_tshirts = 115.80 :=
by
  sorry

end carrie_spent_l65_65368


namespace car_speed_l65_65136

variable (Distance : ℕ) (Time : ℕ)
variable (h1 : Distance = 495)
variable (h2 : Time = 5)

theorem car_speed (Distance Time : ℕ) (h1 : Distance = 495) (h2 : Time = 5) : 
  Distance / Time = 99 :=
by
  sorry

end car_speed_l65_65136


namespace wrapping_paper_amount_l65_65930

theorem wrapping_paper_amount (x : ℝ) (h : x + (3/4) * x + (x + (3/4) * x) = 7) : x = 2 :=
by
  sorry

end wrapping_paper_amount_l65_65930


namespace total_money_is_102_l65_65868

-- Defining the amounts of money each person has
def Jack_money : ℕ := 26
def Ben_money : ℕ := Jack_money - 9
def Eric_money : ℕ := Ben_money - 10
def Anna_money : ℕ := Jack_money * 2

-- Defining the total amount of money
def total_money : ℕ := Eric_money + Ben_money + Jack_money + Anna_money

-- Proving the total money is 102
theorem total_money_is_102 : total_money = 102 :=
by
  -- this is where the proof would go
  sorry

end total_money_is_102_l65_65868


namespace jimmy_points_lost_for_bad_behavior_l65_65481

theorem jimmy_points_lost_for_bad_behavior (points_per_exam : ℕ) (num_exams : ℕ) (points_needed : ℕ)
  (extra_points_allowed : ℕ) (total_points_earned : ℕ) (current_points : ℕ)
  (h1 : points_per_exam = 20) (h2 : num_exams = 3) (h3 : points_needed = 50)
  (h4 : extra_points_allowed = 5) (h5 : total_points_earned = points_per_exam * num_exams)
  (h6 : current_points = points_needed + extra_points_allowed) :
  total_points_earned - current_points = 5 :=
by
  sorry

end jimmy_points_lost_for_bad_behavior_l65_65481


namespace g_difference_l65_65865

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 3) * (n + 5) + 2

theorem g_difference (s : ℕ) : g s - g (s - 1) = (3 * s^2 + 9 * s + 8) / 4 :=
by
  -- skip the proof
  sorry

end g_difference_l65_65865


namespace painting_price_after_new_discount_l65_65463

namespace PaintingPrice

-- Define the original price and the price Sarah paid
def original_price (x : ℕ) : Prop := x / 5 = 15

-- Define the new discounted price
def new_discounted_price (y x : ℕ) : Prop := y = x * 2 / 3

-- Theorem to prove the final price considering both conditions
theorem painting_price_after_new_discount (x y : ℕ) 
  (h1 : original_price x)
  (h2 : new_discounted_price y x) : y = 50 :=
by
  sorry

end PaintingPrice

end painting_price_after_new_discount_l65_65463


namespace problem1_problem2_problem3_problem4_l65_65973

theorem problem1 : (-3 + 8 - 7 - 15) = -17 := 
sorry

theorem problem2 : (23 - 6 * (-3) + 2 * (-4)) = 33 := 
sorry

theorem problem3 : (-8 / (4 / 5) * (-2 / 3)) = 20 / 3 := 
sorry

theorem problem4 : (-2^2 - 9 * (-1 / 3)^2 + abs (-4)) = -1 := 
sorry

end problem1_problem2_problem3_problem4_l65_65973


namespace quadratic_root_a_value_l65_65440

theorem quadratic_root_a_value (a : ℝ) :
  (∃ x : ℝ, x = -2 ∧ x^2 + (3 / 2) * a * x - a^2 = 0) → (a = 1 ∨ a = -4) := 
by
  intro h
  sorry

end quadratic_root_a_value_l65_65440


namespace ratio_problem_l65_65245

theorem ratio_problem 
  (a b c d : ℚ)
  (h1 : a / b = 3)
  (h2 : b / c = 2 / 5)
  (h3 : c / d = 9) : 
  d / a = 5 / 54 :=
by
  sorry

end ratio_problem_l65_65245


namespace molly_total_cost_l65_65582

def cost_per_package : ℕ := 5
def num_parents : ℕ := 2
def num_brothers : ℕ := 3
def num_children_per_brother : ℕ := 2
def num_spouse_per_brother : ℕ := 1

def total_num_relatives : ℕ := 
  let parents_and_siblings := num_parents + num_brothers
  let additional_relatives := num_brothers * (1 + num_spouse_per_brother + num_children_per_brother)
  parents_and_siblings + additional_relatives

def total_cost : ℕ :=
  total_num_relatives * cost_per_package

theorem molly_total_cost : total_cost = 85 := sorry

end molly_total_cost_l65_65582


namespace geometric_sequence_sum_inverse_equals_l65_65298

variable (a : ℕ → ℝ)
variable (n : ℕ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃(r : ℝ), ∀ (n : ℕ), a (n + 1) = r * a n

theorem geometric_sequence_sum_inverse_equals (a : ℕ → ℝ)
  (h_geo : is_geometric_sequence a)
  (h_sum : a 5 + a 6 + a 7 + a 8 = 15 / 8)
  (h_prod : a 6 * a 7 = -9 / 8) :
  (1 / a 5) + (1 / a 6) + (1 / a 7) + (1 / a 8) = -5 / 3 :=
by
  sorry

end geometric_sequence_sum_inverse_equals_l65_65298


namespace probability_prime_and_cube_is_correct_l65_65246

-- Conditions based on the problem
def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_cube (n : ℕ) : Prop :=
  n = 1 ∨ n = 8

def possible_outcomes := 8 * 8
def successful_outcomes := 4 * 2

noncomputable def probability_of_prime_and_cube :=
  (successful_outcomes : ℝ) / (possible_outcomes : ℝ)

theorem probability_prime_and_cube_is_correct :
  probability_of_prime_and_cube = 1 / 8 :=
by
  sorry

end probability_prime_and_cube_is_correct_l65_65246


namespace number_of_green_fish_l65_65479

theorem number_of_green_fish (total_fish : ℕ) (blue_fish : ℕ) (orange_fish : ℕ) (green_fish : ℕ)
  (h1 : total_fish = 80)
  (h2 : blue_fish = total_fish / 2)
  (h3 : orange_fish = blue_fish - 15)
  (h4 : green_fish = total_fish - blue_fish - orange_fish)
  : green_fish = 15 :=
by sorry

end number_of_green_fish_l65_65479


namespace integer_solutions_equation_l65_65677

theorem integer_solutions_equation : 
  (∃ x y : ℤ, (1 / (2022 : ℚ) = 1 / (x : ℚ) + 1 / (y : ℚ))) → 
  ∃! (n : ℕ), n = 53 :=
by
  sorry

end integer_solutions_equation_l65_65677


namespace g_difference_l65_65267

def g (n : ℕ) : ℚ := (1 / 4) * n * (n + 1) * (n + 2) * (n + 3)

theorem g_difference (s : ℕ) : g s - g (s - 1) = s * (s + 1) * (s + 2) := 
by sorry

end g_difference_l65_65267


namespace cost_of_painting_l65_65916

def area_of_house : ℕ := 484
def price_per_sqft : ℕ := 20

theorem cost_of_painting : area_of_house * price_per_sqft = 9680 := by
  sorry

end cost_of_painting_l65_65916


namespace function_passes_through_vertex_l65_65910

theorem function_passes_through_vertex (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) : a^(2 - 2) + 1 = 2 :=
by
  sorry

end function_passes_through_vertex_l65_65910


namespace factorization_2109_two_digit_l65_65675

theorem factorization_2109_two_digit (a b: ℕ) : 
  2109 = a * b ∧ 10 ≤ a ∧ a ≤ 99 ∧ 10 ≤ b ∧ b ≤ 99 → false :=
by
  sorry

end factorization_2109_two_digit_l65_65675


namespace combined_tax_rate_l65_65408

-- Definitions of the problem conditions
def tax_rate_Mork : ℝ := 0.40
def tax_rate_Mindy : ℝ := 0.25

-- Asserts the condition that Mindy earned 4 times as much as Mork
def income_ratio (income_Mindy income_Mork : ℝ) := income_Mindy = 4 * income_Mork

-- The theorem to be proved: The combined tax rate is 28%.
theorem combined_tax_rate (income_Mork income_Mindy total_income total_tax : ℝ)
  (h_income_ratio : income_ratio income_Mindy income_Mork)
  (total_income_eq : total_income = income_Mork + income_Mindy)
  (total_tax_eq : total_tax = tax_rate_Mork * income_Mork + tax_rate_Mindy * income_Mindy) :
  total_tax / total_income = 0.28 := sorry

end combined_tax_rate_l65_65408


namespace father_l65_65432

theorem father's_age (M F : ℕ) (h1 : M = 2 * F / 5) (h2 : M + 6 = (F + 6) / 2) : F = 30 :=
by
  sorry

end father_l65_65432


namespace circle_outside_hexagon_area_l65_65607

theorem circle_outside_hexagon_area :
  let r := (Real.sqrt 2) / 2
  let s := 1
  let area_circle := π * r^2
  let area_hexagon := 3 * Real.sqrt 3 / 2 * s^2
  area_circle - area_hexagon = (π / 2) - (3 * Real.sqrt 3 / 2) :=
by
  sorry

end circle_outside_hexagon_area_l65_65607


namespace product_polynomials_l65_65907

theorem product_polynomials (x : ℝ) : 
  (1 + x^3) * (1 - 2 * x + x^4) = 1 - 2 * x + x^3 - x^4 + x^7 :=
by sorry

end product_polynomials_l65_65907


namespace eiffel_tower_model_height_l65_65006

theorem eiffel_tower_model_height 
  (H1 : ℝ) (W1 : ℝ) (W2 : ℝ) (H2 : ℝ)
  (h1 : H1 = 324)
  (w1 : W1 = 8000000)  -- converted 8000 tons to 8000000 kg
  (w2 : W2 = 1)
  (h_eq : (H2 / H1)^3 = W2 / W1) : 
  H2 = 1.62 :=
by
  rw [h1, w1, w2] at h_eq
  sorry

end eiffel_tower_model_height_l65_65006


namespace unique_function_f_l65_65543

theorem unique_function_f (f : ℝ → ℝ)
    (h1 : ∀ x : ℝ, f x = -f (-x))
    (h2 : ∀ x : ℝ, f (x + 1) = f x + 1)
    (h3 : ∀ x : ℝ, x ≠ 0 → f (1 / x) = 1 / x^2 * f x) :
    ∀ x : ℝ, f x = x := 
sorry

end unique_function_f_l65_65543


namespace tangent_line_at_0_maximum_integer_value_of_a_l65_65374

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - a*x + 2

-- Part (1)
-- Prove that the equation of the tangent line to f(x) at x = 0 is x + y - 2 = 0 when a = 2
theorem tangent_line_at_0 {a : ℝ} (h : a = 2) : ∀ x y : ℝ, (y = f x a) → (x = 0) → (y = 2 - x) :=
by 
  sorry

-- Part (2)
-- Prove that if f(x) + 2x + x log(x+1) ≥ 0 holds for all x ≥ 0, then the maximum integer value of a is 4
theorem maximum_integer_value_of_a 
  (h : ∀ x : ℝ, x ≥ 0 → f x a + 2 * x + x * Real.log (x + 1) ≥ 0) : a ≤ 4 :=
by
  sorry

end tangent_line_at_0_maximum_integer_value_of_a_l65_65374


namespace percentage_decrease_of_b_l65_65525

theorem percentage_decrease_of_b (a b x m : ℝ) (p : ℝ) 
  (a_pos : 0 < a) (b_pos : 0 < b)
  (h1 : a / b = 4 / 5)
  (h2 : x = a + 0.25 * a)
  (h3 : m = b * (1 - p / 100))
  (h4 : m / x = 0.4) :
  p = 60 :=
by
  sorry

end percentage_decrease_of_b_l65_65525


namespace intersection_of_sets_l65_65073

noncomputable def setM : Set ℝ := { x | x + 1 > 0 }
noncomputable def setN : Set ℝ := { x | 2 * x - 1 < 0 }

theorem intersection_of_sets : setM ∩ setN = { x : ℝ | -1 < x ∧ x < 1 / 2 } := by
  sorry

end intersection_of_sets_l65_65073


namespace nasadkas_in_barrel_l65_65112

def capacity (B N V : ℚ) :=
  (B + 20 * V = 3 * B) ∧ (19 * B + N + 15.5 * V = 20 * B + 8 * V)

theorem nasadkas_in_barrel (B N V : ℚ) (h : capacity B N V) : B / N = 4 :=
by
  sorry

end nasadkas_in_barrel_l65_65112


namespace right_triangle_wy_expression_l65_65884

theorem right_triangle_wy_expression (α β : ℝ) (u v w y : ℝ)
    (h1 : (∀ x : ℝ, x^2 - u * x + v = 0 → x = Real.sin α ∨ x = Real.sin β))
    (h2 : (∀ x : ℝ, x^2 - w * x + y = 0 → x = Real.cos α ∨ x = Real.cos β))
    (h3 : α + β = Real.pi / 2) :
    w * y = u * v :=
sorry

end right_triangle_wy_expression_l65_65884


namespace compute_fraction_l65_65404

theorem compute_fraction : ((5 * 7) - 3) / 9 = 32 / 9 := by
  sorry

end compute_fraction_l65_65404


namespace find_ray_solutions_l65_65639

noncomputable def polynomial (a x : ℝ) : ℝ :=
  x^3 - (a^2 + a + 1) * x^2 + (a^3 + a^2 + a) * x - a^3

theorem find_ray_solutions (a : ℝ) :
  (∀ x : ℝ, polynomial a x ≥ 0 → ∃ b : ℝ, ∀ y ≥ b, polynomial a y ≥ 0) ↔ a = 1 ∨ a = -1 :=
sorry

end find_ray_solutions_l65_65639


namespace problem1_problem2_l65_65561

-- Define propositions P and Q under the given conditions
def P (a x : ℝ) : Prop := 2 * x^2 - 5 * a * x - 3 * a^2 < 0

def Q (x : ℝ) : Prop := (2 * Real.sin x > 1) ∧ (x^2 - x - 2 < 0)

-- Problem 1: Prove that if a = 2 and p ∧ q holds true, then the range of x is (π/6, 2)
theorem problem1 (x : ℝ) (hx1 : P 2 x ∧ Q x) : (Real.pi / 6 < x ∧ x < 2) :=
sorry

-- Problem 2: Prove that if ¬P is a sufficient but not necessary condition for ¬Q, then the range of a is [2/3, ∞)
theorem problem2 (a : ℝ) (h₁ : ∀ x, Q x → P a x) (h₂ : ∃ x, Q x → ¬P a x) : a ≥ 2 / 3 :=
sorry

end problem1_problem2_l65_65561


namespace product_of_four_integers_l65_65874

theorem product_of_four_integers:
  ∃ (A B C D : ℚ) (x : ℚ), 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D ∧
  A + B + C + D = 40 ∧
  A - 3 = x ∧ B + 3 = x ∧ C / 2 = x ∧ D * 2 = x ∧
  A * B * C * D = (9089600 / 6561) := by
  sorry

end product_of_four_integers_l65_65874


namespace distinct_digits_solution_l65_65132

theorem distinct_digits_solution (A B C : ℕ)
  (h1 : A + B = 10)
  (h2 : C + A = 9)
  (h3 : B + C = 9)
  (h4 : A ≠ B)
  (h5 : B ≠ C)
  (h6 : C ≠ A)
  (h7 : 0 < A)
  (h8 : 0 < B)
  (h9 : 0 < C)
  : A = 1 ∧ B = 9 ∧ C = 8 := 
  by sorry

end distinct_digits_solution_l65_65132


namespace boat_speed_in_still_water_l65_65019

/-- Given a boat's speed along the stream and against the stream, prove its speed in still water. -/
theorem boat_speed_in_still_water (b s : ℝ) 
  (h1 : b + s = 11)
  (h2 : b - s = 5) : b = 8 :=
sorry

end boat_speed_in_still_water_l65_65019


namespace square_pieces_placement_l65_65984

theorem square_pieces_placement (n : ℕ) (H : n = 8) :
  {m : ℕ // m = 17} :=
sorry

end square_pieces_placement_l65_65984


namespace cubic_polynomials_l65_65853

theorem cubic_polynomials (a b c a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
    (h1 : a - 1/b = r₁ ∧ b - 1/c = r₂ ∧ c - 1/a = r₃)
    (h2 : r₁ + r₂ + r₃ = 5)
    (h3 : r₁ * r₂ + r₂ * r₃ + r₃ * r₁ = -15)
    (h4 : r₁ * r₂ * r₃ = -3)
    (h5 : a₁ * b₁ * c₁ = 1 + Real.sqrt 2 ∨ a₁ * b₁ * c₁ = 1 - Real.sqrt 2)
    (h6 : a₂ * b₂ * c₂ = 1 + Real.sqrt 2 ∨ a₂ * b₂ * c₂ = 1 - Real.sqrt 2) :
    (-(a₁ * b₁ * c₁))^3 + (-(a₂ * b₂ * c₂))^3 = -14 := sorry

end cubic_polynomials_l65_65853


namespace matrix_addition_correct_l65_65687

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 4, -2], ![5, -3, 1]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![ -3,  2, -4], ![ 1, -6,  3], ![-2,  4,  0]]

def expectedSum : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![-1,  1, -1], ![ 1, -2,  1], ![ 3,  1,  1]]

theorem matrix_addition_correct :
  A + B = expectedSum := by
  sorry

end matrix_addition_correct_l65_65687


namespace inverse_function_point_l65_65208

theorem inverse_function_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ ∀ y, (∀ x, y = a^(x-3) + 1) → (2, 3) ∈ {(y, x) | y = a^(x-3) + 1} :=
by
  sorry

end inverse_function_point_l65_65208


namespace smallest_number_condition_l65_65741

theorem smallest_number_condition :
  ∃ n : ℕ, (n + 1) % 12 = 0 ∧
           (n + 1) % 18 = 0 ∧
           (n + 1) % 24 = 0 ∧
           (n + 1) % 32 = 0 ∧
           (n + 1) % 40 = 0 ∧
           n = 2879 :=
sorry

end smallest_number_condition_l65_65741


namespace sale_in_first_month_l65_65674

theorem sale_in_first_month (sale1 sale2 sale3 sale4 sale5 : ℕ) 
  (h1 : sale1 = 5660) (h2 : sale2 = 6200) (h3 : sale3 = 6350) (h4 : sale4 = 6500) 
  (h_avg : (sale1 + sale2 + sale3 + sale4 + sale5) / 5 = 6000) : 
  sale5 = 5290 := 
by
  sorry

end sale_in_first_month_l65_65674


namespace division_problem_l65_65220

theorem division_problem :
  (0.25 / 0.005) / 0.1 = 500 := by
  sorry

end division_problem_l65_65220


namespace more_silverfish_than_goldfish_l65_65295

variable (n G S R : ℕ)

-- Condition 1: If the cat eats all the goldfish, the number of remaining fish is \(\frac{2}{3}\)n - 1
def condition1 := n - G = (2 * n) / 3 - 1

-- Condition 2: If the cat eats all the redfish, the number of remaining fish is \(\frac{2}{3}\)n + 4
def condition2 := n - R = (2 * n) / 3 + 4

-- The goal: Silverfish are more numerous than goldfish by 2
theorem more_silverfish_than_goldfish (h1 : condition1 n G) (h2 : condition2 n R) :
  S = (n / 3) + 3 → G = (n / 3) + 1 → S - G = 2 :=
by
  sorry

end more_silverfish_than_goldfish_l65_65295


namespace max_stamps_without_discount_theorem_l65_65928

def total_money := 5000
def price_per_stamp := 50
def max_stamps_without_discount := 100

theorem max_stamps_without_discount_theorem :
  price_per_stamp * max_stamps_without_discount ≤ total_money ∧
  ∀ n, n > max_stamps_without_discount → price_per_stamp * n > total_money := by
  sorry

end max_stamps_without_discount_theorem_l65_65928


namespace blue_to_red_face_area_ratio_l65_65936

theorem blue_to_red_face_area_ratio :
  let original_cube_dim := 13
  let red_face_area := 6 * original_cube_dim^2
  let total_faces := 6 * original_cube_dim^3
  let blue_face_area := total_faces - red_face_area
  (blue_face_area / red_face_area) = 12 :=
by
  sorry

end blue_to_red_face_area_ratio_l65_65936


namespace find_a_l65_65113

theorem find_a 
  (x : ℤ) 
  (a : ℤ) 
  (h1 : x = 2) 
  (h2 : y = a) 
  (h3 : 2 * x - 3 * y = 5) : a = -1 / 3 := 
by 
  sorry

end find_a_l65_65113


namespace slips_with_3_l65_65717

variable (total_slips : ℕ) (expected_value : ℚ) (num_slips_with_3 : ℕ)

def num_slips_with_9 := total_slips - num_slips_with_3

def expected_value_calc (total_slips expected_value : ℚ) (num_slips_with_3 num_slips_with_9 : ℕ) : ℚ :=
  (num_slips_with_3 / total_slips) * 3 + (num_slips_with_9 / total_slips) * 9

theorem slips_with_3 (h1 : total_slips = 15) (h2 : expected_value = 5.4)
  (h3 : expected_value_calc total_slips expected_value num_slips_with_3 (num_slips_with_9 total_slips num_slips_with_3) = expected_value) :
  num_slips_with_3 = 9 :=
by
  rw [h1, h2] at h3
  sorry

end slips_with_3_l65_65717


namespace largest_number_of_pangs_largest_number_of_pangs_possible_l65_65314

theorem largest_number_of_pangs (x y z : ℕ) 
  (hx : x ≥ 2) 
  (hy : y ≥ 3) 
  (hcost : 3 * x + 4 * y + 9 * z = 100) : 
  z ≤ 9 :=
by sorry

theorem largest_number_of_pangs_possible (x y z : ℕ) 
  (hx : x = 2) 
  (hy : y = 3) 
  (hcost : 3 * x + 4 * y + 9 * z = 100) : 
  z = 9 :=
by sorry

end largest_number_of_pangs_largest_number_of_pangs_possible_l65_65314


namespace lemons_and_oranges_for_100_gallons_l65_65217

-- Given conditions
def lemons_per_gallon := 30 / 40
def oranges_per_gallon := 20 / 40

-- Theorem to be proven
theorem lemons_and_oranges_for_100_gallons : 
  lemons_per_gallon * 100 = 75 ∧ oranges_per_gallon * 100 = 50 := by
  sorry

end lemons_and_oranges_for_100_gallons_l65_65217


namespace rate_of_current_is_5_l65_65269

theorem rate_of_current_is_5 
  (speed_still_water : ℕ)
  (distance_travelled : ℕ)
  (time_travelled : ℚ) 
  (effective_speed_with_current : ℚ) : 
  speed_still_water = 20 ∧ distance_travelled = 5 ∧ time_travelled = 1/5 ∧ 
  effective_speed_with_current = (speed_still_water + 5) →
  effective_speed_with_current * time_travelled = distance_travelled :=
by
  sorry

end rate_of_current_is_5_l65_65269


namespace larger_cube_volume_is_512_l65_65355

def original_cube_volume := 64 -- volume in cubic feet
def scale_factor := 2 -- the factor by which the dimensions are scaled

def side_length (volume : ℕ) : ℕ := volume^(1/3) -- Assuming we have a function to compute cube root

def larger_cube_volume (original_volume : ℕ) (scale_factor : ℕ) : ℕ :=
  let original_side_length := side_length original_volume
  let larger_side_length := scale_factor * original_side_length
  larger_side_length ^ 3

theorem larger_cube_volume_is_512 :
  larger_cube_volume original_cube_volume scale_factor = 512 :=
sorry

end larger_cube_volume_is_512_l65_65355


namespace product_sum_divisible_by_1987_l65_65981

theorem product_sum_divisible_by_1987 :
  let A : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 1) (List.range (1987 + 1)))
  let B : ℕ :=
    List.prod (List.filter (λ x => x % 2 = 0) (List.range (1987 + 1)))
  A + B ≡ 0 [MOD 1987] := by
  -- The proof goes here
  sorry

end product_sum_divisible_by_1987_l65_65981


namespace perpendicular_line_and_plane_implication_l65_65929

variable (l m : Line)
variable (α β : Plane)

-- Given conditions
def line_perpendicular_to_plane (l : Line) (α : Plane) : Prop :=
sorry -- Assume this checks if line l is perpendicular to plane α

def line_in_plane (m : Line) (α : Plane) : Prop :=
sorry -- Assume this checks if line m is included in plane α

def line_perpendicular_to_line (l m : Line) : Prop :=
sorry -- Assume this checks if line l is perpendicular to line m

-- Lean statement for the proof problem
theorem perpendicular_line_and_plane_implication
  (h1 : line_perpendicular_to_plane l α)
  (h2 : line_in_plane m α) :
  line_perpendicular_to_line l m :=
sorry

end perpendicular_line_and_plane_implication_l65_65929


namespace total_ages_l65_65318

variable (Bill_age Caroline_age : ℕ)
variable (h1 : Bill_age = 2 * Caroline_age - 1) (h2 : Bill_age = 17)

theorem total_ages : Bill_age + Caroline_age = 26 :=
by
  sorry

end total_ages_l65_65318


namespace Ian_money_left_l65_65270

-- Definitions based on the conditions
def hours_worked : ℕ := 8
def rate_per_hour : ℕ := 18
def total_money_made : ℕ := hours_worked * rate_per_hour
def money_left : ℕ := total_money_made / 2

-- The statement to be proved 
theorem Ian_money_left : money_left = 72 :=
by
  sorry

end Ian_money_left_l65_65270


namespace quotient_of_a_by_b_l65_65058

-- Definitions based on given conditions
def a : ℝ := 0.0204
def b : ℝ := 17

-- Statement to be proven
theorem quotient_of_a_by_b : a / b = 0.0012 := 
by
  sorry

end quotient_of_a_by_b_l65_65058


namespace integer_difference_divisible_by_n_l65_65495

theorem integer_difference_divisible_by_n (n : ℕ) (h : n > 0) (a : Fin (n+1) → ℤ) :
  ∃ (i j : Fin (n+1)), i ≠ j ∧ (a i - a j) % n = 0 :=
by
  sorry

end integer_difference_divisible_by_n_l65_65495


namespace quadratic_vertex_a_l65_65213

theorem quadratic_vertex_a
  (a b c : ℝ)
  (h1 : ∀ x, (a * x^2 + b * x + c = a * (x - 2)^2 + 5))
  (h2 : a * 0^2 + b * 0 + c = 0) :
  a = -5/4 :=
by
  -- Use the given conditions to outline the proof (proof not provided here as per instruction)
  sorry

end quadratic_vertex_a_l65_65213


namespace ratio_of_part_diminished_by_4_l65_65546

theorem ratio_of_part_diminished_by_4 (N P : ℕ) (h1 : N = 160)
    (h2 : (1/5 : ℝ) * N + 4 = P - 4) : (P - 4) / N = 9 / 40 := 
by
  sorry

end ratio_of_part_diminished_by_4_l65_65546


namespace coord_sum_D_l65_65467

def is_midpoint (M C D : ℝ × ℝ) := M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

theorem coord_sum_D (M C D : ℝ × ℝ) (h : is_midpoint M C D) (hM : M = (4, 6)) (hC : C = (10, 2)) :
  D.1 + D.2 = 8 :=
sorry

end coord_sum_D_l65_65467


namespace evaluate_expression_l65_65259

theorem evaluate_expression :
  12 - 5 * 3^2 + 8 / 2 - 7 + 4^2 = -20 :=
by
  sorry

end evaluate_expression_l65_65259


namespace double_inequality_l65_65179

variable (a b c : ℝ)

theorem double_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b ≤ 1) (hbc : b + c ≤ 1) (hca : c + a ≤ 1) :
  a^2 + b^2 + c^2 ≤ a + b + c - a * b - b * c - c * a ∧ 
  a + b + c - a * b - b * c - c * a ≤ 1 / 2 * (1 + a^2 + b^2 + c^2) := 
sorry

end double_inequality_l65_65179


namespace suff_not_nec_cond_l65_65260

theorem suff_not_nec_cond (a : ℝ) : (a > 6 → a^2 > 36) ∧ (a^2 > 36 → (a > 6 ∨ a < -6)) := by
  sorry

end suff_not_nec_cond_l65_65260


namespace plumber_fix_cost_toilet_l65_65911

noncomputable def fixCost_Sink : ℕ := 30
noncomputable def fixCost_Shower : ℕ := 40

theorem plumber_fix_cost_toilet
  (T : ℕ)
  (Earnings1 : ℕ := 3 * T + 3 * fixCost_Sink)
  (Earnings2 : ℕ := 2 * T + 5 * fixCost_Sink)
  (Earnings3 : ℕ := T + 2 * fixCost_Shower + 3 * fixCost_Sink)
  (MaxEarnings : ℕ := 250) :
  Earnings2 = MaxEarnings → T = 50 :=
by
  sorry

end plumber_fix_cost_toilet_l65_65911


namespace multiply_eq_four_l65_65881

variables (a b c d : ℝ)

theorem multiply_eq_four (h1 : a = d) 
                         (h2 : b = c) 
                         (h3 : d + d = c * d) 
                         (h4 : b = d) 
                         (h5 : d + d = d * d) 
                         (h6 : c = 3) :
                         a * b = 4 := 
by 
  sorry

end multiply_eq_four_l65_65881


namespace problem1_problem2_l65_65265

-- Problem 1: Lean 4 Statement
theorem problem1 (n : ℕ) (hn : n > 0) : 20 ∣ (4 * 6^n + 5^(n + 1) - 9) :=
sorry

-- Problem 2: Lean 4 Statement
theorem problem2 : (3^100 % 7) = 4 :=
sorry

end problem1_problem2_l65_65265


namespace sum_of_integers_between_neg20_5_and_10_5_l65_65457

theorem sum_of_integers_between_neg20_5_and_10_5 :
  let a := -20
  let l := 10
  let n := (l - a) / 1 + 1
  let S := n / 2 * (a + l)
  S = -155 := by
{
  sorry
}

end sum_of_integers_between_neg20_5_and_10_5_l65_65457


namespace max_value_of_f_l65_65708

variable (n : ℕ)

-- Define the quadratic function with coefficients a, b, and c.
noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
axiom f_n : ∃ a b c, f n a b c = 6
axiom f_n1 : ∃ a b c, f (n + 1) a b c = 14
axiom f_n2 : ∃ a b c, f (n + 2) a b c = 14

-- The main goal is to prove the maximum value of f(x) is 15.
theorem max_value_of_f : ∃ a b c, (∀ x : ℝ, f x a b c ≤ 15) :=
by
  sorry

end max_value_of_f_l65_65708


namespace sufficient_but_not_necessary_for_ax_square_pos_l65_65760

variables (a x : ℝ)

theorem sufficient_but_not_necessary_for_ax_square_pos (h : a > 0) : 
  (a > 0 → ax^2 > 0) ∧ ((ax^2 > 0) → a > 0) :=
sorry

end sufficient_but_not_necessary_for_ax_square_pos_l65_65760


namespace relay_team_order_count_l65_65004

theorem relay_team_order_count :
  ∃ (orders : ℕ), orders = 6 :=
by
  let team_members := 4
  let remaining_members := team_members - 1  -- Excluding Lisa
  let first_lap_choices := remaining_members.choose 3  -- Choices for the first lap
  let third_lap_choices := (remaining_members - 1).choose 2  -- Choices for the third lap
  let fourth_lap_choices := (remaining_members - 2).choose 1  -- The last remaining member choices
  have orders := first_lap_choices * third_lap_choices * fourth_lap_choices
  use orders
  sorry

end relay_team_order_count_l65_65004


namespace infinite_double_perfect_squares_l65_65689

def is_double_number (n : ℕ) : Prop :=
  ∃ k m : ℕ, m > 0 ∧ n = m * 10^k + m

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem infinite_double_perfect_squares : ∀ n : ℕ, ∃ m, n < m ∧ is_double_number m ∧ is_perfect_square m :=
  sorry

end infinite_double_perfect_squares_l65_65689


namespace cells_count_after_9_days_l65_65556

theorem cells_count_after_9_days :
  let a := 5
  let r := 3
  let n := 3
  a * r^(n-1) = 45 :=
by
  let a := 5
  let r := 3
  let n := 3
  sorry

end cells_count_after_9_days_l65_65556


namespace max_and_min_A_l65_65588

noncomputable def B := {B : ℕ // B > 22222222 ∧ gcd B 18 = 1}
noncomputable def A (B : B) : ℕ := 10^8 * ((B.val % 10)) + (B.val / 10)

noncomputable def A_max := 999999998
noncomputable def A_min := 122222224

theorem max_and_min_A : 
  (∃ B : B, A B = A_max) ∧ (∃ B : B, A B = A_min) := sorry

end max_and_min_A_l65_65588


namespace largest_integral_ratio_l65_65625

theorem largest_integral_ratio (P A : ℕ) (rel_prime_sides : ∃ (a b c : ℕ), gcd a b = 1 ∧ gcd b c = 1 ∧ gcd c a = 1 ∧ a^2 + b^2 = c^2 ∧ P = a + b + c ∧ A = a * b / 2) :
  (∃ (k : ℕ), k = 45 ∧ ∀ l, l < 45 → l ≠ (P^2 / A)) :=
sorry

end largest_integral_ratio_l65_65625


namespace find_n_l65_65007

def x := 3
def y := 1
def n := x - 3 * y^(x - y) + 1

theorem find_n : n = 1 :=
by
  unfold n x y
  sorry

end find_n_l65_65007


namespace range_of_a_l65_65684

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, ¬ (x^2 - 2 * x + 3 ≤ a^2 - 2 * a - 1)) ↔ (-1 < a ∧ a < 3) :=
sorry

end range_of_a_l65_65684


namespace total_students_l65_65604

theorem total_students (N : ℕ)
  (h1 : (84 + 128 + 13 = 15 * N))
  : N = 15 :=
sorry

end total_students_l65_65604


namespace ratio_A_BC_1_to_4_l65_65109

/-
We will define the conditions and prove the ratio.
-/

def A := 20
def total := 100

-- defining the conditions
variables (B C : ℝ)
def condition1 := A + B + C = total
def condition2 := B = 3 / 5 * (A + C)

-- the theorem to prove
theorem ratio_A_BC_1_to_4 (h1 : condition1 B C) (h2 : condition2 B C) : A / (B + C) = 1 / 4 :=
by
  sorry

end ratio_A_BC_1_to_4_l65_65109


namespace initial_volume_of_mixture_l65_65527

theorem initial_volume_of_mixture (p q : ℕ) (x : ℕ) (h_ratio1 : p = 5 * x) (h_ratio2 : q = 3 * x) (h_added : q + 15 = 6 * x) (h_new_ratio : 5 * (3 * x + 15) = 6 * 5 * x) : 
  p + q = 40 :=
by
  sorry

end initial_volume_of_mixture_l65_65527


namespace average_weight_of_16_boys_l65_65578

theorem average_weight_of_16_boys :
  ∃ A : ℝ,
    (16 * A + 8 * 45.15 = 24 * 48.55) ∧
    A = 50.25 :=
by {
  -- Proof skipped, using sorry to denote the proof is required.
  sorry
}

end average_weight_of_16_boys_l65_65578


namespace number_of_teams_l65_65015

theorem number_of_teams (n : ℕ) (h1 : ∀ k, k = 10) (h2 : n * 10 * (n - 1) / 2 = 1900) : n = 20 :=
by
  sorry

end number_of_teams_l65_65015


namespace add_decimals_l65_65658

theorem add_decimals : 5.763 + 2.489 = 8.252 := 
by
  sorry

end add_decimals_l65_65658


namespace complement_union_intersection_l65_65133

open Set

def A : Set ℝ := {x | 3 ≤ x ∧ x < 5}
def B : Set ℝ := {x | 2 < x ∧ x < 9}

theorem complement_union_intersection :
  (compl (A ∪ B) = {x | x ≤ 2 ∨ 9 ≤ x}) ∧
  (compl (A ∩ B) = {x | x < 3 ∨ 5 ≤ x}) :=
by
  sorry

end complement_union_intersection_l65_65133


namespace smallest_k_sum_of_squares_multiple_of_200_l65_65571

-- Define the sum of squares for positive integer k
def sum_of_squares (k : ℕ) : ℕ := (k * (k + 1) * (2 * k + 1)) / 6

-- Prove that the sum of squares for k = 112 is a multiple of 200
theorem smallest_k_sum_of_squares_multiple_of_200 :
  ∃ k : ℕ, sum_of_squares k = sum_of_squares 112 ∧ 200 ∣ sum_of_squares 112 :=
sorry

end smallest_k_sum_of_squares_multiple_of_200_l65_65571


namespace problem_statement_l65_65141

theorem problem_statement (x y : ℝ) : (x * y < 18) → (x < 2 ∨ y < 9) :=
sorry

end problem_statement_l65_65141


namespace odd_number_as_difference_of_squares_l65_65719

theorem odd_number_as_difference_of_squares (n : ℤ) (h : ∃ k : ℤ, n = 2 * k + 1) :
  ∃ a b : ℤ, n = a^2 - b^2 :=
by
  sorry

end odd_number_as_difference_of_squares_l65_65719


namespace clock_strikes_l65_65986

theorem clock_strikes (t n : ℕ) (h_t : 13 * t = 26) (h_n : 2 * n - 1 * t = 22) : n = 6 :=
by
  sorry

end clock_strikes_l65_65986


namespace sum_of_cubes_l65_65768

theorem sum_of_cubes (a b c : ℝ) (h1 : a + b + c = 3) (h2 : ab + ac + bc = 5) (h3 : abc = -6) : a^3 + b^3 + c^3 = -36 :=
sorry

end sum_of_cubes_l65_65768


namespace jellybean_proof_l65_65762

def number_vanilla_jellybeans : ℕ := 120

def number_grape_jellybeans (V : ℕ) : ℕ := 5 * V + 50

def number_strawberry_jellybeans (V : ℕ) : ℕ := (2 * V) / 3

def total_number_jellybeans (V G S : ℕ) : ℕ := V + G + S

def cost_per_vanilla_jellybean : ℚ := 0.05

def cost_per_grape_jellybean : ℚ := 0.08

def cost_per_strawberry_jellybean : ℚ := 0.07

def total_cost_jellybeans (V G S : ℕ) : ℚ := 
  (cost_per_vanilla_jellybean * V) + 
  (cost_per_grape_jellybean * G) + 
  (cost_per_strawberry_jellybean * S)

theorem jellybean_proof :
  ∃ (V G S : ℕ), 
    V = number_vanilla_jellybeans ∧
    G = number_grape_jellybeans V ∧
    S = number_strawberry_jellybeans V ∧
    total_number_jellybeans V G S = 850 ∧
    total_cost_jellybeans V G S = 63.60 :=
by
  sorry

end jellybean_proof_l65_65762


namespace exists_fraction_equal_to_d_minus_1_l65_65536

theorem exists_fraction_equal_to_d_minus_1 (n d : ℕ) (hdiv : d > 0 ∧ n % d = 0) :
  ∃ k : ℕ, k < n ∧ (n - k) / (n - (n - k)) = d - 1 :=
by
  sorry

end exists_fraction_equal_to_d_minus_1_l65_65536


namespace john_draw_on_back_l65_65885

theorem john_draw_on_back (total_pictures front_pictures : ℕ) (h1 : total_pictures = 15) (h2 : front_pictures = 6) : total_pictures - front_pictures = 9 :=
  by
  sorry

end john_draw_on_back_l65_65885


namespace expand_expression_l65_65993

theorem expand_expression (x y : ℝ) : 24 * (3 * x - 4 * y + 6) = 72 * x - 96 * y + 144 := 
by
  sorry

end expand_expression_l65_65993


namespace monotonically_decreasing_iff_l65_65569

noncomputable def f (a x : ℝ) : ℝ := (x^2 - 2 * a * x) * Real.exp x

theorem monotonically_decreasing_iff (a : ℝ) : (∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ f a (-1) ∧ f a x ≤ f a 1) ↔ (a ≥ 3 / 4) :=
by
  sorry

end monotonically_decreasing_iff_l65_65569


namespace least_repeating_block_of_8_over_11_l65_65751

theorem least_repeating_block_of_8_over_11 : (∃ n : ℕ, (∀ m : ℕ, m < n → ¬(∃ a b : ℤ, (10^m - 1) * (8 * 10^n - b * 11 * 10^(n - t)) = a * 11 * 10^(m - t))) ∧ n ≤ 2) :=
by
  sorry

end least_repeating_block_of_8_over_11_l65_65751


namespace second_player_wins_l65_65714

-- Piles of balls and game conditions
def two_pile_game (pile1 pile2 : ℕ) : Prop :=
  ∀ (player1_turn : ℕ → Prop) (player2_turn : ℕ → Prop),
    (∀ n : ℕ, player1_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player1's move
    (∀ n : ℕ, player2_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player2's move
    -- - Second player has a winning strategy
    ∃ (win_strategy : ℕ → ℕ), ∀ k : ℕ, player1_turn k → player2_turn (win_strategy k) 

-- Lean statement of the problem
theorem second_player_wins : ∀ (pile1 pile2 : ℕ), pile1 = 30 ∧ pile2 = 30 → two_pile_game pile1 pile2 :=
  by
    intros pile1 pile2 h
    sorry  -- Placeholder for the proof


end second_player_wins_l65_65714


namespace parallel_lines_condition_l65_65250

theorem parallel_lines_condition (a : ℝ) :
  ( ∀ x y : ℝ, (a * x + 2 * y + 2 = 0 → ∃ C₁ : ℝ, x - 2 * y = C₁) 
  ∧ (x + (a - 1) * y + 1 = 0 → ∃ C₂ : ℝ, x - 2 * y = C₂) )
  ↔ a = -1 :=
sorry

end parallel_lines_condition_l65_65250


namespace tan_alpha_tan_beta_l65_65792

theorem tan_alpha_tan_beta (α β : ℝ) (h1 : Real.cos (α + β) = 3 / 5) (h2 : Real.cos (α - β) = 4 / 5) :
  Real.tan α * Real.tan β = 1 / 7 := by
  sorry

end tan_alpha_tan_beta_l65_65792


namespace solve_for_x_l65_65296

theorem solve_for_x (x : ℝ) : 
  (x - 35) / 3 = (3 * x + 10) / 8 → x = -310 := by
  sorry

end solve_for_x_l65_65296


namespace women_per_table_l65_65186

theorem women_per_table 
  (total_tables : ℕ)
  (men_per_table : ℕ)
  (total_customers : ℕ) 
  (h_total_tables : total_tables = 6)
  (h_men_per_table : men_per_table = 5)
  (h_total_customers : total_customers = 48) :
  (total_customers - (men_per_table * total_tables)) / total_tables = 3 :=
by
  subst h_total_tables
  subst h_men_per_table
  subst h_total_customers
  sorry

end women_per_table_l65_65186


namespace sin_cos_identity_l65_65366

variable {α : ℝ}

/-- Given 1 / sin(α) + 1 / cos(α) = √3, then sin(α) * cos(α) = -1 / 3 -/
theorem sin_cos_identity (h : 1 / Real.sin α + 1 / Real.cos α = Real.sqrt 3) : 
  Real.sin α * Real.cos α = -1 / 3 := 
sorry

end sin_cos_identity_l65_65366


namespace set_range_of_three_numbers_l65_65054

theorem set_range_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : (a + b + c) / 3 = 6) 
(h4 : b = 6) (h5 : c = 10) : c - a = 8 := by
  sorry

end set_range_of_three_numbers_l65_65054


namespace infinite_slips_have_repeated_numbers_l65_65342

theorem infinite_slips_have_repeated_numbers
  (slips : Set ℕ) (h_inf_slips : slips.Infinite)
  (h_sub_infinite_imp_repeats : ∀ s : Set ℕ, s.Infinite → ∃ x ∈ s, ∃ y ∈ s, x ≠ y ∧ x = y) :
  ∃ n : ℕ, {x ∈ slips | x = n}.Infinite :=
by sorry

end infinite_slips_have_repeated_numbers_l65_65342


namespace determine_max_weight_l65_65812

theorem determine_max_weight {a b : ℕ} (n : ℕ) (x : ℕ) (ha : a > 0) (hb : b > 0) (hx : 1 ≤ x ∧ x ≤ n) :
  n = 9 :=
sorry

end determine_max_weight_l65_65812


namespace minEmployees_correct_l65_65059

noncomputable def minEmployees (seaTurtles birdMigration bothTurtlesBirds turtlesPlants allThree : ℕ) : ℕ :=
  let onlySeaTurtles := seaTurtles - (bothTurtlesBirds + turtlesPlants - allThree)
  let onlyBirdMigration := birdMigration - (bothTurtlesBirds + allThree - turtlesPlants)
  onlySeaTurtles + onlyBirdMigration + bothTurtlesBirds + turtlesPlants + allThree

theorem minEmployees_correct :
  minEmployees 120 90 30 50 15 = 245 := by
  sorry

end minEmployees_correct_l65_65059


namespace first_folder_number_l65_65247

theorem first_folder_number (stickers : ℕ) (folders : ℕ) : stickers = 999 ∧ folders = 369 → 100 = 100 :=
by sorry

end first_folder_number_l65_65247


namespace fermat_prime_solution_unique_l65_65542

def is_fermat_prime (p : ℕ) : Prop :=
  ∃ r : ℕ, p = 2^(2^r) + 1

def problem_statement (p n k : ℕ) : Prop :=
  is_fermat_prime p ∧ p^n + n = (n + 1)^k

theorem fermat_prime_solution_unique (p n k : ℕ) :
  problem_statement p n k → (p, n, k) = (3, 1, 2) ∨ (p, n, k) = (5, 2, 3) :=
by
  sorry

end fermat_prime_solution_unique_l65_65542


namespace lake_width_l65_65623

theorem lake_width
  (W : ℝ)
  (janet_speed : ℝ) (sister_speed : ℝ) (wait_time : ℝ)
  (h1 : janet_speed = 30)
  (h2 : sister_speed = 12)
  (h3 : wait_time = 3)
  (h4 : W / sister_speed = W / janet_speed + wait_time) :
  W = 60 := 
sorry

end lake_width_l65_65623


namespace division_result_l65_65745

theorem division_result (a b : ℕ) (ha : a = 7) (hb : b = 3) :
    ((a^3 + b^3) / (a^2 - a * b + b^2) = 10) := 
by
  sorry

end division_result_l65_65745


namespace kendall_total_change_l65_65550

-- Definition of values of coins
def value_of_quarters (q : ℕ) : ℝ := q * 0.25
def value_of_dimes (d : ℕ) : ℝ := d * 0.10
def value_of_nickels (n : ℕ) : ℝ := n * 0.05

-- Conditions
def quarters := 10
def dimes := 12
def nickels := 6

-- Theorem statement
theorem kendall_total_change : 
  value_of_quarters quarters + value_of_dimes dimes + value_of_nickels nickels = 4.00 :=
by
  sorry

end kendall_total_change_l65_65550


namespace find_F_l65_65620

theorem find_F (F C : ℝ) (hC_eq : C = (4/7) * (F - 40)) (hC_val : C = 35) : F = 101.25 :=
by
  sorry

end find_F_l65_65620


namespace general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l65_65870

-- Definitions based on conditions
def a (n : ℕ) : ℕ := 2 * n + 1

def b (n : ℕ) : ℕ := 2 ^ (a n)

noncomputable def S (n : ℕ) : ℕ := (n * (2 * n + 2)) / 2

noncomputable def T (n : ℕ) : ℕ := (8 * (4 ^ n - 1)) / 3

-- Statements to be proved
theorem general_formula_an : ∀ n : ℕ, a n = 2 * n + 1 := sorry

theorem geometric_sequence_bn : ∀ n : ℕ, b n = 2 ^ (2 * n + 1) := sorry

theorem sum_of_geometric_sequence_Tn : ∀ n : ℕ, T n = (8 * (4 ^ n - 1)) / 3 := sorry

end general_formula_an_geometric_sequence_bn_sum_of_geometric_sequence_Tn_l65_65870


namespace min_dist_circle_to_line_l65_65976

noncomputable def circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y

noncomputable def line_eq (x y : ℝ) := x + y - 8

theorem min_dist_circle_to_line : 
  (∀ x y : ℝ, circle_eq x y = 0 → ∃ d : ℝ, d ≥ 0 ∧ 
    (∀ x₁ y₁ : ℝ, circle_eq x₁ y₁ = 0 → ∀ x₂ y₂ : ℝ, line_eq x₂ y₂ = 0 → d ≤ dist (x₁, y₁) (x₂, y₂)) ∧ 
    d = 2 * Real.sqrt 2) :=
by
  sorry

end min_dist_circle_to_line_l65_65976


namespace remainder_avg_is_correct_l65_65319

-- Definitions based on the conditions
variables (total_avg : ℝ) (first_part_avg : ℝ) (second_part_avg : ℝ) (first_part_percent : ℝ) (second_part_percent : ℝ)

-- The conditions stated mathematically
def overall_avg_contribution 
  (remainder_avg : ℝ) : Prop :=
  first_part_percent * first_part_avg + 
  second_part_percent * second_part_avg + 
  (1 - first_part_percent - second_part_percent) * remainder_avg =  total_avg
  
-- The question
theorem remainder_avg_is_correct : overall_avg_contribution 75 80 65 0.25 0.50 90 := sorry

end remainder_avg_is_correct_l65_65319


namespace complement_of_angle_l65_65358

theorem complement_of_angle (α : ℝ) (h : α = 23 + 36 / 60) : 180 - α = 156.4 := 
by
  sorry

end complement_of_angle_l65_65358


namespace snow_probability_at_least_once_l65_65937

theorem snow_probability_at_least_once :
  let p := 3 / 4
  let prob_no_snow_single_day := 1 - p
  let prob_no_snow_all_days := prob_no_snow_single_day ^ 5
  let prob_snow_at_least_once := 1 - prob_no_snow_all_days
  prob_snow_at_least_once = 1023 / 1024 :=
by
  sorry

end snow_probability_at_least_once_l65_65937


namespace lily_milk_amount_l65_65858

def initial_milk : ℚ := 5
def milk_given_to_james : ℚ := 18 / 4
def milk_received_from_neighbor : ℚ := 7 / 4

theorem lily_milk_amount : (initial_milk - milk_given_to_james + milk_received_from_neighbor) = 9 / 4 :=
by
  sorry

end lily_milk_amount_l65_65858


namespace find_face_value_l65_65947

-- Define the conditions as variables in Lean
variable (BD TD FV : ℝ)
variable (hBD : BD = 36)
variable (hTD : TD = 30)
variable (hRel : BD = TD + (TD * BD / FV))

-- State the theorem we want to prove
theorem find_face_value (BD TD : ℝ) (FV : ℝ) 
  (hBD : BD = 36) (hTD : TD = 30) (hRel : BD = TD + (TD * BD / FV)) : 
  FV = 180 := 
  sorry

end find_face_value_l65_65947


namespace polynomial_equivalence_l65_65451

theorem polynomial_equivalence (x y : ℝ) (h : y = x + 1/x) :
  (x^2 * (y^2 + 2*y - 5) = 0) ↔ (x^4 + 2*x^3 - 3*x^2 + 2*x + 1 = 0) :=
by
  sorry

end polynomial_equivalence_l65_65451


namespace kevin_food_spending_l65_65423

theorem kevin_food_spending :
  let total_budget := 20
  let samuel_ticket := 14
  let samuel_food_and_drinks := 6
  let kevin_drinks := 2
  let kevin_food_and_drinks := total_budget - (samuel_ticket + samuel_food_and_drinks) - kevin_drinks
  kevin_food_and_drinks = 4 :=
by
  sorry

end kevin_food_spending_l65_65423


namespace max_triangles_in_graph_l65_65652

def points : Finset Point := sorry
def no_coplanar (points : Finset Point) : Prop := sorry
def no_tetrahedron (points : Finset Point) : Prop := sorry
def triangles (points : Finset Point) : ℕ := sorry

theorem max_triangles_in_graph (points : Finset Point) 
  (H1 : points.card = 9) 
  (H2 : no_coplanar points) 
  (H3 : no_tetrahedron points) : 
  triangles points ≤ 27 := 
sorry

end max_triangles_in_graph_l65_65652


namespace find_AG_l65_65509

theorem find_AG (AE CE BD CD AB AG : ℝ) (h1 : AE = 3)
    (h2 : CE = 1) (h3 : BD = 2) (h4 : CD = 2) (h5 : AB = 5) :
    AG = (3 * Real.sqrt 66) / 7 :=
  sorry

end find_AG_l65_65509


namespace radius_of_inscribed_sphere_l65_65045

theorem radius_of_inscribed_sphere (a b c s : ℝ)
  (h1: 2 * (a * b + a * c + b * c) = 616)
  (h2: a + b + c = 40)
  : s = Real.sqrt 246 ↔ (2 * s) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 :=
by
  sorry

end radius_of_inscribed_sphere_l65_65045


namespace found_bottle_caps_is_correct_l65_65308

def initial_bottle_caps : ℕ := 6
def total_bottle_caps : ℕ := 28

theorem found_bottle_caps_is_correct : total_bottle_caps - initial_bottle_caps = 22 := by
  sorry

end found_bottle_caps_is_correct_l65_65308


namespace calculate_loss_percentage_l65_65863

theorem calculate_loss_percentage
  (CP SP₁ SP₂ : ℝ)
  (h₁ : SP₁ = CP * 1.05)
  (h₂ : SP₂ = 1140) :
  (CP = 1200) → (SP₁ = 1260) → ((CP - SP₂) / CP * 100 = 5) :=
by
  intros h1 h2
  -- Here, we will eventually provide the actual proof steps.
  sorry

end calculate_loss_percentage_l65_65863


namespace chocolates_difference_l65_65927

theorem chocolates_difference (robert_chocolates : ℕ) (nickel_chocolates : ℕ)
  (h1 : robert_chocolates = 7) (h2 : nickel_chocolates = 3) :
  robert_chocolates - nickel_chocolates = 4 :=
by
  sorry

end chocolates_difference_l65_65927


namespace cos_triple_angle_l65_65551

theorem cos_triple_angle (θ : ℝ) (h : Real.cos θ = 3 / 5) : Real.cos (3 * θ) = -117 / 125 :=
by
  sorry

end cos_triple_angle_l65_65551


namespace monotonic_m_range_l65_65207

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x - 12

-- Prove the range of m where f(x) is monotonic on [m, m+4]
theorem monotonic_m_range {m : ℝ} :
  (∀ x y : ℝ, m ≤ x ∧ x ≤ m + 4 ∧ m ≤ y ∧ y ≤ m + 4 → (x ≤ y → f x ≤ f y ∨ f x ≥ f y))
  ↔ (m ≤ -5 ∨ m ≥ 2) :=
sorry

end monotonic_m_range_l65_65207


namespace arcsin_arccos_eq_l65_65886

theorem arcsin_arccos_eq (x : ℝ) (h : Real.arcsin x + Real.arcsin (2 * x - 1) = Real.arccos x) : x = 1 := by
  sorry

end arcsin_arccos_eq_l65_65886


namespace find_chord_eq_l65_65879

-- Given conditions 
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def point_p : (ℝ × ℝ) := (3, 2)
def midpoint_chord (p1 p2 p : (ℝ × ℝ)) : Prop := p.fst = (p1.fst + p2.fst) / 2 ∧ p.snd = (p1.snd + p2.snd) / 2

-- Conditions in Lean definition
def conditions (x1 y1 x2 y2 : ℝ) : Prop :=
  ellipse_eq x1 y1 ∧ ellipse_eq x2 y2 ∧ midpoint_chord (x1,y1) (x2,y2) point_p

-- The statement to prove
theorem find_chord_eq (x1 y1 x2 y2 : ℝ) (h : conditions x1 y1 x2 y2) :
  ∃ m b : ℝ, (m = -2 / 3) ∧ b = 2 - m * 3 ∧ (∀ x y : ℝ, y = m * x + b → 2 * x + 3 * y - 12 = 0) :=
by {
  sorry
}

end find_chord_eq_l65_65879


namespace correct_answer_l65_65572

variables (x y : ℝ)

def cost_equations (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 120) ∧ (2 * x - y = 20)

theorem correct_answer : cost_equations x y :=
sorry

end correct_answer_l65_65572


namespace find_a_plus_2b_l65_65738

open Real

theorem find_a_plus_2b 
  (a b : ℝ) 
  (ha : 0 < a ∧ a < π / 2) 
  (hb : 0 < b ∧ b < π / 2) 
  (h1 : 4 * (sin a)^2 + 3 * (sin b)^2 = 1) 
  (h2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0) :
  a + 2 * b = π / 2 :=
sorry

end find_a_plus_2b_l65_65738


namespace range_of_a_no_real_roots_l65_65304

theorem range_of_a_no_real_roots (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 + ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
by
  sorry

end range_of_a_no_real_roots_l65_65304


namespace river_current_speed_l65_65902

theorem river_current_speed 
  (downstream_distance upstream_distance still_water_speed : ℝ)
  (H1 : still_water_speed = 20)
  (H2 : downstream_distance = 100)
  (H3 : upstream_distance = 60)
  (H4 : (downstream_distance / (still_water_speed + x)) = (upstream_distance / (still_water_speed - x)))
  : x = 5 :=
by
  sorry

end river_current_speed_l65_65902


namespace find_other_number_l65_65190

theorem find_other_number (B : ℕ) (hcf_cond : Nat.gcd 36 B = 14) (lcm_cond : Nat.lcm 36 B = 396) : B = 66 :=
sorry

end find_other_number_l65_65190


namespace center_circle_sum_l65_65939

theorem center_circle_sum (h k : ℝ) :
  (∃ h k : ℝ, h + k = 6 ∧ ∃ R, (x - h)^2 + (y - k)^2 = R^2) ↔ ∃ h k : ℝ, h = 3 ∧ k = 3 ∧ h + k = 6 := 
by
  sorry

end center_circle_sum_l65_65939


namespace geometric_sequence_b_value_l65_65194

theorem geometric_sequence_b_value 
  (b : ℝ)
  (h1 : b > 0)
  (h2 : ∃ r : ℝ, 160 * r = b ∧ b * r = 1)
  : b = 4 * Real.sqrt 10 := 
sorry

end geometric_sequence_b_value_l65_65194


namespace security_to_bag_ratio_l65_65794

noncomputable def U_house : ℕ := 10
noncomputable def U_airport : ℕ := 5 * U_house
noncomputable def C_bag : ℕ := 15
noncomputable def W_boarding : ℕ := 20
noncomputable def W_takeoff : ℕ := 2 * W_boarding
noncomputable def T_total : ℕ := 180
noncomputable def T_known : ℕ := U_house + U_airport + C_bag + W_boarding + W_takeoff
noncomputable def T_security : ℕ := T_total - T_known

theorem security_to_bag_ratio : T_security / C_bag = 3 :=
by sorry

end security_to_bag_ratio_l65_65794


namespace solution_of_abs_square_inequality_l65_65085

def solution_set := {x : ℝ | (1 ≤ x ∧ x ≤ 3) ∨ x = -2}

theorem solution_of_abs_square_inequality (x : ℝ) :
  (abs (x^2 - 4) ≤ x + 2) ↔ (x ∈ solution_set) :=
by
  sorry

end solution_of_abs_square_inequality_l65_65085


namespace non_deg_ellipse_b_l65_65945

theorem non_deg_ellipse_b (b : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 - 6*x + 27*y = b ∧ (∀ x y : ℝ, (x - 3)^2 + 9*(y + 3/2)^2 = b + 145/4)) → b > -145/4 :=
sorry

end non_deg_ellipse_b_l65_65945


namespace find_side_c_and_area_S_find_sinA_plus_cosB_l65_65828

-- Definitions for the conditions given
structure Triangle :=
  (a b c : ℝ)
  (angleA angleB angleC : ℝ)

noncomputable def givenTriangle : Triangle :=
  { a := 2, b := 4, c := 2 * Real.sqrt 3, angleA := 30, angleB := 90, angleC := 60 }

-- Prove the length of side c and the area S
theorem find_side_c_and_area_S (t : Triangle) (h : t = givenTriangle) :
  t.c = 2 * Real.sqrt 3 ∧ (1 / 2) * t.a * t.b * Real.sin (t.angleC * Real.pi / 180) = 2 * Real.sqrt 3 :=
by
  sorry

-- Prove the value of sin A + cos B
theorem find_sinA_plus_cosB (t : Triangle) (h : t = givenTriangle) :
  Real.sin (t.angleA * Real.pi / 180) + Real.cos (t.angleB * Real.pi / 180) = 1 / 2 :=
by
  sorry

end find_side_c_and_area_S_find_sinA_plus_cosB_l65_65828


namespace function_value_at_2018_l65_65696

theorem function_value_at_2018 (f : ℝ → ℝ)
  (h1 : f 4 = 2 - Real.sqrt 3)
  (h2 : ∀ x, f (x + 2) = 1 / (- f x)) :
  f 2018 = -2 - Real.sqrt 3 :=
by
  sorry

end function_value_at_2018_l65_65696


namespace min_value_frac_sum_l65_65309

open Real

theorem min_value_frac_sum (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 2 * a + b = 1) :
  1 / a + 2 / b = 8 :=
sorry

end min_value_frac_sum_l65_65309


namespace area_of_trapezium_l65_65856

-- Definitions for the given conditions
def parallel_side_a : ℝ := 18  -- in cm
def parallel_side_b : ℝ := 20  -- in cm
def distance_between_sides : ℝ := 5  -- in cm

-- Statement to prove the area is 95 cm²
theorem area_of_trapezium : 
  let a := parallel_side_a
  let b := parallel_side_b
  let h := distance_between_sides
  (1 / 2 * (a + b) * h = 95) :=
by
  sorry  -- Proof is not required here

end area_of_trapezium_l65_65856


namespace sum_of_digits_of_x_l65_65648

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem sum_of_digits_of_x (x : ℕ) (h1 : 100 ≤ x) (h2 : x ≤ 949)
  (h3 : is_palindrome x) (h4 : is_palindrome (x + 50)) :
  sum_of_digits x = 19 :=
sorry

end sum_of_digits_of_x_l65_65648


namespace range_of_a_l65_65345

variable (a : ℝ)

def p : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ + 1 = 0

def q : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - 2 * a * x + a^2 + 1 ≥ 1

theorem range_of_a : ¬(p a ∨ q a) → -2 < a ∧ a < 0 := by
  sorry

end range_of_a_l65_65345


namespace rectangle_side_lengths_l65_65835

variables (x y m n S : ℝ) (hx_y_ratio : x / y = m / n) (hxy_area : x * y = S)

theorem rectangle_side_lengths :
  x = Real.sqrt (m * S / n) ∧ y = Real.sqrt (n * S / m) :=
sorry

end rectangle_side_lengths_l65_65835


namespace groups_needed_for_sampling_l65_65010

def total_students : ℕ := 600
def sample_size : ℕ := 20

theorem groups_needed_for_sampling : (total_students / sample_size = 30) :=
by
  sorry

end groups_needed_for_sampling_l65_65010


namespace least_pos_int_with_12_pos_factors_is_72_l65_65361

def least_positive_integer_with_12_factors (n : ℕ) : Prop :=
  ∃ (m : ℕ), m > 0 ∧ m ≠ n ∧ (∀ d : ℕ, d ∣ m → (d = n ∨ d = 1)) ∧
  ((∀ d : ℕ, d ∣ m → ∃ e : ℕ, e = 1) → n = 72)

theorem least_pos_int_with_12_pos_factors_is_72 (n : ℕ) :
  least_positive_integer_with_12_factors n → n = 72 := by
  sorry

end least_pos_int_with_12_pos_factors_is_72_l65_65361


namespace increasing_interval_of_f_l65_65649

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 4)

theorem increasing_interval_of_f :
  ∀ x, x ∈ Set.Icc (-3 * Real.pi / 4) (Real.pi / 4) → MonotoneOn f (Set.Icc (-3 * Real.pi / 4) (Real.pi / 4)) :=
by
  sorry

end increasing_interval_of_f_l65_65649


namespace amount_of_juice_p_in_a_l65_65122

  def total_p : ℚ := 24
  def total_v : ℚ := 25
  def ratio_a : ℚ := 4 / 1
  def ratio_y : ℚ := 1 / 5

  theorem amount_of_juice_p_in_a :
    ∃ P_a : ℚ, ∃ V_a : ℚ, ∃ P_y : ℚ, ∃ V_y : ℚ,
      P_a / V_a = ratio_a ∧ P_y / V_y = ratio_y ∧
      P_a + P_y = total_p ∧ V_a + V_y = total_v ∧ P_a = 20 :=
  by
    sorry
  
end amount_of_juice_p_in_a_l65_65122


namespace polygon_sides_l65_65266

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 1080) : n = 8 :=
sorry

end polygon_sides_l65_65266


namespace restaurant_sales_l65_65777

theorem restaurant_sales (monday tuesday wednesday thursday : ℕ) 
  (h1 : monday = 40) 
  (h2 : tuesday = monday + 40) 
  (h3 : wednesday = tuesday / 2) 
  (h4 : monday + tuesday + wednesday + thursday = 203) : 
  thursday = wednesday + 3 := 
by sorry

end restaurant_sales_l65_65777


namespace circle_radius_doubling_l65_65951

theorem circle_radius_doubling (r : ℝ) : 
  let new_radius := 2 * r
  let original_circumference := 2 * Real.pi * r
  let new_circumference := 2 * Real.pi * new_radius
  let original_area := Real.pi * r^2
  let new_area := Real.pi * (new_radius)^2
  (new_circumference = 2 * original_circumference) ∧ (new_area = 4 * original_area) :=
by
  let new_radius := 2 * r
  let original_circumference := 2 * Real.pi * r
  let new_circumference := 2 * Real.pi * new_radius
  let original_area := Real.pi * r^2
  let new_area := Real.pi * (new_radius)^2
  have hc : new_circumference = 2 * original_circumference := by
    sorry
  have ha : new_area = 4 * original_area := by
    sorry
  exact ⟨hc, ha⟩

end circle_radius_doubling_l65_65951


namespace center_of_circle_is_correct_l65_65433

-- Define the conditions as Lean functions and statements
def is_tangent (x y : ℝ) : Prop :=
  (3 * x + 4 * y = 48) ∨ (3 * x + 4 * y = -12)

def is_on_line (x y : ℝ) : Prop := x = y

-- Define the proof statement
theorem center_of_circle_is_correct (x y : ℝ) (h1 : is_tangent x y) (h2 : is_on_line x y) :
  (x, y) = (18 / 7, 18 / 7) :=
sorry

end center_of_circle_is_correct_l65_65433


namespace marble_cut_percentage_l65_65135

theorem marble_cut_percentage
  (initial_weight : ℝ)
  (final_weight : ℝ)
  (x : ℝ)
  (first_week_cut : ℝ)
  (second_week_cut : ℝ)
  (third_week_cut : ℝ) :
  initial_weight = 190 →
  final_weight = 109.0125 →
  first_week_cut = (1 - x / 100) →
  second_week_cut = 0.85 →
  third_week_cut = 0.9 →
  (initial_weight * first_week_cut * second_week_cut * third_week_cut = final_weight) →
  x = 24.95 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end marble_cut_percentage_l65_65135


namespace shells_total_l65_65780

variable (x y : ℝ)

theorem shells_total (h1 : y = x + (x + 32)) : y = 2 * x + 32 :=
sorry

end shells_total_l65_65780


namespace inscribed_circle_probability_l65_65209

theorem inscribed_circle_probability (r : ℝ) (h : r > 0) : 
  let square_area := 4 * r^2
  let circle_area := π * r^2
  (circle_area / square_area) = π / 4 := by
  sorry

end inscribed_circle_probability_l65_65209


namespace find_m_l65_65503

theorem find_m (m : ℝ) : (Real.tan (20 * Real.pi / 180) + m * Real.sin (20 * Real.pi / 180) = Real.sqrt 3) → m = 4 :=
by
  sorry

end find_m_l65_65503


namespace least_integer_greater_than_sqrt_500_l65_65534

theorem least_integer_greater_than_sqrt_500 : 
  ∃ n : ℤ, (∀ m : ℤ, m * m ≤ 500 → m < n) ∧ n = 23 :=
by
  sorry

end least_integer_greater_than_sqrt_500_l65_65534


namespace value_2_std_dev_less_than_mean_l65_65767

-- Define the mean and standard deviation as constants
def mean : ℝ := 14.5
def std_dev : ℝ := 1.5

-- State the theorem (problem)
theorem value_2_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.5 := by
  sorry

end value_2_std_dev_less_than_mean_l65_65767


namespace martin_rings_big_bell_l65_65580

/-
Problem Statement:
Martin rings the small bell 4 times more than 1/3 as often as the big bell.
If he rings both of them a combined total of 52 times, prove that he rings the big bell 36 times.
-/

theorem martin_rings_big_bell (s b : ℕ) 
  (h1 : s + b = 52) 
  (h2 : s = 4 + (1 / 3 : ℚ) * b) : 
  b = 36 := 
by
  sorry

end martin_rings_big_bell_l65_65580


namespace minimize_product_of_roots_of_quadratic_eq_l65_65730

theorem minimize_product_of_roots_of_quadratic_eq (k : ℝ) :
  (∃ x y : ℝ, 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0) 
  → k = 25 / 8 :=
sorry

end minimize_product_of_roots_of_quadratic_eq_l65_65730


namespace students_taking_geometry_or_science_but_not_both_l65_65963

def students_taking_both : ℕ := 15
def students_taking_geometry : ℕ := 30
def students_taking_science_only : ℕ := 18

theorem students_taking_geometry_or_science_but_not_both : students_taking_geometry - students_taking_both + students_taking_science_only = 33 := by
  sorry

end students_taking_geometry_or_science_but_not_both_l65_65963


namespace statement1_statement2_statement3_statement4_statement5_statement6_l65_65918

/-
Correct syntax statements in pseudo code
-/

def correct_assignment1 (A B : ℤ) : Prop :=
  B = A ∧ A = 50

def correct_assignment2 (x y z : ℕ) : Prop :=
  x = 1 ∧ y = 2 ∧ z = 3

def correct_input1 (s : String) (x : ℕ) : Prop :=
  s = "How old are you?" ∧ x ≥ 0

def correct_input2 (x : ℕ) : Prop :=
  x ≥ 0

def correct_print1 (s1 : String) (C : ℤ) : Prop :=
  s1 = "A+B=" ∧ C < 100  -- additional arbitrary condition for C

def correct_print2 (s2 : String) : Prop :=
  s2 = "Good-bye!"

theorem statement1 (A : ℤ) : ∃ B, correct_assignment1 A B :=
sorry

theorem statement2 : ∃ (x y z : ℕ), correct_assignment2 x y z :=
sorry

theorem statement3 (x : ℕ) : ∃ s, correct_input1 s x :=
sorry

theorem statement4 (x : ℕ) : correct_input2 x :=
sorry

theorem statement5 (C : ℤ) : ∃ s1, correct_print1 s1 C :=
sorry

theorem statement6 : ∃ s2, correct_print2 s2 :=
sorry

end statement1_statement2_statement3_statement4_statement5_statement6_l65_65918


namespace point_in_second_quadrant_range_l65_65875

theorem point_in_second_quadrant_range (m : ℝ) :
  (m - 3 < 0 ∧ m + 1 > 0) ↔ (-1 < m ∧ m < 3) :=
by
  sorry

end point_in_second_quadrant_range_l65_65875


namespace ab_is_square_l65_65297

theorem ab_is_square (a b c : ℕ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) 
  (h_main : a + b = b * (a - c)) (h_prime : ∃ p : ℕ, Prime p ∧ c + 1 = p^2) :
  ∃ k : ℕ, a + b = k^2 :=
by
  sorry

end ab_is_square_l65_65297


namespace plant_lamp_arrangements_l65_65380

/-- Rachel has two identical basil plants and an aloe plant.
Additionally, she has two identical white lamps, two identical red lamps, and 
two identical blue lamps she can put each plant under 
(she can put more than one plant under a lamp, but each plant is under exactly one lamp). 
-/
theorem plant_lamp_arrangements : 
  let plants := ["basil", "basil", "aloe"]
  let lamps := ["white", "white", "red", "red", "blue", "blue"]
  ∃ n, n = 27 := by
  sorry

end plant_lamp_arrangements_l65_65380


namespace simplify_fraction_l65_65210

theorem simplify_fraction (a b c : ℕ) (h1 : a = 222) (h2 : b = 8888) (h3 : c = 44) : 
  (a : ℚ) / b * c = 111 / 101 := 
by 
  sorry

end simplify_fraction_l65_65210


namespace function_additive_of_tangential_property_l65_65283

open Set

variable {f : ℝ → ℝ}

def is_tangential_quadrilateral_sides (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ (a + c = b + d)

theorem function_additive_of_tangential_property
  (h : ∀ (a b c d : ℝ), is_tangential_quadrilateral_sides a b c d → f (a + b + c + d) = f a + f b + f c + f d) :
  ∀ (x y : ℝ), 0 < x → 0 < y → f (x + y) = f x + f y :=
by
  sorry

end function_additive_of_tangential_property_l65_65283


namespace second_day_hike_ratio_l65_65753

theorem second_day_hike_ratio (full_hike_distance first_day_distance third_day_distance : ℕ) 
(h_full_hike: full_hike_distance = 50)
(h_first_day: first_day_distance = 10)
(h_third_day: third_day_distance = 15) : 
(full_hike_distance - (first_day_distance + third_day_distance)) / full_hike_distance = 1 / 2 := by
  sorry

end second_day_hike_ratio_l65_65753


namespace set_theorem_l65_65633

noncomputable def set_A : Set ℕ := {1, 2}
noncomputable def set_B : Set ℕ := {1, 2, 3}
noncomputable def set_C : Set ℕ := {2, 3, 4}

theorem set_theorem : (set_A ∩ set_B) ∪ set_C = {1, 2, 3, 4} := by
  sorry

end set_theorem_l65_65633


namespace Igor_colored_all_cells_l65_65406

theorem Igor_colored_all_cells (m n : ℕ) (h1 : 9 * m = 12 * n) (h2 : 0 < m ∧ m ≤ 4) (h3 : 0 < n ∧ n ≤ 3) :
  m = 4 ∧ n = 3 :=
by {
  sorry
}

end Igor_colored_all_cells_l65_65406


namespace total_fish_l65_65968

theorem total_fish (goldfish bluefish : ℕ) (h1 : goldfish = 15) (h2 : bluefish = 7) : goldfish + bluefish = 22 := 
by
  sorry

end total_fish_l65_65968


namespace coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l65_65729

def coprime_distinct_remainders (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : Prop :=
  ∃ (a : Fin m → ℤ) (b : Fin k → ℤ),
    (∀ (i : Fin m) (j : Fin k), ∀ (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k))

def not_coprime_congruent_product (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : Prop :=
  ∀ (a : Fin m → ℤ) (b : Fin k → ℤ),
    ∃ (i : Fin m) (j : Fin k) (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a s * b t) % (m * k)

-- Example statement to assert the existence of the above properties
theorem coprime_mk_has_distinct_products 
  (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : coprime_distinct_remainders m k coprime_mk :=
sorry

theorem not_coprime_mk_has_congruent_products 
  (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : not_coprime_congruent_product m k not_coprime_mk :=
sorry

end coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l65_65729


namespace div_simplify_l65_65746

theorem div_simplify (a b : ℝ) (h : a ≠ 0) : (8 * a * b) / (2 * a) = 4 * b :=
by
  sorry

end div_simplify_l65_65746


namespace jack_bill_age_difference_l65_65990

theorem jack_bill_age_difference :
  ∃ (a b : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ (7 * a - 29 * b = 14) ∧ ((10 * a + b) - (10 * b + a) = 36) :=
by
  sorry

end jack_bill_age_difference_l65_65990


namespace max_possible_value_of_C_l65_65904

theorem max_possible_value_of_C (A B C D : ℕ) (h₁ : A + B + C + D = 200) (h₂ : A + B = 70) (h₃ : 0 < A) (h₄ : 0 < B) (h₅ : 0 < C) (h₆ : 0 < D) :
  C ≤ 129 :=
by
  sorry

end max_possible_value_of_C_l65_65904


namespace intersection_points_on_circle_l65_65630

theorem intersection_points_on_circle
  (x y : ℝ)
  (h1 : y = (x + 2)^2)
  (h2 : x + 2 = (y - 1)^2) :
  (x + 2)^2 + (y - 1)^2 = 2 :=
sorry

end intersection_points_on_circle_l65_65630


namespace ratio_of_pens_to_pencils_l65_65305

-- Define the conditions
def total_items : ℕ := 13
def pencils : ℕ := 4
def eraser : ℕ := 1
def pens : ℕ := total_items - pencils - eraser

-- Prove the ratio of pens to pencils is 2:1
theorem ratio_of_pens_to_pencils : pens = 2 * pencils :=
by
  -- indicate that the proof is omitted
  sorry

end ratio_of_pens_to_pencils_l65_65305


namespace Jamie_correct_percentage_l65_65935

theorem Jamie_correct_percentage (y : ℕ) : ((8 * y - 2 * y : ℕ) / (8 * y : ℕ) : ℚ) * 100 = 75 := by
  sorry

end Jamie_correct_percentage_l65_65935


namespace find_x_l65_65686

structure Point where
  x : ℝ
  y : ℝ

def vector (P Q : Point) : Point :=
  ⟨Q.x - P.x, Q.y - P.y⟩

def parallel (v w : Point) : Prop :=
  v.x * w.y = v.y * w.x

theorem find_x (A B C : Point) (hA : A = ⟨0, -3⟩) (hB : B = ⟨3, 3⟩) (hC : C = ⟨x, -1⟩) (h_parallel : parallel (vector A B) (vector A C)) : x = 1 := 
by
  sorry

end find_x_l65_65686


namespace trigonometric_identity_l65_65831

theorem trigonometric_identity :
  (Real.sin (18 * Real.pi / 180) * Real.sin (78 * Real.pi / 180)) -
  (Real.cos (162 * Real.pi / 180) * Real.cos (78 * Real.pi / 180)) = 1 / 2 := by
  sorry

end trigonometric_identity_l65_65831


namespace four_star_three_l65_65118

def star (a b : ℕ) : ℕ := a^2 - a * b + b^2 + 2 * a * b

theorem four_star_three : star 4 3 = 37 :=
by
  -- here we would normally provide the proof steps
  sorry

end four_star_three_l65_65118


namespace solve_for_2a_plus_b_l65_65075

variable (a b : ℝ)

theorem solve_for_2a_plus_b (h1 : 4 * a ^ 2 - b ^ 2 = 12) (h2 : 2 * a - b = 4) : 2 * a + b = 3 := 
by
  sorry

end solve_for_2a_plus_b_l65_65075


namespace original_population_l65_65142

-- Define the initial setup
variable (P : ℝ)

-- The conditions given in the problem
axiom ten_percent_died (P : ℝ) : (1 - 0.1) * P = 0.9 * P
axiom twenty_percent_left (P : ℝ) : (1 - 0.2) * (0.9 * P) = 0.9 * P * 0.8

-- Define the final condition
axiom final_population (P : ℝ) : 0.9 * P * 0.8 = 3240

-- The proof problem
theorem original_population : P = 4500 :=
by
  sorry

end original_population_l65_65142


namespace find_q_l65_65212

variable (p q : ℝ)
variable (h1 : 1 < p)
variable (h2 : p < q)
variable (h3 : 1 / p + 1 / q = 1)
variable (h4 : p * q = 8)

theorem find_q : q = 4 + 2 * Real.sqrt 2 :=
by
  sorry

end find_q_l65_65212


namespace number_of_students_l65_65977

theorem number_of_students 
  (P S : ℝ)
  (total_cost : ℝ) 
  (percent_free : ℝ) 
  (lunch_cost : ℝ)
  (h1 : percent_free = 0.40)
  (h2 : total_cost = 210)
  (h3 : lunch_cost = 7)
  (h4 : P = 0.60 * S)
  (h5 : P * lunch_cost = total_cost) :
  S = 50 :=
by
  sorry

end number_of_students_l65_65977


namespace geometric_seq_a9_l65_65610

theorem geometric_seq_a9 
  (a : ℕ → ℤ)  -- The sequence definition
  (h_geometric : ∀ n : ℕ, a (n+1) = a 1 * (a 2 ^ n) / a 1 ^ n)  -- Geometric sequence property
  (h_a1 : a 1 = 2)  -- Given a₁ = 2
  (h_a5 : a 5 = 18)  -- Given a₅ = 18
: a 9 = 162 := sorry

end geometric_seq_a9_l65_65610


namespace rate_of_stream_l65_65435

theorem rate_of_stream (x : ℝ) (h1 : ∀ (distance : ℝ), (24 : ℝ) > 0) (h2 : ∀ (distance : ℝ), (distance / (24 - x)) = 3 * (distance / (24 + x))) : x = 12 :=
by
  sorry

end rate_of_stream_l65_65435


namespace burgers_ordered_l65_65950

theorem burgers_ordered (H : ℕ) (Ht : H + 2 * H = 45) : 2 * H = 30 := by
  sorry

end burgers_ordered_l65_65950


namespace line_ellipse_intersect_l65_65147

theorem line_ellipse_intersect (m k : ℝ) (h₀ : ∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ x^2 / 5 + y^2 / m = 1) : m ≥ 1 ∧ m ≠ 5 :=
sorry

end line_ellipse_intersect_l65_65147


namespace terminating_decimals_nat_l65_65637

theorem terminating_decimals_nat (n : ℕ) (h1 : ∃ a b : ℕ, n = 2^a * 5^b)
  (h2 : ∃ c d : ℕ, n + 1 = 2^c * 5^d) : n = 1 ∨ n = 4 :=
by
  sorry

end terminating_decimals_nat_l65_65637


namespace frac_eval_eq_l65_65389

theorem frac_eval_eq :
  let a := 19
  let b := 8
  let c := 35
  let d := 19 * 8 / 35
  ( (⌈a / b - ⌈c / d⌉⌉) / ⌈c / b + ⌈d⌉⌉) = (1 / 10) := by
  sorry

end frac_eval_eq_l65_65389


namespace new_car_travel_distance_l65_65673

-- Define the distance traveled by the older car
def distance_older_car : ℝ := 150

-- Define the percentage increase
def percentage_increase : ℝ := 0.30

-- Define the condition for the newer car's travel distance
def distance_newer_car (d_old : ℝ) (perc_inc : ℝ) : ℝ :=
  d_old * (1 + perc_inc)

-- Prove the main statement
theorem new_car_travel_distance :
  distance_newer_car distance_older_car percentage_increase = 195 := by
  -- Skip the proof body as instructed
  sorry

end new_car_travel_distance_l65_65673


namespace max_value_of_expression_l65_65276

theorem max_value_of_expression (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) : 
  (∃ x : ℝ, x = 3 → 
    ∀ A : ℝ, A = (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80 * (a * b * c)^(4/3)) → 
      A ≤ x) :=
by
  sorry

end max_value_of_expression_l65_65276


namespace tournament_ranking_sequences_l65_65807

def total_fair_ranking_sequences (A B C D : Type) : Nat :=
  let saturday_outcomes := 2
  let sunday_outcomes := 4 -- 2 possibilities for (first, second) and 2 for (third, fourth)
  let tiebreaker_effect := 2 -- swap second and third
  saturday_outcomes * sunday_outcomes * tiebreaker_effect

theorem tournament_ranking_sequences (A B C D : Type) :
  total_fair_ranking_sequences A B C D = 32 := 
by
  sorry

end tournament_ranking_sequences_l65_65807


namespace simplify_expression_l65_65576

theorem simplify_expression :
  1 + 1 / (1 + 1 / (2 + 2)) = 9 / 5 := by
  sorry

end simplify_expression_l65_65576


namespace no_integer_points_between_A_and_B_on_line_l65_65803

theorem no_integer_points_between_A_and_B_on_line
  (A : ℕ × ℕ) (B : ℕ × ℕ)
  (hA : A = (2, 3))
  (hB : B = (50, 500)) :
  ∀ (P : ℕ × ℕ), P.1 > 2 ∧ P.1 < 50 ∧ 
    (P.2 * 48 - P.1 * 497 = 2 * 497 - 3 * 48) →
    false := 
by
  sorry

end no_integer_points_between_A_and_B_on_line_l65_65803


namespace greatest_power_of_2_factor_of_expr_l65_65938

theorem greatest_power_of_2_factor_of_expr :
  (∃ k, 2 ^ k ∣ 12 ^ 600 - 8 ^ 400 ∧ ∀ m, 2 ^ m ∣ 12 ^ 600 - 8 ^ 400 → m ≤ 1204) :=
sorry

end greatest_power_of_2_factor_of_expr_l65_65938


namespace range_of_m_l65_65781

open Real Set

variable (x m : ℝ)

def p (x : ℝ) := (x + 1) * (x - 1) ≤ 0
def q (x m : ℝ) := (x + 1) * (x - (3 * m - 1)) ≤ 0 ∧ m > 0

theorem range_of_m (hpsuffq : ∀ x, p x → q x m) (hqnotsuffp : ∃ x, q x m ∧ ¬ p x) : m > 2 / 3 := by
  sorry

end range_of_m_l65_65781


namespace base_3_is_most_economical_l65_65301

theorem base_3_is_most_economical (m d : ℕ) (h : d ≥ 1) (h_m_div_d : m % d = 0) :
  3^(m / 3) ≥ d^(m / d) :=
sorry

end base_3_is_most_economical_l65_65301


namespace moon_speed_conversion_l65_65498

theorem moon_speed_conversion :
  ∀ (moon_speed_kps : ℝ) (seconds_in_minute : ℕ) (minutes_in_hour : ℕ),
  moon_speed_kps = 0.9 →
  seconds_in_minute = 60 →
  minutes_in_hour = 60 →
  (moon_speed_kps * (seconds_in_minute * minutes_in_hour) = 3240) := by
  sorry

end moon_speed_conversion_l65_65498


namespace smallest_integer_divisible_l65_65197

theorem smallest_integer_divisible:
  ∃ n : ℕ, n > 1 ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 6 = 1) ∧ n = 61 :=
by
  sorry

end smallest_integer_divisible_l65_65197


namespace FruitKeptForNextWeek_l65_65866

/-- Define the variables and conditions -/
def total_fruit : ℕ := 10
def fruit_eaten : ℕ := 5
def fruit_brought_on_friday : ℕ := 3

/-- Define what we need to prove -/
theorem FruitKeptForNextWeek : 
  ∃ k, total_fruit - fruit_eaten - fruit_brought_on_friday = k ∧ k = 2 :=
by
  sorry

end FruitKeptForNextWeek_l65_65866


namespace math_problem_real_solution_l65_65381

theorem math_problem_real_solution (x y : ℝ) (h : x^2 * y^2 - x * y - x / y - y / x = 4) : 
  (x - 2) * (y - 2) = 3 - 2 * Real.sqrt 2 :=
sorry

end math_problem_real_solution_l65_65381


namespace find_x_l65_65137

theorem find_x (a b x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < x)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : 
    x = 16 * a^(3 / 2) :=
by 
  sorry

end find_x_l65_65137


namespace calc_expr_l65_65038

theorem calc_expr :
  (-1) * (-3) + 3^2 / (8 - 5) = 6 :=
by
  sorry

end calc_expr_l65_65038


namespace least_number_modular_l65_65234

theorem least_number_modular 
  (n : ℕ)
  (h1 : n % 34 = 4)
  (h2 : n % 48 = 6)
  (h3 : n % 5 = 2) : n = 4082 :=
by
  sorry

end least_number_modular_l65_65234


namespace vehicle_count_l65_65949

theorem vehicle_count (T B : ℕ) (h1 : T + B = 15) (h2 : 3 * T + 2 * B = 40) : T = 10 ∧ B = 5 :=
by
  sorry

end vehicle_count_l65_65949


namespace Bill_has_39_dollars_l65_65817

noncomputable def Frank_initial_money : ℕ := 42
noncomputable def pizza_cost : ℕ := 11
noncomputable def num_pizzas : ℕ := 3
noncomputable def Bill_initial_money : ℕ := 30

noncomputable def Frank_spent : ℕ := pizza_cost * num_pizzas
noncomputable def Frank_remaining_money : ℕ := Frank_initial_money - Frank_spent
noncomputable def Bill_final_money : ℕ := Bill_initial_money + Frank_remaining_money

theorem Bill_has_39_dollars :
  Bill_final_money = 39 :=
by
  sorry

end Bill_has_39_dollars_l65_65817


namespace area_of_inscribed_triangle_l65_65609

noncomputable def area_of_triangle_inscribed_in_circle_with_arcs (a b c : ℕ) := 
  let circum := a + b + c
  let r := circum / (2 * Real.pi)
  let θ := 360 / (a + b + c)
  let angle1 := 4 * θ
  let angle2 := 6 * θ
  let angle3 := 8 * θ
  let sin80 := Real.sin (80 * Real.pi / 180)
  let sin120 := Real.sin (120 * Real.pi / 180)
  let sin160 := Real.sin (160 * Real.pi / 180)
  let approx_vals := sin80 + sin120 + sin160
  (1 / 2) * r^2 * approx_vals

theorem area_of_inscribed_triangle : 
  area_of_triangle_inscribed_in_circle_with_arcs 4 6 8 = 90.33 / Real.pi^2 :=
by sorry

end area_of_inscribed_triangle_l65_65609


namespace a_3_equals_35_l65_65565

noncomputable def S (n : ℕ) : ℕ := 5 * n ^ 2 + 10 * n
noncomputable def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_3_equals_35 : a 3 = 35 := by
  sorry

end a_3_equals_35_l65_65565


namespace trees_in_garden_l65_65399

theorem trees_in_garden (yard_length : ℕ) (distance_between_trees : ℕ) (H1 : yard_length = 400) (H2 : distance_between_trees = 16) : 
  (yard_length / distance_between_trees) + 1 = 26 :=
by
  -- Adding sorry to skip the proof
  sorry

end trees_in_garden_l65_65399


namespace rain_at_least_once_l65_65196

theorem rain_at_least_once (p : ℚ) (h : p = 3/4) : 
    (1 - (1 - p)^4) = 255/256 :=
by
  sorry

end rain_at_least_once_l65_65196


namespace number_of_10_digit_integers_with_consecutive_twos_l65_65889

open Nat

-- Define the total number of 10-digit integers using only '1' and '2's
def total_10_digit_numbers : ℕ := 2^10

-- Define the Fibonacci function
def fibonacci : ℕ → ℕ
| 0    => 1
| 1    => 2
| n+2  => fibonacci (n+1) + fibonacci n

-- Calculate the 10th Fibonacci number for the problem context
def F_10 : ℕ := fibonacci 9 + fibonacci 8

-- Prove that the number of 10-digit integers with at least one pair of consecutive '2's is 880
theorem number_of_10_digit_integers_with_consecutive_twos :
  total_10_digit_numbers - F_10 = 880 :=
by
  sorry

end number_of_10_digit_integers_with_consecutive_twos_l65_65889


namespace no_two_right_angles_in_triangle_l65_65101

theorem no_two_right_angles_in_triangle 
  (α β γ : ℝ)
  (h1 : α + β + γ = 180) :
  ¬ (α = 90 ∧ β = 90) :=
by
  sorry

end no_two_right_angles_in_triangle_l65_65101


namespace solve_for_x_l65_65205

theorem solve_for_x (x : ℝ) (h : 3 * x + 15 = (1 / 3) * (4 * x + 28)) : 
  x = -17 / 5 := 
by 
  sorry

end solve_for_x_l65_65205


namespace modified_monotonous_count_l65_65801

def is_modified_monotonous (n : ℕ) : Prop :=
  -- Definition that determines if a number is modified-monotonous
  -- Must include digit '5', and digits must form a strictly increasing or decreasing sequence
  sorry 

def count_modified_monotonous (n : ℕ) : ℕ :=
  2 * (8 * (2^8) + 2^8) + 1 -- Formula for counting modified-monotonous numbers including '5'

theorem modified_monotonous_count : count_modified_monotonous 5 = 4609 := 
  by 
    sorry

end modified_monotonous_count_l65_65801


namespace calculate_total_cost_l65_65106

def total_cost (num_boxes : ℕ) (packs_per_box : ℕ) (tissues_per_pack : ℕ) (cost_per_tissue : ℝ) : ℝ :=
  num_boxes * packs_per_box * tissues_per_pack * cost_per_tissue

theorem calculate_total_cost :
  total_cost 10 20 100 0.05 = 1000 := 
by
  sorry

end calculate_total_cost_l65_65106


namespace power_div_ex_l65_65473

theorem power_div_ex (a b c : ℕ) (h1 : a = 2^4) (h2 : b = 2^3) (h3 : c = 2^2) :
  ((a^4) * (b^6)) / (c^12) = 1024 := 
sorry

end power_div_ex_l65_65473


namespace dad_strawberries_weight_proof_l65_65083

/-
Conditions:
1. total_weight (the combined weight of Marco's and his dad's strawberries) is 23 pounds.
2. marco_weight (the weight of Marco's strawberries) is 14 pounds.
We need to prove that dad_weight (the weight of dad's strawberries) is 9 pounds.
-/

def total_weight : ℕ := 23
def marco_weight : ℕ := 14

def dad_weight : ℕ := total_weight - marco_weight

theorem dad_strawberries_weight_proof : dad_weight = 9 := by
  sorry

end dad_strawberries_weight_proof_l65_65083


namespace remainder_123456789012_mod_252_l65_65168

theorem remainder_123456789012_mod_252 :
  let M := 123456789012 
  (M % 252) = 87 := by
  sorry

end remainder_123456789012_mod_252_l65_65168


namespace ellipse_area_quadrants_eq_zero_l65_65020

theorem ellipse_area_quadrants_eq_zero 
(E : Type)
(x y : E → ℝ) 
(h_ellipse : ∀ (x y : ℝ), (x - 19)^2 / (19 * 1998) + (y - 98)^2 / (98 * 1998) = 1998) 
(R1 R2 R3 R4 : ℝ)
(H1 : ∀ (R1 R2 R3 R4 : ℝ), R1 = R_ellipse / 4 ∧ R2 = R_ellipse / 4 ∧ R3 = R_ellipse / 4 ∧ R4 = R_ellipse / 4)
: R1 - R2 + R3 - R4 = 0 := 
by 
sorry

end ellipse_area_quadrants_eq_zero_l65_65020


namespace binomial_square_coefficients_l65_65338

noncomputable def a : ℝ := 13.5
noncomputable def b : ℝ := 18

theorem binomial_square_coefficients (c d : ℝ) :
  (∀ x : ℝ, 6 * x ^ 2 + 18 * x + a = (c * x + d) ^ 2) ∧ 
  (∀ x : ℝ, 3 * x ^ 2 + b * x + 4 = (c * x + d) ^ 2)  → 
  a = 13.5 ∧ b = 18 := sorry

end binomial_square_coefficients_l65_65338


namespace power_inequality_l65_65159

theorem power_inequality (n : ℕ) (x : ℝ) (h1 : 0 < n) (h2 : x > -1) : (1 + x)^n ≥ 1 + n * x :=
sorry

end power_inequality_l65_65159


namespace estimate_larger_than_difference_l65_65282

variable {x y : ℝ}

theorem estimate_larger_than_difference (h1 : x > y) (h2 : y > 0) :
    ⌈x⌉ - ⌊y⌋ > x - y := by
  sorry

end estimate_larger_than_difference_l65_65282


namespace inradius_semicircle_relation_l65_65124

theorem inradius_semicircle_relation 
  (a b c : ℝ)
  (h_acute: a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2)
  (S : ℝ)
  (p : ℝ)
  (r : ℝ)
  (ra rb rc : ℝ)
  (h_def_semi_perim : p = (a + b + c) / 2)
  (h_area : S = p * r)
  (h_ra : ra = (2 * S) / (b + c))
  (h_rb : rb = (2 * S) / (a + c))
  (h_rc : rc = (2 * S) / (a + b)) :
  2 / r = 1 / ra + 1 / rb + 1 / rc :=
by
  sorry

end inradius_semicircle_relation_l65_65124


namespace inequality_proof_l65_65200

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l65_65200


namespace teacher_already_graded_worksheets_l65_65704

-- Define the conditions
def num_worksheets : ℕ := 9
def problems_per_worksheet : ℕ := 4
def remaining_problems : ℕ := 16
def total_problems := num_worksheets * problems_per_worksheet

-- Define the required proof
theorem teacher_already_graded_worksheets :
  (total_problems - remaining_problems) / problems_per_worksheet = 5 :=
by sorry

end teacher_already_graded_worksheets_l65_65704


namespace number_of_lines_l65_65224

-- Define point P
structure Point where
  x : ℝ
  y : ℝ

-- Define a line by its equation ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ
  nonzero: a ≠ 0 ∨ b ≠ 0

-- Definition of a line passing through a point P
def passes_through (l : Line) (P : Point) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Definition of a line having equal intercepts on x-axis and y-axis
def equal_intercepts (l : Line) : Prop :=
  l.a ≠ 0 ∧ l.b ≠ 0 ∧ l.a = l.b

-- Definition of a specific point P
def P : Point := { x := 1, y := 2 }

-- The theorem statement
theorem number_of_lines : ∃ (lines : Finset Line), (∀ l ∈ lines, passes_through l P ∧ equal_intercepts l) ∧ lines.card = 2 := by
  sorry

end number_of_lines_l65_65224


namespace ratio_of_areas_of_similar_triangles_l65_65506

theorem ratio_of_areas_of_similar_triangles (m1 m2 : ℝ) (med_ratio : m1 / m2 = 1 / Real.sqrt 2) :
    let area_ratio := (m1 / m2) ^ 2
    area_ratio = 1 / 2 := by
  sorry

end ratio_of_areas_of_similar_triangles_l65_65506


namespace condition_neither_sufficient_nor_necessary_l65_65876
-- Import necessary library

-- Define the function and conditions
def f (x a : ℝ) : ℝ := x^2 + a * x + 1

-- State the proof problem
theorem condition_neither_sufficient_nor_necessary :
  ∀ a : ℝ, (∀ x : ℝ, f x a = 0 -> x = 1/2) ↔ a^2 - 4 = 0 ∧ a ≤ -2 := sorry

end condition_neither_sufficient_nor_necessary_l65_65876


namespace geometric_sequence_S6_l65_65237

noncomputable def sum_of_first_n_terms (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_S6 (a r : ℝ) (h1 : sum_of_first_n_terms a r 2 = 6) (h2 : sum_of_first_n_terms a r 4 = 30) : 
  sum_of_first_n_terms a r 6 = 126 :=
sorry

end geometric_sequence_S6_l65_65237


namespace smallest_c_over_a_plus_b_l65_65470

theorem smallest_c_over_a_plus_b (a b c : ℝ) (h : a^2 + b^2 = c^2) :
  ∃ d : ℝ, d = (c / (a + b)) ∧ d = (Real.sqrt 2 / 2) :=
by
  sorry

end smallest_c_over_a_plus_b_l65_65470


namespace trig_identity_l65_65264

theorem trig_identity :
  (Real.cos (105 * Real.pi / 180) * Real.cos (45 * Real.pi / 180) + Real.sin (45 * Real.pi / 180) * Real.sin (105 * Real.pi / 180)) = 1 / 2 :=
  sorry

end trig_identity_l65_65264


namespace max_rect_area_with_given_perimeter_l65_65612

-- Define the variables used in the problem
def length_of_wire := 12
def max_area (x : ℝ) := -(x - 3)^2 + 9

-- Lean Statement for the problem
theorem max_rect_area_with_given_perimeter : ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 6 → (x * (6 - x) ≤ A)) ∧ A = 9 :=
by
  sorry

end max_rect_area_with_given_perimeter_l65_65612


namespace avg_speed_4_2_l65_65587

noncomputable def avg_speed_round_trip (D : ℝ) : ℝ :=
  let speed_up := 3
  let speed_down := 7
  let total_distance := 2 * D
  let total_time := D / speed_up + D / speed_down
  total_distance / total_time

theorem avg_speed_4_2 (D : ℝ) (hD : D > 0) : avg_speed_round_trip D = 4.2 := by
  sorry

end avg_speed_4_2_l65_65587


namespace prime_p_perfect_cube_l65_65312

theorem prime_p_perfect_cube (p : ℕ) (hp : Nat.Prime p) (h : ∃ n : ℕ, 13 * p + 1 = n^3) :
  p = 2 ∨ p = 211 :=
by
  sorry

end prime_p_perfect_cube_l65_65312


namespace division_remainder_l65_65995

theorem division_remainder :
  ∃ (R D Q : ℕ), D = 3 * Q ∧ D = 3 * R + 3 ∧ 251 = D * Q + R ∧ R = 8 := by
  sorry

end division_remainder_l65_65995


namespace complex_number_proof_l65_65638

open Complex

noncomputable def problem_complex (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) : ℂ :=
  (z - 1) * (z^2 - 1) * (z^3 - 1) * (z^4 - 1) * (z^5 - 1) * (z^6 - 1)

theorem complex_number_proof (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) :
  problem_complex z h1 h2 = 8 :=
  sorry

end complex_number_proof_l65_65638


namespace range_of_m_l65_65385

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 → ((m^2 - m) * 4^x - 2^x < 0)) → (-1 < m ∧ m < 2) :=
by
  sorry

end range_of_m_l65_65385


namespace unique_solution_condition_l65_65472

theorem unique_solution_condition (c d : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + c = d * x + 2) ↔ d ≠ 4 :=
by
  sorry

end unique_solution_condition_l65_65472


namespace football_players_count_l65_65786

-- Define the given conditions
def total_students : ℕ := 39
def long_tennis_players : ℕ := 20
def both_sports : ℕ := 17
def play_neither : ℕ := 10

-- Define a theorem to prove the number of football players is 26
theorem football_players_count : 
  ∃ (F : ℕ), F = 26 ∧ 
  (total_students - play_neither) = (F - both_sports) + (long_tennis_players - both_sports) + both_sports :=
by {
  sorry
}

end football_players_count_l65_65786


namespace shaded_area_l65_65606

theorem shaded_area (area_large : ℝ) (area_small : ℝ) (n_small_squares : ℕ) 
  (n_triangles: ℕ) (area_total : ℝ) : 
  area_large = 16 → 
  area_small = 1 → 
  n_small_squares = 4 → 
  n_triangles = 4 → 
  area_total = 4 → 
  4 * area_small = 4 →
  area_large - (area_total + (n_small_squares * area_small)) = 4 :=
by
  intros
  sorry

end shaded_area_l65_65606


namespace vegetarian_count_l65_65535

variables (v_only v_nboth vegan pesc nvboth : ℕ)
variables (hv_only : v_only = 13) (hv_nboth : v_nboth = 8)
          (hvegan_tot : vegan = 5) (hvegan_v : vveg1 = 3)
          (hpesc_tot : pesc = 4) (hpesc_vnboth : nvboth = 2)

theorem vegetarian_count (total_veg : ℕ) 
  (H_total : total_veg = v_only + v_nboth + (vegan - vveg1)) :
  total_veg = 23 :=
sorry

end vegetarian_count_l65_65535


namespace carrots_picked_next_day_l65_65573

-- Definitions based on conditions
def initial_carrots : Nat := 48
def carrots_thrown_away : Nat := 45
def total_carrots_next_day : Nat := 45

-- The proof problem statement
theorem carrots_picked_next_day : 
  (initial_carrots - carrots_thrown_away + x = total_carrots_next_day) → (x = 42) :=
by 
  sorry

end carrots_picked_next_day_l65_65573


namespace left_seats_equals_15_l65_65178

variable (L : ℕ)

noncomputable def num_seats_left (L : ℕ) : Prop :=
  ∃ L, 3 * L + 3 * (L - 3) + 8 = 89

theorem left_seats_equals_15 : num_seats_left L → L = 15 :=
by
  intro h
  sorry

end left_seats_equals_15_l65_65178


namespace find_m_l65_65151

theorem find_m (x1 x2 m : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 + 3 * x2 = 5) : m = 7 / 4 :=
  sorry

end find_m_l65_65151


namespace not_divisible_by_pow_two_l65_65353

theorem not_divisible_by_pow_two (n : ℕ) (h : n > 1) : ¬ (2^n ∣ (3^n + 1)) :=
by
  sorry

end not_divisible_by_pow_two_l65_65353


namespace no_daily_coverage_l65_65288

theorem no_daily_coverage (ranks : Nat → Nat)
  (h_ranks_ordered : ∀ i, ranks (i+1) ≥ 3 * ranks i)
  (h_cycle : ∀ i, ∃ N : Nat, ranks i = N ∧ ∃ k : Nat, k = N ∧ ∀ m, m % (2 * N) < N → (¬ ∃ j, ranks j ≤ N))
  : ¬ (∀ d : Nat, ∃ j : Nat, (∃ k : Nat, d % (2 * (ranks j)) < ranks j))
  := sorry

end no_daily_coverage_l65_65288


namespace distance_proof_l65_65447

-- Definitions from the conditions
def avg_speed_to_retreat := 50
def avg_speed_back_home := 75
def total_round_trip_time := 10
def distance_between_home_and_retreat := 300

-- Theorem stating the problem
theorem distance_proof 
  (D : ℝ)
  (h1 : D / avg_speed_to_retreat + D / avg_speed_back_home = total_round_trip_time) :
  D = distance_between_home_and_retreat :=
sorry

end distance_proof_l65_65447


namespace max_value_y_on_interval_l65_65772

noncomputable def y (x: ℝ) : ℝ := x^4 - 8 * x^2 + 2

theorem max_value_y_on_interval : 
  ∃ x ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y x = 11 ∧ ∀ z ∈ Set.Icc (-1 : ℝ) (3 : ℝ), y z ≤ 11 := 
sorry

end max_value_y_on_interval_l65_65772


namespace marble_weight_l65_65163

-- Define the weights of marbles and waffle irons
variables (m w : ℝ)

-- Given conditions
def condition1 : Prop := 9 * m = 4 * w
def condition2 : Prop := 3 * w = 75 

-- The theorem we want to prove
theorem marble_weight (h1 : condition1 m w) (h2 : condition2 w) : m = 100 / 9 :=
by
  sorry

end marble_weight_l65_65163


namespace largest_constant_c_l65_65218

theorem largest_constant_c :
  ∃ c : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x^2 + y^2 = 1 → x^6 + y^6 ≥ c * x * y) ∧ c = 1 / 2 :=
sorry

end largest_constant_c_l65_65218


namespace find_a_b_sum_l65_65146

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 6 * x - 6

theorem find_a_b_sum (a b : ℝ)
  (h1 : f a = 1)
  (h2 : f b = -5) :
  a + b = 2 :=
  sorry

end find_a_b_sum_l65_65146


namespace Diana_additional_video_game_time_l65_65822

theorem Diana_additional_video_game_time 
    (original_reward_per_hour : ℕ := 30)
    (raise_percentage : ℕ := 20)
    (hours_read : ℕ := 12)
    (minutes_per_hour : ℕ := 60) :
    let raise := (raise_percentage * original_reward_per_hour) / 100
    let new_reward_per_hour := original_reward_per_hour + raise
    let total_time_after_raise := new_reward_per_hour * hours_read
    let total_time_before_raise := original_reward_per_hour * hours_read
    let additional_minutes := total_time_after_raise - total_time_before_raise
    additional_minutes = 72 :=
by sorry

end Diana_additional_video_game_time_l65_65822


namespace Randy_drew_pictures_l65_65980

variable (P Q R: ℕ)

def Peter_drew_pictures (P : ℕ) : Prop := P = 8
def Quincy_drew_pictures (Q P : ℕ) : Prop := Q = P + 20
def Total_drawing (R P Q : ℕ) : Prop := R + P + Q = 41

theorem Randy_drew_pictures
  (P_eq : Peter_drew_pictures P)
  (Q_eq : Quincy_drew_pictures Q P)
  (Total_eq : Total_drawing R P Q) :
  R = 5 :=
by 
  sorry

end Randy_drew_pictures_l65_65980


namespace expand_expression_l65_65174

theorem expand_expression :
  (3 * t^2 - 2 * t + 3) * (-2 * t^2 + 3 * t - 4) = -6 * t^4 + 13 * t^3 - 24 * t^2 + 17 * t - 12 :=
by sorry

end expand_expression_l65_65174


namespace jane_donuts_l65_65965

def croissant_cost := 60
def donut_cost := 90
def days := 6

theorem jane_donuts (c d k : ℤ) 
  (h1 : c + d = days)
  (h2 : donut_cost * d + croissant_cost * c = 100 * k + 50) :
  d = 3 :=
sorry

end jane_donuts_l65_65965


namespace additional_chair_frequency_l65_65679

theorem additional_chair_frequency 
  (workers : ℕ)
  (chairs_per_worker_per_hour : ℕ)
  (hours : ℕ)
  (total_chairs : ℕ) 
  (additional_chairs_rate : ℕ)
  (h_workers : workers = 3) 
  (h_chairs_per_worker : chairs_per_worker_per_hour = 4) 
  (h_hours : hours = 6 ) 
  (h_total_chairs : total_chairs = 73) :
  additional_chairs_rate = 6 :=
by
  sorry

end additional_chair_frequency_l65_65679


namespace yellow_pill_cost_22_5_l65_65591

-- Definitions based on conditions
def number_of_days := 3 * 7
def total_cost := 903
def daily_cost := total_cost / number_of_days
def blue_pill_cost (yellow_pill_cost : ℝ) := yellow_pill_cost - 2

-- Prove that the cost of one yellow pill is 22.5 dollars
theorem yellow_pill_cost_22_5 : 
  ∃ (yellow_pill_cost : ℝ), 
    number_of_days = 21 ∧
    total_cost = 903 ∧ 
    (∀ yellow_pill_cost, daily_cost = yellow_pill_cost + blue_pill_cost yellow_pill_cost → yellow_pill_cost = 22.5) :=
by 
  sorry

end yellow_pill_cost_22_5_l65_65591


namespace polygon_properties_l65_65248

theorem polygon_properties
    (each_exterior_angle : ℝ)
    (h1 : each_exterior_angle = 24) :
    ∃ n : ℕ, n = 15 ∧ (180 * (n - 2) = 2340) :=
  by
    sorry

end polygon_properties_l65_65248


namespace arithmetic_sequence_sum_mul_three_eq_3480_l65_65427

theorem arithmetic_sequence_sum_mul_three_eq_3480 :
  let a := 50
  let d := 3
  let l := 95
  let n := ((l - a) / d + 1 : ℕ)
  let sum := n * (a + l) / 2
  3 * sum = 3480 := by
  sorry

end arithmetic_sequence_sum_mul_three_eq_3480_l65_65427


namespace solve_fiftieth_term_l65_65386

variable (a₇ a₂₁ : ℤ) (d : ℚ)

-- The conditions stated in the problem
def seventh_term : a₇ = 10 := by sorry
def twenty_first_term : a₂₁ = 34 := by sorry

-- The fifty term calculation assuming the common difference d
def fiftieth_term_is_fraction (d : ℚ) : ℚ := 10 + 43 * d

-- Translate the condition a₂₁ = a₇ + 14 * d
theorem solve_fiftieth_term : a₂₁ = a₇ + 14 * d → 
                              fiftieth_term_is_fraction d = 682 / 7 := by sorry


end solve_fiftieth_term_l65_65386


namespace fourth_person_height_l65_65450

theorem fourth_person_height (h : ℝ)
  (h2 : h + 2 = h₂)
  (h3 : h + 4 = h₃)
  (h4 : h + 10 = h₄)
  (average_height : (h + h₂ + h₃ + h₄) / 4 = 77) :
  h₄ = 83 :=
by
  sorry

end fourth_person_height_l65_65450


namespace certain_event_abs_nonneg_l65_65061

theorem certain_event_abs_nonneg (x : ℝ) : |x| ≥ 0 :=
by
  sorry

end certain_event_abs_nonneg_l65_65061


namespace find_certain_number_l65_65181

theorem find_certain_number (n : ℕ) (h : 9823 + n = 13200) : n = 3377 :=
by
  sorry

end find_certain_number_l65_65181


namespace quadrilateral_area_l65_65646

structure Point :=
  (x : ℝ)
  (y : ℝ)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

def area_of_quadrilateral (A B C D : Point) : ℝ :=
  area_of_triangle A B C + area_of_triangle A C D

def A : Point := ⟨2, 2⟩
def B : Point := ⟨2, -1⟩
def C : Point := ⟨3, -1⟩
def D : Point := ⟨2007, 2008⟩

theorem quadrilateral_area :
  area_of_quadrilateral A B C D = 2008006.5 :=
by
  sorry

end quadrilateral_area_l65_65646


namespace expected_yield_of_carrots_l65_65484

def steps_to_feet (steps : ℕ) (step_size : ℕ) : ℕ :=
  steps * step_size

def garden_area (length width : ℕ) : ℕ :=
  length * width

def yield_of_carrots (area : ℕ) (yield_rate : ℚ) : ℚ :=
  area * yield_rate

theorem expected_yield_of_carrots :
  steps_to_feet 18 3 * steps_to_feet 25 3 = 4050 →
  yield_of_carrots 4050 (3 / 4) = 3037.5 :=
by
  sorry

end expected_yield_of_carrots_l65_65484


namespace proportional_relationships_l65_65193

-- Let l, v, t be real numbers indicating distance, velocity, and time respectively.
variables (l v t : ℝ)

-- Define the relationships according to the given formulas
def distance_formula := l = v * t
def velocity_formula := v = l / t
def time_formula := t = l / v

-- Definitions of proportionality
def directly_proportional (x y : ℝ) := ∃ k : ℝ, x = k * y
def inversely_proportional (x y : ℝ) := ∃ k : ℝ, x * y = k

-- The main theorem
theorem proportional_relationships (const_t const_v const_l : ℝ) :
  (distance_formula l v const_t → directly_proportional l v) ∧
  (distance_formula l const_v t → directly_proportional l t) ∧
  (velocity_formula const_l v t → inversely_proportional v t) :=
by
  sorry

end proportional_relationships_l65_65193


namespace power_sum_divisible_by_five_l65_65047

theorem power_sum_divisible_by_five : 
  (3^444 + 4^333) % 5 = 0 := 
by 
  sorry

end power_sum_divisible_by_five_l65_65047


namespace point_coordinates_l65_65641

-- Definitions based on conditions
def on_x_axis (P : ℝ × ℝ) : Prop := P.2 = 0
def dist_to_y_axis (P : ℝ × ℝ) (d : ℝ) : Prop := abs P.1 = d

-- Lean 4 statement
theorem point_coordinates {P : ℝ × ℝ} (h1 : on_x_axis P) (h2 : dist_to_y_axis P 3) :
  P = (3, 0) ∨ P = (-3, 0) :=
by sorry

end point_coordinates_l65_65641


namespace last_popsicle_melts_32_times_faster_l65_65841

theorem last_popsicle_melts_32_times_faster (t : ℕ) : 
  let time_first := t
  let time_sixth := t / 2^5
  (time_first / time_sixth) = 32 :=
by
  sorry

end last_popsicle_melts_32_times_faster_l65_65841


namespace oak_trees_in_park_l65_65948

theorem oak_trees_in_park (planting_today : ℕ) (total_trees : ℕ) 
  (h1 : planting_today = 4) (h2 : total_trees = 9) : 
  total_trees - planting_today = 5 :=
by
  -- proof goes here
  sorry

end oak_trees_in_park_l65_65948


namespace find_x_l65_65994

variable (x : ℝ)

theorem find_x (h : 2 * x - 12 = -(x + 3)) : x = 3 := 
sorry

end find_x_l65_65994


namespace find_line_AB_l65_65203

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 16

-- Define the line AB
def lineAB (x y : ℝ) : Prop := 2 * x + y + 1 = 0

-- Proof statement: Line AB is the correct line through the intersection points of the two circles
theorem find_line_AB :
  (∃ x y, circle1 x y ∧ circle2 x y) →
  (∀ x y, (circle1 x y ∧ circle2 x y) ↔ lineAB x y) :=
by
  sorry

end find_line_AB_l65_65203


namespace propositions_correctness_l65_65960

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}
def P : Prop := ∃ x : ℝ, x^2 - x - 1 > 0
def negP : Prop := ∀ x : ℝ, x^2 - x - 1 ≤ 0

theorem propositions_correctness :
    (∀ a, a ∈ M → a ∈ N) = false ∧
    (∀ a b, (a ∈ M → b ∉ M) ↔ (b ∈ M → a ∉ M)) ∧
    (∀ p q, ¬(p ∧ q) → ¬p ∧ ¬q) = false ∧ 
    (¬P ↔ negP) :=
by
  sorry

end propositions_correctness_l65_65960


namespace ten_differences_le_100_exists_l65_65966

theorem ten_differences_le_100_exists (s : Finset ℤ) (h_card : s.card = 101) (h_range : ∀ x ∈ s, 0 ≤ x ∧ x ≤ 1000) :
∃ S : Finset ℕ, S.card = 10 ∧ (∀ y ∈ S, y ≤ 100) :=
by {
  sorry
}

end ten_differences_le_100_exists_l65_65966


namespace quadratic_roots_expression_l65_65958

theorem quadratic_roots_expression :
  ∀ (x₁ x₂ : ℝ), 
  (x₁ + x₂ = 3) →
  (x₁ * x₂ = -1) →
  (x₁^2 * x₂ + x₁ * x₂^2 = -3) :=
by
  intros x₁ x₂ h1 h2
  sorry

end quadratic_roots_expression_l65_65958


namespace original_pumpkins_count_l65_65232

def pumpkins_eaten_by_rabbits : ℕ := 23
def pumpkins_left : ℕ := 20
def original_pumpkins : ℕ := pumpkins_left + pumpkins_eaten_by_rabbits

theorem original_pumpkins_count :
  original_pumpkins = 43 :=
sorry

end original_pumpkins_count_l65_65232


namespace bug_probability_at_A_after_8_meters_l65_65115

noncomputable def P : ℕ → ℚ 
| 0 => 1
| (n + 1) => (1 / 3) * (1 - P n)

theorem bug_probability_at_A_after_8_meters :
  P 8 = 547 / 2187 := 
sorry

end bug_probability_at_A_after_8_meters_l65_65115


namespace average_student_headcount_l65_65507

variable (headcount_02_03 headcount_03_04 headcount_04_05 headcount_05_06 : ℕ)
variable {h_02_03 : headcount_02_03 = 10900}
variable {h_03_04 : headcount_03_04 = 10500}
variable {h_04_05 : headcount_04_05 = 10700}
variable {h_05_06 : headcount_05_06 = 11300}

theorem average_student_headcount : 
  (headcount_02_03 + headcount_03_04 + headcount_04_05 + headcount_05_06) / 4 = 10850 := 
by 
  sorry

end average_student_headcount_l65_65507


namespace average_problem_l65_65566

theorem average_problem
  (h : (20 + 40 + 60) / 3 = (x + 50 + 45) / 3 + 5) :
  x = 10 :=
by
  sorry

end average_problem_l65_65566


namespace seeds_total_l65_65785

variable (seedsInBigGarden : Nat)
variable (numSmallGardens : Nat)
variable (seedsPerSmallGarden : Nat)

theorem seeds_total (h1 : seedsInBigGarden = 36) (h2 : numSmallGardens = 3) (h3 : seedsPerSmallGarden = 2) : 
  seedsInBigGarden + numSmallGardens * seedsPerSmallGarden = 42 := by
  sorry

end seeds_total_l65_65785


namespace add_base8_l65_65720

-- Define x and y in base 8 and their sum in base 8
def x := 24 -- base 8
def y := 157 -- base 8
def result := 203 -- base 8

theorem add_base8 : (x + y) = result := 
by sorry

end add_base8_l65_65720


namespace point_a_number_l65_65126

theorem point_a_number (x : ℝ) (h : abs (x - 2) = 6) : x = 8 ∨ x = -4 :=
sorry

end point_a_number_l65_65126


namespace solve_system_l65_65840

theorem solve_system : ∃ x y : ℚ, 
  (2 * x + 3 * y = 7 - 2 * x + 7 - 3 * y) ∧ 
  (3 * x - 2 * y = x - 2 + y - 2) ∧ 
  x = 3 / 4 ∧ 
  y = 11 / 6 := 
by 
  sorry

end solve_system_l65_65840


namespace normals_intersect_at_single_point_l65_65896

-- Definitions of points on the parabola and distinct condition
variables {a b c : ℝ}

-- Condition stating that A, B, C are distinct points
def distinct_points (a b c : ℝ) : Prop :=
  (a - b) ≠ 0 ∧ (b - c) ≠ 0 ∧ (c - a) ≠ 0

-- Statement to be proved
theorem normals_intersect_at_single_point (habc : distinct_points a b c) :
  a + b + c = 0 :=
sorry

end normals_intersect_at_single_point_l65_65896


namespace find_speed_grocery_to_gym_l65_65690

variables (v : ℝ) (speed_grocery_to_gym : ℝ)
variables (d_home_to_grocery : ℝ) (d_grocery_to_gym : ℝ)
variables (time_diff : ℝ)

def problem_conditions : Prop :=
  d_home_to_grocery = 840 ∧
  d_grocery_to_gym = 480 ∧
  time_diff = 40 ∧
  speed_grocery_to_gym = 2 * v

def correct_answer : Prop :=
  speed_grocery_to_gym = 30

theorem find_speed_grocery_to_gym :
  problem_conditions v speed_grocery_to_gym d_home_to_grocery d_grocery_to_gym time_diff →
  correct_answer speed_grocery_to_gym :=
by
  sorry

end find_speed_grocery_to_gym_l65_65690


namespace frac_pattern_2_11_frac_pattern_general_l65_65129

theorem frac_pattern_2_11 :
  (2 / 11) = (1 / 6) + (1 / 66) :=
sorry

theorem frac_pattern_general (n : ℕ) (hn : n ≥ 3) :
  (2 / (2 * n - 1)) = (1 / n) + (1 / (n * (2 * n - 1))) :=
sorry

end frac_pattern_2_11_frac_pattern_general_l65_65129


namespace distance_difference_l65_65560

-- Definition of speeds and time
def speed_alberto : ℕ := 16
def speed_clara : ℕ := 12
def time_hours : ℕ := 5

-- Distance calculation functions
def distance (speed time : ℕ) : ℕ := speed * time

-- Main theorem statement
theorem distance_difference : 
  distance speed_alberto time_hours - distance speed_clara time_hours = 20 :=
by
  sorry

end distance_difference_l65_65560


namespace solve_equation_l65_65148

theorem solve_equation (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
sorry

end solve_equation_l65_65148


namespace average_of_remaining_two_numbers_l65_65340

theorem average_of_remaining_two_numbers
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.9)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.45 :=
sorry

end average_of_remaining_two_numbers_l65_65340


namespace evaluate_expression_l65_65628

/-
  Define the expressions from the conditions.
  We define the numerator and denominator separately.
-/
def expr_numerator : ℚ := 1 - (1 / 4)
def expr_denominator : ℚ := 1 - (1 / 3)

/-
  Define the original expression to be proven.
  This is our main expression to evaluate.
-/
def expr : ℚ := expr_numerator / expr_denominator

/-
  State the final proof problem that the expression is equal to 9/8.
-/
theorem evaluate_expression : expr = 9 / 8 := sorry

end evaluate_expression_l65_65628


namespace mrs_franklin_gave_38_packs_l65_65952

-- Define the initial number of Valentines
def initial_valentines : Int := 450

-- Define the remaining Valentines after giving some away
def remaining_valentines : Int := 70

-- Define the size of each pack
def pack_size : Int := 10

-- Define the number of packs given away
def packs_given (initial remaining pack_size : Int) : Int :=
  (initial - remaining) / pack_size

theorem mrs_franklin_gave_38_packs :
  packs_given 450 70 10 = 38 := sorry

end mrs_franklin_gave_38_packs_l65_65952


namespace total_weight_is_correct_l65_65074

def siblings_suitcases : Nat := 1 + 2 + 3 + 4 + 5 + 6
def weight_per_sibling_suitcase : Nat := 10
def total_weight_siblings : Nat := siblings_suitcases * weight_per_sibling_suitcase

def parents : Nat := 2
def suitcases_per_parent : Nat := 3
def weight_per_parent_suitcase : Nat := 12
def total_weight_parents : Nat := parents * suitcases_per_parent * weight_per_parent_suitcase

def grandparents : Nat := 2
def suitcases_per_grandparent : Nat := 2
def weight_per_grandparent_suitcase : Nat := 8
def total_weight_grandparents : Nat := grandparents * suitcases_per_grandparent * weight_per_grandparent_suitcase

def other_relatives_suitcases : Nat := 8
def weight_per_other_relatives_suitcase : Nat := 15
def total_weight_other_relatives : Nat := other_relatives_suitcases * weight_per_other_relatives_suitcase

def total_weight_all_suitcases : Nat := total_weight_siblings + total_weight_parents + total_weight_grandparents + total_weight_other_relatives

theorem total_weight_is_correct : total_weight_all_suitcases = 434 := by {
  sorry
}

end total_weight_is_correct_l65_65074


namespace general_term_a_n_sum_b_n_terms_l65_65150

-- Given definitions based on the conditions
def a (n : ℕ) : ℕ := 2^(n-1)

def b (n : ℕ) : ℕ := (2^(2*n-1))^2

def b_sum (n : ℕ) : (ℕ → ℕ) := 
  (fun b : ℕ => match b with 
                | 1 => 4 
                | 2 => 64 
                | _ => (4^(2*(b - 2 + 1) - 1)))

def T (n : ℕ) : ℕ := (4 / 15) * (16^n - 1)

-- First part: Proving the general term of {a_n} is 2^(n-1)
theorem general_term_a_n (n : ℕ) : a n = 2^(n-1) := by
  sorry

-- Second part: Proving the sum of the first n terms of {b_n} is (4/15)*(16^n - 1)
theorem sum_b_n_terms (n : ℕ) : T n = (4 / 15) * (16^n - 1) := by 
  sorry

end general_term_a_n_sum_b_n_terms_l65_65150


namespace steve_more_than_wayne_first_time_at_2004_l65_65836

def initial_steve_money (year: ℕ) := if year = 2000 then 100 else 0
def initial_wayne_money (year: ℕ) := if year = 2000 then 10000 else 0

def steve_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_steve_money year
  else 2 * steve_money (year - 1)

def wayne_money (year: ℕ) : ℕ :=
  if year < 2000 then 0
  else if year = 2000 then initial_wayne_money year
  else wayne_money (year - 1) / 2

theorem steve_more_than_wayne_first_time_at_2004 :
  ∃ (year: ℕ), year = 2004 ∧ steve_money year > wayne_money year := by
  sorry

end steve_more_than_wayne_first_time_at_2004_l65_65836


namespace dave_books_about_outer_space_l65_65412

theorem dave_books_about_outer_space (x : ℕ) 
  (H1 : 8 + 3 = 11) 
  (H2 : 11 * 6 = 66) 
  (H3 : 102 - 66 = 36) 
  (H4 : 36 / 6 = x) : 
  x = 6 := 
by
  sorry

end dave_books_about_outer_space_l65_65412


namespace rectangle_difference_l65_65402

theorem rectangle_difference (L B : ℝ) (h1 : 2 * (L + B) = 266) (h2 : L * B = 4290) :
  L - B = 23 :=
sorry

end rectangle_difference_l65_65402


namespace chess_amateurs_play_with_l65_65488

theorem chess_amateurs_play_with :
  ∃ n : ℕ, ∃ total_players : ℕ, total_players = 6 ∧
  (total_players * (total_players - 1)) / 2 = 12 ∧
  (n = total_players - 1 ∧ n = 5) :=
by
  sorry

end chess_amateurs_play_with_l65_65488


namespace robot_possible_path_lengths_l65_65436

theorem robot_possible_path_lengths (n : ℕ) (valid_path: ∀ (i : ℕ), i < n → (i % 4 = 0 ∨ i % 4 = 1 ∨ i % 4 = 2 ∨ i % 4 = 3)) :
  (n % 4 = 0) :=
by
  sorry

end robot_possible_path_lengths_l65_65436


namespace area_of_AFCH_l65_65873

-- Define the sides of the rectangles ABCD and EFGH
def AB : ℝ := 9
def BC : ℝ := 5
def EF : ℝ := 3
def FG : ℝ := 10

-- Define the area of quadrilateral AFCH
def area_AFCH : ℝ := 52.5

-- The theorem we want to prove
theorem area_of_AFCH :
  AB = 9 ∧ BC = 5 ∧ EF = 3 ∧ FG = 10 → (area_AFCH = 52.5) :=
by
  sorry

end area_of_AFCH_l65_65873


namespace difference_sixth_seventh_l65_65080

theorem difference_sixth_seventh
  (A1 A2 A3 A4 A5 A6 A7 A8 : ℕ)
  (h_avg_8 : (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8) / 8 = 25)
  (h_avg_2 : (A1 + A2) / 2 = 20)
  (h_avg_3 : (A3 + A4 + A5) / 3 = 26)
  (h_A8 : A8 = 30)
  (h_A6_A8 : A6 = A8 - 6) :
  A7 - A6 = 4 :=
by
  sorry

end difference_sixth_seventh_l65_65080


namespace ratio_first_term_l65_65050

theorem ratio_first_term (x : ℕ) (r : ℕ × ℕ) (h₀ : r = (6 - x, 7 - x)) 
        (h₁ : x ≥ 3) (h₂ : r.1 < r.2) : r.1 < 4 :=
by
  sorry

end ratio_first_term_l65_65050


namespace mixture_replacement_l65_65256

theorem mixture_replacement (A B x : ℕ) (hA : A = 32) (h_ratio1 : A / B = 4) (h_ratio2 : A / (B + x) = 2 / 3) : x = 40 :=
by
  sorry

end mixture_replacement_l65_65256


namespace find_x_floor_mult_eq_45_l65_65199

theorem find_x_floor_mult_eq_45 (x : ℝ) (h1 : 0 < x) (h2 : ⌊x⌋ * x = 45) : x = 7.5 :=
sorry

end find_x_floor_mult_eq_45_l65_65199


namespace sampling_methods_correct_l65_65694

def condition1 : Prop :=
  ∃ yogurt_boxes : ℕ, yogurt_boxes = 10 ∧ ∃ sample_boxes : ℕ, sample_boxes = 3

def condition2 : Prop :=
  ∃ rows seats_per_row attendees sample_size : ℕ,
    rows = 32 ∧ seats_per_row = 40 ∧ attendees = rows * seats_per_row ∧ sample_size = 32

def condition3 : Prop :=
  ∃ liberal_arts_classes science_classes total_classes sample_size : ℕ,
    liberal_arts_classes = 4 ∧ science_classes = 8 ∧ total_classes = liberal_arts_classes + science_classes ∧ sample_size = 50

def simple_random_sampling (s : Prop) : Prop := sorry -- definition for simple random sampling
def systematic_sampling (s : Prop) : Prop := sorry -- definition for systematic sampling
def stratified_sampling (s : Prop) : Prop := sorry -- definition for stratified sampling

theorem sampling_methods_correct :
  (condition1 → simple_random_sampling condition1) ∧
  (condition2 → systematic_sampling condition2) ∧
  (condition3 → stratified_sampling condition3) :=
by {
  sorry
}

end sampling_methods_correct_l65_65694


namespace sum_of_decimals_l65_65654

theorem sum_of_decimals :
  let a := 0.35
  let b := 0.048
  let c := 0.0072
  a + b + c = 0.4052 := by
  sorry

end sum_of_decimals_l65_65654


namespace pow_mult_same_base_l65_65023

theorem pow_mult_same_base (a b : ℕ) : 10^a * 10^b = 10^(a + b) := by 
  sorry

example : 10^655 * 10^652 = 10^1307 :=
  pow_mult_same_base 655 652

end pow_mult_same_base_l65_65023


namespace reciprocal_of_fraction_sum_l65_65847

theorem reciprocal_of_fraction_sum : 
  (1 / (1 / 3 + 1 / 4 - 1 / 12)) = 2 := sorry

end reciprocal_of_fraction_sum_l65_65847


namespace sum_of_squares_largest_multiple_of_7_l65_65417

theorem sum_of_squares_largest_multiple_of_7
  (N : ℕ) (a : ℕ) (h1 : N = a^2 + (a + 1)^2 + (a + 2)^2)
  (h2 : N < 10000)
  (h3 : 7 ∣ N) :
  N = 8750 := sorry

end sum_of_squares_largest_multiple_of_7_l65_65417


namespace eggs_remainder_and_full_cartons_l65_65663

def abigail_eggs := 48
def beatrice_eggs := 63
def carson_eggs := 27
def carton_size := 15

theorem eggs_remainder_and_full_cartons :
  let total_eggs := abigail_eggs + beatrice_eggs + carson_eggs
  ∃ (full_cartons left_over : ℕ),
    total_eggs = full_cartons * carton_size + left_over ∧
    left_over = 3 ∧
    full_cartons = 9 :=
by
  sorry

end eggs_remainder_and_full_cartons_l65_65663


namespace complex_number_arithmetic_l65_65749

theorem complex_number_arithmetic (i : ℂ) (h : i^2 = -1) : (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end complex_number_arithmetic_l65_65749


namespace no_integers_p_q_l65_65316

theorem no_integers_p_q :
  ¬ ∃ p q : ℤ, ∀ x : ℤ, 3 ∣ (x^2 + p * x + q) :=
by
  sorry

end no_integers_p_q_l65_65316


namespace cos_beta_half_l65_65233

theorem cos_beta_half (α β : ℝ) (hα_ac : 0 < α ∧ α < π / 2) (hβ_ac : 0 < β ∧ β < π / 2) 
  (h1 : Real.tan α = 4 * Real.sqrt 3) (h2 : Real.cos (α + β) = -11 / 14) : 
  Real.cos β = 1 / 2 :=
by
  sorry

end cos_beta_half_l65_65233


namespace chris_did_not_get_A_l65_65597

variable (A : Prop) (MC_correct : Prop) (Essay80 : Prop)

-- The condition provided by professor
axiom condition : A ↔ (MC_correct ∧ Essay80)

-- The theorem we need to prove based on the statement (B) from the solution
theorem chris_did_not_get_A 
    (h : ¬ A) : ¬ MC_correct ∨ ¬ Essay80 :=
by sorry

end chris_did_not_get_A_l65_65597


namespace games_in_each_box_l65_65583

theorem games_in_each_box (start_games sold_games total_boxes remaining_games games_per_box : ℕ) 
  (h_start: start_games = 35) (h_sold: sold_games = 19) (h_boxes: total_boxes = 2) 
  (h_remaining: remaining_games = start_games - sold_games) 
  (h_per_box: games_per_box = remaining_games / total_boxes) : games_per_box = 8 :=
by
  sorry

end games_in_each_box_l65_65583


namespace crackers_per_friend_l65_65943

theorem crackers_per_friend (Total_crackers Left_crackers Friends : ℕ) (h1 : Total_crackers = 23) (h2 : Left_crackers = 11) (h3 : Friends = 2):
  (Total_crackers - Left_crackers) / Friends = 6 :=
by
  sorry

end crackers_per_friend_l65_65943


namespace Linda_original_savings_l65_65562

variable (TV_cost : ℝ := 200) -- TV cost
variable (savings : ℝ) -- Linda's original savings

-- Prices, Discounts, Taxes
variable (sofa_price : ℝ := 600)
variable (sofa_discount : ℝ := 0.20)
variable (sofa_tax : ℝ := 0.05)

variable (dining_table_price : ℝ := 400)
variable (dining_table_discount : ℝ := 0.15)
variable (dining_table_tax : ℝ := 0.06)

variable (chair_set_price : ℝ := 300)
variable (chair_set_discount : ℝ := 0.25)
variable (chair_set_tax : ℝ := 0.04)

variable (coffee_table_price : ℝ := 100)
variable (coffee_table_discount : ℝ := 0.10)
variable (coffee_table_tax : ℝ := 0.03)

variable (service_charge_rate : ℝ := 0.02) -- Service charge rate

noncomputable def discounted_price_with_tax (price discount tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

noncomputable def total_furniture_cost : ℝ :=
  let sofa_cost := discounted_price_with_tax sofa_price sofa_discount sofa_tax
  let dining_table_cost := discounted_price_with_tax dining_table_price dining_table_discount dining_table_tax
  let chair_set_cost := discounted_price_with_tax chair_set_price chair_set_discount chair_set_tax
  let coffee_table_cost := discounted_price_with_tax coffee_table_price coffee_table_discount coffee_table_tax
  let combined_cost := sofa_cost + dining_table_cost + chair_set_cost + coffee_table_cost
  combined_cost * (1 + service_charge_rate)

theorem Linda_original_savings : savings = 4 * TV_cost ∧ savings / 4 * 3 = total_furniture_cost :=
by
  sorry -- Proof skipped

end Linda_original_savings_l65_65562


namespace sum_of_special_right_triangle_areas_l65_65987

noncomputable def is_special_right_triangle (a b : ℕ) : Prop :=
  let area := (a * b) / 2
  area = 3 * (a + b)

noncomputable def special_right_triangle_areas : List ℕ :=
  [(18, 9), (9, 18), (15, 10), (10, 15), (12, 12)].map (λ p => (p.1 * p.2) / 2)

theorem sum_of_special_right_triangle_areas : 
  special_right_triangle_areas.eraseDups.sum = 228 := by
  sorry

end sum_of_special_right_triangle_areas_l65_65987


namespace Hector_gumballs_l65_65005

theorem Hector_gumballs :
  ∃ (total_gumballs : ℕ)
  (gumballs_Todd : ℕ) (gumballs_Alisha : ℕ) (gumballs_Bobby : ℕ) (gumballs_remaining : ℕ),
  gumballs_Todd = 4 ∧
  gumballs_Alisha = 2 * gumballs_Todd ∧
  gumballs_Bobby = 4 * gumballs_Alisha - 5 ∧
  gumballs_remaining = 6 ∧
  total_gumballs = gumballs_Todd + gumballs_Alisha + gumballs_Bobby + gumballs_remaining ∧
  total_gumballs = 45 :=
by
  sorry

end Hector_gumballs_l65_65005


namespace ratio_S15_S5_l65_65912

variable {a : ℕ → ℝ}  -- The geometric sequence
variable {S : ℕ → ℝ}  -- The sum of the first n terms of the geometric sequence

-- Define the conditions:
axiom sum_of_first_n_terms (n : ℕ) : S n = a 0 * (1 - (a 1)^n) / (1 - a 1)
axiom ratio_S10_S5 : S 10 / S 5 = 1 / 2

-- Define the math proof problem:
theorem ratio_S15_S5 : S 15 / S 5 = 3 / 4 :=
  sorry

end ratio_S15_S5_l65_65912


namespace sum_of_first_10_bn_l65_65830

def a (n : ℕ) : ℚ :=
  (2 / 5) * n + (3 / 5)

def b (n : ℕ) : ℤ :=
  ⌊a n⌋

def sum_first_10_b : ℤ :=
  (b 1) + (b 2) + (b 3) + (b 4) + (b 5) + (b 6) + (b 7) + (b 8) + (b 9) + (b 10)

theorem sum_of_first_10_bn : sum_first_10_b = 24 :=
  by sorry

end sum_of_first_10_bn_l65_65830


namespace overall_percent_change_l65_65600

theorem overall_percent_change (W : ℝ) : 
  (W * 0.6 * 1.3 * 0.8 * 1.1) / W = 0.624 :=
by {
  sorry
}

end overall_percent_change_l65_65600


namespace number_of_devices_bought_l65_65732

-- Define the essential parameters
def original_price : Int := 800000
def discounted_price : Int := 450000
def total_discount : Int := 16450000

-- Define the main statement to prove
theorem number_of_devices_bought : (total_discount / (original_price - discounted_price) = 47) :=
by
  -- The essential proof is skipped here with sorry
  sorry

end number_of_devices_bought_l65_65732


namespace no_consecutive_even_square_and_three_times_square_no_consecutive_square_and_seven_times_square_l65_65667

-- Problem 1: Square of an even number followed by three times a square number
theorem no_consecutive_even_square_and_three_times_square :
  ∀ (k n : ℕ), ¬(3 * n ^ 2 = 4 * k ^ 2 + 1) :=
by sorry

-- Problem 2: Square number followed by seven times another square number
theorem no_consecutive_square_and_seven_times_square :
  ∀ (r s : ℕ), ¬(7 * s ^ 2 = r ^ 2 + 1) :=
by sorry

end no_consecutive_even_square_and_three_times_square_no_consecutive_square_and_seven_times_square_l65_65667


namespace largest_sum_fraction_l65_65281

open Rat

theorem largest_sum_fraction :
  let a := (2:ℚ) / 5
  let c1 := (1:ℚ) / 6
  let c2 := (1:ℚ) / 3
  let c3 := (1:ℚ) / 7
  let c4 := (1:ℚ) / 8
  let c5 := (1:ℚ) / 9
  max (a + c1) (max (a + c2) (max (a + c3) (max (a + c4) (a + c5)))) = a + c2
  ∧ a + c2 = (11:ℚ) / 15 := by
  sorry

end largest_sum_fraction_l65_65281


namespace gathering_people_total_l65_65420

theorem gathering_people_total (W S B : ℕ) (hW : W = 26) (hS : S = 22) (hB : B = 17) :
  W + S - B = 31 :=
by
  sorry

end gathering_people_total_l65_65420


namespace drink_all_tea_l65_65369

theorem drink_all_tea (cups : Fin 30 → Prop) (red blue : Fin 30 → Prop)
  (h₀ : ∀ n, cups n ↔ (red n ↔ ¬ blue n))
  (h₁ : ∃ a b, a ≠ b ∧ red a ∧ blue b)
  (h₂ : ∀ n, red n → red (n + 2))
  (h₃ : ∀ n, blue n → blue (n + 2)) :
  ∃ sequence : ℕ → Fin 30, (∀ n, cups (sequence n)) ∧ (sequence 0 ≠ sequence 1) 
  ∧ (∀ n, cups (sequence (n+1))) :=
by
  sorry

end drink_all_tea_l65_65369


namespace wrapping_paper_area_correct_l65_65395

variable (w h : ℝ) -- Define the base length and height of the box.

-- Lean statement for the problem asserting that the area of the wrapping paper is \(2(w+h)^2\).
def wrapping_paper_area (w h : ℝ) : ℝ := 2 * (w + h) ^ 2

-- Theorem stating that the derived formula for the area of the wrapping paper is correct.
theorem wrapping_paper_area_correct (w h : ℝ) : wrapping_paper_area w h = 2 * (w + h) ^ 2 :=
by
  sorry -- Proof is omitted

end wrapping_paper_area_correct_l65_65395


namespace km_to_leaps_l65_65123

theorem km_to_leaps (a b c d e f : ℕ) :
  (2 * a) * strides = (3 * b) * leaps →
  (4 * c) * dashes = (5 * d) * strides →
  (6 * e) * dashes = (7 * f) * kilometers →
  1 * kilometers = (90 * b * d * e) / (56 * a * c * f) * leaps :=
by
  -- Using the given conditions to derive the answer
  intro h1 h2 h3
  sorry

end km_to_leaps_l65_65123


namespace drink_total_amount_l65_65955

theorem drink_total_amount (parts_coke parts_sprite parts_mountain_dew ounces_coke total_parts : ℕ)
  (h1 : parts_coke = 2) (h2 : parts_sprite = 1) (h3 : parts_mountain_dew = 3)
  (h4 : total_parts = parts_coke + parts_sprite + parts_mountain_dew)
  (h5 : ounces_coke = 6) :
  ( ounces_coke * total_parts ) / parts_coke = 18 :=
by
  sorry

end drink_total_amount_l65_65955


namespace joan_final_oranges_l65_65736

def joan_oranges_initial := 75
def tom_oranges := 42
def sara_sold := 40
def christine_added := 15

theorem joan_final_oranges : joan_oranges_initial + tom_oranges - sara_sold + christine_added = 92 :=
by 
  sorry

end joan_final_oranges_l65_65736


namespace cost_price_250_l65_65056

theorem cost_price_250 (C : ℝ) (h1 : 0.90 * C = C - 0.10 * C) (h2 : 1.10 * C = C + 0.10 * C) (h3 : 1.10 * C - 0.90 * C = 50) : C = 250 := 
by
  sorry

end cost_price_250_l65_65056


namespace units_digit_seven_pow_ten_l65_65651

theorem units_digit_seven_pow_ten : ∃ u : ℕ, (7^10) % 10 = u ∧ u = 9 :=
by
  use 9
  sorry

end units_digit_seven_pow_ten_l65_65651


namespace max_value_of_expr_l65_65915

theorem max_value_of_expr  
  (a b c : ℝ) 
  (h₀ : 0 ≤ a)
  (h₁ : 0 ≤ b)
  (h₂ : 0 ≤ c)
  (h₃ : a + 2 * b + 3 * c = 1) :
  a + b^3 + c^4 ≤ 0.125 := 
sorry

end max_value_of_expr_l65_65915


namespace solve_m_l65_65084

theorem solve_m (m : ℝ) :
  (∃ x > 0, (2 * m - 4) ^ 2 = x ∧ (3 * m - 1) ^ 2 = x) →
  (m = -3 ∨ m = 1) :=
by 
  sorry

end solve_m_l65_65084


namespace strawberry_picking_l65_65722

theorem strawberry_picking 
  (e : ℕ) (n : ℕ) (p : ℕ) (A : ℕ) (w : ℕ) 
  (h1 : e = 4) 
  (h2 : n = 3) 
  (h3 : p = 20) 
  (h4 : A = 128) 
  : w = 7 :=
by 
  -- proof steps to be filled in
  sorry

end strawberry_picking_l65_65722


namespace initial_pennies_indeterminate_l65_65024

-- Conditions
def initial_nickels : ℕ := 7
def dad_nickels : ℕ := 9
def mom_nickels : ℕ := 2
def total_nickels_now : ℕ := 18

-- Proof problem statement
theorem initial_pennies_indeterminate :
  ∀ (initial_nickels dad_nickels mom_nickels total_nickels_now : ℕ), 
  initial_nickels = 7 → dad_nickels = 9 → mom_nickels = 2 → total_nickels_now = 18 → 
  (∃ (initial_pennies : ℕ), true) → false :=
by
  sorry

end initial_pennies_indeterminate_l65_65024


namespace stones_on_perimeter_of_square_l65_65820

theorem stones_on_perimeter_of_square (n : ℕ) (h : n = 5) : 
  4 * n - 4 = 16 :=
by
  sorry

end stones_on_perimeter_of_square_l65_65820


namespace roots_quadratic_expression_l65_65770

theorem roots_quadratic_expression (α β : ℝ) (hα : α^2 - 3 * α - 2 = 0) (hβ : β^2 - 3 * β - 2 = 0) :
    7 * α^4 + 10 * β^3 = 544 := 
sorry

end roots_quadratic_expression_l65_65770


namespace sum_of_solutions_l65_65008

theorem sum_of_solutions (x : ℝ) :
  (∀ x : ℝ, x^3 + x^2 - 6*x - 20 = 4*x + 24) →
  let polynomial := (x^3 + x^2 - 10*x - 44);
  (polynomial = 0) →
  let a := 1;
  let b := 1;
  -b/a = -1 :=
sorry

end sum_of_solutions_l65_65008


namespace solution_set_abs_inequality_l65_65131

theorem solution_set_abs_inequality (x : ℝ) :
  |2 * x + 1| < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end solution_set_abs_inequality_l65_65131


namespace intersection_is_4_l65_65189

-- Definitions of the sets
def U : Set Int := {0, 1, 2, 4, 6, 8}
def M : Set Int := {0, 4, 6}
def N : Set Int := {0, 1, 6}

-- Definition of the complement
def complement_U_N : Set Int := U \ N

-- Definition of the intersection
def intersection_M_complement_U_N : Set Int := M ∩ complement_U_N

-- Statement of the theorem
theorem intersection_is_4 : intersection_M_complement_U_N = {4} :=
by
  sorry

end intersection_is_4_l65_65189


namespace triangle_region_areas_l65_65726

open Real

theorem triangle_region_areas (A B C : ℝ) 
  (h1 : 20^2 + 21^2 = 29^2)
  (h2 : ∃ (triangle_area : ℝ), triangle_area = 210)
  (h3 : C > A)
  (h4 : C > B)
  : A + B + 210 = C := 
sorry

end triangle_region_areas_l65_65726


namespace positive_integer_solutions_eq_8_2_l65_65442

-- Define the variables and conditions in the problem
def positive_integer_solution_count_eq (n m : ℕ) : Prop :=
  ∀ (x₁ x₂ x₃ x₄ : ℕ),
    x₂ = m →
    (x₁ + x₂ + x₃ + x₄ = n) →
    (x₁ > 0 ∧ x₃ > 0 ∧ x₄ > 0) →
    -- Number of positive integer solutions should be 10
    (x₁ + x₃ + x₄ = 6)

-- Statement of the theorem
theorem positive_integer_solutions_eq_8_2 : positive_integer_solution_count_eq 8 2 := sorry

end positive_integer_solutions_eq_8_2_l65_65442


namespace subtraction_correct_l65_65552

theorem subtraction_correct :
  2222222222222 - 1111111111111 = 1111111111111 := by
  sorry

end subtraction_correct_l65_65552


namespace ratio_of_areas_l65_65755

theorem ratio_of_areas (r : ℝ) (h1 : r > 0) : 
  let OX := r / 3
  let area_OP := π * r ^ 2
  let area_OX := π * (OX) ^ 2
  (area_OX / area_OP) = 1 / 9 :=
by
  sorry

end ratio_of_areas_l65_65755


namespace ratio_correct_l65_65311

def my_age : ℕ := 35
def son_age_next_year : ℕ := 8
def son_age_now : ℕ := son_age_next_year - 1
def ratio_of_ages : ℕ := my_age / son_age_now

theorem ratio_correct : ratio_of_ages = 5 :=
by
  -- Add proof here
  sorry

end ratio_correct_l65_65311


namespace trig_identity_l65_65145

theorem trig_identity (f : ℝ → ℝ) (ϕ : ℝ) (h₁ : ∀ x, f x = 2 * Real.sin (2 * x + ϕ)) (h₂ : 0 < ϕ) (h₃ : ϕ < π) (h₄ : f 0 = 1) :
  f ϕ = 2 :=
sorry

end trig_identity_l65_65145


namespace true_propositions_l65_65845

noncomputable def discriminant_leq_zero : Prop :=
  let a := 1
  let b := -1
  let c := 2
  b^2 - 4 * a * c ≤ 0

def proposition_1 : Prop := discriminant_leq_zero

def proposition_2 (x : ℝ) : Prop :=
  abs x ≥ 0 → x ≥ 0

def proposition_3 : Prop :=
  5 > 2 ∧ 3 < 7

theorem true_propositions : proposition_1 ∧ proposition_3 ∧ ¬∀ x : ℝ, proposition_2 x :=
by
  sorry

end true_propositions_l65_65845


namespace incoming_class_student_count_l65_65384

theorem incoming_class_student_count (n : ℕ) :
  n < 1000 ∧ n % 25 = 18 ∧ n % 28 = 26 → n = 418 :=
by
  sorry

end incoming_class_student_count_l65_65384


namespace a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l65_65706

-- Definitions for the point P and movements
def move (P : ℤ) (flip : Bool) : ℤ :=
  if flip then P + 1 else -P

-- Definitions for probabilities
def probability_of_event (events : ℕ) (successful : ℕ) : ℚ :=
  successful / events

def probability_a3_zero : ℚ :=
  probability_of_event 8 2  -- 2 out of 8 sequences lead to a3 = 0

def probability_a4_one : ℚ :=
  probability_of_event 16 2  -- 2 out of 16 sequences lead to a4 = 1

noncomputable def probability_an_n_minus_3 (n : ℕ) : ℚ :=
  if n < 3 then 0 else (n - 1) / (2 ^ n)

-- Statements to prove
theorem a3_probability_is_one_fourth : probability_a3_zero = 1/4 := by
  sorry

theorem a4_probability_is_one_eighth : probability_a4_one = 1/8 := by
  sorry

theorem an_n_minus_3_probability (n : ℕ) (hn : n ≥ 3) : probability_an_n_minus_3 n = (n - 1) / (2^n) := by
  sorry

end a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l65_65706


namespace quadratic_rewrite_sum_l65_65100

theorem quadratic_rewrite_sum (a b c : ℝ) (x : ℝ) :
  -3 * x^2 + 15 * x + 75 = a * (x + b)^2 + c → (a + b + c) = 88.25 :=
sorry

end quadratic_rewrite_sum_l65_65100


namespace water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l65_65341

noncomputable def tiered_water_bill (usage : ℕ) : ℝ :=
  if usage <= 20 then
    2.3 * usage
  else if usage <= 30 then
    2.3 * 20 + 3.45 * (usage - 20)
  else
    2.3 * 20 + 3.45 * 10 + 4.6 * (usage - 30)

-- (1) Prove that if Xiao Ming's family used 32 cubic meters of water in August, 
-- their water bill is 89.7 yuan.
theorem water_bill_august_32m_cubed : tiered_water_bill 32 = 89.7 := by
  sorry

-- (2) Prove that if Xiao Ming's family paid 59.8 yuan for their water bill in October, 
-- they used 24 cubic meters of water.
theorem water_usage_october_59_8_yuan : ∃ x : ℕ, tiered_water_bill x = 59.8 ∧ x = 24 := by
  use 24
  sorry

end water_bill_august_32m_cubed_water_usage_october_59_8_yuan_l65_65341


namespace no_common_root_of_polynomials_l65_65072

theorem no_common_root_of_polynomials (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) : 
  ∀ x : ℝ, ¬ (x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0) :=
by
  intro x
  sorry

end no_common_root_of_polynomials_l65_65072


namespace abs_conditions_iff_l65_65793

theorem abs_conditions_iff (x y : ℝ) :
  (|x| < 1 ∧ |y| < 1) ↔ (|x + y| + |x - y| < 2) :=
by
  sorry

end abs_conditions_iff_l65_65793


namespace exists_sequence_a_l65_65644

-- Define the sequence and properties
def sequence_a (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  a 18 = 2019 ∧
  ∀ k, 3 ≤ k → k ≤ 18 → ∃ i j, 1 ≤ i → i < j → j < k → a k = a i + a j

-- The main theorem statement
theorem exists_sequence_a : ∃ (a : ℕ → ℤ), sequence_a a := 
sorry

end exists_sequence_a_l65_65644


namespace trigonometric_identity_l65_65635

theorem trigonometric_identity (α : ℝ) (h : (1 + Real.tan α) / (1 - Real.tan α) = 2012) : 
  (1 / Real.cos (2 * α)) + Real.tan (2 * α) = 2012 := 
by
  -- This will be the proof body which we omit with sorry
  sorry

end trigonometric_identity_l65_65635


namespace find_k_l65_65424

-- Define the vector operations and properties

def vector_add (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
def vector_smul (k : ℝ) (a : ℝ × ℝ) : ℝ × ℝ := (k * a.1, k * a.2)
def vectors_parallel (a b : ℝ × ℝ) : Prop := 
  (a.1 * b.2 = a.2 * b.1)

-- Given vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Statement of the problem
theorem find_k (k : ℝ) : 
  vectors_parallel (vector_add (vector_smul k a) b) (vector_add a (vector_smul (-3) b)) 
  → k = -1 / 3 :=
by
  sorry

end find_k_l65_65424


namespace lighter_dog_weight_l65_65021

theorem lighter_dog_weight
  (x y z : ℕ)
  (h1 : x + y + z = 36)
  (h2 : y + z = 3 * x)
  (h3 : x + z = 2 * y) :
  x = 9 :=
by
  sorry

end lighter_dog_weight_l65_65021


namespace simultaneous_equations_solution_l65_65188

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = (2 * m - 1) * x + 4) ↔ m ≠ 1 :=
by
  sorry

end simultaneous_equations_solution_l65_65188


namespace relationship_y1_y2_l65_65676

theorem relationship_y1_y2 :
  ∀ (b y1 y2 : ℝ), 
  (∃ b y1 y2, y1 = -2023 * (-2) + b ∧ y2 = -2023 * (-1) + b) → y1 > y2 :=
by
  intro b y1 y2 h
  sorry

end relationship_y1_y2_l65_65676


namespace candle_length_sum_l65_65775

theorem candle_length_sum (l s : ℕ) (x : ℤ) 
  (h1 : l = s + 32)
  (h2 : s = (5 * x)) 
  (h3 : l = (7 * (3 * x))) :
  l + s = 52 := 
sorry

end candle_length_sum_l65_65775


namespace min_value_a2_b2_l65_65683

theorem min_value_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (h : a^2 - 2015 * a = b^2 - 2015 * b) : 
  a^2 + b^2 ≥ 2015^2 / 2 := 
sorry

end min_value_a2_b2_l65_65683


namespace num_new_terms_in_sequence_l65_65842

theorem num_new_terms_in_sequence (k : ℕ) (h : k ≥ 2) : 
  (2^(k+1) - 1) - (2^k - 1) = 2^k := by
  sorry

end num_new_terms_in_sequence_l65_65842


namespace angle_triple_complement_l65_65971

theorem angle_triple_complement (x : ℝ) (h1 : x = 3 * (90 - x)) : x = 67.5 := 
by {
  sorry
}

end angle_triple_complement_l65_65971


namespace volume_of_larger_cube_l65_65494

theorem volume_of_larger_cube (s : ℝ) (V : ℝ) :
  (∀ (n : ℕ), n = 125 →
    ∀ (v_sm : ℝ), v_sm = 1 →
    V = n * v_sm →
    V = s^3 →
    s = 5 →
    ∀ (sa_large : ℝ), sa_large = 6 * s^2 →
    sa_large = 150 →
    ∀ (sa_sm_total : ℝ), sa_sm_total = n * (6 * v_sm^(2/3)) →
    sa_sm_total = 750 →
    sa_sm_total - sa_large = 600 →
    V = 125) :=
by
  intros n n125 v_sm v1 Vdef Vcube sc5 sa_large sa_large_def sa_large150 sa_sm_total sa_sm_total_def sa_sm_total750 diff600
  simp at *
  sorry

end volume_of_larger_cube_l65_65494


namespace monkey_reaches_top_in_19_minutes_l65_65128

theorem monkey_reaches_top_in_19_minutes (pole_height : ℕ) (ascend_first_min : ℕ) (slip_every_alternate_min : ℕ) 
    (total_minutes : ℕ) (net_gain_two_min : ℕ) : 
    pole_height = 10 ∧ ascend_first_min = 2 ∧ slip_every_alternate_min = 1 ∧ net_gain_two_min = 1 ∧ total_minutes = 19 →
    (net_gain_two_min * (total_minutes - 1) / 2 + ascend_first_min = pole_height) := 
by
    intros
    sorry

end monkey_reaches_top_in_19_minutes_l65_65128


namespace total_phd_time_l65_65322

-- Definitions for the conditions
def acclimation_period : ℕ := 1
def basics_period : ℕ := 2
def research_period := basics_period + (3 * basics_period / 4)
def dissertation_period := acclimation_period / 2

-- Main statement to prove
theorem total_phd_time : acclimation_period + basics_period + research_period + dissertation_period = 7 := by
  -- Here should be the proof (skipped with sorry)
  sorry

end total_phd_time_l65_65322


namespace fraction_value_l65_65905

def x : ℚ := 4 / 7
def y : ℚ := 8 / 11

theorem fraction_value : (7 * x + 11 * y) / (49 * x * y) = 231 / 56 := by
  sorry

end fraction_value_l65_65905


namespace length_stationary_l65_65365

def speed : ℝ := 64.8
def time_pole : ℝ := 5
def time_stationary : ℝ := 25

def length_moving : ℝ := speed * time_pole
def length_combined : ℝ := speed * time_stationary

theorem length_stationary : length_combined - length_moving = 1296 :=
by
  sorry

end length_stationary_l65_65365


namespace two_roots_iff_a_greater_than_neg1_l65_65362

theorem two_roots_iff_a_greater_than_neg1 (a : ℝ) :
  (∃! x : ℝ, x^2 + 2*x + 2*|x + 1| = a) ↔ a > -1 :=
sorry

end two_roots_iff_a_greater_than_neg1_l65_65362


namespace not_approximately_equal_exp_l65_65154

noncomputable def multinomial_approximation (n k₁ k₂ k₃ k₄ k₅ : ℕ) : ℝ :=
  (n.factorial : ℝ) / ((k₁.factorial : ℝ) * (k₂.factorial : ℝ) * (k₃.factorial : ℝ) * (k₄.factorial : ℝ) * (k₅.factorial : ℝ))

theorem not_approximately_equal_exp (e : ℝ) (h1 : e > 0) :
  e ^ 2737 ≠ multinomial_approximation 1000 70 270 300 220 140 :=
by 
  sorry  

end not_approximately_equal_exp_l65_65154


namespace cows_with_no_spot_l65_65155

theorem cows_with_no_spot (total_cows : ℕ) (percent_red_spot : ℚ) (percent_blue_spot : ℚ) :
  total_cows = 140 ∧ percent_red_spot = 0.40 ∧ percent_blue_spot = 0.25 → 
  ∃ (no_spot_cows : ℕ), no_spot_cows = 63 :=
by 
  sorry

end cows_with_no_spot_l65_65155


namespace sum_of_cubes_l65_65410

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) : x^3 + y^3 = -10 :=
by
  sorry

end sum_of_cubes_l65_65410


namespace sports_popularity_order_l65_65660

theorem sports_popularity_order :
  let soccer := (13 : ℚ) / 40
  let baseball := (9 : ℚ) / 30
  let basketball := (7 : ℚ) / 20
  let volleyball := (3 : ℚ) / 10
  basketball > soccer ∧ soccer > baseball ∧ baseball = volleyball :=
by
  sorry

end sports_popularity_order_l65_65660


namespace pears_total_l65_65491

-- Conditions
def keith_initial_pears : ℕ := 47
def keith_given_pears : ℕ := 46
def mike_initial_pears : ℕ := 12

-- Define the remaining pears
def keith_remaining_pears : ℕ := keith_initial_pears - keith_given_pears
def mike_remaining_pears : ℕ := mike_initial_pears

-- Theorem statement
theorem pears_total :
  keith_remaining_pears + mike_remaining_pears = 13 :=
by
  sorry

end pears_total_l65_65491


namespace number_of_roots_l65_65198

def S : Set ℚ := { x : ℚ | 0 < x ∧ x < (5 : ℚ)/8 }

def f (x : ℚ) : ℚ := 
  match x.num, x.den with
  | num, den => num / den + 1

theorem number_of_roots (h : ∀ q p, (p, q) = 1 → (q : ℚ) / p ∈ S → ((q + 1 : ℚ) / p = (2 : ℚ) / 3)) :
  ∃ n : ℕ, n = 7 :=
sorry

end number_of_roots_l65_65198


namespace area_of_inscribed_triangle_l65_65302

-- Define the square with a given diagonal
def diagonal (d : ℝ) : Prop := d = 16
def side_length_of_square (s : ℝ) : Prop := s = 8 * Real.sqrt 2
def side_length_of_equilateral_triangle (a : ℝ) : Prop := a = 8 * Real.sqrt 2

-- Define the area of the equilateral triangle
def area_of_equilateral_triangle (area : ℝ) : Prop :=
  area = 32 * Real.sqrt 3

-- The theorem: Given the above conditions, prove the area of the equilateral triangle
theorem area_of_inscribed_triangle (d s a area : ℝ) 
  (h1 : diagonal d) 
  (h2 : side_length_of_square s) 
  (h3 : side_length_of_equilateral_triangle a) 
  (h4 : s = a) : 
  area_of_equilateral_triangle area :=
sorry

end area_of_inscribed_triangle_l65_65302


namespace find_k_for_circle_radius_l65_65619

theorem find_k_for_circle_radius (k : ℝ) :
  (∃ x y : ℝ, x^2 + 14*x + y^2 + 8*y - k = 0 ∧ (x + 7)^2 + (y + 4)^2 = 10^2) ↔ k = 35 :=
by
  sorry

end find_k_for_circle_radius_l65_65619


namespace gcd_143_144_l65_65913

def a : ℕ := 143
def b : ℕ := 144

theorem gcd_143_144 : Nat.gcd a b = 1 :=
by
  sorry

end gcd_143_144_l65_65913


namespace factor_quadratic_expression_l65_65497

theorem factor_quadratic_expression (x y : ℝ) :
  5 * x^2 + 6 * x * y - 8 * y^2 = (x + 2 * y) * (5 * x - 4 * y) :=
by
  sorry

end factor_quadratic_expression_l65_65497


namespace parameterized_line_equation_l65_65979

theorem parameterized_line_equation (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 * t + 6) 
  (h2 : y = 5 * t - 7) : 
  y = (5 / 3) * x - 17 :=
sorry

end parameterized_line_equation_l65_65979


namespace measure_of_angle_l65_65444

theorem measure_of_angle (x : ℝ) (h : 90 - x = 3 * x - 10) : x = 25 :=
by
  sorry

end measure_of_angle_l65_65444


namespace problem1_problem2_l65_65656

-- Problem statement 1: Prove (a-2)(a-6) < (a-3)(a-5)
theorem problem1 (a : ℝ) : (a - 2) * (a - 6) < (a - 3) * (a - 5) :=
by
  sorry

-- Problem statement 2: Prove the range of values for 2x - y given -2 < x < 1 and 1 < y < 2 is (-6, 1)
theorem problem2 (x y : ℝ) (hx : -2 < x) (hx1 : x < 1) (hy : 1 < y) (hy1 : y < 2) : -6 < 2 * x - y ∧ 2 * x - y < 1 :=
by
  sorry

end problem1_problem2_l65_65656


namespace student_score_is_64_l65_65500

-- Define the total number of questions and correct responses.
def total_questions : ℕ := 100
def correct_responses : ℕ := 88

-- Function to calculate the score based on the grading rule.
def calculate_score (total : ℕ) (correct : ℕ) : ℕ :=
  correct - 2 * (total - correct)

-- The theorem that states the score for the given conditions.
theorem student_score_is_64 :
  calculate_score total_questions correct_responses = 64 :=
by
  sorry

end student_score_is_64_l65_65500


namespace find_number_l65_65456

theorem find_number (N : ℕ) (h1 : N / 3 = 8) (h2 : N / 8 = 3) : N = 24 :=
by
  sorry

end find_number_l65_65456


namespace average_white_paper_per_ton_trees_saved_per_ton_l65_65325

-- Define the given conditions
def waste_paper_tons : ℕ := 5
def produced_white_paper_tons : ℕ := 4
def saved_trees : ℕ := 40

-- State the theorems that need to be proved
theorem average_white_paper_per_ton :
  (produced_white_paper_tons : ℚ) / waste_paper_tons = 0.8 := 
sorry

theorem trees_saved_per_ton :
  (saved_trees : ℚ) / waste_paper_tons = 8 := 
sorry

end average_white_paper_per_ton_trees_saved_per_ton_l65_65325


namespace y_affected_by_other_factors_l65_65033

-- Given the linear regression model
def linear_regression_model (b a e x : ℝ) : ℝ := b * x + a + e

-- Theorem: Prove that the dependent variable \( y \) may be affected by factors other than the independent variable \( x \)
theorem y_affected_by_other_factors (b a e x : ℝ) :
  ∃ y, (y = linear_regression_model b a e x ∧ e ≠ 0) :=
sorry

end y_affected_by_other_factors_l65_65033


namespace percentage_increase_is_50_l65_65647

def papaya_growth (P : ℝ) : Prop :=
  let growth1 := 2
  let growth2 := 2 * (1 + P / 100)
  let growth3 := 1.5 * growth2
  let growth4 := 2 * growth3
  let growth5 := 0.5 * growth4
  growth1 + growth2 + growth3 + growth4 + growth5 = 23

theorem percentage_increase_is_50 :
  ∃ (P : ℝ), papaya_growth P ∧ P = 50 := by
  sorry

end percentage_increase_is_50_l65_65647


namespace find_g_eq_minus_x_l65_65892

-- Define the function g and the given conditions.
def g (x : ℝ) : ℝ := sorry

axiom g0 : g 0 = 2
axiom g_xy : ∀ (x y : ℝ), g (x * y) = g ((x^2 + 2 * y^2) / 3) + 3 * (x - y)^2

-- State the problem: proving that g(x) = -x.
theorem find_g_eq_minus_x : ∀ (x : ℝ), g x = -x := by
  sorry

end find_g_eq_minus_x_l65_65892


namespace inscribed_circle_radius_l65_65176

theorem inscribed_circle_radius (AB BC CD DA: ℝ) (hAB: AB = 13) (hBC: BC = 10) (hCD: CD = 8) (hDA: DA = 11) :
  ∃ r, r = 2 * Real.sqrt 7 :=
by
  sorry

end inscribed_circle_radius_l65_65176


namespace min_score_needed_l65_65165

theorem min_score_needed 
  (s1 s2 s3 s4 s5 : ℕ)
  (next_test_goal_increment : ℕ)
  (current_scores_sum : ℕ)
  (desired_average : ℕ)
  (total_tests : ℕ)
  (required_total_sum : ℕ)
  (required_next_score : ℕ)
  (current_scores : s1 = 88 ∧ s2 = 92 ∧ s3 = 75 ∧ s4 = 85 ∧ s5 = 80)
  (increment_eq : next_test_goal_increment = 5)
  (current_sum_eq : current_scores_sum = s1 + s2 + s3 + s4 + s5)
  (desired_average_eq : desired_average = (current_scores_sum / 5) + next_test_goal_increment)
  (total_tests_eq : total_tests = 6)
  (required_total_sum_eq : required_total_sum = desired_average * total_tests)
  (required_next_score_eq : required_next_score = required_total_sum - current_scores_sum) :
  required_next_score = 114 := by
    sorry

end min_score_needed_l65_65165


namespace evaluate_fraction_sum_l65_65060

-- Define the problem conditions and target equation
theorem evaluate_fraction_sum
    (p q r : ℝ)
    (h : p / (30 - p) + q / (75 - q) + r / (45 - r) = 8) :
    6 / (30 - p) + 15 / (75 - q) + 9 / (45 - r) = 11 / 5 := by
  sorry

end evaluate_fraction_sum_l65_65060


namespace ratio_female_to_male_l65_65982

theorem ratio_female_to_male
  (a b c : ℕ)
  (ha : a = 60)
  (hb : b = 80)
  (hc : c = 65) :
  f / m = 1 / 3 := 
by
  sorry

end ratio_female_to_male_l65_65982


namespace good_permutation_exists_iff_power_of_two_l65_65857

def is_good_permutation (n : ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ i j k : ℕ, i < j → j < k → k < n → ¬ (↑n ∣ (a i + a k - 2 * a j))

theorem good_permutation_exists_iff_power_of_two (n : ℕ) (h : n ≥ 3) :
  (∃ a : ℕ → ℕ, (∀ i, i < n → a i < n) ∧ is_good_permutation n a) ↔ ∃ b : ℕ, 2 ^ b = n :=
sorry

end good_permutation_exists_iff_power_of_two_l65_65857


namespace greatest_divisor_condition_gcd_of_numbers_l65_65502

theorem greatest_divisor_condition (n : ℕ) (h100 : n ∣ 100) (h225 : n ∣ 225) (h150 : n ∣ 150) : n ≤ 25 :=
  sorry

theorem gcd_of_numbers : Nat.gcd (Nat.gcd 100 225) 150 = 25 :=
  sorry

end greatest_divisor_condition_gcd_of_numbers_l65_65502


namespace c_completion_days_l65_65737

noncomputable def work_rate (days: ℕ) := (1 : ℝ) / days

theorem c_completion_days : 
  ∀ (W : ℝ) (Ra Rb Rc : ℝ) (Dc : ℕ),
  Ra = work_rate 30 → Rb = work_rate 30 → Rc = work_rate Dc →
  (Ra + Rb + Rc) * 8 + (Ra + Rb) * 4 = W → 
  Dc = 40 :=
by
  intros W Ra Rb Rc Dc hRa hRb hRc hW
  sorry

end c_completion_days_l65_65737


namespace remainder_mod_7_l65_65352

theorem remainder_mod_7 : (9^7 + 8^8 + 7^9) % 7 = 3 :=
by sorry

end remainder_mod_7_l65_65352


namespace find_second_number_l65_65698

theorem find_second_number (x : ℕ) : 9548 + x = 3362 + 13500 → x = 7314 := by
  sorry

end find_second_number_l65_65698


namespace geometric_figure_perimeter_l65_65771

theorem geometric_figure_perimeter (A : ℝ) (n : ℝ) (area : ℝ) (side_length : ℝ) (perimeter : ℝ) : 
  A = 216 ∧ n = 6 ∧ area = A / n ∧ side_length = Real.sqrt area ∧ perimeter = 2 * (3 * side_length + 2 * side_length) + 2 * side_length →
  perimeter = 72 := 
by 
  sorry

end geometric_figure_perimeter_l65_65771


namespace total_dining_bill_before_tip_l65_65107

-- Define total number of people
def numberOfPeople : ℕ := 6

-- Define the individual payment
def individualShare : ℝ := 25.48

-- Define the total payment
def totalPayment : ℝ := numberOfPeople * individualShare

-- Define the tip percentage
def tipPercentage : ℝ := 0.10

-- Total payment including tip expressed in terms of the original bill B
def totalPaymentWithTip (B : ℝ) : ℝ := B + B * tipPercentage

-- Prove the total dining bill before the tip
theorem total_dining_bill_before_tip : 
    ∃ B : ℝ, totalPayment = totalPaymentWithTip B ∧ B = 139.89 :=
by
    sorry

end total_dining_bill_before_tip_l65_65107


namespace square_diff_correctness_l65_65540

theorem square_diff_correctness (x y : ℝ) :
  let A := (x + y) * (x - 2*y)
  let B := (x + y) * (-x + y)
  let C := (x + y) * (-x - y)
  let D := (-x + y) * (x - y)
  (∃ (a b : ℝ), B = (a + b) * (a - b)) ∧ (∀ (p q : ℝ), A ≠ (p + q) * (p - q)) ∧ (∀ (r s : ℝ), C ≠ (r + s) * (r - s)) ∧ (∀ (t u : ℝ), D ≠ (t + u) * (t - u)) :=
by
  sorry

end square_diff_correctness_l65_65540


namespace impossible_load_two_coins_l65_65743

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l65_65743


namespace sum_arithmetic_series_base8_l65_65585

theorem sum_arithmetic_series_base8 : 
  let n := 36
  let a := 1
  let l := 30 -- 36_8 in base 10 is 30
  let S := (n * (a + l)) / 2
  let sum_base10 := 558
  let sum_base8 := 1056 -- 558 in base 8 is 1056
  S = sum_base10 ∧ sum_base10 = 1056 :=
by
  sorry

end sum_arithmetic_series_base8_l65_65585


namespace degree_of_product_l65_65387

-- Definitions for the conditions
def isDegree (p : Polynomial ℝ) (n : ℕ) : Prop :=
  p.degree = n

variable {h j : Polynomial ℝ}

-- Given conditions
axiom h_deg : isDegree h 3
axiom j_deg : isDegree j 6

-- The theorem to prove
theorem degree_of_product : h.degree = 3 → j.degree = 6 → (Polynomial.degree (Polynomial.comp h (Polynomial.X ^ 4) * Polynomial.comp j (Polynomial.X ^ 3)) = 30) :=
by
  intros h3 j6
  sorry

end degree_of_product_l65_65387


namespace river_width_l65_65263

noncomputable def width_of_river (d: ℝ) (f: ℝ) (v: ℝ) : ℝ :=
  v / (d * (f * 1000 / 60))

theorem river_width : width_of_river 2 2 3000 = 45 := by
  sorry

end river_width_l65_65263


namespace quadratic_discriminant_l65_65000

theorem quadratic_discriminant : 
  let a := 4
  let b := -6
  let c := 9
  (b^2 - 4 * a * c = -108) := 
by
  sorry

end quadratic_discriminant_l65_65000


namespace circular_film_diameter_l65_65471

-- Definition of the problem conditions
def liquidVolume : ℝ := 576  -- volume of liquid Y in cm^3
def filmThickness : ℝ := 0.2  -- thickness of the film in cm

-- Statement of the theorem to prove the diameter of the film
theorem circular_film_diameter :
  2 * Real.sqrt (2880 / Real.pi) = 2 * Real.sqrt (liquidVolume / (filmThickness * Real.pi)) := by
  sorry

end circular_film_diameter_l65_65471


namespace desiree_age_l65_65526

-- Definitions of the given variables and conditions
variables (D C : ℝ)

-- Given conditions
def condition1 : Prop := D = 2 * C
def condition2 : Prop := D + 30 = 0.6666666 * (C + 30) + 14
def condition3 : Prop := D = 2.99999835

-- Main theorem to prove
theorem desiree_age : D = 2.99999835 :=
by
  { sorry }

end desiree_age_l65_65526


namespace parameterize_line_l65_65486

theorem parameterize_line (f : ℝ → ℝ) (t : ℝ) (x y : ℝ)
  (h1 : y = 2 * x - 30)
  (h2 : (x, y) = (f t, 20 * t - 10)) :
  f t = 10 * t + 10 :=
sorry

end parameterize_line_l65_65486


namespace five_cds_cost_with_discount_l65_65354

theorem five_cds_cost_with_discount
  (price_2_cds : ℝ)
  (discount_rate : ℝ)
  (num_cds : ℕ)
  (total_cost : ℝ) 
  (h1 : price_2_cds = 40)
  (h2 : discount_rate = 0.10)
  (h3 : num_cds = 5)
  : total_cost = 90 :=
by
  sorry

end five_cds_cost_with_discount_l65_65354


namespace other_acute_angle_in_right_triangle_l65_65067

theorem other_acute_angle_in_right_triangle (α : ℝ) (β : ℝ) (γ : ℝ) 
  (h1 : α + β + γ = 180) (h2 : γ = 90) (h3 : α = 30) : β = 60 := 
sorry

end other_acute_angle_in_right_triangle_l65_65067


namespace julia_more_kids_on_Monday_l65_65390

def kids_played_on_Tuesday : Nat := 14
def kids_played_on_Monday : Nat := 22

theorem julia_more_kids_on_Monday : kids_played_on_Monday - kids_played_on_Tuesday = 8 :=
by {
  sorry
}

end julia_more_kids_on_Monday_l65_65390


namespace contrapositive_of_proposition_is_false_l65_65379

theorem contrapositive_of_proposition_is_false (x y : ℝ) 
  (h₀ : (x + y > 0) → (x > 0 ∧ y > 0)) : 
  ¬ ((x ≤ 0 ∨ y ≤ 0) → (x + y ≤ 0)) :=
by
  sorry

end contrapositive_of_proposition_is_false_l65_65379


namespace remainder_of_expression_l65_65476

theorem remainder_of_expression (n : ℤ) : (10 + n^2) % 7 = (3 + n^2) % 7 := 
by {
  sorry
}

end remainder_of_expression_l65_65476


namespace problem1_problem2_l65_65898

noncomputable def f (x a b : ℝ) := |x + a^2| + |x - b^2|

theorem problem1 (a b x : ℝ) (h : a^2 + b^2 - 2 * a + 2 * b + 2 = 0) :
  f x a b >= 3 ↔ x <= -0.5 ∨ x >= 1.5 :=
sorry

theorem problem2 (a b x : ℝ) (h : a + b = 4) :
  f x a b >= 8 :=
sorry

end problem1_problem2_l65_65898


namespace smallest_three_digit_number_with_property_l65_65682

theorem smallest_three_digit_number_with_property :
  ∃ a : ℕ, 100 ≤ a ∧ a < 1000 ∧ ∃ n : ℕ, 1000 * a + (a + 1) = n^2 ∧ a = 183 :=
sorry

end smallest_three_digit_number_with_property_l65_65682


namespace find_m_l65_65367

theorem find_m (m : ℝ) : 
  (m^2 + 3 * m + 3 ≠ 0) ∧ (m^2 + 2 * m - 3 ≠ 0) ∧ 
  (m^2 + 3 * m + 3 = 1) → m = -2 := 
by
  sorry

end find_m_l65_65367


namespace quadrilateral_area_l65_65764

-- Define the number of interior and boundary points
def interior_points : ℕ := 5
def boundary_points : ℕ := 4

-- State the theorem to prove the area of the quadrilateral using Pick's Theorem
theorem quadrilateral_area : interior_points + (boundary_points / 2) - 1 = 6 := by sorry

end quadrilateral_area_l65_65764


namespace proof_problem_l65_65294

variable (a b c d x : ℤ)

-- Conditions
axiom condition1 : a - b = c + d + x
axiom condition2 : a + b = c - d - 3
axiom condition3 : a - c = 3
axiom answer_eq : x = 9

-- Proof statement
theorem proof_problem : (a - b) = (c + d + 9) :=
by
  sorry

end proof_problem_l65_65294


namespace gcd_a_b_l65_65811

def a : ℕ := 6666666
def b : ℕ := 999999999

theorem gcd_a_b : Nat.gcd a b = 3 := by
  sorry

end gcd_a_b_l65_65811


namespace number_of_incorrect_statements_l65_65492

-- Conditions
def cond1 (p q : Prop) : Prop := (p ∨ q) → (p ∧ q)

def cond2 (x : ℝ) : Prop := x > 5 → x^2 - 4*x - 5 > 0

def cond3 : Prop := ∃ x0 : ℝ, x0^2 + x0 - 1 < 0

def cond3_neg : Prop := ∀ x : ℝ, x^2 + x - 1 ≥ 0

def cond4 (x : ℝ) : Prop := (x ≠ 1 ∨ x ≠ 2) → (x^2 - 3*x + 2 ≠ 0)

-- Proof problem
theorem number_of_incorrect_statements : 
  (¬ cond1 (p := true) (q := false)) ∧ (cond2 (x := 6)) ∧ (cond3 → cond3_neg) ∧ (¬ cond4 (x := 0)) → 
  2 = 2 :=
by
  sorry

end number_of_incorrect_statements_l65_65492


namespace floor_sqrt_23_squared_l65_65238

theorem floor_sqrt_23_squared : (Int.floor (Real.sqrt 23))^2 = 16 := 
by
  -- conditions
  have h1 : 4^2 = 16 := by norm_num
  have h2 : 5^2 = 25 := by norm_num
  have h3 : 16 < 23 := by norm_num
  have h4 : 23 < 25 := by norm_num
  -- statement (goal)
  sorry

end floor_sqrt_23_squared_l65_65238


namespace find_amount_l65_65655

theorem find_amount (x : ℝ) (h1 : 0.25 * x = 0.15 * 1500 - 30) (h2 : x = 780) : 30 = 30 :=
by
  sorry

end find_amount_l65_65655


namespace ingrid_income_l65_65891

theorem ingrid_income (combined_tax_rate : ℝ)
  (john_income : ℝ) (john_tax_rate : ℝ)
  (ingrid_tax_rate : ℝ)
  (combined_income : ℝ)
  (combined_tax : ℝ) :
  combined_tax_rate = 0.35581395348837205 →
  john_income = 57000 →
  john_tax_rate = 0.3 →
  ingrid_tax_rate = 0.4 →
  combined_income = john_income + (combined_income - john_income) →
  combined_tax = (john_tax_rate * john_income) + (ingrid_tax_rate * (combined_income - john_income)) →
  combined_tax_rate = combined_tax / combined_income →
  combined_income = 57000 + 72000 :=
by
  sorry

end ingrid_income_l65_65891


namespace ratio_product_even_odd_composite_l65_65013

theorem ratio_product_even_odd_composite :
  (4 * 6 * 8 * 10 * 12) / (9 * 15 * 21 * 25 * 27) = (2^10) / (3^6 * 5^2 * 7) :=
by
  sorry

end ratio_product_even_odd_composite_l65_65013


namespace tan_4x_eq_cos_x_has_9_solutions_l65_65482

theorem tan_4x_eq_cos_x_has_9_solutions :
  ∃ (s : Finset ℝ), s.card = 9 ∧ ∀ x ∈ s, (0 ≤ x ∧ x ≤ 2 * Real.pi) ∧ (Real.tan (4 * x) = Real.cos x) :=
sorry

end tan_4x_eq_cos_x_has_9_solutions_l65_65482


namespace intersection_x_value_l65_65070

theorem intersection_x_value :
  ∀ x y: ℝ,
    (y = 3 * x - 15) ∧ (3 * x + y = 120) → x = 22.5 := by
  sorry

end intersection_x_value_l65_65070


namespace cecile_apples_l65_65672

theorem cecile_apples (C D : ℕ) (h1 : D = C + 20) (h2 : C + D = 50) : C = 15 :=
by
  -- Proof steps would go here
  sorry

end cecile_apples_l65_65672


namespace inequality_proof_l65_65725

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 0) :
  (33 * a^2 - a) / (33 * a^2 + 1) + (33 * b^2 - b) / (33 * b^2 + 1) + (33 * c^2 - c) / (33 * c^2 + 1) ≥ 0 :=
sorry

end inequality_proof_l65_65725


namespace explicit_formula_for_f_l65_65028

def f (k : ℕ) : ℚ :=
  if k = 1 then 4 / 3
  else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3

theorem explicit_formula_for_f (k : ℕ) (hk : k ≥ 1) : 
  (f k = if k = 1 then 4 / 3 else (5 ^ (k + 1) / 81) * (8 * 10 ^ (1 - k) - 9 * k + 1) + (2 ^ (k + 1)) / 3) ∧ 
  ∀ k ≥ 2, 2 * f k = f (k - 1) - k * 5^k + 2^k :=
by {
  sorry
}

end explicit_formula_for_f_l65_65028


namespace range_of_a_l65_65405

theorem range_of_a (a : ℝ)
  (h : ∀ x y : ℝ, (4 * x - 3 * y - 2 = 0) → (x^2 + y^2 - 2 * a * x + 4 * y + a^2 - 12 = 0) → x ≠ y) :
  -6 < a ∧ a < 4 :=
by
  sorry

end range_of_a_l65_65405


namespace relationship_of_squares_and_products_l65_65323

theorem relationship_of_squares_and_products (a b x : ℝ) (h1 : b < x) (h2 : x < a) (h3 : a < 0) : 
  x^2 > ax ∧ ax > b^2 :=
by
  sorry

end relationship_of_squares_and_products_l65_65323


namespace animal_sale_money_l65_65183

theorem animal_sale_money (G S : ℕ) (h1 : G + S = 360) (h2 : 5 * S = 7 * G) : 
  (1/2 * G * 40) + (2/3 * S * 30) = 7200 := 
by
  sorry

end animal_sale_money_l65_65183


namespace greatest_common_ratio_l65_65063

theorem greatest_common_ratio {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
  (h4 : (b = (a + c) / 2 → b^2 = a * c) ∨ (c = (a + b) / 2 ∧ b = -a / 2)) :
  ∃ r : ℝ, r = -2 :=
by
  sorry

end greatest_common_ratio_l65_65063


namespace square_side_length_l65_65991

-- Define the given dimensions and total length
def rectangle_width : ℕ := 2
def total_length : ℕ := 7

-- Define the unknown side length of the square
variable (Y : ℕ)

-- State the problem and provide the conclusion
theorem square_side_length : Y + rectangle_width = total_length -> Y = 5 :=
by 
  sorry

end square_side_length_l65_65991


namespace cone_volume_proof_l65_65797

noncomputable def slant_height := 21
noncomputable def horizontal_semi_axis := 10
noncomputable def vertical_semi_axis := 12
noncomputable def equivalent_radius :=
  Real.sqrt (horizontal_semi_axis * vertical_semi_axis)
noncomputable def cone_height :=
  Real.sqrt (slant_height ^ 2 - equivalent_radius ^ 2)

noncomputable def cone_volume :=
  (1 / 3) * Real.pi * horizontal_semi_axis * vertical_semi_axis * cone_height

theorem cone_volume_proof :
  cone_volume = 2250.24 * Real.pi := sorry

end cone_volume_proof_l65_65797


namespace inequality_proof_l65_65434

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z) ^ 2 * (y * z + z * x + x * y) ^ 2 ≤ 
  3 * (y^2 + y * z + z^2) * (z^2 + z * x + x^2) * (x^2 + x * y + y^2) := 
sorry

end inequality_proof_l65_65434


namespace sum_of_modified_numbers_l65_65613

theorem sum_of_modified_numbers (x y R : ℝ) (h : x + y = R) : 
  2 * (x + 4) + 2 * (y + 5) = 2 * R + 18 :=
by
  sorry

end sum_of_modified_numbers_l65_65613


namespace fermat_little_theorem_l65_65577

theorem fermat_little_theorem (p : ℕ) (a : ℤ) (hp : Nat.Prime p) (hcoprime : Int.gcd a p = 1) : 
  (a ^ (p - 1)) % p = 1 % p := 
sorry

end fermat_little_theorem_l65_65577


namespace ratio_of_pages_given_l65_65170

variable (Lana_initial_pages : ℕ) (Duane_initial_pages : ℕ) (Lana_final_pages : ℕ)

theorem ratio_of_pages_given
  (h1 : Lana_initial_pages = 8)
  (h2 : Duane_initial_pages = 42)
  (h3 : Lana_final_pages = 29) :
  (Lana_final_pages - Lana_initial_pages) / Duane_initial_pages = 1 / 2 :=
  by
  -- Placeholder for the proof
  sorry

end ratio_of_pages_given_l65_65170


namespace geometric_sequence_a4_l65_65668

theorem geometric_sequence_a4 {a : ℕ → ℝ} (q : ℝ) (h₁ : q > 0)
  (h₂ : ∀ n, a (n + 1) = a 1 * q ^ (n)) (h₃ : a 1 = 2) 
  (h₄ : a 2 + 4 = (a 1 + a 3) / 2) : a 4 = 54 := 
by
  sorry

end geometric_sequence_a4_l65_65668


namespace fourth_person_height_l65_65053

variable (H : ℝ)
variable (height1 height2 height3 height4 : ℝ)

theorem fourth_person_height
  (h1 : height1 = H)
  (h2 : height2 = H + 2)
  (h3 : height3 = H + 4)
  (h4 : height4 = H + 10)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 78) :
  height4 = 84 :=
by
  sorry

end fourth_person_height_l65_65053


namespace percentage_per_annum_is_correct_l65_65882

-- Define the conditions of the problem
def banker_gain : ℝ := 24
def present_worth : ℝ := 600
def time : ℕ := 2

-- Define the formula for the amount due
def amount_due (r : ℝ) (t : ℕ) (PW : ℝ) : ℝ := PW * (1 + r * t)

-- Define the given conditions translated from the problem
def given_conditions (r : ℝ) : Prop :=
  amount_due r time present_worth = present_worth + banker_gain

-- Lean statement of the problem to be proved
theorem percentage_per_annum_is_correct :
  ∃ r : ℝ, given_conditions r ∧ r = 0.02 :=
by {
  sorry
}

end percentage_per_annum_is_correct_l65_65882


namespace point_coordinates_in_second_quadrant_l65_65932

theorem point_coordinates_in_second_quadrant (P : ℝ × ℝ)
  (hx : P.1 ≤ 0)
  (hy : P.2 ≥ 0)
  (dist_x_axis : abs P.2 = 3)
  (dist_y_axis : abs P.1 = 10) :
  P = (-10, 3) :=
by
  sorry

end point_coordinates_in_second_quadrant_l65_65932


namespace ptolemys_inequality_l65_65160

variable {A B C D : Type} [OrderedRing A]
variable (AB BC CD DA AC BD : A)

/-- Ptolemy's inequality for a quadrilateral -/
theorem ptolemys_inequality 
  (AB_ BC_ CD_ DA_ AC_ BD_ : A) :
  AC * BD ≤ AB * CD + BC * AD :=
  sorry

end ptolemys_inequality_l65_65160


namespace car_discount_l65_65517

variable (P D : ℝ)

theorem car_discount (h1 : 0 < P)
                     (h2 : (P - D) * 1.45 = 1.16 * P) :
                     D = 0.2 * P := by
  sorry

end car_discount_l65_65517


namespace cooking_time_l65_65262

theorem cooking_time
  (total_potatoes : ℕ) (cooked_potatoes : ℕ) (remaining_time : ℕ) (remaining_potatoes : ℕ)
  (h_total : total_potatoes = 15)
  (h_cooked : cooked_potatoes = 8)
  (h_remaining_time : remaining_time = 63)
  (h_remaining_potatoes : remaining_potatoes = total_potatoes - cooked_potatoes) :
  remaining_time / remaining_potatoes = 9 :=
by
  sorry

end cooking_time_l65_65262


namespace b_range_condition_l65_65699

theorem b_range_condition (b : ℝ) : 
  -2 * Real.sqrt 6 < b ∧ b < 2 * Real.sqrt 6 ↔ (b^2 - 24) < 0 :=
by
  sorry

end b_range_condition_l65_65699


namespace pastries_sold_correctly_l65_65504

def cupcakes : ℕ := 4
def cookies : ℕ := 29
def total_pastries : ℕ := cupcakes + cookies
def left_over : ℕ := 24
def sold_pastries : ℕ := total_pastries - left_over

theorem pastries_sold_correctly : sold_pastries = 9 :=
by sorry

end pastries_sold_correctly_l65_65504


namespace find_b_l65_65458

theorem find_b (b : ℝ) (y : ℝ) : (4 * 3 + 2 * y = b) ∧ (3 * 3 + 6 * y = 3 * b) → b = 27 :=
by
sorry

end find_b_l65_65458


namespace bird_family_problem_l65_65799

def initial_bird_families (f s i : Nat) : Prop :=
  i = f + s

theorem bird_family_problem : initial_bird_families 32 35 67 :=
by
  -- Proof would go here
  sorry

end bird_family_problem_l65_65799


namespace eliminate_denominator_correctness_l65_65601

-- Define the initial equality with fractions
def initial_equation (x : ℝ) := (2 * x - 3) / 5 = (2 * x) / 3 - 3

-- Define the resulting expression after eliminating the denominators
def eliminated_denominators (x : ℝ) := 3 * (2 * x - 3) = 5 * 2 * x - 3 * 15

-- The theorem states that given the initial equation, the eliminated denomination expression holds true
theorem eliminate_denominator_correctness (x : ℝ) :
  initial_equation x → eliminated_denominators x := by
  sorry

end eliminate_denominator_correctness_l65_65601


namespace taxi_fare_function_l65_65570

theorem taxi_fare_function (x : ℝ) (h : x > 3) : 
  ∃ y : ℝ, y = 2 * x + 4 :=
by
  sorry

end taxi_fare_function_l65_65570


namespace sticks_picked_up_l65_65944

variable (original_sticks left_sticks picked_sticks : ℕ)

theorem sticks_picked_up :
  original_sticks = 99 → left_sticks = 61 → picked_sticks = original_sticks - left_sticks → picked_sticks = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sticks_picked_up_l65_65944


namespace max_stamps_l65_65329

theorem max_stamps (price_per_stamp : ℕ) (total_money : ℕ) (h_price : price_per_stamp = 37) (h_total : total_money = 4000) : 
  ∃ max_stamps : ℕ, max_stamps = 108 ∧ max_stamps * price_per_stamp ≤ total_money ∧ ∀ n : ℕ, n * price_per_stamp ≤ total_money → n ≤ max_stamps :=
by
  sorry

end max_stamps_l65_65329


namespace Ruby_apples_remaining_l65_65375

def Ruby_original_apples : ℕ := 6357912
def Emily_takes_apples : ℕ := 2581435
def Ruby_remaining_apples (R E : ℕ) : ℕ := R - E

theorem Ruby_apples_remaining : Ruby_remaining_apples Ruby_original_apples Emily_takes_apples = 3776477 := by
  sorry

end Ruby_apples_remaining_l65_65375


namespace percent_increase_equilateral_triangles_l65_65617

theorem percent_increase_equilateral_triangles :
  let s₁ := 3
  let s₂ := 2 * s₁
  let s₃ := 2 * s₂
  let s₄ := 2 * s₃
  let P₁ := 3 * s₁
  let P₄ := 3 * s₄
  (P₄ - P₁) / P₁ * 100 = 700 :=
by
  sorry

end percent_increase_equilateral_triangles_l65_65617


namespace soft_lenses_more_than_hard_l65_65157

-- Define the problem conditions as Lean definitions
def total_sales (S H : ℕ) : Prop := 150 * S + 85 * H = 1455
def total_pairs (S H : ℕ) : Prop := S + H = 11

-- The theorem we need to prove
theorem soft_lenses_more_than_hard (S H : ℕ) (h1 : total_sales S H) (h2 : total_pairs S H) : S - H = 5 :=
by
  sorry

end soft_lenses_more_than_hard_l65_65157


namespace find_b_l65_65970

theorem find_b (a b c y1 y2 : ℝ) (h1 : y1 = a * 2^2 + b * 2 + c) 
              (h2 : y2 = a * (-2)^2 + b * (-2) + c) 
              (h3 : y1 - y2 = -12) : b = -3 :=
by 
  sorry

end find_b_l65_65970


namespace sum_of_three_numbers_l65_65823

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156)
  (h2 : a * b + b * c + c * a = 50) :
  a + b + c = 16 :=
by
  sorry

end sum_of_three_numbers_l65_65823


namespace total_tagged_numbers_l65_65226

theorem total_tagged_numbers:
  let W := 200
  let X := W / 2
  let Y := X + W
  let Z := 400
  W + X + Y + Z = 1000 := by 
    sorry

end total_tagged_numbers_l65_65226


namespace arithmetic_sequence_a15_l65_65519

theorem arithmetic_sequence_a15 {a : ℕ → ℝ} (d : ℝ) (a7 a23 : ℝ) 
    (h1 : a 7 = 8) (h2 : a 23 = 22) : 
    a 15 = 15 := 
by
  sorry

end arithmetic_sequence_a15_l65_65519


namespace bird_families_difference_l65_65069

-- Define the conditions
def bird_families_to_africa : ℕ := 47
def bird_families_to_asia : ℕ := 94

-- The proof statement
theorem bird_families_difference : (bird_families_to_asia - bird_families_to_africa = 47) :=
by
  sorry

end bird_families_difference_l65_65069


namespace fraction_product_l65_65239

theorem fraction_product : (1/2) * (3/5) * (7/11) * (4/13) = 84/1430 := by
  sorry

end fraction_product_l65_65239


namespace daniel_sales_tax_l65_65370

theorem daniel_sales_tax :
  let total_cost := 25
  let tax_rate := 0.05
  let tax_free_cost := 18.7
  let tax_paid := 0.3
  exists (taxable_cost : ℝ), 
    18.7 + taxable_cost + 0.05 * taxable_cost = total_cost ∧
    taxable_cost * tax_rate = tax_paid :=
by
  sorry

end daniel_sales_tax_l65_65370


namespace value_of_a_l65_65093

theorem value_of_a (a : ℝ) :
  (∃ (l1 l2 : (ℝ × ℝ × ℝ)),
   l1 = (1, -a, a) ∧ l2 = (3, 1, 2) ∧
   (∃ (m1 m2 : ℝ), 
    (m1 = (1 : ℝ) / a ∧ m2 = -3) ∧ 
    (m1 * m2 = -1))) → a = 3 :=
by sorry

end value_of_a_l65_65093


namespace xiao_ming_speed_difference_l65_65221

noncomputable def distance_school : ℝ := 9.3
noncomputable def time_cycling : ℝ := 0.6
noncomputable def distance_park : ℝ := 0.9
noncomputable def time_walking : ℝ := 0.2

noncomputable def cycling_speed : ℝ := distance_school / time_cycling
noncomputable def walking_speed : ℝ := distance_park / time_walking
noncomputable def speed_difference : ℝ := cycling_speed - walking_speed

theorem xiao_ming_speed_difference : speed_difference = 11 := by
  sorry

end xiao_ming_speed_difference_l65_65221


namespace sum_max_min_a_l65_65152

theorem sum_max_min_a (a : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x - 20 * a^2 < 0)
  (h2 : ∀ x1 x2 : ℝ, x1^2 - a * x1 - 20 * a^2 = 0 → x2^2 - a * x2 - 20 * a^2 = 0 → |x1 - x2| ≤ 9) :
    -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → (1 + -1) = 0 :=
by
  sorry

end sum_max_min_a_l65_65152


namespace probability_two_same_number_l65_65608

theorem probability_two_same_number :
  let rolls := 5
  let sides := 8
  let total_outcomes := sides ^ rolls
  let favorable_outcomes := 8 * 7 * 6 * 5 * 4
  let probability_all_different := (favorable_outcomes : ℚ) / total_outcomes
  let probability_at_least_two_same := 1 - probability_all_different
  probability_at_least_two_same = (3256 : ℚ) / 4096 :=
by 
  sorry

end probability_two_same_number_l65_65608


namespace lcm_inequality_l65_65735

open Nat

-- Assume positive integers n and m, with n > m
theorem lcm_inequality (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n > m) :
  Nat.lcm m n + Nat.lcm (m+1) (n+1) ≥ 2 * m * Real.sqrt n := 
  sorry

end lcm_inequality_l65_65735


namespace square_side_length_l65_65728

theorem square_side_length :
  ∀ (s : ℝ), (∃ w l : ℝ, w = 6 ∧ l = 24 ∧ s^2 = w * l) → s = 12 := by 
  sorry

end square_side_length_l65_65728


namespace pq_plus_four_mul_l65_65258

open Real

theorem pq_plus_four_mul {p q : ℝ} (h1 : (x - 4) * (3 * x + 11) = x ^ 2 - 19 * x + 72) 
  (hpq1 : 2 * p ^ 2 + 18 * p - 116 = 0) (hpq2 : 2 * q ^ 2 + 18 * q - 116 = 0) (hpq_ne : p ≠ q) : 
  (p + 4) * (q + 4) = -78 := 
sorry

end pq_plus_four_mul_l65_65258


namespace sum_of_possible_values_l65_65859

theorem sum_of_possible_values (x y : ℝ) (h : x * y - (2 * x) / (y ^ 3) - (2 * y) / (x ^ 3) = 6) :
  ∃ (a b : ℝ), (a - 2) * (b - 2) = 4 ∧ (a - 2) * (b - 2) = 9 ∧ 4 + 9 = 13 :=
sorry

end sum_of_possible_values_l65_65859


namespace factor_theorem_example_l65_65026

theorem factor_theorem_example (t : ℚ) : (4 * t^3 + 6 * t^2 + 11 * t - 6 = 0) ↔ (t = 1/2) :=
by sorry

end factor_theorem_example_l65_65026


namespace largest_x_satisfying_inequality_l65_65579

theorem largest_x_satisfying_inequality :
  (∃ x : ℝ, 
    (∀ y : ℝ, |(y^2 - 4 * y - 39601)| ≥ |(y^2 + 4 * y - 39601)| → y ≤ x) ∧ 
    |(x^2 - 4 * x - 39601)| ≥ |(x^2 + 4 * x - 39601)|
  ) → x = 199 := 
sorry

end largest_x_satisfying_inequality_l65_65579


namespace no_angle_sat_sin_cos_eq_sin_40_l65_65261

open Real

theorem no_angle_sat_sin_cos_eq_sin_40 :
  ¬∃ α : ℝ, sin α * cos α = sin (40 * π / 180) := 
by 
  sorry

end no_angle_sat_sin_cos_eq_sin_40_l65_65261


namespace part1_part2_l65_65903

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*m*x + 2 - m

theorem part1 (m : ℝ) : (∀ x : ℝ, f x m ≥ x - m*x) → -7 ≤ m ∧ m ≤ 1 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x m) → m ≤ 1 :=
by
  sorry

end part1_part2_l65_65903


namespace min_m_plus_n_l65_65401

theorem min_m_plus_n (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m * n - 2 * m - 3 * n = 20) : 
  m + n = 20 :=
sorry

end min_m_plus_n_l65_65401


namespace jackson_points_l65_65834

theorem jackson_points (team_total_points : ℕ) (other_players_count : ℕ) (other_players_avg_score : ℕ) 
  (total_points_by_team : team_total_points = 72) 
  (total_points_by_others : other_players_count = 7) 
  (avg_points_by_others : other_players_avg_score = 6) :
  ∃ points_by_jackson : ℕ, points_by_jackson = 30 :=
by
  sorry

end jackson_points_l65_65834


namespace range_of_m_l65_65094

-- Define the conditions for p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧ 
  (x₁^2 + 2 * m * x₁ + 1 = 0) ∧ (x₂^2 + 2 * m * x₂ + 1 = 0)

def q (m : ℝ) : Prop := ¬ ∃ x : ℝ, x^2 + 2 * (m-2) * x - 3 * m + 10 = 0

-- The main theorem
theorem range_of_m (m : ℝ) : (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ 
  (m ≤ -2 ∨ (-1 ≤ m ∧ m < 3)) := 
by
  sorry

end range_of_m_l65_65094


namespace binomial_last_three_terms_sum_l65_65158

theorem binomial_last_three_terms_sum (n : ℕ) :
  (1 + n + (n * (n - 1)) / 2 = 79) → n = 12 :=
by
  sorry

end binomial_last_three_terms_sum_l65_65158


namespace turtles_remaining_l65_65989

/-- 
In one nest, there are x baby sea turtles, while in the other nest, there are 2x baby sea turtles.
One-fourth of the turtles in the first nest and three-sevenths of the turtles in the second nest
got swept to the sea. Prove the total number of turtles still on the sand is (53/28)x.
-/
theorem turtles_remaining (x : ℕ) (h1 : ℕ := x) (h2 : ℕ := 2 * x) : ((3/4) * x + (8/7) * (2 * x)) = (53/28) * x :=
by
  sorry

end turtles_remaining_l65_65989


namespace find_number_of_students_l65_65810

variables (n : ℕ)
variables (avg_A avg_B avg_C excl_avg_A excl_avg_B excl_avg_C : ℕ)
variables (new_avg_A new_avg_B new_avg_C : ℕ)
variables (excluded_students : ℕ)

theorem find_number_of_students :
  avg_A = 80 ∧ avg_B = 85 ∧ avg_C = 75 ∧
  excl_avg_A = 20 ∧ excl_avg_B = 25 ∧ excl_avg_C = 15 ∧
  excluded_students = 5 ∧
  new_avg_A = 90 ∧ new_avg_B = 95 ∧ new_avg_C = 85 →
  n = 35 :=
by
  sorry

end find_number_of_students_l65_65810


namespace black_lambs_count_l65_65713

/-- Definition of the total number of lambs. -/
def total_lambs : Nat := 6048

/-- Definition of the number of white lambs. -/
def white_lambs : Nat := 193

/-- Prove that the number of black lambs is 5855. -/
theorem black_lambs_count : total_lambs - white_lambs = 5855 := by
  sorry

end black_lambs_count_l65_65713


namespace gcd_qr_l65_65798

theorem gcd_qr (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 770) : Nat.gcd q r = 70 := sorry

end gcd_qr_l65_65798


namespace extremum_condition_l65_65449

noncomputable def y (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

theorem extremum_condition (a : ℝ) :
  (∃ x : ℝ, (y a x = 0) ∧ ∀ x' > x, y a x' < y a x) → a < -3 :=
by
  sorry

end extremum_condition_l65_65449


namespace monthly_growth_rate_l65_65575

-- Definitions based on the conditions given in the original problem.
def final_height : ℝ := 80
def current_height : ℝ := 20
def months_in_year : ℕ := 12

-- Prove the monthly growth rate.
theorem monthly_growth_rate : (final_height - current_height) / months_in_year = 5 := by
  sorry

end monthly_growth_rate_l65_65575


namespace no_sum_of_two_squares_l65_65409

theorem no_sum_of_two_squares (n : ℤ) (h : n % 4 = 3) : ¬∃ a b : ℤ, n = a^2 + b^2 := 
by
  sorry

end no_sum_of_two_squares_l65_65409


namespace cut_scene_length_proof_l65_65125

noncomputable def original_length : ℕ := 60
noncomputable def final_length : ℕ := 57
noncomputable def cut_scene_length := original_length - final_length

theorem cut_scene_length_proof : cut_scene_length = 3 := by
  sorry

end cut_scene_length_proof_l65_65125


namespace work_completion_l65_65326

noncomputable def efficiency (p q: ℕ) := q = 3 * p / 5

theorem work_completion (p q : ℕ) (h1 : efficiency p q) (h2: p * 24 = 100) :
  2400 / (p + q) = 15 :=
by 
  sorry

end work_completion_l65_65326


namespace set_intersection_union_eq_complement_l65_65636

def A : Set ℝ := {x | 2 * x^2 + x - 3 = 0}
def B : Set ℝ := {i | i^2 ≥ 4}
def complement_C : Set ℝ := {-1, 1, 3/2}

theorem set_intersection_union_eq_complement :
  A ∩ B ∪ complement_C = complement_C :=
by
  sorry

end set_intersection_union_eq_complement_l65_65636


namespace provenance_of_positive_test_l65_65862

noncomputable def pr_disease : ℚ := 1 / 200
noncomputable def pr_no_disease : ℚ := 1 - pr_disease
noncomputable def pr_test_given_disease : ℚ := 1
noncomputable def pr_test_given_no_disease : ℚ := 0.05
noncomputable def pr_test : ℚ := pr_test_given_disease * pr_disease + pr_test_given_no_disease * pr_no_disease
noncomputable def pr_disease_given_test : ℚ := 
  (pr_test_given_disease * pr_disease) / pr_test

theorem provenance_of_positive_test : pr_disease_given_test = 20 / 219 :=
by
  sorry

end provenance_of_positive_test_l65_65862


namespace total_distance_walked_l65_65105

-- Condition 1: Distance in feet
def distance_feet : ℝ := 30

-- Condition 2: Conversion factor from feet to meters
def feet_to_meters : ℝ := 0.3048

-- Condition 3: Number of trips
def trips : ℝ := 4

-- Question: Total distance walked in meters
theorem total_distance_walked :
  distance_feet * feet_to_meters * trips = 36.576 :=
sorry

end total_distance_walked_l65_65105


namespace find_capacity_l65_65788

noncomputable def pool_capacity (V1 V2 q : ℝ) : Prop :=
  V1 = q / 120 ∧ V2 = V1 + 50 ∧ V1 + V2 = q / 48

theorem find_capacity (q : ℝ) : ∃ V1 V2, pool_capacity V1 V2 q → q = 12000 :=
by 
  sorry

end find_capacity_l65_65788


namespace number_of_students_l65_65364

theorem number_of_students (S G : ℕ) (h1 : G = 2 * S / 3) (h2 : 8 = 2 * G / 5) : S = 30 :=
by
  sorry

end number_of_students_l65_65364


namespace rectangular_prism_diagonals_l65_65096

structure RectangularPrism :=
  (faces : ℕ)
  (edges : ℕ)
  (vertices : ℕ)
  (length : ℝ)
  (height : ℝ)
  (width : ℝ)
  (length_ne_height : length ≠ height)
  (height_ne_width : height ≠ width)
  (width_ne_length : width ≠ length)

def diagonals (rp : RectangularPrism) : ℕ :=
  let face_diagonals := 12
  let space_diagonals := 4
  face_diagonals + space_diagonals

theorem rectangular_prism_diagonals (rp : RectangularPrism) :
  rp.faces = 6 →
  rp.edges = 12 →
  rp.vertices = 8 →
  diagonals rp = 16 ∧ 4 = 4 :=
by
  intros
  sorry

end rectangular_prism_diagonals_l65_65096


namespace minimum_value_is_six_l65_65838

noncomputable def minimum_value (m n : ℝ) (h : m > 2 * n) : ℝ :=
  m + (4 * n ^ 2 - 2 * m * n + 9) / (m - 2 * n)

theorem minimum_value_is_six (m n : ℝ) (h : m > 2 * n) : minimum_value m n h = 6 := 
sorry

end minimum_value_is_six_l65_65838


namespace marta_should_buy_84_ounces_l65_65765

/-- Definition of the problem's constants and assumptions --/
def apple_weight : ℕ := 4
def orange_weight : ℕ := 3
def bag_capacity : ℕ := 49
def num_bags : ℕ := 3

-- Marta wants to put the same number of apples and oranges in each bag
def equal_fruit (A O : ℕ) := A = O

-- Each bag should hold up to 49 ounces of fruit
def bag_limit (n : ℕ) := 4 * n + 3 * n ≤ 49

-- Marta's total apple weight based on the number of apples per bag and number of bags
def total_apple_weight (A : ℕ) : ℕ := (A * 3 * 4)

/-- Statement of the proof problem: 
Marta should buy 84 ounces of apples --/
theorem marta_should_buy_84_ounces : total_apple_weight 7 = 84 :=
by
  sorry

end marta_should_buy_84_ounces_l65_65765


namespace hyperbola_center_l65_65922

theorem hyperbola_center :
  ∃ (h : ℝ × ℝ), h = (9 / 2, 2) ∧
  (∃ (x y : ℝ), 9 * x^2 - 81 * x - 16 * y^2 + 64 * y + 144 = 0) :=
  sorry

end hyperbola_center_l65_65922


namespace find_y_intercept_l65_65448

theorem find_y_intercept (a b : ℝ) (h1 : (3 : ℝ) ≠ (7 : ℝ))
  (h2 : -2 = a * 3 + b) (h3 : 14 = a * 7 + b) :
  b = -14 :=
sorry

end find_y_intercept_l65_65448


namespace least_positive_number_of_linear_combination_of_24_20_l65_65956

-- Define the conditions as integers
def problem_statement (x y : ℤ) : Prop := 24 * x + 20 * y = 4

theorem least_positive_number_of_linear_combination_of_24_20 :
  ∃ (x y : ℤ), (24 * x + 20 * y = 4) := 
by
  sorry

end least_positive_number_of_linear_combination_of_24_20_l65_65956


namespace range_of_h_l65_65595

noncomputable def h : ℝ → ℝ
| x => if x = -7 then 0 else 2 * (x - 3)

theorem range_of_h :
  (Set.range h) = Set.univ \ {-20} :=
sorry

end range_of_h_l65_65595


namespace odd_function_symmetry_l65_65446

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then x^2 else sorry

theorem odd_function_symmetry (x : ℝ) (k : ℕ) (h1 : ∀ y, f (-y) = -f y)
  (h2 : ∀ y, f y = f (2 - y)) (h3 : ∀ y, 0 < y ∧ y ≤ 1 → f y = y^2) :
  k = 45 / 4 → f k = -9 / 16 :=
by
  intros _
  sorry

end odd_function_symmetry_l65_65446


namespace trigonometric_ineq_l65_65285

theorem trigonometric_ineq (h₁ : (Real.pi / 4) < 1.5) (h₂ : 1.5 < (Real.pi / 2)) : 
  Real.cos 1.5 < Real.sin 1.5 ∧ Real.sin 1.5 < Real.tan 1.5 := 
sorry

end trigonometric_ineq_l65_65285


namespace ruby_initial_apples_l65_65848

theorem ruby_initial_apples (apples_taken : ℕ) (apples_left : ℕ) (initial_apples : ℕ) 
  (h1 : apples_taken = 55) (h2 : apples_left = 8) (h3 : initial_apples = apples_taken + apples_left) : 
  initial_apples = 63 := 
by
  sorry

end ruby_initial_apples_l65_65848


namespace gross_profit_percentage_l65_65621

theorem gross_profit_percentage :
  ∀ (selling_price wholesale_cost : ℝ),
  selling_price = 28 →
  wholesale_cost = 24.14 →
  (selling_price - wholesale_cost) / wholesale_cost * 100 = 15.99 :=
by
  intros selling_price wholesale_cost h1 h2
  rw [h1, h2]
  norm_num
  sorry

end gross_profit_percentage_l65_65621


namespace only_nonneg_int_solution_l65_65854

theorem only_nonneg_int_solution (x y z : ℕ) (h : x^3 = 3 * y^3 + 9 * z^3) : x = 0 ∧ y = 0 ∧ z = 0 := 
sorry

end only_nonneg_int_solution_l65_65854


namespace framed_painting_ratio_l65_65156

-- Define the conditions and the problem
theorem framed_painting_ratio:
  ∀ (x : ℝ),
    (30 + 2 * x) * (20 + 4 * x) = 1500 →
    (20 + 4 * x) / (30 + 2 * x) = 4 / 5 := 
by sorry

end framed_painting_ratio_l65_65156


namespace percentage_of_other_investment_l65_65289

theorem percentage_of_other_investment (investment total_interest interest_5 interest_other percentage_other : ℝ) 
  (h1 : investment = 18000)
  (h2 : interest_5 = 6000 * 0.05)
  (h3 : total_interest = 660)
  (h4 : percentage_other / 100 * (investment - 6000) = 360) : 
  percentage_other = 3 :=
by
  sorry

end percentage_of_other_investment_l65_65289


namespace function_passes_through_point_l65_65883

theorem function_passes_through_point (a : ℝ) (x y : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  (x = 1 ∧ y = 4) ↔ (y = a^(x-1) + 3) :=
sorry

end function_passes_through_point_l65_65883


namespace parallelogram_side_problem_l65_65211

theorem parallelogram_side_problem (y z : ℝ) (h1 : 4 * z + 1 = 15) (h2 : 3 * y - 2 = 15) :
  y + z = 55 / 6 :=
sorry

end parallelogram_side_problem_l65_65211


namespace inequality_xyz_l65_65789

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x * y / z) + (y * z / x) + (z * x / y) > 2 * ((x ^ 3 + y ^ 3 + z ^ 3) ^ (1 / 3)) :=
by
  sorry

end inequality_xyz_l65_65789


namespace amaya_total_marks_l65_65541

theorem amaya_total_marks 
  (m_a s_a a m m_s : ℕ) 
  (h_music : m_a = 70)
  (h_social_studies : s_a = m_a + 10)
  (h_maths_art_diff : m = a - 20)
  (h_maths_fraction : m = a - 1/10 * a)
  (h_maths_eq_fraction : m = 9/10 * a)
  (h_arts : 9/10 * a = a - 20)
  (h_total : m_a + s_a + a + m = 530) :
  m_a + s_a + a + m = 530 :=
by
  -- Proof to be completed
  sorry

end amaya_total_marks_l65_65541


namespace gas_cost_per_gallon_l65_65557

-- Define the conditions as Lean definitions
def miles_per_gallon : ℕ := 32
def total_miles : ℕ := 336
def total_cost : ℕ := 42

-- Prove the cost of gas per gallon, which is $4 per gallon
theorem gas_cost_per_gallon : total_cost / (total_miles / miles_per_gallon) = 4 :=
by
  sorry

end gas_cost_per_gallon_l65_65557


namespace factorial_expression_l65_65710

theorem factorial_expression :
  7 * (Nat.factorial 7) + 6 * (Nat.factorial 6) + 2 * (Nat.factorial 6) = 41040 := by
  sorry

end factorial_expression_l65_65710


namespace max_students_l65_65306

theorem max_students (A B C : ℕ) (A_left B_left C_left : ℕ)
  (hA : A = 38) (hB : B = 78) (hC : C = 128)
  (hA_left : A_left = 2) (hB_left : B_left = 6) (hC_left : C_left = 20) :
  gcd (A - A_left) (gcd (B - B_left) (C - C_left)) = 36 :=
by {
  sorry
}

end max_students_l65_65306


namespace system_of_equations_l65_65894

theorem system_of_equations (x y : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x + 2 * y = 4) : 
  x + y = 3 :=
sorry

end system_of_equations_l65_65894


namespace remove_terms_to_make_sum_l65_65437

theorem remove_terms_to_make_sum (a b c d e f : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/5) (h₃ : c = 1/7) (h₄ : d = 1/9) (h₅ : e = 1/11) (h₆ : f = 1/13) :
  a + b + c + d + e + f - e - f = 3/2 :=
by
  sorry

end remove_terms_to_make_sum_l65_65437


namespace gcd_128_144_256_l65_65460

theorem gcd_128_144_256 : Nat.gcd (Nat.gcd 128 144) 256 = 128 :=
  sorry

end gcd_128_144_256_l65_65460


namespace time_to_cross_platform_l65_65219

variable (l t p : ℝ) -- Define relevant variables

-- Conditions as definitions in Lean 4
def length_of_train := l
def time_to_pass_man := t
def length_of_platform := p

-- Assume given values in the problem
def cond1 : length_of_train = 186 := by sorry
def cond2 : time_to_pass_man = 8 := by sorry
def cond3 : length_of_platform = 279 := by sorry

-- Statement that represents the target theorem to be proved
theorem time_to_cross_platform (h₁ : length_of_train = 186) (h₂ : time_to_pass_man = 8) (h₃ : length_of_platform = 279) : 
  let speed := length_of_train / time_to_pass_man
  let total_distance := length_of_train + length_of_platform
  let time_to_cross := total_distance / speed
  time_to_cross = 20 :=
by sorry

end time_to_cross_platform_l65_65219


namespace shortest_part_length_l65_65693

theorem shortest_part_length (total_length : ℝ) (r1 r2 r3 : ℝ) (shortest_length : ℝ) :
  total_length = 196.85 → r1 = 3.6 → r2 = 8.4 → r3 = 12 → shortest_length = 29.5275 :=
by
  sorry

end shortest_part_length_l65_65693


namespace evaluate_expression_l65_65335

theorem evaluate_expression :
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  (a + b + c - d) + (a + b - c + d) + (a - b + c + d) + (-a + b + c + d) = 2222 :=
by
  let a := 1
  let b := 10
  let c := 100
  let d := 1000
  sorry

end evaluate_expression_l65_65335


namespace find_N_l65_65584

theorem find_N (N : ℕ) (h : (Real.sqrt 3 - 1)^N = 4817152 - 2781184 * Real.sqrt 3) : N = 16 :=
sorry

end find_N_l65_65584


namespace long_furred_and_brown_dogs_l65_65512

-- Define the total number of dogs.
def total_dogs : ℕ := 45

-- Define the number of long-furred dogs.
def long_furred_dogs : ℕ := 26

-- Define the number of brown dogs.
def brown_dogs : ℕ := 22

-- Define the number of dogs that are neither long-furred nor brown.
def neither_long_furred_nor_brown_dogs : ℕ := 8

-- Prove that the number of dogs that are both long-furred and brown is 11.
theorem long_furred_and_brown_dogs : 
  (long_furred_dogs + brown_dogs) - (total_dogs - neither_long_furred_nor_brown_dogs) = 11 :=
by
  sorry

end long_furred_and_brown_dogs_l65_65512


namespace maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l65_65703

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l65_65703


namespace balazs_missed_number_l65_65360

theorem balazs_missed_number (n k : ℕ) 
  (h1 : n * (n + 1) / 2 = 3000 + k)
  (h2 : 1 ≤ k)
  (h3 : k < n) : k = 3 := by
  sorry

end balazs_missed_number_l65_65360


namespace gcd_84_120_eq_12_l65_65715

theorem gcd_84_120_eq_12 : Int.gcd 84 120 = 12 := by
  sorry

end gcd_84_120_eq_12_l65_65715


namespace sum_of_repeating_decimals_l65_65692

noncomputable def x := (2 : ℚ) / (3 : ℚ)
noncomputable def y := (5 : ℚ) / (11 : ℚ)

theorem sum_of_repeating_decimals : x + y = (37 : ℚ) / (33 : ℚ) :=
by {
  sorry
}

end sum_of_repeating_decimals_l65_65692


namespace original_price_of_stamp_l65_65130

theorem original_price_of_stamp (original_price : ℕ) (h : original_price * (1 / 5 : ℚ) = 6) : original_price = 30 :=
by
  sorry

end original_price_of_stamp_l65_65130


namespace octagon_side_length_eq_l65_65776

theorem octagon_side_length_eq (AB BC : ℝ) (AE FB s : ℝ) :
  AE = FB → AE < 5 → AB = 10 → BC = 12 →
  s = -11 + Real.sqrt 242 →
  EF = (10.5 - (Real.sqrt 242) / 2) :=
by
  -- Identified parameters and included all conditions from step a)
  intros h1 h2 h3 h4 h5
  -- statement of the theorem to be proven
  let EF := (10.5 - (Real.sqrt 242) / 2)
  sorry  -- placeholder for proof

end octagon_side_length_eq_l65_65776


namespace expression_equals_base10_l65_65844

-- Define numbers in various bases
def base7ToDec (n : ℕ) : ℕ := 1 * (7^2) + 6 * (7^1) + 5 * (7^0)
def base2ToDec (n : ℕ) : ℕ := 1 * (2^1) + 1 * (2^0)
def base6ToDec (n : ℕ) : ℕ := 1 * (6^2) + 2 * (6^1) + 1 * (6^0)
def base3ToDec (n : ℕ) : ℕ := 2 * (3^1) + 1 * (3^0)

-- Prove the given expression equals 39 in base 10
theorem expression_equals_base10 :
  (base7ToDec 165 / base2ToDec 11) + (base6ToDec 121 / base3ToDec 21) = 39 :=
by
  -- Convert the base n numbers to base 10
  let num1 := base7ToDec 165
  let den1 := base2ToDec 11
  let num2 := base6ToDec 121
  let den2 := base3ToDec 21
  
  -- Simplify the expression (skipping actual steps for brevity, replaced by sorry)
  sorry

end expression_equals_base10_l65_65844


namespace each_dog_food_intake_l65_65782

theorem each_dog_food_intake (total_food : ℝ) (dog_count : ℕ) (equal_amount : ℝ) : total_food = 0.25 → dog_count = 2 → (total_food / dog_count) = equal_amount → equal_amount = 0.125 :=
by
  intros h1 h2 h3
  sorry

end each_dog_food_intake_l65_65782


namespace solution_set_of_inequality_l65_65721

theorem solution_set_of_inequality :
  {x : ℝ | -1 < x ∧ x < 2} = {x : ℝ | (x - 2) / (x + 1) < 0} :=
sorry

end solution_set_of_inequality_l65_65721


namespace chosen_number_is_30_l65_65671

theorem chosen_number_is_30 (x : ℤ) 
  (h1 : 8 * x - 138 = 102) : x = 30 := 
sorry

end chosen_number_is_30_l65_65671


namespace perpendicular_lines_intersect_at_point_l65_65478

theorem perpendicular_lines_intersect_at_point :
  ∀ (d k : ℝ), 
  (∀ x y, 3 * x - 4 * y = d ↔ 8 * x + k * y = d) → 
  (∃ x y, x = 2 ∧ y = -3 ∧ 3 * x - 4 * y = d ∧ 8 * x + k * y = d) → 
  d = -2 :=
by sorry

end perpendicular_lines_intersect_at_point_l65_65478


namespace intersection_point_l65_65599

theorem intersection_point (k : ℚ) :
  (∃ x y : ℚ, x + k * y = 0 ∧ 2 * x + 3 * y + 8 = 0 ∧ x - y - 1 = 0) ↔ (k = -1/2) :=
by sorry

end intersection_point_l65_65599


namespace three_lines_pass_through_point_and_intersect_parabola_l65_65657

-- Define the point (0,1)
def point : ℝ × ℝ := (0, 1)

-- Define the parabola y^2 = 4x as a set of points
def parabola (p : ℝ × ℝ) : Prop :=
  (p.snd)^2 = 4 * (p.fst)

-- Define the condition for the line passing through (0,1)
def line_through_point (line_eq : ℝ → ℝ) : Prop :=
  line_eq 0 = 1

-- Define the condition for the line intersecting the parabola at only one point
def intersects_once (line_eq : ℝ → ℝ) : Prop :=
  ∃! x : ℝ, parabola (x, line_eq x)

-- The main theorem statement
theorem three_lines_pass_through_point_and_intersect_parabola :
  ∃ (f1 f2 f3 : ℝ → ℝ), 
    line_through_point f1 ∧ line_through_point f2 ∧ line_through_point f3 ∧
    intersects_once f1 ∧ intersects_once f2 ∧ intersects_once f3 ∧
    (∀ (f : ℝ → ℝ), (line_through_point f ∧ intersects_once f) ->
      (f = f1 ∨ f = f2 ∨ f = f3)) :=
sorry

end three_lines_pass_through_point_and_intersect_parabola_l65_65657


namespace students_prefer_dogs_l65_65533

theorem students_prefer_dogs (total_students : ℕ) (perc_dogs_vg perc_dogs_mv : ℕ) (h_total: total_students = 30)
  (h_perc_dogs_vg: perc_dogs_vg = 50) (h_perc_dogs_mv: perc_dogs_mv = 10) :
  total_students * perc_dogs_vg / 100 + total_students * perc_dogs_mv / 100 = 18 := by
  sorry

end students_prefer_dogs_l65_65533


namespace retirement_year_2020_l65_65756

-- Given conditions
def femaleRetirementAge := 55
def initialRetirementYear (birthYear : ℕ) := birthYear + femaleRetirementAge
def delayedRetirementYear (baseYear additionalYears : ℕ) := baseYear + additionalYears

def postponementStep := 3
def delayStartYear := 2018
def retirementAgeIn2045 := 65
def retirementYear (birthYear : ℕ) : ℕ :=
  let originalRetirementYear := initialRetirementYear birthYear
  let delayYears := ((originalRetirementYear - delayStartYear) / postponementStep) + 1
  delayedRetirementYear originalRetirementYear delayYears

-- Main theorem to prove
theorem retirement_year_2020 : retirementYear 1964 = 2020 := sorry

end retirement_year_2020_l65_65756


namespace circle_area_from_equation_l65_65356

theorem circle_area_from_equation :
  (∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = -9) →
  ∃ (r : ℝ), (r = 2) ∧
    (∃ (A : ℝ), A = π * r^2 ∧ A = 4 * π) :=
by {
  -- Conditions included as hypothesis
  sorry -- Proof to be provided here
}

end circle_area_from_equation_l65_65356


namespace partial_fraction_sum_eq_zero_l65_65274

theorem partial_fraction_sum_eq_zero (A B C D E : ℂ) :
  (∀ x : ℂ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ 4 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x - 4)) =
      A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x - 4)) →
  A + B + C + D + E = 0 :=
by
  sorry

end partial_fraction_sum_eq_zero_l65_65274


namespace tax_rate_equals_65_l65_65528

def tax_rate_percentage := 65
def tax_rate_per_dollars (rate_percentage : ℕ) : ℕ :=
  (rate_percentage / 100) * 100

theorem tax_rate_equals_65 :
  tax_rate_per_dollars tax_rate_percentage = 65 := by
  sorry

end tax_rate_equals_65_l65_65528


namespace gravel_amount_l65_65524

theorem gravel_amount (total_material sand gravel : ℝ) 
  (h1 : total_material = 14.02) 
  (h2 : sand = 8.11) 
  (h3 : gravel = total_material - sand) : 
  gravel = 5.91 :=
  sorry

end gravel_amount_l65_65524


namespace inequality_solution_l65_65691

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  sorry

end inequality_solution_l65_65691


namespace reciprocal_neg_one_over_2011_l65_65214

theorem reciprocal_neg_one_over_2011 : 1 / (- (1 / 2011)) = -2011 :=
by
  sorry

end reciprocal_neg_one_over_2011_l65_65214


namespace emily_and_eli_probability_l65_65310

noncomputable def probability_same_number : ℚ :=
  let count_multiples (n k : ℕ) := (k - 1) / n
  let emily_count := count_multiples 20 250
  let eli_count := count_multiples 30 250
  let common_lcm := Nat.lcm 20 30
  let common_count := count_multiples common_lcm 250
  common_count / (emily_count * eli_count : ℚ)

theorem emily_and_eli_probability :
  let probability := probability_same_number
  probability = 1 / 24 :=
by
  sorry

end emily_and_eli_probability_l65_65310


namespace jamies_mother_twice_age_l65_65887

theorem jamies_mother_twice_age (y : ℕ) :
  ∀ (jamie_age_2010 mother_age_2010 : ℕ), 
  jamie_age_2010 = 10 → 
  mother_age_2010 = 5 * jamie_age_2010 → 
  mother_age_2010 + y = 2 * (jamie_age_2010 + y) → 
  2010 + y = 2040 :=
by
  intros jamie_age_2010 mother_age_2010 h_jamie h_mother h_eq
  sorry

end jamies_mother_twice_age_l65_65887


namespace swimmer_speed_in_still_water_l65_65860

-- Define the various given conditions as constants in Lean
def swimmer_distance : ℝ := 3
def river_current_speed : ℝ := 1.7
def time_taken : ℝ := 2.3076923076923075

-- Define what we need to prove: the swimmer's speed in still water
theorem swimmer_speed_in_still_water (v : ℝ) :
  swimmer_distance = (v - river_current_speed) * time_taken → 
  v = 3 := by
  sorry

end swimmer_speed_in_still_water_l65_65860


namespace sally_total_expense_l65_65992

-- Definitions based on the problem conditions
def peaches_price_after_coupon : ℝ := 12.32
def peaches_coupon : ℝ := 3.00
def cherries_weight : ℝ := 2.00
def cherries_price_per_kg : ℝ := 11.54
def apples_weight : ℝ := 4.00
def apples_price_per_kg : ℝ := 5.00
def apples_discount_percentage : ℝ := 0.15
def oranges_count : ℝ := 6.00
def oranges_price_per_unit : ℝ := 1.25
def oranges_promotion : ℝ := 3.00 -- Buy 2, get 1 free means she pays for 4 out of 6

-- Calculation of the total expense
def total_expense : ℝ :=
  (peaches_price_after_coupon + peaches_coupon) + 
  (cherries_weight * cherries_price_per_kg) + 
  ((apples_weight * apples_price_per_kg) * (1 - apples_discount_percentage)) +
  (4 * oranges_price_per_unit)

-- Statement to verify total expense
theorem sally_total_expense : total_expense = 60.40 := by
  sorry

end sally_total_expense_l65_65992


namespace decreasing_interval_l65_65888

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 4)

theorem decreasing_interval :
  ∀ x ∈ Set.Icc 0 (2 * Real.pi), x ∈ Set.Icc (3 * Real.pi / 4) (2 * Real.pi) ↔ (∀ ε > 0, f x > f (x + ε)) := 
sorry

end decreasing_interval_l65_65888


namespace system_solution_l65_65499

theorem system_solution (a x0 : ℝ) (h : a ≠ 0) 
  (h1 : 3 * x0 + 2 * x0 = 15 * a) 
  (h2 : 1 / a * x0 + x0 = 9) 
  : x0 = 6 ∧ a = 2 :=
by {
  sorry
}

end system_solution_l65_65499


namespace farmer_plough_rate_l65_65071

theorem farmer_plough_rate (x : ℝ) (h1 : 85 * ((1400 / x) + 2) + 40 = 1400) : x = 100 :=
by
  sorry

end farmer_plough_rate_l65_65071


namespace parabola_distance_l65_65242

theorem parabola_distance (p : ℝ) (hp : 0 < p) (hf : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  dist P (0, p / 2) = 16) (hx : ∀ P : ℝ × ℝ, P ∈ {Q : ℝ × ℝ | Q.1^2 = 2 * p * Q.2} →
  P.2 = 10) : p = 12 :=
sorry

end parabola_distance_l65_65242


namespace king_plan_feasibility_l65_65229

-- Create a predicate for the feasibility of the king's plan
def feasible (n : ℕ) : Prop :=
  (n = 6 ∧ true) ∨ (n = 2004 ∧ false)

theorem king_plan_feasibility :
  ∀ n : ℕ, feasible n :=
by
  intro n
  sorry

end king_plan_feasibility_l65_65229


namespace calculation_result_l65_65763

theorem calculation_result:
  5 * 301 + 4 * 301 + 3 * 301 + 300 = 3912 :=
by
  sorry

end calculation_result_l65_65763


namespace simplify_fraction_l65_65733

theorem simplify_fraction : 
  (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) :=
by
  -- Proof will go here
  sorry

end simplify_fraction_l65_65733


namespace sum_proof_l65_65839

theorem sum_proof (X Y : ℝ) (hX : 0.45 * X = 270) (hY : 0.35 * Y = 210) : 
  (0.75 * X) + (0.55 * Y) = 780 := by
  sorry

end sum_proof_l65_65839


namespace least_value_of_a_l65_65709

theorem least_value_of_a (a : ℝ) (h : a^2 - 12 * a + 35 ≤ 0) : 5 ≤ a :=
by {
  sorry
}

end least_value_of_a_l65_65709


namespace smallest_n_exists_l65_65036

theorem smallest_n_exists :
  ∃ (a1 a2 a3 a4 a5 : ℤ), a1 + a2 + a3 + a4 + a5 = 1990 ∧ a1 * a2 * a3 * a4 * a5 = 1990 :=
sorry

end smallest_n_exists_l65_65036


namespace potatoes_left_l65_65466

theorem potatoes_left (initial_potatoes : ℕ) (potatoes_for_salads : ℕ) (potatoes_for_mashed : ℕ)
  (h1 : initial_potatoes = 52)
  (h2 : potatoes_for_salads = 15)
  (h3 : potatoes_for_mashed = 24) :
  initial_potatoes - (potatoes_for_salads + potatoes_for_mashed) = 13 := by
  sorry

end potatoes_left_l65_65466


namespace equal_binomial_terms_l65_65257

theorem equal_binomial_terms (p q : ℝ) (h1 : 0 < p) (h2 : 0 < q) (h3 : p + q = 1)
    (h4 : 55 * p^9 * q^2 = 165 * p^8 * q^3) : p = 3 / 4 :=
by
  sorry

end equal_binomial_terms_l65_65257


namespace nonnegative_integers_with_abs_value_less_than_4_l65_65134

theorem nonnegative_integers_with_abs_value_less_than_4 :
  {n : ℕ | abs (n : ℤ) < 4} = {0, 1, 2, 3} :=
by {
  sorry
}

end nonnegative_integers_with_abs_value_less_than_4_l65_65134


namespace find_width_of_rect_box_l65_65222

-- Define the dimensions of the wooden box in meters
def wooden_box_length_m : ℕ := 8
def wooden_box_width_m : ℕ := 7
def wooden_box_height_m : ℕ := 6

-- Define the dimensions of the rectangular boxes in centimeters (with unknown width W)
def rect_box_length_cm : ℕ := 8
def rect_box_height_cm : ℕ := 6

-- Define the maximum number of rectangular boxes
def max_boxes : ℕ := 1000000

-- Define the constraint that the total volume of the boxes should not exceed the volume of the wooden box
theorem find_width_of_rect_box (W : ℕ) (wooden_box_volume : ℕ := (wooden_box_length_m * 100) * (wooden_box_width_m * 100) * (wooden_box_height_m * 100)) : 
  (rect_box_length_cm * W * rect_box_height_cm) * max_boxes = wooden_box_volume → W = 7 :=
by
  sorry

end find_width_of_rect_box_l65_65222


namespace circle_equation_standard_form_l65_65998

theorem circle_equation_standard_form (x y : ℝ) :
  (∃ (center : ℝ × ℝ), center.1 = -1 ∧ center.2 = 2 * center.1 ∧ (center.2 = -2) ∧ (center.1 + 1)^2 + center.2^2 = 4 ∧ (center.1 = -1) ∧ (center.2 = -2)) ->
  (x + 1)^2 + (y + 2)^2 = 4 :=
sorry

end circle_equation_standard_form_l65_65998


namespace largest_x_value_l65_65967

-- Definition of the equation
def equation (x : ℚ) : Prop := 3 * (9 * x^2 + 10 * x + 11) = x * (9 * x - 45)

-- The problem to prove is that the largest value of x satisfying the equation is -1/2
theorem largest_x_value : ∃ x : ℚ, equation x ∧ ∀ y : ℚ, equation y → y ≤ -1/2 := by
  sorry

end largest_x_value_l65_65967


namespace max_covered_squares_by_tetromino_l65_65805

-- Definition of the grid size
def grid_size := (5, 5)

-- Definition of S-Tetromino (Z-Tetromino) coverage covering four contiguous squares
def is_STetromino (coords: List (Nat × Nat)) : Prop := 
  coords.length = 4 ∧ ∃ (x y : Nat), coords = [(x, y), (x, y+1), (x+1, y+1), (x+1, y+2)]

-- Definition of the coverage constraint
def no_more_than_two_tiles (cover: List (Nat × Nat)) : Prop :=
  ∀ (coord: Nat × Nat), cover.count coord ≤ 2

-- Definition of the total tiled squares covered by at least one tile
def tiles_covered (cover: List (Nat × Nat)) : Nat := 
  cover.toFinset.card 

-- Definition of the problem using proof equivalence
theorem max_covered_squares_by_tetromino
  (cover: List (List (Nat × Nat)))
  (H_tiles: ∀ t, t ∈ cover → is_STetromino t)
  (H_coverage: no_more_than_two_tiles (cover.join)) :
  tiles_covered (cover.join) = 24 :=
sorry 

end max_covered_squares_by_tetromino_l65_65805


namespace jersey_sum_adjacent_gt_17_l65_65095

theorem jersey_sum_adjacent_gt_17 (a : ℕ → ℕ) (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ n, 0 < a n ∧ a n ≤ 10) (h_circle : ∀ n, a n = a (n % 10)) :
  ∃ n, a n + a (n+1) + a (n+2) > 17 :=
by
  sorry

end jersey_sum_adjacent_gt_17_l65_65095


namespace monotonic_intervals_a_eq_1_range_of_a_no_zero_points_in_interval_l65_65899

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

theorem monotonic_intervals_a_eq_1 :
  ∀ x : ℝ, (0 < x ∧ x ≤ 2 → (f x 1) < (f 2 1)) ∧ 
           (2 ≤ x → (f x 1) > (f 2 1)) :=
by
  sorry

theorem range_of_a_no_zero_points_in_interval :
  ∀ a : ℝ, (∀ x : ℝ, (0 < x ∧ x < 1/3) → ((2 - a) * (x - 1) - 2 * Real.log x) > 0) ↔ 2 - 3 * Real.log 3 ≤ a :=
by
  sorry

end monotonic_intervals_a_eq_1_range_of_a_no_zero_points_in_interval_l65_65899


namespace geometric_sequence_common_ratio_l65_65469

theorem geometric_sequence_common_ratio (a : ℕ → ℤ) (q : ℤ)  
  (h1 : a 1 = 3) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h3 : 4 * a 1 + a 3 = 4 * a 2) : 
  q = 2 := 
by {
  -- Proof is omitted here
  sorry
}

end geometric_sequence_common_ratio_l65_65469


namespace solve_eq1_solve_eq2_l65_65452

theorem solve_eq1 : (2 * (x - 3) = 3 * x * (x - 3)) → (x = 3 ∨ x = 2 / 3) :=
by
  intro h
  sorry

theorem solve_eq2 : (2 * x ^ 2 - 3 * x + 1 = 0) → (x = 1 ∨ x = 1 / 2) :=
by
  intro h
  sorry

end solve_eq1_solve_eq2_l65_65452


namespace math_proof_problem_l65_65487

variable {a b c : ℝ}

theorem math_proof_problem (h₁ : a * b * c * (a + b) * (b + c) * (c + a) ≠ 0)
  (h₂ : (a + b + c) * (1 / a + 1 / b + 1 / c) = 1007 / 1008) :
  (a * b / ((a + c) * (b + c)) + b * c / ((b + a) * (c + a)) + c * a / ((c + b) * (a + b))) = 2017 := 
sorry

end math_proof_problem_l65_65487


namespace fourth_arithmetic_sequence_equation_l65_65846

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
variables (a : ℕ → ℝ) (h : is_arithmetic_sequence a)
variable (h1 : a 1 - 2 * a 2 + a 3 = 0)
variable (h2 : a 1 - 3 * a 2 + 3 * a 3 - a 4 = 0)
variable (h3 : a 1 - 4 * a 2 + 6 * a 3 - 4 * a 4 + a 5 = 0)

-- Theorem statement to be proven
theorem fourth_arithmetic_sequence_equation : a 1 - 5 * a 2 + 10 * a 3 - 10 * a 4 + 5 * a 5 - a 6 = 0 :=
by
  sorry

end fourth_arithmetic_sequence_equation_l65_65846


namespace parabola_vertex_relationship_l65_65475

theorem parabola_vertex_relationship (m x y : ℝ) :
  (y = x^2 - 2*m*x + 2*m^2 - 3*m + 1) → (y = x^2 - 3*x + 1) :=
by
  intro h
  sorry

end parabola_vertex_relationship_l65_65475


namespace combined_value_of_cookies_l65_65321

theorem combined_value_of_cookies
  (total_boxes_sold : ℝ)
  (plain_boxes_sold : ℝ)
  (price_chocolate_chip : ℝ)
  (price_plain : ℝ)
  (h1 : total_boxes_sold = 1585)
  (h2 : plain_boxes_sold = 793.375)
  (h3 : price_chocolate_chip = 1.25)
  (h4 : price_plain = 0.75) :
  (plain_boxes_sold * price_plain) + ((total_boxes_sold - plain_boxes_sold) * price_chocolate_chip) = 1584.5625 :=
by
  sorry

end combined_value_of_cookies_l65_65321


namespace jerry_total_miles_l65_65240

def monday : ℕ := 15
def tuesday : ℕ := 18
def wednesday : ℕ := 25
def thursday : ℕ := 12
def friday : ℕ := 10

def total : ℕ := monday + tuesday + wednesday + thursday + friday

theorem jerry_total_miles : total = 80 := by
  sorry

end jerry_total_miles_l65_65240


namespace max_prime_factors_of_c_l65_65042

-- Definitions of conditions
variables (c d : ℕ)
variable (prime_factor_count : ℕ → ℕ)
variable (gcd : ℕ → ℕ → ℕ)
variable (lcm : ℕ → ℕ → ℕ)

-- Conditions
axiom gcd_condition : prime_factor_count (gcd c d) = 11
axiom lcm_condition : prime_factor_count (lcm c d) = 44
axiom fewer_prime_factors : prime_factor_count c < prime_factor_count d

-- Proof statement
theorem max_prime_factors_of_c : prime_factor_count c ≤ 27 := 
sorry

end max_prime_factors_of_c_l65_65042


namespace find_larger_number_l65_65087

variable (x y : ℕ)

theorem find_larger_number (h1 : x = 7) (h2 : x + y = 15) : y = 8 := by
  sorry

end find_larger_number_l65_65087


namespace geometric_sequence_a6_l65_65480

theorem geometric_sequence_a6
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : a 1 = 1)
  (S3 : S 3 = 7 / 4)
  (sum_S3 : S 3 = a 1 + a 1 * a 2 + a 1 * (a 2)^2) :
  a 6 = 1 / 32 := by
  sorry

end geometric_sequence_a6_l65_65480


namespace biology_physics_ratio_l65_65391

theorem biology_physics_ratio (boys_bio : ℕ) (girls_bio : ℕ) (total_bio : ℕ) (total_phys : ℕ) 
  (h1 : boys_bio = 25) 
  (h2 : girls_bio = 3 * boys_bio) 
  (h3 : total_bio = boys_bio + girls_bio) 
  (h4 : total_phys = 200) : 
  total_bio / total_phys = 1 / 2 :=
by
  sorry

end biology_physics_ratio_l65_65391


namespace gcd_18_30_45_l65_65191

-- Define the conditions
def a := 18
def b := 30
def c := 45

-- Prove that the gcd of a, b, and c is 3
theorem gcd_18_30_45 : Nat.gcd (Nat.gcd a b) c = 3 :=
by
  -- Skip the proof itself
  sorry

end gcd_18_30_45_l65_65191


namespace minimize_base_side_length_l65_65184

theorem minimize_base_side_length (V : ℝ) (a h : ℝ) 
  (volume_eq : V = a ^ 2 * h) (V_given : V = 256) (h_eq : h = 256 / (a ^ 2)) :
  a = 8 :=
by
  -- Recognize that for a given volume, making it a cube minimizes the surface area.
  -- As the volume of the cube a^3 = 256, solving for a gives 8.
  -- a := (256:ℝ) ^ (1/3:ℝ)
  sorry

end minimize_base_side_length_l65_65184


namespace total_cost_cardshop_l65_65957

theorem total_cost_cardshop : 
  let price_A := 1.25
  let price_B := 1.50
  let price_C := 2.25
  let price_D := 2.50
  let discount_10_percent := 0.10
  let discount_15_percent := 0.15
  let sales_tax_rate := 0.06
  let qty_A := 6
  let qty_B := 4
  let qty_C := 10
  let qty_D := 12
  let total_before_discounts := qty_A * price_A + qty_B * price_B + qty_C * price_C + qty_D * price_D
  let discount_A := if qty_A >= 5 then qty_A * price_A * discount_10_percent else 0
  let discount_C := if qty_C >= 8 then qty_C * price_C * discount_15_percent else 0
  let discount_D := if qty_D >= 8 then qty_D * price_D * discount_15_percent else 0
  let total_discounts := discount_A + discount_C + discount_D
  let total_after_discounts := total_before_discounts - total_discounts
  let tax := total_after_discounts * sales_tax_rate
  let total_cost := total_after_discounts + tax
  total_cost = 60.82
:= 
by
  have price_A : ℝ := 1.25
  have price_B : ℝ := 1.50
  have price_C : ℝ := 2.25
  have price_D : ℝ := 2.50
  have discount_10_percent : ℝ := 0.10
  have discount_15_percent : ℝ := 0.15
  have sales_tax_rate : ℝ := 0.06
  have qty_A : ℕ := 6
  have qty_B : ℕ := 4
  have qty_C : ℕ := 10
  have qty_D : ℕ := 12
  let total_before_discounts := qty_A * price_A + qty_B * price_B + qty_C * price_C + qty_D * price_D
  let discount_A := if qty_A >= 5 then qty_A * price_A * discount_10_percent else 0
  let discount_C := if qty_C >= 8 then qty_C * price_C * discount_15_percent else 0
  let discount_D := if qty_D >= 8 then qty_D * price_D * discount_15_percent else 0
  let total_discounts := discount_A + discount_C + discount_D
  let total_after_discounts := total_before_discounts - total_discounts
  let tax := total_after_discounts * sales_tax_rate
  let total_cost := total_after_discounts + tax
  sorry

end total_cost_cardshop_l65_65957


namespace geometric_series_common_ratio_l65_65394

theorem geometric_series_common_ratio (a r S : ℝ) (hS : S = a / (1 - r)) (hRemove : (ar^4) / (1 - r) = S / 81) :
  r = 1/3 :=
sorry

end geometric_series_common_ratio_l65_65394


namespace value_in_parentheses_l65_65645

theorem value_in_parentheses (x : ℝ) (h : x / Real.sqrt 18 = Real.sqrt 2) : x = 6 :=
sorry

end value_in_parentheses_l65_65645


namespace perfect_square_trinomial_iff_l65_65852

theorem perfect_square_trinomial_iff (m : ℤ) :
  (∃ a b : ℤ, 4 = a^2 ∧ 121 = b^2 ∧ (4 = a^2 ∧ 121 = b^2) ∧ m = 2 * a * b ∨ m = -2 * a * b) ↔ (m = 44 ∨ m = -44) :=
by sorry

end perfect_square_trinomial_iff_l65_65852


namespace work_done_by_gas_l65_65747

theorem work_done_by_gas (n : ℕ) (R T0 Pa : ℝ) (V0 : ℝ) (W : ℝ) :
  -- Conditions
  n = 1 ∧
  R = 8.314 ∧
  T0 = 320 ∧
  Pa * V0 = n * R * T0 ∧
  -- Question Statement and Correct Answer
  W = Pa * V0 / 2 →
  W = 665 :=
by sorry

end work_done_by_gas_l65_65747


namespace solve_for_x_l65_65120

theorem solve_for_x (x : ℝ) (hx : x ≠ 0) (h : (5*x)^10 = (10*x)^5) : x = 2/5 :=
sorry

end solve_for_x_l65_65120


namespace football_game_cost_l65_65099

theorem football_game_cost :
  ∀ (total_spent strategy_game_cost batman_game_cost football_game_cost : ℝ),
  total_spent = 35.52 →
  strategy_game_cost = 9.46 →
  batman_game_cost = 12.04 →
  total_spent - strategy_game_cost - batman_game_cost = football_game_cost →
  football_game_cost = 13.02 :=
by
  intros total_spent strategy_game_cost batman_game_cost football_game_cost h1 h2 h3 h4
  have : football_game_cost = 13.02 := sorry
  exact this

end football_game_cost_l65_65099


namespace simplify_fraction_l65_65062

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l65_65062


namespace geometric_sequence_product_l65_65419

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geom : ∀ n : ℕ, a (n + 1) = r * a n)
variable (h_condition : a 5 * a 14 = 5)

theorem geometric_sequence_product :
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  sorry

end geometric_sequence_product_l65_65419


namespace find_n_l65_65078

theorem find_n :
  ∃ n : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = n ^ 5 ∧ 
  (∀ m : ℕ, 120 ^ 5 + 105 ^ 5 + 78 ^ 5 + 33 ^ 5 = m ^ 5 → m = 144) :=
by
  sorry

end find_n_l65_65078


namespace cody_tickets_l65_65027

theorem cody_tickets (initial_tickets spent_tickets won_tickets : ℕ) (h_initial : initial_tickets = 49) (h_spent : spent_tickets = 25) (h_won : won_tickets = 6) : initial_tickets - spent_tickets + won_tickets = 30 := 
by 
  sorry

end cody_tickets_l65_65027


namespace number_added_multiplied_l65_65766

theorem number_added_multiplied (x : ℕ) (h : (7/8 : ℚ) * x = 28) : ((x + 16) * (5/16 : ℚ)) = 15 :=
by
  sorry

end number_added_multiplied_l65_65766


namespace spotted_mushrooms_ratio_l65_65167

theorem spotted_mushrooms_ratio 
  (total_mushrooms : ℕ) 
  (gilled_mushrooms : ℕ) 
  (spotted_mushrooms : ℕ) 
  (total_mushrooms_eq : total_mushrooms = 30) 
  (gilled_mushrooms_eq : gilled_mushrooms = 3) 
  (spots_and_gills_exclusive : ∀ x, x = spotted_mushrooms ∨ x = gilled_mushrooms) : 
  spotted_mushrooms / gilled_mushrooms = 9 := 
by
  sorry

end spotted_mushrooms_ratio_l65_65167


namespace problem_l65_65806

def p : Prop := 0 % 2 = 0
def q : Prop := ¬(3 % 2 = 0)

theorem problem : p ∨ q :=
by
  sorry

end problem_l65_65806


namespace find_k_if_lines_parallel_l65_65851

theorem find_k_if_lines_parallel (k : ℝ) : (∀ x y, y = 5 * x + 3 → y = (3 * k) * x + 7) → k = 5 / 3 :=
by {
  sorry
}

end find_k_if_lines_parallel_l65_65851


namespace solve_ff_eq_x_l65_65510

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x - 5

theorem solve_ff_eq_x (x : ℝ) :
  (f (f x) = x) ↔ 
  (x = (5 + 3 * Real.sqrt 5) / 2 ∨
   x = (5 - 3 * Real.sqrt 5) / 2 ∨
   x = (3 + Real.sqrt 41) / 2 ∨ 
   x = (3 - Real.sqrt 41) / 2) := 
by
  sorry

end solve_ff_eq_x_l65_65510


namespace solve_equation_l65_65284

noncomputable def a := 3 + Real.sqrt 8
noncomputable def b := 3 - Real.sqrt 8

theorem solve_equation (x : ℝ) :
  (Real.sqrt (a^x) + Real.sqrt (b^x) = 6) ↔ (x = 2 ∨ x = -2) := 
  by
  sorry

end solve_equation_l65_65284


namespace abs_neg_three_l65_65111

theorem abs_neg_three : abs (-3) = 3 := 
by
  -- according to the definition of absolute value, abs(-3) = 3
  sorry

end abs_neg_three_l65_65111


namespace find_m_value_l65_65166

noncomputable def fx (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem find_m_value (m : ℝ) : (∀ x > 0, fx m x > fx m 0) → m = 2 := by
  sorry

end find_m_value_l65_65166


namespace longest_collection_has_more_pages_l65_65066

noncomputable def miles_pages_per_inch := 5
noncomputable def daphne_pages_per_inch := 50
noncomputable def miles_height_inches := 240
noncomputable def daphne_height_inches := 25

noncomputable def miles_total_pages := miles_height_inches * miles_pages_per_inch
noncomputable def daphne_total_pages := daphne_height_inches * daphne_pages_per_inch

theorem longest_collection_has_more_pages :
  max miles_total_pages daphne_total_pages = 1250 := by
  -- Skip the proof
  sorry

end longest_collection_has_more_pages_l65_65066


namespace max_of_a_l65_65255

theorem max_of_a (a b c d : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c ≥ d) (h4 : d > 0)
  (h5 : a + b + c + d = 4) (h6 : a^2 + b^2 + c^2 + d^2 = 8) : a ≤ 1 + Real.sqrt 3 :=
sorry

end max_of_a_l65_65255


namespace new_ratio_boarders_to_day_students_l65_65926

-- Given conditions
def initial_ratio_boarders_to_day_students : ℚ := 2 / 5
def initial_boarders : ℕ := 120
def new_boarders : ℕ := 30

-- Derived definitions
def initial_day_students : ℕ :=
  (initial_boarders * (5 : ℕ)) / 2

def total_boarders : ℕ := initial_boarders + new_boarders
def total_day_students : ℕ := initial_day_students

-- Theorem to prove the new ratio
theorem new_ratio_boarders_to_day_students : total_boarders / total_day_students = 1 / 2 :=
  sorry

end new_ratio_boarders_to_day_students_l65_65926


namespace fraction_compare_l65_65173

theorem fraction_compare : 
  let a := (1 : ℝ) / 4
  let b := 250000025 / (10^9)
  let diff := a - b
  diff = (1 : ℝ) / (4 * 10^7) :=
by
  sorry

end fraction_compare_l65_65173


namespace solve_y_l65_65052

theorem solve_y :
  ∀ y : ℚ, 6 * (4 * y - 1) - 3 = 3 * (2 - 5 * y) ↔ y = 5 / 13 :=
by
  sorry

end solve_y_l65_65052


namespace make_polynomial_perfect_square_l65_65724

theorem make_polynomial_perfect_square (m : ℝ) :
  m = 196 → ∃ (f : ℝ → ℝ), ∀ x : ℝ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = (f x) ^ 2 :=
by
  sorry

end make_polynomial_perfect_square_l65_65724


namespace min_value_expression_l65_65773

theorem min_value_expression (x y : ℝ) :
  ∃ m, (m = 104) ∧ (∀ x y : ℝ, (x + 3)^2 + 2 * (y - 2)^2 + 4 * (x - 7)^2 + (y + 4)^2 ≥ m) :=
sorry

end min_value_expression_l65_65773


namespace ac_bd_sum_l65_65819

theorem ac_bd_sum (a b c d : ℝ) (h1 : a + b + c = 6) (h2 : a + b + d = -3) (h3 : a + c + d = 0) (h4 : b + c + d = -9) : 
  a * c + b * d = 23 := 
sorry

end ac_bd_sum_l65_65819


namespace arithmetic_sqrt_sqrt_16_l65_65897

theorem arithmetic_sqrt_sqrt_16 : Real.sqrt (Real.sqrt 16) = 2 := 
by
  sorry

end arithmetic_sqrt_sqrt_16_l65_65897


namespace mixed_number_fraction_division_and_subtraction_l65_65029

theorem mixed_number_fraction_division_and_subtraction :
  ( (11 / 6) / (11 / 4) ) - (1 / 2) = 1 / 6 := 
sorry

end mixed_number_fraction_division_and_subtraction_l65_65029


namespace total_amount_spent_on_cookies_l65_65030

def days_in_april : ℕ := 30
def cookies_per_day : ℕ := 3
def cost_per_cookie : ℕ := 18

theorem total_amount_spent_on_cookies : days_in_april * cookies_per_day * cost_per_cookie = 1620 := by
  sorry

end total_amount_spent_on_cookies_l65_65030


namespace product_square_preceding_div_by_12_l65_65426

theorem product_square_preceding_div_by_12 (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) :=
by
  sorry

end product_square_preceding_div_by_12_l65_65426


namespace geometric_sequence_problem_l65_65681

variable {a : ℕ → ℝ}

-- Given conditions
def geometric_sequence (a : ℕ → ℝ) :=
  ∃ q r, (∀ n, a (n + 1) = q * a n ∧ a 0 = r)

-- Define the conditions from the problem
def condition1 (a : ℕ → ℝ) :=
  a 3 + a 6 = 6

def condition2 (a : ℕ → ℝ) :=
  a 5 + a 8 = 9

-- Theorem to be proved
theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (hgeom : geometric_sequence a)
  (h1 : condition1 a)
  (h2 : condition2 a) :
  a 7 + a 10 = 27 / 2 :=
sorry

end geometric_sequence_problem_l65_65681


namespace initial_liquid_A_quantity_l65_65081

theorem initial_liquid_A_quantity
  (x : ℝ)
  (init_A init_B init_C : ℝ)
  (removed_A removed_B removed_C : ℝ)
  (added_B added_C : ℝ)
  (new_A new_B new_C : ℝ)
  (h1 : init_A / init_B = 7 / 5)
  (h2 : init_A / init_C = 7 / 3)
  (h3 : init_A + init_B + init_C = 15 * x)
  (h4 : removed_A = 7 / 15 * 9)
  (h5 : removed_B = 5 / 15 * 9)
  (h6 : removed_C = 3 / 15 * 9)
  (h7 : new_A = init_A - removed_A)
  (h8 : new_B = init_B - removed_B + added_B)
  (h9 : new_C = init_C - removed_C + added_C)
  (h10 : new_A / (new_B + new_C) = 7 / 10)
  (h11 : added_B = 6)
  (h12 : added_C = 3) : 
  init_A = 35.7 :=
sorry

end initial_liquid_A_quantity_l65_65081


namespace positive_integers_no_common_factor_l65_65849

theorem positive_integers_no_common_factor (X Y Z : ℕ) 
    (X_pos : 0 < X) (Y_pos : 0 < Y) (Z_pos : 0 < Z)
    (coprime_XYZ : Nat.gcd (Nat.gcd X Y) Z = 1)
    (eqn : X * (Real.log 3 / Real.log 100) + Y * (Real.log 4 / Real.log 100) = Z^2) :
    X + Y + Z = 4 :=
sorry

end positive_integers_no_common_factor_l65_65849


namespace value_of_3_W_4_l65_65567

def W (a b : ℤ) : ℤ := b + 5 * a - 3 * a ^ 2

theorem value_of_3_W_4 : W 3 4 = -8 :=
by
  sorry

end value_of_3_W_4_l65_65567


namespace largest_proper_divisor_condition_l65_65097

def is_proper_divisor (n k : ℕ) : Prop :=
  k > 1 ∧ k < n ∧ n % k = 0

theorem largest_proper_divisor_condition (n p : ℕ) (hp : is_proper_divisor n p) (hl : ∀ k, is_proper_divisor n k → k ≤ n / p):
  n = 12 ∨ n = 33 :=
by
  -- Placeholder for proof
  sorry

end largest_proper_divisor_condition_l65_65097


namespace Kindergarten_Students_l65_65138

theorem Kindergarten_Students (X : ℕ) (h1 : 40 * X + 40 * 10 + 40 * 11 = 1200) : X = 9 :=
by
  sorry

end Kindergarten_Students_l65_65138


namespace geom_prog_common_ratio_l65_65678

variable {α : Type*} [Field α]

theorem geom_prog_common_ratio (x y z r : α) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) 
  (h1 : x * (y + z) = a) (h2 : y * (z + x) = a * r) (h3 : z * (x + y) = a * r^2) :
  r^2 + r + 1 = 0 :=
by
  sorry

end geom_prog_common_ratio_l65_65678


namespace same_terminal_side_angle_l65_65833

theorem same_terminal_side_angle (k : ℤ) : 
  ∃ (θ : ℤ), θ = k * 360 + 257 ∧ (θ % 360 = (-463) % 360) :=
by
  sorry

end same_terminal_side_angle_l65_65833


namespace tangent_subtraction_identity_l65_65548

theorem tangent_subtraction_identity (α β : ℝ) 
  (h1 : Real.tan α = -3/4) 
  (h2 : Real.tan (Real.pi - β) = 1/2) : 
  Real.tan (α - β) = -2/11 := 
sorry

end tangent_subtraction_identity_l65_65548


namespace ineq_10_3_minus_9_5_l65_65140

variable {a b c : ℝ}

/-- Given \(a, b, c\) are positive real numbers and \(a + b + c = 1\), prove \(10(a^3 + b^3 + c^3) - 9(a^5 + b^5 + c^5) \geq 1\). -/
theorem ineq_10_3_minus_9_5 (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 1) :
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := 
sorry

end ineq_10_3_minus_9_5_l65_65140


namespace sum_of_terms_l65_65754

def geometric_sequence (a b c d : ℝ) :=
  ∃ q : ℝ, a = b / q ∧ c = b * q ∧ d = c * q

def symmetric_sequence_of_length_7 (s : Fin 8 → ℝ) :=
  ∀ i : Fin 8, s i = s (Fin.mk (7 - i) sorry)

def sequence_conditions (s : Fin 8 → ℝ) :=
  symmetric_sequence_of_length_7 s ∧
  geometric_sequence (s ⟨1,sorry⟩) (s ⟨2,sorry⟩) (s ⟨3,sorry⟩) (s ⟨4,sorry⟩) ∧
  s ⟨1,sorry⟩ = 2 ∧
  s ⟨3,sorry⟩ = 8

theorem sum_of_terms (s : Fin 8 → ℝ) (h : sequence_conditions s) :
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = 44 ∨
  s 0 + s 1 + s 2 + s 3 + s 4 + s 5 + s 6 = -4 :=
sorry

end sum_of_terms_l65_65754


namespace combined_population_after_two_years_l65_65415

def population_after_years (initial_population : ℕ) (yearly_changes : List (ℕ → ℕ)) : ℕ :=
  yearly_changes.foldl (fun pop change => change pop) initial_population

def townA_change_year1 (pop : ℕ) : ℕ :=
  pop + (pop * 8 / 100) + 200 - 100

def townA_change_year2 (pop : ℕ) : ℕ :=
  pop + (pop * 10 / 100) + 200 - 100

def townB_change_year1 (pop : ℕ) : ℕ :=
  pop - (pop * 2 / 100) + 50 - 200

def townB_change_year2 (pop : ℕ) : ℕ :=
  pop - (pop * 1 / 100) + 50 - 200

theorem combined_population_after_two_years :
  population_after_years 15000 [townA_change_year1, townA_change_year2] +
  population_after_years 10000 [townB_change_year1, townB_change_year2] = 27433 := 
  sorry

end combined_population_after_two_years_l65_65415


namespace fraction_value_l65_65972

theorem fraction_value : (20 * 21) / (2 + 0 + 2 + 1) = 84 := by
  sorry

end fraction_value_l65_65972


namespace find_theta_l65_65919

theorem find_theta (R h : ℝ) (θ : ℝ) 
  (r1_def : r1 = R * Real.cos θ)
  (r2_def : r2 = (R + h) * Real.cos θ)
  (s_def : s = 2 * π * h * Real.cos θ)
  (s_eq_h : s = h) : 
  θ = Real.arccos (1 / (2 * π)) :=
by
  sorry

end find_theta_l65_65919


namespace multiply_polynomials_l65_65332

theorem multiply_polynomials (x : ℝ) : (x^4 + 50 * x^2 + 625) * (x^2 - 25) = x^6 - 15625 := by
  sorry

end multiply_polynomials_l65_65332


namespace least_num_subtracted_l65_65102

theorem least_num_subtracted 
  (n : ℕ) 
  (h1 : n = 642) 
  (rem_cond : ∀ k, (k = 638) → n - k = 4): 
  n - 638 = 4 := 
by sorry

end least_num_subtracted_l65_65102


namespace solve_for_B_l65_65564

theorem solve_for_B (B : ℕ) (h : 3 * B + 2 = 20) : B = 6 :=
by 
  -- This is just a placeholder, the proof will go here
  sorry

end solve_for_B_l65_65564


namespace smallest_N_for_abs_x_squared_minus_4_condition_l65_65661

theorem smallest_N_for_abs_x_squared_minus_4_condition (x : ℝ) 
  (h : abs (x - 2) < 0.01) : abs (x^2 - 4) < 0.0401 := 
sorry

end smallest_N_for_abs_x_squared_minus_4_condition_l65_65661


namespace carter_siblings_oldest_age_l65_65818

theorem carter_siblings_oldest_age
    (avg_age : ℕ)
    (sibling1 : ℕ)
    (sibling2 : ℕ)
    (sibling3 : ℕ)
    (sibling4 : ℕ) :
    avg_age = 9 →
    sibling1 = 5 →
    sibling2 = 8 →
    sibling3 = 7 →
    ((sibling1 + sibling2 + sibling3 + sibling4) / 4) = avg_age →
    sibling4 = 16 := by
  intros
  sorry

end carter_siblings_oldest_age_l65_65818


namespace no_real_solution_l65_65908

noncomputable def augmented_matrix (m : ℝ) : Matrix (Fin 2) (Fin 3) ℝ :=
  ![![m, 4, m+2], ![1, m, m]]

theorem no_real_solution (m : ℝ) :
  (∀ (a b : ℝ), ¬ ∃ (x y : ℝ), a * x + b * y = m ∧ a * x + b * y = 4 ∧ a * x + b * y = m + 2) ↔ m = 2 :=
by
sorry

end no_real_solution_l65_65908


namespace total_animals_correct_l65_65483

def L := 10

def C := 2 * L + 4

def Merry_lambs := L
def Merry_cows := C
def Merry_pigs (P : ℕ) := P
def Brother_lambs := L + 3

def Brother_chickens (R : ℕ) := R * Brother_lambs
def Brother_goats (Q : ℕ) := 2 * Brother_lambs + Q

def Merry_total (P : ℕ) := Merry_lambs + Merry_cows + Merry_pigs P
def Brother_total (R Q : ℕ) := Brother_lambs + Brother_chickens R + Brother_goats Q

def Total_animals (P R Q : ℕ) := Merry_total P + Brother_total R Q

theorem total_animals_correct (P R Q : ℕ) : 
  Total_animals P R Q = 73 + P + R * 13 + Q := by
  sorry

end total_animals_correct_l65_65483


namespace election_ratio_l65_65744

theorem election_ratio (X Y : ℝ) 
  (h : 0.74 * X + 0.5000000000000002 * Y = 0.66 * (X + Y)) : 
  X / Y = 2 :=
by sorry

end election_ratio_l65_65744


namespace total_raining_time_correct_l65_65602

-- Define individual durations based on given conditions
def duration_day1 : ℕ := 10        -- 17:00 - 07:00 = 10 hours
def duration_day2 : ℕ := duration_day1 + 2    -- Second day: 10 hours + 2 hours = 12 hours
def duration_day3 : ℕ := duration_day2 * 2    -- Third day: 12 hours * 2 = 24 hours

-- Define the total raining time over three days
def total_raining_time : ℕ := duration_day1 + duration_day2 + duration_day3

-- Formally state the theorem to prove the total rain time is 46 hours
theorem total_raining_time_correct : total_raining_time = 46 := by
  sorry

end total_raining_time_correct_l65_65602


namespace solve_cubic_fraction_l65_65455

noncomputable def problem_statement (x : ℝ) :=
  (x = (-(3:ℝ) + Real.sqrt 13) / 4) ∨ (x = (-(3:ℝ) - Real.sqrt 13) / 4)

theorem solve_cubic_fraction (x : ℝ) (h : (x^3 + 2*x^2 + 3*x + 5) / (x + 2) = x + 4) : 
  problem_statement x :=
by
  sorry

end solve_cubic_fraction_l65_65455


namespace bricks_of_other_types_l65_65769

theorem bricks_of_other_types (A B total other: ℕ) (hA: A = 40) (hB: B = A / 2) (hTotal: total = 150) (hSum: total = A + B + other): 
  other = 90 :=
by sorry

end bricks_of_other_types_l65_65769


namespace find_unique_positive_integer_pair_l65_65615

theorem find_unique_positive_integer_pair :
  ∃! (b c : ℕ), b > 0 ∧ c > 0 ∧ c > b^2 ∧ b > c^2 :=
sorry

end find_unique_positive_integer_pair_l65_65615


namespace dot_product_vec1_vec2_l65_65946

-- Define the vectors
def vec1 := (⟨-4, -1⟩ : ℤ × ℤ)
def vec2 := (⟨6, 8⟩ : ℤ × ℤ)

-- Define the dot product function
def dot_product (v1 v2 : ℤ × ℤ) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Prove that the dot product of vec1 and vec2 is -32
theorem dot_product_vec1_vec2 : dot_product vec1 vec2 = -32 :=
by
  sorry

end dot_product_vec1_vec2_l65_65946


namespace total_price_of_books_l65_65418

theorem total_price_of_books
  (total_books : ℕ)
  (math_books_cost : ℕ)
  (history_books_cost : ℕ)
  (math_books_bought : ℕ)
  (total_books_eq : total_books = 80)
  (math_books_cost_eq : math_books_cost = 4)
  (history_books_cost_eq : history_books_cost = 5)
  (math_books_bought_eq : math_books_bought = 10) :
  (math_books_bought * math_books_cost + (total_books - math_books_bought) * history_books_cost = 390) := 
by
  sorry

end total_price_of_books_l65_65418


namespace cat_weight_l65_65376

theorem cat_weight 
  (weight1 weight2 : ℕ)
  (total_weight : ℕ)
  (h1 : weight1 = 2)
  (h2 : weight2 = 7)
  (h3 : total_weight = 13) : 
  ∃ weight3 : ℕ, weight3 = 4 := 
by
  sorry

end cat_weight_l65_65376


namespace hyperbola_center_l65_65611

theorem hyperbola_center :
  ∃ (c : ℝ × ℝ), c = (3, 5) ∧
  (9 * (x - c.1)^2 - 36 * (y - c.2)^2 - (1244 - 243 - 1001) = 0) :=
sorry

end hyperbola_center_l65_65611


namespace value_fraction_l65_65251

variables {x y : ℝ}
variables (hx : x ≠ 0) (hy : y ≠ 0) (h : (4 * x + 2 * y) / (x - 4 * y) = 3)

theorem value_fraction : (x + 4 * y) / (4 * x - y) = 10 / 57 :=
by { sorry }

end value_fraction_l65_65251


namespace series_sum_l65_65171

theorem series_sum :
  ∑' n : ℕ,  n ≠ 0 → (6 * n + 2) / ((6 * n - 1)^2 * (6 * n + 5)^2) = 1 / 600 := by
  sorry

end series_sum_l65_65171


namespace inequality_correct_l65_65330

variable (a b : ℝ)

theorem inequality_correct (h : a < b) : 2 - a > 2 - b :=
by
  sorry

end inequality_correct_l65_65330


namespace find_f_2023_l65_65414

def is_strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ {a b : ℕ}, a < b → f a < f b

theorem find_f_2023 (f : ℕ → ℕ)
  (h_inc : is_strictly_increasing f)
  (h_relation : ∀ m n : ℕ, f (n + f m) = f n + m + 1) :
  f 2023 = 2024 :=
sorry

end find_f_2023_l65_65414


namespace todd_savings_l65_65529

def original_price : ℝ := 125
def sale_discount : ℝ := 0.20
def coupon : ℝ := 10
def credit_card_discount : ℝ := 0.10
def rebate : ℝ := 0.05
def sales_tax : ℝ := 0.08

def calculate_savings (original_price sale_discount coupon credit_card_discount rebate sales_tax : ℝ) : ℝ :=
  let after_sale := original_price * (1 - sale_discount)
  let after_coupon := after_sale - coupon
  let after_credit_card := after_coupon * (1 - credit_card_discount)
  let after_rebate := after_credit_card * (1 - rebate)
  let tax := after_credit_card * sales_tax
  let final_price := after_rebate + tax
  original_price - final_price

theorem todd_savings : calculate_savings 125 0.20 10 0.10 0.05 0.08 = 41.57 :=
by
  sorry

end todd_savings_l65_65529


namespace pool_filling_time_l65_65012

theorem pool_filling_time :
  let pool_capacity := 12000 -- in cubic meters
  let first_valve_time := 120 -- in minutes
  let first_valve_rate := pool_capacity / first_valve_time -- in cubic meters per minute
  let second_valve_rate := first_valve_rate + 50 -- in cubic meters per minute
  let combined_rate := first_valve_rate + second_valve_rate -- in cubic meters per minute
  let time_to_fill := pool_capacity / combined_rate -- in minutes
  time_to_fill = 48 :=
by
  sorry

end pool_filling_time_l65_65012


namespace total_fruits_in_30_days_l65_65752

-- Define the number of oranges Sophie receives each day
def sophie_daily_oranges : ℕ := 20

-- Define the number of grapes Hannah receives each day
def hannah_daily_grapes : ℕ := 40

-- Define the number of days
def number_of_days : ℕ := 30

-- Calculate the total number of fruits received by Sophie and Hannah in 30 days
theorem total_fruits_in_30_days :
  (sophie_daily_oranges * number_of_days) + (hannah_daily_grapes * number_of_days) = 1800 :=
by
  sorry

end total_fruits_in_30_days_l65_65752


namespace sin_double_alpha_l65_65603

variable (α β : ℝ)

theorem sin_double_alpha (h1 : Real.pi / 2 < β ∧ β < α ∧ α < 3 * Real.pi / 4)
        (h2 : Real.cos (α - β) = 12 / 13) 
        (h3 : Real.sin (α + β) = -3 / 5) : 
        Real.sin (2 * α) = -56 / 65 := by
  sorry

end sin_double_alpha_l65_65603


namespace top_black_second_red_probability_l65_65215

-- Define the problem conditions in Lean
def num_standard_cards : ℕ := 52
def num_jokers : ℕ := 2
def num_total_cards : ℕ := num_standard_cards + num_jokers

def num_black_cards : ℕ := 26
def num_red_cards : ℕ := 26

-- Lean statement
theorem top_black_second_red_probability :
  (num_black_cards / num_total_cards * num_red_cards / (num_total_cards - 1)) = 338 / 1431 := by
  sorry

end top_black_second_red_probability_l65_65215


namespace Mia_biking_speed_l65_65827

theorem Mia_biking_speed
    (Eugene_speed : ℝ)
    (Carlos_ratio : ℝ)
    (Mia_ratio : ℝ)
    (Mia_speed : ℝ)
    (h1 : Eugene_speed = 5)
    (h2 : Carlos_ratio = 3 / 4)
    (h3 : Mia_ratio = 4 / 3)
    (h4 : Mia_speed = Mia_ratio * (Carlos_ratio * Eugene_speed)) :
    Mia_speed = 5 :=
by
  sorry

end Mia_biking_speed_l65_65827


namespace amy_l65_65035

theorem amy's_speed (a b : ℝ) (s : ℝ) 
  (h1 : ∀ (major minor : ℝ), major = 2 * minor) 
  (h2 : ∀ (w : ℝ), w = 4) 
  (h3 : ∀ (t_diff : ℝ), t_diff = 48) 
  (h4 : 2 * a + 2 * Real.pi * Real.sqrt ((4 * b^2 + b^2) / 2) - (2 * a + 2 * Real.pi * Real.sqrt (((2 * b + 8)^2 + (b + 4)^2) / 2)) = 48 * s) :
  s = Real.pi / 2 := sorry

end amy_l65_65035


namespace log4_80_cannot_be_found_without_additional_values_l65_65185

-- Conditions provided in the problem
def log4_16 : Real := 2
def log4_32 : Real := 2.5

-- Lean statement of the proof problem
theorem log4_80_cannot_be_found_without_additional_values :
  ¬(∃ (log4_80 : Real), log4_80 = log4_16 + log4_5) :=
sorry

end log4_80_cannot_be_found_without_additional_values_l65_65185


namespace triangle_BPC_area_l65_65114

universe u

variables {T : Type u} [LinearOrderedField T]

-- Define the points
variables (A B C E F P : T)
variables (area : T → T → T → T) -- A function to compute the area of a triangle

-- Hypotheses
def conditions :=
  E ∈ [A, B] ∧
  F ∈ [A, C] ∧
  (∃ P, P ∈ [B, F] ∧ P ∈ [C, E]) ∧
  area A E P + area E P F + area P F A = 4 ∧ -- AEPF
  area B E P = 4 ∧ -- BEP
  area C F P = 4   -- CFP

-- The theorem to prove
theorem triangle_BPC_area (h : conditions A B C E F P area) : area B P C = 12 :=
sorry

end triangle_BPC_area_l65_65114


namespace find_x_l65_65043

theorem find_x (x : ℝ) (h1 : x ≠ 0) (h2 : x = (1 / x) * (-x) + 3) : x = 2 :=
by
  sorry

end find_x_l65_65043


namespace roof_length_width_difference_l65_65759

variable (w l : ℕ)

theorem roof_length_width_difference (h1 : l = 7 * w) (h2 : l * w = 847) : l - w = 66 :=
by 
  sorry

end roof_length_width_difference_l65_65759


namespace inequality_holds_l65_65411

variable (a b c d : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)
variable (hc : c > 0)
variable (hd : d > 0)
variable (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2)

theorem inequality_holds (ha : a > 0)
  (hb : b > 0) 
  (hc : c > 0) 
  (hd : d > 0) 
  (h : 1 / (a^3 + 1) + 1 / (b^3 + 1) + 1 / (c^3 + 1) + 1 / (d^3 + 1) = 2) :
  (1 - a) / (a^2 - a + 1) + (1 - b) / (b^2 - b + 1) + (1 - c) / (c^2 - c + 1) + (1 - d) / (d^2 - d + 1) ≥ 0 := by
  sorry

end inequality_holds_l65_65411


namespace distinct_primes_p_q_r_l65_65941

theorem distinct_primes_p_q_r (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) (eqn : r * p^3 + p^2 + p = 2 * r * q^2 + q^2 + q) : p * q * r = 2014 :=
by
  sorry

end distinct_primes_p_q_r_l65_65941


namespace average_paychecks_l65_65169

def first_paychecks : Nat := 6
def remaining_paychecks : Nat := 20
def total_paychecks : Nat := 26
def amount_first : Nat := 750
def amount_remaining : Nat := 770

theorem average_paychecks : 
  (first_paychecks * amount_first + remaining_paychecks * amount_remaining) / total_paychecks = 765 :=
by
  sorry

end average_paychecks_l65_65169


namespace find_n_l65_65642

theorem find_n {x n : ℕ} (h1 : 3 * x - 4 = 8) (h2 : 7 * x - 15 = 13) (h3 : 4 * x + 2 = 18) 
  (h4 : n = 803) : 8 + (n - 1) * 5 = 4018 := by
  sorry

end find_n_l65_65642


namespace novel_cost_l65_65121

-- Given conditions
variable (N : ℕ) -- cost of the novel
variable (lunch_cost : ℕ) -- cost of lunch

-- Conditions
axiom gift_amount : N + lunch_cost + 29 = 50
axiom lunch_cost_eq : lunch_cost = 2 * N

-- Question and answer tuple as a theorem
theorem novel_cost : N = 7 := 
by
  sorry -- Proof estaps are to be filled in.

end novel_cost_l65_65121


namespace integer_range_2014_l65_65774

theorem integer_range_2014 : 1000 < 2014 ∧ 2014 < 10000 := by
  sorry

end integer_range_2014_l65_65774


namespace solve_for_x_l65_65187

theorem solve_for_x 
  (x : ℝ) 
  (h : (2/7) * (1/4) * x = 8) : 
  x = 112 :=
sorry

end solve_for_x_l65_65187


namespace trigonometric_identity_l65_65416

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + Real.cos θ) = 5 / 2 :=
by
  sorry

end trigonometric_identity_l65_65416


namespace track_length_proof_l65_65700

noncomputable def track_length : ℝ :=
  let x := 541.67
  x

theorem track_length_proof
  (p : ℝ)
  (q : ℝ)
  (h1 : p = 1 / 4)
  (h2 : q = 120)
  (h3 : ¬(p = q))
  (h4 : ∃ r : ℝ, r = 180)
  (speed_constant : ∃ b_speed, ∃ s_speed, b_speed * t = q ∧ s_speed * t = r) :
  track_length = 541.67 :=
sorry

end track_length_proof_l65_65700


namespace sum_A_B_l65_65586

theorem sum_A_B (A B : ℕ) 
  (h1 : (1 / 4 : ℚ) * (1 / 8) = 1 / (4 * A))
  (h2 : 1 / (4 * A) = 1 / B) : A + B = 40 := 
by
  sorry

end sum_A_B_l65_65586


namespace max_value_of_sum_l65_65659

theorem max_value_of_sum (x y z : ℝ) (h : x^2 + 4 * y^2 + 9 * z^2 = 3) : x + 2 * y + 3 * z ≤ 3 :=
sorry

end max_value_of_sum_l65_65659


namespace trajectory_of_P_l65_65254

theorem trajectory_of_P (M P : ℝ × ℝ) (OM OP : ℝ) (x y : ℝ) :
  (M = (4, y)) →
  (P = (x, y)) →
  (OM = Real.sqrt (4^2 + y^2)) →
  (OP = Real.sqrt ((x - 4)^2 + y^2)) →
  (OM * OP = 16) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end trajectory_of_P_l65_65254


namespace sum_inverse_terms_l65_65339

theorem sum_inverse_terms : 
  (∑' n : ℕ, if n = 0 then (0 : ℝ) else (1 / (n * (n + 3) : ℝ))) = 11 / 18 :=
by {
  -- proof to be filled in
  sorry
}

end sum_inverse_terms_l65_65339


namespace loss_percentage_is_11_l65_65538

-- Constants for the given problem conditions
def cost_price : ℝ := 1500
def selling_price : ℝ := 1335

-- Formulation of the proof problem
theorem loss_percentage_is_11 :
  ((cost_price - selling_price) / cost_price) * 100 = 11 := by
  sorry

end loss_percentage_is_11_l65_65538


namespace frac_sum_eq_l65_65695

theorem frac_sum_eq (a b : ℝ) (h1 : a^2 + a - 1 = 0) (h2 : b^2 + b - 1 = 0) : 
  (a / b + b / a = 2) ∨ (a / b + b / a = -3) := 
sorry

end frac_sum_eq_l65_65695


namespace opposite_of_neg2_l65_65961

def opposite (y : ℤ) : ℤ := -y

theorem opposite_of_neg2 : opposite (-2) = 2 := by
  sorry

end opposite_of_neg2_l65_65961


namespace mutually_exclusive_not_complementary_l65_65554

-- Define the basic events and conditions
structure Pocket :=
(red : ℕ)
(black : ℕ)

-- Define the event type
inductive Event
| atleast_one_black : Event
| both_black : Event
| atleast_one_red : Event
| both_red : Event
| exactly_one_black : Event
| exactly_two_black : Event
| none_black : Event

def is_mutually_exclusive (e1 e2 : Event) : Prop :=
  match e1, e2 with
  | Event.exactly_one_black, Event.exactly_two_black => true
  | Event.exactly_two_black, Event.exactly_one_black => true
  | _, _ => false

def is_complementary (e1 e2 : Event) : Prop :=
  e1 = Event.none_black ∧ e2 = Event.both_red ∨
  e1 = Event.both_red ∧ e2 = Event.none_black

-- Given conditions
def pocket : Pocket := { red := 2, black := 2 }

-- Proof problem setup
theorem mutually_exclusive_not_complementary : 
  is_mutually_exclusive Event.exactly_one_black Event.exactly_two_black ∧
  ¬ is_complementary Event.exactly_one_black Event.exactly_two_black :=
by
  sorry

end mutually_exclusive_not_complementary_l65_65554


namespace factor_polynomial_l65_65650

variable {R : Type*} [CommRing R]

theorem factor_polynomial (a b c : R) :
  a^4 * (b^3 - c^3) + b^4 * (c^3 - a^3) + c^4 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (-(a + b + c) * (a^2 + b^2 + c^2 + ab + bc + ac)) :=
by
  sorry

end factor_polynomial_l65_65650


namespace total_pieces_of_pizza_l65_65175

def pieces_per_pizza : ℕ := 6
def pizzas_per_fourthgrader : ℕ := 20
def fourthgraders_count : ℕ := 10

theorem total_pieces_of_pizza :
  fourthgraders_count * (pieces_per_pizza * pizzas_per_fourthgrader) = 1200 :=
by
  /-
  We have:
  - pieces_per_pizza = 6
  - pizzas_per_fourthgrader = 20
  - fourthgraders_count = 10

  Therefore,
  10 * (6 * 20) = 1200
  -/
  sorry

end total_pieces_of_pizza_l65_65175


namespace permutations_PERCEPTION_l65_65346

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end permutations_PERCEPTION_l65_65346


namespace unique_line_through_point_odd_x_prime_y_intercepts_l65_65539

theorem unique_line_through_point_odd_x_prime_y_intercepts :
  ∃! (a b : ℕ), 0 < b ∧ Nat.Prime b ∧ a % 2 = 1 ∧
  (4 * b + 3 * a = a * b) :=
sorry

end unique_line_through_point_odd_x_prime_y_intercepts_l65_65539


namespace point_on_curve_l65_65116

-- Define the equation of the curve
def curve (x y : ℝ) := x^2 - x * y + 2 * y + 1 = 0

-- State that point (3, 10) satisfies the given curve equation
theorem point_on_curve : curve 3 10 :=
by
  -- this is where the proof would go but we will skip it for now
  sorry

end point_on_curve_l65_65116


namespace series_sum_l65_65900

theorem series_sum :
  ∑' n : ℕ, (2 ^ n) / (3 ^ (2 ^ n) + 1) = 1 / 2 :=
sorry

end series_sum_l65_65900


namespace find_value_l65_65787

theorem find_value (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2 * m^2 + 2010 = 2011 := 
sorry

end find_value_l65_65787


namespace shirts_bought_by_peter_l65_65464

-- Define the constants and assumptions
variables (P S x : ℕ)

-- State the conditions given in the problem
def condition1 : P = 6 :=
by sorry

def condition2 : 2 * S = 20 :=
by sorry

def condition3 : 2 * P + x * S = 62 :=
by sorry

-- State the theorem to be proven
theorem shirts_bought_by_peter : x = 5 :=
by sorry

end shirts_bought_by_peter_l65_65464


namespace blocks_difference_l65_65350

def blocks_house := 89
def blocks_tower := 63

theorem blocks_difference : (blocks_house - blocks_tower = 26) :=
by sorry

end blocks_difference_l65_65350


namespace range_of_a_min_value_of_a_l65_65804

variable (f : ℝ → ℝ) (a x : ℝ)

-- Part 1
theorem range_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₁ : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ 3) : 0 ≤ a ∧ a ≤ 4 :=
sorry

-- Part 2
theorem min_value_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₂ : ∀ x, f (x - a) + f (x + a) ≥ 1 - a) : a ≥ 1/3 :=
sorry

end range_of_a_min_value_of_a_l65_65804


namespace sandbox_width_l65_65574

theorem sandbox_width :
  ∀ (length area width : ℕ), length = 312 → area = 45552 →
  area = length * width → width = 146 :=
by
  intros length area width h_length h_area h_eq
  sorry

end sandbox_width_l65_65574


namespace smallest_n_l65_65712

theorem smallest_n (n : ℕ) : 
  (2^n + 5^n - n) % 1000 = 0 ↔ n = 797 :=
sorry

end smallest_n_l65_65712


namespace find_a_if_f_even_l65_65634

noncomputable def f (x a : ℝ) : ℝ := (x + a) * Real.log (((2 * x) - 1) / ((2 * x) + 1))

theorem find_a_if_f_even (a : ℝ) :
  (∀ x : ℝ, (x > 1/2 ∨ x < -1/2) → f x a = f (-x) a) → a = 0 :=
by
  intro h1
  -- This is where the mathematical proof would go, but it's omitted as per the requirements.
  sorry

end find_a_if_f_even_l65_65634


namespace laila_utility_l65_65206

theorem laila_utility (u : ℝ) :
  (2 * u * (10 - 2 * u) = 2 * (4 - 2 * u) * (2 * u + 4)) → u = 4 := 
by 
  sorry

end laila_utility_l65_65206


namespace graph_is_hyperbola_l65_65718

theorem graph_is_hyperbola : ∀ (x y : ℝ), x^2 - 18 * y^2 - 6 * x + 4 * y + 9 = 0 → ∃ a b c d : ℝ, a * (x - b)^2 - c * (y - d)^2 = 1 :=
by
  -- Proof is omitted
  sorry

end graph_is_hyperbola_l65_65718


namespace proof_problem_l65_65275

def polar_curve_C (ρ : ℝ) : Prop := ρ = 5

def point_P (x y : ℝ) : Prop := x = -3 ∧ y = -3 / 2

def line_l_through_P (x y : ℝ) (k : ℝ) : Prop := y + 3 / 2 = k * (x + 3)

def distance_AB (A B : ℝ × ℝ) : Prop := (A.1 - B.1)^2 + (A.2 - B.2)^2 = 64

theorem proof_problem
  (ρ : ℝ) (x y : ℝ) (k : ℝ)
  (A B : ℝ × ℝ)
  (h1 : polar_curve_C ρ)
  (h2 : point_P (-3) (-3 / 2))
  (h3 : ∃ k, line_l_through_P x y k)
  (h4 : distance_AB A B) :
  ∃ (x y : ℝ), (x^2 + y^2 = 25) ∧ ((x = -3) ∨ (3 * x + 4 * y + 15 = 0)) := 
sorry

end proof_problem_l65_65275


namespace radio_loss_percentage_l65_65618

theorem radio_loss_percentage (cost_price selling_price : ℕ) (h1 : cost_price = 1500) (h2 : selling_price = 1305) : 
  (cost_price - selling_price) * 100 / cost_price = 13 := by
  sorry

end radio_loss_percentage_l65_65618


namespace math_problem_l65_65778

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => if a n < 2 * n then a n + 1 else a n

theorem math_problem (n : ℕ) (hn : n > 0) (ha_inc : ∀ m, m > 0 → a m < a (m + 1)) 
  (ha_rec : ∀ m, m > 0 → a (m + 1) ≤ 2 * m) : 
  ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ n = a p - a q := sorry

end math_problem_l65_65778


namespace find_f_inv_486_l65_65201

-- Assuming function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Given conditions
axiom f_cond1 : f 4 = 2
axiom f_cond2 : ∀ x : ℝ, f (3 * x) = 3 * f x

-- Proof problem: Prove that f⁻¹(486) = 972
theorem find_f_inv_486 : (∃ x : ℝ, f x = 486 ∧ x = 972) :=
sorry

end find_f_inv_486_l65_65201


namespace cookies_from_dough_l65_65459

theorem cookies_from_dough :
  ∀ (length width : ℕ), length = 24 → width = 18 →
  ∃ (side : ℕ), side = Nat.gcd length width ∧ (length / side) * (width / side) = 12 :=
by
  intros length width h_length h_width
  simp only [h_length, h_width]
  use Nat.gcd length width
  simp only [Nat.gcd_rec]
  sorry

end cookies_from_dough_l65_65459


namespace unique_odd_number_between_500_and_1000_l65_65901

theorem unique_odd_number_between_500_and_1000 :
  ∃! x : ℤ, 500 ≤ x ∧ x ≤ 1000 ∧ x % 25 = 6 ∧ x % 9 = 7 ∧ x % 2 = 1 :=
sorry

end unique_odd_number_between_500_and_1000_l65_65901


namespace Problem_l65_65895

def f (x : ℕ) : ℕ := x ^ 2 + 1
def g (x : ℕ) : ℕ := 2 * x - 1

theorem Problem : f (g (3 + 1)) = 50 := by
  sorry

end Problem_l65_65895


namespace suitable_storage_temp_l65_65279

theorem suitable_storage_temp : -5 ≤ -1 ∧ -1 ≤ 1 := by {
  sorry
}

end suitable_storage_temp_l65_65279


namespace distance_AC_100_l65_65791

theorem distance_AC_100 (d_AB : ℝ) (t1 : ℝ) (t2 : ℝ) (AC : ℝ) (CB : ℝ) :
  d_AB = 150 ∧ t1 = 3 ∧ t2 = 12 ∧ d_AB = AC + CB ∧ AC / 3 = CB / 12 → AC = 100 := 
by
  sorry

end distance_AC_100_l65_65791


namespace quadratic_roots_square_l65_65553

theorem quadratic_roots_square (q : ℝ) :
  (∃ a : ℝ, a + a^2 = 12 ∧ q = a * a^2) → (q = 27 ∨ q = -64) :=
by
  sorry

end quadratic_roots_square_l65_65553


namespace intersecting_lines_l65_65104

theorem intersecting_lines (m : ℝ) :
  (∃ (x y : ℝ), y = 2 * x ∧ x + y = 3 ∧ m * x + 2 * y + 5 = 0) ↔ (m = -9) :=
by
  sorry

end intersecting_lines_l65_65104


namespace arithmetic_sequence_y_solution_l65_65164

theorem arithmetic_sequence_y_solution : 
  ∃ y : ℚ, (y + 2 - - (1 / 3)) = (4 * y - (y + 2)) ∧ y = 13 / 6 :=
by
  sorry

end arithmetic_sequence_y_solution_l65_65164


namespace golf_balls_dozen_count_l65_65048

theorem golf_balls_dozen_count (n d : Nat) (h1 : n = 108) (h2 : d = 12) : n / d = 9 :=
by
  sorry

end golf_balls_dozen_count_l65_65048


namespace volume_of_cube_in_pyramid_l65_65453

open Real

noncomputable def side_length_of_base := 2
noncomputable def height_of_equilateral_triangle := sqrt 6
noncomputable def cube_side_length := sqrt 6 / 3
noncomputable def volume_of_cube := cube_side_length ^ 3

theorem volume_of_cube_in_pyramid 
  (side_length_of_base : ℝ) (height_of_equilateral_triangle : ℝ) (cube_side_length : ℝ) :
  volume_of_cube = 2 * sqrt 6 / 9 := 
by
  sorry

end volume_of_cube_in_pyramid_l65_65453


namespace intersection_is_correct_l65_65513

def A : Set ℤ := {-1, 1, 2, 4}
def B : Set ℤ := {-1, 0, 2}

theorem intersection_is_correct : A ∩ B = {-1, 2} := 
by 
  -- proof goes here 
  sorry

end intersection_is_correct_l65_65513


namespace unique_triplet_exists_l65_65779

theorem unique_triplet_exists (a b p : ℕ) (hp : Nat.Prime p) : 
  (a + b)^p = p^a + p^b → (a = 1 ∧ b = 1 ∧ p = 2) :=
by sorry

end unique_triplet_exists_l65_65779


namespace xy_condition_l65_65917

theorem xy_condition (x y : ℝ) (h : x * y + x / y + y / x = -3) : (x - 2) * (y - 2) = 3 :=
sorry

end xy_condition_l65_65917


namespace probability_colors_match_l65_65117

section ProbabilityJellyBeans

structure JellyBeans where
  green : ℕ
  blue : ℕ
  red : ℕ

def total_jellybeans (jb : JellyBeans) : ℕ :=
  jb.green + jb.blue + jb.red

-- Define the situation using structures
def lila_jellybeans : JellyBeans := { green := 1, blue := 1, red := 1 }
def max_jellybeans : JellyBeans := { green := 2, blue := 1, red := 3 }

-- Define probabilities
noncomputable def probability (count : ℕ) (total : ℕ) : ℚ :=
  if total = 0 then 0 else (count : ℚ) / (total : ℚ)

-- Main theorem
theorem probability_colors_match :
  probability lila_jellybeans.green (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.green (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.blue (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.blue (total_jellybeans max_jellybeans) +
  probability lila_jellybeans.red (total_jellybeans lila_jellybeans) *
  probability max_jellybeans.red (total_jellybeans max_jellybeans) = 1 / 3 :=
by sorry

end ProbabilityJellyBeans

end probability_colors_match_l65_65117


namespace range_of_a_l65_65003

theorem range_of_a :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 2 * x * (3 * x + a) < 1) → a < 1 :=
by
  sorry

end range_of_a_l65_65003


namespace intersection_correct_l65_65983

noncomputable def set_M : Set ℝ := { x | x^2 + x - 6 ≤ 0 }
noncomputable def set_N : Set ℝ := { x | abs (2 * x + 1) > 3 }
noncomputable def set_intersection : Set ℝ := { x | (x ∈ set_M) ∧ (x ∈ set_N) }

theorem intersection_correct : 
  set_intersection = { x : ℝ | (-3 ≤ x ∧ x < -2) ∨ (1 < x ∧ x ≤ 2) } := 
by 
  sorry

end intersection_correct_l65_65983


namespace no_solution_xyz_l65_65662

theorem no_solution_xyz : ∀ (x y z : Nat), (1 ≤ x) → (x ≤ 9) → (0 ≤ y) → (y ≤ 9) → (0 ≤ z) → (z ≤ 9) →
    100 * x + 10 * y + z ≠ 10 * x * y + x * z :=
by
  intros x y z hx1 hx9 hy1 hy9 hz1 hz9
  sorry

end no_solution_xyz_l65_65662


namespace sum_of_three_consecutive_odds_l65_65934

theorem sum_of_three_consecutive_odds (a : ℤ) (h : a % 2 = 1) (ha_mod : (a + 4) % 2 = 1) (h_sum : a + (a + 4) = 150) : a + (a + 2) + (a + 4) = 225 :=
sorry

end sum_of_three_consecutive_odds_l65_65934


namespace least_value_expression_l65_65407

theorem least_value_expression : ∃ x : ℝ, ∀ (x : ℝ), (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 ≥ 2094
∧ ∃ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2023 = 2094 := by
  sorry

end least_value_expression_l65_65407


namespace total_pencils_children_l65_65273

theorem total_pencils_children :
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  c1 + c2 + c3 + c4 + c5 = 60 :=
by
  let c1 := 6
  let c2 := 9
  let c3 := 12
  let c4 := 15
  let c5 := 18
  show c1 + c2 + c3 + c4 + c5 = 60
  sorry

end total_pencils_children_l65_65273


namespace car_first_hour_speed_l65_65665

theorem car_first_hour_speed
  (x speed2 : ℝ)
  (avgSpeed : ℝ)
  (h_speed2 : speed2 = 60)
  (h_avgSpeed : avgSpeed = 35) :
  (avgSpeed = (x + speed2) / 2) → x = 10 :=
by
  sorry

end car_first_hour_speed_l65_65665


namespace number_with_20_multiples_l65_65978

theorem number_with_20_multiples : ∃ n : ℕ, (∀ k : ℕ, (1 ≤ k) → (k ≤ 100) → (n ∣ k) → (k / n ≤ 20) ) ∧ n = 5 := 
  sorry

end number_with_20_multiples_l65_65978


namespace employees_six_years_or_more_percentage_l65_65669

theorem employees_six_years_or_more_percentage 
  (Y : ℕ)
  (Total : ℝ := (3 * Y:ℝ) + (4 * Y:ℝ) + (7 * Y:ℝ) - (2 * Y:ℝ) + (6 * Y:ℝ) + (1 * Y:ℝ))
  (Employees_Six_Years : ℝ := (6 * Y:ℝ) + (1 * Y:ℝ))
  : Employees_Six_Years / Total * 100 = 36.84 :=
by
  sorry

end employees_six_years_or_more_percentage_l65_65669


namespace find_base_l65_65531

theorem find_base (r : ℕ) : 
  (2 * r^2 + 1 * r + 0) + (2 * r^2 + 6 * r + 0) = 5 * r^2 + 0 * r + 0 → r = 7 :=
by
  sorry

end find_base_l65_65531


namespace perpendicular_lines_a_eq_neg6_l65_65707

theorem perpendicular_lines_a_eq_neg6 
  (a : ℝ) 
  (h1 : ∀ x y : ℝ, ax + 2*y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + 3*y - 2 = 0) 
  (h_perpendicular : ∀ m1 m2 : ℝ, m1 * m2 = -1) : 
  a = -6 := 
by 
  sorry

end perpendicular_lines_a_eq_neg6_l65_65707


namespace fraction_division_l65_65230

theorem fraction_division (a b c d : ℚ) (h1 : a = 3) (h2 : b = 8) (h3 : c = 5) (h4 : d = 12) :
  (a / b) / (c / d) = 9 / 10 :=
by
  sorry

end fraction_division_l65_65230


namespace white_stones_count_l65_65549

/-- We define the total number of stones as a constant. -/
def total_stones : ℕ := 120

/-- We define the difference between white and black stones as a constant. -/
def white_minus_black : ℕ := 36

/-- The theorem states that if there are 120 go stones in total and 
    36 more white go stones than black go stones, then there are 78 white go stones. -/
theorem white_stones_count (W B : ℕ) (h1 : W = B + white_minus_black) (h2 : B + W = total_stones) : W = 78 := 
sorry

end white_stones_count_l65_65549


namespace total_birds_in_tree_l65_65558

def initial_birds := 14
def additional_birds := 21

theorem total_birds_in_tree : initial_birds + additional_birds = 35 := by
  sorry

end total_birds_in_tree_l65_65558


namespace value_of_m_l65_65398

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 2 * x + 1 else 2 * (-x) + 1

theorem value_of_m (m : ℝ) (heven : ∀ x : ℝ, f (-x) = f x)
  (hpos : ∀ x : ℝ, x ≥ 0 → f x = 2 * x + 1)
  (hfm : f m = 5) : m = 2 ∨ m = -2 :=
sorry

end value_of_m_l65_65398


namespace arithmetic_sequence_sum_l65_65180

-- Define arithmetic sequence and sum of first n terms
def arithmetic_seq (a d : ℕ → ℕ) :=
  ∀ n, a (n + 1) = a n + d 1

def arithmetic_sum (a d : ℕ → ℕ) (n : ℕ) :=
  (n * (a 1 + a n)) / 2

-- Conditions from the problem
variables {a : ℕ → ℕ} {d : ℕ}

axiom condition : a 3 + a 7 + a 11 = 6

-- Definition of a_7 as derived in the solution
def a_7 : ℕ := 2

-- Proof problem equivalent statement
theorem arithmetic_sequence_sum : arithmetic_sum a d 13 = 26 :=
by
  -- These steps would involve setting up and proving the calculation details
  sorry

end arithmetic_sequence_sum_l65_65180


namespace perimeter_of_staircase_region_l65_65758

-- Definitions according to the conditions.
def staircase_region.all_right_angles : Prop := True -- Given condition that all angles are right angles.
def staircase_region.side_length : ℕ := 1 -- Given condition that the side length of each congruent side is 1 foot.
def staircase_region.total_area : ℕ := 120 -- Given condition that the total area of the region is 120 square feet.
def num_sides : ℕ := 12 -- Number of congruent sides.

-- The question is to prove that the perimeter of the region is 36 feet.
theorem perimeter_of_staircase_region : 
  (num_sides * staircase_region.side_length + 
    15 + -- length added to complete the larger rectangle assuming x = 15
    9   -- length added to complete the larger rectangle assuming y = 9
  ) = 36 := 
by
  -- Given and facts are already logically considered to prove (conditions and right angles are trivial)
  sorry

end perimeter_of_staircase_region_l65_65758


namespace ratio_s_to_t_l65_65088

theorem ratio_s_to_t (b : ℝ) (s t : ℝ)
  (h1 : s = -b / 10)
  (h2 : t = -b / 6) :
  s / t = 3 / 5 :=
by sorry

end ratio_s_to_t_l65_65088


namespace maximum_marks_l65_65439

theorem maximum_marks (M : ℝ) (h1 : 212 + 25 = 237) (h2 : 0.30 * M = 237) : M = 790 := 
by
  sorry

end maximum_marks_l65_65439


namespace find_LN_l65_65629

noncomputable def LM : ℝ := 25
noncomputable def sinN : ℝ := 4 / 5

theorem find_LN (LN : ℝ) (h_sin : sinN = LM / LN) : LN = 125 / 4 :=
by
  sorry

end find_LN_l65_65629


namespace simplify_and_evaluate_expression_l65_65855

theorem simplify_and_evaluate_expression (a b : ℝ) (h : (a + 1)^2 + |b + 1| = 0) : 
  1 - (a^2 + 2 * a * b + b^2) / (a^2 - a * b) / ((a + b) / (a - b)) = -1 := 
sorry

end simplify_and_evaluate_expression_l65_65855


namespace geometric_seq_xyz_eq_neg_two_l65_65877

open Real

noncomputable def geometric_seq (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_seq_xyz_eq_neg_two (x y z : ℝ) :
  geometric_seq (-1) x y z (-2) → x * y * z = -2 :=
by
  intro h
  obtain ⟨r, hx, hy, hz, he⟩ := h
  rw [hx, hy, hz, he] at *
  sorry

end geometric_seq_xyz_eq_neg_two_l65_65877


namespace compute_fraction_l65_65018

theorem compute_fraction :
  ( (11^4 + 400) * (25^4 + 400) * (37^4 + 400) * (49^4 + 400) * (61^4 + 400) ) /
  ( (5^4 + 400) * (17^4 + 400) * (29^4 + 400) * (41^4 + 400) * (53^4 + 400) ) = 799 := 
by
  sorry

end compute_fraction_l65_65018


namespace range_of_a_for_monotonicity_l65_65228

noncomputable def f (x : ℝ) (a : ℝ) := (Real.sqrt (x^2 + 1)) - a * x

theorem range_of_a_for_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x a < f y a) ↔ a ≥ 1 := sorry

end range_of_a_for_monotonicity_l65_65228


namespace common_factor_polynomials_l65_65632

theorem common_factor_polynomials (a : ℝ) :
  (∀ p : ℝ, p ≠ 0 ∧ 
           (p^3 - p - a = 0) ∧ 
           (p^2 + p - a = 0)) → 
  (a = 0 ∨ a = 10 ∨ a = -2) := by
  sorry

end common_factor_polynomials_l65_65632


namespace max_value_f_l65_65514

noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 + x + 1)

theorem max_value_f : ∀ x : ℝ, f x ≤ 4 / 3 :=
sorry

end max_value_f_l65_65514


namespace sufficient_but_not_necessary_l65_65624

variable (p q : Prop)

theorem sufficient_but_not_necessary : (¬p → ¬(p ∧ q)) ∧ (¬(¬p) → ¬(p ∧ q) → False) :=
by {
  sorry
}

end sufficient_but_not_necessary_l65_65624


namespace gigi_ate_33_bananas_l65_65544

def gigi_bananas (total_bananas : ℕ) (days : ℕ) (diff : ℕ) (bananas_day_7 : ℕ) : Prop :=
  ∃ b, (days * b + diff * ((days * (days - 1)) / 2)) = total_bananas ∧ 
       (b + 6 * diff) = bananas_day_7

theorem gigi_ate_33_bananas :
  gigi_bananas 150 7 4 33 :=
by {
  sorry
}

end gigi_ate_33_bananas_l65_65544


namespace weight_of_A_l65_65400

theorem weight_of_A
  (W_A W_B W_C W_D W_E : ℕ)
  (H_A H_B H_C H_D : ℕ)
  (Age_A Age_B Age_C Age_D : ℕ)
  (hw1 : (W_A + W_B + W_C) / 3 = 84)
  (hh1 : (H_A + H_B + H_C) / 3 = 170)
  (ha1 : (Age_A + Age_B + Age_C) / 3 = 30)
  (hw2 : (W_A + W_B + W_C + W_D) / 4 = 80)
  (hh2 : (H_A + H_B + H_C + H_D) / 4 = 172)
  (ha2 : (Age_A + Age_B + Age_C + Age_D) / 4 = 28)
  (hw3 : (W_B + W_C + W_D + W_E) / 4 = 79)
  (hh3 : (H_B + H_C + H_D + H_E) / 4 = 173)
  (ha3 : (Age_B + Age_C + Age_D + (Age_A - 3)) / 4 = 27)
  (hw4 : W_E = W_D + 7)
  : W_A = 79 := 
sorry

end weight_of_A_l65_65400


namespace sum_of_cubes_l65_65532

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 20) : x^3 + y^3 = 1008 := 
sorry

end sum_of_cubes_l65_65532


namespace find_divisor_l65_65268

theorem find_divisor (n x : ℕ) (h1 : n = 3) (h2 : (n / x : ℝ) * 12 = 9): x = 4 := by
  sorry

end find_divisor_l65_65268


namespace option_C_correct_l65_65403

variable (a b : ℝ)

theorem option_C_correct (h : a > b) : -15 * a < -15 * b := 
  sorry

end option_C_correct_l65_65403


namespace timothy_tea_cups_l65_65428

theorem timothy_tea_cups (t : ℕ) (h : 6 * t + 60 = 120) : t + 12 = 22 :=
by
  sorry

end timothy_tea_cups_l65_65428


namespace percent_profit_l65_65277

theorem percent_profit (C S : ℝ) (h : 60 * C = 50 * S):
  (((S - C) / C) * 100) = 20 :=
by 
  sorry

end percent_profit_l65_65277


namespace not_sufficient_not_necessary_l65_65016

theorem not_sufficient_not_necessary (a : ℝ) :
  ¬ ((a^2 > 1) → (1/a > 0)) ∧ ¬ ((1/a > 0) → (a^2 > 1)) := sorry

end not_sufficient_not_necessary_l65_65016


namespace gcd_lcm_45_75_l65_65914

theorem gcd_lcm_45_75 : gcd 45 75 = 15 ∧ lcm 45 75 = 1125 :=
by sorry

end gcd_lcm_45_75_l65_65914


namespace jack_pays_back_expected_amount_l65_65734

-- Definitions from the conditions
def principal : ℝ := 1200
def interest_rate : ℝ := 0.10

-- Definition for proof
def interest : ℝ := principal * interest_rate
def total_amount : ℝ := principal + interest

-- Lean statement for the proof problem
theorem jack_pays_back_expected_amount : total_amount = 1320 := by
  sorry

end jack_pays_back_expected_amount_l65_65734


namespace parabola_no_real_intersection_l65_65041

theorem parabola_no_real_intersection (a b c : ℝ) (h₁ : a = 1) (h₂ : b = -4) (h₃ : c = 5) :
  ∀ (x : ℝ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end parabola_no_real_intersection_l65_65041


namespace polynomial_degree_rational_coefficients_l65_65593

theorem polynomial_degree_rational_coefficients :
  ∃ p : Polynomial ℚ,
    (Polynomial.aeval (2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (-2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (3 + Real.sqrt 11) p = 0) ∧
    (Polynomial.aeval (3 - Real.sqrt 11) p = 0) ∧
    p.degree = 6 :=
sorry

end polynomial_degree_rational_coefficients_l65_65593


namespace train_speed_l65_65393

theorem train_speed (length : ℝ) (time : ℝ) (speed : ℝ) (h_length : length = 975) (h_time : time = 48) (h_speed : speed = length / time * 3.6) : 
  speed = 73.125 := 
by 
  sorry

end train_speed_l65_65393


namespace geometric_series_terms_l65_65740

theorem geometric_series_terms 
    (b1 q : ℝ)
    (h₁ : (b1^2 / (1 + q + q^2)) = 12)
    (h₂ : (b1^2 / (1 + q^2)) = (36 / 5)) :
    (b1 = 3 ∨ b1 = -3) ∧ q = -1/2 :=
by
  sorry

end geometric_series_terms_l65_65740


namespace three_digit_subtraction_l65_65223

theorem three_digit_subtraction (c d : ℕ) (H1 : 0 ≤ c ∧ c ≤ 9) (H2 : 0 ≤ d ∧ d ≤ 9) :
  (745 - (300 + c * 10 + 4) = (400 + d * 10 + 1)) ∧ ((4 + 1) - d % 11 = 0) → 
  c + d = 14 := 
sorry

end three_digit_subtraction_l65_65223


namespace area_of_rectangular_field_l65_65880

theorem area_of_rectangular_field 
  (P L W : ℕ) 
  (hP : P = 120) 
  (hL : L = 3 * W) 
  (hPerimeter : 2 * L + 2 * W = P) : 
  (L * W = 675) :=
by 
  sorry

end area_of_rectangular_field_l65_65880


namespace tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l65_65640

-- Conditions
variables {O : ℝ × ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
          {P Q : ℝ × ℝ} (p : ℝ)
          (hp : 0 < p)
          (hA : A.1 ^ 2 = 2 * p * A.2)
          (hB : B = (0, -1))
          (hP : P.2 = P.1 ^ 2 / (2 * p))
          (hQ : Q.2 = Q.1 ^ 2 / (2 * p))

-- Proof problem statements
theorem tangent_line_AB
  (hAB_tangent : ∀ x : ℝ, x ^ 2 / (2 * p) = 2 * x - 1 → x = 1) : true :=
by sorry

theorem op_oq_leq_oa_squared 
  (h_op_oq_leq : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + (P.1 ^ 2 / (2 * p)) ^ 2) * (Q.1 ^ 2 + (Q.1 ^ 2 / (2 * p)) ^ 2) ≤ 2) : true :=
by sorry

theorem bp_bq_gt_ba_squared 
  ( h_bp_bq_gt : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + ((P.1 ^ 2 / (2 * p)) + 1) ^ 2) * (Q.1 ^ 2 + ((Q.1 ^ 2 / (2 * p)) +1 ) ^ 2) > 5 ) : true :=
by sorry

end tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l65_65640


namespace min_value_fraction_l65_65697

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y = x + 2 * y + 6) : 
  (∃ (z : ℝ), z = 1 / x + 1 / (2 * y) ∧ z ≥ 1 / 3) :=
sorry

end min_value_fraction_l65_65697


namespace smallest_natrural_number_cube_ends_888_l65_65317

theorem smallest_natrural_number_cube_ends_888 :
  ∃ n : ℕ, (n^3 % 1000 = 888) ∧ (∀ m : ℕ, (m^3 % 1000 = 888) → n ≤ m) := 
sorry

end smallest_natrural_number_cube_ends_888_l65_65317


namespace num_new_books_not_signed_l65_65037

theorem num_new_books_not_signed (adventure_books mystery_books science_fiction_books non_fiction_books used_books signed_books : ℕ)
    (h1 : adventure_books = 13)
    (h2 : mystery_books = 17)
    (h3 : science_fiction_books = 25)
    (h4 : non_fiction_books = 10)
    (h5 : used_books = 42)
    (h6 : signed_books = 10) : 
    (adventure_books + mystery_books + science_fiction_books + non_fiction_books) - used_books - signed_books = 13 := 
by
  sorry

end num_new_books_not_signed_l65_65037


namespace championship_winner_is_902_l65_65025

namespace BasketballMatch

inductive Class : Type
| c901
| c902
| c903
| c904

open Class

def A_said (champ third : Class) : Prop :=
  champ = c902 ∧ third = c904

def B_said (fourth runner_up : Class) : Prop :=
  fourth = c901 ∧ runner_up = c903

def C_said (third champ : Class) : Prop :=
  third = c903 ∧ champ = c904

def half_correct (P Q : Prop) : Prop := 
  (P ∧ ¬Q) ∨ (¬P ∧ Q)

theorem championship_winner_is_902 (A_third B_fourth B_runner_up C_third : Class) 
  (H_A : half_correct (A_said c902 A_third) (A_said A_third c902))
  (H_B : half_correct (B_said B_fourth B_runner_up) (B_said B_runner_up B_fourth))
  (H_C : half_correct (C_said C_third c904) (C_said c904 C_third)) :
  ∃ winner, winner = c902 :=
sorry

end BasketballMatch

end championship_winner_is_902_l65_65025


namespace max_value_y_l65_65327

theorem max_value_y (x y : ℕ) (h₁ : 9 * (x + y) > 17 * x) (h₂ : 15 * x < 8 * (x + y)) :
  y ≤ 112 :=
sorry

end max_value_y_l65_65327


namespace cleaning_project_l65_65431

theorem cleaning_project (x : ℕ) : 12 + x = 2 * (15 - x) := sorry

end cleaning_project_l65_65431


namespace ratio_square_l65_65800

theorem ratio_square (x y : ℕ) (h1 : x * (x + y) = 40) (h2 : y * (x + y) = 90) (h3 : 2 * y = 3 * x) : (x + y) ^ 2 = 100 := 
by 
  sorry

end ratio_square_l65_65800


namespace fractional_identity_l65_65110

theorem fractional_identity (m n r t : ℚ) 
  (h₁ : m / n = 5 / 2) 
  (h₂ : r / t = 8 / 5) : 
  (2 * m * r - 3 * n * t) / (5 * n * t - 4 * m * r) = -5 / 11 :=
by 
  sorry

end fractional_identity_l65_65110


namespace bob_is_47_5_l65_65057

def bob_age (a b : ℝ) := b = 3 * a - 20
def sum_of_ages (a b : ℝ) := b + a = 70

theorem bob_is_47_5 (a b : ℝ) (h1 : bob_age a b) (h2 : sum_of_ages a b) : b = 47.5 :=
by
  sorry

end bob_is_47_5_l65_65057


namespace distance_between_trees_l65_65076

-- Variables representing the total length of the yard and the number of trees.
variable (length_of_yard : ℕ) (number_of_trees : ℕ)

-- The given conditions
def yard_conditions (length_of_yard number_of_trees : ℕ) :=
  length_of_yard = 700 ∧ number_of_trees = 26

-- The proof statement: If the yard is 700 meters long and there are 26 trees, 
-- then the distance between two consecutive trees is 28 meters.
theorem distance_between_trees (length_of_yard : ℕ) (number_of_trees : ℕ)
  (h : yard_conditions length_of_yard number_of_trees) : 
  (length_of_yard / (number_of_trees - 1)) = 28 := 
by
  sorry

end distance_between_trees_l65_65076


namespace rose_bushes_planted_l65_65748

-- Define the conditions as variables
variable (current_bushes planted_bushes total_bushes : Nat)
variable (h1 : current_bushes = 2) (h2 : total_bushes = 6)
variable (h3 : total_bushes = current_bushes + planted_bushes)

theorem rose_bushes_planted : planted_bushes = 4 := by
  sorry

end rose_bushes_planted_l65_65748


namespace inequality_imply_positive_a_l65_65462

theorem inequality_imply_positive_a 
  (a b d : ℝ) (hb : b ≠ 0) (hd : d ≠ 0) (h_d_pos : d > 0) 
  (h : a / b > -3 / (2 * d)) : a > 0 :=
sorry

end inequality_imply_positive_a_l65_65462


namespace ypsilon_calendar_l65_65940

theorem ypsilon_calendar (x y z : ℕ) 
  (h1 : 28 * x + 30 * y + 31 * z = 365) : x + y + z = 12 :=
sorry

end ypsilon_calendar_l65_65940


namespace length_of_faster_train_proof_l65_65443

-- Definitions based on the given conditions
def faster_train_speed_kmh := 72 -- in km/h
def slower_train_speed_kmh := 36 -- in km/h
def time_to_cross_seconds := 18 -- in seconds

-- Conversion factor from km/h to m/s
def kmh_to_ms := 5 / 18

-- Define the relative speed in m/s
def relative_speed_ms := (faster_train_speed_kmh - slower_train_speed_kmh) * kmh_to_ms

-- Length of the faster train in meters
def length_of_faster_train := relative_speed_ms * time_to_cross_seconds

-- The theorem statement for the Lean prover
theorem length_of_faster_train_proof : length_of_faster_train = 180 := by
  sorry

end length_of_faster_train_proof_l65_65443


namespace percentage_increase_formula_l65_65850

theorem percentage_increase_formula (A B C : ℝ) (h1 : A = 3 * B) (h2 : C = B - 30) :
  100 * ((A - C) / C) = 200 + 9000 / C := 
by 
  sorry

end percentage_increase_formula_l65_65850


namespace elasticity_ratio_is_correct_l65_65278

-- Definitions of the given elasticities
def e_OGBR_QN : ℝ := 1.27
def e_OGBR_PN : ℝ := 0.76

-- Theorem stating the ratio of elasticities equals 1.7
theorem elasticity_ratio_is_correct : (e_OGBR_QN / e_OGBR_PN) = 1.7 := sorry

end elasticity_ratio_is_correct_l65_65278


namespace find_calories_per_slice_l65_65216

/-- Defining the number of slices and their respective calories. -/
def slices_in_cake : ℕ := 8
def calories_per_brownie : ℕ := 375
def brownies_in_pan : ℕ := 6
def extra_calories_in_cake : ℕ := 526

/-- Defining the total calories in cake and brownies -/
def total_calories_in_brownies : ℕ := brownies_in_pan * calories_per_brownie
def total_calories_in_cake (c : ℕ) : ℕ := slices_in_cake * c

/-- The equation from the given problem -/
theorem find_calories_per_slice (c : ℕ) :
  total_calories_in_cake c = total_calories_in_brownies + extra_calories_in_cake → c = 347 :=
by
  sorry

end find_calories_per_slice_l65_65216


namespace dice_product_composite_probability_l65_65143

def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ m : ℕ, m > 1 ∧ m < n ∧ n % m = 0

-- This function calculates the probability of an event occurring by counting the favorable and total outcomes.
def probability (favorable total : ℕ) : ℚ :=
  favorable / total

noncomputable def probability_of_composite_product : ℚ :=
  probability 1283 1296

theorem dice_product_composite_probability : probability_of_composite_product = 1283 / 1296 := sorry

end dice_product_composite_probability_l65_65143


namespace arithmetic_sequence_common_difference_l65_65272

theorem arithmetic_sequence_common_difference  (a_n : ℕ → ℝ)
  (h1 : a_n 1 + a_n 6 = 12)
  (h2 : a_n 4 = 7) :
  ∃ d : ℝ, ∀ n : ℕ, a_n n = a_n 1 + (n - 1) * d ∧ d = 2 := 
sorry

end arithmetic_sequence_common_difference_l65_65272


namespace ratio_of_x_to_y_l65_65702

theorem ratio_of_x_to_y (x y : ℤ) (h : (12 * x - 5 * y) / (17 * x - 3 * y) = 5 / 7) : x / y = -20 :=
by
  sorry

end ratio_of_x_to_y_l65_65702


namespace identity_function_l65_65231

theorem identity_function (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (y - f x) = f x - 2 * x + f (f y)) :
  ∀ y : ℝ, f y = y :=
by 
  sorry

end identity_function_l65_65231


namespace abs_not_eq_three_implies_x_not_eq_three_l65_65954

theorem abs_not_eq_three_implies_x_not_eq_three (x : ℝ) (h : |x| ≠ 3) : x ≠ 3 :=
sorry

end abs_not_eq_three_implies_x_not_eq_three_l65_65954


namespace age_problem_l65_65832

theorem age_problem (x y : ℕ) (h1 : y - 5 = 2 * (x - 5)) (h2 : x + y + 16 = 50) : x = 13 :=
by sorry

end age_problem_l65_65832


namespace annual_expenditure_l65_65320

theorem annual_expenditure (x y : ℝ) (h1 : y = 0.8 * x + 0.1) (h2 : x = 15) : y = 12.1 :=
by
  sorry

end annual_expenditure_l65_65320


namespace math_competition_question_1_math_competition_question_2_l65_65545

noncomputable def participant_score_probabilities : Prop :=
  let P1 := (3 / 5)^2 * (2 / 5)^2
  let P2 := 2 * (3 / 5) * (2 / 5)
  let P3 := 2 * (3 / 5) * (2 / 5)^2
  let P4 := (3 / 5)^2
  P1 + P2 + P3 + P4 = 208 / 625

noncomputable def winning_probabilities : Prop :=
  let P_100_or_more := (4 / 5)^8 * (3 / 5)^3 + 3 * (4 / 5)^8 * (3 / 5)^2 * (2 / 5) + 
                      (8 * (4 / 5)^7 * (1/5) * (3 / 5)^3 + 
                      28 * (4 / 5)^6 * (1/5)^2 * (3 / 5)^3)
  let winning_if_100_or_more := P_100_or_more * (9 / 10)
  let winning_if_less_100 := (1 - P_100_or_more) * (2 / 5)
  winning_if_100_or_more + winning_if_less_100 ≥ 1 / 2

theorem math_competition_question_1 : participant_score_probabilities :=
by sorry

theorem math_competition_question_2 : winning_probabilities :=
by sorry

end math_competition_question_1_math_competition_question_2_l65_65545


namespace kamal_chemistry_marks_l65_65626

variables (english math physics biology average total numSubjects : ℕ)

theorem kamal_chemistry_marks 
  (marks_in_english : english = 66)
  (marks_in_math : math = 65)
  (marks_in_physics : physics = 77)
  (marks_in_biology : biology = 75)
  (avg_marks : average = 69)
  (number_of_subjects : numSubjects = 5)
  (total_marks_known : total = 283) :
  ∃ chemistry : ℕ, chemistry = 62 := 
by 
  sorry

end kamal_chemistry_marks_l65_65626


namespace max_rectangle_area_l65_65559

theorem max_rectangle_area (P : ℕ) (hP : P = 40) (l w : ℕ) (h : 2 * l + 2 * w = P) : ∃ A, A = l * w ∧ ∀ l' w', 2 * l' + 2 * w' = P → l' * w' ≤ 100 :=
by 
  sorry

end max_rectangle_area_l65_65559


namespace strawberry_cost_l65_65252

theorem strawberry_cost (price_per_basket : ℝ) (num_baskets : ℕ) (total_cost : ℝ)
  (h1 : price_per_basket = 16.50) (h2 : num_baskets = 4) : total_cost = 66.00 :=
by
  sorry

end strawberry_cost_l65_65252


namespace cubic_sum_l65_65347

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = 1) (h3 : a * b * c = -2) : 
  a^3 + b^3 + c^3 = -6 :=
sorry

end cubic_sum_l65_65347


namespace delaney_bus_miss_theorem_l65_65092

def delaneyMissesBus : Prop :=
  let busDeparture := 8 * 60               -- bus departure time in minutes (8:00 a.m.)
  let travelTime := 30                     -- travel time in minutes
  let departureTime := 7 * 60 + 50         -- departure time from home in minutes (7:50 a.m.)
  let arrivalTime := departureTime + travelTime -- arrival time at the pick-up point
  arrivalTime - busDeparture = 20 -- he misses the bus by 20 minutes

theorem delaney_bus_miss_theorem : delaneyMissesBus := sorry

end delaney_bus_miss_theorem_l65_65092


namespace optionD_is_deductive_l65_65280

-- Conditions related to the reasoning options
inductive ReasoningProcess where
  | optionA : ReasoningProcess
  | optionB : ReasoningProcess
  | optionC : ReasoningProcess
  | optionD : ReasoningProcess

-- Definitions matching the equivalent Lean problem
def isDeductiveReasoning (rp : ReasoningProcess) : Prop :=
  match rp with
  | ReasoningProcess.optionA => False
  | ReasoningProcess.optionB => False
  | ReasoningProcess.optionC => False
  | ReasoningProcess.optionD => True

-- The proposition we need to prove
theorem optionD_is_deductive :
  isDeductiveReasoning ReasoningProcess.optionD = True := by
  sorry

end optionD_is_deductive_l65_65280


namespace evaluate_f_at_2_l65_65039

-- Define the polynomial function f(x)
def f (x : ℝ) : ℝ := 3 * x^6 - 2 * x^5 + x^3 + 1

theorem evaluate_f_at_2 : f 2 = 34 :=
by
  -- Insert proof here
  sorry

end evaluate_f_at_2_l65_65039


namespace quadrilateral_divided_similarity_iff_trapezoid_l65_65942

noncomputable def convex_quadrilateral (A B C D : Type) : Prop := sorry
noncomputable def is_trapezoid (A B C D : Type) : Prop := sorry
noncomputable def similar_quadrilaterals (E F A B C D : Type) : Prop := sorry

theorem quadrilateral_divided_similarity_iff_trapezoid {A B C D E F : Type}
  (h1 : convex_quadrilateral A B C D)
  (h2 : similar_quadrilaterals E F A B C D): 
  is_trapezoid A B C D ↔ similar_quadrilaterals E F A B C D :=
sorry

end quadrilateral_divided_similarity_iff_trapezoid_l65_65942


namespace correct_value_of_3_dollar_neg4_l65_65988

def special_operation (x y : Int) : Int :=
  x * (y + 2) + x * y + x

theorem correct_value_of_3_dollar_neg4 : special_operation 3 (-4) = -15 :=
by
  sorry

end correct_value_of_3_dollar_neg4_l65_65988


namespace abs_eq_solutions_l65_65177

theorem abs_eq_solutions (x : ℝ) (hx : |x - 5| = 3 * x + 6) :
  x = -11 / 2 ∨ x = -1 / 4 :=
sorry

end abs_eq_solutions_l65_65177


namespace lcm_1_to_10_l65_65421

theorem lcm_1_to_10 : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
by
  sorry

end lcm_1_to_10_l65_65421


namespace leftmost_rectangle_is_B_l65_65814

def isLeftmostRectangle (wA wB wC wD wE : ℕ) : Prop := 
  wB < wD ∧ wB < wE

theorem leftmost_rectangle_is_B :
  let wA := 5
  let wB := 2
  let wC := 4
  let wD := 9
  let wE := 10
  let xA := 2
  let xB := 1
  let xC := 7
  let xD := 6
  let xE := 4
  let yA := 8
  let yB := 6
  let yC := 3
  let yD := 5
  let yE := 7
  let zA := 10
  let zB := 9
  let zC := 0
  let zD := 11
  let zE := 2
  isLeftmostRectangle wA wB wC wD wE :=
by
  simp only
  sorry

end leftmost_rectangle_is_B_l65_65814


namespace find_difference_l65_65522

noncomputable def g : ℝ → ℝ := sorry    -- Definition of the function g (since it's graph-based and specific)

-- Given conditions
variables (c d : ℝ)
axiom h1 : Function.Injective g          -- g is an invertible function (injective functions have inverses)
axiom h2 : g c = d
axiom h3 : g d = 6

-- Theorem to prove
theorem find_difference : c - d = -2 :=
by {
  -- sorry is needed since the exact proof steps are not provided
  sorry
}

end find_difference_l65_65522


namespace train_passing_time_l65_65711

-- Definitions based on the conditions
def length_T1 : ℕ := 800
def speed_T1_kmph : ℕ := 108
def length_T2 : ℕ := 600
def speed_T2_kmph : ℕ := 72

-- Converting kmph to mps
def convert_kmph_to_mps (speed_kmph : ℕ) : ℕ := speed_kmph * 1000 / 3600
def speed_T1_mps : ℕ := convert_kmph_to_mps speed_T1_kmph
def speed_T2_mps : ℕ := convert_kmph_to_mps speed_T2_kmph

-- Calculating relative speed and total length
def relative_speed_T1_T2 : ℕ := speed_T1_mps - speed_T2_mps
def total_length_T1_T2 : ℕ := length_T1 + length_T2

-- Proving the time to pass
theorem train_passing_time : total_length_T1_T2 / relative_speed_T1_T2 = 140 := by
  sorry

end train_passing_time_l65_65711


namespace find_some_number_l65_65139

def simplify_expr (x : ℚ) : Prop :=
  1 / 2 + ((2 / 3 * (3 / 8)) + x) - (8 / 16) = 4.25

theorem find_some_number :
  ∃ x : ℚ, simplify_expr x ∧ x = 4 :=
by
  sorry

end find_some_number_l65_65139


namespace floor_width_l65_65161

theorem floor_width
  (widthX lengthX : ℝ) (widthY lengthY : ℝ)
  (hX : widthX = 10) (lX : lengthX = 18) (lY : lengthY = 20)
  (h : lengthX * widthX = lengthY * widthY) :
  widthY = 9 := 
by
  -- proof goes here
  sorry

end floor_width_l65_65161


namespace expression_value_l65_65742

theorem expression_value (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
by
  sorry

end expression_value_l65_65742


namespace coffee_consumption_l65_65014

theorem coffee_consumption (h1 h2 g1 h3: ℕ) (k : ℕ) (g2 : ℕ) :
  (k = h1 * g1) → (h1 = 9) → (g1 = 2) → (h2 = 6) → (k / h2 = g2) → (g2 = 3) :=
by
  sorry

end coffee_consumption_l65_65014


namespace solution_set_inequality_l65_65688

theorem solution_set_inequality {a b c x : ℝ} (h1 : a < 0)
  (h2 : -b / a = 1 + 2) (h3 : c / a = 1 * 2) :
  a - c * (x^2 - x - 1) - b * x ≥ 0 ↔ x ≤ -3 / 2 ∨ x ≥ 1 := by
  sorry

end solution_set_inequality_l65_65688


namespace myopia_relation_l65_65363

def myopia_data := 
  [(1.00, 100), (0.50, 200), (0.25, 400), (0.20, 500), (0.10, 1000)]

noncomputable def myopia_function (x : ℝ) : ℝ :=
  100 / x

theorem myopia_relation (h₁ : 100 = (1.00 : ℝ) * 100)
    (h₂ : 100 = (0.50 : ℝ) * 200)
    (h₃ : 100 = (0.25 : ℝ) * 400)
    (h₄ : 100 = (0.20 : ℝ) * 500)
    (h₅ : 100 = (0.10 : ℝ) * 1000) :
  (∀ x > 0, myopia_function x = 100 / x) ∧ (myopia_function 250 = 0.4) :=
by
  sorry

end myopia_relation_l65_65363


namespace sum_f_alpha_beta_gamma_neg_l65_65614

theorem sum_f_alpha_beta_gamma_neg (f : ℝ → ℝ)
  (h_f : ∀ x, f x = -x - x^3)
  (α β γ : ℝ)
  (h1 : α + β > 0)
  (h2 : β + γ > 0)
  (h3 : γ + α > 0) :
  f α + f β + f γ < 0 := 
sorry

end sum_f_alpha_beta_gamma_neg_l65_65614


namespace computer_cost_l65_65872

theorem computer_cost (C : ℝ) (h1 : 0.10 * C = a) (h2 : 3 * C = b) (h3 : b - 1.10 * C = 2700) : 
  C = 2700 / 2.90 :=
by
  sorry

end computer_cost_l65_65872


namespace find_number_l65_65351

theorem find_number (x : ℝ) (h : x - x / 3 = x - 24) : x = 72 := 
by 
  sorry

end find_number_l65_65351


namespace fraction_milk_in_mug1_is_one_fourth_l65_65813

-- Condition definitions
def initial_tea_mug1 := 6 -- ounces
def initial_milk_mug2 := 6 -- ounces
def tea_transferred_mug1_to_mug2 := initial_tea_mug1 / 3
def tea_remaining_mug1 := initial_tea_mug1 - tea_transferred_mug1_to_mug2
def total_liquid_mug2 := initial_milk_mug2 + tea_transferred_mug1_to_mug2
def portion_transferred_back := total_liquid_mug2 / 4
def tea_ratio_mug2 := tea_transferred_mug1_to_mug2 / total_liquid_mug2
def milk_ratio_mug2 := initial_milk_mug2 / total_liquid_mug2
def tea_transferred_back := portion_transferred_back * tea_ratio_mug2
def milk_transferred_back := portion_transferred_back * milk_ratio_mug2
def final_tea_mug1 := tea_remaining_mug1 + tea_transferred_back
def final_milk_mug1 := milk_transferred_back
def final_total_liquid_mug1 := final_tea_mug1 + final_milk_mug1

-- Lean statement of the problem
theorem fraction_milk_in_mug1_is_one_fourth :
  final_milk_mug1 / final_total_liquid_mug1 = 1 / 4 :=
by
  sorry

end fraction_milk_in_mug1_is_one_fourth_l65_65813


namespace relationship_f_2011_2014_l65_65333

noncomputable def quadratic_func : Type := ℝ → ℝ

variable (f : quadratic_func)

-- The function is symmetric about x = 2013
axiom symmetry (x : ℝ) : f (2013 + x) = f (2013 - x)

-- The function opens upward (convexity)
axiom opens_upward (a b : ℝ) : f ((a + b) / 2) ≤ (f a + f b) / 2

theorem relationship_f_2011_2014 :
  f 2011 > f 2014 := 
sorry

end relationship_f_2011_2014_l65_65333


namespace area_of_triangle_is_2_l65_65144

-- Define the conditions of the problem
variable (a b c : ℝ)
variable (A B C : ℝ)  -- Angles in radians

-- Conditions for the triangle ABC
variable (sin_A : ℝ) (sin_C : ℝ)
variable (c2sinA_eq_5sinC : c^2 * sin_A = 5 * sin_C)
variable (a_plus_c_squared_eq_16_plus_b_squared : (a + c)^2 = 16 + b^2)
variable (ac_eq_5 : a * c = 5)
variable (cos_B : ℝ)
variable (sin_B : ℝ)

-- Sine and Cosine law results
variable (cos_B_def : cos_B = (a^2 + c^2 - b^2) / (2 * a * c))
variable (sin_B_def : sin_B = Real.sqrt (1 - cos_B^2))

-- Area of the triangle
noncomputable def area_triangle_ABC := (1/2) * a * c * sin_B

-- Theorem to prove the area
theorem area_of_triangle_is_2 :
  area_triangle_ABC a c sin_B = 2 :=
by
  rw [area_triangle_ABC]
  sorry

end area_of_triangle_is_2_l65_65144


namespace inequality_proof_l65_65287

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 :=
by
  sorry

end inequality_proof_l65_65287


namespace smallest_divisor_of_2880_that_gives_perfect_square_is_5_l65_65017

theorem smallest_divisor_of_2880_that_gives_perfect_square_is_5 :
  (∃ x : ℕ, x ≠ 0 ∧ 2880 % x = 0 ∧ (∃ y : ℕ, 2880 / x = y * y) ∧ x = 5) := by
  sorry

end smallest_divisor_of_2880_that_gives_perfect_square_is_5_l65_65017


namespace convert_to_base10_sum_l65_65397

def base8_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 8^2 + d1 * 8^1 + d0 * 8^0

def base13_to_dec (d2 d1 d0 : Nat) : Nat :=
  d2 * 13^2 + d1 * 13^1 + d0 * 13^0

def convert_537_8 : Nat :=
  base8_to_dec 5 3 7

def convert_4C5_13 : Nat :=
  base13_to_dec 4 12 5

theorem convert_to_base10_sum : 
  convert_537_8 + convert_4C5_13 = 1188 := 
by 
  sorry

end convert_to_base10_sum_l65_65397


namespace continuous_implies_defined_defined_does_not_imply_continuous_l65_65796

-- Define function continuity at a point x = a
def continuous_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - a) < δ → abs (f x - f a) < ε

-- Prove that if f is continuous at x = a, then f is defined at x = a
theorem continuous_implies_defined (f : ℝ → ℝ) (a : ℝ) : 
  continuous_at f a → ∃ y, f a = y :=
by
  sorry  -- Proof omitted

-- Prove that the definition of f at x = a does not guarantee continuity at x = a
theorem defined_does_not_imply_continuous (f : ℝ → ℝ) (a : ℝ) :
  (∃ y, f a = y) → ¬ continuous_at f a :=
by
  sorry  -- Proof omitted

end continuous_implies_defined_defined_does_not_imply_continuous_l65_65796


namespace total_number_of_shells_l65_65225

variable (David Mia Ava Alice : ℕ)
variable (hd : David = 15)
variable (hm : Mia = 4 * David)
variable (ha : Ava = Mia + 20)
variable (hAlice : Alice = Ava / 2)

theorem total_number_of_shells :
  David + Mia + Ava + Alice = 195 :=
by
  sorry

end total_number_of_shells_l65_65225


namespace A_share_of_gain_l65_65182

-- Given problem conditions
def investment_A (x : ℝ) : ℝ := x * 12
def investment_B (x : ℝ) : ℝ := 2 * x * 6
def investment_C (x : ℝ) : ℝ := 3 * x * 4
def total_investment (x : ℝ) : ℝ := investment_A x + investment_B x + investment_C x
def total_gain : ℝ := 21000

-- Mathematically equivalent proof problem statement
theorem A_share_of_gain (x : ℝ) : (investment_A x) / (total_investment x) * total_gain = 7000 :=
by
  sorry

end A_share_of_gain_l65_65182


namespace cricketer_runs_l65_65809

theorem cricketer_runs (R x : ℝ) : 
  (R / 85 = 12.4) →
  ((R + x) / 90 = 12.0) →
  x = 26 := 
by
  sorry

end cricketer_runs_l65_65809


namespace price_of_70_cans_l65_65445

noncomputable def regular_price_per_can : ℝ := 0.55
noncomputable def discount_rate_case : ℝ := 0.25
noncomputable def bulk_discount_rate : ℝ := 0.10
noncomputable def cans_per_case : ℕ := 24
noncomputable def total_cans_purchased : ℕ := 70

theorem price_of_70_cans :
  let discounted_price_per_can := regular_price_per_can * (1 - discount_rate_case)
  let discounted_price_for_cases := 48 * discounted_price_per_can
  let bulk_discount := if 70 >= 3 * cans_per_case then discounted_price_for_cases * bulk_discount_rate else 0
  let final_price_for_cases := discounted_price_for_cases - bulk_discount
  let additional_cans := total_cans_purchased % cans_per_case
  let price_for_additional_cans := additional_cans * discounted_price_per_can
  final_price_for_cases + price_for_additional_cans = 26.895 :=
by sorry

end price_of_70_cans_l65_65445


namespace abcd_eq_eleven_l65_65490

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry
noncomputable def d : ℝ := sorry

-- Conditions on a, b, c, d
axiom cond_a : a = Real.sqrt (4 + Real.sqrt (5 + a))
axiom cond_b : b = Real.sqrt (4 - Real.sqrt (5 + b))
axiom cond_c : c = Real.sqrt (4 + Real.sqrt (5 - c))
axiom cond_d : d = Real.sqrt (4 - Real.sqrt (5 - d))

-- Theorem to prove
theorem abcd_eq_eleven : a * b * c * d = 11 :=
by
  sorry

end abcd_eq_eleven_l65_65490


namespace dimes_left_l65_65631

-- Definitions based on the conditions
def Initial_dimes : ℕ := 8
def Sister_borrowed : ℕ := 4
def Friend_borrowed : ℕ := 2

-- The proof problem statement (without the proof)
theorem dimes_left (Initial_dimes Sister_borrowed Friend_borrowed : ℕ) : 
  Initial_dimes = 8 → Sister_borrowed = 4 → Friend_borrowed = 2 →
  Initial_dimes - (Sister_borrowed + Friend_borrowed) = 2 :=
by
  intros
  sorry

end dimes_left_l65_65631


namespace Carol_width_eq_24_l65_65505

-- Given conditions
def Carol_length : ℕ := 5
def Jordan_length : ℕ := 2
def Jordan_width : ℕ := 60

-- Required proof: Carol's width is 24 considering equal areas of both rectangles
theorem Carol_width_eq_24 (w : ℕ) (h : Carol_length * w = Jordan_length * Jordan_width) : w = 24 := 
by sorry

end Carol_width_eq_24_l65_65505


namespace problem1_problem2_l65_65474

-- Define the main assumptions and the proof problem for Lean 4
theorem problem1 (a : ℝ) (h : a ≠ 0) : (a^2)^3 / (-a)^2 = a^4 := sorry

theorem problem2 (a b : ℝ) : (a + 2 * b) * (a + b) - 3 * a * (a + b) = -2 * a^2 + 2 * b^2 := sorry

end problem1_problem2_l65_65474


namespace area_enclosed_by_abs_eq_l65_65909

theorem area_enclosed_by_abs_eq (x y : ℝ) : 
  (|x| + |3 * y| = 12) → (∃ area : ℝ, area = 96) :=
by
  sorry

end area_enclosed_by_abs_eq_l65_65909


namespace num_even_multiples_of_four_perfect_squares_lt_5000_l65_65485

theorem num_even_multiples_of_four_perfect_squares_lt_5000 : 
  ∃ (k : ℕ), k = 17 ∧ ∀ (n : ℕ), (0 < n ∧ 16 * n^2 < 5000) ↔ (1 ≤ n ∧ n ≤ 17) :=
by
  sorry

end num_even_multiples_of_four_perfect_squares_lt_5000_l65_65485


namespace minNumberOfGloves_l65_65373

-- Define the number of participants
def numParticipants : ℕ := 43

-- Define the number of gloves needed per participant
def glovesPerParticipant : ℕ := 2

-- Define the total number of gloves
def totalGloves (participants glovesPerParticipant : ℕ) : ℕ := 
  participants * glovesPerParticipant

-- Theorem proving the minimum number of gloves required
theorem minNumberOfGloves : totalGloves numParticipants glovesPerParticipant = 86 :=
by
  sorry

end minNumberOfGloves_l65_65373


namespace solve_conjugate_l65_65253
open Complex

-- Problem definition:
def Z (a : ℝ) : ℂ := ⟨a, 1⟩  -- Z = a + i

def conj_Z (a : ℝ) : ℂ := ⟨a, -1⟩  -- conjugate of Z

theorem solve_conjugate (a : ℝ) (h : Z a + conj_Z a = 4) : conj_Z 2 = 2 - I := by
  sorry

end solve_conjugate_l65_65253


namespace Bryan_deposit_amount_l65_65032

theorem Bryan_deposit_amount (deposit_mark : ℕ) (deposit_bryan : ℕ)
  (h1 : deposit_mark = 88)
  (h2 : deposit_bryan = 5 * deposit_mark - 40) : 
  deposit_bryan = 400 := 
by
  sorry

end Bryan_deposit_amount_l65_65032


namespace neznaika_mistake_correct_numbers_l65_65243

theorem neznaika_mistake_correct_numbers (N : ℕ) :
  (10 ≤ N) ∧ (N ≤ 99) ∧
  ¬ (N % 30 = 0) ∧
  (N % 3 = 0) ∧
  ¬ (N % 10 = 0) ∧
  ((N % 9 = 0) ∨ (N % 15 = 0) ∨ (N % 18 = 0)) ∧
  (N % 5 ≠ 0) ∧
  (N % 4 ≠ 0) → N = 36 ∨ N = 45 ∨ N = 72 := by
  sorry

end neznaika_mistake_correct_numbers_l65_65243


namespace simplify_tangent_expression_l65_65307

theorem simplify_tangent_expression :
  (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by
  sorry

end simplify_tangent_expression_l65_65307


namespace hiring_probabilities_l65_65521

-- Define the candidates and their abilities
inductive Candidate : Type
| Strong
| Moderate
| Weak

open Candidate

-- Define the ordering rule and hiring rule
def interviewOrders : List (Candidate × Candidate × Candidate) :=
  [(Strong, Moderate, Weak), (Strong, Weak, Moderate), 
   (Moderate, Strong, Weak), (Moderate, Weak, Strong),
   (Weak, Strong, Moderate), (Weak, Moderate, Strong)]

def hiresStrong (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Moderate, Strong, Weak) => true
  | (Moderate, Weak, Strong) => true
  | (Weak, Strong, Moderate) => true
  | _ => false

def hiresModerate (order : Candidate × Candidate × Candidate) : Bool :=
  match order with
  | (Strong, Weak, Moderate) => true
  | (Weak, Moderate, Strong) => true
  | _ => false

-- The main theorem to be proved
theorem hiring_probabilities :
  let orders := interviewOrders
  let p := (orders.filter hiresStrong).length / orders.length
  let q := (orders.filter hiresModerate).length / orders.length
  p = 1 / 2 ∧ q = 1 / 3 := by
  sorry

end hiring_probabilities_l65_65521


namespace problem_remainder_3_l65_65392

theorem problem_remainder_3 :
  88 % 5 = 3 :=
by
  sorry

end problem_remainder_3_l65_65392


namespace solution_set_quadratic_inequality_l65_65313

theorem solution_set_quadratic_inequality :
  {x : ℝ | (x^2 - 3*x + 2) < 0} = {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end solution_set_quadratic_inequality_l65_65313


namespace largest_among_options_l65_65594

theorem largest_among_options :
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  D > A ∧ D > B ∧ D > C ∧ D > E := by
{
  let A := 15679 + (1 / 3579)
  let B := 15679 - (1 / 3579)
  let C := 15679 * (1 / 3579)
  let D := 15679 / (1 / 3579)
  let E := 15679 * 1.03
  sorry
}

end largest_among_options_l65_65594


namespace find_m_l65_65716

theorem find_m (m x_1 x_2 : ℝ) 
  (h1 : x_1^2 + m * x_1 - 3 = 0) 
  (h2 : x_2^2 + m * x_2 - 3 = 0) 
  (h3 : x_1 + x_2 - x_1 * x_2 = 5) : 
  m = -2 :=
sorry

end find_m_l65_65716


namespace B_alone_completion_l65_65893

-- Define the conditions:
def A_efficiency_rel_to_B (A B: ℕ → Prop) : Prop :=
  ∀ (x: ℕ), B x → A (2 * x)

def together_job_completion (A B: ℕ → Prop) : Prop :=
  ∀ (t: ℕ), t = 20 → (∃ (x y : ℕ), B x ∧ A y ∧ (1/x + 1/y = 1/t))

-- Define the theorem:
theorem B_alone_completion (A B: ℕ → Prop) (h1 : A_efficiency_rel_to_B A B) (h2 : together_job_completion A B) :
  ∃ (x: ℕ), B x ∧ x = 30 :=
sorry

end B_alone_completion_l65_65893


namespace problem_statement_l65_65153

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 1

theorem problem_statement (x : ℝ) (h : x ≠ 0) : f x > 0 :=
by sorry

end problem_statement_l65_65153


namespace red_ball_count_l65_65739

theorem red_ball_count (w : ℕ) (f : ℝ) (total : ℕ) (r : ℕ) 
  (hw : w = 60)
  (hf : f = 0.25)
  (ht : total = w / (1 - f))
  (hr : r = total * f) : 
  r = 20 :=
by 
  -- Lean doesn't require a proof for the problem statement
  sorry

end red_ball_count_l65_65739


namespace gnuff_tutoring_rate_l65_65723

theorem gnuff_tutoring_rate (flat_rate : ℕ) (total_paid : ℕ) (minutes : ℕ) :
  flat_rate = 20 → total_paid = 146 → minutes = 18 → (total_paid - flat_rate) / minutes = 7 :=
by
  intros
  sorry

end gnuff_tutoring_rate_l65_65723


namespace pure_imaginary_denom_rationalization_l65_65727

theorem pure_imaginary_denom_rationalization (a : ℝ) : 
  (∃ b : ℝ, 1 - a * Complex.I * Complex.I = b * Complex.I) → a = 0 :=
by
  sorry

end pure_imaginary_denom_rationalization_l65_65727


namespace min_people_for_no_empty_triplet_60_l65_65430

noncomputable def min_people_for_no_empty_triplet (total_chairs : ℕ) : ℕ :=
  if h : total_chairs % 3 = 0 then total_chairs / 3 else sorry

theorem min_people_for_no_empty_triplet_60 :
  min_people_for_no_empty_triplet 60 = 20 :=
by
  sorry

end min_people_for_no_empty_triplet_60_l65_65430


namespace chocolates_bought_in_a_month_l65_65343

theorem chocolates_bought_in_a_month :
  ∀ (chocolates_for_her: ℕ)
    (chocolates_for_sister: ℕ)
    (chocolates_for_charlie: ℕ)
    (weeks_in_a_month: ℕ), 
  weeks_in_a_month = 4 →
  chocolates_for_her = 2 →
  chocolates_for_sister = 1 →
  chocolates_for_charlie = 10 →
  (chocolates_for_her * weeks_in_a_month + chocolates_for_sister * weeks_in_a_month + chocolates_for_charlie) = 22 :=
by
  intros chocolates_for_her chocolates_for_sister chocolates_for_charlie weeks_in_a_month
  intros h_weeks h_her h_sister h_charlie
  sorry

end chocolates_bought_in_a_month_l65_65343


namespace rest_duration_per_kilometer_l65_65049

theorem rest_duration_per_kilometer
  (speed : ℕ)
  (total_distance : ℕ)
  (total_time : ℕ)
  (walking_time : ℕ := total_distance / speed * 60)  -- walking_time in minutes
  (rest_time : ℕ := total_time - walking_time)  -- total resting time in minutes
  (number_of_rests : ℕ := total_distance - 1)  -- number of rests after each kilometer
  (duration_per_rest : ℕ := rest_time / number_of_rests)
  (h1 : speed = 10)
  (h2 : total_distance = 5)
  (h3 : total_time = 50) : 
  (duration_per_rest = 5) := 
sorry

end rest_duration_per_kilometer_l65_65049


namespace grant_received_money_l65_65959

theorem grant_received_money :
  let total_teeth := 20
  let lost_teeth := 2
  let first_tooth_amount := 20
  let other_tooth_amount_per_tooth := 2
  let remaining_teeth := total_teeth - lost_teeth - 1
  let total_amount_received := first_tooth_amount + remaining_teeth * other_tooth_amount_per_tooth
  total_amount_received = 54 :=
by  -- Start the proof mode
  sorry  -- This is where the actual proof would go

end grant_received_money_l65_65959


namespace product_eq_one_of_abs_log_eq_l65_65149

theorem product_eq_one_of_abs_log_eq (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |Real.log a| = |Real.log b|) : a * b = 1 := 
sorry

end product_eq_one_of_abs_log_eq_l65_65149


namespace sarah_dimes_l65_65511

theorem sarah_dimes (d n : ℕ) (h1 : d + n = 50) (h2 : 10 * d + 5 * n = 200) : d = 10 :=
sorry

end sarah_dimes_l65_65511


namespace two_point_form_eq_l65_65303

theorem two_point_form_eq (x y : ℝ) : 
  let A := (5, 6)
  let B := (-1, 2)
  (y - 6) / (2 - 6) = (x - 5) / (-1 - 5) := 
  sorry

end two_point_form_eq_l65_65303


namespace b_negative_l65_65974

variable {R : Type*} [LinearOrderedField R]

theorem b_negative (a b : R) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∀ x : R, 0 ≤ x → (x - a) * (x - b) * (x - (2*a + b)) ≥ 0) : b < 0 := 
sorry

end b_negative_l65_65974


namespace june_spent_on_music_books_l65_65921

theorem june_spent_on_music_books
  (total_budget : ℤ)
  (math_books_cost : ℤ)
  (science_books_cost : ℤ)
  (art_books_cost : ℤ)
  (music_books_cost : ℤ)
  (h_total_budget : total_budget = 500)
  (h_math_books_cost : math_books_cost = 80)
  (h_science_books_cost : science_books_cost = 100)
  (h_art_books_cost : art_books_cost = 160)
  (h_total_cost : music_books_cost = total_budget - (math_books_cost + science_books_cost + art_books_cost)) :
  music_books_cost = 160 :=
sorry

end june_spent_on_music_books_l65_65921


namespace stream_speed_l65_65119

theorem stream_speed (x : ℝ) (d : ℝ) (v_b : ℝ) (t : ℝ) (h : v_b = 8) (h1 : d = 210) (h2 : t = 56) : x = 2 :=
by
  sorry

end stream_speed_l65_65119


namespace original_price_proof_l65_65750

noncomputable def original_price (profit selling_price : ℝ) : ℝ :=
  (profit / 0.20)

theorem original_price_proof (P : ℝ) : 
  original_price 600 (P + 600) = 3000 :=
by
  sorry

end original_price_proof_l65_65750


namespace rental_lower_amount_eq_50_l65_65315

theorem rental_lower_amount_eq_50 (L : ℝ) (total_rent : ℝ) (reduction : ℝ) (rooms_changed : ℕ) (diff_per_room : ℝ)
  (h1 : total_rent = 400)
  (h2 : reduction = 0.25 * total_rent)
  (h3 : rooms_changed = 10)
  (h4 : diff_per_room = reduction / ↑rooms_changed)
  (h5 : 60 - L = diff_per_room) :
  L = 50 :=
  sorry

end rental_lower_amount_eq_50_l65_65315


namespace alice_bob_not_next_to_each_other_l65_65086

open Nat

theorem alice_bob_not_next_to_each_other (A B C D E : Type) :
  let arrangements := 5!
  let together := 4! * 2
  arrangements - together = 72 :=
by
  let arrangements := 5!
  let together := 4! * 2
  sorry

end alice_bob_not_next_to_each_other_l65_65086


namespace inequality_must_hold_l65_65861

theorem inequality_must_hold (a b c : ℝ) (h : (a / c^2) > (b / c^2)) (hc : c ≠ 0) : a^2 > b^2 :=
sorry

end inequality_must_hold_l65_65861


namespace time_after_2345_minutes_l65_65924

-- Define the constants
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24
def startTime : Nat := 0 -- midnight on January 1, 2022, treated as 0 minutes.

-- Prove the equivalent time after 2345 minutes
theorem time_after_2345_minutes :
    let totalMinutes := 2345
    let totalHours := totalMinutes / minutesInHour
    let remainingMinutes := totalMinutes % minutesInHour
    let totalDays := totalHours / hoursInDay
    let remainingHours := totalHours % hoursInDay
    startTime + totalDays * hoursInDay * minutesInHour + remainingHours * minutesInHour + remainingMinutes = startTime + 1 * hoursInDay * minutesInHour + 15 * minutesInHour + 5 :=
    by
    sorry

end time_after_2345_minutes_l65_65924


namespace symmetric_line_l65_65055

theorem symmetric_line (x y : ℝ) : 
  (∀ (x y  : ℝ), 2 * x + y - 1 = 0) ∧ (∀ (x  : ℝ), x = 1) → (2 * x - y - 3 = 0) :=
by
  sorry

end symmetric_line_l65_65055


namespace painting_clock_57_painting_clock_1913_l65_65605

-- Part (a)
theorem painting_clock_57 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 57) % 12))) :
  ∃ m : ℕ, m = 4 :=
by { sorry }

-- Part (b)
theorem painting_clock_1913 (h : ∀ n : ℕ, (n = 12 ∨ (exists k : ℕ, n = (12 + k * 1913) % 12))) :
  ∃ m : ℕ, m = 12 :=
by { sorry }

end painting_clock_57_painting_clock_1913_l65_65605


namespace line_sum_slope_intercept_l65_65065

theorem line_sum_slope_intercept (m b : ℝ) (x y : ℝ)
  (hm : m = 3)
  (hpoint : (x, y) = (-2, 4))
  (heq : y = m * x + b) :
  m + b = 13 :=
by
  sorry

end line_sum_slope_intercept_l65_65065


namespace parabola_focus_l65_65422

theorem parabola_focus (a : ℝ) (h1 : ∀ x y, x^2 = a * y ↔ y = x^2 / a)
(h2 : focus_coordinates = (0, 5)) : a = 20 := 
sorry

end parabola_focus_l65_65422


namespace f_values_sum_l65_65291

noncomputable def f : ℝ → ℝ := sorry

-- defining the properties
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- given conditions
axiom f_odd : is_odd f
axiom f_periodic : is_periodic f 2

-- statement to prove
theorem f_values_sum : f 1 + f 2 + f 3 = 0 :=
by
  sorry

end f_values_sum_l65_65291


namespace peter_erasers_l65_65377

theorem peter_erasers (initial_erasers : ℕ) (extra_erasers : ℕ) (final_erasers : ℕ)
  (h1 : initial_erasers = 8) (h2 : extra_erasers = 3) : final_erasers = 11 :=
by
  sorry

end peter_erasers_l65_65377


namespace factorize_one_factorize_two_l65_65293

variable (m x y : ℝ)

-- Problem statement for Question 1
theorem factorize_one (m : ℝ) : 
  2 * m^2 - 8 = 2 * (m + 2) * (m - 2) := 
sorry

-- Problem statement for Question 2
theorem factorize_two (x y : ℝ) : 
  (x + y)^2 - 4 * (x + y) + 4 = (x + y - 2)^2 := 
sorry

end factorize_one_factorize_two_l65_65293


namespace initial_red_marbles_l65_65331

theorem initial_red_marbles (R : ℕ) (blue_marbles_initial : ℕ) (red_marbles_removed : ℕ) :
  blue_marbles_initial = 30 →
  red_marbles_removed = 3 →
  (R - red_marbles_removed) + (blue_marbles_initial - 4 * red_marbles_removed) = 35 →
  R = 20 :=
by
  intros h_blue h_red h_total
  sorry

end initial_red_marbles_l65_65331


namespace dive_has_five_judges_l65_65923

noncomputable def number_of_judges 
  (scores : List ℝ)
  (difficulty : ℝ)
  (point_value : ℝ) : ℕ := sorry

theorem dive_has_five_judges :
  number_of_judges [7.5, 8.0, 9.0, 6.0, 8.8] 3.2 77.76 = 5 :=
by
  sorry

end dive_has_five_judges_l65_65923


namespace T_30_is_13515_l65_65843

def sequence_first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

def sequence_last_element (n : ℕ) : ℕ := sequence_first_element n + n - 1

def sum_sequence_set (n : ℕ) : ℕ :=
  n * (sequence_first_element n + sequence_last_element n) / 2

theorem T_30_is_13515 : sum_sequence_set 30 = 13515 := by
  sorry

end T_30_is_13515_l65_65843


namespace total_reading_materials_l65_65596

def reading_materials (magazines newspapers books pamphlets : ℕ) : ℕ :=
  magazines + newspapers + books + pamphlets

theorem total_reading_materials:
  reading_materials 425 275 150 75 = 925 := by
  sorry

end total_reading_materials_l65_65596


namespace wine_division_l65_65563

theorem wine_division (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0) :
  (∃ k, k = (m + n) / 2 ∧ k * 2 = (m + n) ∧ k % Nat.gcd m n = 0) ↔ 
  (m + n) % 2 = 0 ∧ ((m + n) / 2) % Nat.gcd m n = 0 :=
by
  sorry

end wine_division_l65_65563


namespace university_math_students_l65_65808

theorem university_math_students
  (total_students : ℕ)
  (math_only : ℕ)
  (stats_only : ℕ)
  (both_courses : ℕ)
  (H1 : total_students = 75)
  (H2 : math_only + stats_only + both_courses = total_students)
  (H3 : math_only = 2 * (stats_only + both_courses))
  (H4 : both_courses = 9) :
  math_only + both_courses = 53 :=
by
  sorry

end university_math_students_l65_65808


namespace range_of_a_l65_65825

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → a < x + (1 / x)) → a < 2 :=
by
  sorry

end range_of_a_l65_65825


namespace value_of_b_plus_c_l65_65590

variable {a b c d : ℝ}

theorem value_of_b_plus_c (h1 : a + b = 4) (h2 : c + d = 5) (h3 : a + d = 2) : b + c = 7 :=
sorry

end value_of_b_plus_c_l65_65590


namespace length_of_train_l65_65664

-- declare constants
variables (L S : ℝ)

-- state conditions
def condition1 : Prop := L = S * 50
def condition2 : Prop := L + 500 = S * 100

-- state the theorem to prove
theorem length_of_train (h1 : condition1 L S) (h2 : condition2 L S) : L = 500 :=
by sorry

end length_of_train_l65_65664


namespace area_ratio_l65_65337

theorem area_ratio (l w h : ℝ) (h1 : w * h = 288) (h2 : l * w = 432) (h3 : l * w * h = 5184) :
  (l * h) / (l * w) = 1 / 2 :=
sorry

end area_ratio_l65_65337


namespace georgia_makes_muffins_l65_65969

-- Definitions based on conditions
def muffinRecipeMakes : ℕ := 6
def numberOfStudents : ℕ := 24
def durationInMonths : ℕ := 9

-- Theorem to prove the given problem
theorem georgia_makes_muffins :
  (numberOfStudents / muffinRecipeMakes) * durationInMonths = 36 :=
by
  -- We'll skip the proof with sorry
  sorry

end georgia_makes_muffins_l65_65969


namespace total_journey_length_l65_65518

theorem total_journey_length (y : ℚ)
  (h1 : y * 1 / 4 + 30 + y * 1 / 7 = y) : 
  y = 840 / 17 :=
by 
  sorry

end total_journey_length_l65_65518


namespace kim_total_water_intake_l65_65378

def quarts_to_ounces (q : ℝ) : ℝ := q * 32

theorem kim_total_water_intake :
  (quarts_to_ounces 1.5) + 12 = 60 := 
by
  -- proof step 
  sorry

end kim_total_water_intake_l65_65378


namespace polar_to_rectangular_l65_65784

theorem polar_to_rectangular (r θ : ℝ) (hr : r = 6) (hθ : θ = 5 * Real.pi / 3) :
  (r * Real.cos θ, r * Real.sin θ) = (3, -3 * Real.sqrt 3) :=
by
  -- Definitions and assertions from the conditions
  have cos_theta : Real.cos (5 * Real.pi / 3) = 1 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted
  have sin_theta : Real.sin (5 * Real.pi / 3) = - Real.sqrt 3 / 2 :=
    by sorry  -- detailed trigonometric proof is omitted

  -- Proof that the converted coordinates match the expected result
  rw [hr, hθ, cos_theta, sin_theta]
  simp
  -- Detailed proof steps to verify (6 * (1 / 2), 6 * (- Real.sqrt 3 / 2)) = (3, -3 * Real.sqrt 3) omitted
  sorry

end polar_to_rectangular_l65_65784


namespace fraction_power_multiplication_l65_65324

theorem fraction_power_multiplication :
  ( (1 / 3) ^ 4 * (1 / 5) = 1 / 405 ) :=
by
  sorry

end fraction_power_multiplication_l65_65324


namespace vasya_numbers_l65_65622

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l65_65622


namespace find_angle3_l65_65555

theorem find_angle3 (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 + angle2 = 90)
  (h2 : angle2 + angle3 = 180)
  (h3 : angle1 = 20) :
  angle3 = 110 :=
sorry

end find_angle3_l65_65555


namespace dilution_problem_l65_65871
-- Definitions of the conditions
def volume_initial : ℝ := 15
def concentration_initial : ℝ := 0.60
def concentration_final : ℝ := 0.40
def amount_alcohol_initial : ℝ := volume_initial * concentration_initial

-- Proof problem statement in Lean 4
theorem dilution_problem : 
  ∃ (x : ℝ), x = 7.5 ∧ 
              amount_alcohol_initial = concentration_final * (volume_initial + x) :=
sorry

end dilution_problem_l65_65871


namespace rectangle_semicircle_problem_l65_65508

/--
Rectangle ABCD and a semicircle with diameter AB are coplanar and have nonoverlapping interiors.
Let R denote the region enclosed by the semicircle and the rectangle.
Line ℓ meets the semicircle, segment AB, and segment CD at distinct points P, V, and S, respectively.
Line ℓ divides region R into two regions with areas in the ratio 3:1.
Suppose that AV = 120, AP = 180, and VB = 240.
Prove the length of DA = 90 * sqrt(6).
-/
theorem rectangle_semicircle_problem (DA : ℝ) (AV AP VB : ℝ) (h₁ : AV = 120) (h₂ : AP = 180) (h₃ : VB = 240) :
  DA = 90 * Real.sqrt 6 := by
  sorry

end rectangle_semicircle_problem_l65_65508


namespace binomial_p_value_l65_65372

noncomputable def binomial_expected_value (n : ℕ) (p : ℝ) : ℝ := n * p

theorem binomial_p_value (p : ℝ) : (binomial_expected_value 18 p = 9) → p = 1/2 :=
by
  intro h
  sorry

end binomial_p_value_l65_65372


namespace james_painted_area_l65_65192

-- Define the dimensions of the wall and windows
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window1_height : ℕ := 3
def window1_length : ℕ := 5
def window2_height : ℕ := 2
def window2_length : ℕ := 6

-- Calculate the areas
def wall_area : ℕ := wall_height * wall_length
def window1_area : ℕ := window1_height * window1_length
def window2_area : ℕ := window2_height * window2_length
def total_window_area : ℕ := window1_area + window2_area
def painted_area : ℕ := wall_area - total_window_area

theorem james_painted_area : painted_area = 123 :=
by
  -- The proof is omitted
  sorry

end james_painted_area_l65_65192


namespace evaluate_expression_l65_65666

theorem evaluate_expression (x y z : ℚ) (hx : x = 1 / 2) (hy : y = 1 / 3) (hz : z = -3) :
  (2 * x)^2 * (y^2)^3 * z^2 = 1 / 81 :=
by
  -- Proof omitted
  sorry

end evaluate_expression_l65_65666


namespace fish_remaining_l65_65438

theorem fish_remaining
  (initial_guppies : ℕ)
  (initial_angelfish : ℕ)
  (initial_tiger_sharks : ℕ)
  (initial_oscar_fish : ℕ)
  (sold_guppies : ℕ)
  (sold_angelfish : ℕ)
  (sold_tiger_sharks : ℕ)
  (sold_oscar_fish : ℕ)
  (initial_total : ℕ := initial_guppies + initial_angelfish + initial_tiger_sharks + initial_oscar_fish)
  (sold_total : ℕ := sold_guppies + sold_angelfish + sold_tiger_sharks + sold_oscar_fish)
  (remaining : ℕ := initial_total - sold_total) :
  initial_guppies = 94 →
  initial_angelfish = 76 →
  initial_tiger_sharks = 89 →
  initial_oscar_fish = 58 →
  sold_guppies = 30 →
  sold_angelfish = 48 →
  sold_tiger_sharks = 17 →
  sold_oscar_fish = 24 →
  remaining = 198 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  rw [h1, h2, h3, h4, h5, h6, h7, h8]
  norm_num
  sorry

end fish_remaining_l65_65438


namespace number_13_on_top_after_folds_l65_65040

/-
A 5x5 grid of numbers from 1 to 25 with the following sequence of folds:
1. Fold along the diagonal from bottom-left to top-right
2. Fold the left half over the right half
3. Fold the top half over the bottom half
4. Fold the bottom half over the top half
Prove that the number 13 ends up on top after all folds.
-/

def grid := (⟨ 5, 5 ⟩ : Nat × Nat)

def initial_grid : ℕ → ℕ := λ n => if 1 ≤ n ∧ n ≤ 25 then n else 0

def fold_diagonal (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 1 fold

def fold_left_over_right (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 2 fold

def fold_top_over_bottom (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 3 fold

def fold_bottom_over_top (g : ℕ → ℕ) : ℕ → ℕ := sorry -- implement function to represent step 4 fold

theorem number_13_on_top_after_folds : (fold_bottom_over_top (fold_top_over_bottom (fold_left_over_right (fold_diagonal initial_grid)))) 13 = 13 :=
by {
  sorry
}

end number_13_on_top_after_folds_l65_65040


namespace cost_of_3000_pencils_l65_65300

-- Define the cost per box and the number of pencils per box
def cost_per_box : ℝ := 36
def pencils_per_box : ℕ := 120

-- Define the number of pencils to buy
def pencils_to_buy : ℕ := 3000

-- Define the total cost to prove
def total_cost_to_prove : ℝ := 900

-- The theorem to prove
theorem cost_of_3000_pencils : 
  (cost_per_box / pencils_per_box) * pencils_to_buy = total_cost_to_prove :=
by
  sorry

end cost_of_3000_pencils_l65_65300


namespace total_molecular_weight_correct_l65_65002

-- Defining the molecular weights of elements
def mol_weight_C : ℝ := 12.01
def mol_weight_H : ℝ := 1.01
def mol_weight_Cl : ℝ := 35.45
def mol_weight_O : ℝ := 16.00

-- Defining the number of moles of compounds
def moles_C2H5Cl : ℝ := 15
def moles_O2 : ℝ := 12

-- Calculating the molecular weights of compounds
def mol_weight_C2H5Cl : ℝ := (2 * mol_weight_C) + (5 * mol_weight_H) + mol_weight_Cl
def mol_weight_O2 : ℝ := 2 * mol_weight_O

-- Calculating the total weight of each compound
def total_weight_C2H5Cl : ℝ := moles_C2H5Cl * mol_weight_C2H5Cl
def total_weight_O2 : ℝ := moles_O2 * mol_weight_O2

-- Defining the final total weight
def total_weight : ℝ := total_weight_C2H5Cl + total_weight_O2

-- Statement to prove
theorem total_molecular_weight_correct :
  total_weight = 1351.8 := by
  sorry

end total_molecular_weight_correct_l65_65002


namespace percentage_less_than_l65_65925

theorem percentage_less_than (x y : ℝ) (h : y = 1.80 * x) : (x / y) * 100 = 100 - 44.44 :=
by
  sorry

end percentage_less_than_l65_65925


namespace Matt_overall_profit_l65_65249

def initialValue : ℕ := 8 * 6

def valueGivenAwayTrade1 : ℕ := 2 * 6
def valueReceivedTrade1 : ℕ := 3 * 2 + 9

def valueGivenAwayTrade2 : ℕ := 2 + 6
def valueReceivedTrade2 : ℕ := 2 * 5 + 8

def valueGivenAwayTrade3 : ℕ := 5 + 9
def valueReceivedTrade3 : ℕ := 3 * 3 + 10 + 1

def valueGivenAwayTrade4 : ℕ := 2 * 3 + 8
def valueReceivedTrade4 : ℕ := 2 * 7 + 4

def overallProfit : ℕ :=
  (valueReceivedTrade1 - valueGivenAwayTrade1) +
  (valueReceivedTrade2 - valueGivenAwayTrade2) +
  (valueReceivedTrade3 - valueGivenAwayTrade3) +
  (valueReceivedTrade4 - valueGivenAwayTrade4)

theorem Matt_overall_profit : overallProfit = 23 :=
by
  unfold overallProfit valueReceivedTrade1 valueGivenAwayTrade1 valueReceivedTrade2 valueGivenAwayTrade2 valueReceivedTrade3 valueGivenAwayTrade3 valueReceivedTrade4 valueGivenAwayTrade4
  linarith

end Matt_overall_profit_l65_65249


namespace reflect_across_y_axis_l65_65931

theorem reflect_across_y_axis (x y : ℝ) :
  (x, y) = (1, 2) → (-x, y) = (-1, 2) :=
by
  intro h
  cases h
  sorry

end reflect_across_y_axis_l65_65931


namespace number_of_neutrons_eq_l65_65685

variable (A n x : ℕ)

/-- The number of neutrons N in the nucleus of an atom R, given that:
  1. A is the atomic mass number of R.
  2. The ion RO3^(n-) contains x outer electrons. -/
theorem number_of_neutrons_eq (N : ℕ) (h : A - N + 24 + n = x) : N = A + n + 24 - x :=
by sorry

end number_of_neutrons_eq_l65_65685


namespace find_varphi_l65_65044

theorem find_varphi (ϕ : ℝ) (h0 : 0 < ϕ ∧ ϕ < π / 2) :
  (∀ x₁ x₂, |(2 * Real.cos (2 * x₁)) - (2 * Real.cos (2 * x₂ - 2 * ϕ))| = 4 → 
    ∃ (x₁ x₂ : ℝ), |x₁ - x₂| = π / 6 
  ) → ϕ = π / 3 :=
by
  sorry

end find_varphi_l65_65044


namespace sum_of_two_primes_unique_l65_65236

theorem sum_of_two_primes_unique (n : ℕ) (h : n = 10003) :
  (∃ p1 p2 : ℕ, Prime p1 ∧ Prime p2 ∧ n = p1 + p2 ∧ p1 = 2 ∧ Prime (n - 2)) ↔ 
  (p1 = 2 ∧ p2 = 10001 ∧ Prime 10001) := 
by
  sorry

end sum_of_two_primes_unique_l65_65236


namespace solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l65_65413

-- Define the conditions
variables (a b c x y z: ℝ) 

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) : Prop :=
  (a * y + b * x = c) ∧
  (c * x + a * z = b) ∧
  (b * z + c * y = a)

-- Define the general solution
def solution (a b c x y z : ℝ) : Prop :=
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧
  z = (a^2 + b^2 - c^2) / (2 * a * b)

-- Define the proof problem statement
theorem solve_system_of_equations_general (a b c x y z : ℝ) (h : system_of_equations a b c x y z) 
      (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : solution a b c x y z :=
  sorry

-- Special cases
theorem solve_system_of_equations_zero_case_1 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_2 (a b c x y z : ℝ) (h1 : a = 0) (h2 : b = 0) (h3: c ≠ 0) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_3 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : x = c / b ∧ 
      (c * x = b) :=
  sorry

-- Following special cases more concisely
theorem solve_system_of_equations_special_cases (a b c x y z : ℝ) 
      (h : system_of_equations a b c x y z) (h1: a = 0 ∨ b = 0 ∨ c = 0): 
      (∃ k : ℝ, x = k ∧ y = -k ∧ z = k)  
    ∨ (∃ k : ℝ, x = k ∧ y = k ∧ z = -k)
    ∨ (∃ k : ℝ, x = -k ∧ y = k ∧ z = k) :=
  sorry

end solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l65_65413


namespace find_U_l65_65271

-- Declare the variables and conditions
def digits : Set ℤ := {1, 2, 3, 4, 5, 6}

theorem find_U (P Q R S T U : ℤ) :
  -- Condition: Digits are distinct and each is in {1, 2, 3, 4, 5, 6}
  (P ∈ digits) ∧ (Q ∈ digits) ∧ (R ∈ digits) ∧ (S ∈ digits) ∧ (T ∈ digits) ∧ (U ∈ digits) ∧
  (P ≠ Q) ∧ (P ≠ R) ∧ (P ≠ S) ∧ (P ≠ T) ∧ (P ≠ U) ∧
  (Q ≠ R) ∧ (Q ≠ S) ∧ (Q ≠ T) ∧ (Q ≠ U) ∧
  (R ≠ S) ∧ (R ≠ T) ∧ (R ≠ U) ∧ (S ≠ T) ∧ (S ≠ U) ∧ (T ≠ U) ∧
  -- Condition: The three-digit number PQR is divisible by 9
  (100 * P + 10 * Q + R) % 9 = 0 ∧
  -- Condition: The three-digit number QRS is divisible by 4
  (10 * Q + R) % 4 = 0 ∧
  -- Condition: The three-digit number RST is divisible by 3
  (10 * R + S) % 3 = 0 ∧
  -- Condition: The sum of the digits is divisible by 5
  (P + Q + R + S + T + U) % 5 = 0
  -- Conclusion: U = 4
  → U = 4 :=
by sorry

end find_U_l65_65271


namespace peanuts_added_l65_65204

theorem peanuts_added (a b x : ℕ) (h1 : a = 4) (h2 : b = 6) (h3 : a + x = b) : x = 2 :=
by
  sorry

end peanuts_added_l65_65204


namespace series_2023_power_of_3_squared_20_equals_653_l65_65383

def series (A : ℕ → ℕ) : Prop :=
  A 0 = 1 ∧ 
  ∀ n > 0, 
  A n = A (n / 2023) + A (n / 2023^2) + A (n / 2023^3)

theorem series_2023_power_of_3_squared_20_equals_653 (A : ℕ → ℕ) (h : series A) : A (2023 ^ (3^2) + 20) = 653 :=
by
  -- placeholder for proof
  sorry

end series_2023_power_of_3_squared_20_equals_653_l65_65383


namespace value_of_a_plus_b_l65_65864

theorem value_of_a_plus_b (a b : ℝ) (h : (2 * a + 2 * b - 1) * (2 * a + 2 * b + 1) = 99) :
  a + b = 5 ∨ a + b = -5 :=
sorry

end value_of_a_plus_b_l65_65864


namespace sin_pow_cos_pow_eq_l65_65962

theorem sin_pow_cos_pow_eq (x : ℝ) (h : Real.sin x ^ 10 + Real.cos x ^ 10 = 11 / 36) : 
  Real.sin x ^ 14 + Real.cos x ^ 14 = 41 / 216 := by
  sorry

end sin_pow_cos_pow_eq_l65_65962


namespace largest_n_l65_65011

noncomputable def a (n : ℕ) (x : ℤ) : ℤ := 2 + (n - 1) * x
noncomputable def b (n : ℕ) (y : ℤ) : ℤ := 3 + (n - 1) * y

theorem largest_n {n : ℕ} (x y : ℤ) :
  a 1 x = 2 ∧ b 1 y = 3 ∧ 3 * a 2 x < 2 * b 2 y ∧ a n x * b n y = 4032 →
  n = 367 :=
sorry

end largest_n_l65_65011


namespace sum_of_first_six_terms_l65_65382

theorem sum_of_first_six_terms 
  {S : ℕ → ℝ} 
  (h_arith_seq : ∀ n, S n = n * (-2) + (n * (n - 1) * 3 ))
  (S_2_eq_2 : S 2 = 2)
  (S_4_eq_10 : S 4 = 10) : S 6 = 18 := 
  sorry

end sum_of_first_six_terms_l65_65382


namespace intersection_M_N_l65_65790

-- Define set M
def M : Set ℝ := {y | ∃ x > 0, y = 2^x}

-- Define set N
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- Prove the intersection of M and N equals (1, 2)
theorem intersection_M_N :
  ∀ x, x ∈ M ∩ N ↔ 1 < x ∧ x < 2 :=
by
  -- Skipping the proof here
  sorry

end intersection_M_N_l65_65790


namespace selection_methods_l65_65292

theorem selection_methods (students : ℕ) (boys : ℕ) (girls : ℕ) (selected : ℕ) (h1 : students = 8) (h2 : boys = 6) (h3 : girls = 2) (h4 : selected = 4) : 
  ∃ methods, methods = 40 :=
by
  have h5 : students = boys + girls := by linarith
  sorry

end selection_methods_l65_65292


namespace min_sum_of_factors_l65_65920

theorem min_sum_of_factors (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z = 1806) :
  x + y + z ≥ 72 := 
sorry

end min_sum_of_factors_l65_65920


namespace triangle_angle_D_l65_65568

theorem triangle_angle_D (F E D : ℝ) (hF : F = 15) (hE : E = 3 * F) (h_triangle : D + E + F = 180) : D = 120 := by
  sorry

end triangle_angle_D_l65_65568


namespace geometric_sequence_11th_term_l65_65022

theorem geometric_sequence_11th_term (a r : ℝ) (h₁ : a * r ^ 4 = 8) (h₂ : a * r ^ 7 = 64) : 
  a * r ^ 10 = 512 :=
by sorry

end geometric_sequence_11th_term_l65_65022


namespace shadow_length_of_flagpole_is_correct_l65_65001

noncomputable def length_of_shadow_flagpole : ℕ :=
  let h_flagpole : ℕ := 18
  let shadow_building : ℕ := 60
  let h_building : ℕ := 24
  let similar_conditions : Prop := true
  45

theorem shadow_length_of_flagpole_is_correct :
  length_of_shadow_flagpole = 45 := by
  sorry

end shadow_length_of_flagpole_is_correct_l65_65001


namespace snow_probability_l65_65761

theorem snow_probability :
  let p1_snow := 1 / 3
  let p2_snow := 1 / 4
  let p1_prob_no_snow := 1 - p1_snow
  let p2_prob_no_snow := 1 - p2_snow
  let p_no_snow_first_three := p1_prob_no_snow ^ 3
  let p_no_snow_next_four := p2_prob_no_snow ^ 4
  let p_no_snow_week := p_no_snow_first_three * p_no_snow_next_four
  1 - p_no_snow_week = 29 / 32 :=
by
  let p1_snow := 1 / 3
  let p2_snow := 1 / 4
  let p1_prob_no_snow := 1 - p1_snow
  let p2_prob_no_snow := 1 - p2_snow
  let p_no_snow_first_three := p1_prob_no_snow ^ 3
  let p_no_snow_next_four := p2_prob_no_snow ^ 4
  let p_no_snow_week := p_no_snow_first_three * p_no_snow_next_four
  have p_no_snow_week_eq : p_no_snow_week = 3 / 32 := sorry
  have p_snow_at_least_once_week : 1 - p_no_snow_week = 29 / 32 := sorry
  exact p_snow_at_least_once_week

end snow_probability_l65_65761


namespace range_of_a_l65_65869

noncomputable def f (x a : ℝ) : ℝ := x^2 * Real.exp x - a

theorem range_of_a 
 (h : ∃ a, (∀ x₀ x₁ x₂, x₀ ≠ x₁ ∧ x₀ ≠ x₂ ∧ x₁ ≠ x₂ ∧ f x₀ a = 0 ∧ f x₁ a = 0 ∧ f x₂ a = 0)) :
  ∃ a, 0 < a ∧ a < 4 / Real.exp 2 :=
by
  sorry

end range_of_a_l65_65869


namespace uncle_zhang_age_l65_65867

theorem uncle_zhang_age (z l : ℕ) (h1 : z + l = 56) (h2 : z = l - (l / 2)) : z = 24 :=
by sorry

end uncle_zhang_age_l65_65867


namespace value_of_y_l65_65826

-- Problem: Prove that given the conditions \( x - y = 8 \) and \( x + y = 16 \),
-- the value of \( y \) is 4.
theorem value_of_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := 
sorry

end value_of_y_l65_65826


namespace divides_lcm_condition_l65_65477

theorem divides_lcm_condition (x y : ℕ) (h₀ : 1 < x) (h₁ : 1 < y)
  (h₂ : Nat.lcm (x+2) (y+2) - Nat.lcm (x+1) (y+1) = Nat.lcm (x+1) (y+1) - Nat.lcm x y) :
  x ∣ y ∨ y ∣ x := 
sorry

end divides_lcm_condition_l65_65477


namespace min_value_sin_sq_l65_65286

theorem min_value_sin_sq (A B : ℝ) (h : A + B = π / 2) :
  4 / (Real.sin A)^2 + 9 / (Real.sin B)^2 ≥ 25 :=
sorry

end min_value_sin_sq_l65_65286


namespace find_x_l65_65227

def operation (a b : Int) : Int := 2 * a + b

theorem find_x :
  ∃ x : Int, operation 3 (operation 4 x) = -1 :=
  sorry

end find_x_l65_65227


namespace building_time_l65_65705

theorem building_time (b p : ℕ) 
  (h1 : b = 3 * p - 5) 
  (h2 : b + p = 67) 
  : b = 49 := 
by 
  sorry

end building_time_l65_65705


namespace no_common_multiples_of_3_l65_65348

-- Define the sets X and Y
def SetX : Set ℤ := {n | 1 ≤ n ∧ n ≤ 24 ∧ n % 2 = 1}
def SetY : Set ℤ := {n | 0 ≤ n ∧ n ≤ 40 ∧ n % 2 = 0}

-- Define the condition for being a multiple of 3
def isMultipleOf3 (n : ℤ) : Prop := n % 3 = 0

-- Define the intersection of SetX and SetY that are multiples of 3
def intersectionMultipleOf3 : Set ℤ := {n | n ∈ SetX ∧ n ∈ SetY ∧ isMultipleOf3 n}

-- Prove that the set is empty
theorem no_common_multiples_of_3 : intersectionMultipleOf3 = ∅ := by
  sorry

end no_common_multiples_of_3_l65_65348


namespace cricketer_average_increase_l65_65598

theorem cricketer_average_increase (A : ℝ) (H1 : 18 * A + 98 = 19 * 26) :
  26 - A = 4 :=
by
  sorry

end cricketer_average_increase_l65_65598


namespace Marilyn_has_40_bananas_l65_65241

-- Definitions of the conditions
def boxes : ℕ := 8
def bananas_per_box : ℕ := 5

-- Statement of the proof problem
theorem Marilyn_has_40_bananas : (boxes * bananas_per_box) = 40 := by
  sorry

end Marilyn_has_40_bananas_l65_65241


namespace arithmetic_seq_sum_l65_65328

theorem arithmetic_seq_sum (a : ℕ → ℤ) (h_arith_seq : ∀ m n p q : ℕ, m + n = p + q → a m + a n = a p + a q) (h_a5 : a 5 = 15) : a 2 + a 4 + a 6 + a 8 = 60 := 
by
  sorry

end arithmetic_seq_sum_l65_65328


namespace point_in_second_or_third_quadrant_l65_65515

theorem point_in_second_or_third_quadrant (k b : ℝ) (h₁ : k < 0) (h₂ : b ≠ 0) : 
  (k < 0 ∧ b > 0) ∨ (k < 0 ∧ b < 0) :=
by
  sorry

end point_in_second_or_third_quadrant_l65_65515


namespace range_of_m_l65_65878

theorem range_of_m (m : ℝ) :
  let M := {x : ℝ | x ≤ m}
  let P := {x : ℝ | x ≥ -1}
  (M ∩ P = ∅) → m < -1 :=
by
  sorry

end range_of_m_l65_65878


namespace mike_daily_work_hours_l65_65202

def total_hours_worked : ℕ := 15
def number_of_days_worked : ℕ := 5

theorem mike_daily_work_hours : total_hours_worked / number_of_days_worked = 3 :=
by
  sorry

end mike_daily_work_hours_l65_65202


namespace number_of_5_dollar_bills_l65_65816

theorem number_of_5_dollar_bills (x y : ℝ) (h1 : x + y = 54) (h2 : 5 * x + 20 * y = 780) : x = 20 :=
sorry

end number_of_5_dollar_bills_l65_65816


namespace smallest_n_l65_65592

theorem smallest_n (n : ℕ) (h1 : n > 2016) (h2 : (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0) : n = 2020 :=
sorry

end smallest_n_l65_65592


namespace todd_has_40_left_after_paying_back_l65_65999

def todd_snowcone_problem : Prop :=
  let borrowed := 100
  let repay := 110
  let cost_ingredients := 75
  let snowcones_sold := 200
  let price_per_snowcone := 0.75
  let total_earnings := snowcones_sold * price_per_snowcone
  let remaining_money := total_earnings - repay
  remaining_money = 40

theorem todd_has_40_left_after_paying_back : todd_snowcone_problem :=
by
  -- Add proof here if needed
  sorry

end todd_has_40_left_after_paying_back_l65_65999


namespace relationship_between_abc_l65_65701

noncomputable def a : ℝ := (0.6 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (1.5 : ℝ)
noncomputable def c : ℝ := (1.5 : ℝ) ^ (0.6 : ℝ)

theorem relationship_between_abc : c > a ∧ a > b := sorry

end relationship_between_abc_l65_65701


namespace value_of_f_ln3_l65_65454

def f : ℝ → ℝ := sorry

theorem value_of_f_ln3 (f_symm : ∀ x : ℝ, f (x + 1) = f (-x + 1))
  (f_exp : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = Real.exp (-x)) :
  f (Real.log 3) = 3 * Real.exp (-2) :=
by
  sorry

end value_of_f_ln3_l65_65454


namespace third_number_in_first_set_is_42_l65_65103

theorem third_number_in_first_set_is_42 (x y : ℕ) :
  (28 + x + y + 78 + 104) / 5 = 90 →
  (128 + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 42 :=
by { sorry }

end third_number_in_first_set_is_42_l65_65103


namespace inequality_abc_l65_65031

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
    a^2 + b^2 + c^2 + 3 ≥ (1 / a) + (1 / b) + (1 / c) + a + b + c :=
sorry

end inequality_abc_l65_65031


namespace manny_marbles_l65_65195

theorem manny_marbles (total_marbles : ℕ) (ratio_m : ℕ) (ratio_n : ℕ) (manny_gives : ℕ) 
  (h_total : total_marbles = 36) (h_ratio_m : ratio_m = 4) (h_ratio_n : ratio_n = 5) (h_manny_gives : manny_gives = 2) : 
  (total_marbles * ratio_n / (ratio_m + ratio_n)) - manny_gives = 18 :=
by
  sorry

end manny_marbles_l65_65195


namespace units_digit_k_cube_plus_2_to_k_plus_1_mod_10_l65_65837

def k : ℕ := 2012 ^ 2 + 2 ^ 2012

theorem units_digit_k_cube_plus_2_to_k_plus_1_mod_10 : (k ^ 3 + 2 ^ (k + 1)) % 10 = 2 := 
by sorry

end units_digit_k_cube_plus_2_to_k_plus_1_mod_10_l65_65837


namespace bag_of_food_costs_two_dollars_l65_65344

theorem bag_of_food_costs_two_dollars
  (cost_puppy : ℕ)
  (total_cost : ℕ)
  (daily_food : ℚ)
  (bag_food_quantity : ℚ)
  (weeks : ℕ)
  (h1 : cost_puppy = 10)
  (h2 : total_cost = 14)
  (h3 : daily_food = 1/3)
  (h4 : bag_food_quantity = 3.5)
  (h5 : weeks = 3) :
  (total_cost - cost_puppy) / (21 * daily_food / bag_food_quantity) = 2 := 
  by sorry

end bag_of_food_costs_two_dollars_l65_65344


namespace sum_series_l65_65034

theorem sum_series :
  ∑' n : ℕ, (2^n / (3^(2^n) + 1)) = 1 / 2 :=
sorry

end sum_series_l65_65034


namespace find_sticker_price_l65_65162

variable (x : ℝ)

def price_at_store_A (x : ℝ) : ℝ := 0.80 * x - 120
def price_at_store_B (x : ℝ) : ℝ := 0.70 * x
def savings (x : ℝ) : ℝ := price_at_store_B x - price_at_store_A x

theorem find_sticker_price (h : savings x = 30) : x = 900 :=
by
  -- proof can be filled in here
  sorry

end find_sticker_price_l65_65162


namespace polygon_diagonals_eq_sum_sides_and_right_angles_l65_65795

-- Define the number of sides of the polygon
variables (n : ℕ)

-- Definition of the number of diagonals in a convex n-sided polygon
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Definition of the sum of interior angles of an n-sided polygon
def sum_interior_angles (n : ℕ) : ℕ := (n - 2) * 180

-- Definition of equivalent right angles for interior angles
def num_right_angles (n : ℕ) : ℕ := 2 * (n - 2)

-- The proof statement: prove that the equation holds for n
theorem polygon_diagonals_eq_sum_sides_and_right_angles (h : 3 ≤ n) :
  num_diagonals n = n + num_right_angles n :=
sorry

end polygon_diagonals_eq_sum_sides_and_right_angles_l65_65795


namespace members_even_and_divisible_l65_65465

structure ClubMember (α : Type) := 
  (friend : α) 
  (enemy : α)

def is_even (n : Nat) : Prop :=
  n % 2 = 0

def can_be_divided_into_two_subclubs (members : List (ClubMember Nat)) : Prop :=
sorry -- Definition of dividing into two subclubs here

theorem members_even_and_divisible (members : List (ClubMember Nat)) :
  is_even members.length ∧ can_be_divided_into_two_subclubs members :=
sorry

end members_even_and_divisible_l65_65465


namespace limit_of_p_n_is_tenth_l65_65523

noncomputable def p_n (n : ℕ) : ℝ := sorry -- Definition of p_n needs precise formulation.

def tends_to_tenth_as_n_infty (p : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, abs (p n - 1/10) < ε

theorem limit_of_p_n_is_tenth : tends_to_tenth_as_n_infty p_n := sorry

end limit_of_p_n_is_tenth_l65_65523


namespace relationship_between_a_and_b_l65_65108

theorem relationship_between_a_and_b 
  (x a b : ℝ)
  (hx : 0 < x)
  (ha : 0 < a)
  (hb : 0 < b)
  (hax : a^x < b^x) 
  (hbx : b^x < 1) : 
  a < b ∧ b < 1 := 
sorry

end relationship_between_a_and_b_l65_65108


namespace b_is_perfect_square_l65_65468

noncomputable def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

theorem b_is_perfect_square (a b : ℕ)
  (h_positive : 0 < a) (h_positive_b : 0 < b)
  (h_gcd_lcm_multiple : (Nat.gcd a b + Nat.lcm a b) % (a + 1) = 0)
  (h_le : b ≤ a) : is_perfect_square b :=
sorry

end b_is_perfect_square_l65_65468


namespace equation_of_parallel_line_through_point_l65_65670

theorem equation_of_parallel_line_through_point :
  ∃ m b, (∀ x y, y = m * x + b → (∃ k, k = 3 ^ 2 - 9 * 2 + 1)) ∧ 
         (∀ x y, y = 3 * x + b → y - 0 = 3 * (x - (-2))) :=
sorry

end equation_of_parallel_line_through_point_l65_65670


namespace large_cross_area_is_60_cm_squared_l65_65098

noncomputable def small_square_area (s : ℝ) := s * s
noncomputable def large_square_area (s : ℝ) := 4 * small_square_area s
noncomputable def small_cross_area (s : ℝ) := 5 * small_square_area s
noncomputable def large_cross_area (s : ℝ) := 5 * large_square_area s
noncomputable def remaining_area (s : ℝ) := large_cross_area s - small_cross_area s

theorem large_cross_area_is_60_cm_squared :
  ∃ (s : ℝ), remaining_area s = 45 → large_cross_area s = 60 :=
by
  sorry

end large_cross_area_is_60_cm_squared_l65_65098


namespace repeating_decimal_subtraction_simplified_l65_65731

theorem repeating_decimal_subtraction_simplified :
  let x := (567 / 999 : ℚ)
  let y := (234 / 999 : ℚ)
  let z := (891 / 999 : ℚ)
  x - y - z = -186 / 333 :=
by
  sorry

end repeating_decimal_subtraction_simplified_l65_65731


namespace william_total_tickets_l65_65429

def initial_tickets : ℕ := 15
def additional_tickets : ℕ := 3
def total_tickets : ℕ := initial_tickets + additional_tickets

theorem william_total_tickets :
  total_tickets = 18 := by
  -- proof goes here
  sorry

end william_total_tickets_l65_65429


namespace drawings_in_five_pages_l65_65964

theorem drawings_in_five_pages :
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  a₁ + a₂ + a₃ + a₄ + a₅ = 155 :=
by
  let a₁ := 5
  let a₂ := 2 * a₁
  let a₃ := 2 * a₂
  let a₄ := 2 * a₃
  let a₅ := 2 * a₄
  sorry

end drawings_in_five_pages_l65_65964


namespace is_divisible_by_N2_l65_65051

def are_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

noncomputable def eulers_totient (n : ℕ) : ℕ :=
  Nat.totient n

theorem is_divisible_by_N2 (N1 N2 : ℕ) (h_coprime : are_coprime N1 N2) 
  (k := eulers_totient N2) : 
  (N1 ^ k - 1) % N2 = 0 :=
by
  sorry

end is_divisible_by_N2_l65_65051


namespace pillows_from_feathers_l65_65680

def feathers_per_pound : ℕ := 300
def feathers_total : ℕ := 3600
def pounds_per_pillow : ℕ := 2

theorem pillows_from_feathers :
  (feathers_total / feathers_per_pound / pounds_per_pillow) = 6 :=
by
  sorry

end pillows_from_feathers_l65_65680


namespace total_volume_needed_l65_65821

def box_length : ℕ := 20
def box_width : ℕ := 20
def box_height : ℕ := 12
def box_cost : ℕ := 50 -- in cents to avoid using floats
def total_spent : ℕ := 20000 -- $200 in cents

def volume_of_box : ℕ := box_length * box_width * box_height
def number_of_boxes : ℕ := total_spent / box_cost

theorem total_volume_needed : number_of_boxes * volume_of_box = 1920000 := by
  sorry

end total_volume_needed_l65_65821


namespace greater_than_neg4_1_l65_65996

theorem greater_than_neg4_1 (k : ℤ) (h1 : k = -4) : k > (-4.1 : ℝ) :=
by sorry

end greater_than_neg4_1_l65_65996


namespace f_is_zero_l65_65516

noncomputable def f (x : ℝ) : ℝ := sorry

theorem f_is_zero 
  (H1 : ∀ a b : ℝ, f (a * b) = a * f b + b * f a)
  (H2 : ∀ x : ℝ, |f x| ≤ 1) : ∀ x : ℝ, f x = 0 := 
sorry

end f_is_zero_l65_65516


namespace flavored_drink_ratio_l65_65009

theorem flavored_drink_ratio :
  ∃ (F C W: ℚ), F / C = 1 / 7.5 ∧ F / W = 1 / 56.25 ∧ C/W = 6/90 ∧ F / C / 3 = ((F / W) * 2)
:= sorry

end flavored_drink_ratio_l65_65009


namespace evaluate_expression_l65_65244

theorem evaluate_expression : 
  -3 * 5 - (-4 * -2) + (-15 * -3) / 3 = -8 :=
by
  sorry

end evaluate_expression_l65_65244


namespace fixed_point_l65_65824

noncomputable def function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (x : ℝ) : ℝ :=
  a ^ (x - 1) + 1

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  function a h_pos h_ne_one 1 = 2 :=
by
  sorry

end fixed_point_l65_65824


namespace gamma_suff_not_nec_for_alpha_l65_65953

variable {α β γ : Prop}

theorem gamma_suff_not_nec_for_alpha
  (h1 : β → α)
  (h2 : γ ↔ β) :
  (γ → α) ∧ (¬(α → γ)) :=
by {
  sorry
}

end gamma_suff_not_nec_for_alpha_l65_65953


namespace John_height_in_feet_after_growth_spurt_l65_65172

def John_initial_height : ℕ := 66
def growth_rate_per_month : ℕ := 2
def number_of_months : ℕ := 3
def inches_per_foot : ℕ := 12

theorem John_height_in_feet_after_growth_spurt :
  (John_initial_height + growth_rate_per_month * number_of_months) / inches_per_foot = 6 := by
  sorry

end John_height_in_feet_after_growth_spurt_l65_65172


namespace arithmetic_sequence_sum_ratio_l65_65064

theorem arithmetic_sequence_sum_ratio
  (S : ℕ → ℚ)
  (a : ℕ → ℚ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : a 5 / a 3 = 7 / 3) :
  S 5 / S 3 = 5 := 
by
  sorry

end arithmetic_sequence_sum_ratio_l65_65064


namespace prob_statement_l65_65975

open Set

-- Definitions from the conditions
def A : Set ℝ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | x^2 + 2 * x < 0}

-- Proposition to be proved
theorem prob_statement : A ∩ (Bᶜ) = {-2, 0, 1, 2} :=
by
  sorry

end prob_statement_l65_65975


namespace tangent_points_l65_65349

theorem tangent_points (x y : ℝ) (h : y = x^3 - 3 * x) (slope_zero : 3 * x^2 - 3 = 0) :
  (x = -1 ∧ y = 2) ∨ (x = 1 ∧ y = -2) :=
sorry

end tangent_points_l65_65349


namespace units_digit_of_expression_l65_65091

theorem units_digit_of_expression :
  (6 * 16 * 1986 - 6 ^ 4) % 10 = 0 := 
sorry

end units_digit_of_expression_l65_65091


namespace calculate_difference_of_squares_l65_65520

theorem calculate_difference_of_squares :
  (153^2 - 147^2) = 1800 :=
by
  sorry

end calculate_difference_of_squares_l65_65520


namespace part1_part2_l65_65933

-- Definitions corresponding to the conditions
def angle_A := 35
def angle_B1 := 40
def three_times_angle_triangle (A B C : ℕ) : Prop :=
  A + B + C = 180 ∧ (A = 3 * B ∨ B = 3 * A ∨ C = 3 * A ∨ A = 3 * C ∨ B = 3 * C ∨ C = 3 * B)

-- Part 1: Checking if triangle ABC is a "three times angle triangle".
theorem part1 : three_times_angle_triangle angle_A angle_B1 (180 - angle_A - angle_B1) :=
  sorry

-- Definitions corresponding to the new conditions
def angle_B2 := 60

-- Part 2: Finding the smallest interior angle in triangle ABC.
theorem part2 (angle_A angle_C : ℕ) :
  three_times_angle_triangle angle_A angle_B2 angle_C → (angle_A = 20 ∨ angle_A = 30 ∨ angle_C = 20 ∨ angle_C = 30) :=
  sorry

end part1_part2_l65_65933


namespace k_value_if_perfect_square_l65_65643

theorem k_value_if_perfect_square (k : ℤ) (x : ℝ) (h : ∃ (a : ℝ), x^2 + k * x + 25 = a^2) : k = 10 ∨ k = -10 := by
  sorry

end k_value_if_perfect_square_l65_65643


namespace largest_number_l65_65079

theorem largest_number (P Q R S T : ℕ) 
  (hP_digits_prime : ∃ p1 p2, P = 10 * p1 + p2 ∧ Prime P ∧ Prime (p1 + p2))
  (hQ_multiple_of_5 : Q % 5 = 0)
  (hR_odd_non_prime : Odd R ∧ ¬ Prime R)
  (hS_prime_square : ∃ p, Prime p ∧ S = p * p)
  (hT_mean_prime : T = (P + Q) / 2 ∧ Prime T)
  (hP_range : 10 ≤ P ∧ P ≤ 99)
  (hQ_range : 2 ≤ Q ∧ Q ≤ 19)
  (hR_range : 2 ≤ R ∧ R ≤ 19)
  (hS_range : 2 ≤ S ∧ S ≤ 19)
  (hT_range : 2 ≤ T ∧ T ≤ 19) :
  max P (max Q (max R (max S T))) = Q := 
by 
  sorry

end largest_number_l65_65079


namespace probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l65_65815

def total_balls := 20
def red_balls := 10
def yellow_balls := 6
def white_balls := 4
def initial_white_balls_probability := (white_balls : ℚ) / total_balls
def initial_yellow_or_red_balls_probability := (yellow_balls + red_balls : ℚ) / total_balls

def removed_red_balls := 2
def removed_white_balls := 2
def remaining_balls := total_balls - (removed_red_balls + removed_white_balls)
def remaining_white_balls := white_balls - removed_white_balls
def remaining_white_balls_probability := (remaining_white_balls : ℚ) / remaining_balls

theorem probability_white_ball_initial : initial_white_balls_probability = 1 / 5 := by sorry
theorem probability_yellow_or_red_ball_initial : initial_yellow_or_red_balls_probability = 4 / 5 := by sorry
theorem probability_white_ball_after_removal : remaining_white_balls_probability = 1 / 8 := by sorry

end probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l65_65815


namespace set_representation_listing_method_l65_65547

def is_in_set (a : ℤ) : Prop := 0 < 2 * a - 1 ∧ 2 * a - 1 ≤ 5

def M : Set ℤ := {a | is_in_set a}

theorem set_representation_listing_method :
  M = {1, 2, 3} :=
sorry

end set_representation_listing_method_l65_65547


namespace olivia_grocery_cost_l65_65336

theorem olivia_grocery_cost :
  let cost_bananas := 12
  let cost_bread := 9
  let cost_milk := 7
  let cost_apples := 14
  cost_bananas + cost_bread + cost_milk + cost_apples = 42 :=
by
  rfl

end olivia_grocery_cost_l65_65336
