import Mathlib

namespace min_max_transformation_a_min_max_transformation_b_l2253_225390

theorem min_max_transformation_a {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≥ a) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x^3 - 1) / (x^6 + 1) → z ≤ b) :=
sorry

theorem min_max_transformation_b {a b : ℝ} (hmin : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≥ a))
  (hmax : ∀ x : ℝ, ∀ z : ℝ, (z = (x - 1) / (x^2 + 1)) → (z ≤ b)) :
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≥ -b) ∧
  (∀ x : ℝ, ∀ z : ℝ, z = (x + 1) / (x^2 + 1) → z ≤ -a) :=
sorry

end min_max_transformation_a_min_max_transformation_b_l2253_225390


namespace system_solution_unique_l2253_225340

theorem system_solution_unique (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (eq1 : x ^ 3 + 2 * y ^ 2 + 1 / (4 * z) = 1)
  (eq2 : y ^ 3 + 2 * z ^ 2 + 1 / (4 * x) = 1)
  (eq3 : z ^ 3 + 2 * x ^ 2 + 1 / (4 * y) = 1) :
  (x, y, z) = ( ( (-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2), ((-1 + Real.sqrt 3) / 2) ) := 
by
  sorry

end system_solution_unique_l2253_225340


namespace tank_empty_time_l2253_225324

theorem tank_empty_time 
  (time_to_empty_leak : ℝ) 
  (inlet_rate_per_minute : ℝ) 
  (tank_volume : ℝ) 
  (net_time_to_empty : ℝ) : 
  time_to_empty_leak = 7 → 
  inlet_rate_per_minute = 6 → 
  tank_volume = 6048.000000000001 → 
  net_time_to_empty = 12 :=
by
  intros h1 h2 h3
  sorry

end tank_empty_time_l2253_225324


namespace intersection_M_N_l2253_225367

-- Define the sets based on the given conditions
def M : Set ℝ := {x | x + 2 < 0}
def N : Set ℝ := {x | x + 1 < 0}

-- State the theorem to prove the intersection
theorem intersection_M_N :
  M ∩ N = {x | x < -2} := by
sorry

end intersection_M_N_l2253_225367


namespace sum_of_smallest_and_largest_eq_2y_l2253_225382

variable (a n y : ℤ) (hn_even : Even n) (hy : y = a + n - 1)

theorem sum_of_smallest_and_largest_eq_2y : a + (a + 2 * (n - 1)) = 2 * y := 
by
  sorry

end sum_of_smallest_and_largest_eq_2y_l2253_225382


namespace quadratic_roots_l2253_225339

theorem quadratic_roots : ∀ x : ℝ, x * (x - 2) = 2 - x ↔ (x = 2 ∨ x = -1) := by
  intros
  sorry

end quadratic_roots_l2253_225339


namespace solve_prime_equation_l2253_225374

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem solve_prime_equation (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h : p ^ 2 - 6 * p * q + q ^ 2 + 3 * q - 1 = 0) :
  (p = 17 ∧ q = 3) :=
sorry

end solve_prime_equation_l2253_225374


namespace secret_reaches_2186_students_on_seventh_day_l2253_225391

/-- 
Alice tells a secret to three friends on Sunday. The next day, each of those friends tells the secret to three new friends.
Each time a person hears the secret, they tell three other new friends the following day.
On what day will 2186 students know the secret?
-/
theorem secret_reaches_2186_students_on_seventh_day :
  ∃ (n : ℕ), 1 + 3 * ((3^n - 1)/2) = 2186 ∧ n = 7 :=
by
  sorry

end secret_reaches_2186_students_on_seventh_day_l2253_225391


namespace parallel_vectors_eq_l2253_225308

theorem parallel_vectors_eq (x : ℝ) :
  let a := (x, 1)
  let b := (2, 4)
  (a.1 / b.1 = a.2 / b.2) → x = 1 / 2 :=
by
  intros h
  sorry

end parallel_vectors_eq_l2253_225308


namespace correct_student_mark_l2253_225375

theorem correct_student_mark :
  ∀ (total_marks total_correct_marks incorrect_mark correct_average students : ℝ)
  (h1 : total_marks = students * 100)
  (h2 : incorrect_mark = 60)
  (h3 : correct_average = 95)
  (h4 : total_correct_marks = students * correct_average),
  total_marks - incorrect_mark + (total_correct_marks - (total_marks - incorrect_mark)) = 10 :=
by
  intros total_marks total_correct_marks incorrect_mark correct_average students h1 h2 h3 h4
  sorry

end correct_student_mark_l2253_225375


namespace sum_of_remainders_eq_24_l2253_225335

theorem sum_of_remainders_eq_24 (a b c : ℕ) 
  (h1 : a % 30 = 13) (h2 : b % 30 = 19) (h3 : c % 30 = 22) :
  (a + b + c) % 30 = 24 :=
by
  sorry

end sum_of_remainders_eq_24_l2253_225335


namespace find_point_P_l2253_225346

def f (x : ℝ) : ℝ := x^4 - 2 * x

def tangent_line_perpendicular (x y : ℝ) : Prop :=
  (f x) = y ∧ (4 * x^3 - 2 = 2)

theorem find_point_P :
  ∃ (x y : ℝ), tangent_line_perpendicular x y ∧ x = 1 ∧ y = -1 :=
sorry

end find_point_P_l2253_225346


namespace problem_part1_problem_part2_l2253_225313

open Real

theorem problem_part1 (A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) : 
  C = 5 * π / 8 := 
sorry

theorem problem_part2 (a b c A B C : ℝ) (h1 : sin C * sin (A - B) = sin B * sin (C - A)) (h2 : A = 2 * B) (h3 : A + B + C = π):
  2 * a^2 = b^2 + c^2 :=
sorry

end problem_part1_problem_part2_l2253_225313


namespace sides_of_triangle_inequality_l2253_225363

theorem sides_of_triangle_inequality (a b c : ℝ) (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 ∧ 
  (a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c) :=
sorry

end sides_of_triangle_inequality_l2253_225363


namespace cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l2253_225319

-- Problem 1
theorem cylinder_lateral_area (C H : ℝ) (hC : C = 1.8) (hH : H = 1.5) :
  C * H = 2.7 := by sorry 

-- Problem 2
theorem cylinder_volume (D H : ℝ) (hD : D = 3) (hH : H = 8) :
  (3.14 * ((D * 10 / 2) ^ 2) * H) = 5652 :=
by sorry

-- Problem 3
theorem cylinder_surface_area (r h : ℝ) (hr : r = 6) (hh : h = 5) :
    (3.14 * r * 2 * h + 3.14 * r ^ 2 * 2) = 414.48 :=
by sorry

-- Problem 4
theorem cone_volume (B H : ℝ) (hB : B = 18.84) (hH : H = 6) :
  (1 / 3 * B * H) = 37.68 :=
by sorry

end cylinder_lateral_area_cylinder_volume_cylinder_surface_area_cone_volume_l2253_225319


namespace Tigers_Sharks_min_games_l2253_225332

open Nat

theorem Tigers_Sharks_min_games (N : ℕ) : 
  (let total_games := 3 + N
   let sharks_wins := 1 + N
   sharks_wins * 20 ≥ total_games * 19) ↔ N ≥ 37 := 
by
  sorry

end Tigers_Sharks_min_games_l2253_225332


namespace percent_increase_l2253_225315

variable (P : ℝ)
def firstQuarterPrice := 1.20 * P
def secondQuarterPrice := 1.50 * P

theorem percent_increase:
  ((secondQuarterPrice P - firstQuarterPrice P) / firstQuarterPrice P) * 100 = 25 := by
  sorry

end percent_increase_l2253_225315


namespace line_tangent_to_ellipse_l2253_225321

theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6 → ∃! y : ℝ, 3 * x^2 + 6 * y^2 = 6) →
  m^2 = 3 / 2 :=
by
  sorry

end line_tangent_to_ellipse_l2253_225321


namespace max_gold_coins_l2253_225389

theorem max_gold_coins (k : ℕ) (n : ℕ) (h : n = 13 * k + 3 ∧ n < 150) : n = 146 :=
by 
  sorry

end max_gold_coins_l2253_225389


namespace emmy_rosa_ipods_total_l2253_225317

theorem emmy_rosa_ipods_total :
  ∃ (emmy_initial rosa_current : ℕ), 
    emmy_initial = 14 ∧ 
    (emmy_initial - 6) / 2 = rosa_current ∧ 
    (emmy_initial - 6) + rosa_current = 12 :=
by
  sorry

end emmy_rosa_ipods_total_l2253_225317


namespace smallest_factor_l2253_225328

theorem smallest_factor (x : ℕ) (h1 : 936 = 2^3 * 3^1 * 13^1)
  (h2 : ∃ (x : ℕ), (936 * x) % 2^5 = 0 ∧ (936 * x) % 3^3 = 0 ∧ (936 * x) % 13^2 = 0) : x = 468 := 
sorry

end smallest_factor_l2253_225328


namespace possible_amounts_l2253_225300

theorem possible_amounts (n : ℕ) : 
  ¬ (∃ x y : ℕ, 3 * x + 5 * y = n) ↔ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 7 :=
sorry

end possible_amounts_l2253_225300


namespace sqrt_4_eq_pm2_l2253_225301

theorem sqrt_4_eq_pm2 : {y : ℝ | y^2 = 4} = {2, -2} :=
by
  sorry

end sqrt_4_eq_pm2_l2253_225301


namespace pies_from_apples_l2253_225365

theorem pies_from_apples 
  (initial_apples : ℕ) (handed_out_apples : ℕ) (apples_per_pie : ℕ) 
  (remaining_apples := initial_apples - handed_out_apples) 
  (pies := remaining_apples / apples_per_pie) 
  (h1 : initial_apples = 75) 
  (h2 : handed_out_apples = 19) 
  (h3 : apples_per_pie = 8) : 
  pies = 7 :=
by
  rw [h1, h2, h3]
  sorry

end pies_from_apples_l2253_225365


namespace find_x_l2253_225381

theorem find_x (a b x : ℝ) (h1 : 2^a = x) (h2 : 3^b = x) (h3 : 1/a + 1/b = 1) : x = 6 :=
sorry

end find_x_l2253_225381


namespace increasing_interval_of_f_l2253_225353

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * Real.pi / 3 - 2 * x)

theorem increasing_interval_of_f :
  ∃ a b : ℝ, f x = 3 * Real.sin (2 * Real.pi / 3 - 2 * x) ∧ (a = 7 * Real.pi / 12) ∧ (b = 13 * Real.pi / 12) ∧ ∀ x1 x2, a ≤ x1 ∧ x1 < x2 ∧ x2 ≤ b → f x1 < f x2 := 
sorry

end increasing_interval_of_f_l2253_225353


namespace payment_methods_20_yuan_l2253_225380

theorem payment_methods_20_yuan :
  let ten_yuan_note := 10
  let five_yuan_note := 5
  let one_yuan_note := 1
  ∃ (methods : Nat), 
    methods = 9 ∧ 
    ∃ (num_10 num_5 num_1 : Nat),
      (num_10 * ten_yuan_note + num_5 * five_yuan_note + num_1 * one_yuan_note = 20) →
      methods = 9 :=
sorry

end payment_methods_20_yuan_l2253_225380


namespace alpha_plus_beta_l2253_225318

theorem alpha_plus_beta :
  (∃ α β : ℝ, 
    (∀ x : ℝ, x ≠ -β ∧ x ≠ 45 → (x - α) / (x + β) = (x^2 - 90 * x + 1980) / (x^2 + 70 * x - 3570))
  ) → (∃ α β : ℝ, α + β = 123) :=
by {
  sorry
}

end alpha_plus_beta_l2253_225318


namespace cells_after_3_hours_l2253_225398

noncomputable def cell_division_problem (t : ℕ) : ℕ :=
  2 ^ (t * 2)

theorem cells_after_3_hours : cell_division_problem 3 = 64 := by
  sorry

end cells_after_3_hours_l2253_225398


namespace coplanar_condition_l2253_225372

-- Definitions representing points A, B, C, D and the origin O in a vector space over the reals
variables {V : Type*} [AddCommGroup V] [Module ℝ V] (O A B C D : V)

-- The main statement of the problem
theorem coplanar_condition (h : (2 : ℝ) • (A - O) - (3 : ℝ) • (B - O) + (7 : ℝ) • (C - O) + k • (D - O) = 0) :
  k = -6 :=
sorry

end coplanar_condition_l2253_225372


namespace parallel_lines_condition_l2253_225307

theorem parallel_lines_condition (a : ℝ) :
  (a = 3 / 2) ↔ (∀ x y : ℝ, (x + 2 * a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0) → (a = 3 / 2)) :=
sorry

end parallel_lines_condition_l2253_225307


namespace fraction_decimal_equivalent_l2253_225364

theorem fraction_decimal_equivalent : (7 / 16 : ℝ) = 0.4375 :=
by
  sorry

end fraction_decimal_equivalent_l2253_225364


namespace foundation_cost_l2253_225397

theorem foundation_cost (volume_per_house : ℝ)
    (density : ℝ)
    (cost_per_pound : ℝ)
    (num_houses : ℕ) 
    (dimension_len : ℝ)
    (dimension_wid : ℝ)
    (dimension_height : ℝ)
    : cost_per_pound = 0.02 → density = 150 → dimension_len = 100 → dimension_wid = 100 → dimension_height = 0.5 → num_houses = 3
    → volume_per_house = dimension_len * dimension_wid * dimension_height 
    → (num_houses : ℝ) * (volume_per_house * density * cost_per_pound) = 45000 := 
by 
  sorry

end foundation_cost_l2253_225397


namespace billy_horses_l2253_225384

theorem billy_horses (each_horse_oats_per_meal : ℕ) (meals_per_day : ℕ) (total_oats_needed : ℕ) (days : ℕ) 
    (h_each_horse_oats_per_meal : each_horse_oats_per_meal = 4)
    (h_meals_per_day : meals_per_day = 2)
    (h_total_oats_needed : total_oats_needed = 96)
    (h_days : days = 3) :
    (total_oats_needed / (days * (each_horse_oats_per_meal * meals_per_day)) = 4) :=
by
  sorry

end billy_horses_l2253_225384


namespace ahn_largest_number_l2253_225371

def largest_number_ahn_can_get : ℕ :=
  let n := 10
  2 * (200 - n)

theorem ahn_largest_number :
  (10 ≤ 99) →
  (10 ≤ 99) →
  largest_number_ahn_can_get = 380 := 
by
-- Conditions: n is a two-digit integer with range 10 ≤ n ≤ 99
-- Proof is skipped
  sorry

end ahn_largest_number_l2253_225371


namespace area_new_rectangle_l2253_225331

theorem area_new_rectangle (a b : ℝ) :
  (b + 2 * a) * (b - a) = b^2 + a * b - 2 * a^2 := by
sorry

end area_new_rectangle_l2253_225331


namespace constant_fraction_condition_l2253_225336

theorem constant_fraction_condition 
    (a1 b1 c1 a2 b2 c2 : ℝ) : 
    (∀ x : ℝ, (a1 * x^2 + b1 * x + c1) / (a2 * x^2 + b2 * x + c2) = k) ↔ 
    (a1 / a2 = b1 / b2 ∧ b1 / b2 = c1 / c2) :=
by
  sorry

end constant_fraction_condition_l2253_225336


namespace total_silver_dollars_l2253_225329

-- Definitions based on conditions
def chiu_silver_dollars : ℕ := 56
def phung_silver_dollars : ℕ := chiu_silver_dollars + 16
def ha_silver_dollars : ℕ := phung_silver_dollars + 5

-- Theorem statement
theorem total_silver_dollars : chiu_silver_dollars + phung_silver_dollars + ha_silver_dollars = 205 :=
by
  -- We use "sorry" to fill in the proof part as instructed
  sorry

end total_silver_dollars_l2253_225329


namespace find_ratio_l2253_225326

theorem find_ratio (x y c d : ℝ) (h1 : 8 * x - 6 * y = c) (h2 : 12 * y - 18 * x = d) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) : c / d = -1 := by
  sorry

end find_ratio_l2253_225326


namespace problem1_problem2_problem3_l2253_225394

-- Definition of the sequence
def a (n : ℕ) (k : ℚ) : ℚ := (k * n - 3) / (n - 3 / 2)

-- The first condition proof problem
theorem problem1 (k : ℚ) : (∀ n : ℕ, a n k = (a (n + 1) k + a (n - 1) k) / 2) → k = 2 :=
sorry

-- The second condition proof problem
theorem problem2 (k : ℚ) : 
  k ≠ 2 → 
  (if k > 2 then (a 1 k < k ∧ a 2 k = max (a 1 k) (a 2 k))
   else if k < 2 then (a 2 k < k ∧ a 1 k = max (a 1 k) (a 2 k))
   else False) :=
sorry

-- The third condition proof problem
theorem problem3 (k : ℚ) : 
  (∀ n : ℕ, n > 0 → a n k > (k * 2^n + (-1)^n) / 2^n) → 
  101 / 48 < k ∧ k < 13 / 6 :=
sorry

end problem1_problem2_problem3_l2253_225394


namespace average_weight_of_B_C_D_E_l2253_225358

theorem average_weight_of_B_C_D_E 
    (W_A W_B W_C W_D W_E : ℝ)
    (h1 : (W_A + W_B + W_C)/3 = 60)
    (h2 : W_A = 87)
    (h3 : (W_A + W_B + W_C + W_D)/4 = 65)
    (h4 : W_E = W_D + 3) :
    (W_B + W_C + W_D + W_E)/4 = 64 :=
by {
    sorry
}

end average_weight_of_B_C_D_E_l2253_225358


namespace man_speed_l2253_225311

theorem man_speed (time_in_minutes : ℝ) (distance_in_km : ℝ) (T : time_in_minutes = 24) (D : distance_in_km = 4) : 
  (distance_in_km / (time_in_minutes / 60)) = 10 := by
  sorry

end man_speed_l2253_225311


namespace geometric_common_ratio_l2253_225334

noncomputable def geo_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_common_ratio (a₁ : ℝ) (q : ℝ) (n : ℕ) 
  (h : 2 * geo_sum a₁ q n = geo_sum a₁ q (n + 1) + geo_sum a₁ q (n + 2)) : q = -2 :=
by
  sorry

end geometric_common_ratio_l2253_225334


namespace blue_notebook_cost_l2253_225304

theorem blue_notebook_cost 
  (total_spent : ℕ)
  (total_notebooks : ℕ)
  (red_notebooks : ℕ)
  (cost_per_red : ℕ)
  (green_notebooks : ℕ)
  (cost_per_green : ℕ)
  (blue_notebooks : ℕ)
  (total_cost_blue : ℕ)
  (cost_per_blue : ℕ)
  (h1 : total_spent = 37)
  (h2 : total_notebooks = 12)
  (h3 : red_notebooks = 3)
  (h4 : cost_per_red = 4)
  (h5 : green_notebooks = 2)
  (h6 : cost_per_green = 2)
  (h7 : total_cost_blue = total_spent - (red_notebooks * cost_per_red + green_notebooks * cost_per_green))
  (h8 : blue_notebooks = total_notebooks - (red_notebooks + green_notebooks))
  (h9 : cost_per_blue = total_cost_blue / blue_notebooks)
  : cost_per_blue = 3 :=
sorry

end blue_notebook_cost_l2253_225304


namespace average_age_of_three_l2253_225322

theorem average_age_of_three (Tonya_age John_age Mary_age : ℕ)
  (h1 : John_age = 2 * Mary_age)
  (h2 : Tonya_age = 2 * John_age)
  (h3 : Tonya_age = 60) :
  (Tonya_age + John_age + Mary_age) / 3 = 35 := by
  sorry

end average_age_of_three_l2253_225322


namespace number_of_ordered_triplets_l2253_225355

theorem number_of_ordered_triplets :
  ∃ count : ℕ, (∀ (a b c : ℕ), lcm a b = 1000 ∧ lcm b c = 2000 ∧ lcm c a = 2000 →
  count = 70) :=
sorry

end number_of_ordered_triplets_l2253_225355


namespace divisible_bc_ad_l2253_225387

theorem divisible_bc_ad
  (a b c d u : ℤ)
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) :
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end divisible_bc_ad_l2253_225387


namespace length_QF_l2253_225302

-- Define parabola C as y^2 = 8x
def is_on_parabola (P : ℝ × ℝ) : Prop :=
  P.2 * P.2 = 8 * P.1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define the condition that Q is on the parabola and the line PF in the first quadrant
def is_intersection_and_in_first_quadrant (Q : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  is_on_parabola Q ∧ Q.1 - Q.2 - 2 = 0 ∧ Q.1 > 0 ∧ Q.2 > 0

-- Define the vector relation between P, Q, and F
def vector_relation (P Q F : ℝ × ℝ) : Prop :=
  let vPQ := (Q.1 - P.1, Q.2 - P.2)
  let vQF := (F.1 - Q.1, F.2 - Q.2)
  (vPQ.1^2 + vPQ.2^2) = 2 * (vQF.1^2 + vQF.2^2)

-- Lean 4 statement of the proof problem
theorem length_QF (Q : ℝ × ℝ) (P : ℝ × ℝ) :
  is_on_parabola Q ∧ is_intersection_and_in_first_quadrant Q P ∧ vector_relation P Q focus → 
  dist Q focus = 8 + 4 * Real.sqrt 2 :=
by
  sorry

end length_QF_l2253_225302


namespace NumberOfRootsForEquation_l2253_225338

noncomputable def numRootsAbsEq : ℕ :=
  let f := (fun x : ℝ => abs (abs (abs (abs (x - 1) - 9) - 9) - 3))
  let roots : List ℝ := [27, -25, 11, -9, 9, -7]
  roots.length

theorem NumberOfRootsForEquation : numRootsAbsEq = 6 := by
  sorry

end NumberOfRootsForEquation_l2253_225338


namespace radical_axis_eq_l2253_225343

-- Definitions of the given circles
def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 6 * y = 0
def circle2_eq (x y : ℝ) : Prop := x^2 + y^2 - 6 * x = 0

-- The theorem proving that the equation of the radical axis is 3x - y - 9 = 0
theorem radical_axis_eq (x y : ℝ) :
  (circle1_eq x y) ∧ (circle2_eq x y) → 3 * x - y - 9 = 0 :=
sorry

end radical_axis_eq_l2253_225343


namespace total_books_l2253_225320

def keith_books : ℕ := 20
def jason_books : ℕ := 21

theorem total_books : keith_books + jason_books = 41 :=
by
  sorry

end total_books_l2253_225320


namespace solve_equation1_solve_equation2_l2253_225377

theorem solve_equation1 (x : ℝ) (h1 : 2 * x - 9 = 4 * x) : x = -9 / 2 :=
by
  sorry

theorem solve_equation2 (x : ℝ) (h2 : 5 / 2 * x - 7 / 3 * x = 4 / 3 * 5 - 5) : x = 10 :=
by
  sorry

end solve_equation1_solve_equation2_l2253_225377


namespace work_completion_in_8_days_l2253_225303

/-- Definition of the individual work rates and the combined work rate. -/
def work_rate_A := 1 / 12
def work_rate_B := 1 / 24
def combined_work_rate := work_rate_A + work_rate_B

/-- The main theorem stating that A and B together complete the job in 8 days. -/
theorem work_completion_in_8_days (h1 : work_rate_A = 1 / 12) (h2 : work_rate_B = 1 / 24) : 
  1 / combined_work_rate = 8 :=
by
  sorry

end work_completion_in_8_days_l2253_225303


namespace fixed_chord_property_l2253_225392

theorem fixed_chord_property (d : ℝ) (h₁ : d = 3 / 2) :
  ∀ (x1 x2 m : ℝ) (h₀ : x1 + x2 = m) (h₂ : x1 * x2 = 1 - d),
    ((1 / ((x1 ^ 2) + (m * x1) ^ 2)) + (1 / ((x2 ^ 2) + (m * x2) ^ 2))) = 4 / 9 :=
by
  sorry

end fixed_chord_property_l2253_225392


namespace elena_probability_at_least_one_correct_l2253_225373

-- Conditions
def total_questions := 30
def choices_per_question := 4
def guessed_questions := 6
def incorrect_probability_single := 3 / 4

-- Expression for the probability of missing all guessed questions
def probability_all_incorrect := (incorrect_probability_single) ^ guessed_questions

-- Calculation from the solution
def probability_at_least_one_correct := 1 - probability_all_incorrect

-- Problem statement to prove
theorem elena_probability_at_least_one_correct : probability_at_least_one_correct = 3367 / 4096 :=
by sorry

end elena_probability_at_least_one_correct_l2253_225373


namespace derek_alice_pair_l2253_225325

-- Variables and expressions involved
variable (x b c : ℝ)

-- Definitions of the conditions
def derek_eq := |x + 3| = 5 
def alice_eq := ∀ a, (a - 2) * (a + 8) = a^2 + b * a + c

-- The theorem to prove
theorem derek_alice_pair : derek_eq x → alice_eq b c → (b, c) = (6, -16) :=
by
  intros h1 h2
  sorry

end derek_alice_pair_l2253_225325


namespace fold_point_area_sum_l2253_225350

noncomputable def fold_point_area (AB AC : ℝ) (angle_B : ℝ) : ℝ :=
  let BC := Real.sqrt (AB ^ 2 + AC ^ 2)
  -- Assuming the fold point area calculation as per the problem's solution
  let q := 270
  let r := 324
  let s := 3
  q * Real.pi - r * Real.sqrt s

theorem fold_point_area_sum (AB AC : ℝ) (angle_B : ℝ) (hAB : AB = 36) (hAC : AC = 72) (hangle_B : angle_B = π / 2) :
  let S := fold_point_area AB AC angle_B
  ∃ q r s : ℕ, S = q * Real.pi - r * Real.sqrt s ∧ q + r + s = 597 :=
by
  sorry

end fold_point_area_sum_l2253_225350


namespace distance_between_stations_l2253_225396

theorem distance_between_stations :
  ∀ (x t : ℕ), 
    (20 * t = x) ∧ 
    (25 * t = x + 70) →
    (2 * x + 70 = 630) :=
by
  sorry

end distance_between_stations_l2253_225396


namespace find_a_in_terms_of_y_l2253_225361

theorem find_a_in_terms_of_y (a b y : ℝ) (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * y^3) (h3 : a - b = 3 * y) :
  a = 3 * y :=
sorry

end find_a_in_terms_of_y_l2253_225361


namespace square_root_of_4_is_pm2_l2253_225323

theorem square_root_of_4_is_pm2 : ∃ (x : ℤ), x * x = 4 ∧ (x = 2 ∨ x = -2) := by
  sorry

end square_root_of_4_is_pm2_l2253_225323


namespace lcm_48_180_l2253_225305

theorem lcm_48_180 : Nat.lcm 48 180 = 720 :=
by 
  sorry

end lcm_48_180_l2253_225305


namespace solve_system_equations_l2253_225351

theorem solve_system_equations :
  ∃ x y : ℚ, (5 * x * (y + 6) = 0 ∧ 2 * x + 3 * y = 1) ∧
  (x = 0 ∧ y = 1 / 3 ∨ x = 19 / 2 ∧ y = -6) :=
by
  sorry

end solve_system_equations_l2253_225351


namespace sum_of_three_sqrt_139_l2253_225385

theorem sum_of_three_sqrt_139 {x y z : ℝ} (h1 : x >= 0) (h2 : y >= 0) (h3 : z >= 0)
  (hx : x^2 + y^2 + z^2 = 75) (hy : x * y + y * z + z * x = 32) : x + y + z = Real.sqrt 139 := 
by
  sorry

end sum_of_three_sqrt_139_l2253_225385


namespace journey_total_distance_l2253_225333

theorem journey_total_distance (D : ℝ) (h_train : D * (3 / 5) = t) (h_bus : D * (7 / 20) = b) (h_walk : D * (1 - ((3 / 5) + (7 / 20))) = 6.5) : D = 130 :=
by
  sorry

end journey_total_distance_l2253_225333


namespace correct_proposition_l2253_225341

-- Definitions of the propositions p and q
def p : Prop := ∀ x : ℝ, (x > 1 → x > 2)
def q : Prop := ∀ x y : ℝ, (x + y ≠ 2 → x ≠ 1 ∨ y ≠ 1)

-- The proof problem statement
theorem correct_proposition : ¬p ∧ q :=
by
  -- Assuming p is false (i.e., ¬p is true) and q is true
  sorry

end correct_proposition_l2253_225341


namespace height_of_given_cylinder_l2253_225383

noncomputable def height_of_cylinder (P d : ℝ) : ℝ :=
  let r := P / (2 * Real.pi)
  let l := P
  let h := Real.sqrt (d^2 - l^2)
  h

theorem height_of_given_cylinder : height_of_cylinder 6 10 = 8 :=
by
  show height_of_cylinder 6 10 = 8
  sorry

end height_of_given_cylinder_l2253_225383


namespace min_value_of_a_l2253_225330

theorem min_value_of_a : 
  ∃ (a : ℤ), ∃ x y : ℤ, x ≠ y ∧ |x| ≤ 10 ∧ (x - y^2 = a) ∧ (y - x^2 = a) ∧ a = -111 :=
by
  sorry

end min_value_of_a_l2253_225330


namespace residue_system_mod_3n_l2253_225347

theorem residue_system_mod_3n (n : ℕ) (h_odd : n % 2 = 1) :
  ∃ (a b : ℕ → ℕ) (k : ℕ), 
  (∀ i, a i = 3 * i - 2) ∧ 
  (∀ i, b i = 3 * i - 3) ∧
  (∀ i (k : ℕ), 0 < k ∧ k < n → 
    (a i + a (i + 1)) % (3 * n) ≠ (a i + b i) % (3 * n) ∧ 
    (a i + b i) % (3 * n) ≠ (b i + b (i + k)) % (3 * n) ∧ 
    (a i + a (i + 1)) % (3 * n) ≠ (b i + b (i + k)) % (3 * n)) :=
sorry

end residue_system_mod_3n_l2253_225347


namespace function_range_ge_4_l2253_225379

variable {x : ℝ}

theorem function_range_ge_4 (h : x > 0) : 2 * x + 2 * x⁻¹ ≥ 4 :=
sorry

end function_range_ge_4_l2253_225379


namespace base7_calculation_result_l2253_225370

-- Define the base 7 addition and multiplication
def base7_add (a b : ℕ) := (a + b)
def base7_mul (a b : ℕ) := (a * b)

-- Represent the given numbers in base 10 for calculations:
def num1 : ℕ := 2 * 7 + 5 -- 25 in base 7
def num2 : ℕ := 3 * 7^2 + 3 * 7 + 4 -- 334 in base 7
def mul_factor : ℕ := 2 -- 2 in base 7

-- Addition result
def sum : ℕ := base7_add num1 num2

-- Multiplication result
def result : ℕ := base7_mul sum mul_factor

-- Proving the result is equal to the final answer in base 7
theorem base7_calculation_result : result = 6 * 7^2 + 6 * 7 + 4 := 
by sorry

end base7_calculation_result_l2253_225370


namespace sum_geometric_series_l2253_225312

-- Given the conditions
def q : ℕ := 2
def a3 : ℕ := 16
def n : ℕ := 2017
def a1 : ℕ := 4

-- Define the sum of the first n terms of a geometric series
noncomputable def geometricSeriesSum (a1 q n : ℕ) : ℕ :=
  a1 * (1 - q^n) / (1 - q)

-- State the problem
theorem sum_geometric_series :
  geometricSeriesSum a1 q n = 2^2019 - 4 :=
sorry

end sum_geometric_series_l2253_225312


namespace real_roots_iff_l2253_225359

theorem real_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 + 2 * k * x + 3 * k^2 + 2 * k = 0) ↔ (-1 ≤ k ∧ k ≤ 0) :=
by sorry

end real_roots_iff_l2253_225359


namespace plane_passing_through_A_perpendicular_to_BC_l2253_225362

-- Define the points A, B, and C
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def A : Point3D := { x := -3, y := 7, z := 2 }
def B : Point3D := { x := 3, y := 5, z := 1 }
def C : Point3D := { x := 4, y := 5, z := 3 }

-- Define the vector BC as the difference between points C and B
def vectorBC (B C : Point3D) : Point3D :=
{ x := C.x - B.x,
  y := C.y - B.y,
  z := C.z - B.z }

-- Define the equation of the plane passing through point A and 
-- perpendicular to vector BC
def plane_eq (A : Point3D) (n : Point3D) (x y z : ℝ) : Prop :=
n.x * (x - A.x) + n.y * (y - A.y) + n.z * (z - A.z) = 0

-- Define the proof problem
theorem plane_passing_through_A_perpendicular_to_BC :
  ∀ (x y z : ℝ), plane_eq A (vectorBC B C) x y z ↔ x + 2 * z - 1 = 0 :=
by
  -- the proof part
  sorry

end plane_passing_through_A_perpendicular_to_BC_l2253_225362


namespace vertical_angles_equal_l2253_225386

-- Given: Definition for pairs of adjacent angles summing up to 180 degrees
def adjacent_add_to_straight_angle (α β : ℝ) : Prop := 
  α + β = 180

-- Given: Two intersecting lines forming angles
variables (α β γ δ : ℝ)

-- Given: Relationship of adjacent angles being supplementary
axiom adj1 : adjacent_add_to_straight_angle α β
axiom adj2 : adjacent_add_to_straight_angle β γ
axiom adj3 : adjacent_add_to_straight_angle γ δ
axiom adj4 : adjacent_add_to_straight_angle δ α

-- Question: Prove that vertical angles are equal
theorem vertical_angles_equal : α = γ :=
by sorry

end vertical_angles_equal_l2253_225386


namespace line_intercepts_of_3x_minus_y_plus_6_eq_0_l2253_225360

theorem line_intercepts_of_3x_minus_y_plus_6_eq_0 :
  (∃ y, 3 * 0 - y + 6 = 0 ∧ y = 6) ∧ (∃ x, 3 * x - 0 + 6 = 0 ∧ x = -2) :=
by
  sorry

end line_intercepts_of_3x_minus_y_plus_6_eq_0_l2253_225360


namespace solve_equation_error_step_l2253_225354

theorem solve_equation_error_step 
  (equation : ∀ x : ℝ, (x - 1) / 2 + 1 = (2 * x + 1) / 3) :
  ∃ (step : ℕ), step = 1 ∧
  let s1 := ((x - 1) / 2 + 1) * 6;
  ∀ (x : ℝ), s1 ≠ (((2 * x + 1) / 3) * 6) :=
by
  sorry

end solve_equation_error_step_l2253_225354


namespace part1_part2_l2253_225337

-- Definition of the function f
def f (x m : ℝ) : ℝ := abs (x - m) + abs (x + 3)

-- Part 1: For m = 1, the solution set of f(x) >= 6
theorem part1 (x : ℝ) : f x 1 ≥ 6 ↔ x ≤ -4 ∨ x ≥ 2 := 
by 
  sorry

-- Part 2: If the inequality f(x) ≤ 2m - 5 has a solution with respect to x, then m ≥ 8
theorem part2 (m : ℝ) (h : ∃ x, f x m ≤ 2 * m - 5) : m ≥ 8 :=
by
  sorry

end part1_part2_l2253_225337


namespace avg_weight_increase_l2253_225327

theorem avg_weight_increase
  (A : ℝ) -- Initial average weight
  (n : ℕ) -- Initial number of people
  (w_old : ℝ) -- Weight of the person being replaced
  (w_new : ℝ) -- Weight of the new person
  (h_n : n = 8) -- Initial number of people is 8
  (h_w_old : w_old = 85) -- Weight of the replaced person is 85
  (h_w_new : w_new = 105) -- Weight of the new person is 105
  : ((8 * A + (w_new - w_old)) / 8) - A = 2.5 := 
sorry

end avg_weight_increase_l2253_225327


namespace paperboy_delivery_count_l2253_225309

def no_miss_four_consecutive (n : ℕ) (E : ℕ → ℕ) : Prop :=
  ∀ k > 3, E k = E (k - 1) + E (k - 2) + E (k - 3)

def base_conditions (E : ℕ → ℕ) : Prop :=
  E 1 = 2 ∧ E 2 = 4 ∧ E 3 = 8

theorem paperboy_delivery_count : ∃ (E : ℕ → ℕ), 
  base_conditions E ∧ no_miss_four_consecutive 12 E ∧ E 12 = 1854 :=
by
  sorry

end paperboy_delivery_count_l2253_225309


namespace solve_r_l2253_225348

variable (r : ℝ)

theorem solve_r : (r + 3) / (r - 2) = (r - 1) / (r + 1) → r = -1/7 := by
  sorry

end solve_r_l2253_225348


namespace shaded_area_quadrilateral_l2253_225306

theorem shaded_area_quadrilateral :
  let large_square_area := 11 * 11
  let small_square_area_1 := 1 * 1
  let small_square_area_2 := 2 * 2
  let small_square_area_3 := 3 * 3
  let small_square_area_4 := 4 * 4
  let other_non_shaded_areas := 12 + 15 + 14
  let total_non_shaded := small_square_area_1 + small_square_area_2 + small_square_area_3 + small_square_area_4 + other_non_shaded_areas
  let shaded_area := large_square_area - total_non_shaded
  shaded_area = 35 := by
  sorry

end shaded_area_quadrilateral_l2253_225306


namespace simplify_and_evaluate_l2253_225314

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -1) (hy : y = -1/3) :
  ((3 * x^2 + x * y + 2 * y) - 2 * (5 * x * y - 4 * x^2 + y)) = 8 := by
  sorry

end simplify_and_evaluate_l2253_225314


namespace part1_part2_part3_l2253_225376

noncomputable def f (x : ℝ) : ℝ := 3 * x - Real.exp x + 1

theorem part1 :
  ∃ x0 > 0, f x0 = 0 :=
sorry

theorem part2 (x0 : ℝ) (h1 : f x0 = 0) :
  ∀ x, f x ≤ (3 - Real.exp x0) * (x - x0) :=
sorry

theorem part3 (m x1 x2 : ℝ) (h1 : m > 0) (h2 : x1 < x2) (h3 : f x1 = m) (h4 : f x2 = m):
  x2 - x1 < 2 - 3 * m / 4 :=
sorry

end part1_part2_part3_l2253_225376


namespace initial_pups_per_mouse_l2253_225349

-- Definitions from the problem's conditions
def initial_mice : ℕ := 8
def stress_factor : ℕ := 2
def second_round_pups : ℕ := 6
def total_mice : ℕ := 280

-- Define a variable for the initial number of pups each mouse had
variable (P : ℕ)

-- Lean statement to prove the number of initial pups per mouse
theorem initial_pups_per_mouse (P : ℕ) (initial_mice stress_factor second_round_pups total_mice : ℕ) :
  total_mice = initial_mice + initial_mice * P + (initial_mice + initial_mice * P) * second_round_pups - stress_factor * (initial_mice + initial_mice * P) → 
  P = 6 := 
by
  sorry

end initial_pups_per_mouse_l2253_225349


namespace solution_set_of_inequality_l2253_225366

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

theorem solution_set_of_inequality (H1 : f 1 = 1)
  (H2 : ∀ x : ℝ, x * f' x < 1 / 2) :
  {x : ℝ | f (Real.log x ^ 2) < (Real.log x ^ 2) / 2 + 1 / 2} = 
  {x : ℝ | 0 < x ∧ x < 1 / 10} ∪ {x : ℝ | x > 10} :=
sorry

end solution_set_of_inequality_l2253_225366


namespace minimum_value_l2253_225388

theorem minimum_value (x : ℝ) (hx : 0 < x) : ∃ y, (y = x + 4 / (x + 1)) ∧ (∀ z, (x > 0 → z = x + 4 / (x + 1)) → 3 ≤ z) := sorry

end minimum_value_l2253_225388


namespace no_parallelogram_on_convex_graph_l2253_225399

-- Definition of strictly convex function
def is_strictly_convex (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x t y : ℝ⦄, (x < t ∧ t < y) → f t < ((f y - f x) / (y - x)) * (t - x) + f x

-- The main statement of the problem
theorem no_parallelogram_on_convex_graph (f : ℝ → ℝ) :
  is_strictly_convex f →
  ¬ ∃ (a b c d : ℝ), a < b ∧ b < c ∧ c < d ∧
    (f b < (f c - f a) / (c - a) * (b - a) + f a) ∧
    (f c < (f d - f b) / (d - b) * (c - b) + f b) :=
sorry

end no_parallelogram_on_convex_graph_l2253_225399


namespace boxes_A_B_cost_condition_boxes_B_profit_condition_l2253_225368

/-
Part 1: Prove the number of brand A boxes is 60 and number of brand B boxes is 40 given the cost condition.
-/
theorem boxes_A_B_cost_condition (x : ℕ) (y : ℕ) :
  80 * x + 130 * y = 10000 ∧ x + y = 100 → x = 60 ∧ y = 40 :=
by sorry

/-
Part 2: Prove the number of brand B boxes should be at least 54 given the profit condition.
-/
theorem boxes_B_profit_condition (y : ℕ) :
  40 * (100 - y) + 70 * y ≥ 5600 → y ≥ 54 :=
by sorry

end boxes_A_B_cost_condition_boxes_B_profit_condition_l2253_225368


namespace xyz_value_l2253_225310

theorem xyz_value (x y z : ℕ) (h1 : x + 2 * y = z) (h2 : x^2 - 4 * y^2 + z^2 = 310) :
  xyz = 4030 ∨ xyz = 23870 :=
by
  -- placeholder for proof steps
  sorry

end xyz_value_l2253_225310


namespace value_of_x_plus_y_pow_2023_l2253_225378

theorem value_of_x_plus_y_pow_2023 (x y : ℝ) (h : abs (x - 2) + abs (y + 3) = 0) : 
  (x + y) ^ 2023 = -1 := 
sorry

end value_of_x_plus_y_pow_2023_l2253_225378


namespace sum_of_four_primes_div_by_60_l2253_225352

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_four_primes_div_by_60
  (p q r s : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (hs : is_prime s)
  (horder : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) :
  (p + q + r + s) % 60 = 0 :=
by
  sorry


end sum_of_four_primes_div_by_60_l2253_225352


namespace expected_value_equals_51_l2253_225357

noncomputable def expected_value_8_sided_die : ℝ :=
  (1 / 8) * (2 * 1^2 + 2 * 2^2 + 2 * 3^2 + 2 * 4^2 + 2 * 5^2 + 2 * 6^2 + 2 * 7^2 + 2 * 8^2)

theorem expected_value_equals_51 :
  expected_value_8_sided_die = 51 := 
  by 
    sorry

end expected_value_equals_51_l2253_225357


namespace solve_for_x_l2253_225342

-- Define the variables and conditions
variable (x : ℚ)

-- Define the given condition
def condition : Prop := (x + 4)/(x - 3) = (x - 2)/(x + 2)

-- State the theorem that x = -2/11 is a solution to the condition
theorem solve_for_x (h : condition x) : x = -2 / 11 := by
  sorry

end solve_for_x_l2253_225342


namespace polygon_area_is_correct_l2253_225316

def points : List (ℕ × ℕ) := [
  (0, 0), (10, 0), (20, 0), (30, 10),
  (0, 20), (10, 20), (20, 30), (10, 30),
  (0, 30), (20, 10), (30, 20), (10, 10)
]

def polygon_area (ps : List (ℕ × ℕ)) : ℕ := sorry

theorem polygon_area_is_correct :
  polygon_area points = 9 := sorry

end polygon_area_is_correct_l2253_225316


namespace smallest_angle_satisfying_trig_eqn_l2253_225369

theorem smallest_angle_satisfying_trig_eqn :
  ∃ x : ℝ, 0 < x ∧ 8 * (Real.sin x)^2 * (Real.cos x)^4 - 8 * (Real.sin x)^4 * (Real.cos x)^2 = 1 ∧ x = 10 :=
by
  sorry

end smallest_angle_satisfying_trig_eqn_l2253_225369


namespace find_x_l2253_225393

theorem find_x (x : ℝ) : 0.003 + 0.158 + x = 2.911 → x = 2.750 :=
by
  sorry

end find_x_l2253_225393


namespace original_inhabitants_7200_l2253_225345

noncomputable def original_inhabitants (X : ℝ) : Prop :=
  let initial_decrease := 0.9 * X
  let final_decrease := 0.75 * initial_decrease
  final_decrease = 4860

theorem original_inhabitants_7200 : ∃ X : ℝ, original_inhabitants X ∧ X = 7200 := by
  use 7200
  unfold original_inhabitants
  simp
  sorry

end original_inhabitants_7200_l2253_225345


namespace worker_surveys_per_week_l2253_225344

theorem worker_surveys_per_week :
  let regular_rate := 30
  let cellphone_rate := regular_rate + 0.20 * regular_rate
  let surveys_with_cellphone := 50
  let earnings := 3300
  cellphone_rate = regular_rate + 0.20 * regular_rate →
  earnings = surveys_with_cellphone * cellphone_rate →
  regular_rate = 30 →
  surveys_with_cellphone = 50 →
  earnings = 3300 →
  surveys_with_cellphone = 50 := sorry

end worker_surveys_per_week_l2253_225344


namespace greatest_third_side_l2253_225356

-- Given data and the Triangle Inequality theorem
theorem greatest_third_side (c : ℕ) (h1 : 8 < c) (h2 : c < 22) : c = 21 :=
by
  sorry

end greatest_third_side_l2253_225356


namespace store_breaks_even_l2253_225395

-- Defining the conditions based on the problem statement.
def cost_price_piece1 (profitable : ℝ → Prop) : Prop :=
  ∃ x, profitable x ∧ 1.5 * x = 150

def cost_price_piece2 (loss : ℝ → Prop) : Prop :=
  ∃ y, loss y ∧ 0.75 * y = 150

def profitable (x : ℝ) : Prop := x + 0.5 * x = 150
def loss (y : ℝ) : Prop := y - 0.25 * y = 150

-- Store breaks even if the total cost price equals the total selling price
theorem store_breaks_even (x y : ℝ)
  (P1 : cost_price_piece1 profitable)
  (P2 : cost_price_piece2 loss) :
  (x + y = 100 + 200) → (150 + 150) = 300 :=
by
  sorry

end store_breaks_even_l2253_225395
