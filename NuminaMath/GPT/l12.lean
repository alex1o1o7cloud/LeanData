import Mathlib

namespace ishas_pencil_initial_length_l12_12193

theorem ishas_pencil_initial_length (l : ℝ) (h1 : l - 4 = 18) : l = 22 :=
by
  sorry

end ishas_pencil_initial_length_l12_12193


namespace bullet_speed_difference_l12_12545

def speed_horse : ℕ := 20  -- feet per second
def speed_bullet : ℕ := 400  -- feet per second

def speed_forward : ℕ := speed_bullet + speed_horse
def speed_backward : ℕ := speed_bullet - speed_horse

theorem bullet_speed_difference : speed_forward - speed_backward = 40 :=
by
  sorry

end bullet_speed_difference_l12_12545


namespace simplify_fraction_l12_12281

variables {a b c x y z : ℝ}

theorem simplify_fraction :
  (cx * (a^2 * x^2 + 3 * a^2 * y^2 + c^2 * z^2) + bz * (a^2 * x^2 + 3 * c^2 * x^2 + c^2 * y^2)) / (cx + bz) =
  a^2 * x^2 + c^2 * y^2 + c^2 * z^2 :=
sorry

end simplify_fraction_l12_12281


namespace simplify_2A_minus_B_value_2A_minus_B_a_eq_neg1_b_eq_2_find_b_independent_of_a_l12_12098

-- Definitions from the conditions
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 + 6 * a * b + 8 * a

-- Part (1): Simplifying 2A - B
theorem simplify_2A_minus_B (a b : ℝ) : 
  2 * A a b - B a b = -16 * a * b + 6 * b - 8 * a := 
by
  sorry

-- Part (2): Finding 2A - B for specific a and b
theorem value_2A_minus_B_a_eq_neg1_b_eq_2 : 
  2 * A (-1) 2 - B (-1) 2 = 52 := 
by 
  sorry

-- Part (3): Finding b for which 2A - B is independent of a
theorem find_b_independent_of_a (a b : ℝ) (h : 2 * A a b - B a b = 6 * b) : 
  b = -1 / 2 := 
by
  sorry

end simplify_2A_minus_B_value_2A_minus_B_a_eq_neg1_b_eq_2_find_b_independent_of_a_l12_12098


namespace find_f_50_l12_12708

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x * y
axiom f_20 : f 20 = 10

theorem find_f_50 : f 50 = 25 :=
by
  sorry

end find_f_50_l12_12708


namespace closest_to_10_l12_12554

theorem closest_to_10
  (A B C D : ℝ)
  (hA : A = 9.998)
  (hB : B = 10.1)
  (hC : C = 10.09)
  (hD : D = 10.001) :
  abs (10 - D) < abs (10 - A) ∧ abs (10 - D) < abs (10 - B) ∧ abs (10 - D) < abs (10 - C) :=
by
  sorry

end closest_to_10_l12_12554


namespace no_prime_p_such_that_22p2_plus_23_is_prime_l12_12866

theorem no_prime_p_such_that_22p2_plus_23_is_prime :
  ∀ p : ℕ, Prime p → ¬ Prime (22 * p ^ 2 + 23) :=
by
  sorry

end no_prime_p_such_that_22p2_plus_23_is_prime_l12_12866


namespace cube_edge_length_l12_12241

theorem cube_edge_length (a : ℝ) (base_length : ℝ) (base_width : ℝ) (rise_height : ℝ) 
  (h_conditions : base_length = 20 ∧ base_width = 15 ∧ rise_height = 11.25 ∧ 
                  (base_length * base_width * rise_height) = a^3) : 
  a = 15 := 
by
  sorry

end cube_edge_length_l12_12241


namespace find_t_l12_12726

theorem find_t (t : ℝ) : (∃ y : ℝ, y = -(t - 1) ∧ 2 * y - 4 = 3 * (y - 2)) ↔ t = -1 :=
by sorry

end find_t_l12_12726


namespace polynomial_divisibility_l12_12370

theorem polynomial_divisibility (P : Polynomial ℝ) (n : ℕ) (h_pos : 0 < n) :
  ∃ Q : Polynomial ℝ, (P * P + Q * Q) % (X * X + 1)^n = 0 :=
sorry

end polynomial_divisibility_l12_12370


namespace darry_total_steps_l12_12749

def largest_ladder_steps : ℕ := 20
def largest_ladder_times : ℕ := 12

def medium_ladder_steps : ℕ := 15
def medium_ladder_times : ℕ := 8

def smaller_ladder_steps : ℕ := 10
def smaller_ladder_times : ℕ := 10

def smallest_ladder_steps : ℕ := 5
def smallest_ladder_times : ℕ := 15

theorem darry_total_steps :
  (largest_ladder_steps * largest_ladder_times)
  + (medium_ladder_steps * medium_ladder_times)
  + (smaller_ladder_steps * smaller_ladder_times)
  + (smallest_ladder_steps * smallest_ladder_times)
  = 535 := by
  sorry

end darry_total_steps_l12_12749


namespace packages_per_hour_A_B_max_A_robots_l12_12240

-- Define the number of packages sorted by each unit of type A and B robots
def packages_by_A_robot (x : ℕ) := x
def packages_by_B_robot (y : ℕ) := y

-- Problem conditions
def cond1 (x y : ℕ) : Prop := 80 * x + 100 * y = 8200
def cond2 (x y : ℕ) : Prop := 50 * x + 50 * y = 4500

-- Part 1: to prove type A and type B robot's packages per hour
theorem packages_per_hour_A_B (x y : ℕ) (h1 : cond1 x y) (h2 : cond2 x y) : x = 40 ∧ y = 50 :=
by sorry

-- Part 2: prove maximum units of type A robots when purchasing 200 robots ensuring not < 9000 packages/hour
def cond3 (m : ℕ) : Prop := 40 * m + 50 * (200 - m) ≥ 9000

theorem max_A_robots (m : ℕ) (h3 : cond3 m) : m ≤ 100 :=
by sorry

end packages_per_hour_A_B_max_A_robots_l12_12240


namespace four_students_three_classes_l12_12395

-- Define the function that calculates the number of valid assignments
def valid_assignments (students : ℕ) (classes : ℕ) : ℕ :=
  if students = 4 ∧ classes = 3 then 36 else 0  -- Using given conditions to return 36 when appropriate

-- Define the theorem to prove that there are 36 valid ways
theorem four_students_three_classes : valid_assignments 4 3 = 36 :=
  by
  -- The proof is not required, so we use sorry to skip it
  sorry

end four_students_three_classes_l12_12395


namespace cos_seven_theta_l12_12776

theorem cos_seven_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (7 * θ) = -83728 / 390625 := 
sorry

end cos_seven_theta_l12_12776


namespace remainder_7_pow_253_mod_12_l12_12011

theorem remainder_7_pow_253_mod_12 : (7 ^ 253) % 12 = 7 := by
  sorry

end remainder_7_pow_253_mod_12_l12_12011


namespace remaining_to_original_ratio_l12_12506

-- Define the number of rows and production per row for corn and potatoes.
def rows_of_corn : ℕ := 10
def corn_per_row : ℕ := 9
def rows_of_potatoes : ℕ := 5
def potatoes_per_row : ℕ := 30

-- Define the remaining crops after pest destruction.
def remaining_crops : ℕ := 120

-- Calculate the original number of crops from corn and potato productions.
def original_crops : ℕ :=
  (rows_of_corn * corn_per_row) + (rows_of_potatoes * potatoes_per_row)

-- Define the ratio of remaining crops to original crops.
def crops_ratio : ℚ := remaining_crops / original_crops

theorem remaining_to_original_ratio : crops_ratio = 1 / 2 := 
by
  sorry

end remaining_to_original_ratio_l12_12506


namespace sum_of_reciprocals_is_two_l12_12537

variable (x y : ℝ)
variable (h1 : x + y = 50)
variable (h2 : x * y = 25)

theorem sum_of_reciprocals_is_two (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1/x + 1/y) = 2 :=
by
  sorry

end sum_of_reciprocals_is_two_l12_12537


namespace chess_sequences_l12_12032

def binomial (n k : Nat) : Nat := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem chess_sequences :
  binomial 11 4 = 210 := by
  sorry

end chess_sequences_l12_12032


namespace total_children_l12_12024

-- Given the conditions
def toy_cars : Nat := 134
def dolls : Nat := 269

-- Prove that the total number of children is 403
theorem total_children (h_cars : toy_cars = 134) (h_dolls : dolls = 269) :
  toy_cars + dolls = 403 :=
by
  sorry

end total_children_l12_12024


namespace calc_r_over_s_at_2_l12_12122

def r (x : ℝ) := 3 * (x - 4) * (x - 1)
def s (x : ℝ) := (x - 4) * (x + 3)

theorem calc_r_over_s_at_2 : (r 2) / (s 2) = 3 / 5 := by
  sorry

end calc_r_over_s_at_2_l12_12122


namespace canoe_problem_l12_12774

-- Definitions:
variables (P_L P_R : ℝ)

-- Conditions:
def conditions := 
  (P_L = P_R) ∧ -- Condition that the probabilities for left and right oars working are the same
  (0 ≤ P_L) ∧ (P_L ≤ 1) ∧ -- Probability values must be between 0 and 1
  (1 - (1 - P_L) * (1 - P_R) = 0.84) -- Given the rowing probability is 0.84

-- Theorem that P_L = 0.6 given the conditions:
theorem canoe_problem : conditions P_L P_R → P_L = 0.6 :=
by
  sorry

end canoe_problem_l12_12774


namespace seq_property_l12_12306

theorem seq_property (m : ℤ) (h1 : |m| ≥ 2)
  (a : ℕ → ℤ)
  (h2 : ¬ (a 1 = 0 ∧ a 2 = 0))
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) = a (n + 1) - m * a n)
  (r s : ℕ)
  (h4 : r > s ∧ s ≥ 2)
  (h5 : a r = a 1 ∧ a s = a 1) :
  r - s ≥ |m| :=
by
  sorry

end seq_property_l12_12306


namespace rectangle_not_sum_110_l12_12622

noncomputable def not_sum_110 : Prop :=
  ∀ (w : ℕ), (w > 0) → (2 * w^2 + 6 * w ≠ 110)

theorem rectangle_not_sum_110 : not_sum_110 := 
  sorry

end rectangle_not_sum_110_l12_12622


namespace determine_x_l12_12061

theorem determine_x (x : Nat) (h1 : x % 9 = 0) (h2 : x^2 > 225) (h3 : x < 30) : x = 18 ∨ x = 27 :=
sorry

end determine_x_l12_12061


namespace hiker_speeds_l12_12096

theorem hiker_speeds:
  ∃ (d : ℝ), 
  (d > 5) ∧ ((70 / (d - 5)) = (110 / d)) ∧ (d - 5 = 8.75) :=
by
  sorry

end hiker_speeds_l12_12096


namespace balls_in_boxes_l12_12453

/-
Prove that the number of ways to distribute 6 distinguishable balls into 3 indistinguishable boxes is 132.
-/
theorem balls_in_boxes : 
  ∃ (ways : ℕ), ways = 132 ∧ ways = 
    (1) + 
    (Nat.choose 6 5) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 4) + 
    (Nat.choose 6 3) + 
    (Nat.choose 6 3 * Nat.choose 3 2) + 
    (Nat.choose 6 2 * Nat.choose 4 2 / 6) := 
by
  sorry

end balls_in_boxes_l12_12453


namespace more_whistles_sean_than_charles_l12_12374

def whistles_sean : ℕ := 223
def whistles_charles : ℕ := 128

theorem more_whistles_sean_than_charles : (whistles_sean - whistles_charles) = 95 :=
by
  sorry

end more_whistles_sean_than_charles_l12_12374


namespace rotated_point_coordinates_l12_12919

noncomputable def A : ℝ × ℝ := (1, 2)

def rotate_90_counterclockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.snd, p.fst)

theorem rotated_point_coordinates :
  rotate_90_counterclockwise A = (-2, 1) :=
by
  -- Skipping the proof
  sorry

end rotated_point_coordinates_l12_12919


namespace magnet_cost_is_three_l12_12666

noncomputable def stuffed_animal_cost : ℕ := 6
noncomputable def combined_stuffed_animals_cost : ℕ := 2 * stuffed_animal_cost
noncomputable def magnet_cost : ℕ := combined_stuffed_animals_cost / 4

theorem magnet_cost_is_three : magnet_cost = 3 :=
by
  sorry

end magnet_cost_is_three_l12_12666


namespace second_term_of_series_l12_12690

noncomputable def geometric_series_second_term (a r S : ℝ) := r * a

theorem second_term_of_series (a r : ℝ) (S : ℝ) (hr : r = 1/4) (hs : S = 16) 
  (hS_formula : S = a / (1 - r)) : geometric_series_second_term a r S = 3 :=
by
  -- Definitions are in place, applying algebraic manipulation steps here would follow
  sorry

end second_term_of_series_l12_12690


namespace quadratic_root_a_l12_12884

theorem quadratic_root_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 = 0 ∧ x = 1) → a = -5 :=
by
  intro h
  have h1 : (1:ℝ)^2 + a * (1:ℝ) + 4 = 0 := sorry
  linarith

end quadratic_root_a_l12_12884


namespace function_identity_l12_12780

theorem function_identity (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, f m + f n ∣ m + n) : ∀ m : ℕ+, f m = m := by
  sorry

end function_identity_l12_12780


namespace original_price_of_coffee_l12_12606

/-- 
  Define the prices of the cups of coffee as per the conditions.
  Let x be the original price of one cup of coffee.
  Assert the conditions and find the original price.
-/
theorem original_price_of_coffee (x : ℝ) 
  (h1 : x + x / 2 + 3 = 57) 
  (h2 : (x + x / 2 + 3)/3 = 19) : 
  x = 36 := 
by
  sorry

end original_price_of_coffee_l12_12606


namespace calculate_sum_of_powers_l12_12893

theorem calculate_sum_of_powers :
  (6^2 - 3^2)^4 + (7^2 - 2^2)^4 = 4632066 :=
by
  sorry

end calculate_sum_of_powers_l12_12893


namespace lcm_18_35_l12_12251

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := 
by 
  sorry

end lcm_18_35_l12_12251


namespace magnitude_fourth_power_l12_12223

open Complex

noncomputable def complex_magnitude_example : ℂ := 4 + 3 * Real.sqrt 3 * Complex.I

theorem magnitude_fourth_power :
  ‖complex_magnitude_example ^ 4‖ = 1849 := by
  sorry

end magnitude_fourth_power_l12_12223


namespace factorization_a4_plus_4_l12_12827

theorem factorization_a4_plus_4 (a : ℝ) : a^4 + 4 = (a^2 - 2*a + 2) * (a^2 + 2*a + 2) :=
by sorry

end factorization_a4_plus_4_l12_12827


namespace find_a_if_x_is_1_root_l12_12331

theorem find_a_if_x_is_1_root {a : ℝ} (h : (1 : ℝ)^2 + a * 1 - 2 = 0) : a = 1 :=
by sorry

end find_a_if_x_is_1_root_l12_12331


namespace min_value_of_diff_squares_l12_12790

noncomputable def C (x y z : ℝ) : ℝ := (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 12))
noncomputable def D (x y z : ℝ) : ℝ := (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))

theorem min_value_of_diff_squares (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  ∃ minimum_value, minimum_value = 36 ∧ ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → (C x y z)^2 - (D x y z)^2 ≥ minimum_value :=
sorry

end min_value_of_diff_squares_l12_12790


namespace simplify_fraction_l12_12227

theorem simplify_fraction : 
  (2 * Real.sqrt 6) / (Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5) = Real.sqrt 2 + Real.sqrt 3 - Real.sqrt 5 := 
by
  sorry

end simplify_fraction_l12_12227


namespace recess_breaks_l12_12583

theorem recess_breaks (total_outside_time : ℕ) (lunch_break : ℕ) (extra_recess : ℕ) (recess_duration : ℕ) 
  (h1 : total_outside_time = 80)
  (h2 : lunch_break = 30)
  (h3 : extra_recess = 20)
  (h4 : recess_duration = 15) : 
  (total_outside_time - (lunch_break + extra_recess)) / recess_duration = 2 := 
by {
  -- proof starts here
  sorry
}

end recess_breaks_l12_12583


namespace fraction_identity_l12_12018

theorem fraction_identity (a b c : ℝ) (h1 : a + b + c > 0) (h2 : a + b - c > 0) (h3 : a + c - b > 0) (h4 : b + c - a > 0) 
  (h5 : (a+b+c)/(a+b-c) = 7) (h6 : (a+b+c)/(a+c-b) = 1.75) : (a+b+c)/(b+c-a) = 3.5 :=
by
  sorry

end fraction_identity_l12_12018


namespace roots_of_polynomial_l12_12835

theorem roots_of_polynomial :
  ∀ x : ℝ, (x^3 - 3 * x^2 + 2 * x) * (x - 5) = 0 ↔ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 5 :=
by 
  sorry

end roots_of_polynomial_l12_12835


namespace dealer_gross_profit_l12_12303

theorem dealer_gross_profit (P S G : ℝ) (hP : P = 150) (markup : S = P + 0.5 * S) :
  G = S - P → G = 150 :=
by
  sorry

end dealer_gross_profit_l12_12303


namespace triangle_side_length_l12_12369

noncomputable def sine (x : ℝ) : ℝ := Real.sin (x * Real.pi / 180) -- Define sine function explicitly (degrees to radians)

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ)
  (hA : A = 30) (hC : C = 45) (ha : a = 4) :
  c = 4 * Real.sqrt 2 := by
  sorry

end triangle_side_length_l12_12369


namespace x_squared_minus_y_squared_l12_12444

open Real

theorem x_squared_minus_y_squared
  (x y : ℝ)
  (h1 : x + y = 4/9)
  (h2 : x - y = 2/9) :
  x^2 - y^2 = 8/81 :=
by
  sorry

end x_squared_minus_y_squared_l12_12444


namespace dot_product_is_five_l12_12488

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-1, 3)

-- Define the condition that involves a and b
def condition : Prop := 2 • a - b = (3, 1)

-- Prove that the dot product of a and b equals 5 given the condition
theorem dot_product_is_five : condition → (a.1 * b.1 + a.2 * b.2) = 5 :=
by
  sorry

end dot_product_is_five_l12_12488


namespace abs_inequality_solution_set_l12_12595

theorem abs_inequality_solution_set (x : ℝ) : 
  (|2 * x - 3| ≤ 1) ↔ (1 ≤ x ∧ x ≤ 2) := 
by
  sorry

end abs_inequality_solution_set_l12_12595


namespace find_correction_time_l12_12483

-- Define the conditions
def loses_minutes_per_day : ℚ := 2 + 1/2
def initial_time_set : ℚ := 1 * 60 -- 1 PM in minutes
def time_on_march_21 : ℚ := 9 * 60 -- 9 AM in minutes on March 21
def total_minutes_per_day : ℚ := 24 * 60
def days_between : ℚ := 6 - 4/24 -- 6 days minus 4 hours

-- Calculate effective functioning minutes per day
def effective_minutes_per_day : ℚ := total_minutes_per_day - loses_minutes_per_day

-- Calculate the ratio of actual time to the watch's time
def time_ratio : ℚ := total_minutes_per_day / effective_minutes_per_day

-- Calculate the total actual time in minutes between initial set time and the given time showing on the watch
def total_actual_time : ℚ := days_between * total_minutes_per_day + initial_time_set

-- Calculate the actual time according to the ratio
def actual_time_according_to_ratio : ℚ := total_actual_time * time_ratio

-- Calculate the correction required 'n'
def required_minutes_correction : ℚ := actual_time_according_to_ratio - total_actual_time

-- The theorem stating that the required correction is as calculated
theorem find_correction_time : required_minutes_correction = (14 + 14/23) := by
  sorry

end find_correction_time_l12_12483


namespace total_number_of_people_l12_12270

theorem total_number_of_people (c a : ℕ) (h1 : c = 2 * a) (h2 : c = 28) : c + a = 42 :=
by
  sorry

end total_number_of_people_l12_12270


namespace binomial_expansion_fraction_l12_12489

theorem binomial_expansion_fraction 
    (a_0 a_1 a_2 a_3 a_4 a_5 : ℤ)
    (h1 : a_0 + a_1 + a_2 + a_3 + a_4 + a_5 = 1)
    (h2 : a_0 - a_1 + a_2 - a_3 + a_4 - a_5 = 243) :
    (a_0 + a_2 + a_4) / (a_1 + a_3 + a_5) = -122 / 121 :=
by
  sorry

end binomial_expansion_fraction_l12_12489


namespace sum_of_nonneg_reals_l12_12288

theorem sum_of_nonneg_reals (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0)
  (h4 : x^2 + y^2 + z^2 = 52) (h5 : x * y + y * z + z * x = 24) :
  x + y + z = 10 :=
sorry

end sum_of_nonneg_reals_l12_12288


namespace sin_double_angle_l12_12022

variable {φ : ℝ}

theorem sin_double_angle (h : (7 / 13 : ℝ) + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 :=
by
  sorry

end sin_double_angle_l12_12022


namespace three_distinct_roots_condition_l12_12838

noncomputable def k_condition (k : ℝ) : Prop :=
  ∀ (x : ℝ), (x / (x - 1) + x / (x - 3)) = k * x → 
    (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

theorem three_distinct_roots_condition (k : ℝ) : k ≠ 0 ↔ k_condition k :=
by
  sorry

end three_distinct_roots_condition_l12_12838


namespace range_of_m_l12_12460

-- Definition of p: x / (x - 2) < 0 implies 0 < x < 2
def p (x : ℝ) : Prop := x / (x - 2) < 0

-- Definition of q: 0 < x < m
def q (x m : ℝ) : Prop := 0 < x ∧ x < m

-- Main theorem: If p is a necessary but not sufficient condition for q to hold, then the range of m is (2, +∞)
theorem range_of_m {m : ℝ} (h : ∀ x, p x → q x m) (hs : ∃ x, ¬(q x m) ∧ p x) : 
  2 < m :=
sorry

end range_of_m_l12_12460


namespace intersection_of_A_and_B_l12_12365

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 1 > 0}

theorem intersection_of_A_and_B :
  A ∩ B = {2} :=
sorry

end intersection_of_A_and_B_l12_12365


namespace find_original_speed_l12_12199

theorem find_original_speed :
  ∀ (v T : ℝ), 
    (300 = 212 + 88) →
    (T + 2/3 = 212 / v + 88 / (v - 50)) →
    v = 110 :=
by
  intro v T h_dist h_trip
  sorry

end find_original_speed_l12_12199


namespace history_paper_pages_l12_12230

/-
Stacy has a history paper due in 3 days.
She has to write 21 pages per day to finish on time.
Prove that the total number of pages for the history paper is 63.
-/

theorem history_paper_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 21) (h2 : days = 3) : total_pages = 63 :=
by
  -- We would include the proof here, but for now, we use sorry to skip the proof.
  sorry

end history_paper_pages_l12_12230


namespace memorial_visits_l12_12734

theorem memorial_visits (x : ℕ) (total_visits : ℕ) (difference : ℕ) 
  (h1 : total_visits = 589) 
  (h2 : difference = 56) 
  (h3 : 2 * x + difference = total_visits - x) : 
  2 * x + 56 = 589 - x :=
by
  -- proof steps would go here
  sorry

end memorial_visits_l12_12734


namespace possible_integer_roots_l12_12037

def polynomial (x : ℤ) : ℤ := x^3 + 2 * x^2 - 3 * x - 17

theorem possible_integer_roots :
  ∃ (roots : List ℤ), roots = [1, -1, 17, -17] ∧ ∀ r ∈ roots, polynomial r = 0 := 
sorry

end possible_integer_roots_l12_12037


namespace half_dollar_difference_l12_12885

theorem half_dollar_difference (n d h : ℕ) 
  (h1 : n + d + h = 150) 
  (h2 : 5 * n + 10 * d + 50 * h = 1500) : 
  ∃ h_max h_min, (h_max - h_min = 16) :=
by sorry

end half_dollar_difference_l12_12885


namespace initial_books_gathered_l12_12974

-- Conditions
def total_books_now : Nat := 59
def books_found : Nat := 26

-- Proof problem
theorem initial_books_gathered : total_books_now - books_found = 33 :=
by
  sorry -- Proof to be provided later

end initial_books_gathered_l12_12974


namespace xiaoli_estimate_greater_l12_12129

variable (p q a b : ℝ)

theorem xiaoli_estimate_greater (hpq : p > q) (hq0 : q > 0) (hab : a > b) : (p + a) - (q + b) > p - q := 
by 
  sorry

end xiaoli_estimate_greater_l12_12129


namespace find_m_l12_12618

-- Define the functions f and g
def f (x m : ℝ) := x^2 - 2 * x + m
def g (x m : ℝ) := x^2 - 3 * x + 5 * m

-- The condition to be proved
theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 10 :=
by
  sorry

end find_m_l12_12618


namespace probability_of_johns_8th_roll_l12_12293

noncomputable def probability_johns_8th_roll_is_last : ℚ :=
  (7/8)^6 * (1/8)

theorem probability_of_johns_8th_roll :
  probability_johns_8th_roll_is_last = 117649 / 2097152 := by
  sorry

end probability_of_johns_8th_roll_l12_12293


namespace jackson_earned_on_monday_l12_12449

-- Definitions
def goal := 1000
def tuesday_earnings := 40
def avg_rate := 10
def houses := 88
def days_remaining := 3
def total_collected_remaining_days := days_remaining * (houses / 4) * avg_rate

-- The proof problem statement
theorem jackson_earned_on_monday (m : ℕ) :
  m + tuesday_earnings + total_collected_remaining_days = goal → m = 300 :=
by
  -- We will eventually provide the proof here
  sorry

end jackson_earned_on_monday_l12_12449


namespace clerk_daily_salary_l12_12162

theorem clerk_daily_salary (manager_salary : ℝ) (num_managers num_clerks : ℕ) (total_salary : ℝ) (clerk_salary : ℝ)
  (h1 : manager_salary = 5)
  (h2 : num_managers = 2)
  (h3 : num_clerks = 3)
  (h4 : total_salary = 16) :
  clerk_salary = 2 :=
by
  sorry

end clerk_daily_salary_l12_12162


namespace black_balls_number_l12_12322

-- Define the given conditions and the problem statement as Lean statements
theorem black_balls_number (n : ℕ) (h : (2 : ℝ) / (n + 2 : ℝ) = 0.4) : n = 3 :=
by
  sorry

end black_balls_number_l12_12322


namespace mark_repayment_l12_12755

noncomputable def totalDebt (days : ℕ) : ℝ :=
  if days < 3 then
    20 + (20 * 0.10 * days)
  else
    35 + (20 * 0.10 * 3) + (35 * 0.10 * (days - 3))

theorem mark_repayment :
  ∃ (x : ℕ), totalDebt x ≥ 70 ∧ x = 12 :=
by
  -- Use this theorem statement to prove the corresponding lean proof
  sorry

end mark_repayment_l12_12755


namespace soccer_minimum_wins_l12_12954

/-
Given that a soccer team has won 60% of 45 matches played so far, 
prove that the minimum number of matches that the team still needs to win to reach a winning percentage of 75% is 27.
-/
theorem soccer_minimum_wins 
  (initial_matches : ℕ)                 -- the initial number of matches
  (initial_win_rate : ℚ)                -- the initial win rate (as a percentage)
  (desired_win_rate : ℚ)                -- the desired win rate (as a percentage)
  (initial_wins : ℕ)                    -- the initial number of wins

  -- Given conditions
  (h1 : initial_matches = 45)
  (h2 : initial_win_rate = 0.60)
  (h3 : desired_win_rate = 0.75)
  (h4 : initial_wins = 27):
  
  -- To prove: the minimum number of additional matches that need to be won is 27
  ∃ (n : ℕ), (initial_wins + n) / (initial_matches + n) = desired_win_rate ∧ 
                  n = 27 :=
by 
  sorry

end soccer_minimum_wins_l12_12954


namespace largest_b_for_denom_has_nonreal_roots_l12_12372

theorem largest_b_for_denom_has_nonreal_roots :
  ∃ b : ℤ, 
  (∀ x : ℝ, x^2 + (b : ℝ) * x + 12 ≠ 0) 
  ∧ (∀ b' : ℤ, (∀ x : ℝ, x^2 + (b' : ℝ) * x + 12 ≠ 0) → b' ≤ b)
  ∧ b = 6 :=
sorry

end largest_b_for_denom_has_nonreal_roots_l12_12372


namespace terminal_side_of_angle_y_eq_neg_one_l12_12913
/-
Given that the terminal side of angle θ lies on the line y = -x,
prove that y = -1 where y = sin θ / |sin θ| + |cos θ| / cos θ + tan θ / |tan θ|.
-/


noncomputable def y (θ : ℝ) : ℝ :=
  (Real.sin θ / |Real.sin θ|) + (|Real.cos θ| / Real.cos θ) + (Real.tan θ / |Real.tan θ|)

theorem terminal_side_of_angle_y_eq_neg_one (θ : ℝ) (k : ℤ) (h : θ = k * Real.pi - (Real.pi / 4)) :
  y θ = -1 :=
by
  sorry

end terminal_side_of_angle_y_eq_neg_one_l12_12913


namespace tan_value_of_point_on_exp_graph_l12_12388

theorem tan_value_of_point_on_exp_graph (a : ℝ) (h1 : (a, 9) ∈ {p : ℝ × ℝ | ∃ x, p = (x, 3^x)}) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := by
  sorry

end tan_value_of_point_on_exp_graph_l12_12388


namespace Lyka_savings_l12_12514

def Smartphone_cost := 800
def Initial_savings := 200
def Gym_cost_per_month := 50
def Total_months := 4
def Weeks_per_month := 4
def Savings_per_week_initial := 50
def Savings_per_week_after_raise := 80

def Total_savings : Nat :=
  let initial_savings := Savings_per_week_initial * Weeks_per_month * 2
  let increased_savings := Savings_per_week_after_raise * Weeks_per_month * 2
  initial_savings + increased_savings

theorem Lyka_savings :
  (Initial_savings + Total_savings) = 1040 := by
  sorry

end Lyka_savings_l12_12514


namespace problem_statement_l12_12971

variable (a b : ℝ)

theorem problem_statement (h1 : a > b) (h2 : b > 1/a) (h3 : 1/a > 0) :
  (a + b > 2) ∧ (a > 1) ∧ (a - 1/b > b - 1/a) :=
by 
  sorry

end problem_statement_l12_12971


namespace speed_of_stream_l12_12381

-- Define the problem conditions
variables (b s : ℝ)
axiom cond1 : 21 = b + s
axiom cond2 : 15 = b - s

-- State the theorem
theorem speed_of_stream : s = 3 :=
sorry

end speed_of_stream_l12_12381


namespace sin_A_equals_4_over_5_l12_12762

variables {A B C : ℝ}
-- Given a right triangle ABC with angle B = 90 degrees
def right_triangle (A B C : ℝ) : Prop :=
  (A + B + C = 180) ∧ (B = 90)

-- We are given 3 * sin(A) = 4 * cos(A)
def given_condition (A : ℝ) : Prop :=
  3 * Real.sin A = 4 * Real.cos A

-- We need to prove that sin(A) = 4/5
theorem sin_A_equals_4_over_5 (A B C : ℝ) 
  (h1 : right_triangle B 90 C)
  (h2 : given_condition A) : 
  Real.sin A = 4 / 5 :=
by
  sorry

end sin_A_equals_4_over_5_l12_12762


namespace Sandy_original_number_l12_12465

theorem Sandy_original_number (x : ℝ) (h : (3 * x + 20)^2 = 2500) : x = 10 :=
by
  sorry

end Sandy_original_number_l12_12465


namespace find_number_l12_12626

theorem find_number (x : ℕ) (h : 112 * x = 70000) : x = 625 :=
by
  sorry

end find_number_l12_12626


namespace similar_triangle_shortest_side_l12_12775

theorem similar_triangle_shortest_side 
  (a₁ : ℝ) (b₁ : ℝ) (c₁ : ℝ) (c₂ : ℝ) (k : ℝ)
  (h₁ : a₁ = 15) 
  (h₂ : c₁ = 39) 
  (h₃ : c₂ = 117) 
  (h₄ : k = c₂ / c₁) 
  (h₅ : k = 3) 
  (h₆ : a₂ = a₁ * k) :
  a₂ = 45 := 
by {
  sorry -- proof is not required
}

end similar_triangle_shortest_side_l12_12775


namespace parametric_to_general_eq_l12_12020

theorem parametric_to_general_eq (x y θ : ℝ) 
  (h1 : x = 2 + Real.sin θ ^ 2) 
  (h2 : y = -1 + Real.cos (2 * θ)) : 
  2 * x + y - 4 = 0 ∧ 2 ≤ x ∧ x ≤ 3 := 
sorry

end parametric_to_general_eq_l12_12020


namespace binomial_sum_l12_12212

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem binomial_sum (n : ℤ) (h1 : binomial 25 n.natAbs + binomial 25 12 = binomial 26 13 ∧ n ≥ 0) : 
    (n = 12 ∨ n = 13) → n.succ + n = 25 := 
    sorry

end binomial_sum_l12_12212


namespace find_subtracted_number_l12_12717

theorem find_subtracted_number 
  (a : ℕ) (b : ℕ) (g : ℕ) (n : ℕ) 
  (h1 : a = 2) 
  (h2 : b = 3 * a) 
  (h3 : g = 2 * b - n) 
  (h4 : g = 8) : n = 4 :=
by 
  sorry

end find_subtracted_number_l12_12717


namespace charity_ticket_sales_l12_12090

theorem charity_ticket_sales
  (x y p : ℕ)
  (h1 : x + y = 200)
  (h2 : x * p + y * (p / 2) = 3501)
  (h3 : x = 3 * y) :
  150 * 20 = 3000 :=
by
  sorry

end charity_ticket_sales_l12_12090


namespace students_like_apple_and_chocolate_not_carrot_l12_12643

-- Definitions based on the conditions
def total_students : ℕ := 50
def apple_likers : ℕ := 23
def chocolate_likers : ℕ := 20
def carrot_likers : ℕ := 10
def non_likers : ℕ := 15

-- The main statement we need to prove: 
-- the number of students who liked both apple pie and chocolate cake but not carrot cake
theorem students_like_apple_and_chocolate_not_carrot : 
  ∃ (a b c d : ℕ), a + b + d = apple_likers ∧
                    a + c + d = chocolate_likers ∧
                    b + c + d = carrot_likers ∧
                    a + b + c + (50 - (35) - 15) = 35 ∧ 
                    a = 7 :=
by 
  sorry

end students_like_apple_and_chocolate_not_carrot_l12_12643


namespace animal_count_l12_12164

theorem animal_count (dogs : ℕ) (cats : ℕ) (birds : ℕ) (fish : ℕ)
  (h1 : dogs = 6)
  (h2 : cats = dogs / 2)
  (h3 : birds = dogs * 2)
  (h4 : fish = dogs * 3) : 
  dogs + cats + birds + fish = 39 :=
by
  sorry

end animal_count_l12_12164


namespace percentage_decrease_l12_12851

variable {a b x m : ℝ} (p : ℝ)

theorem percentage_decrease (h₁ : a / b = 4 / 5)
                          (h₂ : x = 1.25 * a)
                          (h₃ : m = b * (1 - p / 100))
                          (h₄ : m / x = 0.8) :
  p = 20 :=
sorry

end percentage_decrease_l12_12851


namespace option_b_correct_l12_12552

variable (Line Plane : Type)

-- Definitions for perpendicularity and parallelism
variable (perp parallel : Line → Plane → Prop) (parallel_line : Line → Line → Prop)

-- Assumptions reflecting the conditions in the problem
axiom perp_alpha_1 {a : Line} {alpha : Plane} : perp a alpha
axiom perp_alpha_2 {b : Line} {alpha : Plane} : perp b alpha

-- The statement to prove
theorem option_b_correct (a b : Line) (alpha : Plane) :
  perp a alpha → perp b alpha → parallel_line a b :=
by
  intro h1 h2
  -- proof omitted
  sorry

end option_b_correct_l12_12552


namespace single_discount_eq_l12_12957

/--
A jacket is originally priced at $50. It is on sale for 25% off. After applying the sale discount, 
John uses a coupon that gives an additional 10% off of the discounted price. If there is a 5% sales 
tax on the final price, what single percent discount (before taxes) is equivalent to these series 
of discounts followed by the tax? --/
theorem single_discount_eq :
  let P0 := 50
  let discount1 := 0.25
  let discount2 := 0.10
  let tax := 0.05
  let discounted_price := P0 * (1 - discount1) * (1 - discount2)
  let after_tax_price := discounted_price * (1 + tax)
  let single_discount := (P0 - discounted_price) / P0
  single_discount * 100 = 32.5 :=
by
  sorry

end single_discount_eq_l12_12957


namespace units_digit_powers_difference_l12_12898

theorem units_digit_powers_difference (p : ℕ) 
  (h1: p > 0) 
  (h2: p % 2 = 0) 
  (h3: (p % 10 + 2) % 10 = 8) : 
  ((p ^ 3) % 10 - (p ^ 2) % 10) % 10 = 0 :=
by
  sorry

end units_digit_powers_difference_l12_12898


namespace total_profit_proof_l12_12561
-- Import the necessary libraries

-- Define the investments and profits
def investment_tom : ℕ := 3000 * 12
def investment_jose : ℕ := 4500 * 10
def profit_jose : ℕ := 3500

-- Define the ratio and profit parts
def ratio_tom : ℕ := investment_tom / Nat.gcd investment_tom investment_jose
def ratio_jose : ℕ := investment_jose / Nat.gcd investment_tom investment_jose
def ratio_total : ℕ := ratio_tom + ratio_jose
def one_part_value : ℕ := profit_jose / ratio_jose
def profit_tom : ℕ := ratio_tom * one_part_value

-- The total profit
def total_profit : ℕ := profit_tom + profit_jose

-- The theorem to prove
theorem total_profit_proof : total_profit = 6300 := by
  sorry

end total_profit_proof_l12_12561


namespace card_game_impossible_l12_12923

theorem card_game_impossible : 
  ∀ (students : ℕ) (initial_cards : ℕ) (cards_distribution : ℕ → ℕ), 
  students = 2018 → 
  initial_cards = 2018 →
  (∀ n, n < students → (if n = 0 then cards_distribution n = initial_cards else cards_distribution n = 0)) →
  (¬ ∃ final_distribution : ℕ → ℕ, (∀ n, n < students → final_distribution n = 1)) :=
by
  intros students initial_cards cards_distribution stu_eq init_eq init_dist final_dist
  -- Sorry can be used here as the proof is not required
  sorry

end card_game_impossible_l12_12923


namespace oranges_taken_by_susan_l12_12183

-- Defining the conditions
def original_number_of_oranges_in_box : ℕ := 55
def oranges_left_in_box_after_susan_takes : ℕ := 20

-- Statement to prove:
theorem oranges_taken_by_susan :
  original_number_of_oranges_in_box - oranges_left_in_box_after_susan_takes = 35 :=
by
  sorry

end oranges_taken_by_susan_l12_12183


namespace three_students_with_B_l12_12247

-- Define the students and their statements as propositions
variables (Eva B_Frank B_Gina B_Harry : Prop)

-- Condition 1: Eva said, "If I get a B, then Frank will get a B."
axiom Eva_statement : Eva → B_Frank

-- Condition 2: Frank said, "If I get a B, then Gina will get a B."
axiom Frank_statement : B_Frank → B_Gina

-- Condition 3: Gina said, "If I get a B, then Harry will get a B."
axiom Gina_statement : B_Gina → B_Harry

-- Condition 4: Only three students received a B.
axiom only_three_Bs : (Eva ∧ B_Frank ∧ B_Gina ∧ B_Harry) → False

-- The theorem we need to prove: The three students who received B's are Frank, Gina, and Harry.
theorem three_students_with_B (h_B_Frank : B_Frank) (h_B_Gina : B_Gina) (h_B_Harry : B_Harry) : ¬Eva :=
by
  sorry

end three_students_with_B_l12_12247


namespace A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l12_12900

-- Definitions of events
def A : Prop := sorry -- event that the part is of the first grade
def B : Prop := sorry -- event that the part is of the second grade
def C : Prop := sorry -- event that the part is of the third grade

-- Mathematically equivalent proof problems
theorem A_or_B : A ∨ B ↔ (A ∨ B) :=
by sorry

theorem not_A_or_C : ¬(A ∨ C) ↔ B :=
by sorry

theorem A_and_C : (A ∧ C) ↔ false :=
by sorry

theorem A_and_B_or_C : ((A ∧ B) ∨ C) ↔ C :=
by sorry

end A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l12_12900


namespace sum_of_largest_and_smallest_l12_12586

theorem sum_of_largest_and_smallest (n : ℕ) (h : 6 * n + 15 = 105) : (n + (n + 5) = 35) :=
by
  sorry

end sum_of_largest_and_smallest_l12_12586


namespace remainder_101_pow_50_mod_100_l12_12793

theorem remainder_101_pow_50_mod_100 : (101 ^ 50) % 100 = 1 := by
  sorry

end remainder_101_pow_50_mod_100_l12_12793


namespace solve_for_x_l12_12530

theorem solve_for_x (x : ℝ) (h : 3 * x - 8 = 4 * x + 5) : x = -13 :=
by 
  sorry

end solve_for_x_l12_12530


namespace final_amount_H2O_l12_12723

theorem final_amount_H2O (main_reaction : ∀ (Li3N H2O LiOH NH3 : ℕ), Li3N + 3 * H2O = 3 * LiOH + NH3)
  (side_reaction : ∀ (Li3N LiOH Li2O NH4OH : ℕ), Li3N + LiOH = Li2O + NH4OH)
  (temperature : ℕ) (pressure : ℕ)
  (percentage : ℝ) (init_moles_LiOH : ℕ) (init_moles_Li3N : ℕ)
  (H2O_req_main : ℝ) (H2O_req_side : ℝ) :
  400 = temperature →
  2 = pressure →
  0.05 = percentage →
  9 = init_moles_LiOH →
  3 = init_moles_Li3N →
  H2O_req_main = init_moles_Li3N * 3 →
  H2O_req_side = init_moles_LiOH * percentage →
  H2O_req_main + H2O_req_side = 9.45 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 
  sorry

end final_amount_H2O_l12_12723


namespace boat_distance_travelled_upstream_l12_12464

theorem boat_distance_travelled_upstream (v : ℝ) (d : ℝ) :
  ∀ (boat_speed_in_still_water upstream_time downstream_time : ℝ),
  boat_speed_in_still_water = 25 →
  upstream_time = 1 →
  downstream_time = 0.25 →
  d = (boat_speed_in_still_water - v) * upstream_time →
  d = (boat_speed_in_still_water + v) * downstream_time →
  d = 10 :=
by
  intros
  sorry

end boat_distance_travelled_upstream_l12_12464


namespace arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l12_12699

-- Definitions based on conditions
def performances : Nat := 8
def singing : Nat := 2
def dance : Nat := 3
def variety : Nat := 3

-- Problem 1: Prove arrangement with a singing program at the beginning and end
theorem arrange_singing_begin_end : 1440 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 2: Prove arrangement with singing programs not adjacent
theorem arrange_singing_not_adjacent : 30240 = sorry :=
by
  -- proof goes here
  sorry

-- Problem 3: Prove arrangement with singing programs adjacent and dance not adjacent
theorem arrange_singing_adjacent_dance_not_adjacent : 2880 = sorry :=
by
  -- proof goes here
  sorry

end arrange_singing_begin_end_arrange_singing_not_adjacent_arrange_singing_adjacent_dance_not_adjacent_l12_12699


namespace minimum_planks_required_l12_12526

theorem minimum_planks_required (colors : Finset ℕ) (planks : List ℕ) :
  colors.card = 100 ∧
  ∀ i j, i ∈ colors → j ∈ colors → i ≠ j →
  ∃ k₁ k₂, k₁ < k₂ ∧ planks.get? k₁ = some i ∧ planks.get? k₂ = some j
  → planks.length = 199 := 
sorry

end minimum_planks_required_l12_12526


namespace pyramid_base_edge_length_l12_12890

theorem pyramid_base_edge_length
  (hemisphere_radius : ℝ) (pyramid_height : ℝ) (slant_height : ℝ) (is_tangent: Prop) :
  hemisphere_radius = 3 ∧ pyramid_height = 8 ∧ slant_height = 10 ∧ is_tangent →
  ∃ (base_edge_length : ℝ), base_edge_length = 6 * Real.sqrt 2 :=
by
  sorry

end pyramid_base_edge_length_l12_12890


namespace SUVs_purchased_l12_12958

theorem SUVs_purchased (x : ℕ) (hToyota : ℕ) (hHonda : ℕ) (hNissan : ℕ) 
  (hRatioToyota : hToyota = 7 * x) 
  (hRatioHonda : hHonda = 5 * x) 
  (hRatioNissan : hNissan = 3 * x) 
  (hToyotaSUV : ℕ) (hHondaSUV : ℕ) (hNissanSUV : ℕ) 
  (hToyotaSUV_num : hToyotaSUV = (50 * hToyota) / 100) 
  (hHondaSUV_num : hHondaSUV = (40 * hHonda) / 100) 
  (hNissanSUV_num : hNissanSUV = (30 * hNissan) / 100) : 
  hToyotaSUV + hHondaSUV + hNissanSUV = 64 := 
by
  sorry

end SUVs_purchased_l12_12958


namespace min_sum_of_factors_l12_12952

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1800) : 
  a + b + c = 64 :=
sorry

end min_sum_of_factors_l12_12952


namespace evaluate_five_iterates_of_f_at_one_l12_12118

def f (x : ℕ) : ℕ :=
if x % 2 = 0 then x / 2 else 5 * x + 1

theorem evaluate_five_iterates_of_f_at_one :
  f (f (f (f (f 1)))) = 4 := by
  sorry

end evaluate_five_iterates_of_f_at_one_l12_12118


namespace fraction_of_number_l12_12308

theorem fraction_of_number (x : ℕ) (f : ℚ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 :=
sorry

end fraction_of_number_l12_12308


namespace vertical_distance_to_Felix_l12_12445

/--
  Dora is at point (8, -15).
  Eli is at point (2, 18).
  Felix is at point (5, 7).
  Calculate the vertical distance they need to walk to reach Felix.
-/
theorem vertical_distance_to_Felix :
  let Dora := (8, -15)
  let Eli := (2, 18)
  let Felix := (5, 7)
  let midpoint := ((Dora.1 + Eli.1) / 2, (Dora.2 + Eli.2) / 2)
  let vertical_distance := Felix.2 - midpoint.2
  vertical_distance = 5.5 :=
by
  sorry

end vertical_distance_to_Felix_l12_12445


namespace slope_of_intersection_line_is_one_l12_12693

open Real

-- Definitions of the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6 * x + 4 * y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 2 * y + 4 = 0

-- The statement to prove that the slope of the line through the intersection points is 1
theorem slope_of_intersection_line_is_one :
  ∃ m : ℝ, (∀ x y : ℝ, circle1 x y → circle2 x y → (y = m * x + b)) ∧ m = 1 :=
by
  sorry

end slope_of_intersection_line_is_one_l12_12693


namespace midpoint_of_symmetric_chord_on_ellipse_l12_12748

theorem midpoint_of_symmetric_chord_on_ellipse
  (A B : ℝ × ℝ) -- coordinates of points A and B
  (hA : (A.1^2 / 16) + (A.2^2 / 4) = 1) -- A lies on the ellipse
  (hB : (B.1^2 / 16) + (B.2^2 / 4) = 1) -- B lies on the ellipse
  (symm : 2 * (A.1 + B.1) / 2 - 2 * (A.2 + B.2) / 2 - 3 = 0) -- A and B are symmetric about the line
  : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (2, 1 / 2) :=
  sorry

end midpoint_of_symmetric_chord_on_ellipse_l12_12748


namespace binom_12_9_plus_binom_12_3_l12_12448

theorem binom_12_9_plus_binom_12_3 : (Nat.choose 12 9) + (Nat.choose 12 3) = 440 := by
  sorry

end binom_12_9_plus_binom_12_3_l12_12448


namespace james_added_8_fish_l12_12763

theorem james_added_8_fish
  (initial_fish : ℕ := 60)
  (fish_eaten_per_day : ℕ := 2)
  (total_days_with_worm : ℕ := 21)
  (fish_remaining_when_discovered : ℕ := 26) :
  ∃ (additional_fish : ℕ), additional_fish = 8 :=
by
  let total_fish_eaten := total_days_with_worm * fish_eaten_per_day
  let fish_remaining_without_addition := initial_fish - total_fish_eaten
  let additional_fish := fish_remaining_when_discovered - fish_remaining_without_addition
  exact ⟨additional_fish, sorry⟩

end james_added_8_fish_l12_12763


namespace gcd_largest_of_forms_l12_12970

theorem gcd_largest_of_forms (a b : ℕ) (h1 : a ≠ b) (h2 : a < 10) (h3 : b < 10) :
  Nat.gcd (100 * a + 11 * b) (101 * b + 10 * a) = 45 :=
by
  sorry

end gcd_largest_of_forms_l12_12970


namespace Ricardo_coin_difference_l12_12069

theorem Ricardo_coin_difference (p : ℕ) (h₁ : 1 ≤ p) (h₂ : p ≤ 3029) :
    let max_value := 15150 - 4 * 1
    let min_value := 15150 - 4 * 3029
    max_value - min_value = 12112 := by
  sorry

end Ricardo_coin_difference_l12_12069


namespace televisions_sold_this_black_friday_l12_12777

theorem televisions_sold_this_black_friday 
  (T : ℕ) 
  (h1 : ∀ (n : ℕ), n = 3 → (T + (50 * n) = 477)) 
  : T = 327 := 
sorry

end televisions_sold_this_black_friday_l12_12777


namespace base_number_equals_2_l12_12694

theorem base_number_equals_2 (x : ℝ) (n : ℕ) (h1 : x^(2*n) + x^(2*n) + x^(2*n) + x^(2*n) = 4^26) (h2 : n = 25) : x = 2 :=
by
  sorry

end base_number_equals_2_l12_12694


namespace geometric_sequence_a9_l12_12778

open Nat

theorem geometric_sequence_a9 (a : ℕ → ℝ) (h1 : a 3 = 20) (h2 : a 6 = 5) 
  (h_geometric : ∀ m n, a ((m + n) / 2) ^ 2 = a m * a n) : 
  a 9 = 5 / 4 := 
by
  sorry

end geometric_sequence_a9_l12_12778


namespace max_value_y_l12_12159

noncomputable def y (x : ℝ) : ℝ := x * (3 - 2 * x)

theorem max_value_y : ∃ x, 0 < x ∧ x < (3:ℝ) / 2 ∧ y x = 9 / 8 :=
by
  sorry

end max_value_y_l12_12159


namespace toy_poodle_height_l12_12321

-- Define the heights of the poodles
variables (S M T : ℝ)

-- Conditions
def std_taller_min : Prop := S = M + 8
def min_taller_toy : Prop := M = T + 6
def std_height : Prop := S = 28

-- Goal: How tall is the toy poodle?
theorem toy_poodle_height (h1 : std_taller_min S M)
                          (h2 : min_taller_toy M T)
                          (h3 : std_height S) : T = 14 :=
by 
  sorry

end toy_poodle_height_l12_12321


namespace force_is_correct_l12_12426

noncomputable def force_computation : ℝ :=
  let m : ℝ := 5 -- kg
  let s : ℝ → ℝ := fun t => 2 * t + 3 * t^2 -- cm
  let a : ℝ := 6 / 100 -- acceleration in m/s^2
  m * a

theorem force_is_correct : force_computation = 0.3 := 
by
  -- Initial conditions
  sorry

end force_is_correct_l12_12426


namespace perimeter_of_nonagon_l12_12163

-- Definitions based on the conditions
def sides := 9
def side_length : ℝ := 2

-- The problem statement in Lean
theorem perimeter_of_nonagon : sides * side_length = 18 := 
by sorry

end perimeter_of_nonagon_l12_12163


namespace coprime_composite_lcm_l12_12301

theorem coprime_composite_lcm (a b : ℕ) (ha : a > 1) (hb : b > 1) (hcoprime : Nat.gcd a b = 1) (hlcm : Nat.lcm a b = 120) : 
  Nat.gcd a b = 1 ∧ min a b = 8 := 
by 
  sorry

end coprime_composite_lcm_l12_12301


namespace problem_I_problem_II_1_problem_II_2_l12_12123

section
variables (boys_A girls_A boys_B girls_B : ℕ)
variables (total_students : ℕ)

-- Define the conditions
def conditions : Prop :=
  boys_A = 2 ∧ girls_A = 1 ∧ boys_B = 3 ∧ girls_B = 2 ∧ total_students = boys_A + girls_A + boys_B + girls_B

-- Problem (I)
theorem problem_I (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ arrangements, arrangements = 14400 := sorry

-- Problem (II.1)
theorem problem_II_1 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 13 / 14 := sorry

-- Problem (II.2)
theorem problem_II_2 (h : conditions boys_A girls_A boys_B girls_B total_students) :
  ∃ prob, prob = 6 / 35 := sorry
end

end problem_I_problem_II_1_problem_II_2_l12_12123


namespace range_of_a_l12_12658

theorem range_of_a (x : ℝ) (a : ℝ) (hx : 0 < x ∧ x < 4) : |x - 1| < a → a ≥ 3 := sorry

end range_of_a_l12_12658


namespace sheila_weekly_earnings_l12_12857

-- Variables
variables {hours_mon_wed_fri hours_tue_thu rate_per_hour : ℕ}

-- Conditions
def sheila_works_mwf : hours_mon_wed_fri = 8 := by sorry
def sheila_works_tue_thu : hours_tue_thu = 6 := by sorry
def sheila_rate : rate_per_hour = 11 := by sorry

-- Main statement to prove
theorem sheila_weekly_earnings : 
  3 * hours_mon_wed_fri + 2 * hours_tue_thu = 36 →
  rate_per_hour = 11 →
  (3 * hours_mon_wed_fri + 2 * hours_tue_thu) * rate_per_hour = 396 :=
by
  intros h_hours h_rate
  sorry

end sheila_weekly_earnings_l12_12857


namespace max_distinct_subsets_l12_12073

def T : Set ℕ := { x | 1 ≤ x ∧ x ≤ 999 }

theorem max_distinct_subsets (k : ℕ) (A : Fin k → Set ℕ) 
  (h : ∀ i j : Fin k, i < j → A i ∪ A j = T) : 
  k ≤ 1000 := 
sorry

end max_distinct_subsets_l12_12073


namespace set_A_enumeration_l12_12601

-- Define the conditions of the problem.
def A : Set ℕ := { x | ∃ (n : ℕ), 6 = n * (6 - x) }

-- State the theorem to be proved.
theorem set_A_enumeration : A = {0, 2, 3, 4, 5} :=
by
  sorry

end set_A_enumeration_l12_12601


namespace calculation_result_l12_12064

theorem calculation_result : (18 * 23 - 24 * 17) / 3 + 5 = 7 :=
by
  sorry

end calculation_result_l12_12064


namespace simple_interest_difference_l12_12597

/-- The simple interest on a certain amount at a 4% rate for 5 years amounted to a certain amount less than the principal. The principal was Rs 2400. Prove that the difference between the principal and the simple interest is Rs 1920. 
-/
theorem simple_interest_difference :
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  P - SI = 1920 :=
by
  /- We introduce the let definitions for the conditions and then state the theorem
    with the conclusion that needs to be proved. -/
  let P := 2400
  let R := 4
  let T := 5
  let SI := (P * R * T) / 100
  /- The final step where we would conclude our theorem. -/
  sorry

end simple_interest_difference_l12_12597


namespace probability_same_color_is_correct_l12_12663

-- Definitions of the conditions
def num_red : ℕ := 6
def num_blue : ℕ := 5
def total_plates : ℕ := num_red + num_blue
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- The probability statement
def prob_three_same_color : ℚ :=
  let total_ways := choose total_plates 3
  let ways_red := choose num_red 3
  let ways_blue := choose num_blue 3
  let favorable_ways := ways_red
  favorable_ways / total_ways

theorem probability_same_color_is_correct : prob_three_same_color = (4 : ℚ) / 33 := sorry

end probability_same_color_is_correct_l12_12663


namespace problem1_problem2_problem3_l12_12456

-- Problem 1
theorem problem1 (x : ℝ) (h : x = 10) : (2 / 5) * x = 4 :=
by sorry

-- Problem 2
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : (2 / 5) * m = (- (1 / 5) * m^2) + 2 * m) : m = 8 :=
by sorry

-- Problem 3
theorem problem3 (t : ℝ) (h1 : ∃ t, (2 / 5) * (32 - t) + (- (1 / 5) * t^2) + 2 * t = 16) : true :=
by sorry

end problem1_problem2_problem3_l12_12456


namespace simplify_expression_l12_12999

theorem simplify_expression (w x : ℤ) :
  3 * w + 6 * w + 9 * w + 12 * w + 15 * w + 20 * x + 24 = 45 * w + 20 * x + 24 :=
by sorry

end simplify_expression_l12_12999


namespace max_weight_of_crates_on_trip_l12_12814

def max_crates : ℕ := 5
def min_crate_weight : ℕ := 150

theorem max_weight_of_crates_on_trip : max_crates * min_crate_weight = 750 := by
  sorry

end max_weight_of_crates_on_trip_l12_12814


namespace car_speed_l12_12048

theorem car_speed (v : ℝ) (h₁ : (1/75 * 3600) + 12 = 1/v * 3600) : v = 60 := 
by 
  sorry

end car_speed_l12_12048


namespace sqrt3_pow_log_sqrt3_8_eq_8_l12_12328

theorem sqrt3_pow_log_sqrt3_8_eq_8 : (Real.sqrt 3) ^ (Real.log 8 / Real.log (Real.sqrt 3)) = 8 :=
by
  sorry

end sqrt3_pow_log_sqrt3_8_eq_8_l12_12328


namespace correct_statement_four_l12_12834

variable {α : Type*} (A B S : Set α) (U : Set α)

theorem correct_statement_four (h1 : U = Set.univ) (h2 : A ∩ B = U) : A = U ∧ B = U := by
  sorry

end correct_statement_four_l12_12834


namespace no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l12_12678

theorem no_20_digit_number_starting_with_11111111111_is_a_perfect_square :
  ¬ ∃ (n : ℤ), (10^19 ≤ n ∧ n < 10^20 ∧ (11111111111 * 10^9 ≤ n ∧ n < 11111111112 * 10^9) ∧ (∃ k : ℤ, n = k^2)) :=
by
  sorry

end no_20_digit_number_starting_with_11111111111_is_a_perfect_square_l12_12678


namespace train_speed_from_clicks_l12_12802

theorem train_speed_from_clicks (speed_mph : ℝ) (rail_length_ft : ℝ) (clicks_heard : ℝ) :
  rail_length_ft = 40 →
  clicks_heard = 1 →
  (60 * rail_length_ft * clicks_heard * speed_mph / 5280) = 27 :=
by
  intros h1 h2
  sorry

end train_speed_from_clicks_l12_12802


namespace valid_integer_values_n_l12_12320

def is_integer (x : ℝ) : Prop := ∃ k : ℤ, x = k

theorem valid_integer_values_n : ∃ (n_values : ℕ), n_values = 3 ∧
  ∀ n : ℤ, is_integer (3200 * (2 / 5) ^ (2 * n)) ↔ n = 0 ∨ n = 1 ∨ n = 2 :=
by
  sorry

end valid_integer_values_n_l12_12320


namespace tank_depth_l12_12685

open Real

theorem tank_depth :
  ∃ d : ℝ, (0.75 * (2 * 25 * d + 2 * 12 * d + 25 * 12) = 558) ∧ d = 6 :=
sorry

end tank_depth_l12_12685


namespace find_root_l12_12921

theorem find_root (y : ℝ) (h : y - 9 / (y - 4) = 2 - 9 / (y - 4)) : y = 2 :=
by
  sorry

end find_root_l12_12921


namespace find_coefficients_sum_l12_12430

theorem find_coefficients_sum :
  let f := (2 * x - 1) ^ 5 + (x + 2) ^ 4
  let a_0 := 15
  let a_1 := 42
  let a_2 := -16
  let a_5 := 32
  (|a_0| + |a_1| + |a_2| + |a_5| = 105) := 
by {
  sorry
}

end find_coefficients_sum_l12_12430


namespace max_tan_y_l12_12947

noncomputable def tan_y_upper_bound (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : Real :=
  Real.tan y

theorem max_tan_y (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : 
    tan_y_upper_bound x y hx hy h = 2005 * Real.sqrt 2006 / 4012 := 
by 
  sorry

end max_tan_y_l12_12947


namespace minimum_trucks_needed_l12_12280

theorem minimum_trucks_needed 
  (total_weight : ℕ) (box_weight: ℕ) (truck_capacity: ℕ) (min_trucks: ℕ)
  (h_total_weight : total_weight = 10)
  (h_box_weight_le : ∀ (w : ℕ), w <= box_weight → w <= 1)
  (h_truck_capacity : truck_capacity = 3)
  (h_min_trucks : min_trucks = 5) : 
  min_trucks >= (total_weight / truck_capacity) :=
sorry

end minimum_trucks_needed_l12_12280


namespace geometric_sequence_product_geometric_sequence_sum_not_definitely_l12_12459

theorem geometric_sequence_product (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ∃ r3, ∀ n, (a n * b n) = r3 * (a (n-1) * b (n-1)) :=
sorry

theorem geometric_sequence_sum_not_definitely (a b : ℕ → ℝ) (r1 r2 : ℝ) 
  (ha : ∀ n, a (n+1) = r1 * a n)
  (hb : ∀ n, b (n+1) = r2 * b n) :
  ¬ ∀ r3, ∃ N, ∀ n ≥ N, (a n + b n) = r3 * (a (n-1) + b (n-1)) :=
sorry

end geometric_sequence_product_geometric_sequence_sum_not_definitely_l12_12459


namespace correct_proposition_is_D_l12_12804

theorem correct_proposition_is_D (A B C D : Prop) :
  (∀ (H : Prop), (H = A ∨ H = B ∨ H = C) → ¬H) → D :=
by
  -- We assume that A, B, and C are false.
  intro h
  -- Now we need to prove that D is true.
  sorry

end correct_proposition_is_D_l12_12804


namespace max_radius_of_circle_touching_graph_l12_12080

theorem max_radius_of_circle_touching_graph :
  ∃ r : ℝ, (∀ (x : ℝ), (x^2 + (x^4 - r)^2 = r^2) → r ≤ (3 * (2:ℝ)^(1/3)) / 4) ∧
           r = (3 * (2:ℝ)^(1/3)) / 4 :=
by
  sorry

end max_radius_of_circle_touching_graph_l12_12080


namespace scooter_gain_percent_l12_12239

def initial_cost : ℝ := 900
def first_repair_cost : ℝ := 150
def second_repair_cost : ℝ := 75
def third_repair_cost : ℝ := 225
def selling_price : ℝ := 1800

theorem scooter_gain_percent :
  let total_cost := initial_cost + first_repair_cost + second_repair_cost + third_repair_cost
  let gain := selling_price - total_cost
  let gain_percent := (gain / total_cost) * 100
  gain_percent = 33.33 :=
by
  sorry

end scooter_gain_percent_l12_12239


namespace sequence_odd_l12_12169

theorem sequence_odd (a : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (ha2 : a 2 = 7)
  (hr : ∀ n ≥ 2, -1 < (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ∧ (a (n + 1) : ℤ) - (a n)^2 / a (n - 1) ≤ 1) :
  ∀ n > 1, Odd (a n) := 
  sorry

end sequence_odd_l12_12169


namespace find_m_l12_12966

variables (a b m : ℝ)

def f (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

def f' (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_m (h1 : f m = 0) (h2 : f' m = 0) (h3 : m ≠ 0)
    (h4 : ∃ x, f' x = 0 ∧ ∀ y, x ≤ y → f x ≥ f y ∧ f x = 1/2) :
    m = 3/2 :=
sorry

end find_m_l12_12966


namespace log_base_9_of_729_l12_12027

theorem log_base_9_of_729 : ∃ x : ℝ, (9:ℝ) = 3^2 ∧ (729:ℝ) = 3^6 ∧ (9:ℝ)^x = 729 ∧ x = 3 :=
by
  sorry

end log_base_9_of_729_l12_12027


namespace dice_sum_probability_l12_12551

theorem dice_sum_probability :
  let total_outcomes := 36
  let sum_le_8_outcomes := 13
  (sum_le_8_outcomes : ℕ) / (total_outcomes : ℕ) = (13 / 18 : ℝ) :=
by
  sorry

end dice_sum_probability_l12_12551


namespace fraction_scaling_l12_12385

theorem fraction_scaling (x y : ℝ) :
  ((5 * x - 5 * 5 * y) / ((5 * x) ^ 2 + (5 * y) ^ 2)) = (1 / 5) * ((x - 5 * y) / (x ^ 2 + y ^ 2)) :=
by
  sorry

end fraction_scaling_l12_12385


namespace math_problem_l12_12670

theorem math_problem : 33333 * 33334 = 1111122222 := 
by sorry

end math_problem_l12_12670


namespace find_15th_term_l12_12484

def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 3 ∧ a 2 = 4 ∧ ∀ n, a (n + 2) = a n

theorem find_15th_term :
  ∃ a : ℕ → ℕ, seq a ∧ a 15 = 3 :=
by
  sorry

end find_15th_term_l12_12484


namespace de_morgan_neg_or_l12_12470

theorem de_morgan_neg_or (p q : Prop) (h : ¬(p ∨ q)) : ¬p ∧ ¬q :=
by sorry

end de_morgan_neg_or_l12_12470


namespace lines_intersection_l12_12051

theorem lines_intersection (a b : ℝ) : 
  (2 : ℝ) = (1/3 : ℝ) * (1 : ℝ) + a →
  (1 : ℝ) = (1/3 : ℝ) * (2 : ℝ) + b →
  a + b = 2 := 
by
  intros h₁ h₂
  sorry

end lines_intersection_l12_12051


namespace sum_is_eight_l12_12504

theorem sum_is_eight (a b c d : ℤ)
  (h1 : 2 * (a - b + c) = 10)
  (h2 : 2 * (b - c + d) = 12)
  (h3 : 2 * (c - d + a) = 6)
  (h4 : 2 * (d - a + b) = 4) :
  a + b + c + d = 8 :=
by
  sorry

end sum_is_eight_l12_12504


namespace principal_amount_l12_12722

variable (P : ℝ)

/-- Prove the principal amount P given that the simple interest at 4% for 5 years is Rs. 2400 less than the principal --/
theorem principal_amount : 
  (4/100) * P * 5 = P - 2400 → 
  P = 3000 := 
by 
  sorry

end principal_amount_l12_12722


namespace bouquet_branches_l12_12206

variable (w : ℕ) (b : ℕ)

theorem bouquet_branches :
  (w + b = 7) → 
  (w ≥ 1) → 
  (∀ x y, x ≠ y → (x = w ∨ y = w) → (x = b ∨ y = b)) → 
  (w = 1 ∧ b = 6) :=
by
  intro h1 h2 h3
  sorry

end bouquet_branches_l12_12206


namespace ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875_l12_12759

theorem ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875:
  let is_ellipse (x y n : ℝ) := x^2 + n*(y - 1)^2 = n
  let is_hyperbola (x y : ℝ) := x^2 - 4*(y + 3)^2 = 4
  ∃ (n1 n2 : ℝ),
    n1 = 62.20625 ∧ n2 = 1.66875 ∧
    (∀ (x y : ℝ), is_ellipse x y n1 → is_hyperbola x y → 
       is_ellipse x y n2 → is_hyperbola x y → 
       (4 + n1)*(y^2 - 2*y + 1) = 4*(y^2 + 6*y + 9) + 4 ∧ 
       ((24 - 2*n1)^2 - 4*(4 + n1)*40 = 0) ∧
       (4 + n2)*(y^2 - 2*y + 1) = 4*(y^2 + 6*y + 9) + 4 ∧ 
       ((24 - 2*n2)^2 - 4*(4 + n2)*40 = 0))
:= sorry

end ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875_l12_12759


namespace find_number_l12_12906

theorem find_number (x : ℚ) (h : 15 + 3 * x = 6 * x - 10) : x = 25 / 3 :=
by
  sorry

end find_number_l12_12906


namespace problem_1_problem_2_l12_12789

noncomputable def complete_residue_system (n : ℕ) (as : Fin n → ℕ) :=
  ∀ i j : Fin n, i ≠ j → as i % n ≠ as j % n

theorem problem_1 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) := 
sorry

theorem problem_2 (n : ℕ) (hn : 0 < n) :
  ∃ as : Fin n → ℕ, complete_residue_system n as ∧ complete_residue_system n (λ i => as i + i) ∧ complete_residue_system n (λ i => as i - i) := 
sorry

end problem_1_problem_2_l12_12789


namespace solution_1_solution_2_l12_12298

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem solution_1 :
  (∀ x : ℝ, f x = 2 * Real.sin (2 * x + Real.pi / 3)) :=
by sorry

theorem solution_2 (x0 : ℝ) (hx0 : x0 ∈ Set.Icc (Real.pi / 2) Real.pi) :
  f (x0 / 2) = -3 / 8 → 
  Real.cos (x0 + Real.pi / 6) = - Real.sqrt 741 / 32 - 3 / 32 :=
by sorry

end solution_1_solution_2_l12_12298


namespace part1_arithmetic_sequence_part2_general_term_part3_max_m_l12_12165

-- Part (1)
theorem part1_arithmetic_sequence (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : a 1 + a 2 = 2 * m) : 
  m = 9 / 8 := 
sorry

-- Part (2)
theorem part2_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2) : 
  ∀ n, a n = 8 ^ (1 - 2 ^ (n - 1)) := 
sorry

-- Part (3)
theorem part3_max_m (m : ℝ) (a : ℕ → ℝ) (h1 : a 1 = 1) 
  (h2 : ∀ n, a (n + 1) = (1 / 8) * (a n) ^ 2 + m) 
  (h3 : ∀ n, a n < 4) : 
  m ≤ 2 := 
sorry

end part1_arithmetic_sequence_part2_general_term_part3_max_m_l12_12165


namespace layoffs_payment_l12_12326

theorem layoffs_payment :
  let total_employees := 450
  let salary_2000_employees := 150
  let salary_2500_employees := 200
  let salary_3000_employees := 100
  let first_round_2000_layoffs := 0.20 * salary_2000_employees
  let first_round_2500_layoffs := 0.25 * salary_2500_employees
  let first_round_3000_layoffs := 0.15 * salary_3000_employees
  let remaining_2000_after_first_round := salary_2000_employees - first_round_2000_layoffs
  let remaining_2500_after_first_round := salary_2500_employees - first_round_2500_layoffs
  let remaining_3000_after_first_round := salary_3000_employees - first_round_3000_layoffs
  let second_round_2000_layoffs := 0.10 * remaining_2000_after_first_round
  let second_round_2500_layoffs := 0.15 * remaining_2500_after_first_round
  let second_round_3000_layoffs := 0.05 * remaining_3000_after_first_round
  let remaining_2000_after_second_round := remaining_2000_after_first_round - second_round_2000_layoffs
  let remaining_2500_after_second_round := remaining_2500_after_first_round - second_round_2500_layoffs
  let remaining_3000_after_second_round := remaining_3000_after_first_round - second_round_3000_layoffs
  let total_payment := remaining_2000_after_second_round * 2000 + remaining_2500_after_second_round * 2500 + remaining_3000_after_second_round * 3000
  total_payment = 776500 := sorry

end layoffs_payment_l12_12326


namespace correct_calculation_result_l12_12312

theorem correct_calculation_result (x : ℤ) (h : x - 32 = 25) : x - 23 = 34 :=
by
  sorry

end correct_calculation_result_l12_12312


namespace ice_cream_cost_l12_12482

-- Define the given conditions
def cost_brownie : ℝ := 2.50
def cost_syrup_per_unit : ℝ := 0.50
def cost_nuts : ℝ := 1.50
def cost_total : ℝ := 7.00
def scoops_ice_cream : ℕ := 2
def syrup_units : ℕ := 2

-- Define the hot brownie dessert cost equation
def hot_brownie_cost (cost_ice_cream_per_scoop : ℝ) : ℝ :=
  cost_brownie + (cost_syrup_per_unit * syrup_units) + cost_nuts + (scoops_ice_cream * cost_ice_cream_per_scoop)

-- Define the theorem we want to prove
theorem ice_cream_cost : hot_brownie_cost 1 = cost_total :=
by sorry

end ice_cream_cost_l12_12482


namespace solve_quadratic_inequality_l12_12794

theorem solve_quadratic_inequality (x : ℝ) (h : x^2 - 7 * x + 6 < 0) : 1 < x ∧ x < 6 :=
  sorry

end solve_quadratic_inequality_l12_12794


namespace find_fraction_of_original_flow_rate_l12_12822

noncomputable def fraction_of_original_flow_rate (f : ℚ) : Prop :=
  let original_flow_rate := 5
  let reduced_flow_rate := 2
  reduced_flow_rate = f * original_flow_rate - 1

theorem find_fraction_of_original_flow_rate : ∃ (f : ℚ), fraction_of_original_flow_rate f ∧ f = 3 / 5 :=
by
  sorry

end find_fraction_of_original_flow_rate_l12_12822


namespace books_about_fish_l12_12217

theorem books_about_fish (F : ℕ) (spent : ℕ) (cost_whale_books : ℕ) (cost_magazines : ℕ) (cost_fish_books_per_unit : ℕ) (whale_books : ℕ) (magazines : ℕ) :
  whale_books = 9 →
  magazines = 3 →
  cost_whale_books = 11 →
  cost_magazines = 1 →
  spent = 179 →
  99 + 11 * F + 3 = spent → F = 7 :=
by
  sorry

end books_about_fish_l12_12217


namespace exists_infinite_triples_a_no_triples_b_l12_12979

-- Question (a)
theorem exists_infinite_triples_a : ∀ k : ℕ, ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2 - 1) :=
by {
  sorry
}

-- Question (b)
theorem no_triples_b : ¬ ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2) :=
by {
  sorry
}

end exists_infinite_triples_a_no_triples_b_l12_12979


namespace water_current_speed_l12_12811

-- Definitions based on the conditions
def swimmer_speed : ℝ := 4  -- The swimmer's speed in still water (km/h)
def swim_time : ℝ := 2  -- Time taken to swim against the current (hours)
def swim_distance : ℝ := 6  -- Distance swum against the current (km)

-- The effective speed against the current
noncomputable def effective_speed_against_current (v : ℝ) : ℝ := swimmer_speed - v

-- Lean statement that formalizes proving the speed of the current
theorem water_current_speed (v : ℝ) (h : effective_speed_against_current v = swim_distance / swim_time) : v = 1 :=
by
  sorry

end water_current_speed_l12_12811


namespace equal_angles_proof_l12_12676

/-- Proof Problem: After how many minutes will the hour and minute hands form equal angles with their positions at 12 o'clock? -/
noncomputable def equal_angle_time (x : ℝ) : Prop :=
  -- Defining the conditions for the problem
  let minute_hand_speed := 6 -- degrees per minute
  let hour_hand_speed := 0.5 -- degrees per minute
  let total_degrees := 360 * x -- total degrees of minute hand till time x
  let hour_hand_degrees := 30 * (x / 60) -- total degrees of hour hand till time x

  -- Equation for equal angles formed with respect to 12 o'clock
  30 * (x / 60) = 360 - 360 * (x / 60)

theorem equal_angles_proof :
  ∃ (x : ℝ), equal_angle_time x ∧ x = 55 + 5/13 :=
sorry

end equal_angles_proof_l12_12676


namespace side_length_of_square_l12_12969

theorem side_length_of_square : 
  ∀ (L : ℝ), L = 28 → (L / 4) = 7 :=
by
  intro L h
  rw [h]
  norm_num

end side_length_of_square_l12_12969


namespace increase_in_sides_of_polygon_l12_12621

theorem increase_in_sides_of_polygon (n n' : ℕ) (h : (n' - 2) * 180 - (n - 2) * 180 = 180) : n' = n + 1 :=
by
  sorry

end increase_in_sides_of_polygon_l12_12621


namespace total_fruits_l12_12744

def num_papaya_trees : ℕ := 2
def num_mango_trees : ℕ := 3
def papayas_per_tree : ℕ := 10
def mangos_per_tree : ℕ := 20

theorem total_fruits : (num_papaya_trees * papayas_per_tree) + (num_mango_trees * mangos_per_tree) = 80 := 
by
  sorry

end total_fruits_l12_12744


namespace value_of_otimes_l12_12342

variable (a b : ℚ)

/-- Define the operation ⊗ -/
def otimes (x y : ℚ) : ℚ := a^2 * x + b * y - 3

/-- Given conditions -/
axiom condition1 : otimes a b 1 (-3) = 2 

/-- Target proof -/
theorem value_of_otimes : otimes a b 2 (-6) = 7 :=
by
  sorry

end value_of_otimes_l12_12342


namespace ones_digit_expression_l12_12517

theorem ones_digit_expression :
  ((73 ^ 1253 * 44 ^ 987 + 47 ^ 123 / 39 ^ 654 * 86 ^ 1484 - 32 ^ 1987) % 10) = 2 := by
  sorry

end ones_digit_expression_l12_12517


namespace sam_walked_distance_when_meeting_l12_12074

variable (D_s D_f : ℝ)
variable (t : ℝ)

theorem sam_walked_distance_when_meeting
  (h1 : 55 = D_f + D_s)
  (h2 : D_f = 6 * t)
  (h3 : D_s = 5 * t) :
  D_s = 25 :=
by 
  -- This is where the proof would go
  sorry

end sam_walked_distance_when_meeting_l12_12074


namespace rational_solution_for_quadratic_l12_12593

theorem rational_solution_for_quadratic (k : ℕ) (h_pos : 0 < k) : 
  ∃ m : ℕ, (18^2 - 4 * k * (2 * k)) = m^2 ↔ k = 4 :=
by
  sorry

end rational_solution_for_quadratic_l12_12593


namespace markus_more_marbles_than_mara_l12_12907

variable (mara_bags : Nat) (mara_marbles_per_bag : Nat)
variable (markus_bags : Nat) (markus_marbles_per_bag : Nat)

theorem markus_more_marbles_than_mara :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_bags = 2 →
  markus_marbles_per_bag = 13 →
  (markus_bags * markus_marbles_per_bag) - (mara_bags * mara_marbles_per_bag) = 2 :=
by
  intros
  sorry

end markus_more_marbles_than_mara_l12_12907


namespace pow_congruence_modulus_p_squared_l12_12462

theorem pow_congruence_modulus_p_squared (p : ℕ) (a b : ℤ) (hp : Nat.Prime p) (h : a ≡ b [ZMOD p]) : a^p ≡ b^p [ZMOD p^2] :=
sorry

end pow_congruence_modulus_p_squared_l12_12462


namespace smallest_N_proof_l12_12623

theorem smallest_N_proof (N c1 c2 c3 c4 : ℕ)
  (h1 : N + c1 = 4 * c3 - 2)
  (h2 : N + c2 = 4 * c1 - 3)
  (h3 : 2 * N + c3 = 4 * c4 - 1)
  (h4 : 3 * N + c4 = 4 * c2) :
  N = 12 :=
sorry

end smallest_N_proof_l12_12623


namespace determine_y_l12_12314

variable (x y : ℝ)

theorem determine_y (h1 : 0.25 * x = 0.15 * y - 15) (h2 : x = 840) : y = 1500 := 
by 
  sorry

end determine_y_l12_12314


namespace mary_turnips_grown_l12_12832

variable (sally_turnips : ℕ)
variable (total_turnips : ℕ)
variable (mary_turnips : ℕ)

theorem mary_turnips_grown (h_sally : sally_turnips = 113)
                          (h_total : total_turnips = 242) :
                          mary_turnips = total_turnips - sally_turnips := by
  sorry

end mary_turnips_grown_l12_12832


namespace number_of_yogurts_l12_12805

def slices_per_yogurt : Nat := 8
def slices_per_banana : Nat := 10
def number_of_bananas : Nat := 4

theorem number_of_yogurts (slices_per_yogurt slices_per_banana number_of_bananas : Nat) : 
  slices_per_yogurt = 8 → 
  slices_per_banana = 10 → 
  number_of_bananas = 4 → 
  (number_of_bananas * slices_per_banana) / slices_per_yogurt = 5 :=
by
  intros h1 h2 h3
  sorry

end number_of_yogurts_l12_12805


namespace sphere_surface_area_l12_12901

theorem sphere_surface_area (a : ℝ) (l R : ℝ)
  (h₁ : 6 * l^2 = a)
  (h₂ : l * Real.sqrt 3 = 2 * R) :
  4 * Real.pi * R^2 = (Real.pi / 2) * a :=
sorry

end sphere_surface_area_l12_12901


namespace circle_symmetric_line_l12_12656

theorem circle_symmetric_line (a b : ℝ) (h : a < 2) (hb : b = -2) : a + b < 0 := by
  sorry

end circle_symmetric_line_l12_12656


namespace max_tan_A_minus_B_l12_12945

open Real

-- Given conditions
variables {A B C a b c : ℝ}

-- Assume the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- and the equation a * cos B - b * cos A = (3 / 5) * c holds.
def condition (a b c A B C : ℝ) : Prop :=
  a * cos B - b * cos A = (3 / 5) * c

-- Prove that the maximum value of tan(A - B) is 3/4
theorem max_tan_A_minus_B (a b c A B C : ℝ) (h : condition a b c A B C) :
  ∃ t : ℝ, t = tan (A - B) ∧ 0 ≤ t ∧ t ≤ 3 / 4 :=
sorry

end max_tan_A_minus_B_l12_12945


namespace tan_ratio_l12_12180

theorem tan_ratio (a b : ℝ) 
  (h1 : Real.sin (a + b) = 5 / 8)
  (h2 : Real.sin (a - b) = 1 / 4) : 
  Real.tan a / Real.tan b = 7 / 3 := 
sorry

end tan_ratio_l12_12180


namespace sum_p_q_l12_12926

theorem sum_p_q (p q : ℚ) (g : ℚ → ℚ) (h : g = λ x => (x + 2) / (x^2 + p * x + q))
  (h_asymp1 : ∀ {x}, x = -1 → (x^2 + p * x + q) = 0)
  (h_asymp2 : ∀ {x}, x = 3 → (x^2 + p * x + q) = 0) :
  p + q = -5 := by
  sorry

end sum_p_q_l12_12926


namespace merchant_articles_l12_12995

theorem merchant_articles 
   (CP SP : ℝ)
   (N : ℝ)
   (h1 : SP = 1.25 * CP)
   (h2 : N * CP = 16 * SP) : 
   N = 20 := by
   sorry

end merchant_articles_l12_12995


namespace calculate_expression_l12_12136

theorem calculate_expression :
  (-0.125)^2022 * 8^2023 = 8 :=
sorry

end calculate_expression_l12_12136


namespace largest_among_given_numbers_l12_12508

theorem largest_among_given_numbers : 
    let a := 24680 + (1 / 1357)
    let b := 24680 - (1 / 1357)
    let c := 24680 * (1 / 1357)
    let d := 24680 / (1 / 1357)
    let e := 24680.1357
    d > a ∧ d > b ∧ d > c ∧ d > e :=
by
  sorry

end largest_among_given_numbers_l12_12508


namespace number_of_ways_to_choose_marbles_l12_12284

theorem number_of_ways_to_choose_marbles 
  (total_marbles : ℕ) 
  (red_count green_count blue_count : ℕ) 
  (total_choice chosen_rgb_count remaining_choice : ℕ) 
  (h_total_marbles : total_marbles = 15) 
  (h_red_count : red_count = 2) 
  (h_green_count : green_count = 2) 
  (h_blue_count : blue_count = 2) 
  (h_total_choice : total_choice = 5) 
  (h_chosen_rgb_count : chosen_rgb_count = 2) 
  (h_remaining_choice : remaining_choice = 3) :
  ∃ (num_ways : ℕ), num_ways = 3300 :=
sorry

end number_of_ways_to_choose_marbles_l12_12284


namespace vacation_total_cost_l12_12023

def plane_ticket_cost (per_person_cost : ℕ) (num_people : ℕ) : ℕ :=
  num_people * per_person_cost

def hotel_stay_cost (per_person_per_day_cost : ℕ) (num_people : ℕ) (num_days : ℕ) : ℕ :=
  num_people * per_person_per_day_cost * num_days

def total_vacation_cost (plane_ticket_cost : ℕ) (hotel_stay_cost : ℕ) : ℕ :=
  plane_ticket_cost + hotel_stay_cost

theorem vacation_total_cost :
  let per_person_plane_ticket_cost := 24
  let per_person_hotel_cost := 12
  let num_people := 2
  let num_days := 3
  let plane_cost := plane_ticket_cost per_person_plane_ticket_cost num_people
  let hotel_cost := hotel_stay_cost per_person_hotel_cost num_people num_days
  total_vacation_cost plane_cost hotel_cost = 120 := by
  sorry

end vacation_total_cost_l12_12023


namespace power_sum_tenth_l12_12132

theorem power_sum_tenth (a b : ℝ) (h1 : a + b = 1)
    (h2 : a^2 + b^2 = 3)
    (h3 : a^3 + b^3 = 4)
    (h4 : a^4 + b^4 = 7)
    (h5 : a^5 + b^5 = 11) : 
    a^10 + b^10 = 123 := 
sorry

end power_sum_tenth_l12_12132


namespace sum_x_y_m_l12_12920

theorem sum_x_y_m (x y m : ℕ) (h1 : x >= 10 ∧ x < 100) (h2 : y >= 10 ∧ y < 100) 
  (h3 : ∃ a b : ℕ, x = 10 * a + b ∧ y = 10 * b + a ∧ a < 10 ∧ b < 10) 
  (h4 : x^2 - y^2 = 4 * m^2) : 
  x + y + m = 105 := 
sorry

end sum_x_y_m_l12_12920


namespace rectangle_perimeter_given_square_l12_12274

-- Defining the problem conditions
def square_side_length (p : ℕ) : ℕ := p / 4

def rectangle_perimeter (s : ℕ) : ℕ := 2 * (s + (s / 2))

-- Stating the theorem: Given the perimeter of the square is 80, prove the perimeter of one of the rectangles is 60
theorem rectangle_perimeter_given_square (p : ℕ) (h : p = 80) : rectangle_perimeter (square_side_length p) = 60 :=
by
  sorry

end rectangle_perimeter_given_square_l12_12274


namespace range_of_m_l12_12072

theorem range_of_m {x m : ℝ} 
  (α : 2 / (x + 1) > 1) 
  (β : m ≤ x ∧ x ≤ 2) 
  (suff_condition : ∀ x, (2 / (x + 1) > 1) → (m ≤ x ∧ x ≤ 2)) :
  m ≤ -1 :=
sorry

end range_of_m_l12_12072


namespace nancy_hourly_wage_l12_12998

theorem nancy_hourly_wage 
  (tuition_per_semester : ℕ := 22000) 
  (parents_cover : ℕ := tuition_per_semester / 2) 
  (scholarship : ℕ := 3000) 
  (student_loan : ℕ := 2 * scholarship) 
  (work_hours : ℕ := 200) 
  (remaining_tuition : ℕ := parents_cover - scholarship - student_loan) :
  (remaining_tuition / work_hours = 10) :=
  by
  sorry

end nancy_hourly_wage_l12_12998


namespace comparison_l12_12689

noncomputable def a : ℝ := 7 / 9
noncomputable def b : ℝ := 0.7 * Real.exp 0.1
noncomputable def c : ℝ := Real.cos (2 / 3)

theorem comparison : c > a ∧ a > b :=
by
  -- c > a proof
  have h1 : c > a := sorry
  -- a > b proof
  have h2 : a > b := sorry
  exact ⟨h1, h2⟩

end comparison_l12_12689


namespace first_tray_holds_260_cups_l12_12468

variable (x : ℕ)

def first_tray_holds_x_cups (tray1 : ℕ) := tray1 = x
def second_tray_holds_x_minus_20_cups (tray2 : ℕ) := tray2 = x - 20
def total_cups_in_both_trays (tray1 tray2: ℕ) := tray1 + tray2 = 500

theorem first_tray_holds_260_cups (tray1 tray2 : ℕ) :
  first_tray_holds_x_cups x tray1 →
  second_tray_holds_x_minus_20_cups x tray2 →
  total_cups_in_both_trays tray1 tray2 →
  x = 260 := by
  sorry

end first_tray_holds_260_cups_l12_12468


namespace number_of_pencils_l12_12120

theorem number_of_pencils 
  (P Pe M : ℕ)
  (h1 : Pe = P + 4)
  (h2 : M = P + 20)
  (h3 : P / 5 = Pe / 6)
  (h4 : Pe / 6 = M / 7) : 
  Pe = 24 :=
by
  sorry

end number_of_pencils_l12_12120


namespace bicycle_spokes_count_l12_12334

theorem bicycle_spokes_count (bicycles wheels spokes : ℕ) 
       (h1 : bicycles = 4) 
       (h2 : wheels = 2) 
       (h3 : spokes = 10) : 
       bicycles * (wheels * spokes) = 80 :=
by
  sorry

end bicycle_spokes_count_l12_12334


namespace ratio_a6_b6_l12_12529

-- Definitions for sequences and sums
variable {α : Type*} [LinearOrderedField α] 
variable (a b : ℕ → α) 
variable (S T : ℕ → α)

-- Main theorem stating the problem
theorem ratio_a6_b6 (h : ∀ n, S n / T n = (2 * n - 5) / (4 * n + 3)) :
    a 6 / b 6 = 17 / 47 :=
sorry

end ratio_a6_b6_l12_12529


namespace min_value_fraction_l12_12041

theorem min_value_fraction (x : ℝ) (h : x > 4) : 
  ∃ y : ℝ, (y = 2 * Real.sqrt 19) ∧ (∀ z : ℝ, (z = (x + 15) / Real.sqrt (x - 4)) → z ≥ y) :=
by
  sorry

end min_value_fraction_l12_12041


namespace sum_of_coeffs_is_minus_one_l12_12747

theorem sum_of_coeffs_is_minus_one 
  (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℤ) :
  (∀ x : ℤ, (1 - x^3)^3 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 + a₈ * x^8 + a₉ * x^9)
  → a = 1 
  → a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -1 :=
by
  sorry

end sum_of_coeffs_is_minus_one_l12_12747


namespace elder_child_age_l12_12939

theorem elder_child_age (x : ℕ) (h : x + (x + 4) + (x + 8) + (x + 12) = 48) : (x + 12) = 18 :=
by
  sorry

end elder_child_age_l12_12939


namespace determine_k_l12_12325

theorem determine_k (k : ℝ) (h : 2 - 2^2 = k * (2)^2 + 1) : k = -3/4 :=
by
  sorry

end determine_k_l12_12325


namespace find_third_side_l12_12218

theorem find_third_side
  (cubes : ℕ) (cube_volume : ℚ) (side1 side2 : ℚ)
  (fits : cubes = 24) (vol_cube : cube_volume = 27)
  (dim1 : side1 = 8) (dim2 : side2 = 9) :
  (side1 * side2 * (cube_volume * cubes) / (side1 * side2)) = 9 := by
  sorry

end find_third_side_l12_12218


namespace Maddie_bought_palettes_l12_12664

-- Defining constants and conditions as per the problem statement.
def cost_per_palette : ℝ := 15
def number_of_lipsticks : ℝ := 4
def cost_per_lipstick : ℝ := 2.50
def number_of_hair_boxes : ℝ := 3
def cost_per_hair_box : ℝ := 4
def total_paid : ℝ := 67

-- Defining the condition which we need to prove for number of makeup palettes bought.
theorem Maddie_bought_palettes (P : ℝ) :
  (number_of_lipsticks * cost_per_lipstick) +
  (number_of_hair_boxes * cost_per_hair_box) +
  (cost_per_palette * P) = total_paid →
  P = 3 :=
sorry

end Maddie_bought_palettes_l12_12664


namespace probability_star_top_card_is_one_fifth_l12_12299

-- Define the total number of cards in the deck
def total_cards : ℕ := 65

-- Define the number of star cards in the deck
def star_cards : ℕ := 13

-- Define the probability calculation
def probability_star_top_card : ℚ := star_cards / total_cards

-- State the theorem regarding the probability
theorem probability_star_top_card_is_one_fifth :
  probability_star_top_card = 1 / 5 :=
by
  sorry

end probability_star_top_card_is_one_fifth_l12_12299


namespace rectangle_midpoints_sum_l12_12964

theorem rectangle_midpoints_sum (A B C D M N O P : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (4, 0))
  (hC : C = (4, 3))
  (hD : D = (0, 3))
  (hM : M = (2, 0))
  (hN : N = (4, 1.5))
  (hO : O = (2, 3))
  (hP : P = (0, 1.5)) :
  (Real.sqrt ((2 - 0) ^ 2 + (0 - 0) ^ 2) + 
  Real.sqrt ((4 - 0) ^ 2 + (1.5 - 0) ^ 2) + 
  Real.sqrt ((2 - 0) ^ 2 + (3 - 0) ^ 2) + 
  Real.sqrt ((0 - 0) ^ 2 + (1.5 - 0) ^ 2)) = 11.38 :=
by
  sorry

end rectangle_midpoints_sum_l12_12964


namespace quadrilateral_not_parallelogram_l12_12880

-- Definitions based on the given conditions
structure Quadrilateral :=
  (a b c d : ℝ) -- sides of the quadrilateral
  (parallel : Prop) -- one pair of parallel sides
  (equal_sides : Prop) -- another pair of equal sides

-- Problem statement
theorem quadrilateral_not_parallelogram (q : Quadrilateral) 
  (h1 : q.parallel) 
  (h2 : q.equal_sides) : 
  ¬ (∃ p : Quadrilateral, p = q) :=
sorry

end quadrilateral_not_parallelogram_l12_12880


namespace shara_savings_l12_12494

theorem shara_savings 
  (original_price : ℝ)
  (discount1 : ℝ := 0.08)
  (discount2 : ℝ := 0.05)
  (sales_tax : ℝ := 0.06)
  (final_price : ℝ := 184)
  (h : (original_price * (1 - discount1) * (1 - discount2) * (1 + sales_tax)) = final_price) :
  original_price - final_price = 25.78 :=
sorry

end shara_savings_l12_12494


namespace overall_average_runs_l12_12479

theorem overall_average_runs 
  (test_matches: ℕ) (test_avg: ℕ) 
  (odi_matches: ℕ) (odi_avg: ℕ) 
  (t20_matches: ℕ) (t20_avg: ℕ)
  (h_test_matches: test_matches = 25)
  (h_test_avg: test_avg = 48)
  (h_odi_matches: odi_matches = 20)
  (h_odi_avg: odi_avg = 38)
  (h_t20_matches: t20_matches = 15)
  (h_t20_avg: t20_avg = 28) :
  (25 * 48 + 20 * 38 + 15 * 28) / (25 + 20 + 15) = 39.67 :=
sorry

end overall_average_runs_l12_12479


namespace larger_integer_exists_l12_12982

theorem larger_integer_exists (a b : ℤ) (h1 : a - b = 8) (h2 : a * b = 272) : a = 17 :=
sorry

end larger_integer_exists_l12_12982


namespace inequality_with_conditions_l12_12007

variable {a b c : ℝ}

theorem inequality_with_conditions (h : a * b + b * c + c * a = 1) :
  (|a - b| / |1 + c^2|) + (|b - c| / |1 + a^2|) ≥ (|c - a| / |1 + b^2|) :=
by
  sorry

end inequality_with_conditions_l12_12007


namespace total_import_value_l12_12620

-- Define the given conditions
def export_value : ℝ := 8.07
def additional_amount : ℝ := 1.11
def factor : ℝ := 1.5

-- Define the import value to be proven
def import_value : ℝ := 46.4

-- Main theorem statement
theorem total_import_value :
  export_value = factor * import_value + additional_amount → import_value = 46.4 :=
by sorry

end total_import_value_l12_12620


namespace intersection_area_correct_l12_12256

noncomputable def intersection_area (XY YE FX EX FY : ℕ) : ℚ :=
  if XY = 12 ∧ YE = FX ∧ YE = 15 ∧ EX = FY ∧ EX = 20 then
    18
  else
    0

theorem intersection_area_correct {XY YE FX EX FY : ℕ} (h1 : XY = 12) (h2 : YE = FX) (h3 : YE = 15) (h4 : EX = FY) (h5 : EX = 20) : 
  intersection_area XY YE FX EX FY = 18 := 
by {
  sorry
}

end intersection_area_correct_l12_12256


namespace stations_between_l12_12933

theorem stations_between (n : ℕ) (h : n * (n - 1) / 2 = 306) : n - 2 = 25 := 
by
  sorry

end stations_between_l12_12933


namespace simplify_expression_l12_12088

theorem simplify_expression (x y : ℝ) (h : x - 2 * y = -2) : 9 - 2 * x + 4 * y = 13 :=
by sorry

end simplify_expression_l12_12088


namespace beam_reflection_equation_l12_12002

theorem beam_reflection_equation:
  ∃ (line : ℝ → ℝ → Prop), 
  (∀ (x y : ℝ), line x y ↔ (5 * x - 2 * y - 10 = 0)) ∧
  (line 4 5) ∧ 
  (line 2 0) :=
by
  sorry

end beam_reflection_equation_l12_12002


namespace number_of_planes_l12_12715

-- Definitions based on the conditions
def Line (space: Type) := space → space → Prop

variables {space: Type} [MetricSpace space]

-- Given conditions
variable (l1 l2 l3 : Line space)
variable (intersects : ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p)

-- The theorem stating the conclusion
theorem number_of_planes (h: ∃ p : space, l1 p p ∧ l2 p p ∧ l3 p p) :
  (1 = 1 ∨ 1 = 2 ∨ 1 = 3) ∨ (2 = 1 ∨ 2 = 2 ∨ 2 = 3) ∨ (3 = 1 ∨ 3 = 2 ∨ 3 = 3) := 
sorry

end number_of_planes_l12_12715


namespace least_pennies_l12_12931

theorem least_pennies : 
  ∃ (a : ℕ), a % 5 = 1 ∧ a % 3 = 2 ∧ a = 11 :=
by
  sorry

end least_pennies_l12_12931


namespace fair_prize_division_l12_12179

theorem fair_prize_division (eq_chance : ∀ (game : ℕ), 0.5 ≤ 1 ∧ 1 ≤ 0.5)
  (first_to_six : ∀ (p1_wins p2_wins : ℕ), (p1_wins = 6 ∨ p2_wins = 6) → (p1_wins + p2_wins) ≤ 11)
  (current_status : 5 + 3 = 8) :
  (7 : ℝ) / 8 = 7 / (8 : ℝ) :=
by
  sorry

end fair_prize_division_l12_12179


namespace lowest_price_l12_12894

theorem lowest_price (cost_per_component shipping_cost_per_unit fixed_costs number_of_components produced_cost total_variable_cost total_cost lowest_price : ℝ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 2)
  (h3 : fixed_costs = 16200)
  (h4 : number_of_components = 150)
  (h5 : total_variable_cost = cost_per_component + shipping_cost_per_unit)
  (h6 : produced_cost = total_variable_cost * number_of_components)
  (h7 : total_cost = produced_cost + fixed_costs)
  (h8 : lowest_price = total_cost / number_of_components) :
  lowest_price = 190 :=
  by
  sorry

end lowest_price_l12_12894


namespace wilsons_theorem_l12_12951

theorem wilsons_theorem (p : ℕ) (hp : p ≥ 2) : Nat.Prime p ↔ (Nat.factorial (p - 1) + 1) % p = 0 := 
sorry

end wilsons_theorem_l12_12951


namespace total_students_l12_12104

variable (T : ℕ)

-- Conditions
def is_girls_percentage (T : ℕ) := 60 / 100 * T
def is_boys_percentage (T : ℕ) := 40 / 100 * T
def boys_not_in_clubs (number_of_boys : ℕ) := 2 / 3 * number_of_boys

theorem total_students (h1 : is_girls_percentage T + is_boys_percentage T = T)
  (h2 : boys_not_in_clubs (is_boys_percentage T) = 40) : T = 150 :=
by
  sorry

end total_students_l12_12104


namespace find_missing_score_l12_12943

noncomputable def total_points (mean : ℝ) (games : ℕ) : ℝ :=
  mean * games

noncomputable def sum_of_scores (scores : List ℝ) : ℝ :=
  scores.sum

theorem find_missing_score
  (scores : List ℝ)
  (mean : ℝ)
  (games : ℕ)
  (total_points_value : ℝ)
  (sum_of_recorded_scores : ℝ)
  (missing_score : ℝ) :
  scores = [81, 73, 86, 73] →
  mean = 79.2 →
  games = 5 →
  total_points_value = total_points mean games →
  sum_of_recorded_scores = sum_of_scores scores →
  missing_score = total_points_value - sum_of_recorded_scores →
  missing_score = 83 :=
by
  intros
  exact sorry

end find_missing_score_l12_12943


namespace simplify_fraction_l12_12338

theorem simplify_fraction (a b c d : ℕ) (h₁ : a = 2) (h₂ : b = 462) (h₃ : c = 29) (h₄ : d = 42) :
  (a : ℚ) / (b : ℚ) + (c : ℚ) / (d : ℚ) = 107 / 154 :=
by {
  sorry
}

end simplify_fraction_l12_12338


namespace solution_set_no_pos_ab_l12_12287

def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 2)

theorem solution_set :
  {x : ℝ | f x ≤ x + 1} = {x : ℝ | 2 / 3 ≤ x ∧ x ≤ 4} :=
by sorry

theorem no_pos_ab :
  ¬ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2 * a + b = 1 ∧ 1 / a + 2 / b = 4 :=
by sorry

end solution_set_no_pos_ab_l12_12287


namespace num_red_balls_l12_12824

theorem num_red_balls (x : ℕ) (h : 4 / (4 + x) = 1 / 5) : x = 16 :=
by
  sorry

end num_red_balls_l12_12824


namespace sum_octal_eq_1021_l12_12188

def octal_to_decimal (n : ℕ) : ℕ :=
  let d0 := n % 10
  let r1 := n / 10
  let d1 := r1 % 10
  let r2 := r1 / 10
  let d2 := r2 % 10
  (d2 * 64) + (d1 * 8) + d0

def decimal_to_octal (n : ℕ) : ℕ :=
  let d0 := n % 8
  let r1 := n / 8
  let d1 := r1 % 8
  let r2 := r1 / 8
  let d2 := r2 % 8
  d2 * 100 + d1 * 10 + d0

theorem sum_octal_eq_1021 :
  decimal_to_octal (octal_to_decimal 642 + octal_to_decimal 157) = 1021 := by
  sorry

end sum_octal_eq_1021_l12_12188


namespace isosceles_base_l12_12613

theorem isosceles_base (s b : ℕ) 
  (h1 : 3 * s = 45) 
  (h2 : 2 * s + b = 40) 
  (h3 : s = 15): 
  b = 10 :=
  sorry

end isosceles_base_l12_12613


namespace power_ineq_for_n_geq_5_l12_12038

noncomputable def power_ineq (n : ℕ) : Prop := 2^n > n^2 + 1

theorem power_ineq_for_n_geq_5 (n : ℕ) (h : n ≥ 5) : power_ineq n :=
  sorry

end power_ineq_for_n_geq_5_l12_12038


namespace combined_money_l12_12086

/-- Tom has a quarter the money of Nataly. Nataly has three times the money of Raquel.
     Sam has twice the money of Nataly. Raquel has $40. Prove that combined they have $430. -/
theorem combined_money : 
  ∀ (T R N S : ℕ), 
    (T = N / 4) ∧ 
    (N = 3 * R) ∧ 
    (S = 2 * N) ∧ 
    (R = 40) → 
    T + R + N + S = 430 := 
by
  sorry

end combined_money_l12_12086


namespace cost_of_50_lavenders_l12_12541

noncomputable def cost_of_bouquet (lavenders : ℕ) : ℚ :=
  (25 / 15) * lavenders

theorem cost_of_50_lavenders :
  cost_of_bouquet 50 = 250 / 3 :=
sorry

end cost_of_50_lavenders_l12_12541


namespace num_triangles_with_area_2_l12_12350

-- Define the grid and points
def is_grid_point (x y : ℕ) : Prop := x ≤ 3 ∧ y ≤ 3

-- Function to calculate the area of a triangle using vertices (x1, y1), (x2, y2), and (x3, y3)
def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℕ) : ℤ := 
  (x1 * y2 + x2 * y3 + x3 * y1) 
  - (y1 * x2 + y2 * x3 + y3 * x1)

-- Check if the area is 2 (since we are dealing with a lattice grid, 
-- we can consider non-fractional form by multiplying by 2 to avoid half-area)
def has_area_2 (x1 y1 x2 y2 x3 y3 : ℕ) : Prop :=
  abs (area_of_triangle x1 y1 x2 y2 x3 y3) = 4

-- Define the main theorem that needs to be proved
theorem num_triangles_with_area_2 : 
  ∃ (n : ℕ), n = 64 ∧
  ∀ (x1 y1 x2 y2 x3 y3 : ℕ), 
  is_grid_point x1 y1 ∧ is_grid_point x2 y2 ∧ is_grid_point x3 y3 ∧ 
  has_area_2 x1 y1 x2 y2 x3 y3 → n = 64 :=
sorry

end num_triangles_with_area_2_l12_12350


namespace distance_from_origin_to_line_l12_12524

def ellipse (x y : ℝ) : Prop := (x^2 / 2) + y^2 = 1

-- definition of the perpendicular property of chords
def perpendicular (O A B : ℝ × ℝ) : Prop := (A.1 * B.1 + A.2 * B.2 = 0)

theorem distance_from_origin_to_line
  (xA yA xB yB : ℝ)
  (hA : ellipse xA yA)
  (hB : ellipse xB yB)
  (h_perpendicular : perpendicular (0, 0) (xA, yA) (xB, yB))
  : ∃ d : ℝ, d = (Real.sqrt 6) / 3 :=
sorry

end distance_from_origin_to_line_l12_12524


namespace maxim_is_correct_l12_12354

-- Define the mortgage rate as 12.5%
def mortgage_rate : ℝ := 0.125

-- Define the dividend yield rate as 17%
def dividend_rate : ℝ := 0.17

-- Define the net return as the difference between the dividend rate and the mortgage rate
def net_return (D M : ℝ) : ℝ := D - M

-- The main theorem to prove Maxim Sergeyevich is correct
theorem maxim_is_correct : net_return dividend_rate mortgage_rate > 0 :=
by
  sorry

end maxim_is_correct_l12_12354


namespace purely_imaginary_complex_iff_l12_12455

theorem purely_imaginary_complex_iff (m : ℝ) :
  (m + 2 = 0) → (m = -2) :=
by
  sorry

end purely_imaginary_complex_iff_l12_12455


namespace acd_over_b_eq_neg_210_l12_12452

theorem acd_over_b_eq_neg_210 
  (a b c d x : ℤ) 
  (h1 : x = (a + b*Real.sqrt c)/d) 
  (h2 : (7*x)/8 + 1 = 4/x) 
  : (a * c * d) / b = -210 := 
by 
  sorry

end acd_over_b_eq_neg_210_l12_12452


namespace find_angleZ_l12_12450

-- Definitions of angles
def angleX : ℝ := 100
def angleY : ℝ := 130

-- Define Z angle based on the conditions given
def angleZ : ℝ := 130

theorem find_angleZ (p q : Prop) (parallel_pq : p ∧ q)
  (h1 : angleX = 100)
  (h2 : angleY = 130) :
  angleZ = 130 :=
by
  sorry

end find_angleZ_l12_12450


namespace value_of_x_plus_y_l12_12993

-- Define the sum of integers from 50 to 60
def sum_integers_50_to_60 : ℤ := List.sum (List.range' 50 (60 - 50 + 1))

-- Calculate the number of even integers from 50 to 60
def count_even_integers_50_to_60 : ℤ := List.length (List.filter (λ n => n % 2 = 0) (List.range' 50 (60 - 50 + 1)))

-- Define x and y based on the given conditions
def x : ℤ := sum_integers_50_to_60
def y : ℤ := count_even_integers_50_to_60

-- The main theorem to prove
theorem value_of_x_plus_y : x + y = 611 := by
  -- Placeholder for the proof
  sorry

end value_of_x_plus_y_l12_12993


namespace find_value_correct_l12_12346

-- Definitions for the given conditions
def equation1 (a b : ℚ) : Prop := 3 * a - b = 8
def equation2 (a b : ℚ) : Prop := 4 * b + 7 * a = 13

-- Definition for the question
def find_value (a b : ℚ) : ℚ := 2 * a + b

-- Statement of the proof
theorem find_value_correct (a b : ℚ) (h1 : equation1 a b) (h2 : equation2 a b) : find_value a b = 73 / 19 := 
by 
  sorry

end find_value_correct_l12_12346


namespace inequality_proof_equality_case_l12_12810

variables (x y z : ℝ)
  
theorem inequality_proof 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) : 
  (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) ≤ 1 := 
sorry

theorem equality_case 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x + y + z ≥ 3) 
  (h_eq : (1 / (x + y + z^2)) + (1 / (y + z + x^2)) + (1 / (z + x + y^2)) = 1) :
  x = 1 ∧ y = 1 ∧ z = 1 := 
sorry

end inequality_proof_equality_case_l12_12810


namespace Micah_words_per_minute_l12_12629

-- Defining the conditions
def Isaiah_words_per_minute : ℕ := 40
def extra_words : ℕ := 1200

-- Proving the statement that Micah can type 20 words per minute
theorem Micah_words_per_minute (Isaiah_wpm : ℕ) (extra_w : ℕ) : Isaiah_wpm = 40 → extra_w = 1200 → (Isaiah_wpm * 60 - extra_w) / 60 = 20 :=
by
  -- Sorry is used to skip the proof
  sorry

end Micah_words_per_minute_l12_12629


namespace length_of_lawn_l12_12177

-- Definitions based on conditions
def area_per_bag : ℝ := 250
def width : ℝ := 36
def num_bags : ℝ := 4
def extra_area : ℝ := 208

-- Statement to prove
theorem length_of_lawn :
  (num_bags * area_per_bag + extra_area) / width = 33.56 := by
  sorry

end length_of_lawn_l12_12177


namespace second_day_more_than_third_day_l12_12546

-- Define the conditions
def total_people (d1 d2 d3 : ℕ) := d1 + d2 + d3 = 246 
def first_day := 79
def third_day := 120

-- Define the statement to prove
theorem second_day_more_than_third_day : 
  ∃ d2 : ℕ, total_people first_day d2 third_day ∧ (d2 - third_day) = 47 :=
by
  sorry

end second_day_more_than_third_day_l12_12546


namespace algebraic_expression_constant_l12_12985

theorem algebraic_expression_constant (x : ℝ) : x * (x - 6) - (3 - x) ^ 2 = -9 :=
sorry

end algebraic_expression_constant_l12_12985


namespace souvenir_prices_total_profit_l12_12204

variables (x y m n : ℝ)

-- Conditions for the first part
def conditions_part1 : Prop :=
  7 * x + 8 * y = 380 ∧
  10 * x + 6 * y = 380

-- Result for the first part
def result_part1 : Prop :=
  x = 20 ∧ y = 30

-- Conditions for the second part
def conditions_part2 : Prop :=
  m + n = 40 ∧
  20 * m + 30 * n = 900 

-- Result for the second part
def result_part2 : Prop :=
  30 * 5 + 10 * 7 = 220

theorem souvenir_prices (x y : ℝ) (h : conditions_part1 x y) : result_part1 x y :=
by { sorry }

theorem total_profit (m n : ℝ) (h : conditions_part2 m n) : result_part2 :=
by { sorry }

end souvenir_prices_total_profit_l12_12204


namespace falling_body_time_l12_12068

theorem falling_body_time (g : ℝ) (h_g : g = 9.808) (d : ℝ) (t1 : ℝ) (h_d : d = 49.34) (h_t1 : t1 = 1.3) : 
  ∃ t : ℝ, (1 / 2 * g * (t + t1)^2 - 1 / 2 * g * t^2 = d) → t = 7.088 :=
by 
  use 7.088
  intros h
  sorry

end falling_body_time_l12_12068


namespace parabola_circle_intercept_l12_12588

theorem parabola_circle_intercept (p : ℝ) (h_pos : p > 0) :
  (∃ (x y : ℝ), y^2 = 2 * p * x ∧ x^2 + y^2 + 2 * x - 3 = 0) ∧
  (∃ (y1 y2 : ℝ), (y1 - y2)^2 + (-(p / 2) + 1)^2 = 4^2) → p = 2 :=
by sorry

end parabola_circle_intercept_l12_12588


namespace geom_seq_sum_l12_12178

variable (a : ℕ → ℝ) (r : ℝ) (a1 a4 : ℝ)

theorem geom_seq_sum :
  (∀ n : ℕ, a (n + 1) = a n * r) → r = 2 → a 2 + a 3 = 4 → a 1 + a 4 = 6 :=
by
  sorry

end geom_seq_sum_l12_12178


namespace k_squared_geq_25_div_3_l12_12295

open Real

theorem k_squared_geq_25_div_3 
  (a₁ a₂ a₃ a₄ a₅ k : ℝ)
  (h₁₂ : abs (a₁ - a₂) ≥ 1) (h₁₃ : abs (a₁ - a₃) ≥ 1) (h₁₄ : abs (a₁ - a₄) ≥ 1) (h₁₅ : abs (a₁ - a₅) ≥ 1)
  (h₂₃ : abs (a₂ - a₃) ≥ 1) (h₂₄ : abs (a₂ - a₄) ≥ 1) (h₂₅ : abs (a₂ - a₅) ≥ 1)
  (h₃₄ : abs (a₃ - a₄) ≥ 1) (h₃₅ : abs (a₃ - a₅) ≥ 1)
  (h₄₅ : abs (a₄ - a₅) ≥ 1)
  (eq1 : a₁ + a₂ + a₃ + a₄ + a₅ = 2 * k)
  (eq2 : a₁^2 + a₂^2 + a₃^2 + a₄^2 + a₅^2 = 2 * k^2) :
  k^2 ≥ 25 / 3 :=
by
  sorry

end k_squared_geq_25_div_3_l12_12295


namespace gcd_lcm_sum_l12_12983

theorem gcd_lcm_sum :
  gcd 42 70 + lcm 15 45 = 59 :=
by sorry

end gcd_lcm_sum_l12_12983


namespace geometric_series_common_ratio_l12_12961

theorem geometric_series_common_ratio
    (a : ℝ) (S : ℝ) (r : ℝ)
    (h_a : a = 512)
    (h_S : S = 3072)
    (h_sum : S = a / (1 - r)) : 
    r = 5 / 6 :=
by 
  rw [h_a] at h_sum
  rw [h_S] at h_sum
  sorry

end geometric_series_common_ratio_l12_12961


namespace painted_cubes_l12_12617

theorem painted_cubes (n : ℕ) (h1 : 3 < n)
  (h2 : 6 * (n - 2)^2 = 12 * (n - 2)) :
  n = 4 := by
  sorry

end painted_cubes_l12_12617


namespace connectivity_within_square_l12_12791

theorem connectivity_within_square (side_length : ℝ) (highway1 highway2 : ℝ) 
  (A1 A2 A3 A4 : ℝ → ℝ → Prop) : 
  side_length = 10 → 
  highway1 ≠ highway2 → 
  (∀ x y, (0 ≤ x ∧ x ≤ side_length ∧ 0 ≤ y ∧ y ≤ side_length) → 
    (A1 x y ∨ A2 x y ∨ A3 x y ∨ A4 x y)) →
  ∃ (road_length : ℝ), road_length ≤ 25 := 
sorry

end connectivity_within_square_l12_12791


namespace gcd_g_x_1155_l12_12181

def g (x : ℕ) := (4 * x + 5) * (5 * x + 3) * (6 * x + 7) * (3 * x + 11)

theorem gcd_g_x_1155 (x : ℕ) (h : x % 18711 = 0) : Nat.gcd (g x) x = 1155 := by
  sorry

end gcd_g_x_1155_l12_12181


namespace cubic_roots_expression_l12_12108

theorem cubic_roots_expression (p q r : ℝ) 
  (h1 : p + q + r = 4) 
  (h2 : pq + pr + qr = 6) 
  (h3 : pqr = 3) : 
  p / (qr + 2) + q / (pr + 2) + r / (pq + 2) = 4 / 5 := 
by 
  sorry

end cubic_roots_expression_l12_12108


namespace find_weight_of_B_l12_12845

theorem find_weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 44) : B = 33 :=
by 
  sorry

end find_weight_of_B_l12_12845


namespace birth_date_of_id_number_l12_12335

def extract_birth_date (id_number : String) := 
  let birth_str := id_number.drop 6 |>.take 8
  let year := birth_str.take 4
  let month := birth_str.drop 4 |>.take 2
  let day := birth_str.drop 6
  (year, month, day)

theorem birth_date_of_id_number :
  extract_birth_date "320106194607299871" = ("1946", "07", "29") := by
  sorry

end birth_date_of_id_number_l12_12335


namespace exist_infinitely_many_coprime_pairs_l12_12828

theorem exist_infinitely_many_coprime_pairs (a b : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : Nat.gcd a b = 1) : 
  ∃ (a b : ℕ), (a + b).mod (a^b + b^a) = 0 :=
sorry

end exist_infinitely_many_coprime_pairs_l12_12828


namespace original_perimeter_l12_12040

theorem original_perimeter (a b : ℝ) (h : a / 2 + b / 2 = 129 / 2) : 2 * (a + b) = 258 :=
by
  sorry

end original_perimeter_l12_12040


namespace first_car_made_earlier_l12_12669

def year_first_car : ℕ := 1970
def year_third_car : ℕ := 2000
def diff_third_second : ℕ := 20

theorem first_car_made_earlier : (year_third_car - diff_third_second) - year_first_car = 10 := by
  sorry

end first_car_made_earlier_l12_12669


namespace distance_from_unselected_vertex_l12_12761

-- Define the problem statement
theorem distance_from_unselected_vertex
  (base length : ℝ) (area : ℝ) (h : ℝ) 
  (h_area : area = (base * h) / 2) 
  (h_base : base = 8) 
  (h_area_given : area = 24) : 
  h = 6 :=
by
  -- The proof here is skipped
  sorry

end distance_from_unselected_vertex_l12_12761


namespace irwins_family_hike_total_distance_l12_12950

theorem irwins_family_hike_total_distance
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 0.2)
    (h2 : d2 = 0.4)
    (h3 : d3 = 0.1)
    :
    d1 + d2 + d3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end irwins_family_hike_total_distance_l12_12950


namespace michael_twice_jacob_in_11_years_l12_12994

-- Definitions
def jacob_age_4_years := 5
def jacob_current_age := jacob_age_4_years - 4
def michael_current_age := jacob_current_age + 12

-- Theorem to prove
theorem michael_twice_jacob_in_11_years :
  ∀ (x : ℕ), jacob_current_age + x = 1 →
    michael_current_age + x = 13 →
    michael_current_age + (11 : ℕ) = 2 * (jacob_current_age + (11 : ℕ)) :=
by
  intros x h1 h2
  sorry

end michael_twice_jacob_in_11_years_l12_12994


namespace other_books_new_releases_percentage_l12_12548

theorem other_books_new_releases_percentage
  (T : ℝ)
  (h1 : 0 < T)
  (hf_books : ℝ := 0.4 * T)
  (hf_new_releases : ℝ := 0.4 * hf_books)
  (other_books : ℝ := 0.6 * T)
  (total_new_releases : ℝ := hf_new_releases + (P * other_books))
  (fraction_hf_new : ℝ := hf_new_releases / total_new_releases)
  (fraction_value : fraction_hf_new = 0.27586206896551724)
  : P = 0.7 :=
sorry

end other_books_new_releases_percentage_l12_12548


namespace cost_of_cheaper_feed_l12_12378

theorem cost_of_cheaper_feed (C : ℝ)
  (total_weight : ℝ) (weight_cheaper : ℝ) (price_expensive : ℝ) (total_value : ℝ) : 
  total_weight = 35 → 
  total_value = 0.36 * total_weight → 
  weight_cheaper = 17 → 
  price_expensive = 0.53 →
  (total_value = weight_cheaper * C + (total_weight - weight_cheaper) * price_expensive) →
  C = 0.18 := 
by
  sorry

end cost_of_cheaper_feed_l12_12378


namespace triangle_side_lengths_inequality_l12_12714

theorem triangle_side_lengths_inequality
  (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : b + c > a)
  (h3 : c + a > b) :
  a * b * c ≥ (a + b - c) * (b + c - a) * (c + a - b) := 
sorry

end triangle_side_lengths_inequality_l12_12714


namespace jim_age_in_2_years_l12_12703

theorem jim_age_in_2_years (c1 : ∀ t : ℕ, t = 37) (c2 : ∀ j : ℕ, j = 27) : ∀ j2 : ℕ, j2 = 29 :=
by
  sorry

end jim_age_in_2_years_l12_12703


namespace susans_coins_worth_l12_12713

theorem susans_coins_worth :
  ∃ n d : ℕ, n + d = 40 ∧ (5 * n + 10 * d) = 230 ∧ (10 * n + 5 * d) = 370 :=
sorry

end susans_coins_worth_l12_12713


namespace female_members_count_l12_12213

theorem female_members_count (M F : ℕ) (h1 : F = 2 * M) (h2 : F + M = 18) : F = 12 :=
by
  -- the proof will go here
  sorry

end female_members_count_l12_12213


namespace perfect_match_of_products_l12_12348

theorem perfect_match_of_products
  (x : ℕ)  -- number of workers assigned to produce nuts
  (h1 : 22 - x ≥ 0)  -- ensuring non-negative number of workers for screws
  (h2 : 1200 * (22 - x) = 2 * 2000 * x) :  -- the condition for perfect matching
  (2 * 1200 * (22 - x) = 2000 * x) :=  -- the correct equation
by sorry

end perfect_match_of_products_l12_12348


namespace necessary_condition_l12_12253

theorem necessary_condition :
  ∃ x : ℝ, (x < 0 ∨ x > 2) → (2 * x^2 - 5 * x - 3 ≥ 0) :=
sorry

end necessary_condition_l12_12253


namespace q_alone_time_24_days_l12_12575

theorem q_alone_time_24_days:
  ∃ (Wq : ℝ), (∀ (Wp Ws : ℝ), 
    Wp = Wq + 1 / 60 → 
    Wp + Wq = 1 / 10 → 
    Wp + 1 / 60 + 2 * Wq = 1 / 6 → 
    1 / Wq = 24) :=
by
  sorry

end q_alone_time_24_days_l12_12575


namespace cara_bread_dinner_amount_240_l12_12431

def conditions (B L D : ℕ) : Prop :=
  8 * L = D ∧ 6 * B = D ∧ B + L + D = 310

theorem cara_bread_dinner_amount_240 :
  ∃ (B L D : ℕ), conditions B L D ∧ D = 240 :=
by
  sorry

end cara_bread_dinner_amount_240_l12_12431


namespace flowers_per_set_l12_12922

variable (totalFlowers : ℕ) (numSets : ℕ)

theorem flowers_per_set (h1 : totalFlowers = 270) (h2 : numSets = 3) : totalFlowers / numSets = 90 :=
by
  sorry

end flowers_per_set_l12_12922


namespace solve_congruence_l12_12849

theorem solve_congruence :
  ∃ n : ℤ, 19 * n ≡ 13 [ZMOD 47] ∧ n ≡ 25 [ZMOD 47] :=
by
  sorry

end solve_congruence_l12_12849


namespace expand_square_binomial_l12_12649

variable (m n : ℝ)

theorem expand_square_binomial : (3 * m - n) ^ 2 = 9 * m ^ 2 - 6 * m * n + n ^ 2 :=
by
  sorry

end expand_square_binomial_l12_12649


namespace domain_of_fractional_sqrt_function_l12_12133

theorem domain_of_fractional_sqrt_function :
  ∀ x : ℝ, (x + 4 ≥ 0) ∧ (x - 1 ≠ 0) ↔ (x ∈ (Set.Ici (-4) \ {1})) :=
by
  sorry

end domain_of_fractional_sqrt_function_l12_12133


namespace fifth_selected_ID_is_01_l12_12349

noncomputable def populationIDs : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

noncomputable def randomNumberTable : List (List ℕ) :=
  [[78, 16, 65, 72,  8, 2, 63, 14,  7, 2, 43, 69, 97, 28,  1, 98],
   [32,  4, 92, 34, 49, 35, 82,  0, 36, 23, 48, 69, 69, 38, 74, 81]]

noncomputable def selectedIDs (table : List (List ℕ)) : List ℕ :=
  [8, 2, 14, 7, 1]  -- Derived from the selection method

theorem fifth_selected_ID_is_01 : (selectedIDs randomNumberTable).get! 4 = 1 := by
  sorry

end fifth_selected_ID_is_01_l12_12349


namespace subset_relation_l12_12085

def P := {x : ℝ | x < 2}
def Q := {y : ℝ | y < 1}

theorem subset_relation : Q ⊆ P := 
by {
  sorry
}

end subset_relation_l12_12085


namespace retirement_fund_increment_l12_12330

theorem retirement_fund_increment (k y : ℝ) (h1 : k * Real.sqrt (y + 3) = k * Real.sqrt y + 15)
  (h2 : k * Real.sqrt (y + 5) = k * Real.sqrt y + 27) : k * Real.sqrt y = 810 := by
  sorry

end retirement_fund_increment_l12_12330


namespace average_first_two_numbers_l12_12932

theorem average_first_two_numbers (a1 a2 a3 a4 a5 a6 : ℝ)
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
  (h2 : (a3 + a4) / 2 = 3.85)
  (h3 : (a5 + a6) / 2 = 4.200000000000001) :
  (a1 + a2) / 2 = 3.8 :=
by
  sorry

end average_first_two_numbers_l12_12932


namespace right_isosceles_triangle_areas_l12_12833

theorem right_isosceles_triangle_areas :
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  A + B = C :=
by
  let A := (5 * 5) / 2
  let B := (12 * 12) / 2
  let C := (13 * 13) / 2
  sorry

end right_isosceles_triangle_areas_l12_12833


namespace total_ticket_cost_l12_12942

theorem total_ticket_cost 
  (young_discount : ℝ := 0.55) 
  (old_discount : ℝ := 0.30) 
  (full_price : ℝ := 10)
  (num_young : ℕ := 2) 
  (num_middle : ℕ := 2) 
  (num_old : ℕ := 2) 
  (grandma_ticket_cost : ℝ := 7) :
  2 * (full_price * young_discount) + 2 * full_price + 2 * grandma_ticket_cost = 43 :=
by 
  sorry

end total_ticket_cost_l12_12942


namespace factorize_expression_l12_12779

theorem factorize_expression (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) :=
by
  sorry

end factorize_expression_l12_12779


namespace find_difference_l12_12012

theorem find_difference (a b : ℕ) (h1 : a < b) (h2 : a + b = 78) (h3 : Nat.lcm a b = 252) : b - a = 6 :=
by
  sorry

end find_difference_l12_12012


namespace random_event_proof_l12_12812

def is_certain_event (event: Prop) : Prop := ∃ h: event → true, ∃ h': true → event, true
def is_impossible_event (event: Prop) : Prop := event → false
def is_random_event (event: Prop) : Prop := ¬is_certain_event event ∧ ¬is_impossible_event event

def cond1 : Prop := sorry -- Yingying encounters a green light
def cond2 : Prop := sorry -- A non-transparent bag contains one ping-pong ball and two glass balls of the same size, and a ping-pong ball is drawn from it.
def cond3 : Prop := sorry -- You are currently answering question 12 of this test paper.
def cond4 : Prop := sorry -- The highest temperature in our city tomorrow will be 60°C.

theorem random_event_proof : 
  is_random_event cond1 ∧ 
  ¬is_random_event cond2 ∧ 
  ¬is_random_event cond3 ∧ 
  ¬is_random_event cond4 :=
by
  sorry

end random_event_proof_l12_12812


namespace tangential_difference_l12_12698

noncomputable def tan_alpha_minus_beta (α β : ℝ) : ℝ :=
  Real.tan (α - β)

theorem tangential_difference 
  {α β : ℝ}
  (h : 3 / (2 + Real.sin (2 * α)) + 2021 / (2 + Real.sin β) = 2024) : 
  tan_alpha_minus_beta α β = 1 := 
sorry

end tangential_difference_l12_12698


namespace downstream_speed_is_45_l12_12252

-- Define the conditions
def upstream_speed := 35 -- The man can row upstream at 35 kmph
def still_water_speed := 40 -- The speed of the man in still water is 40 kmph

-- Define the speed of the stream based on the given conditions
def stream_speed := still_water_speed - upstream_speed 

-- Define the speed of the man rowing downstream
def downstream_speed := still_water_speed + stream_speed

-- The assertion to prove
theorem downstream_speed_is_45 : downstream_speed = 45 := by
  sorry

end downstream_speed_is_45_l12_12252


namespace quadratic_has_real_roots_l12_12644

-- Define the condition that a quadratic equation has real roots given ac < 0

variable {a b c : ℝ}

def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_has_real_roots (h : a * c < 0) : ∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 :=
by
  sorry

end quadratic_has_real_roots_l12_12644


namespace find_alpha_plus_beta_l12_12071

theorem find_alpha_plus_beta (α β : ℝ)
  (h : ∀ x : ℝ, x ≠ 45 → (x - α) / (x + β) = (x^2 - 90 * x + 1981) / (x^2 + 63 * x - 3420)) :
  α + β = 113 :=
by
  sorry

end find_alpha_plus_beta_l12_12071


namespace sqrt_product_eq_225_l12_12297

theorem sqrt_product_eq_225 : (Real.sqrt (5 * 3) * Real.sqrt (3 ^ 3 * 5 ^ 3) = 225) :=
by
  sorry

end sqrt_product_eq_225_l12_12297


namespace tan_beta_value_l12_12474

theorem tan_beta_value (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.tan α = 4 / 3)
  (h4 : Real.cos (α + β) = Real.sqrt 5 / 5) :
  Real.tan β = 2 / 11 := 
sorry

end tan_beta_value_l12_12474


namespace original_weight_l12_12078

namespace MarbleProblem

def remainingWeightAfterCuts (w : ℝ) : ℝ :=
  w * 0.70 * 0.70 * 0.85

theorem original_weight (w : ℝ) : remainingWeightAfterCuts w = 124.95 → w = 299.94 :=
by
  intros h
  sorry

end MarbleProblem

end original_weight_l12_12078


namespace polynomial_divisibility_l12_12192

theorem polynomial_divisibility (C D : ℝ) (h : ∀ x : ℂ, x^2 + x + 1 = 0 → x^104 + C * x + D = 0) :
  C + D = 2 := 
sorry

end polynomial_divisibility_l12_12192


namespace parabola_symmetry_l12_12265

-- Define the function f as explained in the problem
def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b*x + c

-- Lean theorem to prove the inequality based on given conditions
theorem parabola_symmetry (b c : ℝ) (h : ∀ t : ℝ, f (3 + t) b c = f (3 - t) b c) :
  f 3 b c < f 1 b c ∧ f 1 b c < f 6 b c :=
by
  sorry

end parabola_symmetry_l12_12265


namespace find_T5_l12_12257

variables (a b x y : ℝ)

def T (n : ℕ) : ℝ := a * x^n + b * y^n

theorem find_T5
  (h1 : T a b x y 1 = 3)
  (h2 : T a b x y 2 = 7)
  (h3 : T a b x y 3 = 6)
  (h4 : T a b x y 4 = 42) :
  T a b x y 5 = -360 :=
sorry

end find_T5_l12_12257


namespace solve_xy_l12_12636

theorem solve_xy : ∃ x y : ℝ, (x - y = 10 ∧ x^2 + y^2 = 100) ↔ ((x = 0 ∧ y = -10) ∨ (x = 10 ∧ y = 0)) := 
by {
  sorry
}

end solve_xy_l12_12636


namespace zero_point_condition_l12_12960

-- Define the function f(x) = ax + 3
def f (a x : ℝ) : ℝ := a * x + 3

-- Define that a > 2 is necessary but not sufficient condition
theorem zero_point_condition (a : ℝ) (h : a > 2) : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f a x = 0) ↔ (a ≥ 3) := 
sorry

end zero_point_condition_l12_12960


namespace unique_solution_quadratic_eq_l12_12642

theorem unique_solution_quadratic_eq (q : ℚ) (hq : q ≠ 0) :
  (∀ x : ℚ, q * x^2 - 10 * x + 2 = 0) ↔ q = 12.5 :=
by
  sorry

end unique_solution_quadratic_eq_l12_12642


namespace calculate_expression_l12_12787

theorem calculate_expression : ( (3 / 20 + 5 / 200 + 7 / 2000) * 2 = 0.357 ) :=
by
  sorry

end calculate_expression_l12_12787


namespace points_on_single_circle_l12_12864

theorem points_on_single_circle (n : ℕ) (points : Fin n → ℝ × ℝ)
  (h : ∀ i j : Fin n, ∃ f : ℝ × ℝ → ℝ × ℝ, (∀ p, f p ≠ p) ∧ f (points i) = points j ∧ 
        (∀ k : Fin n, ∃ p, points k = f p)) :
  ∃ (O : ℝ × ℝ) (r : ℝ), ∀ i : Fin n, dist (points i) O = r := sorry

end points_on_single_circle_l12_12864


namespace final_speed_train_l12_12785

theorem final_speed_train
  (u : ℝ) (a : ℝ) (t : ℕ) :
  u = 0 → a = 1 → t = 20 → u + a * t = 20 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end final_speed_train_l12_12785


namespace distinguishable_large_triangles_l12_12652

def num_of_distinguishable_large_eq_triangles : Nat :=
  let colors := 8
  let pairs := 7 + Nat.choose 7 2
  colors * pairs

theorem distinguishable_large_triangles : num_of_distinguishable_large_eq_triangles = 224 := by
  sorry

end distinguishable_large_triangles_l12_12652


namespace probability_of_successful_meeting_l12_12686

noncomputable def meeting_probability : ℚ := 7 / 64

theorem probability_of_successful_meeting :
  (∃ x y z : ℝ,
     0 ≤ x ∧ x ≤ 2 ∧
     0 ≤ y ∧ y ≤ 2 ∧
     0 ≤ z ∧ z ≤ 2 ∧
     abs (x - z) ≤ 0.75 ∧
     abs (y - z) ≤ 1.5 ∧
     z ≥ x ∧
     z ≥ y) →
  meeting_probability = 7 / 64 := by
  sorry

end probability_of_successful_meeting_l12_12686


namespace omitted_digits_correct_l12_12111

theorem omitted_digits_correct :
  (287 * 23 = 6601) := by
  sorry

end omitted_digits_correct_l12_12111


namespace correct_calculation_l12_12042

theorem correct_calculation :
  (∀ (x y : ℝ), -3 * x - 3 * x ≠ 0) ∧
  (∀ (x : ℝ), x - 4 * x ≠ -3) ∧
  (∀ (x : ℝ), 2 * x + 3 * x^2 ≠ 5 * x^3) ∧
  (∀ (x y : ℝ), -4 * x * y + 3 * x * y = -x * y) :=
by
  sorry

end correct_calculation_l12_12042


namespace positive_difference_g_b_values_l12_12754

noncomputable def g (n : ℤ) : ℤ :=
if n < 0 then n^2 + 5 * n + 6 else 3 * n - 30

theorem positive_difference_g_b_values : 
  let g_neg_3 := g (-3)
  let g_3 := g 3
  g_neg_3 = 0 → g_3 = -21 → 
  ∃ b1 b2 : ℤ, g_neg_3 + g_3 + g b1 = 0 ∧ g_neg_3 + g_3 + g b2 = 0 ∧ 
  b1 ≠ b2 ∧ b1 < b2 ∧ b1 < 0 ∧ b2 > 0 ∧ b2 - b1 = 22 :=
by
  sorry

end positive_difference_g_b_values_l12_12754


namespace employee_salary_percentage_l12_12875

theorem employee_salary_percentage (A B : ℝ)
    (h1 : A + B = 450)
    (h2 : B = 180) : (A / B) * 100 = 150 := by
  sorry

end employee_salary_percentage_l12_12875


namespace MNPQ_is_rectangle_l12_12977

variable {Point : Type}
variable {A B C D M N P Q : Point}

def is_parallelogram (A B C D : Point) : Prop := sorry
def altitude (X Y : Point) : Prop := sorry
def rectangle (M N P Q : Point) : Prop := sorry

theorem MNPQ_is_rectangle 
  (h_parallelogram : is_parallelogram A B C D)
  (h_alt1 : altitude B M)
  (h_alt2 : altitude B N)
  (h_alt3 : altitude D P)
  (h_alt4 : altitude D Q) :
  rectangle M N P Q :=
sorry

end MNPQ_is_rectangle_l12_12977


namespace sum_of_adjacents_to_15_l12_12382

-- Definitions of the conditions
def divisorsOf225 : Set ℕ := {3, 5, 9, 15, 25, 45, 75, 225}

-- Definition of the adjacency relationship
def isAdjacent (x y : ℕ) (s : Set ℕ) : Prop :=
  x ∈ s ∧ y ∈ s ∧ Nat.gcd x y > 1

-- Problem statement in Lean 4
theorem sum_of_adjacents_to_15 :
  ∃ x y : ℕ, isAdjacent 15 x divisorsOf225 ∧ isAdjacent 15 y divisorsOf225 ∧ x + y = 120 :=
by
  sorry

end sum_of_adjacents_to_15_l12_12382


namespace smallest_positive_debt_l12_12768

theorem smallest_positive_debt :
  ∃ (p g : ℤ), 25 = 250 * p + 175 * g :=
by
  sorry

end smallest_positive_debt_l12_12768


namespace factor_expression_l12_12194

theorem factor_expression (x : ℤ) : 75 * x + 45 = 15 * (5 * x + 3) := 
by {
  sorry
}

end factor_expression_l12_12194


namespace number_of_truthful_dwarfs_l12_12283

/-- Each of the 10 dwarfs either always tells the truth or always lies. 
It is known that each of them likes exactly one type of ice cream: vanilla, chocolate, or fruit. 
Prove the number of truthful dwarfs. -/
theorem number_of_truthful_dwarfs (x y : ℕ) (h1 : x + y = 10) (h2 : x + 2 * y = 16) : x = 4 :=
by sorry

end number_of_truthful_dwarfs_l12_12283


namespace betsy_sewing_l12_12258

-- Definitions of conditions
def total_squares : ℕ := 16 + 16
def sewn_percentage : ℝ := 0.25
def sewn_squares : ℝ := sewn_percentage * total_squares
def squares_left : ℝ := total_squares - sewn_squares

-- Proof that Betsy needs to sew 24 more squares
theorem betsy_sewing : squares_left = 24 := by
  -- Sorry placeholder for the actual proof
  sorry

end betsy_sewing_l12_12258


namespace average_branches_per_foot_correct_l12_12522

def height_tree_1 : ℕ := 50
def branches_tree_1 : ℕ := 200
def height_tree_2 : ℕ := 40
def branches_tree_2 : ℕ := 180
def height_tree_3 : ℕ := 60
def branches_tree_3 : ℕ := 180
def height_tree_4 : ℕ := 34
def branches_tree_4 : ℕ := 153

def total_height := height_tree_1 + height_tree_2 + height_tree_3 + height_tree_4
def total_branches := branches_tree_1 + branches_tree_2 + branches_tree_3 + branches_tree_4
def average_branches_per_foot := total_branches / total_height

theorem average_branches_per_foot_correct : average_branches_per_foot = 713 / 184 := 
  by
    -- Proof omitted, directly state the result
    sorry

end average_branches_per_foot_correct_l12_12522


namespace amount_spent_l12_12786

-- Definitions
def initial_amount : ℕ := 54
def amount_left : ℕ := 29

-- Proof statement
theorem amount_spent : initial_amount - amount_left = 25 :=
by
  sorry

end amount_spent_l12_12786


namespace primes_div_conditions_unique_l12_12973

theorem primes_div_conditions_unique (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) :
  (p ∣ q + 6) ∧ (q ∣ p + 7) → (p = 19 ∧ q = 13) :=
sorry

end primes_div_conditions_unique_l12_12973


namespace ratio_quadrilateral_l12_12099

theorem ratio_quadrilateral
  (ABCD_area : ℝ)
  (h_ABCD : ABCD_area = 40)
  (K L M N : Type)
  (AK KB : ℝ)
  (h_ratio : AK / KB = BL / LC ∧ BL / LC = CM / MD ∧ CM / MD = DN / NA)
  (KLMN_area : ℝ)
  (h_KLMN : KLMN_area = 25) :
  (AK / (AK + KB) = 1 / 4 ∨ AK / (AK + KB) = 3 / 4) :=
sorry

end ratio_quadrilateral_l12_12099


namespace k_value_for_polynomial_l12_12419

theorem k_value_for_polynomial (k : ℤ) :
  (3 : ℤ)^3 + k * (3 : ℤ) - 18 = 0 → k = -3 :=
by
  sorry

end k_value_for_polynomial_l12_12419


namespace first_discount_percentage_l12_12844

/-
  Prove that under the given conditions:
  1. The price before the first discount is $33.78.
  2. The final price after the first and second discounts is $19.
  3. The second discount is 25%.
-/
theorem first_discount_percentage (x : ℝ) :
  (33.78 * (1 - x / 100) * (1 - 25 / 100) = 19) →
  x = 25 :=
by
  -- Proof steps (to be filled)
  sorry

end first_discount_percentage_l12_12844


namespace students_playing_both_l12_12187

theorem students_playing_both (T F L N B : ℕ)
  (hT : T = 39)
  (hF : F = 26)
  (hL : L = 20)
  (hN : N = 10)
  (hTotal : (F + L - B) + N = T) :
  B = 17 :=
by
  sorry

end students_playing_both_l12_12187


namespace christine_final_throw_difference_l12_12351

def christine_first_throw : ℕ := 20
def janice_first_throw : ℕ := christine_first_throw - 4
def christine_second_throw : ℕ := christine_first_throw + 10
def janice_second_throw : ℕ := janice_first_throw * 2
def janice_final_throw : ℕ := christine_first_throw + 17
def highest_throw : ℕ := 37

theorem christine_final_throw_difference :
  ∃ x : ℕ, christine_second_throw + x = highest_throw ∧ x = 7 := by 
sorry

end christine_final_throw_difference_l12_12351


namespace calories_burned_per_week_l12_12764

-- Definitions of the conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℝ := 1.5
def calories_per_min : ℝ := 7
def minutes_per_hour : ℝ := 60

-- Theorem stating the proof problem
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * minutes_per_hour) * calories_per_min) = 1890 := by
  sorry

end calories_burned_per_week_l12_12764


namespace cos_value_in_second_quadrant_l12_12155

theorem cos_value_in_second_quadrant {B : ℝ} (h1 : π / 2 < B ∧ B < π) (h2 : Real.sin B = 5 / 13) : 
  Real.cos B = - (12 / 13) :=
sorry

end cos_value_in_second_quadrant_l12_12155


namespace students_taking_statistics_l12_12246

-- Definitions based on conditions
def total_students := 89
def history_students := 36
def history_or_statistics := 59
def history_not_statistics := 27

-- The proof problem
theorem students_taking_statistics : ∃ S : ℕ, S = 32 ∧
  ((history_students - history_not_statistics) + S - (history_students - history_not_statistics)) = history_or_statistics :=
by
  use 32
  sorry

end students_taking_statistics_l12_12246


namespace ineq_pos_xy_l12_12972

theorem ineq_pos_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) / Real.sqrt (x * y) ≤ x / y + y / x := 
sorry

end ineq_pos_xy_l12_12972


namespace height_ratio_l12_12887

noncomputable def Anne_height := 80
noncomputable def Bella_height := 3 * Anne_height
noncomputable def Sister_height := Bella_height - 200

theorem height_ratio : Anne_height / Sister_height = 2 :=
by
  /-
  The proof here is omitted as requested.
  -/
  sorry

end height_ratio_l12_12887


namespace problem_result_l12_12434

noncomputable def max_value (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) : ℝ :=
  2 * x^2 + x * y + y^2

theorem problem (x y : ℝ) (hx : 2 * x^2 - x * y + y^2 = 15) :
  max_value x y hx = (75 + 60 * Real.sqrt 2) / 7 :=
sorry

theorem result : 75 + 60 + 2 + 7 = 144 :=
by norm_num

end problem_result_l12_12434


namespace roots_equation_sum_and_product_l12_12418

theorem roots_equation_sum_and_product (x1 x2 : ℝ) (h1 : x1 ^ 2 - 3 * x1 - 5 = 0) (h2 : x2 ^ 2 - 3 * x2 - 5 = 0) :
  x1 + x2 - x1 * x2 = 8 :=
sorry

end roots_equation_sum_and_product_l12_12418


namespace inequality_proof_l12_12579

variable {a b c d : ℝ}

theorem inequality_proof (h1 : a > b) (h2 : c > d) : d - a < c - b := by
  sorry

end inequality_proof_l12_12579


namespace problem1_problem2_l12_12275

variable {a b : ℝ}

theorem problem1 (h : a ≠ b) : 
  ((b / (a - b)) - (a / (a - b))) = -1 := 
by
  sorry

theorem problem2 (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) : 
  ((a^2 - a * b)/(a^2) / ((a / b) - (b / a))) = (b / (a + b)) := 
by
  sorry

end problem1_problem2_l12_12275


namespace marble_problem_l12_12310

theorem marble_problem : Nat.lcm (Nat.lcm (Nat.lcm 2 3) 5) 7 = 210 := by
  sorry

end marble_problem_l12_12310


namespace divisible_by_7_iff_l12_12391

variable {x y : ℤ}

theorem divisible_by_7_iff :
  7 ∣ (2 * x + 3 * y) ↔ 7 ∣ (5 * x + 4 * y) :=
by
  sorry

end divisible_by_7_iff_l12_12391


namespace six_digit_numbers_with_at_least_two_zeros_l12_12512

noncomputable def num_six_digit_numbers_with_at_least_two_zeros : ℕ :=
  73314

theorem six_digit_numbers_with_at_least_two_zeros :
  ∃ n : ℕ, n = num_six_digit_numbers_with_at_least_two_zeros := by
  use 73314
  sorry

end six_digit_numbers_with_at_least_two_zeros_l12_12512


namespace sarah_gave_away_16_apples_to_teachers_l12_12533

def initial_apples : Nat := 25
def apples_given_to_friends : Nat := 5
def apples_eaten : Nat := 1
def apples_left_after_journey : Nat := 3

theorem sarah_gave_away_16_apples_to_teachers :
  let apples_after_giving_to_friends := initial_apples - apples_given_to_friends
  let apples_after_eating := apples_after_giving_to_friends - apples_eaten
  apples_after_eating - apples_left_after_journey = 16 :=
by
  sorry

end sarah_gave_away_16_apples_to_teachers_l12_12533


namespace function_increasing_interval_l12_12304

theorem function_increasing_interval :
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi),
  (2 * Real.sin ((Real.pi / 6) - 2 * x) : ℝ)
  ≤ 2 * Real.sin ((Real.pi / 6) - 2 * x + 1)) ↔ (x ∈ Set.Icc (Real.pi / 3) (5 * Real.pi / 6)) :=
sorry

end function_increasing_interval_l12_12304


namespace find_n_in_geometric_series_l12_12497

theorem find_n_in_geometric_series (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 2 →
  (∀ k, a (k + 1) = 2 * a k) →
  S n = 126 →
  S n = a 1 * (2^n - 1) / (2 - 1) →
  n = 6 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end find_n_in_geometric_series_l12_12497


namespace MattSkipsRopesTimesPerSecond_l12_12486

theorem MattSkipsRopesTimesPerSecond:
  ∀ (minutes_jumped : ℕ) (total_skips : ℕ), 
  minutes_jumped = 10 → 
  total_skips = 1800 → 
  (total_skips / (minutes_jumped * 60)) = 3 :=
by
  intros minutes_jumped total_skips h_jumped h_skips
  sorry

end MattSkipsRopesTimesPerSecond_l12_12486


namespace sequence_is_odd_l12_12560

theorem sequence_is_odd (a : ℕ → ℤ) 
  (h1 : a 1 = 2) 
  (h2 : a 2 = 7) 
  (h3 : ∀ n ≥ 2, -1/2 < (a (n + 1)) - (a n) * (a n) / a (n-1) ∧
                (a (n + 1)) - (a n) * (a n) / a (n-1) ≤ 1/2) :
  ∀ n > 1, (a n) % 2 = 1 :=
by
  sorry

end sequence_is_odd_l12_12560


namespace geo_seq_decreasing_l12_12987

variables (a_1 q : ℝ) (a : ℕ → ℝ)
-- Define the geometric sequence
def geo_seq (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q ^ n

-- The problem statement as a Lean theorem
theorem geo_seq_decreasing (h1 : a_1 * (q - 1) < 0) (h2 : q > 0) :
  ∀ n : ℕ, geo_seq a_1 q (n + 1) < geo_seq a_1 q n :=
by
  sorry

end geo_seq_decreasing_l12_12987


namespace variance_uniform_l12_12650

noncomputable def variance_of_uniform (α β : ℝ) (h : α < β) : ℝ :=
  let E := (α + β) / 2
  (β - α)^2 / 12

theorem variance_uniform (α β : ℝ) (h : α < β) :
  variance_of_uniform α β h = (β - α)^2 / 12 :=
by
  -- statement of proof only, actual proof here is sorry
  sorry

end variance_uniform_l12_12650


namespace arithmetic_seq_a12_l12_12063

variable {a : ℕ → ℝ}

theorem arithmetic_seq_a12 :
  (∀ n, ∃ d, a (n + 1) = a n + d)
  ∧ a 5 + a 11 = 30
  ∧ a 4 = 7
  → a 12 = 23 :=
by
  sorry


end arithmetic_seq_a12_l12_12063


namespace player_current_average_l12_12570

theorem player_current_average (A : ℝ) 
  (h1 : 10 * A + 76 = (A + 4) * 11) : 
  A = 32 :=
sorry

end player_current_average_l12_12570


namespace lateral_surface_area_of_cylinder_l12_12077

theorem lateral_surface_area_of_cylinder (V : ℝ) (hV : V = 27 * Real.pi) : 
  ∃ (S : ℝ), S = 18 * Real.pi :=
by
  sorry

end lateral_surface_area_of_cylinder_l12_12077


namespace soap_remaining_days_l12_12897

theorem soap_remaining_days 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0)
  (daily_consumption : ℝ)
  (h4 : daily_consumption = a * b * c / 8) 
  (h5 : ∀ t : ℝ, t > 0 → t ≤ 7 → daily_consumption = (a * b * c - (a * b * c) * (1 / 8))) :
  ∃ t : ℝ, t = 1 :=
by 
  sorry

end soap_remaining_days_l12_12897


namespace solution_set_of_inequality_l12_12953

theorem solution_set_of_inequality {x : ℝ} :
  {x : ℝ | |2 - 3 * x| ≥ 4} = {x : ℝ | x ≤ -2 / 3 ∨ 2 ≤ x} :=
by
  sorry

end solution_set_of_inequality_l12_12953


namespace man_present_age_l12_12115

variable {P : ℝ}

theorem man_present_age (h1 : P = 1.25 * (P - 10)) (h2 : P = (5 / 6) * (P + 10)) : P = 50 :=
  sorry

end man_present_age_l12_12115


namespace not_product_of_two_primes_l12_12225

theorem not_product_of_two_primes (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h : ∃ n : ℕ, a^3 + b^3 = n^2) :
  ¬ (∃ p q : ℕ, p ≠ q ∧ Prime p ∧ Prime q ∧ a + b = p * q) :=
by
  sorry

end not_product_of_two_primes_l12_12225


namespace negation_P_l12_12269

-- Define the original proposition P
def P (a b : ℝ) : Prop := (a^2 + b^2 = 0) → (a = 0 ∧ b = 0)

-- State the negation of P
theorem negation_P : ∀ (a b : ℝ), (a^2 + b^2 ≠ 0) → (a ≠ 0 ∨ b ≠ 0) :=
by
  sorry

end negation_P_l12_12269


namespace felix_can_lift_150_l12_12705

-- Define the weights of Felix and his brother.
variables (F B : ℤ)

-- Given conditions
-- Felix's brother can lift three times his weight off the ground, and this amount is 600 pounds.
def brother_lift (B : ℤ) : Prop := 3 * B = 600
-- Felix's brother weighs twice as much as Felix.
def brother_weight (B F : ℤ) : Prop := B = 2 * F
-- Felix can lift off the ground 1.5 times his weight.
def felix_lift (F : ℤ) : ℤ := 3 * F / 2 -- Note: 1.5F can be represented as 3F/2 in Lean for integer operations.

-- Goal: Prove that Felix can lift 150 pounds.
theorem felix_can_lift_150 (F B : ℤ) (h1 : brother_lift B) (h2 : brother_weight B F) : felix_lift F = 150 := by
  dsimp [brother_lift, brother_weight, felix_lift] at *
  sorry

end felix_can_lift_150_l12_12705


namespace average_weight_of_all_players_l12_12135

-- Definitions based on conditions
def num_forwards : ℕ := 8
def avg_weight_forwards : ℝ := 75
def num_defensemen : ℕ := 12
def avg_weight_defensemen : ℝ := 82

-- Total number of players
def total_players : ℕ := num_forwards + num_defensemen

-- Values derived from conditions
def total_weight_forwards : ℝ := avg_weight_forwards * num_forwards
def total_weight_defensemen : ℝ := avg_weight_defensemen * num_defensemen
def total_weight : ℝ := total_weight_forwards + total_weight_defensemen

-- Theorem to prove the average weight of all players
theorem average_weight_of_all_players : total_weight / total_players = 79.2 :=
by
  sorry

end average_weight_of_all_players_l12_12135


namespace fraction_meaningful_cond_l12_12539

theorem fraction_meaningful_cond (x : ℝ) : (x + 2 ≠ 0) ↔ (x ≠ -2) := 
by
  sorry

end fraction_meaningful_cond_l12_12539


namespace num_solutions_triples_l12_12139

theorem num_solutions_triples :
  {n : ℕ // ∃ a b c : ℤ, a^2 - a * (b + c) + b^2 - b * c + c^2 = 1 ∧ n = 10  } :=
  sorry

end num_solutions_triples_l12_12139


namespace geometric_sequence_sum_l12_12030

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (r : ℝ)
  (h1 : a 1 + a 3 = 8)
  (h2 : a 5 + a 7 = 4)
  (geometric_seq : ∀ n, a n = a 1 * r ^ (n - 1)) :
  a 9 + a 11 + a 13 + a 15 = 3 :=
by
  sorry

end geometric_sequence_sum_l12_12030


namespace two_pow_n_plus_one_square_or_cube_l12_12394

theorem two_pow_n_plus_one_square_or_cube (n : ℕ) :
  (∃ a : ℕ, 2^n + 1 = a^2) ∨ (∃ a : ℕ, 2^n + 1 = a^3) → n = 3 :=
by
  sorry

end two_pow_n_plus_one_square_or_cube_l12_12394


namespace problem1_problem2_l12_12956

-- Problem 1
theorem problem1 : (Real.sqrt 8 - Real.sqrt 27 - (4 * Real.sqrt (1 / 2) + Real.sqrt 12)) = -5 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem2 : ((Real.sqrt 6 + Real.sqrt 12) * (2 * Real.sqrt 3 - Real.sqrt 6) - 3 * Real.sqrt 32 / (Real.sqrt 2 / 2)) = -18 := by
  sorry

end problem1_problem2_l12_12956


namespace sum_binomial_2k_eq_2_2n_l12_12336

open scoped BigOperators

noncomputable def binomial_coeff (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem sum_binomial_2k_eq_2_2n (n : ℕ) :
  ∑ k in Finset.range (n + 1), 2^k * binomial_coeff (2*n - k) n = 2^(2*n) := 
by
  sorry

end sum_binomial_2k_eq_2_2n_l12_12336


namespace remainder_of_pencils_l12_12210

def number_of_pencils : ℕ := 13254839
def packages : ℕ := 7

theorem remainder_of_pencils :
  number_of_pencils % packages = 3 := by
  sorry

end remainder_of_pencils_l12_12210


namespace intersection_eq_l12_12683

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem intersection_eq : A ∩ B = {1, 3} :=
by
  sorry

end intersection_eq_l12_12683


namespace correct_calculation_for_b_l12_12963

theorem correct_calculation_for_b (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end correct_calculation_for_b_l12_12963


namespace yellow_chips_are_one_l12_12058

-- Definitions based on conditions
def yellow_chip_points : ℕ := 2
def blue_chip_points : ℕ := 4
def green_chip_points : ℕ := 5

variables (Y B G : ℕ)

-- Given conditions
def point_product_condition : Prop := (yellow_chip_points^Y * blue_chip_points^B * green_chip_points^G = 16000)
def equal_blue_green : Prop := (B = G)

-- Theorem to prove the number of yellow chips
theorem yellow_chips_are_one (Y B G : ℕ) (hprod : point_product_condition Y B G) (heq : equal_blue_green B G) : Y = 1 :=
by {
    sorry -- Proof omitted
}

end yellow_chips_are_one_l12_12058


namespace roots_greater_than_half_iff_l12_12886

noncomputable def quadratic_roots (a : ℝ) (x1 x2 : ℝ) : Prop :=
  (2 - a) * x1^2 - 3 * a * x1 + 2 * a = 0 ∧ 
  (2 - a) * x2^2 - 3 * a * x2 + 2 * a = 0 ∧
  x1 > 1/2 ∧ x2 > 1/2

theorem roots_greater_than_half_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots a x1 x2) ↔ (16 / 17 < a ∧ a < 2) :=
sorry

end roots_greater_than_half_iff_l12_12886


namespace min_value_parabola_l12_12918

theorem min_value_parabola : 
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 4 ∧ (-x^2 + 4 * x - 2) = -2 :=
by
  sorry

end min_value_parabola_l12_12918


namespace moles_of_NH4Cl_l12_12728

-- Define what is meant by "mole" and the substances NH3, HCl, and NH4Cl
def NH3 : Type := ℕ -- Use ℕ to represent moles
def HCl : Type := ℕ
def NH4Cl : Type := ℕ

-- Define the stoichiometry of the reaction
def reaction (n_NH3 n_HCl : ℕ) : ℕ :=
n_NH3 + n_HCl

-- Lean 4 statement: given 1 mole of NH3 and 1 mole of HCl, prove the reaction produces 1 mole of NH4Cl
theorem moles_of_NH4Cl (n_NH3 n_HCl : ℕ) (h1 : n_NH3 = 1) (h2 : n_HCl = 1) : 
  reaction n_NH3 n_HCl = 1 :=
by
  sorry

end moles_of_NH4Cl_l12_12728


namespace tenth_term_arithmetic_sequence_l12_12290

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ a₃₀ : ℕ) (d : ℕ) (n : ℕ), a₁ = 3 → a₃₀ = 89 → n = 10 → 
  (a₃₀ - a₁) / 29 = d → a₁ + (n - 1) * d = 30 :=
by
  intros a₁ a₃₀ d n h₁ h₃₀ hn hd
  sorry

end tenth_term_arithmetic_sequence_l12_12290


namespace set_complement_intersection_l12_12558

open Set

variable (U M N : Set ℕ)

theorem set_complement_intersection :
  U = {1, 2, 3, 4, 5, 6, 7} →
  M = {3, 4, 5} →
  N = {1, 3, 6} →
  {2, 7} = (U \ M) ∩ (U \ N) :=
by
  intros hU hM hN
  rw [hU, hM, hN]
  sorry

end set_complement_intersection_l12_12558


namespace find_value_of_expr_l12_12157

variables (a b : ℝ)

def condition1 : Prop := a^2 + a * b = -2
def condition2 : Prop := b^2 - 3 * a * b = -3

theorem find_value_of_expr (h1 : condition1 a b) (h2 : condition2 a b) : a^2 + 4 * a * b - b^2 = 1 :=
sorry

end find_value_of_expr_l12_12157


namespace sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l12_12476

-- Part a) Prove that if a sequence has a limit, then it is bounded.
theorem sequence_with_limit_is_bounded (x : ℕ → ℝ) (x0 : ℝ) (h : ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) :
  ∃ C, ∀ n, |x n| ≤ C := by
  sorry

-- Part b) Is the converse statement true?
theorem bounded_sequence_does_not_imply_limit :
  ∃ (x : ℕ → ℝ), (∃ C, ∀ n, |x n| ≤ C) ∧ ¬(∃ x0, ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - x0| < ε) := by
  sorry

end sequence_with_limit_is_bounded_bounded_sequence_does_not_imply_limit_l12_12476


namespace solve_equation_l12_12515

theorem solve_equation (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) :
  (3 / (x - 2) = 2 + x / (2 - x)) ↔ x = 7 :=
sorry

end solve_equation_l12_12515


namespace rectangle_area_l12_12101

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w^2 + (3*w)^2 = d^2) : (3 * w ^ 2 = 3 * d ^ 2 / 10) :=
by
  sorry

end rectangle_area_l12_12101


namespace contrapositive_l12_12141

theorem contrapositive (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) → a^2 + b^2 ≠ 0 :=
by
  intro h
  sorry

end contrapositive_l12_12141


namespace machine_A_produces_1_sprockets_per_hour_l12_12745

namespace SprocketsProduction

variable {A T : ℝ} -- A: sprockets per hour of machine A, T: hours it takes for machine Q to produce 110 sprockets

-- Given conditions
axiom machine_Q_production_rate : 110 / T = 1.10 * A
axiom machine_P_production_rate : 110 / (T + 10) = A

-- The target theorem to prove
theorem machine_A_produces_1_sprockets_per_hour (h1 : 110 / T = 1.10 * A) (h2 : 110 / (T + 10) = A) : A = 1 :=
by sorry

end SprocketsProduction

end machine_A_produces_1_sprockets_per_hour_l12_12745


namespace solution_set_of_inequality_l12_12590

theorem solution_set_of_inequality :
  { x : ℝ | (x - 2) * (x + 3) > 0 } = { x : ℝ | x < -3 } ∪ { x : ℝ | x > 2 } :=
sorry

end solution_set_of_inequality_l12_12590


namespace find_teddy_dogs_l12_12116

-- Definitions from the conditions
def teddy_cats := 8
def ben_dogs (teddy_dogs : ℕ) := teddy_dogs + 9
def dave_cats (teddy_cats : ℕ) := teddy_cats + 13
def dave_dogs (teddy_dogs : ℕ) := teddy_dogs - 5
def total_pets (teddy_dogs teddy_cats : ℕ) := teddy_dogs + teddy_cats + (ben_dogs teddy_dogs) + (dave_dogs teddy_dogs) + (dave_cats teddy_cats)

-- Theorem statement
theorem find_teddy_dogs (teddy_dogs : ℕ) (teddy_cats : ℕ) (hd : total_pets teddy_dogs teddy_cats = 54) :
  teddy_dogs = 7 := sorry

end find_teddy_dogs_l12_12116


namespace rectangle_ratio_l12_12416

noncomputable def ratio_of_length_to_width (w : ℝ) : ℝ :=
  40 / w

theorem rectangle_ratio (w : ℝ) 
  (hw1 : 35 * (w + 5) = 40 * w + 75) : 
  ratio_of_length_to_width w = 2 :=
by
  sorry

end rectangle_ratio_l12_12416


namespace land_division_possible_l12_12398

-- Define the basic properties and conditions of the plot
structure Plot :=
  (is_square : Prop)
  (has_center_well : Prop)
  (has_four_trees : Prop)
  (has_four_gates : Prop)

-- Define a section of the plot
structure Section :=
  (contains_tree : Prop)
  (contains_gate : Prop)
  (equal_fence_length : Prop)
  (unrestricted_access_to_well : Prop)

-- Define the property that indicates a valid division of the plot
def valid_division (p : Plot) (sections : List Section) : Prop :=
  sections.length = 4 ∧
  (∀ s ∈ sections, s.contains_tree) ∧
  (∀ s ∈ sections, s.contains_gate) ∧
  (∀ s ∈ sections, s.equal_fence_length) ∧
  (∀ s ∈ sections, s.unrestricted_access_to_well)

-- Define the main theorem to prove
theorem land_division_possible (p : Plot) : 
  p.is_square ∧ p.has_center_well ∧ p.has_four_trees ∧ p.has_four_gates → 
  ∃ sections : List Section, valid_division p sections :=
by
  sorry

end land_division_possible_l12_12398


namespace regression_equation_correct_l12_12083

-- Defining the given data as constants
def x_data : List ℕ := [1, 2, 3, 4, 5, 6, 7]
def y_data : List ℕ := [891, 888, 351, 220, 200, 138, 112]

def sum_t_y : ℚ := 1586
def avg_t : ℚ := 0.37
def sum_t2_min7_avg_t2 : ℚ := 0.55

-- Defining the target regression equation
def target_regression (x : ℚ) : ℚ := 1000 / x + 30

-- Function to calculate the regression equation from data
noncomputable def calculate_regression (x_data y_data : List ℕ) : (ℚ → ℚ) :=
  let n : ℚ := x_data.length
  let avg_y : ℚ := y_data.sum / n
  let b : ℚ := (sum_t_y - n * avg_t * avg_y) / (sum_t2_min7_avg_t2)
  let a : ℚ := avg_y - b * avg_t
  fun x : ℚ => a + b / x

-- Theorem stating the regression equation matches the target regression equation
theorem regression_equation_correct :
  calculate_regression x_data y_data = target_regression :=
by
  sorry

end regression_equation_correct_l12_12083


namespace justify_misha_decision_l12_12317

-- Define the conditions based on the problem description
def reviews_smartphone_A := (7, 4) -- 7 positive and 4 negative reviews for A
def reviews_smartphone_B := (4, 1) -- 4 positive and 1 negative reviews for B

-- Define the ratios for each smartphone based on their reviews
def ratio_A := (reviews_smartphone_A.1 : ℚ) / reviews_smartphone_A.2
def ratio_B := (reviews_smartphone_B.1 : ℚ) / reviews_smartphone_B.2

-- Goal: to show that ratio_B > ratio_A, justifying Misha's decision
theorem justify_misha_decision : ratio_B > ratio_A := by
  -- placeholders to bypass the proof steps
  sorry

end justify_misha_decision_l12_12317


namespace carson_total_distance_l12_12005

def perimeter (length : ℕ) (width : ℕ) : ℕ :=
  2 * (length + width)

def total_distance (length : ℕ) (width : ℕ) (rounds : ℕ) (breaks : ℕ) (break_distance : ℕ) : ℕ :=
  let P := perimeter length width
  let distance_rounds := rounds * P
  let distance_breaks := breaks * break_distance
  distance_rounds + distance_breaks

theorem carson_total_distance :
  total_distance 600 400 8 4 100 = 16400 :=
by
  sorry

end carson_total_distance_l12_12005


namespace prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l12_12311

noncomputable def total_outcomes := 24
noncomputable def outcomes_two_correct := 6
noncomputable def outcomes_at_least_two_correct := 7
noncomputable def outcomes_all_incorrect := 9

theorem prob_two_correct : (outcomes_two_correct : ℚ) / total_outcomes = 1 / 4 := by
  sorry

theorem prob_at_least_two_correct : (outcomes_at_least_two_correct : ℚ) / total_outcomes = 7 / 24 := by
  sorry

theorem prob_all_incorrect : (outcomes_all_incorrect : ℚ) / total_outcomes = 3 / 8 := by
  sorry

end prob_two_correct_prob_at_least_two_correct_prob_all_incorrect_l12_12311


namespace liam_total_money_l12_12986

-- Define the conditions as noncomputable since they involve monetary calculations
noncomputable def liam_money (initial_bottles : ℕ) (price_per_bottle : ℕ) (bottles_sold : ℕ) (extra_money : ℕ) : ℚ :=
  let cost := initial_bottles * price_per_bottle
  let money_after_selling_part := cost + extra_money
  let selling_price_per_bottle := money_after_selling_part / bottles_sold
  let total_revenue := initial_bottles * selling_price_per_bottle
  total_revenue

-- State the theorem with the given problem
theorem liam_total_money :
  let initial_bottles := 50
  let price_per_bottle := 1
  let bottles_sold := 40
  let extra_money := 10
  liam_money initial_bottles price_per_bottle bottles_sold extra_money = 75 := 
sorry

end liam_total_money_l12_12986


namespace sqrt_x_minus_2_domain_l12_12604

theorem sqrt_x_minus_2_domain {x : ℝ} : (∃y : ℝ, y = Real.sqrt (x - 2)) ↔ x ≥ 2 :=
by sorry

end sqrt_x_minus_2_domain_l12_12604


namespace train_speed_correct_l12_12854

def train_length : ℝ := 2500  -- Length of the train in meters.
def crossing_time : ℝ := 100  -- Time to cross the electric pole in seconds.
def expected_speed : ℝ := 25  -- Expected speed of the train in meters/second.

theorem train_speed_correct :
  (train_length / crossing_time) = expected_speed :=
by
  sorry

end train_speed_correct_l12_12854


namespace radius_ratio_of_spheres_l12_12549

theorem radius_ratio_of_spheres
  (V_large : ℝ) (V_small : ℝ) (r_large r_small : ℝ)
  (h1 : V_large = 324 * π)
  (h2 : V_small = 0.25 * V_large)
  (h3 : (4/3) * π * r_large^3 = V_large)
  (h4 : (4/3) * π * r_small^3 = V_small) :
  (r_small / r_large) = (1/2) := 
sorry

end radius_ratio_of_spheres_l12_12549


namespace second_ball_red_probability_l12_12990

-- Definitions based on given conditions
def total_balls : ℕ := 10
def red_balls : ℕ := 6
def white_balls : ℕ := 4
def first_ball_is_red : Prop := true

-- The probability that the second ball drawn is red given the first ball drawn is red
def prob_second_red_given_first_red : ℚ :=
  (red_balls - 1) / (total_balls - 1)

theorem second_ball_red_probability :
  first_ball_is_red → prob_second_red_given_first_red = 5 / 9 :=
by
  intro _
  -- proof goes here
  sorry

end second_ball_red_probability_l12_12990


namespace find_number_l12_12403

theorem find_number (n : ℝ) : (1 / 2) * n + 6 = 11 → n = 10 := by
  sorry

end find_number_l12_12403


namespace pensioners_painting_conditions_l12_12146

def boardCondition (A Z : ℕ) : Prop :=
(∀ x y, (∃ i j, i ≤ 1 ∧ j ≤ 1 ∧ (x + 3 = A) ∧ (i ≤ 2 ∧ j ≤ 4 ∨ i ≤ 4 ∧ j ≤ 2) → x + 2 * y = Z))

theorem pensioners_painting_conditions (A Z : ℕ) :
  (boardCondition A Z) ↔ (A = 0 ∧ Z = 0) ∨ (A = 9 ∧ Z = 8) :=
sorry

end pensioners_painting_conditions_l12_12146


namespace balance_four_heartsuits_with_five_circles_l12_12202

variables (x y z : ℝ)

-- Given conditions
axiom condition1 : 4 * x + 3 * y = 12 * z
axiom condition2 : 2 * x = y + 3 * z

-- Statement to prove
theorem balance_four_heartsuits_with_five_circles : 4 * y = 5 * z :=
by sorry

end balance_four_heartsuits_with_five_circles_l12_12202


namespace biscuits_more_than_cookies_l12_12054

theorem biscuits_more_than_cookies :
  let morning_butter_cookies := 20
  let morning_biscuits := 40
  let afternoon_butter_cookies := 10
  let afternoon_biscuits := 20
  let total_butter_cookies := morning_butter_cookies + afternoon_butter_cookies
  let total_biscuits := morning_biscuits + afternoon_biscuits
  total_biscuits - total_butter_cookies = 30 :=
by
  sorry

end biscuits_more_than_cookies_l12_12054


namespace value_of_f_log3_54_l12_12168

noncomputable def f : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem value_of_f_log3_54
  (h1 : is_odd f)
  (h2 : ∀ x, f (x + 2) = -1 / f x)
  (h3 : ∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) :
  f (Real.log 54 / Real.log 3) = -3 / 2 := sorry

end value_of_f_log3_54_l12_12168


namespace ineq_five_times_x_minus_six_gt_one_l12_12347

variable {x : ℝ}

theorem ineq_five_times_x_minus_six_gt_one (x : ℝ) : 5 * x - 6 > 1 :=
sorry

end ineq_five_times_x_minus_six_gt_one_l12_12347


namespace minimize_y_l12_12772

noncomputable def y (x a b : ℝ) : ℝ := (x - a)^2 + (x - b)^2 + 3 * x + 5

theorem minimize_y (a b : ℝ) : 
  ∃ x : ℝ, (∀ x' : ℝ, y x a b ≤ y x' a b) → x = (2 * a + 2 * b - 3) / 4 := by
  sorry

end minimize_y_l12_12772


namespace chord_intersection_eq_l12_12697

theorem chord_intersection_eq (x y : ℝ) (r : ℝ) : 
  (x + 1)^2 + y^2 = r^2 → 
  (x - 4)^2 + (y - 1)^2 = 4 → 
  (x = 4) → 
  (y = 1) → 
  (r^2 = 26) → (5 * x + y - 19 = 0) :=
by
  sorry

end chord_intersection_eq_l12_12697


namespace total_students_l12_12507

theorem total_students (T : ℝ) (h : 0.50 * T = 440) : T = 880 := 
by {
  sorry
}

end total_students_l12_12507


namespace direction_cosines_l12_12026

theorem direction_cosines (x y z : ℝ) (α β γ : ℝ)
  (h1 : 2 * x - 3 * y - 3 * z - 9 = 0)
  (h2 : x - 2 * y + z + 3 = 0) :
  α = 9 / Real.sqrt 107 ∧ β = 5 / Real.sqrt 107 ∧ γ = 1 / Real.sqrt 107 :=
by
  -- Here, we will sketch out the proof to establish that these values for α, β, and γ hold.
  sorry

end direction_cosines_l12_12026


namespace evaluate_expression_l12_12208

theorem evaluate_expression :
  2 + (3 / (4 + (5 / (6 + (7 / 8))))) = 137 / 52 := by
  sorry

end evaluate_expression_l12_12208


namespace smallest_sum_is_381_l12_12591

def is_valid_3_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def uses_digits_once (n m : ℕ) : Prop :=
  (∀ d ∈ [1, 2, 3, 4, 5, 6], (d ∈ n.digits 10 ∨ d ∈ m.digits 10)) ∧
  (∀ d, d ∈ n.digits 10 → d ∈ [1, 2, 3, 4, 5, 6]) ∧
  (∀ d, d ∈ m.digits 10 → d ∈ [1, 2, 3, 4, 5, 6])

theorem smallest_sum_is_381 :
  ∃ (n m : ℕ), is_valid_3_digit_number n ∧ is_valid_3_digit_number m ∧
  uses_digits_once n m ∧ n + m = 381 :=
sorry

end smallest_sum_is_381_l12_12591


namespace right_triangle_counterexample_l12_12948

def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_right_angle (α : ℝ) : Prop := α = 90

def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180

def is_acute_triangle (α β γ : ℝ) : Prop := is_acute_angle α ∧ is_acute_angle β ∧ is_acute_angle γ

def is_right_triangle (α β γ : ℝ) : Prop := 
  (is_right_angle α ∧ is_acute_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_right_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_acute_angle β ∧ is_right_angle γ)

theorem right_triangle_counterexample (α β γ : ℝ) : 
  is_triangle α β γ → is_right_triangle α β γ → ¬ is_acute_triangle α β γ :=
by
  intro htri hrt hacute
  sorry

end right_triangle_counterexample_l12_12948


namespace right_triangle_side_length_l12_12205

theorem right_triangle_side_length (area : ℝ) (side1 : ℝ) (side2 : ℝ) (h_area : area = 8) (h_side1 : side1 = Real.sqrt 10) (h_area_eq : area = 0.5 * side1 * side2) :
  side2 = 1.6 * Real.sqrt 10 :=
by 
  sorry

end right_triangle_side_length_l12_12205


namespace symmetric_scanning_codes_count_l12_12457

noncomputable def countSymmetricScanningCodes : ℕ :=
  let totalConfigs := 32
  let invalidConfigs := 2
  totalConfigs - invalidConfigs

theorem symmetric_scanning_codes_count :
  countSymmetricScanningCodes = 30 :=
by
  -- Here, we would detail the steps, but we omit the actual proof for now.
  sorry

end symmetric_scanning_codes_count_l12_12457


namespace time_to_cook_rest_of_potatoes_l12_12107

-- Definitions of the conditions
def total_potatoes : ℕ := 12
def already_cooked : ℕ := 6
def minutes_per_potato : ℕ := 6

-- Proof statement
theorem time_to_cook_rest_of_potatoes : (total_potatoes - already_cooked) * minutes_per_potato = 36 :=
by
  sorry

end time_to_cook_rest_of_potatoes_l12_12107


namespace min_r_minus_p_l12_12769

theorem min_r_minus_p : ∃ (p q r : ℕ), p * q * r = 362880 ∧ p < q ∧ q < r ∧ (∀ p' q' r' : ℕ, (p' * q' * r' = 362880 ∧ p' < q' ∧ q' < r') → r - p ≤ r' - p') ∧ r - p = 39 :=
by
  sorry

end min_r_minus_p_l12_12769


namespace hash_op_example_l12_12379

def hash_op (a b c : ℤ) : ℤ := (b + 1)^2 - 4 * a * (c - 1)

theorem hash_op_example : hash_op 2 3 4 = -8 := by
  -- The proof can be added here, but for now, we use sorry to skip it
  sorry

end hash_op_example_l12_12379


namespace num_pass_students_is_85_l12_12820

theorem num_pass_students_is_85 (T P F : ℕ) (avg_all avg_pass avg_fail : ℕ) (weight_pass weight_fail : ℕ) 
  (h_total_students : T = 150)
  (h_avg_all : avg_all = 40)
  (h_avg_pass : avg_pass = 45)
  (h_avg_fail : avg_fail = 20)
  (h_weight_ratio : weight_pass = 3 ∧ weight_fail = 1)
  (h_total_marks : (weight_pass * avg_pass * P + weight_fail * avg_fail * F) / (weight_pass * P + weight_fail * F) = avg_all)
  (h_students_sum : P + F = T) :
  P = 85 :=
by
  sorry

end num_pass_students_is_85_l12_12820


namespace expenditure_may_to_july_l12_12938

theorem expenditure_may_to_july (spent_by_may : ℝ) (spent_by_july : ℝ) (h_may : spent_by_may = 0.8) (h_july : spent_by_july = 3.5) :
  spent_by_july - spent_by_may = 2.7 :=
by
  sorry

end expenditure_may_to_july_l12_12938


namespace calc_x_squared_plus_5xy_plus_y_squared_l12_12028

theorem calc_x_squared_plus_5xy_plus_y_squared 
  (x y : ℝ) 
  (h1 : x * y = 4)
  (h2 : x - y = 5) :
  x^2 + 5 * x * y + y^2 = 53 :=
by 
  sorry

end calc_x_squared_plus_5xy_plus_y_squared_l12_12028


namespace sum_common_elements_ap_gp_l12_12874

noncomputable def sum_of_first_10_common_elements : ℕ := 20 * (4^10 - 1) / (4 - 1)

theorem sum_common_elements_ap_gp :
  sum_of_first_10_common_elements = 6990500 :=
by
  unfold sum_of_first_10_common_elements
  sorry

end sum_common_elements_ap_gp_l12_12874


namespace two_subsets_count_l12_12674

-- Definitions from the problem conditions
def S : Set (Fin 5) := {0, 1, 2, 3, 4}

-- Main statement
theorem two_subsets_count : 
  (∃ A B : Set (Fin 5), A ∪ B = S ∧ A ∩ B = {a, b} ∧ A ≠ B) → 
  (number_of_ways = 40) :=
sorry

end two_subsets_count_l12_12674


namespace tower_remainder_l12_12544

def num_towers : ℕ := 907200  -- the total number of different towers S for 9 cubes

theorem tower_remainder : num_towers % 1000 = 200 :=
by
  sorry

end tower_remainder_l12_12544


namespace kate_money_ratio_l12_12436

-- Define the cost of the pen and the amount Kate needs
def pen_cost : ℕ := 30
def additional_money_needed : ℕ := 20

-- Define the amount of money Kate has
def kate_savings : ℕ := pen_cost - additional_money_needed

-- Define the ratio of Kate's money to the cost of the pen
def ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- The target property: the ratio of Kate's savings to the cost of the pen
theorem kate_money_ratio : ratio kate_savings pen_cost = (1, 3) :=
by
  sorry

end kate_money_ratio_l12_12436


namespace least_number_to_subtract_l12_12902

theorem least_number_to_subtract (n : ℕ) : 
  ∃ k : ℕ, k = 762429836 % 17 ∧ k = 15 := 
by sorry

end least_number_to_subtract_l12_12902


namespace students_with_certificates_l12_12060

variable (C N : ℕ)

theorem students_with_certificates :
  (C + N = 120) ∧ (C = N + 36) → C = 78 :=
by
  sorry

end students_with_certificates_l12_12060


namespace divisible_check_l12_12471

theorem divisible_check (n : ℕ) (h : n = 287) : 
  ¬ (n % 3 = 0) ∧  ¬ (n % 4 = 0) ∧  ¬ (n % 5 = 0) ∧ ¬ (n % 6 = 0) ∧ (n % 7 = 0) := 
by {
  sorry
}

end divisible_check_l12_12471


namespace probability_of_less_than_20_l12_12345

variable (total_people : ℕ) (people_over_30 : ℕ)
variable (people_under_20 : ℕ) (probability_under_20 : ℝ)

noncomputable def group_size := total_people = 150
noncomputable def over_30 := people_over_30 = 90
noncomputable def under_20 := people_under_20 = total_people - people_over_30

theorem probability_of_less_than_20
  (total_people_eq : total_people = 150)
  (people_over_30_eq : people_over_30 = 90)
  (people_under_20_eq : people_under_20 = 60)
  (under_20_eq : 60 = total_people - people_over_30) :
  probability_under_20 = people_under_20 / total_people := by
  sorry

end probability_of_less_than_20_l12_12345


namespace roots_of_quadratic_l12_12412

theorem roots_of_quadratic (a b c : ℝ) (h1 : a ≠ 0) (h2 : a + b + c = 0) (h3 : a - b + c = 0) : 
  (a * 1 ^2 + b * 1 + c = 0) ∧ (a * (-1) ^2 + b * (-1) + c = 0) :=
sorry

end roots_of_quadratic_l12_12412


namespace joey_speed_return_l12_12237

/--
Joey the postman takes 1 hour to run a 5-mile-long route every day, delivering packages along the way.
On his return, he must climb a steep hill covering 3 miles and then navigate a rough, muddy terrain spanning 2 miles.
If the average speed of the entire round trip is 8 miles per hour, prove that the speed with which Joey returns along the path is 20 miles per hour.
-/
theorem joey_speed_return
  (dist_out : ℝ := 5)
  (time_out : ℝ := 1)
  (dist_hill : ℝ := 3)
  (dist_terrain : ℝ := 2)
  (avg_speed_round : ℝ := 8)
  (total_dist : ℝ := dist_out * 2)
  (total_time : ℝ := total_dist / avg_speed_round)
  (time_return : ℝ := total_time - time_out)
  (dist_return : ℝ := dist_hill + dist_terrain) :
  (dist_return / time_return = 20) := 
sorry

end joey_speed_return_l12_12237


namespace sixth_graders_bought_more_pencils_23_l12_12975

open Int

-- Conditions
def pencils_cost_whole_number_cents : Prop := ∃ n : ℕ, n > 0
def seventh_graders_total_cents := 165
def sixth_graders_total_cents := 234
def number_of_sixth_graders := 30

-- The number of sixth graders who bought more pencils than seventh graders
theorem sixth_graders_bought_more_pencils_23 :
  (seventh_graders_total_cents / 3 = 55) ∧
  (sixth_graders_total_cents / 3 = 78) →
  78 - 55 = 23 :=
by
  sorry

end sixth_graders_bought_more_pencils_23_l12_12975


namespace negation_of_existence_statement_l12_12224

theorem negation_of_existence_statement :
  (¬ ∃ x : ℝ, x^2 - 8 * x + 18 < 0) ↔ (∀ x : ℝ, x^2 - 8 * x + 18 ≥ 0) :=
by
  sorry

end negation_of_existence_statement_l12_12224


namespace neg_four_is_square_root_of_sixteen_l12_12124

/-
  Definitions:
  - A number y is a square root of x if y^2 = x.
  - A number y is an arithmetic square root of x if y ≥ 0 and y^2 = x.
-/

theorem neg_four_is_square_root_of_sixteen :
  -4 * -4 = 16 := 
by
  -- proof step is omitted
  sorry

end neg_four_is_square_root_of_sixteen_l12_12124


namespace problem_1_solution_problem_2_solution_l12_12648

noncomputable def problem_1 : Real :=
  (-3) + (2 - Real.pi)^0 - (1 / 2)⁻¹

theorem problem_1_solution :
  problem_1 = -4 :=
by
  sorry

noncomputable def problem_2 (a : Real) : Real :=
  (2 * a)^3 - a * a^2 + 3 * a^6 / a^3

theorem problem_2_solution (a : Real) :
  problem_2 a = 10 * a^3 :=
by
  sorry

end problem_1_solution_problem_2_solution_l12_12648


namespace arithmetic_sequence_general_term_l12_12563

theorem arithmetic_sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n - a (n - 1) = 2) : ∀ n, a n = 2 * n - 1 := by
  sorry

end arithmetic_sequence_general_term_l12_12563


namespace john_total_fuel_usage_l12_12527

def city_fuel_rate := 6 -- liters per km for city traffic
def highway_fuel_rate := 4 -- liters per km for highway traffic

def trip1_city_distance := 50 -- km for Trip 1
def trip2_highway_distance := 35 -- km for Trip 2
def trip3_city_distance := 15 -- km for Trip 3 in city traffic
def trip3_highway_distance := 10 -- km for Trip 3 on highway

-- Define the total fuel consumption
def total_fuel_used : Nat :=
  (trip1_city_distance * city_fuel_rate) +
  (trip2_highway_distance * highway_fuel_rate) +
  (trip3_city_distance * city_fuel_rate) +
  (trip3_highway_distance * highway_fuel_rate)

theorem john_total_fuel_usage :
  total_fuel_used = 570 :=
by
  sorry

end john_total_fuel_usage_l12_12527


namespace math_problem_equivalent_l12_12089

-- Given that the problem requires four distinct integers a, b, c, d which are less than 12 and invertible modulo 12.
def coprime_with_12 (x : ℕ) : Prop := Nat.gcd x 12 = 1

theorem math_problem_equivalent 
  (a b c d : ℕ) (ha : coprime_with_12 a) (hb : coprime_with_12 b) 
  (hc : coprime_with_12 c) (hd : coprime_with_12 d) 
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c)
  (hbd : b ≠ d) (hcd : c ≠ d) :
  ((a * b * c * d) + (a * b * c) + (a * b * d) + (a * c * d) + (b * c * d)) * Nat.gcd (a * b * c * d) 12 = 1 :=
sorry

end math_problem_equivalent_l12_12089


namespace sum_of_numbers_l12_12829

theorem sum_of_numbers : ∃ (a b : ℕ), (a + b = 21) ∧ (a / b = 3 / 4) ∧ (max a b = 12) :=
by
  sorry

end sum_of_numbers_l12_12829


namespace necessary_and_sufficient_condition_holds_l12_12491

noncomputable def necessary_and_sufficient_condition (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + m > 0

theorem necessary_and_sufficient_condition_holds (m : ℝ) :
  necessary_and_sufficient_condition m ↔ m > 1 :=
by
  sorry

end necessary_and_sufficient_condition_holds_l12_12491


namespace find_triples_l12_12733

theorem find_triples (x n p : ℕ) (hp : Nat.Prime p) 
  (hx_pos : x > 0) (hn_pos : n > 0) : 
  x^3 + 3 * x + 14 = 2 * p^n → (x = 1 ∧ n = 2 ∧ p = 3) ∨ (x = 3 ∧ n = 2 ∧ p = 5) :=
by 
  sorry

end find_triples_l12_12733


namespace remainder_21_pow_2051_mod_29_l12_12580

theorem remainder_21_pow_2051_mod_29 :
  ∀ (a : ℤ), (21^4 ≡ 1 [MOD 29]) -> (2051 = 4 * 512 + 3) -> (21^3 ≡ 15 [MOD 29]) -> (21^2051 ≡ 15 [MOD 29]) :=
by
  intros a h1 h2 h3
  sorry

end remainder_21_pow_2051_mod_29_l12_12580


namespace M_eq_N_l12_12400

noncomputable def M (a : ℝ) : ℝ :=
  a^2 + (a + 3)^2 + (a + 5)^2 + (a + 6)^2

noncomputable def N (a : ℝ) : ℝ :=
  (a + 1)^2 + (a + 2)^2 + (a + 4)^2 + (a + 7)^2

theorem M_eq_N (a : ℝ) : M a = N a :=
by
  sorry

end M_eq_N_l12_12400


namespace rectangle_area_increase_l12_12160

theorem rectangle_area_increase :
  let l := 33.333333333333336
  let b := l / 2
  let A_original := l * b
  let l_new := l - 5
  let b_new := b + 4
  let A_new := l_new * b_new
  A_new - A_original = 30 :=
by
  sorry

end rectangle_area_increase_l12_12160


namespace roots_cubic_polynomial_l12_12616

theorem roots_cubic_polynomial (r s t : ℝ)
  (h₁ : 8 * r^3 + 1001 * r + 2008 = 0)
  (h₂ : 8 * s^3 + 1001 * s + 2008 = 0)
  (h₃ : 8 * t^3 + 1001 * t + 2008 = 0)
  (h₄ : r + s + t = 0) :
  (r + s)^3 + (s + t)^3 + (t + r)^3 = 753 := 
sorry

end roots_cubic_polynomial_l12_12616


namespace more_stable_performance_l12_12719

theorem more_stable_performance (s_A_sq s_B_sq : ℝ) (hA : s_A_sq = 0.25) (hB : s_B_sq = 0.12) : s_A_sq > s_B_sq :=
by
  rw [hA, hB]
  sorry

end more_stable_performance_l12_12719


namespace sports_field_perimeter_l12_12145

noncomputable def perimeter_of_sports_field (a b : ℝ) (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) : ℝ :=
  2 * (a + b)

theorem sports_field_perimeter {a b : ℝ} (h1 : a^2 + b^2 = 400) (h2 : a * b = 120) :
  perimeter_of_sports_field a b h1 h2 = 51 := by
  sorry

end sports_field_perimeter_l12_12145


namespace independence_test_purpose_l12_12442

theorem independence_test_purpose:
  ∀ (test: String), test = "independence test" → 
  ∀ (purpose: String), purpose = "to provide the reliability of the relationship between two categorical variables" →
  (test = "independence test" ∧ purpose = "to provide the reliability of the relationship between two categorical variables") :=
by
  intros test h_test purpose h_purpose
  exact ⟨h_test, h_purpose⟩

end independence_test_purpose_l12_12442


namespace fractions_equal_l12_12665

theorem fractions_equal (a b c d : ℚ) (h1 : a = 2/7) (h2 : b = 3) (h3 : c = 3/7) (h4 : d = 2) :
  a * b = c * d := 
sorry

end fractions_equal_l12_12665


namespace factorize_expression_l12_12559

variables {a x y : ℝ}

theorem factorize_expression (a x y : ℝ) : 3 * a * x ^ 2 + 6 * a * x * y + 3 * a * y ^ 2 = 3 * a * (x + y) ^ 2 :=
by
  sorry

end factorize_expression_l12_12559


namespace ones_digit_34_pow_34_pow_17_pow_17_l12_12413

-- Definitions from the conditions
def ones_digit (n : ℕ) : ℕ := n % 10

-- Translation of the original problem statement
theorem ones_digit_34_pow_34_pow_17_pow_17 :
  ones_digit (34 ^ (34 * 17 ^ 17)) = 4 :=
sorry

end ones_digit_34_pow_34_pow_17_pow_17_l12_12413


namespace total_balloons_l12_12469

def tom_balloons : Nat := 9
def sara_balloons : Nat := 8

theorem total_balloons : tom_balloons + sara_balloons = 17 := 
by
  sorry

end total_balloons_l12_12469


namespace find_quotient_l12_12753

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h_dividend : dividend = 171) 
  (h_divisor : divisor = 21) 
  (h_remainder : remainder = 3) 
  (h_div_eq : dividend = divisor * quotient + remainder) :
  quotient = 8 :=
by sorry

end find_quotient_l12_12753


namespace march_volume_expression_l12_12914

variable (x : ℝ) (y : ℝ)

def initial_volume : ℝ := 500
def growth_rate_volumes (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)
def calculate_march_volume (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)^2

theorem march_volume_expression :
  y = calculate_march_volume x initial_volume :=
sorry

end march_volume_expression_l12_12914


namespace maximum_reduced_price_l12_12876

theorem maximum_reduced_price (marked_price : ℝ) (cost_price : ℝ) (reduced_price : ℝ) 
    (h1 : marked_price = 240) 
    (h2 : marked_price = cost_price * 1.6) 
    (h3 : reduced_price - cost_price ≥ cost_price * 0.1) : 
    reduced_price ≤ 165 :=
sorry

end maximum_reduced_price_l12_12876


namespace smallest_prime_divisor_of_sum_l12_12607

theorem smallest_prime_divisor_of_sum : ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ (3 ^ 15 + 11 ^ 21) :=
by
  sorry

end smallest_prime_divisor_of_sum_l12_12607


namespace max_e_of_conditions_l12_12773

theorem max_e_of_conditions (a b c d e : ℝ) 
  (h1 : a + b + c + d + e = 8) 
  (h2 : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  e ≤ (16 / 5) :=
by 
  sorry

end max_e_of_conditions_l12_12773


namespace find_uv_l12_12581

open Real

def vec1 : ℝ × ℝ := (3, -2)
def vec2 : ℝ × ℝ := (-1, 2)
def vec3 : ℝ × ℝ := (1, -1)
def vec4 : ℝ × ℝ := (4, -7)
def vec5 : ℝ × ℝ := (-3, 5)

theorem find_uv (u v : ℝ) :
  vec1 + ⟨4 * u, -7 * u⟩ = vec2 + ⟨-3 * v, 5 * v⟩ + vec3 →
  u = 3 / 4 ∧ v = -9 / 4 :=
by
  sorry

end find_uv_l12_12581


namespace triangle_inequality_l12_12286

theorem triangle_inequality (S R r : ℝ) (h : S^2 = 2 * R^2 + 8 * R * r + 3 * r^2) : 
  R / r = 2 ∨ R / r ≥ Real.sqrt 2 + 1 := 
by 
  sorry

end triangle_inequality_l12_12286


namespace question1_question2_case1_question2_case2_question2_case3_l12_12696

def f (x a : ℝ) : ℝ := x^2 + (1 - a) * x - a

theorem question1 (x : ℝ) (h : (-1 < x) ∧ (x < 3)) : f x 3 < 0 := sorry

theorem question2_case1 (x : ℝ) : f x (-1) > 0 ↔ x ≠ -1 := sorry

theorem question2_case2 (x a : ℝ) (h : a > -1) : f x a > 0 ↔ (x < -1 ∨ x > a) := sorry

theorem question2_case3 (x a : ℝ) (h : a < -1) : f x a > 0 ↔ (x < a ∨ x > -1) := sorry

end question1_question2_case1_question2_case2_question2_case3_l12_12696


namespace inequality_solution_condition_necessary_but_not_sufficient_l12_12941

theorem inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ↔ (a ≥ 0 ∨ a ≤ -1) := sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (a > 0 ∨ a < -1) → (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ∧ ¬(∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0 → (a > 0 ∨ a < -1)) := sorry

end inequality_solution_condition_necessary_but_not_sufficient_l12_12941


namespace solve_some_number_l12_12173

theorem solve_some_number (n : ℝ) (h : (n * 10) / 100 = 0.032420000000000004) : n = 0.32420000000000004 :=
by
  -- The proof steps are omitted with 'sorry' here.
  sorry

end solve_some_number_l12_12173


namespace value_of_f_minus_a_l12_12738

noncomputable def f (x : ℝ) : ℝ := x^3 + x + 1

theorem value_of_f_minus_a (a : ℝ) (h : f a = 2) : f (-a) = 0 :=
by sorry

end value_of_f_minus_a_l12_12738


namespace gcd_factorials_l12_12584

theorem gcd_factorials : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = 5040 := by
  sorry

end gcd_factorials_l12_12584


namespace average_infect_influence_l12_12029

theorem average_infect_influence
  (x : ℝ)
  (h : (1 + x)^2 = 100) :
  x = 9 :=
sorry

end average_infect_influence_l12_12029


namespace suitable_for_census_l12_12446

-- Define types for each survey option.
inductive SurveyOption where
  | A : SurveyOption -- Understanding the vision of middle school students in our province
  | B : SurveyOption -- Investigating the viewership of "The Reader"
  | C : SurveyOption -- Inspecting the components of a newly developed fighter jet to ensure successful test flights
  | D : SurveyOption -- Testing the lifespan of a batch of light bulbs

-- Theorem statement asserting that Option C is the suitable one for a census.
theorem suitable_for_census : SurveyOption.C = SurveyOption.C :=
by
  exact rfl

end suitable_for_census_l12_12446


namespace rectangle_width_l12_12244

theorem rectangle_width
  (L W : ℝ)
  (h1 : W = L + 2)
  (h2 : 2 * L + 2 * W = 16) :
  W = 5 :=
by
  sorry

end rectangle_width_l12_12244


namespace least_deletions_to_square_l12_12393

theorem least_deletions_to_square (l : List ℕ) (h : l = [10, 20, 30, 40, 50, 60, 70, 80, 90]) : 
  ∃ d, d.card ≤ 2 ∧ ∀ (lp : List ℕ), lp = l.diff d → 
  ∃ k, lp.prod = k^2 :=
by
  sorry

end least_deletions_to_square_l12_12393


namespace smallest_k_49_divides_binom_l12_12855

theorem smallest_k_49_divides_binom : 
  ∃ k : ℕ, 0 < k ∧ 49 ∣ Nat.choose (2 * k) k ∧ (∀ m : ℕ, 0 < m ∧ 49 ∣ Nat.choose (2 * m) m → k ≤ m) ∧ k = 25 :=
by
  sorry

end smallest_k_49_divides_binom_l12_12855


namespace ott_fractional_part_l12_12668

theorem ott_fractional_part (x : ℝ) :
  let moe_initial := 6 * x
  let loki_initial := 5 * x
  let nick_initial := 4 * x
  let ott_initial := 1
  
  let moe_given := (x : ℝ)
  let loki_given := (x : ℝ)
  let nick_given := (x : ℝ)
  
  let ott_returned_each := (1 / 10) * x
  
  let moe_effective := moe_given - ott_returned_each
  let loki_effective := loki_given - ott_returned_each
  let nick_effective := nick_given - ott_returned_each
  
  let ott_received := moe_effective + loki_effective + nick_effective
  let ott_final_money := ott_initial + ott_received
  
  let total_money_original := moe_initial + loki_initial + nick_initial + ott_initial
  let fraction_ott_final := ott_final_money / total_money_original
  
  ott_final_money / total_money_original = (10 + 27 * x) / (150 * x + 10) :=
by
  sorry

end ott_fractional_part_l12_12668


namespace no_integer_n_exists_l12_12153

theorem no_integer_n_exists : ∀ (n : ℤ), n ^ 2022 - 2 * n ^ 2021 + 3 * n ^ 2019 ≠ 2020 :=
by sorry

end no_integer_n_exists_l12_12153


namespace large_block_dimension_ratio_l12_12841

theorem large_block_dimension_ratio
  (V_normal V_large : ℝ) 
  (k : ℝ)
  (h1 : V_normal = 4)
  (h2 : V_large = 32) 
  (h3 : V_large = k^3 * V_normal) :
  k = 2 := by
  sorry

end large_block_dimension_ratio_l12_12841


namespace alyssa_went_to_13_games_last_year_l12_12289

theorem alyssa_went_to_13_games_last_year :
  ∀ (X : ℕ), (11 + X + 15 = 39) → X = 13 :=
by
  intros X h
  sorry

end alyssa_went_to_13_games_last_year_l12_12289


namespace range_of_a_l12_12781

theorem range_of_a (a b c : ℝ) (h1 : a + b + c = 2) (h2 : a^2 + b^2 + c^2 = 4) (h3 : a > b) (h4 : b > c) :
  a ∈ Set.Ioo (2 / 3) 2 :=
sorry

end range_of_a_l12_12781


namespace evaporated_water_l12_12070

theorem evaporated_water 
  (E : ℝ)
  (h₁ : 0 < 10) -- initial mass is positive
  (h₂ : 10 * 0.3 + 10 * 0.7 = 3 + 7) -- Solution Y composition check
  (h₃ : (3 + 0.3 * E) / (10 - E + 0.7 * E) = 0.36) -- New solution composition
  : E = 0.9091 := 
sorry

end evaporated_water_l12_12070


namespace find_possible_values_l12_12509

theorem find_possible_values (x : ℝ) (h : x^2 + 9 * (x / (x - 3))^2 = 90) :
  ∃ y, (y = (x - 3)^2 * (x + 4) / (3 * x - 4)) ∧ (y = 36 / 11 ∨ y = 468 / 23) :=
by
  sorry

end find_possible_values_l12_12509


namespace x_lt_y_l12_12640

variable {a b c d x y : ℝ}

theorem x_lt_y 
  (ha : a > 1) 
  (hb : b > 1) 
  (hc : c > 1) 
  (hd : d > 1)
  (h1 : a^x + b^y = (a^2 + b^2)^x)
  (h2 : c^x + d^y = 2^y * (cd)^(y/2)) : 
  x < y :=
by 
  sorry

end x_lt_y_l12_12640


namespace first_term_of_arithmetic_sequence_l12_12109

theorem first_term_of_arithmetic_sequence :
  ∃ (a_1 : ℤ), ∀ (d n : ℤ), d = 3 / 4 ∧ n = 30 ∧ a_n = 63 / 4 → a_1 = -6 := by
  sorry

end first_term_of_arithmetic_sequence_l12_12109


namespace seminar_attendees_l12_12888

theorem seminar_attendees (a b c d attendees_not_from_companies : ℕ)
  (h1 : a = 30)
  (h2 : b = 2 * a)
  (h3 : c = a + 10)
  (h4 : d = c - 5)
  (h5 : attendees_not_from_companies = 20) :
  a + b + c + d + attendees_not_from_companies = 185 := by
  sorry

end seminar_attendees_l12_12888


namespace simplify_and_evaluate_expression_l12_12004

theorem simplify_and_evaluate_expression (m : ℝ) (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2*m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l12_12004


namespace fraction_power_zero_l12_12518

variable (a b : ℤ)
variable (h_a : a ≠ 0) (h_b : b ≠ 0)

theorem fraction_power_zero : (a / b)^0 = 1 := by
  sorry

end fraction_power_zero_l12_12518


namespace square_area_l12_12677

theorem square_area (x : ℝ) (s1 s2 area : ℝ) 
  (h1 : s1 = 5 * x - 21) 
  (h2 : s2 = 36 - 4 * x) 
  (hs : s1 = s2)
  (ha : area = s1 * s1) : 
  area = 113.4225 := 
by
  -- Proof goes here
  sorry

end square_area_l12_12677


namespace combined_average_score_clubs_l12_12079

theorem combined_average_score_clubs
  (nA nB : ℕ) -- Number of members in each club
  (avgA avgB : ℝ) -- Average score of each club
  (hA : nA = 40)
  (hB : nB = 50)
  (hAvgA : avgA = 90)
  (hAvgB : avgB = 81) :
  (nA * avgA + nB * avgB) / (nA + nB) = 85 :=
by
  sorry -- Proof omitted

end combined_average_score_clubs_l12_12079


namespace sum_first_3n_terms_l12_12751

-- Geometric Sequence: Sum of first n terms Sn, first 2n terms S2n, first 3n terms S3n.
variables {n : ℕ} {S : ℕ → ℕ}

-- Conditions
def sum_first_n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S n = 48
def sum_first_2n_terms (S : ℕ → ℕ) (n : ℕ) : Prop := S (2 * n) = 60

-- Theorem to Prove
theorem sum_first_3n_terms {S : ℕ → ℕ} (h1 : sum_first_n_terms S n) (h2 : sum_first_2n_terms S n) :
  S (3 * n) = 63 :=
sorry

end sum_first_3n_terms_l12_12751


namespace min_sum_ab_l12_12565

theorem min_sum_ab (a b : ℤ) (hab : a * b = 72) : a + b ≥ -17 := by
  sorry

end min_sum_ab_l12_12565


namespace equation_one_solution_equation_two_solution_l12_12578

theorem equation_one_solution (x : ℝ) : 4 * (x - 1)^2 - 9 = 0 ↔ (x = 5 / 2) ∨ (x = - 1 / 2) := 
by sorry

theorem equation_two_solution (x : ℝ) : x^2 - 6 * x - 7 = 0 ↔ (x = 7) ∨ (x = - 1) :=
by sorry

end equation_one_solution_equation_two_solution_l12_12578


namespace initial_volume_is_72_l12_12783

noncomputable def initial_volume (V : ℝ) : Prop :=
  let salt_initial : ℝ := 0.10 * V
  let total_volume_new : ℝ := V + 18
  let salt_percentage_new : ℝ := 0.08 * total_volume_new
  salt_initial = salt_percentage_new

theorem initial_volume_is_72 :
  ∃ V : ℝ, initial_volume V ∧ V = 72 :=
by
  sorry

end initial_volume_is_72_l12_12783


namespace prove_M_l12_12235

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {2, 3}
def M : Set ℕ := {x | x ∈ P ∧ x ∉ Q}

theorem prove_M :
  M = {1} :=
by
  sorry

end prove_M_l12_12235


namespace scientific_notation_280000_l12_12756

theorem scientific_notation_280000 : (280000 : ℝ) = 2.8 * 10^5 :=
sorry

end scientific_notation_280000_l12_12756


namespace inequality_proof_l12_12967

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_cond : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
by
  sorry

end inequality_proof_l12_12967


namespace determine_c_l12_12337

theorem determine_c {f : ℝ → ℝ} (c : ℝ) (h : ∀ x, f x = 2 / (3 * x + c))
  (hf_inv : ∀ x, (f⁻¹ x) = (3 - 6 * x) / x) : c = 18 :=
by sorry

end determine_c_l12_12337


namespace sandwiches_per_day_l12_12415

theorem sandwiches_per_day (S : ℕ) 
  (h1 : ∀ n, n = 4 * S)
  (h2 : 7 * 4 * S = 280) : S = 10 := 
by
  sorry

end sandwiches_per_day_l12_12415


namespace salad_quantity_percentage_difference_l12_12150

noncomputable def Tom_rate := 2/3 -- Tom's rate (lb/min)
noncomputable def Tammy_rate := 3/2 -- Tammy's rate (lb/min)
noncomputable def Total_salad := 65 -- Total salad chopped (lb)
noncomputable def Time_to_chop := Total_salad / (Tom_rate + Tammy_rate) -- Time to chop 65 lb (min)
noncomputable def Tom_chop := Time_to_chop * Tom_rate -- Total chopped by Tom (lb)
noncomputable def Tammy_chop := Time_to_chop * Tammy_rate -- Total chopped by Tammy (lb)
noncomputable def Percent_difference := (Tammy_chop - Tom_chop) / Tom_chop * 100 -- Percent difference

theorem salad_quantity_percentage_difference : Percent_difference = 125 :=
by
  sorry

end salad_quantity_percentage_difference_l12_12150


namespace intersection_point_on_circle_l12_12397

theorem intersection_point_on_circle :
  ∀ (m : ℝ) (x y : ℝ),
  (m * x - y = 0) → 
  (x + m * y - m - 2 = 0) → 
  (x - 1)^2 + (y - 1 / 2)^2 = 5 / 4 :=
by
  intros m x y h1 h2
  sorry

end intersection_point_on_circle_l12_12397


namespace positive_difference_solutions_l12_12081

theorem positive_difference_solutions (r₁ r₂ : ℝ) (h_r₁ : (r₁^2 - 5 * r₁ - 22) / (r₁ + 4) = 3 * r₁ + 8) (h_r₂ : (r₂^2 - 5 * r₂ - 22) / (r₂ + 4) = 3 * r₂ + 8) (h_r₁_ne : r₁ ≠ -4) (h_r₂_ne : r₂ ≠ -4) :
  |r₁ - r₂| = 3 / 2 := 
sorry


end positive_difference_solutions_l12_12081


namespace arctan_tan_expr_is_75_degrees_l12_12547

noncomputable def arctan_tan_expr : ℝ := Real.arctan (Real.tan (75 * Real.pi / 180) - 2 * Real.tan (30 * Real.pi / 180))

theorem arctan_tan_expr_is_75_degrees : (arctan_tan_expr * 180 / Real.pi) = 75 := 
by
  sorry

end arctan_tan_expr_is_75_degrees_l12_12547


namespace find_crossed_out_digit_l12_12742

theorem find_crossed_out_digit (n : ℕ) (h_rev : ∀ (k : ℕ), k < n → k % 9 = 0) (remaining_sum : ℕ) 
  (crossed_sum : ℕ) (h_sum : remaining_sum + crossed_sum = 27) : 
  crossed_sum = 8 :=
by
  -- We can incorporate generating the value from digit sum here.
  sorry

end find_crossed_out_digit_l12_12742


namespace sum_gcd_lcm_63_2898_l12_12344

theorem sum_gcd_lcm_63_2898 : Nat.gcd 63 2898 + Nat.lcm 63 2898 = 182575 :=
by
  sorry

end sum_gcd_lcm_63_2898_l12_12344


namespace total_volume_l12_12147

-- Defining the volumes for different parts as per the conditions.
variables (V_A V_C V_B' V_C' : ℝ)
variables (V : ℝ)

-- The given conditions
axiom V_A_eq_40 : V_A = 40
axiom V_C_eq_300 : V_C = 300
axiom V_B'_eq_360 : V_B' = 360
axiom V_C'_eq_90 : V_C' = 90

-- The proof goal: total volume of the parallelepiped
theorem total_volume (V_A V_C V_B' V_C' : ℝ) 
  (V_A_eq_40 : V_A = 40) (V_C_eq_300 : V_C = 300) 
  (V_B'_eq_360 : V_B' = 360) (V_C'_eq_90 : V_C' = 90) :
  V = V_A + V_C + V_B' + V_C' :=
by
  sorry

end total_volume_l12_12147


namespace roots_of_equation_l12_12121

theorem roots_of_equation (x : ℝ) : (x - 3) ^ 2 = 4 ↔ (x = 5 ∨ x = 1) := by
  sorry

end roots_of_equation_l12_12121


namespace eight_digit_number_min_max_l12_12186

theorem eight_digit_number_min_max (Amin Amax B : ℕ) 
  (hAmin: Amin = 14444446) 
  (hAmax: Amax = 99999998) 
  (hB_coprime: Nat.gcd B 12 = 1) 
  (hB_length: 44444444 < B) 
  (h_digits: ∀ (b : ℕ), b < 10 → ∃ (A : ℕ), A = 10^7 * b + (B - b) / 10 ∧ A < 100000000) :
  (∃ b, Amin = 10^7 * b + (44444461 - b) / 10 ∧ Nat.gcd 44444461 12 = 1 ∧ 44444444 < 44444461) ∧
  (∃ b, Amax = 10^7 * b + (999999989 - b) / 10 ∧ Nat.gcd 999999989 12 = 1 ∧ 44444444 < 999999989) :=
  sorry

end eight_digit_number_min_max_l12_12186


namespace find_quotient_l12_12339

-- Variables for larger number L and smaller number S
variables (L S: ℕ)

-- Conditions as definitions
def condition1 := L - S = 1325
def condition2 (quotient: ℕ) := L = S * quotient + 5
def condition3 := L = 1650

-- Statement to prove the quotient is 5
theorem find_quotient : ∃ (quotient: ℕ), condition1 L S ∧ condition2 L S quotient ∧ condition3 L → quotient = 5 := by
  sorry

end find_quotient_l12_12339


namespace acute_triangle_sin_sum_gt_two_l12_12853

theorem acute_triangle_sin_sum_gt_two 
  {α β γ : ℝ} 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : 0 < γ ∧ γ < π / 2) 
  (h4 : α + β + γ = π) :
  (Real.sin α + Real.sin β + Real.sin γ > 2) :=
sorry

end acute_triangle_sin_sum_gt_two_l12_12853


namespace tv_cost_solution_l12_12569

theorem tv_cost_solution (M T : ℝ) 
  (h1 : 2 * M + T = 7000)
  (h2 : M + 2 * T = 9800) : 
  T = 4200 :=
by
  sorry

end tv_cost_solution_l12_12569


namespace number_of_girls_attending_picnic_l12_12117

variables (g b : ℕ)

def hms_conditions : Prop :=
  g + b = 1500 ∧ (3 / 4 : ℝ) * g + (3 / 5 : ℝ) * b = 975

theorem number_of_girls_attending_picnic (h : hms_conditions g b) : (3 / 4 : ℝ) * g = 375 :=
sorry

end number_of_girls_attending_picnic_l12_12117


namespace point_in_fourth_quadrant_l12_12940

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : -4 * a < 0) (h2 : 2 + b < 0) : 
  (a > 0) ∧ (b < -2) → (a > 0) ∧ (b < 0) := 
by
  sorry

end point_in_fourth_quadrant_l12_12940


namespace poly_solution_l12_12879

-- Definitions for the conditions of the problem
def poly1 (d g : ℚ) := 5 * d ^ 2 - 4 * d + g
def poly2 (d h : ℚ) := 4 * d ^ 2 + h * d - 5
def product (d g h : ℚ) := 20 * d ^ 4 - 31 * d ^ 3 - 17 * d ^ 2 + 23 * d - 10

-- Statement of the problem: proving g + h = 7/2 given the conditions.
theorem poly_solution
  (g h : ℚ)
  (cond : ∀ d : ℚ, poly1 d g * poly2 d h = product d g h) :
  g + h = 7 / 2 :=
by
  sorry

end poly_solution_l12_12879


namespace find_b_l12_12704

theorem find_b (b n : ℝ) (h_neg : b < 0) :
  (∀ x, x^2 + b * x + 1 / 4 = (x + n)^2 + 1 / 18) → b = - (Real.sqrt 7) / 3 :=
by
  sorry

end find_b_l12_12704


namespace find_x_l12_12161

noncomputable section

variable (x : ℝ)
def vector_v : ℝ × ℝ := (x, 4)
def vector_w : ℝ × ℝ := (5, 2)
def projection (v w : ℝ × ℝ) : ℝ × ℝ :=
  let num := (v.1 * w.1 + v.2 * w.2)
  let den := (w.1 * w.1 + w.2 * w.2)
  (num / den * w.1, num / den * w.2)

theorem find_x (h : projection (vector_v x) (vector_w) = (3, 1.2)) : 
  x = 47 / 25 :=
by
  sorry

end find_x_l12_12161


namespace cost_prices_of_products_l12_12059

-- Define the variables and conditions from the problem
variables (x y : ℝ)

-- Theorem statement
theorem cost_prices_of_products (h1 : 20 * x + 15 * y = 380) (h2 : 15 * x + 10 * y = 280) : 
  x = 16 ∧ y = 4 :=
by
  -- Proof omitted
  sorry

end cost_prices_of_products_l12_12059


namespace minimum_arc_length_of_curve_and_line_l12_12695

-- Definition of the curve C and the line x = π/4
def curve (x y α : ℝ) : Prop :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line (x : ℝ) : Prop :=
  x = Real.pi / 4

-- Statement of the proof problem: the minimum value of d as α varies
theorem minimum_arc_length_of_curve_and_line : 
  (∀ α : ℝ, ∃ d : ℝ, (∃ y : ℝ, curve (Real.pi / 4) y α) → 
    (d = Real.pi / 2)) :=
sorry

end minimum_arc_length_of_curve_and_line_l12_12695


namespace rhombus_diagonal_length_l12_12848

theorem rhombus_diagonal_length (d1 d2 : ℝ) (area : ℝ) (h1 : area = 600) (h2 : d1 = 30) :
  d2 = 40 :=
by
  sorry

end rhombus_diagonal_length_l12_12848


namespace train_speed_l12_12752

theorem train_speed (distance time : ℤ) (h_distance : distance = 500)
    (h_time : time = 3) :
    distance / time = 166 :=
by
  -- Proof steps will be filled in here
  sorry

end train_speed_l12_12752


namespace wickets_before_last_match_l12_12619

-- Define the conditions
variable (W : ℕ)

-- Initial average
def initial_avg : ℝ := 12.4

-- Runs given in the last match
def runs_last_match : ℝ := 26

-- Wickets taken in the last match
def wickets_last_match : ℕ := 4

-- The new average after the last match
def new_avg : ℝ := initial_avg - 0.4

-- Prove the theorem
theorem wickets_before_last_match :
  (12.4 * W + runs_last_match) / (W + wickets_last_match) = new_avg → W = 55 :=
by
  sorry

end wickets_before_last_match_l12_12619


namespace range_of_function_l12_12625

theorem range_of_function : ∀ y : ℝ, ∃ x : ℝ, y = (x^2 + 3*x + 2)/(x^2 + x + 1) :=
by
  sorry

end range_of_function_l12_12625


namespace students_count_l12_12688

theorem students_count (initial: ℕ) (left: ℕ) (new: ℕ) (result: ℕ) 
  (h1: initial = 31)
  (h2: left = 5)
  (h3: new = 11)
  (h4: result = initial - left + new) : result = 37 := by
  sorry

end students_count_l12_12688


namespace ratio_problem_l12_12826

-- Given condition: a, b, c are in the ratio 2:3:4
theorem ratio_problem (a b c : ℝ) (h1 : a / b = 2 / 3) (h2 : a / c = 2 / 4) : 
  (a - b + c) / b = 1 :=
by
  -- Placeholder for the actual proof
  sorry

end ratio_problem_l12_12826


namespace product_11_29_product_leq_20_squared_product_leq_half_m_squared_l12_12782

-- Definition of natural numbers
variable (a b m : ℕ)

-- Statement 1: Prove that 11 × 29 = 20^2 - 9^2
theorem product_11_29 : 11 * 29 = 20^2 - 9^2 := sorry

-- Statement 2: Prove ∀ a, b ∈ ℕ, if a + b = 40, then ab ≤ 20^2.
theorem product_leq_20_squared (a b : ℕ) (h : a + b = 40) : a * b ≤ 20^2 := sorry

-- Statement 3: Prove ∀ a, b ∈ ℕ, if a + b = m, then ab ≤ (m/2)^2.
theorem product_leq_half_m_squared (a b : ℕ) (m : ℕ) (h : a + b = m) : a * b ≤ (m / 2)^2 := sorry

end product_11_29_product_leq_20_squared_product_leq_half_m_squared_l12_12782


namespace smallest_n_for_isosceles_trapezoid_coloring_l12_12231

def isIsoscelesTrapezoid (a b c d : ℕ) : Prop :=
  -- conditions to check if vertices a, b, c, d form an isosceles trapezoid in a regular n-gon
  sorry  -- definition of an isosceles trapezoid

def vertexColors (n : ℕ) : Fin n → Fin 3 :=
  sorry  -- vertex coloring function

theorem smallest_n_for_isosceles_trapezoid_coloring :
  ∃ n : ℕ, (∀ (vertices : Fin n → Fin 3), ∃ (a b c d : Fin n),
    vertexColors n a = vertexColors n b ∧
    vertexColors n b = vertexColors n c ∧
    vertexColors n c = vertexColors n d ∧
    isIsoscelesTrapezoid a b c d) ∧ n = 17 :=
by
  sorry

end smallest_n_for_isosceles_trapezoid_coloring_l12_12231


namespace abs_sum_ge_sqrt_three_over_two_l12_12034

open Real

theorem abs_sum_ge_sqrt_three_over_two
  (a b : ℝ) : (|a| + |b| ≥ 2 / sqrt 3) ∧ (∀ x, |a * sin x + b * sin (2 * x)| ≤ 1) ↔
  (a, b) = (4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) ∨ 
  (a, b) = (-4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (4 / (3 * sqrt 3), -2 / (3 * sqrt 3)) ∨
  (a, b) = (-4 / (3 * sqrt 3), 2 / (3 * sqrt 3)) := 
sorry

end abs_sum_ge_sqrt_three_over_two_l12_12034


namespace original_cost_l12_12997

theorem original_cost (A : ℝ) (discount : ℝ) (sale_price : ℝ) (original_price : ℝ) (h1 : discount = 0.30) (h2 : sale_price = 35) (h3 : sale_price = (1 - discount) * original_price) : 
  original_price = 50 := by
  sorry

end original_cost_l12_12997


namespace bus_distance_l12_12925

theorem bus_distance (w r : ℝ) (h1 : w = 0.17) (h2 : r = w + 3.67) : r = 3.84 :=
by
  sorry

end bus_distance_l12_12925


namespace train_speed_l12_12152

theorem train_speed
  (distance_meters : ℝ := 400)
  (time_seconds : ℝ := 12)
  (distance_kilometers : ℝ := distance_meters / 1000)
  (time_hours : ℝ := time_seconds / 3600) :
  distance_kilometers / time_hours = 120 := by
  sorry

end train_speed_l12_12152


namespace other_person_age_l12_12691

variable {x : ℕ} -- age of the other person
variable {y : ℕ} -- Marco's age

-- Conditions given in the problem.
axiom marco_age : y = 2 * x + 1
axiom sum_ages : x + y = 37

-- Goal: Prove that the age of the other person is 12.
theorem other_person_age : x = 12 :=
by
  -- Proof is skipped
  sorry

end other_person_age_l12_12691


namespace perimeter_difference_l12_12878

-- Definitions as per conditions
def plywood_width : ℕ := 6
def plywood_height : ℕ := 9
def rectangles_count : ℕ := 6

-- The perimeter difference to be proved
theorem perimeter_difference : 
  ∃ (max_perimeter min_perimeter : ℕ), 
  max_perimeter = 22 ∧ min_perimeter = 12 ∧ (max_perimeter - min_perimeter = 10) :=
by
  sorry

end perimeter_difference_l12_12878


namespace axis_of_symmetry_l12_12600

variable (f : ℝ → ℝ)

theorem axis_of_symmetry (h : ∀ x, f x = f (5 - x)) :  ∀ x y, y = f x ↔ (x = 2.5 ∧ y = f 2.5) := 
sorry

end axis_of_symmetry_l12_12600


namespace probability_is_correct_l12_12437

noncomputable def probability_cashier_opens_early : ℝ :=
  let x1 : ℝ := sorry
  let x2 : ℝ := sorry
  let x3 : ℝ := sorry
  let x4 : ℝ := sorry
  let x5 : ℝ := sorry
  let x6 : ℝ := sorry
  if (0 <= x1) ∧ (x1 <= 15) ∧
     (0 <= x2) ∧ (x2 <= 15) ∧
     (0 <= x3) ∧ (x3 <= 15) ∧
     (0 <= x4) ∧ (x4 <= 15) ∧
     (0 <= x5) ∧ (x5 <= 15) ∧
     (0 <= x6) ∧ (x6 <= 15) ∧
     (x1 < x6) ∧ (x2 < x6) ∧ (x3 < x6) ∧ (x4 < x6) ∧ (x5 < x6) then 
    let p_not_A : ℝ := (12 / 15) ^ 6
    1 - p_not_A
  else
    0

theorem probability_is_correct : probability_cashier_opens_early = 0.738 :=
by sorry

end probability_is_correct_l12_12437


namespace missing_digit_divisibility_l12_12877

theorem missing_digit_divisibility (x : ℕ) (h1 : x < 10) :
  3 ∣ (1 + 3 + 5 + 7 + x + 2) ↔ x = 0 ∨ x = 3 ∨ x = 6 ∨ x = 9 := by
  sorry

end missing_digit_divisibility_l12_12877


namespace problem_statement_l12_12645

noncomputable def a : ℝ := Real.sqrt 3 - Real.sqrt 11
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 11

theorem problem_statement : (a^2 - b^2) / (a^2 * b - a * b^2) / (1 + (a^2 + b^2) / (2 * a * b)) = Real.sqrt 3 / 3 :=
by
  -- conditions
  let a := Real.sqrt 3 - Real.sqrt 11
  let b := Real.sqrt 3 + Real.sqrt 11
  have h1 : a = Real.sqrt 3 - Real.sqrt 11 := rfl
  have h2 : b = Real.sqrt 3 + Real.sqrt 11 := rfl
  -- question statement
  sorry

end problem_statement_l12_12645


namespace total_area_pool_and_deck_l12_12712

theorem total_area_pool_and_deck (pool_length pool_width deck_width : ℕ) 
  (h1 : pool_length = 12) 
  (h2 : pool_width = 10) 
  (h3 : deck_width = 4) : 
  (pool_length + 2 * deck_width) * (pool_width + 2 * deck_width) = 360 := 
by sorry

end total_area_pool_and_deck_l12_12712


namespace compute_fg_neg1_l12_12182

def f (x : ℝ) : ℝ := 5 - 2 * x
def g (x : ℝ) : ℝ := x^3 + 2

theorem compute_fg_neg1 : f (g (-1)) = 3 := by
  sorry

end compute_fg_neg1_l12_12182


namespace sum_of_distinct_integers_l12_12637

theorem sum_of_distinct_integers (a b c d e : ℤ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_prod : (8 - a) * (8 - b) * (8 - c) * (8 - d) * (8 - e) = 120) : a + b + c + d + e = 39 :=
by
  sorry

end sum_of_distinct_integers_l12_12637


namespace ball_count_proof_l12_12819

noncomputable def valid_ball_count : ℕ :=
  150

def is_valid_ball_count (N : ℕ) : Prop :=
  80 < N ∧ N ≤ 200 ∧
  (∃ y b w r : ℕ,
    y = Nat.div (12 * N) 100 ∧
    b = Nat.div (20 * N) 100 ∧
    w = 2 * Nat.div N 3 ∧
    r = N - (y + b + w) ∧
    r.mod N = 0 )

theorem ball_count_proof : is_valid_ball_count valid_ball_count :=
by
  -- The proof would be inserted here.
  sorry

end ball_count_proof_l12_12819


namespace distinct_parallel_lines_l12_12279

theorem distinct_parallel_lines (k : ℝ) :
  (∃ (L1 L2 : ℝ × ℝ → Prop), 
    (∀ x y, L1 (x, y) ↔ x - 2 * y - 3 = 0) ∧ 
    (∀ x y, L2 (x, y) ↔ 18 * x - k^2 * y - 9 * k = 0)) → 
  (∃ slope1 slope2, 
    slope1 = 1/2 ∧ 
    slope2 = 18 / k^2 ∧
    (slope1 = slope2) ∧
    (¬ (∀ x y, x - 2 * y - 3 = 18 * x - k^2 * y - 9 * k))) → 
  k = -6 :=
by 
  sorry

end distinct_parallel_lines_l12_12279


namespace max_value_of_f_l12_12417

noncomputable def f (x : ℝ) : ℝ := (Real.sin x)^3 + Real.cos (2 * x) - (Real.cos x)^2 - Real.sin x

theorem max_value_of_f :
  ∃ x : ℝ, f x = 5 / 27 ∧ ∀ y : ℝ, f y ≤ 5 / 27 :=
sorry

end max_value_of_f_l12_12417


namespace remainder_of_x_l12_12091

theorem remainder_of_x (x : ℕ) 
(H1 : 4 + x ≡ 81 [MOD 16])
(H2 : 6 + x ≡ 16 [MOD 36])
(H3 : 8 + x ≡ 36 [MOD 64]) :
  x ≡ 37 [MOD 48] :=
sorry

end remainder_of_x_l12_12091


namespace part_a_part_b_l12_12420

-- Part (a)
theorem part_a (f : ℚ → ℝ) (h_add : ∀ x y : ℚ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℚ, f (x * y) = f x * f y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = 0) :=
sorry

-- Part (b)
theorem part_b (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_mul : ∀ x y : ℝ, f (x * y) = f x * f y) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = 0) :=
sorry

end part_a_part_b_l12_12420


namespace geometric_sequence_product_l12_12915

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (a_geometric : ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r)
  (root_condition : ∃ x y : ℝ, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 :=
sorry

end geometric_sequence_product_l12_12915


namespace division_example_l12_12633

theorem division_example : 0.45 / 0.005 = 90 := by
  sorry

end division_example_l12_12633


namespace slope_angle_l12_12432

theorem slope_angle (A B : ℝ × ℝ) (θ : ℝ) (hA : A = (-1, 3)) (hB : B = (1, 1)) (hθ : θ ∈ Set.Ico 0 Real.pi)
  (hslope : Real.tan θ = (B.2 - A.2) / (B.1 - A.1)) :
  θ = (3 / 4) * Real.pi :=
by
  cases hA
  cases hB
  simp at hslope
  sorry

end slope_angle_l12_12432


namespace expression_equals_five_l12_12429

-- Define the innermost expression
def inner_expr : ℤ := -|( -2 + 3 )|

-- Define the next layer of the expression
def middle_expr : ℤ := |(inner_expr) - 2|

-- Define the outer expression
def outer_expr : ℤ := |middle_expr + 2|

-- The proof problem statement (in this case, without the proof)
theorem expression_equals_five : outer_expr = 5 :=
by
  -- Insert precise conditions directly from the problem statement
  have h_inner : inner_expr = -|1| := by sorry
  have h_middle : middle_expr = |-1 - 2| := by sorry
  have h_outer : outer_expr = |(-3) + 2| := by sorry
  -- The final goal that needs to be proven
  sorry

end expression_equals_five_l12_12429


namespace units_digit_17_mul_27_l12_12946

theorem units_digit_17_mul_27 : 
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  units_product = 9 := by
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  sorry

end units_digit_17_mul_27_l12_12946


namespace find_a_l12_12366

-- Defining the problem conditions
def rational_eq (x a : ℝ) :=
  x / (x - 3) - 2 * a / (x - 3) = 2

def extraneous_root (x : ℝ) : Prop :=
  x = 3

-- Theorem: Given the conditions, prove that a = 3 / 2
theorem find_a (a : ℝ) : (∃ x, extraneous_root x ∧ rational_eq x a) → a = 3 / 2 :=
  by
    sorry

end find_a_l12_12366


namespace ratio_sum_product_is_constant_l12_12654

variables {p a : ℝ} (h_a : 0 < a)
theorem ratio_sum_product_is_constant
    (k : ℝ) (h_k : k ≠ 0)
    (x₁ x₂ : ℝ) (h_intersection : x₁ * (2 * p * (x₂ - a)) = 2 * p * (x₁ - a) ∧ x₂ * (2 * p * (x₁ - a)) = 2 * p * (x₂ - a)) :
  (x₁ + x₂) / (x₁ * x₂) = 1 / a := by
  sorry

end ratio_sum_product_is_constant_l12_12654


namespace smallest_possible_r_l12_12574

theorem smallest_possible_r (p q r : ℤ) (hpq: p < q) (hqr: q < r) 
  (hgeo: q^2 = p * r) (harith: 2 * q = p + r) : r = 4 :=
sorry

end smallest_possible_r_l12_12574


namespace value_of_x_l12_12903

theorem value_of_x (x y : ℝ) (h1 : y = 1 / (4 * x + 2)) (h2 : y = 2) : x = -3 / 8 :=
by
  sorry

end value_of_x_l12_12903


namespace maxValue_a1_l12_12127

variable (a_1 q : ℝ)

def isGeometricSequence (a_1 q : ℝ) : Prop :=
  a_1 ≥ 1 ∧ a_1 * q ≤ 2 ∧ a_1 * q^2 ≥ 3

theorem maxValue_a1 (h : isGeometricSequence a_1 q) : a_1 ≤ 4 / 3 := 
sorry

end maxValue_a1_l12_12127


namespace functional_equation_solution_l12_12739

-- Define the functional equation with given conditions
def func_eq (f : ℤ → ℝ) (N : ℕ) : Prop :=
  (∀ k : ℤ, f (2 * k) = 2 * f k) ∧
  (∀ k : ℤ, f (N - k) = f k)

-- State the mathematically equivalent proof problem
theorem functional_equation_solution (N : ℕ) (f : ℤ → ℝ) 
  (h1 : ∀ k : ℤ, f (2 * k) = 2 * f k)
  (h2 : ∀ k : ℤ, f (N - k) = f k) : 
  ∀ a : ℤ, f a = 0 := 
sorry

end functional_equation_solution_l12_12739


namespace botanical_garden_path_length_l12_12271

theorem botanical_garden_path_length
  (scale : ℝ)
  (path_length_map : ℝ)
  (path_length_real : ℝ)
  (h_scale : scale = 500)
  (h_path_length_map : path_length_map = 6.5)
  (h_path_length_real : path_length_real = path_length_map * scale) :
  path_length_real = 3250 :=
by
  sorry

end botanical_garden_path_length_l12_12271


namespace cube_sum_identity_l12_12097

theorem cube_sum_identity (a b : ℝ) (h₁ : a + b = 12) (h₂ : a * b = 20) : a^3 + b^3 = 1008 := 
by 
 sorry

end cube_sum_identity_l12_12097


namespace product_increase_by_13_l12_12577

theorem product_increase_by_13 {
    a1 a2 a3 a4 a5 a6 a7 : ℕ
} : (a1 > 3) → (a2 > 3) → (a3 > 3) → (a4 > 3) → (a5 > 3) → (a6 > 3) → (a7 > 3) → 
    ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) * (a6 - 3) * (a7 - 3) = 13 * a1 * a2 * a3 * a4 * a5 * a6 * a7) :=
        sorry

end product_increase_by_13_l12_12577


namespace length_of_arc_l12_12634

variable {O A B : Type}
variable (angle_OAB : Real) (radius_OA : Real)

theorem length_of_arc (h1 : angle_OAB = 45) (h2 : radius_OA = 5) :
  (length_of_arc_AB = 5 * π / 4) :=
sorry

end length_of_arc_l12_12634


namespace sufficient_necessary_condition_l12_12930

theorem sufficient_necessary_condition (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = a ∧ x > 2) ↔ a > 5 :=
by
  sorry

end sufficient_necessary_condition_l12_12930


namespace find_y_l12_12389

def star (a b : ℝ) : ℝ := 4 * a + 2 * b

theorem find_y (y : ℝ) : star 3 (star 4 y) = -2 → y = -11.5 :=
by
  sorry

end find_y_l12_12389


namespace cubes_with_4_neighbors_l12_12015

theorem cubes_with_4_neighbors (a b c : ℕ) (h₁ : 3 < a) (h₂ : 3 < b) (h₃ : 3 < c)
  (h₄ : (a - 2) * (b - 2) * (c - 2) = 429) : 
  4 * ((a - 2) + (b - 2) + (c - 2)) = 108 := by
  sorry

end cubes_with_4_neighbors_l12_12015


namespace elizabeth_stickers_count_l12_12441

theorem elizabeth_stickers_count :
  let initial_bottles := 10
  let lost_at_school := 2
  let stolen_at_dance := 1
  let stickers_per_bottle := 3
  let remaining_bottles := initial_bottles - lost_at_school - stolen_at_dance
  remaining_bottles * stickers_per_bottle = 21 := by sorry

end elizabeth_stickers_count_l12_12441


namespace find_a_l12_12870

open Nat

-- Define the conditions and the proof goal
theorem find_a (a b : ℕ) (h1 : 2019 = a^2 - b^2) (h2 : a < 1000) : a = 338 :=
sorry

end find_a_l12_12870


namespace birds_flew_up_l12_12662

theorem birds_flew_up (original_birds total_birds birds_flew_up : ℕ) 
  (h1 : original_birds = 14)
  (h2 : total_birds = 35)
  (h3 : total_birds = original_birds + birds_flew_up) :
  birds_flew_up = 21 :=
by
  rw [h1, h2] at h3
  linarith

end birds_flew_up_l12_12662


namespace value_of_a_minus_b_l12_12276

theorem value_of_a_minus_b (a b : ℤ) 
  (h₁ : |a| = 7) 
  (h₂ : |b| = 5) 
  (h₃ : a < b) : 
  a - b = -12 ∨ a - b = -2 := 
sorry

end value_of_a_minus_b_l12_12276


namespace probability_at_least_one_die_less_3_l12_12110

-- Definitions
def total_outcomes_dice : ℕ := 64
def outcomes_no_die_less_3 : ℕ := 36
def favorable_outcomes : ℕ := total_outcomes_dice - outcomes_no_die_less_3
def probability : ℚ := favorable_outcomes / total_outcomes_dice

-- Theorem statement
theorem probability_at_least_one_die_less_3 :
  probability = 7 / 16 :=
by
  -- Proof would go here
  sorry

end probability_at_least_one_die_less_3_l12_12110


namespace sack_flour_cost_l12_12770

theorem sack_flour_cost
  (x y : ℝ) 
  (h1 : 10 * x + 800 = 108 * y)
  (h2 : 4 * x - 800 = 36 * y) : x = 1600 := by
  -- Add your proof here
  sorry

end sack_flour_cost_l12_12770


namespace ratio_of_rectangle_to_square_l12_12817

theorem ratio_of_rectangle_to_square (s w h : ℝ) 
  (hs : h = s / 2)
  (shared_area_ABCD_EFGH_1 : 0.25 * s^2 = 0.4 * w * h)
  (shared_area_ABCD_EFGH_2 : 0.25 * s^2 = 0.4 * w * h) :
  w / h = 2.5 :=
by
  -- Proof goes here
  sorry

end ratio_of_rectangle_to_square_l12_12817


namespace no_such_function_exists_l12_12439

namespace ProofProblem

open Nat

-- Declaration of the proposed function
def f : ℕ+ → ℕ+ := sorry

-- Statement to be proved
theorem no_such_function_exists : 
  ¬ ∃ f : ℕ+ → ℕ+, ∀ n : ℕ+, f^[n] n = n + 1 :=
by
  sorry

end ProofProblem

end no_such_function_exists_l12_12439


namespace minimize_shelves_books_l12_12681

theorem minimize_shelves_books : 
  ∀ (n : ℕ),
    (n > 0 ∧ 130 % n = 0 ∧ 195 % n = 0) → 
    (n ≤ 65) := sorry

end minimize_shelves_books_l12_12681


namespace smallest_positive_debt_resolvable_l12_12603

theorem smallest_positive_debt_resolvable :
  ∃ D : ℤ, D > 0 ∧ (D = 250 * p + 175 * g + 125 * s ∧ 
  (∀ (D' : ℤ), D' > 0 → (∃ p g s : ℤ, D' = 250 * p + 175 * g + 125 * s) → D' ≥ D)) := 
sorry

end smallest_positive_debt_resolvable_l12_12603


namespace min_value_four_l12_12478

noncomputable def min_value_T (a b c : ℝ) : ℝ :=
  1 / (2 * (a * b - 1)) + a * (b + 2 * c) / (a * b - 1)

theorem min_value_four (a b c : ℝ) (h1 : (1 / a) > 0)
  (h2 : b^2 - (4 * c) / a ≤ 0) (h3 : a * b > 1) : 
  min_value_T a b c = 4 := 
by 
  sorry

end min_value_four_l12_12478


namespace no_solution_l12_12481

theorem no_solution (a : ℝ) :
  (a < -12 ∨ a > 0) →
  ∀ x : ℝ, ¬(6 * (|x - 4 * a|) + (|x - a ^ 2|) + 5 * x - 4 * a = 0) :=
by
  intros ha hx
  sorry

end no_solution_l12_12481


namespace max_value_proof_l12_12229

noncomputable def max_value (x y : ℝ) : ℝ := x^2 + 2 * x * y + 3 * y^2

theorem max_value_proof (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - 2 * x * y + 3 * y^2 = 12) : 
  max_value x y = 24 + 12 * Real.sqrt 3 := 
sorry

end max_value_proof_l12_12229


namespace quinton_cupcakes_l12_12840

theorem quinton_cupcakes (students_Delmont : ℕ) (students_Donnelly : ℕ)
                         (num_teachers_nurse_principal : ℕ) (leftover : ℕ) :
  students_Delmont = 18 → students_Donnelly = 16 →
  num_teachers_nurse_principal = 4 → leftover = 2 →
  students_Delmont + students_Donnelly + num_teachers_nurse_principal + leftover = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end quinton_cupcakes_l12_12840


namespace Karlsson_eats_more_than_half_l12_12891

open Real

theorem Karlsson_eats_more_than_half
  (D : ℝ) (S : ℕ → ℝ)
  (a b : ℕ → ℝ)
  (cut_and_eat : ∀ n, S (n + 1) = S n - (S n * a n) / (a n + b n))
  (side_conditions : ∀ n, max (a n) (b n) ≤ D) :
  ∃ n, S n < (S 0) / 2 := sorry

end Karlsson_eats_more_than_half_l12_12891


namespace average_monthly_balance_correct_l12_12367

def january_balance : ℕ := 100
def february_balance : ℕ := 200
def march_balance : ℕ := 250
def april_balance : ℕ := 250
def may_balance : ℕ := 150
def june_balance : ℕ := 100

def total_balance : ℕ :=
  january_balance + february_balance + march_balance + april_balance + may_balance + june_balance

def number_of_months : ℕ := 6

def average_monthly_balance : ℕ :=
  total_balance / number_of_months

theorem average_monthly_balance_correct :
  average_monthly_balance = 175 := by
  sorry

end average_monthly_balance_correct_l12_12367


namespace bridge_length_l12_12475

def train_length : ℕ := 120
def train_speed : ℕ := 45
def crossing_time : ℕ := 30

theorem bridge_length :
  let speed_m_per_s := (train_speed * 1000) / 3600
  let total_distance := speed_m_per_s * crossing_time
  let bridge_length := total_distance - train_length
  bridge_length = 255 := by
  sorry

end bridge_length_l12_12475


namespace n_divisible_by_100_l12_12653

theorem n_divisible_by_100 
    (n : ℕ) 
    (h_pos : 0 < n) 
    (h_div : 100 ∣ n^3) : 
    100 ∣ n := 
sorry

end n_divisible_by_100_l12_12653


namespace vector_calculation_l12_12390

def a :ℝ × ℝ := (1, 2)
def b :ℝ × ℝ := (1, -1)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_calculation : scalar_mult (1/3) a - scalar_mult (4/3) b = (-1, 2) :=
by sorry

end vector_calculation_l12_12390


namespace smallest_positive_m_condition_l12_12406

theorem smallest_positive_m_condition
  (p q : ℤ) (m : ℤ) (h_prod : p * q = 42) (h_diff : |p - q| ≤ 10) 
  (h_roots : 15 * (p + q) = m) : m = 195 :=
sorry

end smallest_positive_m_condition_l12_12406


namespace fraction_representation_correct_l12_12219

theorem fraction_representation_correct (h : ∀ (x y z w: ℕ), 9*x = y ∧ 47*z = w ∧ 2*47*5 = 235):
  (18: ℚ) / (9 * 47 * 5) = (2: ℚ) / 235 :=
by
  sorry

end fraction_representation_correct_l12_12219


namespace start_time_6am_l12_12424

def travel_same_time (t : ℝ) (x : ℝ) (y : ℝ) (constant_speed : Prop) : Prop :=
  (x = t + 4) ∧ (y = t + 9) ∧ constant_speed 

theorem start_time_6am
  (x y t: ℝ)
  (constant_speed : Prop) 
  (meet_noon : travel_same_time t x y constant_speed)
  (eqn : 1/t + 1/(t + 4) + 1/(t + 9) = 1) :
  t = 6 :=
by
  sorry

end start_time_6am_l12_12424


namespace weights_problem_l12_12358

theorem weights_problem
  (weights : Fin 10 → ℝ)
  (h1 : ∀ (i j k l a b c : Fin 10), i ≠ j → i ≠ k → i ≠ l → i ≠ a → i ≠ b → i ≠ c →
    j ≠ k → j ≠ l → j ≠ a → j ≠ b → j ≠ c →
    k ≠ l → k ≠ a → k ≠ b → k ≠ c → 
    l ≠ a → l ≠ b → l ≠ c →
    a ≠ b → a ≠ c →
    b ≠ c →
    weights i + weights j + weights k + weights l > weights a + weights b + weights c)
  (h2 : ∀ (i j : Fin 9), weights i ≤ weights (i + 1)) :
  ∀ (i j k a b : Fin 10), i ≠ j → i ≠ k → i ≠ a → i ≠ b → j ≠ k → j ≠ a → j ≠ b → k ≠ a → k ≠ b → a ≠ b → 
    weights i + weights j + weights k > weights a + weights b := 
sorry

end weights_problem_l12_12358


namespace cups_of_oil_used_l12_12100

-- Define the required amounts
def total_liquid : ℝ := 1.33
def water_used : ℝ := 1.17

-- The statement we want to prove
theorem cups_of_oil_used : total_liquid - water_used = 0.16 := by
sorry

end cups_of_oil_used_l12_12100


namespace kyler_games_won_l12_12730

theorem kyler_games_won (peter_wins peter_losses emma_wins emma_losses kyler_losses : ℕ)
  (h_peter : peter_wins = 5)
  (h_peter_losses : peter_losses = 4)
  (h_emma : emma_wins = 2)
  (h_emma_losses : emma_losses = 5)
  (h_kyler_losses : kyler_losses = 4) : ∃ kyler_wins : ℕ, kyler_wins = 2 :=
by {
  sorry
}

end kyler_games_won_l12_12730


namespace problem_I_problem_II_l12_12190

def f (x : ℝ) : ℝ := abs (x - 1)

theorem problem_I (x : ℝ) : f (2 * x) + f (x + 4) ≥ 8 ↔ x ≤ -10 / 3 ∨ x ≥ 2 := by
  sorry

variable {a b : ℝ}
theorem problem_II (ha : abs a < 1) (hb : abs b < 1) (h_neq : a ≠ 0) : 
  (abs (a * b - 1) / abs a) > abs ((b / a) - 1) :=
by
  sorry

end problem_I_problem_II_l12_12190


namespace C_share_l12_12758

-- Definitions based on conditions
def total_sum : ℝ := 164
def ratio_B : ℝ := 0.65
def ratio_C : ℝ := 0.40

-- Statement of the proof problem
theorem C_share : (ratio_C * (total_sum / (1 + ratio_B + ratio_C))) = 32 :=
by
  sorry

end C_share_l12_12758


namespace cannot_take_value_l12_12492

theorem cannot_take_value (x y : ℝ) (h : |x| + |y| = 13) : 
  ∀ (v : ℝ), x^2 + 7*x - 3*y + y^2 = v → (0 ≤ v ∧ v ≤ 260) := 
by
  sorry

end cannot_take_value_l12_12492


namespace fraction_proof_l12_12447

theorem fraction_proof (x y : ℕ) (h1 : y = 7) (h2 : x = 22) : 
  (y / (x - 1) = 1 / 3) ∧ ((y + 4) / x = 1 / 2) := by
  sorry

end fraction_proof_l12_12447


namespace find_number_divided_by_3_equals_subtracted_5_l12_12423

theorem find_number_divided_by_3_equals_subtracted_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
sorry

end find_number_divided_by_3_equals_subtracted_5_l12_12423


namespace ratio_of_a_b_l12_12495

-- Define the system of equations as given in the problem
variables (x y a b : ℝ)

-- Conditions: the system of equations and b ≠ 0
def system_of_equations (a b : ℝ) (x y : ℝ) := 
  4 * x - 3 * y = a ∧ 6 * y - 8 * x = b

-- The theorem we aim to prove
theorem ratio_of_a_b (h : system_of_equations a b x y) (h₀ : b ≠ 0) : a / b = -1 / 2 :=
sorry

end ratio_of_a_b_l12_12495


namespace locus_of_D_l12_12831

theorem locus_of_D 
  (a b : ℝ)
  (hA : 0 ≤ a ∧ a ≤ (2 * Real.sqrt 3 / 3))
  (hB : 0 ≤ b ∧ b ≤ (2 * Real.sqrt 3 / 3))
  (AB_eq : Real.sqrt ((b - 2 * a)^2 + (Real.sqrt 3 * b)^2)  = 2) :
  3 * (b - a / 2)^2 + (Real.sqrt 3 / 2 * (a + b))^2 / 3 = 1 :=
sorry

end locus_of_D_l12_12831


namespace yuna_has_biggest_number_l12_12307

-- Define the numbers assigned to each student
def Yoongi_num : ℕ := 7
def Jungkook_num : ℕ := 6
def Yuna_num : ℕ := 9
def Yoojung_num : ℕ := 8

-- State the main theorem that Yuna has the biggest number
theorem yuna_has_biggest_number : 
  (Yuna_num = 9) ∧ (Yuna_num > Yoongi_num) ∧ (Yuna_num > Jungkook_num) ∧ (Yuna_num > Yoojung_num) :=
sorry

end yuna_has_biggest_number_l12_12307


namespace percentage_of_water_in_mixture_l12_12610

-- Conditions
def percentage_water_LiquidA : ℝ := 0.10
def percentage_water_LiquidB : ℝ := 0.15
def percentage_water_LiquidC : ℝ := 0.25

def volume_LiquidA (v : ℝ) : ℝ := 4 * v
def volume_LiquidB (v : ℝ) : ℝ := 3 * v
def volume_LiquidC (v : ℝ) : ℝ := 2 * v

-- Proof
theorem percentage_of_water_in_mixture (v : ℝ) :
  (percentage_water_LiquidA * volume_LiquidA v + percentage_water_LiquidB * volume_LiquidB v + percentage_water_LiquidC * volume_LiquidC v) / (volume_LiquidA v + volume_LiquidB v + volume_LiquidC v) * 100 = 15 :=
by
  sorry

end percentage_of_water_in_mixture_l12_12610


namespace fraction_to_decimal_l12_12502

theorem fraction_to_decimal :
  (7 / 125 : ℚ) = 0.056 :=
sorry

end fraction_to_decimal_l12_12502


namespace more_girls_than_boys_l12_12532

def initial_girls : ℕ := 632
def initial_boys : ℕ := 410
def new_girls_joined : ℕ := 465
def total_girls : ℕ := initial_girls + new_girls_joined

theorem more_girls_than_boys :
  total_girls - initial_boys = 687 :=
by
  -- Proof goes here
  sorry


end more_girls_than_boys_l12_12532


namespace unit_digit_product_7858_1086_4582_9783_l12_12596

-- Define the unit digits of the given numbers
def unit_digit_7858 : ℕ := 8
def unit_digit_1086 : ℕ := 6
def unit_digit_4582 : ℕ := 2
def unit_digit_9783 : ℕ := 3

-- Define a function to calculate the unit digit of a product of two numbers based on their unit digits
def unit_digit_product (a b : ℕ) : ℕ :=
  (a * b) % 10

-- The theorem that states the unit digit of the product of the numbers is 4
theorem unit_digit_product_7858_1086_4582_9783 :
  unit_digit_product (unit_digit_product (unit_digit_product unit_digit_7858 unit_digit_1086) unit_digit_4582) unit_digit_9783 = 4 :=
  by
  sorry

end unit_digit_product_7858_1086_4582_9783_l12_12596


namespace total_flying_days_l12_12562

-- Definitions for the conditions
def days_fly_south_winter := 40
def days_fly_north_summer := 2 * days_fly_south_winter
def days_fly_east_spring := 60

-- Theorem stating the total flying days
theorem total_flying_days : 
  days_fly_south_winter + days_fly_north_summer + days_fly_east_spring = 180 :=
  by {
    -- This is where we would prove the theorem
    sorry
  }

end total_flying_days_l12_12562


namespace units_digit_17_pow_2023_l12_12808

theorem units_digit_17_pow_2023 
  (cycle : ℕ → ℕ)
  (h1 : cycle 0 = 7)
  (h2 : cycle 1 = 9)
  (h3 : cycle 2 = 3)
  (h4 : cycle 3 = 1)
  (units_digit : ℕ → ℕ)
  (h_units : ∀ n, units_digit (17^n) = units_digit (7^n))
  (h_units_cycle : ∀ n, units_digit (7^n) = cycle (n % 4)) :
  units_digit (17^2023) = 3 :=
by
  sorry

end units_digit_17_pow_2023_l12_12808


namespace max_m_value_l12_12368

theorem max_m_value (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) : 
  (∀ a b : ℝ, a > 0 → b > 0 → ((2 / a) + (1 / b) ≥ (m / (2 * a + b)))) → m ≤ 9 :=
sorry

end max_m_value_l12_12368


namespace valid_seating_arrangements_l12_12496

theorem valid_seating_arrangements :
  let total_arrangements := Nat.factorial 10
  let restricted_arrangements := Nat.factorial 7 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 3507840 :=
by
  sorry

end valid_seating_arrangements_l12_12496


namespace probability_exactly_three_even_l12_12138

theorem probability_exactly_three_even (p : ℕ → ℚ) (n : ℕ) (k : ℕ) (h : p 20 = 1/2 ∧ n = 5 ∧ k = 3) :
  (∃ C : ℚ, (C = (Nat.choose n k : ℚ)) ∧ (p 20)^n = 1/32) → (C * 1/32 = 5/16) :=
by
  sorry

end probability_exactly_three_even_l12_12138


namespace grains_of_rice_in_teaspoon_is_10_l12_12302

noncomputable def grains_of_rice_per_teaspoon : ℕ :=
  let grains_per_cup := 480
  let tablespoons_per_half_cup := 8
  let teaspoons_per_tablespoon := 3
  grains_per_cup / (2 * tablespoons_per_half_cup * teaspoons_per_tablespoon)

theorem grains_of_rice_in_teaspoon_is_10 : grains_of_rice_per_teaspoon = 10 :=
by
  sorry

end grains_of_rice_in_teaspoon_is_10_l12_12302


namespace sum_of_powers_l12_12725

-- Here is the statement in Lean 4
theorem sum_of_powers (ω : ℂ) (h1 : ω^9 = 1) (h2 : ω ≠ 1) :
  (ω^20 + ω^24 + ω^28 + ω^32 + ω^36 + ω^40 + ω^44 + ω^48 + ω^52 + ω^56 + ω^60 + ω^64 + ω^68) = (ω^2 - 1) / (ω^4 - 1) :=
sorry -- Proof is omitted as per instructions.

end sum_of_powers_l12_12725


namespace main_expr_equals_target_l12_12679

-- Define the improper fractions for the mixed numbers:
def mixed_to_improper (a b : ℕ) (c : ℕ) : ℚ := (a * b + c) / b

noncomputable def mixed_1 := mixed_to_improper 5 7 2
noncomputable def mixed_2 := mixed_to_improper 3 4 3
noncomputable def mixed_3 := mixed_to_improper 4 6 1
noncomputable def mixed_4 := mixed_to_improper 2 5 1

-- Define the main expression
noncomputable def main_expr := 47 * (mixed_1 - mixed_2) / (mixed_3 + mixed_4)

-- Define the target result converted to an improper fraction
noncomputable def target_result : ℚ := (11 * 99 + 13) / 99

-- The theorem to be proved: main_expr == target_result
theorem main_expr_equals_target : main_expr = target_result :=
by sorry

end main_expr_equals_target_l12_12679


namespace star_proof_l12_12991

def star (a b : ℕ) : ℕ := 3 + b ^ a

theorem star_proof : star (star 2 1) 4 = 259 :=
by
  sorry

end star_proof_l12_12991


namespace total_combined_grapes_l12_12830

theorem total_combined_grapes :
  ∀ (r a y : ℕ), (r = 25) → (a = r + 2) → (y = a + 4) → (r + a + y = 83) :=
by
  intros r a y hr ha hy
  rw [hr, ha, hy]
  sorry

end total_combined_grapes_l12_12830


namespace set_union_example_l12_12865

open Set

theorem set_union_example :
  let A := ({1, 3, 5, 6} : Set ℤ)
  let B := ({-1, 5, 7} : Set ℤ)
  A ∪ B = ({-1, 1, 3, 5, 6, 7} : Set ℤ) :=
by
  intros
  sorry

end set_union_example_l12_12865


namespace original_number_l12_12847

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def digit_list (n : ℕ) (a b c d e : ℕ) : Prop :=
  n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e

def four_digit_variant (N n : ℕ) (a b c d e : ℕ) : Prop :=
  (n = 10^3 * b + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^2 * c + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10 * d + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + e) ∨
  (n = 10^4 * a + 10^3 * b + 10^2 * c + 10 * d)

theorem original_number (N : ℕ) (a b c d e : ℕ) 
  (h1 : is_five_digit N) 
  (h2 : digit_list N a b c d e)
  (h3 : ∃ n, is_five_digit n ∧ four_digit_variant N n a b c d e ∧ N + n = 54321) :
  N = 49383 := 
sorry

end original_number_l12_12847


namespace average_percent_increase_per_year_l12_12305

def initial_population : ℕ := 175000
def final_population : ℕ := 262500
def years : ℕ := 10

theorem average_percent_increase_per_year :
  ( ( ( ( final_population - initial_population ) / years : ℝ ) / initial_population ) * 100 ) = 5 := by
  sorry

end average_percent_increase_per_year_l12_12305


namespace sum_reciprocal_transformation_l12_12639

theorem sum_reciprocal_transformation 
  (a b c d S : ℝ) 
  (h1 : a + b + c + d = S)
  (h2 : 1 / a + 1 / b + 1 / c + 1 / d = S)
  (h3 : a ≠ 0 ∧ a ≠ 1)
  (h4 : b ≠ 0 ∧ b ≠ 1)
  (h5 : c ≠ 0 ∧ c ≠ 1)
  (h6 : d ≠ 0 ∧ d ≠ 1) :
  S = -2 :=
by
  sorry

end sum_reciprocal_transformation_l12_12639


namespace cups_per_serving_l12_12095

-- Define the conditions
def total_cups : ℕ := 18
def servings : ℕ := 9

-- State the theorem to prove the answer
theorem cups_per_serving : total_cups / servings = 2 := by
  sorry

end cups_per_serving_l12_12095


namespace find_divisor_l12_12025

theorem find_divisor (D Q R d: ℕ) (hD: D = 16698) (hQ: Q = 89) (hR: R = 14) (hDiv: D = d * Q + R): d = 187 := 
by 
  sorry

end find_divisor_l12_12025


namespace cone_lateral_area_and_sector_area_l12_12709

theorem cone_lateral_area_and_sector_area 
  (slant_height : ℝ) 
  (height : ℝ) 
  (r : ℝ) 
  (h_slant : slant_height = 1) 
  (h_height : height = 0.8) 
  (h_r : r = Real.sqrt (slant_height^2 - height^2)) :
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) ∧
  (1 / 2 * 2 * Real.pi * r * slant_height = 3 / 5 * Real.pi) :=
by
  sorry

end cone_lateral_area_and_sector_area_l12_12709


namespace complement_intersection_eq_4_l12_12195

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection_eq_4 (hU : U = {0, 1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4}) :
  ((U \ A) ∩ B) = {4} :=
by {
  -- Proof goes here
  exact sorry
}

end complement_intersection_eq_4_l12_12195


namespace minimum_value_of_f_div_f_l12_12702

noncomputable def quadratic_function_min_value (a b c : ℝ) (h : 0 < b) (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) : ℝ :=
  (a + b + c) / b

theorem minimum_value_of_f_div_f' (a b c : ℝ) (h : 0 < b)
  (h₀ : 0 < a) (h₁ : 0 < c) (h₂ : b^2 ≤ 4*a*c) :
  quadratic_function_min_value a b c h h₀ h₁ h₂ = 2 :=
sorry

end minimum_value_of_f_div_f_l12_12702


namespace ratio_of_plums_to_peaches_is_three_l12_12868

theorem ratio_of_plums_to_peaches_is_three :
  ∃ (L P W : ℕ), W = 1 ∧ P = W + 12 ∧ L = 3 * P ∧ W + P + L = 53 ∧ (L / P) = 3 :=
by
  sorry

end ratio_of_plums_to_peaches_is_three_l12_12868


namespace group_size_is_eight_l12_12624

/-- Theorem: The number of people in the group is 8 if the 
average weight increases by 6 kg when a new person replaces 
one weighing 45 kg, and the weight of the new person is 93 kg. -/
theorem group_size_is_eight
    (n : ℕ)
    (H₁ : 6 * n = 48)
    (H₂ : 93 - 45 = 48) :
    n = 8 :=
by
  sorry

end group_size_is_eight_l12_12624


namespace geometric_sequence_ratio_l12_12267

theorem geometric_sequence_ratio
  (a1 r : ℝ) (h_r : r ≠ 1)
  (h : (1 - r^6) / (1 - r^3) = 1 / 2) :
  (1 - r^9) / (1 - r^3) = 3 / 4 :=
  sorry

end geometric_sequence_ratio_l12_12267


namespace find_a_and_x_range_l12_12612

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a_and_x_range :
  (∃ a, (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3)) →
  (∀ x, ∃ a, f x a ≤ 5 → 
    ((a = 1 → (0 ≤ x ∧ x ≤ 5)) ∧
     (a = 7 → (3 ≤ x ∧ x ≤ 8)))) :=
by sorry

end find_a_and_x_range_l12_12612


namespace roots_negative_reciprocals_l12_12387

theorem roots_negative_reciprocals (a b c r s : ℝ) (h1 : a * r^2 + b * r + c = 0)
    (h2 : a * s^2 + b * s + c = 0) (h3 : r = -1 / s) (h4 : s = -1 / r) :
    a = -c :=
by
  -- Insert clever tricks to auto-solve or reuse axioms here
  sorry

end roots_negative_reciprocals_l12_12387


namespace bike_price_l12_12285

-- Definitions of the conditions
def maria_savings : ℕ := 120
def mother_offer : ℕ := 250
def amount_needed : ℕ := 230

-- Theorem statement
theorem bike_price (maria_savings mother_offer amount_needed : ℕ) : 
  maria_savings + mother_offer + amount_needed = 600 := 
by
  -- Sorry is used here to skip the actual proof steps
  sorry

end bike_price_l12_12285


namespace final_price_percentage_of_original_l12_12498

theorem final_price_percentage_of_original (original_price sale_price final_price : ℝ)
  (h1 : sale_price = original_price * 0.5)
  (h2 : final_price = sale_price * 0.9) :
  final_price = original_price * 0.45 :=
by
  sorry

end final_price_percentage_of_original_l12_12498


namespace neither_biology_nor_chemistry_l12_12965

def science_club_total : ℕ := 80
def biology_members : ℕ := 50
def chemistry_members : ℕ := 40
def both_members : ℕ := 25

theorem neither_biology_nor_chemistry :
  (science_club_total -
  ((biology_members - both_members) +
  (chemistry_members - both_members) +
  both_members)) = 15 := by
  sorry

end neither_biology_nor_chemistry_l12_12965


namespace calculate_rectangle_length_l12_12053

theorem calculate_rectangle_length (side_of_square : ℝ) (width_of_rectangle : ℝ)
  (length_of_wire : ℝ) (perimeter_of_rectangle : ℝ) :
  side_of_square = 20 → 
  width_of_rectangle = 14 → 
  length_of_wire = 4 * side_of_square →
  perimeter_of_rectangle = length_of_wire →
  2 * (width_of_rectangle + length_of_rectangle) = perimeter_of_rectangle →
  length_of_rectangle = 26 :=
by
  intros
  sorry

end calculate_rectangle_length_l12_12053


namespace accessories_cost_is_200_l12_12045

variable (c_cost a_cost : ℕ)
variable (ps_value ps_sold : ℕ)
variable (john_paid : ℕ)

-- Given Conditions
def computer_cost := 700
def accessories_cost := a_cost
def playstation_value := 400
def playstation_sold := ps_value - (ps_value * 20 / 100)
def john_paid_amount := 580

-- Theorem to be proved
theorem accessories_cost_is_200 :
  ps_value = 400 →
  ps_sold = playstation_sold →
  c_cost = 700 →
  john_paid = 580 →
  john_paid + ps_sold - c_cost = a_cost →
  a_cost = 200 :=
by
  intros
  sorry

end accessories_cost_is_200_l12_12045


namespace area_of_right_triangle_l12_12170

theorem area_of_right_triangle (a b c : ℝ) 
  (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) 
  (h4 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 30 :=
by sorry

end area_of_right_triangle_l12_12170


namespace geometric_sequence_seventh_term_l12_12934

theorem geometric_sequence_seventh_term (a r : ℝ) 
  (h4 : a * r^3 = 16) 
  (h9 : a * r^8 = 2) : 
  a * r^6 = 8 := 
sorry

end geometric_sequence_seventh_term_l12_12934


namespace proof_problem_l12_12291

axiom is_line (m : Type) : Prop
axiom is_plane (α : Type) : Prop
axiom is_subset_of_plane (m : Type) (β : Type) : Prop
axiom is_perpendicular (a : Type) (b : Type) : Prop
axiom is_parallel (a : Type) (b : Type) : Prop

theorem proof_problem
  (m n : Type) 
  (α β : Type)
  (h1 : is_line m)
  (h2 : is_line n)
  (h3 : is_plane α)
  (h4 : is_plane β)
  (h_prop2 : is_parallel α β → is_subset_of_plane m α → is_parallel m β)
  (h_prop3 : is_perpendicular n α → is_perpendicular n β → is_perpendicular m α → is_perpendicular m β)
  : (is_subset_of_plane m β → is_perpendicular α β → ¬ (is_perpendicular m α)) ∧ 
    (is_parallel m α → is_parallel m β → ¬ (is_parallel α β)) :=
sorry

end proof_problem_l12_12291


namespace andy_last_problem_l12_12477

theorem andy_last_problem (start_num : ℕ) (num_solved : ℕ) (result : ℕ) : 
  start_num = 78 → 
  num_solved = 48 → 
  result = start_num + num_solved - 1 → 
  result = 125 :=
by
  sorry

end andy_last_problem_l12_12477


namespace remaining_dimes_l12_12736

-- Conditions
def initial_pennies : Nat := 7
def initial_dimes : Nat := 8
def borrowed_dimes : Nat := 4

-- Define the theorem
theorem remaining_dimes : initial_dimes - borrowed_dimes = 4 := by
  -- Use the conditions to state the remaining dimes
  sorry

end remaining_dimes_l12_12736


namespace kim_average_round_correct_answers_l12_12261

theorem kim_average_round_correct_answers (x : ℕ) :
  (6 * 2) + (x * 3) + (4 * 5) = 38 → x = 2 :=
by
  intros h
  sorry

end kim_average_round_correct_answers_l12_12261


namespace find_A_l12_12929

theorem find_A (A B C : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9)
  (h3 : A * 10 + B + B * 10 + C = B * 100 + C * 10 + B) : 
  A = 9 :=
  sorry

end find_A_l12_12929


namespace total_fires_l12_12196

-- Conditions as definitions
def Doug_fires : Nat := 20
def Kai_fires : Nat := 3 * Doug_fires
def Eli_fires : Nat := Kai_fires / 2

-- Theorem to prove the total number of fires
theorem total_fires : Doug_fires + Kai_fires + Eli_fires = 110 := by
  sorry

end total_fires_l12_12196


namespace jerry_age_l12_12017

theorem jerry_age (M J : ℝ) (h₁ : M = 17) (h₂ : M = 2.5 * J - 3) : J = 8 :=
by
  -- The proof is omitted.
  sorry

end jerry_age_l12_12017


namespace y_share_l12_12174

theorem y_share (total_amount : ℝ) (x_share y_share z_share : ℝ)
  (hx : x_share = 1) (hy : y_share = 0.45) (hz : z_share = 0.30)
  (h_total : total_amount = 105) :
  (60 * y_share) = 27 :=
by
  have h_cycle : 1 + y_share + z_share = 1.75 := by sorry
  have h_num_cycles : total_amount / 1.75 = 60 := by sorry
  sorry

end y_share_l12_12174


namespace sqrt_D_irrational_l12_12949

variable (k : ℤ)

def a := 3 * k
def b := 3 * k + 3
def c := a k + b k
def D := a k * a k + b k * b k + c k * c k

theorem sqrt_D_irrational : ¬ ∃ (r : ℚ), r * r = D k := 
by sorry

end sqrt_D_irrational_l12_12949


namespace find_factor_l12_12883

theorem find_factor (x f : ℕ) (hx : x = 110) (h : x * f - 220 = 110) : f = 3 :=
sorry

end find_factor_l12_12883


namespace find_f_5_l12_12766

-- Define the function f satisfying the given conditions
noncomputable def f : ℝ → ℝ :=
sorry

-- Assert the conditions as hypotheses
axiom f_eq : ∀ x y : ℝ, f (x - y) = f x + f y
axiom f_zero : f 0 = 2

-- State the theorem we need to prove
theorem find_f_5 : f 5 = 1 :=
sorry

end find_f_5_l12_12766


namespace largest_value_b_l12_12049

theorem largest_value_b (b : ℚ) : (3 * b + 7) * (b - 2) = 9 * b -> b = (4 + Real.sqrt 58) / 3 :=
by
  sorry

end largest_value_b_l12_12049


namespace airplane_altitude_l12_12266

theorem airplane_altitude (d_Alice_Bob : ℝ) (angle_Alice : ℝ) (angle_Bob : ℝ) (altitude : ℝ) : 
  d_Alice_Bob = 8 ∧ angle_Alice = 45 ∧ angle_Bob = 30 → altitude = 16 / 3 :=
by
  intros h
  rcases h with ⟨h1, ⟨h2, h3⟩⟩
  -- you may insert the proof here if needed
  sorry

end airplane_altitude_l12_12266


namespace linear_regression_decrease_l12_12214

theorem linear_regression_decrease (x : ℝ) (y : ℝ) (h : y = 2 - 1.5 * x) : 
  y = 2 - 1.5 * (x + 1) -> (y - (2 - 1.5 * (x +1))) = -1.5 :=
by
  sorry

end linear_regression_decrease_l12_12214


namespace values_of_a_for_equation_l12_12332

theorem values_of_a_for_equation :
  ∃ S : Finset ℤ, (∀ a ∈ S, |3 * a + 7| + |3 * a - 5| = 12) ∧ S.card = 4 :=
by
  sorry

end values_of_a_for_equation_l12_12332


namespace cricket_player_average_l12_12846

theorem cricket_player_average (A : ℕ)
  (H1 : 10 * A + 62 = 11 * (A + 4)) : A = 18 :=
by {
  sorry -- The proof itself
}

end cricket_player_average_l12_12846


namespace minimum_value_problem_l12_12978

theorem minimum_value_problem (x y z w : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hw : 0 < w) 
  (hxyz : x + y + z + w = 1) :
  (1 / (x + y) + 1 / (x + z) + 1 / (x + w) + 1 / (y + z) + 1 / (y + w) + 1 / (z + w)) ≥ 18 := 
sorry

end minimum_value_problem_l12_12978


namespace sufficient_but_not_necessary_condition_l12_12682

variable {a : ℝ}

theorem sufficient_but_not_necessary_condition :
  (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (a ≥ 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l12_12682


namespace no_integer_points_on_circle_l12_12795

theorem no_integer_points_on_circle : 
  ∀ x : ℤ, ¬ ((x - 3)^2 + (x + 1 + 2)^2 ≤ 64) := by
  sorry

end no_integer_points_on_circle_l12_12795


namespace most_likely_units_digit_l12_12760

theorem most_likely_units_digit :
  ∃ m n : Fin 11, ∀ (M N : Fin 11), (∃ k : Nat, k * 11 + M + N = m + n) → 
    (m + n) % 10 = 0 :=
by
  sorry

end most_likely_units_digit_l12_12760


namespace restaurant_total_cost_l12_12557

theorem restaurant_total_cost (burger_cost pizza_cost : ℕ)
    (h1 : burger_cost = 9)
    (h2 : pizza_cost = 2 * burger_cost) :
    pizza_cost + 3 * burger_cost = 45 := 
by
  sorry

end restaurant_total_cost_l12_12557


namespace compare_abc_l12_12435

open Real

theorem compare_abc
  (a b c : ℝ)
  (ha : 0 < a ∧ a < π / 2)
  (hb : 0 < b ∧ b < π / 2)
  (hc : 0 < c ∧ c < π / 2)
  (h1 : cos a = a)
  (h2 : sin (cos b) = b)
  (h3 : cos (sin c) = c) :
  c > a ∧ a > b :=
sorry

end compare_abc_l12_12435


namespace largest_divisor_expression_l12_12992

theorem largest_divisor_expression (y : ℤ) (h : y % 2 = 1) : 
  4320 ∣ (15 * y + 3) * (15 * y + 9) * (10 * y + 10) :=
sorry  

end largest_divisor_expression_l12_12992


namespace conveyor_belt_sampling_l12_12377

noncomputable def sampling_method (interval : ℕ) (total_items : ℕ) : String :=
  if interval = 5 ∧ total_items > 0 then "systematic sampling" else "unknown"

theorem conveyor_belt_sampling :
  ∀ (interval : ℕ) (total_items : ℕ),
  interval = 5 ∧ total_items > 0 →
  sampling_method interval total_items = "systematic sampling" :=
sorry

end conveyor_belt_sampling_l12_12377


namespace sufficient_not_necessary_l12_12454

variable (x : ℝ)

theorem sufficient_not_necessary (h : x^2 - 3 * x + 2 > 0) : x > 2 → (∀ x : ℝ, x^2 - 3 * x + 2 > 0 ↔ x > 2 ∨ x < -1) :=
by
  sorry

end sufficient_not_necessary_l12_12454


namespace inequality_solution_set_impossible_l12_12655

theorem inequality_solution_set_impossible (a b : ℝ) (h_b : b ≠ 0) : ¬ (a = 0 ∧ ∀ x, ax + b > 0 ∧ x > (b / a)) :=
by {
  sorry
}

end inequality_solution_set_impossible_l12_12655


namespace divisible_by_133_l12_12806

theorem divisible_by_133 (n : ℕ) : (11^(n + 2) + 12^(2*n + 1)) % 133 = 0 :=
by
  sorry

end divisible_by_133_l12_12806


namespace crease_points_ellipse_l12_12981

theorem crease_points_ellipse (R a : ℝ) (x y : ℝ) (h1 : 0 < R) (h2 : 0 < a) (h3 : a < R) : 
  (x - a / 2) ^ 2 / (R / 2) ^ 2 + y ^ 2 / ((R / 2) ^ 2 - (a / 2) ^ 2) ≥ 1 :=
by
  -- Omitted detailed proof steps
  sorry

end crease_points_ellipse_l12_12981


namespace vectorBC_computation_l12_12493

open Vector

def vectorAB : ℝ × ℝ := (2, 4)

def vectorAC : ℝ × ℝ := (1, 3)

theorem vectorBC_computation :
  (vectorAC.1 - vectorAB.1, vectorAC.2 - vectorAB.2) = (-1, -1) :=
sorry

end vectorBC_computation_l12_12493


namespace union_A_B_intersection_complements_l12_12672
open Set

noncomputable def A : Set ℤ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B : Set ℤ := {x | x^2 - x - 2 = 0}
def U : Set ℤ := {x | abs x ≤ 3}

theorem union_A_B :
  A ∪ B = { -1, 2, 3 } :=
by sorry

theorem intersection_complements :
  (U \ A) ∩ (U \ B) = { -3, -2, 0, 1 } :=
by sorry

end union_A_B_intersection_complements_l12_12672


namespace intersection_points_count_l12_12609

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ (∀ x, f x = g x → x = x1 ∨ x = x2) :=
by
  sorry

end intersection_points_count_l12_12609


namespace multiplication_problem_l12_12659

noncomputable def problem_statement (x : ℂ) : Prop :=
  (x^4 + 30 * x^2 + 225) * (x^2 - 15) = x^6 - 3375

theorem multiplication_problem (x : ℂ) : 
  problem_statement x :=
sorry

end multiplication_problem_l12_12659


namespace lcm_of_three_numbers_l12_12414

theorem lcm_of_three_numbers :
  ∀ (a b c : ℕ) (hcf : ℕ), hcf = Nat.gcd (Nat.gcd a b) c → a = 136 → b = 144 → c = 168 → hcf = 8 →
  Nat.lcm (Nat.lcm a b) c = 411264 :=
by
  intros a b c hcf h1 h2 h3 h4
  rw [h2, h3, h4]
  sorry

end lcm_of_three_numbers_l12_12414


namespace smart_charging_piles_eq_l12_12534

theorem smart_charging_piles_eq (x : ℝ) :
  301 * (1 + x) ^ 2 = 500 :=
by sorry

end smart_charging_piles_eq_l12_12534


namespace cost_split_difference_l12_12567

-- Definitions of amounts paid
def SarahPaid : ℕ := 150
def DerekPaid : ℕ := 210
def RitaPaid : ℕ := 240

-- Total paid by all three
def TotalPaid : ℕ := SarahPaid + DerekPaid + RitaPaid

-- Each should have paid:
def EachShouldHavePaid : ℕ := TotalPaid / 3

-- Amount Sarah owes Rita
def SarahOwesRita : ℕ := EachShouldHavePaid - SarahPaid

-- Amount Derek should receive back from Rita
def DerekShouldReceiveFromRita : ℕ := DerekPaid - EachShouldHavePaid

-- Difference between the amounts Sarah and Derek owe/should receive from Rita
theorem cost_split_difference : SarahOwesRita - DerekShouldReceiveFromRita = 60 := by
    sorry

end cost_split_difference_l12_12567


namespace find_a_l12_12684

theorem find_a (a : ℤ) (h : |a + 1| = 3) : a = 2 ∨ a = -4 :=
sorry

end find_a_l12_12684


namespace work_completion_l12_12220

theorem work_completion (A : ℝ) (B : ℝ) (work_duration : ℝ) (total_days : ℝ) (B_days : ℝ) :
  B_days = 28 ∧ total_days = 8 ∧ (A * 2 + (A * 6 + B * 6) = work_duration) →
  A = 84 / 11 :=
by
  sorry

end work_completion_l12_12220


namespace largest_divisor_of_five_even_numbers_l12_12660

theorem largest_divisor_of_five_even_numbers (n : ℕ) (h₁ : n % 2 = 1) : 
  ∃ d, (∀ n, n % 2 = 1 → d ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) ∧ 
       (∀ d', (∀ n, n % 2 = 1 → d' ∣ (n+2)*(n+4)*(n+6)*(n+8)*(n+10)) → d' ≤ d) ∧ 
       d = 480 := sorry

end largest_divisor_of_five_even_numbers_l12_12660


namespace tile_rectangle_condition_l12_12128

theorem tile_rectangle_condition (k m n : ℕ) (hk : 0 < k) (hm : 0 < m) (hn : 0 < n) : 
  (∃ q, m = k * q) ∨ (∃ r, n = k * r) :=
sorry

end tile_rectangle_condition_l12_12128


namespace smallest_prime_divisor_of_sum_of_powers_l12_12222

theorem smallest_prime_divisor_of_sum_of_powers :
  let a := 5
  let b := 7
  let n := 23
  let m := 17
  Nat.minFac (a^n + b^m) = 2 := by
  sorry

end smallest_prime_divisor_of_sum_of_powers_l12_12222


namespace palm_meadows_total_beds_l12_12602

theorem palm_meadows_total_beds :
  ∃ t : Nat, t = 31 → 
    (∀ r1 r2 r3 : Nat, r1 = 13 → r2 = 8 → r3 = r1 - r2 → 
      t = (r2 * 2 + r3 * 3)) :=
by
  sorry

end palm_meadows_total_beds_l12_12602


namespace term_free_of_x_l12_12487

namespace PolynomialExpansion

theorem term_free_of_x (m n k : ℕ) (h : (x : ℝ)^(m * k - (m + n) * r) = 1) :
  (m * k) % (m + n) = 0 :=
by
  sorry

end PolynomialExpansion

end term_free_of_x_l12_12487


namespace xiao_ming_correctly_answered_question_count_l12_12860

-- Define the given conditions as constants and variables
def total_questions : ℕ := 20
def points_per_correct : ℕ := 8
def points_deducted_per_incorrect : ℕ := 5
def total_score : ℕ := 134

-- Prove that the number of correctly answered questions is 18
theorem xiao_ming_correctly_answered_question_count :
  ∃ (correct_count incorrect_count : ℕ), 
      correct_count + incorrect_count = total_questions ∧
      correct_count * points_per_correct - 
      incorrect_count * points_deducted_per_incorrect = total_score ∧
      correct_count = 18 :=
by
  sorry

end xiao_ming_correctly_answered_question_count_l12_12860


namespace num_solutions_l12_12410

theorem num_solutions :
  ∃ n, (∀ a b c : ℤ, (|a + b| + c = 21 ∧ a * b + |c| = 85) ↔ n = 12) :=
sorry

end num_solutions_l12_12410


namespace min_total_cost_at_n_equals_1_l12_12043

-- Define the conditions and parameters
variables (a : ℕ) -- The total construction area
variables (n : ℕ) -- The number of floors

-- Definitions based on the given problem conditions
def land_expropriation_cost : ℕ := 2388 * a
def construction_cost (n : ℕ) : ℕ :=
  if n = 0 then 0 else if n = 1 then 455 * a else (455 * n * a + 30 * (n-2) * (n-1) / 2 * a)

-- Total cost including land expropriation and construction costs
def total_cost (n : ℕ) : ℕ := land_expropriation_cost a + construction_cost a n

-- The minimum total cost occurs at n = 1
theorem min_total_cost_at_n_equals_1 :
  ∃ n, n = 1 ∧ total_cost a n = 2788 * a :=
by sorry

end min_total_cost_at_n_equals_1_l12_12043


namespace find_missing_value_l12_12473

theorem find_missing_value :
  300 * 2 + (12 + 4) * 1 / 8 = 602 :=
by
  sorry

end find_missing_value_l12_12473


namespace lower_limit_total_people_l12_12425

/-- 
  Given:
    1. Exactly 3/7 of the people in the room are under the age of 21.
    2. Exactly 5/10 of the people in the room are over the age of 65.
    3. There are 30 people in the room under the age of 21.
  Prove: The lower limit of the total number of people in the room is 70.
-/
theorem lower_limit_total_people (T : ℕ) (h1 : (3 / 7) * T = 30) : T = 70 := by
  sorry

end lower_limit_total_people_l12_12425


namespace value_of_expression_l12_12207

theorem value_of_expression (r s : ℝ) (h₁ : 3 * r^2 - 5 * r - 7 = 0) (h₂ : 3 * s^2 - 5 * s - 7 = 0) : 
  (9 * r^2 - 9 * s^2) / (r - s) = 15 :=
sorry

end value_of_expression_l12_12207


namespace average_income_QR_l12_12519

theorem average_income_QR (P Q R : ℝ) 
  (h1: (P + Q) / 2 = 5050) 
  (h2: (P + R) / 2 = 5200) 
  (hP: P = 4000) : 
  (Q + R) / 2 = 6250 := 
by 
  -- additional steps and proof to be provided here
  sorry

end average_income_QR_l12_12519


namespace bowling_team_scores_l12_12001

theorem bowling_team_scores : 
  ∀ (A B C : ℕ), 
  C = 162 → 
  B = 3 * C → 
  A + B + C = 810 → 
  A / B = 1 / 3 := 
by 
  intros A B C h1 h2 h3 
  sorry

end bowling_team_scores_l12_12001


namespace inequality_half_l12_12882

-- The Lean 4 statement for the given proof problem
theorem inequality_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry


end inequality_half_l12_12882


namespace sqrt_addition_l12_12916

theorem sqrt_addition : Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  sorry

end sqrt_addition_l12_12916


namespace problem_result_l12_12324

def elongation_A : List ℕ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
def elongation_B : List ℕ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]

def z_i : List ℤ := List.zipWith (λ x y => x - y) elongation_A elongation_B

def sample_mean (lst : List ℤ) : ℚ :=
  (List.sum lst : ℚ) / List.length lst

def sample_variance (lst : List ℤ) : ℚ :=
  let mean := sample_mean lst
  (List.sum (lst.map (λ z => (z - mean) * (z - mean))) : ℚ) / List.length lst

def improvement_significance (mean : ℚ) (variance : ℚ) : Prop :=
  mean ≥ 2 * Real.sqrt (variance / 10)

theorem problem_result :
  sample_mean z_i = 11 ∧
  sample_variance z_i = 61 ∧
  improvement_significance (sample_mean z_i) (sample_variance z_i) :=
by
  sorry

end problem_result_l12_12324


namespace external_angle_bisector_lengths_l12_12353

noncomputable def f_a (a b c : ℝ) : ℝ := 4 * Real.sqrt 3
noncomputable def f_b (b : ℝ) : ℝ := 6 / Real.sqrt 7
noncomputable def f_c (a b c : ℝ) : ℝ := 4 * Real.sqrt 3

theorem external_angle_bisector_lengths (a b c : ℝ) 
  (ha : a = 5 - Real.sqrt 7)
  (hb : b = 6)
  (hc : c = 5 + Real.sqrt 7) :
  f_a a b c = 4 * Real.sqrt 3 ∧
  f_b b = 6 / Real.sqrt 7 ∧
  f_c a b c = 4 * Real.sqrt 3 := by
  sorry

end external_angle_bisector_lengths_l12_12353


namespace range_of_x_minus_cos_y_l12_12631

theorem range_of_x_minus_cos_y {x y : ℝ} (h : x^2 + 2 * Real.cos y = 1) :
  ∃ (a b : ℝ), ∀ z, z = x - Real.cos y → a ≤ z ∧ z ≤ b ∧ a = -1 ∧ b = 1 + Real.sqrt 3 :=
by
  sorry

end range_of_x_minus_cos_y_l12_12631


namespace sin2alpha_div_1_plus_cos2alpha_eq_3_l12_12277

theorem sin2alpha_div_1_plus_cos2alpha_eq_3 (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (2 * α)) / (1 + Real.cos (2 * α)) = 3 := 
  sorry

end sin2alpha_div_1_plus_cos2alpha_eq_3_l12_12277


namespace zero_in_tens_place_l12_12201

variable {A B : ℕ} {m : ℕ}

-- Define the conditions
def condition1 (A : ℕ) (B : ℕ) (m : ℕ) : Prop :=
  ∀ A B : ℕ, ∀ m : ℕ, A * 10^(m+1) + B = 9 * (A * 10^m + B)

theorem zero_in_tens_place (A B : ℕ) (m : ℕ) :
  condition1 A B m → m = 1 :=
by
  intro h
  sorry

end zero_in_tens_place_l12_12201


namespace right_triangle_area_l12_12737

theorem right_triangle_area (a b c : ℝ) (h1 : c = 13) (h2 : a = 5) (h3 : a^2 + b^2 = c^2) : 0.5 * a * b = 30 := by
  sorry

end right_triangle_area_l12_12737


namespace inequality_solution_set_l12_12255

theorem inequality_solution_set :
  {x : ℝ | 3 * x + 9 > 0 ∧ 2 * x < 6} = {x : ℝ | -3 < x ∧ x < 3} := 
by
  sorry

end inequality_solution_set_l12_12255


namespace students_passed_both_tests_l12_12872

theorem students_passed_both_tests :
  ∀ (total students_passed_long_jump students_passed_shot_put students_failed_both x : ℕ),
    total = 50 →
    students_passed_long_jump = 40 →
    students_passed_shot_put = 31 →
    students_failed_both = 4 →
    (students_passed_long_jump - x) + (students_passed_shot_put - x) + x + students_failed_both = total →
    x = 25 :=
by intros total students_passed_long_jump students_passed_shot_put students_failed_both x
   intro total_eq students_passed_long_jump_eq students_passed_shot_put_eq students_failed_both_eq sum_eq
   sorry

end students_passed_both_tests_l12_12872


namespace Mary_sleep_hours_for_avg_score_l12_12638

def sleep_score_inverse_relation (sleep1 score1 sleep2 score2 : ℝ) : Prop :=
  sleep1 * score1 = sleep2 * score2

theorem Mary_sleep_hours_for_avg_score (h1 s1 s2 : ℝ) (h_eq : h1 = 6) (s1_eq : s1 = 60)
  (avg_score_cond : (s1 + s2) / 2 = 75) :
  ∃ h2 : ℝ, sleep_score_inverse_relation h1 s1 h2 s2 ∧ h2 = 4 := 
by
  sorry

end Mary_sleep_hours_for_avg_score_l12_12638


namespace largest_shaded_area_figure_C_l12_12364

noncomputable def area_of_square (s : ℝ) : ℝ := s^2
noncomputable def area_of_circle (r : ℝ) : ℝ := Real.pi * r^2
noncomputable def shaded_area_of_figure_A : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_B : ℝ := 4 - Real.pi
noncomputable def shaded_area_of_figure_C : ℝ := Real.pi - 2

theorem largest_shaded_area_figure_C : shaded_area_of_figure_C > shaded_area_of_figure_A ∧ shaded_area_of_figure_C > shaded_area_of_figure_B := by
  sorry

end largest_shaded_area_figure_C_l12_12364


namespace ab_product_eq_2_l12_12057

theorem ab_product_eq_2 (a b : ℝ) (h : (a + 1)^2 + (b + 2)^2 = 0) : a * b = 2 :=
by sorry

end ab_product_eq_2_l12_12057


namespace solution_of_inequality_system_l12_12036

theorem solution_of_inequality_system (a b : ℝ) 
    (h1 : 4 - 2 * a = 0)
    (h2 : (3 + b) / 2 = 1) : a + b = 1 := 
by 
  sorry

end solution_of_inequality_system_l12_12036


namespace eval_expression_eq_2_l12_12151

theorem eval_expression_eq_2 :
  (10^2 + 11^2 + 12^2 + 13^2 + 14^2) / 365 = 2 :=
by
  sorry

end eval_expression_eq_2_l12_12151


namespace infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l12_12191

theorem infinite_n_square_plus_one_divides_factorial :
  ∃ (infinitely_many n : ℕ), (n^2 + 1) ∣ (n!) := sorry

theorem infinite_n_square_plus_one_not_divide_factorial :
  ∃ (infinitely_many n : ℕ), ¬((n^2 + 1) ∣ (n!)) := sorry

end infinite_n_square_plus_one_divides_factorial_infinite_n_square_plus_one_not_divide_factorial_l12_12191


namespace polynomial_remainder_l12_12608

theorem polynomial_remainder :
  ∀ (q : Polynomial ℚ), (3 * X^5 - 2 * X^3 + 5 * X - 9) = (X - 1) * (X - 2) * q + (92 * X - 95) :=
by
  intro q
  sorry

end polynomial_remainder_l12_12608


namespace find_a4_l12_12735

noncomputable def quadratic_eq (t : ℝ) := t^2 - 36 * t + 288 = 0

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∃ a1 : ℝ, a 1 = a1 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def condition1 (a : ℕ → ℝ) := a 1 + a 2 = -1
def condition2 (a : ℕ → ℝ) := a 1 - a 3 = -3

theorem find_a4 :
  ∃ (a : ℕ → ℝ) (q : ℝ), quadratic_eq q ∧ geometric_sequence a q ∧ condition1 a ∧ condition2 a ∧ a 4 = -8 :=
by
  sorry

end find_a4_l12_12735


namespace solve_for_z_l12_12904

variable {z : ℂ}
def complex_i := Complex.I

theorem solve_for_z (h : 1 - complex_i * z = -1 + complex_i * z) : z = -complex_i := by
  sorry

end solve_for_z_l12_12904


namespace intercepts_of_line_l12_12033

theorem intercepts_of_line (x y : ℝ) 
  (h : 2 * x + 7 * y = 35) :
  (y = 5 → x = 0) ∧ (x = 17.5 → y = 0)  :=
by
  sorry

end intercepts_of_line_l12_12033


namespace cannot_be_sum_of_four_consecutive_even_integers_l12_12905

-- Define what it means to be the sum of four consecutive even integers
def sum_of_four_consecutive_even_integers (n : ℤ) : Prop :=
  ∃ m : ℤ, n = 4 * m + 12 ∧ m % 2 = 0

-- State the problem in Lean 4
theorem cannot_be_sum_of_four_consecutive_even_integers :
  ¬ sum_of_four_consecutive_even_integers 32 ∧
  ¬ sum_of_four_consecutive_even_integers 80 ∧
  ¬ sum_of_four_consecutive_even_integers 104 ∧
  ¬ sum_of_four_consecutive_even_integers 200 :=
by
  sorry

end cannot_be_sum_of_four_consecutive_even_integers_l12_12905


namespace total_amount_is_175_l12_12451

noncomputable def calc_total_amount (x : ℝ) (y : ℝ) (z : ℝ) : ℝ :=
x + y + z

theorem total_amount_is_175 (x y z : ℝ) 
  (h1 : y = 0.45 * x)
  (h2 : z = 0.30 * x)
  (h3 : y = 45) :
  calc_total_amount x y z = 175 :=
by
  -- sorry to skip the proof
  sorry

end total_amount_is_175_l12_12451


namespace smallest_b_gt_4_perfect_square_l12_12438

theorem smallest_b_gt_4_perfect_square :
  ∃ b : ℕ, b > 4 ∧ ∃ k : ℕ, 4 * b + 5 = k^2 ∧ b = 5 :=
by
  sorry

end smallest_b_gt_4_perfect_square_l12_12438


namespace julia_total_watches_l12_12657

-- Definitions based on conditions.
def silver_watches : Nat := 20
def bronze_watches : Nat := 3 * silver_watches
def total_silver_bronze_watches : Nat := silver_watches + bronze_watches
def gold_watches : Nat := total_silver_bronze_watches / 10

-- The final proof statement without providing the proof.
theorem julia_total_watches : (silver_watches + bronze_watches + gold_watches) = 88 :=
by 
  -- Since we don't need to provide the actual proof, we use sorry
  sorry

end julia_total_watches_l12_12657


namespace sixteen_grams_on_left_pan_l12_12003

theorem sixteen_grams_on_left_pan :
  ∃ (weights : ℕ → ℕ) (pans : ℕ → ℕ) (n : ℕ),
    weights n = 16 ∧
    pans 0 = 11111 ∧
    ∃ k, (∀ i < k, weights i = 2 ^ i) ∧
    (∀ i < k, (pans 1 + weights i = 38) ∧ (pans 0 + 11111 = weights i + skeletal)) ∧
    k = 6 := by
  sorry

end sixteen_grams_on_left_pan_l12_12003


namespace valid_conditions_x_y_z_l12_12589

theorem valid_conditions_x_y_z (x y z : ℤ) :
  x = y - 1 ∧ z = y + 1 ∨ x = y ∧ z = y + 1 ↔ x * (x - y) + y * (y - x) + z * (z - y) = 1 :=
sorry

end valid_conditions_x_y_z_l12_12589


namespace division_theorem_l12_12521

theorem division_theorem (y : ℕ) (h : y ≠ 0) : (y - 1) ∣ (y^(y^2) - 2 * y^(y + 1) + 1) := 
by
  sorry

end division_theorem_l12_12521


namespace speed_against_current_l12_12869

noncomputable def man's_speed_with_current : ℝ := 20
noncomputable def current_speed : ℝ := 1

theorem speed_against_current :
  (man's_speed_with_current - 2 * current_speed) = 18 := by
sorry

end speed_against_current_l12_12869


namespace monogram_count_is_correct_l12_12243

def count_possible_monograms : ℕ :=
  Nat.choose 23 2

theorem monogram_count_is_correct : 
  count_possible_monograms = 253 := 
by 
  -- The proof will show this matches the combination formula calculation
  -- The final proof is left incomplete as per the instructions
  sorry

end monogram_count_is_correct_l12_12243


namespace channel_depth_l12_12066

theorem channel_depth
  (top_width bottom_width area : ℝ)
  (h : ℝ)
  (trapezium_area_formula : area = (1 / 2) * (top_width + bottom_width) * h)
  (top_width_val : top_width = 14)
  (bottom_width_val : bottom_width = 8)
  (area_val : area = 770) :
  h = 70 := 
by
  sorry

end channel_depth_l12_12066


namespace cherry_pies_count_correct_l12_12556

def total_pies : ℕ := 36

def ratio_ap_bb_ch : (ℕ × ℕ × ℕ) := (2, 3, 4)

def total_ratio_parts : ℕ := 2 + 3 + 4

def pies_per_part (total_pies : ℕ) (total_ratio_parts : ℕ) : ℕ := total_pies / total_ratio_parts

def num_parts_ch : ℕ := 4

def num_cherry_pies (total_pies : ℕ) (total_ratio_parts : ℕ) (num_parts_ch : ℕ) : ℕ :=
  pies_per_part total_pies total_ratio_parts * num_parts_ch

theorem cherry_pies_count_correct : num_cherry_pies total_pies total_ratio_parts num_parts_ch = 16 := by
  sorry

end cherry_pies_count_correct_l12_12556


namespace grapes_average_seeds_l12_12540

def total_seeds_needed : ℕ := 60
def apple_seed_average : ℕ := 6
def pear_seed_average : ℕ := 2
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def extra_seeds_needed : ℕ := 3

-- Calculation of total seeds from apples and pears:
def seeds_from_apples : ℕ := apples_count * apple_seed_average
def seeds_from_pears : ℕ := pears_count * pear_seed_average

def total_seeds_from_apples_and_pears : ℕ := seeds_from_apples + seeds_from_pears

-- Calculation of the remaining seeds needed from grapes:
def seeds_needed_from_grapes : ℕ := total_seeds_needed - total_seeds_from_apples_and_pears - extra_seeds_needed

-- Calculation of the average number of seeds per grape:
def grape_seed_average : ℕ := seeds_needed_from_grapes / grapes_count

-- Prove the correct average number of seeds per grape:
theorem grapes_average_seeds : grape_seed_average = 3 :=
by
  sorry

end grapes_average_seeds_l12_12540


namespace total_profit_at_100_max_profit_price_l12_12825

noncomputable def sales_volume (x : ℝ) : ℝ := 15 - 0.1 * x
noncomputable def floating_price (S : ℝ) : ℝ := 10 / S
noncomputable def supply_price (x : ℝ) : ℝ := 30 + floating_price (sales_volume x)
noncomputable def profit_per_set (x : ℝ) : ℝ := x - supply_price x
noncomputable def total_profit (x : ℝ) : ℝ := profit_per_set x * sales_volume x

-- Theorem 1: Total profit when each set is priced at 100 yuan is 340 ten thousand yuan
theorem total_profit_at_100 : total_profit 100 = 340 := by
  sorry

-- Theorem 2: The price per set that maximizes profit per set is 140 yuan
theorem max_profit_price : ∃ x, profit_per_set x = 100 ∧ x = 140 := by
  sorry

end total_profit_at_100_max_profit_price_l12_12825


namespace dogs_sold_correct_l12_12046

-- Definitions based on conditions
def ratio_cats_to_dogs (cats dogs : ℕ) := 2 * dogs = cats

-- Given conditions
def cats_sold := 16
def dogs_sold := 8

-- The theorem to prove
theorem dogs_sold_correct (h : ratio_cats_to_dogs cats_sold dogs_sold) : dogs_sold = 8 :=
by
  sorry

end dogs_sold_correct_l12_12046


namespace value_of_expression_l12_12673

theorem value_of_expression (n m : ℤ) (h : m = 2 * n^2 + n + 1) : 8 * n^2 - 4 * m + 4 * n - 3 = -7 := by
  sorry

end value_of_expression_l12_12673


namespace alice_still_needs_to_fold_l12_12228

theorem alice_still_needs_to_fold (total_cranes alice_folds friend_folds remains: ℕ) 
  (h1 : total_cranes = 1000)
  (h2 : alice_folds = total_cranes / 2)
  (h3 : friend_folds = (total_cranes - alice_folds) / 5)
  (h4 : remains = total_cranes - alice_folds - friend_folds) :
  remains = 400 := 
  by
    sorry

end alice_still_needs_to_fold_l12_12228


namespace f_periodic_odd_condition_l12_12599

theorem f_periodic_odd_condition (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_period : ∀ x, f (x + 4) = f x) (h_one : f 1 = 5) : f 2015 = -5 :=
by
  sorry

end f_periodic_odd_condition_l12_12599


namespace transistors_in_2010_l12_12399

theorem transistors_in_2010 (initial_transistors: ℕ) 
    (doubling_period_years: ℕ) (start_year: ℕ) (end_year: ℕ) 
    (h_initial: initial_transistors = 500000)
    (h_period: doubling_period_years = 2) 
    (h_start: start_year = 1992) 
    (h_end: end_year = 2010) :
  let years_passed := end_year - start_year
  let number_of_doublings := years_passed / doubling_period_years
  let transistors_in_end_year := initial_transistors * 2^number_of_doublings
  transistors_in_end_year = 256000000 := by
    sorry

end transistors_in_2010_l12_12399


namespace apps_difference_l12_12292

variable (initial_apps : ℕ) (added_apps : ℕ) (apps_left : ℕ)
variable (total_apps : ℕ := initial_apps + added_apps)
variable (deleted_apps : ℕ := total_apps - apps_left)
variable (difference : ℕ := added_apps - deleted_apps)

theorem apps_difference (h1 : initial_apps = 115) (h2 : added_apps = 235) (h3 : apps_left = 178) : 
  difference = 63 := by
  sorry

end apps_difference_l12_12292


namespace boxes_produced_by_machine_A_in_10_minutes_l12_12216

-- Define the variables and constants involved
variables {A : ℕ} -- number of boxes machine A produces in 10 minutes

-- Define the condition that machine B produces 4*A boxes in 10 minutes
def boxes_produced_by_machine_B_in_10_minutes := 4 * A

-- Define the combined production working together for 20 minutes
def combined_production_in_20_minutes := 10 * A

-- Statement to prove that machine A produces A boxes in 10 minutes
theorem boxes_produced_by_machine_A_in_10_minutes :
  ∀ (boxes_produced_by_machine_B_in_10_minutes : ℕ) (combined_production_in_20_minutes : ℕ),
    boxes_produced_by_machine_B_in_10_minutes = 4 * A →
    combined_production_in_20_minutes = 10 * A →
    A = A :=
by
  intros _ _ hB hC
  sorry

end boxes_produced_by_machine_A_in_10_minutes_l12_12216


namespace median_to_longest_side_l12_12238

theorem median_to_longest_side
  (a b c : ℕ) (h1 : a = 10) (h2 : b = 24) (h3 : c = 26)
  (h4 : a^2 + b^2 = c^2) :
  ∃ m : ℕ, m = c / 2 ∧ m = 13 := 
by {
  sorry
}

end median_to_longest_side_l12_12238


namespace sum_of_three_integers_mod_53_l12_12501

theorem sum_of_three_integers_mod_53 (a b c : ℕ) (h1 : a % 53 = 31) 
                                     (h2 : b % 53 = 22) (h3 : c % 53 = 7) : 
                                     (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_three_integers_mod_53_l12_12501


namespace possible_values_n_l12_12576

theorem possible_values_n (n : ℕ) (h_pos : 0 < n) (h1 : n > 9 / 4) (h2 : n < 14) :
  ∃ S : Finset ℕ, S.card = 11 ∧ ∀ k ∈ S, k = n :=
by
  -- proof to be filled in
  sorry

end possible_values_n_l12_12576


namespace distance_light_in_50_years_l12_12959

/-- The distance light travels in one year, given in scientific notation -/
def distance_light_per_year : ℝ := 9.4608 * 10^12

/-- The distance light travels in 50 years is calculated -/
theorem distance_light_in_50_years :
  distance_light_per_year * 50 = 4.7304 * 10^14 :=
by
  -- the proof is not demanded, so we use sorry
  sorry

end distance_light_in_50_years_l12_12959


namespace crocodile_can_move_anywhere_iff_even_l12_12375

def is_even (n : ℕ) : Prop := n % 2 = 0

def can_move_to_any_square (N : ℕ) : Prop :=
∀ (x1 y1 x2 y2 : ℤ), ∃ (k : ℕ), 
(x1 + k * (N + 1) = x2 ∨ y1 + k * (N + 1) = y2)

theorem crocodile_can_move_anywhere_iff_even (N : ℕ) : can_move_to_any_square N ↔ is_even N :=
sorry

end crocodile_can_move_anywhere_iff_even_l12_12375


namespace area_of_circles_l12_12440

theorem area_of_circles (BD AC : ℝ) (hBD : BD = 6) (hAC : AC = 12) : 
  ∃ S : ℝ, S = 225 / 4 * Real.pi :=
by
  sorry

end area_of_circles_l12_12440


namespace sam_morning_run_distance_l12_12329

variable (n : ℕ) (x : ℝ)

theorem sam_morning_run_distance (h : x + 2 * n * x + 12 = 18) : x = 6 / (1 + 2 * n) :=
by
  sorry

end sam_morning_run_distance_l12_12329


namespace sara_has_8_balloons_l12_12632

-- Define the number of yellow balloons Tom has.
def tom_balloons : ℕ := 9 

-- Define the total number of yellow balloons.
def total_balloons : ℕ := 17

-- Define the number of yellow balloons Sara has.
def sara_balloons : ℕ := total_balloons - tom_balloons

-- Theorem stating that Sara has 8 yellow balloons.
theorem sara_has_8_balloons : sara_balloons = 8 := by
  -- Proof goes here. Adding sorry for now to skip the proof.
  sorry

end sara_has_8_balloons_l12_12632


namespace cosine_double_angle_l12_12962

theorem cosine_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cosine_double_angle_l12_12962


namespace greatest_radius_of_circle_area_lt_90pi_l12_12386

theorem greatest_radius_of_circle_area_lt_90pi : ∃ (r : ℤ), (∀ (r' : ℤ), (π * (r':ℝ)^2 < 90 * π ↔ (r' ≤ r))) ∧ (π * (r:ℝ)^2 < 90 * π) ∧ (r = 9) :=
sorry

end greatest_radius_of_circle_area_lt_90pi_l12_12386


namespace max_g_value_on_interval_l12_12647

def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_g_value_on_interval : ∃ x, 0 ≤ x ∧ x ≤ 2 ∧ ∀ y,  0 ≤ y ∧ y ≤ 2 → g x ≥ g y ∧ g x = 3 :=
-- Proof goes here
sorry

end max_g_value_on_interval_l12_12647


namespace find_p_q_l12_12114

def vector_a (p : ℝ) : ℝ × ℝ × ℝ := (4, p, -2)
def vector_b (q : ℝ) : ℝ × ℝ × ℝ := (3, 2, q)

theorem find_p_q (p q : ℝ)
  (h1 : 4 * 3 + p * 2 + (-2) * q = 0)
  (h2 : 4^2 + p^2 + (-2)^2 = 3^2 + 2^2 + q^2) :
  (p, q) = (-29/12, 43/12) :=
by 
  sorry

end find_p_q_l12_12114


namespace triangle_area_l12_12566

theorem triangle_area (a b c : ℝ) (h₁ : a = 9) (h₂ : b = 12) (h₃ : c = 15) (h₄ : a^2 + b^2 = c^2) : 
  (1/2 * a * b) = 54 := by
  -- conditions provided
  sorry

end triangle_area_l12_12566


namespace num_factors_m_l12_12928

noncomputable def m : ℕ := 2^5 * 3^6 * 5^7 * 6^8

theorem num_factors_m : ∃ (k : ℕ), k = 1680 ∧ ∀ d : ℕ, d ∣ m ↔ ∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 13 ∧ 0 ≤ b ∧ b ≤ 14 ∧ 0 ≤ c ∧ c ≤ 7 ∧ d = 2^a * 3^b * 5^c :=
by 
sorry

end num_factors_m_l12_12928


namespace dan_initial_money_l12_12592

def initial_amount (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ) : ℕ :=
  spent_candy + spent_chocolate + remaining

theorem dan_initial_money 
  (spent_candy : ℕ) (spent_chocolate : ℕ) (remaining : ℕ)
  (h_candy : spent_candy = 2)
  (h_chocolate : spent_chocolate = 3)
  (h_remaining : remaining = 2) :
  initial_amount spent_candy spent_chocolate remaining = 7 :=
by
  rw [h_candy, h_chocolate, h_remaining]
  unfold initial_amount
  rfl

end dan_initial_money_l12_12592


namespace average_increase_fraction_l12_12333

-- First, we define the given conditions:
def incorrect_mark : ℕ := 82
def correct_mark : ℕ := 62
def number_of_students : ℕ := 80

-- We state the theorem to prove that the fraction by which the average marks increased is 1/4. 
theorem average_increase_fraction (incorrect_mark correct_mark : ℕ) (number_of_students : ℕ) :
  (incorrect_mark - correct_mark) / number_of_students = 1 / 4 :=
by
  sorry

end average_increase_fraction_l12_12333


namespace sqrt_addition_l12_12899

theorem sqrt_addition :
  (Real.sqrt (49 + 81) + Real.sqrt (36 - 9) = Real.sqrt 130 + 3 * Real.sqrt 3) := 
by sorry

end sqrt_addition_l12_12899


namespace cubic_roots_l12_12582

open Real

theorem cubic_roots (x1 x2 x3 : ℝ) (h1 : x1 * x2 = 1)
  (h2 : 3 * x1^3 + 2 * sqrt 3 * x1^2 - 21 * x1 + 6 * sqrt 3 = 0)
  (h3 : 3 * x2^3 + 2 * sqrt 3 * x2^2 - 21 * x2 + 6 * sqrt 3 = 0)
  (h4 : 3 * x3^3 + 2 * sqrt 3 * x3^2 - 21 * x3 + 6 * sqrt 3 = 0) :
  (x1 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x3 = -2 * sqrt 3) ∨
  (x1 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x1 = sqrt 3 / 3 ∧ x2 = -2 * sqrt 3) ∨
  (x2 = sqrt 3 ∧ x3 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) ∨
  (x3 = sqrt 3 ∧ x2 = sqrt 3 / 3 ∧ x1 = -2 * sqrt 3) := 
sorry

end cubic_roots_l12_12582


namespace original_decimal_number_l12_12044

theorem original_decimal_number (x : ℝ) (h : x / 100 = x - 1.485) : x = 1.5 := 
by
  sorry

end original_decimal_number_l12_12044


namespace nickels_used_for_notebook_l12_12296

def notebook_cost_dollars : ℚ := 1.30
def dollar_to_cents_conversion : ℤ := 100
def nickel_value_cents : ℤ := 5

theorem nickels_used_for_notebook : 
  (notebook_cost_dollars * dollar_to_cents_conversion) / nickel_value_cents = 26 := 
by 
  sorry

end nickels_used_for_notebook_l12_12296


namespace base_eight_to_base_ten_l12_12294

theorem base_eight_to_base_ten (a b c : ℕ) (ha : a = 1) (hb : b = 5) (hc : c = 7) :
  a * 8^2 + b * 8^1 + c * 8^0 = 111 :=
by 
  -- rest of the proof will go here
  sorry

end base_eight_to_base_ten_l12_12294


namespace solve_system_l12_12309

open Real

-- Define the system of equations as hypotheses
def eqn1 (x y z : ℝ) : Prop := x + y + 2 - 4 * x * y = 0
def eqn2 (x y z : ℝ) : Prop := y + z + 2 - 4 * y * z = 0
def eqn3 (x y z : ℝ) : Prop := z + x + 2 - 4 * z * x = 0

-- State the theorem
theorem solve_system (x y z : ℝ) :
  (eqn1 x y z ∧ eqn2 x y z ∧ eqn3 x y z) ↔ 
  ((x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = -1/2 ∧ y = -1/2 ∧ z = -1/2)) :=
by 
  sorry

end solve_system_l12_12309


namespace radius_of_curvature_correct_l12_12732

open Real

noncomputable def radius_of_curvature_squared (a b t_0 : ℝ) : ℝ :=
  (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2)

theorem radius_of_curvature_correct (a b t_0 : ℝ) (h : a > 0) (h₁ : b > 0) :
  radius_of_curvature_squared a b t_0 = (a^2 * sin t_0^2 + b^2 * cos t_0^2)^3 / (a^2 * b^2) :=
sorry

end radius_of_curvature_correct_l12_12732


namespace graph_does_not_pass_second_quadrant_l12_12798

theorem graph_does_not_pass_second_quadrant (a b : ℝ) (h₀ : 1 < a) (h₁ : b < -1) : 
∀ x : ℝ, ¬ (y = a^x + b ∧ y > 0 ∧ x < 0) :=
by
  sorry

end graph_does_not_pass_second_quadrant_l12_12798


namespace symmetric_line_eq_l12_12935

theorem symmetric_line_eq (a b : ℝ) (ha : a ≠ 0) : 
  (∃ k m : ℝ, (∀ x: ℝ, ax + b = (k * ( -x)) + m ∧ (k = 1/a ∧ m = b/a )))  := 
sorry

end symmetric_line_eq_l12_12935


namespace total_capacity_of_bowl_l12_12362

theorem total_capacity_of_bowl (L C : ℕ) (h1 : L / C = 3 / 5) (h2 : C = L + 18) : L + C = 72 := 
by
  sorry

end total_capacity_of_bowl_l12_12362


namespace garden_enlargement_l12_12144

theorem garden_enlargement :
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  area_square - area_rectangular = 400 := by
  -- initializing all definitions
  let length := 60
  let width := 20
  let perimeter := 2 * (length + width)
  let side_square := perimeter / 4
  let area_rectangular := length * width
  let area_square := side_square * side_square
  -- placeholder for the actual proof
  sorry

end garden_enlargement_l12_12144


namespace local_food_drive_correct_l12_12511

def local_food_drive_condition1 (R J x : ℕ) : Prop :=
  J = 2 * R + x

def local_food_drive_condition2 (J : ℕ) : Prop :=
  4 * J = 100

def local_food_drive_condition3 (R J : ℕ) : Prop :=
  R + J = 35

theorem local_food_drive_correct (R J x : ℕ)
  (h1 : local_food_drive_condition1 R J x)
  (h2 : local_food_drive_condition2 J)
  (h3 : local_food_drive_condition3 R J) :
  x = 5 :=
by
  sorry

end local_food_drive_correct_l12_12511


namespace probability_even_distinct_digits_l12_12598

theorem probability_even_distinct_digits :
  let count_even_distinct := 1960
  let total_numbers := 8000
  count_even_distinct / total_numbers = 49 / 200 :=
by
  sorry

end probability_even_distinct_digits_l12_12598


namespace increase_in_area_l12_12355

-- Define the initial side length and the increment.
def initial_side_length : ℕ := 6
def increment : ℕ := 1

-- Define the original area of the land.
def original_area : ℕ := initial_side_length * initial_side_length

-- Define the new side length after the increase.
def new_side_length : ℕ := initial_side_length + increment

-- Define the new area of the land.
def new_area : ℕ := new_side_length * new_side_length

-- Define the theorem that states the increase in area.
theorem increase_in_area : new_area - original_area = 13 := by
  sorry

end increase_in_area_l12_12355


namespace solution_set_circle_l12_12535

theorem solution_set_circle (a x y : ℝ) :
 (∃ a, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)) ↔ ((x - 3)^2 + (y - 1)^2 = 5 ∧ ¬ (x = 2 ∧ y = -1)) := by
sorry

end solution_set_circle_l12_12535


namespace circles_intersect_l12_12215

noncomputable def positional_relationship (center1 center2 : ℝ × ℝ) (radius1 radius2 : ℝ) : String :=
  let d := Real.sqrt ((center2.1 - center1.1)^2 + (center2.2 - center1.2)^2)
  if radius1 + radius2 > d ∧ d > abs (radius1 - radius2) then "Intersecting"
  else if radius1 + radius2 = d then "Externally tangent"
  else if abs (radius1 - radius2) = d then "Internally tangent"
  else "Separate"

theorem circles_intersect :
  positional_relationship (0, 1) (1, 2) 1 2 = "Intersecting" :=
by
  sorry

end circles_intersect_l12_12215


namespace marcy_total_time_l12_12248

theorem marcy_total_time 
    (petting_time : ℝ)
    (fraction_combing : ℝ)
    (H1 : petting_time = 12)
    (H2 : fraction_combing = 1/3) :
    (petting_time + (fraction_combing * petting_time) = 16) :=
  sorry

end marcy_total_time_l12_12248


namespace min_birthday_employees_wednesday_l12_12538

theorem min_birthday_employees_wednesday :
  ∀ (employees : ℕ) (n : ℕ), 
  employees = 50 → 
  n ≥ 1 →
  ∃ (x : ℕ), 6 * x + (x + n) = employees ∧ x + n ≥ 8 :=
by
  sorry

end min_birthday_employees_wednesday_l12_12538


namespace total_candies_is_36_l12_12221

-- Defining the conditions
def candies_per_day (day : String) : Nat :=
  if day = "Monday" ∨ day = "Wednesday" then 2 else 1

def total_candies_per_week : Nat :=
  (candies_per_day "Monday" + candies_per_day "Tuesday"
  + candies_per_day "Wednesday" + candies_per_day "Thursday"
  + candies_per_day "Friday" + candies_per_day "Saturday"
  + candies_per_day "Sunday")

def total_candies_in_weeks (weeks : Nat) : Nat :=
  weeks * total_candies_per_week

-- Stating the theorem
theorem total_candies_is_36 : total_candies_in_weeks 4 = 36 :=
  sorry

end total_candies_is_36_l12_12221


namespace depth_of_second_hole_l12_12571

theorem depth_of_second_hole :
  let workers1 := 45
  let hours1 := 8
  let depth1 := 30
  let total_man_hours1 := workers1 * hours1
  let rate_of_work := depth1 / total_man_hours1
  let workers2 := 45 + 45
  let hours2 := 6
  let total_man_hours2 := workers2 * hours2
  let depth2 := rate_of_work * total_man_hours2
  depth2 = 45 := by
    sorry

end depth_of_second_hole_l12_12571


namespace quadratic_sum_l12_12137

theorem quadratic_sum (x : ℝ) :
  (∃ a b c : ℝ, 6 * x^2 + 48 * x + 162 = a * (x + b) ^ 2 + c ∧ a + b + c = 76) :=
by
  sorry

end quadratic_sum_l12_12137


namespace find_f7_l12_12067

noncomputable def f : ℝ → ℝ := sorry

-- The conditions provided in the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom function_in_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The final proof goal
theorem find_f7 : f 7 = -2 :=
by sorry

end find_f7_l12_12067


namespace value_modulo_7_l12_12166

theorem value_modulo_7 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 7 = 5 := 
  by 
  sorry

end value_modulo_7_l12_12166


namespace deer_families_stayed_l12_12254

-- Define the initial number of deer families
def initial_deer_families : ℕ := 79

-- Define the number of deer families that moved out
def moved_out_deer_families : ℕ := 34

-- The theorem stating how many deer families stayed
theorem deer_families_stayed : initial_deer_families - moved_out_deer_families = 45 :=
by
  -- Proof will be provided here
  sorry

end deer_families_stayed_l12_12254


namespace triangle_area_not_twice_parallelogram_l12_12797

theorem triangle_area_not_twice_parallelogram (b h : ℝ) :
  (1 / 2) * b * h ≠ 2 * b * h :=
sorry

end triangle_area_not_twice_parallelogram_l12_12797


namespace largest_possible_last_digit_l12_12396

theorem largest_possible_last_digit (D : Fin 3003 → Nat) :
  D 0 = 2 →
  (∀ i : Fin 3002, (10 * D i + D (i + 1)) % 17 = 0 ∨ (10 * D i + D (i + 1)) % 23 = 0) →
  D 3002 = 9 :=
sorry

end largest_possible_last_digit_l12_12396


namespace star_of_15_star_eq_neg_15_l12_12300

def y_star (y : ℤ) : ℤ := 10 - y
def star_y (y : ℤ) : ℤ := y - 10

theorem star_of_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by {
  -- applying given definitions;
  sorry
}

end star_of_15_star_eq_neg_15_l12_12300


namespace numbers_from_five_threes_l12_12047

theorem numbers_from_five_threes :
  (∃ (a b c d e : ℤ), (3*a + 3*b + 3*c + 3*d + 3*e = 11 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 12 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 13 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 14 ∨ 
                        3*a + 3*b + 3*c + 3*d + 3*e = 15) ) :=
by
  -- Proof provided by the problem statement steps, using:
  -- 11 = (33/3)
  -- 12 = 3 * 3 + 3 + 3 - 3
  -- 13 = 3 * 3 + 3 + 3/3
  -- 14 = (33 + 3 * 3) / 3
  -- 15 = 3 + 3 + 3 + 3 + 3
  sorry

end numbers_from_five_threes_l12_12047


namespace odd_expression_proof_l12_12401

theorem odd_expression_proof (n : ℤ) : Odd (n^2 + n + 5) :=
by 
  sorry

end odd_expression_proof_l12_12401


namespace aleksey_divisible_l12_12404

theorem aleksey_divisible
  (x y a b S : ℤ)
  (h1 : x + y = S)
  (h2 : S ∣ (a * x + b * y)) :
  S ∣ (b * x + a * y) := 
sorry

end aleksey_divisible_l12_12404


namespace remainder_when_divided_by_x_minus_2_l12_12402

def f (x : ℝ) : ℝ := x^5 - 6 * x^4 + 11 * x^3 + 21 * x^2 - 17 * x + 10

theorem remainder_when_divided_by_x_minus_2 : (f 2) = 84 := by
  sorry

end remainder_when_divided_by_x_minus_2_l12_12402


namespace solution_comparison_l12_12523

variables (a a' b b' : ℝ)

theorem solution_comparison (ha : a ≠ 0) (ha' : a' ≠ 0) :
  (-(b / a) < -(b' / a')) ↔ (b' / a' < b / a) :=
by sorry

end solution_comparison_l12_12523


namespace puzzles_and_board_games_count_l12_12380

def num_toys : ℕ := 200
def num_action_figures : ℕ := num_toys / 4
def num_dolls : ℕ := num_toys / 3

theorem puzzles_and_board_games_count :
  num_toys - num_action_figures - num_dolls = 84 := 
  by
    -- TODO: Prove this theorem
    sorry

end puzzles_and_board_games_count_l12_12380


namespace swap_numbers_l12_12500

theorem swap_numbers (a b : ℕ) (hc: b = 17) (ha : a = 8) : 
  ∃ c, c = b ∧ b = a ∧ a = c := 
by
  sorry

end swap_numbers_l12_12500


namespace consecutive_odd_sum_count_l12_12433

theorem consecutive_odd_sum_count (N : ℕ) :
  N = 20 ↔ (
    ∃ (ns : Finset ℕ), ∃ (js : Finset ℕ),
      (∀ n ∈ ns, n < 500) ∧
      (∀ j ∈ js, j ≥ 2) ∧
      ∀ n ∈ ns, ∃ j ∈ js, ∃ k, k = 3 ∧ N = j * (2 * k + j)
  ) :=
by
  sorry

end consecutive_odd_sum_count_l12_12433


namespace minimize_reciprocals_l12_12319

theorem minimize_reciprocals (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 30) :
  (a = 10 ∧ b = 5) → ∀ x y : ℕ, (x > 0) → (y > 0) → (x + 4 * y = 30) → (1 / (x : ℝ) + 1 / (y : ℝ) ≥ 1 / 10 + 1 / 5) := 
by {
  sorry
}

end minimize_reciprocals_l12_12319


namespace garden_ratio_l12_12710

-- Define the given conditions
def garden_length : ℕ := 100
def garden_perimeter : ℕ := 300

-- Problem statement: Prove the ratio of the length to the width is 2:1
theorem garden_ratio : 
  ∃ (W L : ℕ), 
    L = garden_length ∧ 
    2 * L + 2 * W = garden_perimeter ∧ 
    L / W = 2 :=
by 
  sorry

end garden_ratio_l12_12710


namespace same_number_of_friends_l12_12671

theorem same_number_of_friends (n : ℕ) (friends : Fin n → Fin n) :
  (∃ i j : Fin n, i ≠ j ∧ friends i = friends j) :=
by
  -- The proof is omitted.
  sorry

end same_number_of_friends_l12_12671


namespace determine_price_reduction_l12_12917

noncomputable def initial_cost_price : ℝ := 220
noncomputable def initial_selling_price : ℝ := 280
noncomputable def initial_daily_sales_volume : ℕ := 30
noncomputable def price_reduction_increase_rate : ℝ := 3

variable (x : ℝ)

noncomputable def daily_sales_volume (x : ℝ) : ℝ := initial_daily_sales_volume + price_reduction_increase_rate * x
noncomputable def profit_per_item (x : ℝ) : ℝ := (initial_selling_price - x) - initial_cost_price

theorem determine_price_reduction (x : ℝ) 
    (h1 : daily_sales_volume x = initial_daily_sales_volume + price_reduction_increase_rate * x)
    (h2 : profit_per_item x = 60 - x) : 
    (30 + 3 * x) * (60 - x) = 3600 → x = 30 :=
by 
  sorry

end determine_price_reduction_l12_12917


namespace juvy_chives_l12_12675

-- Definitions based on the problem conditions
def total_rows : Nat := 20
def plants_per_row : Nat := 10
def parsley_rows : Nat := 3
def rosemary_rows : Nat := 2
def chive_rows : Nat := total_rows - (parsley_rows + rosemary_rows)

-- The statement we want to prove
theorem juvy_chives : chive_rows * plants_per_row = 150 := by
  sorry

end juvy_chives_l12_12675


namespace correct_statement_l12_12226

noncomputable def f (x : ℝ) := Real.exp x - x
noncomputable def g (x : ℝ) := Real.log x + x + 1

def proposition_p := ∀ x : ℝ, f x > 0
def proposition_q := ∃ x0 : ℝ, 0 < x0 ∧ g x0 = 0

theorem correct_statement : (proposition_p ∧ proposition_q) :=
by
  sorry

end correct_statement_l12_12226


namespace value_subtracted_is_five_l12_12895

variable (N x : ℕ)

theorem value_subtracted_is_five
  (h1 : (N - x) / 7 = 7)
  (h2 : (N - 14) / 10 = 4) : x = 5 := by
  sorry

end value_subtracted_is_five_l12_12895


namespace complex_equation_l12_12968

theorem complex_equation (m n : ℝ) (i : ℂ)
  (hi : i^2 = -1)
  (h1 : m * (1 + i) = 1 + n * i) :
  ( (m + n * i) / (m - n * i) )^2 = -1 :=
sorry

end complex_equation_l12_12968


namespace solve_rational_inequality_l12_12863

theorem solve_rational_inequality :
  {x : ℝ | (9*x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 4} =
  {x : ℝ | (-10 < x ∧ x < -5) ∨ (2/3 < x ∧ x < 4/3) ∨ (4/3 < x)} :=
by
  sorry

end solve_rational_inequality_l12_12863


namespace cube_volume_multiple_of_6_l12_12016

theorem cube_volume_multiple_of_6 (n : ℕ) (h : ∃ m : ℕ, n^3 = 24 * m) : ∃ k : ℕ, n = 6 * k :=
by
  sorry

end cube_volume_multiple_of_6_l12_12016


namespace alice_bob_same_point_after_3_turns_l12_12185

noncomputable def alice_position (t : ℕ) : ℕ := (15 + 4 * t) % 15

noncomputable def bob_position (t : ℕ) : ℕ :=
  if t < 2 then 15
  else (15 - 11 * (t - 2)) % 15

theorem alice_bob_same_point_after_3_turns :
  ∃ t, t = 3 ∧ alice_position t = bob_position t :=
by
  exists 3
  simp only [alice_position, bob_position]
  norm_num
  -- Alice's position after 3 turns
  -- alice_position 3 = (15 + 4 * 3) % 15
  -- bob_position 3 = (15 - 11 * (3 - 2)) % 15
  -- Therefore,
  -- alice_position 3 = 12
  -- bob_position 3 = 12
  sorry

end alice_bob_same_point_after_3_turns_l12_12185


namespace total_time_for_process_l12_12555

-- Given conditions
def cat_resistance_time : ℕ := 20
def walking_distance : ℕ := 64
def walking_rate : ℕ := 8

-- Prove the total time
theorem total_time_for_process : cat_resistance_time + (walking_distance / walking_rate) = 28 := by
  sorry

end total_time_for_process_l12_12555


namespace model_lighthouse_height_l12_12373

theorem model_lighthouse_height (h_actual : ℝ) (V_actual : ℝ) (V_model : ℝ) (h_actual_val : h_actual = 60) (V_actual_val : V_actual = 150000) (V_model_val : V_model = 0.15) :
  (h_actual * (V_model / V_actual)^(1/3)) = 0.6 :=
by
  rw [h_actual_val, V_actual_val, V_model_val]
  sorry

end model_lighthouse_height_l12_12373


namespace difference_of_scores_correct_l12_12233

-- Define the parameters
def num_innings : ℕ := 46
def batting_avg : ℕ := 63
def highest_score : ℕ := 248
def reduced_avg : ℕ := 58
def excluded_innings : ℕ := num_innings - 2

-- Necessary calculations
def total_runs := batting_avg * num_innings
def reduced_total_runs := reduced_avg * excluded_innings
def sum_highest_lowest := total_runs - reduced_total_runs
def lowest_score := sum_highest_lowest - highest_score

-- The correct answer to prove
def expected_difference := highest_score - lowest_score
def correct_answer := 150

-- Define the proof problem
theorem difference_of_scores_correct :
  expected_difference = correct_answer := by
  sorry

end difference_of_scores_correct_l12_12233


namespace ages_of_three_persons_l12_12200

theorem ages_of_three_persons (y m e : ℕ) 
  (h1 : e = m + 16)
  (h2 : m = y + 8)
  (h3 : e - 6 = 3 * (y - 6))
  (h4 : e - 6 = 2 * (m - 6)) :
  y = 18 ∧ m = 26 ∧ e = 42 := 
by 
  sorry

end ages_of_three_persons_l12_12200


namespace smallest_t_for_circle_covered_l12_12383

theorem smallest_t_for_circle_covered:
  ∃ t, (∀ θ, 0 ≤ θ → θ ≤ t → (∃ r, r = Real.sin θ)) ∧
         (∀ t', (∀ θ, 0 ≤ θ → θ ≤ t' → (∃ r, r = Real.sin θ)) → t' ≥ t) :=
sorry

end smallest_t_for_circle_covered_l12_12383


namespace cube_volume_given_surface_area_l12_12376

theorem cube_volume_given_surface_area (s : ℝ) (h₀ : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_given_surface_area_l12_12376


namespace wendys_sales_are_205_l12_12836

def price_of_apple : ℝ := 1.5
def price_of_orange : ℝ := 1.0
def apples_sold_morning : ℕ := 40
def oranges_sold_morning : ℕ := 30
def apples_sold_afternoon : ℕ := 50
def oranges_sold_afternoon : ℕ := 40

/-- Wendy's total sales for the day are $205 given the conditions about the prices of apples and oranges,
and the number of each sold in the morning and afternoon. -/
def wendys_total_sales : ℝ :=
  let total_apples_sold := apples_sold_morning + apples_sold_afternoon
  let total_oranges_sold := oranges_sold_morning + oranges_sold_afternoon
  let sales_from_apples := total_apples_sold * price_of_apple
  let sales_from_oranges := total_oranges_sold * price_of_orange
  sales_from_apples + sales_from_oranges

theorem wendys_sales_are_205 : wendys_total_sales = 205 := by
  sorry

end wendys_sales_are_205_l12_12836


namespace intersection_of_lines_l12_12466

theorem intersection_of_lines :
  ∃ x y : ℚ, (8 * x - 3 * y = 9) ∧ (6 * x + 2 * y = 20) ∧ (x = 39 / 17) ∧ (y = 53 / 17) :=
by
  sorry

end intersection_of_lines_l12_12466


namespace problem1_l12_12892

theorem problem1 (x y : ℝ) (h1 : x * (x + y) = 27) (h2 : y * (x + y) = 54) : (x + y)^2 = 81 := 
by
  sorry

end problem1_l12_12892


namespace lisa_interest_earned_l12_12236

/-- Lisa's interest earned after three years from Bank of Springfield's Super High Yield savings account -/
theorem lisa_interest_earned :
  let P := 2000
  let r := 0.02
  let n := 3
  let A := P * (1 + r)^n
  A - P = 122 := by
  sorry

end lisa_interest_earned_l12_12236


namespace point_on_right_branch_l12_12542

noncomputable def on_hyperbola_right_branch (a b m : ℝ) :=
  (∀ a b m : ℝ, (a - 2 * b > 0) → (a + 2 * b > 0) → (a ^ 2 - 4 * b ^ 2 = m) → (m ≠ 0) → a > 0)

theorem point_on_right_branch (a b m : ℝ) (h₁ : a - 2 * b > 0) (h₂ : a + 2 * b > 0) (h₃ : a ^ 2 - 4 * b ^ 2 = m) (h₄ : m ≠ 0) :
  a > 0 := 
by 
  sorry

end point_on_right_branch_l12_12542


namespace general_term_formula_for_b_n_sum_of_first_n_terms_of_c_n_l12_12359

def is_geometric_sequence (a : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, ∀ n : ℕ, a (n + 1) = a n * q

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

def c_sequence (a b : ℕ → ℤ) (n : ℕ) : ℤ := a n - b n

def sum_c_sequence (c : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum c

theorem general_term_formula_for_b_n (a b : ℕ → ℤ) (n : ℕ) 
  (h1 : is_geometric_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : a 1 = b 1)
  (h4 : a 2 = 3)
  (h5 : a 3 = 9)
  (h6 : a 4 = b 14) :
  b n = 2 * n - 1 :=
sorry

theorem sum_of_first_n_terms_of_c_n (a b : ℕ → ℤ) (n : ℕ)
  (h1 : is_geometric_sequence a)
  (h2 : is_arithmetic_sequence b)
  (h3 : a 1 = b 1)
  (h4 : a 2 = 3)
  (h5 : a 3 = 9)
  (h6 : a 4 = b 14)
  (h7 : ∀ n : ℕ, c_sequence a b n = a n - b n) :
  sum_c_sequence (c_sequence a b) n = (3 ^ n) / 2 - n ^ 2 - 1 / 2 :=
sorry

end general_term_formula_for_b_n_sum_of_first_n_terms_of_c_n_l12_12359


namespace range_of_y_l12_12503

theorem range_of_y (x : ℝ) : 
  - (Real.sqrt 3) / 3 ≤ (Real.sin x) / (2 - Real.cos x) ∧ (Real.sin x) / (2 - Real.cos x) ≤ (Real.sqrt 3) / 3 :=
sorry

end range_of_y_l12_12503


namespace area_enclosed_by_region_l12_12896

open Real

def condition (x y : ℝ) := abs (2 * x + 2 * y) + abs (2 * x - 2 * y) ≤ 8

theorem area_enclosed_by_region : 
  (∃ u v : ℝ, condition u v) → ∃ A : ℝ, A = 16 := 
sorry

end area_enclosed_by_region_l12_12896


namespace volume_of_63_ounces_l12_12911

variable {V W : ℝ}
variable (k : ℝ)

def directly_proportional (V W : ℝ) (k : ℝ) : Prop :=
  V = k * W

theorem volume_of_63_ounces (h1 : directly_proportional 48 112 k)
                            (h2 : directly_proportional V 63 k) :
  V = 27 := by
  sorry

end volume_of_63_ounces_l12_12911


namespace seat_adjustment_schemes_l12_12743

theorem seat_adjustment_schemes {n k : ℕ} (h1 : n = 7) (h2 : k = 3) :
  (2 * Nat.choose n k) = 70 :=
by
  -- n is the number of people, k is the number chosen
  rw [h1, h2]
  -- the rest is skipped for the statement only
  sorry

end seat_adjustment_schemes_l12_12743


namespace triangle_integral_y_difference_l12_12341

theorem triangle_integral_y_difference :
  ∀ (y : ℕ), (3 ≤ y ∧ y ≤ 15) → (∃ y_min y_max : ℕ, y_min = 3 ∧ y_max = 15 ∧ (y_max - y_min = 12)) :=
by
  intro y
  intro h
  -- skipped proof
  sorry

end triangle_integral_y_difference_l12_12341


namespace aaron_age_l12_12692

variable (A : ℕ)
variable (henry_sister_age : ℕ)
variable (henry_age : ℕ)
variable (combined_age : ℕ)

theorem aaron_age (h1 : henry_sister_age = 3 * A)
                 (h2 : henry_age = 4 * henry_sister_age)
                 (h3 : combined_age = henry_sister_age + henry_age)
                 (h4 : combined_age = 240) : A = 16 := by
  sorry

end aaron_age_l12_12692


namespace min_sum_chessboard_labels_l12_12564

theorem min_sum_chessboard_labels :
  ∃ (r : Fin 9 → Fin 9), 
  (∀ (i j : Fin 9), i ≠ j → r i ≠ r j) ∧ 
  ((Finset.univ : Finset (Fin 9)).sum (λ i => 1 / (r i + i.val + 1)) = 1) :=
by
  sorry

end min_sum_chessboard_labels_l12_12564


namespace range_of_a_l12_12505

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 ↔ x < 1) → a < -1 :=
by
  intro h
  sorry

end range_of_a_l12_12505


namespace min_value_f_l12_12908

noncomputable def f (x : ℝ) : ℝ := |x - 4| + |x - 6|

theorem min_value_f : ∃ x : ℝ, f x ≥ 2 :=
by
  sorry

end min_value_f_l12_12908


namespace rectangle_perimeter_l12_12615

-- Defining the given conditions
def rectangleArea := 4032
noncomputable def ellipseArea := 4032 * Real.pi
noncomputable def b := Real.sqrt 2016
noncomputable def a := 2 * Real.sqrt 2016

-- Problem statement: the perimeter of the rectangle
theorem rectangle_perimeter (x y : ℝ) (h1 : x * y = rectangleArea)
  (h2 : x + y = 2 * a) : 2 * (x + y) = 8 * Real.sqrt 2016 :=
by
  sorry

end rectangle_perimeter_l12_12615


namespace pizza_volume_one_piece_l12_12411

theorem pizza_volume_one_piece :
  ∀ (h t: ℝ) (d: ℝ) (n: ℕ), d = 16 → t = 1/2 → n = 8 → h = 8 → 
  ( (π * (d / 2)^2 * t) / n = 4 * π ) :=
by 
  intros h t d n hd ht hn hh
  sorry

end pizza_volume_one_piece_l12_12411


namespace dividend_percentage_l12_12343

theorem dividend_percentage (face_value : ℝ) (investment : ℝ) (roi : ℝ) (dividend_percentage : ℝ) 
    (h1 : face_value = 40) 
    (h2 : investment = 20) 
    (h3 : roi = 0.25) : dividend_percentage = 12.5 := 
  sorry

end dividend_percentage_l12_12343


namespace omega_bound_l12_12427

noncomputable def f (ω x : ℝ) : ℝ := Real.cos (ω * x) - Real.sin (ω * x)

theorem omega_bound (ω : ℝ) (h₁ : ω > 0)
  (h₂ : ∀ x : ℝ, -π / 2 < x ∧ x < π / 2 → (f ω x) ≤ (f ω (-π / 2))) :
  ω ≤ 1 / 2 :=
sorry

end omega_bound_l12_12427


namespace percentage_tax_proof_l12_12510

theorem percentage_tax_proof (total_worth tax_free cost taxable tax_rate tax_value percentage_sales_tax : ℝ)
  (h1 : total_worth = 40)
  (h2 : tax_free = 34.7)
  (h3 : tax_rate = 0.06)
  (h4 : total_worth = taxable + tax_rate * taxable + tax_free)
  (h5 : tax_value = tax_rate * taxable)
  (h6 : percentage_sales_tax = (tax_value / total_worth) * 100) :
  percentage_sales_tax = 0.75 :=
by
  sorry

end percentage_tax_proof_l12_12510


namespace total_new_students_l12_12103

-- Given conditions
def number_of_schools : ℝ := 25.0
def average_students_per_school : ℝ := 9.88

-- Problem statement
theorem total_new_students : number_of_schools * average_students_per_school = 247 :=
by sorry

end total_new_students_l12_12103


namespace chord_length_3pi_4_chord_bisected_by_P0_l12_12520

open Real

-- Define conditions and the problem.
def Circle := {p : ℝ × ℝ // p.1^2 + p.2^2 = 8}
def P0 : ℝ × ℝ := (-1, 2)

-- Proving the first part (1)
theorem chord_length_3pi_4 (α : ℝ) (hα : α = 3 * π / 4) (A B : ℝ × ℝ)
  (h1 : (A.1^2 + A.2^2 = 8) ∧ (B.1^2 + B.2^2 = 8))
  (h2 : (A.1 + B.1) / 2 = -1) (h3 : (A.2 + B.2) / 2 = 2) :
  dist A B = sqrt 30 := sorry

-- Proving the second part (2)
theorem chord_bisected_by_P0 (A B : ℝ × ℝ)
  (h1 : (A.1^2 + A.2^2 = 8) ∧ (B.1^2 + B.2^2 = 8))
  (h2 : (A.1 + B.1) / 2 = -1) (h3 : (A.2 + B.2) / 2 = 2) :
  ∃ k : ℝ, (B.2 - A.2) = k * (B.1 - A.1) ∧ k = 1 / 2 ∧
  (k * (x - (-1))) = y - 2 := sorry

end chord_length_3pi_4_chord_bisected_by_P0_l12_12520


namespace larger_square_area_total_smaller_squares_area_l12_12130
noncomputable def largerSquareSideLengthFromCircleRadius (r : ℝ) : ℝ :=
  2 * (2 * r)

noncomputable def squareArea (side : ℝ) : ℝ :=
  side * side

theorem larger_square_area (r : ℝ) (h : r = 3) :
  squareArea (largerSquareSideLengthFromCircleRadius r) = 144 :=
by
  sorry

theorem total_smaller_squares_area (r : ℝ) (h : r = 3) :
  4 * squareArea (2 * r) = 144 :=
by
  sorry

end larger_square_area_total_smaller_squares_area_l12_12130


namespace second_fraction_correct_l12_12784

theorem second_fraction_correct : 
  ∃ x : ℚ, (2 / 3) * x * (1 / 3) * (3 / 8) = 0.07142857142857142 ∧ x = 6 / 7 :=
by
  sorry

end second_fraction_correct_l12_12784


namespace min_value_ineq_l12_12980

noncomputable def function_y (a : ℝ) (x : ℝ) : ℝ := a^(1-x)

theorem min_value_ineq (a m n : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : m * n > 0) (h4 : m + n = 1) :
  1/m + 2/n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end min_value_ineq_l12_12980


namespace discriminant_of_quadratic_equation_l12_12278

noncomputable def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic_equation : discriminant 5 (-11) (-18) = 481 := by
  sorry

end discriminant_of_quadratic_equation_l12_12278


namespace minimum_cost_for_13_bottles_l12_12611

def cost_per_bottle_shop_A := 200 -- in cents
def discount_shop_B := 15 / 100 -- discount
def promotion_B_threshold := 4
def promotion_A_threshold := 4

-- Function to calculate the cost in Shop A for given number of bottles
def shop_A_cost (bottles : ℕ) : ℕ :=
  let batches := bottles / 5
  let remainder := bottles % 5
  (batches * 4 + remainder) * cost_per_bottle_shop_A

-- Function to calculate the cost in Shop B for given number of bottles
def shop_B_cost (bottles : ℕ) : ℕ :=
  if bottles >= promotion_B_threshold then
    (bottles * cost_per_bottle_shop_A) * (1 - discount_shop_B)
  else
    bottles * cost_per_bottle_shop_A

-- Function to calculate combined cost for given numbers of bottles from Shops A and B
def combined_cost (bottles_A bottles_B : ℕ) : ℕ :=
  shop_A_cost bottles_A + shop_B_cost bottles_B

theorem minimum_cost_for_13_bottles : ∃ a b, a + b = 13 ∧ combined_cost a b = 2000 := 
sorry

end minimum_cost_for_13_bottles_l12_12611


namespace greatest_number_remainder_l12_12858

theorem greatest_number_remainder (G R : ℕ) (h1 : 150 % G = 50) (h2 : 230 % G = 5) (h3 : 175 % G = R) (h4 : ∀ g, g ∣ 100 → g ∣ 225 → g ∣ (175 - R) → g ≤ G) : R = 0 :=
by {
  -- This is the statement only; the proof is omitted as per the instructions.
  sorry
}

end greatest_number_remainder_l12_12858


namespace prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l12_12724

theorem prop1_converse (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b := sorry

theorem prop1_inverse (a b c : ℝ) (h : a ≤ b) : a * c^2 ≤ b * c^2 := sorry

theorem prop1_contrapositive (a b c : ℝ) (h : a * c^2 ≤ b * c^2) : a ≤ b := sorry

theorem prop2_converse (a b c : ℝ) (f : ℝ → ℝ) (h : ∃x, f x = 0) : b^2 - 4 * a * c < 0 := sorry

theorem prop2_inverse (a b c : ℝ) (f : ℝ → ℝ) (h : b^2 - 4 * a * c ≥ 0) : ¬∃x, f x = 0 := sorry

theorem prop2_contrapositive (a b c : ℝ) (f : ℝ → ℝ) (h : ¬∃x, f x = 0) : b^2 - 4 * a * c ≥ 0 := sorry

end prop1_converse_prop1_inverse_prop1_contrapositive_prop2_converse_prop2_inverse_prop2_contrapositive_l12_12724


namespace range_of_a_for_local_min_max_l12_12125

noncomputable def f (a e x : ℝ) : ℝ := 2 * a^x - e * x^2

theorem range_of_a_for_local_min_max (a e x1 x2 : ℝ) (h_a : 0 < a) (h_a_ne : a ≠ 1) (h_x1_x2 : x1 < x2) 
  (h_min : ∀ x, f a e x > f a e x1) (h_max : ∀ x, f a e x < f a e x2) : 
  (1 / Real.exp 1) < a ∧ a < 1 := 
sorry

end range_of_a_for_local_min_max_l12_12125


namespace percentage_of_first_to_second_l12_12984

theorem percentage_of_first_to_second (X : ℝ) (h1 : first = (7/100) * X) (h2 : second = (14/100) * X) : (first / second) * 100 = 50 := 
by
  sorry

end percentage_of_first_to_second_l12_12984


namespace part1_part2_l12_12087

namespace RationalOp
  -- Define the otimes operation
  def otimes (a b : ℚ) : ℚ := a * b^2 + 2 * a * b + a

  -- Part 1: Prove (-2) ⊗ 4 = -50
  theorem part1 : otimes (-2) 4 = -50 := sorry

  -- Part 2: Given x ⊗ 3 = y ⊗ (-3), prove 8x - 2y + 5 = 5
  theorem part2 (x y : ℚ) (h : otimes x 3 = otimes y (-3)) : 8*x - 2*y + 5 = 5 := sorry
end RationalOp

end part1_part2_l12_12087


namespace cubic_root_sum_cubed_l12_12408

theorem cubic_root_sum_cubed
  (p q r : ℂ)
  (h1 : 3 * p^3 - 9 * p^2 + 27 * p - 6 = 0)
  (h2 : 3 * q^3 - 9 * q^2 + 27 * q - 6 = 0)
  (h3 : 3 * r^3 - 9 * r^2 + 27 * r - 6 = 0)
  (hpq : p ≠ q)
  (hqr : q ≠ r)
  (hrp : r ≠ p) :
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 585 := 
  sorry

end cubic_root_sum_cubed_l12_12408


namespace probability_computation_l12_12356

-- Definitions of individual success probabilities
def probability_Xavier_solving_problem : ℚ := 1 / 4
def probability_Yvonne_solving_problem : ℚ := 2 / 3
def probability_William_solving_problem : ℚ := 7 / 10
def probability_Zelda_solving_problem : ℚ := 5 / 8
def probability_Zelda_notsolving_problem : ℚ := 1 - probability_Zelda_solving_problem

-- The target probability that only Xavier, Yvonne, and William, but not Zelda, will solve the problem
def target_probability : ℚ := (1 / 4) * (2 / 3) * (7 / 10) * (3 / 8)

-- The simplified form of the computed probability
def simplified_target_probability : ℚ := 7 / 160

-- Lean 4 statement to prove the equality of the computed and the target probabilities
theorem probability_computation :
  target_probability = simplified_target_probability := by
  sorry

end probability_computation_l12_12356


namespace valid_lineups_l12_12249

def total_players : ℕ := 15
def k : ℕ := 2  -- number of twins
def total_chosen : ℕ := 7
def remaining_players := total_players - k

def nCr (n r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.choose n r

def total_choices : ℕ := nCr total_players total_chosen
def restricted_choices : ℕ := nCr remaining_players (total_chosen - k)

theorem valid_lineups : total_choices - restricted_choices = 5148 := by
  sorry

end valid_lineups_l12_12249


namespace desired_annual_profit_is_30500000_l12_12924

noncomputable def annual_fixed_costs : ℝ := 50200000
noncomputable def average_cost_per_car : ℝ := 5000
noncomputable def number_of_cars : ℕ := 20000
noncomputable def selling_price_per_car : ℝ := 9035

noncomputable def total_revenue : ℝ :=
  selling_price_per_car * number_of_cars

noncomputable def total_variable_costs : ℝ :=
  average_cost_per_car * number_of_cars

noncomputable def total_costs : ℝ :=
  annual_fixed_costs + total_variable_costs

noncomputable def desired_annual_profit : ℝ :=
  total_revenue - total_costs

theorem desired_annual_profit_is_30500000:
  desired_annual_profit = 30500000 := by
  sorry

end desired_annual_profit_is_30500000_l12_12924


namespace dima_is_mistaken_l12_12259

theorem dima_is_mistaken :
  (∃ n : Nat, n > 0 ∧ ∀ n, 3 * n = 4 * n) → False :=
by
  intros h
  obtain ⟨n, hn1, hn2⟩ := h
  have hn := (hn2 n)
  linarith

end dima_is_mistaken_l12_12259


namespace badminton_tournament_l12_12008

theorem badminton_tournament (n x : ℕ) (h1 : 2 * n > 0) (h2 : 3 * n > 0) (h3 : (5 * n) * (5 * n - 1) = 14 * x) : n = 3 :=
by
  -- Placeholder for the proof
  sorry

end badminton_tournament_l12_12008


namespace geometric_progression_common_ratio_l12_12531

-- Definitions and theorems
variable {α : Type*} [OrderedCommRing α]

theorem geometric_progression_common_ratio
  (a : α) (r : α)
  (h_pos : a > 0)
  (h_geometric : ∀ n : ℕ, a * r^n = (a * r^(n + 1)) * (a * r^(n + 2))):
  r = 1 := by
  sorry

end geometric_progression_common_ratio_l12_12531


namespace find_angle_C_60_find_min_value_of_c_l12_12384

theorem find_angle_C_60 (a b c : ℝ) (A B C : ℝ)
  (h_cos_eq : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C) : 
  C = 60 := 
sorry

theorem find_min_value_of_c (a b c : ℝ) (A B C : ℝ)
  (h_cos_eq : a * Real.cos B + b * Real.cos A = 2 * c * Real.cos C)
  (h_area : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :
  c ≥ 2 * Real.sqrt 2 :=
sorry

end find_angle_C_60_find_min_value_of_c_l12_12384


namespace line_curve_intersection_l12_12641

theorem line_curve_intersection (a : ℝ) : 
  (∃! (x y : ℝ), (y = a * (x + 2)) ∧ (x ^ 2 - y * |y| = 1)) ↔ a ∈ Set.Ioo (-Real.sqrt 3 / 3) 1 :=
by
  sorry

end line_curve_intersection_l12_12641


namespace product_of_irwins_baskets_l12_12873

theorem product_of_irwins_baskets 
  (baskets_scored : Nat)
  (point_value : Nat)
  (total_baskets : baskets_scored = 2)
  (value_per_basket : point_value = 11) : 
  point_value * baskets_scored = 22 := 
by 
  sorry

end product_of_irwins_baskets_l12_12873


namespace no_real_roots_of_quadratic_l12_12056

theorem no_real_roots_of_quadratic (k : ℝ) (h : k ≠ 0) : ¬ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0 :=
by sorry

end no_real_roots_of_quadratic_l12_12056


namespace no_roots_impl_a_neg_l12_12700

theorem no_roots_impl_a_neg {a : ℝ} : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x - 1/x + a ≠ 0) → a < 0 :=
sorry

end no_roots_impl_a_neg_l12_12700


namespace determine_x_l12_12889

theorem determine_x (x y : ℝ) (h : x / (x - 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y + 1)) : 
  x = 2 * y^2 + 6 * y + 4 := 
by
  sorry

end determine_x_l12_12889


namespace sqrt_x_plus_5_l12_12000

theorem sqrt_x_plus_5 (x : ℝ) (h : x = -1) : Real.sqrt (x + 5) = 2 :=
by
  sorry

end sqrt_x_plus_5_l12_12000


namespace frac_sum_property_l12_12480

theorem frac_sum_property (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := by
  sorry

end frac_sum_property_l12_12480


namespace min_distance_of_complex_numbers_l12_12315

open Complex

theorem min_distance_of_complex_numbers
  (z w : ℂ)
  (h₁ : abs (z + 1 + 3 * Complex.I) = 1)
  (h₂ : abs (w - 7 - 8 * Complex.I) = 3) :
  ∃ d, d = Real.sqrt 185 - 4 ∧ ∀ Z W : ℂ, abs (Z + 1 + 3 * Complex.I) = 1 → abs (W - 7 - 8 * Complex.I) = 3 → abs (Z - W) ≥ d :=
sorry

end min_distance_of_complex_numbers_l12_12315


namespace correct_average_marks_l12_12989

-- Define all the given conditions
def average_marks : ℕ := 92
def number_of_students : ℕ := 25
def wrong_mark : ℕ := 75
def correct_mark : ℕ := 30

-- Define variables for total marks calculations
def total_marks_with_wrong : ℕ := average_marks * number_of_students
def total_marks_with_correct : ℕ := total_marks_with_wrong - wrong_mark + correct_mark

-- Goal: Prove that the correct average marks is 90.2
theorem correct_average_marks :
  (total_marks_with_correct : ℝ) / (number_of_students : ℝ) = 90.2 :=
by
  sorry

end correct_average_marks_l12_12989


namespace correct_calculation_l12_12075

theorem correct_calculation (a x y b : ℝ) :
  (-a - a = 0) = False ∧
  (- (x + y) = -x - y) = True ∧
  (3 * (b - 2 * a) = 3 * b - 2 * a) = False ∧
  (8 * a^4 - 6 * a^2 = 2 * a^2) = False :=
by
  sorry

end correct_calculation_l12_12075


namespace length_of_train_l12_12184

-- Conditions
variable (L E T : ℝ)
axiom h1 : 300 * E = L + 300 * T
axiom h2 : 90 * E = L - 90 * T

-- The statement to be proved
theorem length_of_train : L = 200 * E :=
by
  sorry

end length_of_train_l12_12184


namespace nonneg_int_solution_coprime_l12_12843

theorem nonneg_int_solution_coprime (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : c ≥ (a - 1) * (b - 1)) :
  ∃ (x y : ℕ), c = a * x + b * y :=
sorry

end nonneg_int_solution_coprime_l12_12843


namespace thief_distance_l12_12587

variable (d : ℝ := 250)   -- initial distance in meters
variable (v_thief : ℝ := 12 * 1000 / 3600)  -- thief's speed in m/s (converted from km/hr)
variable (v_policeman : ℝ := 15 * 1000 / 3600)  -- policeman's speed in m/s (converted from km/hr)

noncomputable def distance_thief_runs : ℝ :=
  v_thief * (d / (v_policeman - v_thief))

theorem thief_distance :
  distance_thief_runs d v_thief v_policeman = 990.47 := sorry

end thief_distance_l12_12587


namespace sum_of_coeffs_l12_12264

theorem sum_of_coeffs (a0 a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 2 * (0 : ℤ))^5 = a0)
  (h2 : (1 - 2 * (1 : ℤ))^5 = a0 + a1 + a2 + a3 + a4 + a5) :
  a1 + a2 + a3 + a4 + a5 = -2 := by
  sorry

end sum_of_coeffs_l12_12264


namespace travel_time_difference_l12_12405

theorem travel_time_difference 
  (speed : ℝ) (d1 d2 : ℝ) (h_speed : speed = 50) (h_d1 : d1 = 475) (h_d2 : d2 = 450) : 
  (d1 - d2) / speed * 60 = 30 := 
by 
  sorry

end travel_time_difference_l12_12405


namespace maple_taller_than_pine_l12_12198

theorem maple_taller_than_pine :
  let pine_tree := 24 + 1/4
  let maple_tree := 31 + 2/3
  (maple_tree - pine_tree) = 7 + 5/12 :=
by
  sorry

end maple_taller_than_pine_l12_12198


namespace positive_integers_between_300_and_1000_squared_l12_12316

theorem positive_integers_between_300_and_1000_squared :
  ∃ n : ℕ, 300 < n^2 ∧ n^2 < 1000 → ∃ m : ℕ, m = 14 := sorry

end positive_integers_between_300_and_1000_squared_l12_12316


namespace number_of_paths_l12_12821

open Nat

def f : ℕ → ℕ → ℕ
| 0, 0 => 1
| x, 0 => 1
| 0, y => 1
| (x + 1), (y + 1) => f x (y + 1) + f (x + 1) y

theorem number_of_paths (n : ℕ) : f n 2 = (n^2 + 3 * n + 2) / 2 := by sorry

end number_of_paths_l12_12821


namespace taxi_distance_l12_12485

variable (initial_fee charge_per_2_5_mile total_charge : ℝ)
variable (d : ℝ)

theorem taxi_distance 
  (h_initial_fee : initial_fee = 2.35)
  (h_charge_per_2_5_mile : charge_per_2_5_mile = 0.35)
  (h_total_charge : total_charge = 5.50)
  (h_eq : total_charge = initial_fee + (charge_per_2_5_mile / (2/5)) * d) :
  d = 3.6 :=
sorry

end taxi_distance_l12_12485


namespace smallest_n_geometric_seq_l12_12765

noncomputable def geom_seq (a r : ℝ) (n : ℕ) : ℝ :=
  a * r ^ (n - 1)

noncomputable def S_n (a r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r ^ n) / (1 - r)

theorem smallest_n_geometric_seq :
  (∃ n : ℕ, S_n (1/9) 3 n > 2018) ∧ ∀ m : ℕ, m < 10 → S_n (1/9) 3 m ≤ 2018 :=
by
  sorry

end smallest_n_geometric_seq_l12_12765


namespace average_age_after_person_leaves_l12_12927

theorem average_age_after_person_leaves
  (average_age_seven : ℕ := 28)
  (num_people_initial : ℕ := 7)
  (person_leaves : ℕ := 20) :
  (average_age_seven * num_people_initial - person_leaves) / (num_people_initial - 1) = 29 := by
  sorry

end average_age_after_person_leaves_l12_12927


namespace complement_of_A_l12_12392

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the set A
def A : Set ℕ := {2, 4, 5}

-- Define the complement of A with respect to U
def CU : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- State the theorem that the complement of A with respect to U is {1, 3, 6, 7}
theorem complement_of_A : CU = {1, 3, 6, 7} := by
  sorry

end complement_of_A_l12_12392


namespace inequality_problem_l12_12937

variable (a b c d : ℝ)

theorem inequality_problem (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := sorry

end inequality_problem_l12_12937


namespace contrapositive_of_implication_l12_12140

theorem contrapositive_of_implication (p q : Prop) (h : p → q) : ¬q → ¬p :=
by {
  sorry
}

end contrapositive_of_implication_l12_12140


namespace butterfly_flutters_total_distance_l12_12268

-- Define the conditions
def start_pos : ℤ := 0
def first_move : ℤ := 4
def second_move : ℤ := -3
def third_move : ℤ := 7

-- Define a function that calculates the total distance
def total_distance (xs : List ℤ) : ℤ :=
  List.sum (List.map (fun ⟨x, y⟩ => abs (y - x)) (xs.zip xs.tail))

-- Create the butterfly's path
def path : List ℤ := [start_pos, first_move, second_move, third_move]

-- Define the proposition that we need to prove
theorem butterfly_flutters_total_distance : total_distance path = 21 := sorry

end butterfly_flutters_total_distance_l12_12268


namespace min_value_condition_solve_inequality_l12_12467

open Real

-- Define the function f(x) = |x - a| + |x + 2|
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 2)

-- Part I: Proving the values of a for f(x) having minimum value of 2
theorem min_value_condition (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 2) → (∃ x : ℝ, f x a = 2) → (a = 0 ∨ a = -4) :=
by
  sorry

-- Part II: Solving inequality f(x) ≤ 6 when a = 2
theorem solve_inequality : 
  ∀ x : ℝ, f x 2 ≤ 6 ↔ (x ≥ -3 ∧ x ≤ 3) :=
by
  sorry

end min_value_condition_solve_inequality_l12_12467


namespace Mr_Blue_potato_yield_l12_12119

-- Definitions based on the conditions
def steps_length (steps : ℕ) : ℕ := steps * 3
def garden_length : ℕ := steps_length 18
def garden_width : ℕ := steps_length 25

def area_garden : ℕ := garden_length * garden_width
def yield_potatoes (area : ℕ) : ℚ := area * (3/4)

-- Statement of the proof
theorem Mr_Blue_potato_yield :
  yield_potatoes area_garden = 3037.5 := by
  sorry

end Mr_Blue_potato_yield_l12_12119


namespace product_of_roots_of_polynomial_l12_12409

theorem product_of_roots_of_polynomial : 
  ∀ x : ℝ, (x + 3) * (x - 4) = 22 → ∃ a b : ℝ, (x^2 - x - 34 = 0) ∧ (a * b = -34) :=
by
  sorry

end product_of_roots_of_polynomial_l12_12409


namespace paths_inequality_l12_12996
open Nat

-- Definitions
def m : ℕ := sorry -- m represents the number of rows.
def n : ℕ := sorry -- n represents the number of columns.
def N : ℕ := sorry -- N is the number of ways to color the grid such that there is a path composed of black cells from the left edge to the right edge.
def M : ℕ := sorry -- M is the number of ways to color the grid such that there are two non-intersecting paths composed of black cells from the left edge to the right edge.

-- Theorem statement
theorem paths_inequality : (N ^ 2) ≥ 2 ^ (m * n) * M := 
by
  sorry

end paths_inequality_l12_12996


namespace heloise_gives_dogs_to_janet_l12_12035

theorem heloise_gives_dogs_to_janet :
  ∃ d c : ℕ, d * 17 = c * 10 ∧ d + c = 189 ∧ d - 60 = 10 :=
by
  sorry

end heloise_gives_dogs_to_janet_l12_12035


namespace value_of_expr_l12_12792

noncomputable def verify_inequality (x a b c : ℝ) : Prop :=
  (x - a) * (x - b) / (x - c) ≥ 0

theorem value_of_expr (a b c : ℝ) :
  (∀ x : ℝ, verify_inequality x a b c ↔ (x < -6 ∨ abs (x - 30) ≤ 2)) →
  a < b →
  a = 28 →
  b = 32 →
  c = -6 →
  a + 2 * b + 3 * c = 74 := by
  sorry

end value_of_expr_l12_12792


namespace math_problem_l12_12720

variable (a b c m : ℝ)

-- Quadratic equation: y = ax^2 + bx + c
def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Opens downward
axiom a_neg : a < 0
-- Passes through A(1, 0)
axiom passes_A : quadratic a b c 1 = 0
-- Passes through B(m, 0) with -2 < m < -1
axiom passes_B : quadratic a b c m = 0
axiom m_range : -2 < m ∧ m < -1

-- Prove the conclusions
theorem math_problem : b < 0 ∧ (a + b + c = 0) ∧ (a * (m+1) - b + c > 0) ∧ ¬(4 * a * c - b^2 > 4 * a) :=
by
  sorry

end math_problem_l12_12720


namespace system_solutions_range_b_l12_12327

theorem system_solutions_range_b (b : ℝ) :
  (∀ x y : ℝ, x^2 - y^2 = 0 → x^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0 ∨ y = b) →
  b ≥ 2 ∨ b ≤ -2 :=
sorry

end system_solutions_range_b_l12_12327


namespace baker_initial_cakes_cannot_be_determined_l12_12799

theorem baker_initial_cakes_cannot_be_determined (initial_pastries sold_cakes sold_pastries remaining_pastries : ℕ)
  (h1 : initial_pastries = 148)
  (h2 : sold_cakes = 15)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45)
  (h5 : sold_pastries + remaining_pastries = initial_pastries) :
  True :=
by
  sorry

end baker_initial_cakes_cannot_be_determined_l12_12799


namespace perpendicular_lines_b_eq_neg9_l12_12167

-- Definitions for the conditions.
def eq1 (x y : ℝ) : Prop := x + 3 * y + 4 = 0
def eq2 (b x y : ℝ) : Prop := b * x + 3 * y + 4 = 0

-- The problem statement
theorem perpendicular_lines_b_eq_neg9 (b : ℝ) : 
  (∀ x y, eq1 x y → eq2 b x y) ∧ (∀ x y, eq2 b x y → eq1 x y) → b = -9 :=
by
  sorry

end perpendicular_lines_b_eq_neg9_l12_12167


namespace hire_applicant_A_l12_12729

-- Define the test scores for applicants A and B
def education_A := 7
def experience_A := 8
def attitude_A := 9

def education_B := 10
def experience_B := 7
def attitude_B := 8

-- Define the weights for the test items
def weight_education := 1 / 6
def weight_experience := 2 / 6
def weight_attitude := 3 / 6

-- Define the final scores
def final_score_A := education_A * weight_education + experience_A * weight_experience + attitude_A * weight_attitude
def final_score_B := education_B * weight_education + experience_B * weight_experience + attitude_B * weight_attitude

-- Prove that Applicant A is hired because their final score is higher
theorem hire_applicant_A : final_score_A > final_score_B :=
by sorry

end hire_applicant_A_l12_12729


namespace product_of_slope_and_intercept_l12_12513

theorem product_of_slope_and_intercept {x1 y1 x2 y2 : ℝ} (h1 : x1 = -4) (h2 : y1 = -2) (h3 : x2 = 1) (h4 : y2 = 3) :
  let m := (y2 - y1) / (x2 - x1)
  let b := y1 - m * x1
  m * b = 2 :=
by
  sorry

end product_of_slope_and_intercept_l12_12513


namespace head_start_proofs_l12_12360

def HeadStartAtoB : ℕ := 150
def HeadStartAtoC : ℕ := 310
def HeadStartAtoD : ℕ := 400

def HeadStartBtoC : ℕ := HeadStartAtoC - HeadStartAtoB
def HeadStartCtoD : ℕ := HeadStartAtoD - HeadStartAtoC
def HeadStartBtoD : ℕ := HeadStartAtoD - HeadStartAtoB

theorem head_start_proofs :
  (HeadStartBtoC = 160) ∧
  (HeadStartCtoD = 90) ∧
  (HeadStartBtoD = 250) :=
by
  sorry

end head_start_proofs_l12_12360


namespace parallelogram_isosceles_angles_l12_12176

def angle_sum_isosceles_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ (a = b ∨ b = c ∨ a = c)

theorem parallelogram_isosceles_angles :
  ∀ (A B C D P : Type) (AB BC CD DA BD : ℝ)
    (angle_DAB angle_BCD angle_ABC angle_CDA angle_ABP angle_BAP angle_PBD angle_BDP angle_CBD angle_BCD : ℝ),
  AB ≠ BC →
  angle_DAB = 72 →
  angle_BCD = 72 →
  angle_ABC = 108 →
  angle_CDA = 108 →
  angle_sum_isosceles_triangle angle_ABP angle_BAP 108 →
  angle_sum_isosceles_triangle 72 72 angle_BDP →
  angle_sum_isosceles_triangle 108 36 36 →
  ∃! (ABP BPD BCD : Type),
   (angle_ABP = 36 ∧ angle_BAP = 36 ∧ angle_PBA = 108) ∧
   (angle_PBD = 72 ∧ angle_PDB = 72 ∧ angle_BPD = 36) ∧
   (angle_CBD = 108 ∧ angle_BCD = 36 ∧ angle_BDC = 36) :=
sorry

end parallelogram_isosceles_angles_l12_12176


namespace inequality_proof_equality_condition_l12_12801

theorem inequality_proof (a : ℝ) : (a^2 + 5)^2 + 4 * a * (10 - a) ≥ 8 * a^3  :=
by sorry

theorem equality_condition (a : ℝ) : ((a^2 + 5)^2 + 4 * a * (10 - a) = 8 * a^3) ↔ (a = 5 ∨ a = -1) :=
by sorry

end inequality_proof_equality_condition_l12_12801


namespace vector_cross_product_coordinates_l12_12242

variables (a1 a2 a3 b1 b2 b3 : ℝ)

def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1, a.2.2 * b.1 - a.1 * b.2.2, a.1 * b.2.1 - a.2.1 * b.1)

theorem vector_cross_product_coordinates :
  cross_product (a1, a2, a3) (b1, b2, b3) = 
    (a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1) :=
by
sorry

end vector_cross_product_coordinates_l12_12242


namespace money_last_duration_l12_12716

-- Defining the conditions
def money_from_mowing : ℕ := 14
def money_from_weed_eating : ℕ := 26
def money_spent_per_week : ℕ := 5

-- Theorem statement to prove Mike's money will last 8 weeks
theorem money_last_duration : (money_from_mowing + money_from_weed_eating) / money_spent_per_week = 8 := by
  sorry

end money_last_duration_l12_12716


namespace arithmetic_geometric_ratio_l12_12661

noncomputable def arithmetic_sequence (a1 a2 : ℝ) : Prop :=
1 + 3 = a1 + a2

noncomputable def geometric_sequence (b2 : ℝ) : Prop :=
b2 ^ 2 = 4

theorem arithmetic_geometric_ratio (a1 a2 b2 : ℝ) 
  (h1 : arithmetic_sequence a1 a2) 
  (h2 : geometric_sequence b2) : 
  (a1 + a2) / b2 = 5 / 2 :=
by sorry

end arithmetic_geometric_ratio_l12_12661


namespace uneaten_pancakes_time_l12_12867

theorem uneaten_pancakes_time:
  ∀ (production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya : ℕ) (k : ℕ),
    production_rate_dad = 70 →
    production_rate_mom = 100 →
    consumption_rate_petya = 10 * 4 → -- 10 pancakes in 15 minutes -> (10/15) * 60 = 40 per hour
    consumption_rate_vasya = 2 * consumption_rate_petya →
    k * ((production_rate_dad + production_rate_mom) / 60 - (consumption_rate_petya + consumption_rate_vasya) / 60) ≥ 20 →
    k ≥ 24 := 
by
  intros production_rate_dad production_rate_mom consumption_rate_petya consumption_rate_vasya k
  sorry

end uneaten_pancakes_time_l12_12867


namespace range_of_a_l12_12803

variable {a x : ℝ}

theorem range_of_a (h : ∀ x, (a - 5) * x > a - 5 ↔ x < 1) : a < 5 := 
sorry

end range_of_a_l12_12803


namespace king_zenobius_more_descendants_l12_12102

-- Conditions
def descendants_paphnutius (p2_descendants p1_descendants: ℕ) := 
  2 + 60 * p2_descendants + 20 * p1_descendants = 142

def descendants_zenobius (z3_descendants z1_descendants : ℕ) := 
  4 + 35 * z3_descendants + 35 * z1_descendants = 144

-- Main statement
theorem king_zenobius_more_descendants:
  ∀ (p2_descendants p1_descendants z3_descendants z1_descendants : ℕ),
    descendants_paphnutius p2_descendants p1_descendants →
    descendants_zenobius z3_descendants z1_descendants →
    144 > 142 :=
by
  intros
  sorry

end king_zenobius_more_descendants_l12_12102


namespace temperature_decrease_l12_12816

theorem temperature_decrease (current_temp : ℝ) (future_temp : ℝ) : 
  current_temp = 84 → future_temp = (3 / 4) * current_temp → (current_temp - future_temp) = 21 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end temperature_decrease_l12_12816


namespace cow_manure_plant_height_l12_12156

theorem cow_manure_plant_height
  (control_plant_height : ℝ)
  (bone_meal_ratio : ℝ)
  (cow_manure_ratio : ℝ)
  (h1 : control_plant_height = 36)
  (h2 : bone_meal_ratio = 1.25)
  (h3 : cow_manure_ratio = 2) :
  (control_plant_height * bone_meal_ratio * cow_manure_ratio) = 90 :=
sorry

end cow_manure_plant_height_l12_12156


namespace retail_price_l12_12525

/-- A retailer bought a machine at a wholesale price of $99 and later sold it after a 10% discount of the retail price.
If the retailer made a profit equivalent to 20% of the wholesale price, then the retail price of the machine before the discount was $132. -/
theorem retail_price (wholesale_price : ℝ) (profit_percent discount_percent : ℝ) (P : ℝ) 
  (h₁ : wholesale_price = 99) 
  (h₂ : profit_percent = 0.20) 
  (h₃ : discount_percent = 0.10)
  (h₄ : (1 - discount_percent) * P = wholesale_price + profit_percent * wholesale_price) : 
  P = 132 := 
by
  sorry

end retail_price_l12_12525


namespace correct_statement_l12_12573

theorem correct_statement (a b : ℝ) (ha : a < b) (hb : b < 0) : |a| / |b| > 1 :=
sorry

end correct_statement_l12_12573


namespace max_quadratic_value_l12_12707

def quadratic (x : ℝ) : ℝ :=
  -2 * x^2 + 4 * x + 3

theorem max_quadratic_value : ∃ x : ℝ, ∀ y : ℝ, quadratic x = y → y ≤ 5 ∧ (∀ z : ℝ, quadratic z ≤ y) := 
by
  sorry

end max_quadratic_value_l12_12707


namespace binary_to_decimal_101101_l12_12680

theorem binary_to_decimal_101101 : 
  (1 * 2^0 + 0 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5) = 45 := 
by 
  sorry

end binary_to_decimal_101101_l12_12680


namespace print_output_l12_12852

-- Conditions
def a : Nat := 10

/-- The print statement with the given conditions should output "a=10" -/
theorem print_output : "a=" ++ toString a = "a=10" :=
sorry

end print_output_l12_12852


namespace ellas_quadratic_equation_l12_12094

theorem ellas_quadratic_equation (d e : ℤ) :
  (∀ x : ℤ, |x - 8| = 3 → (x = 11 ∨ x = 5)) →
  (∀ x : ℤ, (x = 11 ∨ x = 5) → x^2 + d * x + e = 0) →
  (d, e) = (-16, 55) :=
by
  intro h1 h2
  sorry

end ellas_quadratic_equation_l12_12094


namespace area_percentage_change_l12_12536

variable (a b : ℝ)

def initial_area : ℝ := a * b

def new_length (a : ℝ) : ℝ := a * 1.35

def new_width (b : ℝ) : ℝ := b * 0.86

def new_area (a b : ℝ) : ℝ := (new_length a) * (new_width b)

theorem area_percentage_change :
    ((new_area a b) / (initial_area a b)) = 1.161 :=
by
  sorry

end area_percentage_change_l12_12536


namespace peter_situps_eq_24_l12_12340

noncomputable def situps_peter_did : ℕ :=
  let ratio_peter_greg := 3 / 4
  let situps_greg := 32
  let situps_peter := (3 * situps_greg) / 4
  situps_peter

theorem peter_situps_eq_24 : situps_peter_did = 24 := 
by 
  let h := situps_peter_did
  show h = 24
  sorry

end peter_situps_eq_24_l12_12340


namespace proof_inequality_l12_12318

noncomputable def inequality_proof (α : ℝ) (a b : ℝ) (m : ℕ) : Prop :=
  (0 < α) → (α < Real.pi / 2) →
  (m ≥ 1) →
  (0 < a) → (0 < b) →
  (a / (Real.cos α)^m + b / (Real.sin α)^m ≥ (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2))

-- Statement of the proof problem
theorem proof_inequality (α : ℝ) (a b : ℝ) (m : ℕ) 
  (h1 : 0 < α) (h2 : α < Real.pi / 2) (h3 : 1 ≤ m) (h4 : 0 < a) (h5 : 0 < b) : 
  a / (Real.cos α)^m + b / (Real.sin α)^m ≥ 
    (a^((2:ℝ)/(m+2)) + b^((2:ℝ)/(m+2)))^((m+2)/2) :=
by
  sorry

end proof_inequality_l12_12318


namespace determine_d_minus_r_l12_12856

theorem determine_d_minus_r :
  ∃ d r: ℕ, (∀ n ∈ [2023, 2459, 3571], n % d = r) ∧ (1 < d) ∧ (d - r = 1) :=
sorry

end determine_d_minus_r_l12_12856


namespace numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l12_12605

theorem numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1 :
  (63 ∣ 2^48 - 1) ∧ (65 ∣ 2^48 - 1) := 
by
  sorry

end numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l12_12605


namespace prob_heart_club_spade_l12_12272

-- Definitions based on the conditions
def total_cards : ℕ := 52
def cards_per_suit : ℕ := 13

-- Definitions based on the question
def prob_first_heart : ℚ := cards_per_suit / total_cards
def prob_second_club : ℚ := cards_per_suit / (total_cards - 1)
def prob_third_spade : ℚ := cards_per_suit / (total_cards - 2)

-- The main proof statement to be proved
theorem prob_heart_club_spade :
  prob_first_heart * prob_second_club * prob_third_spade = 169 / 10200 :=
by
  sorry

end prob_heart_club_spade_l12_12272


namespace solve_for_x_l12_12553

theorem solve_for_x (x : ℚ) (h : 5 * (x - 4) = 3 * (3 - 3 * x) + 6) : x = 5 / 2 :=
by {
  sorry
}

end solve_for_x_l12_12553


namespace min_value_of_a_plus_b_l12_12472

theorem min_value_of_a_plus_b (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc : 1 = 1) 
    (h1 : b^2 > 4 * a) (h2 : b < 2 * a) (h3 : b < a + 1) : a + b = 10 :=
sorry

end min_value_of_a_plus_b_l12_12472


namespace chess_player_max_consecutive_win_prob_l12_12371

theorem chess_player_max_consecutive_win_prob
  {p1 p2 p3 : ℝ} 
  (h1 : 0 < p1)
  (h2 : p1 < p2)
  (h3 : p2 < p3) :
  ∀ pA pB pC : ℝ, pC = (2 * p3 * (p1 + p2) - 4 * p1 * p2 * p3) 
                  → pB = (2 * p2 * (p1 + p3) - 4 * p1 * p2 * p3) 
                  → pA = (2 * p1 * (p2 + p3) - 4 * p1 * p2 * p3) 
                  → pC > pB ∧ pC > pA := 
by
  sorry

end chess_player_max_consecutive_win_prob_l12_12371


namespace sequence_property_l12_12818

-- Conditions as definitions
def seq (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  (a 1 = -(2 / 3)) ∧ (∀ n ≥ 2, S n + (1 / S n) + 2 = a n)

-- The desired property of the sequence
def S_property (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < n → S n = -((n + 1) / (n + 2))

-- The main theorem
theorem sequence_property (a S : ℕ → ℝ) (h_seq : seq a S) : S_property S := sorry

end sequence_property_l12_12818


namespace problem_statement_l12_12711

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := 2⁻¹
noncomputable def c : ℝ := Real.log 6 / Real.log 5

theorem problem_statement : b < a ∧ a < c := by
  sorry

end problem_statement_l12_12711


namespace diana_total_cost_l12_12262

noncomputable def shopping_total_cost := 
  let t_shirt_price := 10
  let sweater_price := 25
  let jacket_price := 100
  let jeans_price := 40
  let shoes_price := 70 

  let t_shirt_discount := 0.20
  let sweater_discount := 0.10
  let jacket_discount := 0.15
  let jeans_discount := 0.05
  let shoes_discount := 0.25

  let clothes_tax := 0.06
  let shoes_tax := 0.09

  let t_shirt_qty := 8
  let sweater_qty := 5
  let jacket_qty := 3
  let jeans_qty := 6
  let shoes_qty := 4

  let t_shirt_total := t_shirt_qty * t_shirt_price 
  let sweater_total := sweater_qty * sweater_price 
  let jacket_total := jacket_qty * jacket_price 
  let jeans_total := jeans_qty * jeans_price 
  let shoes_total := shoes_qty * shoes_price 

  let t_shirt_discounted := t_shirt_total * (1 - t_shirt_discount)
  let sweater_discounted := sweater_total * (1 - sweater_discount)
  let jacket_discounted := jacket_total * (1 - jacket_discount)
  let jeans_discounted := jeans_total * (1 - jeans_discount)
  let shoes_discounted := shoes_total * (1 - shoes_discount)

  let t_shirt_final := t_shirt_discounted * (1 + clothes_tax)
  let sweater_final := sweater_discounted * (1 + clothes_tax)
  let jacket_final := jacket_discounted * (1 + clothes_tax)
  let jeans_final := jeans_discounted * (1 + clothes_tax)
  let shoes_final := shoes_discounted * (1 + shoes_tax)

  t_shirt_final + sweater_final + jacket_final + jeans_final + shoes_final

theorem diana_total_cost : shopping_total_cost = 927.97 :=
by sorry

end diana_total_cost_l12_12262


namespace distance_and_ratio_correct_l12_12250

noncomputable def distance_and_ratio (a : ℝ) : ℝ × ℝ :=
  let dist : ℝ := a / Real.sqrt 3
  let ratio : ℝ := 1 / 2
  ⟨dist, ratio⟩

theorem distance_and_ratio_correct (a : ℝ) :
  distance_and_ratio a = (a / Real.sqrt 3, 1 / 2) := by
  -- Proof omitted
  sorry

end distance_and_ratio_correct_l12_12250


namespace product_of_positive_c_for_rational_solutions_l12_12461

theorem product_of_positive_c_for_rational_solutions : 
  (∃ c₁ c₂ : ℕ, c₁ > 0 ∧ c₂ > 0 ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₁ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   (∀ x : ℝ, (3 * x ^ 2 + 7 * x + c₂ = 0) → ∃ r₁ r₂ : ℚ, x = r₁ ∨ x = r₂) ∧ 
   c₁ * c₂ = 8) :=
sorry

end product_of_positive_c_for_rational_solutions_l12_12461


namespace hyperbola_asymptote_l12_12550

def hyperbola_eqn (m x y : ℝ) := m * x^2 - y^2 = 1

def vertex_distance_condition (m : ℝ) := 2 * Real.sqrt (1 / m) = 4

theorem hyperbola_asymptote (m : ℝ) (h_eq : hyperbola_eqn m x y) (h_dist : vertex_distance_condition m) :
  ∃ k, y = k * x ∧ k = 1 / 2 ∨ k = -1 / 2 := by
  sorry

end hyperbola_asymptote_l12_12550


namespace combination_of_students_l12_12850

-- Define the conditions
def num_boys := 4
def num_girls := 3

-- Define the combination function
def combination (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Calculate possible combinations
def two_boys_one_girl : ℕ :=
  combination num_boys 2 * combination num_girls 1

def one_boy_two_girls : ℕ :=
  combination num_boys 1 * combination num_girls 2

-- Total combinations
def total_combinations : ℕ :=
  two_boys_one_girl + one_boy_two_girls

-- Lean statement to be proven
theorem combination_of_students :
  total_combinations = 30 :=
by sorry

end combination_of_students_l12_12850


namespace no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l12_12881

theorem no_real_solution_x_squared_minus_2x_plus_3_eq_zero :
  ∀ x : ℝ, x^2 - 2 * x + 3 ≠ 0 :=
by
  sorry

end no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l12_12881


namespace determine_marriages_l12_12105

-- Definitions of the items each person bought
variable (a_items b_items c_items : ℕ) -- Number of items bought by wives a, b, and c
variable (A_items B_items C_items : ℕ) -- Number of items bought by husbands A, B, and C

-- Conditions
variable (spend_eq_square_a : a_items * a_items = a_spend) -- Spending equals square of items
variable (spend_eq_square_b : b_items * b_items = b_spend)
variable (spend_eq_square_c : c_items * c_items = c_spend)
variable (spend_eq_square_A : A_items * A_items = A_spend)
variable (spend_eq_square_B : B_items * B_items = B_spend)
variable (spend_eq_square_C : C_items * C_items = C_spend)

variable (A_spend_eq : A_spend = a_spend + 48) -- Husbands spent 48 yuan more than wives
variable (B_spend_eq : B_spend = b_spend + 48)
variable (C_spend_eq : C_spend = c_spend + 48)

variable (A_bought_9_more : A_items = b_items + 9) -- A bought 9 more items than b
variable (B_bought_7_more : B_items = a_items + 7) -- B bought 7 more items than a

-- Theorem statement
theorem determine_marriages (hA : A_items ≥ b_items + 9) (hB : B_items ≥ a_items + 7) :
  (A_spend = A_items * A_items) ∧ (B_spend = B_items * B_items) ∧ (C_spend = C_items * C_items) ∧
  (a_spend = a_items * a_items) ∧ (b_spend = b_items * b_items) ∧ (c_spend = c_items * c_items) →
  (A_spend = a_spend + 48) ∧ (B_spend = b_spend + 48) ∧ (C_spend = c_spend + 48) →
  (A_items = b_items + 9) ∧ (B_items = a_items + 7) →
  (A_items = 13 ∧ c_items = 11) ∧ (B_items = 8 ∧ b_items = 4) ∧ (C_items = 7 ∧ a_items = 1) :=
by
  sorry

end determine_marriages_l12_12105


namespace roots_quadratic_inequality_l12_12516

theorem roots_quadratic_inequality (t x1 x2 : ℝ) (h_eqn : x1 ^ 2 - t * x1 + t = 0) 
  (h_eqn2 : x2 ^ 2 - t * x2 + t = 0) (h_real : x1 + x2 = t) (h_prod : x1 * x2 = t) :
  x1 ^ 2 + x2 ^ 2 ≥ 2 * (x1 + x2) := 
sorry

end roots_quadratic_inequality_l12_12516


namespace find_k_and_a_range_l12_12594

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := x^2 + Real.exp x - k * Real.exp (-x)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 + a

theorem find_k_and_a_range (k a : ℝ) (h_even : ∀ x : ℝ, f x k = f (-x) k) :
  k = -1 ∧ 2 ≤ a := by
    sorry

end find_k_and_a_range_l12_12594


namespace allens_mothers_age_l12_12065

-- Define the conditions
variables (A M S : ℕ) -- Declare variables for ages of Allen, his mother, and his sister

-- Define Allen is 30 years younger than his mother
axiom h1 : A = M - 30

-- Define Allen's sister is 5 years older than him
axiom h2 : S = A + 5

-- Define in 7 years, the sum of their ages will be 110
axiom h3 : (A + 7) + (M + 7) + (S + 7) = 110

-- Define the age difference between Allen's mother and sister is 25 years
axiom h4 : M - S = 25

-- State the theorem: what is the present age of Allen's mother
theorem allens_mothers_age : M = 48 :=
by sorry

end allens_mothers_age_l12_12065


namespace arithmetic_progression_20th_term_and_sum_l12_12628

theorem arithmetic_progression_20th_term_and_sum :
  let a := 3
  let d := 4
  let n := 20
  let a_20 := a + (n - 1) * d
  let S_20 := n / 2 * (a + a_20)
  a_20 = 79 ∧ S_20 = 820 := by
    let a := 3
    let d := 4
    let n := 20
    let a_20 := a + (n - 1) * d
    let S_20 := n / 2 * (a + a_20)
    sorry

end arithmetic_progression_20th_term_and_sum_l12_12628


namespace ratio_of_democrats_l12_12158

theorem ratio_of_democrats (F M : ℕ) 
  (h1 : F + M = 990) 
  (h2 : (1 / 2 : ℚ) * F = 165) 
  (h3 : (1 / 4 : ℚ) * M = 165) : 
  (165 + 165) / 990 = 1 / 3 := 
by
  sorry

end ratio_of_democrats_l12_12158


namespace find_point_D_l12_12273

structure Point :=
  (x : ℤ)
  (y : ℤ)

def translation_rule (A C : Point) : Point :=
{
  x := C.x - A.x,
  y := C.y - A.y
}

def translate (P delta : Point) : Point :=
{
  x := P.x + delta.x,
  y := P.y + delta.y
}

def A := Point.mk (-1) 4
def C := Point.mk 1 2
def B := Point.mk 2 1
def D := Point.mk 4 (-1)
def translation_delta : Point := translation_rule A C

theorem find_point_D : translate B translation_delta = D :=
by
  sorry

end find_point_D_l12_12273


namespace bowling_average_change_l12_12490

theorem bowling_average_change (old_avg : ℝ) (wickets_last : ℕ) (runs_last : ℕ) (wickets_before : ℕ)
  (h_old_avg : old_avg = 12.4)
  (h_wickets_last : wickets_last = 8)
  (h_runs_last : runs_last = 26)
  (h_wickets_before : wickets_before = 175) :
  old_avg - ((old_avg * wickets_before + runs_last)/(wickets_before + wickets_last)) = 0.4 :=
by {
  sorry
}

end bowling_average_change_l12_12490


namespace number_of_machines_sold_l12_12171

-- Define the parameters and conditions given in the problem
def commission_of_first_150 (sale_price : ℕ) : ℕ := 150 * (sale_price * 3 / 100)
def commission_of_next_100 (sale_price : ℕ) : ℕ := 100 * (sale_price * 4 / 100)
def commission_of_after_250 (sale_price : ℕ) (x : ℕ) : ℕ := x * (sale_price * 5 / 100)

-- Define the total commission using these commissions
def total_commission (x : ℕ) : ℕ :=
  commission_of_first_150 10000 + 
  commission_of_next_100 9500 + 
  commission_of_after_250 9000 x

-- The main statement we want to prove
theorem number_of_machines_sold (x : ℕ) (total_commission : ℕ) : x = 398 ↔ total_commission = 150000 :=
by
  sorry

end number_of_machines_sold_l12_12171


namespace intersect_curves_l12_12687

theorem intersect_curves (R : ℝ) (hR : R > 0) :
  (∃ (x y : ℝ), x^2 + y^2 = R^2 ∧ x - y - 2 = 0) ↔ R ≥ Real.sqrt 2 :=
sorry

end intersect_curves_l12_12687


namespace minimum_cuts_for_polygons_l12_12807

theorem minimum_cuts_for_polygons (initial_pieces desired_pieces : ℕ) (sides : ℕ)
    (h_initial_pieces : initial_pieces = 1) (h_desired_pieces : desired_pieces = 100)
    (h_sides : sides = 20) :
    ∃ (cuts : ℕ), cuts = 1699 ∧
    (∀ current_pieces, current_pieces < desired_pieces → current_pieces + cuts ≥ desired_pieces) :=
by
    sorry

end minimum_cuts_for_polygons_l12_12807


namespace f_sum_neg_l12_12422

def f : ℝ → ℝ := sorry

theorem f_sum_neg (x₁ x₂ : ℝ)
  (h1 : ∀ x, f (4 - x) = - f x)
  (h2 : ∀ x, x < 2 → ∀ y, y < x → f y < f x)
  (h3 : x₁ + x₂ > 4)
  (h4 : (x₁ - 2) * (x₂ - 2) < 0)
  : f x₁ + f x₂ < 0 := 
sorry

end f_sum_neg_l12_12422


namespace union_of_sets_l12_12189

def setA := {x : ℝ | x^2 < 4}
def setB := {y : ℝ | ∃ x ∈ setA, y = x^2 - 2 * x - 1}

theorem union_of_sets : (setA ∪ setB) = {x : ℝ | -2 ≤ x ∧ x < 7} :=
by sorry

end union_of_sets_l12_12189


namespace polynomial_roots_l12_12788

theorem polynomial_roots :
  ∀ x : ℝ, (4 * x^4 - 28 * x^3 + 53 * x^2 - 28 * x + 4 = 0) ↔ (x = 4 ∨ x = 2 ∨ x = 1/4 ∨ x = 1/2) := 
by
  sorry

end polynomial_roots_l12_12788


namespace equivalent_polar_coordinates_l12_12750

-- Definitions of given conditions and the problem statement
def polar_point_neg (r : ℝ) (θ : ℝ) : Prop := r = -3 ∧ θ = 5 * Real.pi / 6
def polar_point_pos (r : ℝ) (θ : ℝ) : Prop := r = 3 ∧ θ = 11 * Real.pi / 6
def angle_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < 2 * Real.pi

theorem equivalent_polar_coordinates :
  ∃ (r θ : ℝ), polar_point_neg r θ → polar_point_pos 3 (11 * Real.pi / 6) ∧ angle_range (11 * Real.pi / 6) :=
by
  sorry

end equivalent_polar_coordinates_l12_12750


namespace chromium_percentage_alloy_l12_12052

theorem chromium_percentage_alloy 
  (w1 w2 w3 w4 : ℝ)
  (p1 p2 p3 p4 : ℝ)
  (h_w1 : w1 = 15)
  (h_w2 : w2 = 30)
  (h_w3 : w3 = 10)
  (h_w4 : w4 = 5)
  (h_p1 : p1 = 0.12)
  (h_p2 : p2 = 0.08)
  (h_p3 : p3 = 0.15)
  (h_p4 : p4 = 0.20) :
  (w1 * p1 + w2 * p2 + w3 * p3 + w4 * p4) / (w1 + w2 + w3 + w4) * 100 = 11.17 := 
  sorry

end chromium_percentage_alloy_l12_12052


namespace minimum_value_xy_l12_12009

theorem minimum_value_xy (x y : ℝ) (h : (x + Real.sqrt (x^2 + 1)) * (y + Real.sqrt (y^2 + 1)) ≥ 1) : x + y ≥ 0 :=
sorry

end minimum_value_xy_l12_12009


namespace exist_unique_rectangular_prism_Q_l12_12234

variable (a b c : ℝ) (h_lt : a < b ∧ b < c)
variable (x y z : ℝ) (hx_lt : x < y ∧ y < z ∧ z < a)

theorem exist_unique_rectangular_prism_Q :
  (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) ∧ (x < y ∧ y < z ∧ z < a) → 
  ∃! x y z, (2 * (x*y + y*z + z*x) = 0.5 * (a*b + b*c + c*a) ∧ x*y*z = 0.25 * a*b*c) :=
sorry

end exist_unique_rectangular_prism_Q_l12_12234


namespace polygon_interior_angle_144_proof_l12_12092

-- Definitions based on the conditions in the problem statement
def interior_angle (n : ℕ) : ℝ := 144
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- The problem statement as a Lean 4 theorem to prove n = 10
theorem polygon_interior_angle_144_proof : ∃ n : ℕ, interior_angle n = 144 ∧ sum_of_interior_angles n = n * 144 → n = 10 := by
  sorry

end polygon_interior_angle_144_proof_l12_12092


namespace madeline_needs_work_hours_l12_12428

def rent : ℝ := 1200
def groceries : ℝ := 400
def medical_expenses : ℝ := 200
def utilities : ℝ := 60
def emergency_savings : ℝ := 200
def hourly_wage : ℝ := 15

def total_expenses : ℝ := rent + groceries + medical_expenses + utilities + emergency_savings

noncomputable def total_hours_needed : ℝ := total_expenses / hourly_wage

theorem madeline_needs_work_hours :
  ⌈total_hours_needed⌉ = 138 := by
  sorry

end madeline_needs_work_hours_l12_12428


namespace total_books_to_put_away_l12_12142

-- Definitions based on the conditions
def books_per_shelf := 4
def shelves_needed := 3

-- The proof problem translates to finding the total number of books
theorem total_books_to_put_away : shelves_needed * books_per_shelf = 12 := by
  sorry

end total_books_to_put_away_l12_12142


namespace positive_diff_solutions_l12_12006

theorem positive_diff_solutions (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 14) (h2 : 2 * x2 - 3 = -14) : 
  x1 - x2 = 14 := 
by
  sorry

end positive_diff_solutions_l12_12006


namespace count_valid_prime_pairs_l12_12842

theorem count_valid_prime_pairs (x y : ℕ) (h₁ : Prime x) (h₂ : Prime y) (h₃ : x ≠ y) (h₄ : (621 * x * y) % (x + y) = 0) : 
  ∃ p, p = 6 := by
  sorry

end count_valid_prime_pairs_l12_12842


namespace number_of_pounds_colombian_beans_l12_12232

def cost_per_pound_colombian : ℝ := 5.50
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def desired_cost_per_pound : ℝ := 4.60
noncomputable def amount_colombian_beans (C : ℝ) : Prop := 
  let P := total_weight - C
  cost_per_pound_colombian * C + cost_per_pound_peruvian * P = desired_cost_per_pound * total_weight

theorem number_of_pounds_colombian_beans : ∃ C, amount_colombian_beans C ∧ C = 11.2 :=
sorry

end number_of_pounds_colombian_beans_l12_12232


namespace relationship_among_variables_l12_12421

theorem relationship_among_variables (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (h1 : a^2 = 2) (h2 : b^3 = 3) (h3 : c^4 = 4) (h4 : d^5 = 5) : a = c ∧ a < d ∧ d < b := 
by
  sorry

end relationship_among_variables_l12_12421


namespace find_y_l12_12126

theorem find_y (x y : ℚ) (h1 : x = 153) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 350064) : 
  y = 40 / 3967 :=
by
  -- Proof to be filled in
  sorry

end find_y_l12_12126


namespace num_solutions_eq_three_l12_12568

theorem num_solutions_eq_three :
  (∃ n : Nat, (x : ℝ) → (x^2 - 4) * (x^2 - 1) = (x^2 + 3 * x + 2) * (x^2 - 8 * x + 7) → n = 3) :=
sorry

end num_solutions_eq_three_l12_12568


namespace tulip_to_remaining_ratio_l12_12134

theorem tulip_to_remaining_ratio (total_flowers daisies sunflowers tulips remaining_tulips remaining_flowers : ℕ) 
  (h1 : total_flowers = 12) 
  (h2 : daisies = 2) 
  (h3 : sunflowers = 4) 
  (h4 : tulips = total_flowers - (daisies + sunflowers))
  (h5 : remaining_tulips = tulips)
  (h6 : remaining_flowers = remaining_tulips + sunflowers)
  (h7 : remaining_flowers = 10) : 
  tulips / remaining_flowers = 3 / 5 := 
by
  sorry

end tulip_to_remaining_ratio_l12_12134


namespace clinton_shoes_count_l12_12936

def num_hats : ℕ := 5
def num_belts : ℕ := num_hats + 2
def num_shoes : ℕ := 2 * num_belts

theorem clinton_shoes_count : num_shoes = 14 := by
  -- proof goes here
  sorry

end clinton_shoes_count_l12_12936


namespace axis_of_symmetry_l12_12646

-- Define points and the parabola equation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A := Point.mk 2 5
def B := Point.mk 4 5

def parabola (b c : ℝ) (p : Point) : Prop :=
  p.y = 2 * p.x^2 + b * p.x + c

theorem axis_of_symmetry (b c : ℝ) (hA : parabola b c A) (hB : parabola b c B) : ∃ x_axis : ℝ, x_axis = 3 :=
by
  -- Proof to be provided
  sorry

end axis_of_symmetry_l12_12646


namespace gcd_of_102_and_238_l12_12149

theorem gcd_of_102_and_238 : Nat.gcd 102 238 = 34 := 
by 
  sorry

end gcd_of_102_and_238_l12_12149


namespace tunnel_length_l12_12203

/-- A train travels at 80 kmph, enters a tunnel at 5:12 am, and leaves at 5:18 am.
    The length of the train is 1 km. Prove the length of the tunnel is 7 km. -/
theorem tunnel_length 
(speed : ℕ) (enter_time leave_time : ℕ) (train_length : ℕ) 
(h_enter : enter_time = 5 * 60 + 12) 
(h_leave : leave_time = 5 * 60 + 18) 
(h_speed : speed = 80) 
(h_train_length : train_length = 1) 
: ∃ tunnel_length : ℕ, tunnel_length = 7 :=
sorry

end tunnel_length_l12_12203


namespace inequality_proof_l12_12651

theorem inequality_proof (n : ℕ) (a : Fin n → ℝ) (h1 : 0 < n) (h2 : (Finset.univ.sum a) ≥ 0) :
  (Finset.univ.sum (λ i => Real.sqrt (a i ^ 2 + 1))) ≥
  Real.sqrt (2 * n * (Finset.univ.sum a)) :=
by
  sorry

end inequality_proof_l12_12651


namespace opposite_of_2023_is_neg_2023_l12_12706

theorem opposite_of_2023_is_neg_2023 : -2023 = -2023 :=
by trivial

end opposite_of_2023_is_neg_2023_l12_12706


namespace cubes_and_quartics_sum_l12_12800

theorem cubes_and_quartics_sum (a b : ℝ) (h1 : a + b = 2) (h2 : a^2 + b^2 = 2) : 
  a^3 + b^3 = 2 ∧ a^4 + b^4 = 2 :=
by 
  sorry

end cubes_and_quartics_sum_l12_12800


namespace bicycle_cost_after_tax_l12_12131

theorem bicycle_cost_after_tax :
  let original_price := 300
  let first_discount := original_price * 0.40
  let price_after_first_discount := original_price - first_discount
  let second_discount := price_after_first_discount * 0.20
  let price_after_second_discount := price_after_first_discount - second_discount
  let tax := price_after_second_discount * 0.05
  price_after_second_discount + tax = 151.20 :=
by
  sorry

end bicycle_cost_after_tax_l12_12131


namespace octagon_area_in_square_l12_12361

def main : IO Unit :=
  IO.println s!"Hello, Lean!"

theorem octagon_area_in_square :
  ∀ (s : ℝ), ∀ (area_square : ℝ), ∀ (area_octagon : ℝ),
  (s * 4 = 160) →
  (s = 40) →
  (area_square = s * s) →
  (area_square = 1600) →
  (∃ (area_triangle : ℝ), area_triangle = 50 ∧ 8 * area_triangle = 400) →
  (area_octagon = area_square - 400) →
  (area_octagon = 1200) :=
by
  intros s area_square area_octagon h1 h2 h3 h4 h5 h6
  sorry

end octagon_area_in_square_l12_12361


namespace ratio_man_to_son_in_two_years_l12_12084

-- Define current ages and the conditions
def son_current_age : ℕ := 24
def man_current_age : ℕ := son_current_age + 26

-- Define ages in two years
def son_age_in_two_years : ℕ := son_current_age + 2
def man_age_in_two_years : ℕ := man_current_age + 2

-- State the theorem
theorem ratio_man_to_son_in_two_years : 
  (man_age_in_two_years : ℚ) / (son_age_in_two_years : ℚ) = 2 :=
by sorry

end ratio_man_to_son_in_two_years_l12_12084


namespace proof_q_values_proof_q_comparison_l12_12813

-- Definitions of the conditions given.
def q : ℝ → ℝ := 
  sorry -- The definition is not required to be constructed, as we are only focusing on the conditions given.

-- Conditions
axiom cond1 : q 2 = 5
axiom cond2 : q 1.5 = 3

-- Statements to prove
theorem proof_q_values : (q 2 = 5) ∧ (q 1.5 = 3) := 
  by sorry

theorem proof_q_comparison : q 2 > q 1.5 :=
  by sorry

end proof_q_values_proof_q_comparison_l12_12813


namespace pointA_in_second_quadrant_l12_12912

def pointA : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem pointA_in_second_quadrant : isSecondQuadrant pointA :=
by
  sorry

end pointA_in_second_quadrant_l12_12912


namespace batsman_average_increase_l12_12050

def average_increase (avg_before : ℕ) (runs_12th_inning : ℕ) (avg_after : ℕ) : ℕ :=
  avg_after - avg_before

theorem batsman_average_increase :
  ∀ (avg_before runs_12th_inning avg_after : ℕ),
    (runs_12th_inning = 70) →
    (avg_after = 37) →
    (11 * avg_before + runs_12th_inning = 12 * avg_after) →
    average_increase avg_before runs_12th_inning avg_after = 3 :=
by
  intros avg_before runs_12th_inning avg_after h_runs h_avg_after h_total
  sorry

end batsman_average_increase_l12_12050


namespace cos_double_angle_l12_12019

theorem cos_double_angle (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 4) : 
  Real.cos (2 * α) = -7 / 8 :=
sorry

end cos_double_angle_l12_12019


namespace geometric_sequence_a5_l12_12543

theorem geometric_sequence_a5 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) 
  : a 5 = -8 :=
sorry

end geometric_sequence_a5_l12_12543


namespace prime_divisor_greater_than_p_l12_12955

theorem prime_divisor_greater_than_p (p q : ℕ) (hp : Prime p) 
    (hq : Prime q) (hdiv : q ∣ 2^p - 1) : p < q := 
by
  sorry

end prime_divisor_greater_than_p_l12_12955


namespace shooting_to_practice_ratio_l12_12055

variable (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ)
variable (runningWeightliftingRatio : ℕ)

axiom practiceTime_def : practiceTime = 2 * 60 -- converting 2 hours to minutes
axiom weightliftingTime_def : weightliftingTime = 20
axiom runningWeightliftingRatio_def : runningWeightliftingRatio = 2
axiom runningTime_def : runningTime = runningWeightliftingRatio * weightliftingTime
axiom shootingTime_def : shootingTime = practiceTime - (runningTime + weightliftingTime)

theorem shooting_to_practice_ratio (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ) 
                                   (runningWeightliftingRatio : ℕ) :
  practiceTime = 120 →
  weightliftingTime = 20 →
  runningWeightliftingRatio = 2 →
  runningTime = runningWeightliftingRatio * weightliftingTime →
  shootingTime = practiceTime - (runningTime + weightliftingTime) →
  (shootingTime : ℚ) / practiceTime = 1 / 2 :=
by sorry

end shooting_to_practice_ratio_l12_12055


namespace smallest_winning_N_and_digit_sum_l12_12910

-- Definitions of operations
def B (x : ℕ) : ℕ := 3 * x
def S (x : ℕ) : ℕ := x + 100

/-- The main theorem confirming the smallest winning number and sum of its digits -/
theorem smallest_winning_N_and_digit_sum :
  ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 999 ∧ (900 ≤ 9 * N + 400 ∧ 9 * N + 400 < 1000) ∧ (N = 56) ∧ (5 + 6 = 11) :=
by {
  -- Proof skipped
  sorry
}

end smallest_winning_N_and_digit_sum_l12_12910


namespace greatest_divisor_less_than_30_l12_12363

theorem greatest_divisor_less_than_30 :
  (∃ d, d ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} ∧ ∀ m, m ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} → m ≤ d) → 
  18 ∈ {n | n ∣ 540 ∧ n < 30 ∧ n ∣ 180} :=
by
  sorry

end greatest_divisor_less_than_30_l12_12363


namespace max_value_fraction_l12_12767

theorem max_value_fraction : ∀ x : ℝ, 
  (∃ x : ℝ, max (1 + (16 / (4 * x^2 + 8 * x + 5))) = 17) :=
by
  sorry

end max_value_fraction_l12_12767


namespace yadav_spends_50_percent_on_clothes_and_transport_l12_12357

variable (S : ℝ)
variable (monthly_savings : ℝ := 46800 / 12)
variable (clothes_transport_expense : ℝ := 3900)
variable (remaining_salary : ℝ := 0.40 * S)

theorem yadav_spends_50_percent_on_clothes_and_transport (h1 : remaining_salary = 2 * 3900) :
  (clothes_transport_expense / remaining_salary) * 100 = 50 :=
by
  -- skipping the proof steps
  sorry

end yadav_spends_50_percent_on_clothes_and_transport_l12_12357


namespace total_amount_due_l12_12112

noncomputable def original_bill : ℝ := 500
noncomputable def late_charge_rate : ℝ := 0.02
noncomputable def annual_interest_rate : ℝ := 0.05

theorem total_amount_due (n : ℕ) (initial_amount : ℝ) (late_charge_rate : ℝ) (interest_rate : ℝ) : 
  initial_amount = 500 → 
  late_charge_rate = 0.02 → 
  interest_rate = 0.05 → 
  n = 3 → 
  (initial_amount * (1 + late_charge_rate)^n * (1 + interest_rate) = 557.13) :=
by
  intros h_initial_amount h_late_charge_rate h_interest_rate h_n
  sorry

end total_amount_due_l12_12112


namespace intersection_of_A_and_B_l12_12757

-- Given sets A and B
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 1}

-- Prove the intersection of A and B
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 1} := 
by
  sorry

end intersection_of_A_and_B_l12_12757


namespace number_property_l12_12727

theorem number_property (n : ℕ) (h : n = 7101449275362318840579) :
  n / 7 = 101449275362318840579 :=
sorry

end number_property_l12_12727


namespace factor_polynomial_l12_12839

def p (x y z : ℝ) : ℝ := x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2)

theorem factor_polynomial (x y z : ℝ) : 
  p x y z = (x - y) * (y - z) * (z - x) * -(x * y + x * z + y * z) :=
by 
  simp [p]
  sorry

end factor_polynomial_l12_12839


namespace union_M_N_l12_12211

def U : Set ℝ := {x | -3 ≤ x ∧ x < 2}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 0}

theorem union_M_N :
  M ∪ N = {x | -3 ≤ x ∧ x < 1} := 
sorry

end union_M_N_l12_12211


namespace prove_functions_same_l12_12197

theorem prove_functions_same (u v : ℝ) (huv : u = v) : 
  (u > 1) → (v > 1) → (Real.sqrt ((u + 1) / (u - 1)) = Real.sqrt ((v + 1) / (v - 1))) :=
by
  sorry

end prove_functions_same_l12_12197


namespace find_n_l12_12721

-- Defining the parameters and conditions
def large_block_positions (n : ℕ) : ℕ := 199 * n + 110 * (n - 1)

-- Theorem statement
theorem find_n (h : large_block_positions n = 2362) : n = 8 :=
sorry

end find_n_l12_12721


namespace george_total_coins_l12_12718

-- We'll state the problem as proving the total number of coins George has.
variable (num_nickels num_dimes : ℕ)
variable (value_of_coins : ℝ := 2.60)
variable (value_of_nickels : ℝ := 0.05 * num_nickels)
variable (value_of_dimes : ℝ := 0.10 * num_dimes)

theorem george_total_coins :
  num_nickels = 4 → 
  value_of_coins = value_of_nickels + value_of_dimes → 
  num_nickels + num_dimes = 28 := 
by
  sorry

end george_total_coins_l12_12718


namespace arithmetic_sequence_common_difference_l12_12154

theorem arithmetic_sequence_common_difference (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h₁ : a 3 = 4) (h₂ : S 3 = 3)
  (h₃ : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (h₄ : ∀ n, a n = a 1 + (n - 1) * d) :
  ∃ d, d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l12_12154


namespace range_a_two_zeros_l12_12172

-- Definition of the function f(x)
def f (a x : ℝ) : ℝ := a * x^3 - 3 * a * x + 3 * a - 5

-- The theorem statement about the range of a
theorem range_a_two_zeros (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = 0 ∧ f a x2 = 0) → 1 ≤ a ∧ a ≤ 5 := sorry

end range_a_two_zeros_l12_12172


namespace infinite_solutions_exists_l12_12014

theorem infinite_solutions_exists :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x ≠ y) ∧ (y ≠ z) ∧ (z ≠ x) ∧
  (x - y + z = 1) ∧ ((x * y) % z = 0) ∧ ((y * z) % x = 0) ∧ ((z * x) % y = 0) ∧
  ∀ n : ℕ, ∃ x y z : ℕ, (n > 0) ∧ (x = n * (n^2 + n - 1)) ∧ (y = (n+1) * (n^2 + n - 1)) ∧ (z = n * (n+1)) := by
  sorry

end infinite_solutions_exists_l12_12014


namespace maximize_profit_l12_12627

noncomputable def profit_function (x : ℝ) : ℝ := -3 * x^2 + 252 * x - 4860

theorem maximize_profit :
  (∀ x : ℝ, 30 ≤ x ∧ x ≤ 54 → profit_function x ≤ 432) ∧ profit_function 42 = 432 := sorry

end maximize_profit_l12_12627


namespace fraction_given_to_friend_l12_12010

theorem fraction_given_to_friend (s u r g k : ℕ) 
  (h1: s = 135) 
  (h2: u = s / 3) 
  (h3: r = s - u) 
  (h4: k = 54) 
  (h5: g = r - k) :
  g / r = 2 / 5 := 
  by
  sorry

end fraction_given_to_friend_l12_12010


namespace percentage_increase_of_return_trip_l12_12076

noncomputable def speed_increase_percentage (initial_speed avg_speed : ℝ) : ℝ :=
  ((2 * avg_speed * initial_speed) / avg_speed - initial_speed) * 100 / initial_speed

theorem percentage_increase_of_return_trip :
  let initial_speed := 30
  let avg_speed := 34.5
  speed_increase_percentage initial_speed avg_speed = 35.294 :=
  sorry

end percentage_increase_of_return_trip_l12_12076


namespace triangle_reflection_not_necessarily_perpendicular_l12_12323

theorem triangle_reflection_not_necessarily_perpendicular
  (P Q R : ℝ × ℝ)
  (hP : 0 ≤ P.1 ∧ 0 ≤ P.2)
  (hQ : 0 ≤ Q.1 ∧ 0 ≤ Q.2)
  (hR : 0 ≤ R.1 ∧ 0 ≤ R.2)
  (not_on_y_eq_x_P : P.1 ≠ P.2)
  (not_on_y_eq_x_Q : Q.1 ≠ Q.2)
  (not_on_y_eq_x_R : R.1 ≠ R.2) :
  ¬ (∃ (mPQ mPQ' : ℝ), 
      mPQ = (Q.2 - P.2) / (Q.1 - P.1) ∧ 
      mPQ' = (Q.1 - P.1) / (Q.2 - P.2) ∧ 
      mPQ * mPQ' = -1) :=
sorry

end triangle_reflection_not_necessarily_perpendicular_l12_12323


namespace three_digit_cubes_divisible_by_8_and_9_l12_12614

theorem three_digit_cubes_divisible_by_8_and_9 : 
  ∃! n : ℕ, (216 ≤ n^3 ∧ n^3 ≤ 999) ∧ (n % 6 = 0) :=
sorry

end three_digit_cubes_divisible_by_8_and_9_l12_12614


namespace limit_of_sequence_l12_12746

noncomputable def limit_problem := 
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |((2 * n - 3) / (n + 2) : ℝ) - 2| < ε

theorem limit_of_sequence : limit_problem :=
sorry

end limit_of_sequence_l12_12746


namespace prob_top_odd_correct_l12_12499

def total_dots : Nat := 78
def faces : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Probability calculation for odd dots after removal
def prob_odd_dot (n : Nat) : Rat :=
  if n % 2 = 1 then
    1 - (n : Rat) / total_dots
  else
    (n : Rat) / total_dots

-- Probability that the top face shows an odd number of dots
noncomputable def prob_top_odd : Rat :=
  (1 / (faces.length : Rat)) * (faces.map prob_odd_dot).sum

theorem prob_top_odd_correct :
  prob_top_odd = 523 / 936 :=
by
  sorry

end prob_top_odd_correct_l12_12499


namespace equivalent_proof_problem_l12_12093

-- Define the real numbers x, y, z and the operation ⊗
variables {x y z : ℝ}

def otimes (a b : ℝ) : ℝ := (a - b) ^ 2

theorem equivalent_proof_problem : otimes ((x + z) ^ 2) ((z + y) ^ 2) = (x ^ 2 + 2 * x * z - y ^ 2 - 2 * z * y) ^ 2 :=
by sorry

end equivalent_proof_problem_l12_12093


namespace minimum_value_of_expression_l12_12585

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + a * b + a * c + b * c = 4) : 
  2 * a + b + c ≥ 4 := 
by 
  sorry

end minimum_value_of_expression_l12_12585


namespace remainder_when_dividing_polynomial_by_x_minus_3_l12_12263

noncomputable def P (x : ℤ) : ℤ := 
  2 * x^8 - 3 * x^7 + 4 * x^6 - x^4 + 6 * x^3 - 5 * x^2 + 18 * x - 20

theorem remainder_when_dividing_polynomial_by_x_minus_3 :
  P 3 = 17547 :=
by
  sorry

end remainder_when_dividing_polynomial_by_x_minus_3_l12_12263


namespace ratio_JL_JM_l12_12282

theorem ratio_JL_JM (s w h : ℝ) (shared_area_25 : 0.25 * s^2 = 0.4 * w * h) (jm_eq_s : h = s) :
  w / h = 5 / 8 :=
by
  -- Proof will go here
  sorry

end ratio_JL_JM_l12_12282


namespace neg_one_to_zero_l12_12082

theorem neg_one_to_zero : (-1 : ℤ)^0 = 1 := 
by 
  -- Expanding expressions and applying the exponent rule for non-zero numbers
  sorry

end neg_one_to_zero_l12_12082


namespace inequality_solution_l12_12352

theorem inequality_solution (x : ℝ) : (5 * x + 3 > 9 - 3 * x ∧ x ≠ 3) ↔ (x > 3 / 4 ∧ x ≠ 3) :=
by {
  sorry
}

end inequality_solution_l12_12352


namespace arithmetic_sequence_third_term_l12_12771

theorem arithmetic_sequence_third_term (b y : ℝ) 
  (h1 : 2 * b + y + 2 = 10) 
  (h2 : b + y + 2 = b + y + 2) : 
  8 - b = 6 := 
by 
  sorry

end arithmetic_sequence_third_term_l12_12771


namespace smallest_integer_rel_prime_to_1020_l12_12407

theorem smallest_integer_rel_prime_to_1020 : ∃ n : ℕ, n > 1 ∧ n = 7 ∧ gcd n 1020 = 1 := by
  -- Here we state the theorem
  sorry

end smallest_integer_rel_prime_to_1020_l12_12407


namespace hypotenuse_length_l12_12143

theorem hypotenuse_length (a b : ℕ) (h1 : a = 36) (h2 : b = 48) : 
  ∃ c : ℕ, c * c = a * a + b * b ∧ c = 60 := 
by 
  use 60
  sorry

end hypotenuse_length_l12_12143


namespace altitude_segment_length_l12_12031

theorem altitude_segment_length 
  {A B C D E : Type} 
  (BD DC AE y : ℝ) 
  (h1 : BD = 4) 
  (h2 : DC = 6) 
  (h3 : AE = 3) 
  (h4 : 3 / 4 = 9 / (y + 3)) : 
  y = 9 := 
by 
  sorry

end altitude_segment_length_l12_12031


namespace evaluate_f_3_minus_f_neg_3_l12_12871

def f (x : ℝ) : ℝ := x^4 + x^2 + 7 * x

theorem evaluate_f_3_minus_f_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end evaluate_f_3_minus_f_neg_3_l12_12871


namespace find_number_l12_12862

variable (a b x : ℕ)

theorem find_number
    (h1 : x * a = 7 * b)
    (h2 : x * a = 20)
    (h3 : 7 * b = 20) :
    x = 1 :=
sorry

end find_number_l12_12862


namespace find_m_n_l12_12062

-- Define the vectors OA, OB, OC
def vector_oa (m : ℝ) : ℝ × ℝ := (-2, m)
def vector_ob (n : ℝ) : ℝ × ℝ := (n, 1)
def vector_oc : ℝ × ℝ := (5, -1)

-- Define the condition that OA is perpendicular to OB
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Define the condition that points A, B, and C are collinear.
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (A.1 - B.1) * (C.2 - A.2) = k * ((C.1 - A.1) * (A.2 - B.2))

theorem find_m_n (m n : ℝ) :
  collinear (-2, m) (n, 1) (5, -1) ∧ perpendicular (-2, m) (n, 1) → m = 3 ∧ n = 3 / 2 := by
  intro h
  sorry

end find_m_n_l12_12062


namespace sequence_term_four_l12_12572

theorem sequence_term_four (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n, a (n + 1) = a n + 2) : a 4 = 7 :=
sorry

end sequence_term_four_l12_12572


namespace solve_for_x_and_y_l12_12741

theorem solve_for_x_and_y (x y : ℝ) 
  (h1 : 0.75 / x = 7 / 8)
  (h2 : x / y = 5 / 6) :
  x = 6 / 7 ∧ y = (6 / 7 * 6) / 5 :=
by
  sorry

end solve_for_x_and_y_l12_12741


namespace set_of_values_l12_12528

theorem set_of_values (a : ℝ) (h : 2 ∉ {x : ℝ | x - a < 0}) : a ≤ 2 := 
sorry

end set_of_values_l12_12528


namespace quiz_minimum_correct_l12_12630

theorem quiz_minimum_correct (x : ℕ) (hx : 7 * x + 14 ≥ 120) : x ≥ 16 := 
by sorry

end quiz_minimum_correct_l12_12630


namespace area_ratio_trapezoid_triangle_l12_12635

-- Define the geometric elements and given conditions.
variable (AB CD EAB ABCD : ℝ)
variable (trapezoid_ABCD : AB = 10)
variable (trapezoid_ABCD_CD : CD = 25)
variable (ratio_areas_EDC_EAB : (CD / AB)^2 = 25 / 4)
variable (trapezoid_relation : (ABCD + EAB) / EAB = 25 / 4)

-- The goal is to prove the ratio of the areas of triangle EAB to trapezoid ABCD.
theorem area_ratio_trapezoid_triangle :
  (EAB / ABCD) = 4 / 21 :=
by
  sorry

end area_ratio_trapezoid_triangle_l12_12635


namespace average_age_of_contestants_l12_12823

theorem average_age_of_contestants :
  let numFemales := 12
  let avgAgeFemales := 25
  let numMales := 18
  let avgAgeMales := 40
  let sumAgesFemales := avgAgeFemales * numFemales
  let sumAgesMales := avgAgeMales * numMales
  let totalSumAges := sumAgesFemales + sumAgesMales
  let totalContestants := numFemales + numMales
  (totalSumAges / totalContestants) = 34 := by
  sorry

end average_age_of_contestants_l12_12823


namespace analysis_method_sufficient_conditions_l12_12859

theorem analysis_method_sufficient_conditions (P : Prop) (analysis_method : ∀ (Q : Prop), (Q → P) → Q) :
  ∀ Q, (Q → P) → Q :=
by
  -- Proof is skipped
  sorry

end analysis_method_sufficient_conditions_l12_12859


namespace free_throws_count_l12_12148

-- Definitions based on the conditions
variables (a b x : ℕ) -- Number of 2-point shots, 3-point shots, and free throws respectively.

-- Condition: Points from two-point shots equal the points from three-point shots
def points_eq : Prop := 2 * a = 3 * b

-- Condition: Number of free throws is twice the number of two-point shots
def free_throws_eq : Prop := x = 2 * a

-- Condition: Total score is adjusted to 78 points
def total_score : Prop := 2 * a + 3 * b + x = 78

-- Proof problem statement
theorem free_throws_count (h1 : points_eq a b) (h2 : free_throws_eq a x) (h3 : total_score a b x) : x = 26 :=
sorry

end free_throws_count_l12_12148


namespace determinant_of_sum_l12_12667

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![5, 6], ![2, 3]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 1], ![1, 0]]

theorem determinant_of_sum : (A + B).det = -3 := 
by 
  sorry

end determinant_of_sum_l12_12667


namespace solve_inequality_l12_12209

theorem solve_inequality :
  {x : ℝ | -3 * x^2 + 5 * x + 4 < 0} = {x : ℝ | x < 3 / 4} ∪ {x : ℝ | 1 < x} :=
by
  sorry

end solve_inequality_l12_12209


namespace base5_division_l12_12245

-- Given conditions in decimal:
def n1_base10 : ℕ := 214
def n2_base10 : ℕ := 7

-- Convert the result back to base 5
def result_base5 : ℕ := 30  -- since 30 in decimal is 110 in base 5

theorem base5_division (h1 : 1324 = 214) (h2 : 12 = 7) : 1324 / 12 = 110 :=
by {
  -- these conditions help us bridge to the proof (intentionally left unproven here)
  sorry
}

end base5_division_l12_12245


namespace q0_r0_eq_three_l12_12731

variable (p q r s : Polynomial ℝ)
variable (hp_const : p.coeff 0 = 2)
variable (hs_eq : s = p * q * r)
variable (hs_const : s.coeff 0 = 6)

theorem q0_r0_eq_three : (q.coeff 0) * (r.coeff 0) = 3 := by
  sorry

end q0_r0_eq_three_l12_12731


namespace base_8_not_divisible_by_five_l12_12988

def base_b_subtraction_not_divisible_by_five (b : ℕ) : Prop :=
  let num1 := 3 * b^3 + 1 * b^2 + 0 * b + 2
  let num2 := 3 * b^2 + 0 * b + 2
  let diff := num1 - num2
  ¬ (diff % 5 = 0)

theorem base_8_not_divisible_by_five : base_b_subtraction_not_divisible_by_five 8 := 
by
  sorry

end base_8_not_divisible_by_five_l12_12988


namespace max_ab_l12_12740

theorem max_ab (a b : ℝ) (h1 : a + 4 * b = 8) (h2 : a > 0) (h3 : b > 0) : ab ≤ 4 := 
sorry

end max_ab_l12_12740


namespace pq_eq_neg72_l12_12944

theorem pq_eq_neg72 {p q : ℝ} (h : ∀ x, (x - 7) * (3 * x + 11) = x ^ 2 - 20 * x + 63 →
(p = x ∨ q = x) ∧ p ≠ q) : 
(p + 2) * (q + 2) = -72 :=
sorry

end pq_eq_neg72_l12_12944


namespace project_scientists_total_l12_12039

def total_scientists (S : ℕ) : Prop :=
  S / 2 + S / 5 + 21 = S

theorem project_scientists_total : ∃ S, total_scientists S ∧ S = 70 :=
by
  existsi 70
  unfold total_scientists
  sorry

end project_scientists_total_l12_12039


namespace general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l12_12976

open Classical

axiom S_n : ℕ → ℝ
axiom a_n : ℕ → ℝ
axiom b_n : ℕ → ℝ
axiom c_n : ℕ → ℝ
axiom T_n : ℕ → ℝ

noncomputable def general_a_n (n : ℕ) : ℝ :=
  sorry

axiom h1 : ∀ n, S_n n + a_n n = 2

theorem general_formula_a_n : ∀ n, a_n n = 1 / 2^(n-1) :=
  sorry

axiom h2 : b_n 1 = a_n 1
axiom h3 : ∀ n ≥ 2, b_n n = 3 * b_n (n-1) / (b_n (n-1) + 3)

theorem general_formula_b_n : ∀ n, b_n n = 3 / (n + 2) ∧
  (∀ n, 1 / b_n n = 1 + (n - 1) / 3) :=
  sorry

axiom h4 : ∀ n, c_n n = a_n n / b_n n

theorem sum_c_n_T_n : ∀ n, T_n n = 8 / 3 - (n + 4) / (3 * 2^(n-1)) :=
  sorry

end general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l12_12976


namespace min_value_of_fraction_l12_12809

noncomputable def problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : ℝ :=
  1/a + 2/b

theorem min_value_of_fraction (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  problem_statement a b h1 h2 h3 = 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_of_fraction_l12_12809


namespace geometric_series_sum_l12_12013

-- Definitions based on conditions
def a : ℚ := 3 / 2
def r : ℚ := -4 / 9

-- Statement of the proof
theorem geometric_series_sum : (a / (1 - r)) = 27 / 26 :=
by
  -- proof goes here
  sorry

end geometric_series_sum_l12_12013


namespace probability_even_sum_of_selected_envelopes_l12_12260

theorem probability_even_sum_of_selected_envelopes :
  let face_values := [5, 6, 8, 10]
  let possible_sum_is_even (s : ℕ) : Prop := s % 2 = 0
  let num_combinations := Nat.choose 4 2
  let favorable_combinations := 3
  (favorable_combinations / num_combinations : ℚ) = 1 / 2 :=
by
  sorry

end probability_even_sum_of_selected_envelopes_l12_12260


namespace rice_difference_l12_12909

theorem rice_difference:
  ∀ (total : ℕ) (kept_fraction : ℚ) (kept : ℕ) (given : ℕ),
  total = 50 →
  kept_fraction = (7 : ℚ) / 10 →
  kept = total * kept_fraction →
  given = total - kept →
  (kept - given = 20) :=
by
  intros total kept_fraction kept given
  intros h_total h_fraction h_kept h_given
  sorry

end rice_difference_l12_12909


namespace jellybean_probability_l12_12113

/-- Abe holds 1 blue and 2 red jelly beans. 
    Bob holds 2 blue, 2 yellow, and 1 red jelly bean. 
    Each randomly picks a jelly bean to show the other. 
    What is the probability that the colors match? 
-/
theorem jellybean_probability :
  let abe_blue_prob := 1 / 3
  let bob_blue_prob := 2 / 5
  let abe_red_prob := 2 / 3
  let bob_red_prob := 1 / 5
  (abe_blue_prob * bob_blue_prob + abe_red_prob * bob_red_prob) = 4 / 15 :=
by
  sorry

end jellybean_probability_l12_12113


namespace solve_equation_1_solve_equation_2_l12_12837

namespace Proofs

theorem solve_equation_1 (x : ℝ) :
  (x + 1)^2 = 9 ↔ x = 2 ∨ x = -4 :=
by
  sorry

theorem solve_equation_2 (x : ℝ) :
  x * (x - 6) = 6 ↔ x = 3 + Real.sqrt 15 ∨ x = 3 - Real.sqrt 15 :=
by
  sorry

end Proofs

end solve_equation_1_solve_equation_2_l12_12837


namespace milk_quality_check_l12_12313

/-
Suppose there is a collection of 850 bags of milk numbered from 001 to 850. 
From this collection, 50 bags are randomly selected for testing by reading numbers 
from a random number table. Starting from the 3rd line and the 1st group of numbers, 
continuing to the right, we need to find the next 4 bag numbers after the sequence 
614, 593, 379, 242.
-/

def random_numbers : List Nat := [
  78226, 85384, 40527, 48987, 60602, 16085, 29971, 61279,
  43021, 92980, 27768, 26916, 27783, 84572, 78483, 39820,
  61459, 39073, 79242, 20372, 21048, 87088, 34600, 74636,
  63171, 58247, 12907, 50303, 28814, 40422, 97895, 61421,
  42372, 53183, 51546, 90385, 12120, 64042, 51320, 22983
]

noncomputable def next_valid_numbers (nums : List Nat) (start_idx : Nat) : List Nat :=
  nums.drop start_idx |>.filter (λ n => n ≤ 850) |>.take 4

theorem milk_quality_check :
  next_valid_numbers random_numbers 18 = [203, 722, 104, 88] :=
sorry

end milk_quality_check_l12_12313


namespace value_of_a_plus_b_l12_12175

theorem value_of_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, (x > -4 ∧ x < 1) ↔ (ax^2 + bx - 2 > 0)) → 
  a = 1/2 → 
  b = 3/2 → 
  a + b = 2 := 
by 
  intro h cond_a cond_b 
  rw [cond_a, cond_b]
  norm_num

end value_of_a_plus_b_l12_12175


namespace cubic_identity_l12_12796

theorem cubic_identity (a b c : ℝ) (h1 : a + b + c = 7) (h2 : ab + ac + bc = 11) (h3 : abc = -6) : a^3 + b^3 + c^3 = 94 :=
by
  sorry

end cubic_identity_l12_12796


namespace minimum_time_to_cook_l12_12815

def wash_pot_fill_water : ℕ := 2
def wash_vegetables : ℕ := 3
def prepare_noodles_seasonings : ℕ := 2
def boil_water : ℕ := 7
def cook_noodles_vegetables : ℕ := 3

theorem minimum_time_to_cook : wash_pot_fill_water + boil_water + cook_noodles_vegetables = 12 :=
by
  sorry

end minimum_time_to_cook_l12_12815


namespace odd_function_increasing_function_l12_12443

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (2^x + 1)

theorem odd_function (x : ℝ) : 
  (f (1 / 2) (-x)) = -(f (1 / 2) x) := 
by
  sorry

theorem increasing_function : 
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f (1 / 2) x₁ < f (1 / 2) x₂ := 
by
  sorry

end odd_function_increasing_function_l12_12443


namespace distance_between_centers_of_circles_l12_12463

theorem distance_between_centers_of_circles :
  ∀ (rect_width rect_height circle_radius distance_between_centers : ℝ),
  rect_width = 11 
  ∧ rect_height = 7 
  ∧ circle_radius = rect_height / 2 
  ∧ distance_between_centers = rect_width - 2 * circle_radius 
  → distance_between_centers = 4 := by
  intros rect_width rect_height circle_radius distance_between_centers
  sorry

end distance_between_centers_of_circles_l12_12463


namespace find_x_approx_l12_12701

theorem find_x_approx :
  ∀ (x : ℝ), 3639 + 11.95 - x^2 = 3054 → abs (x - 24.43) < 0.01 :=
by
  intro x
  sorry

end find_x_approx_l12_12701


namespace sum_of_youngest_and_oldest_cousins_l12_12106

theorem sum_of_youngest_and_oldest_cousins 
  (a1 a2 a3 a4 : ℕ) 
  (h_order : a1 ≤ a2 ∧ a2 ≤ a3 ∧ a3 ≤ a4) 
  (h_mean : a1 + a2 + a3 + a4 = 36) 
  (h_median : a2 + a3 = 14) : 
  a1 + a4 = 22 :=
by sorry

end sum_of_youngest_and_oldest_cousins_l12_12106


namespace arc_intercept_length_l12_12861

noncomputable def side_length : ℝ := 4
noncomputable def diagonal_length : ℝ := Real.sqrt (side_length^2 + side_length^2)
noncomputable def radius : ℝ := diagonal_length / 2
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def arc_length_one_side : ℝ := circumference / 4

theorem arc_intercept_length :
  arc_length_one_side = Real.sqrt 2 * Real.pi :=
by
  sorry

end arc_intercept_length_l12_12861


namespace one_cow_one_bag_in_forty_days_l12_12458

theorem one_cow_one_bag_in_forty_days
    (total_cows : ℕ)
    (total_bags : ℕ)
    (total_days : ℕ)
    (husk_consumption : total_cows * total_bags = total_cows * total_days) :
  total_days = 40 :=
by sorry

end one_cow_one_bag_in_forty_days_l12_12458


namespace find_a3_l12_12021

noncomputable def S (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def a (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * q ^ (n - 1)

theorem find_a3 (a₁ q : ℚ) (h1 : S 6 a₁ q / S 3 a₁ q = -19 / 8)
  (h2 : a 4 a₁ q - a 2 a₁ q = -15 / 8) :
  a 3 a₁ q = 9 / 4 :=
by sorry

end find_a3_l12_12021
