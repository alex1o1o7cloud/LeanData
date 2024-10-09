import Mathlib

namespace inverse_proportion_function_neg_k_l868_86817

variable {k : ℝ}
variable {y1 y2 : ℝ}

theorem inverse_proportion_function_neg_k
  (h1 : k ≠ 0)
  (h2 : y1 > y2)
  (hA : y1 = k / (-2))
  (hB : y2 = k / 5) :
  k < 0 :=
sorry

end inverse_proportion_function_neg_k_l868_86817


namespace a_7_eq_64_l868_86807

-- Define the problem conditions using variables in Lean
variable {a : ℕ → ℝ} -- defining the sequence as a function from natural numbers to reals
variable {q : ℝ}  -- common ratio

-- The sequence is geometric
axiom geom_seq (n : ℕ) : a (n + 1) = a n * q

-- Conditions given in the problem
axiom condition1 : a 1 + a 2 = 3
axiom condition2 : a 2 + a 3 = 6

-- Target statement to prove
theorem a_7_eq_64 : a 7 = 64 := 
sorry

end a_7_eq_64_l868_86807


namespace well_diameter_l868_86860

noncomputable def calculateDiameter (volume depth : ℝ) : ℝ :=
  2 * Real.sqrt (volume / (Real.pi * depth))

theorem well_diameter :
  calculateDiameter 678.5840131753953 24 = 6 :=
by
  sorry

end well_diameter_l868_86860


namespace circles_symmetric_sin_cos_l868_86858

noncomputable def sin_cos_product (θ : Real) : Real := Real.sin θ * Real.cos θ

theorem circles_symmetric_sin_cos (a θ : Real) 
(h1 : ∃ x1 y1, x1 = -a / 2 ∧ y1 = 0 ∧ 2*x1 - y1 - 1 = 0) 
(h2 : ∃ x2 y2, x2 = -a ∧ y2 = -Real.tan θ / 2 ∧ 2*x2 - y2 - 1 = 0) :
sin_cos_product θ = -2 / 5 := 
sorry

end circles_symmetric_sin_cos_l868_86858


namespace train_speed_km_hr_calc_l868_86842

theorem train_speed_km_hr_calc :
  let length := 175 -- length of the train in meters
  let time := 3.499720022398208 -- time to cross the pole in seconds
  let speed_mps := length / time -- speed in meters per second
  let speed_kmph := speed_mps * 3.6 -- converting speed from m/s to km/hr
  speed_kmph = 180.025923226 := 
sorry

end train_speed_km_hr_calc_l868_86842


namespace simplify_log_expression_l868_86885

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem simplify_log_expression :
  let term1 := 1 / (log_base 20 3 + 1)
  let term2 := 1 / (log_base 12 5 + 1)
  let term3 := 1 / (log_base 8 7 + 1)
  term1 + term2 + term3 = 2 :=
by
  sorry

end simplify_log_expression_l868_86885


namespace soda_original_price_l868_86831

theorem soda_original_price (P : ℝ) (h1 : 1.5 * P = 6) : P = 4 :=
by
  sorry

end soda_original_price_l868_86831


namespace r_sq_plus_s_sq_l868_86839

variable {r s : ℝ}

theorem r_sq_plus_s_sq (h1 : r * s = 16) (h2 : r + s = 10) : r^2 + s^2 = 68 := 
by
  sorry

end r_sq_plus_s_sq_l868_86839


namespace age_difference_l868_86847

theorem age_difference (b_age : ℕ) (bro_age : ℕ) (h1 : b_age = 5) (h2 : b_age + bro_age = 19) : 
  bro_age - b_age = 9 :=
by
  sorry

end age_difference_l868_86847


namespace cyclist_total_heartbeats_l868_86876

theorem cyclist_total_heartbeats
  (heart_rate : ℕ := 120) -- beats per minute
  (race_distance : ℕ := 50) -- miles
  (pace : ℕ := 4) -- minutes per mile
  : (race_distance * pace) * heart_rate = 24000 := by
  sorry

end cyclist_total_heartbeats_l868_86876


namespace pencils_on_desk_l868_86830

theorem pencils_on_desk (pencils_in_drawer pencils_on_desk_initial pencils_total pencils_placed : ℕ)
  (h_drawer : pencils_in_drawer = 43)
  (h_desk_initial : pencils_on_desk_initial = 19)
  (h_total : pencils_total = 78) :
  pencils_placed = 16 := by
  sorry

end pencils_on_desk_l868_86830


namespace amount_first_set_correct_l868_86810

-- Define the amounts as constants
def total_amount : ℝ := 900.00
def amount_second_set : ℝ := 260.00
def amount_third_set : ℝ := 315.00

-- Define the amount given to the first set
def amount_first_set : ℝ :=
  total_amount - amount_second_set - amount_third_set

-- Statement: prove that the amount given to the first set of families equals $325.00
theorem amount_first_set_correct :
  amount_first_set = 325.00 :=
sorry

end amount_first_set_correct_l868_86810


namespace sum_non_solution_values_l868_86822

theorem sum_non_solution_values (A B C : ℝ) (h : ∀ x : ℝ, (x+B) * (A*x+36) / ((x+C) * (x+9)) = 4) :
  ∃ M : ℝ, M = - (B + 9) := 
sorry

end sum_non_solution_values_l868_86822


namespace handshake_problem_l868_86862

theorem handshake_problem :
  ∃ (a b : ℕ), a + b = 20 ∧ (a * (a - 1)) / 2 + (b * (b - 1)) / 2 = 106 ∧ a * b = 84 :=
by
  sorry

end handshake_problem_l868_86862


namespace hcf_of_two_numbers_l868_86855

noncomputable def H : ℕ := 322 / 14

theorem hcf_of_two_numbers (H k : ℕ) (lcm_val : ℕ) :
  lcm_val = H * 13 * 14 ∧ 322 = H * k ∧ 322 / 14 = H → H = 23 :=
by
  sorry

end hcf_of_two_numbers_l868_86855


namespace install_time_for_windows_l868_86844

theorem install_time_for_windows
  (total_windows installed_windows hours_per_window : ℕ)
  (h1 : total_windows = 200)
  (h2 : installed_windows = 65)
  (h3 : hours_per_window = 12) :
  (total_windows - installed_windows) * hours_per_window = 1620 :=
by
  sorry

end install_time_for_windows_l868_86844


namespace intersection_M_N_eq_2_4_l868_86899

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℕ := {x | ∃ y, y = Real.log (6 - x) ∧ x < 6}

theorem intersection_M_N_eq_2_4 : M ∩ N = {2, 4} :=
by sorry

end intersection_M_N_eq_2_4_l868_86899


namespace ellipse_equation_line_equation_l868_86819
-- Import the necessary libraries

-- Problem (I): The equation of the ellipse
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (hA : (1 : ℝ) / a^2 + (9 / 4 : ℝ) / b^2 = 1)
  (h_ecc : b^2 = (3 / 4 : ℝ) * a^2) : 
  (a^2 = 4 ∧ b^2 = 3) :=
by
  sorry

-- Problem (II): The equation of the line
theorem line_equation (k : ℝ) (h_area : (12 * Real.sqrt (2 : ℝ)) / 7 = 12 * abs k / (4 * k^2 + 3)) : 
  k = 1 ∨ k = -1 :=
by
  sorry

end ellipse_equation_line_equation_l868_86819


namespace total_bugs_eaten_l868_86896

theorem total_bugs_eaten :
  let gecko_bugs := 12
  let lizard_bugs := gecko_bugs / 2
  let frog_bugs := lizard_bugs * 3
  let toad_bugs := frog_bugs + (frog_bugs / 2)
  gecko_bugs + lizard_bugs + frog_bugs + toad_bugs = 63 :=
by
  sorry

end total_bugs_eaten_l868_86896


namespace find_g_26_l868_86845

variable {g : ℕ → ℕ}

theorem find_g_26 (hg : ∀ x, g (x + g x) = 5 * g x) (h1 : g 1 = 5) : g 26 = 120 :=
  sorry

end find_g_26_l868_86845


namespace sector_area_l868_86821

theorem sector_area (r : ℝ) (alpha : ℝ) (h_r : r = 2) (h_alpha : alpha = π / 4) : 
  (1 / 2) * r^2 * alpha = π / 2 :=
by
  rw [h_r, h_alpha]
  -- proof steps would go here
  sorry

end sector_area_l868_86821


namespace maddie_total_cost_l868_86850

theorem maddie_total_cost :
  let price_palette := 15
  let price_lipstick := 2.5
  let price_hair_color := 4
  let num_palettes := 3
  let num_lipsticks := 4
  let num_hair_colors := 3
  let total_cost := (num_palettes * price_palette) + (num_lipsticks * price_lipstick) + (num_hair_colors * price_hair_color)
  total_cost = 67 := by
  sorry

end maddie_total_cost_l868_86850


namespace max_min_x_plus_y_l868_86853

theorem max_min_x_plus_y (x y : ℝ) (h : |x + 2| + |1 - x| = 9 - |y - 5| - |1 + y|) :
  -3 ≤ x + y ∧ x + y ≤ 6 := 
sorry

end max_min_x_plus_y_l868_86853


namespace number_of_real_roots_l868_86800

noncomputable def f (x : ℝ) : ℝ := x ^ 3 - 6 * x ^ 2 + 9 * x - 10

theorem number_of_real_roots : ∃! x : ℝ, f x = 0 :=
sorry

end number_of_real_roots_l868_86800


namespace isosceles_triangle_angle_l868_86866

theorem isosceles_triangle_angle (x : ℕ) (h1 : 2 * x + x + x = 180) :
  x = 45 ∧ 2 * x = 90 :=
by
  have h2 : 4 * x = 180 := by linarith
  have h3 : x = 45 := by linarith
  have h4 : 2 * x = 90 := by linarith
  exact ⟨h3, h4⟩

end isosceles_triangle_angle_l868_86866


namespace multiplier_of_first_integer_l868_86869

theorem multiplier_of_first_integer :
  ∃ m x : ℤ, x + 4 = 15 ∧ x * m = 3 + 2 * 15 ∧ m = 3 := by
  sorry

end multiplier_of_first_integer_l868_86869


namespace value_of_a_plus_b_l868_86829

theorem value_of_a_plus_b 
  (a b : ℝ) 
  (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = a * x + b)
  (h₂ : ∀ x, g x = 3 * x - 6)
  (h₃ : ∀ x, g (f x) = 4 * x + 5) : 
  a + b = 5 :=
sorry

end value_of_a_plus_b_l868_86829


namespace train_speed_l868_86864

/--
Given:
- The speed of the first person \(V_p\) is 4 km/h.
- The train takes 9 seconds to pass the first person completely.
- The length of the train is approximately 50 meters (49.999999999999986 meters).

Prove:
- The speed of the train \(V_t\) is 24 km/h.
-/
theorem train_speed (V_p : ℝ) (t : ℝ) (L : ℝ) (V_t : ℝ) 
  (hV_p : V_p = 4) 
  (ht : t = 9)
  (hL : L = 49.999999999999986)
  (hrel_speed : (L / t) * 3.6 = V_t - V_p) :
  V_t = 24 :=
by
  sorry

end train_speed_l868_86864


namespace a_lt_one_l868_86836

-- Define the function f(x) = |x-3| + |x+7|
def f (x : ℝ) : ℝ := |x-3| + |x+7|

-- The statement of the problem
theorem a_lt_one (a : ℝ) :
  (∀ x : ℝ, a < Real.log (f x)) → a < 1 :=
by
  intro h
  have H : f (-7) = 10 := by sorry -- piecewise definition
  have H1 : Real.log (f (-7)) = 1 := by sorry -- minimum value of log
  specialize h (-7)
  rw [H1] at h
  exact h

end a_lt_one_l868_86836


namespace count_heads_at_night_l868_86868

variables (J T D : ℕ)

theorem count_heads_at_night (h1 : 2 * J + 4 * T + 2 * D = 56) : J + T + D = 14 :=
by
  -- Skip the proof
  sorry

end count_heads_at_night_l868_86868


namespace total_amount_l868_86802

-- Definitions directly derived from the conditions in the problem
variable (you_spent friend_spent : ℕ)
variable (h1 : friend_spent = you_spent + 1)
variable (h2 : friend_spent = 8)

-- The goal is to prove that the total amount spent on lunch is $15
theorem total_amount : you_spent + friend_spent = 15 := by
  sorry

end total_amount_l868_86802


namespace correct_statements_count_l868_86826

theorem correct_statements_count :
  (¬(1 = 1) ∧ ¬(1 = 0)) ∧
  (¬(1 = 11)) ∧
  ((1 - 2 + 1 / 2) = 3) ∧
  (2 = 2) →
  2 = ([false, false, true, true].count true) := 
sorry

end correct_statements_count_l868_86826


namespace determine_a_if_derivative_is_even_l868_86875

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x^2 + x

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem determine_a_if_derivative_is_even (a : ℝ) :
  (∀ x : ℝ, f' x a = f' (-x) a) → a = 0 :=
by
  intros h
  sorry

end determine_a_if_derivative_is_even_l868_86875


namespace probability_two_even_multiples_of_five_drawn_l868_86815

-- Definition of conditions
def toys : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
                      39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

def isEvenMultipleOfFive (n : ℕ) : Bool := n % 10 == 0

-- Collect all such numbers from the list
def evenMultiplesOfFive : List ℕ := toys.filter isEvenMultipleOfFive

-- Number of such even multiples of 5
def countEvenMultiplesOfFive : ℕ := evenMultiplesOfFive.length

theorem probability_two_even_multiples_of_five_drawn :
  (countEvenMultiplesOfFive / 50) * ((countEvenMultiplesOfFive - 1) / 49) = 2 / 245 :=
  by sorry

end probability_two_even_multiples_of_five_drawn_l868_86815


namespace area_of_trapezium_l868_86818

-- Defining the lengths of the sides and the distance
def a : ℝ := 12  -- 12 cm
def b : ℝ := 16  -- 16 cm
def h : ℝ := 14  -- 14 cm

-- Statement that the area of the trapezium is 196 cm²
theorem area_of_trapezium : (1 / 2) * (a + b) * h = 196 :=
by
  sorry

end area_of_trapezium_l868_86818


namespace fraction_subtraction_simplification_l868_86805

/-- Given that 57 equals 19 times 3, we want to prove that (8/19) - (5/57) equals 1/3. -/
theorem fraction_subtraction_simplification :
  8 / 19 - 5 / 57 = 1 / 3 := by
  sorry

end fraction_subtraction_simplification_l868_86805


namespace arithmetic_seq_sum_l868_86806

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℤ → ℤ) 
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) 
  (h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 110) : 
  S 15 = 330 := 
by
  sorry

end arithmetic_seq_sum_l868_86806


namespace digits_satisfy_sqrt_l868_86835

theorem digits_satisfy_sqrt (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  (b = 0 ∧ a = 0) ∨ (b = 3 ∧ a = 1) ∨ (b = 6 ∧ a = 4) ∨ (b = 9 ∧ a = 9) ↔ b^2 = 9 * a :=
by
  sorry

end digits_satisfy_sqrt_l868_86835


namespace determine_k_and_a_n_and_T_n_l868_86859

noncomputable def S_n (n : ℕ) (k : ℝ) : ℝ := -0.5 * n^2 + k * n

/-- Given the sequence S_n with sum of the first n terms S_n := -1/2 n^2 + k*n,
where k is a positive natural number. The maximum value of S_n is 8. -/
theorem determine_k_and_a_n_and_T_n (k : ℝ) (h : k = 4) :
  (∀ n : ℕ, S_n n k ≤ 8) ∧ 
  (∀ n : ℕ, ∃ a : ℝ, a = 9/2 - n) ∧
  (∀ n : ℕ, ∃ T : ℝ, T = 4 - (n + 2)/2^(n-1)) :=
by
  sorry

end determine_k_and_a_n_and_T_n_l868_86859


namespace simplify_expr_1_simplify_expr_2_l868_86816

-- The first problem
theorem simplify_expr_1 (a : ℝ) : 2 * a^2 - 3 * a - 5 * a^2 + 6 * a = -3 * a^2 + 3 * a := 
by
  sorry

-- The second problem
theorem simplify_expr_2 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

end simplify_expr_1_simplify_expr_2_l868_86816


namespace probability_of_winning_l868_86865

def total_products_in_box : ℕ := 6
def winning_products_in_box : ℕ := 2

theorem probability_of_winning : (winning_products_in_box : ℚ) / (total_products_in_box : ℚ) = 1 / 3 :=
by sorry

end probability_of_winning_l868_86865


namespace speed_of_first_bus_l868_86884

theorem speed_of_first_bus (v : ℕ) (h : (v + 60) * 4 = 460) : v = 55 :=
by
  sorry

end speed_of_first_bus_l868_86884


namespace two_polygons_sum_of_interior_angles_l868_86824

theorem two_polygons_sum_of_interior_angles
  (n1 n2 : ℕ) (h1 : Even n1) (h2 : Even n2) 
  (h_sum : (n1 - 2) * 180 + (n2 - 2) * 180 = 1800):
  (n1 = 4 ∧ n2 = 10) ∨ (n1 = 6 ∧ n2 = 8) :=
by
  sorry

end two_polygons_sum_of_interior_angles_l868_86824


namespace num_valid_m_values_for_distributing_marbles_l868_86877

theorem num_valid_m_values_for_distributing_marbles : 
  ∃ (m_values : Finset ℕ), m_values.card = 22 ∧ 
  ∀ m ∈ m_values, ∃ n : ℕ, m * n = 360 ∧ n > 1 ∧ m > 1 :=
by
  sorry

end num_valid_m_values_for_distributing_marbles_l868_86877


namespace students_attended_school_l868_86898

-- Definitions based on conditions
def total_students (S : ℕ) : Prop :=
  ∃ (L R : ℕ), 
    (L = S / 2) ∧ 
    (R = L / 4) ∧ 
    (5 = R / 5)

-- Theorem stating the problem
theorem students_attended_school (S : ℕ) : total_students S → S = 200 :=
by
  intro h
  sorry

end students_attended_school_l868_86898


namespace probability_three_draws_one_white_l868_86813

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_white_balls + num_black_balls

def probability_one_white_three_draws : ℚ := 
  (num_white_balls / total_balls) * 
  ((num_black_balls - 1) / (total_balls - 1)) * 
  ((num_black_balls - 2) / (total_balls - 2)) * 3

theorem probability_three_draws_one_white :
  probability_one_white_three_draws = 12 / 35 := by sorry

end probability_three_draws_one_white_l868_86813


namespace solve_system_l868_86833

theorem solve_system :
  ∀ x y : ℚ, (3 * x + 4 * y = 12) ∧ (9 * x - 12 * y = -24) →
  (x = 2 / 3) ∧ (y = 5 / 2) :=
by
  intro x y
  intro h
  sorry

end solve_system_l868_86833


namespace num_four_letter_initials_sets_l868_86801

def num_initials_sets : ℕ := 8 ^ 4

theorem num_four_letter_initials_sets:
  num_initials_sets = 4096 :=
by
  rw [num_initials_sets]
  norm_num

end num_four_letter_initials_sets_l868_86801


namespace equivalent_expression_l868_86814

-- Let a, b, c, d, e be real numbers
variables (a b c d e : ℝ)

-- Condition given in the problem
def condition : Prop := 81 * a - 27 * b + 9 * c - 3 * d + e = -5

-- Objective: Prove that 8 * a - 4 * b + 2 * c - d + e = -5 given the condition
theorem equivalent_expression (h : condition a b c d e) : 8 * a - 4 * b + 2 * c - d + e = -5 :=
sorry

end equivalent_expression_l868_86814


namespace compound_interest_is_correct_l868_86883

noncomputable def compoundInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R)^T - P

theorem compound_interest_is_correct
  (P : ℝ)
  (R : ℝ)
  (T : ℝ)
  (SI : ℝ) : SI = P * R * T / 100 ∧ R = 0.10 ∧ T = 2 ∧ SI = 600 → compoundInterest P R T = 630 :=
by
  sorry

end compound_interest_is_correct_l868_86883


namespace side_length_of_square_l868_86843

theorem side_length_of_square (total_length : ℝ) (sides : ℕ) (h1 : total_length = 100) (h2 : sides = 4) :
  (total_length / (sides : ℝ) = 25) :=
by
  sorry

end side_length_of_square_l868_86843


namespace season_duration_l868_86854

-- Define the given conditions.
def games_per_month : ℕ := 7
def games_per_season : ℕ := 14

-- Define the property we want to prove.
theorem season_duration : games_per_season / games_per_month = 2 :=
by
  sorry

end season_duration_l868_86854


namespace real_solutions_l868_86890

theorem real_solutions :
  ∀ x : ℝ, 
  (1 / ((x - 1) * (x - 2)) + 
   1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 
   1 / ((x - 4) * (x - 5)) = 1 / 10) 
  ↔ (x = 10 ∨ x = -3.5) :=
by
  sorry

end real_solutions_l868_86890


namespace weight_of_bag_l868_86891

-- Definitions
def chicken_price : ℝ := 1.50
def bag_cost : ℝ := 2
def feed_per_chicken : ℝ := 2
def profit_from_50_chickens : ℝ := 65
def total_chickens : ℕ := 50

-- Theorem
theorem weight_of_bag : 
  (bag_cost / (profit_from_50_chickens - 
               (total_chickens * chicken_price)) / 
               (feed_per_chicken * total_chickens)) = 20 := 
sorry

end weight_of_bag_l868_86891


namespace no_burial_needed_for_survivors_l868_86811

def isSurvivor (p : Person) : Bool := sorry
def isBuried (p : Person) : Bool := sorry
variable (p : Person) (accident : Bool)

theorem no_burial_needed_for_survivors (h : accident = true) (hsurvive : isSurvivor p = true) : isBuried p = false :=
sorry

end no_burial_needed_for_survivors_l868_86811


namespace steven_weight_l868_86870

theorem steven_weight (danny_weight : ℝ) (steven_more : ℝ) (steven_weight : ℝ) 
  (h₁ : danny_weight = 40) 
  (h₂ : steven_more = 0.2 * danny_weight) 
  (h₃ : steven_weight = danny_weight + steven_more) : 
  steven_weight = 48 := 
  by 
  sorry

end steven_weight_l868_86870


namespace sin_theta_tan_theta_iff_first_third_quadrant_l868_86892

open Real

-- Definitions from conditions
def in_first_or_third_quadrant (θ : ℝ) : Prop :=
  (0 < θ ∧ θ < π / 2) ∨ (π < θ ∧ θ < 3 * π / 2)

def sin_theta_plus_tan_theta_positive (θ : ℝ) : Prop :=
  sin θ + tan θ > 0

-- Proof statement
theorem sin_theta_tan_theta_iff_first_third_quadrant (θ : ℝ) :
  sin_theta_plus_tan_theta_positive θ ↔ in_first_or_third_quadrant θ :=
sorry

end sin_theta_tan_theta_iff_first_third_quadrant_l868_86892


namespace sqrt_sum_l868_86889

theorem sqrt_sum (m n : ℝ) (h1 : m + n = 0) (h2 : m * n = -2023) : m + 2 * m * n + n = -4046 :=
by sorry

end sqrt_sum_l868_86889


namespace volume_formula_correct_l868_86838

def volume_of_box (x : ℝ) : ℝ :=
  x * (16 - 2 * x) * (12 - 2 * x)

theorem volume_formula_correct (x : ℝ) (h : x ≤ 12 / 5) :
  volume_of_box x = 4 * x^3 - 56 * x^2 + 192 * x :=
by sorry

end volume_formula_correct_l868_86838


namespace simplify_expr_l868_86886

theorem simplify_expr (x y : ℝ) (P Q : ℝ) (hP : P = x^2 + y^2) (hQ : Q = x^2 - y^2) : 
  (P * Q / (P + Q)) + ((P + Q) / (P * Q)) = ((x^4 + y^4) ^ 2) / (2 * x^2 * (x^4 - y^4)) :=
by sorry

end simplify_expr_l868_86886


namespace sum_of_terms_l868_86894

-- Given the condition that the sequence a_n is an arithmetic sequence
-- with Sum S_n of first n terms such that S_3 = 9 and S_6 = 36,
-- prove that a_7 + a_8 + a_9 is 45.

variable (a : ℕ → ℝ) -- arithmetic sequence
variable (S : ℕ → ℝ) -- sum of the first n terms of the sequence

axiom sum_3 : S 3 = 9
axiom sum_6 : S 6 = 36
axiom sum_seq_arith : ∀ n : ℕ, S n = n * (a 1) + (n - 1) * n / 2 * (a 2 - a 1)

theorem sum_of_terms : a 7 + a 8 + a 9 = 45 :=
by {
  sorry
}

end sum_of_terms_l868_86894


namespace probability_black_given_not_white_l868_86871

theorem probability_black_given_not_white
  (total_balls : ℕ)
  (white_balls : ℕ)
  (yellow_balls : ℕ)
  (black_balls : ℕ)
  (H1 : total_balls = 25)
  (H2 : white_balls = 10)
  (H3 : yellow_balls = 5)
  (H4 : black_balls = 10)
  (H5 : total_balls = white_balls + yellow_balls + black_balls)
  (H6 : ¬white_balls = total_balls) :
  (10 / (25 - 10) : ℚ) = 2 / 3 :=
by
  sorry

end probability_black_given_not_white_l868_86871


namespace trigonometric_eq_solution_count_l868_86856

theorem trigonometric_eq_solution_count :
  ∃ B : Finset ℤ, B.card = 250 ∧ ∀ x ∈ B, 2000 ≤ x ∧ x ≤ 3000 ∧ 
  2 * Real.sqrt 2 * Real.sin (Real.pi * x / 4)^3 = Real.sin (Real.pi / 4 * (1 + x)) :=
sorry

end trigonometric_eq_solution_count_l868_86856


namespace smallest_n_divisible_by_one_billion_l868_86878

-- Define the sequence parameters and the common ratio
def first_term : ℚ := 5 / 8
def second_term : ℚ := 50
def common_ratio : ℚ := second_term / first_term -- this is 80

-- Define the n-th term of the geometric sequence
noncomputable def nth_term (n : ℕ) : ℚ :=
  first_term * (common_ratio ^ (n - 1))

-- Define the target divisor (one billion)
def target_divisor : ℤ := 10 ^ 9

-- Prove that the smallest n such that nth_term n is divisible by 10^9 is 9
theorem smallest_n_divisible_by_one_billion :
  ∃ n : ℕ, nth_term n = (first_term * (common_ratio ^ (n - 1))) ∧ 
           (target_divisor : ℚ) ∣ nth_term n ∧
           n = 9 :=
by sorry

end smallest_n_divisible_by_one_billion_l868_86878


namespace sales_not_books_magazines_stationery_l868_86804

variable (books_sales : ℕ := 45)
variable (magazines_sales : ℕ := 30)
variable (stationery_sales : ℕ := 10)
variable (total_sales : ℕ := 100)

theorem sales_not_books_magazines_stationery : 
  books_sales + magazines_sales + stationery_sales < total_sales → 
  total_sales - (books_sales + magazines_sales + stationery_sales) = 15 :=
by
  sorry

end sales_not_books_magazines_stationery_l868_86804


namespace power_comparison_l868_86803

theorem power_comparison : (5 : ℕ) ^ 30 < (3 : ℕ) ^ 50 ∧ (3 : ℕ) ^ 50 < (4 : ℕ) ^ 40 := by
  sorry

end power_comparison_l868_86803


namespace point_in_fourth_quadrant_coords_l868_86887

theorem point_in_fourth_quadrant_coords 
  (P : ℝ × ℝ)
  (h1 : P.2 < 0)
  (h2 : abs P.2 = 2)
  (h3 : P.1 > 0)
  (h4 : abs P.1 = 5) :
  P = (5, -2) :=
sorry

end point_in_fourth_quadrant_coords_l868_86887


namespace Vasya_can_win_l868_86897

theorem Vasya_can_win 
  (a : ℕ → ℕ) -- initial sequence of natural numbers
  (x : ℕ) -- number chosen by Vasya
: ∃ (i : ℕ), ∀ (k : ℕ), ∃ (j : ℕ), (a j + k * x = 1) :=
by
  sorry

end Vasya_can_win_l868_86897


namespace certain_number_eq_14_l868_86834

theorem certain_number_eq_14 (x y : ℤ) (h1 : 4 * x + y = 34) (h2 : y^2 = 4) : 2 * x - y = 14 :=
by
  sorry

end certain_number_eq_14_l868_86834


namespace mod_computation_l868_86888

theorem mod_computation (a b n : ℕ) (h_modulus : n = 7) (h_a : a = 47) (h_b : b = 28) :
  (a^2023 - b^2023) % n = 5 :=
by
  sorry

end mod_computation_l868_86888


namespace weigh_80_grams_is_false_l868_86881

def XiaoGang_weight_grams : Nat := 80000  -- 80 kilograms in grams
def weight_claim : Nat := 80  -- 80 grams claim

theorem weigh_80_grams_is_false : weight_claim ≠ XiaoGang_weight_grams :=
by
  sorry

end weigh_80_grams_is_false_l868_86881


namespace students_not_picked_l868_86849

def total_students : ℕ := 58
def number_of_groups : ℕ := 8
def students_per_group : ℕ := 6

theorem students_not_picked :
  total_students - (number_of_groups * students_per_group) = 10 := by 
  sorry

end students_not_picked_l868_86849


namespace max_value_a_l868_86857

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a = 3 :=
sorry

end max_value_a_l868_86857


namespace batsman_average_increase_l868_86861

theorem batsman_average_increase :
  ∀ (A : ℝ), (10 * A + 110 = 11 * 60) → (60 - A = 5) :=
by
  intros A h
  -- Proof goes here
  sorry

end batsman_average_increase_l868_86861


namespace binary_addition_l868_86823

-- Define the binary numbers as natural numbers
def b1 : ℕ := 0b101  -- 101_2
def b2 : ℕ := 0b11   -- 11_2
def b3 : ℕ := 0b1100 -- 1100_2
def b4 : ℕ := 0b11101 -- 11101_2
def sum_b : ℕ := 0b110001 -- 110001_2

theorem binary_addition :
  b1 + b2 + b3 + b4 = sum_b := 
by
  sorry

end binary_addition_l868_86823


namespace find_original_price_l868_86832

-- Given conditions:
-- 1. 10% cashback
-- 2. $25 mail-in rebate
-- 3. Final cost is $110

def original_price (P : ℝ) (cashback : ℝ) (rebate : ℝ) (final_cost : ℝ) :=
  final_cost = P - (cashback * P + rebate)

theorem find_original_price :
  ∀ (P : ℝ), original_price P 0.10 25 110 → P = 150 :=
by
  sorry

end find_original_price_l868_86832


namespace customer_difference_l868_86874

theorem customer_difference (X Y Z : ℕ) (h1 : X - Y = 10) (h2 : 10 - Z = 4) : X - 4 = 10 :=
by sorry

end customer_difference_l868_86874


namespace distance_traveled_by_light_in_10_seconds_l868_86872

theorem distance_traveled_by_light_in_10_seconds :
  ∃ (a : ℝ) (n : ℕ), (300000 * 10 : ℝ) = a * 10 ^ n ∧ n = 6 :=
sorry

end distance_traveled_by_light_in_10_seconds_l868_86872


namespace monotonic_decreasing_interval_l868_86825

noncomputable def f (x : ℝ) := Real.log x + x^2 - 3 * x

theorem monotonic_decreasing_interval :
  (∃ I : Set ℝ, I = Set.Ioo (1 / 2 : ℝ) 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x < y → f x ≥ f y) := 
by
  sorry

end monotonic_decreasing_interval_l868_86825


namespace pq_ratio_at_0_l868_86880

noncomputable def p (x : ℝ) : ℝ := -3 * (x + 4) * x
noncomputable def q (x : ℝ) : ℝ := (x + 4) * (x - 3)

theorem pq_ratio_at_0 : (p 0) / (q 0) = 0 := by
  sorry

end pq_ratio_at_0_l868_86880


namespace tim_stacked_bales_today_l868_86867

theorem tim_stacked_bales_today (initial_bales : ℕ) (current_bales : ℕ) (initial_eq : initial_bales = 54) (current_eq : current_bales = 82) : 
  current_bales - initial_bales = 28 :=
by
  -- conditions
  have h1 : initial_bales = 54 := initial_eq
  have h2 : current_bales = 82 := current_eq
  sorry

end tim_stacked_bales_today_l868_86867


namespace find_number_l868_86873

theorem find_number (x : ℝ) (h : (1/4) * x = (1/5) * (x + 1) + 1) : x = 24 := 
sorry

end find_number_l868_86873


namespace enrique_commission_l868_86851

-- Define parameters for the problem
def suit_price : ℝ := 700
def suits_sold : ℝ := 2

def shirt_price : ℝ := 50
def shirts_sold : ℝ := 6

def loafer_price : ℝ := 150
def loafers_sold : ℝ := 2

def commission_rate : ℝ := 0.15

-- Calculate total sales for each category
def total_suit_sales : ℝ := suit_price * suits_sold
def total_shirt_sales : ℝ := shirt_price * shirts_sold
def total_loafer_sales : ℝ := loafer_price * loafers_sold

-- Calculate total sales
def total_sales : ℝ := total_suit_sales + total_shirt_sales + total_loafer_sales

-- Calculate commission
def commission : ℝ := commission_rate * total_sales

-- Proof statement that Enrique's commission is $300
theorem enrique_commission : commission = 300 := sorry

end enrique_commission_l868_86851


namespace belle_stickers_l868_86828

theorem belle_stickers (c_stickers : ℕ) (diff : ℕ) (b_stickers : ℕ) (h1 : c_stickers = 79) (h2 : diff = 18) (h3 : c_stickers = b_stickers - diff) : b_stickers = 97 := 
by
  sorry

end belle_stickers_l868_86828


namespace find_N_l868_86879

theorem find_N : (2 + 3 + 4) / 3 = (1990 + 1991 + 1992) / (N : ℚ) → N = 1991 := by
sorry

end find_N_l868_86879


namespace Jakes_height_is_20_l868_86809

-- Define the conditions
def Sara_width : ℤ := 12
def Sara_height : ℤ := 24
def Sara_depth : ℤ := 24
def Jake_width : ℤ := 16
def Jake_depth : ℤ := 18
def volume_difference : ℤ := 1152

-- Volume calculation
def Sara_volume : ℤ := Sara_width * Sara_height * Sara_depth

-- Prove Jake's height is 20 inches
theorem Jakes_height_is_20 :
  ∃ h : ℤ, (Sara_volume - (Jake_width * h * Jake_depth) = volume_difference) ∧ h = 20 :=
by
  sorry

end Jakes_height_is_20_l868_86809


namespace sequence_an_form_l868_86840

-- Definitions based on the given conditions
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ := (n : ℝ)^2 * a n
def a_1 : ℝ := 1

-- The conjecture we need to prove
theorem sequence_an_form (a : ℕ → ℝ) (h₁ : ∀ n ≥ 2, sum_first_n_terms a n = (n : ℝ)^2 * a n)
  (h₂ : a 1 = a_1) :
  ∀ n ≥ 2, a n = 2 / (n * (n + 1)) :=
by
  sorry

end sequence_an_form_l868_86840


namespace smallest_integer_consecutive_set_l868_86820

theorem smallest_integer_consecutive_set :
  ∃ m : ℤ, (m+3 < 3*m - 5) ∧ (∀ n : ℤ, (n+3 < 3*n - 5) → n ≥ m) ∧ m = 5 :=
by
  sorry

end smallest_integer_consecutive_set_l868_86820


namespace problem_a_lt_zero_b_lt_neg_one_l868_86848

theorem problem_a_lt_zero_b_lt_neg_one (a b : ℝ) (ha : a < 0) (hb : b < -1) : 
  ab > a ∧ a > ab^2 := 
by
  sorry

end problem_a_lt_zero_b_lt_neg_one_l868_86848


namespace geometric_sequence_sum_9000_l868_86827

noncomputable def sum_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_9000 (a r : ℝ) (h : r ≠ 1) 
  (h1 : sum_geometric_sequence a r 3000 = 1000)
  (h2 : sum_geometric_sequence a r 6000 = 1900) : 
  sum_geometric_sequence a r 9000 = 2710 :=
sorry

end geometric_sequence_sum_9000_l868_86827


namespace range_of_a_l868_86837

noncomputable def S : Set ℝ := {x | |x - 1| + |x + 2| > 5}
noncomputable def T (a : ℝ) : Set ℝ := {x | |x - a| ≤ 4}

theorem range_of_a (a : ℝ) : 
  (S ∪ T a) = Set.univ ↔ -2 ≤ a ∧ a ≤ 1 := 
by 
  sorry

end range_of_a_l868_86837


namespace fraction_of_rectangle_shaded_l868_86895

theorem fraction_of_rectangle_shaded
  (length : ℕ) (width : ℕ)
  (one_third_part : ℕ) (half_of_third : ℕ)
  (H1 : length = 10) (H2 : width = 15)
  (H3 : one_third_part = (1/3 : ℝ) * (length * width)) 
  (H4 : half_of_third = (1/2 : ℝ) * one_third_part) :
  (half_of_third / (length * width) = 1/6) :=
sorry

end fraction_of_rectangle_shaded_l868_86895


namespace compare_f_neg_x1_neg_x2_l868_86812

noncomputable def f : ℝ → ℝ := sorry

theorem compare_f_neg_x1_neg_x2 
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x)) 
  (h2 : ∀ x y : ℝ, 1 ≤ x → 1 ≤ y → x < y → f x < f y)
  (x1 x2 : ℝ)
  (hx1 : x1 < 0)
  (hx2 : x2 > 0)
  (hx1x2 : x1 + x2 < -2) :
  f (-x1) > f (-x2) :=
by sorry

end compare_f_neg_x1_neg_x2_l868_86812


namespace max_value_of_expression_l868_86893

noncomputable def f (x y : ℝ) := x * y^2 * (x^2 + x + 1) * (y^2 + y + 1)

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  ∃ m, m = 951625 / 256 ∧ ∀ a b : ℝ, a + b = 5 → f a b ≤ m :=
sorry

end max_value_of_expression_l868_86893


namespace loom_weaving_rate_l868_86846

theorem loom_weaving_rate :
  (119.04761904761905 : ℝ) > 0 ∧ (15 : ℝ) > 0 ∧ ∃ rate : ℝ, rate = 15 / 119.04761904761905 → rate = 0.126 :=
by sorry

end loom_weaving_rate_l868_86846


namespace find_numbers_l868_86808

theorem find_numbers (x y z t : ℕ) 
  (h1 : x + t = 37) 
  (h2 : y + z = 36) 
  (h3 : x + z = 2 * y) 
  (h4 : y * t = z * z) : 
  x = 12 ∧ y = 16 ∧ z = 20 ∧ t = 25 :=
by
  sorry

end find_numbers_l868_86808


namespace min_S_value_l868_86841

theorem min_S_value (n : ℕ) (h₁ : n ≥ 375) :
    let R := 3000
    let S := 9 * n - R
    let dice_sum (s : ℕ) := ∃ L : List ℕ, (∀ x ∈ L, 1 ≤ x ∧ x ≤ 8) ∧ L.sum = s
    dice_sum R ∧ S = 375 := 
by
  sorry

end min_S_value_l868_86841


namespace dinesh_loop_l868_86882

noncomputable def number_of_pentagons (n : ℕ) : ℕ :=
  if (20 * n) % 11 = 0 then 10 else 0

theorem dinesh_loop (n : ℕ) : number_of_pentagons n = 10 :=
by sorry

end dinesh_loop_l868_86882


namespace gcd_20244_46656_l868_86852

theorem gcd_20244_46656 : Nat.gcd 20244 46656 = 54 := by
  sorry

end gcd_20244_46656_l868_86852


namespace polynomial_real_root_inequality_l868_86863

theorem polynomial_real_root_inequality (a b : ℝ) : 
  (∃ x : ℝ, x^4 - a * x^3 + 2 * x^2 - b * x + 1 = 0) → (a^2 + b^2 ≥ 8) :=
sorry

end polynomial_real_root_inequality_l868_86863
