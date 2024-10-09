import Mathlib

namespace sum_of_squares_l1163_116378

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 52) : x^2 + y^2 = 220 :=
sorry

end sum_of_squares_l1163_116378


namespace range_of_a_range_of_m_l1163_116347

def f (x : ℝ) : ℝ := |2 * x + 1| + |2 * x - 3|

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, f x < |1 - 2 * a|) ↔ a ∈ (Set.Iic (-3/2) ∪ Set.Ici (5/2)) := by sorry

theorem range_of_m (m : ℝ) : 
  (∃ t : ℝ, t^2 - 2 * Real.sqrt 6 * t + f m = 0) ↔ m ∈ (Set.Icc (-1) 2) := by sorry

end range_of_a_range_of_m_l1163_116347


namespace functions_same_function_C_functions_same_function_D_l1163_116377

theorem functions_same_function_C (x : ℝ) : (x^2) = (x^6)^(1/3) :=
by sorry

theorem functions_same_function_D (x : ℝ) : x = (x^3)^(1/3) :=
by sorry

end functions_same_function_C_functions_same_function_D_l1163_116377


namespace find_m_symmetry_l1163_116389

theorem find_m_symmetry (A B : ℝ × ℝ) (m : ℝ)
  (hA : A = (-3, m)) (hB : B = (3, 4)) (hy : A.2 = B.2) : m = 4 :=
sorry

end find_m_symmetry_l1163_116389


namespace bob_coloring_l1163_116399

/-
  Problem:
  Find the number of ways to color five points in {(x, y) | 1 ≤ x, y ≤ 5} blue 
  such that the distance between any two blue points is not an integer.
-/

def is_integer_distance (p1 p2 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let d := Int.gcd ((x2 - x1)^2 + (y2 - y1)^2)
  d ≠ 1

def valid_coloring (points : List (ℤ × ℤ)) : Prop :=
  points.length = 5 ∧ 
  (∀ (p1 p2 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ¬ is_integer_distance p1 p2)

theorem bob_coloring : ∃ (points : List (ℤ × ℤ)), valid_coloring points ∧ points.length = 80 :=
sorry

end bob_coloring_l1163_116399


namespace sum_of_coords_D_eq_eight_l1163_116303

def point := (ℝ × ℝ)

def N : point := (4, 6)
def C : point := (10, 2)

def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

theorem sum_of_coords_D_eq_eight
  (D : point)
  (h_midpoint : is_midpoint N C D) :
  D.1 + D.2 = 8 :=
by 
  sorry

end sum_of_coords_D_eq_eight_l1163_116303


namespace points_symmetric_about_y_eq_x_l1163_116332

theorem points_symmetric_about_y_eq_x (x y r : ℝ) :
  (x^2 + y^2 ≤ r^2 ∧ x + y > 0) →
  (∃ p q : ℝ, (q = p ∧ p + q = 0) ∨ (p = q ∧ q = -p)) :=
sorry

end points_symmetric_about_y_eq_x_l1163_116332


namespace angles_in_order_l1163_116391

-- α1, α2, α3 are real numbers representing the angles of inclination of lines
variable (α1 α2 α3 : ℝ)

-- Conditions given in the problem
axiom tan_α1 : Real.tan α1 = 1
axiom tan_α2 : Real.tan α2 = -1
axiom tan_α3 : Real.tan α3 = -2

-- Theorem to prove
theorem angles_in_order : α1 < α3 ∧ α3 < α2 := 
by
  sorry

end angles_in_order_l1163_116391


namespace num_integers_between_sqrt10_sqrt100_l1163_116371

theorem num_integers_between_sqrt10_sqrt100 : 
  ∃ n : ℕ, n = 7 ∧ ∀ x : ℤ, (⌊Real.sqrt 10⌋ + 1 <= x) ∧ (x <= ⌈Real.sqrt 100⌉ - 1) ↔ (4 <= x ∧ x <= 10) := 
by 
  sorry

end num_integers_between_sqrt10_sqrt100_l1163_116371


namespace donuts_per_box_l1163_116321

-- Define the conditions and the theorem
theorem donuts_per_box :
  (10 * 12 - 12 - 8) / 10 = 10 :=
by
  sorry

end donuts_per_box_l1163_116321


namespace sum_of_midpoint_coordinates_l1163_116322

theorem sum_of_midpoint_coordinates :
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  sum_of_coordinates = 17 :=
by
  let x1 := 3
  let y1 := 4
  let x2 := 9
  let y2 := 18
  let midpoint := ((x1 + x2) / 2, (y1 + y2) / 2)
  let sum_of_coordinates := midpoint.fst + midpoint.snd
  show sum_of_coordinates = 17
  sorry

end sum_of_midpoint_coordinates_l1163_116322


namespace min_value_proven_l1163_116331

open Real

noncomputable def min_value (x y : ℝ) (h1 : log x + log y = 1) : Prop :=
  2 * x + 5 * y ≥ 20 ∧ (2 * x + 5 * y = 20 ↔ 2 * x = 5 * y ∧ x * y = 10)

theorem min_value_proven (x y : ℝ) (h1 : log x + log y = 1) :
  min_value x y h1 :=
sorry

end min_value_proven_l1163_116331


namespace gcd_m_n_l1163_116315

noncomputable def m := 55555555
noncomputable def n := 111111111

theorem gcd_m_n : Int.gcd m n = 1 := by
  sorry

end gcd_m_n_l1163_116315


namespace aiyanna_more_cookies_than_alyssa_l1163_116309

-- Definitions of the conditions
def alyssa_cookies : ℕ := 129
def aiyanna_cookies : ℕ := 140

-- The proof problem statement
theorem aiyanna_more_cookies_than_alyssa : (aiyanna_cookies - alyssa_cookies) = 11 := sorry

end aiyanna_more_cookies_than_alyssa_l1163_116309


namespace reflected_ray_bisects_circle_circumference_l1163_116393

open Real

noncomputable def equation_of_line_reflected_ray : Prop :=
  ∃ (m b : ℝ), (m = 2 / (-3 + 1)) ∧ (b = (3/(-5 + 5)) + 1) ∧ ((-5, -3) = (-5, (-5*m + b))) ∧ ((1, 1) = (1, (1*m + b)))

theorem reflected_ray_bisects_circle_circumference :
  equation_of_line_reflected_ray ↔ ∃ a b c : ℝ, (a = 2) ∧ (b = -3) ∧ (c = 1) ∧ (a*x + b*y + c = 0) :=
by
  sorry

end reflected_ray_bisects_circle_circumference_l1163_116393


namespace number_of_students_l1163_116305

theorem number_of_students (left_pos right_pos total_pos : ℕ) 
  (h₁ : left_pos = 5) 
  (h₂ : right_pos = 3) 
  (h₃ : total_pos = left_pos - 1 + 1 + (right_pos - 1)) : 
  total_pos = 7 :=
by
  rw [h₁, h₂] at h₃
  simp at h₃
  exact h₃

end number_of_students_l1163_116305


namespace geometric_progression_condition_l1163_116343

theorem geometric_progression_condition {b : ℕ → ℝ} (b1_ne_b2 : b 1 ≠ b 2) (h : ∀ n, b (n + 2) = b n / b (n + 1)) :
  (∀ n, b (n+1) / b n = b 2 / b 1) ↔ b 1 = b 2^3 := sorry

end geometric_progression_condition_l1163_116343


namespace box_third_dimension_length_l1163_116339

noncomputable def box_height (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ) : ℝ :=
  let total_volume := num_cubes * cube_volume
  total_volume / (length * width)

theorem box_third_dimension_length (num_cubes : ℕ) (cube_volume : ℝ) (length : ℝ) (width : ℝ)
  (h_num_cubes : num_cubes = 24)
  (h_cube_volume : cube_volume = 27)
  (h_length : length = 8)
  (h_width : width = 12) :
  box_height num_cubes cube_volume length width = 6.75 :=
by {
  -- proof skipped
  sorry
}

end box_third_dimension_length_l1163_116339


namespace sum_of_coefficients_l1163_116316

theorem sum_of_coefficients :
  (∃ a b c d e : ℤ, 512 * x ^ 3 + 27 = a * x * (c * x ^ 2 + d * x + e) + b * (c * x ^ 2 + d * x + e)) →
  (a = 8 ∧ b = 3 ∧ c = 64 ∧ d = -24 ∧ e = 9) →
  a + b + c + d + e = 60 :=
by
  intro h1 h2
  sorry

end sum_of_coefficients_l1163_116316


namespace zero_a_and_b_l1163_116376

theorem zero_a_and_b (a b : ℝ) (h : a^2 + |b| = 0) : a = 0 ∧ b = 0 :=
by
  sorry

end zero_a_and_b_l1163_116376


namespace estimate_students_less_than_2_hours_probability_one_male_one_female_l1163_116334

-- Definitions from the conditions
def total_students_surveyed : ℕ := 40
def total_grade_ninth_students : ℕ := 400
def freq_0_1 : ℕ := 8
def freq_1_2 : ℕ := 20
def freq_2_3 : ℕ := 7
def freq_3_4 : ℕ := 5
def male_students_at_least_3_hours : ℕ := 2
def female_students_at_least_3_hours : ℕ := 3

-- Question 1 proof statement
theorem estimate_students_less_than_2_hours :
  total_grade_ninth_students * (freq_0_1 + freq_1_2) / total_students_surveyed = 280 :=
by sorry

-- Question 2 proof statement
theorem probability_one_male_one_female :
  (male_students_at_least_3_hours * female_students_at_least_3_hours) / (Nat.choose 5 2) = (3 / 5) :=
by sorry

end estimate_students_less_than_2_hours_probability_one_male_one_female_l1163_116334


namespace sin_sum_angle_eq_sqrt15_div5_l1163_116355

variable {x : Real}
variable (h1 : 0 < x ∧ x < Real.pi) (h2 : Real.sin (2 * x) = 1 / 5)

theorem sin_sum_angle_eq_sqrt15_div5 : Real.sin (Real.pi / 4 + x) = Real.sqrt 15 / 5 := by
  -- The proof is omitted as instructed.
  sorry

end sin_sum_angle_eq_sqrt15_div5_l1163_116355


namespace effective_percentage_change_l1163_116329

def original_price (P : ℝ) : ℝ := P
def annual_sale_discount (P : ℝ) : ℝ := 0.70 * P
def clearance_event_discount (P : ℝ) : ℝ := 0.80 * (annual_sale_discount P)
def sales_tax (P : ℝ) : ℝ := 1.10 * (clearance_event_discount P)

theorem effective_percentage_change (P : ℝ) :
  (sales_tax P) = 0.616 * P := by
  sorry

end effective_percentage_change_l1163_116329


namespace jump_rope_difference_l1163_116381

noncomputable def cindy_jump_time : ℕ := 12
noncomputable def betsy_jump_time : ℕ := cindy_jump_time / 2
noncomputable def tina_jump_time : ℕ := 3 * betsy_jump_time

theorem jump_rope_difference : tina_jump_time - cindy_jump_time = 6 :=
by
  -- proof steps would go here
  sorry

end jump_rope_difference_l1163_116381


namespace sin_half_angle_product_lt_quarter_l1163_116317

theorem sin_half_angle_product_lt_quarter (A B C : ℝ) (h : A + B + C = 180) :
    Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := 
    sorry

end sin_half_angle_product_lt_quarter_l1163_116317


namespace power_function_value_l1163_116311

noncomputable def f (x : ℝ) : ℝ := x ^ (1/2)

-- Given the condition
axiom passes_through_point : f 3 = Real.sqrt 3

-- Prove that f(9) = 3
theorem power_function_value : f 9 = 3 := by
  sorry

end power_function_value_l1163_116311


namespace emily_final_lives_l1163_116374

/-- Initial number of lives Emily had. --/
def initialLives : ℕ := 42

/-- Number of lives Emily lost in the hard part of the game. --/
def livesLost : ℕ := 25

/-- Number of lives Emily gained in the next level. --/
def livesGained : ℕ := 24

/-- Final number of lives Emily should have after the changes. --/
def finalLives : ℕ := (initialLives - livesLost) + livesGained

theorem emily_final_lives : finalLives = 41 := by
  /-
  Proof is omitted as per instructions.
  Prove that the final number of lives Emily has is 41.
  -/
  sorry

end emily_final_lives_l1163_116374


namespace mrs_hilt_more_l1163_116380

-- Define the values of the pennies, nickels, and dimes.
def value_penny : ℝ := 0.01
def value_nickel : ℝ := 0.05
def value_dime : ℝ := 0.10

-- Define the count of coins Mrs. Hilt has.
def mrs_hilt_pennies : ℕ := 2
def mrs_hilt_nickels : ℕ := 2
def mrs_hilt_dimes : ℕ := 2

-- Define the count of coins Jacob has.
def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1

-- Calculate the total amount of money Mrs. Hilt has.
def mrs_hilt_total : ℝ :=
  mrs_hilt_pennies * value_penny
  + mrs_hilt_nickels * value_nickel
  + mrs_hilt_dimes * value_dime

-- Calculate the total amount of money Jacob has.
def jacob_total : ℝ :=
  jacob_pennies * value_penny
  + jacob_nickels * value_nickel
  + jacob_dimes * value_dime

-- Prove that Mrs. Hilt has $0.13 more than Jacob.
theorem mrs_hilt_more : mrs_hilt_total - jacob_total = 0.13 := by
  sorry

end mrs_hilt_more_l1163_116380


namespace function_intersects_y_axis_at_0_neg4_l1163_116392

theorem function_intersects_y_axis_at_0_neg4 :
  (∃ x y : ℝ, y = 4 * x - 4 ∧ x = 0 ∧ y = -4) :=
sorry

end function_intersects_y_axis_at_0_neg4_l1163_116392


namespace inscribed_circle_radius_in_quarter_circle_l1163_116304

theorem inscribed_circle_radius_in_quarter_circle (R r : ℝ) (hR : R = 4) :
  (r + r * Real.sqrt 2 = R) ↔ r = 4 * Real.sqrt 2 - 4 := by
  sorry

end inscribed_circle_radius_in_quarter_circle_l1163_116304


namespace original_price_of_sarees_l1163_116324

theorem original_price_of_sarees 
  (P : ℝ) 
  (h1 : 0.72 * P = 144) : 
  P = 200 := 
sorry

end original_price_of_sarees_l1163_116324


namespace value_of_f_at_112_5_l1163_116312

noncomputable def f : ℝ → ℝ := sorry

lemma f_even_func (x : ℝ) : f x = f (-x) := sorry
lemma f_func_eq (x : ℝ) : f x + f (x + 1) = 4 := sorry
lemma f_interval : ∀ x, -3 ≤ x ∧ x ≤ -2 → f x = 4 * x + 12 := sorry

theorem value_of_f_at_112_5 : f 112.5 = 2 := sorry

end value_of_f_at_112_5_l1163_116312


namespace p_interval_satisfies_inequality_l1163_116363

theorem p_interval_satisfies_inequality :
  ∀ (p q : ℝ), 0 ≤ p ∧ p < 2.232 ∧ q > 0 ∧ p + q ≠ 0 →
    (4 * (p * q ^ 2 + p ^ 2 * q + 4 * q ^ 2 + 4 * p * q)) / (p + q) > 5 * p ^ 2 * q :=
by sorry

end p_interval_satisfies_inequality_l1163_116363


namespace zero_in_interval_l1163_116368

noncomputable def f (x : ℝ) : ℝ := Real.log (3 * x / 2) - 2 / x

theorem zero_in_interval :
  (Real.log (3 / 2) - 2 < 0) ∧ (Real.log 3 - 2 / 3 > 0) →
  ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  -- conditions from the problem statement
  intros h
  -- proving the result
  sorry

end zero_in_interval_l1163_116368


namespace largest_number_by_replacement_l1163_116340

theorem largest_number_by_replacement 
  (n : ℝ) (n_1 n_3 n_6 n_8 : ℝ)
  (h : n = -0.3168)
  (h1 : n_1 = -0.3468)
  (h3 : n_3 = -0.4168)
  (h6 : n_6 = -0.3148)
  (h8 : n_8 = -0.3164)
  : n_6 > n_1 ∧ n_6 > n_3 ∧ n_6 > n_8 := 
by {
  -- Proof goes here
  sorry
}

end largest_number_by_replacement_l1163_116340


namespace shaniqua_styles_count_l1163_116397

variable (S : ℕ)

def shaniqua_haircuts (haircuts : ℕ) : ℕ := 12 * haircuts
def shaniqua_styles (styles : ℕ) : ℕ := 25 * styles

theorem shaniqua_styles_count (total_money haircuts : ℕ) (styles : ℕ) :
  total_money = shaniqua_haircuts haircuts + shaniqua_styles styles → haircuts = 8 → total_money = 221 → S = 5 :=
by
  sorry

end shaniqua_styles_count_l1163_116397


namespace num_teams_is_seventeen_l1163_116325

-- Each team faces all other teams 10 times and there are 1360 games in total.
def total_teams (n : ℕ) : Prop := 1360 = (n * (n - 1) * 10) / 2

theorem num_teams_is_seventeen : ∃ n : ℕ, total_teams n ∧ n = 17 := 
by 
  sorry

end num_teams_is_seventeen_l1163_116325


namespace county_population_percentage_l1163_116367

theorem county_population_percentage 
    (percent_less_than_20000 : ℝ)
    (percent_20000_to_49999 : ℝ) 
    (h1 : percent_less_than_20000 = 35) 
    (h2 : percent_20000_to_49999 = 40) : 
    percent_less_than_20000 + percent_20000_to_49999 = 75 := 
by
  sorry

end county_population_percentage_l1163_116367


namespace quadratic_transformation_l1163_116346

theorem quadratic_transformation (a b c : ℝ) (h : a * x^2 + b * x + c = 5 * (x + 2)^2 - 7) :
  ∃ (n m g : ℝ), 2 * a * x^2 + 2 * b * x + 2 * c = n * (x - g)^2 + m ∧ g = -2 :=
by
  sorry

end quadratic_transformation_l1163_116346


namespace Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l1163_116345

-- Define \( S_n \) following the given conditions
def S (n : ℕ) : ℕ :=
  let a := 2^n + 1 -- first term
  let b := 2^(n+1) - 1 -- last term
  let m := b - a + 1 -- number of terms
  (m * (a + b)) / 2 -- sum of the arithmetic series

-- The first part: Prove that \( S_n \) is divisible by 3 for all positive integers \( n \)
theorem Sn_divisible_by_3 (n : ℕ) (hn : 0 < n) : 3 ∣ S n := sorry

-- The second part: Prove that \( S_n \) is divisible by 9 if and only if \( n \) is even
theorem Sn_divisible_by_9_iff_even (n : ℕ) (hn : 0 < n) : 9 ∣ S n ↔ Even n := sorry

end Sn_divisible_by_3_Sn_divisible_by_9_iff_even_l1163_116345


namespace units_digit_17_pow_2023_l1163_116390

theorem units_digit_17_pow_2023 : (17^2023 % 10) = 3 := sorry

end units_digit_17_pow_2023_l1163_116390


namespace total_amount_received_l1163_116338
noncomputable section

variables (B : ℕ) (H1 : (1 / 3 : ℝ) * B = 50)
theorem total_amount_received (H2 : (2 / 3 : ℝ) * B = 100) (H3 : ∀ (x : ℕ), x = 5): 
  100 * 5 = 500 := 
by
  sorry

end total_amount_received_l1163_116338


namespace Suzanne_runs_5_kilometers_l1163_116386

theorem Suzanne_runs_5_kilometers 
  (a : ℕ) 
  (r : ℕ) 
  (total_donation : ℕ) 
  (n : ℕ)
  (h1 : a = 10) 
  (h2 : r = 2) 
  (h3 : total_donation = 310) 
  (h4 : total_donation = a * (1 - r^n) / (1 - r)) 
  : n = 5 :=
by
  sorry

end Suzanne_runs_5_kilometers_l1163_116386


namespace solve_equation_l1163_116301

noncomputable def equation (x : ℝ) : ℝ :=
(13 * x - x^2) / (x + 1) * (x + (13 - x) / (x + 1))

theorem solve_equation :
  equation 1 = 42 ∧ equation 6 = 42 ∧ equation (3 + Real.sqrt 2) = 42 ∧ equation (3 - Real.sqrt 2) = 42 :=
by
  sorry

end solve_equation_l1163_116301


namespace todd_money_after_repay_l1163_116333

-- Definitions as conditions
def borrowed_amount : ℤ := 100
def repay_amount : ℤ := 110
def ingredients_cost : ℤ := 75
def snow_cones_sold : ℤ := 200
def price_per_snow_cone : ℚ := 0.75

-- Function to calculate money left after transactions
def money_left_after_repay (borrowed : ℤ) (repay : ℤ) (cost : ℤ) (sold : ℤ) (price : ℚ) : ℚ :=
  let left_after_cost := borrowed - cost
  let earnings := sold * price
  let total_before_repay := left_after_cost + earnings
  total_before_repay - repay

-- The theorem stating the problem
theorem todd_money_after_repay :
  money_left_after_repay borrowed_amount repay_amount ingredients_cost snow_cones_sold price_per_snow_cone = 65 := 
by
  -- This is a placeholder for the actual proof
  sorry

end todd_money_after_repay_l1163_116333


namespace circle_radius_tangent_l1163_116375

theorem circle_radius_tangent (A B O M X : Type) (AB AM MB r : ℝ)
  (hL1 : AB = 2) (hL2 : AM = 1) (hL3 : MB = 1) (hMX : MX = 1/2)
  (hTangent1 : OX = 1/2 + r) (hTangent2 : OM = 1 - r)
  (hPythagorean : OM^2 + MX^2 = OX^2) :
  r = 1/3 :=
by
  sorry

end circle_radius_tangent_l1163_116375


namespace percentage_of_part_whole_l1163_116383

theorem percentage_of_part_whole (part whole : ℝ) (h_part : part = 75) (h_whole : whole = 125) : 
  (part / whole) * 100 = 60 :=
by
  rw [h_part, h_whole]
  -- Simplification steps would follow, but we substitute in the placeholders
  sorry

end percentage_of_part_whole_l1163_116383


namespace average_height_of_trees_l1163_116356

theorem average_height_of_trees :
  ∃ (h : ℕ → ℕ), (h 2 = 12) ∧ (∀ i, h i = 2 * h (i+1) ∨ h i = h (i+1) / 2) ∧ (h 1 * h 2 * h 3 * h 4 * h 5 * h 6 = 4608) →
  (h 1 + h 2 + h 3 + h 4 + h 5 + h 6) / 6 = 21 :=
sorry

end average_height_of_trees_l1163_116356


namespace log_m_n_iff_m_minus_1_n_minus_1_l1163_116328

theorem log_m_n_iff_m_minus_1_n_minus_1 (m n : ℝ) (h1 : m > 0) (h2 : m ≠ 1) (h3 : n > 0) :
  (Real.log n / Real.log m < 0) ↔ ((m - 1) * (n - 1) < 0) :=
sorry

end log_m_n_iff_m_minus_1_n_minus_1_l1163_116328


namespace sum_of_areas_of_tangent_circles_l1163_116388

theorem sum_of_areas_of_tangent_circles
  (r s t : ℝ)
  (h1 : r + s = 6)
  (h2 : s + t = 8)
  (h3 : r + t = 10) :
  π * (r^2 + s^2 + t^2) = 56 * π :=
by
  sorry

end sum_of_areas_of_tangent_circles_l1163_116388


namespace area_of_garden_l1163_116319

variable (w l : ℕ)
variable (h1 : l = 3 * w) 
variable (h2 : 2 * (l + w) = 72)

theorem area_of_garden : l * w = 243 := by
  sorry

end area_of_garden_l1163_116319


namespace ticket_distribution_count_l1163_116394

-- Defining the parameters
def tickets : Finset ℕ := {1, 2, 3, 4, 5, 6}
def people : ℕ := 4

-- Condition: Each person gets at least 1 ticket and at most 2 tickets, consecutive if 2.
def valid_distribution (dist: Finset (Finset ℕ)) :=
  dist.card = 4 ∧ ∀ s ∈ dist, s.card >= 1 ∧ s.card <= 2 ∧ (s.card = 1 ∨ (∃ x, s = {x, x+1}))

-- Question: Prove that there are 144 valid ways to distribute the tickets.
theorem ticket_distribution_count :
  ∃ dist: Finset (Finset ℕ), valid_distribution dist ∧ dist.card = 144 :=
by {
  sorry -- Proof is omitted as per instructions.
}

-- This statement checks distribution of 6 tickets to 4 people with given constraints is precisely 144

end ticket_distribution_count_l1163_116394


namespace simplification_evaluation_l1163_116372

noncomputable def simplify_and_evaluate (x : ℤ) : ℚ :=
  (1 - 1 / (x - 1)) * ((x - 1) / ((x - 2) * (x - 2)))

theorem simplification_evaluation (x : ℤ) (h1 : x > 0) (h2 : 3 - x ≥ 0) : 
  simplify_and_evaluate x = 1 :=
by
  have h3 : x = 3 := sorry
  rw [simplify_and_evaluate, h3]
  simp [h3]
  sorry

end simplification_evaluation_l1163_116372


namespace tom_average_score_increase_l1163_116353

def initial_scores : List ℕ := [72, 78, 81]
def fourth_exam_score : ℕ := 90

theorem tom_average_score_increase :
  let initial_avg := (initial_scores.sum : ℚ) / (initial_scores.length : ℚ)
  let total_score_after_fourth := initial_scores.sum + fourth_exam_score
  let new_avg := (total_score_after_fourth : ℚ) / (initial_scores.length + 1 : ℚ)
  new_avg - initial_avg = 3.25 := by 
  -- Proof goes here
  sorry

end tom_average_score_increase_l1163_116353


namespace drum_capacity_ratio_l1163_116323

variable {C_X C_Y : ℝ}

theorem drum_capacity_ratio (h1 : C_X / 2 + C_Y / 2 = 3 * C_Y / 4) : C_Y / C_X = 2 :=
by
  have h2: C_X / 2 = C_Y / 4 := by
    sorry
  have h3: C_X = C_Y / 2 := by
    sorry
  rw [h3]
  have h4: C_Y / (C_Y / 2) = 2 := by
    sorry
  exact h4

end drum_capacity_ratio_l1163_116323


namespace find_number_of_students_l1163_116344

theorem find_number_of_students (N : ℕ) (h1 : T = 80 * N) (h2 : (T - 350) / (N - 5) = 90) 
: N = 10 := 
by 
  -- Proof steps would go here. Omitted as per the instruction.
  sorry

end find_number_of_students_l1163_116344


namespace fraction_exponentiation_l1163_116395

theorem fraction_exponentiation : (3 / 4) ^ 5 = 243 / 1024 := by
  sorry

end fraction_exponentiation_l1163_116395


namespace largest_three_digit_multiple_of_6_sum_15_l1163_116366

-- Statement of the problem in Lean
theorem largest_three_digit_multiple_of_6_sum_15 : 
  ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 6 = 0 ∧ (Nat.digits 10 n).sum = 15 ∧ 
  ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ m % 6 = 0 ∧ (Nat.digits 10 m).sum = 15 → m ≤ n :=
by
  sorry -- proof not required

end largest_three_digit_multiple_of_6_sum_15_l1163_116366


namespace num_ordered_pairs_l1163_116326

theorem num_ordered_pairs (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x * y = 4410) : 
  ∃ (n : ℕ), n = 36 :=
sorry

end num_ordered_pairs_l1163_116326


namespace total_marbles_l1163_116349

theorem total_marbles
  (R B Y : ℕ)  -- Red, Blue, and Yellow marbles as natural numbers
  (h_ratio : 2 * (R + B + Y) = 9 * Y)  -- The ratio condition translated
  (h_yellow : Y = 36)  -- The number of yellow marbles condition
  : R + B + Y = 81 :=  -- Statement that the total number of marbles is 81
sorry

end total_marbles_l1163_116349


namespace geometric_progression_ineq_l1163_116365

variable (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ)

-- Condition: \(b_n\) is an increasing positive geometric progression
-- \( q > 1 \) because the progression is increasing
variable (q_pos : q > 1) 

-- Recursive definitions for the geometric progression
variable (geom_b₂ : b₂ = b₁ * q)
variable (geom_b₃ : b₃ = b₁ * q^2)
variable (geom_b₄ : b₄ = b₁ * q^3)
variable (geom_b₅ : b₅ = b₁ * q^4)
variable (geom_b₆ : b₆ = b₁ * q^5)

-- Given condition from the problem
variable (condition : b₄ + b₃ - b₂ - b₁ = 5)

-- Statement to prove
theorem geometric_progression_ineq (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) 
  (q_pos : q > 1) 
  (geom_b₂ : b₂ = b₁ * q)
  (geom_b₃ : b₃ = b₁ * q^2)
  (geom_b₄ : b₄ = b₁ * q^3)
  (geom_b₅ : b₅ = b₁ * q^4)
  (geom_b₆ : b₆ = b₁ * q^5)
  (condition : b₃ + b₄ - b₂ - b₁ = 5) : b₆ + b₅ ≥ 20 := by
    sorry

end geometric_progression_ineq_l1163_116365


namespace prove_students_second_and_third_l1163_116362

namespace MonicaClasses

def Monica := 
  let classes_per_day := 6
  let students_first_class := 20
  let students_fourth_class := students_first_class / 2
  let students_fifth_class := 28
  let students_sixth_class := 28
  let total_students := 136
  let known_students := students_first_class + students_fourth_class + students_fifth_class + students_sixth_class
  let students_second_and_third := total_students - known_students
  students_second_and_third = 50

theorem prove_students_second_and_third : Monica :=
  by
    sorry

end MonicaClasses

end prove_students_second_and_third_l1163_116362


namespace cube_side_length_l1163_116318

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = (6 * n^3) / 3) : n = 3 :=
sorry

end cube_side_length_l1163_116318


namespace regular_pay_limit_l1163_116302

theorem regular_pay_limit (x : ℝ) : 3 * x + 6 * 13 = 198 → x = 40 :=
by
  intro h
  -- proof skipped
  sorry

end regular_pay_limit_l1163_116302


namespace find_d_l1163_116358

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x - 3

theorem find_d (c d : ℝ) (h : ∀ x, f (g x c) c = 15 * x + d) : d = -12 :=
by
  have h1 : ∀ x, f (g x c) c = 5 * (c * x - 3) + c := by intros; simp [f, g]
  have h2 : ∀ x, 5 * (c * x - 3) + c = 5 * c * x + c - 15 := by intros; ring
  specialize h 0
  rw [h1, h2] at h
  sorry

end find_d_l1163_116358


namespace abcd_mod_7_zero_l1163_116382

theorem abcd_mod_7_zero
  (a b c d : ℕ)
  (h1 : a + 2 * b + 3 * c + 4 * d ≡ 1 [MOD 7])
  (h2 : 2 * a + 3 * b + c + 2 * d ≡ 5 [MOD 7])
  (h3 : 3 * a + b + 2 * c + 3 * d ≡ 3 [MOD 7])
  (h4 : 4 * a + 2 * b + d + c ≡ 2 [MOD 7])
  (ha : a < 7) (hb : b < 7) (hc : c < 7) (hd : d < 7) :
  (a * b * c * d) % 7 = 0 :=
by sorry

end abcd_mod_7_zero_l1163_116382


namespace find_a_range_l1163_116341

theorem find_a_range (a : ℝ) (x : ℝ) (h1 : a * x < 6) (h2 : (3 * x - 6 * a) / 2 > a / 3 - 1) :
  a ≤ -3 / 2 :=
sorry

end find_a_range_l1163_116341


namespace boy_run_time_l1163_116335

section
variables {d1 d2 d3 d4 : ℝ} -- distances
variables {v1 v2 v3 v4 : ℝ} -- velocities
variables {t : ℝ} -- time

-- Define conditions
def distances_and_velocities (d1 d2 d3 d4 v1 v2 v3 v4 : ℝ) :=
  d1 = 25 ∧ d2 = 30 ∧ d3 = 40 ∧ d4 = 35 ∧
  v1 = 3.33 ∧ v2 = 3.33 ∧ v3 = 2.78 ∧ v4 = 2.22

-- Problem statement
theorem boy_run_time
  (h : distances_and_velocities d1 d2 d3 d4 v1 v2 v3 v4) :
  t = (d1 / v1) + (d2 / v2) + (d3 / v3) + (d4 / v4) := 
sorry
end

end boy_run_time_l1163_116335


namespace sum_first_10_terms_l1163_116342

noncomputable def a (n : ℕ) := 1 / (4 * (n + 1) ^ 2 - 1)

theorem sum_first_10_terms : (Finset.range 10).sum a = 10 / 21 :=
by
  sorry

end sum_first_10_terms_l1163_116342


namespace min_value_x2_sub_xy_add_y2_l1163_116330

/-- Given positive real numbers x and y such that x^2 + xy + 3y^2 = 10, 
prove that the minimum value of x^2 - xy + y^2 is 2. -/
theorem min_value_x2_sub_xy_add_y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + x * y + 3 * y^2 = 10) : 
  ∃ (value : ℝ), value = x^2 - x * y + y^2 ∧ value = 2 := 
by 
  sorry

end min_value_x2_sub_xy_add_y2_l1163_116330


namespace maximum_value_expression_l1163_116320

theorem maximum_value_expression (a b : ℝ) (h : a^2 + b^2 = 9) : 
  ∃ x, x = 5 ∧ ∀ y, y = ab - b + a → y ≤ x :=
by
  sorry

end maximum_value_expression_l1163_116320


namespace james_writes_pages_per_hour_l1163_116396

theorem james_writes_pages_per_hour (hours_per_night : ℕ) (days_per_week : ℕ) (weeks : ℕ) (total_pages : ℕ) (total_hours : ℕ) :
  hours_per_night = 3 → 
  days_per_week = 7 → 
  weeks = 7 → 
  total_pages = 735 → 
  total_hours = 147 → 
  total_hours = hours_per_night * days_per_week * weeks → 
  total_pages / total_hours = 5 :=
by sorry

end james_writes_pages_per_hour_l1163_116396


namespace smaller_area_l1163_116354

theorem smaller_area (A B : ℝ) (total_area : A + B = 1800) (diff_condition : B - A = (A + B) / 6) :
  A = 750 := 
by
  sorry

end smaller_area_l1163_116354


namespace usual_time_eight_l1163_116360

/-- Define the parameters used in the problem -/
def usual_speed (S : ℝ) : ℝ := S
def usual_time (T : ℝ) : ℝ := T
def reduced_speed (S : ℝ) := 0.25 * S
def reduced_time (T : ℝ) := T + 24

/-- The main theorem that we need to prove -/
theorem usual_time_eight (S T : ℝ) 
  (h1 : usual_speed S = S)
  (h2 : usual_time T = T)
  (h3 : reduced_speed S = 0.25 * S)
  (h4 : reduced_time T = T + 24)
  (h5 : S / (0.25 * S) = (T + 24) / T) : T = 8 :=
by 
  sorry -- Proof omitted for brevity. Refers to the solution steps.


end usual_time_eight_l1163_116360


namespace no_snow_three_days_l1163_116369

noncomputable def probability_no_snow_first_two_days : ℚ := 1 - 2/3
noncomputable def probability_no_snow_third_day : ℚ := 1 - 3/5

theorem no_snow_three_days : 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_first_two_days) * 
  (probability_no_snow_third_day) = 2/45 :=
by
  sorry

end no_snow_three_days_l1163_116369


namespace jodi_walks_days_l1163_116337

section
variables {d : ℕ} -- d is the number of days Jodi walks per week

theorem jodi_walks_days (h : 1 * d + 2 * d + 3 * d + 4 * d = 60) : d = 6 := by
  sorry

end

end jodi_walks_days_l1163_116337


namespace percentage_increase_sale_l1163_116313

theorem percentage_increase_sale (P S : ℝ) (hP : P > 0) (hS : S > 0) 
  (h1 : ∀ P S : ℝ, 0.7 * P * S * (1 + X / 100) = 1.26 * P * S) : 
  X = 80 := 
by
  sorry

end percentage_increase_sale_l1163_116313


namespace speed_of_second_train_l1163_116373

-- Definitions of given conditions
def length_first_train : ℝ := 60 
def length_second_train : ℝ := 280 
def speed_first_train : ℝ := 30 
def time_clear : ℝ := 16.998640108791296 

-- The Lean statement for the proof problem
theorem speed_of_second_train : 
  let relative_distance_km := (length_first_train + length_second_train) / 1000
  let time_clear_hr := time_clear / 3600
  (speed_first_train + (relative_distance_km / time_clear_hr)) = 72.00588235294118 → 
  ∃ V : ℝ, V = 42.00588235294118 :=
by 
  -- Placeholder for the proof
  sorry

end speed_of_second_train_l1163_116373


namespace radius_of_circle_l1163_116364

-- Definitions based on conditions
def center_in_first_quadrant (C : ℝ × ℝ) : Prop :=
  C.1 > 0 ∧ C.2 > 0

def intersects_x_axis (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = Real.sqrt ((C.1 - 1)^2 + (C.2)^2) ∧ r = Real.sqrt ((C.1 - 3)^2 + (C.2)^2)

def tangent_to_line (C : ℝ × ℝ) (r : ℝ) : Prop :=
  r = abs (C.1 - C.2 + 1) / Real.sqrt 2

-- Main statement
theorem radius_of_circle (C : ℝ × ℝ) (r : ℝ) 
  (h1 : center_in_first_quadrant C)
  (h2 : intersects_x_axis C r)
  (h3 : tangent_to_line C r) : 
  r = Real.sqrt 2 := 
sorry

end radius_of_circle_l1163_116364


namespace chairs_left_l1163_116351

-- Conditions
def red_chairs : Nat := 4
def yellow_chairs : Nat := 2 * red_chairs
def blue_chairs : Nat := yellow_chairs - 2
def lisa_borrows : Nat := 3

-- Theorem
theorem chairs_left (chairs_left : Nat) : chairs_left = red_chairs + yellow_chairs + blue_chairs - lisa_borrows :=
by
  sorry

end chairs_left_l1163_116351


namespace lending_rate_is_7_percent_l1163_116306

-- Conditions
def principal : ℝ := 5000
def borrowing_rate : ℝ := 0.04  -- 4% p.a. simple interest
def time : ℕ := 2  -- 2 years
def gain_per_year : ℝ := 150

-- Proof of the final statement
theorem lending_rate_is_7_percent :
  let borrowing_interest := principal * borrowing_rate * time / 100
  let interest_per_year := borrowing_interest / 2
  let total_interest_earned_per_year := interest_per_year + gain_per_year
  (total_interest_earned_per_year * 100) / principal = 7 :=
by
  sorry

end lending_rate_is_7_percent_l1163_116306


namespace simplify_div_l1163_116310

theorem simplify_div : (27 * 10^12) / (9 * 10^4) = 3 * 10^8 := 
by
  sorry

end simplify_div_l1163_116310


namespace travel_cost_from_B_to_C_l1163_116336

noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

noncomputable def travel_cost_by_air (distance : ℝ) (booking_fee : ℝ) (per_km_cost : ℝ) : ℝ :=
  booking_fee + (distance * per_km_cost)

theorem travel_cost_from_B_to_C :
  let AC := 4000
  let AB := 4500
  let BC := Real.sqrt (AB^2 - AC^2)
  let booking_fee := 120
  let per_km_cost := 0.12
  travel_cost_by_air BC booking_fee per_km_cost = 367.39 := by
  sorry

end travel_cost_from_B_to_C_l1163_116336


namespace area_increase_is_50_l1163_116352

def length := 13
def width := 10
def length_new := length + 2
def width_new := width + 2
def area_original := length * width
def area_new := length_new * width_new
def area_increase := area_new - area_original

theorem area_increase_is_50 : area_increase = 50 :=
by
  -- Here we will include the steps to prove the theorem if required
  sorry

end area_increase_is_50_l1163_116352


namespace circle_equation_center_xaxis_radius_2_l1163_116327

theorem circle_equation_center_xaxis_radius_2 (a x y : ℝ) :
  (0:ℝ) < 2 ∧ (a - 1)^2 + 2^2 = 4 -> (x - 1)^2 + y^2 = 4 :=
by
  sorry

end circle_equation_center_xaxis_radius_2_l1163_116327


namespace perpendicular_vecs_l1163_116379

open Real

def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (3, 4)
def lambda := 1 / 2

theorem perpendicular_vecs : 
  let vec_diff := (vec_a.1 - lambda * vec_b.1, vec_a.2 - lambda * vec_b.2)
  let vec_opp := (vec_b.1 - vec_a.1, vec_b.2 - vec_a.2)
  (vec_diff.1 * vec_opp.1 + vec_diff.2 * vec_opp.2) = 0 := 
by 
  let vec_diff := (vec_a.1 - lambda * vec_b.1, vec_a.2 - lambda * vec_b.2)
  let vec_opp := (vec_b.1 - vec_a.1, vec_b.2 - vec_a.2)
  show (vec_diff.1 * vec_opp.1 + vec_diff.2 * vec_opp.2) = 0
  sorry

end perpendicular_vecs_l1163_116379


namespace quadratic_function_a_value_l1163_116300

theorem quadratic_function_a_value (a : ℝ) (h₁ : a ≠ 1) :
  (∀ x : ℝ, ∃ c₀ c₁ c₂ : ℝ, (a-1) * x^(a^2 + 1) + 2 * x + 3 = c₂ * x^2 + c₁ * x + c₀) → a = -1 :=
by
  sorry

end quadratic_function_a_value_l1163_116300


namespace evaluate_using_horners_method_l1163_116370

def f (x : ℝ) : ℝ := 3 * x^6 + 12 * x^5 + 8 * x^4 - 3.5 * x^3 + 7.2 * x^2 + 5 * x - 13

theorem evaluate_using_horners_method :
  f 6 = 243168.2 :=
by
  sorry

end evaluate_using_horners_method_l1163_116370


namespace least_positive_linear_combination_24_18_l1163_116350

theorem least_positive_linear_combination_24_18 (x y : ℤ) :
  ∃ (a : ℤ) (b : ℤ), 24 * a + 18 * b = 6 :=
by
  use 1
  use -1
  sorry

end least_positive_linear_combination_24_18_l1163_116350


namespace symmetric_circle_eq_l1163_116361

theorem symmetric_circle_eq :
  (∃ x y : ℝ, (x + 2)^2 + (y - 1)^2 = 4) :=
by
  sorry

end symmetric_circle_eq_l1163_116361


namespace professional_pay_per_hour_l1163_116308

def professionals : ℕ := 2
def hours_per_day : ℕ := 6
def days : ℕ := 7
def total_cost : ℕ := 1260

theorem professional_pay_per_hour :
  (total_cost / (professionals * hours_per_day * days) = 15) :=
by
  sorry

end professional_pay_per_hour_l1163_116308


namespace fg_minus_gf_l1163_116314

-- Definitions provided by the conditions
def f (x : ℝ) : ℝ := 4 * x + 8
def g (x : ℝ) : ℝ := 2 * x - 3

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -17 := 
  sorry

end fg_minus_gf_l1163_116314


namespace rhombus_diagonal_l1163_116359

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 14) (h2 : area = 126) (h3 : area = (d1 * d2) / 2) : d2 = 18 := 
by
  -- h1, h2, and h3 are the conditions
  sorry

end rhombus_diagonal_l1163_116359


namespace Beto_can_determine_xy_l1163_116398

theorem Beto_can_determine_xy (m n : ℤ) :
  (∃ k t : ℤ, 0 < t ∧ m = 2 * k + 1 ∧ n = 2 * t * (2 * k + 1)) ↔ 
  (∀ x y : ℝ, (∃ a b : ℝ, a ≠ b ∧ x = a ∧ y = b) →
    ∃ xy_val : ℝ, (x^m + y^m = xy_val) ∧ (x^n + y^n = xy_val)) := 
sorry

end Beto_can_determine_xy_l1163_116398


namespace geometric_sequence_term_l1163_116357

theorem geometric_sequence_term 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_seq : ∀ n, a (n+1) = a n * q)
  (h_a2 : a 2 = 8) 
  (h_a5 : a 5 = 64) : 
  a 3 = 16 := 
by 
  sorry

end geometric_sequence_term_l1163_116357


namespace find_correct_day_l1163_116385

def tomorrow_is_not_September (d : String) : Prop :=
  d ≠ "September"

def in_a_week_is_September (d : String) : Prop :=
  d = "September"

def day_after_tomorrow_is_not_Wednesday (d : String) : Prop :=
  d ≠ "Wednesday"

theorem find_correct_day :
    ((∀ d, tomorrow_is_not_September d) ∧ 
    (∀ d, in_a_week_is_September d) ∧ 
    (∀ d, day_after_tomorrow_is_not_Wednesday d)) → 
    "Wednesday, August 25" = "Wednesday, August 25" :=
by
sorry

end find_correct_day_l1163_116385


namespace solve_eq_simplify_expression_l1163_116384

-- Part 1: Prove the solution to the given equation

theorem solve_eq (x : ℚ) : (1 / (x - 1) + 1 = 3 / (2 * x - 2)) → x = 3 / 2 :=
sorry

-- Part 2: Prove the simplified value of the given expression when x=1/2

theorem simplify_expression : (x = 1/2) →
  ((x^2 / (1 + x) - x) / ((x^2 - 1) / (x^2 + 2 * x + 1)) = 1) :=
sorry

end solve_eq_simplify_expression_l1163_116384


namespace remainder_division_l1163_116387

theorem remainder_division : ∃ (r : ℕ), 271 = 30 * 9 + r ∧ r = 1 :=
by
  -- Details of the proof would be filled here
  sorry

end remainder_division_l1163_116387


namespace translation_correct_l1163_116348

-- Define the points in the Cartesian coordinate system
structure Point where
  x : ℤ
  y : ℤ

-- Given points A and B
def A : Point := { x := -1, y := 0 }
def B : Point := { x := 1, y := 2 }

-- Translated point A' (A₁)
def A₁ : Point := { x := 2, y := -1 }

-- Define the translation applied to a point
def translate (p : Point) (v : Point) : Point :=
  { x := p.x + v.x, y := p.y + v.y }

-- Calculate the translation vector from A to A'
def translationVector : Point :=
  { x := A₁.x - A.x, y := A₁.y - A.y }

-- Define the expected point B' (B₁)
def B₁ : Point := { x := 4, y := 1 }

-- Theorem statement
theorem translation_correct :
  translate B translationVector = B₁ :=
by
  -- proof goes here
  sorry

end translation_correct_l1163_116348


namespace minimum_value_of_f_l1163_116307

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  ∃ a > 2, (∀ x > 2, f x ≥ f a) ∧ a = 3 := by
sorry

end minimum_value_of_f_l1163_116307
