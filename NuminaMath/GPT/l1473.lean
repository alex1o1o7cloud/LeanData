import Mathlib

namespace NUMINAMATH_GPT_min_41x_2y_eq_nine_l1473_147345

noncomputable def min_value_41x_2y (x y : ℝ) : ℝ :=
  41*x + 2*y

theorem min_41x_2y_eq_nine (x y : ℝ) (h : ∀ n : ℕ, 0 < n →  n*x + (1/n)*y ≥ 1) :
  min_value_41x_2y x y = 9 :=
sorry

end NUMINAMATH_GPT_min_41x_2y_eq_nine_l1473_147345


namespace NUMINAMATH_GPT_cube_volume_from_surface_area_l1473_147325

theorem cube_volume_from_surface_area (s : ℕ) (h : 6 * s^2 = 864) : s^3 = 1728 :=
by {
  -- Proof begins here
  sorry
}

end NUMINAMATH_GPT_cube_volume_from_surface_area_l1473_147325


namespace NUMINAMATH_GPT_angles_cosine_sum_l1473_147314

theorem angles_cosine_sum (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 1)
  (h2 : Real.cos A + Real.cos B = 0) :
  12 * Real.cos (2 * A) + 4 * Real.cos (2 * B) = 8 :=
sorry

end NUMINAMATH_GPT_angles_cosine_sum_l1473_147314


namespace NUMINAMATH_GPT_percentage_decrease_revenue_l1473_147338

theorem percentage_decrease_revenue (old_revenue new_revenue : Float) (h_old : old_revenue = 69.0) (h_new : new_revenue = 42.0) : 
  (old_revenue - new_revenue) / old_revenue * 100 = 39.13 := by
  rw [h_old, h_new]
  norm_num
  sorry

end NUMINAMATH_GPT_percentage_decrease_revenue_l1473_147338


namespace NUMINAMATH_GPT_total_points_after_3_perfect_games_l1473_147392

def perfect_score := 21
def number_of_games := 3

theorem total_points_after_3_perfect_games : perfect_score * number_of_games = 63 := 
by
  sorry

end NUMINAMATH_GPT_total_points_after_3_perfect_games_l1473_147392


namespace NUMINAMATH_GPT_total_distance_combined_l1473_147321

/-- The conditions for the problem
Each car has 50 liters of fuel.
Car U has a fuel efficiency of 20 liters per 100 kilometers.
Car V has a fuel efficiency of 25 liters per 100 kilometers.
Car W has a fuel efficiency of 5 liters per 100 kilometers.
Car X has a fuel efficiency of 10 liters per 100 kilometers.
-/
theorem total_distance_combined (fuel_U fuel_V fuel_W fuel_X : ℕ) (eff_U eff_V eff_W eff_X : ℕ) (fuel : ℕ)
  (hU : fuel_U = 50) (hV : fuel_V = 50) (hW : fuel_W = 50) (hX : fuel_X = 50)
  (eU : eff_U = 20) (eV : eff_V = 25) (eW : eff_W = 5) (eX : eff_X = 10) :
  (fuel_U * 100 / eff_U) + (fuel_V * 100 / eff_V) + (fuel_W * 100 / eff_W) + (fuel_X * 100 / eff_X) = 1950 := by 
  sorry

end NUMINAMATH_GPT_total_distance_combined_l1473_147321


namespace NUMINAMATH_GPT_expand_polynomial_l1473_147355

variable (x : ℝ)

theorem expand_polynomial :
  (7 * x - 3) * (2 * x ^ 3 + 5 * x ^ 2 - 4) = 14 * x ^ 4 + 29 * x ^ 3 - 15 * x ^ 2 - 28 * x + 12 := by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l1473_147355


namespace NUMINAMATH_GPT_steve_book_sales_l1473_147332

theorem steve_book_sales
  (copies_price : ℝ)
  (agent_rate : ℝ)
  (total_earnings : ℝ)
  (net_per_copy : ℝ := copies_price * (1 - agent_rate))
  (total_copies_sold : ℝ := total_earnings / net_per_copy) :
  copies_price = 2 → agent_rate = 0.10 → total_earnings = 1620000 → total_copies_sold = 900000 :=
by
  intros
  sorry

end NUMINAMATH_GPT_steve_book_sales_l1473_147332


namespace NUMINAMATH_GPT_sum_rows_7_8_pascal_triangle_l1473_147323

theorem sum_rows_7_8_pascal_triangle : (2^7 + 2^8 = 384) :=
by
  sorry

end NUMINAMATH_GPT_sum_rows_7_8_pascal_triangle_l1473_147323


namespace NUMINAMATH_GPT_sum_of_distinct_prime_factors_l1473_147343

-- Definition of the expression
def expression : ℤ := 7^4 - 7^2

-- Statement of the theorem
theorem sum_of_distinct_prime_factors : 
  Nat.sum (List.eraseDup (Nat.factors expression.natAbs)) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_distinct_prime_factors_l1473_147343


namespace NUMINAMATH_GPT_thabo_books_l1473_147383

theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 280) : H = 55 :=
by
  sorry

end NUMINAMATH_GPT_thabo_books_l1473_147383


namespace NUMINAMATH_GPT_spherical_coordinates_neg_z_l1473_147379

theorem spherical_coordinates_neg_z (x y z : ℝ) (h₀ : ρ = 5) (h₁ : θ = 3 * Real.pi / 4) (h₂ : φ = Real.pi / 3)
  (hx : x = ρ * Real.sin φ * Real.cos θ) 
  (hy : y = ρ * Real.sin φ * Real.sin θ) 
  (hz : z = ρ * Real.cos φ) : 
  (ρ, θ, π - φ) = (5, 3 * Real.pi / 4, 2 * Real.pi / 3) :=
by
  sorry

end NUMINAMATH_GPT_spherical_coordinates_neg_z_l1473_147379


namespace NUMINAMATH_GPT_problem_solution_l1473_147301

-- We assume x and y are real numbers.
variables (x y : ℝ)

-- Our conditions
def condition1 : Prop := |x| - x + y = 6
def condition2 : Prop := x + |y| + y = 8

-- The goal is to prove that x + y = 30 under the given conditions.
theorem problem_solution (hx : condition1 x y) (hy : condition2 x y) : x + y = 30 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1473_147301


namespace NUMINAMATH_GPT_find_other_number_l1473_147374

def integers_three_and_four_sum (a b : ℤ) : Prop :=
  3 * a + 4 * b = 131

def one_of_the_numbers_is (x : ℤ) : Prop :=
  x = 17

theorem find_other_number (a b : ℤ) (h1 : integers_three_and_four_sum a b) (h2 : one_of_the_numbers_is a ∨ one_of_the_numbers_is b) :
  (a = 21 ∨ b = 21) :=
sorry

end NUMINAMATH_GPT_find_other_number_l1473_147374


namespace NUMINAMATH_GPT_positive_integer_solution_l1473_147327

theorem positive_integer_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≤ y ∧ y ≤ z) (h_eq : 5 * (x * y + y * z + z * x) = 4 * x * y * z) :
  (x = 2 ∧ y = 5 ∧ z = 10) ∨ (x = 2 ∧ y = 4 ∧ z = 20) :=
sorry

end NUMINAMATH_GPT_positive_integer_solution_l1473_147327


namespace NUMINAMATH_GPT_solve_equation_l1473_147341

theorem solve_equation (x : ℝ) : (x + 1) * (x - 3) = 5 ↔ (x = 4 ∨ x = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1473_147341


namespace NUMINAMATH_GPT_second_number_is_sixty_l1473_147389

theorem second_number_is_sixty (x : ℕ) (h_sum : 2 * x + x + (2 / 3) * x = 220) : x = 60 :=
by
  sorry

end NUMINAMATH_GPT_second_number_is_sixty_l1473_147389


namespace NUMINAMATH_GPT_trapezoid_angles_sum_l1473_147315

theorem trapezoid_angles_sum {α β γ δ : ℝ} (h : α + β + γ + δ = 360) (h1 : α = 60) (h2 : β = 120) :
  γ + δ = 180 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_angles_sum_l1473_147315


namespace NUMINAMATH_GPT_geometric_progression_nonzero_k_l1473_147363

theorem geometric_progression_nonzero_k (k : ℝ) : k ≠ 0 ↔ (40*k)^2 = (10*k) * (160*k) := by sorry

end NUMINAMATH_GPT_geometric_progression_nonzero_k_l1473_147363


namespace NUMINAMATH_GPT_mirella_orange_books_read_l1473_147347

-- Definitions based on the conditions in a)
def purpleBookPages : ℕ := 230
def orangeBookPages : ℕ := 510
def purpleBooksRead : ℕ := 5
def extraOrangePages : ℕ := 890

-- The total number of purple pages read
def purplePagesRead := purpleBooksRead * purpleBookPages

-- The number of orange books read
def orangeBooksRead (O : ℕ) := O * orangeBookPages

-- Statement to be proved
theorem mirella_orange_books_read (O : ℕ) :
  orangeBooksRead O = purplePagesRead + extraOrangePages → O = 4 :=
by
  sorry

end NUMINAMATH_GPT_mirella_orange_books_read_l1473_147347


namespace NUMINAMATH_GPT_weeks_to_work_l1473_147371

-- Definitions of conditions as per step a)
def isabelle_ticket_cost : ℕ := 20
def brother_ticket_cost : ℕ := 10
def brothers_total_savings : ℕ := 5
def isabelle_savings : ℕ := 5
def job_pay_per_week : ℕ := 3
def total_ticket_cost := isabelle_ticket_cost + 2 * brother_ticket_cost
def total_savings := isabelle_savings + brothers_total_savings
def remaining_amount := total_ticket_cost - total_savings

-- Theorem statement to match the question
theorem weeks_to_work : remaining_amount / job_pay_per_week = 10 := by
  -- Lean expects a proof here, replaced with sorry to skip it
  sorry

end NUMINAMATH_GPT_weeks_to_work_l1473_147371


namespace NUMINAMATH_GPT_triangle_angle_property_l1473_147339

variables {a b c : ℝ}
variables {A B C : ℝ} -- angles in triangle ABC

-- definition of a triangle side condition
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- condition given in the problem
def satisfies_condition (a b c : ℝ) : Prop := b^2 = a^2 + c^2

-- angle property based on given problem
def angle_B_is_right (A B C : ℝ) : Prop := B = 90

theorem triangle_angle_property (a b c : ℝ) (A B C : ℝ)
  (ht : triangle a b c) 
  (hc : satisfies_condition a b c) : 
  angle_B_is_right A B C :=
sorry

end NUMINAMATH_GPT_triangle_angle_property_l1473_147339


namespace NUMINAMATH_GPT_benjamin_collects_6_dozen_eggs_l1473_147375

theorem benjamin_collects_6_dozen_eggs (B : ℕ) (h : B + 3 * B + (B - 4) = 26) : B = 6 :=
by sorry

end NUMINAMATH_GPT_benjamin_collects_6_dozen_eggs_l1473_147375


namespace NUMINAMATH_GPT_parabola_equation_l1473_147386

theorem parabola_equation (x y : ℝ) :
  (∃p : ℝ, x = 4 ∧ y = -2 ∧ (x^2 = -2 * p * y ∨ y^2 = 2 * p * x) → (x^2 = -8 * y ∨ y^2 = x)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_equation_l1473_147386


namespace NUMINAMATH_GPT_xy_series_16_l1473_147336

noncomputable def series (x y : ℝ) : ℝ := ∑' n : ℕ, (n + 1) * (x * y)^n

theorem xy_series_16 (x y : ℝ) (h_series : series x y = 16) (h_abs : |x * y| < 1) :
  (x = 3 / 4 ∧ (y = 1 ∨ y = -1)) :=
sorry

end NUMINAMATH_GPT_xy_series_16_l1473_147336


namespace NUMINAMATH_GPT_remainder_of_5_pow_2023_mod_6_l1473_147304

theorem remainder_of_5_pow_2023_mod_6 : 5^2023 % 6 = 5 := 
by sorry

end NUMINAMATH_GPT_remainder_of_5_pow_2023_mod_6_l1473_147304


namespace NUMINAMATH_GPT_bread_baked_on_monday_l1473_147344

def loaves_wednesday : ℕ := 5
def loaves_thursday : ℕ := 7
def loaves_friday : ℕ := 10
def loaves_saturday : ℕ := 14
def loaves_sunday : ℕ := 19

def increment (n m : ℕ) : ℕ := m - n

theorem bread_baked_on_monday : 
  increment loaves_wednesday loaves_thursday = 2 →
  increment loaves_thursday loaves_friday = 3 →
  increment loaves_friday loaves_saturday = 4 →
  increment loaves_saturday loaves_sunday = 5 →
  loaves_sunday + 6 = 25 :=
by 
  sorry

end NUMINAMATH_GPT_bread_baked_on_monday_l1473_147344


namespace NUMINAMATH_GPT_Karen_tote_weight_l1473_147369

variable (B T F : ℝ)
variable (Papers Laptop : ℝ)

theorem Karen_tote_weight (h1: T = 2 * B)
                         (h2: F = 2 * T)
                         (h3: Papers = (1 / 6) * F)
                         (h4: Laptop = T + 2)
                         (h5: F = B + Laptop + Papers):
  T = 12 := 
sorry

end NUMINAMATH_GPT_Karen_tote_weight_l1473_147369


namespace NUMINAMATH_GPT_jade_transactions_l1473_147307

theorem jade_transactions :
  ∀ (transactions_mabel transactions_anthony transactions_cal transactions_jade : ℕ),
    transactions_mabel = 90 →
    transactions_anthony = transactions_mabel + transactions_mabel / 10 →
    transactions_cal = (transactions_anthony * 2) / 3 →
    transactions_jade = transactions_cal + 19 →
    transactions_jade = 85 :=
by
  intros transactions_mabel transactions_anthony transactions_cal transactions_jade
  intros h_mabel h_anthony h_cal h_jade
  sorry

end NUMINAMATH_GPT_jade_transactions_l1473_147307


namespace NUMINAMATH_GPT_total_candies_in_store_l1473_147377

-- Define the quantities of chocolates in each box
def box_chocolates_1 := 200
def box_chocolates_2 := 320
def box_chocolates_3 := 500
def box_chocolates_4 := 500
def box_chocolates_5 := 768
def box_chocolates_6 := 768

-- Define the quantities of candies in each tub
def tub_candies_1 := 1380
def tub_candies_2 := 1150
def tub_candies_3 := 1150
def tub_candies_4 := 1720

-- Sum of all chocolates and candies
def total_chocolates := box_chocolates_1 + box_chocolates_2 + box_chocolates_3 + box_chocolates_4 + box_chocolates_5 + box_chocolates_6
def total_candies := tub_candies_1 + tub_candies_2 + tub_candies_3 + tub_candies_4
def total_store_candies := total_chocolates + total_candies

theorem total_candies_in_store : total_store_candies = 8456 := by
  sorry

end NUMINAMATH_GPT_total_candies_in_store_l1473_147377


namespace NUMINAMATH_GPT_abs_sum_values_l1473_147320

theorem abs_sum_values (x y : ℚ) (h1 : |x| = 5) (h2 : |y| = 2) (h3 : |x - y| = x - y) : 
  x + y = 7 ∨ x + y = 3 := 
by
  sorry

end NUMINAMATH_GPT_abs_sum_values_l1473_147320


namespace NUMINAMATH_GPT_polynomial_properties_l1473_147395

noncomputable def p (x : ℕ) : ℕ := 2 * x^3 + x + 4

theorem polynomial_properties :
  p 1 = 7 ∧ p 10 = 2014 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_polynomial_properties_l1473_147395


namespace NUMINAMATH_GPT_desired_interest_percentage_l1473_147398

-- Definitions based on conditions
def face_value : ℝ := 20
def dividend_rate : ℝ := 0.09  -- 9% converted to fraction
def market_value : ℝ := 15

-- The main statement
theorem desired_interest_percentage : 
  ((dividend_rate * face_value) / market_value) * 100 = 12 :=
by
  sorry

end NUMINAMATH_GPT_desired_interest_percentage_l1473_147398


namespace NUMINAMATH_GPT_probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l1473_147394

noncomputable def p_a : ℝ := 0.18
noncomputable def p_b : ℝ := 0.5
noncomputable def p_b_given_a : ℝ := 0.2
noncomputable def p_c : ℝ := 0.3
noncomputable def p_c_given_a : ℝ := 0.4
noncomputable def p_c_given_b : ℝ := 0.6

noncomputable def p_a_and_b : ℝ := p_a * p_b_given_a
noncomputable def p_a_and_b_and_c : ℝ := p_c_given_a * p_a_and_b
noncomputable def p_a_and_b_given_c : ℝ := p_a_and_b_and_c / p_c
noncomputable def p_a_and_c_given_b : ℝ := p_a_and_b_and_c / p_b

theorem probability_a_and_b_and_c : p_a_and_b_and_c = 0.0144 := by
  sorry

theorem probability_a_and_b_given_c : p_a_and_b_given_c = 0.048 := by
  sorry

theorem probability_a_and_c_given_b : p_a_and_c_given_b = 0.0288 := by
  sorry

end NUMINAMATH_GPT_probability_a_and_b_and_c_probability_a_and_b_given_c_probability_a_and_c_given_b_l1473_147394


namespace NUMINAMATH_GPT_tan_theta_eq_l1473_147351

variables (k θ : ℝ)

-- Condition: k > 0
axiom k_pos : k > 0

-- Condition: k * cos θ = 12
axiom k_cos_theta : k * Real.cos θ = 12

-- Condition: k * sin θ = 5
axiom k_sin_theta : k * Real.sin θ = 5

-- To prove: tan θ = 5 / 12
theorem tan_theta_eq : Real.tan θ = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_tan_theta_eq_l1473_147351


namespace NUMINAMATH_GPT_sin_18_eq_l1473_147382

theorem sin_18_eq : ∃ x : Real, x = (Real.sin (Real.pi / 10)) ∧ x = (Real.sqrt 5 - 1) / 4 := by
  sorry

end NUMINAMATH_GPT_sin_18_eq_l1473_147382


namespace NUMINAMATH_GPT_smallest_n_l1473_147334

theorem smallest_n (n : ℕ) (h : 10 - n ≥ 0) : 
  (9 / 10) * (8 / 9) * (7 / 8) * (6 / 7) * (5 / 6) * (4 / 5) < 0.5 → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1473_147334


namespace NUMINAMATH_GPT_find_p_l1473_147306

theorem find_p (m n p : ℝ) 
  (h₁ : m = 5 * n + 5) 
  (h₂ : m + 2 = 5 * (n + p) + 5) :
  p = 2 / 5 :=
by sorry

end NUMINAMATH_GPT_find_p_l1473_147306


namespace NUMINAMATH_GPT_distance_MN_is_2R_l1473_147360

-- Definitions for the problem conditions
variable (R : ℝ) (A B C M N : ℝ) (alpha : ℝ)
variable (AC AB : ℝ)

-- Assumptions based on the problem statement
axiom circle_radius (r : ℝ) : r = R
axiom chord_length_AC (ch_AC : ℝ) : ch_AC = AC
axiom chord_length_AB (ch_AB : ℝ) : ch_AB = AB
axiom distance_M_to_AC (d_M_AC : ℝ) : d_M_AC = AC
axiom distance_N_to_AB (d_N_AB : ℝ) : d_N_AB = AB
axiom angle_BAC (ang_BAC : ℝ) : ang_BAC = alpha

-- To prove: the distance between M and N is 2R
theorem distance_MN_is_2R : |MN| = 2 * R := sorry

end NUMINAMATH_GPT_distance_MN_is_2R_l1473_147360


namespace NUMINAMATH_GPT_birds_problem_l1473_147359

theorem birds_problem 
  (x y z : ℕ) 
  (h1 : x + y + z = 30) 
  (h2 : (1 / 3 : ℚ) * x + (1 / 2 : ℚ) * y + 2 * z = 30) 
  : x = 9 ∧ y = 10 ∧ z = 11 := 
  by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_birds_problem_l1473_147359


namespace NUMINAMATH_GPT_min_value_of_xy_l1473_147308

theorem min_value_of_xy (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h : 2 * x + y + 6 = x * y) : 18 ≤ x * y :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_xy_l1473_147308


namespace NUMINAMATH_GPT_smallest_magnitude_z_theorem_l1473_147385

noncomputable def smallest_magnitude_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem smallest_magnitude_z_theorem : 
  ∃ z : ℂ, (Complex.abs (z - 9) + Complex.abs (z - 4 * Complex.I) = 15) ∧
  smallest_magnitude_z z = 36 / Real.sqrt 97 := 
sorry

end NUMINAMATH_GPT_smallest_magnitude_z_theorem_l1473_147385


namespace NUMINAMATH_GPT_table_tennis_basketball_teams_l1473_147390

theorem table_tennis_basketball_teams (X Y : ℕ)
  (h1 : X + Y = 50) 
  (h2 : 7 * Y = 3 * X)
  (h3 : 2 * (X - 8) = 3 * (Y + 8)) :
  X = 35 ∧ Y = 15 :=
by
  sorry

end NUMINAMATH_GPT_table_tennis_basketball_teams_l1473_147390


namespace NUMINAMATH_GPT_find_n_l1473_147326

theorem find_n {n : ℕ} (avg1 : ℕ) (avg2 : ℕ) (S : ℕ) :
  avg1 = 7 →
  avg2 = 6 →
  S = 7 * n →
  6 = (S - 11) / (n + 1) →
  n = 17 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_n_l1473_147326


namespace NUMINAMATH_GPT_new_average_age_l1473_147368

theorem new_average_age:
  ∀ (initial_avg_age new_persons_avg_age : ℝ) (initial_count new_persons_count : ℕ),
    initial_avg_age = 16 →
    new_persons_avg_age = 15 →
    initial_count = 20 →
    new_persons_count = 20 →
    (initial_avg_age * initial_count + new_persons_avg_age * new_persons_count) / 
    (initial_count + new_persons_count) = 15.5 :=
by
  intros initial_avg_age new_persons_avg_age initial_count new_persons_count
  intros h1 h2 h3 h4
  
  sorry

end NUMINAMATH_GPT_new_average_age_l1473_147368


namespace NUMINAMATH_GPT_triangle_third_side_l1473_147396

theorem triangle_third_side (AB AC AD : ℝ) (hAB : AB = 25) (hAC : AC = 30) (hAD : AD = 24) :
  ∃ BC : ℝ, (BC = 25 ∨ BC = 11) :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_l1473_147396


namespace NUMINAMATH_GPT_cost_per_yellow_ink_l1473_147358

def initial_amount : ℕ := 50
def cost_per_black_ink : ℕ := 11
def num_black_inks : ℕ := 2
def cost_per_red_ink : ℕ := 15
def num_red_inks : ℕ := 3
def additional_amount_needed : ℕ := 43
def num_yellow_inks : ℕ := 2

theorem cost_per_yellow_ink :
  let total_cost_needed := initial_amount + additional_amount_needed
  let total_black_ink_cost := cost_per_black_ink * num_black_inks
  let total_red_ink_cost := cost_per_red_ink * num_red_inks
  let total_non_yellow_cost := total_black_ink_cost + total_red_ink_cost
  let total_yellow_ink_cost := total_cost_needed - total_non_yellow_cost
  let cost_per_yellow_ink := total_yellow_ink_cost / num_yellow_inks
  cost_per_yellow_ink = 13 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_yellow_ink_l1473_147358


namespace NUMINAMATH_GPT_geometric_sequence_n_terms_l1473_147376

/-- In a geometric sequence with the first term a₁ and common ratio q,
the number of terms n for which the nth term aₙ has a given value -/
theorem geometric_sequence_n_terms (a₁ aₙ q : ℚ) (n : ℕ)
  (h1 : a₁ = 9/8)
  (h2 : aₙ = 1/3)
  (h3 : q = 2/3)
  (h_seq : aₙ = a₁ * q^(n-1)) :
  n = 4 := sorry

end NUMINAMATH_GPT_geometric_sequence_n_terms_l1473_147376


namespace NUMINAMATH_GPT_percent_increase_l1473_147302

theorem percent_increase (new_value old_value : ℕ) (h_new : new_value = 480) (h_old : old_value = 320) :
  ((new_value - old_value) / old_value) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_percent_increase_l1473_147302


namespace NUMINAMATH_GPT_determinant_is_zero_l1473_147364

-- Define the matrix
def my_matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, x + z, y - z],
    ![1, x + y + z, y - z],
    ![1, x + z, x + y]]

-- Define the property to prove
theorem determinant_is_zero (x y z : ℝ) :
  Matrix.det (my_matrix x y z) = 0 :=
by sorry

end NUMINAMATH_GPT_determinant_is_zero_l1473_147364


namespace NUMINAMATH_GPT_dolls_count_l1473_147381

theorem dolls_count (lisa_dolls : ℕ) (vera_dolls : ℕ) (sophie_dolls : ℕ) (aida_dolls : ℕ)
  (h1 : vera_dolls = 2 * lisa_dolls)
  (h2 : sophie_dolls = 2 * vera_dolls)
  (h3 : aida_dolls = 2 * sophie_dolls)
  (hl : lisa_dolls = 20) :
  aida_dolls + sophie_dolls + vera_dolls + lisa_dolls = 300 :=
by
  sorry

end NUMINAMATH_GPT_dolls_count_l1473_147381


namespace NUMINAMATH_GPT_circus_total_tickets_sold_l1473_147324

-- Definitions from the conditions
def revenue_total : ℕ := 2100
def lower_seat_tickets_sold : ℕ := 50
def price_lower : ℕ := 30
def price_upper : ℕ := 20

-- Definition derived from the conditions
def tickets_total (L U : ℕ) : ℕ := L + U

-- The theorem we need to prove
theorem circus_total_tickets_sold (L U : ℕ) (hL: L = lower_seat_tickets_sold)
    (h₁ : price_lower * L + price_upper * U = revenue_total) : 
    tickets_total L U = 80 :=
by
  sorry  -- Proof omitted

end NUMINAMATH_GPT_circus_total_tickets_sold_l1473_147324


namespace NUMINAMATH_GPT_total_seats_value_l1473_147322

noncomputable def students_per_bus : ℝ := 14.0
noncomputable def number_of_buses : ℝ := 2.0
noncomputable def total_seats : ℝ := students_per_bus * number_of_buses

theorem total_seats_value : total_seats = 28.0 :=
by
  sorry

end NUMINAMATH_GPT_total_seats_value_l1473_147322


namespace NUMINAMATH_GPT_minimum_g_value_l1473_147354

noncomputable def g (x : ℝ) := (9 * x^2 + 18 * x + 20) / (4 * (2 + x))

theorem minimum_g_value :
  ∀ x ≥ (1 : ℝ), g x = (47 / 16) := sorry

end NUMINAMATH_GPT_minimum_g_value_l1473_147354


namespace NUMINAMATH_GPT_fourth_guard_distance_l1473_147330

theorem fourth_guard_distance (d1 d2 d3 : ℕ) (d4 : ℕ) (h1 : d1 + d2 + d3 + d4 = 1000) (h2 : d1 + d2 + d3 = 850) : d4 = 150 :=
sorry

end NUMINAMATH_GPT_fourth_guard_distance_l1473_147330


namespace NUMINAMATH_GPT_shortest_tree_height_is_correct_l1473_147399

-- Definitions of the tree heights
def tallest_tree_height : ℕ := 150
def middle_tree_height : ℕ := (2 * tallest_tree_height) / 3
def shortest_tree_height : ℕ := middle_tree_height / 2

-- Theorem statement
theorem shortest_tree_height_is_correct :
  shortest_tree_height = 50 :=
by
  sorry

end NUMINAMATH_GPT_shortest_tree_height_is_correct_l1473_147399


namespace NUMINAMATH_GPT_remainder_31_l1473_147346

theorem remainder_31 (x : ℤ) (h : x % 62 = 7) : (x + 11) % 31 = 18 := by
  sorry

end NUMINAMATH_GPT_remainder_31_l1473_147346


namespace NUMINAMATH_GPT_solve_inequality_I_solve_inequality_II_l1473_147316

def f (x : ℝ) : ℝ := |x - 1| - |2 * x + 3|

theorem solve_inequality_I (x : ℝ) : f x > 2 ↔ -2 < x ∧ x < -4 / 3 :=
by sorry

theorem solve_inequality_II (a : ℝ) : ∀ x, f x ≤ (3 / 2) * a^2 - a ↔ a ≥ 5 / 3 :=
by sorry

end NUMINAMATH_GPT_solve_inequality_I_solve_inequality_II_l1473_147316


namespace NUMINAMATH_GPT_sum_of_solutions_l1473_147337

theorem sum_of_solutions : 
  (∀ x : ℝ, (x^2 - 5 * x + 3)^(x^2 - 6 * x + 4) = 1) → 
  (∃ s : ℝ, s = 16) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_solutions_l1473_147337


namespace NUMINAMATH_GPT_quarters_dimes_equivalence_l1473_147388

theorem quarters_dimes_equivalence (m : ℕ) (h : 25 * 30 + 10 * 20 = 25 * 15 + 10 * m) : m = 58 :=
by
  sorry

end NUMINAMATH_GPT_quarters_dimes_equivalence_l1473_147388


namespace NUMINAMATH_GPT_union_sets_a_l1473_147331

theorem union_sets_a (P S : Set ℝ) (a : ℝ) :
  P = {1, 5, 10} →
  S = {1, 3, a^2 + 1} →
  S ∪ P = {1, 3, 5, 10} →
  a = 2 ∨ a = -2 ∨ a = 3 ∨ a = -3 :=
by
  intros hP hS hUnion 
  sorry

end NUMINAMATH_GPT_union_sets_a_l1473_147331


namespace NUMINAMATH_GPT_find_f_seven_l1473_147397

theorem find_f_seven 
  (f : ℝ → ℝ)
  (hf : ∀ x : ℝ, f (2 * x + 3) = x^2 - 2 * x + 3) :
  f 7 = 3 := 
sorry

end NUMINAMATH_GPT_find_f_seven_l1473_147397


namespace NUMINAMATH_GPT_actual_price_of_good_l1473_147328

theorem actual_price_of_good (P : ℝ) (h : 0.684 * P = 6600) : P = 9649.12 :=
sorry

end NUMINAMATH_GPT_actual_price_of_good_l1473_147328


namespace NUMINAMATH_GPT_problem1_l1473_147329

theorem problem1 (a b : ℝ) (h1 : (a + b)^2 = 6) (h2 : (a - b)^2 = 2) : a^2 + b^2 = 4 ∧ a * b = 1 := 
by
  sorry

end NUMINAMATH_GPT_problem1_l1473_147329


namespace NUMINAMATH_GPT_area_triangle_CMB_eq_105_l1473_147342

noncomputable def area_of_triangle (C M B : ℝ × ℝ) : ℝ :=
  0.5 * (M.1 * B.2 - M.2 * B.1)

theorem area_triangle_CMB_eq_105 :
  let C : ℝ × ℝ := (0, 0)
  let M : ℝ × ℝ := (10, 0)
  let B : ℝ × ℝ := (10, 21)
  area_of_triangle C M B = 105 := by
  sorry

end NUMINAMATH_GPT_area_triangle_CMB_eq_105_l1473_147342


namespace NUMINAMATH_GPT_min_value_x_3y_min_value_x_3y_iff_l1473_147391

theorem min_value_x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y ≥ 25 :=
sorry

theorem min_value_x_3y_iff (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 3 * x + 4 * y = x * y) : x + 3 * y = 25 ↔ x = 10 ∧ y = 5 :=
sorry

end NUMINAMATH_GPT_min_value_x_3y_min_value_x_3y_iff_l1473_147391


namespace NUMINAMATH_GPT_greatest_x_lcm_l1473_147373

theorem greatest_x_lcm (x : ℕ) (hx : x > 0) :
  (∀ x, lcm (lcm x 15) (gcd x 21) = 105) ↔ x = 105 := 
sorry

end NUMINAMATH_GPT_greatest_x_lcm_l1473_147373


namespace NUMINAMATH_GPT_no_integer_roots_l1473_147372

theorem no_integer_roots (a b x : ℤ) : 2 * a * b * x^4 - a^2 * x^2 - b^2 - 1 ≠ 0 :=
sorry

end NUMINAMATH_GPT_no_integer_roots_l1473_147372


namespace NUMINAMATH_GPT_area_triangle_AEB_l1473_147333

theorem area_triangle_AEB :
  ∀ (A B C D F G E : Type)
    (AB AD BC CD : ℝ) 
    (AF BG : ℝ) 
    (triangle_AEB : ℝ),
  (AB = 7) →
  (BC = 4) →
  (CD = 7) →
  (AD = 4) →
  (DF = 2) →
  (GC = 1) →
  (triangle_AEB = 1/2 * 7 * (4 + 16/3)) →
  (triangle_AEB = 98 / 3) :=
by
  intros A B C D F G E AB AD BC CD AF BG triangle_AEB
  sorry

end NUMINAMATH_GPT_area_triangle_AEB_l1473_147333


namespace NUMINAMATH_GPT_total_market_cost_l1473_147387

-- Defining the variables for the problem
def pounds_peaches : Nat := 5 * 3
def pounds_apples : Nat := 4 * 3
def pounds_blueberries : Nat := 3 * 3

def cost_per_pound_peach := 2
def cost_per_pound_apple := 1
def cost_per_pound_blueberry := 1

-- Defining the total costs
def cost_peaches : Nat := pounds_peaches * cost_per_pound_peach
def cost_apples : Nat := pounds_apples * cost_per_pound_apple
def cost_blueberries : Nat := pounds_blueberries * cost_per_pound_blueberry

-- Total cost
def total_cost : Nat := cost_peaches + cost_apples + cost_blueberries

-- Theorem to prove the total cost is $51.00
theorem total_market_cost : total_cost = 51 := by
  sorry

end NUMINAMATH_GPT_total_market_cost_l1473_147387


namespace NUMINAMATH_GPT_min_fraction_value_l1473_147311

theorem min_fraction_value 
    (a : ℕ → ℝ) 
    (S : ℕ → ℝ) 
    (d : ℝ) 
    (n : ℕ) 
    (h1 : ∀ {n}, a n = 5 + (n - 1) * d)
    (h2 : (a 2) * (a 10) = (a 4 - 1)^2) 
    (h3 : S n = (n * (a 1 + a n)) / 2)
    (h4 : a 1 = 5)
    (h5 : d > 0) :
    2 * S n + n + 32 ≥ (20 / 3) * (a n + 1) := sorry

end NUMINAMATH_GPT_min_fraction_value_l1473_147311


namespace NUMINAMATH_GPT_proper_fraction_and_condition_l1473_147300

theorem proper_fraction_and_condition (a b : ℤ) (h1 : 1 < a) (h2 : b = 2 * a - 1) :
  0 < a ∧ a < b ∧ (a - 1 : ℚ) / (b - 1) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_proper_fraction_and_condition_l1473_147300


namespace NUMINAMATH_GPT_horse_bags_problem_l1473_147370

theorem horse_bags_problem (x y : ℤ) 
  (h1 : x - 1 = y + 1) : 
  x + 1 = 2 * (y - 1) :=
sorry

end NUMINAMATH_GPT_horse_bags_problem_l1473_147370


namespace NUMINAMATH_GPT_range_of_m_l1473_147353

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, 4 * x - m < 0 ∧ -1 ≤ x ∧ x ≤ 2) →
  (∃ x : ℝ, x^2 - x - 2 > 0) →
  (∀ x : ℝ, 4 * x - m < 0 → -1 ≤ x ∧ x ≤ 2) →
  m > 8 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1473_147353


namespace NUMINAMATH_GPT_rate_of_interest_is_4_l1473_147393

theorem rate_of_interest_is_4 (R : ℝ) : 
  ∀ P : ℝ, ∀ T : ℝ, P = 3000 → T = 5 → (P * R * T / 100 = P - 2400) → R = 4 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_is_4_l1473_147393


namespace NUMINAMATH_GPT_vacation_cost_eq_l1473_147318

theorem vacation_cost_eq (C : ℕ) (h : C / 3 - C / 5 = 50) : C = 375 :=
sorry

end NUMINAMATH_GPT_vacation_cost_eq_l1473_147318


namespace NUMINAMATH_GPT_class_committee_selection_l1473_147335

theorem class_committee_selection :
  let members := ["A", "B", "C", "D", "E"]
  let admissible_entertainment_candidates := ["C", "D", "E"]
  ∃ (entertainment : String) (study : String) (sports : String),
    entertainment ∈ admissible_entertainment_candidates ∧
    study ∈ members.erase entertainment ∧
    sports ∈ (members.erase entertainment).erase study ∧
    (3 * 4 * 3 = 36) :=
sorry

end NUMINAMATH_GPT_class_committee_selection_l1473_147335


namespace NUMINAMATH_GPT_cost_of_concessions_l1473_147317

theorem cost_of_concessions (total_cost : ℕ) (adult_ticket_cost : ℕ) (child_ticket_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  total_cost = 76 →
  adult_ticket_cost = 10 →
  child_ticket_cost = 7 →
  num_adults = 5 →
  num_children = 2 →
  total_cost - (num_adults * adult_ticket_cost + num_children * child_ticket_cost) = 12 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_cost_of_concessions_l1473_147317


namespace NUMINAMATH_GPT_tank_capacity_l1473_147303

theorem tank_capacity (C : ℝ) (h₁ : 3/4 * C + 7 = 7/8 * C) : C = 56 :=
by
  sorry

end NUMINAMATH_GPT_tank_capacity_l1473_147303


namespace NUMINAMATH_GPT_range_of_k_l1473_147384

theorem range_of_k (k : ℝ) : (∀ x : ℝ, |x + 1| + |x - 2| > k) → k > 3 := 
sorry

end NUMINAMATH_GPT_range_of_k_l1473_147384


namespace NUMINAMATH_GPT_sin_product_ge_one_l1473_147356

theorem sin_product_ge_one (x : ℝ) (n : ℤ) :
  (∀ α, |Real.sin α| ≤ 1) →
  ∀ x,
  (Real.sin x) * (Real.sin (1755 * x)) * (Real.sin (2011 * x)) ≥ 1 ↔
  ∃ n : ℤ, x = π / 2 + 2 * π * n := by {
    sorry
}

end NUMINAMATH_GPT_sin_product_ge_one_l1473_147356


namespace NUMINAMATH_GPT_translate_right_l1473_147350

-- Definition of the initial point and translation distance
def point_A : ℝ × ℝ := (2, -1)
def translation_distance : ℝ := 3

-- The proof statement
theorem translate_right (x_A y_A : ℝ) (d : ℝ) 
  (h1 : point_A = (x_A, y_A))
  (h2 : translation_distance = d) : 
  (x_A + d, y_A) = (5, -1) := 
sorry

end NUMINAMATH_GPT_translate_right_l1473_147350


namespace NUMINAMATH_GPT_enthalpy_change_correct_l1473_147312

def CC_bond_energy : ℝ := 347
def CO_bond_energy : ℝ := 358
def OH_bond_energy_CH2OH : ℝ := 463
def CO_double_bond_energy_COOH : ℝ := 745
def OH_bond_energy_COOH : ℝ := 467
def OO_double_bond_energy : ℝ := 498
def OH_bond_energy_H2O : ℝ := 467

def total_bond_energy_reactants : ℝ :=
  CC_bond_energy + CO_bond_energy + OH_bond_energy_CH2OH + 1.5 * OO_double_bond_energy

def total_bond_energy_products : ℝ :=
  CO_double_bond_energy_COOH + OH_bond_energy_COOH + OH_bond_energy_H2O

def deltaH : ℝ := total_bond_energy_reactants - total_bond_energy_products

theorem enthalpy_change_correct :
  deltaH = 236 := by
  sorry

end NUMINAMATH_GPT_enthalpy_change_correct_l1473_147312


namespace NUMINAMATH_GPT_calc_j_inverse_l1473_147352

noncomputable def i : ℂ := Complex.I  -- Equivalent to i^2 = -1 definition of complex imaginary unit
noncomputable def j : ℂ := i + 1      -- Definition of j

theorem calc_j_inverse :
  (j - j⁻¹)⁻¹ = (-3 * i + 1) / 5 :=
by 
  -- The statement here only needs to declare the equivalence, 
  -- without needing the proof
  sorry

end NUMINAMATH_GPT_calc_j_inverse_l1473_147352


namespace NUMINAMATH_GPT_divisible_by_27_l1473_147378

theorem divisible_by_27 (n : ℕ) : 27 ∣ (2^(5*n+1) + 5^(n+2)) :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_27_l1473_147378


namespace NUMINAMATH_GPT_greatest_power_of_2_divides_10_1004_minus_4_502_l1473_147349

theorem greatest_power_of_2_divides_10_1004_minus_4_502 :
  ∃ k, 10^1004 - 4^502 = 2^1007 * k :=
sorry

end NUMINAMATH_GPT_greatest_power_of_2_divides_10_1004_minus_4_502_l1473_147349


namespace NUMINAMATH_GPT_problem1_proof_problem2_proof_l1473_147340

noncomputable def problem1 : Real :=
  Real.sqrt 2 * Real.sqrt 3 + Real.sqrt 24

theorem problem1_proof : problem1 = 3 * Real.sqrt 6 :=
  sorry

noncomputable def problem2 : Real :=
  (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3)

theorem problem2_proof : problem2 = 6 :=
  sorry

end NUMINAMATH_GPT_problem1_proof_problem2_proof_l1473_147340


namespace NUMINAMATH_GPT_height_percentage_differences_l1473_147305

variable (B : ℝ) (A : ℝ) (R : ℝ)
variable (h1 : A = 1.25 * B) (h2 : R = 1.0625 * B)

theorem height_percentage_differences :
  (100 * (A - B) / B = 25) ∧
  (100 * (A - R) / A = 15) ∧
  (100 * (R - B) / B = 6.25) :=
by
  sorry

end NUMINAMATH_GPT_height_percentage_differences_l1473_147305


namespace NUMINAMATH_GPT_snow_probability_first_week_l1473_147310

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end NUMINAMATH_GPT_snow_probability_first_week_l1473_147310


namespace NUMINAMATH_GPT_old_hen_weight_unit_l1473_147367

theorem old_hen_weight_unit (w : ℕ) (units : String) (opt1 opt2 opt3 opt4 : String)
  (h_opt1 : opt1 = "grams") (h_opt2 : opt2 = "kilograms") (h_opt3 : opt3 = "tons") (h_opt4 : opt4 = "meters") (h_w : w = 2) : 
  (units = opt2) :=
sorry

end NUMINAMATH_GPT_old_hen_weight_unit_l1473_147367


namespace NUMINAMATH_GPT_sweets_ratio_l1473_147380

theorem sweets_ratio (total_sweets : ℕ) (mother_ratio : ℚ) (eldest_sweets second_sweets : ℕ)
  (h1 : total_sweets = 27) (h2 : mother_ratio = 1 / 3) (h3 : eldest_sweets = 8) (h4 : second_sweets = 6) :
  let mother_sweets := mother_ratio * total_sweets
  let remaining_sweets := total_sweets - mother_sweets
  let other_sweets := eldest_sweets + second_sweets
  let youngest_sweets := remaining_sweets - other_sweets
  youngest_sweets / eldest_sweets = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sweets_ratio_l1473_147380


namespace NUMINAMATH_GPT_sum_of_sequence_l1473_147348

noncomputable def f (n x : ℝ) : ℝ := (1 / (8 * n)) * x^2 + 2 * n * x

theorem sum_of_sequence (n : ℕ) (hn : n > 0) :
  let a : ℝ := 1 / (8 * n)
  let b : ℝ := 2 * n
  let f' := 2 * a * ((-n : ℝ )) + b 
  ∃ S : ℝ, S = (n - 1) * 2^(n + 1) + 2 := 
sorry

end NUMINAMATH_GPT_sum_of_sequence_l1473_147348


namespace NUMINAMATH_GPT_opposite_of_negative_fraction_l1473_147362

theorem opposite_of_negative_fraction :
  -(-1 / 2023) = (1 / 2023) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_negative_fraction_l1473_147362


namespace NUMINAMATH_GPT_surface_area_of_brick_l1473_147357

namespace SurfaceAreaProof

def brick_length : ℝ := 8
def brick_width : ℝ := 6
def brick_height : ℝ := 2

theorem surface_area_of_brick :
  2 * (brick_length * brick_width + brick_length * brick_height + brick_width * brick_height) = 152 :=
by
  sorry

end SurfaceAreaProof

end NUMINAMATH_GPT_surface_area_of_brick_l1473_147357


namespace NUMINAMATH_GPT_partition_exists_iff_l1473_147366

theorem partition_exists_iff (k : ℕ) :
  (∃ (A B : Finset ℕ), A ∪ B = Finset.range (1990 + k + 1) ∧ A ∩ B = ∅ ∧ 
  (A.sum id + 1990 * A.card = B.sum id + 1990 * B.card)) ↔ 
  (k % 4 = 3 ∨ (k % 4 = 0 ∧ k ≥ 92)) :=
by
  sorry

end NUMINAMATH_GPT_partition_exists_iff_l1473_147366


namespace NUMINAMATH_GPT_juanitas_dessert_cost_is_correct_l1473_147361

noncomputable def brownie_cost := 2.50
noncomputable def regular_scoop_cost := 1.00
noncomputable def premium_scoop_cost := 1.25
noncomputable def deluxe_scoop_cost := 1.50
noncomputable def syrup_cost := 0.50
noncomputable def nuts_cost := 1.50
noncomputable def whipped_cream_cost := 0.75
noncomputable def cherry_cost := 0.25
noncomputable def discount_tuesday := 0.10

noncomputable def total_cost_of_juanitas_dessert :=
    let discounted_brownie := brownie_cost * (1 - discount_tuesday)
    let ice_cream_cost := 2 * regular_scoop_cost + premium_scoop_cost
    let syrup_total := 2 * syrup_cost
    let additional_toppings := nuts_cost + whipped_cream_cost + cherry_cost
    discounted_brownie + ice_cream_cost + syrup_total + additional_toppings
   
theorem juanitas_dessert_cost_is_correct:
  total_cost_of_juanitas_dessert = 9.00 := by
  sorry

end NUMINAMATH_GPT_juanitas_dessert_cost_is_correct_l1473_147361


namespace NUMINAMATH_GPT_hyperbola_condition_l1473_147313

theorem hyperbola_condition (m n : ℝ) : (m < 0 ∧ 0 < n) → (∀ x y : ℝ, nx^2 + my^2 = 1 → (n * x^2 - m * y^2 > 0)) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l1473_147313


namespace NUMINAMATH_GPT_equivalence_of_complements_union_l1473_147309

open Set

-- Definitions as per the conditions
def U : Set ℝ := univ
def M : Set ℝ := { x | x ≥ 1 }
def N : Set ℝ := { x | 0 ≤ x ∧ x < 5 }
def complement_U (S : Set ℝ) : Set ℝ := U \ S

-- Mathematical statement to be proved
theorem equivalence_of_complements_union :
  (complement_U M ∪ complement_U N) = { x : ℝ | x < 1 ∨ x ≥ 5 } :=
by
  -- Non-trivial proof, hence skipped with sorry
  sorry

end NUMINAMATH_GPT_equivalence_of_complements_union_l1473_147309


namespace NUMINAMATH_GPT_rationalize_denominator_l1473_147365

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 2) = -(Real.sqrt 3 + 2) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l1473_147365


namespace NUMINAMATH_GPT_non_formable_triangle_sticks_l1473_147319

theorem non_formable_triangle_sticks 
  (sticks : Fin 8 → ℕ) 
  (h_no_triangle : ∀ (i j k : Fin 8), i < j → j < k → sticks i + sticks j ≤ sticks k) : 
  ∃ (max_length : ℕ), (max_length = sticks (Fin.mk 7 (by norm_num))) ∧ max_length = 21 := 
by 
  sorry

end NUMINAMATH_GPT_non_formable_triangle_sticks_l1473_147319
