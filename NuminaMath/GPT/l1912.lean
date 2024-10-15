import Mathlib

namespace NUMINAMATH_GPT_x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l1912_191232

theorem x_sq_plus_3x_minus_2_ge_zero (x : ℝ) (h : x ≥ 1) : x^2 + 3 * x - 2 ≥ 0 :=
sorry

theorem neg_x_sq_plus_3x_minus_2_lt_zero (x : ℝ) (h : x < 1) : x^2 + 3 * x - 2 < 0 :=
sorry

end NUMINAMATH_GPT_x_sq_plus_3x_minus_2_ge_zero_neg_x_sq_plus_3x_minus_2_lt_zero_l1912_191232


namespace NUMINAMATH_GPT_highest_number_paper_l1912_191287

theorem highest_number_paper
  (n : ℕ)
  (P : ℝ)
  (hP : P = 0.010309278350515464)
  (hP_formula : 1 / n = P) :
  n = 97 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_highest_number_paper_l1912_191287


namespace NUMINAMATH_GPT_area_difference_of_square_screens_l1912_191265

theorem area_difference_of_square_screens (d1 d2 : ℝ) (A1 A2 : ℝ) 
  (h1 : d1 = 18) (h2 : d2 = 16) 
  (hA1 : A1 = d1^2 / 2) (hA2 : A2 = d2^2 / 2) : 
  A1 - A2 = 34 := by
  sorry

end NUMINAMATH_GPT_area_difference_of_square_screens_l1912_191265


namespace NUMINAMATH_GPT_find_interest_rate_l1912_191253

theorem find_interest_rate
  (P : ℝ) (t : ℕ) (I : ℝ)
  (hP : P = 3000)
  (ht : t = 5)
  (hI : I = 750) :
  ∃ r : ℝ, I = P * r * t / 100 ∧ r = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_interest_rate_l1912_191253


namespace NUMINAMATH_GPT_prop1_prop3_l1912_191258

def custom_op (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

theorem prop1 (x y : ℝ) : custom_op x y = custom_op y x :=
by sorry

theorem prop3 (x : ℝ) : custom_op (x + 1) (x - 1) = custom_op x x - 1 :=
by sorry

end NUMINAMATH_GPT_prop1_prop3_l1912_191258


namespace NUMINAMATH_GPT_find_d_l1912_191291

theorem find_d (
  x : ℝ
) (
  h1 : 3 * x + 8 = 5
) (
  d : ℝ
) (
  h2 : d * x - 15 = -7
) : d = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_d_l1912_191291


namespace NUMINAMATH_GPT_remainder_of_k_divided_by_7_l1912_191241

theorem remainder_of_k_divided_by_7 :
  ∃ k < 42, k % 5 = 2 ∧ k % 6 = 5 ∧ k % 7 = 3 :=
by {
  -- The proof is supplied here
  sorry
}

end NUMINAMATH_GPT_remainder_of_k_divided_by_7_l1912_191241


namespace NUMINAMATH_GPT_cost_price_of_ball_l1912_191249

theorem cost_price_of_ball (x : ℝ) (h : 17 * x - 5 * x = 720) : x = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_price_of_ball_l1912_191249


namespace NUMINAMATH_GPT_sin_theta_add_pi_over_3_l1912_191256

theorem sin_theta_add_pi_over_3 (θ : ℝ) (h : Real.cos (π / 6 - θ) = 2 / 3) : 
  Real.sin (θ + π / 3) = 2 / 3 :=
sorry

end NUMINAMATH_GPT_sin_theta_add_pi_over_3_l1912_191256


namespace NUMINAMATH_GPT_calculate_length_QR_l1912_191294

noncomputable def length_QR (A : ℝ) (h : ℝ) (PQ : ℝ) (RS : ℝ) : ℝ :=
  21 - 0.5 * (Real.sqrt (PQ ^ 2 - h ^ 2) + Real.sqrt (RS ^ 2 - h ^ 2))

theorem calculate_length_QR :
  length_QR 210 10 12 21 = 21 - 0.5 * (Real.sqrt 44 + Real.sqrt 341) :=
by
  sorry

end NUMINAMATH_GPT_calculate_length_QR_l1912_191294


namespace NUMINAMATH_GPT_dot_product_of_a_and_b_l1912_191284

-- Define the vectors
def a : ℝ × ℝ := (1, -3)
def b : ℝ × ℝ := (3, 7)

-- Define the dot product function
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

-- State the theorem
theorem dot_product_of_a_and_b : dot_product a b = -18 := by
  sorry

end NUMINAMATH_GPT_dot_product_of_a_and_b_l1912_191284


namespace NUMINAMATH_GPT_wheat_grains_approximation_l1912_191200

theorem wheat_grains_approximation :
  let total_grains : ℕ := 1536
  let wheat_per_sample : ℕ := 28
  let sample_size : ℕ := 224
  let wheat_estimate : ℕ := total_grains * wheat_per_sample / sample_size
  wheat_estimate = 169 := by
  sorry

end NUMINAMATH_GPT_wheat_grains_approximation_l1912_191200


namespace NUMINAMATH_GPT_percentage_increase_l1912_191203

theorem percentage_increase (d : ℝ) (v_current v_reduce v_increase t_reduce t_increase : ℝ) (h1 : d = 96)
  (h2 : v_current = 8) (h3 : v_reduce = v_current - 4) (h4 : t_reduce = d / v_reduce) 
  (h5 : t_increase = d / v_increase) (h6 : t_reduce = t_current + 16) (h7 : t_increase = t_current - 16) :
  (v_increase - v_current) / v_current * 100 = 50 := 
sorry

end NUMINAMATH_GPT_percentage_increase_l1912_191203


namespace NUMINAMATH_GPT_number_of_squares_is_five_l1912_191248

-- A function that computes the number of squares obtained after the described operations on a piece of paper.
def folded_and_cut_number_of_squares (initial_shape : Type) (folds : ℕ) (cuts : ℕ) : ℕ :=
  -- sorry is used here as a placeholder for the actual implementation
  sorry

-- The main theorem stating that after two folds and two cuts, we obtain five square pieces.
theorem number_of_squares_is_five (initial_shape : Type) (h_initial_square : initial_shape = square)
  (h_folds : folds = 2) (h_cuts : cuts = 2) : folded_and_cut_number_of_squares initial_shape folds cuts = 5 :=
  sorry

end NUMINAMATH_GPT_number_of_squares_is_five_l1912_191248


namespace NUMINAMATH_GPT_xiaoli_estimate_smaller_l1912_191262

variable (x y z : ℝ)
variable (hx : x > y) (hz : z > 0)

theorem xiaoli_estimate_smaller :
  (x - z) - (y + z) < x - y := 
by
  sorry

end NUMINAMATH_GPT_xiaoli_estimate_smaller_l1912_191262


namespace NUMINAMATH_GPT_prob_two_red_balls_in_four_draws_l1912_191233

noncomputable def probability_red_balls (draws : ℕ) (red_in_draw : ℕ) (total_balls : ℕ) (red_balls : ℕ) : ℝ :=
  let prob_red := (red_balls : ℝ) / (total_balls : ℝ)
  let prob_white := 1 - prob_red
  (Nat.choose draws red_in_draw : ℝ) * (prob_red ^ red_in_draw) * (prob_white ^ (draws - red_in_draw))

theorem prob_two_red_balls_in_four_draws :
  probability_red_balls 4 2 10 4 = 0.3456 :=
by
  sorry

end NUMINAMATH_GPT_prob_two_red_balls_in_four_draws_l1912_191233


namespace NUMINAMATH_GPT_find_m_eq_zero_l1912_191239

-- Given two sets A and B
def A (m : ℝ) : Set ℝ := {3, m}
def B (m : ℝ) : Set ℝ := {3 * m, 3}

-- The assumption that A equals B
axiom A_eq_B (m : ℝ) : A m = B m

-- Prove that m = 0
theorem find_m_eq_zero (m : ℝ) (h : A m = B m) : m = 0 := by
  sorry

end NUMINAMATH_GPT_find_m_eq_zero_l1912_191239


namespace NUMINAMATH_GPT_polygon_diagonals_30_l1912_191266

-- Define the properties and conditions of the problem
def sides := 30

-- Define the number of diagonals calculation function
def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- The proof statement to check the number of diagonals in a 30-sided convex polygon
theorem polygon_diagonals_30 : num_diagonals sides = 375 := by
  sorry

end NUMINAMATH_GPT_polygon_diagonals_30_l1912_191266


namespace NUMINAMATH_GPT_heartsuit_3_8_l1912_191222

def heartsuit (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem heartsuit_3_8 : heartsuit 3 8 = 60 := by
  sorry

end NUMINAMATH_GPT_heartsuit_3_8_l1912_191222


namespace NUMINAMATH_GPT_integer_solution_of_floor_equation_l1912_191210

theorem integer_solution_of_floor_equation (n : ℤ) : 
  (⌊n^2 / 4⌋ - ⌊n / 2⌋^2 = 5) ↔ (n = 11) :=
by sorry

end NUMINAMATH_GPT_integer_solution_of_floor_equation_l1912_191210


namespace NUMINAMATH_GPT_women_left_l1912_191280

-- Definitions for initial numbers of men and women
def initial_men (M : ℕ) : Prop := M + 2 = 14
def initial_women (W : ℕ) (M : ℕ) : Prop := 5 * M = 4 * W

-- Definition for the final state after women left
def final_state (M W : ℕ) (X : ℕ) : Prop := 2 * (W - X) = 24

-- The problem statement in Lean 4
theorem women_left (M W X : ℕ) (h_men : initial_men M) 
  (h_women : initial_women W M) (h_final : final_state M W X) : X = 3 :=
sorry

end NUMINAMATH_GPT_women_left_l1912_191280


namespace NUMINAMATH_GPT_solve_for_a_l1912_191251

theorem solve_for_a (a x : ℝ) (h : 2 * x + 3 * a = 10) (hx : x = 2) : a = 2 :=
by
  rw [hx] at h
  linarith

end NUMINAMATH_GPT_solve_for_a_l1912_191251


namespace NUMINAMATH_GPT_min_value_abs_sum_pqr_inequality_l1912_191269

theorem min_value_abs_sum (x : ℝ) : |x + 1| + |x - 2| ≥ 3 :=
by
  sorry

theorem pqr_inequality (p q r : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := 
by
  have f_min : ∀ x, |x + 1| + |x - 2| ≥ 3 := min_value_abs_sum
  sorry

end NUMINAMATH_GPT_min_value_abs_sum_pqr_inequality_l1912_191269


namespace NUMINAMATH_GPT_m_n_solution_l1912_191290

theorem m_n_solution (m n : ℝ) (h1 : m - n = -5) (h2 : m^2 + n^2 = 13) : m^4 + n^4 = 97 :=
by
  sorry

end NUMINAMATH_GPT_m_n_solution_l1912_191290


namespace NUMINAMATH_GPT_race_head_start_l1912_191281

theorem race_head_start (Va Vb L H : ℚ) (h : Va = 30 / 17 * Vb) :
  H = 13 / 30 * L :=
by
  sorry

end NUMINAMATH_GPT_race_head_start_l1912_191281


namespace NUMINAMATH_GPT_diego_annual_savings_l1912_191261

-- Definitions based on conditions
def monthly_deposit := 5000
def monthly_expense := 4600
def months_in_year := 12

-- Prove that Diego's annual savings is $4800
theorem diego_annual_savings : (monthly_deposit - monthly_expense) * months_in_year = 4800 := by
  sorry

end NUMINAMATH_GPT_diego_annual_savings_l1912_191261


namespace NUMINAMATH_GPT_intersection_interval_l1912_191240

noncomputable def f (x: ℝ) : ℝ := Real.log x
noncomputable def g (x: ℝ) : ℝ := 7 - 2 * x

theorem intersection_interval : ∃ x : ℝ, 3 < x ∧ x < 4 ∧ f x = g x := 
sorry

end NUMINAMATH_GPT_intersection_interval_l1912_191240


namespace NUMINAMATH_GPT_sin_arcsin_plus_arctan_l1912_191225

theorem sin_arcsin_plus_arctan (a b : ℝ) (ha : a = Real.arcsin (4/5)) (hb : b = Real.arctan (1/2)) :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = (11 * Real.sqrt 5) / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_arcsin_plus_arctan_l1912_191225


namespace NUMINAMATH_GPT_mukesh_total_debt_l1912_191286

-- Define the initial principal, additional loan, interest rate, and time periods
def principal₁ : ℝ := 10000
def principal₂ : ℝ := 12000
def rate : ℝ := 0.06
def time₁ : ℝ := 2
def time₂ : ℝ := 3

-- Define the interest calculations
def interest₁ : ℝ := principal₁ * rate * time₁
def total_after_2_years : ℝ := principal₁ + interest₁ + principal₂
def interest₂ : ℝ := total_after_2_years * rate * time₂

-- Define the total amount owed after 5 years
def amount_owed : ℝ := total_after_2_years + interest₂

-- The goal is to prove that Mukesh owes 27376 Rs after 5 years
theorem mukesh_total_debt : amount_owed = 27376 := by sorry

end NUMINAMATH_GPT_mukesh_total_debt_l1912_191286


namespace NUMINAMATH_GPT_angle_between_vectors_45_degrees_l1912_191293

open Real

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def vec_mag (v : ℝ × ℝ) : ℝ := sqrt (vec_dot v v)

noncomputable def vec_angle (v w : ℝ × ℝ) : ℝ := arccos (vec_dot v w / (vec_mag v * vec_mag w))

theorem angle_between_vectors_45_degrees 
  (e1 e2 : ℝ × ℝ)
  (h1 : vec_mag e1 = 1)
  (h2 : vec_mag e2 = 1)
  (h3 : vec_dot e1 e2 = 0)
  (a : ℝ × ℝ := (3, 0) - (0, 1))  -- (3 * e1 - e2) is represented in a direct vector form (3, -1)
  (b : ℝ × ℝ := (2, 0) + (0, 1)): -- (2 * e1 + e2) is represented in a direct vector form (2, 1)
  vec_angle a b = π / 4 :=  -- π / 4 radians is equivalent to 45 degrees
sorry

end NUMINAMATH_GPT_angle_between_vectors_45_degrees_l1912_191293


namespace NUMINAMATH_GPT_smallest_positive_integer_l1912_191292

theorem smallest_positive_integer (n : ℕ) (h : 721 * n % 30 = 1137 * n % 30) :
  ∃ k : ℕ, k > 0 ∧ n = 2 * k :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1912_191292


namespace NUMINAMATH_GPT_students_still_inward_l1912_191244

theorem students_still_inward (num_students : ℕ) (turns : ℕ) : (num_students = 36) ∧ (turns = 36) → ∃ n, n = 26 :=
by
  sorry

end NUMINAMATH_GPT_students_still_inward_l1912_191244


namespace NUMINAMATH_GPT_log7_18_l1912_191299

theorem log7_18 (a b : ℝ) (h1 : Real.log 2 / Real.log 10 = a) (h2 : Real.log 3 / Real.log 10 = b) : 
  Real.log 18 / Real.log 7 = (a + 2 * b) / (1 - a) :=
by
  -- proof to be completed
  sorry

end NUMINAMATH_GPT_log7_18_l1912_191299


namespace NUMINAMATH_GPT_volume_second_cube_l1912_191298

open Real

-- Define the ratio of the edges of the cubes
def edge_ratio (a b : ℝ) := a / b = 3 / 1

-- Define the volume of the first cube
def volume_first_cube (a : ℝ) := a^3 = 27

-- Define the edge of the second cube based on the edge of the first cube
def edge_second_cube (a b : ℝ) := a / 3 = b

-- Statement of the problem in Lean 4
theorem volume_second_cube 
  (a b : ℝ) 
  (h_edge_ratio : edge_ratio a b) 
  (h_volume_first : volume_first_cube a) 
  (h_edge_second : edge_second_cube a b) : 
  b^3 = 1 := 
sorry

end NUMINAMATH_GPT_volume_second_cube_l1912_191298


namespace NUMINAMATH_GPT_apples_needed_for_two_weeks_l1912_191272

theorem apples_needed_for_two_weeks :
  ∀ (apples_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ),
  apples_per_day = 1 → days_per_week = 7 → weeks = 2 →
  apples_per_day * days_per_week * weeks = 14 :=
by
  intros apples_per_day days_per_week weeks h1 h2 h3
  sorry

end NUMINAMATH_GPT_apples_needed_for_two_weeks_l1912_191272


namespace NUMINAMATH_GPT_original_number_of_people_l1912_191273

-- Define the conditions as Lean definitions
def two_thirds_left (x : ℕ) : ℕ := (2 * x) / 3
def one_fourth_dancing_left (x : ℕ) : ℕ := ((x / 3) - (x / 12))

-- The problem statement as Lean theorem
theorem original_number_of_people (x : ℕ) (h : x / 4 = 15) : x = 60 :=
by sorry

end NUMINAMATH_GPT_original_number_of_people_l1912_191273


namespace NUMINAMATH_GPT_bridget_heavier_than_martha_l1912_191263

def bridget_weight := 39
def martha_weight := 2

theorem bridget_heavier_than_martha :
  bridget_weight - martha_weight = 37 :=
by
  sorry

end NUMINAMATH_GPT_bridget_heavier_than_martha_l1912_191263


namespace NUMINAMATH_GPT_find_pair_l1912_191264

theorem find_pair :
  ∃ x y : ℕ, (1984 * x - 1983 * y = 1985) ∧ (x = 27764) ∧ (y = 27777) :=
by
  sorry

end NUMINAMATH_GPT_find_pair_l1912_191264


namespace NUMINAMATH_GPT_parabola_properties_l1912_191216

noncomputable def parabola (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties
  (a b c t m n x₀ : ℝ)
  (ha : a > 0)
  (h1 : parabola a b c 1 = m)
  (h4 : parabola a b c 4 = n)
  (ht : t = -b / (2 * a))
  (h3ab : 3 * a + b = 0) 
  (hmnc : m < c ∧ c < n)
  (hx₀ym : parabola a b c x₀ = m) :
  m < n ∧ (1 / 2) < t ∧ t < 2 ∧ 0 < x₀ ∧ x₀ < 3 :=
  sorry

end NUMINAMATH_GPT_parabola_properties_l1912_191216


namespace NUMINAMATH_GPT_sum_first_60_natural_numbers_l1912_191277

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end NUMINAMATH_GPT_sum_first_60_natural_numbers_l1912_191277


namespace NUMINAMATH_GPT_relationship_among_x_y_z_w_l1912_191279

theorem relationship_among_x_y_z_w (x y z w : ℝ) (h : (x + y) / (y + z) = (z + w) / (w + x)) :
  x = z ∨ x + y + w + z = 0 :=
sorry

end NUMINAMATH_GPT_relationship_among_x_y_z_w_l1912_191279


namespace NUMINAMATH_GPT_identify_false_condition_l1912_191245

-- Define the quadratic function
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
def condition_A (a b c : ℝ) : Prop := quadratic_function a b c (-1) = 0
def condition_B (a b c : ℝ) : Prop := 2 * a + b = 0
def condition_C (a b c : ℝ) : Prop := quadratic_function a b c 1 = 3
def condition_D (a b c : ℝ) : Prop := quadratic_function a b c 2 = 8

-- Main theorem stating which condition is false
theorem identify_false_condition (a b c : ℝ) (ha : a ≠ 0) : ¬ condition_A a b c ∨ ¬ condition_B a b c ∨ ¬ condition_C a b c ∨  ¬ condition_D a b c :=
by
sorry

end NUMINAMATH_GPT_identify_false_condition_l1912_191245


namespace NUMINAMATH_GPT_part1_part2_l1912_191214

-- Condition definitions
def income2017 : ℝ := 2500
def income2019 : ℝ := 3600
def n : ℕ := 2

-- Part 1: Prove the annual growth rate
theorem part1 (x : ℝ) (hx : income2019 = income2017 * (1 + x) ^ n) : x = 0.2 :=
by sorry

-- Part 2: Prove reaching 4200 yuan with the same growth rate
theorem part2 (hx : income2019 = income2017 * (1 + 0.2) ^ n) : 3600 * (1 + 0.2) ≥ 4200 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l1912_191214


namespace NUMINAMATH_GPT_no_such_polyhedron_l1912_191230

theorem no_such_polyhedron (n : ℕ) (S : Fin n → ℝ) (H : ∀ i j : Fin n, i ≠ j → S i ≥ 2 * S j) : False :=
by
  sorry

end NUMINAMATH_GPT_no_such_polyhedron_l1912_191230


namespace NUMINAMATH_GPT_time_against_current_l1912_191209

-- Define the conditions:
def swimming_speed_still_water : ℝ := 6  -- Speed in still water (km/h)
def current_speed : ℝ := 2  -- Speed of the water current (km/h)
def time_with_current : ℝ := 3.5  -- Time taken to swim with the current (hours)

-- Define effective speeds:
def effective_speed_against_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water - current_speed

def effective_speed_with_current (swimming_speed_still_water current_speed: ℝ) : ℝ :=
  swimming_speed_still_water + current_speed

-- Calculate the distance covered with the current:
def distance_with_current (time_with_current effective_speed_with_current: ℝ) : ℝ :=
  time_with_current * effective_speed_with_current

-- Define the proof goal:
theorem time_against_current (h1 : swimming_speed_still_water = 6) (h2 : current_speed = 2)
  (h3 : time_with_current = 3.5) :
  ∃ (t : ℝ), t = 7 := by
  sorry

end NUMINAMATH_GPT_time_against_current_l1912_191209


namespace NUMINAMATH_GPT_solution_set_inequality_l1912_191271

theorem solution_set_inequality (x : ℝ) : (3 * x^2 + 7 * x ≤ 2) ↔ (-2 ≤ x ∧ x ≤ 1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1912_191271


namespace NUMINAMATH_GPT_sally_initial_poems_l1912_191242

theorem sally_initial_poems (recited: ℕ) (forgotten: ℕ) (h1 : recited = 3) (h2 : forgotten = 5) : 
  recited + forgotten = 8 := 
by
  sorry

end NUMINAMATH_GPT_sally_initial_poems_l1912_191242


namespace NUMINAMATH_GPT_kate_bought_wands_l1912_191276

theorem kate_bought_wands (price_per_wand : ℕ)
                           (additional_cost : ℕ)
                           (total_money_collected : ℕ)
                           (number_of_wands_sold : ℕ)
                           (total_wands_bought : ℕ) :
  price_per_wand = 60 → additional_cost = 5 → total_money_collected = 130 → 
  number_of_wands_sold = total_money_collected / (price_per_wand + additional_cost) →
  total_wands_bought = number_of_wands_sold + 1 →
  total_wands_bought = 3 := by
  sorry

end NUMINAMATH_GPT_kate_bought_wands_l1912_191276


namespace NUMINAMATH_GPT_trapezium_area_l1912_191228

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 15) :
  (1/2) * (a + b) * h = 285 := by
  sorry

end NUMINAMATH_GPT_trapezium_area_l1912_191228


namespace NUMINAMATH_GPT_abs_sum_leq_abs_l1912_191268

theorem abs_sum_leq_abs (a b : ℝ) (h1 : a < b) (h2 : b < 0) :
  |a| + |b| ≤ |a + b| :=
sorry

end NUMINAMATH_GPT_abs_sum_leq_abs_l1912_191268


namespace NUMINAMATH_GPT_measure_of_angle_C_l1912_191270

theorem measure_of_angle_C (C D : ℝ) (h1 : C + D = 180) (h2 : C = 12 * D) : C = 2160 / 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_measure_of_angle_C_l1912_191270


namespace NUMINAMATH_GPT_find_angle_and_area_of_triangle_l1912_191243

theorem find_angle_and_area_of_triangle (a b : ℝ) 
  (h_a : a = Real.sqrt 7) (h_b : b = 2)
  (angle_A : ℝ) (angle_A_eq : angle_A = Real.pi / 3)
  (angle_B : ℝ)
  (vec_m : ℝ × ℝ := (a, Real.sqrt 3 * b))
  (vec_n : ℝ × ℝ := (Real.cos angle_A, Real.sin angle_B))
  (colinear : vec_m.1 * vec_n.2 = vec_m.2 * vec_n.1)
  (sin_A : Real.sin angle_A = (Real.sqrt 3) / 2)
  (cos_A : Real.cos angle_A = 1 / 2) :
  angle_A = Real.pi / 3 ∧ 
  ∃ (c : ℝ), c = 3 ∧
  (1/2) * b * c * Real.sin angle_A = 3 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_and_area_of_triangle_l1912_191243


namespace NUMINAMATH_GPT_find_b_l1912_191267

def point := ℝ × ℝ

def dir_vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def scale_vector (v : point) (s : ℝ) : point := (s * v.1, s * v.2)

theorem find_b (p1 p2 : point) (b : ℝ) :
  p1 = (-5, 0) → p2 = (-2, 2) →
  dir_vector p1 p2 = (3, 2) →
  scale_vector (3, 2) (2 / 3) = (2, b) →
  b = 4 / 3 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_find_b_l1912_191267


namespace NUMINAMATH_GPT_initial_percentage_liquid_X_l1912_191235

theorem initial_percentage_liquid_X (P : ℝ) :
  let original_solution_kg := 8
  let evaporated_water_kg := 2
  let added_solution_kg := 2
  let remaining_solution_kg := original_solution_kg - evaporated_water_kg
  let new_solution_kg := remaining_solution_kg + added_solution_kg
  let new_solution_percentage := 0.25
  let initial_liquid_X_kg := (P / 100) * original_solution_kg
  let final_liquid_X_kg := initial_liquid_X_kg + (P / 100) * added_solution_kg
  let final_liquid_X_kg' := new_solution_percentage * new_solution_kg
  (final_liquid_X_kg = final_liquid_X_kg') → 
  P = 20 :=
by
  intros
  let original_solution_kg_p0 := 8
  let evaporated_water_kg_p1 := 2
  let added_solution_kg_p2 := 2
  let remaining_solution_kg_p3 := (original_solution_kg_p0 - evaporated_water_kg_p1)
  let new_solution_kg_p4 := (remaining_solution_kg_p3 + added_solution_kg_p2)
  let new_solution_percentage : ℝ := 0.25
  let initial_liquid_X_kg_p6 := ((P / 100) * original_solution_kg_p0)
  let final_liquid_X_kg_p7 := initial_liquid_X_kg_p6 + ((P / 100) * added_solution_kg_p2)
  let final_liquid_X_kg_p8 := (new_solution_percentage * new_solution_kg_p4)
  exact sorry

end NUMINAMATH_GPT_initial_percentage_liquid_X_l1912_191235


namespace NUMINAMATH_GPT_length_of_third_side_l1912_191211

theorem length_of_third_side (A B C : ℝ) (a b c : ℝ)
  (h1 : a = 12) (h2 : c = 18) (h3 : B = 2 * C) :
  ∃ a, a = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_length_of_third_side_l1912_191211


namespace NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_l1912_191247

theorem solve_quadratic_eq1 (x : ℝ) :
  x^2 - 4 * x + 3 = 0 ↔ (x = 3 ∨ x = 1) :=
sorry

theorem solve_quadratic_eq2 (x : ℝ) :
  x^2 - x - 3 = 0 ↔ (x = (1 + Real.sqrt 13) / 2 ∨ x = (1 - Real.sqrt 13) / 2) :=
sorry

end NUMINAMATH_GPT_solve_quadratic_eq1_solve_quadratic_eq2_l1912_191247


namespace NUMINAMATH_GPT_age_of_youngest_child_l1912_191282

theorem age_of_youngest_child (x : ℕ) 
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 65) : x = 7 :=
sorry

end NUMINAMATH_GPT_age_of_youngest_child_l1912_191282


namespace NUMINAMATH_GPT_train_length_l1912_191212

theorem train_length (time : ℝ) (speed_kmh : ℝ) (speed_ms : ℝ) (length : ℝ) : 
  time = 3.499720022398208 ∧ 
  speed_kmh = 144 ∧ 
  speed_ms = 40 ∧ 
  length = speed_ms * time → 
  length = 139.98880089592832 :=
by sorry

end NUMINAMATH_GPT_train_length_l1912_191212


namespace NUMINAMATH_GPT_jordan_width_l1912_191285

-- Definitions based on conditions
def area_of_carols_rectangle : ℝ := 15 * 20
def jordan_length_feet : ℝ := 6
def feet_to_inches (feet: ℝ) : ℝ := feet * 12
def jordan_length_inches : ℝ := feet_to_inches jordan_length_feet

-- Main statement
theorem jordan_width :
  ∃ w : ℝ, w = 300 / 72 :=
sorry

end NUMINAMATH_GPT_jordan_width_l1912_191285


namespace NUMINAMATH_GPT_factorization_c_minus_d_l1912_191215

theorem factorization_c_minus_d : 
  ∃ (c d : ℤ), (∀ (x : ℤ), (4 * x^2 - 17 * x - 15 = (4 * x + c) * (x + d))) ∧ (c - d = 8) :=
by
  sorry

end NUMINAMATH_GPT_factorization_c_minus_d_l1912_191215


namespace NUMINAMATH_GPT_complement_A_is_correct_l1912_191255

-- Let A be the set representing the domain of the function y = log2(x - 1)
def A : Set ℝ := { x : ℝ | x > 1 }

-- The universal set is ℝ
def U : Set ℝ := Set.univ

-- Complement of A with respect to ℝ
def complement_A (U : Set ℝ) (A : Set ℝ) : Set ℝ := U \ A

-- Prove that the complement of A with respect to ℝ is (-∞, 1]
theorem complement_A_is_correct : complement_A U A = { x : ℝ | x ≤ 1 } :=
by {
 sorry
}

end NUMINAMATH_GPT_complement_A_is_correct_l1912_191255


namespace NUMINAMATH_GPT_total_participants_l1912_191218

theorem total_participants
  (F M : ℕ) 
  (half_female_democrats : F / 2 = 125)
  (one_third_democrats : (F + M) / 3 = (125 + M / 4))
  : F + M = 1750 :=
by
  sorry

end NUMINAMATH_GPT_total_participants_l1912_191218


namespace NUMINAMATH_GPT_cubic_meter_to_cubic_centimeters_l1912_191236

theorem cubic_meter_to_cubic_centimeters :
  (1 : ℝ) ^ 3 = (100 : ℝ) ^ 3 := by
  sorry

end NUMINAMATH_GPT_cubic_meter_to_cubic_centimeters_l1912_191236


namespace NUMINAMATH_GPT_gcd_a_b_l1912_191250

def a (n : ℤ) : ℤ := n^5 + 6 * n^3 + 8 * n
def b (n : ℤ) : ℤ := n^4 + 4 * n^2 + 3

theorem gcd_a_b (n : ℤ) : ∃ d : ℤ, d = Int.gcd (a n) (b n) ∧ (d = 1 ∨ d = 3) :=
by
  sorry

end NUMINAMATH_GPT_gcd_a_b_l1912_191250


namespace NUMINAMATH_GPT_cubic_coefficient_determination_l1912_191224

def f (x : ℚ) (A B C D : ℚ) : ℚ := A*x^3 + B*x^2 + C*x + D

theorem cubic_coefficient_determination {A B C D : ℚ}
  (h1 : f 1 A B C D = 0)
  (h2 : f (2/3) A B C D = -4)
  (h3 : f (4/5) A B C D = -16/5) :
  A = 15 ∧ B = -37 ∧ C = 30 ∧ D = -8 :=
  sorry

end NUMINAMATH_GPT_cubic_coefficient_determination_l1912_191224


namespace NUMINAMATH_GPT_y_value_solution_l1912_191221

theorem y_value_solution (y : ℝ) (h : (3 / y) - ((4 / y) * (2 / y)) = 1.5) : 
  y = 1 + Real.sqrt (19 / 3) := 
sorry

end NUMINAMATH_GPT_y_value_solution_l1912_191221


namespace NUMINAMATH_GPT_proof_goal_l1912_191252

noncomputable def exp_value (k m n : ℕ) : ℤ :=
  (6^k - k^6 + 2^m - 4^m + n^3 - 3^n : ℤ)

theorem proof_goal (k m n : ℕ) (h_k : 18^k ∣ 624938) (h_m : 24^m ∣ 819304) (h_n : n = 2 * k + m) :
  exp_value k m n = 0 := by
  sorry

end NUMINAMATH_GPT_proof_goal_l1912_191252


namespace NUMINAMATH_GPT_total_cartons_accepted_l1912_191204

theorem total_cartons_accepted (total_cartons : ℕ) (customers : ℕ) (damaged_cartons_per_customer : ℕ) (initial_cartons_per_customer accepted_cartons_per_customer total_accepted_cartons : ℕ) :
    total_cartons = 400 →
    customers = 4 →
    damaged_cartons_per_customer = 60 →
    initial_cartons_per_customer = total_cartons / customers →
    accepted_cartons_per_customer = initial_cartons_per_customer - damaged_cartons_per_customer →
    total_accepted_cartons = accepted_cartons_per_customer * customers →
    total_accepted_cartons = 160 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_total_cartons_accepted_l1912_191204


namespace NUMINAMATH_GPT_age_twice_in_two_years_l1912_191246

-- conditions
def father_age (S : ℕ) : ℕ := S + 24
def present_son_age : ℕ := 22
def present_father_age : ℕ := father_age present_son_age

-- theorem statement
theorem age_twice_in_two_years (S M Y : ℕ) (h1 : S = present_son_age) (h2 : M = present_father_age) : 
  M + 2 = 2 * (S + 2) :=
by
  sorry

end NUMINAMATH_GPT_age_twice_in_two_years_l1912_191246


namespace NUMINAMATH_GPT_multiplier_for_average_grade_l1912_191223

/-- Conditions -/
def num_of_grades_2 : ℕ := 3
def num_of_grades_3 : ℕ := 4
def num_of_grades_4 : ℕ := 1
def num_of_grades_5 : ℕ := 1
def cash_reward : ℕ := 15

-- Definitions for sums and averages based on the conditions
def sum_of_grades : ℕ :=
  num_of_grades_2 * 2 + num_of_grades_3 * 3 + num_of_grades_4 * 4 + num_of_grades_5 * 5

def total_grades : ℕ :=
  num_of_grades_2 + num_of_grades_3 + num_of_grades_4 + num_of_grades_5

def average_grade : ℕ :=
  sum_of_grades / total_grades

/-- Proof statement -/
theorem multiplier_for_average_grade : cash_reward / average_grade = 5 := by
  sorry

end NUMINAMATH_GPT_multiplier_for_average_grade_l1912_191223


namespace NUMINAMATH_GPT_average_non_prime_squares_approx_l1912_191257

-- Define a function to check if a number is prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the list of non-prime numbers between 50 and 100
def non_prime_numbers : List ℕ :=
  [51, 52, 54, 55, 56, 57, 58, 60, 62, 63, 64, 65, 66, 68, 69, 70,
   72, 74, 75, 76, 77, 78, 80, 81, 82, 84, 85, 86, 87, 88, 90, 91,
   92, 93, 94, 95, 96, 98, 99]

-- Define the sum of squares of the elements in a list
def sum_of_squares (l : List ℕ) : ℕ :=
  l.foldr (λ x acc => x * x + acc) 0

-- Define the count of non-prime numbers
def count_non_prime : ℕ :=
  non_prime_numbers.length

-- Calculate the average
def average_non_prime_squares : ℚ :=
  sum_of_squares non_prime_numbers / count_non_prime

-- Theorem to state that the average of the sum of squares of non-prime numbers
-- between 50 and 100 is approximately 6417.67
theorem average_non_prime_squares_approx :
  abs ((average_non_prime_squares : ℝ) - 6417.67) < 0.01 := 
  sorry

end NUMINAMATH_GPT_average_non_prime_squares_approx_l1912_191257


namespace NUMINAMATH_GPT_smallest_positive_angle_equivalent_neg_1990_l1912_191237

theorem smallest_positive_angle_equivalent_neg_1990:
  ∃ k : ℤ, 0 ≤ (θ : ℤ) ∧ θ < 360 ∧ -1990 + 360 * k = θ := by
  use 6
  sorry

end NUMINAMATH_GPT_smallest_positive_angle_equivalent_neg_1990_l1912_191237


namespace NUMINAMATH_GPT_union_of_setA_and_setB_l1912_191226

def setA : Set ℕ := {1, 2, 4}
def setB : Set ℕ := {2, 6}

theorem union_of_setA_and_setB :
  setA ∪ setB = {1, 2, 4, 6} :=
by sorry

end NUMINAMATH_GPT_union_of_setA_and_setB_l1912_191226


namespace NUMINAMATH_GPT_age_of_b_l1912_191238

-- Definition of conditions
variable (a b c : ℕ)
variable (h1 : a = b + 2)
variable (h2 : b = 2 * c)
variable (h3 : a + b + c = 12)

-- The statement of the proof problem
theorem age_of_b : b = 4 :=
by {
   sorry
}

end NUMINAMATH_GPT_age_of_b_l1912_191238


namespace NUMINAMATH_GPT_marble_ratio_l1912_191234

theorem marble_ratio (A J C : ℕ) (h1 : 3 * (A + J + C) = 60) (h2 : A = 4) (h3 : C = 8) : A / J = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_marble_ratio_l1912_191234


namespace NUMINAMATH_GPT_trig_identity_l1912_191229

theorem trig_identity :
  (Real.sin (17 * Real.pi / 180) * Real.cos (47 * Real.pi / 180) - 
   Real.sin (73 * Real.pi / 180) * Real.cos (43 * Real.pi / 180)) = -1/2 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1912_191229


namespace NUMINAMATH_GPT_sector_to_cone_base_area_l1912_191254

theorem sector_to_cone_base_area
  (r_sector : ℝ) (theta : ℝ) (h1 : r_sector = 2) (h2 : theta = 120) :
  ∃ (A : ℝ), A = (4 / 9) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_sector_to_cone_base_area_l1912_191254


namespace NUMINAMATH_GPT_minimum_value_inequality_l1912_191220

theorem minimum_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h : x + 2 * y + 3 * z = 1) :
  (16 / x^3 + 81 / (8 * y^3) + 1 / (27 * z^3)) ≥ 1296 := sorry

end NUMINAMATH_GPT_minimum_value_inequality_l1912_191220


namespace NUMINAMATH_GPT_evaluate_expression_l1912_191205

-- Defining the conditions and constants as per the problem statement
def factor_power_of_2 (n : ℕ) : ℕ :=
  if n % 8 = 0 then 3 else 0 -- Greatest power of 2 in 360
  
def factor_power_of_5 (n : ℕ) : ℕ :=
  if n % 5 = 0 then 1 else 0 -- Greatest power of 5 in 360

def expression (b a : ℕ) : ℚ := (2 / 3)^(b - a)

noncomputable def target_value : ℚ := 9 / 4

theorem evaluate_expression : expression (factor_power_of_5 360) (factor_power_of_2 360) = target_value := 
  by
    sorry

end NUMINAMATH_GPT_evaluate_expression_l1912_191205


namespace NUMINAMATH_GPT_banker_l1912_191259

theorem banker's_discount (BD TD FV : ℝ) (hBD : BD = 18) (hTD : TD = 15) 
(h : BD = TD + (TD^2 / FV)) : FV = 75 := by
  sorry

end NUMINAMATH_GPT_banker_l1912_191259


namespace NUMINAMATH_GPT_jellybean_total_count_l1912_191207

theorem jellybean_total_count :
  let black := 8
  let green := 2 * black
  let orange := (2 * green) - 5
  let red := orange + 3
  let yellow := black / 2
  let purple := red + 4
  let brown := (green + purple) - 3
  black + green + orange + red + yellow + purple + brown = 166 := by
  -- skipping proof for brevity
  sorry

end NUMINAMATH_GPT_jellybean_total_count_l1912_191207


namespace NUMINAMATH_GPT_Lauren_total_revenue_l1912_191289

noncomputable def LaurenMondayEarnings (views subscriptions : ℕ) : ℝ :=
  (views * 0.40) + (subscriptions * 0.80)

noncomputable def LaurenTuesdayEarningsEUR (views subscriptions : ℕ) : ℝ :=
  (views * 0.40) + (subscriptions * 0.75)

noncomputable def convertEURtoUSD (eur : ℝ) : ℝ :=
  eur * (1 / 0.85)

noncomputable def convertGBPtoUSD (gbp : ℝ) : ℝ :=
  gbp * 1.38

noncomputable def LaurenWeekendEarnings (sales : ℝ) : ℝ :=
  (sales * 0.10)

theorem Lauren_total_revenue :
  let monday_views := 80
  let monday_subscriptions := 20
  let tuesday_views := 100
  let tuesday_subscriptions := 27
  let weekend_sales := 100

  let monday_earnings := LaurenMondayEarnings monday_views monday_subscriptions
  let tuesday_earnings_eur := LaurenTuesdayEarningsEUR tuesday_views tuesday_subscriptions
  let tuesday_earnings_usd := convertEURtoUSD tuesday_earnings_eur
  let weekend_earnings_gbp := LaurenWeekendEarnings weekend_sales
  let weekend_earnings_usd := convertGBPtoUSD weekend_earnings_gbp

  monday_earnings + tuesday_earnings_usd + weekend_earnings_usd = 132.68 :=
by
  sorry

end NUMINAMATH_GPT_Lauren_total_revenue_l1912_191289


namespace NUMINAMATH_GPT_polyhedron_inequality_proof_l1912_191213

noncomputable def polyhedron_inequality (B : ℕ) (P : ℕ) (T : ℕ) : Prop :=
  B * Real.sqrt (P + T) ≥ 2 * P

theorem polyhedron_inequality_proof (B P T : ℕ) 
  (h1 : 0 < B) (h2 : 0 < P) (h3 : 0 < T) 
  (condition_is_convex_polyhedron : true) : 
  polyhedron_inequality B P T :=
sorry

end NUMINAMATH_GPT_polyhedron_inequality_proof_l1912_191213


namespace NUMINAMATH_GPT_circle_in_quad_radius_l1912_191283

theorem circle_in_quad_radius (AB BC CD DA : ℝ) (r : ℝ) (h₁ : AB = 15) (h₂ : BC = 10) (h₃ : CD = 8) (h₄ : DA = 13) :
  r = 2 * Real.sqrt 10 := 
by {
  sorry
  }

end NUMINAMATH_GPT_circle_in_quad_radius_l1912_191283


namespace NUMINAMATH_GPT_garrison_reinforcement_l1912_191208

theorem garrison_reinforcement (x : ℕ) (h1 : ∀ (n m p : ℕ), n * m = p → x = n - m) :
  (150 * (31 - x) = 450 * 5) → x = 16 :=
by sorry

end NUMINAMATH_GPT_garrison_reinforcement_l1912_191208


namespace NUMINAMATH_GPT_factorization_correct_l1912_191206

theorem factorization_correct (m : ℤ) : m^2 - 1 = (m - 1) * (m + 1) :=
by {
  -- sorry, this is a place-holder for the proof.
  sorry
}

end NUMINAMATH_GPT_factorization_correct_l1912_191206


namespace NUMINAMATH_GPT_value_of_m_l1912_191217

theorem value_of_m :
  ∃ m : ℝ, (3 - 1) / (m + 2) = 1 → m = 0 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_m_l1912_191217


namespace NUMINAMATH_GPT_vertices_divisible_by_three_l1912_191260

namespace PolygonDivisibility

theorem vertices_divisible_by_three (v : Fin 2018 → ℤ) 
  (h_initial : (Finset.univ.sum v) = 1) 
  (h_move : ∀ i : Fin 2018, ∃ j : Fin 2018, abs (v i - v j) = 1) :
  ¬ ∃ (k : Fin 2018 → ℤ), (∀ n : Fin 2018, k n % 3 = 0) :=
by {
  sorry
}

end PolygonDivisibility

end NUMINAMATH_GPT_vertices_divisible_by_three_l1912_191260


namespace NUMINAMATH_GPT_amy_red_balloons_l1912_191288

theorem amy_red_balloons (total_balloons green_balloons blue_balloons : ℕ) (h₁ : total_balloons = 67) (h₂: green_balloons = 17) (h₃ : blue_balloons = 21) : (total_balloons - (green_balloons + blue_balloons)) = 29 :=
by
  sorry

end NUMINAMATH_GPT_amy_red_balloons_l1912_191288


namespace NUMINAMATH_GPT_xy_sum_cases_l1912_191295

theorem xy_sum_cases (x y : ℕ) (hxy1 : 0 < x) (hxy2 : x < 30)
                      (hy1 : 0 < y) (hy2 : y < 30)
                      (h : x + y + x * y = 119) : (x + y = 24) ∨ (x + y = 20) :=
sorry

end NUMINAMATH_GPT_xy_sum_cases_l1912_191295


namespace NUMINAMATH_GPT_amount_collected_from_ii_and_iii_class_l1912_191219

theorem amount_collected_from_ii_and_iii_class
  (P1 P2 P3 : ℕ) (F1 F2 F3 : ℕ) (total_amount amount_ii_iii : ℕ)
  (H1 : P1 / P2 = 1 / 50)
  (H2 : P1 / P3 = 1 / 100)
  (H3 : F1 / F2 = 5 / 2)
  (H4 : F1 / F3 = 5 / 1)
  (H5 : total_amount = 3575)
  (H6 : total_amount = (P1 * F1) + (P2 * F2) + (P3 * F3))
  (H7 : amount_ii_iii = (P2 * F2) + (P3 * F3)) :
  amount_ii_iii = 3488 := sorry

end NUMINAMATH_GPT_amount_collected_from_ii_and_iii_class_l1912_191219


namespace NUMINAMATH_GPT_find_maximum_marks_l1912_191296

theorem find_maximum_marks (M : ℝ) 
  (h1 : 0.60 * M = 270)
  (h2 : ∀ x : ℝ, 220 + 50 = x → x = 270) : 
  M = 450 :=
by
  sorry

end NUMINAMATH_GPT_find_maximum_marks_l1912_191296


namespace NUMINAMATH_GPT_prime_eq_sum_of_two_squares_l1912_191202

theorem prime_eq_sum_of_two_squares (p : ℕ) (hp_prime : Nat.Prime p) (hp_mod : p % 4 = 1) : 
  ∃ a b : ℤ, p = a^2 + b^2 := 
sorry

end NUMINAMATH_GPT_prime_eq_sum_of_two_squares_l1912_191202


namespace NUMINAMATH_GPT_factory_output_decrease_l1912_191297

noncomputable def original_output (O : ℝ) : ℝ :=
  O

noncomputable def increased_output_10_percent (O : ℝ) : ℝ :=
  O * 1.1

noncomputable def increased_output_30_percent (O : ℝ) : ℝ :=
  increased_output_10_percent O * 1.3

noncomputable def percentage_decrease_needed (original new_output : ℝ) : ℝ :=
  ((new_output - original) / new_output) * 100

theorem factory_output_decrease (O : ℝ) : 
  abs (percentage_decrease_needed (original_output O) (increased_output_30_percent O) - 30.07) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_factory_output_decrease_l1912_191297


namespace NUMINAMATH_GPT_sum_of_remainders_l1912_191275

theorem sum_of_remainders (a b c : ℕ) (h1 : a % 47 = 25) (h2 : b % 47 = 20) (h3 : c % 47 = 3) : 
  (a + b + c) % 47 = 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_sum_of_remainders_l1912_191275


namespace NUMINAMATH_GPT_reflected_line_equation_l1912_191278

def line_reflection_about_x_axis (x y : ℝ) : Prop :=
  x - y + 1 = 0 → y = -x - 1

theorem reflected_line_equation :
  ∀ (x y : ℝ), x - y + 1 = 0 → x + y + 1 = 0 :=
by
  intros x y h
  suffices y = -x - 1 by
    linarith
  sorry

end NUMINAMATH_GPT_reflected_line_equation_l1912_191278


namespace NUMINAMATH_GPT_find_line_equation_l1912_191201

theorem find_line_equation (a b : ℝ) :
  (2 * a + 3 * b = 0 ∧ a * b < 0) ↔ (3 * a - 2 * b = 0 ∨ a - b + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_line_equation_l1912_191201


namespace NUMINAMATH_GPT_determine_n_l1912_191274

theorem determine_n (n : ℕ) (h : 17^(4 * n) = (1 / 17)^(n - 30)) : n = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_determine_n_l1912_191274


namespace NUMINAMATH_GPT_merchant_marking_percentage_l1912_191227

theorem merchant_marking_percentage (L : ℝ) (p : ℝ) (d : ℝ) (c : ℝ) (profit : ℝ) 
  (purchase_price : ℝ) (selling_price : ℝ) (marked_price : ℝ) (list_price : ℝ) : 
  L = 100 ∧ p = 30 ∧ d = 20 ∧ c = 20 ∧ profit = 20 ∧ 
  purchase_price = L - L * (p / 100) ∧ 
  marked_price = 109.375 ∧ 
  selling_price = marked_price - marked_price * (d / 100) ∧ 
  selling_price - purchase_price = profit * (selling_price / 100) 
  → marked_price = 109.375 := by sorry

end NUMINAMATH_GPT_merchant_marking_percentage_l1912_191227


namespace NUMINAMATH_GPT_no_three_nat_numbers_with_sum_power_of_three_l1912_191231

noncomputable def powers_of_3 (n : ℕ) : ℕ := 3^n

theorem no_three_nat_numbers_with_sum_power_of_three :
  ¬ ∃ (a b c : ℕ) (k m n : ℕ), a + b = powers_of_3 k ∧ b + c = powers_of_3 m ∧ c + a = powers_of_3 n :=
by
  sorry

end NUMINAMATH_GPT_no_three_nat_numbers_with_sum_power_of_three_l1912_191231
