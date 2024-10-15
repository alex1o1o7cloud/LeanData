import Mathlib

namespace NUMINAMATH_GPT_derivative_at_1_l163_16371

noncomputable def f (x : ℝ) : ℝ := (Real.log x + 2^x) / x^2

theorem derivative_at_1 : (deriv f 1) = 2 * Real.log 2 - 3 := 
sorry

end NUMINAMATH_GPT_derivative_at_1_l163_16371


namespace NUMINAMATH_GPT_length_OD1_l163_16334

-- Define the hypothesis of the problem
noncomputable def sphere_center : Point := sorry -- center O of the sphere
noncomputable def radius_sphere : ℝ := 10 -- radius of the sphere

-- Define face intersection properties
noncomputable def face_AA1D1D_radius : ℝ := 1
noncomputable def face_A1B1C1D1_radius : ℝ := 1
noncomputable def face_CDD1C1_radius : ℝ := 3

-- Define the coordinates of D1 (or in abstract form, we'll assume it is a known point)
noncomputable def segment_OD1 : ℝ := sorry -- Length of OD1 segment to be calculated

-- The main theorem to prove
theorem length_OD1 : 
  -- Given conditions
  (face_AA1D1D_radius = 1) ∧ 
  (face_A1B1C1D1_radius = 1) ∧ 
  (face_CDD1C1_radius = 3) ∧ 
  (radius_sphere = 10) →
  -- Prove the length of segment OD1 is 17
  segment_OD1 = 17 :=
by
  sorry

end NUMINAMATH_GPT_length_OD1_l163_16334


namespace NUMINAMATH_GPT_cos_17pi_over_4_l163_16385

theorem cos_17pi_over_4 : Real.cos (17 * Real.pi / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_cos_17pi_over_4_l163_16385


namespace NUMINAMATH_GPT_bus_speed_including_stoppages_l163_16351

theorem bus_speed_including_stoppages
  (speed_without_stoppages : ℝ)
  (stoppage_time : ℝ)
  (remaining_time_ratio : ℝ)
  (h1 : speed_without_stoppages = 12)
  (h2 : stoppage_time = 0.5)
  (h3 : remaining_time_ratio = 1 - stoppage_time) :
  (speed_without_stoppages * remaining_time_ratio) = 6 := 
by
  sorry

end NUMINAMATH_GPT_bus_speed_including_stoppages_l163_16351


namespace NUMINAMATH_GPT_fraction_of_field_planted_l163_16308

theorem fraction_of_field_planted (AB AC : ℕ) (x : ℕ) (shortest_dist : ℕ) (hypotenuse : ℕ)
  (S : ℕ) (total_area : ℕ) (planted_area : ℕ) :
  AB = 5 ∧ AC = 12 ∧ hypotenuse = 13 ∧ shortest_dist = 2 ∧ x * x = S ∧ 
  total_area = 30 ∧ planted_area = total_area - S →
  (planted_area / total_area : ℚ) = 2951 / 3000 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_field_planted_l163_16308


namespace NUMINAMATH_GPT_flour_for_recipe_l163_16339

theorem flour_for_recipe (flour_needed shortening_have : ℚ)
  (flour_ratio shortening_ratio : ℚ) 
  (ratio : flour_ratio / shortening_ratio = 5)
  (shortening_used : shortening_ratio = 2 / 3) :
  flour_needed = 10 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_flour_for_recipe_l163_16339


namespace NUMINAMATH_GPT_calculate_expression_l163_16349

theorem calculate_expression : 5 * 7 + 9 * 4 - 36 / 3 + 48 / 4 = 71 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l163_16349


namespace NUMINAMATH_GPT_magnitude_of_a_l163_16348

variable (a b : EuclideanSpace ℝ (Fin 2))
variable (theta : ℝ)
variable (hθ : theta = π / 3)
variable (hb : ‖b‖ = 1)
variable (hab : ‖a + 2 • b‖ = 2 * sqrt 3)

theorem magnitude_of_a :
  ‖a‖ = 2 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_a_l163_16348


namespace NUMINAMATH_GPT_polynomial_root_arithmetic_sequence_l163_16343

theorem polynomial_root_arithmetic_sequence :
  (∃ (a d : ℝ), 
    (64 * (a - d)^3 + 144 * (a - d)^2 + 92 * (a - d) + 15 = 0) ∧
    (64 * a^3 + 144 * a^2 + 92 * a + 15 = 0) ∧
    (64 * (a + d)^3 + 144 * (a + d)^2 + 92 * (a + d) + 15 = 0) ∧
    (2 * d = 1)) := sorry

end NUMINAMATH_GPT_polynomial_root_arithmetic_sequence_l163_16343


namespace NUMINAMATH_GPT_cost_of_socks_l163_16314

theorem cost_of_socks (cost_shirt_no_discount cost_pants_no_discount cost_shirt_discounted cost_pants_discounted cost_socks_discounted total_savings team_size socks_cost_no_discount : ℝ) 
    (h1 : cost_shirt_no_discount = 7.5)
    (h2 : cost_pants_no_discount = 15)
    (h3 : cost_shirt_discounted = 6.75)
    (h4 : cost_pants_discounted = 13.5)
    (h5 : cost_socks_discounted = 3.75)
    (h6 : total_savings = 36)
    (h7 : team_size = 12)
    (h8 : 12 * (7.5 + 15 + socks_cost_no_discount) - 12 * (6.75 + 13.5 + 3.75) = 36)
    : socks_cost_no_discount = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_socks_l163_16314


namespace NUMINAMATH_GPT_decimalToFrac_l163_16318

theorem decimalToFrac : (145 / 100 : ℚ) = 29 / 20 := by
  sorry

end NUMINAMATH_GPT_decimalToFrac_l163_16318


namespace NUMINAMATH_GPT_find_original_number_l163_16397

def original_four_digit_number (N : ℕ) : Prop :=
  N >= 1000 ∧ N < 10000 ∧ (70000 + N) - (10 * N + 7) = 53208

theorem find_original_number (N : ℕ) (h : original_four_digit_number N) : N = 1865 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l163_16397


namespace NUMINAMATH_GPT_cos_diff_to_product_l163_16312

open Real

theorem cos_diff_to_product (a b : ℝ) : 
  cos (a + b) - cos (a - b) = -2 * sin a * sin b := 
  sorry

end NUMINAMATH_GPT_cos_diff_to_product_l163_16312


namespace NUMINAMATH_GPT_age_double_in_years_l163_16353

theorem age_double_in_years (S M X: ℕ) (h1: M = S + 22) (h2: S = 20) (h3: M + X = 2 * (S + X)) : X = 2 :=
by 
  sorry

end NUMINAMATH_GPT_age_double_in_years_l163_16353


namespace NUMINAMATH_GPT_hyperbola_condition_l163_16345

noncomputable def a_b_sum (a b : ℝ) : ℝ :=
  a + b

theorem hyperbola_condition
  (a b : ℝ)
  (h1 : a^2 - b^2 = 1)
  (h2 : abs (a - b) = 2)
  (h3 : a > b) :
  a_b_sum a b = 1/2 :=
sorry

end NUMINAMATH_GPT_hyperbola_condition_l163_16345


namespace NUMINAMATH_GPT_length_of_train_75_l163_16309

variable (L : ℝ) -- Length of the train in meters

-- Condition 1: The train crosses a bridge of length 150 m in 7.5 seconds
def crosses_bridge (L: ℝ) : Prop := (L + 150) / 7.5 = L / 2.5

-- Condition 2: The train crosses a lamp post in 2.5 seconds
def crosses_lamp (L: ℝ) : Prop := L / 2.5 = L / 2.5

theorem length_of_train_75 (L : ℝ) (h1 : crosses_bridge L) (h2 : crosses_lamp L) : L = 75 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_train_75_l163_16309


namespace NUMINAMATH_GPT_correct_time_fraction_l163_16346

theorem correct_time_fraction :
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  (correct_hours * correct_minutes_per_hour : ℝ) / (hours * minutes_per_hour) = (5 / 36 : ℝ) :=
by
  let hours := 24
  let correct_hours := 10
  let minutes_per_hour := 60
  let correct_minutes_per_hour := 20
  sorry

end NUMINAMATH_GPT_correct_time_fraction_l163_16346


namespace NUMINAMATH_GPT_prove_a_zero_l163_16350

-- Define two natural numbers a and b
variables (a b : ℕ)

-- Condition: For every natural number n, 2^n * a + b is a perfect square
def condition := ∀ n : ℕ, ∃ k : ℕ, 2^n * a + b = k^2

-- Statement to prove: a = 0
theorem prove_a_zero (h : condition a b) : a = 0 := sorry

end NUMINAMATH_GPT_prove_a_zero_l163_16350


namespace NUMINAMATH_GPT_longest_segment_CD_l163_16321

theorem longest_segment_CD
  (ABD_angle : ℝ) (ADB_angle : ℝ) (BDC_angle : ℝ) (CBD_angle : ℝ)
  (angle_proof_ABD : ABD_angle = 50)
  (angle_proof_ADB : ADB_angle = 40)
  (angle_proof_BDC : BDC_angle = 35)
  (angle_proof_CBD : CBD_angle = 70) :
  true := 
by
  sorry

end NUMINAMATH_GPT_longest_segment_CD_l163_16321


namespace NUMINAMATH_GPT_stream_speed_l163_16324

def upstream_time : ℝ := 4  -- time in hours
def downstream_time : ℝ := 4  -- time in hours
def upstream_distance : ℝ := 32  -- distance in km
def downstream_distance : ℝ := 72  -- distance in km

-- Speed equations based on given conditions
def effective_speed_upstream (vj vs : ℝ) : Prop := vj - vs = upstream_distance / upstream_time
def effective_speed_downstream (vj vs : ℝ) : Prop := vj + vs = downstream_distance / downstream_time

theorem stream_speed (vj vs : ℝ)  
  (h1 : effective_speed_upstream vj vs)
  (h2 : effective_speed_downstream vj vs) : 
  vs = 5 := sorry

end NUMINAMATH_GPT_stream_speed_l163_16324


namespace NUMINAMATH_GPT_evaluate_g_at_neg2_l163_16370

-- Definition of the polynomial g
def g (x : ℝ) : ℝ := 3 * x^5 - 20 * x^4 + 40 * x^3 - 25 * x^2 - 75 * x + 90

-- Statement to prove using the condition
theorem evaluate_g_at_neg2 : g (-2) = -596 := 
by 
   sorry

end NUMINAMATH_GPT_evaluate_g_at_neg2_l163_16370


namespace NUMINAMATH_GPT_problem_statement_l163_16396

theorem problem_statement (n : ℕ) : (-1 : ℤ) ^ n * (-1) ^ (2 * n + 1) * (-1) ^ (n + 1) = 1 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l163_16396


namespace NUMINAMATH_GPT_number_2120_in_33rd_group_l163_16322

def last_number_in_group (n : ℕ) := 2 * n * (n + 1)

theorem number_2120_in_33rd_group :
  ∃ n, n = 33 ∧ (last_number_in_group (n - 1) < 2120) ∧ (2120 <= last_number_in_group n) :=
sorry

end NUMINAMATH_GPT_number_2120_in_33rd_group_l163_16322


namespace NUMINAMATH_GPT_find_b_value_l163_16399

theorem find_b_value (x y b : ℝ) (h1 : (7 * x + b * y) / (x - 2 * y) = 29) (h2 : x / (2 * y) = 3 / 2) : b = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l163_16399


namespace NUMINAMATH_GPT_small_cube_edge_length_l163_16369

theorem small_cube_edge_length 
  (m n : ℕ)
  (h1 : 12 % m = 0) 
  (h2 : n = 12 / m) 
  (h3 : 6 * (n - 2)^2 = 12 * (n - 2)) 
  : m = 3 :=
by 
  sorry

end NUMINAMATH_GPT_small_cube_edge_length_l163_16369


namespace NUMINAMATH_GPT_megatek_manufacturing_percentage_proof_l163_16387

def megatek_employee_percentage
  (total_degrees_in_circle : ℕ)
  (manufacturing_degrees : ℕ) : ℚ :=
  (manufacturing_degrees / total_degrees_in_circle : ℚ) * 100

theorem megatek_manufacturing_percentage_proof (h1 : total_degrees_in_circle = 360)
  (h2 : manufacturing_degrees = 54) :
  megatek_employee_percentage total_degrees_in_circle manufacturing_degrees = 15 := 
by
  sorry

end NUMINAMATH_GPT_megatek_manufacturing_percentage_proof_l163_16387


namespace NUMINAMATH_GPT_two_legged_birds_count_l163_16330

def count_birds (b m i : ℕ) : Prop :=
  b + m + i = 300 ∧ 2 * b + 4 * m + 6 * i = 680 → b = 280

theorem two_legged_birds_count : ∃ b m i : ℕ, count_birds b m i :=
by
  have h1 : count_birds 280 0 20 := sorry
  exact ⟨280, 0, 20, h1⟩

end NUMINAMATH_GPT_two_legged_birds_count_l163_16330


namespace NUMINAMATH_GPT_members_in_both_sets_are_23_l163_16368

variable (U A B : Finset ℕ)
variable (count_U count_A count_B count_neither count_both : ℕ)

theorem members_in_both_sets_are_23 (hU : count_U = 192)
    (hA : count_A = 107) (hB : count_B = 49) (hNeither : count_neither = 59) :
    count_both = 23 :=
by
  sorry

end NUMINAMATH_GPT_members_in_both_sets_are_23_l163_16368


namespace NUMINAMATH_GPT_count_integers_satisfying_sqrt_condition_l163_16302

theorem count_integers_satisfying_sqrt_condition :
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℕ), (3 < Real.sqrt x ∧ Real.sqrt x < 5) → (9 < x ∧ x < 25) :=
by
  sorry

end NUMINAMATH_GPT_count_integers_satisfying_sqrt_condition_l163_16302


namespace NUMINAMATH_GPT_problem1_problem2_l163_16364

theorem problem1 :
  ( (1/2) ^ (-2) - 0.01 ^ (-1) + (-(1 + 1/7)) ^ (0)) = -95 := by
  sorry

theorem problem2 (x : ℝ) :
  (x - 2) * (x + 1) - (x - 1) ^ 2 = x - 3 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l163_16364


namespace NUMINAMATH_GPT_probability_top_card_is_joker_l163_16341

def deck_size : ℕ := 54
def joker_count : ℕ := 2

theorem probability_top_card_is_joker :
  (joker_count : ℝ) / (deck_size : ℝ) = 1 / 27 :=
by
  sorry

end NUMINAMATH_GPT_probability_top_card_is_joker_l163_16341


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_and_m_l163_16379

theorem arithmetic_sequence_common_difference_and_m (S : ℕ → ℤ) (a : ℕ → ℤ) (m d : ℕ) 
(h1 : S (m-1) = -2) (h2 : S m = 0) (h3 : S (m+1) = 3) :
  d = 1 ∧ m = 5 :=
by sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_and_m_l163_16379


namespace NUMINAMATH_GPT_overall_gain_percentage_l163_16376

theorem overall_gain_percentage (cost_A cost_B cost_C sp_A sp_B sp_C : ℕ)
  (hA : cost_A = 1000)
  (hB : cost_B = 3000)
  (hC : cost_C = 6000)
  (hsA : sp_A = 2000)
  (hsB : sp_B = 4500)
  (hsC : sp_C = 8000) :
  ((sp_A + sp_B + sp_C - (cost_A + cost_B + cost_C) : ℝ) / (cost_A + cost_B + cost_C) * 100) = 45 :=
by sorry

end NUMINAMATH_GPT_overall_gain_percentage_l163_16376


namespace NUMINAMATH_GPT_euler_sum_of_squares_euler_sum_of_quads_l163_16358

theorem euler_sum_of_squares :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^2 = π^2 / 6 := sorry

theorem euler_sum_of_quads :
  ∑' n : ℕ, 1 / (n.succ : ℚ)^4 = π^4 / 90 := sorry

end NUMINAMATH_GPT_euler_sum_of_squares_euler_sum_of_quads_l163_16358


namespace NUMINAMATH_GPT_afternoon_emails_l163_16331

theorem afternoon_emails (A : ℕ) (five_morning_emails : ℕ) (two_more : five_morning_emails + 2 = A) : A = 7 :=
by
  sorry

end NUMINAMATH_GPT_afternoon_emails_l163_16331


namespace NUMINAMATH_GPT_smallest_positive_integer_x_for_cube_l163_16332

theorem smallest_positive_integer_x_for_cube (x : ℕ) (h1 : 1512 = 2^3 * 3^3 * 7) (h2 : ∀ n : ℕ, n > 0 → ∃ k : ℕ, 1512 * n = k^3) : x = 49 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_x_for_cube_l163_16332


namespace NUMINAMATH_GPT_cos_36_is_correct_l163_16356

noncomputable def cos_36_eq : Prop :=
  let b := Real.cos (Real.pi * 36 / 180)
  let a := Real.cos (Real.pi * 72 / 180)
  (a = 2 * b^2 - 1) ∧ (b = (1 + Real.sqrt 5) / 4)

theorem cos_36_is_correct : cos_36_eq :=
by sorry

end NUMINAMATH_GPT_cos_36_is_correct_l163_16356


namespace NUMINAMATH_GPT_single_bacteria_colony_days_to_limit_l163_16303

theorem single_bacteria_colony_days_to_limit (n : ℕ) (h : ∀ t : ℕ, t ≤ 21 → (2 ^ t = 2 * 2 ^ (t - 1))) : n = 22 :=
by
  sorry

end NUMINAMATH_GPT_single_bacteria_colony_days_to_limit_l163_16303


namespace NUMINAMATH_GPT_rachel_plants_lamps_l163_16363

-- Define the conditions as types
def plants : Type := { fern1 : Prop // true } × { fern2 : Prop // true } × { cactus : Prop // true }
def lamps : Type := { yellow1 : Prop // true } × { yellow2 : Prop // true } × { blue1 : Prop // true } × { blue2 : Prop // true }

-- A function that counts the distribution of plants under lamps
noncomputable def count_ways (p : plants) (l : lamps) : ℕ :=
  -- Here we should define the function that counts the number of configurations, 
  -- but since we are only defining the problem here we'll skip this part.
  sorry

-- The statement to prove
theorem rachel_plants_lamps :
  ∀ (p : plants) (l : lamps), count_ways p l = 14 :=
by
  sorry

end NUMINAMATH_GPT_rachel_plants_lamps_l163_16363


namespace NUMINAMATH_GPT_range_b_values_l163_16344

theorem range_b_values (f g : ℝ → ℝ) (a b : ℝ) 
  (hf : ∀ x, f x = Real.exp x - 1) 
  (hg : ∀ x, g x = -x^2 + 4*x - 3) 
  (h : f a = g b) : 
  b ∈ Set.univ :=
by sorry

end NUMINAMATH_GPT_range_b_values_l163_16344


namespace NUMINAMATH_GPT_value_of_x_l163_16393

theorem value_of_x (x y : ℝ) (h1 : y = 2) (h2 : y = 1 / (4 * x + 2)) : x = -3 / 8 := 
by
  sorry

end NUMINAMATH_GPT_value_of_x_l163_16393


namespace NUMINAMATH_GPT_range_of_a_maximum_of_z_l163_16307

-- Problem 1
theorem range_of_a (a b : ℝ) (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) :
  -2 < a ∧ a < 1 :=
sorry

-- Problem 2
theorem maximum_of_z (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 9) :
  ∃ z, z = a * b^2 ∧ z ≤ 27 :=
sorry


end NUMINAMATH_GPT_range_of_a_maximum_of_z_l163_16307


namespace NUMINAMATH_GPT_matrix_power_50_l163_16398

-- Defining the matrix A.
def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![5, 1], 
    ![-12, -3]]

-- Statement of the theorem
theorem matrix_power_50 :
  A ^ 50 = ![![301, 50], 
               ![-900, -301]] :=
by
  sorry

end NUMINAMATH_GPT_matrix_power_50_l163_16398


namespace NUMINAMATH_GPT_factor_expression_l163_16337

theorem factor_expression (x : ℝ) : 6 * x ^ 3 - 54 * x = 6 * x * (x + 3) * (x - 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_factor_expression_l163_16337


namespace NUMINAMATH_GPT_johns_total_pay_l163_16366

-- Define the given conditions
def lastYearBonus : ℝ := 10000
def CAGR : ℝ := 0.05
def numYears : ℕ := 1
def projectsCompleted : ℕ := 8
def bonusPerProject : ℝ := 2000
def thisYearSalary : ℝ := 200000

-- Define the calculation for the first part of the bonus using the CAGR formula
def firstPartBonus (presentValue : ℝ) (growthRate : ℝ) (years : ℕ) : ℝ :=
  presentValue * (1 + growthRate)^years

-- Define the calculation for the second part of the bonus
def secondPartBonus (numProjects : ℕ) (bonusPerProject : ℝ) : ℝ :=
  numProjects * bonusPerProject

-- Define the total pay calculation
def totalPay (salary : ℝ) (bonus1 : ℝ) (bonus2 : ℝ) : ℝ :=
  salary + bonus1 + bonus2

-- The proof statement, given the conditions, prove the total pay is $226,500
theorem johns_total_pay : totalPay thisYearSalary (firstPartBonus lastYearBonus CAGR numYears) (secondPartBonus projectsCompleted bonusPerProject) = 226500 := 
by
  -- insert proof here
  sorry

end NUMINAMATH_GPT_johns_total_pay_l163_16366


namespace NUMINAMATH_GPT_prove_range_of_p_l163_16354

noncomputable def f (x : ℝ) : ℝ := -x^2 + 2 * x - 1

def A (x : ℝ) : Prop := x > 2
def no_pre_image_in_A (p : ℝ) : Prop := ∀ x, A x → f x ≠ p

theorem prove_range_of_p (p : ℝ) : no_pre_image_in_A p ↔ p > -1 := by
  sorry

end NUMINAMATH_GPT_prove_range_of_p_l163_16354


namespace NUMINAMATH_GPT_min_value_Px_Py_l163_16357

def P (τ : ℝ) : ℝ := (τ + 1)^3

theorem min_value_Px_Py (x y : ℝ) (h : x + y = 0) : P x + P y = 2 :=
sorry

end NUMINAMATH_GPT_min_value_Px_Py_l163_16357


namespace NUMINAMATH_GPT_prove_f2_l163_16384

def func_condition (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x y : ℝ, f (f x + y) = f (x ^ 2 - y) + 2 * c * f x * y

theorem prove_f2 (c : ℝ) (f : ℝ → ℝ)
  (hf : func_condition f c) :
  (f 2 = 0 ∨ f 2 = 4) ∧ (2 * (if f 2 = 0 then 4 else if f 2 = 4 then 4 else 0) = 8) :=
by {
  sorry
}

end NUMINAMATH_GPT_prove_f2_l163_16384


namespace NUMINAMATH_GPT_complement_N_subset_M_l163_16386

-- Definitions for the sets M and N
def M : Set ℝ := {x | x * (x - 3) < 0}
def N : Set ℝ := {x | x < 1 ∨ x ≥ 3}

-- Complement of N in ℝ
def complement_N : Set ℝ := {x | ¬(x < 1 ∨ x ≥ 3)}

-- The theorem stating that complement_N is a subset of M
theorem complement_N_subset_M : complement_N ⊆ M :=
by
  sorry

end NUMINAMATH_GPT_complement_N_subset_M_l163_16386


namespace NUMINAMATH_GPT_luke_played_rounds_l163_16355

theorem luke_played_rounds (total_points : ℕ) (points_per_round : ℕ) (result : ℕ)
  (h1 : total_points = 154)
  (h2 : points_per_round = 11)
  (h3 : result = total_points / points_per_round) :
  result = 14 :=
by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_luke_played_rounds_l163_16355


namespace NUMINAMATH_GPT_wait_time_at_least_8_l163_16389

-- Define the conditions
variables (p₀ p : ℝ) (r x : ℝ)

-- Given conditions
def initial_BAC := p₀ = 89
def BAC_after_2_hours := p = 61
def BAC_decrease := p = p₀ * (Real.exp (r * x))
def decrease_in_2_hours := p = 89 * (Real.exp (r * 2))

-- The main goal to prove the time required is at least 8 hours
theorem wait_time_at_least_8 (h1 : p₀ = 89) (h2 : p = 61) (h3 : p = p₀ * Real.exp (r * x)) (h4 : 61 = 89 * Real.exp (2 * r)) : 
  ∃ x, 89 * Real.exp (r * x) < 20 ∧ x ≥ 8 :=
sorry

end NUMINAMATH_GPT_wait_time_at_least_8_l163_16389


namespace NUMINAMATH_GPT_max_intersections_circle_quadrilateral_max_intersections_correct_l163_16392

-- Define the intersection property of a circle and a line segment
def max_intersections_per_side (circle : Type) (line_segment : Type) : ℕ := 2

-- Define a quadrilateral as a shape having four sides
def sides_of_quadrilateral : ℕ := 4

-- The theorem stating the maximum number of intersection points between a circle and a quadrilateral
theorem max_intersections_circle_quadrilateral (circle : Type) (quadrilateral : Type) : Prop :=
  max_intersections_per_side circle quadrilateral * sides_of_quadrilateral = 8

-- Proof is skipped with 'sorry'
theorem max_intersections_correct (circle : Type) (quadrilateral : Type) :
  max_intersections_circle_quadrilateral circle quadrilateral :=
by
  sorry

end NUMINAMATH_GPT_max_intersections_circle_quadrilateral_max_intersections_correct_l163_16392


namespace NUMINAMATH_GPT_range_of_a_iff_l163_16313

def cubic_inequality (x : ℝ) : Prop := x^3 + 3 * x^2 - x - 3 > 0

def quadratic_inequality (x a : ℝ) : Prop := x^2 - 2 * a * x - 1 ≤ 0

def integer_solution_condition (x : ℤ) (a : ℝ) : Prop := 
  x^3 + 3 * x^2 - x - 3 > 0 ∧ x^2 - 2 * a * x - 1 ≤ 0

def range_of_a (a : ℝ) : Prop := (3 / 4 : ℝ) ≤ a ∧ a < (4 / 3 : ℝ)

theorem range_of_a_iff : 
  (∃ x : ℤ, integer_solution_condition x a) ↔ range_of_a a := 
sorry

end NUMINAMATH_GPT_range_of_a_iff_l163_16313


namespace NUMINAMATH_GPT_C_plus_D_l163_16338

theorem C_plus_D (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → C / (x - 3) + D * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) : 
  C + D = -10 := by
  sorry

end NUMINAMATH_GPT_C_plus_D_l163_16338


namespace NUMINAMATH_GPT_number_with_specific_places_l163_16335

theorem number_with_specific_places :
  ∃ (n : Real), 
    (n / 10 % 10 = 6) ∧ -- tens place
    (n / 1 % 10 = 0) ∧  -- ones place
    (n * 10 % 10 = 0) ∧  -- tenths place
    (n * 100 % 10 = 6) →  -- hundredths place
    n = 60.06 :=
by
  sorry

end NUMINAMATH_GPT_number_with_specific_places_l163_16335


namespace NUMINAMATH_GPT_total_combined_rainfall_l163_16320

theorem total_combined_rainfall :
  let monday_hours := 5
  let monday_rate := 1
  let tuesday_hours := 3
  let tuesday_rate := 1.5
  let wednesday_hours := 4
  let wednesday_rate := 2 * monday_rate
  let thursday_hours := 6
  let thursday_rate := tuesday_rate / 2
  let friday_hours := 2
  let friday_rate := 1.5 * wednesday_rate
  let monday_rain := monday_hours * monday_rate
  let tuesday_rain := tuesday_hours * tuesday_rate
  let wednesday_rain := wednesday_hours * wednesday_rate
  let thursday_rain := thursday_hours * thursday_rate
  let friday_rain := friday_hours * friday_rate
  monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain = 28 := by
  sorry

end NUMINAMATH_GPT_total_combined_rainfall_l163_16320


namespace NUMINAMATH_GPT_sin_identity_l163_16342

open Real

noncomputable def alpha : ℝ := π  -- since we are considering angles in radians

theorem sin_identity (h1 : sin α = 3/5) (h2 : π/2 < α ∧ α < 3 * π / 2) :
  sin (5 * π / 2 - α) = -4 / 5 :=
by sorry

end NUMINAMATH_GPT_sin_identity_l163_16342


namespace NUMINAMATH_GPT_johns_initial_money_l163_16306

theorem johns_initial_money (X : ℝ) 
  (h₁ : (1 / 2) * X + (1 / 3) * X + (1 / 10) * X + 10 = X) : X = 150 :=
sorry

end NUMINAMATH_GPT_johns_initial_money_l163_16306


namespace NUMINAMATH_GPT_range_of_function_l163_16391

theorem range_of_function : 
  (∀ x, (Real.pi / 4) ≤ x ∧ x ≤ (Real.pi / 2) → 
   1 ≤ (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x ∧ 
    (Real.sin x)^2 + Real.sqrt 3 * Real.sin x * Real.cos x ≤ 3 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_function_l163_16391


namespace NUMINAMATH_GPT_find_interest_rate_l163_16329

theorem find_interest_rate 
    (P : ℝ) (T : ℝ) (known_rate : ℝ) (diff : ℝ) (R : ℝ) :
    P = 7000 → T = 2 → known_rate = 0.18 → diff = 840 → (P * known_rate * T - (P * (R/100) * T) = diff) → R = 12 :=
by
  intros P_eq T_eq kr_eq diff_eq interest_eq
  simp only [P_eq, T_eq, kr_eq, diff_eq] at interest_eq
-- Solving equation is not required
  sorry

end NUMINAMATH_GPT_find_interest_rate_l163_16329


namespace NUMINAMATH_GPT_minimum_value_frac_l163_16301

theorem minimum_value_frac (a b : ℝ) (h₁ : 2 * a - b + 2 * 0 = 0) 
  (h₂ : a > 0) (h₃ : b > 0) (h₄ : a + b = 1) : 
  (1 / a) + (1 / b) = 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_frac_l163_16301


namespace NUMINAMATH_GPT_no_solution_abs_eq_l163_16315

theorem no_solution_abs_eq : ∀ y : ℝ, |y - 2| ≠ |y - 1| + |y - 4| :=
by
  intros y
  sorry

end NUMINAMATH_GPT_no_solution_abs_eq_l163_16315


namespace NUMINAMATH_GPT_eq_factorial_sum_l163_16372

theorem eq_factorial_sum (k l m n : ℕ) (hk : k > 0) (hl : l > 0) (hm : m > 0) (hn : n > 0) :
  (1 / (Nat.factorial k : ℝ) + 1 / (Nat.factorial l : ℝ) + 1 / (Nat.factorial m : ℝ) = 1 / (Nat.factorial n : ℝ))
  ↔ (k = 3 ∧ l = 3 ∧ m = 3 ∧ n = 2) :=
by
  sorry

end NUMINAMATH_GPT_eq_factorial_sum_l163_16372


namespace NUMINAMATH_GPT_solution_set_of_inequality_l163_16388

theorem solution_set_of_inequality (f : ℝ → ℝ) :
  (∀ x, f x = f (-x)) →
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ x1 ≤ x2 → f x2 ≤ f x1) →
  (f 1 = 0) →
  {x : ℝ | f (x - 3) ≥ 0} = {x : ℝ | 2 ≤ x ∧ x ≤ 4} :=
by
  intros h_even h_mono h_f1
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l163_16388


namespace NUMINAMATH_GPT_divides_iff_l163_16310

open Int

theorem divides_iff (n m : ℤ) : (9 ∣ (2 * n + 5 * m)) ↔ (9 ∣ (5 * n + 8 * m)) := 
sorry

end NUMINAMATH_GPT_divides_iff_l163_16310


namespace NUMINAMATH_GPT_smallest_integer_base_cube_l163_16347

theorem smallest_integer_base_cube (b : ℤ) (h1 : b > 5) (h2 : ∃ k : ℤ, 1 * b + 2 = k^3) : b = 6 :=
sorry

end NUMINAMATH_GPT_smallest_integer_base_cube_l163_16347


namespace NUMINAMATH_GPT_length_of_ST_l163_16319

theorem length_of_ST (PQ PS : ℝ) (ST : ℝ) (hPQ : PQ = 8) (hPS : PS = 7) 
  (h_area_eq : (1 / 2) * PQ * (PS * (1 / PS) * 8) = PQ * PS) : 
  ST = 2 * Real.sqrt 65 := 
by
  -- proof steps (to be written)
  sorry

end NUMINAMATH_GPT_length_of_ST_l163_16319


namespace NUMINAMATH_GPT_abs_neg_five_halves_l163_16327

theorem abs_neg_five_halves : abs (-5 / 2) = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_abs_neg_five_halves_l163_16327


namespace NUMINAMATH_GPT_eval_diff_squares_l163_16360

theorem eval_diff_squares : 81^2 - 49^2 = 4160 :=
by
  sorry

end NUMINAMATH_GPT_eval_diff_squares_l163_16360


namespace NUMINAMATH_GPT_train_speeds_l163_16333

noncomputable def c1 : ℝ := sorry  -- speed of the passenger train in km/min
noncomputable def c2 : ℝ := sorry  -- speed of the freight train in km/min
noncomputable def c3 : ℝ := sorry  -- speed of the express train in km/min

def conditions : Prop :=
  (5 / c1 + 5 / c2 = 15) ∧
  (5 / c2 + 5 / c3 = 11) ∧
  (c2 ≤ c1) ∧
  (c3 ≤ 2.5)

-- The theorem to be proved
theorem train_speeds :
  conditions →
  (40 / 60 ≤ c1 ∧ c1 ≤ 50 / 60) ∧ 
  (100 / 3 / 60 ≤ c2 ∧ c2 ≤ 40 / 60) ∧ 
  (600 / 7 / 60 ≤ c3 ∧ c3 ≤ 150 / 60) :=
sorry

end NUMINAMATH_GPT_train_speeds_l163_16333


namespace NUMINAMATH_GPT_function_increasing_value_of_a_function_decreasing_value_of_a_l163_16305

-- Part 1: Prove that if \( f(x) = x^3 - ax - 1 \) is increasing on the interval \( (1, +\infty) \), then \( a \leq 3 \)
theorem function_increasing_value_of_a (a : ℝ) :
  (∀ x > 1, 3 * x^2 - a ≥ 0) → a ≤ 3 := by
  sorry

-- Part 2: Prove that if the decreasing interval of \( f(x) = x^3 - ax - 1 \) is \( (-1, 1) \), then \( a = 3 \)
theorem function_decreasing_value_of_a (a : ℝ) :
  (∀ x, -1 < x ∧ x < 1 → 3 * x^2 - a < 0) ∧ (3 * (-1)^2 - a = 0 ∧ 3 * (1)^2 - a = 0) → a = 3 := by
  sorry

end NUMINAMATH_GPT_function_increasing_value_of_a_function_decreasing_value_of_a_l163_16305


namespace NUMINAMATH_GPT_animals_total_l163_16394

-- Given definitions and conditions
def ducks : ℕ := 25
def rabbits : ℕ := 8
def chickens := 4 * ducks

-- Proof statement
theorem animals_total (h1 : chickens = 4 * ducks)
                     (h2 : ducks - 17 = rabbits)
                     (h3 : rabbits = 8) :
  chickens + ducks + rabbits = 133 := by
  sorry

end NUMINAMATH_GPT_animals_total_l163_16394


namespace NUMINAMATH_GPT_problem_integer_condition_l163_16380

theorem problem_integer_condition (a : ℤ) (h1 : 0 ≤ a ∧ a ≤ 14)
  (h2 : (235935623 * 74^0 + 2 * 74^1 + 6 * 74^2 + 5 * 74^3 + 3 * 74^4 + 9 * 74^5 + 
         5 * 74^6 + 3 * 74^7 + 2 * 74^8 - a) % 15 = 0) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_integer_condition_l163_16380


namespace NUMINAMATH_GPT_reciprocal_neg_5_l163_16326

theorem reciprocal_neg_5 : ∃ x : ℚ, -5 * x = 1 ∧ x = -1/5 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_neg_5_l163_16326


namespace NUMINAMATH_GPT_combined_operation_l163_16383

def f (x : ℚ) := (3 / 4) * x
def g (x : ℚ) := (5 / 3) * x

theorem combined_operation (x : ℚ) : g (f x) = (5 / 4) * x :=
by
    unfold f g
    sorry

end NUMINAMATH_GPT_combined_operation_l163_16383


namespace NUMINAMATH_GPT_solution_set_of_inequality_range_of_a_for_gx_zero_l163_16395

-- Define f(x) and g(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) + abs (x + a)

def g (x : ℝ) (a : ℝ) : ℝ := f x a - abs (3 + a)

-- The first Lean statement
theorem solution_set_of_inequality (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, f x a > 6 ↔ x < -4 ∨ (-3 < x ∧ x < 1) ∨ 2 < x := by
  sorry

-- The second Lean statement
theorem range_of_a_for_gx_zero (a : ℝ) :
  (∃ x : ℝ, g x a = 0) ↔ a ≥ -2 := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_range_of_a_for_gx_zero_l163_16395


namespace NUMINAMATH_GPT_find_x_l163_16377

theorem find_x (x : ℚ) : (8 + 12 + 24) / 3 = (16 + x) / 2 → x = 40 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_x_l163_16377


namespace NUMINAMATH_GPT_gifts_from_Pedro_l163_16340

theorem gifts_from_Pedro (gifts_from_Emilio gifts_from_Jorge total_gifts : ℕ)
  (h1 : gifts_from_Emilio = 11)
  (h2 : gifts_from_Jorge = 6)
  (h3 : total_gifts = 21) :
  total_gifts - (gifts_from_Emilio + gifts_from_Jorge) = 4 := by
  sorry

end NUMINAMATH_GPT_gifts_from_Pedro_l163_16340


namespace NUMINAMATH_GPT_relationship_among_a_ab_ab2_l163_16362

theorem relationship_among_a_ab_ab2 (a b : ℝ) (h_a : a < 0) (h_b1 : -1 < b) (h_b2 : b < 0) :
  a < a * b ∧ a * b < a * b^2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_a_ab_ab2_l163_16362


namespace NUMINAMATH_GPT_panels_per_home_panels_needed_per_home_l163_16352

theorem panels_per_home (P : ℕ) (total_homes : ℕ) (shortfall : ℕ) (homes_installed : ℕ) :
  total_homes = 20 →
  shortfall = 50 →
  homes_installed = 15 →
  (P - shortfall) / homes_installed = P / total_homes →
  P = 200 :=
by
  intro h1 h2 h3 h4
  sorry

theorem panels_needed_per_home :
  (200 / 20) = 10 :=
by
  sorry

end NUMINAMATH_GPT_panels_per_home_panels_needed_per_home_l163_16352


namespace NUMINAMATH_GPT_point_C_number_l163_16365

theorem point_C_number (B C: ℝ) (h1 : B = 3) (h2 : |C - B| = 2) :
  C = 1 ∨ C = 5 := 
by {
  sorry
}

end NUMINAMATH_GPT_point_C_number_l163_16365


namespace NUMINAMATH_GPT_smallest_digit_never_in_units_place_of_odd_numbers_l163_16323

-- Define the set of units digits of odd numbers
def units_digits_odd_numbers : Set ℕ := {1, 3, 5, 7, 9}

-- Prove that the smallest digit not found in the units place of an odd number is 0
theorem smallest_digit_never_in_units_place_of_odd_numbers : ∀ d, d ∉ units_digits_odd_numbers → d = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_digit_never_in_units_place_of_odd_numbers_l163_16323


namespace NUMINAMATH_GPT_large_paintings_count_l163_16382

-- Define the problem conditions
def paint_per_large : Nat := 3
def paint_per_small : Nat := 2
def small_paintings : Nat := 4
def total_paint : Nat := 17

-- Question to find number of large paintings (L)
theorem large_paintings_count :
  ∃ L : Nat, (paint_per_large * L + paint_per_small * small_paintings = total_paint) → L = 3 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_large_paintings_count_l163_16382


namespace NUMINAMATH_GPT_fraction_of_male_gerbils_is_correct_l163_16375

def total_pets := 90
def total_gerbils := 66
def total_hamsters := total_pets - total_gerbils
def fraction_hamsters_male := 1/3
def total_males := 25
def male_hamsters := fraction_hamsters_male * total_hamsters
def male_gerbils := total_males - male_hamsters
def fraction_gerbils_male := male_gerbils / total_gerbils

theorem fraction_of_male_gerbils_is_correct : fraction_gerbils_male = 17 / 66 := by
  sorry

end NUMINAMATH_GPT_fraction_of_male_gerbils_is_correct_l163_16375


namespace NUMINAMATH_GPT_count_house_numbers_l163_16378

def isPrime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def twoDigitPrimesBetween40And60 : List ℕ :=
  [41, 43, 47, 53, 59]

theorem count_house_numbers : 
  ∃ n : ℕ, n = 20 ∧ 
  ∀ (AB CD : ℕ), 
  AB ∈ twoDigitPrimesBetween40And60 → 
  CD ∈ twoDigitPrimesBetween40And60 → 
  AB ≠ CD → 
  true :=
by
  sorry

end NUMINAMATH_GPT_count_house_numbers_l163_16378


namespace NUMINAMATH_GPT_Eugene_buys_two_pairs_of_shoes_l163_16381

theorem Eugene_buys_two_pairs_of_shoes :
  let tshirt_price : ℕ := 20
  let pants_price : ℕ := 80
  let shoes_price : ℕ := 150
  let discount_rate : ℕ := 10
  let discounted_price (price : ℕ) := price - (price * discount_rate / 100)
  let total_price (count1 count2 count3 : ℕ) (price1 price2 price3 : ℕ) :=
    (count1 * price1) + (count2 * price2) + (count3 * price3)
  let total_amount_paid : ℕ := 558
  let tshirts_bought : ℕ := 4
  let pants_bought : ℕ := 3
  let amount_left := total_amount_paid - discounted_price (tshirts_bought * tshirt_price + pants_bought * pants_price)
  let shoes_bought := amount_left / discounted_price shoes_price
  shoes_bought = 2 := 
sorry

end NUMINAMATH_GPT_Eugene_buys_two_pairs_of_shoes_l163_16381


namespace NUMINAMATH_GPT_original_decimal_number_l163_16359

theorem original_decimal_number (x : ℝ) (h₁ : 0 < x) (h₂ : 100 * x = 9 * (1 / x)) : x = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_original_decimal_number_l163_16359


namespace NUMINAMATH_GPT_road_trip_ratio_l163_16361

-- Problem Definitions
variable (x d3 total grand_total : ℕ)
variable (hx1 : total = x + 2 * x + d3 + 2 * (x + 2 * x + d3))
variable (hx2 : d3 = 40)
variable (hx3 : total = 560)
variable (hx4 : grand_total = d3 / x)

-- Proof Statement
theorem road_trip_ratio (hx1 : total = x + 2 * x + d3 + 2 * (x + 2 * x + d3)) 
  (hx2 : d3 = 40) (hx3 : total = 560) : grand_total = 9 / 11 := by
  sorry

end NUMINAMATH_GPT_road_trip_ratio_l163_16361


namespace NUMINAMATH_GPT_average_pages_per_hour_l163_16390

theorem average_pages_per_hour 
  (P : ℕ) (H : ℕ) (hP : P = 30000) (hH : H = 150) : 
  P / H = 200 := 
by 
  sorry

end NUMINAMATH_GPT_average_pages_per_hour_l163_16390


namespace NUMINAMATH_GPT_fraction_taken_by_kiley_l163_16317

-- Define the constants and conditions
def total_crayons : ℕ := 48
def remaining_crayons_after_joe : ℕ := 18

-- Define the main statement to be proven
theorem fraction_taken_by_kiley (f : ℚ) : 
  (48 - (48 * f)) / 2 = 18 → f = 1 / 4 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_fraction_taken_by_kiley_l163_16317


namespace NUMINAMATH_GPT_tangent_line_count_l163_16304

noncomputable def circles_tangent_lines (r1 r2 d : ℝ) : ℕ :=
if d = |r1 - r2| then 1 else 0 -- Define the function based on the problem statement

theorem tangent_line_count :
  circles_tangent_lines 4 5 3 = 1 := 
by
  -- Placeholder for the proof, which we are skipping as per instructions
  sorry

end NUMINAMATH_GPT_tangent_line_count_l163_16304


namespace NUMINAMATH_GPT_cindy_correct_answer_l163_16325

-- Define the conditions given in the problem
def x : ℤ := 272 -- Cindy's miscalculated number

-- The outcome of Cindy's incorrect operation
def cindy_incorrect (x : ℤ) : Prop := (x - 7) = 53 * 5

-- The outcome of Cindy's correct operation
def cindy_correct (x : ℤ) : ℤ := (x - 5) / 7

-- The main theorem to prove
theorem cindy_correct_answer : cindy_incorrect x → cindy_correct x = 38 :=
by
  sorry

end NUMINAMATH_GPT_cindy_correct_answer_l163_16325


namespace NUMINAMATH_GPT_find_a6_l163_16311

variable {a : ℕ → ℝ}

-- Define what it means for a sequence to be arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
def given_condition (a : ℕ → ℝ) (d : ℝ) : Prop :=
  2 * (a 1 + a 3 + a 5) + 3 * (a 8 + a 10) = 36

theorem find_a6 (d : ℝ) :
  is_arithmetic_sequence a d →
  given_condition a d →
  a 6 = 3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_find_a6_l163_16311


namespace NUMINAMATH_GPT_num_combinations_L_shape_l163_16336

theorem num_combinations_L_shape (n : ℕ) (k : ℕ) (grid_size : ℕ) (L_shape_blocks : ℕ) 
  (h1 : n = 6) (h2 : k = 4) (h3 : grid_size = 36) (h4 : L_shape_blocks = 4) : 
  ∃ (total_combinations : ℕ), total_combinations = 1800 := by
  sorry

end NUMINAMATH_GPT_num_combinations_L_shape_l163_16336


namespace NUMINAMATH_GPT_mr_smith_children_l163_16374

noncomputable def gender_probability (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let equal_gender_ways := Nat.choose n (n / 2)
  let favourable_outcomes := total_outcomes - equal_gender_ways
  favourable_outcomes / total_outcomes

theorem mr_smith_children (n : ℕ) (h : n = 8) : 
  gender_probability n = 93 / 128 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_mr_smith_children_l163_16374


namespace NUMINAMATH_GPT_power_modulo_l163_16328

theorem power_modulo (h : 3 ^ 4 ≡ 1 [MOD 10]) : 3 ^ 2023 ≡ 7 [MOD 10] :=
by
  sorry

end NUMINAMATH_GPT_power_modulo_l163_16328


namespace NUMINAMATH_GPT_proof_problem_l163_16367

open Set

variable {R : Set ℝ} (A B : Set ℝ) (complement_B : Set ℝ)

-- Defining set A
def setA : Set ℝ := { x | 1 < x ∧ x < 3 }

-- Defining set B based on the given functional relationship
def setB : Set ℝ := { x | 2 < x } 

-- Defining the complement of set B (in the universal set R)
def complementB : Set ℝ := { x | x ≤ 2 }

-- The intersection we need to prove is equivalent to the given answer
def intersection_result : Set ℝ := { x | 1 < x ∧ x ≤ 2 }

-- The theorem statement (no proof)
theorem proof_problem : setA ∩ complementB = intersection_result := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l163_16367


namespace NUMINAMATH_GPT_cost_price_of_article_l163_16373

theorem cost_price_of_article (C : ℝ) (SP : ℝ) (C_new : ℝ) (SP_new : ℝ) :
  SP = 1.05 * C →
  C_new = 0.95 * C →
  SP_new = SP - 3 →
  SP_new = 1.045 * C →
  C = 600 :=
by
  intro h1 h2 h3 h4
  -- statement to be proved
  sorry

end NUMINAMATH_GPT_cost_price_of_article_l163_16373


namespace NUMINAMATH_GPT_average_of_remaining_two_l163_16316

theorem average_of_remaining_two (S S3 : ℚ) (h1 : S / 5 = 6) (h2 : S3 / 3 = 4) : (S - S3) / 2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_l163_16316


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l163_16300

theorem common_ratio_of_geometric_series (a₁ q : ℝ) 
  (S_3 : ℝ) (S_2 : ℝ) 
  (hS3 : S_3 = a₁ * (1 - q^3) / (1 - q)) 
  (hS2 : S_2 = a₁ * (1 - q^2) / (1 - q)) 
  (h_ratio : S_3 / S_2 = 3 / 2) :
  q = 1 ∨ q = -1/2 :=
by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l163_16300
