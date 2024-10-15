import Mathlib

namespace NUMINAMATH_GPT_find_angle_sum_l1133_113345

theorem find_angle_sum (c d : ℝ) (hc : 0 < c ∧ c < π/2) (hd : 0 < d ∧ d < π/2)
    (h1 : 4 * (Real.cos c)^2 + 3 * (Real.sin d)^2 = 1)
    (h2 : 4 * Real.sin (2 * c) = 3 * Real.cos (2 * d)) :
    2 * c + 3 * d = π / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_sum_l1133_113345


namespace NUMINAMATH_GPT_collinear_points_l1133_113371

-- Define collinear points function
def collinear (x1 y1 z1 x2 y2 z2 x3 y3 z3: ℝ) : Prop :=
  ∀ (a b c : ℝ), a * (y2 - y1) * (z3 - z1) + b * (z2 - z1) * (x3 - x1) + c * (x2 - x1) * (y3 - y1) = 0

-- Problem statement
theorem collinear_points (a b : ℝ)
  (h : collinear 2 a b a 3 b a b 4) :
  a + b = -2 :=
sorry

end NUMINAMATH_GPT_collinear_points_l1133_113371


namespace NUMINAMATH_GPT_carB_highest_avg_speed_l1133_113302

-- Define the distances and times for each car
def distanceA : ℕ := 715
def timeA : ℕ := 11
def distanceB : ℕ := 820
def timeB : ℕ := 12
def distanceC : ℕ := 950
def timeC : ℕ := 14

-- Define the average speeds
def avgSpeedA : ℚ := distanceA / timeA
def avgSpeedB : ℚ := distanceB / timeB
def avgSpeedC : ℚ := distanceC / timeC

theorem carB_highest_avg_speed : avgSpeedB > avgSpeedA ∧ avgSpeedB > avgSpeedC :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_carB_highest_avg_speed_l1133_113302


namespace NUMINAMATH_GPT_percentage_answered_first_correctly_l1133_113300

variable (A B C D : ℝ)

-- Conditions translated to Lean
variable (hB : B = 0.65)
variable (hC : C = 0.20)
variable (hD : D = 0.60)

-- Statement to prove
theorem percentage_answered_first_correctly (hI : A + B - D = 1 - C) : A = 0.75 := by
  -- import conditions
  rw [hB, hC, hD] at hI
  -- solve the equation
  sorry

end NUMINAMATH_GPT_percentage_answered_first_correctly_l1133_113300


namespace NUMINAMATH_GPT_find_length_of_second_train_l1133_113387

def length_of_second_train (L : ℝ) : Prop :=
  let speed_first_train := 33.33 -- Speed in m/s
  let speed_second_train := 22.22 -- Speed in m/s
  let relative_speed := speed_first_train + speed_second_train -- Relative speed in m/s
  let time_to_cross := 9 -- time in seconds
  let length_first_train := 260 -- Length in meters
  length_first_train + L = relative_speed * time_to_cross

theorem find_length_of_second_train : length_of_second_train 239.95 :=
by
  admit -- To be completed (proof)

end NUMINAMATH_GPT_find_length_of_second_train_l1133_113387


namespace NUMINAMATH_GPT_division_of_mixed_numbers_l1133_113350

noncomputable def mixed_to_improper (n : ℕ) (a b : ℕ) : ℚ :=
  n + (a / b)

theorem division_of_mixed_numbers : 
  (mixed_to_improper 7 1 3) / (mixed_to_improper 2 1 2) = 44 / 15 :=
by
  sorry

end NUMINAMATH_GPT_division_of_mixed_numbers_l1133_113350


namespace NUMINAMATH_GPT_cooking_time_eq_80_l1133_113343

-- Define the conditions
def hushpuppies_per_guest : Nat := 5
def number_of_guests : Nat := 20
def hushpuppies_per_batch : Nat := 10
def time_per_batch : Nat := 8

-- Calculate total number of hushpuppies needed
def total_hushpuppies : Nat := hushpuppies_per_guest * number_of_guests

-- Calculate number of batches needed
def number_of_batches : Nat := total_hushpuppies / hushpuppies_per_batch

-- Calculate total time needed
def total_time_needed : Nat := number_of_batches * time_per_batch

-- Statement to prove the correctness
theorem cooking_time_eq_80 : total_time_needed = 80 := by
  sorry

end NUMINAMATH_GPT_cooking_time_eq_80_l1133_113343


namespace NUMINAMATH_GPT_swimmer_speed_in_still_water_l1133_113368

variable (distance : ℝ) (time : ℝ) (current_speed : ℝ) (swimmer_speed_still_water : ℝ)

-- Define the given conditions
def conditions := 
  distance = 8 ∧
  time = 5 ∧
  current_speed = 1.4 ∧
  (distance / time = swimmer_speed_still_water - current_speed)

-- The theorem we want to prove
theorem swimmer_speed_in_still_water : 
  conditions distance time current_speed swimmer_speed_still_water → 
  swimmer_speed_still_water = 3 := 
by 
  -- Skipping the actual proof
  sorry

end NUMINAMATH_GPT_swimmer_speed_in_still_water_l1133_113368


namespace NUMINAMATH_GPT_ratio_perimeter_to_breadth_l1133_113314

-- Definitions of the conditions
def area_of_rectangle (length breadth : ℝ) := length * breadth
def perimeter_of_rectangle (length breadth : ℝ) := 2 * (length + breadth)

-- The problem statement: prove the ratio of perimeter to breadth
theorem ratio_perimeter_to_breadth (L B : ℝ) (hL : L = 18) (hA : area_of_rectangle L B = 216) :
  (perimeter_of_rectangle L B) / B = 5 :=
by 
  -- Given definitions and conditions, we skip the proof.
  sorry

end NUMINAMATH_GPT_ratio_perimeter_to_breadth_l1133_113314


namespace NUMINAMATH_GPT_sum_of_roots_l1133_113356

variable {h b : ℝ}
variable {x₁ x₂ : ℝ}

-- Definition of the distinct property
def distinct (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂

-- Definition of the original equations given the conditions
def satisfies_equation (x : ℝ) (h b : ℝ) : Prop := 3 * x^2 - h * x = b

-- Main theorem statement translating the given mathematical problem
theorem sum_of_roots (h b : ℝ) (x₁ x₂ : ℝ) (h₁ : satisfies_equation x₁ h b) 
  (h₂ : satisfies_equation x₂ h b) (h₃ : distinct x₁ x₂) : x₁ + x₂ = h / 3 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1133_113356


namespace NUMINAMATH_GPT_sphere_radius_eq_cylinder_radius_l1133_113396

theorem sphere_radius_eq_cylinder_radius
  (r h d : ℝ) (h_eq_d : h = 16) (d_eq_h : d = 16)
  (sphere_surface_area_eq_cylinder : 4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h) : 
  r = 8 :=
by
  sorry

end NUMINAMATH_GPT_sphere_radius_eq_cylinder_radius_l1133_113396


namespace NUMINAMATH_GPT_find_percentage_l1133_113394

theorem find_percentage (P N : ℝ) (h1 : (P / 100) * N = 60) (h2 : 0.80 * N = 240) : P = 20 :=
sorry

end NUMINAMATH_GPT_find_percentage_l1133_113394


namespace NUMINAMATH_GPT_arithmetic_progression_x_value_l1133_113366

theorem arithmetic_progression_x_value :
  ∀ (x : ℝ), (3 * x + 2) - (2 * x - 4) = (5 * x - 1) - (3 * x + 2) → x = 9 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_arithmetic_progression_x_value_l1133_113366


namespace NUMINAMATH_GPT_probability_of_odd_sum_given_even_product_l1133_113304

open Nat

noncomputable def probability_odd_sum_given_even_product : ℚ :=
  let total_outcomes := 6^5
  let odd_outcomes := 3^5
  let even_outcomes := total_outcomes - odd_outcomes
  let favorable_outcomes := 15 * 3^5
  favorable_outcomes / even_outcomes

theorem probability_of_odd_sum_given_even_product :
  probability_odd_sum_given_even_product = 91 / 324 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_odd_sum_given_even_product_l1133_113304


namespace NUMINAMATH_GPT_non_black_cows_l1133_113318

-- Define the main problem conditions
def total_cows : ℕ := 18
def black_cows : ℕ := (total_cows / 2) + 5

-- Statement to prove the number of non-black cows
theorem non_black_cows :
  total_cows - black_cows = 4 :=
by
  sorry

end NUMINAMATH_GPT_non_black_cows_l1133_113318


namespace NUMINAMATH_GPT_yards_green_correct_l1133_113338

-- Define the conditions
def total_yards_silk := 111421
def yards_pink := 49500

-- Define the question as a theorem statement
theorem yards_green_correct :
  (total_yards_silk - yards_pink = 61921) :=
by
  sorry

end NUMINAMATH_GPT_yards_green_correct_l1133_113338


namespace NUMINAMATH_GPT_sum_of_arith_geo_progression_l1133_113373

noncomputable def sum_two_numbers (a b : ℝ) : ℝ :=
  a + b

theorem sum_of_arith_geo_progression : 
  ∃ (a b : ℝ), (∃ d : ℝ, a = 4 + d ∧ b = 4 + 2 * d) ∧ 
  (∃ r : ℝ, a * r = b ∧ b * r = 16) ∧ 
  sum_two_numbers a b = 8 + 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_arith_geo_progression_l1133_113373


namespace NUMINAMATH_GPT_percentage_problem_l1133_113306

theorem percentage_problem (N : ℕ) (P : ℕ) (h1 : N = 25) (h2 : N = (P * N / 100) + 21) : P = 16 :=
sorry

end NUMINAMATH_GPT_percentage_problem_l1133_113306


namespace NUMINAMATH_GPT_range_independent_variable_l1133_113392

theorem range_independent_variable (x : ℝ) (h : x + 1 > 0) : x > -1 :=
sorry

end NUMINAMATH_GPT_range_independent_variable_l1133_113392


namespace NUMINAMATH_GPT_parabola_value_l1133_113391

theorem parabola_value (b c : ℝ) (h : 3 = -(-2) ^ 2 + b * -2 + c) : 2 * c - 4 * b - 9 = 5 := by
  sorry

end NUMINAMATH_GPT_parabola_value_l1133_113391


namespace NUMINAMATH_GPT_sum_of_ratios_of_squares_l1133_113312

theorem sum_of_ratios_of_squares (r : ℚ) (a b c : ℤ) (h1 : r = 45 / 64) 
  (h2 : r = (a * (Real.sqrt b)) / c) 
  (ha : a = 3) 
  (hb : b = 5) 
  (hc : c = 8) : a + b + c = 16 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_ratios_of_squares_l1133_113312


namespace NUMINAMATH_GPT_Tracy_sold_paintings_l1133_113301

theorem Tracy_sold_paintings (num_people : ℕ) (group1_customers : ℕ) (group1_paintings : ℕ)
    (group2_customers : ℕ) (group2_paintings : ℕ) (group3_customers : ℕ) (group3_paintings : ℕ) 
    (total_paintings : ℕ) :
    num_people = 20 →
    group1_customers = 4 →
    group1_paintings = 2 →
    group2_customers = 12 →
    group2_paintings = 1 →
    group3_customers = 4 →
    group3_paintings = 4 →
    total_paintings = (group1_customers * group1_paintings) + (group2_customers * group2_paintings) + 
                      (group3_customers * group3_paintings) →
    total_paintings = 36 :=
by
  intros 
  -- including this to ensure the lean code passes syntax checks
  sorry

end NUMINAMATH_GPT_Tracy_sold_paintings_l1133_113301


namespace NUMINAMATH_GPT_problem_statement_l1133_113369

def complex_number (m : ℂ) : ℂ :=
  (m^2 - 3*m - 4) + (m^2 - 5*m - 6) * Complex.I

theorem problem_statement (m : ℂ) :
  (complex_number m).im = m^2 - 5*m - 6 →
  (complex_number m).re = 0 →
  m ≠ -1 ∧ m ≠ 6 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1133_113369


namespace NUMINAMATH_GPT_sum_third_column_l1133_113320

variable (a b c d e f g h i : ℕ)

theorem sum_third_column :
  (a + b + c = 24) →
  (d + e + f = 26) →
  (g + h + i = 40) →
  (a + d + g = 27) →
  (b + e + h = 20) →
  (c + f + i = 43) :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_third_column_l1133_113320


namespace NUMINAMATH_GPT_Sahil_transportation_charges_l1133_113341

theorem Sahil_transportation_charges
  (cost_machine : ℝ)
  (cost_repair : ℝ)
  (actual_selling_price : ℝ)
  (profit_percentage : ℝ)
  (transportation_charges : ℝ)
  (h1 : cost_machine = 12000)
  (h2 : cost_repair = 5000)
  (h3 : profit_percentage = 0.50)
  (h4 : actual_selling_price = 27000)
  (h5 : transportation_charges + (cost_machine + cost_repair) * (1 + profit_percentage) = actual_selling_price) :
  transportation_charges = 1500 :=
by
  sorry

end NUMINAMATH_GPT_Sahil_transportation_charges_l1133_113341


namespace NUMINAMATH_GPT_initially_calculated_average_height_l1133_113358

theorem initially_calculated_average_height
  (A : ℝ)
  (h1 : ∀ heights : List ℝ, heights.length = 35 → (heights.sum + (106 - 166) = heights.sum) → (heights.sum / 35) = 180) :
  A = 181.71 :=
sorry

end NUMINAMATH_GPT_initially_calculated_average_height_l1133_113358


namespace NUMINAMATH_GPT_value_of_X_l1133_113325

theorem value_of_X (X : ℝ) (h : ((X + 0.064)^2 - (X - 0.064)^2) / (X * 0.064) = 4.000000000000002) : X ≠ 0 :=
sorry

end NUMINAMATH_GPT_value_of_X_l1133_113325


namespace NUMINAMATH_GPT_sqrt_product_l1133_113313

theorem sqrt_product (h1 : Real.sqrt 81 = 9) 
                     (h2 : Real.sqrt 16 = 4) 
                     (h3 : Real.sqrt (Real.sqrt (Real.sqrt 64)) = 2 * Real.sqrt 2) : 
                     Real.sqrt 81 * Real.sqrt 16 * Real.sqrt (Real.sqrt (Real.sqrt 64)) = 72 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_product_l1133_113313


namespace NUMINAMATH_GPT_T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l1133_113342

def T (r n : ℕ) : ℕ :=
  sorry -- Define the function T_r(n) according to the problem's condition

-- Specific cases given in the problem statement
noncomputable def T_0_2006 : ℕ := T 0 2006
noncomputable def T_1_2006 : ℕ := T 1 2006
noncomputable def T_2_2006 : ℕ := T 2 2006

-- Theorems stating the result
theorem T_0_2006_correct : T_0_2006 = 1764 := sorry
theorem T_1_2006_correct : T_1_2006 = 122 := sorry
theorem T_2_2006_correct : T_2_2006 = 121 := sorry

end NUMINAMATH_GPT_T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l1133_113342


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l1133_113370

-- Sum of the first n terms of the sequence
noncomputable def S_n (n : ℕ) (c : ℤ) : ℤ := (n + 1) * (n + 1) + c

-- The nth term of the sequence
noncomputable def a_n (n : ℕ) (c : ℤ) : ℤ := S_n n c - (S_n (n - 1) c)

-- Define the sequence being arithmetic
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) - a n = d

theorem necessary_and_sufficient_condition (c : ℤ) :
  (∀ n ≥ 1, a_n n c - a_n (n-1) c = 2) ↔ (c = -1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l1133_113370


namespace NUMINAMATH_GPT_value_of_expression_is_one_l1133_113317

theorem value_of_expression_is_one : 
  ∃ (a b c d : ℚ), (a = 1) ∧ (b = -1) ∧ (c = 0) ∧ (d = 1 ∨ d = -1) ∧ (a - b + c^2 - |d| = 1) :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_is_one_l1133_113317


namespace NUMINAMATH_GPT_neg_p_iff_forall_l1133_113363

-- Define the proposition p
def p : Prop := ∃ (x : ℝ), x > 1 ∧ x^2 - 1 > 0

-- State the negation of p as a theorem
theorem neg_p_iff_forall : ¬ p ↔ ∀ (x : ℝ), x > 1 → x^2 - 1 ≤ 0 :=
by sorry

end NUMINAMATH_GPT_neg_p_iff_forall_l1133_113363


namespace NUMINAMATH_GPT_binom_60_3_eq_34220_l1133_113336

theorem binom_60_3_eq_34220 : (Nat.choose 60 3) = 34220 := 
by sorry

end NUMINAMATH_GPT_binom_60_3_eq_34220_l1133_113336


namespace NUMINAMATH_GPT_stratified_sampling_result_l1133_113307

-- Define the total number of students in each grade
def students_grade10 : ℕ := 1600
def students_grade11 : ℕ := 1200
def students_grade12 : ℕ := 800

-- Define the condition
def stratified_sampling (x : ℕ) : Prop :=
  (x / (students_grade10 + students_grade11 + students_grade12) = (20 / students_grade12))

-- The main statement to be proven
theorem stratified_sampling_result 
  (students_grade10 : ℕ)
  (students_grade11 : ℕ)
  (students_grade12 : ℕ)
  (sampled_from_grade12 : ℕ)
  (h_sampling : stratified_sampling 90)
  (h_sampled12 : sampled_from_grade12 = 20) :
  (90 - sampled_from_grade12 = 70) :=
  by
    sorry

end NUMINAMATH_GPT_stratified_sampling_result_l1133_113307


namespace NUMINAMATH_GPT_fixed_point_line_passes_through_range_of_t_l1133_113398

-- Definition for first condition: Line with slope k (k ≠ 0)
variables {k : ℝ} (hk : k ≠ 0)

-- Definition for second condition: Ellipse C
def ellipse_C (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Third condition: Intersections M and N
variables (M N : ℝ × ℝ)
variables (intersection_M : ellipse_C M.1 M.2)
variables (intersection_N : ellipse_C N.1 N.2)

-- Fourth condition: Slopes are k1 and k2
variables {k1 k2 : ℝ}
variables (hk1 : k1 = M.2 / M.1)
variables (hk2 : k2 = N.2 / N.1)

-- Fifth condition: Given equation 3(k1 + k2) = 8k
variables (h_eq : 3 * (k1 + k2) = 8 * k)

-- Proof for question 1: Line passes through a fixed point
theorem fixed_point_line_passes_through 
    (h_eq : 3 * (k1 + k2) = 8 * k) : 
    ∃ n : ℝ, n = 1/2 ∨ n = -1/2 := sorry

-- Additional conditions for question 2
variables {D : ℝ × ℝ} (hD : D = (1, 0))
variables (t : ℝ)
variables (area_ratio : (M.2 / N.2) = t)
variables (h_ineq : k^2 < 5 / 12)

-- Proof for question 2: Range for t
theorem range_of_t
    (hD : D = (1, 0))
    (area_ratio : (M.2 / N.2) = t)
    (h_ineq : k^2 < 5 / 12) : 
    2 < t ∧ t < 3 ∨ 1 / 3 < t ∧ t < 1 / 2 := sorry

end NUMINAMATH_GPT_fixed_point_line_passes_through_range_of_t_l1133_113398


namespace NUMINAMATH_GPT_parabola_x_intercepts_incorrect_l1133_113339

-- Define the given quadratic function
noncomputable def f (x : ℝ) : ℝ := -1 / 2 * (x - 1)^2 + 2

-- The Lean statement for the problem
theorem parabola_x_intercepts_incorrect :
  ¬ ((f 3 = 0) ∧ (f (-3) = 0)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_x_intercepts_incorrect_l1133_113339


namespace NUMINAMATH_GPT_eighteenth_entry_of_sequence_l1133_113386

def r_7 (n : ℕ) : ℕ := n % 7

theorem eighteenth_entry_of_sequence : ∃ n : ℕ, (r_7 (4 * n) ≤ 3) ∧ (∀ m : ℕ, m < 18 → (r_7 (4 * m) ≤ 3) → m ≠ n) ∧ n = 30 := 
by 
  sorry

end NUMINAMATH_GPT_eighteenth_entry_of_sequence_l1133_113386


namespace NUMINAMATH_GPT_granger_buys_3_jars_of_peanut_butter_l1133_113381

theorem granger_buys_3_jars_of_peanut_butter :
  ∀ (spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count: ℕ),
    spam_cost = 3 → peanut_butter_cost = 5 → bread_cost = 2 →
    spam_count = 12 → loaf_count = 4 → total_cost = 59 →
    spam_cost * spam_count + bread_cost * loaf_count + peanut_butter_cost * peanut_butter_count = total_cost →
    peanut_butter_count = 3 :=
by
  intros spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count
  intros hspam_cost hpeanut_butter_cost hbread_cost hspam_count hloaf_count htotal_cost htotal
  sorry  -- The proof step is omitted as requested.

end NUMINAMATH_GPT_granger_buys_3_jars_of_peanut_butter_l1133_113381


namespace NUMINAMATH_GPT_coloring_scheme_count_l1133_113305

/-- Given the set of points in the Cartesian plane, where each point (m, n) with
    1 <= m, n <= 6 is colored either red or blue, the number of ways to color these points
    such that each unit square has exactly two red vertices is 126. -/
theorem coloring_scheme_count 
  (color : Fin 6 → Fin 6 → Bool)
  (colored_correctly : ∀ m n, (1 ≤ m ∧ m ≤ 6) ∧ (1 ≤ n ∧ n ≤ 6) ∧ 
    (color m n = true ∨ color m n = false) :=
    sorry
  )
  : (∃ valid_coloring : Nat, valid_coloring = 126) :=
  sorry

end NUMINAMATH_GPT_coloring_scheme_count_l1133_113305


namespace NUMINAMATH_GPT_gallons_in_pond_after_50_days_l1133_113327

def initial_amount : ℕ := 500
def evaporation_rate : ℕ := 1
def days_passed : ℕ := 50
def total_evaporation : ℕ := days_passed * evaporation_rate
def final_amount : ℕ := initial_amount - total_evaporation

theorem gallons_in_pond_after_50_days : final_amount = 450 := by
  sorry

end NUMINAMATH_GPT_gallons_in_pond_after_50_days_l1133_113327


namespace NUMINAMATH_GPT_units_digit_odd_product_l1133_113344

theorem units_digit_odd_product (l : List ℕ) (h_odds : ∀ n ∈ l, n % 2 = 1) :
  (∀ x ∈ l, x % 10 = 5) ↔ (5 ∈ l) := by
  sorry

end NUMINAMATH_GPT_units_digit_odd_product_l1133_113344


namespace NUMINAMATH_GPT_fraction_invariant_l1133_113365

variable {R : Type*} [Field R]
variables (x y : R)

theorem fraction_invariant : (2 * x) / (3 * x - y) = (6 * x) / (9 * x - 3 * y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_invariant_l1133_113365


namespace NUMINAMATH_GPT_melissa_total_cost_l1133_113349

-- Definitions based on conditions
def daily_rental_rate : ℝ := 15
def mileage_rate : ℝ := 0.10
def number_of_days : ℕ := 3
def number_of_miles : ℕ := 300

-- Theorem statement to prove the total cost
theorem melissa_total_cost : daily_rental_rate * number_of_days + mileage_rate * number_of_miles = 75 := 
by 
  sorry

end NUMINAMATH_GPT_melissa_total_cost_l1133_113349


namespace NUMINAMATH_GPT_divisors_of_90_l1133_113323

def num_pos_divisors (n : ℕ) : ℕ :=
  let factors := if n = 90 then [(2, 1), (3, 2), (5, 1)] else []
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

theorem divisors_of_90 : num_pos_divisors 90 = 12 := by
  sorry

end NUMINAMATH_GPT_divisors_of_90_l1133_113323


namespace NUMINAMATH_GPT_largest_constant_C_l1133_113353

theorem largest_constant_C :
  ∃ C : ℝ, 
    (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z - 1)) 
      ∧ (∀ D : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ D * (x + y + z - 1)) → C ≥ D)
    ∧ C = (2 + 2 * Real.sqrt 7) / 3 :=
sorry

end NUMINAMATH_GPT_largest_constant_C_l1133_113353


namespace NUMINAMATH_GPT_exists_same_color_points_at_unit_distance_l1133_113367

theorem exists_same_color_points_at_unit_distance
  (color : ℝ × ℝ → ℕ)
  (coloring : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end NUMINAMATH_GPT_exists_same_color_points_at_unit_distance_l1133_113367


namespace NUMINAMATH_GPT_simplify_expression_l1133_113361

theorem simplify_expression (x : ℝ) : (2 * x)^5 - (5 * x) * (x^4) = 27 * x^5 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1133_113361


namespace NUMINAMATH_GPT_find_a_l1133_113348

-- Define the polynomial expansion term conditions
def binomial_coefficient (n k : ℕ) := Nat.choose n k

def fourth_term_coefficient (x a : ℝ) : ℝ :=
  binomial_coefficient 9 3 * x^6 * a^3

theorem find_a (a : ℝ) (x : ℝ) (h : fourth_term_coefficient x a = 84) : a = 1 :=
by
  unfold fourth_term_coefficient at h
  sorry

end NUMINAMATH_GPT_find_a_l1133_113348


namespace NUMINAMATH_GPT_simplify_expression_l1133_113333

open Real

-- Assuming lg refers to the common logarithm log base 10
noncomputable def problem_expression : ℝ :=
  log 4 + 2 * log 5 + 4^(-1/2:ℝ)

theorem simplify_expression : problem_expression = 5 / 2 :=
by
  -- Placeholder proof, actual steps not required
  sorry

end NUMINAMATH_GPT_simplify_expression_l1133_113333


namespace NUMINAMATH_GPT_find_f_2005_1000_l1133_113372

-- Define the real-valued function and its properties
def f (x y : ℝ) : ℝ := sorry

-- The condition given in the problem
axiom condition :
  ∀ x y z : ℝ, f x y = f x z - 2 * f y z - 2 * z

-- The target we need to prove
theorem find_f_2005_1000 : f 2005 1000 = 5 := 
by 
  -- all necessary logical steps (detailed in solution) would go here
  sorry

end NUMINAMATH_GPT_find_f_2005_1000_l1133_113372


namespace NUMINAMATH_GPT_scissor_count_l1133_113352

theorem scissor_count :
  let initial_scissors := 54 
  let added_scissors := 22
  let removed_scissors := 15
  initial_scissors + added_scissors - removed_scissors = 61 := by
  sorry

end NUMINAMATH_GPT_scissor_count_l1133_113352


namespace NUMINAMATH_GPT_problem1_problem2_l1133_113329

-- Problem 1: Prove that \(\sqrt{27}+3\sqrt{\frac{1}{3}}-\sqrt{24} \times \sqrt{2} = 0\)
theorem problem1 : Real.sqrt 27 + 3 * Real.sqrt (1 / 3) - Real.sqrt 24 * Real.sqrt 2 = 0 := 
by sorry

-- Problem 2: Prove that \((\sqrt{5}-2)(2+\sqrt{5})-{(\sqrt{3}-1)}^{2} = -3 + 2\sqrt{3}\)
theorem problem2 : (Real.sqrt 5 - 2) * (2 + Real.sqrt 5) - (Real.sqrt 3 - 1) ^ 2 = -3 + 2 * Real.sqrt 3 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1133_113329


namespace NUMINAMATH_GPT_jessica_marbles_62_l1133_113335

-- Definitions based on conditions
def marbles_kurt (marbles_dennis : ℕ) : ℕ := marbles_dennis - 45
def marbles_laurie (marbles_kurt : ℕ) : ℕ := marbles_kurt + 12
def marbles_jessica (marbles_laurie : ℕ) : ℕ := marbles_laurie + 25

-- Given marbles for Dennis
def marbles_dennis : ℕ := 70

-- Proof statement: Prove that Jessica has 62 marbles given the conditions
theorem jessica_marbles_62 : marbles_jessica (marbles_laurie (marbles_kurt marbles_dennis)) = 62 := 
by
  sorry

end NUMINAMATH_GPT_jessica_marbles_62_l1133_113335


namespace NUMINAMATH_GPT_hyperbola_min_value_l1133_113377

def hyperbola_condition : Prop :=
  ∀ (m : ℝ), ∀ (x y : ℝ), (4 * x + 3 * y + m = 0 → (x^2 / 9 - y^2 / 16 = 1) → false)

noncomputable def minimum_value : ℝ :=
  2 * Real.sqrt 37 - 6

theorem hyperbola_min_value :
  hyperbola_condition → minimum_value =  2 * Real.sqrt 37 - 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_hyperbola_min_value_l1133_113377


namespace NUMINAMATH_GPT_project_completion_l1133_113380

theorem project_completion (a b c d e : ℕ) 
  (h₁ : 1 / (a : ℝ) + 1 / b + 1 / c + 1 / d = 1 / 6)
  (h₂ : 1 / (b : ℝ) + 1 / c + 1 / d + 1 / e = 1 / 8)
  (h₃ : 1 / (a : ℝ) + 1 / e = 1 / 12) : 
  e = 48 :=
sorry

end NUMINAMATH_GPT_project_completion_l1133_113380


namespace NUMINAMATH_GPT_two_is_four_percent_of_fifty_l1133_113337

theorem two_is_four_percent_of_fifty : (2 / 50) * 100 = 4 := 
by
  sorry

end NUMINAMATH_GPT_two_is_four_percent_of_fifty_l1133_113337


namespace NUMINAMATH_GPT_compound_interest_two_years_l1133_113399

/-- Given the initial amount, and year-wise interest rates, 
     we want to find the amount in 2 years and prove it equals to a specific value. -/
theorem compound_interest_two_years 
  (P : ℝ) (R1 : ℝ) (R2 : ℝ) (T1 : ℝ) (T2 : ℝ) 
  (initial_amount : P = 7644) 
  (interest_rate_first_year : R1 = 0.04) 
  (interest_rate_second_year : R2 = 0.05) 
  (time_first_year : T1 = 1) 
  (time_second_year : T2 = 1) : 
  (P + (P * R1 * T1) + ((P + (P * R1 * T1)) * R2 * T2) = 8347.248) := 
by 
  sorry

end NUMINAMATH_GPT_compound_interest_two_years_l1133_113399


namespace NUMINAMATH_GPT_inequality_log_equality_log_l1133_113357

theorem inequality_log (x : ℝ) (hx : x < 0 ∨ x > 0) :
  max 0 (Real.log (|x|)) ≥ 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2) := 
sorry

theorem equality_log (x : ℝ) :
  (max 0 (Real.log (|x|)) = 
  ((Real.sqrt 5 - 1) / (2 * Real.sqrt 5)) * Real.log (|x|) + 
  (1 / (2 * Real.sqrt 5)) * Real.log (|x^2 - 1|) + 
  (1 / 2) * Real.log ((Real.sqrt 5 + 1) / 2)) ↔ 
  (x = (Real.sqrt 5 + 1) / 2 ∨ x = (Real.sqrt 5 - 1) / 2 ∨ x = -(Real.sqrt 5 + 1) / 2 ∨ x = -(Real.sqrt 5 - 1) / 2) := 
sorry

end NUMINAMATH_GPT_inequality_log_equality_log_l1133_113357


namespace NUMINAMATH_GPT_min_value_f_a_neg3_max_value_g_ge_7_l1133_113351

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp x) * (x^2 + a * x + 1)

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := 2 * x^3 + 3 * (b + 1) * x^2 + 6 * b * x + 6

theorem min_value_f_a_neg3 (h : -3 ≤ -1) : 
  (∀ x : ℝ, f x (-3) ≥ -Real.exp 2) := 
sorry

theorem max_value_g_ge_7 (a : ℝ) (h : a ≤ -1) (b : ℝ) (h_b : b = a + 1) :
  ∃ m : ℝ, (∀ x : ℝ, g x b ≤ m) ∧ (m ≥ 7) := 
sorry

end NUMINAMATH_GPT_min_value_f_a_neg3_max_value_g_ge_7_l1133_113351


namespace NUMINAMATH_GPT_Mary_younger_by_14_l1133_113385

variable (Betty_age : ℕ) (Albert_age : ℕ) (Mary_age : ℕ)

theorem Mary_younger_by_14 :
  (Betty_age = 7) →
  (Albert_age = 4 * Betty_age) →
  (Albert_age = 2 * Mary_age) →
  (Albert_age - Mary_age = 14) :=
by
  intros
  sorry

end NUMINAMATH_GPT_Mary_younger_by_14_l1133_113385


namespace NUMINAMATH_GPT_sufficient_condition_l1133_113384

theorem sufficient_condition (a : ℝ) (h : a ≥ 10) : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → x^2 - a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_l1133_113384


namespace NUMINAMATH_GPT_fraction_simplification_l1133_113389

noncomputable def x : ℚ := 0.714714714 -- Repeating decimal representation for x
noncomputable def y : ℚ := 2.857857857 -- Repeating decimal representation for y

theorem fraction_simplification :
  (x / y) = (714 / 2855) :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1133_113389


namespace NUMINAMATH_GPT_line_equation_with_slope_angle_135_and_y_intercept_neg1_l1133_113310

theorem line_equation_with_slope_angle_135_and_y_intercept_neg1 :
  ∃ k b : ℝ, k = -1 ∧ b = -1 ∧ ∀ x y : ℝ, y = k * x + b ↔ y = -x - 1 :=
by
  sorry

end NUMINAMATH_GPT_line_equation_with_slope_angle_135_and_y_intercept_neg1_l1133_113310


namespace NUMINAMATH_GPT_length_of_train_is_135_l1133_113308

noncomputable def length_of_train (v : ℝ) (t : ℝ) : ℝ :=
  ((v * 1000) / 3600) * t

theorem length_of_train_is_135 :
  length_of_train 140 3.4711508793582233 = 135 :=
sorry

end NUMINAMATH_GPT_length_of_train_is_135_l1133_113308


namespace NUMINAMATH_GPT_number_of_solutions_abs_eq_l1133_113390

theorem number_of_solutions_abs_eq (f : ℝ → ℝ) (g : ℝ → ℝ) : 
  (∀ x : ℝ, f x = |3 * x| ∧ g x = |x - 2| ∧ (f x + g x = 4) → 
  ∃! x1 x2 : ℝ, 
    ((0 < x1 ∧ x1 < 2 ∧ f x1 + g x1 = 4 ) ∨ 
    (x2 < 0 ∧ f x2 + g x2 = 4) ∧ x1 ≠ x2)) :=
by
  sorry

end NUMINAMATH_GPT_number_of_solutions_abs_eq_l1133_113390


namespace NUMINAMATH_GPT_ones_digit_of_8_pow_47_l1133_113330

theorem ones_digit_of_8_pow_47 : (8^47) % 10 = 2 := 
  sorry

end NUMINAMATH_GPT_ones_digit_of_8_pow_47_l1133_113330


namespace NUMINAMATH_GPT_hexagon_angle_U_l1133_113315

theorem hexagon_angle_U 
  (F I U G E R : ℝ)
  (h1 : F = I) 
  (h2 : I = U)
  (h3 : G + E = 180)
  (h4 : R + U = 180)
  (h5 : F + I + G + U + R + E = 720) :
  U = 120 := by
  sorry

end NUMINAMATH_GPT_hexagon_angle_U_l1133_113315


namespace NUMINAMATH_GPT_arcsin_one_half_l1133_113395

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_one_half_l1133_113395


namespace NUMINAMATH_GPT_ethanol_percentage_fuel_B_l1133_113374

noncomputable def percentage_ethanol_in_fuel_B : ℝ :=
  let tank_capacity := 208
  let ethanol_in_fuelA := 0.12
  let total_ethanol := 30
  let volume_fuelA := 82
  let ethanol_from_fuelA := volume_fuelA * ethanol_in_fuelA
  let ethanol_from_fuelB := total_ethanol - ethanol_from_fuelA
  let volume_fuelB := tank_capacity - volume_fuelA
  (ethanol_from_fuelB / volume_fuelB) * 100

theorem ethanol_percentage_fuel_B :
  percentage_ethanol_in_fuel_B = 16 :=
by
  sorry

end NUMINAMATH_GPT_ethanol_percentage_fuel_B_l1133_113374


namespace NUMINAMATH_GPT_total_supermarkets_FGH_chain_l1133_113378

variable (US_supermarkets : ℕ) (Canada_supermarkets : ℕ)
variable (total_supermarkets : ℕ)

-- Conditions
def condition1 := US_supermarkets = 37
def condition2 := US_supermarkets = Canada_supermarkets + 14

-- Goal
theorem total_supermarkets_FGH_chain
    (h1 : condition1 US_supermarkets)
    (h2 : condition2 US_supermarkets Canada_supermarkets) :
    total_supermarkets = US_supermarkets + Canada_supermarkets :=
sorry

end NUMINAMATH_GPT_total_supermarkets_FGH_chain_l1133_113378


namespace NUMINAMATH_GPT_ones_digit_of_4567_times_3_is_1_l1133_113321

theorem ones_digit_of_4567_times_3_is_1 :
  let n := 4567
  let m := 3
  (n * m) % 10 = 1 :=
by
  let n := 4567
  let m := 3
  have h : (n * m) % 10 = ((4567 * 3) % 10) := by rfl -- simplifying the product
  sorry -- this is where the proof would go, if required

end NUMINAMATH_GPT_ones_digit_of_4567_times_3_is_1_l1133_113321


namespace NUMINAMATH_GPT_max_a4a7_value_l1133_113309

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n m : ℕ, a (n + 1) = a n + d

-- Given conditions
def given_conditions (a : ℕ → ℝ) (d : ℝ) : Prop := 
  arithmetic_sequence a d ∧ a 5 = 4 -- a6 = 4 so we use index 5 since Lean is 0-indexed

-- Define the product a4 * a7
def a4a7_product (a : ℕ → ℝ) (d : ℝ) : ℝ := (a 5 - 2 * d) * (a 5 + d)

-- The maximum value of a4 * a7
def max_a4a7 (a : ℕ → ℝ) (d : ℝ) : ℝ := 18

-- The proof problem statement
theorem max_a4a7_value (a : ℕ → ℝ) (d : ℝ) :
  given_conditions a d → a4a7_product a d = max_a4a7 a d :=
by
  sorry

end NUMINAMATH_GPT_max_a4a7_value_l1133_113309


namespace NUMINAMATH_GPT_quadratic_distinct_roots_l1133_113360

theorem quadratic_distinct_roots (k : ℝ) : 
  (∃ x y : ℝ, k * x^2 - 2 * x - 1 = 0 ∧ k * y^2 - 2 * y - 1 = 0 ∧ x ≠ y) ↔ k > -1 ∧ k ≠ 0 := 
sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_l1133_113360


namespace NUMINAMATH_GPT_ratio_of_amount_lost_l1133_113397

noncomputable def amount_lost (initial_amount spent_motorcycle spent_concert after_loss : ℕ) : ℕ :=
  let remaining_after_motorcycle := initial_amount - spent_motorcycle
  let remaining_after_concert := remaining_after_motorcycle / 2
  remaining_after_concert - after_loss

noncomputable def ratio (a b : ℕ) : ℕ × ℕ :=
  let g := Nat.gcd a b
  (a / g, b / g)

theorem ratio_of_amount_lost 
  (initial_amount spent_motorcycle spent_concert after_loss : ℕ)
  (h1 : initial_amount = 5000)
  (h2 : spent_motorcycle = 2800)
  (h3 : spent_concert = (initial_amount - spent_motorcycle) / 2)
  (h4 : after_loss = 825) :
  ratio (amount_lost initial_amount spent_motorcycle spent_concert after_loss)
        spent_concert = (1, 4) := by
  sorry

end NUMINAMATH_GPT_ratio_of_amount_lost_l1133_113397


namespace NUMINAMATH_GPT_solution_set_inequality_l1133_113376

theorem solution_set_inequality (x : ℝ) : 4 * x^2 - 3 * x > 5 ↔ x < -5/4 ∨ x > 1 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1133_113376


namespace NUMINAMATH_GPT_find_x_value_l1133_113362

open Real

theorem find_x_value (a : ℝ) (x : ℝ) (h : a > 0) (h_eq : 10^x = log (10 * a) + log (a⁻¹)) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_x_value_l1133_113362


namespace NUMINAMATH_GPT_max_value_of_determinant_l1133_113393

noncomputable def determinant_of_matrix (θ : ℝ) : ℝ :=
  Matrix.det ![
    ![1, 1, 1],
    ![1, 1 + Real.sin (2 * θ), 1],
    ![1, 1, 1 + Real.cos (2 * θ)]
  ]

theorem max_value_of_determinant : 
  ∃ θ : ℝ, (∀ θ : ℝ, determinant_of_matrix θ ≤ (1 / 2)) ∧ determinant_of_matrix (θ_at_maximum) = (1 / 2) :=
sorry

end NUMINAMATH_GPT_max_value_of_determinant_l1133_113393


namespace NUMINAMATH_GPT_least_five_digit_congruent_l1133_113383

theorem least_five_digit_congruent (x : ℕ) (h1 : x ≥ 10000) (h2 : x < 100000) (h3 : x % 17 = 8) : x = 10004 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_five_digit_congruent_l1133_113383


namespace NUMINAMATH_GPT_perpendicular_vectors_x_value_l1133_113340

theorem perpendicular_vectors_x_value 
  (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (x, -1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_x_value_l1133_113340


namespace NUMINAMATH_GPT_factorial_divides_exponential_difference_l1133_113331

theorem factorial_divides_exponential_difference (n : ℕ) : n! ∣ 2^(2 * n!) - 2^n! :=
by
  sorry

end NUMINAMATH_GPT_factorial_divides_exponential_difference_l1133_113331


namespace NUMINAMATH_GPT_lunas_phone_bill_percentage_l1133_113311

variables (H F P : ℝ)

theorem lunas_phone_bill_percentage :
  F = 0.60 * H ∧ H + F = 240 ∧ H + F + P = 249 →
  (P / F) * 100 = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lunas_phone_bill_percentage_l1133_113311


namespace NUMINAMATH_GPT_line_equation_through_point_slope_l1133_113334

theorem line_equation_through_point_slope :
  ∃ (a b c : ℝ), (a, b) ≠ (0, 0) ∧ (a * 1 + b * 3 + c = 0) ∧ (y = -4 * x → k = -4 / 9) ∧ (∀ (x y : ℝ), y - 3 = k * (x - 1) → 4 * x + 3 * y - 13 = 0) :=
sorry

end NUMINAMATH_GPT_line_equation_through_point_slope_l1133_113334


namespace NUMINAMATH_GPT_xyz_value_l1133_113303

theorem xyz_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 3) (h3 : z + 1/x = 2) :
  x * y * z = 10 + 3 * Real.sqrt 11 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l1133_113303


namespace NUMINAMATH_GPT_mike_ride_equals_42_l1133_113328

-- Define the costs as per the conditions
def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie : ℝ := 2.50 + 5.00 + 0.25 * 22

-- State the theorem that needs to be proved
theorem mike_ride_equals_42 : ∃ M : ℕ, cost_mike M = cost_annie ∧ M = 42 :=
by
  sorry

end NUMINAMATH_GPT_mike_ride_equals_42_l1133_113328


namespace NUMINAMATH_GPT_projection_magnitude_of_a_onto_b_equals_neg_three_l1133_113322

variables {a b : ℝ}

def vector_magnitude (v : ℝ) : ℝ := abs v

def dot_product (a b : ℝ) : ℝ := a * b

noncomputable def projection (a b : ℝ) : ℝ := (dot_product a b) / (vector_magnitude b)

theorem projection_magnitude_of_a_onto_b_equals_neg_three
  (ha : vector_magnitude a = 5)
  (hb : vector_magnitude b = 3)
  (hab : dot_product a b = -9) :
  projection a b = -3 :=
by sorry

end NUMINAMATH_GPT_projection_magnitude_of_a_onto_b_equals_neg_three_l1133_113322


namespace NUMINAMATH_GPT_cosine_product_inequality_l1133_113355

theorem cosine_product_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  8 * Real.cos A * Real.cos B * Real.cos C ≤ 1 := 
sorry

end NUMINAMATH_GPT_cosine_product_inequality_l1133_113355


namespace NUMINAMATH_GPT_problem_S_equal_102_l1133_113375

-- Define the values in Lean
def S : ℕ := 1 * 3^1 + 2 * 3^2 + 3 * 3^3

-- Theorem to prove that S is equal to 102
theorem problem_S_equal_102 : S = 102 :=
by
  sorry

end NUMINAMATH_GPT_problem_S_equal_102_l1133_113375


namespace NUMINAMATH_GPT_tangent_line_to_ellipse_l1133_113319

theorem tangent_line_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 1 → x^2 + 4 * y^2 = 1 → (x^2 + 4 * (m * x + 1)^2 = 1)) →
  m^2 = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_ellipse_l1133_113319


namespace NUMINAMATH_GPT_total_books_proof_l1133_113316

-- Define the number of books Lily finished last month.
def books_last_month : ℕ := 4

-- Define the number of books Lily wants to finish this month.
def books_this_month : ℕ := books_last_month * 2

-- Define the total number of books Lily will finish in two months.
def total_books_two_months : ℕ := books_last_month + books_this_month

-- Theorem to prove the total number of books Lily will finish in two months is 12.
theorem total_books_proof : total_books_two_months = 12 := by
  -- Here would be the proof steps.
  sorry

end NUMINAMATH_GPT_total_books_proof_l1133_113316


namespace NUMINAMATH_GPT_red_pairs_count_l1133_113324

theorem red_pairs_count (students_green : ℕ) (students_red : ℕ) (total_students : ℕ) (total_pairs : ℕ)
(pairs_green_green : ℕ) : 
students_green = 63 →
students_red = 69 →
total_students = 132 →
total_pairs = 66 →
pairs_green_green = 21 →
∃ (pairs_red_red : ℕ), pairs_red_red = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_red_pairs_count_l1133_113324


namespace NUMINAMATH_GPT_truck_speed_on_dirt_road_l1133_113346

theorem truck_speed_on_dirt_road 
  (total_distance: ℝ) (time_on_dirt: ℝ) (time_on_paved: ℝ) (speed_difference: ℝ)
  (h1: total_distance = 200) (h2: time_on_dirt = 3) (h3: time_on_paved = 2) (h4: speed_difference = 20) : 
  ∃ v: ℝ, (time_on_dirt * v + time_on_paved * (v + speed_difference) = total_distance) ∧ v = 32 := 
sorry

end NUMINAMATH_GPT_truck_speed_on_dirt_road_l1133_113346


namespace NUMINAMATH_GPT_base_area_of_cuboid_l1133_113382

theorem base_area_of_cuboid (V h : ℝ) (hv : V = 144) (hh : h = 8) : ∃ A : ℝ, A = 18 := by
  sorry

end NUMINAMATH_GPT_base_area_of_cuboid_l1133_113382


namespace NUMINAMATH_GPT_rent_increase_percentage_l1133_113388

theorem rent_increase_percentage (a x: ℝ) (h1: a ≠ 0) (h2: (9 / 10) * a = (4 / 5) * a * (1 + x / 100)) : x = 12.5 :=
sorry

end NUMINAMATH_GPT_rent_increase_percentage_l1133_113388


namespace NUMINAMATH_GPT_georgie_ghost_ways_l1133_113332

-- Define the total number of windows and locked windows
def total_windows : ℕ := 8
def locked_windows : ℕ := 2

-- Define the number of usable windows
def usable_windows : ℕ := total_windows - locked_windows

-- Define the theorem to prove the number of ways Georgie the Ghost can enter and exit
theorem georgie_ghost_ways :
  usable_windows * (usable_windows - 1) = 30 := by
  sorry

end NUMINAMATH_GPT_georgie_ghost_ways_l1133_113332


namespace NUMINAMATH_GPT_harkamal_payment_l1133_113354

theorem harkamal_payment :
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  total_payment = 1125 :=
by
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  sorry

end NUMINAMATH_GPT_harkamal_payment_l1133_113354


namespace NUMINAMATH_GPT_sqrt_81_eq_pm_9_l1133_113326

theorem sqrt_81_eq_pm_9 (x : ℤ) (hx : x^2 = 81) : x = 9 ∨ x = -9 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_81_eq_pm_9_l1133_113326


namespace NUMINAMATH_GPT_decreasing_function_l1133_113364

theorem decreasing_function (m : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → (m + 3) * x1 - 2 > (m + 3) * x2 - 2) ↔ m < -3 :=
by
  sorry

end NUMINAMATH_GPT_decreasing_function_l1133_113364


namespace NUMINAMATH_GPT_matrix_determinant_eq_9_l1133_113359

theorem matrix_determinant_eq_9 (x : ℝ) :
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  (a * d - b * c = 9) → x = -2 :=
by 
  let a := x - 1
  let b := 2
  let c := 3
  let d := -5
  sorry

end NUMINAMATH_GPT_matrix_determinant_eq_9_l1133_113359


namespace NUMINAMATH_GPT_frank_change_l1133_113347

theorem frank_change (n_c n_b money_given c_c c_b : ℕ) 
  (h1 : n_c = 5) 
  (h2 : n_b = 2) 
  (h3 : money_given = 20) 
  (h4 : c_c = 2) 
  (h5 : c_b = 3) : 
  money_given - (n_c * c_c + n_b * c_b) = 4 := 
by
  sorry

end NUMINAMATH_GPT_frank_change_l1133_113347


namespace NUMINAMATH_GPT_surveyDSuitableForComprehensiveSurvey_l1133_113379

inductive Survey where
| A : Survey
| B : Survey
| C : Survey
| D : Survey

def isComprehensiveSurvey (s : Survey) : Prop :=
  match s with
  | Survey.A => False
  | Survey.B => False
  | Survey.C => False
  | Survey.D => True

theorem surveyDSuitableForComprehensiveSurvey : isComprehensiveSurvey Survey.D :=
by
  sorry

end NUMINAMATH_GPT_surveyDSuitableForComprehensiveSurvey_l1133_113379
