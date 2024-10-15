import Mathlib

namespace NUMINAMATH_GPT_coloring_problem_l2243_224397

theorem coloring_problem (a : ℕ → ℕ) (n t : ℕ) 
  (h1 : ∀ i j, i < j → a i < a j) 
  (h2 : ∀ x : ℤ, ∃ i, 0 < i ∧ i ≤ n ∧ ((x + a (i - 1)) % t) = 0) : 
  n ∣ t :=
by
  sorry

end NUMINAMATH_GPT_coloring_problem_l2243_224397


namespace NUMINAMATH_GPT_complex_pure_imaginary_solution_l2243_224340

theorem complex_pure_imaginary_solution (m : ℝ) 
  (h_real_part : m^2 + 2*m - 3 = 0) 
  (h_imaginary_part : m - 1 ≠ 0) : 
  m = -3 :=
sorry

end NUMINAMATH_GPT_complex_pure_imaginary_solution_l2243_224340


namespace NUMINAMATH_GPT_apartment_building_count_l2243_224368

theorem apartment_building_count 
  (floors_per_building : ℕ) 
  (apartments_per_floor : ℕ) 
  (doors_per_apartment : ℕ) 
  (total_doors_needed : ℕ) 
  (doors_per_building : ℕ) 
  (number_of_buildings : ℕ)
  (h1 : floors_per_building = 12)
  (h2 : apartments_per_floor = 6) 
  (h3 : doors_per_apartment = 7) 
  (h4 : total_doors_needed = 1008) 
  (h5 : doors_per_building = apartments_per_floor * doors_per_apartment * floors_per_building)
  (h6 : number_of_buildings = total_doors_needed / doors_per_building) : 
  number_of_buildings = 2 := 
by 
  rw [h1, h2, h3] at h5 
  rw [h5, h4] at h6 
  exact h6

end NUMINAMATH_GPT_apartment_building_count_l2243_224368


namespace NUMINAMATH_GPT_solve_for_a_l2243_224337

theorem solve_for_a (a : ℝ) (h : (a + 3)^(a + 1) = 1) : a = -2 ∨ a = -1 :=
by {
  -- proof here
  sorry
}

end NUMINAMATH_GPT_solve_for_a_l2243_224337


namespace NUMINAMATH_GPT_roots_of_transformed_quadratic_l2243_224381

theorem roots_of_transformed_quadratic (a b p q s1 s2 : ℝ)
    (h_quad_eq : s1 ^ 2 + a * s1 + b = 0 ∧ s2 ^ 2 + a * s2 + b = 0)
    (h_sum_roots : s1 + s2 = -a)
    (h_prod_roots : s1 * s2 = b) :
        p = -(a ^ 4 - 4 * a ^ 2 * b + 2 * b ^ 2) ∧ 
        q = b ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_transformed_quadratic_l2243_224381


namespace NUMINAMATH_GPT_farmer_crops_saved_l2243_224329

noncomputable def average_corn_per_row := (10 + 14) / 2
noncomputable def average_potato_per_row := (35 + 45) / 2
noncomputable def average_wheat_per_row := (55 + 65) / 2

noncomputable def avg_reduction_corn := (40 + 60 + 25) / 3 / 100
noncomputable def avg_reduction_potato := (50 + 30 + 60) / 3 / 100
noncomputable def avg_reduction_wheat := (20 + 55 + 35) / 3 / 100

noncomputable def saved_corn_per_row := average_corn_per_row * (1 - avg_reduction_corn)
noncomputable def saved_potato_per_row := average_potato_per_row * (1 - avg_reduction_potato)
noncomputable def saved_wheat_per_row := average_wheat_per_row * (1 - avg_reduction_wheat)

def rows_corn := 30
def rows_potato := 24
def rows_wheat := 36

noncomputable def total_saved_corn := saved_corn_per_row * rows_corn
noncomputable def total_saved_potatoes := saved_potato_per_row * rows_potato
noncomputable def total_saved_wheat := saved_wheat_per_row * rows_wheat

noncomputable def total_crops_saved := total_saved_corn + total_saved_potatoes + total_saved_wheat

theorem farmer_crops_saved : total_crops_saved = 2090 := by
  sorry

end NUMINAMATH_GPT_farmer_crops_saved_l2243_224329


namespace NUMINAMATH_GPT_part_a_part_b_l2243_224312

/-- Part (a) statement: -/
theorem part_a (x : Fin 100 → ℕ) :
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) →
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) :=
by
  sorry

/-- Part (b) statement: -/
theorem part_b (x : Fin 100 → ℕ) :
  (∀ i j a b c : Fin 100, i ≠ j ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c →
    x i + x j < x a + x b + x c) →
  (∀ i j k a b c d : Fin 100, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d →
    x i + x j + x k < x a + x b + x c + x d) :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l2243_224312


namespace NUMINAMATH_GPT_ratio_of_boys_in_class_l2243_224302

noncomputable def boy_to_total_ratio (p_boy p_girl : ℚ) : ℚ :=
p_boy / (p_boy + p_girl)

theorem ratio_of_boys_in_class (p_boy p_girl total_students : ℚ)
    (h1 : p_boy = (3/4) * p_girl)
    (h2 : p_boy + p_girl = 1)
    (h3 : total_students = 1) :
    boy_to_total_ratio p_boy p_girl = 3/7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_in_class_l2243_224302


namespace NUMINAMATH_GPT_parallelogram_area_l2243_224319

noncomputable def area_parallelogram (b s θ : ℝ) : ℝ := b * (s * Real.sin θ)

theorem parallelogram_area : area_parallelogram 20 10 (Real.pi / 6) = 100 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l2243_224319


namespace NUMINAMATH_GPT_sum_of_three_consecutive_numbers_l2243_224306

theorem sum_of_three_consecutive_numbers (smallest : ℕ) (h : smallest = 29) :
  (smallest + (smallest + 1) + (smallest + 2)) = 90 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_numbers_l2243_224306


namespace NUMINAMATH_GPT_diff_quotient_remainder_n_75_l2243_224380

theorem diff_quotient_remainder_n_75 :
  ∃ n q r p : ℕ,  n = 75 ∧ n = 5 * q ∧ n = 34 * p + r ∧ q > r ∧ (q - r = 8) :=
by
  sorry

end NUMINAMATH_GPT_diff_quotient_remainder_n_75_l2243_224380


namespace NUMINAMATH_GPT_fraction_to_terminating_decimal_l2243_224371

theorem fraction_to_terminating_decimal :
  (53 : ℚ)/160 = 0.33125 :=
by sorry

end NUMINAMATH_GPT_fraction_to_terminating_decimal_l2243_224371


namespace NUMINAMATH_GPT_find_f_inv_value_l2243_224388

noncomputable def f (x : ℝ) : ℝ := 8^x
noncomputable def f_inv (y : ℝ) : ℝ := Real.logb 8 y

theorem find_f_inv_value (a : ℝ) (h : a = 8^(1/3)) : f_inv (a + 2) = Real.logb 8 (8^(1/3) + 2) := by
  sorry

end NUMINAMATH_GPT_find_f_inv_value_l2243_224388


namespace NUMINAMATH_GPT_john_ate_2_bags_for_dinner_l2243_224348

variable (x y : ℕ)
variable (h1 : x + y = 3)
variable (h2 : y ≥ 1)

theorem john_ate_2_bags_for_dinner : x = 2 := 
by sorry

end NUMINAMATH_GPT_john_ate_2_bags_for_dinner_l2243_224348


namespace NUMINAMATH_GPT_power_function_half_value_l2243_224325

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem power_function_half_value (a : ℝ) (h : (f 4 a) / (f 2 a) = 3) :
  f (1 / 2) a = 1 / 3 :=
by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_power_function_half_value_l2243_224325


namespace NUMINAMATH_GPT_integer_roots_7_values_of_a_l2243_224351

theorem integer_roots_7_values_of_a :
  (∃ a : ℝ, (∀ r s : ℤ, (r + s = -a ∧ (r * s = 8 * a))) ∧ (∃ n : ℕ, n = 7)) :=
sorry

end NUMINAMATH_GPT_integer_roots_7_values_of_a_l2243_224351


namespace NUMINAMATH_GPT_problem_l2243_224378

-- Definitions for angles A, B, C and sides a, b, c of a triangle.
variables {A B C : ℝ} {a b c : ℝ}
-- Given condition
variables (h : a = b * Real.cos C + c * Real.sin B)

-- Triangle inequality and angle conditions
variables (ha : 0 < A) (hb : 0 < B) (hc : 0 < C)
variables (suma : A + B + C = Real.pi)

-- Goal: to prove that under the given condition, angle B is π/4
theorem problem : B = Real.pi / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l2243_224378


namespace NUMINAMATH_GPT_quarters_per_jar_l2243_224330

/-- Jenn has 5 jars full of quarters. Each jar can hold a certain number of quarters.
    The bike costs 180 dollars, and she will have 20 dollars left over after buying it.
    Prove that each jar can hold 160 quarters. -/
theorem quarters_per_jar (num_jars : ℕ) (cost_bike : ℕ) (left_over : ℕ)
  (quarters_per_dollar : ℕ) (total_quarters : ℕ) (quarters_per_jar : ℕ) :
  num_jars = 5 → cost_bike = 180 → left_over = 20 → quarters_per_dollar = 4 →
  total_quarters = ((cost_bike + left_over) * quarters_per_dollar) →
  quarters_per_jar = (total_quarters / num_jars) →
  quarters_per_jar = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_quarters_per_jar_l2243_224330


namespace NUMINAMATH_GPT_milan_total_bill_correct_l2243_224326

-- Define the monthly fee, the per minute rate, and the number of minutes used last month
def monthly_fee : ℝ := 2
def per_minute_rate : ℝ := 0.12
def minutes_used : ℕ := 178

-- Define the total bill calculation
def total_bill : ℝ := minutes_used * per_minute_rate + monthly_fee

-- The proof statement
theorem milan_total_bill_correct :
  total_bill = 23.36 := 
by
  sorry

end NUMINAMATH_GPT_milan_total_bill_correct_l2243_224326


namespace NUMINAMATH_GPT_ratio_length_breadth_l2243_224395

-- Define the conditions
def length := 135
def area := 6075

-- Define the breadth in terms of the area and length
def breadth := area / length

-- The problem statement as a Lean 4 theorem to prove the ratio
theorem ratio_length_breadth : length / breadth = 3 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ratio_length_breadth_l2243_224395


namespace NUMINAMATH_GPT_solve_simultaneous_eqns_l2243_224301

theorem solve_simultaneous_eqns :
  ∀ (x y : ℝ), 
  (1/x - 1/(2*y) = 2*y^4 - 2*x^4 ∧ 1/x + 1/(2*y) = (3*x^2 + y^2) * (x^2 + 3*y^2)) 
  ↔ 
  (x = (3^(1/5) + 1) / 2 ∧ y = (3^(1/5) - 1) / 2) :=
by sorry

end NUMINAMATH_GPT_solve_simultaneous_eqns_l2243_224301


namespace NUMINAMATH_GPT_measure_smaller_angle_east_northwest_l2243_224344

/-- A mathematical structure for a circle with 12 rays forming congruent central angles. -/
structure CircleWithRays where
  rays : Finset (Fin 12)  -- There are 12 rays
  congruent_angles : ∀ i, i ∈ rays

/-- The measure of the central angle formed by each ray is 30 degrees (since 360/12 = 30). -/
def central_angle_measure : ℝ := 30

/-- The measure of the smaller angle formed between the ray pointing East and the ray pointing Northwest is 150 degrees. -/
theorem measure_smaller_angle_east_northwest (c : CircleWithRays) : 
  ∃ angle : ℝ, angle = 150 := by
  sorry

end NUMINAMATH_GPT_measure_smaller_angle_east_northwest_l2243_224344


namespace NUMINAMATH_GPT_similar_triangle_shortest_side_l2243_224349

theorem similar_triangle_shortest_side {a b c : ℝ} (h₁ : a = 24) (h₂ : b = 32) (h₃ : c = 80) :
  let hypotenuse₁ := Real.sqrt (a ^ 2 + b ^ 2)
  let scale_factor := c / hypotenuse₁
  let shortest_side₂ := scale_factor * a
  shortest_side₂ = 48 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangle_shortest_side_l2243_224349


namespace NUMINAMATH_GPT_find_f_at_one_l2243_224356

noncomputable def f (x : ℝ) (m n : ℝ) : ℝ := m * x^3 + n * x + 1

theorem find_f_at_one (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : f (-1) m n = 5) : f (1) m n = 7 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_f_at_one_l2243_224356


namespace NUMINAMATH_GPT_problem_statement_l2243_224373

-- Define the arithmetic sequence conditions
variables (a : ℕ → ℕ) (d : ℕ)
axiom h1 : a 1 = 2
axiom h2 : a 2018 = 2019
axiom arithmetic_seq : ∀ n, a (n + 1) = a n + d

-- Define the sum of the first n terms of the sequence
def sum_seq (n : ℕ) : ℕ := (n * a 1) + (n * (n-1) * d / 2)

theorem problem_statement : sum_seq a 5 + a 2014 = 2035 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l2243_224373


namespace NUMINAMATH_GPT_YoongiHasSevenPets_l2243_224393

def YoongiPets (dogs cats : ℕ) : ℕ := dogs + cats

theorem YoongiHasSevenPets : YoongiPets 5 2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_YoongiHasSevenPets_l2243_224393


namespace NUMINAMATH_GPT_last_digit_p_minus_q_not_5_l2243_224355

theorem last_digit_p_minus_q_not_5 (p q : ℕ) (n : ℕ) 
  (h1 : p * q = 10^n) 
  (h2 : ¬ (p % 10 = 0))
  (h3 : ¬ (q % 10 = 0))
  (h4 : p > q) : (p - q) % 10 ≠ 5 :=
by sorry

end NUMINAMATH_GPT_last_digit_p_minus_q_not_5_l2243_224355


namespace NUMINAMATH_GPT_fraction_meaningful_iff_l2243_224304

theorem fraction_meaningful_iff (x : ℝ) : (∃ y, y = x / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_iff_l2243_224304


namespace NUMINAMATH_GPT_total_practice_hours_l2243_224328

-- Definitions based on conditions
def weekday_practice_hours : ℕ := 3
def saturday_practice_hours : ℕ := 5
def weekdays_per_week : ℕ := 5
def weeks_until_game : ℕ := 3

-- Theorem statement
theorem total_practice_hours : (weekday_practice_hours * weekdays_per_week + saturday_practice_hours) * weeks_until_game = 60 := 
by sorry

end NUMINAMATH_GPT_total_practice_hours_l2243_224328


namespace NUMINAMATH_GPT_central_angle_of_sector_l2243_224394

theorem central_angle_of_sector 
  (r : ℝ) (s : ℝ) (c : ℝ)
  (h1 : r = 5)
  (h2 : s = 15)
  (h3 : c = 2 * π * r) :
  ∃ n : ℝ, (n * s * π / 180 = c) ∧ n = 120 :=
by
  use 120
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l2243_224394


namespace NUMINAMATH_GPT_sin_polar_circle_l2243_224361

theorem sin_polar_circle (t : ℝ) (θ : ℝ) (r : ℝ) (h : ∀ θ, 0 ≤ θ ∧ θ ≤ t → r = Real.sin θ) :
  t = Real.pi := 
by
  sorry

end NUMINAMATH_GPT_sin_polar_circle_l2243_224361


namespace NUMINAMATH_GPT_unattainable_y_l2243_224390

theorem unattainable_y (x : ℝ) (h : 4 * x + 5 ≠ 0) : 
  (y = (3 - x) / (4 * x + 5)) → (y ≠ -1/4) :=
sorry

end NUMINAMATH_GPT_unattainable_y_l2243_224390


namespace NUMINAMATH_GPT_converse_of_propositions_is_true_l2243_224392

theorem converse_of_propositions_is_true :
  (∀ x : ℝ, (x = 1 ∨ x = 2) ↔ (x^2 - 3 * x + 2 = 0)) ∧
  (∀ x y : ℝ, (x^2 + y^2 = 0) ↔ (x = 0 ∧ y = 0)) := 
by {
  sorry
}

end NUMINAMATH_GPT_converse_of_propositions_is_true_l2243_224392


namespace NUMINAMATH_GPT_highest_temperature_l2243_224338

theorem highest_temperature (lowest_temp : ℝ) (max_temp_diff : ℝ) :
  lowest_temp = 18 → max_temp_diff = 4 → lowest_temp + max_temp_diff = 22 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_highest_temperature_l2243_224338


namespace NUMINAMATH_GPT_arrangement_of_BANANA_l2243_224305

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_of_BANANA_l2243_224305


namespace NUMINAMATH_GPT_possible_values_l2243_224335

theorem possible_values (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) : 
  ∃ S : Set ℝ, S = {x : ℝ | 4 ≤ x} ∧ (1 / a + 1 / b) ∈ S :=
by
  sorry

end NUMINAMATH_GPT_possible_values_l2243_224335


namespace NUMINAMATH_GPT_students_with_same_grade_l2243_224398

theorem students_with_same_grade :
  let total_students := 40
  let students_with_same_A := 3
  let students_with_same_B := 2
  let students_with_same_C := 6
  let students_with_same_D := 1
  let total_same_grade_students := students_with_same_A + students_with_same_B + students_with_same_C + students_with_same_D
  total_same_grade_students = 12 →
  (total_same_grade_students / total_students) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_students_with_same_grade_l2243_224398


namespace NUMINAMATH_GPT_find_c_l2243_224343

theorem find_c (x c : ℝ) (h1 : 3 * x + 8 = 5) (h2 : c * x - 15 = -3) : c = -12 := 
by
  -- Equations and conditions
  have h1 : 3 * x + 8 = 5 := h1
  have h2 : c * x - 15 = -3 := h2
  -- The proof script would go here
  sorry

end NUMINAMATH_GPT_find_c_l2243_224343


namespace NUMINAMATH_GPT_max_tiles_on_floor_l2243_224399

theorem max_tiles_on_floor
  (tile_w tile_h floor_w floor_h : ℕ)
  (h_tile_w : tile_w = 25)
  (h_tile_h : tile_h = 65)
  (h_floor_w : floor_w = 150)
  (h_floor_h : floor_h = 390) :
  max ((floor_h / tile_h) * (floor_w / tile_w))
      ((floor_h / tile_w) * (floor_w / tile_h)) = 36 :=
by
  -- Given conditions and calculations will be proved in the proof.
  sorry

end NUMINAMATH_GPT_max_tiles_on_floor_l2243_224399


namespace NUMINAMATH_GPT_prove_m_plus_n_eq_one_l2243_224357

-- Define coordinates of points A and B
def A (m n : ℝ) : ℝ × ℝ := (1 + m, 1 - n)
def B : ℝ × ℝ := (-3, 2)

-- Define symmetry about the y-axis condition
def symmetric_about_y_axis (P Q : ℝ × ℝ) : Prop :=
  P.1 = -Q.1 ∧ P.2 = Q.2

-- Given conditions
def conditions (m n : ℝ) : Prop :=
  symmetric_about_y_axis (A m n) B

-- Statement to prove
theorem prove_m_plus_n_eq_one (m n : ℝ) (h : conditions m n) : m + n = 1 := 
by 
  sorry

end NUMINAMATH_GPT_prove_m_plus_n_eq_one_l2243_224357


namespace NUMINAMATH_GPT_altitude_correct_l2243_224387

-- Define the given sides and area of the triangle
def AB : ℝ := 30
def BC : ℝ := 17
def AC : ℝ := 25
def area_ABC : ℝ := 120

-- The length of the altitude from the vertex C to the base AB
def height_C_to_AB : ℝ := 8

-- Problem statement to be proven
theorem altitude_correct : (1 / 2) * AB * height_C_to_AB = area_ABC :=
by
  sorry

end NUMINAMATH_GPT_altitude_correct_l2243_224387


namespace NUMINAMATH_GPT_mul_99_105_l2243_224375

theorem mul_99_105 : 99 * 105 = 10395 := 
by
  -- Annotations and imports are handled; only the final Lean statement provided as requested.
  sorry

end NUMINAMATH_GPT_mul_99_105_l2243_224375


namespace NUMINAMATH_GPT_max_value_of_m_l2243_224332

theorem max_value_of_m :
  (∃ (t : ℝ), ∀ (x : ℝ), 2 ≤ x ∧ x ≤ m → (x + t)^2 ≤ 2 * x) → m ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_value_of_m_l2243_224332


namespace NUMINAMATH_GPT_second_part_lent_years_l2243_224365

theorem second_part_lent_years 
  (P1 P2 T : ℝ)
  (h1 : P1 + P2 = 2743)
  (h2 : P2 = 1688)
  (h3 : P1 * 0.03 * 8 = P2 * 0.05 * T) 
  : T = 3 :=
sorry

end NUMINAMATH_GPT_second_part_lent_years_l2243_224365


namespace NUMINAMATH_GPT_determine_q_l2243_224346

theorem determine_q (q : ℝ) (x1 x2 x3 x4 : ℝ) 
  (h_first_eq : x1^2 - 5 * x1 + q = 0 ∧ x2^2 - 5 * x2 + q = 0)
  (h_second_eq : x3^2 - 7 * x3 + 2 * q = 0 ∧ x4^2 - 7 * x4 + 2 * q = 0)
  (h_relation : x3 = 2 * x1) : 
  q = 6 :=
by
  sorry

end NUMINAMATH_GPT_determine_q_l2243_224346


namespace NUMINAMATH_GPT_candy_initial_amount_l2243_224379

namespace CandyProblem

variable (initial_candy given_candy left_candy : ℕ)

theorem candy_initial_amount (h1 : given_candy = 10) (h2 : left_candy = 68) (h3 : left_candy = initial_candy - given_candy) : initial_candy = 78 := 
  sorry
end CandyProblem

end NUMINAMATH_GPT_candy_initial_amount_l2243_224379


namespace NUMINAMATH_GPT_determine_k_value_l2243_224345

theorem determine_k_value (x y z k : ℝ) 
  (h1 : 5 / (x + y) = k / (x - z))
  (h2 : k / (x - z) = 9 / (z + y)) :
  k = 14 :=
sorry

end NUMINAMATH_GPT_determine_k_value_l2243_224345


namespace NUMINAMATH_GPT_youngest_son_trips_l2243_224342

theorem youngest_son_trips 
  (p : ℝ) (n_oldest : ℝ) (c : ℝ) (Y : ℝ)
  (h1 : p = 100)
  (h2 : n_oldest = 35)
  (h3 : c = 4)
  (h4 : p / c = Y) :
  Y = 25 := sorry

end NUMINAMATH_GPT_youngest_son_trips_l2243_224342


namespace NUMINAMATH_GPT_mike_bricks_l2243_224374

theorem mike_bricks (total_bricks bricks_A bricks_B bricks_other: ℕ) 
  (h1 : bricks_A = 40) 
  (h2 : bricks_B = bricks_A / 2)
  (h3 : total_bricks = 150) 
  (h4 : total_bricks = bricks_A + bricks_B + bricks_other) : bricks_other = 90 := 
by 
  sorry

end NUMINAMATH_GPT_mike_bricks_l2243_224374


namespace NUMINAMATH_GPT_circle_radius_l2243_224314

theorem circle_radius (x y : ℝ) : x^2 + 8*x + y^2 - 10*y + 32 = 0 → ∃ r : ℝ, r = 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l2243_224314


namespace NUMINAMATH_GPT_johnny_earnings_l2243_224350

theorem johnny_earnings :
  let job1 := 3 * 7
  let job2 := 2 * 10
  let job3 := 4 * 12
  let daily_earnings := job1 + job2 + job3
  let total_earnings := 5 * daily_earnings
  total_earnings = 445 :=
by
  sorry

end NUMINAMATH_GPT_johnny_earnings_l2243_224350


namespace NUMINAMATH_GPT_find_value_of_m_l2243_224327

/-- Given the universal set U, set A, and the complement of A in U, we prove that m = -2. -/
theorem find_value_of_m (m : ℤ) (U : Set ℤ) (A : Set ℤ) (complement_U_A : Set ℤ) 
  (h1 : U = {2, 3, m^2 + m - 4})
  (h2 : A = {m, 2})
  (h3 : complement_U_A = {3}) 
  (h4 : U = A ∪ complement_U_A) 
  (h5 : A ∩ complement_U_A = ∅) 
  : m = -2 :=
sorry

end NUMINAMATH_GPT_find_value_of_m_l2243_224327


namespace NUMINAMATH_GPT_jerry_cut_maple_trees_l2243_224362

theorem jerry_cut_maple_trees :
  (∀ pine maple walnut : ℕ, 
    pine = 8 * 80 ∧ 
    walnut = 4 * 100 ∧ 
    1220 = pine + walnut + maple * 60) → 
  maple = 3 := 
by 
  sorry

end NUMINAMATH_GPT_jerry_cut_maple_trees_l2243_224362


namespace NUMINAMATH_GPT_scientific_notation_110_billion_l2243_224320

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ 110 * 10^8 = a * 10^n

theorem scientific_notation_110_billion :
  ∃ (a : ℝ) (n : ℤ), scientific_notation_form a n ∧ a = 1.1 ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_110_billion_l2243_224320


namespace NUMINAMATH_GPT_sequence_not_generated_l2243_224334

theorem sequence_not_generated (a : ℕ → ℝ) :
  (a 1 = 2) ∧ (a 2 = 0) ∧ (a 3 = 2) ∧ (a 4 = 0) → 
  (∀ n, a n ≠ (1 - Real.cos (n * Real.pi)) + (n - 1) * (n - 2)) :=
by sorry

end NUMINAMATH_GPT_sequence_not_generated_l2243_224334


namespace NUMINAMATH_GPT_simplify_expression_l2243_224339

theorem simplify_expression :
  (2 : ℝ) * (2 * a) * (4 * a^2) * (3 * a^3) * (6 * a^4) = 288 * a^10 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l2243_224339


namespace NUMINAMATH_GPT_heartsuit_xx_false_l2243_224386

def heartsuit (x y : ℝ) : ℝ := |x - y|

theorem heartsuit_xx_false (x : ℝ) : heartsuit x x ≠ x :=
by sorry

end NUMINAMATH_GPT_heartsuit_xx_false_l2243_224386


namespace NUMINAMATH_GPT_area_comparison_l2243_224315

def point := (ℝ × ℝ)

def quadrilateral_I_vertices : List point := [(0, 0), (2, 0), (2, 2), (0, 2)]

def quadrilateral_I_area : ℝ := 4

def quadrilateral_II_vertices : List point := [(1, 0), (4, 0), (4, 4), (1, 3)]

noncomputable def quadrilateral_II_area : ℝ := 10.5

theorem area_comparison :
  quadrilateral_I_area < quadrilateral_II_area :=
  by
    sorry

end NUMINAMATH_GPT_area_comparison_l2243_224315


namespace NUMINAMATH_GPT_simplify_expression_l2243_224354

variable (a b c x y z : ℝ)

theorem simplify_expression :
  (cz * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + bz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cz + bz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cz * a^3 * y^3 + 3 * bz * c^3 * x^3) / (cz + bz) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2243_224354


namespace NUMINAMATH_GPT_find_x_l2243_224307

theorem find_x (x y : ℝ) (hx : x ≠ 0) (h1 : x / 2 = y^2) (h2 : x / 4 = 4 * y) : x = 128 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2243_224307


namespace NUMINAMATH_GPT_length_AC_correct_l2243_224382

noncomputable def length_AC (A B C D : Type) : ℝ := 105 / 17

variable {A B C D : Type}
variables (angle_BAC angle_ADB length_AD length_BC : ℝ)

theorem length_AC_correct
  (h1 : angle_BAC = 60)
  (h2 : angle_ADB = 30)
  (h3 : length_AD = 3)
  (h4 : length_BC = 9) :
  length_AC A B C D = 105 / 17 :=
sorry

end NUMINAMATH_GPT_length_AC_correct_l2243_224382


namespace NUMINAMATH_GPT_cost_per_serving_l2243_224383

-- Define the costs
def pasta_cost : ℝ := 1.00
def sauce_cost : ℝ := 2.00
def meatball_cost : ℝ := 5.00

-- Define the number of servings
def servings : ℝ := 8.0

-- State the theorem
theorem cost_per_serving : (pasta_cost + sauce_cost + meatball_cost) / servings = 1.00 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_serving_l2243_224383


namespace NUMINAMATH_GPT_initial_garrison_men_l2243_224336

theorem initial_garrison_men (M : ℕ) (H1 : ∃ provisions : ℕ, provisions = M * 60)
  (H2 : ∃ provisions_15 : ℕ, provisions_15 = M * 45)
  (H3 : ∀ provisions_15 (new_provisions: ℕ), (provisions_15 = M * 45 ∧ new_provisions = 20 * (M + 1250)) → provisions_15 = new_provisions) :
  M = 1000 :=
by
  sorry

end NUMINAMATH_GPT_initial_garrison_men_l2243_224336


namespace NUMINAMATH_GPT_suzanne_donation_total_l2243_224358

theorem suzanne_donation_total : 
  (10 + 10 * 2 + 10 * 2^2 + 10 * 2^3 + 10 * 2^4 = 310) :=
by
  sorry

end NUMINAMATH_GPT_suzanne_donation_total_l2243_224358


namespace NUMINAMATH_GPT_line_circle_no_intersection_l2243_224364

theorem line_circle_no_intersection : 
  (∀ x y : ℝ, 3 * x + 4 * y = 12 → (x - 1)^2 + (y + 1)^2 ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_line_circle_no_intersection_l2243_224364


namespace NUMINAMATH_GPT_perfect_square_fraction_l2243_224353

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem perfect_square_fraction (a b : ℕ) 
  (h_pos_a: 0 < a) 
  (h_pos_b: 0 < b) 
  (h_div : (a * b + 1) ∣ (a^2 + b^2)) : 
  is_perfect_square ((a^2 + b^2) / (a * b + 1)) := 
sorry

end NUMINAMATH_GPT_perfect_square_fraction_l2243_224353


namespace NUMINAMATH_GPT_cost_per_tissue_box_l2243_224360

-- Given conditions
def rolls_toilet_paper : ℝ := 10
def cost_per_toilet_paper : ℝ := 1.5
def rolls_paper_towels : ℝ := 7
def cost_per_paper_towel : ℝ := 2
def boxes_tissues : ℝ := 3
def total_cost : ℝ := 35

-- Deduction of individual costs
def cost_toilet_paper := rolls_toilet_paper * cost_per_toilet_paper
def cost_paper_towels := rolls_paper_towels * cost_per_paper_towel
def cost_tissues := total_cost - cost_toilet_paper - cost_paper_towels

-- Prove the cost for one box of tissues
theorem cost_per_tissue_box : (cost_tissues / boxes_tissues) = 2 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_tissue_box_l2243_224360


namespace NUMINAMATH_GPT_intersection_complement_A_B_subset_A_C_l2243_224384

-- Definition of sets A, B, and complements in terms of conditions
def setA : Set ℝ := { x | 3 ≤ x ∧ x < 7 }
def setB : Set ℝ := { x | 2 < x ∧ x < 10 }
def complement_A : Set ℝ := { x | x < 3 ∨ x ≥ 7 }

-- Proof Problem (1)
theorem intersection_complement_A_B :
  ((complement_A) ∩ setB) = { x | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10) } := 
  sorry

-- Definition of set C 
def setC (a : ℝ) : Set ℝ := { x | x < a }
-- Proof Problem (2)
theorem subset_A_C {a : ℝ} (h : setA ⊆ setC a) : a ≥ 7 :=
  sorry

end NUMINAMATH_GPT_intersection_complement_A_B_subset_A_C_l2243_224384


namespace NUMINAMATH_GPT_range_of_a_l2243_224317

def A := {x : ℝ | x * (4 - x) ≥ 3}
def B (a : ℝ) := {x : ℝ | x > a}

theorem range_of_a (a : ℝ) : (A ∩ B a = A) ↔ (a < 1) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l2243_224317


namespace NUMINAMATH_GPT_ab_value_l2243_224376

theorem ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by 
  sorry

end NUMINAMATH_GPT_ab_value_l2243_224376


namespace NUMINAMATH_GPT_find_a_b_a_b_values_l2243_224369

/-
Define the matrix M as given in the problem.
Define the constants a and b, and state the condition that proves their correct values such that M_inv = a * M + b * I.
-/

open Matrix

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ :=
  !![2, 0;
     1, -3]

noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  !![1/2, 0;
     1/6, -1/3]

theorem find_a_b :
  ∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
sorry

theorem a_b_values :
  (∃ (a b : ℚ), (M⁻¹) = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ)) ∧
  (∃ a b : ℚ, a = 1/6 ∧ b = 1/6) :=
sorry

end NUMINAMATH_GPT_find_a_b_a_b_values_l2243_224369


namespace NUMINAMATH_GPT_greatest_possible_value_x_l2243_224367

theorem greatest_possible_value_x :
  ∀ x : ℚ, (∃ y : ℚ, y = (5 * x - 25) / (4 * x - 5) ∧ y^2 + y = 18) →
  x ≤ 55 / 29 :=
by sorry

end NUMINAMATH_GPT_greatest_possible_value_x_l2243_224367


namespace NUMINAMATH_GPT_sqrt_real_domain_l2243_224323

theorem sqrt_real_domain (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 := 
sorry

end NUMINAMATH_GPT_sqrt_real_domain_l2243_224323


namespace NUMINAMATH_GPT_calculate_sum_l2243_224359

theorem calculate_sum :
  (1 : ℚ) + 3 / 6 + 5 / 12 + 7 / 20 + 9 / 30 + 11 / 42 + 13 / 56 + 15 / 72 + 17 / 90 = 81 + 2 / 5 :=
sorry

end NUMINAMATH_GPT_calculate_sum_l2243_224359


namespace NUMINAMATH_GPT_rationalize_denominator_l2243_224303

theorem rationalize_denominator (h : Real.sqrt 200 = 10 * Real.sqrt 2) : 
  (7 / Real.sqrt 200) = (7 * Real.sqrt 2 / 20) :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l2243_224303


namespace NUMINAMATH_GPT_quadratic_has_one_real_root_positive_value_of_m_l2243_224316

theorem quadratic_has_one_real_root (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 0 ∨ m = 1/4 := by
  sorry

theorem positive_value_of_m (m : ℝ) (h : (4 * m) * (4 * m) - 4 * 1 * m = 0) : m = 1/4 := by
  have root_cases := quadratic_has_one_real_root m h
  cases root_cases
  · exfalso
    -- We know m = 0 cannot be the positive m we are looking for.
    sorry
  · assumption

end NUMINAMATH_GPT_quadratic_has_one_real_root_positive_value_of_m_l2243_224316


namespace NUMINAMATH_GPT_measure_of_obtuse_angle_APB_l2243_224331

-- Define the triangle type and conditions
structure Triangle :=
  (A B C : Point)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)

-- Define the point type
structure Point :=
  (x y : ℝ)

-- Property of the triangle is isotropic and it contains right angles 90 degrees 
def IsoscelesRightTriangle (T : Triangle) : Prop :=
  T.angle_A = 45 ∧ T.angle_B = 45 ∧ T.angle_C = 90

-- Define the angle bisector intersection point P
def AngleBisectorIntersection (T : Triangle) (P : Point) : Prop :=
  -- (dummy properties assuming necessary geometric constructions can be proven)
  true

-- Statement we want to prove
theorem measure_of_obtuse_angle_APB (T : Triangle) (P : Point) 
    (h1 : IsoscelesRightTriangle T) (h2 : AngleBisectorIntersection T P) :
  ∃ APB : ℝ, APB = 135 :=
  sorry

end NUMINAMATH_GPT_measure_of_obtuse_angle_APB_l2243_224331


namespace NUMINAMATH_GPT_ways_to_climb_four_steps_l2243_224391

theorem ways_to_climb_four_steps (ways_to_climb : ℕ → ℕ) 
  (h1 : ways_to_climb 1 = 1) 
  (h2 : ways_to_climb 2 = 2) 
  (h3 : ways_to_climb 3 = 3) 
  (h_step : ∀ n, ways_to_climb n = ways_to_climb (n - 1) + ways_to_climb (n - 2)) : 
  ways_to_climb 4 = 5 := 
sorry

end NUMINAMATH_GPT_ways_to_climb_four_steps_l2243_224391


namespace NUMINAMATH_GPT_johns_average_speed_is_correct_l2243_224341

noncomputable def johnsAverageSpeed : ℝ :=
  let total_time : ℝ := 6 + 0.5 -- Total driving time in hours
  let total_distance : ℝ := 210 -- Total distance covered in miles
  total_distance / total_time -- Average speed formula

theorem johns_average_speed_is_correct :
  johnsAverageSpeed = 32.31 :=
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_johns_average_speed_is_correct_l2243_224341


namespace NUMINAMATH_GPT_original_length_of_tape_l2243_224322

-- Given conditions
variables (L : Real) (used_by_Remaining_yesterday : L * (1 - 1 / 5) = 4 / 5 * L)
          (remaining_after_today : 1.5 = 4 / 5 * L * 1 / 4)

-- The theorem to prove
theorem original_length_of_tape (L : Real) 
  (used_by_Remaining_yesterday : L * (1 - 1 / 5) = 4 / 5 * L)
  (remaining_after_today : 1.5 = 4 / 5 * L * 1 / 4) :
  L = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_original_length_of_tape_l2243_224322


namespace NUMINAMATH_GPT_cube_volume_l2243_224352

theorem cube_volume (s : ℝ) (h1 : 6 * s^2 = 1734) : s^3 = 4913 := by
  sorry

end NUMINAMATH_GPT_cube_volume_l2243_224352


namespace NUMINAMATH_GPT_fencing_rate_correct_l2243_224310

noncomputable def rate_of_fencing_per_meter (area_hectares : ℝ) (total_cost : ℝ) : ℝ :=
  let area_sqm := area_hectares * 10000
  let r_squared := area_sqm / Real.pi
  let r := Real.sqrt r_squared
  let circumference := 2 * Real.pi * r
  total_cost / circumference

theorem fencing_rate_correct :
  rate_of_fencing_per_meter 13.86 6070.778380479544 = 4.60 :=
by
  sorry

end NUMINAMATH_GPT_fencing_rate_correct_l2243_224310


namespace NUMINAMATH_GPT_lucy_found_shells_l2243_224377

theorem lucy_found_shells (original current : ℕ) (h1 : original = 68) (h2 : current = 89) : current - original = 21 :=
by {
    sorry
}

end NUMINAMATH_GPT_lucy_found_shells_l2243_224377


namespace NUMINAMATH_GPT_problem_solution_l2243_224300

theorem problem_solution
  {a b c d : ℝ}
  (h1 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h2 : (a^2012 - c^2012) * (a^2012 - d^2012) = 2011)
  (h3 : (b^2012 - c^2012) * (b^2012 - d^2012) = 2011) :
  (c * d)^2012 - (a * b)^2012 = 2011 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l2243_224300


namespace NUMINAMATH_GPT_number_of_blue_balls_l2243_224389

theorem number_of_blue_balls (b : ℕ) 
  (h1 : 0 < b ∧ b ≤ 15)
  (prob : (b / 15) * ((b - 1) / 14) = 1 / 21) :
  b = 5 := sorry

end NUMINAMATH_GPT_number_of_blue_balls_l2243_224389


namespace NUMINAMATH_GPT_Geraldine_more_than_Jazmin_l2243_224318

-- Define the number of dolls Geraldine and Jazmin have
def Geraldine_dolls : ℝ := 2186.0
def Jazmin_dolls : ℝ := 1209.0

-- State the theorem we need to prove
theorem Geraldine_more_than_Jazmin :
  Geraldine_dolls - Jazmin_dolls = 977.0 := 
by
  sorry

end NUMINAMATH_GPT_Geraldine_more_than_Jazmin_l2243_224318


namespace NUMINAMATH_GPT_expensive_time_8_l2243_224347

variable (x : ℝ) -- x represents the time to pick an expensive handcuff lock

-- Conditions
def cheap_time := 6
def total_time := 42
def cheap_pairs := 3
def expensive_pairs := 3

-- Total time for cheap handcuffs
def total_cheap_time := cheap_pairs * cheap_time

-- Total time for expensive handcuffs
def total_expensive_time := total_time - total_cheap_time

-- Equation relating x to total_expensive_time
def expensive_equation := expensive_pairs * x = total_expensive_time

-- Proof goal
theorem expensive_time_8 : expensive_equation x -> x = 8 := by
  sorry

end NUMINAMATH_GPT_expensive_time_8_l2243_224347


namespace NUMINAMATH_GPT_ratio_of_x_and_y_l2243_224333

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_and_y_l2243_224333


namespace NUMINAMATH_GPT_find_tangent_c_l2243_224396

theorem find_tangent_c (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → (-12)^2 - 4 * (1) * (12 * c) = 0) → c = 3 :=
sorry

end NUMINAMATH_GPT_find_tangent_c_l2243_224396


namespace NUMINAMATH_GPT_alok_age_proof_l2243_224308

variable (A B C : ℕ)

theorem alok_age_proof (h1 : B = 6 * A) (h2 : B + 10 = 2 * (C + 10)) (h3 : C = 10) : A = 5 :=
by
  sorry

end NUMINAMATH_GPT_alok_age_proof_l2243_224308


namespace NUMINAMATH_GPT_rectangle_area_solution_l2243_224385

theorem rectangle_area_solution (x : ℝ) (h1 : (x + 3) * (2*x - 1) = 12*x + 5) : 
  x = (7 + Real.sqrt 113) / 4 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_solution_l2243_224385


namespace NUMINAMATH_GPT_max_value_of_f_range_of_m_l2243_224324

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x - b * x^2

theorem max_value_of_f (a b : ℝ) (x : ℝ) (h1 : 1 / Real.exp 1 ≤ x ∧ x ≤ Real.exp 1)
  (h_tangent : ∀ (x : ℝ), f a b x - ((-1/2) * x + (Real.log 1 - 1/2)) = 0) : 
  ∃ x_max, f a b x_max = -1/2 := sorry

theorem range_of_m (m : ℝ) 
  (h_ineq : ∀ (a : ℝ) (x : ℝ), 1 ≤ a ∧ a ≤ 3 / 2 ∧ 1 ≤ x ∧ x ≤ Real.exp 2 → a * Real.log x ≥ m + x) : 
  m ≤ 2 - Real.exp 2 := sorry

end NUMINAMATH_GPT_max_value_of_f_range_of_m_l2243_224324


namespace NUMINAMATH_GPT_flower_pattern_perimeter_l2243_224321

theorem flower_pattern_perimeter (r : ℝ) (θ : ℝ) (h_r : r = 3) (h_θ : θ = 45) : 
    let arc_length := (360 - θ) / 360 * 2 * π * r
    let total_perimeter := arc_length + 2 * r
    total_perimeter = (21 / 4 * π) + 6 := 
by
  -- Definitions from conditions
  let arc_length := (360 - θ) / 360 * 2 * π * r
  let total_perimeter := arc_length + 2 * r

  -- Assertions to reach the target conclusion
  have h_arc_length: arc_length = (21 / 4 * π) :=
    by
      sorry

  -- Incorporate the radius
  have h_total: total_perimeter = (21 / 4 * π) + 6 :=
    by
      sorry

  exact h_total

end NUMINAMATH_GPT_flower_pattern_perimeter_l2243_224321


namespace NUMINAMATH_GPT_arithmetic_progression_implies_equality_l2243_224363

theorem arithmetic_progression_implies_equality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ((a + b) / 2) = ((Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2)) / 2) → a = b :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_implies_equality_l2243_224363


namespace NUMINAMATH_GPT_no_real_solution_l2243_224311

-- Given conditions as definitions in Lean 4
def eq1 (x : ℝ) : Prop := x^5 + 3 * x^4 + 5 * x^3 + 5 * x^2 + 6 * x + 2 = 0
def eq2 (x : ℝ) : Prop := x^3 + 3 * x^2 + 4 * x + 1 = 0

-- The theorem to prove
theorem no_real_solution : ¬ ∃ x : ℝ, eq1 x ∧ eq2 x :=
by sorry

end NUMINAMATH_GPT_no_real_solution_l2243_224311


namespace NUMINAMATH_GPT_led_message_count_l2243_224366

theorem led_message_count : 
  let n := 7
  let colors := 2
  let lit_leds := 3
  let non_adjacent_combinations := 10
  (non_adjacent_combinations * (colors ^ lit_leds)) = 80 :=
by
  sorry

end NUMINAMATH_GPT_led_message_count_l2243_224366


namespace NUMINAMATH_GPT_circle_trajectory_l2243_224370

theorem circle_trajectory (a b : ℝ) :
  ∃ x y : ℝ, (b - 3)^2 + a^2 = (b + 3)^2 → x^2 = 12 * y := 
sorry

end NUMINAMATH_GPT_circle_trajectory_l2243_224370


namespace NUMINAMATH_GPT_karen_starts_late_by_4_minutes_l2243_224309

-- Define conditions as Lean 4 variables/constants
noncomputable def karen_speed : ℝ := 60 -- in mph
noncomputable def tom_speed : ℝ := 45 -- in mph
noncomputable def tom_distance : ℝ := 24 -- in miles
noncomputable def karen_lead : ℝ := 4 -- in miles

-- Main theorem statement
theorem karen_starts_late_by_4_minutes : 
  ∃ (minutes_late : ℝ), minutes_late = 4 :=
by
  -- Calculations based on given conditions provided in the problem
  let t := tom_distance / tom_speed -- Time for Tom to drive 24 miles
  let tk := (tom_distance + karen_lead) / karen_speed -- Time for Karen to drive 28 miles
  let time_difference := t - tk -- Time difference between Tom and Karen
  let minutes_late := time_difference * 60 -- Convert time difference to minutes
  existsi minutes_late -- Existential quantifier to state the existence of such a time
  have h : minutes_late = 4 := sorry -- Placeholder for demonstrating equality
  exact h

end NUMINAMATH_GPT_karen_starts_late_by_4_minutes_l2243_224309


namespace NUMINAMATH_GPT_triangle_third_side_l2243_224313

theorem triangle_third_side (x : ℕ) : 
  (3 < x) ∧ (x < 17) → 
  (x = 11) :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_l2243_224313


namespace NUMINAMATH_GPT_arithmetic_sequence_8th_term_l2243_224372

theorem arithmetic_sequence_8th_term 
    (a₁ : ℝ) (a₅ : ℝ) (n : ℕ) (a₈ : ℝ) 
    (h₁ : a₁ = 3) 
    (h₂ : a₅ = 78) 
    (h₃ : n = 25) : 
    a₈ = 24.875 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_8th_term_l2243_224372
