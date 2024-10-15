import Mathlib

namespace NUMINAMATH_GPT_relationship_between_y1_y2_y3_l553_55347

-- Define the parabola equation and points
def parabola (x c : ℝ) : ℝ := 2 * (x - 1)^2 + c

-- Define the points
def point1 := -2
def point2 := 0
def point3 := 5 / 3

-- Define the y values at these points
def y1 (c : ℝ) := parabola point1 c
def y2 (c : ℝ) := parabola point2 c
def y3 (c : ℝ) := parabola point3 c

-- Proof statement
theorem relationship_between_y1_y2_y3 (c : ℝ) : 
  y1 c > y2 c ∧ y2 c > y3 c :=
sorry

end NUMINAMATH_GPT_relationship_between_y1_y2_y3_l553_55347


namespace NUMINAMATH_GPT_equiangular_polygons_unique_solution_l553_55324

theorem equiangular_polygons_unique_solution :
  ∃! (n1 n2 : ℕ), (n1 ≠ 0 ∧ n2 ≠ 0) ∧ (180 / n1 + 360 / n2 = 90) :=
by
  sorry

end NUMINAMATH_GPT_equiangular_polygons_unique_solution_l553_55324


namespace NUMINAMATH_GPT_percentage_increase_sale_l553_55359

theorem percentage_increase_sale (P S : ℝ) (hP : 0 < P) (hS : 0 < S) :
  let new_price := 0.65 * P
  let original_revenue := P * S
  let new_revenue := 1.17 * original_revenue
  let percentage_increase := 80 / 100
  let new_sales := S * (1 + percentage_increase)
  new_price * new_sales = new_revenue :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_sale_l553_55359


namespace NUMINAMATH_GPT_suitable_for_systematic_sampling_l553_55307

def city_districts : ℕ := 2000
def student_ratio : List ℕ := [3, 2, 8, 2]
def sample_size_city : ℕ := 200
def total_components : ℕ := 2000

def condition_A : Prop := 
  city_districts = 2000 ∧ 
  student_ratio = [3, 2, 8, 2] ∧ 
  sample_size_city = 200

def condition_B : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 5

def condition_C : Prop := 
  ∃ (n : ℕ), n = total_components ∧ n = 200

def condition_D : Prop := 
  ∃ (n : ℕ), n = 20 ∧ n = 5

theorem suitable_for_systematic_sampling : condition_C :=
by
  sorry

end NUMINAMATH_GPT_suitable_for_systematic_sampling_l553_55307


namespace NUMINAMATH_GPT_first_car_speed_l553_55351

theorem first_car_speed
  (highway_length : ℝ)
  (second_car_speed : ℝ)
  (meeting_time : ℝ)
  (D1 D2 : ℝ) :
  highway_length = 45 → second_car_speed = 16 → meeting_time = 1.5 → D2 = second_car_speed * meeting_time → D1 + D2 = highway_length → D1 = 14 * meeting_time :=
by
  intros h_highway h_speed h_time h_D2 h_sum
  sorry

end NUMINAMATH_GPT_first_car_speed_l553_55351


namespace NUMINAMATH_GPT_find_number_l553_55380

theorem find_number (x : ℕ) (h : 5 * x = 100) : x = 20 :=
sorry

end NUMINAMATH_GPT_find_number_l553_55380


namespace NUMINAMATH_GPT_eunji_class_total_students_l553_55384

variable (A B : Finset ℕ) (universe_students : Finset ℕ)

axiom students_play_instrument_a : A.card = 24
axiom students_play_instrument_b : B.card = 17
axiom students_play_both_instruments : (A ∩ B).card = 8
axiom no_students_without_instruments : A ∪ B = universe_students

theorem eunji_class_total_students : universe_students.card = 33 := by
  sorry

end NUMINAMATH_GPT_eunji_class_total_students_l553_55384


namespace NUMINAMATH_GPT_vacation_cost_per_person_l553_55309

theorem vacation_cost_per_person (airbnb_cost car_cost : ℝ) (num_people : ℝ) 
  (h1 : airbnb_cost = 3200) (h2 : car_cost = 800) (h3 : num_people = 8) : 
  (airbnb_cost + car_cost) / num_people = 500 := 
by 
  sorry

end NUMINAMATH_GPT_vacation_cost_per_person_l553_55309


namespace NUMINAMATH_GPT_find_first_dimension_l553_55311

variable (w h cost_per_sqft total_cost : ℕ)

def surface_area (l w h : ℕ) : ℕ := 2 * l * w + 2 * l * h + 2 * w * h

def insulation_cost (A cost_per_sqft : ℕ) : ℕ := A * cost_per_sqft

theorem find_first_dimension 
  (w := 7) (h := 2) (cost_per_sqft := 20) (total_cost := 1640) : 
  (∃ l : ℕ, insulation_cost (surface_area l w h) cost_per_sqft = total_cost) → 
  l = 3 := 
sorry

end NUMINAMATH_GPT_find_first_dimension_l553_55311


namespace NUMINAMATH_GPT_smallest_x_l553_55302

theorem smallest_x (x y : ℝ) (h1 : 4 < x) (h2 : x < 6) (h3 : 6 < y) (h4 : y < 10) (h5 : y - x = 5) :
  x = 4 :=
sorry

end NUMINAMATH_GPT_smallest_x_l553_55302


namespace NUMINAMATH_GPT_unique_A3_zero_l553_55386

variable {F : Type*} [Field F]

theorem unique_A3_zero (A : Matrix (Fin 2) (Fin 2) F) 
  (h1 : A ^ 4 = 0) 
  (h2 : Matrix.trace A = 0) : 
  A ^ 3 = 0 :=
sorry

end NUMINAMATH_GPT_unique_A3_zero_l553_55386


namespace NUMINAMATH_GPT_price_per_foot_of_fence_l553_55372

theorem price_per_foot_of_fence (area : ℝ) (total_cost : ℝ) (side_length : ℝ) (perimeter : ℝ) (price_per_foot : ℝ) 
  (h1 : area = 289) (h2 : total_cost = 3672) (h3 : side_length = Real.sqrt area) (h4 : perimeter = 4 * side_length) (h5 : price_per_foot = total_cost / perimeter) :
  price_per_foot = 54 := by
  sorry

end NUMINAMATH_GPT_price_per_foot_of_fence_l553_55372


namespace NUMINAMATH_GPT_medical_team_selection_l553_55334

theorem medical_team_selection : 
  let male_doctors := 6
  let female_doctors := 5
  let choose_male := Nat.choose male_doctors 2
  let choose_female := Nat.choose female_doctors 1
  choose_male * choose_female = 75 := 
by 
  sorry

end NUMINAMATH_GPT_medical_team_selection_l553_55334


namespace NUMINAMATH_GPT_prob_Z_l553_55396

theorem prob_Z (P_X P_Y P_W P_Z : ℚ) (hX : P_X = 1/4) (hY : P_Y = 1/3) (hW : P_W = 1/6) 
(hSum : P_X + P_Y + P_Z + P_W = 1) : P_Z = 1/4 := 
by
  -- The proof will be filled in later
  sorry

end NUMINAMATH_GPT_prob_Z_l553_55396


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l553_55326

theorem largest_angle_in_triangle (a b c : ℝ)
  (h1 : a + b = (4 / 3) * 90)
  (h2 : b = a + 36)
  (h3 : a + b + c = 180) :
  max a (max b c) = 78 :=
sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l553_55326


namespace NUMINAMATH_GPT_solve_congruence_l553_55398

theorem solve_congruence (x : ℤ) : 
  (10 * x + 3) % 18 = 11 % 18 → x % 9 = 8 % 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_congruence_l553_55398


namespace NUMINAMATH_GPT_inv_matrix_eq_l553_55303

variable (a : ℝ)
variable (A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 3; 1, a])
variable (A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![a, -3; -1, a])

theorem inv_matrix_eq : (A⁻¹ = A_inv) → (a = 2) := 
by 
  sorry

end NUMINAMATH_GPT_inv_matrix_eq_l553_55303


namespace NUMINAMATH_GPT_first_marvelous_monday_after_school_starts_l553_55381

def is_marvelous_monday (year : ℕ) (month : ℕ) (day : ℕ) (start_day : ℕ) : Prop :=
  let days_in_month := if month = 9 then 30 else if month = 10 then 31 else 0
  let fifth_monday := start_day + 28
  let is_monday := (fifth_monday - 1) % 7 = 0
  month = 10 ∧ day = 30 ∧ is_monday

theorem first_marvelous_monday_after_school_starts :
  ∃ (year month day : ℕ),
    year = 2023 ∧ month = 10 ∧ day = 30 ∧ is_marvelous_monday year month day 4 := sorry

end NUMINAMATH_GPT_first_marvelous_monday_after_school_starts_l553_55381


namespace NUMINAMATH_GPT_sum_of_a_and_b_l553_55385

variables {a b m : ℝ}

theorem sum_of_a_and_b (h1 : a^2 + a * b = 16 + m) (h2 : b^2 + a * b = 9 - m) : a + b = 5 ∨ a + b = -5 :=
by sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l553_55385


namespace NUMINAMATH_GPT_num_pens_l553_55339

theorem num_pens (pencils : ℕ) (students : ℕ) (pens : ℕ)
  (h_pencils : pencils = 520)
  (h_students : students = 40)
  (h_div : pencils % students = 0)
  (h_pens_per_student : pens = (pencils / students) * students) :
  pens = 520 := by
  sorry

end NUMINAMATH_GPT_num_pens_l553_55339


namespace NUMINAMATH_GPT_find_a10_l553_55305

-- Conditions
variables (S : ℕ → ℕ) (a : ℕ → ℕ)
variables (hS9 : S 9 = 81) (ha2 : a 2 = 3)

-- Arithmetic sequence sum definition
def arithmetic_sequence_sum (n : ℕ) (a1 : ℕ) (d : ℕ) :=
  n * (2 * a1 + (n - 1) * d) / 2

-- a_n formula definition
def a_n (n a1 d : ℕ) := a1 + (n - 1) * d

-- Proof statement
theorem find_a10 (a1 d : ℕ) (hS9' : 9 * (2 * a1 + 8 * d) / 2 = 81) (ha2' : a1 + d = 3) :
  a 10 = a1 + 9 * d :=
sorry

end NUMINAMATH_GPT_find_a10_l553_55305


namespace NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l553_55337

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 20) (h2 : x * y = 9) : x^2 + y^2 = 418 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_squared_plus_y_squared_l553_55337


namespace NUMINAMATH_GPT_number_of_multiples_of_6_between_5_and_125_l553_55345

theorem number_of_multiples_of_6_between_5_and_125 : 
  ∃ k : ℕ, (5 < 6 * k ∧ 6 * k < 125) → k = 20 :=
sorry

end NUMINAMATH_GPT_number_of_multiples_of_6_between_5_and_125_l553_55345


namespace NUMINAMATH_GPT_leo_current_weight_l553_55383

variables (L K J : ℝ)

def condition1 := L + 12 = 1.7 * K
def condition2 := L + K + J = 270
def condition3 := J = K + 30

theorem leo_current_weight (h1 : condition1 L K)
                           (h2 : condition2 L K J)
                           (h3 : condition3 K J) : L = 103.6 :=
sorry

end NUMINAMATH_GPT_leo_current_weight_l553_55383


namespace NUMINAMATH_GPT_gcd_lcm_45_150_l553_55322

theorem gcd_lcm_45_150 : Nat.gcd 45 150 = 15 ∧ Nat.lcm 45 150 = 450 :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_45_150_l553_55322


namespace NUMINAMATH_GPT_max_intersections_two_circles_three_lines_l553_55352

theorem max_intersections_two_circles_three_lines :
  ∀ (C1 C2 : ℝ × ℝ × ℝ) (L1 L2 L3 : ℝ × ℝ × ℝ), 
  C1 ≠ C2 → L1 ≠ L2 → L2 ≠ L3 → L1 ≠ L3 →
  ∃ (P : ℕ), P = 17 :=
by 
  sorry

end NUMINAMATH_GPT_max_intersections_two_circles_three_lines_l553_55352


namespace NUMINAMATH_GPT_problem_to_prove_l553_55364

theorem problem_to_prove
  (a b c : ℝ)
  (h1 : a + b + c = -3)
  (h2 : a * b + b * c + c * a = -10)
  (h3 : a * b * c = -5) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 = 70 :=
by
  sorry

end NUMINAMATH_GPT_problem_to_prove_l553_55364


namespace NUMINAMATH_GPT_sqrt_domain_condition_l553_55301

theorem sqrt_domain_condition (x : ℝ) : (2 * x - 6 ≥ 0) ↔ (x ≥ 3) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_domain_condition_l553_55301


namespace NUMINAMATH_GPT_sum_of_cubes_l553_55300

variable (a b c : ℝ)

theorem sum_of_cubes (h1 : a^2 + 3 * b = 2) (h2 : b^2 + 5 * c = 3) (h3 : c^2 + 7 * a = 6) :
  a^3 + b^3 + c^3 = -0.875 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l553_55300


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l553_55325

theorem sufficient_but_not_necessary_condition (a b : ℝ) (h : a > b ∧ b > 0) : (1 / a < 1 / b) ∧ ¬ (1 / a < 1 / b → a > b ∧ b > 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l553_55325


namespace NUMINAMATH_GPT_real_solutions_of_equation_l553_55342

theorem real_solutions_of_equation : 
  ∃! x₁ x₂ : ℝ, (3 * x₁^2 - 10 * x₁ + 7 = 0) ∧ (3 * x₂^2 - 10 * x₂ + 7 = 0) ∧ x₁ ≠ x₂ :=
sorry

end NUMINAMATH_GPT_real_solutions_of_equation_l553_55342


namespace NUMINAMATH_GPT_arcsin_half_eq_pi_six_arccos_sqrt_three_over_two_eq_pi_six_l553_55371

theorem arcsin_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := by
  sorry

theorem arccos_sqrt_three_over_two_eq_pi_six : Real.arccos (Real.sqrt 3 / 2) = Real.pi / 6 := by
  sorry

end NUMINAMATH_GPT_arcsin_half_eq_pi_six_arccos_sqrt_three_over_two_eq_pi_six_l553_55371


namespace NUMINAMATH_GPT_number_of_females_l553_55304

-- Definitions
variable (F : ℕ) -- ℕ = Natural numbers, ensuring F is a non-negative integer
variable (h_male : ℕ := 2 * F)
variable (h_total : F + 2 * F = 18)
variable (h_female_pos : F > 0)

-- Theorem
theorem number_of_females (F : ℕ) (h_male : ℕ := 2 * F) (h_total : F + 2 * F = 18) (h_female_pos : F > 0) : F = 6 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_females_l553_55304


namespace NUMINAMATH_GPT_least_possible_b_l553_55331

theorem least_possible_b (a b : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (h_a_factors : ∃ k, a = p^k ∧ k + 1 = 3) (h_b_factors : ∃ m, b = p^m ∧ m + 1 = a) (h_divisible : b % a = 0) : 
  b = 8 := 
by 
  sorry

end NUMINAMATH_GPT_least_possible_b_l553_55331


namespace NUMINAMATH_GPT_equation_solution_l553_55328

theorem equation_solution (x : ℝ) (h : x ≠ 1 ∧ x ≠ 2 ∧ x ≠ 3 ∧ x ≠ 4 ∧ x ≠ 5) :
  (1 / ((x - 1) * (x - 2)) + 1 / ((x - 2) * (x - 3)) + 
   1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 4) ↔ 
  (x = 3 + 2 * Real.sqrt 5 ∨ x = 3 - 2 * Real.sqrt 5) := 
sorry

end NUMINAMATH_GPT_equation_solution_l553_55328


namespace NUMINAMATH_GPT_simplify_rationalize_l553_55361

theorem simplify_rationalize
  : (1 / (1 + (1 / (Real.sqrt 5 + 2)))) = ((Real.sqrt 5 + 1) / 4) := 
sorry

end NUMINAMATH_GPT_simplify_rationalize_l553_55361


namespace NUMINAMATH_GPT_count_whole_numbers_between_4_and_18_l553_55317

theorem count_whole_numbers_between_4_and_18 :
  ∀ (x : ℕ), 4 < x ∧ x < 18 ↔ ∃ n : ℕ, n = 13 :=
by sorry

end NUMINAMATH_GPT_count_whole_numbers_between_4_and_18_l553_55317


namespace NUMINAMATH_GPT_range_of_a_l553_55346

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 < 0) ↔ (a ∈ [-1, 3]) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l553_55346


namespace NUMINAMATH_GPT_original_portion_al_l553_55330

variable (a b c : ℕ)

theorem original_portion_al :
  a + b + c = 1200 ∧
  a - 150 + 3 * b + 3 * c = 1800 ∧
  c = 2 * b →
  a = 825 :=
by
  sorry

end NUMINAMATH_GPT_original_portion_al_l553_55330


namespace NUMINAMATH_GPT_find_k_l553_55354

theorem find_k 
  (m_eq : ∀ x : ℝ, ∃ y : ℝ, y = 4 * x + 2)
  (n_eq : ∀ x : ℝ, ∃ y : ℝ, y = k * x - 8)
  (intersect : ∃ x y : ℝ, x = -2 ∧ y = -6 ∧ 4 * x + 2 = y ∧ k * x - 8 = y) :
  k = -1 := 
sorry

end NUMINAMATH_GPT_find_k_l553_55354


namespace NUMINAMATH_GPT_time_saved_by_both_trains_trainB_distance_l553_55358

-- Define the conditions
def trainA_speed_reduced := 360 / 12  -- 30 miles/hour
def trainB_speed_reduced := 360 / 8   -- 45 miles/hour

def trainA_speed := trainA_speed_reduced / (2 / 3)  -- 45 miles/hour
def trainB_speed := trainB_speed_reduced / (1 / 2)  -- 90 miles/hour

def trainA_time_saved := 12 - (360 / trainA_speed)  -- 4 hours
def trainB_time_saved := 8 - (360 / trainB_speed)   -- 4 hours

-- Prove that total time saved by both trains running at their own speeds is 8 hours
theorem time_saved_by_both_trains : trainA_time_saved + trainB_time_saved = 8 := by
  sorry

-- Prove that the distance between Town X and Town Y for Train B is 360 miles
theorem trainB_distance : 360 = 360 := by
  rfl

end NUMINAMATH_GPT_time_saved_by_both_trains_trainB_distance_l553_55358


namespace NUMINAMATH_GPT_sum_even_odd_functions_l553_55382

theorem sum_even_odd_functions (f g : ℝ → ℝ) (h₁ : ∀ x, f (-x) = f x) (h₂ : ∀ x, g (-x) = -g x) (h₃ : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  f 1 + g 1 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_sum_even_odd_functions_l553_55382


namespace NUMINAMATH_GPT_sin_cos_75_eq_quarter_l553_55365

theorem sin_cos_75_eq_quarter : (Real.sin (75 * Real.pi / 180)) * (Real.cos (75 * Real.pi / 180)) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_75_eq_quarter_l553_55365


namespace NUMINAMATH_GPT_problem1_problem2_l553_55362

noncomputable def setA (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def setB : Set ℝ := {x | x^2 - 5 * x + 4 ≤ 0}

theorem problem1 (a : ℝ) (h : a = 1) : setA a ∪ setB = {x | 0 ≤ x ∧ x ≤ 4} := by
  sorry

theorem problem2 (a : ℝ) : (∀ x, x ∈ setA a → x ∈ setB) ↔ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l553_55362


namespace NUMINAMATH_GPT_root_in_interval_l553_55388

noncomputable def f (x : ℝ) := x^2 + 12 * x - 15

theorem root_in_interval :
  (f 1.1 = -0.59) → (f 1.2 = 0.84) →
  ∃ c, 1.1 < c ∧ c < 1.2 ∧ f c = 0 :=
by
  intros h1 h2
  let h3 := h1
  let h4 := h2
  sorry

end NUMINAMATH_GPT_root_in_interval_l553_55388


namespace NUMINAMATH_GPT_find_n_for_divisibility_by_33_l553_55397

theorem find_n_for_divisibility_by_33 (n : ℕ) (hn_range : n < 10) (div11 : (12 - n) % 11 = 0) (div3 : (20 + n) % 3 = 0) : n = 1 :=
by {
  -- Proof steps go here
  sorry
}

end NUMINAMATH_GPT_find_n_for_divisibility_by_33_l553_55397


namespace NUMINAMATH_GPT_widow_share_l553_55314

theorem widow_share (w d s : ℝ) (h_sum : w + 5 * s + 4 * d = 8000)
  (h1 : d = 2 * w)
  (h2 : s = 3 * d) :
  w = 8000 / 39 := by
sorry

end NUMINAMATH_GPT_widow_share_l553_55314


namespace NUMINAMATH_GPT_minimum_value_l553_55312

theorem minimum_value {a b c : ℝ} (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * b * c = 1 / 2) :
  ∃ x, x = a^2 + 4 * a * b + 9 * b^2 + 8 * b * c + 3 * c^2 ∧ x = 13.5 :=
sorry

end NUMINAMATH_GPT_minimum_value_l553_55312


namespace NUMINAMATH_GPT_deepak_present_age_l553_55366

theorem deepak_present_age (x : ℕ) (Rahul_age Deepak_age : ℕ) 
  (h1 : Rahul_age = 4 * x) (h2 : Deepak_age = 3 * x) 
  (h3 : Rahul_age + 4 = 32) : Deepak_age = 21 := by
  sorry

end NUMINAMATH_GPT_deepak_present_age_l553_55366


namespace NUMINAMATH_GPT_correct_proposition_four_l553_55310

universe u

-- Definitions
variable {Point : Type u} (A B : Point) (a α : Set Point)
variable (h5 : A ∉ α)
variable (h6 : a ⊂ α)

-- The statement to be proved
theorem correct_proposition_four : A ∉ a :=
sorry

end NUMINAMATH_GPT_correct_proposition_four_l553_55310


namespace NUMINAMATH_GPT_inequality_abc_l553_55308

theorem inequality_abc (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a * b * c = 8) :
  (a^2 / Real.sqrt ((1 + a^3) * (1 + b^3))) + (b^2 / Real.sqrt ((1 + b^3) * (1 + c^3))) +
  (c^2 / Real.sqrt ((1 + c^3) * (1 + a^3))) ≥ 4 / 3 :=
sorry

end NUMINAMATH_GPT_inequality_abc_l553_55308


namespace NUMINAMATH_GPT_students_in_band_l553_55394

theorem students_in_band (total_students : ℕ) (band_percentage : ℚ) (h_total_students : total_students = 840) (h_band_percentage : band_percentage = 0.2) : ∃ band_students : ℕ, band_students = 168 ∧ band_students = band_percentage * total_students := 
sorry

end NUMINAMATH_GPT_students_in_band_l553_55394


namespace NUMINAMATH_GPT_parabola_symmetry_l553_55344

theorem parabola_symmetry (a h m : ℝ) (A_on_parabola : 4 = a * (-1 - 3)^2 + h) (B_on_parabola : 4 = a * (m - 3)^2 + h) : 
  m = 7 :=
by 
  sorry

end NUMINAMATH_GPT_parabola_symmetry_l553_55344


namespace NUMINAMATH_GPT_sparrow_grains_l553_55315

theorem sparrow_grains (x : ℤ) : 9 * x < 1001 ∧ 10 * x > 1100 → x = 111 :=
by
  sorry

end NUMINAMATH_GPT_sparrow_grains_l553_55315


namespace NUMINAMATH_GPT_sin_double_angle_l553_55374

-- Given Conditions
variable {α : ℝ}
variable (h1 : 0 < α ∧ α < π / 2) -- α is in the first quadrant
variable (h2 : Real.sin α = 3 / 5) -- sin(α) = 3/5

-- Theorem statement
theorem sin_double_angle (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin α = 3 / 5) : 
  Real.sin (2 * α) = 24 / 25 := 
sorry

end NUMINAMATH_GPT_sin_double_angle_l553_55374


namespace NUMINAMATH_GPT_fourth_vertex_of_square_l553_55320

def A : ℂ := 2 - 3 * Complex.I
def B : ℂ := 3 + 2 * Complex.I
def C : ℂ := -3 + 2 * Complex.I

theorem fourth_vertex_of_square : ∃ D : ℂ, 
  (D - B) = (B - A) * Complex.I ∧ 
  (D - C) = (C - A) * Complex.I ∧ 
  (D = -3 + 8 * Complex.I) :=
sorry

end NUMINAMATH_GPT_fourth_vertex_of_square_l553_55320


namespace NUMINAMATH_GPT_square_plot_area_l553_55353

theorem square_plot_area (price_per_foot : ℝ) (total_cost : ℝ) (s : ℝ) (A : ℝ)
  (h1 : price_per_foot = 58)
  (h2 : total_cost = 1160)
  (h3 : total_cost = 4 * s * price_per_foot)
  (h4 : A = s * s) :
  A = 25 := by
  sorry

end NUMINAMATH_GPT_square_plot_area_l553_55353


namespace NUMINAMATH_GPT_dodecahedron_edge_probability_l553_55390

def numVertices := 20
def pairsChosen := Nat.choose 20 2  -- Calculates combination (20 choose 2)
def edgesPerVertex := 3
def numEdges := (numVertices * edgesPerVertex) / 2
def probability : ℚ := numEdges / pairsChosen

theorem dodecahedron_edge_probability :
  probability = 3 / 19 :=
by
  -- The proof is skipped as per the instructions
  sorry

end NUMINAMATH_GPT_dodecahedron_edge_probability_l553_55390


namespace NUMINAMATH_GPT_max_vertices_of_divided_triangle_l553_55357

theorem max_vertices_of_divided_triangle (n : ℕ) (h : n ≥ 1) : 
  (∀ t : ℕ, t = 1000 → exists T : ℕ, T = (n + 2)) :=
by sorry

end NUMINAMATH_GPT_max_vertices_of_divided_triangle_l553_55357


namespace NUMINAMATH_GPT_problem_statement_l553_55370

theorem problem_statement (n : ℕ) (a b c : ℕ → ℤ)
  (h1 : n > 0)
  (h2 : ∀ i j, i ≠ j → ¬ (a i - a j) % n = 0 ∧
                           ¬ ((b i + c i) - (b j + c j)) % n = 0 ∧
                           ¬ (b i - b j) % n = 0 ∧
                           ¬ ((c i + a i) - (c j + a i)) % n = 0 ∧
                           ¬ (c i - c j) % n = 0 ∧
                           ¬ ((a i + b i) - (a j + b i)) % n = 0 ∧
                           ¬ ((a i + b i + c i) - (a j + b i + c j)) % n = 0) :
  (Odd n) ∧ (¬ ∃ k, n = 3 * k) :=
by sorry

end NUMINAMATH_GPT_problem_statement_l553_55370


namespace NUMINAMATH_GPT_simplified_value_l553_55392

-- Define the operation ∗
def operation (m n p q : ℚ) : ℚ :=
  m * p * (n / q)

-- Prove that the simplified value of 5/4 ∗ 6/2 is 60
theorem simplified_value : operation 5 4 6 2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_simplified_value_l553_55392


namespace NUMINAMATH_GPT_cheryl_material_leftover_l553_55375

theorem cheryl_material_leftover :
  let material1 := (5 / 9 : ℚ)
  let material2 := (1 / 3 : ℚ)
  let total_bought := material1 + material2
  let used := (0.5555555555555556 : ℝ)
  let total_bought_decimal := (8 / 9 : ℝ)
  let leftover := total_bought_decimal - used
  leftover = 0.3333333333333332 := by
sorry

end NUMINAMATH_GPT_cheryl_material_leftover_l553_55375


namespace NUMINAMATH_GPT_incorrect_statement_among_ABCD_l553_55349

theorem incorrect_statement_among_ABCD :
  ¬ (-3 = Real.sqrt ((-3)^2)) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_statement_among_ABCD_l553_55349


namespace NUMINAMATH_GPT_sample_variance_l553_55318

theorem sample_variance (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) :
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_sample_variance_l553_55318


namespace NUMINAMATH_GPT_factor_count_x9_minus_x_l553_55313

theorem factor_count_x9_minus_x :
  ∃ (factors : List (Polynomial ℤ)), x^9 - x = factors.prod ∧ factors.length = 5 :=
sorry

end NUMINAMATH_GPT_factor_count_x9_minus_x_l553_55313


namespace NUMINAMATH_GPT_cos_20_cos_10_minus_sin_160_sin_10_l553_55335

theorem cos_20_cos_10_minus_sin_160_sin_10 : 
  (Real.cos (20 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) - 
   Real.sin (160 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) = 
   Real.cos (30 * Real.pi / 180) :=
by
  sorry

end NUMINAMATH_GPT_cos_20_cos_10_minus_sin_160_sin_10_l553_55335


namespace NUMINAMATH_GPT_xyz_value_l553_55327

variables {x y z : ℂ}

theorem xyz_value (h1 : x * y + 2 * y = -8)
                  (h2 : y * z + 2 * z = -8)
                  (h3 : z * x + 2 * x = -8) :
  x * y * z = 32 :=
by
  sorry

end NUMINAMATH_GPT_xyz_value_l553_55327


namespace NUMINAMATH_GPT_fraction_of_house_painted_l553_55368

theorem fraction_of_house_painted (total_time : ℝ) (paint_time : ℝ) (house : ℝ) (h1 : total_time = 60) (h2 : paint_time = 15) (h3 : house = 1) : 
  (paint_time / total_time) * house = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_house_painted_l553_55368


namespace NUMINAMATH_GPT_cards_probability_comparison_l553_55323

noncomputable def probability_case_a : ℚ :=
  (Nat.choose 13 10) * (Nat.choose 39 3) / Nat.choose 52 13

noncomputable def probability_case_b : ℚ :=
  4 ^ 13 / Nat.choose 52 13

theorem cards_probability_comparison :
  probability_case_b > probability_case_a :=
  sorry

end NUMINAMATH_GPT_cards_probability_comparison_l553_55323


namespace NUMINAMATH_GPT_initial_balloons_blown_up_l553_55377
-- Import necessary libraries

-- Define the statement
theorem initial_balloons_blown_up (x : ℕ) (hx : x + 13 = 60) : x = 47 :=
by
  sorry

end NUMINAMATH_GPT_initial_balloons_blown_up_l553_55377


namespace NUMINAMATH_GPT_photos_per_week_in_february_l553_55391

def january_photos : ℕ := 31 * 2

def total_photos (jan_feb_photos : ℕ) : ℕ := jan_feb_photos - january_photos

theorem photos_per_week_in_february (jan_feb_photos : ℕ) (weeks_in_february : ℕ)
  (h1 : jan_feb_photos = 146)
  (h2 : weeks_in_february = 4) :
  total_photos jan_feb_photos / weeks_in_february = 21 := by
  sorry

end NUMINAMATH_GPT_photos_per_week_in_february_l553_55391


namespace NUMINAMATH_GPT_boxes_needed_to_sell_l553_55376

theorem boxes_needed_to_sell (total_bars : ℕ) (bars_per_box : ℕ) (target_boxes : ℕ) (h₁ : total_bars = 710) (h₂ : bars_per_box = 5) : target_boxes = 142 :=
by
  sorry

end NUMINAMATH_GPT_boxes_needed_to_sell_l553_55376


namespace NUMINAMATH_GPT_exists_k_l553_55348

theorem exists_k (m n : ℕ) : ∃ k : ℕ, (Real.sqrt m + Real.sqrt (m - 1)) ^ n = Real.sqrt k + Real.sqrt (k - 1) := by
  sorry

end NUMINAMATH_GPT_exists_k_l553_55348


namespace NUMINAMATH_GPT_jade_transactions_l553_55360

theorem jade_transactions 
    (mabel_transactions : ℕ)
    (anthony_transactions : ℕ)
    (cal_transactions : ℕ)
    (jade_transactions : ℕ)
    (h_mabel : mabel_transactions = 90)
    (h_anthony : anthony_transactions = mabel_transactions + mabel_transactions / 10)
    (h_cal : cal_transactions = 2 * anthony_transactions / 3)
    (h_jade : jade_transactions = cal_transactions + 14) : 
    jade_transactions = 80 :=
sorry

end NUMINAMATH_GPT_jade_transactions_l553_55360


namespace NUMINAMATH_GPT_fractions_sum_identity_l553_55367

theorem fractions_sum_identity (a b c : ℝ) (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / ((b - c) ^ 2) + b / ((c - a) ^ 2) + c / ((a - b) ^ 2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_fractions_sum_identity_l553_55367


namespace NUMINAMATH_GPT_petya_time_l553_55389

variable (a V : ℝ)

noncomputable def planned_time := a / V
noncomputable def real_time := (a / (2.5 * V)) + (a / (1.6 * V))

theorem petya_time (hV : V > 0) (ha : a > 0) : real_time a V > planned_time a V :=
by
  sorry

end NUMINAMATH_GPT_petya_time_l553_55389


namespace NUMINAMATH_GPT_children_attended_l553_55332

theorem children_attended (A C : ℕ) (h1 : C = 2 * A) (h2 : A + C = 42) : C = 28 :=
by
  sorry

end NUMINAMATH_GPT_children_attended_l553_55332


namespace NUMINAMATH_GPT_mowing_time_approximately_correct_l553_55379

noncomputable def timeToMowLawn 
  (length width : ℝ) -- dimensions of the lawn in feet
  (swath overlap : ℝ) -- swath width and overlap in inches
  (speed : ℝ) : ℝ :=  -- walking speed in feet per hour
  (length * (width / ((swath - overlap) / 12))) / speed

theorem mowing_time_approximately_correct
  (h_length : ∀ (length : ℝ), length = 100)
  (h_width : ∀ (width : ℝ), width = 120)
  (h_swath : ∀ (swath : ℝ), swath = 30)
  (h_overlap : ∀ (overlap : ℝ), overlap = 6)
  (h_speed : ∀ (speed : ℝ), speed = 4500) :
  abs (timeToMowLawn 100 120 30 6 4500 - 1.33) < 0.01 := -- assert the answer is approximately 1.33 with a tolerance
by
  intros
  have length := h_length 100
  have width := h_width 120
  have swath := h_swath 30
  have overlap := h_overlap 6
  have speed := h_speed 4500
  rw [length, width, swath, overlap, speed]
  simp [timeToMowLawn]
  sorry

end NUMINAMATH_GPT_mowing_time_approximately_correct_l553_55379


namespace NUMINAMATH_GPT_sin_segment_ratio_is_rel_prime_l553_55319

noncomputable def sin_segment_ratio : ℕ × ℕ :=
  let p := 1
  let q := 8
  (p, q)
  
theorem sin_segment_ratio_is_rel_prime :
  1 < 8 ∧ gcd 1 8 = 1 ∧ sin_segment_ratio = (1, 8) :=
by
  -- gcd 1 8 = 1
  have h1 : gcd 1 8 = 1 := by exact gcd_one_right 8
  -- 1 < 8
  have h2 : 1 < 8 := by decide
  -- final tuple
  have h3 : sin_segment_ratio = (1, 8) := by rfl
  exact ⟨h2, h1, h3⟩

end NUMINAMATH_GPT_sin_segment_ratio_is_rel_prime_l553_55319


namespace NUMINAMATH_GPT_power_log_simplification_l553_55387

theorem power_log_simplification (x : ℝ) (h : x > 0) : (16^(Real.log x / Real.log 2))^(1/4) = x :=
by sorry

end NUMINAMATH_GPT_power_log_simplification_l553_55387


namespace NUMINAMATH_GPT_analysis_method_correct_answer_l553_55395

axiom analysis_def (conclusion: Prop): 
  ∃ sufficient_conditions: (Prop → Prop), 
    (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)

theorem analysis_method_correct_answer :
  ∀ (conclusion : Prop) , ∃ sufficient_conditions: (Prop → Prop), 
  (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)
:= by 
  intros 
  sorry

end NUMINAMATH_GPT_analysis_method_correct_answer_l553_55395


namespace NUMINAMATH_GPT_age_difference_l553_55399

def JobAge := 5
def StephanieAge := 4 * JobAge
def FreddyAge := 18

theorem age_difference : StephanieAge - FreddyAge = 2 := by
  sorry

end NUMINAMATH_GPT_age_difference_l553_55399


namespace NUMINAMATH_GPT_smallest_n_for_multiple_of_5_l553_55373

theorem smallest_n_for_multiple_of_5 (x y : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 5]) (h2 : y - 2 ≡ 0 [ZMOD 5]) :
  ∃ n : ℕ, n > 0 ∧ x^2 + x * y + y^2 + n ≡ 0 [ZMOD 5] ∧ n = 1 := 
sorry

end NUMINAMATH_GPT_smallest_n_for_multiple_of_5_l553_55373


namespace NUMINAMATH_GPT_rabbit_excursion_time_l553_55363

theorem rabbit_excursion_time 
  (line_length : ℝ := 40) 
  (line_speed : ℝ := 3) 
  (rabbit_speed : ℝ := 5) : 
  -- The time calculated for the rabbit to return is 25 seconds
  (line_length / (rabbit_speed - line_speed) + line_length / (rabbit_speed + line_speed)) = 25 :=
by
  -- Placeholder for the proof, to be filled in with a detailed proof later
  sorry

end NUMINAMATH_GPT_rabbit_excursion_time_l553_55363


namespace NUMINAMATH_GPT_find_m_l553_55369

theorem find_m (f : ℝ → ℝ) (m : ℝ) 
  (h_even : ∀ x, f (-x) = f x) 
  (h_fx : ∀ x, 0 < x → f x = 4^(m - x)) 
  (h_f_neg2 : f (-2) = 1/8) : 
  m = 1/2 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_l553_55369


namespace NUMINAMATH_GPT_intersection_A_B_l553_55333

def A : Set ℝ := { x | x ≤ 1 }
def B : Set ℝ := {0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l553_55333


namespace NUMINAMATH_GPT_carol_mike_equal_savings_weeks_l553_55336

theorem carol_mike_equal_savings_weeks :
  ∃ x : ℕ, (60 + 9 * x = 90 + 3 * x) ↔ x = 5 := 
by
  sorry

end NUMINAMATH_GPT_carol_mike_equal_savings_weeks_l553_55336


namespace NUMINAMATH_GPT_trains_at_start_2016_l553_55329

def traversal_time_red := 7
def traversal_time_blue := 8
def traversal_time_green := 9

def return_period_red := 2 * traversal_time_red
def return_period_blue := 2 * traversal_time_blue
def return_period_green := 2 * traversal_time_green

def train_start_pos_time := 2016
noncomputable def lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)

theorem trains_at_start_2016 :
  train_start_pos_time % lcm_period = 0 :=
by
  have return_period_red := 2 * traversal_time_red
  have return_period_blue := 2 * traversal_time_blue
  have return_period_green := 2 * traversal_time_green
  have lcm_period := Nat.lcm return_period_red (Nat.lcm return_period_blue return_period_green)
  have train_start_pos_time := 2016
  exact sorry

end NUMINAMATH_GPT_trains_at_start_2016_l553_55329


namespace NUMINAMATH_GPT_average_time_per_stop_l553_55356

-- Definitions from the conditions
def pizzas : Nat := 12
def stops_with_two_pizzas : Nat := 2
def total_delivery_time : Nat := 40

-- Using the conditions to define what needs to be proved
theorem average_time_per_stop : 
  let single_pizza_stops := pizzas - stops_with_two_pizzas * 2
  let total_stops := single_pizza_stops + stops_with_two_pizzas
  let average_time := total_delivery_time / total_stops
  average_time = 4 := by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_average_time_per_stop_l553_55356


namespace NUMINAMATH_GPT_ramu_profit_percent_l553_55338

def ramu_bought_car : ℝ := 48000
def ramu_repair_cost : ℝ := 14000
def ramu_selling_price : ℝ := 72900

theorem ramu_profit_percent :
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  profit_percent = 17.58 := 
by
  -- Definitions and setting up the proof environment
  let total_cost := ramu_bought_car + ramu_repair_cost
  let profit := ramu_selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  sorry

end NUMINAMATH_GPT_ramu_profit_percent_l553_55338


namespace NUMINAMATH_GPT_find_f_a_plus_1_l553_55343

def f (x : ℝ) : ℝ := x^2 + 1

theorem find_f_a_plus_1 (a : ℝ) : f (a + 1) = a^2 + 2 * a + 2 := by
  sorry

end NUMINAMATH_GPT_find_f_a_plus_1_l553_55343


namespace NUMINAMATH_GPT_problem_statement_l553_55306

-- Define a set S
variable {S : Type*}

-- Define the binary operation on S
variable (mul : S → S → S)

-- Assume the given condition: (a * b) * a = b for all a, b in S
axiom given_condition : ∀ (a b : S), (mul (mul a b) a) = b

-- Prove that a * (b * a) = b for all a, b in S
theorem problem_statement : ∀ (a b : S), mul a (mul b a) = b :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l553_55306


namespace NUMINAMATH_GPT_n_fraction_sum_l553_55316

theorem n_fraction_sum {n : ℝ} {lst : List ℝ} (h_len : lst.length = 21) 
(h_mem : n ∈ lst) 
(h_avg : n = 4 * (lst.erase n).sum / 20) :
  n = (lst.sum) / 6 :=
by
  sorry

end NUMINAMATH_GPT_n_fraction_sum_l553_55316


namespace NUMINAMATH_GPT_determine_squirrel_color_l553_55355

-- Define the types for Squirrel species and the nuts in hollows
inductive Squirrel
| red
| gray

def tells_truth (s : Squirrel) : Prop :=
  s = Squirrel.red

def lies (s : Squirrel) : Prop :=
  s = Squirrel.gray

-- Statements made by the squirrel in front of the second hollow
def statement1 (s : Squirrel) (no_nuts_in_first : Prop) : Prop :=
  tells_truth s → no_nuts_in_first ∧ (lies s → ¬no_nuts_in_first)

def statement2 (s : Squirrel) (nuts_in_either : Prop) : Prop :=
  tells_truth s → nuts_in_either ∧ (lies s → ¬nuts_in_either)

-- Given a squirrel that says the statements and the information about truth and lies
theorem determine_squirrel_color (s : Squirrel) (no_nuts_in_first : Prop) (nuts_in_either : Prop) :
  (statement1 s no_nuts_in_first) ∧ (statement2 s nuts_in_either) → s = Squirrel.red :=
by
  sorry

end NUMINAMATH_GPT_determine_squirrel_color_l553_55355


namespace NUMINAMATH_GPT_opposite_of_num_l553_55341

-- Define the number whose opposite we are calculating
def num := -1 / 2

-- Theorem statement that the opposite of num is 1/2
theorem opposite_of_num : -num = 1 / 2 := by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_opposite_of_num_l553_55341


namespace NUMINAMATH_GPT_first_problem_l553_55393

-- Definitions for the first problem
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable (h_pos : ∀ n, a n > 0)
variable (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1))

-- Theorem statement for the first problem
theorem first_problem (h_pos : ∀ n, a n > 0) (h_seq : ∀ n, (a n + 1)^2 = 4 * (S n + 1)) :
  ∃ d, ∀ n, a (n + 1) - a n = d := sorry

end NUMINAMATH_GPT_first_problem_l553_55393


namespace NUMINAMATH_GPT_molecular_weight_N2O5_l553_55350

variable {x : ℕ}

theorem molecular_weight_N2O5 (hx : 10 * 108 = 1080) : (108 * x = 1080 * x / 10) :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_N2O5_l553_55350


namespace NUMINAMATH_GPT_last_term_of_sequence_l553_55340

theorem last_term_of_sequence (u₀ : ℤ) (diffs : List ℤ) (sum_diffs : ℤ) :
  u₀ = 0 → diffs = [2, 4, -1, 0, -5, -3, 3] → sum_diffs = diffs.sum → 
  u₀ + sum_diffs = 0 := by
  sorry

end NUMINAMATH_GPT_last_term_of_sequence_l553_55340


namespace NUMINAMATH_GPT_sum_leq_two_l553_55321

open Classical

theorem sum_leq_two (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^3 + b^3 = 2) : a + b ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_leq_two_l553_55321


namespace NUMINAMATH_GPT_volume_of_cone_l553_55378

theorem volume_of_cone (l : ℝ) (A : ℝ) (r : ℝ) (h : ℝ) : 
  l = 10 → A = 60 * Real.pi → (r = 6) → (h = Real.sqrt (10^2 - 6^2)) → 
  (1 / 3 * Real.pi * r^2 * h) = 96 * Real.pi :=
by
  intros
  -- here the proof would be written
  sorry

end NUMINAMATH_GPT_volume_of_cone_l553_55378
