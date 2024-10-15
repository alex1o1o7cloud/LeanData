import Mathlib

namespace NUMINAMATH_GPT_scalene_triangle_angle_difference_l674_67435

def scalene_triangle (a b c : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem scalene_triangle_angle_difference (x y : ℝ) :
  (x + y = 100) → scalene_triangle x y 80 → (x - y = 80) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_scalene_triangle_angle_difference_l674_67435


namespace NUMINAMATH_GPT_axel_vowels_written_l674_67425

theorem axel_vowels_written (total_alphabets number_of_vowels n : ℕ) (h1 : total_alphabets = 10) (h2 : number_of_vowels = 5) (h3 : total_alphabets = number_of_vowels * n) : n = 2 :=
by
  sorry

end NUMINAMATH_GPT_axel_vowels_written_l674_67425


namespace NUMINAMATH_GPT_solve_equation_l674_67415

theorem solve_equation :
  ∃ x : ℝ, x = (Real.sqrt (x - 1/x)) + (Real.sqrt (1 - 1/x)) ∧ x = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l674_67415


namespace NUMINAMATH_GPT_domain_of_f_comp_l674_67419

theorem domain_of_f_comp (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → -2 ≤ x^2 - 2 ∧ x^2 - 2 ≤ -1) →
  (∀ x, - (4 : ℝ) / 3 ≤ x ∧ x ≤ -1 → -2 ≤ 3 * x + 2 ∧ 3 * x + 2 ≤ -1) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_comp_l674_67419


namespace NUMINAMATH_GPT_log_base_3_domain_is_minus_infinity_to_3_l674_67487

noncomputable def log_base_3_domain (x : ℝ) : Prop :=
  3 - x > 0

theorem log_base_3_domain_is_minus_infinity_to_3 :
  ∀ x : ℝ, log_base_3_domain x ↔ x < 3 :=
by
  sorry

end NUMINAMATH_GPT_log_base_3_domain_is_minus_infinity_to_3_l674_67487


namespace NUMINAMATH_GPT_minimum_choir_size_l674_67432

theorem minimum_choir_size : ∃ (choir_size : ℕ), 
  (choir_size % 9 = 0) ∧ 
  (choir_size % 11 = 0) ∧ 
  (choir_size % 13 = 0) ∧ 
  (choir_size % 10 = 0) ∧ 
  (choir_size = 12870) :=
by
  sorry

end NUMINAMATH_GPT_minimum_choir_size_l674_67432


namespace NUMINAMATH_GPT_prime_sum_of_primes_l674_67479

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def distinct_primes (p q r s : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime s ∧ p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s

theorem prime_sum_of_primes (p q r s : ℕ) :
  distinct_primes p q r s →
  is_prime (p + q + r + s) →
  is_square (p^2 + q * s) →
  is_square (p^2 + q * r) →
  (p = 2 ∧ q = 7 ∧ r = 11 ∧ s = 3) ∨ (p = 2 ∧ q = 7 ∧ r = 3 ∧ s = 11) :=
by
  sorry

end NUMINAMATH_GPT_prime_sum_of_primes_l674_67479


namespace NUMINAMATH_GPT_find_a_l674_67438

theorem find_a (x a : ℝ) : 
  (a + 2 = 0) ↔ (a = -2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l674_67438


namespace NUMINAMATH_GPT_value_of_a_l674_67403

theorem value_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + x + a^2 - 1 = 0 → x = 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l674_67403


namespace NUMINAMATH_GPT_prime_square_minus_one_divisible_by_24_l674_67414

theorem prime_square_minus_one_divisible_by_24 (n : ℕ) (h_prime : Prime n) (h_n_neq_2 : n ≠ 2) (h_n_neq_3 : n ≠ 3) : 24 ∣ (n^2 - 1) :=
sorry

end NUMINAMATH_GPT_prime_square_minus_one_divisible_by_24_l674_67414


namespace NUMINAMATH_GPT_study_time_for_average_l674_67481

theorem study_time_for_average
    (study_time_exam1 score_exam1 : ℕ)
    (study_time_exam2 score_exam2 average_score desired_average : ℝ)
    (relation : score_exam1 = 20 * study_time_exam1)
    (direct_relation : score_exam2 = 20 * study_time_exam2)
    (total_exams : ℕ)
    (average_condition : (score_exam1 + score_exam2) / total_exams = desired_average) :
    study_time_exam2 = 4.5 :=
by
  have : total_exams = 2 := by sorry
  have : score_exam1 = 60 := by sorry
  have : desired_average = 75 := by sorry
  have : score_exam2 = 90 := by sorry
  sorry

end NUMINAMATH_GPT_study_time_for_average_l674_67481


namespace NUMINAMATH_GPT_fish_to_apples_l674_67466

variables (f l r a : ℝ)

theorem fish_to_apples (h1 : 3 * f = 2 * l) (h2 : l = 5 * r) (h3 : l = 3 * a) : f = 2 * a :=
by
  -- We assume the conditions as hypotheses and aim to prove the final statement
  sorry

end NUMINAMATH_GPT_fish_to_apples_l674_67466


namespace NUMINAMATH_GPT_problem_statement_l674_67436

theorem problem_statement (pi : ℝ) (h : pi = 4 * Real.sin (52 * Real.pi / 180)) :
  (2 * pi * Real.sqrt (16 - pi ^ 2) - 8 * Real.sin (44 * Real.pi / 180)) /
  (Real.sqrt 3 - 2 * Real.sqrt 3 * (Real.sin (22 * Real.pi / 180)) ^ 2) = 8 * Real.sqrt 3 := 
  sorry

end NUMINAMATH_GPT_problem_statement_l674_67436


namespace NUMINAMATH_GPT_maximum_value_of_expression_l674_67413

-- Define the given condition
def condition (a b c : ℝ) : Prop := a + 3 * b + c = 5

-- Define the objective function
def objective (a b c : ℝ) : ℝ := a * b + a * c + b * c

-- Main theorem statement
theorem maximum_value_of_expression (a b c : ℝ) (h : condition a b c) : 
  ∃ (a b c : ℝ), condition a b c ∧ objective a b c = 25 / 3 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_expression_l674_67413


namespace NUMINAMATH_GPT_initial_bottle_caps_correct_l674_67437

-- Defining the variables based on the conditions
def bottle_caps_found : ℕ := 7
def total_bottle_caps_now : ℕ := 32
def initial_bottle_caps : ℕ := 25

-- Statement of the theorem
theorem initial_bottle_caps_correct:
  total_bottle_caps_now - bottle_caps_found = initial_bottle_caps :=
sorry

end NUMINAMATH_GPT_initial_bottle_caps_correct_l674_67437


namespace NUMINAMATH_GPT_problem_statement_l674_67400

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def has_minimum_value_at (f : ℝ → ℝ) (a : ℝ) := ∀ x : ℝ, f a ≤ f x
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem problem_statement : is_even_function f4 ∧ has_minimum_value_at f4 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l674_67400


namespace NUMINAMATH_GPT_remainder_of_5032_div_28_l674_67428

theorem remainder_of_5032_div_28 : 5032 % 28 = 20 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_5032_div_28_l674_67428


namespace NUMINAMATH_GPT_joyce_initial_eggs_l674_67453

theorem joyce_initial_eggs :
  ∃ E : ℕ, (E + 6 = 14) ∧ E = 8 :=
sorry

end NUMINAMATH_GPT_joyce_initial_eggs_l674_67453


namespace NUMINAMATH_GPT_largest_equal_cost_l674_67411

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_digit_sum (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem largest_equal_cost :
  ∃ (n : ℕ), n < 500 ∧ digit_sum n = binary_digit_sum n ∧ ∀ m < 500, digit_sum m = binary_digit_sum m → m ≤ 247 :=
by
  sorry

end NUMINAMATH_GPT_largest_equal_cost_l674_67411


namespace NUMINAMATH_GPT_square_root_condition_l674_67460

theorem square_root_condition (x : ℝ) : (6 + x ≥ 0) ↔ (x ≥ -6) :=
by sorry

end NUMINAMATH_GPT_square_root_condition_l674_67460


namespace NUMINAMATH_GPT_initial_cell_count_l674_67470

theorem initial_cell_count (f : ℕ → ℕ) (h₁ : ∀ n, f (n + 1) = 2 * (f n - 2)) (h₂ : f 5 = 164) : f 0 = 9 :=
sorry

end NUMINAMATH_GPT_initial_cell_count_l674_67470


namespace NUMINAMATH_GPT_episodes_per_season_l674_67402

theorem episodes_per_season (S : ℕ) (E : ℕ) (H1 : S = 12) (H2 : 2/3 * E = 160) : E / S = 20 :=
by
  sorry

end NUMINAMATH_GPT_episodes_per_season_l674_67402


namespace NUMINAMATH_GPT_impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l674_67446

theorem impossible_to_get_60_pieces :
  ¬ ∃ (n m : ℕ), 1 + 7 * n + 11 * m = 60 :=
sorry

theorem possible_to_get_more_than_60_pieces :
  ∀ k > 60, ∃ (n m : ℕ), 1 + 7 * n + 11 * m = k :=
sorry

end NUMINAMATH_GPT_impossible_to_get_60_pieces_possible_to_get_more_than_60_pieces_l674_67446


namespace NUMINAMATH_GPT_copier_cost_l674_67422

noncomputable def total_time : ℝ := 4 + 25 / 60
noncomputable def first_quarter_hour_cost : ℝ := 6
noncomputable def hourly_cost : ℝ := 8
noncomputable def time_after_first_quarter_hour : ℝ := total_time - 0.25
noncomputable def remaining_cost : ℝ := time_after_first_quarter_hour * hourly_cost
noncomputable def total_cost : ℝ := first_quarter_hour_cost + remaining_cost

theorem copier_cost :
  total_cost = 39.33 :=
by
  -- This statement remains to be proved.
  sorry

end NUMINAMATH_GPT_copier_cost_l674_67422


namespace NUMINAMATH_GPT_percentage_loss_l674_67496

variable (CP SP : ℝ)
variable (HCP : CP = 1600)
variable (HSP : SP = 1408)

theorem percentage_loss (HCP : CP = 1600) (HSP : SP = 1408) : 
  (CP - SP) / CP * 100 = 12 := by
sorry

end NUMINAMATH_GPT_percentage_loss_l674_67496


namespace NUMINAMATH_GPT_tan_ratio_l674_67465

-- Given conditions
variables {p q : ℝ} (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3)

-- The theorem we need to prove
theorem tan_ratio (h1 : Real.cos (p + q) = 1 / 3) (h2 : Real.cos (p - q) = 2 / 3) : 
  Real.tan p / Real.tan q = -1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_ratio_l674_67465


namespace NUMINAMATH_GPT_isosceles_triangle_l674_67489

-- Let ∆ABC be a triangle with angles A, B, and C
variables {A B C : ℝ}

-- Given condition: 2 * cos B * sin A = sin C
def condition (A B C : ℝ) : Prop := 2 * Real.cos B * Real.sin A = Real.sin C

-- Problem: Given the condition, we need to prove that ∆ABC is an isosceles triangle, meaning A = B.
theorem isosceles_triangle (A B C : ℝ) (h : condition A B C) : A = B :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_l674_67489


namespace NUMINAMATH_GPT_sewers_handle_rain_l674_67475

theorem sewers_handle_rain (total_capacity : ℕ) (runoff_per_hour : ℕ) : 
  total_capacity = 240000 → 
  runoff_per_hour = 1000 → 
  total_capacity / runoff_per_hour / 24 = 10 :=
by 
  intro h1 h2
  sorry

end NUMINAMATH_GPT_sewers_handle_rain_l674_67475


namespace NUMINAMATH_GPT_line_equation_M_l674_67404

theorem line_equation_M (x y : ℝ) : 
  (∃ c1 m1 : ℝ, m1 = 2 / 3 ∧ c1 = 4 ∧ 
  (∃ m2 c2 : ℝ, m2 = 2 * m1 ∧ c2 = (1 / 2) * c1 ∧ y = m2 * x + c2)) → 
  y = (4 / 3) * x + 2 := 
sorry

end NUMINAMATH_GPT_line_equation_M_l674_67404


namespace NUMINAMATH_GPT_regular_polygon_sides_l674_67410

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ n, 180 * (n - 2) = 150 * n) : n = 12 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l674_67410


namespace NUMINAMATH_GPT_max_distance_covered_l674_67409

theorem max_distance_covered 
  (D : ℝ)
  (h1 : (D / 2) / 5 + (D / 2) / 4 = 6) : 
  D = 40 / 3 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_covered_l674_67409


namespace NUMINAMATH_GPT_design_height_lower_part_l674_67464

theorem design_height_lower_part (H : ℝ) (H_eq : H = 2) (L : ℝ) 
  (ratio : (H - L) / L = L / H) : L = Real.sqrt 5 - 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_design_height_lower_part_l674_67464


namespace NUMINAMATH_GPT_matrix_multiplication_example_l674_67433

def matrix1 : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![-4, 5]]
def vector1 : Fin 2 → ℤ := ![4, -2]
def scalar : ℤ := 2
def result : Fin 2 → ℤ := ![32, -52]

theorem matrix_multiplication_example :
  scalar • (matrix1.mulVec vector1) = result := by
  sorry

end NUMINAMATH_GPT_matrix_multiplication_example_l674_67433


namespace NUMINAMATH_GPT_determine_ab_l674_67472

theorem determine_ab (a b : ℕ) (h1: a + b = 30) (h2: 2 * a * b + 14 * a = 5 * b + 290) : a * b = 104 := by
  -- the proof would be written here
  sorry

end NUMINAMATH_GPT_determine_ab_l674_67472


namespace NUMINAMATH_GPT_greatest_mass_l674_67420

theorem greatest_mass (V : ℝ) (h : ℝ) (l : ℝ) 
    (ρ_Hg ρ_H2O ρ_Oil : ℝ) 
    (V1 V2 V3 : ℝ) 
    (m_Hg m_H2O m_Oil : ℝ)
    (ρ_Hg_val : ρ_Hg = 13.59) 
    (ρ_H2O_val : ρ_H2O = 1) 
    (ρ_Oil_val : ρ_Oil = 0.915) 
    (height_layers_equal : h = l) :
    ∀ V1 V2 V3 m_Hg m_H2O m_Oil, 
    V1 + V2 + V3 = 27 * (l^3) → 
    V2 = 7 * V1 → 
    V3 = 19 * V1 → 
    m_Hg = ρ_Hg * V1 → 
    m_H2O = ρ_H2O * V2 → 
    m_Oil = ρ_Oil * V3 → 
    m_Oil > m_Hg ∧ m_Oil > m_H2O := 
by 
    intros
    sorry

end NUMINAMATH_GPT_greatest_mass_l674_67420


namespace NUMINAMATH_GPT_almond_butter_servings_l674_67439

def servings_of_almond_butter (tbsp_in_container : ℚ) (tbsp_per_serving : ℚ) : ℚ :=
  tbsp_in_container / tbsp_per_serving

def container_holds : ℚ := 37 + 2/3

def serving_size : ℚ := 3

theorem almond_butter_servings :
  servings_of_almond_butter container_holds serving_size = 12 + 5/9 := 
by
  sorry

end NUMINAMATH_GPT_almond_butter_servings_l674_67439


namespace NUMINAMATH_GPT_sugar_bought_l674_67461

noncomputable def P : ℝ := 0.50
noncomputable def S : ℝ := 2.0

theorem sugar_bought : 
  (1.50 * S + 5 * P = 5.50) ∧ 
  (3 * 1.50 + P = 5) ∧
  ((1.50 : ℝ) ≠ 0) → (S = 2) :=
by
  sorry

end NUMINAMATH_GPT_sugar_bought_l674_67461


namespace NUMINAMATH_GPT_triangle_area_l674_67492

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (7, 4)
def C : ℝ × ℝ := (7, -4)

-- Statement to prove the area of the triangle is 32 square units
theorem triangle_area :
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  (1 / 2 : ℝ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)| = 32 := by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_triangle_area_l674_67492


namespace NUMINAMATH_GPT_number_of_terms_is_13_l674_67442

-- Define sum of first three terms
def sum_first_three (a d : ℤ) : ℤ := a + (a + d) + (a + 2 * d)

-- Define sum of last three terms when the number of terms is n
def sum_last_three (a d : ℤ) (n : ℕ) : ℤ := (a + (n - 3) * d) + (a + (n - 2) * d) + (a + (n - 1) * d)

-- Define sum of all terms in the sequence
def sum_all_terms (a d : ℤ) (n : ℕ) : ℤ := n / 2 * (2 * a + (n - 1) * d)

-- Given conditions
def condition_one (a d : ℤ) : Prop := sum_first_three a d = 34
def condition_two (a d : ℤ) (n : ℕ) : Prop := sum_last_three a d n = 146
def condition_three (a d : ℤ) (n : ℕ) : Prop := sum_all_terms a d n = 390

-- Theorem to prove that n = 13
theorem number_of_terms_is_13 (a d : ℤ) (n : ℕ) :
  condition_one a d →
  condition_two a d n →
  condition_three a d n →
  n = 13 :=
by sorry

end NUMINAMATH_GPT_number_of_terms_is_13_l674_67442


namespace NUMINAMATH_GPT_inequality_proof_l674_67447

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
variable (hSum : a + b + c = 1)

theorem inequality_proof :
  (a^2 + b^2 + c^2) * (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l674_67447


namespace NUMINAMATH_GPT_train_speed_proof_l674_67416

noncomputable def speed_of_train (train_length : ℝ) (time_seconds : ℝ) (man_speed : ℝ) : ℝ :=
  let train_length_km := train_length / 1000
  let time_hours := time_seconds / 3600
  let relative_speed := train_length_km / time_hours
  relative_speed - man_speed

theorem train_speed_proof :
  speed_of_train 605 32.99736021118311 6 = 60.028 :=
by
  unfold speed_of_train
  -- Direct substitution and expected numerical simplification
  norm_num
  sorry

end NUMINAMATH_GPT_train_speed_proof_l674_67416


namespace NUMINAMATH_GPT_abs_inequality_solution_l674_67448

theorem abs_inequality_solution :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l674_67448


namespace NUMINAMATH_GPT_uma_income_is_20000_l674_67459

/-- Given that the ratio of the incomes of Uma and Bala is 4 : 3, 
the ratio of their expenditures is 3 : 2, and both save $5000 at the end of the year, 
prove that Uma's income is $20000. -/
def uma_bala_income : Prop :=
  ∃ (x y : ℕ), (4 * x - 3 * y = 5000) ∧ (3 * x - 2 * y = 5000) ∧ (4 * x = 20000)
  
theorem uma_income_is_20000 : uma_bala_income :=
  sorry

end NUMINAMATH_GPT_uma_income_is_20000_l674_67459


namespace NUMINAMATH_GPT_no_int_solutions_for_equation_l674_67421

theorem no_int_solutions_for_equation : 
  ∀ x y : ℤ, x ^ 2022 + y^2 = 2 * y + 2 → false := 
by
  -- By the given steps in the solution, we can conclude that no integer solutions exist
  sorry

end NUMINAMATH_GPT_no_int_solutions_for_equation_l674_67421


namespace NUMINAMATH_GPT_find_a_tangent_line_eq_l674_67441

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + x - 1) * Real.exp x

theorem find_a (a : ℝ) : f 1 (-3) = 0 → a = 1 := by
  sorry

theorem tangent_line_eq (x : ℝ) (e : ℝ) : x = 1 ∧ f 1 x = Real.exp 1 → 
    (4 * Real.exp 1 * x - y - 3 * Real.exp 1 = 0) := by
  sorry

end NUMINAMATH_GPT_find_a_tangent_line_eq_l674_67441


namespace NUMINAMATH_GPT_triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l674_67469

theorem triangle_side_ratio_impossible (a b c : ℝ) (h₁ : a = b / 2) (h₂ : a = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_2 (a b c : ℝ) (h₁ : b = a / 2) (h₂ : b = c / 3) : false :=
by
  sorry

theorem triangle_side_ratio_impossible_3 (a b c : ℝ) (h₁ : c = a / 2) (h₂ : c = b / 3) : false :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_ratio_impossible_triangle_side_ratio_impossible_2_triangle_side_ratio_impossible_3_l674_67469


namespace NUMINAMATH_GPT_no_fermat_in_sequence_l674_67452

def sequence_term (n k : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

def is_fermat_number (a : ℕ) : Prop :=
  ∃ m : ℕ, a = 2^(2^m) + 1

theorem no_fermat_in_sequence (k n : ℕ) (hk : k > 2) (hn : n > 2) :
  ¬ is_fermat_number (sequence_term n k) :=
sorry

end NUMINAMATH_GPT_no_fermat_in_sequence_l674_67452


namespace NUMINAMATH_GPT_line_equation_l674_67468

theorem line_equation (b r S : ℝ) (h : ℝ) (m : ℝ) (eq_one : S = 1/2 * b * h) (eq_two : h = 2*S / b) (eq_three : |m| = r / b) 
  (eq_four : m = r / b) : 
  (∀ x y : ℝ, y = m * (x - b) → b > 0 → r > 0 → S > 0 → rx - bry - rb = 0) := 
sorry

end NUMINAMATH_GPT_line_equation_l674_67468


namespace NUMINAMATH_GPT_smallest_integer_in_range_l674_67491

theorem smallest_integer_in_range :
  ∃ (n : ℕ), n > 1 ∧ n % 3 = 2 ∧ n % 7 = 2 ∧ n % 8 = 2 ∧ 131 ≤ n ∧ n ≤ 170 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_in_range_l674_67491


namespace NUMINAMATH_GPT_solve_proportion_l674_67427

noncomputable def x : ℝ := 0.6

theorem solve_proportion (x : ℝ) (h : 0.75 / x = 10 / 8) : x = 0.6 :=
by
  sorry

end NUMINAMATH_GPT_solve_proportion_l674_67427


namespace NUMINAMATH_GPT_batsman_average_after_17_l674_67444

variable (x : ℝ)
variable (total_runs_16 : ℝ := 16 * x)
variable (runs_17 : ℝ := 90)
variable (new_total_runs : ℝ := total_runs_16 + runs_17)
variable (new_average : ℝ := new_total_runs / 17)

theorem batsman_average_after_17 :
  (total_runs_16 + runs_17 = 17 * (x + 3)) → new_average = x + 3 → new_average = 42 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_batsman_average_after_17_l674_67444


namespace NUMINAMATH_GPT_alice_winning_strategy_l674_67494

theorem alice_winning_strategy (n : ℕ) (h : n ≥ 2) :
  (∃ strategy : Π (s : ℕ), s < n → (ℕ × ℕ), 
    ∀ (k : ℕ) (hk : k < n), ¬(strategy k hk).fst = (strategy k hk).snd) ↔ (n % 4 = 3) :=
sorry

end NUMINAMATH_GPT_alice_winning_strategy_l674_67494


namespace NUMINAMATH_GPT_unique_pair_fraction_l674_67471

theorem unique_pair_fraction (p : ℕ) (hprime : Nat.Prime p) (hodd : p % 2 = 1) :
  ∃! (n m : ℕ), (n ≠ m) ∧ (2 / (p : ℚ) = 1 / (n : ℚ) + 1 / (m : ℚ)) ∧
  ((n = (p + 1) / 2 ∧ m = p * (p + 1) / 2) ∨ (m = (p + 1) / 2 ∧ n = p * (p + 1) / 2)) := sorry

end NUMINAMATH_GPT_unique_pair_fraction_l674_67471


namespace NUMINAMATH_GPT_no_hikers_in_morning_l674_67401

-- Given Conditions
def morning_rowers : ℕ := 13
def afternoon_rowers : ℕ := 21
def total_rowers : ℕ := 34

-- Statement to be proven
theorem no_hikers_in_morning : (total_rowers - afternoon_rowers = morning_rowers) →
                              (total_rowers - afternoon_rowers = morning_rowers) →
                              0 = 34 - 21 - morning_rowers :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_no_hikers_in_morning_l674_67401


namespace NUMINAMATH_GPT_parabola_vertex_sum_l674_67474

theorem parabola_vertex_sum (p q r : ℝ)
  (h1 : ∃ a : ℝ, ∀ x y : ℝ, y = a * (x - 3)^2 + 4 → y = p * x^2 + q * x + r)
  (h2 : ∀ y1 : ℝ, y1 = p * (1 : ℝ)^2 + q * (1 : ℝ) + r → y1 = 10)
  (h3 : ∀ y2 : ℝ, y2 = p * (-1 : ℝ)^2 + q * (-1 : ℝ) + r → y2 = 14) :
  p + q + r = 10 :=
sorry

end NUMINAMATH_GPT_parabola_vertex_sum_l674_67474


namespace NUMINAMATH_GPT_inequality_proof_l674_67431

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 1) :
  (a / (a + 2 * b)^(1/3) + b / (b + 2 * c)^(1/3) + c / (c + 2 * a)^(1/3)) ≥ 1 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l674_67431


namespace NUMINAMATH_GPT_arc_length_ratio_l674_67480

theorem arc_length_ratio
  (h_circ : ∀ (x y : ℝ), (x - 1)^2 + y^2 = 1)
  (h_line : ∀ x y : ℝ, x - y = 0) :
  let shorter_arc := (1 / 4) * (2 * Real.pi)
  let longer_arc := 2 * Real.pi - shorter_arc
  shorter_arc / longer_arc = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arc_length_ratio_l674_67480


namespace NUMINAMATH_GPT_Mary_is_2_l674_67429

variable (M J : ℕ)

/-- Given the conditions from the problem, Mary's age can be determined to be 2. -/
theorem Mary_is_2 (h1 : J - 5 = M + 2) (h2 : J = 2 * M + 5) : M = 2 := by
  sorry

end NUMINAMATH_GPT_Mary_is_2_l674_67429


namespace NUMINAMATH_GPT_Anna_phone_chargers_l674_67498

-- Define the conditions and the goal in Lean
theorem Anna_phone_chargers (P L : ℕ) (h1 : L = 5 * P) (h2 : P + L = 24) : P = 4 :=
by
  sorry

end NUMINAMATH_GPT_Anna_phone_chargers_l674_67498


namespace NUMINAMATH_GPT_billboards_color_schemes_is_55_l674_67405

def adjacent_color_schemes (n : ℕ) : ℕ :=
  if h : n = 8 then 55 else 0

theorem billboards_color_schemes_is_55 :
  adjacent_color_schemes 8 = 55 :=
sorry

end NUMINAMATH_GPT_billboards_color_schemes_is_55_l674_67405


namespace NUMINAMATH_GPT_right_triangle_shorter_leg_l674_67467

theorem right_triangle_shorter_leg (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) : a = 25 ∨ b = 25 :=
sorry

end NUMINAMATH_GPT_right_triangle_shorter_leg_l674_67467


namespace NUMINAMATH_GPT_jump_difference_l674_67477

variable (runningRicciana jumpRicciana runningMargarita : ℕ)

theorem jump_difference :
  (runningMargarita + (2 * jumpRicciana - 1)) - (runningRicciana + jumpRicciana) = 1 :=
by
  -- Given conditions
  let runningRicciana := 20
  let jumpRicciana := 4
  let runningMargarita := 18
  -- The proof is omitted (using 'sorry')
  sorry

end NUMINAMATH_GPT_jump_difference_l674_67477


namespace NUMINAMATH_GPT_arithmetic_sequence_50th_term_l674_67493

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 4
  let a_n (n : ℕ) := a_1 + (n - 1) * d
  a_n 50 = 199 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_50th_term_l674_67493


namespace NUMINAMATH_GPT_frog_jump_plan_l674_67455

-- Define the vertices of the hexagon
inductive Vertex
| A | B | C | D | E | F

open Vertex

-- Define adjacency in the regular hexagon
def adjacent (v1 v2 : Vertex) : Prop :=
  match v1, v2 with
  | A, B | A, F | B, C | B, A | C, D | C, B | D, E | D, C | E, F | E, D | F, A | F, E => true
  | _, _ => false

-- Define the problem
def frog_jump_sequences_count : ℕ :=
  26

theorem frog_jump_plan : frog_jump_sequences_count = 26 := 
  sorry

end NUMINAMATH_GPT_frog_jump_plan_l674_67455


namespace NUMINAMATH_GPT_parabola_intersection_l674_67482

def parabola1 (x : ℝ) : ℝ := 3 * x ^ 2 - 4 * x + 2
def parabola2 (x : ℝ) : ℝ := 9 * x ^ 2 + 6 * x + 2

theorem parabola_intersection :
  ∃ (x y : ℝ), (parabola1 x = y ∧ parabola2 x = y) ∧ 
                ((x = 0 ∧ y = 2) ∨ (x = -5 / 3 ∧ y = 17)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_intersection_l674_67482


namespace NUMINAMATH_GPT_expression_equals_sqrt2_l674_67454

theorem expression_equals_sqrt2 :
  (1 + Real.pi)^0 + 2 - abs (-3) + 2 * Real.sin (Real.pi / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_expression_equals_sqrt2_l674_67454


namespace NUMINAMATH_GPT_wilma_garden_rows_l674_67426

theorem wilma_garden_rows :
  ∃ (rows : ℕ),
    (∃ (yellow green red total : ℕ),
      yellow = 12 ∧
      green = 2 * yellow ∧
      red = 42 ∧
      total = yellow + green + red ∧
      total / 13 = rows ∧
      rows = 6) :=
sorry

end NUMINAMATH_GPT_wilma_garden_rows_l674_67426


namespace NUMINAMATH_GPT_find_m_l674_67445

theorem find_m (m l : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, m)) (h_b : b = (l, -2))
  (h_parallel : ∃ k : ℝ, k ≠ 0 ∧ a = k • (a + 2 • b)) :
  m = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l674_67445


namespace NUMINAMATH_GPT_hyperbola_focus_and_asymptotes_l674_67473

def is_focus_on_y_axis (a b : ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
∃ c : ℝ, eq (c^2 * a) (c^2 * b)

def are_asymptotes_perpendicular (eq : ℝ → ℝ → Prop) : Prop :=
∃ k1 k2 : ℝ, (k1 != 0 ∧ k2 != 0 ∧ eq k1 k2 ∧ eq (-k1) k2)

theorem hyperbola_focus_and_asymptotes :
  is_focus_on_y_axis 1 (-1) (fun y x => y^2 - x^2 = 4) ∧ are_asymptotes_perpendicular (fun y x => y = x) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focus_and_asymptotes_l674_67473


namespace NUMINAMATH_GPT_minimum_elements_union_l674_67418

open Set

def A : Finset ℕ := sorry
def B : Finset ℕ := sorry

variable (size_A : A.card = 25)
variable (size_B : B.card = 18)
variable (at_least_10_not_in_A : (B \ A).card ≥ 10)

theorem minimum_elements_union : (A ∪ B).card = 35 :=
by
  sorry

end NUMINAMATH_GPT_minimum_elements_union_l674_67418


namespace NUMINAMATH_GPT_parametric_to_standard_equation_l674_67450

theorem parametric_to_standard_equation (x y t : ℝ) 
(h1 : x = 4 * t + 1) 
(h2 : y = -2 * t - 5) : 
x + 2 * y + 9 = 0 :=
by
  sorry

end NUMINAMATH_GPT_parametric_to_standard_equation_l674_67450


namespace NUMINAMATH_GPT_find_prime_factors_l674_67485

-- Define n and the prime numbers p and q
def n : ℕ := 400000001
def p : ℕ := 20201
def q : ℕ := 19801

-- Main theorem statement
theorem find_prime_factors (hn : n = p * q) 
  (hp : Prime p) 
  (hq : Prime q) : 
  n = 400000001 ∧ p = 20201 ∧ q = 19801 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_prime_factors_l674_67485


namespace NUMINAMATH_GPT_min_value_of_f_l674_67408

noncomputable def f (x : ℝ) : ℝ :=
  1 / (Real.sqrt (x^2 + 2)) + Real.sqrt (x^2 + 2)

theorem min_value_of_f :
  ∃ x : ℝ, f x = 3 * Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l674_67408


namespace NUMINAMATH_GPT_positive_integer_solutions_l674_67417

theorem positive_integer_solutions (a b : ℕ) (h_pos_ab : 0 < a ∧ 0 < b) :
  (∃ k : ℕ, k = a^2 / (2 * a * b^2 - b^3 + 1) ∧ 0 < k) ↔
  ∃ n : ℕ, (a = 2 * n ∧ b = 1) ∨ (a = n ∧ b = 2 * n) ∨ (a = 8 * n^4 - n ∧ b = 2 * n) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solutions_l674_67417


namespace NUMINAMATH_GPT_num_children_proof_l674_67424

-- Definitions and Main Problem
def legs_of_javier : ℕ := 2
def legs_of_wife : ℕ := 2
def legs_per_child : ℕ := 2
def legs_per_dog : ℕ := 4
def legs_of_cat : ℕ := 4
def num_dogs : ℕ := 2
def num_cats : ℕ := 1
def total_legs : ℕ := 22

-- Proof problem: Prove that the number of children (num_children) is equal to 3
theorem num_children_proof : ∃ num_children : ℕ, legs_of_javier + legs_of_wife + (num_children * legs_per_child) + (num_dogs * legs_per_dog) + (num_cats * legs_of_cat) = total_legs ∧ num_children = 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_num_children_proof_l674_67424


namespace NUMINAMATH_GPT_find_p_l674_67462

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem find_p (p : ℕ) (h : is_prime p) (hpgt1 : 1 < p) :
  8 * p^4 - 3003 = 1997 ↔ p = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l674_67462


namespace NUMINAMATH_GPT_baseball_card_devaluation_l674_67451

variable (x : ℝ) -- Note: x will represent the yearly percent decrease in decimal form (e.g., x = 0.10 for 10%)

theorem baseball_card_devaluation :
  (1 - x) * (1 - x) = 0.81 → x = 0.10 :=
by
  sorry

end NUMINAMATH_GPT_baseball_card_devaluation_l674_67451


namespace NUMINAMATH_GPT_what_percent_of_y_l674_67456

-- Given condition
axiom y_pos : ℝ → Prop

noncomputable def math_problem (y : ℝ) (h : y_pos y) : Prop :=
  (8 * y / 20 + 3 * y / 10 = 0.7 * y)

-- The theorem to be proved
theorem what_percent_of_y (y : ℝ) (h : y > 0) : 8 * y / 20 + 3 * y / 10 = 0.7 * y :=
by
  sorry

end NUMINAMATH_GPT_what_percent_of_y_l674_67456


namespace NUMINAMATH_GPT_integer_pairs_solution_l674_67423

def is_satisfied_solution (x y : ℤ) : Prop :=
  x^2 + y^2 = x + y + 2

theorem integer_pairs_solution :
  ∀ (x y : ℤ), is_satisfied_solution x y ↔ (x, y) = (-1, 0) ∨ (x, y) = (-1, 1) ∨ (x, y) = (0, -1) ∨ (x, y) = (0, 2) ∨ (x, y) = (1, -1) ∨ (x, y) = (1, 2) ∨ (x, y) = (2, 0) ∨ (x, y) = (2, 1) :=
by
  sorry

end NUMINAMATH_GPT_integer_pairs_solution_l674_67423


namespace NUMINAMATH_GPT_count_correct_statements_l674_67476

theorem count_correct_statements :
  ∃ (M: ℚ) (M1: ℚ) (M2: ℚ) (M3: ℚ) (M4: ℚ)
    (a b c d e : ℚ) (hacb : c ≠ 0) (habc: a ≠ 0) (hbcb : b ≠ 0) (hdcb: d ≠ 0) (hec: e ≠ 0),
  M = (ac + bd - ce) / c 
  ∧ M1 = (-bc - ad - ce) / c 
  ∧ M2 = (-dc - ab - ce) / c 
  ∧ M3 = (-dc - ab - de) / d 
  ∧ M4 = (ce - bd - ac) / (-c)
  ∧ M4 = M
  ∧ (M ≠ M3)
  ∧ (∀ M1, M1 = (-bc - ad - ce) / c → ((a = c ∨ b = d) ↔ b = d))
  ∧ (M4 = (ac + bd - ce)/c) :=
sorry

end NUMINAMATH_GPT_count_correct_statements_l674_67476


namespace NUMINAMATH_GPT_problem_statement_l674_67497

def M : Set ℝ := {x | (x + 3) * (x - 1) < 0}
def N : Set ℝ := {x | x <= -3}
def R (S : Set ℝ) : Set ℝ := {x | ∃ y ∈ S, x = y}

theorem problem_statement : R (M ∪ N) = {x | x >= 1} :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l674_67497


namespace NUMINAMATH_GPT_tate_total_years_eq_12_l674_67488

-- Definitions based on conditions
def high_school_normal_years : ℕ := 4
def high_school_years : ℕ := high_school_normal_years - 1
def college_years : ℕ := 3 * high_school_years
def total_years : ℕ := high_school_years + college_years

-- Statement to prove
theorem tate_total_years_eq_12 : total_years = 12 := by
  sorry

end NUMINAMATH_GPT_tate_total_years_eq_12_l674_67488


namespace NUMINAMATH_GPT_white_surface_fraction_l674_67463

-- Definition of the problem conditions
def larger_cube_surface_area : ℕ := 54
def white_cubes : ℕ := 6
def white_surface_area_minimized : ℕ := 5

-- Theorem statement proving the fraction of white surface area
theorem white_surface_fraction : (white_surface_area_minimized / larger_cube_surface_area : ℚ) = 5 / 54 := 
by
  sorry

end NUMINAMATH_GPT_white_surface_fraction_l674_67463


namespace NUMINAMATH_GPT_solve_floor_equation_l674_67486

theorem solve_floor_equation (x : ℝ) :
  (⌊⌊2 * x⌋ - 1 / 2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
sorry

end NUMINAMATH_GPT_solve_floor_equation_l674_67486


namespace NUMINAMATH_GPT_find_n_l674_67458

theorem find_n (n : ℕ) :
  (2^n - 1) % 3 = 0 ∧ (∃ m : ℤ, (2^n - 1) / 3 ∣ 4 * m^2 + 1) →
  ∃ j : ℕ, n = 2^j :=
by
  sorry

end NUMINAMATH_GPT_find_n_l674_67458


namespace NUMINAMATH_GPT_mask_donation_equation_l674_67412

theorem mask_donation_equation (x : ℝ) : 
  1 + (1 + x) + (1 + x)^2 = 4.75 :=
sorry

end NUMINAMATH_GPT_mask_donation_equation_l674_67412


namespace NUMINAMATH_GPT_eugene_payment_correct_l674_67484

noncomputable def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price - (original_price * discount_rate)

noncomputable def total_cost (quantity : ℕ) (price : ℝ) : ℝ :=
  quantity * price

noncomputable def eugene_total_cost : ℝ :=
  let tshirt_price := discounted_price 20 0.10
  let pants_price := discounted_price 80 0.10
  let shoes_price := discounted_price 150 0.15
  let hat_price := discounted_price 25 0.05
  let jacket_price := discounted_price 120 0.20
  let total_cost_before_tax := 
    total_cost 4 tshirt_price + 
    total_cost 3 pants_price + 
    total_cost 2 shoes_price + 
    total_cost 3 hat_price + 
    total_cost 1 jacket_price
  total_cost_before_tax + (total_cost_before_tax * 0.06)

theorem eugene_payment_correct : eugene_total_cost = 752.87 := by
  sorry

end NUMINAMATH_GPT_eugene_payment_correct_l674_67484


namespace NUMINAMATH_GPT_maximize_area_l674_67483

variable (x : ℝ)
def fence_length : ℝ := 240 - 2 * x
def area (x : ℝ) : ℝ := x * fence_length x

theorem maximize_area : fence_length 60 = 120 :=
  sorry

end NUMINAMATH_GPT_maximize_area_l674_67483


namespace NUMINAMATH_GPT_pairs_count_1432_1433_l674_67478

def PairsCount (n : ℕ) : ℕ :=
  -- The implementation would count the pairs (x, y) such that |x^2 - y^2| = n
  sorry

-- We write down the theorem that expresses what we need to prove
theorem pairs_count_1432_1433 : PairsCount 1432 = 8 ∧ PairsCount 1433 = 4 := by
  sorry

end NUMINAMATH_GPT_pairs_count_1432_1433_l674_67478


namespace NUMINAMATH_GPT_f_continuous_on_interval_f_not_bounded_variation_l674_67406

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else x * Real.sin (1 / x)

theorem f_continuous_on_interval : ContinuousOn f (Set.Icc 0 1) :=
sorry

theorem f_not_bounded_variation : ¬ BoundedVariationOn f (Set.Icc 0 1) :=
sorry

end NUMINAMATH_GPT_f_continuous_on_interval_f_not_bounded_variation_l674_67406


namespace NUMINAMATH_GPT_curve_line_and_circle_l674_67449

theorem curve_line_and_circle : 
  ∀ x y : ℝ, (x^3 + x * y^2 = 2 * x) ↔ (x = 0 ∨ x^2 + y^2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_curve_line_and_circle_l674_67449


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l674_67495

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (h1 : S 4 = 3) (h2 : S 8 = 7) : S 12 = 12 :=
by
  -- placeholder for the proof, details omitted
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l674_67495


namespace NUMINAMATH_GPT_sequence_a_n_l674_67499

theorem sequence_a_n (a : ℤ) (h : (-1)^1 * 1 + a + (-1)^4 * 4 + a = 3 * ( (-1)^2 * 2 + a )) :
  a = -3 ∧ ((-1)^100 * 100 + a) = 97 :=
by
  sorry  -- proof is omitted

end NUMINAMATH_GPT_sequence_a_n_l674_67499


namespace NUMINAMATH_GPT_second_shirt_price_l674_67430

-- Define the conditions
def price_first_shirt := 82
def price_third_shirt := 90
def min_avg_price_remaining_shirts := 104
def total_shirts := 10
def desired_avg_price := 100

-- Prove the price of the second shirt
theorem second_shirt_price : 
  ∀ (P : ℝ), 
  (price_first_shirt + P + price_third_shirt + 7 * min_avg_price_remaining_shirts = total_shirts * desired_avg_price) → 
  P = 100 :=
by
  sorry

end NUMINAMATH_GPT_second_shirt_price_l674_67430


namespace NUMINAMATH_GPT_line_perpendicular_intersection_l674_67434

noncomputable def line_equation (x y : ℝ) := 3 * x + y + 2 = 0

def is_perpendicular (m1 m2 : ℝ) := m1 * m2 = -1

theorem line_perpendicular_intersection (x y : ℝ) :
  (x - y + 2 = 0) →
  (2 * x + y + 1 = 0) →
  is_perpendicular (1 / 3) (-3) →
  line_equation x y := 
sorry

end NUMINAMATH_GPT_line_perpendicular_intersection_l674_67434


namespace NUMINAMATH_GPT_sin_2phi_l674_67440

theorem sin_2phi (φ : ℝ) (h : (7 : ℝ) / 13 + Real.sin φ = Real.cos φ) : Real.sin (2 * φ) = 120 / 169 := 
by 
  sorry

end NUMINAMATH_GPT_sin_2phi_l674_67440


namespace NUMINAMATH_GPT_evaluate_f_neg_a_l674_67407

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem evaluate_f_neg_a (a : ℝ) (h : f a = 1 / 3) : f (-a) = 5 / 3 :=
by sorry

end NUMINAMATH_GPT_evaluate_f_neg_a_l674_67407


namespace NUMINAMATH_GPT_count_big_boxes_l674_67457

theorem count_big_boxes (B : ℕ) (h : 7 * B + 4 * 9 = 71) : B = 5 :=
sorry

end NUMINAMATH_GPT_count_big_boxes_l674_67457


namespace NUMINAMATH_GPT_solve_rational_equation_solve_quadratic_equation_l674_67443

-- Statement for the first equation
theorem solve_rational_equation (x : ℝ) (h : x ≠ 1) : 
  (x / (x - 1) + 2 / (1 - x) = 2) → (x = 0) :=
by intro h1; sorry

-- Statement for the second equation
theorem solve_quadratic_equation (x : ℝ) : 
  (2 * x^2 + 6 * x - 3 = 0) → (x = 1/2 ∨ x = -3) :=
by intro h1; sorry

end NUMINAMATH_GPT_solve_rational_equation_solve_quadratic_equation_l674_67443


namespace NUMINAMATH_GPT_infinitely_many_not_2a_3b_5c_l674_67490

theorem infinitely_many_not_2a_3b_5c : ∃ᶠ x : ℤ in Filter.cofinite, ∀ a b c : ℕ, x % 120 ≠ (2^a + 3^b - 5^c) % 120 :=
by
  sorry

end NUMINAMATH_GPT_infinitely_many_not_2a_3b_5c_l674_67490
