import Mathlib

namespace NUMINAMATH_GPT_perp_tangents_l918_91814

theorem perp_tangents (a b : ℝ) (h : a + b = 5) (tangent_perp : ∀ x y : ℝ, x = 1 ∧ y = 1) :
  a / b = 1 / 3 :=
sorry

end NUMINAMATH_GPT_perp_tangents_l918_91814


namespace NUMINAMATH_GPT_domain_of_function_l918_91811

theorem domain_of_function :
  ∀ x : ℝ, (x > 0) ∧ (x ≤ 2) ∧ (x ≠ 1) ↔ ∀ x, (∃ y : ℝ, y = (1 / (Real.log x / Real.log 10) + Real.sqrt (2 - x))) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l918_91811


namespace NUMINAMATH_GPT_find_y_given_z_25_l918_91825

theorem find_y_given_z_25 (k m x y z : ℝ) 
  (hk : y = k * x) 
  (hm : z = m * x)
  (hy5 : y = 10) 
  (hx5z15 : z = 15) 
  (hz25 : z = 25) : 
  y = 50 / 3 := 
  by sorry

end NUMINAMATH_GPT_find_y_given_z_25_l918_91825


namespace NUMINAMATH_GPT_two_digit_product_l918_91851

theorem two_digit_product (x y : ℕ) (h₁ : 10 ≤ x) (h₂ : x < 100) (h₃ : 10 ≤ y) (h₄ : y < 100) (h₅ : x * y = 4320) :
  (x = 60 ∧ y = 72) ∨ (x = 72 ∧ y = 60) :=
sorry

end NUMINAMATH_GPT_two_digit_product_l918_91851


namespace NUMINAMATH_GPT_geometric_sum_eight_terms_l918_91845

theorem geometric_sum_eight_terms :
  let a0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 4
  let n := 8
  let S_n := a0 * (1 - r^n) / (1 - r)
  S_n = 65535 / 147456 := by
  sorry

end NUMINAMATH_GPT_geometric_sum_eight_terms_l918_91845


namespace NUMINAMATH_GPT_waiter_customers_l918_91887

variable (initial_customers left_customers new_customers : ℕ)

theorem waiter_customers 
  (h1 : initial_customers = 33)
  (h2 : left_customers = 31)
  (h3 : new_customers = 26) :
  (initial_customers - left_customers + new_customers = 28) := 
by
  sorry

end NUMINAMATH_GPT_waiter_customers_l918_91887


namespace NUMINAMATH_GPT_infinite_indices_exist_l918_91807

theorem infinite_indices_exist (a : ℕ → ℕ) (h_seq : ∀ n, a n < a (n + 1)) :
  ∃ᶠ m in ⊤, ∃ x y h k : ℕ, 0 < h ∧ h < k ∧ k < m ∧ a m = x * a h + y * a k :=
by sorry

end NUMINAMATH_GPT_infinite_indices_exist_l918_91807


namespace NUMINAMATH_GPT_average_speed_correct_l918_91865

-- Definitions of distances and speeds
def distance1 := 50 -- miles
def speed1 := 20 -- miles per hour
def distance2 := 20 -- miles
def speed2 := 40 -- miles per hour
def distance3 := 30 -- miles
def speed3 := 15 -- miles per hour

-- Definition of total distance and total time
def total_distance := distance1 + distance2 + distance3
def time1 := distance1 / speed1
def time2 := distance2 / speed2
def time3 := distance3 / speed3
def total_time := time1 + time2 + time3

-- Definition of average speed
def average_speed := total_distance / total_time

-- Statement to be proven
theorem average_speed_correct : average_speed = 20 := 
by 
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_average_speed_correct_l918_91865


namespace NUMINAMATH_GPT_center_and_radius_of_circle_l918_91801

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 6 * y + 6 = 0

-- State the theorem
theorem center_and_radius_of_circle :
  (∃ x₀ y₀ r, (∀ x y, circle_eq x y ↔ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
  x₀ = 1 ∧ y₀ = -3 ∧ r = 2) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_center_and_radius_of_circle_l918_91801


namespace NUMINAMATH_GPT_files_remaining_l918_91818

def initial_music_files : ℕ := 27
def initial_video_files : ℕ := 42
def initial_doc_files : ℕ := 12
def compression_ratio_music : ℕ := 2
def compression_ratio_video : ℕ := 3
def files_deleted : ℕ := 11

def compressed_music_files : ℕ := initial_music_files * compression_ratio_music
def compressed_video_files : ℕ := initial_video_files * compression_ratio_video
def total_compressed_files : ℕ := compressed_music_files + compressed_video_files + initial_doc_files

theorem files_remaining : total_compressed_files - files_deleted = 181 := by
  -- we skip the proof for now
  sorry

end NUMINAMATH_GPT_files_remaining_l918_91818


namespace NUMINAMATH_GPT_INPUT_is_input_statement_l918_91867

-- Define what constitutes each type of statement
def isOutputStatement (stmt : String) : Prop :=
  stmt = "PRINT"

def isInputStatement (stmt : String) : Prop :=
  stmt = "INPUT"

def isConditionalStatement (stmt : String) : Prop :=
  stmt = "THEN"

def isEndStatement (stmt : String) : Prop :=
  stmt = "END"

-- The main theorem
theorem INPUT_is_input_statement : isInputStatement "INPUT" := by
  sorry

end NUMINAMATH_GPT_INPUT_is_input_statement_l918_91867


namespace NUMINAMATH_GPT_problem1_part1_problem1_part2_problem2_l918_91824

noncomputable def problem1_condition1 (m : ℕ) (a : ℕ) : Prop := 4^m = a
noncomputable def problem1_condition2 (n : ℕ) (b : ℕ) : Prop := 8^n = b

theorem problem1_part1 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(2*m + 3*n) = a * b :=
by sorry

theorem problem1_part2 (m n a b : ℕ) (h1 : 4^m = a) (h2 : 8^n = b) : 2^(4*m - 6*n) = (a^2) / (b^2) :=
by sorry

theorem problem2 (x : ℕ) (h : 2 * 8^x * 16 = 2^23) : x = 6 :=
by sorry

end NUMINAMATH_GPT_problem1_part1_problem1_part2_problem2_l918_91824


namespace NUMINAMATH_GPT_number_of_small_pizzas_ordered_l918_91895

-- Define the problem conditions
def benBrothers : Nat := 2
def slicesPerPerson : Nat := 12
def largePizzaSlices : Nat := 14
def smallPizzaSlices : Nat := 8
def numLargePizzas : Nat := 2

-- Define the statement to prove
theorem number_of_small_pizzas_ordered : 
  ∃ (s : Nat), (benBrothers + 1) * slicesPerPerson - numLargePizzas * largePizzaSlices = s * smallPizzaSlices ∧ s = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_small_pizzas_ordered_l918_91895


namespace NUMINAMATH_GPT_word_count_with_a_l918_91841

-- Defining the constants for the problem
def alphabet_size : ℕ := 26
def no_a_size : ℕ := 25

-- Calculating words that contain 'A' for lengths 1 to 5
def words_with_a (len : ℕ) : ℕ :=
  alphabet_size ^ len - no_a_size ^ len

-- The main theorem statement
theorem word_count_with_a : words_with_a 1 + words_with_a 2 + words_with_a 3 + words_with_a 4 + words_with_a 5 = 2186085 :=
by
  -- Calculations are established in the problem statement
  sorry

end NUMINAMATH_GPT_word_count_with_a_l918_91841


namespace NUMINAMATH_GPT_arithmetic_geometric_sequence_l918_91810

theorem arithmetic_geometric_sequence (a : ℕ → ℕ) (S : ℕ → ℕ) (q : ℕ)
  (h₀ : ∀ n, a n = 2^(n-1))
  (h₁ : a 1 = 1)
  (h₂ : a 1 + a 2 + a 3 = 7)
  (h₃ : q > 0) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, S n = 2^n - 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_arithmetic_geometric_sequence_l918_91810


namespace NUMINAMATH_GPT_system_of_equations_solution_l918_91808

theorem system_of_equations_solution (x y z : ℤ) :
  x^2 - 9 * y^2 - z^2 = 0 ∧ z = x - 3 * y ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (∃ k : ℤ, x = 3 * k ∧ y = k ∧ z = 0) := 
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l918_91808


namespace NUMINAMATH_GPT_parallelogram_area_correct_l918_91847

noncomputable def parallelogram_area (s1 s2 : ℝ) (a : ℝ) : ℝ :=
s2 * (2 * s2 * Real.sin a)

theorem parallelogram_area_correct (s2 a : ℝ) (h_pos_s2 : 0 < s2) :
  parallelogram_area (2 * s2) s2 a = 2 * s2^2 * Real.sin a :=
by
  unfold parallelogram_area
  sorry

end NUMINAMATH_GPT_parallelogram_area_correct_l918_91847


namespace NUMINAMATH_GPT_two_a_plus_two_d_eq_zero_l918_91840

theorem two_a_plus_two_d_eq_zero
  (a b c d : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h₃ : d ≠ 0)
  (h₄ : ∀ x : ℝ, (2 * a * ((2 * a * x + b) / (3 * c * x + 2 * d)) + b)
                 / (3 * c * ((2 * a * x + b) / (3 * c * x + 2 * d)) + 2 * d) = x) :
  2 * a + 2 * d = 0 :=
by sorry

end NUMINAMATH_GPT_two_a_plus_two_d_eq_zero_l918_91840


namespace NUMINAMATH_GPT_find_n_values_l918_91822

-- Define a function that calculates the polynomial expression
def prime_expression (n : ℕ) : ℕ :=
  n^4 - 27 * n^2 + 121

-- State the problem as a theorem
theorem find_n_values (n : ℕ) (h : Nat.Prime (prime_expression n)) : n = 2 ∨ n = 5 :=
  sorry

end NUMINAMATH_GPT_find_n_values_l918_91822


namespace NUMINAMATH_GPT_brownies_total_l918_91881

theorem brownies_total :
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  after_mooney_ate + additional_brownies = 36 :=
by
  let initial_brownies := 2 * 12
  let after_father_ate := initial_brownies - 8
  let after_mooney_ate := after_father_ate - 4
  let additional_brownies := 2 * 12
  show after_mooney_ate + additional_brownies = 36
  sorry

end NUMINAMATH_GPT_brownies_total_l918_91881


namespace NUMINAMATH_GPT_exterior_angle_measure_l918_91802

theorem exterior_angle_measure (sum_interior_angles : ℝ) (h : sum_interior_angles = 1260) :
  ∃ (n : ℕ) (d : ℝ), (n - 2) * 180 = sum_interior_angles ∧ d = 360 / n ∧ d = 40 := 
by
  sorry

end NUMINAMATH_GPT_exterior_angle_measure_l918_91802


namespace NUMINAMATH_GPT_fraction_power_multiplication_l918_91823

theorem fraction_power_multiplication :
  ((1 : ℝ) / 3) ^ 4 * ((1 : ℝ) / 5) = ((1 : ℝ) / 405) := by
  sorry

end NUMINAMATH_GPT_fraction_power_multiplication_l918_91823


namespace NUMINAMATH_GPT_min_value_cos_sin_l918_91857

noncomputable def min_value_expression : ℝ :=
  -1 / 2

theorem min_value_cos_sin (θ : ℝ) (hθ1 : 0 ≤ θ) (hθ2 : θ ≤ 3 * Real.pi / 2) :
  ∃ (y : ℝ), y = Real.cos (θ / 3) * (1 - Real.sin θ) ∧ y = min_value_expression :=
sorry

end NUMINAMATH_GPT_min_value_cos_sin_l918_91857


namespace NUMINAMATH_GPT_quadratic_equal_roots_iff_l918_91821

theorem quadratic_equal_roots_iff (k : ℝ) :
  (∃ x : ℝ, x^2 - k * x + 9 = 0 ∧ x^2 - k * x + 9 = 0 ∧ x = x) ↔ k^2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equal_roots_iff_l918_91821


namespace NUMINAMATH_GPT_find_second_number_l918_91894

-- The Lean statement for the given math problem:

theorem find_second_number
  (x y z : ℝ)  -- Represent the three numbers
  (h1 : x = 2 * y)  -- The first number is twice the second
  (h2 : z = (1/3) * x)  -- The third number is one-third of the first
  (h3 : x + y + z = 110)  -- The sum of the three numbers is 110
  : y = 30 :=  -- The second number is 30
sorry

end NUMINAMATH_GPT_find_second_number_l918_91894


namespace NUMINAMATH_GPT_option_C_forms_a_set_l918_91834

-- Definition of the criteria for forming a set
def well_defined (criterion : Prop) : Prop := criterion

-- Criteria for option C: all female students in grade one of Jiu Middle School
def grade_one_students_criteria (is_female : Prop) (is_grade_one_student : Prop) : Prop :=
  is_female ∧ is_grade_one_student

-- Proof statement
theorem option_C_forms_a_set :
  ∀ (is_female : Prop) (is_grade_one_student : Prop), well_defined (grade_one_students_criteria is_female is_grade_one_student) :=
  by sorry

end NUMINAMATH_GPT_option_C_forms_a_set_l918_91834


namespace NUMINAMATH_GPT_distance_from_neg6_to_origin_l918_91828

theorem distance_from_neg6_to_origin :
  abs (-6) = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_neg6_to_origin_l918_91828


namespace NUMINAMATH_GPT_slope_of_line_determined_by_solutions_l918_91831

theorem slope_of_line_determined_by_solutions (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : 3 / x₁ +  4 / y₁ = 0)
  (h₂ : 3 / x₂ + 4 / y₂ = 0) :
  (y₂ - y₁) / (x₂ - x₁) = -4 / 3 :=
sorry

end NUMINAMATH_GPT_slope_of_line_determined_by_solutions_l918_91831


namespace NUMINAMATH_GPT_homework_duration_equation_l918_91875

-- Given conditions
def initial_duration : ℝ := 120
def final_duration : ℝ := 60
variable (x : ℝ)

-- The goal is to prove that the appropriate equation holds
theorem homework_duration_equation : initial_duration * (1 - x)^2 = final_duration := 
sorry

end NUMINAMATH_GPT_homework_duration_equation_l918_91875


namespace NUMINAMATH_GPT_smallest_n_l918_91800

theorem smallest_n {n : ℕ} (h1 : n ≡ 4 [MOD 6]) (h2 : n ≡ 3 [MOD 7]) (h3 : n > 10) : n = 52 :=
sorry

end NUMINAMATH_GPT_smallest_n_l918_91800


namespace NUMINAMATH_GPT_pollen_scientific_notation_correct_l918_91874

def moss_flower_pollen_diameter := 0.0000084
def pollen_scientific_notation := 8.4 * 10^(-6)

theorem pollen_scientific_notation_correct :
  moss_flower_pollen_diameter = pollen_scientific_notation :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_pollen_scientific_notation_correct_l918_91874


namespace NUMINAMATH_GPT_min_value_of_expression_l918_91876

theorem min_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) : 
  (1 / x + 4 / y) ≥ 9 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l918_91876


namespace NUMINAMATH_GPT_mod_equiv_pow_five_l918_91855

theorem mod_equiv_pow_five (m : ℤ) (hm : 0 ≤ m ∧ m < 11) (h : 12^5 ≡ m [ZMOD 11]) : m = 1 :=
by
  sorry

end NUMINAMATH_GPT_mod_equiv_pow_five_l918_91855


namespace NUMINAMATH_GPT_inverse_relation_a1600_inverse_relation_a400_l918_91899

variable (a b : ℝ)

def k := 400 

theorem inverse_relation_a1600 : (a * b = k) → (a = 1600) → (b = 0.25) :=
by
  sorry

theorem inverse_relation_a400 : (a * b = k) → (a = 400) → (b = 1) :=
by
  sorry

end NUMINAMATH_GPT_inverse_relation_a1600_inverse_relation_a400_l918_91899


namespace NUMINAMATH_GPT_probability_of_two_black_balls_is_one_fifth_l918_91884

noncomputable def probability_of_two_black_balls (W B : Nat) : ℚ :=
  let total_balls := W + B
  let prob_black1 := (B : ℚ) / total_balls
  let prob_black2_given_black1 := (B - 1 : ℚ) / (total_balls - 1)
  prob_black1 * prob_black2_given_black1

theorem probability_of_two_black_balls_is_one_fifth : 
  probability_of_two_black_balls 8 7 = 1 / 5 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_two_black_balls_is_one_fifth_l918_91884


namespace NUMINAMATH_GPT_factor_1024_count_l918_91837

theorem factor_1024_count :
  ∃ (n : ℕ), 
  (∀ (a b c : ℕ), (a >= b) → (b >= c) → (2^a * 2^b * 2^c = 1024) → a + b + c = 10) ∧ n = 14 :=
sorry

end NUMINAMATH_GPT_factor_1024_count_l918_91837


namespace NUMINAMATH_GPT_hockey_team_helmets_l918_91890

theorem hockey_team_helmets (r b : ℕ) 
  (h1 : b = r - 6) 
  (h2 : r * 3 = b * 5) : 
  r + b = 24 :=
by
  sorry

end NUMINAMATH_GPT_hockey_team_helmets_l918_91890


namespace NUMINAMATH_GPT_factor_expression_l918_91862

variable (x : ℝ)

theorem factor_expression :
  (4 * x ^ 3 + 100 * x ^ 2 - 28) - (-9 * x ^ 3 + 2 * x ^ 2 - 28) = 13 * x ^ 2 * (x + 7) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l918_91862


namespace NUMINAMATH_GPT_smallest_sum_of_18_consecutive_integers_is_perfect_square_l918_91888

theorem smallest_sum_of_18_consecutive_integers_is_perfect_square 
  (n : ℕ) 
  (S : ℕ) 
  (h1 : S = 9 * (2 * n + 17)) 
  (h2 : ∃ k : ℕ, 2 * n + 17 = k^2) 
  (h3 : ∀ m : ℕ, m < 5 → 2 * n + 17 ≠ m^2) : 
  S = 225 := 
by
  sorry

end NUMINAMATH_GPT_smallest_sum_of_18_consecutive_integers_is_perfect_square_l918_91888


namespace NUMINAMATH_GPT_company_x_total_employees_l918_91891

-- Definitions for conditions
def initial_percentage : ℝ := 0.60
def Q2_hiring_males : ℕ := 30
def Q2_new_percentage : ℝ := 0.57
def Q3_hiring_females : ℕ := 50
def Q3_new_percentage : ℝ := 0.62
def Q4_hiring_males : ℕ := 40
def Q4_hiring_females : ℕ := 10
def Q4_new_percentage : ℝ := 0.58

-- Statement of the proof problem
theorem company_x_total_employees :
  ∃ (E : ℕ) (F : ℕ), 
    (F = initial_percentage * E ∧
     F = Q2_new_percentage * (E + Q2_hiring_males) ∧
     F + Q3_hiring_females = Q3_new_percentage * (E + Q2_hiring_males + Q3_hiring_females) ∧
     F + Q3_hiring_females + Q4_hiring_females = Q4_new_percentage * (E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females)) →
    E + Q2_hiring_males + Q3_hiring_females + Q4_hiring_males + Q4_hiring_females = 700 :=
sorry

end NUMINAMATH_GPT_company_x_total_employees_l918_91891


namespace NUMINAMATH_GPT_find_k_l918_91842

theorem find_k : ∃ b k : ℝ, (∀ x : ℝ, (x + b)^2 = x^2 - 20 * x + k) ∧ k = 100 := by
  sorry

end NUMINAMATH_GPT_find_k_l918_91842


namespace NUMINAMATH_GPT_head_start_ratio_l918_91819

variable (Va Vb L H : ℕ)

-- Conditions
def speed_relation : Prop := Va = (4 * Vb) / 3

-- The head start fraction that makes A and B finish the race at the same time given the speed relation
theorem head_start_ratio (Va Vb L H : ℕ)
  (h1 : speed_relation Va Vb)
  (h2 : L > 0) : (H = L / 4) :=
sorry

end NUMINAMATH_GPT_head_start_ratio_l918_91819


namespace NUMINAMATH_GPT_probability_five_heads_in_six_tosses_is_09375_l918_91858

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

noncomputable def probability_exact_heads (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * (p^k) * ((1-p)^(n-k))
  
theorem probability_five_heads_in_six_tosses_is_09375 :
  probability_exact_heads 6 5 0.5 = 0.09375 :=
by
  sorry

end NUMINAMATH_GPT_probability_five_heads_in_six_tosses_is_09375_l918_91858


namespace NUMINAMATH_GPT_find_a_given_coefficient_l918_91863

theorem find_a_given_coefficient (a : ℝ) (h : (a^3 * 10 = 80)) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_given_coefficient_l918_91863


namespace NUMINAMATH_GPT_eventually_periodic_sequence_l918_91836

noncomputable def eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ N k : ℕ, k > 0 ∧ ∀ m ≥ N, a m = a (m + k)

theorem eventually_periodic_sequence
  (a : ℕ → ℕ)
  (h_pos : ∀ n, a n > 0)
  (h_condition : ∀ n, a n * a (n + 1) = a (n + 2) * a (n + 3)) :
  eventually_periodic a :=
sorry

end NUMINAMATH_GPT_eventually_periodic_sequence_l918_91836


namespace NUMINAMATH_GPT_messages_after_noon_l918_91872

theorem messages_after_noon (t n : ℕ) (h1 : t = 39) (h2 : n = 21) : t - n = 18 := by
  sorry

end NUMINAMATH_GPT_messages_after_noon_l918_91872


namespace NUMINAMATH_GPT_equilateral_triangle_perimeter_l918_91889

theorem equilateral_triangle_perimeter (s : ℝ) 
  (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  -- Proof steps (omitted)
  sorry

end NUMINAMATH_GPT_equilateral_triangle_perimeter_l918_91889


namespace NUMINAMATH_GPT_correct_choice_l918_91812

variable (a b : ℝ) (p q : Prop) (x : ℝ)

-- Proposition A: Incorrect because x > 3 is a sufficient condition for x > 2.
def propositionA : Prop := (∀ x : ℝ, x > 3 → x > 2) ∧ ¬ (∀ x : ℝ, x > 2 → x > 3)

-- Proposition B: Incorrect negation form.
def propositionB : Prop := ¬ (¬p → ¬q) ∧ (q → p)

-- Proposition C: Incorrect because it should be 1/a > 1/b given 0 < a < b.
def propositionC : Prop := (a > 0 ∧ b < 0) ∧ ¬ (1/a < 1/b)

-- Proposition D: Correct negation form.
def propositionD_negation_correct : Prop := 
  (¬ ∃ x : ℝ, x^2 = 1) = ( ∀ x : ℝ, x^2 ≠ 1)

theorem correct_choice : propositionD_negation_correct := by
  sorry

end NUMINAMATH_GPT_correct_choice_l918_91812


namespace NUMINAMATH_GPT_joe_initial_money_l918_91860

theorem joe_initial_money (cost_notebook cost_book money_left : ℕ) 
                          (num_notebooks num_books : ℕ)
                          (h1 : cost_notebook = 4) 
                          (h2 : cost_book = 7)
                          (h3 : num_notebooks = 7) 
                          (h4 : num_books = 2) 
                          (h5 : money_left = 14) :
  (num_notebooks * cost_notebook + num_books * cost_book + money_left) = 56 := by
  sorry

end NUMINAMATH_GPT_joe_initial_money_l918_91860


namespace NUMINAMATH_GPT_geom_sequence_a1_value_l918_91869

-- Define the conditions and the statement
theorem geom_sequence_a1_value (a_1 a_6 : ℚ) (a_3 a_4 : ℚ)
  (h1 : a_1 + a_6 = 11)
  (h2 : a_3 * a_4 = 32 / 9) :
  (a_1 = 32 / 3 ∨ a_1 = 1 / 3) :=
by 
-- We will prove the theorem here (skipped with sorry)
sorry

end NUMINAMATH_GPT_geom_sequence_a1_value_l918_91869


namespace NUMINAMATH_GPT_heather_payment_per_weed_l918_91892

noncomputable def seconds_in_hour : ℕ := 60 * 60

noncomputable def weeds_per_hour (seconds_per_weed : ℕ) : ℕ :=
  seconds_in_hour / seconds_per_weed

noncomputable def payment_per_weed (hourly_pay : ℕ) (weeds_per_hour : ℕ) : ℚ :=
  hourly_pay / weeds_per_hour

theorem heather_payment_per_weed (seconds_per_weed : ℕ) (hourly_pay : ℕ) :
  seconds_per_weed = 18 ∧ hourly_pay = 10 → payment_per_weed hourly_pay (weeds_per_hour seconds_per_weed) = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_heather_payment_per_weed_l918_91892


namespace NUMINAMATH_GPT_scientific_notation_correct_l918_91879

theorem scientific_notation_correct :
  1200000000 = 1.2 * 10^9 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_correct_l918_91879


namespace NUMINAMATH_GPT_area_of_triangle_XPQ_l918_91878
open Real

/-- Given a triangle XYZ with area 15 square units and points P, Q, R on sides XY, YZ, and ZX respectively,
where XP = 3, PY = 6, and triangles XPQ and quadrilateral PYRQ have equal areas, 
prove that the area of triangle XPQ is 5/3 square units. -/
theorem area_of_triangle_XPQ 
  (Area_XYZ : ℝ) (h1 : Area_XYZ = 15)
  (XP PY : ℝ) (h2 : XP = 3) (h3 : PY = 6)
  (h4 : ∃ (Area_XPQ : ℝ) (Area_PYRQ : ℝ), Area_XPQ = Area_PYRQ) :
  ∃ (Area_XPQ : ℝ), Area_XPQ = 5/3 :=
sorry

end NUMINAMATH_GPT_area_of_triangle_XPQ_l918_91878


namespace NUMINAMATH_GPT_digit_150_in_17_div_70_l918_91803

noncomputable def repeating_sequence_170 : List Nat := [2, 4, 2, 8, 5, 7]

theorem digit_150_in_17_div_70 : (repeating_sequence_170.get? (150 % 6 - 1) = some 7) := by
  sorry

end NUMINAMATH_GPT_digit_150_in_17_div_70_l918_91803


namespace NUMINAMATH_GPT_unique_integer_solution_m_l918_91830

theorem unique_integer_solution_m {m : ℤ} (h : ∀ x : ℤ, |2 * x - m| ≤ 1 → x = 2) : m = 4 := 
sorry

end NUMINAMATH_GPT_unique_integer_solution_m_l918_91830


namespace NUMINAMATH_GPT_solve_for_m_l918_91877

theorem solve_for_m :
  (∀ (m : ℕ), 
   ((1:ℚ)^(m+1) / 5^(m+1) * 1^18 / 4^18 = 1 / (2 * 10^35)) → m = 34) := 
by apply sorry

end NUMINAMATH_GPT_solve_for_m_l918_91877


namespace NUMINAMATH_GPT_find_goods_train_speed_l918_91854

-- Definition of given conditions
def speed_of_man_train_kmph : ℝ := 120
def time_goods_train_seconds : ℝ := 9
def length_goods_train_meters : ℝ := 350

-- The proof statement
theorem find_goods_train_speed :
  let relative_speed_mps := (speed_of_man_train_kmph + goods_train_speed_kmph) * (5 / 18)
  ∃ (goods_train_speed_kmph : ℝ), relative_speed_mps = length_goods_train_meters / time_goods_train_seconds ∧ goods_train_speed_kmph = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_goods_train_speed_l918_91854


namespace NUMINAMATH_GPT_smallest_integer_is_840_l918_91896

def is_prime_digit (d : ℕ) : Prop :=
  d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

def all_divide (N : ℕ) : Prop :=
  (2 ∣ N) ∧ (3 ∣ N) ∧ (5 ∣ N) ∧ (7 ∣ N)

def no_prime_digit (N : ℕ) : Prop :=
  ∀ d ∈ N.digits 10, ¬ is_prime_digit d

def smallest_satisfying_N (N : ℕ) : Prop :=
  no_prime_digit N ∧ all_divide N ∧ ∀ M, no_prime_digit M → all_divide M → N ≤ M

theorem smallest_integer_is_840 : smallest_satisfying_N 840 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_is_840_l918_91896


namespace NUMINAMATH_GPT_statement_b_statement_e_l918_91843

-- Statement (B): ∀ x, if x^3 > 0 then x > 0.
theorem statement_b (x : ℝ) : x^3 > 0 → x > 0 := sorry

-- Statement (E): ∀ x, if x < 1 then x^3 < x.
theorem statement_e (x : ℝ) : x < 1 → x^3 < x := sorry

end NUMINAMATH_GPT_statement_b_statement_e_l918_91843


namespace NUMINAMATH_GPT_index_difference_l918_91816

theorem index_difference (n f m : ℕ) (h_n : n = 25) (h_f : f = 8) (h_m : m = 25 - 8) :
  (n - f) / n - (n - m) / n = 9 / 25 :=
by
  -- The proof is to be completed here.
  sorry

end NUMINAMATH_GPT_index_difference_l918_91816


namespace NUMINAMATH_GPT_find_cost_of_fourth_cd_l918_91849

variables (cost1 cost2 cost3 cost4 : ℕ)
variables (h1 : (cost1 + cost2 + cost3) / 3 = 15)
variables (h2 : (cost1 + cost2 + cost3 + cost4) / 4 = 16)

theorem find_cost_of_fourth_cd : cost4 = 19 := 
by 
  sorry

end NUMINAMATH_GPT_find_cost_of_fourth_cd_l918_91849


namespace NUMINAMATH_GPT_calculate_V3_at_2_l918_91897

def polynomial (x : ℕ) : ℕ :=
  (((((2 * x + 5) * x + 6) * x + 23) * x - 8) * x + 10) * x - 3

theorem calculate_V3_at_2 : polynomial 2 = 71 := by
  sorry

end NUMINAMATH_GPT_calculate_V3_at_2_l918_91897


namespace NUMINAMATH_GPT_johns_cycling_speed_needed_l918_91844

theorem johns_cycling_speed_needed 
  (swim_speed : Float := 3)
  (swim_distance : Float := 0.5)
  (run_speed : Float := 8)
  (run_distance : Float := 4)
  (total_time : Float := 3)
  (bike_distance : Float := 20) :
  (bike_distance / (total_time - (swim_distance / swim_speed + run_distance / run_speed))) = 60 / 7 := 
  by
  sorry

end NUMINAMATH_GPT_johns_cycling_speed_needed_l918_91844


namespace NUMINAMATH_GPT_non_obtuse_triangle_medians_ge_4R_l918_91813

theorem non_obtuse_triangle_medians_ge_4R
  (A B C : Type*)
  (triangle_non_obtuse : ∀ (α β γ : ℝ), α ≤ 90 ∧ β ≤ 90 ∧ γ ≤ 90)
  (m_a m_b m_c : ℝ)
  (R : ℝ)
  (h1 : AO + BO ≤ AM + BM)
  (h2 : AM = 2 * m_a / 3 ∧ BM = 2 * m_b / 3)
  (h3 : AO + BO = 2 * R)
  (h4 : m_c ≥ R) : 
  m_a + m_b + m_c ≥ 4 * R :=
by
  sorry

end NUMINAMATH_GPT_non_obtuse_triangle_medians_ge_4R_l918_91813


namespace NUMINAMATH_GPT_sum_first_six_terms_geometric_seq_l918_91883

theorem sum_first_six_terms_geometric_seq :
  let a := (1 : ℚ) / 4 
  let r := (1 : ℚ) / 4 
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (1365 / 16384 : ℚ) :=
by
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  sorry

end NUMINAMATH_GPT_sum_first_six_terms_geometric_seq_l918_91883


namespace NUMINAMATH_GPT_continuity_of_f_at_2_l918_91850

def f (x : ℝ) := -2 * x^2 - 5

theorem continuity_of_f_at_2 : ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by {
  sorry
}

end NUMINAMATH_GPT_continuity_of_f_at_2_l918_91850


namespace NUMINAMATH_GPT_minimum_value_l918_91861

theorem minimum_value (x : ℝ) (h : x > 1) : 2 * x + 7 / (x - 1) ≥ 2 * Real.sqrt 14 + 2 := by
  sorry

end NUMINAMATH_GPT_minimum_value_l918_91861


namespace NUMINAMATH_GPT_part1_max_value_part2_three_distinct_real_roots_l918_91885

def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem part1_max_value (m : ℝ) (h_max : ∀ x, f x m ≤ f 2 m) : m = 6 := by
  sorry

theorem part2_three_distinct_real_roots (a : ℝ) (h_m : (m = 6))
  (h_a : ∀ x₁ x₂ x₃ : ℝ, f x₁ m = a ∧ f x₂ m = a ∧ f x₃ m = a →
     x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) : 0 < a ∧ a < 32 := by
  sorry

end NUMINAMATH_GPT_part1_max_value_part2_three_distinct_real_roots_l918_91885


namespace NUMINAMATH_GPT_arithmetic_expression_equiv_l918_91804

theorem arithmetic_expression_equiv :
  (-1:ℤ)^2009 * (-3) + 1 - 2^2 * 3 + (1 - 2^2) / 3 + (1 - 2 * 3)^2 = 16 := by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_equiv_l918_91804


namespace NUMINAMATH_GPT_vector_equation_proof_l918_91832

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B C P : V)

/-- The given condition. -/
def given_condition : Prop :=
  (P - A) + 2 • (P - B) + 3 • (P - C) = 0

/-- The target equality we want to prove. -/
theorem vector_equation_proof (h : given_condition A B C P) :
  P - A = (1 / 3 : ℝ) • (B - A) + (1 / 2 : ℝ) • (C - A) :=
sorry

end NUMINAMATH_GPT_vector_equation_proof_l918_91832


namespace NUMINAMATH_GPT_application_schemes_eq_l918_91820

noncomputable def number_of_application_schemes (graduates : ℕ) (universities : ℕ) : ℕ :=
  universities ^ graduates

theorem application_schemes_eq : 
  number_of_application_schemes 5 3 = 3 ^ 5 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_application_schemes_eq_l918_91820


namespace NUMINAMATH_GPT_number_of_women_in_preston_after_one_year_l918_91815

def preston_is_25_times_leesburg (preston leesburg : ℕ) : Prop := 
  preston = 25 * leesburg

def leesburg_population : ℕ := 58940

def women_percentage_leesburg : ℕ := 40

def women_percentage_preston : ℕ := 55

def growth_rate_leesburg : ℝ := 0.025

def growth_rate_preston : ℝ := 0.035

theorem number_of_women_in_preston_after_one_year : 
  ∀ (preston leesburg : ℕ), 
  preston_is_25_times_leesburg preston leesburg → 
  leesburg = 58940 → 
  (women_percentage_preston : ℝ) / 100 * (preston * (1 + growth_rate_preston) : ℝ) = 838788 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_women_in_preston_after_one_year_l918_91815


namespace NUMINAMATH_GPT_committee_formation_l918_91809

/-- Problem statement: In how many ways can a 5-person executive committee be formed if one of the 
members must be the president, given there are 30 members. --/
theorem committee_formation (n : ℕ) (k : ℕ) (h : n = 30) (h2 : k = 5) : 
  (n * Nat.choose (n - 1) (k - 1) = 712530 ) :=
by
  sorry

end NUMINAMATH_GPT_committee_formation_l918_91809


namespace NUMINAMATH_GPT_solve_for_x_l918_91859

theorem solve_for_x (x : ℝ) (h : 3 * x - 4 = -2 * x + 11) : x = 3 := 
sorry

end NUMINAMATH_GPT_solve_for_x_l918_91859


namespace NUMINAMATH_GPT_John_bought_new_socks_l918_91886

theorem John_bought_new_socks (initial_socks : ℕ) (thrown_away_socks : ℕ) (current_socks : ℕ) :
    initial_socks = 33 → thrown_away_socks = 19 → current_socks = 27 → 
    current_socks = (initial_socks - thrown_away_socks) + 13 :=
by
  sorry

end NUMINAMATH_GPT_John_bought_new_socks_l918_91886


namespace NUMINAMATH_GPT_percentage_silver_cars_after_shipment_l918_91817

-- Definitions for conditions
def initialCars : ℕ := 40
def initialSilverPerc : ℝ := 0.15
def newShipmentCars : ℕ := 80
def newShipmentNonSilverPerc : ℝ := 0.30

-- Proof statement that needs to be proven
theorem percentage_silver_cars_after_shipment :
  let initialSilverCars := initialSilverPerc * initialCars
  let newShipmentSilverPerc := 1 - newShipmentNonSilverPerc
  let newShipmentSilverCars := newShipmentSilverPerc * newShipmentCars
  let totalSilverCars := initialSilverCars + newShipmentSilverCars
  let totalCars := initialCars + newShipmentCars
  (totalSilverCars / totalCars) * 100 = 51.67 :=
by
  sorry

end NUMINAMATH_GPT_percentage_silver_cars_after_shipment_l918_91817


namespace NUMINAMATH_GPT_balloons_lost_l918_91827

theorem balloons_lost (initial remaining : ℕ) (h_initial : initial = 9) (h_remaining : remaining = 7) : initial - remaining = 2 := by
  sorry

end NUMINAMATH_GPT_balloons_lost_l918_91827


namespace NUMINAMATH_GPT_max_ounces_amber_can_get_l918_91852

theorem max_ounces_amber_can_get :
  let money := 7
  let candy_cost := 1
  let candy_ounces := 12
  let chips_cost := 1.40
  let chips_ounces := 17
  let max_ounces := max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces)
  max_ounces = 85 := 
by
  sorry

end NUMINAMATH_GPT_max_ounces_amber_can_get_l918_91852


namespace NUMINAMATH_GPT_three_digit_numbers_divisible_by_11_are_550_or_803_l918_91880

theorem three_digit_numbers_divisible_by_11_are_550_or_803 :
  ∀ (N : ℕ), (100 ≤ N ∧ N < 1000 ∧ ∃ (a b c : ℕ), N = 100 * a + 10 * b + c ∧ a ≠ 0 ∧ 11 ∣ N ∧ (N / 11 = a^2 + b^2 + c^2)) → (N = 550 ∨ N = 803) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_divisible_by_11_are_550_or_803_l918_91880


namespace NUMINAMATH_GPT_TrainTravelDays_l918_91839

-- Definition of the problem conditions
def train_start (days: ℕ) : ℕ := 
  if days = 0 then 0 -- no trains to meet on the first day
  else days -- otherwise, meet 'days' number of trains

/-- 
  Prove that if a train comes across 4 trains on its way from Amritsar to Bombay and starts at 9 am, 
  then it takes 5 days for the train to reach its destination.
-/
theorem TrainTravelDays (meet_train_count : ℕ) : meet_train_count = 4 → train_start (meet_train_count) + 1 = 5 :=
by
  intro h
  rw [h]
  sorry

end NUMINAMATH_GPT_TrainTravelDays_l918_91839


namespace NUMINAMATH_GPT_total_weight_of_balls_l918_91864

theorem total_weight_of_balls :
  let weight_blue := 6
  let weight_brown := 3.12
  let weight_green := 4.5
  weight_blue + weight_brown + weight_green = 13.62 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_balls_l918_91864


namespace NUMINAMATH_GPT_probability_fully_lit_l918_91870

-- define the conditions of the problem
def characters : List String := ["K", "y", "o", "t", "o", " ", "G", "r", "a", "n", "d", " ", "H", "o", "t", "e", "l"]

-- define the length of the sequence
def length_sequence : ℕ := characters.length

-- theorem stating the probability of seeing the fully lit sign
theorem probability_fully_lit : (1 / length_sequence) = 1 / 5 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_probability_fully_lit_l918_91870


namespace NUMINAMATH_GPT_find_XY_squared_l918_91846

variables {A B C T X Y : Type}

-- Conditions
variables (is_acute_scalene_triangle : ∀ A B C : Type, Prop) -- Assume scalene and acute properties
variable  (circumcircle : ∀ A B C : Type, Type) -- Circumcircle of the triangle
variable  (tangent_at : ∀ (ω : Type) B C, Type) -- Tangents at B and C
variables (BT CT : ℝ)
variables (BC : ℝ)
variables (projections : ∀ T (line : Type), Type)
variables (TX TY XY : ℝ)

-- Given conditions
axiom BT_value : BT = 18
axiom CT_value : CT = 18
axiom BC_value : BC = 24
axiom final_equation : TX^2 + TY^2 + XY^2 = 1552

-- Goal
theorem find_XY_squared : XY^2 = 884 := by
  sorry

end NUMINAMATH_GPT_find_XY_squared_l918_91846


namespace NUMINAMATH_GPT_sum_of_squares_of_solutions_l918_91873

theorem sum_of_squares_of_solutions :
  (∃ s₁ s₂ : ℝ, s₁ ≠ s₂ ∧ s₁ + s₂ = 17 ∧ s₁ * s₂ = 22) →
  ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 245 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_solutions_l918_91873


namespace NUMINAMATH_GPT_exponentiation_and_multiplication_of_fractions_l918_91882

-- Let's define the required fractions
def a : ℚ := 3 / 4
def b : ℚ := 1 / 5

-- Define the expected result
def expected_result : ℚ := 81 / 1280

-- State the theorem to prove
theorem exponentiation_and_multiplication_of_fractions : (a^4) * b = expected_result := by 
  sorry

end NUMINAMATH_GPT_exponentiation_and_multiplication_of_fractions_l918_91882


namespace NUMINAMATH_GPT_focus_of_parabola_y_eq_9x2_plus_6_l918_91871

noncomputable def focus_of_parabola (a b : ℝ) : ℝ × ℝ :=
  (0, b + (1 / (4 * a)))

theorem focus_of_parabola_y_eq_9x2_plus_6 :
  focus_of_parabola 9 6 = (0, 217 / 36) :=
by
  sorry

end NUMINAMATH_GPT_focus_of_parabola_y_eq_9x2_plus_6_l918_91871


namespace NUMINAMATH_GPT_roger_individual_pouches_per_pack_l918_91848

variable (members : ℕ) (coaches : ℕ) (helpers : ℕ) (packs : ℕ)

-- Given conditions
def total_people (members coaches helpers : ℕ) : ℕ := members + coaches + helpers
def pouches_per_pack (total_people packs : ℕ) : ℕ := total_people / packs

-- Specific values from the problem
def roger_total_people : ℕ := total_people 13 3 2
def roger_packs : ℕ := 3

-- The problem statement to prove:
theorem roger_individual_pouches_per_pack : pouches_per_pack roger_total_people roger_packs = 6 :=
by
  sorry

end NUMINAMATH_GPT_roger_individual_pouches_per_pack_l918_91848


namespace NUMINAMATH_GPT_range_of_a_l918_91829

theorem range_of_a (a b : ℝ) (h : a - 4 * Real.sqrt b = 2 * Real.sqrt (a - b)) : 
  a ∈ {x | 0 ≤ x} ∧ ((a = 0) ∨ (4 ≤ a ∧ a ≤ 20)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l918_91829


namespace NUMINAMATH_GPT_rhombus_area_l918_91866

theorem rhombus_area (R r : ℝ) : 
  ∃ A : ℝ, A = (8 * R^3 * r^3) / ((R^2 + r^2)^2) :=
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l918_91866


namespace NUMINAMATH_GPT_smaller_rectangle_area_l918_91868

-- Define the lengths and widths of the rectangles
def bigRectangleLength : ℕ := 40
def bigRectangleWidth : ℕ := 20
def smallRectangleLength : ℕ := bigRectangleLength / 2
def smallRectangleWidth : ℕ := bigRectangleWidth / 2

-- Define the area of the rectangles
def area (length width : ℕ) : ℕ := length * width

-- Prove the area of the smaller rectangle
theorem smaller_rectangle_area : area smallRectangleLength smallRectangleWidth = 200 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_smaller_rectangle_area_l918_91868


namespace NUMINAMATH_GPT_prime_sum_product_l918_91806

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 101) : p * q = 194 :=
sorry

end NUMINAMATH_GPT_prime_sum_product_l918_91806


namespace NUMINAMATH_GPT_inequality_always_holds_l918_91805

theorem inequality_always_holds (m : ℝ) : (-6 < m ∧ m ≤ 0) ↔ ∀ x : ℝ, 2 * m * x^2 + m * x - 3 / 4 < 0 := 
sorry

end NUMINAMATH_GPT_inequality_always_holds_l918_91805


namespace NUMINAMATH_GPT_exponentiation_addition_zero_l918_91853

theorem exponentiation_addition_zero : (-2)^(3^2) + 2^(3^2) = 0 := 
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_exponentiation_addition_zero_l918_91853


namespace NUMINAMATH_GPT_subset_proof_l918_91856

-- Define the set B
def B : Set ℝ := { x | x ≥ 0 }

-- Define the set A as the set {1, 2}
def A : Set ℝ := {1, 2}

-- The proof problem: Prove that A ⊆ B
theorem subset_proof : A ⊆ B := sorry

end NUMINAMATH_GPT_subset_proof_l918_91856


namespace NUMINAMATH_GPT_trig_expression_value_l918_91893

theorem trig_expression_value (α : ℝ) (h₁ : Real.tan (α + π / 4) = -1/2) (h₂ : π / 2 < α ∧ α < π) :
  (Real.sin (2 * α) - 2 * (Real.cos α)^2) / Real.sin (α - π / 4) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end NUMINAMATH_GPT_trig_expression_value_l918_91893


namespace NUMINAMATH_GPT_largest_common_factor_462_330_l918_91838

-- Define the factors of 462
def factors_462 : Set ℕ := {1, 2, 3, 6, 7, 14, 21, 33, 42, 66, 77, 154, 231, 462}

-- Define the factors of 330
def factors_330 : Set ℕ := {1, 2, 3, 5, 6, 10, 11, 15, 30, 33, 55, 66, 110, 165, 330}

-- Define the statement of the theorem
theorem largest_common_factor_462_330 : 
  (∀ d : ℕ, d ∈ (factors_462 ∩ factors_330) → d ≤ 66) ∧
  66 ∈ (factors_462 ∩ factors_330) :=
sorry

end NUMINAMATH_GPT_largest_common_factor_462_330_l918_91838


namespace NUMINAMATH_GPT_points_on_opposite_sides_of_line_l918_91835

theorem points_on_opposite_sides_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) < 0 ↔ -7 < a ∧ a < 24 :=
by 
  sorry

end NUMINAMATH_GPT_points_on_opposite_sides_of_line_l918_91835


namespace NUMINAMATH_GPT_x_coord_sum_l918_91898

noncomputable def sum_x_coordinates (x : ℕ) : Prop :=
  (0 ≤ x ∧ x < 20) ∧ (∃ y, y ≡ 7 * x + 3 [MOD 20] ∧ y ≡ 13 * x + 18 [MOD 20])

theorem x_coord_sum : ∃ (x : ℕ), sum_x_coordinates x ∧ x = 15 := by 
  sorry

end NUMINAMATH_GPT_x_coord_sum_l918_91898


namespace NUMINAMATH_GPT_find_rstu_l918_91826

theorem find_rstu (a x y c : ℝ) (r s t u : ℤ) (hc : a^10 * x * y - a^8 * y - a^7 * x = a^6 * (c^3 - 1)) :
  (a^r * x - a^s) * (a^t * y - a^u) = a^6 * c^3 ∧ r * s * t * u = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_rstu_l918_91826


namespace NUMINAMATH_GPT_freshman_to_sophomore_ratio_l918_91833

variable (f s : ℕ)

-- Define the participants from freshmen and sophomores
def freshmen_participants : ℕ := (3 * f) / 7
def sophomores_participants : ℕ := (2 * s) / 3

-- Theorem: There are 14/9 times as many freshmen as sophomores
theorem freshman_to_sophomore_ratio (h : freshmen_participants f = sophomores_participants s) : 
  9 * f = 14 * s :=
by
  sorry

end NUMINAMATH_GPT_freshman_to_sophomore_ratio_l918_91833
