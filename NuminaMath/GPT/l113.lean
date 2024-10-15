import Mathlib

namespace NUMINAMATH_GPT_divisible_by_6_and_sum_15_l113_11363

theorem divisible_by_6_and_sum_15 (A B : ℕ) (h1 : A + B = 15) (h2 : (10 * A + B) % 6 = 0) :
  (A * B = 56) ∨ (A * B = 54) :=
by sorry

end NUMINAMATH_GPT_divisible_by_6_and_sum_15_l113_11363


namespace NUMINAMATH_GPT_tens_digit_seven_last_digit_six_l113_11366

theorem tens_digit_seven_last_digit_six (n : ℕ) (h : ((n * n) / 10) % 10 = 7) :
  (n * n) % 10 = 6 :=
sorry

end NUMINAMATH_GPT_tens_digit_seven_last_digit_six_l113_11366


namespace NUMINAMATH_GPT_value_of_f_of_x_minus_3_l113_11360

theorem value_of_f_of_x_minus_3 (x : ℝ) (f : ℝ → ℝ) (h : ∀ y : ℝ, f y = y^2) : f (x - 3) = x^2 - 6*x + 9 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_of_x_minus_3_l113_11360


namespace NUMINAMATH_GPT_geometric_sequence_a2_value_l113_11380

theorem geometric_sequence_a2_value
  (a : ℕ → ℝ)
  (a1 a2 a3 : ℝ)
  (h1 : a 1 = a1)
  (h2 : a 2 = a2)
  (h3 : a 3 = a3)
  (h_pos : ∀ n, 0 < a n)
  (h_geo : ∀ n, a (n + 1) = a 1 * (a 2) ^ n)
  (h_sum : a1 + a2 + a3 = 18)
  (h_inverse_sum : 1/a1 + 1/a2 + 1/a3 = 2)
  : a2 = 3 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a2_value_l113_11380


namespace NUMINAMATH_GPT_zoo_problem_l113_11345

theorem zoo_problem :
  let parrots := 8
  let snakes := 3 * parrots
  let monkeys := 2 * snakes
  let elephants := (parrots + snakes) / 2
  let zebras := monkeys - 35
  elephants - zebras = 3 :=
by
  sorry

end NUMINAMATH_GPT_zoo_problem_l113_11345


namespace NUMINAMATH_GPT_binary_1011_is_11_decimal_124_is_174_l113_11338

-- Define the conversion from binary to decimal
def binaryToDecimal (n : Nat) : Nat :=
  (n % 10) * 2^0 + ((n / 10) % 10) * 2^1 + ((n / 100) % 10) * 2^2 + ((n / 1000) % 10) * 2^3

-- Define the conversion from decimal to octal through division and remainder
noncomputable def decimalToOctal (n : Nat) : String := 
  let rec aux (n : Nat) (acc : List Nat) : List Nat :=
    if n = 0 then acc else aux (n / 8) ((n % 8) :: acc)
  (aux n []).foldr (fun d s => s ++ d.repr) ""

-- Prove that the binary number 1011 (base 2) equals the decimal number 11
theorem binary_1011_is_11 : binaryToDecimal 1011 = 11 := by
  sorry

-- Prove that the decimal number 124 equals the octal number 174 (base 8)
theorem decimal_124_is_174 : decimalToOctal 124 = "174" := by
  sorry

end NUMINAMATH_GPT_binary_1011_is_11_decimal_124_is_174_l113_11338


namespace NUMINAMATH_GPT_square_of_binomial_l113_11305

theorem square_of_binomial (c : ℝ) (h : c = 3600) :
  ∃ a : ℝ, (x : ℝ) → (x + a)^2 = x^2 + 120 * x + c := by
  sorry

end NUMINAMATH_GPT_square_of_binomial_l113_11305


namespace NUMINAMATH_GPT_distinct_students_27_l113_11357

variable (students_euler : ℕ) (students_fibonacci : ℕ) (students_gauss : ℕ) (overlap_euler_fibonacci : ℕ)

-- Conditions
def conditions : Prop := 
  students_euler = 12 ∧ 
  students_fibonacci = 10 ∧ 
  students_gauss = 11 ∧ 
  overlap_euler_fibonacci = 3

-- Question and correct answer
def distinct_students (students_euler students_fibonacci students_gauss overlap_euler_fibonacci : ℕ) : ℕ :=
  (students_euler + students_fibonacci + students_gauss) - overlap_euler_fibonacci

theorem distinct_students_27 : conditions students_euler students_fibonacci students_gauss overlap_euler_fibonacci →
  distinct_students students_euler students_fibonacci students_gauss overlap_euler_fibonacci = 27 :=
by
  sorry

end NUMINAMATH_GPT_distinct_students_27_l113_11357


namespace NUMINAMATH_GPT_chef_earns_2_60_less_l113_11373

/--
At Joe's Steakhouse, the hourly wage for a chef is 20% greater than that of a dishwasher,
and the hourly wage of a dishwasher is half as much as the hourly wage of a manager.
If a manager's wage is $6.50 per hour, prove that a chef earns $2.60 less per hour than a manager.
-/
theorem chef_earns_2_60_less {w_manager w_dishwasher w_chef : ℝ} 
  (h1 : w_dishwasher = w_manager / 2)
  (h2 : w_chef = w_dishwasher * 1.20)
  (h3 : w_manager = 6.50) :
  w_manager - w_chef = 2.60 :=
by
  sorry

end NUMINAMATH_GPT_chef_earns_2_60_less_l113_11373


namespace NUMINAMATH_GPT_avg_cost_equals_0_22_l113_11336

-- Definitions based on conditions
def num_pencils : ℕ := 150
def cost_pencils : ℝ := 24.75
def shipping_cost : ℝ := 8.50

-- Calculating total cost and average cost
noncomputable def total_cost : ℝ := cost_pencils + shipping_cost
noncomputable def avg_cost_per_pencil : ℝ := total_cost / num_pencils

-- Lean theorem statement
theorem avg_cost_equals_0_22 : avg_cost_per_pencil = 0.22 :=
by
  sorry

end NUMINAMATH_GPT_avg_cost_equals_0_22_l113_11336


namespace NUMINAMATH_GPT_remainder_proof_l113_11333

theorem remainder_proof : 1234567 % 12 = 7 := sorry

end NUMINAMATH_GPT_remainder_proof_l113_11333


namespace NUMINAMATH_GPT_proof_l113_11324

def statement : Prop :=
  ∀ (a : ℝ),
    (¬ (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 - 3 * a - x + 1 > 0) ∧
    ¬ (a^2 - 4 ≥ 0 ∧
    (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 - 3 * a - x + 1 > 0)))
    → (1 ≤ a ∧ a < 2)

theorem proof : statement :=
by
  sorry

end NUMINAMATH_GPT_proof_l113_11324


namespace NUMINAMATH_GPT_total_videos_watched_l113_11326

variable (Ekon Uma Kelsey : ℕ)

theorem total_videos_watched
  (hKelsey : Kelsey = 160)
  (hKelsey_Ekon : Kelsey = Ekon + 43)
  (hEkon_Uma : Ekon = Uma - 17) :
  Kelsey + Ekon + Uma = 411 := by
  sorry

end NUMINAMATH_GPT_total_videos_watched_l113_11326


namespace NUMINAMATH_GPT_range_of_a_l113_11364

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x > a
def q (x : ℝ) : Prop := x^2 + x - 2 > 0

theorem range_of_a 
  (h_sufficient : ∀ x, p x a → q x)
  (h_not_necessary : ∃ x, q x ∧ ¬ p x a) :
  a ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l113_11364


namespace NUMINAMATH_GPT_area_smallest_region_enclosed_l113_11303

theorem area_smallest_region_enclosed {x y : ℝ} (circle_eq : x^2 + y^2 = 9) (abs_line_eq : y = |x|) :
  ∃ area, area = (9 * Real.pi) / 4 :=
by
  sorry

end NUMINAMATH_GPT_area_smallest_region_enclosed_l113_11303


namespace NUMINAMATH_GPT_transformed_graph_passes_point_l113_11369

theorem transformed_graph_passes_point (f : ℝ → ℝ) 
  (h₁ : f 1 = 3) :
  f (-1) + 1 = 4 :=
by
  sorry

end NUMINAMATH_GPT_transformed_graph_passes_point_l113_11369


namespace NUMINAMATH_GPT_option_A_option_C_option_D_l113_11390

noncomputable def ratio_12_11 := (12 : ℝ) / 11
noncomputable def ratio_11_10 := (11 : ℝ) / 10

theorem option_A : ratio_12_11^11 > ratio_11_10^10 := sorry

theorem option_C : ratio_12_11^10 > ratio_11_10^9 := sorry

theorem option_D : ratio_11_10^12 > ratio_12_11^13 := sorry

end NUMINAMATH_GPT_option_A_option_C_option_D_l113_11390


namespace NUMINAMATH_GPT_midpoint_x_sum_l113_11378

variable {p q r s : ℝ}

theorem midpoint_x_sum (h : p + q + r + s = 20) :
  ((p + q) / 2 + (q + r) / 2 + (r + s) / 2 + (s + p) / 2) = 20 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_x_sum_l113_11378


namespace NUMINAMATH_GPT_asymptote_sum_l113_11397

noncomputable def f (x : ℝ) : ℝ := (x^3 + 4*x^2 + 3*x) / (x^3 + x^2 - 2*x)

def holes := 0 -- a
def vertical_asymptotes := 2 -- b
def horizontal_asymptotes := 1 -- c
def oblique_asymptotes := 0 -- d

theorem asymptote_sum : holes + 2 * vertical_asymptotes + 3 * horizontal_asymptotes + 4 * oblique_asymptotes = 7 :=
by
  unfold holes vertical_asymptotes horizontal_asymptotes oblique_asymptotes
  norm_num

end NUMINAMATH_GPT_asymptote_sum_l113_11397


namespace NUMINAMATH_GPT_negation_example_l113_11310

variable {I : Set ℝ}

theorem negation_example (h : ∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) : ¬(∀ x ∈ I, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x ∈ I, x^3 - x^2 + 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l113_11310


namespace NUMINAMATH_GPT_f_zero_eq_one_f_pos_all_f_increasing_l113_11308

noncomputable def f : ℝ → ℝ := sorry

axiom f_nonzero : f 0 ≠ 0
axiom f_pos : ∀ x, 0 < x → 1 < f x
axiom f_mul : ∀ a b : ℝ, f (a + b) = f a * f b

theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem f_pos_all : ∀ x : ℝ, 0 < f x :=
sorry

theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ :=
sorry

end NUMINAMATH_GPT_f_zero_eq_one_f_pos_all_f_increasing_l113_11308


namespace NUMINAMATH_GPT_solve_for_m_l113_11384

open Real

theorem solve_for_m (a b m : ℝ)
  (h1 : (1/2)^a = m)
  (h2 : 3^b = m)
  (h3 : 1/a - 1/b = 2) :
  m = sqrt 6 / 6 := 
  sorry

end NUMINAMATH_GPT_solve_for_m_l113_11384


namespace NUMINAMATH_GPT_ordered_pair_solution_l113_11351

theorem ordered_pair_solution :
  ∃ (x y : ℚ), 
  (3 * x - 2 * y = (6 - 2 * x) + (6 - 2 * y)) ∧
  (x + 3 * y = (2 * x + 1) - (2 * y + 1)) ∧
  x = 12 / 5 ∧
  y = 12 / 25 :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_solution_l113_11351


namespace NUMINAMATH_GPT_simplify_product_l113_11304

theorem simplify_product : 
  18 * (8 / 15) * (2 / 27) = 32 / 45 :=
by
  sorry

end NUMINAMATH_GPT_simplify_product_l113_11304


namespace NUMINAMATH_GPT_correct_operation_l113_11388

theorem correct_operation (x : ℝ) : (2 * x ^ 3) ^ 2 = 4 * x ^ 6 := 
  sorry

end NUMINAMATH_GPT_correct_operation_l113_11388


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l113_11358

theorem quadratic_inequality_solution (x : ℝ) : 3 * x^2 - 5 * x - 8 > 0 ↔ x < -4/3 ∨ x > 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l113_11358


namespace NUMINAMATH_GPT_second_candidate_more_marks_30_l113_11309

noncomputable def total_marks : ℝ := 600
def passing_marks_approx : ℝ := 240

def candidate_marks (percentage : ℝ) (total : ℝ) : ℝ :=
  percentage * total

def more_marks (second_candidate : ℝ) (passing : ℝ) : ℝ :=
  second_candidate - passing

theorem second_candidate_more_marks_30 :
  more_marks (candidate_marks 0.45 total_marks) passing_marks_approx = 30 := by
  sorry

end NUMINAMATH_GPT_second_candidate_more_marks_30_l113_11309


namespace NUMINAMATH_GPT_tan_double_angle_l113_11314

theorem tan_double_angle (theta : ℝ) (h : 2 * Real.sin theta + Real.cos theta = 0) :
  Real.tan (2 * theta) = - 4 / 3 :=
sorry

end NUMINAMATH_GPT_tan_double_angle_l113_11314


namespace NUMINAMATH_GPT_count_of_integer_values_not_satisfying_inequality_l113_11334

theorem count_of_integer_values_not_satisfying_inequality :
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℤ, (3 * x^2 + 11 * x + 10 ≤ 17) ↔ (x = -7 ∨ x = -6 ∨ x = -5 ∨ x = -4 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0) :=
by sorry

end NUMINAMATH_GPT_count_of_integer_values_not_satisfying_inequality_l113_11334


namespace NUMINAMATH_GPT_statistical_hypothesis_independence_l113_11362

def independence_test_statistical_hypothesis (A B: Prop) (independence_test: Prop) : Prop :=
  (independence_test ∧ A ∧ B) → (A = B)

theorem statistical_hypothesis_independence (A B: Prop) (independence_test: Prop) :
  (independence_test ∧ A ∧ B) → (A = B) :=
by
  sorry

end NUMINAMATH_GPT_statistical_hypothesis_independence_l113_11362


namespace NUMINAMATH_GPT_students_not_like_any_l113_11396

variables (F B P T F_cap_B F_cap_P F_cap_T B_cap_P B_cap_T P_cap_T F_cap_B_cap_P_cap_T : ℕ)

def total_students := 30

def students_like_F := 18
def students_like_B := 12
def students_like_P := 14
def students_like_T := 10

def students_like_F_and_B := 8
def students_like_F_and_P := 6
def students_like_F_and_T := 4
def students_like_B_and_P := 5
def students_like_B_and_T := 3
def students_like_P_and_T := 7

def students_like_all_four := 2

theorem students_not_like_any :
  total_students - ((students_like_F + students_like_B + students_like_P + students_like_T)
                    - (students_like_F_and_B + students_like_F_and_P + students_like_F_and_T
                      + students_like_B_and_P + students_like_B_and_T + students_like_P_and_T)
                    + students_like_all_four) = 11 :=
by sorry

end NUMINAMATH_GPT_students_not_like_any_l113_11396


namespace NUMINAMATH_GPT_domain_of_f_l113_11347

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 - 5 * x + 6)

theorem domain_of_f :
  {x : ℝ | x^2 - 5 * x + 6 ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l113_11347


namespace NUMINAMATH_GPT_min_value_expression_l113_11301

theorem min_value_expression :
  ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ (5 * Real.sqrt 6) / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l113_11301


namespace NUMINAMATH_GPT_nathaniel_initial_tickets_l113_11399

theorem nathaniel_initial_tickets (a b c : ℕ) (h1 : a = 2) (h2 : b = 4) (h3 : c = 3) :
  a * b + c = 11 :=
by
  sorry

end NUMINAMATH_GPT_nathaniel_initial_tickets_l113_11399


namespace NUMINAMATH_GPT_percentage_of_first_solution_l113_11371

theorem percentage_of_first_solution (P : ℕ) 
  (h1 : 28 * P / 100 + 12 * 80 / 100 = 40 * 45 / 100) : 
  P = 30 :=
sorry

end NUMINAMATH_GPT_percentage_of_first_solution_l113_11371


namespace NUMINAMATH_GPT_find_parallel_line_l113_11398

-- Definition of the point (0, 1)
def point : ℝ × ℝ := (0, 1)

-- Definition of the original line equation
def original_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Definition of the desired line equation
def desired_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement: defining the desired line based on the point and parallelism condition
theorem find_parallel_line (x y : ℝ) (hx : point.fst = 0) (hy : point.snd = 1) :
  ∃ m : ℝ, (2 * x + y + m = 0) ∧ (2 * 0 + 1 + m = 0) → desired_line x y :=
sorry

end NUMINAMATH_GPT_find_parallel_line_l113_11398


namespace NUMINAMATH_GPT_lilly_can_buy_flowers_l113_11387

-- Define variables
def days_until_birthday : ℕ := 22
def daily_savings : ℕ := 2
def flower_cost : ℕ := 4

-- Statement: Given the conditions, prove the number of flowers Lilly can buy.
theorem lilly_can_buy_flowers :
  (days_until_birthday * daily_savings) / flower_cost = 11 := 
by
  -- proof steps
  sorry

end NUMINAMATH_GPT_lilly_can_buy_flowers_l113_11387


namespace NUMINAMATH_GPT_prove_p_and_q_l113_11300

def p (m : ℝ) : Prop :=
  (∀ x : ℝ, x^2 + x + m > 0) → m > 1 / 4

def q (A B : ℝ) : Prop :=
  A > B ↔ Real.sin A > Real.sin B

theorem prove_p_and_q :
  (∀ m : ℝ, p m) ∧ (∀ A B : ℝ, q A B) :=
by
  sorry

end NUMINAMATH_GPT_prove_p_and_q_l113_11300


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l113_11323

theorem sqrt_meaningful_range (x : ℝ) : x + 2 ≥ 0 → x ≥ -2 :=
by 
  intro h
  linarith [h]

end NUMINAMATH_GPT_sqrt_meaningful_range_l113_11323


namespace NUMINAMATH_GPT_no_real_roots_of_f_l113_11329

def f (x : ℝ) : ℝ := (x + 1) * |x + 1| - x * |x| + 1

theorem no_real_roots_of_f :
  ∀ x : ℝ, f x ≠ 0 := by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_f_l113_11329


namespace NUMINAMATH_GPT_find_analytical_expression_of_f_l113_11302

variable (f : ℝ → ℝ)

theorem find_analytical_expression_of_f
  (h : ∀ x : ℝ, f (2 * x + 1) = 4 * x^2 + 4 * x) :
  ∀ x : ℝ, f x = x^2 - 1 :=
sorry

end NUMINAMATH_GPT_find_analytical_expression_of_f_l113_11302


namespace NUMINAMATH_GPT_total_prep_time_is_8_l113_11321

-- Defining the conditions
def prep_vocab_sentence_eq := 3
def prep_analytical_writing := 2
def prep_quantitative_reasoning := 3

-- Stating the total preparation time
def total_prep_time := prep_vocab_sentence_eq + prep_analytical_writing + prep_quantitative_reasoning

-- The Lean statement of the mathematical proof problem
theorem total_prep_time_is_8 : total_prep_time = 8 := by
  sorry

end NUMINAMATH_GPT_total_prep_time_is_8_l113_11321


namespace NUMINAMATH_GPT_math_problem_l113_11361

theorem math_problem (n a b : ℕ) (hn_pos : n > 0) (h1 : 3 * n + 1 = a^2) (h2 : 5 * n - 1 = b^2) :
  (∃ x y: ℕ, 7 * n + 13 = x * y ∧ 1 < x ∧ 1 < y) ∧
  (∃ p q: ℕ, 8 * (17 * n^2 + 3 * n) = p^2 + q^2) :=
  sorry

end NUMINAMATH_GPT_math_problem_l113_11361


namespace NUMINAMATH_GPT_total_coin_tosses_l113_11320

variable (heads : ℕ) (tails : ℕ)

theorem total_coin_tosses (h_head : heads = 9) (h_tail : tails = 5) : heads + tails = 14 := by
  sorry

end NUMINAMATH_GPT_total_coin_tosses_l113_11320


namespace NUMINAMATH_GPT_polynomial_characterization_l113_11343

theorem polynomial_characterization (P : ℝ → ℝ) :
  (∀ a b c : ℝ, ab + bc + ca = 0 → P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c)) →
  ∃ (α β : ℝ), ∀ x : ℝ, P x = α * x^4 + β * x^2 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_characterization_l113_11343


namespace NUMINAMATH_GPT_brianna_more_chocolates_than_alix_l113_11311

def Nick_ClosetA : ℕ := 10
def Nick_ClosetB : ℕ := 6
def Alix_ClosetA : ℕ := 3 * Nick_ClosetA
def Alix_ClosetB : ℕ := 3 * Nick_ClosetA
def Mom_Takes_From_AlixA : ℚ := (1/4:ℚ) * Alix_ClosetA
def Brianna_ClosetA : ℚ := 2 * (Nick_ClosetA + Alix_ClosetA - Mom_Takes_From_AlixA)
def Brianna_ClosetB_after : ℕ := 18
def Brianna_ClosetB : ℚ := Brianna_ClosetB_after / (0.8:ℚ)

def Brianna_Total : ℚ := Brianna_ClosetA + Brianna_ClosetB
def Alix_Total : ℚ := Alix_ClosetA + Alix_ClosetB
def Difference : ℚ := Brianna_Total - Alix_Total

theorem brianna_more_chocolates_than_alix : Difference = 35 := by
  sorry

end NUMINAMATH_GPT_brianna_more_chocolates_than_alix_l113_11311


namespace NUMINAMATH_GPT_f_six_equals_twenty_two_l113_11367

-- Definitions as per conditions
variable (n : ℕ) (f : ℕ → ℕ)

-- Conditions of the problem
-- n is a natural number greater than or equal to 3
-- f(n) satisfies the properties defined in the given solution
axiom f_base : f 1 = 2
axiom f_recursion {k : ℕ} (hk : k ≥ 1) : f (k + 1) = f k + (k + 1)

-- Goal to prove
theorem f_six_equals_twenty_two : f 6 = 22 := sorry

end NUMINAMATH_GPT_f_six_equals_twenty_two_l113_11367


namespace NUMINAMATH_GPT_lcm_36_105_l113_11372

noncomputable def factorize_36 : List (ℕ × ℕ) := [(2, 2), (3, 2)]
noncomputable def factorize_105 : List (ℕ × ℕ) := [(3, 1), (5, 1), (7, 1)]

theorem lcm_36_105 : Nat.lcm 36 105 = 1260 :=
by
  have h_36 : 36 = 2^2 * 3^2 := by norm_num
  have h_105 : 105 = 3^1 * 5^1 * 7^1 := by norm_num
  sorry

end NUMINAMATH_GPT_lcm_36_105_l113_11372


namespace NUMINAMATH_GPT_unique_root_range_l113_11381

theorem unique_root_range (a : ℝ) :
  (x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → (∃! x : ℝ, x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → - (Real.sqrt 3) / 2 < a ∧ a < (Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_root_range_l113_11381


namespace NUMINAMATH_GPT_determine_y_l113_11328

theorem determine_y (y : ℕ) : (8^5 + 8^5 + 2 * 8^5 = 2^y) → y = 17 := 
by {
  sorry
}

end NUMINAMATH_GPT_determine_y_l113_11328


namespace NUMINAMATH_GPT_triangle_area_is_18_l113_11318

noncomputable def area_triangle : ℝ :=
  let vertices : List (ℝ × ℝ) := [(1, 2), (7, 6), (1, 8)]
  let base := (8 - 2) -- Length between (1, 2) and (1, 8)
  let height := (7 - 1) -- Perpendicular distance from (7, 6) to x = 1
  (1 / 2) * base * height

theorem triangle_area_is_18 : area_triangle = 18 := by
  sorry

end NUMINAMATH_GPT_triangle_area_is_18_l113_11318


namespace NUMINAMATH_GPT_john_total_payment_l113_11353

-- Definitions of the conditions
def yearly_cost_first_8_years : ℕ := 10000
def yearly_cost_9_to_18_years : ℕ := 2 * yearly_cost_first_8_years
def university_tuition : ℕ := 250000
def total_cost := (8 * yearly_cost_first_8_years) + (10 * yearly_cost_9_to_18_years) + university_tuition

-- John pays half of the total cost
def johns_total_cost := total_cost / 2

-- Theorem stating the total cost John pays
theorem john_total_payment : johns_total_cost = 265000 := by
  sorry

end NUMINAMATH_GPT_john_total_payment_l113_11353


namespace NUMINAMATH_GPT_possible_shapes_l113_11312

def is_valid_shapes (T S C : ℕ) : Prop :=
  T + S + C = 24 ∧ T = 7 * S

theorem possible_shapes :
  ∃ (T S C : ℕ), is_valid_shapes T S C ∧ 
    (T = 0 ∧ S = 0 ∧ C = 24) ∨
    (T = 7 ∧ S = 1 ∧ C = 16) ∨
    (T = 14 ∧ S = 2 ∧ C = 8) ∨
    (T = 21 ∧ S = 3 ∧ C = 0) :=
by
  sorry

end NUMINAMATH_GPT_possible_shapes_l113_11312


namespace NUMINAMATH_GPT_remaining_payment_l113_11306

theorem remaining_payment (deposit_percent : ℝ) (deposit_amount : ℝ) (total_percent : ℝ) (total_price : ℝ) :
  deposit_percent = 5 ∧ deposit_amount = 50 ∧ total_percent = 100 → total_price - deposit_amount = 950 :=
by {
  sorry
}

end NUMINAMATH_GPT_remaining_payment_l113_11306


namespace NUMINAMATH_GPT_find_remainder_l113_11315

-- Define the numbers
def a := 98134
def b := 98135
def c := 98136
def d := 98137
def e := 98138
def f := 98139

-- Theorem statement
theorem find_remainder :
  (a + b + c + d + e + f) % 9 = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_remainder_l113_11315


namespace NUMINAMATH_GPT_star_operation_example_l113_11354

-- Define the operation ☆
def star (a b : ℚ) : ℚ := a - b + 1

-- The theorem to prove
theorem star_operation_example : star (star 2 3) 2 = -1 := by
  sorry

end NUMINAMATH_GPT_star_operation_example_l113_11354


namespace NUMINAMATH_GPT_count_three_digit_concave_numbers_l113_11356

def is_concave_number (a b c : ℕ) : Prop :=
  a > b ∧ c > b

theorem count_three_digit_concave_numbers : 
  (∃! n : ℕ, n = 240) := by
  sorry

end NUMINAMATH_GPT_count_three_digit_concave_numbers_l113_11356


namespace NUMINAMATH_GPT_folder_cost_l113_11342

theorem folder_cost (cost_pens : ℕ) (cost_notebooks : ℕ) (total_spent : ℕ) (folders : ℕ) :
  cost_pens = 3 → cost_notebooks = 12 → total_spent = 25 → folders = 2 →
  ∃ (cost_per_folder : ℕ), cost_per_folder = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_folder_cost_l113_11342


namespace NUMINAMATH_GPT_find_honeydews_left_l113_11350

theorem find_honeydews_left 
  (cantaloupe_price : ℕ)
  (honeydew_price : ℕ)
  (initial_cantaloupes : ℕ)
  (initial_honeydews : ℕ)
  (dropped_cantaloupes : ℕ)
  (rotten_honeydews : ℕ)
  (end_cantaloupes : ℕ)
  (total_revenue : ℕ)
  (honeydews_left : ℕ) :
  cantaloupe_price = 2 →
  honeydew_price = 3 →
  initial_cantaloupes = 30 →
  initial_honeydews = 27 →
  dropped_cantaloupes = 2 →
  rotten_honeydews = 3 →
  end_cantaloupes = 8 →
  total_revenue = 85 →
  honeydews_left = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_honeydews_left_l113_11350


namespace NUMINAMATH_GPT_external_tangent_inequality_l113_11385

variable (x y z : ℝ)
variable (a b c T : ℝ)

-- Definitions based on conditions
def a_def : a = x + y := sorry
def b_def : b = y + z := sorry
def c_def : c = z + x := sorry
def T_def : T = π * x^2 + π * y^2 + π * z^2 := sorry

-- The theorem to prove
theorem external_tangent_inequality
    (a_def : a = x + y) 
    (b_def : b = y + z) 
    (c_def : c = z + x) 
    (T_def : T = π * x^2 + π * y^2 + π * z^2) : 
    π * (a + b + c) ^ 2 ≤ 12 * T := 
sorry

end NUMINAMATH_GPT_external_tangent_inequality_l113_11385


namespace NUMINAMATH_GPT_red_grapes_count_l113_11383

theorem red_grapes_count (G : ℕ) (total_fruit : ℕ) (red_grapes : ℕ) (raspberries : ℕ)
  (h1 : red_grapes = 3 * G + 7) 
  (h2 : raspberries = G - 5) 
  (h3 : total_fruit = G + red_grapes + raspberries) 
  (h4 : total_fruit = 102) : 
  red_grapes = 67 :=
by
  sorry

end NUMINAMATH_GPT_red_grapes_count_l113_11383


namespace NUMINAMATH_GPT_largest_angle_in_triangle_l113_11331

theorem largest_angle_in_triangle 
  (A B C : ℝ)
  (h_sum_angles: 2 * A + 20 = 105)
  (h_triangle_sum: A + (A + 20) + C = 180)
  (h_A_ge_0: A ≥ 0)
  (h_B_ge_0: B ≥ 0)
  (h_C_ge_0: C ≥ 0) : 
  max A (max (A + 20) C) = 75 := 
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_largest_angle_in_triangle_l113_11331


namespace NUMINAMATH_GPT_total_coins_l113_11376

theorem total_coins (x y : ℕ) (h : x ≠ y) (h1 : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := by
  sorry

end NUMINAMATH_GPT_total_coins_l113_11376


namespace NUMINAMATH_GPT_x_plus_y_equals_22_l113_11313

theorem x_plus_y_equals_22 (x y : ℕ) (h1 : 2^x = 4^(y + 2)) (h2 : 27^y = 9^(x - 7)) : x + y = 22 := 
sorry

end NUMINAMATH_GPT_x_plus_y_equals_22_l113_11313


namespace NUMINAMATH_GPT_value_of_k_h_5_l113_11325

def h (x : ℝ) : ℝ := 4 * x + 6
def k (x : ℝ) : ℝ := 6 * x - 8

theorem value_of_k_h_5 : k (h 5) = 148 :=
by
  have h5 : h 5 = 4 * 5 + 6 := rfl
  simp [h5, h, k]
  sorry

end NUMINAMATH_GPT_value_of_k_h_5_l113_11325


namespace NUMINAMATH_GPT_ratio_of_x_y_l113_11370

theorem ratio_of_x_y (x y : ℚ) (h : (3 * x - 2 * y) / (2 * x + 3 * y + 1) = 4 / 5) : x / y = 22 / 7 :=
sorry

end NUMINAMATH_GPT_ratio_of_x_y_l113_11370


namespace NUMINAMATH_GPT_original_faculty_members_l113_11392

theorem original_faculty_members
  (x : ℝ) (h : 0.87 * x = 195) : x = 224 := sorry

end NUMINAMATH_GPT_original_faculty_members_l113_11392


namespace NUMINAMATH_GPT_largest_fraction_l113_11389

theorem largest_fraction (p q r s : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) :
  (∃ (x : ℝ), x = (r + s) / (p + q) ∧ 
  (x > (p + s) / (q + r)) ∧ 
  (x > (p + q) / (r + s)) ∧ 
  (x > (q + r) / (p + s)) ∧ 
  (x > (q + s) / (p + r))) :=
sorry

end NUMINAMATH_GPT_largest_fraction_l113_11389


namespace NUMINAMATH_GPT_ball_returns_to_bob_after_13_throws_l113_11375

theorem ball_returns_to_bob_after_13_throws:
  ∃ n : ℕ, n = 13 ∧ (∀ k, k < 13 → (1 + 3 * k) % 13 = 0) :=
sorry

end NUMINAMATH_GPT_ball_returns_to_bob_after_13_throws_l113_11375


namespace NUMINAMATH_GPT_boat_speed_still_water_l113_11322

def effective_upstream_speed (b c : ℝ) : ℝ := b - c
def effective_downstream_speed (b c : ℝ) : ℝ := b + c

theorem boat_speed_still_water :
  ∃ b c : ℝ, effective_upstream_speed b c = 9 ∧ effective_downstream_speed b c = 15 ∧ b = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_boat_speed_still_water_l113_11322


namespace NUMINAMATH_GPT_part1_part2_l113_11346

-- (1) Prove that if 2 ∈ M and M is the solution set of ax^2 + 5x - 2 > 0, then a > -2.
theorem part1 (a : ℝ) (h : 2 * (a * 4 + 10) - 2 > 0) : a > -2 :=
sorry

-- (2) Given M = {x | 1/2 < x < 2} and M is the solution set of ax^2 + 5x - 2 > 0,
-- prove that the solution set of ax^2 - 5x + a^2 - 1 > 0 is -3 < x < 1/2
theorem part2 (a : ℝ) (h1 : ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ ax^2 + 5*x - 2 > 0) (h2 : a = -2) :
  ∀ x : ℝ, (-3 < x ∧ x < 1/2) ↔ (-2 * x^2 - 5 * x + 3 > 0) :=
sorry

end NUMINAMATH_GPT_part1_part2_l113_11346


namespace NUMINAMATH_GPT_sufficient_condition_for_solution_l113_11339

theorem sufficient_condition_for_solution 
  (a : ℝ) (f g h : ℝ → ℝ) (h_a : 1 < a)
  (h_fg_h : ∀ x : ℝ, 0 ≤ f x + g x + h x) 
  (h_common_root : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) : 
  ∃ x : ℝ, a^(f x) + a^(g x) + a^(h x) = 3 := 
by
  sorry

end NUMINAMATH_GPT_sufficient_condition_for_solution_l113_11339


namespace NUMINAMATH_GPT_ratio_ab_l113_11330

variable (x y a b : ℝ)
variable (h1 : 4 * x - 2 * y = a)
variable (h2 : 6 * y - 12 * x = b)
variable (h3 : b ≠ 0)

theorem ratio_ab : 4 * x - 2 * y = a ∧ 6 * y - 12 * x = b ∧ b ≠ 0 → a / b = -1 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_ab_l113_11330


namespace NUMINAMATH_GPT_scientific_notation_9600000_l113_11344

theorem scientific_notation_9600000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 9600000 = a * 10 ^ n ∧ a = 9.6 ∧ n = 6 :=
by
  exists 9.6
  exists 6
  simp
  sorry

end NUMINAMATH_GPT_scientific_notation_9600000_l113_11344


namespace NUMINAMATH_GPT_real_nums_inequality_l113_11307

theorem real_nums_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a ^ 2000 + b ^ 2000 = a ^ 1998 + b ^ 1998) :
  a ^ 2 + b ^ 2 ≤ 2 :=
sorry

end NUMINAMATH_GPT_real_nums_inequality_l113_11307


namespace NUMINAMATH_GPT_albert_age_l113_11349

theorem albert_age
  (A : ℕ)
  (dad_age : ℕ)
  (h1 : dad_age = 48)
  (h2 : dad_age - 4 = 4 * (A - 4)) :
  A = 15 :=
by
  sorry

end NUMINAMATH_GPT_albert_age_l113_11349


namespace NUMINAMATH_GPT_cubic_roots_expression_l113_11377

theorem cubic_roots_expression (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a * b + a * c + b * c = -1) (h3 : a * b * c = 2) :
  2 * a * (b - c) ^ 2 + 2 * b * (c - a) ^ 2 + 2 * c * (a - b) ^ 2 = -36 :=
by
  sorry

end NUMINAMATH_GPT_cubic_roots_expression_l113_11377


namespace NUMINAMATH_GPT_eight_hash_four_eq_ten_l113_11386

def operation (a b : ℚ) : ℚ := a + a / b

theorem eight_hash_four_eq_ten : operation 8 4 = 10 :=
by
  sorry

end NUMINAMATH_GPT_eight_hash_four_eq_ten_l113_11386


namespace NUMINAMATH_GPT_percentage_of_students_wearing_blue_shirts_l113_11332

theorem percentage_of_students_wearing_blue_shirts :
  ∀ (total_students red_percent green_percent students_other_colors : ℕ),
  total_students = 800 →
  red_percent = 23 →
  green_percent = 15 →
  students_other_colors = 136 →
  ((total_students - students_other_colors) - (red_percent + green_percent) = 45) :=
by
  intros total_students red_percent green_percent students_other_colors h_total h_red h_green h_other
  have h_other_percent : (students_other_colors * 100 / total_students) = 17 :=
    sorry
  exact sorry

end NUMINAMATH_GPT_percentage_of_students_wearing_blue_shirts_l113_11332


namespace NUMINAMATH_GPT_length_to_width_ratio_l113_11316

-- Define the conditions: perimeter and length
variable (P : ℕ) (l : ℕ) (w : ℕ)

-- Given conditions
def conditions : Prop := (P = 100) ∧ (l = 40) ∧ (P = 2 * l + 2 * w)

-- The proposition we want to prove
def ratio : Prop := l / w = 4

-- The main theorem
theorem length_to_width_ratio (h : conditions P l w) : ratio l w :=
by sorry

end NUMINAMATH_GPT_length_to_width_ratio_l113_11316


namespace NUMINAMATH_GPT_inheritance_amount_l113_11393

theorem inheritance_amount (x : ℝ) 
    (federal_tax : ℝ := 0.25 * x) 
    (remaining_after_federal_tax : ℝ := x - federal_tax) 
    (state_tax : ℝ := 0.15 * remaining_after_federal_tax) 
    (total_taxes : ℝ := federal_tax + state_tax) 
    (taxes_paid : total_taxes = 15000) : 
    x = 41379 :=
sorry

end NUMINAMATH_GPT_inheritance_amount_l113_11393


namespace NUMINAMATH_GPT_cans_left_to_be_loaded_l113_11317

def cartons_total : ℕ := 50
def cartons_loaded : ℕ := 40
def cans_per_carton : ℕ := 20

theorem cans_left_to_be_loaded : (cartons_total - cartons_loaded) * cans_per_carton = 200 := by
  sorry

end NUMINAMATH_GPT_cans_left_to_be_loaded_l113_11317


namespace NUMINAMATH_GPT_exists_triangle_with_edges_l113_11391

variable {A B C D: Type}
variables (AB AC AD BC BD CD : ℝ)
variables (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)

def x := AB * CD
def y := AC * BD
def z := AD * BC

theorem exists_triangle_with_edges :
  ∃ (x y z : ℝ), 
  ∃ (A B C D: Type),
  ∃ (AB AC AD BC BD CD : ℝ) (tetrahedron : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D),
  x = AB * CD ∧ y = AC * BD ∧ z = AD * BC → 
  (x + y > z ∧ y + z > x ∧ z + x > y) :=
by
  sorry

end NUMINAMATH_GPT_exists_triangle_with_edges_l113_11391


namespace NUMINAMATH_GPT_solution_set_l113_11368

open Set Real

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_at_two : f 2 = 0
axiom f_cond : ∀ x : ℝ, 0 < x → x * (deriv (deriv f) x) + f x < 0

theorem solution_set :
  {x : ℝ | x * f x > 0} = Ioo (-2 : ℝ) 0 ∪ Ioo 0 2 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l113_11368


namespace NUMINAMATH_GPT_solve_for_y_l113_11374

theorem solve_for_y (y : ℝ) (h : 5^(3 * y) = Real.sqrt 125) : y = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_solve_for_y_l113_11374


namespace NUMINAMATH_GPT_quadratic_sum_eq_504_l113_11379

theorem quadratic_sum_eq_504 :
  ∃ (a b c : ℝ), (∀ x : ℝ, 20 * x^2 + 160 * x + 800 = a * (x + b)^2 + c) ∧ a + b + c = 504 :=
by sorry

end NUMINAMATH_GPT_quadratic_sum_eq_504_l113_11379


namespace NUMINAMATH_GPT_B_listing_method_l113_11382

-- Definitions for given conditions
def A : Set ℤ := {-2, -1, 1, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t*t}

-- The mathematically equivalent proof problem
theorem B_listing_method :
  B = {4, 1, 9, 16} := 
by {
  sorry
}

end NUMINAMATH_GPT_B_listing_method_l113_11382


namespace NUMINAMATH_GPT_find_y_l113_11335

theorem find_y (k p y : ℝ) (hk : k ≠ 0) (hp : p ≠ 0) 
  (h : (y - 2 * k)^2 - (y - 3 * k)^2 = 4 * k^2 - p) : 
  y = -(p + k^2) / (2 * k) :=
sorry

end NUMINAMATH_GPT_find_y_l113_11335


namespace NUMINAMATH_GPT_nguyen_fabric_yards_l113_11340

open Nat

theorem nguyen_fabric_yards :
  let fabric_per_pair := 8.5
  let pairs_needed := 7
  let fabric_still_needed := 49
  let total_fabric_needed := pairs_needed * fabric_per_pair
  let fabric_already_have := total_fabric_needed - fabric_still_needed
  let yards_of_fabric := fabric_already_have / 3
  yards_of_fabric = 3.5 := by
    sorry

end NUMINAMATH_GPT_nguyen_fabric_yards_l113_11340


namespace NUMINAMATH_GPT_problem1_l113_11355

theorem problem1 (f : ℚ → ℚ) (a : Fin 7 → ℚ) (h₁ : ∀ x, f x = (1 - 3 * x) * (1 + x) ^ 5)
  (h₂ : ∀ x, f x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6) :
  a 0 + (1/3) * a 1 + (1/3^2) * a 2 + (1/3^3) * a 3 + (1/3^4) * a 4 + (1/3^5) * a 5 + (1/3^6) * a 6 = 
  (1 - 3 * (1/3)) * (1 + (1/3))^5 :=
by sorry

end NUMINAMATH_GPT_problem1_l113_11355


namespace NUMINAMATH_GPT_find_W_l113_11337

noncomputable def volumeOutsideCylinder (r_cylinder r_sphere : ℝ) : ℝ :=
  let h := 2 * Real.sqrt (r_sphere^2 - r_cylinder^2)
  let V_sphere := (4 / 3) * Real.pi * r_sphere^3
  let V_cylinder := Real.pi * r_cylinder^2 * h
  V_sphere - V_cylinder

theorem find_W : 
  volumeOutsideCylinder 4 7 = (1372 / 3 - 32 * Real.sqrt 33) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_find_W_l113_11337


namespace NUMINAMATH_GPT_fg_of_3_is_94_l113_11365

def g (x : ℕ) : ℕ := 4 * x + 5
def f (x : ℕ) : ℕ := 6 * x - 8

theorem fg_of_3_is_94 : f (g 3) = 94 := by
  sorry

end NUMINAMATH_GPT_fg_of_3_is_94_l113_11365


namespace NUMINAMATH_GPT_divisibility_condition_l113_11359

theorem divisibility_condition (n : ℕ) : 
  13 ∣ (4 * 3^(2^n) + 3 * 4^(2^n)) ↔ Even n := 
sorry

end NUMINAMATH_GPT_divisibility_condition_l113_11359


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l113_11395

/-
Problem: Prove ( (a + 1) / (a - 1) + 1 ) / ( 2a / (a^2 - 1) ) = 2024 given a = 2023.
-/

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  ( (a + 1) / (a - 1) + 1 ) / ( 2 * a / (a^2 - 1) ) = 2024 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l113_11395


namespace NUMINAMATH_GPT_train_passes_jogger_in_approx_36_seconds_l113_11348

noncomputable def jogger_speed_kmph : ℝ := 8
noncomputable def train_speed_kmph : ℝ := 55
noncomputable def distance_ahead_m : ℝ := 340
noncomputable def train_length_m : ℝ := 130

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 3600

noncomputable def jogger_speed_mps : ℝ :=
  kmph_to_mps jogger_speed_kmph

noncomputable def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps - jogger_speed_mps

noncomputable def total_distance_m : ℝ :=
  distance_ahead_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ :=
  total_distance_m / relative_speed_mps

theorem train_passes_jogger_in_approx_36_seconds : 
  abs (time_to_pass_jogger_s - 36) < 1 := 
sorry

end NUMINAMATH_GPT_train_passes_jogger_in_approx_36_seconds_l113_11348


namespace NUMINAMATH_GPT_gcd_72_108_150_l113_11319

theorem gcd_72_108_150 : Nat.gcd (Nat.gcd 72 108) 150 = 6 := by
  sorry

end NUMINAMATH_GPT_gcd_72_108_150_l113_11319


namespace NUMINAMATH_GPT_parabola_focus_l113_11341

theorem parabola_focus (a : ℝ) (h : a = 4) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / (4 * a)) ∧ y = 4 * x^2 → f = (0, 1 / 16) := 
by {
  sorry
}

end NUMINAMATH_GPT_parabola_focus_l113_11341


namespace NUMINAMATH_GPT_proof_system_l113_11394

-- Define the system of equations
def system_of_equations (x y : ℝ) : Prop :=
  6 * x - 2 * y = 1 ∧ 2 * x + y = 2

-- Define the solution to the system of equations
def solution_equations (x y : ℝ) : Prop :=
  x = 0.5 ∧ y = 1

-- Define the system of inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  2 * x - 10 < 0 ∧ (x + 1) / 3 < x - 1

-- Define the solution set for the system of inequalities
def solution_inequalities (x : ℝ) : Prop :=
  2 < x ∧ x < 5

-- The final theorem to be proved
theorem proof_system :
  ∃ x y : ℝ, system_of_equations x y ∧ solution_equations x y ∧ system_of_inequalities x ∧ solution_inequalities x :=
by
  sorry

end NUMINAMATH_GPT_proof_system_l113_11394


namespace NUMINAMATH_GPT_rolls_in_package_l113_11327

theorem rolls_in_package (n : ℕ) :
  (9 : ℝ) = (n : ℝ) * (1 - 0.25) → n = 12 :=
by
  sorry

end NUMINAMATH_GPT_rolls_in_package_l113_11327


namespace NUMINAMATH_GPT_max_food_per_guest_l113_11352

theorem max_food_per_guest (total_food : ℕ) (min_guests : ℕ)
    (H1 : total_food = 406) (H2 : min_guests = 163) :
    2 ≤ total_food / min_guests ∧ total_food / min_guests < 3 := by
  sorry

end NUMINAMATH_GPT_max_food_per_guest_l113_11352
