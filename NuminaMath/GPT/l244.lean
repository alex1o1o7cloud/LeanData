import Mathlib

namespace NUMINAMATH_GPT_rectangle_area_increase_l244_24418

-- Definitions to match the conditions
variables {l w : ℝ}

-- The statement 
theorem rectangle_area_increase (h1 : l > 0) (h2 : w > 0) :
  (((1.15 * l) * (1.2 * w) - (l * w)) / (l * w)) * 100 = 38 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_increase_l244_24418


namespace NUMINAMATH_GPT_fraction_equal_decimal_l244_24496

theorem fraction_equal_decimal : (1 / 4) = 0.25 :=
sorry

end NUMINAMATH_GPT_fraction_equal_decimal_l244_24496


namespace NUMINAMATH_GPT_general_formula_sequence_l244_24429

variable {a : ℕ → ℝ}

-- Definitions and assumptions
def recurrence_relation (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a n - 2 * a (n + 1) + a (n + 2) = 0

def initial_conditions (a : ℕ → ℝ) : Prop :=
  a 1 = 2 ∧ a 2 = 4

-- The proof problem
theorem general_formula_sequence (a : ℕ → ℝ)
  (h1 : recurrence_relation a)
  (h2 : initial_conditions a) :
  ∀ n : ℕ, a n = 2 * n :=

sorry

end NUMINAMATH_GPT_general_formula_sequence_l244_24429


namespace NUMINAMATH_GPT_length_of_platform_l244_24481

theorem length_of_platform (length_of_train speed_of_train time_to_cross : ℕ) 
    (h1 : length_of_train = 450) (h2 : speed_of_train = 126) (h3 : time_to_cross = 20) :
    ∃ length_of_platform : ℕ, length_of_platform = 250 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_platform_l244_24481


namespace NUMINAMATH_GPT_compute_expression_l244_24461

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l244_24461


namespace NUMINAMATH_GPT_polar_conversion_equiv_l244_24491

noncomputable def polar_convert (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
if r < 0 then (-r, θ + Real.pi) else (r, θ)

theorem polar_conversion_equiv : polar_convert (-3) (Real.pi / 4) = (3, 5 * Real.pi / 4) :=
by
  sorry

end NUMINAMATH_GPT_polar_conversion_equiv_l244_24491


namespace NUMINAMATH_GPT_Nicole_fewer_questions_l244_24451

-- Definitions based on the given conditions
def Nicole_correct : ℕ := 22
def Cherry_correct : ℕ := 17
def Kim_correct : ℕ := Cherry_correct + 8

-- Theorem to prove the number of fewer questions Nicole answered compared to Kim
theorem Nicole_fewer_questions : Kim_correct - Nicole_correct = 3 :=
by
  -- We set up the definitions
  let Nicole_correct := 22
  let Cherry_correct := 17
  let Kim_correct := Cherry_correct + 8
  -- The proof will be filled in here. 
  -- The goal theorem statement is filled with 'sorry' to bypass the actual proof.
  have : Kim_correct - Nicole_correct = 3 := sorry
  exact this

end NUMINAMATH_GPT_Nicole_fewer_questions_l244_24451


namespace NUMINAMATH_GPT_sin_600_eq_l244_24489

theorem sin_600_eq : Real.sin (600 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_600_eq_l244_24489


namespace NUMINAMATH_GPT_constantin_mother_deposit_return_l244_24442

theorem constantin_mother_deposit_return :
  (10000 : ℝ) * 58.15 = 581500 :=
by
  sorry

end NUMINAMATH_GPT_constantin_mother_deposit_return_l244_24442


namespace NUMINAMATH_GPT_division_expression_is_7_l244_24490

noncomputable def evaluate_expression : ℝ :=
  1 / 2 / 3 / 4 / 5 / (6 / 7 / 8 / 9 / 10)

theorem division_expression_is_7 : evaluate_expression = 7 :=
by
  sorry

end NUMINAMATH_GPT_division_expression_is_7_l244_24490


namespace NUMINAMATH_GPT_fraction_value_l244_24497

theorem fraction_value : (1998 - 998) / 1000 = 1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l244_24497


namespace NUMINAMATH_GPT_survey_method_correct_l244_24437

/-- Definitions to represent the options in the survey method problem. -/
inductive SurveyMethod
| A
| B
| C
| D

/-- The function to determine the correct survey method. -/
def appropriate_survey_method : SurveyMethod :=
  SurveyMethod.C

/-- The theorem stating that the appropriate survey method is indeed option C. -/
theorem survey_method_correct : appropriate_survey_method = SurveyMethod.C :=
by
  /- The actual proof is omitted as per instruction. -/
  sorry

end NUMINAMATH_GPT_survey_method_correct_l244_24437


namespace NUMINAMATH_GPT_sequence_expression_l244_24455

theorem sequence_expression {a : ℕ → ℝ} (h1 : ∀ n, a (n + 1) ^ 2 = a n ^ 2 + 4)
  (h2 : a 1 = 1) (h3 : ∀ n, a n > 0) : ∀ n, a n = Real.sqrt (4 * n - 3) := by
  sorry

end NUMINAMATH_GPT_sequence_expression_l244_24455


namespace NUMINAMATH_GPT_bob_deli_total_cost_l244_24457

-- Definitions based on the problem's conditions
def sandwich_cost : ℕ := 5
def soda_cost : ℕ := 3
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 10
def discount_threshold : ℕ := 50
def discount_amount : ℕ := 10

-- The total initial cost without discount
def initial_total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_sodas * soda_cost)

-- The final cost after applying discount if applicable
def final_cost : ℕ :=
  if initial_total_cost > discount_threshold then
    initial_total_cost - discount_amount
  else
    initial_total_cost

-- Statement to prove
theorem bob_deli_total_cost : final_cost = 55 := by
  sorry

end NUMINAMATH_GPT_bob_deli_total_cost_l244_24457


namespace NUMINAMATH_GPT_correct_statements_l244_24439

-- Definitions
noncomputable def f (x b c : ℝ) : ℝ := abs x * x + b * x + c

-- Proof statements
theorem correct_statements (b c : ℝ) :
  (b > 0 → ∀ x y : ℝ, x ≤ y → f x b c ≤ f y b c) ∧
  (b < 0 → ¬ (∀ x : ℝ, ∃ m : ℝ, f x b c = m)) ∧
  (b = 0 → ∀ x : ℝ, f (x) b c = f (-x) b c) ∧
  (∃ x1 x2 x3 : ℝ, f x1 b c = 0 ∧ f x2 b c = 0 ∧ f x3 b c = 0) :=
sorry

end NUMINAMATH_GPT_correct_statements_l244_24439


namespace NUMINAMATH_GPT_simplify_expression_eq_l244_24452

noncomputable def simplified_expression (b : ℝ) : ℝ :=
  (Real.rpow (Real.rpow (b ^ 16) (1 / 8)) (1 / 4)) ^ 3 *
  (Real.rpow (Real.rpow (b ^ 16) (1 / 4)) (1 / 8)) ^ 3

theorem simplify_expression_eq (b : ℝ) (hb : 0 < b) :
  simplified_expression b = b ^ 3 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_eq_l244_24452


namespace NUMINAMATH_GPT_total_number_of_items_l244_24428

-- Definitions based on the problem conditions
def number_of_notebooks : ℕ := 40
def pens_more_than_notebooks : ℕ := 80
def pencils_more_than_notebooks : ℕ := 45

-- Total items calculation based on the conditions
def number_of_pens : ℕ := number_of_notebooks + pens_more_than_notebooks
def number_of_pencils : ℕ := number_of_notebooks + pencils_more_than_notebooks
def total_items : ℕ := number_of_notebooks + number_of_pens + number_of_pencils

-- Statement to be proved
theorem total_number_of_items : total_items = 245 := 
by 
  sorry

end NUMINAMATH_GPT_total_number_of_items_l244_24428


namespace NUMINAMATH_GPT_custom_mul_2021_1999_l244_24495

axiom custom_mul : ℕ → ℕ → ℕ

axiom custom_mul_id1 : ∀ (A : ℕ), custom_mul A A = 0
axiom custom_mul_id2 : ∀ (A B C : ℕ), custom_mul A (custom_mul B C) = custom_mul A B + C

theorem custom_mul_2021_1999 : custom_mul 2021 1999 = 22 := by
  sorry

end NUMINAMATH_GPT_custom_mul_2021_1999_l244_24495


namespace NUMINAMATH_GPT_number_division_l244_24486

theorem number_division (n : ℕ) (h1 : n / 25 = 5) (h2 : n % 25 = 2) : n = 127 :=
by
  sorry

end NUMINAMATH_GPT_number_division_l244_24486


namespace NUMINAMATH_GPT_penultimate_digit_of_quotient_l244_24408

theorem penultimate_digit_of_quotient :
  (4^1994 + 7^1994) / 10 % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_penultimate_digit_of_quotient_l244_24408


namespace NUMINAMATH_GPT_find_natural_solution_l244_24440

theorem find_natural_solution (x y : ℕ) (h : y^6 + 2 * y^3 - y^2 + 1 = x^3) : x = 1 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_solution_l244_24440


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l244_24480

/-- Problem 1: Calculate 25 * 26 * 8 and show it equals 5200 --/
theorem problem1 : 25 * 26 * 8 = 5200 := 
sorry

/-- Problem 2: Calculate 340 * 40 / 17 and show it equals 800 --/
theorem problem2 : 340 * 40 / 17 = 800 := 
sorry

/-- Problem 3: Calculate 440 * 15 + 480 * 15 + 79 * 15 + 15 and show it equals 15000 --/
theorem problem3 : 440 * 15 + 480 * 15 + 79 * 15 + 15 = 15000 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l244_24480


namespace NUMINAMATH_GPT_one_third_way_l244_24485

theorem one_third_way (x₁ x₂ : ℚ) (w₁ w₂ : ℕ) (h₁ : x₁ = 1/4) (h₂ : x₂ = 3/4) (h₃ : w₁ = 2) (h₄ : w₂ = 1) : 
  (w₁ * x₁ + w₂ * x₂) / (w₁ + w₂) = 5 / 12 :=
by 
  rw [h₁, h₂, h₃, h₄]
  -- Simplification of the weighted average to get 5/12
  sorry

end NUMINAMATH_GPT_one_third_way_l244_24485


namespace NUMINAMATH_GPT_sequence_general_term_l244_24483

theorem sequence_general_term (a : ℕ → ℤ) : 
  (∀ n, a n = (-1)^(n + 1) * (3 * n - 2)) ↔ 
  (a 1 = 1 ∧ a 2 = -4 ∧ a 3 = 7 ∧ a 4 = -10 ∧ a 5 = 13) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l244_24483


namespace NUMINAMATH_GPT_max_product_of_sum_300_l244_24414

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_max_product_of_sum_300_l244_24414


namespace NUMINAMATH_GPT_maximum_x_plus_2y_l244_24473

theorem maximum_x_plus_2y 
  (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x^2 + 8 * y^2 + x * y = 2) :
  x + 2 * y ≤ 4 / 3 :=
sorry

end NUMINAMATH_GPT_maximum_x_plus_2y_l244_24473


namespace NUMINAMATH_GPT_number_of_poles_l244_24400

theorem number_of_poles (side_length : ℝ) (distance_between_poles : ℝ) 
  (h1 : side_length = 150) (h2 : distance_between_poles = 30) : 
  ((4 * side_length) / distance_between_poles) = 20 :=
by 
  -- Placeholder to indicate missing proof
  sorry

end NUMINAMATH_GPT_number_of_poles_l244_24400


namespace NUMINAMATH_GPT_ellipse_equation_standard_form_l244_24434

theorem ellipse_equation_standard_form :
  ∃ (a b : ℝ) (h k : ℝ), 
    a = (Real.sqrt 146 + Real.sqrt 242) / 2 ∧ 
    b = Real.sqrt ((Real.sqrt 146 + Real.sqrt 242) / 2)^2 - 9 ∧ 
    h = 1 ∧ 
    k = 4 ∧ 
    (∀ x y : ℝ, (x, y) = (12, -4) → 
      ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1)) :=
  sorry

end NUMINAMATH_GPT_ellipse_equation_standard_form_l244_24434


namespace NUMINAMATH_GPT_f_odd_f_shift_f_in_range_find_f_7_5_l244_24454

def f : ℝ → ℝ := sorry  -- We define the function f (implementation is not needed here)

theorem f_odd (x : ℝ) : f (-x) = -f x := sorry

theorem f_shift (x : ℝ) : f (x + 2) = -f x := sorry

theorem f_in_range (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f x = x := sorry

theorem find_f_7_5 : f 7.5 = 0.5 :=
by
  sorry

end NUMINAMATH_GPT_f_odd_f_shift_f_in_range_find_f_7_5_l244_24454


namespace NUMINAMATH_GPT_alice_needs_7_fills_to_get_3_cups_l244_24403

theorem alice_needs_7_fills_to_get_3_cups (needs : ℚ) (cup_size : ℚ) (has : ℚ) :
  needs = 3 ∧ cup_size = 1 / 3 ∧ has = 2 / 3 →
  (needs - has) / cup_size = 7 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end NUMINAMATH_GPT_alice_needs_7_fills_to_get_3_cups_l244_24403


namespace NUMINAMATH_GPT_train_pass_time_is_38_seconds_l244_24471

noncomputable def speed_of_jogger_kmhr : ℝ := 9
noncomputable def speed_of_train_kmhr : ℝ := 45
noncomputable def lead_distance_m : ℝ := 260
noncomputable def train_length_m : ℝ := 120

noncomputable def speed_of_jogger_ms : ℝ := speed_of_jogger_kmhr * (1000 / 3600)
noncomputable def speed_of_train_ms : ℝ := speed_of_train_kmhr * (1000 / 3600)

noncomputable def relative_speed_ms : ℝ := speed_of_train_ms - speed_of_jogger_ms
noncomputable def total_distance_m : ℝ := lead_distance_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ := total_distance_m / relative_speed_ms

theorem train_pass_time_is_38_seconds :
  time_to_pass_jogger_s = 38 := 
sorry

end NUMINAMATH_GPT_train_pass_time_is_38_seconds_l244_24471


namespace NUMINAMATH_GPT_census_survey_is_suitable_l244_24479

def suitable_for_census (s: String) : Prop :=
  s = "Understand the vision condition of students in a class"

theorem census_survey_is_suitable :
  suitable_for_census "Understand the vision condition of students in a class" :=
by
  sorry

end NUMINAMATH_GPT_census_survey_is_suitable_l244_24479


namespace NUMINAMATH_GPT_isosceles_triangle_height_ratio_l244_24466

theorem isosceles_triangle_height_ratio (a b : ℝ) (h₁ : b = (4 / 3) * a) :
  ∃ m n : ℝ, b / 2 = m + n ∧ m = (2 / 3) * a ∧ n = (1 / 3) * a ∧ (m / n) = 2 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_height_ratio_l244_24466


namespace NUMINAMATH_GPT_digit_one_not_in_mean_l244_24475

def seq : List ℕ := [5, 55, 555, 5555, 55555, 555555, 5555555, 55555555, 555555555]

noncomputable def arithmetic_mean (l : List ℕ) : ℕ := l.sum / l.length

theorem digit_one_not_in_mean :
  ¬(∃ d, d ∈ (arithmetic_mean seq).digits 10 ∧ d = 1) :=
sorry

end NUMINAMATH_GPT_digit_one_not_in_mean_l244_24475


namespace NUMINAMATH_GPT_average_speed_l244_24411

theorem average_speed (v : ℝ) (h : 500 / v - 500 / (v + 10) = 2) : v = 45.25 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l244_24411


namespace NUMINAMATH_GPT_alicia_total_deductions_in_cents_l244_24406

def Alicia_hourly_wage : ℝ := 25
def local_tax_rate : ℝ := 0.015
def retirement_contribution_rate : ℝ := 0.03

theorem alicia_total_deductions_in_cents :
  let wage_cents := Alicia_hourly_wage * 100
  let tax_deduction := wage_cents * local_tax_rate
  let after_tax_earnings := wage_cents - tax_deduction
  let retirement_contribution := after_tax_earnings * retirement_contribution_rate
  let total_deductions := tax_deduction + retirement_contribution
  total_deductions = 111 :=
by
  sorry

end NUMINAMATH_GPT_alicia_total_deductions_in_cents_l244_24406


namespace NUMINAMATH_GPT_twice_a_minus_4_nonnegative_l244_24487

theorem twice_a_minus_4_nonnegative (a : ℝ) : 2 * a - 4 ≥ 0 ↔ 2 * a - 4 = 0 ∨ 2 * a - 4 > 0 := 
by
  sorry

end NUMINAMATH_GPT_twice_a_minus_4_nonnegative_l244_24487


namespace NUMINAMATH_GPT_initial_positions_2048_l244_24458

noncomputable def number_of_initial_positions (n : ℕ) : ℤ :=
  2 ^ n - 2

theorem initial_positions_2048 : number_of_initial_positions 2048 = 2 ^ 2048 - 2 :=
by
  sorry

end NUMINAMATH_GPT_initial_positions_2048_l244_24458


namespace NUMINAMATH_GPT_log_expression_simplifies_to_one_l244_24402

theorem log_expression_simplifies_to_one :
  (Real.log 5)^2 + Real.log 50 * Real.log 2 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_log_expression_simplifies_to_one_l244_24402


namespace NUMINAMATH_GPT_swim_distance_downstream_l244_24443

theorem swim_distance_downstream 
  (V_m V_s : ℕ) 
  (t d : ℕ) 
  (h1 : V_m = 9) 
  (h2 : t = 3) 
  (h3 : 3 * (V_m - V_s) = 18) : 
  t * (V_m + V_s) = 36 := 
by 
  sorry

end NUMINAMATH_GPT_swim_distance_downstream_l244_24443


namespace NUMINAMATH_GPT_abs_diff_p_q_l244_24446

theorem abs_diff_p_q (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
by 
  sorry

end NUMINAMATH_GPT_abs_diff_p_q_l244_24446


namespace NUMINAMATH_GPT_rate_per_kg_first_batch_l244_24465

/-- This theorem proves the rate per kg of the first batch of wheat. -/
theorem rate_per_kg_first_batch (x : ℝ) 
  (h1 : 30 * x + 20 * 14.25 = 285 + 30 * x) 
  (h2 : (30 * x + 285) * 1.3 = 819) : 
  x = 11.5 := 
sorry

end NUMINAMATH_GPT_rate_per_kg_first_batch_l244_24465


namespace NUMINAMATH_GPT_simplify_expression_l244_24468

theorem simplify_expression :
  (8 * 10^12) / (4 * 10^4) + 2 * 10^3 = 200002000 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l244_24468


namespace NUMINAMATH_GPT_convert_quadratic_to_general_form_l244_24421

theorem convert_quadratic_to_general_form
  (x : ℝ)
  (h : 3 * x * (x - 3) = 4) :
  3 * x ^ 2 - 9 * x - 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_convert_quadratic_to_general_form_l244_24421


namespace NUMINAMATH_GPT_joseph_drives_more_l244_24436

def joseph_speed : ℝ := 50
def joseph_time : ℝ := 2.5
def kyle_speed : ℝ := 62
def kyle_time : ℝ := 2

def joseph_distance : ℝ := joseph_speed * joseph_time
def kyle_distance : ℝ := kyle_speed * kyle_time

theorem joseph_drives_more : (joseph_distance - kyle_distance) = 1 := by
  sorry

end NUMINAMATH_GPT_joseph_drives_more_l244_24436


namespace NUMINAMATH_GPT_relatively_prime_perfect_squares_l244_24472

theorem relatively_prime_perfect_squares (a b c : ℤ) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_eq : (1:ℚ) / a + (1:ℚ) / b = (1:ℚ) / c) :
    ∃ x y z : ℤ, (a + b = x^2 ∧ a - c = y^2 ∧ b - c = z^2) :=
  sorry

end NUMINAMATH_GPT_relatively_prime_perfect_squares_l244_24472


namespace NUMINAMATH_GPT_proof_problem_l244_24453

noncomputable def f (x y k : ℝ) : ℝ := k * x + (1 / y)

theorem proof_problem
  (a b k : ℝ) (h1 : f a b k = f b a k) (h2 : a ≠ b) :
  f (a * b) 1 k = 0 :=
sorry

end NUMINAMATH_GPT_proof_problem_l244_24453


namespace NUMINAMATH_GPT_percent_increase_in_sales_l244_24407

theorem percent_increase_in_sales (sales_this_year : ℕ) (sales_last_year : ℕ) (percent_increase : ℚ) :
  sales_this_year = 400 ∧ sales_last_year = 320 → percent_increase = 25 :=
by
  sorry

end NUMINAMATH_GPT_percent_increase_in_sales_l244_24407


namespace NUMINAMATH_GPT_fencing_cost_proof_l244_24462

noncomputable def totalCostOfFencing (length : ℕ) (breadth : ℕ) (costPerMeter : ℚ) : ℚ :=
  2 * (length + breadth) * costPerMeter

theorem fencing_cost_proof : totalCostOfFencing 56 (56 - 12) 26.50 = 5300 := by
  sorry

end NUMINAMATH_GPT_fencing_cost_proof_l244_24462


namespace NUMINAMATH_GPT_dagger_evaluation_l244_24425

def dagger (a b : ℚ) : ℚ :=
match a, b with
| ⟨m, n, _, _⟩, ⟨p, q, _, _⟩ => (m * p : ℚ) * (q / n : ℚ)

theorem dagger_evaluation : dagger (3/7) (11/4) = 132/7 := by
  sorry

end NUMINAMATH_GPT_dagger_evaluation_l244_24425


namespace NUMINAMATH_GPT_total_playtime_l244_24492

noncomputable def lena_playtime_minutes : ℕ := 210
noncomputable def brother_playtime_minutes (lena_playtime: ℕ) : ℕ := lena_playtime + 17
noncomputable def sister_playtime_minutes (brother_playtime: ℕ) : ℕ := 2 * brother_playtime

theorem total_playtime
  (lena_playtime : ℕ)
  (brother_playtime : ℕ)
  (sister_playtime : ℕ)
  (h_lena : lena_playtime = lena_playtime_minutes)
  (h_brother : brother_playtime = brother_playtime_minutes lena_playtime)
  (h_sister : sister_playtime = sister_playtime_minutes brother_playtime) :
  lena_playtime + brother_playtime + sister_playtime = 891 := 
  by sorry

end NUMINAMATH_GPT_total_playtime_l244_24492


namespace NUMINAMATH_GPT_xy_range_l244_24427

theorem xy_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 1/x + y + 1/y = 5) :
  1/4 ≤ x * y ∧ x * y ≤ 4 :=
sorry

end NUMINAMATH_GPT_xy_range_l244_24427


namespace NUMINAMATH_GPT_sin_y_gt_half_x_l244_24494

theorem sin_y_gt_half_x (x y : ℝ) (hx : x ≤ 90) (h : Real.sin y = (3 / 4) * Real.sin x) : y > x / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_y_gt_half_x_l244_24494


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l244_24488

-- Problem 1
theorem problem1 (x : ℤ) (h : 4 * x = 20) : x = 5 :=
sorry

-- Problem 2
theorem problem2 (x : ℤ) (h : x - 18 = 40) : x = 58 :=
sorry

-- Problem 3
theorem problem3 (x : ℤ) (h : x / 7 = 12) : x = 84 :=
sorry

-- Problem 4
theorem problem4 (n : ℚ) (h : 8 * n / 2 = 15) : n = 15 / 4 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l244_24488


namespace NUMINAMATH_GPT_number_of_BMWs_sold_l244_24450

theorem number_of_BMWs_sold (total_cars_sold : ℕ)
  (percent_Ford percent_Nissan percent_Chevrolet : ℕ)
  (h_total : total_cars_sold = 300)
  (h_percent_Ford : percent_Ford = 18)
  (h_percent_Nissan : percent_Nissan = 25)
  (h_percent_Chevrolet : percent_Chevrolet = 20) :
  (300 * (100 - (percent_Ford + percent_Nissan + percent_Chevrolet)) / 100) = 111 :=
by
  -- We assert that the calculated number of BMWs is 111
  sorry

end NUMINAMATH_GPT_number_of_BMWs_sold_l244_24450


namespace NUMINAMATH_GPT_area_not_covered_by_small_squares_l244_24426

def large_square_side_length : ℕ := 10
def small_square_side_length : ℕ := 4
def large_square_area : ℕ := large_square_side_length ^ 2
def small_square_area : ℕ := small_square_side_length ^ 2
def uncovered_area : ℕ := large_square_area - small_square_area

theorem area_not_covered_by_small_squares :
  uncovered_area = 84 := by
  sorry

end NUMINAMATH_GPT_area_not_covered_by_small_squares_l244_24426


namespace NUMINAMATH_GPT_merchant_profit_percentage_l244_24493

-- Given
def initial_cost_price : ℝ := 100
def marked_price : ℝ := initial_cost_price + 0.50 * initial_cost_price
def discount_percentage : ℝ := 0.20
def discount : ℝ := discount_percentage * marked_price
def selling_price : ℝ := marked_price - discount

-- Prove
theorem merchant_profit_percentage :
  ((selling_price - initial_cost_price) / initial_cost_price) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_merchant_profit_percentage_l244_24493


namespace NUMINAMATH_GPT_triangle_inequality_l244_24431

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : 
  5 * (a^2 + b^2 + c^2) + 18 * a * b * c ≥ 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l244_24431


namespace NUMINAMATH_GPT_range_of_a_l244_24474

theorem range_of_a (a : ℝ) :
  (1 ∉ {x : ℝ | x^2 - 2 * x + a > 0}) → a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l244_24474


namespace NUMINAMATH_GPT_journey_time_ratio_l244_24430

theorem journey_time_ratio (D : ℝ) (h₁ : D > 0) :
  let T1 := D / 48
  let T2 := D / 32
  (T2 / T1) = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_journey_time_ratio_l244_24430


namespace NUMINAMATH_GPT_no_positive_integers_satisfy_equation_l244_24413

theorem no_positive_integers_satisfy_equation :
  ¬ ∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ a^2 = b^11 + 23 :=
by
  sorry

end NUMINAMATH_GPT_no_positive_integers_satisfy_equation_l244_24413


namespace NUMINAMATH_GPT_maximum_absolute_sum_l244_24477

theorem maximum_absolute_sum (x y z : ℝ) (h : x^2 + y^2 + z^2 = 1) : |x| + |y| + |z| ≤ 2 :=
sorry

end NUMINAMATH_GPT_maximum_absolute_sum_l244_24477


namespace NUMINAMATH_GPT_maximize_area_l244_24484

-- Define the variables and constants
variables {x y p : ℝ}

-- Define the conditions
def perimeter (x y p : ℝ) := (2 * x + 2 * y = p)
def area (x y : ℝ) := x * y

-- The theorem statement with conditions
theorem maximize_area (h : perimeter x y p) : x = y → x = p / 4 :=
by
  sorry

end NUMINAMATH_GPT_maximize_area_l244_24484


namespace NUMINAMATH_GPT_time_to_cross_same_direction_l244_24415

-- Defining the conditions
def speed_train1 : ℝ := 60 -- kmph
def speed_train2 : ℝ := 40 -- kmph
def time_opposite_directions : ℝ := 10.000000000000002 -- seconds 
def relative_speed_opposite_directions : ℝ := speed_train1 + speed_train2 -- 100 kmph
def relative_speed_same_direction : ℝ := speed_train1 - speed_train2 -- 20 kmph

-- Defining the proof statement
theorem time_to_cross_same_direction : 
  (time_opposite_directions * (relative_speed_opposite_directions / relative_speed_same_direction)) = 50 :=
by
  sorry

end NUMINAMATH_GPT_time_to_cross_same_direction_l244_24415


namespace NUMINAMATH_GPT_problem_statement_l244_24438

variable (a : ℕ → ℝ)

-- Defining sequences {b_n} and {c_n}
def b (n : ℕ) := a n - a (n + 2)
def c (n : ℕ) := a n + 2 * a (n + 1) + 3 * a (n + 2)

-- Defining that a sequence is arithmetic
def is_arithmetic (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) - seq n = d

-- Problem statement
theorem problem_statement :
  is_arithmetic a ↔ (is_arithmetic (c a) ∧ ∀ n, b a n ≤ b a (n + 1)) :=
sorry

end NUMINAMATH_GPT_problem_statement_l244_24438


namespace NUMINAMATH_GPT_log_expression_value_l244_24441

theorem log_expression_value : 
  let log4_3 := (Real.log 3) / (Real.log 4)
  let log8_3 := (Real.log 3) / (Real.log 8)
  let log3_2 := (Real.log 2) / (Real.log 3)
  let log9_2 := (Real.log 2) / (Real.log 9)
  (log4_3 + log8_3) * (log3_2 + log9_2) = 5 / 4 := 
by
  sorry

end NUMINAMATH_GPT_log_expression_value_l244_24441


namespace NUMINAMATH_GPT_area_of_L_shaped_figure_l244_24456

theorem area_of_L_shaped_figure :
  let large_rect_area := 10 * 7
  let small_rect_area := 4 * 3
  large_rect_area - small_rect_area = 58 := by
  sorry

end NUMINAMATH_GPT_area_of_L_shaped_figure_l244_24456


namespace NUMINAMATH_GPT_sum_coordinates_is_60_l244_24435

theorem sum_coordinates_is_60 :
  let points := [(5 + Real.sqrt 91, 13), (5 - Real.sqrt 91, 13), (5 + Real.sqrt 91, 7), (5 - Real.sqrt 91, 7)]
  let x_coords_sum := (5 + Real.sqrt 91) + (5 - Real.sqrt 91) + (5 + Real.sqrt 91) + (5 - Real.sqrt 91)
  let y_coords_sum := 13 + 13 + 7 + 7
  x_coords_sum + y_coords_sum = 60 :=
by
  sorry

end NUMINAMATH_GPT_sum_coordinates_is_60_l244_24435


namespace NUMINAMATH_GPT_collinear_vectors_m_n_sum_l244_24482

theorem collinear_vectors_m_n_sum (m n : ℕ)
  (h1 : (2, 3, m) = (2 * n, 6, 8)) :
  m + n = 6 :=
sorry

end NUMINAMATH_GPT_collinear_vectors_m_n_sum_l244_24482


namespace NUMINAMATH_GPT_R2_area_l244_24409

-- Definitions for the conditions
def R1_side1 : ℝ := 4
def R1_area : ℝ := 16
def R2_diagonal : ℝ := 10
def similar_rectangles (R1 R2 : ℝ × ℝ) : Prop := (R1.fst / R1.snd = R2.fst / R2.snd)

-- Main theorem
theorem R2_area {a b : ℝ} 
  (R1_side1 : a = 4)
  (R1_area : a * a = 16) 
  (R2_diagonal : b = 10)
  (h : similar_rectangles (a, a) (b / (10 / (2 : ℝ)), b / (10 / (2 : ℝ)))) : 
  b * b / (2 : ℝ) = 50 :=
by
  sorry

end NUMINAMATH_GPT_R2_area_l244_24409


namespace NUMINAMATH_GPT_card_at_42_is_8_spade_l244_24423

-- Conditions Definition
def cards_sequence : List String := 
  ["A♥", "A♠", "2♥", "2♠", "3♥", "3♠", "4♥", "4♠", "5♥", "5♠", "6♥", "6♠", "7♥", "7♠", "8♥", "8♠",
   "9♥", "9♠", "10♥", "10♠", "J♥", "J♠", "Q♥", "Q♠", "K♥", "K♠"]

-- Proposition to be proved
theorem card_at_42_is_8_spade :
  cards_sequence[(41 % 26)] = "8♠" :=
by sorry

end NUMINAMATH_GPT_card_at_42_is_8_spade_l244_24423


namespace NUMINAMATH_GPT_fraction_ordering_l244_24498

theorem fraction_ordering :
  (6:ℚ)/29 < (8:ℚ)/25 ∧ (8:ℚ)/25 < (10:ℚ)/31 :=
by
  sorry

end NUMINAMATH_GPT_fraction_ordering_l244_24498


namespace NUMINAMATH_GPT_visited_both_countries_l244_24420

theorem visited_both_countries (total_people visited_Iceland visited_Norway visited_neither : ℕ) 
(h_total: total_people = 60)
(h_visited_Iceland: visited_Iceland = 35)
(h_visited_Norway: visited_Norway = 23)
(h_visited_neither: visited_neither = 33) : 
total_people - visited_neither = visited_Iceland + visited_Norway - (visited_Iceland + visited_Norway - (total_people - visited_neither)) :=
by sorry

end NUMINAMATH_GPT_visited_both_countries_l244_24420


namespace NUMINAMATH_GPT_coin_tosses_l244_24447

theorem coin_tosses (n : ℤ) (h : (1/2 : ℝ)^n = 0.125) : n = 3 :=
by
  sorry

end NUMINAMATH_GPT_coin_tosses_l244_24447


namespace NUMINAMATH_GPT_distinct_digits_sum_l244_24412

theorem distinct_digits_sum (A B C D G : ℕ) (AB CD GGG : ℕ)
  (h1: AB = 10 * A + B)
  (h2: CD = 10 * C + D)
  (h3: GGG = 111 * G)
  (h4: AB * CD = GGG)
  (h5: A ≠ B)
  (h6: A ≠ C)
  (h7: A ≠ D)
  (h8: A ≠ G)
  (h9: B ≠ C)
  (h10: B ≠ D)
  (h11: B ≠ G)
  (h12: C ≠ D)
  (h13: C ≠ G)
  (h14: D ≠ G)
  (hA: A < 10)
  (hB: B < 10)
  (hC: C < 10)
  (hD: D < 10)
  (hG: G < 10)
  : A + B + C + D + G = 17 := sorry

end NUMINAMATH_GPT_distinct_digits_sum_l244_24412


namespace NUMINAMATH_GPT_rectangle_perimeter_l244_24449

theorem rectangle_perimeter {a b c width : ℕ} (h₁: a = 15) (h₂: b = 20) (h₃: c = 25) (w : ℕ) (h₄: w = 5) :
  let area_triangle := (a * b) / 2
  let length := area_triangle / w
  let perimeter := 2 * (length + w)
  perimeter = 70 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l244_24449


namespace NUMINAMATH_GPT_find_central_angle_l244_24417

-- We define the given conditions.
def radius : ℝ := 2
def area : ℝ := 8

-- We state the theorem that we need to prove.
theorem find_central_angle (R : ℝ) (A : ℝ) (hR : R = radius) (hA : A = area) :
  ∃ α : ℝ, α = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_central_angle_l244_24417


namespace NUMINAMATH_GPT_log_sqrt_7_of_343sqrt7_l244_24405

noncomputable def log_sqrt_7 (y : ℝ) : ℝ := 
  Real.log y / Real.log (Real.sqrt 7)

theorem log_sqrt_7_of_343sqrt7 : log_sqrt_7 (343 * Real.sqrt 7) = 4 :=
by
  sorry

end NUMINAMATH_GPT_log_sqrt_7_of_343sqrt7_l244_24405


namespace NUMINAMATH_GPT_problem_statement_l244_24464

def binary_op (a b : ℚ) : ℚ := (a^2 + b^2) / (a^2 - b^2)

theorem problem_statement : binary_op (binary_op 8 6) 2 = 821 / 429 := 
by sorry

end NUMINAMATH_GPT_problem_statement_l244_24464


namespace NUMINAMATH_GPT_angle_B_value_l244_24478

noncomputable def degree_a (A : ℝ) : Prop := A = 30 ∨ A = 60

noncomputable def degree_b (A B : ℝ) : Prop := B = 3 * A - 60

theorem angle_B_value (A B : ℝ) 
  (h1 : B = 3 * A - 60)
  (h2 : A = 30 ∨ A = 60) :
  B = 30 ∨ B = 120 :=
by
  sorry

end NUMINAMATH_GPT_angle_B_value_l244_24478


namespace NUMINAMATH_GPT_eq_solutions_of_equation_l244_24444

open Int

theorem eq_solutions_of_equation (x y : ℤ) :
  ((x, y) = (0, -4) ∨ (x, y) = (0, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-4, 8) ∨
   (x, y) = (-2, 0) ∨ (x, y) = (-6, 6) ∨
   (x, y) = (0, 0) ∨ (x, y) = (-10, 4)) ↔
  (x - y) * (x - y) = (x - y + 6) * (x + y) :=
sorry

end NUMINAMATH_GPT_eq_solutions_of_equation_l244_24444


namespace NUMINAMATH_GPT_total_number_of_participants_l244_24410

theorem total_number_of_participants (boys_achieving_distance : ℤ) (frequency : ℝ) (h1 : boys_achieving_distance = 8) (h2 : frequency = 0.4) : 
  (boys_achieving_distance : ℝ) / frequency = 20 := 
by 
  sorry

end NUMINAMATH_GPT_total_number_of_participants_l244_24410


namespace NUMINAMATH_GPT_OddPrimeDivisorCondition_l244_24422

theorem OddPrimeDivisorCondition (n : ℕ) (h_pos : 0 < n) (h_div : ∀ d : ℕ, d ∣ n → d + 1 ∣ n + 1) : 
  ∃ p : ℕ, Prime p ∧ n = p ∧ ¬ Even p :=
sorry

end NUMINAMATH_GPT_OddPrimeDivisorCondition_l244_24422


namespace NUMINAMATH_GPT_muffin_is_twice_as_expensive_as_banana_l244_24404

variable (m b : ℚ)
variable (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12)
variable (h2 : 3 * m + 5 * b = S)

theorem muffin_is_twice_as_expensive_as_banana (h1 : 4 * m + 10 * b = 3 * m + 5 * b + 12) : m = 2 * b :=
by
  sorry

end NUMINAMATH_GPT_muffin_is_twice_as_expensive_as_banana_l244_24404


namespace NUMINAMATH_GPT_price_of_first_doughnut_l244_24460

theorem price_of_first_doughnut 
  (P : ℕ)  -- Price of the first doughnut
  (total_doughnuts : ℕ := 48)  -- Total number of doughnuts
  (price_per_dozen : ℕ := 6)  -- Price per dozen of additional doughnuts
  (total_cost : ℕ := 24)  -- Total cost spent
  (doughnuts_left : ℕ := total_doughnuts - 1)  -- Doughnuts left after the first one
  (dozens : ℕ := doughnuts_left / 12)  -- Number of whole dozens
  (cost_of_dozens : ℕ := dozens * price_per_dozen)  -- Cost of the dozens of doughnuts
  (cost_after_first : ℕ := total_cost - cost_of_dozens)  -- Remaining cost after dozens
  : P = 6 := 
by
  -- Proof to be filled in
  sorry

end NUMINAMATH_GPT_price_of_first_doughnut_l244_24460


namespace NUMINAMATH_GPT_tank_emptying_time_l244_24467

theorem tank_emptying_time
  (initial_volume : ℝ)
  (filling_rate : ℝ)
  (emptying_rate : ℝ)
  (initial_fraction_full : initial_volume = 1 / 5)
  (pipe_a_rate : filling_rate = 1 / 10)
  (pipe_b_rate : emptying_rate = 1 / 6) :
  (initial_volume / (filling_rate - emptying_rate) = 3) :=
by
  sorry

end NUMINAMATH_GPT_tank_emptying_time_l244_24467


namespace NUMINAMATH_GPT_circle_equation_AB_diameter_l244_24433

theorem circle_equation_AB_diameter (A B : ℝ × ℝ) :
  A = (1, -4) → B = (-5, 4) →
  ∃ C : ℝ × ℝ, C = (-2, 0) ∧ ∃ r : ℝ, r = 5 ∧ (∀ x y : ℝ, (x + 2)^2 + y^2 = 25) :=
by intros h1 h2; sorry

end NUMINAMATH_GPT_circle_equation_AB_diameter_l244_24433


namespace NUMINAMATH_GPT_correct_calculation_l244_24463

theorem correct_calculation (a b m : ℤ) : 
  (¬((a^3)^2 = a^5)) ∧ ((-2 * m^3)^2 = 4 * m^6) ∧ (¬(a^6 / a^2 = a^3)) ∧ (¬((a + b)^2 = a^2 + b^2)) := 
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l244_24463


namespace NUMINAMATH_GPT_exist_two_quadrilaterals_l244_24469

-- Define the structure of a quadrilateral with four sides and two diagonals
structure Quadrilateral :=
  (s1 : ℝ) -- side 1
  (s2 : ℝ) -- side 2
  (s3 : ℝ) -- side 3
  (s4 : ℝ) -- side 4
  (d1 : ℝ) -- diagonal 1
  (d2 : ℝ) -- diagonal 2

-- The theorem stating the existence of two quadrilaterals satisfying the given conditions
theorem exist_two_quadrilaterals :
  ∃ (quad1 quad2 : Quadrilateral),
  quad1.s1 < quad2.s1 ∧ quad1.s2 < quad2.s2 ∧ quad1.s3 < quad2.s3 ∧ quad1.s4 < quad2.s4 ∧
  quad1.d1 > quad2.d1 ∧ quad1.d2 > quad2.d2 :=
by
  sorry

end NUMINAMATH_GPT_exist_two_quadrilaterals_l244_24469


namespace NUMINAMATH_GPT_equilateral_triangle_sum_l244_24401

theorem equilateral_triangle_sum (x y : ℕ) (h1 : x + 5 = 14) (h2 : y + 11 = 14) : x + y = 12 :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_sum_l244_24401


namespace NUMINAMATH_GPT_evaluate_pow_l244_24476

theorem evaluate_pow : (-64 : ℝ)^(4/3) = 256 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_pow_l244_24476


namespace NUMINAMATH_GPT_geometric_sequence_a5_l244_24432

theorem geometric_sequence_a5 (a : ℕ → ℝ) (q : ℝ) 
  (h₀ : ∀ n, a n + a (n + 1) = 3 * (1 / 2) ^ n)
  (h₁ : ∀ n, a (n + 1) = a n * q)
  (h₂ : q = 1 / 2) :
  a 5 = 1 / 16 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a5_l244_24432


namespace NUMINAMATH_GPT_locus_of_midpoint_l244_24448

open Real

noncomputable def circumcircle_eq (A B C : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let a := 1
  let b := 3
  let r2 := 5
  (a, b, r2)

theorem locus_of_midpoint (A B C N : ℝ × ℝ) :
  N = (6, 2) ∧ A = (0, 1) ∧ B = (2, 1) ∧ C = (3, 4) → 
  let P := (7 / 2, 5 / 2)
  let r2 := 5 / 4
  ∃ x y : ℝ, 
  (x, y) = P ∧ (x - 7 / 2)^2 + (y - 5 / 2)^2 = r2 :=
by sorry

end NUMINAMATH_GPT_locus_of_midpoint_l244_24448


namespace NUMINAMATH_GPT_consecutive_integers_equality_l244_24416

theorem consecutive_integers_equality (n : ℕ) (h_eq : (n - 3) + (n - 2) + (n - 1) + n = (n + 1) + (n + 2) + (n + 3)) : n = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_consecutive_integers_equality_l244_24416


namespace NUMINAMATH_GPT_eval_neg64_pow_two_thirds_l244_24424

theorem eval_neg64_pow_two_thirds : (-64 : Real)^(2/3) = 16 := 
by 
  sorry

end NUMINAMATH_GPT_eval_neg64_pow_two_thirds_l244_24424


namespace NUMINAMATH_GPT_simplify_and_find_ratio_l244_24445

theorem simplify_and_find_ratio (m : ℤ) (c d : ℤ) (h : (5 * m + 15) / 5 = c * m + d) : d / c = 3 := by
  sorry

end NUMINAMATH_GPT_simplify_and_find_ratio_l244_24445


namespace NUMINAMATH_GPT_john_drinks_42_quarts_per_week_l244_24470

def gallons_per_day : ℝ := 1.5
def quarts_per_gallon : ℝ := 4
def days_per_week : ℕ := 7

theorem john_drinks_42_quarts_per_week :
  gallons_per_day * quarts_per_gallon * days_per_week = 42 := sorry

end NUMINAMATH_GPT_john_drinks_42_quarts_per_week_l244_24470


namespace NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l244_24419

theorem smallest_n_for_terminating_decimal :
  ∃ (n : ℕ), 0 < n ∧ ∀ m : ℕ, (0 < m ∧ m < n+53) → (∃ a b : ℕ, n + 53 = 2^a * 5^b) → n = 11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_terminating_decimal_l244_24419


namespace NUMINAMATH_GPT_total_cost_is_15_75_l244_24499

def price_sponge : ℝ := 4.20
def price_shampoo : ℝ := 7.60
def price_soap : ℝ := 3.20
def tax_rate : ℝ := 0.05
def total_cost_before_tax : ℝ := price_sponge + price_shampoo + price_soap
def tax_amount : ℝ := tax_rate * total_cost_before_tax
def total_cost_including_tax : ℝ := total_cost_before_tax + tax_amount

theorem total_cost_is_15_75 : total_cost_including_tax = 15.75 :=
by sorry

end NUMINAMATH_GPT_total_cost_is_15_75_l244_24499


namespace NUMINAMATH_GPT_Martha_should_buy_84oz_of_apples_l244_24459

theorem Martha_should_buy_84oz_of_apples 
  (apple_weight : ℕ)
  (orange_weight : ℕ)
  (bag_capacity : ℕ)
  (num_bags : ℕ)
  (equal_fruits : Prop) 
  (total_weight : ℕ :=
    num_bags * bag_capacity)
  (pair_weight : ℕ :=
    apple_weight + orange_weight)
  (num_pairs : ℕ :=
    total_weight / pair_weight)
  (total_apple_weight : ℕ :=
    num_pairs * apple_weight) :
  apple_weight = 4 → 
  orange_weight = 3 → 
  bag_capacity = 49 → 
  num_bags = 3 → 
  equal_fruits → 
  total_apple_weight = 84 := 
by sorry

end NUMINAMATH_GPT_Martha_should_buy_84oz_of_apples_l244_24459
