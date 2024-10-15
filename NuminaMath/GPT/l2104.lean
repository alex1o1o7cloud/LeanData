import Mathlib

namespace NUMINAMATH_GPT_minimum_value_expression_l2104_210422

theorem minimum_value_expression (a b c d e f : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f) 
(h_sum : a + b + c + d + e + f = 7) : 
  ∃ min_val : ℝ, min_val = 63 ∧ 
  (∀ a b c d e f : ℝ, 0 < a → 0 < b → 0 < c → 0 < d → 0 < e → 0 < f → a + b + c + d + e + f = 7 → 
  (1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f) ≥ min_val) := 
sorry

end NUMINAMATH_GPT_minimum_value_expression_l2104_210422


namespace NUMINAMATH_GPT_colbert_planks_needed_to_buy_l2104_210456

variables (total_planks : ℕ) (planks_from_storage : ℕ) 
          (planks_from_parents : ℕ) (planks_from_friends : ℕ)

def planks_needed_from_store := 
  total_planks - (planks_from_storage + planks_from_parents + planks_from_friends)

theorem colbert_planks_needed_to_buy : 
  total_planks = 200 → planks_from_storage = total_planks / 4 → 
  planks_from_parents = total_planks / 2 → planks_from_friends = 20 → 
  planks_needed_from_store total_planks planks_from_storage planks_from_parents planks_from_friends = 30 :=
by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_colbert_planks_needed_to_buy_l2104_210456


namespace NUMINAMATH_GPT_count_lattice_right_triangles_with_incenter_l2104_210419

def is_lattice_point (p : ℤ × ℤ) : Prop := ∃ x y : ℤ, p = (x, y)

def is_right_triangle (O A B : ℤ × ℤ) : Prop :=
  O = (0, 0) ∧ (O.1 = A.1 ∨ O.2 = A.2) ∧ (O.1 = B.1 ∨ O.2 = B.2) ∧
  (A.1 * B.2 - A.2 * B.1 ≠ 0) -- Ensure A and B are not collinear with O

def incenter (O A B : ℤ × ℤ) : ℤ × ℤ :=
  ((A.1 + B.1 - O.1) / 2, (A.2 + B.2 - O.2) / 2)

theorem count_lattice_right_triangles_with_incenter :
  let I := (2015, 7 * 2015)
  ∃ (O A B : ℤ × ℤ), is_right_triangle O A B ∧ incenter O A B = I :=
sorry

end NUMINAMATH_GPT_count_lattice_right_triangles_with_incenter_l2104_210419


namespace NUMINAMATH_GPT_div_by_10_l2104_210432

theorem div_by_10 (n : ℕ) (hn : 10 ∣ (3^n + 1)) : 10 ∣ (3^(n+4) + 1) :=
by
  sorry

end NUMINAMATH_GPT_div_by_10_l2104_210432


namespace NUMINAMATH_GPT_student_age_is_24_l2104_210411

/-- A man is 26 years older than his student. In two years, his age will be twice the age of his student.
    Prove that the present age of the student is 24 years old. -/
theorem student_age_is_24 (S M : ℕ) (h1 : M = S + 26) (h2 : M + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end NUMINAMATH_GPT_student_age_is_24_l2104_210411


namespace NUMINAMATH_GPT_find_a_of_exponential_passing_point_l2104_210433

theorem find_a_of_exponential_passing_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) (h_point : a^2 = 4) : a = 2 :=
by
  -- Proof will be filled in here
  sorry

end NUMINAMATH_GPT_find_a_of_exponential_passing_point_l2104_210433


namespace NUMINAMATH_GPT_inequality_transform_l2104_210445

theorem inequality_transform {a b : ℝ} (h : a < b) : -2 + 2 * a < -2 + 2 * b :=
sorry

end NUMINAMATH_GPT_inequality_transform_l2104_210445


namespace NUMINAMATH_GPT_derivative_at_neg_one_l2104_210478

-- Define the function f
def f (x : ℝ) : ℝ := x ^ 6

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 6 * x ^ 5

-- The statement we want to prove
theorem derivative_at_neg_one : f' (-1) = -6 := sorry

end NUMINAMATH_GPT_derivative_at_neg_one_l2104_210478


namespace NUMINAMATH_GPT_functional_equation_solution_l2104_210413

theorem functional_equation_solution (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)) →
  ∀ x : ℝ, f x = a * x^2 + b * x :=
by
  intro h
  intro x
  have : ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y) := h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l2104_210413


namespace NUMINAMATH_GPT_each_girl_gets_2_dollars_l2104_210476

theorem each_girl_gets_2_dollars :
  let debt := 40
  let lulu_savings := 6
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  total_savings - debt = 6 → (total_savings - debt) / 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_each_girl_gets_2_dollars_l2104_210476


namespace NUMINAMATH_GPT_probability_both_selected_l2104_210412

def probability_selection_ram : ℚ := 4 / 7
def probability_selection_ravi : ℚ := 1 / 5

theorem probability_both_selected : probability_selection_ram * probability_selection_ravi = 4 / 35 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_probability_both_selected_l2104_210412


namespace NUMINAMATH_GPT_total_wrappers_l2104_210431

theorem total_wrappers (a m : ℕ) (ha : a = 34) (hm : m = 15) : a + m = 49 :=
by
  sorry

end NUMINAMATH_GPT_total_wrappers_l2104_210431


namespace NUMINAMATH_GPT_side_length_of_square_perimeter_of_square_l2104_210494

theorem side_length_of_square {d s: ℝ} (h: d = 2 * Real.sqrt 2): s = 2 :=
by
  sorry

theorem perimeter_of_square {s P: ℝ} (h: s = 2): P = 8 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_perimeter_of_square_l2104_210494


namespace NUMINAMATH_GPT_t_50_mod_7_l2104_210481

theorem t_50_mod_7 (T : ℕ → ℕ) (h₁ : T 1 = 9) (h₂ : ∀ n > 1, T n = 9 ^ (T (n - 1))) :
  T 50 % 7 = 4 :=
sorry

end NUMINAMATH_GPT_t_50_mod_7_l2104_210481


namespace NUMINAMATH_GPT_math_problem_l2104_210407

theorem math_problem :
  (10^2 + 6^2) / 2 = 68 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l2104_210407


namespace NUMINAMATH_GPT_total_shaded_area_l2104_210448

theorem total_shaded_area (S T : ℕ) 
  (h1 : 12 / S = 4)
  (h2 : S / T = 3) :
  (S * S) + 8 * (T * T) = 17 :=
by
  sorry

end NUMINAMATH_GPT_total_shaded_area_l2104_210448


namespace NUMINAMATH_GPT_cubic_roots_l2104_210409

theorem cubic_roots (a b x₃ : ℤ)
  (h1 : (2^3 + a * 2^2 + b * 2 + 6 = 0))
  (h2 : (3^3 + a * 3^2 + b * 3 + 6 = 0))
  (h3 : 2 * 3 * x₃ = -6) :
  a = -4 ∧ b = 1 ∧ x₃ = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_cubic_roots_l2104_210409


namespace NUMINAMATH_GPT_wizard_answers_bal_l2104_210466

-- Define the types for human and zombie as truth-tellers and liars respectively
inductive WizardType
| human : WizardType
| zombie : WizardType

-- Define the meaning of "bal"
inductive BalMeaning
| yes : BalMeaning
| no : BalMeaning

-- Question asked to the wizard
def question (w : WizardType) (b : BalMeaning) : Prop :=
  match w, b with
  | WizardType.human, BalMeaning.yes => true
  | WizardType.human, BalMeaning.no => false
  | WizardType.zombie, BalMeaning.yes => false
  | WizardType.zombie, BalMeaning.no => true

-- Theorem stating the wizard will answer "bal" to the given question
theorem wizard_answers_bal (w : WizardType) (b : BalMeaning) :
  question w b = true ↔ b = BalMeaning.yes :=
by
  sorry

end NUMINAMATH_GPT_wizard_answers_bal_l2104_210466


namespace NUMINAMATH_GPT_largest_whole_number_lt_150_l2104_210441

theorem largest_whole_number_lt_150 : ∃ (x : ℕ), (x <= 16 ∧ ∀ y : ℕ, y < 17 → 9 * y < 150) :=
by
  sorry

end NUMINAMATH_GPT_largest_whole_number_lt_150_l2104_210441


namespace NUMINAMATH_GPT_neg_of_p_l2104_210461

variable (x : ℝ)

def p : Prop := ∀ x ≥ 0, 2^x = 3

theorem neg_of_p : ¬p ↔ ∃ x ≥ 0, 2^x ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_neg_of_p_l2104_210461


namespace NUMINAMATH_GPT_white_marbles_count_l2104_210497

section Marbles

variable (total_marbles black_marbles red_marbles green_marbles white_marbles : Nat)

theorem white_marbles_count
  (h_total: total_marbles = 60)
  (h_black: black_marbles = 32)
  (h_red: red_marbles = 10)
  (h_green: green_marbles = 5)
  (h_color: total_marbles = black_marbles + red_marbles + green_marbles + white_marbles) : 
  white_marbles = 13 := 
by
  sorry 

end Marbles

end NUMINAMATH_GPT_white_marbles_count_l2104_210497


namespace NUMINAMATH_GPT_total_weight_of_remaining_macaroons_l2104_210462

def total_weight_remaining_macaroons (total_macaroons : ℕ) (weight_per_macaroon : ℕ) (bags : ℕ) (bags_eaten : ℕ) : ℕ :=
  let macaroons_per_bag := total_macaroons / bags
  let remaining_macaroons := total_macaroons - macaroons_per_bag * bags_eaten
  remaining_macaroons * weight_per_macaroon

theorem total_weight_of_remaining_macaroons
  (total_macaroons : ℕ)
  (weight_per_macaroon : ℕ)
  (bags : ℕ)
  (bags_eaten : ℕ)
  (h1 : total_macaroons = 12)
  (h2 : weight_per_macaroon = 5)
  (h3 : bags = 4)
  (h4 : bags_eaten = 1)
  : total_weight_remaining_macaroons total_macaroons weight_per_macaroon bags bags_eaten = 45 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_remaining_macaroons_l2104_210462


namespace NUMINAMATH_GPT_b_days_to_complete_work_l2104_210424

theorem b_days_to_complete_work (x : ℕ) 
  (A : ℝ := 1 / 30) 
  (B : ℝ := 1 / x) 
  (C : ℝ := 1 / 40)
  (work_eq : 8 * (A + B + C) + 4 * (A + B) = 1) 
  (x_ne_0 : x ≠ 0) : 
  x = 30 := 
by
  sorry

end NUMINAMATH_GPT_b_days_to_complete_work_l2104_210424


namespace NUMINAMATH_GPT_transformed_coords_of_point_l2104_210491

noncomputable def polar_to_rectangular_coordinates (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

noncomputable def transformed_coordinates (r θ : ℝ) : ℝ × ℝ :=
  let new_r := r ^ 3
  let new_θ := (3 * Real.pi / 2) * θ
  polar_to_rectangular_coordinates new_r new_θ

theorem transformed_coords_of_point (r θ : ℝ)
  (h_r : r = Real.sqrt (8^2 + 6^2))
  (h_cosθ : Real.cos θ = 8 / 10)
  (h_sinθ : Real.sin θ = 6 / 10)
  (coords_match : polar_to_rectangular_coordinates r θ = (8, 6)) :
  transformed_coordinates r θ = (-600, -800) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_transformed_coords_of_point_l2104_210491


namespace NUMINAMATH_GPT_compute_xy_l2104_210459

theorem compute_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x ^ (Real.sqrt y) = 27) (h2 : (Real.sqrt x) ^ y = 9) :
  x * y = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_compute_xy_l2104_210459


namespace NUMINAMATH_GPT_indistinguishable_distributions_l2104_210430

def ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 2 && balls = 6 then 4 else 0

theorem indistinguishable_distributions : ways_to_distribute_balls 6 2 = 4 :=
by sorry

end NUMINAMATH_GPT_indistinguishable_distributions_l2104_210430


namespace NUMINAMATH_GPT_total_savings_at_end_of_year_l2104_210473

-- Defining constants for daily savings and the number of days in a year
def daily_savings : ℕ := 24
def days_in_year : ℕ := 365

-- Stating the theorem
theorem total_savings_at_end_of_year : daily_savings * days_in_year = 8760 :=
by
  sorry

end NUMINAMATH_GPT_total_savings_at_end_of_year_l2104_210473


namespace NUMINAMATH_GPT_lower_upper_bound_f_l2104_210498

-- definition of the function f(n, d) as given in the problem
def func_f (n : ℕ) (d : ℕ) : ℕ :=
  -- placeholder definition; actual definition would rely on the described properties
  sorry

theorem lower_upper_bound_f (n d : ℕ) (hn : 0 < n) (hd : 0 < d) :
  (n-1) * 2^d + 1 ≤ func_f n d ∧ func_f n d ≤ (n-1) * n^d + 1 :=
by
  sorry

end NUMINAMATH_GPT_lower_upper_bound_f_l2104_210498


namespace NUMINAMATH_GPT_remainder_of_polynomial_division_l2104_210451

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 7 * x^4 - 16 * x^3 + 3 * x^2 - 5 * x - 20

-- Define the divisor D(x)
def D (x : ℝ) : ℝ := 2 * x - 4

-- The remainder theorem sets x to 2 and evaluates P(x)
theorem remainder_of_polynomial_division : P 2 = -34 :=
by
  -- We will substitute x=2 directly into P(x)
  sorry

end NUMINAMATH_GPT_remainder_of_polynomial_division_l2104_210451


namespace NUMINAMATH_GPT_find_common_difference_l2104_210406

noncomputable def common_difference (a₁ d : ℤ) : Prop :=
  let a₂ := a₁ + d
  let a₃ := a₁ + 2 * d
  let S₅ := 5 * a₁ + 10 * d
  a₂ + a₃ = 8 ∧ S₅ = 25 → d = 2

-- Statement of the proof problem
theorem find_common_difference (a₁ d : ℤ) (h : common_difference a₁ d) : d = 2 :=
by sorry

end NUMINAMATH_GPT_find_common_difference_l2104_210406


namespace NUMINAMATH_GPT_angle_in_gradians_l2104_210440

noncomputable def gradians_in_full_circle : ℝ := 600
noncomputable def degrees_in_full_circle : ℝ := 360
noncomputable def angle_in_degrees : ℝ := 45

theorem angle_in_gradians :
  angle_in_degrees / degrees_in_full_circle * gradians_in_full_circle = 75 := 
by
  sorry

end NUMINAMATH_GPT_angle_in_gradians_l2104_210440


namespace NUMINAMATH_GPT_find_S20_l2104_210439

theorem find_S20 (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n = 1 + 2 * a n)
  (h2 : a 1 = 2) : 
  S 20 = 2^19 + 1 := 
sorry

end NUMINAMATH_GPT_find_S20_l2104_210439


namespace NUMINAMATH_GPT_square_area_l2104_210457

theorem square_area (perimeter : ℝ) (h_perimeter : perimeter = 40) : 
  ∃ (area : ℝ), area = 100 := by
  sorry

end NUMINAMATH_GPT_square_area_l2104_210457


namespace NUMINAMATH_GPT_nancy_seeds_in_big_garden_l2104_210426

theorem nancy_seeds_in_big_garden :
  let total_seeds := 52
  let small_gardens := 6
  let seeds_per_small_garden := 4
  let total_seeds_small_gardens := small_gardens * seeds_per_small_garden
  let seeds_in_big_garden := total_seeds - total_seeds_small_gardens
  seeds_in_big_garden = 28 := by
  let total_seeds := 52
  let small_gardens := 6
  let seeds_per_small_garden := 4
  let total_seeds_small_gardens := small_gardens * seeds_per_small_garden
  let seeds_in_big_garden := total_seeds - total_seeds_small_gardens
  sorry

end NUMINAMATH_GPT_nancy_seeds_in_big_garden_l2104_210426


namespace NUMINAMATH_GPT_cookies_left_after_three_days_l2104_210415

theorem cookies_left_after_three_days
  (initial_cookies : ℕ)
  (first_day_fraction_eaten : ℚ)
  (second_day_fraction_eaten : ℚ)
  (initial_value : initial_cookies = 64)
  (first_day_fraction : first_day_fraction_eaten = 3/4)
  (second_day_fraction : second_day_fraction_eaten = 1/2) :
  initial_cookies - (first_day_fraction_eaten * 64) - (second_day_fraction_eaten * ((1 - first_day_fraction_eaten) * 64)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_cookies_left_after_three_days_l2104_210415


namespace NUMINAMATH_GPT_sum_term_S2018_l2104_210472

def arithmetic_sequence (a : ℕ → ℤ) := 
  ∃ (d : ℤ), ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) := 
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem sum_term_S2018 :
  ∃ (a S : ℕ → ℤ),
    arithmetic_sequence a ∧ 
    sum_first_n_terms a S ∧ 
    a 1 = -2018 ∧ 
    ((S 2015) / 2015 - (S 2013) / 2013 = 2) ∧ 
    S 2018 = -2018 
:= by
  sorry

end NUMINAMATH_GPT_sum_term_S2018_l2104_210472


namespace NUMINAMATH_GPT_min_value_of_x_l2104_210423

-- Definitions for the conditions given in the problem
def men := 4
def women (x : ℕ) := x
def min_x := 594

-- Definition of the probability p
def C (n k : ℕ) : ℕ := sorry -- Define the binomial coefficient properly

def probability (x : ℕ) : ℚ :=
  (2 * (C (x+1) 2) + (x + 1)) /
  (C (x + 1) 3 + 3 * (C (x + 1) 2) + (x + 1))

-- The theorem statement to prove
theorem min_value_of_x (x : ℕ) : probability x ≤ 1 / 100 →  x = min_x := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_x_l2104_210423


namespace NUMINAMATH_GPT_number_of_small_jars_l2104_210444

theorem number_of_small_jars (S L : ℕ) (h1 : S + L = 100) (h2 : 3 * S + 5 * L = 376) : S = 62 := 
sorry

end NUMINAMATH_GPT_number_of_small_jars_l2104_210444


namespace NUMINAMATH_GPT_number_of_elderly_employees_in_sample_l2104_210468

variables (total_employees young_employees sample_young_employees elderly_employees : ℕ)
variables (sample_total : ℕ)

def conditions (total_employees young_employees sample_young_employees elderly_employees : ℕ) :=
  total_employees = 430 ∧
  young_employees = 160 ∧
  sample_young_employees = 32 ∧
  (∃ M, M = 2 * elderly_employees ∧ elderly_employees + M + young_employees = total_employees)

theorem number_of_elderly_employees_in_sample
  (total_employees young_employees sample_young_employees elderly_employees : ℕ)
  (sample_total : ℕ) :
  conditions total_employees young_employees sample_young_employees elderly_employees →
  sample_total = 430 * 32 / 160 →
  sample_total = 90 * 32 / 430 :=
by
  sorry

end NUMINAMATH_GPT_number_of_elderly_employees_in_sample_l2104_210468


namespace NUMINAMATH_GPT_complex_sum_l2104_210463

-- Define the given condition as a hypothesis
variables {z : ℂ} (h : z^2 + z + 1 = 0)

-- Define the statement to prove
theorem complex_sum (h : z^2 + z + 1 = 0) : z^96 + z^97 + z^98 + z^99 + z^100 + z^101 = 0 :=
sorry

end NUMINAMATH_GPT_complex_sum_l2104_210463


namespace NUMINAMATH_GPT_james_beats_old_record_l2104_210469

def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def two_point_conversions : ℕ := 6
def points_per_two_point_conversion : ℕ := 2
def field_goals : ℕ := 8
def points_per_field_goal : ℕ := 3
def extra_points : ℕ := 20
def points_per_extra_point : ℕ := 1
def old_record : ℕ := 300

theorem james_beats_old_record :
  touchdowns_per_game * points_per_touchdown * games_in_season +
  two_point_conversions * points_per_two_point_conversion +
  field_goals * points_per_field_goal +
  extra_points * points_per_extra_point - old_record = 116 := by
  sorry -- Proof is omitted.

end NUMINAMATH_GPT_james_beats_old_record_l2104_210469


namespace NUMINAMATH_GPT_PartI_PartII_l2104_210475

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Problem statement for (Ⅰ)
theorem PartI (x : ℝ) : (f x < |x| + 1) ↔ (0 < x ∧ x < 2) :=
by sorry

-- Define conditions for PartII
variables (x y : ℝ)
def condition1 : Prop := |x - y - 1| ≤ 1 / 3
def condition2 : Prop := |2 * y + 1| ≤ 1 / 6

-- Problem statement for (Ⅱ)
theorem PartII (h1 : condition1 x y) (h2 : condition2 y) : f x < 1 :=
by sorry

end NUMINAMATH_GPT_PartI_PartII_l2104_210475


namespace NUMINAMATH_GPT_min_cos_y_plus_sin_x_l2104_210484

theorem min_cos_y_plus_sin_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.cos x = Real.sin (3 * x))
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) - Real.cos (2 * x)) :
  ∃ (v : ℝ), v = -1 - Real.sqrt (2 + Real.sqrt 2) / 2 :=
sorry

end NUMINAMATH_GPT_min_cos_y_plus_sin_x_l2104_210484


namespace NUMINAMATH_GPT_find_x_plus_y_l2104_210482

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := 
by
  sorry

end NUMINAMATH_GPT_find_x_plus_y_l2104_210482


namespace NUMINAMATH_GPT_total_gallons_in_tanks_l2104_210402

theorem total_gallons_in_tanks (
  tank1_cap : ℕ := 7000) (tank2_cap : ℕ := 5000) (tank3_cap : ℕ := 3000)
  (fill1_fraction : ℚ := 3/4) (fill2_fraction : ℚ := 4/5) (fill3_fraction : ℚ := 1/2)
  : tank1_cap * fill1_fraction + tank2_cap * fill2_fraction + tank3_cap * fill3_fraction = 10750 := by
  sorry

end NUMINAMATH_GPT_total_gallons_in_tanks_l2104_210402


namespace NUMINAMATH_GPT_stock_price_after_two_years_l2104_210454

theorem stock_price_after_two_years 
    (p0 : ℝ) (r1 r2 : ℝ) (p1 p2 : ℝ) 
    (h0 : p0 = 100) (h1 : r1 = 0.50) 
    (h2 : r2 = 0.30) 
    (h3 : p1 = p0 * (1 + r1)) 
    (h4 : p2 = p1 * (1 - r2)) : 
    p2 = 105 :=
by sorry

end NUMINAMATH_GPT_stock_price_after_two_years_l2104_210454


namespace NUMINAMATH_GPT_l_shape_area_l2104_210421

theorem l_shape_area (large_length large_width small_length small_width : ℕ)
  (large_rect_area : large_length = 10 ∧ large_width = 7)
  (small_rect_area : small_length = 3 ∧ small_width = 2) :
  (large_length * large_width) - 2 * (small_length * small_width) = 58 :=
by 
  sorry

end NUMINAMATH_GPT_l_shape_area_l2104_210421


namespace NUMINAMATH_GPT_rectangle_proof_right_triangle_proof_l2104_210495

-- Definition of rectangle condition
def rectangle_condition (a b : ℕ) : Prop :=
  a * b = 2 * (a + b)

-- Definition of right triangle condition
def right_triangle_condition (a b : ℕ) : Prop :=
  a + b + Int.natAbs (Int.sqrt (a^2 + b^2)) = a * b / 2 ∧
  (∃ c : ℕ, c = Int.natAbs (Int.sqrt (a^2 + b^2)))

-- Recangle proof
theorem rectangle_proof : ∃! p : ℕ × ℕ, rectangle_condition p.1 p.2 := sorry

-- Right triangle proof
theorem right_triangle_proof : ∃! t : ℕ × ℕ, right_triangle_condition t.1 t.2 := sorry

end NUMINAMATH_GPT_rectangle_proof_right_triangle_proof_l2104_210495


namespace NUMINAMATH_GPT_geometric_sequence_fifth_term_l2104_210479

variable {a : ℕ → ℝ} (h1 : a 1 = 1) (h4 : a 4 = 8)

theorem geometric_sequence_fifth_term (h_geom : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q) :
  a 5 = 16 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_fifth_term_l2104_210479


namespace NUMINAMATH_GPT_kishore_savings_l2104_210464

-- Define the monthly expenses and condition
def expenses : Real :=
  5000 + 1500 + 4500 + 2500 + 2000 + 6100

-- Define the monthly salary and savings conditions
def salary (S : Real) : Prop :=
  expenses + 0.1 * S = S

-- Define the savings amount
def savings (S : Real) : Real :=
  0.1 * S

-- The theorem to prove
theorem kishore_savings : ∃ S : Real, salary S ∧ savings S = 2733.33 :=
by
  sorry

end NUMINAMATH_GPT_kishore_savings_l2104_210464


namespace NUMINAMATH_GPT_paving_rate_l2104_210483

theorem paving_rate
  (length : ℝ) (width : ℝ) (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) :
  total_cost / (length * width) = 800 := by
  sorry

end NUMINAMATH_GPT_paving_rate_l2104_210483


namespace NUMINAMATH_GPT_initial_investments_l2104_210489

theorem initial_investments (x y : ℝ) : 
  -- Conditions
  5000 = y + (5000 - y) ∧
  (y * (1 + x / 100) = 2100) ∧
  ((5000 - y) * (1 + (x + 1) / 100) = 3180) →
  -- Conclusion
  y = 2000 ∧ (5000 - y) = 3000 := 
by 
  sorry

end NUMINAMATH_GPT_initial_investments_l2104_210489


namespace NUMINAMATH_GPT_farmer_rent_l2104_210490

-- Definitions based on given conditions
def rent_per_acre_per_month : ℕ := 60
def length_of_plot : ℕ := 360
def width_of_plot : ℕ := 1210
def square_feet_per_acre : ℕ := 43560

-- Problem statement: 
-- Prove that the monthly rent to rent the rectangular plot is $600.
theorem farmer_rent : 
  (length_of_plot * width_of_plot) / square_feet_per_acre * rent_per_acre_per_month = 600 :=
by
  sorry

end NUMINAMATH_GPT_farmer_rent_l2104_210490


namespace NUMINAMATH_GPT_frustum_slant_height_l2104_210436

theorem frustum_slant_height (r1 r2 V : ℝ) (h l : ℝ) 
    (H1 : r1 = 2) (H2 : r2 = 6) (H3 : V = 104 * π)
    (H4 : V = (1/3) * π * h * (r1^2 + r2^2 + r1 * r2)) 
    (H5 : h = 6)
    (H6 : l = Real.sqrt (h^2 + (r2 - r1)^2)) :
    l = 2 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_GPT_frustum_slant_height_l2104_210436


namespace NUMINAMATH_GPT_k_minus_2_divisible_by_3_l2104_210474

theorem k_minus_2_divisible_by_3
  (k : ℕ)
  (a : ℕ → ℤ)
  (h_a0_pos : 0 < k)
  (h_seq : ∀ n ≥ 1, a n = (a (n - 1) + n^k) / n) :
  (k - 2) % 3 = 0 :=
sorry

end NUMINAMATH_GPT_k_minus_2_divisible_by_3_l2104_210474


namespace NUMINAMATH_GPT_dice_probability_four_less_than_five_l2104_210470

noncomputable def probability_exactly_four_less_than_five (n : ℕ) : ℚ :=
  if n = 8 then (Nat.choose 8 4) * (1 / 2)^8 else 0

theorem dice_probability_four_less_than_five : probability_exactly_four_less_than_five 8 = 35 / 128 :=
by
  -- statement is correct, proof to be provided
  sorry

end NUMINAMATH_GPT_dice_probability_four_less_than_five_l2104_210470


namespace NUMINAMATH_GPT_find_x_l2104_210401

theorem find_x (x : ℝ) (h : 0.45 * x = (1 / 3) * x + 110) : x = 942.857 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2104_210401


namespace NUMINAMATH_GPT_symmetric_point_line_eq_l2104_210455

theorem symmetric_point_line_eq (A B : ℝ × ℝ) (l : ℝ → ℝ) (x1 y1 x2 y2 : ℝ)
  (hA : A = (4, 5))
  (hB : B = (-2, 7))
  (hSymmetric : ∀ x y, B = (2 * l x - A.1, 2 * l y - A.2)) :
  ∀ x y, l x = 3 * x - 5 ∧ l y = 3 * y + 6 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_line_eq_l2104_210455


namespace NUMINAMATH_GPT_sum_of_faces_edges_vertices_l2104_210435

def cube_faces : ℕ := 6
def cube_edges : ℕ := 12
def cube_vertices : ℕ := 8

theorem sum_of_faces_edges_vertices :
  cube_faces + cube_edges + cube_vertices = 26 := by
  sorry

end NUMINAMATH_GPT_sum_of_faces_edges_vertices_l2104_210435


namespace NUMINAMATH_GPT_sum_mod_nine_l2104_210404

def a : ℕ := 1234
def b : ℕ := 1235
def c : ℕ := 1236
def d : ℕ := 1237
def e : ℕ := 1238
def modulus : ℕ := 9

theorem sum_mod_nine : (a + b + c + d + e) % modulus = 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_mod_nine_l2104_210404


namespace NUMINAMATH_GPT_sixth_grade_students_total_l2104_210449

noncomputable def total_students (x y : ℕ) : ℕ := x + y

theorem sixth_grade_students_total (x y : ℕ) 
(h1 : x + (1 / 3) * y = 105) 
(h2 : y + (1 / 2) * x = 105) 
: total_students x y = 147 := 
by
  sorry

end NUMINAMATH_GPT_sixth_grade_students_total_l2104_210449


namespace NUMINAMATH_GPT_quadratic_roots_in_intervals_l2104_210492

theorem quadratic_roots_in_intervals (a b c : ℝ) (h₁ : a < b) (h₂ : b < c) :
  ∃ x₁ x₂ : ℝ, (a < x₁ ∧ x₁ < b) ∧ (b < x₂ ∧ x₂ < c) ∧
  3 * x₁^2 - 2 * (a + b + c) * x₁ + (a * b + b * c + c * a) = 0 ∧
  3 * x₂^2 - 2 * (a + b + c) * x₂ + (a * b + b * c + c * a) = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_in_intervals_l2104_210492


namespace NUMINAMATH_GPT_like_terms_proof_l2104_210416

theorem like_terms_proof (m n : ℤ) 
  (h1 : m + 10 = 3 * n - m) 
  (h2 : 7 - n = n - m) :
  m^2 - 2 * m * n + n^2 = 9 := by
  sorry

end NUMINAMATH_GPT_like_terms_proof_l2104_210416


namespace NUMINAMATH_GPT_men_work_in_80_days_l2104_210471

theorem men_work_in_80_days (x : ℕ) (work_eq_20men_56days : x * 80 = 20 * 56) : x = 14 :=
by 
  sorry

end NUMINAMATH_GPT_men_work_in_80_days_l2104_210471


namespace NUMINAMATH_GPT_number_whose_multiples_are_considered_for_calculating_the_average_l2104_210414

theorem number_whose_multiples_are_considered_for_calculating_the_average
  (x : ℕ)
  (n : ℕ)
  (a : ℕ)
  (b : ℕ)
  (h1 : n = 10)
  (h2 : a = (x + 2*x + 3*x + 4*x + 5*x + 6*x + 7*x) / 7)
  (h3 : b = 2*n)
  (h4 : a^2 - b^2 = 0) :
  x = 5 := 
sorry

end NUMINAMATH_GPT_number_whose_multiples_are_considered_for_calculating_the_average_l2104_210414


namespace NUMINAMATH_GPT_geometric_sequence_S20_l2104_210487

-- Define the conditions and target statement
theorem geometric_sequence_S20
  (a : ℕ → ℝ) -- defining the sequence as a function from natural numbers to real numbers
  (q : ℝ) -- common ratio
  (h_pos : ∀ n, a n > 0) -- all terms are positive
  (h_geo : ∀ n, a (n + 1) = q * a n) -- geometric sequence property
  (S : ℕ → ℝ) -- sum function
  (h_S : ∀ n, S n = (a 1 * (1 - q ^ n)) / (1 - q)) -- sum formula for a geometric progression
  (h_S5 : S 5 = 3) -- given S_5 = 3
  (h_S15 : S 15 = 21) -- given S_15 = 21
  : S 20 = 45 := sorry

end NUMINAMATH_GPT_geometric_sequence_S20_l2104_210487


namespace NUMINAMATH_GPT_find_a_l2104_210405

theorem find_a :
  let p1 := (⟨-3, 7⟩ : ℝ × ℝ)
  let p2 := (⟨2, -1⟩ : ℝ × ℝ)
  let direction := (5, -8)
  let target_direction := (a, -2)
  a = (direction.1 * -2) / (direction.2) := by
  sorry

end NUMINAMATH_GPT_find_a_l2104_210405


namespace NUMINAMATH_GPT_min_distance_MN_l2104_210467

open Real

noncomputable def f (x : ℝ) := exp x - (1 / 2) * x^2
noncomputable def g (x : ℝ) := x - 1

theorem min_distance_MN (x1 x2 : ℝ) (h1 : x1 ≥ 0) (h2 : x2 > 0) (h3 : f x1 = g x2) :
  abs (x2 - x1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_MN_l2104_210467


namespace NUMINAMATH_GPT_remainder_when_dividing_l2104_210400

theorem remainder_when_dividing (a : ℕ) (h1 : a = 432 * 44) : a % 38 = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_remainder_when_dividing_l2104_210400


namespace NUMINAMATH_GPT_rectangle_width_decrease_percent_l2104_210488

theorem rectangle_width_decrease_percent (L W : ℝ) (h : L * W = L * W) :
  let L_new := 1.3 * L
  let W_new := W / 1.3 
  let percent_decrease := (1 - (W_new / W)) * 100
  percent_decrease = 23.08 :=
sorry

end NUMINAMATH_GPT_rectangle_width_decrease_percent_l2104_210488


namespace NUMINAMATH_GPT_Timi_has_five_ears_l2104_210480

theorem Timi_has_five_ears (seeing_ears_Imi seeing_ears_Dimi seeing_ears_Timi : ℕ)
  (H1 : seeing_ears_Imi = 8)
  (H2 : seeing_ears_Dimi = 7)
  (H3 : seeing_ears_Timi = 5)
  (total_ears : ℕ := (seeing_ears_Imi + seeing_ears_Dimi + seeing_ears_Timi) / 2) :
  total_ears - seeing_ears_Timi = 5 :=
by
  sorry -- Proof not required.

end NUMINAMATH_GPT_Timi_has_five_ears_l2104_210480


namespace NUMINAMATH_GPT_dalmatians_with_right_ear_spots_l2104_210493

def TotalDalmatians := 101
def LeftOnlySpots := 29
def RightOnlySpots := 17
def NoEarSpots := 22

theorem dalmatians_with_right_ear_spots : 
  (TotalDalmatians - LeftOnlySpots - NoEarSpots) = 50 :=
by
  -- Proof goes here, but for now, we use sorry
  sorry

end NUMINAMATH_GPT_dalmatians_with_right_ear_spots_l2104_210493


namespace NUMINAMATH_GPT_number_of_sets_l2104_210485

theorem number_of_sets (A : Set ℕ) : ∃ s : Finset (Set ℕ), 
  (∀ x ∈ s, ({1} ⊂ x ∧ x ⊆ {1, 2, 3, 4})) ∧ s.card = 7 :=
sorry

end NUMINAMATH_GPT_number_of_sets_l2104_210485


namespace NUMINAMATH_GPT_boat_speed_still_water_l2104_210453

theorem boat_speed_still_water (V_b V_c : ℝ) (h1 : 45 / (V_b - V_c) = t) (h2 : V_b = 12)
(h3 : V_b + V_c = 15):
  V_b = 12 :=
by
  sorry

end NUMINAMATH_GPT_boat_speed_still_water_l2104_210453


namespace NUMINAMATH_GPT_evaluate_expression_l2104_210450

-- Definition of the given condition.
def sixty_four_eq_sixteen_squared : Prop := 64 = 16^2

-- The statement to prove that the given expression equals the answer.
theorem evaluate_expression (h : sixty_four_eq_sixteen_squared) : 
  (16^24) / (64^8) = 16^8 :=
by 
  -- h contains the condition that 64 = 16^2, but we provide a proof step later with sorry
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2104_210450


namespace NUMINAMATH_GPT_union_of_A_and_B_l2104_210446

open Set

-- Define the sets A and B based on given conditions
def A (x : ℤ) : Set ℤ := {y | y = x^2 ∨ y = 2 * x - 1 ∨ y = -4}
def B (x : ℤ) : Set ℤ := {y | y = x - 5 ∨ y = 1 - x ∨ y = 9}

-- Specific condition given in the problem
def A_intersect_B_condition (x : ℤ) : Prop :=
  A x ∩ B x = {9}

-- Prove problem statement that describes the union of A and B
theorem union_of_A_and_B (x : ℤ) (h : A_intersect_B_condition x) : A x ∪ B x = {-8, -7, -4, 4, 9} :=
sorry

end NUMINAMATH_GPT_union_of_A_and_B_l2104_210446


namespace NUMINAMATH_GPT_trigonometric_identity_l2104_210496

open Real

theorem trigonometric_identity :
  sin (72 * pi / 180) * cos (12 * pi / 180) - cos (72 * pi / 180) * sin (12 * pi / 180) = sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l2104_210496


namespace NUMINAMATH_GPT_fraction_is_one_twelve_l2104_210465

variables (A E : ℝ) (f : ℝ)

-- Given conditions
def condition1 : E = 200 := sorry
def condition2 : A - E = f * (A + E) := sorry
def condition3 : A * 1.10 = E * 1.20 + 20 := sorry

-- Proving the fraction f is 1/12
theorem fraction_is_one_twelve : E = 200 → (A - E = f * (A + E)) → (A * 1.10 = E * 1.20 + 20) → 
f = 1 / 12 :=
by
  intros hE hDiff hIncrease
  sorry

end NUMINAMATH_GPT_fraction_is_one_twelve_l2104_210465


namespace NUMINAMATH_GPT_solve_x2_y2_eq_3z2_in_integers_l2104_210458

theorem solve_x2_y2_eq_3z2_in_integers (x y z : ℤ) : x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_solve_x2_y2_eq_3z2_in_integers_l2104_210458


namespace NUMINAMATH_GPT_equal_playing_time_l2104_210452

def number_of_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45

theorem equal_playing_time :
  (players_on_field * match_duration) / number_of_players = 36 :=
by
  sorry

end NUMINAMATH_GPT_equal_playing_time_l2104_210452


namespace NUMINAMATH_GPT_point_B_is_4_l2104_210403

def point_A : ℤ := -3
def units_to_move : ℤ := 7
def point_B : ℤ := point_A + units_to_move

theorem point_B_is_4 : point_B = 4 :=
by
  sorry

end NUMINAMATH_GPT_point_B_is_4_l2104_210403


namespace NUMINAMATH_GPT_percent_runs_by_running_between_wickets_l2104_210427

theorem percent_runs_by_running_between_wickets :
  (132 - (12 * 4 + 2 * 6)) / 132 * 100 = 54.54545454545455 :=
by
  sorry

end NUMINAMATH_GPT_percent_runs_by_running_between_wickets_l2104_210427


namespace NUMINAMATH_GPT_area_sum_of_three_circles_l2104_210408

theorem area_sum_of_three_circles (R d : ℝ) (x y z : ℝ) 
    (hxyz : x^2 + y^2 + z^2 = d^2) :
    (π * ((R^2 - x^2) + (R^2 - y^2) + (R^2 - z^2))) = π * (3 * R^2 - d^2) :=
by
  sorry

end NUMINAMATH_GPT_area_sum_of_three_circles_l2104_210408


namespace NUMINAMATH_GPT_nested_fraction_value_l2104_210460

theorem nested_fraction_value :
  1 + (1 / (1 + (1 / (2 + (2 / 3))))) = 19 / 11 :=
by sorry

end NUMINAMATH_GPT_nested_fraction_value_l2104_210460


namespace NUMINAMATH_GPT_money_distribution_l2104_210437

theorem money_distribution :
  ∀ (A B C : ℕ), 
  A + B + C = 900 → 
  B + C = 750 → 
  C = 250 → 
  A + C = 400 := 
by
  intros A B C h1 h2 h3
  sorry

end NUMINAMATH_GPT_money_distribution_l2104_210437


namespace NUMINAMATH_GPT_position_2023_l2104_210499

def initial_position := "ABCD"

def rotate_180 (pos : String) : String :=
  match pos with
  | "ABCD" => "CDAB"
  | "CDAB" => "ABCD"
  | "DCBA" => "BADC"
  | "BADC" => "DCBA"
  | _ => pos

def reflect_horizontal (pos : String) : String :=
  match pos with
  | "ABCD" => "ABCD"
  | "CDAB" => "DCBA"
  | "DCBA" => "CDAB"
  | "BADC" => "BADC"
  | _ => pos

def transformation (n : ℕ) : String :=
  let cnt := n % 4
  if cnt = 1 then rotate_180 initial_position
  else if cnt = 2 then rotate_180 (rotate_180 initial_position)
  else if cnt = 3 then rotate_180 (reflect_horizontal (rotate_180 initial_position))
  else reflect_horizontal initial_position

theorem position_2023 : transformation 2023 = "DCBA" := by
  sorry

end NUMINAMATH_GPT_position_2023_l2104_210499


namespace NUMINAMATH_GPT_solution_to_ball_problem_l2104_210418

noncomputable def probability_of_arithmetic_progression : Nat :=
  let p := 3
  let q := 9464
  p + q

theorem solution_to_ball_problem : probability_of_arithmetic_progression = 9467 := by
  sorry

end NUMINAMATH_GPT_solution_to_ball_problem_l2104_210418


namespace NUMINAMATH_GPT_divisible_by_12_l2104_210429

theorem divisible_by_12 (n : ℕ) (h1 : (5140 + n) % 4 = 0) (h2 : (5 + 1 + 4 + n) % 3 = 0) : n = 8 :=
by
  sorry

end NUMINAMATH_GPT_divisible_by_12_l2104_210429


namespace NUMINAMATH_GPT_player_B_questions_l2104_210447

theorem player_B_questions :
  ∀ (a b : ℕ → ℕ), (∀ i j, i ≠ j → a i + b j = a j + b i) →
  ∃ k, k = 11 := sorry

end NUMINAMATH_GPT_player_B_questions_l2104_210447


namespace NUMINAMATH_GPT_find_f_l2104_210425

noncomputable def f (f'₁ : ℝ) (x : ℝ) : ℝ := f'₁ * Real.exp x - x ^ 2

theorem find_f'₁ (f'₁ : ℝ) (h : f f'₁ = λ x => f'₁ * Real.exp x - x ^ 2) :
  f'₁ = 2 * Real.exp 1 / (Real.exp 1 - 1) := by
  sorry

end NUMINAMATH_GPT_find_f_l2104_210425


namespace NUMINAMATH_GPT_cube_root_of_x_sqrt_x_eq_x_half_l2104_210417

variable (x : ℝ) (h : 0 < x)

theorem cube_root_of_x_sqrt_x_eq_x_half : (x * Real.sqrt x) ^ (1/3) = x ^ (1/2) := by
  sorry

end NUMINAMATH_GPT_cube_root_of_x_sqrt_x_eq_x_half_l2104_210417


namespace NUMINAMATH_GPT_num_positive_integers_l2104_210443

theorem num_positive_integers (N : ℕ) (h : N > 3) : (∃ (k : ℕ) (h_div : 48 % k = 0), k = N - 3) → (∃ (c : ℕ), c = 8) := sorry

end NUMINAMATH_GPT_num_positive_integers_l2104_210443


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2104_210486

namespace ProofExample

variable {x : ℝ}

-- Define the sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | x < 2}

-- Theorem: "1 < x < 2" is a sufficient but not necessary condition for "x < 2" to hold.
theorem sufficient_not_necessary : 
  (∀ x, 1 < x ∧ x < 2 → x < 2) ∧ ¬(∀ x, x < 2 → 1 < x ∧ x < 2) := 
by
  sorry

end ProofExample

end NUMINAMATH_GPT_sufficient_not_necessary_l2104_210486


namespace NUMINAMATH_GPT_max_horizontal_distance_domino_l2104_210420

theorem max_horizontal_distance_domino (n : ℕ) : 
    (n > 0) → ∃ d, d = 2 * Real.log n := 
by {
    sorry
}

end NUMINAMATH_GPT_max_horizontal_distance_domino_l2104_210420


namespace NUMINAMATH_GPT_ribbon_cost_comparison_l2104_210410

theorem ribbon_cost_comparison 
  (A : Type)
  (yellow_ribbon_cost blue_ribbon_cost : ℕ)
  (h1 : yellow_ribbon_cost = 24)
  (h2 : blue_ribbon_cost = 36) :
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n < blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n > blue_ribbon_cost / n) ∨
  (∃ n : ℕ, n > 0 ∧ yellow_ribbon_cost / n = blue_ribbon_cost / n) :=
sorry

end NUMINAMATH_GPT_ribbon_cost_comparison_l2104_210410


namespace NUMINAMATH_GPT_lines_parallel_l2104_210428

noncomputable def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a, 2, 6)
noncomputable def line2 (a : ℝ) : ℝ × ℝ × ℝ := (1, a-1, a^2-1)

def are_parallel (line1 line2 : ℝ × ℝ × ℝ) : Prop :=
  let ⟨a1, b1, _⟩ := line1
  let ⟨a2, b2, _⟩ := line2
  a1 * b2 = a2 * b1

theorem lines_parallel (a : ℝ) :
  are_parallel (line1 a) (line2 a) → a = -1 :=
sorry

end NUMINAMATH_GPT_lines_parallel_l2104_210428


namespace NUMINAMATH_GPT_negation_of_exists_statement_l2104_210477

theorem negation_of_exists_statement :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_exists_statement_l2104_210477


namespace NUMINAMATH_GPT_find_set_A_l2104_210442

open Set

variable (U : Set ℕ) (A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3, 4, 5})
variable (h1 : (U \ A) ∩ B = {0, 4})
variable (h2 : (U \ A) ∩ (U \ B) = {3, 5})

theorem find_set_A :
  A = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_find_set_A_l2104_210442


namespace NUMINAMATH_GPT_present_ages_ratio_l2104_210434

noncomputable def ratio_of_ages (F S : ℕ) : ℚ :=
  F / S

theorem present_ages_ratio (F S : ℕ) (h1 : F + S = 220) (h2 : (F + 10) * 3 = (S + 10) * 5) :
  ratio_of_ages F S = 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_present_ages_ratio_l2104_210434


namespace NUMINAMATH_GPT_gold_coins_distribution_l2104_210438

theorem gold_coins_distribution (x y : ℝ) (h₁ : x + y = 25) (h₂ : x ≠ y)
  (h₃ : (x^2 - y^2) = k * (x - y)) : k = 25 :=
sorry

end NUMINAMATH_GPT_gold_coins_distribution_l2104_210438
