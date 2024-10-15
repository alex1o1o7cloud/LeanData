import Mathlib

namespace NUMINAMATH_GPT_distribution_ways_l694_69493

def count_distributions (n : ℕ) (k : ℕ) : ℕ :=
-- Calculation for count distributions will be implemented here
sorry

theorem distribution_ways (items bags : ℕ) (cond : items = 6 ∧ bags = 3):
  count_distributions items bags = 75 :=
by
  -- Proof would be implemented here
  sorry

end NUMINAMATH_GPT_distribution_ways_l694_69493


namespace NUMINAMATH_GPT_max_value_f_period_f_l694_69478

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos x) ^ 2 - (Real.cos x) ^ 4

theorem max_value_f : ∃ x : ℝ, (f x) = 1 / 4 :=
sorry

theorem period_f : ∃ p : ℝ, p = π / 2 ∧ ∀ x : ℝ, f (x + p) = f x :=
sorry

end NUMINAMATH_GPT_max_value_f_period_f_l694_69478


namespace NUMINAMATH_GPT_option_d_correct_l694_69429

theorem option_d_correct (a b : ℝ) : 2 * a^2 * b - 4 * a^2 * b = -2 * a^2 * b :=
by
  sorry

end NUMINAMATH_GPT_option_d_correct_l694_69429


namespace NUMINAMATH_GPT_correct_grammatical_phrase_l694_69443

-- Define the conditions as lean definitions 
def number_of_cars_produced_previous_year : ℕ := sorry  -- number of cars produced in previous year
def number_of_cars_produced_2004 : ℕ := 3 * number_of_cars_produced_previous_year  -- number of cars produced in 2004

-- Define the theorem stating the correct phrase to describe the production numbers
theorem correct_grammatical_phrase : 
  (3 * number_of_cars_produced_previous_year = number_of_cars_produced_2004) → 
  ("three times as many cars" = "three times as many cars") := 
by
  sorry

end NUMINAMATH_GPT_correct_grammatical_phrase_l694_69443


namespace NUMINAMATH_GPT_time_worked_on_thursday_l694_69483

/-
  Given:
  - Monday: 3/4 hour
  - Tuesday: 1/2 hour
  - Wednesday: 2/3 hour
  - Friday: 75 minutes
  - Total (Monday to Friday): 4 hours = 240 minutes
  
  The time Mr. Willson worked on Thursday is 50 minutes.
-/

noncomputable def time_worked_monday : ℝ := (3 / 4) * 60
noncomputable def time_worked_tuesday : ℝ := (1 / 2) * 60
noncomputable def time_worked_wednesday : ℝ := (2 / 3) * 60
noncomputable def time_worked_friday : ℝ := 75
noncomputable def total_time_worked : ℝ := 4 * 60

theorem time_worked_on_thursday :
  time_worked_monday + time_worked_tuesday + time_worked_wednesday + time_worked_friday + 50 = total_time_worked :=
by
  sorry

end NUMINAMATH_GPT_time_worked_on_thursday_l694_69483


namespace NUMINAMATH_GPT_ThaboRatio_l694_69439

-- Define the variables
variables (P_f P_nf H_nf : ℕ)

-- Define the conditions as hypotheses
def ThaboConditions := P_f + P_nf + H_nf = 280 ∧ P_nf = H_nf + 20 ∧ H_nf = 55

-- State the theorem we want to prove
theorem ThaboRatio (h : ThaboConditions P_f P_nf H_nf) : (P_f / P_nf) = 2 :=
by sorry

end NUMINAMATH_GPT_ThaboRatio_l694_69439


namespace NUMINAMATH_GPT_min_value_of_expression_l694_69479

noncomputable def minValueExpr (a b c : ℝ) : ℝ :=
  a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem min_value_of_expression (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minValueExpr a b c >= 60 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l694_69479


namespace NUMINAMATH_GPT_parking_fines_l694_69434

theorem parking_fines (total_citations littering_citations offleash_dog_citations parking_fines : ℕ) 
  (h1 : total_citations = 24) 
  (h2 : littering_citations = 4) 
  (h3 : offleash_dog_citations = 4) 
  (h4 : total_citations = littering_citations + offleash_dog_citations + parking_fines) : 
  parking_fines = 16 := 
by 
  sorry

end NUMINAMATH_GPT_parking_fines_l694_69434


namespace NUMINAMATH_GPT_min_marked_cells_l694_69470

theorem min_marked_cells (marking : Fin 15 → Fin 15 → Prop) :
  (∀ i : Fin 15, ∃ j : Fin 15, ∀ k : Fin 10, marking i (j + k % 15)) ∧
  (∀ j : Fin 15, ∃ i : Fin 15, ∀ k : Fin 10, marking (i + k % 15) j) →
  ∃s : Finset (Fin 15 × Fin 15), s.card = 20 ∧ ∀ i : Fin 15, (∃ j, (i, j) ∈ s ∨ (j, i) ∈ s) :=
sorry

end NUMINAMATH_GPT_min_marked_cells_l694_69470


namespace NUMINAMATH_GPT_remainder_two_when_divided_by_3_l694_69427

-- Define the main theorem stating that for any positive integer n,
-- n^3 + 3/2 * n^2 + 1/2 * n - 1 leaves a remainder of 2 when divided by 3.

theorem remainder_two_when_divided_by_3 (n : ℕ) (h : n > 0) : 
  (n^3 + (3 / 2) * n^2 + (1 / 2) * n - 1) % 3 = 2 := 
sorry

end NUMINAMATH_GPT_remainder_two_when_divided_by_3_l694_69427


namespace NUMINAMATH_GPT_binomial_sum_l694_69425

theorem binomial_sum (n k : ℕ) (h : n = 10) (hk : k = 3) :
  Nat.choose n k + Nat.choose n (n - k) = 240 :=
by
  -- placeholder for actual proof
  sorry

end NUMINAMATH_GPT_binomial_sum_l694_69425


namespace NUMINAMATH_GPT_least_positive_three_digit_multiple_of_8_l694_69441

theorem least_positive_three_digit_multiple_of_8 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ n % 8 = 0 ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ m % 8 = 0) → n ≤ m) ∧ n = 104 :=
by
  sorry

end NUMINAMATH_GPT_least_positive_three_digit_multiple_of_8_l694_69441


namespace NUMINAMATH_GPT_max_value_of_x3_div_y4_l694_69415

theorem max_value_of_x3_div_y4 (x y : ℝ) (h1 : 3 ≤ x * y^2) (h2 : x * y^2 ≤ 8) (h3 : 4 ≤ x^2 / y) (h4 : x^2 / y ≤ 9) :
  ∃ (k : ℝ), k = 27 ∧ ∀ (z : ℝ), z = x^3 / y^4 → z ≤ k :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_x3_div_y4_l694_69415


namespace NUMINAMATH_GPT_train_cross_platform_time_l694_69486

noncomputable def kmph_to_mps (s : ℚ) : ℚ :=
  (s * 1000) / 3600

theorem train_cross_platform_time :
  let train_length := 110
  let speed_kmph := 52
  let platform_length := 323.36799999999994
  let speed_mps := kmph_to_mps 52
  let total_distance := train_length + platform_length
  let time := total_distance / speed_mps
  time = 30 := 
by
  sorry

end NUMINAMATH_GPT_train_cross_platform_time_l694_69486


namespace NUMINAMATH_GPT_quadratic_real_roots_l694_69403

theorem quadratic_real_roots (k : ℝ) : 
  (∃ x : ℝ, (k + 1) * x^2 - 2 * x + 1 = 0) → (k ≤ 0 ∧ k ≠ -1) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l694_69403


namespace NUMINAMATH_GPT_algebraic_expression_value_l694_69457

theorem algebraic_expression_value 
  (x1 x2 : ℝ)
  (h1 : x1^2 - x1 - 2022 = 0)
  (h2 : x2^2 - x2 - 2022 = 0) :
  x1^3 - 2022 * x1 + x2^2 = 4045 :=
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l694_69457


namespace NUMINAMATH_GPT_find_m_l694_69469

theorem find_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + x - m > 0 ↔ x < -3 ∨ x > 2) → m = 6 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_m_l694_69469


namespace NUMINAMATH_GPT_seeds_in_pots_l694_69489

theorem seeds_in_pots (x : ℕ) (total_seeds : ℕ) (seeds_fourth_pot : ℕ) 
  (h1 : total_seeds = 10) (h2 : seeds_fourth_pot = 1) 
  (h3 : 3 * x + seeds_fourth_pot = total_seeds) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_seeds_in_pots_l694_69489


namespace NUMINAMATH_GPT_value_of_ab_l694_69487

theorem value_of_ab (a b : ℤ) (h1 : |a| = 5) (h2 : b = -3) (h3 : a < b) : a * b = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_ab_l694_69487


namespace NUMINAMATH_GPT_range_of_m_l694_69402

theorem range_of_m {x : ℝ} (m : ℝ) :
  (∀ x, |x - 1| + |x - 2| + |x - 3| ≥ m) ↔ m ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l694_69402


namespace NUMINAMATH_GPT_solution_set_of_inequality_l694_69430

variable (a b c : ℝ)

theorem solution_set_of_inequality 
  (h1 : a < 0)
  (h2 : b = a)
  (h3 : c = -2 * a)
  (h4 : ∀ x : ℝ, -2 < x ∧ x < 1 → ax^2 + bx + c > 0) :
  ∀ x : ℝ, (x ≤ -1 / 2 ∨ x ≥ 1) ↔ cx^2 + ax + b ≥ 0 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l694_69430


namespace NUMINAMATH_GPT_marks_lost_per_wrong_answer_l694_69446

theorem marks_lost_per_wrong_answer
    (total_questions : ℕ)
    (correct_questions : ℕ)
    (total_marks : ℕ)
    (marks_per_correct : ℕ)
    (marks_lost : ℕ)
    (x : ℕ)
    (h1 : total_questions = 60)
    (h2 : correct_questions = 44)
    (h3 : total_marks = 160)
    (h4 : marks_per_correct = 4)
    (h5 : marks_lost = 176 - total_marks)
    (h6 : marks_lost = x * (total_questions - correct_questions)) :
    x = 1 := by
  sorry

end NUMINAMATH_GPT_marks_lost_per_wrong_answer_l694_69446


namespace NUMINAMATH_GPT_m_and_n_relationship_l694_69475

-- Define the function f
def f (x m : ℝ) := x^2 - 4*x + 4 + m

-- State the conditions and required proof
theorem m_and_n_relationship (m n : ℝ) (h_domain : ∀ x, 2 ≤ x ∧ x ≤ n → 2 ≤ f x m ∧ f x m ≤ n) :
  m^n = 8 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_m_and_n_relationship_l694_69475


namespace NUMINAMATH_GPT_fred_dimes_l694_69428

theorem fred_dimes (initial_dimes borrowed_dimes : ℕ) (h1 : initial_dimes = 7) (h2 : borrowed_dimes = 3) :
  initial_dimes - borrowed_dimes = 4 :=
by
  sorry

end NUMINAMATH_GPT_fred_dimes_l694_69428


namespace NUMINAMATH_GPT_problem_statement_l694_69466

variable {α : Type*} [LinearOrderedCommRing α]

theorem problem_statement (a b c d e : α) (h : a * b^2 * c^3 * d^4 * e^5 < 0) : a * b^2 * c * d^4 * e < 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l694_69466


namespace NUMINAMATH_GPT_intersection_M_P_l694_69454

def is_natural (x : ℤ) : Prop := x ≥ 0

def M (x : ℤ) : Prop := (x - 1)^2 < 4 ∧ is_natural x

def P := ({-1, 0, 1, 2, 3} : Set ℤ)

theorem intersection_M_P :
  {x : ℤ | M x} ∩ P = {0, 1, 2} :=
  sorry

end NUMINAMATH_GPT_intersection_M_P_l694_69454


namespace NUMINAMATH_GPT_max_value_of_g_l694_69433

noncomputable def g (x : ℝ) : ℝ := 4 * x - x^4

theorem max_value_of_g : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → g x ≤ 3 ∧ (∃ x0, x0 = 1 ∧ g x0 = 3) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_g_l694_69433


namespace NUMINAMATH_GPT_correct_answer_is_B_l694_69480

-- Definitions for each set of line segments
def setA := (2, 2, 4)
def setB := (8, 6, 3)
def setC := (2, 6, 3)
def setD := (11, 4, 6)

-- Triangle inequality theorem checking function
def is_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statements to verify each set
lemma check_setA : ¬ is_triangle 2 2 4 := by sorry
lemma check_setB : is_triangle 8 6 3 := by sorry
lemma check_setC : ¬ is_triangle 2 6 3 := by sorry
lemma check_setD : ¬ is_triangle 11 4 6 := by sorry

-- Final theorem combining all checks to match the given problem
theorem correct_answer_is_B : 
  ¬ is_triangle 2 2 4 ∧ is_triangle 8 6 3 ∧ ¬ is_triangle 2 6 3 ∧ ¬ is_triangle 11 4 6 :=
by sorry

end NUMINAMATH_GPT_correct_answer_is_B_l694_69480


namespace NUMINAMATH_GPT_solve_for_x_l694_69499

theorem solve_for_x (x : ℝ) (h : (1 / 4) + (5 / x) = (12 / x) + (1 / 15)) : x = 420 / 11 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l694_69499


namespace NUMINAMATH_GPT_pump_without_leak_time_l694_69419

variables (P : ℝ) (effective_rate_with_leak : ℝ) (leak_rate : ℝ)
variable (pump_filling_time : ℝ)

-- Define the conditions
def conditions :=
  effective_rate_with_leak = 3/7 ∧
  leak_rate = 1/14 ∧
  pump_filling_time = P

-- Define the theorem
theorem pump_without_leak_time (h : conditions P effective_rate_with_leak leak_rate pump_filling_time) : 
  P = 2 :=
sorry

end NUMINAMATH_GPT_pump_without_leak_time_l694_69419


namespace NUMINAMATH_GPT_range_of_m_l694_69423

-- Define the conditions:

/-- Proposition p: the equation represents an ellipse with foci on y-axis -/
def proposition_p (m : ℝ) : Prop :=
  0 < m ∧ m < 9 ∧ 9 - m > 2 * m ∧ 2 * m > 0

/-- Proposition q: the eccentricity of the hyperbola is in the interval (\sqrt(3)/2, \sqrt(2)) -/
def proposition_q (m : ℝ) : Prop :=
  0 < m ∧ (5 / 2 < m ∧ m < 5)

def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def p_and_q (m : ℝ) : Prop := proposition_p m ∧ proposition_q m

-- Mathematically equivalent proof problem in Lean 4:

theorem range_of_m (m : ℝ) : (p_or_q m ∧ ¬p_and_q m) ↔ (m ∈ Set.Ioc 0 (5 / 2) ∪ Set.Icc 3 5) := sorry

end NUMINAMATH_GPT_range_of_m_l694_69423


namespace NUMINAMATH_GPT_correct_calculation_l694_69491

variable (a b : ℝ)

theorem correct_calculation : ((-a^2)^3 = -a^6) :=
by sorry

end NUMINAMATH_GPT_correct_calculation_l694_69491


namespace NUMINAMATH_GPT_min_value_of_fraction_sum_l694_69442

theorem min_value_of_fraction_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / (b + c) + b / (c + a) + c / (a + b)) ≥ 3 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_fraction_sum_l694_69442


namespace NUMINAMATH_GPT_minimum_students_lost_all_items_l694_69473

def smallest_number (N A B C : ℕ) (x : ℕ) : Prop :=
  N = 30 ∧ A = 26 ∧ B = 23 ∧ C = 21 → x ≥ 10

theorem minimum_students_lost_all_items (N A B C : ℕ) : 
  smallest_number N A B C 10 := 
by {
  sorry
}

end NUMINAMATH_GPT_minimum_students_lost_all_items_l694_69473


namespace NUMINAMATH_GPT_sin_alpha_second_quadrant_l694_69409

theorem sin_alpha_second_quadrant (α : ℝ) (h_α_quad_2 : π / 2 < α ∧ α < π) (h_cos_α : Real.cos α = -1 / 3) : Real.sin α = 2 * Real.sqrt 2 / 3 := 
sorry

end NUMINAMATH_GPT_sin_alpha_second_quadrant_l694_69409


namespace NUMINAMATH_GPT_find_n_l694_69411

theorem find_n (n : ℕ) : 2^(2 * n) + 2^(2 * n) + 2^(2 * n) + 2^(2 * n) = 4^22 → n = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l694_69411


namespace NUMINAMATH_GPT_tiffany_final_lives_l694_69417

def initial_lives : ℕ := 43
def lost_lives : ℕ := 14
def gained_lives : ℕ := 27

theorem tiffany_final_lives : (initial_lives - lost_lives + gained_lives) = 56 := by
    sorry

end NUMINAMATH_GPT_tiffany_final_lives_l694_69417


namespace NUMINAMATH_GPT_line_circle_intersections_l694_69460

-- Define the line equation as a predicate
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

-- Define the circle equation as a predicate
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 16

-- The goal is to prove the number of intersections of the line and the circle
theorem line_circle_intersections : (∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧ 
                                   (∃ x y : ℝ, line_eq x y ∧ circle_eq x y ∧ x ≠ y) :=
sorry

end NUMINAMATH_GPT_line_circle_intersections_l694_69460


namespace NUMINAMATH_GPT_total_spent_on_clothing_l694_69453

-- Define the individual costs
def shorts_cost : ℝ := 15
def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51

-- Define the proof problem to show the total cost
theorem total_spent_on_clothing : shorts_cost + jacket_cost + shirt_cost = 42.33 := by
  sorry

end NUMINAMATH_GPT_total_spent_on_clothing_l694_69453


namespace NUMINAMATH_GPT_symmetric_line_eq_l694_69445

theorem symmetric_line_eq (x y : ℝ) (c : ℝ) (P : ℝ × ℝ)
  (h₁ : 3 * x - y - 4 = 0)
  (h₂ : P = (2, -1))
  (h₃ : 3 * x - y + c = 0)
  (h : 3 * 2 - (-1) + c = 0) : 
  c = -7 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_eq_l694_69445


namespace NUMINAMATH_GPT_smallest_positive_integer_divisible_by_14_15_18_l694_69405

theorem smallest_positive_integer_divisible_by_14_15_18 : 
  ∃ n : ℕ, n > 0 ∧ (14 ∣ n) ∧ (15 ∣ n) ∧ (18 ∣ n) ∧ n = 630 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_divisible_by_14_15_18_l694_69405


namespace NUMINAMATH_GPT_polynomial_divisibility_n_l694_69413

theorem polynomial_divisibility_n :
  ∀ (n : ℤ), (∀ x, x = 2 → 3 * x^2 - 4 * x + n = 0) → n = -4 :=
by
  intros n h
  have h2 : 3 * 2^2 - 4 * 2 + n = 0 := h 2 rfl
  linarith

end NUMINAMATH_GPT_polynomial_divisibility_n_l694_69413


namespace NUMINAMATH_GPT_joy_can_choose_17_rods_for_quadrilateral_l694_69435

theorem joy_can_choose_17_rods_for_quadrilateral :
  ∃ (possible_rods : Finset ℕ), 
    possible_rods.card = 17 ∧
    ∀ rod ∈ possible_rods, 
      rod > 0 ∧ rod <= 30 ∧
      (rod ≠ 3 ∧ rod ≠ 7 ∧ rod ≠ 15) ∧
      (rod > 15 - (3 + 7)) ∧
      (rod < 3 + 7 + 15) :=
by
  sorry

end NUMINAMATH_GPT_joy_can_choose_17_rods_for_quadrilateral_l694_69435


namespace NUMINAMATH_GPT_steve_marbles_l694_69410

-- Define the initial condition variables
variables (S Steve_initial Sam_initial Sally_initial Sarah_initial Steve_now : ℕ)

-- Conditions
def cond1 : Sam_initial = 2 * Steve_initial := by sorry
def cond2 : Sally_initial = Sam_initial - 5 := by sorry
def cond3 : Sarah_initial = Steve_initial + 3 := by sorry
def cond4 : Steve_now = Steve_initial + 3 := by sorry
def cond5 : Sam_initial - (3 + 3 + 4) = 6 := by sorry

-- Goal
theorem steve_marbles : Steve_now = 11 := by sorry

end NUMINAMATH_GPT_steve_marbles_l694_69410


namespace NUMINAMATH_GPT_divisibility_by_91_l694_69412

theorem divisibility_by_91 (n : ℕ) : ∃ k : ℤ, 9^(n + 2) + 10^(2 * n + 1) = 91 * k := by
  sorry

end NUMINAMATH_GPT_divisibility_by_91_l694_69412


namespace NUMINAMATH_GPT_insphere_radius_l694_69484

theorem insphere_radius (V S : ℝ) (hV : V > 0) (hS : S > 0) : 
  ∃ r : ℝ, r = 3 * V / S := by
  sorry

end NUMINAMATH_GPT_insphere_radius_l694_69484


namespace NUMINAMATH_GPT_smallest_five_digit_palindrome_div_4_thm_l694_69401

def is_palindrome (n : ℕ) : Prop :=
  n = (n % 10) * 10000 + ((n / 10) % 10) * 1000 + ((n / 100) % 10) * 100 + ((n / 1000) % 10) * 10 + (n / 10000)

def smallest_five_digit_palindrome_div_4 : ℕ :=
  18881

theorem smallest_five_digit_palindrome_div_4_thm :
  is_palindrome smallest_five_digit_palindrome_div_4 ∧
  10000 ≤ smallest_five_digit_palindrome_div_4 ∧
  smallest_five_digit_palindrome_div_4 < 100000 ∧
  smallest_five_digit_palindrome_div_4 % 4 = 0 ∧
  ∀ n, is_palindrome n ∧ 10000 ≤ n ∧ n < 100000 ∧ n % 4 = 0 → n ≥ smallest_five_digit_palindrome_div_4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_palindrome_div_4_thm_l694_69401


namespace NUMINAMATH_GPT_cards_given_l694_69474

def initial_cards : ℕ := 304
def remaining_cards : ℕ := 276
def given_cards : ℕ := initial_cards - remaining_cards

theorem cards_given :
  given_cards = 28 :=
by
  unfold given_cards
  unfold initial_cards
  unfold remaining_cards
  sorry

end NUMINAMATH_GPT_cards_given_l694_69474


namespace NUMINAMATH_GPT_dissimilar_terms_expansion_count_l694_69455

noncomputable def num_dissimilar_terms_in_expansion (a b c d : ℝ) : ℕ :=
  let n := 8
  let k := 4
  Nat.choose (n + k - 1) (k - 1)

theorem dissimilar_terms_expansion_count : 
  num_dissimilar_terms_in_expansion a b c d = 165 := by
  sorry

end NUMINAMATH_GPT_dissimilar_terms_expansion_count_l694_69455


namespace NUMINAMATH_GPT_gnomes_and_ponies_l694_69407

theorem gnomes_and_ponies (g p : ℕ) (h1 : g + p = 15) (h2 : 2 * g + 4 * p = 36) : g = 12 ∧ p = 3 :=
by
  sorry

end NUMINAMATH_GPT_gnomes_and_ponies_l694_69407


namespace NUMINAMATH_GPT_inequality_2_pow_ge_n_sq_l694_69463

theorem inequality_2_pow_ge_n_sq (n : ℕ) (hn : n ≠ 3) : 2^n ≥ n^2 :=
sorry

end NUMINAMATH_GPT_inequality_2_pow_ge_n_sq_l694_69463


namespace NUMINAMATH_GPT_rectangle_sides_l694_69464

theorem rectangle_sides (a b : ℝ) (h1 : a < b) (h2 : a * b = 2 * a + 2 * b) : a < 4 ∧ b > 4 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_sides_l694_69464


namespace NUMINAMATH_GPT_circle_tangent_line_l694_69471

theorem circle_tangent_line 
    (center : ℝ × ℝ) (line_eq : ℝ → ℝ → ℝ) 
    (tangent_eq : ℝ) :
    center = (-1, 1) →
    line_eq 1 (-1)= 0 →
    tangent_eq = 2 :=
  let h := -1;
  let k := 1;
  let radius := Real.sqrt 2;
  sorry

end NUMINAMATH_GPT_circle_tangent_line_l694_69471


namespace NUMINAMATH_GPT_valueOf_seq_l694_69448

variable (a : ℕ → ℝ)
variable (h_arith_seq : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n)
variable (h_positive : ∀ n : ℕ, a n > 0)
variable (h_arith_subseq : 2 * a 5 = a 3 + a 6)

theorem valueOf_seq (a : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n)
  (h_positive : ∀ n : ℕ, a n > 0)
  (h_arith_subseq : 2 * a 5 = a 3 + a 6) :
  (∃ q : ℝ, q = 1 ∨ q = (1 + Real.sqrt 5) / 2 ∧ (a 3 + a 5) / (a 4 + a 6) = 1 / q) → 
  (∃ q : ℝ, (a 3 + a 5) / (a 4 + a 6) = 1 ∨ (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2) :=
by
  sorry

end NUMINAMATH_GPT_valueOf_seq_l694_69448


namespace NUMINAMATH_GPT_remaining_tickets_l694_69438

-- Define initial tickets and used tickets
def initial_tickets := 13
def used_tickets := 6

-- Declare the theorem we want to prove
theorem remaining_tickets (initial_tickets used_tickets : ℕ) (h1 : initial_tickets = 13) (h2 : used_tickets = 6) : initial_tickets - used_tickets = 7 :=
by
  sorry

end NUMINAMATH_GPT_remaining_tickets_l694_69438


namespace NUMINAMATH_GPT_exponentiation_division_l694_69488

variable (a b : ℝ)

theorem exponentiation_division (a b : ℝ) : ((2 * a) / b) ^ 4 = (16 * a ^ 4) / (b ^ 4) := by
  sorry

end NUMINAMATH_GPT_exponentiation_division_l694_69488


namespace NUMINAMATH_GPT_S_is_positive_rationals_l694_69476

variable {S : Set ℚ}

-- Defining the conditions as axioms
axiom cond1 (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : (a + b ∈ S) ∧ (a * b ∈ S)
axiom cond2 {r : ℚ} : (r ∈ S) ∨ (-r ∈ S) ∨ (r = 0)

-- The theorem to prove
theorem S_is_positive_rationals : S = { r : ℚ | r > 0 } := sorry

end NUMINAMATH_GPT_S_is_positive_rationals_l694_69476


namespace NUMINAMATH_GPT_xy_sum_is_one_l694_69468

theorem xy_sum_is_one (x y : ℤ) (h1 : 2021 * x + 2025 * y = 2029) (h2 : 2023 * x + 2027 * y = 2031) : 
  x + y = 1 :=
by sorry

end NUMINAMATH_GPT_xy_sum_is_one_l694_69468


namespace NUMINAMATH_GPT_previous_year_height_l694_69451

noncomputable def previous_height (H_current : ℝ) (g : ℝ) : ℝ :=
  H_current / (1 + g)

theorem previous_year_height :
  previous_height 147 0.05 = 140 :=
by
  unfold previous_height
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_previous_year_height_l694_69451


namespace NUMINAMATH_GPT_wage_difference_l694_69492

theorem wage_difference (P Q H: ℝ) (h1: P = 1.5 * Q) (h2: P * H = 300) (h3: Q * (H + 10) = 300) : P - Q = 5 :=
by
  sorry

end NUMINAMATH_GPT_wage_difference_l694_69492


namespace NUMINAMATH_GPT_no_preimage_iff_k_less_than_neg2_l694_69481

theorem no_preimage_iff_k_less_than_neg2 (k : ℝ) :
  ¬∃ x : ℝ, x^2 - 2 * x - 1 = k ↔ k < -2 :=
sorry

end NUMINAMATH_GPT_no_preimage_iff_k_less_than_neg2_l694_69481


namespace NUMINAMATH_GPT_car_cost_l694_69461

/--
A group of six friends planned to buy a car. They plan to share the cost equally. 
They had a car wash to help raise funds, which would be taken out of the total cost. 
The remaining cost would be split between the six friends. At the car wash, they earn $500. 
However, Brad decided not to join in the purchase of the car, and now each friend has to pay $40 more. 
What is the cost of the car?
-/
theorem car_cost 
  (C : ℝ) 
  (h1 : 6 * ((C - 500) / 5) = 5 * (C / 6 + 40)) : 
  C = 4200 := 
by 
  sorry

end NUMINAMATH_GPT_car_cost_l694_69461


namespace NUMINAMATH_GPT_minimum_weight_of_grass_seed_l694_69432

-- Definitions of cost and weights
def price_5_pound_bag : ℝ := 13.85
def price_10_pound_bag : ℝ := 20.43
def price_25_pound_bag : ℝ := 32.20
def max_weight : ℝ := 80
def min_cost : ℝ := 98.68

-- Lean proposition to prove the minimum weight given the conditions
theorem minimum_weight_of_grass_seed (w : ℝ) :
  w = 75 ↔ (w ≤ max_weight ∧
            ∃ (n5 n10 n25 : ℕ), 
              w = 5 * n5 + 10 * n10 + 25 * n25 ∧
              min_cost ≤ n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ∧
              n5 * price_5_pound_bag + n10 * price_10_pound_bag + n25 * price_25_pound_bag ≤ min_cost) := 
by
  sorry

end NUMINAMATH_GPT_minimum_weight_of_grass_seed_l694_69432


namespace NUMINAMATH_GPT_parabola_focus_hyperbola_equation_l694_69472

-- Problem 1
theorem parabola_focus (p : ℝ) (h₀ : p > 0) (h₁ : 2 * p - 0 - 4 = 0) : p = 2 :=
by
  sorry

-- Problem 2
theorem hyperbola_equation (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
  (h₂ : b / a = 3 / 4) (h₃ : a^2 / a = 16 / 5) (h₄ : a^2 + b^2 = 1) :
  (x^2 / 16) - (y^2 / 9) = 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_hyperbola_equation_l694_69472


namespace NUMINAMATH_GPT_calculate_expression_l694_69496

theorem calculate_expression : (5 + 7 + 3) / 3 - 2 / 3 = 13 / 3 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l694_69496


namespace NUMINAMATH_GPT_initial_price_of_phone_l694_69440

theorem initial_price_of_phone (P : ℝ) (h : 0.20 * P = 480) : P = 2400 :=
sorry

end NUMINAMATH_GPT_initial_price_of_phone_l694_69440


namespace NUMINAMATH_GPT_chris_leftover_money_l694_69467

def chris_will_have_leftover : Prop :=
  let video_game_cost := 60
  let candy_cost := 5
  let hourly_wage := 8
  let hours_worked := 9
  let total_earned := hourly_wage * hours_worked
  let total_cost := video_game_cost + candy_cost
  let leftover := total_earned - total_cost
  leftover = 7

theorem chris_leftover_money : chris_will_have_leftover := 
  by
    sorry

end NUMINAMATH_GPT_chris_leftover_money_l694_69467


namespace NUMINAMATH_GPT_gcd_102_238_eq_34_l694_69477

theorem gcd_102_238_eq_34 :
  Int.gcd 102 238 = 34 :=
sorry

end NUMINAMATH_GPT_gcd_102_238_eq_34_l694_69477


namespace NUMINAMATH_GPT_attendance_changes_l694_69426

theorem attendance_changes :
  let m := 25  -- Monday attendance
  let t := 31  -- Tuesday attendance
  let w := 20  -- initial Wednesday attendance
  let th := 28  -- Thursday attendance
  let f := 22  -- Friday attendance
  let sa := 26  -- Saturday attendance
  let w_new := 30  -- corrected Wednesday attendance
  let initial_total := m + t + w + th + f + sa
  let new_total := m + t + w_new + th + f + sa
  let initial_mean := initial_total / 6
  let new_mean := new_total / 6
  let mean_increase := new_mean - initial_mean
  let initial_median := (25 + 26) / 2  -- median of [20, 22, 25, 26, 28, 31]
  let new_median := (26 + 28) / 2  -- median of [22, 25, 26, 28, 30, 31]
  let median_increase := new_median - initial_median
  mean_increase = 1.667 ∧ median_increase = 1.5 := by
sorry

end NUMINAMATH_GPT_attendance_changes_l694_69426


namespace NUMINAMATH_GPT_inequality_l694_69418

-- Define the real variables p, q, r and the condition that their product is 1
variables {p q r : ℝ} (h : p * q * r = 1)

-- State the theorem
theorem inequality (h : p * q * r = 1) :
  (1 / (1 - p))^2 + (1 / (1 - q))^2 + (1 / (1 - r))^2 ≥ 1 := 
sorry

end NUMINAMATH_GPT_inequality_l694_69418


namespace NUMINAMATH_GPT_exists_isosceles_triangle_containing_l694_69404

variables {A B C X Y Z : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField C]

noncomputable def triangle (a b c : A) := a + b + c

def is_triangle (a b c : A) := a + b > c ∧ b + c > a ∧ c + a > b

def isosceles_triangle (a b c : A) := (a = b ∨ b = c ∨ c = a) ∧ a + b > c ∧ b + c > a ∧ c + a > b

theorem exists_isosceles_triangle_containing
  (a b c : A)
  (h1 : a < 1)
  (h2 : b < 1)
  (h3 : c < 1)
  (h_ABC : is_triangle a b c)
  : ∃ (x y z : A), isosceles_triangle x y z ∧ is_triangle x y z ∧ a < x ∧ b < y ∧ c < z ∧ x < 1 ∧ y < 1 ∧ z < 1 :=
sorry

end NUMINAMATH_GPT_exists_isosceles_triangle_containing_l694_69404


namespace NUMINAMATH_GPT_remainder_by_19_l694_69431

theorem remainder_by_19 (N : ℤ) (k : ℤ) (h : N = 779 * k + 47) : N % 19 = 9 :=
by sorry

end NUMINAMATH_GPT_remainder_by_19_l694_69431


namespace NUMINAMATH_GPT_sine_double_angle_inequality_l694_69437

theorem sine_double_angle_inequality {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 4) : 
  Real.sin (2 * α) < 2 * Real.sin α :=
by
  sorry

end NUMINAMATH_GPT_sine_double_angle_inequality_l694_69437


namespace NUMINAMATH_GPT_simplify_sqrt7_pow6_l694_69456

theorem simplify_sqrt7_pow6 : (Real.sqrt 7)^6 = (343 : Real) :=
by 
  -- we'll fill in the proof later
  sorry

end NUMINAMATH_GPT_simplify_sqrt7_pow6_l694_69456


namespace NUMINAMATH_GPT_ratio_of_expenditure_l694_69400

variable (A B AE BE : ℕ)

theorem ratio_of_expenditure (h1 : A = 2000) 
    (h2 : A / B = 5 / 4) 
    (h3 : A - AE = 800) 
    (h4: B - BE = 800) :
    AE / BE = 3 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_expenditure_l694_69400


namespace NUMINAMATH_GPT_find_width_of_chalkboard_l694_69465

variable (w : ℝ) (l : ℝ)

-- Given conditions
def length_eq_twice_width (w l : ℝ) : Prop := l = 2 * w
def area_eq_eighteen (w l : ℝ) : Prop := w * l = 18

-- Theorem statement
theorem find_width_of_chalkboard (h1 : length_eq_twice_width w l) (h2 : area_eq_eighteen w l) : w = 3 :=
by sorry

end NUMINAMATH_GPT_find_width_of_chalkboard_l694_69465


namespace NUMINAMATH_GPT_remainder_250_div_k_l694_69498

theorem remainder_250_div_k {k : ℕ} (h1 : 0 < k) (h2 : 180 % (k * k) = 12) : 250 % k = 10 := by
  sorry

end NUMINAMATH_GPT_remainder_250_div_k_l694_69498


namespace NUMINAMATH_GPT_oomyapeck_eyes_count_l694_69424

-- Define the various conditions
def number_of_people : ℕ := 3
def fish_per_person : ℕ := 4
def eyes_per_fish : ℕ := 2
def eyes_given_to_dog : ℕ := 2

-- Compute the total number of fish
def total_fish : ℕ := number_of_people * fish_per_person

-- Compute the total number of eyes from the total number of fish
def total_eyes : ℕ := total_fish * eyes_per_fish

-- Compute the number of eyes Oomyapeck eats
def eyes_eaten_by_oomyapeck : ℕ := total_eyes - eyes_given_to_dog

-- The proof statement
theorem oomyapeck_eyes_count : eyes_eaten_by_oomyapeck = 22 := by
  sorry

end NUMINAMATH_GPT_oomyapeck_eyes_count_l694_69424


namespace NUMINAMATH_GPT_problem_statement_l694_69490

open Complex

noncomputable def z : ℂ := ((1 - I)^2 + 3 * (1 + I)) / (2 - I)

theorem problem_statement :
  z = 1 + I ∧ (∀ (a b : ℝ), (z^2 + a * z + b = 1 - I) → (a = -3 ∧ b = 4)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l694_69490


namespace NUMINAMATH_GPT_max_value_ab_bc_cd_da_l694_69497

theorem max_value_ab_bc_cd_da (a b c d : ℝ) (a_nonneg : 0 ≤ a) (b_nonneg : 0 ≤ b) (c_nonneg : 0 ≤ c)
  (d_nonneg : 0 ≤ d) (sum_eq_200 : a + b + c + d = 200) : 
  ab + bc + cd + 0.5 * d * a ≤ 11250 := 
sorry


end NUMINAMATH_GPT_max_value_ab_bc_cd_da_l694_69497


namespace NUMINAMATH_GPT_solution_to_system_of_equations_l694_69421

theorem solution_to_system_of_equations :
  ∃ x y : ℤ, 4 * x - 3 * y = 11 ∧ 2 * x + y = 13 ∧ x = 5 ∧ y = 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_to_system_of_equations_l694_69421


namespace NUMINAMATH_GPT_solve_real_eq_l694_69447

theorem solve_real_eq (x : ℝ) :
  (8 * x ^ 2 + 150 * x + 3) / (3 * x + 56) = 4 * x + 2 ↔ x = -1.5 ∨ x = -18.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_real_eq_l694_69447


namespace NUMINAMATH_GPT_profit_per_meter_is_35_l694_69422

-- defining the conditions
def meters_sold : ℕ := 85
def selling_price : ℕ := 8925
def cost_price_per_meter : ℕ := 70
def total_cost_price := cost_price_per_meter * meters_sold
def total_selling_price := selling_price
def total_profit := total_selling_price - total_cost_price
def profit_per_meter := total_profit / meters_sold

-- Theorem stating the profit per meter of cloth
theorem profit_per_meter_is_35 : profit_per_meter = 35 := 
by
  sorry

end NUMINAMATH_GPT_profit_per_meter_is_35_l694_69422


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l694_69444

noncomputable def arithmetic_sequence (a3 a5_a7_sum : ℝ) : Prop :=
  ∃ (a d : ℝ), a + 2*d = a3 ∧ 2*a + 10*d = a5_a7_sum

noncomputable def sequence_a_n (a d n : ℝ) : ℝ := a + (n - 1)*d

noncomputable def sum_S_n (a d n : ℝ) : ℝ := n/2 * (2*a + (n-1)*d)

noncomputable def sequence_b_n (a d n : ℝ) : ℝ := 1 / (sequence_a_n a d n ^ 2 - 1)

noncomputable def sum_T_n (a d n : ℝ) : ℝ :=
  (1 / 4) * (1 - 1/(n+1))

theorem arithmetic_sequence_properties :
  (arithmetic_sequence 7 26) →
  (∀ n : ℕ+, sequence_a_n 3 2 n = 2 * n + 1) ∧
  (∀ n : ℕ+, sum_S_n 3 2 n = n^2 + 2 * n) ∧
  (∀ n : ℕ+, sum_T_n 3 2 n = n / (4 * (n + 1))) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l694_69444


namespace NUMINAMATH_GPT_number_of_dolls_combined_l694_69495

-- Defining the given conditions as variables
variables (aida sophie vera : ℕ)

-- Given conditions
def condition1 : Prop := aida = 2 * sophie
def condition2 : Prop := sophie = 2 * vera
def condition3 : Prop := vera = 20

-- The final proof statement we need to prove
theorem number_of_dolls_combined (h1 : condition1 aida sophie) (h2 : condition2 sophie vera) (h3 : condition3 vera) : 
  aida + sophie + vera = 140 :=
  by sorry

end NUMINAMATH_GPT_number_of_dolls_combined_l694_69495


namespace NUMINAMATH_GPT_classify_abc_l694_69459

theorem classify_abc (a b c : ℝ) 
  (h1 : (a > 0 ∨ a < 0 ∨ a = 0) ∧ (b > 0 ∨ b < 0 ∨ b = 0) ∧ (c > 0 ∨ c < 0 ∨ c = 0))
  (h2 : (a > 0 ∧ b < 0 ∧ c = 0) ∨ (a > 0 ∧ b = 0 ∧ c < 0) ∨ (a < 0 ∧ b > 0 ∧ c = 0) ∨
        (a < 0 ∧ b = 0 ∧ c > 0) ∨ (a = 0 ∧ b > 0 ∧ c < 0) ∨ (a = 0 ∧ b < 0 ∧ c > 0))
  (h3 : |a| = b^2 * (b - c)) : 
  a < 0 ∧ b > 0 ∧ c = 0 :=
by 
  sorry

end NUMINAMATH_GPT_classify_abc_l694_69459


namespace NUMINAMATH_GPT_count_valid_A_l694_69450

theorem count_valid_A : 
  ∃! (count : ℕ), count = 4 ∧ ∀ A : ℕ, (1 ≤ A ∧ A ≤ 9) → 
  (∃ x1 x2 : ℕ, x1 + x2 = 2 * A + 1 ∧ x1 * x2 = 2 * A ∧ x1 > 0 ∧ x2 > 0) → A = 1 ∨ A = 2 ∨ A = 3 ∨ A = 4 :=
sorry

end NUMINAMATH_GPT_count_valid_A_l694_69450


namespace NUMINAMATH_GPT_sandy_spent_on_repairs_l694_69436

theorem sandy_spent_on_repairs (initial_cost : ℝ) (selling_price : ℝ) (gain_percent : ℝ) (repair_cost : ℝ) :
  initial_cost = 800 → selling_price = 1400 → gain_percent = 40 → selling_price = 1.4 * (initial_cost + repair_cost) → repair_cost = 200 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sandy_spent_on_repairs_l694_69436


namespace NUMINAMATH_GPT_ninth_graders_science_only_l694_69414

theorem ninth_graders_science_only 
    (total_students : ℕ := 120)
    (science_students : ℕ := 80)
    (programming_students : ℕ := 75) 
    : (science_students - (science_students + programming_students - total_students)) = 45 :=
by
  sorry

end NUMINAMATH_GPT_ninth_graders_science_only_l694_69414


namespace NUMINAMATH_GPT_wins_per_girl_l694_69416

theorem wins_per_girl (a b c d : ℕ) (h1 : a + b = 8) (h2 : a + c = 10) (h3 : b + c = 12) (h4 : a + d = 12) (h5 : b + d = 14) (h6 : c + d = 16) : 
  a = 3 ∧ b = 5 ∧ c = 7 ∧ d = 9 :=
sorry

end NUMINAMATH_GPT_wins_per_girl_l694_69416


namespace NUMINAMATH_GPT_probability_of_shaded_triangle_l694_69485

def triangle (name: String) := name

def triangles := ["AEC", "AEB", "BED", "BEC", "BDC", "ABD"]
def shaded_triangles := ["BEC", "BDC", "ABD"]

theorem probability_of_shaded_triangle :
  (shaded_triangles.length : ℚ) / (triangles.length : ℚ) = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_shaded_triangle_l694_69485


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_2008_terms_l694_69458

theorem sum_of_arithmetic_sequence_2008_terms :
  let a := -1776
  let d := 11
  let n := 2008
  let l := a + (n - 1) * d
  let S := (n / 2) * (a + l)
  S = 18599100 := by
  sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_2008_terms_l694_69458


namespace NUMINAMATH_GPT_multiply_203_197_square_neg_699_l694_69420

theorem multiply_203_197 : 203 * 197 = 39991 := by
  sorry

theorem square_neg_699 : (-69.9)^2 = 4886.01 := by
  sorry

end NUMINAMATH_GPT_multiply_203_197_square_neg_699_l694_69420


namespace NUMINAMATH_GPT_remainder_of_sum_l694_69482

theorem remainder_of_sum (a b c : ℕ) (h1 : a % 15 = 8) (h2 : b % 15 = 12) (h3 : c % 15 = 13) : (a + b + c) % 15 = 3 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_l694_69482


namespace NUMINAMATH_GPT_determine_c_l694_69494

theorem determine_c (c y : ℝ) : (∀ y : ℝ, 3 * (3 + 2 * c * y) = 18 * y + 9) → c = 3 := by
  sorry

end NUMINAMATH_GPT_determine_c_l694_69494


namespace NUMINAMATH_GPT_side_length_of_square_l694_69462

theorem side_length_of_square (s : ℝ) (h : s^2 = 2 * (4 * s)) : s = 8 := 
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l694_69462


namespace NUMINAMATH_GPT_number_of_beakers_calculation_l694_69452

-- Conditions
def solution_per_test_tube : ℕ := 7
def number_of_test_tubes : ℕ := 6
def solution_per_beaker : ℕ := 14

-- Total amount of solution
def total_solution : ℕ := solution_per_test_tube * number_of_test_tubes

-- Number of beakers is the fraction of total solution and solution per beaker
def number_of_beakers : ℕ := total_solution / solution_per_beaker

-- Statement of the problem
theorem number_of_beakers_calculation : number_of_beakers = 3 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_beakers_calculation_l694_69452


namespace NUMINAMATH_GPT_P_desert_but_not_Coffee_is_0_15_l694_69406

-- Define the relevant probabilities as constants
def P_desert_and_coffee := 0.60
def P_not_desert := 0.2500000000000001
def P_desert := 1 - P_not_desert
def P_desert_but_not_coffee := P_desert - P_desert_and_coffee

-- The theorem to prove that the probability of ordering dessert but not coffee is 0.15
theorem P_desert_but_not_Coffee_is_0_15 :
  P_desert_but_not_coffee = 0.15 :=
by 
  -- calculation steps can be filled in here eventually
  sorry

end NUMINAMATH_GPT_P_desert_but_not_Coffee_is_0_15_l694_69406


namespace NUMINAMATH_GPT_product_of_two_numbers_l694_69408

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 400) : x * y = 88 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_l694_69408


namespace NUMINAMATH_GPT_evaluate_expression_l694_69449

theorem evaluate_expression :
  - (20 / 2 * (6^2 + 10) - 120 + 5 * 6) = -370 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l694_69449
