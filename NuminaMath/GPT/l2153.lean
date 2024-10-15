import Mathlib

namespace NUMINAMATH_GPT_find_a_l2153_215337

theorem find_a (a : ℝ) (h : (∃ x : ℝ, (a - 3) * x ^ |a - 2| + 4 = 0) ∧ |a-2| = 1) : a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l2153_215337


namespace NUMINAMATH_GPT_domain_of_function_l2153_215339

-- Definitions of the conditions
def condition1 (x : ℝ) : Prop := x - 5 ≠ 0
def condition2 (x : ℝ) : Prop := x - 2 > 0

-- The theorem stating the domain of the function
theorem domain_of_function (x : ℝ) : condition1 x ∧ condition2 x ↔ 2 < x ∧ x ≠ 5 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l2153_215339


namespace NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l2153_215310

theorem condition_necessary_but_not_sufficient (a : ℝ) :
  ((1 / a > 1) → (a < 1)) ∧ (∃ (a : ℝ), a < 1 ∧ 1 / a < 1) :=
by
  sorry

end NUMINAMATH_GPT_condition_necessary_but_not_sufficient_l2153_215310


namespace NUMINAMATH_GPT_addilynn_eggs_initial_l2153_215312

theorem addilynn_eggs_initial (E : ℕ) (H1 : ∃ (E : ℕ), (E / 2) - 15 = 21) : E = 72 :=
by
  sorry

end NUMINAMATH_GPT_addilynn_eggs_initial_l2153_215312


namespace NUMINAMATH_GPT_flowers_not_roses_percentage_l2153_215315

def percentage_non_roses (roses tulips daisies : Nat) : Nat :=
  let total := roses + tulips + daisies
  let non_roses := total - roses
  (non_roses * 100) / total

theorem flowers_not_roses_percentage :
  percentage_non_roses 25 40 35 = 75 :=
by
  sorry

end NUMINAMATH_GPT_flowers_not_roses_percentage_l2153_215315


namespace NUMINAMATH_GPT_solve_for_m_l2153_215304

theorem solve_for_m {m : ℝ} (h : ∀ x : ℝ, (m - 5) * x = 0) : m = 5 :=
sorry

end NUMINAMATH_GPT_solve_for_m_l2153_215304


namespace NUMINAMATH_GPT_equivalent_angle_l2153_215345

theorem equivalent_angle (theta : ℤ) (k : ℤ) : 
  (∃ k : ℤ, (-525 + k * 360 = 195)) :=
by
  sorry

end NUMINAMATH_GPT_equivalent_angle_l2153_215345


namespace NUMINAMATH_GPT_book_cost_l2153_215383

theorem book_cost (initial_money : ℕ) (remaining_money : ℕ) (num_books : ℕ) 
  (h1 : initial_money = 79) (h2 : remaining_money = 16) (h3 : num_books = 9) :
  (initial_money - remaining_money) / num_books = 7 :=
by
  sorry

end NUMINAMATH_GPT_book_cost_l2153_215383


namespace NUMINAMATH_GPT_prop1_prop2_prop3_prop4_final_l2153_215342

variables (a b c : ℝ) (h_a : a ≠ 0)

-- Proposition ①
theorem prop1 (h1 : a + b + c = 0) : b^2 - 4 * a * c ≥ 0 := 
sorry

-- Proposition ②
theorem prop2 (h2 : ∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) : 2 * a + c = 0 := 
sorry

-- Proposition ③
theorem prop3 (h3 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + c = 0 ∧ a * x2^2 + c = 0) : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 :=
sorry

-- Proposition ④
theorem prop4 (h4 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ ∃! x : ℝ, a * x^2 + b * x + c = 0) : ¬ (∃ x : ℝ, a * x^2 + b * x + c = 1 ∧ a * x^2 + b * x + 1 = 0) :=
sorry

-- Collectively checking that ①, ②, and ③ are true, and ④ is false
theorem final (h1 : a + b + c = 0)
              (h2 : ∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0)
              (h3 : ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + c = 0 ∧ a * x2^2 + c = 0)
              (h4 : ∃ x : ℝ, a * x^2 + b * x + c = 0 ∧ ∃! x : ℝ, a * x^2 + b * x + c = 0) : 
  (b^2 - 4 * a * c ≥ 0 ∧ 2 * a + c = 0 ∧ 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0) ∧ 
  ¬ (∃ x : ℝ, a * x^2 + b * x + c = 1 ∧ a * x^2 + b * x + 1 = 0)) :=
sorry

end NUMINAMATH_GPT_prop1_prop2_prop3_prop4_final_l2153_215342


namespace NUMINAMATH_GPT_basketball_game_points_half_l2153_215328

theorem basketball_game_points_half (a d b r : ℕ) (h_arith_seq : a + (a + d) + (a + 2 * d) + (a + 3 * d) ≤ 100)
    (h_geo_seq : b + b * r + b * r^2 + b * r^3 ≤ 100)
    (h_win_by_two : 4 * a + 6 * d = b * (1 + r + r^2 + r^3) + 2) :
    (a + (a + d)) + (b + b * r) = 14 :=
sorry

end NUMINAMATH_GPT_basketball_game_points_half_l2153_215328


namespace NUMINAMATH_GPT_simplify_abs_sum_l2153_215325

theorem simplify_abs_sum (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  |c - a - b| + |c + b - a| = 2 * b :=
sorry

end NUMINAMATH_GPT_simplify_abs_sum_l2153_215325


namespace NUMINAMATH_GPT_smallest_number_of_students_l2153_215318

theorem smallest_number_of_students 
    (ratio_9th_10th : Nat := 3 / 2)
    (ratio_9th_11th : Nat := 5 / 4)
    (ratio_9th_12th : Nat := 7 / 6) :
  ∃ N9 N10 N11 N12 : Nat, 
  N9 / N10 = 3 / 2 ∧ N9 / N11 = 5 / 4 ∧ N9 / N12 = 7 / 6 ∧ N9 + N10 + N11 + N12 = 349 :=
by {
  sorry
}

#print axioms smallest_number_of_students

end NUMINAMATH_GPT_smallest_number_of_students_l2153_215318


namespace NUMINAMATH_GPT_john_total_time_l2153_215390

noncomputable def total_time_spent : ℝ :=
  let landscape_pictures := 10
  let landscape_drawing_time := 2
  let landscape_coloring_time := landscape_drawing_time * 0.7
  let landscape_enhancing_time := 0.75
  let total_landscape_time := (landscape_drawing_time + landscape_coloring_time + landscape_enhancing_time) * landscape_pictures
  
  let portrait_pictures := 15
  let portrait_drawing_time := 3
  let portrait_coloring_time := portrait_drawing_time * 0.75
  let portrait_enhancing_time := 1.0
  let total_portrait_time := (portrait_drawing_time + portrait_coloring_time + portrait_enhancing_time) * portrait_pictures
  
  let abstract_pictures := 20
  let abstract_drawing_time := 1.5
  let abstract_coloring_time := abstract_drawing_time * 0.6
  let abstract_enhancing_time := 0.5
  let total_abstract_time := (abstract_drawing_time + abstract_coloring_time + abstract_enhancing_time) * abstract_pictures
  
  total_landscape_time + total_portrait_time + total_abstract_time

theorem john_total_time : total_time_spent = 193.25 :=
by sorry

end NUMINAMATH_GPT_john_total_time_l2153_215390


namespace NUMINAMATH_GPT_num_int_values_n_terminated_l2153_215356

theorem num_int_values_n_terminated (N : ℕ) (hN1 : 1 ≤ N) (hN2 : N ≤ 500) :
  ∃ n : ℕ, n = 10 ∧ ∀ k, 0 ≤ k → k < n → ∃ (m : ℕ), N = m * 49 :=
sorry

end NUMINAMATH_GPT_num_int_values_n_terminated_l2153_215356


namespace NUMINAMATH_GPT_find_y_l2153_215386

theorem find_y (x y : ℚ) (h1 : x = 151) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 342200) : 
  y = 342200 / 3354151 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2153_215386


namespace NUMINAMATH_GPT_june_initial_stickers_l2153_215313

theorem june_initial_stickers (J b g t : ℕ) (h_b : b = 63) (h_g : g = 25) (h_t : t = 189) : 
  (J + g) + (b + g) = t → J = 76 :=
by
  sorry

end NUMINAMATH_GPT_june_initial_stickers_l2153_215313


namespace NUMINAMATH_GPT_number_of_parallel_lines_l2153_215361

theorem number_of_parallel_lines (n : ℕ) (h : (n * (n - 1) / 2) * (8 * 7 / 2) = 784) : n = 8 :=
sorry

end NUMINAMATH_GPT_number_of_parallel_lines_l2153_215361


namespace NUMINAMATH_GPT_fencing_required_l2153_215359

theorem fencing_required (L W A F : ℝ) (hL : L = 20) (hA : A = 390) (hArea : A = L * W) (hF : F = 2 * W + L) : F = 59 :=
by
  sorry

end NUMINAMATH_GPT_fencing_required_l2153_215359


namespace NUMINAMATH_GPT_division_of_fraction_simplified_l2153_215391

theorem division_of_fraction_simplified :
  12 / (2 / (5 - 3)) = 12 := 
by
  sorry

end NUMINAMATH_GPT_division_of_fraction_simplified_l2153_215391


namespace NUMINAMATH_GPT_exponential_inequality_l2153_215311

-- Define the problem conditions and the proof goal
theorem exponential_inequality (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_eq : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := 
sorry

end NUMINAMATH_GPT_exponential_inequality_l2153_215311


namespace NUMINAMATH_GPT_perpendicular_lines_l2153_215302

theorem perpendicular_lines (m : ℝ) :
  (∃ k l : ℝ, k * m + (1 - m) * l = 3 ∧ (m - 1) * k + (2 * m + 3) * l = 2) → m = -3 ∨ m = 1 :=
by sorry

end NUMINAMATH_GPT_perpendicular_lines_l2153_215302


namespace NUMINAMATH_GPT_a_plus_d_eq_five_l2153_215329

theorem a_plus_d_eq_five (a b c d k : ℝ) (hk : 0 < k) 
  (h1 : a + b = 11) 
  (h2 : b^2 + c^2 = k) 
  (h3 : b + c = 9) 
  (h4 : c + d = 3) : 
  a + d = 5 :=
by
  sorry

end NUMINAMATH_GPT_a_plus_d_eq_five_l2153_215329


namespace NUMINAMATH_GPT_range_of_k_in_first_quadrant_l2153_215301

theorem range_of_k_in_first_quadrant (k : ℝ) (h₁ : k ≠ -1) :
  (∃ x y : ℝ, y = k * x - 1 ∧ x + y - 1 = 0 ∧ x > 0 ∧ y > 0) ↔ 1 < k := by sorry

end NUMINAMATH_GPT_range_of_k_in_first_quadrant_l2153_215301


namespace NUMINAMATH_GPT_find_xy_yz_xz_l2153_215334

-- Define the conditions given in the problem
variables (x y z : ℝ)
variable (hxyz_pos : x > 0 ∧ y > 0 ∧ z > 0)
variable (h1 : x^2 + x * y + y^2 = 12)
variable (h2 : y^2 + y * z + z^2 = 16)
variable (h3 : z^2 + z * x + x^2 = 28)

-- State the theorem to be proved
theorem find_xy_yz_xz : x * y + y * z + x * z = 16 :=
by {
    -- Proof will be done here
    sorry
}

end NUMINAMATH_GPT_find_xy_yz_xz_l2153_215334


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l2153_215369

noncomputable def f : ℝ → ℝ := sorry -- Define your function here satisfying the conditions

theorem problem1 (h1 : ∀ x : ℝ, f (2 - x) = f x)
                 (h2 : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x)) :
  f (-1) = 1 - Real.log 3 := sorry

theorem problem2 (h1 : ∀ x : ℝ, f (2 - x) = f x)
                 (h2 : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x)) :
  ∀ x : ℝ, f (2 - 2 * x) < f (x + 3) ↔ x ∈ Set.Ico (-1/3) 3 := sorry

theorem problem3 (h1 : ∀ x : ℝ, f (2 - x) = f x)
                 (h2 : ∀ x : ℝ, x ≥ 1 → f x = Real.log (x + 1/x))
                 (a : ℝ) :
  (∃ x : ℝ, 1 < x ∧ f x = Real.log (a / x + 2 * a)) ↔ a > 2/3 := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l2153_215369


namespace NUMINAMATH_GPT_infinite_geometric_series_sum_l2153_215324

theorem infinite_geometric_series_sum (a r S : ℚ) (ha : a = 1 / 4) (hr : r = 1 / 3) :
  (S = a / (1 - r)) → (S = 3 / 8) :=
by
  sorry

end NUMINAMATH_GPT_infinite_geometric_series_sum_l2153_215324


namespace NUMINAMATH_GPT_mr_johnson_fencing_l2153_215358

variable (Length Width : ℕ)

def perimeter_of_rectangle (Length Width : ℕ) : ℕ :=
  2 * (Length + Width)

theorem mr_johnson_fencing
  (hLength : Length = 25)
  (hWidth : Width = 15) :
  perimeter_of_rectangle Length Width = 80 := by
  sorry

end NUMINAMATH_GPT_mr_johnson_fencing_l2153_215358


namespace NUMINAMATH_GPT_Benny_spent_95_dollars_l2153_215382

theorem Benny_spent_95_dollars
    (amount_initial : ℕ)
    (amount_left : ℕ)
    (amount_spent : ℕ) :
    amount_initial = 120 →
    amount_left = 25 →
    amount_spent = amount_initial - amount_left →
    amount_spent = 95 :=
by
  intros h_initial h_left h_spent
  rw [h_initial, h_left] at h_spent
  exact h_spent

end NUMINAMATH_GPT_Benny_spent_95_dollars_l2153_215382


namespace NUMINAMATH_GPT_largest_divisor_of_composite_sum_and_square_l2153_215305

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

theorem largest_divisor_of_composite_sum_and_square (n : ℕ) (h : is_composite n) : ( ∃ (k : ℕ), ∀ n : ℕ, is_composite n → ∃ m : ℕ, n + n^2 = m * k) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_largest_divisor_of_composite_sum_and_square_l2153_215305


namespace NUMINAMATH_GPT_find_other_root_l2153_215360

theorem find_other_root (z : ℂ) (z_squared : z^2 = -91 + 104 * I) (root1 : z = 7 + 10 * I) : z = -7 - 10 * I :=
by
  sorry

end NUMINAMATH_GPT_find_other_root_l2153_215360


namespace NUMINAMATH_GPT_square_side_length_l2153_215323

variable (s d k : ℝ)

theorem square_side_length {s d k : ℝ} (h1 : s + d = k) (h2 : d = s * Real.sqrt 2) : 
  s = k / (1 + Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_square_side_length_l2153_215323


namespace NUMINAMATH_GPT_range_of_c_l2153_215344

noncomputable def is_monotonically_decreasing (c: ℝ) : Prop := ∀ x1 x2: ℝ, x1 < x2 → c^x2 ≤ c^x1

def inequality_holds (c: ℝ) : Prop := ∀ x: ℝ, x^2 + x + (1/2)*c > 0

theorem range_of_c (c: ℝ) (h1: c > 0) :
  ((is_monotonically_decreasing c ∨ inequality_holds c) ∧ ¬(is_monotonically_decreasing c ∧ inequality_holds c)) 
  → (0 < c ∧ c ≤ 1/2 ∨ c ≥ 1) := 
sorry

end NUMINAMATH_GPT_range_of_c_l2153_215344


namespace NUMINAMATH_GPT_smallest_m_for_integral_solutions_l2153_215374

theorem smallest_m_for_integral_solutions :
  ∃ m : ℕ, m > 0 ∧ (∃ p q : ℤ, 10 * p * q = 660 ∧ p + q = m/10) ∧ m = 170 :=
by
  sorry

end NUMINAMATH_GPT_smallest_m_for_integral_solutions_l2153_215374


namespace NUMINAMATH_GPT_mother_returns_to_freezer_l2153_215364

noncomputable def probability_return_to_freezer : ℝ :=
  1 - ((5 / 17) * (4 / 16) * (3 / 15) * (2 / 14) * (1 / 13))

theorem mother_returns_to_freezer :
  abs (probability_return_to_freezer - 0.99979) < 0.00001 :=
by
    sorry

end NUMINAMATH_GPT_mother_returns_to_freezer_l2153_215364


namespace NUMINAMATH_GPT_complex_quadrant_l2153_215314

-- Declare the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Declare the complex number z as per the condition
noncomputable def z : ℂ := (2 * i) / (i - 1)

-- State and prove that the complex number z lies in the fourth quadrant
theorem complex_quadrant : (z.re > 0) ∧ (z.im < 0) :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l2153_215314


namespace NUMINAMATH_GPT_mean_of_points_scored_l2153_215353

def mean (lst : List ℕ) : ℚ :=
  (lst.sum : ℚ) / lst.length

theorem mean_of_points_scored (lst : List ℕ)
  (h1 : lst = [81, 73, 83, 86, 73]) : 
  mean lst = 79.2 :=
by
  rw [h1, mean]
  sorry

end NUMINAMATH_GPT_mean_of_points_scored_l2153_215353


namespace NUMINAMATH_GPT_pat_stickers_at_end_of_week_l2153_215377

def initial_stickers : ℕ := 39
def monday_transaction : ℕ := 15
def tuesday_transaction : ℕ := 22
def wednesday_transaction : ℕ := 10
def thursday_trade_net_loss : ℕ := 4
def friday_find : ℕ := 5

def final_stickers (initial : ℕ) (mon : ℕ) (tue : ℕ) (wed : ℕ) (thu : ℕ) (fri : ℕ) : ℕ :=
  initial + mon - tue + wed - thu + fri

theorem pat_stickers_at_end_of_week :
  final_stickers initial_stickers 
                 monday_transaction 
                 tuesday_transaction 
                 wednesday_transaction 
                 thursday_trade_net_loss 
                 friday_find = 43 :=
by
  sorry

end NUMINAMATH_GPT_pat_stickers_at_end_of_week_l2153_215377


namespace NUMINAMATH_GPT_g_1000_is_1820_l2153_215399

-- Definitions and conditions from the problem
def g (n : ℕ) : ℕ := sorry -- exact definition is unknown, we will assume conditions

-- Conditions as given
axiom g_g (n : ℕ) : g (g n) = 3 * n
axiom g_3n_plus_1 (n : ℕ) : g (3 * n + 1) = 3 * n + 2

-- Statement to prove
theorem g_1000_is_1820 : g 1000 = 1820 :=
by
  sorry

end NUMINAMATH_GPT_g_1000_is_1820_l2153_215399


namespace NUMINAMATH_GPT_investment_rate_l2153_215380

theorem investment_rate (r : ℝ) (A : ℝ) (income_diff : ℝ) (total_invested : ℝ) (eight_percent_invested : ℝ) :
  total_invested = 2000 → 
  eight_percent_invested = 750 → 
  income_diff = 65 → 
  A = total_invested - eight_percent_invested → 
  (A * r) - (eight_percent_invested * 0.08) = income_diff → 
  r = 0.1 :=
by
  intros h_total h_eight h_income_diff h_A h_income_eq
  sorry

end NUMINAMATH_GPT_investment_rate_l2153_215380


namespace NUMINAMATH_GPT_new_acute_angle_l2153_215393

/- Definitions -/
def initial_angle_A (ACB : ℝ) (angle_CAB : ℝ) := angle_CAB = 40
def rotation_degrees (rotation : ℝ) := rotation = 480

/- Theorem Statement -/
theorem new_acute_angle (ACB : ℝ) (angle_CAB : ℝ) (rotation : ℝ) :
  initial_angle_A angle_CAB ACB ∧ rotation_degrees rotation → angle_CAB = 80 := 
by
  intros h
  -- This is where you'd provide the proof steps, but we use 'sorry' to indicate the proof is skipped.
  sorry

end NUMINAMATH_GPT_new_acute_angle_l2153_215393


namespace NUMINAMATH_GPT_problem_1_problem_2_l2153_215350

section proof_problem

variables (a b c d : ℤ)
variables (op : ℤ → ℤ → ℤ)
variables (add : ℤ → ℤ → ℤ)

-- Define the given conditions
axiom op_idem : ∀ (a : ℤ), op a a = a
axiom op_zero : ∀ (a : ℤ), op a 0 = 2 * a
axiom op_add : ∀ (a b c d : ℤ), add (op a b) (op c d) = op (a + c) (b + d)

-- Define the problems to prove
theorem problem_1 : add (op 2 3) (op 0 3) = -2 := sorry
theorem problem_2 : op 1024 48 = 2000 := sorry

end proof_problem

end NUMINAMATH_GPT_problem_1_problem_2_l2153_215350


namespace NUMINAMATH_GPT_evaluate_expression_l2153_215347

theorem evaluate_expression :
  ((-2: ℤ)^2) ^ (1 ^ (0 ^ 2)) + 3 ^ (0 ^(1 ^ 2)) = 5 :=
by
  -- sorry allows us to skip the proof
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2153_215347


namespace NUMINAMATH_GPT_find_a_l2153_215333

theorem find_a (x a : ℕ) (h : (x + 4) + 4 = (5 * x + a + 38) / 5) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l2153_215333


namespace NUMINAMATH_GPT_lottery_numbers_bound_l2153_215396

theorem lottery_numbers_bound (s : ℕ) (k : ℕ) (num_tickets : ℕ) (num_numbers : ℕ) (nums_per_ticket : ℕ)
  (h_tickets : num_tickets = 100) (h_numbers : num_numbers = 90) (h_nums_per_ticket : nums_per_ticket = 5)
  (h_s : s = num_tickets) (h_k : k = 49) :
  ∃ n : ℕ, n ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_lottery_numbers_bound_l2153_215396


namespace NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l2153_215379

def rotate (n : Nat) : Nat := 
  sorry -- Function definition for rotating the last digit to the start
def add_1001 (n : Nat) : Nat := 
  sorry -- Function definition for adding 1001
def subtract_1001 (n : Nat) : Nat := 
  sorry -- Function definition for subtracting 1001

theorem problem_a :
  ∃ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) ∧ (List.foldl (λacc step => step acc) 202122 steps = 313233) :=
sorry

theorem problem_b :
  ∃ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) ∧ (steps.length = 8) ∧ (List.foldl (λacc step => step acc) 999999 steps = 000000) :=
sorry

theorem problem_c (n : Nat) (hn : n % 11 = 0) : 
  ∀ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) → (List.foldl (λacc step => step acc) n steps) % 11 = 0 :=
sorry

theorem problem_d : 
  ∀ (steps : List (Nat → Nat)), 
    (∀ step ∈ steps, step = rotate ∨ step = add_1001 ∨ step = subtract_1001) → ¬(List.foldl (λacc step => step acc) 112233 steps = 000000) :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_problem_c_problem_d_l2153_215379


namespace NUMINAMATH_GPT_determine_real_numbers_l2153_215349

theorem determine_real_numbers (x y : ℝ) (h1 : x + y = 1) (h2 : x^3 + y^3 = 19) :
    (x = 3 ∧ y = -2) ∨ (x = -2 ∧ y = 3) :=
sorry

end NUMINAMATH_GPT_determine_real_numbers_l2153_215349


namespace NUMINAMATH_GPT_distance_between_trees_l2153_215387

theorem distance_between_trees (yard_length : ℕ) (num_trees : ℕ) (yard_length_eq : yard_length = 325) (num_trees_eq : num_trees = 26) :
  (yard_length / (num_trees - 1)) = 13 := by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l2153_215387


namespace NUMINAMATH_GPT_daily_earnings_c_l2153_215317

theorem daily_earnings_c (A B C : ℕ) (h1 : A + B + C = 600) (h2 : A + C = 400) (h3 : B + C = 300) : C = 100 :=
sorry

end NUMINAMATH_GPT_daily_earnings_c_l2153_215317


namespace NUMINAMATH_GPT_Jim_time_to_fill_pool_l2153_215375

-- Definitions for the work rates of Sue, Tony, and their combined work rate.
def Sue_work_rate : ℚ := 1 / 45
def Tony_work_rate : ℚ := 1 / 90
def Combined_work_rate : ℚ := 1 / 15

-- Proving the time it takes for Jim to fill the pool alone.
theorem Jim_time_to_fill_pool : ∃ J : ℚ, 1 / J + Sue_work_rate + Tony_work_rate = Combined_work_rate ∧ J = 30 :=
by {
  sorry
}

end NUMINAMATH_GPT_Jim_time_to_fill_pool_l2153_215375


namespace NUMINAMATH_GPT_minimize_abs_difference_and_product_l2153_215392

theorem minimize_abs_difference_and_product (x y : ℤ) (n : ℤ) 
(h1 : 20 * x + 19 * y = 2019)
(h2 : |x - y| = 18) 
: x * y = 2623 :=
sorry

end NUMINAMATH_GPT_minimize_abs_difference_and_product_l2153_215392


namespace NUMINAMATH_GPT_angle_supplement_complement_l2153_215309

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end NUMINAMATH_GPT_angle_supplement_complement_l2153_215309


namespace NUMINAMATH_GPT_no_positive_integer_n_eqn_l2153_215338

theorem no_positive_integer_n_eqn (n : ℕ) : (120^5 + 97^5 + 79^5 + 44^5 ≠ n^5) ∨ n = 144 :=
by
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_no_positive_integer_n_eqn_l2153_215338


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l2153_215341
-- Import the Mathlib library to use fractional arithmetic

-- Define the problem in Lean
theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 8
  let b := (5 : ℚ) / 9
  (a + b) / 2 = 67 / 144 :=
by
  let a : ℚ := 3 / 8
  let b : ℚ := 5 / 9
  have := (a + b) / 2 = 67 / 144
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l2153_215341


namespace NUMINAMATH_GPT_p_and_q_and_not_not_p_or_q_l2153_215351

theorem p_and_q_and_not_not_p_or_q (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by
  sorry

end NUMINAMATH_GPT_p_and_q_and_not_not_p_or_q_l2153_215351


namespace NUMINAMATH_GPT_part1_part2_l2153_215336

-- Conditions
def U := ℝ
def A : Set ℝ := {x | 0 < Real.log x / Real.log 2 ∧ Real.log x / Real.log 2 < 2}
def B (m : ℝ) : Set ℝ := {x | x ≤ 3 * m - 4 ∨ x ≥ 8 + m}
def complement_U (B : Set ℝ) : Set ℝ := {x | ¬(x ∈ B)}
def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

-- Assertions
theorem part1 (m : ℝ) (h1 : m = 2) : intersection A (complement_U (B m)) = {x | 2 < x ∧ x < 4} :=
  sorry

theorem part2 (h : intersection A (complement_U (B m)) = ∅) : -4 ≤ m ∧ m ≤ 5 / 3 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l2153_215336


namespace NUMINAMATH_GPT_polynomial_product_evaluation_l2153_215370

theorem polynomial_product_evaluation :
  let p1 := (2*x^3 - 3*x^2 + 5*x - 1)
  let p2 := (8 - 3*x)
  let product := p1 * p2
  let a := -6
  let b := 25
  let c := -39
  let d := 43
  let e := -8
  (16 * a + 8 * b + 4 * c + 2 * d + e) = 26 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_product_evaluation_l2153_215370


namespace NUMINAMATH_GPT_find_sum_l2153_215395

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem find_sum (h₁ : a * b = 2 * (a + b))
                (h₂ : b * c = 3 * (b + c))
                (h₃ : c * a = 4 * (a + c))
                (ha : a ≠ 0)
                (hb : b ≠ 0)
                (hc : c ≠ 0) 
                : a + b + c = 1128 / 35 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_l2153_215395


namespace NUMINAMATH_GPT_johns_quadratic_l2153_215381

theorem johns_quadratic (d e : ℤ) (h1 : d^2 = 16) (h2 : 2 * d * e = -40) : d * e = -20 :=
sorry

end NUMINAMATH_GPT_johns_quadratic_l2153_215381


namespace NUMINAMATH_GPT_circle_center_sum_l2153_215378

theorem circle_center_sum {x y : ℝ} (h : x^2 + y^2 - 10*x + 4*y + 15 = 0) :
  (x, y) = (5, -2) ∧ x + y = 3 :=
by
  sorry

end NUMINAMATH_GPT_circle_center_sum_l2153_215378


namespace NUMINAMATH_GPT_store_owner_uniforms_l2153_215373

theorem store_owner_uniforms (U E : ℕ) (h1 : U + 1 = 2 * E) (h2 : U % 2 = 1) : U = 3 := 
sorry

end NUMINAMATH_GPT_store_owner_uniforms_l2153_215373


namespace NUMINAMATH_GPT_incorrect_rational_number_statement_l2153_215321

theorem incorrect_rational_number_statement :
  ¬ (∀ x : ℚ, x > 0 ∨ x < 0) := by
sorry

end NUMINAMATH_GPT_incorrect_rational_number_statement_l2153_215321


namespace NUMINAMATH_GPT_square_side_length_l2153_215319

theorem square_side_length (s : ℝ) (h : 8 * s^2 = 3200) : s = 20 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_l2153_215319


namespace NUMINAMATH_GPT_constructible_triangle_and_area_bound_l2153_215303

noncomputable def triangle_inequality_sine (α β γ : ℝ) : Prop :=
  (Real.sin α + Real.sin β > Real.sin γ) ∧
  (Real.sin β + Real.sin γ > Real.sin α) ∧
  (Real.sin γ + Real.sin α > Real.sin β)

theorem constructible_triangle_and_area_bound 
  (α β γ : ℝ) (h_pos : 0 < α) (h_pos_β : 0 < β) (h_pos_γ : 0 < γ)
  (h_sum : α + β + γ < Real.pi)
  (h_ineq1 : α + β > γ)
  (h_ineq2 : β + γ > α)
  (h_ineq3 : γ + α > β) :
  triangle_inequality_sine α β γ ∧
  (Real.sin α * Real.sin β * Real.sin γ) / 4 ≤ (1 / 8) * (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) :=
sorry

end NUMINAMATH_GPT_constructible_triangle_and_area_bound_l2153_215303


namespace NUMINAMATH_GPT_chewbacca_gum_l2153_215376

variable {y : ℝ}

theorem chewbacca_gum (h1 : 25 - 2 * y ≠ 0) (h2 : 40 + 4 * y ≠ 0) :
    25 - 2 * y/40 = 25/(40 + 4 * y) → y = 2.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_chewbacca_gum_l2153_215376


namespace NUMINAMATH_GPT_grayson_unanswered_l2153_215398

noncomputable def unanswered_questions : ℕ :=
  let total_questions := 200
  let first_set_questions := 50
  let first_set_time := first_set_questions * 1 -- 1 minute per question
  let second_set_questions := 50
  let second_set_time := second_set_questions * (90 / 60) -- convert 90 seconds to minutes
  let third_set_questions := 25
  let third_set_time := third_set_questions * 2 -- 2 minutes per question
  let total_answered_time := first_set_time + second_set_time + third_set_time
  let total_time_available := 4 * 60 -- 4 hours in minutes 
  let unanswered := total_questions - (first_set_questions + second_set_questions + third_set_questions)
  unanswered

theorem grayson_unanswered : unanswered_questions = 75 := 
by 
  sorry

end NUMINAMATH_GPT_grayson_unanswered_l2153_215398


namespace NUMINAMATH_GPT_geometric_sequence_properties_l2153_215332

-- Given conditions as definitions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 * a 3 = a 4 ∧ a 3 = 8

-- Prove the common ratio and the sum of the first n terms
theorem geometric_sequence_properties (a : ℕ → ℝ)
  (h : seq a) :
  (∃ q, ∀ n, a n = a 1 * q ^ (n - 1) ∧ q = 2) ∧
  (∀ S_n, S_n = (1 - (2 : ℝ) ^ S_n) / (1 - 2) ∧ S_n = 2 ^ S_n - 1) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_properties_l2153_215332


namespace NUMINAMATH_GPT_kenny_total_liquid_l2153_215354

def total_liquid (oil_per_recipe water_per_recipe : ℚ) (times : ℕ) : ℚ :=
  (oil_per_recipe + water_per_recipe) * times

theorem kenny_total_liquid :
  total_liquid 0.17 1.17 12 = 16.08 := by
  sorry

end NUMINAMATH_GPT_kenny_total_liquid_l2153_215354


namespace NUMINAMATH_GPT_find_p_l2153_215368

def parabola_def (p : ℝ) : Prop := p > 0 ∧ ∀ (m : ℝ), (2 - (-p/2) = 4)

theorem find_p (p : ℝ) (m : ℝ) (h₁ : parabola_def p) (h₂ : (m ^ 2) = 2 * p * 2) 
(h₃ : (m ^ 2) = 2 * p * 2 → dist (2, m) (p / 2, 0) = 4) :
p = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l2153_215368


namespace NUMINAMATH_GPT_seedling_prices_l2153_215308

theorem seedling_prices (x y : ℝ) (a b : ℝ) 
  (h1 : 3 * x + 2 * y = 12)
  (h2 : x + 3 * y = 11) 
  (h3 : a + b = 200) 
  (h4 : 2 * 100 * a + 3 * 100 * b ≥ 50000) :
  x = 2 ∧ y = 3 ∧ b ≥ 100 := 
sorry

end NUMINAMATH_GPT_seedling_prices_l2153_215308


namespace NUMINAMATH_GPT_total_spent_at_music_store_l2153_215362

-- Defining the costs
def clarinet_cost : ℝ := 130.30
def song_book_cost : ℝ := 11.24

-- The main theorem to prove
theorem total_spent_at_music_store : clarinet_cost + song_book_cost = 141.54 :=
by
  sorry

end NUMINAMATH_GPT_total_spent_at_music_store_l2153_215362


namespace NUMINAMATH_GPT_tangent_line_to_circle_polar_l2153_215348

-- Definitions
def polar_circle_equation (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def point_polar_coordinates (ρ θ : ℝ) : Prop := ρ = 2 * Real.sqrt 2 ∧ θ = Real.pi / 4
def tangent_line_polar_equation (ρ θ : ℝ) : Prop := ρ * Real.cos θ = 2

-- Theorem Statement
theorem tangent_line_to_circle_polar {ρ θ : ℝ} :
  (∃ ρ θ, polar_circle_equation ρ θ) →
  (∃ ρ θ, point_polar_coordinates ρ θ) →
  tangent_line_polar_equation ρ θ :=
sorry

end NUMINAMATH_GPT_tangent_line_to_circle_polar_l2153_215348


namespace NUMINAMATH_GPT_greatest_three_digit_number_l2153_215343

theorem greatest_three_digit_number :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ N % 8 = 2 ∧ N % 7 = 4 ∧ N = 978 :=
by
  sorry

end NUMINAMATH_GPT_greatest_three_digit_number_l2153_215343


namespace NUMINAMATH_GPT_quadratic_function_min_value_at_1_l2153_215326

-- Define the quadratic function y = (x - 1)^2 - 3
def quadratic_function (x : ℝ) : ℝ :=
  (x - 1) ^ 2 - 3

-- The theorem to prove is that this quadratic function reaches its minimum value when x = 1.
theorem quadratic_function_min_value_at_1 : ∃ x : ℝ, quadratic_function x = quadratic_function 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_min_value_at_1_l2153_215326


namespace NUMINAMATH_GPT_probability_of_difference_three_l2153_215307

def is_valid_pair (a b : ℕ) : Prop :=
  (a = 3 ∧ b = 6) ∨ (a = 4 ∧ b = 1) ∨ (a = 5 ∧ b = 2) ∨ (a = 6 ∧ b = 3)

def number_of_successful_outcomes : ℕ := 4

def total_number_of_outcomes : ℕ := 36

def probability_of_valid_pairs : ℚ := number_of_successful_outcomes / total_number_of_outcomes

theorem probability_of_difference_three : probability_of_valid_pairs = 1 / 9 := by
  sorry

end NUMINAMATH_GPT_probability_of_difference_three_l2153_215307


namespace NUMINAMATH_GPT_friend_spent_more_than_you_l2153_215340

-- Define the total amount spent by both
def total_spent : ℤ := 19

-- Define the amount spent by your friend
def friend_spent : ℤ := 11

-- Define the amount spent by you
def you_spent : ℤ := total_spent - friend_spent

-- Define the difference in spending
def difference_in_spending : ℤ := friend_spent - you_spent

-- Prove that the difference in spending is $3
theorem friend_spent_more_than_you : difference_in_spending = 3 :=
by
  sorry

end NUMINAMATH_GPT_friend_spent_more_than_you_l2153_215340


namespace NUMINAMATH_GPT_sequence_general_term_l2153_215357

-- Define the sequence based on the given conditions
def seq (n : ℕ) : ℚ := if n = 0 then 1 else (n : ℚ) / (2 * n - 1)

theorem sequence_general_term (n : ℕ) :
  seq (n + 1) = (n + 1) / (2 * (n + 1) - 1) :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2153_215357


namespace NUMINAMATH_GPT_correct_ranking_l2153_215365

-- Definitions for the colleagues
structure Colleague :=
  (name : String)
  (seniority : ℕ)

-- Colleagues: Julia, Kevin, Lana
def Julia := Colleague.mk "Julia" 1
def Kevin := Colleague.mk "Kevin" 0
def Lana := Colleague.mk "Lana" 2

-- Statements definitions
def Statement_I (c1 c2 c3 : Colleague) := c2.seniority < c1.seniority ∧ c1.seniority < c3.seniority 
def Statement_II (c1 c2 c3 : Colleague) := c1.seniority > c3.seniority
def Statement_III (c1 c2 c3 : Colleague) := c1.seniority ≠ c1.seniority

-- Exactly one of the statements is true
def Exactly_One_True (s1 s2 s3 : Prop) := (s1 ∨ s2 ∨ s3) ∧ ¬(s1 ∧ s2 ∨ s1 ∧ s3 ∨ s2 ∧ s3) ∧ ¬(s1 ∧ s2 ∧ s3)

-- The theorem to be proved
theorem correct_ranking :
  Exactly_One_True (Statement_I Kevin Lana Julia) (Statement_II Kevin Lana Julia) (Statement_III Kevin Lana Julia) →
  (Kevin.seniority < Lana.seniority ∧ Lana.seniority < Julia.seniority) := 
  by  sorry

end NUMINAMATH_GPT_correct_ranking_l2153_215365


namespace NUMINAMATH_GPT_total_daisies_l2153_215306

-- Define the conditions
def white_daisies : ℕ := 6
def pink_daisies : ℕ := 9 * white_daisies
def red_daisies : ℕ := 4 * pink_daisies - 3

-- Main statement to be proved
theorem total_daisies : white_daisies + pink_daisies + red_daisies = 273 := by
  sorry

end NUMINAMATH_GPT_total_daisies_l2153_215306


namespace NUMINAMATH_GPT_odd_square_not_sum_of_five_odd_squares_l2153_215320

theorem odd_square_not_sum_of_five_odd_squares :
  ∀ (n : ℤ), (∃ k : ℤ, k^2 % 8 = n % 8 ∧ n % 8 = 1) →
             ¬(∃ a b c d e : ℤ, (a^2 % 8 = 1) ∧ (b^2 % 8 = 1) ∧ (c^2 % 8 = 1) ∧ (d^2 % 8 = 1) ∧ 
               (e^2 % 8 = 1) ∧ (n % 8 = (a^2 + b^2 + c^2 + d^2 + e^2) % 8)) :=
by
  sorry

end NUMINAMATH_GPT_odd_square_not_sum_of_five_odd_squares_l2153_215320


namespace NUMINAMATH_GPT_jovana_shells_l2153_215389

variable (initial_shells : Nat) (additional_shells : Nat)

theorem jovana_shells (h1 : initial_shells = 5) (h2 : additional_shells = 12) : initial_shells + additional_shells = 17 := 
by 
  sorry

end NUMINAMATH_GPT_jovana_shells_l2153_215389


namespace NUMINAMATH_GPT_james_total_earnings_l2153_215352

-- Assume the necessary info for January, February, and March earnings
-- Definitions given as conditions in a)
def January_earnings : ℝ := 4000

def February_earnings : ℝ := January_earnings * 1.5 * 1.2

def March_earnings : ℝ := February_earnings * 0.8

-- The total earnings to be calculated
def Total_earnings : ℝ := January_earnings + February_earnings + March_earnings

-- Prove the total earnings is $16960
theorem james_total_earnings : Total_earnings = 16960 := by
  sorry

end NUMINAMATH_GPT_james_total_earnings_l2153_215352


namespace NUMINAMATH_GPT_find_angle_A_l2153_215355

-- Conditions
def is_triangle (A B C : ℝ) : Prop := A + B + C = 180
def B_is_two_C (B C : ℝ) : Prop := B = 2 * C
def B_is_80 (B : ℝ) : Prop := B = 80

-- Theorem statement
theorem find_angle_A (A B C : ℝ) (h₁ : is_triangle A B C) (h₂ : B_is_two_C B C) (h₃ : B_is_80 B) : A = 60 := by
  sorry

end NUMINAMATH_GPT_find_angle_A_l2153_215355


namespace NUMINAMATH_GPT_stamps_in_last_page_l2153_215331

-- Define the total number of books, pages per book, and stamps per original page.
def total_books : ℕ := 6
def pages_per_book : ℕ := 30
def original_stamps_per_page : ℕ := 7

-- Define the new stamps per page after reorganization.
def new_stamps_per_page : ℕ := 9

-- Define the number of fully filled books and pages in the fourth book.
def filled_books : ℕ := 3
def pages_in_fourth_book : ℕ := 26

-- Define the total number of stamps originally.
def total_original_stamps : ℕ := total_books * pages_per_book * original_stamps_per_page

-- Prove that the last page in the fourth book contains 9 stamps under the given conditions.
theorem stamps_in_last_page : 
  total_original_stamps / new_stamps_per_page - (filled_books * pages_per_book + pages_in_fourth_book) * new_stamps_per_page = 9 :=
by
  sorry

end NUMINAMATH_GPT_stamps_in_last_page_l2153_215331


namespace NUMINAMATH_GPT_sin_double_angle_neg_l2153_215385

variable (α : Real)
variable (h1 : Real.tan α < 0)
variable (h2 : Real.sin α = -Real.sqrt 3 / 3)

theorem sin_double_angle_neg (h1 : Real.tan α < 0) (h2 : Real.sin α = -Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2 * Real.sqrt 2 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_neg_l2153_215385


namespace NUMINAMATH_GPT_circles_are_intersecting_l2153_215394

-- Define the circles and the distances given
def radius_O1 : ℝ := 3
def radius_O2 : ℝ := 5
def distance_O1O2 : ℝ := 2

-- Define the positional relationships
inductive PositionalRelationship
| externally_tangent
| intersecting
| internally_tangent
| contained_within_each_other

open PositionalRelationship

-- State the theorem to be proved
theorem circles_are_intersecting :
  distance_O1O2 > 0 ∧ distance_O1O2 < (radius_O1 + radius_O2) ∧ distance_O1O2 > abs (radius_O1 - radius_O2) →
  PositionalRelationship := 
by
  intro h
  exact PositionalRelationship.intersecting

end NUMINAMATH_GPT_circles_are_intersecting_l2153_215394


namespace NUMINAMATH_GPT_cookies_sold_by_Lucy_l2153_215330

theorem cookies_sold_by_Lucy :
  let cookies_first_round := 34
  let cookies_second_round := 27
  cookies_first_round + cookies_second_round = 61 := by
  sorry

end NUMINAMATH_GPT_cookies_sold_by_Lucy_l2153_215330


namespace NUMINAMATH_GPT_inequality_A_if_ab_pos_inequality_D_if_ab_pos_l2153_215322

variable (a b : ℝ)

theorem inequality_A_if_ab_pos (h : a * b > 0) : a^2 + b^2 ≥ 2 * a * b := 
sorry

theorem inequality_D_if_ab_pos (h : a * b > 0) : (b / a) + (a / b) ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_A_if_ab_pos_inequality_D_if_ab_pos_l2153_215322


namespace NUMINAMATH_GPT_probability_same_color_is_correct_l2153_215372

/- Given that there are 5 balls in total, where 3 are white and 2 are black, and two balls are drawn randomly from the bag, we need to prove that the probability of drawing two balls of the same color is 2/5. -/

def total_balls : ℕ := 5
def white_balls : ℕ := 3
def black_balls : ℕ := 2

def total_ways (n r : ℕ) : ℕ := n.choose r
def white_ways : ℕ := total_ways white_balls 2
def black_ways : ℕ := total_ways black_balls 2
def same_color_ways : ℕ := white_ways + black_ways
def total_draws : ℕ := total_ways total_balls 2

def probability_same_color := ((same_color_ways : ℚ) / total_draws)
def expected_probability := (2 : ℚ) / 5

theorem probability_same_color_is_correct :
  probability_same_color = expected_probability :=
by
  sorry

end NUMINAMATH_GPT_probability_same_color_is_correct_l2153_215372


namespace NUMINAMATH_GPT_marks_difference_l2153_215327

variable (P C M : ℕ)

-- Conditions
def total_marks_more_than_physics := P + C + M > P
def average_chemistry_mathematics := (C + M) / 2 = 65

-- Proof Statement
theorem marks_difference (h1 : total_marks_more_than_physics P C M) (h2 : average_chemistry_mathematics C M) : 
  P + C + M = P + 130 := by
  sorry

end NUMINAMATH_GPT_marks_difference_l2153_215327


namespace NUMINAMATH_GPT_proof_problem_l2153_215371

variables {R : Type*} [Field R] (p q r u v w : R)

theorem proof_problem (h₁ : 15*u + q*v + r*w = 0)
                      (h₂ : p*u + 25*v + r*w = 0)
                      (h₃ : p*u + q*v + 50*w = 0)
                      (hp : p ≠ 15)
                      (hu : u ≠ 0) : 
                      (p / (p - 15) + q / (q - 25) + r / (r - 50)) = 1 := 
by sorry

end NUMINAMATH_GPT_proof_problem_l2153_215371


namespace NUMINAMATH_GPT_find_segment_XY_length_l2153_215397

theorem find_segment_XY_length (A B C D X Y : Type) 
  [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] [DecidableEq X] [DecidableEq Y]
  (line_l : Type) (BX : ℝ) (DY : ℝ) (AB : ℝ) (BC : ℝ) (l : line_l)
  (hBX : BX = 4) (hDY : DY = 10) (hBC : BC = 2 * AB) :
  XY = 13 :=
  sorry

end NUMINAMATH_GPT_find_segment_XY_length_l2153_215397


namespace NUMINAMATH_GPT_tan_subtraction_l2153_215366

theorem tan_subtraction (α β : ℝ) (h₁ : Real.tan α = 9) (h₂ : Real.tan β = 6) :
  Real.tan (α - β) = 3 / 55 :=
by
  sorry

end NUMINAMATH_GPT_tan_subtraction_l2153_215366


namespace NUMINAMATH_GPT_compute_abs_ab_eq_2_sqrt_111_l2153_215335

theorem compute_abs_ab_eq_2_sqrt_111 (a b : ℝ) 
  (h1 : b^2 - a^2 = 25)
  (h2 : a^2 + b^2 = 49) : 
  |a * b| = 2 * Real.sqrt 111 := 
sorry

end NUMINAMATH_GPT_compute_abs_ab_eq_2_sqrt_111_l2153_215335


namespace NUMINAMATH_GPT_calc_triple_hash_30_l2153_215384

def hash_fn (N : ℝ) : ℝ := 0.6 * N + 2

theorem calc_triple_hash_30 :
  hash_fn (hash_fn (hash_fn 30)) = 10.4 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_calc_triple_hash_30_l2153_215384


namespace NUMINAMATH_GPT_line_equation_l2153_215346

-- Given conditions
def param_x (t : ℝ) : ℝ := 3 * t + 6
def param_y (t : ℝ) : ℝ := 5 * t - 7

-- Proof problem: for any real t, the parameterized line can be described by the equation y = 5x/3 - 17.
theorem line_equation (t : ℝ) : ∃ (m b : ℝ), (∃ t : ℝ, param_y t = m * (param_x t) + b) ∧ m = 5 / 3 ∧ b = -17 :=
by
  exists 5 / 3
  exists -17
  sorry

end NUMINAMATH_GPT_line_equation_l2153_215346


namespace NUMINAMATH_GPT_second_intersection_of_parabola_l2153_215367

theorem second_intersection_of_parabola (x_vertex_Pi1 x_vertex_Pi2 : ℝ) : 
  (∀ x : ℝ, x = (10 + 13) / 2 → x_vertex_Pi1 = x) →
  (∀ y : ℝ, y = (x_vertex_Pi2 / 2) → x_vertex_Pi1 = y) →
  (x_vertex_Pi2 = 2 * x_vertex_Pi1) →
  (13 + 33) / 2 = x_vertex_Pi2 :=
by
  sorry

end NUMINAMATH_GPT_second_intersection_of_parabola_l2153_215367


namespace NUMINAMATH_GPT_arcsin_sqrt2_div2_l2153_215363

theorem arcsin_sqrt2_div2 :
  Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_arcsin_sqrt2_div2_l2153_215363


namespace NUMINAMATH_GPT_system_of_equations_solution_l2153_215388

theorem system_of_equations_solution (x y : ℝ) (h1 : 2 * x ^ 2 - 5 * x + 3 = 0) (h2 : y = 3 * x + 1) : 
  (x = 1.5 ∧ y = 5.5) ∨ (x = 1 ∧ y = 4) :=
sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2153_215388


namespace NUMINAMATH_GPT_eq_cont_fracs_l2153_215316

noncomputable def cont_frac : Nat -> Rat
| 0       => 0
| (n + 1) => (n : Rat) + 1 / (cont_frac n)

theorem eq_cont_fracs (n : Nat) : 
  1 - cont_frac n = cont_frac n - 1 :=
sorry

end NUMINAMATH_GPT_eq_cont_fracs_l2153_215316


namespace NUMINAMATH_GPT_output_sequence_value_l2153_215300

theorem output_sequence_value (x y : Int) (seq : List (Int × Int))
  (h : (x, y) ∈ seq) (h_y : y = -10) : x = 32 :=
by
  sorry

end NUMINAMATH_GPT_output_sequence_value_l2153_215300
