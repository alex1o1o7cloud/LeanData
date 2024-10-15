import Mathlib

namespace NUMINAMATH_GPT_balloon_permutations_l1453_145393

theorem balloon_permutations : 
  (Nat.factorial 7) / ((Nat.factorial 2) * (Nat.factorial 3)) = 420 :=
by
  sorry

end NUMINAMATH_GPT_balloon_permutations_l1453_145393


namespace NUMINAMATH_GPT_eq_of_line_through_points_l1453_145374

noncomputable def line_eqn (x y : ℝ) : Prop :=
  x - y + 3 = 0

theorem eq_of_line_through_points :
  ∀ (x1 y1 x2 y2 : ℝ), 
    x1 = -1 → y1 = 2 → x2 = 2 → y2 = 5 → 
    line_eqn (x1 + y1 - x2) (y2 - y1) :=
by
  intros x1 y1 x2 y2 hx1 hy1 hx2 hy2
  rw [hx1, hy1, hx2, hy2]
  sorry -- Proof steps would go here.

end NUMINAMATH_GPT_eq_of_line_through_points_l1453_145374


namespace NUMINAMATH_GPT_original_number_l1453_145315

theorem original_number (x : ℕ) (h : x / 3 = 42) : x = 126 :=
sorry

end NUMINAMATH_GPT_original_number_l1453_145315


namespace NUMINAMATH_GPT_find_rate_l1453_145341

def plan1_cost (minutes : ℕ) : ℝ :=
  if minutes <= 500 then 50 else 50 + (minutes - 500) * 0.35

def plan2_cost (minutes : ℕ) (x : ℝ) : ℝ :=
  if minutes <= 1000 then 75 else 75 + (minutes - 1000) * x

theorem find_rate (x : ℝ) :
  plan1_cost 2500 = plan2_cost 2500 x → x = 0.45 := by
  sorry

end NUMINAMATH_GPT_find_rate_l1453_145341


namespace NUMINAMATH_GPT_algebraic_expr_pos_int_vals_l1453_145346

noncomputable def algebraic_expr_ineq (x : ℕ) : Prop :=
  x > 0 ∧ ((x + 1)/3 - (2*x - 1)/4 ≥ (x - 3)/6)

theorem algebraic_expr_pos_int_vals : {x : ℕ | algebraic_expr_ineq x} = {1, 2, 3} :=
sorry

end NUMINAMATH_GPT_algebraic_expr_pos_int_vals_l1453_145346


namespace NUMINAMATH_GPT_integer_roots_l1453_145349

noncomputable def is_quadratic_root (p q x : ℝ) : Prop :=
  x^2 + p * x + q = 0

theorem integer_roots (p q x1 x2 : ℝ)
  (hq1 : is_quadratic_root p q x1)
  (hq2 : is_quadratic_root p q x2)
  (hd : x1 ≠ x2)
  (hx : |x1 - x2| = 1)
  (hpq : |p - q| = 1) :
  (∃ (p_int q_int x1_int x2_int : ℤ), 
      p = p_int ∧ q = q_int ∧ x1 = x1_int ∧ x2 = x2_int) :=
sorry

end NUMINAMATH_GPT_integer_roots_l1453_145349


namespace NUMINAMATH_GPT_sqrt8_sub_sqrt2_eq_sqrt2_l1453_145342

theorem sqrt8_sub_sqrt2_eq_sqrt2 : Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_sqrt8_sub_sqrt2_eq_sqrt2_l1453_145342


namespace NUMINAMATH_GPT_max_divisions_circle_and_lines_l1453_145344

theorem max_divisions_circle_and_lines (n : ℕ) (h₁ : n = 5) : 
  let R_lines := n * (n + 1) / 2 + 1 -- Maximum regions formed by n lines
  let R_circle_lines := 2 * n       -- Additional regions formed by a circle intersecting n lines
  R_lines + R_circle_lines = 26 := by
  sorry

end NUMINAMATH_GPT_max_divisions_circle_and_lines_l1453_145344


namespace NUMINAMATH_GPT_added_water_is_18_l1453_145300

def capacity : ℕ := 40

def initial_full_percent : ℚ := 0.30

def final_full_fraction : ℚ := 3/4

def initial_water (capacity : ℕ) (initial_full_percent : ℚ) : ℚ :=
  initial_full_percent * capacity

def final_water (capacity : ℕ) (final_full_fraction : ℚ) : ℚ :=
  final_full_fraction * capacity

def water_added (initial_water : ℚ) (final_water : ℚ) : ℚ :=
  final_water - initial_water

theorem added_water_is_18 :
  water_added (initial_water capacity initial_full_percent) (final_water capacity final_full_fraction) = 18 := by
  sorry

end NUMINAMATH_GPT_added_water_is_18_l1453_145300


namespace NUMINAMATH_GPT_bucket_problem_l1453_145339

variable (A B C : ℝ)

theorem bucket_problem :
  (A - 6 = (1 / 3) * (B + 6)) →
  (B - 6 = (1 / 2) * (A + 6)) →
  (C - 8 = (1 / 2) * (A + 8)) →
  A = 13.2 :=
by
  sorry

end NUMINAMATH_GPT_bucket_problem_l1453_145339


namespace NUMINAMATH_GPT_order_of_variables_l1453_145383

variable (a b c d : ℝ)

theorem order_of_variables (h : a - 1 = b + 2 ∧ b + 2 = c - 3 ∧ c - 3 = d + 4) : c > a ∧ a > b ∧ b > d :=
by
  sorry

end NUMINAMATH_GPT_order_of_variables_l1453_145383


namespace NUMINAMATH_GPT_range_of_expression_l1453_145311

theorem range_of_expression (a : ℝ) : (∃ a : ℝ, a + 1 ≥ 0 ∧ a - 2 ≠ 0) → (a ≥ -1 ∧ a ≠ 2) := 
by sorry

end NUMINAMATH_GPT_range_of_expression_l1453_145311


namespace NUMINAMATH_GPT_dot_product_expression_max_value_of_dot_product_l1453_145367

variable (x : ℝ)
variable (k : ℤ)
variable (a : ℝ × ℝ := (Real.cos x, -1 + Real.sin x))
variable (b : ℝ × ℝ := (2 * Real.cos x, Real.sin x))

theorem dot_product_expression :
  (a.1 * b.1 + a.2 * b.2) = 2 - 3 * (Real.sin x)^2 - (Real.sin x) := 
sorry

theorem max_value_of_dot_product :
  ∃ (x : ℝ), 2 - 3 * (Real.sin x)^2 - (Real.sin x) = 9 / 4 ∧ 
  (Real.sin x = -1/2 ∧ 
  (x = 7 * Real.pi / 6 + 2 * k * Real.pi ∨ x = 11 * Real.pi / 6 + 2 * k * Real.pi)) := 
sorry

end NUMINAMATH_GPT_dot_product_expression_max_value_of_dot_product_l1453_145367


namespace NUMINAMATH_GPT_find_k_l1453_145366

-- Define point type and distances
structure Point :=
(x : ℝ)
(y : ℝ)

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition: H is the orthocenter of triangle ABC
variable (A B C H Q : Point)
variable (H_is_orthocenter : ∀ P : Point, dist P H = dist P A + dist P B - dist A B)

-- Prove the given equation
theorem find_k :
  dist Q A + dist Q B + dist Q C = 3 * dist Q H + dist H A + dist H B + dist H C :=
sorry

end NUMINAMATH_GPT_find_k_l1453_145366


namespace NUMINAMATH_GPT_problem1_problem2_l1453_145318

-- Definition for the first problem
def isPerfectSquare (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- First Lean 4 statement for 2^n + 3 = x^2
theorem problem1 (n : ℕ) (h : isPerfectSquare (2^n + 3)) : n = 0 :=
sorry

-- Second Lean 4 statement for 2^n + 1 = x^2
theorem problem2 (n : ℕ) (h : isPerfectSquare (2^n + 1)) : n = 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1453_145318


namespace NUMINAMATH_GPT_ones_digit_of_3_pow_52_l1453_145396

theorem ones_digit_of_3_pow_52 : (3 ^ 52 % 10) = 1 := 
by sorry

end NUMINAMATH_GPT_ones_digit_of_3_pow_52_l1453_145396


namespace NUMINAMATH_GPT_sum_of_consecutive_even_integers_l1453_145320

theorem sum_of_consecutive_even_integers (a : ℤ) (h : a + (a + 6) = 136) :
  a + (a + 2) + (a + 4) + (a + 6) = 272 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_even_integers_l1453_145320


namespace NUMINAMATH_GPT_set_union_example_l1453_145398

theorem set_union_example (M N : Set ℕ) (hM : M = {1, 2}) (hN : N = {2, 3}) : M ∪ N = {1, 2, 3} := 
by
  sorry

end NUMINAMATH_GPT_set_union_example_l1453_145398


namespace NUMINAMATH_GPT_jack_second_half_time_l1453_145330

variable (time_half1 time_half2 time_jack_total time_jill_total : ℕ)

theorem jack_second_half_time (h1 : time_half1 = 19) 
                              (h2 : time_jill_total = 32) 
                              (h3 : time_jack_total + 7 = time_jill_total) :
  time_jack_total = time_half1 + time_half2 → time_half2 = 6 := by
  sorry

end NUMINAMATH_GPT_jack_second_half_time_l1453_145330


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1453_145363

theorem quadratic_inequality_solution (x : ℝ) : x^2 - 36 * x + 318 ≤ 0 ↔ 18 - Real.sqrt 6 ≤ x ∧ x ≤ 18 + Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1453_145363


namespace NUMINAMATH_GPT_parabola_distance_l1453_145382

theorem parabola_distance (x y : ℝ) (h_parabola : y^2 = 8 * x)
  (h_distance_focus : ∀ x y, (x - 2)^2 + y^2 = 6^2) :
  abs x = 4 :=
by sorry

end NUMINAMATH_GPT_parabola_distance_l1453_145382


namespace NUMINAMATH_GPT_mia_12th_roll_last_is_approximately_027_l1453_145361

noncomputable def mia_probability_last_roll_on_12th : ℚ :=
  (5/6) ^ 10 * (1/6)

theorem mia_12th_roll_last_is_approximately_027 : 
  abs (mia_probability_last_roll_on_12th - 0.027) < 0.001 :=
sorry

end NUMINAMATH_GPT_mia_12th_roll_last_is_approximately_027_l1453_145361


namespace NUMINAMATH_GPT_prime_factors_of_expression_l1453_145309

theorem prime_factors_of_expression
  (p : ℕ) (prime_p : Nat.Prime p) :
  (∀ x y : ℕ, 0 < x → 0 < y → p ∣ ((x + y)^19 - x^19 - y^19)) ↔ (p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) :=
by
  sorry

end NUMINAMATH_GPT_prime_factors_of_expression_l1453_145309


namespace NUMINAMATH_GPT_sum_of_smallest_and_largest_even_l1453_145375

theorem sum_of_smallest_and_largest_even (n : ℤ) (h : n + (n + 2) + (n + 4) = 1194) : n + (n + 4) = 796 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_smallest_and_largest_even_l1453_145375


namespace NUMINAMATH_GPT_value_of_x_l1453_145323

theorem value_of_x (x : ℝ) : abs (4 * x - 8) ≤ 0 ↔ x = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_x_l1453_145323


namespace NUMINAMATH_GPT_constant_expression_l1453_145384

-- Suppose x is a real number
variable {x : ℝ}

-- Define the expression sum
def expr_sum (x : ℝ) : ℝ :=
|3 * x - 1| + |4 * x - 1| + |5 * x - 1| + |6 * x - 1| + 
|7 * x - 1| + |8 * x - 1| + |9 * x - 1| + |10 * x - 1| + 
|11 * x - 1| + |12 * x - 1| + |13 * x - 1| + |14 * x - 1| + 
|15 * x - 1| + |16 * x - 1| + |17 * x - 1|

-- The Lean statement of the problem to be proven
theorem constant_expression : (∃ x : ℝ, expr_sum x = 5) :=
sorry

end NUMINAMATH_GPT_constant_expression_l1453_145384


namespace NUMINAMATH_GPT_find_a_l1453_145327

theorem find_a (a : ℝ) (h : (a - 1) ≠ 0) :
  (∃ x : ℝ, ((a - 1) * x^2 + x + a^2 - 1 = 0) ∧ x = 0) → a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1453_145327


namespace NUMINAMATH_GPT_pencil_length_l1453_145350

theorem pencil_length (L : ℝ) (h1 : (1 / 8) * L + (1 / 2) * (7 / 8) * L + (7 / 2) = L) : L = 16 :=
by
  sorry

end NUMINAMATH_GPT_pencil_length_l1453_145350


namespace NUMINAMATH_GPT_plot_length_l1453_145314

variable (b length : ℝ)

theorem plot_length (h1 : length = b + 10)
  (fence_N_cost : ℝ := 26.50 * (b + 10))
  (fence_E_cost : ℝ := 32 * b)
  (fence_S_cost : ℝ := 22 * (b + 10))
  (fence_W_cost : ℝ := 30 * b)
  (total_cost : ℝ := fence_N_cost + fence_E_cost + fence_S_cost + fence_W_cost)
  (h2 : 1.05 * total_cost = 7500) :
  length = 70.25 := by
  sorry

end NUMINAMATH_GPT_plot_length_l1453_145314


namespace NUMINAMATH_GPT_johns_father_fraction_l1453_145306

theorem johns_father_fraction (total_money : ℝ) (given_to_mother_fraction remaining_after_father : ℝ) :
  total_money = 200 →
  given_to_mother_fraction = 3 / 8 →
  remaining_after_father = 65 →
  ((total_money - given_to_mother_fraction * total_money) - remaining_after_father) / total_money
  = 3 / 10 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_johns_father_fraction_l1453_145306


namespace NUMINAMATH_GPT_floor_of_neg_sqrt_frac_l1453_145308

theorem floor_of_neg_sqrt_frac :
  (Int.floor (-Real.sqrt (64 / 9)) = -3) :=
by
  sorry

end NUMINAMATH_GPT_floor_of_neg_sqrt_frac_l1453_145308


namespace NUMINAMATH_GPT_twenty_five_percent_less_than_80_one_fourth_more_l1453_145390

theorem twenty_five_percent_less_than_80_one_fourth_more (n : ℕ) (h : (5 / 4 : ℝ) * n = 60) : n = 48 :=
by
  sorry

end NUMINAMATH_GPT_twenty_five_percent_less_than_80_one_fourth_more_l1453_145390


namespace NUMINAMATH_GPT_exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l1453_145377

variable (m : ℝ)
def f (x : ℝ) : ℝ := x^2 + m*x + 1

theorem exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2 :
  (∃ x0 : ℝ, x0 > 0 ∧ f m x0 < 0) → m < -2 := by
  sorry

end NUMINAMATH_GPT_exists_x0_gt_0_f_x0_lt_0_implies_m_lt_neg_2_l1453_145377


namespace NUMINAMATH_GPT_area_of_square_STUV_l1453_145389

-- Defining the conditions
variable (C L : ℝ)
variable (h1 : 2 * (C + L) = 40)

-- The goal is to prove the area of the square STUV
theorem area_of_square_STUV : (C + L) * (C + L) = 400 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_STUV_l1453_145389


namespace NUMINAMATH_GPT_compound_interest_correct_l1453_145397

noncomputable def compound_interest (P R T : ℝ) : ℝ :=
  P * (1 + R / 100)^T - P

theorem compound_interest_correct (SI R T : ℝ) (hSI : SI = 58) (hR : R = 5) (hT : T = 2) : 
  compound_interest (SI * 100 / (R * T)) R T = 59.45 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_correct_l1453_145397


namespace NUMINAMATH_GPT_eiffel_tower_model_ratio_l1453_145376

/-- Define the conditions of the problem as a structure -/
structure ModelCondition where
  eiffelTowerHeight : ℝ := 984 -- in feet
  modelHeight : ℝ := 6        -- in inches

/-- The main theorem statement -/
theorem eiffel_tower_model_ratio (cond : ModelCondition) : cond.eiffelTowerHeight / cond.modelHeight = 164 := 
by
  -- We can leave the proof out with 'sorry' for now.
  sorry

end NUMINAMATH_GPT_eiffel_tower_model_ratio_l1453_145376


namespace NUMINAMATH_GPT_find_m_l1453_145316

theorem find_m (n : ℝ) : 21 * (m + n) + 21 = 21 * (-m + n) + 21 → m = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1453_145316


namespace NUMINAMATH_GPT_ethan_hours_per_day_l1453_145353

-- Define the known constants
def hourly_wage : ℝ := 18
def work_days_per_week : ℕ := 5
def total_earnings : ℝ := 3600
def weeks_worked : ℕ := 5

-- Define the main theorem
theorem ethan_hours_per_day :
  (∃ hours_per_day : ℝ, 
    hours_per_day = total_earnings / (weeks_worked * work_days_per_week * hourly_wage)) →
  hours_per_day = 8 :=
by
  sorry

end NUMINAMATH_GPT_ethan_hours_per_day_l1453_145353


namespace NUMINAMATH_GPT_monotonically_increasing_range_l1453_145328

theorem monotonically_increasing_range (a : ℝ) : 
  (0 < a ∧ a < 1) ∧ (∀ x : ℝ, 0 < x → (a^x + (1 + a)^x) ≥ (a^(x - 1) + (1 + a)^(x - 1))) → 
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) :=
by
  sorry

end NUMINAMATH_GPT_monotonically_increasing_range_l1453_145328


namespace NUMINAMATH_GPT_part1_part2_l1453_145358

-- Conditions: Definitions of A and B
def A (a b : ℝ) : ℝ := 2 * a^2 - 5 * a * b + 3 * b
def B (a b : ℝ) : ℝ := 4 * a^2 - 6 * a * b - 8 * a

-- Theorem statements
theorem part1 (a b : ℝ) :  2 * A a b - B a b = -4 * a * b + 6 * b + 8 * a := sorry

theorem part2 (a : ℝ) (h : ∀ a, 2 * A a 2 - B a 2 = - 4 * a * 2 + 6 * 2 + 8 * a) : 2 = 2 := sorry

end NUMINAMATH_GPT_part1_part2_l1453_145358


namespace NUMINAMATH_GPT_prob_all_fail_prob_at_least_one_pass_l1453_145332

def prob_pass := 1 / 2
def prob_fail := 1 - prob_pass

def indep (A B C : Prop) : Prop := true -- Usually we prove independence in a detailed manner, but let's assume it's given as true.

theorem prob_all_fail (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : (prob_fail * prob_fail * prob_fail) = 1 / 8 :=
by
  sorry

theorem prob_at_least_one_pass (A B C : Prop) (hA : prob_pass = 1 / 2) (hB : prob_pass = 1 / 2) (hC : prob_pass = 1 / 2) 
  (indepABC : indep A B C) : 1 - (prob_fail * prob_fail * prob_fail) = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_prob_all_fail_prob_at_least_one_pass_l1453_145332


namespace NUMINAMATH_GPT_shaina_chocolate_l1453_145322

-- Define the conditions
def total_chocolate : ℚ := 48 / 5
def number_of_piles : ℚ := 4

-- Define the assertion to prove
theorem shaina_chocolate : (total_chocolate / number_of_piles) = (12 / 5) := 
by 
  sorry

end NUMINAMATH_GPT_shaina_chocolate_l1453_145322


namespace NUMINAMATH_GPT_car_count_l1453_145395

theorem car_count (x y : ℕ) (h1 : x + y = 36) (h2 : 6 * x + 4 * y = 176) :
  x = 16 ∧ y = 20 :=
by
  sorry

end NUMINAMATH_GPT_car_count_l1453_145395


namespace NUMINAMATH_GPT_skyscraper_anniversary_l1453_145317

theorem skyscraper_anniversary 
  (years_since_built : ℕ)
  (target_years : ℕ)
  (years_before_200th : ℕ)
  (years_future : ℕ) 
  (h1 : years_since_built = 100) 
  (h2 : target_years = 200 - 5) 
  (h3 : years_future = target_years - years_since_built) : 
  years_future = 95 :=
by
  sorry

end NUMINAMATH_GPT_skyscraper_anniversary_l1453_145317


namespace NUMINAMATH_GPT_min_sum_equals_nine_l1453_145380

theorem min_sum_equals_nine (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 4 * a + b - a * b = 0) : a + b = 9 :=
by
  sorry

end NUMINAMATH_GPT_min_sum_equals_nine_l1453_145380


namespace NUMINAMATH_GPT_find_a_from_function_property_l1453_145368

theorem find_a_from_function_property {a : ℝ} (h : ∀ (x : ℝ), (0 ≤ x → x ≤ 1 → ax ≤ 3) ∧ (0 ≤ x → x ≤ 1 → ax ≥ 3)) :
  a = 3 :=
sorry

end NUMINAMATH_GPT_find_a_from_function_property_l1453_145368


namespace NUMINAMATH_GPT_negate_proposition_l1453_145391

variable (x : ℝ)

theorem negate_proposition :
  (¬ (∃ x₀ : ℝ, x₀^2 - x₀ + 1/4 ≤ 0)) ↔ ∀ x : ℝ, x^2 - x + 1/4 > 0 :=
by
  sorry

end NUMINAMATH_GPT_negate_proposition_l1453_145391


namespace NUMINAMATH_GPT_pencils_distributed_per_container_l1453_145321

noncomputable def total_pencils (initial_pencils : ℕ) (additional_pencils : ℕ) : ℕ :=
  initial_pencils + additional_pencils

noncomputable def pencils_per_container (total_pencils : ℕ) (num_containers : ℕ) : ℕ :=
  total_pencils / num_containers

theorem pencils_distributed_per_container :
  let initial_pencils := 150
  let additional_pencils := 30
  let num_containers := 5
  let total := total_pencils initial_pencils additional_pencils
  let pencils_per_container := pencils_per_container total num_containers
  pencils_per_container = 36 :=
by {
  -- sorry is used to skip the proof
  -- the actual proof is not required
  sorry
}

end NUMINAMATH_GPT_pencils_distributed_per_container_l1453_145321


namespace NUMINAMATH_GPT_insects_in_lab_l1453_145364

theorem insects_in_lab (total_legs number_of_legs_per_insect : ℕ) (h1 : total_legs = 36) (h2 : number_of_legs_per_insect = 6) : (total_legs / number_of_legs_per_insect) = 6 :=
by
  sorry

end NUMINAMATH_GPT_insects_in_lab_l1453_145364


namespace NUMINAMATH_GPT_cucumbers_count_l1453_145307

theorem cucumbers_count (c : ℕ) (n : ℕ) (additional : ℕ) (initial_cucumbers : ℕ) (total_cucumbers : ℕ) :
  c = 4 → n = 10 → additional = 2 → initial_cucumbers = n - c → total_cucumbers = initial_cucumbers + additional → total_cucumbers = 8 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2] at h4
  simp at h4
  rw [h4, h3] at h5
  simp at h5
  exact h5

end NUMINAMATH_GPT_cucumbers_count_l1453_145307


namespace NUMINAMATH_GPT_gyeonghun_climbing_l1453_145310

variable (t_up t_down d_up d_down : ℝ)
variable (h1 : t_up + t_down = 4) 
variable (h2 : d_down = d_up + 2)
variable (h3 : t_up = d_up / 3)
variable (h4 : t_down = d_down / 4)

theorem gyeonghun_climbing (h1 : t_up + t_down = 4) (h2 : d_down = d_up + 2) (h3 : t_up = d_up / 3) (h4 : t_down = d_down / 4) :
  t_up = 2 :=
by
  sorry

end NUMINAMATH_GPT_gyeonghun_climbing_l1453_145310


namespace NUMINAMATH_GPT_exists_100_integers_with_distinct_pairwise_sums_l1453_145329

-- Define number of integers and the constraint limit
def num_integers : ℕ := 100
def max_value : ℕ := 25000

-- Define the predicate for all pairwise sums being different
def pairwise_different_sums (as : Fin num_integers → ℕ) : Prop :=
  ∀ i j k l : Fin num_integers, i ≠ j ∧ k ≠ l → as i + as j ≠ as k + as l

-- Main theorem statement
theorem exists_100_integers_with_distinct_pairwise_sums :
  ∃ as : Fin num_integers → ℕ, (∀ i : Fin num_integers, as i > 0 ∧ as i ≤ max_value) ∧ pairwise_different_sums as :=
sorry

end NUMINAMATH_GPT_exists_100_integers_with_distinct_pairwise_sums_l1453_145329


namespace NUMINAMATH_GPT_coordinates_of_point_l1453_145303

noncomputable def point_on_x_axis (x : ℝ) :=
  (x, 0)

theorem coordinates_of_point (x : ℝ) (hx : abs x = 3) :
  point_on_x_axis x = (3, 0) ∨ point_on_x_axis x = (-3, 0) :=
  sorry

end NUMINAMATH_GPT_coordinates_of_point_l1453_145303


namespace NUMINAMATH_GPT_random_event_sum_gt_six_l1453_145370

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def selection (s : List ℕ) := s.length = 3 ∧ s ⊆ numbers

def sum_is_greater_than_six (s : List ℕ) : Prop := s.sum > 6

theorem random_event_sum_gt_six :
  ∀ (s : List ℕ), selection s → (sum_is_greater_than_six s ∨ ¬ sum_is_greater_than_six s) := 
by
  intros s h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_random_event_sum_gt_six_l1453_145370


namespace NUMINAMATH_GPT_ratio_roots_l1453_145394

theorem ratio_roots (p q r s : ℤ)
    (h1 : p ≠ 0)
    (h_roots : ∀ x : ℤ, (x = -1 ∨ x = 3 ∨ x = 4) → (p*x^3 + q*x^2 + r*x + s = 0)) : 
    (r : ℚ) / s = -5 / 12 :=
by sorry

end NUMINAMATH_GPT_ratio_roots_l1453_145394


namespace NUMINAMATH_GPT_calculate_spadesuit_l1453_145378

def spadesuit (x y : ℝ) : ℝ :=
  (x + y) * (x - y)

theorem calculate_spadesuit : spadesuit 3 (spadesuit 5 6) = -112 := by
  sorry

end NUMINAMATH_GPT_calculate_spadesuit_l1453_145378


namespace NUMINAMATH_GPT_correct_option_B_l1453_145385

def linear_function (x : ℝ) : ℝ := -x + 2

theorem correct_option_B :
  ∃ x : ℝ, linear_function x = 0 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_correct_option_B_l1453_145385


namespace NUMINAMATH_GPT_minimize_f_l1453_145373

noncomputable def f : ℝ → ℝ := λ x => (3/2) * x^2 - 9 * x + 7

theorem minimize_f : ∀ x, f x ≥ f 3 :=
by 
  intro x
  sorry

end NUMINAMATH_GPT_minimize_f_l1453_145373


namespace NUMINAMATH_GPT_tom_dollars_more_than_jerry_l1453_145347

theorem tom_dollars_more_than_jerry (total_slices : ℕ)
  (jerry_slices : ℕ)
  (tom_slices : ℕ)
  (plain_cost : ℕ)
  (pineapple_additional_cost : ℕ)
  (total_cost : ℕ)
  (cost_per_slice : ℚ)
  (cost_jerry : ℚ)
  (cost_tom : ℚ)
  (jerry_ate_plain : jerry_slices = 5)
  (tom_ate_pineapple : tom_slices = 5)
  (total_slices_10 : total_slices = 10)
  (plain_cost_10 : plain_cost = 10)
  (pineapple_additional_cost_3 : pineapple_additional_cost = 3)
  (total_cost_13 : total_cost = plain_cost + pineapple_additional_cost)
  (cost_per_slice_calc : cost_per_slice = total_cost / total_slices)
  (cost_jerry_calc : cost_jerry = cost_per_slice * jerry_slices)
  (cost_tom_calc : cost_tom = cost_per_slice * tom_slices) :
  cost_tom - cost_jerry = 0 := by
  sorry

end NUMINAMATH_GPT_tom_dollars_more_than_jerry_l1453_145347


namespace NUMINAMATH_GPT_find_m_l1453_145343

-- Define the operation a * b
def star (a b : ℝ) : ℝ := a * b + a - 2 * b

theorem find_m (m : ℝ) (h : star 3 m = 17) : m = 14 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_m_l1453_145343


namespace NUMINAMATH_GPT_total_cantaloupes_l1453_145388

def Fred_grew_38 : ℕ := 38
def Tim_grew_44 : ℕ := 44

theorem total_cantaloupes : Fred_grew_38 + Tim_grew_44 = 82 := by
  sorry

end NUMINAMATH_GPT_total_cantaloupes_l1453_145388


namespace NUMINAMATH_GPT_line_condition_l1453_145372

/-- Given a line l1 passing through points A(-2, m) and B(m, 4),
    a line l2 given by the equation 2x + y - 1 = 0,
    and a line l3 given by the equation x + ny + 1 = 0,
    if l1 is parallel to l2 and l2 is perpendicular to l3,
    then the value of m + n is -10. -/
theorem line_condition (m n : ℝ) (h1 : (4 - m) / (m + 2) = -2)
  (h2 : (2 * -1) * (-1 / n) = -1) : m + n = -10 := 
sorry

end NUMINAMATH_GPT_line_condition_l1453_145372


namespace NUMINAMATH_GPT_contrapositive_of_squared_sum_eq_zero_l1453_145369

theorem contrapositive_of_squared_sum_eq_zero (a b : ℝ) :
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) ↔ (a ≠ 0 ∨ b ≠ 0 → a^2 + b^2 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_squared_sum_eq_zero_l1453_145369


namespace NUMINAMATH_GPT_smallest_d_for_inequality_l1453_145338

open Real

theorem smallest_d_for_inequality :
  (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + 1 * |x^2 - y^2| ≥ exp ((x + y) / 2)) ∧
  (∀ d > 0, (∀ (x y : ℝ), 0 ≤ x → 0 ≤ y → exp (x * y) + d * |x^2 - y^2| ≥ exp ((x + y) / 2)) → d ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_d_for_inequality_l1453_145338


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l1453_145357

noncomputable def f (x : ℝ) (m : ℝ) := Real.sqrt (|x + 2| + |x - 4| - m)

theorem problem_part1 (m : ℝ) : 
  (∀ x : ℝ, |x + 2| + |x - 4| - m ≥ 0) ↔ m ≤ 6 := 
by
  sorry

theorem problem_part2 (a b : ℕ) (n : ℝ) (h1 : (0 < a) ∧ (0 < b)) (h2 : n = 6) 
  (h3 : (4 / (a + 5 * b)) + (1 / (3 * a + 2 * b)) = n) : 
  ∃ (value : ℝ), 4 * a + 7 * b = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l1453_145357


namespace NUMINAMATH_GPT_running_time_difference_l1453_145340

theorem running_time_difference :
  ∀ (distance speed usual_speed : ℝ), 
  distance = 30 →
  usual_speed = 10 →
  speed = (distance / (usual_speed / 2)) - (distance / (usual_speed * 1.5)) →
  speed = 4 :=
by
  intros distance speed usual_speed hd hu hs
  sorry

end NUMINAMATH_GPT_running_time_difference_l1453_145340


namespace NUMINAMATH_GPT_mrs_hilt_found_nickels_l1453_145301

theorem mrs_hilt_found_nickels : 
  ∀ (total cents quarter cents dime cents nickel cents : ℕ), 
    total = 45 → 
    quarter = 25 → 
    dime = 10 → 
    nickel = 5 → 
    ((total - (quarter + dime)) / nickel) = 2 := 
by
  intros total quarter dime nickel h_total h_quarter h_dime h_nickel
  sorry

end NUMINAMATH_GPT_mrs_hilt_found_nickels_l1453_145301


namespace NUMINAMATH_GPT_infinite_perfect_squares_in_ap_l1453_145356

open Nat

def is_arithmetic_progression (a d : ℕ) (an : ℕ → ℕ) : Prop :=
  ∀ n, an n = a + n * d

def is_perfect_square (x : ℕ) : Prop :=
  ∃ m, m * m = x

theorem infinite_perfect_squares_in_ap (a d : ℕ) (an : ℕ → ℕ) (m : ℕ)
  (h_arith_prog : is_arithmetic_progression a d an)
  (h_initial_square : a = m * m) :
  ∃ (f : ℕ → ℕ), ∀ n, is_perfect_square (an (f n)) :=
sorry

end NUMINAMATH_GPT_infinite_perfect_squares_in_ap_l1453_145356


namespace NUMINAMATH_GPT_min_total_balls_l1453_145387

theorem min_total_balls (R G B : Nat) (hG : G = 12) (hRG : R + G < 24) : 23 ≤ R + G + B :=
by {
  sorry
}

end NUMINAMATH_GPT_min_total_balls_l1453_145387


namespace NUMINAMATH_GPT_negation_proposition_l1453_145334

theorem negation_proposition (l : ℝ) (h : l = 1) : 
  (¬ ∃ x : ℝ, x + l ≥ 0) = (∀ x : ℝ, x + l < 0) := by 
  sorry

end NUMINAMATH_GPT_negation_proposition_l1453_145334


namespace NUMINAMATH_GPT_determine_a_l1453_145305

theorem determine_a (a : ℚ) (x : ℚ) : 
  (∃ r s : ℚ, (r*x + s)^2 = a*x^2 + 18*x + 16) → 
  a = 81/16 := 
sorry

end NUMINAMATH_GPT_determine_a_l1453_145305


namespace NUMINAMATH_GPT_work_done_together_in_six_days_l1453_145351

theorem work_done_together_in_six_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
  1 / (A + B) = 6 :=
by
  sorry

end NUMINAMATH_GPT_work_done_together_in_six_days_l1453_145351


namespace NUMINAMATH_GPT_largest_plot_area_l1453_145335

def plotA_area : Real := 10
def plotB_area : Real := 10 + 1
def plotC_area : Real := 9 + 1.5
def plotD_area : Real := 12
def plotE_area : Real := 11 + 1

theorem largest_plot_area :
  max (max (max (max plotA_area plotB_area) plotC_area) plotD_area) plotE_area = 12 ∧ 
  (plotD_area = 12 ∧ plotE_area = 12) := by sorry

end NUMINAMATH_GPT_largest_plot_area_l1453_145335


namespace NUMINAMATH_GPT_sum_first_six_terms_geometric_sequence_l1453_145399

theorem sum_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * ((1 - r^n) / (1 - r))
  S_n = 455 / 1365 := by
  sorry

end NUMINAMATH_GPT_sum_first_six_terms_geometric_sequence_l1453_145399


namespace NUMINAMATH_GPT_cost_per_use_l1453_145333

def cost : ℕ := 30
def uses_in_a_week : ℕ := 3
def weeks : ℕ := 2
def total_uses : ℕ := uses_in_a_week * weeks

theorem cost_per_use : cost / total_uses = 5 := by
  sorry

end NUMINAMATH_GPT_cost_per_use_l1453_145333


namespace NUMINAMATH_GPT_y_plus_z_value_l1453_145371

theorem y_plus_z_value (v w x y z S : ℕ) 
  (h1 : 196 + x + y = S)
  (h2 : 269 + z + 123 = S)
  (h3 : 50 + x + z = S) : 
  y + z = 196 := 
sorry

end NUMINAMATH_GPT_y_plus_z_value_l1453_145371


namespace NUMINAMATH_GPT_group_is_abelian_l1453_145337

variable {G : Type} [Group G]
variable (e : G)
variable (h : ∀ x : G, x * x = e)

theorem group_is_abelian (a b : G) : a * b = b * a :=
sorry

end NUMINAMATH_GPT_group_is_abelian_l1453_145337


namespace NUMINAMATH_GPT_sticker_price_l1453_145392

theorem sticker_price (x : ℝ) (h1 : 0.8 * x - 100 = 0.7 * x - 25) : x = 750 :=
by
  sorry

end NUMINAMATH_GPT_sticker_price_l1453_145392


namespace NUMINAMATH_GPT_dogs_food_consumption_l1453_145359

theorem dogs_food_consumption :
  (let cups_per_meal_momo_fifi := 1.5
   let meals_per_day := 3
   let cups_per_meal_gigi := 2
   let cups_to_pounds := 3
   let daily_food_momo_fifi := cups_per_meal_momo_fifi * meals_per_day * 2
   let daily_food_gigi := cups_per_meal_gigi * meals_per_day
   daily_food_momo_fifi + daily_food_gigi) / cups_to_pounds = 5 :=
by
  sorry

end NUMINAMATH_GPT_dogs_food_consumption_l1453_145359


namespace NUMINAMATH_GPT_smallest_angle_in_triangle_l1453_145352

open Real

theorem smallest_angle_in_triangle
  (a b c : ℝ)
  (h : a = (b + c) / 3)
  (triangle_inequality_1 : a + b > c)
  (triangle_inequality_2 : a + c > b)
  (triangle_inequality_3 : b + c > a) :
  ∃ A B C α β γ : ℝ, -- A, B, C are the angles opposite to sides a, b, c respectively
  0 < α ∧ α < β ∧ α < γ :=
sorry

end NUMINAMATH_GPT_smallest_angle_in_triangle_l1453_145352


namespace NUMINAMATH_GPT_derivative_at_one_is_four_l1453_145354

-- Define the function y = x^2 + 2x + 1
def f (x : ℝ) := x^2 + 2*x + 1

-- State the theorem: The derivative of f at x = 1 is 4
theorem derivative_at_one_is_four : (deriv f 1) = 4 :=
by
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_derivative_at_one_is_four_l1453_145354


namespace NUMINAMATH_GPT_fill_half_jar_in_18_days_l1453_145386

-- Define the doubling condition and the days required to fill half the jar
variable (area : ℕ → ℕ)
variable (doubling : ∀ t, area (t + 1) = 2 * area t)
variable (full_jar : area 19 = 2^19)
variable (half_jar : area 18 = 2^18)

theorem fill_half_jar_in_18_days :
  ∃ n, n = 18 ∧ area n = 2^18 :=
by {
  -- The proof is omitted, but we state the goal
  sorry
}

end NUMINAMATH_GPT_fill_half_jar_in_18_days_l1453_145386


namespace NUMINAMATH_GPT_complex_number_solution_l1453_145324

theorem complex_number_solution (a : ℝ) (h : (⟨a, 1⟩ : ℂ) * ⟨1, -a⟩ = (2 : ℂ)) : a = 1 :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l1453_145324


namespace NUMINAMATH_GPT_opposite_of_4_l1453_145319

theorem opposite_of_4 : ∃ x, 4 + x = 0 ∧ x = -4 :=
by sorry

end NUMINAMATH_GPT_opposite_of_4_l1453_145319


namespace NUMINAMATH_GPT_weight_of_berries_l1453_145336

theorem weight_of_berries (total_weight : ℝ) (melon_weight : ℝ) : total_weight = 0.63 → melon_weight = 0.25 → total_weight - melon_weight = 0.38 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_weight_of_berries_l1453_145336


namespace NUMINAMATH_GPT_bird_cages_count_l1453_145312

-- Definitions based on the conditions provided
def num_parrots_per_cage : ℕ := 2
def num_parakeets_per_cage : ℕ := 7
def total_birds_per_cage : ℕ := num_parrots_per_cage + num_parakeets_per_cage
def total_birds_in_store : ℕ := 54
def num_bird_cages : ℕ := total_birds_in_store / total_birds_per_cage

-- The proof we need to derive
theorem bird_cages_count : num_bird_cages = 6 := by
  sorry

end NUMINAMATH_GPT_bird_cages_count_l1453_145312


namespace NUMINAMATH_GPT_right_triangle_other_acute_angle_l1453_145355

theorem right_triangle_other_acute_angle (A B C : ℝ) (r : A + B + C = 180) (h : A = 90) (a : B = 30) :
  C = 60 :=
sorry

end NUMINAMATH_GPT_right_triangle_other_acute_angle_l1453_145355


namespace NUMINAMATH_GPT_probability_of_exactly_three_blue_marbles_l1453_145325

-- Define the conditions
def total_marbles : ℕ := 15
def blue_marbles : ℕ := 8
def red_marbles : ℕ := 7
def total_selections : ℕ := 6
def blue_selections : ℕ := 3
def blue_probability : ℚ := 8 / 15
def red_probability : ℚ := 7 / 15
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Define the binomial probability formula calculation
def binomial_probability : ℚ :=
  binomial_coefficient total_selections blue_selections * (blue_probability ^ blue_selections) * (red_probability ^ (total_selections - blue_selections))

-- The hypothesis (conditions) and conclusion (the solution)
theorem probability_of_exactly_three_blue_marbles :
  binomial_probability = (3512320 / 11390625) :=
by sorry

end NUMINAMATH_GPT_probability_of_exactly_three_blue_marbles_l1453_145325


namespace NUMINAMATH_GPT_measure_of_two_equal_angles_l1453_145379

noncomputable def measure_of_obtuse_angle (θ : ℝ) : ℝ := θ + (0.6 * θ)

-- Given conditions
def is_obtuse_isosceles_triangle (θ : ℝ) : Prop :=
  θ = 90 ∧ measure_of_obtuse_angle 90 = 144 ∧ 180 - 144 = 36

-- The main theorem
theorem measure_of_two_equal_angles :
  ∀ θ, is_obtuse_isosceles_triangle θ → 36 / 2 = 18 :=
by
  intros θ h
  sorry

end NUMINAMATH_GPT_measure_of_two_equal_angles_l1453_145379


namespace NUMINAMATH_GPT_allison_upload_ratio_l1453_145348

theorem allison_upload_ratio :
  ∃ (x y : ℕ), (x + y = 30) ∧ (10 * x + 20 * y = 450) ∧ (x / 30 = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_allison_upload_ratio_l1453_145348


namespace NUMINAMATH_GPT_certain_number_eq_40_l1453_145313

theorem certain_number_eq_40 (x : ℝ) 
    (h : (20 + x + 60) / 3 = (20 + 60 + 25) / 3 + 5) : x = 40 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_eq_40_l1453_145313


namespace NUMINAMATH_GPT_area_of_rhombus_with_diagonals_6_and_8_l1453_145365

theorem area_of_rhombus_with_diagonals_6_and_8 : 
  ∀ (d1 d2 : ℕ), d1 = 6 → d2 = 8 → (1 / 2 : ℝ) * d1 * d2 = 24 :=
by
  intros d1 d2 h1 h2
  sorry

end NUMINAMATH_GPT_area_of_rhombus_with_diagonals_6_and_8_l1453_145365


namespace NUMINAMATH_GPT_intersection_volume_is_zero_l1453_145362

-- Definitions of the regions
def region1 (x y z : ℝ) : Prop := |x| + |y| + |z| ≤ 2
def region2 (x y z : ℝ) : Prop := |x| + |y| + |z - 2| ≤ 1

-- Main theorem stating the volume of their intersection
theorem intersection_volume_is_zero : 
  ∀ (x y z : ℝ), region1 x y z ∧ region2 x y z → (x = 0 ∧ y = 0 ∧ z = 2) := 
sorry

end NUMINAMATH_GPT_intersection_volume_is_zero_l1453_145362


namespace NUMINAMATH_GPT_sector_area_l1453_145326

theorem sector_area (radius : ℝ) (central_angle : ℝ) (h1 : radius = 3) (h2 : central_angle = 2 * Real.pi / 3) : 
    (1 / 2) * radius^2 * central_angle = 6 * Real.pi :=
by
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_sector_area_l1453_145326


namespace NUMINAMATH_GPT_solve_for_s_l1453_145331

theorem solve_for_s : ∃ (s t : ℚ), (8 * s + 7 * t = 160) ∧ (s = t - 3) ∧ (s = 139 / 15) := by
  sorry

end NUMINAMATH_GPT_solve_for_s_l1453_145331


namespace NUMINAMATH_GPT_student_correct_answers_l1453_145302

-- Definitions based on the conditions
def total_questions : ℕ := 100
def score (correct incorrect : ℕ) : ℕ := correct - 2 * incorrect
def studentScore : ℕ := 73

-- Main theorem to prove
theorem student_correct_answers (C I : ℕ) (h1 : C + I = total_questions) (h2 : score C I = studentScore) : C = 91 :=
by
  sorry

end NUMINAMATH_GPT_student_correct_answers_l1453_145302


namespace NUMINAMATH_GPT_p_sufficient_for_q_iff_l1453_145381

-- Definitions based on conditions
def p (x : ℝ) : Prop := x^2 - 2 * x - 8 ≤ 0
def q (x : ℝ) (m : ℝ) : Prop := (x - (1 - m)) * (x - (1 + m)) ≤ 0
def m_condition (m : ℝ) : Prop := m < 0

-- The statement to prove
theorem p_sufficient_for_q_iff (m : ℝ) :
  (∀ x, p x → q x m) ↔ m <= -3 :=
by
  sorry

-- noncomputable theory is not necessary here since all required functions are computable.

end NUMINAMATH_GPT_p_sufficient_for_q_iff_l1453_145381


namespace NUMINAMATH_GPT_probability_suitable_joint_given_physique_l1453_145345

noncomputable def total_children : ℕ := 20
noncomputable def suitable_physique : ℕ := 4
noncomputable def suitable_joint_structure : ℕ := 5
noncomputable def both_physique_and_joint : ℕ := 2

noncomputable def P (n m : ℕ) : ℚ := n / m

theorem probability_suitable_joint_given_physique :
  P both_physique_and_joint total_children / P suitable_physique total_children = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_probability_suitable_joint_given_physique_l1453_145345


namespace NUMINAMATH_GPT_stratified_sampling_correct_l1453_145304

def num_students := 500
def num_male_students := 500
def num_female_students := 400
def ratio_male_female := num_male_students / num_female_students

def selected_male_students := 25
def selected_female_students := (selected_male_students * num_female_students) / num_male_students

theorem stratified_sampling_correct :
  selected_female_students = 20 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_correct_l1453_145304


namespace NUMINAMATH_GPT_circumradius_of_right_triangle_l1453_145360

theorem circumradius_of_right_triangle (a b c : ℕ) (h : a = 8 ∧ b = 15 ∧ c = 17) : 
  ∃ R : ℝ, R = 8.5 :=
by
  sorry

end NUMINAMATH_GPT_circumradius_of_right_triangle_l1453_145360
