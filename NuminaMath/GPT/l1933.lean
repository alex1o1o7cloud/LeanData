import Mathlib

namespace NUMINAMATH_GPT_unique_solution_implies_relation_l1933_193327

theorem unique_solution_implies_relation (a b : ℝ)
    (h : ∃! (x y : ℝ), y = x^2 + a * x + b ∧ x = y^2 + a * y + b) : 
    a^2 = 2 * (a + 2 * b) - 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_implies_relation_l1933_193327


namespace NUMINAMATH_GPT_charge_difference_is_51_l1933_193314

-- Define the charges and calculations for print shop X
def print_shop_x_cost (n : ℕ) : ℝ :=
  if n ≤ 50 then n * 1.20 else 50 * 1.20 + (n - 50) * 0.90

-- Define the charges and calculations for print shop Y
def print_shop_y_cost (n : ℕ) : ℝ :=
  10 + n * 1.70

-- Define the difference in charges for 70 copies
def charge_difference : ℝ :=
  print_shop_y_cost 70 - print_shop_x_cost 70

-- The proof statement
theorem charge_difference_is_51 : charge_difference = 51 :=
by
  sorry

end NUMINAMATH_GPT_charge_difference_is_51_l1933_193314


namespace NUMINAMATH_GPT_find_m_l1933_193337

noncomputable def geometric_sequence_solution (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) (m : ℕ) : Prop :=
  (S 3 + S 6 = 2 * S 9) ∧ (a 2 + a 5 = 2 * a m)

theorem find_m (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ) (m : ℕ) (h1 : S 3 + S 6 = 2 * S 9)
  (h2 : a 2 + a 5 = 2 * a m) : m = 8 :=
sorry

end NUMINAMATH_GPT_find_m_l1933_193337


namespace NUMINAMATH_GPT_lines_are_skew_l1933_193396

def line1 (a t : ℝ) : ℝ × ℝ × ℝ :=
  (2 + 3 * t, 3 + 4 * t, a + 5 * t)

def line2 (u : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 6 * u, 2 + 2 * u, 1 + 2 * u)

theorem lines_are_skew (a : ℝ) :
  ¬(∃ t u : ℝ, line1 a t = line2 u) ↔ a ≠ 5 / 3 :=
sorry

end NUMINAMATH_GPT_lines_are_skew_l1933_193396


namespace NUMINAMATH_GPT_no_more_beverages_needed_l1933_193358

namespace HydrationPlan

def daily_water_need := 9
def daily_juice_need := 5
def daily_soda_need := 3
def days := 60

def total_water_needed := daily_water_need * days
def total_juice_needed := daily_juice_need * days
def total_soda_needed := daily_soda_need * days

def water_already_have := 617
def juice_already_have := 350
def soda_already_have := 215

theorem no_more_beverages_needed :
  (water_already_have >= total_water_needed) ∧ 
  (juice_already_have >= total_juice_needed) ∧ 
  (soda_already_have >= total_soda_needed) :=
by 
  -- proof goes here
  sorry

end HydrationPlan

end NUMINAMATH_GPT_no_more_beverages_needed_l1933_193358


namespace NUMINAMATH_GPT_valid_two_digit_numbers_l1933_193380

-- Define the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  (n % 10) + (n / 10 % 10) + (n / 100 % 10) + (n / 1000 % 10)

-- Prove the statement about two-digit numbers satisfying the condition
theorem valid_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A ≤ 99 ∧ (sum_of_digits A)^2 = sum_of_digits (A^2)} =
  {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by
  sorry

end NUMINAMATH_GPT_valid_two_digit_numbers_l1933_193380


namespace NUMINAMATH_GPT_coordinates_of_point_P_l1933_193352

noncomputable def tangent_slope_4 : Prop :=
  ∀ (x y : ℝ), y = 1 / x → (-1 / (x^2)) = -4 → (x = 1 / 2 ∧ y = 2) ∨ (x = -1 / 2 ∧ y = -2)

theorem coordinates_of_point_P : tangent_slope_4 :=
by sorry

end NUMINAMATH_GPT_coordinates_of_point_P_l1933_193352


namespace NUMINAMATH_GPT_sum_of_squares_remainder_l1933_193324

theorem sum_of_squares_remainder (n : ℕ) : 
  ((n - 1) ^ 2 + n ^ 2 + (n + 1) ^ 2) % 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_remainder_l1933_193324


namespace NUMINAMATH_GPT_inequality_proof_l1933_193390

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c ≤ 3) : 
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) ≥ 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1933_193390


namespace NUMINAMATH_GPT_trig_expression_equality_l1933_193309

theorem trig_expression_equality :
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  2 * tan_60 + tan_45 - 4 * cos_30 = 1 := by
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  sorry

end NUMINAMATH_GPT_trig_expression_equality_l1933_193309


namespace NUMINAMATH_GPT_sector_area_l1933_193365

noncomputable def area_of_sector (arc_length : ℝ) (central_angle : ℝ) (radius : ℝ) : ℝ :=
  1 / 2 * arc_length * radius

theorem sector_area (R : ℝ)
  (arc_length : ℝ) (central_angle : ℝ)
  (h_arc : arc_length = 4 * Real.pi)
  (h_angle : central_angle = Real.pi / 3)
  (h_radius : arc_length = central_angle * R) :
  area_of_sector arc_length central_angle 12 = 24 * Real.pi :=
by
  -- Proof skipped
  sorry

#check sector_area

end NUMINAMATH_GPT_sector_area_l1933_193365


namespace NUMINAMATH_GPT_john_task_completion_l1933_193362

theorem john_task_completion (J : ℝ) (h : 5 * (1 / J + 1 / 10) + 5 * (1 / J) = 1) : J = 20 :=
by
  sorry

end NUMINAMATH_GPT_john_task_completion_l1933_193362


namespace NUMINAMATH_GPT_saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l1933_193304

noncomputable def bread_saving (n_days : ℕ) : ℕ :=
  (1 / 2) * n_days

theorem saving_20_days :
  bread_saving 20 = 10 :=
by
  -- proof steps for bread_saving 20 = 10
  sorry

theorem cost_saving_20_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 20 * cost_per_loaf) = 350 :=
by
  -- proof steps for cost_saving_20_days
  sorry

theorem saving_60_days :
  bread_saving 60 = 30 :=
by
  -- proof steps for bread_saving 60 = 30
  sorry

theorem cost_saving_60_days (cost_per_loaf : ℕ) :
  cost_per_loaf = 35 → (bread_saving 60 * cost_per_loaf) = 1050 :=
by
  -- proof steps for cost_saving_60_days
  sorry

end NUMINAMATH_GPT_saving_20_days_cost_saving_20_days_saving_60_days_cost_saving_60_days_l1933_193304


namespace NUMINAMATH_GPT_time_to_cross_signal_pole_l1933_193385

/-- Definitions representing the given conditions --/
def length_of_train : ℕ := 300
def time_to_cross_platform : ℕ := 39
def length_of_platform : ℕ := 350
def total_distance := length_of_train + length_of_platform
def speed_of_train := total_distance / time_to_cross_platform

/-- Main statement to be proven --/
theorem time_to_cross_signal_pole : length_of_train / speed_of_train = 18 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_signal_pole_l1933_193385


namespace NUMINAMATH_GPT_min_f_triangle_sides_l1933_193311

noncomputable def f (x : ℝ) : ℝ :=
  let a := (2 * Real.cos x ^ 2, Real.sqrt 3)
  let b := (1, Real.sin (2 * x))
  (a.1 * b.1 + a.2 * b.2) - 2

theorem min_f (x : ℝ) (h1 : -Real.pi / 6 ≤ x) (h2 : x ≤ Real.pi / 3) :
  ∃ x₀, f x₀ = -2 ∧ ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → f x ≥ -2 :=
  sorry

theorem triangle_sides (a b C : ℝ) (h1 : f C = 1) (h2 : C = Real.pi / 6)
  (h3 : 1 = 1) (h4 : a * b = 2 * Real.sqrt 3) (h5 : a > b) :
  a = 2 ∧ b = Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_min_f_triangle_sides_l1933_193311


namespace NUMINAMATH_GPT_main_line_train_probability_l1933_193320

noncomputable def probability_catching_main_line (start_main_line start_harbor_line : Nat) (frequency : Nat) : ℝ :=
  if start_main_line % frequency = 0 ∧ start_harbor_line % frequency = 2 then 1 / 2 else 0

theorem main_line_train_probability :
  probability_catching_main_line 0 2 10 = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_main_line_train_probability_l1933_193320


namespace NUMINAMATH_GPT_largest_remainder_a_correct_l1933_193333

def largest_remainder_a (n : ℕ) (h : n < 150) : ℕ :=
  (269 % n)

theorem largest_remainder_a_correct : ∃ n < 150, largest_remainder_a n sorry = 133 :=
  sorry

end NUMINAMATH_GPT_largest_remainder_a_correct_l1933_193333


namespace NUMINAMATH_GPT_largest_y_coordinate_l1933_193343

theorem largest_y_coordinate (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 :=
sorry

end NUMINAMATH_GPT_largest_y_coordinate_l1933_193343


namespace NUMINAMATH_GPT_pairs_of_different_positives_l1933_193323

def W (x : ℕ) : ℕ := x^4 - 3 * x^3 + 5 * x^2 - 9 * x

theorem pairs_of_different_positives (a b : ℕ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b)
  (hW : W a = W b) : (a, b) = (1, 2) ∨ (a, b) = (2, 1) := 
sorry

end NUMINAMATH_GPT_pairs_of_different_positives_l1933_193323


namespace NUMINAMATH_GPT_scientific_notation_of_361000000_l1933_193339

theorem scientific_notation_of_361000000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ abs a) ∧ (abs a < 10) ∧ (361000000 = a * 10^n) ∧ (a = 3.61) ∧ (n = 8) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_361000000_l1933_193339


namespace NUMINAMATH_GPT_find_relationship_l1933_193300

theorem find_relationship (n m : ℕ) (a : ℚ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_pos_m : 0 < m) :
  (n > m ↔ (1 / n < a)) → m = ⌊1 / a⌋ :=
sorry

end NUMINAMATH_GPT_find_relationship_l1933_193300


namespace NUMINAMATH_GPT_smallest_possible_n_l1933_193378

-- Definitions needed for the problem
variable (x n : ℕ) (hpos : 0 < x)
variable (m : ℕ) (hm : m = 72)

-- The conditions as already stated
def gcd_cond := Nat.gcd 72 n = x + 8
def lcm_cond := Nat.lcm 72 n = x * (x + 8)

-- The proof statement
theorem smallest_possible_n (h_gcd : gcd_cond x n) (h_lcm : lcm_cond x n) : n = 8 :=
by 
  -- Intuitively outline the proof
  sorry

end NUMINAMATH_GPT_smallest_possible_n_l1933_193378


namespace NUMINAMATH_GPT_solution_l1933_193368

theorem solution
  (a b c d : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hd : 0 < d)
  (H : (1 / a + 1 / b) * (1 / c + 1 / d) + 1 / (a * b) + 1 / (c * d) = 6 / Real.sqrt (a * b * c * d)) :
  (a^2 + a * c + c^2) / (b^2 - b * d + d^2) = 3 :=
sorry

end NUMINAMATH_GPT_solution_l1933_193368


namespace NUMINAMATH_GPT_perp_line_eq_l1933_193383

theorem perp_line_eq (m : ℝ) (L1 : ∀ (x y : ℝ), m * x - m^2 * y = 1) (P : ℝ × ℝ) (P_def : P = (2, 1)) :
  ∃ d : ℝ, (∀ (x y : ℝ), x + y = d) ∧ P.fst + P.snd = d :=
by
  sorry

end NUMINAMATH_GPT_perp_line_eq_l1933_193383


namespace NUMINAMATH_GPT_line_through_circle_center_l1933_193359

theorem line_through_circle_center (a : ℝ) :
  (∃ (x y : ℝ), 3 * x + y + a = 0 ∧ x^2 + y^2 + 2 * x - 4 * y = 0) ↔ (a = 1) :=
by
  sorry

end NUMINAMATH_GPT_line_through_circle_center_l1933_193359


namespace NUMINAMATH_GPT_a_in_M_sufficient_not_necessary_l1933_193351

-- Defining the sets M and N
def M := {x : ℝ | x^2 < 3 * x}
def N := {x : ℝ | abs (x - 1) < 2}

-- Stating that a ∈ M is a sufficient but not necessary condition for a ∈ N
theorem a_in_M_sufficient_not_necessary (a : ℝ) (h : a ∈ M) : a ∈ N :=
by sorry

end NUMINAMATH_GPT_a_in_M_sufficient_not_necessary_l1933_193351


namespace NUMINAMATH_GPT_room_length_l1933_193377

def area_four_walls (L: ℕ) (w: ℕ) (h: ℕ) : ℕ :=
  2 * (L * h) + 2 * (w * h)

def area_door (d_w: ℕ) (d_h: ℕ) : ℕ :=
  d_w * d_h

def area_windows (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  num_windows * (win_w * win_h)

def total_area_to_whitewash (L: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) : ℕ :=
  area_four_walls L w h - area_door d_w d_h - area_windows win_w win_h num_windows

theorem room_length (cost: ℕ) (rate: ℕ) (w: ℕ) (h: ℕ) (d_w: ℕ) (d_h: ℕ) (win_w: ℕ) (win_h: ℕ) (num_windows: ℕ) (L: ℕ) :
  cost = rate * total_area_to_whitewash L w h d_w d_h win_w win_h num_windows →
  L = 25 :=
by
  have h1 : total_area_to_whitewash 25 15 12 6 3 4 3 3 = 24 * 25 + 306 := sorry
  have h2 : rate * (24 * 25 + 306) = 5436 := sorry
  sorry

end NUMINAMATH_GPT_room_length_l1933_193377


namespace NUMINAMATH_GPT_final_answer_l1933_193369

def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem final_answer : f 0 = 0 ∧ f 1 = 0 ∧ (∀ x : ℝ, f (-x) = f x) ∧ ¬ (∃ ε > 0, ∀ h : ℝ, abs h < ε → f h ≥ f 0) := 
by
  -- omit the proof steps that were provided in the solution
  sorry

end NUMINAMATH_GPT_final_answer_l1933_193369


namespace NUMINAMATH_GPT_oranges_apples_ratio_l1933_193315

variable (A O P : ℕ)
variable (n : ℚ)
variable (h1 : O = n * A)
variable (h2 : P = 4 * O)
variable (h3 : A = (0.08333333333333333 : ℚ) * P)

theorem oranges_apples_ratio (A O P : ℕ) (n : ℚ) 
  (h1 : O = n * A) (h2 : P = 4 * O) (h3 : A = (0.08333333333333333 : ℚ) * P) : n = 3 := 
by
  sorry

end NUMINAMATH_GPT_oranges_apples_ratio_l1933_193315


namespace NUMINAMATH_GPT_find_number_l1933_193367

theorem find_number (x : ℝ) : ((1.5 * x) / 7 = 271.07142857142856) → x = 1265 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1933_193367


namespace NUMINAMATH_GPT_decimal_to_vulgar_fraction_l1933_193392

theorem decimal_to_vulgar_fraction (h : (34 / 100 : ℚ) = 0.34) : (0.34 : ℚ) = 17 / 50 := by
  sorry

end NUMINAMATH_GPT_decimal_to_vulgar_fraction_l1933_193392


namespace NUMINAMATH_GPT_positive_integers_expressible_l1933_193335

theorem positive_integers_expressible :
  ∃ (x y : ℕ), (x > 0) ∧ (y > 0) ∧ (x^2 + y) / (x * y + 1) = 1 ∧
  ∃ (x' y' : ℕ), (x' > 0) ∧ (y' > 0) ∧ (x' ≠ x ∨ y' ≠ y) ∧ (x'^2 + y') / (x' * y' + 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_positive_integers_expressible_l1933_193335


namespace NUMINAMATH_GPT_compare_M_N_l1933_193330

theorem compare_M_N (a : ℝ) : 
  let M := 2 * a * (a - 2) + 7
  let N := (a - 2) * (a - 3)
  M > N :=
by
  sorry

end NUMINAMATH_GPT_compare_M_N_l1933_193330


namespace NUMINAMATH_GPT_S_2012_value_l1933_193325

-- Define the first term of the arithmetic sequence
def a1 : ℤ := -2012

-- Define the common difference
def d : ℤ := 2

-- Define the sequence a_n
def a (n : ℕ) : ℤ := a1 + d * (n - 1)

-- Define the sum of the first n terms S_n
def S (n : ℕ) : ℤ := n * (a1 + a n) / 2

-- Formalize the given problem as a Lean statement
theorem S_2012_value : S 2012 = -2012 :=
by 
{
  -- The proof is omitted as requested
  sorry
}

end NUMINAMATH_GPT_S_2012_value_l1933_193325


namespace NUMINAMATH_GPT_solve_for_r_l1933_193338

theorem solve_for_r (r : ℚ) (h : (r + 4) / (r - 3) = (r - 2) / (r + 2)) : r = -2/11 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_r_l1933_193338


namespace NUMINAMATH_GPT_teresa_age_when_michiko_born_l1933_193354

theorem teresa_age_when_michiko_born (teresa_current_age morio_current_age morio_age_when_michiko_born : ℕ) 
  (h1 : teresa_current_age = 59) 
  (h2 : morio_current_age = 71) 
  (h3 : morio_age_when_michiko_born = 38) : 
  teresa_current_age - (morio_current_age - morio_age_when_michiko_born) = 26 := 
by 
  sorry

end NUMINAMATH_GPT_teresa_age_when_michiko_born_l1933_193354


namespace NUMINAMATH_GPT_quadratic_roots_identity_l1933_193395

theorem quadratic_roots_identity:
  ∀ {x₁ x₂ : ℝ}, (x₁^2 - 2 * x₁ - 8 = 0) ∧ (x₂^2 - 2 * x₂ - 8 = 0) → (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by
  intros x₁ x₂ h
  sorry

end NUMINAMATH_GPT_quadratic_roots_identity_l1933_193395


namespace NUMINAMATH_GPT_intersection_P_Q_l1933_193326

open Set

noncomputable def P : Set ℝ := {1, 2, 3, 4}

noncomputable def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {1, 2} := 
by {
  sorry
}

end NUMINAMATH_GPT_intersection_P_Q_l1933_193326


namespace NUMINAMATH_GPT_original_salary_condition_l1933_193366

variable (S: ℝ)

theorem original_salary_condition (h: 1.10 * 1.08 * 0.95 * 0.93 * S = 6270) :
  S = 6270 / (1.10 * 1.08 * 0.95 * 0.93) :=
by
  sorry

end NUMINAMATH_GPT_original_salary_condition_l1933_193366


namespace NUMINAMATH_GPT_expand_product_l1933_193334

theorem expand_product (x : ℝ) : (x + 3) * (x + 6) = x^2 + 9 * x + 18 := 
by sorry

end NUMINAMATH_GPT_expand_product_l1933_193334


namespace NUMINAMATH_GPT_minValue_at_least_9_minValue_is_9_l1933_193384

noncomputable def minValue (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) : ℝ :=
  1 / a + 4 / b + 9 / c

theorem minValue_at_least_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4) :
  minValue a b c h_pos h_sum ≥ 9 :=
by
  sorry

theorem minValue_is_9 (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 4)
  (h_abc : a = 2/3 ∧ b = 4/3 ∧ c = 2) : minValue a b c h_pos h_sum = 9 :=
by
  sorry

end NUMINAMATH_GPT_minValue_at_least_9_minValue_is_9_l1933_193384


namespace NUMINAMATH_GPT_simplify_expression_l1933_193374

theorem simplify_expression (x y : ℝ) (hx : x = 5) (hy : y = 2) :
  (10 * x * y^3) / (15 * x^2 * y^2) = 4 / 15 :=
by
  rw [hx, hy]
  -- here we would simplify but leave a hole
  sorry

end NUMINAMATH_GPT_simplify_expression_l1933_193374


namespace NUMINAMATH_GPT_total_points_scored_l1933_193319

theorem total_points_scored :
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  a + b + c + d + e + f + g + h = 54 :=
by
  let a := 7
  let b := 8
  let c := 2
  let d := 11
  let e := 6
  let f := 12
  let g := 1
  let h := 7
  sorry

end NUMINAMATH_GPT_total_points_scored_l1933_193319


namespace NUMINAMATH_GPT_box_with_20_aluminium_80_plastic_weighs_494_l1933_193331

def weight_of_box_with_100_aluminium_balls := 510 -- in grams
def weight_of_box_with_100_plastic_balls := 490 -- in grams
def number_of_aluminium_balls := 100
def number_of_plastic_balls := 100

-- Define the weights per ball type by subtracting the weight of the box
def weight_per_aluminium_ball := (weight_of_box_with_100_aluminium_balls - weight_of_box_with_100_plastic_balls) / number_of_aluminium_balls
def weight_per_plastic_ball := (weight_of_box_with_100_plastic_balls - weight_of_box_with_100_plastic_balls) / number_of_plastic_balls

-- Condition: The weight of the box alone (since it's present in both conditions)
def weight_of_empty_box := weight_of_box_with_100_plastic_balls - (weight_per_plastic_ball * number_of_plastic_balls)

-- Function to compute weight of the box with given number of aluminium and plastic balls
def total_weight (num_al : ℕ) (num_pl : ℕ) : ℕ :=
  weight_of_empty_box + (weight_per_aluminium_ball * num_al) + (weight_per_plastic_ball * num_pl)

-- The theorem to be proven
theorem box_with_20_aluminium_80_plastic_weighs_494 :
  total_weight 20 80 = 494 := sorry

end NUMINAMATH_GPT_box_with_20_aluminium_80_plastic_weighs_494_l1933_193331


namespace NUMINAMATH_GPT_range_of_m_l1933_193305

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x >= (4 + m)) ∧ (x <= 3 * (x - 2) + 4) → (x ≥ 2)) →
  (-3 < m ∧ m <= -2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1933_193305


namespace NUMINAMATH_GPT_range_of_a_l1933_193398

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then x + a / x + 7 else x + a / x - 7

theorem range_of_a (a : ℝ) (ha : 0 < a)
  (hodd : ∀ x : ℝ, f (-x) a = -f x a)
  (hcond : ∀ x : ℝ, 0 ≤ x → f x a ≥ 1 - a) :
  4 ≤ a := sorry

end NUMINAMATH_GPT_range_of_a_l1933_193398


namespace NUMINAMATH_GPT_polygon_side_count_eq_six_l1933_193379

theorem polygon_side_count_eq_six (n : ℕ) 
  (h1 : (n - 2) * 180 = 2 * 360) : n = 6 :=
sorry

end NUMINAMATH_GPT_polygon_side_count_eq_six_l1933_193379


namespace NUMINAMATH_GPT_q_at_2_equals_9_l1933_193344

-- Define the sign function
noncomputable def sgn (x : ℝ) : ℝ :=
if x < 0 then -1 else if x = 0 then 0 else 1

-- Define the function q(x)
noncomputable def q (x : ℝ) : ℝ :=
sgn (3 * x - 1) * |3 * x - 1| ^ (1/2) +
3 * sgn (3 * x - 1) * |3 * x - 1| ^ (1/3) +
|3 * x - 1| ^ (1/4)

-- The theorem stating that q(2) equals 9
theorem q_at_2_equals_9 : q 2 = 9 :=
by sorry

end NUMINAMATH_GPT_q_at_2_equals_9_l1933_193344


namespace NUMINAMATH_GPT_prod_gcd_lcm_eq_864_l1933_193386

theorem prod_gcd_lcm_eq_864 : 
  let a := 24
  let b := 36
  let gcd_ab := Nat.gcd a b
  let lcm_ab := Nat.lcm a b
  gcd_ab * lcm_ab = 864 :=
by
  sorry

end NUMINAMATH_GPT_prod_gcd_lcm_eq_864_l1933_193386


namespace NUMINAMATH_GPT_quadratic_minimum_l1933_193382

-- Define the constants p and q as positive real numbers
variables (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

-- Define the quadratic function f
def f (x : ℝ) : ℝ := 3 * x^2 + p * x + q

-- Assertion to prove: the function f reaches its minimum at x = -p / 6
theorem quadratic_minimum : 
  ∃ x : ℝ, x = -p / 6 ∧ (∀ y : ℝ, f y ≥ f x) :=
sorry

end NUMINAMATH_GPT_quadratic_minimum_l1933_193382


namespace NUMINAMATH_GPT_original_number_is_120_l1933_193357

theorem original_number_is_120 (N k : ℤ) (hk : N - 33 = 87 * k) : N = 120 :=
by
  have h : N - 33 = 87 * 1 := by sorry
  have N_eq : N = 87 + 33 := by sorry
  have N_val : N = 120 := by sorry
  exact N_val

end NUMINAMATH_GPT_original_number_is_120_l1933_193357


namespace NUMINAMATH_GPT_find_x_minus_y_l1933_193303

theorem find_x_minus_y (x y : ℤ) (h1 : |x| = 5) (h2 : y^2 = 16) (h3 : x + y > 0) : x - y = 1 ∨ x - y = 9 := 
by sorry

end NUMINAMATH_GPT_find_x_minus_y_l1933_193303


namespace NUMINAMATH_GPT_nicole_answers_correctly_l1933_193301

theorem nicole_answers_correctly :
  ∀ (C K N : ℕ), C = 17 → K = C + 8 → N = K - 3 → N = 22 :=
by
  intros C K N hC hK hN
  sorry

end NUMINAMATH_GPT_nicole_answers_correctly_l1933_193301


namespace NUMINAMATH_GPT_milk_amount_at_beginning_l1933_193388

theorem milk_amount_at_beginning (H: 0.69 = 0.6 * total_milk) : total_milk = 1.15 :=
sorry

end NUMINAMATH_GPT_milk_amount_at_beginning_l1933_193388


namespace NUMINAMATH_GPT_min_value_of_squares_l1933_193340

theorem min_value_of_squares (a b c : ℝ) (h : a^3 + b^3 + c^3 - 3 * a * b * c = 8) : 
  ∃ m, m ≥ 4 ∧ ∀ a b c, a^3 + b^3 + c^3 - 3 * a * b * c = 8 → a^2 + b^2 + c^2 ≥ m :=
sorry

end NUMINAMATH_GPT_min_value_of_squares_l1933_193340


namespace NUMINAMATH_GPT_pool_capacity_l1933_193391

-- Conditions
variables (C : ℝ) -- total capacity of the pool in gallons
variables (h1 : 300 = 0.75 * C - 0.45 * C) -- the pool requires an additional 300 gallons to be filled to 75%
variables (h2 : 300 = 0.30 * C) -- pumping in these additional 300 gallons will increase the amount of water by 30%

-- Goal
theorem pool_capacity : C = 1000 :=
by sorry

end NUMINAMATH_GPT_pool_capacity_l1933_193391


namespace NUMINAMATH_GPT_least_possible_value_of_y_l1933_193370

theorem least_possible_value_of_y (x y z : ℤ) (hx : Even x) (hy : Odd y) (hz : Odd z) 
  (h1 : y - x > 5) (h2 : z - x ≥ 9) : y ≥ 7 :=
by {
  -- sorry allows us to skip the proof
  sorry
}

end NUMINAMATH_GPT_least_possible_value_of_y_l1933_193370


namespace NUMINAMATH_GPT_olive_charged_10_hours_l1933_193308

/-- If Olive charges her phone for 3/5 of the time she charged last night, and that results
    in 12 hours of use, where each hour of charge results in 2 hours of phone usage,
    then the time Olive charged her phone last night was 10 hours. -/
theorem olive_charged_10_hours (x : ℝ) 
  (h1 : 2 * (3 / 5) * x = 12) : 
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_olive_charged_10_hours_l1933_193308


namespace NUMINAMATH_GPT_cost_of_single_room_l1933_193349

theorem cost_of_single_room
  (total_rooms : ℕ)
  (double_rooms : ℕ)
  (cost_double_room : ℕ)
  (revenue_total : ℕ)
  (cost_single_room : ℕ)
  (H1 : total_rooms = 260)
  (H2 : double_rooms = 196)
  (H3 : cost_double_room = 60)
  (H4 : revenue_total = 14000)
  (H5 : revenue_total = (total_rooms - double_rooms) * cost_single_room + double_rooms * cost_double_room)
  : cost_single_room = 35 :=
sorry

end NUMINAMATH_GPT_cost_of_single_room_l1933_193349


namespace NUMINAMATH_GPT_solve_for_x_l1933_193371

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.05) : x = 0.18 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1933_193371


namespace NUMINAMATH_GPT_pure_imaginary_solution_l1933_193364

theorem pure_imaginary_solution (m : ℝ) 
  (h : ∃ m : ℝ, (m^2 + m - 2 = 0) ∧ (m^2 - 1 ≠ 0)) : m = -2 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_solution_l1933_193364


namespace NUMINAMATH_GPT_ways_to_draw_balls_eq_total_ways_l1933_193376

noncomputable def ways_to_draw_balls (n : Nat) :=
  if h : n = 15 then (15 * 14 * 13 * 12) else 0

noncomputable def valid_combinations : Nat := sorry

noncomputable def total_ways_to_draw : Nat :=
  valid_combinations * 24

theorem ways_to_draw_balls_eq_total_ways :
  ways_to_draw_balls 15 = total_ways_to_draw :=
sorry

end NUMINAMATH_GPT_ways_to_draw_balls_eq_total_ways_l1933_193376


namespace NUMINAMATH_GPT_xiao_ming_shopping_l1933_193381

theorem xiao_ming_shopping :
  ∃ x : ℕ, x ≤ 16 ∧ 6 * x ≤ 100 ∧ 100 - 6 * x = 28 :=
by
  -- Given that:
  -- 1. x is the same amount spent in each of the six stores.
  -- 2. Total money spent, 6 * x, must be less than or equal to 100.
  -- 3. We seek to prove that Xiao Ming has 28 yuan left.
  sorry

end NUMINAMATH_GPT_xiao_ming_shopping_l1933_193381


namespace NUMINAMATH_GPT_average_of_remaining_two_l1933_193313

theorem average_of_remaining_two
  (a b c d e f : ℝ) 
  (h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
  (h_avg_2_1 : (a + b) / 2 = 4.2)
  (h_avg_2_2 : (c + d) / 2 = 3.85) : 
  ((e + f) / 2) = 3.8 :=
by
  sorry

end NUMINAMATH_GPT_average_of_remaining_two_l1933_193313


namespace NUMINAMATH_GPT_solve_for_b_l1933_193302

theorem solve_for_b (a b c : ℝ) (cosC : ℝ) (h_a : a = 3) (h_c : c = 4) (h_cosC : cosC = -1/4) :
    c^2 = a^2 + b^2 - 2 * a * b * cosC → b = 7 / 2 :=
by 
  intro h_cosine_theorem
  sorry

end NUMINAMATH_GPT_solve_for_b_l1933_193302


namespace NUMINAMATH_GPT_exist_five_natural_numbers_sum_and_product_equal_ten_l1933_193341

theorem exist_five_natural_numbers_sum_and_product_equal_ten : 
  ∃ (n_1 n_2 n_3 n_4 n_5 : ℕ), 
  n_1 + n_2 + n_3 + n_4 + n_5 = 10 ∧ 
  n_1 * n_2 * n_3 * n_4 * n_5 = 10 := 
sorry

end NUMINAMATH_GPT_exist_five_natural_numbers_sum_and_product_equal_ten_l1933_193341


namespace NUMINAMATH_GPT_find_a_b_c_sum_l1933_193336

theorem find_a_b_c_sum (a b c : ℝ) 
  (h_vertex : ∀ x, y = a * x^2 + b * x + c ↔ y = a * (x - 3)^2 + 5)
  (h_passes : a * 1^2 + b * 1 + c = 2) :
  a + b + c = 35 / 4 :=
sorry

end NUMINAMATH_GPT_find_a_b_c_sum_l1933_193336


namespace NUMINAMATH_GPT_alice_winning_strategy_l1933_193372

theorem alice_winning_strategy (N : ℕ) (hN : N > 0) : 
  (∃! n : ℕ, N = n * n) ↔ (∀ (k : ℕ), ∃ (m : ℕ), m ≠ k ∧ (m ∣ k ∨ k ∣ m)) :=
sorry

end NUMINAMATH_GPT_alice_winning_strategy_l1933_193372


namespace NUMINAMATH_GPT_rect_side_ratio_square_l1933_193356

theorem rect_side_ratio_square (a b d : ℝ) (h1 : b = 2 * a) (h2 : d = a * Real.sqrt 5) : (b / a) ^ 2 = 4 := 
by sorry

end NUMINAMATH_GPT_rect_side_ratio_square_l1933_193356


namespace NUMINAMATH_GPT_cabbage_production_l1933_193353

theorem cabbage_production (x y : ℕ) 
  (h1 : y^2 - x^2 = 127) 
  (h2 : y - x = 1) 
  (h3 : 2 * y = 128) : y^2 = 4096 := by
  sorry

end NUMINAMATH_GPT_cabbage_production_l1933_193353


namespace NUMINAMATH_GPT_area_two_layers_l1933_193355

-- Given conditions
variables (A_total A_covered A_three_layers : ℕ)

-- Conditions from the problem
def condition_1 : Prop := A_total = 204
def condition_2 : Prop := A_covered = 140
def condition_3 : Prop := A_three_layers = 20

-- Mathematical equivalent proof problem
theorem area_two_layers (A_total A_covered A_three_layers : ℕ) 
  (h1 : condition_1 A_total) 
  (h2 : condition_2 A_covered) 
  (h3 : condition_3 A_three_layers) : 
  ∃ A_two_layers : ℕ, A_two_layers = 24 :=
by sorry

end NUMINAMATH_GPT_area_two_layers_l1933_193355


namespace NUMINAMATH_GPT_cover_large_square_l1933_193387

theorem cover_large_square :
  ∃ (small_squares : Fin 8 → Set (ℝ × ℝ)),
    (∀ i, small_squares i = {p : ℝ × ℝ | (p.1 - x_i)^2 + (p.2 - y_i)^2 < (3/2)^2}) ∧
    (∃ (large_square : Set (ℝ × ℝ)),
      large_square = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 7 ∧ 0 ≤ p.2 ∧ p.2 ≤ 7} ∧
      large_square ⊆ ⋃ i, small_squares i) :=
sorry

end NUMINAMATH_GPT_cover_large_square_l1933_193387


namespace NUMINAMATH_GPT_binom_15_4_l1933_193329

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  n.factorial / (k.factorial * (n - k).factorial)

-- Define the theorem stating the specific binomial coefficient value
theorem binom_15_4 : binom 15 4 = 1365 :=
by
  sorry

end NUMINAMATH_GPT_binom_15_4_l1933_193329


namespace NUMINAMATH_GPT_roots_of_cubic_l1933_193307

-- Define the cubic equation having roots 3 and -2
def cubic_eq (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The proof problem statement
theorem roots_of_cubic (a b c d : ℝ) (h₁ : a ≠ 0)
  (h₂ : cubic_eq a b c d 3)
  (h₃ : cubic_eq a b c d (-2)) : 
  (b + c) / a = -7 := 
sorry

end NUMINAMATH_GPT_roots_of_cubic_l1933_193307


namespace NUMINAMATH_GPT_perimeter_of_tangents_triangle_l1933_193348

theorem perimeter_of_tangents_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) :
    (4 * a * Real.sqrt (a * b)) / (a - b) = 4 * a * (Real.sqrt (a * b) / (a - b)) := 
sorry

end NUMINAMATH_GPT_perimeter_of_tangents_triangle_l1933_193348


namespace NUMINAMATH_GPT_lukas_points_in_5_games_l1933_193310

theorem lukas_points_in_5_games (avg_points_per_game : ℕ) (games_played : ℕ) (total_points : ℕ)
  (h_avg : avg_points_per_game = 12) (h_games : games_played = 5) : total_points = 60 :=
by
  sorry

end NUMINAMATH_GPT_lukas_points_in_5_games_l1933_193310


namespace NUMINAMATH_GPT_quadratic_sum_is_zero_l1933_193363

-- Definition of a quadratic function with given conditions and the final result to prove
theorem quadratic_sum_is_zero {a b c : ℝ} 
  (h₁ : ∀ x : ℝ, x = 3 → a * (x - 1) * (x - 5) = 36) 
  (h₂ : a * 1^2 + b * 1 + c = 0) 
  (h₃ : a * 5^2 + b * 5 + c = 0) : 
  a + b + c = 0 := 
sorry

end NUMINAMATH_GPT_quadratic_sum_is_zero_l1933_193363


namespace NUMINAMATH_GPT_complex_pow_i_2019_l1933_193389

theorem complex_pow_i_2019 : (Complex.I)^2019 = -Complex.I := 
by
  sorry

end NUMINAMATH_GPT_complex_pow_i_2019_l1933_193389


namespace NUMINAMATH_GPT_circles_externally_tangent_l1933_193317

theorem circles_externally_tangent
  (r1 r2 d : ℝ)
  (hr1 : r1 = 2) (hr2 : r2 = 3)
  (hd : d = 5) :
  r1 + r2 = d :=
by
  sorry

end NUMINAMATH_GPT_circles_externally_tangent_l1933_193317


namespace NUMINAMATH_GPT_quadratic_equation_roots_l1933_193361

theorem quadratic_equation_roots (a b c : ℝ) : 
  (b ^ 6 > 4 * (a ^ 3) * (c ^ 3)) → (b ^ 10 > 4 * (a ^ 5) * (c ^ 5)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_roots_l1933_193361


namespace NUMINAMATH_GPT_option_c_correct_l1933_193345

theorem option_c_correct (a b : ℝ) (h : a > b) : 2 + a > 2 + b :=
by sorry

end NUMINAMATH_GPT_option_c_correct_l1933_193345


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1933_193312

-- Define the solutions to the given quadratic equations

theorem solve_eq1 (x : ℝ) : 2 * x ^ 2 - 8 = 0 ↔ x = 2 ∨ x = -2 :=
by sorry

theorem solve_eq2 (x : ℝ) : x ^ 2 + 10 * x + 9 = 0 ↔ x = -9 ∨ x = -1 :=
by sorry

theorem solve_eq3 (x : ℝ) : 5 * x ^ 2 - 4 * x - 1 = 0 ↔ x = -1 / 5 ∨ x = 1 :=
by sorry

theorem solve_eq4 (x : ℝ) : x * (x - 2) + x - 2 = 0 ↔ x = 2 ∨ x = -1 :=
by sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_solve_eq3_solve_eq4_l1933_193312


namespace NUMINAMATH_GPT_geometric_sequence_second_term_value_l1933_193375

theorem geometric_sequence_second_term_value
  (a : ℝ) 
  (r : ℝ) 
  (h1 : 30 * r = a) 
  (h2 : a * r = 7 / 4) 
  (h3 : 0 < a) : 
  a = 7.5 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_second_term_value_l1933_193375


namespace NUMINAMATH_GPT_h_at_0_l1933_193360

noncomputable def h (x : ℝ) : ℝ := sorry -- the actual polynomial
-- Conditions for h(x)
axiom h_cond1 : h (-2) = -4
axiom h_cond2 : h (1) = -1
axiom h_cond3 : h (-3) = -9
axiom h_cond4 : h (3) = -9
axiom h_cond5 : h (5) = -25

-- Statement of the proof problem
theorem h_at_0 : h (0) = -90 := sorry

end NUMINAMATH_GPT_h_at_0_l1933_193360


namespace NUMINAMATH_GPT_find_d_l1933_193373

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
noncomputable def g (x : ℝ) (c : ℝ) : ℝ := c * x + 1

theorem find_d (c d : ℝ) (hx : ∀ x, f (g x c) c = 15 * x + d) : d = 8 :=
sorry

end NUMINAMATH_GPT_find_d_l1933_193373


namespace NUMINAMATH_GPT_next_meeting_time_at_B_l1933_193346

-- Definitions of conditions
def perimeter := 800 -- Perimeter of the block in meters
def t1 := 1 -- They meet for the first time after 1 minute
def AB := 100 -- Length of side AB in meters
def BC := 300 -- Length of side BC in meters
def CD := 100 -- Length of side CD in meters
def DA := 300 -- Length of side DA in meters

-- Main theorem statement
theorem next_meeting_time_at_B :
  ∃ t : ℕ, t = 9 ∧ (∃ m1 m2 : ℕ, ((t = m1 * m2 + 1) ∧ m2 = 800 / (t1 * (AB + BC + CD + DA))) ∧ m1 = 9) :=
sorry

end NUMINAMATH_GPT_next_meeting_time_at_B_l1933_193346


namespace NUMINAMATH_GPT_area_enclosed_by_graph_eq_2pi_l1933_193306

theorem area_enclosed_by_graph_eq_2pi :
  (∃ (x y : ℝ), x^2 + y^2 = 2 * |x| + 2 * |y| ) →
  ∀ (A : ℝ), A = 2 * Real.pi :=
sorry

end NUMINAMATH_GPT_area_enclosed_by_graph_eq_2pi_l1933_193306


namespace NUMINAMATH_GPT_cylinder_surface_area_l1933_193393

variable (height1 height2 radius1 radius2 : ℝ)
variable (π : ℝ)
variable (C1 : height1 = 6 * π)
variable (C2 : radius1 = 3)
variable (C3 : height2 = 4 * π)
variable (C4 : radius2 = 2)

theorem cylinder_surface_area : 
  (6 * π * 4 * π + 2 * π * radius1 ^ 2) = 24 * π ^ 2 + 18 * π ∨
  (4 * π * 6 * π + 2 * π * radius2 ^ 2) = 24 * π ^ 2 + 8 * π :=
by
  intros
  sorry

end NUMINAMATH_GPT_cylinder_surface_area_l1933_193393


namespace NUMINAMATH_GPT_ads_on_first_web_page_l1933_193394

theorem ads_on_first_web_page 
  (A : ℕ)
  (second_page_ads : ℕ := 2 * A)
  (third_page_ads : ℕ := 2 * A + 24)
  (fourth_page_ads : ℕ := 3 * A / 2)
  (total_ads : ℕ := 68 * 3 / 2)
  (sum_of_ads : A + 2 * A + (2 * A + 24) + 3 * A / 2 = total_ads) :
  A = 12 := 
by
  sorry

end NUMINAMATH_GPT_ads_on_first_web_page_l1933_193394


namespace NUMINAMATH_GPT_simplified_expression_l1933_193321

-- Non-computable context since we are dealing with square roots and division
noncomputable def expr (x : ℝ) : ℝ := ((x / (x - 1)) - 1) / ((x^2 + 2 * x + 1) / (x^2 - 1))

theorem simplified_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : expr x = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_simplified_expression_l1933_193321


namespace NUMINAMATH_GPT_solve_for_d_l1933_193332

theorem solve_for_d (r s t d c : ℝ)
  (h1 : (t = -r - s))
  (h2 : (c = rs + rt + st))
  (h3 : (t - 1 = -(r + 5) - (s - 4)))
  (h4 : (c = (r + 5) * (s - 4) + (r + 5) * (t - 1) + (s - 4) * (t - 1)))
  (h5 : (d = -r * s * t))
  (h6 : (d + 210 = -(r + 5) * (s - 4) * (t - 1))) :
  d = 240 ∨ d = 420 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_d_l1933_193332


namespace NUMINAMATH_GPT_speed_in_m_per_s_eq_l1933_193322

theorem speed_in_m_per_s_eq : (1 : ℝ) / 3.6 = (0.27777 : ℝ) :=
by sorry

end NUMINAMATH_GPT_speed_in_m_per_s_eq_l1933_193322


namespace NUMINAMATH_GPT_find_digit_B_l1933_193347

def six_digit_number (B : ℕ) : ℕ := 303200 + B

def is_prime_six_digit (B : ℕ) : Prop := Prime (six_digit_number B)

theorem find_digit_B :
  ∃ B : ℕ, (B ≤ 9) ∧ (is_prime_six_digit B) ∧ (B = 9) :=
sorry

end NUMINAMATH_GPT_find_digit_B_l1933_193347


namespace NUMINAMATH_GPT_minimum_oranges_to_profit_l1933_193342

/-- 
A boy buys 4 oranges for 12 cents and sells 6 oranges for 25 cents. 
Calculate the minimum number of oranges he needs to sell to make a profit of 150 cents.
--/
theorem minimum_oranges_to_profit (cost_oranges : ℕ) (cost_cents : ℕ)
  (sell_oranges : ℕ) (sell_cents : ℕ) (desired_profit : ℚ) :
  cost_oranges = 4 → cost_cents = 12 →
  sell_oranges = 6 → sell_cents = 25 →
  desired_profit = 150 →
  (∃ n : ℕ, n = 129) :=
by
  sorry

end NUMINAMATH_GPT_minimum_oranges_to_profit_l1933_193342


namespace NUMINAMATH_GPT_elements_of_set_A_l1933_193318

theorem elements_of_set_A (A : Set ℝ) (h₁ : ∀ a : ℝ, a ∈ A → (1 + a) / (1 - a) ∈ A)
(h₂ : -3 ∈ A) : A = {-3, -1/2, 1/3, 2} := by
  sorry

end NUMINAMATH_GPT_elements_of_set_A_l1933_193318


namespace NUMINAMATH_GPT_geometric_series_sum_l1933_193328

theorem geometric_series_sum : 
  let a := 6
  let r := - (2 / 5)
  let s := a / (1 - r)
  s = 30 / 7 :=
by
  let a := 6
  let r := -(2 / 5)
  let s := a / (1 - r)
  show s = 30 / 7
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1933_193328


namespace NUMINAMATH_GPT_tom_charges_per_lawn_l1933_193316

theorem tom_charges_per_lawn (gas_cost earnings_from_weeding total_profit lawns_mowed : ℕ) (charge_per_lawn : ℤ) 
  (h1 : gas_cost = 17)
  (h2 : earnings_from_weeding = 10)
  (h3 : total_profit = 29)
  (h4 : lawns_mowed = 3)
  (h5 : total_profit = ((lawns_mowed * charge_per_lawn) + earnings_from_weeding) - gas_cost) :
  charge_per_lawn = 12 := 
by
  sorry

end NUMINAMATH_GPT_tom_charges_per_lawn_l1933_193316


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1933_193399

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  ((∀ x : ℝ, (1 < x) → (x^2 - m * x + 1 > 0)) ↔ (-2 < m ∧ m < 2)) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1933_193399


namespace NUMINAMATH_GPT_problem_statement_l1933_193350

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement :
  (∀ x y : ℝ, f x + f y = f (x + y)) →
  f 3 = 4 →
  f 0 + f (-3) = -4 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_problem_statement_l1933_193350


namespace NUMINAMATH_GPT_q_sufficient_not_necessary_for_p_l1933_193397

def p (x : ℝ) : Prop := abs x < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

theorem q_sufficient_not_necessary_for_p (x : ℝ) : (q x → p x) ∧ ¬(p x → q x) := 
by
  sorry

end NUMINAMATH_GPT_q_sufficient_not_necessary_for_p_l1933_193397
