import Mathlib

namespace NUMINAMATH_GPT_litter_patrol_total_l1225_122554

theorem litter_patrol_total (glass_bottles : Nat) (aluminum_cans : Nat) 
  (h1 : glass_bottles = 10) (h2 : aluminum_cans = 8) : 
  glass_bottles + aluminum_cans = 18 :=
by
  sorry

end NUMINAMATH_GPT_litter_patrol_total_l1225_122554


namespace NUMINAMATH_GPT_car_travels_more_l1225_122557

theorem car_travels_more (train_speed : ℕ) (car_speed : ℕ) (time : ℕ)
  (h1 : train_speed = 60)
  (h2 : car_speed = 2 * train_speed)
  (h3 : time = 3) :
  car_speed * time - train_speed * time = 180 :=
by
  sorry

end NUMINAMATH_GPT_car_travels_more_l1225_122557


namespace NUMINAMATH_GPT_equilateral_triangle_l1225_122549

theorem equilateral_triangle
  (A B C : Type)
  (angle_A : ℝ)
  (side_BC : ℝ)
  (perimeter : ℝ)
  (h1 : angle_A = 60)
  (h2 : side_BC = 1/3 * perimeter)
  (side_AB : ℝ)
  (side_AC : ℝ)
  (h3 : perimeter = side_BC + side_AB + side_AC) :
  (side_AB = side_BC) ∧ (side_AC = side_BC) :=
by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_l1225_122549


namespace NUMINAMATH_GPT_solve_g_l1225_122527

def g (a b : ℚ) : ℚ :=
if a + b ≤ 4 then (a * b - 2 * a + 3) / (3 * a)
else (a * b - 3 * b - 1) / (-3 * b)

theorem solve_g :
  g 3 1 + g 1 5 = 11 / 15 :=
by
  -- Here we just set up the theorem statement. Proof is not included.
  sorry

end NUMINAMATH_GPT_solve_g_l1225_122527


namespace NUMINAMATH_GPT_melissa_total_score_l1225_122589

theorem melissa_total_score (games : ℕ) (points_per_game : ℕ) 
  (h_games : games = 3) (h_points_per_game : points_per_game = 27) : 
  points_per_game * games = 81 := 
by 
  sorry

end NUMINAMATH_GPT_melissa_total_score_l1225_122589


namespace NUMINAMATH_GPT_dayan_sequence_20th_term_l1225_122534

theorem dayan_sequence_20th_term (a : ℕ → ℕ) (h1 : a 0 = 0)
    (h2 : a 1 = 2) (h3 : a 2 = 4) (h4 : a 3 = 8) (h5 : a 4 = 12)
    (h6 : a 5 = 18) (h7 : a 6 = 24) (h8 : a 7 = 32) (h9 : a 8 = 40) (h10 : a 9 = 50)
    (h_even : ∀ n : ℕ, a (2 * n) = 2 * n^2) :
  a 20 = 200 :=
  sorry

end NUMINAMATH_GPT_dayan_sequence_20th_term_l1225_122534


namespace NUMINAMATH_GPT_find_h_l1225_122543

def infinite_sqrt_series (b : ℝ) : ℝ := sorry -- Placeholder for infinite series sqrt(b + sqrt(b + ...))

def diamond (a b : ℝ) : ℝ :=
  a^2 + infinite_sqrt_series b

theorem find_h (h : ℝ) : diamond 3 h = 12 → h = 6 :=
by
  intro h_condition
  -- Further steps will be used during proof
  sorry

end NUMINAMATH_GPT_find_h_l1225_122543


namespace NUMINAMATH_GPT_sum_of_ages_l1225_122514

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
def condition1 := a = 16 + b + c
def condition2 := a^2 = 1632 + (b + c)^2

-- Define the theorem to prove the question
theorem sum_of_ages : condition1 a b c → condition2 a b c → a + b + c = 102 := 
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sum_of_ages_l1225_122514


namespace NUMINAMATH_GPT_f_strictly_increasing_solve_inequality_l1225_122520

variable (f : ℝ → ℝ)

-- Conditions
axiom ax1 : ∀ a b : ℝ, f (a + b) = f a + f b - 1
axiom ax2 : ∀ x : ℝ, x > 0 → f x > 1
axiom ax3 : f 4 = 3

-- Prove monotonicity
theorem f_strictly_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by
  sorry

-- Solve the inequality
theorem solve_inequality (m : ℝ) : -2/3 < m ∧ m < 2 ↔ f (3 * m^2 - m - 2) < 2 := by
  sorry

end NUMINAMATH_GPT_f_strictly_increasing_solve_inequality_l1225_122520


namespace NUMINAMATH_GPT_minimum_value_inequality_l1225_122588

variable {x y z : ℝ}
variable (h1 : x > 0) (h2 : y > 0) (h3 : z > 0)

theorem minimum_value_inequality : (x + y + z) * (1 / (x + y) + 1 / (y + z) + 1 / (z + x)) ≥ 9 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l1225_122588


namespace NUMINAMATH_GPT_least_integer_x_l1225_122504

theorem least_integer_x (x : ℤ) (h : 3 * |x| - 2 * x + 8 < 23) : x = -3 :=
sorry

end NUMINAMATH_GPT_least_integer_x_l1225_122504


namespace NUMINAMATH_GPT_x_add_one_greater_than_x_l1225_122524

theorem x_add_one_greater_than_x (x : ℝ) : x + 1 > x :=
by
  sorry

end NUMINAMATH_GPT_x_add_one_greater_than_x_l1225_122524


namespace NUMINAMATH_GPT_inequality_of_reals_l1225_122580

theorem inequality_of_reals (a b c d : ℝ) : 
  (a + b + c + d) / ((1 + a^2) * (1 + b^2) * (1 + c^2) * (1 + d^2)) < 1 := 
  sorry

end NUMINAMATH_GPT_inequality_of_reals_l1225_122580


namespace NUMINAMATH_GPT_fraction_meaningful_range_l1225_122551

theorem fraction_meaningful_range (x : ℝ) : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_range_l1225_122551


namespace NUMINAMATH_GPT_range_of_m_l1225_122592

theorem range_of_m (m : ℝ) : 
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ (x^3 - 3 * x + m = 0)) → (m ≥ -2 ∧ m ≤ 2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l1225_122592


namespace NUMINAMATH_GPT_initial_guinea_fowls_l1225_122586

theorem initial_guinea_fowls (initial_chickens initial_turkeys : ℕ) 
  (initial_guinea_fowls : ℕ) (lost_chickens lost_turkeys lost_guinea_fowls : ℕ) 
  (total_birds_end : ℕ) (days : ℕ)
  (hc : initial_chickens = 300) (ht : initial_turkeys = 200) 
  (lc : lost_chickens = 20) (lt : lost_turkeys = 8) (lg : lost_guinea_fowls = 5) 
  (d : days = 7) (tb : total_birds_end = 349) :
  initial_guinea_fowls = 80 := 
by 
  sorry

end NUMINAMATH_GPT_initial_guinea_fowls_l1225_122586


namespace NUMINAMATH_GPT_find_base_b4_l1225_122538

theorem find_base_b4 (b_4 : ℕ) : (b_4 - 1) * (b_4 - 2) * (b_4 - 3) = 168 → b_4 = 8 :=
by
  intro h
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_base_b4_l1225_122538


namespace NUMINAMATH_GPT_power_modulo_l1225_122568

theorem power_modulo (k : ℕ) : 7^32 % 19 = 1 → 7^2050 % 19 = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_power_modulo_l1225_122568


namespace NUMINAMATH_GPT_total_items_at_bakery_l1225_122598

theorem total_items_at_bakery (bread_rolls : ℕ) (croissants : ℕ) (bagels : ℕ) (h1 : bread_rolls = 49) (h2 : croissants = 19) (h3 : bagels = 22) : bread_rolls + croissants + bagels = 90 :=
by
  sorry

end NUMINAMATH_GPT_total_items_at_bakery_l1225_122598


namespace NUMINAMATH_GPT_combined_time_l1225_122562

theorem combined_time {t_car t_train t_combined : ℝ} 
  (h1: t_car = 4.5) 
  (h2: t_train = t_car + 2) 
  (h3: t_combined = t_car + t_train) : 
  t_combined = 11 := by
  sorry

end NUMINAMATH_GPT_combined_time_l1225_122562


namespace NUMINAMATH_GPT_D_is_quadratic_l1225_122574

-- Define the equations
def eq_A (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0
def eq_B (x : ℝ) : Prop := 2 * x^2 - 3 * x = 2 * (x^2 - 2)
def eq_C (x : ℝ) : Prop := x^3 - 2 * x + 7 = 0
def eq_D (x : ℝ) : Prop := (x - 2)^2 - 4 = 0

-- Define what it means to be a quadratic equation
def is_quadratic (f : ℝ → Prop) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x ↔ a * x^2 + b * x + c = 0)

theorem D_is_quadratic : is_quadratic eq_D :=
sorry

end NUMINAMATH_GPT_D_is_quadratic_l1225_122574


namespace NUMINAMATH_GPT_ten_fact_minus_nine_fact_l1225_122595

-- Definitions corresponding to the conditions
def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Condition for 9!
def nine_factorial : ℕ := 362880

-- 10! can be expressed in terms of 9!
noncomputable def ten_factorial : ℕ := 10 * nine_factorial

-- Proof statement we need to show
theorem ten_fact_minus_nine_fact : ten_factorial - nine_factorial = 3265920 :=
by
  unfold ten_factorial
  unfold nine_factorial
  sorry

end NUMINAMATH_GPT_ten_fact_minus_nine_fact_l1225_122595


namespace NUMINAMATH_GPT_smaller_number_is_270_l1225_122548

theorem smaller_number_is_270 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 15) : S = 270 :=
sorry

end NUMINAMATH_GPT_smaller_number_is_270_l1225_122548


namespace NUMINAMATH_GPT_combined_total_value_of_items_l1225_122523

theorem combined_total_value_of_items :
  let V1 := 87.50 / 0.07
  let V2 := 144 / 0.12
  let V3 := 50 / 0.05
  let total1 := 1000 + V1
  let total2 := 1000 + V2
  let total3 := 1000 + V3
  total1 + total2 + total3 = 6450 := 
by
  sorry

end NUMINAMATH_GPT_combined_total_value_of_items_l1225_122523


namespace NUMINAMATH_GPT_sixth_graders_forgot_homework_percentage_l1225_122572

-- Definitions of the conditions
def num_students_A : ℕ := 20
def num_students_B : ℕ := 80
def percent_forgot_A : ℚ := 20 / 100
def percent_forgot_B : ℚ := 15 / 100

-- Statement to be proven
theorem sixth_graders_forgot_homework_percentage :
  (num_students_A * percent_forgot_A + num_students_B * percent_forgot_B) /
  (num_students_A + num_students_B) = 16 / 100 :=
by
  sorry

end NUMINAMATH_GPT_sixth_graders_forgot_homework_percentage_l1225_122572


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l1225_122502

-- Definitions for periodic decimals
def repeating_five := 5 / 9
def repeating_seven := 7 / 9

-- Theorem statement
theorem sum_of_repeating_decimals : (repeating_five + repeating_seven) = 4 / 3 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l1225_122502


namespace NUMINAMATH_GPT_minimum_value_of_expression_l1225_122539

noncomputable def min_value_expr (a b : ℝ) : ℝ :=
  (a + 1/b) * (a + 1/b - 2023) + (b + 1/a) * (b + 1/a - 2023)

theorem minimum_value_of_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  min_value_expr a b = -2031948.5 :=
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l1225_122539


namespace NUMINAMATH_GPT_smallest_positive_period_max_value_in_interval_min_value_in_interval_l1225_122583

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x + Real.pi / 6)

theorem smallest_positive_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem max_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = 2 := by
  sorry

theorem min_value_in_interval :
  ∃ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x = -1 := by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_max_value_in_interval_min_value_in_interval_l1225_122583


namespace NUMINAMATH_GPT_largest_element_sum_of_digits_in_E_l1225_122550
open BigOperators
open Nat

def E : Set ℕ := { n | ∃ (r₉ r₁₀ r₁₁ : ℕ), 0 < r₉ ∧ r₉ ≤ 9 ∧ 0 < r₁₀ ∧ r₁₀ ≤ 10 ∧ 0 < r₁₁ ∧ r₁₁ ≤ 11 ∧
  r₉ = n % 9 ∧ r₁₀ = n % 10 ∧ r₁₁ = n % 11 ∧
  (r₉ > 1) ∧ (r₁₀ > 1) ∧ (r₁₁ > 1) ∧
  ∃ (a : ℕ) (b : ℕ) (c : ℕ), r₉ = a ∧ r₁₀ = a * b ∧ r₁₁ = a * b * c ∧ b ≠ 1 ∧ c ≠ 1 }

noncomputable def N : ℕ := 
  max (max (74 % 990) (134 % 990)) (526 % 990)

def sum_of_digits (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem largest_element_sum_of_digits_in_E :
  sum_of_digits N = 13 :=
sorry

end NUMINAMATH_GPT_largest_element_sum_of_digits_in_E_l1225_122550


namespace NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1225_122596

theorem find_x_squared_plus_y_squared (x y : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y + x + y = 17) (h4 : x^2 * y + x * y^2 = 72) : x^2 + y^2 = 65 := 
  sorry

end NUMINAMATH_GPT_find_x_squared_plus_y_squared_l1225_122596


namespace NUMINAMATH_GPT_problem1_problem2_l1225_122559

-- Definitions of the sets A and B
def set_A (x : ℝ) : Prop := x ≤ -1 ∨ x ≥ 4
def set_B (x a : ℝ) : Prop := 2 * a ≤ x ∧ x ≤ a + 2

-- Problem 1: If A ∩ B ≠ ∅, find the range of a
theorem problem1 (a : ℝ) : (∃ x : ℝ, set_A x ∧ set_B x a) → a ≤ -1 / 2 ∨ a = 2 :=
sorry

-- Problem 2: If A ∩ B = B, find the value of a
theorem problem2 (a : ℝ) : (∀ x : ℝ, set_B x a → set_A x) → a ≤ -1 / 2 ∨ a ≥ 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1225_122559


namespace NUMINAMATH_GPT_actual_time_l1225_122515

def digit_in_range (a b : ℕ) : Prop := 
  (a = b + 1 ∨ a = b - 1)

def time_malfunctioned (h m : ℕ) : Prop :=
  digit_in_range 0 (h / 10) ∧ -- tens of hour digit (0 -> 1 or 9)
  digit_in_range 0 (h % 10) ∧ -- units of hour digit (0 -> 1 or 9)
  digit_in_range 5 (m / 10) ∧ -- tens of minute digit (5 -> 4 or 6)
  digit_in_range 9 (m % 10)   -- units of minute digit (9 -> 8 or 0)

theorem actual_time : ∃ h m : ℕ, time_malfunctioned h m ∧ h = 11 ∧ m = 48 :=
by
  sorry

end NUMINAMATH_GPT_actual_time_l1225_122515


namespace NUMINAMATH_GPT_predicted_holiday_shoppers_l1225_122545

-- Conditions
def packages_per_bulk_box : Nat := 25
def every_third_shopper_buys_package : Nat := 3
def bulk_boxes_ordered : Nat := 5

-- Number of predicted holiday shoppers
theorem predicted_holiday_shoppers (pbb : packages_per_bulk_box = 25)
                                   (etsbp : every_third_shopper_buys_package = 3)
                                   (bbo : bulk_boxes_ordered = 5) :
  (bulk_boxes_ordered * packages_per_bulk_box * every_third_shopper_buys_package) = 375 :=
by 
  -- Proof steps can be added here
  sorry

end NUMINAMATH_GPT_predicted_holiday_shoppers_l1225_122545


namespace NUMINAMATH_GPT_find_x_when_y_30_l1225_122518

variable (x y k : ℝ)

noncomputable def inversely_proportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, x * y = k

theorem find_x_when_y_30
  (h_inv_prop : inversely_proportional x y) 
  (h_known_values : x = 5 ∧ y = 15) :
  ∃ x : ℝ, (∃ y : ℝ, y = 30) ∧ x = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_find_x_when_y_30_l1225_122518


namespace NUMINAMATH_GPT_ratio_Ryn_Nikki_l1225_122570

def Joyce_movie_length (M : ℝ) : ℝ := M + 2
def Nikki_movie_length (M : ℝ) : ℝ := 3 * M
def Ryn_movie_fraction (F : ℝ) (Nikki_movie_length : ℝ) : ℝ := F * Nikki_movie_length

theorem ratio_Ryn_Nikki 
  (M : ℝ) 
  (Nikki_movie_is_30 : Nikki_movie_length M = 30) 
  (total_movie_hours_is_76 : M + Joyce_movie_length M + Nikki_movie_length M + Ryn_movie_fraction F (Nikki_movie_length M) = 76) 
  : F = 4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_Ryn_Nikki_l1225_122570


namespace NUMINAMATH_GPT_intersection_A_B_l1225_122508

-- define the set A
def A : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 3 * x - y = 7 }

-- define the set B
def B : Set (ℝ × ℝ) := { p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x + y = 3 }

-- Prove the intersection
theorem intersection_A_B :
  A ∩ B = { (2, -1) } :=
by
  -- We will insert the proof here
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1225_122508


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l1225_122540

theorem smallest_four_digit_divisible_by_53 : ∃ n : ℤ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧
  (∀ m : ℤ, 1000 ≤ m ∧ m < 10000 ∧ m % 53 = 0 → n ≤ m) :=
by
  use 1007
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_l1225_122540


namespace NUMINAMATH_GPT_order_of_abc_l1225_122532

noncomputable def a : ℝ := 0.1 * Real.exp 0.1
noncomputable def b : ℝ := 1 / 9
noncomputable def c : ℝ := -Real.log 0.9

theorem order_of_abc : b > a ∧ a > c :=
by
  sorry

end NUMINAMATH_GPT_order_of_abc_l1225_122532


namespace NUMINAMATH_GPT_walls_per_room_is_8_l1225_122530

-- Definitions and conditions
def total_rooms : Nat := 10
def green_rooms : Nat := 3 * total_rooms / 5
def purple_rooms : Nat := total_rooms - green_rooms
def purple_walls : Nat := 32
def walls_per_room : Nat := purple_walls / purple_rooms

-- Theorem to prove
theorem walls_per_room_is_8 : walls_per_room = 8 := by
  sorry

end NUMINAMATH_GPT_walls_per_room_is_8_l1225_122530


namespace NUMINAMATH_GPT_count_positive_bases_for_log_1024_l1225_122593

-- Define the conditions 
def is_positive_integer_log_base (b n : ℕ) : Prop := b^n = 1024 ∧ n > 0

-- State that there are exactly 4 positive integers b that satisfy the condition
theorem count_positive_bases_for_log_1024 :
  (∃ b1 b2 b3 b4 : ℕ, b1 ≠ b2 ∧ b1 ≠ b3 ∧ b1 ≠ b4 ∧ b2 ≠ b3 ∧ b2 ≠ b4 ∧ b3 ≠ b4 ∧
    (∀ b, is_positive_integer_log_base b 1 ∨ is_positive_integer_log_base b 2 ∨ is_positive_integer_log_base b 5 ∨ is_positive_integer_log_base b 10) ∧
    (is_positive_integer_log_base b1 1 ∨ is_positive_integer_log_base b1 2 ∨ is_positive_integer_log_base b1 5 ∨ is_positive_integer_log_base b1 10) ∧
    (is_positive_integer_log_base b2 1 ∨ is_positive_integer_log_base b2 2 ∨ is_positive_integer_log_base b2 5 ∨ is_positive_integer_log_base b2 10) ∧
    (is_positive_integer_log_base b3 1 ∨ is_positive_integer_log_base b3 2 ∨ is_positive_integer_log_base b3 5 ∨ is_positive_integer_log_base b3 10) ∧
    (is_positive_integer_log_base b4 1 ∨ is_positive_integer_log_base b4 2 ∨ is_positive_integer_log_base b4 5 ∨ is_positive_integer_log_base b4 10)) :=
sorry

end NUMINAMATH_GPT_count_positive_bases_for_log_1024_l1225_122593


namespace NUMINAMATH_GPT_fraction_question_l1225_122571

theorem fraction_question :
  ((3 / 8 + 5 / 6) / (5 / 12 + 1 / 4) = 29 / 16) :=
by
  -- This is where we will put the proof steps 
  sorry

end NUMINAMATH_GPT_fraction_question_l1225_122571


namespace NUMINAMATH_GPT_prime_factor_of_sum_of_four_consecutive_integers_l1225_122585

theorem prime_factor_of_sum_of_four_consecutive_integers (n : ℤ) : 
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) :=
by 
  sorry

end NUMINAMATH_GPT_prime_factor_of_sum_of_four_consecutive_integers_l1225_122585


namespace NUMINAMATH_GPT_females_with_advanced_degrees_l1225_122566

noncomputable def total_employees := 200
noncomputable def total_females := 120
noncomputable def total_advanced_degrees := 100
noncomputable def males_college_degree_only := 40

theorem females_with_advanced_degrees :
  (total_employees - total_females) - males_college_degree_only = 
  total_employees - total_females - males_college_degree_only ∧ 
  total_females = 120 ∧ 
  total_advanced_degrees = 100 ∧ 
  total_employees = 200 ∧ 
  males_college_degree_only = 40 ∧
  total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60 :=
sorry

end NUMINAMATH_GPT_females_with_advanced_degrees_l1225_122566


namespace NUMINAMATH_GPT_factor_expression_l1225_122544

theorem factor_expression (b : ℤ) : 
  (8 * b ^ 3 + 120 * b ^ 2 - 14) - (9 * b ^ 3 - 2 * b ^ 2 + 14) 
  = -1 * (b ^ 3 - 122 * b ^ 2 + 28) := 
by {
  sorry
}

end NUMINAMATH_GPT_factor_expression_l1225_122544


namespace NUMINAMATH_GPT_manuscript_pages_l1225_122564

theorem manuscript_pages (P : ℕ)
  (h1 : 30 = 30)
  (h2 : 20 = 20)
  (h3 : 50 = 30 + 20)
  (h4 : 710 = 5 * (P - 50) + 30 * 8 + 20 * 11) :
  P = 100 :=
by
  sorry

end NUMINAMATH_GPT_manuscript_pages_l1225_122564


namespace NUMINAMATH_GPT_condition_for_a_pow_zero_eq_one_l1225_122565

theorem condition_for_a_pow_zero_eq_one (a : Real) : a ≠ 0 ↔ a^0 = 1 :=
by
  sorry

end NUMINAMATH_GPT_condition_for_a_pow_zero_eq_one_l1225_122565


namespace NUMINAMATH_GPT_janet_lunch_cost_l1225_122584

theorem janet_lunch_cost 
  (num_children : ℕ) (num_chaperones : ℕ) (janet : ℕ) (extra_lunches : ℕ) (cost_per_lunch : ℕ) : 
  num_children = 35 → num_chaperones = 5 → janet = 1 → extra_lunches = 3 → cost_per_lunch = 7 → 
  cost_per_lunch * (num_children + num_chaperones + janet + extra_lunches) = 308 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_janet_lunch_cost_l1225_122584


namespace NUMINAMATH_GPT_enter_exit_ways_correct_l1225_122505

-- Defining the problem conditions
def num_entrances := 4

-- Defining the problem question and answer
def enter_exit_ways (n : Nat) : Nat := n * (n - 1)

-- Statement: Prove the number of different ways to enter and exit is 12
theorem enter_exit_ways_correct : enter_exit_ways num_entrances = 12 := by
  -- Proof
  sorry

end NUMINAMATH_GPT_enter_exit_ways_correct_l1225_122505


namespace NUMINAMATH_GPT_value_of_x_for_zero_expression_l1225_122522

theorem value_of_x_for_zero_expression (x : ℝ) (h : (x-5 = 0)) (h2 : (6*x - 12 ≠ 0)) :
  x = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_x_for_zero_expression_l1225_122522


namespace NUMINAMATH_GPT_train_pass_bridge_l1225_122500

-- Define variables and conditions
variables (train_length bridge_length : ℕ)
          (train_speed_kmph : ℕ)

-- Convert speed from km/h to m/s
def train_speed_mps(train_speed_kmph : ℕ) : ℚ :=
  (train_speed_kmph * 1000) / 3600

-- Total distance to cover
def total_distance(train_length bridge_length : ℕ) : ℕ :=
  train_length + bridge_length

-- Time to pass the bridge
def time_to_pass_bridge(train_length bridge_length : ℕ) (train_speed_kmph : ℕ) : ℚ :=
  (total_distance train_length bridge_length) / (train_speed_mps train_speed_kmph)

-- The proof statement
theorem train_pass_bridge :
  time_to_pass_bridge 360 140 50 = 36 := 
by
  -- actual proof would go here
  sorry

end NUMINAMATH_GPT_train_pass_bridge_l1225_122500


namespace NUMINAMATH_GPT_swimming_pool_radius_l1225_122579

theorem swimming_pool_radius 
  (r : ℝ)
  (h1 : ∀ (r : ℝ), r > 0)
  (h2 : π * (r + 4)^2 - π * r^2 = (11 / 25) * π * r^2) :
  r = 20 := 
sorry

end NUMINAMATH_GPT_swimming_pool_radius_l1225_122579


namespace NUMINAMATH_GPT_count_int_values_not_satisfying_ineq_l1225_122511

theorem count_int_values_not_satisfying_ineq :
  ∃ (s : Finset ℤ), (∀ x ∈ s, 3 * x^2 + 14 * x + 8 ≤ 17) ∧ (s.card = 10) :=
by
  sorry

end NUMINAMATH_GPT_count_int_values_not_satisfying_ineq_l1225_122511


namespace NUMINAMATH_GPT_apples_in_first_group_l1225_122541

variable (A O : ℝ) (X : ℕ)

-- Given conditions
axiom h1 : A = 0.21
axiom h2 : X * A + 3 * O = 1.77
axiom h3 : 2 * A + 5 * O = 1.27 

-- Goal: Prove that the number of apples in the first group is 6
theorem apples_in_first_group : X = 6 := 
by 
  sorry

end NUMINAMATH_GPT_apples_in_first_group_l1225_122541


namespace NUMINAMATH_GPT_number_of_digits_in_product_l1225_122542

open Nat

noncomputable def num_digits (n : ℕ) : ℕ :=
if n = 0 then 1 else Nat.log 10 n + 1

def compute_product : ℕ := 234567 * 123^3

theorem number_of_digits_in_product : num_digits compute_product = 13 := by 
  sorry

end NUMINAMATH_GPT_number_of_digits_in_product_l1225_122542


namespace NUMINAMATH_GPT_range_of_a_l1225_122507

open Set

variable (a : ℝ)

def P(a : ℝ) : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0

def Q(a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (a : ℝ) (hP : P a) (hQ : Q a) : a ≤ -2 ∨ a = 1 := sorry

end NUMINAMATH_GPT_range_of_a_l1225_122507


namespace NUMINAMATH_GPT_complement_union_l1225_122521

theorem complement_union (U A B : Set ℕ) (hU : U = {1, 2, 3, 4}) (hA : A = {1, 2}) (hB : B = {2, 3}) :
  U \ (A ∪ B) = {4} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l1225_122521


namespace NUMINAMATH_GPT_girls_maple_grove_correct_l1225_122582

variables (total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge : ℕ)
variables (girls_maple_grove : ℕ)

-- Conditions
def conditions : Prop :=
  total_students = 150 ∧ 
  boys = 82 ∧ 
  girls = 68 ∧ 
  pine_ridge_students = 70 ∧ 
  maple_grove_students = 80 ∧ 
  boys_pine_ridge = 36 ∧ 
  girls_maple_grove = girls - (pine_ridge_students - boys_pine_ridge)

-- Question and Answer translated to a proposition
def proof_problem : Prop :=
  conditions total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove → 
  girls_maple_grove = 34

-- Statement
theorem girls_maple_grove_correct : proof_problem total_students boys girls pine_ridge_students maple_grove_students boys_pine_ridge girls_maple_grove :=
by {
  sorry -- Proof omitted
}

end NUMINAMATH_GPT_girls_maple_grove_correct_l1225_122582


namespace NUMINAMATH_GPT_reduction_percentage_40_l1225_122575

theorem reduction_percentage_40 (P : ℝ) : 
  1500 * 1.20 - (P / 100 * (1500 * 1.20)) = 1080 ↔ P = 40 :=
by
  sorry

end NUMINAMATH_GPT_reduction_percentage_40_l1225_122575


namespace NUMINAMATH_GPT_second_quadrant_implies_value_of_m_l1225_122509

theorem second_quadrant_implies_value_of_m (m : ℝ) : 4 - m < 0 → m = 5 := by
  intro h
  have ineq : m > 4 := by
    linarith
  sorry

end NUMINAMATH_GPT_second_quadrant_implies_value_of_m_l1225_122509


namespace NUMINAMATH_GPT_find_p_l1225_122536

noncomputable def f (p : ℝ) : ℝ := 2 * p^2 + 20 * Real.sin p

theorem find_p : ∃ p : ℝ, f (f (f (f p))) = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_p_l1225_122536


namespace NUMINAMATH_GPT_star_computation_l1225_122591

def star (x y : ℝ) := x * y - 3 * x + y

theorem star_computation :
  (star 5 8) - (star 8 5) = 12 := by
  sorry

end NUMINAMATH_GPT_star_computation_l1225_122591


namespace NUMINAMATH_GPT_find_innings_l1225_122547

noncomputable def calculate_innings (A : ℕ) (n : ℕ) : Prop :=
  (n * A + 140 = (n + 1) * (A + 8)) ∧ (A + 8 = 28)

theorem find_innings (n : ℕ) (A : ℕ) :
  calculate_innings A n → n = 14 :=
by
  intros h
  -- Here you would prove that h implies n = 14, but we use sorry to skip the proof steps.
  sorry

end NUMINAMATH_GPT_find_innings_l1225_122547


namespace NUMINAMATH_GPT_inequality_m_le_minus3_l1225_122516

theorem inequality_m_le_minus3 (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1 → x^2 - 4 * x ≥ m) → m ≤ -3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_m_le_minus3_l1225_122516


namespace NUMINAMATH_GPT_lowest_score_85_avg_l1225_122533

theorem lowest_score_85_avg (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 = 79) (h2 : a2 = 88) (h3 : a3 = 94) 
  (h4 : a4 = 91) (h5 : 75 ≤ a5) (h6 : 75 ≤ a6) 
  (h7 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 85) : (a5 = 75 ∨ a6 = 75) ∧ (a5 = 75 ∨ a5 > 75) := 
by
  sorry

end NUMINAMATH_GPT_lowest_score_85_avg_l1225_122533


namespace NUMINAMATH_GPT_lines_parallel_distinct_l1225_122513

theorem lines_parallel_distinct (a : ℝ) : 
  (∀ x y : ℝ, (2 * x - a * y + 1 = 0) → ((a - 1) * x - y + a = 0)) ↔ 
  a = 2 := 
sorry

end NUMINAMATH_GPT_lines_parallel_distinct_l1225_122513


namespace NUMINAMATH_GPT_complex_fraction_value_l1225_122503

theorem complex_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 7 / 4 :=
sorry

end NUMINAMATH_GPT_complex_fraction_value_l1225_122503


namespace NUMINAMATH_GPT_tan_identity_equality_l1225_122517

theorem tan_identity_equality
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 :=
by
  sorry

end NUMINAMATH_GPT_tan_identity_equality_l1225_122517


namespace NUMINAMATH_GPT_shrink_ray_coffee_l1225_122512

theorem shrink_ray_coffee (num_cups : ℕ) (ounces_per_cup : ℕ) (shrink_factor : ℝ) 
  (h1 : num_cups = 5) 
  (h2 : ounces_per_cup = 8) 
  (h3 : shrink_factor = 0.5) 
  : num_cups * ounces_per_cup * shrink_factor = 20 :=
by
  rw [h1, h2, h3]
  simp
  norm_num

end NUMINAMATH_GPT_shrink_ray_coffee_l1225_122512


namespace NUMINAMATH_GPT_sum_arithmetic_sequence_l1225_122578

theorem sum_arithmetic_sequence {a : ℕ → ℤ} (S : ℕ → ℤ) :
  (∀ n, S n = n * (a 1 + a n) / 2) →
  S 13 = S 2000 →
  S 2013 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_arithmetic_sequence_l1225_122578


namespace NUMINAMATH_GPT_katie_pink_marbles_l1225_122576

-- Define variables for the problem
variables (P O R : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  O = P - 9 ∧
  R = 4 * (P - 9) ∧
  P + O + R = 33

-- Desired result
def result : Prop :=
  P = 13

-- Proof statement
theorem katie_pink_marbles : conditions P O R → result P :=
by
  intros h
  sorry

end NUMINAMATH_GPT_katie_pink_marbles_l1225_122576


namespace NUMINAMATH_GPT_distinct_positive_integers_count_l1225_122526

-- Define the digits' ranges
def digit (n : ℤ) : Prop := 0 ≤ n ∧ n ≤ 9
def nonzero_digit (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 9

-- Define the 4-digit numbers ABCD and DCBA
def ABCD (A B C D : ℤ) := 1000 * A + 100 * B + 10 * C + D
def DCBA (A B C D : ℤ) := 1000 * D + 100 * C + 10 * B + A

-- Define the difference
def difference (A B C D : ℤ) := ABCD A B C D - DCBA A B C D

-- The theorem to be proven
theorem distinct_positive_integers_count :
  ∃ n : ℤ, n = 161 ∧
  ∀ A B C D : ℤ,
  nonzero_digit A → nonzero_digit D → digit B → digit C → 
  0 < difference A B C D → (∃! x : ℤ, x = difference A B C D) :=
sorry

end NUMINAMATH_GPT_distinct_positive_integers_count_l1225_122526


namespace NUMINAMATH_GPT_developer_break_even_price_l1225_122567

theorem developer_break_even_price :
  let acres := 4
  let cost_per_acre := 1863
  let total_cost := acres * cost_per_acre
  let num_lots := 9
  let cost_per_lot := total_cost / num_lots
  cost_per_lot = 828 :=
by {
  sorry  -- This is where the proof would go.
} 

end NUMINAMATH_GPT_developer_break_even_price_l1225_122567


namespace NUMINAMATH_GPT_num_ways_to_factor_2210_l1225_122558

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_ways_to_factor_2210 : ∃! (a b : ℕ), a * b = 2210 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end NUMINAMATH_GPT_num_ways_to_factor_2210_l1225_122558


namespace NUMINAMATH_GPT_find_number_l1225_122501

theorem find_number (x : ℝ) (h : 0.50 * x = 0.30 * 50 + 13) : x = 56 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1225_122501


namespace NUMINAMATH_GPT_find_f_neg_one_l1225_122594

theorem find_f_neg_one (f : ℝ → ℝ) (h1 : ∀ x, f (-x) + x^2 = - (f x + x^2)) (h2 : f 1 = 1) : f (-1) = -3 := by
  sorry

end NUMINAMATH_GPT_find_f_neg_one_l1225_122594


namespace NUMINAMATH_GPT_all_points_below_line_l1225_122525

theorem all_points_below_line (a b : ℝ) (n : ℕ) (x y : ℕ → ℝ)
  (h1 : b > a)
  (h2 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → x k = a + ((k : ℝ) * (b - a) / (n + 1)))
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k = a * (b / a) ^ (k / (n + 1))) :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → y k < x k := 
sorry

end NUMINAMATH_GPT_all_points_below_line_l1225_122525


namespace NUMINAMATH_GPT_sum_first_n_abs_terms_arithmetic_seq_l1225_122555

noncomputable def sum_abs_arithmetic_sequence (n : ℕ) (h : n ≥ 3) : ℚ :=
  if n = 1 ∨ n = 2 then (n * (4 + 7 - 3 * n)) / 2
  else (3 * n^2 - 11 * n + 20) / 2

theorem sum_first_n_abs_terms_arithmetic_seq (n : ℕ) (h : n ≥ 3) :
  sum_abs_arithmetic_sequence n h = (3 * n^2) / 2 - (11 * n) / 2 + 10 :=
sorry

end NUMINAMATH_GPT_sum_first_n_abs_terms_arithmetic_seq_l1225_122555


namespace NUMINAMATH_GPT_number_of_dogs_l1225_122529

-- Define variables for the number of cats (C) and dogs (D)
variables (C D : ℕ)

-- Define the conditions from the problem statement
def condition1 : Prop := C = D - 6
def condition2 : Prop := C * 3 = D * 2

-- State the theorem that D should be 18 given the conditions
theorem number_of_dogs (h1 : condition1 C D) (h2 : condition2 C D) : D = 18 :=
  sorry

end NUMINAMATH_GPT_number_of_dogs_l1225_122529


namespace NUMINAMATH_GPT_tangent_x_axis_l1225_122556

noncomputable def curve (k : ℝ) : ℝ → ℝ := λ x => Real.log x - k * x + 3

theorem tangent_x_axis (k : ℝ) : 
  ∃ t : ℝ, curve k t = 0 ∧ deriv (curve k) t = 0 → k = Real.exp 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_x_axis_l1225_122556


namespace NUMINAMATH_GPT_total_wet_surface_area_l1225_122546

def length : ℝ := 8
def width : ℝ := 4
def depth : ℝ := 1.25

theorem total_wet_surface_area : length * width + 2 * (length * depth) + 2 * (width * depth) = 62 :=
by
  sorry

end NUMINAMATH_GPT_total_wet_surface_area_l1225_122546


namespace NUMINAMATH_GPT_product_of_integers_around_sqrt_50_l1225_122590

theorem product_of_integers_around_sqrt_50 :
  (∃ (x y : ℕ), x + 1 = y ∧ ↑x < Real.sqrt 50 ∧ Real.sqrt 50 < ↑y ∧ x * y = 56) :=
by
  sorry

end NUMINAMATH_GPT_product_of_integers_around_sqrt_50_l1225_122590


namespace NUMINAMATH_GPT_horner_rule_polynomial_polynomial_value_at_23_l1225_122519

def polynomial (x : ℤ) : ℤ := 7 * x ^ 3 + 3 * x ^ 2 - 5 * x + 11

def horner_polynomial (x : ℤ) : ℤ := x * ((7 * x + 3) * x - 5) + 11

theorem horner_rule_polynomial (x : ℤ) : polynomial x = horner_polynomial x :=
by 
  -- The proof steps would go here,
  -- demonstrating that polynomial x = horner_polynomial x.
  sorry

-- Instantiation of the theorem for a specific value of x
theorem polynomial_value_at_23 : polynomial 23 = horner_polynomial 23 :=
by 
  -- Using the previously established theorem
  apply horner_rule_polynomial

end NUMINAMATH_GPT_horner_rule_polynomial_polynomial_value_at_23_l1225_122519


namespace NUMINAMATH_GPT_diet_soda_bottles_l1225_122537

-- Define the conditions and then state the problem
theorem diet_soda_bottles (R D : ℕ) (h1 : R = 67) (h2 : R = D + 58) : D = 9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_diet_soda_bottles_l1225_122537


namespace NUMINAMATH_GPT_x_plus_y_eq_20_l1225_122531

theorem x_plus_y_eq_20 (x y : ℝ) (hxy : x ≠ y) (hdet : (Matrix.det ![
  ![2, 3, 7],
  ![4, x, y],
  ![4, y, x]]) = 0) : x + y = 20 :=
by
  sorry

end NUMINAMATH_GPT_x_plus_y_eq_20_l1225_122531


namespace NUMINAMATH_GPT_largest_interior_angle_l1225_122597

theorem largest_interior_angle (x : ℝ) (h_ratio : (5*x + 4*x + 3*x = 360)) :
  let e1 := 3 * x
  let e2 := 4 * x
  let e3 := 5 * x
  let i1 := 180 - e1
  let i2 := 180 - e2
  let i3 := 180 - e3
  max i1 (max i2 i3) = 90 :=
sorry

end NUMINAMATH_GPT_largest_interior_angle_l1225_122597


namespace NUMINAMATH_GPT_triangle_acute_angle_exists_l1225_122552

theorem triangle_acute_angle_exists (a b c d e : ℝ)
  (h_abc : a + b > c ∧ a + c > b ∧ b + c > a)
  (h_abd : a + b > d ∧ a + d > b ∧ b + d > a)
  (h_abe : a + b > e ∧ a + e > b ∧ b + e > a)
  (h_acd : a + c > d ∧ a + d > c ∧ c + d > a)
  (h_ace : a + c > e ∧ a + e > c ∧ c + e > a)
  (h_ade : a + d > e ∧ a + e > d ∧ d + e > a)
  (h_bcd : b + c > d ∧ b + d > c ∧ c + d > b)
  (h_bce : b + c > e ∧ b + e > c ∧ c + e > b)
  (h_bde : b + d > e ∧ b + e > d ∧ d + e > b)
  (h_cde : c + d > e ∧ c + e > d ∧ d + e > c) :
  ∃ x y z, (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
           (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
           (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧
           x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
           x + y > z ∧ x + z > y ∧ y + z > x ∧
           (x * x + y * y > z * z ∧ y * y + z * z > x * x ∧ z * z + x * x > y * y) := 
sorry

end NUMINAMATH_GPT_triangle_acute_angle_exists_l1225_122552


namespace NUMINAMATH_GPT_paving_stone_width_l1225_122569

theorem paving_stone_width
  (courtyard_length : ℝ) (courtyard_width : ℝ)
  (num_paving_stones : ℕ) (paving_stone_length : ℝ)
  (courtyard_area : ℝ) (paving_stone_area : ℝ)
  (width : ℝ)
  (h1 : courtyard_length = 30)
  (h2 : courtyard_width = 16.5)
  (h3 : num_paving_stones = 99)
  (h4 : paving_stone_length = 2.5)
  (h5 : courtyard_area = courtyard_length * courtyard_width)
  (h6 : courtyard_area = 495)
  (h7 : paving_stone_area = courtyard_area / num_paving_stones)
  (h8 : paving_stone_area = 5)
  (h9 : paving_stone_area = paving_stone_length * width) :
  width = 2 := by
  sorry

end NUMINAMATH_GPT_paving_stone_width_l1225_122569


namespace NUMINAMATH_GPT_statues_painted_l1225_122535

-- Definitions based on the conditions provided in the problem
def paint_remaining : ℚ := 1/2
def paint_per_statue : ℚ := 1/4

-- The theorem that answers the question
theorem statues_painted (h : paint_remaining = 1/2 ∧ paint_per_statue = 1/4) : 
  (paint_remaining / paint_per_statue) = 2 := 
sorry

end NUMINAMATH_GPT_statues_painted_l1225_122535


namespace NUMINAMATH_GPT_func_value_sum_l1225_122599

noncomputable def f (x : ℝ) : ℝ :=
  -x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2 + 1

theorem func_value_sum : f (1/2) + f (-1/2) = 2 :=
by
  sorry

end NUMINAMATH_GPT_func_value_sum_l1225_122599


namespace NUMINAMATH_GPT_reciprocal_of_neg_three_l1225_122510

theorem reciprocal_of_neg_three : (1:ℝ) / (-3:ℝ) = -1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_three_l1225_122510


namespace NUMINAMATH_GPT_square_window_side_length_is_24_l1225_122573

noncomputable def side_length_square_window
  (num_panes_per_row : ℕ) (pane_height_ratio : ℝ) (border_width : ℝ) (x : ℝ) : ℝ :=
  num_panes_per_row * x + (num_panes_per_row + 1) * border_width

theorem square_window_side_length_is_24
  (num_panes_per_row : ℕ)
  (pane_height_ratio : ℝ)
  (border_width : ℝ) 
  (pane_width : ℝ)
  (pane_height : ℝ)
  (window_side_length : ℝ) : 
  (num_panes_per_row = 3) →
  (pane_height_ratio = 3) →
  (border_width = 3) →
  (pane_height = pane_height_ratio * pane_width) →
  (window_side_length = side_length_square_window num_panes_per_row pane_height_ratio border_width pane_width) →
  (window_side_length = 24) :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_square_window_side_length_is_24_l1225_122573


namespace NUMINAMATH_GPT_fish_weight_l1225_122577

theorem fish_weight (W : ℝ) (h : W = 2 + W / 3) : W = 3 :=
by
  sorry

end NUMINAMATH_GPT_fish_weight_l1225_122577


namespace NUMINAMATH_GPT_antisymmetric_function_multiplication_cauchy_solution_l1225_122561

variable (f : ℤ → ℤ)
variable (h : ∀ x y : ℤ, f (x + y) = f x + f y)

theorem antisymmetric : ∀ x : ℤ, f (-x) = -f x := by
  sorry

theorem function_multiplication : ∀ x y : ℤ, f (x * y) = x * f y := by
  sorry

theorem cauchy_solution : ∃ c : ℤ, ∀ x : ℤ, f x = c * x := by
  sorry

end NUMINAMATH_GPT_antisymmetric_function_multiplication_cauchy_solution_l1225_122561


namespace NUMINAMATH_GPT_remainder_of_large_power_l1225_122587

theorem remainder_of_large_power :
  (2^(2^(2^2))) % 500 = 36 :=
sorry

end NUMINAMATH_GPT_remainder_of_large_power_l1225_122587


namespace NUMINAMATH_GPT_student_correct_answers_l1225_122560

noncomputable def correct_answers : ℕ := 58

theorem student_correct_answers (C W : ℕ) (h1 : C + W = 100) (h2 : 5 * C - 2 * W = 210) : C = correct_answers :=
by {
  -- placeholder for actual proof
  sorry
}

end NUMINAMATH_GPT_student_correct_answers_l1225_122560


namespace NUMINAMATH_GPT_problem1_range_problem2_range_l1225_122581

theorem problem1_range (x y : ℝ) (h : y = 2*|x-1| - |x-4|) : -3 ≤ y := sorry

theorem problem2_range (x a : ℝ) (h : ∀ x, 2*|x-1| - |x-a| ≥ -1) : 0 ≤ a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_problem1_range_problem2_range_l1225_122581


namespace NUMINAMATH_GPT_jerusha_and_lottie_earnings_l1225_122528

theorem jerusha_and_lottie_earnings :
  let J := 68
  let L := J / 4
  J + L = 85 := 
by
  sorry

end NUMINAMATH_GPT_jerusha_and_lottie_earnings_l1225_122528


namespace NUMINAMATH_GPT_part1_tangent_line_part2_monotonicity_l1225_122553

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * (x ^ 2 - 2 * a * x) * Real.log x - x ^ 2 + 4 * a * x + 1

theorem part1_tangent_line (a : ℝ) (h : a = 0) :
  let e := Real.exp 1
  let f_x := f e 0
  let tangent_line := 4 * e - 3 * e ^ 2 + 1
  tangent_line = 4 * e * (x - e) + f_x :=
sorry

theorem part2_monotonicity (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → f (x) a > 0 ↔ a ≤ 0) ∧
  (∀ x : ℝ, 0 < x → x < a → f (x) a > 0 ↔ 0 < a ∧ a < 1) ∧
  (∀ x : ℝ, 1 < x → x < a → f (x) a < 0 ↔ a > 1) ∧
  (∀ x : ℝ, 0 < x → 1 < x → x < a → f (x) a < 0 ↔ (a > 1)) ∧
  (∀ x : ℝ, x > 1 → f (x) a > 0 ↔ (a < 1)) :=
sorry

end NUMINAMATH_GPT_part1_tangent_line_part2_monotonicity_l1225_122553


namespace NUMINAMATH_GPT_radius_of_smaller_circle_l1225_122506

open Real

-- Definitions based on the problem conditions
def large_circle_radius : ℝ := 10
def pattern := "square"

-- Statement of the problem in Lean 4
theorem radius_of_smaller_circle :
  ∀ (r : ℝ), (large_circle_radius = 10) → (pattern = "square") → r = 5 * sqrt 2 →  ∃ r, r = 5 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_smaller_circle_l1225_122506


namespace NUMINAMATH_GPT_cars_cleaned_per_day_l1225_122563

theorem cars_cleaned_per_day
  (money_per_car : ℕ)
  (total_money : ℕ)
  (days : ℕ)
  (h1 : money_per_car = 5)
  (h2 : total_money = 2000)
  (h3 : days = 5) :
  (total_money / (money_per_car * days)) = 80 := by
  sorry

end NUMINAMATH_GPT_cars_cleaned_per_day_l1225_122563
