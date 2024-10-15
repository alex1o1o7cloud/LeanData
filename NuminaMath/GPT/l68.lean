import Mathlib

namespace NUMINAMATH_GPT_number_is_4_less_than_opposite_l68_6805

-- Define the number and its opposite relationship
def opposite_relation (x : ℤ) : Prop := x = -x + (-4)

-- Theorem stating that the given number is 4 less than its opposite
theorem number_is_4_less_than_opposite (x : ℤ) : opposite_relation x :=
sorry

end NUMINAMATH_GPT_number_is_4_less_than_opposite_l68_6805


namespace NUMINAMATH_GPT_evaluate_expression_l68_6878

-- Definition of variables a, b, c as given in conditions
def a : ℕ := 7
def b : ℕ := 11
def c : ℕ := 13

-- The theorem to prove the given expression equals 31
theorem evaluate_expression : 
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 31 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l68_6878


namespace NUMINAMATH_GPT_parallel_condition_l68_6833

def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem parallel_condition (x : ℝ) : 
  let a := (2, 1)
  let b := (3 * x ^ 2 - 1, x)
  (x = 1 → are_parallel a b) ∧ 
  ∃ x', x' ≠ 1 ∧ are_parallel a (3 * x' ^ 2 - 1, x') :=
by
  sorry

end NUMINAMATH_GPT_parallel_condition_l68_6833


namespace NUMINAMATH_GPT_alice_bob_speed_l68_6841

theorem alice_bob_speed (x : ℝ) (h : x = 3 + 2 * Real.sqrt 7) :
  x^2 - 5 * x - 14 = 8 + 2 * Real.sqrt 7 - 5 := by
sorry

end NUMINAMATH_GPT_alice_bob_speed_l68_6841


namespace NUMINAMATH_GPT_find_first_4_hours_speed_l68_6896

noncomputable def average_speed_first_4_hours
  (total_avg_speed : ℝ)
  (first_4_hours_avg_speed : ℝ)
  (remaining_hours_avg_speed : ℝ)
  (total_time : ℕ)
  (first_4_hours : ℕ)
  (remaining_hours : ℕ) : Prop :=
  total_avg_speed * total_time = first_4_hours_avg_speed * first_4_hours + remaining_hours * remaining_hours_avg_speed

theorem find_first_4_hours_speed :
  average_speed_first_4_hours 50 35 53 24 4 20 :=
by
  sorry

end NUMINAMATH_GPT_find_first_4_hours_speed_l68_6896


namespace NUMINAMATH_GPT_parabola_point_distance_eq_l68_6890

open Real

theorem parabola_point_distance_eq (P : ℝ × ℝ) (V : ℝ × ℝ) (F : ℝ × ℝ)
    (hV: V = (0, 0)) (hF : F = (0, 2)) (P_on_parabola : P.1 ^ 2 = 8 * P.2) 
    (hPf : dist P F = 150) (P_in_first_quadrant : 0 ≤ P.1 ∧ 0 ≤ P.2) :
    P = (sqrt 1184, 148) :=
sorry

end NUMINAMATH_GPT_parabola_point_distance_eq_l68_6890


namespace NUMINAMATH_GPT_paper_needed_l68_6844

theorem paper_needed : 26 + 26 + 10 = 62 := by
  sorry

end NUMINAMATH_GPT_paper_needed_l68_6844


namespace NUMINAMATH_GPT_correct_table_count_l68_6838

def stools_per_table : ℕ := 8
def chairs_per_table : ℕ := 2
def legs_per_stool : ℕ := 3
def legs_per_chair : ℕ := 4
def legs_per_table : ℕ := 4
def total_legs : ℕ := 656

theorem correct_table_count (t : ℕ) :
  stools_per_table * legs_per_stool * t +
  chairs_per_table * legs_per_chair * t +
  legs_per_table * t = total_legs → t = 18 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_correct_table_count_l68_6838


namespace NUMINAMATH_GPT_P_subset_Q_l68_6866

def P : Set ℝ := {m | -1 < m ∧ m < 0}

def Q : Set ℝ := {m | ∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end NUMINAMATH_GPT_P_subset_Q_l68_6866


namespace NUMINAMATH_GPT_least_integer_l68_6846

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end NUMINAMATH_GPT_least_integer_l68_6846


namespace NUMINAMATH_GPT_cone_height_correct_l68_6879

noncomputable def cone_height (radius : ℝ) (central_angle : ℝ) : ℝ := 
  let base_radius := (central_angle * radius) / (2 * Real.pi)
  let height := Real.sqrt (radius ^ 2 - base_radius ^ 2)
  height

theorem cone_height_correct:
  cone_height 3 (2 * Real.pi / 3) = 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_cone_height_correct_l68_6879


namespace NUMINAMATH_GPT_max_min_PA_l68_6836

open Classical

variables (A B P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace P]
          (dist_AB : ℝ) (dist_PA_PB : ℝ)

noncomputable def max_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry
noncomputable def min_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry

theorem max_min_PA (A B : Type) [MetricSpace A] [MetricSpace B] [Inhabited P]
                   (dist_AB : ℝ) (dist_PA_PB : ℝ) :
  dist_AB = 4 → dist_PA_PB = 6 → max_PA A B 4 = 5 ∧ min_PA A B 4 = 1 :=
by
  intros h_AB h_PA_PB
  sorry

end NUMINAMATH_GPT_max_min_PA_l68_6836


namespace NUMINAMATH_GPT_four_r_eq_sum_abcd_l68_6895

theorem four_r_eq_sum_abcd (a b c d r : ℤ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : (r - a) * (r - b) * (r - c) * (r - d) = 4) : 
  4 * r = a + b + c + d :=
by 
  sorry

end NUMINAMATH_GPT_four_r_eq_sum_abcd_l68_6895


namespace NUMINAMATH_GPT_luncheon_cost_l68_6845

theorem luncheon_cost (s c p : ℝ) (h1 : 5 * s + 9 * c + 2 * p = 5.95)
  (h2 : 7 * s + 12 * c + 2 * p = 7.90) (h3 : 3 * s + 5 * c + p = 3.50) :
  s + c + p = 1.05 :=
sorry

end NUMINAMATH_GPT_luncheon_cost_l68_6845


namespace NUMINAMATH_GPT_same_solution_m_l68_6863

theorem same_solution_m (m x : ℤ) : 
  (8 - m = 2 * (x + 1)) ∧ (2 * (2 * x - 3) - 1 = 1 - 2 * x) → m = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_same_solution_m_l68_6863


namespace NUMINAMATH_GPT_volume_of_pyramid_correct_l68_6867

noncomputable def volume_of_pyramid (lateral_surface_area base_area inscribed_circle_area radius : ℝ) : ℝ :=
  if lateral_surface_area = 3 * base_area ∧ inscribed_circle_area = radius then
    (2 * Real.sqrt 6) / (Real.pi ^ 3)
  else
    0

theorem volume_of_pyramid_correct
  (lateral_surface_area base_area inscribed_circle_area radius : ℝ)
  (h1 : lateral_surface_area = 3 * base_area)
  (h2 : inscribed_circle_area = radius) :
  volume_of_pyramid lateral_surface_area base_area inscribed_circle_area radius = (2 * Real.sqrt 6) / (Real.pi ^ 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_volume_of_pyramid_correct_l68_6867


namespace NUMINAMATH_GPT_sum_of_logs_l68_6862

open Real

noncomputable def log_base (b a : ℝ) : ℝ := log a / log b

theorem sum_of_logs (x y z : ℝ)
  (h1 : log_base 2 (log_base 4 (log_base 5 x)) = 0)
  (h2 : log_base 3 (log_base 5 (log_base 2 y)) = 0)
  (h3 : log_base 4 (log_base 2 (log_base 3 z)) = 0) :
  x + y + z = 666 := sorry

end NUMINAMATH_GPT_sum_of_logs_l68_6862


namespace NUMINAMATH_GPT_rectangle_perimeter_ratio_l68_6823

theorem rectangle_perimeter_ratio (side_length : ℝ) (h : side_length = 4) :
  let small_rectangle_perimeter := 2 * (side_length + (side_length / 4))
  let large_rectangle_perimeter := 2 * (side_length + (side_length / 2))
  small_rectangle_perimeter / large_rectangle_perimeter = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_ratio_l68_6823


namespace NUMINAMATH_GPT_houses_without_features_l68_6835

-- Definitions for the given conditions
def N : ℕ := 70
def G : ℕ := 50
def P : ℕ := 40
def GP : ℕ := 35

-- The statement of the proof problem
theorem houses_without_features : N - (G + P - GP) = 15 := by
  sorry

end NUMINAMATH_GPT_houses_without_features_l68_6835


namespace NUMINAMATH_GPT_find_x_value_l68_6853

theorem find_x_value (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_find_x_value_l68_6853


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l68_6887

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l68_6887


namespace NUMINAMATH_GPT_largest_number_of_gold_coins_l68_6877

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end NUMINAMATH_GPT_largest_number_of_gold_coins_l68_6877


namespace NUMINAMATH_GPT_rug_area_is_24_l68_6889

def length_floor : ℕ := 12
def width_floor : ℕ := 10
def strip_width : ℕ := 3

theorem rug_area_is_24 :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := 
by
  sorry

end NUMINAMATH_GPT_rug_area_is_24_l68_6889


namespace NUMINAMATH_GPT_factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l68_6880

-- Given condition and question, prove equality for the first expression
theorem factorize_x4_minus_16y4 (x y : ℝ) :
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by sorry

-- Given condition and question, prove equality for the second expression
theorem factorize_minus_2a3_plus_12a2_minus_16a (a : ℝ) :
  -2 * a^3 + 12 * a^2 - 16 * a = -2 * a * (a - 2) * (a - 4) := 
by sorry

end NUMINAMATH_GPT_factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l68_6880


namespace NUMINAMATH_GPT_required_circle_properties_l68_6826

-- Define the two given circles' equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of the required circle lies
def line (x y : ℝ) : Prop :=
  x - y - 4 = 0

-- The equation of the required circle
def required_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - x + 7*y - 32 = 0

-- Prove that the required circle satisfies the conditions
theorem required_circle_properties (x y : ℝ) (hx : required_circle x y) :
  (∃ x y, circle1 x y ∧ circle2 x y ∧ required_circle x y) ∧
  (∃ x y, required_circle x y ∧ line x y) :=
by
  sorry

end NUMINAMATH_GPT_required_circle_properties_l68_6826


namespace NUMINAMATH_GPT_men_with_6_boys_work_l68_6843

theorem men_with_6_boys_work (m b : ℚ) (x : ℕ) :
  2 * m + 4 * b = 1 / 4 →
  x * m + 6 * b = 1 / 3 →
  2 * b = 5 * m →
  x = 1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_men_with_6_boys_work_l68_6843


namespace NUMINAMATH_GPT_cookies_with_new_flour_l68_6870

-- Definitions for the conditions
def cookies_per_cup (total_cookies : ℕ) (total_flour : ℕ) : ℕ :=
  total_cookies / total_flour

noncomputable def cookies_from_flour (cookies_per_cup : ℕ) (flour : ℕ) : ℕ :=
  cookies_per_cup * flour

-- Given data
def total_cookies := 24
def total_flour := 4
def new_flour := 3

-- Theorem (problem statement)
theorem cookies_with_new_flour : cookies_from_flour (cookies_per_cup total_cookies total_flour) new_flour = 18 :=
by
  sorry

end NUMINAMATH_GPT_cookies_with_new_flour_l68_6870


namespace NUMINAMATH_GPT_x1_x2_product_l68_6831

theorem x1_x2_product (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : x1^2 - 2006 * x1 = 1) (h3 : x2^2 - 2006 * x2 = 1) : x1 * x2 = -1 := 
by
  sorry

end NUMINAMATH_GPT_x1_x2_product_l68_6831


namespace NUMINAMATH_GPT_solve_x_l68_6849

theorem solve_x (x y : ℝ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 16) : x = 16 := by
  sorry

end NUMINAMATH_GPT_solve_x_l68_6849


namespace NUMINAMATH_GPT_solve_system_of_equations_l68_6810

theorem solve_system_of_equations (x y z t : ℝ) :
  xy - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18 ↔ (x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∨ t = 0) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l68_6810


namespace NUMINAMATH_GPT_max_x2_y2_z4_l68_6874

theorem max_x2_y2_z4 (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hxyz : x + y + z = 1) :
  x^2 + y^2 + z^4 ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_x2_y2_z4_l68_6874


namespace NUMINAMATH_GPT_tables_count_l68_6854

def total_tables (four_legged_tables three_legged_tables : Nat) : Nat :=
  four_legged_tables + three_legged_tables

theorem tables_count
  (four_legged_tables three_legged_tables : Nat)
  (total_legs : Nat)
  (h1 : four_legged_tables = 16)
  (h2 : total_legs = 124)
  (h3 : 4 * four_legged_tables + 3 * three_legged_tables = total_legs) :
  total_tables four_legged_tables three_legged_tables = 36 :=
by
  sorry

end NUMINAMATH_GPT_tables_count_l68_6854


namespace NUMINAMATH_GPT_john_caffeine_consumption_l68_6894

noncomputable def caffeine_consumed : ℝ :=
let drink1_ounces : ℝ := 12
let drink1_caffeine : ℝ := 250
let drink2_ratio : ℝ := 3
let drink2_ounces : ℝ := 2

-- Calculate caffeine per ounce in the first drink
let caffeine1_per_ounce : ℝ := drink1_caffeine / drink1_ounces

-- Calculate caffeine per ounce in the second drink
let caffeine2_per_ounce : ℝ := caffeine1_per_ounce * drink2_ratio

-- Calculate total caffeine in the second drink
let drink2_caffeine : ℝ := caffeine2_per_ounce * drink2_ounces

-- Total caffeine from both drinks
let total_drinks_caffeine : ℝ := drink1_caffeine + drink2_caffeine

-- Caffeine in the pill is as much as the total from both drinks
let pill_caffeine : ℝ := total_drinks_caffeine

-- Total caffeine consumed
(drink1_caffeine + drink2_caffeine) + pill_caffeine

theorem john_caffeine_consumption :
  caffeine_consumed = 749.96 := by
    -- Proof is omitted
    sorry

end NUMINAMATH_GPT_john_caffeine_consumption_l68_6894


namespace NUMINAMATH_GPT_complement_union_A_B_is_correct_l68_6802

-- Define the set of real numbers R
def R : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | ∃ (y : ℝ), y = Real.log (x + 3) }

-- Simplified definition for A to reflect x > -3
def A_simplified : Set ℝ := { x | x > -3 }

-- Define set B
def B : Set ℝ := { x | x ≥ 2 }

-- Define the union of A and B
def union_A_B : Set ℝ := A_simplified ∪ B

-- Define the complement of the union in R
def complement_R_union_A_B : Set ℝ := R \ union_A_B

-- State the theorem
theorem complement_union_A_B_is_correct :
  complement_R_union_A_B = { x | x ≤ -3 } := by
  sorry

end NUMINAMATH_GPT_complement_union_A_B_is_correct_l68_6802


namespace NUMINAMATH_GPT_k_is_even_set_l68_6837

open Set -- using Set from Lean library

noncomputable def kSet (s : Set ℤ) :=
  (∀ g ∈ ({5, 8, 7, 1} : Set ℤ), ∀ k ∈ s, (g * k) % 2 = 0)

theorem k_is_even_set (s : Set ℤ) :
  (∀ g ∈ ({5, 8, 7, 1} : Set ℤ), ∀ k ∈ s, (g * k) % 2 = 0) →
  ∀ k ∈ s, k % 2 = 0 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_k_is_even_set_l68_6837


namespace NUMINAMATH_GPT_largest_prime_factor_sum_of_four_digit_numbers_l68_6847

theorem largest_prime_factor_sum_of_four_digit_numbers 
  (a b c d : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
  (h3 : 1 ≤ b) (h4 : b ≤ 9) 
  (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 1 ≤ d) (h8 : d ≤ 9) 
  (h_diff : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  : Nat.gcd 6666 (a + b + c + d) = 101 :=
sorry

end NUMINAMATH_GPT_largest_prime_factor_sum_of_four_digit_numbers_l68_6847


namespace NUMINAMATH_GPT_seats_per_table_l68_6842

-- Definitions based on conditions
def tables := 4
def total_people := 32

-- Statement to prove
theorem seats_per_table : (total_people / tables) = 8 :=
by 
  sorry

end NUMINAMATH_GPT_seats_per_table_l68_6842


namespace NUMINAMATH_GPT_students_in_class_l68_6848

-- Define the relevant variables and conditions
variables (P H W T A S : ℕ)

-- Given conditions
axiom poetry_club : P = 22
axiom history_club : H = 27
axiom writing_club : W = 28
axiom two_clubs : T = 6
axiom all_clubs : A = 6

-- Statement to prove
theorem students_in_class
  (poetry_club : P = 22)
  (history_club : H = 27)
  (writing_club : W = 28)
  (two_clubs : T = 6)
  (all_clubs : A = 6) :
  S = P + H + W - T - 2 * A :=
sorry

end NUMINAMATH_GPT_students_in_class_l68_6848


namespace NUMINAMATH_GPT_eval_f_four_times_l68_6883

noncomputable def f (z : Complex) : Complex := 
if z.im ≠ 0 then z * z else -(z * z)

theorem eval_f_four_times : 
  f (f (f (f (Complex.mk 2 1)))) = Complex.mk 164833 354192 := 
by 
  sorry

end NUMINAMATH_GPT_eval_f_four_times_l68_6883


namespace NUMINAMATH_GPT_size_of_first_type_package_is_5_l68_6856

noncomputable def size_of_first_type_package (total_coffee : ℕ) (num_first_type : ℕ) (num_second_type : ℕ) (size_second_type : ℕ) : ℕ :=
  (total_coffee - num_second_type * size_second_type) / num_first_type

theorem size_of_first_type_package_is_5 :
  size_of_first_type_package 70 (4 + 2) 4 10 = 5 :=
by
  sorry

end NUMINAMATH_GPT_size_of_first_type_package_is_5_l68_6856


namespace NUMINAMATH_GPT_investment_percentage_change_l68_6803

/-- 
Isabel's investment problem statement:
Given an initial investment, and percentage changes over three years,
prove that the overall percentage change in Isabel's investment is 1.2% gain.
-/
theorem investment_percentage_change (initial_investment : ℝ) (gain1 : ℝ) (loss2 : ℝ) (gain3 : ℝ) 
    (final_investment : ℝ) :
    initial_investment = 500 →
    gain1 = 0.10 →
    loss2 = 0.20 →
    gain3 = 0.15 →
    final_investment = initial_investment * (1 + gain1) * (1 - loss2) * (1 + gain3) →
    ((final_investment - initial_investment) / initial_investment) * 100 = 1.2 :=
by
  intros h_init h_gain1 h_loss2 h_gain3 h_final
  sorry

end NUMINAMATH_GPT_investment_percentage_change_l68_6803


namespace NUMINAMATH_GPT_rectangle_parallelepiped_angles_l68_6857

theorem rectangle_parallelepiped_angles 
  (a b c d : ℝ) 
  (α β : ℝ) 
  (h_a : a = d * Real.sin β)
  (h_b : b = d * Real.sin α)
  (h_d : d^2 = (d * Real.sin β)^2 + c^2 + (d * Real.sin α)^2) :
  (α > 0 ∧ β > 0 ∧ α + β < 90) := sorry

end NUMINAMATH_GPT_rectangle_parallelepiped_angles_l68_6857


namespace NUMINAMATH_GPT_petya_wins_l68_6829

theorem petya_wins (n : ℕ) : n = 111 → (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → ∃ x : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ (n - k - x) % 10 = 0) → wins_optimal_play := sorry

end NUMINAMATH_GPT_petya_wins_l68_6829


namespace NUMINAMATH_GPT_max_profit_under_budget_max_profit_no_budget_l68_6875

-- Definitions from conditions
def sales_revenue (x1 x2 : ℝ) : ℝ :=
  -2 * x1^2 - x2^2 + 13 * x1 + 11 * x2 - 28

def profit (x1 x2 : ℝ) : ℝ :=
  sales_revenue x1 x2 - x1 - x2

-- Statements for the conditions
theorem max_profit_under_budget :
  (∀ x1 x2 : ℝ, x1 + x2 = 5 → profit x1 x2 ≤ 9) ∧
  (profit 2 3 = 9) :=
by sorry

theorem max_profit_no_budget :
  (∀ x1 x2 : ℝ, profit x1 x2 ≤ 15) ∧
  (profit 3 5 = 15) :=
by sorry

end NUMINAMATH_GPT_max_profit_under_budget_max_profit_no_budget_l68_6875


namespace NUMINAMATH_GPT_cos_double_angle_of_parallel_vectors_l68_6824

variables {α : Type*}

/-- Given vectors a and b specified by the problem, if they are parallel, then cos 2α = 7/9. -/
theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (1/3, Real.tan α)) 
  (hb : b = (Real.cos α, 1)) 
  (parallel : a.1 * b.2 = a.2 * b.1) : 
  Real.cos (2 * α) = 7/9 := 
by 
  sorry

end NUMINAMATH_GPT_cos_double_angle_of_parallel_vectors_l68_6824


namespace NUMINAMATH_GPT_n_to_the_4_plus_4_to_the_n_composite_l68_6884

theorem n_to_the_4_plus_4_to_the_n_composite (n : ℕ) (h : n ≥ 2) : ¬Prime (n^4 + 4^n) := 
sorry

end NUMINAMATH_GPT_n_to_the_4_plus_4_to_the_n_composite_l68_6884


namespace NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_4_l68_6859

theorem smallest_four_digit_number_divisible_by_4 : 
  ∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ (n % 4 = 0) ∧ n = 1000 := by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_number_divisible_by_4_l68_6859


namespace NUMINAMATH_GPT_distinct_positive_integers_mod_1998_l68_6816

theorem distinct_positive_integers_mod_1998
  (a : Fin 93 → ℕ)
  (h_distinct : Function.Injective a) :
  ∃ m n p q : Fin 93, (m ≠ n ∧ p ≠ q) ∧ (a m - a n) * (a p - a q) % 1998 = 0 :=
by
  sorry

end NUMINAMATH_GPT_distinct_positive_integers_mod_1998_l68_6816


namespace NUMINAMATH_GPT_problem_f8_f2018_l68_6827

theorem problem_f8_f2018 (f : ℕ → ℝ) (h₀ : ∀ n, f (n + 3) = (f n - 1) / (f n + 1)) 
  (h₁ : f 1 ≠ 0) (h₂ : f 1 ≠ 1) (h₃ : f 1 ≠ -1) : 
  f 8 * f 2018 = -1 :=
sorry

end NUMINAMATH_GPT_problem_f8_f2018_l68_6827


namespace NUMINAMATH_GPT_island_width_l68_6858

theorem island_width (area length width : ℕ) (h₁ : area = 50) (h₂ : length = 10) : width = area / length := by 
  sorry

end NUMINAMATH_GPT_island_width_l68_6858


namespace NUMINAMATH_GPT_find_x_l68_6881

theorem find_x (y : ℝ) (x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y - 2)) :
  x = (y^2 + 2 * y + 3) / 5 := by
  sorry

end NUMINAMATH_GPT_find_x_l68_6881


namespace NUMINAMATH_GPT_mean_of_four_numbers_l68_6806

theorem mean_of_four_numbers (a b c d : ℝ) (h : (a + b + c + d + 130) / 5 = 90) : (a + b + c + d) / 4 = 80 := by
  sorry

end NUMINAMATH_GPT_mean_of_four_numbers_l68_6806


namespace NUMINAMATH_GPT_original_quantity_of_ghee_l68_6811

theorem original_quantity_of_ghee
  (Q : ℝ) 
  (H1 : (0.5 * Q) = (0.3 * (Q + 20))) : 
  Q = 30 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_original_quantity_of_ghee_l68_6811


namespace NUMINAMATH_GPT_harmony_implication_at_least_N_plus_1_zero_l68_6897

noncomputable def is_harmony (A B : ℕ → ℕ) (i : ℕ) : Prop :=
  A i = (1 / (2 * B i + 1)) * (Finset.range (2 * B i + 1)).sum (fun s => A (i + s - B i))

theorem harmony_implication_at_least_N_plus_1_zero {N : ℕ} (A B : ℕ → ℕ)
  (hN : N ≥ 2) 
  (h_nonneg_A : ∀ i, 0 ≤ A i)
  (h_nonneg_B : ∀ i, 0 ≤ B i)
  (h_periodic_A : ∀ i, A i = A ((i % N) + 1))
  (h_periodic_B : ∀ i, B i = B ((i % N) + 1))
  (h_harmony_AB : ∀ i, is_harmony A B i)
  (h_harmony_BA : ∀ i, is_harmony B A i)
  (h_not_constant_A : ¬ ∀ i j, A i = A j)
  (h_not_constant_B : ¬ ∀ i j, B i = B j) :
  Finset.card (Finset.filter (fun i => A i = 0 ∨ B i = 0) (Finset.range (N * 2))) ≥ N + 1 := by
  sorry

end NUMINAMATH_GPT_harmony_implication_at_least_N_plus_1_zero_l68_6897


namespace NUMINAMATH_GPT_remainder_of_4n_minus_6_l68_6821

theorem remainder_of_4n_minus_6 (n : ℕ) (h : n % 9 = 5) : (4 * n - 6) % 9 = 5 :=
sorry

end NUMINAMATH_GPT_remainder_of_4n_minus_6_l68_6821


namespace NUMINAMATH_GPT_sum_of_points_probabilities_l68_6899

-- Define probabilities for the sums of 2, 3, and 4
def P_A : ℚ := 1 / 36
def P_B : ℚ := 2 / 36
def P_C : ℚ := 3 / 36

-- Theorem statement
theorem sum_of_points_probabilities :
  (P_A < P_B) ∧ (P_B < P_C) :=
  sorry

end NUMINAMATH_GPT_sum_of_points_probabilities_l68_6899


namespace NUMINAMATH_GPT_fraction_increases_l68_6809

theorem fraction_increases (a : ℝ) (h : ℝ) (ha : a > -1) (hh : h > 0) : 
  (a + h) / (a + h + 1) > a / (a + 1) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_increases_l68_6809


namespace NUMINAMATH_GPT_complex_quadrant_l68_6860

theorem complex_quadrant (a b : ℝ) (h : (a + Complex.I) / (b - Complex.I) = 2 - Complex.I) :
  (a < 0 ∧ b < 0) :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l68_6860


namespace NUMINAMATH_GPT_most_numerous_fruit_l68_6861

-- Define the number of boxes
def num_boxes_tangerines := 5
def num_boxes_apples := 3
def num_boxes_pears := 4

-- Define the number of fruits per box
def tangerines_per_box := 30
def apples_per_box := 20
def pears_per_box := 15

-- Calculate the total number of each fruit
def total_tangerines := num_boxes_tangerines * tangerines_per_box
def total_apples := num_boxes_apples * apples_per_box
def total_pears := num_boxes_pears * pears_per_box

-- State the theorem and prove it
theorem most_numerous_fruit :
  total_tangerines = 150 ∧ total_tangerines > total_apples ∧ total_tangerines > total_pears :=
by
  -- Add here the necessary calculations to verify the conditions
  sorry

end NUMINAMATH_GPT_most_numerous_fruit_l68_6861


namespace NUMINAMATH_GPT_martin_distance_l68_6814

def speed : ℝ := 12.0  -- Speed in miles per hour
def time : ℝ := 6.0    -- Time in hours

theorem martin_distance : (speed * time) = 72.0 :=
by
  sorry

end NUMINAMATH_GPT_martin_distance_l68_6814


namespace NUMINAMATH_GPT_mike_picked_64_peaches_l68_6813

theorem mike_picked_64_peaches :
  ∀ (initial peaches_given total final_picked : ℕ),
    initial = 34 →
    peaches_given = 12 →
    total = 86 →
    final_picked = total - (initial - peaches_given) →
    final_picked = 64 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mike_picked_64_peaches_l68_6813


namespace NUMINAMATH_GPT_digit_problem_l68_6804

theorem digit_problem (A B C D E F : ℕ) (hABC : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ D ≠ E ∧ D ≠ F ∧ E ≠ F) 
    (h1 : 100 * A + 10 * B + C = D * 100000 + A * 10000 + E * 1000 + C * 100 + F * 10 + B)
    (h2 : 100 * C + 10 * B + A = E * 100000 + D * 10000 + C * 1000 + A * 100 + B * 10 + F) : 
    A = 3 ∧ B = 6 ∧ C = 4 ∧ D = 1 ∧ E = 2 ∧ F = 9 := 
sorry

end NUMINAMATH_GPT_digit_problem_l68_6804


namespace NUMINAMATH_GPT_factorized_polynomial_sum_of_squares_l68_6852

theorem factorized_polynomial_sum_of_squares :
  ∃ a b c d e f : ℤ, 
    (729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
    (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210) :=
sorry

end NUMINAMATH_GPT_factorized_polynomial_sum_of_squares_l68_6852


namespace NUMINAMATH_GPT_second_batch_jelly_beans_weight_l68_6869

theorem second_batch_jelly_beans_weight (J : ℝ) (h1 : 2 * 3 + J > 0) (h2 : (6 + J) * 2 = 16) : J = 2 :=
sorry

end NUMINAMATH_GPT_second_batch_jelly_beans_weight_l68_6869


namespace NUMINAMATH_GPT_truckToCarRatio_l68_6817

-- Conditions
def liftsCar (C : ℕ) : Prop := C = 5
def peopleNeeded (C T : ℕ) : Prop := 6 * C + 3 * T = 60

-- Theorem statement
theorem truckToCarRatio (C T : ℕ) (hc : liftsCar C) (hp : peopleNeeded C T) : T / C = 2 :=
by
  sorry

end NUMINAMATH_GPT_truckToCarRatio_l68_6817


namespace NUMINAMATH_GPT_Ofelia_savings_l68_6851

theorem Ofelia_savings (X : ℝ) (h : 16 * X = 160) : X = 10 :=
by
  sorry

end NUMINAMATH_GPT_Ofelia_savings_l68_6851


namespace NUMINAMATH_GPT_compare_neg_fractions_l68_6812

theorem compare_neg_fractions : (-5 / 4) < (-4 / 5) := sorry

end NUMINAMATH_GPT_compare_neg_fractions_l68_6812


namespace NUMINAMATH_GPT_length_gh_parallel_lines_l68_6822

theorem length_gh_parallel_lines (
    AB CD EF GH : ℝ
) (
    h1 : AB = 300
) (
    h2 : CD = 200
) (
    h3 : EF = (AB + CD) / 2 * (1 / 2)
) (
    h4 : GH = EF * (1 - 1 / 4)
) :
    GH = 93.75 :=
by
    sorry

end NUMINAMATH_GPT_length_gh_parallel_lines_l68_6822


namespace NUMINAMATH_GPT_min_age_of_youngest_person_l68_6886

theorem min_age_of_youngest_person
  {a b c d e : ℕ}
  (h_sum : a + b + c + d + e = 256)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h_diff : 2 ≤ (b - a) ∧ (b - a) ≤ 10 ∧ 
            2 ≤ (c - b) ∧ (c - b) ≤ 10 ∧ 
            2 ≤ (d - c) ∧ (d - c) ≤ 10 ∧ 
            2 ≤ (e - d) ∧ (e - d) ≤ 10) : 
  a = 32 :=
sorry

end NUMINAMATH_GPT_min_age_of_youngest_person_l68_6886


namespace NUMINAMATH_GPT_johns_out_of_pocket_expense_l68_6834

theorem johns_out_of_pocket_expense :
  let computer_cost := 700
  let accessories_cost := 200
  let playstation_value := 400
  let playstation_loss_percent := 0.2
  (computer_cost + accessories_cost - playstation_value * (1 - playstation_loss_percent) = 580) :=
by {
  sorry
}

end NUMINAMATH_GPT_johns_out_of_pocket_expense_l68_6834


namespace NUMINAMATH_GPT_arithmetic_progression_terms_l68_6865

theorem arithmetic_progression_terms
  (n : ℕ) (a d : ℝ)
  (hn_odd : n % 2 = 1)
  (sum_odd_terms : n / 2 * (2 * a + (n / 2 - 1) * d) = 30)
  (sum_even_terms : (n / 2 - 1) * (2 * (a + d) + (n / 2 - 2) * d) = 36)
  (sum_all_terms : n / 2 * (2 * a + (n - 1) * d) = 66)
  (last_first_diff : (n - 1) * d = 12) :
  n = 9 := sorry

end NUMINAMATH_GPT_arithmetic_progression_terms_l68_6865


namespace NUMINAMATH_GPT_merchant_printer_count_l68_6808

theorem merchant_printer_count (P : ℕ) 
  (cost_keyboards : 15 * 20 = 300)
  (total_cost : 300 + 70 * P = 2050) :
  P = 25 := 
by
  sorry

end NUMINAMATH_GPT_merchant_printer_count_l68_6808


namespace NUMINAMATH_GPT_projectile_height_time_l68_6839

theorem projectile_height_time :
  ∃ t, t ≥ 0 ∧ -16 * t^2 + 80 * t = 72 ↔ t = 1 := 
by sorry

end NUMINAMATH_GPT_projectile_height_time_l68_6839


namespace NUMINAMATH_GPT_shorter_side_length_l68_6825

theorem shorter_side_length (L W : ℝ) (h1 : L * W = 91) (h2 : 2 * L + 2 * W = 40) :
  min L W = 7 :=
by
  sorry

end NUMINAMATH_GPT_shorter_side_length_l68_6825


namespace NUMINAMATH_GPT_total_fishes_caught_l68_6888

def melanieCatches : ℕ := 8
def tomCatches : ℕ := 3 * melanieCatches
def totalFishes : ℕ := melanieCatches + tomCatches

theorem total_fishes_caught : totalFishes = 32 := by
  sorry

end NUMINAMATH_GPT_total_fishes_caught_l68_6888


namespace NUMINAMATH_GPT_D_cows_grazed_l68_6819

-- Defining the given conditions:
def A_cows := 24
def A_months := 3
def A_rent := 1440

def B_cows := 10
def B_months := 5

def C_cows := 35
def C_months := 4

def D_months := 3

def total_rent := 6500

-- Calculate the cost per cow per month (CPCM)
def CPCM := A_rent / (A_cows * A_months)

-- Proving the number of cows D grazed
theorem D_cows_grazed : ∃ x : ℕ, (x * D_months * CPCM + A_rent + (B_cows * B_months * CPCM) + (C_cows * C_months * CPCM) = total_rent) ∧ x = 21 := by
  sorry

end NUMINAMATH_GPT_D_cows_grazed_l68_6819


namespace NUMINAMATH_GPT_factor_diff_of_squares_l68_6820

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end NUMINAMATH_GPT_factor_diff_of_squares_l68_6820


namespace NUMINAMATH_GPT_eq_to_general_quadratic_l68_6893

theorem eq_to_general_quadratic (x : ℝ) : (x - 1) * (x + 1) = 1 → x^2 - 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_eq_to_general_quadratic_l68_6893


namespace NUMINAMATH_GPT_still_water_speed_l68_6871

-- The conditions as given in the problem
variables (V_m V_r V'_r : ℝ)
axiom upstream_speed : V_m - V_r = 20
axiom downstream_increased_speed : V_m + V_r = 30
axiom downstream_reduced_speed : V_m + V'_r = 26

-- Prove that the man's speed in still water is 25 km/h
theorem still_water_speed : V_m = 25 :=
by
  sorry

end NUMINAMATH_GPT_still_water_speed_l68_6871


namespace NUMINAMATH_GPT_find_m_through_point_l68_6892

theorem find_m_through_point :
  ∃ m : ℝ, ∀ (x y : ℝ), ((y = (m - 1) * x - 4) ∧ (x = 2) ∧ (y = 4)) → m = 5 :=
by 
  -- Sorry can be used here to skip the proof as instructed
  sorry

end NUMINAMATH_GPT_find_m_through_point_l68_6892


namespace NUMINAMATH_GPT_part1_part2_l68_6850

-- Part 1: Prove values of m and n.
theorem part1 (m n : ℝ) :
  (∀ x : ℝ, |x - m| ≤ n ↔ 0 ≤ x ∧ x ≤ 4) → m = 2 ∧ n = 2 :=
by
  intro h
  -- Proof omitted
  sorry

-- Part 2: Prove the minimum value of a + b.
theorem part2 (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 2) :
  a + b = (2 / a) + (2 / b) → a + b ≥ 2 * Real.sqrt 2 :=
by
  intro h
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_part1_part2_l68_6850


namespace NUMINAMATH_GPT_compatible_polynomial_count_l68_6807

theorem compatible_polynomial_count (n : ℕ) : 
  ∃ num_polynomials : ℕ, num_polynomials = (n / 2) + 1 :=
by
  sorry

end NUMINAMATH_GPT_compatible_polynomial_count_l68_6807


namespace NUMINAMATH_GPT_max_value_expression_l68_6872

theorem max_value_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hsum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27 / 8 :=
sorry

end NUMINAMATH_GPT_max_value_expression_l68_6872


namespace NUMINAMATH_GPT_fewest_posts_l68_6864

def grazingAreaPosts (length width post_interval rock_wall_length : ℕ) : ℕ :=
  let side1 := width / post_interval + 1
  let side2 := length / post_interval
  side1 + 2 * side2

theorem fewest_posts (length width post_interval rock_wall_length posts : ℕ) :
  length = 70 ∧ width = 50 ∧ post_interval = 10 ∧ rock_wall_length = 150 ∧ posts = 18 →
  grazingAreaPosts length width post_interval rock_wall_length = posts := 
by
  intros h
  obtain ⟨hl, hw, hp, hr, ht⟩ := h
  simp [grazingAreaPosts, hl, hw, hp, hr]
  sorry

end NUMINAMATH_GPT_fewest_posts_l68_6864


namespace NUMINAMATH_GPT_linear_function_iff_l68_6882

variable {x : ℝ} (m : ℝ)

def f (m : ℝ) (x : ℝ) : ℝ := (m + 2) * x + 4 * x - 5

theorem linear_function_iff (m : ℝ) : 
  (∃ c d, ∀ x, f m x = c * x + d) ↔ m ≠ -6 :=
by 
  sorry

end NUMINAMATH_GPT_linear_function_iff_l68_6882


namespace NUMINAMATH_GPT_no_solutions_xyz_l68_6800

theorem no_solutions_xyz :
  ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_no_solutions_xyz_l68_6800


namespace NUMINAMATH_GPT_red_car_speed_is_10mph_l68_6832

noncomputable def speed_of_red_car (speed_black : ℝ) (initial_distance : ℝ) (time_to_overtake : ℝ) : ℝ :=
  (speed_black * time_to_overtake - initial_distance) / time_to_overtake

theorem red_car_speed_is_10mph :
  ∀ (speed_black initial_distance time_to_overtake : ℝ),
  speed_black = 50 →
  initial_distance = 20 →
  time_to_overtake = 0.5 →
  speed_of_red_car speed_black initial_distance time_to_overtake = 10 :=
by
  intros speed_black initial_distance time_to_overtake hb hd ht
  rw [hb, hd, ht]
  norm_num
  sorry

end NUMINAMATH_GPT_red_car_speed_is_10mph_l68_6832


namespace NUMINAMATH_GPT_renovation_cost_distribution_l68_6801

/-- A mathematical proof that if Team A works alone for 3 weeks, followed by both Team A and Team B working together, and the total renovation cost is 4000 yuan, then the payment should be distributed equally between Team A and Team B, each receiving 2000 yuan. -/
theorem renovation_cost_distribution :
  let time_A := 18
  let time_B := 12
  let initial_time_A := 3
  let total_cost := 4000
  ∃ x, (1 / time_A * (x + initial_time_A) + 1 / time_B * x = 1) ∧
       let work_A := 1 / time_A * (x + initial_time_A)
       let work_B := 1 / time_B * x
       work_A = work_B ∧
       total_cost / 2 = 2000 :=
by
  sorry

end NUMINAMATH_GPT_renovation_cost_distribution_l68_6801


namespace NUMINAMATH_GPT_tom_has_hours_to_spare_l68_6830

-- Conditions as definitions
def numberOfWalls : Nat := 5
def wallWidth : Nat := 2 -- in meters
def wallHeight : Nat := 3 -- in meters
def paintingRate : Nat := 10 -- in minutes per square meter
def totalAvailableTime : Nat := 10 -- in hours

-- Lean 4 statement of the problem
theorem tom_has_hours_to_spare :
  let areaOfOneWall := wallWidth * wallHeight -- 2 * 3
  let totalArea := numberOfWalls * areaOfOneWall -- 5 * (2 * 3)
  let totalTimeToPaint := (totalArea * paintingRate) / 60 -- (30 * 10) / 60
  totalAvailableTime - totalTimeToPaint = 5 :=
by
  sorry

end NUMINAMATH_GPT_tom_has_hours_to_spare_l68_6830


namespace NUMINAMATH_GPT_parabola_area_l68_6891

theorem parabola_area (m p : ℝ) (h1 : p > 0) (h2 : (1:ℝ)^2 = 2 * p * m)
    (h3 : (1/2) * (m + p / 2) = 1/2) : p = 1 :=
  by
    sorry

end NUMINAMATH_GPT_parabola_area_l68_6891


namespace NUMINAMATH_GPT_sum_two_numbers_eq_twelve_l68_6876

theorem sum_two_numbers_eq_twelve (x y : ℕ) (h1 : x^2 + y^2 = 90) (h2 : x * y = 27) : x + y = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_two_numbers_eq_twelve_l68_6876


namespace NUMINAMATH_GPT_find_a4_l68_6885

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = a 1 - a 0

theorem find_a4 (h1 : arithmetic_sequence a) (h2 : a 2 + a 6 = 2) : a 4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a4_l68_6885


namespace NUMINAMATH_GPT_problem_statement_l68_6868

theorem problem_statement (x y : ℝ) (h1 : 4 * x + y = 12) (h2 : x + 4 * y = 18) :
  17 * x ^ 2 + 24 * x * y + 17 * y ^ 2 = 532 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l68_6868


namespace NUMINAMATH_GPT_crescent_perimeter_l68_6855

def radius_outer : ℝ := 10.5
def radius_inner : ℝ := 6.7

theorem crescent_perimeter : (radius_outer + radius_inner) * Real.pi = 54.037 :=
by
  sorry

end NUMINAMATH_GPT_crescent_perimeter_l68_6855


namespace NUMINAMATH_GPT_distance_to_place_l68_6840

-- Define the conditions
def speed_boat_standing_water : ℝ := 16
def speed_stream : ℝ := 2
def total_time_taken : ℝ := 891.4285714285714

-- Define the calculated speeds
def downstream_speed : ℝ := speed_boat_standing_water + speed_stream
def upstream_speed : ℝ := speed_boat_standing_water - speed_stream

-- Define the variable for the distance
variable (D : ℝ)

-- State the theorem to prove
theorem distance_to_place :
  D / downstream_speed + D / upstream_speed = total_time_taken →
  D = 7020 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_distance_to_place_l68_6840


namespace NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_l68_6873

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end NUMINAMATH_GPT_cos_135_eq_neg_sqrt2_div_2_l68_6873


namespace NUMINAMATH_GPT_evaluate_g_3_times_l68_6818

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1
  else 2 * n + 3

theorem evaluate_g_3_times : g (g (g 3)) = 65 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_3_times_l68_6818


namespace NUMINAMATH_GPT_geom_seq_sum_problem_l68_6898

noncomputable def geom_sum_first_n_terms (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

noncomputable def geom_sum_specific_terms (a₃ q : ℕ) (n m : ℕ) : ℕ :=
  a₃ * ((1 - (q^m) ^ n) / (1 - q^m))

theorem geom_seq_sum_problem :
  ∀ (a₁ q S₈₇ : ℕ),
  q = 2 →
  S₈₇ = 140 →
  geom_sum_first_n_terms a₁ q 87 = S₈₇ →
  ∃ a₃, a₃ = ((q * q) * a₁) →
  geom_sum_specific_terms a₃ q 29 3 = 80 := 
by
  intros a₁ q S₈₇ hq₁ hS₈₇ hsum
  -- Further proof would go here
  sorry

end NUMINAMATH_GPT_geom_seq_sum_problem_l68_6898


namespace NUMINAMATH_GPT_kth_term_in_sequence_l68_6815

theorem kth_term_in_sequence (k : ℕ) (hk : 0 < k) : ℚ :=
  (2 * k) / (2 * k + 1)

end NUMINAMATH_GPT_kth_term_in_sequence_l68_6815


namespace NUMINAMATH_GPT_range_of_a_l68_6828

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l68_6828
