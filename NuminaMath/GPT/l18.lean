import Mathlib

namespace NUMINAMATH_GPT_width_of_rectangle_11_l18_1858

variable (L W : ℕ)

-- The conditions: 
-- 1. The perimeter is 48cm
-- 2. Width is 2 cm shorter than length
def is_rectangle (L W : ℕ) : Prop :=
  2 * L + 2 * W = 48 ∧ W = L - 2

-- The statement we need to prove
theorem width_of_rectangle_11 (L W : ℕ) (h : is_rectangle L W) : W = 11 :=
by
  sorry

end NUMINAMATH_GPT_width_of_rectangle_11_l18_1858


namespace NUMINAMATH_GPT_neg_cos_ge_a_l18_1888

theorem neg_cos_ge_a (a : ℝ) : (¬ ∃ x : ℝ, Real.cos x ≥ a) ↔ a = 2 := 
sorry

end NUMINAMATH_GPT_neg_cos_ge_a_l18_1888


namespace NUMINAMATH_GPT_rectangle_area_l18_1855

theorem rectangle_area (x : ℝ) (w : ℝ) (h_diag : (3 * w) ^ 2 + w ^ 2 = x ^ 2) : 
  3 * w ^ 2 = (3 / 10) * x ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l18_1855


namespace NUMINAMATH_GPT_coeff_x2y2_in_expansion_l18_1856

-- Define the coefficient of a specific term in the binomial expansion
def coeff_binom (n k : ℕ) (a b : ℤ) (x y : ℕ) : ℤ :=
  (Nat.choose n k) * (a ^ (n - k)) * (b ^ k)

theorem coeff_x2y2_in_expansion : coeff_binom 4 2 1 (-2) 2 2 = 24 := by
  sorry

end NUMINAMATH_GPT_coeff_x2y2_in_expansion_l18_1856


namespace NUMINAMATH_GPT_boxes_per_case_l18_1867

theorem boxes_per_case (total_boxes : ℕ) (total_cases : ℕ) (h1 : total_boxes = 24) (h2 : total_cases = 3) : (total_boxes / total_cases) = 8 :=
by 
  sorry

end NUMINAMATH_GPT_boxes_per_case_l18_1867


namespace NUMINAMATH_GPT_total_pens_count_l18_1898

def total_pens (red black blue : ℕ) : ℕ :=
  red + black + blue

theorem total_pens_count :
  let red := 8
  let black := red + 10
  let blue := red + 7
  total_pens red black blue = 41 :=
by
  sorry

end NUMINAMATH_GPT_total_pens_count_l18_1898


namespace NUMINAMATH_GPT_not_prime_for_large_n_l18_1871

theorem not_prime_for_large_n {n : ℕ} (h : n > 1) : ¬ Prime (n^4 + n^2 + 1) :=
sorry

end NUMINAMATH_GPT_not_prime_for_large_n_l18_1871


namespace NUMINAMATH_GPT_weight_of_b_l18_1885

theorem weight_of_b (a b c : ℝ) (h1 : a + b + c = 135) (h2 : a + b = 80) (h3 : b + c = 82) : b = 27 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_b_l18_1885


namespace NUMINAMATH_GPT_solve_equation_l18_1854

theorem solve_equation (x : ℝ) (h1: (6 * x) ^ 18 = (12 * x) ^ 9) (h2 : x ≠ 0) : x = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_solve_equation_l18_1854


namespace NUMINAMATH_GPT_remaining_pieces_total_l18_1853

noncomputable def initial_pieces : Nat := 16
noncomputable def kennedy_lost_pieces : Nat := 4 + 1 + 2
noncomputable def riley_lost_pieces : Nat := 1 + 1 + 1

theorem remaining_pieces_total : (initial_pieces - kennedy_lost_pieces) + (initial_pieces - riley_lost_pieces) = 22 := by
  sorry

end NUMINAMATH_GPT_remaining_pieces_total_l18_1853


namespace NUMINAMATH_GPT_find_difference_of_max_and_min_values_l18_1835

noncomputable def v (a b : Int) : Int := a * (-4) + b

theorem find_difference_of_max_and_min_values :
  let v0 := 3
  let v1 := v v0 12
  let v2 := v v1 6
  let v3 := v v2 10
  let v4 := v v3 (-8)
  (max (max (max (max v0 v1) v2) v3) v4) - (min (min (min (min v0 v1) v2) v3) v4) = 62 :=
by
  sorry

end NUMINAMATH_GPT_find_difference_of_max_and_min_values_l18_1835


namespace NUMINAMATH_GPT_trigonometric_identity_l18_1881

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (Real.sin θ * Real.cos θ) / (1 + Real.sin θ ^ 2) = 2 / 9 := 
sorry

end NUMINAMATH_GPT_trigonometric_identity_l18_1881


namespace NUMINAMATH_GPT_limit_an_to_a_l18_1876

theorem limit_an_to_a (ε : ℝ) (hε : ε > 0) : 
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N →
  |(9 - (n^3 : ℝ)) / (1 + 2 * (n^3 : ℝ)) + 1/2| < ε :=
sorry

end NUMINAMATH_GPT_limit_an_to_a_l18_1876


namespace NUMINAMATH_GPT_max_value_of_f_l18_1808

noncomputable def f (x : ℝ) : ℝ := 3 * x - x ^ 3

theorem max_value_of_f (a b : ℝ) (ha : ∀ x, f x ≤ b) (hfa : f a = b) : a - b = -1 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l18_1808


namespace NUMINAMATH_GPT_average_words_per_hour_l18_1878

-- Define the given conditions
variables (W : ℕ) (H : ℕ)

-- State constants for the known values
def words := 60000
def writing_hours := 100

-- Define theorem to prove the average words per hour during the writing phase
theorem average_words_per_hour (h : W = words) (h2 : H = writing_hours) : (W / H) = 600 := by
  sorry

end NUMINAMATH_GPT_average_words_per_hour_l18_1878


namespace NUMINAMATH_GPT_polynomial_value_l18_1837

-- Define the conditions as Lean definitions
def condition (x : ℝ) : Prop := x^2 + 2 * x + 1 = 4

-- State the theorem to be proved
theorem polynomial_value (x : ℝ) (h : condition x) : 2 * x^2 + 4 * x + 5 = 11 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_polynomial_value_l18_1837


namespace NUMINAMATH_GPT_total_games_in_season_l18_1877

theorem total_games_in_season {n : ℕ} {k : ℕ} (h1 : n = 25) (h2 : k = 15) :
  (n * (n - 1) / 2) * k = 4500 :=
by
  sorry

end NUMINAMATH_GPT_total_games_in_season_l18_1877


namespace NUMINAMATH_GPT_find_number_being_divided_l18_1864

theorem find_number_being_divided (divisor quotient remainder : ℕ) (h1: divisor = 15) (h2: quotient = 9) (h3: remainder = 1) : 
  divisor * quotient + remainder = 136 :=
by
  -- Simplification and computation would follow here
  sorry

end NUMINAMATH_GPT_find_number_being_divided_l18_1864


namespace NUMINAMATH_GPT_recreation_percentage_l18_1819

variable (W : ℝ) -- John's wages last week
variable (recreation_last_week : ℝ := 0.35 * W) -- Amount spent on recreation last week
variable (wages_this_week : ℝ := 0.70 * W) -- Wages this week
variable (recreation_this_week : ℝ := 0.25 * wages_this_week) -- Amount spent on recreation this week

theorem recreation_percentage :
  (recreation_this_week / recreation_last_week) * 100 = 50 := by
  sorry

end NUMINAMATH_GPT_recreation_percentage_l18_1819


namespace NUMINAMATH_GPT_angle_measure_l18_1831

theorem angle_measure (x : ℝ) (h : x + (3 * x - 10) = 180) : x = 47.5 := 
by
  sorry

end NUMINAMATH_GPT_angle_measure_l18_1831


namespace NUMINAMATH_GPT_discount_rate_on_pony_jeans_l18_1821

-- Define the conditions as Lean definitions
def fox_price : ℝ := 15
def pony_price : ℝ := 18
def total_savings : ℝ := 8.91
def total_discount_rate : ℝ := 22
def number_of_fox_pairs : ℕ := 3
def number_of_pony_pairs : ℕ := 2

-- Given definitions of the discount rates on Fox and Pony jeans
variable (F P : ℝ)

-- The system of equations based on the conditions
axiom sum_of_discount_rates : F + P = total_discount_rate
axiom savings_equation : 
  number_of_fox_pairs * (fox_price * F / 100) + number_of_pony_pairs * (pony_price * P / 100) = total_savings

-- The theorem to prove
theorem discount_rate_on_pony_jeans : P = 11 := by
  sorry

end NUMINAMATH_GPT_discount_rate_on_pony_jeans_l18_1821


namespace NUMINAMATH_GPT_percentage_of_children_speaking_only_Hindi_l18_1815

/-
In a class of 60 children, 30% of children can speak only English,
20% can speak both Hindi and English, and 42 children can speak Hindi.
Prove that the percentage of children who can speak only Hindi is 50%.
-/
theorem percentage_of_children_speaking_only_Hindi :
  let total_children := 60
  let english_only := 0.30 * total_children
  let both_languages := 0.20 * total_children
  let hindi_only := 42 - both_languages
  (hindi_only / total_children) * 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_children_speaking_only_Hindi_l18_1815


namespace NUMINAMATH_GPT_division_problem_l18_1861

theorem division_problem (x : ℕ) (h : x / 5 = 30 + x / 6) : x = 900 :=
sorry

end NUMINAMATH_GPT_division_problem_l18_1861


namespace NUMINAMATH_GPT_highest_power_of_3_l18_1846

-- Define the integer M formed by concatenating the 3-digit numbers from 100 to 250
def M : ℕ := sorry  -- We should define it in a way that represents the concatenation

-- Define a proof that the highest power of 3 that divides M is 3^1
theorem highest_power_of_3 (n : ℕ) (h : M = n) : ∃ m : ℕ, 3^m ∣ n ∧ ¬ (3^(m + 1) ∣ n) ∧ m = 1 :=
by sorry  -- We will not provide proofs; we're only writing the statement

end NUMINAMATH_GPT_highest_power_of_3_l18_1846


namespace NUMINAMATH_GPT_digit_C_equals_one_l18_1840

-- Define the scope of digits
def is_digit (n : ℕ) : Prop := n < 10

-- Define the equality for sums of digits
def sum_of_digits (A B C : ℕ) : Prop := A + B + C = 10

-- Main theorem to prove C = 1
theorem digit_C_equals_one (A B C : ℕ) (hA : is_digit A) (hB : is_digit B) (hC : is_digit C) (hSum : sum_of_digits A B C) : C = 1 :=
sorry

end NUMINAMATH_GPT_digit_C_equals_one_l18_1840


namespace NUMINAMATH_GPT_intersection_of_sets_l18_1810

theorem intersection_of_sets :
  let A := {y : ℝ | ∃ x : ℝ, y = Real.sin x}
  let B := {y : ℝ | ∃ x : ℝ, y = Real.sqrt (-(x^2 - 4*x + 3))}
  A ∩ B = {y : ℝ | 0 ≤ y ∧ y ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l18_1810


namespace NUMINAMATH_GPT_train_A_total_distance_l18_1822

variables (Speed_A : ℝ) (Time_meet : ℝ) (Total_Distance : ℝ)

def Distance_A_to_C (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Distance_B_to_C (Total_Distance Distance_A_to_C : ℝ) : ℝ := Total_Distance - Distance_A_to_C
def Additional_Distance_A (Speed_A Time_meet : ℝ) : ℝ := Speed_A * Time_meet
def Total_Distance_A (Distance_A_to_C Additional_Distance_A : ℝ) : ℝ :=
  Distance_A_to_C + Additional_Distance_A

theorem train_A_total_distance
  (h1 : Speed_A = 50)
  (h2 : Time_meet = 0.5)
  (h3 : Total_Distance = 120) :
  Total_Distance_A (Distance_A_to_C Speed_A Time_meet)
                   (Additional_Distance_A Speed_A Time_meet) = 50 :=
by 
  rw [Distance_A_to_C, Additional_Distance_A, Total_Distance_A]
  rw [h1, h2]
  norm_num

end NUMINAMATH_GPT_train_A_total_distance_l18_1822


namespace NUMINAMATH_GPT_fraction_to_decimal_l18_1847

theorem fraction_to_decimal : (5 / 16 : ℝ) = 0.3125 :=
by sorry

end NUMINAMATH_GPT_fraction_to_decimal_l18_1847


namespace NUMINAMATH_GPT_cost_of_banana_l18_1880

theorem cost_of_banana (B : ℝ) (apples bananas oranges total_pieces total_cost : ℝ) 
  (h1 : apples = 12) (h2 : bananas = 4) (h3 : oranges = 4) 
  (h4 : total_pieces = 20) (h5 : total_cost = 40)
  (h6 : 2 * apples + 3 * oranges + bananas * B = total_cost)
  : B = 1 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_banana_l18_1880


namespace NUMINAMATH_GPT_smallest_prime_sum_of_three_different_primes_is_19_l18_1886

theorem smallest_prime_sum_of_three_different_primes_is_19 :
  ∃ (p : ℕ), Prime p ∧ p = 19 ∧ (∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → Prime a → Prime b → Prime c → a + b + c = p → p ≥ 19) :=
by
  sorry

end NUMINAMATH_GPT_smallest_prime_sum_of_three_different_primes_is_19_l18_1886


namespace NUMINAMATH_GPT_arithmetic_progression_common_difference_l18_1800

theorem arithmetic_progression_common_difference 
  (x y : ℤ) 
  (h1 : 280 * x^2 - 61 * x * y + 3 * y^2 - 13 = 0) 
  (h2 : ∃ a d : ℤ, x = a + 3 * d ∧ y = a + 8 * d) : 
  ∃ d : ℤ, d = -5 := 
sorry

end NUMINAMATH_GPT_arithmetic_progression_common_difference_l18_1800


namespace NUMINAMATH_GPT_train_crosses_platform_l18_1824

theorem train_crosses_platform :
  ∀ (L : ℕ), 
  (300 + L) / (50 / 3) = 48 → 
  L = 500 := 
by
  sorry

end NUMINAMATH_GPT_train_crosses_platform_l18_1824


namespace NUMINAMATH_GPT_andrew_age_l18_1849

-- Definitions based on the conditions
variables (a g : ℝ)

-- The conditions
def condition1 : Prop := g = 9 * a
def condition2 : Prop := g - a = 63

-- The theorem we want to prove
theorem andrew_age (h1 : condition1 a g) (h2 : condition2 a g) : a = 63 / 8 :=
by
  intros
  sorry

end NUMINAMATH_GPT_andrew_age_l18_1849


namespace NUMINAMATH_GPT_mean_score_of_seniors_l18_1859

theorem mean_score_of_seniors (num_students : ℕ) (mean_score : ℚ) 
  (ratio_non_seniors_seniors : ℚ) (ratio_mean_seniors_non_seniors : ℚ) (total_score_seniors : ℚ) :
  num_students = 200 →
  mean_score = 80 →
  ratio_non_seniors_seniors = 1.25 →
  ratio_mean_seniors_non_seniors = 1.2 →
  total_score_seniors = 7200 →
  let num_seniors := (num_students : ℚ) / (1 + ratio_non_seniors_seniors)
  let mean_score_seniors := total_score_seniors / num_seniors
  mean_score_seniors = 80.9 :=
by 
  sorry

end NUMINAMATH_GPT_mean_score_of_seniors_l18_1859


namespace NUMINAMATH_GPT_determinant_of_matrix_A_l18_1873

noncomputable def matrix_A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![x + 2, x + 1, x], 
    ![x, x + 2, x + 1], 
    ![x + 1, x, x + 2]]

theorem determinant_of_matrix_A (x : ℝ) :
  (matrix_A x).det = x^2 + 11 * x + 9 :=
by sorry

end NUMINAMATH_GPT_determinant_of_matrix_A_l18_1873


namespace NUMINAMATH_GPT_distance_to_right_focus_l18_1891

open Real

-- Define the elements of the problem
variable (a c : ℝ)
variable (P : ℝ × ℝ) -- Point P on the hyperbola
variable (F1 F2 : ℝ × ℝ) -- Left and right foci
variable (D : ℝ) -- The left directrix

-- Define conditions as Lean statements
def hyperbola_eq : Prop := (a ≠ 0) ∧ (c ≠ 0) ∧ (P.1^2 / a^2 - P.2^2 / 16 = 1)
def point_on_right_branch : Prop := P.1 > 0
def distance_diff : Prop := abs (dist P F1 - dist P F2) = 6
def distance_to_left_directrix : Prop := abs (P.1 - D) = 34 / 5

-- Define theorem to prove the distance from P to the right focus
theorem distance_to_right_focus
  (hp : hyperbola_eq a c P)
  (hbranch : point_on_right_branch P)
  (hdiff : distance_diff P F1 F2)
  (hdirectrix : distance_to_left_directrix P D) :
  dist P F2 = 16 / 3 :=
sorry

end NUMINAMATH_GPT_distance_to_right_focus_l18_1891


namespace NUMINAMATH_GPT_solution_set_f_ge_0_l18_1803

variables {f : ℝ → ℝ}

-- Conditions
axiom h1 : ∀ x : ℝ, f (-x) = -f x  -- f is odd function
axiom h2 : ∀ x y : ℝ, 0 < x → x < y → f x < f y  -- f is monotonically increasing on (0, +∞)
axiom h3 : f 3 = 0  -- f(3) = 0

theorem solution_set_f_ge_0 : { x : ℝ | f x ≥ 0 } = { x : ℝ | -3 ≤ x ∧ x ≤ 0 } ∪ { x : ℝ | 3 ≤ x } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_ge_0_l18_1803


namespace NUMINAMATH_GPT_marbles_solution_l18_1830

open Nat

def marbles_problem : Prop :=
  ∃ J_k J_j : Nat, (J_k = 3) ∧ (J_k = J_j - 4) ∧ (J_k + J_j = 10)

theorem marbles_solution : marbles_problem := by
  sorry

end NUMINAMATH_GPT_marbles_solution_l18_1830


namespace NUMINAMATH_GPT_action_figures_more_than_books_l18_1816

variable (initialActionFigures : Nat) (newActionFigures : Nat) (books : Nat)

def totalActionFigures (initialActionFigures newActionFigures : Nat) : Nat :=
  initialActionFigures + newActionFigures

theorem action_figures_more_than_books :
  initialActionFigures = 5 → newActionFigures = 7 → books = 9 →
  totalActionFigures initialActionFigures newActionFigures - books = 3 :=
by
  intros h_initial h_new h_books
  rw [h_initial, h_new, h_books]
  sorry

end NUMINAMATH_GPT_action_figures_more_than_books_l18_1816


namespace NUMINAMATH_GPT_intersection_points_l18_1823

noncomputable def curve1 (x y : ℝ) : Prop := x^2 + 4 * y^2 = 1
noncomputable def curve2 (x y : ℝ) : Prop := 4 * x^2 + y^2 = 4

theorem intersection_points : 
  ∃ (points : Finset (ℝ × ℝ)), 
  (∀ p ∈ points, curve1 p.1 p.2 ∧ curve2 p.1 p.2) ∧ points.card = 2 := 
by 
  sorry

end NUMINAMATH_GPT_intersection_points_l18_1823


namespace NUMINAMATH_GPT_arccos_sqrt_3_over_2_eq_pi_over_6_l18_1870

open Real

theorem arccos_sqrt_3_over_2_eq_pi_over_6 :
  ∀ (x : ℝ), x = (sqrt 3) / 2 → arccos x = π / 6 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_arccos_sqrt_3_over_2_eq_pi_over_6_l18_1870


namespace NUMINAMATH_GPT_Mo_tea_cups_l18_1897

theorem Mo_tea_cups (n t : ℕ) 
  (h1 : 2 * n + 5 * t = 36)
  (h2 : 5 * t = 2 * n + 14) : 
  t = 5 :=
by
  sorry

end NUMINAMATH_GPT_Mo_tea_cups_l18_1897


namespace NUMINAMATH_GPT_second_number_exists_l18_1841

theorem second_number_exists (x : ℕ) (h : 150 / x = 15) : x = 10 :=
sorry

end NUMINAMATH_GPT_second_number_exists_l18_1841


namespace NUMINAMATH_GPT_yujin_wire_length_is_correct_l18_1895

def junhoe_wire_length : ℝ := 134.5
def multiplicative_factor : ℝ := 1.06
def yujin_wire_length (junhoe_length : ℝ) (factor : ℝ) : ℝ := junhoe_length * factor

theorem yujin_wire_length_is_correct : 
  yujin_wire_length junhoe_wire_length multiplicative_factor = 142.57 := 
by 
  sorry

end NUMINAMATH_GPT_yujin_wire_length_is_correct_l18_1895


namespace NUMINAMATH_GPT_repeating_decimal_sum_l18_1805

/--
The number 3.17171717... can be written as a reduced fraction x/y where x = 314 and y = 99.
We aim to prove that the sum of x and y is 413.
-/
theorem repeating_decimal_sum : 
  let x := 314
  let y := 99
  (x + y) = 413 := 
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l18_1805


namespace NUMINAMATH_GPT_number_of_ways_to_divide_day_l18_1826

theorem number_of_ways_to_divide_day (n m : ℕ) (hn : 0 < n) (hm : 0 < m) (h : n * m = 1440) : 
  ∃ (pairs : List (ℕ × ℕ)), (pairs.length = 36) ∧
  (∀ (p : ℕ × ℕ), p ∈ pairs → (p.1 * p.2 = 1440)) :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_divide_day_l18_1826


namespace NUMINAMATH_GPT_total_wheels_l18_1833

def cars : Nat := 15
def bicycles : Nat := 3
def trucks : Nat := 8
def tricycles : Nat := 1
def wheels_per_car_or_truck : Nat := 4
def wheels_per_bicycle : Nat := 2
def wheels_per_tricycle : Nat := 3

theorem total_wheels : cars * wheels_per_car_or_truck + trucks * wheels_per_car_or_truck + bicycles * wheels_per_bicycle + tricycles * wheels_per_tricycle = 101 :=
by
  sorry

end NUMINAMATH_GPT_total_wheels_l18_1833


namespace NUMINAMATH_GPT_calc_length_RS_l18_1893

-- Define the trapezoid properties
def trapezoid (PQRS : Type) (PR QS : ℝ) (h A : ℝ) : Prop :=
  PR = 12 ∧ QS = 20 ∧ h = 10 ∧ A = 180

-- Define the length of the side RS
noncomputable def length_RS (PQRS : Type) (PR QS h A : ℝ) : ℝ :=
  18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3

-- Define the theorem statement
theorem calc_length_RS {PQRS : Type} (PR QS h A : ℝ) :
  trapezoid PQRS PR QS h A → length_RS PQRS PR QS h A = 18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3 :=
by
  intros
  exact Eq.refl (18 - 0.5 * Real.sqrt 44 - 5 * Real.sqrt 3)

end NUMINAMATH_GPT_calc_length_RS_l18_1893


namespace NUMINAMATH_GPT_malcolm_brushes_teeth_l18_1872

theorem malcolm_brushes_teeth :
  (∃ (M : ℕ), M = 180 ∧ (∃ (N : ℕ), N = 90 ∧ (M / N = 2))) :=
by
  sorry

end NUMINAMATH_GPT_malcolm_brushes_teeth_l18_1872


namespace NUMINAMATH_GPT_find_a_l18_1892

theorem find_a (a b c : ℤ) (h : (∀ x : ℝ, (x - a) * (x - 5) + 4 = (x + b) * (x + c))) :
  a = 0 ∨ a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l18_1892


namespace NUMINAMATH_GPT_intersection_shape_is_rectangle_l18_1811

noncomputable def curve1 (x y : ℝ) : Prop := x * y = 16
noncomputable def curve2 (x y : ℝ) : Prop := x^2 + y^2 = 34

theorem intersection_shape_is_rectangle (x y : ℝ) :
  (curve1 x y ∧ curve2 x y) → 
  ∃ p1 p2 p3 p4 : ℝ × ℝ,
    (curve1 p1.1 p1.2 ∧ curve1 p2.1 p2.2 ∧ curve1 p3.1 p3.2 ∧ curve1 p4.1 p4.2) ∧
    (curve2 p1.1 p1.2 ∧ curve2 p2.1 p2.2 ∧ curve2 p3.1 p3.2 ∧ curve2 p4.1 p4.2) ∧ 
    (dist p1 p2 = dist p3 p4 ∧ dist p2 p3 = dist p4 p1) ∧ 
    (∃ m : ℝ, p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.1 ≠ m ∧ p2.1 ≠ m) := sorry

end NUMINAMATH_GPT_intersection_shape_is_rectangle_l18_1811


namespace NUMINAMATH_GPT_number_of_wickets_last_match_l18_1832

noncomputable def bowling_average : ℝ := 12.4
noncomputable def runs_taken_last_match : ℝ := 26
noncomputable def wickets_before_last_match : ℕ := 175
noncomputable def decrease_in_average : ℝ := 0.4
noncomputable def new_average : ℝ := bowling_average - decrease_in_average

theorem number_of_wickets_last_match (w : ℝ) :
  (175 + w) > 0 → 
  ((wickets_before_last_match * bowling_average + runs_taken_last_match) / (wickets_before_last_match + w) = new_average) →
  w = 8 := 
sorry

end NUMINAMATH_GPT_number_of_wickets_last_match_l18_1832


namespace NUMINAMATH_GPT_determine_a_l18_1814

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if h : x = 3 then a else 2 / |x - 3|

theorem determine_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ 3 ∧ x2 ≠ 3 ∧ (f x1 a - 4 = 0) ∧ (f x2 a - 4 = 0) ∧ f 3 a - 4 = 0) →
  a = 4 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l18_1814


namespace NUMINAMATH_GPT_intersection_P_Q_correct_l18_1802

-- Define sets P and Q based on given conditions
def is_in_P (x : ℝ) : Prop := x > 1
def is_in_Q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

-- Define the intersection P ∩ Q and the correct answer
def P_inter_Q (x : ℝ) : Prop := is_in_P x ∧ is_in_Q x
def correct_ans (x : ℝ) : Prop := 1 < x ∧ x ≤ 2

-- Prove that P ∩ Q = (1, 2]
theorem intersection_P_Q_correct : ∀ x : ℝ, P_inter_Q x ↔ correct_ans x :=
by sorry

end NUMINAMATH_GPT_intersection_P_Q_correct_l18_1802


namespace NUMINAMATH_GPT_wife_catch_up_l18_1829

/-- A man drives at a speed of 40 miles/hr.
His wife left 30 minutes late with a speed of 50 miles/hr.
Prove that they will meet 2 hours after the wife starts driving. -/
theorem wife_catch_up (t : ℝ) (speed_man speed_wife : ℝ) (late_time : ℝ) :
  speed_man = 40 →
  speed_wife = 50 →
  late_time = 0.5 →
  50 * t = 40 * (t + 0.5) →
  t = 2 :=
by
  intros h_man h_wife h_late h_eq
  -- Actual proof goes here. 
  -- (Skipping the proof as requested, leaving it as a placeholder)
  sorry

end NUMINAMATH_GPT_wife_catch_up_l18_1829


namespace NUMINAMATH_GPT_ellipse_focal_point_l18_1863

theorem ellipse_focal_point (m : ℝ) (m_pos : m > 0)
  (h : ∃ f : ℝ × ℝ, f = (1, 0) ∧ ∀ x y : ℝ, (x^2 / 4) + (y^2 / m^2) = 1 → 
    (x - 1)^2 + y^2 = (x^2 / 4) + (y^2 / m^2)) :
  m = Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_ellipse_focal_point_l18_1863


namespace NUMINAMATH_GPT_find_position_2002_l18_1809

def T (n : ℕ) : ℕ := n * (n + 1) / 2
def a (n : ℕ) : ℕ := T n + 1

theorem find_position_2002 : ∃ row col : ℕ, 1 ≤ row ∧ 1 ≤ col ∧ (a (row - 1) + (col - 1) = 2002 ∧ row = 15 ∧ col = 49) := 
sorry

end NUMINAMATH_GPT_find_position_2002_l18_1809


namespace NUMINAMATH_GPT_average_is_20_l18_1884

-- Define the numbers and the variable n
def a := 3
def b := 16
def c := 33
def n := 27
def d := n + 1

-- Define the sum of the numbers
def sum := a + b + c + d

-- Define the average as sum divided by 4
def average := sum / 4

-- Prove that the average is 20
theorem average_is_20 : average = 20 := by
  sorry

end NUMINAMATH_GPT_average_is_20_l18_1884


namespace NUMINAMATH_GPT_max_value_inequality_l18_1869

theorem max_value_inequality
  (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = ⅟2) :
  (|x1 + y1 - 1| / Real.sqrt 2) + (|x2 + y2 - 1| / Real.sqrt 2) ≤ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_value_inequality_l18_1869


namespace NUMINAMATH_GPT_coffee_vacation_days_l18_1896

theorem coffee_vacation_days 
  (pods_per_day : ℕ := 3)
  (pods_per_box : ℕ := 30)
  (box_cost : ℝ := 8.00)
  (total_spent : ℝ := 32) :
  (total_spent / box_cost) * pods_per_box / pods_per_day = 40 := 
by 
  sorry

end NUMINAMATH_GPT_coffee_vacation_days_l18_1896


namespace NUMINAMATH_GPT_ratio_of_cans_l18_1857

theorem ratio_of_cans (martha_cans : ℕ) (total_required : ℕ) (remaining_cans : ℕ) (diego_cans : ℕ) (ratio : ℚ) 
  (h1 : martha_cans = 90) 
  (h2 : total_required = 150) 
  (h3 : remaining_cans = 5) 
  (h4 : martha_cans + diego_cans = total_required - remaining_cans) 
  (h5 : ratio = (diego_cans : ℚ) / martha_cans) : 
  ratio = 11 / 18 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_cans_l18_1857


namespace NUMINAMATH_GPT_domain_of_sqrt_function_l18_1827

noncomputable def domain_of_function : Set ℝ :=
  {x : ℝ | 3 - 2 * x - x^2 ≥ 0}

theorem domain_of_sqrt_function : domain_of_function = {x : ℝ | -3 ≤ x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_sqrt_function_l18_1827


namespace NUMINAMATH_GPT_num_divisible_by_10_l18_1882

theorem num_divisible_by_10 (a b d : ℕ) (h1 : 100 ≤ a) (h2 : a ≤ 500) (h3 : 100 ≤ b) (h4 : b ≤ 500) (h5 : Nat.gcd d 10 = 10) :
  (b - a) / d + 1 = 41 := by
  sorry

end NUMINAMATH_GPT_num_divisible_by_10_l18_1882


namespace NUMINAMATH_GPT_coffee_bags_per_week_l18_1851

def bags_morning : Nat := 3
def bags_afternoon : Nat := 3 * bags_morning
def bags_evening : Nat := 2 * bags_morning
def bags_per_day : Nat := bags_morning + bags_afternoon + bags_evening
def days_per_week : Nat := 7

theorem coffee_bags_per_week : bags_per_day * days_per_week = 126 := by
  sorry

end NUMINAMATH_GPT_coffee_bags_per_week_l18_1851


namespace NUMINAMATH_GPT_lcm_of_8_12_15_l18_1875

theorem lcm_of_8_12_15 : Nat.lcm 8 (Nat.lcm 12 15) = 120 :=
by
  -- This is where the proof steps would go
  sorry

end NUMINAMATH_GPT_lcm_of_8_12_15_l18_1875


namespace NUMINAMATH_GPT_num_people_on_boats_l18_1806

-- Definitions based on the conditions
def boats := 5
def people_per_boat := 3

-- Theorem stating the problem to be solved
theorem num_people_on_boats : boats * people_per_boat = 15 :=
by sorry

end NUMINAMATH_GPT_num_people_on_boats_l18_1806


namespace NUMINAMATH_GPT_brick_surface_area_l18_1820

variable (X Y Z : ℝ)

#check 4 * X + 4 * Y + 2 * Z = 72 → 
       4 * X + 2 * Y + 4 * Z = 96 → 
       2 * X + 4 * Y + 4 * Z = 102 →
       2 * (X + Y + Z) = 54

theorem brick_surface_area (h1 : 4 * X + 4 * Y + 2 * Z = 72)
                           (h2 : 4 * X + 2 * Y + 4 * Z = 96)
                           (h3 : 2 * X + 4 * Y + 4 * Z = 102) :
                           2 * (X + Y + Z) = 54 := by
  sorry

end NUMINAMATH_GPT_brick_surface_area_l18_1820


namespace NUMINAMATH_GPT_total_amount_paid_l18_1843

-- Define the conditions of the problem
def cost_without_discount (quantity : ℕ) (unit_price : ℚ) : ℚ :=
  quantity * unit_price

def cost_with_discount (quantity : ℕ) (unit_price : ℚ) (discount_rate : ℚ) : ℚ :=
  let total_cost := cost_without_discount quantity unit_price
  total_cost - (total_cost * discount_rate)

-- Define each category's cost after discount
def pens_cost : ℚ := cost_with_discount 7 1.5 0.10
def notebooks_cost : ℚ := cost_without_discount 4 5
def water_bottles_cost : ℚ := cost_with_discount 2 8 0.30
def backpack_cost : ℚ := cost_with_discount 1 25 0.15
def socks_cost : ℚ := cost_with_discount 3 3 0.25

-- Prove the total amount paid is $68.65
theorem total_amount_paid : pens_cost + notebooks_cost + water_bottles_cost + backpack_cost + socks_cost = 68.65 := by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l18_1843


namespace NUMINAMATH_GPT_age_difference_two_children_l18_1852

/-!
# Age difference between two children in a family

## Given:
- 10 years ago, the average age of a family of 4 members was 24 years.
- Two children have been born since then.
- The present average age of the family (now 6 members) is the same, 24 years.
- The present age of the youngest child (Y1) is 3 years.

## Prove:
The age difference between the two children is 2 years.
-/

theorem age_difference_two_children :
  let Y1 := 3
  let Y2 := 5
  let total_age_10_years_ago := 4 * 24
  let total_age_now := 6 * 24
  let increase_age_10_years := total_age_now - total_age_10_years_ago
  let increase_due_to_original_members := 4 * 10
  let increase_due_to_children := increase_age_10_years - increase_due_to_original_members
  Y1 + Y2 = increase_due_to_children
  → Y2 - Y1 = 2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_age_difference_two_children_l18_1852


namespace NUMINAMATH_GPT_sqrt_expression_eq_three_l18_1825

theorem sqrt_expression_eq_three (h: (Real.sqrt 81) = 9) : Real.sqrt ((Real.sqrt 81 + Real.sqrt 81) / 2) = 3 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_expression_eq_three_l18_1825


namespace NUMINAMATH_GPT_circular_garden_area_l18_1887

theorem circular_garden_area (r : ℝ) (A C : ℝ) (h_radius : r = 6) (h_relationship : C = (1 / 3) * A) 
  (h_circumference : C = 2 * Real.pi * r) (h_area : A = Real.pi * r ^ 2) : 
  A = 36 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_circular_garden_area_l18_1887


namespace NUMINAMATH_GPT_jonah_poured_total_pitchers_l18_1817

theorem jonah_poured_total_pitchers :
  (0.25 + 0.125) + (0.16666666666666666 + 0.08333333333333333 + 0.16666666666666666) + 
  (0.25 + 0.125) + (0.3333333333333333 + 0.08333333333333333 + 0.16666666666666666) = 1.75 :=
by
  sorry

end NUMINAMATH_GPT_jonah_poured_total_pitchers_l18_1817


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l18_1836

theorem sufficient_but_not_necessary (a b : ℝ) (h : a * b ≠ 0) : 
  (¬ (a = 0)) ∧ ¬ ((a ≠ 0) → (a * b ≠ 0)) :=
by {
  -- The proof will be constructed here and is omitted as per the instructions
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_l18_1836


namespace NUMINAMATH_GPT_Jason_has_22_5_toys_l18_1834

noncomputable def RachelToys : ℝ := 1
noncomputable def JohnToys : ℝ := RachelToys + 6.5
noncomputable def JasonToys : ℝ := 3 * JohnToys

theorem Jason_has_22_5_toys : JasonToys = 22.5 := sorry

end NUMINAMATH_GPT_Jason_has_22_5_toys_l18_1834


namespace NUMINAMATH_GPT_find_a_l18_1838

def lambda : Set ℝ := { x | ∃ (a b : ℤ), x = a + b * Real.sqrt 3 }

theorem find_a (a : ℤ) (x : ℝ)
  (h1 : x = 7 + a * Real.sqrt 3)
  (h2 : x ∈ lambda)
  (h3 : (1 / x) ∈ lambda) :
  a = 4 ∨ a = -4 :=
sorry

end NUMINAMATH_GPT_find_a_l18_1838


namespace NUMINAMATH_GPT_evaluate_polynomial_at_neg_one_l18_1890

-- Define the polynomial f(x)
def f (x : ℝ) : ℝ := 1 + 2 * x + x^2 - 3 * x^3 + 2 * x^4

-- Define the value x at which we want to evaluate f
def x_val : ℝ := -1

-- State the theorem with the result using Horner's method
theorem evaluate_polynomial_at_neg_one : f x_val = 6 :=
by
  -- Approach to solution is in solution steps, skipped here
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_neg_one_l18_1890


namespace NUMINAMATH_GPT_divide_equal_parts_l18_1868

theorem divide_equal_parts (m n: ℕ) (h₁: (m + n) % 2 = 0) (h₂: gcd m n ∣ ((m + n) / 2)) : ∃ a b: ℕ, a = b ∧ a + b = m + n ∧ a ≤ m + n ∧ b ≤ m + n :=
sorry

end NUMINAMATH_GPT_divide_equal_parts_l18_1868


namespace NUMINAMATH_GPT_value_of_x_l18_1801

theorem value_of_x (n x : ℝ) (h1: x = 3 * n) (h2: 2 * n + 3 = 0.2 * 25) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l18_1801


namespace NUMINAMATH_GPT_fg_value_correct_l18_1844

def f_table (x : ℕ) : ℕ :=
  if x = 1 then 3
  else if x = 3 then 7
  else if x = 5 then 9
  else if x = 7 then 13
  else if x = 9 then 17
  else 0  -- Default value to handle unexpected inputs

def g_table (x : ℕ) : ℕ :=
  if x = 1 then 54
  else if x = 3 then 9
  else if x = 5 then 25
  else if x = 7 then 19
  else if x = 9 then 44
  else 0  -- Default value to handle unexpected inputs

theorem fg_value_correct : f_table (g_table 3) = 17 := 
by sorry

end NUMINAMATH_GPT_fg_value_correct_l18_1844


namespace NUMINAMATH_GPT_inequality_holds_l18_1850

theorem inequality_holds (x y : ℝ) (hx₀ : 0 < x) (hy₀ : 0 < y) (hxy : x + y = 1) :
  (1 / x^2 - 1) * (1 / y^2 - 1) ≥ 9 :=
sorry

end NUMINAMATH_GPT_inequality_holds_l18_1850


namespace NUMINAMATH_GPT_mass_percentage_H_in_chlorous_acid_l18_1860

noncomputable def mass_percentage_H_in_HClO2 : ℚ :=
  let molar_mass_H : ℚ := 1.01
  let molar_mass_Cl : ℚ := 35.45
  let molar_mass_O : ℚ := 16.00
  let molar_mass_HClO2 : ℚ := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  (molar_mass_H / molar_mass_HClO2) * 100

theorem mass_percentage_H_in_chlorous_acid :
  mass_percentage_H_in_HClO2 = 1.475 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_H_in_chlorous_acid_l18_1860


namespace NUMINAMATH_GPT_f_periodic_if_is_bounded_and_satisfies_fe_l18_1813

variable {f : ℝ → ℝ}

-- Condition 1: f is a bounded real function, i.e., it is bounded above and below
def is_bounded (f : ℝ → ℝ) : Prop := ∃ M, ∀ x, |f x| ≤ M

-- Condition 2: The functional equation given for all x.
def functional_eq (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 1/3) + f (x + 1/2) = f x + f (x + 5/6)

-- We need to show that f is periodic with period 1.
theorem f_periodic_if_is_bounded_and_satisfies_fe (h_bounded : is_bounded f) (h_fe : functional_eq f) : 
  ∀ x, f (x + 1) = f x :=
sorry

end NUMINAMATH_GPT_f_periodic_if_is_bounded_and_satisfies_fe_l18_1813


namespace NUMINAMATH_GPT_pure_imaginary_solution_l18_1804

theorem pure_imaginary_solution (m : ℝ) (h₁ : m^2 - m - 4 = 0) (h₂ : m^2 - 5 * m - 6 ≠ 0) :
  m = (1 + Real.sqrt 17) / 2 ∨ m = (1 - Real.sqrt 17) / 2 :=
sorry

end NUMINAMATH_GPT_pure_imaginary_solution_l18_1804


namespace NUMINAMATH_GPT_polynomial_sum_l18_1828

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3
def g (x : ℝ) : ℝ := -3 * x^2 + 7 * x - 6
def h (x : ℝ) : ℝ := 3 * x^2 - 3 * x + 2
def j (x : ℝ) : ℝ := x^2 + x - 1

theorem polynomial_sum (x : ℝ) : f x + g x + h x + j x = 3 * x^2 + x - 2 := by
  sorry

end NUMINAMATH_GPT_polynomial_sum_l18_1828


namespace NUMINAMATH_GPT_hours_per_batch_l18_1865

noncomputable section

def gallons_per_batch : ℕ := 3 / 2   -- 1.5 gallons expressed as a rational number
def ounces_per_gallon : ℕ := 128
def jack_consumption_per_2_days : ℕ := 96
def total_days : ℕ := 24
def time_spent_hours : ℕ := 120

def total_ounces : ℕ := gallons_per_batch * ounces_per_gallon
def total_ounces_consumed_24_days : ℕ := jack_consumption_per_2_days * (total_days / 2)
def number_of_batches : ℕ := total_ounces_consumed_24_days / total_ounces

theorem hours_per_batch :
  (time_spent_hours / number_of_batches) = 20 := by
  sorry

end NUMINAMATH_GPT_hours_per_batch_l18_1865


namespace NUMINAMATH_GPT_ab_bd_ratio_l18_1812

-- Definitions based on the conditions
variables {A B C D : ℝ}
variables (h1 : A / B = 1 / 2) (h2 : B / C = 8 / 5)

-- Math equivalence proving AB/BD = 4/13 based on given conditions
theorem ab_bd_ratio
  (h1 : A / B = 1 / 2)
  (h2 : B / C = 8 / 5) :
  A / (B + C) = 4 / 13 :=
by
  sorry

end NUMINAMATH_GPT_ab_bd_ratio_l18_1812


namespace NUMINAMATH_GPT_tiffany_ate_pies_l18_1883

theorem tiffany_ate_pies (baking_days : ℕ) (pies_per_day : ℕ) (wc_per_pie : ℕ) 
                         (remaining_wc : ℕ) (total_pies : ℕ) (total_wc : ℕ) :
  baking_days = 11 → pies_per_day = 3 → wc_per_pie = 2 → remaining_wc = 58 →
  total_pies = pies_per_day * baking_days → total_wc = total_pies * wc_per_pie →
  (total_wc - remaining_wc) / wc_per_pie = 4 :=
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_tiffany_ate_pies_l18_1883


namespace NUMINAMATH_GPT_verify_value_of_sum_l18_1874

noncomputable def value_of_sum (a b c d e f : ℕ) (values : Finset ℕ) : ℕ :=
if h : a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
        b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
        c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
        d ≠ e ∧ d ≠ f ∧
        e ≠ f ∧
        a + b = c ∧
        b + c = d ∧
        c + e = f
then a + c + f
else 0

theorem verify_value_of_sum :
  ∃ (a b c d e f : ℕ) (values : Finset ℕ),
  values = {4, 12, 15, 27, 31, 39} ∧
  a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ e ∈ values ∧ f ∈ values ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a + b = c ∧
  b + c = d ∧
  c + e = f ∧
  value_of_sum a b c d e f values = 73 :=
by
  sorry

end NUMINAMATH_GPT_verify_value_of_sum_l18_1874


namespace NUMINAMATH_GPT_greatest_possible_bent_strips_l18_1889

theorem greatest_possible_bent_strips (strip_count : ℕ) (cube_length cube_faces flat_strip_cover : ℕ) 
  (unit_squares_per_face total_squares flat_strips unit_squares_covered_by_flats : ℕ):
  strip_count = 18 →
  cube_length = 3 →
  cube_faces = 6 →
  flat_strip_cover = 3 →
  unit_squares_per_face = cube_length * cube_length →
  total_squares = cube_faces * unit_squares_per_face →
  flat_strips = 4 →
  unit_squares_covered_by_flats = flat_strips * flat_strip_cover →
  ∃ bent_strips,
  flat_strips * flat_strip_cover + bent_strips * flat_strip_cover = total_squares 
  ∧ bent_strips = 14 := by
  intros
  -- skipped proof
  sorry

end NUMINAMATH_GPT_greatest_possible_bent_strips_l18_1889


namespace NUMINAMATH_GPT_combined_mpg_l18_1845

theorem combined_mpg (m : ℕ) (ray_mpg tom_mpg : ℕ) (h1 : m = 200) (h2 : ray_mpg = 40) (h3 : tom_mpg = 20) :
  (m / (m / (2 * ray_mpg) + m / (2 * tom_mpg))) = 80 / 3 :=
by
  sorry

end NUMINAMATH_GPT_combined_mpg_l18_1845


namespace NUMINAMATH_GPT_area_correct_l18_1807

noncomputable def area_of_30_60_90_triangle (hypotenuse : ℝ) (angle : ℝ) : ℝ :=
if hypotenuse = 10 ∧ angle = 30 then 25 * Real.sqrt 3 / 2 else 0

theorem area_correct {hypotenuse angle : ℝ} (h1 : hypotenuse = 10) (h2 : angle = 30) :
  area_of_30_60_90_triangle hypotenuse angle = 25 * Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_correct_l18_1807


namespace NUMINAMATH_GPT_abs_b_leq_one_l18_1899

theorem abs_b_leq_one (a b : ℝ) (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (3 * x) ≤ 1) : |b| ≤ 1 := 
sorry

end NUMINAMATH_GPT_abs_b_leq_one_l18_1899


namespace NUMINAMATH_GPT_midpoint_of_hyperbola_segment_l18_1848

theorem midpoint_of_hyperbola_segment :
  (∃ A B : ℝ × ℝ, (A.1 ^ 2 - (A.2 ^ 2 / 9) = 1) ∧ (B.1 ^ 2 - (B.2 ^ 2 / 9) = 1)
    ∧ (A.1 + B.1) / 2 = -1 ∧ (A.2 + B.2) / 2 = -4) :=
by
  sorry

end NUMINAMATH_GPT_midpoint_of_hyperbola_segment_l18_1848


namespace NUMINAMATH_GPT_part1_part2_part3_l18_1879

-- Definitions for the conditions
def not_divisible_by_2_or_3 (k : ℤ) : Prop :=
  ¬(k % 2 = 0 ∨ k % 3 = 0)

def form_6n1_or_6n5 (k : ℤ) : Prop :=
  ∃ (n : ℤ), k = 6 * n + 1 ∨ k = 6 * n + 5

-- Part 1
theorem part1 (k : ℤ) (h : not_divisible_by_2_or_3 k) : form_6n1_or_6n5 k :=
sorry

-- Part 2
def form_6n1 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 1

def form_6n5 (a : ℤ) : Prop :=
  ∃ (n : ℤ), a = 6 * n + 5

theorem part2 (a b : ℤ) (ha : form_6n1 a ∨ form_6n5 a) (hb : form_6n1 b ∨ form_6n5 b) :
  form_6n1 (a * b) :=
sorry

-- Part 3
theorem part3 (a b : ℤ) (ha : form_6n1 a) (hb : form_6n5 b) :
  form_6n5 (a * b) :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l18_1879


namespace NUMINAMATH_GPT_ratio_area_ADE_BCED_is_8_over_9_l18_1839

noncomputable def ratio_area_ADE_BCED 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) : ℝ := 
  sorry

theorem ratio_area_ADE_BCED_is_8_over_9 
  (AB BC AC AD AE : ℝ)
  (hAB : AB = 30)
  (hBC : BC = 45)
  (hAC : AC = 54)
  (hAD : AD = 20)
  (hAE : AE = 24) :
  ratio_area_ADE_BCED AB BC AC AD AE hAB hBC hAC hAD hAE = 8 / 9 :=
  sorry

end NUMINAMATH_GPT_ratio_area_ADE_BCED_is_8_over_9_l18_1839


namespace NUMINAMATH_GPT_number_of_deleted_apps_l18_1842

def initial_apps := 16
def remaining_apps := 8

def deleted_apps : ℕ := initial_apps - remaining_apps

theorem number_of_deleted_apps : deleted_apps = 8 := 
by
  unfold deleted_apps initial_apps remaining_apps
  rfl

end NUMINAMATH_GPT_number_of_deleted_apps_l18_1842


namespace NUMINAMATH_GPT_velocity_at_2_l18_1894

variable (t : ℝ) (s : ℝ)

noncomputable def displacement (t : ℝ) : ℝ := t^2 + 3 / t

noncomputable def velocity (t : ℝ) : ℝ := (deriv displacement) t

theorem velocity_at_2 : velocity t = 2 * 2 - (3 / 4) := by
  sorry

end NUMINAMATH_GPT_velocity_at_2_l18_1894


namespace NUMINAMATH_GPT_find_extrema_l18_1866

-- Define the variables and the constraints
variables (x y z : ℝ)

-- Define the inequalities as conditions
def cond1 := -1 ≤ 2 * x + y - z ∧ 2 * x + y - z ≤ 8
def cond2 := 2 ≤ x - y + z ∧ x - y + z ≤ 9
def cond3 := -3 ≤ x + 2 * y - z ∧ x + 2 * y - z ≤ 7

-- Define the function f
def f (x y z : ℝ) := 7 * x + 5 * y - 2 * z

-- State the theorem that needs to be proved
theorem find_extrema :
  (∃ x y z, cond1 x y z ∧ cond2 x y z ∧ cond3 x y z) →
  (-6 ≤ f x y z ∧ f x y z ≤ 47) :=
by sorry

end NUMINAMATH_GPT_find_extrema_l18_1866


namespace NUMINAMATH_GPT_instantaneous_velocity_at_t_5_l18_1862

noncomputable def s (t : ℝ) : ℝ := (1/4) * t^4 - 3

theorem instantaneous_velocity_at_t_5 : 
  (deriv s 5) = 125 :=
by
  sorry

end NUMINAMATH_GPT_instantaneous_velocity_at_t_5_l18_1862


namespace NUMINAMATH_GPT_trigonometric_expression_value_l18_1818

noncomputable def trigonometric_expression (α : ℝ) : ℝ :=
  (|Real.tan α| / Real.tan α) + (Real.sin α / Real.sqrt ((1 - Real.cos (2 * α)) / 2))

theorem trigonometric_expression_value (α : ℝ) (h : Real.sin α = -Real.cos α) : 
  trigonometric_expression α = 0 ∨ trigonometric_expression α = -2 :=
by 
  sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l18_1818
