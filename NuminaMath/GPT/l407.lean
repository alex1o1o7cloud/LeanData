import Mathlib

namespace NUMINAMATH_GPT_cos_alpha_minus_7pi_over_2_l407_40724

-- Given conditions
variable (α : Real) (h : Real.sin α = 3/5)

-- Statement to prove
theorem cos_alpha_minus_7pi_over_2 : Real.cos (α - 7 * Real.pi / 2) = -3/5 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_7pi_over_2_l407_40724


namespace NUMINAMATH_GPT_minimum_a_l407_40774

noncomputable def func (t a : ℝ) := 5 * (t + 1) ^ 2 + a / (t + 1) ^ 5

theorem minimum_a (a : ℝ) (h: ∀ t ≥ 0, func t a ≥ 24) :
  a = 2 * Real.sqrt ((24 / 7) ^ 7) :=
sorry

end NUMINAMATH_GPT_minimum_a_l407_40774


namespace NUMINAMATH_GPT_range_of_given_function_l407_40751

noncomputable def given_function (x : ℝ) : ℝ :=
  abs (Real.sin x) / (Real.sin x) + Real.cos x / abs (Real.cos x) + abs (Real.tan x) / Real.tan x

theorem range_of_given_function : Set.range given_function = {-1, 3} :=
by
  sorry

end NUMINAMATH_GPT_range_of_given_function_l407_40751


namespace NUMINAMATH_GPT_roots_real_and_equal_l407_40736

theorem roots_real_and_equal (a b c : ℝ) (h_eq : a = 1) (h_b : b = -4 * Real.sqrt 2) (h_c : c = 8) :
  ∃ x : ℝ, (a * x^2 + b * x + c = 0) ∧ (b^2 - 4 * a * c = 0) :=
by
  have h_a : a = 1 := h_eq;
  have h_b : b = -4 * Real.sqrt 2 := h_b;
  have h_c : c = 8 := h_c;
  sorry

end NUMINAMATH_GPT_roots_real_and_equal_l407_40736


namespace NUMINAMATH_GPT_find_y_l407_40730

/-- Given (2 ^ x) - (2 ^ y) = 3 * (2 ^ 10) and x = 12, prove that y = 10 -/
theorem find_y (x y : ℕ) (h : (2 ^ x) - (2 ^ y) = 3 * (2 ^ 10)) (hx : x = 12) : y = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l407_40730


namespace NUMINAMATH_GPT_gcd_180_450_l407_40727

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end NUMINAMATH_GPT_gcd_180_450_l407_40727


namespace NUMINAMATH_GPT_inequality_pos_xy_l407_40719

theorem inequality_pos_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    (1 + x / y)^3 + (1 + y / x)^3 ≥ 16 := 
by {
    sorry
}

end NUMINAMATH_GPT_inequality_pos_xy_l407_40719


namespace NUMINAMATH_GPT_algae_coverage_day_21_l407_40768

-- Let "algae_coverage n" denote the percentage of lake covered by algae on day n.
noncomputable def algaeCoverage : ℕ → ℝ
| 0 => 1 -- initial state on day 0 taken as baseline (can be adjusted accordingly)
| (n+1) => 2 * algaeCoverage n

-- Define the problem statement
theorem algae_coverage_day_21 :
  algaeCoverage 24 = 100 → algaeCoverage 21 = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_algae_coverage_day_21_l407_40768


namespace NUMINAMATH_GPT_problem_l407_40787

theorem problem (a : ℝ) (h : a^2 - 5 * a - 1 = 0) : 3 * a^2 - 15 * a = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_l407_40787


namespace NUMINAMATH_GPT_value_of_g_at_x_minus_5_l407_40725

-- Definition of the function g
def g (x : ℝ) : ℝ := -3

-- The theorem we need to prove
theorem value_of_g_at_x_minus_5 (x : ℝ) : g (x - 5) = -3 := by
  sorry

end NUMINAMATH_GPT_value_of_g_at_x_minus_5_l407_40725


namespace NUMINAMATH_GPT_multiplication_vs_subtraction_difference_l407_40782

variable (x : ℕ)
variable (h : x = 10)

theorem multiplication_vs_subtraction_difference :
  3 * x - (26 - x) = 14 := by
  sorry

end NUMINAMATH_GPT_multiplication_vs_subtraction_difference_l407_40782


namespace NUMINAMATH_GPT_garden_area_l407_40789

/-- A rectangular garden is 350 cm long and 50 cm wide. Determine its area in square meters. -/
theorem garden_area (length_cm width_cm : ℝ) (h_length : length_cm = 350) (h_width : width_cm = 50) : (length_cm / 100) * (width_cm / 100) = 1.75 :=
by
  sorry

end NUMINAMATH_GPT_garden_area_l407_40789


namespace NUMINAMATH_GPT_smallest_N_value_proof_l407_40760

def smallest_value_N (N : ℕ) : Prop :=
  N > 70 ∧ (21 * N) % 70 = 0

theorem smallest_N_value_proof : ∃ N, smallest_value_N N ∧ (∀ M, smallest_value_N M → N ≤ M) :=
  sorry

end NUMINAMATH_GPT_smallest_N_value_proof_l407_40760


namespace NUMINAMATH_GPT_problem_equiv_l407_40763

variable (a b c d e f : ℝ)

theorem problem_equiv :
  a * b * c = 65 → 
  b * c * d = 65 → 
  c * d * e = 1000 → 
  d * e * f = 250 → 
  (a * f) / (c * d) = 1 / 4 := 
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_problem_equiv_l407_40763


namespace NUMINAMATH_GPT_subway_train_speed_l407_40772

theorem subway_train_speed (s : ℕ) (h1 : 0 ≤ s ∧ s ≤ 7) (h2 : s^2 + 2*s = 63) : s = 7 :=
by
  sorry

end NUMINAMATH_GPT_subway_train_speed_l407_40772


namespace NUMINAMATH_GPT_range_of_a_l407_40765

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x| ≥ a * x) ↔ -1 ≤ a ∧ a ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l407_40765


namespace NUMINAMATH_GPT_base_h_addition_eq_l407_40785

theorem base_h_addition_eq (h : ℕ) :
  let n1 := 7 * h^3 + 3 * h^2 + 6 * h + 4
  let n2 := 8 * h^3 + 4 * h^2 + 2 * h + 1
  let sum := 1 * h^4 + 7 * h^3 + 2 * h^2 + 8 * h + 5
  n1 + n2 = sum → h = 8 :=
by
  intros n1 n2 sum h_eq
  sorry

end NUMINAMATH_GPT_base_h_addition_eq_l407_40785


namespace NUMINAMATH_GPT_correct_average_weight_l407_40767

theorem correct_average_weight (n : ℕ) (incorrect_avg_weight : ℝ) (initial_avg_weight : ℝ)
  (misread_weight correct_weight : ℝ) (boys_count : ℕ) :
  incorrect_avg_weight = 58.4 →
  n = 20 →
  misread_weight = 56 →
  correct_weight = 65 →
  boys_count = n →
  initial_avg_weight = (incorrect_avg_weight * n + (correct_weight - misread_weight)) / boys_count →
  initial_avg_weight = 58.85 :=
by
  intro h1 h2 h3 h4 h5 h_avg
  sorry

end NUMINAMATH_GPT_correct_average_weight_l407_40767


namespace NUMINAMATH_GPT_all_points_equal_l407_40766

-- Define the problem conditions and variables
variable (P : Type) -- points in the plane
variable [MetricSpace P] -- the plane is a metric space
variable (f : P → ℝ) -- assignment of numbers to points
variable (incenter : P → P → P → P) -- calculates incenter of a nondegenerate triangle

-- Condition: the value at the incenter of a triangle is the arithmetic mean of the values at the vertices
axiom incenter_mean_property : ∀ (A B C : P), 
  (A ≠ B) → (B ≠ C) → (A ≠ C) →
  f (incenter A B C) = (f A + f B + f C) / 3

-- The theorem to be proved
theorem all_points_equal : ∀ x y : P, f x = f y :=
by
  sorry

end NUMINAMATH_GPT_all_points_equal_l407_40766


namespace NUMINAMATH_GPT_function_evaluation_l407_40737

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x - 1) = x^2 - 1) : ∀ x : ℝ, f x = x^2 + 2 * x :=
by
  sorry

end NUMINAMATH_GPT_function_evaluation_l407_40737


namespace NUMINAMATH_GPT_thor_jumps_to_exceed_29000_l407_40731

theorem thor_jumps_to_exceed_29000 :
  ∃ (n : ℕ), (3 ^ n) > 29000 ∧ n = 10 := sorry

end NUMINAMATH_GPT_thor_jumps_to_exceed_29000_l407_40731


namespace NUMINAMATH_GPT_jessica_routes_count_l407_40780

def line := Type

def valid_route_count (p q r s t u : line) : ℕ := 9 + 36 + 36

theorem jessica_routes_count (p q r s t u : line) :
  valid_route_count p q r s t u = 81 :=
by
  sorry

end NUMINAMATH_GPT_jessica_routes_count_l407_40780


namespace NUMINAMATH_GPT_rectangular_solid_diagonal_l407_40795

theorem rectangular_solid_diagonal (p q r : ℝ) (d : ℝ) :
  p^2 + q^2 + r^2 = d^2 :=
sorry

end NUMINAMATH_GPT_rectangular_solid_diagonal_l407_40795


namespace NUMINAMATH_GPT_problem_divisible_by_1946_l407_40703

def F (n : ℕ) : ℤ := 1492 ^ n - 1770 ^ n - 1863 ^ n + 2141 ^ n

theorem problem_divisible_by_1946 
  (n : ℕ) 
  (hn : n ≤ 1945) : 
  1946 ∣ F n :=
sorry

end NUMINAMATH_GPT_problem_divisible_by_1946_l407_40703


namespace NUMINAMATH_GPT_work_completion_time_l407_40717

variable (p q : Type)

def efficient (p q : Type) : Prop :=
  ∃ (Wp Wq : ℝ), Wp = 1.5 * Wq ∧ Wp = 1 / 25

def work_done_together (p q : Type) := 1/15

theorem work_completion_time {p q : Type} (h1 : efficient p q) :
  ∃ d : ℝ, d = 15 :=
  sorry

end NUMINAMATH_GPT_work_completion_time_l407_40717


namespace NUMINAMATH_GPT_problem_l407_40788

noncomputable def y := 2 + Real.sqrt 3

theorem problem (c d : ℤ) (hc : c > 0) (hd : d > 0) (h : y = c + Real.sqrt d)
  (hy_eq : y^2 + 2*y + 2/y + 1/y^2 = 20) : c + d = 5 :=
  sorry

end NUMINAMATH_GPT_problem_l407_40788


namespace NUMINAMATH_GPT_total_worth_of_stock_l407_40707

theorem total_worth_of_stock (W : ℝ) 
    (h1 : 0.2 * W * 0.1 = 0.02 * W)
    (h2 : 0.6 * (0.8 * W) * 0.05 = 0.024 * W)
    (h3 : 0.2 * (0.8 * W) = 0.16 * W)
    (h4 : (0.024 * W) - (0.02 * W) = 400) 
    : W = 100000 := 
sorry

end NUMINAMATH_GPT_total_worth_of_stock_l407_40707


namespace NUMINAMATH_GPT_parabola_distance_l407_40775

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end NUMINAMATH_GPT_parabola_distance_l407_40775


namespace NUMINAMATH_GPT_scientists_arrival_probability_l407_40797

open Real

theorem scientists_arrival_probability (x y z : ℕ) (n : ℝ) (h : z ≠ 0)
  (hz : ¬ ∃ p : ℕ, Nat.Prime p ∧ p ^ 2 ∣ z)
  (h1 : n = x - y * sqrt z)
  (h2 : ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 120 ∧ 0 ≤ b ∧ b ≤ 120 ∧
    |a - b| ≤ n)
  (h3 : (120 - n)^2 / (120 ^ 2) = 0.7) :
  x + y + z = 202 := sorry

end NUMINAMATH_GPT_scientists_arrival_probability_l407_40797


namespace NUMINAMATH_GPT_average_children_in_families_with_children_l407_40776

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end NUMINAMATH_GPT_average_children_in_families_with_children_l407_40776


namespace NUMINAMATH_GPT_triangle_ineq_l407_40704

theorem triangle_ineq (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 2 * (a^2 + b^2) > c^2 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_ineq_l407_40704


namespace NUMINAMATH_GPT_find_smaller_number_l407_40783

variable (x y : ℕ)

theorem find_smaller_number (h1 : ∃ k : ℕ, x = 2 * k ∧ y = 5 * k) (h2 : x + y = 21) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_smaller_number_l407_40783


namespace NUMINAMATH_GPT_geom_seq_solution_l407_40734

theorem geom_seq_solution (a b x y : ℝ) 
  (h1 : x * (1 + y + y^2) = a) 
  (h2 : x^2 * (1 + y^2 + y^4) = b) :
  x = 1 / (4 * a) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨ 
  x = 1 / (4 * a) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∧
  y = 1 / (2 * (a^2 - b)) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨
  y = 1 / (2 * (a^2 - b)) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) := 
  sorry

end NUMINAMATH_GPT_geom_seq_solution_l407_40734


namespace NUMINAMATH_GPT_min_AP_squared_sum_value_l407_40781

-- Definitions based on given problem conditions
def A : ℝ := 0
def B : ℝ := 2
def C : ℝ := 4
def D : ℝ := 7
def E : ℝ := 15

def distance_squared (x y : ℝ) : ℝ := (x - y)^2

noncomputable def min_AP_squared_sum (r : ℝ) : ℝ :=
  r^2 + distance_squared r B + distance_squared r C + distance_squared r D + distance_squared r E

theorem min_AP_squared_sum_value : ∃ (r : ℝ), (min_AP_squared_sum r) = 137.2 :=
by
  existsi 5.6
  sorry

end NUMINAMATH_GPT_min_AP_squared_sum_value_l407_40781


namespace NUMINAMATH_GPT_three_digit_number_multiple_of_eleven_l407_40756

theorem three_digit_number_multiple_of_eleven:
  ∃ (a b c : ℕ), (1 ≤ a) ∧ (a ≤ 9) ∧ (0 ≤ b) ∧ (b ≤ 9) ∧ (0 ≤ c) ∧ (c ≤ 9) ∧
                  (100 * a + 10 * b + c = 11 * (a + b + c) ∧ (100 * a + 10 * b + c = 198)) :=
by
  use 1
  use 9
  use 8
  sorry

end NUMINAMATH_GPT_three_digit_number_multiple_of_eleven_l407_40756


namespace NUMINAMATH_GPT_greatest_divisor_of_product_of_four_consecutive_integers_l407_40759

theorem greatest_divisor_of_product_of_four_consecutive_integers :
  ∀ n : Nat, 24 ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  intro n
  sorry

end NUMINAMATH_GPT_greatest_divisor_of_product_of_four_consecutive_integers_l407_40759


namespace NUMINAMATH_GPT_area_of_field_with_tomatoes_l407_40777

theorem area_of_field_with_tomatoes :
  let length := 3.6
  let width := 2.5 * length
  let total_area := length * width
  let area_with_tomatoes := total_area / 2
  area_with_tomatoes = 16.2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_field_with_tomatoes_l407_40777


namespace NUMINAMATH_GPT_roots_cubed_l407_40779

noncomputable def q (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * x + b^2 - c^2
noncomputable def p (b c : ℝ) (x : ℝ) : ℝ := x^2 - 2 * b * (b^2 + 3 * c^2) * x + (b^2 - c^2)^3 
def x1 (b c : ℝ) := b + c
def x2 (b c : ℝ) := b - c

theorem roots_cubed (b c : ℝ) :
  (q b c (x1 b c) = 0 ∧ q b c (x2 b c) = 0) →
  (p b c ((x1 b c)^3) = 0 ∧ p b c ((x2 b c)^3) = 0) :=
by
  sorry

end NUMINAMATH_GPT_roots_cubed_l407_40779


namespace NUMINAMATH_GPT_John_l407_40723

theorem John's_earnings_on_Saturday :
  ∃ S : ℝ, (S + S / 2 + 20 = 47) ∧ (S = 18) := by
    sorry

end NUMINAMATH_GPT_John_l407_40723


namespace NUMINAMATH_GPT_uncle_bob_can_park_l407_40715

-- Define the conditions
def total_spaces : Nat := 18
def cars : Nat := 15
def rv_spaces : Nat := 3

-- Define a function to calculate the probability (without implementation)
noncomputable def probability_RV_can_park (total_spaces cars rv_spaces : Nat) : Rat :=
  if h : rv_spaces <= total_spaces - cars then
    -- The probability calculation logic would go here
    16 / 51
  else
    0

-- The theorem stating the desired result
theorem uncle_bob_can_park : probability_RV_can_park total_spaces cars rv_spaces = 16 / 51 :=
  sorry

end NUMINAMATH_GPT_uncle_bob_can_park_l407_40715


namespace NUMINAMATH_GPT_car_drive_time_60_kmh_l407_40758

theorem car_drive_time_60_kmh
  (t : ℝ)
  (avg_speed : ℝ := 80)
  (dist_speed_60 : ℝ := 60 * t)
  (time_speed_90 : ℝ := 2 / 3)
  (dist_speed_90 : ℝ := 90 * time_speed_90)
  (total_distance : ℝ := dist_speed_60 + dist_speed_90)
  (total_time : ℝ := t + time_speed_90)
  (avg_speed_eq : avg_speed = total_distance / total_time) :
  t = 1 / 3 := 
sorry

end NUMINAMATH_GPT_car_drive_time_60_kmh_l407_40758


namespace NUMINAMATH_GPT_function_zero_solution_l407_40792

-- Define the statement of the problem
theorem function_zero_solution (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → ∀ y : ℝ, f (x ^ 2 + y) ≥ (1 / x + 1) * f y) →
  (∀ x : ℝ, f x = 0) :=
by
  -- The proof of this theorem will be inserted here.
  sorry

end NUMINAMATH_GPT_function_zero_solution_l407_40792


namespace NUMINAMATH_GPT_width_of_foil_covered_prism_l407_40712

noncomputable def foil_covered_prism_width : ℕ :=
  let (l, w, h) := (4, 8, 4)
  let inner_width := 2 * l
  let increased_width := w + 2
  increased_width

theorem width_of_foil_covered_prism : foil_covered_prism_width = 10 := 
by
  let l := 4
  let w := 2 * l
  let h := w / 2
  have volume : l * w * h = 128 := by
    sorry
  have width_foil_covered := w + 2
  have : foil_covered_prism_width = width_foil_covered := by
    sorry
  sorry

end NUMINAMATH_GPT_width_of_foil_covered_prism_l407_40712


namespace NUMINAMATH_GPT_binomial_divisible_by_prime_l407_40743

theorem binomial_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  (Nat.choose n p) - (n / p) % p = 0 := 
sorry

end NUMINAMATH_GPT_binomial_divisible_by_prime_l407_40743


namespace NUMINAMATH_GPT_percentage_decrease_l407_40778

theorem percentage_decrease (x : ℝ) (h : x > 0) : ∃ p : ℝ, p = 0.20 ∧ ((1.25 * x) * (1 - p) = x) :=
by
  sorry

end NUMINAMATH_GPT_percentage_decrease_l407_40778


namespace NUMINAMATH_GPT_solve_x_from_operation_l407_40794

def operation (a b c d : ℝ) : ℝ := a * c + b * d

theorem solve_x_from_operation :
  ∀ x : ℝ, operation (2 * x) 3 3 (-1) = 3 → x = 1 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_solve_x_from_operation_l407_40794


namespace NUMINAMATH_GPT_range_of_a_l407_40753

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) → (0 < a ∧ a < 1) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l407_40753


namespace NUMINAMATH_GPT_Q_subset_P_l407_40752

def P : Set ℝ := {x | x < 2}
def Q : Set ℝ := {y | y < 1}

theorem Q_subset_P : Q ⊆ P := by
  sorry

end NUMINAMATH_GPT_Q_subset_P_l407_40752


namespace NUMINAMATH_GPT_possible_length_of_third_side_l407_40796

theorem possible_length_of_third_side (a b c : ℤ) (h1 : a - b = 7) (h2 : (a + b + c) % 2 = 1) : c = 8 :=
sorry

end NUMINAMATH_GPT_possible_length_of_third_side_l407_40796


namespace NUMINAMATH_GPT_students_and_confucius_same_arrival_time_l407_40746

noncomputable def speed_of_students_walking (x : ℝ) : ℝ := x

noncomputable def speed_of_bullock_cart (x : ℝ) : ℝ := 1.5 * x

noncomputable def time_for_students_to_school (x : ℝ) : ℝ := 30 / x

noncomputable def time_for_confucius_to_school (x : ℝ) : ℝ := 30 / (1.5 * x) + 1

theorem students_and_confucius_same_arrival_time (x : ℝ) (h1 : 0 < x) :
  30 / x = 30 / (1.5 * x) + 1 :=
by
  sorry

end NUMINAMATH_GPT_students_and_confucius_same_arrival_time_l407_40746


namespace NUMINAMATH_GPT_my_age_now_l407_40708

theorem my_age_now (Y S : ℕ) (h1 : Y - 9 = 5 * (S - 9)) (h2 : Y = 3 * S) : Y = 54 := by
  sorry

end NUMINAMATH_GPT_my_age_now_l407_40708


namespace NUMINAMATH_GPT_longer_diagonal_eq_l407_40773

variable (a b : ℝ)
variable (h_cd : CD = a) (h_bc : BC = b) (h_diag : AC = a) (h_ad : AD = 2 * b)

theorem longer_diagonal_eq (CD BC AC AD BD : ℝ) (h_cd : CD = a)
  (h_bc : BC = b) (h_diag : AC = CD) (h_ad : AD = 2 * b) :
  BD = Real.sqrt (a^2 + 3 * b^2) :=
sorry

end NUMINAMATH_GPT_longer_diagonal_eq_l407_40773


namespace NUMINAMATH_GPT_price_of_most_expensive_book_l407_40769

-- Define the conditions
def number_of_books := 41
def price_increment := 3

-- Define the price of the n-th book as a function of the price of the first book
def price (c : ℕ) (n : ℕ) : ℕ := c + price_increment * (n - 1)

-- Define a theorem stating the result
theorem price_of_most_expensive_book (c : ℕ) :
  c = 30 → price c number_of_books = 150 :=
by {
  sorry
}

end NUMINAMATH_GPT_price_of_most_expensive_book_l407_40769


namespace NUMINAMATH_GPT_computation_problem_points_l407_40709

def num_problems : ℕ := 30
def points_per_word_problem : ℕ := 5
def total_points : ℕ := 110
def num_computation_problems : ℕ := 20

def points_per_computation_problem : ℕ := 3

theorem computation_problem_points :
  ∃ x : ℕ, (num_computation_problems * x + (num_problems - num_computation_problems) * points_per_word_problem = total_points) ∧ x = points_per_computation_problem :=
by
  use points_per_computation_problem
  simp
  sorry

end NUMINAMATH_GPT_computation_problem_points_l407_40709


namespace NUMINAMATH_GPT_ribbon_per_gift_l407_40755

theorem ribbon_per_gift
  (total_ribbon : ℕ)
  (number_of_gifts : ℕ)
  (ribbon_left : ℕ)
  (used_ribbon := total_ribbon - ribbon_left)
  (ribbon_per_gift := used_ribbon / number_of_gifts)
  (h_total : total_ribbon = 18)
  (h_gifts : number_of_gifts = 6)
  (h_left : ribbon_left = 6) :
  ribbon_per_gift = 2 := by
  sorry

end NUMINAMATH_GPT_ribbon_per_gift_l407_40755


namespace NUMINAMATH_GPT_solve_y_equation_l407_40728

theorem solve_y_equation :
  ∃ y : ℚ, 4 * (5 * y + 3) - 3 = -3 * (2 - 8 * y) ∧ y = 15 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_y_equation_l407_40728


namespace NUMINAMATH_GPT_price_change_l407_40705

theorem price_change (P : ℝ) : 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  P4 = P * 0.9216 := 
by 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  show P4 = P * 0.9216
  sorry

end NUMINAMATH_GPT_price_change_l407_40705


namespace NUMINAMATH_GPT_powers_of_two_div7_l407_40750

theorem powers_of_two_div7 (n : ℕ) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, n = 3 * k := sorry

end NUMINAMATH_GPT_powers_of_two_div7_l407_40750


namespace NUMINAMATH_GPT_scientific_notation_of_42_trillion_l407_40744

theorem scientific_notation_of_42_trillion : (42.1 * 10^12) = 4.21 * 10^13 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_42_trillion_l407_40744


namespace NUMINAMATH_GPT_neg_universal_to_existential_l407_40714

theorem neg_universal_to_existential :
  (¬ (∀ x : ℝ, 2 * x^4 - x^2 + 1 < 0)) ↔ (∃ x : ℝ, 2 * x^4 - x^2 + 1 ≥ 0) :=
by 
  sorry

end NUMINAMATH_GPT_neg_universal_to_existential_l407_40714


namespace NUMINAMATH_GPT_tariffs_impact_but_no_timeframe_l407_40757

noncomputable def cost_of_wine_today : ℝ := 20.00
noncomputable def increase_percentage : ℝ := 0.25
noncomputable def bottles_count : ℕ := 5
noncomputable def price_increase_for_bottles : ℝ := 25.00

theorem tariffs_impact_but_no_timeframe :
  ¬ ∃ (t : ℝ), (cost_of_wine_today * (1 + increase_percentage) - cost_of_wine_today) * bottles_count = price_increase_for_bottles →
  (t = sorry) :=
by 
  sorry

end NUMINAMATH_GPT_tariffs_impact_but_no_timeframe_l407_40757


namespace NUMINAMATH_GPT_vector_operation_result_l407_40742

-- Definitions of vectors a and b
def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (2, -3)

-- The operation 2a - b
def operation (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 - b.1, 2 * a.2 - b.2)

-- The theorem stating the result of the operation
theorem vector_operation_result : operation a b = (-4, 5) :=
by
  sorry

end NUMINAMATH_GPT_vector_operation_result_l407_40742


namespace NUMINAMATH_GPT_probability_one_instrument_l407_40722

-- Definitions based on conditions
def total_people : Nat := 800
def play_at_least_one : Nat := total_people / 5
def play_two_or_more : Nat := 32
def play_exactly_one : Nat := play_at_least_one - play_two_or_more

-- Target statement to prove the equivalence
theorem probability_one_instrument: (play_exactly_one : ℝ) / (total_people : ℝ) = 0.16 := by
  sorry

end NUMINAMATH_GPT_probability_one_instrument_l407_40722


namespace NUMINAMATH_GPT_Tim_age_l407_40716

theorem Tim_age : ∃ (T : ℕ), (T = (3 * T + 2 - 12)) ∧ (T = 5) :=
by
  existsi 5
  sorry

end NUMINAMATH_GPT_Tim_age_l407_40716


namespace NUMINAMATH_GPT_min_value_polynomial_l407_40749

theorem min_value_polynomial (a b : ℝ) : 
  ∃ c, (∀ a b, c ≤ a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) ∧
       (∀ a b, a = -1 ∧ b = -1 → c = a^2 + 2 * b^2 + 2 * a + 4 * b + 2008) :=
sorry

end NUMINAMATH_GPT_min_value_polynomial_l407_40749


namespace NUMINAMATH_GPT_meeting_point_l407_40733

theorem meeting_point :
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  (Paul_start.1 + Lisa_start.1) / 2 = -2 ∧ (Paul_start.2 + Lisa_start.2) / 2 = 3 :=
by
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  have x_coord : (Paul_start.1 + Lisa_start.1) / 2 = -2 := sorry
  have y_coord : (Paul_start.2 + Lisa_start.2) / 2 = 3 := sorry
  exact ⟨x_coord, y_coord⟩

end NUMINAMATH_GPT_meeting_point_l407_40733


namespace NUMINAMATH_GPT_square_difference_example_l407_40762

theorem square_difference_example : 601^2 - 599^2 = 2400 := 
by sorry

end NUMINAMATH_GPT_square_difference_example_l407_40762


namespace NUMINAMATH_GPT_leaves_decrease_by_four_fold_l407_40790

theorem leaves_decrease_by_four_fold (x y : ℝ) (h1 : y ≤ x / 4) : 
  9 * y ≤ (9 * x) / 4 := by 
  sorry

end NUMINAMATH_GPT_leaves_decrease_by_four_fold_l407_40790


namespace NUMINAMATH_GPT_parabola_translation_l407_40701

theorem parabola_translation :
  ∀ (x : ℝ),
  (∃ x' y' : ℝ, x' = x - 1 ∧ y' = 2 * x' ^ 2 - 3 ∧ y = y' + 3) →
  (y = 2 * x ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_translation_l407_40701


namespace NUMINAMATH_GPT_factor_polynomial_l407_40738

theorem factor_polynomial (x : ℝ) :
  (x^2 + 4*x + 3) * (x^2 + 8*x + 15) + (x^2 + 6*x - 8) = 
  (x^2 + 6*x + 4) * (x^2 + 6*x + 11) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l407_40738


namespace NUMINAMATH_GPT_david_spent_difference_l407_40726

-- Define the initial amount, remaining amount, amount spent and the correct answer
def initial_amount : Real := 1800
def remaining_amount : Real := 500
def spent_amount : Real := initial_amount - remaining_amount
def correct_difference : Real := spent_amount - remaining_amount

-- Prove that the difference between the amount spent and the remaining amount is $800
theorem david_spent_difference : correct_difference = 800 := by
  sorry

end NUMINAMATH_GPT_david_spent_difference_l407_40726


namespace NUMINAMATH_GPT_jimmy_fill_bucket_time_l407_40761

-- Definitions based on conditions
def pool_volume : ℕ := 84
def bucket_volume : ℕ := 2
def total_time_minutes : ℕ := 14
def total_time_seconds : ℕ := total_time_minutes * 60
def trips : ℕ := pool_volume / bucket_volume

-- Theorem statement
theorem jimmy_fill_bucket_time : (total_time_seconds / trips) = 20 := by
  sorry

end NUMINAMATH_GPT_jimmy_fill_bucket_time_l407_40761


namespace NUMINAMATH_GPT_solution_exists_l407_40711

def age_problem (S F Y : ℕ) : Prop :=
  S = 12 ∧ S = F / 3 ∧ S - Y = (F - Y) / 5 ∧ Y = 6

theorem solution_exists : ∃ (Y : ℕ), ∃ (S F : ℕ), age_problem S F Y :=
by sorry

end NUMINAMATH_GPT_solution_exists_l407_40711


namespace NUMINAMATH_GPT_profit_percent_300_l407_40770

theorem profit_percent_300 (SP : ℝ) (CP : ℝ) (h : CP = 0.25 * SP) : ((SP - CP) / CP) * 100 = 300 :=
by
  sorry

end NUMINAMATH_GPT_profit_percent_300_l407_40770


namespace NUMINAMATH_GPT_calculation_simplifies_l407_40718

theorem calculation_simplifies :
  120 * (120 - 12) - (120 * 120 - 12) = -1428 := by
  sorry

end NUMINAMATH_GPT_calculation_simplifies_l407_40718


namespace NUMINAMATH_GPT_find_q_l407_40745

variable (x : ℝ)

def f (x : ℝ) := (5 * x^4 + 15 * x^3 + 30 * x^2 + 10 * x + 10)
def g (x : ℝ) := (2 * x^6 + 4 * x^4 + 10 * x^2)
def q (x : ℝ) := (-2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)

theorem find_q :
  (∀ x, q x + g x = f x) ↔ (∀ x, q x = -2 * x^6 + x^4 + 15 * x^3 + 20 * x^2 + 10 * x + 10)
:= sorry

end NUMINAMATH_GPT_find_q_l407_40745


namespace NUMINAMATH_GPT_range_of_m_satisfying_obtuse_triangle_l407_40710

theorem range_of_m_satisfying_obtuse_triangle (m : ℝ) 
(h_triangle: m > 0 
  → m + (m + 1) > (m + 2) 
  ∧ m + (m + 2) > (m + 1) 
  ∧ (m + 1) + (m + 2) > m
  ∧ (m + 2) ^ 2 > m ^ 2 + (m + 1) ^ 2) : 1 < m ∧ m < 1.5 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_satisfying_obtuse_triangle_l407_40710


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l407_40735

noncomputable def a : ℝ := 1 / 3
noncomputable def b : ℝ := Real.sin (1 / 3)
noncomputable def c : ℝ := 1 / Real.pi

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l407_40735


namespace NUMINAMATH_GPT_orangeade_ratio_l407_40739

theorem orangeade_ratio (O W : ℝ) (price1 price2 : ℝ) (revenue1 revenue2 : ℝ)
  (h1 : price1 = 0.30) (h2 : price2 = 0.20)
  (h3 : revenue1 = revenue2)
  (glasses1 glasses2 : ℝ)
  (V : ℝ) :
  glasses1 = (O + W) / V → glasses2 = (O + 2 * W) / V →
  revenue1 = glasses1 * price1 → revenue2 = glasses2 * price2 →
  (O + W) * price1 = (O + 2 * W) * price2 → O / W = 1 :=
by sorry

end NUMINAMATH_GPT_orangeade_ratio_l407_40739


namespace NUMINAMATH_GPT_digit_five_occurrences_l407_40702

variable (fives_ones fives_tens fives_hundreds : ℕ)

def count_fives := fives_ones + fives_tens + fives_hundreds

theorem digit_five_occurrences :
  ( ∀ (fives_ones fives_tens fives_hundreds : ℕ), 
    fives_ones = 100 ∧ fives_tens = 100 ∧ fives_hundreds = 100 → 
    count_fives fives_ones fives_tens fives_hundreds = 300 ) :=
by
  sorry

end NUMINAMATH_GPT_digit_five_occurrences_l407_40702


namespace NUMINAMATH_GPT_cube_faces_sum_l407_40720

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : a = 12) (h2 : b = 13) (h3 : c = 14)
  (h4 : d = 15) (h5 : e = 16) (h6 : f = 17)
  (h_pairs : a + f = b + e ∧ b + e = c + d) :
  a + b + c + d + e + f = 87 := by
  sorry

end NUMINAMATH_GPT_cube_faces_sum_l407_40720


namespace NUMINAMATH_GPT_range_of_a_l407_40793

variable {a : ℝ}

-- Proposition p: The solution set of the inequality x^2 - (a+1)x + 1 ≤ 0 is empty
def prop_p (a : ℝ) : Prop := (a + 1) ^ 2 - 4 < 0 

-- Proposition q: The function f(x) = (a+1)^x is increasing within its domain
def prop_q (a : ℝ) : Prop := a > 0 

-- The combined conditions
def combined_conditions (a : ℝ) : Prop := (prop_p a) ∨ (prop_q a) ∧ ¬(prop_p a ∧ prop_q a)

-- The range of values for a
theorem range_of_a (h : combined_conditions a) : -3 < a ∧ a ≤ 0 ∨ a ≥ 1 :=
  sorry

end NUMINAMATH_GPT_range_of_a_l407_40793


namespace NUMINAMATH_GPT_smallest_total_cells_marked_l407_40729

-- Definitions based on problem conditions
def grid_height : ℕ := 8
def grid_width : ℕ := 13

def squares_per_height : ℕ := grid_height / 2
def squares_per_width : ℕ := grid_width / 2

def initial_marked_cells_per_square : ℕ := 1
def additional_marked_cells_per_square : ℕ := 1

def number_of_squares : ℕ := squares_per_height * squares_per_width
def initial_marked_cells : ℕ := number_of_squares * initial_marked_cells_per_square
def additional_marked_cells : ℕ := number_of_squares * additional_marked_cells_per_square

def total_marked_cells : ℕ := initial_marked_cells + additional_marked_cells

-- Statement of the proof problem
theorem smallest_total_cells_marked : total_marked_cells = 48 := by 
    -- Proof is not required as per the instruction
    sorry

end NUMINAMATH_GPT_smallest_total_cells_marked_l407_40729


namespace NUMINAMATH_GPT_algebraic_expression_l407_40791

variable (m n x y : ℤ)

theorem algebraic_expression (h1 : x = m) (h2 : y = n) (h3 : x - y = 2) : n - m = -2 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_l407_40791


namespace NUMINAMATH_GPT_max_vx_minus_yz_l407_40741

-- Define the set A
def A : Set ℤ := {-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5}

-- Define the conditions
variables (v w x y z : ℤ)
#check v ∈ A -- v belongs to set A
#check w ∈ A -- w belongs to set A
#check x ∈ A -- x belongs to set A
#check y ∈ A -- y belongs to set A
#check z ∈ A -- z belongs to set A

-- vw = x
axiom vw_eq_x : v * w = x

-- w ≠ 0
axiom w_ne_zero : w ≠ 0

-- The target problem
theorem max_vx_minus_yz : ∃ v w x y z : ℤ, v ∈ A ∧ w ∈ A ∧ x ∈ A ∧ y ∈ A ∧ z ∈ A ∧ v * w = x ∧ w ≠ 0 ∧ (v * x - y * z) = 150 := by
  sorry

end NUMINAMATH_GPT_max_vx_minus_yz_l407_40741


namespace NUMINAMATH_GPT_find_x_l407_40713

theorem find_x (x y : ℝ) (h1 : y = 1) (h2 : 4 * x - 2 * y + 3 = 3 * x + 3 * y) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l407_40713


namespace NUMINAMATH_GPT_problem_l407_40799

theorem problem : (112^2 - 97^2) / 15 = 209 := by
  sorry

end NUMINAMATH_GPT_problem_l407_40799


namespace NUMINAMATH_GPT_fifth_digit_is_one_l407_40764

def self_descriptive_seven_digit_number (A B C D E F G : ℕ) : Prop :=
  A = 3 ∧ B = 2 ∧ C = 2 ∧ D = 1 ∧ E = 1 ∧ [A, B, C, D, E, F, G].count 0 = A ∧
  [A, B, C, D, E, F, G].count 1 = B ∧ [A, B, C, D, E, F, G].count 2 = C ∧
  [A, B, C, D, E, F, G].count 3 = D ∧ [A, B, C, D, E, F, G].count 4 = E

theorem fifth_digit_is_one
  (A B C D E F G : ℕ) (h : self_descriptive_seven_digit_number A B C D E F G) : E = 1 := by
  sorry

end NUMINAMATH_GPT_fifth_digit_is_one_l407_40764


namespace NUMINAMATH_GPT_inverse_proportion_y_relation_l407_40747

theorem inverse_proportion_y_relation (x₁ x₂ y₁ y₂ : ℝ) 
  (hA : y₁ = -4 / x₁) 
  (hB : y₂ = -4 / x₂)
  (h₁ : x₁ < 0) 
  (h₂ : 0 < x₂) : 
  y₁ > y₂ := 
sorry

end NUMINAMATH_GPT_inverse_proportion_y_relation_l407_40747


namespace NUMINAMATH_GPT_sum_original_numbers_is_five_l407_40721

noncomputable def sum_original_numbers (a b c d : ℤ) : ℤ :=
  a + b + c + d

theorem sum_original_numbers_is_five (a b c d : ℤ) (hab : 10 * a + b = overline_ab) 
  (h : 100 * (10 * a + b) + 10 * c + 7 * d = 2024) : sum_original_numbers a b c d = 5 :=
sorry

end NUMINAMATH_GPT_sum_original_numbers_is_five_l407_40721


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l407_40706

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)
variable (d : α)

-- Condition definitions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = d

def sum_condition (a : ℕ → α) : Prop :=
  a 2 + a 5 + a 8 = 39

-- The goal statement to prove
theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_sum : sum_condition a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 117 :=
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l407_40706


namespace NUMINAMATH_GPT_charles_finishes_in_11_days_l407_40748

theorem charles_finishes_in_11_days : 
  ∀ (total_pages : ℕ) (pages_mon : ℕ) (pages_tue : ℕ) (pages_wed : ℕ) (pages_thu : ℕ) 
    (does_not_read_on_weekend : Prop),
  total_pages = 96 →
  pages_mon = 7 →
  pages_tue = 12 →
  pages_wed = 10 →
  pages_thu = 6 →
  does_not_read_on_weekend →
  ∃ days_to_finish : ℕ, days_to_finish = 11 :=
by
  intros
  sorry

end NUMINAMATH_GPT_charles_finishes_in_11_days_l407_40748


namespace NUMINAMATH_GPT_buses_needed_40_buses_needed_30_l407_40786

-- Define the number of students
def number_of_students : ℕ := 186

-- Define the function to calculate minimum buses needed
def min_buses_needed (n : ℕ) : ℕ := (number_of_students + n - 1) / n

-- Theorem statements for the specific cases
theorem buses_needed_40 : min_buses_needed 40 = 5 := 
by 
  sorry

theorem buses_needed_30 : min_buses_needed 30 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_buses_needed_40_buses_needed_30_l407_40786


namespace NUMINAMATH_GPT_problem_l407_40784

-- Define sets A and B
def A : Set ℝ := { x | x > 1 }
def B : Set ℝ := { y | y <= -1 }

-- Define set C as a function of a
def C (a : ℝ) : Set ℝ := { x | x < -a / 2 }

-- The statement of the problem: if B ⊆ C, then a < 2
theorem problem (a : ℝ) : (B ⊆ C a) → a < 2 :=
by sorry

end NUMINAMATH_GPT_problem_l407_40784


namespace NUMINAMATH_GPT_min_buses_needed_l407_40732

theorem min_buses_needed (total_students : ℕ) (bus45_capacity : ℕ) (bus40_capacity : ℕ) : 
  total_students = 530 ∧ bus45_capacity = 45 ∧ bus40_capacity = 40 → 
  ∃ (n : ℕ), n = 12 :=
by 
  intro h
  obtain ⟨htotal, hbus45, hbus40⟩ := h
  -- Proof would go here...
  sorry

end NUMINAMATH_GPT_min_buses_needed_l407_40732


namespace NUMINAMATH_GPT_batsman_average_after_25th_innings_l407_40771

theorem batsman_average_after_25th_innings (A : ℝ) (runs_25th : ℝ) (increase : ℝ) (not_out_innings : ℕ) 
    (total_innings : ℕ) (average_increase_condition : 24 * A + runs_25th = 25 * (A + increase)) :       
    runs_25th = 150 ∧ increase = 3 ∧ not_out_innings = 3 ∧ total_innings = 25 → 
    ∃ avg : ℝ, avg = 88.64 := by 
  sorry

end NUMINAMATH_GPT_batsman_average_after_25th_innings_l407_40771


namespace NUMINAMATH_GPT_maximize_expression_l407_40740

-- Given the condition
theorem maximize_expression (x y : ℝ) (h : x + y = 1) : (x^3 + 1) * (y^3 + 1) ≤ (1)^3 + 1 * (0)^3 + 1 * (0)^3 + 1 :=
sorry

end NUMINAMATH_GPT_maximize_expression_l407_40740


namespace NUMINAMATH_GPT_cost_price_for_one_meter_l407_40754

variable (meters_sold : Nat) (selling_price : Nat) (loss_per_meter : Nat) (total_cost_price : Nat)
variable (cost_price_per_meter : Rat)

theorem cost_price_for_one_meter (h1 : meters_sold = 200)
                                  (h2 : selling_price = 12000)
                                  (h3 : loss_per_meter = 12)
                                  (h4 : total_cost_price = selling_price + loss_per_meter * meters_sold)
                                  (h5 : cost_price_per_meter = total_cost_price / meters_sold) :
  cost_price_per_meter = 72 := by
  sorry

end NUMINAMATH_GPT_cost_price_for_one_meter_l407_40754


namespace NUMINAMATH_GPT_max_value_fraction_l407_40798

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∀ z, z = (x / (2 * x + y) + y / (x + 2 * y)) → z ≤ (2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_max_value_fraction_l407_40798


namespace NUMINAMATH_GPT_Felicity_used_23_gallons_l407_40700

variable (A Felicity : ℕ)
variable (h1 : Felicity = 4 * A - 5)
variable (h2 : A + Felicity = 30)

theorem Felicity_used_23_gallons : Felicity = 23 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_Felicity_used_23_gallons_l407_40700
