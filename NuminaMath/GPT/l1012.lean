import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l1012_101296

noncomputable def inequality_condition (x : ℝ) (a : ℝ) : Prop :=
  a - 2 * x - |Real.log x| ≤ 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0 → inequality_condition x a) ↔ a ≤ 1 + Real.log 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1012_101296


namespace NUMINAMATH_GPT_alice_age_multiple_sum_l1012_101239

theorem alice_age_multiple_sum (B : ℕ) (C : ℕ := 3) (A : ℕ := B + 2) (next_multiple_age : ℕ := A + (3 - (A % 3))) :
  B % C = 0 ∧ A = B + 2 ∧ C = 3 → 
  (next_multiple_age % 3 = 0 ∧
   (next_multiple_age / 10) + (next_multiple_age % 10) = 6) := 
by
  intros h
  sorry

end NUMINAMATH_GPT_alice_age_multiple_sum_l1012_101239


namespace NUMINAMATH_GPT_num_8tuples_satisfying_condition_l1012_101275

theorem num_8tuples_satisfying_condition :
  (∃! (y : Fin 8 → ℝ),
    (2 - y 0)^2 + (y 0 - y 1)^2 + (y 1 - y 2)^2 + 
    (y 2 - y 3)^2 + (y 3 - y 4)^2 + (y 4 - y 5)^2 + 
    (y 5 - y 6)^2 + (y 6 - y 7)^2 + y 7^2 = 4 / 9) :=
sorry

end NUMINAMATH_GPT_num_8tuples_satisfying_condition_l1012_101275


namespace NUMINAMATH_GPT_distance_covered_downstream_l1012_101273

-- Conditions
def boat_speed_still_water : ℝ := 16
def stream_rate : ℝ := 5
def time_downstream : ℝ := 6

-- Effective speed downstream
def effective_speed_downstream := boat_speed_still_water + stream_rate

-- Distance covered downstream
def distance_downstream := effective_speed_downstream * time_downstream

-- Theorem to prove
theorem distance_covered_downstream :
  (distance_downstream = 126) :=
by
  sorry

end NUMINAMATH_GPT_distance_covered_downstream_l1012_101273


namespace NUMINAMATH_GPT_negation_proposition_l1012_101251

open Classical

variable (x : ℝ)

theorem negation_proposition :
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l1012_101251


namespace NUMINAMATH_GPT_simplify_expression_l1012_101221

theorem simplify_expression (y : ℝ) :
  (18 * y^3) * (9 * y^2) * (1 / (6 * y)^2) = (9 / 2) * y^3 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1012_101221


namespace NUMINAMATH_GPT_digit_b_divisible_by_7_l1012_101253

theorem digit_b_divisible_by_7 (B : ℕ) (h : 0 ≤ B ∧ B ≤ 9) 
  (hdiv : (4000 + 110 * B + 3) % 7 = 0) : B = 0 :=
by
  sorry

end NUMINAMATH_GPT_digit_b_divisible_by_7_l1012_101253


namespace NUMINAMATH_GPT_men_build_fountain_l1012_101284

theorem men_build_fountain (m1 m2 : ℕ) (l1 l2 d1 d2 : ℕ) (work_rate : ℚ)
  (h1 : m1 * d1 = l1 * work_rate)
  (h2 : work_rate = 56 / (20 * 7))
  (h3 : l1 = 56)
  (h4 : l2 = 42)
  (h5 : m1 = 20)
  (h6 : m2 = 35)
  (h7 : d1 = 7)
  : d2 = 3 :=
sorry

end NUMINAMATH_GPT_men_build_fountain_l1012_101284


namespace NUMINAMATH_GPT_gcd_lcm_of_consecutive_naturals_l1012_101285

theorem gcd_lcm_of_consecutive_naturals (m : ℕ) (h : m > 0) (n : ℕ) (hn : n = m + 1) :
  gcd m n = 1 ∧ lcm m n = m * n :=
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_of_consecutive_naturals_l1012_101285


namespace NUMINAMATH_GPT_parabola_standard_eq_l1012_101241

theorem parabola_standard_eq (h : ∃ (x y : ℝ), x - 2 * y - 4 = 0 ∧ (
                         (y = 0 ∧ x = 4 ∧ y^2 = 16 * x) ∨ 
                         (x = 0 ∧ y = -2 ∧ x^2 = -8 * y))
                         ) :
                         (y^2 = 16 * x) ∨ (x^2 = -8 * y) :=
by 
  sorry

end NUMINAMATH_GPT_parabola_standard_eq_l1012_101241


namespace NUMINAMATH_GPT_toys_in_row_l1012_101205

theorem toys_in_row (n_left n_right : ℕ) (hy : 10 = n_left + 1) (hy' : 7 = n_right + 1) :
  n_left + n_right + 1 = 16 :=
by
  -- Fill in the proof here
  sorry

end NUMINAMATH_GPT_toys_in_row_l1012_101205


namespace NUMINAMATH_GPT_num_prime_divisors_50_fact_l1012_101289
open Nat -- To simplify working with natural numbers

-- We define the prime numbers less than or equal to 50.
def primes_le_50 := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

-- The problem statement: Prove that the number of prime divisors of 50! which are less than or equal to 50 is 15.
theorem num_prime_divisors_50_fact : (primes_le_50.length = 15) :=
by 
  -- Here we use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_num_prime_divisors_50_fact_l1012_101289


namespace NUMINAMATH_GPT_desks_per_row_calc_l1012_101213

theorem desks_per_row_calc :
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  (total_desks / 4 = 6) :=
by
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  show total_desks / 4 = 6
  sorry

end NUMINAMATH_GPT_desks_per_row_calc_l1012_101213


namespace NUMINAMATH_GPT_mean_difference_is_882_l1012_101291

variable (S : ℤ) (N : ℤ) (S_N_correct : N = 1000)

def actual_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 98000) / N

def incorrect_mean (S : ℤ) (N : ℤ) : ℚ :=
  (S + 980000) / N

theorem mean_difference_is_882 
  (S : ℤ) 
  (N : ℤ) 
  (S_N_correct : N = 1000) 
  (S_in_range : 8200 ≤ S) 
  (S_actual : S + 98000 ≤ 980000) :
  incorrect_mean S N - actual_mean S N = 882 := 
by
  /- Proof steps would go here -/
  sorry

end NUMINAMATH_GPT_mean_difference_is_882_l1012_101291


namespace NUMINAMATH_GPT_find_B_share_l1012_101278

-- Definitions for the conditions
def proportion (a b c d : ℕ) := 6 * a = 3 * b ∧ 3 * b = 5 * c ∧ 5 * c = 4 * d

def condition (c d : ℕ) := c = d + 1000

-- Statement of the problem
theorem find_B_share (A B C D : ℕ) (x : ℕ) 
  (h1 : proportion (6*x) (3*x) (5*x) (4*x)) 
  (h2 : condition (5*x) (4*x)) : 
  B = 3000 :=
by 
  sorry

end NUMINAMATH_GPT_find_B_share_l1012_101278


namespace NUMINAMATH_GPT_ed_more_marbles_l1012_101224

-- Define variables for initial number of marbles
variables {E D : ℕ}

-- Ed had some more marbles than Doug initially.
-- Doug lost 8 of his marbles at the playground.
-- Now Ed has 30 more marbles than Doug.
theorem ed_more_marbles (h : E = (D - 8) + 30) : E - D = 22 :=
by
  sorry

end NUMINAMATH_GPT_ed_more_marbles_l1012_101224


namespace NUMINAMATH_GPT_area_of_triangle_bounded_by_coordinate_axes_and_line_l1012_101279

def area_of_triangle (b h : ℕ) : ℕ := (b * h) / 2

theorem area_of_triangle_bounded_by_coordinate_axes_and_line :
  area_of_triangle 4 6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_bounded_by_coordinate_axes_and_line_l1012_101279


namespace NUMINAMATH_GPT_find_decimal_decrease_l1012_101236

noncomputable def tax_diminished_percentage (T C : ℝ) (X : ℝ) : Prop :=
  let new_tax := T * (1 - X / 100)
  let new_consumption := C * 1.15
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  new_revenue = original_revenue * 0.943

theorem find_decimal_decrease (T C : ℝ) (X : ℝ) :
  tax_diminished_percentage T C X → X = 18 := sorry

end NUMINAMATH_GPT_find_decimal_decrease_l1012_101236


namespace NUMINAMATH_GPT_compare_xyz_l1012_101297

noncomputable def x := (0.5 : ℝ)^(0.5 : ℝ)
noncomputable def y := (0.5 : ℝ)^(1.3 : ℝ)
noncomputable def z := (1.3 : ℝ)^(0.5 : ℝ)

theorem compare_xyz : z > x ∧ x > y := by
  sorry

end NUMINAMATH_GPT_compare_xyz_l1012_101297


namespace NUMINAMATH_GPT_balance_two_diamonds_three_bullets_l1012_101208

-- Define the variables
variables (a b c : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := 3 * a + b = 9 * c
def condition2 : Prop := a = b + c

-- Goal is to prove two diamonds (2 * b) balance three bullets (3 * c)
theorem balance_two_diamonds_three_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 
  2 * b = 3 * c := 
by 
  sorry

end NUMINAMATH_GPT_balance_two_diamonds_three_bullets_l1012_101208


namespace NUMINAMATH_GPT_right_triangle_5_12_13_l1012_101231

theorem right_triangle_5_12_13 (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) : a^2 + b^2 = c^2 := 
by 
   sorry

end NUMINAMATH_GPT_right_triangle_5_12_13_l1012_101231


namespace NUMINAMATH_GPT_find_m_minus_n_l1012_101217

theorem find_m_minus_n (m n : ℤ) (h1 : |m| = 14) (h2 : |n| = 23) (h3 : m + n > 0) : m - n = -9 ∨ m - n = -37 := 
sorry

end NUMINAMATH_GPT_find_m_minus_n_l1012_101217


namespace NUMINAMATH_GPT_sandy_correct_sums_l1012_101282

theorem sandy_correct_sums (x y : ℕ) (h1 : x + y = 30) (h2 : 3 * x - 2 * y = 50) : x = 22 :=
  by
  sorry

end NUMINAMATH_GPT_sandy_correct_sums_l1012_101282


namespace NUMINAMATH_GPT_time_for_trains_to_clear_l1012_101229

noncomputable def train_length_1 : ℕ := 120
noncomputable def train_length_2 : ℕ := 320
noncomputable def train_speed_1_kmph : ℚ := 42
noncomputable def train_speed_2_kmph : ℚ := 30

noncomputable def kmph_to_mps (speed: ℚ) : ℚ := (5/18) * speed

noncomputable def train_speed_1_mps : ℚ := kmph_to_mps train_speed_1_kmph
noncomputable def train_speed_2_mps : ℚ := kmph_to_mps train_speed_2_kmph

noncomputable def total_length : ℕ := train_length_1 + train_length_2
noncomputable def relative_speed : ℚ := train_speed_1_mps + train_speed_2_mps

noncomputable def collision_time : ℚ := total_length / relative_speed

theorem time_for_trains_to_clear : collision_time = 22 := by
  sorry

end NUMINAMATH_GPT_time_for_trains_to_clear_l1012_101229


namespace NUMINAMATH_GPT_incorrect_statement_C_l1012_101228

theorem incorrect_statement_C : 
  (∀ x : ℝ, |x| = x → x = 0 ∨ x = 1) ↔ False :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_incorrect_statement_C_l1012_101228


namespace NUMINAMATH_GPT_geometric_sequence_ratio_l1012_101249

theorem geometric_sequence_ratio (a b c q : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ b + c - a = x * q ∧ c + a - b = x * q^2 ∧ a + b - c = x * q^3 ∧ a + b + c = x) →
  q^3 + q^2 + q = 1 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_ratio_l1012_101249


namespace NUMINAMATH_GPT_cuberoot_eight_is_512_l1012_101298

-- Define the condition on x
def cuberoot_is_eight (x : ℕ) : Prop := 
  x^(1 / 3) = 8

-- The statement to be proved
theorem cuberoot_eight_is_512 : ∃ x : ℕ, cuberoot_is_eight x ∧ x = 512 := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_cuberoot_eight_is_512_l1012_101298


namespace NUMINAMATH_GPT_sum_first_and_third_angle_l1012_101246

-- Define the conditions
variable (A : ℕ)
axiom C1 : A + 2 * A + (A - 40) = 180

-- State the theorem to be proven
theorem sum_first_and_third_angle : A + (A - 40) = 70 :=
by
  sorry

end NUMINAMATH_GPT_sum_first_and_third_angle_l1012_101246


namespace NUMINAMATH_GPT_total_dots_not_visible_l1012_101247

theorem total_dots_not_visible :
  let total_dots := 4 * 21
  let visible_sum := 1 + 2 + 3 + 3 + 4 + 5 + 5 + 6
  total_dots - visible_sum = 55 :=
by
  sorry

end NUMINAMATH_GPT_total_dots_not_visible_l1012_101247


namespace NUMINAMATH_GPT_increasing_function_range_l1012_101274

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) (h₃ : ∀ x y : ℝ, x < y → f a x < f a y) : 
  3 / 2 ≤ a ∧ a < 2 := by
  sorry

end NUMINAMATH_GPT_increasing_function_range_l1012_101274


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1012_101259

theorem sufficient_not_necessary (p q : Prop) (h : p ∧ q) : (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1012_101259


namespace NUMINAMATH_GPT_min_draw_to_ensure_one_red_l1012_101257

theorem min_draw_to_ensure_one_red (b y r : ℕ) (h1 : b + y + r = 20) (h2 : b = y / 6) (h3 : r < y) : 
  ∃ n : ℕ, n = 15 ∧ ∀ d : ℕ, d < 15 → ∀ drawn : Finset (ℕ × ℕ × ℕ), drawn.card = d → ∃ card ∈ drawn, card.2 = r := 
sorry

end NUMINAMATH_GPT_min_draw_to_ensure_one_red_l1012_101257


namespace NUMINAMATH_GPT_cyclist_wait_time_l1012_101211

theorem cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (wait_time : ℝ) (catch_up_time : ℝ) 
  (hiker_speed_eq : hiker_speed = 4) 
  (cyclist_speed_eq : cyclist_speed = 12) 
  (wait_time_eq : wait_time = 5 / 60) 
  (catch_up_time_eq : catch_up_time = (2 / 3) / (1 / 15)) 
  : catch_up_time * 60 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_cyclist_wait_time_l1012_101211


namespace NUMINAMATH_GPT_parabola_directrix_l1012_101293

theorem parabola_directrix (p : ℝ) :
  (∀ y x : ℝ, y^2 = 2 * p * x ↔ x = -1 → p = 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1012_101293


namespace NUMINAMATH_GPT_find_triples_l1012_101214

-- Define the conditions in Lean 4
def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_positive_integer (n : ℕ) : Prop := n > 0

-- Define the math proof problem
theorem find_triples (m n p : ℕ) (hp : is_prime p) (hm : is_positive_integer m) (hn : is_positive_integer n) : 
  p^n + 3600 = m^2 ↔ (m = 61 ∧ n = 2 ∧ p = 11) ∨ (m = 65 ∧ n = 4 ∧ p = 5) ∨ (m = 68 ∧ n = 10 ∧ p = 2) :=
by
  sorry

end NUMINAMATH_GPT_find_triples_l1012_101214


namespace NUMINAMATH_GPT_find_common_difference_find_minimum_sum_minimum_sum_value_l1012_101225

-- Defining the arithmetic sequence and its properties
def a (n : ℕ) (d : ℚ) := (-3 : ℚ) + n * d

-- Given conditions
def condition_1 : ℚ := -3
def condition_2 (d : ℚ) := 11 * a 4 d = 5 * a 7 d - 13
def common_difference : ℚ := 31 / 9

-- Sum of the first n terms of an arithmetic sequence
def S (n : ℕ) (d : ℚ) := n * (-3 + (n - 1) * d / 2)

-- Defining the necessary theorems
theorem find_common_difference (d : ℚ) : condition_2 d → d = common_difference := by
  sorry

theorem find_minimum_sum (n : ℕ) : S n common_difference ≥ S 2 common_difference := by
  sorry

theorem minimum_sum_value : S 2 common_difference = -23 / 9 := by
  sorry

end NUMINAMATH_GPT_find_common_difference_find_minimum_sum_minimum_sum_value_l1012_101225


namespace NUMINAMATH_GPT_molecular_weight_of_7_moles_of_NH4_2SO4_l1012_101292

theorem molecular_weight_of_7_moles_of_NH4_2SO4 :
  let N_weight := 14.01
  let H_weight := 1.01
  let S_weight := 32.07
  let O_weight := 16.00
  let N_atoms := 2
  let H_atoms := 8
  let S_atoms := 1
  let O_atoms := 4
  let moles := 7
  let molecular_weight := (N_weight * N_atoms) + (H_weight * H_atoms) + (S_weight * S_atoms) + (O_weight * O_atoms)
  let total_weight := molecular_weight * moles
  total_weight = 924.19 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_7_moles_of_NH4_2SO4_l1012_101292


namespace NUMINAMATH_GPT_circle_center_coordinates_l1012_101220

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0 ↔ (x - h)^2 + (y - k)^2 = 13) ∧ h = 2 ∧ k = -3 :=
sorry

end NUMINAMATH_GPT_circle_center_coordinates_l1012_101220


namespace NUMINAMATH_GPT_black_white_ratio_extended_pattern_l1012_101295

theorem black_white_ratio_extended_pattern
  (original_black : ℕ) (original_white : ℕ) (added_black : ℕ)
  (h1 : original_black = 10)
  (h2 : original_white = 26)
  (h3 : added_black = 20) :
  (original_black + added_black) / original_white = 30 / 26 :=
by sorry

end NUMINAMATH_GPT_black_white_ratio_extended_pattern_l1012_101295


namespace NUMINAMATH_GPT_root_equation_m_l1012_101288

theorem root_equation_m (m : ℝ) : 
  (∃ (x : ℝ), x = -1 ∧ m*x^2 + x - m^2 + 1 = 0) → m = 1 :=
by 
  sorry

end NUMINAMATH_GPT_root_equation_m_l1012_101288


namespace NUMINAMATH_GPT_find_b_l1012_101223

def h (x : ℝ) : ℝ := 4 * x - 5

theorem find_b (b : ℝ) (h_b : h b = 1) : b = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l1012_101223


namespace NUMINAMATH_GPT_cos_value_l1012_101237

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 3) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 9 := by
  sorry

end NUMINAMATH_GPT_cos_value_l1012_101237


namespace NUMINAMATH_GPT_decreasing_function_condition_l1012_101218

theorem decreasing_function_condition (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, x ≤ 3 → deriv f x ≤ 0) ↔ (m ≥ 1) :=
by 
  sorry

end NUMINAMATH_GPT_decreasing_function_condition_l1012_101218


namespace NUMINAMATH_GPT_exists_same_color_points_one_meter_apart_l1012_101203

-- Declare the colors as an enumeration
inductive Color
| red : Color
| black : Color

-- Define the function that assigns a color to each point in the plane
def color (point : ℝ × ℝ) : Color := sorry

-- The theorem to be proven
theorem exists_same_color_points_one_meter_apart :
  ∃ x y : ℝ × ℝ, x ≠ y ∧ dist x y = 1 ∧ color x = color y :=
sorry

end NUMINAMATH_GPT_exists_same_color_points_one_meter_apart_l1012_101203


namespace NUMINAMATH_GPT_negation_of_existence_l1012_101271

theorem negation_of_existence :
  (¬ ∃ x₀ : ℝ, x₀^2 + 2*x₀ + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l1012_101271


namespace NUMINAMATH_GPT_volume_ratio_of_spheres_l1012_101260

theorem volume_ratio_of_spheres (r1 r2 r3 : ℝ) 
  (h : r1 / r2 = 1 / 2 ∧ r2 / r3 = 2 / 3) : 
  (4/3 * π * r3^3) = 3 * (4/3 * π * r1^3 + 4/3 * π * r2^3) :=
by
  sorry

end NUMINAMATH_GPT_volume_ratio_of_spheres_l1012_101260


namespace NUMINAMATH_GPT_range_of_m_length_of_chord_l1012_101248

-- Definition of Circle C
def CircleC (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0

-- Definition of Circle D
def CircleD (x y : ℝ) := (x + 3)^2 + (y + 1)^2 = 16

-- Definition of Line l
def LineL (x y : ℝ) := x + 2*y - 4 = 0

-- Problem 1: Prove range of values for m
theorem range_of_m (m : ℝ) : (∀ x y, CircleC x y m) → m < 5 := by
  sorry

-- Problem 2: Prove length of chord MN
theorem length_of_chord (x y : ℝ) :
  CircleC x y 4 ∧ CircleD x y ∧ LineL x y →
  (∃ MN, MN = (4*Real.sqrt 5) / 5) := by
    sorry

end NUMINAMATH_GPT_range_of_m_length_of_chord_l1012_101248


namespace NUMINAMATH_GPT_positive_integer_solution_l1012_101233

theorem positive_integer_solution (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (1 / (x * x : ℝ) + 1 / (y * y : ℝ) + 1 / (z * z : ℝ) + 1 / (t * t : ℝ) = 1) ↔ (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2) :=
by
  sorry

end NUMINAMATH_GPT_positive_integer_solution_l1012_101233


namespace NUMINAMATH_GPT_num_distinct_ordered_pairs_l1012_101204

theorem num_distinct_ordered_pairs (a b c : ℕ) (h₀ : a + b + c = 50) (h₁ : c = 10) (h₂ : 0 < a ∧ 0 < b) :
  ∃ n : ℕ, n = 39 := 
sorry

end NUMINAMATH_GPT_num_distinct_ordered_pairs_l1012_101204


namespace NUMINAMATH_GPT_find_pairs_of_square_numbers_l1012_101265

theorem find_pairs_of_square_numbers (a b k : ℕ) (hk : k ≥ 2) 
  (h_eq : (a * a + b * b) = k * k * (a * b + 1)) : 
  (a = k ∧ b = k * k * k) ∨ (b = k ∧ a = k * k * k) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_of_square_numbers_l1012_101265


namespace NUMINAMATH_GPT_birch_count_is_87_l1012_101226

def num_trees : ℕ := 130
def incorrect_signs (B L : ℕ) : Prop := B + L = num_trees ∧ L + 1 = num_trees - 1 ∧ B = 87

theorem birch_count_is_87 (B L : ℕ) (h1 : B + L = num_trees) (h2 : L + 1 = num_trees - 1) :
  B = 87 :=
sorry

end NUMINAMATH_GPT_birch_count_is_87_l1012_101226


namespace NUMINAMATH_GPT_sandy_correct_sums_l1012_101200

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 45) : c = 21 :=
  sorry

end NUMINAMATH_GPT_sandy_correct_sums_l1012_101200


namespace NUMINAMATH_GPT_number_of_black_squares_in_58th_row_l1012_101264

theorem number_of_black_squares_in_58th_row :
  let pattern := [1, 0, 0] -- pattern where 1 represents a black square
  let n := 58
  let total_squares := 2 * n - 1 -- total squares in the 58th row
  let black_count := total_squares / 3 -- number of black squares in the repeating pattern
  black_count = 38 :=
by
  let pattern := [1, 0, 0]
  let n := 58
  let total_squares := 2 * n - 1
  let black_count := total_squares / 3
  have black_count_eq_38 : 38 = (115 / 3) := by sorry
  exact black_count_eq_38.symm

end NUMINAMATH_GPT_number_of_black_squares_in_58th_row_l1012_101264


namespace NUMINAMATH_GPT_left_handed_classical_music_lovers_l1012_101210

-- Define the conditions
variables (total_people left_handed classical_music right_handed_dislike : ℕ)
variables (x : ℕ) -- x will represent the number of left-handed classical music lovers

-- State the assumptions based on conditions
axiom h1 : total_people = 30
axiom h2 : left_handed = 12
axiom h3 : classical_music = 20
axiom h4 : right_handed_dislike = 3
axiom h5 : 30 = x + (12 - x) + (20 - x) + 3

-- State the theorem to prove
theorem left_handed_classical_music_lovers : x = 5 :=
by {
  -- Skip the proof using sorry
  sorry
}

end NUMINAMATH_GPT_left_handed_classical_music_lovers_l1012_101210


namespace NUMINAMATH_GPT_problem_l1012_101263

def count_numbers_with_more_ones_than_zeros (n : ℕ) : ℕ :=
  -- function that counts numbers less than or equal to 'n'
  -- whose binary representation has more '1's than '0's
  sorry

theorem problem (M := count_numbers_with_more_ones_than_zeros 1500) : 
  M % 1000 = 884 :=
sorry

end NUMINAMATH_GPT_problem_l1012_101263


namespace NUMINAMATH_GPT_min_rectangle_area_l1012_101232

theorem min_rectangle_area : 
  ∃ (x y : ℕ), 2 * (x + y) = 80 ∧ x * y = 39 :=
by
  sorry

end NUMINAMATH_GPT_min_rectangle_area_l1012_101232


namespace NUMINAMATH_GPT_inv_composition_l1012_101243

theorem inv_composition (f g : ℝ → ℝ) (hf : Function.Bijective f) (hg : Function.Bijective g) (h : ∀ x, f⁻¹ (g x) = 2 * x - 4) : 
  g⁻¹ (f (-3)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_inv_composition_l1012_101243


namespace NUMINAMATH_GPT_evaluate_x2_y2_l1012_101286

theorem evaluate_x2_y2 (x y : ℝ) (h1 : x + y = 12) (h2 : 3 * x + y = 18) : x^2 - y^2 = -72 := 
sorry

end NUMINAMATH_GPT_evaluate_x2_y2_l1012_101286


namespace NUMINAMATH_GPT_initial_price_of_sugar_per_kg_l1012_101276

theorem initial_price_of_sugar_per_kg
  (initial_price : ℝ)
  (final_price : ℝ)
  (required_reduction : ℝ)
  (initial_price_eq : initial_price = 6)
  (final_price_eq : final_price = 7.5)
  (required_reduction_eq : required_reduction = 0.19999999999999996) :
  initial_price = 6 :=
by
  sorry

end NUMINAMATH_GPT_initial_price_of_sugar_per_kg_l1012_101276


namespace NUMINAMATH_GPT_sum_of_2x2_table_is_zero_l1012_101258

theorem sum_of_2x2_table_is_zero {a b c d : ℤ} 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_eq : a + b = c + d)
  (prod_eq : a * c = b * d) :
  a + b + c + d = 0 :=
by sorry

end NUMINAMATH_GPT_sum_of_2x2_table_is_zero_l1012_101258


namespace NUMINAMATH_GPT_remaining_budget_for_public_spaces_l1012_101206

noncomputable def total_budget : ℝ := 32
noncomputable def policing_budget : ℝ := total_budget / 2
noncomputable def education_budget : ℝ := 12
noncomputable def remaining_budget : ℝ := total_budget - (policing_budget + education_budget)

theorem remaining_budget_for_public_spaces : remaining_budget = 4 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_remaining_budget_for_public_spaces_l1012_101206


namespace NUMINAMATH_GPT_stanley_sold_4_cups_per_hour_l1012_101281

theorem stanley_sold_4_cups_per_hour (S : ℕ) (Carl_Hour : ℕ) :
  (Carl_Hour = 7) →
  21 = (Carl_Hour * 3) →
  (21 - 9) = (S * 3) →
  S = 4 :=
by
  intros Carl_Hour_eq Carl_3hours Stanley_eq
  sorry

end NUMINAMATH_GPT_stanley_sold_4_cups_per_hour_l1012_101281


namespace NUMINAMATH_GPT_total_earnings_l1012_101240

variable (phone_cost : ℕ) (laptop_cost : ℕ) (computer_cost : ℕ)
variable (num_phone_repairs : ℕ) (num_laptop_repairs : ℕ) (num_computer_repairs : ℕ)

theorem total_earnings (h1 : phone_cost = 11) (h2 : laptop_cost = 15) 
                       (h3 : computer_cost = 18) (h4 : num_phone_repairs = 5) 
                       (h5 : num_laptop_repairs = 2) (h6 : num_computer_repairs = 2) :
                       (num_phone_repairs * phone_cost + num_laptop_repairs * laptop_cost + num_computer_repairs * computer_cost) = 121 := 
by
  sorry

end NUMINAMATH_GPT_total_earnings_l1012_101240


namespace NUMINAMATH_GPT_smallest_n_gcd_l1012_101216

theorem smallest_n_gcd (n : ℕ) :
  (∃ n > 0, gcd (11 * n - 3) (8 * n + 2) > 1) ∧ (∀ m > 0, gcd (11 * m - 3) (8 * m + 2) > 1 → m ≥ n) ↔ n = 19 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_gcd_l1012_101216


namespace NUMINAMATH_GPT_sqrt_two_irrational_l1012_101219

theorem sqrt_two_irrational :
  ¬ ∃ (p q : ℕ), p ≠ 0 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (↑q / ↑p) ^ 2 = (2:ℝ) :=
sorry

end NUMINAMATH_GPT_sqrt_two_irrational_l1012_101219


namespace NUMINAMATH_GPT_number_of_valid_pairs_l1012_101252

theorem number_of_valid_pairs (a b : ℝ) :
  (∃ x y : ℤ, a * (x : ℝ) + b * (y : ℝ) = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) →
  ∃! pairs_count : ℕ, pairs_count = 72 :=
by
  sorry

end NUMINAMATH_GPT_number_of_valid_pairs_l1012_101252


namespace NUMINAMATH_GPT_sum_of_circle_areas_l1012_101262

theorem sum_of_circle_areas (a b c: ℝ)
  (h1: a + b = 6)
  (h2: b + c = 8)
  (h3: a + c = 10) :
  π * a^2 + π * b^2 + π * c^2 = 56 * π := 
by
  sorry

end NUMINAMATH_GPT_sum_of_circle_areas_l1012_101262


namespace NUMINAMATH_GPT_Taehyung_age_l1012_101209

variable (T U : Nat)

-- Condition 1: Taehyung is 17 years younger than his uncle
def condition1 : Prop := U = T + 17

-- Condition 2: Four years later, the sum of their ages is 43
def condition2 : Prop := (T + 4) + (U + 4) = 43

-- The goal is to prove that Taehyung's current age is 9, given the conditions above
theorem Taehyung_age : condition1 T U ∧ condition2 T U → T = 9 := by
  sorry

end NUMINAMATH_GPT_Taehyung_age_l1012_101209


namespace NUMINAMATH_GPT_zoo_camels_l1012_101272

theorem zoo_camels (x y : ℕ) (h1 : x - y = 10) (h2 : x + 2 * y = 55) : x + y = 40 :=
by sorry

end NUMINAMATH_GPT_zoo_camels_l1012_101272


namespace NUMINAMATH_GPT_percent_value_in_quarters_l1012_101280

def nickel_value : ℕ := 5
def quarter_value : ℕ := 25
def num_nickels : ℕ := 80
def num_quarters : ℕ := 40

def value_in_nickels : ℕ := num_nickels * nickel_value
def value_in_quarters : ℕ := num_quarters * quarter_value
def total_value : ℕ := value_in_nickels + value_in_quarters

theorem percent_value_in_quarters :
  (value_in_quarters : ℚ) / total_value = 5 / 7 :=
by
  sorry

end NUMINAMATH_GPT_percent_value_in_quarters_l1012_101280


namespace NUMINAMATH_GPT_number_properties_l1012_101287

-- Define what it means for a digit to be in a specific place
def digit_at_place (n place : ℕ) (d : ℕ) : Prop := 
  (n / 10 ^ place) % 10 = d

-- The given number
def specific_number : ℕ := 670154500

-- Conditions: specific number has specific digit in defined places
theorem number_properties : (digit_at_place specific_number 7 7) ∧ (digit_at_place specific_number 2 5) :=
by
  -- Proof of the theorem
  sorry

end NUMINAMATH_GPT_number_properties_l1012_101287


namespace NUMINAMATH_GPT_shaded_area_is_correct_l1012_101212

-- Definitions based on the conditions
def is_square (s : ℝ) (area : ℝ) : Prop := s * s = area
def rect_area (l w : ℝ) : ℝ := l * w

variables (s : ℝ) (area_s : ℝ) (rect1_l rect1_w rect2_l rect2_w : ℝ)

-- Given conditions
def square := is_square s area_s
def rect1 := rect_area rect1_l rect1_w
def rect2 := rect_area rect2_l rect2_w

-- Problem statement: Prove the area of the shaded region
theorem shaded_area_is_correct
  (s: ℝ)
  (rect1_l rect1_w rect2_l rect2_w : ℝ)
  (h_square: is_square s 16)
  (h_rect1: rect_area rect1_l rect1_w = 6)
  (h_rect2: rect_area rect2_l rect2_w = 2) :
  (16 - (6 + 2) = 8) := 
  sorry

end NUMINAMATH_GPT_shaded_area_is_correct_l1012_101212


namespace NUMINAMATH_GPT_quadratic_inequality_l1012_101222

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_inequality (h : f b c (-1) = f b c 3) : f b c 1 < c ∧ c < f b c 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_l1012_101222


namespace NUMINAMATH_GPT_range_of_m_l1012_101227

def has_solution_in_interval (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), x^2 - 2 * x - 1 + m ≤ 0 

theorem range_of_m (m : ℝ) : has_solution_in_interval m ↔ m ≤ 2 := by 
  sorry

end NUMINAMATH_GPT_range_of_m_l1012_101227


namespace NUMINAMATH_GPT_octal_subtraction_l1012_101266

theorem octal_subtraction : (53 - 27 : ℕ) = 24 :=
by sorry

end NUMINAMATH_GPT_octal_subtraction_l1012_101266


namespace NUMINAMATH_GPT_container_ratio_l1012_101294

theorem container_ratio (A B : ℝ) (h : (4 / 5) * A = (2 / 3) * B) : (A / B) = (5 / 6) :=
by
  sorry

end NUMINAMATH_GPT_container_ratio_l1012_101294


namespace NUMINAMATH_GPT_pencils_purchased_l1012_101255

variable (P : ℕ)

theorem pencils_purchased (misplaced broke found bought left : ℕ) (h1 : misplaced = 7) (h2 : broke = 3) (h3 : found = 4) (h4 : bought = 2) (h5 : left = 16) :
  P - misplaced - broke + found + bought = left → P = 22 :=
by
  intros h
  have h_eq : P - 7 - 3 + 4 + 2 = 16 := by
    rw [h1, h2, h3, h4, h5] at h; exact h
  sorry

end NUMINAMATH_GPT_pencils_purchased_l1012_101255


namespace NUMINAMATH_GPT_power_function_value_l1012_101277

theorem power_function_value {α : ℝ} (h : 3^α = Real.sqrt 3) : (9 : ℝ)^α = 3 :=
by sorry

end NUMINAMATH_GPT_power_function_value_l1012_101277


namespace NUMINAMATH_GPT_distance_between_stations_l1012_101283

/-- Two trains start at the same time from two stations and proceed towards each other. 
    The first train travels at 20 km/hr and the second train travels at 25 km/hr. 
    When they meet, the second train has traveled 60 km more than the first train. -/
theorem distance_between_stations
    (t : ℝ) -- The time in hours when they meet
    (x : ℝ) -- The distance traveled by the slower train
    (d1 d2 : ℝ) -- Distances traveled by the two trains respectively
    (h1 : 20 * t = x)
    (h2 : 25 * t = x + 60) :
  d1 + d2 = 540 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_stations_l1012_101283


namespace NUMINAMATH_GPT_expr1_correct_expr2_correct_expr3_correct_l1012_101268

-- Define the expressions and corresponding correct answers
def expr1 : Int := 58 + 15 * 4
def expr2 : Int := 216 - 72 / 8
def expr3 : Int := (358 - 295) / 7

-- State the proof goals
theorem expr1_correct : expr1 = 118 := by
  sorry

theorem expr2_correct : expr2 = 207 := by
  sorry

theorem expr3_correct : expr3 = 9 := by
  sorry

end NUMINAMATH_GPT_expr1_correct_expr2_correct_expr3_correct_l1012_101268


namespace NUMINAMATH_GPT_sum_of_abc_l1012_101244

theorem sum_of_abc (a b c : ℕ) (h : a + b + c = 12) 
  (area_ratio : ℝ) (side_length_ratio : ℝ) 
  (ha : area_ratio = 50 / 98) 
  (hb : side_length_ratio = (Real.sqrt 50) / (Real.sqrt 98))
  (hc : side_length_ratio = (a * (Real.sqrt b)) / c) :
  a + b + c = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_abc_l1012_101244


namespace NUMINAMATH_GPT_polar_to_cartesian_coordinates_l1012_101242

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_coordinates :
  polar_to_cartesian 2 (2 / 3 * Real.pi) = (-1, Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_coordinates_l1012_101242


namespace NUMINAMATH_GPT_fred_seashells_l1012_101230

-- Definitions based on conditions
def tom_seashells : Nat := 15
def total_seashells : Nat := 58

-- The theorem we want to prove
theorem fred_seashells : (15 + F = 58) → F = 43 := 
by
  intro h
  have h1 : F = 58 - 15 := by linarith
  exact h1

end NUMINAMATH_GPT_fred_seashells_l1012_101230


namespace NUMINAMATH_GPT_root_sum_abs_gt_6_l1012_101201

variables (r1 r2 p : ℝ)

theorem root_sum_abs_gt_6 
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 9)
  (h3 : p^2 > 36) :
  |r1 + r2| > 6 :=
by sorry

end NUMINAMATH_GPT_root_sum_abs_gt_6_l1012_101201


namespace NUMINAMATH_GPT_gcd_of_360_and_150_is_30_l1012_101238

theorem gcd_of_360_and_150_is_30 : Nat.gcd 360 150 = 30 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_360_and_150_is_30_l1012_101238


namespace NUMINAMATH_GPT_calculation_correct_l1012_101267

theorem calculation_correct : 
  ((2 * (15^2 + 35^2 + 21^2) - (3^4 + 5^4 + 7^4)) / (3 + 5 + 7)) = 45 := by
  sorry

end NUMINAMATH_GPT_calculation_correct_l1012_101267


namespace NUMINAMATH_GPT_total_ingredients_l1012_101290

theorem total_ingredients (water : ℕ) (flour : ℕ) (salt : ℕ)
  (h_water : water = 10)
  (h_flour : flour = 16)
  (h_salt : salt = flour / 2) :
  water + flour + salt = 34 :=
by
  sorry

end NUMINAMATH_GPT_total_ingredients_l1012_101290


namespace NUMINAMATH_GPT_limit_sum_perimeters_areas_of_isosceles_triangles_l1012_101256

theorem limit_sum_perimeters_areas_of_isosceles_triangles (b s h : ℝ) : 
  ∃ P A : ℝ, 
    (P = 2*(b + 2*s)) ∧ 
    (A = (2/3)*b*h) :=
  sorry

end NUMINAMATH_GPT_limit_sum_perimeters_areas_of_isosceles_triangles_l1012_101256


namespace NUMINAMATH_GPT_one_third_of_1206_is_100_5_percent_of_400_l1012_101270

theorem one_third_of_1206_is_100_5_percent_of_400 (n m : ℕ) (f : ℝ) :
  n = 1206 → m = 400 → f = 1 / 3 → (n * f) / m * 100 = 100.5 :=
by
  intros h_n h_m h_f
  rw [h_n, h_m, h_f]
  sorry

end NUMINAMATH_GPT_one_third_of_1206_is_100_5_percent_of_400_l1012_101270


namespace NUMINAMATH_GPT_sin_cos_alpha_frac_l1012_101269

theorem sin_cos_alpha_frac (α : ℝ) (h : Real.tan (Real.pi - α) = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 3 := 
by
  sorry

end NUMINAMATH_GPT_sin_cos_alpha_frac_l1012_101269


namespace NUMINAMATH_GPT_find_subtracted_number_l1012_101235

theorem find_subtracted_number (x y : ℝ) (h1 : x = 62.5) (h2 : (2 * (x + 5)) / 5 - y = 22) : y = 5 :=
sorry

end NUMINAMATH_GPT_find_subtracted_number_l1012_101235


namespace NUMINAMATH_GPT_cyclist_downhill_speed_l1012_101261

noncomputable def downhill_speed (d uphill_speed avg_speed : ℝ) : ℝ :=
  let downhill_speed := (2 * d * uphill_speed) / (avg_speed * d - uphill_speed * 2)
  -- We want to prove
  downhill_speed

theorem cyclist_downhill_speed :
  downhill_speed 150 25 35 = 58.33 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_cyclist_downhill_speed_l1012_101261


namespace NUMINAMATH_GPT_time_to_fill_pond_l1012_101215

noncomputable def pond_capacity : ℝ := 200
noncomputable def normal_pump_rate : ℝ := 6
noncomputable def restriction_factor : ℝ := 2 / 3
noncomputable def restricted_pump_rate : ℝ := restriction_factor * normal_pump_rate

theorem time_to_fill_pond : pond_capacity / restricted_pump_rate = 50 := 
by 
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_time_to_fill_pond_l1012_101215


namespace NUMINAMATH_GPT_spencer_total_distance_l1012_101254

-- Define the individual segments of Spencer's travel
def walk1 : ℝ := 1.2
def bike1 : ℝ := 1.8
def bus1 : ℝ := 3
def walk2 : ℝ := 0.4
def walk3 : ℝ := 0.6
def bike2 : ℝ := 2
def walk4 : ℝ := 1.5

-- Define the conversion factors
def bike_to_walk_conversion : ℝ := 0.5
def bus_to_walk_conversion : ℝ := 0.8

-- Calculate the total walking distance
def total_walking_distance : ℝ := walk1 + walk2 + walk3 + walk4

-- Calculate the total biking distance as walking equivalent
def total_biking_distance_as_walking : ℝ := (bike1 + bike2) * bike_to_walk_conversion

-- Calculate the total bus distance as walking equivalent
def total_bus_distance_as_walking : ℝ := bus1 * bus_to_walk_conversion

-- Define the total walking equivalent distance
def total_distance : ℝ := total_walking_distance + total_biking_distance_as_walking + total_bus_distance_as_walking

-- Theorem stating the total distance covered is 8 miles
theorem spencer_total_distance : total_distance = 8 := by
  unfold total_distance
  unfold total_walking_distance
  unfold total_biking_distance_as_walking
  unfold total_bus_distance_as_walking
  norm_num
  sorry

end NUMINAMATH_GPT_spencer_total_distance_l1012_101254


namespace NUMINAMATH_GPT_part_b_part_c_l1012_101250

-- Statement for part b: In how many ways can the figure be properly filled with the numbers from 1 to 5?
def proper_fill_count_1_to_5 : Nat :=
  8

-- Statement for part c: In how many ways can the figure be properly filled with the numbers from 1 to 7?
def proper_fill_count_1_to_7 : Nat :=
  48

theorem part_b :
  proper_fill_count_1_to_5 = 8 :=
sorry

theorem part_c :
  proper_fill_count_1_to_7 = 48 :=
sorry

end NUMINAMATH_GPT_part_b_part_c_l1012_101250


namespace NUMINAMATH_GPT_no_square_number_divisible_by_six_in_range_l1012_101202

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (6 ∣ x) ∧ (50 < x) ∧ (x < 120) :=
by
  sorry

end NUMINAMATH_GPT_no_square_number_divisible_by_six_in_range_l1012_101202


namespace NUMINAMATH_GPT_num_first_graders_in_class_l1012_101299

def numKindergartners := 14
def numSecondGraders := 4
def totalStudents := 42

def numFirstGraders : Nat := totalStudents - (numKindergartners + numSecondGraders)

theorem num_first_graders_in_class :
  numFirstGraders = 24 :=
by
  sorry

end NUMINAMATH_GPT_num_first_graders_in_class_l1012_101299


namespace NUMINAMATH_GPT_trajectory_of_point_P_l1012_101234

theorem trajectory_of_point_P :
  ∀ (x y : ℝ), 
  (∀ (m n : ℝ), n = 2 * m - 4 → (1 - m, -n) = (x - 1, y)) → 
  y = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_point_P_l1012_101234


namespace NUMINAMATH_GPT_friend_spent_more_l1012_101207

theorem friend_spent_more (total_spent friend_spent: ℝ) (h_total: total_spent = 15) (h_friend: friend_spent = 10) :
  friend_spent - (total_spent - friend_spent) = 5 :=
by
  sorry

end NUMINAMATH_GPT_friend_spent_more_l1012_101207


namespace NUMINAMATH_GPT_thirteen_coins_value_l1012_101245

theorem thirteen_coins_value :
  ∃ (p n d q : ℕ), p + n + d + q = 13 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 141 ∧ 
                   2 ≤ p ∧ 2 ≤ n ∧ 2 ≤ d ∧ 2 ≤ q ∧ 
                   d = 3 :=
  sorry

end NUMINAMATH_GPT_thirteen_coins_value_l1012_101245
