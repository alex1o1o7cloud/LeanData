import Mathlib

namespace NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l1645_164573

def proposition_p (x : ℝ) := x - 1 = 0
def proposition_q (x : ℝ) := (x - 1) * (x + 2) = 0

theorem p_sufficient_but_not_necessary_for_q :
  ( (∀ x, proposition_p x → proposition_q x) ∧ ¬(∀ x, proposition_p x ↔ proposition_q x) ) := 
by
  sorry

end NUMINAMATH_GPT_p_sufficient_but_not_necessary_for_q_l1645_164573


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l1645_164548

theorem parabola_focus_coordinates :
  ∀ x y : ℝ, y^2 = -8 * x → (x, y) = (-2, 0) := by
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l1645_164548


namespace NUMINAMATH_GPT_inequality_solution_l1645_164599

theorem inequality_solution (x : ℝ) (hx1 : x ≥ -1/2) (hx2 : x ≠ 0) :
  (4 * x^2 / (1 - Real.sqrt (1 + 2 * x))^2 < 2 * x + 9) ↔ 
  (-1/2 ≤ x ∧ x < 0) ∨ (0 < x ∧ x < 45/8) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1645_164599


namespace NUMINAMATH_GPT_sales_tax_difference_l1645_164570

theorem sales_tax_difference (P : ℝ) (d t1 t2 : ℝ) :
  let discounted_price := P * (1 - d)
  let total_cost1 := discounted_price * (1 + t1)
  let total_cost2 := discounted_price * (1 + t2)
  t1 = 0.08 ∧ t2 = 0.075 ∧ P = 50 ∧ d = 0.05 →
  abs ((total_cost1 - total_cost2) - 0.24) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_sales_tax_difference_l1645_164570


namespace NUMINAMATH_GPT_vertex_of_parabola_minimum_value_for_x_ge_2_l1645_164529

theorem vertex_of_parabola :
  ∀ x y : ℝ, y = x^2 + 2*x - 3 → ∃ (vx vy : ℝ), (vx = -1) ∧ (vy = -4) :=
by
  sorry

theorem minimum_value_for_x_ge_2 :
  ∀ x : ℝ, x ≥ 2 → y = x^2 + 2*x - 3 → ∃ (min_val : ℝ), min_val = 5 :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_minimum_value_for_x_ge_2_l1645_164529


namespace NUMINAMATH_GPT_Rebecca_worked_56_l1645_164518

-- Define the conditions
variables (x : ℕ)
def Toby_hours := 2 * x - 10
def Rebecca_hours := Toby_hours - 8
def Total_hours := x + Toby_hours + Rebecca_hours

-- Theorem stating that under the given conditions, Rebecca worked 56 hours
theorem Rebecca_worked_56 
  (h : Total_hours = 157) 
  (hx : x = 37) : Rebecca_hours = 56 :=
by sorry

end NUMINAMATH_GPT_Rebecca_worked_56_l1645_164518


namespace NUMINAMATH_GPT_nabla_example_l1645_164520

def nabla (a b : ℕ) : ℕ := 2 + b ^ a

theorem nabla_example : nabla (nabla 1 2) 3 = 83 :=
  by
  sorry

end NUMINAMATH_GPT_nabla_example_l1645_164520


namespace NUMINAMATH_GPT_max_children_l1645_164512

/-- Total quantities -/
def total_apples : ℕ := 55
def total_cookies : ℕ := 114
def total_chocolates : ℕ := 83

/-- Leftover quantities after distribution -/
def leftover_apples : ℕ := 3
def leftover_cookies : ℕ := 10
def leftover_chocolates : ℕ := 5

/-- Distributed quantities -/
def distributed_apples : ℕ := total_apples - leftover_apples
def distributed_cookies : ℕ := total_cookies - leftover_cookies
def distributed_chocolates : ℕ := total_chocolates - leftover_chocolates

/-- The theorem states the maximum number of children -/
theorem max_children : Nat.gcd (Nat.gcd distributed_apples distributed_cookies) distributed_chocolates = 26 :=
by
  sorry

end NUMINAMATH_GPT_max_children_l1645_164512


namespace NUMINAMATH_GPT_find_exponent_l1645_164598

theorem find_exponent (y : ℝ) (exponent : ℝ) :
  (12^1 * 6^exponent / 432 = y) → (y = 36) → (exponent = 3) :=
by 
  intros h₁ h₂ 
  sorry

end NUMINAMATH_GPT_find_exponent_l1645_164598


namespace NUMINAMATH_GPT_problem_l1645_164558

theorem problem (a b c d : ℝ) 
  (h1 : a + b + c = 5) 
  (h2 : a + b + d = 1) 
  (h3 : a + c + d = 16) 
  (h4 : b + c + d = 9) : 
  a * b + c * d = 734 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_problem_l1645_164558


namespace NUMINAMATH_GPT_n_values_satisfy_condition_l1645_164528

-- Define the exponential functions
def exp1 (n : ℤ) : ℚ := (-1/2) ^ n
def exp2 (n : ℤ) : ℚ := (-1/5) ^ n

-- Define the set of possible values for n
def valid_n : List ℤ := [-2, -1, 0, 1, 2, 3]

-- Define the condition for n to satisfy the inequality
def satisfies_condition (n : ℤ) : Prop := exp1 n > exp2 n

-- Prove that the only values of n that satisfy the condition are -1 and 2
theorem n_values_satisfy_condition :
  ∀ n ∈ valid_n, satisfies_condition n ↔ (n = -1 ∨ n = 2) :=
by
  intro n
  sorry

end NUMINAMATH_GPT_n_values_satisfy_condition_l1645_164528


namespace NUMINAMATH_GPT_fraction_value_l1645_164532

theorem fraction_value (a b : ℝ) (h : 1 / a - 1 / b = 4) : 
    (a - 2 * a * b - b) / (2 * a + 7 * a * b - 2 * b) = 6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_value_l1645_164532


namespace NUMINAMATH_GPT_total_flowers_eaten_l1645_164550

-- Definitions based on conditions
def num_bugs : ℕ := 3
def flowers_per_bug : ℕ := 2

-- Statement asserting the total number of flowers eaten
theorem total_flowers_eaten : num_bugs * flowers_per_bug = 6 := by
  sorry

end NUMINAMATH_GPT_total_flowers_eaten_l1645_164550


namespace NUMINAMATH_GPT_max_pqrs_squared_l1645_164569

theorem max_pqrs_squared (p q r s : ℝ)
  (h1 : p + q = 18)
  (h2 : pq + r + s = 85)
  (h3 : pr + qs = 190)
  (h4 : rs = 120) :
  p^2 + q^2 + r^2 + s^2 ≤ 886 :=
sorry

end NUMINAMATH_GPT_max_pqrs_squared_l1645_164569


namespace NUMINAMATH_GPT_sum_odd_numbers_to_2019_is_correct_l1645_164510

-- Define the sequence sum
def sum_first_n_odd (n : ℕ) : ℕ := n * n

-- Define the specific problem
theorem sum_odd_numbers_to_2019_is_correct : sum_first_n_odd 1010 = 1020100 :=
by
  -- Sorry placeholder for the proof
  sorry

end NUMINAMATH_GPT_sum_odd_numbers_to_2019_is_correct_l1645_164510


namespace NUMINAMATH_GPT_vacation_animals_total_l1645_164557

noncomputable def lisa := 40
noncomputable def alex := lisa / 2
noncomputable def jane := alex + 10
noncomputable def rick := 3 * jane
noncomputable def tim := 2 * rick
noncomputable def you := 5 * tim
noncomputable def total_animals := lisa + alex + jane + rick + tim + you

theorem vacation_animals_total : total_animals = 1260 := by
  sorry

end NUMINAMATH_GPT_vacation_animals_total_l1645_164557


namespace NUMINAMATH_GPT_danny_total_bottle_caps_l1645_164587

def danny_initial_bottle_caps : ℕ := 37
def danny_found_bottle_caps : ℕ := 18

theorem danny_total_bottle_caps : danny_initial_bottle_caps + danny_found_bottle_caps = 55 := by
  sorry

end NUMINAMATH_GPT_danny_total_bottle_caps_l1645_164587


namespace NUMINAMATH_GPT_BoatsRUs_total_canoes_l1645_164533

def totalCanoesBuiltByJuly (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem BoatsRUs_total_canoes :
  totalCanoesBuiltByJuly 5 3 7 = 5465 :=
by
  sorry

end NUMINAMATH_GPT_BoatsRUs_total_canoes_l1645_164533


namespace NUMINAMATH_GPT_problem1_problem2_l1645_164551

variable (x y : ℝ)

-- Problem 1
theorem problem1 : (x + y) ^ 2 + x * (x - 2 * y) = 2 * x ^ 2 + y ^ 2 := by
  sorry

variable (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 0) -- to ensure the denominators are non-zero

-- Problem 2
theorem problem2 : (x ^ 2 - 6 * x + 9) / (x - 2) / (x + 2 - (3 * x - 4) / (x - 2)) = (x - 3) / x := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1645_164551


namespace NUMINAMATH_GPT_Lucy_total_groceries_l1645_164555

theorem Lucy_total_groceries :
  let packs_of_cookies := 12
  let packs_of_noodles := 16
  let boxes_of_cereals := 5
  let packs_of_crackers := 45
  (packs_of_cookies + packs_of_noodles + packs_of_crackers + boxes_of_cereals) = 78 :=
by
  sorry

end NUMINAMATH_GPT_Lucy_total_groceries_l1645_164555


namespace NUMINAMATH_GPT_ceil_sqrt_180_eq_14_l1645_164506

theorem ceil_sqrt_180_eq_14
  (h : 13 < Real.sqrt 180 ∧ Real.sqrt 180 < 14) :
  Int.ceil (Real.sqrt 180) = 14 :=
  sorry

end NUMINAMATH_GPT_ceil_sqrt_180_eq_14_l1645_164506


namespace NUMINAMATH_GPT_problem1_problem2_l1645_164594

-- Problem 1
theorem problem1 : -9 + (-4 * 5) = -29 :=
by
  sorry

-- Problem 2
theorem problem2 : (-(6) * -2) / (2 / 3) = -18 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1645_164594


namespace NUMINAMATH_GPT_coordinates_of_P_l1645_164513

theorem coordinates_of_P (m : ℝ) (P : ℝ × ℝ) :
  P = (2 * m, m + 8) ∧ 2 * m = 0 → P = (0, 8) := by
  intros hm
  sorry

end NUMINAMATH_GPT_coordinates_of_P_l1645_164513


namespace NUMINAMATH_GPT_compare_logs_l1645_164584

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem compare_logs (a b c : ℝ) (h1 : a = log_base 4 1.25) (h2 : b = log_base 5 1.2) (h3 : c = log_base 4 8) :
  c > a ∧ a > b :=
by
  sorry

end NUMINAMATH_GPT_compare_logs_l1645_164584


namespace NUMINAMATH_GPT_magnitude_z_l1645_164527

open Complex

theorem magnitude_z
  (z w : ℂ)
  (h1 : abs (2 * z - w) = 25)
  (h2 : abs (z + 2 * w) = 5)
  (h3 : abs (z + w) = 2) : abs z = 9 := 
by 
  sorry

end NUMINAMATH_GPT_magnitude_z_l1645_164527


namespace NUMINAMATH_GPT_compute_expression_l1645_164576

theorem compute_expression : 2 * ((3 + 7) ^ 2 + (3 ^ 2 + 7 ^ 2)) = 316 := 
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1645_164576


namespace NUMINAMATH_GPT_charge_move_increases_energy_l1645_164540

noncomputable def energy_increase_when_charge_moved : ℝ :=
  let initial_energy := 15
  let energy_per_pair := initial_energy / 3
  let new_energy_AB := energy_per_pair
  let new_energy_AC := 2 * energy_per_pair
  let new_energy_BC := 2 * energy_per_pair
  let final_energy := new_energy_AB + new_energy_AC + new_energy_BC
  final_energy - initial_energy

theorem charge_move_increases_energy :
  energy_increase_when_charge_moved = 10 :=
by
  sorry

end NUMINAMATH_GPT_charge_move_increases_energy_l1645_164540


namespace NUMINAMATH_GPT_valid_outfits_count_l1645_164565

noncomputable def number_of_valid_outfits (shirt_count: ℕ) (pant_colors: List String) (hat_count: ℕ) : ℕ :=
  let total_combinations := shirt_count * (pant_colors.length) * hat_count
  let matching_outfits := List.length (List.filter (λ c => c ∈ pant_colors) ["tan", "black", "blue", "gray"])
  total_combinations - matching_outfits

theorem valid_outfits_count :
    number_of_valid_outfits 8 ["tan", "black", "blue", "gray"] 8 = 252 := by
  sorry

end NUMINAMATH_GPT_valid_outfits_count_l1645_164565


namespace NUMINAMATH_GPT_top_leftmost_rectangle_is_B_l1645_164553

-- Definitions for the side lengths of each rectangle
def A_w : ℕ := 6
def A_x : ℕ := 2
def A_y : ℕ := 7
def A_z : ℕ := 10

def B_w : ℕ := 2
def B_x : ℕ := 1
def B_y : ℕ := 4
def B_z : ℕ := 8

def C_w : ℕ := 5
def C_x : ℕ := 11
def C_y : ℕ := 6
def C_z : ℕ := 3

def D_w : ℕ := 9
def D_x : ℕ := 7
def D_y : ℕ := 5
def D_z : ℕ := 9

def E_w : ℕ := 11
def E_x : ℕ := 4
def E_y : ℕ := 9
def E_z : ℕ := 1

-- The problem statement to prove
theorem top_leftmost_rectangle_is_B : 
  (B_w = 2 ∧ B_y = 4) ∧ 
  (A_w = 6 ∨ D_w = 9 ∨ C_w = 5 ∨ E_w = 11) ∧
  (A_y = 7 ∨ D_y = 5 ∨ C_y = 6 ∨ E_y = 9) → 
  (B_w = 2 ∧ ∀ w : ℕ, w = 6 ∨ w = 5 ∨ w = 9 ∨ w = 11 → B_w < w) :=
by {
  -- skipping the proof
  sorry
}

end NUMINAMATH_GPT_top_leftmost_rectangle_is_B_l1645_164553


namespace NUMINAMATH_GPT_smallest_five_digit_number_divisible_by_first_five_primes_l1645_164500

theorem smallest_five_digit_number_divisible_by_first_five_primes : 
  ∃ n, (n >= 10000) ∧ (n < 100000) ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 11550 :=
by
  sorry

end NUMINAMATH_GPT_smallest_five_digit_number_divisible_by_first_five_primes_l1645_164500


namespace NUMINAMATH_GPT_power_inequality_l1645_164592

variable (a b c : ℝ)
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_ineq : a^19 / b^19 + b^19 / c^19 + c^19 / a^19 ≤ a^19 / c^19 + b^19 / a^19 + c^19 / b^19)

theorem power_inequality :
  a^20 / b^20 + b^20 / c^20 + c^20 / a^20 ≤ a^20 / c^20 + b^20 / a^20 + c^20 / b^20 :=
by
  sorry

end NUMINAMATH_GPT_power_inequality_l1645_164592


namespace NUMINAMATH_GPT_eight_odot_six_eq_ten_l1645_164585

-- Define the operation ⊙ as given in the problem statement
def operation (a b : ℕ) : ℕ := a + (3 * a) / (2 * b)

-- State the theorem to prove
theorem eight_odot_six_eq_ten : operation 8 6 = 10 :=
by
  -- Here you will provide the proof, but we skip it with sorry
  sorry

end NUMINAMATH_GPT_eight_odot_six_eq_ten_l1645_164585


namespace NUMINAMATH_GPT_minimize_shoes_l1645_164547

-- Definitions for inhabitants, one-legged inhabitants, and shoe calculations
def total_inhabitants := 10000
def P (percent_one_legged : ℕ) := (percent_one_legged * total_inhabitants) / 100
def non_one_legged (percent_one_legged : ℕ) := total_inhabitants - (P percent_one_legged)
def non_one_legged_with_shoes (percent_one_legged : ℕ) := (non_one_legged percent_one_legged) / 2
def shoes_needed (percent_one_legged : ℕ) := 
  (P percent_one_legged) + 2 * (non_one_legged_with_shoes percent_one_legged)

-- Theorem to prove that 100% one-legged minimizes the shoes required
theorem minimize_shoes : ∀ (percent_one_legged : ℕ), shoes_needed percent_one_legged = total_inhabitants → percent_one_legged = 100 :=
by
  intros percent_one_legged h
  sorry

end NUMINAMATH_GPT_minimize_shoes_l1645_164547


namespace NUMINAMATH_GPT_blue_lipstick_count_l1645_164537

def total_students : Nat := 200

def colored_lipstick_students (total : Nat) : Nat :=
  total / 2

def red_lipstick_students (colored : Nat) : Nat :=
  colored / 4

def blue_lipstick_students (red : Nat) : Nat :=
  red / 5

theorem blue_lipstick_count :
  blue_lipstick_students (red_lipstick_students (colored_lipstick_students total_students)) = 5 := 
sorry

end NUMINAMATH_GPT_blue_lipstick_count_l1645_164537


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1645_164545

theorem necessary_but_not_sufficient
  (x y : ℝ) :
  (x^2 + y^2 ≤ 2*x → x^2 + y^2 ≤ 4) ∧ ¬ (x^2 + y^2 ≤ 4 → x^2 + y^2 ≤ 2*x) :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1645_164545


namespace NUMINAMATH_GPT_part1_even_function_part2_two_distinct_zeros_l1645_164568

noncomputable def f (x a : ℝ) : ℝ := (4^x + a) / 2^x
noncomputable def g (x a : ℝ) : ℝ := f x a - (a + 1)

theorem part1_even_function (a : ℝ) :
  (∀ x : ℝ, f (-x) a = f x a) ↔ a = 1 :=
sorry

theorem part2_two_distinct_zeros (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ -1 ≤ x1 ∧ x1 ≤ 1 ∧ -1 ≤ x2 ∧ x2 ≤ 1 ∧ g x1 a = 0 ∧ g x2 a = 0) ↔ (a ∈ Set.Icc (1/2) 1 ∪ Set.Icc 1 2) :=
sorry

end NUMINAMATH_GPT_part1_even_function_part2_two_distinct_zeros_l1645_164568


namespace NUMINAMATH_GPT_min_value_expression_l1645_164572

theorem min_value_expression (x y : ℝ) (h1 : x < 0) (h2 : y < 0) (h3 : x + y = -1) :
  xy + (1 / xy) = 17 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l1645_164572


namespace NUMINAMATH_GPT_mark_paintable_area_l1645_164562

theorem mark_paintable_area :
  let num_bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let area_excluded := 70
  let area_wall_one_bedroom := 2 * (length * height) + 2 * (width * height) - area_excluded 
  (area_wall_one_bedroom * num_bedrooms) = 1520 :=
by
  sorry

end NUMINAMATH_GPT_mark_paintable_area_l1645_164562


namespace NUMINAMATH_GPT_chessboard_tiling_impossible_l1645_164577

theorem chessboard_tiling_impossible :
  ¬ ∃ (cover : (Fin 5 × Fin 7 → Prop)), 
    (cover (0, 3) = false) ∧
    (∀ i j, (cover (i, j) → cover (i + 1, j) ∨ cover (i, j + 1)) ∧
             ∀ x y z w, cover (x, y) → cover (z, w) → (x ≠ z ∨ y ≠ w)) :=
sorry

end NUMINAMATH_GPT_chessboard_tiling_impossible_l1645_164577


namespace NUMINAMATH_GPT_andrew_purchase_grapes_l1645_164530

theorem andrew_purchase_grapes (G : ℕ) (h : 70 * G + 495 = 1055) : G = 8 :=
by
  sorry

end NUMINAMATH_GPT_andrew_purchase_grapes_l1645_164530


namespace NUMINAMATH_GPT_harry_weekly_earnings_l1645_164522

def dogs_walked_MWF := 7
def dogs_walked_Tue := 12
def dogs_walked_Thu := 9
def pay_per_dog := 5

theorem harry_weekly_earnings : 
  dogs_walked_MWF * pay_per_dog * 3 + dogs_walked_Tue * pay_per_dog + dogs_walked_Thu * pay_per_dog = 210 :=
by
  sorry

end NUMINAMATH_GPT_harry_weekly_earnings_l1645_164522


namespace NUMINAMATH_GPT_interchange_digits_product_l1645_164566

-- Definition of the proof problem
theorem interchange_digits_product (n a b k : ℤ) (h1 : n = 10 * a + b) (h2 : n = (k + 1) * (a + b)) :
  ∃ x : ℤ, (10 * b + a) = x * (a + b) ∧ x = 10 - k :=
by
  existsi (10 - k)
  sorry

end NUMINAMATH_GPT_interchange_digits_product_l1645_164566


namespace NUMINAMATH_GPT_tammy_average_speed_second_day_l1645_164597

theorem tammy_average_speed_second_day :
  ∃ v t : ℝ, 
  t + (t - 2) + (t + 1) = 20 ∧
  v * t + (v + 0.5) * (t - 2) + (v - 0.5) * (t + 1) = 80 ∧
  (v + 0.5) = 4.575 :=
by 
  sorry

end NUMINAMATH_GPT_tammy_average_speed_second_day_l1645_164597


namespace NUMINAMATH_GPT_surface_area_implies_side_length_diagonal_l1645_164571

noncomputable def cube_side_length_diagonal (A : ℝ) := 
  A = 864 → ∃ s d : ℝ, s = 12 ∧ d = 12 * Real.sqrt 3

theorem surface_area_implies_side_length_diagonal : 
  cube_side_length_diagonal 864 := by
  sorry

end NUMINAMATH_GPT_surface_area_implies_side_length_diagonal_l1645_164571


namespace NUMINAMATH_GPT_problem_solution_l1645_164526

noncomputable def expr := 
  (Real.tan (Real.pi / 15) - Real.sqrt 3) / ((4 * (Real.cos (Real.pi / 15))^2 - 2) * Real.sin (Real.pi / 15))

theorem problem_solution : expr = -4 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l1645_164526


namespace NUMINAMATH_GPT_least_subtraction_for_divisibility_l1645_164531

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 964807) : ∃ k, k = 7 ∧ (n - k) % 8 = 0 :=
by 
  sorry

end NUMINAMATH_GPT_least_subtraction_for_divisibility_l1645_164531


namespace NUMINAMATH_GPT_star_polygon_points_l1645_164542

theorem star_polygon_points (n : ℕ) (A B : ℕ → ℝ) 
  (h_angles_congruent_A : ∀ i j, A i = A j)
  (h_angles_congruent_B : ∀ i j, B i = B j)
  (h_angle_relation : ∀ i, A i = B i - 15) :
  n = 24 :=
by
  sorry

end NUMINAMATH_GPT_star_polygon_points_l1645_164542


namespace NUMINAMATH_GPT_meters_conversion_equivalence_l1645_164589

-- Define the conditions
def meters_to_decimeters (m : ℝ) : ℝ := m * 10
def meters_to_centimeters (m : ℝ) : ℝ := m * 100

-- State the problem
theorem meters_conversion_equivalence :
  7.34 = 7 + (meters_to_decimeters 0.3) / 10 + (meters_to_centimeters 0.04) / 100 :=
sorry

end NUMINAMATH_GPT_meters_conversion_equivalence_l1645_164589


namespace NUMINAMATH_GPT_set_intersection_l1645_164574

open Set

def U := {x : ℝ | True}
def A := {x : ℝ | x^2 - 2 * x < 0}
def B := {x : ℝ | x - 1 ≥ 0}
def complement (U B : Set ℝ) := {x : ℝ | x ∉ B}
def intersection (A B : Set ℝ) := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem set_intersection :
  intersection A (complement U B) = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_set_intersection_l1645_164574


namespace NUMINAMATH_GPT_number_of_ways_to_choose_bases_l1645_164507

-- Definitions of the conditions
def num_students : Nat := 4
def num_bases : Nat := 3

-- The main statement that we need to prove
theorem number_of_ways_to_choose_bases : (num_bases ^ num_students) = 81 := by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_choose_bases_l1645_164507


namespace NUMINAMATH_GPT_rectangle_perimeter_eq_30sqrt10_l1645_164563

theorem rectangle_perimeter_eq_30sqrt10 (A : ℝ) (l : ℝ) (w : ℝ) 
  (hA : A = 500) (hlw : l = 2 * w) (hArea : A = l * w) : 
  2 * (l + w) = 30 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_eq_30sqrt10_l1645_164563


namespace NUMINAMATH_GPT_find_a3_l1645_164556

variable (a_n : ℕ → ℤ) (a1 a4 a5 : ℤ)
variable (d : ℤ := -2)

-- Conditions
axiom h1 : ∀ n : ℕ, a_n (n + 1) = a_n n + d
axiom h2 : a4 = a1 + 3 * d
axiom h3 : a5 = a1 + 4 * d
axiom h4 : a4 * a4 = a1 * a5

-- Question to prove
theorem find_a3 : (a_n 3) = 5 := by
  sorry

end NUMINAMATH_GPT_find_a3_l1645_164556


namespace NUMINAMATH_GPT_valid_grid_count_l1645_164596

def is_adjacent (i j : ℕ) (n : ℕ) : Prop :=
  (i = j + 1 ∨ i + 1 = j ∨ (i = n - 1 ∧ j = 0) ∨ (i = 0 ∧ j = n - 1))

def valid_grid (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 4 ∧ 0 ≤ j ∧ j < 4 →
         (is_adjacent i (i+1) 4 → grid i (i+1) * grid i (i+1) = 0) ∧ 
         (is_adjacent j (j+1) 4 → grid (j+1) j * grid (j+1) j = 0)

theorem valid_grid_count : 
  ∃ s : ℕ, s = 1234 ∧
    (∃ grid : ℕ → ℕ → ℕ, valid_grid grid) :=
sorry

end NUMINAMATH_GPT_valid_grid_count_l1645_164596


namespace NUMINAMATH_GPT_minimum_area_triangle_AOB_l1645_164581

theorem minimum_area_triangle_AOB : 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) ∧ (∀ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 / a + 2 / b = 1) → (1/2 * a * b ≥ 12)) := 
sorry

end NUMINAMATH_GPT_minimum_area_triangle_AOB_l1645_164581


namespace NUMINAMATH_GPT_solution_set_for_absolute_value_inequality_l1645_164554

theorem solution_set_for_absolute_value_inequality :
  {x : ℝ | |2 * x - 1| ≤ 3} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by 
  sorry

end NUMINAMATH_GPT_solution_set_for_absolute_value_inequality_l1645_164554


namespace NUMINAMATH_GPT_percent_of_z_l1645_164535

variable (x y z : ℝ)

theorem percent_of_z :
  x = 1.20 * y →
  y = 0.40 * z →
  x = 0.48 * z :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_percent_of_z_l1645_164535


namespace NUMINAMATH_GPT_playground_children_count_l1645_164560

theorem playground_children_count (boys girls : ℕ) (h_boys : boys = 27) (h_girls : girls = 35) : boys + girls = 62 := by
  sorry

end NUMINAMATH_GPT_playground_children_count_l1645_164560


namespace NUMINAMATH_GPT_bases_for_204_base_b_l1645_164534

theorem bases_for_204_base_b (b : ℕ) : (∃ n : ℤ, 2 * b^2 + 4 = n^2) ↔ b = 4 ∨ b = 6 ∨ b = 8 ∨ b = 10 :=
by
  sorry

end NUMINAMATH_GPT_bases_for_204_base_b_l1645_164534


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1645_164543

theorem solve_equation_1 (x : ℝ) : (x + 2) ^ 2 = 3 * (x + 2) ↔ x = -2 ∨ x = 1 := by
  sorry

theorem solve_equation_2 (x : ℝ) : x ^ 2 - 8 * x + 3 = 0 ↔ x = 4 + Real.sqrt 13 ∨ x = 4 - Real.sqrt 13 := by
  sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1645_164543


namespace NUMINAMATH_GPT_third_root_of_cubic_equation_l1645_164544

-- Definitions
variable (a b : ℚ) -- We use rational numbers due to the fractions involved
def cubic_equation (x : ℚ) : ℚ := a * x^3 + (a + 3 * b) * x^2 + (2 * b - 4 * a) * x + (10 - a)

-- Conditions
axiom h1 : cubic_equation a b (-1) = 0
axiom h2 : cubic_equation a b 4 = 0

-- The theorem we aim to prove
theorem third_root_of_cubic_equation : ∃ (c : ℚ), c = -62 / 19 ∧ cubic_equation a b c = 0 :=
sorry

end NUMINAMATH_GPT_third_root_of_cubic_equation_l1645_164544


namespace NUMINAMATH_GPT_box_height_l1645_164591

theorem box_height (x : ℝ) (hx : x + 5 = 10)
  (surface_area : 2*x^2 + 4*x*(x + 5) ≥ 150) : x + 5 = 10 :=
sorry

end NUMINAMATH_GPT_box_height_l1645_164591


namespace NUMINAMATH_GPT_age_of_older_friend_l1645_164502

theorem age_of_older_friend (a b : ℕ) (h1 : a - b = 2) (h2 : a + b = 74) : a = 38 :=
by
  sorry

end NUMINAMATH_GPT_age_of_older_friend_l1645_164502


namespace NUMINAMATH_GPT_problem1_l1645_164516

theorem problem1 (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 15) : 
  (x * y = 5) ∧ ((x - y)^2 = 5) :=
by
  sorry

end NUMINAMATH_GPT_problem1_l1645_164516


namespace NUMINAMATH_GPT_find_value_of_15b_minus_2a_l1645_164580

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if 1 ≤ x ∧ x < 2 then x + a / x
else if 2 ≤ x ∧ x ≤ 3 then b * x - 3
else 0

theorem find_value_of_15b_minus_2a (a b : ℝ)
  (h_periodic : ∀ x : ℝ, f x a b = f (x + 2) a b)
  (h_condition : f (7 / 2) a b = f (-7 / 2) a b) :
  15 * b - 2 * a = 41 :=
sorry

end NUMINAMATH_GPT_find_value_of_15b_minus_2a_l1645_164580


namespace NUMINAMATH_GPT_perpendicular_planes_l1645_164538

variables (b c : Line) (α β : Plane)
axiom line_in_plane (b : Line) (α : Plane) : Prop -- b ⊆ α
axiom line_parallel_plane (c : Line) (α : Plane) : Prop -- c ∥ α
axiom lines_are_skew (b c : Line) : Prop -- b and c could be skew
axiom planes_are_perpendicular (α β : Plane) : Prop -- α ⊥ β
axiom line_perpendicular_plane (c : Line) (β : Plane) : Prop -- c ⊥ β

theorem perpendicular_planes (hcα : line_in_plane c α) (hcβ : line_perpendicular_plane c β) : planes_are_perpendicular α β := 
sorry

end NUMINAMATH_GPT_perpendicular_planes_l1645_164538


namespace NUMINAMATH_GPT_remainder_of_product_l1645_164541

theorem remainder_of_product (a b n : ℕ) (ha : a % n = 7) (hb : b % n = 1) :
  ((a * b) % n) = 7 :=
by
  -- Definitions as per the conditions
  let a := 63
  let b := 65
  let n := 8
  /- Now prove the statement -/
  sorry

end NUMINAMATH_GPT_remainder_of_product_l1645_164541


namespace NUMINAMATH_GPT_power_function_result_l1645_164590
noncomputable def f (x : ℝ) (k : ℝ) (n : ℝ) : ℝ := k * x ^ n

theorem power_function_result (k n : ℝ) (h1 : f 27 k n = 3) : f 8 k (1/3) = 2 :=
by 
  sorry

end NUMINAMATH_GPT_power_function_result_l1645_164590


namespace NUMINAMATH_GPT_intersection_eq_l1645_164514

-- Define the sets M and N using the given conditions
def M : Set ℝ := { x | x < 1 / 2 }
def N : Set ℝ := { x | x ≥ -4 }

-- The goal is to prove that the intersection of M and N is { x | -4 ≤ x < 1 / 2 }
theorem intersection_eq : M ∩ N = { x | -4 ≤ x ∧ x < (1 / 2) } :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l1645_164514


namespace NUMINAMATH_GPT_lowest_exam_score_l1645_164552

theorem lowest_exam_score 
  (first_exam_score : ℕ := 90) 
  (second_exam_score : ℕ := 108) 
  (third_exam_score : ℕ := 102) 
  (max_score_per_exam : ℕ := 120) 
  (desired_average : ℕ := 100) 
  (total_exams : ℕ := 5) 
  (total_score_needed : ℕ := desired_average * total_exams) : 
  ∃ (lowest_score : ℕ), lowest_score = 80 :=
by
  sorry

end NUMINAMATH_GPT_lowest_exam_score_l1645_164552


namespace NUMINAMATH_GPT_only_solution_l1645_164511

theorem only_solution (a b c : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
    (h_le : a ≤ b ∧ b ≤ c) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_div_a2b : a^3 + b^3 + c^3 % (a^2 * b) = 0)
    (h_div_b2c : a^3 + b^3 + c^3 % (b^2 * c) = 0)
    (h_div_c2a : a^3 + b^3 + c^3 % (c^2 * a) = 0) : 
    a = 1 ∧ b = 1 ∧ c = 1 :=
  by
  sorry

end NUMINAMATH_GPT_only_solution_l1645_164511


namespace NUMINAMATH_GPT_silverware_probability_l1645_164539

def numWaysTotal (totalPieces : ℕ) (choosePieces : ℕ) : ℕ :=
  Nat.choose totalPieces choosePieces

def numWaysForks (forks : ℕ) (chooseForks : ℕ) : ℕ :=
  Nat.choose forks chooseForks

def numWaysSpoons (spoons : ℕ) (chooseSpoons : ℕ) : ℕ :=
  Nat.choose spoons chooseSpoons

def numWaysKnives (knives : ℕ) (chooseKnives : ℕ) : ℕ :=
  Nat.choose knives chooseKnives

def favorableOutcomes (forks : ℕ) (spoons : ℕ) (knives : ℕ) : ℕ :=
  numWaysForks forks 2 * numWaysSpoons spoons 1 * numWaysKnives knives 1

def probability (totalWays : ℕ) (favorableWays : ℕ) : ℚ :=
  favorableWays / totalWays

theorem silverware_probability :
  probability (numWaysTotal 18 4) (favorableOutcomes 5 7 6) = 7 / 51 := by
  sorry

end NUMINAMATH_GPT_silverware_probability_l1645_164539


namespace NUMINAMATH_GPT_solve_for_x_l1645_164567

theorem solve_for_x (x y : ℝ) (h1 : 9^y = x^12) (h2 : y = 6) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1645_164567


namespace NUMINAMATH_GPT_problem_statement_l1645_164588

variable (a b : Type) [LinearOrder a] [LinearOrder b]
variable (α β : Type) [LinearOrder α] [LinearOrder β]

-- Given conditions
def line_perpendicular_to_plane (l : Type) (p : Type) [LinearOrder l] [LinearOrder p] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

def lines_parallel (l1 : Type) (l2 : Type) [LinearOrder l1] [LinearOrder l2] : Prop :=
True -- This is a placeholder. Actual geometry definition required.

theorem problem_statement (a b α : Type) [LinearOrder a] [LinearOrder b] [LinearOrder α]
(val_perp1 : line_perpendicular_to_plane a α)
(val_perp2 : line_perpendicular_to_plane b α)
: lines_parallel a b :=
sorry

end NUMINAMATH_GPT_problem_statement_l1645_164588


namespace NUMINAMATH_GPT_alexis_shirt_expense_l1645_164549

theorem alexis_shirt_expense :
  let B := 200
  let E_pants := 46
  let E_coat := 38
  let E_socks := 11
  let E_belt := 18
  let E_shoes := 41
  let L := 16
  let S := B - (E_pants + E_coat + E_socks + E_belt + E_shoes + L)
  S = 30 :=
by
  sorry

end NUMINAMATH_GPT_alexis_shirt_expense_l1645_164549


namespace NUMINAMATH_GPT_initial_average_weight_l1645_164504

theorem initial_average_weight 
    (W : ℝ)
    (a b c d e : ℝ)
    (h1 : (a + b + c) / 3 = W)
    (h2 : (a + b + c + d) / 4 = W)
    (h3 : (b + c + d + (d + 3)) / 4 = 68)
    (h4 : a = 81) :
    W = 70 := 
sorry

end NUMINAMATH_GPT_initial_average_weight_l1645_164504


namespace NUMINAMATH_GPT_find_a_and_b_nth_equation_conjecture_l1645_164523

theorem find_a_and_b {a b : ℤ} (h1 : 1^2 + 2^2 - 3^2 = 1 * a - b)
                                        (h2 : 2^2 + 3^2 - 4^2 = 2 * 0 - b)
                                        (h3 : 3^2 + 4^2 - 5^2 = 3 * 1 - b)
                                        (h4 : 4^2 + 5^2 - 6^2 = 4 * 2 - b):
    a = -1 ∧ b = 3 :=
    sorry

theorem nth_equation_conjecture (n : ℤ) :
  n^2 + (n+1)^2 - (n+2)^2 = n * (n-2) - 3 :=
  sorry

end NUMINAMATH_GPT_find_a_and_b_nth_equation_conjecture_l1645_164523


namespace NUMINAMATH_GPT_combined_annual_income_l1645_164505

-- Define the given conditions and verify the combined annual income
def A_ratio : ℤ := 5
def B_ratio : ℤ := 2
def C_ratio : ℤ := 3
def D_ratio : ℤ := 4

def C_income : ℤ := 15000
def B_income : ℤ := 16800
def A_income : ℤ := 25000
def D_income : ℤ := 21250

theorem combined_annual_income :
  (A_income + B_income + C_income + D_income) * 12 = 936600 :=
by
  sorry

end NUMINAMATH_GPT_combined_annual_income_l1645_164505


namespace NUMINAMATH_GPT_complement_intersection_l1645_164508

-- Definitions of sets and complements
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 5}
def C_U_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}
def C_U_B : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- The proof statement
theorem complement_intersection {U A B C_U_A C_U_B : Set ℕ} (h1 : U = {1, 2, 3, 4, 5}) (h2 : A = {1, 2, 3}) (h3 : B = {2, 5}) (h4 : C_U_A = {x | x ∈ U ∧ x ∉ A}) (h5 : C_U_B = {x | x ∈ U ∧ x ∉ B}) : 
  (C_U_A ∩ C_U_B) = {4} :=
by 
  sorry

end NUMINAMATH_GPT_complement_intersection_l1645_164508


namespace NUMINAMATH_GPT_cos_beta_eq_sqrt10_over_10_l1645_164546

-- Define the conditions and the statement
theorem cos_beta_eq_sqrt10_over_10 
  (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (h_tan : Real.tan α = 2)
  (h_sin_sum : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 :=
sorry

end NUMINAMATH_GPT_cos_beta_eq_sqrt10_over_10_l1645_164546


namespace NUMINAMATH_GPT_sum_of_digits_a_l1645_164586

def a : ℕ := 10^10 - 47

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_a : sum_of_digits a = 81 := 
  by 
    sorry

end NUMINAMATH_GPT_sum_of_digits_a_l1645_164586


namespace NUMINAMATH_GPT_equal_striped_areas_l1645_164515

theorem equal_striped_areas (A B C D : ℝ) (h_AD_DB : D = A + B) (h_CD2 : C^2 = A * B) :
  (π * C^2 / 4 = π * B^2 / 8 - π * A^2 / 8 - π * D^2 / 8) := 
sorry

end NUMINAMATH_GPT_equal_striped_areas_l1645_164515


namespace NUMINAMATH_GPT_trisha_hourly_wage_l1645_164509

theorem trisha_hourly_wage (annual_take_home_pay : ℝ) (percent_withheld : ℝ)
  (hours_per_week : ℝ) (weeks_per_year : ℝ) (hourly_wage : ℝ) :
  annual_take_home_pay = 24960 ∧ 
  percent_withheld = 0.20 ∧ 
  hours_per_week = 40 ∧ 
  weeks_per_year = 52 ∧ 
  hourly_wage = (annual_take_home_pay / (0.80 * (hours_per_week * weeks_per_year))) → 
  hourly_wage = 15 :=
by sorry

end NUMINAMATH_GPT_trisha_hourly_wage_l1645_164509


namespace NUMINAMATH_GPT_cost_of_two_other_puppies_l1645_164559

theorem cost_of_two_other_puppies (total_cost : ℕ) (sale_price : ℕ) (num_puppies : ℕ) (num_sale_puppies : ℕ) (remaining_puppies_cost : ℕ) :
  total_cost = 800 →
  sale_price = 150 →
  num_puppies = 5 →
  num_sale_puppies = 3 →
  remaining_puppies_cost = (total_cost - num_sale_puppies * sale_price) →
  (remaining_puppies_cost / (num_puppies - num_sale_puppies)) = 175 :=
by
  intros
  sorry

end NUMINAMATH_GPT_cost_of_two_other_puppies_l1645_164559


namespace NUMINAMATH_GPT_seahorse_penguin_ratio_l1645_164578

theorem seahorse_penguin_ratio :
  ∃ S P : ℕ, S = 70 ∧ P = S + 85 ∧ Nat.gcd 70 (S + 85) = 5 ∧ 70 / Nat.gcd 70 (S + 85) = 14 ∧ (S + 85) / Nat.gcd 70 (S + 85) = 31 :=
by
  sorry

end NUMINAMATH_GPT_seahorse_penguin_ratio_l1645_164578


namespace NUMINAMATH_GPT_cost_of_sculpture_cny_l1645_164519

def exchange_rate_usd_to_nad := 8 -- 1 USD = 8 NAD
def exchange_rate_usd_to_cny := 5  -- 1 USD = 5 CNY
def cost_of_sculpture_nad := 160  -- Cost of sculpture in NAD

theorem cost_of_sculpture_cny : (cost_of_sculpture_nad / exchange_rate_usd_to_nad) * exchange_rate_usd_to_cny = 100 := by
  sorry

end NUMINAMATH_GPT_cost_of_sculpture_cny_l1645_164519


namespace NUMINAMATH_GPT_smallest_xym_sum_l1645_164579

def is_two_digit_integer (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

def reversed_digits (x y : ℤ) : Prop :=
  ∃ a b : ℤ, x = 10 * a + b ∧ y = 10 * b + a ∧ 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9

def odd_multiple_of_9 (n : ℤ) : Prop :=
  ∃ k : ℤ, k % 2 = 1 ∧ n = 9 * k

theorem smallest_xym_sum :
  ∃ (x y m : ℤ), is_two_digit_integer x ∧ is_two_digit_integer y ∧ reversed_digits x y ∧ x^2 + y^2 = m^2 ∧ odd_multiple_of_9 (x + y) ∧ x + y + m = 169 :=
by
  sorry

end NUMINAMATH_GPT_smallest_xym_sum_l1645_164579


namespace NUMINAMATH_GPT_correct_simplification_l1645_164524

theorem correct_simplification (m a b x y : ℝ) :
  ¬ (4 * m - m = 3) ∧
  ¬ (a^2 * b - a * b^2 = 0) ∧
  ¬ (2 * a^3 - 3 * a^3 = a^3) ∧
  (x * y - 2 * x * y = - x * y) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_simplification_l1645_164524


namespace NUMINAMATH_GPT_range_of_m_l1645_164595

theorem range_of_m {x m : ℝ} (h : ∀ x, x^2 - 2*x + 2*m - 1 ≥ 0) : m ≥ 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1645_164595


namespace NUMINAMATH_GPT_cost_of_senior_ticket_l1645_164593

theorem cost_of_senior_ticket (x : ℤ) (total_tickets : ℤ) (cost_regular_ticket : ℤ) (total_sales : ℤ) (senior_tickets_sold : ℤ) (regular_tickets_sold : ℤ) :
  total_tickets = 65 →
  cost_regular_ticket = 15 →
  total_sales = 855 →
  senior_tickets_sold = 24 →
  regular_tickets_sold = total_tickets - senior_tickets_sold →
  total_sales = senior_tickets_sold * x + regular_tickets_sold * cost_regular_ticket →
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_senior_ticket_l1645_164593


namespace NUMINAMATH_GPT_circles_intersect_l1645_164583

theorem circles_intersect (R r d: ℝ) (hR: R = 7) (hr: r = 4) (hd: d = 8) : (R - r < d) ∧ (d < R + r) :=
by
  rw [hR, hr, hd]
  exact ⟨by linarith, by linarith⟩

end NUMINAMATH_GPT_circles_intersect_l1645_164583


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1645_164575

theorem sufficient_but_not_necessary (a : ℝ) : (a > 6 → a^2 > 36) ∧ ¬(a^2 > 36 → a > 6) := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1645_164575


namespace NUMINAMATH_GPT_find_other_number_l1645_164521

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 12 n = 60) (h_hcf : Nat.gcd 12 n = 3) : n = 15 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l1645_164521


namespace NUMINAMATH_GPT_jon_original_number_l1645_164517

theorem jon_original_number :
  ∃ y : ℤ, (5 * (3 * y + 6) - 8 = 142) ∧ (y = 8) :=
sorry

end NUMINAMATH_GPT_jon_original_number_l1645_164517


namespace NUMINAMATH_GPT_determine_day_from_statements_l1645_164564

/-- Define the days of the week as an inductive type. -/
inductive Day where
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday
  deriving DecidableEq, Repr

open Day

/-- Define the properties of the lion lying on specific days. -/
def lion_lies (d : Day) : Prop :=
  d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Define the properties of the lion telling the truth on specific days. -/
def lion_truth (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday ∨ d = Sunday

/-- Define the properties of the unicorn lying on specific days. -/
def unicorn_lies (d : Day) : Prop :=
  d = Thursday ∨ d = Friday ∨ d = Saturday

/-- Define the properties of the unicorn telling the truth on specific days. -/
def unicorn_truth (d : Day) : Prop :=
  d = Sunday ∨ d = Monday ∨ d = Tuesday ∨ d = Wednesday

/-- Function to determine the day before a given day. -/
def yesterday (d : Day) : Day :=
  match d with
  | Monday    => Sunday
  | Tuesday   => Monday
  | Wednesday => Tuesday
  | Thursday  => Wednesday
  | Friday    => Thursday
  | Saturday  => Friday
  | Sunday    => Saturday

/-- Define the lion's statement: "Yesterday was a day when I lied." -/
def lion_statement (d : Day) : Prop :=
  lion_lies (yesterday d)

/-- Define the unicorn's statement: "Yesterday was a day when I lied." -/
def unicorn_statement (d : Day) : Prop :=
  unicorn_lies (yesterday d)

/-- Prove that today must be Thursday given the conditions and statements. -/
theorem determine_day_from_statements (d : Day) :
    lion_statement d ∧ unicorn_statement d → d = Thursday := by
  sorry

end NUMINAMATH_GPT_determine_day_from_statements_l1645_164564


namespace NUMINAMATH_GPT_find_sample_size_l1645_164501

-- Define the frequencies
def frequencies (k : ℕ) : List ℕ := [2 * k, 3 * k, 4 * k, 6 * k, 4 * k, k]

-- Define the sum of the first three frequencies
def sum_first_three_frequencies (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k

-- Define the total number of data points
def total_data_points (k : ℕ) : ℕ := 2 * k + 3 * k + 4 * k + 6 * k + 4 * k + k

-- Define the main theorem
theorem find_sample_size (n k : ℕ) (h1 : sum_first_three_frequencies k = 27)
  (h2 : total_data_points k = n) : n = 60 := by
  sorry

end NUMINAMATH_GPT_find_sample_size_l1645_164501


namespace NUMINAMATH_GPT_largest_n_multiple_3_l1645_164525

theorem largest_n_multiple_3 (n : ℕ) (h1 : n < 100000) (h2 : (8 * (n + 2)^5 - n^2 + 14 * n - 30) % 3 = 0) : n = 99999 := 
sorry

end NUMINAMATH_GPT_largest_n_multiple_3_l1645_164525


namespace NUMINAMATH_GPT_nature_of_graph_l1645_164582

theorem nature_of_graph :
  ∀ (x y : ℝ), (x^2 - 3 * y) * (x - y + 1) = (y^2 - 3 * x) * (x - y + 1) →
    (y = -x - 3 ∨ y = x ∨ y = x + 1) ∧ ¬( (y = -x - 3) ∧ (y = x) ∧ (y = x + 1) ) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_nature_of_graph_l1645_164582


namespace NUMINAMATH_GPT_abs_triangle_inequality_l1645_164561

theorem abs_triangle_inequality {a : ℝ} (h : ∀ x : ℝ, |x - 3| + |x + 1| > a) : a < 4 :=
sorry

end NUMINAMATH_GPT_abs_triangle_inequality_l1645_164561


namespace NUMINAMATH_GPT_black_pens_per_student_l1645_164503

theorem black_pens_per_student (number_of_students : ℕ)
                               (red_pens_per_student : ℕ)
                               (taken_first_month : ℕ)
                               (taken_second_month : ℕ)
                               (pens_after_splitting : ℕ)
                               (initial_black_pens_per_student : ℕ) : 
  number_of_students = 3 → 
  red_pens_per_student = 62 → 
  taken_first_month = 37 → 
  taken_second_month = 41 → 
  pens_after_splitting = 79 → 
  initial_black_pens_per_student = 43 :=
by sorry

end NUMINAMATH_GPT_black_pens_per_student_l1645_164503


namespace NUMINAMATH_GPT_no_such_integers_l1645_164536

def p (x : ℤ) : ℤ := x^2 + x - 70

theorem no_such_integers : ¬ (∃ m n : ℤ, 0 < m ∧ m < n ∧ n ∣ p m ∧ (n + 1) ∣ p (m + 1)) :=
by
  sorry

end NUMINAMATH_GPT_no_such_integers_l1645_164536
