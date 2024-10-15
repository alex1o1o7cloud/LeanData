import Mathlib

namespace NUMINAMATH_GPT_no_integer_roots_l1966_196635

theorem no_integer_roots : ∀ x : ℤ, x^3 - 3 * x^2 - 16 * x + 20 ≠ 0 := by
  intro x
  sorry

end NUMINAMATH_GPT_no_integer_roots_l1966_196635


namespace NUMINAMATH_GPT_line_bisects_circle_perpendicular_l1966_196664

theorem line_bisects_circle_perpendicular :
  (∃ l : ℝ → ℝ, (∀ x y : ℝ, x^2 + y^2 + x - 2*y + 1 = 0 → l x = y)
               ∧ (∀ x y : ℝ, x + 2*y + 3 = 0 → x ∈ { x | ∃ k:ℝ, y = -1/2 * k + l x})
               ∧ (∀ x y : ℝ, l x = 2 * x - 2)) :=
sorry

end NUMINAMATH_GPT_line_bisects_circle_perpendicular_l1966_196664


namespace NUMINAMATH_GPT_anya_kolya_apples_l1966_196684

theorem anya_kolya_apples (A K : ℕ) (h1 : A = (K * 100) / (A + K)) (h2 : K = (A * 100) / (A + K)) : A = 50 ∧ K = 50 :=
sorry

end NUMINAMATH_GPT_anya_kolya_apples_l1966_196684


namespace NUMINAMATH_GPT_sum_of_three_numbers_l1966_196636

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  n % 10 = d ∨ n / 10 % 10 = d ∨ n / 100 = d

theorem sum_of_three_numbers (A B C : ℕ) :
  (100 ≤ A ∧ A < 1000 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100) ∧
  (∃ (B7 C7 : ℕ), B7 + C7 = 208 ∧ (contains_digit A 7 ∨ contains_digit B7 7 ∨ contains_digit C7 7)) ∧
  (∃ (B3 C3 : ℕ), B3 + C3 = 76 ∧ (contains_digit B3 3 ∨ contains_digit C3 3)) →
  A + B + C = 247 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l1966_196636


namespace NUMINAMATH_GPT_chicken_price_per_pound_l1966_196615

theorem chicken_price_per_pound (beef_pounds chicken_pounds : ℕ) (beef_price chicken_price : ℕ)
    (total_amount : ℕ)
    (h_beef_quantity : beef_pounds = 1000)
    (h_beef_cost : beef_price = 8)
    (h_chicken_quantity : chicken_pounds = 2 * beef_pounds)
    (h_total_price : 1000 * beef_price + chicken_pounds * chicken_price = total_amount)
    (h_total_amount : total_amount = 14000) : chicken_price = 3 :=
by
  sorry

end NUMINAMATH_GPT_chicken_price_per_pound_l1966_196615


namespace NUMINAMATH_GPT_total_weight_of_ripe_apples_is_1200_l1966_196676

def total_apples : Nat := 14
def weight_ripe_apple : Nat := 150
def weight_unripe_apple : Nat := 120
def unripe_apples : Nat := 6
def ripe_apples : Nat := total_apples - unripe_apples
def total_weight_ripe_apples : Nat := ripe_apples * weight_ripe_apple

theorem total_weight_of_ripe_apples_is_1200 :
  total_weight_ripe_apples = 1200 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_ripe_apples_is_1200_l1966_196676


namespace NUMINAMATH_GPT_max_abs_value_of_quadratic_function_l1966_196613

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def point_in_band_region (y k l : ℝ) : Prop := k ≤ y ∧ y ≤ l

theorem max_abs_value_of_quadratic_function (a b c t : ℝ) (h1 : point_in_band_region (quadratic_function a b c (-2) + 2) 0 4)
                                             (h2 : point_in_band_region (quadratic_function a b c 0 + 2) 0 4)
                                             (h3 : point_in_band_region (quadratic_function a b c 2 + 2) 0 4)
                                             (h4 : point_in_band_region (t + 1) (-1) 3) :
  |quadratic_function a b c t| ≤ 5 / 2 :=
sorry

end NUMINAMATH_GPT_max_abs_value_of_quadratic_function_l1966_196613


namespace NUMINAMATH_GPT_cos_angle_difference_l1966_196669

theorem cos_angle_difference (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3 / 2) 
  (h2 : Real.cos A + Real.cos B = 1): 
  Real.cos (A - B) = 5 / 8 := 
by sorry

end NUMINAMATH_GPT_cos_angle_difference_l1966_196669


namespace NUMINAMATH_GPT_problem1_problem2_l1966_196629

-- Problem 1: Calculation Proof
theorem problem1 : (3 - Real.pi)^0 - Real.sqrt 4 + 4 * Real.sin (Real.pi * 60 / 180) + |Real.sqrt 3 - 3| = 2 + Real.sqrt 3 :=
by
  sorry

-- Problem 2: Inequality Systems Proof
theorem problem2 (x : ℝ) :
  (5 * (x + 3) > 4 * x + 8) ∧ (x / 6 - 1 < (x - 2) / 3) → x > -2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1966_196629


namespace NUMINAMATH_GPT_animals_remaining_correct_l1966_196699

-- Definitions from the conditions
def initial_cows : ℕ := 184
def initial_dogs : ℕ := initial_cows / 2

def cows_sold : ℕ := initial_cows / 4
def remaining_cows : ℕ := initial_cows - cows_sold

def dogs_sold : ℕ := (3 * initial_dogs) / 4
def remaining_dogs : ℕ := initial_dogs - dogs_sold

def total_remaining_animals : ℕ := remaining_cows + remaining_dogs

-- Theorem to be proved
theorem animals_remaining_correct : total_remaining_animals = 161 := 
by
  sorry

end NUMINAMATH_GPT_animals_remaining_correct_l1966_196699


namespace NUMINAMATH_GPT_problem_conditions_l1966_196622

theorem problem_conditions (a : ℕ → ℤ) :
  (1 + x)^6 = a 0 + a 1 * (1 - x) + a 2 * (1 - x)^2 + a 3 * (1 - x)^3 + a 4 * (1 - x)^4 + a 5 * (1 - x)^5 + a 6 * (1 - x)^6 →
  a 6 = 1 ∧ a 1 + a 3 + a 5 = -364 :=
by sorry

end NUMINAMATH_GPT_problem_conditions_l1966_196622


namespace NUMINAMATH_GPT_complex_expression_power_48_l1966_196663

open Complex

noncomputable def complex_expression := (1 + I) / Real.sqrt 2

theorem complex_expression_power_48 : complex_expression ^ 48 = 1 := by
  sorry

end NUMINAMATH_GPT_complex_expression_power_48_l1966_196663


namespace NUMINAMATH_GPT_obtain_any_natural_from_4_l1966_196649

/-- Definitions of allowed operations:
  - Append the digit 4.
  - Append the digit 0.
  - Divide by 2, if the number is even.
--/
def append4 (n : ℕ) : ℕ := 10 * n + 4
def append0 (n : ℕ) : ℕ := 10 * n
def divide2 (n : ℕ) : ℕ := n / 2

/-- We'll also define if a number is even --/
def is_even (n : ℕ) : Prop := n % 2 = 0

/-- Define the set of operations applied on a number --/
inductive operations : ℕ → ℕ → Prop
| initial : operations 4 4
| append4_step (n m : ℕ) : operations n m → operations n (append4 m)
| append0_step (n m : ℕ) : operations n m → operations n (append0 m)
| divide2_step (n m : ℕ) : is_even m → operations n m → operations n (divide2 m)

/-- The main theorem proving that any natural number can be obtained from 4 using the allowed operations --/
theorem obtain_any_natural_from_4 (n : ℕ) : ∃ m, operations 4 m ∧ m = n :=
by sorry

end NUMINAMATH_GPT_obtain_any_natural_from_4_l1966_196649


namespace NUMINAMATH_GPT_product_of_real_roots_l1966_196668

theorem product_of_real_roots : 
  (∃ x y : ℝ, (x ^ Real.log x = Real.exp 1) ∧ (y ^ Real.log y = Real.exp 1) ∧ x ≠ y ∧ x * y = 1) :=
by
  sorry

end NUMINAMATH_GPT_product_of_real_roots_l1966_196668


namespace NUMINAMATH_GPT_fraction_value_l1966_196618

theorem fraction_value : (2020 / (20 * 20 : ℝ)) = 5.05 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l1966_196618


namespace NUMINAMATH_GPT_sqrt_sq_eq_abs_l1966_196651

theorem sqrt_sq_eq_abs (a : ℝ) : Real.sqrt (a^2) = |a| :=
sorry

end NUMINAMATH_GPT_sqrt_sq_eq_abs_l1966_196651


namespace NUMINAMATH_GPT_inequality_positive_real_l1966_196680

theorem inequality_positive_real (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
sorry

end NUMINAMATH_GPT_inequality_positive_real_l1966_196680


namespace NUMINAMATH_GPT_number_of_hens_l1966_196685

theorem number_of_hens (H C : ℕ) (h1 : H + C = 46) (h2 : 2 * H + 4 * C = 136) : H = 24 :=
by
  sorry

end NUMINAMATH_GPT_number_of_hens_l1966_196685


namespace NUMINAMATH_GPT_domain_of_tan_l1966_196644

noncomputable def is_excluded_from_domain (x : ℝ) : Prop :=
  ∃ k : ℤ, x = 1 + 6 * k

theorem domain_of_tan {x : ℝ} :
  ∀ x, ¬ is_excluded_from_domain x ↔ ¬ ∃ k : ℤ, x = 1 + 6 * k := 
by 
  sorry

end NUMINAMATH_GPT_domain_of_tan_l1966_196644


namespace NUMINAMATH_GPT_initial_goal_proof_l1966_196605

def marys_collection (k : ℕ) : ℕ := 5 * k
def scotts_collection (m : ℕ) : ℕ := m / 3
def total_collected (k : ℕ) (m : ℕ) (s : ℕ) : ℕ := k + m + s
def initial_goal (total : ℕ) (excess : ℕ) : ℕ := total - excess

theorem initial_goal_proof : 
  initial_goal (total_collected 600 (marys_collection 600) (scotts_collection (marys_collection 600))) 600 = 4000 :=
by
  sorry

end NUMINAMATH_GPT_initial_goal_proof_l1966_196605


namespace NUMINAMATH_GPT_a_2_is_minus_1_l1966_196696
open Nat

variable (a S : ℕ → ℤ)

-- Conditions
axiom sum_first_n (n : ℕ) (hn : n > 0) : 2 * S n - n * a n = n
axiom S_20 : S 20 = -360

-- The problem statement to prove
theorem a_2_is_minus_1 : a 2 = -1 :=
by 
  sorry

end NUMINAMATH_GPT_a_2_is_minus_1_l1966_196696


namespace NUMINAMATH_GPT_benny_birthday_money_l1966_196632

-- Define conditions
def spent_on_gear : ℕ := 47
def left_over : ℕ := 32

-- Define the total amount Benny received
def total_money_received : ℕ := 79

-- Theorem statement
theorem benny_birthday_money (spent_on_gear : ℕ) (left_over : ℕ) : spent_on_gear + left_over = total_money_received :=
by
  sorry

end NUMINAMATH_GPT_benny_birthday_money_l1966_196632


namespace NUMINAMATH_GPT_extremum_of_function_l1966_196658

theorem extremum_of_function (k : ℝ) (h₀ : k ≠ 1) :
  (k > 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≤ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) ∧
  (k < 1 → ∃ x : ℝ, ∀ y : ℝ, ((k-1) * x^2 - 2 * (k-1) * x - k) ≥ ((k-1) * y^2 - 2 * (k-1) * y - k) ∧ ((k-1) * x^2 - 2 * (k-1) * x - k) = -2*k + 1) :=
by
  sorry

end NUMINAMATH_GPT_extremum_of_function_l1966_196658


namespace NUMINAMATH_GPT_rectangle_length_l1966_196616

theorem rectangle_length :
  ∀ (side : ℕ) (width : ℕ) (length : ℕ), 
  side = 4 → 
  width = 8 → 
  side * side = width * length → 
  length = 2 := 
by
  -- sorry to skip the proof
  intros side width length h1 h2 h3
  sorry

end NUMINAMATH_GPT_rectangle_length_l1966_196616


namespace NUMINAMATH_GPT_find_xy_l1966_196603

theorem find_xy (x y : ℝ) (h : (x^2 + 6 * x + 12) * (5 * y^2 + 2 * y + 1) = 12 / 5) : 
    x * y = 3 / 5 :=
sorry

end NUMINAMATH_GPT_find_xy_l1966_196603


namespace NUMINAMATH_GPT_net_change_of_Toronto_Stock_Exchange_l1966_196698

theorem net_change_of_Toronto_Stock_Exchange :
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  (monday + tuesday + wednesday + thursday + friday) = -119 :=
by
  let monday := -150
  let tuesday := 106
  let wednesday := -47
  let thursday := 182
  let friday := -210
  have h : (monday + tuesday + wednesday + thursday + friday) = -119 := sorry
  exact h

end NUMINAMATH_GPT_net_change_of_Toronto_Stock_Exchange_l1966_196698


namespace NUMINAMATH_GPT_first_discount_percentage_l1966_196692

theorem first_discount_percentage (normal_price sale_price : ℝ) (second_discount : ℝ) (first_discount : ℝ) :
  normal_price = 149.99999999999997 →
  sale_price = 108 →
  second_discount = 0.20 →
  (1 - second_discount) * (1 - first_discount) * normal_price = sale_price →
  first_discount = 0.10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l1966_196692


namespace NUMINAMATH_GPT_tarun_garden_area_l1966_196660

theorem tarun_garden_area :
  ∀ (side : ℝ), 
  (1500 / 8 = 4 * side) → 
  (30 * side = 1500) → 
  side^2 = 2197.265625 :=
by
  sorry

end NUMINAMATH_GPT_tarun_garden_area_l1966_196660


namespace NUMINAMATH_GPT_factor_correct_l1966_196638

def factor_expression (x : ℝ) : Prop :=
  x * (x - 3) - 5 * (x - 3) = (x - 5) * (x - 3)

theorem factor_correct (x : ℝ) : factor_expression x :=
  by sorry

end NUMINAMATH_GPT_factor_correct_l1966_196638


namespace NUMINAMATH_GPT_min_AP_BP_l1966_196604

-- Definitions based on conditions in the problem
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 6)
def parabola (x y : ℝ) : Prop := y^2 = 8 * x

-- The theorem to prove the minimum value of AP + BP
theorem min_AP_BP
  (P : ℝ × ℝ)
  (hP_parabola : parabola P.1 P.2) :
  dist P A + dist P B ≥ 9 :=
sorry

end NUMINAMATH_GPT_min_AP_BP_l1966_196604


namespace NUMINAMATH_GPT_binom_product_l1966_196694

open Nat

-- Define binomial coefficient
def binom (n k : ℕ) : ℕ := n.choose k

theorem binom_product :
  binom 10 3 * binom 8 3 = 6720 := by
  sorry

end NUMINAMATH_GPT_binom_product_l1966_196694


namespace NUMINAMATH_GPT_triangle_is_isosceles_l1966_196641

open Real

-- Define the basic setup of the triangle and the variables involved
variables {A B C : ℝ} -- Angles of the triangle
variables {a b c : ℝ} -- Sides opposite to A, B, and C respectively
variables (h1 : a * cos B = b * cos A) -- Given condition: a * cos B = b * cos A

-- The theorem stating that the given condition implies the triangle is isosceles
theorem triangle_is_isosceles (h1 : a * cos B = b * cos A) : A = B :=
sorry

end NUMINAMATH_GPT_triangle_is_isosceles_l1966_196641


namespace NUMINAMATH_GPT_calculate_expression_l1966_196654

theorem calculate_expression : 3 * ((-3)^4 - (-3)^3 + (-3)^2 - 3^2 + 3^3 - 3^4) = 81 := 
by sorry

end NUMINAMATH_GPT_calculate_expression_l1966_196654


namespace NUMINAMATH_GPT_part1_part2_l1966_196624

variable {x m : ℝ}

def P (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10
def S (x : ℝ) (m : ℝ) : Prop := -m + 1 ≤ x ∧ x ≤ m + 1

theorem part1 (h : ∀ x, P x → P x ∨ S x m) : m ≤ 0 :=
sorry

theorem part2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (P x ↔ S x m) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1966_196624


namespace NUMINAMATH_GPT_second_discount_percentage_is_20_l1966_196609

theorem second_discount_percentage_is_20 
    (normal_price : ℝ)
    (final_price : ℝ)
    (first_discount : ℝ)
    (first_discount_percentage : ℝ)
    (h1 : normal_price = 149.99999999999997)
    (h2 : final_price = 108)
    (h3 : first_discount_percentage = 10)
    (h4 : first_discount = normal_price * (first_discount_percentage / 100)) :
    (((normal_price - first_discount) - final_price) / (normal_price - first_discount)) * 100 = 20 := by
  sorry

end NUMINAMATH_GPT_second_discount_percentage_is_20_l1966_196609


namespace NUMINAMATH_GPT_fraction_greater_than_decimal_l1966_196631

theorem fraction_greater_than_decimal :
  (1 / 4 : ℝ) > (24999999 / (10^8 : ℝ)) + (1 / (4 * (10^8 : ℝ))) :=
by
  sorry

end NUMINAMATH_GPT_fraction_greater_than_decimal_l1966_196631


namespace NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_is_perfect_square_l1966_196691

theorem sum_of_squares_of_consecutive_integers_is_perfect_square (x : ℤ) :
  ∃ k : ℤ, k ^ 2 = x ^ 2 + (x + 1) ^ 2 + (x ^ 2 * (x + 1) ^ 2) :=
by
  use (x^2 + x + 1)
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_consecutive_integers_is_perfect_square_l1966_196691


namespace NUMINAMATH_GPT_actual_distance_between_cities_l1966_196689

-- Define the scale and distance on the map as constants
def distance_on_map : ℝ := 20
def scale_inch_miles : ℝ := 12  -- Because 1 inch = 12 miles derived from the scale 0.5 inches = 6 miles

-- Define the actual distance calculation
def actual_distance (distance_inch : ℝ) (scale : ℝ) : ℝ :=
  distance_inch * scale

-- Example theorem to prove the actual distance between the cities
theorem actual_distance_between_cities :
  actual_distance distance_on_map scale_inch_miles = 240 := by
  sorry

end NUMINAMATH_GPT_actual_distance_between_cities_l1966_196689


namespace NUMINAMATH_GPT_only_integer_solution_is_trivial_l1966_196611

theorem only_integer_solution_is_trivial (a b c : ℤ) (h : 5 * a^2 + 9 * b^2 = 13 * c^2) : a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end NUMINAMATH_GPT_only_integer_solution_is_trivial_l1966_196611


namespace NUMINAMATH_GPT_find_a_inverse_function_l1966_196642

theorem find_a_inverse_function
  (a : ℝ)
  (f : ℝ → ℝ)
  (h1 : ∀ x y, y = f x ↔ x = a * y)
  (h2 : f 4 = 2) :
  a = 2 := 
sorry

end NUMINAMATH_GPT_find_a_inverse_function_l1966_196642


namespace NUMINAMATH_GPT_f_zero_is_one_l1966_196617

def f (n : ℕ) : ℕ := sorry

theorem f_zero_is_one (f : ℕ → ℕ)
  (h1 : ∀ n : ℕ, f (f n) + f n = 2 * n + 3)
  (h2 : f 2015 = 2016) : f 0 = 1 := 
by {
  -- proof not required
  sorry
}

end NUMINAMATH_GPT_f_zero_is_one_l1966_196617


namespace NUMINAMATH_GPT_remainder_when_divided_by_9_l1966_196675

theorem remainder_when_divided_by_9 (x : ℕ) (h : 4 * x % 9 = 2) : x % 9 = 5 :=
by sorry

end NUMINAMATH_GPT_remainder_when_divided_by_9_l1966_196675


namespace NUMINAMATH_GPT_M_lt_N_l1966_196690

/-- M is the coefficient of x^4 y^2 in the expansion of (x^2 + x + 2y)^5 -/
def M : ℕ := 120

/-- N is the sum of the coefficients in the expansion of (3/x - x)^7 -/
def N : ℕ := 128

/-- The relationship between M and N -/
theorem M_lt_N : M < N := by 
  dsimp [M, N]
  sorry

end NUMINAMATH_GPT_M_lt_N_l1966_196690


namespace NUMINAMATH_GPT_mary_regular_hours_l1966_196666

theorem mary_regular_hours (x y : ℕ) :
  8 * x + 10 * y = 760 ∧ x + y = 80 → x = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_mary_regular_hours_l1966_196666


namespace NUMINAMATH_GPT_problem1_problem2_l1966_196634

theorem problem1 : (Real.sqrt 2) * (Real.sqrt 6) + (Real.sqrt 3) = 3 * (Real.sqrt 3) :=
  sorry

theorem problem2 : (1 - Real.sqrt 2) * (2 - Real.sqrt 2) = 4 - 3 * (Real.sqrt 2) :=
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1966_196634


namespace NUMINAMATH_GPT_simple_interest_rate_l1966_196656

theorem simple_interest_rate (P : ℝ) (SI : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : SI = P / 5)
  (h2 : SI = P * R * T / 100)
  (h3 : T = 7) : 
  R = 20 / 7 :=
by 
  sorry

end NUMINAMATH_GPT_simple_interest_rate_l1966_196656


namespace NUMINAMATH_GPT_fraction_eq_l1966_196633

theorem fraction_eq (x : ℝ) (h1 : x * 180 = 24) (h2 : x < 20 / 100) : x = 2 / 15 :=
sorry

end NUMINAMATH_GPT_fraction_eq_l1966_196633


namespace NUMINAMATH_GPT_solve_for_y_l1966_196602

theorem solve_for_y (y : ℝ) (h : 6 * y^(1/4) - 3 * (y / y^(3/4)) = 12 + y^(1/4)) : y = 1296 := by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1966_196602


namespace NUMINAMATH_GPT_range_neg_square_l1966_196693

theorem range_neg_square (x : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) : 
  -9 ≤ -x^2 ∧ -x^2 ≤ 0 :=
sorry

end NUMINAMATH_GPT_range_neg_square_l1966_196693


namespace NUMINAMATH_GPT_age_ratio_in_two_years_l1966_196648

-- Definitions of conditions
def son_present_age : ℕ := 26
def age_difference : ℕ := 28
def man_present_age : ℕ := son_present_age + age_difference

-- Future ages after 2 years
def son_future_age : ℕ := son_present_age + 2
def man_future_age : ℕ := man_present_age + 2

-- The theorem to prove
theorem age_ratio_in_two_years : (man_future_age / son_future_age) = 2 := 
by
  -- Step-by-Step proof would go here
  sorry

end NUMINAMATH_GPT_age_ratio_in_two_years_l1966_196648


namespace NUMINAMATH_GPT_fathers_age_l1966_196625

variable (S F : ℕ)
variable (h1 : F = 3 * S)
variable (h2 : F + 15 = 2 * (S + 15))

theorem fathers_age : F = 45 :=
by
  -- the proof steps would go here
  sorry

end NUMINAMATH_GPT_fathers_age_l1966_196625


namespace NUMINAMATH_GPT_power_function_increasing_iff_l1966_196672

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := x^a

theorem power_function_increasing_iff (a : ℝ) : 
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → power_function a x1 < power_function a x2) ↔ a > 0 := 
by
  sorry

end NUMINAMATH_GPT_power_function_increasing_iff_l1966_196672


namespace NUMINAMATH_GPT_possible_values_l1966_196683

def seq_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = 2 * a (n + 2) * a (n + 3) + 2016

theorem possible_values (a : ℕ → ℤ) (h : seq_condition a) :
  (a 1, a 2) = (0, 2016) ∨
  (a 1, a 2) = (-14, 70) ∨
  (a 1, a 2) = (-69, 15) ∨
  (a 1, a 2) = (-2015, 1) ∨
  (a 1, a 2) = (2016, 0) ∨
  (a 1, a 2) = (70, -14) ∨
  (a 1, a 2) = (15, -69) ∨
  (a 1, a 2) = (1, -2015) :=
sorry

end NUMINAMATH_GPT_possible_values_l1966_196683


namespace NUMINAMATH_GPT_proof_neg_q_l1966_196657

variable (f : ℝ → ℝ)
variable (x : ℝ)

def proposition_p (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x < f y

def proposition_q : Prop := ∃ x : ℝ, (deriv fun y => 1 / y) x > 0

theorem proof_neg_q : ¬ proposition_q := 
by
  intro h
  -- proof omitted for brevity
  sorry

end NUMINAMATH_GPT_proof_neg_q_l1966_196657


namespace NUMINAMATH_GPT_area_of_original_rectangle_l1966_196678

theorem area_of_original_rectangle 
  (L W : ℝ)
  (h1 : 2 * L * (3 * W) = 1800) :
  L * W = 300 :=
by
  sorry

end NUMINAMATH_GPT_area_of_original_rectangle_l1966_196678


namespace NUMINAMATH_GPT_coconut_to_almond_ratio_l1966_196688

-- Conditions
def number_of_coconut_candles (C : ℕ) : Prop :=
  ∃ L A : ℕ, L = 2 * C ∧ A = 10

-- Question
theorem coconut_to_almond_ratio (C : ℕ) (h : number_of_coconut_candles C) :
  ∃ r : ℚ, r = C / 10 := by
  sorry

end NUMINAMATH_GPT_coconut_to_almond_ratio_l1966_196688


namespace NUMINAMATH_GPT_quotient_calculation_l1966_196673

theorem quotient_calculation
  (dividend : ℕ)
  (divisor : ℕ)
  (remainder : ℕ)
  (h_dividend : dividend = 176)
  (h_divisor : divisor = 14)
  (h_remainder : remainder = 8) :
  ∃ q, dividend = divisor * q + remainder ∧ q = 12 :=
by
  sorry

end NUMINAMATH_GPT_quotient_calculation_l1966_196673


namespace NUMINAMATH_GPT_range_of_a_l1966_196627

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x < 2 → (x + a < 0))) → (a ≤ -2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1966_196627


namespace NUMINAMATH_GPT_geometric_body_is_cylinder_l1966_196639

def top_view_is_circle : Prop := sorry

def is_prism_or_cylinder : Prop := sorry

theorem geometric_body_is_cylinder 
  (h1 : top_view_is_circle) 
  (h2 : is_prism_or_cylinder) 
  : Cylinder := 
sorry

end NUMINAMATH_GPT_geometric_body_is_cylinder_l1966_196639


namespace NUMINAMATH_GPT_count_valid_abcd_is_zero_l1966_196628

def valid_digits := {a // 1 ≤ a ∧ a ≤ 9} 
def zero_to_nine := {n // 0 ≤ n ∧ n ≤ 9}

noncomputable def increasing_arithmetic_sequence_with_difference_5 (a b c d : ℕ) : Prop := 
  10 * a + b + 5 = 10 * b + c ∧ 
  10 * b + c + 5 = 10 * c + d

theorem count_valid_abcd_is_zero :
  ∀ (a : valid_digits) (b c d : zero_to_nine),
    ¬ increasing_arithmetic_sequence_with_difference_5 a.val b.val c.val d.val := 
sorry

end NUMINAMATH_GPT_count_valid_abcd_is_zero_l1966_196628


namespace NUMINAMATH_GPT_round_trip_time_l1966_196687

def boat_speed_still_water : ℝ := 16
def stream_speed : ℝ := 2
def distance_to_place : ℝ := 7560

theorem round_trip_time : (distance_to_place / (boat_speed_still_water + stream_speed) + distance_to_place / (boat_speed_still_water - stream_speed)) = 960 := by
  sorry

end NUMINAMATH_GPT_round_trip_time_l1966_196687


namespace NUMINAMATH_GPT_common_pts_above_curve_l1966_196667

open Real

theorem common_pts_above_curve {x y t : ℝ} (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : 0 ≤ y ∧ y ≤ 1) (h3 : 0 < t ∧ t < 1) :
  (∀ t, y ≥ (t-1)/t * x + 1 - t) ↔ (sqrt x + sqrt y ≥ 1) := 
by
  sorry

end NUMINAMATH_GPT_common_pts_above_curve_l1966_196667


namespace NUMINAMATH_GPT_ellipse_domain_l1966_196607

theorem ellipse_domain (m : ℝ) :
  (-1 < m ∧ m < 2 ∧ m ≠ 1 / 2) -> 
  ∃ a b : ℝ, (a = 2 - m) ∧ (b = m + 1) ∧ a > 0 ∧ b > 0 ∧ a ≠ b :=
by
  sorry

end NUMINAMATH_GPT_ellipse_domain_l1966_196607


namespace NUMINAMATH_GPT_total_balls_l1966_196606

theorem total_balls (S V B Total : ℕ) (hS : S = 68) (hV : S = V - 12) (hB : S = B + 23) : 
  Total = S + V + B := by
  sorry

end NUMINAMATH_GPT_total_balls_l1966_196606


namespace NUMINAMATH_GPT_complex_exp_l1966_196659

theorem complex_exp {i : ℂ} (h : i^2 = -1) : (1 + i)^30 + (1 - i)^30 = 0 := by
  sorry

end NUMINAMATH_GPT_complex_exp_l1966_196659


namespace NUMINAMATH_GPT_supplementary_angle_proof_l1966_196640

noncomputable def complementary_angle (α : ℝ) : ℝ := 125 + 12 / 60

noncomputable def calculate_angle (c : ℝ) := 180 - c

noncomputable def supplementary_angle (α : ℝ) := 90 - α

theorem supplementary_angle_proof :
    let α := calculate_angle (complementary_angle α)
    supplementary_angle α = 35 + 12 / 60 := 
by
  sorry

end NUMINAMATH_GPT_supplementary_angle_proof_l1966_196640


namespace NUMINAMATH_GPT_number_of_men_in_group_l1966_196652

-- Define the conditions
variable (n : ℕ) -- number of men in the group
variable (A : ℝ) -- original average age of the group
variable (increase_in_years : ℝ := 2) -- the increase in the average age
variable (ages_before_replacement : ℝ := 21 + 23) -- total age of the men replaced
variable (ages_after_replacement : ℝ := 2 * 37) -- total age of the new men

-- Define the theorem using the conditions
theorem number_of_men_in_group 
  (h1 : n * increase_in_years = ages_after_replacement - ages_before_replacement) :
  n = 15 :=
sorry

end NUMINAMATH_GPT_number_of_men_in_group_l1966_196652


namespace NUMINAMATH_GPT_minimum_value_x_l1966_196670

theorem minimum_value_x (a b x : ℝ) (h₀ : a > 0) (h₁ : b > 0) 
    (H : 4 * a + b * (1 - a) = 0) 
    (Hinequality : ∀ (a b : ℝ), a > 0 → b > 0 → 
        (4 * a + b * (1 - a) = 0 → 
        (1 / a^2 + 16 / b^2 ≥ 1 + x / 2 - x^2))) : 
    x >= 1 := 
sorry

end NUMINAMATH_GPT_minimum_value_x_l1966_196670


namespace NUMINAMATH_GPT_math_problem_l1966_196614

-- Arithmetic sequence {a_n}
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 2 = 8 ∧ a 3 + a 5 = 4 * a 2

-- General term of the arithmetic sequence {a_n}
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n = 4 * n

-- Geometric sequence {b_n}
def geometric_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  b 4 = a 1 ∧ b 6 = a 4

-- The sum S_n of the first n terms of the sequence {b_n - a_n}
def sum_sequence (b : ℕ → ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (2 ^ (n - 1) - 1 / 2 - 2 * n ^ 2 - 2 * n)

-- Full proof statement
theorem math_problem (a : ℕ → ℕ) (b : ℕ → ℝ) (S : ℕ → ℝ) :
  arithmetic_sequence a →
  general_term a →
  ∀ a_n : ℕ → ℝ, a_n 1 = 4 ∧ a_n 4 = 16 →
  geometric_sequence b a_n →
  sum_sequence b a_n S :=
by
  intros h_arith_seq h_gen_term h_a_n h_geom_seq
  sorry

end NUMINAMATH_GPT_math_problem_l1966_196614


namespace NUMINAMATH_GPT_combined_room_size_l1966_196682

theorem combined_room_size (M J S : ℝ) 
  (h1 : M + J + S = 800) 
  (h2 : J = M + 100) 
  (h3 : S = M - 50) : 
  J + S = 550 := 
by
  sorry

end NUMINAMATH_GPT_combined_room_size_l1966_196682


namespace NUMINAMATH_GPT_framing_required_l1966_196621

/- 
  Problem: A 5-inch by 7-inch picture is enlarged by quadrupling its dimensions.
  A 3-inch-wide border is then placed around each side of the enlarged picture.
  What is the minimum number of linear feet of framing that must be purchased
  to go around the perimeter of the border?
-/
def original_width : ℕ := 5
def original_height : ℕ := 7
def enlargement_factor : ℕ := 4
def border_width : ℕ := 3

theorem framing_required : (2 * ((original_width * enlargement_factor + 2 * border_width) + (original_height * enlargement_factor + 2 * border_width))) / 12 = 10 :=
by
  sorry

end NUMINAMATH_GPT_framing_required_l1966_196621


namespace NUMINAMATH_GPT_line_tangent_constant_sum_l1966_196665

noncomputable def parabolaEquation (x y : ℝ) : Prop :=
  y ^ 2 = 4 * x

noncomputable def circleEquation (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + y ^ 2 = 4

noncomputable def isTangent (l : ℝ → ℝ) (x y : ℝ) : Prop :=
  l x = y ∧ ((x - 2) ^ 2 + y ^ 2 = 4)

theorem line_tangent_constant_sum (l : ℝ → ℝ) (A B P : ℝ × ℝ) (x₁ y₁ x₂ y₂ : ℝ) :
  parabolaEquation x₁ y₁ →
  parabolaEquation x₂ y₂ →
  isTangent l (4 / 5) (8 / 5) →
  A = (x₁, y₁) →
  B = (x₂, y₂) →
  let F := (1, 0)
  let distance (p1 p2 : ℝ × ℝ) : ℝ := (Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2))
  (distance F A) + (distance F B) - (distance A B) = 2 :=
sorry

end NUMINAMATH_GPT_line_tangent_constant_sum_l1966_196665


namespace NUMINAMATH_GPT_solve_inequality_l1966_196630

theorem solve_inequality (x : ℝ) :
  (x - 2) / (x + 5) ≤ 1 / 2 ↔ x ∈ Set.Ioc (-5 : ℝ) 9 :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1966_196630


namespace NUMINAMATH_GPT_odd_number_as_diff_of_squares_l1966_196608

theorem odd_number_as_diff_of_squares (n : ℤ) : ∃ a b : ℤ, a^2 - b^2 = 2 * n + 1 :=
by
  use (n + 1), n
  sorry

end NUMINAMATH_GPT_odd_number_as_diff_of_squares_l1966_196608


namespace NUMINAMATH_GPT_manufacturer_cost_price_l1966_196679

theorem manufacturer_cost_price
    (C : ℝ)
    (h1 : C > 0)
    (h2 : 1.18 * 1.20 * 1.25 * C = 30.09) :
    |C - 17| < 0.01 :=
by
    sorry

end NUMINAMATH_GPT_manufacturer_cost_price_l1966_196679


namespace NUMINAMATH_GPT_david_bike_distance_l1966_196626

noncomputable def david_time_hours : ℝ := 2 + 1 / 3
noncomputable def david_speed_mph : ℝ := 6.998571428571427
noncomputable def david_distance : ℝ := 16.33

theorem david_bike_distance :
  david_speed_mph * david_time_hours = david_distance :=
by
  sorry

end NUMINAMATH_GPT_david_bike_distance_l1966_196626


namespace NUMINAMATH_GPT_primes_between_4900_8100_l1966_196681

theorem primes_between_4900_8100 :
  ∃ (count : ℕ),
  count = 5 ∧ ∀ n : ℤ, 70 < n ∧ n < 90 ∧ (n * n > 4900 ∧ n * n < 8100 ∧ Prime n) → count = 5 :=
by
  sorry

end NUMINAMATH_GPT_primes_between_4900_8100_l1966_196681


namespace NUMINAMATH_GPT_min_sum_x_y_condition_l1966_196671

theorem min_sum_x_y_condition {x y : ℝ} (h₁ : x > 0) (h₂ : y > 0) (h₃ : 1 / x + 9 / y = 1) : x + y = 16 :=
by
  sorry -- proof skipped

end NUMINAMATH_GPT_min_sum_x_y_condition_l1966_196671


namespace NUMINAMATH_GPT_probability_event_occurs_l1966_196650

def in_interval (x : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 2 * Real.pi

def event_occurs (x : ℝ) : Prop :=
  Real.cos (x + Real.pi / 3) + Real.sqrt 3 * Real.sin (x + Real.pi / 3) ≥ 1

theorem probability_event_occurs : 
  (∀ x, in_interval x → event_occurs x) → 
  (∃ p, p = 1/3) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_probability_event_occurs_l1966_196650


namespace NUMINAMATH_GPT_sum_congruence_example_l1966_196661

theorem sum_congruence_example (a b c : ℤ) (h1 : a % 15 = 7) (h2 : b % 15 = 3) (h3 : c % 15 = 9) : 
  (a + b + c) % 15 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_sum_congruence_example_l1966_196661


namespace NUMINAMATH_GPT_scientific_notation_of_1_300_000_l1966_196697

-- Define the condition: 1.3 million equals 1,300,000
def one_point_three_million : ℝ := 1300000

-- The theorem statement for the question
theorem scientific_notation_of_1_300_000 :
  one_point_three_million = 1.3 * 10^6 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_1_300_000_l1966_196697


namespace NUMINAMATH_GPT_correct_assignment_statements_l1966_196600

-- Defining what constitutes an assignment statement in this context.
def is_assignment_statement (s : String) : Prop :=
  s ∈ ["x ← 1", "y ← 2", "z ← 3", "i ← i + 2"]

-- Given statements
def statements : List String :=
  ["x ← 1, y ← 2, z ← 3", "S^2 ← 4", "i ← i + 2", "x + 1 ← x"]

-- The Lean Theorem statement that these are correct assignment statements.
theorem correct_assignment_statements (s₁ s₃ : String) (h₁ : s₁ = "x ← 1, y ← 2, z ← 3") (h₃ : s₃ = "i ← i + 2") :
  is_assignment_statement s₁ ∧ is_assignment_statement s₃ :=
by
  sorry

end NUMINAMATH_GPT_correct_assignment_statements_l1966_196600


namespace NUMINAMATH_GPT_total_area_rectABCD_l1966_196601

theorem total_area_rectABCD (BF CF : ℝ) (X Y : ℝ)
  (h1 : BF = 3 * CF)
  (h2 : 3 * X - Y - (X - Y) = 96)
  (h3 : X + 3 * X = 192) :
  X + 3 * X = 192 :=
by
  sorry

end NUMINAMATH_GPT_total_area_rectABCD_l1966_196601


namespace NUMINAMATH_GPT_cos_product_value_l1966_196637

open Real

theorem cos_product_value (α : ℝ) (h : sin α = 1 / 3) : 
  cos (π / 4 + α) * cos (π / 4 - α) = 7 / 18 :=
by
  sorry

end NUMINAMATH_GPT_cos_product_value_l1966_196637


namespace NUMINAMATH_GPT_max_value_m_n_squared_sum_l1966_196620

theorem max_value_m_n_squared_sum (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m * n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end NUMINAMATH_GPT_max_value_m_n_squared_sum_l1966_196620


namespace NUMINAMATH_GPT_bus_capacity_l1966_196686

def left_side_seats : ℕ := 15
def seats_difference : ℕ := 3
def people_per_seat : ℕ := 3
def back_seat_capacity : ℕ := 7

theorem bus_capacity : left_side_seats + (left_side_seats - seats_difference) * people_per_seat + back_seat_capacity = 88 := 
by
  sorry

end NUMINAMATH_GPT_bus_capacity_l1966_196686


namespace NUMINAMATH_GPT_cargo_per_truck_is_2_5_l1966_196662

-- Define our instance conditions
variables (x : ℝ) (n : ℕ)

-- Conditions extracted from the problem
def truck_capacity_change : Prop :=
  55 ≤ x ∧ x ≤ 64 ∧
  (x = (x / n - 0.5) * (n + 4))

-- Objective based on these conditions
theorem cargo_per_truck_is_2_5 :
  truck_capacity_change x n → (x = 60) → (n + 4 = 24) → (x / 24 = 2.5) :=
by 
  sorry

end NUMINAMATH_GPT_cargo_per_truck_is_2_5_l1966_196662


namespace NUMINAMATH_GPT_factor_expression_l1966_196674

theorem factor_expression (y : ℝ) : 
  5 * y * (y - 2) + 11 * (y - 2) = (y - 2) * (5 * y + 11) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1966_196674


namespace NUMINAMATH_GPT_smaller_angle_is_85_l1966_196695

-- Conditions
def isParallelogram (α β : ℝ) : Prop :=
  α + β = 180

def angleExceedsBy10 (α β : ℝ) : Prop :=
  β = α + 10

-- Proof Problem
theorem smaller_angle_is_85 (α β : ℝ)
  (h1 : isParallelogram α β)
  (h2 : angleExceedsBy10 α β) :
  α = 85 :=
by
  sorry

end NUMINAMATH_GPT_smaller_angle_is_85_l1966_196695


namespace NUMINAMATH_GPT_factorize_difference_of_squares_l1966_196619

theorem factorize_difference_of_squares (x : ℝ) :
  4 * x^2 - 1 = (2 * x + 1) * (2 * x - 1) :=
sorry

end NUMINAMATH_GPT_factorize_difference_of_squares_l1966_196619


namespace NUMINAMATH_GPT_no_real_roots_of_x_squared_plus_5_l1966_196623

theorem no_real_roots_of_x_squared_plus_5 : ¬ ∃ (x : ℝ), x^2 + 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_x_squared_plus_5_l1966_196623


namespace NUMINAMATH_GPT_lemango_eating_mangos_l1966_196677

theorem lemango_eating_mangos :
  ∃ (mangos_eaten : ℕ → ℕ), 
    (mangos_eaten 1 * (2^6 - 1) = 364 * (2 - 1)) ∧
    (mangos_eaten 6 = 128) :=
by
  sorry

end NUMINAMATH_GPT_lemango_eating_mangos_l1966_196677


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l1966_196653

-- Define the inequality for part (1)
def ineq_part1 (x : ℝ) : Prop := 1 - (4 / (x + 1)) < 0

-- Define the solution set P for part (1)
def P (x : ℝ) : Prop := -1 < x ∧ x < 3

-- Prove that the solution set for the inequality is P
theorem part1_solution :
  ∀ (x : ℝ), ineq_part1 x ↔ P x :=
by
  -- proof omitted
  sorry

-- Define the inequality for part (2)
def ineq_part2 (x : ℝ) : Prop := abs (x + 2) < 3

-- Define the solution set Q for part (2)
def Q (x : ℝ) : Prop := -5 < x ∧ x < 1

-- Define P as depending on some parameter a
def P_param (a : ℝ) (x : ℝ) : Prop := -1 < x ∧ x < a

-- Prove the range of a given P ∪ Q = Q 
theorem part2_solution :
  ∀ a : ℝ, (∀ x : ℝ, (P_param a x ∨ Q x) ↔ Q x) → 
    (0 < a ∧ a ≤ 1) :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l1966_196653


namespace NUMINAMATH_GPT_find_N_l1966_196610

theorem find_N (x N : ℝ) (h1 : x + 1 / x = N) (h2 : x^2 + 1 / x^2 = 2) : N = 2 :=
sorry

end NUMINAMATH_GPT_find_N_l1966_196610


namespace NUMINAMATH_GPT_probability_non_adjacent_l1966_196647

def total_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n m 

def non_adjacent_arrangements (n m : ℕ) : ℕ :=
  Nat.choose n (m - 1)

def probability_zeros_non_adjacent (n m : ℕ) : ℚ :=
  (non_adjacent_arrangements n m : ℚ) / (total_arrangements n m : ℚ)

theorem probability_non_adjacent (a b : ℕ) (h₁ : a = 4) (h₂ : b = 2) :
  probability_zeros_non_adjacent 5 2 = 2 / 3 := 
by 
  rw [probability_zeros_non_adjacent]
  rw [non_adjacent_arrangements, total_arrangements]
  sorry

end NUMINAMATH_GPT_probability_non_adjacent_l1966_196647


namespace NUMINAMATH_GPT_range_a_monotonically_increasing_l1966_196646

def g (a x : ℝ) : ℝ := a * x^3 + a * x^2 + x

theorem range_a_monotonically_increasing (a : ℝ) : 
  (∀ x : ℝ, 3 * a * x^2 + 2 * a * x + 1 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 3) := 
sorry

end NUMINAMATH_GPT_range_a_monotonically_increasing_l1966_196646


namespace NUMINAMATH_GPT_maria_change_l1966_196612

def cost_per_apple : ℝ := 0.75
def number_of_apples : ℕ := 5
def amount_paid : ℝ := 10.0
def total_cost := number_of_apples * cost_per_apple
def change_received := amount_paid - total_cost

theorem maria_change :
  change_received = 6.25 :=
sorry

end NUMINAMATH_GPT_maria_change_l1966_196612


namespace NUMINAMATH_GPT_geometric_seq_a5_value_l1966_196645

theorem geometric_seq_a5_value 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_seq : ∀ n : ℕ, a (n+1) = a n * q)
  (h_pos : ∀ n : ℕ, a n > 0) 
  (h1 : a 1 * a 8 = 4 * a 5)
  (h2 : (a 4 + 2 * a 6) / 2 = 18) 
  : a 5 = 16 := 
sorry

end NUMINAMATH_GPT_geometric_seq_a5_value_l1966_196645


namespace NUMINAMATH_GPT_geometric_sequence_sum_8_l1966_196655

variable {a : ℝ} 

-- conditions
def geometric_series_sum_4 (r : ℝ) (a : ℝ) : ℝ :=
  a + a * r + a * r^2 + a * r^3

def geometric_series_sum_8 (r : ℝ) (a : ℝ) : ℝ :=
  a + a * r + a * r^2 + a * r^3 + a * r^4 + a * r^5 + a * r^6 + a * r^7

theorem geometric_sequence_sum_8 (r : ℝ) (S4 : ℝ) (S8 : ℝ) (hr : r = 2) (hS4 : S4 = 1) :
  (∃ a : ℝ, geometric_series_sum_4 r a = S4 ∧ geometric_series_sum_8 r a = S8) → S8 = 17 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_8_l1966_196655


namespace NUMINAMATH_GPT_equivalent_solution_eq1_eqC_l1966_196643

-- Define the given equation
def eq1 (x y : ℝ) : Prop := 4 * x - 8 * y - 5 = 0

-- Define the candidate equations
def eqA (x y : ℝ) : Prop := 8 * x - 8 * y - 10 = 0
def eqB (x y : ℝ) : Prop := 8 * x - 16 * y - 5 = 0
def eqC (x y : ℝ) : Prop := 8 * x - 16 * y - 10 = 0
def eqD (x y : ℝ) : Prop := 12 * x - 24 * y - 10 = 0

-- The theorem that we need to prove
theorem equivalent_solution_eq1_eqC : ∀ x y, eq1 x y ↔ eqC x y :=
by
  sorry

end NUMINAMATH_GPT_equivalent_solution_eq1_eqC_l1966_196643
