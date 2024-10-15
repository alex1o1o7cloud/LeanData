import Mathlib

namespace NUMINAMATH_GPT_distance_to_left_focus_l1227_122798

theorem distance_to_left_focus (P : ℝ × ℝ) 
  (h1 : P.1^2 / 100 + P.2^2 / 36 = 1) 
  (h2 : dist P (50 - 100 / 9, P.2) = 17 / 2) :
  dist P (-50 - 100 / 9, P.2) = 66 / 5 :=
sorry

end NUMINAMATH_GPT_distance_to_left_focus_l1227_122798


namespace NUMINAMATH_GPT_isosceles_triangle_of_cosine_equality_l1227_122746

variable {A B C : ℝ}
variable {a b c : ℝ}

/-- Prove that in triangle ABC, if a*cos(B) = b*cos(A), then a = b implying A = B --/
theorem isosceles_triangle_of_cosine_equality 
(h1 : a * Real.cos B = b * Real.cos A) :
a = b :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_of_cosine_equality_l1227_122746


namespace NUMINAMATH_GPT_robert_salary_loss_l1227_122742

variable (S : ℝ)

theorem robert_salary_loss : 
  let decreased_salary := 0.80 * S
  let increased_salary := decreased_salary * 1.20
  let percentage_loss := 100 - (increased_salary / S) * 100
  percentage_loss = 4 :=
by
  sorry

end NUMINAMATH_GPT_robert_salary_loss_l1227_122742


namespace NUMINAMATH_GPT_num_squares_in_6x6_grid_l1227_122768

/-- Define the number of kxk squares in an nxn grid -/
def num_squares (n k : ℕ) : ℕ := (n + 1 - k) * (n + 1 - k)

/-- Prove the total number of different squares in a 6x6 grid is 86 -/
theorem num_squares_in_6x6_grid : 
  (num_squares 6 1) + (num_squares 6 2) + (num_squares 6 3) + (num_squares 6 4) = 86 :=
by sorry

end NUMINAMATH_GPT_num_squares_in_6x6_grid_l1227_122768


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1227_122704

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x + 1

theorem necessary_but_not_sufficient (a : ℝ) :
  ((a ≥ 4 ∨ a ≤ 0) ↔ (∃ x : ℝ, f a x = 0)) ∧ ¬((a ≥ 4 ∨ a ≤ 0) → (∃ x : ℝ, f a x = 0)) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1227_122704


namespace NUMINAMATH_GPT_eval_sqrt_4_8_pow_12_l1227_122760

theorem eval_sqrt_4_8_pow_12 : ((8 : ℝ)^(1 / 4))^12 = 512 :=
by
  -- This is where the proof steps would go 
  sorry

end NUMINAMATH_GPT_eval_sqrt_4_8_pow_12_l1227_122760


namespace NUMINAMATH_GPT_find_valid_pairs_l1227_122793

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def valid_pair (p q : ℕ) : Prop :=
  p < 2005 ∧ q < 2005 ∧ is_prime p ∧ is_prime q ∧ q ∣ p^2 + 8 ∧ p ∣ q^2 + 8

theorem find_valid_pairs :
  ∀ p q, valid_pair p q → (p, q) = (2, 2) ∨ (p, q) = (881, 89) ∨ (p, q) = (89, 881) :=
sorry

end NUMINAMATH_GPT_find_valid_pairs_l1227_122793


namespace NUMINAMATH_GPT_difference_of_two_numbers_l1227_122779

theorem difference_of_two_numbers
  (L : ℕ) (S : ℕ) 
  (hL : L = 1596) 
  (hS : 6 * S + 15 = 1596) : 
  L - S = 1333 := 
by
  sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l1227_122779


namespace NUMINAMATH_GPT_inequality_among_three_vars_l1227_122770

theorem inequality_among_three_vars 
  (x y z : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hz : 0 < z) 
  (h : x + y + z ≥ 3) : 
  (
    1 / (x + y + z ^ 2) + 
    1 / (y + z + x ^ 2) + 
    1 / (z + x + y ^ 2) 
  ) ≤ 1 := 
  sorry

end NUMINAMATH_GPT_inequality_among_three_vars_l1227_122770


namespace NUMINAMATH_GPT_integer_roots_of_quadratic_eq_l1227_122729

theorem integer_roots_of_quadratic_eq (a : ℤ) :
  (∃ x1 x2 : ℤ, x1 + x2 = a ∧ x1 * x2 = 9 * a) ↔
  a = 100 ∨ a = -64 ∨ a = 48 ∨ a = -12 ∨ a = 36 ∨ a = 0 :=
by sorry

end NUMINAMATH_GPT_integer_roots_of_quadratic_eq_l1227_122729


namespace NUMINAMATH_GPT_farm_needs_horse_food_per_day_l1227_122741

-- Definition of conditions
def ratio_sheep_to_horses := 4 / 7
def food_per_horse := 230
def number_of_sheep := 32

-- Number of horses based on ratio
def number_of_horses := (number_of_sheep * 7) / 4

-- Proof Statement
theorem farm_needs_horse_food_per_day :
  (number_of_horses * food_per_horse) = 12880 :=
by
  -- skipping the proof steps
  sorry

end NUMINAMATH_GPT_farm_needs_horse_food_per_day_l1227_122741


namespace NUMINAMATH_GPT_min_value_expression_ge_512_l1227_122783

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  (a^3 + 4*a^2 + a + 1) * (b^3 + 4*b^2 + b + 1) * (c^3 + 4*c^2 + c + 1) / (a * b * c)

theorem min_value_expression_ge_512 {a b c : ℝ} 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) : 
  min_value_expression a b c ≥ 512 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_ge_512_l1227_122783


namespace NUMINAMATH_GPT_minimum_seats_occupied_l1227_122796

theorem minimum_seats_occupied (total_seats : ℕ) (h : total_seats = 180) : 
  ∃ occupied_seats : ℕ, occupied_seats = 45 ∧ 
  ∀ additional_person,
    (∀ i : ℕ, i < total_seats → 
     (occupied_seats ≤ i → i < occupied_seats + 1 ∨ i > occupied_seats + 1)) →
    additional_person = occupied_seats + 1  :=
by
  sorry

end NUMINAMATH_GPT_minimum_seats_occupied_l1227_122796


namespace NUMINAMATH_GPT_find_a_l1227_122719

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 2*(a-1)*x + 2

theorem find_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≤ 2 → 2 ≤ x2 → quadratic_function a x1 ≥ quadratic_function a 2 ∧ quadratic_function a 2 ≤ quadratic_function a x2) →
  a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1227_122719


namespace NUMINAMATH_GPT_chris_mixed_raisins_l1227_122762

-- Conditions
variables (R C : ℝ)

-- 1. Chris mixed some pounds of raisins with 3 pounds of nuts.
-- 2. A pound of nuts costs 3 times as much as a pound of raisins.
-- 3. The total cost of the raisins was 0.25 of the total cost of the mixture.

-- Problem statement: Prove that R = 3 given the conditions
theorem chris_mixed_raisins :
  R * C = 0.25 * (R * C + 3 * 3 * C) → R = 3 :=
by
  sorry

end NUMINAMATH_GPT_chris_mixed_raisins_l1227_122762


namespace NUMINAMATH_GPT_problem1_problem2_l1227_122745

-- Using the conditions from a) and the correct answers from b):
-- 1. Given an angle α with a point P(-4,3) on its terminal side

theorem problem1 (α : ℝ) (x y r : ℝ) (h₁ : x = -4) (h₂ : y = 3) (h₃ : r = 5) 
  (hx : r = Real.sqrt (x^2 + y^2)) 
  (hsin : Real.sin α = y / r) 
  (hcos : Real.cos α = x / r) 
  : (Real.cos (π / 2 + α) * Real.sin (-π - α)) / (Real.cos (11 * π / 2 - α) * Real.sin (9 * π / 2 + α)) = -3 / 4 :=
by sorry

-- 2. Let k be an integer
theorem problem2 (α : ℝ) (k : ℤ)
  : (Real.sin (k * π - α) * Real.cos ((k + 1) * π - α)) / (Real.sin ((k - 1) * π + α) * Real.cos (k * π + α)) = -1 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1227_122745


namespace NUMINAMATH_GPT_value_of_expression_l1227_122774

def f (x : ℝ) : ℝ := x^2 - 3*x + 7
def g (x : ℝ) : ℝ := x + 2

theorem value_of_expression : f (g 3) - g (f 3) = 8 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1227_122774


namespace NUMINAMATH_GPT_selling_price_for_target_profit_l1227_122738

-- Defining the conditions
def purchase_price : ℝ := 200
def annual_cost : ℝ := 40000
def annual_sales_volume (x : ℝ) := 800 - x
def annual_profit (x : ℝ) : ℝ := (x - purchase_price) * annual_sales_volume x - annual_cost

-- The theorem to prove
theorem selling_price_for_target_profit : ∃ x : ℝ, annual_profit x = 40000 ∧ x = 400 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_for_target_profit_l1227_122738


namespace NUMINAMATH_GPT_minimum_value_frac_l1227_122759

theorem minimum_value_frac (x y z : ℝ) (h : 2 * x * y + y * z > 0) : 
  (x^2 + y^2 + z^2) / (2 * x * y + y * z) ≥ 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_frac_l1227_122759


namespace NUMINAMATH_GPT_max_profit_l1227_122772

noncomputable def profit (x : ℝ) : ℝ :=
  10 * (x - 40) * (100 - x)

theorem max_profit (x : ℝ) (hx : x > 40) :
  (profit 70 = 9000) ∧ ∀ y > 40, profit y ≤ 9000 := by
  sorry

end NUMINAMATH_GPT_max_profit_l1227_122772


namespace NUMINAMATH_GPT_chess_tournament_total_players_l1227_122718

theorem chess_tournament_total_players :
  ∃ n : ℕ,
    n + 12 = 35 ∧
    ∀ p : ℕ,
      (∃ pts : ℕ,
        p = n + 12 ∧
        pts = (p * (p - 1)) / 2 ∧
        pts = n^2 - n + 132) ∧
      ( ∃ (gained_half_points : ℕ → Prop),
          (∀ k ≤ 12, gained_half_points k) ∧
          (∀ k > 12, ¬ gained_half_points k)) :=
by
  sorry

end NUMINAMATH_GPT_chess_tournament_total_players_l1227_122718


namespace NUMINAMATH_GPT_max_value_expression_l1227_122750

theorem max_value_expression (x1 x2 x3 : ℝ) (h1 : x1 + x2 + x3 = 1) (h2 : 0 < x1) (h3 : 0 < x2) (h4 : 0 < x3) :
    x1 * x2^2 * x3 + x1 * x2 * x3^2 ≤ 27 / 1024 :=
sorry

end NUMINAMATH_GPT_max_value_expression_l1227_122750


namespace NUMINAMATH_GPT_factorize_xy2_minus_x_l1227_122782

theorem factorize_xy2_minus_x (x y : ℝ) : xy^2 - x = x * (y - 1) * (y + 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_xy2_minus_x_l1227_122782


namespace NUMINAMATH_GPT_add_eq_pm_three_max_sub_eq_five_l1227_122763

-- Define the conditions for m and n
variables (m n : ℤ)
def abs_m_eq_one : Prop := |m| = 1
def abs_n_eq_four : Prop := |n| = 4

-- State the first theorem regarding m + n given mn < 0
theorem add_eq_pm_three (h_m : abs_m_eq_one m) (h_n : abs_n_eq_four n) (h_mn : m * n < 0) : m + n = 3 ∨ m + n = -3 := 
sorry

-- State the second theorem regarding maximum value of m - n
theorem max_sub_eq_five (h_m : abs_m_eq_one m) (h_n : abs_n_eq_four n) : ∃ max_val, (max_val = m - n) ∧ (∀ (x : ℤ), (|m| = 1) ∧ (|n| = 4)  → (x = m - n)  → x ≤ max_val) ∧ max_val = 5 := 
sorry

end NUMINAMATH_GPT_add_eq_pm_three_max_sub_eq_five_l1227_122763


namespace NUMINAMATH_GPT_roots_polynomial_expression_l1227_122776

theorem roots_polynomial_expression (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a * b + a * c + b * c = -1)
  (h3 : a * b * c = -2) :
  a * (b - c)^2 + b * (c - a)^2 + c * (a - b)^2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_roots_polynomial_expression_l1227_122776


namespace NUMINAMATH_GPT_rakesh_fixed_deposit_percentage_l1227_122730

-- Definitions based on the problem statement
def salary : ℝ := 4000
def cash_in_hand : ℝ := 2380
def spent_on_groceries : ℝ := 0.30

-- The theorem to prove
theorem rakesh_fixed_deposit_percentage (x : ℝ) 
  (H1 : cash_in_hand = 0.70 * (salary - (x / 100) * salary)) : 
  x = 15 := 
sorry

end NUMINAMATH_GPT_rakesh_fixed_deposit_percentage_l1227_122730


namespace NUMINAMATH_GPT_sum_of_roots_l1227_122708

theorem sum_of_roots (a b c : ℝ) (h : 3 * x^2 - 7 * x + 2 = 0) : -b / a = 7 / 3 :=
by sorry

end NUMINAMATH_GPT_sum_of_roots_l1227_122708


namespace NUMINAMATH_GPT_quadratic_equation_terms_l1227_122753

theorem quadratic_equation_terms (x : ℝ) :
  (∃ a b c : ℝ, a = 3 ∧ b = -6 ∧ c = -7 ∧ a * x^2 + b * x + c = 0) →
  (∃ (a : ℝ), a = 3 ∧ ∀ (x : ℝ), 3 * x^2 - 6 * x - 7 = a * x^2 - 6 * x - 7) ∧
  (∃ (c : ℝ), c = -7 ∧ ∀ (x : ℝ), 3 * x^2 - 6 * x - 7 = 3 * x^2 - 6 * x + c) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_equation_terms_l1227_122753


namespace NUMINAMATH_GPT_find_solutions_l1227_122799

def equation (x : ℝ) : Prop :=
  (1 / (x^2 + 12*x - 8)) + (1 / (x^2 + 3*x - 8)) + (1 / (x^2 - 14*x - 8)) = 0

theorem find_solutions : {x : ℝ | equation x} = {2, -4, 1, -8} :=
  by
  sorry

end NUMINAMATH_GPT_find_solutions_l1227_122799


namespace NUMINAMATH_GPT_number_of_divisible_factorials_l1227_122784

theorem number_of_divisible_factorials:
  ∃ (count : ℕ), count = 36 ∧ ∀ n, 1 ≤ n ∧ n ≤ 50 → (∃ k : ℕ, n! = k * (n * (n + 1)) / 2) ↔ n ≤ n - 14 :=
sorry

end NUMINAMATH_GPT_number_of_divisible_factorials_l1227_122784


namespace NUMINAMATH_GPT_coefficient_x2_is_negative_40_l1227_122791

noncomputable def x2_coefficient_in_expansion (a : ℕ) : ℤ :=
  (-1)^3 * a^2 * Nat.choose 5 2

theorem coefficient_x2_is_negative_40 :
  x2_coefficient_in_expansion 2 = -40 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_x2_is_negative_40_l1227_122791


namespace NUMINAMATH_GPT_find_total_students_l1227_122752

theorem find_total_students (n : ℕ) : n < 550 ∧ n % 19 = 15 ∧ n % 17 = 10 → n = 509 :=
by 
  sorry

end NUMINAMATH_GPT_find_total_students_l1227_122752


namespace NUMINAMATH_GPT_max_saturdays_l1227_122720

theorem max_saturdays (days_in_month : ℕ) (month : string) (is_leap_year : Prop) (start_day : ℕ) : 
  (days_in_month = 29 → is_leap_year → start_day = 6 → true) ∧ -- February in a leap year starts on Saturday
  (days_in_month = 30 → (start_day = 5 ∨ start_day = 6) → true) ∧ -- 30-day months start on Friday or Saturday
  (days_in_month = 31 → (start_day = 4 ∨ start_day = 5 ∨ start_day = 6) → true) ∧ -- 31-day months start on Thursday, Friday, or Saturday
  (31 ≤ days_in_month ∧ days_in_month ≤ 28 → false) → -- Other case should be false
  ∃ n : ℕ, n = 5 := -- Maximum number of Saturdays is 5
sorry

end NUMINAMATH_GPT_max_saturdays_l1227_122720


namespace NUMINAMATH_GPT_coincide_foci_of_parabola_and_hyperbola_l1227_122726

theorem coincide_foci_of_parabola_and_hyperbola (p : ℝ) (hpos : p > 0) :
  (∃ x y : ℝ, (x, y) = (4, 0) ∧ y^2 = 2 * p * x) →
  (∃ x y : ℝ, (x, y) = (4, 0) ∧ (x^2 / 12) - (y^2 / 4) = 1) →
  p = 8 := 
sorry

end NUMINAMATH_GPT_coincide_foci_of_parabola_and_hyperbola_l1227_122726


namespace NUMINAMATH_GPT_solve_system_of_equations_l1227_122700

theorem solve_system_of_equations (x y z : ℝ) : 
  (y * z = 3 * y + 2 * z - 8) ∧
  (z * x = 4 * z + 3 * x - 8) ∧
  (x * y = 2 * x + y - 1) ↔ 
  ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 5 / 2 ∧ z = -1)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1227_122700


namespace NUMINAMATH_GPT_C_pow_eq_target_l1227_122727

open Matrix

-- Define the specific matrix C
def C : Matrix (Fin 2) (Fin 2) ℤ := !![3, 1; -4, -1]

-- Define the target matrix for the formula we need to prove
def C_power_50 : Matrix (Fin 2) (Fin 2) ℤ := !![101, 50; -200, -99]

-- Prove that C^50 equals to the target matrix
theorem C_pow_eq_target (n : ℕ) (h : n = 50) : C ^ n = C_power_50 := by
  rw [h]
  sorry

end NUMINAMATH_GPT_C_pow_eq_target_l1227_122727


namespace NUMINAMATH_GPT_train_length_correct_l1227_122751

noncomputable def length_of_train (speed_train_kmph : ℕ) (time_to_cross_bridge_sec : ℝ) (length_of_bridge_m : ℝ) : ℝ :=
let speed_train_mps := (speed_train_kmph : ℝ) * (1000 / 3600)
let total_distance := speed_train_mps * time_to_cross_bridge_sec
total_distance - length_of_bridge_m

theorem train_length_correct :
  length_of_train 90 32.99736021118311 660 = 164.9340052795778 :=
by
  have speed_train_mps : ℝ := 90 * (1000 / 3600)
  have total_distance := speed_train_mps * 32.99736021118311
  have length_of_train := total_distance - 660
  exact sorry

end NUMINAMATH_GPT_train_length_correct_l1227_122751


namespace NUMINAMATH_GPT_minimum_transfers_required_l1227_122732

def initial_quantities : List ℕ := [2, 12, 12, 12, 12]
def target_quantity := 10
def min_transfers := 4

theorem minimum_transfers_required :
  ∃ transfers : ℕ, transfers = min_transfers ∧
  ∀ quantities : List ℕ, List.sum initial_quantities = List.sum quantities →
  (∀ q ∈ quantities, q = target_quantity) :=
by
  sorry

end NUMINAMATH_GPT_minimum_transfers_required_l1227_122732


namespace NUMINAMATH_GPT_parabola_min_perimeter_l1227_122747

noncomputable def focus_of_parabola (p : ℝ) (hp : p > 0) : ℝ × ℝ :=
(1, 0)

noncomputable def A : ℝ × ℝ := (3, 2)

noncomputable def is_on_parabola (P : ℝ × ℝ) (p : ℝ) : Prop :=
P.2 ^ 2 = 2 * p * P.1

noncomputable def area_of_triangle (A P F : ℝ × ℝ) : ℝ :=
0.5 * abs (A.1 * (P.2 - F.2) + P.1 * (F.2 - A.2) + F.1 * (A.2 - P.2))

noncomputable def perimeter (A P F : ℝ × ℝ) : ℝ := 
abs (A.1 - P.1) + abs (A.1 - F.1) + abs (P.1 - F.1)

theorem parabola_min_perimeter 
  {p : ℝ} (hp : p > 0)
  (A : ℝ × ℝ) (ha : A = (3,2))
  (P : ℝ × ℝ) (hP : is_on_parabola P p)
  {F : ℝ × ℝ} (hF : F = focus_of_parabola p hp)
  (harea : area_of_triangle A P F = 1)
  (hmin : ∀ P', is_on_parabola P' p → 
    perimeter A P' F ≥ perimeter A P F) :
  abs (P.1 - F.1) = 5/2 :=
sorry

end NUMINAMATH_GPT_parabola_min_perimeter_l1227_122747


namespace NUMINAMATH_GPT_students_remaining_after_third_stop_l1227_122706

theorem students_remaining_after_third_stop
  (initial_students : ℕ)
  (third : ℚ) (stops : ℕ)
  (one_third_off : third = 1 / 3)
  (initial_students_eq : initial_students = 64)
  (stops_eq : stops = 3)
  : 64 * ((2 / 3) ^ 3) = 512 / 27 :=
by 
  sorry

end NUMINAMATH_GPT_students_remaining_after_third_stop_l1227_122706


namespace NUMINAMATH_GPT_find_e_l1227_122739

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) (h1 : 3 + d + e + f = -6)
  (h2 : - f / 3 = -6)
  (h3 : 9 = f)
  (h4 : - d / 3 = -18) : e = -72 :=
by
  sorry

end NUMINAMATH_GPT_find_e_l1227_122739


namespace NUMINAMATH_GPT_total_travel_cost_l1227_122735

noncomputable def calculate_cost : ℕ :=
  let cost_length_road :=
    (30 * 10 * 4) +  -- first segment
    (40 * 10 * 5) +  -- second segment
    (30 * 10 * 6)    -- third segment
  let cost_breadth_road :=
    (20 * 10 * 3) +  -- first segment
    (40 * 10 * 2)    -- second segment
  cost_length_road + cost_breadth_road

theorem total_travel_cost :
  calculate_cost = 6400 :=
by
  sorry

end NUMINAMATH_GPT_total_travel_cost_l1227_122735


namespace NUMINAMATH_GPT_quadratic_function_monotonicity_l1227_122705

theorem quadratic_function_monotonicity
  (a b : ℝ)
  (h1 : ∀ x y : ℝ, x ≤ y ∧ y ≤ -1 → a * x^2 + b * x + 3 ≤ a * y^2 + b * y + 3)
  (h2 : ∀ x y : ℝ, -1 ≤ x ∧ x ≤ y → a * x^2 + b * x + 3 ≥ a * y^2 + b * y + 3) :
  b = 2 * a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_quadratic_function_monotonicity_l1227_122705


namespace NUMINAMATH_GPT_infinite_unlucky_numbers_l1227_122715

def is_unlucky (n : ℕ) : Prop :=
  ¬(∃ x y : ℕ, x > 1 ∧ y > 1 ∧ (n = x^2 - 1 ∨ n = y^2 - 1))

theorem infinite_unlucky_numbers : ∀ᶠ n in at_top, is_unlucky n := sorry

end NUMINAMATH_GPT_infinite_unlucky_numbers_l1227_122715


namespace NUMINAMATH_GPT_problem_solution_l1227_122754

theorem problem_solution (x y z : ℝ)
  (h1 : 1/x + 1/y + 1/z = 2)
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 = 1) :
  1/(x*y) + 1/(y*z) + 1/(z*x) = 3/2 :=
sorry

end NUMINAMATH_GPT_problem_solution_l1227_122754


namespace NUMINAMATH_GPT_mixed_number_multiplication_l1227_122786

def mixed_to_improper (a : Int) (b : Int) (c : Int) : Rat :=
  a + (b / c)

theorem mixed_number_multiplication : 
  let a := 5
  let b := mixed_to_improper 7 2 5
  a * b = (37 : Rat) :=
by
  intros
  sorry

end NUMINAMATH_GPT_mixed_number_multiplication_l1227_122786


namespace NUMINAMATH_GPT_two_digit_number_solution_l1227_122731

theorem two_digit_number_solution : ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ 10 * x + y = 10 * 5 + 3 ∧ 10 * y + x = 10 * 3 + 5 ∧ 3 * z = 3 * 15 ∧ 2 * z = 2 * 15 := by
  sorry

end NUMINAMATH_GPT_two_digit_number_solution_l1227_122731


namespace NUMINAMATH_GPT_eating_possible_values_l1227_122785

def A : Set ℝ := {-1, 1 / 2, 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x ^ 2 = 1 ∧ a ≥ 0}

-- Full eating relation: B ⊆ A
-- Partial eating relation: (B ∩ A).Nonempty ∧ ¬(B ⊆ A ∨ A ⊆ B)

def is_full_eating (a : ℝ) : Prop := B a ⊆ A
def is_partial_eating (a : ℝ) : Prop :=
  (B a ∩ A).Nonempty ∧ ¬(B a ⊆ A ∨ A ⊆ B a)

theorem eating_possible_values :
  {a : ℝ | is_full_eating a ∨ is_partial_eating a} = {0, 1, 4} :=
by
  sorry

end NUMINAMATH_GPT_eating_possible_values_l1227_122785


namespace NUMINAMATH_GPT_simon_age_l1227_122722

theorem simon_age : 
  ∃ s : ℕ, 
  ∀ a : ℕ,
    a = 30 → 
    s = (a / 2) - 5 → 
    s = 10 := 
by
  sorry

end NUMINAMATH_GPT_simon_age_l1227_122722


namespace NUMINAMATH_GPT_factor_polynomial_l1227_122724

theorem factor_polynomial (y : ℝ) : 3 * y ^ 2 - 75 = 3 * (y - 5) * (y + 5) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1227_122724


namespace NUMINAMATH_GPT_range_of_a_l1227_122748

theorem range_of_a : 
  (∃ a : ℝ, (∃ x : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (x^2 + a ≤ a*x - 3))) ↔ (a ≥ 7) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1227_122748


namespace NUMINAMATH_GPT_length_of_first_train_is_270_04_l1227_122733

noncomputable def length_of_first_train (speed_first_train_kmph : ℕ) (speed_second_train_kmph : ℕ) 
  (time_seconds : ℕ) (length_second_train_m : ℕ) : ℕ :=
  let combined_speed_mps := ((speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600) 
  let combined_length := combined_speed_mps * time_seconds
  combined_length - length_second_train_m

theorem length_of_first_train_is_270_04 :
  length_of_first_train 120 80 9 230 = 270 :=
by
  sorry

end NUMINAMATH_GPT_length_of_first_train_is_270_04_l1227_122733


namespace NUMINAMATH_GPT_gear_C_rotation_direction_gear_C_rotation_count_l1227_122764

/-- Definition of the radii of the gears -/
def radius_A : ℝ := 15
def radius_B : ℝ := 10 
def radius_C : ℝ := 5

/-- Gear \( A \) drives gear \( B \) and gear \( B \) drives gear \( C \) -/
def drives (x y : ℝ) := x * y

/-- Direction of rotation of gear \( C \) when gear \( A \) rotates clockwise -/
theorem gear_C_rotation_direction : drives radius_A radius_B = drives radius_C radius_B → drives radius_A radius_B > 0 → drives radius_C radius_B > 0 := by
  sorry

/-- Number of rotations of gear \( C \) when gear \( A \) makes one complete turn -/
theorem gear_C_rotation_count : ∀ n : ℝ, drives radius_A radius_B = drives radius_C radius_B → (n * radius_A)*(radius_B / radius_C) = 3 * n := by
  sorry

end NUMINAMATH_GPT_gear_C_rotation_direction_gear_C_rotation_count_l1227_122764


namespace NUMINAMATH_GPT_bus_problem_initial_buses_passengers_l1227_122702

theorem bus_problem_initial_buses_passengers : 
  ∃ (m n : ℕ), m ≥ 2 ∧ n ≤ 32 ∧ 22 * m + 1 = n * (m - 1) ∧ n * (m - 1) = 529 ∧ m = 24 :=
sorry

end NUMINAMATH_GPT_bus_problem_initial_buses_passengers_l1227_122702


namespace NUMINAMATH_GPT_hcf_of_48_and_64_is_16_l1227_122712

theorem hcf_of_48_and_64_is_16
  (lcm_value : Nat)
  (hcf_value : Nat)
  (a : Nat)
  (b : Nat)
  (h_lcm : lcm_value = Nat.lcm a b)
  (hcf_def : hcf_value = Nat.gcd a b)
  (h_lcm_value : lcm_value = 192)
  (h_a : a = 48)
  (h_b : b = 64)
  : hcf_value = 16 := by
  sorry

end NUMINAMATH_GPT_hcf_of_48_and_64_is_16_l1227_122712


namespace NUMINAMATH_GPT_dollars_sum_l1227_122777

theorem dollars_sum : 
  (5 / 8 : ℝ) + (2 / 5) = 1.025 :=
by
  sorry

end NUMINAMATH_GPT_dollars_sum_l1227_122777


namespace NUMINAMATH_GPT_number_of_chinese_l1227_122769

theorem number_of_chinese (total americans australians chinese : ℕ) 
    (h_total : total = 49)
    (h_americans : americans = 16)
    (h_australians : australians = 11)
    (h_chinese : chinese = total - americans - australians) :
    chinese = 22 :=
by
    rw [h_total, h_americans, h_australians] at h_chinese
    exact h_chinese

end NUMINAMATH_GPT_number_of_chinese_l1227_122769


namespace NUMINAMATH_GPT_union_of_A_and_B_l1227_122744

/-- Given sets A and B defined as follows: A = {x | -1 <= x <= 3} and B = {x | 0 < x < 4}.
Prove that their union A ∪ B is the interval [-1, 4). -/
theorem union_of_A_and_B :
  let A := {x : ℝ | -1 ≤ x ∧ x ≤ 3}
  let B := {x : ℝ | 0 < x ∧ x < 4}
  A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 4} :=
by
  sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1227_122744


namespace NUMINAMATH_GPT_Craig_initial_apples_l1227_122781

variable (j : ℕ) (shared : ℕ) (left : ℕ)

theorem Craig_initial_apples (HJ : j = 11) (HS : shared = 7) (HL : left = 13) :
  shared + left = 20 := by
  sorry

end NUMINAMATH_GPT_Craig_initial_apples_l1227_122781


namespace NUMINAMATH_GPT_sum_of_three_integers_with_product_5_pow_4_l1227_122773

noncomputable def a : ℕ := 1
noncomputable def b : ℕ := 5
noncomputable def c : ℕ := 125

theorem sum_of_three_integers_with_product_5_pow_4 (h : a * b * c = 5^4) : 
  a + b + c = 131 := by
  have ha : a = 1 := rfl
  have hb : b = 5 := rfl
  have hc : c = 125 := rfl
  rw [ha, hb, hc, mul_assoc] at h
  exact sorry

end NUMINAMATH_GPT_sum_of_three_integers_with_product_5_pow_4_l1227_122773


namespace NUMINAMATH_GPT_inequality_solution_set_l1227_122717

theorem inequality_solution_set :
  { x : ℝ | -x^2 + 2*x > 0 } = { x : ℝ | 0 < x ∧ x < 2 } :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1227_122717


namespace NUMINAMATH_GPT_fraction_simplification_l1227_122761

theorem fraction_simplification : 
  ((2 * 7) * (6 * 14)) / ((14 * 6) * (2 * 7)) = 1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l1227_122761


namespace NUMINAMATH_GPT_smallest_n_not_divisible_by_10_l1227_122775

theorem smallest_n_not_divisible_by_10 :
  ∃ n > 2016, (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := 
sorry

end NUMINAMATH_GPT_smallest_n_not_divisible_by_10_l1227_122775


namespace NUMINAMATH_GPT_log_product_identity_l1227_122737

theorem log_product_identity :
    (Real.log 9 / Real.log 8) * (Real.log 32 / Real.log 9) = 5 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_log_product_identity_l1227_122737


namespace NUMINAMATH_GPT_point_on_y_axis_l1227_122794

theorem point_on_y_axis (a : ℝ) :
  (a + 2 = 0) -> a = -2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_point_on_y_axis_l1227_122794


namespace NUMINAMATH_GPT_sum_of_first_2009_terms_l1227_122707

variable (a : ℕ → ℝ) (d : ℝ)

-- conditions: arithmetic sequence and specific sum condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def sum_condition (a : ℕ → ℝ) : Prop :=
  a 1004 + a 1005 + a 1006 = 3

-- sum of the first 2009 terms
noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n.succ * (a 0 + a n.succ) / 2)

-- proof problem
theorem sum_of_first_2009_terms (h1 : is_arithmetic_sequence a d) (h2 : sum_condition a) :
  sum_first_n_terms a 2008 = 2009 :=
sorry

end NUMINAMATH_GPT_sum_of_first_2009_terms_l1227_122707


namespace NUMINAMATH_GPT_cos_alpha_value_l1227_122723

theorem cos_alpha_value (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
  (hcos : Real.cos (α + π / 3) = -2 / 3) : Real.cos α = (Real.sqrt 15 - 2) / 6 := 
  by 
  sorry

end NUMINAMATH_GPT_cos_alpha_value_l1227_122723


namespace NUMINAMATH_GPT_monotonicity_of_g_l1227_122736

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) / (x ^ 2)

theorem monotonicity_of_g (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x : ℝ, x > 0 → (g a x) < (g a (x + 1))) ∧ (∀ x : ℝ, x < 0 → (g a x) > (g a (x - 1))) :=
  sorry

end NUMINAMATH_GPT_monotonicity_of_g_l1227_122736


namespace NUMINAMATH_GPT_simplify_expression_l1227_122765

theorem simplify_expression (x : ℝ) : 5 * x + 7 * x - 3 * x = 9 * x :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1227_122765


namespace NUMINAMATH_GPT_cookie_cost_proof_l1227_122789

def cost_per_cookie (total_spent : ℕ) (days : ℕ) (cookies_per_day : ℕ) : ℕ :=
  total_spent / (days * cookies_per_day)

theorem cookie_cost_proof : cost_per_cookie 1395 31 3 = 15 := by
  sorry

end NUMINAMATH_GPT_cookie_cost_proof_l1227_122789


namespace NUMINAMATH_GPT_union_of_A_and_B_l1227_122756

def A : Set ℕ := {1, 2, 3, 5, 7}
def B : Set ℕ := {3, 4, 5}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4, 5, 7} :=
by sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1227_122756


namespace NUMINAMATH_GPT_dad_eyes_l1227_122714

def mom_eyes : ℕ := 1
def kids_eyes : ℕ := 3 * 4
def total_eyes : ℕ := 16

theorem dad_eyes :
  mom_eyes + kids_eyes + (total_eyes - (mom_eyes + kids_eyes)) = total_eyes :=
by 
  -- The proof part is omitted as per instructions
  sorry

example : (total_eyes - (mom_eyes + kids_eyes)) = 3 :=
by 
  -- The proof part is omitted as per instructions
  sorry

end NUMINAMATH_GPT_dad_eyes_l1227_122714


namespace NUMINAMATH_GPT_factor_is_given_sum_l1227_122780

theorem factor_is_given_sum (P Q : ℤ)
  (h1 : ∀ x : ℝ, (x^2 + 3 * x + 7) * (x^2 + (-3) * x + 7) = x^4 + P * x^2 + Q) :
  P + Q = 54 := 
sorry

end NUMINAMATH_GPT_factor_is_given_sum_l1227_122780


namespace NUMINAMATH_GPT_triangle_right_if_condition_l1227_122767

variables (a b c : ℝ) (A B C : ℝ)
-- Condition: Given 1 + cos A = (b + c) / c
axiom h1 : 1 + Real.cos A = (b + c) / c 

-- To prove: a^2 + b^2 = c^2
theorem triangle_right_if_condition (h1 : 1 + Real.cos A = (b + c) / c) : a^2 + b^2 = c^2 :=
  sorry

end NUMINAMATH_GPT_triangle_right_if_condition_l1227_122767


namespace NUMINAMATH_GPT_mutually_exclusive_A_C_l1227_122795

-- Definitions based on the given conditions
def all_not_defective (A : Prop) : Prop := A
def all_defective (B : Prop) : Prop := B
def at_least_one_defective (C : Prop) : Prop := C

-- Theorem to prove A and C are mutually exclusive
theorem mutually_exclusive_A_C (A B C : Prop) 
  (H1 : all_not_defective A) 
  (H2 : all_defective B) 
  (H3 : at_least_one_defective C) : 
  (A ∧ C) → False :=
sorry

end NUMINAMATH_GPT_mutually_exclusive_A_C_l1227_122795


namespace NUMINAMATH_GPT_hyperbola_asymptotes_eq_l1227_122757

theorem hyperbola_asymptotes_eq (M : ℝ) :
  (4 / 3 = 5 / Real.sqrt M) → M = 225 / 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_eq_l1227_122757


namespace NUMINAMATH_GPT_polygon_sides_l1227_122725

theorem polygon_sides (n : ℕ) : 
  (∃ D, D = 104) ∧ (D = (n - 1) * (n - 4) / 2)  → n = 17 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1227_122725


namespace NUMINAMATH_GPT_contribution_per_person_correct_l1227_122790

-- Definitions from conditions
def total_fundraising_goal : ℕ := 2400
def number_of_participants : ℕ := 8
def administrative_fee_per_person : ℕ := 20

-- Desired answer
def total_contribution_per_person : ℕ := total_fundraising_goal / number_of_participants + administrative_fee_per_person

-- Proof statement
theorem contribution_per_person_correct :
  total_contribution_per_person = 320 :=
by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_contribution_per_person_correct_l1227_122790


namespace NUMINAMATH_GPT_smallest_integer_n_l1227_122721

theorem smallest_integer_n (n : ℕ) : (1 / 2 : ℝ) < n / 9 ↔ n ≥ 5 := 
sorry

end NUMINAMATH_GPT_smallest_integer_n_l1227_122721


namespace NUMINAMATH_GPT_value_of_1_plus_i_cubed_l1227_122716

-- Definition of the imaginary unit i
def i : ℂ := Complex.I

-- Condition: i^2 = -1
lemma i_squared : i ^ 2 = -1 := by
  unfold i
  exact Complex.I_sq

-- The proof statement
theorem value_of_1_plus_i_cubed : 1 + i ^ 3 = 1 - i := by
  sorry

end NUMINAMATH_GPT_value_of_1_plus_i_cubed_l1227_122716


namespace NUMINAMATH_GPT_truth_values_of_p_and_q_l1227_122758

variable {p q : Prop}

theorem truth_values_of_p_and_q (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by
  sorry

end NUMINAMATH_GPT_truth_values_of_p_and_q_l1227_122758


namespace NUMINAMATH_GPT_sheet_length_proof_l1227_122734

noncomputable def length_of_sheet (L : ℝ) : ℝ := 48

theorem sheet_length_proof (L : ℝ) (w : ℝ) (s : ℝ) (V : ℝ) (h : ℝ) (new_w : ℝ) :
  w = 36 →
  s = 8 →
  V = 5120 →
  h = s →
  new_w = w - 2 * s →
  V = (L - 2 * s) * new_w * h →
  L = 48 :=
by
  intros hw hs hV hh h_new_w h_volume
  -- conversion of the mathematical equivalent proof problem to Lean's theorem
  sorry

end NUMINAMATH_GPT_sheet_length_proof_l1227_122734


namespace NUMINAMATH_GPT_black_more_than_blue_l1227_122755

noncomputable def number_of_pencils := 8
noncomputable def number_of_blue_pens := 2 * number_of_pencils
noncomputable def number_of_red_pens := number_of_pencils - 2
noncomputable def total_pens := 48

-- Given the conditions
def satisfies_conditions (K B P : ℕ) : Prop :=
  P = number_of_pencils ∧
  B = number_of_blue_pens ∧
  K + B + number_of_red_pens = total_pens

-- Prove the number of more black pens than blue pens
theorem black_more_than_blue (K B P : ℕ) : satisfies_conditions K B P → (K - B) = 10 := by
  sorry

end NUMINAMATH_GPT_black_more_than_blue_l1227_122755


namespace NUMINAMATH_GPT_prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l1227_122778

noncomputable def P_A : ℝ := 0.5
noncomputable def P_B_not_A : ℝ := 0.3
noncomputable def P_B : ℝ := 0.6  -- given from solution step
noncomputable def P_C : ℝ := 1 - (1 - P_A) * (1 - P_B)
noncomputable def P_D : ℝ := (1 - P_A) * (1 - P_B)
noncomputable def P_E : ℝ := 3 * P_D * (P_C ^ 2)

theorem prob_insurance_A_or_B :
  P_C = 0.8 :=
by
  sorry

theorem prob_exactly_one_no_insurance_out_of_three :
  P_E = 0.384 :=
by
  sorry

end NUMINAMATH_GPT_prob_insurance_A_or_B_prob_exactly_one_no_insurance_out_of_three_l1227_122778


namespace NUMINAMATH_GPT_sunday_saturday_ratio_is_two_to_one_l1227_122797

-- Define the conditions as given in the problem
def total_pages : ℕ := 360
def saturday_morning_read : ℕ := 40
def saturday_night_read : ℕ := 10
def remaining_pages : ℕ := 210

-- Define Ethan's total pages read so far
def total_read : ℕ := total_pages - remaining_pages

-- Define pages read on Saturday
def saturday_total_read : ℕ := saturday_morning_read + saturday_night_read

-- Define pages read on Sunday
def sunday_total_read : ℕ := total_read - saturday_total_read

-- Define the ratio of pages read on Sunday to pages read on Saturday
def sunday_to_saturday_ratio : ℕ := sunday_total_read / saturday_total_read

-- Theorem statement: ratio of pages read on Sunday to pages read on Saturday is 2:1
theorem sunday_saturday_ratio_is_two_to_one : sunday_to_saturday_ratio = 2 :=
by
  -- This part should contain the detailed proof
  sorry

end NUMINAMATH_GPT_sunday_saturday_ratio_is_two_to_one_l1227_122797


namespace NUMINAMATH_GPT_quadratic_root_and_a_value_l1227_122703

theorem quadratic_root_and_a_value (a : ℝ) (h1 : (a + 3) * 0^2 - 4 * 0 + a^2 - 9 = 0) (h2 : a + 3 ≠ 0) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_and_a_value_l1227_122703


namespace NUMINAMATH_GPT_frosting_cupcakes_l1227_122743

noncomputable def rate_cagney := 1 / 25  -- Cagney's rate in cupcakes per second
noncomputable def rate_lacey := 1 / 20  -- Lacey's rate in cupcakes per second

noncomputable def break_time := 30      -- Break time in seconds
noncomputable def work_period := 180    -- Work period in seconds before a break
noncomputable def total_time := 600     -- Total time in seconds (10 minutes)

noncomputable def combined_rate := rate_cagney + rate_lacey -- Combined rate in cupcakes per second

-- Effective work time after considering breaks
noncomputable def effective_work_time :=
  total_time - (total_time / work_period) * break_time

-- Total number of cupcakes frosted in the effective work time
noncomputable def total_cupcakes := combined_rate * effective_work_time

theorem frosting_cupcakes : total_cupcakes = 48 :=
by
  sorry

end NUMINAMATH_GPT_frosting_cupcakes_l1227_122743


namespace NUMINAMATH_GPT_projection_of_b_onto_a_l1227_122711
-- Import the entire library for necessary functions and definitions.

-- Define the problem in Lean 4, using relevant conditions and statement.
theorem projection_of_b_onto_a (m : ℝ) (h : (1 : ℝ) * 3 + (Real.sqrt 3) * m = 6) : m = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_projection_of_b_onto_a_l1227_122711


namespace NUMINAMATH_GPT_smallest_positive_multiple_of_3_4_5_is_60_l1227_122713

theorem smallest_positive_multiple_of_3_4_5_is_60 :
  ∃ n : ℕ, n > 0 ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ n = 60 :=
by
  use 60
  sorry

end NUMINAMATH_GPT_smallest_positive_multiple_of_3_4_5_is_60_l1227_122713


namespace NUMINAMATH_GPT_coefficient_of_monomial_l1227_122728

theorem coefficient_of_monomial : 
  ∀ (m n : ℝ), -((2 * Real.pi) / 3) * m * (n ^ 5) = -((2 * Real.pi) / 3) * m * (n ^ 5) :=
by
  sorry

end NUMINAMATH_GPT_coefficient_of_monomial_l1227_122728


namespace NUMINAMATH_GPT_min_value_is_2_sqrt_2_l1227_122701

noncomputable def min_value (a b : ℝ) : ℝ :=
  a^2 + b^2 / (a - b)

theorem min_value_is_2_sqrt_2 (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : a * b = 1) : 
  min_value a b = 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_min_value_is_2_sqrt_2_l1227_122701


namespace NUMINAMATH_GPT_ducks_to_total_ratio_l1227_122787

-- Definitions based on the given conditions
def totalBirds : ℕ := 15
def costPerChicken : ℕ := 2
def totalCostForChickens : ℕ := 20

-- Proving the desired ratio of ducks to total number of birds
theorem ducks_to_total_ratio : (totalCostForChickens / costPerChicken) + d = totalBirds → d = 15 - (totalCostForChickens / costPerChicken) → 
  (totalCostForChickens / costPerChicken) + d = totalBirds → d = totalBirds - (totalCostForChickens / costPerChicken) →
  d = 5 → (totalBirds - (totalCostForChickens / costPerChicken)) / totalBirds = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ducks_to_total_ratio_l1227_122787


namespace NUMINAMATH_GPT_asymptotes_of_C2_l1227_122710

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def C1 (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)
noncomputable def C2 (x y : ℝ) : Prop := (y^2 / a^2 - x^2 / b^2 = 1)
noncomputable def ecc1 : ℝ := (Real.sqrt (a^2 - b^2)) / a
noncomputable def ecc2 : ℝ := (Real.sqrt (a^2 + b^2)) / a

theorem asymptotes_of_C2 :
  a > b → b > 0 → ecc1 * ecc2 = Real.sqrt 3 / 2 → by exact (∀ x y : ℝ, C2 x y → x = - Real.sqrt 2 * y ∨ x = Real.sqrt 2 * y) :=
sorry

end NUMINAMATH_GPT_asymptotes_of_C2_l1227_122710


namespace NUMINAMATH_GPT_sum_of_products_equal_l1227_122771

theorem sum_of_products_equal 
  (a1 a2 a3 b1 b2 b3 c1 c2 c3 : ℝ)
  (h1 : a1 + a2 + a3 = b1 + b2 + b3)
  (h2 : b1 + b2 + b3 = c1 + c2 + c3)
  (h3 : c1 + c2 + c3 = a1 + b1 + c1)
  (h4 : a1 + b1 + c1 = a2 + b2 + c2)
  (h5 : a2 + b2 + c2 = a3 + b3 + c3) :
  a1 * b1 * c1 + a2 * b2 * c2 + a3 * b3 * c3 = a1 * a2 * a3 + b1 * b2 * b3 + c1 * c2 * c3 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_products_equal_l1227_122771


namespace NUMINAMATH_GPT_min_arg_z_l1227_122709

noncomputable def z (x y : ℝ) := x + y * Complex.I

def satisfies_condition (x y : ℝ) : Prop :=
  Complex.abs (z x y + 3 - Real.sqrt 3 * Complex.I) = Real.sqrt 3

theorem min_arg_z (x y : ℝ) (h : satisfies_condition x y) :
  Complex.arg (z x y) = 5 * Real.pi / 6 := 
sorry

end NUMINAMATH_GPT_min_arg_z_l1227_122709


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1227_122740

theorem arithmetic_sequence_common_difference 
  (a : Nat → Int)
  (a1 : a 1 = 5)
  (a6_a8_sum : a 6 + a 8 = 58) :
  ∃ d, ∀ n, a n = 5 + (n - 1) * d ∧ d = 4 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1227_122740


namespace NUMINAMATH_GPT_product_of_roots_eq_20_l1227_122766

open Real

theorem product_of_roots_eq_20 :
  (∀ x : ℝ, (x^2 + 18 * x + 30 = 2 * sqrt (x^2 + 18 * x + 45)) → 
  (x^2 + 18 * x + 20 = 0)) → 
  ∀ α β : ℝ, (α ≠ β ∧ α * β = 20) :=
by
  intros h x hx
  sorry

end NUMINAMATH_GPT_product_of_roots_eq_20_l1227_122766


namespace NUMINAMATH_GPT_find_value_of_x_y_l1227_122788

theorem find_value_of_x_y (x y : ℝ) (h1 : |x| + x + y = 10) (h2 : |y| + x - y = 12) : x + y = 18 / 5 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_x_y_l1227_122788


namespace NUMINAMATH_GPT_find_x_l1227_122749

-- Define the vectors a and b
def vec_a : ℝ × ℝ := (3, 5)
def vec_b (x : ℝ) : ℝ × ℝ := (1, x)

-- Define what it means for two vectors to be parallel
def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (a.1 = k * b.1) ∧ (a.2 = k * b.2)

-- Given condition: vectors a and b are parallel
theorem find_x (x : ℝ) (h : vectors_parallel vec_a (vec_b x)) : x = 5 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1227_122749


namespace NUMINAMATH_GPT_find_smallest_value_l1227_122792

noncomputable def smallest_value (a b c d : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2

theorem find_smallest_value (a b c d : ℝ) (h1: a + b = 18)
  (h2: ab + c + d = 85) (h3: ad + bc = 180) (h4: cd = 104) :
  smallest_value a b c d = 484 :=
sorry

end NUMINAMATH_GPT_find_smallest_value_l1227_122792
