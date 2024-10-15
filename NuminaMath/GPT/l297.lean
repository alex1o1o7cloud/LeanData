import Mathlib

namespace NUMINAMATH_GPT_largest_root_in_range_l297_29791

-- Define the conditions for the equation parameters
variables (a0 a1 a2 : ℝ)
-- Define the conditions for the absolute value constraints
variables (h0 : |a0| < 2) (h1 : |a1| < 2) (h2 : |a2| < 2)

-- Define the equation
def cubic_equation (x : ℝ) : ℝ := x^3 + a2 * x^2 + a1 * x + a0

-- Define the property we want to prove about the largest positive root r
theorem largest_root_in_range :
  ∃ r > 0, (∃ x, cubic_equation a0 a1 a2 x = 0 ∧ r = x) ∧ (5 / 2 < r ∧ r < 3) :=
by sorry

end NUMINAMATH_GPT_largest_root_in_range_l297_29791


namespace NUMINAMATH_GPT_find_numbers_l297_29720

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end NUMINAMATH_GPT_find_numbers_l297_29720


namespace NUMINAMATH_GPT_ages_correct_l297_29765

def ages : List ℕ := [5, 8, 13, 15]
def Tanya : ℕ := 13
def Yura : ℕ := 8
def Sveta : ℕ := 5
def Lena : ℕ := 15

theorem ages_correct (h1 : Tanya ∈ ages) 
                     (h2: Yura ∈ ages)
                     (h3: Sveta ∈ ages)
                     (h4: Lena ∈ ages)
                     (h5: Tanya ≠ Yura)
                     (h6: Tanya ≠ Sveta)
                     (h7: Tanya ≠ Lena)
                     (h8: Yura ≠ Sveta)
                     (h9: Yura ≠ Lena)
                     (h10: Sveta ≠ Lena)
                     (h11: Sveta = 5)
                     (h12: Tanya > Yura)
                     (h13: (Tanya + Sveta) % 3 = 0) :
                     Tanya = 13 ∧ Yura = 8 ∧ Sveta = 5 ∧ Lena = 15 := by
  sorry

end NUMINAMATH_GPT_ages_correct_l297_29765


namespace NUMINAMATH_GPT_remaining_pie_portion_l297_29764

theorem remaining_pie_portion (Carlos_takes: ℝ) (fraction_Maria: ℝ) :
  Carlos_takes = 0.60 →
  fraction_Maria = 0.25 →
  (1 - Carlos_takes) * (1 - fraction_Maria) = 0.30 := by
  intros h1 h2
  rw [h1, h2]
  simp
  sorry

end NUMINAMATH_GPT_remaining_pie_portion_l297_29764


namespace NUMINAMATH_GPT_mary_needs_more_apples_l297_29745

theorem mary_needs_more_apples :
  let pies := 15
  let apples_per_pie := 10
  let harvested_apples := 40
  let total_apples_needed := pies * apples_per_pie
  let more_apples_needed := total_apples_needed - harvested_apples
  more_apples_needed = 110 :=
by
  sorry

end NUMINAMATH_GPT_mary_needs_more_apples_l297_29745


namespace NUMINAMATH_GPT_trig_eq_solution_l297_29725

open Real

theorem trig_eq_solution (x : ℝ) :
    (∃ k : ℤ, x = -arccos ((sqrt 13 - 1) / 4) + 2 * k * π) ∨ 
    (∃ k : ℤ, x = -arccos ((1 - sqrt 13) / 4) + 2 * k * π) ↔ 
    (cos 5 * x - cos 7 * x) / (sin 4 * x + sin 2 * x) = 2 * abs (sin 2 * x) := by
  sorry

end NUMINAMATH_GPT_trig_eq_solution_l297_29725


namespace NUMINAMATH_GPT_subcommittee_formation_l297_29772

/-- A Senate committee consists of 10 Republicans and 7 Democrats.
    The number of ways to form a subcommittee with 4 Republicans and 3 Democrats is 7350. -/
theorem subcommittee_formation :
  (Nat.choose 10 4) * (Nat.choose 7 3) = 7350 :=
by
  sorry

end NUMINAMATH_GPT_subcommittee_formation_l297_29772


namespace NUMINAMATH_GPT_moles_of_Cl2_required_l297_29786

theorem moles_of_Cl2_required (n_C2H6 n_HCl : ℕ) (balance : n_C2H6 = 3) (HCl_needed : n_HCl = 6) :
  ∃ n_Cl2 : ℕ, n_Cl2 = 9 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_Cl2_required_l297_29786


namespace NUMINAMATH_GPT_sqrt_of_six_l297_29758

theorem sqrt_of_six : Real.sqrt 6 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_GPT_sqrt_of_six_l297_29758


namespace NUMINAMATH_GPT_frequency_of_sixth_group_l297_29712

theorem frequency_of_sixth_group :
  ∀ (total_data_points : ℕ)
    (freq1 freq2 freq3 freq4 : ℕ)
    (freq5_ratio : ℝ),
    total_data_points = 40 →
    freq1 = 10 →
    freq2 = 5 →
    freq3 = 7 →
    freq4 = 6 →
    freq5_ratio = 0.10 →
    (total_data_points - (freq1 + freq2 + freq3 + freq4) - (total_data_points * freq5_ratio)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_sixth_group_l297_29712


namespace NUMINAMATH_GPT_b8_expression_l297_29707

theorem b8_expression (a b : ℕ → ℚ)
  (ha0 : a 0 = 2)
  (hb0 : b 0 = 3)
  (ha : ∀ n, a (n + 1) = (a n) ^ 2 / (b n))
  (hb : ∀ n, b (n + 1) = (b n) ^ 2 / (a n)) :
  b 8 = 3 ^ 3281 / 2 ^ 3280 :=
by
  sorry

end NUMINAMATH_GPT_b8_expression_l297_29707


namespace NUMINAMATH_GPT_power_mod_l297_29729

theorem power_mod (x n m : ℕ) : (x^n) % m = x % m := by 
  sorry

example : 5^2023 % 150 = 5 % 150 :=
by exact power_mod 5 2023 150

end NUMINAMATH_GPT_power_mod_l297_29729


namespace NUMINAMATH_GPT_number_of_boxes_needed_l297_29723

theorem number_of_boxes_needed 
  (students : ℕ) (cookies_per_student : ℕ) (cookies_per_box : ℕ) 
  (total_students : students = 134) 
  (cookies_each : cookies_per_student = 7) 
  (cookies_in_box : cookies_per_box = 28) 
  (total_cookies : students * cookies_per_student = 938)
  : Nat.ceil (938 / 28) = 34 := 
by
  sorry

end NUMINAMATH_GPT_number_of_boxes_needed_l297_29723


namespace NUMINAMATH_GPT_smallest_number_is_16_l297_29780

theorem smallest_number_is_16 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ (a + b + c) / 3 = 24 ∧ 
  (b = 25) ∧ (c = b + 6) ∧ min a (min b c) = 16 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_is_16_l297_29780


namespace NUMINAMATH_GPT_sum_of_squares_ineq_l297_29779

theorem sum_of_squares_ineq (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum_sq : a^2 + b^2 + c^2 = 3) :
  a + b + c ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_ineq_l297_29779


namespace NUMINAMATH_GPT_knight_will_be_freed_l297_29781

/-- Define a structure to hold the state of the piles -/
structure PileState where
  pile1_magical : ℕ
  pile1_non_magical : ℕ
  pile2_magical : ℕ
  pile2_non_magical : ℕ
deriving Repr

-- Function to move one coin from pile1 to pile2
def move_coin (state : PileState) : PileState :=
  if state.pile1_magical > 0 then
    { state with
      pile1_magical := state.pile1_magical - 1,
      pile2_magical := state.pile2_magical + 1 }
  else if state.pile1_non_magical > 0 then
    { state with
      pile1_non_magical := state.pile1_non_magical - 1,
      pile2_non_magical := state.pile2_non_magical + 1 }
  else
    state -- If no coins to move, the state remains unchanged

-- The initial state of the piles
def initial_state : PileState :=
  { pile1_magical := 0, pile1_non_magical := 49, pile2_magical := 50, pile2_non_magical := 1 }

-- Check if the knight can be freed (both piles have the same number of magical or non-magical coins)
def knight_free (state : PileState) : Prop :=
  state.pile1_magical = state.pile2_magical ∨ state.pile1_non_magical = state.pile2_non_magical

noncomputable def knight_can_be_freed_by_25th_day : Prop :=
  exists n : ℕ, n ≤ 25 ∧ knight_free (Nat.iterate move_coin n initial_state)

theorem knight_will_be_freed : knight_can_be_freed_by_25th_day :=
  sorry

end NUMINAMATH_GPT_knight_will_be_freed_l297_29781


namespace NUMINAMATH_GPT_true_propositions_l297_29747

-- Definitions for the propositions
def proposition1 (a b : ℝ) : Prop := a > b → a^2 > b^2
def proposition2 (a b : ℝ) : Prop := a^2 > b^2 → |a| > |b|
def proposition3 (a b c : ℝ) : Prop := (a > b ↔ a + c > b + c)

-- Theorem to state the true propositions
theorem true_propositions (a b c : ℝ) :
  -- Proposition 3 is true
  (proposition3 a b c) →
  -- Assert that the serial number of the true propositions is 3
  {3} = { i | (i = 1 ∧ proposition1 a b) ∨ (i = 2 ∧ proposition2 a b) ∨ (i = 3 ∧ proposition3 a b c)} :=
by
  sorry

end NUMINAMATH_GPT_true_propositions_l297_29747


namespace NUMINAMATH_GPT_worker_times_l297_29734

-- Define the problem
theorem worker_times (x y : ℝ) (h1 : (1 / x + 1 / y = 1 / 8)) (h2 : x = y - 12) :
    x = 24 ∧ y = 12 :=
by
  sorry

end NUMINAMATH_GPT_worker_times_l297_29734


namespace NUMINAMATH_GPT_find_f_4_l297_29798

-- Lean code to encapsulate the conditions and the goal
theorem find_f_4 (f : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), x * f y = y * f x)
  (h2 : f 12 = 24) : 
  f 4 = 8 :=
sorry

end NUMINAMATH_GPT_find_f_4_l297_29798


namespace NUMINAMATH_GPT_find_second_number_l297_29774

-- Define the two numbers A and B
variables (A B : ℝ)

-- Define the conditions
def condition1 := 0.20 * A = 0.30 * B + 80
def condition2 := A = 580

-- Define the goal
theorem find_second_number (h1 : condition1 A B) (h2 : condition2 A) : B = 120 :=
by sorry

end NUMINAMATH_GPT_find_second_number_l297_29774


namespace NUMINAMATH_GPT_cube_surface_area_calc_l297_29767

-- Edge length of the cube
def edge_length : ℝ := 7

-- Definition of the surface area formula for a cube
def surface_area (a : ℝ) : ℝ := 6 * (a ^ 2)

-- The main theorem stating the surface area of the cube with given edge length
theorem cube_surface_area_calc : surface_area edge_length = 294 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_calc_l297_29767


namespace NUMINAMATH_GPT_original_six_digit_number_is_105262_l297_29737

def is_valid_number (N : ℕ) : Prop :=
  ∃ A : ℕ, A < 100000 ∧ (N = 10 * A + 2) ∧ (200000 + A = 2 * N + 2)

theorem original_six_digit_number_is_105262 :
  ∃ N : ℕ, is_valid_number N ∧ N = 105262 :=
by
  sorry

end NUMINAMATH_GPT_original_six_digit_number_is_105262_l297_29737


namespace NUMINAMATH_GPT_printer_z_time_l297_29794

theorem printer_z_time (t_z : ℝ)
  (hx : (∀ (p : ℝ), p = 16))
  (hy : (∀ (q : ℝ), q = 12))
  (ratio : (16 / (1 /  ((1 / 12) + (1 / t_z)))) = 10 / 3) :
  t_z = 8 := by
  sorry

end NUMINAMATH_GPT_printer_z_time_l297_29794


namespace NUMINAMATH_GPT_smallest_k_for_sum_of_squares_multiple_of_360_l297_29713

theorem smallest_k_for_sum_of_squares_multiple_of_360 :
  ∃ k : ℕ, k > 0 ∧ (k * (k + 1) * (2 * k + 1)) / 6 % 360 = 0 ∧ ∀ n : ℕ, n > 0 → (n * (n + 1) * (2 * n + 1)) / 6 % 360 = 0 → k ≤ n :=
by sorry

end NUMINAMATH_GPT_smallest_k_for_sum_of_squares_multiple_of_360_l297_29713


namespace NUMINAMATH_GPT_solve_system_of_equations_l297_29761

theorem solve_system_of_equations : 
  ∃ (x y : ℤ), 2 * x + 5 * y = 8 ∧ 3 * x - 5 * y = -13 ∧ x = -1 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l297_29761


namespace NUMINAMATH_GPT_linear_function_quadrants_passing_through_l297_29728

theorem linear_function_quadrants_passing_through :
  ∀ (x : ℝ) (y : ℝ), (y = 2 * x + 3 → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0)) :=
by
  sorry

end NUMINAMATH_GPT_linear_function_quadrants_passing_through_l297_29728


namespace NUMINAMATH_GPT_minimum_value_of_F_l297_29740

theorem minimum_value_of_F (f g : ℝ → ℝ) (a b : ℝ) (h_odd_f : ∀ x, f (-x) = -f x) 
  (h_odd_g : ∀ x, g (-x) = -g x) (h_max_F : ∃ x > 0, a * f x + b * g x + 3 = 10) 
  : ∃ x < 0, a * f x + b * g x + 3 = -4 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_F_l297_29740


namespace NUMINAMATH_GPT_percent_females_employed_l297_29743

noncomputable def employed_percent (population: ℕ) : ℚ := 0.60
noncomputable def employed_males_percent (population: ℕ) : ℚ := 0.48

theorem percent_females_employed (population: ℕ) : ((employed_percent population) - (employed_males_percent population)) / (employed_percent population) = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_percent_females_employed_l297_29743


namespace NUMINAMATH_GPT_find_a_l297_29716

theorem find_a (a : ℝ) : (∀ x : ℝ, |x - a| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  intro h
  have h1 : |(-1 : ℝ) - a| = 3 := sorry
  have h2 : |(5 : ℝ) - a| = 3 := sorry
  sorry

end NUMINAMATH_GPT_find_a_l297_29716


namespace NUMINAMATH_GPT_amount_of_CaO_required_l297_29771

theorem amount_of_CaO_required (n_H2O : ℝ) (n_CaOH2 : ℝ) (n_CaO : ℝ) 
  (h1 : n_H2O = 2) (h2 : n_CaOH2 = 2) :
  n_CaO = 2 :=
by
  sorry

end NUMINAMATH_GPT_amount_of_CaO_required_l297_29771


namespace NUMINAMATH_GPT_exit_time_correct_l297_29703

def time_to_exit_wide : ℝ := 6
def time_to_exit_narrow : ℝ := 10

theorem exit_time_correct :
  ∃ x y : ℝ, x = 6 ∧ y = 10 ∧ 
  (1 / x + 1 / y = 4 / 15) ∧ 
  (y = x + 4) ∧ 
  (3.75 * (1 / x + 1 / y) = 1) :=
by
  use time_to_exit_wide
  use time_to_exit_narrow
  sorry

end NUMINAMATH_GPT_exit_time_correct_l297_29703


namespace NUMINAMATH_GPT_right_triangles_product_hypotenuses_square_l297_29754

/-- 
Given two right triangles T₁ and T₂ with areas 2 and 8 respectively. 
The hypotenuse of T₁ is congruent to one leg of T₂.
The shorter leg of T₁ is congruent to the hypotenuse of T₂.
Prove that the square of the product of the lengths of their hypotenuses is 4624.
-/
theorem right_triangles_product_hypotenuses_square :
  ∃ x y z u : ℝ, 
    (1 / 2) * x * y = 2 ∧
    (1 / 2) * y * u = 8 ∧
    x^2 + y^2 = z^2 ∧
    y^2 + (16 / y)^2 = z^2 ∧ 
    (z^2)^2 = 4624 := 
sorry

end NUMINAMATH_GPT_right_triangles_product_hypotenuses_square_l297_29754


namespace NUMINAMATH_GPT_value_of_x_plus_y_l297_29753

noncomputable def x : ℝ := 1 / 2
noncomputable def y : ℝ := 3

theorem value_of_x_plus_y
  (hx : 1 / x = 2)
  (hy : 1 / x + 3 / y = 3) :
  x + y = 7 / 2 :=
  sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l297_29753


namespace NUMINAMATH_GPT_gardener_cabbages_l297_29744

theorem gardener_cabbages (area_this_year : ℕ) (side_length_this_year : ℕ) (side_length_last_year : ℕ) (area_last_year : ℕ) (additional_cabbages : ℕ) :
  area_this_year = 9801 →
  side_length_this_year = 99 →
  side_length_last_year = side_length_this_year - 1 →
  area_last_year = side_length_last_year * side_length_last_year →
  additional_cabbages = area_this_year - area_last_year →
  additional_cabbages = 197 :=
by
  sorry

end NUMINAMATH_GPT_gardener_cabbages_l297_29744


namespace NUMINAMATH_GPT_four_fold_application_of_f_l297_29701

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 then
    x / 3
  else
    5 * x + 2

theorem four_fold_application_of_f : f (f (f (f 3))) = 187 := 
  by
    sorry

end NUMINAMATH_GPT_four_fold_application_of_f_l297_29701


namespace NUMINAMATH_GPT_right_triangle_area_l297_29783

theorem right_triangle_area (hypotenuse : ℝ)
  (angle_deg : ℝ)
  (h_hyp : hypotenuse = 10 * Real.sqrt 2)
  (h_angle : angle_deg = 45) : 
  (1 / 2) * (hypotenuse / Real.sqrt 2)^2 = 50 := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_area_l297_29783


namespace NUMINAMATH_GPT_simplify_expression_l297_29704

theorem simplify_expression (x : ℝ) : (3 * x + 6 - 5 * x) / 3 = - (2 / 3) * x + 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l297_29704


namespace NUMINAMATH_GPT_remainder_when_divided_by_6_eq_5_l297_29796

theorem remainder_when_divided_by_6_eq_5 (k : ℕ) (hk1 : k % 5 = 2) (hk2 : k < 41) (hk3 : k % 7 = 3) : k % 6 = 5 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_6_eq_5_l297_29796


namespace NUMINAMATH_GPT_bread_count_at_end_of_day_l297_29738

def initial_loaves : ℕ := 2355
def sold_loaves : ℕ := 629
def delivered_loaves : ℕ := 489

theorem bread_count_at_end_of_day : 
  initial_loaves - sold_loaves + delivered_loaves = 2215 := by
  sorry

end NUMINAMATH_GPT_bread_count_at_end_of_day_l297_29738


namespace NUMINAMATH_GPT_quadratic_roots_identity_l297_29741

noncomputable def sum_of_roots (a b : ℝ) : Prop := a + b = -10
noncomputable def product_of_roots (a b : ℝ) : Prop := a * b = 5

theorem quadratic_roots_identity (a b : ℝ)
  (h₁ : sum_of_roots a b)
  (h₂ : product_of_roots a b) :
  (a / b + b / a) = 18 :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_identity_l297_29741


namespace NUMINAMATH_GPT_polynomial_division_l297_29736

-- Define the polynomials P and D
noncomputable def P : Polynomial ℤ := 5 * Polynomial.X ^ 4 - 3 * Polynomial.X ^ 3 + 7 * Polynomial.X ^ 2 - 9 * Polynomial.X + 12
noncomputable def D : Polynomial ℤ := Polynomial.X - 3
noncomputable def Q : Polynomial ℤ := 5 * Polynomial.X ^ 3 + 12 * Polynomial.X ^ 2 + 43 * Polynomial.X + 120
def R : ℤ := 372

-- State the theorem
theorem polynomial_division :
  P = D * Q + Polynomial.C R := 
sorry

end NUMINAMATH_GPT_polynomial_division_l297_29736


namespace NUMINAMATH_GPT_find_initial_oranges_l297_29795

variable (O : ℕ)
variable (reserved_fraction : ℚ := 1 / 4)
variable (sold_fraction : ℚ := 3 / 7)
variable (rotten_oranges : ℕ := 4)
variable (good_oranges_today : ℕ := 32)

-- Define the total oranges before finding the rotten oranges
def oranges_before_rotten := good_oranges_today + rotten_oranges

-- Define the remaining fraction of oranges after reserving for friends and selling some
def remaining_fraction := (1 - reserved_fraction) * (1 - sold_fraction)

-- State the theorem to be proven
theorem find_initial_oranges (h : remaining_fraction * O = oranges_before_rotten) : O = 84 :=
sorry

end NUMINAMATH_GPT_find_initial_oranges_l297_29795


namespace NUMINAMATH_GPT_find_integer_x_l297_29793

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧
  2 < x ∧ x < 15 ∧
  -1 < x ∧ x < 7 ∧
  0 < x ∧ x < 4 ∧
  x + 1 < 5 → 
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_x_l297_29793


namespace NUMINAMATH_GPT_least_positive_integer_divisible_by_primes_gt_5_l297_29705

theorem least_positive_integer_divisible_by_primes_gt_5 : ∃ n : ℕ, n = 7 * 11 * 13 ∧ ∀ k : ℕ, (k > 0 ∧ (k % 7 = 0) ∧ (k % 11 = 0) ∧ (k % 13 = 0)) → k ≥ 1001 := 
sorry

end NUMINAMATH_GPT_least_positive_integer_divisible_by_primes_gt_5_l297_29705


namespace NUMINAMATH_GPT_value_of_f2008_plus_f2009_l297_29751

variable {f : ℤ → ℤ}

-- Conditions
axiom h1 : ∀ x : ℤ, f (-(x) + 2) = -f (x + 2)
axiom h2 : ∀ x : ℤ, f (6 - x) = f x
axiom h3 : f 3 = 2

-- The theorem to prove
theorem value_of_f2008_plus_f2009 : f 2008 + f 2009 = -2 :=
  sorry

end NUMINAMATH_GPT_value_of_f2008_plus_f2009_l297_29751


namespace NUMINAMATH_GPT_no_integral_value_2001_l297_29784

noncomputable def P (x : ℤ) : ℤ := sorry -- Polynomial definition needs to be filled in

theorem no_integral_value_2001 (a0 a1 a2 a3 a4 : ℤ) (x1 x2 x3 x4 : ℤ) :
  (P x1 = 2020) ∧ (P x2 = 2020) ∧ (P x3 = 2020) ∧ (P x4 = 2020) ∧ 
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 → 
  ¬ ∃ x : ℤ, P x = 2001 :=
sorry

end NUMINAMATH_GPT_no_integral_value_2001_l297_29784


namespace NUMINAMATH_GPT_sum_of_digits_next_l297_29724

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

theorem sum_of_digits_next (n : ℕ) (h : sum_of_digits n = 1399) : 
  sum_of_digits (n + 1) = 1402 :=
sorry

end NUMINAMATH_GPT_sum_of_digits_next_l297_29724


namespace NUMINAMATH_GPT_cyclic_inequality_l297_29730

theorem cyclic_inequality (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  (a^3 * b^3 * (a * b - a * c - b * c + c^2) +
   b^3 * c^3 * (b * c - b * a - c * a + a^2) +
   c^3 * a^3 * (c * a - c * b - a * b + b^2)) ≥ 0 :=
sorry

end NUMINAMATH_GPT_cyclic_inequality_l297_29730


namespace NUMINAMATH_GPT_probability_equal_white_black_probability_white_ge_black_l297_29726

/-- Part (a) -/
theorem probability_equal_white_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (2 * m) / (n + m)) := 
  sorry

/-- Part (b) -/
theorem probability_white_ge_black (n m : ℕ) (h : n ≥ m) :
  (∃ p, p = (n - m + 1) / (n + 1)) := 
  sorry

end NUMINAMATH_GPT_probability_equal_white_black_probability_white_ge_black_l297_29726


namespace NUMINAMATH_GPT_max_books_l297_29749

theorem max_books (price_per_book available_money : ℕ) (h1 : price_per_book = 15) (h2 : available_money = 200) :
  ∃ n : ℕ, n = 13 ∧ n ≤ available_money / price_per_book :=
by {
  sorry
}

end NUMINAMATH_GPT_max_books_l297_29749


namespace NUMINAMATH_GPT_no_more_than_four_intersection_points_l297_29789

noncomputable def conic1 (a b c d e f : ℝ) (x y : ℝ) : Prop := 
  a * x^2 + 2 * b * x * y + c * y^2 + 2 * d * x + 2 * e * y = f

noncomputable def conic2_param (P Q A : ℝ → ℝ) (t : ℝ) : ℝ × ℝ :=
  (P t / A t, Q t / A t)

theorem no_more_than_four_intersection_points (a b c d e f : ℝ)
  (P Q A : ℝ → ℝ) :
  (∃ t1 t2 t3 t4 t5,
    conic1 a b c d e f (P t1 / A t1) (Q t1 / A t1) ∧
    conic1 a b c d e f (P t2 / A t2) (Q t2 / A t2) ∧
    conic1 a b c d e f (P t3 / A t3) (Q t3 / A t3) ∧
    conic1 a b c d e f (P t4 / A t4) (Q t4 / A t4) ∧
    conic1 a b c d e f (P t5 / A t5) (Q t5 / A t5)) → false :=
sorry

end NUMINAMATH_GPT_no_more_than_four_intersection_points_l297_29789


namespace NUMINAMATH_GPT_evaluate_expression_l297_29756

theorem evaluate_expression : -25 - 7 * (4 + 2) = -67 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l297_29756


namespace NUMINAMATH_GPT_unique_solution_quadratic_l297_29739

theorem unique_solution_quadratic (q : ℚ) :
  (∃ x : ℚ, q ≠ 0 ∧ q * x^2 - 16 * x + 9 = 0) ∧ (∀ y z : ℚ, (q * y^2 - 16 * y + 9 = 0 ∧ q * z^2 - 16 * z + 9 = 0) → y = z) → q = 64 / 9 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_quadratic_l297_29739


namespace NUMINAMATH_GPT_calculate_total_money_made_l297_29759

def original_price : ℕ := 51
def discount : ℕ := 8
def num_tshirts_sold : ℕ := 130
def discounted_price : ℕ := original_price - discount
def total_money_made : ℕ := discounted_price * num_tshirts_sold

theorem calculate_total_money_made :
  total_money_made = 5590 := 
sorry

end NUMINAMATH_GPT_calculate_total_money_made_l297_29759


namespace NUMINAMATH_GPT_isosceles_triangle_largest_angle_l297_29719

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h : A = B) (h₁ : C + A + B = 180) (h₂ : C = 30) : 
  180 - 2 * 30 = 120 :=
by sorry

end NUMINAMATH_GPT_isosceles_triangle_largest_angle_l297_29719


namespace NUMINAMATH_GPT_train_length_l297_29785

theorem train_length (S L : ℝ)
  (h1 : L = S * 11)
  (h2 : L + 120 = S * 22) : 
  L = 120 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_train_length_l297_29785


namespace NUMINAMATH_GPT_B_coordinates_when_A_is_origin_l297_29708

-- Definitions based on the conditions
def A_coordinates_when_B_is_origin := (2, 5)

-- Theorem to prove the coordinates of B when A is the origin
theorem B_coordinates_when_A_is_origin (x y : ℤ) :
    A_coordinates_when_B_is_origin = (2, 5) →
    (x, y) = (-2, -5) :=
by
  intro h
  -- skipping the proof steps
  sorry

end NUMINAMATH_GPT_B_coordinates_when_A_is_origin_l297_29708


namespace NUMINAMATH_GPT_calculate_initial_budget_l297_29706

-- Definitions based on conditions
def cost_of_chicken := 12
def cost_per_pound_beef := 3
def pounds_of_beef := 5
def amount_left := 53

-- Derived definition for total cost of beef
def cost_of_beef := cost_per_pound_beef * pounds_of_beef
-- Derived definition for total spent
def total_spent := cost_of_chicken + cost_of_beef
-- Final calculation for initial budget
def initial_budget := total_spent + amount_left

-- Statement to prove
theorem calculate_initial_budget : initial_budget = 80 :=
by
  sorry

end NUMINAMATH_GPT_calculate_initial_budget_l297_29706


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l297_29773

theorem problem_1 : 
  ∀ x : ℝ, x^2 - 2 * x + 5 = (x - 1)^2 + 4 := 
sorry

theorem problem_2 (n : ℝ) (h : ∀ x : ℝ, x^2 + 2 * n * x + 3 = (x + 5)^2 - 25 + 3) : 
  n = -5 := 
sorry

theorem problem_3 (a : ℝ) (h : ∀ x : ℝ, (x^2 + 6 * x + 9) * (x^2 - 4 * x + 4) = ((x + a)^2 + b)^2) : 
  a = -1/2 := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l297_29773


namespace NUMINAMATH_GPT_sum_of_prime_factors_172944_l297_29727

theorem sum_of_prime_factors_172944 : 
  (∃ (a b c : ℕ), 2^a * 3^b * 1201^c = 172944 ∧ a = 4 ∧ b = 2 ∧ c = 1) → 2 + 3 + 1201 = 1206 := 
by 
  intros h 
  exact sorry

end NUMINAMATH_GPT_sum_of_prime_factors_172944_l297_29727


namespace NUMINAMATH_GPT_white_chocolate_bars_sold_l297_29790

theorem white_chocolate_bars_sold (W D : ℕ) (h1 : D = 15) (h2 : W / D = 4 / 3) : W = 20 :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_white_chocolate_bars_sold_l297_29790


namespace NUMINAMATH_GPT_remainder_when_divided_by_7_l297_29700

theorem remainder_when_divided_by_7 (n : ℕ) (h : (2 * n) % 7 = 4) : n % 7 = 2 :=
  by sorry

end NUMINAMATH_GPT_remainder_when_divided_by_7_l297_29700


namespace NUMINAMATH_GPT_find_k_value_l297_29732

-- Define the lines l1 and l2 with given conditions
def line1 (x y : ℝ) : Prop := x + 3 * y - 7 = 0
def line2 (k x y : ℝ) : Prop := k * x - y - 2 = 0

-- Define the condition for the quadrilateral to be circumscribed by a circle
def is_circumscribed (k : ℝ) : Prop :=
  ∃ (x y : ℝ), line1 x y ∧ line2 k x y ∧ 0 < x ∧ 0 < y

theorem find_k_value (k : ℝ) : is_circumscribed k → k = 3 := 
sorry

end NUMINAMATH_GPT_find_k_value_l297_29732


namespace NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_eq_l297_29770

open Nat

noncomputable def sum_arithmetic_sequence (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  if h: n = 0 then 0 else n * a₁ + (n * (n - 1) * d) / 2

theorem sum_first_n_terms_arithmetic_sequence_eq 
  (a₁ a₃ a₆ : ℝ) (d : ℝ) (n : ℕ) 
  (h₀ : d ≠ 0)
  (h₁ : a₁ = 4)
  (h₂ : a₃ = a₁ + 2 * d)
  (h₃ : a₆ = a₁ + 5 * d)
  (h₄ : a₃^2 = a₁ * a₆) :
  sum_arithmetic_sequence a₁ a₃ a₆ d n = (n^2 + 7 * n) / 2 := 
by
  sorry

end NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_eq_l297_29770


namespace NUMINAMATH_GPT_additional_hours_equal_five_l297_29792

-- The total hovering time constraint over two days
def total_time : ℕ := 24

-- Hovering times for each zone on the first day
def day1_mountain_time : ℕ := 3
def day1_central_time : ℕ := 4
def day1_eastern_time : ℕ := 2

-- Additional hours on the second day (variables M, C, E)
variables (M C E : ℕ)

-- The main proof statement
theorem additional_hours_equal_five 
  (h : day1_mountain_time + M + day1_central_time + C + day1_eastern_time + E = total_time) :
  M = 5 ∧ C = 5 ∧ E = 5 :=
by
  sorry

end NUMINAMATH_GPT_additional_hours_equal_five_l297_29792


namespace NUMINAMATH_GPT_min_max_product_l297_29799

noncomputable def min_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the minimum value of 3x^2 + 4xy + 3y^2
  sorry

noncomputable def max_value (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) : ℝ :=
  -- Implementation to find the maximum value of 3x^2 + 4xy + 3y^2
  sorry

theorem min_max_product (x y : ℝ) (h : 9 * x^2 + 12 * x * y + 8 * y^2 = 1) :
  min_value x y h * max_value x y h = 7 / 16 :=
sorry

end NUMINAMATH_GPT_min_max_product_l297_29799


namespace NUMINAMATH_GPT_Julio_current_age_l297_29797

theorem Julio_current_age (J : ℕ) (James_current_age : ℕ) (h1 : James_current_age = 11)
    (h2 : J + 14 = 2 * (James_current_age + 14)) : 
    J = 36 := 
by 
  sorry

end NUMINAMATH_GPT_Julio_current_age_l297_29797


namespace NUMINAMATH_GPT_problem_statement_l297_29714

theorem problem_statement (p q : ℚ) (hp : 3 / p = 4) (hq : 3 / q = 18) : p - q = 7 / 12 := 
by
  sorry

end NUMINAMATH_GPT_problem_statement_l297_29714


namespace NUMINAMATH_GPT_gibraltar_initial_population_stable_l297_29702
-- Import necessary libraries

-- Define constants based on conditions
def full_capacity := 300 * 4
def initial_population := (full_capacity / 3) - 100
def population := 300 -- This is the final answer we need to validate

-- The main theorem to prove
theorem gibraltar_initial_population_stable : initial_population = population :=
by 
  -- Proof is skipped as requested
  sorry

end NUMINAMATH_GPT_gibraltar_initial_population_stable_l297_29702


namespace NUMINAMATH_GPT_parallel_lines_m_l297_29731

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, 2 * x + 3 * y + 1 = 0 → 6 ≠ 0) ∧ 
  (∀ x y : ℝ, m * x + 6 * y - 5 = 0 → 6 ≠ 0) → 
  m = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_parallel_lines_m_l297_29731


namespace NUMINAMATH_GPT_find_number_l297_29762

-- Definitions from the conditions
def condition1 (x : ℝ) := 16 * x = 3408
def condition2 (x : ℝ) := 1.6 * x = 340.8

-- The statement to prove
theorem find_number (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x = 213 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l297_29762


namespace NUMINAMATH_GPT_doughnut_problem_l297_29755

theorem doughnut_problem :
  ∀ (total_doughnuts first_two_box_doughnuts boxes : ℕ),
  total_doughnuts = 72 →
  first_two_box_doughnuts = 12 →
  boxes = 4 →
  (total_doughnuts - 2 * first_two_box_doughnuts) / boxes = 12 :=
by
  intros total_doughnuts first_two_box_doughnuts boxes ht12 hb12 b4
  sorry

end NUMINAMATH_GPT_doughnut_problem_l297_29755


namespace NUMINAMATH_GPT_sin_A_value_l297_29763

theorem sin_A_value
  (f : ℝ → ℝ)
  (cos_B : ℝ)
  (f_C_div_2 : ℝ)
  (C_acute : Prop) :
  (∀ x, f x = Real.cos (2 * x + Real.pi / 3) + Real.sin x ^ 2) →
  cos_B = 1 / 3 →
  f (C / 2) = -1 / 4 →
  (0 < C ∧ C < Real.pi / 2) →
  Real.sin (Real.arcsin (Real.sqrt 3 / 2) + Real.arcsin (2 * Real.sqrt 2 / 3)) = (2 * Real.sqrt 2 + Real.sqrt 3) / 6 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sin_A_value_l297_29763


namespace NUMINAMATH_GPT_integral_solution_l297_29777

noncomputable def definite_integral : ℝ :=
  ∫ x in (-2 : ℝ)..(0 : ℝ), (x + 2)^2 * (Real.cos (3 * x))

theorem integral_solution :
  definite_integral = (12 - 2 * Real.sin 6) / 27 :=
sorry

end NUMINAMATH_GPT_integral_solution_l297_29777


namespace NUMINAMATH_GPT_eight_step_paths_board_l297_29782

theorem eight_step_paths_board (P Q : ℕ) (hP : P = 0) (hQ : Q = 7) : 
  ∃ (paths : ℕ), paths = 70 :=
by
  sorry

end NUMINAMATH_GPT_eight_step_paths_board_l297_29782


namespace NUMINAMATH_GPT_james_older_brother_is_16_l297_29788

variables (John James James_older_brother : ℕ)

-- Given conditions
def current_age_john : ℕ := 39
def three_years_ago_john (caj : ℕ) : ℕ := caj - 3
def twice_as_old_condition (ja : ℕ) (james_age_in_6_years : ℕ) : Prop :=
  ja = 2 * james_age_in_6_years
def james_age_in_6_years (jc : ℕ) : ℕ := jc + 6
def james_older_brother_age (jc : ℕ) : ℕ := jc + 4

-- Theorem to be proved
theorem james_older_brother_is_16
  (H1 : current_age_john = John)
  (H2 : three_years_ago_john current_age_john = 36)
  (H3 : twice_as_old_condition 36 (james_age_in_6_years James))
  (H4 : james_older_brother_age James = James_older_brother) :
  James_older_brother = 16 := sorry

end NUMINAMATH_GPT_james_older_brother_is_16_l297_29788


namespace NUMINAMATH_GPT_diagonals_in_decagon_l297_29768

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem diagonals_in_decagon : number_of_diagonals 10 = 35 := by
  sorry

end NUMINAMATH_GPT_diagonals_in_decagon_l297_29768


namespace NUMINAMATH_GPT_algebraic_expression_value_l297_29715

theorem algebraic_expression_value (m: ℝ) (h: m^2 + m - 1 = 0) : 2023 - m^2 - m = 2022 := 
by 
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l297_29715


namespace NUMINAMATH_GPT_marias_profit_l297_29746

theorem marias_profit 
  (initial_loaves : ℕ)
  (morning_price : ℝ)
  (afternoon_discount : ℝ)
  (late_afternoon_price : ℝ)
  (cost_per_loaf : ℝ)
  (loaves_sold_morning : ℕ)
  (loaves_sold_afternoon : ℕ)
  (loaves_remaining : ℕ)
  (revenue_morning : ℝ)
  (revenue_afternoon : ℝ)
  (revenue_late_afternoon : ℝ)
  (total_revenue : ℝ)
  (total_cost : ℝ)
  (profit : ℝ) :
  initial_loaves = 60 →
  morning_price = 3.0 →
  afternoon_discount = 0.75 →
  late_afternoon_price = 1.50 →
  cost_per_loaf = 1.0 →
  loaves_sold_morning = initial_loaves / 3 →
  loaves_sold_afternoon = (initial_loaves - loaves_sold_morning) / 2 →
  loaves_remaining = initial_loaves - loaves_sold_morning - loaves_sold_afternoon →
  revenue_morning = loaves_sold_morning * morning_price →
  revenue_afternoon = loaves_sold_afternoon * (afternoon_discount * morning_price) →
  revenue_late_afternoon = loaves_remaining * late_afternoon_price →
  total_revenue = revenue_morning + revenue_afternoon + revenue_late_afternoon →
  total_cost = initial_loaves * cost_per_loaf →
  profit = total_revenue - total_cost →
  profit = 75 := sorry

end NUMINAMATH_GPT_marias_profit_l297_29746


namespace NUMINAMATH_GPT_solve_equation_naturals_l297_29778

theorem solve_equation_naturals :
  ∀ (X Y Z : ℕ), X^Y + Y^Z = X * Y * Z ↔ 
    (X = 1 ∧ Y = 1 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 2) ∨ 
    (X = 2 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 3) ∨ 
    (X = 4 ∧ Y = 2 ∧ Z = 4) := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_naturals_l297_29778


namespace NUMINAMATH_GPT_incorrect_average_initially_calculated_l297_29733

theorem incorrect_average_initially_calculated :
  ∀ (S' S : ℕ) (n : ℕ) (incorrect_correct_difference : ℕ),
  n = 10 →
  incorrect_correct_difference = 30 →
  S = 200 →
  S' = S - incorrect_correct_difference →
  (S' / n) = 17 :=
by
  intros S' S n incorrect_correct_difference h_n h_diff h_S h_S' 
  sorry

end NUMINAMATH_GPT_incorrect_average_initially_calculated_l297_29733


namespace NUMINAMATH_GPT_find_polynomial_l297_29776

-- Define the polynomial conditions
structure CubicPolynomial :=
  (P : ℝ → ℝ)
  (P0 : ℝ)
  (P1 : ℝ)
  (P2 : ℝ)
  (P3 : ℝ)
  (cubic_eq : ∀ x, P x = P0 + P1 * x + P2 * x^2 + P3 * x^3)

theorem find_polynomial (P : CubicPolynomial) (h_neg1 : P.P (-1) = 2) (h0 : P.P 0 = 3) (h1 : P.P 1 = 1) (h2 : P.P 2 = 15) :
  ∀ x, P.P x = 3 + x - 2 * x^2 - x^3 :=
sorry

end NUMINAMATH_GPT_find_polynomial_l297_29776


namespace NUMINAMATH_GPT_chickens_and_sheep_are_ten_l297_29722

noncomputable def chickens_and_sheep_problem (C S : ℕ) : Prop :=
  (C + 4 * S = 2 * C) ∧ (2 * C + 4 * (S - 4) = 16 * (S - 4)) → (S + 2 = 10)

theorem chickens_and_sheep_are_ten (C S : ℕ) : chickens_and_sheep_problem C S :=
sorry

end NUMINAMATH_GPT_chickens_and_sheep_are_ten_l297_29722


namespace NUMINAMATH_GPT_regions_formed_l297_29752

theorem regions_formed (radii : ℕ) (concentric_circles : ℕ) (total_regions : ℕ) 
  (h_radii : radii = 16) (h_concentric_circles : concentric_circles = 10) 
  (h_total_regions : total_regions = radii * (concentric_circles + 1)) : 
  total_regions = 176 := 
by
  rw [h_radii, h_concentric_circles] at h_total_regions
  exact h_total_regions

end NUMINAMATH_GPT_regions_formed_l297_29752


namespace NUMINAMATH_GPT_inequality_sum_squares_l297_29711

theorem inequality_sum_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 :=
sorry

end NUMINAMATH_GPT_inequality_sum_squares_l297_29711


namespace NUMINAMATH_GPT_volume_triangular_pyramid_correctness_l297_29769

noncomputable def volume_of_regular_triangular_pyramid 
  (a α l : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α

theorem volume_triangular_pyramid_correctness (a α l : ℝ) : volume_of_regular_triangular_pyramid a α l =
  (a ^ 3 * Real.sqrt 3 / 8) * Real.tan α := 
sorry

end NUMINAMATH_GPT_volume_triangular_pyramid_correctness_l297_29769


namespace NUMINAMATH_GPT_base9_square_multiple_of_3_ab4c_l297_29717

theorem base9_square_multiple_of_3_ab4c (a b c : ℕ) (N : ℕ) (h1 : a ≠ 0)
  (h2 : N = a * 9^3 + b * 9^2 + 4 * 9 + c)
  (h3 : ∃ k : ℕ, N = k^2)
  (h4 : N % 3 = 0) :
  c = 0 :=
sorry

end NUMINAMATH_GPT_base9_square_multiple_of_3_ab4c_l297_29717


namespace NUMINAMATH_GPT_square_neg_2x_squared_l297_29787

theorem square_neg_2x_squared (x : ℝ) : (-2 * x ^ 2) ^ 2 = 4 * x ^ 4 :=
by
  sorry

end NUMINAMATH_GPT_square_neg_2x_squared_l297_29787


namespace NUMINAMATH_GPT_percentage_equivalence_l297_29748

theorem percentage_equivalence (A B C P : ℝ)
  (hA : A = 0.80 * 600)
  (hB : B = 480)
  (hC : C = 960)
  (hP : P = (B / C) * 100) :
  A = P * 10 :=  -- Since P is the percentage, we use it to relate A to C
sorry

end NUMINAMATH_GPT_percentage_equivalence_l297_29748


namespace NUMINAMATH_GPT_min_radius_circle_condition_l297_29735

theorem min_radius_circle_condition (r : ℝ) (a b : ℝ) 
    (h_circle : (a - (r + 1))^2 + b^2 = r^2)
    (h_condition : b^2 ≥ 4 * a) :
    r ≥ 4 := 
sorry

end NUMINAMATH_GPT_min_radius_circle_condition_l297_29735


namespace NUMINAMATH_GPT_flower_bee_difference_proof_l297_29718

variable (flowers bees : ℕ)

def flowers_bees_difference (flowers bees : ℕ) : ℕ :=
  flowers - bees

theorem flower_bee_difference_proof : flowers_bees_difference 5 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_flower_bee_difference_proof_l297_29718


namespace NUMINAMATH_GPT_find_q_l297_29775

noncomputable def q_value (m q : ℕ) : Prop := 
  ((1 ^ m) / (5 ^ m)) * ((1 ^ 16) / (4 ^ 16)) = 1 / (q * 10 ^ 31)

theorem find_q (m : ℕ) (q : ℕ) (h1 : m = 31) (h2 : q_value m q) : q = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l297_29775


namespace NUMINAMATH_GPT_remainder_when_2519_divided_by_3_l297_29721

theorem remainder_when_2519_divided_by_3 :
  2519 % 3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_2519_divided_by_3_l297_29721


namespace NUMINAMATH_GPT_farm_produce_weeks_l297_29760

def eggs_needed_per_week (saly_eggs ben_eggs ked_eggs : ℕ) : ℕ :=
  saly_eggs + ben_eggs + ked_eggs

def number_of_weeks (total_eggs : ℕ) (weekly_eggs : ℕ) : ℕ :=
  total_eggs / weekly_eggs

theorem farm_produce_weeks :
  let saly_eggs := 10
  let ben_eggs := 14
  let ked_eggs := 14 / 2
  let total_eggs := 124
  let weekly_eggs := eggs_needed_per_week saly_eggs ben_eggs ked_eggs
  number_of_weeks total_eggs weekly_eggs = 4 :=
by
  sorry 

end NUMINAMATH_GPT_farm_produce_weeks_l297_29760


namespace NUMINAMATH_GPT_case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l297_29766

-- Conditions for Case (a)
def corner_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is a corner cell
  sorry

theorem case_a_second_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  corner_cell board starting_cell → 
  player = 2 :=
by
  sorry
  
-- Conditions for Case (b)
def initial_setup_according_to_figure (board : Type) (starting_cell : board) : Prop :=
  -- definition to determine if a cell setup matches the figure
  sorry

theorem case_b_first_player_wins (board : Type) (starting_cell : board) (player : ℕ) :
  initial_setup_according_to_figure board starting_cell → 
  player = 1 :=
by
  sorry

-- Conditions for Case (c)
def black_cell (board : Type) (cell : board) : Prop :=
  -- definition to determine if a cell is black
  sorry

theorem case_c_winner_based_on_cell_color (board : Type) (starting_cell : board) (player : ℕ) :
  (black_cell board starting_cell → player = 1) ∧ (¬ black_cell board starting_cell → player = 2) :=
by
  sorry
  
-- Conditions for Case (d)
def same_starting_cell_two_games (board : Type) (starting_cell : board) : Prop :=
  -- definition for same starting cell but different outcomes in games
  sorry

theorem case_d_examples (board : Type) (starting_cell : board) (player1 player2 : ℕ) :
  (same_starting_cell_two_games board starting_cell → (player1 = 1 ∧ player2 = 2)) ∨ 
  (same_starting_cell_two_games board starting_cell → (player1 = 2 ∧ player2 = 1)) :=
by
  sorry

end NUMINAMATH_GPT_case_a_second_player_wins_case_b_first_player_wins_case_c_winner_based_on_cell_color_case_d_examples_l297_29766


namespace NUMINAMATH_GPT_sum_of_first_n_terms_l297_29710

variable (a : ℕ → ℤ) (b : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a_n_arith : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom a_3 : a 3 = -6
axiom a_6 : a 6 = 0
axiom b_1 : b 1 = -8
axiom b_2 : b 2 = a 1 + a 2 + a 3

-- Correct answer to prove
theorem sum_of_first_n_terms : S n = 4 * (1 - 3^n) := sorry

end NUMINAMATH_GPT_sum_of_first_n_terms_l297_29710


namespace NUMINAMATH_GPT_star_is_addition_l297_29757

variable {α : Type} [AddCommGroup α]

-- Define the binary operation star
variable (star : α → α → α)

-- Define the condition given in the problem
axiom star_condition : ∀ (a b c : α), star (star a b) c = a + b + c

-- Prove that star is the same as usual addition
theorem star_is_addition : ∀ (a b : α), star a b = a + b :=
  sorry

end NUMINAMATH_GPT_star_is_addition_l297_29757


namespace NUMINAMATH_GPT_casey_nail_decorating_time_l297_29750

/-- Given the conditions:
1. Casey wants to apply three coats: a base coat, a coat of paint, and a coat of glitter.
2. Each coat takes 20 minutes to apply.
3. Each coat requires 20 minutes of drying time before the next one can be applied.

Prove that the total time taken by Casey to finish decorating her fingernails and toenails is 120 minutes.
-/
theorem casey_nail_decorating_time
  (application_time : ℕ)
  (drying_time : ℕ)
  (num_coats : ℕ)
  (total_time : ℕ)
  (h_app_time : application_time = 20) 
  (h_dry_time : drying_time = 20)
  (h_num_coats : num_coats = 3)
  (h_total_time_eq : total_time = num_coats * (application_time + drying_time)) :
  total_time = 120 :=
sorry

end NUMINAMATH_GPT_casey_nail_decorating_time_l297_29750


namespace NUMINAMATH_GPT_perimeter_of_square_l297_29742

theorem perimeter_of_square (s : ℝ) (area : s^2 = 468) : 4 * s = 24 * Real.sqrt 13 := 
by
  sorry

end NUMINAMATH_GPT_perimeter_of_square_l297_29742


namespace NUMINAMATH_GPT_outdoor_tables_count_l297_29709

variable (numIndoorTables : ℕ) (chairsPerIndoorTable : ℕ) (totalChairs : ℕ)
variable (chairsPerOutdoorTable : ℕ)

theorem outdoor_tables_count 
  (h1 : numIndoorTables = 8) 
  (h2 : chairsPerIndoorTable = 3) 
  (h3 : totalChairs = 60) 
  (h4 : chairsPerOutdoorTable = 3) :
  ∃ (numOutdoorTables : ℕ), numOutdoorTables = 12 := by
  admit

end NUMINAMATH_GPT_outdoor_tables_count_l297_29709
