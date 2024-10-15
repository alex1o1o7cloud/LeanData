import Mathlib

namespace NUMINAMATH_GPT_postage_stamp_problem_l925_92504

theorem postage_stamp_problem
  (x y z : ℕ) (h1: y = 10 * x) (h2: x + 2 * y + 5 * z = 100) :
  x = 5 ∧ y = 50 ∧ z = 0 :=
by
  sorry

end NUMINAMATH_GPT_postage_stamp_problem_l925_92504


namespace NUMINAMATH_GPT_yoongi_age_l925_92576

theorem yoongi_age (H Yoongi : ℕ) : H = Yoongi + 2 ∧ H + Yoongi = 18 → Yoongi = 8 :=
by
  sorry

end NUMINAMATH_GPT_yoongi_age_l925_92576


namespace NUMINAMATH_GPT_problem_l925_92598

noncomputable def F (x : ℝ) : ℝ :=
  (1 + x^2 - x^3) / (2 * x * (1 - x))

theorem problem (x : ℝ) (hx0 : x ≠ 0) (hx1 : x ≠ 1) :
  F x + F ((x - 1) / x) = 1 + x :=
by
  sorry

end NUMINAMATH_GPT_problem_l925_92598


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l925_92551

def P (x : ℝ) : Prop := x^3 + 2 * x ≥ 0

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 0 ≤ x → P x) ↔ (∃ x : ℝ, 0 ≤ x ∧ ¬ P x) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l925_92551


namespace NUMINAMATH_GPT_no_integer_solutions_l925_92511

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), (x ≠ 1 ∧ (x^7 - 1) / (x - 1) = (y^5 - 1)) :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l925_92511


namespace NUMINAMATH_GPT_selected_student_in_eighteenth_group_l925_92524

def systematic_sampling (first_number common_difference nth_term : ℕ) : ℕ :=
  first_number + (nth_term - 1) * common_difference

theorem selected_student_in_eighteenth_group :
  systematic_sampling 22 50 18 = 872 :=
by
  sorry

end NUMINAMATH_GPT_selected_student_in_eighteenth_group_l925_92524


namespace NUMINAMATH_GPT_log_inequality_solution_l925_92509

variable {a x : ℝ}

theorem log_inequality_solution (ha_pos : a > 0) (ha_ne_one : a ≠ 1) :
  (1 + Real.log (a ^ x - 1) / Real.log 2 ≤ Real.log (4 - a ^ x) / Real.log 2) →
  ((1 < a ∧ x ≤ Real.log (7 / 4) / Real.log a) ∨ (0 < a ∧ a < 1 ∧ x ≥ Real.log (7 / 4) / Real.log a)) :=
sorry

end NUMINAMATH_GPT_log_inequality_solution_l925_92509


namespace NUMINAMATH_GPT_max_value_expression_l925_92536

theorem max_value_expression (x : ℝ) : 
  (∃ y : ℝ, y = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16) ∧ 
                ∀ z : ℝ, 
                (∃ x : ℝ, z = x^4 / (x^8 + 2 * x^6 + 4 * x^4 + 4 * x^2 + 16)) → 
                y ≥ z) → 
  ∃ y : ℝ, y = 1 / 16 := 
sorry

end NUMINAMATH_GPT_max_value_expression_l925_92536


namespace NUMINAMATH_GPT_tan_of_angle_l925_92592

noncomputable def tan_val (α : ℝ) : ℝ := Real.tan α

theorem tan_of_angle (α : ℝ) (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) (h2 : Real.cos (2 * α) = -3 / 5) :
  tan_val α = -2 := by
  sorry

end NUMINAMATH_GPT_tan_of_angle_l925_92592


namespace NUMINAMATH_GPT_Mrs_Martin_pays_32_l925_92553

def kiddie_scoop_cost : ℕ := 3
def regular_scoop_cost : ℕ := 4
def double_scoop_cost : ℕ := 6

def num_regular_scoops : ℕ := 2
def num_kiddie_scoops : ℕ := 2
def num_double_scoops : ℕ := 3

def total_cost : ℕ := 
  (num_regular_scoops * regular_scoop_cost) + 
  (num_kiddie_scoops * kiddie_scoop_cost) + 
  (num_double_scoops * double_scoop_cost)

theorem Mrs_Martin_pays_32 :
  total_cost = 32 :=
by
  sorry

end NUMINAMATH_GPT_Mrs_Martin_pays_32_l925_92553


namespace NUMINAMATH_GPT_possible_second_game_scores_count_l925_92527

theorem possible_second_game_scores_count :
  ∃ (A1 A3 B2 : ℕ),
  (A1 + A3 = 22) ∧ (B2 = 11) ∧ (A1 < 11) ∧ (A3 < 11) ∧ ((B2 - A2 = 2) ∨ (B2 >= A2 + 2)) ∧ (A1 + B1 + A2 + B2 + A3 + B3 = 62) :=
  sorry

end NUMINAMATH_GPT_possible_second_game_scores_count_l925_92527


namespace NUMINAMATH_GPT_purely_imaginary_complex_is_two_l925_92541

theorem purely_imaginary_complex_is_two
  (a : ℝ)
  (h_imag : (a^2 - 3 * a + 2) + (a - 1) * I = (a - 1) * I) :
  a = 2 := by
  sorry

end NUMINAMATH_GPT_purely_imaginary_complex_is_two_l925_92541


namespace NUMINAMATH_GPT_cone_surface_area_l925_92520

theorem cone_surface_area {h : ℝ} {A_base : ℝ} (h_eq : h = 4) (A_base_eq : A_base = 9 * Real.pi) :
  let r := Real.sqrt (A_base / Real.pi)
  let l := Real.sqrt (r^2 + h^2)
  let lateral_area := Real.pi * r * l
  let total_surface_area := lateral_area + A_base
  total_surface_area = 24 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_surface_area_l925_92520


namespace NUMINAMATH_GPT_equation_of_line_AB_l925_92560

-- Definition of the given circle
def circle1 : Type := { p : ℝ × ℝ // p.1^2 + (p.2 - 2)^2 = 4 }

-- Definition of the center and point on the second circle
def center : ℝ × ℝ := (0, 2)
def point : ℝ × ℝ := (-2, 6)

-- Definition of the second circle with diameter endpoints
def circle2_eq (x y : ℝ) : Prop := (x + 1)^2 + (y - 4)^2 = 5

-- Statement to be proved
theorem equation_of_line_AB :
  ∃ x y : ℝ, (x^2 + (y - 2)^2 = 4) ∧ ((x + 1)^2 + (y - 4)^2 = 5) ∧ (x - 2*y + 6 = 0) := 
sorry

end NUMINAMATH_GPT_equation_of_line_AB_l925_92560


namespace NUMINAMATH_GPT_B_lap_time_l925_92567

-- Definitions based on given conditions.
def time_to_complete_lap_A := 40
def meeting_interval := 15

-- The theorem states that given the conditions, B takes 24 seconds to complete the track.
theorem B_lap_time (l : ℝ) (t : ℝ) (h1 : t = 24)
                    (h2 : l / time_to_complete_lap_A + l / t = l / meeting_interval):
  t = 24 := by sorry

end NUMINAMATH_GPT_B_lap_time_l925_92567


namespace NUMINAMATH_GPT_cookies_in_one_row_l925_92502

theorem cookies_in_one_row
  (num_trays : ℕ) (rows_per_tray : ℕ) (total_cookies : ℕ)
  (h_trays : num_trays = 4) (h_rows : rows_per_tray = 5) (h_cookies : total_cookies = 120) :
  total_cookies / (num_trays * rows_per_tray) = 6 := by
  sorry

end NUMINAMATH_GPT_cookies_in_one_row_l925_92502


namespace NUMINAMATH_GPT_no_solution_condition_l925_92534

theorem no_solution_condition (n : ℝ) : ¬(∃ x y z : ℝ, n^2 * x + y = 1 ∧ n * y + z = 1 ∧ x + n^2 * z = 1) ↔ n = -1 := 
by {
    sorry
}

end NUMINAMATH_GPT_no_solution_condition_l925_92534


namespace NUMINAMATH_GPT_kola_age_l925_92566

variables (x y : ℕ)

-- Condition 1: Kolya is twice as old as Olya was when Kolya was as old as Olya is now
def condition1 : Prop := x = 2 * (2 * y - x)

-- Condition 2: When Olya is as old as Kolya is now, their combined age will be 36 years.
def condition2 : Prop := (3 * x - y = 36)

theorem kola_age : condition1 x y → condition2 x y → x = 16 :=
by { sorry }

end NUMINAMATH_GPT_kola_age_l925_92566


namespace NUMINAMATH_GPT_minimum_value_of_2m_plus_n_solution_set_for_inequality_l925_92549

namespace MathProof

-- Definitions and conditions
def f (x m n : ℝ) : ℝ := |x + m| + |2 * x - n|

-- Part (I)
theorem minimum_value_of_2m_plus_n
  (m n : ℝ)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_f_nonneg : ∀ x : ℝ, f x m n ≥ 1) :
  2 * m + n ≥ 2 :=
sorry

-- Part (II)
theorem solution_set_for_inequality
  (x : ℝ) :
  (f x 2 3 > 5 ↔ (x < 0 ∨ x > 2)) :=
sorry

end MathProof

end NUMINAMATH_GPT_minimum_value_of_2m_plus_n_solution_set_for_inequality_l925_92549


namespace NUMINAMATH_GPT_problem_i_problem_ii_l925_92537

noncomputable def f (m x : ℝ) := (Real.log x / Real.log m) ^ 2 + 2 * (Real.log x / Real.log m) - 3

theorem problem_i (x : ℝ) : f 2 x < 0 ↔ (1 / 8) < x ∧ x < 2 :=
by sorry

theorem problem_ii (m : ℝ) (H : ∀ x, 2 ≤ x ∧ x ≤ 4 → f m x < 0) : 
  (0 < m ∧ m < 4^(1/3)) ∨ (4 < m) :=
by sorry

end NUMINAMATH_GPT_problem_i_problem_ii_l925_92537


namespace NUMINAMATH_GPT_maximum_students_l925_92562

-- Definitions for conditions
def students (n : ℕ) := Fin n → Prop

-- Condition: Among any six students, there are two who are not friends
def not_friend_in_six (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 6 → ∃ (a b : Fin n), a ∈ s ∧ b ∈ s ∧ ¬ friend a b

-- Condition: For any pair of students not friends, there is a student who is friends with both
def friend_of_two_not_friends (n : ℕ) (friend : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b : Fin n), ¬ friend a b → ∃ (c : Fin n), c ≠ a ∧ c ≠ b ∧ friend c a ∧ friend c b

-- Theorem stating the main result
theorem maximum_students (n : ℕ) (friend : Fin n → Fin n → Prop) :
  not_friend_in_six n friend ∧ friend_of_two_not_friends n friend → n ≤ 25 := 
sorry

end NUMINAMATH_GPT_maximum_students_l925_92562


namespace NUMINAMATH_GPT_martin_total_waste_is_10_l925_92503

def martinWastesTrafficTime : Nat := 2
def martinWastesFreewayTime : Nat := 4 * martinWastesTrafficTime
def totalTimeWasted : Nat := martinWastesTrafficTime + martinWastesFreewayTime

theorem martin_total_waste_is_10 : totalTimeWasted = 10 := 
by 
  sorry

end NUMINAMATH_GPT_martin_total_waste_is_10_l925_92503


namespace NUMINAMATH_GPT_largest_side_of_rectangle_l925_92557

theorem largest_side_of_rectangle :
  ∃ (l w : ℝ), (2 * l + 2 * w = 240) ∧ (l * w = 12 * 240) ∧ (l = 86.835 ∨ w = 86.835) :=
by
  -- Actual proof would be here
  sorry

end NUMINAMATH_GPT_largest_side_of_rectangle_l925_92557


namespace NUMINAMATH_GPT_solving_linear_equations_problems_l925_92530

def num_total_math_problems : ℕ := 140
def percent_algebra_problems : ℝ := 0.40
def fraction_solving_linear_equations : ℝ := 0.50

theorem solving_linear_equations_problems :
  let num_algebra_problems := percent_algebra_problems * num_total_math_problems
  let num_solving_linear_equations := fraction_solving_linear_equations * num_algebra_problems
  num_solving_linear_equations = 28 :=
by
  sorry

end NUMINAMATH_GPT_solving_linear_equations_problems_l925_92530


namespace NUMINAMATH_GPT_quadratic_function_properties_l925_92586

-- We define the primary conditions
def axis_of_symmetry (f : ℝ → ℝ) (x_sym : ℝ) : Prop := 
  ∀ x, f x = f (2 * x_sym - x)

def minimum_value (f : ℝ → ℝ) (y_min : ℝ) (x_min : ℝ) : Prop := 
  ∀ x, f x_min ≤ f x

def passes_through (f : ℝ → ℝ) (pt : ℝ × ℝ) : Prop := 
  f pt.1 = pt.2

-- We need to prove that a quadratic function exists with the given properties and find intersections
theorem quadratic_function_properties :
  ∃ f : ℝ → ℝ,
    axis_of_symmetry f (-1) ∧
    minimum_value f (-4) (-1) ∧
    passes_through f (-2, 5) ∧
    (∀ y : ℝ, f 0 = y → y = 5) ∧
    (∀ x : ℝ, f x = 0 → (x = -5/3 ∨ x = -1/3)) :=
sorry

end NUMINAMATH_GPT_quadratic_function_properties_l925_92586


namespace NUMINAMATH_GPT_interest_equality_l925_92518

-- Definitions based on the conditions
def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ := P * r * t

-- Constants for the problem
def P1 : ℝ := 200 -- 200 Rs is the principal of the first case
def r1 : ℝ := 0.1 -- 10% converted to a decimal
def t1 : ℝ := 12 -- 12 years

def P2 : ℝ := 1000 -- Correct answer for the other amount
def r2 : ℝ := 0.12 -- 12% converted to a decimal
def t2 : ℝ := 2 -- 2 years

-- Theorem stating that the interest generated is the same
theorem interest_equality : 
  simple_interest P1 r1 t1 = simple_interest P2 r2 t2 :=
by 
  -- Skip the proof since it is not required
  sorry

end NUMINAMATH_GPT_interest_equality_l925_92518


namespace NUMINAMATH_GPT_max_a_avoiding_lattice_points_l925_92596

def is_lattice_point (x y : ℤ) : Prop :=
  true  -- Placeholder for (x, y) being in lattice points.

def passes_through_lattice_point (m : ℚ) (x : ℤ) : Prop :=
  is_lattice_point x (⌊m * x + 2⌋)

theorem max_a_avoiding_lattice_points :
  ∀ {a : ℚ}, (∀ x : ℤ, (0 < x ∧ x ≤ 100) → ¬passes_through_lattice_point ((1 : ℚ) / 2) x ∧ ¬passes_through_lattice_point (a - 1) x) →
  a = 50 / 99 :=
by
  sorry

end NUMINAMATH_GPT_max_a_avoiding_lattice_points_l925_92596


namespace NUMINAMATH_GPT_square_area_with_circles_l925_92558

theorem square_area_with_circles (r : ℝ) (h : r = 8) : (2 * (2 * r))^2 = 1024 := 
by 
  sorry

end NUMINAMATH_GPT_square_area_with_circles_l925_92558


namespace NUMINAMATH_GPT_problem1_l925_92547

theorem problem1 (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) : 
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end NUMINAMATH_GPT_problem1_l925_92547


namespace NUMINAMATH_GPT_baker_sales_difference_l925_92593

/-!
  Prove that the difference in dollars between the baker's daily average sales and total sales for today is 48 dollars.
-/

theorem baker_sales_difference :
  let price_pastry := 2
  let price_bread := 4
  let avg_pastries := 20
  let avg_bread := 10
  let today_pastries := 14
  let today_bread := 25
  let daily_avg_sales := avg_pastries * price_pastry + avg_bread * price_bread
  let today_sales := today_pastries * price_pastry + today_bread * price_bread
  daily_avg_sales - today_sales = 48 :=
sorry

end NUMINAMATH_GPT_baker_sales_difference_l925_92593


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l925_92574

-- Part (a)
theorem part_a : ∃ a b, a * b = 80 ∧ (a = 8 ∨ a = 4) ∧ (b = 10 ∨ b = 5) :=
by sorry

-- Part (b)
theorem part_b : ∃ a b c, (a * b) / c = 50 ∧ (a = 10 ∨ a = 5) ∧ (b = 10 ∨ b = 5) ∧ (c = 2 ∨ c = 1) :=
by sorry

-- Part (c)
theorem part_c : ∃ n, n = 4 ∧ ∀ a b c, (a + b) / c = 23 :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l925_92574


namespace NUMINAMATH_GPT_contrapositive_of_lt_l925_92546

theorem contrapositive_of_lt (a b c : ℝ) :
  (a < b → a + c < b + c) → (a + c ≥ b + c → a ≥ b) :=
by
  intro h₀ h₁
  sorry

end NUMINAMATH_GPT_contrapositive_of_lt_l925_92546


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l925_92587

theorem repeating_decimal_fraction :
  (5 + 341 / 999) = (5336 / 999) :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l925_92587


namespace NUMINAMATH_GPT_average_xy_l925_92565

theorem average_xy (x y : ℝ) 
  (h : (4 + 6 + 9 + x + y) / 5 = 20) : (x + y) / 2 = 40.5 :=
sorry

end NUMINAMATH_GPT_average_xy_l925_92565


namespace NUMINAMATH_GPT_find_f3_l925_92571

theorem find_f3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, x * f y = y * f x) (h2 : f 15 = 20) : f 3 = 4 := 
  sorry

end NUMINAMATH_GPT_find_f3_l925_92571


namespace NUMINAMATH_GPT_vector_sum_to_zero_l925_92510

variable {V : Type}
variable [AddCommGroup V] [Module ℝ V] {A B C : V}

theorem vector_sum_to_zero (AB BC CA : V) (hAB : AB = B - A) (hBC : BC = C - B) (hCA : CA = A - C) :
  AB + BC + CA = 0 := by
  sorry

end NUMINAMATH_GPT_vector_sum_to_zero_l925_92510


namespace NUMINAMATH_GPT_three_brothers_pizza_slices_l925_92563

theorem three_brothers_pizza_slices :
  let large_pizza_slices := 14
  let small_pizza_slices := 8
  let num_brothers := 3
  let total_slices := small_pizza_slices + 2 * large_pizza_slices
  total_slices / num_brothers = 12 := by
  sorry

end NUMINAMATH_GPT_three_brothers_pizza_slices_l925_92563


namespace NUMINAMATH_GPT_ivan_running_distance_l925_92568

theorem ivan_running_distance (x MondayDistance TuesdayDistance WednesdayDistance ThursdayDistance FridayDistance : ℝ) 
  (h1 : MondayDistance = x)
  (h2 : TuesdayDistance = 2 * x)
  (h3 : WednesdayDistance = x)
  (h4 : ThursdayDistance = (1 / 2) * x)
  (h5 : FridayDistance = x)
  (hShortest : ThursdayDistance = 5) :
  MondayDistance + TuesdayDistance + WednesdayDistance + ThursdayDistance + FridayDistance = 55 :=
by
  sorry

end NUMINAMATH_GPT_ivan_running_distance_l925_92568


namespace NUMINAMATH_GPT_compute_combination_l925_92532

def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem compute_combination : combination 9 5 = 126 := by
  sorry

end NUMINAMATH_GPT_compute_combination_l925_92532


namespace NUMINAMATH_GPT_abc_plus_2_gt_a_plus_b_plus_c_l925_92552

theorem abc_plus_2_gt_a_plus_b_plus_c (a b c : ℝ) (ha : -1 < a) (ha' : a < 1) (hb : -1 < b) (hb' : b < 1) (hc : -1 < c) (hc' : c < 1) :
  a * b * c + 2 > a + b + c :=
sorry

end NUMINAMATH_GPT_abc_plus_2_gt_a_plus_b_plus_c_l925_92552


namespace NUMINAMATH_GPT_total_goats_l925_92539

theorem total_goats (W: ℕ) (H_W: W = 180) (H_P: W + 70 = 250) : W + (W + 70) = 430 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_goats_l925_92539


namespace NUMINAMATH_GPT_rs_division_l925_92506

theorem rs_division (a b c : ℝ) 
  (h1 : a = 1 / 2 * b)
  (h2 : b = 1 / 2 * c)
  (h3 : a + b + c = 700) : 
  c = 400 :=
sorry

end NUMINAMATH_GPT_rs_division_l925_92506


namespace NUMINAMATH_GPT_hotel_r_charge_percentage_l925_92516

-- Let P, R, and G be the charges for a single room at Hotels P, R, and G respectively
variables (P R G : ℝ)

-- Given conditions:
-- 1. The charge for a single room at Hotel P is 55% less than the charge for a single room at Hotel R.
-- 2. The charge for a single room at Hotel P is 10% less than the charge for a single room at Hotel G.
axiom h1 : P = 0.45 * R
axiom h2 : P = 0.90 * G

-- The charge for a single room at Hotel R is what percent greater than the charge for a single room at Hotel G.
theorem hotel_r_charge_percentage : (R - G) / G * 100 = 100 :=
sorry

end NUMINAMATH_GPT_hotel_r_charge_percentage_l925_92516


namespace NUMINAMATH_GPT_rows_needed_correct_l925_92582

variable (pencils rows_needed : Nat)

def total_pencils : Nat := 35
def pencils_per_row : Nat := 5
def rows_expected : Nat := 7

theorem rows_needed_correct : rows_needed = total_pencils / pencils_per_row →
  rows_needed = rows_expected := by
  sorry

end NUMINAMATH_GPT_rows_needed_correct_l925_92582


namespace NUMINAMATH_GPT_donovan_points_needed_l925_92535

-- Definitions based on conditions
def average_points := 26
def games_played := 15
def total_games := 20
def goal_average := 30

-- Assertion
theorem donovan_points_needed :
  let total_points_needed := goal_average * total_games
  let points_already_scored := average_points * games_played
  let remaining_games := total_games - games_played
  let remaining_points_needed := total_points_needed - points_already_scored
  let points_per_game_needed := remaining_points_needed / remaining_games
  points_per_game_needed = 42 :=
  by
    -- Proof skipped
    sorry

end NUMINAMATH_GPT_donovan_points_needed_l925_92535


namespace NUMINAMATH_GPT_michael_choose_classes_l925_92531

-- Michael's scenario setup
def total_classes : ℕ := 10
def compulsory_class : ℕ := 1
def remaining_classes : ℕ := total_classes - compulsory_class
def total_to_choose : ℕ := 4
def additional_to_choose : ℕ := total_to_choose - compulsory_class

-- Correct answer based on the conditions
def correct_answer : ℕ := 84

-- Function to compute the binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove the number of ways Michael can choose his classes
theorem michael_choose_classes : binomial 9 3 = correct_answer := by
  rw [binomial, Nat.factorial]
  sorry

end NUMINAMATH_GPT_michael_choose_classes_l925_92531


namespace NUMINAMATH_GPT_power_function_const_coeff_l925_92545

theorem power_function_const_coeff (m : ℝ) (h1 : m^2 + 2 * m - 2 = 1) (h2 : m ≠ 1) : m = -3 :=
  sorry

end NUMINAMATH_GPT_power_function_const_coeff_l925_92545


namespace NUMINAMATH_GPT_proof_problem_l925_92505

noncomputable def valid_x (x : ℝ) : Prop :=
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 / 3 ∧ x ≤ 2

theorem proof_problem (x : ℝ) (h : valid_x x) :
  (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ≤ 2 :=
sorry

end NUMINAMATH_GPT_proof_problem_l925_92505


namespace NUMINAMATH_GPT_graph_is_two_lines_l925_92554

theorem graph_is_two_lines : ∀ (x y : ℝ), (x ^ 2 - 25 * y ^ 2 - 20 * x + 100 = 0) ↔ (x = 10 + 5 * y ∨ x = 10 - 5 * y) := 
by 
  intro x y
  sorry

end NUMINAMATH_GPT_graph_is_two_lines_l925_92554


namespace NUMINAMATH_GPT_average_score_is_correct_l925_92575

-- Define the given conditions
def numbers_of_students : List ℕ := [12, 28, 40, 35, 20, 10, 5]
def scores : List ℕ := [95, 85, 75, 65, 55, 45, 35]

-- Function to calculate the total score
def total_score (scores numbers : List ℕ) : ℕ :=
  List.sum (List.zipWith (λ a b => a * b) scores numbers)

-- Calculate the average percent score
def average_percent_score (total number_of_students : ℕ) : ℕ :=
  total / number_of_students

-- Prove that the average percentage score is 70
theorem average_score_is_correct :
  average_percent_score (total_score scores numbers_of_students) 150 = 70 :=
by
  sorry

end NUMINAMATH_GPT_average_score_is_correct_l925_92575


namespace NUMINAMATH_GPT_gallons_left_l925_92544

theorem gallons_left (initial_gallons : ℚ) (gallons_given : ℚ) (gallons_left : ℚ) : 
  initial_gallons = 4 ∧ gallons_given = 16/3 → gallons_left = -4/3 :=
by
  sorry

end NUMINAMATH_GPT_gallons_left_l925_92544


namespace NUMINAMATH_GPT_middle_part_l925_92512

theorem middle_part (x : ℝ) (h : 2 * x + (2 / 3) * x + (2 / 9) * x = 120) : 
  (2 / 3) * x = 27.6 :=
by
  -- Assuming the given conditions
  sorry

end NUMINAMATH_GPT_middle_part_l925_92512


namespace NUMINAMATH_GPT_meaningful_range_l925_92543

noncomputable def isMeaningful (x : ℝ) : Prop :=
  (x + 1 ≥ 0) ∧ (x - 2 ≠ 0)

theorem meaningful_range (x : ℝ) : isMeaningful x ↔ (x ≥ -1) ∧ (x ≠ 2) :=
by
  sorry

end NUMINAMATH_GPT_meaningful_range_l925_92543


namespace NUMINAMATH_GPT_prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l925_92579

noncomputable def prob_win_single_game : ℚ := 7 / 10
noncomputable def prob_lose_single_game : ℚ := 3 / 10

theorem prob_win_all_6_games : (prob_win_single_game ^ 6) = 117649 / 1000000 :=
by
  sorry

theorem prob_win_exactly_5_out_of_6_games : (6 * (prob_win_single_game ^ 5) * prob_lose_single_game) = 302526 / 1000000 :=
by
  sorry

end NUMINAMATH_GPT_prob_win_all_6_games_prob_win_exactly_5_out_of_6_games_l925_92579


namespace NUMINAMATH_GPT_part1_l925_92523

theorem part1 (a : ℝ) : 
  (∀ x ∈ Set.Ici (1/2 : ℝ), 2 * x + a / (x + 1) ≥ 0) → a ≥ -3 / 2 :=
sorry

end NUMINAMATH_GPT_part1_l925_92523


namespace NUMINAMATH_GPT_greatest_servings_l925_92569

def servings (ingredient_amount recipe_amount: ℚ) (recipe_servings: ℕ) : ℚ :=
  (ingredient_amount / recipe_amount) * recipe_servings

theorem greatest_servings (chocolate_new_recipe sugar_new_recipe water_new_recipe milk_new_recipe : ℚ)
                         (servings_new_recipe : ℕ)
                         (chocolate_jordan sugar_jordan milk_jordan : ℚ)
                         (lots_of_water : Prop) :
  chocolate_new_recipe = 3 ∧ sugar_new_recipe = 1/3 ∧ water_new_recipe = 1.5 ∧ milk_new_recipe = 5 ∧
  servings_new_recipe = 6 ∧ chocolate_jordan = 8 ∧ sugar_jordan = 3 ∧ milk_jordan = 12 ∧ lots_of_water →
  max (servings chocolate_jordan chocolate_new_recipe servings_new_recipe)
      (max (servings sugar_jordan sugar_new_recipe servings_new_recipe)
           (servings milk_jordan milk_new_recipe servings_new_recipe)) = 16 :=
by
  sorry

end NUMINAMATH_GPT_greatest_servings_l925_92569


namespace NUMINAMATH_GPT_piles_3_stones_impossible_l925_92585

theorem piles_3_stones_impossible :
  ∀ n : ℕ, ∀ piles : ℕ → ℕ,
  (piles 0 = 1001) →
  (∀ k : ℕ, k > 0 → ∃ i j : ℕ, piles (k-1) > 1 → piles k = i + j ∧ i > 0 ∧ j > 0) →
  ¬ (∀ m : ℕ, piles m ≠ 3) :=
by
  sorry

end NUMINAMATH_GPT_piles_3_stones_impossible_l925_92585


namespace NUMINAMATH_GPT_length_of_second_train_is_229_95_l925_92533

noncomputable def length_of_second_train (length_first_train : ℝ) 
                                          (speed_first_train : ℝ) 
                                          (speed_second_train : ℝ) 
                                          (time_to_cross : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train * (1000 / 3600)
  let speed_second_train_mps := speed_second_train * (1000 / 3600)
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance_covered := relative_speed * time_to_cross
  total_distance_covered - length_first_train

theorem length_of_second_train_is_229_95 :
  length_of_second_train 270 120 80 9 = 229.95 :=
by
  sorry

end NUMINAMATH_GPT_length_of_second_train_is_229_95_l925_92533


namespace NUMINAMATH_GPT_sin_585_eq_neg_sqrt_two_div_two_l925_92517

theorem sin_585_eq_neg_sqrt_two_div_two : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_585_eq_neg_sqrt_two_div_two_l925_92517


namespace NUMINAMATH_GPT_find_num_female_workers_l925_92522

-- Defining the given constants and equations
def num_male_workers : Nat := 20
def num_child_workers : Nat := 5
def wage_male_worker : Nat := 35
def wage_female_worker : Nat := 20
def wage_child_worker : Nat := 8
def avg_wage_paid : Nat := 26

-- Defining the total number of workers and total daily wage
def total_workers (num_female_workers : Nat) : Nat := 
  num_male_workers + num_female_workers + num_child_workers

def total_wage (num_female_workers : Nat) : Nat :=
  (num_male_workers * wage_male_worker) + (num_female_workers * wage_female_worker) + (num_child_workers * wage_child_worker)

-- Proving the number of female workers given the average wage
theorem find_num_female_workers (F : Nat) 
  (h : avg_wage_paid * total_workers F = total_wage F) : 
  F = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_num_female_workers_l925_92522


namespace NUMINAMATH_GPT_coins_difference_l925_92573

theorem coins_difference (p n d : ℕ) (h1 : p + n + d = 3030)
  (h2 : 1 ≤ p) (h3 : 1 ≤ n) (h4 : 1 ≤ d) (h5 : p ≤ 3029) (h6 : n ≤ 3029) (h7 : d ≤ 3029) :
  (max (p + 5 * n + 10 * d) (max (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) - 
  (min (p + 5 * n + 10 * d) (min (p + 5 * n + 10 * (3030 - p - n)) (3030 - n - d + 5 * d + 10 * p))) = 27243 := 
sorry

end NUMINAMATH_GPT_coins_difference_l925_92573


namespace NUMINAMATH_GPT_unoccupied_garden_area_is_correct_l925_92508

noncomputable def area_unoccupied_by_pond_trees_bench (π : ℝ) : ℝ :=
  let garden_area := 144
  let pond_area_rectangle := 6
  let pond_area_semi_circle := 2 * π
  let trees_area := 3
  let bench_area := 3
  garden_area - (pond_area_rectangle + pond_area_semi_circle + trees_area + bench_area)

theorem unoccupied_garden_area_is_correct : 
  area_unoccupied_by_pond_trees_bench Real.pi = 132 - 2 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_unoccupied_garden_area_is_correct_l925_92508


namespace NUMINAMATH_GPT_distance_from_A_to_y_axis_l925_92580

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the distance function from a point to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- State the theorem
theorem distance_from_A_to_y_axis :
  distance_to_y_axis point_A = 3 :=
  by
    -- This part will contain the proof, but we omit it with 'sorry' for now.
    sorry

end NUMINAMATH_GPT_distance_from_A_to_y_axis_l925_92580


namespace NUMINAMATH_GPT_pi_irrational_l925_92572

theorem pi_irrational :
  ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (π = a / b) :=
by
  sorry

end NUMINAMATH_GPT_pi_irrational_l925_92572


namespace NUMINAMATH_GPT_chord_ratio_l925_92559

theorem chord_ratio (EQ GQ HQ FQ : ℝ) (h1 : EQ = 5) (h2 : GQ = 12) (h3 : HQ = 3) (h4 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 12 / 5 := by
  sorry

end NUMINAMATH_GPT_chord_ratio_l925_92559


namespace NUMINAMATH_GPT_solve_for_A_l925_92561

def spadesuit (A B : ℝ) : ℝ := 4*A + 3*B + 6

theorem solve_for_A (A : ℝ) : spadesuit A 5 = 79 → A = 14.5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_solve_for_A_l925_92561


namespace NUMINAMATH_GPT_price_of_whole_pizza_l925_92584

theorem price_of_whole_pizza
    (price_per_slice : ℕ)
    (num_slices_sold : ℕ)
    (num_whole_pizzas_sold : ℕ)
    (total_revenue : ℕ) 
    (H : price_per_slice * num_slices_sold + num_whole_pizzas_sold * P = total_revenue) : 
    P = 15 :=
by
  let price_per_slice := 3
  let num_slices_sold := 24
  let num_whole_pizzas_sold := 3
  let total_revenue := 117
  sorry

end NUMINAMATH_GPT_price_of_whole_pizza_l925_92584


namespace NUMINAMATH_GPT_harkamal_total_amount_l925_92578

-- Conditions
def cost_grapes : ℝ := 8 * 80
def cost_mangoes : ℝ := 9 * 55
def cost_apples_before_discount : ℝ := 6 * 120
def cost_oranges : ℝ := 4 * 75
def discount_apples : ℝ := 0.10 * cost_apples_before_discount
def cost_apples_after_discount : ℝ := cost_apples_before_discount - discount_apples

def total_cost_before_tax : ℝ :=
  cost_grapes + cost_mangoes + cost_apples_after_discount + cost_oranges

def sales_tax : ℝ := 0.05 * total_cost_before_tax

def total_amount_paid : ℝ := total_cost_before_tax + sales_tax

-- Question translated into a Lean statement
theorem harkamal_total_amount:
  total_amount_paid = 2187.15 := 
sorry

end NUMINAMATH_GPT_harkamal_total_amount_l925_92578


namespace NUMINAMATH_GPT_perfect_square_quadratic_l925_92501

theorem perfect_square_quadratic (a : ℝ) :
  ∃ (b : ℝ), (x : ℝ) → (x^2 - ax + 16) = (x + b)^2 ∨ (x^2 - ax + 16) = (x - b)^2 → a = 8 ∨ a = -8 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_quadratic_l925_92501


namespace NUMINAMATH_GPT_part1_part2_l925_92570

variables (α β : Real)

theorem part1 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos α * Real.cos β = 7 / 12 := 
sorry

theorem part2 (h1 : Real.cos (α + β) = 1 / 3) (h2 : Real.sin α * Real.sin β = 1 / 4) :
  Real.cos (2 * α - 2 * β) = 7 / 18 := 
sorry

end NUMINAMATH_GPT_part1_part2_l925_92570


namespace NUMINAMATH_GPT_kindergarten_students_percentage_is_correct_l925_92529

-- Definitions based on conditions
def total_students_annville : ℕ := 150
def total_students_cleona : ℕ := 250
def percent_kindergarten_annville : ℕ := 14
def percent_kindergarten_cleona : ℕ := 10

-- Calculation of number of kindergarten students
def kindergarten_students_annville : ℕ := total_students_annville * percent_kindergarten_annville / 100
def kindergarten_students_cleona : ℕ := total_students_cleona * percent_kindergarten_cleona / 100
def total_kindergarten_students : ℕ := kindergarten_students_annville + kindergarten_students_cleona
def total_students : ℕ := total_students_annville + total_students_cleona
def kindergarten_percentage : ℚ := (total_kindergarten_students * 100) / total_students

-- The theorem to be proved using the conditions
theorem kindergarten_students_percentage_is_correct : kindergarten_percentage = 11.5 := by
  sorry

end NUMINAMATH_GPT_kindergarten_students_percentage_is_correct_l925_92529


namespace NUMINAMATH_GPT_yellow_marbles_at_least_zero_l925_92555

noncomputable def total_marbles := 30
def blue_marbles (n : ℕ) := n / 3
def red_marbles (n : ℕ) := n / 3
def green_marbles := 10
def yellow_marbles (n : ℕ) := n - ((2 * n) / 3 + 10)

-- Conditions
axiom h1 : total_marbles % 3 = 0
axiom h2 : total_marbles = 30

-- Prove the smallest number of yellow marbles is 0
theorem yellow_marbles_at_least_zero : yellow_marbles total_marbles = 0 := by
  sorry

end NUMINAMATH_GPT_yellow_marbles_at_least_zero_l925_92555


namespace NUMINAMATH_GPT_linda_five_dollar_bills_l925_92528

theorem linda_five_dollar_bills (x y : ℕ) (h1 : x + y = 12) (h2 : 5 * x + 10 * y = 80) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_linda_five_dollar_bills_l925_92528


namespace NUMINAMATH_GPT_mutually_exclusive_probability_zero_l925_92548

theorem mutually_exclusive_probability_zero {A B : Prop} (p1 p2 : ℝ) 
  (hA : 0 ≤ p1 ∧ p1 ≤ 1) 
  (hB : 0 ≤ p2 ∧ p2 ≤ 1) 
  (hAB : A ∧ B → False) : 
  (A ∧ B) = False :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_probability_zero_l925_92548


namespace NUMINAMATH_GPT_exists_integer_cube_ends_with_2007_ones_l925_92577

theorem exists_integer_cube_ends_with_2007_ones :
  ∃ x : ℕ, x^3 % 10^2007 = 10^2007 - 1 :=
sorry

end NUMINAMATH_GPT_exists_integer_cube_ends_with_2007_ones_l925_92577


namespace NUMINAMATH_GPT_fresh_fruit_water_content_l925_92564

theorem fresh_fruit_water_content (W N : ℝ) 
  (fresh_weight_dried: W + N = 50) 
  (dried_weight: (0.80 * 5) = N) : 
  ((W / (W + N)) * 100 = 92) :=
by
  sorry

end NUMINAMATH_GPT_fresh_fruit_water_content_l925_92564


namespace NUMINAMATH_GPT_walked_8_miles_if_pace_4_miles_per_hour_l925_92590

-- Define the conditions
def walked_some_miles_in_2_hours (d : ℝ) : Prop :=
  d = 2

def pace_same_4_miles_per_hour (p : ℝ) : Prop :=
  p = 4

-- Define the proof problem
theorem walked_8_miles_if_pace_4_miles_per_hour :
  ∀ (d p : ℝ), walked_some_miles_in_2_hours d → pace_same_4_miles_per_hour p → (p * d = 8) :=
by
  intros d p h1 h2
  rw [h1, h2]
  exact sorry

end NUMINAMATH_GPT_walked_8_miles_if_pace_4_miles_per_hour_l925_92590


namespace NUMINAMATH_GPT_average_of_four_numbers_l925_92507

theorem average_of_four_numbers (a b c d : ℝ) 
  (h1 : b + c + d = 24) (h2 : a + c + d = 36)
  (h3 : a + b + d = 28) (h4 : a + b + c = 32) :
  (a + b + c + d) / 4 = 10 := 
sorry

end NUMINAMATH_GPT_average_of_four_numbers_l925_92507


namespace NUMINAMATH_GPT_theater_rows_l925_92526

theorem theater_rows (R : ℕ) (h1 : R < 30 → ∃ r : ℕ, r < R ∧ r * 2 ≥ 30) (h2 : R ≥ 29 → 26 + 3 ≤ R) : R = 29 :=
by
  sorry

end NUMINAMATH_GPT_theater_rows_l925_92526


namespace NUMINAMATH_GPT_gwen_money_received_from_dad_l925_92599

variables (D : ℕ)

-- Conditions
def mom_received := 8
def mom_more_than_dad := 3

-- Question and required proof
theorem gwen_money_received_from_dad : 
  (mom_received = D + mom_more_than_dad) -> D = 5 := 
by
  sorry

end NUMINAMATH_GPT_gwen_money_received_from_dad_l925_92599


namespace NUMINAMATH_GPT_solve_for_y_l925_92589

noncomputable def x : ℝ := 20
noncomputable def y : ℝ := 40

theorem solve_for_y 
  (h₁ : 1.5 * x = 0.75 * y) 
  (h₂ : x = 20) : 
  y = 40 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l925_92589


namespace NUMINAMATH_GPT_minimum_value_l925_92525

def f (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 1487

theorem minimum_value : ∃ x : ℝ, f x = 1484 := 
sorry

end NUMINAMATH_GPT_minimum_value_l925_92525


namespace NUMINAMATH_GPT_customer_saves_7_906304_percent_l925_92588

variable {P : ℝ} -- Define the base retail price as a variable

-- Define the percentage reductions and additions
def reduced_price (P : ℝ) : ℝ := 0.88 * P
def further_discount_price (P : ℝ) : ℝ := reduced_price P * 0.95
def price_with_dealers_fee (P : ℝ) : ℝ := further_discount_price P * 1.02
def final_price (P : ℝ) : ℝ := price_with_dealers_fee P * 1.08

-- Define the final price factor
def final_price_factor : ℝ := 0.88 * 0.95 * 1.02 * 1.08

noncomputable def total_savings (P : ℝ) : ℝ :=
  P - (final_price_factor * P)

theorem customer_saves_7_906304_percent (P : ℝ) :
  total_savings P = P * 0.07906304 := by
  sorry -- Proof to be added

end NUMINAMATH_GPT_customer_saves_7_906304_percent_l925_92588


namespace NUMINAMATH_GPT_sin_thirty_deg_l925_92538

theorem sin_thirty_deg : Real.sin (π / 6) = 1 / 2 := 
by 
-- Assume that the unit circle point corresponding to 30° is (√3/2, 1/2)
-- Sine of an angle in the unit circle is the y-coordinate of the corresponding point
sorry

end NUMINAMATH_GPT_sin_thirty_deg_l925_92538


namespace NUMINAMATH_GPT_MrsHilt_money_left_l925_92519

theorem MrsHilt_money_left (initial_amount pencil_cost remaining_amount : ℕ) 
  (h_initial : initial_amount = 15) 
  (h_cost : pencil_cost = 11) 
  (h_remaining : remaining_amount = 4) : 
  initial_amount - pencil_cost = remaining_amount := 
by 
  sorry

end NUMINAMATH_GPT_MrsHilt_money_left_l925_92519


namespace NUMINAMATH_GPT_orange_face_probability_correct_l925_92513

-- Define the number of faces
def total_faces : ℕ := 12
def green_faces : ℕ := 5
def orange_faces : ℕ := 4
def purple_faces : ℕ := 3

-- Define the probability of rolling an orange face
def probability_of_orange_face : ℚ := orange_faces / total_faces

-- Statement of the theorem
theorem orange_face_probability_correct :
  probability_of_orange_face = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_orange_face_probability_correct_l925_92513


namespace NUMINAMATH_GPT_find_two_digit_number_l925_92556

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_multiple_of_9 (n : ℕ) : Prop := n % 9 = 0

def product_of_digits (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

theorem find_two_digit_number :
  ∃ N : ℕ, is_two_digit N ∧ is_odd N ∧ is_multiple_of_9 N ∧ is_perfect_square (product_of_digits N) ∧ N = 99 :=
sorry

end NUMINAMATH_GPT_find_two_digit_number_l925_92556


namespace NUMINAMATH_GPT_extremum_at_one_and_value_at_two_l925_92583

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_one_and_value_at_two (a b : ℝ) (h_deriv : 3 + 2*a + b = 0) (h_value : 1 + a + b + a^2 = 10) : 
  f 2 a b = 18 := 
by 
  sorry

end NUMINAMATH_GPT_extremum_at_one_and_value_at_two_l925_92583


namespace NUMINAMATH_GPT_possible_values_of_a_l925_92515

theorem possible_values_of_a :
  ∃ a b c : ℤ, 
    (∀ x : ℤ, (x - a) * (x - 5) + 1 = (x + b) * (x + c)) ↔ 
    (a = 3 ∨ a = 7) :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_a_l925_92515


namespace NUMINAMATH_GPT_total_rooms_to_paint_l925_92550

theorem total_rooms_to_paint :
  ∀ (hours_per_room hours_remaining rooms_painted : ℕ),
    hours_per_room = 7 →
    hours_remaining = 63 →
    rooms_painted = 2 →
    rooms_painted + hours_remaining / hours_per_room = 11 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_rooms_to_paint_l925_92550


namespace NUMINAMATH_GPT_how_fast_is_a_l925_92540

variable (a b : ℝ) (k : ℝ)

theorem how_fast_is_a (h1 : a = k * b) (h2 : a + b = 1 / 30) (h3 : a = 1 / 40) : k = 3 := sorry

end NUMINAMATH_GPT_how_fast_is_a_l925_92540


namespace NUMINAMATH_GPT_solve_abs_equation_l925_92581

theorem solve_abs_equation (x : ℝ) : 2 * |x - 5| = 6 ↔ x = 2 ∨ x = 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_equation_l925_92581


namespace NUMINAMATH_GPT_credit_limit_l925_92597

theorem credit_limit (paid_tuesday : ℕ) (paid_thursday : ℕ) (remaining_payment : ℕ) (full_payment : ℕ) 
  (h1 : paid_tuesday = 15) 
  (h2 : paid_thursday = 23) 
  (h3 : remaining_payment = 62) 
  (h4 : full_payment = paid_tuesday + paid_thursday + remaining_payment) : 
  full_payment = 100 := 
by
  sorry

end NUMINAMATH_GPT_credit_limit_l925_92597


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l925_92594

def P (x : ℝ) : Prop := 0 < x ∧ x < 5
def Q (x : ℝ) : Prop := |x - 2| < 3

theorem sufficient_but_not_necessary_condition
  (x : ℝ) : (P x → Q x) ∧ ¬(Q x → P x) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l925_92594


namespace NUMINAMATH_GPT_distance_between_places_l925_92542

theorem distance_between_places
  (d : ℝ) -- let d be the distance between A and B
  (v : ℝ) -- let v be the original speed
  (h1 : v * 4 = d) -- initially, speed * time = distance
  (h2 : (v + 20) * 3 = d) -- after speed increase, speed * new time = distance
  : d = 240 :=
sorry

end NUMINAMATH_GPT_distance_between_places_l925_92542


namespace NUMINAMATH_GPT_top_square_is_9_l925_92591

def initial_grid : List (List ℕ) := 
  [[1, 2, 3],
   [4, 5, 6],
   [7, 8, 9]]

def fold_step_1 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col3 := grid.map (fun row => row.get! 2)
  let col2 := grid.map (fun row => row.get! 1)
  [[col1.get! 0, col3.get! 0, col2.get! 0],
   [col1.get! 1, col3.get! 1, col2.get! 1],
   [col1.get! 2, col3.get! 2, col2.get! 2]]

def fold_step_2 (grid : List (List ℕ)) : List (List ℕ) :=
  let col1 := grid.map (fun row => row.get! 0)
  let col2 := grid.map (fun row => row.get! 1)
  let col3 := grid.map (fun row => row.get! 2)
  [[col2.get! 0, col1.get! 0, col3.get! 0],
   [col2.get! 1, col1.get! 1, col3.get! 1],
   [col2.get! 2, col1.get! 2, col3.get! 2]]

def fold_step_3 (grid : List (List ℕ)) : List (List ℕ) :=
  let row1 := grid.get! 0
  let row2 := grid.get! 1
  let row3 := grid.get! 2
  [row3, row2, row1]

def folded_grid : List (List ℕ) :=
  fold_step_3 (fold_step_2 (fold_step_1 initial_grid))

theorem top_square_is_9 : folded_grid.get! 0 = [9, 7, 8] :=
  sorry

end NUMINAMATH_GPT_top_square_is_9_l925_92591


namespace NUMINAMATH_GPT_range_of_a_l925_92595

noncomputable def p (x : ℝ) : Prop := (3*x - 1)/(x - 2) ≤ 1
noncomputable def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) < 0

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, ¬ q x a) → (¬ ∃ x : ℝ, ¬ p x) → -1/2 ≤ a ∧ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l925_92595


namespace NUMINAMATH_GPT_bob_shucks_240_oysters_in_2_hours_l925_92514

-- Definitions based on conditions provided:
def oysters_per_minute (oysters : ℕ) (minutes : ℕ) : ℕ :=
  oysters / minutes

def minutes_in_hour : ℕ :=
  60

def oysters_in_two_hours (oysters_per_minute : ℕ) (hours : ℕ) : ℕ :=
  oysters_per_minute * (hours * minutes_in_hour)

-- Parameters given in the problem:
def initial_oysters : ℕ := 10
def initial_minutes : ℕ := 5
def hours : ℕ := 2

-- The main theorem we need to prove:
theorem bob_shucks_240_oysters_in_2_hours :
  oysters_in_two_hours (oysters_per_minute initial_oysters initial_minutes) hours = 240 :=
by
  sorry

end NUMINAMATH_GPT_bob_shucks_240_oysters_in_2_hours_l925_92514


namespace NUMINAMATH_GPT_find_ordered_pairs_l925_92500

theorem find_ordered_pairs (a b x : ℕ) (h1 : b > a) (h2 : a + b = 15) (h3 : (a - 2 * x) * (b - 2 * x) = 2 * a * b / 3) :
  (a, b) = (8, 7) :=
by
  sorry

end NUMINAMATH_GPT_find_ordered_pairs_l925_92500


namespace NUMINAMATH_GPT_ratio_a_to_c_l925_92521

variable (a b c : ℕ)

theorem ratio_a_to_c (h1 : a / b = 8 / 3) (h2 : b / c = 1 / 5) : a / c = 8 / 15 := 
sorry

end NUMINAMATH_GPT_ratio_a_to_c_l925_92521
