import Mathlib

namespace complex_magnitude_l1670_167032

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z with the given condition
variable (z : ℂ) (h : z * (1 + i) = 2 * i)

-- Statement of the problem: Prove that |z + 2 * i| = √10
theorem complex_magnitude (z : ℂ) (h : z * (1 + i) = 2 * i) : Complex.abs (z + 2 * i) = Real.sqrt 10 := 
sorry

end complex_magnitude_l1670_167032


namespace total_paint_is_correct_l1670_167075

/-- Given conditions -/
def paint_per_large_canvas := 3
def paint_per_small_canvas := 2
def large_paintings := 3
def small_paintings := 4

/-- Define total paint used using the given conditions -/
noncomputable def total_paint_used : ℕ := 
  (paint_per_large_canvas * large_paintings) + (paint_per_small_canvas * small_paintings)

/-- Theorem statement to show the total paint used equals 17 ounces -/
theorem total_paint_is_correct : total_paint_used = 17 := by
  sorry

end total_paint_is_correct_l1670_167075


namespace exists_n_in_range_multiple_of_11_l1670_167009

def is_multiple_of_11 (n : ℕ) : Prop :=
  (3 * n^5 + 4 * n^4 + 5 * n^3 + 7 * n^2 + 6 * n + 2) % 11 = 0

theorem exists_n_in_range_multiple_of_11 : ∃ n : ℕ, (2 ≤ n ∧ n ≤ 101) ∧ is_multiple_of_11 n :=
sorry

end exists_n_in_range_multiple_of_11_l1670_167009


namespace distinct_nat_numbers_l1670_167048

theorem distinct_nat_numbers 
  (a b c : ℕ) (p q r : ℤ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_sum : a + b + c = 55) 
  (h_ab : a + b = p * p) 
  (h_bc : b + c = q * q) 
  (h_ca : c + a = r * r) : 
  a = 19 ∧ b = 6 ∧ c = 30 :=
sorry

end distinct_nat_numbers_l1670_167048


namespace subtraction_proof_l1670_167093

theorem subtraction_proof :
  2000000000000 - 1111111111111 - 222222222222 = 666666666667 :=
by sorry

end subtraction_proof_l1670_167093


namespace find_f_of_2_l1670_167039

-- Given definitions:
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

def defined_on_neg_inf_to_0 (f : ℝ → ℝ) : Prop := ∀ x, x < 0 → f x = 2 * x^3 + x^2

-- The main theorem to prove:
theorem find_f_of_2 (f : ℝ → ℝ) 
  (h_odd : odd_function f)
  (h_def : defined_on_neg_inf_to_0 f) :
  f 2 = 12 :=
sorry

end find_f_of_2_l1670_167039


namespace quadratic_real_roots_exists_l1670_167088

theorem quadratic_real_roots_exists :
  ∃ (x1 x2 : ℝ), (x1 ≠ x2) ∧ (x1 * x1 - 6 * x1 + 8 = 0) ∧ (x2 * x2 - 6 * x2 + 8 = 0) :=
by
  sorry

end quadratic_real_roots_exists_l1670_167088


namespace toms_dog_is_12_l1670_167068

def toms_cat_age : ℕ := 8
def toms_rabbit_age : ℕ := toms_cat_age / 2
def toms_dog_age : ℕ := toms_rabbit_age * 3

theorem toms_dog_is_12 : toms_dog_age = 12 :=
by
  sorry

end toms_dog_is_12_l1670_167068


namespace christmas_bonus_remainder_l1670_167063

theorem christmas_bonus_remainder (P : ℕ) (h : P % 5 = 2) : (3 * P) % 5 = 1 :=
by
  sorry

end christmas_bonus_remainder_l1670_167063


namespace symmetric_point_A_is_B_l1670_167089

/-
  Define the symmetric point function for reflecting a point across the origin.
  Define the coordinate of point A.
  Assert that the symmetric point of A has coordinates (-2, 6).
-/

structure Point where
  x : ℤ
  y : ℤ

def symmetric_point (p : Point) : Point :=
  Point.mk (-p.x) (-p.y)

def A : Point := ⟨2, -6⟩

def B : Point := ⟨-2, 6⟩

theorem symmetric_point_A_is_B : symmetric_point A = B := by
  sorry

end symmetric_point_A_is_B_l1670_167089


namespace sum_inf_evaluation_eq_9_by_80_l1670_167004

noncomputable def infinite_sum_evaluation : ℝ := ∑' n, (2 * n) / (n^4 + 16)

theorem sum_inf_evaluation_eq_9_by_80 :
  infinite_sum_evaluation = 9 / 80 :=
by
  sorry

end sum_inf_evaluation_eq_9_by_80_l1670_167004


namespace find_A_l1670_167058

theorem find_A (J : ℤ := 15)
  (JAVA_pts : ℤ := 50)
  (AJAX_pts : ℤ := 53)
  (AXLE_pts : ℤ := 40)
  (L : ℤ := 12)
  (JAVA_eq : ∀ A V : ℤ, 2 * A + V + J = JAVA_pts)
  (AJAX_eq : ∀ A X : ℤ, 2 * A + X + J = AJAX_pts)
  (AXLE_eq : ∀ A X E : ℤ, A + X + L + E = AXLE_pts) : A = 21 :=
sorry

end find_A_l1670_167058


namespace problem_statement_l1670_167094

theorem problem_statement :
  (∀ x : ℝ, |x| < 2 → x < 3) ∧
  (∀ x : ℝ, ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ∧
  (-1 < m ∧ m < 0 → ∀ a b : ℝ, a ≠ b → (a * b > 0)) :=
by
  sorry

end problem_statement_l1670_167094


namespace garden_area_l1670_167078

theorem garden_area (w l : ℕ) (h1 : l = 3 * w + 30) (h2 : 2 * (w + l) = 780) : 
  w * l = 27000 := 
by 
  sorry

end garden_area_l1670_167078


namespace factorial_expression_simplification_l1670_167051

theorem factorial_expression_simplification : (3 * (Nat.factorial 5) + 15 * (Nat.factorial 4)) / (Nat.factorial 6) = 1 := by
  sorry

end factorial_expression_simplification_l1670_167051


namespace base_h_equation_l1670_167085

theorem base_h_equation (h : ℕ) : 
  (5 * h^3 + 7 * h^2 + 3 * h + 4) + (6 * h^3 + 4 * h^2 + 2 * h + 1) = 
  1 * h^4 + 4 * h^3 + 1 * h^2 + 5 * h + 5 → 
  h = 10 := 
sorry

end base_h_equation_l1670_167085


namespace find_A_plus_B_plus_C_plus_D_l1670_167015

noncomputable def A : ℤ := -7
noncomputable def B : ℕ := 8
noncomputable def C : ℤ := 21
noncomputable def D : ℕ := 1

def conditions_satisfied : Prop :=
  D > 0 ∧
  ¬∃ p : ℕ, Nat.Prime p ∧ p^2 ∣ B ∧ p ≠ 1 ∧ p ≠ B ∧ p ≥ 2 ∧
  Int.gcd A (Int.gcd C (Int.ofNat D)) = 1

theorem find_A_plus_B_plus_C_plus_D : conditions_satisfied → A + B + C + D = 23 :=
by
  intro h
  sorry

end find_A_plus_B_plus_C_plus_D_l1670_167015


namespace morgan_first_sat_score_l1670_167018

theorem morgan_first_sat_score (x : ℝ) (h : 1.10 * x = 1100) : x = 1000 :=
sorry

end morgan_first_sat_score_l1670_167018


namespace complete_square_solution_l1670_167056

theorem complete_square_solution :
  ∀ (x : ℝ), (x^2 + 8*x + 9 = 0) → ((x + 4)^2 = 7) :=
by
  intro x h_eq
  sorry

end complete_square_solution_l1670_167056


namespace remainder_when_3_pow_2020_div_73_l1670_167001

theorem remainder_when_3_pow_2020_div_73 :
  (3^2020 % 73) = 8 := 
sorry

end remainder_when_3_pow_2020_div_73_l1670_167001


namespace quadrilateral_area_inequality_l1670_167086

theorem quadrilateral_area_inequality 
  (T : ℝ) (a b c d e f : ℝ) (φ : ℝ) 
  (hT : T = (1/2) * e * f * Real.sin φ) 
  (hptolemy : e * f ≤ a * c + b * d) : 
  2 * T ≤ a * c + b * d := 
sorry

end quadrilateral_area_inequality_l1670_167086


namespace find_f_neg_half_l1670_167073

def is_odd_function {α β : Type*} [AddGroup α] [Neg β] (f : α → β) : Prop :=
  ∀ x : α, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 0

theorem find_f_neg_half (f_odd : is_odd_function f) (f_pos : ∀ x > 0, f x = Real.log x / Real.log 2) :
  f (-1/2) = 1 := by
  sorry

end find_f_neg_half_l1670_167073


namespace saree_original_price_l1670_167097

theorem saree_original_price :
  ∃ P : ℝ, (0.95 * 0.88 * P = 334.4) ∧ (P = 400) :=
by
  sorry

end saree_original_price_l1670_167097


namespace set_contains_one_implies_values_l1670_167057

theorem set_contains_one_implies_values (x : ℝ) (A : Set ℝ) (hA : A = {x, x^2}) (h1 : 1 ∈ A) : x = 1 ∨ x = -1 := by
  sorry

end set_contains_one_implies_values_l1670_167057


namespace find_z_to_8_l1670_167029

noncomputable def complex_number_z (z : ℂ) : Prop :=
  z + z⁻¹ = 2 * Complex.cos (Real.pi / 4)

theorem find_z_to_8 (z : ℂ) (h : complex_number_z z) : (z ^ 8 + (z ^ 8)⁻¹ = 2) :=
by
  sorry

end find_z_to_8_l1670_167029


namespace lucas_fib_relation_l1670_167026

noncomputable def α := (1 + Real.sqrt 5) / 2
noncomputable def β := (1 - Real.sqrt 5) / 2
def Fib : ℕ → ℝ
| 0       => 0
| 1       => 1
| (n + 2) => Fib n + Fib (n + 1)

def Lucas : ℕ → ℝ
| 0       => 2
| 1       => 1
| (n + 2) => Lucas n + Lucas (n + 1)

theorem lucas_fib_relation (n : ℕ) (hn : 1 ≤ n) :
  Lucas (2 * n + 1) + (-1)^(n+1) = Fib (2 * n) * Fib (2 * n + 1) := sorry

end lucas_fib_relation_l1670_167026


namespace sum_of_number_and_radical_conjugate_l1670_167034

theorem sum_of_number_and_radical_conjugate : 
  (10 - Real.sqrt 2018) + (10 + Real.sqrt 2018) = 20 := 
by 
  sorry

end sum_of_number_and_radical_conjugate_l1670_167034


namespace find_r_power_4_l1670_167007

variable {r : ℝ}

theorem find_r_power_4 (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := 
sorry

end find_r_power_4_l1670_167007


namespace cost_price_of_bicycle_l1670_167087

variables {CP_A SP_AB SP_BC : ℝ}

theorem cost_price_of_bicycle (h1 : SP_AB = CP_A * 1.2)
                             (h2 : SP_BC = SP_AB * 1.25)
                             (h3 : SP_BC = 225) :
                             CP_A = 150 :=
by sorry

end cost_price_of_bicycle_l1670_167087


namespace cylinder_problem_l1670_167019

theorem cylinder_problem (r h : ℝ) (h1 : π * r^2 * h = 2) (h2 : 2 * π * r * h + 2 * π * r^2 = 12) :
  1 / r + 1 / h = 3 :=
sorry

end cylinder_problem_l1670_167019


namespace frac_plus_a_ge_seven_l1670_167067

theorem frac_plus_a_ge_seven (a : ℝ) (h : a > 3) : 4 / (a - 3) + a ≥ 7 := 
by
  sorry

end frac_plus_a_ge_seven_l1670_167067


namespace largest_n_for_factoring_l1670_167081

theorem largest_n_for_factoring :
  ∃ (n : ℤ), (∀ (A B : ℤ), (3 * A + B = n) → (3 * A * B = 90) → n = 271) :=
by sorry

end largest_n_for_factoring_l1670_167081


namespace smallest_integer_is_17_l1670_167076

theorem smallest_integer_is_17
  (a b c d : ℕ)
  (h1 : b = 33)
  (h2 : d = b + 3)
  (h3 : (a + b + c + d) = 120)
  (h4 : a ≤ b)
  (h5 : c > b)
  : a = 17 :=
sorry

end smallest_integer_is_17_l1670_167076


namespace probability_one_in_first_20_rows_l1670_167008

theorem probability_one_in_first_20_rows :
  let total_elements := 210
  let number_of_ones := 39
  (number_of_ones / total_elements : ℚ) = 13 / 70 :=
by
  sorry

end probability_one_in_first_20_rows_l1670_167008


namespace ticket_cost_is_nine_l1670_167043

theorem ticket_cost_is_nine (bought_tickets : ℕ) (left_tickets : ℕ) (spent_dollars : ℕ) 
  (h1 : bought_tickets = 6) 
  (h2 : left_tickets = 3) 
  (h3 : spent_dollars = 27) : 
  spent_dollars / (bought_tickets - left_tickets) = 9 :=
by
  -- Using the imported library and the given conditions
  sorry

end ticket_cost_is_nine_l1670_167043


namespace find_number_l1670_167024

theorem find_number (x : ℝ) (h : (3/4 : ℝ) * x = 93.33333333333333) : x = 124.44444444444444 := 
by
  -- Proof to be filled in
  sorry

end find_number_l1670_167024


namespace conference_games_scheduled_l1670_167066

theorem conference_games_scheduled
  (divisions : ℕ)
  (teams_per_division : ℕ)
  (intra_games_per_pair : ℕ)
  (inter_games_per_pair : ℕ)
  (h_div : divisions = 3)
  (h_teams : teams_per_division = 4)
  (h_intra : intra_games_per_pair = 3)
  (h_inter : inter_games_per_pair = 2) :
  let intra_division_games := (teams_per_division * (teams_per_division - 1) / 2) * intra_games_per_pair
  let intra_division_total := intra_division_games * divisions
  let inter_division_games := teams_per_division * (teams_per_division * (divisions - 1)) * inter_games_per_pair
  let inter_division_total := inter_division_games * divisions / 2
  let total_games := intra_division_total + inter_division_total
  total_games = 150 :=
by
  sorry

end conference_games_scheduled_l1670_167066


namespace problem_A_eq_7_problem_A_eq_2012_l1670_167035

open Nat

-- Problem statement for A = 7
theorem problem_A_eq_7 (n k : ℕ) :
  (n! + 7 * n = n^k) ↔ ((n, k) = (2, 4) ∨ (n, k) = (3, 3)) :=
sorry

-- Problem statement for A = 2012
theorem problem_A_eq_2012 (n k : ℕ) :
  ¬ (n! + 2012 * n = n^k) :=
sorry

end problem_A_eq_7_problem_A_eq_2012_l1670_167035


namespace men_complete_units_per_day_l1670_167084

noncomputable def UnitsCompletedByMen (total_units : ℕ) (units_by_women : ℕ) : ℕ :=
  total_units - units_by_women

theorem men_complete_units_per_day :
  UnitsCompletedByMen 12 3 = 9 := by
  -- Proof skipped
  sorry

end men_complete_units_per_day_l1670_167084


namespace product_of_last_two_digits_l1670_167021

theorem product_of_last_two_digits (A B : ℕ) (h1 : A + B = 12) (h2 : (10 * A + B) % 8 = 0) : A * B = 32 :=
by
  sorry

end product_of_last_two_digits_l1670_167021


namespace fraction_subtraction_equivalence_l1670_167040

theorem fraction_subtraction_equivalence :
  (8 / 19) - (5 / 57) = 1 / 3 :=
by sorry

end fraction_subtraction_equivalence_l1670_167040


namespace polynomial_simplification_l1670_167045

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 9 * x - 8) + (-x^5 + x^4 - 2 * x^3 + 4 * x^2 - 6 * x + 14) = 
  -x^5 + 3 * x^4 + x^3 - x^2 + 3 * x + 6 :=
by
  sorry

end polynomial_simplification_l1670_167045


namespace initial_red_marbles_l1670_167061

theorem initial_red_marbles (r g : ℕ) (h1 : r * 3 = 7 * g) (h2 : 4 * (r - 14) = g + 30) : r = 24 := 
sorry

end initial_red_marbles_l1670_167061


namespace roy_cat_finishes_food_on_wednesday_l1670_167042

-- Define the conditions
def morning_consumption := (1 : ℚ) / 5
def evening_consumption := (1 : ℚ) / 6
def total_cans := 10

-- Define the daily consumption calculation
def daily_consumption := morning_consumption + evening_consumption

-- Define the day calculation function
def day_cat_finishes_food : String :=
  let total_days := total_cans / daily_consumption
  if total_days ≤ 7 then "certain day within a week"
  else if total_days ≤ 14 then "Wednesday next week"
  else "later"

-- The main theorem to prove
theorem roy_cat_finishes_food_on_wednesday : day_cat_finishes_food = "Wednesday next week" := sorry

end roy_cat_finishes_food_on_wednesday_l1670_167042


namespace rectangle_area_l1670_167028

theorem rectangle_area (a b : ℝ) (h : 2 * a^2 - 11 * a + 5 = 0) (hb : 2 * b^2 - 11 * b + 5 = 0) : a * b = 5 / 2 :=
sorry

end rectangle_area_l1670_167028


namespace motorist_travel_time_l1670_167052

noncomputable def total_time (dist1 dist2 speed1 speed2 : ℝ) : ℝ :=
  (dist1 / speed1) + (dist2 / speed2)

theorem motorist_travel_time (speed1 speed2 : ℝ) (total_dist : ℝ) (half_dist : ℝ) :
  speed1 = 60 → speed2 = 48 → total_dist = 324 → half_dist = total_dist / 2 →
  total_time half_dist half_dist speed1 speed2 = 6.075 :=
by
  intros h1 h2 h3 h4
  simp [total_time, h1, h2, h3, h4]
  sorry

end motorist_travel_time_l1670_167052


namespace octagon_area_sum_l1670_167006

theorem octagon_area_sum :
  let A1 := 2024
  let a := 1012
  let b := 506
  let c := 2
  a + b + c = 1520 := by
    sorry

end octagon_area_sum_l1670_167006


namespace smallest_number_of_coins_l1670_167054

theorem smallest_number_of_coins (d q : ℕ) (h₁ : 10 * d + 25 * q = 265) (h₂ : d > q) :
  d + q = 16 :=
sorry

end smallest_number_of_coins_l1670_167054


namespace number_of_female_students_l1670_167020

variable (n m : ℕ)

theorem number_of_female_students (hn : n ≥ 0) (hm : m ≥ 0) (hmn : m ≤ n) : n - m = n - m :=
by
  sorry

end number_of_female_students_l1670_167020


namespace difference_apples_peaches_pears_l1670_167096

-- Definitions based on the problem conditions
def apples : ℕ := 60
def peaches : ℕ := 3 * apples
def pears : ℕ := apples / 2

-- Statement of the proof problem
theorem difference_apples_peaches_pears : (apples + peaches) - pears = 210 := by
  sorry

end difference_apples_peaches_pears_l1670_167096


namespace original_area_of_circle_l1670_167005

theorem original_area_of_circle
  (A₀ : ℝ) -- original area
  (r₀ r₁ : ℝ) -- original and new radius
  (π : ℝ := 3.14)
  (h_area : A₀ = π * r₀^2)
  (h_area_increase : π * r₁^2 = 9 * A₀)
  (h_circumference_increase : 2 * π * r₁ - 2 * π * r₀ = 50.24) :
  A₀ = 50.24 :=
by
  sorry

end original_area_of_circle_l1670_167005


namespace marigold_ratio_l1670_167031

theorem marigold_ratio :
  ∃ x, 14 + 25 + x = 89 ∧ x / 25 = 2 := by
  sorry

end marigold_ratio_l1670_167031


namespace net_sales_revenue_l1670_167003

-- Definition of the conditions
def regression (x : ℝ) : ℝ := 8.5 * x + 17.5

-- Statement of the theorem
theorem net_sales_revenue (x : ℝ) (h : x = 10) : (regression x - x) = 92.5 :=
by {
  -- No proof required as per instruction; use sorry.
  sorry
}

end net_sales_revenue_l1670_167003


namespace find_f_65_l1670_167077

theorem find_f_65 (f : ℝ → ℝ) (h_eq : ∀ x y : ℝ, f (x * y) = x * f y) (h_f1 : f 1 = 40) : f 65 = 2600 :=
by
  sorry

end find_f_65_l1670_167077


namespace find_common_real_root_l1670_167010

theorem find_common_real_root :
  ∃ (m a : ℝ), (a^2 + m * a + 2 = 0) ∧ (a^2 + 2 * a + m = 0) ∧ m = -3 ∧ a = 1 :=
by
  -- Skipping the proof
  sorry

end find_common_real_root_l1670_167010


namespace necessary_condition_not_sufficient_condition_l1670_167027

noncomputable def zero_point (a : ℝ) : Prop :=
  ∃ x : ℝ, 3^x + a - 1 = 0

noncomputable def decreasing_log (a : ℝ) : Prop :=
  0 < a ∧ a < 1

theorem necessary_condition (a : ℝ) (h : zero_point a) : 0 < a ∧ a < 1 := sorry

theorem not_sufficient_condition (a : ℝ) (h : 0 < a ∧ a < 1) : ¬(zero_point a) := sorry

end necessary_condition_not_sufficient_condition_l1670_167027


namespace problem_min_x_plus_2y_l1670_167053

theorem problem_min_x_plus_2y (x y : ℝ) (h : x^2 + 4 * y^2 - 2 * x + 8 * y + 1 = 0) : 
  x + 2 * y ≥ -2 * Real.sqrt 2 - 1 :=
sorry

end problem_min_x_plus_2y_l1670_167053


namespace Petya_tore_out_sheets_l1670_167036

theorem Petya_tore_out_sheets (n m : ℕ) (h1 : n = 185) (h2 : m = 518)
  (h3 : m.digits = n.digits) : (m - n + 1) / 2 = 167 :=
by
  sorry

end Petya_tore_out_sheets_l1670_167036


namespace find_k_l1670_167065

-- Define the vectors
def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 2)

-- Define the condition for vectors to be parallel
def is_parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

-- Translate the problem condition
def problem_condition (k : ℝ) : Prop :=
  let lhs := (k * a.1 + b.1, k * a.2 + b.2)
  let rhs := (a.1 - 3 * b.1, a.2 - 3 * b.2)
  is_parallel lhs rhs

-- The goal is to find k such that the condition holds
theorem find_k : problem_condition (-1/3) :=
by
  sorry

end find_k_l1670_167065


namespace monthly_salary_l1670_167037

variable (S : ℝ)
variable (Saves : ℝ)
variable (NewSaves : ℝ)

open Real

theorem monthly_salary (h1 : Saves = 0.30 * S)
                       (h2 : NewSaves = Saves - 0.25 * Saves)
                       (h3 : NewSaves = 400) :
    S = 1777.78 := by
    sorry

end monthly_salary_l1670_167037


namespace problem_intersection_l1670_167083

theorem problem_intersection (a b : ℝ) 
    (h1 : b = - 2 / a) 
    (h2 : b = a + 3) 
    : 1 / a - 1 / b = -3 / 2 :=
by
  sorry

end problem_intersection_l1670_167083


namespace find_greatest_consecutive_integer_l1670_167046

theorem find_greatest_consecutive_integer (n : ℤ) 
  (h : n^2 + (n + 1)^2 = 452) : n + 1 = 15 :=
sorry

end find_greatest_consecutive_integer_l1670_167046


namespace pow_sum_nineteen_eq_zero_l1670_167002

variable {a b c : ℝ}

theorem pow_sum_nineteen_eq_zero (h₁ : a + b + c = 0) (h₂ : a^3 + b^3 + c^3 = 0) : a^19 + b^19 + c^19 = 0 :=
sorry

end pow_sum_nineteen_eq_zero_l1670_167002


namespace three_digit_ends_in_5_divisible_by_5_l1670_167023

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def ends_in_5 (n : ℕ) : Prop := n % 10 = 5

def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0

theorem three_digit_ends_in_5_divisible_by_5 (N : ℕ) 
  (h1 : is_three_digit N) 
  (h2 : ends_in_5 N) : is_divisible_by_5 N := 
sorry

end three_digit_ends_in_5_divisible_by_5_l1670_167023


namespace part1_l1670_167091

variables {a b c : ℝ}
theorem part1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a / (b + c) = b / (c + a) - c / (a + b)) : 
    b / (c + a) ≥ (Real.sqrt 17 - 1) / 4 :=
sorry

end part1_l1670_167091


namespace original_proposition_contrapositive_converse_inverse_negation_false_l1670_167038

variable {a b c : ℝ}

-- Original Proposition
theorem original_proposition (h : a < b) : a + c < b + c :=
sorry

-- Contrapositive
theorem contrapositive (h : a + c >= b + c) : a >= b :=
sorry

-- Converse
theorem converse (h : a + c < b + c) : a < b :=
sorry

-- Inverse
theorem inverse (h : a >= b) : a + c >= b + c :=
sorry

-- Negation is false
theorem negation_false (h : a < b) : ¬ (a + c >= b + c) :=
sorry

end original_proposition_contrapositive_converse_inverse_negation_false_l1670_167038


namespace sin_x_correct_l1670_167014

noncomputable def sin_x (a b c : ℝ) (x : ℝ) : ℝ :=
  2 * a * b * c / Real.sqrt (a^4 + 2 * a^2 * b^2 * (c^2 - 1) + b^4)

theorem sin_x_correct (a b c x : ℝ) 
  (h₁ : a > b) 
  (h₂ : b > 0) 
  (h₃ : c > 0) 
  (h₄ : 0 < x ∧ x < Real.pi / 2) 
  (h₅ : Real.tan x = 2 * a * b * c / (a^2 - b^2)) :
  Real.sin x = sin_x a b c x :=
sorry

end sin_x_correct_l1670_167014


namespace percentage_of_second_discount_is_correct_l1670_167049

def car_original_price : ℝ := 12000
def first_discount : ℝ := 0.20
def final_price_after_discounts : ℝ := 7752
def third_discount : ℝ := 0.05

def solve_percentage_second_discount : Prop := 
  ∃ (second_discount : ℝ), 
    (car_original_price * (1 - first_discount) * (1 - second_discount) * (1 - third_discount) = final_price_after_discounts) ∧ 
    (second_discount * 100 = 15)

theorem percentage_of_second_discount_is_correct : solve_percentage_second_discount :=
  sorry

end percentage_of_second_discount_is_correct_l1670_167049


namespace pencils_ratio_l1670_167059

theorem pencils_ratio
  (Sarah_pencils : ℕ)
  (Tyrah_pencils : ℕ)
  (Tim_pencils : ℕ)
  (h1 : Tyrah_pencils = 12)
  (h2 : Tim_pencils = 16)
  (h3 : Tim_pencils = 8 * Sarah_pencils) :
  Tyrah_pencils / Sarah_pencils = 6 :=
by
  sorry

end pencils_ratio_l1670_167059


namespace mean_of_three_l1670_167074

theorem mean_of_three (x y z a : ℝ)
  (h₁ : (x + y) / 2 = 5)
  (h₂ : (y + z) / 2 = 9)
  (h₃ : (z + x) / 2 = 10) :
  (x + y + z) / 3 = 8 :=
by
  sorry

end mean_of_three_l1670_167074


namespace hannah_dogs_food_total_l1670_167064

def first_dog_food : ℝ := 1.5
def second_dog_food : ℝ := 2 * first_dog_food
def third_dog_food : ℝ := second_dog_food + 2.5

theorem hannah_dogs_food_total : first_dog_food + second_dog_food + third_dog_food = 10 := by
  sorry

end hannah_dogs_food_total_l1670_167064


namespace travel_days_l1670_167013

variable (a b d : ℕ)

theorem travel_days (h1 : a + d = 11) (h2 : b + d = 21) (h3 : a + b = 12) : a + b + d = 22 :=
by sorry

end travel_days_l1670_167013


namespace determine_a_l1670_167090

theorem determine_a (a : ℕ)
  (h1 : 2 / (2 + 3 + a) = 1 / 3) : a = 1 :=
by
  sorry

end determine_a_l1670_167090


namespace slope_product_constant_l1670_167069

noncomputable def parabola_equation (p : ℝ) (hp : p > 0) : Prop :=
  ∀ x y : ℝ, (x ^ 2 = 2 * y ↔ x ^ 2 = 2 * p * y)

theorem slope_product_constant :
  ∀ (x1 y1 x2 y2 k1 k2 : ℝ) (P A B : ℝ × ℝ),
  P = (2, 2) →
  A = (x1, y1) →
  B = (x2, y2) →
  (∀ k: ℝ, y1 = k * (x1 + 2) + 4 ∧ y2 = k * (x2 + 2) + 4) →
  k1 = (y1 - 2) / (x1 - 2) →
  k2 = (y2 - 2) / (x2 - 2) →
  (x1 + x2 = 2 * k) →
  (x1 * x2 = -4 * k - 8) →
  k1 * k2 = -1 := 
  sorry

end slope_product_constant_l1670_167069


namespace N_subset_M_l1670_167098

-- Definitions of sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x * x - x < 0 }

-- Proof statement: N is a subset of M
theorem N_subset_M : N ⊆ M :=
sorry

end N_subset_M_l1670_167098


namespace max_abs_eq_one_vertices_l1670_167072

theorem max_abs_eq_one_vertices (x y : ℝ) :
  (max (|x + y|) (|x - y|) = 1) ↔ (x = -1 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 1) :=
sorry

end max_abs_eq_one_vertices_l1670_167072


namespace symmetric_points_sum_l1670_167082

theorem symmetric_points_sum {c e : ℤ} 
  (P : ℤ × ℤ × ℤ) 
  (sym_xoy : ℤ × ℤ × ℤ) 
  (sym_y : ℤ × ℤ × ℤ) 
  (hP : P = (-4, -2, 3)) 
  (h_sym_xoy : sym_xoy = (-4, -2, -3)) 
  (h_sym_y : sym_y = (4, -2, 3)) 
  (hc : c = -3) 
  (he : e = 4) : 
  c + e = 1 :=
by
  -- Proof goes here
  sorry

end symmetric_points_sum_l1670_167082


namespace polar_center_coordinates_l1670_167070

-- Define polar coordinate system equation
def polar_circle (ρ θ : ℝ) := ρ = 2 * Real.sin θ

-- Define the theorem: Given the equation of a circle in polar coordinates, its center in polar coordinates.
theorem polar_center_coordinates :
  (∀ (θ : ℝ), 0 ≤ θ ∧ θ < 2 * Real.pi → ∃ ρ, polar_circle ρ θ) →
  (∀ ρ θ, polar_circle ρ θ → 0 ≤ θ ∧ θ < 2 * Real.pi → (ρ = 1 ∧ θ = Real.pi / 2) ∨ (ρ = -1 ∧ θ = 3 * Real.pi / 2)) :=
by {
  sorry 
}

end polar_center_coordinates_l1670_167070


namespace tabitha_honey_days_l1670_167033

noncomputable def days_of_honey (cups_per_day servings_per_cup total_servings : ℕ) : ℕ :=
  total_servings / (cups_per_day * servings_per_cup)

theorem tabitha_honey_days :
  let cups_per_day := 3
  let servings_per_cup := 1
  let ounces_container := 16
  let servings_per_ounce := 6
  let total_servings := ounces_container * servings_per_ounce
  days_of_honey cups_per_day servings_per_cup total_servings = 32 :=
by
  sorry

end tabitha_honey_days_l1670_167033


namespace prime_numbers_eq_l1670_167011

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_numbers_eq 
  (p q r : ℕ)
  (hp : is_prime p)
  (hq : is_prime q)
  (hr : is_prime r)
  (h : p * (p - 7) + q * (q - 7) = r * (r - 7)) :
  (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 5 ∧ r = 7) ∨
  (p = 7 ∧ q = 5 ∧ r = 5) ∨ (p = 5 ∧ q = 7 ∧ r = 5) ∨
  (p = 5 ∧ q = 7 ∧ r = 2) ∨ (p = 7 ∧ q = 5 ∧ r = 2) ∨
  (p = 7 ∧ q = 3 ∧ r = 3) ∨ (p = 3 ∧ q = 7 ∧ r = 3) ∨
  (∃ (a : ℕ), is_prime a ∧ p = a ∧ q = 7 ∧ r = a) ∨
  (∃ (a : ℕ), is_prime a ∧ p = 7 ∧ q = a ∧ r = a) :=
sorry

end prime_numbers_eq_l1670_167011


namespace geometric_sequence_sum_S8_l1670_167099

noncomputable def sum_geometric_seq (a q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_S8 (a q : ℝ) (h1 : q ≠ 1)
  (h2 : sum_geometric_seq a q 4 = -5)
  (h3 : sum_geometric_seq a q 6 = 21 * sum_geometric_seq a q 2) :
  sum_geometric_seq a q 8 = -85 :=
sorry

end geometric_sequence_sum_S8_l1670_167099


namespace how_many_grapes_l1670_167044

-- Define the conditions given in the problem
def apples_to_grapes :=
  (3 / 4) * 12 = 6

-- Define the result to prove
def grapes_value :=
  (1 / 3) * 9 = 2

-- The statement combining the conditions and the problem to be proven
theorem how_many_grapes : apples_to_grapes → grapes_value :=
by
  intro h
  sorry

end how_many_grapes_l1670_167044


namespace roots_quad_sum_abs_gt_four_sqrt_three_l1670_167079

theorem roots_quad_sum_abs_gt_four_sqrt_three
  (p r1 r2 : ℝ)
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 12)
  (h3 : p^2 > 48) : 
  |r1 + r2| > 4 * Real.sqrt 3 := 
by 
  sorry

end roots_quad_sum_abs_gt_four_sqrt_three_l1670_167079


namespace number_of_correct_statements_l1670_167041

theorem number_of_correct_statements:
  (¬∀ (a : ℝ), -a < 0) ∧
  (∀ (x : ℝ), |x| = -x → x < 0) ∧
  (∀ (a : ℚ), (∀ (b : ℚ), |b| ≥ |a|) → a = 0) ∧
  (∀ (x y : ℝ), 5 * x^2 * y ≠ 0 → 2 + 1 = 3) →
  2 = 2 := sorry

end number_of_correct_statements_l1670_167041


namespace polygon_sides_l1670_167030

theorem polygon_sides (n : Nat) (h : (360 : ℝ) / (180 * (n - 2)) = 2 / 9) : n = 11 :=
by
  sorry

end polygon_sides_l1670_167030


namespace no_int_solutions_l1670_167055

theorem no_int_solutions (c x y : ℤ) (h1 : 0 < c) (h2 : c % 2 = 1) : x ^ 2 - y ^ 3 ≠ (2 * c) ^ 3 - 1 :=
sorry

end no_int_solutions_l1670_167055


namespace nearest_integer_ratio_l1670_167095

variable (a b : ℝ)

-- Given condition and constraints
def condition : Prop := (a > b) ∧ (b > 0) ∧ (a + b) / 2 = 3 * Real.sqrt (a * b)

-- Main statement to prove
theorem nearest_integer_ratio (h : condition a b) : Int.floor (a / b) = 34 ∨ Int.floor (a / b) = 33 := sorry

end nearest_integer_ratio_l1670_167095


namespace time_to_fill_tank_l1670_167017

-- Define the rates of the pipes
def rate_first_fill : ℚ := 1 / 15
def rate_second_fill : ℚ := 1 / 15
def rate_outlet_empty : ℚ := -1 / 45

-- Define the combined rate
def combined_rate : ℚ := rate_first_fill + rate_second_fill + rate_outlet_empty

-- Define the time to fill the tank
def fill_time (rate : ℚ) : ℚ := 1 / rate

theorem time_to_fill_tank : fill_time combined_rate = 9 := 
by 
  -- Proof omitted
  sorry

end time_to_fill_tank_l1670_167017


namespace prime_divisors_of_50_fact_eq_15_l1670_167012

theorem prime_divisors_of_50_fact_eq_15 :
  ∃ P : Finset Nat, (∀ p ∈ P, Prime p ∧ p ∣ (Nat.factorial 50)) ∧ P.card = 15 := by
  sorry

end prime_divisors_of_50_fact_eq_15_l1670_167012


namespace min_distance_between_intersections_range_of_a_l1670_167092

variable {a : ℝ}

/-- Given the function f(x) = x^2 - 2ax - 2(a + 1), 
1. Prove that the graph of function f(x) always intersects the x-axis at two distinct points.
2. For all x in the interval (-1, ∞), prove that f(x) + 3 ≥ 0 implies a ≤ sqrt 2 - 1. --/

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 2 * (a + 1)

theorem min_distance_between_intersections (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, (f x₁ a = 0) ∧ (f x₂ a = 0) ∧ (x₁ ≠ x₂) ∧ (dist x₁ x₂ = 2) := sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, -1 < x → f x a + 3 ≥ 0) → a ≤ Real.sqrt 2 - 1 := sorry

end min_distance_between_intersections_range_of_a_l1670_167092


namespace logarithmic_relationship_l1670_167071

theorem logarithmic_relationship
  (a b c : ℝ) (m n r : ℝ)
  (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) (h4 : 1 < c)
  (h5 : m = Real.log c / Real.log a)
  (h6 : n = Real.log c / Real.log b)
  (h7 : r = a ^ c) :
  n < m ∧ m < r :=
sorry

end logarithmic_relationship_l1670_167071


namespace positive_integer_solutions_inequality_l1670_167080

theorem positive_integer_solutions_inequality :
  {x : ℕ | 2 * x + 9 ≥ 3 * (x + 2)} = {1, 2, 3} :=
by
  sorry

end positive_integer_solutions_inequality_l1670_167080


namespace total_cats_correct_l1670_167025

-- Jamie's cats
def Jamie_Persian_cats : ℕ := 4
def Jamie_Maine_Coons : ℕ := 2

-- Gordon's cats
def Gordon_Persian_cats : ℕ := Jamie_Persian_cats / 2
def Gordon_Maine_Coons : ℕ := Jamie_Maine_Coons + 1

-- Hawkeye's cats
def Hawkeye_Persian_cats : ℕ := 0
def Hawkeye_Maine_Coons : ℕ := Gordon_Maine_Coons - 1

-- Total cats for each person
def Jamie_total_cats : ℕ := Jamie_Persian_cats + Jamie_Maine_Coons
def Gordon_total_cats : ℕ := Gordon_Persian_cats + Gordon_Maine_Coons
def Hawkeye_total_cats : ℕ := Hawkeye_Persian_cats + Hawkeye_Maine_Coons

-- Proof that the total number of cats is 13
theorem total_cats_correct : Jamie_total_cats + Gordon_total_cats + Hawkeye_total_cats = 13 :=
by sorry

end total_cats_correct_l1670_167025


namespace M_values_l1670_167016

theorem M_values (m n p M : ℝ) (h1 : M = m / (n + p)) (h2 : M = n / (p + m)) (h3 : M = p / (m + n)) :
  M = 1 / 2 ∨ M = -1 :=
by
  sorry

end M_values_l1670_167016


namespace line_equation_l1670_167062

theorem line_equation (b : ℝ) :
  (∃ b, (∀ x y, y = (3/4) * x + b) ∧ 
  (1/2) * |b| * |- (4/3) * b| = 6 →
  (3 * x - 4 * y + 12 = 0 ∨ 3 * x - 4 * y - 12 = 0)) := 
sorry

end line_equation_l1670_167062


namespace cheaper_to_buy_more_l1670_167050

def cost (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 30 then 15 * n
  else if 31 ≤ n ∧ n ≤ 60 then 13 * n
  else if 61 ≤ n ∧ n ≤ 90 then 12 * n
  else if 91 ≤ n then 11 * n
  else 0

theorem cheaper_to_buy_more (n : ℕ) : 
  (∃ m, m < n ∧ cost (m + 1) < cost m) ↔ n = 9 := sorry

end cheaper_to_buy_more_l1670_167050


namespace quilt_cost_calculation_l1670_167000

theorem quilt_cost_calculation :
  let length := 12
  let width := 15
  let cost_per_sq_foot := 70
  let sales_tax_rate := 0.05
  let discount_rate := 0.10
  let area := length * width
  let cost_before_discount := area * cost_per_sq_foot
  let discount_amount := cost_before_discount * discount_rate
  let cost_after_discount := cost_before_discount - discount_amount
  let sales_tax_amount := cost_after_discount * sales_tax_rate
  let total_cost := cost_after_discount + sales_tax_amount
  total_cost = 11907 := by
  {
    sorry
  }

end quilt_cost_calculation_l1670_167000


namespace smallest_class_number_l1670_167060

theorem smallest_class_number (x : ℕ)
  (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) + (x + 15) = 57) :
  x = 2 :=
by sorry

end smallest_class_number_l1670_167060


namespace two_m_plus_three_b_l1670_167047

noncomputable def m : ℚ := (-(3/2) - (1/2)) / (2 - (-1))

noncomputable def b : ℚ := (1/2) - m * (-1)

theorem two_m_plus_three_b :
  2 * m + 3 * b = -11 / 6 :=
by
  sorry

end two_m_plus_three_b_l1670_167047


namespace residue_11_pow_1234_mod_19_l1670_167022

theorem residue_11_pow_1234_mod_19 : 
  (11 ^ 1234) % 19 = 11 := 
by
  sorry

end residue_11_pow_1234_mod_19_l1670_167022
