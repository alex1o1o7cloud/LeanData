import Mathlib

namespace NUMINAMATH_GPT_find_c_l1530_153024

-- Given that the function f(x) = 2^x + c passes through the point (2,5),
-- Prove that c = 1.
theorem find_c (c : ℝ) : (∃ (f : ℝ → ℝ), (∀ x, f x = 2^x + c) ∧ (f 2 = 5)) → c = 1 := by
  sorry

end NUMINAMATH_GPT_find_c_l1530_153024


namespace NUMINAMATH_GPT_Roe_total_savings_l1530_153023

-- Define savings amounts per period
def savings_Jan_to_Jul : Int := 7 * 10
def savings_Aug_to_Nov : Int := 4 * 15
def savings_Dec : Int := 20

-- Define total savings for the year
def total_savings : Int := savings_Jan_to_Jul + savings_Aug_to_Nov + savings_Dec

-- Prove that Roe's total savings for the year is $150
theorem Roe_total_savings : total_savings = 150 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_Roe_total_savings_l1530_153023


namespace NUMINAMATH_GPT_max_min_x1_x2_squared_l1530_153039

theorem max_min_x1_x2_squared (k : ℝ) (x1 x2 : ℝ) 
  (h1 : x1^2 - (k-2)*x1 + (k^2 + 3*k + 5) = 0)
  (h2 : x2^2 - (k-2)*x2 + (k^2 + 3*k + 5) = 0)
  (h3 : -4 ≤ k ∧ k ≤ -4/3) : 
  (∃ (k_max k_min : ℝ), 
    k = -4 → x1^2 + x2^2 = 18 ∧ k = -4/3 → x1^2 + x2^2 = 50/9) :=
sorry

end NUMINAMATH_GPT_max_min_x1_x2_squared_l1530_153039


namespace NUMINAMATH_GPT_mary_chopped_tables_l1530_153056

-- Define the constants based on the conditions
def chairs_sticks := 6
def tables_sticks := 9
def stools_sticks := 2
def burn_rate := 5

-- Define the quantities of items Mary chopped up
def chopped_chairs := 18
def chopped_stools := 4
def warm_hours := 34
def sticks_from_chairs := chopped_chairs * chairs_sticks
def sticks_from_stools := chopped_stools * stools_sticks
def total_needed_sticks := warm_hours * burn_rate
def sticks_from_tables (chopped_tables : ℕ) := chopped_tables * tables_sticks

-- Define the proof goal
theorem mary_chopped_tables : ∃ chopped_tables, sticks_from_chairs + sticks_from_stools + sticks_from_tables chopped_tables = total_needed_sticks ∧ chopped_tables = 6 :=
by
  sorry

end NUMINAMATH_GPT_mary_chopped_tables_l1530_153056


namespace NUMINAMATH_GPT_work_completion_days_l1530_153015

structure WorkProblem :=
  (total_work : ℝ := 1) -- Assume total work to be 1 unit
  (days_A : ℝ := 30)
  (days_B : ℝ := 15)
  (days_together : ℝ := 5)

noncomputable def total_days_taken (wp : WorkProblem) : ℝ :=
  let work_per_day_A := 1 / wp.days_A
  let work_per_day_B := 1 / wp.days_B
  let work_per_day_together := work_per_day_A + work_per_day_B
  let work_done_together := wp.days_together * work_per_day_together
  let remaining_work := wp.total_work - work_done_together
  let days_for_A := remaining_work / work_per_day_A
  wp.days_together + days_for_A

theorem work_completion_days (wp : WorkProblem) : total_days_taken wp = 20 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_days_l1530_153015


namespace NUMINAMATH_GPT_positive_difference_perimeters_l1530_153066

theorem positive_difference_perimeters :
  let w1 := 3
  let h1 := 2
  let w2 := 6
  let h2 := 1
  let P1 := 2 * (w1 + h1)
  let P2 := 2 * (w2 + h2)
  P2 - P1 = 4 := by
  sorry

end NUMINAMATH_GPT_positive_difference_perimeters_l1530_153066


namespace NUMINAMATH_GPT_power_sum_zero_l1530_153027

theorem power_sum_zero (n : ℕ) (h : 0 < n) : (-1:ℤ)^(2*n) + (-1:ℤ)^(2*n+1) = 0 := 
by 
  sorry

end NUMINAMATH_GPT_power_sum_zero_l1530_153027


namespace NUMINAMATH_GPT_find_common_ratio_l1530_153051

variable {a : ℕ → ℝ} {q : ℝ}

-- Define that a is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem find_common_ratio
  (h1 : is_geometric_sequence a q)
  (h2 : 0 < q)
  (h3 : a 1 * a 3 = 1)
  (h4 : sum_first_n_terms a 3 = 7) :
  q = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_common_ratio_l1530_153051


namespace NUMINAMATH_GPT_sum_of_gcd_and_lcm_of_180_and_4620_l1530_153061

def gcd_180_4620 : ℕ := Nat.gcd 180 4620
def lcm_180_4620 : ℕ := Nat.lcm 180 4620
def sum_gcd_lcm_180_4620 : ℕ := gcd_180_4620 + lcm_180_4620

theorem sum_of_gcd_and_lcm_of_180_and_4620 :
  sum_gcd_lcm_180_4620 = 13920 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_gcd_and_lcm_of_180_and_4620_l1530_153061


namespace NUMINAMATH_GPT_debby_ate_candy_l1530_153078

theorem debby_ate_candy (initial_candy : ℕ) (remaining_candy : ℕ) (debby_initial : initial_candy = 12) (debby_remaining : remaining_candy = 3) : initial_candy - remaining_candy = 9 :=
by
  sorry

end NUMINAMATH_GPT_debby_ate_candy_l1530_153078


namespace NUMINAMATH_GPT_determine_C_for_identity_l1530_153070

theorem determine_C_for_identity :
  (∀ (x : ℝ), (1/2 * (Real.sin x)^2 + C = -1/4 * Real.cos (2 * x))) → C = -1/4 :=
by
  sorry

end NUMINAMATH_GPT_determine_C_for_identity_l1530_153070


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1530_153082

def set_M : Set ℝ := {x : ℝ | x^2 - x ≥ 0}
def set_N : Set ℝ := {x : ℝ | x < 2}

theorem intersection_of_M_and_N :
  set_M ∩ set_N = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1530_153082


namespace NUMINAMATH_GPT_rectangle_to_rhombus_l1530_153069

def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ D.2 = C.2 ∧ C.1 = B.1 ∧ B.2 = A.2

def is_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2) ≠ 0

def is_rhombus (A B C D : ℝ × ℝ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A

theorem rectangle_to_rhombus (A B C D : ℝ × ℝ) (h1 : is_rectangle A B C D) :
  ∃ X Y Z W : ℝ × ℝ, is_triangle A B C ∧ is_triangle A D C ∧ is_rhombus X Y Z W :=
by
  sorry

end NUMINAMATH_GPT_rectangle_to_rhombus_l1530_153069


namespace NUMINAMATH_GPT_simplify_expression_l1530_153013

theorem simplify_expression (x y : ℝ) (h_x_ne_0 : x ≠ 0) (h_y_ne_0 : y ≠ 0) :
  (25*x^3*y) * (8*x*y) * (1 / (5*x*y^2)^2) = 8*x^2 / y^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1530_153013


namespace NUMINAMATH_GPT_math_problem_l1530_153073

noncomputable def answer := 21

theorem math_problem 
  (a b c d x : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |x| = 3) : 
  2 * x^2 - (a * b - c - d) + |a * b + 3| = answer := 
sorry

end NUMINAMATH_GPT_math_problem_l1530_153073


namespace NUMINAMATH_GPT_optimal_tablet_combination_exists_l1530_153053

/-- Define the daily vitamin requirement structure --/
structure Vitamins (A B C D : ℕ)

theorem optimal_tablet_combination_exists {x y : ℕ} :
  (∃ (x y : ℕ), 
    (3 * x ≥ 3) ∧ (x + y ≥ 9) ∧ (x + 3 * y ≥ 15) ∧ (2 * y ≥ 2) ∧
    (x + y = 9) ∧ 
    (20 * x + 60 * y = 3) ∧ 
    (x + 2 * y = 12) ∧ 
    (x = 6 ∧ y = 3)) := 
  by
  sorry

end NUMINAMATH_GPT_optimal_tablet_combination_exists_l1530_153053


namespace NUMINAMATH_GPT_solve_problem_l1530_153079

-- Define the polynomial p(x)
noncomputable def p (x : ℂ) : ℂ := x^2 - x + 1

-- Define the root condition
def is_root (α : ℂ) : Prop := p (p (p (p α))) = 0

-- Define the expression to evaluate
noncomputable def expression (α : ℂ) : ℂ := (p α - 1) * p α * p (p α) * p (p (p α))

-- State the theorem asserting the required equality
theorem solve_problem (α : ℂ) (hα : is_root α) : expression α = -1 :=
sorry

end NUMINAMATH_GPT_solve_problem_l1530_153079


namespace NUMINAMATH_GPT_cost_equation_l1530_153026

def cost (W : ℕ) : ℕ :=
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10

theorem cost_equation (W : ℕ) : cost W = 
  if W ≤ 10 then 5 * W + 10 else 7 * W - 10 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_cost_equation_l1530_153026


namespace NUMINAMATH_GPT_holds_for_even_positive_l1530_153086

variable {n : ℕ}
variable (p : ℕ → Prop)

-- Conditions
axiom base_case : p 2
axiom inductive_step : ∀ k, p k → p (k + 2)

-- Theorem to prove
theorem holds_for_even_positive (n : ℕ) (h : n > 0) (h_even : n % 2 = 0) : p n :=
sorry

end NUMINAMATH_GPT_holds_for_even_positive_l1530_153086


namespace NUMINAMATH_GPT_compute_expression_l1530_153002

theorem compute_expression (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 49 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l1530_153002


namespace NUMINAMATH_GPT_february_first_day_of_week_l1530_153003

theorem february_first_day_of_week 
  (feb13_is_wednesday : ∃ day, day = 13 ∧ day_of_week = "Wednesday") :
  ∃ day, day = 1 ∧ day_of_week = "Friday" :=
sorry

end NUMINAMATH_GPT_february_first_day_of_week_l1530_153003


namespace NUMINAMATH_GPT_gcd_45345_34534_l1530_153072

theorem gcd_45345_34534 : Nat.gcd 45345 34534 = 71 := by
  sorry

end NUMINAMATH_GPT_gcd_45345_34534_l1530_153072


namespace NUMINAMATH_GPT_solve_x_l1530_153021

theorem solve_x (x : ℝ) (A B : ℝ × ℝ) (a : ℝ × ℝ) 
  (hA : A = (1, 3)) (hB : B = (2, 4))
  (ha : a = (2 * x - 1, x ^ 2 + 3 * x - 3))
  (hab : a = (B.1 - A.1, B.2 - A.2)) : x = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_x_l1530_153021


namespace NUMINAMATH_GPT_div_by_6_for_all_k_l1530_153045

def b_n_sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem div_by_6_for_all_k : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 → (b_n_sum_of_squares k) % 6 = 0 :=
by
  intros k hk
  sorry

end NUMINAMATH_GPT_div_by_6_for_all_k_l1530_153045


namespace NUMINAMATH_GPT_shifted_linear_func_is_2x_l1530_153065

-- Define the initial linear function
def linear_func (x : ℝ) : ℝ := 2 * x - 3

-- Define the shifted linear function
def shifted_linear_func (x : ℝ) : ℝ := linear_func x + 3

theorem shifted_linear_func_is_2x (x : ℝ) : shifted_linear_func x = 2 * x := by
  -- Proof would go here, but we use sorry to skip it
  sorry

end NUMINAMATH_GPT_shifted_linear_func_is_2x_l1530_153065


namespace NUMINAMATH_GPT_problem_equivalent_l1530_153044

theorem problem_equivalent
  (x : ℚ)
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) :
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 - Real.sqrt (x^4 - 4)) = 289 / 8 := 
by
  sorry

end NUMINAMATH_GPT_problem_equivalent_l1530_153044


namespace NUMINAMATH_GPT_symmetric_line_equation_l1530_153083

theorem symmetric_line_equation (x y : ℝ) (h : 4 * x - 3 * y + 5 = 0):
  4 * x + 3 * y + 5 = 0 :=
sorry

end NUMINAMATH_GPT_symmetric_line_equation_l1530_153083


namespace NUMINAMATH_GPT_sum_of_palindromic_primes_less_than_70_l1530_153048

def is_prime (n : ℕ) : Prop := Nat.Prime n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_prime (reverse_digits n)

theorem sum_of_palindromic_primes_less_than_70 :
  let palindromic_primes := [11, 13, 31, 37]
  (∀ p ∈ palindromic_primes, is_palindromic_prime p ∧ p < 70) →
  palindromic_primes.sum = 92 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_palindromic_primes_less_than_70_l1530_153048


namespace NUMINAMATH_GPT_common_z_values_l1530_153049

theorem common_z_values (z : ℝ) :
  (∃ x : ℝ, x^2 + z^2 = 9 ∧ x^2 = 4*z - 5) ↔ (z = -2 + 3*Real.sqrt 2 ∨ z = -2 - 3*Real.sqrt 2) := 
sorry

end NUMINAMATH_GPT_common_z_values_l1530_153049


namespace NUMINAMATH_GPT_six_letter_word_combinations_l1530_153011

theorem six_letter_word_combinations : ∃ n : ℕ, n = 26 * 26 * 26 := 
sorry

end NUMINAMATH_GPT_six_letter_word_combinations_l1530_153011


namespace NUMINAMATH_GPT_books_sold_in_store_on_saturday_l1530_153089

namespace BookshopInventory

def initial_inventory : ℕ := 743
def saturday_online_sales : ℕ := 128
def sunday_online_sales : ℕ := 162
def shipment_received : ℕ := 160
def final_inventory : ℕ := 502

-- Define the total number of books sold
def total_books_sold (S : ℕ) : ℕ := S + saturday_online_sales + 2 * S + sunday_online_sales

-- Net change in inventory equals total books sold minus shipment received
def net_change_in_inventory (S : ℕ) : ℕ := total_books_sold S - shipment_received

-- Prove that the difference between initial and final inventories equals the net change in inventory
theorem books_sold_in_store_on_saturday : ∃ S : ℕ, net_change_in_inventory S = initial_inventory - final_inventory ∧ S = 37 :=
by
  sorry

end BookshopInventory

end NUMINAMATH_GPT_books_sold_in_store_on_saturday_l1530_153089


namespace NUMINAMATH_GPT_smallest_positive_n_l1530_153090

theorem smallest_positive_n (n : ℕ) (h : 19 * n ≡ 789 [MOD 11]) : n = 1 := 
by
  sorry

end NUMINAMATH_GPT_smallest_positive_n_l1530_153090


namespace NUMINAMATH_GPT_lisa_punch_l1530_153005

theorem lisa_punch (x : ℝ) (H : x = 0.125) :
  (0.3 + x) / (2 + x) = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_lisa_punch_l1530_153005


namespace NUMINAMATH_GPT_sales_neither_notebooks_nor_markers_l1530_153096

theorem sales_neither_notebooks_nor_markers (percent_notebooks percent_markers percent_staplers : ℝ) 
  (h1 : percent_notebooks = 25)
  (h2 : percent_markers = 40)
  (h3 : percent_staplers = 15) : 
  percent_staplers + (100 - (percent_notebooks + percent_markers + percent_staplers)) = 35 :=
by
  sorry

end NUMINAMATH_GPT_sales_neither_notebooks_nor_markers_l1530_153096


namespace NUMINAMATH_GPT_min_discount_70_percent_l1530_153012

theorem min_discount_70_percent
  (P S : ℝ) (M : ℝ)
  (hP : P = 800)
  (hS : S = 1200)
  (hM : M = 0.05) :
  ∃ D : ℝ, D = 0.7 ∧ S * D - P ≥ P * M :=
by sorry

end NUMINAMATH_GPT_min_discount_70_percent_l1530_153012


namespace NUMINAMATH_GPT_number_of_10_people_rows_l1530_153034

theorem number_of_10_people_rows (x r : ℕ) (h1 : r = 54) (h2 : ∀ i : ℕ, i * 9 + x * 10 = 54) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_number_of_10_people_rows_l1530_153034


namespace NUMINAMATH_GPT_number_of_solutions_eq_one_l1530_153014

theorem number_of_solutions_eq_one :
  (∃! y : ℝ, (y ≠ 0) ∧ (y ≠ 3) ∧ ((3 * y^2 - 15 * y) / (y^2 - 3 * y) = y + 1)) :=
  sorry

end NUMINAMATH_GPT_number_of_solutions_eq_one_l1530_153014


namespace NUMINAMATH_GPT_saving_is_zero_cents_l1530_153080

-- Define the in-store and online prices
def in_store_price : ℝ := 129.99
def online_payment_per_installment : ℝ := 29.99
def shipping_and_handling : ℝ := 11.99

-- Define the online total price
def online_total_price : ℝ := 4 * online_payment_per_installment + shipping_and_handling

-- Define the saving in cents
def saving_in_cents : ℝ := (in_store_price - online_total_price) * 100

-- State the theorem to prove the number of cents saved
theorem saving_is_zero_cents : saving_in_cents = 0 := by
  sorry

end NUMINAMATH_GPT_saving_is_zero_cents_l1530_153080


namespace NUMINAMATH_GPT_servings_in_bottle_l1530_153047

theorem servings_in_bottle (total_revenue : ℕ) (price_per_serving : ℕ) (h1 : total_revenue = 98) (h2 : price_per_serving = 8) : Nat.floor (total_revenue / price_per_serving) = 12 :=
by
  sorry

end NUMINAMATH_GPT_servings_in_bottle_l1530_153047


namespace NUMINAMATH_GPT_part1_part2_l1530_153081

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x - a - 1|

-- Question 1: Prove that f(x) ≥ 3/4
theorem part1 (x a : ℝ) : f x a ≥ 3 / 4 := 
sorry

-- Question 2: Given f(4) < 13, find the range of a
theorem part2 (a : ℝ) (h : f 4 a < 13) : -2 < a ∧ a < 3 := 
sorry

end NUMINAMATH_GPT_part1_part2_l1530_153081


namespace NUMINAMATH_GPT_problem1_problem2_l1530_153054

noncomputable def f (x a : ℝ) := |x - 1| + |x - a|

theorem problem1 (x : ℝ) : f x (-1) ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5 :=
sorry

theorem problem2 (a : ℝ) : (∀ x, f x a ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1530_153054


namespace NUMINAMATH_GPT_xiao_ming_climb_stairs_8_l1530_153052

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | (n+2) => fibonacci n + fibonacci (n + 1)

theorem xiao_ming_climb_stairs_8 :
  fibonacci 8 = 34 :=
sorry

end NUMINAMATH_GPT_xiao_ming_climb_stairs_8_l1530_153052


namespace NUMINAMATH_GPT_total_area_of_combined_shape_l1530_153043

theorem total_area_of_combined_shape
  (length_rectangle : ℝ) (width_rectangle : ℝ) (side_square : ℝ)
  (h_length : length_rectangle = 0.45)
  (h_width : width_rectangle = 0.25)
  (h_side : side_square = 0.15) :
  (length_rectangle * width_rectangle + side_square * side_square) = 0.135 := 
by 
  sorry

end NUMINAMATH_GPT_total_area_of_combined_shape_l1530_153043


namespace NUMINAMATH_GPT_inscribed_sphere_radius_l1530_153025

noncomputable def radius_inscribed_sphere (S1 S2 S3 S4 V : ℝ) : ℝ :=
  3 * V / (S1 + S2 + S3 + S4)

theorem inscribed_sphere_radius (S1 S2 S3 S4 V R : ℝ) :
  R = radius_inscribed_sphere S1 S2 S3 S4 V :=
by
  sorry

end NUMINAMATH_GPT_inscribed_sphere_radius_l1530_153025


namespace NUMINAMATH_GPT_reciprocals_sum_of_roots_l1530_153031

theorem reciprocals_sum_of_roots (r s γ δ : ℚ) (h1 : 7 * r^2 + 5 * r + 3 = 0) (h2 : 7 * s^2 + 5 * s + 3 = 0) (h3 : γ = 1/r) (h4 : δ = 1/s) :
  γ + δ = -5/3 := 
  by 
    sorry

end NUMINAMATH_GPT_reciprocals_sum_of_roots_l1530_153031


namespace NUMINAMATH_GPT_polynomial_integer_roots_l1530_153076

theorem polynomial_integer_roots
  (b c : ℤ)
  (x1 x2 x1' x2' : ℤ)
  (h_eq1 : x1 * x2 > 0)
  (h_eq2 : x1' * x2' > 0)
  (h_eq3 : x1^2 + b * x1 + c = 0)
  (h_eq4 : x2^2 + b * x2 + c = 0)
  (h_eq5 : x1'^2 + c * x1' + b = 0)
  (h_eq6 : x2'^2 + c * x2' + b = 0)
  : x1 < 0 ∧ x2 < 0 ∧ b - 1 ≤ c ∧ c ≤ b + 1 ∧ 
    ((b = 5 ∧ c = 6) ∨ (b = 6 ∧ c = 5) ∨ (b = 4 ∧ c = 4)) := 
sorry

end NUMINAMATH_GPT_polynomial_integer_roots_l1530_153076


namespace NUMINAMATH_GPT_Y_pdf_from_X_pdf_l1530_153006

/-- Given random variable X with PDF p(x), prove PDF of Y = X^3 -/
noncomputable def X_pdf (σ : ℝ) (x : ℝ) : ℝ := 
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2))

noncomputable def Y_pdf (σ : ℝ) (y : ℝ) : ℝ :=
  (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2))

theorem Y_pdf_from_X_pdf (σ : ℝ) (y : ℝ) :
  ∀ x : ℝ, X_pdf σ x = (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x ^ 2) / (2 * σ ^ 2)) →
  Y_pdf σ y = (1 / (3 * σ * y^(2/3) * Real.sqrt (2 * Real.pi))) * Real.exp (-(y^(2/3)) / (2 * σ ^ 2)) :=
sorry

end NUMINAMATH_GPT_Y_pdf_from_X_pdf_l1530_153006


namespace NUMINAMATH_GPT_tan_sin_difference_l1530_153001

theorem tan_sin_difference :
  let tan_60 := Real.tan (60 * Real.pi / 180)
  let sin_60 := Real.sin (60 * Real.pi / 180)
  tan_60 - sin_60 = (Real.sqrt 3 / 2) := by
sorry

end NUMINAMATH_GPT_tan_sin_difference_l1530_153001


namespace NUMINAMATH_GPT_problem1_problem2_l1530_153033

open Set Real

-- Definition of sets A, B, and C
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

-- Problem 1: Prove A ∪ B = { x | 1 ≤ x < 10 }
theorem problem1 : A ∪ B = { x : ℝ | 1 ≤ x ∧ x < 10 } :=
sorry

-- Problem 2: Prove the range of a given the conditions
theorem problem2 (a : ℝ) (h1 : (A ∩ C a) ≠ ∅) (h2 : (B ∩ C a) = ∅) : 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1530_153033


namespace NUMINAMATH_GPT_figure_total_area_l1530_153063

theorem figure_total_area (a : ℝ) (h : a^2 - (3/2 * a^2) = 0.6) : 
  5 * a^2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_figure_total_area_l1530_153063


namespace NUMINAMATH_GPT_max_value_expression_l1530_153050

theorem max_value_expression (θ : ℝ) : 
  2 ≤ 5 + 3 * Real.sin θ ∧ 5 + 3 * Real.sin θ ≤ 8 → 
  (∃ θ, (14 / (5 + 3 * Real.sin θ)) = 7) := 
sorry

end NUMINAMATH_GPT_max_value_expression_l1530_153050


namespace NUMINAMATH_GPT_inequality_solution_l1530_153058

theorem inequality_solution {x : ℝ} :
  {x | (2 * x - 8) * (x - 4) / x ≥ 0} = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1530_153058


namespace NUMINAMATH_GPT_fill_table_with_numbers_l1530_153099

-- Define the main theorem based on the conditions and question.
theorem fill_table_with_numbers (numbers : Finset ℤ) (table : ℕ → ℕ → ℤ)
  (h_numbers_card : numbers.card = 100)
  (h_sum_1x3_horizontal : ∀ i j, (table i j + table i (j + 1) + table i (j + 2) ∈ numbers))
  (h_sum_1x3_vertical : ∀ i j, (table i j + table (i + 1) j + table (i + 2) j ∈ numbers)):
  ∃ (t : ℕ → ℕ → ℤ), (∀ k, 1 ≤ k ∧ k ≤ 6 → ∃ i j, t i j = k) :=
sorry

end NUMINAMATH_GPT_fill_table_with_numbers_l1530_153099


namespace NUMINAMATH_GPT_problem_statement_l1530_153038

theorem problem_statement (x y : ℝ) (h : x / y = 2) : (x - y) / x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1530_153038


namespace NUMINAMATH_GPT_problem_l1530_153059

theorem problem (a b : ℤ) (ha : a = 4) (hb : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := 
by
  -- Provide proof here
  sorry

end NUMINAMATH_GPT_problem_l1530_153059


namespace NUMINAMATH_GPT_number_of_players_l1530_153067
-- Importing the necessary library

-- Define the number of games formula for the tournament
def number_of_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- The theorem to prove the number of players given the conditions
theorem number_of_players (n : ℕ) (h : number_of_games n = 306) : n = 18 :=
by
  sorry

end NUMINAMATH_GPT_number_of_players_l1530_153067


namespace NUMINAMATH_GPT_max_distance_between_circle_centers_l1530_153004

theorem max_distance_between_circle_centers :
  let rect_width := 20
  let rect_height := 16
  let circle_diameter := 8
  let horiz_distance := rect_width - circle_diameter
  let vert_distance := rect_height - circle_diameter
  let max_distance := Real.sqrt (horiz_distance ^ 2 + vert_distance ^ 2)
  max_distance = 4 * Real.sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_between_circle_centers_l1530_153004


namespace NUMINAMATH_GPT_green_caps_percentage_l1530_153032

variable (total_caps : ℕ) (red_caps : ℕ)

def green_caps (total_caps red_caps: ℕ) : ℕ :=
  total_caps - red_caps

def percentage_of_green_caps (total_caps green_caps: ℕ) : ℕ :=
  (green_caps * 100) / total_caps

theorem green_caps_percentage :
  (total_caps = 125) →
  (red_caps = 50) →
  percentage_of_green_caps total_caps (green_caps total_caps red_caps) = 60 :=
by
  intros h1 h2
  rw [h1, h2]
  exact sorry  -- The proof is omitted 

end NUMINAMATH_GPT_green_caps_percentage_l1530_153032


namespace NUMINAMATH_GPT_negation_of_both_even_l1530_153008

-- Definitions
def even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Main statement
theorem negation_of_both_even (a b : ℕ) : ¬ (even a ∧ even b) ↔ (¬even a ∨ ¬even b) :=
by sorry

end NUMINAMATH_GPT_negation_of_both_even_l1530_153008


namespace NUMINAMATH_GPT_rotated_intersection_point_l1530_153035

theorem rotated_intersection_point (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) : 
  ∃ P : ℝ × ℝ, P = (-Real.sin θ, Real.cos θ) ∧ 
    ∃ φ : ℝ, φ = θ + π / 2 ∧ 
      P = (Real.cos φ, Real.sin φ) := 
by
  sorry

end NUMINAMATH_GPT_rotated_intersection_point_l1530_153035


namespace NUMINAMATH_GPT_z_pow12_plus_inv_z_pow12_l1530_153091

open Complex

theorem z_pow12_plus_inv_z_pow12 (z: ℂ) (h: z + z⁻¹ = 2 * cos (10 * Real.pi / 180)) :
  z^12 + z⁻¹^12 = -1 := by
  sorry

end NUMINAMATH_GPT_z_pow12_plus_inv_z_pow12_l1530_153091


namespace NUMINAMATH_GPT_largest_alternating_geometric_four_digit_number_l1530_153068

theorem largest_alternating_geometric_four_digit_number :
  ∃ (a b c d : ℕ), 
  (9 = 2 * b) ∧ (b = 2 * c) ∧ (a = 3) ∧ (9 * d = b * c) ∧ 
  (a > b) ∧ (b < c) ∧ (c > d) ∧ (1000 * a + 100 * b + 10 * c + d = 9632) := sorry

end NUMINAMATH_GPT_largest_alternating_geometric_four_digit_number_l1530_153068


namespace NUMINAMATH_GPT_smallest_perimeter_of_consecutive_even_triangle_l1530_153040

theorem smallest_perimeter_of_consecutive_even_triangle (n : ℕ) :
  (2 * n + 2 * n + 2 > 2 * n + 4) ∧
  (2 * n + 2 * n + 4 > 2 * n + 2) ∧
  (2 * n + 2 + 2 * n + 4 > 2 * n) →
  2 * n + (2 * n + 2) + (2 * n + 4) = 18 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_perimeter_of_consecutive_even_triangle_l1530_153040


namespace NUMINAMATH_GPT_right_triangle_num_array_l1530_153057

theorem right_triangle_num_array (n : ℕ) (hn : 0 < n) 
    (a : ℕ → ℕ → ℝ) 
    (h1 : a 1 1 = 1/4)
    (hd : ∀ i j, 0 < j → j <= i → a (i+1) 1 = a i 1 + 1/4)
    (hq : ∀ i j, 2 < i → 0 < j → j ≤ i → a i (j+1) = a i j * (1/2)) :
  a n 3 = n / 16 := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_num_array_l1530_153057


namespace NUMINAMATH_GPT_range_of_a_l1530_153075

-- Define the function f(x) and its condition
def f (x a : ℝ) : ℝ := x^2 + (a + 2) * x + (a - 1)

-- Given condition: f(-1, a) = -2
def condition (a : ℝ) : Prop := f (-1) a = -2

-- Requirement for the domain of g(x) = ln(f(x) + 3) being ℝ
def domain_requirement (a : ℝ) : Prop := ∀ x : ℝ, f x a + 3 > 0

-- Main theorem to prove the range of a
theorem range_of_a : {a : ℝ // condition a ∧ domain_requirement a} = {a : ℝ // -2 < a ∧ a < 2} :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1530_153075


namespace NUMINAMATH_GPT_travel_time_l1530_153084

-- Definitions of the conditions
variables (x : ℝ) (speed_elder speed_younger : ℝ)
variables (time_elder_total time_younger_total : ℝ)

def elder_speed_condition : Prop := speed_elder = x
def younger_speed_condition : Prop := speed_younger = x - 4
def elder_distance : Prop := 42 / speed_elder + 1 = time_elder_total
def younger_distance : Prop := 42 / speed_younger + 1 / 3 = time_younger_total

-- The main theorem we want to prove
theorem travel_time : ∀ (x : ℝ), 
  elder_speed_condition x speed_elder → 
  younger_speed_condition x speed_younger → 
  elder_distance speed_elder time_elder_total → 
  younger_distance speed_younger time_younger_total → 
  time_elder_total = time_younger_total ∧ time_elder_total = (10 / 3) :=
sorry

end NUMINAMATH_GPT_travel_time_l1530_153084


namespace NUMINAMATH_GPT_grid_rows_l1530_153093

theorem grid_rows (R : ℕ) :
  let squares_per_row := 15
  let red_squares := 4 * 6
  let blue_squares := 4 * squares_per_row
  let green_squares := 66
  let total_squares := red_squares + blue_squares + green_squares 
  total_squares = squares_per_row * R →
  R = 10 :=
by
  intros
  sorry

end NUMINAMATH_GPT_grid_rows_l1530_153093


namespace NUMINAMATH_GPT_apple_equation_l1530_153028

-- Conditions directly from a)
def condition1 (x : ℕ) : Prop := (x - 1) % 3 = 0
def condition2 (x : ℕ) : Prop := (x + 2) % 4 = 0

theorem apple_equation (x : ℕ) (h1 : condition1 x) (h2 : condition2 x) : 
  (x - 1) / 3 = (x + 2) / 4 := 
sorry

end NUMINAMATH_GPT_apple_equation_l1530_153028


namespace NUMINAMATH_GPT_solve_inequality_l1530_153055

theorem solve_inequality :
  {x : ℝ | (x - 3)*(x - 4)*(x - 5) / ((x - 2)*(x - 6)*(x - 7)) > 0} =
  {x : ℝ | x < 2} ∪ {x : ℝ | 4 < x ∧ x < 5} ∪ {x : ℝ | 6 < x ∧ x < 7} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l1530_153055


namespace NUMINAMATH_GPT_minimum_throws_to_ensure_same_sum_twice_l1530_153064

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end NUMINAMATH_GPT_minimum_throws_to_ensure_same_sum_twice_l1530_153064


namespace NUMINAMATH_GPT_required_bricks_l1530_153095

def brick_volume (length width height : ℝ) : ℝ := length * width * height

def wall_volume (length width height : ℝ) : ℝ := length * width * height

theorem required_bricks : 
  let brick_length := 25
  let brick_width := 11.25
  let brick_height := 6
  let wall_length := 850
  let wall_width := 600
  let wall_height := 22.5
  (wall_volume wall_length wall_width wall_height) / 
  (brick_volume brick_length brick_width brick_height) = 6800 :=
by
  sorry

end NUMINAMATH_GPT_required_bricks_l1530_153095


namespace NUMINAMATH_GPT_decreasing_interval_l1530_153097

theorem decreasing_interval (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 - 2 * x) :
  {x | deriv f x < 0} = {x | x < 1} :=
by
  sorry

end NUMINAMATH_GPT_decreasing_interval_l1530_153097


namespace NUMINAMATH_GPT_find_angle_C_l1530_153007

variable {A B C : ℝ} -- Angles of triangle ABC
variable {a b c : ℝ} -- Sides opposite the respective angles

theorem find_angle_C
  (h1 : 2 * c * Real.cos B = 2 * a + b) : 
  C = 120 :=
  sorry

end NUMINAMATH_GPT_find_angle_C_l1530_153007


namespace NUMINAMATH_GPT_ratio_r_to_pq_l1530_153029

theorem ratio_r_to_pq (p q r : ℕ) (h₁ : p + q + r = 5000) (h₂ : r = 2000) :
  r / (p + q) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_ratio_r_to_pq_l1530_153029


namespace NUMINAMATH_GPT_field_perimeter_l1530_153062

noncomputable def outer_perimeter (posts : ℕ) (post_width_inches : ℝ) (spacing_feet : ℝ) : ℝ :=
  let posts_per_side := posts / 4
  let gaps_per_side := posts_per_side - 1
  let post_width_feet := post_width_inches / 12
  let side_length := gaps_per_side * spacing_feet + posts_per_side * post_width_feet
  4 * side_length

theorem field_perimeter : 
  outer_perimeter 32 5 4 = 125 + 1/3 := 
by
  sorry

end NUMINAMATH_GPT_field_perimeter_l1530_153062


namespace NUMINAMATH_GPT_simplify_complex_expression_l1530_153009

variables (x y : ℝ) (i : ℂ)

theorem simplify_complex_expression (h : i^2 = -1) :
  (x + 3 * i * y) * (x - 3 * i * y) = x^2 - 9 * y^2 :=
by sorry

end NUMINAMATH_GPT_simplify_complex_expression_l1530_153009


namespace NUMINAMATH_GPT_feeding_ways_correct_l1530_153037

def total_feeding_ways : Nat :=
  (5 * 6 * (5 * 4 * 3 * 2 * 1)^2)

theorem feeding_ways_correct :
  total_feeding_ways = 432000 :=
by
  -- Proof is omitted here
  sorry

end NUMINAMATH_GPT_feeding_ways_correct_l1530_153037


namespace NUMINAMATH_GPT_alice_height_after_growth_l1530_153022

/-- Conditions: Bob and Alice were initially the same height. Bob has grown by 25%, Alice 
has grown by one third as many inches as Bob, and Bob is now 75 inches tall. --/
theorem alice_height_after_growth (initial_height : ℕ)
  (bob_growth_rate : ℚ)
  (alice_growth_ratio : ℚ)
  (bob_final_height : ℕ) :
  bob_growth_rate = 0.25 →
  alice_growth_ratio = 1 / 3 →
  bob_final_height = 75 →
  initial_height + (bob_final_height - initial_height) / 3 = 65 :=
by
  sorry

end NUMINAMATH_GPT_alice_height_after_growth_l1530_153022


namespace NUMINAMATH_GPT_common_ratio_geom_series_l1530_153030

theorem common_ratio_geom_series :
  let a₁ : ℚ := 4/7
  let a₂ : ℚ := -16/21
  let a₃ : ℚ := -64/63
  ∃ r : ℚ, r = a₂ / a₁ ∧ r = a₃ / a₂ ∧ r = -4/3 := 
by
  sorry

end NUMINAMATH_GPT_common_ratio_geom_series_l1530_153030


namespace NUMINAMATH_GPT_sum_of_decimals_as_fraction_l1530_153010

theorem sum_of_decimals_as_fraction :
  (0.2 : ℚ) + 0.04 + 0.006 + 0.0008 + 0.00010 = 2469 / 10000 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_as_fraction_l1530_153010


namespace NUMINAMATH_GPT_trig_identity_example_l1530_153094

theorem trig_identity_example :
  (Real.sin (36 * Real.pi / 180) * Real.cos (6 * Real.pi / 180) -
   Real.sin (54 * Real.pi / 180) * Real.cos (84 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_example_l1530_153094


namespace NUMINAMATH_GPT_range_of_function_l1530_153060

theorem range_of_function :
  ∃ (S : Set ℝ), (∀ x : ℝ, (1 / 2)^(x^2 - 2) ∈ S) ∧ S = Set.Ioc 0 4 := by
  sorry

end NUMINAMATH_GPT_range_of_function_l1530_153060


namespace NUMINAMATH_GPT_correct_statement_l1530_153046

def correct_input_format_1 (s : String) : Prop :=
  s = "INPUT a, b, c"

def correct_input_format_2 (s : String) : Prop :=
  s = "INPUT x="

def correct_output_format_1 (s : String) : Prop :=
  s = "PRINT A="

def correct_output_format_2 (s : String) : Prop :=
  s = "PRINT 3*2"

theorem correct_statement : (correct_input_format_1 "INPUT a; b; c" = false) ∧
                            (correct_input_format_2 "INPUT x=3" = false) ∧
                            (correct_output_format_1 "PRINT“A=4”" = false) ∧
                            (correct_output_format_2 "PRINT 3*2" = true) :=
by sorry

end NUMINAMATH_GPT_correct_statement_l1530_153046


namespace NUMINAMATH_GPT_pizza_combinations_l1530_153088

/-- The number of unique pizzas that can be made with exactly 5 toppings from a selection of 8 is 56. -/
theorem pizza_combinations : (Nat.choose 8 5) = 56 := by
  sorry

end NUMINAMATH_GPT_pizza_combinations_l1530_153088


namespace NUMINAMATH_GPT_sum_of_coordinates_B_l1530_153098

theorem sum_of_coordinates_B :
  ∃ (x y : ℝ), (3, 5) = ((x + 6) / 2, (y + 8) / 2) ∧ x + y = 2 := by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_B_l1530_153098


namespace NUMINAMATH_GPT_inequality_solution_l1530_153041

theorem inequality_solution (x : ℝ) (h : x ≠ 0) : 
  (1 / (x^2 + 1) > 2 * x^2 / x + 13 / 10) ↔ (x ∈ Set.Ioo (-1.6) 0 ∨ x ∈ Set.Ioi 0.8) :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l1530_153041


namespace NUMINAMATH_GPT_shares_sum_4000_l1530_153016

variables (w x y z : ℝ)

def relation_z_w : Prop := z = 1.20 * w
def relation_y_z : Prop := y = 1.25 * z
def relation_x_y : Prop := x = 1.35 * y
def w_after_3_years : ℝ := 8 * w
def z_after_3_years : ℝ := 8 * z
def y_after_3_years : ℝ := 8 * y
def x_after_3_years : ℝ := 8 * x

theorem shares_sum_4000 (w : ℝ) :
  relation_z_w w z →
  relation_y_z z y →
  relation_x_y y x →
  x_after_3_years x + y_after_3_years y + z_after_3_years z + w_after_3_years w = 4000 :=
by
  intros h_z_w h_y_z h_x_y
  rw [relation_z_w, relation_y_z, relation_x_y] at *
  sorry

end NUMINAMATH_GPT_shares_sum_4000_l1530_153016


namespace NUMINAMATH_GPT_probability_four_ones_in_five_rolls_l1530_153071

-- Define the probability of rolling a 1 on a fair six-sided die
def prob_one_roll_one : ℚ := 1 / 6

-- Define the probability of not rolling a 1 on a fair six-sided die
def prob_one_roll_not_one : ℚ := 5 / 6

-- Define the number of successes needed, here 4 ones in 5 rolls
def num_successes : ℕ := 4

-- Define the total number of trials, here 5 rolls
def num_trials : ℕ := 5

-- Binomial probability calculation for 4 successes in 5 trials with probability of success prob_one_roll_one
def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_four_ones_in_five_rolls : binomial_prob num_trials num_successes prob_one_roll_one = 25 / 7776 := 
by
  sorry

end NUMINAMATH_GPT_probability_four_ones_in_five_rolls_l1530_153071


namespace NUMINAMATH_GPT_max_value_l1530_153085

theorem max_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 2) : 
  2 * x * y + 2 * y * z * Real.sqrt 3 ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_l1530_153085


namespace NUMINAMATH_GPT_total_apples_eaten_l1530_153019

def simone_consumption (days: ℕ) (consumption_per_day: ℚ) : ℚ := days * consumption_per_day
def lauri_consumption (days: ℕ) (consumption_per_day: ℚ) : ℚ := days * consumption_per_day

theorem total_apples_eaten :
  simone_consumption 16 (1/2) + lauri_consumption 15 (1/3) = 13 := by
  sorry

end NUMINAMATH_GPT_total_apples_eaten_l1530_153019


namespace NUMINAMATH_GPT_joan_total_spending_l1530_153017

def basketball_game_price : ℝ := 5.20
def basketball_game_discount : ℝ := 0.15 * basketball_game_price
def basketball_game_discounted : ℝ := basketball_game_price - basketball_game_discount

def racing_game_price : ℝ := 4.23
def racing_game_discount : ℝ := 0.10 * racing_game_price
def racing_game_discounted : ℝ := racing_game_price - racing_game_discount

def puzzle_game_price : ℝ := 3.50

def total_before_tax : ℝ := basketball_game_discounted + racing_game_discounted + puzzle_game_price
def sales_tax : ℝ := 0.08 * total_before_tax
def total_with_tax : ℝ := total_before_tax + sales_tax

theorem joan_total_spending : (total_with_tax : ℝ) = 12.67 := by
  sorry

end NUMINAMATH_GPT_joan_total_spending_l1530_153017


namespace NUMINAMATH_GPT_sum_of_g1_l1530_153018

noncomputable def g : ℝ → ℝ := sorry

lemma problem_condition : ∀ x y : ℝ, g (g (x - y)) = g x + g y - g x * g y - x * y := sorry

theorem sum_of_g1 : g 1 = 1 := 
by
  -- Provide the necessary proof steps to show g(1) = 1
  sorry

end NUMINAMATH_GPT_sum_of_g1_l1530_153018


namespace NUMINAMATH_GPT_largest_non_factor_product_of_factors_of_100_l1530_153087

theorem largest_non_factor_product_of_factors_of_100 :
  ∃ x y : ℕ, 
  (x ≠ y) ∧ 
  (0 < x ∧ 0 < y) ∧ 
  (x ∣ 100 ∧ y ∣ 100) ∧ 
  ¬(x * y ∣ 100) ∧ 
  (∀ a b : ℕ, 
    (a ≠ b) ∧ 
    (0 < a ∧ 0 < b) ∧ 
    (a ∣ 100 ∧ b ∣ 100) ∧ 
    ¬(a * b ∣ 100) → 
    (x * y) ≥ (a * b)) ∧ 
  (x * y) = 40 :=
by
  sorry

end NUMINAMATH_GPT_largest_non_factor_product_of_factors_of_100_l1530_153087


namespace NUMINAMATH_GPT_samantha_birth_year_l1530_153042

theorem samantha_birth_year 
  (first_amc8 : ℕ)
  (amc8_annual : ∀ n : ℕ, n ≥ first_amc8)
  (seventh_amc8 : ℕ)
  (samantha_age : ℕ)
  (samantha_birth_year : ℕ)
  (move_year : ℕ)
  (h1 : first_amc8 = 1983)
  (h2 : seventh_amc8 = first_amc8 + 6)
  (h3 : seventh_amc8 = 1989)
  (h4 : samantha_age = 14)
  (h5 : samantha_birth_year = seventh_amc8 - samantha_age)
  (h6 : move_year = seventh_amc8 - 3) :
  samantha_birth_year = 1975 :=
sorry

end NUMINAMATH_GPT_samantha_birth_year_l1530_153042


namespace NUMINAMATH_GPT_coprime_n_minus_2_n_squared_minus_n_minus_1_l1530_153020

theorem coprime_n_minus_2_n_squared_minus_n_minus_1 (n : ℕ) : n - 2 ∣ n^2 - n - 1 → False :=
by
-- proof omitted as per instructions
sorry

end NUMINAMATH_GPT_coprime_n_minus_2_n_squared_minus_n_minus_1_l1530_153020


namespace NUMINAMATH_GPT_hyperbola_asymptotes_l1530_153000

theorem hyperbola_asymptotes :
  ∀ {x y : ℝ},
    (x^2 / 9 - y^2 / 16 = 1) →
    (y = (4 / 3) * x ∨ y = -(4 / 3) * x) :=
by
  intro x y h
  sorry

end NUMINAMATH_GPT_hyperbola_asymptotes_l1530_153000


namespace NUMINAMATH_GPT_number_of_children_coming_to_show_l1530_153036

theorem number_of_children_coming_to_show :
  ∀ (cost_adult cost_child : ℕ) (number_adults total_cost : ℕ),
  cost_adult = 12 →
  cost_child = 10 →
  number_adults = 3 →
  total_cost = 66 →
  ∃ (c : ℕ), 3 = c := by
    sorry

end NUMINAMATH_GPT_number_of_children_coming_to_show_l1530_153036


namespace NUMINAMATH_GPT_pentagon_area_is_correct_l1530_153077

noncomputable def area_of_pentagon : ℕ :=
  let area_trapezoid := (1 / 2) * (25 + 28) * 30
  let area_triangle := (1 / 2) * 18 * 24
  area_trapezoid + area_triangle

theorem pentagon_area_is_correct (s1 s2 s3 s4 s5 : ℕ) (b1 b2 h1 b3 h2 : ℕ)
  (h₀ : s1 = 18) (h₁ : s2 = 25) (h₂ : s3 = 30) (h₃ : s4 = 28) (h₄ : s5 = 25)
  (h₅ : b1 = 25) (h₆ : b2 = 28) (h₇ : h1 = 30) (h₈ : b3 = 18) (h₉ : h2 = 24) :
  area_of_pentagon = 1011 := by
  -- placeholder for actual proof
  sorry

end NUMINAMATH_GPT_pentagon_area_is_correct_l1530_153077


namespace NUMINAMATH_GPT_mike_picked_peaches_l1530_153092

def initial_peaches : ℕ := 34
def total_peaches : ℕ := 86

theorem mike_picked_peaches : total_peaches - initial_peaches = 52 :=
by
  sorry

end NUMINAMATH_GPT_mike_picked_peaches_l1530_153092


namespace NUMINAMATH_GPT_find_pairs_l1530_153074

def regions_divided (h s : ℕ) : ℕ :=
  1 + s * (s + 1) / 2 + h * (s + 1)

theorem find_pairs (h s : ℕ) :
  regions_divided h s = 1992 →
  (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l1530_153074
