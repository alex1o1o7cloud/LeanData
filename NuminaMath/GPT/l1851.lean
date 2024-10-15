import Mathlib

namespace NUMINAMATH_GPT_solution_set_of_inequality_l1851_185161

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 4 * x else (-(x^2 - 4 * x))

theorem solution_set_of_inequality :
  {x : ℝ | f (x - 2) < 5} = {x : ℝ | -3 < x ∧ x < 7} := by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1851_185161


namespace NUMINAMATH_GPT_uv_divisible_by_3_l1851_185176

theorem uv_divisible_by_3
  {u v : ℤ}
  (h : 9 ∣ (u^2 + u * v + v^2)) :
  3 ∣ u ∧ 3 ∣ v :=
sorry

end NUMINAMATH_GPT_uv_divisible_by_3_l1851_185176


namespace NUMINAMATH_GPT_lottery_sample_representativeness_l1851_185128

theorem lottery_sample_representativeness (A B C D : Prop) :
  B :=
by
  sorry

end NUMINAMATH_GPT_lottery_sample_representativeness_l1851_185128


namespace NUMINAMATH_GPT_option_c_correct_l1851_185101

theorem option_c_correct : (3 * Real.sqrt 2) ^ 2 = 18 :=
by 
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_option_c_correct_l1851_185101


namespace NUMINAMATH_GPT_choose_rectangles_l1851_185150

theorem choose_rectangles (n : ℕ) (hn : n ≥ 2) :
  ∃ (chosen_rectangles : Finset (ℕ × ℕ)), 
    (chosen_rectangles.card = 2 * n ∧
     ∀ (r1 r2 : ℕ × ℕ), r1 ∈ chosen_rectangles → r2 ∈ chosen_rectangles →
      (r1.fst ≤ r2.fst ∧ r1.snd ≤ r2.snd) ∨ 
      (r2.fst ≤ r1.fst ∧ r2.snd ≤ r1.snd) ∨ 
      (r1.fst ≤ r2.snd ∧ r1.snd ≤ r2.fst) ∨ 
      (r2.fst ≤ r1.snd ∧ r2.snd <= r1.fst)) :=
sorry

end NUMINAMATH_GPT_choose_rectangles_l1851_185150


namespace NUMINAMATH_GPT_complement_union_l1851_185138

open Set

variable (U : Set ℝ)
variable (A : Set ℝ)
variable (B : Set ℝ)

theorem complement_union (U : Set ℝ) (A : Set ℝ) (B : Set ℝ) (hU : U = univ) 
(hA : A = {x : ℝ | 0 < x}) 
(hB : B = {x : ℝ | -3 < x ∧ x < 1}) : 
compl (A ∪ B) = {x : ℝ | x ≤ -3} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l1851_185138


namespace NUMINAMATH_GPT_fraction_zero_iff_x_neg_one_l1851_185164

theorem fraction_zero_iff_x_neg_one (x : ℝ) (h₀ : x^2 - 1 = 0) (h₁ : x - 1 ≠ 0) :
  (x^2 - 1) / (x - 1) = 0 ↔ x = -1 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_iff_x_neg_one_l1851_185164


namespace NUMINAMATH_GPT_company_KW_price_percentage_l1851_185194

theorem company_KW_price_percentage
  (A B : ℝ)
  (h1 : ∀ P: ℝ, P = 1.9 * A)
  (h2 : ∀ P: ℝ, P = 2 * B) :
  Price = 131.034 / 100 * (A + B) := 
by
  sorry

end NUMINAMATH_GPT_company_KW_price_percentage_l1851_185194


namespace NUMINAMATH_GPT_seating_arrangements_correct_l1851_185139

-- Conditions
def num_children : ℕ := 3
def num_front_seats : ℕ := 2
def num_back_seats : ℕ := 3
def driver_choices : ℕ := 2

-- Function to calculate the number of arrangements
noncomputable def seating_arrangements (children : ℕ) (front_seats : ℕ) (back_seats : ℕ) (driver_choices : ℕ) : ℕ :=
  driver_choices * (children + 1) * (back_seats.factorial)

-- Problem Statement
theorem seating_arrangements_correct : 
  seating_arrangements num_children num_front_seats num_back_seats driver_choices = 48 :=
by
  -- Translate conditions to computation
  have h1: num_children = 3 := rfl
  have h2: num_front_seats = 2 := rfl
  have h3: num_back_seats = 3 := rfl
  have h4: driver_choices = 2 := rfl
  sorry

end NUMINAMATH_GPT_seating_arrangements_correct_l1851_185139


namespace NUMINAMATH_GPT_exists_An_Bn_l1851_185189

theorem exists_An_Bn (n : ℕ) : ∃ (A_n B_n : ℕ), (3 - Real.sqrt 7) ^ n = A_n - B_n * Real.sqrt 7 := by
  sorry

end NUMINAMATH_GPT_exists_An_Bn_l1851_185189


namespace NUMINAMATH_GPT_probability_factor_lt_10_l1851_185143

theorem probability_factor_lt_10 (n : ℕ) (h : n = 90) :
  (∃ factors_lt_10 : ℕ, ∃ total_factors : ℕ,
    factors_lt_10 = 7 ∧ total_factors = 12 ∧ (factors_lt_10 / total_factors : ℚ) = 7 / 12) :=
by sorry

end NUMINAMATH_GPT_probability_factor_lt_10_l1851_185143


namespace NUMINAMATH_GPT_system_of_inequalities_solution_l1851_185141

theorem system_of_inequalities_solution (p : ℝ) (h1 : 19 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 10 / 19 := by
  sorry

end NUMINAMATH_GPT_system_of_inequalities_solution_l1851_185141


namespace NUMINAMATH_GPT_rectangle_length_l1851_185168

theorem rectangle_length : 
  ∃ l b : ℝ, 
    (l = 2 * b) ∧ 
    (20 < l ∧ l < 50) ∧ 
    (10 < b ∧ b < 30) ∧ 
    ((l - 5) * (b + 5) = l * b + 75) ∧ 
    (l = 40) :=
sorry

end NUMINAMATH_GPT_rectangle_length_l1851_185168


namespace NUMINAMATH_GPT_number_of_stickers_used_to_decorate_l1851_185142

def initial_stickers : ℕ := 20
def bought_stickers : ℕ := 12
def birthday_stickers : ℕ := 20
def given_stickers : ℕ := 5
def remaining_stickers : ℕ := 39

theorem number_of_stickers_used_to_decorate :
  (initial_stickers + bought_stickers + birthday_stickers - given_stickers - remaining_stickers) = 8 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_stickers_used_to_decorate_l1851_185142


namespace NUMINAMATH_GPT_smallest_number_diminished_by_16_divisible_l1851_185145

theorem smallest_number_diminished_by_16_divisible (n : ℕ) :
  (∃ n, ∀ k ∈ [4, 6, 8, 10], (n - 16) % k = 0 ∧ n = 136) :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_diminished_by_16_divisible_l1851_185145


namespace NUMINAMATH_GPT_correct_response_percentage_l1851_185134

def number_of_students : List ℕ := [300, 1100, 100, 600, 400]
def total_students : ℕ := number_of_students.sum
def correct_response_students : ℕ := number_of_students.maximum.getD 0

theorem correct_response_percentage :
  (correct_response_students * 100 / total_students) = 44 := by
  sorry

end NUMINAMATH_GPT_correct_response_percentage_l1851_185134


namespace NUMINAMATH_GPT_rectangular_garden_length_l1851_185179

theorem rectangular_garden_length (L P B : ℕ) (h1 : P = 600) (h2 : B = 150) (h3 : P = 2 * (L + B)) : L = 150 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_garden_length_l1851_185179


namespace NUMINAMATH_GPT_extra_apples_proof_l1851_185114

def total_apples (red_apples : ℕ) (green_apples : ℕ) : ℕ :=
  red_apples + green_apples

def apples_taken_by_students (students : ℕ) : ℕ :=
  students

def extra_apples (total_apples : ℕ) (apples_taken : ℕ) : ℕ :=
  total_apples - apples_taken

theorem extra_apples_proof
  (red_apples : ℕ) (green_apples : ℕ) (students : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : students = 21) :
  extra_apples (total_apples red_apples green_apples) (apples_taken_by_students students) = 35 :=
by
  sorry

end NUMINAMATH_GPT_extra_apples_proof_l1851_185114


namespace NUMINAMATH_GPT_arcsin_sqrt_three_over_two_l1851_185185

theorem arcsin_sqrt_three_over_two :
  Real.arcsin (Real.sqrt 3 / 2) = π / 3 :=
sorry

end NUMINAMATH_GPT_arcsin_sqrt_three_over_two_l1851_185185


namespace NUMINAMATH_GPT_find_number_l1851_185163

theorem find_number :
  let f_add (a b : ℝ) : ℝ := a * b
  let f_sub (a b : ℝ) : ℝ := a + b
  let f_mul (a b : ℝ) : ℝ := a / b
  let f_div (a b : ℝ) : ℝ := a - b
  (f_div 9 8) * (f_mul 7 some_number) + (f_sub some_number 10) = 13.285714285714286 :=
  let some_number := 5
  sorry

end NUMINAMATH_GPT_find_number_l1851_185163


namespace NUMINAMATH_GPT_mike_spending_l1851_185103

noncomputable def marbles_cost : ℝ := 9.05
noncomputable def football_cost : ℝ := 4.95
noncomputable def baseball_cost : ℝ := 6.52

noncomputable def toy_car_original_cost : ℝ := 6.50
noncomputable def toy_car_discount : ℝ := 0.20
noncomputable def toy_car_discounted_cost : ℝ := toy_car_original_cost * (1 - toy_car_discount)

noncomputable def puzzle_cost : ℝ := 3.25
noncomputable def puzzle_total_cost : ℝ := puzzle_cost -- 'buy one get one free' condition

noncomputable def action_figure_original_cost : ℝ := 15.00
noncomputable def action_figure_discounted_cost : ℝ := 10.50

noncomputable def total_cost : ℝ := marbles_cost + football_cost + baseball_cost + toy_car_discounted_cost + puzzle_total_cost + action_figure_discounted_cost

theorem mike_spending : total_cost = 39.47 := by
  sorry

end NUMINAMATH_GPT_mike_spending_l1851_185103


namespace NUMINAMATH_GPT_area_of_segment_l1851_185166

theorem area_of_segment (R : ℝ) (hR : R > 0) (h_perimeter : 4 * R = 2 * R + 2 * R) :
  (1 - (1 / 2) * Real.sin 2) * R^2 = (fun R => (1 - (1 / 2) * Real.sin 2) * R^2) R :=
by
  sorry

end NUMINAMATH_GPT_area_of_segment_l1851_185166


namespace NUMINAMATH_GPT_problem_1_exists_a_problem_2_values_of_a_l1851_185124

open Set

-- Definitions for sets A, B, C
def A (a : ℝ) : Set ℝ := {x | x^2 - 2 * a * x + 4 * a^2 - 3 = 0}
def B : Set ℝ := {x | x^2 - x - 2 = 0}
def C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- Lean statements for the two problems
theorem problem_1_exists_a : ∃ a : ℝ, A a ∩ B = A a ∪ B ∧ a = 1/2 := by
  sorry

theorem problem_2_values_of_a (a : ℝ) : 
  (A a ∩ B ≠ ∅ ∧ A a ∩ C = ∅) → 
  (A a = {-1} → a = -1) ∧ (∀ x, A a = {-1, x} → x ≠ 2 → False) := 
  by sorry

end NUMINAMATH_GPT_problem_1_exists_a_problem_2_values_of_a_l1851_185124


namespace NUMINAMATH_GPT_avg_price_of_racket_l1851_185116

theorem avg_price_of_racket (total_revenue : ℝ) (pairs_sold : ℝ) (h1 : total_revenue = 686) (h2 : pairs_sold = 70) : 
  total_revenue / pairs_sold = 9.8 := by
  sorry

end NUMINAMATH_GPT_avg_price_of_racket_l1851_185116


namespace NUMINAMATH_GPT_lara_gives_betty_l1851_185104

variables (X Y : ℝ)

-- Conditions
-- Lara has spent X dollars
-- Betty has spent Y dollars
-- Y is greater than X
theorem lara_gives_betty (h : Y > X) : (Y - X) / 2 = (X + Y) / 2 - X :=
by
  sorry

end NUMINAMATH_GPT_lara_gives_betty_l1851_185104


namespace NUMINAMATH_GPT_find_a_plus_b_l1851_185140

/-- Given the sets M = {x | |x-4| + |x-1| < 5} and N = {x | a < x < 6}, and M ∩ N = {2, b}, 
prove that a + b = 7. -/
theorem find_a_plus_b 
  (M : Set ℝ := { x | |x - 4| + |x - 1| < 5 }) 
  (N : Set ℝ := { x | a < x ∧ x < 6 }) 
  (a b : ℝ)
  (h_inter : M ∩ N = {2, b}) :
  a + b = 7 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l1851_185140


namespace NUMINAMATH_GPT_abs_five_minus_e_l1851_185160

noncomputable def e : ℝ := Real.exp 1

theorem abs_five_minus_e : |5 - e| = 5 - e := by
  sorry

end NUMINAMATH_GPT_abs_five_minus_e_l1851_185160


namespace NUMINAMATH_GPT_coefficient_j_l1851_185175

theorem coefficient_j (j k : ℝ) (p : Polynomial ℝ) (h : p = Polynomial.C 400 + Polynomial.X * Polynomial.C k + Polynomial.X^2 * Polynomial.C j + Polynomial.X^4) :
  (∃ a d : ℝ, (d ≠ 0) ∧ (0 > (4*a + 6*d)) ∧ (p.eval a = 0) ∧ (p.eval (a + d) = 0) ∧ (p.eval (a + 2*d) = 0) ∧ (p.eval (a + 3*d) = 0)) → 
  j = -40 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_j_l1851_185175


namespace NUMINAMATH_GPT_rem_frac_l1851_185154

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_frac : rem (5/7 : ℚ) (3/4 : ℚ) = (5/7 : ℚ) := 
by 
  sorry

end NUMINAMATH_GPT_rem_frac_l1851_185154


namespace NUMINAMATH_GPT_correct_profit_equation_l1851_185123

def total_rooms : ℕ := 50
def initial_price : ℕ := 180
def price_increase_step : ℕ := 10
def cost_per_occupied_room : ℕ := 20
def desired_profit : ℕ := 10890

theorem correct_profit_equation (x : ℕ) : 
  (x - cost_per_occupied_room : ℤ) * (total_rooms - (x - initial_price : ℤ) / price_increase_step) = desired_profit :=
by sorry

end NUMINAMATH_GPT_correct_profit_equation_l1851_185123


namespace NUMINAMATH_GPT_college_girls_count_l1851_185170

theorem college_girls_count 
  (B G : ℕ)
  (h1 : B / G = 8 / 5)
  (h2 : B + G = 455) : 
  G = 175 := 
sorry

end NUMINAMATH_GPT_college_girls_count_l1851_185170


namespace NUMINAMATH_GPT_no_isosceles_triangle_exists_l1851_185188

-- Define the grid size
def grid_size : ℕ := 5

-- Define points A and B such that AB is three units horizontally
structure Point where
  x : ℕ
  y : ℕ

-- Define specific points A and B
def A : Point := ⟨2, 2⟩
def B : Point := ⟨5, 2⟩

-- Define a function to check if a triangle is isosceles
def is_isosceles (p1 p2 p3 : Point) : Prop :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p1.x - p3.x)^2 + (p1.y - p3.y)^2 ∨
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2 ∨
  (p1.x - p3.x)^2 + (p1.y - p3.y)^2 = (p2.x - p3.x)^2 + (p2.y - p3.y)^2

-- Prove that there are no points C that make triangle ABC isosceles
theorem no_isosceles_triangle_exists :
  ¬ ∃ C : Point, C.x ≤ grid_size ∧ C.y ≤ grid_size ∧ is_isosceles A B C :=
by
  sorry

end NUMINAMATH_GPT_no_isosceles_triangle_exists_l1851_185188


namespace NUMINAMATH_GPT_cost_sum_in_WD_l1851_185149

def watch_cost_loss (W : ℝ) : ℝ := 0.9 * W
def watch_cost_gain (W : ℝ) : ℝ := 1.04 * W
def bracelet_cost_gain (B : ℝ) : ℝ := 1.08 * B
def bracelet_cost_reduced_gain (B : ℝ) : ℝ := 1.02 * B

theorem cost_sum_in_WD :
  ∃ W B : ℝ, 
    watch_cost_loss W + 196 = watch_cost_gain W ∧ 
    bracelet_cost_gain B - 100 = bracelet_cost_reduced_gain B ∧ 
    (W + B / 1.5 = 2511.11) :=
sorry

end NUMINAMATH_GPT_cost_sum_in_WD_l1851_185149


namespace NUMINAMATH_GPT_difference_of_two_numbers_l1851_185129

theorem difference_of_two_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x * y = 23) : |x - y| = 22 :=
sorry

end NUMINAMATH_GPT_difference_of_two_numbers_l1851_185129


namespace NUMINAMATH_GPT_tangent_line_eq_l1851_185113

def f (x : ℝ) : ℝ := x^3 + 4 * x + 5

theorem tangent_line_eq (x y : ℝ) (h : (x, y) = (1, 10)) : 
  (7 * x - y + 3 = 0) :=
sorry

end NUMINAMATH_GPT_tangent_line_eq_l1851_185113


namespace NUMINAMATH_GPT_cost_price_equivalence_l1851_185132

theorem cost_price_equivalence (list_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) (cost_price : ℝ) :
  list_price = 132 → discount_rate = 0.1 → profit_rate = 0.1 → 
  (list_price * (1 - discount_rate)) = cost_price * (1 + profit_rate) →
  cost_price = 108 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cost_price_equivalence_l1851_185132


namespace NUMINAMATH_GPT_solve_for_a_l1851_185118

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem solve_for_a (a : ℝ) (f : ℝ → ℝ) (h_def : ∀ x, f x = x / ((x + 1) * (x - a)))
  (h_odd : is_odd_function f) :
  a = 1 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l1851_185118


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1851_185167

theorem sufficient_but_not_necessary (x : ℝ) : (x < -1) → (x < -1 ∨ x > 1) ∧ ¬(∀ y : ℝ, (x < -1 ∨ y > 1) → (y < -1)) :=
by
  -- This means we would prove that if x < -1, then x < -1 ∨ x > 1 holds (sufficient),
  -- and show that there is a case (x > 1) where x < -1 is not necessary for x < -1 ∨ x > 1. 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1851_185167


namespace NUMINAMATH_GPT_monotonicity_intervals_max_m_value_l1851_185155

noncomputable def f (x : ℝ) : ℝ :=  (3 / 2) * x^2 - 3 * Real.log x

theorem monotonicity_intervals :
  (∀ x > (1:ℝ), ∃ ε > (0:ℝ), ∀ y, x < y → y < x + ε → f x < f y)
  ∧ (∀ x, (0:ℝ) < x → x < (1:ℝ) → ∃ ε > (0:ℝ), ∀ y, x - ε < y → y < x → f y < f x) :=
by sorry

theorem max_m_value (m : ℤ) (h : ∀ x > (1:ℝ), f (x * Real.log x + 2 * x - 1) > f (↑m * (x - 1))) :
  m ≤ 4 :=
by sorry

end NUMINAMATH_GPT_monotonicity_intervals_max_m_value_l1851_185155


namespace NUMINAMATH_GPT_solve_for_x_l1851_185178

theorem solve_for_x (x y : ℚ) (h1 : x - y = 8) (h2 : x + 2 * y = 10) : x = 26 / 3 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1851_185178


namespace NUMINAMATH_GPT_smallest_d_for_inverse_g_l1851_185197

def g (x : ℝ) := (x - 3)^2 - 8

theorem smallest_d_for_inverse_g : ∃ d : ℝ, (∀ x y : ℝ, x ≠ y → x ≥ d → y ≥ d → g x ≠ g y) ∧ ∀ d' : ℝ, d' < 3 → ∃ x y : ℝ, x ≠ y ∧ x ≥ d' ∧ y ≥ d' ∧ g x = g y :=
by
  sorry

end NUMINAMATH_GPT_smallest_d_for_inverse_g_l1851_185197


namespace NUMINAMATH_GPT_tan_half_sum_of_angles_l1851_185153

theorem tan_half_sum_of_angles (p q : ℝ) 
    (h1 : Real.cos p + Real.cos q = 3 / 5) 
    (h2 : Real.sin p + Real.sin q = 1 / 4) :
    Real.tan ((p + q) / 2) = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_tan_half_sum_of_angles_l1851_185153


namespace NUMINAMATH_GPT_cost_of_balls_max_basketball_count_l1851_185190

-- Define the prices of basketball and soccer ball
variables (x y : ℕ)

-- Define the conditions given in the problem
def condition1 : Prop := 2 * x + 3 * y = 310
def condition2 : Prop := 5 * x + 2 * y = 500

-- Proving the cost of each basketball and soccer ball
theorem cost_of_balls (h1 : condition1 x y) (h2 : condition2 x y) : x = 80 ∧ y = 50 :=
sorry

-- Define the total number of balls and the inequality constraint
variable (m : ℕ)
def total_balls_condition : Prop := m + (60 - m) = 60
def cost_constraint : Prop := 80 * m + 50 * (60 - m) ≤ 4000

-- Proving the maximum number of basketballs
theorem max_basketball_count (hc : cost_constraint m) (ht : total_balls_condition m) : m ≤ 33 :=
sorry

end NUMINAMATH_GPT_cost_of_balls_max_basketball_count_l1851_185190


namespace NUMINAMATH_GPT_max_points_on_poly_graph_l1851_185107

theorem max_points_on_poly_graph (P : Polynomial ℤ) (h_deg : P.degree = 20):
  ∃ (S : Finset (ℤ × ℤ)), (∀ p ∈ S, 0 ≤ p.snd ∧ p.snd ≤ 10) ∧ S.card ≤ 20 ∧ 
  ∀ S' : Finset (ℤ × ℤ), (∀ p ∈ S', 0 ≤ p.snd ∧ p.snd ≤ 10) → S'.card ≤ 20 :=
by
  sorry

end NUMINAMATH_GPT_max_points_on_poly_graph_l1851_185107


namespace NUMINAMATH_GPT_simplify_expression_to_inverse_abc_l1851_185148

variable (a b c : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem simplify_expression_to_inverse_abc :
  (a + b + c + 3)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ca + 3)⁻¹ * ((ab)⁻¹ + (bc)⁻¹ + (ca)⁻¹ + 3) = (1 : ℝ) / (abc) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_to_inverse_abc_l1851_185148


namespace NUMINAMATH_GPT_order_of_three_numbers_l1851_185157

theorem order_of_three_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a * b) / (a + b) ≤ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≤ (a + b) / 2 :=
by
  sorry

end NUMINAMATH_GPT_order_of_three_numbers_l1851_185157


namespace NUMINAMATH_GPT_fraction_increase_l1851_185117

-- Define the problem conditions and the proof statement
theorem fraction_increase (m n : ℤ) (hnz : n ≠ 0) (hnnz : n ≠ -1) (h : m < n) :
  (m : ℚ) / n < (m + 1 : ℚ) / (n + 1) :=
by sorry

end NUMINAMATH_GPT_fraction_increase_l1851_185117


namespace NUMINAMATH_GPT_chairs_removal_correct_chairs_removal_l1851_185105

theorem chairs_removal (initial_chairs : ℕ) (chairs_per_row : ℕ) (participants : ℕ) : ℕ :=
  let total_chairs := 169
  let per_row := 13
  let attendees := 95
  let needed_chairs := (attendees + per_row - 1) / per_row * per_row
  let chairs_to_remove := total_chairs - needed_chairs
  chairs_to_remove

theorem correct_chairs_removal : chairs_removal 169 13 95 = 65 :=
by
  sorry

end NUMINAMATH_GPT_chairs_removal_correct_chairs_removal_l1851_185105


namespace NUMINAMATH_GPT_pow_sum_ge_mul_l1851_185172

theorem pow_sum_ge_mul (m n : ℕ) : 2^(m + n - 2) ≥ m * n := 
sorry

end NUMINAMATH_GPT_pow_sum_ge_mul_l1851_185172


namespace NUMINAMATH_GPT_cubic_roots_arithmetic_progression_l1851_185159

theorem cubic_roots_arithmetic_progression (a b c : ℝ) :
  (∃ x : ℝ, x^3 + a * x^2 + b * x + c = 0) ∧ 
  (∀ x : ℝ, x^3 + a * x^2 + b * x + c = 0 → 
    (x = p - t ∨ x = p ∨ x = p + t) ∧ 
    (a ≠ 0)) ↔ 
  ((a * b / 3) - 2 * (a^3) / 27 - c = 0 ∧ (a^3 / 3) - b ≥ 0) := 
by sorry

end NUMINAMATH_GPT_cubic_roots_arithmetic_progression_l1851_185159


namespace NUMINAMATH_GPT_exists_four_consecutive_with_square_divisors_l1851_185111

theorem exists_four_consecutive_with_square_divisors :
  ∃ n : ℕ, n = 3624 ∧
  (∃ d1, d1^2 > 1 ∧ d1^2 ∣ n) ∧ 
  (∃ d2, d2^2 > 1 ∧ d2^2 ∣ (n + 1)) ∧ 
  (∃ d3, d3^2 > 1 ∧ d3^2 ∣ (n + 2)) ∧ 
  (∃ d4, d4^2 > 1 ∧ d4^2 ∣ (n + 3)) :=
sorry

end NUMINAMATH_GPT_exists_four_consecutive_with_square_divisors_l1851_185111


namespace NUMINAMATH_GPT_average_height_of_females_at_school_l1851_185106

-- Define the known quantities and conditions
variable (total_avg_height male_avg_height female_avg_height : ℝ)
variable (male_count female_count : ℕ)

-- Given conditions
def conditions :=
  total_avg_height = 180 ∧ 
  male_avg_height = 185 ∧ 
  male_count = 2 * female_count ∧
  (male_count + female_count) * total_avg_height = male_count * male_avg_height + female_count * female_avg_height

-- The theorem we want to prove
theorem average_height_of_females_at_school (total_avg_height male_avg_height female_avg_height : ℝ)
    (male_count female_count : ℕ) (h : conditions total_avg_height male_avg_height female_avg_height male_count female_count) :
    female_avg_height = 170 :=
  sorry

end NUMINAMATH_GPT_average_height_of_females_at_school_l1851_185106


namespace NUMINAMATH_GPT_volume_of_pure_water_added_l1851_185146

theorem volume_of_pure_water_added 
  (V0 : ℝ) (P0 : ℝ) (Pf : ℝ) 
  (V0_eq : V0 = 50) 
  (P0_eq : P0 = 0.30) 
  (Pf_eq : Pf = 0.1875) : 
  ∃ V : ℝ, V = 30 ∧ (15 / (V0 + V)) = Pf := 
by
  sorry

end NUMINAMATH_GPT_volume_of_pure_water_added_l1851_185146


namespace NUMINAMATH_GPT_find_other_factor_l1851_185133

theorem find_other_factor 
    (w : ℕ) 
    (hw_pos : w > 0) 
    (h_factor : ∃ (x y : ℕ), 936 * w = x * y ∧ (2 ^ 5 ∣ x) ∧ (3 ^ 3 ∣ x)) 
    (h_ww : w = 156) : 
    ∃ (other_factor : ℕ), 936 * w = 156 * other_factor ∧ other_factor = 72 := 
by 
    sorry

end NUMINAMATH_GPT_find_other_factor_l1851_185133


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l1851_185151

theorem volume_of_rectangular_prism {l w h : ℝ} 
  (h1 : l * w = 12) 
  (h2 : w * h = 18) 
  (h3 : l * h = 24) : 
  l * w * h = 72 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l1851_185151


namespace NUMINAMATH_GPT_area_ratio_independent_l1851_185108

-- Definitions related to the problem
variables (AB BC CD : ℝ) (e f g : ℝ)

-- Let the lengths be defined as follows
def AB_def : Prop := AB = 2 * e
def BC_def : Prop := BC = 2 * f
def CD_def : Prop := CD = 2 * g

-- Let the areas be defined as follows
def area_quadrilateral (e f g : ℝ) : ℝ :=
  2 * (e + f) * (f + g)

def area_enclosed (e f g : ℝ) : ℝ :=
  (e + f + g) ^ 2 + f ^ 2 - e ^ 2 - g ^ 2

-- Prove the ratio is 2 / π
theorem area_ratio_independent (e f g : ℝ) (h1 : AB_def AB e)
  (h2 : BC_def BC f) (h3 : CD_def CD g) :
  (area_quadrilateral e f g) / ((area_enclosed e f g) * (π / 2)) = 2 / π :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_independent_l1851_185108


namespace NUMINAMATH_GPT_remainder_17_pow_63_mod_7_l1851_185183

theorem remainder_17_pow_63_mod_7 : (17 ^ 63) % 7 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_17_pow_63_mod_7_l1851_185183


namespace NUMINAMATH_GPT_find_B_l1851_185125

variable {U : Set ℕ}

def A : Set ℕ := {1, 3, 5, 7}
def complement_A : Set ℕ := {2, 4, 6}
def complement_B : Set ℕ := {1, 4, 6}
def B : Set ℕ := {2, 3, 5, 7}

theorem find_B
  (hU : U = A ∪ complement_A)
  (A_comp : ∀ x, x ∈ complement_A ↔ x ∉ A)
  (B_comp : ∀ x, x ∈ complement_B ↔ x ∉ B) :
  B = {2, 3, 5, 7} :=
sorry

end NUMINAMATH_GPT_find_B_l1851_185125


namespace NUMINAMATH_GPT_dogs_not_doing_anything_l1851_185199

def total_dogs : ℕ := 500
def dogs_running : ℕ := 18 * total_dogs / 100
def dogs_playing_with_toys : ℕ := (3 * total_dogs) / 20
def dogs_barking : ℕ := 7 * total_dogs / 100
def dogs_digging_holes : ℕ := total_dogs / 10
def dogs_competing : ℕ := 12
def dogs_sleeping : ℕ := (2 * total_dogs) / 25
def dogs_eating_treats : ℕ := total_dogs / 5

def dogs_doing_anything : ℕ := dogs_running + dogs_playing_with_toys + dogs_barking + dogs_digging_holes + dogs_competing + dogs_sleeping + dogs_eating_treats

theorem dogs_not_doing_anything : total_dogs - dogs_doing_anything = 98 :=
by
  -- proof steps would go here
  sorry

end NUMINAMATH_GPT_dogs_not_doing_anything_l1851_185199


namespace NUMINAMATH_GPT_lcm_3_4_6_15_l1851_185119

noncomputable def lcm_is_60 : ℕ := 60

theorem lcm_3_4_6_15 : lcm (lcm (lcm 3 4) 6) 15 = lcm_is_60 := 
by 
    sorry

end NUMINAMATH_GPT_lcm_3_4_6_15_l1851_185119


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_zero_l1851_185158

noncomputable def arithmetic_sequence_sum (S : ℕ → ℤ) : Prop :=
S 20 = S 40

theorem sum_of_arithmetic_sequence_zero {S : ℕ → ℤ} (h : arithmetic_sequence_sum S) : 
  S 60 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_zero_l1851_185158


namespace NUMINAMATH_GPT_original_fraction_l1851_185162

theorem original_fraction (x y : ℝ) (hxy : x / y = 5 / 7)
  (hx : 1.20 * x / (0.90 * y) = 20 / 21) : x / y = 5 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_original_fraction_l1851_185162


namespace NUMINAMATH_GPT_workshop_personnel_l1851_185115

-- Definitions for workshops with their corresponding production constraints
def workshopA_production (x : ℕ) : ℕ := 6 + 11 * (x - 1)
def workshopB_production (y : ℕ) : ℕ := 7 + 10 * (y - 1)

-- The main theorem to be proved
theorem workshop_personnel :
  ∃ (x y : ℕ), workshopA_production x = workshopB_production y ∧
               100 ≤ workshopA_production x ∧ workshopA_production x ≤ 200 ∧
               x = 12 ∧ y = 13 :=
by
  sorry

end NUMINAMATH_GPT_workshop_personnel_l1851_185115


namespace NUMINAMATH_GPT_vector_addition_example_l1851_185127

noncomputable def OA : ℝ × ℝ := (-2, 3)
noncomputable def AB : ℝ × ℝ := (-1, -4)
noncomputable def OB : ℝ × ℝ := (OA.1 + AB.1, OA.2 + AB.2)

theorem vector_addition_example :
  OB = (-3, -1) :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_example_l1851_185127


namespace NUMINAMATH_GPT_exponent_power_rule_l1851_185136

theorem exponent_power_rule (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 :=
by sorry

end NUMINAMATH_GPT_exponent_power_rule_l1851_185136


namespace NUMINAMATH_GPT_greatest_integer_x_l1851_185131

theorem greatest_integer_x (x : ℤ) : 
  (∀ x : ℤ, (8 / 11 : ℝ) > (x / 17) → x ≤ 12) ∧ (8 / 11 : ℝ) > (12 / 17) :=
sorry

end NUMINAMATH_GPT_greatest_integer_x_l1851_185131


namespace NUMINAMATH_GPT_jeremy_can_win_in_4_turns_l1851_185156

noncomputable def game_winnable_in_4_turns (left right : ℕ) : Prop :=
∃ n1 n2 n3 n4 : ℕ,
  n1 > 0 ∧ n2 > 0 ∧ n3 > 0 ∧ n4 > 0 ∧
  (left + n1 + n2 + n3 + n4 = right * n1 * n2 * n3 * n4)

theorem jeremy_can_win_in_4_turns (left right : ℕ) (hleft : left = 17) (hright : right = 5) : game_winnable_in_4_turns left right :=
by
  rw [hleft, hright]
  sorry

end NUMINAMATH_GPT_jeremy_can_win_in_4_turns_l1851_185156


namespace NUMINAMATH_GPT_bed_height_l1851_185137

noncomputable def bed_length : ℝ := 8
noncomputable def bed_width : ℝ := 4
noncomputable def bags_of_soil : ℕ := 16
noncomputable def soil_per_bag : ℝ := 4
noncomputable def total_volume_of_soil : ℝ := bags_of_soil * soil_per_bag
noncomputable def number_of_beds : ℕ := 2
noncomputable def volume_per_bed : ℝ := total_volume_of_soil / number_of_beds

theorem bed_height :
  volume_per_bed / (bed_length * bed_width) = 1 :=
sorry

end NUMINAMATH_GPT_bed_height_l1851_185137


namespace NUMINAMATH_GPT_gdp_scientific_notation_l1851_185182

theorem gdp_scientific_notation :
  (121 * 10^12 : ℝ) = 1.21 * 10^14 := by
  sorry

end NUMINAMATH_GPT_gdp_scientific_notation_l1851_185182


namespace NUMINAMATH_GPT_find_first_remainder_l1851_185122

theorem find_first_remainder (N : ℕ) (R₁ R₂ : ℕ) (h1 : N = 184) (h2 : N % 15 = R₂) (h3 : R₂ = 4) : 
  N % 13 = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_first_remainder_l1851_185122


namespace NUMINAMATH_GPT_right_triangle_area_l1851_185112

theorem right_triangle_area (a b : ℝ) (h : a^2 - 7 * a + 12 = 0 ∧ b^2 - 7 * b + 12 = 0) : 
  ∃ A : ℝ, (A = 6 ∨ A = 3 * (Real.sqrt 7 / 2)) ∧ A = 1 / 2 * a * b := 
by 
  sorry

end NUMINAMATH_GPT_right_triangle_area_l1851_185112


namespace NUMINAMATH_GPT_probability_black_pen_l1851_185187

-- Define the total number of pens and the specific counts
def total_pens : ℕ := 5 + 6 + 7
def green_pens : ℕ := 5
def black_pens : ℕ := 6
def red_pens : ℕ := 7

-- Define the probability calculation
def probability (total : ℕ) (count : ℕ) : ℚ := count / total

-- State the theorem
theorem probability_black_pen :
  probability total_pens black_pens = 1 / 3 :=
by sorry

end NUMINAMATH_GPT_probability_black_pen_l1851_185187


namespace NUMINAMATH_GPT_triangle_BX_in_terms_of_sides_l1851_185121

-- Define the triangle with angles and points
variables {A B C : ℝ}
variables {AB AC BC : ℝ}
variables (X Y : ℝ) (AZ : ℝ)

-- Add conditions as assumptions
variables (angle_A_bisector : 2 * A = (B + C)) -- AZ is the angle bisector of angle A
variables (angle_B_lt_C : B < C) -- angle B < angle C
variables (point_XY : X / AB = Y / AC ∧ X = Y) -- BX = CY and angles BZX = CZY

-- Define the statement to be proved
theorem triangle_BX_in_terms_of_sides :
    BX = CY →
    (AZ < 1 ∧ AZ > 0) →
    A + B + C = π → 
    BX = (BC * BC) / (AB + AC) :=
sorry

end NUMINAMATH_GPT_triangle_BX_in_terms_of_sides_l1851_185121


namespace NUMINAMATH_GPT_hydrogen_atoms_in_compound_l1851_185192

theorem hydrogen_atoms_in_compound :
  ∀ (molecular_weight_of_compound atomic_weight_Al atomic_weight_O atomic_weight_H : ℕ)
    (num_Al num_O num_H : ℕ),
    molecular_weight_of_compound = 78 →
    atomic_weight_Al = 27 →
    atomic_weight_O = 16 →
    atomic_weight_H = 1 →
    num_Al = 1 →
    num_O = 3 →
    molecular_weight_of_compound = 
      (num_Al * atomic_weight_Al) + (num_O * atomic_weight_O) + (num_H * atomic_weight_H) →
    num_H = 3 := by
  intros
  sorry

end NUMINAMATH_GPT_hydrogen_atoms_in_compound_l1851_185192


namespace NUMINAMATH_GPT_joseph_investment_after_two_years_l1851_185173

noncomputable def initial_investment : ℝ := 1000
noncomputable def monthly_addition : ℝ := 100
noncomputable def yearly_interest_rate : ℝ := 0.10
noncomputable def time_in_years : ℕ := 2

theorem joseph_investment_after_two_years :
  let first_year_total := initial_investment + 12 * monthly_addition
  let first_year_interest := first_year_total * yearly_interest_rate
  let end_of_first_year_total := first_year_total + first_year_interest
  let second_year_total := end_of_first_year_total + 12 * monthly_addition
  let second_year_interest := second_year_total * yearly_interest_rate
  let end_of_second_year_total := second_year_total + second_year_interest
  end_of_second_year_total = 3982 := 
by
  sorry

end NUMINAMATH_GPT_joseph_investment_after_two_years_l1851_185173


namespace NUMINAMATH_GPT_fraction_multiplier_l1851_185198

theorem fraction_multiplier (x y : ℝ) :
  (3 * x * 3 * y) / (3 * x + 3 * y) = 3 * (x * y) / (x + y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_multiplier_l1851_185198


namespace NUMINAMATH_GPT_sin_330_l1851_185180

theorem sin_330 : Real.sin (330 * Real.pi / 180) = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_330_l1851_185180


namespace NUMINAMATH_GPT_average_distance_run_l1851_185110

theorem average_distance_run :
  let mickey_lap := 250
  let johnny_lap := 300
  let alex_lap := 275
  let lea_lap := 280
  let johnny_times := 8
  let lea_times := 5
  let mickey_times := johnny_times / 2
  let alex_times := mickey_times + 1 + 2 * lea_times
  let total_distance := johnny_times * johnny_lap + mickey_times * mickey_lap + lea_times * lea_lap + alex_times * alex_lap
  let number_of_participants := 4
  let avg_distance := total_distance / number_of_participants
  avg_distance = 2231.25 := by
  sorry

end NUMINAMATH_GPT_average_distance_run_l1851_185110


namespace NUMINAMATH_GPT_earnings_from_jam_l1851_185196

def betty_strawberries : ℕ := 16
def matthew_additional_strawberries : ℕ := 20
def jar_strawberries : ℕ := 7
def jar_price : ℕ := 4

theorem earnings_from_jam :
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  total_money = 40 :=
by
  let matthew_strawberries := betty_strawberries + matthew_additional_strawberries
  let natalie_strawberries := matthew_strawberries / 2
  let total_strawberries := betty_strawberries + matthew_strawberries + natalie_strawberries
  let total_jars := total_strawberries / jar_strawberries
  let total_money := total_jars * jar_price
  show total_money = 40
  sorry

end NUMINAMATH_GPT_earnings_from_jam_l1851_185196


namespace NUMINAMATH_GPT_prob_not_all_same_correct_l1851_185181

-- Define the number of dice and the number of sides per die
def num_dice : ℕ := 5
def sides_per_die : ℕ := 8

-- Total possible outcomes when rolling num_dice dice with sides_per_die sides each
def total_outcomes : ℕ := sides_per_die ^ num_dice

-- Outcomes where all dice show the same number
def same_number_outcomes : ℕ := sides_per_die

-- Probability that all five dice show the same number
def prob_same : ℚ := same_number_outcomes / total_outcomes

-- Probability that not all five dice show the same number
def prob_not_same : ℚ := 1 - prob_same

-- The expected probability that not all dice show the same number
def expected_prob_not_same : ℚ := 4095 / 4096

-- The main theorem to prove
theorem prob_not_all_same_correct : prob_not_same = expected_prob_not_same := by
  sorry  -- Proof will be filled in by the user

end NUMINAMATH_GPT_prob_not_all_same_correct_l1851_185181


namespace NUMINAMATH_GPT_team_total_points_l1851_185152

theorem team_total_points (three_points_goals: ℕ) (two_points_goals: ℕ) (half_of_total: ℕ) 
  (h1 : three_points_goals = 5) 
  (h2 : two_points_goals = 10) 
  (h3 : half_of_total = (3 * three_points_goals + 2 * two_points_goals) / 2) 
  : 2 * half_of_total = 70 := 
by 
  -- proof to be filled
  sorry

end NUMINAMATH_GPT_team_total_points_l1851_185152


namespace NUMINAMATH_GPT_trimino_tilings_greater_l1851_185193

noncomputable def trimino_tilings (n : ℕ) : ℕ := sorry
noncomputable def domino_tilings (n : ℕ) : ℕ := sorry

theorem trimino_tilings_greater (n : ℕ) (h : n > 1) : trimino_tilings (3 * n) > domino_tilings (2 * n) :=
sorry

end NUMINAMATH_GPT_trimino_tilings_greater_l1851_185193


namespace NUMINAMATH_GPT_ratio_surface_area_volume_l1851_185165

theorem ratio_surface_area_volume (a b : ℕ) (h1 : a^3 = 6 * b^2) (h2 : 6 * a^2 = 6 * b) : 
  (6 * a^2) / (b^3) = 7776 :=
by
  sorry

end NUMINAMATH_GPT_ratio_surface_area_volume_l1851_185165


namespace NUMINAMATH_GPT_geometric_seq_a7_l1851_185169

-- Definitions for the geometric sequence and conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

variables {a : ℕ → ℝ}
axiom a1 : a 1 = 2
axiom a3 : a 3 = 4
axiom geom_seq : geometric_sequence a

-- Statement to prove
theorem geometric_seq_a7 : a 7 = 16 :=
by
  -- proof will be filled in here
  sorry

end NUMINAMATH_GPT_geometric_seq_a7_l1851_185169


namespace NUMINAMATH_GPT_totalPearsPicked_l1851_185177

-- Define the number of pears picked by each individual
def jasonPears : ℕ := 46
def keithPears : ℕ := 47
def mikePears : ℕ := 12

-- State the theorem to prove the total number of pears picked
theorem totalPearsPicked : jasonPears + keithPears + mikePears = 105 := 
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_totalPearsPicked_l1851_185177


namespace NUMINAMATH_GPT_product_of_invertible_labels_l1851_185171

def f1 (x : ℤ) : ℤ := x^3 - 2 * x
def f2 (x : ℤ) : ℤ := x - 2
def f3 (x : ℤ) : ℤ := 2 - x

theorem product_of_invertible_labels :
  (¬ ∃ inv : ℤ → ℤ, f1 (inv 0) = 0 ∧ ∀ x : ℤ, f1 (inv (f1 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f2 (inv 0) = 0 ∧ ∀ x : ℤ, f2 (inv (f2 x)) = x) ∧
  (∃ inv : ℤ → ℤ, f3 (inv 0) = 0 ∧ ∀ x : ℤ, f3 (inv (f3 x)) = x) →
  (2 * 3 = 6) :=
by sorry

end NUMINAMATH_GPT_product_of_invertible_labels_l1851_185171


namespace NUMINAMATH_GPT_marketing_survey_l1851_185186

theorem marketing_survey
  (H_neither : Nat := 80)
  (H_only_A : Nat := 60)
  (H_ratio_Both_to_Only_B : Nat := 3)
  (H_both : Nat := 25) :
  H_neither + H_only_A + (H_ratio_Both_to_Only_B * H_both) + H_both = 240 := 
sorry

end NUMINAMATH_GPT_marketing_survey_l1851_185186


namespace NUMINAMATH_GPT_range_of_a_l1851_185130

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, 0 < x ∧ (1 / 4)^x + (1 / 2)^(x - 1) + a = 0) →
  (-3 < a ∧ a < 0) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1851_185130


namespace NUMINAMATH_GPT_no_grasshopper_at_fourth_vertex_l1851_185191

-- Definitions based on given conditions
def is_vertex_of_square (x : ℝ) (y : ℝ) : Prop :=
  (x = 0 ∨ x = 1) ∧ (y = 0 ∨ y = 1)

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1) ^ 2 + (a.2 - b.2) ^ 2

def leapfrog_jump (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * b.1 - a.1, 2 * b.2 - a.2)

-- Problem statement
theorem no_grasshopper_at_fourth_vertex (a b c : ℝ × ℝ) :
  is_vertex_of_square a.1 a.2 ∧ is_vertex_of_square b.1 b.2 ∧ is_vertex_of_square c.1 c.2 →
  ∃ d : ℝ × ℝ, is_vertex_of_square d.1 d.2 ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c →
  ∀ (n : ℕ) (pos : ℕ → ℝ × ℝ → ℝ × ℝ → ℝ × ℝ), (pos 0 a b = leapfrog_jump a b) ∧
    (pos n a b = leapfrog_jump (pos (n-1) a b) (pos (n-1) b c)) →
    (pos n a b).1 ≠ (d.1) ∨ (pos n a b).2 ≠ (d.2) :=
sorry

end NUMINAMATH_GPT_no_grasshopper_at_fourth_vertex_l1851_185191


namespace NUMINAMATH_GPT_num_students_taking_music_l1851_185100

-- Definitions based on given conditions
def total_students : ℕ := 500
def students_taking_art : ℕ := 20
def students_taking_both_music_and_art : ℕ := 10
def students_taking_neither_music_nor_art : ℕ := 450

-- Theorem statement to prove the number of students taking music
theorem num_students_taking_music :
  ∃ (M : ℕ), M = 40 ∧ 
  (total_students - students_taking_neither_music_nor_art = M + students_taking_art - students_taking_both_music_and_art) := 
by
  sorry

end NUMINAMATH_GPT_num_students_taking_music_l1851_185100


namespace NUMINAMATH_GPT_value_of_y_l1851_185195

theorem value_of_y (x y : ℤ) (h1 : 1.5 * (x : ℝ) = 0.25 * (y : ℝ)) (h2 : x = 24) : y = 144 :=
  sorry

end NUMINAMATH_GPT_value_of_y_l1851_185195


namespace NUMINAMATH_GPT_lcm_36_100_eq_900_l1851_185147

/-- Definition for the prime factorization of 36 -/
def factorization_36 : Prop := 36 = 2^2 * 3^2

/-- Definition for the prime factorization of 100 -/
def factorization_100 : Prop := 100 = 2^2 * 5^2

/-- The least common multiple problem statement -/
theorem lcm_36_100_eq_900 (h₁ : factorization_36) (h₂ : factorization_100) : Nat.lcm 36 100 = 900 := 
by
  sorry

end NUMINAMATH_GPT_lcm_36_100_eq_900_l1851_185147


namespace NUMINAMATH_GPT_mat_weavers_proof_l1851_185120

def mat_weavers_rate
  (num_weavers_1 : ℕ) (num_mats_1 : ℕ) (num_days_1 : ℕ)
  (num_mats_2 : ℕ) (num_days_2 : ℕ) : ℕ :=
  let rate_per_weaver_per_day := num_mats_1 / (num_weavers_1 * num_days_1)
  let num_weavers_2 := num_mats_2 / (rate_per_weaver_per_day * num_days_2)
  num_weavers_2

theorem mat_weavers_proof :
  mat_weavers_rate 4 4 4 36 12 = 12 := by
  sorry

end NUMINAMATH_GPT_mat_weavers_proof_l1851_185120


namespace NUMINAMATH_GPT_factory_X_bulbs_percentage_l1851_185102

theorem factory_X_bulbs_percentage (p : ℝ) (hx : 0.59 * p + 0.65 * (1 - p) = 0.62) : p = 0.5 :=
sorry

end NUMINAMATH_GPT_factory_X_bulbs_percentage_l1851_185102


namespace NUMINAMATH_GPT_unique_real_solution_k_l1851_185174

-- Definitions corresponding to problem conditions:
def is_real_solution (a b k : ℤ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ (∃ (x y : ℝ), x * x = a - 1 ∧ y * y = b - 1 ∧ x + y = Real.sqrt (a * b + k))

-- Theorem statement:
theorem unique_real_solution_k (k : ℤ) : (∀ a b : ℤ, is_real_solution a b k → (a = 2 ∧ b = 2)) ↔ k = 0 :=
sorry

end NUMINAMATH_GPT_unique_real_solution_k_l1851_185174


namespace NUMINAMATH_GPT_div_neg_rev_l1851_185126

theorem div_neg_rev (a b : ℝ) (h : a > b) : (a / -3) < (b / -3) :=
by
  sorry

end NUMINAMATH_GPT_div_neg_rev_l1851_185126


namespace NUMINAMATH_GPT_question_one_question_two_l1851_185135

variable (b x : ℝ)
def f (x : ℝ) : ℝ := x^2 - b * x + 3

theorem question_one (h : f b 0 = f b 4) : ∃ x1 x2 : ℝ, f b x1 = 0 ∧ f b x2 = 0 ∧ (x1 = 3 ∧ x2 = 1) ∨ (x1 = 1 ∧ x2 = 3) := by 
  sorry

theorem question_two (h1 : ∃ x1 x2 : ℝ, x1 > 1 ∧ x2 < 1 ∧ f b x1 = 0 ∧ f b x2 = 0) : b > 4 := by
  sorry

end NUMINAMATH_GPT_question_one_question_two_l1851_185135


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1851_185184

theorem sufficient_not_necessary_condition (x : ℝ) : x - 1 > 0 → (x > 2) ∧ (¬ (x - 1 > 0 → x > 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1851_185184


namespace NUMINAMATH_GPT_min_value_f_l1851_185109

noncomputable def f (x : Fin 5 → ℝ) : ℝ :=
  (x 0 + x 2) / (x 4 + 2 * x 1 + 3 * x 3) +
  (x 1 + x 3) / (x 0 + 2 * x 2 + 3 * x 4) +
  (x 2 + x 4) / (x 1 + 2 * x 3 + 3 * x 0) +
  (x 3 + x 0) / (x 2 + 2 * x 4 + 3 * x 1) +
  (x 4 + x 1) / (x 3 + 2 * x 0 + 3 * x 2)

def min_f (x : Fin 5 → ℝ) : Prop :=
  (∀ i, 0 < x i) → f x = 5 / 3

theorem min_value_f : ∀ x : Fin 5 → ℝ, min_f x :=
by
  intros
  sorry

end NUMINAMATH_GPT_min_value_f_l1851_185109


namespace NUMINAMATH_GPT_find_e_l1851_185144

noncomputable def f (x : ℝ) (c : ℝ) := 5 * x + 2 * c

noncomputable def g (x : ℝ) (c : ℝ) := c * x^2 + 3

noncomputable def fg (x : ℝ) (c : ℝ) := f (g x c) c

theorem find_e (c : ℝ) (e : ℝ) (h1 : f (g x c) c = 15 * x^2 + e) (h2 : 5 * c = 15) : e = 21 :=
by
  sorry

end NUMINAMATH_GPT_find_e_l1851_185144
