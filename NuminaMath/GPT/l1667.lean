import Mathlib

namespace NUMINAMATH_GPT_solve_compound_inequality_l1667_166727

noncomputable def compound_inequality_solution (x : ℝ) : Prop :=
  (3 - (1 / (3 * x + 4)) < 5) ∧ (2 * x + 1 > 0)

theorem solve_compound_inequality (x : ℝ) :
  compound_inequality_solution x ↔ (x > -1/2) :=
by
  sorry

end NUMINAMATH_GPT_solve_compound_inequality_l1667_166727


namespace NUMINAMATH_GPT_pure_imaginary_complex_number_solution_l1667_166711

theorem pure_imaginary_complex_number_solution (m : ℝ) :
  (m^2 - 5 * m + 6 = 0) ∧ (m^2 - 3 * m ≠ 0) → m = 2 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_complex_number_solution_l1667_166711


namespace NUMINAMATH_GPT_non_visible_dots_l1667_166774

-- Define the configuration of the dice
def total_dots_on_one_die : ℕ := 1 + 2 + 3 + 4 + 5 + 6
def total_dots_on_two_dice : ℕ := 2 * total_dots_on_one_die
def visible_dots : ℕ := 2 + 3 + 5

-- The statement to prove
theorem non_visible_dots : total_dots_on_two_dice - visible_dots = 32 := by sorry

end NUMINAMATH_GPT_non_visible_dots_l1667_166774


namespace NUMINAMATH_GPT_never_prime_l1667_166736

theorem never_prime (p : ℕ) (hp : Nat.Prime p) : ¬Nat.Prime (p^2 + 105) := sorry

end NUMINAMATH_GPT_never_prime_l1667_166736


namespace NUMINAMATH_GPT_value_of_a_l1667_166702

variable (x y a : ℝ)

-- Conditions
def condition1 : Prop := (x = 1)
def condition2 : Prop := (y = 2)
def condition3 : Prop := (3 * x - a * y = 1)

-- Theorem stating the equivalence between the conditions and the value of 'a'
theorem value_of_a (h1 : condition1 x) (h2 : condition2 y) (h3 : condition3 x y a) : a = 1 :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_value_of_a_l1667_166702


namespace NUMINAMATH_GPT_cubic_vs_square_ratio_l1667_166763

theorem cubic_vs_square_ratio 
  (s r : ℝ) 
  (hs : 0 < s) 
  (hr : 0 < r) 
  (h : r < s) : 
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) :=
by sorry

end NUMINAMATH_GPT_cubic_vs_square_ratio_l1667_166763


namespace NUMINAMATH_GPT_Cartesian_eq_C2_correct_distance_AB_correct_l1667_166778

-- Part I: Proving the Cartesian equation of curve (C2)
noncomputable def equation_of_C2 (x y : ℝ) (α : ℝ) : Prop :=
  x = 4 * Real.cos α ∧ y = 4 + 4 * Real.sin α

def Cartesian_eq_C2 (x y : ℝ) : Prop :=
  x^2 + (y - 4)^2 = 16

theorem Cartesian_eq_C2_correct (x y α : ℝ) (h : equation_of_C2 x y α) : Cartesian_eq_C2 x y :=
by sorry

-- Part II: Proving the distance |AB| given polar equations
noncomputable def polar_eq_C1 (theta : ℝ) : ℝ :=
  4 * Real.sin theta

noncomputable def polar_eq_C2 (theta : ℝ) : ℝ :=
  8 * Real.sin theta

def distance_AB (rho1 rho2 : ℝ) : ℝ :=
  abs (rho1 - rho2)

theorem distance_AB_correct : distance_AB (polar_eq_C1 (π / 3)) (polar_eq_C2 (π / 3)) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_Cartesian_eq_C2_correct_distance_AB_correct_l1667_166778


namespace NUMINAMATH_GPT_slope_of_parallel_line_l1667_166708

theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : ∃ m : ℝ, m = 1 / 2 := 
by {
  -- Proof body here
  sorry
}

end NUMINAMATH_GPT_slope_of_parallel_line_l1667_166708


namespace NUMINAMATH_GPT_find_marks_in_mathematics_l1667_166716

theorem find_marks_in_mathematics
  (english : ℕ)
  (physics : ℕ)
  (chemistry : ℕ)
  (biology : ℕ)
  (average : ℕ)
  (subjects : ℕ)
  (marks_math : ℕ) :
  english = 96 →
  physics = 82 →
  chemistry = 97 →
  biology = 95 →
  average = 93 →
  subjects = 5 →
  (average * subjects = english + marks_math + physics + chemistry + biology) →
  marks_math = 95 :=
  by
    intros h_eng h_phy h_chem h_bio h_avg h_sub h_eq
    rw [h_eng, h_phy, h_chem, h_bio, h_avg, h_sub] at h_eq
    sorry

end NUMINAMATH_GPT_find_marks_in_mathematics_l1667_166716


namespace NUMINAMATH_GPT_square_line_product_l1667_166796

theorem square_line_product (b : ℝ) (h1 : y = 3) (h2 : y = 7) (h3 := x = 2) (h4 : x = b) : 
  (b = 6 ∨ b = -2) → (6 * -2 = -12) :=
by
  sorry

end NUMINAMATH_GPT_square_line_product_l1667_166796


namespace NUMINAMATH_GPT_absolute_value_expression_l1667_166733

theorem absolute_value_expression : 
  (abs ((-abs (-1 + 2))^2 - 1) = 0) :=
sorry

end NUMINAMATH_GPT_absolute_value_expression_l1667_166733


namespace NUMINAMATH_GPT_find_number_l1667_166729

theorem find_number (x : ℝ) (h : 3 * (2 * x + 5) = 105) : x = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1667_166729


namespace NUMINAMATH_GPT_total_expense_l1667_166720

theorem total_expense (tanya_face_cost : ℕ) (tanya_face_qty : ℕ) (tanya_body_cost : ℕ) (tanya_body_qty : ℕ) 
  (tanya_total_expense : ℕ) (christy_multiplier : ℕ) (christy_total_expense : ℕ) (total_expense : ℕ) :
  tanya_face_cost = 50 →
  tanya_face_qty = 2 →
  tanya_body_cost = 60 →
  tanya_body_qty = 4 →
  tanya_total_expense = tanya_face_qty * tanya_face_cost + tanya_body_qty * tanya_body_cost →
  christy_multiplier = 2 →
  christy_total_expense = christy_multiplier * tanya_total_expense →
  total_expense = christy_total_expense + tanya_total_expense →
  total_expense = 1020 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_expense_l1667_166720


namespace NUMINAMATH_GPT_rectangle_area_l1667_166722

theorem rectangle_area (AB AC : ℝ) (H1 : AB = 15) (H2 : AC = 17) : 
  ∃ (BC : ℝ), (AB * BC = 120) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1667_166722


namespace NUMINAMATH_GPT_total_ducks_in_lake_l1667_166797

/-- 
Problem: Determine the total number of ducks in the lake after more ducks join.

Conditions:
- Initially, there are 13 ducks in the lake.
- 20 more ducks come to join them.
-/

def initial_ducks : Nat := 13

def new_ducks : Nat := 20

theorem total_ducks_in_lake : initial_ducks + new_ducks = 33 := by
  sorry -- Proof to be filled in later

end NUMINAMATH_GPT_total_ducks_in_lake_l1667_166797


namespace NUMINAMATH_GPT_g_of_neg_two_l1667_166793

def f (x : ℝ) : ℝ := 4 * x - 9

def g (x : ℝ) : ℝ := 3 * x ^ 2 + 4 * x - 2

theorem g_of_neg_two : g (-2) = 227 / 16 :=
by
  sorry

end NUMINAMATH_GPT_g_of_neg_two_l1667_166793


namespace NUMINAMATH_GPT_expression_value_l1667_166713

-- Proving the value of the expression using the factorial and sum formulas
theorem expression_value :
  (Nat.factorial 10) / (10 * 11 / 2) = 66069 := 
sorry

end NUMINAMATH_GPT_expression_value_l1667_166713


namespace NUMINAMATH_GPT_optimal_perimeter_proof_l1667_166700

-- Definition of conditions
def fencing_length : Nat := 400
def min_width : Nat := 50
def area : Nat := 8000

-- Definition of the perimeter to be proven as optimal
def optimal_perimeter : Nat := 360

-- Theorem statement to be proven
theorem optimal_perimeter_proof (l w : Nat) (h1 : l * w = area) (h2 : 2 * l + 2 * w <= fencing_length) (h3 : w >= min_width) :
  2 * l + 2 * w = optimal_perimeter :=
sorry

end NUMINAMATH_GPT_optimal_perimeter_proof_l1667_166700


namespace NUMINAMATH_GPT_Murtha_pebbles_problem_l1667_166718

theorem Murtha_pebbles_problem : 
  let a := 3
  let d := 3
  let n := 18
  let a_n := a + (n - 1) * d
  let S_n := n / 2 * (a + a_n)
  S_n = 513 :=
by
  sorry

end NUMINAMATH_GPT_Murtha_pebbles_problem_l1667_166718


namespace NUMINAMATH_GPT_tan_sin_cos_log_expression_simplification_l1667_166782

-- Proof Problem 1 Statement in Lean 4
theorem tan_sin_cos (α : ℝ) (h : Real.tan (Real.pi / 4 + α) = 2) : 
  (Real.sin α + 3 * Real.cos α) / (Real.sin α - Real.cos α) = -5 :=
by
  sorry

-- Proof Problem 2 Statement in Lean 4
theorem log_expression_simplification : 
  Real.logb 3 (Real.sqrt 27) + Real.logb 10 25 + Real.logb 10 4 + 
  (7 : ℝ) ^ Real.logb 7 2 + (-9.8) ^ 0 = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_sin_cos_log_expression_simplification_l1667_166782


namespace NUMINAMATH_GPT_variance_of_data_is_0_02_l1667_166726

def data : List ℝ := [10.1, 9.8, 10, 9.8, 10.2]

theorem variance_of_data_is_0_02 (h : (10.1 + 9.8 + 10 + 9.8 + 10.2) / 5 = 10) : 
  (1 / 5) * ((10.1 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10 - 10) ^ 2 + (9.8 - 10) ^ 2 + (10.2 - 10) ^ 2) = 0.02 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_data_is_0_02_l1667_166726


namespace NUMINAMATH_GPT_value_of_a_plus_b_l1667_166724

theorem value_of_a_plus_b (a b : ℝ) (h1 : |a| = 5) (h2 : |b| = 1) (h3 : a - b < 0) :
  a + b = -6 ∨ a + b = -4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_plus_b_l1667_166724


namespace NUMINAMATH_GPT_min_value_of_3x_plus_4y_l1667_166772

open Real

theorem min_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

end NUMINAMATH_GPT_min_value_of_3x_plus_4y_l1667_166772


namespace NUMINAMATH_GPT_marla_colors_red_squares_l1667_166731

-- Conditions
def total_rows : Nat := 10
def squares_per_row : Nat := 15
def total_squares : Nat := total_rows * squares_per_row

def blue_rows_top : Nat := 2
def blue_rows_bottom : Nat := 2
def total_blue_rows : Nat := blue_rows_top + blue_rows_bottom
def total_blue_squares : Nat := total_blue_rows * squares_per_row

def green_squares : Nat := 66
def red_rows : Nat := 4

-- Theorem to prove 
theorem marla_colors_red_squares : 
  total_squares - total_blue_squares - green_squares = red_rows * 6 :=
by
  sorry -- This skips the proof

end NUMINAMATH_GPT_marla_colors_red_squares_l1667_166731


namespace NUMINAMATH_GPT_girls_not_join_field_trip_l1667_166792

theorem girls_not_join_field_trip (total_students : ℕ) (number_of_boys : ℕ) (number_on_trip : ℕ)
  (h_total : total_students = 18)
  (h_boys : number_of_boys = 8)
  (h_equal : number_on_trip = number_of_boys) :
  total_students - number_of_boys - number_on_trip = 2 := by
sorry

end NUMINAMATH_GPT_girls_not_join_field_trip_l1667_166792


namespace NUMINAMATH_GPT_race_course_length_l1667_166757

theorem race_course_length (v : ℝ) (d : ℝ) (h1 : d = 7 * (d - 120)) : d = 140 :=
sorry

end NUMINAMATH_GPT_race_course_length_l1667_166757


namespace NUMINAMATH_GPT_home_electronics_budget_l1667_166721

theorem home_electronics_budget (deg_ba: ℝ) (b_deg: ℝ) (perc_me: ℝ) (perc_fa: ℝ) (perc_gm: ℝ) (perc_il: ℝ) : 
  deg_ba = 43.2 → 
  b_deg = 360 → 
  perc_me = 12 →
  perc_fa = 15 →
  perc_gm = 29 →
  perc_il = 8 →
  (b_deg / 360 * 100 = 12) → 
  perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100) = 76 →
  100 - (perc_il + perc_fa + perc_gm + perc_il + (b_deg / 360 * 100)) = 24 :=
by
  intro h_deg_ba h_b_deg h_perc_me h_perc_fa h_perc_gm h_perc_il h_ba_12perc h_total_76perc
  sorry

end NUMINAMATH_GPT_home_electronics_budget_l1667_166721


namespace NUMINAMATH_GPT_rectangle_ratio_l1667_166766

theorem rectangle_ratio (a b c d : ℝ)
  (h1 : (a * b) / (c * d) = 0.16)
  (h2 : a / c = b / d) :
  a / c = 0.4 ∧ b / d = 0.4 :=
by 
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l1667_166766


namespace NUMINAMATH_GPT_eccentricity_of_hyperbola_l1667_166743

noncomputable def hyperbola_eccentricity : Prop :=
  ∀ (a b : ℝ), a > 0 → b > 0 → (∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  → (∀ (c : ℝ), c^2 = a^2 + b^2) → b = 3 * a → ∃ e : ℝ, e = Real.sqrt 10

-- Statement of the problem without proof (includes the conditions)
theorem eccentricity_of_hyperbola (a b : ℝ) (h : a > 0) (h2 : b > 0) 
  (h3 : ∀ (x y : ℝ), x - 3 * y + 1 = 0 → y = 3 * x)
  (h4 : ∀ (c : ℝ), c^2 = a^2 + b^2) : hyperbola_eccentricity := 
  sorry

end NUMINAMATH_GPT_eccentricity_of_hyperbola_l1667_166743


namespace NUMINAMATH_GPT_collinear_points_eq_sum_l1667_166717

theorem collinear_points_eq_sum (a b : ℝ) :
  -- Collinearity conditions in ℝ³
  (∃ t1 t2 t3 t4 : ℝ,
    (2, a, b) = (a + t1 * (a - 2), 3 + t1 * (b - 3), b + t1 * (4 - b)) ∧
    (a, 3, b) = (a + t2 * (a - 2), 3 + t2 * (b - 3), b + t2 * (4 - b)) ∧
    (a, b, 4) = (a + t3 * (a - 2), 3 + t3 * (b - 3), b + t3 * (4 - b)) ∧
    (5, b, a) = (a + t4 * (a - 2), 3 + t4 * (b - 3), b + t4 * (4 - b))) →
  a + b = 9 :=
by
  sorry

end NUMINAMATH_GPT_collinear_points_eq_sum_l1667_166717


namespace NUMINAMATH_GPT_isosceles_triangle_base_l1667_166740

theorem isosceles_triangle_base (h_perimeter : 2 * 1.5 + x = 3.74) : x = 0.74 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_l1667_166740


namespace NUMINAMATH_GPT_closest_perfect_square_to_350_l1667_166730

theorem closest_perfect_square_to_350 : ∃ m : ℤ, (m^2 = 361) ∧ (∀ n : ℤ, (n^2 ≤ 350 ∧ 350 - n^2 > 361 - 350) → false)
:= by
  sorry

end NUMINAMATH_GPT_closest_perfect_square_to_350_l1667_166730


namespace NUMINAMATH_GPT_tan_alpha_eq_neg_one_l1667_166771

-- Define the point P and the angle α
def P : ℝ × ℝ := (-1, 1)
def α : ℝ := sorry  -- α is the angle whose terminal side passes through P

-- Statement to be proved
theorem tan_alpha_eq_neg_one (h : (P.1, P.2) = (-1, 1)) : Real.tan α = -1 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg_one_l1667_166771


namespace NUMINAMATH_GPT_cost_of_building_fence_l1667_166761

-- Define the conditions
def area : ℕ := 289
def price_per_foot : ℕ := 60

-- Define the length of one side of the square (since area = side^2)
def side_length (a : ℕ) : ℕ := Nat.sqrt a

-- Define the perimeter of the square (since square has 4 equal sides)
def perimeter (s : ℕ) : ℕ := 4 * s

-- Define the cost of building the fence
def cost (p : ℕ) (ppf : ℕ) : ℕ := p * ppf

-- Prove that the cost of building the fence is Rs. 4080
theorem cost_of_building_fence : cost (perimeter (side_length area)) price_per_foot = 4080 := by
  -- Skip the proof steps
  sorry

end NUMINAMATH_GPT_cost_of_building_fence_l1667_166761


namespace NUMINAMATH_GPT_sequences_power_of_two_l1667_166798

open scoped Classical

theorem sequences_power_of_two (n : ℕ) (a b : Fin n → ℚ)
  (h1 : (∃ i j, i < j ∧ a i = a j) → ∀ i, a i = b i)
  (h2 : {p | ∃ (i j : Fin n), i < j ∧ (a i + a j = p)} = {q | ∃ (i j : Fin n), i < j ∧ (b i + b j = q)})
  (h3 : ∃ i j, i < j ∧ a i ≠ b i) :
  ∃ k : ℕ, n = 2 ^ k := 
sorry

end NUMINAMATH_GPT_sequences_power_of_two_l1667_166798


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1667_166767

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 < x + 6) ↔ (-2 < x ∧ x < 3) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1667_166767


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1667_166795

-- Definitions for the conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ b = c ∨ c = a

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Statement of the theorem
theorem isosceles_triangle_perimeter {a b c : ℕ} (h1 : is_isosceles_triangle a b c) (h2 : valid_triangle a b c) :
  (a = 2 ∧ b = 4 ∧ c = 4 ∨ a = 4 ∧ b = 4 ∧ c = 2 ∨ a = 4 ∧ b = 2 ∧ c = 4) →
  a + b + c = 10 :=
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1667_166795


namespace NUMINAMATH_GPT_new_mean_of_five_numbers_l1667_166781

theorem new_mean_of_five_numbers (a b c d e : ℝ) 
  (h_mean : (a + b + c + d + e) / 5 = 25) :
  ((a + 5) + (b + 10) + (c + 15) + (d + 20) + (e + 25)) / 5 = 40 :=
by
  sorry

end NUMINAMATH_GPT_new_mean_of_five_numbers_l1667_166781


namespace NUMINAMATH_GPT_planned_pigs_correct_l1667_166773

-- Define initial number of animals
def initial_cows : ℕ := 2
def initial_pigs : ℕ := 3
def initial_goats : ℕ := 6

-- Define planned addition of animals
def added_cows : ℕ := 3
def added_goats : ℕ := 2
def total_animals : ℕ := 21

-- Define the total planned number of pigs to verify:
def planned_pigs := 8

-- State the final number of pigs to be proven
theorem planned_pigs_correct : 
  initial_cows + initial_pigs + initial_goats + added_cows + planned_pigs + added_goats = total_animals :=
by
  sorry

end NUMINAMATH_GPT_planned_pigs_correct_l1667_166773


namespace NUMINAMATH_GPT_complement_of_M_l1667_166719

open Set

def U : Set ℝ := univ

def M : Set ℝ := { x | x^2 - x ≥ 0 }

theorem complement_of_M :
  compl M = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_l1667_166719


namespace NUMINAMATH_GPT_find_a_2b_3c_l1667_166799

noncomputable def a : ℝ := 28
noncomputable def b : ℝ := 32
noncomputable def c : ℝ := -3

def ineq_condition (x : ℝ) : Prop := (x < -3) ∨ (abs (x - 30) ≤ 2)

theorem find_a_2b_3c (a b c : ℝ) (h₁ : a < b)
  (h₂ : ∀ x : ℝ, (x < -3 ∨ abs (x - 30) ≤ 2) ↔ ((x - a)*(x - b)/(x - c) ≤ 0)) :
  a + 2 * b + 3 * c = 83 :=
by
  sorry

end NUMINAMATH_GPT_find_a_2b_3c_l1667_166799


namespace NUMINAMATH_GPT_mixed_number_division_l1667_166744

theorem mixed_number_division : 
  let a := 9 / 4
  let b := 3 / 5
  a / b = 15 / 4 :=
by
  sorry

end NUMINAMATH_GPT_mixed_number_division_l1667_166744


namespace NUMINAMATH_GPT_total_insects_eaten_l1667_166775

theorem total_insects_eaten : 
  (5 * 6) + (3 * (2 * 6)) = 66 :=
by
  /- We'll calculate the total number of insects eaten by combining the amounts eaten by the geckos and lizards -/
  sorry

end NUMINAMATH_GPT_total_insects_eaten_l1667_166775


namespace NUMINAMATH_GPT_find_b_l1667_166701

/-- Given the distance between the parallel lines l₁ : x - y = 0
  and l₂ : x - y + b = 0 is √2, prove that b = 2 or b = -2. --/
theorem find_b (b : ℝ) (h : ∀ (x y : ℝ), (x - y = 0) → ∀ (x' y' : ℝ), (x' - y' + b = 0) → (|b| / Real.sqrt 2 = Real.sqrt 2)) :
  b = 2 ∨ b = -2 :=
sorry

end NUMINAMATH_GPT_find_b_l1667_166701


namespace NUMINAMATH_GPT_canoe_row_probability_l1667_166732

-- Definitions based on conditions
def prob_left_works : ℚ := 3 / 5
def prob_right_works : ℚ := 3 / 5

-- The probability that you can still row the canoe
def prob_can_row : ℚ := 
  prob_left_works * prob_right_works +  -- both oars work
  prob_left_works * (1 - prob_right_works) +  -- left works, right breaks
  (1 - prob_left_works) * prob_right_works  -- left breaks, right works
  
theorem canoe_row_probability : prob_can_row = 21 / 25 := by
  -- Skip proof for now
  sorry

end NUMINAMATH_GPT_canoe_row_probability_l1667_166732


namespace NUMINAMATH_GPT_percentage_ownership_l1667_166770

theorem percentage_ownership (total students_cats students_dogs : ℕ) (h1 : total = 500) (h2 : students_cats = 75) (h3 : students_dogs = 125):
  (students_cats / total : ℝ) = 0.15 ∧
  (students_dogs / total : ℝ) = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_ownership_l1667_166770


namespace NUMINAMATH_GPT_sequence_divisibility_l1667_166703

theorem sequence_divisibility (g : ℕ → ℕ) (h₁ : g 1 = 1) 
(h₂ : ∀ n : ℕ, g (n + 1) = g n ^ 2 + g n + 1) 
(n : ℕ) : g n ^ 2 + 1 ∣ g (n + 1) ^ 2 + 1 :=
sorry

end NUMINAMATH_GPT_sequence_divisibility_l1667_166703


namespace NUMINAMATH_GPT_closest_integer_to_a2013_l1667_166734

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 100 ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) + (1 / a (n + 1))

theorem closest_integer_to_a2013 (a : ℕ → ℝ) (h : seq a) : abs (a 2013 - 118) < 0.5 :=
sorry

end NUMINAMATH_GPT_closest_integer_to_a2013_l1667_166734


namespace NUMINAMATH_GPT_average_sales_l1667_166741

-- Define the cost calculation for each special weekend
noncomputable def valentines_day_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20)) / 10

noncomputable def st_patricks_day_sales_per_ticket : Real :=
  ((3 * 2.00) + 6.25 + (8 * 1.00)) / 8

noncomputable def christmas_sales_per_ticket : Real :=
  ((6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 9

-- Define the combined average snack sales
noncomputable def combined_average_sales_per_ticket : Real :=
  ((4 * 2.20) + (6 * 1.50) + (7 * 1.20) + (3 * 2.00) + 6.25 + (8 * 1.00) + (6 * 2.15) + (4.25 + (4.25 / 3.0)) + (9 * 1.10)) / 27

-- Proof problem as a Lean theorem
theorem average_sales : 
  valentines_day_sales_per_ticket = 2.62 ∧ 
  st_patricks_day_sales_per_ticket = 2.53 ∧ 
  christmas_sales_per_ticket = 3.16 ∧ 
  combined_average_sales_per_ticket = 2.78 :=
by 
  sorry

end NUMINAMATH_GPT_average_sales_l1667_166741


namespace NUMINAMATH_GPT_selling_price_of_bracelet_l1667_166737

theorem selling_price_of_bracelet (x : ℝ) 
  (cost_per_bracelet : ℝ) 
  (num_bracelets : ℕ) 
  (box_of_cookies_cost : ℝ) 
  (money_left_after_buying_cookies : ℝ) 
  (total_revenue : ℝ) 
  (total_cost_of_supplies : ℝ) :
  cost_per_bracelet = 1 →
  num_bracelets = 12 →
  box_of_cookies_cost = 3 →
  money_left_after_buying_cookies = 3 →
  total_cost_of_supplies = cost_per_bracelet * num_bracelets →
  total_revenue = 9 →
  x = total_revenue / num_bracelets :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_selling_price_of_bracelet_l1667_166737


namespace NUMINAMATH_GPT_slope_tangent_line_at_zero_l1667_166751

noncomputable def f (x : ℝ) : ℝ := (2 * x - 5) / (x^2 + 1)

theorem slope_tangent_line_at_zero : 
  (deriv f 0) = 2 :=
sorry

end NUMINAMATH_GPT_slope_tangent_line_at_zero_l1667_166751


namespace NUMINAMATH_GPT_kishore_expenses_l1667_166704

noncomputable def total_salary (savings : ℕ) (percent : ℝ) : ℝ :=
savings / percent

noncomputable def total_expenses (rent milk groceries education petrol : ℕ) : ℕ :=
  rent + milk + groceries + education + petrol

noncomputable def miscellaneous_expenses (total_salary : ℝ) (total_expenses : ℕ) (savings : ℕ) : ℝ :=
  total_salary - (total_expenses + savings)

theorem kishore_expenses :
  total_salary 2160 0.1 - (total_expenses 5000 1500 4500 2500 2000 + 2160) = 3940 := by
  sorry

end NUMINAMATH_GPT_kishore_expenses_l1667_166704


namespace NUMINAMATH_GPT_problem_solution_l1667_166762

theorem problem_solution (n : ℕ) (h : n^3 - n = 5814) : (n % 2 = 0) :=
by sorry

end NUMINAMATH_GPT_problem_solution_l1667_166762


namespace NUMINAMATH_GPT_investment_of_c_l1667_166749

variable (P_a P_b P_c C_a C_b C_c : ℝ)

theorem investment_of_c (h1 : P_b = 3500) 
                        (h2 : P_a - P_c = 1399.9999999999998) 
                        (h3 : C_a = 8000) 
                        (h4 : C_b = 10000) 
                        (h5 : P_a / C_a = P_b / C_b) 
                        (h6 : P_c / C_c = P_b / C_b) : 
                        C_c = 40000 := 
by 
  sorry

end NUMINAMATH_GPT_investment_of_c_l1667_166749


namespace NUMINAMATH_GPT_candy_bar_cost_l1667_166747

theorem candy_bar_cost {initial_money left_money cost_bar : ℕ} 
                        (h_initial : initial_money = 4)
                        (h_left : left_money = 3)
                        (h_cost : cost_bar = initial_money - left_money) :
                        cost_bar = 1 :=
by 
  sorry -- Proof is not required as per the instructions

end NUMINAMATH_GPT_candy_bar_cost_l1667_166747


namespace NUMINAMATH_GPT_louis_current_age_l1667_166787

-- Define the constants for years to future and future age of Carla
def years_to_future : ℕ := 6
def carla_future_age : ℕ := 30

-- Define the sum of current ages
def sum_current_ages : ℕ := 55

-- State the theorem
theorem louis_current_age :
  ∃ (c l : ℕ), (c + years_to_future = carla_future_age) ∧ (c + l = sum_current_ages) ∧ (l = 31) :=
sorry

end NUMINAMATH_GPT_louis_current_age_l1667_166787


namespace NUMINAMATH_GPT_quadratic_has_real_roots_l1667_166710

open Real

theorem quadratic_has_real_roots (k : ℝ) (h : k ≠ 0) :
    ∃ x : ℝ, x^2 + k * x + k^2 - 1 = 0 ↔
    -2 / sqrt 3 ≤ k ∧ k ≤ 2 / sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_has_real_roots_l1667_166710


namespace NUMINAMATH_GPT_arithmetic_progression_sum_l1667_166786

theorem arithmetic_progression_sum (a d S n : ℤ) (h_a : a = 32) (h_d : d = -4) (h_S : S = 132) :
  (n = 6 ∨ n = 11) :=
by
  -- Start the proof here
  sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_l1667_166786


namespace NUMINAMATH_GPT_correct_sampling_methods_l1667_166784

def reporter_A_sampling : String :=
  "systematic sampling"

def reporter_B_sampling : String :=
  "systematic sampling"

theorem correct_sampling_methods (constant_flow : Prop)
  (A_interview_method : ∀ t : ℕ, t % 10 = 0)
  (B_interview_method : ∀ n : ℕ, n % 1000 = 0) :
  reporter_A_sampling = "systematic sampling" ∧ reporter_B_sampling = "systematic sampling" :=
by
  sorry

end NUMINAMATH_GPT_correct_sampling_methods_l1667_166784


namespace NUMINAMATH_GPT_perpendicular_vectors_m_value_l1667_166790

theorem perpendicular_vectors_m_value : 
  ∀ (m : ℝ), ((2 : ℝ) * (1 : ℝ) + (m * (1 / 2)) + (1 * 2) = 0) → m = -8 :=
by
  intro m
  intro h
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_m_value_l1667_166790


namespace NUMINAMATH_GPT_sum_of_roots_l1667_166753

theorem sum_of_roots (x1 x2 : ℝ) (h : x1^2 + 5*x1 - 1 = 0 ∧ x2^2 + 5*x2 - 1 = 0) : x1 + x2 = -5 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l1667_166753


namespace NUMINAMATH_GPT_greatest_integer_a_l1667_166777

-- Define formal properties and state the main theorem.
theorem greatest_integer_a (a : ℤ) : (∀ x : ℝ, ¬(x^2 + (a:ℝ) * x + 15 = 0)) → (a ≤ 7) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_greatest_integer_a_l1667_166777


namespace NUMINAMATH_GPT_solve_inequality_l1667_166789

theorem solve_inequality (x : ℝ) : x + 2 < 1 ↔ x < -1 := sorry

end NUMINAMATH_GPT_solve_inequality_l1667_166789


namespace NUMINAMATH_GPT_find_x4_y4_z4_l1667_166707

theorem find_x4_y4_z4
  (x y z : ℝ)
  (h1 : x + y + z = 3)
  (h2 : x^2 + y^2 + z^2 = 5)
  (h3 : x^3 + y^3 + z^3 = 7) :
  x^4 + y^4 + z^4 = 59 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x4_y4_z4_l1667_166707


namespace NUMINAMATH_GPT_every_integer_appears_exactly_once_l1667_166706

-- Define the sequence of integers
variable (a : ℕ → ℤ)

-- Define the conditions
axiom infinite_positives : ∀ n : ℕ, ∃ i > n, a i > 0
axiom infinite_negatives : ∀ n : ℕ, ∃ i > n, a i < 0
axiom distinct_remainders : ∀ n : ℕ, ∀ i j : ℕ, i < n → j < n → i ≠ j → (a i % n) ≠ (a j % n)

-- The proof statement
theorem every_integer_appears_exactly_once :
  ∀ x : ℤ, ∃! i : ℕ, a i = x :=
sorry

end NUMINAMATH_GPT_every_integer_appears_exactly_once_l1667_166706


namespace NUMINAMATH_GPT_find_two_digit_number_l1667_166748

theorem find_two_digit_number (x y : ℕ) (h1 : 10 * x + y = 4 * (x + y) + 3) (h2 : 10 * x + y = 3 * x * y + 5) : 10 * x + y = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_two_digit_number_l1667_166748


namespace NUMINAMATH_GPT_negation_prop_l1667_166791

open Classical

variable (x : ℝ)

theorem negation_prop :
    (∃ x : ℝ, x^2 + 2*x + 2 < 0) = False ↔
    (∀ x : ℝ, x^2 + 2*x + 2 ≥ 0) :=
by
    sorry

end NUMINAMATH_GPT_negation_prop_l1667_166791


namespace NUMINAMATH_GPT_cube_cut_off_edges_l1667_166709

theorem cube_cut_off_edges :
  let original_edges := 12
  let new_edges_per_vertex := 3
  let vertices := 8
  let new_edges := new_edges_per_vertex * vertices
  (original_edges + new_edges) = 36 :=
by
  sorry

end NUMINAMATH_GPT_cube_cut_off_edges_l1667_166709


namespace NUMINAMATH_GPT_prob_B_takes_second_shot_correct_prob_A_takes_ith_shot_correct_expected_A_shots_correct_l1667_166768

noncomputable def prob_A_makes_shot : ℝ := 0.6
noncomputable def prob_B_makes_shot : ℝ := 0.8
noncomputable def prob_A_starts : ℝ := 0.5
noncomputable def prob_B_starts : ℝ := 0.5

noncomputable def prob_B_takes_second_shot : ℝ :=
  prob_A_starts * (1 - prob_A_makes_shot) + prob_B_starts * prob_B_makes_shot

theorem prob_B_takes_second_shot_correct :
  prob_B_takes_second_shot = 0.6 :=
  sorry

noncomputable def prob_A_takes_nth_shot (n : ℕ) : ℝ :=
  let p₁ := 0.5
  let recurring_prob := (1 / 6) * ((2 / 5)^(n-1))
  (1 / 3) + recurring_prob

theorem prob_A_takes_ith_shot_correct (i : ℕ) :
  prob_A_takes_nth_shot i = (1 / 3) + (1 / 6) * ((2 / 5)^(i - 1)) :=
  sorry

noncomputable def expected_A_shots (n : ℕ) : ℝ :=
  let geometric_sum := ((2 / 5)^n - 1) / (1 - (2 / 5))
  (1 / 6) * geometric_sum + (n / 3)

theorem expected_A_shots_correct (n : ℕ) :
  expected_A_shots n = (5 / 18) * (1 - (2 / 5)^n) + (n / 3) :=
  sorry

end NUMINAMATH_GPT_prob_B_takes_second_shot_correct_prob_A_takes_ith_shot_correct_expected_A_shots_correct_l1667_166768


namespace NUMINAMATH_GPT_foma_should_give_ierema_55_coins_l1667_166725

variables (F E Y : ℝ)

-- Conditions
def condition1 : Prop := F - 70 = E + 70
def condition2 : Prop := F - 40 = Y
def condition3 : Prop := E + 70 = Y

theorem foma_should_give_ierema_55_coins
  (h1 : condition1 F E) (h2 : condition2 F Y) (h3 : condition3 E Y) :
  F - (E + 55) = E + 55 :=
by
  sorry

end NUMINAMATH_GPT_foma_should_give_ierema_55_coins_l1667_166725


namespace NUMINAMATH_GPT_boys_and_girls_at_bus_stop_l1667_166794

theorem boys_and_girls_at_bus_stop (H M : ℕ) 
  (h1 : H = 2 * (M - 15)) 
  (h2 : M - 15 = 5 * (H - 45)) : 
  H = 50 ∧ M = 40 := 
by 
  sorry

end NUMINAMATH_GPT_boys_and_girls_at_bus_stop_l1667_166794


namespace NUMINAMATH_GPT_triangle_area_inradius_l1667_166712

theorem triangle_area_inradius
  (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 35)
  (h2 : inradius = 4.5)
  (h3 : area = inradius * (perimeter / 2)) :
  area = 78.75 := by
  sorry

end NUMINAMATH_GPT_triangle_area_inradius_l1667_166712


namespace NUMINAMATH_GPT_pairs_a_eq_b_l1667_166742

theorem pairs_a_eq_b 
  (n : ℕ) (h_n : ¬ ∃ k : ℕ, k^2 = n) (a b : ℕ) 
  (r : ℝ) (h_r_pos : 0 < r) (h_ra_rational : ∃ q₁ : ℚ, r^a + (n:ℝ)^(1/2) = q₁) 
  (h_rb_rational : ∃ q₂ : ℚ, r^b + (n:ℝ)^(1/2) = q₂) : 
  a = b :=
sorry

end NUMINAMATH_GPT_pairs_a_eq_b_l1667_166742


namespace NUMINAMATH_GPT_interest_percentage_face_value_l1667_166738

def face_value : ℝ := 5000
def selling_price : ℝ := 6153.846153846153
def interest_percentage_selling_price : ℝ := 0.065

def interest_amount : ℝ := interest_percentage_selling_price * selling_price

theorem interest_percentage_face_value :
  (interest_amount / face_value) * 100 = 8 :=
by
  sorry

end NUMINAMATH_GPT_interest_percentage_face_value_l1667_166738


namespace NUMINAMATH_GPT_interest_rate_unique_l1667_166788

theorem interest_rate_unique (P r : ℝ) (h₁ : P * (1 + 3 * r) = 300) (h₂ : P * (1 + 8 * r) = 400) : r = 1 / 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_interest_rate_unique_l1667_166788


namespace NUMINAMATH_GPT_original_number_correct_l1667_166746

-- Definitions for the problem conditions
/-
Let N be the original number.
X is the number to be subtracted.
We are given that X = 8.
We need to show that (N - 8) mod 5 = 4, (N - 8) mod 7 = 4, and (N - 8) mod 9 = 4.
-/

-- Declaration of variables
variable (N : ℕ) (X : ℕ)

-- Given conditions
def conditions := (N - X) % 5 = 4 ∧ (N - X) % 7 = 4 ∧ (N - X) % 9 = 4

-- Given the subtracted number X is 8.
def X_val : ℕ := 8

-- Prove that N = 326 meets the conditions
theorem original_number_correct (h : X = X_val) : ∃ N, conditions N X ∧ N = 326 := by
  sorry

end NUMINAMATH_GPT_original_number_correct_l1667_166746


namespace NUMINAMATH_GPT_probability_divisible_by_5_l1667_166758

def is_three_digit_integer (M : ℕ) : Prop :=
  100 ≤ M ∧ M < 1000

def ones_digit_is_4 (M : ℕ) : Prop :=
  (M % 10) = 4

theorem probability_divisible_by_5 (M : ℕ) (h1 : is_three_digit_integer M) (h2 : ones_digit_is_4 M) :
  (∃ p : ℚ, p = 0) :=
by
  sorry

end NUMINAMATH_GPT_probability_divisible_by_5_l1667_166758


namespace NUMINAMATH_GPT_intersection_point_unique_m_l1667_166759

theorem intersection_point_unique_m (m : ℕ) (h1 : m > 0)
  (x y : ℤ) (h2 : 13 * x + 11 * y = 700) (h3 : y = m * x - 1) : m = 6 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_unique_m_l1667_166759


namespace NUMINAMATH_GPT_time_to_travel_to_shop_l1667_166783

-- Define the distance and speed as given conditions
def distance : ℕ := 184
def speed : ℕ := 23

-- Define the time taken for the journey
def time_taken (d : ℕ) (s : ℕ) : ℕ := d / s

-- Statement to prove that the time taken is 8 hours
theorem time_to_travel_to_shop : time_taken distance speed = 8 := by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_time_to_travel_to_shop_l1667_166783


namespace NUMINAMATH_GPT_count_valid_triangles_l1667_166780

/-- 
Define the problem constraints: scalene triangles with side lengths a, b, c, 
where a < b < c, a + c = 2b, and a + b + c ≤ 30.
-/
def is_valid_triangle (a b c : ℕ) : Prop :=
  a < b ∧ b < c ∧ a + c = 2 * b ∧ a + b + c ≤ 30

/-- 
Statement of the problem: Prove that there are 20 distinct triangles satisfying the above constraints. 
-/
theorem count_valid_triangles : ∃ n, n = 20 ∧ (∀ {a b c : ℕ}, is_valid_triangle a b c → n = 20) :=
sorry

end NUMINAMATH_GPT_count_valid_triangles_l1667_166780


namespace NUMINAMATH_GPT_function_behavior_on_negative_interval_l1667_166739

-- Define the necessary conditions and function properties
variables {f : ℝ → ℝ}

-- Conditions: f is even, increasing on [0, 7], and f(7) = 6
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def f7_eq_6 (f : ℝ → ℝ) : Prop := f 7 = 6

-- The theorem to prove
theorem function_behavior_on_negative_interval (h1 : even_function f) (h2 : increasing_on_interval f 0 7) (h3 : f7_eq_6 f) : 
  (∀ x y, -7 ≤ x → x ≤ y → y ≤ 0 → f x ≥ f y) ∧ (∀ x, -7 ≤ x → x ≤ 0 → f x ≤ 6) :=
sorry

end NUMINAMATH_GPT_function_behavior_on_negative_interval_l1667_166739


namespace NUMINAMATH_GPT_negation_of_implication_l1667_166764

theorem negation_of_implication (a b c : ℝ) :
  ¬ (a > b → a + c > b + c) ↔ (a ≤ b → a + c ≤ b + c) :=
by sorry

end NUMINAMATH_GPT_negation_of_implication_l1667_166764


namespace NUMINAMATH_GPT_combined_height_l1667_166756

theorem combined_height (h_John : ℕ) (h_Lena : ℕ) (h_Rebeca : ℕ)
  (cond1 : h_John = 152)
  (cond2 : h_John = h_Lena + 15)
  (cond3 : h_Rebeca = h_John + 6) :
  h_Lena + h_Rebeca = 295 :=
by
  sorry

end NUMINAMATH_GPT_combined_height_l1667_166756


namespace NUMINAMATH_GPT_watermelon_seeds_l1667_166705

variable (G Y B : ℕ)

theorem watermelon_seeds (h1 : Y = 3 * G) (h2 : G > B) (h3 : B = 300) (h4 : G + Y + B = 1660) : G = 340 := by
  sorry

end NUMINAMATH_GPT_watermelon_seeds_l1667_166705


namespace NUMINAMATH_GPT_perpendicular_line_eq_slope_intercept_l1667_166750

/-- Given the line 3x - 6y = 9, find the equation of the line perpendicular to it passing through the point (2, -3) in slope-intercept form. The resulting equation should be y = -2x + 1. -/
theorem perpendicular_line_eq_slope_intercept :
  ∃ (m b : ℝ), (m = -2) ∧ (b = 1) ∧ (∀ x y : ℝ, y = m * x + b ↔ (3 * x - 6 * y = 9) ∧ y = -2 * x + 1) :=
sorry

end NUMINAMATH_GPT_perpendicular_line_eq_slope_intercept_l1667_166750


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1667_166765

theorem quadratic_inequality_solution (x : ℝ) :
  (x^2 - 5 * x - 6 > 0) ↔ (x < -1 ∨ x > 6) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1667_166765


namespace NUMINAMATH_GPT_find_ab_l1667_166715

theorem find_ab (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 27) : a * b = 9 :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_find_ab_l1667_166715


namespace NUMINAMATH_GPT_find_smallest_denominator_difference_l1667_166728

theorem find_smallest_denominator_difference :
  ∃ (r s : ℕ), 
    r > 0 ∧ s > 0 ∧ 
    (5 : ℚ) / 11 < r / s ∧ r / s < (4 : ℚ) / 9 ∧ 
    ¬ ∃ t : ℕ, t < s ∧ (5 : ℚ) / 11 < r / t ∧ r / t < (4 : ℚ) / 9 ∧ 
    s - r = 11 := 
sorry

end NUMINAMATH_GPT_find_smallest_denominator_difference_l1667_166728


namespace NUMINAMATH_GPT_evaluate_functions_l1667_166769

def f (x : ℝ) := x + 2
def g (x : ℝ) := 2 * x^2 - 4
def h (x : ℝ) := x + 1

theorem evaluate_functions : f (g (h 3)) = 30 := by
  sorry

end NUMINAMATH_GPT_evaluate_functions_l1667_166769


namespace NUMINAMATH_GPT_numbers_are_perfect_squares_l1667_166760

/-- Prove that the numbers 49, 4489, 444889, ... obtained by inserting 48 into the 
middle of the previous number are perfect squares. -/
theorem numbers_are_perfect_squares :
  ∀ n : ℕ, ∃ k : ℕ, (k ^ 2) = (Int.ofNat ((20 * (10 : ℕ) ^ n + 1) / 3)) :=
by
  sorry

end NUMINAMATH_GPT_numbers_are_perfect_squares_l1667_166760


namespace NUMINAMATH_GPT_slope_of_line_intersecting_hyperbola_l1667_166776

theorem slope_of_line_intersecting_hyperbola 
  (A B : ℝ × ℝ)
  (hA : A.1^2 - A.2^2 = 1)
  (hB : B.1^2 - B.2^2 = 1)
  (midpoint_condition : (A.1 + B.1) / 2 = 2 ∧ (A.2 + B.2) / 2 = 1) :
  (B.2 - A.2) / (B.1 - A.1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_line_intersecting_hyperbola_l1667_166776


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1667_166752

variable (a b : ℝ)

def proposition_A : Prop := a > 0
def proposition_B : Prop := a > b ∧ a⁻¹ > b⁻¹

theorem necessary_but_not_sufficient : (proposition_B a b → proposition_A a) ∧ ¬(proposition_A a → proposition_B a b) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1667_166752


namespace NUMINAMATH_GPT_bus_stoppage_time_l1667_166745

theorem bus_stoppage_time (speed_excl_stoppages speed_incl_stoppages : ℕ) (h1 : speed_excl_stoppages = 54) (h2 : speed_incl_stoppages = 45) : 
  ∃ (t : ℕ), t = 10 := by
  sorry

end NUMINAMATH_GPT_bus_stoppage_time_l1667_166745


namespace NUMINAMATH_GPT_length_of_circle_l1667_166779

-- Define initial speeds and conditions
variables (V1 V2 : ℝ)
variables (L : ℝ) -- Length of the circle

-- Conditions
def initial_condition : Prop := V1 - V2 = 3
def extra_laps_after_speed_increase : Prop := (V1 + 10) - V2 = V1 - V2 + 10

-- Statement representing the mathematical equivalence
theorem length_of_circle
  (h1 : initial_condition V1 V2) 
  (h2 : extra_laps_after_speed_increase V1 V2) :
  L = 1250 := 
sorry

end NUMINAMATH_GPT_length_of_circle_l1667_166779


namespace NUMINAMATH_GPT_seating_arrangement_l1667_166754

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 7 * y = 112) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangement_l1667_166754


namespace NUMINAMATH_GPT_find_integer_n_l1667_166785

theorem find_integer_n : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] ∧ n = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_integer_n_l1667_166785


namespace NUMINAMATH_GPT_box_inscribed_in_sphere_l1667_166714

theorem box_inscribed_in_sphere (x y z r : ℝ) (surface_area : ℝ)
  (edge_sum : ℝ) (given_x : x = 8) 
  (given_surface_area : surface_area = 432) 
  (given_edge_sum : edge_sum = 104) 
  (surface_area_eq : 2 * (x * y + y * z + z * x) = surface_area)
  (edge_sum_eq : 4 * (x + y + z) = edge_sum) : 
  r = 7 :=
by
  sorry

end NUMINAMATH_GPT_box_inscribed_in_sphere_l1667_166714


namespace NUMINAMATH_GPT_intersection_with_x_axis_l1667_166735

noncomputable def f (x : ℝ) : ℝ := 
  (3 * x - 1) * (Real.sqrt (9 * x^2 - 6 * x + 5) + 1) + 
  (2 * x - 3) * (Real.sqrt (4 * x^2 - 12 * x + 13)) + 1

theorem intersection_with_x_axis :
  ∃ x : ℝ, f x = 0 ∧ x = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_intersection_with_x_axis_l1667_166735


namespace NUMINAMATH_GPT_not_possible_consecutive_results_l1667_166723

theorem not_possible_consecutive_results 
  (dot_counts : ℕ → ℕ)
  (h_identical_conditions : ∀ (i : ℕ), dot_counts i = 1 ∨ dot_counts i = 2 ∨ dot_counts i = 3) 
  (h_correct_dot_distribution : ∀ (i j : ℕ), (i ≠ j → dot_counts i ≠ dot_counts j))
  : ¬ (∃ (consecutive : ℕ → ℕ), 
        (∀ (k : ℕ), k < 6 → consecutive k = dot_counts (4 * k) + dot_counts (4 * k + 1) 
                         + dot_counts (4 * k + 2) + dot_counts (4 * k + 3))
        ∧ (∀ (k : ℕ), k < 5 → consecutive (k + 1) = consecutive k + 1)) := sorry

end NUMINAMATH_GPT_not_possible_consecutive_results_l1667_166723


namespace NUMINAMATH_GPT_price_decrease_percentage_l1667_166755

theorem price_decrease_percentage (P₀ P₁ P₂ : ℝ) (x : ℝ) :
  P₀ = 1 → P₁ = P₀ * 1.25 → P₂ = P₁ * (1 - x / 100) → P₂ = 1 → x = 20 :=
by
  intros h₀ h₁ h₂ h₃
  sorry

end NUMINAMATH_GPT_price_decrease_percentage_l1667_166755
