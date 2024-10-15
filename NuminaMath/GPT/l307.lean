import Mathlib

namespace NUMINAMATH_GPT_min_triangle_perimeter_proof_l307_30755

noncomputable def min_triangle_perimeter (l m n : ℕ) : ℕ :=
  if l > m ∧ m > n ∧ (3^l % 10000 = 3^m % 10000) ∧ (3^m % 10000 = 3^n % 10000) then
    l + m + n
  else
    0

theorem min_triangle_perimeter_proof : ∃ (l m n : ℕ), l > m ∧ m > n ∧ 
  (3^l % 10000 = 3^m % 10000) ∧
  (3^m % 10000 = 3^n % 10000) ∧ min_triangle_perimeter l m n = 3003 :=
  sorry

end NUMINAMATH_GPT_min_triangle_perimeter_proof_l307_30755


namespace NUMINAMATH_GPT_distinct_flavors_count_l307_30706

-- Define the number of available candies
def red_candies := 3
def green_candies := 2
def blue_candies := 4

-- Define what it means for a flavor to be valid: includes at least one candy of each color.
def is_valid_flavor (x y z : Nat) : Prop :=
  x ≥ 1 ∧ y ≥ 1 ∧ z ≥ 1 ∧ x ≤ red_candies ∧ y ≤ green_candies ∧ z ≤ blue_candies

-- Define what it means for two flavors to have the same ratio
def same_ratio (x1 y1 z1 x2 y2 z2 : Nat) : Prop :=
  x1 * y2 * z2 = x2 * y1 * z1

-- Define the proof problem: the number of distinct flavors
theorem distinct_flavors_count :
  ∃ n, n = 21 ∧ ∀ (x y z : Nat), is_valid_flavor x y z ↔ (∃ x' y' z', is_valid_flavor x' y' z' ∧ ¬ same_ratio x y z x' y' z') :=
sorry

end NUMINAMATH_GPT_distinct_flavors_count_l307_30706


namespace NUMINAMATH_GPT_kx2_kx_1_pos_l307_30759

theorem kx2_kx_1_pos (k : ℝ) : (∀ x : ℝ, k * x^2 - k * x + 1 > 0) ↔ (0 ≤ k ∧ k < 4) :=
sorry

end NUMINAMATH_GPT_kx2_kx_1_pos_l307_30759


namespace NUMINAMATH_GPT_original_number_from_sum_l307_30700

variable (a b c : ℕ) (m S : ℕ)

/-- Given a three-digit number, the magician asks the participant to add all permutations -/
def three_digit_number_permutations_sum (a b c : ℕ) : ℕ :=
  100 * a + 10 * b + c + (100 * a + 10 * c + b) + (100 * b + 10 * c + a) +
  (100 * b + 10 * a + c) + (100 * c + 10 * a + b) + (100 * c + 10 * b + a)

/-- Given the sum of all permutations of the three-digit number is 4239, determine the original number -/
theorem original_number_from_sum (S : ℕ) (hS : S = 4239) (Sum_conditions : three_digit_number_permutations_sum a b c = S) :
  (100 * a + 10 * b + c) = 429 := by
  sorry

end NUMINAMATH_GPT_original_number_from_sum_l307_30700


namespace NUMINAMATH_GPT_min_value_of_f_solve_inequality_l307_30796

noncomputable def f (x : ℝ) : ℝ := abs (x - 5/2) + abs (x - 1/2)

theorem min_value_of_f : (∀ x : ℝ, f x ≥ 2) ∧ (∃ x : ℝ, f x = 2) := by
  sorry

theorem solve_inequality (x : ℝ) : (f x ≤ x + 4) ↔ (-1/3 ≤ x ∧ x ≤ 7) := by
  sorry

end NUMINAMATH_GPT_min_value_of_f_solve_inequality_l307_30796


namespace NUMINAMATH_GPT_area_common_to_all_four_circles_l307_30734

noncomputable def common_area (R : ℝ) : ℝ :=
  (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6

theorem area_common_to_all_four_circles (R : ℝ) :
  ∃ (O1 O2 A B : ℝ × ℝ),
    dist O1 O2 = R ∧
    dist O1 A = R ∧
    dist O2 A = R ∧
    dist O1 B = R ∧
    dist O2 B = R ∧
    dist A B = R ∧
    common_area R = (R^2 * (2 * Real.pi - 3 * Real.sqrt 3)) / 6 :=
by
  sorry

end NUMINAMATH_GPT_area_common_to_all_four_circles_l307_30734


namespace NUMINAMATH_GPT_linear_function_product_neg_l307_30748

theorem linear_function_product_neg (a1 b1 a2 b2 : ℝ) (hP : b1 = -3 * a1 + 4) (hQ : b2 = -3 * a2 + 4) :
  (a1 - a2) * (b1 - b2) < 0 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_product_neg_l307_30748


namespace NUMINAMATH_GPT_div_equal_octagons_l307_30751

-- Definitions based on the conditions
def squareArea (n : ℕ) := n * n
def isDivisor (m n : ℕ) := n % m = 0

-- Main statement
theorem div_equal_octagons (n : ℕ) (hn : n = 8) :
  (2 ∣ squareArea n) ∨ (4 ∣ squareArea n) ∨ (8 ∣ squareArea n) ∨ (16 ∣ squareArea n) :=
by
  -- We shall show the divisibility aspect later.
  sorry

end NUMINAMATH_GPT_div_equal_octagons_l307_30751


namespace NUMINAMATH_GPT_letter_puzzle_solutions_l307_30717

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def to_number (B A : ℕ) : ℕ :=
  10 * B + A

theorem letter_puzzle_solutions (A B : ℕ) (h_diff : A ≠ B) (h_digits : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9)
  (h_two_digit : is_two_digit (to_number B A)) :
  A^B = to_number B A ↔ (A = 2 ∧ B = 5 ∨ A = 6 ∧ B = 2 ∨ A = 4 ∧ B = 3) :=
by
  sorry

end NUMINAMATH_GPT_letter_puzzle_solutions_l307_30717


namespace NUMINAMATH_GPT_correct_assertions_l307_30720

variables {A B : Type} (f : A → B)

-- 1. Different elements in set A can have the same image in set B
def statement_1 : Prop := ∃ a1 a2 : A, a1 ≠ a2 ∧ f a1 = f a2

-- 2. A single element in set A can have different images in B
def statement_2 : Prop := ∃ a1 : A, ∃ b1 b2 : B, b1 ≠ b2 ∧ (f a1 = b1 ∧ f a1 = b2)

-- 3. There can be elements in set B that do not have a pre-image in A
def statement_3 : Prop := ∃ b : B, ∀ a : A, f a ≠ b

-- Correct answer is statements 1 and 3 are true, statement 2 is false
theorem correct_assertions : statement_1 f ∧ ¬statement_2 f ∧ statement_3 f := sorry

end NUMINAMATH_GPT_correct_assertions_l307_30720


namespace NUMINAMATH_GPT_smallest_integer_solution_l307_30784

theorem smallest_integer_solution (x : ℤ) : 
  (∃ y : ℤ, (y > 20 / 21 ∧ (y = ↑x ∧ (x = 1)))) → (x = 1) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_solution_l307_30784


namespace NUMINAMATH_GPT_pasta_cost_is_one_l307_30741

-- Define the conditions
def pasta_cost (p : ℝ) : ℝ := p -- The cost of the pasta per box
def sauce_cost : ℝ := 2.00 -- The cost of the sauce
def meatballs_cost : ℝ := 5.00 -- The cost of the meatballs
def servings : ℕ := 8 -- The number of servings
def cost_per_serving : ℝ := 1.00 -- The cost per serving

-- Calculate the total meal cost
def total_meal_cost : ℝ := servings * cost_per_serving

-- Calculate the combined cost of sauce and meatballs
def combined_cost_of_sauce_and_meatballs : ℝ := sauce_cost + meatballs_cost

-- Calculate the cost of the pasta
def pasta_cost_calculation : ℝ := total_meal_cost - combined_cost_of_sauce_and_meatballs

-- The theorem stating that the pasta cost should be $1
theorem pasta_cost_is_one (p : ℝ) (h : pasta_cost_calculation = p) : p = 1 := by
  sorry

end NUMINAMATH_GPT_pasta_cost_is_one_l307_30741


namespace NUMINAMATH_GPT_total_fence_length_l307_30735

variable (Darren Doug : ℝ)

-- Definitions based on given conditions
def Darren_paints_more := Darren = 1.20 * Doug
def Darren_paints_360 := Darren = 360

-- The statement to prove
theorem total_fence_length (h1 : Darren_paints_more Darren Doug) (h2 : Darren_paints_360 Darren) : (Darren + Doug) = 660 := 
by
  sorry

end NUMINAMATH_GPT_total_fence_length_l307_30735


namespace NUMINAMATH_GPT_inequality_proof_l307_30733

noncomputable def f (x m : ℝ) : ℝ := 2 * m * x - Real.log x

theorem inequality_proof (m x₁ x₂ : ℝ) (hm : m ≥ -1) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0)
  (hineq : (f x₁ m + f x₂ m) / 2 ≤ x₁ ^ 2 + x₂ ^ 2 + (3 / 2) * x₁ * x₂) :
  x₁ + x₂ ≥ (Real.sqrt 3 - 1) / 2 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l307_30733


namespace NUMINAMATH_GPT_max_alpha_l307_30747

theorem max_alpha (A B C : ℝ) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hSum : A + B + C = π)
  (hmin : ∀ alpha, alpha = min (2 * A - B) (min (3 * B - 2 * C) (π / 2 - A))) :
  ∃ alpha, alpha = 2 * π / 9 := 
sorry

end NUMINAMATH_GPT_max_alpha_l307_30747


namespace NUMINAMATH_GPT_quadratic_completing_square_sum_l307_30742

theorem quadratic_completing_square_sum (q t : ℝ) :
    (∃ (x : ℝ), 9 * x^2 - 54 * x - 36 = 0 ∧ (x + q)^2 = t) →
    q + t = 10 := sorry

end NUMINAMATH_GPT_quadratic_completing_square_sum_l307_30742


namespace NUMINAMATH_GPT_avg_remaining_two_l307_30708

-- Defining the given conditions
variable (six_num_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ)

-- Defining the known values
axiom avg_val : six_num_avg = 3.95
axiom avg_group1 : group1_avg = 3.6
axiom avg_group2 : group2_avg = 3.85

-- Stating the problem to prove that the average of the remaining 2 numbers is 4.4
theorem avg_remaining_two (h : six_num_avg = 3.95) 
                           (h1: group1_avg = 3.6)
                           (h2: group2_avg = 3.85) : 
  4.4 = ((six_num_avg * 6) - (group1_avg * 2 + group2_avg * 2)) / 2 := 
sorry

end NUMINAMATH_GPT_avg_remaining_two_l307_30708


namespace NUMINAMATH_GPT_area_of_region_l307_30701

-- Define the condition: the equation of the region
def region_equation (x y : ℝ) : Prop := x^2 + y^2 + 10 * x - 4 * y + 9 = 0

-- State the theorem: the area of the region defined by the equation is 20π
theorem area_of_region : ∀ x y : ℝ, region_equation x y → ∃ A : ℝ, A = 20 * Real.pi :=
by sorry

end NUMINAMATH_GPT_area_of_region_l307_30701


namespace NUMINAMATH_GPT_log_expression_simplification_l307_30746

open Real

noncomputable def log_expr (a b c d x y z : ℝ) : ℝ :=
  log (a^2 / b) + log (b^2 / c) + log (c^2 / d) - log (a^2 * y * z / (d^2 * x))

theorem log_expression_simplification (a b c d x y z : ℝ) (h : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
(h4 : d ≠ 0) (h5 : x ≠ 0) (h6 : y ≠ 0) (h7 : z ≠ 0) :
  log_expr a b c d x y z = log (bdx / yz) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_log_expression_simplification_l307_30746


namespace NUMINAMATH_GPT_math_proof_problem_l307_30782

noncomputable def proof_problem (c d : ℝ) : Prop :=
  (∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  ∧ (∃ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3)
  ∧ 100 * c + d = 141
  
theorem math_proof_problem (c d : ℝ) 
  (h1 : ∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  (h2 : ∀ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3) :
  100 * c + d = 141 := 
sorry

end NUMINAMATH_GPT_math_proof_problem_l307_30782


namespace NUMINAMATH_GPT_abs_ineq_subs_ineq_l307_30773

-- Problem 1
theorem abs_ineq (x : ℝ) : -2 ≤ x ∧ x ≤ 2 ↔ |x - 1| + |x + 1| ≤ 4 := 
sorry

-- Problem 2
theorem subs_ineq (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) : 
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a + b + c := 
sorry

end NUMINAMATH_GPT_abs_ineq_subs_ineq_l307_30773


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l307_30732

theorem isosceles_triangle_base_length
  (a b c : ℕ)
  (h_iso : a = b)
  (h_perimeter : a + b + c = 62)
  (h_leg_length : a = 25) :
  c = 12 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l307_30732


namespace NUMINAMATH_GPT_exists_pos_int_n_l307_30764

def sequence_x (x : ℕ → ℝ) : Prop :=
  ∀ n, x (n + 2) = x n + (x (n + 1))^2

def sequence_y (y : ℕ → ℝ) : Prop :=
  ∀ n, y (n + 2) = y n^2 + y (n + 1)

def positive_initial_conditions (x y : ℕ → ℝ) : Prop :=
  x 1 > 1 ∧ x 2 > 1 ∧ y 1 > 1 ∧ y 2 > 1

theorem exists_pos_int_n (x y : ℕ → ℝ) (hx : sequence_x x) (hy : sequence_y y) 
  (ini : positive_initial_conditions x y) : ∃ n, x n > y n := 
sorry

end NUMINAMATH_GPT_exists_pos_int_n_l307_30764


namespace NUMINAMATH_GPT_toby_friends_girls_l307_30739

theorem toby_friends_girls (total_friends : ℕ) (num_boys : ℕ) (perc_boys : ℕ) 
  (h1 : perc_boys = 55) (h2 : num_boys = 33) (h3 : total_friends = 60) : 
  (total_friends - num_boys = 27) :=
by
  sorry

end NUMINAMATH_GPT_toby_friends_girls_l307_30739


namespace NUMINAMATH_GPT_parabola_shift_right_l307_30777

theorem parabola_shift_right (x : ℝ) :
  let original_parabola := - (1 / 2) * x^2
  let shifted_parabola := - (1 / 2) * (x - 1)^2
  original_parabola = shifted_parabola :=
sorry

end NUMINAMATH_GPT_parabola_shift_right_l307_30777


namespace NUMINAMATH_GPT_area_of_given_polygon_l307_30781

def point := (ℝ × ℝ)

def vertices : List point := [(0,0), (5,0), (5,2), (3,2), (3,3), (2,3), (2,2), (0,2), (0,0)]

def polygon_area (vertices : List point) : ℝ := 
  -- Function to compute the area of the given polygon
  -- Implementation of the area computation is assumed to be correct
  sorry

theorem area_of_given_polygon : polygon_area vertices = 11 :=
sorry

end NUMINAMATH_GPT_area_of_given_polygon_l307_30781


namespace NUMINAMATH_GPT_focus_of_parabola_l307_30725

theorem focus_of_parabola (f : ℝ) : 
  (∀ (x: ℝ), x^2 + ((- 1 / 16) * x^2 - f)^2 = ((- 1 / 16) * x^2 - (f + 8))^2) 
  → f = -4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l307_30725


namespace NUMINAMATH_GPT_find_p_geometric_progression_l307_30783

theorem find_p_geometric_progression (p : ℝ) : 
  (p = -1 ∨ p = 40 / 9) ↔ ((9 * p + 10), (3 * p), |p - 8|) ∈ 
  {gp | ∃ r : ℝ, gp = (r, r * r, r * r * r)} :=
by sorry

end NUMINAMATH_GPT_find_p_geometric_progression_l307_30783


namespace NUMINAMATH_GPT_move_line_left_and_up_l307_30713

/--
The equation of the line obtained by moving the line y = 2x - 3
2 units to the left and then 3 units up is y = 2x + 4.
-/
theorem move_line_left_and_up :
  ∀ (x y : ℝ), y = 2*x - 3 → ∃ x' y', x' = x + 2 ∧ y' = y + 3 ∧ y' = 2*x' + 4 :=
by
  sorry

end NUMINAMATH_GPT_move_line_left_and_up_l307_30713


namespace NUMINAMATH_GPT_problem_statement_l307_30794

theorem problem_statement (a : ℕ → ℝ)
  (h_recur : ∀ n, n ≥ 1 → a (n + 1) = a (n - 1) / (1 + n * a (n - 1) * a n))
  (h_initial_0 : a 0 = 1)
  (h_initial_1 : a 1 = 1) :
  1 / (a 190 * a 200) = 19901 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l307_30794


namespace NUMINAMATH_GPT_divisors_form_l307_30710

theorem divisors_form (p n : ℕ) (h_prime : Nat.Prime p) (h_pos : 0 < n) :
  ∃ k : ℕ, (p^n - 1 = 2^k - 1 ∨ p^n - 1 ∣ 48) :=
sorry

end NUMINAMATH_GPT_divisors_form_l307_30710


namespace NUMINAMATH_GPT_fisherman_daily_earnings_l307_30760

theorem fisherman_daily_earnings :
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 :=
by
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  show red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52
  sorry

end NUMINAMATH_GPT_fisherman_daily_earnings_l307_30760


namespace NUMINAMATH_GPT_root_of_linear_eq_l307_30769

variable (a b : ℚ) -- Using rationals for coefficients

-- Define the linear equation
def linear_eq (x : ℚ) : Prop := a * x + b = 0

-- Define the root function
def root_function : ℚ := -b / a

-- State the goal
theorem root_of_linear_eq : linear_eq a b (root_function a b) :=
by
  unfold linear_eq
  unfold root_function
  sorry

end NUMINAMATH_GPT_root_of_linear_eq_l307_30769


namespace NUMINAMATH_GPT_number_of_bricks_l307_30745

theorem number_of_bricks (b1_hours b2_hours combined_hours: ℝ) (reduction_rate: ℝ) (x: ℝ):
  b1_hours = 12 ∧ 
  b2_hours = 15 ∧ 
  combined_hours = 6 ∧ 
  reduction_rate = 15 ∧ 
  (combined_hours * ((x / b1_hours) + (x / b2_hours) - reduction_rate) = x) → 
  x = 1800 :=
by
  sorry

end NUMINAMATH_GPT_number_of_bricks_l307_30745


namespace NUMINAMATH_GPT_find_a_plus_b_l307_30737

open Complex

noncomputable def problem_statement (a b : ℝ) : Prop :=
  (∃ (r1 r2 r3 : ℂ),
     r1 = 1 + I * Real.sqrt 3 ∧
     r2 = 1 - I * Real.sqrt 3 ∧
     r3 = -2 ∧
     (r1 + r2 + r3 = 0) ∧
     (r1 * r2 * r3 = -b) ∧
     (r1 * r2 + r2 * r3 + r3 * r1 = -a))

theorem find_a_plus_b (a b : ℝ) (h : problem_statement a b) : a + b = 8 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l307_30737


namespace NUMINAMATH_GPT_equation_solution_l307_30785

theorem equation_solution (x y : ℝ) (h : x^2 + (1 - y)^2 + (x - y)^2 = (1 / 3)) : 
  x = (1 / 3) ∧ y = (2 / 3) := 
  sorry

end NUMINAMATH_GPT_equation_solution_l307_30785


namespace NUMINAMATH_GPT_stratified_sampling_A_l307_30757

theorem stratified_sampling_A (A B C total_units : ℕ) (propA : A = 400) (propB : B = 300) (propC : C = 200) (units : total_units = 90) :
  let total_families := A + B + C
  let nA := (A * total_units) / total_families
  nA = 40 :=
by
  -- prove the theorem here
  sorry

end NUMINAMATH_GPT_stratified_sampling_A_l307_30757


namespace NUMINAMATH_GPT_train_speed_kmh_l307_30753

theorem train_speed_kmh 
  (L_train : ℝ) (L_bridge : ℝ) (time : ℝ)
  (h_train : L_train = 460)
  (h_bridge : L_bridge = 140)
  (h_time : time = 48) : 
  (L_train + L_bridge) / time * 3.6 = 45 := 
by
  -- Definitions and conditions
  have h_total_dist : L_train + L_bridge = 600 := by sorry
  have h_speed_mps : (L_train + L_bridge) / time = 600 / 48 := by sorry
  have h_speed_mps_simplified : 600 / 48 = 12.5 := by sorry
  have h_speed_kmh : 12.5 * 3.6 = 45 := by sorry
  sorry

end NUMINAMATH_GPT_train_speed_kmh_l307_30753


namespace NUMINAMATH_GPT_min_value_inequality_l307_30730

variable (x y : ℝ)

theorem min_value_inequality (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 1) : 
  ∃ m : ℝ, m = 1 / 4 ∧ (∀ x y, x > 0 → y > 0 → x + y = 1 → (x ^ 2) / (x + 2) + (y ^ 2) / (y + 1) ≥ m) :=
by
  use (1 / 4)
  sorry

end NUMINAMATH_GPT_min_value_inequality_l307_30730


namespace NUMINAMATH_GPT_proof_problem_l307_30704

variable (balls : Finset ℕ) (blackBalls whiteBalls : Finset ℕ)
variable (drawnBalls : Finset ℕ)

/-- There are 6 black balls numbered 1 to 6. -/
def initialBlackBalls : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- There are 4 white balls numbered 7 to 10. -/
def initialWhiteBalls : Finset ℕ := {7, 8, 9, 10}

/-- The total balls (black + white). -/
def totalBalls : Finset ℕ := initialBlackBalls ∪ initialWhiteBalls

/-- The hypergeometric distribution condition for black balls. -/
def hypergeometricBlack : Prop :=
  true  -- placeholder: black balls follow hypergeometric distribution

/-- The probability of drawing 2 white balls is not 1/14. -/
def probDraw2White : Prop :=
  (3 / 7) ≠ (1 / 14)

/-- The probability of the maximum total score (8 points) is 1/14. -/
def probMaxScore : Prop :=
  (15 / 210) = (1 / 14)

/-- Main theorem combining the above conditions for the problem. -/
theorem proof_problem : hypergeometricBlack ∧ probMaxScore :=
by
  unfold hypergeometricBlack
  unfold probMaxScore
  sorry

end NUMINAMATH_GPT_proof_problem_l307_30704


namespace NUMINAMATH_GPT_completing_the_square_sum_l307_30743

theorem completing_the_square_sum :
  ∃ (a b c : ℤ), 64 * (x : ℝ) ^ 2 + 96 * x - 81 = 0 ∧ a > 0 ∧ (8 * x + 6) ^ 2 = c ∧ a = 8 ∧ b = 6 ∧ a + b + c = 131 :=
by
  sorry

end NUMINAMATH_GPT_completing_the_square_sum_l307_30743


namespace NUMINAMATH_GPT_fewer_parking_spaces_on_fourth_level_l307_30791

theorem fewer_parking_spaces_on_fourth_level 
  (spaces_first_level : ℕ) (spaces_second_level : ℕ) (spaces_third_level : ℕ) (spaces_fourth_level : ℕ) 
  (total_spaces_garage : ℕ) (cars_parked : ℕ) 
  (h1 : spaces_first_level = 90)
  (h2 : spaces_second_level = spaces_first_level + 8)
  (h3 : spaces_third_level = spaces_second_level + 12)
  (h4 : total_spaces_garage = 299)
  (h5 : cars_parked = 100)
  (h6 : spaces_first_level + spaces_second_level + spaces_third_level + spaces_fourth_level = total_spaces_garage) :
  spaces_third_level - spaces_fourth_level = 109 := 
by
  sorry

end NUMINAMATH_GPT_fewer_parking_spaces_on_fourth_level_l307_30791


namespace NUMINAMATH_GPT_turtles_on_lonely_island_l307_30709

theorem turtles_on_lonely_island (T : ℕ) (h1 : 60 = 2 * T + 10) : T = 25 := 
by sorry

end NUMINAMATH_GPT_turtles_on_lonely_island_l307_30709


namespace NUMINAMATH_GPT_obtain_angle_10_30_l307_30787

theorem obtain_angle_10_30 (a : ℕ) (h : 100 + a = 135) : a = 35 := 
by sorry

end NUMINAMATH_GPT_obtain_angle_10_30_l307_30787


namespace NUMINAMATH_GPT_kimberly_initial_skittles_l307_30722

theorem kimberly_initial_skittles : 
  ∀ (x : ℕ), (x + 7 = 12) → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_kimberly_initial_skittles_l307_30722


namespace NUMINAMATH_GPT_jordan_annual_income_l307_30798

theorem jordan_annual_income (q : ℝ) (I T : ℝ) 
  (h1 : T = q * 35000 + (q + 3) * (I - 35000))
  (h2 : T = (q + 0.4) * I) : 
  I = 40000 :=
by sorry

end NUMINAMATH_GPT_jordan_annual_income_l307_30798


namespace NUMINAMATH_GPT_total_cakes_served_l307_30766

-- Defining the values for cakes served during lunch and dinner
def lunch_cakes : ℤ := 6
def dinner_cakes : ℤ := 9

-- Stating the theorem that the total number of cakes served today is 15
theorem total_cakes_served : lunch_cakes + dinner_cakes = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_cakes_served_l307_30766


namespace NUMINAMATH_GPT_count_positive_integers_satisfying_inequality_l307_30786

theorem count_positive_integers_satisfying_inequality :
  ∃ n : ℕ, n = 4 ∧ ∀ x : ℕ, (10 < x^2 + 6 * x + 9 ∧ x^2 + 6 * x + 9 < 50) ↔ (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4) := 
by
  sorry

end NUMINAMATH_GPT_count_positive_integers_satisfying_inequality_l307_30786


namespace NUMINAMATH_GPT_purchases_per_customer_l307_30740

noncomputable def number_of_customers_in_cars (num_cars : ℕ) (customers_per_car : ℕ) : ℕ :=
  num_cars * customers_per_car

def total_sales (sports_store_sales : ℕ) (music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

theorem purchases_per_customer {num_cars : ℕ} {customers_per_car : ℕ} {sports_store_sales : ℕ} {music_store_sales : ℕ}
    (h1 : num_cars = 10)
    (h2 : customers_per_car = 5)
    (h3 : sports_store_sales = 20)
    (h4: music_store_sales = 30) :
    (total_sales sports_store_sales music_store_sales / number_of_customers_in_cars num_cars customers_per_car) = 1 :=
by
  sorry

end NUMINAMATH_GPT_purchases_per_customer_l307_30740


namespace NUMINAMATH_GPT_calligraphy_prices_max_brushes_l307_30711

theorem calligraphy_prices 
  (x y : ℝ)
  (h1 : 40 * x + 100 * y = 280)
  (h2 : 30 * x + 200 * y = 260) :
  x = 6 ∧ y = 0.4 := 
by sorry

theorem max_brushes 
  (m : ℝ)
  (h_budget : 6 * m + 0.4 * (200 - m) ≤ 360) :
  m ≤ 50 :=
by sorry

end NUMINAMATH_GPT_calligraphy_prices_max_brushes_l307_30711


namespace NUMINAMATH_GPT_adiabatic_compression_work_l307_30762

noncomputable def adiabatic_work (p1 V1 V2 k : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) : ℝ :=
  (p1 * V1) / (k - 1) * (1 - (V1 / V2)^(k - 1))

theorem adiabatic_compression_work (p1 V1 V2 k W : ℝ) (h₁ : k > 1) (h₂ : V1 > 0) (h₃ : V2 > 0) :
  W = adiabatic_work p1 V1 V2 k h₁ h₂ h₃ :=
sorry

end NUMINAMATH_GPT_adiabatic_compression_work_l307_30762


namespace NUMINAMATH_GPT_Tim_transactions_l307_30712

theorem Tim_transactions
  (Mabel_Monday : ℕ)
  (Mabel_Tuesday : ℕ := Mabel_Monday + Mabel_Monday / 10)
  (Anthony_Tuesday : ℕ := 2 * Mabel_Tuesday)
  (Cal_Tuesday : ℕ := (2 * Anthony_Tuesday) / 3)
  (Jade_Tuesday : ℕ := Cal_Tuesday + 17)
  (Isla_Wednesday : ℕ := Mabel_Tuesday + Cal_Tuesday - 12)
  (Tim_Thursday : ℕ := (Jade_Tuesday + Isla_Wednesday) * 3 / 2)
  : Tim_Thursday = 614 := by sorry

end NUMINAMATH_GPT_Tim_transactions_l307_30712


namespace NUMINAMATH_GPT_problem1_problem2_l307_30719

-- Problem 1: Evaluating an integer arithmetic expression
theorem problem1 : (1 * (-8)) - (-6) + (-3) = -5 := 
by
  sorry

-- Problem 2: Evaluating a mixed arithmetic expression with rational numbers and decimals
theorem problem2 : (5 / 13) - 3.7 + (8 / 13) - (-1.7) = -1 := 
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l307_30719


namespace NUMINAMATH_GPT_center_polar_coordinates_l307_30716

-- Assuming we have a circle defined in polar coordinates
def polar_circle_center (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ + 2 * Real.sin θ

-- The goal is to prove that the center of this circle has the polar coordinates (sqrt 2, π/4)
theorem center_polar_coordinates : ∃ ρ θ, polar_circle_center ρ θ ∧ ρ = Real.sqrt 2 ∧ θ = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_center_polar_coordinates_l307_30716


namespace NUMINAMATH_GPT_product_three_consecutive_integers_divisible_by_six_l307_30749

theorem product_three_consecutive_integers_divisible_by_six
  (n : ℕ) (h_pos : 0 < n) : ∃ k : ℕ, (n - 1) * n * (n + 1) = 6 * k :=
by sorry

end NUMINAMATH_GPT_product_three_consecutive_integers_divisible_by_six_l307_30749


namespace NUMINAMATH_GPT_right_vs_oblique_prism_similarities_and_differences_l307_30770

-- Definitions of Prisms and their properties
structure Prism where
  parallel_bases : Prop
  congruent_bases : Prop
  parallelogram_faces : Prop

structure RightPrism extends Prism where
  rectangular_faces : Prop
  perpendicular_sides : Prop

structure ObliquePrism extends Prism where
  non_perpendicular_sides : Prop

theorem right_vs_oblique_prism_similarities_and_differences 
  (p1 : RightPrism) (p2 : ObliquePrism) : 
    (p1.parallel_bases ↔ p2.parallel_bases) ∧ 
    (p1.congruent_bases ↔ p2.congruent_bases) ∧ 
    (p1.parallelogram_faces ↔ p2.parallelogram_faces) ∧
    (p1.rectangular_faces ∧ p1.perpendicular_sides ↔ p2.non_perpendicular_sides) := 
by 
  sorry

end NUMINAMATH_GPT_right_vs_oblique_prism_similarities_and_differences_l307_30770


namespace NUMINAMATH_GPT_cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l307_30718

theorem cos_eq_neg_1_over_4_of_sin_eq_1_over_4
  (α : ℝ)
  (h : Real.sin (α + π / 3) = 1 / 4) :
  Real.cos (α + 5 * π / 6) = -1 / 4 :=
sorry

end NUMINAMATH_GPT_cos_eq_neg_1_over_4_of_sin_eq_1_over_4_l307_30718


namespace NUMINAMATH_GPT_tetrahedron_has_six_edges_l307_30729

-- Define what a tetrahedron is
inductive Tetrahedron : Type
| mk : Tetrahedron

-- Define the number of edges of a Tetrahedron
def edges_of_tetrahedron (t : Tetrahedron) : Nat := 6

theorem tetrahedron_has_six_edges (t : Tetrahedron) : edges_of_tetrahedron t = 6 := 
by
  sorry

end NUMINAMATH_GPT_tetrahedron_has_six_edges_l307_30729


namespace NUMINAMATH_GPT_integer_solution_count_l307_30738

theorem integer_solution_count {a b c d : ℤ} (h : a ≠ b) :
  (∀ x y : ℤ, (x + a * y + c) * (x + b * y + d) = 2 →
    ∃ a b : ℤ, (|a - b| = 1 ∨ (|a - b| = 2 ∧ (d - c) % 2 = 1))) :=
sorry

end NUMINAMATH_GPT_integer_solution_count_l307_30738


namespace NUMINAMATH_GPT_tile_floor_covering_l307_30727

theorem tile_floor_covering (n : ℕ) (h1 : 10 < n) (h2 : n < 20) (h3 : ∃ x, 9 * x = n^2) : n = 12 ∨ n = 15 ∨ n = 18 := by
  sorry

end NUMINAMATH_GPT_tile_floor_covering_l307_30727


namespace NUMINAMATH_GPT_convert_255_to_base8_l307_30726

-- Define the conversion function from base 10 to base 8
def base10_to_base8 (n : ℕ) : ℕ :=
  let d2 := n / 64
  let r2 := n % 64
  let d1 := r2 / 8
  let r1 := r2 % 8
  d2 * 100 + d1 * 10 + r1

-- Define the specific number and base in the conditions
def num10 : ℕ := 255
def base8_result : ℕ := 377

-- The theorem stating the proof problem
theorem convert_255_to_base8 : base10_to_base8 num10 = base8_result :=
by
  -- You would provide the proof steps here
  sorry

end NUMINAMATH_GPT_convert_255_to_base8_l307_30726


namespace NUMINAMATH_GPT_inequality_square_l307_30788

theorem inequality_square (a b : ℝ) (h : a > |b|) : a^2 > b^2 :=
sorry

end NUMINAMATH_GPT_inequality_square_l307_30788


namespace NUMINAMATH_GPT_cricket_team_age_difference_l307_30715

theorem cricket_team_age_difference :
  ∀ (captain_age : ℕ) (keeper_age : ℕ) (team_size : ℕ) (team_average_age : ℕ) (remaining_size : ℕ),
  captain_age = 28 →
  keeper_age = captain_age + 3 →
  team_size = 11 →
  team_average_age = 25 →
  remaining_size = team_size - 2 →
  (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 24 →
  team_average_age - (team_size * team_average_age - (captain_age + keeper_age)) / remaining_size = 1 :=
by
  intros captain_age keeper_age team_size team_average_age remaining_size h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_cricket_team_age_difference_l307_30715


namespace NUMINAMATH_GPT_city_rentals_cost_per_mile_l307_30799

theorem city_rentals_cost_per_mile (x : ℝ)
  (h₁ : 38.95 + 150 * x = 41.95 + 150 * 0.29) :
  x = 0.31 :=
by sorry

end NUMINAMATH_GPT_city_rentals_cost_per_mile_l307_30799


namespace NUMINAMATH_GPT_candy_difference_l307_30761

-- Defining the conditions as Lean hypotheses
variable (R K B M : ℕ)

-- Given conditions
axiom h1 : K = 4
axiom h2 : B = M - 6
axiom h3 : M = R + 2
axiom h4 : K = B + 2

-- Prove that Robert gets 2 more pieces of candy than Kate
theorem candy_difference : R - K = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_candy_difference_l307_30761


namespace NUMINAMATH_GPT_rubiks_cube_repeats_l307_30750

theorem rubiks_cube_repeats (num_positions : ℕ) (H : num_positions = 43252003274489856000) 
  (moves : ℕ → ℕ) : 
  ∃ n, ∃ m, (∀ P, moves n = moves m → P = moves 0) :=
by
  sorry

end NUMINAMATH_GPT_rubiks_cube_repeats_l307_30750


namespace NUMINAMATH_GPT_find_f_inv_128_l307_30774

open Function

theorem find_f_inv_128 (f : ℕ → ℕ) 
  (h₀ : f 5 = 2) 
  (h₁ : ∀ x, f (2 * x) = 2 * f x) : 
  f⁻¹' {128} = {320} :=
by
  sorry

end NUMINAMATH_GPT_find_f_inv_128_l307_30774


namespace NUMINAMATH_GPT_fourth_derivative_of_function_y_l307_30731

noncomputable def log_base_3 (x : ℝ) : ℝ := (Real.log x) / (Real.log 3)

noncomputable def function_y (x : ℝ) : ℝ := (log_base_3 x) / (x ^ 2)

theorem fourth_derivative_of_function_y (x : ℝ) (h : 0 < x) : 
    (deriv^[4] (fun x => function_y x)) x = (-154 + 120 * (Real.log x)) / (x ^ 6 * Real.log 3) :=
  sorry

end NUMINAMATH_GPT_fourth_derivative_of_function_y_l307_30731


namespace NUMINAMATH_GPT_deductive_reasoning_correct_l307_30758

theorem deductive_reasoning_correct :
  (∀ (s : ℕ), s = 3 ↔
    (s == 1 → DeductiveReasoningGeneralToSpecific ∧
     s == 2 → alwaysCorrect ∧
     s == 3 → InFormOfSyllogism ∧
     s == 4 → ConclusionDependsOnPremisesAndForm)) :=
sorry

end NUMINAMATH_GPT_deductive_reasoning_correct_l307_30758


namespace NUMINAMATH_GPT_seq_b_arithmetic_diff_seq_a_general_term_l307_30707

variable {n : ℕ}

def seq_a (a : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n / (a n + 2)

def seq_b (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = 1 / a n

theorem seq_b_arithmetic_diff (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_a : seq_a a) (h_b : seq_b a b) :
  ∀ n, b (n + 1) - b n = 1 / 2 :=
by
  sorry

theorem seq_a_general_term (a : ℕ → ℝ) (h_a : seq_a a) :
  ∀ n, a n = 2 / (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_seq_b_arithmetic_diff_seq_a_general_term_l307_30707


namespace NUMINAMATH_GPT_real_solution_to_abs_equation_l307_30765

theorem real_solution_to_abs_equation :
  (∃! x : ℝ, |x - 2| = |x - 4| + |x - 6| + |x - 8|) :=
by
  sorry

end NUMINAMATH_GPT_real_solution_to_abs_equation_l307_30765


namespace NUMINAMATH_GPT_total_students_proof_l307_30705

variable (studentsA studentsB : ℕ) (ratioAtoB : ℕ := 3/2)
variable (percentA percentB : ℕ := 10/100)
variable (diffPercent : ℕ := 20/100)
variable (extraStudentsInA : ℕ := 190)
variable (totalStudentsB : ℕ := 650)

theorem total_students_proof :
  (studentsB = totalStudentsB) ∧ 
  ((percentA * studentsA - diffPercent * studentsB = extraStudentsInA) ∧
  (studentsA / studentsB = ratioAtoB)) →
  (studentsA + studentsB = 1625) :=
by
  sorry

end NUMINAMATH_GPT_total_students_proof_l307_30705


namespace NUMINAMATH_GPT_fixed_point_of_f_l307_30797

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-1) + 4

theorem fixed_point_of_f (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) : f a 1 = 5 :=
by
  unfold f
  -- Skip the proof; it will be filled in the subsequent steps
  sorry

end NUMINAMATH_GPT_fixed_point_of_f_l307_30797


namespace NUMINAMATH_GPT_milk_drinks_on_weekdays_l307_30723

-- Defining the number of boxes Lolita drinks on a weekday as a variable W
variable (W : ℕ)

-- Condition: Lolita drinks 30 boxes of milk per week.
axiom total_milk_per_week : 5 * W + 2 * W + 3 * W = 30

-- Proof (Statement) that Lolita drinks 15 boxes of milk on weekdays.
theorem milk_drinks_on_weekdays : 5 * W = 15 :=
by {
  -- Use the given axiom to derive the solution
  sorry
}

end NUMINAMATH_GPT_milk_drinks_on_weekdays_l307_30723


namespace NUMINAMATH_GPT_remaining_stock_weight_l307_30790

def green_beans_weight : ℕ := 80
def rice_weight : ℕ := green_beans_weight - 30
def sugar_weight : ℕ := green_beans_weight - 20
def flour_weight : ℕ := 2 * sugar_weight
def lentils_weight : ℕ := flour_weight - 10

def rice_remaining_weight : ℕ := rice_weight - rice_weight / 3
def sugar_remaining_weight : ℕ := sugar_weight - sugar_weight / 5
def flour_remaining_weight : ℕ := flour_weight - flour_weight / 4
def lentils_remaining_weight : ℕ := lentils_weight - lentils_weight / 6

def total_remaining_weight : ℕ :=
  rice_remaining_weight + sugar_remaining_weight + flour_remaining_weight + lentils_remaining_weight + green_beans_weight

theorem remaining_stock_weight :
  total_remaining_weight = 343 := by
  sorry

end NUMINAMATH_GPT_remaining_stock_weight_l307_30790


namespace NUMINAMATH_GPT_trigonometric_identity_l307_30724

theorem trigonometric_identity (x : ℝ) (h : Real.tan x = 3) : 1 / (Real.sin x ^ 2 - 2 * Real.cos x ^ 2) = 10 / 7 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l307_30724


namespace NUMINAMATH_GPT_rope_cut_into_pieces_l307_30772

theorem rope_cut_into_pieces (length_of_rope_cm : ℕ) (num_equal_pieces : ℕ) (length_equal_piece_mm : ℕ) (length_remaining_piece_mm : ℕ) 
  (h1 : length_of_rope_cm = 1165) (h2 : num_equal_pieces = 150) (h3 : length_equal_piece_mm = 75) (h4 : length_remaining_piece_mm = 100) :
  (num_equal_pieces * length_equal_piece_mm + (11650 - num_equal_pieces * length_equal_piece_mm) / length_remaining_piece_mm = 154) :=
by
  sorry

end NUMINAMATH_GPT_rope_cut_into_pieces_l307_30772


namespace NUMINAMATH_GPT_positive_difference_l307_30776

-- Define the conditions given in the problem
def conditions (x y : ℝ) : Prop :=
  x + y = 40 ∧ 3 * y - 4 * x = 20

-- The theorem to prove
theorem positive_difference (x y : ℝ) (h : conditions x y) : abs (y - x) = 11.42 :=
by
  sorry -- proof omitted

end NUMINAMATH_GPT_positive_difference_l307_30776


namespace NUMINAMATH_GPT_cylinder_lateral_surface_area_l307_30767

theorem cylinder_lateral_surface_area (r l : ℝ) (A : ℝ) (h_r : r = 1) (h_l : l = 2) : A = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cylinder_lateral_surface_area_l307_30767


namespace NUMINAMATH_GPT_max_ab_bc_cd_da_l307_30752

theorem max_ab_bc_cd_da (a b c d : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d) (h_sum : a + b + c + d = 200) :
  ab + bc + cd + da ≤ 10000 :=
by sorry

end NUMINAMATH_GPT_max_ab_bc_cd_da_l307_30752


namespace NUMINAMATH_GPT_total_questions_on_test_l307_30754

theorem total_questions_on_test :
  ∀ (correct incorrect score : ℕ),
  (score = correct - 2 * incorrect) →
  (score = 76) →
  (correct = 92) →
  (correct + incorrect = 100) :=
by
  intros correct incorrect score grading_system score_eq correct_eq
  sorry

end NUMINAMATH_GPT_total_questions_on_test_l307_30754


namespace NUMINAMATH_GPT_apple_pies_count_l307_30771

def total_pies := 13
def pecan_pies := 4
def pumpkin_pies := 7
def apple_pies := total_pies - pecan_pies - pumpkin_pies

theorem apple_pies_count : apple_pies = 2 := by
  sorry

end NUMINAMATH_GPT_apple_pies_count_l307_30771


namespace NUMINAMATH_GPT_largest_divisor_consecutive_odd_squares_l307_30736

theorem largest_divisor_consecutive_odd_squares (m n : ℤ) 
  (hmn : m = n + 2) 
  (hodd_m : m % 2 = 1) 
  (hodd_n : n % 2 = 1) 
  (horder : n < m) : ∃ k : ℤ, m^2 - n^2 = 8 * k :=
by 
  sorry

end NUMINAMATH_GPT_largest_divisor_consecutive_odd_squares_l307_30736


namespace NUMINAMATH_GPT_total_marbles_l307_30789

theorem total_marbles :
  let marbles_second_bowl := 600
  let marbles_first_bowl := (3/4) * marbles_second_bowl
  let total_marbles := marbles_first_bowl + marbles_second_bowl
  total_marbles = 1050 := by
  sorry -- proof skipped

end NUMINAMATH_GPT_total_marbles_l307_30789


namespace NUMINAMATH_GPT_smallest_divisor_after_391_l307_30792

theorem smallest_divisor_after_391 (m : ℕ) (h₁ : 1000 ≤ m ∧ m < 10000) (h₂ : Even m) (h₃ : 391 ∣ m) : 
  ∃ d, d > 391 ∧ d ∣ m ∧ ∀ e, 391 < e ∧ e ∣ m → e ≥ d :=
by
  use 441
  sorry

end NUMINAMATH_GPT_smallest_divisor_after_391_l307_30792


namespace NUMINAMATH_GPT_solve_x_if_alpha_beta_eq_8_l307_30768

variable (x : ℝ)

def alpha (x : ℝ) := 4 * x + 9
def beta (x : ℝ) := 9 * x + 6

theorem solve_x_if_alpha_beta_eq_8 (hx : alpha (beta x) = 8) : x = (-25 / 36) :=
by
  sorry

end NUMINAMATH_GPT_solve_x_if_alpha_beta_eq_8_l307_30768


namespace NUMINAMATH_GPT_find_number_l307_30793

theorem find_number (x : ℝ) (h : 20 / x = 0.8) : x = 25 := 
by
  sorry

end NUMINAMATH_GPT_find_number_l307_30793


namespace NUMINAMATH_GPT_attendance_second_concert_l307_30780

-- Define the given conditions
def attendance_first_concert : ℕ := 65899
def additional_people : ℕ := 119

-- Prove the number of people at the second concert
theorem attendance_second_concert : 
  attendance_first_concert + additional_people = 66018 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_attendance_second_concert_l307_30780


namespace NUMINAMATH_GPT_original_number_is_seven_l307_30744

theorem original_number_is_seven (x : ℤ) (h : 3 * x - 6 = 15) : x = 7 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_seven_l307_30744


namespace NUMINAMATH_GPT_part1_part2_l307_30728

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}

/-- Given sequence properties -/
axiom h1 : a 1 = 5
axiom h2 : ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1) + 2^n - 1

/-- Part (I): Proving the sequence is arithmetic -/
theorem part1 (n : ℕ) : ∃ d, (∀ m ≥ 1, (a (m + 1) - 1) / 2^(m + 1) - (a m - 1) / 2^m = d)
∧ ((a 1 - 1) / 2 = 2) := sorry

/-- Part (II): Sum of the first n terms -/
theorem part2 (n : ℕ) : S n = n * 2^(n+1) := sorry

end NUMINAMATH_GPT_part1_part2_l307_30728


namespace NUMINAMATH_GPT_product_of_numbers_l307_30702

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 157) : x * y = 22 := 
by 
  sorry

end NUMINAMATH_GPT_product_of_numbers_l307_30702


namespace NUMINAMATH_GPT_quadratic_distinct_roots_l307_30714

theorem quadratic_distinct_roots
  (a b c : ℝ)
  (h1 : 5 * a + 3 * b + 2 * c = 0)
  (h2 : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1 ^ 2 + b * x1 + c = 0) ∧ (a * x2 ^ 2 + b * x2 + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_l307_30714


namespace NUMINAMATH_GPT_exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l307_30763

theorem exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4
  (n : ℕ) (hn₁ : Odd n) (hn₂ : 0 < n) :
  (∃ (a b : ℕ), 0 < a ∧ 0 < b ∧ (4 : ℚ) / n = 1 / a + 1 / b) ↔
  ∃ p, p ∣ n ∧ Prime p ∧ p % 4 = 1 :=
by
  sorry

end NUMINAMATH_GPT_exist_a_b_if_and_only_if_n_prime_divisor_1_mod_4_l307_30763


namespace NUMINAMATH_GPT_range_of_m_for_distinct_real_roots_l307_30778

theorem range_of_m_for_distinct_real_roots (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 - 4 * x1 - m = 0 ∧ x2^2 - 4 * x2 - m = 0) ↔ m > -4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_for_distinct_real_roots_l307_30778


namespace NUMINAMATH_GPT_exists_linear_eq_exactly_m_solutions_l307_30703

theorem exists_linear_eq_exactly_m_solutions (m : ℕ) (hm : 0 < m) :
  ∃ (a b c : ℤ), ∀ (x y : ℕ), a * x + b * y = c ↔
    (1 ≤ x ∧ 1 ≤ y ∧ x + y = m + 1) :=
by
  sorry

end NUMINAMATH_GPT_exists_linear_eq_exactly_m_solutions_l307_30703


namespace NUMINAMATH_GPT_coords_of_point_P_l307_30756

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2)

theorem coords_of_point_P :
  ∀ (a : ℝ), 0 < a ∧ a ≠ 1 → ∃ P : ℝ × ℝ, (P = (1, -2) ∧ ∀ y, f (f a (-2)) y = y) :=
by
  sorry

end NUMINAMATH_GPT_coords_of_point_P_l307_30756


namespace NUMINAMATH_GPT_sqrt_of_9_l307_30775

theorem sqrt_of_9 : Real.sqrt 9 = 3 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_of_9_l307_30775


namespace NUMINAMATH_GPT_number_of_students_passed_l307_30795

theorem number_of_students_passed (total_students : ℕ) (failure_frequency : ℝ) (h1 : total_students = 1000) (h2 : failure_frequency = 0.4) : 
  (total_students - (total_students * failure_frequency)) = 600 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_passed_l307_30795


namespace NUMINAMATH_GPT_sin_cos_inequality_l307_30721

open Real

theorem sin_cos_inequality (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * π) :
  (sin (x - π / 6) > cos x) ↔ (π / 3 < x ∧ x < 4 * π / 3) :=
by sorry

end NUMINAMATH_GPT_sin_cos_inequality_l307_30721


namespace NUMINAMATH_GPT_smaller_of_x_y_l307_30779

theorem smaller_of_x_y (x y a b c : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < x) (h4 : 0 < y)
  (h5 : x * y = c) (h6 : x^2 - b * x + a * y = 0) : min x y = c / a :=
by sorry

end NUMINAMATH_GPT_smaller_of_x_y_l307_30779
