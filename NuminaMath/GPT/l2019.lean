import Mathlib

namespace son_l2019_201986

theorem son's_present_age
  (S F : ℤ)
  (h1 : F = S + 45)
  (h2 : F + 10 = 4 * (S + 10))
  (h3 : S + 15 = 2 * S) :
  S = 15 :=
by
  sorry

end son_l2019_201986


namespace number_of_unit_squares_in_50th_ring_l2019_201961

def nth_ring_unit_squares (n : ℕ) : ℕ :=
  8 * n

-- Statement to prove
theorem number_of_unit_squares_in_50th_ring : nth_ring_unit_squares 50 = 400 :=
by
  -- Proof steps (skip with sorry)
  sorry

end number_of_unit_squares_in_50th_ring_l2019_201961


namespace solution_y_values_l2019_201923
-- Import the necessary libraries

-- Define the system of equations and the necessary conditions
def equation1 (x : ℝ) := x^2 - 6*x + 8 = 0
def equation2 (x y : ℝ) := 2*x - y = 6

-- The main theorem to be proven
theorem solution_y_values : ∃ x1 x2 y1 y2 : ℝ, 
  (equation1 x1 ∧ equation1 x2 ∧ equation2 x1 y1 ∧ equation2 x2 y2 ∧ 
  y1 = 2 ∧ y2 = -2) :=
by
  -- Use the provided solutions in the problem statement
  use 4, 2, 2, -2
  sorry  -- The details of the proof are omitted.

end solution_y_values_l2019_201923


namespace value_of_coins_is_77_percent_l2019_201909

theorem value_of_coins_is_77_percent :
  let pennies := 2 * 1  -- value of two pennies in cents
  let nickel := 5       -- value of one nickel in cents
  let dimes := 2 * 10   -- value of two dimes in cents
  let half_dollar := 50 -- value of one half-dollar in cents
  let total_cents := pennies + nickel + dimes + half_dollar
  let dollar_in_cents := 100
  (total_cents / dollar_in_cents) * 100 = 77 :=
by
  sorry

end value_of_coins_is_77_percent_l2019_201909


namespace heather_oranges_l2019_201943

theorem heather_oranges (initial_oranges additional_oranges : ℝ) (h1 : initial_oranges = 60.5) (h2 : additional_oranges = 35.8) :
  initial_oranges + additional_oranges = 96.3 :=
by
  -- sorry is used to indicate the proof is omitted
  sorry

end heather_oranges_l2019_201943


namespace find_divisor_l2019_201952

theorem find_divisor (x : ℕ) (h : 144 = (x * 13) + 1) : x = 11 := by
  sorry

end find_divisor_l2019_201952


namespace sqrt_meaningful_range_l2019_201902

theorem sqrt_meaningful_range (x : ℝ) : 2 * x - 6 ≥ 0 ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_range_l2019_201902


namespace find_scooters_l2019_201968

variables (b t s : ℕ)

theorem find_scooters (h1 : b + t + s = 13) (h2 : 2 * b + 3 * t + 2 * s = 30) : s = 9 :=
sorry

end find_scooters_l2019_201968


namespace coordinates_with_respect_to_origin_l2019_201966

theorem coordinates_with_respect_to_origin :
  ∀ (point : ℝ × ℝ), point = (3, -2) → point = (3, -2) := by
  intro point h
  exact h

end coordinates_with_respect_to_origin_l2019_201966


namespace product_of_c_values_l2019_201954

theorem product_of_c_values :
  ∃ (c1 c2 : ℕ), (c1 > 0 ∧ c2 > 0) ∧
  (∃ (x1 x2 : ℚ), (7 * x1^2 + 15 * x1 + c1 = 0) ∧ (7 * x2^2 + 15 * x2 + c2 = 0)) ∧
  (c1 * c2 = 16) :=
sorry

end product_of_c_values_l2019_201954


namespace consecutive_sums_permutations_iff_odd_l2019_201947

theorem consecutive_sums_permutations_iff_odd (n : ℕ) (h : n ≥ 2) :
  (∃ (a b : Fin n → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ n) ∧ (∀ i, 1 ≤ b i ∧ b i ≤ n) ∧
    ∃ N, ∀ i, a i + b i = N + i) ↔ (Odd n) :=
by
  sorry

end consecutive_sums_permutations_iff_odd_l2019_201947


namespace ladder_base_distance_l2019_201959

noncomputable def length_of_ladder : ℝ := 8.5
noncomputable def height_on_wall : ℝ := 7.5

theorem ladder_base_distance (x : ℝ) (h : x ^ 2 + height_on_wall ^ 2 = length_of_ladder ^ 2) :
  x = 4 :=
by sorry

end ladder_base_distance_l2019_201959


namespace remainder_of_P_div_by_D_is_333_l2019_201901

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := 8 * x^4 - 18 * x^3 + 27 * x^2 - 14 * x - 30

-- Define the divisor D(x) and simplify it, but this is not necessary for the theorem statement.
-- def D (x : ℝ) : ℝ := 4 * x - 12  

-- Prove the remainder is 333 when x = 3
theorem remainder_of_P_div_by_D_is_333 : P 3 = 333 := by
  sorry

end remainder_of_P_div_by_D_is_333_l2019_201901


namespace combined_savings_after_four_weeks_l2019_201956

-- Definitions based on problem conditions
def hourly_wage : ℕ := 10
def daily_hours : ℕ := 10
def days_per_week : ℕ := 5
def weeks : ℕ := 4

def robby_saving_ratio : ℚ := 2/5
def jaylene_saving_ratio : ℚ := 3/5
def miranda_saving_ratio : ℚ := 1/2

-- Definitions derived from the conditions
def daily_earnings : ℕ := hourly_wage * daily_hours
def total_working_days : ℕ := days_per_week * weeks
def monthly_earnings : ℕ := daily_earnings * total_working_days

def robby_savings : ℚ := robby_saving_ratio * monthly_earnings
def jaylene_savings : ℚ := jaylene_saving_ratio * monthly_earnings
def miranda_savings : ℚ := miranda_saving_ratio * monthly_earnings

def total_savings : ℚ := robby_savings + jaylene_savings + miranda_savings

-- The main theorem to prove
theorem combined_savings_after_four_weeks :
  total_savings = 3000 := by sorry

end combined_savings_after_four_weeks_l2019_201956


namespace david_marks_in_physics_l2019_201960

theorem david_marks_in_physics 
  (english_marks mathematics_marks chemistry_marks biology_marks : ℕ)
  (num_subjects : ℕ)
  (average_marks : ℕ)
  (h1 : english_marks = 81)
  (h2 : mathematics_marks = 65)
  (h3 : chemistry_marks = 67)
  (h4 : biology_marks = 85)
  (h5 : num_subjects = 5)
  (h6 : average_marks = 76) :
  ∃ physics_marks : ℕ, physics_marks = 82 :=
by
  sorry

end david_marks_in_physics_l2019_201960


namespace polynomial_divisible_by_7_l2019_201924

theorem polynomial_divisible_by_7 (n : ℤ) : 7 ∣ ((n + 7)^2 - n^2) :=
sorry

end polynomial_divisible_by_7_l2019_201924


namespace min_value_inequality_l2019_201920

theorem min_value_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 3) :
  (1 / x) + (4 / y) + (9 / z) ≥ 12 := 
sorry

end min_value_inequality_l2019_201920


namespace factorize_expression_l2019_201921

theorem factorize_expression (a b : ℝ) : a * b^2 - 9 * a = a * (b + 3) * (b - 3) :=
by 
  sorry

end factorize_expression_l2019_201921


namespace intersection_of_A_B_find_a_b_l2019_201988

-- Lean 4 definitions based on the given conditions
def setA (x : ℝ) : Prop := 4 - x^2 > 0
def setB (x : ℝ) (y : ℝ) : Prop := y = Real.log (-x^2 + 2*x + 3) ∧ -x^2 + 2*x + 3 > 0

-- Prove the intersection of sets A and B
theorem intersection_of_A_B :
  {x : ℝ | setA x} ∩ {x : ℝ | ∃ y : ℝ, setB x y} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

-- On the roots of the quadratic equation and solution interval of inequality
theorem find_a_b (a b : ℝ) :
  (∀ x : ℝ, 2 * x^2 + a * x + b < 0 ↔ -3 < x ∧ x < 1) →
  a = 4 ∧ b = -6 :=
by
  sorry

end intersection_of_A_B_find_a_b_l2019_201988


namespace Jungkook_has_bigger_number_l2019_201962

theorem Jungkook_has_bigger_number : (3 + 6) > 4 :=
by {
  sorry
}

end Jungkook_has_bigger_number_l2019_201962


namespace midpoint_product_zero_l2019_201939

theorem midpoint_product_zero (x y : ℝ) :
  let A := (2, 6)
  let B := (x, y)
  let C := (4, 3)
  (C = ((2 + x) / 2, (6 + y) / 2)) → (x * y = 0) := by
  intros
  sorry

end midpoint_product_zero_l2019_201939


namespace number_of_outcomes_l2019_201976

-- Define the conditions
def students : Nat := 4
def events : Nat := 3

-- Define the problem: number of possible outcomes for the champions
theorem number_of_outcomes : students ^ events = 64 :=
by sorry

end number_of_outcomes_l2019_201976


namespace base_number_of_exponentiation_l2019_201916

theorem base_number_of_exponentiation (n : ℕ) (some_number : ℕ) (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = some_number^22) (h2 : n = 21) : some_number = 4 :=
  sorry

end base_number_of_exponentiation_l2019_201916


namespace book_E_chapters_l2019_201934

def total_chapters: ℕ := 97
def chapters_A: ℕ := 17
def chapters_B: ℕ := chapters_A + 5
def chapters_C: ℕ := chapters_B - 7
def chapters_D: ℕ := chapters_C * 2
def chapters_sum : ℕ := chapters_A + chapters_B + chapters_C + chapters_D

theorem book_E_chapters :
  total_chapters - chapters_sum = 13 :=
by
  sorry

end book_E_chapters_l2019_201934


namespace average_value_of_T_l2019_201982

noncomputable def expected_value_T : ℕ := 22

theorem average_value_of_T (boys girls : ℕ) (boy_pair girl_pair : Prop) (T : ℕ) :
  boys = 9 → girls = 15 →
  boy_pair ∧ girl_pair →
  T = expected_value_T :=
by
  intros h_boys h_girls h_pairs
  sorry

end average_value_of_T_l2019_201982


namespace exists_bijection_l2019_201975

-- Define the non-negative integers set
def N_0 := {n : ℕ // n ≥ 0}

-- Translation of the equivalent proof statement into Lean
theorem exists_bijection (f : N_0 → N_0) :
  (∀ m n : N_0, f ⟨3 * m.val * n.val + m.val + n.val, sorry⟩ = 
   ⟨4 * (f m).val * (f n).val + (f m).val + (f n).val, sorry⟩) :=
sorry

end exists_bijection_l2019_201975


namespace charlie_delta_four_products_l2019_201903

noncomputable def charlie_delta_purchase_ways : ℕ := 1363

theorem charlie_delta_four_products :
  let cakes := 6
  let cookies := 4
  let total := cakes + cookies
  ∃ ways : ℕ, ways = charlie_delta_purchase_ways :=
by
  sorry

end charlie_delta_four_products_l2019_201903


namespace total_number_of_trees_l2019_201915

-- Definitions of the conditions
def side_length : ℝ := 100
def trees_per_sq_meter : ℝ := 4

-- Calculations based on the conditions
def area_of_street : ℝ := side_length * side_length
def area_of_forest : ℝ := 3 * area_of_street

-- The statement to prove
theorem total_number_of_trees : 
  trees_per_sq_meter * area_of_forest = 120000 := 
sorry

end total_number_of_trees_l2019_201915


namespace bag_contains_fifteen_balls_l2019_201945

theorem bag_contains_fifteen_balls 
  (r b : ℕ) 
  (h1 : r + b = 15) 
  (h2 : (r * (r - 1)) / 210 = 1 / 21) 
  : r = 4 := 
sorry

end bag_contains_fifteen_balls_l2019_201945


namespace min_max_calculation_l2019_201983

theorem min_max_calculation
  (p q r s : ℝ)
  (h1 : p + q + r + s = 8)
  (h2 : p^2 + q^2 + r^2 + s^2 = 20) :
  -32 ≤ 5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ∧
  5 * (p^3 + q^3 + r^3 + s^3) - (p^4 + q^4 + r^4 + s^4) ≤ 12 :=
sorry

end min_max_calculation_l2019_201983


namespace product_of_binomials_l2019_201908

-- Definition of the binomials
def binomial1 (x : ℝ) : ℝ := 4 * x - 3
def binomial2 (x : ℝ) : ℝ := x + 7

-- The theorem to be proved
theorem product_of_binomials (x : ℝ) : 
  binomial1 x * binomial2 x = 4 * x^2 + 25 * x - 21 :=
by
  sorry

end product_of_binomials_l2019_201908


namespace ratio_expression_l2019_201906

theorem ratio_expression (a b c : ℝ) (ha : a / b = 20) (hb : b / c = 10) : (a + b) / (b + c) = 210 / 11 := by
  sorry

end ratio_expression_l2019_201906


namespace smallest_prime_perf_sqr_minus_eight_l2019_201946

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_perf_sqr_minus_eight (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k^2 - 8

theorem smallest_prime_perf_sqr_minus_eight :
  ∃ (n : ℕ), is_prime n ∧ is_perf_sqr_minus_eight n ∧ (∀ m : ℕ, is_prime m ∧ is_perf_sqr_minus_eight m → n ≤ m) :=
sorry

end smallest_prime_perf_sqr_minus_eight_l2019_201946


namespace max_min_diff_eq_l2019_201955

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x^2 + 2*x + 2) - Real.sqrt (x^2 - 3*x + 3)

theorem max_min_diff_eq : 
  (∀ x : ℝ, ∃ max min : ℝ, max = Real.sqrt (8 - Real.sqrt 3) ∧ min = -Real.sqrt (8 - Real.sqrt 3) ∧ 
  (max - min = 2 * Real.sqrt (8 - Real.sqrt 3))) :=
sorry

end max_min_diff_eq_l2019_201955


namespace negation_of_p_l2019_201925

theorem negation_of_p :
  (¬ (∀ x > 0, (x+1)*Real.exp x > 1)) ↔ 
  (∃ x ≤ 0, (x+1)*Real.exp x ≤ 1) :=
sorry

end negation_of_p_l2019_201925


namespace weighted_average_correct_l2019_201965

noncomputable def weightedAverage := 
  (5 * (3/5 : ℝ) + 3 * (4/9 : ℝ) + 8 * 0.45 + 4 * 0.067) / (5 + 3 + 8 + 4)

theorem weighted_average_correct :
  weightedAverage = 0.41 :=
by
  sorry

end weighted_average_correct_l2019_201965


namespace no_real_solution_l2019_201991

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 6

-- Lean statement: prove that the equation x^2 - 4x + 6 = 0 has no real solution
theorem no_real_solution : ¬ ∃ x : ℝ, f x = 0 :=
sorry

end no_real_solution_l2019_201991


namespace tiles_walked_on_l2019_201957

/-- 
A park has a rectangular shape with a width of 13 feet and a length of 19 feet.
Square-shaped tiles of dimension 1 foot by 1 foot cover the entire area.
The gardener walks in a straight line from one corner of the rectangle to the opposite corner.
One specific tile in the path is not to be stepped on. 
Prove that the number of tiles the gardener walks on is 30.
-/
theorem tiles_walked_on (width length gcd_width_length tiles_to_avoid : ℕ)
  (h_width : width = 13)
  (h_length : length = 19)
  (h_gcd : gcd width length = 1)
  (h_tiles_to_avoid : tiles_to_avoid = 1) : 
  (width + length - gcd_width_length - tiles_to_avoid = 30) := 
by
  sorry

end tiles_walked_on_l2019_201957


namespace son_work_time_l2019_201913

theorem son_work_time (M S : ℝ) 
  (hM : M = 1 / 4)
  (hCombined : M + S = 1 / 3) : 
  S = 1 / 12 :=
by
  sorry

end son_work_time_l2019_201913


namespace problem1_problem2_l2019_201904

-- Problem 1
theorem problem1 (x : ℚ) (h : x = -1/3) : 6 * x^2 + 5 * x^2 - 2 * (3 * x - 2 * x^2) = 11 / 3 :=
by sorry

-- Problem 2
theorem problem2 (a b : ℚ) (ha : a = -2) (hb : b = -1) : 5 * a^2 - a * b - 2 * (3 * a * b - (a * b - 2 * a^2)) = -6 :=
by sorry

end problem1_problem2_l2019_201904


namespace ratio_perimeter_triangle_square_l2019_201998

/-
  Suppose a square piece of paper with side length 4 units is folded in half diagonally.
  The folded paper is then cut along the fold, producing two right-angled triangles.
  We need to prove that the ratio of the perimeter of one of the triangles to the perimeter of the original square is (1/2) + (sqrt 2 / 4).
-/
theorem ratio_perimeter_triangle_square:
  let side_length := 4
  let triangle_leg := side_length
  let hypotenuse := Real.sqrt (triangle_leg ^ 2 + triangle_leg ^ 2)
  let perimeter_triangle := triangle_leg + triangle_leg + hypotenuse
  let perimeter_square := 4 * side_length
  let ratio := perimeter_triangle / perimeter_square
  ratio = (1 / 2) + (Real.sqrt 2 / 4) :=
by
  sorry

end ratio_perimeter_triangle_square_l2019_201998


namespace cost_of_items_l2019_201919

namespace GardenCost

variables (B T C : ℝ)

/-- Given conditions defining the cost relationships and combined cost,
prove the specific costs of bench, table, and chair. -/
theorem cost_of_items
  (h1 : T + B + C = 650)
  (h2 : T = 2 * B - 50)
  (h3 : C = 1.5 * B - 25) :
  B = 161.11 ∧ T = 272.22 ∧ C = 216.67 :=
sorry

end GardenCost

end cost_of_items_l2019_201919


namespace integer_between_sqrt3_add1_and_sqrt11_l2019_201970

theorem integer_between_sqrt3_add1_and_sqrt11 :
  (∀ x, (1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) → (2 < Real.sqrt 3 + 1 ∧ Real.sqrt 3 + 1 < 3) ∧ (3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4) ∧ x = 3) :=
by
  sorry

end integer_between_sqrt3_add1_and_sqrt11_l2019_201970


namespace system_solution_equation_solution_l2019_201949

-- Proof problem for the first system of equations
theorem system_solution (x y : ℝ) : 
  (2 * x + 3 * y = 8) ∧ (3 * x - 5 * y = -7) → (x = 1 ∧ y = 2) :=
by sorry

-- Proof problem for the second equation
theorem equation_solution (x : ℝ) : 
  ((x - 2) / (x + 2) - 12 / (x^2 - 4) = 1) → (x = -1) :=
by sorry

end system_solution_equation_solution_l2019_201949


namespace minimum_value_omega_l2019_201997

variable (f : ℝ → ℝ) (ω ϕ T : ℝ) (x : ℝ)
variable (h_zero : 0 < ω) (h_phi_range : 0 < ϕ ∧ ϕ < π)
variable (h_period : T = 2 * π / ω)
variable (h_f_period : f T = sqrt 3 / 2)
variable (h_zero_of_f : f (π / 9) = 0)
variable (h_f_def : ∀ x, f x = cos (ω * x + ϕ))

theorem minimum_value_omega : ω = 3 := by sorry

end minimum_value_omega_l2019_201997


namespace base5_addition_correct_l2019_201922

-- Definitions to interpret base-5 numbers
def base5_to_base10 (n : List ℕ) : ℕ :=
  n.reverse.foldl (λ acc d => acc * 5 + d) 0

-- Conditions given in the problem
def num1 : ℕ := base5_to_base10 [2, 0, 1, 4]  -- (2014)_5 in base-10
def num2 : ℕ := base5_to_base10 [2, 2, 3]    -- (223)_5 in base-10

-- Statement to prove
theorem base5_addition_correct :
  base5_to_base10 ([2, 0, 1, 4]) + base5_to_base10 ([2, 2, 3]) = base5_to_base10 ([2, 2, 4, 2]) :=
by
  -- Proof goes here
  sorry

#print axioms base5_addition_correct

end base5_addition_correct_l2019_201922


namespace find_x_values_l2019_201985

def f (x : ℝ) : ℝ := x^2 - 4 * x

theorem find_x_values :
  {x : ℝ | f (f x) = f x} = {0, 4, 5, -1} :=
by
  sorry

end find_x_values_l2019_201985


namespace incorrect_statement_for_proportional_function_l2019_201993

theorem incorrect_statement_for_proportional_function (x y : ℝ) : y = -5 * x →
  ¬ (∀ x, (x > 0 → y > 0) ∧ (x < 0 → y < 0)) :=
by
  sorry

end incorrect_statement_for_proportional_function_l2019_201993


namespace system1_solution_l2019_201907

variable (x y : ℝ)

theorem system1_solution :
  (3 * x - y = -1) ∧ (x + 2 * y = 9) ↔ (x = 1) ∧ (y = 4) := by
  sorry

end system1_solution_l2019_201907


namespace four_digit_numbers_condition_l2019_201910

theorem four_digit_numbers_condition :
  ∃ (N : Nat), (1000 ≤ N ∧ N < 10000) ∧
               (∃ x a : Nat, N = 1000 * a + x ∧ x = 200 * a ∧ 1 ≤ a ∧ a ≤ 4) :=
by
  sorry

end four_digit_numbers_condition_l2019_201910


namespace jade_initial_pieces_l2019_201927

theorem jade_initial_pieces (n w l p : ℕ) (hn : n = 11) (hw : w = 7) (hl : l = 23) (hp : p = n * w + l) : p = 100 :=
by
  sorry

end jade_initial_pieces_l2019_201927


namespace geometric_series_q_and_S6_l2019_201992

theorem geometric_series_q_and_S6 (a : ℕ → ℝ) (q : ℝ) (S_6 : ℝ) 
  (ha_pos : ∀ n, a n > 0)
  (ha2 : a 2 = 3)
  (ha4 : a 4 = 27) :
  q = 3 ∧ S_6 = 364 :=
by
  sorry

end geometric_series_q_and_S6_l2019_201992


namespace alpha_plus_beta_eq_118_l2019_201964

theorem alpha_plus_beta_eq_118 (α β : ℝ) (h : ∀ x : ℝ, (x - α) / (x + β) = (x^2 - 96 * x + 2209) / (x^2 + 63 * x - 3969)) : α + β = 118 :=
by
  sorry

end alpha_plus_beta_eq_118_l2019_201964


namespace circle_chord_intersect_zero_l2019_201948

noncomputable def circle_product (r : ℝ) : ℝ :=
  let O := (0, 0)
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B)

theorem circle_chord_intersect_zero (r : ℝ) :
  let A := (r, 0)
  let B := (-r, 0)
  let C := (0, r)
  let D := (0, -r)
  let P := (0, 0)
  (dist A P) * (dist P B) = 0 :=
by sorry

end circle_chord_intersect_zero_l2019_201948


namespace find_b_l2019_201900

theorem find_b (a b : ℤ) (h1 : 3 * a + 1 = 4) (h2 : b - a = 1) : b = 2 :=
sorry

end find_b_l2019_201900


namespace polynomial_identity_l2019_201941

theorem polynomial_identity (x : ℝ) :
  (x - 2)^5 + 5 * (x - 2)^4 + 10 * (x - 2)^3 + 10 * (x - 2)^2 + 5 * (x - 2) + 1 = (x - 1)^5 := 
by 
  sorry

end polynomial_identity_l2019_201941


namespace sum_a_b_eq_negative_one_l2019_201929

theorem sum_a_b_eq_negative_one 
  (a b : ℝ) 
  (h1 : ∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - a * x - b < 0)
  (h2 : ∀ x : ℝ, x^2 - a * x - b = 0 → x = 2 ∨ x = 3) :
  a + b = -1 := 
sorry

end sum_a_b_eq_negative_one_l2019_201929


namespace solve_system_of_equations_l2019_201940

theorem solve_system_of_equations:
  ∃ (x y : ℚ), 3 * x + 4 * y = 16 ∧ 5 * x - 6 * y = 33 ∧ x = 6 ∧ y = -1/2 :=
by
  sorry

end solve_system_of_equations_l2019_201940


namespace max_length_MN_l2019_201984

theorem max_length_MN (p : ℝ) (h a b c r : ℝ)
  (h_perimeter : a + b + c = 2 * p)
  (h_tangent : r = (a * h) / (2 * p))
  (h_parallel : ∀ h r : ℝ, ∃ k : ℝ, MN = k * (1 - 2 * r / h)) :
  ∀ k : ℝ, MN = (p / 4) :=
sorry

end max_length_MN_l2019_201984


namespace measure_Z_is_19_6_l2019_201953

def measure_angle_X : ℝ := 72
def measure_Y (measure_Z : ℝ) : ℝ := 4 * measure_Z + 10
def angle_sum_condition (measure_Z : ℝ) : Prop :=
  measure_angle_X + (measure_Y measure_Z) + measure_Z = 180

theorem measure_Z_is_19_6 :
  ∃ measure_Z : ℝ, measure_Z = 19.6 ∧ angle_sum_condition measure_Z :=
by
  sorry

end measure_Z_is_19_6_l2019_201953


namespace linear_system_solution_l2019_201990

theorem linear_system_solution (a b : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : 2 * a + 3 * b = 4) : 
  a - b = 1 := 
by
  sorry

end linear_system_solution_l2019_201990


namespace simplify_and_evaluate_l2019_201973

theorem simplify_and_evaluate (x : ℝ) (h : x = Real.sin (Real.pi / 6)) :
  (1 - 2 / (x - 1)) / ((x - 3) / (x^2 - 1)) = 3 / 2 :=
by
  -- simplify and evaluate the expression given the condition on x
  sorry

end simplify_and_evaluate_l2019_201973


namespace miranda_monthly_savings_l2019_201987

noncomputable def total_cost := 260
noncomputable def sister_contribution := 50
noncomputable def months := 3

theorem miranda_monthly_savings : 
  (total_cost - sister_contribution) / months = 70 := 
by
  sorry

end miranda_monthly_savings_l2019_201987


namespace parallel_lines_m_l2019_201981

theorem parallel_lines_m (m : ℝ) :
  (∀ (x y : ℝ), 2 * x + (m + 1) * y + 4 = 0) ∧ (∀ (x y : ℝ), m * x + 3 * y - 2 = 0) →
  (m = -3 ∨ m = 2) :=
by
  sorry

end parallel_lines_m_l2019_201981


namespace find_third_number_l2019_201963

theorem find_third_number (A B C : ℝ) (h1 : (A + B + C) / 3 = 48) (h2 : (A + B) / 2 = 56) : C = 32 :=
by sorry

end find_third_number_l2019_201963


namespace pounds_per_ton_l2019_201926

theorem pounds_per_ton (weight_pounds : ℕ) (weight_tons : ℕ) (h_weight : weight_pounds = 6000) (h_tons : weight_tons = 3) : 
  weight_pounds / weight_tons = 2000 :=
by
  sorry

end pounds_per_ton_l2019_201926


namespace trains_at_starting_positions_after_2016_minutes_l2019_201972

-- Definitions corresponding to conditions
def round_trip_minutes (line: String) : Nat :=
  if line = "red" then 14
  else if line = "blue" then 16
  else if line = "green" then 18
  else 0

def is_multiple_of (n m : Nat) : Prop :=
  n % m = 0

-- Formalize the statement to be proven
theorem trains_at_starting_positions_after_2016_minutes :
  ∀ (line: String), 
  line = "red" ∨ line = "blue" ∨ line = "green" →
  is_multiple_of 2016 (round_trip_minutes line) :=
by
  intro line h
  cases h with
  | inl red =>
    sorry
  | inr hb =>
    cases hb with
    | inl blue =>
      sorry
    | inr green =>
      sorry

end trains_at_starting_positions_after_2016_minutes_l2019_201972


namespace range_of_m_l2019_201931

theorem range_of_m (m : ℝ) : (∀ (x : ℝ), |3 - x| + |5 + x| > m) → m < 8 :=
sorry

end range_of_m_l2019_201931


namespace total_vacations_and_classes_l2019_201995

def kelvin_classes := 90
def grant_vacations := 4 * kelvin_classes
def total := grant_vacations + kelvin_classes

theorem total_vacations_and_classes :
  total = 450 :=
by
  sorry

end total_vacations_and_classes_l2019_201995


namespace minimum_ab_condition_l2019_201958

open Int

theorem minimum_ab_condition 
  (a b : ℕ) 
  (h_pos : 0 < a ∧ 0 < b)
  (h_div7_ab_sum : ab * (a + b) % 7 ≠ 0) 
  (h_div7_expansion : ((a + b) ^ 7 - a ^ 7 - b ^ 7) % 7 = 0) : 
  ab = 18 :=
sorry

end minimum_ab_condition_l2019_201958


namespace gcd_45_75_l2019_201971

theorem gcd_45_75 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_l2019_201971


namespace set_P_equals_set_interval_l2019_201999

def A : Set ℝ := {x | x < 5}
def B : Set ℝ := {x | x <= 1 ∨ x >= 3}
def P : Set ℝ := {x | x ∈ A ∧ ¬ (x ∈ A ∧ x ∈ B)}

theorem set_P_equals_set_interval :
  P = {x | 1 < x ∧ x < 3} :=
sorry

end set_P_equals_set_interval_l2019_201999


namespace kimmie_earnings_l2019_201942

theorem kimmie_earnings (K : ℚ) (h : (1/2 : ℚ) * K + (1/3 : ℚ) * K = 375) : K = 450 := 
by
  sorry

end kimmie_earnings_l2019_201942


namespace time_b_used_l2019_201928

noncomputable def time_b_used_for_proof : ℚ :=
  let C : ℚ := 1
  let C_a : ℚ := 1 / 4 * C
  let t_a : ℚ := 15
  let p_a : ℚ := 1 / 3
  let p_b : ℚ := 2 / 3
  let ratio : ℚ := (C_a * t_a) / ((C - C_a) * (t_a * p_a / p_b))
  t_a * p_a / p_b

theorem time_b_used : time_b_used_for_proof = 10 / 3 := by
  sorry

end time_b_used_l2019_201928


namespace smallest_of_powers_l2019_201974

theorem smallest_of_powers :
  (2:ℤ)^(55) < (3:ℤ)^(44) ∧ (2:ℤ)^(55) < (5:ℤ)^(33) ∧ (2:ℤ)^(55) < (6:ℤ)^(22) := by
  sorry

end smallest_of_powers_l2019_201974


namespace hannah_late_times_l2019_201980

variable (hourly_rate : ℝ)
variable (hours_worked : ℝ)
variable (dock_per_late : ℝ)
variable (actual_pay : ℝ)

theorem hannah_late_times (h1 : hourly_rate = 30)
                          (h2 : hours_worked = 18)
                          (h3 : dock_per_late = 5)
                          (h4 : actual_pay = 525) :
  ((hourly_rate * hours_worked - actual_pay) / dock_per_late) = 3 := 
by
  sorry

end hannah_late_times_l2019_201980


namespace second_number_is_30_l2019_201951

-- Definitions from the conditions
def second_number (x : ℕ) := x
def first_number (x : ℕ) := 2 * x
def third_number (x : ℕ) := (2 * x) / 3
def sum_of_numbers (x : ℕ) := first_number x + second_number x + third_number x

-- Lean statement
theorem second_number_is_30 (x : ℕ) (h1 : sum_of_numbers x = 110) : x = 30 :=
by
  sorry

end second_number_is_30_l2019_201951


namespace find_all_n_l2019_201977

theorem find_all_n (n : ℕ) : 
  (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % n = 0) ↔ (∃ j : ℕ, n = 3^j) :=
by 
  -- proof goes here
  sorry

end find_all_n_l2019_201977


namespace parallel_lines_m_eq_neg4_l2019_201905

theorem parallel_lines_m_eq_neg4 (m : ℝ) (h1 : (m-2) ≠ -m) 
  (h2 : (m-2) / 3 = -m / (m + 2)) : m = -4 :=
sorry

end parallel_lines_m_eq_neg4_l2019_201905


namespace rachel_math_homework_pages_l2019_201979

theorem rachel_math_homework_pages (M : ℕ) 
  (h1 : 23 = M + (M + 3)) : M = 10 :=
by {
  sorry
}

end rachel_math_homework_pages_l2019_201979


namespace number_of_pairs_sold_l2019_201996

-- Define the conditions
def total_amount_made : ℝ := 588
def average_price_per_pair : ℝ := 9.8

-- The theorem we want to prove
theorem number_of_pairs_sold : total_amount_made / average_price_per_pair = 60 := 
by sorry

end number_of_pairs_sold_l2019_201996


namespace part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l2019_201918

def f (a x : ℝ) : ℝ := a * x ^ 2 + (1 - a) * x + a - 2

theorem part1 (a : ℝ) : (∀ x : ℝ, f a x ≥ -2) ↔ a ≥ 1/3 :=
sorry

theorem part2_case1 (a : ℝ) (ha : a = 0) : ∀ x : ℝ, f a x < a - 1 ↔ x < 1 :=
sorry

theorem part2_case2 (a : ℝ) (ha : a > 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (-1 / a < x ∧ x < 1) :=
sorry

theorem part2_case3_1 (a : ℝ) (ha : a = -1) : ∀ x : ℝ, (f a x < a - 1) ↔ x ≠ 1 :=
sorry

theorem part2_case3_2 (a : ℝ) (ha : -1 < a ∧ a < 0) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > -1 / a ∨ x < 1) :=
sorry

theorem part2_case3_3 (a : ℝ) (ha : a < -1) : ∀ x : ℝ, (f a x < a - 1) ↔ (x > 1 ∨ x < -1 / a) :=
sorry

end part1_part2_case1_part2_case2_part2_case3_1_part2_case3_2_part2_case3_3_l2019_201918


namespace alpha_div_beta_is_rational_l2019_201935

noncomputable def alpha_is_multiple (α : ℝ) (k : ℕ) : Prop :=
  ∃ k : ℕ, α = k * (2 * Real.pi / 1996)

noncomputable def beta_is_multiple (β : ℝ) (m : ℕ) : Prop :=
  β ≠ 0 ∧ ∃ m : ℕ, β = m * (2 * Real.pi / 1996)

theorem alpha_div_beta_is_rational (α β : ℝ) (k m : ℕ)
  (hα : alpha_is_multiple α k) (hβ : beta_is_multiple β m) :
  ∃ r : ℚ, α / β = r := by
    sorry

end alpha_div_beta_is_rational_l2019_201935


namespace highlighter_difference_l2019_201936

theorem highlighter_difference :
  ∃ (P : ℕ), 7 + P + (P + 5) = 40 ∧ P - 7 = 7 :=
by
  sorry

end highlighter_difference_l2019_201936


namespace max_b_value_l2019_201937

theorem max_b_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a * b = (2 * a - b) / (2 * a + 3 * b)) : b ≤ 1 / 3 :=
  sorry

end max_b_value_l2019_201937


namespace intersect_x_axis_once_l2019_201933

theorem intersect_x_axis_once (k : ℝ) : 
  (∀ x : ℝ, (k - 3) * x^2 + 2 * x + 1 = 0 → x = 0) → (k = 3 ∨ k = 4) :=
by
  intro h
  sorry

end intersect_x_axis_once_l2019_201933


namespace median_inequality_l2019_201932

variables {α : ℝ} (A B C M : Point) (a b c : ℝ)

-- Definitions and conditions
def isTriangle (A B C : Point) : Prop := -- definition of triangle
sorry

def isMedian (A B C M : Point) : Prop := -- definition of median
sorry

-- Statement we want to prove
theorem median_inequality (h1 : isTriangle A B C) (h2 : isMedian A B C M) :
  2 * AM ≥ (b + c) * Real.cos (α / 2) :=
sorry

end median_inequality_l2019_201932


namespace ratio_of_dimensions_128_l2019_201930

noncomputable def volume128 (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_of_dimensions_128 (w l h : ℕ) (h_volume : volume128 w l h) : 
  ∃ wratio lratio, (w / l = wratio) ∧ (w / h = lratio) :=
sorry

end ratio_of_dimensions_128_l2019_201930


namespace space_shuttle_speed_conversion_l2019_201912

-- Define the given conditions
def speed_km_per_sec : ℕ := 6  -- Speed in km/s
def seconds_per_hour : ℕ := 3600  -- Seconds in an hour

-- Define the computed speed in km/hr
def expected_speed_km_per_hr : ℕ := 21600  -- Expected speed in km/hr

-- The main theorem statement to be proven
theorem space_shuttle_speed_conversion : speed_km_per_sec * seconds_per_hour = expected_speed_km_per_hr := by
  sorry

end space_shuttle_speed_conversion_l2019_201912


namespace find_number_l2019_201950

theorem find_number (x : ℕ) (h : (x + 720) / 125 = 7392 / 462) : x = 1280 :=
sorry

end find_number_l2019_201950


namespace bob_total_profit_l2019_201914

-- Define the given inputs
def n_dogs : ℕ := 2
def c_dog : ℝ := 250.00
def n_puppies : ℕ := 6
def c_food_vac : ℝ := 500.00
def c_ad : ℝ := 150.00
def p_puppy : ℝ := 350.00

-- The statement to prove
theorem bob_total_profit : 
  (n_puppies * p_puppy - (n_dogs * c_dog + c_food_vac + c_ad)) = 950.00 :=
by
  sorry

end bob_total_profit_l2019_201914


namespace general_term_l2019_201967

noncomputable def seq (n : ℕ) : ℤ :=
  if n = 0 then 0 else
  if n = 1 then -1 else
  if n % 2 = 0 then (2 * 2 ^ (n / 2 - 1) - 1) / 3 else 
  (-2)^(n - n / 2) / 3 - 1

-- Conditions
def condition1 : Prop := seq 1 = -1
def condition2 : Prop := seq 2 > seq 1
def condition3 (n : ℕ) : Prop := |seq (n + 1) - seq n| = 2^n
def condition4 : Prop := ∀ m, seq (2*m + 1) > seq (2*m - 1)
def condition5 : Prop := ∀ m, seq (2*m) < seq (2*m + 2)

-- The theorem stating the general term of the sequence
theorem general_term (n : ℕ) :
  condition1 →
  condition2 →
  (∀ n, condition3 n) →
  condition4 →
  condition5 →
  seq n = ( (-2)^n - 1) / 3 :=
by
  sorry

end general_term_l2019_201967


namespace rectangle_error_percent_deficit_l2019_201994

theorem rectangle_error_percent_deficit (L W : ℝ) (p : ℝ) 
    (h1 : L > 0) (h2 : W > 0)
    (h3 : 1.05 * (1 - p) = 1.008) :
    p = 0.04 :=
by
  sorry

end rectangle_error_percent_deficit_l2019_201994


namespace find_x_l2019_201989

theorem find_x (x : ℝ) (h1 : 0 < x) (h2 : ⌈x⌉ * x = 220) : x = 14.67 :=
sorry

end find_x_l2019_201989


namespace intersection_of_M_and_N_l2019_201911

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0, 1} := 
by sorry

end intersection_of_M_and_N_l2019_201911


namespace football_goals_l2019_201938

variable (A : ℚ) (G : ℚ)

theorem football_goals (A G : ℚ) 
    (h1 : G = 14 * A)
    (h2 : G + 3 = (A + 0.08) * 15) :
    G = 25.2 :=
by
  -- Proof here
  sorry

end football_goals_l2019_201938


namespace ball_travel_approximately_80_l2019_201917

noncomputable def ball_travel_distance : ℝ :=
  let h₀ := 20
  let ratio := 2 / 3
  h₀ + -- first descent
  h₀ * ratio + -- first ascent
  h₀ * ratio + -- second descent
  h₀ * ratio^2 + -- second ascent
  h₀ * ratio^2 + -- third descent
  h₀ * ratio^3 + -- third ascent
  h₀ * ratio^3 + -- fourth descent
  h₀ * ratio^4 -- fourth ascent

theorem ball_travel_approximately_80 :
  abs (ball_travel_distance - 80) < 1 :=
sorry

end ball_travel_approximately_80_l2019_201917


namespace general_term_a_n_l2019_201978

open BigOperators

variable {a : ℕ → ℝ}  -- The sequence a_n
variable {S : ℕ → ℝ}  -- The sequence sum S_n

-- Define the sum of the first n terms:
def seq_sum (a : ℕ → ℝ) (n : ℕ) := ∑ k in Finset.range (n + 1), a k

theorem general_term_a_n (h : ∀ n : ℕ, S n = 2 ^ n - 1) (n : ℕ) : a n = 2 ^ (n - 1) :=
by
  sorry

end general_term_a_n_l2019_201978


namespace total_workers_l2019_201969

-- Definitions for the conditions in the problem
def avg_salary_all : ℝ := 8000
def num_technicians : ℕ := 7
def avg_salary_technicians : ℝ := 18000
def avg_salary_non_technicians : ℝ := 6000

-- Main theorem stating the total number of workers
theorem total_workers (W : ℕ) :
  (7 * avg_salary_technicians + (W - 7) * avg_salary_non_technicians = W * avg_salary_all) → W = 42 :=
by
  sorry

end total_workers_l2019_201969


namespace remainder_when_divided_by_5_l2019_201944

theorem remainder_when_divided_by_5 
  (n : ℕ) 
  (h : n % 10 = 7) : 
  n % 5 = 2 := 
by 
  sorry

end remainder_when_divided_by_5_l2019_201944
