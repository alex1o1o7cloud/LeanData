import Mathlib

namespace triangle_angle_from_cosine_relation_l650_65079

theorem triangle_angle_from_cosine_relation (a b c : ℝ) (A B C : ℝ) :
  (0 < a ∧ 0 < b ∧ 0 < c) →
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  (A + B + C = π) →
  (2 * b * Real.cos B = a * Real.cos C + c * Real.cos A) →
  B = π / 3 := by sorry

end triangle_angle_from_cosine_relation_l650_65079


namespace concatenated_number_divisible_by_1980_l650_65091

def concatenated_number : ℕ :=
  -- Definition of the number A as described in the problem
  sorry

theorem concatenated_number_divisible_by_1980 :
  1980 ∣ concatenated_number :=
by
  sorry

end concatenated_number_divisible_by_1980_l650_65091


namespace inequality_proof_l650_65046

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a / (a + b)) * ((a + 2*b) / (a + 3*b)) < Real.sqrt (a / (a + 4*b)) := by
  sorry

end inequality_proof_l650_65046


namespace girls_in_circle_l650_65097

theorem girls_in_circle (total : ℕ) (holding_boys_hand : ℕ) (holding_girls_hand : ℕ) 
  (h1 : total = 40)
  (h2 : holding_boys_hand = 22)
  (h3 : holding_girls_hand = 30) :
  ∃ (girls : ℕ), girls = 24 ∧ 
    girls * 2 = holding_girls_hand * 2 + holding_boys_hand + holding_girls_hand - total :=
by
  sorry

end girls_in_circle_l650_65097


namespace m_range_l650_65036

-- Define propositions p and q
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∃ x : ℝ, 3^x - m + 1 ≤ 0

-- Define the theorem
theorem m_range :
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) →
  (∀ m : ℝ, 1 < m ∧ m ≤ 2 ↔ q m ∧ ¬(p m)) :=
sorry

end m_range_l650_65036


namespace similar_right_triangles_l650_65016

theorem similar_right_triangles (y : ℝ) : 
  (15 : ℝ) / 12 = y / 10 → y = 12.5 := by
sorry

end similar_right_triangles_l650_65016


namespace percent_of_percent_l650_65065

theorem percent_of_percent (x : ℝ) (h : x ≠ 0) : (0.3 * 0.7 * x) / x = 0.21 := by
  sorry

end percent_of_percent_l650_65065


namespace max_fourth_number_l650_65027

def numbers : Finset Nat := {39, 41, 44, 45, 47, 52, 55}

def is_valid_arrangement (arr : List Nat) : Prop :=
  arr.toFinset = numbers ∧
  ∀ i, i + 2 < arr.length → (arr[i]! + arr[i+1]! + arr[i+2]!) % 3 = 0

theorem max_fourth_number :
  ∃ (arr : List Nat), is_valid_arrangement arr ∧
    ∀ (other_arr : List Nat), is_valid_arrangement other_arr →
      arr[3]! ≥ other_arr[3]! ∧ arr[3]! = 47 :=
sorry

end max_fourth_number_l650_65027


namespace quadratic_factorization_l650_65018

theorem quadratic_factorization (C D : ℤ) :
  (∀ y : ℚ, 15 * y^2 - 56 * y + 48 = (C * y - 16) * (D * y - 3)) →
  C * D + C = 20 := by
  sorry

end quadratic_factorization_l650_65018


namespace inequality_solution_set_l650_65081

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (x - a) * (x - 1/a) > 0} = {x : ℝ | a < x ∧ x < 1/a} := by sorry

end inequality_solution_set_l650_65081


namespace quadratic_two_distinct_roots_l650_65039

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 5 = 0 ∧ x₂^2 + m*x₂ - 5 = 0 :=
by
  sorry

end quadratic_two_distinct_roots_l650_65039


namespace max_performances_l650_65058

/-- Represents a performance in the theater festival -/
structure Performance :=
  (students : Finset ℕ)
  (size_eq_six : students.card = 6)

/-- The theater festival -/
structure TheaterFestival :=
  (num_students : ℕ)
  (num_students_eq_twelve : num_students = 12)
  (performances : Finset Performance)
  (common_students : Performance → Performance → Finset ℕ)
  (common_students_le_two : ∀ p1 p2 : Performance, p1 ≠ p2 → (common_students p1 p2).card ≤ 2)

/-- The theorem stating the maximum number of performances -/
theorem max_performances (festival : TheaterFestival) : 
  festival.performances.card ≤ 4 :=
sorry

end max_performances_l650_65058


namespace min_sum_abs_l650_65083

theorem min_sum_abs (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 := by
  sorry

end min_sum_abs_l650_65083


namespace bart_money_theorem_l650_65017

theorem bart_money_theorem :
  ∃ m : ℕ, m > 0 ∧ ∀ n : ℕ, n ≥ m → ∃ a b : ℕ, n = 17 * a + 19 * b := by
  sorry

end bart_money_theorem_l650_65017


namespace smallest_positive_angle_2015_l650_65063

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem smallest_positive_angle_2015 :
  ∃! θ : ℝ, 0 ≤ θ ∧ θ < 360 ∧ same_terminal_side θ (-2015) ∧
  ∀ φ, 0 ≤ φ ∧ φ < 360 ∧ same_terminal_side φ (-2015) → θ ≤ φ :=
by sorry

end smallest_positive_angle_2015_l650_65063


namespace bin_game_expected_win_l650_65021

/-- The number of yellow balls in the bin -/
def yellow_balls : ℕ := 7

/-- The number of blue balls in the bin -/
def blue_balls : ℕ := 3

/-- The amount won when drawing a yellow ball -/
def yellow_win : ℚ := 3

/-- The amount lost when drawing a blue ball -/
def blue_loss : ℚ := 1

/-- The expected amount won from playing the game -/
def expected_win : ℚ := 1

/-- Theorem stating that the expected amount won is 1 dollar
    given the specified number of yellow and blue balls and win/loss amounts -/
theorem bin_game_expected_win :
  (yellow_balls * yellow_win + blue_balls * (-blue_loss)) / (yellow_balls + blue_balls) = expected_win :=
sorry

end bin_game_expected_win_l650_65021


namespace money_combination_l650_65028

theorem money_combination (raquel nataly tom sam : ℝ) : 
  tom = (1/4) * nataly →
  nataly = 3 * raquel →
  sam = 2 * nataly →
  raquel = 40 →
  tom + raquel + nataly + sam = 430 :=
by sorry

end money_combination_l650_65028


namespace equation_solution_l650_65090

theorem equation_solution (m n x : ℝ) (hm : m > 0) (hn : n < 0) :
  (x - m)^2 - (x - n)^2 = (m - n)^2 → x = m :=
by sorry

end equation_solution_l650_65090


namespace feuerbach_circle_equation_l650_65003

/-- The Feuerbach circle (nine-point circle) of a triangle -/
def feuerbach_circle (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * c * (p.1^2 + p.2^2) - (a + b) * c * p.1 + (a * b - c^2) * p.2 = 0}

/-- The vertices of the triangle -/
def triangle_vertices (a b c : ℝ) : Set (ℝ × ℝ) :=
  {(a, 0), (b, 0), (0, c)}

theorem feuerbach_circle_equation (a b c : ℝ) (h : c ≠ 0) :
  ∃ (circle : Set (ℝ × ℝ)), circle = feuerbach_circle a b c ∧
  (∀ (p : ℝ × ℝ), p ∈ circle ↔ 2 * c * (p.1^2 + p.2^2) - (a + b) * c * p.1 + (a * b - c^2) * p.2 = 0) :=
sorry

end feuerbach_circle_equation_l650_65003


namespace largest_sum_is_ten_l650_65057

/-- A structure representing a set of five positive integers -/
structure FiveIntegers where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  d : ℕ+
  e : ℕ+

/-- The property that the sum of the five integers equals their product -/
def hasSumProductProperty (x : FiveIntegers) : Prop :=
  x.a + x.b + x.c + x.d + x.e = x.a * x.b * x.c * x.d * x.e

/-- The sum of the five integers -/
def sum (x : FiveIntegers) : ℕ :=
  x.a + x.b + x.c + x.d + x.e

/-- The theorem stating that (1, 1, 1, 2, 5) has the largest sum among all valid sets -/
theorem largest_sum_is_ten :
  ∀ x : FiveIntegers, hasSumProductProperty x → sum x ≤ 10 :=
sorry

end largest_sum_is_ten_l650_65057


namespace sum_p_q_equals_expected_p_condition_q_condition_l650_65034

/-- A linear function p(x) satisfying p(-1) = -2 -/
def p (x : ℝ) : ℝ := 4 * x - 2

/-- A quadratic function q(x) satisfying q(1) = 3 -/
def q (x : ℝ) : ℝ := 1.5 * x^2 - 1.5

/-- Theorem stating that p(x) + q(x) = 1.5x^2 + 4x - 3.5 -/
theorem sum_p_q_equals_expected : 
  ∀ x : ℝ, p x + q x = 1.5 * x^2 + 4 * x - 3.5 := by
  sorry

/-- Verification of the conditions -/
theorem p_condition : p (-1) = -2 := by
  sorry

theorem q_condition : q 1 = 3 := by
  sorry

end sum_p_q_equals_expected_p_condition_q_condition_l650_65034


namespace percentage_problem_l650_65061

theorem percentage_problem (x : ℝ) (h : 0.20 * x = 1000) : 1.20 * x = 6000 := by
  sorry

end percentage_problem_l650_65061


namespace even_function_symmetric_about_y_axis_l650_65080

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_symmetric_about_y_axis (f : ℝ → ℝ) (h : even_function f) :
  ∀ x y, f x = y ↔ f (-x) = y :=
sorry

end even_function_symmetric_about_y_axis_l650_65080


namespace expression_evaluation_l650_65085

theorem expression_evaluation :
  let x : ℝ := -3
  let numerator := 5 + x * (5 + x) - 5^2
  let denominator := x - 5 + x^2
  numerator / denominator = -26 := by sorry

end expression_evaluation_l650_65085


namespace cos_minus_sin_for_point_l650_65096

theorem cos_minus_sin_for_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = Real.sqrt 3 ∧ r * Real.sin α = -1) →
  Real.cos α - Real.sin α = (Real.sqrt 3 + 1) / 2 := by
sorry

end cos_minus_sin_for_point_l650_65096


namespace multiplicative_inverse_600_mod_4901_l650_65098

theorem multiplicative_inverse_600_mod_4901 :
  ∃ n : ℕ, n < 4901 ∧ (600 * n) % 4901 = 1 ∧ n = 3196 := by sorry

end multiplicative_inverse_600_mod_4901_l650_65098


namespace complementary_events_l650_65073

-- Define the sample space for throwing 3 coins
def CoinOutcome := Fin 2 × Fin 2 × Fin 2

-- Define the event "No more than one head"
def NoMoreThanOneHead (outcome : CoinOutcome) : Prop :=
  (outcome.1 + outcome.2.1 + outcome.2.2 : ℕ) ≤ 1

-- Define the event "At least two heads"
def AtLeastTwoHeads (outcome : CoinOutcome) : Prop :=
  (outcome.1 + outcome.2.1 + outcome.2.2 : ℕ) ≥ 2

-- Theorem stating that the two events are complementary
theorem complementary_events :
  ∀ (outcome : CoinOutcome), NoMoreThanOneHead outcome ↔ ¬(AtLeastTwoHeads outcome) :=
by
  sorry


end complementary_events_l650_65073


namespace inequality_solution_set_l650_65000

theorem inequality_solution_set (x : ℝ) :
  (4 * x^3 + 9 * x^2 - 6 * x < 2) ↔ ((-2 < x ∧ x < -1) ∨ (-1 < x ∧ x < 1/4)) := by
  sorry

end inequality_solution_set_l650_65000


namespace integral_bound_for_differentiable_function_l650_65011

open Set
open MeasureTheory
open Interval
open Real

theorem integral_bound_for_differentiable_function 
  (f : ℝ → ℝ) 
  (hf_diff : DifferentiableOn ℝ f (Icc 0 1))
  (hf_zero : f 0 = 0 ∧ f 1 = 0)
  (hf_deriv_bound : ∀ x ∈ Icc 0 1, abs (deriv f x) ≤ 1) :
  abs (∫ x in Icc 0 1, f x) < (1 / 4) := by
  sorry

end integral_bound_for_differentiable_function_l650_65011


namespace tetrahedron_volume_in_cube_l650_65038

/-- The volume of a tetrahedron formed by non-adjacent vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side : ℝ) (h : cube_side = 8) :
  let tetrahedron_volume := (cube_side^3 * Real.sqrt 2) / 3
  tetrahedron_volume = (512 * Real.sqrt 2) / 3 := by sorry

end tetrahedron_volume_in_cube_l650_65038


namespace three_greater_than_sqrt_seven_l650_65006

theorem three_greater_than_sqrt_seven : 3 > Real.sqrt 7 := by
  sorry

end three_greater_than_sqrt_seven_l650_65006


namespace mariels_dogs_count_l650_65045

/-- The number of dogs Mariel is walking -/
def mariels_dogs : ℕ :=
  let total_legs : ℕ := 36
  let num_walkers : ℕ := 2
  let other_walker_dogs : ℕ := 3
  let human_legs : ℕ := 2
  let dog_legs : ℕ := 4
  let total_dogs : ℕ := (total_legs - num_walkers * human_legs) / dog_legs
  total_dogs - other_walker_dogs

theorem mariels_dogs_count : mariels_dogs = 5 := by
  sorry

end mariels_dogs_count_l650_65045


namespace geometric_sum_divisors_l650_65052

/-- The sum of geometric series from 0 to n with ratio a -/
def geometric_sum (a : ℕ) (n : ℕ) : ℕ :=
  (a^(n+1) - 1) / (a - 1)

/-- The set of all divisors of geometric_sum a n for some n -/
def divisor_set (a : ℕ) : Set ℕ :=
  {m : ℕ | ∃ n : ℕ, (geometric_sum a n) % m = 0}

/-- The set of all natural numbers relatively prime to a -/
def coprime_set (a : ℕ) : Set ℕ :=
  {m : ℕ | Nat.gcd m a = 1}

theorem geometric_sum_divisors (a : ℕ) (h : a > 1) :
  divisor_set a = coprime_set a :=
by sorry

end geometric_sum_divisors_l650_65052


namespace smallest_sum_of_reciprocals_l650_65048

theorem smallest_sum_of_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 → 
    (a : ℕ) + (b : ℕ) ≥ (x : ℕ) + (y : ℕ)) → 
  (x : ℕ) + (y : ℕ) = 64 :=
by sorry

end smallest_sum_of_reciprocals_l650_65048


namespace negation_of_universal_statement_l650_65015

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≥ 0) :=
by sorry

end negation_of_universal_statement_l650_65015


namespace characterize_bijection_condition_l650_65025

/-- Given an even positive integer m, characterize all positive integers n for which
    there exists a bijection f from [1,n] to [1,n] satisfying the condition that
    for all x and y in [1,n] where n divides mx - y, n+1 divides f(x)^m - f(y). -/
theorem characterize_bijection_condition (m : ℕ) (h_m : Even m) (h_m_pos : 0 < m) :
  ∀ n : ℕ, 0 < n →
    (∃ f : Fin n → Fin n, Function.Bijective f ∧
      ∀ x y : Fin n, n ∣ m * x - y →
        (n + 1) ∣ (f x)^m - (f y)) ↔
    Nat.Prime (n + 1) :=
by sorry

end characterize_bijection_condition_l650_65025


namespace negation_of_forall_ge_two_l650_65023

theorem negation_of_forall_ge_two :
  (¬ (∀ x : ℝ, x > 0 → x + 1/x ≥ 2)) ↔ (∃ x : ℝ, x > 0 ∧ x + 1/x < 2) :=
by sorry

end negation_of_forall_ge_two_l650_65023


namespace chord_length_l650_65026

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by
  sorry

end chord_length_l650_65026


namespace probability_A_B_different_groups_l650_65030

def number_of_people : ℕ := 6
def number_of_groups : ℕ := 3

theorem probability_A_B_different_groups :
  let total_ways := (number_of_people.choose 2) * ((number_of_people - 2).choose 2) / (number_of_groups.factorial)
  let ways_same_group := ((number_of_people - 2).choose 2) / ((number_of_groups - 1).factorial)
  (total_ways - ways_same_group) / total_ways = 4 / 5 := by
  sorry

end probability_A_B_different_groups_l650_65030


namespace absolute_value_two_l650_65067

theorem absolute_value_two (m : ℝ) : |m| = 2 → m = 2 ∨ m = -2 := by
  sorry

end absolute_value_two_l650_65067


namespace min_value_fraction_sum_min_value_fraction_sum_achievable_l650_65066

theorem min_value_fraction_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (8 / x + 2 / y) ≥ 18 :=
by sorry

theorem min_value_fraction_sum_achievable :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (8 / x + 2 / y) = 18 :=
by sorry

end min_value_fraction_sum_min_value_fraction_sum_achievable_l650_65066


namespace special_polynomial_value_at_one_l650_65088

/-- A non-constant quadratic polynomial satisfying the given equation -/
def SpecialPolynomial (g : ℝ → ℝ) : Prop :=
  (∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, g x = a * x^2 + b * x + c) ∧
  (∀ x : ℝ, x ≠ 0 → g (x - 1) + g x + g (x + 1) = (g x)^2 / (2023 * x))

theorem special_polynomial_value_at_one
  (g : ℝ → ℝ) (h : SpecialPolynomial g) : g 1 = 3 := by
  sorry

end special_polynomial_value_at_one_l650_65088


namespace photocopy_cost_calculation_l650_65024

/-- The cost of a single photocopy -/
def photocopy_cost : ℝ := 0.02

/-- The discount rate for large orders -/
def discount_rate : ℝ := 0.25

/-- The number of copies in a large order -/
def large_order_threshold : ℕ := 100

/-- The number of copies each person orders -/
def copies_per_person : ℕ := 80

/-- The savings per person when combining orders -/
def savings_per_person : ℝ := 0.40

theorem photocopy_cost_calculation :
  let total_copies := 2 * copies_per_person
  let undiscounted_total := total_copies * photocopy_cost
  let discounted_total := undiscounted_total * (1 - discount_rate)
  discounted_total = undiscounted_total - 2 * savings_per_person :=
by sorry

end photocopy_cost_calculation_l650_65024


namespace cyclic_inequality_l650_65009

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + c*a + a^2)) ≥ (1/3) * (a + b + c) := by
  sorry

end cyclic_inequality_l650_65009


namespace full_time_employees_count_l650_65002

/-- A corporation with part-time and full-time employees -/
structure Corporation where
  total_employees : ℕ
  part_time_employees : ℕ

/-- The number of full-time employees in a corporation -/
def full_time_employees (c : Corporation) : ℕ :=
  c.total_employees - c.part_time_employees

/-- Theorem stating the number of full-time employees in a specific corporation -/
theorem full_time_employees_count (c : Corporation) 
  (h1 : c.total_employees = 65134)
  (h2 : c.part_time_employees = 2041) :
  full_time_employees c = 63093 := by
  sorry

end full_time_employees_count_l650_65002


namespace solution_set_l650_65014

/-- A linear function f(x) = ax + b where a > 0 and f(-2) = 0 -/
def f (a b : ℝ) (ha : a > 0) (hf : a * (-2) + b = 0) (x : ℝ) : ℝ :=
  a * x + b

theorem solution_set (a b x : ℝ) (ha : a > 0) (hf : a * (-2) + b = 0) :
  a * x > b ↔ x > 2 := by
  sorry

end solution_set_l650_65014


namespace library_books_loaned_l650_65093

theorem library_books_loaned (initial_books : ℕ) (return_rate : ℚ) (final_books : ℕ) 
  (h1 : initial_books = 75)
  (h2 : return_rate = 70 / 100)
  (h3 : final_books = 57) :
  ∃ (loaned_books : ℕ), loaned_books = 60 ∧ 
    initial_books - (↑loaned_books * (1 - return_rate)).floor = final_books :=
by sorry

end library_books_loaned_l650_65093


namespace distributive_analogy_l650_65094

theorem distributive_analogy (a b c : ℝ) (h : c ≠ 0) :
  ((a + b) * c = a * c + b * c) ↔ ((a + b) / c = a / c + b / c) :=
sorry

end distributive_analogy_l650_65094


namespace line_not_in_third_quadrant_l650_65082

/-- The line ρ cos θ + 2ρ sin θ = 1 in polar coordinates -/
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.cos θ + 2 * ρ * Real.sin θ = 1

/-- The third quadrant in Cartesian coordinates -/
def third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Theorem: The line ρ cos θ + 2ρ sin θ = 1 does not pass through the third quadrant -/
theorem line_not_in_third_quadrant :
  ¬∃ (x y : ℝ), (∃ (ρ θ : ℝ), polar_line ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) ∧ third_quadrant x y :=
sorry

end line_not_in_third_quadrant_l650_65082


namespace set_operations_l650_65040

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}
def B : Set ℝ := {x | x^2 - 9*x + 14 < 0}

theorem set_operations :
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧
  ((Set.univ \ A) ∩ B = {x | 2 < x ∧ x < 3}) := by
  sorry

end set_operations_l650_65040


namespace total_games_won_l650_65042

def team_games_won (games_played : ℕ) (win_percentage : ℚ) : ℕ :=
  ⌊(games_played : ℚ) * win_percentage⌋₊

theorem total_games_won :
  let team_a := team_games_won 150 (35/100)
  let team_b := team_games_won 110 (45/100)
  let team_c := team_games_won 200 (30/100)
  team_a + team_b + team_c = 163 := by
  sorry

end total_games_won_l650_65042


namespace max_points_in_tournament_l650_65055

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (games_per_pair : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculates the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  t.num_teams.choose 2 * t.games_per_pair

/-- Calculates the total points available in the tournament -/
def total_points (t : Tournament) : ℕ :=
  total_games t * t.points_for_win

/-- Represents the maximum points achievable by top teams -/
def max_points_for_top_teams (t : Tournament) : ℕ :=
  let points_from_top_matches := (t.num_teams - 1) * t.points_for_win
  let points_from_other_matches := (t.num_teams - 3) * 2 * t.points_for_win
  points_from_top_matches + points_from_other_matches

/-- The main theorem to be proved -/
theorem max_points_in_tournament (t : Tournament) 
  (h1 : t.num_teams = 8)
  (h2 : t.games_per_pair = 2)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1)
  (h5 : t.points_for_loss = 0) :
  max_points_for_top_teams t = 38 :=
sorry

end max_points_in_tournament_l650_65055


namespace complex_modulus_equality_l650_65020

theorem complex_modulus_equality (t : ℝ) (ht : t > 0) : 
  t = 3 * Real.sqrt 3 ↔ Complex.abs (-5 + t * Complex.I) = 2 * Real.sqrt 13 := by
sorry

end complex_modulus_equality_l650_65020


namespace range_of_product_l650_65049

theorem range_of_product (x y z : ℝ) 
  (hx : -3 < x) (hxy : x < y) (hy : y < 1) 
  (hz1 : -4 < z) (hz2 : z < 0) : 
  0 < (x - y) * z ∧ (x - y) * z < 16 := by
  sorry

end range_of_product_l650_65049


namespace cindy_solution_l650_65077

def cindy_problem (x : ℝ) : Prop :=
  (x - 12) / 4 = 32 →
  round ((x - 7) / 5) = 27

theorem cindy_solution : ∃ x : ℝ, cindy_problem x := by
  sorry

end cindy_solution_l650_65077


namespace imaginary_part_of_complex_number_l650_65050

theorem imaginary_part_of_complex_number : Complex.im (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end imaginary_part_of_complex_number_l650_65050


namespace remainder_102_104_plus_6_div_9_l650_65001

theorem remainder_102_104_plus_6_div_9 : (102 * 104 + 6) % 9 = 3 := by
  sorry

end remainder_102_104_plus_6_div_9_l650_65001


namespace triangle_angles_l650_65064

-- Define a triangle XYZ
structure Triangle :=
  (X Y Z : Point)

-- Define the angles in the triangle
def angle_YXZ (t : Triangle) : ℝ := sorry
def angle_XYZ (t : Triangle) : ℝ := sorry
def angle_XZY (t : Triangle) : ℝ := sorry

-- State the theorem
theorem triangle_angles (t : Triangle) :
  angle_YXZ t = 40 ∧ angle_XYZ t = 80 → angle_XZY t = 60 :=
by sorry

end triangle_angles_l650_65064


namespace wage_increase_l650_65019

/-- Represents the regression equation for monthly wage based on labor productivity -/
def wage_equation (x : ℝ) : ℝ := 50 + 60 * x

/-- Theorem stating that an increase of 1 in labor productivity results in a 60 yuan increase in monthly wage -/
theorem wage_increase (x : ℝ) : wage_equation (x + 1) = wage_equation x + 60 := by
  sorry

end wage_increase_l650_65019


namespace triangle_equality_condition_l650_65041

/-- In a triangle ABC with sides a, b, and c, the equation 
    (b² * c²) / (2 * b * c * cos(A)) = b² + c² - 2 * b * c * cos(A) 
    holds if and only if a = b or a = c. -/
theorem triangle_equality_condition (a b c : ℝ) (A : ℝ) 
  (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  (b^2 * c^2) / (2 * b * c * Real.cos A) = b^2 + c^2 - 2 * b * c * Real.cos A ↔ 
  a = b ∨ a = c := by
sorry

end triangle_equality_condition_l650_65041


namespace percentage_problem_l650_65033

theorem percentage_problem (P : ℝ) : 
  (0.3 * 200 = P / 100 * 50 + 30) → P = 60 := by
  sorry

end percentage_problem_l650_65033


namespace age_of_15th_person_l650_65076

theorem age_of_15th_person (total_persons : Nat) (avg_age_all : Nat) (group1_size : Nat) 
  (avg_age_group1 : Nat) (group2_size : Nat) (avg_age_group2 : Nat) :
  total_persons = 18 →
  avg_age_all = 15 →
  group1_size = 5 →
  avg_age_group1 = 14 →
  group2_size = 9 →
  avg_age_group2 = 16 →
  (total_persons * avg_age_all) = 
    (group1_size * avg_age_group1) + (group2_size * avg_age_group2) + 56 :=
by
  sorry

#check age_of_15th_person

end age_of_15th_person_l650_65076


namespace stream_speed_calculation_l650_65037

/-- Proves that given a boat with a speed of 20 km/hr in still water,
if it travels 26 km downstream and 14 km upstream in the same time,
then the speed of the stream is 6 km/hr. -/
theorem stream_speed_calculation (boat_speed : ℝ) (downstream_distance : ℝ) (upstream_distance : ℝ) :
  boat_speed = 20 →
  downstream_distance = 26 →
  upstream_distance = 14 →
  (downstream_distance / (boat_speed + x) = upstream_distance / (boat_speed - x)) →
  x = 6 :=
by sorry

end stream_speed_calculation_l650_65037


namespace sum_of_xyz_equals_695_l650_65089

theorem sum_of_xyz_equals_695 (a b : ℝ) (x y z : ℕ+) :
  a^2 = 9/25 →
  b^2 = (3 + Real.sqrt 2)^2 / 14 →
  a < 0 →
  b > 0 →
  (a + b)^3 = (x.val : ℝ) * Real.sqrt y.val / z.val →
  x.val + y.val + z.val = 695 := by
sorry

end sum_of_xyz_equals_695_l650_65089


namespace ferry_tourists_l650_65029

theorem ferry_tourists (a₁ : ℕ) (d : ℕ) (n : ℕ) (h1 : a₁ = 85) (h2 : d = 3) (h3 : n = 5) :
  n * (2 * a₁ + (n - 1) * d) / 2 = 455 := by
  sorry

end ferry_tourists_l650_65029


namespace quadratic_form_minimum_l650_65012

theorem quadratic_form_minimum : ∀ x y : ℝ,
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 ≥ 28 ∧
  ∃ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 = 28 :=
by sorry

end quadratic_form_minimum_l650_65012


namespace complex_product_pure_imaginary_l650_65008

theorem complex_product_pure_imaginary (a : ℝ) : 
  (Complex.I : ℂ)^2 = -1 →
  (↑a - Complex.I) * (1 + Complex.I) = Complex.I * (Complex.ofReal (a - 1)) →
  a = -1 := by
sorry

end complex_product_pure_imaginary_l650_65008


namespace functional_equation_solution_l650_65013

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f (f y / f x + 1) = f (x + y / x + 1) - f x

/-- The main theorem stating the form of the function satisfying the equation -/
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, (∀ x, x > 0 → f x > 0) →
  SatisfiesFunctionalEquation f →
  ∃ a : ℝ, a > 0 ∧ ∀ x, x > 0 → f x = a * x :=
sorry

end functional_equation_solution_l650_65013


namespace sum_from_simple_interest_and_true_discount_l650_65075

/-- Given a sum, time, and rate, if the simple interest is 85 and the true discount is 80, then the sum is 1360 -/
theorem sum_from_simple_interest_and_true_discount 
  (P T R : ℝ) 
  (h_simple_interest : (P * T * R) / 100 = 85)
  (h_true_discount : (P * T * R) / (100 + T * R) = 80) :
  P = 1360 := by
  sorry

end sum_from_simple_interest_and_true_discount_l650_65075


namespace johnnys_age_reference_l650_65060

/-- Proves that Johnny was referring to 3 years ago -/
theorem johnnys_age_reference : 
  ∀ (current_age : ℕ) (years_ago : ℕ),
  current_age = 8 →
  current_age + 2 = 2 * (current_age - years_ago) →
  years_ago = 3 := by
  sorry

end johnnys_age_reference_l650_65060


namespace cube_root_unity_inverse_l650_65070

/-- Given a complex cube root of unity ω, prove that (ω - ω⁻¹)⁻¹ = -(1 + 2ω²)/5 -/
theorem cube_root_unity_inverse (ω : ℂ) (h1 : ω^3 = 1) (h2 : ω ≠ 1) :
  (ω - ω⁻¹)⁻¹ = -(1 + 2*ω^2)/5 := by
  sorry

end cube_root_unity_inverse_l650_65070


namespace arithmetic_sequence_common_difference_l650_65022

/-- Given an arithmetic sequence {a_n} where a₂ = 9 and a₅ = 33, 
    prove that the common difference is 8. -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) -- The arithmetic sequence
  (h1 : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) -- Definition of arithmetic sequence
  (h2 : a 2 = 9) -- Given: a₂ = 9
  (h3 : a 5 = 33) -- Given: a₅ = 33
  : a 2 - a 1 = 8 := by
sorry

end arithmetic_sequence_common_difference_l650_65022


namespace quadratic_root_problem_l650_65087

theorem quadratic_root_problem (a : ℝ) :
  ((-1 : ℝ)^2 - 2*(-1) + a = 0) → 
  (∃ x : ℝ, x^2 - 2*x + a = 0 ∧ x ≠ -1 ∧ x = 3) :=
by sorry

end quadratic_root_problem_l650_65087


namespace f_odd_and_periodic_l650_65059

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
def condition1 (f : ℝ → ℝ) : Prop := ∀ x, f (10 + x) = f (10 - x)
def condition2 (f : ℝ → ℝ) : Prop := ∀ x, f (20 + x) = -f (20 - x)

-- State the theorem
theorem f_odd_and_periodic (h1 : condition1 f) (h2 : condition2 f) :
  (∀ x, f (x + 40) = f x) ∧ (∀ x, f (-x) = -f x) := by
  sorry

end f_odd_and_periodic_l650_65059


namespace sum_reciprocals_squared_l650_65035

-- Define the constants
noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6

-- State the theorem
theorem sum_reciprocals_squared :
  (1/a + 1/b + 1/c + 1/d)^2 = 96/529 := by sorry

end sum_reciprocals_squared_l650_65035


namespace probability_quarter_or_dime_l650_65099

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Dime
  | Nickel

/-- The value of each coin type in cents -/
def coinValue : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Dime => 10
  | Coin.Nickel => 5

/-- The total value of each coin type in the jar in cents -/
def totalValue : Coin → ℕ
  | Coin.Quarter => 500
  | Coin.Dime => 600
  | Coin.Nickel => 200

/-- The number of coins of each type in the jar -/
def coinCount (c : Coin) : ℕ := totalValue c / coinValue c

/-- The total number of coins in the jar -/
def totalCoins : ℕ := coinCount Coin.Quarter + coinCount Coin.Dime + coinCount Coin.Nickel

/-- The probability of selecting either a quarter or a dime from the jar -/
def probQuarterOrDime : ℚ :=
  (coinCount Coin.Quarter + coinCount Coin.Dime : ℚ) / totalCoins

theorem probability_quarter_or_dime :
  probQuarterOrDime = 2 / 3 := by
  sorry


end probability_quarter_or_dime_l650_65099


namespace max_score_is_six_l650_65051

/-- Represents a 5x5 game board -/
def GameBoard : Type := Fin 5 → Fin 5 → Bool

/-- Calculates the sum of a 3x3 sub-square starting at (i, j) -/
def subSquareSum (board : GameBoard) (i j : Fin 3) : ℕ :=
  (Finset.sum (Finset.range 3) fun x =>
    Finset.sum (Finset.range 3) fun y =>
      if board (i + x) (j + y) then 1 else 0)

/-- Calculates the score of a given board (maximum sum of any 3x3 sub-square) -/
def boardScore (board : GameBoard) : ℕ :=
  Finset.sup (Finset.range 3) fun i =>
    Finset.sup (Finset.range 3) fun j =>
      subSquareSum board i j

/-- Represents a strategy for Player 2 -/
def Player2Strategy : Type := GameBoard → Fin 5 → Fin 5

/-- Represents the game play with both players' moves -/
def gamePlay (p2strat : Player2Strategy) : GameBoard :=
  sorry -- Implementation of game play

theorem max_score_is_six :
  ∀ (p2strat : Player2Strategy),
    boardScore (gamePlay p2strat) ≤ 6 ∧
    ∃ (optimal_p2strat : Player2Strategy),
      boardScore (gamePlay optimal_p2strat) = 6 :=
sorry

end max_score_is_six_l650_65051


namespace tangent_line_equation_l650_65007

/-- The equation of the tangent line to y = xe^x + 1 at (1, e+1) -/
theorem tangent_line_equation (x y : ℝ) : 
  (∀ t, y = t * Real.exp t + 1) →  -- Curve equation
  2 * Real.exp 1 * x - y - Real.exp 1 + 1 = 0 -- Tangent line equation
  ↔ 
  (x = 1 ∧ y = Real.exp 1 + 1) -- Point of tangency
  := by sorry

end tangent_line_equation_l650_65007


namespace evaluate_expression_l650_65092

theorem evaluate_expression : 103^4 - 4 * 103^3 + 6 * 103^2 - 4 * 103 + 1 = 108243216 := by
  sorry

end evaluate_expression_l650_65092


namespace equation_system_solution_l650_65084

theorem equation_system_solution (a b x y : ℝ) 
  (h1 : x^2 + x*y + y^2 = a^2)
  (h2 : Real.log (Real.sqrt a) / Real.log (a^(1/x)) + Real.log (Real.sqrt b) / Real.log (b^(1/y)) = a / Real.sqrt 3) :
  x = a * Real.sqrt 3 / 3 ∧ y = a * Real.sqrt 3 / 3 := by
sorry

end equation_system_solution_l650_65084


namespace second_rectangle_height_l650_65074

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.height

/-- Proves that the height of the second rectangle is 6 inches -/
theorem second_rectangle_height (r1 r2 : Rectangle) 
  (h1 : r1.width = 4)
  (h2 : r1.height = 5)
  (h3 : r2.width = 3)
  (h4 : area r1 = area r2 + 2) : 
  r2.height = 6 := by
  sorry

#check second_rectangle_height

end second_rectangle_height_l650_65074


namespace max_value_implies_a_l650_65044

theorem max_value_implies_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, 2 * a^x - 5 ≤ 10) ∧
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, 2 * a^x - 5 = 10) →
  a = Real.sqrt (15/2) ∨ a = 15/2 :=
by sorry

end max_value_implies_a_l650_65044


namespace bus_seating_problem_l650_65062

theorem bus_seating_problem :
  ∀ (bus_seats minibus_seats : ℕ),
    bus_seats = minibus_seats + 20 →
    5 * bus_seats + 5 * minibus_seats = 300 →
    bus_seats = 40 ∧ minibus_seats = 20 :=
by
  sorry

#check bus_seating_problem

end bus_seating_problem_l650_65062


namespace number_solution_l650_65068

theorem number_solution (z s : ℝ) (n : ℝ) : 
  z ≠ 0 → 
  z = Real.sqrt (n * z * s - 9 * s^2) → 
  z = 3 → 
  n = 3 + 3 * s := by
  sorry

end number_solution_l650_65068


namespace inequality_of_five_variables_l650_65095

theorem inequality_of_five_variables (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) ≥ 2 := by
  sorry

end inequality_of_five_variables_l650_65095


namespace equal_hot_dogs_and_buns_l650_65071

/-- The number of hot dogs in each package -/
def hot_dogs_per_package : ℕ := 7

/-- The number of buns in each package -/
def buns_per_package : ℕ := 9

/-- The smallest number of hot dog packages needed to have an equal number of hot dogs and buns -/
def smallest_number_of_packages : ℕ := 9

theorem equal_hot_dogs_and_buns :
  smallest_number_of_packages * hot_dogs_per_package =
  (smallest_number_of_packages * hot_dogs_per_package / buns_per_package) * buns_per_package ∧
  ∀ n : ℕ, n < smallest_number_of_packages →
    n * hot_dogs_per_package ≠
    (n * hot_dogs_per_package / buns_per_package) * buns_per_package :=
by sorry

end equal_hot_dogs_and_buns_l650_65071


namespace monotonic_increasing_condition_l650_65043

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x

-- State the theorem
theorem monotonic_increasing_condition (a : ℝ) : 
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, Monotone (fun x => f a x)) → a ≥ 3 := by
  sorry

end monotonic_increasing_condition_l650_65043


namespace claire_took_eight_photos_l650_65056

/-- The number of photos taken by Claire -/
def claire_photos : ℕ := 8

/-- The number of photos taken by Lisa -/
def lisa_photos : ℕ := 3 * claire_photos

/-- The number of photos taken by Robert -/
def robert_photos : ℕ := claire_photos + 16

/-- Theorem stating that given the conditions, Claire has taken 8 photos -/
theorem claire_took_eight_photos :
  lisa_photos = robert_photos ∧
  lisa_photos = 3 * claire_photos ∧
  robert_photos = claire_photos + 16 →
  claire_photos = 8 := by
  sorry

end claire_took_eight_photos_l650_65056


namespace students_in_all_classes_l650_65047

/-- Represents the number of students registered for a combination of classes -/
structure ClassRegistration where
  history : ℕ
  math : ℕ
  english : ℕ
  historyMath : ℕ
  historyEnglish : ℕ
  mathEnglish : ℕ
  allThree : ℕ

/-- The theorem stating the number of students registered for all three classes -/
theorem students_in_all_classes 
  (total : ℕ) 
  (classes : ClassRegistration) 
  (h1 : total = 86)
  (h2 : classes.history = 12)
  (h3 : classes.math = 17)
  (h4 : classes.english = 36)
  (h5 : classes.historyMath + classes.historyEnglish + classes.mathEnglish = 3)
  (h6 : total = classes.history + classes.math + classes.english - 
        (classes.historyMath + classes.historyEnglish + classes.mathEnglish) + 
        classes.allThree) :
  classes.allThree = 24 := by
  sorry

end students_in_all_classes_l650_65047


namespace solve_equation_l650_65072

-- Define the custom operation *
def star (a b : ℚ) : ℚ := 4 * a - 2 * b

-- State the theorem
theorem solve_equation : ∃ x : ℚ, star 3 (star 6 x) = 2 ∧ x = 19/2 := by
  sorry

end solve_equation_l650_65072


namespace quadratic_function_unique_a_l650_65032

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluates the quadratic function at a given x -/
def QuadraticFunction.eval (f : QuadraticFunction) (x : ℚ) : ℚ :=
  f.a * x^2 + f.b * x + f.c

theorem quadratic_function_unique_a (f : QuadraticFunction) :
  f.eval 1 = 5 → f.eval 0 = 2 → f.a = -3 := by
  sorry

end quadratic_function_unique_a_l650_65032


namespace grade_distribution_l650_65053

theorem grade_distribution (thompson_total : ℕ) (thompson_a : ℕ) (thompson_b : ℕ) (carter_total : ℕ)
  (h1 : thompson_total = 20)
  (h2 : thompson_a = 12)
  (h3 : thompson_b = 5)
  (h4 : carter_total = 30)
  (h5 : thompson_a + thompson_b ≤ thompson_total) :
  ∃ (carter_a carter_b : ℕ),
    carter_a + carter_b ≤ carter_total ∧
    carter_a * thompson_total = thompson_a * carter_total ∧
    carter_b * (thompson_total - thompson_a) = thompson_b * (carter_total - carter_a) ∧
    carter_a = 18 ∧
    carter_b = 8 :=
by sorry

end grade_distribution_l650_65053


namespace extremum_value_l650_65005

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- Define the derivative of f(x)
def f_prime (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem extremum_value (a b : ℝ) : 
  f a b 1 = 10 ∧ f_prime a b 1 = 0 → a = 4 := by
  sorry

end extremum_value_l650_65005


namespace min_value_x_plus_reciprocal_l650_65004

theorem min_value_x_plus_reciprocal (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end min_value_x_plus_reciprocal_l650_65004


namespace cube_sum_theorem_l650_65054

theorem cube_sum_theorem (x y z : ℝ) 
  (sum_condition : x + y + z = 5)
  (prod_sum_condition : x*y + y*z + z*x = 6)
  (prod_condition : x*y*z = -15) :
  x^3 + y^3 + z^3 = -97 := by sorry

end cube_sum_theorem_l650_65054


namespace prob_X_eq_three_l650_65010

/-- A random variable X following a binomial distribution B(6, 1/2) -/
def X : ℕ → ℝ := sorry

/-- The probability mass function for X -/
def pmf (k : ℕ) : ℝ := sorry

/-- Theorem: The probability of X = 3 is 5/16 -/
theorem prob_X_eq_three : pmf 3 = 5/16 := by sorry

end prob_X_eq_three_l650_65010


namespace initial_machines_count_l650_65031

/-- The number of machines initially operating to fill a production order -/
def initial_machines : ℕ := sorry

/-- The total number of machines available -/
def total_machines : ℕ := 7

/-- The time taken by the initial number of machines to fill the order (in hours) -/
def initial_time : ℕ := 42

/-- The time taken by all machines to fill the order (in hours) -/
def all_machines_time : ℕ := 36

/-- The rate at which each machine works (assumed to be constant and positive) -/
def machine_rate : ℝ := sorry

theorem initial_machines_count :
  initial_machines = 6 :=
by sorry

end initial_machines_count_l650_65031


namespace boxes_to_brother_l650_65078

def total_boxes : ℕ := 45
def boxes_to_sister : ℕ := 9
def boxes_to_cousin : ℕ := 7
def boxes_left : ℕ := 17

theorem boxes_to_brother :
  total_boxes - boxes_to_sister - boxes_to_cousin - boxes_left = 12 := by
  sorry

end boxes_to_brother_l650_65078


namespace product_of_solutions_l650_65069

theorem product_of_solutions (x₁ x₂ : ℝ) : 
  (|6 * x₁| + 5 = 47) → (|6 * x₂| + 5 = 47) → x₁ * x₂ = -49 := by
  sorry

end product_of_solutions_l650_65069


namespace unique_pizza_combinations_l650_65086

def num_toppings : ℕ := 8
def toppings_per_pizza : ℕ := 3

theorem unique_pizza_combinations :
  Nat.choose num_toppings toppings_per_pizza = 56 := by
  sorry

end unique_pizza_combinations_l650_65086
