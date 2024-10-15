import Mathlib

namespace NUMINAMATH_CALUDE_root_equality_implies_c_equals_two_l1570_157080

theorem root_equality_implies_c_equals_two :
  ∀ (a b c d : ℕ),
    a > 1 → b > 1 → c > 1 → d > 1 →
    (∀ (M : ℝ), M ≠ 1 →
      (M^(1/a + 1/(a*b) + 1/(a*b*c) + 1/(a*b*c*d)) = M^(37/48))) →
    c = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_equality_implies_c_equals_two_l1570_157080


namespace NUMINAMATH_CALUDE_two_variables_scatter_plot_l1570_157046

-- Define a type for statistical variables
def StatisticalVariable : Type := ℝ

-- Define a type for a dataset of two variables
def Dataset : Type := List (StatisticalVariable × StatisticalVariable)

-- Statement: Any two statistical variables can be represented with a scatter plot
theorem two_variables_scatter_plot (data : Dataset) :
  ∃ (scatter_plot : Dataset → Bool), scatter_plot data = true :=
sorry

end NUMINAMATH_CALUDE_two_variables_scatter_plot_l1570_157046


namespace NUMINAMATH_CALUDE_male_honor_roll_fraction_l1570_157084

theorem male_honor_roll_fraction (total : ℝ) (h1 : total > 0) :
  let female_ratio : ℝ := 2 / 5
  let female_honor_ratio : ℝ := 5 / 6
  let total_honor_ratio : ℝ := 22 / 30
  let male_ratio : ℝ := 1 - female_ratio
  let male_honor_ratio : ℝ := (total_honor_ratio * total - female_honor_ratio * female_ratio * total) / (male_ratio * total)
  male_honor_ratio = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_male_honor_roll_fraction_l1570_157084


namespace NUMINAMATH_CALUDE_smallest_4digit_divisible_by_5_6_2_l1570_157016

def is_divisible (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_4digit_divisible_by_5_6_2 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 →
  (is_divisible n 5 ∧ is_divisible n 6 ∧ is_divisible n 2) →
  1020 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_4digit_divisible_by_5_6_2_l1570_157016


namespace NUMINAMATH_CALUDE_solve_equations_l1570_157006

-- Define the equations
def equation1 (y : ℝ) : Prop := 2.4 * y - 9.8 = 1.4 * y - 9
def equation2 (x : ℝ) : Prop := x - 3 = (3/2) * x + 1

-- State the theorem
theorem solve_equations :
  (∃ y : ℝ, equation1 y ∧ y = 0.8) ∧
  (∃ x : ℝ, equation2 x ∧ x = -8) := by sorry

end NUMINAMATH_CALUDE_solve_equations_l1570_157006


namespace NUMINAMATH_CALUDE_product_inequality_l1570_157091

theorem product_inequality (a a' b b' c c' : ℝ) 
  (h1 : a * a' > 0) 
  (h2 : a * c ≥ b^2) 
  (h3 : a' * c' ≥ b'^2) : 
  (a + a') * (c + c') ≥ (b + b')^2 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1570_157091


namespace NUMINAMATH_CALUDE_divisible_by_25_l1570_157099

theorem divisible_by_25 (n : ℕ) : 25 ∣ (2^(n+2) * 3^n + 5*n - 4) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_25_l1570_157099


namespace NUMINAMATH_CALUDE_exists_student_with_eight_sessions_l1570_157097

/-- A structure representing a club with students and sessions. -/
structure Club where
  students : Finset Nat
  sessions : Finset Nat
  attended : Nat → Finset Nat
  meet_once : ∀ s₁ s₂, s₁ ∈ students → s₂ ∈ students → s₁ ≠ s₂ →
    ∃! session, session ∈ sessions ∧ s₁ ∈ attended session ∧ s₂ ∈ attended session
  not_all_in_one : ∀ session, session ∈ sessions → ∃ s, s ∈ students ∧ s ∉ attended session

/-- Theorem stating that in a club satisfying the given conditions,
    there exists a student who attended at least 8 sessions. -/
theorem exists_student_with_eight_sessions (c : Club) (h : c.students.card = 50) :
  ∃ s, s ∈ c.students ∧ (c.sessions.filter (fun session => s ∈ c.attended session)).card ≥ 8 :=
sorry

end NUMINAMATH_CALUDE_exists_student_with_eight_sessions_l1570_157097


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1570_157078

/-- p is a necessary but not sufficient condition for q -/
theorem p_necessary_not_sufficient_for_q :
  (∀ x, (-1 < x ∧ x < 2) → x < 3) ∧
  ¬(∀ x, x < 3 → (-1 < x ∧ x < 2)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1570_157078


namespace NUMINAMATH_CALUDE_parallel_lines_c_value_l1570_157041

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of c for which the lines y = 8x + 2 and y = (2c)x - 4 are parallel -/
theorem parallel_lines_c_value :
  (∀ x y : ℝ, y = 8 * x + 2 ↔ y = (2 * c) * x - 4) → c = 4 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_c_value_l1570_157041


namespace NUMINAMATH_CALUDE_turquoise_score_difference_is_correct_l1570_157079

/-- Calculates 5/8 of the difference between white and black scores in a turquoise mixture --/
def turquoise_score_difference (total : ℚ) : ℚ :=
  let white_ratio : ℚ := 5
  let black_ratio : ℚ := 3
  let total_ratio : ℚ := white_ratio + black_ratio
  let part_value : ℚ := total / total_ratio
  let white_scores : ℚ := white_ratio * part_value
  let black_scores : ℚ := black_ratio * part_value
  let difference : ℚ := white_scores - black_scores
  (5 : ℚ) / 8 * difference

/-- Theorem stating that 5/8 of the difference between white and black scores is 58.125 --/
theorem turquoise_score_difference_is_correct :
  turquoise_score_difference 372 = 58125 / 1000 :=
by sorry

end NUMINAMATH_CALUDE_turquoise_score_difference_is_correct_l1570_157079


namespace NUMINAMATH_CALUDE_rectangle_width_l1570_157085

theorem rectangle_width (perimeter : ℝ) (length_difference : ℝ) : perimeter = 46 → length_difference = 7 → 
  let length := (perimeter / 2 - length_difference) / 2
  let width := length + length_difference
  width = 15 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_l1570_157085


namespace NUMINAMATH_CALUDE_maria_friends_money_l1570_157031

/-- The amount of money Rene received from Maria -/
def rene_amount : ℕ := 300

/-- The amount of money Florence received from Maria -/
def florence_amount : ℕ := 3 * rene_amount

/-- The amount of money Isha received from Maria -/
def isha_amount : ℕ := florence_amount / 2

/-- The total amount of money Maria gave to her three friends -/
def total_amount : ℕ := isha_amount + florence_amount + rene_amount

/-- Theorem stating that the total amount Maria gave to her friends is $1650 -/
theorem maria_friends_money : total_amount = 1650 := by sorry

end NUMINAMATH_CALUDE_maria_friends_money_l1570_157031


namespace NUMINAMATH_CALUDE_positive_number_square_sum_l1570_157087

theorem positive_number_square_sum : ∃ n : ℕ+, (n : ℝ)^2 + 2*(n : ℝ) = 170 ∧ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_square_sum_l1570_157087


namespace NUMINAMATH_CALUDE_sum_of_a_and_t_is_71_l1570_157036

/-- Given a natural number n, this function represents the equation
    √(n+1 + (n+1)/((n+1)²-1)) = (n+1)√((n+1)/((n+1)²-1)) -/
def equation_pattern (n : ℕ) : Prop :=
  Real.sqrt ((n + 1 : ℝ) + (n + 1) / ((n + 1)^2 - 1)) = (n + 1 : ℝ) * Real.sqrt ((n + 1) / ((n + 1)^2 - 1))

/-- The main theorem stating that given the pattern for n = 1 to 7,
    the sum of a and t in the equation √(8 + a/t) = 8√(a/t) is 71 -/
theorem sum_of_a_and_t_is_71 
  (h1 : equation_pattern 1)
  (h2 : equation_pattern 2)
  (h3 : equation_pattern 3)
  (h4 : equation_pattern 4)
  (h5 : equation_pattern 5)
  (h6 : equation_pattern 6)
  (h7 : equation_pattern 7)
  (a t : ℝ)
  (ha : a > 0)
  (ht : t > 0)
  (h : Real.sqrt (8 + a/t) = 8 * Real.sqrt (a/t)) :
  a + t = 71 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_t_is_71_l1570_157036


namespace NUMINAMATH_CALUDE_complex_multiplication_l1570_157066

theorem complex_multiplication (z : ℂ) (i : ℂ) : 
  z.re = 1 ∧ z.im = 1 ∧ i * i = -1 → z * (1 - i) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l1570_157066


namespace NUMINAMATH_CALUDE_solve_equation_l1570_157083

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((3 / x) + 3) = 5 / 3 → x = -27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1570_157083


namespace NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1570_157055

theorem abs_neg_three_eq_three : |(-3 : ℝ)| = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_three_eq_three_l1570_157055


namespace NUMINAMATH_CALUDE_square_area_ratio_when_tripled_l1570_157075

theorem square_area_ratio_when_tripled (s : ℝ) (h : s > 0) :
  (s^2) / ((3*s)^2) = 1/9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_when_tripled_l1570_157075


namespace NUMINAMATH_CALUDE_prime_divisor_existence_l1570_157021

theorem prime_divisor_existence (p : Nat) (hp : p.Prime ∧ p ≥ 3) :
  ∃ N : Nat, ∀ x ≥ N, ∃ i ∈ Finset.range ((p + 3) / 2), 
    ∃ q : Nat, q.Prime ∧ q > p ∧ q ∣ (x + i + 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_divisor_existence_l1570_157021


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1570_157025

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define set A
def A : Set Nat := {0, 1}

-- Define set B
def B : Set Nat := {1, 2, 3}

-- Theorem statement
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {2, 3} := by
  sorry

-- Note: Aᶜ represents the complement of A in the universal set U

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1570_157025


namespace NUMINAMATH_CALUDE_even_function_property_l1570_157068

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_prop : ∀ x, f (x + 2) = x * f x) : 
  f 1 = 0 := by sorry

end NUMINAMATH_CALUDE_even_function_property_l1570_157068


namespace NUMINAMATH_CALUDE_two_digit_integer_problem_l1570_157038

theorem two_digit_integer_problem (n : ℕ) :
  n ≥ 10 ∧ n ≤ 99 →
  (60 + n) / 2 = 60 + n / 100 →
  min 60 n = 59 := by
sorry

end NUMINAMATH_CALUDE_two_digit_integer_problem_l1570_157038


namespace NUMINAMATH_CALUDE_least_possible_xy_l1570_157054

theorem least_possible_xy (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (2 * y) = (1 : ℚ) / 8) :
  (x * y : ℕ) ≥ 128 ∧ ∃ (a b : ℕ+), (a : ℕ) * (b : ℕ) = 128 ∧ (1 : ℚ) / a + (1 : ℚ) / (2 * b) = (1 : ℚ) / 8 :=
sorry

end NUMINAMATH_CALUDE_least_possible_xy_l1570_157054


namespace NUMINAMATH_CALUDE_shoe_factory_production_l1570_157000

/-- The monthly production plan of a shoe factory. -/
def monthly_plan : ℝ := 5000

/-- The production in the first week as a fraction of the monthly plan. -/
def first_week : ℝ := 0.2

/-- The production in the second week as a fraction of the first week's production. -/
def second_week : ℝ := 1.2

/-- The production in the third week as a fraction of the first two weeks' combined production. -/
def third_week : ℝ := 0.6

/-- The production in the fourth week in pairs of shoes. -/
def fourth_week : ℝ := 1480

/-- Theorem stating that the given production schedule results in the monthly plan. -/
theorem shoe_factory_production :
  first_week * monthly_plan +
  second_week * first_week * monthly_plan +
  third_week * (first_week * monthly_plan + second_week * first_week * monthly_plan) +
  fourth_week = monthly_plan := by sorry

end NUMINAMATH_CALUDE_shoe_factory_production_l1570_157000


namespace NUMINAMATH_CALUDE_range_of_m_l1570_157028

/-- A decreasing function on the open interval (-2, 2) -/
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, -2 < x ∧ x < y ∧ y < 2 → f x > f y

theorem range_of_m (f : ℝ → ℝ) (m : ℝ) 
  (h_decreasing : DecreasingFunction f)
  (h_inequality : f (m - 1) > f (2 * m - 1)) :
  0 < m ∧ m < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1570_157028


namespace NUMINAMATH_CALUDE_triangle_property_l1570_157088

theorem triangle_property (a b c A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  b * Real.cos A + Real.sqrt 3 * b * Real.sin A - c - a = 0 →
  b = Real.sqrt 3 →
  B = π / 3 ∧ ∀ (a' c' : ℝ), a' + c' ≤ 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_property_l1570_157088


namespace NUMINAMATH_CALUDE_gcd_228_1995_l1570_157023

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l1570_157023


namespace NUMINAMATH_CALUDE_sequence_sum_problem_l1570_157012

theorem sequence_sum_problem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0) →
  S 5 = 1 / 11 →
  a 1 = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_problem_l1570_157012


namespace NUMINAMATH_CALUDE_california_new_york_ratio_l1570_157073

/-- Proves that the ratio of Coronavirus cases in California to New York is 1:2 --/
theorem california_new_york_ratio : 
  ∀ (california texas : ℕ), 
  california = texas + 400 →
  2000 + california + texas = 3600 →
  california * 2 = 2000 :=
by
  sorry

end NUMINAMATH_CALUDE_california_new_york_ratio_l1570_157073


namespace NUMINAMATH_CALUDE_sin_graph_symmetry_l1570_157081

theorem sin_graph_symmetry (x : ℝ) :
  let f (x : ℝ) := Real.sin (2 * x)
  let g (x : ℝ) := f (x + π / 6)
  ∀ y : ℝ, g (π / 6 - x) = g (π / 6 + x) := by
sorry

end NUMINAMATH_CALUDE_sin_graph_symmetry_l1570_157081


namespace NUMINAMATH_CALUDE_least_four_digit_multiple_l1570_157018

/-- The least 4-digit number divisible by 15, 25, 40, and 75 is 1200 -/
theorem least_four_digit_multiple : ∃ n : ℕ, 
  (n ≥ 1000 ∧ n < 10000) ∧ 
  15 ∣ n ∧ 25 ∣ n ∧ 40 ∣ n ∧ 75 ∣ n ∧
  (∀ m : ℕ, (m ≥ 1000 ∧ m < 10000) ∧ 15 ∣ m ∧ 25 ∣ m ∧ 40 ∣ m ∧ 75 ∣ m → m ≥ n) ∧
  n = 1200 :=
by sorry

end NUMINAMATH_CALUDE_least_four_digit_multiple_l1570_157018


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1570_157007

theorem complex_equation_solution (z : ℂ) (h : z * (1 - Complex.I) = 2) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1570_157007


namespace NUMINAMATH_CALUDE_solve_equation_l1570_157037

theorem solve_equation (t x : ℝ) (h1 : (5 + x) / (t + x) = 2 / 3) (h2 : t = 13) : x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1570_157037


namespace NUMINAMATH_CALUDE_power_of_ten_division_l1570_157039

theorem power_of_ten_division : (10 ^ 8) / (10 * 10 ^ 5) = 100 := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_division_l1570_157039


namespace NUMINAMATH_CALUDE_rectangle_area_l1570_157015

theorem rectangle_area (breadth : ℝ) (h1 : breadth > 0) : 
  let length := 3 * breadth
  let perimeter := 2 * (length + breadth)
  perimeter = 48 → breadth * length = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1570_157015


namespace NUMINAMATH_CALUDE_exists_monochromatic_triangle_l1570_157035

-- Define the vertices of the hexagon
inductive Vertex : Type
  | A | B | C | D | E | F

-- Define the colors
inductive Color : Type
  | Blue | Yellow

-- Define an edge as a pair of vertices
def Edge : Type := Vertex × Vertex

-- Function to get the color of an edge
def edge_color : Edge → Color := sorry

-- Define the hexagon
def hexagon : Set Edge := sorry

-- Theorem statement
theorem exists_monochromatic_triangle :
  ∃ (v1 v2 v3 : Vertex) (c : Color),
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
    edge_color (v1, v2) = c ∧
    edge_color (v2, v3) = c ∧
    edge_color (v1, v3) = c :=
  sorry

end NUMINAMATH_CALUDE_exists_monochromatic_triangle_l1570_157035


namespace NUMINAMATH_CALUDE_line_intercepts_sum_zero_l1570_157024

/-- Given a line l with equation 2x+(k-3)y-2k+6=0 where k ≠ 3,
    if the sum of its x-intercept and y-intercept is 0, then k = 1 -/
theorem line_intercepts_sum_zero (k : ℝ) (h1 : k ≠ 3) :
  (∃ x y : ℝ, 2*x + (k-3)*y - 2*k + 6 = 0) →
  (∃ x_int y_int : ℝ,
    (2*x_int - 2*k + 6 = 0) ∧
    ((k-3)*y_int - 2*k + 6 = 0) ∧
    (x_int + y_int = 0)) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_zero_l1570_157024


namespace NUMINAMATH_CALUDE_information_spread_l1570_157076

theorem information_spread (population : ℕ) (h : population = 1000000) : 
  ∃ (n : ℕ), (2^(n+1) - 1 ≥ population) ∧ (∀ m : ℕ, m < n → 2^(m+1) - 1 < population) :=
sorry

end NUMINAMATH_CALUDE_information_spread_l1570_157076


namespace NUMINAMATH_CALUDE_star_equality_implies_x_equals_three_l1570_157005

/-- Binary operation ⋆ on ordered pairs of integers -/
def star : (Int × Int) → (Int × Int) → (Int × Int) :=
  fun (a, b) (c, d) ↦ (a + c, b - d)

/-- Theorem stating that if (4,5) ⋆ (1,3) = (x,y) ⋆ (2,1), then x = 3 -/
theorem star_equality_implies_x_equals_three (x y : Int) :
  star (4, 5) (1, 3) = star (x, y) (2, 1) → x = 3 := by
  sorry


end NUMINAMATH_CALUDE_star_equality_implies_x_equals_three_l1570_157005


namespace NUMINAMATH_CALUDE_sum_of_series_equals_three_l1570_157056

theorem sum_of_series_equals_three : 
  ∑' k : ℕ+, (k : ℝ)^2 / 2^(k : ℝ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_series_equals_three_l1570_157056


namespace NUMINAMATH_CALUDE_intersection_length_tangent_line_m_range_l1570_157057

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

-- Define circle C
def circle_C (a x y : ℝ) : Prop := x^2 + y^2 + 2*a*x - 2*a*y + 2*a^2 - 4*a = 0

-- Theorem for part 1
theorem intersection_length :
  ∀ (x1 y1 x2 y2 : ℝ),
  circle_O x1 y1 ∧ circle_O x2 y2 ∧
  circle_C 3 x1 y1 ∧ circle_C 3 x2 y2 →
  ((x1 - x2)^2 + (y1 - y2)^2)^(1/2) = Real.sqrt 94 / 3 := by sorry

-- Theorem for part 2
theorem tangent_line_m_range :
  ∀ (a m : ℝ),
  0 < a ∧ a ≤ 4 ∧
  (∃ (x y : ℝ), line_l m x y ∧ circle_C a x y) ∧
  (∀ (x y : ℝ), line_l m x y → (x + a)^2 + (y - a)^2 ≥ 4*a) →
  -1 ≤ m ∧ m ≤ 8 - 4*Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_intersection_length_tangent_line_m_range_l1570_157057


namespace NUMINAMATH_CALUDE_pizza_consumption_order_l1570_157030

structure Sibling where
  name : String
  pizza_fraction : ℚ

def pizza_problem (alex beth cyril eve dan : Sibling) : Prop :=
  alex.name = "Alex" ∧
  beth.name = "Beth" ∧
  cyril.name = "Cyril" ∧
  eve.name = "Eve" ∧
  dan.name = "Dan" ∧
  alex.pizza_fraction = 1/7 ∧
  beth.pizza_fraction = 1/5 ∧
  cyril.pizza_fraction = 1/6 ∧
  eve.pizza_fraction = 1/9 ∧
  dan.pizza_fraction = 1 - (alex.pizza_fraction + beth.pizza_fraction + cyril.pizza_fraction + eve.pizza_fraction)

theorem pizza_consumption_order (alex beth cyril eve dan : Sibling) 
  (h : pizza_problem alex beth cyril eve dan) :
  dan.pizza_fraction > beth.pizza_fraction ∧
  beth.pizza_fraction > cyril.pizza_fraction ∧
  cyril.pizza_fraction > alex.pizza_fraction ∧
  alex.pizza_fraction > eve.pizza_fraction :=
by sorry

end NUMINAMATH_CALUDE_pizza_consumption_order_l1570_157030


namespace NUMINAMATH_CALUDE_dans_initial_money_l1570_157011

/-- The amount of money Dan spent on the candy bar -/
def candy_bar_cost : ℝ := 1.00

/-- The amount of money Dan has left after buying the candy bar -/
def money_left : ℝ := 2.00

/-- Dan's initial amount of money -/
def initial_money : ℝ := candy_bar_cost + money_left

theorem dans_initial_money : initial_money = 3.00 := by sorry

end NUMINAMATH_CALUDE_dans_initial_money_l1570_157011


namespace NUMINAMATH_CALUDE_penelope_greta_ratio_l1570_157072

/-- The amount of food animals eat per day in pounds -/
structure AnimalFood where
  penelope : ℝ
  greta : ℝ
  milton : ℝ
  elmer : ℝ

/-- The conditions given in the problem -/
def problem_conditions (food : AnimalFood) : Prop :=
  food.penelope = 20 ∧
  food.milton = food.greta / 100 ∧
  food.elmer = 4000 * food.milton ∧
  food.elmer = food.penelope + 60

/-- The theorem to be proved -/
theorem penelope_greta_ratio (food : AnimalFood) :
  problem_conditions food → food.penelope / food.greta = 10 := by
  sorry

end NUMINAMATH_CALUDE_penelope_greta_ratio_l1570_157072


namespace NUMINAMATH_CALUDE_choir_average_age_l1570_157027

theorem choir_average_age 
  (num_females : ℕ) (num_males : ℕ) (num_children : ℕ)
  (avg_age_females : ℚ) (avg_age_males : ℚ) (avg_age_children : ℚ)
  (h1 : num_females = 12)
  (h2 : num_males = 20)
  (h3 : num_children = 8)
  (h4 : avg_age_females = 28)
  (h5 : avg_age_males = 38)
  (h6 : avg_age_children = 10) :
  (num_females * avg_age_females + num_males * avg_age_males + num_children * avg_age_children) / 
  (num_females + num_males + num_children : ℚ) = 1176 / 40 := by
  sorry

end NUMINAMATH_CALUDE_choir_average_age_l1570_157027


namespace NUMINAMATH_CALUDE_power_sum_equals_zero_l1570_157045

theorem power_sum_equals_zero : 1^2009 + (-1)^2009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equals_zero_l1570_157045


namespace NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l1570_157014

/-- The number of ways to distribute n indistinguishable balls into k indistinguishable boxes,
    with at least one ball in each box. -/
def distribute_balls (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 2 ways to distribute 6 indistinguishable balls into 4 indistinguishable boxes,
    with at least one ball in each box. -/
theorem distribute_six_balls_four_boxes :
  distribute_balls 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l1570_157014


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l1570_157044

theorem smallest_n_divisibility (x y z : ℕ+) 
  (h1 : x ∣ y^3) 
  (h2 : y ∣ z^3) 
  (h3 : z ∣ x^3) : 
  (∀ n : ℕ, n ≥ 13 → (x*y*z : ℕ) ∣ (x + y + z : ℕ)^n) ∧ 
  (∀ m : ℕ, m < 13 → ∃ a b c : ℕ+, 
    a ∣ b^3 ∧ b ∣ c^3 ∧ c ∣ a^3 ∧ ¬((a*b*c : ℕ) ∣ (a + b + c : ℕ)^m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l1570_157044


namespace NUMINAMATH_CALUDE_scooter_repair_cost_l1570_157092

/-- Proves that the total repair cost is $11,000 given the conditions of Peter's scooter purchase and sale --/
theorem scooter_repair_cost (C : ℝ) : 
  (0.05 * C + 0.10 * C + 0.07 * C = 0.22 * C) →  -- Total repair cost is 22% of C
  (1.25 * C - C - 0.22 * C = 1500) →              -- Profit equation
  0.22 * C = 11000 :=                             -- Total repair cost is $11,000
by sorry

end NUMINAMATH_CALUDE_scooter_repair_cost_l1570_157092


namespace NUMINAMATH_CALUDE_max_quotient_value_l1570_157042

theorem max_quotient_value (a b : ℝ) 
  (ha : 100 ≤ a ∧ a ≤ 300)
  (hb : 800 ≤ b ∧ b ≤ 1600)
  (hab : a + b ≤ 1800) :
  ∃ (a' b' : ℝ), 
    100 ≤ a' ∧ a' ≤ 300 ∧
    800 ≤ b' ∧ b' ≤ 1600 ∧
    a' + b' ≤ 1800 ∧
    b' / a' = 5 ∧
    ∀ (x y : ℝ), 
      100 ≤ x ∧ x ≤ 300 → 
      800 ≤ y ∧ y ≤ 1600 → 
      x + y ≤ 1800 → 
      y / x ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_max_quotient_value_l1570_157042


namespace NUMINAMATH_CALUDE_return_trip_time_l1570_157067

/-- Represents the time for a plane's journey between two cities -/
structure FlightTime where
  against_wind : ℝ  -- Time flying against the wind
  still_air : ℝ     -- Time flying in still air
  with_wind : ℝ     -- Time flying with the wind

/-- Checks if the flight times are valid according to the problem conditions -/
def is_valid_flight (ft : FlightTime) : Prop :=
  ft.against_wind = 75 ∧ ft.with_wind = ft.still_air - 10

/-- Theorem stating the possible return trip times -/
theorem return_trip_time (ft : FlightTime) :
  is_valid_flight ft → ft.with_wind = 15 ∨ ft.with_wind = 50 := by
  sorry

end NUMINAMATH_CALUDE_return_trip_time_l1570_157067


namespace NUMINAMATH_CALUDE_complex_expression_equals_negative_48_l1570_157022

theorem complex_expression_equals_negative_48 : 
  ((-1/2 * (1/100))^5 * (2/3 * (2/100))^4 * (-3/4 * (3/100))^3 * (4/5 * (4/100))^2 * (-5/6 * (5/100))) * (10^30) = -48 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_negative_48_l1570_157022


namespace NUMINAMATH_CALUDE_production_days_calculation_l1570_157008

theorem production_days_calculation (n : ℕ) : 
  (∀ (k : ℕ), k ≤ n → (60 * k : ℝ) = (60 : ℝ) * k) → 
  ((60 * n + 90 : ℝ) / (n + 1) = 62) → 
  n = 14 :=
sorry

end NUMINAMATH_CALUDE_production_days_calculation_l1570_157008


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1570_157093

theorem cubic_equation_solution (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 24*x^3) 
  (h3 : a - b = x) : 
  a = (x*(3 + Real.sqrt 92))/6 ∨ a = (x*(3 - Real.sqrt 92))/6 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1570_157093


namespace NUMINAMATH_CALUDE_complex_cube_sum_div_product_l1570_157069

theorem complex_cube_sum_div_product (x y z : ℂ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 10)
  (h_diff_sq : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 8 := by
sorry

end NUMINAMATH_CALUDE_complex_cube_sum_div_product_l1570_157069


namespace NUMINAMATH_CALUDE_pedros_test_scores_l1570_157019

theorem pedros_test_scores :
  let scores : List ℕ := [92, 91, 89, 85, 78]
  let first_three : List ℕ := [92, 85, 78]
  ∀ (s : List ℕ),
    s.length = 5 →
    s.take 3 = first_three →
    s.sum / s.length = 87 →
    (∀ x ∈ s, x < 100) →
    s.Nodup →
    s = scores :=
by sorry

end NUMINAMATH_CALUDE_pedros_test_scores_l1570_157019


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_sum_sqrt_l1570_157089

theorem sqrt_sum_equals_sqrt_of_sum_sqrt (a b : ℚ) :
  (Real.sqrt a + Real.sqrt b = Real.sqrt (2 + Real.sqrt 3)) ↔
  ((a = 1/2 ∧ b = 3/2) ∨ (a = 3/2 ∧ b = 1/2)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_sqrt_of_sum_sqrt_l1570_157089


namespace NUMINAMATH_CALUDE_palindrome_with_seven_percentage_l1570_157029

-- Define a palindrome in the range 100 to 999
def IsPalindrome (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

-- Define a number containing at least one 7
def ContainsSeven (n : Nat) : Prop :=
  (n / 100 = 7) ∨ ((n / 10) % 10 = 7) ∨ (n % 10 = 7)

-- Count of palindromes with at least one 7
def PalindromeWithSeven : Nat :=
  19

-- Total count of palindromes between 100 and 999
def TotalPalindromes : Nat :=
  90

-- Theorem statement
theorem palindrome_with_seven_percentage :
  (PalindromeWithSeven : ℚ) / TotalPalindromes = 19 / 90 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_with_seven_percentage_l1570_157029


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1570_157002

theorem arithmetic_calculations :
  (12 - (-18) + (-7) - 15 = 8) ∧
  (5 + 1 / 7 : ℚ) * (7 / 8 : ℚ) / (-8 / 9 : ℚ) / 3 = -27 / 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1570_157002


namespace NUMINAMATH_CALUDE_g_3_equals_9_l1570_157082

-- Define the function g
def g (x : ℝ) : ℝ := 3*x^6 - 2*x^4 + 5*x^2 - 7

-- Theorem statement
theorem g_3_equals_9 (h : g (-3) = 9) : g 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_g_3_equals_9_l1570_157082


namespace NUMINAMATH_CALUDE_problem_statement_l1570_157070

theorem problem_statement : 
  let A := (16 * Real.sqrt 2) ^ (1/3 : ℝ)
  let B := Real.sqrt (9 * 9 ^ (1/3 : ℝ))
  let C := ((2 ^ (1/5 : ℝ)) ^ 2) ^ 2
  A ^ 2 + B ^ 3 + C ^ 5 = 105 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l1570_157070


namespace NUMINAMATH_CALUDE_function_inequality_condition_l1570_157020

open Real

theorem function_inequality_condition (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ →
    (a * exp x₁ / x₁ - x₁) / x₂ - (a * exp x₂ / x₂ - x₂) / x₁ < 0) ↔
  a ≥ -exp 1 :=
sorry

end NUMINAMATH_CALUDE_function_inequality_condition_l1570_157020


namespace NUMINAMATH_CALUDE_sugar_price_increase_l1570_157077

theorem sugar_price_increase (initial_price : ℝ) (initial_quantity : ℝ) : 
  initial_quantity > 0 →
  initial_price > 0 →
  initial_price * initial_quantity = 5 * (0.4 * initial_quantity) →
  initial_price = 2 := by
sorry

end NUMINAMATH_CALUDE_sugar_price_increase_l1570_157077


namespace NUMINAMATH_CALUDE_invitation_combinations_l1570_157061

theorem invitation_combinations (n m : ℕ) (h : n = 10 ∧ m = 6) : 
  (Nat.choose n m) - (Nat.choose (n - 2) (m - 2)) = 140 :=
sorry

end NUMINAMATH_CALUDE_invitation_combinations_l1570_157061


namespace NUMINAMATH_CALUDE_two_quadrilaterals_nine_regions_l1570_157026

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral in the plane -/
structure Quadrilateral :=
  (p1 p2 p3 p4 : Point)

/-- The plane divided by quadrilaterals -/
def PlaneDivision :=
  List Quadrilateral

/-- Count the number of regions in a plane division -/
def countRegions (division : PlaneDivision) : ℕ :=
  sorry

/-- Theorem: There exists a plane division with two quadrilaterals that results in 9 regions -/
theorem two_quadrilaterals_nine_regions :
  ∃ (division : PlaneDivision),
    division.length = 2 ∧ countRegions division = 9 :=
  sorry

end NUMINAMATH_CALUDE_two_quadrilaterals_nine_regions_l1570_157026


namespace NUMINAMATH_CALUDE_max_marks_calculation_l1570_157090

/-- Given a passing threshold, a student's score, and the shortfall to pass,
    calculate the maximum possible marks. -/
theorem max_marks_calculation (passing_threshold : ℚ) (score : ℕ) (shortfall : ℕ) :
  passing_threshold = 30 / 100 →
  score = 212 →
  shortfall = 28 →
  (score + shortfall) / passing_threshold = 800 :=
by sorry

end NUMINAMATH_CALUDE_max_marks_calculation_l1570_157090


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l1570_157003

/-- The number of people sharing the box of chocolate bars -/
def num_people : ℕ := 3

/-- The number of bars two people got combined -/
def bars_two_people : ℕ := 8

/-- The total number of bars in the box -/
def total_bars : ℕ := 16

/-- Theorem stating that the total number of bars is 16 -/
theorem chocolate_bar_count :
  (num_people : ℕ) = 3 →
  (bars_two_people : ℕ) = 8 →
  (total_bars : ℕ) = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l1570_157003


namespace NUMINAMATH_CALUDE_ship_journey_day1_distance_l1570_157053

/-- Represents the distance traveled by a ship over three days -/
structure ShipJourney where
  day1_north : ℝ
  day2_east : ℝ
  day3_east : ℝ

/-- Calculates the total distance traveled by the ship -/
def total_distance (journey : ShipJourney) : ℝ :=
  journey.day1_north + journey.day2_east + journey.day3_east

/-- Theorem stating the distance traveled north on the first day -/
theorem ship_journey_day1_distance :
  ∀ (journey : ShipJourney),
    journey.day2_east = 3 * journey.day1_north →
    journey.day3_east = journey.day2_east + 110 →
    total_distance journey = 810 →
    journey.day1_north = 100 := by
  sorry

end NUMINAMATH_CALUDE_ship_journey_day1_distance_l1570_157053


namespace NUMINAMATH_CALUDE_third_year_sample_size_l1570_157040

/-- Represents the number of students to be selected in a stratified sampling -/
def sample_size : ℕ := 200

/-- Represents the total number of first-year students -/
def first_year_students : ℕ := 700

/-- Represents the total number of second-year students -/
def second_year_students : ℕ := 670

/-- Represents the total number of third-year students -/
def third_year_students : ℕ := 630

/-- Represents the total number of students in all three years -/
def total_students : ℕ := first_year_students + second_year_students + third_year_students

/-- Theorem stating that the number of third-year students to be selected in the stratified sampling is 63 -/
theorem third_year_sample_size :
  (sample_size * third_year_students) / total_students = 63 := by
  sorry

end NUMINAMATH_CALUDE_third_year_sample_size_l1570_157040


namespace NUMINAMATH_CALUDE_sum_in_arithmetic_sequence_l1570_157009

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_in_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 3 + a 7 = 37 →
  a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_sum_in_arithmetic_sequence_l1570_157009


namespace NUMINAMATH_CALUDE_sum_of_ages_l1570_157058

theorem sum_of_ages (bella_age : ℕ) (age_difference : ℕ) : 
  bella_age = 5 → 
  age_difference = 9 → 
  bella_age + (bella_age + age_difference) = 19 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1570_157058


namespace NUMINAMATH_CALUDE_wire_length_l1570_157032

/-- Given that a 75-meter roll of wire weighs 15 kg, 
    this theorem proves that a roll weighing 5 kg is 25 meters long. -/
theorem wire_length (weight : ℝ) (length : ℝ) : 
  (75 : ℝ) / 15 = length / weight → weight = 5 → length = 25 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_l1570_157032


namespace NUMINAMATH_CALUDE_tangent_line_intersection_l1570_157071

noncomputable section

def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m
def g (x : ℝ) : ℝ := 6 * Real.log x - 4 * x

theorem tangent_line_intersection (m : ℝ) : 
  (∃ a : ℝ, a > 0 ∧ 
    f m a = g a ∧ 
    (deriv (f m)) a = (deriv g) a) → 
  m = -5 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_line_intersection_l1570_157071


namespace NUMINAMATH_CALUDE_infinite_solutions_condition_l1570_157051

theorem infinite_solutions_condition (b : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - b) = 3 * (4 * x + 8)) ↔ b = -6 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_condition_l1570_157051


namespace NUMINAMATH_CALUDE_roots_quadratic_equation_l1570_157096

theorem roots_quadratic_equation (α β : ℝ) : 
  (α^2 - 3*α - 2 = 0) → 
  (β^2 - 3*β - 2 = 0) → 
  7*α^4 + 10*β^3 = 544 := by
sorry

end NUMINAMATH_CALUDE_roots_quadratic_equation_l1570_157096


namespace NUMINAMATH_CALUDE_sachin_age_l1570_157049

/-- Represents the ages of Sachin, Rahul, and Praveen -/
structure Ages where
  sachin : ℝ
  rahul : ℝ
  praveen : ℝ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.rahul = ages.sachin + 7 ∧
  ages.sachin / ages.rahul = 7 / 9 ∧
  ages.praveen = 2 * ages.rahul ∧
  ages.sachin / ages.rahul = 7 / 9 ∧
  ages.rahul / ages.praveen = 9 / 18

/-- Theorem stating that if the ages satisfy the conditions, then Sachin's age is 24.5 -/
theorem sachin_age (ages : Ages) : 
  satisfiesConditions ages → ages.sachin = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_sachin_age_l1570_157049


namespace NUMINAMATH_CALUDE_sqrt_64_equals_8_l1570_157048

theorem sqrt_64_equals_8 : Real.sqrt 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_64_equals_8_l1570_157048


namespace NUMINAMATH_CALUDE_sum_first_ten_primes_with_units_digit_three_l1570_157098

/-- A function that returns true if a number has a units digit of 3 -/
def hasUnitsDigitThree (n : ℕ) : Bool :=
  n % 10 = 3

/-- The sequence of prime numbers with a units digit of 3 -/
def primesWithUnitsDigitThree : List ℕ :=
  (List.range 200).filter (λ n => n.Prime && hasUnitsDigitThree n)

/-- The sum of the first ten prime numbers with a units digit of 3 -/
def sumFirstTenPrimesWithUnitsDigitThree : ℕ :=
  (primesWithUnitsDigitThree.take 10).sum

theorem sum_first_ten_primes_with_units_digit_three :
  sumFirstTenPrimesWithUnitsDigitThree = 639 := by
  sorry


end NUMINAMATH_CALUDE_sum_first_ten_primes_with_units_digit_three_l1570_157098


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1570_157062

theorem cube_root_simplification :
  (1 + 27) ^ (1/3) * (1 + 27 ^ (1/3)) ^ (1/3) = 112 ^ (1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1570_157062


namespace NUMINAMATH_CALUDE_vector_parallelism_l1570_157059

theorem vector_parallelism (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![1, x]
  (∃ (k : ℝ), k ≠ 0 ∧ (a + b) = k • (2 • a - b)) → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallelism_l1570_157059


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l1570_157064

theorem merry_go_round_revolutions (outer_radius inner_radius : ℝ) 
  (outer_revolutions : ℕ) (h1 : outer_radius = 30) (h2 : inner_radius = 10) 
  (h3 : outer_revolutions = 25) : 
  ∃ inner_revolutions : ℕ, 
    inner_revolutions * inner_radius * 2 * Real.pi = outer_revolutions * outer_radius * 2 * Real.pi ∧ 
    inner_revolutions = 75 := by
  sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l1570_157064


namespace NUMINAMATH_CALUDE_fraction_simplification_l1570_157065

theorem fraction_simplification (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (8 * a^4 * b^2 * c) / (4 * a^3 * b) = 2 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1570_157065


namespace NUMINAMATH_CALUDE_log_inequality_range_l1570_157047

open Real

theorem log_inequality_range (f : ℝ → ℝ) (t : ℝ) : 
  (∀ x > 0, f x = log x) →
  (∀ x > 0, f x + f t ≤ f (x^2 + t)) →
  0 < t ∧ t ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_range_l1570_157047


namespace NUMINAMATH_CALUDE_gcd_288_123_l1570_157034

theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_288_123_l1570_157034


namespace NUMINAMATH_CALUDE_batsman_average_after_15th_innings_l1570_157010

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored) / (b.innings + 1)

theorem batsman_average_after_15th_innings 
  (b : Batsman)
  (h1 : b.innings = 14)
  (h2 : newAverage b 85 = b.average + 3) :
  newAverage b 85 = 43 := by
  sorry

#check batsman_average_after_15th_innings

end NUMINAMATH_CALUDE_batsman_average_after_15th_innings_l1570_157010


namespace NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l1570_157017

theorem smallest_sum_of_coefficients (a b : ℝ) : 
  a > 0 → b > 0 → 
  (∃ x : ℝ, x^2 + 2*a*x + 3*b = 0) →
  (∃ y : ℝ, y^2 + 3*b*y + 2*a = 0) →
  a + b ≥ 5 := by
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_coefficients_l1570_157017


namespace NUMINAMATH_CALUDE_theater_sales_total_cost_l1570_157043

/-- Represents the theater ticket sales problem --/
structure TheaterSales where
  total_tickets : ℕ
  balcony_surplus : ℕ
  orchestra_price : ℕ
  balcony_price : ℕ

/-- Calculate the total cost of tickets sold --/
def total_cost (sales : TheaterSales) : ℕ :=
  let orchestra_tickets := (sales.total_tickets - sales.balcony_surplus) / 2
  let balcony_tickets := sales.total_tickets - orchestra_tickets
  orchestra_tickets * sales.orchestra_price + balcony_tickets * sales.balcony_price

/-- Theorem stating that the total cost for the given conditions is $3320 --/
theorem theater_sales_total_cost :
  let sales : TheaterSales := {
    total_tickets := 370,
    balcony_surplus := 190,
    orchestra_price := 12,
    balcony_price := 8
  }
  total_cost sales = 3320 := by
  sorry


end NUMINAMATH_CALUDE_theater_sales_total_cost_l1570_157043


namespace NUMINAMATH_CALUDE_trajectory_is_straight_line_l1570_157095

/-- The set of points (x, y) in ℝ² where x + y = 0 forms a straight line -/
theorem trajectory_is_straight_line :
  {p : ℝ × ℝ | p.1 + p.2 = 0} = {p : ℝ × ℝ | ∃ (t : ℝ), p = (t, -t)} := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_straight_line_l1570_157095


namespace NUMINAMATH_CALUDE_sampling_methods_classification_l1570_157094

-- Define the characteristics of sampling methods
def is_systematic_sampling (method : String) : Prop :=
  method = "Samples at equal time intervals"

def is_simple_random_sampling (method : String) : Prop :=
  method = "Selects individuals from a small population with little difference among them"

-- Define the two sampling methods
def sampling_method_1 : String :=
  "Samples a bag for inspection every 30 minutes in a milk production line"

def sampling_method_2 : String :=
  "Selects 3 students from a group of 30 math enthusiasts in a middle school"

-- Theorem to prove
theorem sampling_methods_classification :
  is_systematic_sampling sampling_method_1 ∧
  is_simple_random_sampling sampling_method_2 := by
  sorry


end NUMINAMATH_CALUDE_sampling_methods_classification_l1570_157094


namespace NUMINAMATH_CALUDE_range_of_a_l1570_157033

theorem range_of_a : ∀ a : ℝ, (∀ x : ℝ, x > 1 → x > a) ∧ (∃ x : ℝ, x > a ∧ x ≤ 1) ↔ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1570_157033


namespace NUMINAMATH_CALUDE_unique_non_expressible_l1570_157001

/-- Checks if a number can be expressed as x^2 + y^5 for some integers x and y -/
def isExpressible (n : ℤ) : Prop :=
  ∃ x y : ℤ, n = x^2 + y^5

/-- The list of numbers to check -/
def numberList : List ℤ := [59170, 59149, 59130, 59121, 59012]

/-- Theorem stating that 59121 is the only number in the list that cannot be expressed as x^2 + y^5 -/
theorem unique_non_expressible :
  ∀ n ∈ numberList, n ≠ 59121 → isExpressible n ∧ ¬isExpressible 59121 := by
  sorry

end NUMINAMATH_CALUDE_unique_non_expressible_l1570_157001


namespace NUMINAMATH_CALUDE_tetrahedron_max_volume_edge_ratio_l1570_157086

/-- Given a tetrahedron with volume V and edge lengths a, b, c, d where no three edges are coplanar,
    and L = a + b + c + d, the maximum value of V/L^3 is √2/2592 -/
theorem tetrahedron_max_volume_edge_ratio :
  ∀ (V a b c d L : ℝ),
  V > 0 → a > 0 → b > 0 → c > 0 → d > 0 →
  (∀ (x y z : ℝ), x + y + z ≠ a + b + c + d) →  -- No three edges are coplanar
  L = a + b + c + d →
  (∃ (V' : ℝ), V' = V ∧ V' / L^3 ≤ Real.sqrt 2 / 2592) :=
by sorry

end NUMINAMATH_CALUDE_tetrahedron_max_volume_edge_ratio_l1570_157086


namespace NUMINAMATH_CALUDE_yard_length_is_700_l1570_157060

/-- The length of a yard with trees planted at equal distances -/
def yard_length (num_trees : ℕ) (tree_distance : ℕ) : ℕ :=
  (num_trees - 1) * tree_distance

/-- Proof that the yard length is 700 meters -/
theorem yard_length_is_700 :
  yard_length 26 28 = 700 := by
  sorry

end NUMINAMATH_CALUDE_yard_length_is_700_l1570_157060


namespace NUMINAMATH_CALUDE_bread_slices_proof_l1570_157013

/-- The number of slices Andy ate at each time -/
def slices_eaten_per_time : ℕ := 3

/-- The number of times Andy ate slices -/
def times_andy_ate : ℕ := 2

/-- The number of slices needed to make one piece of toast bread -/
def slices_per_toast : ℕ := 2

/-- The number of pieces of toast bread made -/
def toast_pieces_made : ℕ := 10

/-- The number of slices left after making toast -/
def slices_left : ℕ := 1

/-- The total number of slices in the original loaf of bread -/
def total_slices : ℕ := 27

theorem bread_slices_proof :
  total_slices = 
    slices_eaten_per_time * times_andy_ate + 
    slices_per_toast * toast_pieces_made + 
    slices_left :=
by
  sorry

end NUMINAMATH_CALUDE_bread_slices_proof_l1570_157013


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1570_157050

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n > 2 → 
  interior_angle = 160 →
  (n : ℝ) * interior_angle = 180 * (n - 2) →
  n = 18 := by
sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1570_157050


namespace NUMINAMATH_CALUDE_school_boys_count_l1570_157052

theorem school_boys_count (muslim_percent : ℝ) (hindu_percent : ℝ) (sikh_percent : ℝ) (other_count : ℕ) :
  muslim_percent = 0.44 →
  hindu_percent = 0.28 →
  sikh_percent = 0.10 →
  other_count = 72 →
  ∃ (total : ℕ), 
    (muslim_percent + hindu_percent + sikh_percent + (other_count : ℝ) / total) = 1 ∧
    total = 400 := by
  sorry

end NUMINAMATH_CALUDE_school_boys_count_l1570_157052


namespace NUMINAMATH_CALUDE_percentage_reading_both_books_l1570_157004

theorem percentage_reading_both_books (total_students : ℕ) 
  (read_A : ℕ) (read_B : ℕ) (read_both : ℕ) :
  total_students = 600 →
  read_both = (20 * read_A) / 100 →
  read_A + read_B - read_both = total_students →
  read_A - read_both - (read_B - read_both) = 75 →
  (read_both * 100) / read_B = 25 :=
by sorry

end NUMINAMATH_CALUDE_percentage_reading_both_books_l1570_157004


namespace NUMINAMATH_CALUDE_new_ratio_second_term_l1570_157063

theorem new_ratio_second_term 
  (a b x : ℤ) 
  (h1 : a = 7)
  (h2 : b = 11)
  (h3 : x = 5)
  (h4 : a + x = 3) :
  ∃ y : ℤ, (a + x) * y = 3 * (b + x) ∧ y = 16 := by
  sorry

end NUMINAMATH_CALUDE_new_ratio_second_term_l1570_157063


namespace NUMINAMATH_CALUDE_greatest_base6_digit_sum_l1570_157074

/-- Represents a base-6 digit -/
def Base6Digit := Fin 6

/-- Converts a natural number to its base-6 representation -/
def toBase6 (n : ℕ) : List Base6Digit :=
  sorry

/-- Calculates the sum of digits in a list -/
def digitSum (digits : List Base6Digit) : ℕ :=
  sorry

/-- Theorem: The greatest possible sum of digits in the base-6 representation
    of a positive integer less than 1728 is 20 -/
theorem greatest_base6_digit_sum :
  ∃ (n : ℕ), n > 0 ∧ n < 1728 ∧
  digitSum (toBase6 n) = 20 ∧
  ∀ (m : ℕ), m > 0 → m < 1728 → digitSum (toBase6 m) ≤ 20 :=
sorry

end NUMINAMATH_CALUDE_greatest_base6_digit_sum_l1570_157074
