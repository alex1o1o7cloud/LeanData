import Mathlib

namespace batsman_average_l2981_298170

/-- Represents a batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  averageIncrease : Nat

/-- Calculates the average score of a batsman -/
def average (b : Batsman) : Nat :=
  b.totalRuns / b.innings

/-- Theorem: Given the conditions, prove that the batsman's average after 10 innings is 33 -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 10)
  (h2 : b.totalRuns = (average b * 9) + 60)
  (h3 : average { innings := b.innings, totalRuns := b.totalRuns, averageIncrease := b.averageIncrease } = 
        average { innings := b.innings - 1, totalRuns := b.totalRuns - 60, averageIncrease := b.averageIncrease } + 3) :
  average b = 33 := by
  sorry


end batsman_average_l2981_298170


namespace sqrt_65_bounds_l2981_298108

theorem sqrt_65_bounds (n : ℕ+) : n < Real.sqrt 65 ∧ Real.sqrt 65 < n + 1 → n = 8 := by
  sorry

end sqrt_65_bounds_l2981_298108


namespace binary_remainder_by_8_l2981_298190

/-- The remainder when 101110100101₂ is divided by 8 is 5. -/
theorem binary_remainder_by_8 : (101110100101 : Nat) % 8 = 5 := by
  sorry

end binary_remainder_by_8_l2981_298190


namespace interior_angles_theorem_l2981_298121

/-- The sum of interior angles of a convex polygon with n sides, in degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: Given a convex polygon with n sides where the sum of interior angles is 3600 degrees,
    the sum of interior angles for a polygon with n+2 sides is 3960 degrees,
    and for a polygon with n-2 sides is 3240 degrees. -/
theorem interior_angles_theorem (n : ℕ) (h : sum_interior_angles n = 3600) :
  sum_interior_angles (n + 2) = 3960 ∧ sum_interior_angles (n - 2) = 3240 := by
  sorry

#check interior_angles_theorem

end interior_angles_theorem_l2981_298121


namespace average_of_xyz_l2981_298187

theorem average_of_xyz (x y z : ℝ) (h : (5 / 4) * (x + y + z) = 15) : 
  (x + y + z) / 3 = 4 := by
sorry

end average_of_xyz_l2981_298187


namespace distinct_paths_theorem_l2981_298138

/-- The number of distinct paths in a rectangular grid from point C to point D -/
def distinct_paths (right_steps : ℕ) (up_steps : ℕ) : ℕ :=
  Nat.choose (right_steps + up_steps) up_steps

/-- Theorem: The number of distinct paths from C to D is equal to (10 choose 3) -/
theorem distinct_paths_theorem :
  distinct_paths 7 3 = 120 := by
  sorry

end distinct_paths_theorem_l2981_298138


namespace area_equality_l2981_298180

-- Define a square
structure Square :=
  (A B C D : Point)

-- Define the property of being inside a square
def InsideSquare (P : Point) (s : Square) : Prop := sorry

-- Define the angle between three points
def Angle (P Q R : Point) : ℝ := sorry

-- Define the area of a triangle
def TriangleArea (P Q R : Point) : ℝ := sorry

-- State the theorem
theorem area_equality (s : Square) (P Q : Point) 
  (h_inside_P : InsideSquare P s)
  (h_inside_Q : InsideSquare Q s)
  (h_angle_PAQ : Angle s.A P Q = 45)
  (h_angle_PCQ : Angle s.C P Q = 45) :
  TriangleArea P s.A s.B + TriangleArea P s.C Q + TriangleArea Q s.A s.D =
  TriangleArea Q s.C s.D + TriangleArea P s.A Q + TriangleArea P s.B s.C :=
sorry

end area_equality_l2981_298180


namespace undeveloped_sections_l2981_298194

/-- Proves that the number of undeveloped sections is 3 given the specified conditions -/
theorem undeveloped_sections
  (section_area : ℝ)
  (total_undeveloped_area : ℝ)
  (h1 : section_area = 2435)
  (h2 : total_undeveloped_area = 7305) :
  total_undeveloped_area / section_area = 3 := by
  sorry

end undeveloped_sections_l2981_298194


namespace tan_80_plus_tan_40_minus_sqrt3_tan_80_tan_40_l2981_298173

theorem tan_80_plus_tan_40_minus_sqrt3_tan_80_tan_40 :
  let t80 := Real.tan (80 * π / 180)
  let t40 := Real.tan (40 * π / 180)
  t80 + t40 - Real.sqrt 3 * t80 * t40 = -Real.sqrt 3 := by sorry

end tan_80_plus_tan_40_minus_sqrt3_tan_80_tan_40_l2981_298173


namespace square_of_sum_31_3_l2981_298196

theorem square_of_sum_31_3 : 31^2 + 2*(31*3) + 3^2 = 1156 := by
  sorry

end square_of_sum_31_3_l2981_298196


namespace smallest_prime_in_sum_l2981_298116

theorem smallest_prime_in_sum (p q r s : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → Nat.Prime s →
  p + q + r = 2 * s →
  1 < p → p < q → q < r →
  p = 2 := by
  sorry

end smallest_prime_in_sum_l2981_298116


namespace problem_solution_l2981_298119

/-- The probability that student A solves the problem -/
def prob_A : ℚ := 1/5

/-- The probability that student B solves the problem -/
def prob_B : ℚ := 1/4

/-- The probability that student C solves the problem -/
def prob_C : ℚ := 1/3

/-- The probability that exactly two students solve the problem -/
def prob_two_solve : ℚ := 
  prob_A * prob_B * (1 - prob_C) + 
  prob_A * prob_C * (1 - prob_B) + 
  (1 - prob_A) * prob_B * prob_C

/-- The probability that the problem is not solved -/
def prob_not_solved : ℚ := (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- The probability that the problem is solved -/
def prob_solved : ℚ := 1 - prob_not_solved

theorem problem_solution : 
  prob_two_solve = 3/20 ∧ prob_solved = 3/5 := by sorry

end problem_solution_l2981_298119


namespace triangle_angle_A_l2981_298139

theorem triangle_angle_A (A : Real) : 
  4 * Real.pi * Real.sin A - 3 * Real.arccos (-1/2) = 0 →
  (A = Real.pi / 6 ∨ A = 5 * Real.pi / 6) :=
by sorry

end triangle_angle_A_l2981_298139


namespace greatest_four_digit_number_with_remainders_l2981_298153

theorem greatest_four_digit_number_with_remainders :
  ∃ n : ℕ,
    n ≤ 9999 ∧
    n > 999 ∧
    n % 15 = 2 ∧
    n % 24 = 8 ∧
    (∀ m : ℕ, m ≤ 9999 ∧ m > 999 ∧ m % 15 = 2 → m ≤ n) ∧
    n = 9992 :=
by sorry

end greatest_four_digit_number_with_remainders_l2981_298153


namespace chess_tournament_games_l2981_298132

theorem chess_tournament_games (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 7) : 
  (n * k) / 2 = 35 := by
  sorry

end chess_tournament_games_l2981_298132


namespace number_of_pupils_l2981_298161

/-- Given a program with parents and pupils, calculate the number of pupils -/
theorem number_of_pupils (total_people parents : ℕ) (h1 : parents = 105) (h2 : total_people = 803) :
  total_people - parents = 698 := by
  sorry

end number_of_pupils_l2981_298161


namespace compare_roots_l2981_298145

theorem compare_roots : (2 * Real.sqrt 6 < 5) ∧ (-Real.sqrt 5 < -Real.sqrt 2) := by
  sorry

end compare_roots_l2981_298145


namespace largest_expression_l2981_298118

theorem largest_expression : 
  let expr1 := 2 + (-2)
  let expr2 := 2 - (-2)
  let expr3 := 2 * (-2)
  let expr4 := 2 / (-2)
  expr2 = max expr1 (max expr2 (max expr3 expr4)) := by sorry

end largest_expression_l2981_298118


namespace twelfth_day_is_monday_l2981_298184

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with specific properties -/
structure Month where
  firstDay : DayOfWeek
  lastDay : DayOfWeek
  numDays : Nat
  numFridays : Nat

/-- Given a starting day and a number of days, calculates the resulting day of the week -/
def advanceDays (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  sorry

/-- Theorem stating that under given conditions, the 12th day of the month is a Monday -/
theorem twelfth_day_is_monday (m : Month) 
  (h1 : m.numFridays = 5)
  (h2 : m.firstDay ≠ DayOfWeek.Friday)
  (h3 : m.lastDay ≠ DayOfWeek.Friday)
  (h4 : m.numDays ≥ 12) :
  advanceDays m.firstDay 11 = DayOfWeek.Monday :=
  sorry

end twelfth_day_is_monday_l2981_298184


namespace range_of_3a_plus_4b_l2981_298134

theorem range_of_3a_plus_4b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : 2 * a + 2 * b ≤ 15) (h2 : 4 / a + 3 / b ≤ 2) :
  ∃ (min max : ℝ), min = 24 ∧ max = 27 ∧
  (∀ x, (∃ a' b' : ℝ, a' > 0 ∧ b' > 0 ∧
    2 * a' + 2 * b' ≤ 15 ∧ 4 / a' + 3 / b' ≤ 2 ∧
    x = 3 * a' + 4 * b') → min ≤ x ∧ x ≤ max) :=
sorry

end range_of_3a_plus_4b_l2981_298134


namespace parabola_uniqueness_l2981_298133

/-- A tangent line to a parabola -/
structure Tangent where
  line : Line2D

/-- A parabola in 2D space -/
structure Parabola where
  focus : Point2D
  directrix : Line2D

/-- The vertex tangent of a parabola -/
def vertexTangent (p : Parabola) : Tangent :=
  sorry

/-- Determines if a given tangent is valid for a parabola -/
def isValidTangent (p : Parabola) (t : Tangent) : Prop :=
  sorry

theorem parabola_uniqueness 
  (t : Tangent) (t₁ : Tangent) (t₂ : Tangent) : 
  ∃! p : Parabola, 
    (vertexTangent p = t) ∧ 
    (isValidTangent p t₁) ∧ 
    (isValidTangent p t₂) :=
sorry

end parabola_uniqueness_l2981_298133


namespace inequality_equivalence_l2981_298106

theorem inequality_equivalence (m : ℝ) : (3 * m - 4 < 6) ↔ (m < 6) := by sorry

end inequality_equivalence_l2981_298106


namespace new_ratio_first_term_l2981_298126

/-- Given an original ratio of 7:11, when 5 is added to both terms, 
    the first term of the new ratio is 12. -/
theorem new_ratio_first_term : 
  let original_first : ℕ := 7
  let original_second : ℕ := 11
  let added_number : ℕ := 5
  let new_first : ℕ := original_first + added_number
  new_first = 12 := by sorry

end new_ratio_first_term_l2981_298126


namespace infinite_occurrences_l2981_298185

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => 
    let prev := a n
    let god := (n + 1).factorization.prod (λ p k => p)  -- greatest odd divisor
    if god % 4 = 1 then prev + 1 else prev - 1

-- State the theorem
theorem infinite_occurrences :
  (∀ k : ℕ+, Set.Infinite {n : ℕ | a n = k}) ∧
  Set.Infinite {n : ℕ | a n = 1} := by
  sorry

end infinite_occurrences_l2981_298185


namespace polynomial_remainder_l2981_298158

theorem polynomial_remainder (x : ℝ) : 
  (x^3 - 2*x^2 + 4*x - 1) % (x - 2) = 7 := by
  sorry

end polynomial_remainder_l2981_298158


namespace apple_bags_theorem_l2981_298137

def is_valid_total (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ (∃ k : ℕ, n = 6 * k)

theorem apple_bags_theorem :
  ∀ n : ℕ, is_valid_total n ↔ (n = 72 ∨ n = 78) :=
by sorry

end apple_bags_theorem_l2981_298137


namespace largest_negative_integer_negation_l2981_298124

theorem largest_negative_integer_negation (x : ℤ) : 
  (∀ y : ℤ, y < 0 → y ≤ x) ∧ x < 0 → -(-(-x)) = 1 := by
  sorry

end largest_negative_integer_negation_l2981_298124


namespace table_seating_theorem_l2981_298101

/-- Represents the setup of people around a round table -/
structure TableSetup where
  num_men : ℕ
  num_women : ℕ

/-- Calculates the probability of a specific man being satisfied -/
def prob_man_satisfied (setup : TableSetup) : ℚ :=
  1 - (setup.num_men - 1) / (setup.num_men + setup.num_women - 1) *
      (setup.num_men - 2) / (setup.num_men + setup.num_women - 2)

/-- Calculates the expected number of satisfied men -/
def expected_satisfied_men (setup : TableSetup) : ℚ :=
  setup.num_men * prob_man_satisfied setup

/-- Main theorem about the probability and expectation in the given setup -/
theorem table_seating_theorem (setup : TableSetup) 
    (h_men : setup.num_men = 50) (h_women : setup.num_women = 50) : 
    prob_man_satisfied setup = 25 / 33 ∧ 
    expected_satisfied_men setup = 1250 / 33 := by
  sorry

#eval prob_man_satisfied ⟨50, 50⟩
#eval expected_satisfied_men ⟨50, 50⟩

end table_seating_theorem_l2981_298101


namespace complex_square_equality_l2981_298115

theorem complex_square_equality (c d : ℕ+) :
  (c + d * Complex.I) ^ 2 = 7 + 24 * Complex.I →
  c + d * Complex.I = 4 + 3 * Complex.I := by
  sorry

end complex_square_equality_l2981_298115


namespace line_equation_l2981_298197

/-- Given a line with an angle of inclination of 45° and a y-intercept of 2,
    its equation is x - y + 2 = 0 -/
theorem line_equation (angle : ℝ) (y_intercept : ℝ) :
  angle = 45 ∧ y_intercept = 2 →
  ∀ x y : ℝ, (y = x + y_intercept) ↔ (x - y + y_intercept = 0) :=
by sorry

end line_equation_l2981_298197


namespace general_term_formula_l2981_298136

def S (n : ℕ) : ℤ := 3 * n^2 - 2 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2 else 6 * n - 5

theorem general_term_formula (n : ℕ) :
  (n = 1 ∧ a n = S n) ∨
  (n ≥ 2 ∧ a n = S n - S (n-1)) :=
sorry

end general_term_formula_l2981_298136


namespace total_birds_l2981_298189

theorem total_birds (cardinals : ℕ) (robins : ℕ) (blue_jays : ℕ) (sparrows : ℕ) (pigeons : ℕ) (finches : ℕ) : 
  cardinals = 3 →
  robins = 4 * cardinals →
  blue_jays = 2 * cardinals →
  sparrows = 3 * cardinals + 1 →
  pigeons = 3 * blue_jays →
  finches = robins / 2 →
  cardinals + robins + blue_jays + sparrows + pigeons + finches = 55 := by
sorry

end total_birds_l2981_298189


namespace retirement_percentage_l2981_298157

def gross_pay : ℝ := 1120
def tax_deduction : ℝ := 100
def net_pay : ℝ := 740

theorem retirement_percentage :
  (gross_pay - net_pay - tax_deduction) / gross_pay * 100 = 25 := by
  sorry

end retirement_percentage_l2981_298157


namespace slope_range_for_intersection_l2981_298152

/-- A line with slope k intersects a hyperbola at two distinct points -/
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ ∧ y₂ = k * x₂ ∧
    x₁^2 - y₁^2 = 2 ∧ x₂^2 - y₂^2 = 2

/-- The theorem stating the range of k for which the line intersects the hyperbola at two points -/
theorem slope_range_for_intersection :
  ∀ k : ℝ, intersects_at_two_points k ↔ -1 < k ∧ k < 1 :=
sorry

end slope_range_for_intersection_l2981_298152


namespace parallel_lines_m_values_l2981_298107

/-- Two lines are parallel if their slopes are equal or if they are both vertical -/
def are_parallel (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 = 0 ∧ a2 = 0) ∨ (b1 = 0 ∧ b2 = 0) ∨ (a1 * b2 = a2 * b1 ∧ a1 ≠ 0 ∧ a2 ≠ 0)

/-- The statement to be proved -/
theorem parallel_lines_m_values (m : ℝ) :
  are_parallel (m - 2) (-1) 5 (m - 2) (3 - m) 2 → m = 2 ∨ m = 4 := by
  sorry

end parallel_lines_m_values_l2981_298107


namespace quadrilateral_diagonal_count_l2981_298130

/-- A quadrilateral with side lengths 9, 11, 15, and 14 has exactly 17 possible whole number lengths for a diagonal. -/
theorem quadrilateral_diagonal_count : ∃ (possible_lengths : Finset ℕ),
  (∀ d ∈ possible_lengths, 
    -- Triangle inequality for both triangles formed by the diagonal
    9 + d > 11 ∧ d + 11 > 9 ∧ 9 + 11 > d ∧
    14 + d > 15 ∧ d + 15 > 14 ∧ 14 + 15 > d) ∧
  (∀ d : ℕ, 
    (9 + d > 11 ∧ d + 11 > 9 ∧ 9 + 11 > d ∧
     14 + d > 15 ∧ d + 15 > 14 ∧ 14 + 15 > d) → d ∈ possible_lengths) ∧
  Finset.card possible_lengths = 17 := by
sorry

end quadrilateral_diagonal_count_l2981_298130


namespace functional_equation_solution_l2981_298102

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c * x :=
by sorry

end functional_equation_solution_l2981_298102


namespace triangle_angle_sum_l2981_298111

-- Define the triangle
structure Triangle :=
  (a b c : ℝ)

-- Define the properties of the triangle
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b + t.c = 180

-- Define the specific conditions of our triangle
def our_triangle (t : Triangle) : Prop :=
  is_valid_triangle t ∧
  t.a = 70 ∧
  t.b = 40 ∧
  t.c = 70

-- Theorem statement
theorem triangle_angle_sum (t : Triangle) :
  our_triangle t → t.c = 40 :=
by sorry

end triangle_angle_sum_l2981_298111


namespace degree_of_minus_five_x_squared_y_l2981_298175

def monomial_degree (m : ℤ → ℤ → ℤ) : ℕ :=
  sorry

theorem degree_of_minus_five_x_squared_y :
  monomial_degree (fun x y ↦ -5 * x^2 * y) = 3 :=
sorry

end degree_of_minus_five_x_squared_y_l2981_298175


namespace net_effect_on_sale_value_l2981_298122

/-- Theorem: Net effect on sale value after price reduction and sales increase -/
theorem net_effect_on_sale_value 
  (price_reduction : Real) 
  (sales_increase : Real) 
  (h1 : price_reduction = 0.25)
  (h2 : sales_increase = 0.75) : 
  (1 - price_reduction) * (1 + sales_increase) - 1 = 0.3125 := by
  sorry

#eval (1 - 0.25) * (1 + 0.75) - 1

end net_effect_on_sale_value_l2981_298122


namespace N_satisfies_equation_l2981_298146

def N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 2; 1, 2]

theorem N_satisfies_equation : 
  N^3 - 3 • N^2 + 4 • N = !![6, 12; 3, 6] := by sorry

end N_satisfies_equation_l2981_298146


namespace reflection_across_x_axis_l2981_298140

-- Define the original function g(x)
def g (x : ℝ) : ℝ := x^2 - 4

-- Define the reflected function h(x)
def h (x : ℝ) : ℝ := -x^2 + 4

-- Theorem stating that h(x) is the reflection of g(x) across the x-axis
theorem reflection_across_x_axis :
  ∀ x : ℝ, h x = -(g x) :=
by
  sorry

end reflection_across_x_axis_l2981_298140


namespace triangle_ratio_l2981_298114

theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  b * Real.sin A * Real.sin B + a * (Real.cos B)^2 = 2 * c →
  a / c = 2 := by sorry

end triangle_ratio_l2981_298114


namespace average_playing_time_is_ten_l2981_298150

/-- Represents the number of players --/
def num_players : ℕ := 8

/-- Represents the start time in hours since midnight --/
def start_time : ℕ := 8

/-- Represents the end time in hours since midnight --/
def end_time : ℕ := 18

/-- Represents the number of chess games being played simultaneously --/
def num_games : ℕ := 2

/-- Calculates the average playing time per person --/
def average_playing_time : ℚ :=
  (end_time - start_time : ℚ) * num_games / num_players

theorem average_playing_time_is_ten :
  average_playing_time = 10 := by sorry

end average_playing_time_is_ten_l2981_298150


namespace division_problem_l2981_298164

theorem division_problem (x y : ℕ+) : 
  (x : ℝ) / y = 96.15 → 
  ∃ q : ℕ, x = q * y + 9 →
  y = 60 := by
sorry

end division_problem_l2981_298164


namespace selection_with_girl_count_l2981_298171

def num_boys : Nat := 4
def num_girls : Nat := 3
def num_selected : Nat := 3
def num_tasks : Nat := 3

theorem selection_with_girl_count :
  (Nat.choose (num_boys + num_girls) num_selected * Nat.factorial num_tasks) -
  (Nat.choose num_boys num_selected * Nat.factorial num_tasks) = 186 := by
  sorry

end selection_with_girl_count_l2981_298171


namespace opposite_of_negative_two_l2981_298131

theorem opposite_of_negative_two : 
  ∃ x : ℤ, x + (-2) = 0 ∧ x = 2 := by
  sorry

end opposite_of_negative_two_l2981_298131


namespace quadratic_roots_l2981_298165

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 5*x + m

theorem quadratic_roots (m : ℝ) (h : f m 1 = 0) : 
  ∃ (r₁ r₂ : ℝ), r₁ = 1 ∧ r₂ = 4 ∧ ∀ x, f m x = 0 ↔ x = r₁ ∨ x = r₂ := by
  sorry

end quadratic_roots_l2981_298165


namespace value_of_z_l2981_298195

theorem value_of_z (x y z : ℝ) (hx : x = 3) (hy : y = 2 * x) (hz : z = 3 * y) : z = 18 := by
  sorry

end value_of_z_l2981_298195


namespace triangle_inequality_l2981_298183

/-- Theorem: For any triangle with side lengths a, b, c, and area S,
    the inequality a² + b² + c² ≥ 4√3 S holds, with equality if and only if
    the triangle is equilateral. -/
theorem triangle_inequality (a b c S : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
    (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : S > 0)
    (h_area_def : S = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
    (h_s_def : s = (a + b + c) / 2) :
    a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S ∧
    (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * S ↔ a = b ∧ b = c) :=
  sorry

end triangle_inequality_l2981_298183


namespace area_bounded_by_function_and_double_tangent_l2981_298127

-- Define the function
def f (x : ℝ) : ℝ := -x^4 + 16*x^3 - 78*x^2 + 50*x - 2

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -4*x^3 + 48*x^2 - 156*x + 50

-- Theorem statement
theorem area_bounded_by_function_and_double_tangent :
  ∃ (a b : ℝ),
    a < b ∧
    f' a = f' b ∧
    (f b - f a) / (b - a) = f' a ∧
    (∫ (x : ℝ) in a..b, (((f b - f a) / (b - a)) * (x - a) + f a) - f x) = 1296 / 5 :=
sorry

end area_bounded_by_function_and_double_tangent_l2981_298127


namespace water_formed_is_zero_l2981_298179

-- Define the chemical compounds
inductive Compound
| NH4Cl
| NaOH
| BaNO3_2
| NH4OH
| NaCl
| HNO3
| NH4NO3
| H2O
| NaNO3
| BaCl2

-- Define a reaction
structure Reaction :=
(reactants : List (Compound × ℕ))
(products : List (Compound × ℕ))

-- Define the given reactions
def reaction1 : Reaction :=
{ reactants := [(Compound.NH4Cl, 1), (Compound.NaOH, 1)]
, products := [(Compound.NH4OH, 1), (Compound.NaCl, 1)] }

def reaction2 : Reaction :=
{ reactants := [(Compound.NH4OH, 1), (Compound.HNO3, 1)]
, products := [(Compound.NH4NO3, 1), (Compound.H2O, 1)] }

def reaction3 : Reaction :=
{ reactants := [(Compound.BaNO3_2, 1), (Compound.NaCl, 2)]
, products := [(Compound.NaNO3, 2), (Compound.BaCl2, 1)] }

-- Define the initial reactants
def initialReactants : List (Compound × ℕ) :=
[(Compound.NH4Cl, 3), (Compound.NaOH, 3), (Compound.BaNO3_2, 2)]

-- Define a function to calculate the moles of water formed
def molesOfWaterFormed (initialReactants : List (Compound × ℕ)) 
                       (reactions : List Reaction) : ℕ :=
  sorry

-- Theorem statement
theorem water_formed_is_zero :
  molesOfWaterFormed initialReactants [reaction1, reaction2, reaction3] = 0 :=
sorry

end water_formed_is_zero_l2981_298179


namespace arithmetic_sequence_ratio_l2981_298176

/-- Given arithmetic sequences a and b with sums S and T respectively, 
    if S_n / T_n = (2n-1) / (3n+2) for all n, then a_7 / b_7 = 25 / 41 -/
theorem arithmetic_sequence_ratio 
  (a b : ℕ → ℚ) 
  (S T : ℕ → ℚ) 
  (h1 : ∀ n, S n = (n / 2) * (a 1 + a n)) 
  (h2 : ∀ n, T n = (n / 2) * (b 1 + b n)) 
  (h3 : ∀ n, S n / T n = (2 * n - 1) / (3 * n + 2)) : 
  a 7 / b 7 = 25 / 41 := by
sorry

end arithmetic_sequence_ratio_l2981_298176


namespace truck_problem_l2981_298154

theorem truck_problem (T b c : ℝ) (hT : T > 0) (hb : b > 0) (hc : c > 0) :
  let x := (b * c + Real.sqrt (b^2 * c^2 + 4 * b * c * T)) / (2 * c)
  x * (x - b) * c = T * x ∧ (x - b) * (T / x + c) = T :=
by sorry

end truck_problem_l2981_298154


namespace line_properties_l2981_298144

-- Define the line equation
def line_equation (k x y : ℝ) : Prop := y + 1 = k * (x - 2)

-- Theorem statement
theorem line_properties :
  -- 1. Countless lines through (2, -1)
  (∃ (S : Set ℝ), Infinite S ∧ ∀ k ∈ S, line_equation k 2 (-1)) ∧
  -- 2. Always passes through a fixed point
  (∃ (x₀ y₀ : ℝ), ∀ k, line_equation k x₀ y₀) ∧
  -- 3. Cannot be perpendicular to x-axis
  (∀ k, line_equation k 0 0 → k ≠ 0) :=
sorry

end line_properties_l2981_298144


namespace scientific_notation_79000_l2981_298178

theorem scientific_notation_79000 : 79000 = 7.9 * (10 ^ 4) := by
  sorry

end scientific_notation_79000_l2981_298178


namespace binary_110011_equals_51_l2981_298110

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary_110011 : List Bool := [true, true, false, false, true, true]

theorem binary_110011_equals_51 : binary_to_decimal binary_110011 = 51 := by
  sorry

end binary_110011_equals_51_l2981_298110


namespace palmer_photo_ratio_l2981_298125

/-- Given the information about Palmer's photo collection before and after her trip to Bali,
    prove that the ratio of new pictures taken in the second week to the first week is 3:1. -/
theorem palmer_photo_ratio (initial_photos : ℕ) (final_photos : ℕ) (first_week : ℕ) (third_fourth_weeks : ℕ)
    (h1 : initial_photos = 100)
    (h2 : final_photos = 380)
    (h3 : first_week = 50)
    (h4 : third_fourth_weeks = 80) :
    (final_photos - initial_photos - first_week - third_fourth_weeks) / first_week = 3 := by
  sorry

#check palmer_photo_ratio

end palmer_photo_ratio_l2981_298125


namespace complex_number_coordinates_l2981_298105

theorem complex_number_coordinates : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end complex_number_coordinates_l2981_298105


namespace equidistant_function_property_l2981_298181

def f (a b : ℝ) (z : ℂ) : ℂ := (Complex.mk a b) * z

theorem equidistant_function_property (a b : ℝ) :
  (∀ z : ℂ, Complex.abs (f a b z - z) = Complex.abs (f a b z)) →
  Complex.abs (Complex.mk a b) = 5 →
  b^2 = 99/4 := by sorry

end equidistant_function_property_l2981_298181


namespace division_multiplication_result_l2981_298155

theorem division_multiplication_result : (9 / 6) * 12 = 18 := by
  sorry

end division_multiplication_result_l2981_298155


namespace problem_solution_l2981_298143

def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

theorem problem_solution :
  (∀ a : ℝ, (∀ x : ℝ, f a x + |x - 1| ≥ 2) → a ≤ 0 ∨ a ≥ 4) ∧
  (∀ a : ℝ, a < 2 → (∃ x : ℝ, ∀ y : ℝ, f a x ≤ f a y) → f a (a/2) = a - 1 → a = 4/3) :=
by sorry

end problem_solution_l2981_298143


namespace inequality_solution_set_a_range_for_inequality_l2981_298156

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x + 3|
def g (x : ℝ) : ℝ := |2*x - 1|

-- Statement for the first part of the problem
theorem inequality_solution_set :
  {x : ℝ | f x < g x} = {x : ℝ | x < -2/3 ∨ x > 4} :=
sorry

-- Statement for the second part of the problem
theorem a_range_for_inequality (a : ℝ) :
  (∀ x : ℝ, 2 * f x + g x > a * x + 4) ↔ -1 < a ∧ a ≤ 4 :=
sorry

end inequality_solution_set_a_range_for_inequality_l2981_298156


namespace cos_120_degrees_l2981_298167

theorem cos_120_degrees : Real.cos (2 * Real.pi / 3) = -1 / 2 := by
  sorry

end cos_120_degrees_l2981_298167


namespace f_composition_value_l2981_298109

noncomputable section

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp x else Real.log x

theorem f_composition_value : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end f_composition_value_l2981_298109


namespace arithmetic_sequence_property_l2981_298162

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  num_terms : ℕ
  common_diff : ℝ

/-- Theorem stating the properties of the specific arithmetic sequence -/
theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
  (h1 : seq.first_term = 3)
  (h2 : seq.last_term = 50)
  (h3 : seq.sum = 318) :
  seq.common_diff = 47 / 11 := by
  sorry

#check arithmetic_sequence_property

end arithmetic_sequence_property_l2981_298162


namespace multiples_of_6_not_18_under_350_l2981_298147

def count_multiples (n : ℕ) (m : ℕ) : ℕ :=
  (n - 1) / m

theorem multiples_of_6_not_18_under_350 : 
  (count_multiples 350 6) - (count_multiples 350 18) = 39 := by
  sorry

end multiples_of_6_not_18_under_350_l2981_298147


namespace consecutive_non_primes_l2981_298104

theorem consecutive_non_primes (n : ℕ) (h : n ≥ 1) :
  ∃ (k : ℕ), ∀ (i : ℕ), i ∈ Finset.range n → 
    ¬ Nat.Prime (k + i) ∧ 
    (∀ (j : ℕ), j ∈ Finset.range n → k + i = k + j → i = j) :=
by sorry

end consecutive_non_primes_l2981_298104


namespace volumes_not_equal_implies_cross_sections_not_equal_cross_sections_equal_not_implies_volumes_not_equal_l2981_298188

/-- Represents a geometric shape with height and volume -/
structure GeometricShape where
  height : ℝ
  volume : ℝ

/-- Represents the cross-sectional area of a shape at a given height -/
def crossSectionalArea (shape : GeometricShape) (h : ℝ) : ℝ :=
  sorry

/-- Cavalieri's Principle -/
axiom cavalieri_principle (A B : GeometricShape) :
  A.height = B.height →
  (∀ h, 0 ≤ h ∧ h ≤ A.height → crossSectionalArea A h = crossSectionalArea B h) →
  A.volume = B.volume

theorem volumes_not_equal_implies_cross_sections_not_equal
  (A B : GeometricShape) (h_height : A.height = B.height) :
  A.volume ≠ B.volume →
  ∃ h, 0 ≤ h ∧ h ≤ A.height ∧ crossSectionalArea A h ≠ crossSectionalArea B h :=
sorry

theorem cross_sections_equal_not_implies_volumes_not_equal
  (A B : GeometricShape) (h_height : A.height = B.height) :
  ¬(∀ h, 0 ≤ h ∧ h ≤ A.height → crossSectionalArea A h = crossSectionalArea B h →
    A.volume ≠ B.volume) :=
sorry

end volumes_not_equal_implies_cross_sections_not_equal_cross_sections_equal_not_implies_volumes_not_equal_l2981_298188


namespace inequality_not_hold_l2981_298159

theorem inequality_not_hold (m n a : Real) 
  (h1 : m > n) (h2 : n > 1) (h3 : 0 < a) (h4 : a < 1) : 
  ¬(a^m > a^n) := by
sorry

end inequality_not_hold_l2981_298159


namespace first_equation_is_golden_second_equation_root_l2981_298198

-- Definition of a golden equation
def is_golden_equation (a b c : ℝ) : Prop := a ≠ 0 ∧ a - b + c = 0

-- Theorem 1: 4x^2 + 11x + 7 = 0 is a golden equation
theorem first_equation_is_golden : is_golden_equation 4 11 7 := by sorry

-- Theorem 2: If 3x^2 - mx + n = 0 is a golden equation and m is a root, then m = -1 or m = 3/2
theorem second_equation_root (m n : ℝ) :
  is_golden_equation 3 (-m) n →
  (3 * m^2 - m * m + n = 0) →
  (m = -1 ∨ m = 3/2) := by sorry

end first_equation_is_golden_second_equation_root_l2981_298198


namespace four_team_win_structure_exists_l2981_298186

/-- Represents the result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss

/-- Represents a volleyball tournament -/
structure Tournament where
  teams : Finset Nat
  results : Nat → Nat → MatchResult
  round_robin : ∀ i j, i ≠ j → (results i j = MatchResult.Win ↔ results j i = MatchResult.Loss)

/-- The main theorem to be proved -/
theorem four_team_win_structure_exists (t : Tournament) 
  (h_eight_teams : t.teams.card = 8) :
  ∃ A B C D, A ∈ t.teams ∧ B ∈ t.teams ∧ C ∈ t.teams ∧ D ∈ t.teams ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    t.results A B = MatchResult.Win ∧
    t.results A C = MatchResult.Win ∧
    t.results A D = MatchResult.Win ∧
    t.results B C = MatchResult.Win ∧
    t.results B D = MatchResult.Win ∧
    t.results C D = MatchResult.Win :=
  sorry

end four_team_win_structure_exists_l2981_298186


namespace polynomial_evaluation_l2981_298113

theorem polynomial_evaluation :
  ∀ x : ℝ, x > 0 → x^2 - 3*x - 9 = 0 →
  x^4 - 3*x^3 - 9*x^2 + 27*x - 8 = 8 := by
  sorry

end polynomial_evaluation_l2981_298113


namespace larger_number_proof_l2981_298103

theorem larger_number_proof (A B : ℝ) (h1 : A > 0) (h2 : B > 0) 
  (h3 : A - B = 1660) (h4 : 0.075 * A = 0.125 * B) : A = 4150 := by
  sorry

end larger_number_proof_l2981_298103


namespace scientific_notation_180_million_l2981_298151

/-- Proves that 180 million in scientific notation is equal to 1.8 × 10^8 -/
theorem scientific_notation_180_million :
  (180000000 : ℝ) = 1.8 * (10 ^ 8) := by
  sorry

end scientific_notation_180_million_l2981_298151


namespace perpendicular_planes_l2981_298128

-- Define the types for line and plane
variable (L : Type) [LinearOrder L]
variable (P : Type)

-- Define the relations
variable (perpendicular : L → P → Prop)
variable (contains : P → L → Prop)
variable (perp_planes : P → P → Prop)

-- State the theorem
theorem perpendicular_planes 
  (l : L) (α β : P) 
  (h1 : perpendicular l α) 
  (h2 : contains β l) : 
  perp_planes α β :=
sorry

end perpendicular_planes_l2981_298128


namespace min_diagonal_rectangle_l2981_298199

/-- The minimum diagonal of a rectangle with perimeter 24 -/
theorem min_diagonal_rectangle (l w : ℝ) (h_perimeter : l + w = 12) :
  ∃ (d : ℝ), d = Real.sqrt (l^2 + w^2) ∧ 
  (∀ (l' w' : ℝ), l' + w' = 12 → Real.sqrt (l'^2 + w'^2) ≥ d) ∧
  d = 6 * Real.sqrt 2 := by
  sorry

end min_diagonal_rectangle_l2981_298199


namespace two_week_egg_consumption_l2981_298166

/-- Calculates the total number of eggs consumed over a given number of days,
    given a daily egg consumption rate. -/
def totalEggsConsumed (dailyConsumption : ℕ) (days : ℕ) : ℕ :=
  dailyConsumption * days

/-- Theorem stating that consuming 3 eggs daily for 14 days results in 42 eggs consumed. -/
theorem two_week_egg_consumption :
  totalEggsConsumed 3 14 = 42 := by
  sorry

end two_week_egg_consumption_l2981_298166


namespace one_negative_root_condition_l2981_298129

/-- A polynomial of the form x^4 + 3px^3 + 6x^2 + 3px + 1 -/
def polynomial (p : ℝ) (x : ℝ) : ℝ := x^4 + 3*p*x^3 + 6*x^2 + 3*p*x + 1

/-- The condition that the polynomial has exactly one negative real root -/
def has_one_negative_root (p : ℝ) : Prop :=
  ∃! x : ℝ, x < 0 ∧ polynomial p x = 0

/-- Theorem stating the condition on p for the polynomial to have exactly one negative real root -/
theorem one_negative_root_condition (p : ℝ) :
  has_one_negative_root p ↔ p ≥ 4/3 := by sorry

end one_negative_root_condition_l2981_298129


namespace pencil_boxes_count_l2981_298191

theorem pencil_boxes_count (book_boxes : ℕ) (books_per_box : ℕ) (pencils_per_box : ℕ) (total_items : ℕ) :
  book_boxes = 19 →
  books_per_box = 46 →
  pencils_per_box = 170 →
  total_items = 1894 →
  (total_items - book_boxes * books_per_box) / pencils_per_box = 6 :=
by
  sorry

#check pencil_boxes_count

end pencil_boxes_count_l2981_298191


namespace bailey_dog_treats_l2981_298160

theorem bailey_dog_treats :
  let total_items : ℕ := 4 * 5
  let chew_toys : ℕ := 2
  let rawhide_bones : ℕ := 10
  let dog_treats : ℕ := total_items - (chew_toys + rawhide_bones)
  dog_treats = 8 := by
sorry

end bailey_dog_treats_l2981_298160


namespace unique_prime_B_l2981_298100

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def number_form (B : ℕ) : ℕ := 1034960 + B

theorem unique_prime_B :
  ∃! B : ℕ, B < 10 ∧ is_prime (number_form B) :=
sorry

end unique_prime_B_l2981_298100


namespace max_rectangular_pen_area_l2981_298142

/-- Given 60 feet of fencing, the maximum area of a rectangular pen is 225 square feet. -/
theorem max_rectangular_pen_area (perimeter : ℝ) (h : perimeter = 60) : 
  ∃ (width height : ℝ), 
    width > 0 ∧ 
    height > 0 ∧ 
    2 * (width + height) = perimeter ∧ 
    ∀ (w h : ℝ), w > 0 → h > 0 → 2 * (w + h) = perimeter → w * h ≤ width * height ∧ 
    width * height = 225 :=
sorry

end max_rectangular_pen_area_l2981_298142


namespace perpendicular_distance_to_plane_l2981_298120

/-- The perpendicular distance from a point to a plane --/
def perpendicularDistance (p : ℝ × ℝ × ℝ) (plane : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

/-- The plane containing three points --/
def planeThroughPoints (a b c : ℝ × ℝ × ℝ) : Set (ℝ × ℝ × ℝ) :=
  sorry

theorem perpendicular_distance_to_plane :
  let a : ℝ × ℝ × ℝ := (0, 0, 0)
  let b : ℝ × ℝ × ℝ := (5, 0, 0)
  let c : ℝ × ℝ × ℝ := (0, 3, 0)
  let d : ℝ × ℝ × ℝ := (0, 0, 6)
  let plane := planeThroughPoints a b c
  perpendicularDistance d plane = 6 := by
  sorry

end perpendicular_distance_to_plane_l2981_298120


namespace quadratic_properties_l2981_298174

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties
  (a b c : ℝ)
  (ha : a < 0)
  (h_root : f a b c (-1) = 0)
  (h_sym : -b / (2 * a) = 1) :
  (a - b + c = 0) ∧
  (∀ m : ℝ, f a b c m ≤ -4 * a) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a b c x₁ = -1 → f a b c x₂ = -1 → x₁ < -1 ∧ x₂ > 3) :=
by sorry

end quadratic_properties_l2981_298174


namespace intersection_A_complement_B_l2981_298172

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x^2 - 2*x < 0}

def B : Set ℝ := {x | x ≥ 1}

theorem intersection_A_complement_B : A ∩ Bᶜ = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end intersection_A_complement_B_l2981_298172


namespace exists_fourth_power_product_l2981_298148

def is_not_divisible_by_primes_greater_than_28 (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p > 28 → ¬(p ∣ n)

theorem exists_fourth_power_product 
  (M : Finset ℕ) 
  (h_card : M.card = 2008) 
  (h_distinct : M.card = Finset.card (M.image id))
  (h_positive : ∀ n ∈ M, n > 0)
  (h_not_div : ∀ n ∈ M, is_not_divisible_by_primes_greater_than_28 n) :
  ∃ a b c d : ℕ, a ∈ M ∧ b ∈ M ∧ c ∈ M ∧ d ∈ M ∧ 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  ∃ k : ℕ, a * b * c * d = k^4 :=
sorry

end exists_fourth_power_product_l2981_298148


namespace complex_modulus_l2981_298193

theorem complex_modulus (z : ℂ) (h : (2 - Complex.I) * z = Complex.I) : Complex.abs z = Real.sqrt 5 / 5 := by
  sorry

end complex_modulus_l2981_298193


namespace arccos_one_half_l2981_298169

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by sorry

end arccos_one_half_l2981_298169


namespace quadratic_sum_equals_27_l2981_298112

theorem quadratic_sum_equals_27 (m n : ℝ) (h : m + n = 4) : 
  2 * m^2 + 4 * m * n + 2 * n^2 - 5 = 27 := by
sorry

end quadratic_sum_equals_27_l2981_298112


namespace three_digit_number_theorem_l2981_298141

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def satisfies_condition (n : ℕ) : Prop :=
  is_three_digit n ∧
  2 * n = (n % 100) * 10 + n / 100 + (n / 10 % 10) * 100 + n % 10

def solution_set : Finset ℕ :=
  {111, 222, 333, 370, 407, 444, 481, 518, 555, 592, 629, 666, 777, 888, 999}

theorem three_digit_number_theorem :
  ∀ n : ℕ, satisfies_condition n ↔ n ∈ solution_set := by sorry

end three_digit_number_theorem_l2981_298141


namespace fraction_product_simplification_l2981_298123

theorem fraction_product_simplification :
  (21 : ℚ) / 28 * 14 / 33 * 99 / 42 = 1 := by
  sorry

end fraction_product_simplification_l2981_298123


namespace percentage_passed_all_subjects_l2981_298177

/-- Percentage of students who failed in Hindi -/
def A : ℝ := 30

/-- Percentage of students who failed in English -/
def B : ℝ := 45

/-- Percentage of students who failed in Math -/
def C : ℝ := 25

/-- Percentage of students who failed in Science -/
def D : ℝ := 40

/-- Percentage of students who failed in both Hindi and English -/
def AB : ℝ := 12

/-- Percentage of students who failed in both Hindi and Math -/
def AC : ℝ := 15

/-- Percentage of students who failed in both Hindi and Science -/
def AD : ℝ := 18

/-- Percentage of students who failed in both English and Math -/
def BC : ℝ := 20

/-- Percentage of students who failed in both English and Science -/
def BD : ℝ := 22

/-- Percentage of students who failed in both Math and Science -/
def CD : ℝ := 24

/-- Percentage of students who failed in all four subjects -/
def ABCD : ℝ := 10

/-- The total percentage -/
def total : ℝ := 100

theorem percentage_passed_all_subjects :
  total - (A + B + C + D - (AB + AC + AD + BC + BD + CD) + ABCD) = 61 := by
  sorry

end percentage_passed_all_subjects_l2981_298177


namespace parallelogram_side_comparison_l2981_298182

structure Parallelogram where
  sides : Fin 4 → ℝ
  parallel : sides 0 = sides 2 ∧ sides 1 = sides 3

def inscribed (P Q : Parallelogram) : Prop :=
  ∀ i : Fin 4, P.sides i ≤ Q.sides i

theorem parallelogram_side_comparison 
  (P₁ P₂ P₃ : Parallelogram)
  (h₁ : inscribed P₂ P₁)
  (h₂ : inscribed P₃ P₂)
  (h₃ : ∀ i : Fin 4, P₃.sides i ≤ P₁.sides i) :
  ∃ i : Fin 4, P₁.sides i ≤ 2 * P₃.sides i :=
sorry

end parallelogram_side_comparison_l2981_298182


namespace solve_star_equation_l2981_298192

-- Define the star operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

-- Theorem statement
theorem solve_star_equation :
  ∀ y : ℝ, star 7 y = 47 → y = 5.4 := by
  sorry

end solve_star_equation_l2981_298192


namespace distance_to_town_l2981_298117

theorem distance_to_town (d : ℝ) : 
  (∀ x, x ≥ 6 → d < x) →  -- A's statement is false
  (∀ y, y ≤ 5 → d > y) →  -- B's statement is false
  (∀ z, z ≤ 4 → d > z) →  -- C's statement is false
  d ∈ Set.Ioo 5 6 := by
sorry

end distance_to_town_l2981_298117


namespace q_definition_l2981_298135

/-- Given p: x ≤ 1, and ¬p is a sufficient but not necessary condition for q,
    prove that q can be defined as x > 0 -/
theorem q_definition (x : ℝ) :
  (∃ p : Prop, (p ↔ x ≤ 1) ∧ 
   (∃ q : Prop, (¬p → q) ∧ ¬(q → ¬p))) →
  ∃ q : Prop, q ↔ x > 0 :=
by sorry

end q_definition_l2981_298135


namespace least_xy_value_l2981_298168

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) :
  (x * y : ℕ) ≥ 96 ∧ ∃ (a b : ℕ+), (a : ℚ) / b + (1 : ℚ) / (3 * b) = (1 : ℚ) / 8 ∧ (a * b : ℕ) = 96 :=
sorry

end least_xy_value_l2981_298168


namespace min_a_is_minimum_l2981_298163

/-- The inequality that holds for all x ≥ 0 -/
def inequality (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → x * Real.exp x + a * Real.exp x * Real.log (x + 1) + 1 ≥ Real.exp x * (x + 1) ^ a

/-- The minimum value of a that satisfies the inequality -/
def min_a : ℝ := -1

/-- Theorem stating that min_a is the minimum value satisfying the inequality -/
theorem min_a_is_minimum :
  (∀ a : ℝ, inequality a → a ≥ min_a) ∧ inequality min_a := by sorry

end min_a_is_minimum_l2981_298163


namespace isosceles_triangle_base_length_l2981_298149

/-- An isosceles triangle with perimeter 10 and one side length 2 -/
structure IsoscelesTriangle where
  perimeter : ℝ
  side_length : ℝ
  perimeter_eq : perimeter = 10
  side_length_eq : side_length = 2

/-- The base length of the isosceles triangle -/
def base_length (t : IsoscelesTriangle) : ℝ := 4

theorem isosceles_triangle_base_length (t : IsoscelesTriangle) :
  base_length t = 4 :=
by sorry

end isosceles_triangle_base_length_l2981_298149
