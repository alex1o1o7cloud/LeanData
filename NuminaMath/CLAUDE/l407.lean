import Mathlib

namespace line_tangent_to_circle_l407_40738

theorem line_tangent_to_circle 
  (x₀ y₀ r : ℝ) 
  (h_outside : x₀^2 + y₀^2 > r^2) :
  ∃ (x y : ℝ), 
    x₀*x + y₀*y = r^2 ∧ 
    x^2 + y^2 = r^2 ∧
    ∀ (x' y' : ℝ), x₀*x' + y₀*y' = r^2 ∧ x'^2 + y'^2 = r^2 → (x', y') = (x, y) :=
by sorry

end line_tangent_to_circle_l407_40738


namespace increase_and_subtract_l407_40774

theorem increase_and_subtract (initial : ℝ) (increase_percent : ℝ) (subtract_amount : ℝ) : 
  initial = 837 → 
  increase_percent = 135 → 
  subtract_amount = 250 → 
  (initial * (1 + increase_percent / 100) - subtract_amount) = 1717.95 := by
  sorry

end increase_and_subtract_l407_40774


namespace equation_solution_l407_40739

theorem equation_solution : ∃ x : ℝ, x ≠ 2 ∧ x + 2 = 2 / (x - 2) ↔ x = Real.sqrt 6 ∨ x = -Real.sqrt 6 := by
  sorry

end equation_solution_l407_40739


namespace larger_bill_value_l407_40725

/-- Proves that the value of the larger denomination bill is $10 given the problem conditions --/
theorem larger_bill_value (total_bills : ℕ) (total_value : ℕ) (five_dollar_bills : ℕ) (larger_bills : ℕ) :
  total_bills = 5 + larger_bills →
  total_bills = 12 →
  five_dollar_bills = 4 →
  larger_bills = 8 →
  total_value = 100 →
  total_value = 5 * five_dollar_bills + larger_bills * 10 :=
by sorry

end larger_bill_value_l407_40725


namespace units_digit_of_n_l407_40764

def units_digit (a : ℕ) : ℕ := a % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : units_digit m = 6) :
  units_digit n = 1 := by
  sorry

end units_digit_of_n_l407_40764


namespace total_fruits_three_days_l407_40715

/-- Represents the number of fruits eaten by a dog on a given day -/
def fruits_eaten (initial : ℕ) (day : ℕ) : ℕ :=
  initial * 2^(day - 1)

/-- Represents the total fruits eaten by all dogs over a period of days -/
def total_fruits (bonnies_day1 : ℕ) (days : ℕ) : ℕ :=
  let blueberries_day1 := (3 * bonnies_day1) / 4
  let apples_day1 := 3 * blueberries_day1
  let cherries_day1 := 5 * apples_day1
  (Finset.sum (Finset.range days) (λ d => fruits_eaten bonnies_day1 (d + 1))) +
  (Finset.sum (Finset.range days) (λ d => fruits_eaten blueberries_day1 (d + 1))) +
  (Finset.sum (Finset.range days) (λ d => fruits_eaten apples_day1 (d + 1))) +
  (Finset.sum (Finset.range days) (λ d => fruits_eaten cherries_day1 (d + 1)))

theorem total_fruits_three_days :
  total_fruits 60 3 = 6405 := by
  sorry

end total_fruits_three_days_l407_40715


namespace problem_statement_l407_40763

open Real

theorem problem_statement :
  (¬ (∀ x : ℝ, sin x ≠ 1) ↔ (∃ x : ℝ, sin x = 1)) ∧
  ((∀ α : ℝ, α = π/6 → sin α = 1/2) ∧ ¬(∀ α : ℝ, sin α = 1/2 → α = π/6)) ∧
  ¬(∀ a : ℕ → ℝ, (∀ n : ℕ, a (n+1) = 3 * a n) ↔ (∃ r : ℝ, ∀ n : ℕ, a (n+1) = r * a n)) :=
by
  sorry


end problem_statement_l407_40763


namespace existence_of_squares_with_difference_2023_l407_40796

theorem existence_of_squares_with_difference_2023 :
  ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^2 = y^2 + 2023 ∧
  ((x = 1012 ∧ y = 1011) ∨ (x = 148 ∧ y = 141) ∨ (x = 68 ∧ y = 51)) :=
by sorry

end existence_of_squares_with_difference_2023_l407_40796


namespace ellipse_chord_y_diff_l407_40761

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Theorem: For the given ellipse, if a chord AB passes through the left focus and the
    inscribed circle of triangle ABF₂ has circumference π, then |y₁ - y₂| = 5/4 -/
theorem ellipse_chord_y_diff (e : Ellipse) (A B : Point) (F₁ F₂ : Point) : 
  e.a = 5 → 
  e.b = 4 → 
  F₁.x = -3 → 
  F₁.y = 0 → 
  F₂.x = 3 → 
  F₂.y = 0 → 
  (A.x - F₁.x) * (B.y - F₁.y) = (B.x - F₁.x) * (A.y - F₁.y) →  -- AB passes through F₁
  2 * π * (A.x * (B.y - F₂.y) + B.x * (F₂.y - A.y) + F₂.x * (A.y - B.y)) / 
    (A.x * (B.y - F₂.y) + B.x * (F₂.y - A.y) + F₂.x * (A.y - B.y) + 
     (A.x - F₂.x) * (B.y - F₂.y) - (B.x - F₂.x) * (A.y - F₂.y)) = π →  -- Inscribed circle circumference
  |A.y - B.y| = 5/4 := by
sorry


end ellipse_chord_y_diff_l407_40761


namespace parabola_line_intersection_l407_40767

/-- The parabola equation -/
def parabola (a x y : ℝ) : Prop := y = a * x^2 + 5 * x + 2

/-- The line equation -/
def line (x y : ℝ) : Prop := y = -2 * x + 1

/-- The intersection condition -/
def intersect_once (a : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, parabola a p.1 p.2 ∧ line p.1 p.2

/-- The theorem statement -/
theorem parabola_line_intersection (a : ℝ) :
  intersect_once a ↔ a = 49 / 4 := by sorry

end parabola_line_intersection_l407_40767


namespace quadratic_roots_condition_l407_40760

-- Define the quadratic equation
def quadratic_equation (x a : ℝ) : Prop := x^2 + 3*x - a = 0

-- Define the condition for two distinct real roots
def has_two_distinct_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ quadratic_equation x a ∧ quadratic_equation y a

-- Theorem statement
theorem quadratic_roots_condition (a : ℝ) :
  has_two_distinct_real_roots a ↔ a > -9/4 :=
by sorry

end quadratic_roots_condition_l407_40760


namespace xyz_value_l407_40728

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 8) :
  x * y * z = 16 / 3 := by
sorry

end xyz_value_l407_40728


namespace sphere_surface_area_l407_40705

theorem sphere_surface_area (r : ℝ) (d : ℝ) (h1 : r = 1) (h2 : d = Real.sqrt 3) :
  4 * Real.pi * (r^2 + d^2) = 16 * Real.pi := by
  sorry

end sphere_surface_area_l407_40705


namespace complex_sum_equals_one_l407_40742

theorem complex_sum_equals_one (w : ℂ) (h : w = Complex.exp (Complex.I * (6 * Real.pi / 11))) :
  w / (1 + w^2) + w^3 / (1 + w^6) + w^4 / (1 + w^8) = 1 := by
  sorry

end complex_sum_equals_one_l407_40742


namespace solution_set_f_leq_x_plus_2_range_f_geq_expr_l407_40704

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x + 1|

-- Theorem for the solution set of f(x) ≤ x + 2
theorem solution_set_f_leq_x_plus_2 :
  {x : ℝ | f x ≤ x + 2} = {x : ℝ | 0 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem for the range of x satisfying f(x) ≥ (|a+1| - |2a-1|)/|a| for all non-zero real a
theorem range_f_geq_expr :
  {x : ℝ | ∀ a : ℝ, a ≠ 0 → f x ≥ (|a + 1| - |2*a - 1|) / |a|} =
  {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

end solution_set_f_leq_x_plus_2_range_f_geq_expr_l407_40704


namespace trigonometric_expression_evaluation_l407_40780

theorem trigonometric_expression_evaluation :
  (Real.tan (150 * π / 180)) * (Real.cos (-210 * π / 180)) * (Real.sin (-420 * π / 180)) /
  ((Real.sin (1050 * π / 180)) * (Real.cos (-600 * π / 180))) = -Real.sqrt 3 := by
  sorry

end trigonometric_expression_evaluation_l407_40780


namespace simple_interest_problem_l407_40732

/-- Simple interest calculation problem -/
theorem simple_interest_problem (rate : ℚ) (principal : ℚ) (interest_diff : ℚ) (years : ℚ) :
  rate = 4 / 100 →
  principal = 2400 →
  principal * rate * years = principal - interest_diff →
  interest_diff = 1920 →
  years = 5 := by
  sorry

end simple_interest_problem_l407_40732


namespace hyperbola_equation_l407_40722

/-- Given a hyperbola with one focus at (5,0) and asymptotes y = ± 4/3 x, 
    its equation is x²/9 - y²/16 = 1 -/
theorem hyperbola_equation (F : ℝ × ℝ) (slope : ℝ) :
  F = (5, 0) →
  slope = 4/3 →
  ∀ (x y : ℝ), (x^2 / 9 - y^2 / 16 = 1) ↔ 
    (∃ (a b c : ℝ), 
      a^2 + b^2 = c^2 ∧
      c = 5 ∧
      b / a = slope ∧
      x^2 / a^2 - y^2 / b^2 = 1) :=
by sorry

end hyperbola_equation_l407_40722


namespace rectangle_height_l407_40781

/-- Given a rectangle with width 32 cm and area divided by diagonal 576 cm², prove its height is 36 cm. -/
theorem rectangle_height (w h : ℝ) (area_div_diagonal : ℝ) : 
  w = 32 → 
  area_div_diagonal = 576 →
  (w * h) / 2 = area_div_diagonal →
  h = 36 := by
  sorry

end rectangle_height_l407_40781


namespace xyz_sum_l407_40792

theorem xyz_sum (x y z : ℕ+) (h : (x + y * Complex.I)^2 - 46 * Complex.I = z) :
  x + y + z = 552 := by
  sorry

end xyz_sum_l407_40792


namespace intersection_sum_mod20_l407_40711

/-- The sum of x-coordinates of intersection points of two modular functions -/
theorem intersection_sum_mod20 : ∃ (x₁ x₂ : ℕ),
  (x₁ < 20 ∧ x₂ < 20) ∧
  (∀ (y : ℕ), (7 * x₁ + 3) % 20 = y % 20 ↔ (13 * x₁ + 17) % 20 = y % 20) ∧
  (∀ (y : ℕ), (7 * x₂ + 3) % 20 = y % 20 ↔ (13 * x₂ + 17) % 20 = y % 20) ∧
  x₁ + x₂ = 12 :=
by sorry

end intersection_sum_mod20_l407_40711


namespace average_weight_proof_l407_40743

theorem average_weight_proof (a b c : ℝ) : 
  (a + b + c) / 3 = 45 →
  (b + c) / 2 = 43 →
  b = 35 →
  (a + b) / 2 = 42 := by
sorry

end average_weight_proof_l407_40743


namespace composite_29n_plus_11_l407_40749

theorem composite_29n_plus_11 (n : ℕ) 
  (h1 : ∃ a : ℕ, 3 * n + 1 = a ^ 2) 
  (h2 : ∃ b : ℕ, 10 * n + 1 = b ^ 2) : 
  ¬(Nat.Prime (29 * n + 11)) :=
sorry

end composite_29n_plus_11_l407_40749


namespace students_remaining_in_school_l407_40716

theorem students_remaining_in_school (total_students : ℕ) 
  (h1 : total_students = 1000)
  (h2 : ∃ trip_students : ℕ, trip_students = total_students / 2)
  (h3 : ∃ remaining_after_trip : ℕ, remaining_after_trip = total_students - (total_students / 2))
  (h4 : ∃ sent_home : ℕ, sent_home = remaining_after_trip / 2)
  : total_students - (total_students / 2) - ((total_students - (total_students / 2)) / 2) = 250 := by
  sorry

end students_remaining_in_school_l407_40716


namespace game_wheel_probability_l407_40758

theorem game_wheel_probability : 
  ∀ (p_A p_B p_C p_D p_E : ℚ),
    p_A = 2/7 →
    p_B = 1/7 →
    p_C = p_D →
    p_C = p_E →
    p_A + p_B + p_C + p_D + p_E = 1 →
    p_C = 4/21 := by
  sorry

end game_wheel_probability_l407_40758


namespace faye_pencil_rows_l407_40750

/-- The number of rows that can be made with a given number of pencils and pencils per row. -/
def number_of_rows (total_pencils : ℕ) (pencils_per_row : ℕ) : ℕ :=
  total_pencils / pencils_per_row

/-- Theorem: Faye can make 6 rows with 30 pencils, placing 5 pencils in each row. -/
theorem faye_pencil_rows : number_of_rows 30 5 = 6 := by
  sorry

end faye_pencil_rows_l407_40750


namespace arithmetic_expression_equality_l407_40727

theorem arithmetic_expression_equality : (4 + 6 * 3) - (2 * 3) + 5 = 21 := by
  sorry

end arithmetic_expression_equality_l407_40727


namespace central_cell_value_l407_40753

theorem central_cell_value (n : ℕ) (h1 : n = 29) :
  let total_sum := n * (n * (n + 1) / 2)
  let above_diagonal_sum := 3 * ((total_sum - n * (n + 1) / 2) / 2)
  let below_diagonal_sum := (total_sum - n * (n + 1) / 2) / 2
  let diagonal_sum := total_sum - above_diagonal_sum - below_diagonal_sum
  above_diagonal_sum = 3 * below_diagonal_sum →
  diagonal_sum / n = 15 := by
  sorry

#check central_cell_value

end central_cell_value_l407_40753


namespace ratio_unchanged_l407_40709

/-- Represents the number of animals in the zoo -/
structure ZooPopulation where
  cheetahs : ℕ
  pandas : ℕ

/-- The zoo population 5 years ago -/
def initial_population : ZooPopulation := sorry

/-- The current zoo population -/
def current_population : ZooPopulation :=
  { cheetahs := initial_population.cheetahs + 2,
    pandas := initial_population.pandas + 6 }

/-- The ratio of cheetahs to pandas -/
def cheetah_panda_ratio (pop : ZooPopulation) : ℚ :=
  pop.cheetahs / pop.pandas

theorem ratio_unchanged :
  cheetah_panda_ratio initial_population = cheetah_panda_ratio current_population :=
by sorry

end ratio_unchanged_l407_40709


namespace sqrt_equation_solution_l407_40773

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 + Real.sqrt x) = 4 → x = 121 := by
  sorry

end sqrt_equation_solution_l407_40773


namespace division_remainder_l407_40744

theorem division_remainder : ∃ q : ℕ, 1234567 = 257 * q + 123 ∧ 123 < 257 := by
  sorry

end division_remainder_l407_40744


namespace problem_grid_triangles_l407_40754

/-- Represents a triangular grid with a given number of rows -/
structure TriangularGrid where
  rows : ℕ

/-- Calculates the total number of triangles in a triangular grid -/
def totalTriangles (grid : TriangularGrid) : ℕ :=
  sorry

/-- The specific triangular grid described in the problem -/
def problemGrid : TriangularGrid :=
  { rows := 4 }

theorem problem_grid_triangles :
  totalTriangles problemGrid = 18 := by
  sorry

end problem_grid_triangles_l407_40754


namespace furniture_payment_l407_40786

theorem furniture_payment (a b c d e : ℝ) : 
  a + b + c + d + e = 120 ∧
  a = (1/3) * (b + c + d + e) ∧
  b = (1/4) * (a + c + d + e) ∧
  c = (1/5) * (a + b + d + e) ∧
  d = (1/6) * (a + b + c + e) →
  e = 41.33 := by sorry

end furniture_payment_l407_40786


namespace collinear_points_sum_l407_40707

/-- Three points in 3D space are collinear if they lie on the same line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop := sorry

/-- The main theorem: If the given points are collinear, then c + d = 6. -/
theorem collinear_points_sum (c d : ℝ) : 
  collinear (2, c, d) (c, 3, d) (c, d, 4) → c + d = 6 := by
  sorry

end collinear_points_sum_l407_40707


namespace hyperbola_condition_l407_40731

/-- A curve of the form ax^2 + by^2 = 1 is a hyperbola if ab < 0 -/
def is_hyperbola (a b : ℝ) : Prop := a * b < 0

/-- The curve mx^2 - (m-2)y^2 = 1 -/
def curve (m : ℝ) : (ℝ → ℝ → Prop) := λ x y => m * x^2 - (m - 2) * y^2 = 1

theorem hyperbola_condition (m : ℝ) :
  (∀ m > 3, is_hyperbola m (2 - m)) ∧
  (∃ m ≤ 3, is_hyperbola m (2 - m)) :=
by sorry

end hyperbola_condition_l407_40731


namespace line_segment_param_sum_squares_l407_40720

/-- Given a line segment from (1,2) to (6,9) parameterized by x = pt + q and y = rt + s,
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1,2), prove that p^2 + q^2 + r^2 + s^2 = 79 -/
theorem line_segment_param_sum_squares :
  ∀ (p q r s : ℝ),
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → 
    p * t + q = 1 + 5 * t ∧ 
    r * t + s = 2 + 7 * t) →
  p^2 + q^2 + r^2 + s^2 = 79 := by
sorry

end line_segment_param_sum_squares_l407_40720


namespace parabola_line_intersection_l407_40759

/-- Parabola defined by y² = 2x -/
def parabola (x y : ℝ) : Prop := y^2 = 2*x

/-- Line defined by y = -1/2x + b -/
def line (x y b : ℝ) : Prop := y = -1/2*x + b

/-- Point on both parabola and line -/
def intersection_point (x y b : ℝ) : Prop :=
  parabola x y ∧ line x y b

/-- Circle with diameter AB is tangent to x-axis -/
def circle_tangent_to_x_axis (xA yA xB yB : ℝ) : Prop :=
  (yA + yB) / 2 = (xB - xA) / 4

theorem parabola_line_intersection (b : ℝ) :
  (∃ xA yA xB yB : ℝ,
    intersection_point xA yA b ∧
    intersection_point xB yB b ∧
    xA ≠ xB ∧
    circle_tangent_to_x_axis xA yA xB yB) →
  b = -4/5 := by sorry

end parabola_line_intersection_l407_40759


namespace constant_term_expansion_l407_40799

theorem constant_term_expansion (x : ℝ) (x_ne_zero : x ≠ 0) : 
  ∃ (c : ℕ), c = 17920 ∧ 
  ∃ (f : ℝ → ℝ), (λ x => (2*x + 2/x)^8) = (λ x => c + f x) ∧ 
  (∀ x ≠ 0, f x ≠ 0 → ∃ (n : ℤ), n ≠ 0 ∧ f x = x^n * (f x / x^n)) :=
by sorry

end constant_term_expansion_l407_40799


namespace table_tennis_matches_l407_40719

theorem table_tennis_matches (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 ∧ (n * (n - 1)) / 2 ≠ 10 := by
  sorry

#check table_tennis_matches

end table_tennis_matches_l407_40719


namespace product_of_roots_l407_40710

theorem product_of_roots : Real.sqrt 4 ^ (1/3) * Real.sqrt 8 ^ (1/4) = 2 * Real.sqrt 32 ^ (1/12) := by
  sorry

end product_of_roots_l407_40710


namespace largest_five_digit_congruent_to_18_mod_25_l407_40795

theorem largest_five_digit_congruent_to_18_mod_25 : ∃ n : ℕ,
  n = 99993 ∧
  n ≥ 10000 ∧ n < 100000 ∧
  n % 25 = 18 ∧
  ∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 25 = 18 → m ≤ n :=
by sorry

end largest_five_digit_congruent_to_18_mod_25_l407_40795


namespace binomial_26_6_l407_40718

theorem binomial_26_6 (h1 : Nat.choose 24 5 = 42504) (h2 : Nat.choose 24 6 = 134596) :
  Nat.choose 26 6 = 230230 := by
  sorry

end binomial_26_6_l407_40718


namespace line_inclination_theorem_l407_40735

theorem line_inclination_theorem (a b c : ℝ) (α : ℝ) : 
  (∃ (x y : ℝ), a * x + b * y + c = 0) →  -- Line exists
  (Real.tan α = -a / b) →  -- Angle of inclination
  (Real.sin α + Real.cos α = 0) →  -- Given condition
  (a - b = 0) :=  -- Conclusion to prove
by sorry

end line_inclination_theorem_l407_40735


namespace f_properties_l407_40733

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x) + 2 * Real.sqrt 3 * Real.sin x * Real.cos x

theorem f_properties :
  (∃ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 ∧ f x = 2 ∧ ∀ y, 0 ≤ y ∧ y ≤ Real.pi / 2 → f y ≤ f x) ∧
  (∀ θ : ℝ, 0 < θ ∧ θ < Real.pi / 6 ∧ f θ = 4 / 3 → Real.cos (2 * θ) = (Real.sqrt 15 + 2) / 6) :=
by sorry

end f_properties_l407_40733


namespace fraction_sum_equality_l407_40777

theorem fraction_sum_equality (a b c : ℝ) 
  (h : a / (20 - a) + b / (75 - b) + c / (55 - c) = 8) :
  4 / (20 - a) + 15 / (75 - b) + 11 / (55 - c) = 8.8 := by
  sorry

end fraction_sum_equality_l407_40777


namespace school_supplies_expenditure_l407_40769

theorem school_supplies_expenditure (winnings : ℚ) : 
  (winnings / 2 : ℚ) + -- Amount spent on supplies
  ((winnings - winnings / 2) * 3 / 8 : ℚ) + -- Amount saved
  (2500 : ℚ) -- Remaining amount
  = winnings →
  (winnings / 2 : ℚ) = 4000 := by sorry

end school_supplies_expenditure_l407_40769


namespace point_on_transformed_graph_l407_40726

/-- Given a function f : ℝ → ℝ such that f(3) = -2,
    prove that (1, 0) satisfies the equation 3y = 2f(3x) + 4 -/
theorem point_on_transformed_graph (f : ℝ → ℝ) (h : f 3 = -2) :
  let g : ℝ → ℝ := λ x => (2 * f (3 * x) + 4) / 3
  g 1 = 0 := by sorry

end point_on_transformed_graph_l407_40726


namespace sum_of_composition_equals_negative_ten_l407_40745

def p (x : ℝ) : ℝ := abs x - 2

def q (x : ℝ) : ℝ := -abs x

def evaluation_points : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_composition_equals_negative_ten :
  (evaluation_points.map (λ x => q (p x))).sum = -10 := by sorry

end sum_of_composition_equals_negative_ten_l407_40745


namespace english_to_maths_ratio_l407_40752

/-- Represents the marks obtained in different subjects -/
structure Marks where
  english : ℕ
  science : ℕ
  maths : ℕ

/-- Represents the ratio between two numbers -/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Theorem stating the ratio of English to Maths marks -/
theorem english_to_maths_ratio (m : Marks) : 
  m.science = 17 ∧ 
  m.english = 3 * m.science ∧ 
  m.english + m.science + m.maths = 170 → 
  ∃ r : Ratio, r.numerator = 1 ∧ r.denominator = 2 ∧ 
    r.numerator * m.maths = r.denominator * m.english :=
by sorry

end english_to_maths_ratio_l407_40752


namespace unique_number_exists_l407_40713

-- Define the properties of x
def is_reciprocal_not_less_than_1 (x : ℝ) : Prop := 1 / x ≥ 1
def does_not_contain_6 (x : ℕ) : Prop := ¬ (∃ d : ℕ, d < 10 ∧ d = 6 ∧ ∃ k : ℕ, x = 10 * k + d)
def cube_less_than_221 (x : ℝ) : Prop := x^3 < 221
def is_even (x : ℕ) : Prop := ∃ k : ℕ, x = 2 * k
def is_prime (x : ℕ) : Prop := Nat.Prime x
def is_multiple_of_5 (x : ℕ) : Prop := ∃ k : ℕ, x = 5 * k
def is_irrational (x : ℝ) : Prop := ¬ (∃ p q : ℤ, q ≠ 0 ∧ x = p / q)
def is_less_than_6 (x : ℝ) : Prop := x < 6
def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, x = k^2
def is_greater_than_20 (x : ℝ) : Prop := x > 20
def log_base_10_at_least_2 (x : ℝ) : Prop := Real.log x / Real.log 10 ≥ 2
def is_not_less_than_10 (x : ℝ) : Prop := x ≥ 10

-- Define the theorem
theorem unique_number_exists : ∃! x : ℕ, 
  (is_reciprocal_not_less_than_1 x ∨ does_not_contain_6 x ∨ cube_less_than_221 x) ∧
  (¬is_reciprocal_not_less_than_1 x ∨ ¬does_not_contain_6 x ∨ ¬cube_less_than_221 x) ∧
  (is_even x ∨ is_prime x ∨ is_multiple_of_5 x) ∧
  (¬is_even x ∨ ¬is_prime x ∨ ¬is_multiple_of_5 x) ∧
  (is_irrational x ∨ is_less_than_6 x ∨ is_perfect_square x) ∧
  (¬is_irrational x ∨ ¬is_less_than_6 x ∨ ¬is_perfect_square x) ∧
  (is_greater_than_20 x ∨ log_base_10_at_least_2 x ∨ is_not_less_than_10 x) ∧
  (¬is_greater_than_20 x ∨ ¬log_base_10_at_least_2 x ∨ ¬is_not_less_than_10 x) :=
by sorry


end unique_number_exists_l407_40713


namespace g_f_three_equals_one_l407_40702

-- Define the domain of x
inductive Domain : Type
| one : Domain
| two : Domain
| three : Domain
| four : Domain

-- Define function f
def f : Domain → Domain
| Domain.one => Domain.three
| Domain.two => Domain.four
| Domain.three => Domain.two
| Domain.four => Domain.one

-- Define function g
def g : Domain → ℕ
| Domain.one => 2
| Domain.two => 1
| Domain.three => 6
| Domain.four => 8

-- Theorem to prove
theorem g_f_three_equals_one : g (f Domain.three) = 1 := by
  sorry

end g_f_three_equals_one_l407_40702


namespace line_passes_through_point_three_common_tangents_implies_a_8_l407_40746

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define the line l
def line_l (m x y : ℝ) : Prop := (m + 1) * x + 2 * y - 1 + m = 0

-- Define the second circle
def circle_2 (x y a : ℝ) : Prop := x^2 + y^2 - 2*x + 8*y + a = 0

-- Theorem 1: Line l always passes through the fixed point (-1, 1)
theorem line_passes_through_point :
  ∀ m : ℝ, line_l m (-1) 1 :=
sorry

-- Theorem 2: If circle C and circle_2 have exactly three common tangents, then a = 8
theorem three_common_tangents_implies_a_8 :
  (∃! (t1 t2 t3 : ℝ × ℝ), 
    (∀ x y, circle_C x y → (x - t1.1)^2 + (y - t1.2)^2 = 0 ∨ 
                           (x - t2.1)^2 + (y - t2.2)^2 = 0 ∨ 
                           (x - t3.1)^2 + (y - t3.2)^2 = 0) ∧
    (∀ x y, circle_2 x y a → (x - t1.1)^2 + (y - t1.2)^2 = 0 ∨ 
                              (x - t2.1)^2 + (y - t2.2)^2 = 0 ∨ 
                              (x - t3.1)^2 + (y - t3.2)^2 = 0)) →
  a = 8 :=
sorry

end line_passes_through_point_three_common_tangents_implies_a_8_l407_40746


namespace smallest_n_divisible_l407_40708

theorem smallest_n_divisible (n : ℕ) : 
  (∀ m : ℕ, m > 0 ∧ m < 12 → (¬(72 ∣ m^2) ∨ ¬(1728 ∣ m^3))) ∧ 
  (72 ∣ 12^2) ∧ (1728 ∣ 12^3) := by
  sorry

end smallest_n_divisible_l407_40708


namespace smallest_factorial_divisible_by_2016_smallest_factorial_divisible_by_2016_power_10_l407_40793

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem smallest_factorial_divisible_by_2016 :
  ∀ n : ℕ, n < 8 → ¬(2016 ∣ factorial n) ∧ (2016 ∣ factorial 8) :=
sorry

theorem smallest_factorial_divisible_by_2016_power_10 :
  ∀ n : ℕ, n < 63 → ¬(2016^10 ∣ factorial n) ∧ (2016^10 ∣ factorial 63) :=
sorry

end smallest_factorial_divisible_by_2016_smallest_factorial_divisible_by_2016_power_10_l407_40793


namespace divisible_by_six_ones_digits_l407_40712

theorem divisible_by_six_ones_digits : 
  ∃ (S : Finset ℕ), (∀ n ∈ S, n < 10 ∧ ∃ m : ℕ, 6 ∣ (10 * m + n)) ∧ S.card = 5 :=
sorry

end divisible_by_six_ones_digits_l407_40712


namespace polygon_is_trapezoid_l407_40730

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of a trapezoid: a quadrilateral with at least one pair of parallel sides -/
def is_trapezoid (p1 p2 p3 p4 : Point) : Prop :=
  ∃ (l1 l2 l3 l4 : Line),
    (p1.y = l1.slope * p1.x + l1.intercept) ∧
    (p2.y = l2.slope * p2.x + l2.intercept) ∧
    (p3.y = l3.slope * p3.x + l3.intercept) ∧
    (p4.y = l4.slope * p4.x + l4.intercept) ∧
    ((l1.slope = l2.slope ∧ l1.slope ≠ l3.slope ∧ l1.slope ≠ l4.slope) ∨
     (l1.slope = l3.slope ∧ l1.slope ≠ l2.slope ∧ l1.slope ≠ l4.slope) ∨
     (l1.slope = l4.slope ∧ l1.slope ≠ l2.slope ∧ l1.slope ≠ l3.slope) ∨
     (l2.slope = l3.slope ∧ l2.slope ≠ l1.slope ∧ l2.slope ≠ l4.slope) ∨
     (l2.slope = l4.slope ∧ l2.slope ≠ l1.slope ∧ l2.slope ≠ l3.slope) ∨
     (l3.slope = l4.slope ∧ l3.slope ≠ l1.slope ∧ l3.slope ≠ l2.slope))

theorem polygon_is_trapezoid :
  let l1 : Line := ⟨2, 3⟩
  let l2 : Line := ⟨-2, 3⟩
  let l3 : Line := ⟨2, -1⟩
  let l4 : Line := ⟨0, -1⟩
  ∃ (p1 p2 p3 p4 : Point),
    (p1.y = l1.slope * p1.x + l1.intercept ∨ p1.y = l2.slope * p1.x + l2.intercept ∨
     p1.y = l3.slope * p1.x + l3.intercept ∨ p1.y = l4.slope * p1.x + l4.intercept) ∧
    (p2.y = l1.slope * p2.x + l1.intercept ∨ p2.y = l2.slope * p2.x + l2.intercept ∨
     p2.y = l3.slope * p2.x + l3.intercept ∨ p2.y = l4.slope * p2.x + l4.intercept) ∧
    (p3.y = l1.slope * p3.x + l1.intercept ∨ p3.y = l2.slope * p3.x + l2.intercept ∨
     p3.y = l3.slope * p3.x + l3.intercept ∨ p3.y = l4.slope * p3.x + l4.intercept) ∧
    (p4.y = l1.slope * p4.x + l1.intercept ∨ p4.y = l2.slope * p4.x + l2.intercept ∨
     p4.y = l3.slope * p4.x + l3.intercept ∨ p4.y = l4.slope * p4.x + l4.intercept) ∧
    is_trapezoid p1 p2 p3 p4 :=
by sorry

end polygon_is_trapezoid_l407_40730


namespace new_mean_after_combining_l407_40714

theorem new_mean_after_combining (n1 n2 : ℕ) (mean1 mean2 additional : ℚ) :
  let sum1 : ℚ := n1 * mean1
  let sum2 : ℚ := n2 * mean2
  let total_sum : ℚ := sum1 + sum2 + additional
  let total_count : ℕ := n1 + n2 + 1
  (total_sum / total_count : ℚ) = (n1 * mean1 + n2 * mean2 + additional) / (n1 + n2 + 1) :=
by
  sorry

-- Example usage with the given problem values
example : 
  let n1 : ℕ := 7
  let n2 : ℕ := 9
  let mean1 : ℚ := 15
  let mean2 : ℚ := 28
  let additional : ℚ := 100
  (n1 * mean1 + n2 * mean2 + additional) / (n1 + n2 + 1) = 457 / 17 :=
by
  sorry

end new_mean_after_combining_l407_40714


namespace f_range_f_range_complete_l407_40798

noncomputable def f (x : ℝ) : ℝ :=
  |Real.sin x| / Real.sin x + Real.cos x / |Real.cos x| + |Real.tan x| / Real.tan x

theorem f_range :
  ∀ x : ℝ, Real.sin x ≠ 0 ∧ Real.cos x ≠ 0 →
    f x = -1 ∨ f x = 3 :=
by sorry

theorem f_range_complete :
  ∃ x y : ℝ, Real.sin x ≠ 0 ∧ Real.cos x ≠ 0 ∧
             Real.sin y ≠ 0 ∧ Real.cos y ≠ 0 ∧
             f x = -1 ∧ f y = 3 :=
by sorry

end f_range_f_range_complete_l407_40798


namespace no_simultaneous_doughnut_and_syrup_l407_40736

theorem no_simultaneous_doughnut_and_syrup :
  ¬∃ (x : ℝ), (x^2 - 9*x + 13 < 0) ∧ (x^2 + x - 5 < 0) := by
  sorry

end no_simultaneous_doughnut_and_syrup_l407_40736


namespace angle_cosine_relation_l407_40794

/-- Given a point Q in 3D space with positive coordinates, and angles α, β, γ between OQ and the x, y, z axes respectively, prove that if cos α = 2/5 and cos β = 1/4, then cos γ = √(311)/20 -/
theorem angle_cosine_relation (Q : ℝ × ℝ × ℝ) (α β γ : ℝ) 
  (h_pos : Q.1 > 0 ∧ Q.2.1 > 0 ∧ Q.2.2 > 0)
  (h_α : α = Real.arccos (Q.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_β : β = Real.arccos (Q.2.1 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_γ : γ = Real.arccos (Q.2.2 / Real.sqrt (Q.1^2 + Q.2.1^2 + Q.2.2^2)))
  (h_cos_α : Real.cos α = 2/5)
  (h_cos_β : Real.cos β = 1/4) :
  Real.cos γ = Real.sqrt 311 / 20 := by
  sorry

end angle_cosine_relation_l407_40794


namespace sandy_shirt_cost_l407_40790

/-- The amount Sandy spent on clothes, in cents -/
def total_spent : ℕ := 3356

/-- The cost of shorts, in cents -/
def shorts_cost : ℕ := 1399

/-- The cost of jacket, in cents -/
def jacket_cost : ℕ := 743

/-- The cost of shirt, in cents -/
def shirt_cost : ℕ := total_spent - (shorts_cost + jacket_cost)

theorem sandy_shirt_cost : shirt_cost = 1214 := by
  sorry

end sandy_shirt_cost_l407_40790


namespace existence_of_subset_with_property_P_l407_40723

-- Define the property P for a subset A and a natural number m
def property_P (A : Set ℕ) (m : ℕ) : Prop :=
  ∀ k : ℕ, ∃ a : ℕ → ℕ, 
    (∀ i, i < k → a i ∈ A) ∧
    (∀ i, i < k - 1 → 1 ≤ a (i + 1) - a i ∧ a (i + 1) - a i ≤ m)

-- Main theorem
theorem existence_of_subset_with_property_P 
  (r : ℕ) (partition : Fin r → Set ℕ) 
  (partition_properties : 
    (∀ i j, i ≠ j → partition i ∩ partition j = ∅) ∧ 
    (⋃ i, partition i) = Set.univ) :
  ∃ (i : Fin r) (m : ℕ), property_P (partition i) m :=
sorry

end existence_of_subset_with_property_P_l407_40723


namespace factorization_sum_l407_40768

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 17*x + 72 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 8*x - 63 = (x + b)*(x - c)) →
  a + b + c = 24 := by
  sorry

end factorization_sum_l407_40768


namespace dice_probability_l407_40756

def first_die : Finset ℕ := {1, 3, 5, 6}
def second_die : Finset ℕ := {1, 2, 4, 5, 7, 9}

def sum_in_range (x : ℕ) (y : ℕ) : Bool :=
  let sum := x + y
  8 ≤ sum ∧ sum ≤ 10

def favorable_outcomes : Finset (ℕ × ℕ) :=
  (first_die.product second_die).filter (fun (x, y) ↦ sum_in_range x y)

def total_outcomes : ℕ := (first_die.card * second_die.card : ℕ)

theorem dice_probability :
  (favorable_outcomes.card : ℚ) / total_outcomes = 7 / 18 := by
  sorry

#eval favorable_outcomes
#eval total_outcomes

end dice_probability_l407_40756


namespace fly_speed_fly_speed_problem_l407_40700

/-- The speed of a fly moving between two cyclists --/
theorem fly_speed (cyclist_speed : ℝ) (initial_distance : ℝ) (fly_distance : ℝ) : ℝ :=
  let relative_speed := 2 * cyclist_speed
  let meeting_time := initial_distance / relative_speed
  fly_distance / meeting_time

/-- Given the conditions of the problem, prove that the fly's speed is 15 miles/hour --/
theorem fly_speed_problem : fly_speed 10 50 37.5 = 15 := by
  sorry

end fly_speed_fly_speed_problem_l407_40700


namespace necessary_but_not_sufficient_l407_40717

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, (∀ x : ℝ, x^2 + 2*x + 1 - a^2 < 0 → -1 + a < x ∧ x < -1 - a) → a < 1) ∧
  (∃ a : ℝ, a < 1 ∧ ¬(∀ x : ℝ, x^2 + 2*x + 1 - a^2 < 0 → -1 + a < x ∧ x < -1 - a)) :=
by sorry

end necessary_but_not_sufficient_l407_40717


namespace monkey_climb_theorem_l407_40747

/-- The height of a tree that a monkey can climb in 15 hours, 
    given that it hops 3 ft up and slips 2 ft back each hour except for the last hour. -/
def tree_height : ℕ :=
  let hop_distance : ℕ := 3
  let slip_distance : ℕ := 2
  let total_hours : ℕ := 15
  let net_progress_per_hour : ℕ := hop_distance - slip_distance
  let height_before_last_hour : ℕ := net_progress_per_hour * (total_hours - 1)
  height_before_last_hour + hop_distance

theorem monkey_climb_theorem : tree_height = 17 := by
  sorry

end monkey_climb_theorem_l407_40747


namespace same_side_probability_is_seven_twentyfourths_l407_40737

/-- Represents a 12-sided die with specific colored sides. -/
structure TwelveSidedDie :=
  (maroon : Nat)
  (teal : Nat)
  (cyan : Nat)
  (sparkly : Nat)
  (total_sides : Nat)
  (side_sum : maroon + teal + cyan + sparkly = total_sides)

/-- The probability of two dice showing the same side when rolled. -/
def same_side_probability (d : TwelveSidedDie) : Rat :=
  (d.maroon^2 + d.teal^2 + d.cyan^2 + d.sparkly^2) / d.total_sides^2

/-- The specific die used in the problem. -/
def problem_die : TwelveSidedDie :=
  { maroon := 3
    teal := 4
    cyan := 4
    sparkly := 1
    total_sides := 12
    side_sum := by decide }

/-- Theorem stating that the probability of two problem dice showing the same side is 7/24. -/
theorem same_side_probability_is_seven_twentyfourths :
  same_side_probability problem_die = 7 / 24 := by
  sorry

end same_side_probability_is_seven_twentyfourths_l407_40737


namespace maximal_closely_related_interval_l407_40724

/-- Two functions are closely related on an interval if their difference is bounded by 1 -/
def closely_related (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x ∈ Set.Icc a b, |f x - g x| ≤ 1

/-- The given functions f and g -/
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

/-- The theorem stating that [2, 3] is the maximal closely related interval for f and g -/
theorem maximal_closely_related_interval :
  closely_related f g 2 3 ∧
  ∀ a b : ℝ, a < 2 ∨ b > 3 → ¬(closely_related f g a b) :=
sorry

end maximal_closely_related_interval_l407_40724


namespace tommy_balloons_l407_40772

/-- The number of balloons Tommy initially had -/
def initial_balloons : ℕ := 71

/-- The number of balloons Tommy's mom gave him -/
def mom_balloons : ℕ := 34

/-- The number of balloons Tommy gave to his friends -/
def friend_balloons : ℕ := 15

/-- The number of teddy bears Tommy got after exchanging balloons -/
def teddy_bears : ℕ := 30

/-- The exchange rate of balloons to teddy bears -/
def exchange_rate : ℕ := 3

theorem tommy_balloons : 
  initial_balloons + mom_balloons - friend_balloons = teddy_bears * exchange_rate := by
  sorry

end tommy_balloons_l407_40772


namespace history_score_is_84_percent_l407_40755

/-- Given a student's scores in math and a third subject, along with a desired overall average,
    this function calculates the required score in history. -/
def calculate_history_score (math_score : ℚ) (third_subject_score : ℚ) (desired_average : ℚ) : ℚ :=
  3 * desired_average - math_score - third_subject_score

/-- Theorem stating that given the specific scores and desired average,
    the calculated history score is 84%. -/
theorem history_score_is_84_percent :
  calculate_history_score 72 69 75 = 84 := by
  sorry

#eval calculate_history_score 72 69 75

end history_score_is_84_percent_l407_40755


namespace angle_range_in_special_triangle_l407_40762

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a + b = 2c, then 0 < C ≤ π/3 -/
theorem angle_range_in_special_triangle (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  a + b = 2 * c →
  0 < C ∧ C ≤ π / 3 := by
  sorry


end angle_range_in_special_triangle_l407_40762


namespace gcd_of_three_numbers_l407_40703

theorem gcd_of_three_numbers : Nat.gcd 84 (Nat.gcd 294 315) = 21 := by
  sorry

end gcd_of_three_numbers_l407_40703


namespace solution_value_l407_40776

theorem solution_value (r s : ℝ) : 
  (3 * r^2 - 5 * r = 7) → 
  (3 * s^2 - 5 * s = 7) → 
  r ≠ s →
  (9 * r^2 - 9 * s^2) / (r - s) = 15 := by
sorry

end solution_value_l407_40776


namespace circle_equation_l407_40782

theorem circle_equation (x y θ : ℝ) : 
  (x = 3 + 4 * Real.cos θ ∧ y = -2 + 4 * Real.sin θ) → 
  (x - 3)^2 + (y + 2)^2 = 16 := by
sorry

end circle_equation_l407_40782


namespace circle_properties_l407_40740

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 4

-- Define the center of the circle
def center : ℝ × ℝ := (1, -2)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem statement
theorem circle_properties :
  ∀ x y : ℝ, circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
sorry

end circle_properties_l407_40740


namespace coeff_x_squared_expansion_l407_40791

open Polynomial

/-- The coefficient of x^2 in the expansion of (1-2x)^5(1+3x)^4 is -26 -/
theorem coeff_x_squared_expansion : 
  (coeff ((1 - 2 * X) ^ 5 * (1 + 3 * X) ^ 4) 2) = -26 := by
  sorry

end coeff_x_squared_expansion_l407_40791


namespace undefined_rational_function_l407_40787

theorem undefined_rational_function (x : ℝ) :
  (x^2 - 12*x + 36 = 0) → ¬∃y, y = (3*x^3 + 5) / (x^2 - 12*x + 36) :=
by
  sorry

end undefined_rational_function_l407_40787


namespace divisibility_of_group_difference_l407_40783

/-- Represents a person in the circle, either a boy or a girl -/
inductive Person
| Boy
| Girl

/-- The circle of people -/
def Circle := List Person

/-- Count the number of groups of 3 consecutive people with exactly one boy -/
def countGroupsWithOneBoy (circle : Circle) : Nat :=
  sorry

/-- Count the number of groups of 3 consecutive people with exactly one girl -/
def countGroupsWithOneGirl (circle : Circle) : Nat :=
  sorry

theorem divisibility_of_group_difference (n : Nat) (circle : Circle) 
    (h1 : n ≥ 3)
    (h2 : circle.length = n) :
  let a := countGroupsWithOneBoy circle
  let b := countGroupsWithOneGirl circle
  3 ∣ (a - b) :=
by sorry

end divisibility_of_group_difference_l407_40783


namespace parallel_transitivity_counterexample_l407_40797

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity_counterexample 
  (a : Line) (α β : Plane) :
  ¬(∀ a α β, parallel_line_plane a α → parallel_line_plane a β → 
    parallel_plane_plane α β) :=
sorry

end parallel_transitivity_counterexample_l407_40797


namespace mirror_height_for_full_body_view_l407_40785

/-- 
Theorem: For a person standing upright in front of a vertical mirror, 
the minimum mirror height required to see their full body is exactly 
half of their height.
-/
theorem mirror_height_for_full_body_view 
  (h : ℝ) -- height of the person
  (m : ℝ) -- height of the mirror
  (h_pos : h > 0) -- person's height is positive
  (m_pos : m > 0) -- mirror's height is positive
  (full_view : m ≥ h / 2) -- condition for full body view
  (minimal : ∀ m' : ℝ, m' > 0 → m' < m → ¬(m' ≥ h / 2)) -- m is minimal
  : m = h / 2 := by sorry

end mirror_height_for_full_body_view_l407_40785


namespace sum_inequality_l407_40778

theorem sum_inequality (a b c : ℝ) 
  (ha : 1/Real.sqrt 2 ≤ a ∧ a ≤ Real.sqrt 2)
  (hb : 1/Real.sqrt 2 ≤ b ∧ b ≤ Real.sqrt 2)
  (hc : 1/Real.sqrt 2 ≤ c ∧ c ≤ Real.sqrt 2) :
  (3/(a+2*b) + 3/(b+2*c) + 3/(c+2*a)) ≥ (2/(a+b) + 2/(b+c) + 2/(c+a)) := by
  sorry

end sum_inequality_l407_40778


namespace goats_count_l407_40779

/-- Represents the number of animals on a farm --/
structure Farm where
  goats : ℕ
  cows : ℕ
  pigs : ℕ
  chickens : ℕ
  ducks : ℕ

/-- Represents the conditions given in the problem --/
def farm_conditions (f : Farm) : Prop :=
  f.pigs = 2 * f.cows ∧
  f.cows = f.goats + 4 ∧
  f.chickens = 3 * f.pigs ∧
  f.ducks = (f.cows + f.goats) / 2 ∧
  f.goats + f.cows + f.pigs + f.chickens + f.ducks = 172

/-- The theorem to be proved --/
theorem goats_count (f : Farm) (h : farm_conditions f) : f.goats = 12 := by
  sorry


end goats_count_l407_40779


namespace find_number_l407_40784

theorem find_number : ∃ x : ℝ, 0.20 * x + 0.25 * 60 = 23 ∧ x = 40 := by
  sorry

end find_number_l407_40784


namespace inequality_proof_l407_40741

theorem inequality_proof (a b : ℝ) (n : ℕ+) 
  (ha : a > 0) (hb : b > 0) (hab : 1/a + 1/b = 1) :
  (a + b)^(n : ℝ) - a^(n : ℝ) - b^(n : ℝ) ≥ 2^(2*(n : ℝ)) - 2^((n : ℝ) + 1) :=
by sorry

end inequality_proof_l407_40741


namespace f_simplification_symmetry_condition_g_maximum_condition_l407_40765

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x / 2) * Real.cos (x / 2) - 2 * Real.sqrt 3 * Real.sin (x / 2) ^ 2 + Real.sqrt 3

noncomputable def g (x : ℝ) : ℝ := f x + Real.sin x

theorem f_simplification (x : ℝ) : f x = 2 * Real.sin (x + π / 3) := by sorry

theorem symmetry_condition (φ : ℝ) :
  (∃ k : ℤ, π / 3 + φ + π / 3 = k * π) → φ = π / 3 := by sorry

theorem g_maximum_condition (θ : ℝ) :
  (∀ x : ℝ, g x ≤ g θ) → Real.cos θ = Real.sqrt 3 / Real.sqrt 7 := by sorry

end f_simplification_symmetry_condition_g_maximum_condition_l407_40765


namespace intersection_with_complement_l407_40701

def U : Set ℤ := Set.univ

def A : Set ℤ := {-1, 1, 2}

def B : Set ℤ := {-1, 1}

theorem intersection_with_complement :
  A ∩ (Set.compl B) = {2} := by sorry

end intersection_with_complement_l407_40701


namespace seven_twelfths_decimal_l407_40729

theorem seven_twelfths_decimal : 
  (7 : ℚ) / 12 = 0.5833333333333333 := by sorry

end seven_twelfths_decimal_l407_40729


namespace zhang_bing_special_year_l407_40751

/-- Given that Zhang Bing was born in 1953, this theorem proves the existence and uniqueness of a year between 1953 and 2023 where his age is both a multiple of 9 and equal to the sum of the digits of that year. -/
theorem zhang_bing_special_year : 
  ∃! Y : ℕ, 1953 < Y ∧ Y < 2023 ∧ 
  (∃ k : ℕ, Y - 1953 = 9 * k) ∧
  (Y - 1953 = (Y / 1000) + ((Y % 1000) / 100) + ((Y % 100) / 10) + (Y % 10)) :=
by sorry

end zhang_bing_special_year_l407_40751


namespace M_inequalities_l407_40766

/-- M_(n,k,h) is the maximum number of h-element subsets of an n-element set X with property P_k(X) -/
def M (n k h : ℕ) : ℕ := sorry

/-- The three inequalities for M_(n,k,h) -/
theorem M_inequalities (n k h : ℕ) (hn : n > 0) (hk : k > 0) (hh : h > 0) (hnkh : n ≥ k) (hkh : k ≥ h) :
  (M n k h ≤ (n / h) * M (n-1) (k-1) (h-1)) ∧
  (M n k h ≥ (n / (n-h)) * M (n-1) k h) ∧
  (M n k h ≤ M (n-1) (k-1) (h-1) + M (n-1) k h) :=
sorry

end M_inequalities_l407_40766


namespace interest_calculation_l407_40775

/-- Represents the problem of finding the minimum number of years for a specific interest calculation. -/
theorem interest_calculation (principal1 principal2 rate1 rate2 target_interest : ℚ) :
  principal1 = 800 →
  principal2 = 1400 →
  rate1 = 3 / 100 →
  rate2 = 5 / 100 →
  target_interest = 350 →
  (∃ (n : ℕ), (principal1 * rate1 * n + principal2 * rate2 * n ≥ target_interest) ∧
    (∀ (m : ℕ), m < n → principal1 * rate1 * m + principal2 * rate2 * m < target_interest)) →
  (∃ (n : ℕ), (principal1 * rate1 * n + principal2 * rate2 * n ≥ target_interest) ∧
    (∀ (m : ℕ), m < n → principal1 * rate1 * m + principal2 * rate2 * m < target_interest) ∧
    n = 4) :=
by sorry

end interest_calculation_l407_40775


namespace intersection_S_complement_T_l407_40706

-- Define the universal set U as ℝ
def U := ℝ

-- Define set S
def S : Set ℝ := {x : ℝ | x^2 - x ≤ 0}

-- Define set T
def T : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x ∧ x ≤ 0}

-- State the theorem
theorem intersection_S_complement_T : S ∩ (Set.univ \ T) = {0} := by sorry

end intersection_S_complement_T_l407_40706


namespace geometric_series_product_l407_40770

theorem geometric_series_product (y : ℝ) : y = 9 ↔ 
  (∑' n, (1/3)^n) * (∑' n, (-1/3)^n) = ∑' n, (1/y)^n := by sorry

end geometric_series_product_l407_40770


namespace consecutive_integers_sum_of_powers_l407_40721

theorem consecutive_integers_sum_of_powers (n : ℕ) : 
  (n > 0) →
  ((n - 1)^2 + n^2 + (n + 1)^2 = 9458) →
  ((n - 1)^4 + n^4 + (n + 1)^4 = 30212622) :=
by sorry

end consecutive_integers_sum_of_powers_l407_40721


namespace parallelogram_area_example_l407_40757

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 21 cm and height 11 cm is 231 square centimeters -/
theorem parallelogram_area_example : parallelogram_area 21 11 = 231 := by
  sorry

end parallelogram_area_example_l407_40757


namespace salt_production_average_l407_40789

/-- The salt production problem --/
theorem salt_production_average (initial_production : ℕ) (monthly_increase : ℕ) (months : ℕ) (days_in_year : ℕ) :
  let total_production := initial_production + (monthly_increase * (months * (months - 1)) / 2)
  (total_production : ℚ) / days_in_year = 121.1 := by
  sorry

#check salt_production_average 3000 100 12 365

end salt_production_average_l407_40789


namespace quadratic_real_roots_l407_40734

theorem quadratic_real_roots (k : ℝ) :
  (∃ x : ℝ, k * x^2 - 2 * x - 1 = 0) ↔ k ≥ -1 := by
  sorry

end quadratic_real_roots_l407_40734


namespace fraction_sum_greater_than_sum_fraction_l407_40748

theorem fraction_sum_greater_than_sum_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  1 / a + 1 / b > 1 / (a + b) := by
  sorry

end fraction_sum_greater_than_sum_fraction_l407_40748


namespace exist_mutual_wins_l407_40788

/-- Represents a football tournament --/
structure Tournament :=
  (num_teams : Nat)
  (scores_round1 : Fin num_teams → Nat)
  (scores_round2 : Fin num_teams → Nat)

/-- Properties of the tournament --/
def TournamentProperties (t : Tournament) : Prop :=
  t.num_teams = 20 ∧
  (∀ i j, i ≠ j → t.scores_round1 i ≠ t.scores_round1 j) ∧
  (∃ s, ∀ i, t.scores_round2 i = s)

/-- Theorem stating the existence of two teams that each won one game against the other --/
theorem exist_mutual_wins (t : Tournament) (h : TournamentProperties t) :
  ∃ i j, i ≠ j ∧ 
    t.scores_round2 i - t.scores_round1 i = 2 ∧
    t.scores_round2 j - t.scores_round1 j = 2 :=
by sorry

end exist_mutual_wins_l407_40788


namespace units_digit_of_7_power_2023_l407_40771

theorem units_digit_of_7_power_2023 : ∃ n : ℕ, 7^2023 ≡ 3 [ZMOD 10] :=
by sorry

end units_digit_of_7_power_2023_l407_40771
