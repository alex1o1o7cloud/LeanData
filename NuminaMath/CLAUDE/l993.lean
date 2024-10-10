import Mathlib

namespace inequality_proof_equality_condition_l993_99327

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x * y / (x^2 + y^2 + 2 * z^2)) +
  Real.sqrt (y * z / (y^2 + z^2 + 2 * x^2)) +
  Real.sqrt (z * x / (z^2 + x^2 + 2 * y^2)) ≤ 3 / 2 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  Real.sqrt (x * y / (x^2 + y^2 + 2 * z^2)) +
  Real.sqrt (y * z / (y^2 + z^2 + 2 * x^2)) +
  Real.sqrt (z * x / (z^2 + x^2 + 2 * y^2)) = 3 / 2 ↔
  x = y ∧ y = z :=
by sorry

end inequality_proof_equality_condition_l993_99327


namespace square_sum_value_l993_99371

theorem square_sum_value (m n : ℝ) :
  (m^2 + 3*n^2)^2 - 4*(m^2 + 3*n^2) - 12 = 0 →
  m^2 + 3*n^2 = 6 := by
sorry

end square_sum_value_l993_99371


namespace minimum_balls_to_draw_l993_99353

theorem minimum_balls_to_draw (red green yellow blue white black : ℕ) 
  (h_red : red = 35) (h_green : green = 25) (h_yellow : yellow = 22) 
  (h_blue : blue = 15) (h_white : white = 14) (h_black : black = 12) :
  let total := red + green + yellow + blue + white + black
  let threshold := 18
  ∃ n : ℕ, n = 93 ∧ 
    (∀ m : ℕ, m < n → 
      ∃ (r g y b w k : ℕ), r + g + y + b + w + k = m ∧
        r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black ∧
        r < threshold ∧ g < threshold ∧ y < threshold ∧ 
        b < threshold ∧ w < threshold ∧ k < threshold) ∧
    (∀ m : ℕ, m ≥ n → 
      ∀ (r g y b w k : ℕ), r + g + y + b + w + k = m →
        r ≤ red ∧ g ≤ green ∧ y ≤ yellow ∧ b ≤ blue ∧ w ≤ white ∧ k ≤ black →
        r ≥ threshold ∨ g ≥ threshold ∨ y ≥ threshold ∨ 
        b ≥ threshold ∨ w ≥ threshold ∨ k ≥ threshold) :=
by sorry

end minimum_balls_to_draw_l993_99353


namespace exam_score_97_impossible_l993_99369

theorem exam_score_97_impossible :
  ¬ ∃ (correct unanswered : ℕ),
    correct + unanswered ≤ 20 ∧
    5 * correct + unanswered = 97 :=
by sorry

end exam_score_97_impossible_l993_99369


namespace quadratic_max_condition_l993_99330

/-- Given a quadratic function y = ax² - 2ax + c with a ≠ 0 and maximum value 2,
    prove that c - a = 2 and a < 0 -/
theorem quadratic_max_condition (a c : ℝ) (h1 : a ≠ 0) :
  (∀ x, a * x^2 - 2*a*x + c ≤ 2) ∧ 
  (∃ x, a * x^2 - 2*a*x + c = 2) →
  c - a = 2 ∧ a < 0 := by
  sorry

end quadratic_max_condition_l993_99330


namespace quadratic_roots_condition_l993_99379

-- Define the quadratic equation
def quadratic (m x : ℝ) : ℝ := x^2 - (3*m + 2)*x + 2*(m + 6)

-- Define the property of having two real roots greater than 3
def has_two_roots_greater_than_three (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 3 ∧ x₂ > 3 ∧ quadratic m x₁ = 0 ∧ quadratic m x₂ = 0

-- Theorem statement
theorem quadratic_roots_condition (m : ℝ) :
  has_two_roots_greater_than_three m ↔ 4/3 < m ∧ m < 15/7 := by
  sorry

end quadratic_roots_condition_l993_99379


namespace fraction_equality_l993_99308

theorem fraction_equality (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) 
  (h3 : (4*x + 2*y) / (2*x - 4*y) = 3) : 
  (2*x + 4*y) / (4*x - 2*y) = 9/13 := by
  sorry

end fraction_equality_l993_99308


namespace complement_equivalence_l993_99341

def U (a : ℝ) := {x : ℕ | 0 < x ∧ x ≤ ⌊a⌋}
def P : Set ℕ := {1, 2, 3}
def Q : Set ℕ := {4, 5, 6}

theorem complement_equivalence (a : ℝ) :
  (6 ≤ a ∧ a < 7) ↔ (U a \ P = Q) :=
sorry

end complement_equivalence_l993_99341


namespace at_least_one_positive_l993_99378

theorem at_least_one_positive (a b c : ℝ) :
  (a > 0 ∨ b > 0 ∨ c > 0) ↔ ¬(a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0) := by
  sorry

end at_least_one_positive_l993_99378


namespace line_through_point_parallel_to_line_l993_99303

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) is on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The problem statement -/
theorem line_through_point_parallel_to_line :
  let l1 : Line := { a := 3, b := 1, c := -2 }
  let l2 : Line := { a := 3, b := 1, c := -5 }
  l2.contains 2 (-1) ∧ Line.parallel l1 l2 := by
  sorry

end line_through_point_parallel_to_line_l993_99303


namespace expression_value_at_negative_three_l993_99350

theorem expression_value_at_negative_three :
  let x : ℤ := -3
  let expr := 5 * x - (3 * x - 2 * (2 * x - 3))
  expr = -24 := by sorry

end expression_value_at_negative_three_l993_99350


namespace gcd_problem_l993_99340

theorem gcd_problem (p : Nat) (h : Prime p) :
  Nat.gcd (p^7 + 1) (p^7 + p^3 + 1) = 1 :=
by sorry

end gcd_problem_l993_99340


namespace cube_root_of_one_sixty_fourth_l993_99326

theorem cube_root_of_one_sixty_fourth (x : ℝ) : x^3 = 1/64 → x = 1/4 := by
  sorry

end cube_root_of_one_sixty_fourth_l993_99326


namespace shaded_area_sum_l993_99300

/-- Represents the shaded area in each step of the square division pattern -/
def shadedAreaSeries : ℕ → ℚ
  | 0 => 1/4
  | n + 1 => (shadedAreaSeries n) / 16

/-- The sum of the infinite geometric series representing the total shaded area -/
def totalShadedArea : ℚ := 4/15

theorem shaded_area_sum :
  (∑' n, shadedAreaSeries n) = totalShadedArea := by
  sorry

end shaded_area_sum_l993_99300


namespace sum_of_digits_1_to_5000_l993_99377

def sum_of_digits (n : ℕ) : ℕ := sorry

def sequence_sum (n : ℕ) : ℕ := sorry

theorem sum_of_digits_1_to_5000 : 
  sequence_sum 5000 = 194450 := by sorry

end sum_of_digits_1_to_5000_l993_99377


namespace vector_at_t_5_l993_99388

def line_parameterization (t : ℝ) : ℝ × ℝ := sorry

theorem vector_at_t_5 (h1 : line_parameterization 1 = (2, 7))
                      (h2 : line_parameterization 4 = (8, -5)) :
  line_parameterization 5 = (10, -9) := by sorry

end vector_at_t_5_l993_99388


namespace division_problem_l993_99337

theorem division_problem (a b q : ℕ) (h1 : a - b = 1370) (h2 : a = 1626) (h3 : a = b * q + 15) : q = 6 := by
  sorry

end division_problem_l993_99337


namespace binary_division_and_double_l993_99314

def binary_number : ℕ := 3666 -- 111011010010₂ in decimal

theorem binary_division_and_double :
  (binary_number % 4) * 2 = 4 := by
  sorry

end binary_division_and_double_l993_99314


namespace paint_needed_l993_99399

theorem paint_needed (total_needed : ℕ) (existing : ℕ) (newly_bought : ℕ) 
  (h1 : total_needed = 70)
  (h2 : existing = 36)
  (h3 : newly_bought = 23) :
  total_needed - (existing + newly_bought) = 11 := by
  sorry

end paint_needed_l993_99399


namespace same_last_three_digits_l993_99390

theorem same_last_three_digits (N : ℕ) (h1 : N > 0) :
  (∃ (a b c : ℕ), a < 10 ∧ b < 10 ∧ c < 10 ∧ 
   N % 1000 = 100 * a + 10 * b + c ∧
   (N^2) % 1000 = 100 * a + 10 * b + c) →
  N % 1000 = 873 :=
by sorry

end same_last_three_digits_l993_99390


namespace expression_evaluation_l993_99358

theorem expression_evaluation : 
  2 * (7 ^ (1/3 : ℝ)) + 16 ^ (3/4 : ℝ) + (4 / (Real.sqrt 3 - 1)) ^ (0 : ℝ) + (-3) ^ (-1 : ℝ) = 44/3 := by
  sorry

end expression_evaluation_l993_99358


namespace solution_set_when_a_is_2_range_of_a_for_solution_l993_99306

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 2 * |x - 1|

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x > 5} = {x : ℝ | x < -1/3 ∨ x > 3} :=
sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_solution :
  {a : ℝ | ∃ x, f a x - |x - 1| ≤ |a - 2|} = {a : ℝ | a ≤ 3/2} :=
sorry

end solution_set_when_a_is_2_range_of_a_for_solution_l993_99306


namespace gcd_1721_1733_l993_99352

theorem gcd_1721_1733 : Nat.gcd 1721 1733 = 1 := by
  sorry

end gcd_1721_1733_l993_99352


namespace penguin_arrangements_l993_99332

def word_length : ℕ := 7
def repeated_letter_count : ℕ := 2

theorem penguin_arrangements :
  (word_length.factorial / repeated_letter_count.factorial) = 2520 := by
  sorry

end penguin_arrangements_l993_99332


namespace system_solution_l993_99360

theorem system_solution (a b c : ℝ) : 
  a^2 + a*b + c^2 = 31 ∧ 
  b^2 + a*b - c^2 = 18 ∧ 
  a^2 - b^2 = 7 → 
  c = Real.sqrt 3 ∨ c = -Real.sqrt 3 :=
by sorry

end system_solution_l993_99360


namespace line_equation_l993_99389

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (0, 2)
def line (k : ℝ) (x y : ℝ) : Prop := y = k*x + 2

-- Define a point on both the line and the parabola
def intersection_point (k x y : ℝ) : Prop :=
  parabola x y ∧ line k x y

-- Define the condition for a circle passing through three points
def circle_condition (x1 y1 x2 y2 : ℝ) : Prop :=
  x1*x2 + y1*y2 = 0

theorem line_equation :
  ∀ k : ℝ,
  (∃ x1 y1 x2 y2 : ℝ,
    x1 ≠ x2 ∧
    intersection_point k x1 y1 ∧
    intersection_point k x2 y2 ∧
    circle_condition x1 y1 x2 y2) →
  k = -1 :=
sorry

end line_equation_l993_99389


namespace page_number_added_twice_l993_99380

theorem page_number_added_twice (n : ℕ) (h1 : n > 0) : 
  (∃ p : ℕ, p ≤ n ∧ n * (n + 1) / 2 + p = 2630) → 
  (∃ p : ℕ, p ≤ n ∧ n * (n + 1) / 2 + p = 2630 ∧ p = 2) :=
by sorry

end page_number_added_twice_l993_99380


namespace trigonometric_equation_solution_l993_99336

theorem trigonometric_equation_solution :
  ∀ x : ℝ,
  (Real.sin (2019 * x))^4 + (Real.cos (2022 * x))^2019 * (Real.cos (2019 * x))^2018 = 1 ↔
  (∃ n : ℤ, x = π / 4038 + π * n / 2019) ∨ (∃ k : ℤ, x = π * k / 3) :=
by sorry

end trigonometric_equation_solution_l993_99336


namespace is_circle_center_l993_99354

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 8*y + 9 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (3, -4)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 16 :=
by sorry

end is_circle_center_l993_99354


namespace equilateral_triangle_circumradius_ratio_l993_99361

/-- Given two equilateral triangles with side lengths B and b (B ≠ b) and circumradii S and s respectively,
    the ratio of their circumradii S/s is always equal to the ratio of their side lengths B/b. -/
theorem equilateral_triangle_circumradius_ratio 
  (B b S s : ℝ) 
  (hB : B > 0) 
  (hb : b > 0) 
  (hne : B ≠ b) 
  (hS : S = B * Real.sqrt 3 / 3) 
  (hs : s = b * Real.sqrt 3 / 3) : 
  S / s = B / b := by
  sorry

end equilateral_triangle_circumradius_ratio_l993_99361


namespace triangular_pyramid_angles_l993_99368

/-- 
Given a triangular pyramid with lateral surface area S and lateral edge length l,
if the plane angles at the apex form an arithmetic progression with common difference π/6,
then the angles are as specified.
-/
theorem triangular_pyramid_angles (S l : ℝ) (h_positive_S : S > 0) (h_positive_l : l > 0) :
  let α := Real.arcsin ((S * (Real.sqrt 3 - 1)) / l^2)
  ∃ (θ₁ θ₂ θ₃ : ℝ),
    (θ₁ = α - π/6 ∧ θ₂ = α ∧ θ₃ = α + π/6) ∧
    (θ₁ + θ₂ + θ₃ = π/2) ∧
    (θ₃ - θ₂ = θ₂ - θ₁) ∧
    (θ₃ - θ₂ = π/6) ∧
    (S = (l^2 / 2) * (Real.sin θ₁ + Real.sin θ₂ + Real.sin θ₃)) :=
by sorry

end triangular_pyramid_angles_l993_99368


namespace abs_square_eq_neg_cube_l993_99339

theorem abs_square_eq_neg_cube (a b : ℤ) : |a|^2 = -(b^3) → a = -8 ∧ b = -4 :=
by sorry

end abs_square_eq_neg_cube_l993_99339


namespace intersection_chord_length_l993_99310

/-- Line in parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Circle in polar form -/
structure PolarCircle where
  ρ : ℝ → ℝ

/-- Chord length calculation -/
def chordLength (l : ParametricLine) (c : PolarCircle) : ℝ := sorry

/-- Main theorem -/
theorem intersection_chord_length :
  let l : ParametricLine := { x := fun t => t + 1, y := fun t => t - 3 }
  let c : PolarCircle := { ρ := fun θ => 4 * Real.cos θ }
  chordLength l c = 2 * Real.sqrt 2 := by sorry

end intersection_chord_length_l993_99310


namespace triangle_problem_l993_99338

theorem triangle_problem (a b c A B C : ℝ) : 
  -- Conditions
  (2 * Real.sin (7 * π / 6) * Real.sin (π / 6 + C) + Real.cos C = -1 / 2) →
  (c = Real.sqrt 13) →
  (1 / 2 * a * b * Real.sin C = 3 * Real.sqrt 3) →
  -- Conclusions
  (C = π / 3) ∧ 
  (Real.sin A + Real.sin B = 7 * Real.sqrt 39 / 26) := by
sorry

end triangle_problem_l993_99338


namespace zhang_san_not_losing_probability_l993_99387

theorem zhang_san_not_losing_probability 
  (p_win : ℚ) (p_draw : ℚ) 
  (h_win : p_win = 1/3) 
  (h_draw : p_draw = 1/4) : 
  p_win + p_draw = 7/12 := by
  sorry

end zhang_san_not_losing_probability_l993_99387


namespace cubic_function_properties_l993_99346

/-- A cubic function with a local maximum at x = -1 and a local minimum at x = 3 -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f -/
def f_deriv (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_properties :
  ∃ (a b c : ℝ),
    (f_deriv a b (-1) = 0) ∧
    (f_deriv a b 3 = 0) ∧
    (f a b c (-1) = 7) ∧
    (a = -3) ∧
    (b = -9) ∧
    (c = 2) ∧
    (f a b c 3 = -25) := by
  sorry

end cubic_function_properties_l993_99346


namespace sum_is_parabola_l993_99348

-- Define the original parabola
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the reflected parabola
def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 - b * x + c

-- Define the translated original parabola (f)
def f (a b c : ℝ) (x : ℝ) : ℝ := original_parabola a b c x + 3

-- Define the translated reflected parabola (g)
def g (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c x - 2

-- Theorem: The sum of f and g is a parabola
theorem sum_is_parabola (a b c : ℝ) :
  ∃ (A C : ℝ), ∀ x, f a b c x + g a b c x = A * x^2 + C :=
sorry

end sum_is_parabola_l993_99348


namespace quadratic_inequality_solution_set_l993_99386

theorem quadratic_inequality_solution_set :
  {x : ℝ | -2 * x^2 - x + 6 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 3/2} := by sorry

end quadratic_inequality_solution_set_l993_99386


namespace total_cost_is_correct_l993_99392

def calculate_total_cost (type_a_count : ℕ) (type_b_count : ℕ) (type_c_count : ℕ)
  (type_a_price : ℚ) (type_b_price : ℚ) (type_c_price : ℚ)
  (type_a_discount : ℚ) (type_b_discount : ℚ) (type_c_discount : ℚ)
  (type_a_discount_threshold : ℕ) (type_b_discount_threshold : ℕ) (type_c_discount_threshold : ℕ) : ℚ :=
  let type_a_cost := type_a_count * type_a_price
  let type_b_cost := type_b_count * type_b_price
  let type_c_cost := type_c_count * type_c_price
  let type_a_discounted_cost := if type_a_count > type_a_discount_threshold then type_a_cost * (1 - type_a_discount) else type_a_cost
  let type_b_discounted_cost := if type_b_count > type_b_discount_threshold then type_b_cost * (1 - type_b_discount) else type_b_cost
  let type_c_discounted_cost := if type_c_count > type_c_discount_threshold then type_c_cost * (1 - type_c_discount) else type_c_cost
  type_a_discounted_cost + type_b_discounted_cost + type_c_discounted_cost

theorem total_cost_is_correct :
  calculate_total_cost 150 90 60 2 3 5 0.2 0.15 0.1 100 50 30 = 739.5 := by
  sorry

end total_cost_is_correct_l993_99392


namespace andy_twice_rahims_age_l993_99328

def rahims_current_age : ℕ := 6
def andys_age_difference : ℕ := 1

theorem andy_twice_rahims_age (x : ℕ) : 
  (rahims_current_age + andys_age_difference + x = 2 * rahims_current_age) → x = 5 := by
  sorry

end andy_twice_rahims_age_l993_99328


namespace solution_sets_l993_99374

def solution_set_1 (a b : ℝ) : Set ℝ := {x | a * x - b > 0}
def solution_set_2 (a b : ℝ) : Set ℝ := {x | (a * x + b) / (x - 2) > 0}

theorem solution_sets (a b : ℝ) :
  solution_set_1 a b = Set.Ioi 1 →
  solution_set_2 a b = Set.Iic (-1) ∪ Set.Ioi 2 :=
by sorry

end solution_sets_l993_99374


namespace quadratic_roots_root_of_two_two_as_only_root_l993_99372

/-- The quadratic equation x^2 - 2px + q = 0 -/
def quadratic_equation (p q x : ℝ) : Prop :=
  x^2 - 2*p*x + q = 0

/-- The discriminant of the quadratic equation -/
def discriminant (p q : ℝ) : ℝ :=
  4*p^2 - 4*q

theorem quadratic_roots (p q : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation p q x ∧ quadratic_equation p q y) ↔ q < p^2 :=
sorry

theorem root_of_two (p q : ℝ) :
  quadratic_equation p q 2 ↔ q = 4*p - 4 :=
sorry

theorem two_as_only_root (p q : ℝ) :
  (∀ x : ℝ, quadratic_equation p q x ↔ x = 2) ↔ (p = 2 ∧ q = 4) :=
sorry

end quadratic_roots_root_of_two_two_as_only_root_l993_99372


namespace unique_integer_satisfying_conditions_l993_99373

theorem unique_integer_satisfying_conditions (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : 7 < x ∧ x < 18)
  (h3 : 13 > x ∧ x > 2)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) : 
  x = 10 := by
  sorry

end unique_integer_satisfying_conditions_l993_99373


namespace tennis_ball_order_l993_99393

/-- The number of tennis balls originally ordered by a sports retailer -/
def original_order : ℕ := 288

/-- The number of extra yellow balls sent by mistake -/
def extra_yellow_balls : ℕ := 90

/-- The ratio of white balls to yellow balls after the error -/
def final_ratio : Rat := 8 / 13

theorem tennis_ball_order :
  ∃ (white yellow : ℕ),
    -- The retailer ordered equal numbers of white and yellow tennis balls
    white = yellow ∧
    -- The total original order
    white + yellow = original_order ∧
    -- After the error, the ratio of white to yellow balls is 8/13
    (white : Rat) / ((yellow : Rat) + extra_yellow_balls) = final_ratio :=
by sorry

end tennis_ball_order_l993_99393


namespace workshop_workers_count_l993_99325

/-- The total number of workers in a workshop given specific salary conditions -/
theorem workshop_workers_count : ℕ :=
  let average_salary : ℚ := 1000
  let technician_salary : ℚ := 1200
  let other_salary : ℚ := 820
  let technician_count : ℕ := 10
  let total_workers : ℕ := 21

  have h1 : average_salary * total_workers = 
    technician_salary * technician_count + other_salary * (total_workers - technician_count) := by sorry

  total_workers


end workshop_workers_count_l993_99325


namespace complex_modulus_problem_l993_99383

theorem complex_modulus_problem (z : ℂ) : (1 - Complex.I) * z = 1 + Complex.I → Complex.abs z = 1 := by
  sorry

end complex_modulus_problem_l993_99383


namespace divisible_by_seventeen_l993_99333

theorem divisible_by_seventeen (n : ℕ) : 
  (6^(2*n) + 2^(n+2) + 12 * 2^n) % 17 = 0 := by
  sorry

end divisible_by_seventeen_l993_99333


namespace water_left_after_experiment_l993_99319

/-- 
Proves that if Jori starts with 3 gallons of water and uses 5/4 gallons, 
she will have 7/4 gallons left.
-/
theorem water_left_after_experiment (initial_water : ℚ) (used_water : ℚ) 
  (h1 : initial_water = 3)
  (h2 : used_water = 5/4) : 
  initial_water - used_water = 7/4 := by
  sorry

end water_left_after_experiment_l993_99319


namespace inverse_proportionality_l993_99316

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = -4, then x = -2 when y = 10 -/
theorem inverse_proportionality (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 5 * (-4) = k) :
  10 * x = k → x = -2 := by
sorry

end inverse_proportionality_l993_99316


namespace max_blank_squares_l993_99329

/-- Represents a grid of unit squares -/
structure Grid :=
  (size : ℕ)

/-- Represents a triangle placement on the grid -/
structure TrianglePlacement :=
  (grid : Grid)
  (covers_all_segments : Prop)

/-- Represents the count of squares without triangles -/
def blank_squares (tp : TrianglePlacement) : ℕ := sorry

/-- The main theorem: maximum number of blank squares in a 100x100 grid -/
theorem max_blank_squares :
  ∀ (tp : TrianglePlacement),
    tp.grid.size = 100 →
    tp.covers_all_segments →
    blank_squares tp ≤ 2450 :=
by sorry

end max_blank_squares_l993_99329


namespace circle_touches_angle_sides_l993_99315

-- Define the angle
def Angle : Type := sorry

-- Define a circle
structure Circle (α : Type) where
  center : α
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the property of a circle touching a line
def touches_line (c : Circle Point) (l : Set Point) : Prop := sorry

-- Define the property of two circles touching each other
def circles_touch (c1 c2 : Circle Point) : Prop := sorry

-- Define the circle with diameter AB
def circle_with_diameter (A B : Point) : Circle Point := sorry

-- Define the sides of an angle
def sides_of_angle (a : Angle) : Set (Set Point) := sorry

theorem circle_touches_angle_sides 
  (θ : Angle) 
  (A B : Point) 
  (c1 c2 : Circle Point) 
  (h1 : c1.center = A)
  (h2 : c2.center = B)
  (h3 : ∀ s ∈ sides_of_angle θ, touches_line c1 s ∧ touches_line c2 s)
  (h4 : circles_touch c1 c2) :
  ∀ s ∈ sides_of_angle θ, touches_line (circle_with_diameter A B) s := by
  sorry

end circle_touches_angle_sides_l993_99315


namespace inequality_property_l993_99385

theorem inequality_property (a b : ℝ) : a > b → -5 * a < -5 * b := by
  sorry

end inequality_property_l993_99385


namespace margin_in_terms_of_selling_price_l993_99364

variables (P S M t n : ℝ)

/-- The margin M can be expressed in terms of the selling price S, given the production cost P, tax rate t, and a constant n. -/
theorem margin_in_terms_of_selling_price
  (h1 : S = P * (1 + t/100))  -- Selling price including tax
  (h2 : M = P / n)            -- Margin definition
  (h3 : n > 0)                -- n is positive (implied by the context)
  (h4 : t ≥ 0)                -- Tax rate is non-negative (implied by the context)
  : M = S / (n * (1 + t/100)) :=
sorry

end margin_in_terms_of_selling_price_l993_99364


namespace troy_computer_purchase_l993_99307

/-- The amount of money Troy needs to buy a new computer -/
def additional_money_needed (new_computer_cost saved_amount old_computer_value : ℕ) : ℕ :=
  new_computer_cost - (saved_amount + old_computer_value)

/-- Theorem stating the amount Troy needs to buy the new computer -/
theorem troy_computer_purchase (new_computer_cost saved_amount old_computer_value : ℕ) 
  (h1 : new_computer_cost = 1200)
  (h2 : saved_amount = 450)
  (h3 : old_computer_value = 150) :
  additional_money_needed new_computer_cost saved_amount old_computer_value = 600 := by
  sorry

#eval additional_money_needed 1200 450 150

end troy_computer_purchase_l993_99307


namespace total_cost_of_shed_is_818_25_l993_99397

/-- Calculate the total cost of constructing a shed given the following conditions:
  * 1000 bricks are needed
  * 30% of bricks are at 50% discount off $0.50 each
  * 40% of bricks are at 20% discount off $0.50 each
  * 30% of bricks are at full price of $0.50 each
  * 5% tax on total cost of bricks
  * Additional building materials cost $200
  * 7% tax on additional building materials
  * Labor fees are $20 per hour for 10 hours
-/
def total_cost_of_shed : ℝ :=
  let total_bricks : ℝ := 1000
  let brick_full_price : ℝ := 0.50
  let discounted_bricks_1 : ℝ := 0.30 * total_bricks
  let discounted_bricks_2 : ℝ := 0.40 * total_bricks
  let full_price_bricks : ℝ := 0.30 * total_bricks
  let discount_1 : ℝ := 0.50
  let discount_2 : ℝ := 0.20
  let brick_tax_rate : ℝ := 0.05
  let additional_materials_cost : ℝ := 200
  let materials_tax_rate : ℝ := 0.07
  let labor_rate : ℝ := 20
  let labor_hours : ℝ := 10

  let discounted_price_1 : ℝ := brick_full_price * (1 - discount_1)
  let discounted_price_2 : ℝ := brick_full_price * (1 - discount_2)
  
  let brick_cost : ℝ := 
    discounted_bricks_1 * discounted_price_1 +
    discounted_bricks_2 * discounted_price_2 +
    full_price_bricks * brick_full_price
  
  let brick_tax : ℝ := brick_cost * brick_tax_rate
  let materials_tax : ℝ := additional_materials_cost * materials_tax_rate
  let labor_cost : ℝ := labor_rate * labor_hours

  brick_cost + brick_tax + additional_materials_cost + materials_tax + labor_cost

theorem total_cost_of_shed_is_818_25 : 
  total_cost_of_shed = 818.25 := by
  sorry

end total_cost_of_shed_is_818_25_l993_99397


namespace floor_sqrt_75_l993_99370

theorem floor_sqrt_75 : ⌊Real.sqrt 75⌋ = 8 := by sorry

end floor_sqrt_75_l993_99370


namespace price_reduction_theorem_l993_99321

/-- Given three consecutive price reductions, calculates the overall percentage reduction -/
def overall_reduction (r1 r2 r3 : ℝ) : ℝ :=
  (1 - (1 - r1) * (1 - r2) * (1 - r3)) * 100

/-- Theorem stating that the overall reduction after 25%, 20%, and 15% reductions is 49% -/
theorem price_reduction_theorem : 
  overall_reduction 0.25 0.20 0.15 = 49 := by
  sorry

#eval overall_reduction 0.25 0.20 0.15

end price_reduction_theorem_l993_99321


namespace forum_posts_per_day_l993_99335

/-- A forum with questions and answers -/
structure Forum where
  members : ℕ
  questionsPerHour : ℕ
  answerRatio : ℕ

/-- Calculate the total posts (questions and answers) per day -/
def totalPostsPerDay (f : Forum) : ℕ :=
  let questionsPerDay := f.members * (f.questionsPerHour * 24)
  let answersPerDay := questionsPerDay * f.answerRatio
  questionsPerDay + answersPerDay

/-- Theorem: Given the conditions, the forum has 57600 posts per day -/
theorem forum_posts_per_day :
  ∀ (f : Forum),
    f.members = 200 →
    f.questionsPerHour = 3 →
    f.answerRatio = 3 →
    totalPostsPerDay f = 57600 := by
  sorry

end forum_posts_per_day_l993_99335


namespace real_part_of_z_l993_99344

theorem real_part_of_z (z : ℂ) (h : (3 + 4*I)*z = 5*(1 - I)) : 
  z.re = -1/5 := by
  sorry

end real_part_of_z_l993_99344


namespace min_value_problem_l993_99381

theorem min_value_problem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) :
  3 * x + 4 * y ≥ 5 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 3 * y = 5 * x * y ∧ 3 * x + 4 * y = 5 := by
  sorry

end min_value_problem_l993_99381


namespace power_two_plus_one_div_by_three_l993_99359

theorem power_two_plus_one_div_by_three (n : ℕ) : 
  3 ∣ (2^n + 1) ↔ n % 2 = 1 := by
sorry

end power_two_plus_one_div_by_three_l993_99359


namespace tangent_line_slope_intercept_difference_l993_99311

/-- A line passing through two points and tangent to a circle -/
structure TangentLine where
  a : ℝ
  b : ℝ
  passes_through_first : 7 = a * 5 + b
  passes_through_second : 20 = a * 9 + b
  tangent_at : (5, 7) ∈ {(x, y) | y = a * x + b}

/-- The difference between the slope and y-intercept of the tangent line -/
def slope_intercept_difference (line : TangentLine) : ℝ := line.a - line.b

/-- Theorem stating that the difference between slope and y-intercept is 12.5 -/
theorem tangent_line_slope_intercept_difference :
  ∀ (line : TangentLine), slope_intercept_difference line = 12.5 := by
  sorry

end tangent_line_slope_intercept_difference_l993_99311


namespace nested_sqrt_evaluation_l993_99331

theorem nested_sqrt_evaluation (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x^2 * Real.sqrt (x^2 * Real.sqrt (x^2))) = x^(7/4) := by
  sorry

end nested_sqrt_evaluation_l993_99331


namespace inequality_proof_l993_99334

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b + 3/4) * (b^2 + c + 3/4) * (c^2 + a + 3/4) ≥ (2*a + 1/2) * (2*b + 1/2) * (2*c + 1/2) := by
  sorry

end inequality_proof_l993_99334


namespace suitable_land_size_l993_99395

def previous_property : ℝ := 2
def land_multiplier : ℝ := 10
def pond_size : ℝ := 1

theorem suitable_land_size :
  let new_property := previous_property * land_multiplier
  let suitable_land := new_property - pond_size
  suitable_land = 19 := by sorry

end suitable_land_size_l993_99395


namespace square_area_not_covered_by_circles_l993_99384

/-- The area of a square not covered by circles -/
theorem square_area_not_covered_by_circles (side_length : ℝ) (num_circles : ℕ) : 
  side_length = 16 → num_circles = 9 → 
  side_length^2 - (num_circles : ℝ) * (side_length / 3)^2 * Real.pi = 256 - 64 * Real.pi := by
  sorry

#check square_area_not_covered_by_circles

end square_area_not_covered_by_circles_l993_99384


namespace pqr_product_l993_99302

theorem pqr_product (p q r : ℝ) (h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p)
  (h_nonzero : p ≠ 0 ∧ q ≠ 0 ∧ r ≠ 0)
  (h_eq : p^2 + 2/q = q^2 + 2/r ∧ q^2 + 2/r = r^2 + 2/p) :
  |p * q * r| = 2 := by
  sorry

end pqr_product_l993_99302


namespace students_per_grade_l993_99351

theorem students_per_grade (total_students : ℕ) (total_grades : ℕ) 
  (h1 : total_students = 22800) 
  (h2 : total_grades = 304) : 
  total_students / total_grades = 75 := by
  sorry

end students_per_grade_l993_99351


namespace issacs_pens_l993_99363

theorem issacs_pens (total : ℕ) (pens : ℕ) (pencils : ℕ) : 
  total = 108 →
  total = pens + pencils →
  pencils = 5 * pens + 12 →
  pens = 16 := by
  sorry

end issacs_pens_l993_99363


namespace calculate_first_train_length_l993_99324

/-- The length of the first train given the specified conditions -/
def first_train_length (first_train_speed second_train_speed : ℝ)
                       (second_train_length : ℝ)
                       (crossing_time : ℝ) : ℝ :=
  (first_train_speed - second_train_speed) * crossing_time - second_train_length

/-- Theorem stating the length of the first train under given conditions -/
theorem calculate_first_train_length :
  first_train_length 72 36 300 69.99440044796417 = 399.9440044796417 := by
  sorry

end calculate_first_train_length_l993_99324


namespace platform_length_l993_99322

/-- The length of a platform given train parameters --/
theorem platform_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : 
  train_length = 250 →
  train_speed_kmh = 55 →
  crossing_time = 35.99712023038157 →
  ∃ (platform_length : ℝ), platform_length = 300 := by
  sorry

#check platform_length

end platform_length_l993_99322


namespace zoo_animals_l993_99312

theorem zoo_animals (lions : ℕ) (penguins : ℕ) : 
  lions = 30 →
  11 * lions = 3 * penguins →
  penguins - lions = 80 := by
  sorry

end zoo_animals_l993_99312


namespace scooter_profit_theorem_l993_99301

def scooter_profit_problem (cost_price : ℝ) : Prop :=
  let repair_cost : ℝ := 500
  let profit : ℝ := 1100
  let selling_price : ℝ := cost_price + profit
  (0.1 * cost_price = repair_cost) ∧
  ((profit / cost_price) * 100 = 22)

theorem scooter_profit_theorem :
  ∃ (cost_price : ℝ), scooter_profit_problem cost_price := by
  sorry

end scooter_profit_theorem_l993_99301


namespace quadratic_range_l993_99323

theorem quadratic_range (x : ℝ) (h : x^2 - 4*x + 3 < 0) :
  8 < x^2 + 4*x + 3 ∧ x^2 + 4*x + 3 < 24 := by
  sorry

end quadratic_range_l993_99323


namespace order_of_numbers_l993_99396

theorem order_of_numbers : 70.3 > 0.37 ∧ 0.37 > Real.log 0.3 := by
  sorry

end order_of_numbers_l993_99396


namespace triangle_law_of_sines_l993_99356

theorem triangle_law_of_sines (A B C : Real) (a b c : Real) :
  A = π / 6 →
  a = Real.sqrt 2 →
  b / Real.sin B = 2 * Real.sqrt 2 :=
by sorry

end triangle_law_of_sines_l993_99356


namespace largest_cube_filling_box_l993_99367

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the greatest common divisor of three natural numbers -/
def gcd3 (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the volume of a cube -/
def cubeVolume (edge : ℕ) : ℕ :=
  edge * edge * edge

/-- The main theorem about the largest cube that can fill the box -/
theorem largest_cube_filling_box (box : BoxDimensions) 
  (h_box : box = { length := 102, width := 255, height := 170 }) :
  let maxEdge := gcd3 box.length box.width box.height
  let numCubes := boxVolume box / cubeVolume maxEdge
  maxEdge = 17 ∧ numCubes = 900 := by sorry

end largest_cube_filling_box_l993_99367


namespace balance_proof_l993_99318

/-- The weight of a single diamond -/
def diamond_weight : ℝ := sorry

/-- The weight of a single emerald -/
def emerald_weight : ℝ := sorry

/-- The number of diamonds that balance one emerald -/
def diamonds_per_emerald : ℕ := sorry

theorem balance_proof :
  -- Condition 1 and 2: 9 diamonds balance 4 emeralds
  9 * diamond_weight = 4 * emerald_weight →
  -- Condition 3: 9 diamonds + 1 emerald balance 4 emeralds
  9 * diamond_weight + emerald_weight = 4 * emerald_weight →
  -- Conclusion: 3 diamonds balance 1 emerald
  diamonds_per_emerald = 3 := by sorry

end balance_proof_l993_99318


namespace tan_expression_equality_l993_99304

theorem tan_expression_equality (θ : Real) (h : Real.tan θ = 3) :
  let k : Real := 1/2
  (1 - k * Real.cos θ) / Real.sin θ - (2 * Real.sin θ) / (1 + Real.cos θ) =
  (20 - Real.sqrt 10) / (3 * Real.sqrt 10) - (6 * Real.sqrt 10) / (10 + Real.sqrt 10) :=
by sorry

end tan_expression_equality_l993_99304


namespace cubic_roots_relation_l993_99313

theorem cubic_roots_relation (a b c r s t : ℝ) : 
  (∀ x, x^3 + 5*x^2 + 6*x - 13 = 0 ↔ x = a ∨ x = b ∨ x = c) →
  (∀ x, x^3 + r*x^2 + s*x + t = 0 ↔ x = a+1 ∨ x = b+1 ∨ x = c+1) →
  t = -15 := by
sorry

end cubic_roots_relation_l993_99313


namespace meeting_point_coordinates_l993_99349

/-- The point two-thirds of the way from one point to another -/
def two_thirds_point (x₁ y₁ x₂ y₂ : ℚ) : ℚ × ℚ :=
  (x₁ + 2/3 * (x₂ - x₁), y₁ + 2/3 * (y₂ - y₁))

/-- Prove that the meeting point is at (14/3, 11/3) -/
theorem meeting_point_coordinates :
  two_thirds_point 10 (-3) 2 7 = (14/3, 11/3) := by
  sorry

#check meeting_point_coordinates

end meeting_point_coordinates_l993_99349


namespace ellipse_condition_l993_99309

-- Define the condition for an ellipse with foci on the x-axis
def is_ellipse_x_axis (m n : ℝ) : Prop :=
  ∃ (a b : ℝ), a > b ∧ a > 0 ∧ b > 0 ∧ m = 1 / (a^2) ∧ n = 1 / (b^2)

-- State the theorem
theorem ellipse_condition (m n : ℝ) :
  is_ellipse_x_axis m n ↔ n > m ∧ m > 0 :=
sorry

end ellipse_condition_l993_99309


namespace inscribed_quadrilateral_triangle_l993_99382

/-- Given a quadrilateral inscribed in a circle of radius R, with points P, Q, and M as described,
    and distances a, b, and c from these points to the circle's center,
    prove that the sides of triangle PQM have the given lengths. -/
theorem inscribed_quadrilateral_triangle (R a b c : ℝ) (h_pos : R > 0) :
  ∃ (PQ QM PM : ℝ),
    PQ = Real.sqrt (a^2 + b^2 - 2*R^2) ∧
    QM = Real.sqrt (b^2 + c^2 - 2*R^2) ∧
    PM = Real.sqrt (c^2 + a^2 - 2*R^2) := by
  sorry

end inscribed_quadrilateral_triangle_l993_99382


namespace certain_event_at_least_one_genuine_l993_99375

theorem certain_event_at_least_one_genuine :
  ∀ (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ),
    total = 12 →
    genuine = 10 →
    defective = 2 →
    total = genuine + defective →
    selected = 3 →
    (∀ outcome : Finset (Fin total),
      outcome.card = selected →
      ∃ i ∈ outcome, i.val < genuine) :=
by sorry

end certain_event_at_least_one_genuine_l993_99375


namespace absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_18_l993_99365

theorem absolute_value_equation_solution_difference : ℝ → Prop :=
  fun difference =>
    ∃ x₁ x₂ : ℝ,
      (|2 * x₁ - 3| = 18) ∧
      (|2 * x₂ - 3| = 18) ∧
      (x₁ ≠ x₂) ∧
      (difference = |x₁ - x₂|) ∧
      (difference = 18)

-- The proof goes here
theorem absolute_value_equation_solution_difference_is_18 :
  absolute_value_equation_solution_difference 18 :=
sorry

end absolute_value_equation_solution_difference_absolute_value_equation_solution_difference_is_18_l993_99365


namespace max_value_ab_l993_99357

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (1 / ((2 * a + b) * b)) + (2 / ((2 * b + a) * a)) = 1) :
  ab ≤ 2 - (2 * Real.sqrt 2) / 3 ∧ 
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 
    (1 / ((2 * a₀ + b₀) * b₀)) + (2 / ((2 * b₀ + a₀) * a₀)) = 1 ∧
    a₀ * b₀ = 2 - (2 * Real.sqrt 2) / 3 :=
sorry

end max_value_ab_l993_99357


namespace terminal_side_quadrant_l993_99305

theorem terminal_side_quadrant (α : Real) :
  let P : ℝ × ℝ := (Real.sin 2, Real.cos 2)
  (∃ k : ℝ, k > 0 ∧ P = (k * Real.sin α, k * Real.cos α)) →
  Real.sin α > 0 ∧ Real.cos α < 0 :=
by
  sorry

end terminal_side_quadrant_l993_99305


namespace vector_norm_equation_solutions_l993_99345

theorem vector_norm_equation_solutions :
  let v : ℝ × ℝ := (3, -4)
  let w : ℝ × ℝ := (5, 8)
  let norm_eq : ℝ → Prop := λ k => ‖k • v - w‖ = 5 * Real.sqrt 13
  ∀ k : ℝ, norm_eq k ↔ (k = 123 / 50 ∨ k = -191 / 50) :=
by sorry

end vector_norm_equation_solutions_l993_99345


namespace ship_journey_theorem_l993_99347

/-- A ship's journey over three days -/
structure ShipJourney where
  day1_distance : ℝ
  day2_multiplier : ℝ
  day3_additional : ℝ
  total_distance : ℝ

/-- The solution to the ship's journey problem -/
def ship_journey_solution (j : ShipJourney) : Prop :=
  j.day1_distance = 100 ∧
  j.day2_multiplier = 3 ∧
  j.total_distance = 810 ∧
  j.total_distance = j.day1_distance + (j.day2_multiplier * j.day1_distance) + 
                     (j.day2_multiplier * j.day1_distance + j.day3_additional) ∧
  j.day3_additional = 110

/-- Theorem stating the solution to the ship's journey problem -/
theorem ship_journey_theorem (j : ShipJourney) :
  ship_journey_solution j → j.day3_additional = 110 :=
by
  sorry


end ship_journey_theorem_l993_99347


namespace minas_age_l993_99362

/-- Given the ages of Minho, Suhong, and Mina, prove that Mina is 10 years old -/
theorem minas_age (suhong minho mina : ℕ) : 
  minho = 3 * suhong →  -- Minho's age is three times Suhong's age
  mina = 2 * suhong - 2 →  -- Mina's age is two years younger than twice Suhong's age
  suhong + minho + mina = 34 →  -- The sum of the ages of the three is 34
  mina = 10 := by
sorry


end minas_age_l993_99362


namespace least_lcm_ac_l993_99320

theorem least_lcm_ac (a b c : ℕ+) (h1 : Nat.lcm a b = 18) (h2 : Nat.lcm b c = 20) :
  ∃ (a' c' : ℕ+), Nat.lcm a' b = 18 ∧ Nat.lcm b c' = 20 ∧ 
    Nat.lcm a' c' = 90 ∧ ∀ (x y : ℕ+), Nat.lcm x b = 18 → Nat.lcm b y = 20 → 
      Nat.lcm x y ≥ 90 := by
sorry

end least_lcm_ac_l993_99320


namespace remainder_problem_l993_99355

theorem remainder_problem (k : Nat) (h : k > 0) :
  (90 % (k^2) = 6) → (130 % k = 4) := by
  sorry

end remainder_problem_l993_99355


namespace symmetric_point_to_origin_l993_99366

/-- Given a point M with coordinates (-3, -5), proves that the coordinates of the point symmetric to M with respect to the origin are (3, 5). -/
theorem symmetric_point_to_origin (M : ℝ × ℝ) (h : M = (-3, -5)) :
  (- M.1, - M.2) = (3, 5) := by
  sorry

end symmetric_point_to_origin_l993_99366


namespace deaf_to_blind_ratio_l993_99376

theorem deaf_to_blind_ratio (total_students blind_students : ℕ) 
  (h1 : total_students = 180)
  (h2 : blind_students = 45) :
  (total_students - blind_students) / blind_students = 3 := by
  sorry

end deaf_to_blind_ratio_l993_99376


namespace min_value_fraction_l993_99343

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (4 / a + 9 / b) ≥ 25 := by
  sorry

end min_value_fraction_l993_99343


namespace hannah_shopping_cost_hannah_spent_65_dollars_l993_99317

theorem hannah_shopping_cost : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun num_sweatshirts sweatshirt_cost num_tshirts tshirt_cost total_cost =>
    num_sweatshirts * sweatshirt_cost + num_tshirts * tshirt_cost = total_cost

theorem hannah_spent_65_dollars :
  hannah_shopping_cost 3 15 2 10 65 := by
  sorry

end hannah_shopping_cost_hannah_spent_65_dollars_l993_99317


namespace fraction_equality_l993_99342

theorem fraction_equality : (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) := by
  sorry

end fraction_equality_l993_99342


namespace box_minus_two_zero_three_l993_99391

def box (a b c : ℤ) : ℚ := a^b - b^c + c^a

theorem box_minus_two_zero_three : box (-2) 0 3 = 10/9 := by
  sorry

end box_minus_two_zero_three_l993_99391


namespace cube_volume_from_surface_area_l993_99398

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 864 →
  (∃ (side_length : ℝ), 
    surface_area = 6 * side_length^2 ∧
    volume = side_length^3) →
  volume = 1728 := by
sorry

end cube_volume_from_surface_area_l993_99398


namespace sufficient_not_necessary_l993_99394

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := (a^2 - 4 : ℝ) + (a + 1 : ℝ) * Complex.I

-- Define what it means for a complex number to be purely imaginary
def is_purely_imaginary (w : ℂ) : Prop := w.re = 0 ∧ w.im ≠ 0

-- State the theorem
theorem sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ -2 ∧ is_purely_imaginary (z a)) ∧
  (∀ (a : ℝ), a = -2 → is_purely_imaginary (z a)) :=
sorry

end sufficient_not_necessary_l993_99394
