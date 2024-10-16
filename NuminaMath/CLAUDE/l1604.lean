import Mathlib

namespace NUMINAMATH_CALUDE_bug_return_probability_l1604_160436

/-- Probability of the bug being at the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1/3 * (1 - P n)

/-- The problem statement -/
theorem bug_return_probability : P 8 = 547/2187 := by
  sorry

end NUMINAMATH_CALUDE_bug_return_probability_l1604_160436


namespace NUMINAMATH_CALUDE_lottery_theorem_l1604_160413

-- Define the lottery setup
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3

-- Define the probability of drawing a red ball first given a white ball second
def prob_red_given_white : ℚ := 5/11

-- Define the probabilities for the distribution of red balls drawn
def prob_zero_red : ℚ := 27/125
def prob_one_red : ℚ := 549/1000
def prob_two_red : ℚ := 47/200

-- Define the expected number of red balls drawn
def expected_red_balls : ℚ := 1019/1000

-- Theorem statement
theorem lottery_theorem :
  (total_balls = red_balls + white_balls) →
  (prob_red_given_white = 5/11) ∧
  (prob_zero_red + prob_one_red + prob_two_red = 1) ∧
  (expected_red_balls = 0 * prob_zero_red + 1 * prob_one_red + 2 * prob_two_red) :=
by sorry

end NUMINAMATH_CALUDE_lottery_theorem_l1604_160413


namespace NUMINAMATH_CALUDE_solve_for_y_l1604_160401

theorem solve_for_y (x y : ℝ) (h1 : 3 * x^2 = y - 6) (h2 : x = 4) : y = 54 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l1604_160401


namespace NUMINAMATH_CALUDE_range_of_a_l1604_160433

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1604_160433


namespace NUMINAMATH_CALUDE_symmetry_sine_cosine_function_l1604_160491

/-- Given a function f(x) = a*sin(x) + b*cos(x) where ab ≠ 0, 
    if the graph of f(x) is symmetric about x = π/6 and f(x₀) = 8/5 * a, 
    then sin(2x₀ + π/6) = 7/25 -/
theorem symmetry_sine_cosine_function 
  (a b x₀ : ℝ) 
  (h1 : a * b ≠ 0) 
  (f : ℝ → ℝ) 
  (h2 : ∀ x, f x = a * Real.sin x + b * Real.cos x) 
  (h3 : ∀ x, f (π/3 - x) = f (π/3 + x)) 
  (h4 : f x₀ = 8/5 * a) : 
  Real.sin (2*x₀ + π/6) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sine_cosine_function_l1604_160491


namespace NUMINAMATH_CALUDE_log_inequality_l1604_160450

theorem log_inequality (x : ℝ) :
  (Real.log x / Real.log (1/2) - Real.sqrt (2 - Real.log x / Real.log 4) + 1 ≤ 0) ↔
  (1 / Real.sqrt 2 ≤ x ∧ x ≤ 16) :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_l1604_160450


namespace NUMINAMATH_CALUDE_actual_annual_yield_actual_annual_yield_approx_l1604_160409

/-- Calculates the actual annual yield for a one-year term deposit with varying interest rates and a closing fee. -/
theorem actual_annual_yield (P : ℝ) : ℝ :=
  let first_quarter_rate := 0.12 / 4
  let second_quarter_rate := 0.08 / 4
  let third_semester_rate := 0.06 / 2
  let closing_fee_rate := 0.01
  let final_amount := P * (1 + first_quarter_rate) * (1 + second_quarter_rate) * (1 + third_semester_rate)
  let effective_final_amount := final_amount - (P * closing_fee_rate)
  (effective_final_amount / P) - 1

/-- The actual annual yield is approximately 7.2118% -/
theorem actual_annual_yield_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.00001 ∧ ∀ (P : ℝ), P > 0 → |actual_annual_yield P - 0.072118| < ε :=
sorry

end NUMINAMATH_CALUDE_actual_annual_yield_actual_annual_yield_approx_l1604_160409


namespace NUMINAMATH_CALUDE_expression_evaluation_l1604_160408

theorem expression_evaluation (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h : x = 1 / z) :
  (x^3 - 1/x^3) * (z^3 + 1/z^3) = x^6 - 1/x^6 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1604_160408


namespace NUMINAMATH_CALUDE_sum_and_multiple_l1604_160476

theorem sum_and_multiple : ∃ (a b c : ℕ), 
  (∀ n : ℕ, n < 100 → n ≤ a) ∧ 
  (a < 100) ∧
  (b > 300) ∧ 
  (∀ n : ℕ, n > 300 → n ≥ b) ∧
  (∀ n : ℕ, n ≤ 200 → n ≤ c) ∧ 
  (c ≤ 200) ∧
  (a + b = 2 * c) := by
sorry

end NUMINAMATH_CALUDE_sum_and_multiple_l1604_160476


namespace NUMINAMATH_CALUDE_scientific_notation_of_161000_l1604_160420

/-- The scientific notation representation of 161,000 -/
theorem scientific_notation_of_161000 : 161000 = 1.61 * (10 ^ 5) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_161000_l1604_160420


namespace NUMINAMATH_CALUDE_subtract_negative_l1604_160457

theorem subtract_negative (a b : ℝ) : a - (-b) = a + b := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l1604_160457


namespace NUMINAMATH_CALUDE_kite_diagonal_sum_less_than_largest_sides_sum_l1604_160421

/-- A kite is a quadrilateral with two pairs of adjacent sides of equal length -/
structure Kite where
  sides : Fin 4 → ℝ
  diagonals : Fin 2 → ℝ
  side_positive : ∀ i, sides i > 0
  diagonal_positive : ∀ i, diagonals i > 0
  adjacent_equal : sides 0 = sides 1 ∧ sides 2 = sides 3

theorem kite_diagonal_sum_less_than_largest_sides_sum (k : Kite) :
  k.diagonals 0 + k.diagonals 1 < 
  (max (k.sides 0) (k.sides 2)) + (max (k.sides 1) (k.sides 3)) + 
  (min (max (k.sides 0) (k.sides 2)) (max (k.sides 1) (k.sides 3))) :=
sorry

end NUMINAMATH_CALUDE_kite_diagonal_sum_less_than_largest_sides_sum_l1604_160421


namespace NUMINAMATH_CALUDE_equation_solution_l1604_160471

theorem equation_solution :
  ∃ (y₁ y₂ : ℝ), 
    (4 * (-1)^2 + 3 * y₁^2 + 8 * (-1) - 6 * y₁ + 30 = 50) ∧
    (4 * (-1)^2 + 3 * y₂^2 + 8 * (-1) - 6 * y₂ + 30 = 50) ∧
    (y₁ = 1 + Real.sqrt (29/3)) ∧
    (y₂ = 1 - Real.sqrt (29/3)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1604_160471


namespace NUMINAMATH_CALUDE_square_perimeter_l1604_160444

theorem square_perimeter (area : ℝ) (side : ℝ) : 
  area = 400 ∧ area = side * side → 4 * side = 80 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l1604_160444


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1604_160414

def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (0, 1)

theorem perpendicular_vectors :
  let v1 := (2 * a.1 + b.1, 2 * a.2 + b.2)
  let v2 := (a.1 - 2 * b.1, a.2 - 2 * b.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1604_160414


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1604_160411

/-- Two vectors in ℝ² are perpendicular if their dot product is zero -/
def perpendicular (a b : ℝ × ℝ) : Prop := a.1 * b.1 + a.2 * b.2 = 0

/-- Given two vectors a and b in ℝ², where a = (1,3) and b = (x,-1),
    if a is perpendicular to b, then x = 3 -/
theorem perpendicular_vectors (x : ℝ) : 
  perpendicular (1, 3) (x, -1) → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1604_160411


namespace NUMINAMATH_CALUDE_outfits_from_five_shirts_three_pants_l1604_160469

/-- The number of outfits that can be made from a given number of shirts and pants -/
def number_of_outfits (shirts : ℕ) (pants : ℕ) : ℕ := shirts * pants

/-- Theorem: Given 5 shirts and 3 pairs of pants, the number of outfits is 15 -/
theorem outfits_from_five_shirts_three_pants : 
  number_of_outfits 5 3 = 15 := by
  sorry

end NUMINAMATH_CALUDE_outfits_from_five_shirts_three_pants_l1604_160469


namespace NUMINAMATH_CALUDE_chad_bbq_ice_cost_l1604_160454

/-- The cost of ice for Chad's BBQ --/
def ice_cost (people : ℕ) (ice_per_person : ℕ) (pack_size : ℕ) (cost_per_pack : ℚ) : ℚ :=
  let total_ice := people * ice_per_person
  let packs_needed := (total_ice + pack_size - 1) / pack_size  -- Ceiling division
  packs_needed * cost_per_pack

/-- Theorem stating the cost of ice for Chad's BBQ --/
theorem chad_bbq_ice_cost :
  ice_cost 15 2 10 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_chad_bbq_ice_cost_l1604_160454


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l1604_160461

-- Define the lines
def line1 (x : ℝ) : ℝ := 3 * x - 3
def line2 (x : ℝ) : ℝ := -2 * x + 14
def line3 : ℝ := 0
def line4 : ℝ := 5

-- Define the vertices of the quadrilateral
def vertex1 : ℝ × ℝ := (0, line1 0)
def vertex2 : ℝ × ℝ := (0, line2 0)
def vertex3 : ℝ × ℝ := (line4, line1 line4)
def vertex4 : ℝ × ℝ := (line4, line2 line4)

-- Define the area of the quadrilateral
def quadrilateralArea : ℝ := 80

-- Theorem statement
theorem area_of_quadrilateral :
  let vertices := [vertex1, vertex2, vertex3, vertex4]
  quadrilateralArea = 80 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l1604_160461


namespace NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l1604_160479

/-- The probability of drawing n-1 white marbles followed by a red marble -/
def P (n : ℕ) : ℚ := 1 / (n * (n + 1))

/-- The number of boxes -/
def num_boxes : ℕ := 3000

theorem smallest_n_for_P_less_than_threshold :
  (∀ k < 55, P k ≥ 1 / num_boxes) ∧
  P 55 < 1 / num_boxes :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_P_less_than_threshold_l1604_160479


namespace NUMINAMATH_CALUDE_circle_area_tripled_l1604_160442

theorem circle_area_tripled (r m : ℝ) : 
  (r > 0) → (m > 0) → (π * (r + m)^2 = 3 * π * r^2) → (r = m * (Real.sqrt 3 - 1) / 2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l1604_160442


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1604_160440

theorem quadratic_inequality_solution_set :
  {x : ℝ | x * (x - 3) < 0} = {x : ℝ | 0 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1604_160440


namespace NUMINAMATH_CALUDE_binomial_variance_three_fourths_l1604_160438

def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_three_fourths (p : ℝ) 
  (h1 : 0 ≤ p) (h2 : p ≤ 1) 
  (h3 : binomial_variance 3 p = 3/4) : p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_variance_three_fourths_l1604_160438


namespace NUMINAMATH_CALUDE_triangle_legs_theorem_l1604_160456

/-- A point inside a right angle -/
structure PointInRightAngle where
  /-- Distance from the point to one side of the angle -/
  dist1 : ℝ
  /-- Distance from the point to the other side of the angle -/
  dist2 : ℝ

/-- A triangle formed by a line through a point in a right angle -/
structure TriangleInRightAngle where
  /-- The point inside the right angle -/
  point : PointInRightAngle
  /-- The area of the triangle -/
  area : ℝ

/-- The legs of a right triangle -/
structure RightTriangleLegs where
  /-- Length of one leg -/
  leg1 : ℝ
  /-- Length of the other leg -/
  leg2 : ℝ

/-- Theorem about the legs of a specific triangle in a right angle -/
theorem triangle_legs_theorem (t : TriangleInRightAngle)
    (h1 : t.point.dist1 = 4)
    (h2 : t.point.dist2 = 8)
    (h3 : t.area = 100) :
    (∃ l : RightTriangleLegs, (l.leg1 = 40 ∧ l.leg2 = 5) ∨ (l.leg1 = 10 ∧ l.leg2 = 20)) :=
  sorry

end NUMINAMATH_CALUDE_triangle_legs_theorem_l1604_160456


namespace NUMINAMATH_CALUDE_second_player_winning_strategy_l1604_160496

/-- Represents the n-gon coloring game -/
structure ColoringGame where
  n : ℕ  -- number of sides in the n-gon

/-- Defines when the second player has a winning strategy -/
def second_player_wins (game : ColoringGame) : Prop :=
  ∃ k : ℕ, game.n = 4 + 3 * k

/-- Theorem: The second player has a winning strategy if and only if n = 4 + 3k, where k ≥ 0 -/
theorem second_player_winning_strategy (game : ColoringGame) :
  second_player_wins game ↔ ∃ k : ℕ, game.n = 4 + 3 * k :=
by sorry

end NUMINAMATH_CALUDE_second_player_winning_strategy_l1604_160496


namespace NUMINAMATH_CALUDE_intersection_line_slope_l1604_160407

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 2*y + 4 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧
  circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧
  C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) (h : intersection C D) :
  (D.2 - C.2) / (D.1 - C.1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l1604_160407


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1604_160473

theorem sufficient_not_necessary (m n : ℝ) :
  (∀ m n : ℝ, m / n - 1 = 0 → m - n = 0) ∧
  (∃ m n : ℝ, m - n = 0 ∧ ¬(m / n - 1 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1604_160473


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1604_160466

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) :
  -x₁^2 + 2*x₁ + 4 = 0 ∧ -x₂^2 + 2*x₂ + 4 = 0 → x₁ + x₂ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1604_160466


namespace NUMINAMATH_CALUDE_system_solution_transformation_l1604_160468

theorem system_solution_transformation (a₁ a₂ c₁ c₂ : ℝ) :
  (a₁ * 2 + 3 = c₁ ∧ a₂ * 2 + 3 = c₂) →
  (a₁ * (-1) + (-3) = a₁ - c₁ ∧ a₂ * (-1) + (-3) = a₂ - c₂) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_transformation_l1604_160468


namespace NUMINAMATH_CALUDE_max_a_value_l1604_160470

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - a * x^2 + 3

theorem max_a_value (a : ℝ) :
  (∃ m n : ℝ, m ∈ Set.Icc 1 5 ∧ n ∈ Set.Icc 1 5 ∧ n - m ≥ 2 ∧ f a m = f a n) →
  a ≤ Real.log 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l1604_160470


namespace NUMINAMATH_CALUDE_fibonacci_product_theorem_l1604_160437

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The property we want to prove -/
def satisfies_property (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0 ∧ fib m * fib n = m * n

/-- The theorem statement -/
theorem fibonacci_product_theorem :
  ∀ m n : ℕ, satisfies_property m n ↔ (m = 1 ∧ n = 1) ∨ (m = 5 ∧ n = 5) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_product_theorem_l1604_160437


namespace NUMINAMATH_CALUDE_f_monotone_condition_l1604_160424

-- Define the piecewise function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - 2 else Real.log (|x - m|)

-- Define monotonically increasing property
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

-- State the theorem
theorem f_monotone_condition (m : ℝ) :
  (∀ x y, 0 ≤ x ∧ x < y → f m x ≤ f m y) ↔ m ≤ 9/10 :=
sorry

end NUMINAMATH_CALUDE_f_monotone_condition_l1604_160424


namespace NUMINAMATH_CALUDE_all_events_probability_at_least_one_not_occurring_l1604_160445

-- Define the probabilities of each event
def P_A : ℝ := 0.8
def P_B : ℝ := 0.6
def P_C : ℝ := 0.5

-- Theorem for the probability of all three events occurring
theorem all_events_probability :
  P_A * P_B * P_C = 0.24 :=
sorry

-- Theorem for the probability of at least one event not occurring
theorem at_least_one_not_occurring :
  1 - (P_A * P_B * P_C) = 0.76 :=
sorry

end NUMINAMATH_CALUDE_all_events_probability_at_least_one_not_occurring_l1604_160445


namespace NUMINAMATH_CALUDE_store_revenue_l1604_160434

theorem store_revenue (december : ℝ) (h1 : december > 0) : 
  let november := (2 / 5 : ℝ) * december
  let january := (1 / 3 : ℝ) * november
  let average := (november + january) / 2
  december / average = 15 / 4 := by
sorry

end NUMINAMATH_CALUDE_store_revenue_l1604_160434


namespace NUMINAMATH_CALUDE_inequality_implication_l1604_160487

theorem inequality_implication (a b c : ℝ) (h : a < b) : -a * c^2 ≥ -b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1604_160487


namespace NUMINAMATH_CALUDE_garden_diameter_l1604_160467

/-- The diameter of a circular ground given the area of a surrounding garden -/
theorem garden_diameter (garden_width : ℝ) (garden_area : ℝ) (diameter : ℝ) : 
  garden_width = 2 →
  garden_area = 226.19467105846502 →
  diameter = 34 →
  let radius := diameter / 2
  let outer_radius := radius + garden_width
  garden_area = π * outer_radius^2 - π * radius^2 :=
by sorry

end NUMINAMATH_CALUDE_garden_diameter_l1604_160467


namespace NUMINAMATH_CALUDE_parallel_sides_implies_parallelogram_equal_sides_implies_parallelogram_one_pair_parallel_equal_implies_parallelogram_equal_diagonals_implies_parallelogram_l1604_160426

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define the conditions
def opposite_sides_parallel (q : Quadrilateral) : Prop := sorry
def opposite_sides_equal (q : Quadrilateral) : Prop := sorry
def one_pair_parallel_and_equal (q : Quadrilateral) : Prop := sorry
def diagonals_equal (q : Quadrilateral) : Prop := sorry

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Theorem statements
theorem parallel_sides_implies_parallelogram (q : Quadrilateral) :
  opposite_sides_parallel q → is_parallelogram q := by sorry

theorem equal_sides_implies_parallelogram (q : Quadrilateral) :
  opposite_sides_equal q → is_parallelogram q := by sorry

theorem one_pair_parallel_equal_implies_parallelogram (q : Quadrilateral) :
  one_pair_parallel_and_equal q → is_parallelogram q := by sorry

theorem equal_diagonals_implies_parallelogram (q : Quadrilateral) :
  diagonals_equal q → is_parallelogram q := by sorry

end NUMINAMATH_CALUDE_parallel_sides_implies_parallelogram_equal_sides_implies_parallelogram_one_pair_parallel_equal_implies_parallelogram_equal_diagonals_implies_parallelogram_l1604_160426


namespace NUMINAMATH_CALUDE_parallelogram_area_24_16_l1604_160455

/-- The area of a parallelogram with given base and height -/
def parallelogram_area (base height : ℝ) : ℝ := base * height

/-- Theorem: The area of a parallelogram with base 24 cm and height 16 cm is 384 square centimeters -/
theorem parallelogram_area_24_16 : parallelogram_area 24 16 = 384 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_24_16_l1604_160455


namespace NUMINAMATH_CALUDE_lcm_18_42_l1604_160474

theorem lcm_18_42 : Nat.lcm 18 42 = 126 := by sorry

end NUMINAMATH_CALUDE_lcm_18_42_l1604_160474


namespace NUMINAMATH_CALUDE_cos_shift_odd_condition_l1604_160419

open Real

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem cos_shift_odd_condition (φ : ℝ) :
  (φ = π / 2 → is_odd_function (λ x => cos (x + φ))) ∧
  (∃ φ', φ' ≠ π / 2 ∧ is_odd_function (λ x => cos (x + φ'))) :=
sorry

end NUMINAMATH_CALUDE_cos_shift_odd_condition_l1604_160419


namespace NUMINAMATH_CALUDE_factorial_ratio_l1604_160451

theorem factorial_ratio : Nat.factorial 45 / Nat.factorial 42 = 85140 := by sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1604_160451


namespace NUMINAMATH_CALUDE_optimal_coloring_for_max_polygons_l1604_160477

/-- The number of points on the circle -/
def total_points : ℕ := 1996

/-- The optimal number of colors -/
def optimal_colors : ℕ := 61

/-- The sequence of point counts for each color -/
def color_counts : List ℕ := List.range optimal_colors |>.map (· + 2)

/-- Theorem stating the optimal coloring for maximum inscribed polygons -/
theorem optimal_coloring_for_max_polygons :
  (color_counts.sum = total_points) ∧
  (color_counts.length = optimal_colors) ∧
  (color_counts.Nodup) ∧
  (∀ other_coloring : List ℕ,
    other_coloring.sum = total_points →
    other_coloring.Nodup →
    other_coloring.length ≤ optimal_colors) := by
  sorry


end NUMINAMATH_CALUDE_optimal_coloring_for_max_polygons_l1604_160477


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1604_160485

theorem quadratic_rewrite (d e f : ℤ) :
  (∀ x : ℝ, 4 * x^2 - 16 * x + 2 = (d * x + e)^2 + f) →
  d * e = -8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1604_160485


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l1604_160497

theorem trigonometric_equation_solution (x : ℝ) : 
  3 - 7 * (Real.cos x)^2 * Real.sin x - 3 * (Real.sin x)^3 = 0 ↔ 
  (∃ k : ℤ, x = π / 2 + 2 * k * π) ∨ 
  (∃ k : ℤ, x = (-1)^k * π / 6 + k * π) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l1604_160497


namespace NUMINAMATH_CALUDE_casket_inscription_proof_l1604_160464

/-- Represents a craftsman who can make caskets -/
inductive Craftsman
| Bellini
| Cellini
| CelliniSon

/-- Represents a casket with an inscription -/
structure Casket where
  maker : Craftsman
  inscription : String

/-- Determines if an inscription is true for a pair of caskets -/
def isInscriptionTrue (c1 c2 : Casket) (inscription : String) : Prop :=
  match inscription with
  | "At least one of these boxes was made by Cellini's son" =>
    c1.maker = Craftsman.CelliniSon ∨ c2.maker = Craftsman.CelliniSon
  | _ => False

/-- Cellini's son never engraves true statements -/
axiom celliniSonFalsity (c : Casket) :
  c.maker = Craftsman.CelliniSon → ¬(isInscriptionTrue c c c.inscription)

/-- The inscription that solves the problem -/
def problemInscription : String :=
  "At least one of these boxes was made by Cellini's son"

theorem casket_inscription_proof :
  ∃ (c1 c2 : Casket),
    (c1.inscription = problemInscription) ∧
    (c2.inscription = problemInscription) ∧
    (c1.maker = c2.maker) ∧
    (c1.maker = Craftsman.Bellini ∨ c1.maker = Craftsman.Cellini) ∧
    (¬∃ (c : Casket), c.inscription = problemInscription →
      c.maker = Craftsman.Bellini ∨ c.maker = Craftsman.Cellini) ∧
    (∀ (c : Casket), c.inscription = problemInscription →
      ¬(c.maker = Craftsman.Bellini ∨ c.maker = Craftsman.Cellini)) :=
by
  sorry


end NUMINAMATH_CALUDE_casket_inscription_proof_l1604_160464


namespace NUMINAMATH_CALUDE_projection_composition_l1604_160447

open Matrix

/-- The matrix that projects a vector onto (4, 2) -/
def proj_matrix_1 : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4/5, 2/5; 2/5, 1/5]

/-- The matrix that projects a vector onto (2, 1) -/
def proj_matrix_2 : Matrix (Fin 2) (Fin 2) ℚ :=
  !![4/5, 2/5; 2/5, 1/5]

/-- The theorem stating that the composition of the two projection matrices
    results in the same matrix -/
theorem projection_composition :
  proj_matrix_2 * proj_matrix_1 = !![4/5, 2/5; 2/5, 1/5] := by sorry

end NUMINAMATH_CALUDE_projection_composition_l1604_160447


namespace NUMINAMATH_CALUDE_maximal_cross_section_area_l1604_160499

/-- A triangular prism with vertical edges parallel to the z-axis -/
structure TriangularPrism where
  base : Set (ℝ × ℝ)
  height : ℝ → ℝ

/-- The cross-section of the prism is an equilateral triangle with side length 8 -/
def equilateralBase (p : TriangularPrism) : Prop :=
  ∃ (A B C : ℝ × ℝ),
    A ∈ p.base ∧ B ∈ p.base ∧ C ∈ p.base ∧
    dist A B = 8 ∧ dist B C = 8 ∧ dist C A = 8

/-- The plane that intersects the prism -/
def intersectingPlane (x y z : ℝ) : Prop :=
  3 * x - 5 * y + 2 * z = 30

/-- The cross-section formed by the intersection of the prism and the plane -/
def crossSection (p : TriangularPrism) : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | (x, y) ∈ p.base ∧ z = p.height x ∧ intersectingPlane x y z}

/-- The area of the cross-section -/
noncomputable def crossSectionArea (p : TriangularPrism) : ℝ :=
  sorry

/-- The main theorem stating that the maximal area of the cross-section is 92 -/
theorem maximal_cross_section_area (p : TriangularPrism) 
  (h : equilateralBase p) : 
  crossSectionArea p ≤ 92 ∧ ∃ (p' : TriangularPrism), equilateralBase p' ∧ crossSectionArea p' = 92 :=
sorry

end NUMINAMATH_CALUDE_maximal_cross_section_area_l1604_160499


namespace NUMINAMATH_CALUDE_floor_sum_example_l1604_160412

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l1604_160412


namespace NUMINAMATH_CALUDE_cookie_area_theorem_l1604_160478

/-- Represents a rectangular cookie with length and width -/
structure Cookie where
  length : ℝ
  width : ℝ

/-- Calculates the area of a cookie -/
def Cookie.area (c : Cookie) : ℝ := c.length * c.width

/-- Calculates the circumference of two cookies placed horizontally -/
def combined_circumference (c : Cookie) : ℝ := 2 * (2 * c.length + c.width)

theorem cookie_area_theorem (c : Cookie) 
  (h1 : combined_circumference c = 70)
  (h2 : c.width = 15) : 
  c.area = 150 := by
  sorry

end NUMINAMATH_CALUDE_cookie_area_theorem_l1604_160478


namespace NUMINAMATH_CALUDE_product_a4b4_l1604_160429

theorem product_a4b4 (a₁ a₂ a₃ a₄ b₁ b₂ b₃ b₄ : ℝ) 
  (eq1 : a₁ * b₁ + a₂ * b₃ = 1)
  (eq2 : a₁ * b₂ + a₂ * b₄ = 0)
  (eq3 : a₃ * b₁ + a₄ * b₃ = 0)
  (eq4 : a₃ * b₂ + a₄ * b₄ = 1)
  (eq5 : a₂ * b₃ = 7) :
  a₄ * b₄ = -6 := by sorry

end NUMINAMATH_CALUDE_product_a4b4_l1604_160429


namespace NUMINAMATH_CALUDE_compound_interest_time_calculation_l1604_160431

/-- Proves that the time t satisfies the compound interest equation for the given problem --/
theorem compound_interest_time_calculation 
  (initial_investment : ℝ) 
  (annual_rate : ℝ) 
  (compounding_frequency : ℝ) 
  (final_amount : ℝ) 
  (h1 : initial_investment = 600)
  (h2 : annual_rate = 0.10)
  (h3 : compounding_frequency = 2)
  (h4 : final_amount = 661.5) :
  ∃ t : ℝ, final_amount = initial_investment * (1 + annual_rate / compounding_frequency) ^ (compounding_frequency * t) :=
sorry

end NUMINAMATH_CALUDE_compound_interest_time_calculation_l1604_160431


namespace NUMINAMATH_CALUDE_equal_area_rectangles_l1604_160483

/-- Given two rectangles with equal area, where one rectangle measures 12 inches by 15 inches
    and the other has a length of 9 inches, prove that the width of the second rectangle is 20 inches. -/
theorem equal_area_rectangles (carol_length carol_width jordan_length jordan_width : ℝ)
    (h1 : carol_length = 12)
    (h2 : carol_width = 15)
    (h3 : jordan_length = 9)
    (h4 : carol_length * carol_width = jordan_length * jordan_width) :
    jordan_width = 20 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_l1604_160483


namespace NUMINAMATH_CALUDE_compound_has_six_hydrogen_atoms_l1604_160495

/-- The atomic weight of carbon in g/mol -/
def carbon_weight : ℝ := 12.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The atomic weight of hydrogen in g/mol -/
def hydrogen_weight : ℝ := 1.008

/-- The total molecular weight of the compound in g/mol -/
def total_weight : ℝ := 122

/-- The number of carbon atoms in the compound -/
def carbon_count : ℕ := 7

/-- The number of oxygen atoms in the compound -/
def oxygen_count : ℕ := 2

/-- Calculate the molecular weight of the compound given the number of hydrogen atoms -/
def molecular_weight (hydrogen_count : ℕ) : ℝ :=
  carbon_weight * carbon_count + oxygen_weight * oxygen_count + hydrogen_weight * hydrogen_count

/-- Theorem stating that the compound has 6 hydrogen atoms -/
theorem compound_has_six_hydrogen_atoms :
  ∃ (n : ℕ), molecular_weight n = total_weight ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_compound_has_six_hydrogen_atoms_l1604_160495


namespace NUMINAMATH_CALUDE_different_color_probability_l1604_160484

/-- The probability of drawing two balls of different colors from a bag -/
theorem different_color_probability : 
  let total_balls : ℕ := 4
  let red_balls : ℕ := 2
  let yellow_balls : ℕ := 2
  let drawn_balls : ℕ := 2
  let total_ways := Nat.choose total_balls drawn_balls
  let different_color_ways := red_balls * yellow_balls
  (different_color_ways : ℚ) / total_ways = 2/3 := by sorry

end NUMINAMATH_CALUDE_different_color_probability_l1604_160484


namespace NUMINAMATH_CALUDE_stock_trading_profit_l1604_160422

/-- Represents the stock trading scenario described in the problem -/
def stock_trading (initial_investment : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) (final_sale_rate : ℝ) : ℝ :=
  let first_sale := initial_investment * (1 + profit_rate)
  let second_sale := first_sale * (1 - loss_rate)
  let third_sale := second_sale * final_sale_rate
  let first_profit := first_sale - initial_investment
  let final_loss := second_sale - third_sale
  first_profit - final_loss

/-- Theorem stating that given the conditions in the problem, A's overall profit is 10 yuan -/
theorem stock_trading_profit :
  stock_trading 10000 0.1 0.1 0.9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stock_trading_profit_l1604_160422


namespace NUMINAMATH_CALUDE_f_pi_half_value_l1604_160423

theorem f_pi_half_value : 
  let f : ℝ → ℝ := fun x ↦ x * Real.sin x + Real.cos x
  f (Real.pi / 2) = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_f_pi_half_value_l1604_160423


namespace NUMINAMATH_CALUDE_no_prime_square_product_l1604_160406

theorem no_prime_square_product (p q r : Nat) (hp : Prime p) (hq : Prime q) (hr : Prime r) :
  ¬∃ n : Nat, (p^2 + p) * (q^2 + q) * (r^2 + r) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_square_product_l1604_160406


namespace NUMINAMATH_CALUDE_compound_composition_l1604_160462

/-- Atomic weight of Carbon in g/mol -/
def atomic_weight_C : ℝ := 12.01

/-- Atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- Atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- Number of Carbon atoms in the compound -/
def num_C : ℕ := 6

/-- Number of Oxygen atoms in the compound -/
def num_O : ℕ := 7

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 192

/-- Calculates the number of Hydrogen atoms in the compound -/
def num_H : ℕ := 8

theorem compound_composition :
  (num_C : ℝ) * atomic_weight_C + (num_O : ℝ) * atomic_weight_O + (num_H : ℝ) * atomic_weight_H = molecular_weight := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l1604_160462


namespace NUMINAMATH_CALUDE_tribe_leadership_arrangements_l1604_160493

def tribe_size : ℕ := 15
def num_chiefs : ℕ := 1
def num_supporting_chiefs : ℕ := 2
def inferior_officers_per_chief : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem tribe_leadership_arrangements :
  tribe_size * (tribe_size - 1) * (tribe_size - 2) *
  (choose (tribe_size - 3) inferior_officers_per_chief) *
  (choose (tribe_size - 3 - inferior_officers_per_chief) inferior_officers_per_chief) = 3243240 :=
by sorry

end NUMINAMATH_CALUDE_tribe_leadership_arrangements_l1604_160493


namespace NUMINAMATH_CALUDE_box_interior_surface_area_l1604_160410

theorem box_interior_surface_area :
  let original_length : ℕ := 25
  let original_width : ℕ := 35
  let corner_size : ℕ := 7
  let original_area := original_length * original_width
  let corner_area := corner_size * corner_size
  let total_corner_area := 4 * corner_area
  let remaining_area := original_area - total_corner_area
  remaining_area = 679 := by sorry

end NUMINAMATH_CALUDE_box_interior_surface_area_l1604_160410


namespace NUMINAMATH_CALUDE_triangle_area_l1604_160416

/-- The area of a triangle formed by the points (0,0), (1,1), and (2,1) is 1/2. -/
theorem triangle_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (2, 1)
  let area := (1/2) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))
  area = 1/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l1604_160416


namespace NUMINAMATH_CALUDE_officer_election_proof_l1604_160428

def total_candidates : ℕ := 18
def past_officers : ℕ := 8
def positions_available : ℕ := 6

theorem officer_election_proof :
  (Nat.choose total_candidates positions_available) -
  (Nat.choose (total_candidates - past_officers) positions_available) -
  (Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (positions_available - 1)) = 16338 := by
  sorry

end NUMINAMATH_CALUDE_officer_election_proof_l1604_160428


namespace NUMINAMATH_CALUDE_original_amount_calculation_l1604_160452

theorem original_amount_calculation (total : ℚ) : 
  (3/4 : ℚ) * total - (1/5 : ℚ) * total = 132 → total = 240 := by
  sorry

end NUMINAMATH_CALUDE_original_amount_calculation_l1604_160452


namespace NUMINAMATH_CALUDE_largest_guaranteed_divisor_l1604_160417

def die_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

def is_valid_roll (roll : Finset ℕ) : Prop :=
  roll ⊆ die_numbers ∧ roll.card = 7

def roll_product (roll : Finset ℕ) : ℕ :=
  roll.prod id

theorem largest_guaranteed_divisor :
  ∀ roll : Finset ℕ, is_valid_roll roll →
    ∃ m : ℕ, m = 192 ∧ 
      (∀ n : ℕ, n > 192 → ¬(∀ r : Finset ℕ, is_valid_roll r → n ∣ roll_product r)) ∧
      (192 ∣ roll_product roll) :=
by sorry

end NUMINAMATH_CALUDE_largest_guaranteed_divisor_l1604_160417


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1604_160465

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : 19 / x + 98 / y = 1) : 
  x + y ≥ 117 + 14 * Real.sqrt 38 := by
  sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1604_160465


namespace NUMINAMATH_CALUDE_alice_ice_cream_count_l1604_160404

/-- The number of pints of ice cream Alice had on Wednesday -/
def ice_cream_on_wednesday (sunday_pints : ℕ) : ℕ :=
  let monday_pints := 3 * sunday_pints
  let tuesday_pints := monday_pints / 3
  let total_before_wednesday := sunday_pints + monday_pints + tuesday_pints
  let returned_pints := tuesday_pints / 2
  total_before_wednesday - returned_pints

/-- Theorem stating that Alice had 18 pints of ice cream on Wednesday -/
theorem alice_ice_cream_count : ice_cream_on_wednesday 4 = 18 := by
  sorry

#eval ice_cream_on_wednesday 4

end NUMINAMATH_CALUDE_alice_ice_cream_count_l1604_160404


namespace NUMINAMATH_CALUDE_max_player_salary_l1604_160443

theorem max_player_salary (num_players : ℕ) (min_salary : ℕ) (total_cap : ℕ) :
  num_players = 25 →
  min_salary = 18000 →
  total_cap = 900000 →
  (num_players - 1) * min_salary + (total_cap - (num_players - 1) * min_salary) ≤ total_cap →
  (∀ (salaries : List ℕ), salaries.length = num_players → 
    (∀ s ∈ salaries, s ≥ min_salary) → 
    salaries.sum ≤ total_cap →
    ∀ s ∈ salaries, s ≤ 468000) :=
by sorry

#check max_player_salary

end NUMINAMATH_CALUDE_max_player_salary_l1604_160443


namespace NUMINAMATH_CALUDE_power_seven_mod_nineteen_l1604_160446

theorem power_seven_mod_nineteen : 7^2023 ≡ 4 [ZMOD 19] := by
  sorry

end NUMINAMATH_CALUDE_power_seven_mod_nineteen_l1604_160446


namespace NUMINAMATH_CALUDE_downstream_speed_calculation_l1604_160403

/-- Represents the speed of a person rowing in different conditions -/
structure RowingSpeed where
  upstream : ℝ
  stillWater : ℝ

/-- Calculates the downstream speed given upstream and still water speeds -/
def downstreamSpeed (s : RowingSpeed) : ℝ :=
  2 * s.stillWater - s.upstream

/-- Theorem stating that given the specified upstream and still water speeds, 
    the downstream speed is 53 kmph -/
theorem downstream_speed_calculation (s : RowingSpeed) 
  (h1 : s.upstream = 37) 
  (h2 : s.stillWater = 45) : 
  downstreamSpeed s = 53 := by
  sorry

#eval downstreamSpeed { upstream := 37, stillWater := 45 }

end NUMINAMATH_CALUDE_downstream_speed_calculation_l1604_160403


namespace NUMINAMATH_CALUDE_monotonic_range_a_l1604_160459

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- State the theorem
theorem monotonic_range_a :
  (∀ a : ℝ, ∀ x : ℝ, Monotone (f a)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_range_a_l1604_160459


namespace NUMINAMATH_CALUDE_second_coin_value_l1604_160427

/-- Proves that the value of the second type of coin is 0.5 rupees -/
theorem second_coin_value (total_value : ℝ) (num_coins : ℕ) (coin1_value : ℝ) (coin3_value : ℝ) :
  total_value = 35 →
  num_coins = 20 →
  coin1_value = 1 →
  coin3_value = 0.25 →
  ∃ (coin2_value : ℝ), 
    coin2_value = 0.5 ∧
    num_coins * (coin1_value + coin2_value + coin3_value) = total_value :=
by sorry

end NUMINAMATH_CALUDE_second_coin_value_l1604_160427


namespace NUMINAMATH_CALUDE_even_m_permutation_exists_l1604_160490

/-- A permutation of numbers from 1 to m -/
def Permutation (m : ℕ) := { f : ℕ → ℕ // Function.Bijective f ∧ ∀ i, i ≤ m → f i ≤ m }

/-- Partial sums of a permutation -/
def PartialSums (m : ℕ) (p : Permutation m) : ℕ → ℕ
  | 0 => 0
  | n + 1 => PartialSums m p n + p.val (n + 1)

/-- Different remainders property -/
def DifferentRemainders (m : ℕ) (p : Permutation m) : Prop :=
  ∀ i j, i ≤ m → j ≤ m → i ≠ j → PartialSums m p i % m ≠ PartialSums m p j % m

theorem even_m_permutation_exists (m : ℕ) (h : m > 1) (he : Even m) :
  ∃ p : Permutation m, DifferentRemainders m p := by
  sorry

end NUMINAMATH_CALUDE_even_m_permutation_exists_l1604_160490


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1604_160475

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1604_160475


namespace NUMINAMATH_CALUDE_geometric_sequence_min_value_l1604_160400

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_min_value
  (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0) (h_geom : geometric_sequence a q)
  (h_relation : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n, Real.sqrt (a m * a n) = 4 * a 1) :
  (∀ m n, Real.sqrt (a m * a n) = 4 * a 1 → 1 / m + 5 / n ≥ 7 / 4) ∧
  (∃ m n, Real.sqrt (a m * a n) = 4 * a 1 ∧ 1 / m + 5 / n = 7 / 4) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_min_value_l1604_160400


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1604_160482

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 7/5 of a right angle
  a + b = 7/5 * 90 →
  -- One angle is 20° larger than the other
  b = a + 20 →
  -- All angles are non-negative
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 73°
  max a (max b c) = 73 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1604_160482


namespace NUMINAMATH_CALUDE_white_tiles_count_l1604_160448

theorem white_tiles_count (total : ℕ) (yellow : ℕ) (purple : ℕ) 
  (h_total : total = 20)
  (h_yellow : yellow = 3)
  (h_purple : purple = 6) :
  total - (yellow + (yellow + 1) + purple) = 7 := by
  sorry

end NUMINAMATH_CALUDE_white_tiles_count_l1604_160448


namespace NUMINAMATH_CALUDE_cone_section_area_l1604_160430

-- Define the cone structure
structure Cone where
  -- Axial section is an isosceles right triangle
  axial_section_isosceles_right : Bool
  -- Hypotenuse of axial section
  hypotenuse : ℝ
  -- Angle between section and base
  α : ℝ

-- Define the theorem
theorem cone_section_area (c : Cone) 
  (h1 : c.axial_section_isosceles_right = true) 
  (h2 : c.hypotenuse = 2) 
  (h3 : 0 < c.α ∧ c.α < π / 2) : 
  ∃ (area : ℝ), area = (Real.sqrt 2 / 2) * (1 / (Real.cos c.α)^2) :=
sorry

end NUMINAMATH_CALUDE_cone_section_area_l1604_160430


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1604_160449

theorem sum_of_three_numbers (a b c : ℝ) 
  (sum_of_squares : a^2 + b^2 + c^2 = 149) 
  (sum_of_products : a*b + b*c + a*c = 70) : 
  a + b + c = 17 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1604_160449


namespace NUMINAMATH_CALUDE_fifth_root_of_x_times_fourth_root_l1604_160415

theorem fifth_root_of_x_times_fourth_root (x : ℝ) (hx : x > 0) :
  (x * x^(1/4))^(1/5) = x^(1/4) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_x_times_fourth_root_l1604_160415


namespace NUMINAMATH_CALUDE_stating_at_least_two_different_selections_l1604_160481

/-- The number of available courses -/
def num_courses : ℕ := 6

/-- The number of courses each student must choose -/
def courses_per_student : ℕ := 2

/-- The number of students -/
def num_students : ℕ := 3

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- 
Theorem stating that the number of ways in which at least two out of three students 
can select different combinations of 2 courses from a set of 6 courses is equal to 2520.
-/
theorem at_least_two_different_selections : 
  (choose num_courses courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student * 
   choose num_courses courses_per_student) * num_students - 
  (choose num_courses courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student) * num_students + 
  (choose num_courses courses_per_student * 
   choose (num_courses - courses_per_student) courses_per_student * 
   choose (num_courses - 2 * courses_per_student) courses_per_student) = 2520 :=
by sorry

end NUMINAMATH_CALUDE_stating_at_least_two_different_selections_l1604_160481


namespace NUMINAMATH_CALUDE_mrs_hilt_total_distance_l1604_160439

/-- Calculate the total distance walked by Mrs. Hilt -/
def total_distance (
  water_fountain_dist : ℝ)
  (main_office_dist : ℝ)
  (teachers_lounge_dist : ℝ)
  (water_fountain_increase : ℝ)
  (main_office_increase : ℝ)
  (teachers_lounge_increase : ℝ)
  (water_fountain_visits : ℕ)
  (main_office_visits : ℕ)
  (teachers_lounge_visits : ℕ) : ℝ :=
  let water_fountain_return := water_fountain_dist * (1 + water_fountain_increase)
  let main_office_return := main_office_dist * (1 + main_office_increase)
  let teachers_lounge_return := teachers_lounge_dist * (1 + teachers_lounge_increase)
  (water_fountain_dist + water_fountain_return) * water_fountain_visits +
  (main_office_dist + main_office_return) * main_office_visits +
  (teachers_lounge_dist + teachers_lounge_return) * teachers_lounge_visits

/-- Theorem stating that Mrs. Hilt's total walking distance is 699 feet -/
theorem mrs_hilt_total_distance :
  total_distance 30 50 35 0.15 0.10 0.20 4 2 3 = 699 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_total_distance_l1604_160439


namespace NUMINAMATH_CALUDE_relationship_abc_l1604_160458

theorem relationship_abc : 3^(1/10) > (1/2)^(1/10) ∧ (1/2)^(1/10) > (-1/2)^3 := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l1604_160458


namespace NUMINAMATH_CALUDE_xyz_value_l1604_160402

theorem xyz_value (x y z : ℝ) 
  (h1 : 2 * x + 3 * y + z = 13) 
  (h2 : 4 * x^2 + 9 * y^2 + z^2 - 2 * x + 15 * y + 3 * z = 82) : 
  x * y * z = 12 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1604_160402


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1604_160498

def vector_a : ℝ × ℝ := (3, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem parallel_vectors_x_value :
  ∀ x : ℝ, (vector_a.1 * (vector_b x).2 = vector_a.2 * (vector_b x).1) → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1604_160498


namespace NUMINAMATH_CALUDE_r_value_when_n_is_2_l1604_160435

theorem r_value_when_n_is_2 (n : ℕ) (s r : ℕ) 
  (h1 : s = 3^(n^2) + 1) 
  (h2 : r = 5^s - s) 
  (h3 : n = 2) : 
  r = 5^82 - 82 := by
sorry

end NUMINAMATH_CALUDE_r_value_when_n_is_2_l1604_160435


namespace NUMINAMATH_CALUDE_lily_typing_speed_l1604_160480

/-- Represents Lily's typing scenario -/
structure TypingScenario where
  totalTime : ℕ -- Total time including breaks
  totalWords : ℕ -- Total words typed
  breakInterval : ℕ -- Interval between breaks
  breakDuration : ℕ -- Duration of each break

/-- Calculates the words typed per minute -/
def wordsPerMinute (scenario : TypingScenario) : ℚ :=
  let effectiveTypingTime := scenario.totalTime - (scenario.totalTime / scenario.breakInterval) * scenario.breakDuration
  scenario.totalWords / effectiveTypingTime

/-- Theorem stating that Lily types 15 words per minute -/
theorem lily_typing_speed :
  let scenario : TypingScenario := {
    totalTime := 19
    totalWords := 255
    breakInterval := 10
    breakDuration := 2
  }
  wordsPerMinute scenario = 15 := by
  sorry


end NUMINAMATH_CALUDE_lily_typing_speed_l1604_160480


namespace NUMINAMATH_CALUDE_canoe_kayak_difference_l1604_160453

/-- Represents the daily rental cost of a canoe in dollars -/
def canoe_cost : ℚ := 11

/-- Represents the daily rental cost of a kayak in dollars -/
def kayak_cost : ℚ := 16

/-- Represents the ratio of canoes to kayaks rented -/
def rental_ratio : ℚ := 4 / 3

/-- Represents the total revenue in dollars -/
def total_revenue : ℚ := 460

/-- Represents the number of kayaks rented -/
def kayaks : ℕ := 15

/-- Represents the number of canoes rented -/
def canoes : ℕ := 20

theorem canoe_kayak_difference :
  canoes - kayaks = 5 ∧
  canoe_cost * canoes + kayak_cost * kayaks = total_revenue ∧
  (canoes : ℚ) / kayaks = rental_ratio := by
  sorry

end NUMINAMATH_CALUDE_canoe_kayak_difference_l1604_160453


namespace NUMINAMATH_CALUDE_box_volume_increase_l1604_160472

/-- Proves that for a rectangular box with given dimensions, the value of x that
    satisfies the equation for equal volume increase when increasing length or height is 0. -/
theorem box_volume_increase (l w h : ℝ) (x : ℝ) : 
  l = 6 → w = 4 → h = 5 → ((l + x) * w * h = l * w * (h + x)) → x = 0 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_increase_l1604_160472


namespace NUMINAMATH_CALUDE_triangle_inequality_l1604_160488

theorem triangle_inequality (a b c n : ℝ) (h_triangle : 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b > c ∧ b + c > a ∧ c + a > b) (h_n : 1 ≤ n) :
  let s := (a + b + c) / 2
  (a^n / (b + c) + b^n / (c + a) + c^n / (a + b)) ≥ (2/3)^(n-2) * s^(n-1) := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1604_160488


namespace NUMINAMATH_CALUDE_weight_replacement_l1604_160405

theorem weight_replacement (initial_count : ℕ) (average_increase : ℝ) (new_weight : ℝ) :
  initial_count = 8 →
  average_increase = 2.5 →
  new_weight = 55 →
  ∃ (replaced_weight : ℝ),
    replaced_weight = new_weight - (initial_count * average_increase) ∧
    replaced_weight = 35 :=
by sorry

end NUMINAMATH_CALUDE_weight_replacement_l1604_160405


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1604_160463

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = -1 + Real.sqrt 6 ∧ x₂ = -1 - Real.sqrt 6) ∧ 
  (x₁^2 + 2*x₁ - 5 = 0 ∧ x₂^2 + 2*x₂ - 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1604_160463


namespace NUMINAMATH_CALUDE_min_a_value_l1604_160418

/-- Set A defined by the quadratic inequality -/
def set_A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (1 - a) * p.1^2 + 2 * p.1 * p.2 - a * p.2^2 ≤ 0}

/-- Set B defined by the linear inequality and positivity conditions -/
def set_B : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 3 * p.1 - 5 * p.2 ≥ 0 ∧ p.1 > 0 ∧ p.2 > 0}

/-- Theorem stating the minimum value of a given the subset relationship -/
theorem min_a_value (h : set_B ⊆ set_A a) : a ≥ 55 / 34 := by
  sorry

#check min_a_value

end NUMINAMATH_CALUDE_min_a_value_l1604_160418


namespace NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1604_160486

theorem coefficient_x_squared_in_expansion (x : ℝ) :
  (Finset.range 6).sum (fun k => (Nat.choose 5 k : ℝ) * x^k * (1:ℝ)^(5-k)) =
  10 * x^2 + (Finset.range 6).sum (fun k => if k ≠ 2 then (Nat.choose 5 k : ℝ) * x^k * (1:ℝ)^(5-k) else 0) :=
by sorry


end NUMINAMATH_CALUDE_coefficient_x_squared_in_expansion_l1604_160486


namespace NUMINAMATH_CALUDE_cards_distribution_l1604_160489

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) 
  (h2 : num_people = 9) : 
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l1604_160489


namespace NUMINAMATH_CALUDE_trig_expression_value_quadratic_equation_solutions_quadratic_root_property_l1604_160432

-- Part 1
theorem trig_expression_value : 
  2 * Real.tan (60 * π / 180) * Real.cos (30 * π / 180) - Real.sin (45 * π / 180) ^ 2 = 5/2 := by
sorry

-- Part 2
theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x => 2 * (x + 2)^2 - 3 * (x + 2)
  ∃ x₁ x₂ : ℝ, x₁ = -2 ∧ x₂ = -1/2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ 
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
sorry

-- Part 3
theorem quadratic_root_property :
  ∀ m : ℝ, m^2 - 5*m - 2 = 0 → 2*m^2 - 10*m + 2023 = 2027 := by
sorry

end NUMINAMATH_CALUDE_trig_expression_value_quadratic_equation_solutions_quadratic_root_property_l1604_160432


namespace NUMINAMATH_CALUDE_program_count_l1604_160492

/-- The total number of courses available --/
def total_courses : ℕ := 7

/-- The number of courses in a program --/
def program_size : ℕ := 5

/-- The number of math courses available --/
def math_courses : ℕ := 2

/-- The number of non-math courses available (excluding English) --/
def non_math_courses : ℕ := total_courses - math_courses - 1

/-- The minimum number of math courses required in a program --/
def min_math_courses : ℕ := 2

/-- Calculates the number of ways to choose a program --/
def calculate_programs : ℕ :=
  Nat.choose non_math_courses (program_size - min_math_courses - 1) +
  Nat.choose non_math_courses (program_size - math_courses - 1)

theorem program_count : calculate_programs = 6 := by sorry

end NUMINAMATH_CALUDE_program_count_l1604_160492


namespace NUMINAMATH_CALUDE_women_in_room_l1604_160441

theorem women_in_room (initial_men initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 7 / 9 →
  initial_men + 5 = 23 →
  3 * (initial_women - 4) = 57 :=
by sorry

end NUMINAMATH_CALUDE_women_in_room_l1604_160441


namespace NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_22_l1604_160494

theorem smallest_k_for_64_power_gt_4_power_22 : 
  ∃ k : ℕ, (∀ m : ℕ, 64^m > 4^22 → k ≤ m) ∧ 64^k > 4^22 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_k_for_64_power_gt_4_power_22_l1604_160494


namespace NUMINAMATH_CALUDE_prob_double_is_one_seventh_l1604_160425

/-- The number of integers in the modified domino set -/
def n : ℕ := 13

/-- The total number of domino pairings in the set -/
def total_pairings : ℕ := n * (n + 1) / 2

/-- The number of doubles in the set -/
def num_doubles : ℕ := n

/-- The probability of selecting a double from the modified domino set -/
def prob_double : ℚ := num_doubles / total_pairings

theorem prob_double_is_one_seventh : prob_double = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_double_is_one_seventh_l1604_160425


namespace NUMINAMATH_CALUDE_max_volume_cube_l1604_160460

/-- A rectangular solid with length l, width w, and height h -/
structure RectangularSolid where
  l : ℝ
  w : ℝ
  h : ℝ
  l_pos : 0 < l
  w_pos : 0 < w
  h_pos : 0 < h

/-- The surface area of a rectangular solid -/
def surfaceArea (r : RectangularSolid) : ℝ :=
  2 * (r.l * r.w + r.l * r.h + r.w * r.h)

/-- The volume of a rectangular solid -/
def volume (r : RectangularSolid) : ℝ :=
  r.l * r.w * r.h

/-- Theorem: Among all rectangular solids with a fixed surface area S,
    the cube has the maximum volume, and this maximum volume is (S/6)^(3/2) -/
theorem max_volume_cube (S : ℝ) (h_pos : 0 < S) :
  ∃ (max_vol : ℝ),
    (∀ (r : RectangularSolid), surfaceArea r = S → volume r ≤ max_vol) ∧
    (∃ (cube : RectangularSolid), surfaceArea cube = S ∧ volume cube = max_vol) ∧
    max_vol = (S / 6) ^ (3/2) :=
  sorry

end NUMINAMATH_CALUDE_max_volume_cube_l1604_160460
