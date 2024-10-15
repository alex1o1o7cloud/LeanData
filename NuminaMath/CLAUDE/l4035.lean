import Mathlib

namespace NUMINAMATH_CALUDE_leftover_value_is_zero_l4035_403571

/-- Represents the number of coins in a roll -/
def roll_size : ℕ := 40

/-- Represents Michael's coin counts -/
def michael_quarters : ℕ := 75
def michael_nickels : ℕ := 123

/-- Represents Sarah's coin counts -/
def sarah_quarters : ℕ := 85
def sarah_nickels : ℕ := 157

/-- Calculates the total number of quarters -/
def total_quarters : ℕ := michael_quarters + sarah_quarters

/-- Calculates the total number of nickels -/
def total_nickels : ℕ := michael_nickels + sarah_nickels

/-- Calculates the number of leftover quarters -/
def leftover_quarters : ℕ := total_quarters % roll_size

/-- Calculates the number of leftover nickels -/
def leftover_nickels : ℕ := total_nickels % roll_size

/-- Represents the value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- Represents the value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- Calculates the total value of leftover coins in cents -/
def leftover_value : ℕ := leftover_quarters * quarter_value + leftover_nickels * nickel_value

/-- Theorem stating that the value of leftover coins is $0.00 -/
theorem leftover_value_is_zero : leftover_value = 0 := by sorry

end NUMINAMATH_CALUDE_leftover_value_is_zero_l4035_403571


namespace NUMINAMATH_CALUDE_m_range_l4035_403578

theorem m_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ 
  (∀ x : ℝ, x^2 + m * x + 1 > 0) → 
  -2 < m ∧ m < 0 := by sorry

end NUMINAMATH_CALUDE_m_range_l4035_403578


namespace NUMINAMATH_CALUDE_four_heads_in_five_tosses_l4035_403531

def n : ℕ := 5
def k : ℕ := 4
def p : ℚ := 1/2

def binomial_coefficient (n k : ℕ) : ℕ := 
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binomial_coefficient n k : ℚ) * p^k * (1 - p)^(n - k)

theorem four_heads_in_five_tosses : 
  binomial_probability n k p = 5/32 := by sorry

end NUMINAMATH_CALUDE_four_heads_in_five_tosses_l4035_403531


namespace NUMINAMATH_CALUDE_sum_surface_areas_of_cut_cube_l4035_403508

/-- The sum of surface areas of cuboids resulting from cutting a unit cube -/
theorem sum_surface_areas_of_cut_cube : 
  let n : ℕ := 4  -- number of divisions per side
  let num_cuboids : ℕ := n^3
  let side_length : ℚ := 1 / n
  let surface_area_one_cuboid : ℚ := 6 * side_length^2
  surface_area_one_cuboid * num_cuboids = 24 := by sorry

end NUMINAMATH_CALUDE_sum_surface_areas_of_cut_cube_l4035_403508


namespace NUMINAMATH_CALUDE_min_value_cubic_quadratic_l4035_403594

theorem min_value_cubic_quadratic (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : 57 * a + 88 * b + 125 * c ≥ 1148) : 
  ∀ x y z : ℝ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 57 * x + 88 * y + 125 * z ≥ 1148 →
  a^3 + b^3 + c^3 + 5*a^2 + 5*b^2 + 5*c^2 ≤ x^3 + y^3 + z^3 + 5*x^2 + 5*y^2 + 5*z^2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_cubic_quadratic_l4035_403594


namespace NUMINAMATH_CALUDE_max_player_score_l4035_403589

theorem max_player_score (total_players : ℕ) (total_points : ℕ) (min_points : ℕ) 
  (h1 : total_players = 12)
  (h2 : total_points = 100)
  (h3 : min_points = 7)
  (h4 : ∀ player, player ≥ min_points) :
  ∃ max_score : ℕ, max_score = 23 ∧ 
  (∀ player_score : ℕ, player_score ≤ max_score) ∧
  (∃ player : ℕ, player = max_score) ∧
  (total_points = (total_players - 1) * min_points + max_score) :=
by sorry

end NUMINAMATH_CALUDE_max_player_score_l4035_403589


namespace NUMINAMATH_CALUDE_parking_lot_theorem_l4035_403510

/-- A multi-story parking lot with equal-sized levels -/
structure ParkingLot where
  total_spaces : ℕ
  num_levels : ℕ
  cars_on_one_level : ℕ

/-- Calculates the number of additional cars that can fit on one level -/
def additional_cars (p : ParkingLot) : ℕ :=
  (p.total_spaces / p.num_levels) - p.cars_on_one_level

theorem parking_lot_theorem (p : ParkingLot) 
  (h1 : p.total_spaces = 425)
  (h2 : p.num_levels = 5)
  (h3 : p.cars_on_one_level = 23) :
  additional_cars p = 62 := by
  sorry

#eval additional_cars { total_spaces := 425, num_levels := 5, cars_on_one_level := 23 }

end NUMINAMATH_CALUDE_parking_lot_theorem_l4035_403510


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l4035_403529

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 222) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 22 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l4035_403529


namespace NUMINAMATH_CALUDE_andrews_friends_pizza_l4035_403559

theorem andrews_friends_pizza (total_slices : ℕ) (slices_per_friend : ℕ) (num_friends : ℕ) :
  total_slices = 16 →
  slices_per_friend = 4 →
  total_slices = num_friends * slices_per_friend →
  num_friends = 4 := by
sorry

end NUMINAMATH_CALUDE_andrews_friends_pizza_l4035_403559


namespace NUMINAMATH_CALUDE_park_short_bushes_after_planting_l4035_403505

/-- The number of short bushes in a park after planting new ones. -/
def total_short_bushes (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem stating that the total number of short bushes after planting is 57. -/
theorem park_short_bushes_after_planting :
  total_short_bushes 37 20 = 57 := by
  sorry

end NUMINAMATH_CALUDE_park_short_bushes_after_planting_l4035_403505


namespace NUMINAMATH_CALUDE_day_crew_fraction_is_eight_elevenths_l4035_403570

/-- Represents the fraction of boxes loaded by the day crew given the relative productivity and size of the night crew -/
def day_crew_fraction (night_crew_productivity : ℚ) (night_crew_size : ℚ) : ℚ :=
  1 / (1 + night_crew_productivity * night_crew_size)

theorem day_crew_fraction_is_eight_elevenths :
  day_crew_fraction (3/4) (1/2) = 8/11 := by
  sorry

end NUMINAMATH_CALUDE_day_crew_fraction_is_eight_elevenths_l4035_403570


namespace NUMINAMATH_CALUDE_furniture_shop_cost_price_l4035_403563

/-- Proves that the cost price of an item is 6672 when the selling price is 8340
    and the markup is 25%. -/
theorem furniture_shop_cost_price : 
  ∀ (cost_price selling_price : ℝ),
  selling_price = 8340 →
  selling_price = cost_price * (1 + 0.25) →
  cost_price = 6672 := by sorry

end NUMINAMATH_CALUDE_furniture_shop_cost_price_l4035_403563


namespace NUMINAMATH_CALUDE_roots_of_equation_l4035_403582

theorem roots_of_equation (x : ℝ) : (x - 1)^2 = 1 ↔ x = 0 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_equation_l4035_403582


namespace NUMINAMATH_CALUDE_impossible_score_l4035_403532

/-- Represents the score of a quiz -/
structure QuizScore where
  correct : ℕ
  unanswered : ℕ
  incorrect : ℕ
  total_questions : ℕ
  score : ℤ

/-- The quiz scoring system -/
def quiz_score (qs : QuizScore) : Prop :=
  qs.correct + qs.unanswered + qs.incorrect = qs.total_questions ∧
  qs.score = 5 * qs.correct + 2 * qs.unanswered - qs.incorrect

theorem impossible_score : 
  ∀ qs : QuizScore, 
  qs.total_questions = 25 → 
  quiz_score qs → 
  qs.score ≠ 127 := by
sorry

end NUMINAMATH_CALUDE_impossible_score_l4035_403532


namespace NUMINAMATH_CALUDE_line_points_property_l4035_403545

theorem line_points_property (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ) :
  y₁ = -2 * x₁ + 3 →
  y₂ = -2 * x₂ + 3 →
  y₃ = -2 * x₃ + 3 →
  x₁ < x₂ →
  x₂ < x₃ →
  x₂ * x₃ < 0 →
  y₁ * y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_line_points_property_l4035_403545


namespace NUMINAMATH_CALUDE_trigonometric_product_equality_l4035_403504

theorem trigonometric_product_equality : 
  3.420 * Real.sin (10 * π / 180) * Real.sin (20 * π / 180) * Real.sin (30 * π / 180) * 
  Real.sin (40 * π / 180) * Real.sin (50 * π / 180) * Real.sin (60 * π / 180) * 
  Real.sin (70 * π / 180) * Real.sin (80 * π / 180) = 3 / 256 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_product_equality_l4035_403504


namespace NUMINAMATH_CALUDE_solve_2a_plus_b_l4035_403586

theorem solve_2a_plus_b (a b : ℝ) 
  (h1 : 4 * a^2 - b^2 = 12) 
  (h2 : 2 * a - b = 4) : 
  2 * a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_solve_2a_plus_b_l4035_403586


namespace NUMINAMATH_CALUDE_grid_tiling_condition_l4035_403500

/-- Represents a tile type that can cover a 2x2 or larger area of a grid -/
structure Tile :=
  (width : ℕ)
  (height : ℕ)
  (valid : width ≥ 2 ∧ height ≥ 2)

/-- Represents the set of 6 available tile types -/
def TileSet : Set Tile := sorry

/-- Predicate to check if a grid can be tiled with the given tile set -/
def canBeTiled (m n : ℕ) (tiles : Set Tile) : Prop := sorry

/-- Main theorem: A rectangular grid can be tiled iff 4 divides m or n, and neither is 1 -/
theorem grid_tiling_condition (m n : ℕ) :
  canBeTiled m n TileSet ↔ (4 ∣ m ∨ 4 ∣ n) ∧ m ≠ 1 ∧ n ≠ 1 :=
sorry

end NUMINAMATH_CALUDE_grid_tiling_condition_l4035_403500


namespace NUMINAMATH_CALUDE_q_minus_p_equals_zero_l4035_403553

def P : Set ℕ := {1, 2, 3, 4, 5}
def Q : Set ℕ := {0, 2, 3}

def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem q_minus_p_equals_zero : set_difference Q P = {0} := by sorry

end NUMINAMATH_CALUDE_q_minus_p_equals_zero_l4035_403553


namespace NUMINAMATH_CALUDE_multiplication_formula_98_102_l4035_403554

theorem multiplication_formula_98_102 : 98 * 102 = 9996 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_formula_98_102_l4035_403554


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l4035_403540

theorem tangent_line_to_circle (r : ℝ) (h1 : r > 0) : 
  (∃ (x y : ℝ), x + y = 2*r ∧ (x - 1)^2 + (y - 1)^2 = r^2 ∧ 
   ∀ (x' y' : ℝ), x' + y' = 2*r → (x' - 1)^2 + (y' - 1)^2 ≥ r^2) →
  r = 2 + Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l4035_403540


namespace NUMINAMATH_CALUDE_count_off_ones_l4035_403506

theorem count_off_ones (n : ℕ) (h : n = 1994) : 
  (n / (Nat.lcm 3 4) : ℕ) = 166 := by
  sorry

end NUMINAMATH_CALUDE_count_off_ones_l4035_403506


namespace NUMINAMATH_CALUDE_parabola_focus_l4035_403502

/-- A parabola is defined by the equation x^2 = -8y -/
def parabola (x y : ℝ) : Prop := x^2 = -8*y

/-- The focus of a parabola is a point on its axis of symmetry -/
def is_focus (x y : ℝ) (p : ℝ → ℝ → Prop) : Prop :=
  ∀ (u v : ℝ), p u v → (x = 0 ∧ y = -2)

/-- Theorem: The focus of the parabola x^2 = -8y is located at (0, -2) -/
theorem parabola_focus :
  is_focus 0 (-2) parabola :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_l4035_403502


namespace NUMINAMATH_CALUDE_even_periodic_increasing_function_inequality_l4035_403518

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem even_periodic_increasing_function_inequality (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_period : has_period_two f)
  (h_increasing : increasing_on f (-1) 0) :
  f 3 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f 2 :=
sorry

end NUMINAMATH_CALUDE_even_periodic_increasing_function_inequality_l4035_403518


namespace NUMINAMATH_CALUDE_optimal_production_solution_l4035_403514

/-- Represents the production problem with given parameters -/
structure ProductionProblem where
  total_units : ℕ
  workers : ℕ
  a_per_unit : ℕ
  b_per_unit : ℕ
  c_per_unit : ℕ
  a_per_worker : ℕ
  b_per_worker : ℕ
  c_per_worker : ℕ

/-- Calculates the completion time for a given worker distribution -/
def completion_time (prob : ProductionProblem) (x k : ℕ) : ℚ :=
  max (prob.a_per_unit * prob.total_units / (prob.a_per_worker * x : ℚ))
    (max (prob.b_per_unit * prob.total_units / (prob.b_per_worker * k * x : ℚ))
         (prob.c_per_unit * prob.total_units / (prob.c_per_worker * (prob.workers - (1 + k) * x) : ℚ)))

/-- The main theorem stating the optimal solution -/
theorem optimal_production_solution (prob : ProductionProblem) 
    (h_prob : prob.total_units = 3000 ∧ prob.workers = 200 ∧ 
              prob.a_per_unit = 2 ∧ prob.b_per_unit = 2 ∧ prob.c_per_unit = 1 ∧
              prob.a_per_worker = 6 ∧ prob.b_per_worker = 3 ∧ prob.c_per_worker = 2) :
    ∃ (x : ℕ), x > 0 ∧ x < prob.workers ∧ 
    completion_time prob x 2 = 250 / 11 ∧
    ∀ (y k : ℕ), y > 0 → y < prob.workers → k > 0 → 
    completion_time prob y k ≥ 250 / 11 := by
  sorry

end NUMINAMATH_CALUDE_optimal_production_solution_l4035_403514


namespace NUMINAMATH_CALUDE_largest_inscribed_triangle_l4035_403533

-- Define a convex polygon
def ConvexPolygon (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define an inscribed triangle in a polygon
def InscribedTriangle (T : Set (ℝ × ℝ)) (M : Set (ℝ × ℝ)) : Prop := sorry

-- Define the area of a triangle
def TriangleArea (T : Set (ℝ × ℝ)) : ℝ := sorry

-- Define a triangle formed by three vertices of a polygon
def VertexTriangle (T : Set (ℝ × ℝ)) (M : Set (ℝ × ℝ)) : Prop := sorry

theorem largest_inscribed_triangle (M : Set (ℝ × ℝ)) (h : ConvexPolygon M) :
  ∃ (T : Set (ℝ × ℝ)), VertexTriangle T M ∧
    ∀ (S : Set (ℝ × ℝ)), InscribedTriangle S M → TriangleArea S ≤ TriangleArea T :=
sorry

end NUMINAMATH_CALUDE_largest_inscribed_triangle_l4035_403533


namespace NUMINAMATH_CALUDE_percent_problem_l4035_403535

theorem percent_problem (x : ℝ) (h : 0.4 * x = 160) : 0.5 * x = 200 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l4035_403535


namespace NUMINAMATH_CALUDE_pen_cost_l4035_403598

theorem pen_cost (pen_cost ink_cost : ℝ) 
  (total_cost : pen_cost + ink_cost = 1.10)
  (price_difference : pen_cost = ink_cost + 1) : 
  pen_cost = 1.05 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l4035_403598


namespace NUMINAMATH_CALUDE_number_equation_solution_l4035_403509

theorem number_equation_solution :
  ∃ x : ℝ, 5.4 * x + 0.6 = 108.45000000000003 ∧ x = 19.97222222222222 :=
by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l4035_403509


namespace NUMINAMATH_CALUDE_shape_triangle_area_ratio_l4035_403536

/-- A shape with a certain area -/
structure Shape where
  area : ℝ
  area_pos : area > 0

/-- A triangle with a certain area -/
structure Triangle where
  area : ℝ
  area_pos : area > 0

/-- The theorem stating the relationship between the areas of a shape and a triangle -/
theorem shape_triangle_area_ratio 
  (s : Shape) 
  (t : Triangle) 
  (h : s.area / t.area = 2) : 
  s.area = 2 * t.area := by
  sorry

end NUMINAMATH_CALUDE_shape_triangle_area_ratio_l4035_403536


namespace NUMINAMATH_CALUDE_complex_expression_equality_l4035_403587

theorem complex_expression_equality (y : ℂ) (h : y = Complex.exp (2 * π * I / 9)) :
  (3 * y + y^3) * (3 * y^3 + y^9) * (3 * y^6 + y^18) = 121 + 48 * (y + y^6) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l4035_403587


namespace NUMINAMATH_CALUDE_symmetry_implies_k_and_b_l4035_403564

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Checks if two lines are symmetric with respect to the vertical line x = a -/
def symmetric_lines (l1 l2 : Line) (a : ℝ) : Prop :=
  l1.slope = -l2.slope ∧
  l1.intercept + l2.intercept = 2 * (l1.slope * a + l1.intercept)

/-- The main theorem stating the conditions for symmetry and the resulting values of k and b -/
theorem symmetry_implies_k_and_b (k b : ℝ) :
  symmetric_lines (Line.mk k 3) (Line.mk 2 b) 1 →
  k = -2 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_k_and_b_l4035_403564


namespace NUMINAMATH_CALUDE_distance_between_points_l4035_403524

/-- The distance between the points (2, -1) and (-3, 6) is √74. -/
theorem distance_between_points : Real.sqrt 74 = Real.sqrt ((2 - (-3))^2 + ((-1) - 6)^2) := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l4035_403524


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l4035_403573

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_cond : 2 * a 4 + a 3 - 2 * a 2 - a 1 = 8) :
  ∃ m : ℝ, m = 12 * Real.sqrt 3 ∧ ∀ x : ℝ, 2 * a 5 + a 4 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l4035_403573


namespace NUMINAMATH_CALUDE_tangent_line_equation_l4035_403556

def f (x : ℝ) : ℝ := x^3 - 3*x

theorem tangent_line_equation (P : ℝ × ℝ) (h₁ : P = (-2, -2)) :
  ∃ (m b : ℝ), (∀ x, (m * x + b = 9 * x + 16) ∨ (m * x + b = -2)) ∧
  (∃ x₀, f x₀ = m * x₀ + b ∧ 
         ∀ x, f x ≥ m * x + b ∧ 
         (f x = m * x + b ↔ x = x₀)) ∧
  (m * P.1 + b = P.2) := by
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l4035_403556


namespace NUMINAMATH_CALUDE_original_profit_margin_l4035_403562

theorem original_profit_margin
  (original_price : ℝ)
  (original_margin : ℝ)
  (h_price_decrease : ℝ → ℝ → Prop)
  (h_margin_increase : ℝ → ℝ → Prop) :
  h_price_decrease original_price (original_price * (1 - 0.064)) →
  h_margin_increase original_margin (original_margin + 0.08) →
  original_margin = 0.17 :=
by sorry

end NUMINAMATH_CALUDE_original_profit_margin_l4035_403562


namespace NUMINAMATH_CALUDE_five_balls_four_boxes_l4035_403597

/-- The number of ways to place n distinguishable objects into k distinguishable containers -/
def placement_count (n k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to place 5 distinguishable balls into 4 distinguishable boxes is 4^5 -/
theorem five_balls_four_boxes : placement_count 5 4 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_five_balls_four_boxes_l4035_403597


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4035_403523

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 4) = 20 → ∃ y : ℝ, (y + 3) * (y - 4) = 20 ∧ x + y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l4035_403523


namespace NUMINAMATH_CALUDE_double_average_l4035_403595

theorem double_average (n : Nat) (original_avg : Nat) (h1 : n = 12) (h2 : original_avg = 36) :
  let total := n * original_avg
  let doubled_total := 2 * total
  let new_avg := doubled_total / n
  new_avg = 72 := by
sorry

end NUMINAMATH_CALUDE_double_average_l4035_403595


namespace NUMINAMATH_CALUDE_line_l_equation_l4035_403577

-- Define the types for points and lines
def Point := ℝ × ℝ
def Line := ℝ → ℝ → ℝ

-- Define the given conditions
def point_on_l : Point := (2, 3)
def L1 : Line := λ x y => 2*x - 5*y + 9
def L2 : Line := λ x y => 2*x - 5*y - 7
def midpoint_line : Line := λ x y => x - 4*y - 1

-- Define the line l
def l : Line := λ x y => 4*x - 5*y + 7

-- Theorem statement
theorem line_l_equation : 
  ∃ (A B : Point),
    (L1 A.1 A.2 = 0 ∧ L2 B.1 B.2 = 0) ∧ 
    (midpoint_line ((A.1 + B.1)/2) ((A.2 + B.2)/2) = 0) ∧
    (l point_on_l.1 point_on_l.2 = 0) ∧
    (∀ (x y : ℝ), l x y = 0 ↔ 4*x - 5*y + 7 = 0) :=
by sorry

end NUMINAMATH_CALUDE_line_l_equation_l4035_403577


namespace NUMINAMATH_CALUDE_complex_magnitude_range_l4035_403521

theorem complex_magnitude_range (z : ℂ) (h : Complex.abs z = 1) :
  4 * Real.sqrt 2 ≤ Complex.abs ((z + 1) + Complex.I * (7 - z)) ∧
  Complex.abs ((z + 1) + Complex.I * (7 - z)) ≤ 6 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_complex_magnitude_range_l4035_403521


namespace NUMINAMATH_CALUDE_simplify_expression_l4035_403513

theorem simplify_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^3 - b^3 = a - b) :
  a/b - b/a + 1/(a*b) = -1 + 2/(a*b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4035_403513


namespace NUMINAMATH_CALUDE_alice_numbers_l4035_403583

theorem alice_numbers : ∃ x y : ℝ, x * y = 12 ∧ x + y = 7 ∧ ({x, y} : Set ℝ) = {3, 4} := by
  sorry

end NUMINAMATH_CALUDE_alice_numbers_l4035_403583


namespace NUMINAMATH_CALUDE_definite_integral_semicircle_l4035_403552

theorem definite_integral_semicircle (f : ℝ → ℝ) (r : ℝ) :
  (∀ x, f x = Real.sqrt (r^2 - x^2)) →
  r > 0 →
  ∫ x in (0)..(r), f x = (π * r^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_semicircle_l4035_403552


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l4035_403539

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 25 = 4

/-- The slopes of the asymptotes -/
def asymptote_slopes : Set ℝ := {0.8, -0.8}

/-- Theorem stating that the slopes of the asymptotes of the given hyperbola are ±0.8 -/
theorem hyperbola_asymptote_slopes :
  ∀ (x y : ℝ), hyperbola_eq x y → (∃ (m : ℝ), m ∈ asymptote_slopes ∧ 
    ∃ (b : ℝ), y = m * x + b) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slopes_l4035_403539


namespace NUMINAMATH_CALUDE_discriminant_greater_than_four_l4035_403561

theorem discriminant_greater_than_four (p q : ℝ) 
  (h1 : 999^2 + p * 999 + q < 0) 
  (h2 : 1001^2 + p * 1001 + q < 0) : 
  p^2 - 4*q > 4 := by
sorry

end NUMINAMATH_CALUDE_discriminant_greater_than_four_l4035_403561


namespace NUMINAMATH_CALUDE_largest_even_number_under_300_l4035_403565

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ n % 2 = 0 ∧ n ≤ 300

theorem largest_even_number_under_300 :
  ∀ n : ℕ, is_valid_number n → n ≤ 298 :=
by
  sorry

#check largest_even_number_under_300

end NUMINAMATH_CALUDE_largest_even_number_under_300_l4035_403565


namespace NUMINAMATH_CALUDE_banana_arrangements_l4035_403520

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (repetitions : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (repetitions.map Nat.factorial).prod

/-- Proof that the number of distinct arrangements of "BANANA" is 60 -/
theorem banana_arrangements :
  distinctArrangements 6 [3, 2, 1] = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l4035_403520


namespace NUMINAMATH_CALUDE_print_time_rounded_l4035_403544

/-- The number of pages to be printed -/
def total_pages : ℕ := 350

/-- The number of pages printed per minute -/
def pages_per_minute : ℕ := 25

/-- Rounds a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

/-- The time required to print the pages, in minutes -/
def print_time : ℚ := total_pages / pages_per_minute

theorem print_time_rounded : round_to_nearest print_time = 14 := by
  sorry

end NUMINAMATH_CALUDE_print_time_rounded_l4035_403544


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l4035_403585

/-- The last digit of the decimal expansion of 1/3^15 is 0 -/
theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) : 
  n = 15 → (∃ (k : ℕ), (1 : ℚ) / 3^n = k * (1 / 10^n) + (1 / 10^n)) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l4035_403585


namespace NUMINAMATH_CALUDE_percentage_difference_l4035_403572

theorem percentage_difference (A B x : ℝ) : 
  A > B ∧ B > 0 → A = B * (1 + x / 100) → x = 100 * (A - B) / B := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l4035_403572


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l4035_403566

theorem quadratic_solution_property (k : ℚ) : 
  (∃ a b : ℚ, 
    (5 * a^2 + 7 * a + k = 0) ∧ 
    (5 * b^2 + 7 * b + k = 0) ∧ 
    (abs (a - b) = a^2 + b^2)) ↔ 
  (k = 21/25 ∨ k = -21/25) := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l4035_403566


namespace NUMINAMATH_CALUDE_triangle_side_relation_l4035_403581

theorem triangle_side_relation (a b c : ℝ) : 
  (a > 0 ∧ b > 0 ∧ c > 0) →  -- Positive lengths
  (a + b > c ∧ b + c > a ∧ c + a > b) →  -- Triangle inequality
  a^2 + 4*a*c + 3*c^2 - 3*a*b - 7*b*c + 2*b^2 = 0 →
  a + c - 2*b = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_relation_l4035_403581


namespace NUMINAMATH_CALUDE_min_even_integers_l4035_403599

theorem min_even_integers (a b c d e f g : ℤ) : 
  a + b + c = 30 →
  a + b + c + d + e = 48 →
  a + b + c + d + e + f + g = 60 →
  ∃ (a' b' c' d' e' f' g' : ℤ), 
    a' + b' + c' = 30 ∧
    a' + b' + c' + d' + e' = 48 ∧
    a' + b' + c' + d' + e' + f' + g' = 60 ∧
    Even a' ∧ Even b' ∧ Even c' ∧ Even d' ∧ Even e' ∧ Even f' ∧ Even g' :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l4035_403599


namespace NUMINAMATH_CALUDE_shirt_sale_tax_percentage_l4035_403503

theorem shirt_sale_tax_percentage : 
  let num_fandoms : ℕ := 4
  let shirts_per_fandom : ℕ := 5
  let original_price : ℚ := 15
  let discount_percentage : ℚ := 20 / 100
  let total_paid : ℚ := 264

  let discounted_price : ℚ := original_price * (1 - discount_percentage)
  let total_shirts : ℕ := num_fandoms * shirts_per_fandom
  let total_cost_before_tax : ℚ := discounted_price * total_shirts
  let tax_amount : ℚ := total_paid - total_cost_before_tax
  let tax_percentage : ℚ := tax_amount / total_cost_before_tax * 100

  tax_percentage = 10 := by sorry

end NUMINAMATH_CALUDE_shirt_sale_tax_percentage_l4035_403503


namespace NUMINAMATH_CALUDE_circle_area_relation_l4035_403537

/-- Two circles are tangent if they touch at exactly one point. -/
def CirclesTangent (A B : Set ℝ × ℝ) : Prop := sorry

/-- A circle passes through a point if the point lies on the circle's circumference. -/
def CirclePassesThrough (C : Set ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

/-- The center of a circle. -/
def CircleCenter (C : Set ℝ × ℝ) : ℝ × ℝ := sorry

/-- The area of a circle. -/
def CircleArea (C : Set ℝ × ℝ) : ℝ := sorry

/-- Theorem: Given two circles A and B, where A is tangent to B and passes through B's center,
    if the area of A is 16π, then the area of B is 64π. -/
theorem circle_area_relation (A B : Set ℝ × ℝ) :
  CirclesTangent A B →
  CirclePassesThrough A (CircleCenter B) →
  CircleArea A = 16 * Real.pi →
  CircleArea B = 64 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circle_area_relation_l4035_403537


namespace NUMINAMATH_CALUDE_cindys_math_operation_l4035_403522

theorem cindys_math_operation (x : ℝ) : (x - 12) / 2 = 64 → (x - 6) / 4 = 33.5 := by
  sorry

end NUMINAMATH_CALUDE_cindys_math_operation_l4035_403522


namespace NUMINAMATH_CALUDE_periodic_decimal_to_fraction_l4035_403501

theorem periodic_decimal_to_fraction :
  (0.02 : ℚ) = 2 / 99 →
  (2.06 : ℚ) = 68 / 33 := by
sorry

end NUMINAMATH_CALUDE_periodic_decimal_to_fraction_l4035_403501


namespace NUMINAMATH_CALUDE_tangent_addition_formula_l4035_403558

theorem tangent_addition_formula : 
  (Real.tan (12 * π / 180) + Real.tan (18 * π / 180)) / 
  (1 - Real.tan (12 * π / 180) * Real.tan (18 * π / 180)) = Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_tangent_addition_formula_l4035_403558


namespace NUMINAMATH_CALUDE_largest_triangle_perimeter_l4035_403574

theorem largest_triangle_perimeter (a b x : ℕ) : 
  a = 8 → b = 11 → x ∈ Set.Icc 4 18 → 
  (∀ y : ℕ, y ∈ Set.Icc 4 18 → a + b + y ≤ a + b + x) →
  a + b + x = 37 := by
sorry

end NUMINAMATH_CALUDE_largest_triangle_perimeter_l4035_403574


namespace NUMINAMATH_CALUDE_polynomial_parity_l4035_403517

/-- Represents a polynomial with integer coefficients -/
def IntPolynomial := List Int

/-- Multiplies two polynomials -/
def polyMult (p q : IntPolynomial) : IntPolynomial := sorry

/-- Checks if all elements in a list are even -/
def allEven (l : List Int) : Prop := ∀ x ∈ l, Even x

/-- Checks if all elements in a list are multiples of 4 -/
def allMultiplesOf4 (l : List Int) : Prop := ∀ x ∈ l, ∃ k, x = 4 * k

/-- Checks if at least one element in a list is odd -/
def hasOdd (l : List Int) : Prop := ∃ x ∈ l, Odd x

theorem polynomial_parity (P Q : IntPolynomial) :
  (allEven (polyMult P Q)) ∧ ¬(allMultiplesOf4 (polyMult P Q)) →
  ((allEven P ∧ hasOdd Q) ∨ (allEven Q ∧ hasOdd P)) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_parity_l4035_403517


namespace NUMINAMATH_CALUDE_sandbox_ratio_l4035_403542

/-- A rectangular sandbox with specific dimensions. -/
structure Sandbox where
  width : ℝ
  length : ℝ
  perimeter : ℝ
  length_multiple : ℝ
  width_eq : width = 5
  perimeter_eq : perimeter = 30
  length_eq : length = length_multiple * width

/-- The ratio of length to width for a sandbox with given properties is 2:1. -/
theorem sandbox_ratio (s : Sandbox) : s.length / s.width = 2 := by
  sorry


end NUMINAMATH_CALUDE_sandbox_ratio_l4035_403542


namespace NUMINAMATH_CALUDE_foundation_dig_time_l4035_403543

/-- Represents the time taken to dig a foundation given the number of men -/
def digTime (men : ℕ) : ℝ := sorry

theorem foundation_dig_time :
  (digTime 20 = 6) →  -- It takes 20 men 6 days
  (∀ m₁ m₂ : ℕ, m₁ * digTime m₁ = m₂ * digTime m₂) →  -- Inverse proportion
  digTime 30 = 4 := by sorry

end NUMINAMATH_CALUDE_foundation_dig_time_l4035_403543


namespace NUMINAMATH_CALUDE_gum_cost_proof_l4035_403591

/-- The cost of gum in dollars -/
def cost_in_dollars (pieces : ℕ) (cents_per_piece : ℕ) : ℚ :=
  (pieces * cents_per_piece : ℚ) / 100

/-- Proof that 500 pieces of gum at 2 cents each costs 10 dollars -/
theorem gum_cost_proof : cost_in_dollars 500 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_gum_cost_proof_l4035_403591


namespace NUMINAMATH_CALUDE_farmer_max_profit_l4035_403530

/-- Represents the farmer's problem of maximizing profit given land and budget constraints -/
theorem farmer_max_profit (total_land : ℝ) (rice_yield peanut_yield : ℝ) 
  (rice_cost peanut_cost : ℝ) (rice_price peanut_price : ℝ) (budget : ℝ) :
  total_land = 2 →
  rice_yield = 6000 →
  peanut_yield = 1500 →
  rice_cost = 3600 →
  peanut_cost = 1200 →
  rice_price = 3 →
  peanut_price = 5 →
  budget = 6000 →
  ∃ (rice_area peanut_area : ℝ),
    rice_area = 1.5 ∧
    peanut_area = 0.5 ∧
    rice_area + peanut_area ≤ total_land ∧
    rice_cost * rice_area + peanut_cost * peanut_area ≤ budget ∧
    ∀ (x y : ℝ),
      x + y ≤ total_land →
      rice_cost * x + peanut_cost * y ≤ budget →
      (rice_price * rice_yield - rice_cost) * x + (peanut_price * peanut_yield - peanut_cost) * y ≤
      (rice_price * rice_yield - rice_cost) * rice_area + (peanut_price * peanut_yield - peanut_cost) * peanut_area :=
by
  sorry


end NUMINAMATH_CALUDE_farmer_max_profit_l4035_403530


namespace NUMINAMATH_CALUDE_simplify_expression_l4035_403546

theorem simplify_expression (r : ℝ) : 100*r - 48*r + 10 = 52*r + 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4035_403546


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4035_403534

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 = 20) →
  (a 3 + a 4 = 40) →
  (a 5 + a 6 = 80) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4035_403534


namespace NUMINAMATH_CALUDE_common_tangent_length_l4035_403592

/-- The length of the common tangent of two externally tangent circles -/
theorem common_tangent_length (R r : ℝ) (hR : R > 0) (hr : r > 0) :
  let d := R + r  -- distance between centers
  2 * Real.sqrt (r * R) = Real.sqrt (d^2 - (R - r)^2) :=
by sorry

end NUMINAMATH_CALUDE_common_tangent_length_l4035_403592


namespace NUMINAMATH_CALUDE_cube_inequality_l4035_403525

theorem cube_inequality (x y a : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : a^x < a^y) : x^3 > y^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_inequality_l4035_403525


namespace NUMINAMATH_CALUDE_digit_sum_problem_l4035_403555

theorem digit_sum_problem (x y z u : ℕ) : 
  x < 10 → y < 10 → z < 10 → u < 10 →
  x ≠ y → x ≠ z → x ≠ u → y ≠ z → y ≠ u → z ≠ u →
  10 * x + y + 10 * z + x = 10 * u + x - (10 * z + x) →
  x + y + z + u = 18 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l4035_403555


namespace NUMINAMATH_CALUDE_pencils_found_l4035_403519

theorem pencils_found (initial bought final misplaced broken : ℕ) : 
  initial = 20 →
  bought = 2 →
  final = 16 →
  misplaced = 7 →
  broken = 3 →
  final = initial - misplaced - broken + bought + (final - (initial - misplaced - broken + bought)) →
  final - (initial - misplaced - broken + bought) = 4 :=
by sorry

end NUMINAMATH_CALUDE_pencils_found_l4035_403519


namespace NUMINAMATH_CALUDE_apple_seedling_survival_probability_l4035_403549

/-- Survival rate data for apple seedlings -/
def survival_data : List (ℕ × ℝ) := [
  (100, 0.81),
  (200, 0.78),
  (500, 0.79),
  (1000, 0.8),
  (2000, 0.8)
]

/-- The estimated probability of survival for apple seedlings after transplantation -/
def estimated_survival_probability : ℝ := 0.8

/-- Theorem stating that the estimated probability of survival is 0.8 -/
theorem apple_seedling_survival_probability :
  estimated_survival_probability = 0.8 :=
sorry

end NUMINAMATH_CALUDE_apple_seedling_survival_probability_l4035_403549


namespace NUMINAMATH_CALUDE_goose_egg_count_l4035_403569

/-- The number of goose eggs laid at a certain pond -/
def total_eggs : ℕ := 1000

/-- The fraction of eggs that hatched -/
def hatch_rate : ℚ := 1/4

/-- The fraction of hatched geese that survived the first month -/
def first_month_survival_rate : ℚ := 4/5

/-- The fraction of geese that survived the first month but did not survive the first year -/
def first_year_mortality_rate : ℚ := 2/5

/-- The number of geese that survived the first year -/
def survivors : ℕ := 120

theorem goose_egg_count :
  total_eggs * hatch_rate * first_month_survival_rate * (1 - first_year_mortality_rate) = survivors := by
  sorry

end NUMINAMATH_CALUDE_goose_egg_count_l4035_403569


namespace NUMINAMATH_CALUDE_parallel_vectors_imply_x_l4035_403512

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_imply_x (x : ℝ) :
  let a : ℝ × ℝ := (1, 2*x + 1)
  let b : ℝ × ℝ := (2, 3)
  parallel a b → x = 1/4 := by sorry

end NUMINAMATH_CALUDE_parallel_vectors_imply_x_l4035_403512


namespace NUMINAMATH_CALUDE_tomato_theorem_l4035_403579

def tomato_problem (initial_tomatoes : ℕ) : ℕ :=
  let after_first_birds := initial_tomatoes - initial_tomatoes / 3
  let after_second_birds := after_first_birds - after_first_birds / 2
  let final_tomatoes := after_second_birds + (after_second_birds + 1) / 2
  final_tomatoes

theorem tomato_theorem : tomato_problem 21 = 11 := by
  sorry

end NUMINAMATH_CALUDE_tomato_theorem_l4035_403579


namespace NUMINAMATH_CALUDE_probability_two_defective_out_of_ten_l4035_403538

/-- Given a set of products with some defective ones, this function calculates
    the probability of randomly selecting a defective product. -/
def probability_defective (total : ℕ) (defective : ℕ) : ℚ :=
  defective / total

/-- Theorem stating that for 10 products with 2 defective ones,
    the probability of randomly selecting a defective product is 1/5. -/
theorem probability_two_defective_out_of_ten :
  probability_defective 10 2 = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_defective_out_of_ten_l4035_403538


namespace NUMINAMATH_CALUDE_factors_of_539_l4035_403528

theorem factors_of_539 : 
  ∃ (p q : Nat), p.Prime ∧ q.Prime ∧ p * q = 539 ∧ p = 13 ∧ q = 41 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_539_l4035_403528


namespace NUMINAMATH_CALUDE_smallest_square_side_length_l4035_403584

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The problem statement -/
theorem smallest_square_side_length 
  (rect1 : Rectangle)
  (rect2 : Rectangle)
  (h1 : rect1.width = 2 ∧ rect1.height = 4)
  (h2 : rect2.width = 4 ∧ rect2.height = 5)
  (h3 : ∀ (s : ℝ), s ≥ 0 → 
    (∃ (x1 y1 x2 y2 : ℝ), 
      0 ≤ x1 ∧ x1 + rect1.width ≤ s ∧
      0 ≤ y1 ∧ y1 + rect1.height ≤ s ∧
      0 ≤ x2 ∧ x2 + rect2.width ≤ s ∧
      0 ≤ y2 ∧ y2 + rect2.height ≤ s ∧
      (x1 + rect1.width ≤ x2 ∨ x2 + rect2.width ≤ x1 ∨
       y1 + rect1.height ≤ y2 ∨ y2 + rect2.height ≤ y1))) :
  (∀ (s : ℝ), s ≥ 0 ∧ 
    (∃ (x1 y1 x2 y2 : ℝ), 
      0 ≤ x1 ∧ x1 + rect1.width ≤ s ∧
      0 ≤ y1 ∧ y1 + rect1.height ≤ s ∧
      0 ≤ x2 ∧ x2 + rect2.width ≤ s ∧
      0 ≤ y2 ∧ y2 + rect2.height ≤ s ∧
      (x1 + rect1.width ≤ x2 ∨ x2 + rect2.width ≤ x1 ∨
       y1 + rect1.height ≤ y2 ∨ y2 + rect2.height ≤ y1)) → s ≥ 6) ∧
  (∃ (x1 y1 x2 y2 : ℝ), 
    0 ≤ x1 ∧ x1 + rect1.width ≤ 6 ∧
    0 ≤ y1 ∧ y1 + rect1.height ≤ 6 ∧
    0 ≤ x2 ∧ x2 + rect2.width ≤ 6 ∧
    0 ≤ y2 ∧ y2 + rect2.height ≤ 6 ∧
    (x1 + rect1.width ≤ x2 ∨ x2 + rect2.width ≤ x1 ∨
     y1 + rect1.height ≤ y2 ∨ y2 + rect2.height ≤ y1)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_side_length_l4035_403584


namespace NUMINAMATH_CALUDE_triangle_problem_l4035_403527

theorem triangle_problem (AB BC : ℝ) (θ : ℝ) (h t : ℝ) 
  (hyp1 : AB = 7)
  (hyp2 : BC = 25)
  (hyp3 : 100 * Real.sin θ = t)
  (hyp4 : h = AB * Real.sin θ) :
  t = 96 ∧ h = 168 / 25 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l4035_403527


namespace NUMINAMATH_CALUDE_target_shopping_expense_l4035_403550

/-- The total amount spent by Christy and Tanya at Target -/
def total_spent (tanya_face_moisturizer_price : ℕ) 
                (tanya_face_moisturizer_count : ℕ)
                (tanya_body_lotion_price : ℕ)
                (tanya_body_lotion_count : ℕ) : ℕ :=
  let tanya_total := tanya_face_moisturizer_price * tanya_face_moisturizer_count + 
                     tanya_body_lotion_price * tanya_body_lotion_count
  tanya_total * 3

theorem target_shopping_expense :
  total_spent 50 2 60 4 = 1020 :=
sorry

end NUMINAMATH_CALUDE_target_shopping_expense_l4035_403550


namespace NUMINAMATH_CALUDE_fraction_simplification_l4035_403593

theorem fraction_simplification (x : ℝ) (h : 2 * x - 3 ≠ 0) :
  (18 * x^4 - 9 * x^3 - 86 * x^2 + 16 * x + 96) / (18 * x^4 - 63 * x^3 + 22 * x^2 + 112 * x - 96) = (2 * x + 3) / (2 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4035_403593


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l4035_403588

theorem intersection_implies_a_equals_one (a : ℝ) : 
  let A : Set ℝ := {-1, 1, 3}
  let B : Set ℝ := {a + 2, a^2 + 4}
  (A ∩ B = {3}) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l4035_403588


namespace NUMINAMATH_CALUDE_cubic_function_property_l4035_403551

/-- Given a cubic function f(x) = ax³ + bx² with a maximum at x = 1 and f(1) = 3, prove that a + b = 3 -/
theorem cubic_function_property (a b : ℝ) : 
  let f := fun (x : ℝ) => a * x^3 + b * x^2
  let f' := fun (x : ℝ) => 3 * a * x^2 + 2 * b * x
  (f 1 = 3) → (f' 1 = 0) → (a + b = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l4035_403551


namespace NUMINAMATH_CALUDE_total_damage_cost_l4035_403576

/-- The cost of damages caused by Jack --/
def cost_of_damages (tire_cost : ℕ) (num_tires : ℕ) (window_cost : ℕ) : ℕ :=
  tire_cost * num_tires + window_cost

/-- Theorem stating the total cost of damages --/
theorem total_damage_cost :
  cost_of_damages 250 3 700 = 1450 := by
  sorry

end NUMINAMATH_CALUDE_total_damage_cost_l4035_403576


namespace NUMINAMATH_CALUDE_chocolate_boxes_given_away_tom_chocolate_boxes_l4035_403580

theorem chocolate_boxes_given_away (total_boxes : ℕ) (pieces_per_box : ℕ) (pieces_left : ℕ) : ℕ :=
  let total_pieces := total_boxes * pieces_per_box
  let pieces_given_away := total_pieces - pieces_left
  pieces_given_away / pieces_per_box

theorem tom_chocolate_boxes :
  chocolate_boxes_given_away 14 3 18 = 8 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_boxes_given_away_tom_chocolate_boxes_l4035_403580


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l4035_403596

-- Problem 1
theorem problem_1 : 
  |(-3)| + (-1)^2021 * (Real.pi - 3.14)^0 - (-1/2)⁻¹ = 4 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) : 
  (x + 3)^2 - (x + 2) * (x - 2) = 6 * x + 13 := by sorry

-- Problem 3
theorem problem_3 (x y : ℝ) : 
  (2*x - y + 3) * (2*x + y - 3) = 4*x^2 - y^2 + 6*y - 9 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l4035_403596


namespace NUMINAMATH_CALUDE_vector_equation_solution_l4035_403548

theorem vector_equation_solution :
  ∃ (a b : ℚ),
    (2 : ℚ) * a + (-2 : ℚ) * b = 10 ∧
    (3 : ℚ) * a + (5 : ℚ) * b = -8 ∧
    a = 17/8 ∧ b = -23/8 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l4035_403548


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l4035_403575

-- Define the sequence terms
def term (k : ℕ) (A B : ℝ) : ℝ := (4 + 3 * (k - 1)) * A + (5 + 3 * (k - 1)) * B

-- State the theorem
theorem arithmetic_sequence_15th_term (a b : ℝ) (A B : ℝ) (h1 : A = Real.log a) (h2 : B = Real.log b) :
  (∀ k : ℕ, k ≥ 1 → k ≤ 3 → term k A B = Real.log (a^(4 + 3*(k-1)) * b^(5 + 3*(k-1)))) →
  term 15 A B = Real.log (b^93) := by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l4035_403575


namespace NUMINAMATH_CALUDE_roger_earnings_l4035_403560

theorem roger_earnings : ∀ (rate : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ),
  rate = 9 →
  total_lawns = 14 →
  forgotten_lawns = 8 →
  (total_lawns - forgotten_lawns) * rate = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_roger_earnings_l4035_403560


namespace NUMINAMATH_CALUDE_problem_solving_distribution_l4035_403526

theorem problem_solving_distribution (x y z : ℕ) : 
  x + y + z = 100 →  -- Total problems
  x + 2*y + 3*z = 180 →  -- Sum of problems solved by each person
  x - z = 20  -- Difference between difficult and easy problems
:= by sorry

end NUMINAMATH_CALUDE_problem_solving_distribution_l4035_403526


namespace NUMINAMATH_CALUDE_mary_has_29_nickels_l4035_403515

/-- Calculates the total number of nickels Mary has after receiving gifts and doing chores. -/
def marys_nickels (initial : ℕ) (from_dad : ℕ) (mom_multiplier : ℕ) (from_chores : ℕ) : ℕ :=
  initial + from_dad + (mom_multiplier * from_dad) + from_chores

/-- Theorem stating that Mary has 29 nickels after all transactions. -/
theorem mary_has_29_nickels : 
  marys_nickels 7 5 3 2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_mary_has_29_nickels_l4035_403515


namespace NUMINAMATH_CALUDE_not_pythagorean_triple_l4035_403567

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem not_pythagorean_triple : 
  (is_pythagorean_triple 3 4 5) ∧ 
  (is_pythagorean_triple 5 12 13) ∧ 
  (is_pythagorean_triple 6 8 10) ∧ 
  ¬(is_pythagorean_triple 7 25 26) := by
  sorry

end NUMINAMATH_CALUDE_not_pythagorean_triple_l4035_403567


namespace NUMINAMATH_CALUDE_inequality_solution_l4035_403568

theorem inequality_solution (x : ℝ) : 
  (x^2 / (x - 2) ≥ 3 / (x + 2) + 7 / 5) ↔ (x > -2 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l4035_403568


namespace NUMINAMATH_CALUDE_hammond_statue_weight_l4035_403557

/-- Given Hammond's marble carving scenario, prove the weight of each remaining statue. -/
theorem hammond_statue_weight :
  -- Total weight of marble block
  let total_weight : ℕ := 80
  -- Weight of first statue
  let first_statue : ℕ := 10
  -- Weight of second statue
  let second_statue : ℕ := 18
  -- Weight of discarded marble
  let discarded : ℕ := 22
  -- Number of statues
  let num_statues : ℕ := 4
  -- Weight of each remaining statue
  let remaining_statue_weight : ℕ := (total_weight - first_statue - second_statue - discarded) / (num_statues - 2)
  -- Proof that each remaining statue weighs 15 pounds
  remaining_statue_weight = 15 := by
  sorry

end NUMINAMATH_CALUDE_hammond_statue_weight_l4035_403557


namespace NUMINAMATH_CALUDE_cubes_not_touching_foil_l4035_403590

/-- Represents a rectangular prism with inner and outer dimensions -/
structure RectangularPrism where
  inner_length : ℕ
  inner_width : ℕ
  inner_height : ℕ
  outer_width : ℕ

/-- Creates a RectangularPrism with the given constraints -/
def create_prism (outer_width : ℕ) : RectangularPrism :=
  { inner_length := (outer_width - 2) / 2,
    inner_width := outer_width - 2,
    inner_height := (outer_width - 2) / 2,
    outer_width := outer_width }

/-- Calculates the number of cubes not touching tin foil -/
def inner_cubes (prism : RectangularPrism) : ℕ :=
  prism.inner_length * prism.inner_width * prism.inner_height

/-- Theorem stating the number of cubes not touching tin foil -/
theorem cubes_not_touching_foil :
  inner_cubes (create_prism 10) = 128 := by
  sorry

#eval inner_cubes (create_prism 10)

end NUMINAMATH_CALUDE_cubes_not_touching_foil_l4035_403590


namespace NUMINAMATH_CALUDE_stamp_collection_theorem_l4035_403511

def stamp_collection_value (total_stamps : ℕ) (sample_stamps : ℕ) (sample_value : ℕ) (bonus_per_set : ℕ) : ℕ :=
  let stamp_value : ℕ := sample_value / sample_stamps
  let total_value : ℕ := total_stamps * stamp_value
  let complete_sets : ℕ := total_stamps / sample_stamps
  let bonus : ℕ := complete_sets * bonus_per_set
  total_value + bonus

theorem stamp_collection_theorem :
  stamp_collection_value 21 7 28 5 = 99 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_theorem_l4035_403511


namespace NUMINAMATH_CALUDE_least_multiplier_for_perfect_square_l4035_403507

def original_number : ℕ := 2^5 * 3^6 * 4^3 * 5^3 * 6^7

theorem least_multiplier_for_perfect_square :
  ∀ n : ℕ, n > 0 → (∃ m : ℕ, (original_number * n) = m^2) →
  15 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_multiplier_for_perfect_square_l4035_403507


namespace NUMINAMATH_CALUDE_base_salary_minimum_l4035_403547

/-- The base salary of Tom's new sales job -/
def base_salary : ℝ := 45000

/-- The salary of Tom's previous job -/
def previous_salary : ℝ := 75000

/-- The commission percentage on each sale -/
def commission_percentage : ℝ := 0.15

/-- The price of each sale -/
def sale_price : ℝ := 750

/-- The minimum number of sales required to not lose money -/
def min_sales : ℝ := 266.67

theorem base_salary_minimum : 
  base_salary + min_sales * (commission_percentage * sale_price) ≥ previous_salary :=
sorry

end NUMINAMATH_CALUDE_base_salary_minimum_l4035_403547


namespace NUMINAMATH_CALUDE_jerrys_age_l4035_403541

/-- Given that Mickey's age is 5 years more than 200% of Jerry's age,
    and Mickey is 21 years old, Jerry's age is 8 years. -/
theorem jerrys_age (mickey_age jerry_age : ℕ) : 
  mickey_age = 2 * jerry_age + 5 →
  mickey_age = 21 →
  jerry_age = 8 := by
sorry

end NUMINAMATH_CALUDE_jerrys_age_l4035_403541


namespace NUMINAMATH_CALUDE_find_y_l4035_403516

def rotation_equivalence (y : ℝ) : Prop :=
  (480 % 360 : ℝ) = (360 - y) % 360 ∧ y < 360

theorem find_y : ∃ y : ℝ, rotation_equivalence y ∧ y = 240 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l4035_403516
