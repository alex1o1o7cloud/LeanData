import Mathlib

namespace NUMINAMATH_CALUDE_teeth_removal_theorem_l249_24983

theorem teeth_removal_theorem :
  let total_teeth : ℕ := 32
  let first_person_removed : ℕ := total_teeth / 4
  let second_person_removed : ℕ := total_teeth * 3 / 8
  let third_person_removed : ℕ := total_teeth / 2
  let fourth_person_removed : ℕ := 4
  first_person_removed + second_person_removed + third_person_removed + fourth_person_removed = 40 := by
  sorry

end NUMINAMATH_CALUDE_teeth_removal_theorem_l249_24983


namespace NUMINAMATH_CALUDE_common_inner_tangent_l249_24962

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 16
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 8*y + 24 = 0

-- Define the proposed tangent line
def tangent_line (x y : ℝ) : Prop := 3*x - 4*y - 20 = 0

-- Theorem statement
theorem common_inner_tangent :
  ∀ x y : ℝ, 
  (circle1 x y ∨ circle2 x y) → 
  (tangent_line x y ↔ 
    (∃ t : ℝ, 
      (circle1 (x + t) (y + t) ∧ tangent_line (x + t) (y + t)) ∨
      (circle2 (x + t) (y + t) ∧ tangent_line (x + t) (y + t))))
  := by sorry

end NUMINAMATH_CALUDE_common_inner_tangent_l249_24962


namespace NUMINAMATH_CALUDE_dividend_problem_l249_24931

theorem dividend_problem (total : ℚ) (a b c : ℚ) 
  (h1 : total = 527)
  (h2 : a = (2/3) * b)
  (h3 : b = (1/4) * c)
  (h4 : a + b + c = total) :
  a = 62 := by
sorry

end NUMINAMATH_CALUDE_dividend_problem_l249_24931


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l249_24956

theorem reciprocal_of_negative_2023 : ((-2023)⁻¹ : ℚ) = -1 / 2023 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_2023_l249_24956


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l249_24977

theorem rectangle_shorter_side (a b d : ℝ) : 
  a / b = 3 / 4 →  -- ratio of sides is 3:4
  d^2 = a^2 + b^2 →  -- Pythagorean theorem
  d = 9 →  -- diagonal is 9
  a = 5.4 :=  -- shorter side is 5.4
by sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l249_24977


namespace NUMINAMATH_CALUDE_book_reading_problem_l249_24941

theorem book_reading_problem (n t k : ℕ) 
  (h1 : (k + 1) * n + k * (k + 1) / 2 = 374)
  (h2 : (k + 1) * t + k * (k + 1) / 2 = 319)
  (h3 : n > 0)
  (h4 : t > 0)
  (h5 : k > 0) :
  n + t = 53 := by
sorry

end NUMINAMATH_CALUDE_book_reading_problem_l249_24941


namespace NUMINAMATH_CALUDE_circles_intersect_l249_24985

/-- The circles x^2 + y^2 = -4y and (x-1)^2 + y^2 = 1 are intersecting -/
theorem circles_intersect : ∃ (x y : ℝ),
  (x^2 + y^2 = -4*y) ∧ ((x-1)^2 + y^2 = 1) := by
  sorry


end NUMINAMATH_CALUDE_circles_intersect_l249_24985


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l249_24996

theorem degree_to_radian_conversion (deg : ℝ) (rad : ℝ) : 
  (180 : ℝ) = π → 240 = (4 / 3 : ℝ) * π := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l249_24996


namespace NUMINAMATH_CALUDE_max_value_expression_l249_24928

theorem max_value_expression (a b c d : ℝ) 
  (ha : a ∈ Set.Icc (-5 : ℝ) 5)
  (hb : b ∈ Set.Icc (-5 : ℝ) 5)
  (hc : c ∈ Set.Icc (-5 : ℝ) 5)
  (hd : d ∈ Set.Icc (-5 : ℝ) 5) :
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 110 ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ),
    a₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    b₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    c₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    d₀ ∈ Set.Icc (-5 : ℝ) 5 ∧
    a₀ + 2*b₀ + c₀ + 2*d₀ - a₀*b₀ - b₀*c₀ - c₀*d₀ - d₀*a₀ = 110 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l249_24928


namespace NUMINAMATH_CALUDE_polygon_formation_and_perimeter_l249_24971

-- Define the structures
structure Triangle where
  A : Point
  B : Point
  C : Point

structure Parallelogram where
  O : Point
  X : Point
  Y : Point
  Z : Point

-- Define the function that creates a parallelogram from two points and O
def createParallelogram (O X Y : Point) : Parallelogram := sorry

-- Define the function that checks if a point is inside a triangle
def isPointInTriangle (p : Point) (t : Triangle) : Prop := sorry

-- Define the function that calculates the perimeter of a triangle
def trianglePerimeter (t : Triangle) : ℝ := sorry

-- Define the function that calculates the perimeter of a polygon
def polygonPerimeter (vertices : List Point) : ℝ := sorry

-- Main theorem
theorem polygon_formation_and_perimeter 
  (ABC DEF : Triangle) (O : Point) : 
  ∃ (polygon : List Point),
    (∀ X Y, isPointInTriangle X ABC → isPointInTriangle Y DEF →
      let p := createParallelogram O X Y
      (p.O ∈ polygon ∧ p.X ∈ polygon ∧ p.Y ∈ polygon ∧ p.Z ∈ polygon)) ∧
    (polygon.length = 6) ∧
    (polygonPerimeter polygon = trianglePerimeter ABC + trianglePerimeter DEF) :=
sorry

end NUMINAMATH_CALUDE_polygon_formation_and_perimeter_l249_24971


namespace NUMINAMATH_CALUDE_decagon_game_outcome_dodecagon_game_outcome_l249_24929

/-- Represents the possible outcomes of the game -/
inductive GameOutcome
| FirstPlayerWins
| SecondPlayerWins

/-- Represents a regular polygon with alternating colored vertices -/
structure ColoredPolygon where
  sides : ℕ
  vertices_alternating_colors : sides > 0

/-- The game played on a colored polygon -/
def polygon_segment_game (p : ColoredPolygon) : GameOutcome :=
  sorry

/-- Theorem stating the outcome for a decagon -/
theorem decagon_game_outcome :
  polygon_segment_game ⟨10, by norm_num⟩ = GameOutcome.SecondPlayerWins :=
sorry

/-- Theorem stating the outcome for a dodecagon -/
theorem dodecagon_game_outcome :
  polygon_segment_game ⟨12, by norm_num⟩ = GameOutcome.FirstPlayerWins :=
sorry

end NUMINAMATH_CALUDE_decagon_game_outcome_dodecagon_game_outcome_l249_24929


namespace NUMINAMATH_CALUDE_ellipse_perpendicular_triangle_area_l249_24932

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/49 + y^2/24 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-5, 0)
def F₂ : ℝ × ℝ := (5, 0)

-- Define perpendicularity condition
def is_perpendicular (m n : ℝ) : Prop := (n / (m + 5)) * (n / (m - 5)) = -1

-- Theorem statement
theorem ellipse_perpendicular_triangle_area (m n : ℝ) :
  is_on_ellipse m n → is_perpendicular m n →
  (1/2 : ℝ) * |10 * n| = 24 := by sorry

end NUMINAMATH_CALUDE_ellipse_perpendicular_triangle_area_l249_24932


namespace NUMINAMATH_CALUDE_combined_cost_theorem_l249_24920

def wallet_cost : ℝ := 22
def purse_cost : ℝ := 4 * wallet_cost - 3

theorem combined_cost_theorem : wallet_cost + purse_cost = 107 := by
  sorry

end NUMINAMATH_CALUDE_combined_cost_theorem_l249_24920


namespace NUMINAMATH_CALUDE_polynomial_real_root_l249_24954

theorem polynomial_real_root (a : ℝ) : ∃ x : ℝ, x^4 - a*x^2 + a*x - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l249_24954


namespace NUMINAMATH_CALUDE_yulia_lemonade_stand_expenses_l249_24969

/-- Yulia's lemonade stand financial calculation -/
theorem yulia_lemonade_stand_expenses 
  (net_profit : ℝ) 
  (lemonade_revenue : ℝ) 
  (babysitting_earnings : ℝ) 
  (h1 : net_profit = 44)
  (h2 : lemonade_revenue = 47)
  (h3 : babysitting_earnings = 31) :
  lemonade_revenue + babysitting_earnings - net_profit = 34 :=
by
  sorry

#check yulia_lemonade_stand_expenses

end NUMINAMATH_CALUDE_yulia_lemonade_stand_expenses_l249_24969


namespace NUMINAMATH_CALUDE_min_value_h_l249_24917

theorem min_value_h (x : ℝ) (hx : x > 0) : x + 1/x + 1/(x + 1/x)^2 ≥ 2.25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_h_l249_24917


namespace NUMINAMATH_CALUDE_sum_greater_than_6_is_random_event_l249_24903

def numbers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def is_sum_greater_than_6 (selection : List ℕ) : Bool :=
  selection.sum > 6

theorem sum_greater_than_6_is_random_event :
  ∃ (selection₁ selection₂ : List ℕ),
    selection₁.length = 3 ∧
    selection₂.length = 3 ∧
    (∀ n ∈ selection₁, n ∈ numbers) ∧
    (∀ n ∈ selection₂, n ∈ numbers) ∧
    is_sum_greater_than_6 selection₁ ∧
    ¬is_sum_greater_than_6 selection₂ :=
by
  sorry

end NUMINAMATH_CALUDE_sum_greater_than_6_is_random_event_l249_24903


namespace NUMINAMATH_CALUDE_a_share_of_profit_l249_24939

/-- Calculates the share of profit for an investor in a partnership business -/
def calculate_share_of_profit (investment_A investment_B investment_C total_profit : ℚ) : ℚ :=
  (investment_A / (investment_A + investment_B + investment_C)) * total_profit

/-- Theorem: A's share of the profit is 3780 given the investments and total profit -/
theorem a_share_of_profit (investment_A investment_B investment_C total_profit : ℚ) 
  (h1 : investment_A = 6300)
  (h2 : investment_B = 4200)
  (h3 : investment_C = 10500)
  (h4 : total_profit = 12600) :
  calculate_share_of_profit investment_A investment_B investment_C total_profit = 3780 := by
  sorry

end NUMINAMATH_CALUDE_a_share_of_profit_l249_24939


namespace NUMINAMATH_CALUDE_edge_projection_max_sum_l249_24943

theorem edge_projection_max_sum (a b : ℝ) : 
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 7 ∧ x^2 + y^2 = 6 ∧ 
   a^2 = x^2 + 1 ∧ b^2 = y^2 + 1) →
  a + b ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_edge_projection_max_sum_l249_24943


namespace NUMINAMATH_CALUDE_stationery_shop_sales_percentage_l249_24972

theorem stationery_shop_sales_percentage (pen_sales pencil_sales marker_sales : ℝ) 
  (h_pen : pen_sales = 25)
  (h_pencil : pencil_sales = 30)
  (h_marker : marker_sales = 20)
  (h_total : pen_sales + pencil_sales + marker_sales + (100 - pen_sales - pencil_sales - marker_sales) = 100) :
  100 - pen_sales - pencil_sales - marker_sales = 25 := by
sorry

end NUMINAMATH_CALUDE_stationery_shop_sales_percentage_l249_24972


namespace NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l249_24947

theorem parabola_hyperbola_intersection (p : ℝ) (m : ℝ) (a : ℝ) (b : ℝ) : 
  p > 0 → 
  m^2 = 2*p*1 →
  (1 - p/2)^2 + m^2 = 5^2 →
  2 * (-b/a) = -1 →
  a = 1/4 := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_intersection_l249_24947


namespace NUMINAMATH_CALUDE_max_red_socks_l249_24965

theorem max_red_socks (total : ℕ) (red : ℕ) (blue : ℕ) : 
  total ≤ 2001 →
  total = red + blue →
  (red * (red - 1) + 2 * red * blue) / (total * (total - 1)) = 1/2 →
  red ≤ 990 :=
sorry

end NUMINAMATH_CALUDE_max_red_socks_l249_24965


namespace NUMINAMATH_CALUDE_part_one_part_two_l249_24914

/-- Part I: Minimum value of m for maximum |f(x)| -/
theorem part_one (a : ℝ) (h_a : a ∈ Set.Icc 4 6) :
  ∃ m : ℝ, m ≥ 6 ∧ ∀ x ∈ Set.Icc 1 m, |x + a / x - 4| ≤ |m + a / m - 4| :=
sorry

/-- Part II: Upper bound for k -/
theorem part_two (a : ℝ) (h_a : a ∈ Set.Icc 1 2) (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 2 4 → x₂ ∈ Set.Icc 2 4 → x₁ < x₂ →
    |x₁ + a / x₁ - 4| - |x₂ + a / x₂ - 4| < k * x₁ + 3 - (k * x₂ + 3)) →
  k ≤ 6 - 4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l249_24914


namespace NUMINAMATH_CALUDE_first_donor_coins_l249_24993

theorem first_donor_coins (d1 d2 d3 d4 : ℕ) : 
  d2 = 2 * d1 →
  d3 = 3 * d2 →
  d4 = 4 * d3 →
  d1 + d2 + d3 + d4 = 132 →
  d1 = 4 := by
sorry

end NUMINAMATH_CALUDE_first_donor_coins_l249_24993


namespace NUMINAMATH_CALUDE_birds_nest_eggs_l249_24988

theorem birds_nest_eggs (x : ℕ) : 
  (2 * x + 3 + 4 = 17) → x = 5 := by sorry

end NUMINAMATH_CALUDE_birds_nest_eggs_l249_24988


namespace NUMINAMATH_CALUDE_older_brother_pocket_money_l249_24919

theorem older_brother_pocket_money
  (total_money : ℕ)
  (difference : ℕ)
  (h1 : total_money = 12000)
  (h2 : difference = 1000) :
  ∃ (younger older : ℕ),
    younger + older = total_money ∧
    older = younger + difference ∧
    older = 6500 := by
  sorry

end NUMINAMATH_CALUDE_older_brother_pocket_money_l249_24919


namespace NUMINAMATH_CALUDE_bowling_team_average_weight_l249_24923

theorem bowling_team_average_weight 
  (initial_players : ℕ) 
  (initial_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) 
  (h1 : initial_players = 7) 
  (h2 : initial_average = 94) 
  (h3 : new_player1_weight = 110) 
  (h4 : new_player2_weight = 60) : 
  (initial_players * initial_average + new_player1_weight + new_player2_weight) / (initial_players + 2) = 92 := by
  sorry

end NUMINAMATH_CALUDE_bowling_team_average_weight_l249_24923


namespace NUMINAMATH_CALUDE_y_squared_value_l249_24937

theorem y_squared_value (x y : ℤ) 
  (eq1 : 4 * x + y = 34) 
  (eq2 : 2 * x - y = 20) : 
  y ^ 2 = 4 := by
sorry

end NUMINAMATH_CALUDE_y_squared_value_l249_24937


namespace NUMINAMATH_CALUDE_teacher_budget_theorem_l249_24905

/-- Calculates the remaining budget for a teacher after purchasing school supplies. -/
def remaining_budget (last_year_budget : ℕ) (this_year_budget : ℕ) (supply1_cost : ℕ) (supply2_cost : ℕ) : ℕ :=
  (last_year_budget + this_year_budget) - (supply1_cost + supply2_cost)

/-- Proves that the remaining budget is 19 given the specific conditions. -/
theorem teacher_budget_theorem :
  remaining_budget 6 50 13 24 = 19 := by
  sorry

end NUMINAMATH_CALUDE_teacher_budget_theorem_l249_24905


namespace NUMINAMATH_CALUDE_power_division_nineteen_l249_24949

theorem power_division_nineteen : 19^11 / 19^5 = 47045881 := by
  sorry

end NUMINAMATH_CALUDE_power_division_nineteen_l249_24949


namespace NUMINAMATH_CALUDE_quadratic_polynomial_solution_l249_24952

/-- A quadratic polynomial -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  is_quadratic : a ≠ 0

/-- Evaluation of a quadratic polynomial at a point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem quadratic_polynomial_solution 
  (f g : QuadraticPolynomial) 
  (h1 : ∀ x, f.eval (g.eval x) = (f.eval x) * (g.eval x))
  (h2 : g.eval 3 = 40) :
  g.a = 1 ∧ g.b = 31/2 ∧ g.c = -31/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_solution_l249_24952


namespace NUMINAMATH_CALUDE_smallest_area_right_triangle_l249_24991

/-- The smallest possible area of a right triangle with sides 6 and 8 is 24 square units -/
theorem smallest_area_right_triangle (a b c : ℝ) : 
  a = 6 → b = 8 → c^2 = a^2 + b^2 → (1/2) * a * b = 24 := by sorry

end NUMINAMATH_CALUDE_smallest_area_right_triangle_l249_24991


namespace NUMINAMATH_CALUDE_sequence_product_l249_24980

theorem sequence_product (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = q * a n) →
  a 4 = 2 →
  a 2 * a 3 * a 5 * a 6 = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_product_l249_24980


namespace NUMINAMATH_CALUDE_lisa_notebook_savings_l249_24978

/-- Calculates the savings when buying notebooks with discounts -/
def notebook_savings (
  quantity : ℕ
  ) (original_price : ℚ
  ) (discount_rate : ℚ
  ) (bulk_discount : ℚ
  ) (bulk_threshold : ℕ
  ) : ℚ :=
  let discounted_price := original_price * (1 - discount_rate)
  let total_without_discount := quantity * original_price
  let total_with_discount := quantity * discounted_price
  let final_total := 
    if quantity > bulk_threshold
    then total_with_discount - bulk_discount
    else total_with_discount
  total_without_discount - final_total

/-- Theorem stating the savings for Lisa's notebook purchase -/
theorem lisa_notebook_savings :
  notebook_savings 8 3 (30/100) 5 7 = 61/5 := by
  sorry

end NUMINAMATH_CALUDE_lisa_notebook_savings_l249_24978


namespace NUMINAMATH_CALUDE_inequality_range_l249_24936

theorem inequality_range (t : ℝ) (h1 : t > 0) :
  (∀ x > 0, Real.exp (2 * t * x) - (Real.log 2 + Real.log x) / t ≥ 0) ↔ t ≥ 1 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l249_24936


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_equality_condition_l249_24989

theorem min_value_of_sum_of_roots (x : ℝ) : 
  Real.sqrt ((x - 2)^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) ≥ 4 * Real.sqrt 2 :=
by sorry

theorem equality_condition (x : ℝ) : 
  Real.sqrt ((x - 2)^2 + (2 - x)^2) + Real.sqrt ((2 - x)^2 + (2 + x)^2) = 4 * Real.sqrt 2 ↔ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_equality_condition_l249_24989


namespace NUMINAMATH_CALUDE_det_of_matrix_is_one_l249_24942

-- Define the determinant formula for a 2x2 matrix
def det_2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Define our specific matrix
def matrix : Matrix (Fin 2) (Fin 2) ℝ := !![5, 7; 2, 3]

-- Theorem statement
theorem det_of_matrix_is_one :
  det_2x2 (matrix 0 0) (matrix 0 1) (matrix 1 0) (matrix 1 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_of_matrix_is_one_l249_24942


namespace NUMINAMATH_CALUDE_new_ratio_after_boarders_join_l249_24911

theorem new_ratio_after_boarders_join (initial_boarders : ℕ) (new_boarders : ℕ) :
  initial_boarders = 60 →
  new_boarders = 15 →
  (2 : ℚ) / 5 = initial_boarders / (initial_boarders * 5 / 2) →
  (1 : ℚ) / 2 = (initial_boarders + new_boarders) / (initial_boarders * 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_new_ratio_after_boarders_join_l249_24911


namespace NUMINAMATH_CALUDE_cube_painting_probability_l249_24974

/-- Represents the three possible colors for painting cube faces -/
inductive Color
  | Black
  | White
  | Red

/-- Represents a cube with six faces -/
structure Cube :=
  (faces : Fin 6 → Color)

/-- Checks if a cube has no adjacent red faces -/
def noAdjacentRed (c : Cube) : Prop := sorry

/-- Counts the number of valid cube paintings -/
def validPaintings : ℕ := sorry

/-- Checks if two cubes can be rotated to look identical -/
def canRotateIdentical (c1 c2 : Cube) : Prop := sorry

/-- Counts the number of ways two cubes can be painted to look identical after rotation -/
def identicalAppearances : ℕ := sorry

/-- The main theorem stating the probability of two cubes being painted and rotatable to look identical -/
theorem cube_painting_probability :
  (identicalAppearances : ℚ) / (validPaintings^2 : ℚ) = 1 / 5776 := by sorry

end NUMINAMATH_CALUDE_cube_painting_probability_l249_24974


namespace NUMINAMATH_CALUDE_max_product_constraint_l249_24998

theorem max_product_constraint (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_sum : 6 * a + 8 * b = 72) :
  a * b ≤ 27 ∧ ∃ (a₀ b₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 6 * a₀ + 8 * b₀ = 72 ∧ a₀ * b₀ = 27 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constraint_l249_24998


namespace NUMINAMATH_CALUDE_parallel_vectors_x_coordinate_l249_24992

/-- Given vectors a, b, and c in ℝ², prove that if a + 2b is parallel to c, 
    then the x-coordinate of c is -15. -/
theorem parallel_vectors_x_coordinate 
  (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (2, -1)) 
  (hc : c.2 = 3) 
  (h_parallel : ∃ (k : ℝ), k ≠ 0 ∧ (a.1 + 2*b.1, a.2 + 2*b.2) = (k * c.1, k * c.2)) : 
  c.1 = -15 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_coordinate_l249_24992


namespace NUMINAMATH_CALUDE_sum_a_b_is_8_l249_24987

/-- A quadrilateral PQRS with specific properties -/
structure Quadrilateral where
  a : ℤ
  b : ℤ
  a_gt_b : a > b
  b_pos : b > 0
  is_rectangle : True  -- We assume PQRS is a rectangle
  area_is_32 : 2 * (a - b).natAbs * (a + b).natAbs = 32

/-- The sum of a and b in a quadrilateral with specific properties is 8 -/
theorem sum_a_b_is_8 (q : Quadrilateral) : q.a + q.b = 8 := by
  sorry

#check sum_a_b_is_8

end NUMINAMATH_CALUDE_sum_a_b_is_8_l249_24987


namespace NUMINAMATH_CALUDE_fourth_side_length_l249_24994

/-- A quadrilateral inscribed in a circle with three equal sides -/
structure InscribedQuadrilateral where
  -- The radius of the circumscribed circle
  r : ℝ
  -- The length of three equal sides
  s : ℝ
  -- Assumption that the radius is 150√2
  h1 : r = 150 * Real.sqrt 2
  -- Assumption that the three equal sides have length 150
  h2 : s = 150

/-- The length of the fourth side of the quadrilateral -/
def fourthSide (q : InscribedQuadrilateral) : ℝ := 375

/-- Theorem stating that the fourth side has length 375 -/
theorem fourth_side_length (q : InscribedQuadrilateral) : 
  fourthSide q = 375 := by sorry

end NUMINAMATH_CALUDE_fourth_side_length_l249_24994


namespace NUMINAMATH_CALUDE_valid_numbers_l249_24940

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_valid_number (abcd : ℕ) : Prop :=
  1000 ≤ abcd ∧ abcd < 10000 ∧
  abcd % 11 = 0 ∧
  (abcd / 100 % 10 + abcd / 10 % 10 = abcd / 1000) ∧
  is_perfect_square ((abcd / 100 % 10) * 10 + (abcd / 10 % 10))

theorem valid_numbers :
  {abcd : ℕ | is_valid_number abcd} = {9812, 1012, 4048, 9361, 9097} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l249_24940


namespace NUMINAMATH_CALUDE_quadrilateral_property_l249_24916

-- Define the quadrilateral and its properties
structure Quadrilateral :=
  (area : ℝ)
  (pq : ℝ)
  (rs : ℝ)
  (d : ℝ)
  (m : ℕ)
  (n : ℕ)
  (p : ℕ)

-- Define the theorem
theorem quadrilateral_property (q : Quadrilateral) : 
  q.area = 15 ∧ q.pq = 6 ∧ q.rs = 8 ∧ q.d^2 = q.m + q.n * Real.sqrt q.p → 
  q.m + q.n + q.p = 81 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_property_l249_24916


namespace NUMINAMATH_CALUDE_sum_of_coefficients_3x_minus_4y_power_20_l249_24967

theorem sum_of_coefficients_3x_minus_4y_power_20 :
  let f : ℝ → ℝ → ℝ := λ x y => (3*x - 4*y)^20
  (f 1 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_3x_minus_4y_power_20_l249_24967


namespace NUMINAMATH_CALUDE_egypt_trip_total_cost_l249_24959

def egypt_trip_cost (base_price upgrade_cost transportation_cost : ℕ) 
                    (individual_discount transportation_discount : ℚ) 
                    (num_people : ℕ) : ℚ :=
  let discounted_tour_price := base_price - individual_discount
  let total_per_person := discounted_tour_price + upgrade_cost
  let discounted_transportation := transportation_cost * (1 - transportation_discount)
  (total_per_person + discounted_transportation) * num_people

theorem egypt_trip_total_cost :
  egypt_trip_cost 147 65 80 14 (1/10) 2 = 540 := by
  sorry

end NUMINAMATH_CALUDE_egypt_trip_total_cost_l249_24959


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_odd_primes_l249_24901

theorem smallest_four_digit_divisible_by_smallest_odd_primes : 
  ∃ (n : ℕ), 
    (1000 ≤ n) ∧ 
    (n < 10000) ∧ 
    (n % 3 = 0) ∧ 
    (n % 5 = 0) ∧ 
    (n % 7 = 0) ∧ 
    (n % 11 = 0) ∧
    (∀ m : ℕ, 
      (1000 ≤ m) ∧ 
      (m < 10000) ∧ 
      (m % 3 = 0) ∧ 
      (m % 5 = 0) ∧ 
      (m % 7 = 0) ∧ 
      (m % 11 = 0) → 
      n ≤ m) ∧
    n = 1155 :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_smallest_odd_primes_l249_24901


namespace NUMINAMATH_CALUDE_one_sixths_in_eleven_thirds_l249_24913

theorem one_sixths_in_eleven_thirds : (11 / 3) / (1 / 6) = 22 := by sorry

end NUMINAMATH_CALUDE_one_sixths_in_eleven_thirds_l249_24913


namespace NUMINAMATH_CALUDE_minibus_capacity_insufficient_l249_24995

theorem minibus_capacity_insufficient (students : ℕ) (bus_capacity : ℕ) (num_buses : ℕ) : 
  students = 300 → 
  bus_capacity = 23 → 
  num_buses = 13 → 
  num_buses * bus_capacity < students := by
sorry

end NUMINAMATH_CALUDE_minibus_capacity_insufficient_l249_24995


namespace NUMINAMATH_CALUDE_box_sales_ratio_l249_24925

theorem box_sales_ratio (thursday_sales : ℕ) 
  (h1 : thursday_sales = 1200)
  (h2 : ∃ wednesday_sales : ℕ, wednesday_sales = 2 * thursday_sales)
  (h3 : ∃ tuesday_sales : ℕ, tuesday_sales = 2 * wednesday_sales) :
  ∃ (tuesday_sales wednesday_sales : ℕ),
    tuesday_sales = 2 * wednesday_sales ∧
    wednesday_sales = 2 * thursday_sales :=
by
  sorry

end NUMINAMATH_CALUDE_box_sales_ratio_l249_24925


namespace NUMINAMATH_CALUDE_cubic_divisibility_l249_24961

theorem cubic_divisibility : ∃ (n : ℕ), n > 0 ∧ 84^3 % n = 0 ∧ n = 592704 := by
  sorry

end NUMINAMATH_CALUDE_cubic_divisibility_l249_24961


namespace NUMINAMATH_CALUDE_subsets_and_sum_of_M_l249_24990

def M : Finset Nat := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

theorem subsets_and_sum_of_M :
  (Finset.powerset M).card = 2^10 ∧
  (Finset.powerset M).sum (fun s => s.sum id) = 55 * 2^9 := by
  sorry

end NUMINAMATH_CALUDE_subsets_and_sum_of_M_l249_24990


namespace NUMINAMATH_CALUDE_product_of_logs_l249_24912

theorem product_of_logs (a b : ℕ+) : 
  (b - a = 870) →
  (Real.log b / Real.log a = 2) →
  (a + b : ℕ) = 930 := by
sorry

end NUMINAMATH_CALUDE_product_of_logs_l249_24912


namespace NUMINAMATH_CALUDE_triangle_area_l249_24970

def a : ℝ × ℝ := (3, -2)
def b : ℝ × ℝ := (-1, 5)

theorem triangle_area : 
  let det := a.1 * b.2 - a.2 * b.1
  (1/2 : ℝ) * |det| = 13/2 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l249_24970


namespace NUMINAMATH_CALUDE_range_of_a_l249_24999

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → 
  a ∈ Set.Iic 1 ∪ Set.Ici 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l249_24999


namespace NUMINAMATH_CALUDE_quadratic_properties_l249_24982

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the derivative of the quadratic function
def quadratic_derivative (a b : ℝ) (x : ℝ) : ℝ := 2 * a * x + b

theorem quadratic_properties (a b c : ℝ) :
  -- The function has a minimum at x = 2
  (quadratic_derivative a b 2 = 0) →
  -- The function intersects x-axis at x₁ and x₂
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0) →
  -- tan(CAO) - tan(CBO) = 1
  (∃ x₁ x₂ : ℝ, x₁ < 0 ∧ 0 < x₂ ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0 ∧
    c / x₁ - (-c / x₂) = 1) →
  -- Conclusions
  (b + 4 * a = 0) ∧ (a = 1/4) ∧ (b = -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l249_24982


namespace NUMINAMATH_CALUDE_inequality_proof_l249_24960

theorem inequality_proof (a b : ℤ) 
  (h1 : a > b) 
  (h2 : b > 1) 
  (h3 : (a + b) ∣ (a * b + 1)) 
  (h4 : (a - b) ∣ (a * b - 1)) : 
  a < Real.sqrt 3 * b := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l249_24960


namespace NUMINAMATH_CALUDE_walts_investment_rate_l249_24997

/-- Given Walt's investment scenario, prove that the unknown interest rate is 9% -/
theorem walts_investment_rate : 
  ∀ (total_extra : ℝ) (total_interest : ℝ) (known_amount : ℝ) (known_rate : ℝ),
  total_extra = 9000 →
  total_interest = 770 →
  known_amount = 4000 →
  known_rate = 0.08 →
  ∃ (unknown_rate : ℝ),
    unknown_rate = 0.09 ∧
    total_interest = known_amount * known_rate + (total_extra - known_amount) * unknown_rate :=
by
  sorry

#check walts_investment_rate

end NUMINAMATH_CALUDE_walts_investment_rate_l249_24997


namespace NUMINAMATH_CALUDE_modulus_of_z_l249_24933

theorem modulus_of_z (z : ℂ) (h : z * (2 - 3*I) = 6 + 4*I) : Complex.abs z = 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l249_24933


namespace NUMINAMATH_CALUDE_equation_solution_l249_24910

theorem equation_solution (y z w : ℝ) :
  let f : ℝ → ℝ := λ x => (x + y) / (y + z) - (z + w) / (w + x)
  let sol₁ := (-(w + y) + Real.sqrt ((w + y)^2 + 4*(z - w)*(z - y))) / 2
  let sol₂ := (-(w + y) - Real.sqrt ((w + y)^2 + 4*(z - w)*(z - y))) / 2
  (∀ x, f x = 0 ↔ x = sol₁ ∨ x = sol₂) ∧ (f sol₁ = 0 ∧ f sol₂ = 0) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l249_24910


namespace NUMINAMATH_CALUDE_probability_of_choosing_circle_l249_24924

theorem probability_of_choosing_circle (total : ℕ) (circles : ℕ) 
  (h1 : total = 12) (h2 : circles = 5) : 
  (circles : ℚ) / total = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_choosing_circle_l249_24924


namespace NUMINAMATH_CALUDE_expression_value_theorem_l249_24944

theorem expression_value_theorem (x : ℝ) (h : x = Real.sqrt (19 - 8 * Real.sqrt 3)) :
  (x^4 - 6*x^3 - 2*x^2 + 18*x + 23) / (x^2 - 8*x + 15) = 5 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_theorem_l249_24944


namespace NUMINAMATH_CALUDE_means_and_sum_of_squares_l249_24922

theorem means_and_sum_of_squares
  (x y z : ℝ)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 7)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 385.5 := by
  sorry

end NUMINAMATH_CALUDE_means_and_sum_of_squares_l249_24922


namespace NUMINAMATH_CALUDE_cube_volume_problem_l249_24964

theorem cube_volume_problem (s : ℝ) : 
  s > 0 →
  (s - 2) * s * (s + 2) = s^3 - 12 →
  s^3 = 27 := by
sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l249_24964


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l249_24963

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l249_24963


namespace NUMINAMATH_CALUDE_raghu_investment_l249_24906

theorem raghu_investment (raghu trishul vishal : ℝ) : 
  vishal = 1.1 * trishul →
  trishul = 0.9 * raghu →
  raghu + trishul + vishal = 6358 →
  raghu = 2200 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l249_24906


namespace NUMINAMATH_CALUDE_inverse_function_value_l249_24957

noncomputable def f (x : ℝ) : ℝ := x / (2 * x + 1)

noncomputable def f_inv : ℝ → ℝ := Function.invFun f

theorem inverse_function_value :
  f_inv 2 = -2/3 :=
sorry

end NUMINAMATH_CALUDE_inverse_function_value_l249_24957


namespace NUMINAMATH_CALUDE_impossible_to_reach_all_threes_l249_24902

/-- Represents the state of the game at any point --/
structure GameState where
  numPiles : ℕ
  totalTokens : ℕ

/-- The invariant of the game --/
def invariant (state : GameState) : ℕ :=
  state.numPiles + state.totalTokens

/-- The initial state of the game --/
def initialState : GameState :=
  { numPiles := 1, totalTokens := 1001 }

/-- Theorem stating the impossibility of reaching a state with only piles of 3 tokens --/
theorem impossible_to_reach_all_threes :
  ¬∃ (k : ℕ), invariant initialState = 4 * k :=
sorry

end NUMINAMATH_CALUDE_impossible_to_reach_all_threes_l249_24902


namespace NUMINAMATH_CALUDE_extreme_value_condition_l249_24979

/-- If f(x) = m cos x + (1/2) sin 2x reaches an extreme value at x = π/4, then m = 0 -/
theorem extreme_value_condition (m : ℝ) : 
  let f := fun (x : ℝ) => m * Real.cos x + (1/2) * Real.sin (2*x)
  (∃ (ε : ℝ), ∀ (h : ℝ), 0 < |h| → |h| < ε → f (π/4 + h) ≤ f (π/4)) →
  m = 0 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l249_24979


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l249_24981

theorem simplify_trig_expression : 
  Real.sqrt (1 - Real.sin (160 * π / 180) ^ 2) = Real.cos (20 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l249_24981


namespace NUMINAMATH_CALUDE_camping_trip_percentage_l249_24926

/-- Given a school where:
    - 20% of students went to the camping trip and took more than $100
    - 75% of students who went to the camping trip did not take more than $100
    Prove that 80% of all students went on the camping trip. -/
theorem camping_trip_percentage 
  (total_students : ℕ) 
  (students_more_than_100 : ℕ) 
  (students_not_more_than_100 : ℕ) 
  (h1 : students_more_than_100 = (20 : ℕ) * total_students / 100)
  (h2 : students_not_more_than_100 = (75 : ℕ) * (students_more_than_100 + students_not_more_than_100) / 100) :
  students_more_than_100 + students_not_more_than_100 = (80 : ℕ) * total_students / 100 := by
  sorry

end NUMINAMATH_CALUDE_camping_trip_percentage_l249_24926


namespace NUMINAMATH_CALUDE_line_through_parabola_vertex_l249_24921

/-- The number of values of a for which the line y = x + a passes through
    the vertex of the parabola y = x^2 - 2ax + a^2 -/
theorem line_through_parabola_vertex :
  ∃! a : ℝ, ∀ x y : ℝ,
    (y = x + a) ∧ (y = x^2 - 2*a*x + a^2) →
    (x = a ∧ y = 0) := by sorry

end NUMINAMATH_CALUDE_line_through_parabola_vertex_l249_24921


namespace NUMINAMATH_CALUDE_markup_markdown_l249_24953

theorem markup_markdown (original_price : ℝ) (markup1 markup2 markup3 markdown : ℝ) : 
  markup1 = 0.1 →
  markup2 = 0.1 →
  markup3 = 0.05 →
  original_price > 0 →
  original_price * (1 + markup1) * (1 + markup2) * (1 + markup3) * (1 - markdown) = original_price →
  ∀ x : ℕ, x < 22 → (1 - (x : ℝ) / 100) > 1 - markdown :=
by sorry

end NUMINAMATH_CALUDE_markup_markdown_l249_24953


namespace NUMINAMATH_CALUDE_inequality_proofs_l249_24908

theorem inequality_proofs 
  (a b c d : ℝ) 
  (hab : a > b) 
  (hcd : c > d) 
  (hac2bc2 : a * c^2 < b * c^2) 
  (hab_pos : a > b ∧ b > 0) 
  (hc_pos : c > 0) : 
  (a + c > b + d) ∧ 
  (a < b) ∧ 
  ((b + c) / (a + c) > b / a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proofs_l249_24908


namespace NUMINAMATH_CALUDE_minimum_garden_width_l249_24904

theorem minimum_garden_width (w : ℝ) (l : ℝ) :
  w > 0 →
  l = w + 10 →
  w * l ≥ 120 →
  w ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_minimum_garden_width_l249_24904


namespace NUMINAMATH_CALUDE_hollow_cube_side_length_l249_24927

/-- Represents the number of cubes used to create a hollow cube -/
def hollow_cube_cubes (n : ℕ) : ℕ := 6 * n^2 - (n^2 + 4 * (n - 2))

/-- Theorem stating that if 98 cubes are used to make a hollow cube, its side length is 9 -/
theorem hollow_cube_side_length :
  ∃ (n : ℕ), hollow_cube_cubes n = 98 ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_hollow_cube_side_length_l249_24927


namespace NUMINAMATH_CALUDE_blueberry_jelly_amount_l249_24984

/-- The amount of strawberry jelly in grams -/
def strawberry_jelly : ℕ := 1792

/-- The total amount of jelly in grams -/
def total_jelly : ℕ := 6310

/-- The amount of blueberry jelly in grams -/
def blueberry_jelly : ℕ := total_jelly - strawberry_jelly

theorem blueberry_jelly_amount : blueberry_jelly = 4518 := by
  sorry

end NUMINAMATH_CALUDE_blueberry_jelly_amount_l249_24984


namespace NUMINAMATH_CALUDE_percent_problem_l249_24968

theorem percent_problem (x : ℝ) : (0.15 * 40 = 0.25 * x + 2) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l249_24968


namespace NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l249_24946

theorem product_of_sum_and_sum_of_cubes (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 100) : 
  a * b = -3 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_sum_of_cubes_l249_24946


namespace NUMINAMATH_CALUDE_line_x_axis_intersection_l249_24918

/-- The line equation 2y + 5x = 15 -/
def line_equation (x y : ℝ) : Prop := 2 * y + 5 * x = 15

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (3, 0)

theorem line_x_axis_intersection :
  let (x, y) := intersection_point
  line_equation x y ∧ on_x_axis x y :=
by sorry

end NUMINAMATH_CALUDE_line_x_axis_intersection_l249_24918


namespace NUMINAMATH_CALUDE_white_surface_fraction_is_5_16_l249_24986

/-- Represents a cube with given edge length -/
structure Cube :=
  (edge : ℕ)

/-- Represents the large cube constructed from smaller cubes -/
structure LargeCube :=
  (edge : ℕ)
  (smallCubes : ℕ)
  (redCubes : ℕ)
  (whiteCubes : ℕ)

/-- Calculates the surface area of a cube -/
def surfaceArea (c : Cube) : ℕ :=
  6 * c.edge * c.edge

/-- Calculates the fraction of white surface area -/
def whiteSurfaceFraction (lc : LargeCube) : ℚ :=
  sorry

/-- Theorem stating the fraction of white surface area -/
theorem white_surface_fraction_is_5_16 (lc : LargeCube) 
  (h1 : lc.edge = 4)
  (h2 : lc.smallCubes = 64)
  (h3 : lc.redCubes = 48)
  (h4 : lc.whiteCubes = 16) :
  whiteSurfaceFraction lc = 5 / 16 :=
sorry

end NUMINAMATH_CALUDE_white_surface_fraction_is_5_16_l249_24986


namespace NUMINAMATH_CALUDE_red_sweets_count_l249_24975

theorem red_sweets_count (total : ℕ) (green : ℕ) (neither : ℕ) (red : ℕ) 
  (h1 : total = 285)
  (h2 : green = 59)
  (h3 : neither = 177)
  (h4 : total = red + green + neither) :
  red = 49 := by
  sorry

end NUMINAMATH_CALUDE_red_sweets_count_l249_24975


namespace NUMINAMATH_CALUDE_lillian_candy_distribution_l249_24934

theorem lillian_candy_distribution (initial_candies : ℕ) 
  (father_multiplier : ℕ) (num_friends : ℕ) : 
  initial_candies = 205 → 
  father_multiplier = 2 → 
  num_friends = 7 → 
  (initial_candies + father_multiplier * initial_candies) / num_friends = 87 := by
  sorry

end NUMINAMATH_CALUDE_lillian_candy_distribution_l249_24934


namespace NUMINAMATH_CALUDE_wednesday_dressing_time_l249_24930

/-- Represents the dressing times for each day of the school week -/
structure DressingTimes where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- Calculates the average dressing time for the week -/
def weekAverage (times : DressingTimes) : ℚ :=
  (times.monday + times.tuesday + times.wednesday + times.thursday + times.friday) / 5

/-- Theorem: Given the dressing times for Monday, Tuesday, Thursday, and Friday,
    and the old average dressing time, the dressing time for Wednesday must be 3 minutes
    to maintain the same average over the entire week. -/
theorem wednesday_dressing_time
  (times : DressingTimes)
  (h_monday : times.monday = 2)
  (h_tuesday : times.tuesday = 4)
  (h_thursday : times.thursday = 4)
  (h_friday : times.friday = 2)
  (h_old_avg : weekAverage times = 3) :
  times.wednesday = 3 := by
  sorry

#check wednesday_dressing_time

end NUMINAMATH_CALUDE_wednesday_dressing_time_l249_24930


namespace NUMINAMATH_CALUDE_three_integers_sum_l249_24966

theorem three_integers_sum (a b c : ℕ+) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a * b * c = 125 →
  (a : ℕ) + b + c = 31 := by
  sorry

end NUMINAMATH_CALUDE_three_integers_sum_l249_24966


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l249_24955

/-- Given a geometric sequence {a_n} with S_3 = 9/2 and a_3 = 3/2, prove that the common ratio q satisfies q = 1 or q = -1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (S : ℕ → ℚ) 
  (h1 : S 3 = 9/2) 
  (h2 : a 3 = 3/2) : 
  ∃ q : ℚ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ (q = 1 ∨ q = -1/2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l249_24955


namespace NUMINAMATH_CALUDE_sum_of_roots_equation_l249_24938

theorem sum_of_roots_equation (x : ℝ) : 
  (∃ a b : ℝ, x^2 - 5*x + 7 = 9 ∧ x = a ∨ x = b) → a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equation_l249_24938


namespace NUMINAMATH_CALUDE_intersection_M_N_l249_24948

def M : Set ℝ := {-1, 0, 1}
def N : Set ℝ := {x : ℝ | x^2 ≤ x}

theorem intersection_M_N : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l249_24948


namespace NUMINAMATH_CALUDE_hyperbola_and_k_range_l249_24945

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the hyperbola C
def hyperbola_C (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the dot product condition
def dot_product_condition (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 > 2

theorem hyperbola_and_k_range :
  ∃ (a b c : ℝ),
    (∀ x y, ellipse x y ↔ x^2 / a + y^2 / b = 1) ∧
    (c^2 = a / 2) ∧
    (∀ x y, hyperbola_C x y ↔ x^2 / 3 - y^2 = 1) ∧
    (∀ k,
      (∃ x1 y1 x2 y2,
        x1 ≠ x2 ∧
        hyperbola_C x1 y1 ∧
        hyperbola_C x2 y2 ∧
        line_l k x1 y1 ∧
        line_l k x2 y2 ∧
        dot_product_condition x1 y1 x2 y2) ↔
      (k ∈ Set.Ioo (-1 : ℝ) (-Real.sqrt 3 / 3) ∪ Set.Ioo (Real.sqrt 3 / 3) 1)) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_and_k_range_l249_24945


namespace NUMINAMATH_CALUDE_range_of_sum_l249_24935

theorem range_of_sum (a b : ℝ) (ha : -2 < a ∧ a < -1) (hb : -1 < b ∧ b < 0) :
  -3 < a + b ∧ a + b < -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l249_24935


namespace NUMINAMATH_CALUDE_lynn_travel_time_l249_24907

-- Define the problem parameters
def walk_fraction : ℚ := 1/3
def bike_fraction : ℚ := 2/3
def bike_speed_multiplier : ℚ := 4
def walk_time : ℚ := 9

-- Define the theorem
theorem lynn_travel_time :
  let bike_time := walk_time / bike_speed_multiplier
  walk_time + bike_time = 11.25 := by
  sorry


end NUMINAMATH_CALUDE_lynn_travel_time_l249_24907


namespace NUMINAMATH_CALUDE_gulliver_kefir_consumption_l249_24950

/-- Represents the total number of bottles of kefir Gulliver drinks -/
def total_kefir_bottles (initial_money : ℕ) (initial_price : ℕ) : ℕ :=
  initial_money * 6 / (7 * initial_price)

/-- Theorem stating the total number of kefir bottles Gulliver drinks -/
theorem gulliver_kefir_consumption :
  total_kefir_bottles 7000000 7 = 1166666 := by
  sorry

#eval total_kefir_bottles 7000000 7

end NUMINAMATH_CALUDE_gulliver_kefir_consumption_l249_24950


namespace NUMINAMATH_CALUDE_expansion_coefficients_l249_24909

theorem expansion_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 - 2*x)^7 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ = 1 ∧ a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -128) := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l249_24909


namespace NUMINAMATH_CALUDE_intersection_sum_l249_24915

/-- Given two graphs y = -2|x-a| + b and y = 2|x-c| + d intersecting at (1, 6) and (5, 2), prove a + c = 6 -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -2*|x - a| + b = 2*|x - c| + d → x = 1 ∧ -2*|x - a| + b = 6 ∨ x = 5 ∧ -2*|x - a| + b = 2) →
  a + c = 6 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l249_24915


namespace NUMINAMATH_CALUDE_inequality_solution_set_l249_24900

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 5*a)*(x + a) > 0} = {x : ℝ | x < 5*a ∨ x > -a} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l249_24900


namespace NUMINAMATH_CALUDE_rectangular_field_area_l249_24976

/-- Given a rectangular field with one side of 30 feet and three sides fenced using 
    a total of 78 feet of fencing, prove that the area of the field is 720 square feet. -/
theorem rectangular_field_area (L W : ℝ) : 
  L = 30 →  -- Length of uncovered side
  2 * W + L = 78 →  -- Total fencing equation
  L * W = 720 :=  -- Area of the field
by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l249_24976


namespace NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l249_24958

theorem cos_x_plus_2y_equals_one 
  (x y a : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4)) 
  (h2 : y ∈ Set.Icc (-π/4) (π/4)) 
  (h3 : x^3 + Real.sin x - 2*a = 0) 
  (h4 : 4*y^3 + (1/2) * Real.sin (2*y) + a = 0) : 
  Real.cos (x + 2*y) = 1 := by
sorry

end NUMINAMATH_CALUDE_cos_x_plus_2y_equals_one_l249_24958


namespace NUMINAMATH_CALUDE_square_root_of_difference_l249_24951

theorem square_root_of_difference : 
  Real.sqrt (20212020 * 20202021 - 20212021 * 20202020) = 100 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_difference_l249_24951


namespace NUMINAMATH_CALUDE_binary_sum_equals_158_l249_24973

/-- Converts a binary number (represented as a list of bits) to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The first binary number 1010101₂ -/
def binary1 : List Bool := [true, false, true, false, true, false, true]

/-- The second binary number 1001001₂ -/
def binary2 : List Bool := [true, false, false, true, false, false, true]

/-- Theorem stating that the sum of 1010101₂ and 1001001₂ is 158 in decimal -/
theorem binary_sum_equals_158 :
  binaryToDecimal binary1 + binaryToDecimal binary2 = 158 := by
  sorry

end NUMINAMATH_CALUDE_binary_sum_equals_158_l249_24973
