import Mathlib

namespace NUMINAMATH_CALUDE_parrot_silence_explanation_l2969_296984

-- Define the parrot type
structure Parrot where
  repeats_heard_words : Bool
  is_silent : Bool

-- Define the environment
structure Environment where
  words_spoken : Bool

-- Define the theorem
theorem parrot_silence_explanation (p : Parrot) (e : Environment) :
  p.repeats_heard_words ∧ p.is_silent →
  (¬e.words_spoken ∨ ¬p.repeats_heard_words) :=
by
  sorry

-- The negation of repeats_heard_words represents deafness

end NUMINAMATH_CALUDE_parrot_silence_explanation_l2969_296984


namespace NUMINAMATH_CALUDE_parabola_perpendicular_line_passes_through_point_l2969_296923

/-- The parabola y = x^2 -/
def parabola (p : ℝ × ℝ) : Prop := p.2 = p.1^2

/-- Two points are different -/
def different (p q : ℝ × ℝ) : Prop := p ≠ q

/-- A point is not the origin -/
def not_origin (p : ℝ × ℝ) : Prop := p ≠ (0, 0)

/-- Two vectors are perpendicular -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- A point lies on a line defined by two other points -/
def on_line (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, r = (1 - t) • p + t • q

theorem parabola_perpendicular_line_passes_through_point
  (A B : ℝ × ℝ)
  (h_parabola_A : parabola A)
  (h_parabola_B : parabola B)
  (h_different : different A B)
  (h_not_origin_A : not_origin A)
  (h_not_origin_B : not_origin B)
  (h_perpendicular : perpendicular A B) :
  on_line A B (0, 1) :=
sorry

end NUMINAMATH_CALUDE_parabola_perpendicular_line_passes_through_point_l2969_296923


namespace NUMINAMATH_CALUDE_artist_painting_rate_l2969_296967

/-- Proves that given the specified conditions, the artist can paint 1.5 square meters per hour -/
theorem artist_painting_rate 
  (mural_length : ℝ) 
  (mural_width : ℝ) 
  (paint_cost_per_sqm : ℝ) 
  (artist_hourly_rate : ℝ) 
  (total_mural_cost : ℝ) 
  (h1 : mural_length = 6) 
  (h2 : mural_width = 3) 
  (h3 : paint_cost_per_sqm = 4) 
  (h4 : artist_hourly_rate = 10) 
  (h5 : total_mural_cost = 192) : 
  (mural_length * mural_width) / ((total_mural_cost - (paint_cost_per_sqm * mural_length * mural_width)) / artist_hourly_rate) = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_artist_painting_rate_l2969_296967


namespace NUMINAMATH_CALUDE_square_area_from_oblique_projection_l2969_296921

/-- Represents a square in 2D space -/
structure Square where
  side_length : ℝ
  area : ℝ := side_length ^ 2

/-- Represents a parallelogram in 2D space -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ

/-- Represents an oblique projection transformation -/
def obliqueProjection (s : Square) : Parallelogram :=
  sorry

theorem square_area_from_oblique_projection 
  (s : Square) 
  (p : Parallelogram) 
  (h1 : p = obliqueProjection s) 
  (h2 : p.side1 = 4 ∨ p.side2 = 4) : 
  s.area = 16 ∨ s.area = 64 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_oblique_projection_l2969_296921


namespace NUMINAMATH_CALUDE_inequality_comparison_l2969_296900

theorem inequality_comparison :
  (-14 ≤ 0) ∧
  (-2.1 ≤ -2.01) ∧
  (1/2 ≥ -1/3) ∧
  (-0.6 > -4/5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_comparison_l2969_296900


namespace NUMINAMATH_CALUDE_course_selection_plans_l2969_296911

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of course selection plans -/
def coursePlans (totalCourses restrictedCourses coursesToChoose : ℕ) : ℕ :=
  choose (totalCourses - restrictedCourses) coursesToChoose + 
  restrictedCourses * choose (totalCourses - restrictedCourses) (coursesToChoose - 1)

theorem course_selection_plans :
  coursePlans 8 2 5 = 36 := by sorry

end NUMINAMATH_CALUDE_course_selection_plans_l2969_296911


namespace NUMINAMATH_CALUDE_arrangements_of_opening_rooms_l2969_296991

theorem arrangements_of_opening_rooms (n : ℕ) (hn : n = 6) :
  (Finset.sum (Finset.range 5) (fun k => Nat.choose n (k + 2))) = (2^n - (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_arrangements_of_opening_rooms_l2969_296991


namespace NUMINAMATH_CALUDE_one_third_to_fifth_power_l2969_296932

theorem one_third_to_fifth_power :
  (1 / 3 : ℚ) ^ 5 = 1 / 243 := by sorry

end NUMINAMATH_CALUDE_one_third_to_fifth_power_l2969_296932


namespace NUMINAMATH_CALUDE_det_specific_matrix_l2969_296971

theorem det_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 7; 2, 3]
  Matrix.det A = 1 := by
  sorry

end NUMINAMATH_CALUDE_det_specific_matrix_l2969_296971


namespace NUMINAMATH_CALUDE_adams_bank_account_l2969_296903

/-- Adam's bank account problem -/
theorem adams_bank_account 
  (initial_savings : ℚ) 
  (monday_earnings : ℚ) 
  (tuesday_earnings : ℚ) 
  (wednesday_spending : ℚ) 
  (h1 : initial_savings = 1579.37)
  (h2 : monday_earnings = 21.85)
  (h3 : tuesday_earnings = 33.28)
  (h4 : wednesday_spending = 87.41) :
  initial_savings + monday_earnings + tuesday_earnings - wednesday_spending = 1547.09 := by
  sorry

end NUMINAMATH_CALUDE_adams_bank_account_l2969_296903


namespace NUMINAMATH_CALUDE_one_story_height_l2969_296908

-- Define the parameters
def stories : ℕ := 6
def rope_length : ℝ := 20
def loss_percentage : ℝ := 0.25
def num_ropes : ℕ := 4

-- Define the theorem
theorem one_story_height :
  let total_usable_length := (1 - loss_percentage) * rope_length * num_ropes
  let story_height := total_usable_length / stories
  story_height = 10 := by sorry

end NUMINAMATH_CALUDE_one_story_height_l2969_296908


namespace NUMINAMATH_CALUDE_fruit_count_l2969_296955

/-- Given:
  1. If each bag contains 5 oranges and 7 apples, after packing all the apples, there will be 1 orange left.
  2. If each bag contains 9 oranges and 7 apples, after packing all the oranges, there will be 21 apples left.
Prove that the total number of oranges and apples is 85. -/
theorem fruit_count (oranges apples : ℕ) 
  (h1 : ∃ m : ℕ, oranges = 5 * m + 1 ∧ apples = 7 * m)
  (h2 : ∃ n : ℕ, oranges = 9 * n ∧ apples = 7 * n + 21) :
  oranges + apples = 85 := by
sorry

end NUMINAMATH_CALUDE_fruit_count_l2969_296955


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l2969_296963

/-- Represents the monthly sales quantity as a function of price -/
def sales_quantity (x : ℝ) : ℝ := -10000 * x + 80000

/-- Represents the monthly profit as a function of price -/
def monthly_profit (x : ℝ) : ℝ := (x - 4) * (sales_quantity x)

theorem optimal_price_and_profit :
  let price_1 : ℝ := 5
  let quantity_1 : ℝ := 30000
  let price_2 : ℝ := 6
  let quantity_2 : ℝ := 20000
  let unit_cost : ℝ := 4
  
  -- The sales quantity function is correct
  (∀ x, sales_quantity x = -10000 * x + 80000) ∧
  
  -- The function satisfies the given points
  (sales_quantity price_1 = quantity_1) ∧
  (sales_quantity price_2 = quantity_2) ∧
  
  -- The optimal price is 6
  (∀ x, monthly_profit x ≤ monthly_profit 6) ∧
  
  -- The maximum monthly profit is 40000
  (monthly_profit 6 = 40000) := by
    sorry

end NUMINAMATH_CALUDE_optimal_price_and_profit_l2969_296963


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2969_296945

-- Define the inequality
def inequality (x : ℝ) : Prop := |2*x - 1| < 1

-- Define the solution set
def solution_set : Set ℝ := {x | 0 < x ∧ x < 1}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2969_296945


namespace NUMINAMATH_CALUDE_function_equality_l2969_296934

theorem function_equality (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (2 * x + f y) = x + y + f x) → 
  (∀ x : ℝ, f x = x) := by
sorry

end NUMINAMATH_CALUDE_function_equality_l2969_296934


namespace NUMINAMATH_CALUDE_store_revenue_l2969_296940

theorem store_revenue (N D J : ℝ) 
  (h1 : N = (3/5) * D) 
  (h2 : D = (20/7) * ((N + J) / 2)) : 
  J = (1/6) * N := by
sorry

end NUMINAMATH_CALUDE_store_revenue_l2969_296940


namespace NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l2969_296959

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Define the theorem for part I
theorem solution_set_f_positive :
  {x : ℝ | f x > 0} = {x : ℝ | x < -3 ∨ x > 1/3} := by sorry

-- Define the theorem for part II
theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, |m + 1| ≥ f x + 3*|x - 2|) ↔ m ≤ -6 ∨ m ≥ 4 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_positive_range_of_m_l2969_296959


namespace NUMINAMATH_CALUDE_gear_speed_proportion_l2969_296927

/-- Represents a gear with a number of teeth and angular speed -/
structure Gear where
  teeth : ℕ
  speed : ℝ

/-- Represents a system of four meshed gears -/
structure GearSystem where
  A : Gear
  B : Gear
  C : Gear
  D : Gear
  x : ℕ
  y : ℕ
  z : ℕ
  w : ℕ
  mesh_correctly : A.teeth * A.speed = B.teeth * B.speed ∧
                   B.teeth * B.speed = C.teeth * C.speed ∧
                   C.teeth * C.speed = D.teeth * D.speed

/-- Theorem stating the proportion of angular speeds in a gear system -/
theorem gear_speed_proportion (gs : GearSystem)
  (hA : gs.A.teeth = 10 * gs.x)
  (hB : gs.B.teeth = 15 * gs.y)
  (hC : gs.C.teeth = 12 * gs.z)
  (hD : gs.D.teeth = 20 * gs.w) :
  ∃ (k : ℝ), k > 0 ∧
    gs.A.speed = k * (12 * gs.y * gs.z * gs.w : ℝ) ∧
    gs.B.speed = k * (8 * gs.x * gs.z * gs.w : ℝ) ∧
    gs.C.speed = k * (10 * gs.x * gs.y * gs.w : ℝ) ∧
    gs.D.speed = k * (6 * gs.x * gs.y * gs.z : ℝ) :=
sorry

end NUMINAMATH_CALUDE_gear_speed_proportion_l2969_296927


namespace NUMINAMATH_CALUDE_ellipse_equation_l2969_296979

/-- Given an ellipse with foci on the x-axis, sum of major and minor axes equal to 10,
    and focal distance equal to 4√5, prove that its equation is x²/36 + y²/16 = 1. -/
theorem ellipse_equation (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : a + b = 10) (h4 : 2 * c = 4 * Real.sqrt 5) (h5 : a^2 - b^2 = c^2) :
  ∀ x y : ℝ, (x^2 / 36 + y^2 / 16 = 1) ↔ 
  (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2969_296979


namespace NUMINAMATH_CALUDE_range_of_a_l2969_296951

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → |2*a - 1| ≤ |x + 1/x|) ↔ -1/2 ≤ a ∧ a ≤ 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2969_296951


namespace NUMINAMATH_CALUDE_hexagon_triangle_theorem_l2969_296988

/-- A regular hexagon with side length 1 -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : side_length = 1)

/-- An interior point of the hexagon -/
structure InteriorPoint :=
  (x : ℝ)
  (y : ℝ)
  (is_interior : True)  -- We assume the point is interior without specifying conditions

/-- A triangle formed by connecting the interior point to two vertices of the hexagon -/
structure HexagonTriangle :=
  (side1 : ℝ)
  (side2 : ℝ)
  (side3 : ℝ)

/-- The theorem to be proved -/
theorem hexagon_triangle_theorem (h : RegularHexagon) (p : InteriorPoint) 
  (triangles : Fin 6 → HexagonTriangle) : 
  ∃ (i j : Fin 6), i ≠ j ∧ 
    (triangles i).side1 ≥ 1 ∧ (triangles i).side2 ≥ 1 ∧ (triangles i).side3 ≥ 1 ∧
    (triangles j).side1 ≥ 1 ∧ (triangles j).side2 ≥ 1 ∧ (triangles j).side3 ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_hexagon_triangle_theorem_l2969_296988


namespace NUMINAMATH_CALUDE_number_of_hydroxide_groups_l2969_296997

/-- The atomic weight of aluminum -/
def atomic_weight_Al : ℝ := 27

/-- The molecular weight of a hydroxide group -/
def molecular_weight_OH : ℝ := 17

/-- The molecular weight of the compound Al(OH)n -/
def molecular_weight_compound : ℝ := 78

/-- The number of hydroxide groups in the compound -/
def n : ℕ := sorry

/-- Theorem stating that the number of hydroxide groups in Al(OH)n is 3 -/
theorem number_of_hydroxide_groups :
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_number_of_hydroxide_groups_l2969_296997


namespace NUMINAMATH_CALUDE_sin_cos_sum_shift_l2969_296902

theorem sin_cos_sum_shift (x : ℝ) : 
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * (x + π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_shift_l2969_296902


namespace NUMINAMATH_CALUDE_positive_integer_equation_l2969_296966

theorem positive_integer_equation (N : ℕ+) : 15^4 * 28^2 = 12^2 * N^2 ↔ N = 525 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_equation_l2969_296966


namespace NUMINAMATH_CALUDE_boulder_splash_width_l2969_296909

/-- The width of a boulder's splash given the number of pebbles, rocks, and boulders thrown,
    and the total width of all splashes. -/
theorem boulder_splash_width
  (num_pebbles : ℕ)
  (num_rocks : ℕ)
  (num_boulders : ℕ)
  (total_width : ℝ)
  (pebble_splash : ℝ)
  (rock_splash : ℝ)
  (h1 : num_pebbles = 6)
  (h2 : num_rocks = 3)
  (h3 : num_boulders = 2)
  (h4 : total_width = 7)
  (h5 : pebble_splash = 1/4)
  (h6 : rock_splash = 1/2)
  : (total_width - (num_pebbles * pebble_splash + num_rocks * rock_splash)) / num_boulders = 2 :=
sorry

end NUMINAMATH_CALUDE_boulder_splash_width_l2969_296909


namespace NUMINAMATH_CALUDE_four_Y_three_l2969_296917

def Y (a b : ℝ) : ℝ := (a - b)^3 + 5

theorem four_Y_three : Y 4 3 = 6 := by sorry

end NUMINAMATH_CALUDE_four_Y_three_l2969_296917


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l2969_296942

theorem fraction_to_decimal : (15 : ℚ) / 625 = (24 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l2969_296942


namespace NUMINAMATH_CALUDE_circle_definition_l2969_296924

/-- Definition of a circle in a plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- Theorem: The set of all points in a plane at a fixed distance from a given point forms a circle -/
theorem circle_definition (center : ℝ × ℝ) (radius : ℝ) :
  {p : ℝ × ℝ | Real.sqrt ((p.1 - center.1)^2 + (p.2 - center.2)^2) = radius} = Circle center radius :=
by sorry

end NUMINAMATH_CALUDE_circle_definition_l2969_296924


namespace NUMINAMATH_CALUDE_line_equation_proof_l2969_296916

/-- A line in the 2D plane represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (p : Point) :
  given_line.a = 1 →
  given_line.b = -2 →
  given_line.c = 3 →
  p.x = -1 →
  p.y = 3 →
  ∃ (result_line : Line),
    result_line.a = 1 ∧
    result_line.b = -2 ∧
    result_line.c = 7 ∧
    pointOnLine p result_line ∧
    parallelLines given_line result_line :=
by sorry


end NUMINAMATH_CALUDE_line_equation_proof_l2969_296916


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2969_296985

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 > x^2) ↔ (∃ x : ℝ, x^3 ≤ x^2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2969_296985


namespace NUMINAMATH_CALUDE_workshop_wolf_prize_laureates_l2969_296974

theorem workshop_wolf_prize_laureates 
  (total_scientists : ℕ) 
  (both_wolf_and_nobel : ℕ) 
  (total_nobel : ℕ) 
  (h1 : total_scientists = 50)
  (h2 : both_wolf_and_nobel = 12)
  (h3 : total_nobel = 23)
  (h4 : ∃ (non_wolf_non_nobel : ℕ), 
        non_wolf_non_nobel + (non_wolf_non_nobel + 3) = total_scientists - both_wolf_and_nobel) :
  ∃ (wolf_laureates : ℕ), wolf_laureates = 31 ∧ 
    wolf_laureates + (total_scientists - wolf_laureates) = total_scientists :=
sorry

end NUMINAMATH_CALUDE_workshop_wolf_prize_laureates_l2969_296974


namespace NUMINAMATH_CALUDE_rotation_equivalence_l2969_296930

theorem rotation_equivalence (y : ℝ) : 
  (450 % 360 : ℝ) = (360 - y) % 360 → y < 360 → y = 270 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l2969_296930


namespace NUMINAMATH_CALUDE_wire_length_proof_l2969_296973

theorem wire_length_proof (piece1 piece2 piece3 piece4 : ℝ) 
  (ratio_condition : piece1 / piece4 = 5 / 2 ∧ piece2 / piece4 = 7 / 2 ∧ piece3 / piece4 = 3 / 2)
  (shortest_piece : piece4 = 16) : 
  piece1 + piece2 + piece3 + piece4 = 136 := by
sorry

end NUMINAMATH_CALUDE_wire_length_proof_l2969_296973


namespace NUMINAMATH_CALUDE_ratio_problem_l2969_296922

theorem ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 1)
  (h3 : d / b = 2 / 5) :
  a / c = 25 / 32 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l2969_296922


namespace NUMINAMATH_CALUDE_number_problem_l2969_296906

theorem number_problem : ∃ x : ℝ, 0.50 * x = 0.30 * 50 + 13 ∧ x = 56 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2969_296906


namespace NUMINAMATH_CALUDE_simplify_expression_l2969_296956

theorem simplify_expression (y : ℝ) : 5*y + 8*y + 2*y + 7 = 15*y + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2969_296956


namespace NUMINAMATH_CALUDE_road_repair_equivalence_l2969_296998

/-- The number of persons in the first group -/
def first_group : ℕ := 36

/-- The number of days to complete the work -/
def days : ℕ := 12

/-- The number of hours worked per day by the first group -/
def hours_first : ℕ := 5

/-- The number of hours worked per day by the second group -/
def hours_second : ℕ := 6

/-- The number of persons in the second group -/
def second_group : ℕ := 30

theorem road_repair_equivalence :
  first_group * days * hours_first = second_group * days * hours_second :=
sorry

end NUMINAMATH_CALUDE_road_repair_equivalence_l2969_296998


namespace NUMINAMATH_CALUDE_geometric_relations_l2969_296929

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (line_perp_plane : Line → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (plane_perp_plane : Plane → Plane → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem geometric_relations 
  (m l : Line) (α β : Plane) : 
  (line_perp_plane l α ∧ line_parallel_plane m α → perpendicular l m) ∧
  ¬(parallel m l ∧ line_in_plane m α → line_parallel_plane l α) ∧
  ¬(plane_perp_plane α β ∧ line_in_plane m α ∧ line_in_plane l β → perpendicular m l) ∧
  ¬(perpendicular m l ∧ line_in_plane m α ∧ line_in_plane l β → plane_perp_plane α β) :=
by sorry

end NUMINAMATH_CALUDE_geometric_relations_l2969_296929


namespace NUMINAMATH_CALUDE_inequality_proof_l2969_296952

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  a / (c^2 + 1) > b / (c^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2969_296952


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l2969_296980

-- Define the quadratic function
def f (x : ℝ) : ℝ := (x - 2)^2 - 2

-- Theorem stating that f satisfies the given conditions
theorem quadratic_function_properties :
  (∃ (a : ℝ), f a = -2 ∧ ∀ x, f x ≥ f a) ∧  -- Vertex condition
  f 0 = 2                                   -- Y-intercept condition
  := by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l2969_296980


namespace NUMINAMATH_CALUDE_jasons_lawn_cutting_l2969_296941

/-- The number of lawns Jason can cut in 8 hours, given that it takes 30 minutes to cut one lawn -/
theorem jasons_lawn_cutting (time_per_lawn : ℕ) (total_time_hours : ℕ) : 
  time_per_lawn = 30 → total_time_hours = 8 → (total_time_hours * 60) / time_per_lawn = 16 := by
  sorry

end NUMINAMATH_CALUDE_jasons_lawn_cutting_l2969_296941


namespace NUMINAMATH_CALUDE_binary_subtraction_result_l2969_296943

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 111111111₂ -/
def binary_111111111 : List Bool := [true, true, true, true, true, true, true, true, true]

/-- The binary representation of 111111₂ -/
def binary_111111 : List Bool := [true, true, true, true, true, true]

/-- The theorem stating that the difference between the decimal representations
    of 111111111₂ and 111111₂ is equal to 448 -/
theorem binary_subtraction_result :
  binary_to_decimal binary_111111111 - binary_to_decimal binary_111111 = 448 := by
  sorry

end NUMINAMATH_CALUDE_binary_subtraction_result_l2969_296943


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l2969_296975

theorem rectangle_area_proof (large_square_side : ℝ) 
  (rectangle_length rectangle_width : ℝ) 
  (small_square_side : ℝ) :
  large_square_side = 4 →
  rectangle_length = 1 →
  rectangle_width = 4 →
  small_square_side = 2 →
  large_square_side^2 - (rectangle_length * rectangle_width + small_square_side^2) = 8 :=
by
  sorry

#check rectangle_area_proof

end NUMINAMATH_CALUDE_rectangle_area_proof_l2969_296975


namespace NUMINAMATH_CALUDE_root_quadruples_l2969_296920

theorem root_quadruples : ∀ a b c d : ℝ,
  (a ≠ b ∧ 
   2 * a^2 - 3 * c * a + 8 * d = 0 ∧
   2 * b^2 - 3 * c * b + 8 * d = 0 ∧
   c ≠ d ∧
   2 * c^2 - 3 * a * c + 8 * b = 0 ∧
   2 * d^2 - 3 * a * d + 8 * b = 0) →
  ((a = 4 ∧ b = 8 ∧ c = 4 ∧ d = 8) ∨
   (a = -2 ∧ b = -22 ∧ c = -8 ∧ d = 11) ∨
   (a = -8 ∧ b = 2 ∧ c = -2 ∧ d = -4)) :=
by sorry

end NUMINAMATH_CALUDE_root_quadruples_l2969_296920


namespace NUMINAMATH_CALUDE_contest_ranking_l2969_296965

theorem contest_ranking (A B C D : ℝ) 
  (non_negative : A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ D ≥ 0)
  (sum_equality : B + D = A + C)
  (interchange_inequality : A + B > C + D)
  (dick_exceeds : D > B + C) :
  A > D ∧ D > B ∧ B > C := by
sorry

end NUMINAMATH_CALUDE_contest_ranking_l2969_296965


namespace NUMINAMATH_CALUDE_cos_to_sin_shift_l2969_296931

open Real

theorem cos_to_sin_shift (x : ℝ) : 
  cos (2*x) = sin (2*(x - π/6)) :=
by sorry

end NUMINAMATH_CALUDE_cos_to_sin_shift_l2969_296931


namespace NUMINAMATH_CALUDE_solve_equation_l2969_296901

-- Define the custom operation *
def customMult (a b : ℚ) : ℚ := a + 2 * b

-- Define the equation to be solved
def equation (x : ℚ) : Prop :=
  customMult 3 (customMult 4 x) = 6

-- Theorem statement
theorem solve_equation : ∃ x : ℚ, equation x ∧ x = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2969_296901


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_negative_two_l2969_296977

theorem sum_of_roots_equals_negative_two
  (a b c d : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0)
  (h1 : c^2 + a*c + b = 0)
  (h2 : d^2 + a*d + b = 0)
  (h3 : a^2 + c*a + d = 0)
  (h4 : b^2 + c*b + d = 0) :
  a + b + c + d = -2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_negative_two_l2969_296977


namespace NUMINAMATH_CALUDE_min_n_for_120n_divisibility_l2969_296983

theorem min_n_for_120n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (4 ∣ 120 * m) ∧ (8 ∣ 120 * m) ∧ (12 ∣ 120 * m) → n ≤ m) ∧
  (4 ∣ 120 * n) ∧ (8 ∣ 120 * n) ∧ (12 ∣ 120 * n) :=
by
  -- Proof goes here
  sorry

#check min_n_for_120n_divisibility

end NUMINAMATH_CALUDE_min_n_for_120n_divisibility_l2969_296983


namespace NUMINAMATH_CALUDE_age_problem_l2969_296944

/-- The problem of finding when B was half the age A will be in 10 years -/
theorem age_problem (B_age : ℕ) (A_age : ℕ) (x : ℕ) : 
  B_age = 37 →
  A_age = B_age + 7 →
  B_age - x = (A_age + 10) / 2 →
  x = 10 := by
  sorry

end NUMINAMATH_CALUDE_age_problem_l2969_296944


namespace NUMINAMATH_CALUDE_expression_simplification_l2969_296915

theorem expression_simplification (x : ℝ) : 3*x + 4*x^2 + 2 - (9 - 3*x - 4*x^2) + Real.sin x = 8*x^2 + 6*x - 7 + Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2969_296915


namespace NUMINAMATH_CALUDE_triangle_area_l2969_296987

-- Define the triangle ABC
structure Triangle where
  A : ℝ  -- Angle A
  B : ℝ  -- Angle B
  C : ℝ  -- Angle C
  a : ℝ  -- Side a
  c : ℝ  -- Side c

-- Define the conditions of the problem
def problem_triangle : Triangle where
  A := sorry
  B := sorry
  C := sorry
  a := 2
  c := 5

-- Define the arithmetic sequence property
def is_arithmetic_sequence (t : Triangle) : Prop :=
  t.A + t.C = 2 * t.B

-- Define the angle sum property
def angle_sum (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi

-- Theorem statement
theorem triangle_area (t : Triangle) 
  (h1 : is_arithmetic_sequence t) 
  (h2 : angle_sum t) 
  (h3 : t.a = 2) 
  (h4 : t.c = 5) : 
  (1/2 : ℝ) * t.a * t.c * Real.sin t.B = (5 * Real.sqrt 3) / 2 := by
  sorry

-- Note: The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_triangle_area_l2969_296987


namespace NUMINAMATH_CALUDE_chime_2023_date_l2969_296933

/-- Represents a date with year, month, and day -/
structure Date :=
  (year : Nat) (month : Nat) (day : Nat)

/-- Represents a time with hour and minute -/
structure Time :=
  (hour : Nat) (minute : Nat)

/-- Calculates the number of chimes for a given hour -/
def chimes_for_hour (hour : Nat) : Nat :=
  if hour ≤ 12 then hour else hour - 12

/-- Calculates the total number of chimes in a day with the malfunction -/
def daily_chimes : Nat := 101

/-- Calculates the number of chimes from a given start time to midnight -/
def chimes_until_midnight (start_time : Time) : Nat :=
  sorry -- Implementation details omitted

/-- Calculates the date of the nth chime given a start date and time -/
def date_of_nth_chime (start_date : Date) (start_time : Time) (n : Nat) : Date :=
  sorry -- Implementation details omitted

theorem chime_2023_date :
  let start_date := Date.mk 2003 2 26
  let start_time := Time.mk 14 15
  date_of_nth_chime start_date start_time 2023 = Date.mk 2003 3 18 := by
  sorry

end NUMINAMATH_CALUDE_chime_2023_date_l2969_296933


namespace NUMINAMATH_CALUDE_fold_reflection_sum_l2969_296948

/-- The fold line passing through the midpoint of (0,3) and (5,0) -/
def fold_line (x y : ℝ) : Prop := y = (5/3) * x - 1

/-- The property that (m,n) is the reflection of (8,4) across the fold line -/
def reflection_property (m n : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    fold_line x y ∧ 
    (x = (8 + m) / 2 ∧ y = (4 + n) / 2) ∧
    (n - 4) / (m - 8) = -3/5

theorem fold_reflection_sum (m n : ℝ) 
  (h1 : fold_line 0 3)
  (h2 : fold_line 5 0)
  (h3 : reflection_property m n) :
  m + n = 9.75 := by sorry

end NUMINAMATH_CALUDE_fold_reflection_sum_l2969_296948


namespace NUMINAMATH_CALUDE_triangle_is_right_angled_l2969_296912

theorem triangle_is_right_angled : 
  let A : ℂ := 1
  let B : ℂ := Complex.I * 2
  let C : ℂ := 5 + Complex.I * 2
  let AB : ℂ := B - A
  let BC : ℂ := C - B
  let CA : ℂ := A - C
  Complex.abs AB ^ 2 + Complex.abs CA ^ 2 = Complex.abs BC ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_is_right_angled_l2969_296912


namespace NUMINAMATH_CALUDE_inequality_solution_l2969_296905

-- Define the polynomial function
def f (x : ℝ) := x^3 - 4*x^2 - x + 20

-- Define the set of x satisfying the inequality
def S : Set ℝ := {x | f x > 0}

-- State the theorem
theorem inequality_solution : S = Set.Ioi (-4) ∪ Set.Ioi 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2969_296905


namespace NUMINAMATH_CALUDE_min_value_a_l2969_296958

theorem min_value_a (a b : ℕ) (h : 1176 * a = b^3) : 63 ≤ a := by
  sorry

end NUMINAMATH_CALUDE_min_value_a_l2969_296958


namespace NUMINAMATH_CALUDE_milk_water_mixture_l2969_296928

theorem milk_water_mixture (milk water : ℝ) : 
  milk / water = 2 →
  milk / (water + 10) = 6 / 5 →
  milk = 30 := by
sorry

end NUMINAMATH_CALUDE_milk_water_mixture_l2969_296928


namespace NUMINAMATH_CALUDE_marias_gum_l2969_296976

/-- Represents the number of pieces of gum Maria has -/
def total_gum (initial : ℕ) (x : ℕ) (y : ℕ) : ℕ := initial + x + y

/-- Theorem stating the total number of pieces of gum Maria has -/
theorem marias_gum (x y : ℕ) (h1 : 10 ≤ x ∧ x < 100) (h2 : 10 ≤ y ∧ y < 100) :
  total_gum 58 x y = 58 + x + y := by sorry

end NUMINAMATH_CALUDE_marias_gum_l2969_296976


namespace NUMINAMATH_CALUDE_oil_tank_depth_l2969_296918

/-- Represents a right frustum oil tank -/
structure RightFrustumTank where
  volume : ℝ  -- Volume in liters
  top_edge : ℝ  -- Length of top edge in cm
  bottom_edge : ℝ  -- Length of bottom edge in cm

/-- Calculates the depth of a right frustum oil tank -/
def calculate_depth (tank : RightFrustumTank) : ℝ :=
  sorry

/-- Theorem stating that the depth of the given oil tank is 75 cm -/
theorem oil_tank_depth (tank : RightFrustumTank) 
  (h1 : tank.volume = 190)
  (h2 : tank.top_edge = 60)
  (h3 : tank.bottom_edge = 40) :
  calculate_depth tank = 75 :=
sorry

end NUMINAMATH_CALUDE_oil_tank_depth_l2969_296918


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2969_296962

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 * y + 6) = 5 → y = (19 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2969_296962


namespace NUMINAMATH_CALUDE_f_derivative_at_2014_l2969_296938

noncomputable def f (f'2014 : ℝ) : ℝ → ℝ := 
  λ x => (1/2) * x^2 + 2 * x * f'2014 + 2014 * Real.log x

theorem f_derivative_at_2014 : 
  ∃ f'2014 : ℝ, (deriv (f f'2014)) 2014 = -2015 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_2014_l2969_296938


namespace NUMINAMATH_CALUDE_four_point_lines_l2969_296981

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

/-- Count the number of distinct lines through four points -/
def count_lines (p1 p2 p3 p4 : Point) : ℕ :=
  sorry

/-- Theorem: The number of distinct lines through four points is either 1, 4, or 6 -/
theorem four_point_lines (p1 p2 p3 p4 : Point) :
  count_lines p1 p2 p3 p4 = 1 ∨ count_lines p1 p2 p3 p4 = 4 ∨ count_lines p1 p2 p3 p4 = 6 :=
by sorry

end NUMINAMATH_CALUDE_four_point_lines_l2969_296981


namespace NUMINAMATH_CALUDE_divisors_of_2_pow_n_minus_1_l2969_296957

theorem divisors_of_2_pow_n_minus_1 (n : ℕ) (d : ℕ) (h1 : Odd n) (h2 : d > 0) (h3 : d ∣ (2^n - 1)) :
  d % 8 = 1 ∨ d % 8 = 7 :=
sorry

end NUMINAMATH_CALUDE_divisors_of_2_pow_n_minus_1_l2969_296957


namespace NUMINAMATH_CALUDE_six_meetings_in_middle_l2969_296919

/-- Represents a runner on a circular track -/
structure Runner where
  speed : ℕ  -- Speed in meters per minute

/-- Calculates the number of meetings in the middle for two runners -/
def numberOfMeetings (runner1 runner2 : Runner) : ℕ :=
  sorry

/-- Theorem stating that two runners with given speeds meet 6 times in the middle -/
theorem six_meetings_in_middle :
  let runner1 : Runner := ⟨240⟩
  let runner2 : Runner := ⟨180⟩
  numberOfMeetings runner1 runner2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_six_meetings_in_middle_l2969_296919


namespace NUMINAMATH_CALUDE_raw_materials_cost_l2969_296907

/-- The total amount Kanul had --/
def total : ℝ := 5714.29

/-- The amount spent on machinery --/
def machinery : ℝ := 1000

/-- The percentage of total amount kept as cash --/
def cash_percentage : ℝ := 0.30

/-- The amount spent on raw materials --/
def raw_materials : ℝ := total - machinery - (cash_percentage * total)

/-- Theorem stating that the amount spent on raw materials is approximately $3000.00 --/
theorem raw_materials_cost : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |raw_materials - 3000| < ε := by
  sorry

end NUMINAMATH_CALUDE_raw_materials_cost_l2969_296907


namespace NUMINAMATH_CALUDE_cyclist_average_speed_l2969_296999

/-- Given a cyclist who travels two segments with different distances and speeds, 
    this theorem proves that the average speed for the entire trip is 18 miles per hour. -/
theorem cyclist_average_speed : 
  ∀ (d₁ d₂ v₁ v₂ : ℝ), 
    d₁ = 45 ∧ d₂ = 15 ∧ v₁ = 15 ∧ v₂ = 45 →
    (d₁ + d₂) / ((d₁ / v₁) + (d₂ / v₂)) = 18 := by
  sorry


end NUMINAMATH_CALUDE_cyclist_average_speed_l2969_296999


namespace NUMINAMATH_CALUDE_bakery_flour_usage_l2969_296910

theorem bakery_flour_usage (wheat_flour : Real) (white_flour : Real)
  (h1 : wheat_flour = 0.2)
  (h2 : white_flour = 0.1) :
  wheat_flour + white_flour = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_bakery_flour_usage_l2969_296910


namespace NUMINAMATH_CALUDE_fifth_term_sequence_l2969_296964

theorem fifth_term_sequence (n : ℕ) : 
  let a : ℕ → ℕ := λ k => k * (k + 1) / 2
  a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_sequence_l2969_296964


namespace NUMINAMATH_CALUDE_gear_system_rotation_l2969_296949

/-- Represents the rotation direction of a gear -/
inductive Direction
| Clockwise
| Counterclockwise

/-- Represents a system of gears -/
structure GearSystem :=
  (n : ℕ)  -- number of gears

/-- Returns the direction of the i-th gear in the system -/
def gear_direction (sys : GearSystem) (i : ℕ) : Direction :=
  if i % 2 = 0 then Direction.Counterclockwise else Direction.Clockwise

/-- Checks if the gear system can rotate -/
def can_rotate (sys : GearSystem) : Prop :=
  sys.n % 2 = 0

theorem gear_system_rotation (sys : GearSystem) :
  can_rotate sys ↔ sys.n % 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_gear_system_rotation_l2969_296949


namespace NUMINAMATH_CALUDE_evaluate_f_l2969_296939

def f (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_f : 3 * f 2 + 2 * f (-2) = 98 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_f_l2969_296939


namespace NUMINAMATH_CALUDE_sphere_volume_increase_l2969_296937

/-- The volume of a sphere increases by a factor of 8 when its radius is doubled -/
theorem sphere_volume_increase (r : ℝ) (hr : r > 0) : 
  (4 / 3 * Real.pi * (2 * r)^3) / (4 / 3 * Real.pi * r^3) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_increase_l2969_296937


namespace NUMINAMATH_CALUDE_unique_triples_l2969_296989

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem unique_triples : 
  ∀ a b c : ℕ,
    (is_prime (a^2 - 23)) →
    (is_prime (b^2 - 23)) →
    ((a^2 - 23) * (b^2 - 23) = c^2 - 23) →
    ((a = 5 ∧ b = 6 ∧ c = 7) ∨ (a = 6 ∧ b = 5 ∧ c = 7)) :=
by sorry

end NUMINAMATH_CALUDE_unique_triples_l2969_296989


namespace NUMINAMATH_CALUDE_sequence_fifth_term_l2969_296904

/-- Given a sequence {aₙ} with the following properties:
  1) a₁ = 1
  2) aₙ - aₙ₋₁ = 2 for n ≥ 2, n ∈ ℕ*
  Prove that a₅ = 9 -/
theorem sequence_fifth_term (a : ℕ+ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ+, n ≥ 2 → a n - a (n-1) = 2) :
  a 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sequence_fifth_term_l2969_296904


namespace NUMINAMATH_CALUDE_playground_count_l2969_296936

theorem playground_count (numbers : List Nat) : 
  numbers.length = 6 ∧ 
  numbers.take 5 = [6, 12, 1, 12, 7] ∧ 
  (numbers.sum / numbers.length : ℚ) = 7 →
  numbers.getLast! = 4 := by
sorry

end NUMINAMATH_CALUDE_playground_count_l2969_296936


namespace NUMINAMATH_CALUDE_arcsin_of_one_l2969_296996

theorem arcsin_of_one (π : Real) : Real.arcsin 1 = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_of_one_l2969_296996


namespace NUMINAMATH_CALUDE_pet_shop_grooming_time_l2969_296990

/-- The time it takes to groom all dogs in a pet shop -/
theorem pet_shop_grooming_time 
  (poodle_time : ℝ) 
  (terrier_time : ℝ) 
  (num_poodles : ℕ) 
  (num_terriers : ℕ) 
  (num_employees : ℕ) 
  (h1 : poodle_time = 30) 
  (h2 : terrier_time = poodle_time / 2) 
  (h3 : num_poodles = 3) 
  (h4 : num_terriers = 8) 
  (h5 : num_employees = 4) 
  (h6 : num_employees > 0) :
  (num_poodles * poodle_time + num_terriers * terrier_time) / num_employees = 52.5 := by
  sorry


end NUMINAMATH_CALUDE_pet_shop_grooming_time_l2969_296990


namespace NUMINAMATH_CALUDE_range_of_a_l2969_296950

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2)^2 * Real.exp x + a * Real.exp (-x)

/-- The function g(x) -/
def g (a : ℝ) (x : ℝ) : ℝ := 2 * a * |x - 2|

/-- The theorem stating the range of a -/
theorem range_of_a (a : ℝ) :
  (∃ (s : Finset ℝ), s.card = 6 ∧ ∀ x ∈ s, f a x = g a x) →
  1 < a ∧ a < Real.exp 2 / (2 * Real.exp 1 - 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2969_296950


namespace NUMINAMATH_CALUDE_candy_division_l2969_296935

theorem candy_division (mark peter susan john lucy : ℝ) 
  (h1 : mark = 90)
  (h2 : peter = 120.5)
  (h3 : susan = 74.75)
  (h4 : john = 150)
  (h5 : lucy = 85.25)
  (total_people : ℕ)
  (h6 : total_people = 10) :
  (mark + peter + susan + john + lucy) / total_people = 52.05 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l2969_296935


namespace NUMINAMATH_CALUDE_max_pieces_is_nine_l2969_296926

/-- The size of the large cake in inches -/
def large_cake_size : ℕ := 15

/-- The size of a small piece in inches -/
def small_piece_size : ℕ := 5

/-- The maximum number of small pieces that can be cut from the large cake -/
def max_pieces : ℕ := (large_cake_size * large_cake_size) / (small_piece_size * small_piece_size)

theorem max_pieces_is_nine : max_pieces = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_pieces_is_nine_l2969_296926


namespace NUMINAMATH_CALUDE_complex_number_problem_l2969_296913

theorem complex_number_problem (z : ℂ) :
  Complex.abs z = 1 ∧ (Complex.I * Complex.im ((3 + 4*Complex.I) * z) = (3 + 4*Complex.I) * z) →
  z = 4/5 + 3/5*Complex.I ∨ z = -4/5 - 3/5*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2969_296913


namespace NUMINAMATH_CALUDE_water_container_problem_l2969_296969

theorem water_container_problem :
  let large_capacity : ℚ := 144
  let small_capacity : ℚ := 100
  ∀ x y : ℚ,
  (x + (4/5) * y = large_capacity) →
  (y + (5/12) * x = small_capacity) →
  x = 96 ∧ y = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_container_problem_l2969_296969


namespace NUMINAMATH_CALUDE_square_difference_503_496_l2969_296953

theorem square_difference_503_496 : 503^2 - 496^2 = 6993 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_503_496_l2969_296953


namespace NUMINAMATH_CALUDE_interval_of_decrease_f_left_endpoint_neg_infinity_right_endpoint_is_one_l2969_296968

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 2*x + 4

-- Theorem stating the interval of decrease
theorem interval_of_decrease_f :
  ∀ x y : ℝ, x < y ∧ y ≤ 1 → f x > f y :=
by sorry

-- The left endpoint of the interval is negative infinity
theorem left_endpoint_neg_infinity :
  ∀ M : ℝ, ∃ x : ℝ, x < M ∧ ∀ y : ℝ, x < y ∧ y ≤ 1 → f x > f y :=
by sorry

-- The right endpoint of the interval is 1
theorem right_endpoint_is_one :
  ∀ ε > 0, ∃ x : ℝ, 1 < x ∧ x < 1 + ε ∧ f 1 < f x :=
by sorry

end NUMINAMATH_CALUDE_interval_of_decrease_f_left_endpoint_neg_infinity_right_endpoint_is_one_l2969_296968


namespace NUMINAMATH_CALUDE_two_problems_require_loop_l2969_296946

/-- Represents a problem that may or may not require a loop statement to solve. -/
inductive Problem
| SumGeometricSeries
| CompareNumbers
| PiecewiseFunction
| LargestSquareLessThan100

/-- Determines if a given problem requires a loop statement to solve. -/
def requiresLoop (p : Problem) : Bool :=
  match p with
  | Problem.SumGeometricSeries => true
  | Problem.CompareNumbers => false
  | Problem.PiecewiseFunction => false
  | Problem.LargestSquareLessThan100 => true

/-- The list of all problems given in the original question. -/
def allProblems : List Problem :=
  [Problem.SumGeometricSeries, Problem.CompareNumbers, 
   Problem.PiecewiseFunction, Problem.LargestSquareLessThan100]

theorem two_problems_require_loop : 
  (allProblems.filter requiresLoop).length = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_problems_require_loop_l2969_296946


namespace NUMINAMATH_CALUDE_factorial_sum_perfect_square_l2969_296914

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sumFactorials (m : ℕ) : ℕ := (List.range m).map factorial |>.sum

def isPerfectSquare (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

theorem factorial_sum_perfect_square :
  ∀ m : ℕ, m > 0 → (isPerfectSquare (sumFactorials m) ↔ m = 1 ∨ m = 3) :=
by sorry

end NUMINAMATH_CALUDE_factorial_sum_perfect_square_l2969_296914


namespace NUMINAMATH_CALUDE_spaghetti_cost_l2969_296992

def hamburger_cost : ℝ := 3
def fries_cost : ℝ := 1.20
def soda_cost : ℝ := 0.5
def num_hamburgers : ℕ := 5
def num_fries : ℕ := 4
def num_sodas : ℕ := 5
def num_friends : ℕ := 5
def individual_payment : ℝ := 5

theorem spaghetti_cost : 
  ∃ (spaghetti_price : ℝ),
    spaghetti_price = 
      num_friends * individual_payment - 
      (num_hamburgers * hamburger_cost + 
       num_fries * fries_cost + 
       num_sodas * soda_cost) ∧
    spaghetti_price = 2.70 :=
sorry

end NUMINAMATH_CALUDE_spaghetti_cost_l2969_296992


namespace NUMINAMATH_CALUDE_litter_patrol_theorem_l2969_296982

/-- The total number of litter items picked up by the Litter Patrol -/
def total_litter : ℕ := 40

/-- The number of non-miscellaneous items (glass bottles + aluminum cans + plastic bags) -/
def non_misc_items : ℕ := 30

/-- The percentage of non-miscellaneous items in the total litter -/
def non_misc_percentage : ℚ := 3/4

theorem litter_patrol_theorem :
  (non_misc_items : ℚ) / non_misc_percentage = total_litter := by sorry

end NUMINAMATH_CALUDE_litter_patrol_theorem_l2969_296982


namespace NUMINAMATH_CALUDE_max_points_is_168_l2969_296925

/-- Represents the number of cards of each color chosen by Vasya -/
structure CardChoice where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Calculates the total points for a given card choice -/
def calculatePoints (choice : CardChoice) : ℕ :=
  choice.red + 2 * choice.red * choice.blue + 3 * choice.blue * choice.yellow

/-- Theorem: The maximum number of points Vasya can earn is 168 -/
theorem max_points_is_168 : 
  ∃ (choice : CardChoice), 
    choice.red + choice.blue + choice.yellow = 15 ∧ 
    choice.red ≤ 15 ∧ choice.blue ≤ 15 ∧ choice.yellow ≤ 15 ∧
    calculatePoints choice = 168 ∧
    ∀ (other : CardChoice), 
      other.red + other.blue + other.yellow = 15 → 
      other.red ≤ 15 ∧ other.blue ≤ 15 ∧ other.yellow ≤ 15 →
      calculatePoints other ≤ 168 := by
  sorry


end NUMINAMATH_CALUDE_max_points_is_168_l2969_296925


namespace NUMINAMATH_CALUDE_toys_sold_l2969_296947

theorem toys_sold (selling_price : ℕ) (cost_price : ℕ) (gain : ℕ) :
  selling_price = 16800 →
  gain = 3 * cost_price →
  cost_price = 800 →
  (selling_price - gain) / cost_price = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_toys_sold_l2969_296947


namespace NUMINAMATH_CALUDE_intersection_distance_l2969_296960

/-- Two lines intersecting at 60 degrees --/
structure IntersectingLines :=
  (angle : ℝ)
  (h_angle : angle = 60)

/-- Points on the intersecting lines --/
structure PointsOnLines (l : IntersectingLines) :=
  (A B : ℝ × ℝ)
  (dist_initial : ℝ)
  (dist_after_move : ℝ)
  (move_distance : ℝ)
  (h_initial_dist : dist_initial = 31)
  (h_after_move_dist : dist_after_move = 21)
  (h_move_distance : move_distance = 20)

/-- The theorem to be proved --/
theorem intersection_distance (l : IntersectingLines) (p : PointsOnLines l) :
  ∃ (dist_A dist_B : ℝ),
    dist_A = 35 ∧ dist_B = 24 ∧
    (dist_A - p.move_distance)^2 + dist_B^2 = p.dist_initial^2 ∧
    dist_A^2 + dist_B^2 = p.dist_after_move^2 + p.move_distance^2 :=
sorry

end NUMINAMATH_CALUDE_intersection_distance_l2969_296960


namespace NUMINAMATH_CALUDE_train_length_l2969_296994

theorem train_length (platform_time : ℝ) (pole_time : ℝ) (platform_length : ℝ)
  (h1 : platform_time = 39)
  (h2 : pole_time = 18)
  (h3 : platform_length = 350) :
  ∃ (train_length : ℝ) (train_speed : ℝ),
    train_length = train_speed * pole_time ∧
    train_length + platform_length = train_speed * platform_time ∧
    train_length = 300 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2969_296994


namespace NUMINAMATH_CALUDE_volume_ratio_l2969_296995

theorem volume_ratio (A B C : ℝ) 
  (h1 : 2 * A = B + C) 
  (h2 : 5 * B = A + C) : 
  C / (A + B) = 1 := by sorry

end NUMINAMATH_CALUDE_volume_ratio_l2969_296995


namespace NUMINAMATH_CALUDE_wedge_volume_of_sphere_l2969_296970

/-- The volume of a wedge of a sphere -/
theorem wedge_volume_of_sphere (circumference : ℝ) (num_wedges : ℕ) : 
  circumference = 18 * Real.pi → 
  num_wedges = 6 → 
  (1 / num_wedges : ℝ) * (4 / 3 : ℝ) * Real.pi * (circumference / (2 * Real.pi))^3 = 162 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_wedge_volume_of_sphere_l2969_296970


namespace NUMINAMATH_CALUDE_mode_is_131_l2969_296978

/- Define the structure of a stem-and-leaf plot entry -/
structure StemLeafEntry :=
  (stem : ℕ)
  (leaves : List ℕ)

/- Define the stem-and-leaf plot -/
def stemLeafPlot : List StemLeafEntry := [
  ⟨9, [5, 5, 6]⟩,
  ⟨10, [4, 8]⟩,
  ⟨11, [2, 2, 2, 6, 6, 7]⟩,
  ⟨12, [0, 0, 3, 7, 7, 7]⟩,
  ⟨13, [1, 1, 1, 1]⟩,
  ⟨14, [5, 9]⟩
]

/- Define a function to calculate the mode -/
def calculateMode (plot : List StemLeafEntry) : ℕ :=
  sorry

/- Theorem stating that the mode of the given stem-and-leaf plot is 131 -/
theorem mode_is_131 : calculateMode stemLeafPlot = 131 :=
  sorry

end NUMINAMATH_CALUDE_mode_is_131_l2969_296978


namespace NUMINAMATH_CALUDE_airplane_seat_ratio_l2969_296961

theorem airplane_seat_ratio :
  ∀ (total_seats coach_seats first_class_seats k : ℕ),
    total_seats = 387 →
    coach_seats = 310 →
    coach_seats = k * first_class_seats + 2 →
    first_class_seats + coach_seats = total_seats →
    (coach_seats - 2) / first_class_seats = 4 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seat_ratio_l2969_296961


namespace NUMINAMATH_CALUDE_cookie_jar_final_amount_l2969_296993

theorem cookie_jar_final_amount : 
  let initial_amount : ℚ := 21
  let doris_spent : ℚ := 6
  let martha_spent : ℚ := doris_spent / 2
  let john_added : ℚ := 10
  let john_spent_percentage : ℚ := 1 / 4
  let final_amount : ℚ := 
    (initial_amount - doris_spent - martha_spent + john_added) * 
    (1 - john_spent_percentage)
  final_amount = 33 / 2 := by sorry

end NUMINAMATH_CALUDE_cookie_jar_final_amount_l2969_296993


namespace NUMINAMATH_CALUDE_right_triangle_special_angles_l2969_296954

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- leg 1
  b : ℝ  -- leg 2
  c : ℝ  -- hypotenuse
  h : ℝ  -- altitude to hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem
  altitude_condition : h = c / 4  -- altitude is 4 times smaller than hypotenuse

-- Define the theorem
theorem right_triangle_special_angles (t : RightTriangle) :
  let angle1 := Real.arcsin (t.h / t.c)
  let angle2 := Real.arcsin (t.a / t.c)
  (angle1 = 15 * π / 180 ∧ angle2 = 75 * π / 180) ∨
  (angle1 = 75 * π / 180 ∧ angle2 = 15 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_right_triangle_special_angles_l2969_296954


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l2969_296972

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def count_perfect_square_factors (a b c d : ℕ) : ℕ :=
  (a + 1) * (b + 1) * (c + 1) * (d + 1)

theorem perfect_square_factors_count :
  count_perfect_square_factors 6 7 8 4 = 2520 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l2969_296972


namespace NUMINAMATH_CALUDE_nabla_example_l2969_296986

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_example : nabla (nabla 2 3) 2 = 4099 := by
  sorry

end NUMINAMATH_CALUDE_nabla_example_l2969_296986
