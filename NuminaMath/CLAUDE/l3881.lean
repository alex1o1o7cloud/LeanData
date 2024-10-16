import Mathlib

namespace NUMINAMATH_CALUDE_max_x_on_circle_max_x_achieved_l3881_388145

theorem max_x_on_circle (x y : ℝ) (h : x^2 + y^2 = 18*x + 20*y) :
  x ≤ 9 + Real.sqrt 181 :=
by sorry

theorem max_x_achieved (ε : ℝ) (hε : ε > 0) :
  ∃ (x y : ℝ), x^2 + y^2 = 18*x + 20*y ∧ x > 9 + Real.sqrt 181 - ε :=
by sorry

end NUMINAMATH_CALUDE_max_x_on_circle_max_x_achieved_l3881_388145


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3881_388154

def A : Set ℝ := {-2, -1, 0, 1, 2}

def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≤ 0}

theorem intersection_complement_equality : A ∩ (Set.univ \ B) = {-2} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3881_388154


namespace NUMINAMATH_CALUDE_max_value_condition_l3881_388162

/-- The function f(x) = x^2 - 2x + 3 has a maximum value of 3 on the interval [0, a+2] 
    if and only if -2 < a ≤ 0 -/
theorem max_value_condition (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 (a + 2), x^2 - 2*x + 3 ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 (a + 2), x^2 - 2*x + 3 = 3) ↔ 
  -2 < a ∧ a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_max_value_condition_l3881_388162


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_l3881_388102

theorem sufficiency_not_necessity (p q : Prop) : 
  (¬p ∧ ¬q → ¬(p ∧ q)) ∧ 
  ∃ (p q : Prop), ¬(p ∧ q) ∧ ¬(¬p ∧ ¬q) := by
sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_l3881_388102


namespace NUMINAMATH_CALUDE_special_hexagon_angle_sum_l3881_388129

/-- A hexagon with specific angle properties -/
structure SpecialHexagon where
  /-- Exterior angle measuring 40° -/
  ext_angle : ℝ
  /-- First interior angle measuring 45° -/
  int_angle1 : ℝ
  /-- Second interior angle measuring 80° -/
  int_angle2 : ℝ
  /-- Third interior angle -/
  int_angle3 : ℝ
  /-- Fourth interior angle -/
  int_angle4 : ℝ
  /-- Condition: Exterior angle is 40° -/
  h1 : ext_angle = 40
  /-- Condition: First interior angle is 45° -/
  h2 : int_angle1 = 45
  /-- Condition: Second interior angle is 80° -/
  h3 : int_angle2 = 80
  /-- Sum of interior angles of a hexagon is 720° -/
  h4 : int_angle1 + int_angle2 + int_angle3 + int_angle4 + (180 - ext_angle) + 90 = 720

/-- The sum of the third and fourth interior angles is 15° -/
theorem special_hexagon_angle_sum (h : SpecialHexagon) : h.int_angle3 + h.int_angle4 = 15 := by
  sorry

end NUMINAMATH_CALUDE_special_hexagon_angle_sum_l3881_388129


namespace NUMINAMATH_CALUDE_circle_radius_from_triangle_l3881_388132

/-- Given a right-angled triangle with area 60 cm² and one side 15 cm that touches a circle,
    prove that the radius of the circle is 20 cm. -/
theorem circle_radius_from_triangle (triangle_area : ℝ) (triangle_side : ℝ) (circle_radius : ℝ) :
  triangle_area = 60 →
  triangle_side = 15 →
  -- Additional properties to define the relationship between the triangle and circle
  -- These are simplified representations of the problem conditions
  ∃ (triangle_height : ℝ) (triangle_hypotenuse : ℝ),
    triangle_area = (1/2) * triangle_side * triangle_height ∧
    triangle_hypotenuse^2 = triangle_side^2 + triangle_height^2 ∧
    circle_radius - triangle_height + circle_radius - triangle_side = triangle_hypotenuse →
  circle_radius = 20 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_from_triangle_l3881_388132


namespace NUMINAMATH_CALUDE_shortest_side_of_special_triangle_l3881_388150

theorem shortest_side_of_special_triangle :
  ∀ (a b c : ℕ),
    a = 17 →
    a + b + c = 50 →
    (∃ A : ℕ, A^2 = (a + b + c) * (b + c - a) * (a + c - b) * (a + b - c) / 16) →
    b ≥ 13 ∧ c ≥ 13 :=
by sorry

end NUMINAMATH_CALUDE_shortest_side_of_special_triangle_l3881_388150


namespace NUMINAMATH_CALUDE_train_journey_time_l3881_388182

/-- If a train travels at 4/7 of its usual speed and arrives 9 minutes late, 
    its usual time to cover the journey is 12 minutes. -/
theorem train_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
    (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
    (4 / 7 * usual_speed) * (usual_time + 9) = usual_speed * usual_time → 
    usual_time = 12 := by
  sorry

end NUMINAMATH_CALUDE_train_journey_time_l3881_388182


namespace NUMINAMATH_CALUDE_quadratic_root_value_l3881_388157

theorem quadratic_root_value (c : ℝ) : 
  (∀ x : ℝ, (5/2 * x^2 + 17*x + c = 0) ↔ (x = (-17 + Real.sqrt 23) / 5 ∨ x = (-17 - Real.sqrt 23) / 5)) 
  → c = 26.6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_value_l3881_388157


namespace NUMINAMATH_CALUDE_children_on_bus_after_stop_l3881_388134

theorem children_on_bus_after_stop (initial_children : ℕ) (children_off : ℕ) (extra_children_on : ℕ) : 
  initial_children = 5 →
  children_off = 63 →
  extra_children_on = 9 →
  (initial_children - children_off + (children_off + extra_children_on) : ℤ) = 14 :=
by sorry

end NUMINAMATH_CALUDE_children_on_bus_after_stop_l3881_388134


namespace NUMINAMATH_CALUDE_fraction_product_l3881_388175

theorem fraction_product : (2 : ℚ) / 3 * (4 : ℚ) / 9 = (8 : ℚ) / 27 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_l3881_388175


namespace NUMINAMATH_CALUDE_textile_firm_expenses_l3881_388197

/-- Calculates the monthly manufacturing expenses for a textile manufacturing firm. -/
def monthly_manufacturing_expenses (
  num_looms : ℕ
) (total_sales : ℕ)
  (establishment_charges : ℕ)
  (profit_decrease_one_loom : ℕ) : ℕ :=
  let sales_per_loom := total_sales / num_looms
  let cost_saved_one_loom := sales_per_loom - profit_decrease_one_loom
  cost_saved_one_loom * num_looms

/-- Theorem stating the monthly manufacturing expenses for the given problem. -/
theorem textile_firm_expenses :
  monthly_manufacturing_expenses 125 500000 75000 2800 = 150000 := by
  sorry

end NUMINAMATH_CALUDE_textile_firm_expenses_l3881_388197


namespace NUMINAMATH_CALUDE_fifth_number_in_row_l3881_388169

-- Define Pascal's triangle
def pascal (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Define the row we're interested in
def targetRow : ℕ → ℕ
  | 0 => 1
  | 1 => 15
  | k => pascal 15 (k - 1)

-- State the theorem
theorem fifth_number_in_row : targetRow 5 = 1365 := by
  sorry

end NUMINAMATH_CALUDE_fifth_number_in_row_l3881_388169


namespace NUMINAMATH_CALUDE_hcf_of_three_numbers_l3881_388122

theorem hcf_of_three_numbers (a b c : ℕ) (h_lcm : Nat.lcm (Nat.lcm a b) c = 2^4 * 3^2 * 17 * 7)
  (h_a : a = 136) (h_b : b = 144) (h_c : c = 168) : Nat.gcd (Nat.gcd a b) c = 8 := by
  sorry

end NUMINAMATH_CALUDE_hcf_of_three_numbers_l3881_388122


namespace NUMINAMATH_CALUDE_circle_equation_l3881_388117

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point is on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Check if a circle is tangent to a line -/
def Circle.tangentTo (c : Circle) (l : Line) : Prop :=
  let (cx, cy) := c.center
  |l.a * cx + l.b * cy + l.c| = c.radius * Real.sqrt (l.a^2 + l.b^2)

/-- The main theorem -/
theorem circle_equation (C : Circle) (l : Line) :
  C.contains (0, 0) →
  C.radius^2 * Real.pi = 2 * Real.pi →
  l.a = 1 ∧ l.b = -1 ∧ l.c = 2 →
  C.tangentTo l →
  (C.center = (1, 1) ∧ C.radius^2 = 2) ∨ (C.center = (-1, -1) ∧ C.radius^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l3881_388117


namespace NUMINAMATH_CALUDE_work_completion_time_l3881_388189

theorem work_completion_time (days_B : ℝ) (combined_work : ℝ) (combined_days : ℝ) (days_A : ℝ) : 
  days_B = 45 →
  combined_work = 7 / 18 →
  combined_days = 7 →
  (1 / days_A + 1 / days_B) * combined_days = combined_work →
  days_A = 90 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3881_388189


namespace NUMINAMATH_CALUDE_smallest_cube_box_volume_for_cone_l3881_388180

/-- The volume of the smallest cube-shaped box that can accommodate a cone vertically -/
theorem smallest_cube_box_volume_for_cone (cone_height : ℝ) (cone_base_diameter : ℝ) 
  (h_height : cone_height = 15) 
  (h_diameter : cone_base_diameter = 8) : ℝ := by
  sorry

#check smallest_cube_box_volume_for_cone

end NUMINAMATH_CALUDE_smallest_cube_box_volume_for_cone_l3881_388180


namespace NUMINAMATH_CALUDE_range_of_sum_l3881_388191

theorem range_of_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : a + b + 1/a + 1/b = 5) : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_sum_l3881_388191


namespace NUMINAMATH_CALUDE_max_ab_value_l3881_388120

/-- Given a line ax + by - 6 = 0 (a > 0, b > 0) intercepted by the circle x^2 + y^2 - 2x - 4y = 0
    to form a chord of length 2√5, the maximum value of ab is 9/2 -/
theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ (x y : ℝ), a * x + b * y - 6 = 0 ∧ 
                x^2 + y^2 - 2*x - 4*y = 0 ∧ 
                ∃ (x1 y1 x2 y2 : ℝ), 
                  a * x1 + b * y1 - 6 = 0 ∧ 
                  x1^2 + y1^2 - 2*x1 - 4*y1 = 0 ∧
                  a * x2 + b * y2 - 6 = 0 ∧ 
                  x2^2 + y2^2 - 2*x2 - 4*y2 = 0 ∧
                  (x2 - x1)^2 + (y2 - y1)^2 = 20) →
  a * b ≤ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_value_l3881_388120


namespace NUMINAMATH_CALUDE_isosceles_triangles_independent_of_coloring_l3881_388174

/-- The number of isosceles triangles with vertices of the same color in a regular (6n+1)-gon -/
def num_isosceles_triangles (n : ℕ) (K : ℕ) : ℕ :=
  (1/2) * ((6*n+1 - K)*(6*n - K) + K*(K-1) - K*(6*n+1-K))

/-- Theorem stating that the number of isosceles triangles with vertices of the same color
    in a regular (6n+1)-gon is independent of the coloring scheme -/
theorem isosceles_triangles_independent_of_coloring (n : ℕ) (K : ℕ) 
    (h1 : K ≤ 6*n+1) : 
  ∀ (K' : ℕ), K' ≤ 6*n+1 → num_isosceles_triangles n K = num_isosceles_triangles n K' :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_independent_of_coloring_l3881_388174


namespace NUMINAMATH_CALUDE_line_divides_area_in_half_l3881_388130

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The L-shaped region defined by its vertices -/
def LShapedRegion : List Point2D := [
  ⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point2D) : ℝ :=
  sorry

/-- Calculate the area of a polygon formed by the origin and a line intersecting the L-shaped region -/
def areaAboveLine (slope : ℝ) : ℝ :=
  sorry

/-- The theorem stating that the line with slope 1/9 divides the L-shaped region in half -/
theorem line_divides_area_in_half :
  let totalArea := polygonArea LShapedRegion
  let slope := 1 / 9
  areaAboveLine slope = totalArea / 2 := by
  sorry

end NUMINAMATH_CALUDE_line_divides_area_in_half_l3881_388130


namespace NUMINAMATH_CALUDE_at_least_one_less_than_two_l3881_388136

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  min ((1 + y) / x) ((1 + x) / y) < 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_less_than_two_l3881_388136


namespace NUMINAMATH_CALUDE_division_problem_l3881_388112

theorem division_problem (L S Q : ℕ) (h1 : L - S = 1365) (h2 : L = 1631) (h3 : L = S * Q + 35) : Q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3881_388112


namespace NUMINAMATH_CALUDE_monotonic_cubic_implies_m_range_l3881_388119

/-- A function f : ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing. -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = x^3 + x^2 + mx + 1 is monotonic on ℝ, then m ≥ 1/3. -/
theorem monotonic_cubic_implies_m_range (m : ℝ) :
  Monotonic (fun x => x^3 + x^2 + m*x + 1) → m ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_cubic_implies_m_range_l3881_388119


namespace NUMINAMATH_CALUDE_root_equation_sum_l3881_388165

theorem root_equation_sum (a b c : ℝ) : 
  (∀ x : ℝ, x^2 + 3*x - 1 = 0 → x^4 + a*x^2 + b*x + c = 0) →
  a + b + 4*c = -7 := by
sorry

end NUMINAMATH_CALUDE_root_equation_sum_l3881_388165


namespace NUMINAMATH_CALUDE_crew_diff_1000_tons_crew_estimate_min_tonnage_crew_estimate_max_tonnage_l3881_388148

-- Define the regression equation
def crew_estimate (tonnage : ℝ) : ℝ := 9.5 + 0.0062 * tonnage

-- Define the tonnage range
def min_tonnage : ℝ := 192
def max_tonnage : ℝ := 3246

-- Theorem 1: Difference in crew members for 1000 tons difference
theorem crew_diff_1000_tons : 
  ∀ (x : ℝ), crew_estimate (x + 1000) - crew_estimate x = 6 := by sorry

-- Theorem 2: Estimated crew for minimum tonnage
theorem crew_estimate_min_tonnage : 
  ⌊crew_estimate min_tonnage⌋ = 11 := by sorry

-- Theorem 3: Estimated crew for maximum tonnage
theorem crew_estimate_max_tonnage : 
  ⌊crew_estimate max_tonnage⌋ = 30 := by sorry

end NUMINAMATH_CALUDE_crew_diff_1000_tons_crew_estimate_min_tonnage_crew_estimate_max_tonnage_l3881_388148


namespace NUMINAMATH_CALUDE_sum_fraction_denominator_form_main_result_l3881_388190

def double_factorial : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => (n + 2) * double_factorial n

def sum_fraction (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => (double_factorial (2 * (i + 1))) / (double_factorial (2 * (i + 1) + 1)))

theorem sum_fraction_denominator_form (n : ℕ) :
  ∃ (a b : ℕ), b % 2 = 1 ∧ (sum_fraction n).den = 2^a * b := by sorry

theorem main_result : ∃ (a b : ℕ), b % 2 = 1 ∧
  (sum_fraction 2010).den = 2^a * b ∧ (a * b) / 10 = 0 := by sorry

end NUMINAMATH_CALUDE_sum_fraction_denominator_form_main_result_l3881_388190


namespace NUMINAMATH_CALUDE_chocolate_bar_count_l3881_388160

/-- The number of small boxes in the large box -/
def num_small_boxes : ℕ := 17

/-- The number of chocolate bars in each small box -/
def choc_per_small_box : ℕ := 26

/-- The total number of chocolate bars in the large box -/
def total_chocolate_bars : ℕ := num_small_boxes * choc_per_small_box

theorem chocolate_bar_count : total_chocolate_bars = 442 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_count_l3881_388160


namespace NUMINAMATH_CALUDE_average_daily_low_temperature_l3881_388164

def daily_low_temperatures : List ℝ := [40, 47, 45, 41, 39, 43]

theorem average_daily_low_temperature :
  (daily_low_temperatures.sum / daily_low_temperatures.length : ℝ) = 42.5 := by
  sorry

end NUMINAMATH_CALUDE_average_daily_low_temperature_l3881_388164


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3881_388176

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | x^2 - 2*a*x - 3*a^2 < 0} = {x : ℝ | 3*a < x ∧ x < -a} :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3881_388176


namespace NUMINAMATH_CALUDE_vector_coordinates_l3881_388170

def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)
def C : ℝ × ℝ := (0, 3)

def vector (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

theorem vector_coordinates :
  vector A B = (-4, 3) ∧
  vector B C = (1, -2) ∧
  vector A C = (-3, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_coordinates_l3881_388170


namespace NUMINAMATH_CALUDE_henry_tic_tac_toe_games_l3881_388131

theorem henry_tic_tac_toe_games (wins losses draws : ℕ) 
  (h_wins : wins = 2)
  (h_losses : losses = 2)
  (h_draws : draws = 10) :
  wins + losses + draws = 14 := by
  sorry

end NUMINAMATH_CALUDE_henry_tic_tac_toe_games_l3881_388131


namespace NUMINAMATH_CALUDE_manuscript_cost_theorem_l3881_388167

/-- Calculates the total cost of typing and revising a manuscript. -/
def manuscript_cost (total_pages : ℕ) (first_time_cost : ℕ) (revision_cost : ℕ) 
  (revised_once : ℕ) (revised_twice : ℕ) (revised_thrice : ℕ) : ℕ :=
  total_pages * first_time_cost + 
  revised_once * revision_cost + 
  revised_twice * revision_cost * 2 + 
  revised_thrice * revision_cost * 3

theorem manuscript_cost_theorem : 
  manuscript_cost 500 5 4 200 150 50 = 5100 := by
  sorry

#eval manuscript_cost 500 5 4 200 150 50

end NUMINAMATH_CALUDE_manuscript_cost_theorem_l3881_388167


namespace NUMINAMATH_CALUDE_function_growth_l3881_388179

theorem function_growth (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, f x < deriv f x) (a : ℝ) (ha : 0 < a) : 
  f a > Real.exp a * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l3881_388179


namespace NUMINAMATH_CALUDE_matrix_sum_equality_l3881_388147

def A : Matrix (Fin 2) (Fin 2) ℚ := !![2/3, -1/2; 4, -5/2]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-5/6, 1/4; 3/2, -7/4]

theorem matrix_sum_equality : A + B = !![-1/6, -1/4; 11/2, -17/4] := by
  sorry

end NUMINAMATH_CALUDE_matrix_sum_equality_l3881_388147


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l3881_388140

theorem quadratic_equations_solutions :
  -- Equation 1
  (∃ x : ℝ, x^2 - 4 = 0) ∧
  (∀ x : ℝ, x^2 - 4 = 0 → x = 2 ∨ x = -2) ∧
  -- Equation 2
  (∃ x : ℝ, x^2 - 6*x + 9 = 0) ∧
  (∀ x : ℝ, x^2 - 6*x + 9 = 0 → x = 3) ∧
  -- Equation 3
  (∃ x : ℝ, x^2 - 7*x + 12 = 0) ∧
  (∀ x : ℝ, x^2 - 7*x + 12 = 0 → x = 3 ∨ x = 4) ∧
  -- Equation 4
  (∃ x : ℝ, 2*x^2 - 3*x = 5) ∧
  (∀ x : ℝ, 2*x^2 - 3*x = 5 → x = 5/2 ∨ x = -1) := by
  sorry


end NUMINAMATH_CALUDE_quadratic_equations_solutions_l3881_388140


namespace NUMINAMATH_CALUDE_largest_n_for_inequalities_l3881_388155

theorem largest_n_for_inequalities : ∃ (n : ℕ), n = 4 ∧ 
  (∃ (x : ℝ), ∀ (k : ℕ), k ≤ n → (k : ℝ) < x^k ∧ x^k < (k + 1 : ℝ)) ∧
  (∀ (m : ℕ), m > n → ¬∃ (x : ℝ), ∀ (k : ℕ), k ≤ m → (k : ℝ) < x^k ∧ x^k < (k + 1 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_inequalities_l3881_388155


namespace NUMINAMATH_CALUDE_scissors_count_l3881_388105

/-- The total number of scissors after adding more -/
def total_scissors (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem: The total number of scissors is 76 -/
theorem scissors_count : total_scissors 54 22 = 76 := by
  sorry

end NUMINAMATH_CALUDE_scissors_count_l3881_388105


namespace NUMINAMATH_CALUDE_antonov_remaining_packs_l3881_388187

/-- Calculates the number of candy packs remaining after giving away one pack -/
def remaining_packs (total_candies : ℕ) (candies_per_pack : ℕ) : ℕ :=
  (total_candies - candies_per_pack) / candies_per_pack

/-- Proves that Antonov has 2 packs of candy remaining -/
theorem antonov_remaining_packs :
  let total_candies : ℕ := 60
  let candies_per_pack : ℕ := 20
  remaining_packs total_candies candies_per_pack = 2 := by
  sorry

end NUMINAMATH_CALUDE_antonov_remaining_packs_l3881_388187


namespace NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l3881_388115

/-- Theorem: In a triangle where the angles are in the ratio 3:4:5, the largest angle measures 75°. -/
theorem largest_angle_in_ratio_triangle : ∀ (a b c : ℝ),
  -- The angles are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- The angles are in the ratio 3:4:5
  b = (4/3) * a ∧ c = (5/3) * a →
  -- The sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle (c) measures 75°
  c = 75 := by
sorry


end NUMINAMATH_CALUDE_largest_angle_in_ratio_triangle_l3881_388115


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3881_388128

theorem complex_equation_solution :
  ∃ z : ℂ, (5 - 3 * Complex.I * z = 2 + 5 * Complex.I * z) ∧ (z = -3 * Complex.I / 8) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3881_388128


namespace NUMINAMATH_CALUDE_prudence_total_sleep_l3881_388138

/-- Represents Prudence's sleep schedule --/
structure SleepSchedule where
  weekdaySleep : ℕ  -- Hours of sleep on weekdays (Sun-Thu)
  weekendSleep : ℕ  -- Hours of sleep on weekends (Fri-Sat)
  napDuration : ℕ   -- Duration of nap in hours
  napDays : ℕ       -- Number of days with naps
  weekdayNights : ℕ -- Number of weekday nights
  weekendNights : ℕ -- Number of weekend nights

/-- Calculates total sleep in 4 weeks given a sleep schedule --/
def totalSleepInFourWeeks (schedule : SleepSchedule) : ℕ :=
  4 * (schedule.weekdaySleep * schedule.weekdayNights +
       schedule.weekendSleep * schedule.weekendNights +
       schedule.napDuration * schedule.napDays)

/-- Prudence's actual sleep schedule --/
def prudenceSchedule : SleepSchedule :=
  { weekdaySleep := 6
  , weekendSleep := 9
  , napDuration := 1
  , napDays := 2
  , weekdayNights := 5
  , weekendNights := 2 }

/-- Theorem stating that Prudence's total sleep in 4 weeks is 200 hours --/
theorem prudence_total_sleep :
  totalSleepInFourWeeks prudenceSchedule = 200 := by
  sorry


end NUMINAMATH_CALUDE_prudence_total_sleep_l3881_388138


namespace NUMINAMATH_CALUDE_concatenated_not_palindromic_l3881_388151

/-- Represents the concatenation of integers from 1 to n as a natural number -/
def concatenatedNumber (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is palindromic -/
def isPalindromic (num : ℕ) : Prop := sorry

/-- Theorem stating that the concatenated number is not palindromic for n > 1 -/
theorem concatenated_not_palindromic (n : ℕ) (h : n > 1) : 
  ¬(isPalindromic (concatenatedNumber n)) := by sorry

end NUMINAMATH_CALUDE_concatenated_not_palindromic_l3881_388151


namespace NUMINAMATH_CALUDE_stamp_solution_l3881_388192

def stamp_problem (one_cent two_cent five_cent eight_cent : ℕ) : Prop :=
  two_cent = (3 * one_cent) / 4 ∧
  five_cent = (3 * two_cent) / 4 ∧
  eight_cent = 5 ∧
  one_cent * 1 + two_cent * 2 + five_cent * 5 + eight_cent * 8 = 100000

theorem stamp_solution :
  ∃ (one_cent two_cent five_cent eight_cent : ℕ),
    stamp_problem one_cent two_cent five_cent eight_cent ∧
    one_cent = 18816 ∧
    two_cent = 14112 ∧
    five_cent = 10584 ∧
    eight_cent = 5 :=
  sorry

end NUMINAMATH_CALUDE_stamp_solution_l3881_388192


namespace NUMINAMATH_CALUDE_park_is_square_l3881_388149

/-- A shape with a certain number of 90-degree angles -/
structure Shape :=
  (angles : ℕ)

/-- Definition of a square -/
def is_square (s : Shape) : Prop := s.angles = 4

theorem park_is_square (park : Shape) (square_field : Shape)
  (h1 : is_square square_field)
  (h2 : park.angles + square_field.angles = 8) :
  is_square park :=
sorry

end NUMINAMATH_CALUDE_park_is_square_l3881_388149


namespace NUMINAMATH_CALUDE_factor_x_squared_minus_169_l3881_388100

theorem factor_x_squared_minus_169 (x : ℝ) : x^2 - 169 = (x - 13) * (x + 13) := by
  sorry

end NUMINAMATH_CALUDE_factor_x_squared_minus_169_l3881_388100


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3881_388152

theorem tenth_term_of_sequence (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h : ∀ k, S k = k^2 + 2*k) : 
  a 10 = 21 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3881_388152


namespace NUMINAMATH_CALUDE_cube_plus_inverse_cube_l3881_388142

theorem cube_plus_inverse_cube (a : ℝ) (h : (a + 1 / (3 * a))^2 = 3) : 
  27 * a^3 + 1 / a^3 = 54 * Real.sqrt 3 ∨ 27 * a^3 + 1 / a^3 = -54 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_plus_inverse_cube_l3881_388142


namespace NUMINAMATH_CALUDE_problem_solution_l3881_388163

theorem problem_solution (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 - y^2 = 3*x*y) :
  x^2/y^2 + y^2/x^2 - 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3881_388163


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l3881_388109

theorem triangle_ABC_properties (A B C : ℝ) (p : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle ABC exists
  (∃ x y : ℝ, x^2 + (x+1)*p + 1 = 0 ∧ y^2 + (y+1)*p + 1 = 0 ∧ x = Real.tan A ∧ y = Real.tan B) →
  C = 3*π/4 ∧ p ∈ Set.Ioo (-2 : ℝ) (2 - 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l3881_388109


namespace NUMINAMATH_CALUDE_first_or_third_quadrant_set_l3881_388127

def first_or_third_quadrant (α : ℝ) : Prop :=
  (∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2) ∨
  (∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2)

theorem first_or_third_quadrant_set : 
  {α : ℝ | first_or_third_quadrant α} = 
  {α : ℝ | ∃ k : ℤ, 2 * k * Real.pi < α ∧ α < 2 * k * Real.pi + Real.pi / 2} ∪
  {α : ℝ | ∃ k : ℤ, 2 * k * Real.pi + Real.pi < α ∧ α < 2 * k * Real.pi + 3 * Real.pi / 2} :=
by sorry

end NUMINAMATH_CALUDE_first_or_third_quadrant_set_l3881_388127


namespace NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l3881_388110

/-- The number of ways to distribute n distinguishable balls into k indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 187 ways to distribute 6 distinguishable balls into 4 indistinguishable boxes -/
theorem distribute_six_balls_four_boxes : distribute_balls 6 4 = 187 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_four_boxes_l3881_388110


namespace NUMINAMATH_CALUDE_consecutive_points_distance_l3881_388144

/-- Given 5 consecutive points on a straight line, prove that ac = 11 -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (c - b = 3 * (d - c)) →  -- bc = 3 * cd
  (e - d = 8) →            -- de = 8
  (b - a = 5) →            -- ab = 5
  (e - a = 21) →           -- ae = 21
  (c - a = 11) :=          -- ac = 11
by sorry

end NUMINAMATH_CALUDE_consecutive_points_distance_l3881_388144


namespace NUMINAMATH_CALUDE_replacement_cost_100_movies_l3881_388104

/-- The cost to replace VHS movies with DVDs -/
def replacement_cost (num_movies : ℕ) (vhs_trade_value : ℚ) (dvd_cost : ℚ) : ℚ :=
  num_movies * dvd_cost - num_movies * vhs_trade_value

/-- Theorem: The cost to replace 100 VHS movies with DVDs is $800 -/
theorem replacement_cost_100_movies :
  replacement_cost 100 2 10 = 800 := by
  sorry

end NUMINAMATH_CALUDE_replacement_cost_100_movies_l3881_388104


namespace NUMINAMATH_CALUDE_radical_simplification_l3881_388161

theorem radical_simplification :
  Real.sqrt (4 - 2 * Real.sqrt 3) - Real.sqrt (4 + 2 * Real.sqrt 3) = -2 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3881_388161


namespace NUMINAMATH_CALUDE_solve_fraction_equation_l3881_388107

theorem solve_fraction_equation (x : ℝ) : (1 / 3 - 1 / 4 : ℝ) = 1 / x → x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solve_fraction_equation_l3881_388107


namespace NUMINAMATH_CALUDE_f_range_l3881_388137

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 2*x

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- Define the range
def range : Set ℝ := {y | ∃ x ∈ domain, f x = y}

-- Theorem statement
theorem f_range : range = {y | -1 ≤ y ∧ y ≤ 3} := by sorry

end NUMINAMATH_CALUDE_f_range_l3881_388137


namespace NUMINAMATH_CALUDE_water_tank_capacity_l3881_388166

theorem water_tank_capacity (initial_fraction : Rat) (added_volume : ℝ) (final_fraction : Rat) :
  initial_fraction = 1/3 →
  added_volume = 5 →
  final_fraction = 2/5 →
  ∃ (capacity : ℝ), capacity = 75 ∧ 
    initial_fraction * capacity + added_volume = final_fraction * capacity :=
by sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l3881_388166


namespace NUMINAMATH_CALUDE_optimal_screen_arrangement_l3881_388139

/-- The optimal arrangement of two screens in a corner --/
theorem optimal_screen_arrangement (screen_length : ℝ) (h_length : screen_length = 4) :
  let max_area := 8 * (Real.sqrt 2 + 1)
  let optimal_angle := π / 4
  ∀ angle : ℝ, 0 < angle ∧ angle < π / 2 →
    screen_length * screen_length * Real.sin angle / 2 ≤ max_area ∧
    (screen_length * screen_length * Real.sin angle / 2 = max_area ↔ angle = optimal_angle) :=
by sorry

end NUMINAMATH_CALUDE_optimal_screen_arrangement_l3881_388139


namespace NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l3881_388185

theorem max_value_x_sqrt_1_minus_4x_squared :
  (∃ (x : ℝ), x > 0 ∧ x * Real.sqrt (1 - 4 * x^2) = 1/4) ∧
  (∀ (x : ℝ), x > 0 → x * Real.sqrt (1 - 4 * x^2) ≤ 1/4) := by
  sorry

end NUMINAMATH_CALUDE_max_value_x_sqrt_1_minus_4x_squared_l3881_388185


namespace NUMINAMATH_CALUDE_value_of_expression_l3881_388108

theorem value_of_expression (a : ℝ) (h : a - 1/a = 1) : a^2 - a + 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l3881_388108


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l3881_388141

/-- Fuel efficiency of a car on highway and in city -/
structure CarFuelEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank : ℝ     -- Tank capacity in gallons
  h_positive : highway > 0
  c_positive : city > 0
  t_positive : tank > 0
  city_less : city = highway - 6

/-- Theorem stating the car's fuel efficiency in the city is 18 mpg -/
theorem city_fuel_efficiency 
  (car : CarFuelEfficiency)
  (h_highway : car.highway * car.tank = 448)
  (h_city : car.city * car.tank = 336) :
  car.city = 18 := by
  sorry

end NUMINAMATH_CALUDE_city_fuel_efficiency_l3881_388141


namespace NUMINAMATH_CALUDE_chocolate_mixture_proof_l3881_388124

theorem chocolate_mixture_proof (initial_weight : ℝ) (initial_percentage : ℝ) 
  (final_weight : ℝ) (final_percentage : ℝ) (added_pure_chocolate : ℝ) : 
  initial_weight = 620 →
  initial_percentage = 0.1 →
  final_weight = 1000 →
  final_percentage = 0.7 →
  added_pure_chocolate = 638 →
  (initial_weight * initial_percentage + added_pure_chocolate) / final_weight = final_percentage :=
by
  sorry

#check chocolate_mixture_proof

end NUMINAMATH_CALUDE_chocolate_mixture_proof_l3881_388124


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3881_388153

theorem complex_equation_solution (i : ℂ) (z : ℂ) (h1 : i * i = -1) (h2 : i * z = 1) :
  z = -i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3881_388153


namespace NUMINAMATH_CALUDE_density_of_M_l3881_388123

def M : Set ℝ :=
  {r : ℝ | ∃ (m n : ℕ+), r = (m + n) / Real.sqrt (m^2 + n^2)}

theorem density_of_M : ∀ (x y : ℝ), x ∈ M → y ∈ M → x < y →
  ∃ (z : ℝ), z ∈ M ∧ x < z ∧ z < y :=
by sorry

end NUMINAMATH_CALUDE_density_of_M_l3881_388123


namespace NUMINAMATH_CALUDE_opposite_of_neg_six_l3881_388188

/-- The opposite of a real number -/
def opposite (a : ℝ) : ℝ := -a

/-- Theorem: The opposite of -6 is 6 -/
theorem opposite_of_neg_six : opposite (-6) = 6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_neg_six_l3881_388188


namespace NUMINAMATH_CALUDE_garden_vegetable_difference_l3881_388172

/-- Represents the number of vegetables in a garden -/
structure GardenVegetables where
  potatoes : ℕ
  cucumbers : ℕ
  peppers : ℕ

/-- Theorem stating the difference between potatoes and cucumbers in the garden -/
theorem garden_vegetable_difference (g : GardenVegetables) :
  g.potatoes = 237 →
  g.peppers = 2 * g.cucumbers →
  g.potatoes + g.cucumbers + g.peppers = 768 →
  g.potatoes - g.cucumbers = 60 := by
  sorry

#check garden_vegetable_difference

end NUMINAMATH_CALUDE_garden_vegetable_difference_l3881_388172


namespace NUMINAMATH_CALUDE_not_decreasing_if_f0_lt_f4_l3881_388103

def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

theorem not_decreasing_if_f0_lt_f4 (f : ℝ → ℝ) (h : f 0 < f 4) : ¬ IsDecreasing f := by
  sorry

end NUMINAMATH_CALUDE_not_decreasing_if_f0_lt_f4_l3881_388103


namespace NUMINAMATH_CALUDE_solve_square_equation_solve_cubic_equation_l3881_388116

-- Part 1
theorem solve_square_equation :
  ∀ x : ℝ, (x - 1)^2 = 9 ↔ x = 4 ∨ x = -2 :=
by sorry

-- Part 2
theorem solve_cubic_equation :
  ∀ x : ℝ, (1/3) * (x + 3)^3 - 9 = 0 ↔ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_solve_square_equation_solve_cubic_equation_l3881_388116


namespace NUMINAMATH_CALUDE_complement_union_theorem_l3881_388114

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Set Nat := {1, 2}

-- Define set B
def B : Set Nat := {2, 3, 4}

-- Theorem statement
theorem complement_union_theorem :
  (Set.compl A).union B = {2, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l3881_388114


namespace NUMINAMATH_CALUDE_spinner_probability_l3881_388186

theorem spinner_probability (p_largest p_next_largest p_smallest : ℝ) : 
  p_largest = (1 : ℝ) / 2 →
  p_next_largest = (1 : ℝ) / 3 →
  p_largest + p_next_largest + p_smallest = 1 →
  p_smallest = (1 : ℝ) / 6 := by
sorry

end NUMINAMATH_CALUDE_spinner_probability_l3881_388186


namespace NUMINAMATH_CALUDE_tournament_games_32_teams_l3881_388198

/-- The number of games needed in a single-elimination tournament to declare a winner -/
def games_needed (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 1

/-- Theorem: In a single-elimination tournament with 32 teams, 31 games are needed to declare a winner -/
theorem tournament_games_32_teams :
  games_needed 32 = 31 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_32_teams_l3881_388198


namespace NUMINAMATH_CALUDE_marble_remainder_l3881_388194

theorem marble_remainder (r p : ℕ) : 
  r % 8 = 5 → p % 8 = 6 → (r + p) % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_remainder_l3881_388194


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_fifteen_is_dual_base_palindrome_fifteen_is_smallest_dual_base_palindrome_l3881_388168

/-- Checks if a natural number is a palindrome in the given base. -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  sorry

/-- Converts a natural number to its representation in the given base. -/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

theorem smallest_dual_base_palindrome : 
  ∀ n : ℕ, n > 10 → 
    (isPalindrome n 2 ∧ isPalindrome n 4) → 
    n ≥ 15 :=
by sorry

theorem fifteen_is_dual_base_palindrome : 
  isPalindrome 15 2 ∧ isPalindrome 15 4 :=
by sorry

theorem fifteen_is_smallest_dual_base_palindrome : 
  ∀ n : ℕ, n > 10 → 
    (isPalindrome n 2 ∧ isPalindrome n 4) → 
    n = 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_fifteen_is_dual_base_palindrome_fifteen_is_smallest_dual_base_palindrome_l3881_388168


namespace NUMINAMATH_CALUDE_range_of_a_l3881_388156

def prop_p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a - 1) * x + a^2 > 0

def prop_q (a : ℝ) : Prop :=
  ∀ x y, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

theorem range_of_a (a : ℝ) :
  (prop_p a ∨ prop_q a) ∧ ¬(prop_p a ∧ prop_q a) →
  (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3881_388156


namespace NUMINAMATH_CALUDE_b_completes_in_20_days_l3881_388159

/-- The number of days it takes for worker A to complete the work alone -/
def days_a : ℝ := 15

/-- The number of days A and B work together -/
def days_together : ℝ := 7

/-- The fraction of work left after A and B work together -/
def work_left : ℝ := 0.18333333333333335

/-- The number of days it takes for worker B to complete the work alone -/
def days_b : ℝ := 20

/-- Theorem stating that given the conditions, B can complete the work in 20 days -/
theorem b_completes_in_20_days :
  (days_together * (1 / days_a + 1 / days_b) = 1 - work_left) →
  days_b = 20 := by
  sorry

end NUMINAMATH_CALUDE_b_completes_in_20_days_l3881_388159


namespace NUMINAMATH_CALUDE_jessie_weight_loss_l3881_388173

/-- Calculates the weight loss for Jessie based on her exercise routine --/
def weight_loss (initial_weight : ℝ) (exercise_days : ℕ) (even_day_loss : ℝ) (odd_day_loss : ℝ) : ℝ :=
  let even_days := (exercise_days - 1) / 2
  let odd_days := exercise_days - even_days
  even_days * even_day_loss + odd_days * odd_day_loss

/-- Theorem stating that Jessie's weight loss is 8.1 kg --/
theorem jessie_weight_loss :
  let initial_weight : ℝ := 74
  let exercise_days : ℕ := 25
  let even_day_loss : ℝ := 0.2 + 0.15
  let odd_day_loss : ℝ := 0.3
  weight_loss initial_weight exercise_days even_day_loss odd_day_loss = 8.1 := by
  sorry

#eval weight_loss 74 25 (0.2 + 0.15) 0.3

end NUMINAMATH_CALUDE_jessie_weight_loss_l3881_388173


namespace NUMINAMATH_CALUDE_complex_modulus_product_l3881_388135

theorem complex_modulus_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_product_l3881_388135


namespace NUMINAMATH_CALUDE_angle_measure_proof_l3881_388121

theorem angle_measure_proof (C D : ℝ) : 
  C + D = 180 →  -- Angles are supplementary
  C = 9 * D →    -- C is 9 times D
  C = 162 :=     -- The measure of angle C is 162 degrees
by
  sorry

end NUMINAMATH_CALUDE_angle_measure_proof_l3881_388121


namespace NUMINAMATH_CALUDE_meaningful_expression_l3881_388143

theorem meaningful_expression (a : ℝ) : 
  (∃ x : ℝ, x = (Real.sqrt (a + 1)) / (a - 2)) ↔ (a ≥ -1 ∧ a ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_l3881_388143


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l3881_388178

theorem triangle_angle_measure (D E F : ℝ) : 
  D = 80 →  -- Measure of angle D is 80 degrees
  E = 4 * F + 10 →  -- Measure of angle E is 10 degrees more than four times the measure of angle F
  D + E + F = 180 →  -- Sum of angles in a triangle is 180 degrees
  F = 18 :=  -- Measure of angle F is 18 degrees
by sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l3881_388178


namespace NUMINAMATH_CALUDE_average_weight_of_twenty_boys_l3881_388101

theorem average_weight_of_twenty_boys 
  (num_group1 : ℕ) 
  (num_group2 : ℕ) 
  (avg_weight_group2 : ℝ) 
  (avg_weight_all : ℝ) :
  num_group1 = 20 →
  num_group2 = 8 →
  avg_weight_group2 = 45.15 →
  avg_weight_all = 48.792857142857144 →
  (num_group1 * 50.25 + num_group2 * avg_weight_group2) / (num_group1 + num_group2) = avg_weight_all :=
by sorry

end NUMINAMATH_CALUDE_average_weight_of_twenty_boys_l3881_388101


namespace NUMINAMATH_CALUDE_triangle_rotation_l3881_388193

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle with vertices O, P, and Q -/
structure Triangle where
  O : Point
  P : Point
  Q : Point

/-- Calculates the angle between two vectors -/
def angle (v1 v2 : Point) : ℝ := sorry

/-- Rotates a point 90 degrees counter-clockwise around the origin -/
def rotate90 (p : Point) : Point :=
  { x := -p.y, y := p.x }

/-- The main theorem -/
theorem triangle_rotation (t : Triangle) : 
  t.O = ⟨0, 0⟩ → 
  t.P = ⟨7, 0⟩ → 
  t.Q.x > 0 → 
  t.Q.y > 0 → 
  angle t.P t.Q = π / 2 → 
  angle t.P t.Q - angle t.O t.Q = π / 4 → 
  rotate90 t.Q = ⟨-7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2⟩ := by sorry

end NUMINAMATH_CALUDE_triangle_rotation_l3881_388193


namespace NUMINAMATH_CALUDE_min_value_of_2a_plus_1_l3881_388118

theorem min_value_of_2a_plus_1 (a : ℝ) (h : 9*a^2 + 7*a + 5 = 2) : 
  ∃ (min : ℝ), min = -1 ∧ ∀ (x : ℝ), 9*x^2 + 7*x + 5 = 2 → 2*x + 1 ≥ min := by
sorry

end NUMINAMATH_CALUDE_min_value_of_2a_plus_1_l3881_388118


namespace NUMINAMATH_CALUDE_exact_number_range_l3881_388199

/-- Represents a number rounded to one decimal place -/
def RoundedToOneDecimal (x : ℝ) : Prop :=
  ∃ (n : ℤ), x = (n : ℝ) / 10

/-- The range of an exact number represented by 3.4 when rounded to one decimal place -/
theorem exact_number_range (a : ℝ) (h : RoundedToOneDecimal 3.4) :
  (RoundedToOneDecimal a ∧ a = 3.4) → 3.35 ≤ a ∧ a < 3.45 :=
by sorry

end NUMINAMATH_CALUDE_exact_number_range_l3881_388199


namespace NUMINAMATH_CALUDE_moms_balloons_l3881_388183

/-- The number of balloons Tommy's mom gave him -/
def balloons_from_mom (initial_balloons final_balloons : ℕ) : ℕ :=
  final_balloons - initial_balloons

/-- Proof that Tommy's mom gave him 34 balloons -/
theorem moms_balloons : balloons_from_mom 26 60 = 34 := by
  sorry

end NUMINAMATH_CALUDE_moms_balloons_l3881_388183


namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l3881_388146

theorem longest_side_of_triangle (a b c : ℝ) (perimeter : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a / b = 3 / 2 →
  a / c = 2 →
  b / c = 4 / 3 →
  a + b + c = perimeter →
  perimeter = 104 →
  a = 48 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l3881_388146


namespace NUMINAMATH_CALUDE_muffins_per_pack_is_four_l3881_388181

/-- Represents the muffin selling problem --/
structure MuffinProblem where
  total_amount : ℕ -- Total amount to raise in dollars
  muffin_price : ℕ -- Price of each muffin in dollars
  num_cases : ℕ -- Number of cases to sell
  packs_per_case : ℕ -- Number of packs in each case

/-- Calculates the number of muffins in each pack --/
def muffins_per_pack (p : MuffinProblem) : ℕ :=
  (p.total_amount / p.muffin_price) / (p.num_cases * p.packs_per_case)

/-- Theorem stating that the number of muffins per pack is 4 --/
theorem muffins_per_pack_is_four (p : MuffinProblem) 
  (h1 : p.total_amount = 120)
  (h2 : p.muffin_price = 2)
  (h3 : p.num_cases = 5)
  (h4 : p.packs_per_case = 3) : 
  muffins_per_pack p = 4 := by
  sorry

end NUMINAMATH_CALUDE_muffins_per_pack_is_four_l3881_388181


namespace NUMINAMATH_CALUDE_common_chord_equation_l3881_388133

/-- The equation of the line where the common chord of two circles lies -/
theorem common_chord_equation (x y : ℝ) :
  (x^2 + y^2 + 4*x - 6*y + 12 = 0) ∧ (x^2 + y^2 - 2*x - 14*y + 15 = 0) →
  (6*x + 8*y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_equation_l3881_388133


namespace NUMINAMATH_CALUDE_overlap_area_of_rectangles_l3881_388158

theorem overlap_area_of_rectangles (a b x y : ℝ) : 
  a = 3 ∧ b = 9 ∧  -- Rectangle dimensions
  x^2 + a^2 = y^2 ∧ -- Pythagorean theorem for the corner triangle
  x + y = b ∧ -- Sum of triangle sides equals longer rectangle side
  0 < x ∧ 0 < y -- Positive lengths
  → (b * a - 2 * (x * a / 2)) = 15 := by sorry

end NUMINAMATH_CALUDE_overlap_area_of_rectangles_l3881_388158


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3881_388106

theorem sin_alpha_value (α β : Real) 
  (h1 : (0 : Real) < α ∧ α < Real.pi / 2)
  (h2 : -Real.pi / 2 < β ∧ β < 0)
  (h3 : Real.sin β = -5 / 13)
  (h4 : Real.sqrt ((Real.cos α - Real.cos β)^2 + (Real.sin α - Real.sin β)^2) = 2 * Real.sqrt 5 / 5) :
  Real.sin α = 33 / 65 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3881_388106


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l3881_388113

theorem no_solutions_in_interval (x : Real) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi → 1 / Real.sin x + 1 / Real.cos x ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l3881_388113


namespace NUMINAMATH_CALUDE_tangent_double_angle_subtraction_l3881_388111

theorem tangent_double_angle_subtraction (α β : ℝ) 
  (h1 : Real.tan (α - β) = 2/5) 
  (h2 : Real.tan β = 1/2) : 
  Real.tan (α - 2*β) = -1/12 := by
  sorry

end NUMINAMATH_CALUDE_tangent_double_angle_subtraction_l3881_388111


namespace NUMINAMATH_CALUDE_circle_equation_correct_circle_properties_l3881_388195

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if a point lies on a circle -/
def lies_on_circle (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- The specific circle we're considering -/
def our_circle : Circle :=
  { center := (0, 3)
    radius := 1 }

theorem circle_equation_correct :
  ∀ x y : ℝ, x^2 + (y - 3)^2 = 1 ↔ lies_on_circle our_circle (x, y) :=
sorry

theorem circle_properties :
  our_circle.center.1 = 0 ∧
  our_circle.radius = 1 ∧
  lies_on_circle our_circle (1, 3) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_correct_circle_properties_l3881_388195


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3881_388196

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π/2) (h2 : Real.sin (α - π/6) = 1/3) :
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3881_388196


namespace NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3881_388125

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 20) (h2 : x * y = 16) : x^2 + y^2 = 432 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_from_difference_and_product_l3881_388125


namespace NUMINAMATH_CALUDE_prime_binomial_divisibility_l3881_388171

theorem prime_binomial_divisibility (m : ℕ) (h : m ≥ 2) :
  (∀ n : ℕ, m / 3 ≤ n ∧ n ≤ m / 2 → n ∣ Nat.choose n (m - 2*n)) ↔ Nat.Prime m :=
sorry

end NUMINAMATH_CALUDE_prime_binomial_divisibility_l3881_388171


namespace NUMINAMATH_CALUDE_farm_area_calculation_l3881_388126

/-- Given a farm divided into sections, calculate its total area -/
def farm_total_area (num_sections : ℕ) (section_area : ℕ) : ℕ :=
  num_sections * section_area

/-- Theorem: The total area of a farm with 5 sections of 60 acres each is 300 acres -/
theorem farm_area_calculation : farm_total_area 5 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_farm_area_calculation_l3881_388126


namespace NUMINAMATH_CALUDE_rock_skipping_total_l3881_388184

/-- The number of times Bob can skip a rock -/
def bob_skips : ℕ := 12

/-- The number of times Jim can skip a rock -/
def jim_skips : ℕ := 15

/-- The number of rocks each person skipped -/
def rocks_skipped : ℕ := 10

/-- The total number of skips achieved by Bob and Jim -/
def total_skips : ℕ := bob_skips * rocks_skipped + jim_skips * rocks_skipped

theorem rock_skipping_total : total_skips = 270 := by
  sorry

end NUMINAMATH_CALUDE_rock_skipping_total_l3881_388184


namespace NUMINAMATH_CALUDE_binomial_10_3_l3881_388177

theorem binomial_10_3 : Nat.choose 10 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_3_l3881_388177
