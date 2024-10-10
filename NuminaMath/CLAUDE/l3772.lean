import Mathlib

namespace two_from_four_one_from_pair_l3772_377281

/-- The number of ways to select 2 students from a group of 4, where exactly one is chosen from a specific pair --/
theorem two_from_four_one_from_pair : ℕ := by
  sorry

end two_from_four_one_from_pair_l3772_377281


namespace f_properties_l3772_377219

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1 - 2 * (a ^ x) - (a ^ (2 * x))

theorem f_properties (a : ℝ) (h_a : a > 1) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y ∧ y < 1) ∧
  (∃ x₀ : ℝ, x₀ ∈ Set.Icc (-2) 1 ∧ f a x₀ = -7 →
    a = 2 ∧ ∃ x_max : ℝ, x_max ∈ Set.Icc (-2) 1 ∧ f a x_max = 7/16 ∧
    ∀ x : ℝ, x ∈ Set.Icc (-2) 1 → f a x ≤ 7/16) :=
sorry

end f_properties_l3772_377219


namespace rectangle_area_l3772_377243

/-- A rectangle divided into three identical squares with a perimeter of 120 cm has an area of 675 square centimeters. -/
theorem rectangle_area (side : ℝ) : 
  (8 * side = 120) →  -- perimeter condition
  (3 * side * side = 675) -- area calculation
  := by sorry

end rectangle_area_l3772_377243


namespace problem_solution_l3772_377291

-- Define the set B
def B : Set ℝ := {m | ∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x - m < 0}

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | (x - 3*a) * (x - a - 2) < 0}

-- Main theorem
theorem problem_solution (a : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, x^2 - x - 2 < 0) ∧
  (a < 1) ∧
  (A a ⊆ B) ∧
  (A a ≠ B) →
  (B = Set.Ioi 2) ∧
  (2/3 ≤ a ∧ a < 1) :=
sorry

end problem_solution_l3772_377291


namespace train_bridge_crossing_time_l3772_377255

/-- The time required for a train to cross a bridge -/
theorem train_bridge_crossing_time 
  (train_length : Real) 
  (train_speed_kmh : Real) 
  (bridge_length : Real) 
  (h1 : train_length = 100) 
  (h2 : train_speed_kmh = 45) 
  (h3 : bridge_length = 275) : 
  (train_length + bridge_length) / (train_speed_kmh * 1000 / 3600) = 30 := by
  sorry

end train_bridge_crossing_time_l3772_377255


namespace cubic_factorization_l3772_377252

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end cubic_factorization_l3772_377252


namespace coffee_mix_ratio_l3772_377237

theorem coffee_mix_ratio (x y : ℝ) (h : x > 0 ∧ y > 0) :
  (50 * x + 40 * y) / (x + y) = (55 * x + 34 * y) / (x + y) ↔ x / y = 6 / 5 :=
by sorry

end coffee_mix_ratio_l3772_377237


namespace augmented_matrix_solution_l3772_377239

/-- Given an augmented matrix and its solution, prove that c₁ - c₂ = -1 -/
theorem augmented_matrix_solution (c₁ c₂ : ℝ) : 
  (2 * 2 + 3 * 1 = c₁) → 
  (3 * 2 + 2 * 1 = c₂) → 
  c₁ - c₂ = -1 := by
sorry

end augmented_matrix_solution_l3772_377239


namespace absolute_value_inequality_l3772_377234

theorem absolute_value_inequality (a : ℝ) : 
  (∀ x : ℝ, |2*x - 3| - 2*a > |x + a|) ↔ -3/2 ≤ a ∧ a < -1/2 :=
by sorry

end absolute_value_inequality_l3772_377234


namespace first_condition_second_condition_l3772_377215

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem for the first condition
theorem first_condition (a : ℝ) : 
  (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -2 := by sorry

-- Theorem for the second condition
theorem second_condition (a : ℝ) :
  (A a ∩ B = A a ∩ C) ∧ (A a ∩ B ≠ ∅) → a = -3 := by sorry

end first_condition_second_condition_l3772_377215


namespace lines_cannot_form_triangle_l3772_377208

-- Define the lines
def l₁ (x y : ℝ) : Prop := 4 * x + y = 4
def l₂ (m x y : ℝ) : Prop := m * x + y = 0
def l₃ (m x y : ℝ) : Prop := 2 * x - 3 * m * y = 4

-- Define when lines are parallel
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂

-- Define when three lines intersect at a single point
def intersect_at_point (m : ℝ) : Prop :=
  ∃ x y : ℝ, l₁ x y ∧ l₂ m x y ∧ l₃ m x y

-- Theorem statement
theorem lines_cannot_form_triangle (m : ℝ) : 
  (¬∃ x₁ y₁ x₂ y₂ x₃ y₃ : ℝ, 
    l₁ x₁ y₁ ∧ l₂ m x₂ y₂ ∧ l₃ m x₃ y₃ ∧ 
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧ (x₂ ≠ x₃ ∨ y₂ ≠ y₃) ∧ (x₃ ≠ x₁ ∨ y₃ ≠ y₁)) ↔ 
  (m = 4 ∨ m = -1/6 ∨ m = -1 ∨ m = 2/3) :=
sorry

end lines_cannot_form_triangle_l3772_377208


namespace distance_point_to_line_l3772_377209

/-- Given a line in polar form and a point in polar coordinates, 
    calculate the distance from the point to the line. -/
theorem distance_point_to_line 
  (ρ θ : ℝ) -- polar coordinates of the point
  (h_line : ∀ (ρ' θ' : ℝ), 2 * ρ' * Real.sin (θ' - π/4) = Real.sqrt 2) -- line equation
  (h_point : ρ = 2 * Real.sqrt 2 ∧ θ = 7 * π/4) -- point coordinates
  : let x := ρ * Real.cos θ
    let y := ρ * Real.sin θ
    (y - x - 1) / Real.sqrt 2 = 3 * Real.sqrt 2 / 2 :=
by sorry

end distance_point_to_line_l3772_377209


namespace unique_positive_integer_l3772_377223

theorem unique_positive_integer : ∃! (n : ℕ), n > 0 ∧ 15 * n = n^2 + 56 := by
  sorry

end unique_positive_integer_l3772_377223


namespace books_to_buy_l3772_377236

/-- Given that 3 books cost $18.72 and you have $37.44, prove that you can buy 6 books. -/
theorem books_to_buy (cost_of_three : ℝ) (total_money : ℝ) : 
  cost_of_three = 18.72 → total_money = 37.44 → 
  (total_money / (cost_of_three / 3)) = 6 := by
sorry

end books_to_buy_l3772_377236


namespace y_min_at_a_or_b_l3772_377275

/-- The function y in terms of x, a, and b -/
def y (x a b : ℝ) : ℝ := (x - a)^3 + (x - b)^3

/-- Theorem stating that the minimum of y occurs at either a or b -/
theorem y_min_at_a_or_b (a b : ℝ) :
  ∃ (x : ℝ), (∀ (z : ℝ), y z a b ≥ y x a b) ∧ (x = a ∨ x = b) := by
  sorry

end y_min_at_a_or_b_l3772_377275


namespace complete_graph_10_coloring_l3772_377231

/-- A complete graph with 10 vertices -/
def CompleteGraph10 := Fin 10

/-- The type of edge colorings for CompleteGraph10 -/
def EdgeColoring (k : ℕ) := CompleteGraph10 → CompleteGraph10 → Fin k

/-- Predicate to check if k vertices form a k-colored subgraph -/
def is_k_colored_subgraph (k : ℕ) (coloring : EdgeColoring k) (vertices : Finset CompleteGraph10) : Prop :=
  vertices.card = k ∧
  ∀ (v w : CompleteGraph10), v ∈ vertices → w ∈ vertices → v ≠ w →
    ∃ (c : Fin k), ∀ (x y : CompleteGraph10), x ∈ vertices → y ∈ vertices → x ≠ y →
      coloring x y = c → x = v ∧ y = w

/-- Main theorem: k-coloring of CompleteGraph10 is possible iff k ≥ 5 -/
theorem complete_graph_10_coloring (k : ℕ) :
  (∃ (coloring : EdgeColoring k),
    ∀ (vertices : Finset CompleteGraph10),
      vertices.card = k → is_k_colored_subgraph k coloring vertices) ↔
  k ≥ 5 :=
sorry

end complete_graph_10_coloring_l3772_377231


namespace common_tangent_parabola_log_l3772_377229

theorem common_tangent_parabola_log (a s t : ℝ) : 
  a > 0 → 
  t = a * s^2 → 
  t = Real.log s → 
  (2 * a * s) = (1 / s) → 
  a = 1 / (2 * Real.exp 1) := by
sorry

end common_tangent_parabola_log_l3772_377229


namespace line_segment_endpoint_l3772_377279

/-- Given a line segment from (3, 7) to (-9, y) with length 15 and y > 0, prove y = 16 -/
theorem line_segment_endpoint (y : ℝ) : 
  (((3 : ℝ) - (-9))^2 + (y - 7)^2 = 15^2) → y > 0 → y = 16 :=
by sorry

end line_segment_endpoint_l3772_377279


namespace angle_DAC_measure_l3772_377220

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the point D
def D (t : Triangle) : Point := sorry

-- Define the angles
def angle_BAC (t : Triangle) : ℝ := sorry
def angle_ABC (t : Triangle) : ℝ := sorry
def angle_DAC (t : Triangle) : ℝ := sorry

-- Define the lengths
def length_DA (t : Triangle) : ℝ := sorry
def length_CB (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_DAC_measure (t : Triangle) 
  (h1 : length_DA t = length_CB t)
  (h2 : angle_BAC t = 70)
  (h3 : angle_ABC t = 55) :
  angle_DAC t = 100 := by sorry

end angle_DAC_measure_l3772_377220


namespace equation_solutions_l3772_377213

theorem equation_solutions : ∃ (x₁ x₂ : ℝ), 
  (x₁ - 1) * (x₁ + 3) = 12 ∧ 
  (x₂ - 1) * (x₂ + 3) = 12 ∧ 
  x₁ = -5 ∧ 
  x₂ = 3 := by
  sorry

end equation_solutions_l3772_377213


namespace cube_vertex_distance_to_plane_l3772_377271

/-- Given a cube with side length 15 and three vertices adjacent to vertex A
    at heights 15, 17, and 18 above a plane, the distance from vertex A to the plane is 28/3 -/
theorem cube_vertex_distance_to_plane :
  ∀ (a b c d : ℝ),
  a^2 + b^2 + c^2 = 1 →
  15 * a + d = 15 →
  15 * b + d = 17 →
  15 * c + d = 18 →
  d = 28 / 3 :=
by sorry

end cube_vertex_distance_to_plane_l3772_377271


namespace regular_polygon_sides_l3772_377283

/-- For a regular polygon with an exterior angle of 36°, the number of sides is 10. -/
theorem regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : 
  exterior_angle = 36 → n * exterior_angle = 360 → n = 10 := by
  sorry

end regular_polygon_sides_l3772_377283


namespace fancy_sandwich_cost_l3772_377282

/-- The cost of a fancy ham and cheese sandwich given Teresa's shopping list and total spent --/
theorem fancy_sandwich_cost (num_sandwiches : ℕ) (salami_cost brie_cost olive_price_per_pound feta_price_per_pound bread_cost total_spent : ℚ) 
  (olive_weight feta_weight : ℚ) : 
  num_sandwiches = 2 ∧ 
  salami_cost = 4 ∧ 
  brie_cost = 3 * salami_cost ∧ 
  olive_price_per_pound = 10 ∧ 
  olive_weight = 1/4 ∧ 
  feta_price_per_pound = 8 ∧ 
  feta_weight = 1/2 ∧ 
  bread_cost = 2 ∧ 
  total_spent = 40 → 
  (total_spent - (salami_cost + brie_cost + olive_price_per_pound * olive_weight + 
    feta_price_per_pound * feta_weight + bread_cost)) / num_sandwiches = 7.75 := by
  sorry

end fancy_sandwich_cost_l3772_377282


namespace min_side_difference_l3772_377295

theorem min_side_difference (PQ PR QR : ℕ) : 
  PQ + PR + QR = 3010 →
  PQ < PR →
  PR ≤ QR →
  PQ + PR > QR →
  PQ + QR > PR →
  PR + QR > PQ →
  ∀ PQ' PR' QR' : ℕ, 
    PQ' + PR' + QR' = 3010 →
    PQ' < PR' →
    PR' ≤ QR' →
    PQ' + PR' > QR' →
    PQ' + QR' > PR' →
    PR' + QR' > PQ' →
    QR - PQ ≤ QR' - PQ' :=
by sorry

end min_side_difference_l3772_377295


namespace min_value_a2b_l3772_377203

-- Define the function f
def f (x : ℝ) : ℝ := |x^2 - 6|

-- State the theorem
theorem min_value_a2b (a b : ℝ) (h1 : a < b) (h2 : b < 0) (h3 : f a = f b) :
  ∃ (m : ℝ), m = -4 ∧ ∀ (x y : ℝ), x < y ∧ y < 0 ∧ f x = f y → m ≤ x^2 * y :=
sorry

end min_value_a2b_l3772_377203


namespace ellipse_distance_sum_constant_l3772_377267

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope m passing through point P -/
structure Line where
  m : ℝ
  P : Point

theorem ellipse_distance_sum_constant
  (C : Ellipse)
  (h_ecc : C.a^2 - C.b^2 = (C.a / 2)^2) -- eccentricity is 1/2
  (h_chord : 2 * C.b^2 / C.a = 3) -- chord length condition
  (P : Point)
  (h_P_on_axis : P.y = 0 ∧ P.x^2 ≤ C.a^2) -- P is on the major axis
  (l : Line)
  (h_l_slope : l.m = C.b / C.a) -- line l has slope b/a
  (h_l_through_P : l.P = P) -- line l passes through P
  (A B : Point)
  (h_A_on_C : A.x^2 / C.a^2 + A.y^2 / C.b^2 = 1) -- A is on ellipse C
  (h_B_on_C : B.x^2 / C.a^2 + B.y^2 / C.b^2 = 1) -- B is on ellipse C
  (h_A_on_l : A.y = l.m * (A.x - P.x)) -- A is on line l
  (h_B_on_l : B.y = l.m * (B.x - P.x)) -- B is on line l
  : (A.x - P.x)^2 + (A.y - P.y)^2 + (B.x - P.x)^2 + (B.y - P.y)^2 = C.a^2 + C.b^2 := by
  sorry

end ellipse_distance_sum_constant_l3772_377267


namespace A_symmetry_l3772_377227

/-- A(n, k, r) is the number of integer tuples (x₁, x₂, ..., xₖ) satisfying:
    - x₁ ≥ x₂ ≥ ... ≥ xₖ ≥ 0
    - x₁ + x₂ + ... + xₖ = n
    - x₁ - xₖ ≤ r -/
def A (n k r : ℕ+) : ℕ :=
  sorry

/-- For all positive integers m, s, t, A(m, s, t) = A(m, t, s) -/
theorem A_symmetry (m s t : ℕ+) : A m s t = A m t s := by
  sorry

end A_symmetry_l3772_377227


namespace pizza_slices_l3772_377200

theorem pizza_slices (total_pizzas : ℕ) (total_slices : ℕ) (h1 : total_pizzas = 21) (h2 : total_slices = 168) :
  total_slices / total_pizzas = 8 := by
sorry

end pizza_slices_l3772_377200


namespace room_width_calculation_l3772_377206

theorem room_width_calculation (length width total_area : ℝ) 
  (h1 : length = 4)
  (h2 : total_area = 80)
  (h3 : total_area = length * width) :
  width = 20 := by
sorry

end room_width_calculation_l3772_377206


namespace right_triangle_area_l3772_377260

/-- The area of a right-angled triangle with perpendicular sides of lengths √12 cm and √6 cm is 3√2 square centimeters. -/
theorem right_triangle_area : 
  let side1 : ℝ := Real.sqrt 12
  let side2 : ℝ := Real.sqrt 6
  (1 / 2 : ℝ) * side1 * side2 = 3 * Real.sqrt 2 := by sorry

end right_triangle_area_l3772_377260


namespace parabola_passes_through_points_parabola_general_form_l3772_377286

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Theorem statement
theorem parabola_passes_through_points :
  (parabola (-1) = 0) ∧ (parabola 3 = 0) :=
by
  sorry

-- Verify the general form
theorem parabola_general_form (x : ℝ) :
  ∃ (b c : ℝ), parabola x = x^2 - b*x + c :=
by
  sorry

end parabola_passes_through_points_parabola_general_form_l3772_377286


namespace cone_properties_l3772_377230

/-- Properties of a right circular cone -/
theorem cone_properties (V h r l : ℝ) (hV : V = 16 * Real.pi) (hh : h = 6) 
  (hVol : (1/3) * Real.pi * r^2 * h = V) 
  (hSlant : l^2 = r^2 + h^2) : 
  2 * Real.pi * r = 4 * Real.sqrt 2 * Real.pi ∧ 
  Real.pi * r * l = 4 * Real.sqrt 22 * Real.pi := by
  sorry


end cone_properties_l3772_377230


namespace problem_1_problem_2_l3772_377261

-- Problem 1
theorem problem_1 (x : ℝ) : x^2 * x^3 - x^5 = 0 := by sorry

-- Problem 2
theorem problem_2 (a : ℝ) : (a + 1)^2 + 2*a*(a - 1) = 3*a^2 + 1 := by sorry

end problem_1_problem_2_l3772_377261


namespace adams_earnings_l3772_377274

/-- Adam's lawn mowing earnings problem -/
theorem adams_earnings (earnings_per_lawn : ℕ) (total_lawns : ℕ) (forgotten_lawns : ℕ) :
  earnings_per_lawn = 9 →
  total_lawns = 12 →
  forgotten_lawns = 8 →
  (total_lawns - forgotten_lawns) * earnings_per_lawn = 36 :=
by sorry

end adams_earnings_l3772_377274


namespace investment_percentage_problem_l3772_377246

theorem investment_percentage_problem (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ) 
  (second_rate : ℝ) (third_rate : ℝ) (desired_income : ℝ) (x : ℝ) :
  total_investment = 10000 ∧ 
  first_investment = 4000 ∧ 
  second_investment = 3500 ∧ 
  second_rate = 0.04 ∧ 
  third_rate = 0.064 ∧ 
  desired_income = 500 ∧
  first_investment * (x / 100) + second_investment * second_rate + 
    (total_investment - first_investment - second_investment) * third_rate = desired_income →
  x = 5 := by
sorry

end investment_percentage_problem_l3772_377246


namespace num_closed_lockers_l3772_377289

/-- The number of lockers and students -/
def n : ℕ := 100

/-- A locker is open if and only if its number is a perfect square -/
def is_open (k : ℕ) : Prop := ∃ m : ℕ, k = m^2

/-- The number of perfect squares less than or equal to n -/
def num_perfect_squares (n : ℕ) : ℕ := (n.sqrt : ℕ)

/-- The main theorem: The number of closed lockers is equal to
    the total number of lockers minus the number of perfect squares -/
theorem num_closed_lockers : 
  n - (num_perfect_squares n) = 90 := by sorry

end num_closed_lockers_l3772_377289


namespace rationalize_denominator_l3772_377228

theorem rationalize_denominator :
  ∃ (A B C D E F : ℚ),
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) =
    (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    F > 0 ∧
    A = -1 ∧
    B = -3 ∧
    C = 1 ∧
    D = 2/3 ∧
    E = 33 ∧
    F = 17 :=
by sorry

end rationalize_denominator_l3772_377228


namespace characterization_of_M_inequality_for_product_one_l3772_377240

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 2|

-- Define the set M
def M : Set ℝ := {x | f x ≤ 2}

-- Theorem 1: Characterization of set M
theorem characterization_of_M : M = {x | -5 ≤ x ∧ x ≤ 1} := by sorry

-- Theorem 2: Inequality for positive numbers with product 1
theorem inequality_for_product_one (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_prod : a * b * c = 1) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 1/a + 1/b + 1/c := by sorry

end characterization_of_M_inequality_for_product_one_l3772_377240


namespace journey_average_speed_l3772_377259

/-- Calculates the average speed given distances in meters and times in minutes -/
def average_speed (distances : List Float) (times : List Float) : Float :=
  let total_distance := (distances.sum / 1000)  -- Convert to km
  let total_time := (times.sum / 60)  -- Convert to hours
  total_distance / total_time

/-- Theorem: The average speed for the given journey is 6 km/h -/
theorem journey_average_speed :
  let distances := [1000, 1500, 2000]
  let times := [10, 15, 20]
  average_speed distances times = 6 := by
sorry

#eval average_speed [1000, 1500, 2000] [10, 15, 20]

end journey_average_speed_l3772_377259


namespace fraction_zero_value_l3772_377210

theorem fraction_zero_value (x : ℝ) : 
  (x^2 - 4) / (x - 2) = 0 ∧ x - 2 ≠ 0 → x = -2 := by
sorry

end fraction_zero_value_l3772_377210


namespace five_people_arrangement_l3772_377224

/-- The number of ways to arrange n people in a row. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange two specific people next to each other in a row of n people. -/
def adjacentPairArrangements (n : ℕ) : ℕ := 2 * (n - 1)

/-- The number of ways to arrange 5 people in a row with two specific people next to each other. -/
theorem five_people_arrangement : 
  adjacentPairArrangements 5 * arrangements 3 = 48 := by
  sorry

end five_people_arrangement_l3772_377224


namespace product_terminal_zeros_l3772_377264

/-- The number of terminal zeros in a natural number -/
def terminalZeros (n : ℕ) : ℕ := sorry

/-- The product of 75 and 480 -/
def product : ℕ := 75 * 480

/-- Theorem: The number of terminal zeros in the product of 75 and 480 is 3 -/
theorem product_terminal_zeros : terminalZeros product = 3 := by sorry

end product_terminal_zeros_l3772_377264


namespace variance_scaling_l3772_377277

/-- Given a list of 8 real numbers, compute its variance -/
def variance (xs : List ℝ) : ℝ := sorry

theorem variance_scaling (xs : List ℝ) (h : variance xs = 3) :
  variance (xs.map (· * 2)) = 12 := by sorry

end variance_scaling_l3772_377277


namespace greatest_two_digit_multiple_of_17_l3772_377276

theorem greatest_two_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 99 → n ≥ 10 → n % 17 = 0 → n ≤ 85 :=
by
  sorry

end greatest_two_digit_multiple_of_17_l3772_377276


namespace pure_imaginary_ratio_l3772_377251

theorem pure_imaginary_ratio (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ k : ℝ, (3 - 5*Complex.I) * (a + b*Complex.I) * (1 + 2*Complex.I) = k * Complex.I) →
  a / b = -1 / 7 :=
by sorry

end pure_imaginary_ratio_l3772_377251


namespace only_solution_is_two_l3772_377244

/-- Represents the number constructed in the problem -/
def constructNumber (k : ℕ) : ℕ :=
  (10^2000 - 1) - (10^k - 1) * 10^(2000 - k) - (10^1001 - 1)

/-- The main theorem stating that k = 2 is the only solution -/
theorem only_solution_is_two :
  ∃! k : ℕ, k > 0 ∧ ∃ m : ℕ, constructNumber k = m^2 :=
sorry

end only_solution_is_two_l3772_377244


namespace square_ratio_problem_l3772_377257

theorem square_ratio_problem :
  let area_ratio : ℚ := 18 / 50
  let side_ratio : ℝ := Real.sqrt (area_ratio)
  ∃ (a b c : ℕ), 
    (a : ℝ) * Real.sqrt b / c = side_ratio ∧
    a = 3 ∧ b = 2 ∧ c = 5 ∧
    a + b + c = 10 :=
by sorry

end square_ratio_problem_l3772_377257


namespace count_valid_pairs_l3772_377238

def has_one_solution (b c : ℕ) : Prop :=
  b^2 = 4*c ∨ c^2 = 4*b

def valid_pair (b c : ℕ) : Prop :=
  1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ has_one_solution b c

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ S ↔ valid_pair p.1 p.2) ∧ Finset.card S = 3 :=
sorry

end count_valid_pairs_l3772_377238


namespace journey_possible_l3772_377262

/-- Represents a location along the route -/
structure Location :=
  (distance : ℝ)
  (from_quixajuba : Bool)

/-- Represents a person's state during the journey -/
structure PersonState :=
  (location : Location)
  (has_bicycle : Bool)

/-- Represents the state of the entire system at a given time -/
structure SystemState :=
  (time : ℝ)
  (person_a : PersonState)
  (person_b : PersonState)
  (person_c : PersonState)

/-- Defines the problem parameters -/
def problem_params : (ℝ × ℝ × ℝ) :=
  (24, 6, 18)  -- total_distance, walking_speed, biking_speed

/-- Defines a valid initial state -/
def initial_state : SystemState :=
  { time := 0,
    person_a := { location := { distance := 0, from_quixajuba := true }, has_bicycle := true },
    person_b := { location := { distance := 0, from_quixajuba := true }, has_bicycle := false },
    person_c := { location := { distance := 24, from_quixajuba := false }, has_bicycle := false } }

/-- Defines what it means for a system state to be valid -/
def is_valid_state (params : ℝ × ℝ × ℝ) (state : SystemState) : Prop :=
  let (total_distance, _, _) := params
  0 ≤ state.time ∧
  0 ≤ state.person_a.location.distance ∧ state.person_a.location.distance ≤ total_distance ∧
  0 ≤ state.person_b.location.distance ∧ state.person_b.location.distance ≤ total_distance ∧
  0 ≤ state.person_c.location.distance ∧ state.person_c.location.distance ≤ total_distance ∧
  (state.person_a.has_bicycle ∨ state.person_b.has_bicycle ∨ state.person_c.has_bicycle)

/-- Defines what it means for a system state to be a goal state -/
def is_goal_state (params : ℝ × ℝ × ℝ) (state : SystemState) : Prop :=
  let (total_distance, _, _) := params
  state.person_a.location.distance = total_distance ∧
  state.person_b.location.distance = total_distance ∧
  state.person_c.location.distance = 0 ∧
  state.time ≤ 160/60  -- 2 hours and 40 minutes in decimal hours

/-- The main theorem to be proved -/
theorem journey_possible (params : ℝ × ℝ × ℝ) (init : SystemState) :
  is_valid_state params init →
  ∃ (final : SystemState), is_valid_state params final ∧ is_goal_state params final :=
sorry

end journey_possible_l3772_377262


namespace polynomial_existence_l3772_377298

theorem polynomial_existence (n : ℕ+) :
  ∃ (f g : Polynomial ℤ), (f * (X + 1) ^ (2 ^ n.val) + g * (X ^ (2 ^ n.val) + 1)) = 2 := by
  sorry

end polynomial_existence_l3772_377298


namespace sarah_car_robots_l3772_377284

/-- Prove that Sarah has 125 car robots given the conditions of the problem -/
theorem sarah_car_robots :
  ∀ (tom michael bob sarah : ℕ),
  tom = 15 →
  michael = 2 * tom →
  bob = 8 * michael →
  sarah = (bob / 2) + 5 →
  sarah = 125 := by
sorry

end sarah_car_robots_l3772_377284


namespace sequence_a_10_l3772_377204

/-- A sequence satisfying the given properties -/
def Sequence (a : ℕ+ → ℤ) : Prop :=
  (∀ p q : ℕ+, a (p + q) = a p + a q) ∧ (a 2 = -6)

/-- The theorem to be proved -/
theorem sequence_a_10 (a : ℕ+ → ℤ) (h : Sequence a) : a 10 = -30 := by
  sorry

end sequence_a_10_l3772_377204


namespace smallest_cut_length_l3772_377299

theorem smallest_cut_length (x : ℕ) : x > 0 ∧ x ≤ 12 ∧ (12 - x) + (20 - x) ≤ (24 - x) →
  x ≥ 8 ∧ ∀ y : ℕ, y > 0 ∧ y < x → (12 - y) + (20 - y) > (24 - y) :=
by
  sorry

end smallest_cut_length_l3772_377299


namespace v_3_equals_262_l3772_377294

/-- The polynomial f(x) -/
def f (x : ℝ) : ℝ := 7*x^7 + 6*x^6 + 5*x^5 + 4*x^4 + 3*x^3 + 2*x^2 + x

/-- The value of x -/
def x : ℝ := 3

/-- The value of v_3 using Horner's method for the first three terms -/
def v_3 : ℝ := ((7*x + 6)*x + 5)*x + 4

/-- Theorem stating that v_3 equals 262 -/
theorem v_3_equals_262 : v_3 = 262 := by
  sorry

end v_3_equals_262_l3772_377294


namespace average_salary_proof_l3772_377207

theorem average_salary_proof (salary_a salary_b salary_c salary_d salary_e : ℕ)
  (h1 : salary_a = 10000)
  (h2 : salary_b = 5000)
  (h3 : salary_c = 11000)
  (h4 : salary_d = 7000)
  (h5 : salary_e = 9000) :
  (salary_a + salary_b + salary_c + salary_d + salary_e) / 5 = 8600 := by
  sorry

end average_salary_proof_l3772_377207


namespace hundred_decomposition_l3772_377211

def isPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def isPerfectCube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

def isValidDecomposition (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  isPerfectSquare a ∧ isPerfectSquare b ∧ isPerfectCube c

theorem hundred_decomposition :
  ∃! (a b c : ℕ), a + b + c = 100 ∧ isValidDecomposition a b c :=
sorry

end hundred_decomposition_l3772_377211


namespace second_bus_ride_duration_l3772_377217

def first_bus_wait : ℕ := 12
def first_bus_ride : ℕ := 30

def total_first_bus_time : ℕ := first_bus_wait + first_bus_ride

def second_bus_time : ℕ := total_first_bus_time / 2

theorem second_bus_ride_duration : second_bus_time = 21 := by
  sorry

end second_bus_ride_duration_l3772_377217


namespace new_energy_vehicle_analysis_l3772_377269

def daily_distances : List Int := [-8, -12, -16, 0, 22, 31, 33]
def standard_distance : Int := 50
def gasoline_consumption : Rat := 5.5
def gasoline_price : Rat := 8.4
def electric_consumption : Rat := 15
def electricity_price : Rat := 0.5

theorem new_energy_vehicle_analysis :
  let max_distance := daily_distances.foldl max (daily_distances.head!)
  let min_distance := daily_distances.foldl min (daily_distances.head!)
  let total_distance := daily_distances.sum
  let gasoline_cost := (total_distance : Rat) / 100 * gasoline_consumption * gasoline_price
  let electric_cost := (total_distance : Rat) / 100 * electric_consumption * electricity_price
  (max_distance - min_distance = 49) ∧
  (total_distance = 50) ∧
  (gasoline_cost - electric_cost = 154.8) := by
  sorry


end new_energy_vehicle_analysis_l3772_377269


namespace math_score_calculation_l3772_377296

theorem math_score_calculation (total_subjects : ℕ) (avg_without_math : ℝ) (avg_with_math : ℝ) :
  total_subjects = 5 →
  avg_without_math = 88 →
  avg_with_math = 92 →
  (total_subjects - 1) * avg_without_math + (avg_with_math * total_subjects - (total_subjects - 1) * avg_without_math) = 108 :=
by sorry

end math_score_calculation_l3772_377296


namespace power_of_power_l3772_377247

theorem power_of_power (a : ℝ) : (a^3)^3 = a^9 := by
  sorry

end power_of_power_l3772_377247


namespace circus_receipts_l3772_377265

theorem circus_receipts (total_tickets : ℕ) (adult_ticket_cost : ℕ) (child_ticket_cost : ℕ) (adult_tickets_sold : ℕ) :
  total_tickets = 522 →
  adult_ticket_cost = 15 →
  child_ticket_cost = 8 →
  adult_tickets_sold = 130 →
  (adult_tickets_sold * adult_ticket_cost + (total_tickets - adult_tickets_sold) * child_ticket_cost) = 5086 :=
by sorry

end circus_receipts_l3772_377265


namespace f_composition_value_l3772_377221

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1
  else if x = 0 then Real.pi
  else 0

theorem f_composition_value : f (f (f (-1))) = Real.pi + 1 := by sorry

end f_composition_value_l3772_377221


namespace dogwood_trees_planted_tomorrow_l3772_377288

/-- The number of dogwood trees planted tomorrow to reach the desired total -/
def trees_planted_tomorrow (initial_trees : ℕ) (planted_today : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (initial_trees + planted_today)

/-- Theorem stating the number of trees planted tomorrow -/
theorem dogwood_trees_planted_tomorrow :
  trees_planted_tomorrow 39 41 100 = 20 := by
  sorry

end dogwood_trees_planted_tomorrow_l3772_377288


namespace square_of_sum_17_5_l3772_377212

theorem square_of_sum_17_5 : 17^2 + 2*(17*5) + 5^2 = 484 := by
  sorry

end square_of_sum_17_5_l3772_377212


namespace cos_five_pi_thirds_l3772_377249

theorem cos_five_pi_thirds : Real.cos (5 * π / 3) = 1 / 2 := by sorry

end cos_five_pi_thirds_l3772_377249


namespace intersection_of_A_and_B_l3772_377272

def A : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 + 1}
def B : Set (ℝ × ℝ) := {p | p.2 = p.1 + 3}

theorem intersection_of_A_and_B : ∃! a : ℝ × ℝ, a ∈ A ∧ a ∈ B ∧ a = (2, 5) := by sorry

end intersection_of_A_and_B_l3772_377272


namespace bulb_selection_problem_l3772_377287

theorem bulb_selection_problem (total_bulbs : ℕ) (defective_bulbs : ℕ) (probability : ℚ) :
  total_bulbs = 10 →
  defective_bulbs = 4 →
  probability = 1 / 15 →
  ∃ n : ℕ, (((total_bulbs - defective_bulbs : ℚ) / total_bulbs) ^ n = probability) ∧ n = 5 :=
by sorry

end bulb_selection_problem_l3772_377287


namespace f_lower_bound_f_one_less_than_two_l3772_377292

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1 + a| + |x - a|

-- Part 1
theorem f_lower_bound (x a : ℝ) (h : a ≥ 2) : f x a ≥ 3 := by
  sorry

-- Part 2
theorem f_one_less_than_two (a : ℝ) : 
  (f 1 a < 2) ↔ a ∈ Set.Ioo (-1/2 : ℝ) (3/2 : ℝ) := by
  sorry

end f_lower_bound_f_one_less_than_two_l3772_377292


namespace mauve_red_parts_l3772_377218

/-- Represents the composition of paint mixtures -/
structure PaintMixture where
  red : ℝ
  blue : ℝ

/-- Defines the fuchsia paint mixture -/
def fuchsia : PaintMixture := { red := 5, blue := 3 }

/-- Defines the mauve paint mixture with unknown red parts -/
def mauve (x : ℝ) : PaintMixture := { red := x, blue := 6 }

/-- Theorem stating the number of red parts in mauve paint -/
theorem mauve_red_parts : 
  ∃ (x : ℝ), 
    (16 * (fuchsia.red / (fuchsia.red + fuchsia.blue))) = 
    (x * 20 / (x + (mauve x).blue)) ∧ 
    x = 3 := by sorry

end mauve_red_parts_l3772_377218


namespace employee_pay_calculation_l3772_377214

/-- Given two employees with a total pay of 550 rupees, where one employee is paid 120% of the other,
    prove that the employee with lower pay receives 250 rupees. -/
theorem employee_pay_calculation (total_pay : ℝ) (x y : ℝ) : 
  total_pay = 550 →
  x = 1.2 * y →
  x + y = total_pay →
  y = 250 := by
sorry

end employee_pay_calculation_l3772_377214


namespace minimum_value_complex_l3772_377250

theorem minimum_value_complex (z : ℂ) (h : Complex.abs (z - 3 + Complex.I) = 3) :
  (Complex.abs (z + 2 - 3 * Complex.I))^2 + (Complex.abs (z - 6 + 2 * Complex.I))^2 = 44 := by
  sorry

end minimum_value_complex_l3772_377250


namespace milly_extra_balloons_l3772_377235

theorem milly_extra_balloons (total_packs : ℕ) (balloons_per_pack : ℕ) (floretta_balloons : ℕ) : 
  total_packs = 5 →
  balloons_per_pack = 6 →
  floretta_balloons = 8 →
  (total_packs * balloons_per_pack) / 2 - floretta_balloons = 7 := by
  sorry

end milly_extra_balloons_l3772_377235


namespace f_sum_reciprocal_l3772_377256

theorem f_sum_reciprocal (x : ℝ) (hx : x > 0) : 
  let f := fun (y : ℝ) => y / (y + 1)
  f x + f (1/x) = 1 := by
sorry

end f_sum_reciprocal_l3772_377256


namespace ages_sum_l3772_377216

theorem ages_sum (a b s : ℕ+) : 
  (3 * a + 5 + b = s) →
  (6 * s^2 = 2 * a^2 + 10 * b^2) →
  (Nat.gcd (Nat.gcd a.val b.val) s.val = 1) →
  (a + b + s = 19) := by
  sorry

end ages_sum_l3772_377216


namespace cubic_integer_root_l3772_377273

theorem cubic_integer_root 
  (b c : ℚ) 
  (h1 : ∃ x : ℤ, x^3 + b*x + c = 0) 
  (h2 : (5 - Real.sqrt 11)^3 + b*(5 - Real.sqrt 11) + c = 0) : 
  ∃ x : ℤ, x^3 + b*x + c = 0 ∧ x = -10 := by
sorry

end cubic_integer_root_l3772_377273


namespace circle_center_correct_l3772_377233

/-- The equation of a circle in the form ax² + bx + cy² + dy + e = 0 -/
structure CircleEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ

/-- The center of a circle -/
structure CircleCenter where
  x : ℝ
  y : ℝ

/-- Given a circle equation, return its center -/
def findCenter (eq : CircleEquation) : CircleCenter :=
  sorry

theorem circle_center_correct :
  let eq := CircleEquation.mk 1 4 1 (-6) (-20)
  findCenter eq = CircleCenter.mk (-2) 3 := by
  sorry

end circle_center_correct_l3772_377233


namespace largest_gold_coins_distribution_l3772_377205

theorem largest_gold_coins_distribution (n : ℕ) : 
  (∃ k : ℕ, n = 15 * k + 3) → 
  n < 150 → 
  (∀ m : ℕ, (∃ j : ℕ, m = 15 * j + 3) → m < 150 → m ≤ n) →
  n = 138 := by
sorry

end largest_gold_coins_distribution_l3772_377205


namespace partial_fraction_decomposition_constant_l3772_377253

theorem partial_fraction_decomposition_constant (A B C : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ -5 →
    1 / (x^3 + 2*x^2 - 25*x - 50) = A / (x - 2) + B / (x + 5) + C / ((x + 5)^2)) →
  B = -11/490 := by
sorry

end partial_fraction_decomposition_constant_l3772_377253


namespace four_primes_sum_l3772_377293

theorem four_primes_sum (p₁ p₂ p₃ p₄ : ℕ) : 
  Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ 
  p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₃ ≠ p₄ ∧
  (p₁ * p₂ * p₃ * p₄ ∣ 16^4 + 16^2 + 1) ∧
  p₁ + p₂ + p₃ + p₄ = 264 := by
  sorry

end four_primes_sum_l3772_377293


namespace decimal_equals_fraction_l3772_377280

/-- The decimal representation of the number we're considering -/
def decimal : ℚ := 0.73264264264

/-- The denominator of the fraction we're looking for -/
def denominator : ℕ := 999900

/-- The numerator of the fraction we're looking for -/
def numerator : ℕ := 732635316

/-- Theorem stating that our decimal is equal to the fraction numerator/denominator -/
theorem decimal_equals_fraction : decimal = (numerator : ℚ) / denominator := by
  sorry

end decimal_equals_fraction_l3772_377280


namespace complex_number_problem_l3772_377232

theorem complex_number_problem (a : ℝ) (z : ℂ) (i : ℂ) : 
  a < 0 → 
  i^2 = -1 → 
  z = a * i / (1 - 2 * i) → 
  Complex.abs z = Real.sqrt 5 → 
  a = -5 := by sorry

end complex_number_problem_l3772_377232


namespace hyperbola_equation_special_case_l3772_377270

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  a_pos : 0 < a
  b_pos : 0 < b

/-- The equation of a hyperbola -/
def hyperbola_equation (h : Hyperbola a b) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

/-- The distance from the foci to the asymptotes of a hyperbola -/
def foci_to_asymptote_distance (h : Hyperbola a b) : ℝ :=
  b

/-- The length of the real axis of a hyperbola -/
def real_axis_length (h : Hyperbola a b) : ℝ :=
  2 * a

/-- Theorem: If the distance from the foci to the asymptotes equals the length of the real axis
    and the point (2,2) lies on the hyperbola, then the equation of the hyperbola is x^2/3 - y^2/12 = 1 -/
theorem hyperbola_equation_special_case (h : Hyperbola a b) :
  foci_to_asymptote_distance h = real_axis_length h →
  hyperbola_equation h 2 2 →
  ∀ x y, hyperbola_equation h x y ↔ x^2 / 3 - y^2 / 12 = 1 :=
sorry

end hyperbola_equation_special_case_l3772_377270


namespace chloe_winter_clothing_l3772_377225

/-- Calculates the total number of winter clothing items given the number of boxes and items per box. -/
def total_winter_clothing (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) : ℕ :=
  num_boxes * (scarves_per_box + mittens_per_box)

/-- Proves that Chloe has 32 pieces of winter clothing given the problem conditions. -/
theorem chloe_winter_clothing : 
  total_winter_clothing 4 2 6 = 32 := by
  sorry

#eval total_winter_clothing 4 2 6

end chloe_winter_clothing_l3772_377225


namespace exists_k_sum_of_digits_equal_l3772_377241

/-- Sum of digits function -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number contains the digit 9 -/
def hasNoNine (n : ℕ) : Prop := sorry

/-- Main theorem -/
theorem exists_k_sum_of_digits_equal : 
  ∃ k : ℕ, k > 0 ∧ hasNoNine k ∧ sumOfDigits k = sumOfDigits (2^(24^2017) * k) := by sorry

end exists_k_sum_of_digits_equal_l3772_377241


namespace triangle_perimeter_sum_l3772_377285

theorem triangle_perimeter_sum : 
  ∀ (a b c d e : ℝ),
  a = 6 ∧ b = 8 ∧ 
  c^2 + d^2 = e^2 ∧
  (1/2) * c * d = (1/2) * (1/2) * a * b →
  a + b + (a^2 + b^2).sqrt + c + d + e = 24 + 6 * Real.sqrt 3 + 2 * Real.sqrt 15 :=
by sorry

end triangle_perimeter_sum_l3772_377285


namespace adult_meal_cost_l3772_377254

theorem adult_meal_cost 
  (total_people : ℕ) 
  (kids : ℕ) 
  (total_cost : ℚ) 
  (h1 : total_people = 11) 
  (h2 : kids = 2) 
  (h3 : total_cost = 72) : 
  (total_cost / (total_people - kids) : ℚ) = 8 := by
  sorry

end adult_meal_cost_l3772_377254


namespace power_fraction_plus_two_l3772_377297

theorem power_fraction_plus_two : (5 / 3 : ℚ)^7 + 2 = 82499 / 2187 := by sorry

end power_fraction_plus_two_l3772_377297


namespace solution_for_n_equals_S_plus_U_squared_l3772_377248

def S (n : ℕ) : ℕ := sorry  -- Sum of digits of n

def U (n : ℕ) : ℕ := sorry  -- Unit digit of n

theorem solution_for_n_equals_S_plus_U_squared :
  ∀ n : ℕ, n > 0 → (n = S n + (U n)^2) ↔ (n = 13 ∨ n = 46 ∨ n = 99) := by
  sorry

end solution_for_n_equals_S_plus_U_squared_l3772_377248


namespace bea_earned_more_than_dawn_l3772_377245

/-- Calculates the difference in earnings between two lemonade sellers -/
def lemonade_earnings_difference (bea_price bea_sold dawn_price dawn_sold : ℕ) : ℕ :=
  bea_price * bea_sold - dawn_price * dawn_sold

/-- Proves that Bea earned 26 cents more than Dawn given the conditions -/
theorem bea_earned_more_than_dawn :
  lemonade_earnings_difference 25 10 28 8 = 26 := by
  sorry

end bea_earned_more_than_dawn_l3772_377245


namespace milk_dilution_l3772_377266

/-- Proves that adding 15 liters of pure milk to 10 liters of milk with 5% water content
    results in a final water content of 2% -/
theorem milk_dilution (initial_milk : ℝ) (pure_milk : ℝ) (initial_water_percent : ℝ) :
  initial_milk = 10 →
  pure_milk = 15 →
  initial_water_percent = 5 →
  let total_milk := initial_milk + pure_milk
  let water_volume := initial_milk * (initial_water_percent / 100)
  let final_water_percent := (water_volume / total_milk) * 100
  final_water_percent = 2 := by
sorry

end milk_dilution_l3772_377266


namespace cos_75_cos_15_minus_sin_255_sin_165_l3772_377242

theorem cos_75_cos_15_minus_sin_255_sin_165 :
  Real.cos (75 * π / 180) * Real.cos (15 * π / 180) -
  Real.sin (255 * π / 180) * Real.sin (165 * π / 180) = 1/2 := by
  sorry

end cos_75_cos_15_minus_sin_255_sin_165_l3772_377242


namespace investment_difference_l3772_377268

theorem investment_difference (x y : ℝ) : 
  x = 1000 →
  x + y = 1000 →
  0.02 * x + 0.04 * (x + y) = 92 →
  y = 800 := by
sorry

end investment_difference_l3772_377268


namespace soccer_school_admission_probability_l3772_377263

/-- Represents the probability of being admitted to the soccer school -/
def admission_probability (p_assistant : ℝ) (p_head : ℝ) : ℝ :=
  p_assistant * p_assistant + 2 * p_assistant * (1 - p_assistant) * p_head

/-- The probability of the young soccer enthusiast being admitted to the well-known soccer school is 0.4 -/
theorem soccer_school_admission_probability : 
  admission_probability 0.5 0.3 = 0.4 := by
  sorry

end soccer_school_admission_probability_l3772_377263


namespace set_union_problem_l3772_377222

theorem set_union_problem (S T : Set ℕ) (h1 : S = {0, 1}) (h2 : T = {0}) :
  S ∪ T = {0, 1} := by sorry

end set_union_problem_l3772_377222


namespace divisibility_by_five_l3772_377290

theorem divisibility_by_five (x y : ℤ) :
  (∃ k : ℤ, x^2 - 2*x*y + 2*y^2 = 5*k ∨ x^2 + 2*x*y + 2*y^2 = 5*k) ↔ 
  (∃ a b : ℤ, x = 5*a ∧ y = 5*b) ∨ 
  (∀ k : ℤ, x ≠ 5*k ∧ y ≠ 5*k) :=
by sorry

end divisibility_by_five_l3772_377290


namespace average_difference_l3772_377201

theorem average_difference (a b c : ℝ) 
  (h1 : (a + b) / 2 = 115)
  (h2 : (b + c) / 2 = 160) :
  a - c = -90 := by
sorry

end average_difference_l3772_377201


namespace room_area_difference_l3772_377226

-- Define the dimensions of the rooms
def largest_room_width : ℝ := 45
def largest_room_length : ℝ := 30
def smallest_room_width : ℝ := 15
def smallest_room_length : ℝ := 8

-- Define the area calculation function
def area (width : ℝ) (length : ℝ) : ℝ := width * length

-- Theorem statement
theorem room_area_difference :
  area largest_room_width largest_room_length - area smallest_room_width smallest_room_length = 1230 := by
  sorry

end room_area_difference_l3772_377226


namespace train_speed_theorem_l3772_377202

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_theorem (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 120)
  (h2 : bridge_length = 255)
  (h3 : crossing_time = 30)
  : (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_theorem

end train_speed_theorem_l3772_377202


namespace mean_problem_l3772_377258

theorem mean_problem (x : ℝ) : 
  (48 + 62 + 98 + 124 + x) / 5 = 78 → 
  (28 + x + 42 + 78 + 104) / 5 = 62 := by
sorry

end mean_problem_l3772_377258


namespace smallest_number_l3772_377278

-- Define the numbers in their respective bases
def num_base3 : ℕ := 1 * 3^3 + 0 * 3^2 + 0 * 3^1 + 2 * 3^0
def num_base6 : ℕ := 2 * 6^2 + 1 * 6^1 + 0 * 6^0
def num_base4 : ℕ := 1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0
def num_base2 : ℕ := 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

-- Theorem statement
theorem smallest_number :
  num_base2 < num_base3 ∧ num_base2 < num_base6 ∧ num_base2 < num_base4 :=
by sorry

end smallest_number_l3772_377278
