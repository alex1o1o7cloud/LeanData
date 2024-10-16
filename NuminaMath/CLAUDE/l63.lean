import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_7_24_25_l63_6360

theorem right_triangle_7_24_25 : 
  ∀ (a b c : ℝ), a = 7 ∧ b = 24 ∧ c = 25 → a^2 + b^2 = c^2 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_7_24_25_l63_6360


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l63_6308

/-- The distance between the vertices of the hyperbola (x²/16) - (y²/25) = 1 is 8 -/
theorem hyperbola_vertex_distance : 
  let h : ℝ → ℝ → Prop := λ x y => (x^2 / 16) - (y^2 / 25) = 1
  ∃ x₁ x₂ : ℝ, (h x₁ 0 ∧ h x₂ 0) ∧ |x₁ - x₂| = 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l63_6308


namespace NUMINAMATH_CALUDE_complex_number_equality_l63_6345

theorem complex_number_equality (z : ℂ) (h : z / (1 - Complex.I) = Complex.I) : z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l63_6345


namespace NUMINAMATH_CALUDE_min_sum_of_sides_l63_6375

theorem min_sum_of_sides (a b c : ℝ) (A B C : ℝ) : 
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  ((a + b)^2 - c^2 = 4) →
  (C = Real.pi / 3) →
  (∃ (x : ℝ), (a + b ≥ x) ∧ (∀ y, a + b ≥ y → x ≤ y) ∧ (x = 4 * Real.sqrt 3 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_sides_l63_6375


namespace NUMINAMATH_CALUDE_peanut_butter_jar_size_l63_6326

theorem peanut_butter_jar_size (total_ounces : ℕ) (jar_size_1 jar_size_3 : ℕ) (total_jars : ℕ) :
  total_ounces = 252 →
  jar_size_1 = 16 →
  jar_size_3 = 40 →
  total_jars = 9 →
  ∃ (jar_size_2 : ℕ),
    jar_size_2 = 28 ∧
    total_ounces = (total_jars / 3) * (jar_size_1 + jar_size_2 + jar_size_3) :=
by sorry

end NUMINAMATH_CALUDE_peanut_butter_jar_size_l63_6326


namespace NUMINAMATH_CALUDE_smaller_number_problem_l63_6377

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 40) : 
  min x y = -5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l63_6377


namespace NUMINAMATH_CALUDE_train_speed_problem_l63_6359

theorem train_speed_problem (train_length : ℝ) (faster_speed : ℝ) (passing_time : ℝ) :
  train_length = 100 →
  faster_speed = 45 →
  passing_time = 9.599232061435085 →
  ∃ slower_speed : ℝ,
    slower_speed > 0 ∧
    slower_speed < faster_speed ∧
    (faster_speed + slower_speed) * (passing_time / 3600) = 2 * (train_length / 1000) ∧
    slower_speed = 30 :=
by sorry

end NUMINAMATH_CALUDE_train_speed_problem_l63_6359


namespace NUMINAMATH_CALUDE_least_number_divisor_l63_6340

theorem least_number_divisor (n : ℕ) (h1 : n % 5 = 3) (h2 : n % 67 = 3) (h3 : n % 8 = 3)
  (h4 : ∀ m : ℕ, m < n → (m % 5 = 3 ∧ m % 67 = 3 ∧ m % 8 = 3) → False)
  (h5 : n = 1683) :
  3 = Nat.gcd n (n - 3) :=
sorry

end NUMINAMATH_CALUDE_least_number_divisor_l63_6340


namespace NUMINAMATH_CALUDE_expression_value_l63_6330

theorem expression_value : 
  let x : ℝ := 2
  let y : ℝ := -1
  let z : ℝ := 3
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = -35 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l63_6330


namespace NUMINAMATH_CALUDE_right_triangle_equality_l63_6313

theorem right_triangle_equality (a b c p : ℝ) (h1 : 0 < a) (h2 : a ≤ b) (h3 : b ≤ c) 
  (h4 : a^2 + b^2 = c^2) (h5 : 2*p = a + b + c) : 
  let S := (1/2) * a * b
  p * (p - c) = (p - a) * (p - b) ∧ p * (p - c) = S := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_equality_l63_6313


namespace NUMINAMATH_CALUDE_not_perfect_square_property_l63_6362

def S : Set ℕ := {2, 5, 13}

theorem not_perfect_square_property (d : ℕ) (h1 : d ∉ S) (h2 : d > 0) :
  ∃ a b : ℕ, a ∈ S ∪ {d} ∧ b ∈ S ∪ {d} ∧ a ≠ b ∧ ¬∃ k : ℕ, a * b - 1 = k^2 :=
by sorry

end NUMINAMATH_CALUDE_not_perfect_square_property_l63_6362


namespace NUMINAMATH_CALUDE_intersection_point_l63_6314

/-- The line equation is y = -3x + 3 -/
def line_equation (x y : ℝ) : Prop := y = -3 * x + 3

/-- A point is on the x-axis if its y-coordinate is 0 -/
def on_x_axis (x y : ℝ) : Prop := y = 0

/-- The intersection point of the line y = -3x + 3 with the x-axis is (1, 0) -/
theorem intersection_point :
  ∃ (x y : ℝ), line_equation x y ∧ on_x_axis x y ∧ x = 1 ∧ y = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l63_6314


namespace NUMINAMATH_CALUDE_original_stations_count_l63_6318

def number_of_ticket_types (k : ℕ) : ℕ := k * (k - 1) / 2

theorem original_stations_count 
  (m n : ℕ) 
  (h1 : n > 1) 
  (h2 : number_of_ticket_types (m + n) - number_of_ticket_types m = 58) : 
  m = 14 := by
sorry

end NUMINAMATH_CALUDE_original_stations_count_l63_6318


namespace NUMINAMATH_CALUDE_intersection_P_complement_M_l63_6321

theorem intersection_P_complement_M (U : Set ℤ) (M P : Set ℤ) : 
  U = Set.univ ∧ 
  M = {1, 2} ∧ 
  P = {-2, -1, 0, 1, 2} →
  P ∩ (U \ M) = {-2, -1, 0} := by
sorry

end NUMINAMATH_CALUDE_intersection_P_complement_M_l63_6321


namespace NUMINAMATH_CALUDE_tangent_segment_difference_l63_6341

/-- A quadrilateral inscribed in a circle with an inscribed circle --/
structure CyclicTangentialQuadrilateral where
  -- Sides of the quadrilateral
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  -- Condition: quadrilateral is inscribed in a circle
  is_cyclic : True
  -- Condition: quadrilateral has an inscribed circle
  has_incircle : True

/-- Theorem about the difference of segments on a side --/
theorem tangent_segment_difference
  (q : CyclicTangentialQuadrilateral)
  (h1 : q.a = 80)
  (h2 : q.b = 100)
  (h3 : q.c = 120)
  (h4 : q.d = 140)
  (x y : ℝ)
  (h5 : x + y = q.c)
  : |x - y| = 80 := by
  sorry

end NUMINAMATH_CALUDE_tangent_segment_difference_l63_6341


namespace NUMINAMATH_CALUDE_curve_C_equation_min_area_QAB_l63_6364

-- Define the parabola E
def parabola_E (x y : ℝ) : Prop := y^2 = 8 * x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define point N on parabola E
def point_N (x y : ℝ) : Prop := parabola_E x y

-- Define point O as the origin
def point_O : ℝ × ℝ := (0, 0)

-- Define point P as the midpoint of ON
def point_P (x y : ℝ) : Prop := ∃ (nx ny : ℝ), point_N nx ny ∧ x = nx / 2 ∧ y = ny / 2

-- Define curve C as the trajectory of point P
def curve_C (x y : ℝ) : Prop := point_P x y

-- Define point Q on curve C with x₀ ≥ 5
def point_Q (x₀ y₀ : ℝ) : Prop := curve_C x₀ y₀ ∧ x₀ ≥ 5

-- Theorem for the equation of curve C
theorem curve_C_equation (x y : ℝ) : curve_C x y → y^2 = 4 * x := by sorry

-- Theorem for the minimum area of △QAB
theorem min_area_QAB (x₀ y₀ : ℝ) (hQ : point_Q x₀ y₀) : 
  ∃ (A B : ℝ × ℝ), (∀ (area : ℝ), area ≥ 25/2) := by sorry

end NUMINAMATH_CALUDE_curve_C_equation_min_area_QAB_l63_6364


namespace NUMINAMATH_CALUDE_fruit_eating_arrangements_l63_6324

def num_apples : ℕ := 4
def num_oranges : ℕ := 2
def num_bananas : ℕ := 2

def total_fruits : ℕ := num_apples + num_oranges + num_bananas

theorem fruit_eating_arrangements :
  (total_fruits.factorial) / (num_oranges.factorial * num_bananas.factorial) = 6 :=
by sorry

end NUMINAMATH_CALUDE_fruit_eating_arrangements_l63_6324


namespace NUMINAMATH_CALUDE_freshman_sophomore_percentage_l63_6316

theorem freshman_sophomore_percentage
  (total_students : ℕ)
  (pet_ownership_ratio : ℚ)
  (non_pet_owners : ℕ)
  (h1 : total_students = 400)
  (h2 : pet_ownership_ratio = 1/5)
  (h3 : non_pet_owners = 160) :
  (↑(total_students - non_pet_owners) / (1 - pet_ownership_ratio)) / total_students = 1/2 :=
sorry

end NUMINAMATH_CALUDE_freshman_sophomore_percentage_l63_6316


namespace NUMINAMATH_CALUDE_paving_stone_width_l63_6390

/-- Proves that the width of each paving stone is 2 meters given the courtyard dimensions,
    number of paving stones, and length of each paving stone. -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (num_stones : ℕ)
  (stone_length : ℝ)
  (h1 : courtyard_length = 40)
  (h2 : courtyard_width = 33/2)
  (h3 : num_stones = 132)
  (h4 : stone_length = 5/2)
  : ∃ (stone_width : ℝ), stone_width = 2 ∧ 
    courtyard_length * courtyard_width = (stone_length * stone_width) * num_stones :=
by
  sorry


end NUMINAMATH_CALUDE_paving_stone_width_l63_6390


namespace NUMINAMATH_CALUDE_store_profit_is_33_percent_l63_6370

/-- Calculates the store's profit percentage given the markups, discount, and shipping cost -/
def store_profit_percentage (first_markup : ℝ) (second_markup : ℝ) (discount : ℝ) (shipping_cost : ℝ) : ℝ :=
  let price_after_first_markup := 1 + first_markup
  let price_after_second_markup := price_after_first_markup + second_markup * price_after_first_markup
  let price_after_discount := price_after_second_markup * (1 - discount)
  let total_cost := 1 + shipping_cost
  price_after_discount - total_cost

/-- Theorem stating that the store's profit is 33% of the original cost price -/
theorem store_profit_is_33_percent :
  store_profit_percentage 0.20 0.25 0.08 0.05 = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_store_profit_is_33_percent_l63_6370


namespace NUMINAMATH_CALUDE_option1_cheaper_at_30_l63_6348

/-- Represents the cost calculation for two shopping options -/
def shopping_options (x : ℕ) : Prop :=
  let shoe_price : ℕ := 200
  let sock_price : ℕ := 40
  let num_shoes : ℕ := 20
  let option1_cost : ℕ := sock_price * x + num_shoes * shoe_price
  let option2_cost : ℕ := (sock_price * x * 9 + num_shoes * shoe_price * 9) / 10
  x > num_shoes ∧ option1_cost < option2_cost

/-- Theorem stating that Option 1 is cheaper when buying 30 pairs of socks -/
theorem option1_cheaper_at_30 : shopping_options 30 := by
  sorry

#check option1_cheaper_at_30

end NUMINAMATH_CALUDE_option1_cheaper_at_30_l63_6348


namespace NUMINAMATH_CALUDE_kickball_difference_l63_6349

theorem kickball_difference (wednesday : ℕ) (total : ℕ) : 
  wednesday = 37 →
  total = 65 →
  wednesday - (total - wednesday) = 9 := by
sorry

end NUMINAMATH_CALUDE_kickball_difference_l63_6349


namespace NUMINAMATH_CALUDE_farm_animal_ratio_l63_6309

/-- Given a farm with goats, chickens, ducks, and pigs, prove that the ratio of chickens to goats is 2:1 -/
theorem farm_animal_ratio :
  ∀ (chickens ducks pigs : ℕ),
  66 = pigs + 33 →
  ducks = (66 + chickens) / 2 →
  pigs = ducks / 3 →
  chickens = 2 * 66 :=
by sorry

end NUMINAMATH_CALUDE_farm_animal_ratio_l63_6309


namespace NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l63_6355

theorem cubic_minus_linear_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_minus_linear_factorization_l63_6355


namespace NUMINAMATH_CALUDE_sum_of_products_is_negative_one_l63_6374

-- Define the polynomial Q(x)
def Q (x : ℝ) : ℝ := x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1

-- Define the theorem
theorem sum_of_products_is_negative_one 
  (d₁ d₂ d₃ d₄ e₁ e₂ e₃ e₄ : ℝ) 
  (h : ∀ x : ℝ, Q x = (x^2 + d₁*x + e₁) * (x^2 + d₂*x + e₂) * (x^2 + d₃*x + e₃) * (x^2 + d₄*x + e₄)) : 
  d₁*e₁ + d₂*e₂ + d₃*e₃ + d₄*e₄ = -1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_products_is_negative_one_l63_6374


namespace NUMINAMATH_CALUDE_rectangle_area_l63_6354

/-- A rectangle with perimeter 100 meters and length three times the width has an area of 468.75 square meters. -/
theorem rectangle_area (l w : ℝ) (h1 : 2 * l + 2 * w = 100) (h2 : l = 3 * w) : l * w = 468.75 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l63_6354


namespace NUMINAMATH_CALUDE_max_distinct_distance_selection_l63_6323

/-- A regular polygon inscribed in a circle -/
structure RegularPolygon :=
  (sides : ℕ)
  (vertices : Fin sides → ℝ × ℝ)

/-- The distance between two vertices of a regular polygon -/
def distance (p : RegularPolygon) (i j : Fin p.sides) : ℝ := sorry

/-- A selection of vertices from a regular polygon -/
def VertexSelection (p : RegularPolygon) := Fin p.sides → Bool

/-- The number of vertices in a selection -/
def selectionSize (p : RegularPolygon) (s : VertexSelection p) : ℕ := sorry

/-- Whether all distances between selected vertices are distinct -/
def distinctDistances (p : RegularPolygon) (s : VertexSelection p) : Prop := sorry

theorem max_distinct_distance_selection (p : RegularPolygon) 
  (h : p.sides = 21) :
  (∃ (s : VertexSelection p), selectionSize p s = 5 ∧ distinctDistances p s) ∧
  (∀ (s : VertexSelection p), selectionSize p s > 5 → ¬ distinctDistances p s) :=
sorry

end NUMINAMATH_CALUDE_max_distinct_distance_selection_l63_6323


namespace NUMINAMATH_CALUDE_not_sufficient_for_parallelogram_l63_6336

/-- A quadrilateral with vertices A, B, C, and D -/
structure Quadrilateral (V : Type*) :=
  (A B C D : V)

/-- Parallelism relation between line segments -/
def Parallel {V : Type*} (AB CD : V × V) : Prop := sorry

/-- Equality of line segments -/
def SegmentEqual {V : Type*} (AB CD : V × V) : Prop := sorry

/-- Definition of a parallelogram -/
def IsParallelogram {V : Type*} (quad : Quadrilateral V) : Prop := sorry

/-- The main theorem: AB parallel to CD and AD = BC does not imply ABCD is a parallelogram -/
theorem not_sufficient_for_parallelogram {V : Type*} (quad : Quadrilateral V) :
  Parallel (quad.A, quad.B) (quad.C, quad.D) →
  SegmentEqual (quad.A, quad.D) (quad.B, quad.C) →
  ¬ (IsParallelogram quad) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_for_parallelogram_l63_6336


namespace NUMINAMATH_CALUDE_solve_system_of_equations_no_solution_for_inequalities_l63_6319

-- Part 1: System of equations
theorem solve_system_of_equations :
  ∃! (x y : ℚ), x + y = 5 ∧ 3 * x + 10 * y = 30 :=
by sorry

-- Part 2: System of inequalities
theorem no_solution_for_inequalities :
  ¬∃ (x : ℚ), (x + 7) / 2 < 4 ∧ (3 * x - 1) / 2 ≤ 2 * x - 3 :=
by sorry

end NUMINAMATH_CALUDE_solve_system_of_equations_no_solution_for_inequalities_l63_6319


namespace NUMINAMATH_CALUDE_quadratic_tangent_theorem_l63_6365

/-- Represents a quadratic function of the form ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determines if a quadratic function is tangent to the x-axis -/
def isTangentToXAxis (f : QuadraticFunction) : Prop :=
  f.b^2 - 4*f.a*f.c = 0

/-- Determines if the vertex of a quadratic function is a minimum point -/
def hasMinimumVertex (f : QuadraticFunction) : Prop :=
  f.a > 0

/-- The main theorem to be proved -/
theorem quadratic_tangent_theorem :
  ∀ (d : ℝ),
  let f : QuadraticFunction := ⟨3, 12, d⟩
  isTangentToXAxis f →
  d = 12 ∧ hasMinimumVertex f := by
  sorry

end NUMINAMATH_CALUDE_quadratic_tangent_theorem_l63_6365


namespace NUMINAMATH_CALUDE_projection_parallel_condition_l63_6311

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a 3D line
  -- (simplified for this example)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane
  -- (simplified for this example)

/-- Projection of a line onto a plane -/
def project (l : Line3D) (p : Plane3D) : Line3D :=
  sorry -- Definition of projection

/-- Parallel lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

theorem projection_parallel_condition 
  (a b m n : Line3D) (α : Plane3D) 
  (h1 : a ≠ b)
  (h2 : m = project a α)
  (h3 : n = project b α)
  (h4 : m ≠ n) :
  (∀ (a b : Line3D), parallel a b → parallel (project a α) (project b α)) ∧
  (∃ (a b : Line3D), parallel (project a α) (project b α) ∧ ¬parallel a b) :=
sorry

end NUMINAMATH_CALUDE_projection_parallel_condition_l63_6311


namespace NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l63_6329

/-- The equation represents a hyperbola if both coefficients are nonzero and have opposite signs -/
def is_hyperbola (k : ℝ) : Prop :=
  k - 3 > 0 ∧ k > 0

/-- k > 3 is a sufficient condition for the equation to represent a hyperbola -/
theorem sufficient_condition (k : ℝ) (h : k > 3) : is_hyperbola k :=
sorry

/-- k > 3 is not a necessary condition for the equation to represent a hyperbola -/
theorem not_necessary_condition : ∃ k : ℝ, is_hyperbola k ∧ ¬(k > 3) :=
sorry

/-- k > 3 is a sufficient but not necessary condition for the equation to represent a hyperbola -/
theorem sufficient_but_not_necessary (k : ℝ) : 
  (k > 3 → is_hyperbola k) ∧ ¬(is_hyperbola k → k > 3) :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l63_6329


namespace NUMINAMATH_CALUDE_sequence_integer_count_l63_6310

def sequence_term (n : ℕ) : ℚ :=
  9720 / (3 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem sequence_integer_count :
  (∃ (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n))) ∧
  (∃! (k : ℕ), k > 0 ∧
    (∀ (n : ℕ), n < k → is_integer (sequence_term n)) ∧
    (∀ (n : ℕ), n ≥ k → ¬ is_integer (sequence_term n)) ∧
    k = 6) :=
by sorry

end NUMINAMATH_CALUDE_sequence_integer_count_l63_6310


namespace NUMINAMATH_CALUDE_midpoint_chain_l63_6334

/-- Given a line segment XY, we define points G, H, I, and J as follows:
  G is the midpoint of XY
  H is the midpoint of XG
  I is the midpoint of XH
  J is the midpoint of XI
  If XJ = 4, then XY = 64 -/
theorem midpoint_chain (X Y G H I J : ℝ) : 
  (G = (X + Y) / 2) →  -- G is midpoint of XY
  (H = (X + G) / 2) →  -- H is midpoint of XG
  (I = (X + H) / 2) →  -- I is midpoint of XH
  (J = (X + I) / 2) →  -- J is midpoint of XI
  (J - X = 4) →        -- XJ = 4
  (Y - X = 64) :=      -- XY = 64
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l63_6334


namespace NUMINAMATH_CALUDE_clock_rotation_impossibility_l63_6395

/-- Represents a clock face with 12 numbers -/
def ClockFace : Type := Fin 12

/-- The sum of all numbers on the clock face -/
def clockSum : ℕ := (List.range 12).sum + 12

/-- The target number to be achieved on all positions of the blackboard -/
def target : ℕ := 1984

/-- The number of positions on the clock face and blackboard -/
def numPositions : ℕ := 12

theorem clock_rotation_impossibility : 
  ¬ ∃ (n : ℕ), n * clockSum = numPositions * target := by
  sorry

end NUMINAMATH_CALUDE_clock_rotation_impossibility_l63_6395


namespace NUMINAMATH_CALUDE_people_in_room_l63_6396

theorem people_in_room (total_chairs : ℚ) (occupied_chairs : ℚ) (empty_chairs : ℚ) 
  (h1 : empty_chairs = 5)
  (h2 : occupied_chairs = (2/3) * total_chairs)
  (h3 : empty_chairs = (1/3) * total_chairs)
  (h4 : occupied_chairs = 10) :
  ∃ (total_people : ℚ), total_people = 50/3 ∧ (3/5) * total_people = occupied_chairs := by
  sorry

end NUMINAMATH_CALUDE_people_in_room_l63_6396


namespace NUMINAMATH_CALUDE_solution_to_equation_l63_6333

theorem solution_to_equation : ∃ x : ℝ, ((18 + x) / 3 + 10) / 5 = 4 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l63_6333


namespace NUMINAMATH_CALUDE_tissue_pallet_ratio_l63_6322

theorem tissue_pallet_ratio (total_pallets : ℕ) 
  (paper_towel_pallets : ℕ) (paper_plate_pallets : ℕ) (paper_cup_pallets : ℕ) :
  total_pallets = 20 →
  paper_towel_pallets = total_pallets / 2 →
  paper_plate_pallets = total_pallets / 5 →
  paper_cup_pallets = 1 →
  let tissue_pallets := total_pallets - (paper_towel_pallets + paper_plate_pallets + paper_cup_pallets)
  (tissue_pallets : ℚ) / (total_pallets : ℚ) = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_tissue_pallet_ratio_l63_6322


namespace NUMINAMATH_CALUDE_inequality_proof_l63_6301

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + 2*c)) + (b / (c + 2*a)) + (c / (a + 2*b)) > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l63_6301


namespace NUMINAMATH_CALUDE_pig_year_paintings_distribution_l63_6325

theorem pig_year_paintings_distribution (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 3) :
  let total_outcomes := k^n
  let favorable_outcomes := (n.choose 2) * (k.factorial)
  (favorable_outcomes : ℚ) / total_outcomes = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_pig_year_paintings_distribution_l63_6325


namespace NUMINAMATH_CALUDE_base4_product_l63_6356

-- Define a function to convert from base 4 to decimal
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldr (fun (i, d) acc => acc + d * (4 ^ i)) 0

-- Define a function to convert from decimal to base 4
def decimalToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
  aux n []

-- Define the two base 4 numbers
def num1 : List Nat := [1, 3, 2]  -- 132₄
def num2 : List Nat := [1, 2]     -- 12₄

-- State the theorem
theorem base4_product :
  decimalToBase4 (base4ToDecimal num1 * base4ToDecimal num2) = [2, 3, 1, 0] := by
  sorry

end NUMINAMATH_CALUDE_base4_product_l63_6356


namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l63_6347

theorem larger_number_in_ratio (a b : ℝ) : 
  a / b = 8 / 3 → a + b = 143 → max a b = 104 := by sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l63_6347


namespace NUMINAMATH_CALUDE_infinite_gcd_condition_l63_6304

open Set Function Nat

/-- A permutation of positive integers -/
def PositiveIntegerPermutation := ℕ+ → ℕ+

/-- The set of indices satisfying the GCD condition -/
def GcdConditionSet (a : PositiveIntegerPermutation) : Set ℕ+ :=
  {i | Nat.gcd (a i) (a (i + 1)) ≤ (3 * i) / 4}

/-- The main theorem -/
theorem infinite_gcd_condition (a : PositiveIntegerPermutation) 
  (h : Bijective a) : Infinite (GcdConditionSet a) := by
  sorry


end NUMINAMATH_CALUDE_infinite_gcd_condition_l63_6304


namespace NUMINAMATH_CALUDE_second_plot_germination_rate_l63_6386

/-- Calculates the germination rate of the second plot given the number of seeds in each plot,
    the germination rate of the first plot, and the overall germination rate. -/
theorem second_plot_germination_rate 
  (seeds_first_plot : ℕ)
  (seeds_second_plot : ℕ)
  (germination_rate_first_plot : ℚ)
  (overall_germination_rate : ℚ)
  (h1 : seeds_first_plot = 300)
  (h2 : seeds_second_plot = 200)
  (h3 : germination_rate_first_plot = 25 / 100)
  (h4 : overall_germination_rate = 27 / 100)
  : (overall_germination_rate * (seeds_first_plot + seeds_second_plot) - 
     germination_rate_first_plot * seeds_first_plot) / seeds_second_plot = 30 / 100 := by
  sorry

end NUMINAMATH_CALUDE_second_plot_germination_rate_l63_6386


namespace NUMINAMATH_CALUDE_pentagon_fifth_angle_l63_6361

/-- A pentagon with four known angles -/
structure Pentagon where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ
  angle5 : ℝ
  sum_of_angles : angle1 + angle2 + angle3 + angle4 + angle5 = 540

/-- The theorem to prove -/
theorem pentagon_fifth_angle (p : Pentagon) 
  (h1 : p.angle1 = 270)
  (h2 : p.angle2 = 70)
  (h3 : p.angle3 = 60)
  (h4 : p.angle4 = 90) :
  p.angle5 = 50 := by
  sorry


end NUMINAMATH_CALUDE_pentagon_fifth_angle_l63_6361


namespace NUMINAMATH_CALUDE_inequality_solution_l63_6368

def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then { x | x > 1 }
  else if a > 1 then { x | 1/a < x ∧ x < 1 }
  else if a = 1 then ∅
  else if 0 < a ∧ a < 1 then { x | 1 < x ∧ x < 1/a }
  else { x | x < 1/a ∨ x > 1 }

theorem inequality_solution (a : ℝ) :
  { x : ℝ | (a*x - 1)*(x - 1) < 0 } = solution_set a :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l63_6368


namespace NUMINAMATH_CALUDE_jacobs_flock_total_l63_6371

/-- Represents the composition of Jacob's flock -/
structure Flock where
  goats : ℕ
  sheep : ℕ

/-- Theorem stating the total number of animals in Jacob's flock -/
theorem jacobs_flock_total (f : Flock) 
  (h1 : f.goats = f.sheep / 2)  -- One third of animals are goats, so goats = (sheep + goats) / 3 = sheep / 2
  (h2 : f.sheep = f.goats + 12) -- There are 12 more sheep than goats
  : f.goats + f.sheep = 36 := by
  sorry


end NUMINAMATH_CALUDE_jacobs_flock_total_l63_6371


namespace NUMINAMATH_CALUDE_concert_attendance_l63_6346

/-- The number of buses used for the concert trip -/
def number_of_buses : ℕ := 8

/-- The number of students each bus can carry -/
def students_per_bus : ℕ := 45

/-- The total number of students who went to the concert -/
def total_students : ℕ := number_of_buses * students_per_bus

/-- Theorem stating that the total number of students who went to the concert is 360 -/
theorem concert_attendance : total_students = 360 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_l63_6346


namespace NUMINAMATH_CALUDE_seating_arrangements_count_l63_6369

/-- Represents the number of people to be seated. -/
def num_people : ℕ := 4

/-- Represents the total number of chairs in a row. -/
def total_chairs : ℕ := 8

/-- Represents the number of consecutive empty seats required. -/
def consecutive_empty_seats : ℕ := 3

/-- Calculates the number of seating arrangements for the given conditions. -/
def seating_arrangements (p : ℕ) (c : ℕ) (e : ℕ) : ℕ :=
  (Nat.factorial (p + 1)) * (c - p - e + 1)

/-- Theorem stating the number of seating arrangements for the given conditions. -/
theorem seating_arrangements_count :
  seating_arrangements num_people total_chairs consecutive_empty_seats = 600 :=
by
  sorry


end NUMINAMATH_CALUDE_seating_arrangements_count_l63_6369


namespace NUMINAMATH_CALUDE_infinitely_many_n_squared_divides_b_power_n_plus_one_l63_6305

theorem infinitely_many_n_squared_divides_b_power_n_plus_one
  (b : ℕ) (hb : b > 2) :
  (∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, n^2 ∣ b^n + 1) ↔ ¬∃ k : ℕ, b + 1 = 2^k :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_squared_divides_b_power_n_plus_one_l63_6305


namespace NUMINAMATH_CALUDE_matrix_property_l63_6398

theorem matrix_property (a b c d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  (A.transpose = A⁻¹) → (Matrix.det A = 1) → (a^2 + b^2 + c^2 + d^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_matrix_property_l63_6398


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l63_6389

/-- Prove that the polar equation ρ = 4cosθ is equivalent to the Cartesian equation (x - 2)² + y² = 4 -/
theorem polar_to_cartesian_circle (x y ρ θ : ℝ) :
  (ρ = 4 * Real.cos θ) ∧ (x = ρ * Real.cos θ) ∧ (y = ρ * Real.sin θ) →
  (x - 2)^2 + y^2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l63_6389


namespace NUMINAMATH_CALUDE_distance_swum_against_current_l63_6351

/-- The distance swum against the current given swimming speed, current speed, and time taken -/
theorem distance_swum_against_current 
  (swimming_speed : ℝ) 
  (current_speed : ℝ) 
  (time_taken : ℝ) 
  (h1 : swimming_speed = 4)
  (h2 : current_speed = 2)
  (h3 : time_taken = 6) : 
  (swimming_speed - current_speed) * time_taken = 12 := by
  sorry

#check distance_swum_against_current

end NUMINAMATH_CALUDE_distance_swum_against_current_l63_6351


namespace NUMINAMATH_CALUDE_total_paintable_area_l63_6343

/-- Represents a rectangular surface with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Represents a wall with its dimensions and optional window or door -/
structure Wall where
  dimensions : Rectangle
  opening : Option Rectangle

/-- Calculates the paintable area of a wall -/
def Wall.paintableArea (w : Wall) : ℝ :=
  w.dimensions.area - (match w.opening with
    | some o => o.area
    | none => 0)

/-- The four walls of the room -/
def walls : List Wall := [
  { dimensions := { width := 4, height := 8 },
    opening := some { width := 2, height := 3 } },
  { dimensions := { width := 6, height := 8 },
    opening := some { width := 3, height := 6.5 } },
  { dimensions := { width := 4, height := 8 },
    opening := some { width := 3, height := 4 } },
  { dimensions := { width := 6, height := 8 },
    opening := none }
]

theorem total_paintable_area :
  (walls.map Wall.paintableArea).sum = 122.5 := by sorry

end NUMINAMATH_CALUDE_total_paintable_area_l63_6343


namespace NUMINAMATH_CALUDE_kaiden_first_week_cans_l63_6363

/-- The number of cans collected in the first week of Kaiden's soup can collection -/
def cans_first_week (goal : ℕ) (cans_second_week : ℕ) (cans_needed : ℕ) : ℕ :=
  goal - cans_needed - cans_second_week

/-- Theorem stating that Kaiden collected 158 cans in the first week -/
theorem kaiden_first_week_cans :
  cans_first_week 500 259 83 = 158 := by sorry

end NUMINAMATH_CALUDE_kaiden_first_week_cans_l63_6363


namespace NUMINAMATH_CALUDE_tangent_line_sum_l63_6358

/-- A function with a specific tangent line at x = 1 -/
def HasTangentLineAtOne (f : ℝ → ℝ) : Prop :=
  ∃ (m b : ℝ), m = (1/2) ∧ b = 2 ∧
  ∀ x, f x = m * (x - 1) + f 1

theorem tangent_line_sum (f : ℝ → ℝ) (hf : HasTangentLineAtOne f) :
  f 1 + (deriv f) 1 = 3 := by
  sorry

#check tangent_line_sum

end NUMINAMATH_CALUDE_tangent_line_sum_l63_6358


namespace NUMINAMATH_CALUDE_unique_solution_in_interval_l63_6352

theorem unique_solution_in_interval (x : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  ((2 - Real.sin (2 * x)) * Real.sin (x + Real.pi / 4) = 1) ↔
  (x = Real.pi / 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_in_interval_l63_6352


namespace NUMINAMATH_CALUDE_tom_tim_typing_ratio_l63_6394

/-- 
Given that Tim and Tom can type 12 pages in one hour together,
and 14 pages when Tom increases his speed by 25%,
prove that the ratio of Tom's normal typing speed to Tim's is 2:1
-/
theorem tom_tim_typing_ratio :
  ∀ (tim_speed tom_speed : ℝ),
    tim_speed + tom_speed = 12 →
    tim_speed + (1.25 * tom_speed) = 14 →
    tom_speed / tim_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_tom_tim_typing_ratio_l63_6394


namespace NUMINAMATH_CALUDE_number_line_expressions_l63_6302

theorem number_line_expressions (P Q R S T : ℝ) 
  (hP : P > 3 ∧ P < 4)
  (hQ : Q > 1 ∧ Q < 1.2)
  (hR : R > -0.2 ∧ R < 0)
  (hS : S > 0.8 ∧ S < 1)
  (hT : T > 1.4 ∧ T < 1.6) :
  R / (P * Q) < 0 ∧ (S + T) / R < 0 ∧ P - Q ≥ 0 ∧ P * Q ≥ 0 ∧ (S / Q) * P ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_number_line_expressions_l63_6302


namespace NUMINAMATH_CALUDE_remainder_problem_l63_6342

theorem remainder_problem (N : ℤ) : N % 357 = 36 → N % 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l63_6342


namespace NUMINAMATH_CALUDE_chord_length_on_circle_l63_6337

/-- The length of the chord intercepted by y=x on (x-0)^2+(y-2)^2=4 is 2√2 -/
theorem chord_length_on_circle (x y : ℝ) : 
  (x - 0)^2 + (y - 2)^2 = 4 → y = x → 
  ∃ (a b : ℝ), (a - 0)^2 + (b - 2)^2 = 4 ∧ b = a ∧ 
  Real.sqrt ((a - x)^2 + (b - y)^2) = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_chord_length_on_circle_l63_6337


namespace NUMINAMATH_CALUDE_square_division_l63_6300

/-- A square can be divided into two equal parts in at least four different ways. -/
theorem square_division (s : ℝ) (h : s > 0) :
  ∃ (rect1 rect2 : ℝ × ℝ) (tri1 tri2 : ℝ × ℝ × ℝ),
    -- Vertical division
    rect1 = (s, s/2) ∧
    -- Horizontal division
    rect2 = (s/2, s) ∧
    -- Diagonal division (top-left to bottom-right)
    tri1 = (s, s, Real.sqrt 2 * s) ∧
    -- Diagonal division (top-right to bottom-left)
    tri2 = (s, s, Real.sqrt 2 * s) ∧
    -- All divisions result in equal areas
    s * (s/2) = (s/2) * s ∧
    s * (s/2) = (1/2) * s * s ∧
    -- All divisions are valid (non-negative dimensions)
    s > 0 ∧ s/2 > 0 ∧ Real.sqrt 2 * s > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_square_division_l63_6300


namespace NUMINAMATH_CALUDE_mini_quiz_multiple_choice_count_l63_6307

/-- The number of ways to answer 3 true-false questions where all answers cannot be the same -/
def truefalse_combinations : ℕ := 6

/-- The number of answer choices for each multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- The total number of ways to write the answer key -/
def total_combinations : ℕ := 96

/-- Proves that the number of multiple-choice questions is 2 -/
theorem mini_quiz_multiple_choice_count :
  ∃ (n : ℕ), truefalse_combinations * multiple_choice_options ^ n = total_combinations ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_mini_quiz_multiple_choice_count_l63_6307


namespace NUMINAMATH_CALUDE_arithmetic_progression_with_prime_terms_l63_6303

-- Define an arithmetic progression
def ArithmeticProgression (a k : ℕ) : ℕ → ℕ := fun n => a + k * n

-- Define the property of having infinitely many prime terms at prime indices
def HasInfinitelyManyPrimeTermsAtPrimeIndices (seq : ℕ → ℕ) : Prop :=
  ∃ N : ℕ, ∀ p : ℕ, Prime p → p > N → Prime (seq p)

-- State the theorem
theorem arithmetic_progression_with_prime_terms (seq : ℕ → ℕ) :
  (∃ a k : ℕ, seq = ArithmeticProgression a k) →
  HasInfinitelyManyPrimeTermsAtPrimeIndices seq →
  (∃ P : ℕ, Prime P ∧ seq = fun _ => P) ∨ seq = id :=
sorry

end NUMINAMATH_CALUDE_arithmetic_progression_with_prime_terms_l63_6303


namespace NUMINAMATH_CALUDE_number_remainder_l63_6306

theorem number_remainder (N : ℤ) 
  (h1 : N % 195 = 79)
  (h2 : N % 273 = 109) : 
  N % 39 = 1 := by
  sorry

end NUMINAMATH_CALUDE_number_remainder_l63_6306


namespace NUMINAMATH_CALUDE_swim_team_girls_count_l63_6327

theorem swim_team_girls_count (total : ℕ) (ratio : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 96 → 
  ratio = 5 → 
  girls = ratio * boys → 
  total = girls + boys → 
  girls = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_swim_team_girls_count_l63_6327


namespace NUMINAMATH_CALUDE_mark_change_factor_l63_6332

theorem mark_change_factor (n : ℕ) (initial_avg final_avg : ℚ) (h1 : n = 25) (h2 : initial_avg = 70) (h3 : final_avg = 140) :
  (n * final_avg) / (n * initial_avg) = 2 :=
sorry

end NUMINAMATH_CALUDE_mark_change_factor_l63_6332


namespace NUMINAMATH_CALUDE_derivative_f_at_one_l63_6376

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x

theorem derivative_f_at_one : 
  deriv f 1 = 2 + Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_one_l63_6376


namespace NUMINAMATH_CALUDE_statement_A_statement_B_l63_6393

-- Define the parabola E
def E (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of E
def F : ℝ × ℝ := (1, 0)

-- Define the circle F
def circle_F (x y r : ℝ) : Prop := (x - 1)^2 + y^2 = r^2

-- Define the line l_0
def l_0 (t : ℝ) (x y : ℝ) : Prop := x = t*y + 1

-- Define the intersection points A and B
def A (t : ℝ) : ℝ × ℝ := (t^2 + 1, 2*t)
def B (t : ℝ) : ℝ × ℝ := (t^2 + 1, -2*t)

-- Define the midpoint M
def M (t : ℝ) : ℝ × ℝ := (2*t^2 + 1, 2*t)

-- Define point T
def T : ℝ × ℝ := (0, 1)

-- Theorem for statement A
theorem statement_A (t : ℝ) : 
  let y_1 := (A t).2
  let y_2 := (B t).2
  let y_3 := -1/t
  1/y_1 + 1/y_2 = 1/y_3 :=
sorry

-- Theorem for statement B
theorem statement_B : 
  ∃ a b c : ℝ, ∀ t : ℝ, 
    let (x, y) := M t
    y^2 = a*x + b*y + c :=
sorry

end NUMINAMATH_CALUDE_statement_A_statement_B_l63_6393


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l63_6382

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 4 + 2 * Complex.I) :
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l63_6382


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l63_6367

/-- The lateral surface area of a cone with base radius 5 and height 12 is 65π. -/
theorem cone_lateral_surface_area :
  let r : ℝ := 5
  let h : ℝ := 12
  let l : ℝ := Real.sqrt (r^2 + h^2)
  π * r * l = 65 * π :=
by sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l63_6367


namespace NUMINAMATH_CALUDE_inequality_proof_l63_6380

theorem inequality_proof (x y : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) :
  x + y + 1 / (x * y) ≤ 1 / x + 1 / y + x * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l63_6380


namespace NUMINAMATH_CALUDE_circle_equation_l63_6399

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 2 * x - y + 6 = 0

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define tangency to coordinate axes
def tangent_to_axes (c : Circle) : Prop :=
  c.radius = |c.center.1| ∧ c.radius = |c.center.2|

-- Define the center being on the line
def center_on_line (c : Circle) : Prop :=
  line_equation c.center.1 c.center.2

-- Define the standard equation of a circle
def standard_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

-- Theorem statement
theorem circle_equation :
  ∀ c : Circle,
  tangent_to_axes c →
  center_on_line c →
  (∃ x y : ℝ, standard_equation c x y) →
  (∀ x y : ℝ, standard_equation c x y ↔ 
    ((x + 2)^2 + (y - 2)^2 = 4 ∨ (x + 6)^2 + (y + 6)^2 = 36)) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l63_6399


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_three_l63_6331

theorem arithmetic_sqrt_of_three (x : ℝ) : x = Real.sqrt 3 ↔ x ≥ 0 ∧ x ^ 2 = 3 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_three_l63_6331


namespace NUMINAMATH_CALUDE_john_remaining_money_l63_6339

def base7_to_base10 (n : ℕ) : ℕ :=
  6 * 7^3 + 5 * 7^2 + 3 * 7^1 + 4 * 7^0

theorem john_remaining_money :
  let savings : ℕ := base7_to_base10 6534
  let ticket_cost : ℕ := 1200
  savings - ticket_cost = 1128 := by sorry

end NUMINAMATH_CALUDE_john_remaining_money_l63_6339


namespace NUMINAMATH_CALUDE_multiply_cube_by_negative_l63_6328

/-- For any real number y, 2y³ * (-y) = -2y⁴ -/
theorem multiply_cube_by_negative (y : ℝ) : 2 * y^3 * (-y) = -2 * y^4 := by
  sorry

end NUMINAMATH_CALUDE_multiply_cube_by_negative_l63_6328


namespace NUMINAMATH_CALUDE_fraction_meaningfulness_l63_6335

theorem fraction_meaningfulness (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x + 1)) ↔ x ≠ -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_meaningfulness_l63_6335


namespace NUMINAMATH_CALUDE_set_C_characterization_l63_6373

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}

def B (a : ℝ) : Set ℝ := {x | a*x - 2 = 0}

def C : Set ℝ := {0, 1, 2}

theorem set_C_characterization :
  ∀ a : ℝ, (A ∪ B a = A) ↔ a ∈ C :=
sorry

end NUMINAMATH_CALUDE_set_C_characterization_l63_6373


namespace NUMINAMATH_CALUDE_vector_dot_product_theorem_l63_6357

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 1]
def vector_b (y : ℝ) : Fin 2 → ℝ := ![1, y]
def vector_c : Fin 2 → ℝ := ![3, -6]

def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

def perpendicular (u v : Fin 2 → ℝ) : Prop := dot_product u v = 0

def parallel (u v : Fin 2 → ℝ) : Prop := ∃ (k : ℝ), ∀ (i : Fin 2), u i = k * (v i)

theorem vector_dot_product_theorem (x y : ℝ) :
  perpendicular (vector_a x) vector_c →
  parallel (vector_b y) vector_c →
  dot_product (vector_a x + vector_b y) vector_c = 15 := by
  sorry

end NUMINAMATH_CALUDE_vector_dot_product_theorem_l63_6357


namespace NUMINAMATH_CALUDE_sum_of_max_min_M_l63_6385

/-- The set T of points (x, y) satisfying |x+1| + |y-2| ≤ 3 -/
def T : Set (ℝ × ℝ) := {p | |p.1 + 1| + |p.2 - 2| ≤ 3}

/-- The set M of values x + 2y for (x, y) in T -/
def M : Set ℝ := {z | ∃ p ∈ T, z = p.1 + 2 * p.2}

theorem sum_of_max_min_M : (⨆ z ∈ M, z) + (⨅ z ∈ M, z) = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_M_l63_6385


namespace NUMINAMATH_CALUDE_problem_solution_l63_6312

theorem problem_solution (x y : ℝ) 
  (h1 : Real.sqrt (3 + Real.sqrt x) = 4) 
  (h2 : x + y = 58) : 
  y = -111 := by sorry

end NUMINAMATH_CALUDE_problem_solution_l63_6312


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l63_6391

/-- A line y = 2x + a is tangent to the circle x^2 + y^2 = 9 if and only if a = ±3√5 -/
theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, y = 2*x + a ∧ x^2 + y^2 = 9 → (∀ ε > 0, ∃ δ > 0, ∀ x' y', 
    x'^2 + y'^2 = 9 → (x' - x)^2 + (y' - y)^2 < δ^2 → y' ≠ 2*x' + a)) ↔ 
  a = 3 * Real.sqrt 5 ∨ a = -3 * Real.sqrt 5 :=
sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l63_6391


namespace NUMINAMATH_CALUDE_remainder_2753_div_98_l63_6387

theorem remainder_2753_div_98 : 2753 % 98 = 9 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2753_div_98_l63_6387


namespace NUMINAMATH_CALUDE_voter_distribution_l63_6338

theorem voter_distribution (total_voters : ℝ) (dem_percent : ℝ) (rep_percent : ℝ) 
  (rep_vote_a : ℝ) (total_vote_a : ℝ) (dem_vote_a : ℝ) :
  dem_percent = 0.6 →
  rep_percent = 1 - dem_percent →
  rep_vote_a = 0.2 →
  total_vote_a = 0.5 →
  dem_vote_a * dem_percent + rep_vote_a * rep_percent = total_vote_a →
  dem_vote_a = 0.7 := by
sorry

end NUMINAMATH_CALUDE_voter_distribution_l63_6338


namespace NUMINAMATH_CALUDE_translation_theorem_l63_6344

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation2D where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (p : Point2D) (t : Translation2D) : Point2D :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem (A B : Point2D) (A' : Point2D) :
  A = Point2D.mk 2 2 →
  B = Point2D.mk (-1) 1 →
  A' = Point2D.mk (-2) (-2) →
  let t : Translation2D := { dx := A'.x - A.x, dy := A'.y - A.y }
  applyTranslation B t = Point2D.mk (-5) (-3) := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l63_6344


namespace NUMINAMATH_CALUDE_extreme_point_of_f_l63_6392

/-- The function f(x) = 3/2 * x^2 - ln(x) for x > 0 has an extreme point at x = √3/3 -/
theorem extreme_point_of_f (x : ℝ) (h : x > 0) : 
  let f := fun (x : ℝ) => 3/2 * x^2 - Real.log x
  ∃ (c : ℝ), c = Real.sqrt 3 / 3 ∧ 
    (∀ y > 0, f y ≥ f c) ∨ (∀ y > 0, f y ≤ f c) := by
  sorry


end NUMINAMATH_CALUDE_extreme_point_of_f_l63_6392


namespace NUMINAMATH_CALUDE_unique_base_l63_6388

/-- Converts a number from base h to base 10 --/
def to_base_10 (digits : List Nat) (h : Nat) : Nat :=
  digits.foldr (fun d acc => d + h * acc) 0

/-- The equation in base h --/
def equation_holds (h : Nat) : Prop :=
  h > 9 ∧ 
  to_base_10 [8, 3, 2, 7] h + to_base_10 [9, 4, 6, 1] h = to_base_10 [1, 9, 2, 8, 8] h

theorem unique_base : ∃! h, equation_holds h :=
  sorry

end NUMINAMATH_CALUDE_unique_base_l63_6388


namespace NUMINAMATH_CALUDE_f_derivative_at_fixed_point_l63_6379

noncomputable def f (x : ℝ) : ℝ := Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos (Real.cos x)))))))

theorem f_derivative_at_fixed_point (a : ℝ) (h : a = Real.cos a) :
  deriv f a = a^8 - 4*a^6 + 6*a^4 - 4*a^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_f_derivative_at_fixed_point_l63_6379


namespace NUMINAMATH_CALUDE_additional_seashells_is_8_l63_6381

/-- The number of additional seashells Carina puts in each week -/
def additional_seashells : ℕ := sorry

/-- The number of seashells in the jar this week -/
def initial_seashells : ℕ := 50

/-- The number of seashells in the jar after 4 weeks -/
def final_seashells : ℕ := 130

/-- The number of weeks -/
def weeks : ℕ := 4

/-- Formula for the total number of seashells after n weeks -/
def total_seashells (n : ℕ) : ℕ :=
  initial_seashells + n * additional_seashells + (n * (n - 1) / 2) * additional_seashells

/-- Theorem stating that the number of additional seashells per week is 8 -/
theorem additional_seashells_is_8 :
  additional_seashells = 8 ∧
  (∀ n : ℕ, n ≤ weeks → total_seashells n ≤ total_seashells (n + 1)) ∧
  total_seashells weeks = final_seashells :=
sorry

end NUMINAMATH_CALUDE_additional_seashells_is_8_l63_6381


namespace NUMINAMATH_CALUDE_households_without_car_or_bike_l63_6366

theorem households_without_car_or_bike 
  (total : ℕ) 
  (both : ℕ) 
  (with_car : ℕ) 
  (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : both = 20)
  (h3 : with_car = 44)
  (h4 : bike_only = 35) :
  total - (with_car + bike_only) = 11 :=
by sorry

end NUMINAMATH_CALUDE_households_without_car_or_bike_l63_6366


namespace NUMINAMATH_CALUDE_hyperbola_sum_l63_6320

theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 3 ∧ 
  k = -1 ∧ 
  (3 + Real.sqrt 45 - 3)^2 = c^2 ∧ 
  (6 - 3)^2 = a^2 ∧ 
  b^2 = c^2 - a^2 → 
  h + k + a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l63_6320


namespace NUMINAMATH_CALUDE_outfit_choices_count_l63_6384

/-- The number of colors available for each clothing item -/
def num_colors : ℕ := 5

/-- The number of shirts available -/
def num_shirts : ℕ := 5

/-- The number of pants available -/
def num_pants : ℕ := 5

/-- The number of hats available -/
def num_hats : ℕ := 5

/-- The total number of possible outfit combinations -/
def total_combinations : ℕ := num_shirts * num_pants * num_hats

/-- The number of outfit combinations where all items are the same color -/
def same_color_combinations : ℕ := num_colors

/-- The number of valid outfit choices -/
def valid_outfit_choices : ℕ := total_combinations - same_color_combinations

theorem outfit_choices_count : valid_outfit_choices = 120 := by
  sorry

end NUMINAMATH_CALUDE_outfit_choices_count_l63_6384


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l63_6372

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + (a - 1) * x + (a - 1) < 0) ↔ a < -1/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l63_6372


namespace NUMINAMATH_CALUDE_jerry_always_escapes_l63_6317

/-- Represents the square pool -/
structure Pool :=
  (side : ℝ)
  (is_positive : side > 0)

/-- Represents the speeds of Tom and Jerry -/
structure Speeds :=
  (jerry_swim : ℝ)
  (tom_run : ℝ)
  (speed_ratio : tom_run = 4 * jerry_swim)
  (positive_speeds : jerry_swim > 0 ∧ tom_run > 0)

/-- Represents a point in the pool -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defines whether a point is inside or on the edge of the pool -/
def in_pool (p : Point) (pool : Pool) : Prop :=
  0 ≤ p.x ∧ p.x ≤ pool.side ∧ 0 ≤ p.y ∧ p.y ≤ pool.side

/-- Defines whether Jerry can escape from Tom -/
def can_escape (pool : Pool) (speeds : Speeds) : Prop :=
  ∀ (jerry_start tom_start : Point),
    in_pool jerry_start pool →
    ¬in_pool tom_start pool →
    ∃ (escape_point : Point),
      in_pool escape_point pool ∧
      (escape_point.x = 0 ∨ escape_point.x = pool.side ∨
       escape_point.y = 0 ∨ escape_point.y = pool.side) ∧
      (escape_point.x - jerry_start.x) ^ 2 + (escape_point.y - jerry_start.y) ^ 2 <
      ((escape_point.x - tom_start.x) ^ 2 + (escape_point.y - tom_start.y) ^ 2) * (speeds.jerry_swim / speeds.tom_run) ^ 2

theorem jerry_always_escapes (pool : Pool) (speeds : Speeds) :
  can_escape pool speeds :=
sorry

end NUMINAMATH_CALUDE_jerry_always_escapes_l63_6317


namespace NUMINAMATH_CALUDE_octagon_area_reduction_l63_6383

theorem octagon_area_reduction (x : ℝ) : 
  x > 0 ∧ x < 1 →  -- The smaller square's side length is positive and less than the original square
  4 + 2*x = 1.4 * 4 →  -- Perimeter condition
  (1 - x^2) / 1 = 0.36 :=  -- Area reduction
by sorry

end NUMINAMATH_CALUDE_octagon_area_reduction_l63_6383


namespace NUMINAMATH_CALUDE_chocolate_bar_sales_l63_6315

/-- Calculates the money made from selling chocolate bars -/
def money_made (total_bars : ℕ) (price_per_bar : ℕ) (unsold_bars : ℕ) : ℕ :=
  (total_bars - unsold_bars) * price_per_bar

/-- Proves that selling 4 out of 11 bars at $4 each yields $16 -/
theorem chocolate_bar_sales : money_made 11 4 7 = 16 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_bar_sales_l63_6315


namespace NUMINAMATH_CALUDE_line_intersects_segment_midpoint_l63_6378

theorem line_intersects_segment_midpoint (b : ℝ) : 
  let p1 : ℝ × ℝ := (3, 2)
  let p2 : ℝ × ℝ := (7, 6)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (b = 9) ↔ (midpoint.1 + midpoint.2 = b) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_segment_midpoint_l63_6378


namespace NUMINAMATH_CALUDE_min_value_quadratic_expression_l63_6353

theorem min_value_quadratic_expression :
  ∀ x y : ℝ, (x + 3)^2 + 2*(y - 2)^2 + 4*(x - 7)^2 + (y + 4)^2 ≥ 104 ∧
  ∃ x₀ y₀ : ℝ, (x₀ + 3)^2 + 2*(y₀ - 2)^2 + 4*(x₀ - 7)^2 + (y₀ + 4)^2 = 104 :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_expression_l63_6353


namespace NUMINAMATH_CALUDE_range_of_a_l63_6350

theorem range_of_a (a b c : ℝ) 
  (h1 : b^2 + c^2 = -a^2 + 14*a + 5) 
  (h2 : b*c = a^2 - 2*a + 10) : 
  1 ≤ a ∧ a ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l63_6350


namespace NUMINAMATH_CALUDE_train_speed_calculation_l63_6397

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 100 →
  bridge_length = 150 →
  crossing_time = 49.9960003199744 →
  ∃ (speed : ℝ), (abs (speed - 18) < 0.1 ∧ 
    speed = (train_length + bridge_length) / crossing_time * 3.6) := by
  sorry


end NUMINAMATH_CALUDE_train_speed_calculation_l63_6397
