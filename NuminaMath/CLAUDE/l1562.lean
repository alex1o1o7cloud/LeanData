import Mathlib

namespace NUMINAMATH_CALUDE_candy_bar_problem_l1562_156214

theorem candy_bar_problem (F : ℕ) : 
  (∃ (J : ℕ), 
    J = 10 * (2 * F + 6) ∧ 
    (2 * F + 6) = F + (F + 6) ∧
    (40 * J) / 100 = 120) → 
  F = 12 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_problem_l1562_156214


namespace NUMINAMATH_CALUDE_power_2014_of_abs_one_l1562_156244

theorem power_2014_of_abs_one (a : ℝ) : |a| = 1 → a^2014 = 1 := by sorry

end NUMINAMATH_CALUDE_power_2014_of_abs_one_l1562_156244


namespace NUMINAMATH_CALUDE_tangent_circles_radius_l1562_156231

/-- Two circles are tangent if the distance between their centers equals the sum of their radii -/
def are_tangent (c1_center c2_center : ℝ × ℝ) (r : ℝ) : Prop :=
  Real.sqrt ((c1_center.1 - c2_center.1)^2 + (c1_center.2 - c2_center.2)^2) = 2 * r

theorem tangent_circles_radius (r : ℝ) (h : r > 0) :
  are_tangent (0, 0) (3, -1) r → r = Real.sqrt 10 / 2 := by
  sorry

#check tangent_circles_radius

end NUMINAMATH_CALUDE_tangent_circles_radius_l1562_156231


namespace NUMINAMATH_CALUDE_max_cables_for_given_network_l1562_156262

/-- Represents a computer network with two brands of computers. -/
structure ComputerNetwork where
  total_employees : ℕ
  brand_x_count : ℕ
  brand_y_count : ℕ
  max_connections_per_computer : ℕ
  (total_is_sum : total_employees = brand_x_count + brand_y_count)
  (max_connections_positive : max_connections_per_computer > 0)

/-- The maximum number of cables that can be used in the network. -/
def max_cables (network : ComputerNetwork) : ℕ :=
  min (network.brand_x_count * network.max_connections_per_computer)
      (network.brand_y_count * network.max_connections_per_computer)

/-- The theorem stating the maximum number of cables for the given network configuration. -/
theorem max_cables_for_given_network :
  ∃ (network : ComputerNetwork),
    network.total_employees = 40 ∧
    network.brand_x_count = 25 ∧
    network.brand_y_count = 15 ∧
    network.max_connections_per_computer = 3 ∧
    max_cables network = 45 := by
  sorry

end NUMINAMATH_CALUDE_max_cables_for_given_network_l1562_156262


namespace NUMINAMATH_CALUDE_distribute_5_3_eq_31_l1562_156278

/-- The number of ways to distribute n different items into k identical bags -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 different items into 3 identical bags -/
def distribute_5_3 : ℕ := distribute 5 3

theorem distribute_5_3_eq_31 : distribute_5_3 = 31 := by sorry

end NUMINAMATH_CALUDE_distribute_5_3_eq_31_l1562_156278


namespace NUMINAMATH_CALUDE_seven_equidistant_planes_l1562_156283

/-- A type representing a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A type representing a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Function to check if four points are coplanar -/
def areCoplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Function to check if a plane is equidistant from four points -/
def isEquidistant (plane : Plane3D) (p1 p2 p3 p4 : Point3D) : Prop := sorry

/-- Function to count the number of planes equidistant from four points -/
def countEquidistantPlanes (p1 p2 p3 p4 : Point3D) : ℕ := sorry

/-- Theorem stating that there are exactly 7 equidistant planes for four non-coplanar points -/
theorem seven_equidistant_planes
  (p1 p2 p3 p4 : Point3D)
  (h : ¬ areCoplanar p1 p2 p3 p4) :
  countEquidistantPlanes p1 p2 p3 p4 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_equidistant_planes_l1562_156283


namespace NUMINAMATH_CALUDE_total_crayons_l1562_156297

theorem total_crayons (orange_boxes : Nat) (orange_per_box : Nat)
                      (blue_boxes : Nat) (blue_per_box : Nat)
                      (red_boxes : Nat) (red_per_box : Nat) :
  orange_boxes = 6 →
  orange_per_box = 8 →
  blue_boxes = 7 →
  blue_per_box = 5 →
  red_boxes = 1 →
  red_per_box = 11 →
  orange_boxes * orange_per_box + blue_boxes * blue_per_box + red_boxes * red_per_box = 94 := by
  sorry

end NUMINAMATH_CALUDE_total_crayons_l1562_156297


namespace NUMINAMATH_CALUDE_max_value_ab_l1562_156284

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_geom_mean : Real.sqrt 2 = Real.sqrt (2^a * 2^b)) : 
  (∀ x y : ℝ, x > 0 → y > 0 → Real.sqrt 2 = Real.sqrt (2^x * 2^y) → a * b ≥ x * y) ∧ a * b = 1/4 :=
sorry

end NUMINAMATH_CALUDE_max_value_ab_l1562_156284


namespace NUMINAMATH_CALUDE_triangle_area_bound_l1562_156219

-- Define a triangle with integer coordinates
structure IntTriangle where
  A : ℤ × ℤ
  B : ℤ × ℤ
  C : ℤ × ℤ

-- Define a function to count integer points inside a triangle
def countInteriorPoints (t : IntTriangle) : ℕ := sorry

-- Define a function to count integer points on the edges of a triangle
def countBoundaryPoints (t : IntTriangle) : ℕ := sorry

-- Define a function to calculate the area of a triangle
def triangleArea (t : IntTriangle) : ℚ := sorry

-- Theorem statement
theorem triangle_area_bound (t : IntTriangle) :
  countInteriorPoints t = 1 → triangleArea t ≤ 9/2 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_bound_l1562_156219


namespace NUMINAMATH_CALUDE_unique_solution_iff_prime_l1562_156289

theorem unique_solution_iff_prime (n : ℕ) : 
  (∃! (x y : ℕ), (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / n) ↔ Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_unique_solution_iff_prime_l1562_156289


namespace NUMINAMATH_CALUDE_ndoti_winning_strategy_l1562_156238

/-- Represents a point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square on the plane -/
structure Square where
  a : Point
  b : Point
  c : Point
  d : Point

/-- Represents a quadrilateral on the plane -/
structure Quadrilateral where
  x : Point
  y : Point
  z : Point
  w : Point

/-- Calculates the area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Checks if a point is on a side of the square -/
def isOnSquareSide (p : Point) (s : Square) : Prop := sorry

/-- Ndoti's strategy function -/
def ndotiStrategy (s : Square) (x : Point) : Quadrilateral := sorry

/-- The main theorem stating Ndoti's winning strategy -/
theorem ndoti_winning_strategy (s : Square) :
  ∀ x : Point, isOnSquareSide x s →
    quadrilateralArea (ndotiStrategy s x) < (1/2) * squareArea s :=
by sorry

end NUMINAMATH_CALUDE_ndoti_winning_strategy_l1562_156238


namespace NUMINAMATH_CALUDE_unique_solution_linear_equation_l1562_156269

theorem unique_solution_linear_equation (a b : ℝ) :
  (a * 1 + b * 2 = 2) ∧ (a * 2 + b * 5 = 2) → a = 6 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_linear_equation_l1562_156269


namespace NUMINAMATH_CALUDE_son_father_height_relationship_l1562_156286

/-- Represents the possible types of relationships between variables -/
inductive RelationshipType
  | Deterministic
  | Correlation
  | Functional
  | None

/-- Represents the relationship between a son's height and his father's height -/
structure HeightRelationship where
  type : RelationshipType
  isUncertain : Bool

/-- Theorem: The relationship between a son's height and his father's height is a correlation relationship -/
theorem son_father_height_relationship :
  ∀ (r : HeightRelationship), r.isUncertain → r.type = RelationshipType.Correlation :=
by sorry

end NUMINAMATH_CALUDE_son_father_height_relationship_l1562_156286


namespace NUMINAMATH_CALUDE_dice_sum_symmetry_l1562_156279

def num_dice : ℕ := 8
def min_face : ℕ := 1
def max_face : ℕ := 6

def sum_symmetric (s : ℕ) : ℕ :=
  2 * ((num_dice * min_face + num_dice * max_face) / 2) - s

theorem dice_sum_symmetry :
  sum_symmetric 12 = 44 :=
by sorry

end NUMINAMATH_CALUDE_dice_sum_symmetry_l1562_156279


namespace NUMINAMATH_CALUDE_rectangle_y_value_l1562_156239

theorem rectangle_y_value (y : ℝ) : 
  y > 0 →  -- y is positive
  (5 - (-3)) * (y - 2) = 64 →  -- area of rectangle is 64
  y = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l1562_156239


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_ratio_l1562_156257

-- Define the Triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the similarity relation between triangles
def similar (t1 t2 : Triangle) : Prop := sorry

-- Define the similarity ratio between triangles
def similarityRatio (t1 t2 : Triangle) : ℝ := sorry

-- Define the perimeter of a triangle
def perimeter (t : Triangle) : ℝ := t.a + t.b + t.c

-- Theorem statement
theorem similar_triangles_perimeter_ratio 
  (ABC DEF : Triangle) 
  (h_similar : similar ABC DEF) 
  (h_ratio : similarityRatio ABC DEF = 1 / 2) : 
  perimeter ABC / perimeter DEF = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_ratio_l1562_156257


namespace NUMINAMATH_CALUDE_sandwich_combinations_l1562_156271

theorem sandwich_combinations :
  let meat_types : ℕ := 12
  let cheese_types : ℕ := 12
  let spread_types : ℕ := 5
  let meat_selection : ℕ := meat_types
  let cheese_selection : ℕ := cheese_types.choose 2
  let spread_selection : ℕ := spread_types
  meat_selection * cheese_selection * spread_selection = 3960 :=
by sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l1562_156271


namespace NUMINAMATH_CALUDE_bug_probability_l1562_156266

/-- Probability of the bug being at vertex A after n steps -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - P n)

/-- The probability of the bug being at vertex A after 7 steps is 182/729 -/
theorem bug_probability : P 7 = 182 / 729 := by
  sorry

end NUMINAMATH_CALUDE_bug_probability_l1562_156266


namespace NUMINAMATH_CALUDE_min_socks_for_eight_pairs_l1562_156209

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer :=
  (red : ℕ)
  (yellow : ℕ)
  (green : ℕ)
  (purple : ℕ)

/-- The minimum number of socks needed to guarantee at least n pairs -/
def minSocksForPairs (drawer : SockDrawer) (n : ℕ) : ℕ :=
  sorry

/-- The specific drawer configuration in the problem -/
def problemDrawer : SockDrawer :=
  { red := 50, yellow := 100, green := 70, purple := 30 }

theorem min_socks_for_eight_pairs :
  minSocksForPairs problemDrawer 8 = 28 :=
sorry

end NUMINAMATH_CALUDE_min_socks_for_eight_pairs_l1562_156209


namespace NUMINAMATH_CALUDE_equation_represents_hyperbola_l1562_156234

/-- Given the equation (x+y)^2 = x^2 + y^2 + 2x + 2y, prove it represents a hyperbola -/
theorem equation_represents_hyperbola (x y : ℝ) :
  (x + y)^2 = x^2 + y^2 + 2*x + 2*y ↔ (x - 1) * (y - 1) = 1 :=
by sorry

end NUMINAMATH_CALUDE_equation_represents_hyperbola_l1562_156234


namespace NUMINAMATH_CALUDE_apple_buying_problem_l1562_156261

/-- Proves that given the conditions of the apple-buying problem, each man bought 30 apples. -/
theorem apple_buying_problem (men women man_apples woman_apples total_apples : ℕ) 
  (h1 : men = 2)
  (h2 : women = 3)
  (h3 : man_apples + 20 = woman_apples)
  (h4 : men * man_apples + women * woman_apples = total_apples)
  (h5 : total_apples = 210) :
  man_apples = 30 := by
sorry

end NUMINAMATH_CALUDE_apple_buying_problem_l1562_156261


namespace NUMINAMATH_CALUDE_perpendicular_line_proof_l1562_156227

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-1, 3)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 2 * x + y - 1 = 0

-- Theorem statement
theorem perpendicular_line_proof :
  -- The perpendicular line passes through point P
  perpendicular_line point_P.1 point_P.2 ∧
  -- The perpendicular line is indeed perpendicular to the given line
  ∀ (x₁ y₁ x₂ y₂ : ℝ),
    given_line x₁ y₁ → given_line x₂ y₂ →
    perpendicular_line x₁ y₁ → perpendicular_line x₂ y₂ →
    (x₂ - x₁) * (x₂ - x₁) + (y₂ - y₁) * (y₂ - y₁) ≠ 0 →
    ((x₂ - x₁) * (1) + (y₂ - y₁) * (-2)) * ((x₂ - x₁) * (2) + (y₂ - y₁) * (1)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_proof_l1562_156227


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1562_156222

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2}

theorem complement_of_A_in_U :
  (U \ A) = {0, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1562_156222


namespace NUMINAMATH_CALUDE_tan_product_pi_ninths_l1562_156291

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_pi_ninths_l1562_156291


namespace NUMINAMATH_CALUDE_triangle_problem_l1562_156218

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  A > 0 ∧ B > 0 ∧ C > 0 →  -- Positive angles
  A + B + C = π →  -- Sum of angles in a triangle
  a * Real.cos B = 3 →
  b * Real.cos A = 1 →
  A - B = π / 6 →
  c = 4 ∧ B = π / 6 := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l1562_156218


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_dimensions_l1562_156225

/-- An isosceles trapezoid with legs intersecting at a right angle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- Length of the shorter base -/
  shorterBase : ℝ
  /-- Height of the trapezoid -/
  height : ℝ
  /-- Area of the trapezoid -/
  area : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The legs intersect at a right angle -/
  legsRightAngle : True
  /-- The area is calculated correctly -/
  areaEq : area = (longerBase + shorterBase) * height / 2

/-- Theorem about the dimensions of a specific isosceles trapezoid -/
theorem isosceles_trapezoid_dimensions (t : IsoscelesTrapezoid) 
  (h_area : t.area = 12)
  (h_height : t.height = 2) :
  t.longerBase = 8 ∧ t.shorterBase = 4 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_trapezoid_dimensions_l1562_156225


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1562_156255

theorem no_integer_solutions :
  ∀ x : ℤ, x^5 - 31*x + 2015 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1562_156255


namespace NUMINAMATH_CALUDE_exactly_five_ladybugs_l1562_156296

/-- Represents a ladybug with a specific number of spots -/
inductive Ladybug
  | sixSpots
  | fourSpots

/-- Represents a statement made by a ladybug -/
inductive Statement
  | allSame
  | totalThirty
  | totalTwentySix

/-- The meadow containing ladybugs -/
structure Meadow where
  ladybugs : List Ladybug

/-- Evaluates whether a statement is true for a given meadow -/
def isStatementTrue (m : Meadow) (s : Statement) : Bool :=
  match s with
  | Statement.allSame => sorry
  | Statement.totalThirty => sorry
  | Statement.totalTwentySix => sorry

/-- Counts the number of true statements in a list of statements for a given meadow -/
def countTrueStatements (m : Meadow) (statements : List Statement) : Nat :=
  statements.filter (isStatementTrue m) |>.length

/-- Theorem stating that there are exactly 5 ladybugs in the meadow -/
theorem exactly_five_ladybugs :
  ∃ (m : Meadow),
    m.ladybugs.length = 5 ∧
    (∀ l : Ladybug, l ∈ m.ladybugs → (l = Ladybug.sixSpots ∨ l = Ladybug.fourSpots)) ∧
    countTrueStatements m [Statement.allSame, Statement.totalThirty, Statement.totalTwentySix] = 1 :=
  sorry

end NUMINAMATH_CALUDE_exactly_five_ladybugs_l1562_156296


namespace NUMINAMATH_CALUDE_smaller_number_problem_l1562_156256

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 8) : 
  min x y = 5 := by
sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l1562_156256


namespace NUMINAMATH_CALUDE_product_coefficient_equality_l1562_156240

theorem product_coefficient_equality (m : ℝ) : 
  (∃ a b c d : ℝ, (x^2 - m*x + 2) * (2*x + 1) = a*x^3 + b*x^2 + b*x + d) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_product_coefficient_equality_l1562_156240


namespace NUMINAMATH_CALUDE_first_expedition_duration_l1562_156232

theorem first_expedition_duration (total_days : ℕ) 
  (h1 : total_days = 126) : ∃ (x : ℕ), 
  x * 7 + (x + 2) * 7 + 2 * (x + 2) * 7 = total_days ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_expedition_duration_l1562_156232


namespace NUMINAMATH_CALUDE_total_wax_needed_l1562_156235

def wax_already_has : ℕ := 28
def wax_still_needs : ℕ := 260

theorem total_wax_needed : wax_already_has + wax_still_needs = 288 := by
  sorry

end NUMINAMATH_CALUDE_total_wax_needed_l1562_156235


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l1562_156280

theorem sum_of_x_and_y (x y : ℤ) (h1 : x - y = 60) (h2 : x = 37) : x + y = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l1562_156280


namespace NUMINAMATH_CALUDE_parallel_lines_condition_l1562_156263

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x + 2 * y - 1 = 0
def l₂ (a x y : ℝ) : Prop := x + (a + 1) * y + 4 = 0

-- Define parallel lines
def parallel (f g : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), f x y ↔ g (k * x) (k * y)

-- State the theorem
theorem parallel_lines_condition (a : ℝ) :
  (a = 1 ↔ parallel (l₁ a) (l₂ a)) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_condition_l1562_156263


namespace NUMINAMATH_CALUDE_unique_solution_to_diophantine_equation_l1562_156215

theorem unique_solution_to_diophantine_equation :
  ∃! (a b c : ℕ+), 11^(a:ℕ) + 3^(b:ℕ) = (c:ℕ)^2 ∧ a = 4 ∧ b = 5 ∧ c = 122 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_diophantine_equation_l1562_156215


namespace NUMINAMATH_CALUDE_number_problem_l1562_156246

theorem number_problem (n : ℝ) : (0.4 * (3/5) * n = 36) → n = 150 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1562_156246


namespace NUMINAMATH_CALUDE_late_passengers_l1562_156259

theorem late_passengers (total : ℕ) (on_time : ℕ) (h1 : total = 14720) (h2 : on_time = 14507) :
  total - on_time = 213 := by
  sorry

end NUMINAMATH_CALUDE_late_passengers_l1562_156259


namespace NUMINAMATH_CALUDE_max_value_when_a_is_one_a_values_when_max_is_two_l1562_156210

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^2 + 2*a*x + 1 - a

-- Part 1
theorem max_value_when_a_is_one :
  ∃ (max : ℝ), (∀ x, f 1 x ≤ max) ∧ (∃ x, f 1 x = max) ∧ max = 1 := by sorry

-- Part 2
theorem a_values_when_max_is_two :
  (∃ (max : ℝ), (∀ x ∈ Set.Icc 0 1, f a x ≤ max) ∧ 
   (∃ x ∈ Set.Icc 0 1, f a x = max) ∧ max = 2) → (a = -1 ∨ a = 2) := by sorry

end NUMINAMATH_CALUDE_max_value_when_a_is_one_a_values_when_max_is_two_l1562_156210


namespace NUMINAMATH_CALUDE_prob_two_red_shoes_l1562_156268

/-- The probability of drawing two red shoes from a set of 4 red shoes and 4 green shoes -/
theorem prob_two_red_shoes : 
  let total_shoes : ℕ := 4 + 4
  let red_shoes : ℕ := 4
  let draw_count : ℕ := 2
  let total_ways := Nat.choose total_shoes draw_count
  let red_ways := Nat.choose red_shoes draw_count
  (red_ways : ℚ) / total_ways = 3 / 14 := by sorry

end NUMINAMATH_CALUDE_prob_two_red_shoes_l1562_156268


namespace NUMINAMATH_CALUDE_solve_shelves_problem_l1562_156216

def shelves_problem (initial_stock : ℕ) (books_sold : ℕ) (books_per_shelf : ℕ) : Prop :=
  let remaining_books := initial_stock - books_sold
  remaining_books / books_per_shelf = 5

theorem solve_shelves_problem :
  shelves_problem 40 20 4 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_shelves_problem_l1562_156216


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l1562_156236

theorem simplify_sqrt_expression (m : ℝ) (h : m < 1) : 
  Real.sqrt (m^2 - 2*m + 1) = 1 - m := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l1562_156236


namespace NUMINAMATH_CALUDE_remaining_amount_after_purchase_l1562_156264

def lollipop_price : ℚ := 1.5
def gummy_pack_price : ℚ := 2
def chips_price : ℚ := 1.25
def chocolate_price : ℚ := 1.75
def discount_rate : ℚ := 0.1
def tax_rate : ℚ := 0.05
def initial_amount : ℚ := 25

def total_cost : ℚ := 4 * lollipop_price + 2 * gummy_pack_price + 3 * chips_price + chocolate_price

def discounted_cost : ℚ := total_cost * (1 - discount_rate)

def final_cost : ℚ := discounted_cost * (1 + tax_rate)

theorem remaining_amount_after_purchase : 
  initial_amount - final_cost = 10.35 := by sorry

end NUMINAMATH_CALUDE_remaining_amount_after_purchase_l1562_156264


namespace NUMINAMATH_CALUDE_smallest_square_area_l1562_156211

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side * side

/-- Checks if two rectangles can fit in a square of given side length without overlapping -/
def can_fit_in_square (r1 r2 : Rectangle) (side : ℝ) : Prop :=
  (min r1.width r1.height + min r2.width r2.height ≤ side) ∧
  (max r1.width r1.height + max r2.width r2.height ≤ side)

theorem smallest_square_area (r1 r2 : Rectangle) : 
  r1.width = 3 ∧ r1.height = 4 ∧ r2.width = 4 ∧ r2.height = 5 →
  ∃ (side : ℝ), 
    can_fit_in_square r1 r2 side ∧ 
    square_area side = 49 ∧
    ∀ (s : ℝ), can_fit_in_square r1 r2 s → square_area s ≥ 49 := by
  sorry

end NUMINAMATH_CALUDE_smallest_square_area_l1562_156211


namespace NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1562_156217

/-- The greatest possible distance between the centers of two circles in a rectangle -/
theorem greatest_distance_between_circle_centers 
  (rectangle_width : ℝ) 
  (rectangle_height : ℝ) 
  (circle_diameter : ℝ) 
  (h1 : rectangle_width = 20) 
  (h2 : rectangle_height = 15) 
  (h3 : circle_diameter = 10) :
  ∃ (d : ℝ), d = 5 * Real.sqrt 5 ∧ 
  ∀ (d' : ℝ), d' ≤ d ∧ 
  ∃ (x1 y1 x2 y2 : ℝ), 
    0 ≤ x1 ∧ x1 ≤ rectangle_width ∧
    0 ≤ y1 ∧ y1 ≤ rectangle_height ∧
    0 ≤ x2 ∧ x2 ≤ rectangle_width ∧
    0 ≤ y2 ∧ y2 ≤ rectangle_height ∧
    circle_diameter / 2 ≤ x1 ∧ x1 ≤ rectangle_width - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ y1 ∧ y1 ≤ rectangle_height - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ x2 ∧ x2 ≤ rectangle_width - circle_diameter / 2 ∧
    circle_diameter / 2 ≤ y2 ∧ y2 ≤ rectangle_height - circle_diameter / 2 ∧
    d' = Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2) :=
by sorry

end NUMINAMATH_CALUDE_greatest_distance_between_circle_centers_l1562_156217


namespace NUMINAMATH_CALUDE_compute_fraction_power_l1562_156270

theorem compute_fraction_power : 8 * (1 / 3)^4 = 8 / 81 := by
  sorry

end NUMINAMATH_CALUDE_compute_fraction_power_l1562_156270


namespace NUMINAMATH_CALUDE_log_inequality_equiv_solution_set_l1562_156224

def log_inequality (x y : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ x ≠ 1 ∧ x ≠ y ∧ Real.log y / Real.log x ≥ (Real.log x + Real.log y) / (Real.log x - Real.log y)

def solution_set (x y : ℝ) : Prop :=
  (0 < x ∧ x < 1 ∧ 0 < y ∧ y < x) ∨ (x > 1 ∧ y > x)

theorem log_inequality_equiv_solution_set :
  ∀ x y : ℝ, log_inequality x y ↔ solution_set x y :=
sorry

end NUMINAMATH_CALUDE_log_inequality_equiv_solution_set_l1562_156224


namespace NUMINAMATH_CALUDE_car_trip_mpg_l1562_156248

/-- Represents the miles per gallon for a car trip -/
structure MPG where
  ab : ℝ  -- Miles per gallon from A to B
  bc : ℝ  -- Miles per gallon from B to C
  total : ℝ  -- Overall miles per gallon for the entire trip

/-- Represents the distance for a car trip -/
structure Distance where
  ab : ℝ  -- Distance from A to B
  bc : ℝ  -- Distance from B to C

theorem car_trip_mpg (d : Distance) (mpg : MPG) :
  d.bc = d.ab / 2 →  -- Distance from B to C is half of A to B
  mpg.ab = 40 →  -- MPG from A to B is 40
  mpg.total = 300 / 7 →  -- Overall MPG is 300/7 (approx. 42.857142857142854)
  d.ab > 0 →  -- Distance from A to B is positive
  mpg.bc = 100 / 9 :=  -- MPG from B to C is 100/9 (approx. 11.11)
by sorry

end NUMINAMATH_CALUDE_car_trip_mpg_l1562_156248


namespace NUMINAMATH_CALUDE_min_max_x_sum_l1562_156290

theorem min_max_x_sum (x y z : ℝ) 
  (sum_eq : x + y + z = 6) 
  (sum_sq_eq : x^2 + y^2 + z^2 = 10) : 
  ∃ (x_min x_max : ℝ), 
    (∀ x', ∃ y' z', x' + y' + z' = 6 ∧ x'^2 + y'^2 + z'^2 = 10 → x_min ≤ x') ∧
    (∀ x', ∃ y' z', x' + y' + z' = 6 ∧ x'^2 + y'^2 + z'^2 = 10 → x' ≤ x_max) ∧
    x_min = 8/3 ∧ 
    x_max = 2 ∧ 
    x_min + x_max = 14/3 := by
  sorry

end NUMINAMATH_CALUDE_min_max_x_sum_l1562_156290


namespace NUMINAMATH_CALUDE_min_value_of_f_l1562_156241

/-- The quadratic function f(x) = 8x^2 - 32x + 2023 -/
def f (x : ℝ) : ℝ := 8 * x^2 - 32 * x + 2023

/-- Theorem stating that the minimum value of f(x) is 1991 -/
theorem min_value_of_f :
  ∃ (m : ℝ), m = 1991 ∧ ∀ (x : ℝ), f x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1562_156241


namespace NUMINAMATH_CALUDE_investment_calculation_l1562_156230

theorem investment_calculation (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ) (total_dividend : ℝ) :
  face_value = 100 →
  premium_rate = 0.2 →
  dividend_rate = 0.07 →
  total_dividend = 840.0000000000001 →
  (total_dividend / (face_value * dividend_rate)) * (face_value * (1 + premium_rate)) = 14400 :=
by sorry

end NUMINAMATH_CALUDE_investment_calculation_l1562_156230


namespace NUMINAMATH_CALUDE_angle_A_is_60_degrees_max_area_is_5_exists_triangle_with_max_area_l1562_156292

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.a = t.b + t.c - 2

theorem angle_A_is_60_degrees (t : Triangle) 
  (h : triangle_condition t) : t.A = 60 := by sorry

theorem max_area_is_5 (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.a = 2) : 
  ∀ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A → s ≤ 5 := by sorry

theorem exists_triangle_with_max_area (t : Triangle) 
  (h1 : triangle_condition t) 
  (h2 : t.a = 2) : 
  ∃ (s : ℝ), s = (1/2) * t.b * t.c * Real.sin t.A ∧ s = 5 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_60_degrees_max_area_is_5_exists_triangle_with_max_area_l1562_156292


namespace NUMINAMATH_CALUDE_smallest_abs_z_l1562_156212

theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 15) + Complex.abs (z + 3*I) = 20) :
  ∃ (min_abs : ℝ), min_abs = 2.25 ∧ ∀ w : ℂ, Complex.abs (w - 15) + Complex.abs (w + 3*I) = 20 → Complex.abs w ≥ min_abs :=
sorry

end NUMINAMATH_CALUDE_smallest_abs_z_l1562_156212


namespace NUMINAMATH_CALUDE_unique_solution_l1562_156243

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 10000 ∧ n < 100000 ∧ 
  (∃ (a b c d e : ℕ), 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e)

theorem unique_solution :
  ∃! (n : ℕ), is_valid_number n ∧ n * 3 = 100000 * n + n + 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l1562_156243


namespace NUMINAMATH_CALUDE_irreducibility_of_polynomial_l1562_156206

theorem irreducibility_of_polynomial :
  ¬∃ (p q : Polynomial ℤ), (Polynomial.degree p ≥ 1) ∧ (Polynomial.degree q ≥ 1) ∧ (p * q = X^5 + 2*X + 1) :=
by sorry

end NUMINAMATH_CALUDE_irreducibility_of_polynomial_l1562_156206


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l1562_156253

/-- The equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def is_circle_equation (h k r : ℝ) (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The given equation -/
def given_equation (k : ℝ) (x y : ℝ) : ℝ :=
  x^2 + 14*x + y^2 + 8*y - k

theorem circle_equation_k_value :
  ∃! k : ℝ, is_circle_equation (-7) (-4) 5 (given_equation k) ∧ k = -40 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l1562_156253


namespace NUMINAMATH_CALUDE_souvenir_optimal_price_l1562_156251

/-- Represents the optimization problem for a souvenir's selling price --/
def SouvenirOptimization (a : ℝ) : Prop :=
  ∃ (x : ℝ),
    0 < x ∧ x < 1 ∧
    (∀ (z : ℝ), 0 < z ∧ z < 1 →
      5 * a * (1 + 4 * x - x^2 - 4 * x^3) ≥ 5 * a * (1 + 4 * z - z^2 - 4 * z^3)) ∧
    20 * (1 + x) = 30

theorem souvenir_optimal_price (a : ℝ) (h : a > 0) : SouvenirOptimization a := by
  sorry

end NUMINAMATH_CALUDE_souvenir_optimal_price_l1562_156251


namespace NUMINAMATH_CALUDE_triangle_sum_property_l1562_156233

theorem triangle_sum_property : ∃ (a b c d e f : ℤ),
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  a = b + c + d ∧
  b = a + c + e ∧
  c = a + b + f :=
by sorry

end NUMINAMATH_CALUDE_triangle_sum_property_l1562_156233


namespace NUMINAMATH_CALUDE_square_on_circle_radius_l1562_156252

theorem square_on_circle_radius (S : ℝ) (x : ℝ) (R : ℝ) : 
  S = 256 →  -- Area of the square
  x^2 = S →  -- Side length of the square
  (x - R)^2 = R^2 - (x/2)^2 →  -- Pythagorean theorem relation
  R = 10 := by
  sorry

end NUMINAMATH_CALUDE_square_on_circle_radius_l1562_156252


namespace NUMINAMATH_CALUDE_point_not_in_second_quadrant_l1562_156237

theorem point_not_in_second_quadrant (a : ℝ) :
  ¬(a < 0 ∧ 2*a - 1 > 0) :=
by sorry

end NUMINAMATH_CALUDE_point_not_in_second_quadrant_l1562_156237


namespace NUMINAMATH_CALUDE_correct_testing_schemes_l1562_156277

/-- The number of genuine products -/
def genuine_products : ℕ := 5

/-- The number of defective products -/
def defective_products : ℕ := 4

/-- The position at which the last defective product is detected -/
def last_defective_position : ℕ := 6

/-- The number of ways to arrange products such that the last defective product
    is at the specified position and all defective products are included -/
def testing_schemes : ℕ := defective_products * (genuine_products.choose 2) * (last_defective_position - 1).factorial

theorem correct_testing_schemes :
  testing_schemes = 4800 := by sorry

end NUMINAMATH_CALUDE_correct_testing_schemes_l1562_156277


namespace NUMINAMATH_CALUDE_unique_integer_solution_to_equation_l1562_156260

theorem unique_integer_solution_to_equation :
  ∀ x y : ℤ, x^4 + y^4 = 3*x^3*y → x = 0 ∧ y = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_solution_to_equation_l1562_156260


namespace NUMINAMATH_CALUDE_equation_solutions_l1562_156208

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 - 2*x = 3 ↔ x = -1 ∨ x = 3) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1562_156208


namespace NUMINAMATH_CALUDE_seconds_in_week_scientific_correct_l1562_156226

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number of seconds in a week -/
def seconds_in_week : ℕ := 604800

/-- The scientific notation representation of the number of seconds in a week -/
def seconds_in_week_scientific : ScientificNotation :=
  { coefficient := 6.048
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem seconds_in_week_scientific_correct :
  (seconds_in_week_scientific.coefficient * (10 : ℝ) ^ seconds_in_week_scientific.exponent) = seconds_in_week := by
  sorry

end NUMINAMATH_CALUDE_seconds_in_week_scientific_correct_l1562_156226


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1562_156213

theorem sufficient_not_necessary :
  (∀ x y : ℝ, x > 0 ∧ y > 0 → y / x + x / y ≥ 2) ∧
  (∃ x y : ℝ, y / x + x / y ≥ 2 ∧ ¬(x > 0 ∧ y > 0)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1562_156213


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l1562_156274

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation for a line in a plane
variable (subset_line_plane : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_parallel_planes
  (l m : Line) (α β : Plane)
  (h1 : perp_line_plane l α)
  (h2 : subset_line_plane m β)
  (h3 : parallel_planes α β) :
  perp_lines l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_parallel_planes_l1562_156274


namespace NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l1562_156299

/-- The length of a diagonal in a quadrilateral with given offsets and area -/
theorem diagonal_length_of_quadrilateral (offset1 offset2 area : ℝ) 
  (h1 : offset1 = 9)
  (h2 : offset2 = 6)
  (h3 : area = 195) :
  ∃ d : ℝ, d = 26 ∧ (1/2 * d * offset1) + (1/2 * d * offset2) = area :=
by sorry

end NUMINAMATH_CALUDE_diagonal_length_of_quadrilateral_l1562_156299


namespace NUMINAMATH_CALUDE_increasing_f_implies_m_bound_l1562_156288

/-- A cubic function parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * m * x^2 + 6 * x

/-- The derivative of f with respect to x -/
def f_deriv (m : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * m * x + 6

theorem increasing_f_implies_m_bound :
  (∀ x > 2, ∀ y > x, f m y > f m x) →
  m ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_m_bound_l1562_156288


namespace NUMINAMATH_CALUDE_pure_imaginary_value_l1562_156220

theorem pure_imaginary_value (x : ℝ) : 
  (((x^2 - 1) : ℂ) + (x + 1) * Complex.I).re = 0 ∧ 
  (((x^2 - 1) : ℂ) + (x + 1) * Complex.I).im ≠ 0 → 
  x = 1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_value_l1562_156220


namespace NUMINAMATH_CALUDE_all_graphs_different_l1562_156221

-- Define the three equations
def eq_I (x y : ℝ) : Prop := y = x - 3
def eq_II (x y : ℝ) : Prop := y = (x^2 - 9) / (x + 3)
def eq_III (x y : ℝ) : Prop := (x + 3) * y = x^2 - 9

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem stating that all graphs are different
theorem all_graphs_different :
  ¬(same_graph eq_I eq_II) ∧ 
  ¬(same_graph eq_I eq_III) ∧ 
  ¬(same_graph eq_II eq_III) :=
sorry

end NUMINAMATH_CALUDE_all_graphs_different_l1562_156221


namespace NUMINAMATH_CALUDE_arithmetic_progression_ratio_l1562_156200

/-- The sum of the first n terms of an arithmetic progression -/
def arithmeticSum (a d : ℚ) (n : ℕ) : ℚ := n / 2 * (2 * a + (n - 1) * d)

/-- Theorem: In an arithmetic progression where the sum of the first 15 terms
    is three times the sum of the first 8 terms, the ratio of the first term
    to the common difference is 7:3 -/
theorem arithmetic_progression_ratio (a d : ℚ) :
  arithmeticSum a d 15 = 3 * arithmeticSum a d 8 → a / d = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_ratio_l1562_156200


namespace NUMINAMATH_CALUDE_parallel_lines_slope_parallel_line_k_value_l1562_156250

/-- A line through two points is parallel to another line if and only if their slopes are equal -/
theorem parallel_lines_slope (x1 y1 x2 y2 a b c : ℝ) :
  (∀ x y, a * x + b * y = c → y = (-a/b) * x + c/b) →
  (y2 - y1) / (x2 - x1) = -a/b ↔ 
  (∀ x y, y - y1 = ((y2 - y1) / (x2 - x1)) * (x - x1) → a * x + b * y = c) :=
sorry

/-- The value of k for which the line through (4, 3) and (k, -5) is parallel to 3x - 2y = 6 -/
theorem parallel_line_k_value : 
  (∃! k : ℝ, ((-5) - 3) / (k - 4) = (-3) / (-2) ∧ 
              ∀ x y : ℝ, y - 3 = ((-5) - 3) / (k - 4) * (x - 4) → 
                3 * x + (-2) * y = 6) ∧
  (∃! k : ℝ, ((-5) - 3) / (k - 4) = (-3) / (-2) ∧ 
              ∀ x y : ℝ, y - 3 = ((-5) - 3) / (k - 4) * (x - 4) → 
                3 * x + (-2) * y = 6) → k = -4/3 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_slope_parallel_line_k_value_l1562_156250


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1562_156229

theorem tan_alpha_value (α : Real) (h : Real.tan (α + π/4) = 2) : Real.tan α = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1562_156229


namespace NUMINAMATH_CALUDE_new_oranges_added_l1562_156293

theorem new_oranges_added (initial : ℕ) (thrown_away : ℕ) (final : ℕ) : 
  initial = 50 → thrown_away = 40 → final = 34 → 
  final - (initial - thrown_away) = 24 := by
  sorry

end NUMINAMATH_CALUDE_new_oranges_added_l1562_156293


namespace NUMINAMATH_CALUDE_even_Z_tetrominoes_l1562_156282

/-- Represents a lattice polygon -/
structure LatticePolygon where
  -- Add necessary fields here
  mk :: -- Constructor

/-- Represents an S-tetromino -/
inductive STetromino

/-- Represents a Z-tetromino -/
inductive ZTetromino

/-- Represents either an S-tetromino or a Z-tetromino -/
inductive Tetromino
  | S : STetromino → Tetromino
  | Z : ZTetromino → Tetromino

/-- Predicate indicating if a lattice polygon can be tiled with S-tetrominoes -/
def canBeTiledWithS (P : LatticePolygon) : Prop := sorry

/-- Represents a tiling of a lattice polygon using S- and Z-tetrominoes -/
def Tiling (P : LatticePolygon) := List Tetromino

/-- Counts the number of Z-tetrominoes in a tiling -/
def countZTetrominoes (tiling : Tiling P) : Nat := sorry

/-- Main theorem: For any lattice polygon that can be tiled with S-tetrominoes,
    any tiling using S- and Z-tetrominoes will contain an even number of Z-tetrominoes -/
theorem even_Z_tetrominoes (P : LatticePolygon) (h : canBeTiledWithS P) :
  ∀ (tiling : Tiling P), Even (countZTetrominoes tiling) := by
  sorry

end NUMINAMATH_CALUDE_even_Z_tetrominoes_l1562_156282


namespace NUMINAMATH_CALUDE_division_problem_l1562_156258

theorem division_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 12 → 
  divisor = 17 → 
  remainder = 7 → 
  dividend = divisor * quotient + remainder →
  quotient = 0 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1562_156258


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l1562_156294

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 2*a - 2 = 0) → 
  (b^3 - 2*b - 2 = 0) → 
  (c^3 - 2*c - 2 = 0) → 
  a*(b - c)^2 + b*(c - a)^2 + c*(a - b)^2 = -18 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l1562_156294


namespace NUMINAMATH_CALUDE_find_S_l1562_156276

def f : ℕ → ℕ
  | 0 => 0
  | n + 1 => f n + 3

theorem find_S (S : ℕ) (h : 2 * f S = 3996) : S = 666 := by
  sorry

end NUMINAMATH_CALUDE_find_S_l1562_156276


namespace NUMINAMATH_CALUDE_point_P_coordinates_l1562_156201

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of point P -/
def P (m : ℝ) : Point :=
  { x := 3 * m - 6, y := m + 1 }

/-- Definition of point A -/
def A : Point :=
  { x := 1, y := -2 }

/-- A point lies on the y-axis if its x-coordinate is 0 -/
def lies_on_y_axis (p : Point) : Prop :=
  p.x = 0

/-- Two points form a line parallel to the x-axis if they have the same y-coordinate -/
def parallel_to_x_axis (p1 p2 : Point) : Prop :=
  p1.y = p2.y

theorem point_P_coordinates :
  (∃ m : ℝ, lies_on_y_axis (P m) ∧ P m = { x := 0, y := 3 }) ∧
  (∃ m : ℝ, parallel_to_x_axis (P m) A ∧ P m = { x := -15, y := -2 }) :=
sorry

end NUMINAMATH_CALUDE_point_P_coordinates_l1562_156201


namespace NUMINAMATH_CALUDE_cos_angle_AMB_formula_l1562_156295

/-- Regular square pyramid with vertex A and square base BCDE -/
structure RegularSquarePyramid where
  s : ℝ  -- side length of the base
  h : ℝ  -- height of the pyramid
  l : ℝ  -- slant height of the pyramid

/-- Point M is the midpoint of diagonal BD -/
def midpoint_M (p : RegularSquarePyramid) : ℝ × ℝ × ℝ := sorry

/-- Angle AMB in the regular square pyramid -/
def angle_AMB (p : RegularSquarePyramid) : ℝ := sorry

theorem cos_angle_AMB_formula (p : RegularSquarePyramid) :
  Real.cos (angle_AMB p) = (p.l^2 + p.h^2) / (2 * p.l * Real.sqrt (p.h^2 + p.s^2 / 2)) :=
sorry

end NUMINAMATH_CALUDE_cos_angle_AMB_formula_l1562_156295


namespace NUMINAMATH_CALUDE_remainder_46_pow_925_mod_21_l1562_156254

theorem remainder_46_pow_925_mod_21 : 46^925 % 21 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_46_pow_925_mod_21_l1562_156254


namespace NUMINAMATH_CALUDE_trigonometric_calculations_l1562_156247

theorem trigonometric_calculations :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + Real.sin (60 * π / 180) + (Real.tan (60 * π / 180))^2 = 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_calculations_l1562_156247


namespace NUMINAMATH_CALUDE_max_students_l1562_156223

theorem max_students (n : ℕ) : n < 100 ∧ n % 9 = 4 ∧ n % 7 = 3 → n ≤ 94 := by
  sorry

end NUMINAMATH_CALUDE_max_students_l1562_156223


namespace NUMINAMATH_CALUDE_cos_beta_value_l1562_156249

theorem cos_beta_value (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan α = 2) (h4 : Real.sin (α + β) = Real.sqrt 2 / 2) :
  Real.cos β = Real.sqrt 10 / 10 := by
sorry

end NUMINAMATH_CALUDE_cos_beta_value_l1562_156249


namespace NUMINAMATH_CALUDE_complex_square_equality_l1562_156205

theorem complex_square_equality (a b : ℝ) : 
  (a + Complex.I = 2 - b * Complex.I) → (a + b * Complex.I)^2 = 3 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_square_equality_l1562_156205


namespace NUMINAMATH_CALUDE_bike_ride_time_l1562_156273

/-- Represents the problem of calculating the time to cover a highway stretch on a bike --/
theorem bike_ride_time (highway_length : Real) (highway_width : Real) (bike_speed : Real) :
  highway_length = 2 → -- 2 miles
  highway_width = 60 / 5280 → -- 60 feet converted to miles
  bike_speed = 6 → -- 6 miles per hour
  (π * highway_length) / bike_speed = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_bike_ride_time_l1562_156273


namespace NUMINAMATH_CALUDE_inverse_true_converse_false_l1562_156207

-- Define the universe of shapes
variable (Shape : Type)

-- Define predicates for being a circle and having corners
variable (is_circle : Shape → Prop)
variable (has_corners : Shape → Prop)

-- Given statement
axiom circle_no_corners : ∀ s : Shape, is_circle s → ¬(has_corners s)

-- Theorem to prove
theorem inverse_true_converse_false :
  (∀ s : Shape, ¬(is_circle s) → has_corners s) ∧
  ¬(∀ s : Shape, ¬(has_corners s) → is_circle s) :=
sorry

end NUMINAMATH_CALUDE_inverse_true_converse_false_l1562_156207


namespace NUMINAMATH_CALUDE_only_courses_form_set_l1562_156298

-- Define a type for the universe of discourse
def Universe : Type := Unit

-- Define predicates for each option
def likes_airplanes (x : Universe) : Prop := sorry
def is_sufficiently_small_negative (x : ℝ) : Prop := sorry
def has_poor_eyesight (x : Universe) : Prop := sorry
def is_course_of_class_on_day (x : Universe) : Prop := sorry

-- Define what it means for a predicate to form a well-defined set
def forms_well_defined_set {α : Type} (P : α → Prop) : Prop := sorry

-- State the theorem
theorem only_courses_form_set :
  ¬(forms_well_defined_set likes_airplanes) ∧
  ¬(forms_well_defined_set is_sufficiently_small_negative) ∧
  ¬(forms_well_defined_set has_poor_eyesight) ∧
  (forms_well_defined_set is_course_of_class_on_day) :=
sorry

end NUMINAMATH_CALUDE_only_courses_form_set_l1562_156298


namespace NUMINAMATH_CALUDE_quadratic_decreasing_interval_l1562_156267

-- Define the quadratic function
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_decreasing_interval (b c : ℝ) :
  (f b c 1 = 0) → (f b c 3 = 0) →
  ∃ (x : ℝ), ∀ (y : ℝ), y < x → (∀ (z : ℝ), y < z → f b c y > f b c z) ∧ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_interval_l1562_156267


namespace NUMINAMATH_CALUDE_ethans_work_hours_l1562_156204

/-- Proves that Ethan works 8 hours per day given his earnings and work schedule --/
theorem ethans_work_hours 
  (hourly_rate : ℝ) 
  (days_per_week : ℕ) 
  (total_earnings : ℝ) 
  (total_weeks : ℕ) 
  (h1 : hourly_rate = 18)
  (h2 : days_per_week = 5)
  (h3 : total_earnings = 3600)
  (h4 : total_weeks = 5) :
  (total_earnings / total_weeks) / days_per_week / hourly_rate = 8 := by
  sorry

#check ethans_work_hours

end NUMINAMATH_CALUDE_ethans_work_hours_l1562_156204


namespace NUMINAMATH_CALUDE_cubic_odd_and_increasing_l1562_156203

def f (x : ℝ) : ℝ := x^3

theorem cubic_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_odd_and_increasing_l1562_156203


namespace NUMINAMATH_CALUDE_sum_of_solution_l1562_156285

theorem sum_of_solution (a b : ℝ) : 
  3 * a + 7 * b = 1977 → 
  5 * a + b = 2007 → 
  a + b = 498 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solution_l1562_156285


namespace NUMINAMATH_CALUDE_digit_150_is_5_l1562_156242

/-- The decimal representation of 5/13 as a list of digits -/
def decimal_rep_5_13 : List Nat := [3, 8, 4, 6, 1, 5]

/-- The length of the repeating sequence in the decimal representation of 5/13 -/
def repeat_length : Nat := 6

/-- The 150th digit after the decimal point in the decimal representation of 5/13 -/
def digit_150 : Nat :=
  decimal_rep_5_13[(150 - 1) % repeat_length]

theorem digit_150_is_5 : digit_150 = 5 := by sorry

end NUMINAMATH_CALUDE_digit_150_is_5_l1562_156242


namespace NUMINAMATH_CALUDE_expression_evaluation_l1562_156275

theorem expression_evaluation (a b c : ℝ) 
  (h1 : c = b - 11)
  (h2 : b = a + 3)
  (h3 : a = 5)
  (h4 : a + 1 ≠ 0)
  (h5 : b - 3 ≠ 0)
  (h6 : c + 7 ≠ 0) :
  ((a + 3) / (a + 1)) * ((b - 2) / (b - 3)) * ((c + 9) / (c + 7)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1562_156275


namespace NUMINAMATH_CALUDE_square_fraction_is_perfect_square_l1562_156272

theorem square_fraction_is_perfect_square (a b : ℕ+) 
  (h : ∃ k : ℕ, (a + b)^2 = k * (4 * a * b + 1)) : 
  ∃ n : ℕ, (a + b)^2 / (4 * a * b + 1) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_square_fraction_is_perfect_square_l1562_156272


namespace NUMINAMATH_CALUDE_article_price_reduction_l1562_156287

theorem article_price_reduction (reduced_price : ℝ) (reduction_percentage : ℝ) (original_price : ℝ) : 
  reduced_price = 608 ∧ 
  reduction_percentage = 24 ∧ 
  reduced_price = original_price * (1 - reduction_percentage / 100) → 
  original_price = 800 := by
  sorry

end NUMINAMATH_CALUDE_article_price_reduction_l1562_156287


namespace NUMINAMATH_CALUDE_max_value_theorem_l1562_156202

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (2*x*y - 1)^2 = (5*y + 2)*(y - 2)) : 
  x + 1/(2*y) ≤ 3/2 * Real.sqrt 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1562_156202


namespace NUMINAMATH_CALUDE_chicken_price_per_pound_l1562_156245

/-- Given John's food order for a restaurant, prove the price per pound of chicken --/
theorem chicken_price_per_pound (beef_quantity : ℕ) (beef_price : ℚ) 
  (total_cost : ℚ) (chicken_quantity : ℕ) (chicken_price : ℚ) : chicken_price = 3 :=
by
  have h1 : beef_quantity = 1000 := by sorry
  have h2 : beef_price = 8 := by sorry
  have h3 : chicken_quantity = 2 * beef_quantity := by sorry
  have h4 : total_cost = 14000 := by sorry
  have h5 : total_cost = beef_quantity * beef_price + chicken_quantity * chicken_price := by sorry
  sorry

end NUMINAMATH_CALUDE_chicken_price_per_pound_l1562_156245


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1562_156281

theorem product_of_three_numbers (x y z n : ℝ) : 
  x + y + z = 150 ∧ 
  x ≤ y ∧ x ≤ z ∧ 
  z ≤ y ∧
  7 * x = n ∧ 
  y - 10 = n ∧ 
  z + 10 = n → 
  x * y * z = 48000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1562_156281


namespace NUMINAMATH_CALUDE_eight_lines_form_784_parallelograms_intersecting_parallel_lines_theorem_l1562_156265

/-- The number of parallelograms formed by two sets of intersecting parallel lines -/
def parallelogramsCount (n m : ℕ) : ℕ := (n.choose 2) * (m.choose 2)

/-- Theorem stating that 8 lines in each set form 784 parallelograms -/
theorem eight_lines_form_784_parallelograms (n : ℕ) :
  parallelogramsCount n 8 = 784 → n = 8 := by
  sorry

/-- Main theorem proving that given 8 lines in one set and 784 parallelograms, 
    the other set must have 8 lines -/
theorem intersecting_parallel_lines_theorem :
  ∃ (n : ℕ), parallelogramsCount n 8 = 784 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_lines_form_784_parallelograms_intersecting_parallel_lines_theorem_l1562_156265


namespace NUMINAMATH_CALUDE_matrix_product_sum_l1562_156228

def A (y : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; y, 4]
def B (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![x, 6; 7, 8]

theorem matrix_product_sum (x y : ℝ) :
  A y * B x = !![19, 22; 43, 50] →
  x + y = 8 := by sorry

end NUMINAMATH_CALUDE_matrix_product_sum_l1562_156228
