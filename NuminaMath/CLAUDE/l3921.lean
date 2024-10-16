import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_expression_simplification_l3921_392100

theorem trigonometric_expression_simplification :
  let original_expression := (Real.sin (20 * π / 180) + Real.sin (40 * π / 180) + 
                              Real.sin (60 * π / 180) + Real.sin (80 * π / 180)) / 
                             (Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * 
                              Real.cos (30 * π / 180) * Real.cos (40 * π / 180))
  let simplified_expression := (4 * Real.sin (50 * π / 180)) / 
                               (Real.cos (30 * π / 180) * Real.cos (40 * π / 180))
  original_expression = simplified_expression := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_simplification_l3921_392100


namespace NUMINAMATH_CALUDE_current_velocity_l3921_392147

-- Define the rowing speeds
def downstream_speed (v c : ℝ) : ℝ := v + c
def upstream_speed (v c : ℝ) : ℝ := v - c

-- Define the conditions of the problem
def downstream_distance : ℝ := 32
def upstream_distance : ℝ := 14
def trip_time : ℝ := 6

-- Theorem statement
theorem current_velocity :
  ∃ (v c : ℝ),
    downstream_speed v c * trip_time = downstream_distance ∧
    upstream_speed v c * trip_time = upstream_distance ∧
    c = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_current_velocity_l3921_392147


namespace NUMINAMATH_CALUDE_rope_cut_theorem_l3921_392165

/-- Given a rope of 60 meters cut into two pieces, where the longer piece is twice
    the length of the shorter piece, prove that the length of the shorter piece is 20 meters. -/
theorem rope_cut_theorem (total_length : ℝ) (short_piece : ℝ) (long_piece : ℝ) : 
  total_length = 60 →
  long_piece = 2 * short_piece →
  total_length = short_piece + long_piece →
  short_piece = 20 := by
  sorry

end NUMINAMATH_CALUDE_rope_cut_theorem_l3921_392165


namespace NUMINAMATH_CALUDE_remaining_balloons_l3921_392167

def initial_balloons : ℕ := 30
def given_balloons : ℕ := 16

theorem remaining_balloons : initial_balloons - given_balloons = 14 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l3921_392167


namespace NUMINAMATH_CALUDE_circumcircle_equation_l3921_392118

-- Define the vertices of the triangle
def A : ℝ × ℝ := (-1, 5)
def B : ℝ × ℝ := (5, 5)
def C : ℝ × ℝ := (6, -2)

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Theorem statement
theorem circumcircle_equation :
  (circle_equation A.1 A.2) ∧
  (circle_equation B.1 B.2) ∧
  (circle_equation C.1 C.2) ∧
  (∀ (x y : ℝ), circle_equation x y → 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_circumcircle_equation_l3921_392118


namespace NUMINAMATH_CALUDE_namjoon_cookies_l3921_392152

/-- The number of cookies Namjoon had initially -/
def initial_cookies : ℕ := 24

/-- The number of cookies Namjoon ate -/
def eaten_cookies : ℕ := 8

/-- The number of cookies Namjoon gave to Hoseok -/
def given_cookies : ℕ := 7

/-- The number of cookies left after eating and giving away -/
def remaining_cookies : ℕ := 9

theorem namjoon_cookies : 
  initial_cookies - eaten_cookies - given_cookies = remaining_cookies :=
by sorry

end NUMINAMATH_CALUDE_namjoon_cookies_l3921_392152


namespace NUMINAMATH_CALUDE_quadratic_completion_sum_l3921_392187

/-- For the quadratic x^2 - 24x + 50, when written as (x+b)^2 + c, b+c equals -106 -/
theorem quadratic_completion_sum (b c : ℝ) : 
  (∀ x, x^2 - 24*x + 50 = (x+b)^2 + c) → b + c = -106 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completion_sum_l3921_392187


namespace NUMINAMATH_CALUDE_division_multiplication_result_l3921_392116

theorem division_multiplication_result : (2 : ℚ) / 3 * (-1/3) = -2/9 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l3921_392116


namespace NUMINAMATH_CALUDE_disk_intersection_theorem_l3921_392149

-- Define a type for colors
inductive Color
  | Red
  | White
  | Green

-- Define a type for disks
structure Disk where
  color : Color
  center : ℝ × ℝ
  radius : ℝ

-- Define a function to check if two disks intersect
def intersect (d1 d2 : Disk) : Prop :=
  let (x1, y1) := d1.center
  let (x2, y2) := d2.center
  (x1 - x2) ^ 2 + (y1 - y2) ^ 2 ≤ (d1.radius + d2.radius) ^ 2

-- Define a function to check if three disks have a common point
def commonPoint (d1 d2 d3 : Disk) : Prop :=
  ∃ (x y : ℝ), 
    (x - d1.center.1) ^ 2 + (y - d1.center.2) ^ 2 ≤ d1.radius ^ 2 ∧
    (x - d2.center.1) ^ 2 + (y - d2.center.2) ^ 2 ≤ d2.radius ^ 2 ∧
    (x - d3.center.1) ^ 2 + (y - d3.center.2) ^ 2 ≤ d3.radius ^ 2

-- State the theorem
theorem disk_intersection_theorem (disks : Finset Disk) :
  (disks.card = 6) →
  (∃ (r1 r2 w1 w2 g1 g2 : Disk), 
    r1 ∈ disks ∧ r2 ∈ disks ∧ w1 ∈ disks ∧ w2 ∈ disks ∧ g1 ∈ disks ∧ g2 ∈ disks ∧
    r1.color = Color.Red ∧ r2.color = Color.Red ∧
    w1.color = Color.White ∧ w2.color = Color.White ∧
    g1.color = Color.Green ∧ g2.color = Color.Green) →
  (∀ (r w g : Disk), r ∈ disks → w ∈ disks → g ∈ disks →
    r.color = Color.Red → w.color = Color.White → g.color = Color.Green →
    commonPoint r w g) →
  (∃ (c : Color), ∃ (d1 d2 : Disk), d1 ∈ disks ∧ d2 ∈ disks ∧
    d1.color = c ∧ d2.color = c ∧ intersect d1 d2) :=
by sorry

end NUMINAMATH_CALUDE_disk_intersection_theorem_l3921_392149


namespace NUMINAMATH_CALUDE_fraction_value_for_quadratic_relation_l3921_392131

theorem fraction_value_for_quadratic_relation (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2 - 2*a*b - 3*b^2 = 0) : (2*a + 3*b) / (2*a - b) = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_for_quadratic_relation_l3921_392131


namespace NUMINAMATH_CALUDE_system_solution_l3921_392150

theorem system_solution :
  ∃! (x y : ℝ), (x + 2 * Real.sqrt y = 2) ∧ (2 * Real.sqrt x + y = 2) ∧ (x = 4 - 2 * Real.sqrt 3) ∧ (y = 4 - 2 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3921_392150


namespace NUMINAMATH_CALUDE_floor_paving_cost_l3921_392156

/-- The cost of paving a rectangular floor -/
theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (rate : ℝ) 
  (h1 : length = 5.5) 
  (h2 : width = 3.75) 
  (h3 : rate = 1200) : 
  length * width * rate = 24750 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l3921_392156


namespace NUMINAMATH_CALUDE_folded_hexagon_result_verify_interior_angle_sum_l3921_392103

/-- Represents the possible polygons resulting from folding a regular hexagon in half -/
inductive FoldedHexagonShape
  | Quadrilateral
  | Pentagon

/-- Calculates the sum of interior angles for a polygon with n sides -/
def sumOfInteriorAngles (n : ℕ) : ℕ := (n - 2) * 180

/-- Represents the result of folding a regular hexagon in half -/
structure FoldedHexagonResult where
  shape : FoldedHexagonShape
  interiorAngleSum : ℕ

/-- Theorem stating the possible results of folding a regular hexagon in half -/
theorem folded_hexagon_result :
  ∃ (result : FoldedHexagonResult),
    (result.shape = FoldedHexagonShape.Quadrilateral ∧ result.interiorAngleSum = 360) ∨
    (result.shape = FoldedHexagonShape.Pentagon ∧ result.interiorAngleSum = 540) :=
by
  sorry

/-- Verification that the sum of interior angles is correct for each shape -/
theorem verify_interior_angle_sum :
  ∀ (result : FoldedHexagonResult),
    (result.shape = FoldedHexagonShape.Quadrilateral → result.interiorAngleSum = sumOfInteriorAngles 4) ∧
    (result.shape = FoldedHexagonShape.Pentagon → result.interiorAngleSum = sumOfInteriorAngles 5) :=
by
  sorry

end NUMINAMATH_CALUDE_folded_hexagon_result_verify_interior_angle_sum_l3921_392103


namespace NUMINAMATH_CALUDE_cos_thirty_degrees_l3921_392117

theorem cos_thirty_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirty_degrees_l3921_392117


namespace NUMINAMATH_CALUDE_whitewashing_cost_example_l3921_392174

/-- Calculate the cost of white washing a room's walls given its dimensions and openings. -/
def whitewashingCost (length width height doorWidth doorHeight windowWidth windowHeight : ℝ)
  (numWindows : ℕ) (costPerSquareFoot : ℝ) : ℝ :=
  let wallArea := 2 * (length * height + width * height)
  let doorArea := doorWidth * doorHeight
  let windowArea := numWindows * (windowWidth * windowHeight)
  let areaToPaint := wallArea - doorArea - windowArea
  areaToPaint * costPerSquareFoot

/-- The cost of white washing the room is Rs. 2718. -/
theorem whitewashing_cost_example :
  whitewashingCost 25 15 12 6 3 4 3 3 3 = 2718 := by
  sorry

end NUMINAMATH_CALUDE_whitewashing_cost_example_l3921_392174


namespace NUMINAMATH_CALUDE_pair_conditions_l3921_392106

def satisfies_conditions (a b : ℚ) : Prop :=
  a * b = 24 ∧ a + b > 0

theorem pair_conditions :
  ¬(satisfies_conditions (-6) (-4)) ∧
  (satisfies_conditions 3 8) ∧
  ¬(satisfies_conditions (-3/2) (-16)) ∧
  (satisfies_conditions 2 12) ∧
  (satisfies_conditions (4/3) 18) :=
by sorry

end NUMINAMATH_CALUDE_pair_conditions_l3921_392106


namespace NUMINAMATH_CALUDE_square_sum_equality_l3921_392173

theorem square_sum_equality (y : ℝ) : 
  (y - 2)^2 + 2*(y - 2)*(5 + y) + (5 + y)^2 = (2*y + 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3921_392173


namespace NUMINAMATH_CALUDE_square_area_from_rectangle_perimeter_l3921_392180

/-- If a square is cut into two identical rectangles, each with a perimeter of 24 cm,
    then the area of the original square is 64 cm². -/
theorem square_area_from_rectangle_perimeter :
  ∀ (side : ℝ), side > 0 →
  (2 * (side + side / 2) = 24) →
  side * side = 64 := by
sorry

end NUMINAMATH_CALUDE_square_area_from_rectangle_perimeter_l3921_392180


namespace NUMINAMATH_CALUDE_perpendicular_sum_l3921_392153

/-- Given vectors a and b in ℝ², if a + b is perpendicular to a, 
    then the second component of b is -1. -/
theorem perpendicular_sum (a b : ℝ × ℝ) (h : a = (1, 0)) :
  (a + b) • a = 0 → b.1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_sum_l3921_392153


namespace NUMINAMATH_CALUDE_least_subtraction_for_divisibility_solution_9876543210_and_29_l3921_392102

theorem least_subtraction_for_divisibility (n : ℕ) (d : ℕ) (h : d > 0) :
  let r := n % d
  (∃ (k : ℕ), (n - r) = d * k) ∧ (∀ (m : ℕ), m < r → ¬(∃ (k : ℕ), (n - m) = d * k)) :=
by
  sorry

theorem solution_9876543210_and_29 :
  let n : ℕ := 9876543210
  let d : ℕ := 29
  let r : ℕ := n % d
  r = 6 ∧
  (∃ (k : ℕ), (n - r) = d * k) ∧
  (∀ (m : ℕ), m < r → ¬(∃ (k : ℕ), (n - m) = d * k)) :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtraction_for_divisibility_solution_9876543210_and_29_l3921_392102


namespace NUMINAMATH_CALUDE_max_rotation_surface_area_l3921_392104

/-- Represents a triangle inscribed in a circle -/
structure InscribedTriangle where
  r : ℝ  -- radius of the circumscribed circle
  A : ℝ × ℝ  -- coordinates of point A
  B : ℝ × ℝ  -- coordinates of point B
  C : ℝ × ℝ  -- coordinates of point C

/-- Calculates the surface area generated by rotating side BC around the tangent at A -/
def rotationSurfaceArea (triangle : InscribedTriangle) : ℝ :=
  sorry

/-- Theorem: The maximum surface area generated by rotating side BC of an inscribed triangle
    around the tangent at A is achieved when the triangle is equilateral and equals 3r²π√3 -/
theorem max_rotation_surface_area (triangle : InscribedTriangle) :
  rotationSurfaceArea triangle ≤ 3 * triangle.r^2 * Real.pi * Real.sqrt 3 ∧
  (rotationSurfaceArea triangle = 3 * triangle.r^2 * Real.pi * Real.sqrt 3 ↔
   triangle.A.1^2 + triangle.A.2^2 = triangle.r^2 ∧
   triangle.B.1^2 + triangle.B.2^2 = triangle.r^2 ∧
   triangle.C.1^2 + triangle.C.2^2 = triangle.r^2 ∧
   (triangle.A.1 - triangle.B.1)^2 + (triangle.A.2 - triangle.B.2)^2 =
   (triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2 ∧
   (triangle.A.1 - triangle.C.1)^2 + (triangle.A.2 - triangle.C.2)^2 =
   (triangle.B.1 - triangle.C.1)^2 + (triangle.B.2 - triangle.C.2)^2) :=
by sorry


end NUMINAMATH_CALUDE_max_rotation_surface_area_l3921_392104


namespace NUMINAMATH_CALUDE_can_obtain_all_graphs_l3921_392113

/-- Represents a candidate in the election -/
structure Candidate where
  id : Nat

/-- Represents a voter's ranking of candidates -/
structure Ranking where
  preferences : List Candidate

/-- Represents the election system -/
structure ElectionSystem where
  candidates : Finset Candidate
  voters : Finset Nat
  rankings : Nat → Ranking

/-- Represents a directed graph -/
structure DirectedGraph where
  vertices : Finset Candidate
  edges : Candidate → Candidate → Bool

/-- Counts the number of votes where a is ranked higher than b -/
def countPreferences (system : ElectionSystem) (a b : Candidate) : Nat :=
  sorry

/-- Checks if there should be an edge from a to b based on majority preference -/
def hasEdge (system : ElectionSystem) (a b : Candidate) : Bool :=
  2 * countPreferences system a b > system.voters.card

/-- Constructs a directed graph based on the election system -/
def constructGraph (system : ElectionSystem) : DirectedGraph :=
  sorry

/-- Theorem stating that any connected complete directed graph can be obtained -/
theorem can_obtain_all_graphs (n : Nat) :
  ∃ (system : ElectionSystem),
    system.candidates.card = n ∧
    system.voters.card = n ∧
    ∀ (g : DirectedGraph),
      g.vertices = system.candidates →
      ∃ (newSystem : ElectionSystem),
        newSystem.candidates = system.candidates ∧
        constructGraph newSystem = g :=
  sorry

end NUMINAMATH_CALUDE_can_obtain_all_graphs_l3921_392113


namespace NUMINAMATH_CALUDE_equation_roots_and_ellipse_condition_l3921_392127

theorem equation_roots_and_ellipse_condition (m n : ℝ) : 
  ¬(((m^2 - 4*n ≥ 0 ∧ m > 0 ∧ n > 0) → (m > 0 ∧ n > 0 ∧ m ≠ n)) ∧ 
    ((m > 0 ∧ n > 0 ∧ m ≠ n) → (m^2 - 4*n ≥ 0 ∧ m > 0 ∧ n > 0))) :=
by sorry

end NUMINAMATH_CALUDE_equation_roots_and_ellipse_condition_l3921_392127


namespace NUMINAMATH_CALUDE_noProblemProbabilityIs377Over729_l3921_392108

/-- Recursive function to calculate the number of valid arrangements for n people --/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | (n+2) => 3 * validArrangements (n+1) - validArrangements n

/-- The number of chairs and people --/
def numChairs : ℕ := 6

/-- The total number of possible arrangements --/
def totalArrangements : ℕ := 3^numChairs

/-- The probability of no problematic seating arrangement --/
def noProblemProbability : ℚ := validArrangements numChairs / totalArrangements

theorem noProblemProbabilityIs377Over729 : 
  noProblemProbability = 377 / 729 := by sorry

end NUMINAMATH_CALUDE_noProblemProbabilityIs377Over729_l3921_392108


namespace NUMINAMATH_CALUDE_coefficient_sum_l3921_392109

theorem coefficient_sum (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (x - 2)^5 = a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 31 := by
sorry

end NUMINAMATH_CALUDE_coefficient_sum_l3921_392109


namespace NUMINAMATH_CALUDE_edmund_gift_wrapping_l3921_392130

/-- Given Edmund's gift wrapping scenario, prove he can wrap 15 gift boxes in 3 days -/
theorem edmund_gift_wrapping 
  (inches_per_box : ℕ) 
  (inches_per_day : ℕ) 
  (days : ℕ) 
  (h1 : inches_per_box = 18) 
  (h2 : inches_per_day = 90) 
  (h3 : days = 3) : 
  (inches_per_day / inches_per_box) * days = 15 := by
  sorry

#check edmund_gift_wrapping

end NUMINAMATH_CALUDE_edmund_gift_wrapping_l3921_392130


namespace NUMINAMATH_CALUDE_rect_to_polar_conversion_l3921_392199

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_conversion :
  ∀ (x y : ℝ), x = 2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 →
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 4 ∧ θ = π / 4 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry


end NUMINAMATH_CALUDE_rect_to_polar_conversion_l3921_392199


namespace NUMINAMATH_CALUDE_same_color_probability_l3921_392129

/-- The probability of drawing two balls of the same color from a bag with 2 red and 2 white balls, with replacement -/
theorem same_color_probability (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 2 →
  white_balls = 2 →
  (red_balls : ℚ) / total_balls * (red_balls : ℚ) / total_balls +
  (white_balls : ℚ) / total_balls * (white_balls : ℚ) / total_balls = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_same_color_probability_l3921_392129


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l3921_392195

theorem quadratic_root_implies_m (m : ℚ) : 
  ((-2 : ℚ)^2 - m*(-2) - 3 = 0) → m = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l3921_392195


namespace NUMINAMATH_CALUDE_student_bicycle_speed_l3921_392191

/-- Given two students A and B traveling 12 km, where A's speed is 1.2 times B's,
    and A arrives 1/6 hour earlier, B's speed is 12 km/h. -/
theorem student_bicycle_speed (distance : ℝ) (speed_ratio : ℝ) (time_diff : ℝ) :
  distance = 12 →
  speed_ratio = 1.2 →
  time_diff = 1/6 →
  ∃ (speed_B : ℝ),
    distance / speed_B - distance / (speed_ratio * speed_B) = time_diff ∧
    speed_B = 12 := by
  sorry

end NUMINAMATH_CALUDE_student_bicycle_speed_l3921_392191


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l3921_392164

noncomputable def x₁ (t C₁ C₂ : ℝ) : ℝ :=
  (C₁ + C₂ - 2 * t^2) / (2 * (C₁ - t^2) * (C₂ - t^2))

noncomputable def x₂ (t C₁ C₂ : ℝ) : ℝ :=
  (C₂ - C₁) / (2 * (C₁ - t^2) * (C₂ - t^2))

theorem solution_satisfies_system (t C₁ C₂ : ℝ) :
  deriv (fun t => x₁ t C₁ C₂) t = 2 * (x₁ t C₁ C₂)^2 * t + 2 * (x₂ t C₁ C₂)^2 * t ∧
  deriv (fun t => x₂ t C₁ C₂) t = 4 * (x₁ t C₁ C₂) * (x₂ t C₁ C₂) * t :=
by sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l3921_392164


namespace NUMINAMATH_CALUDE_meiosis_fertilization_importance_l3921_392194

structure ReproductiveProcess where
  meiosis : Bool
  fertilization : Bool

structure BiologicalImportance where
  chromosome_maintenance : Bool
  organism_biology : Bool

structure GenerationalEffect where
  somatic_cell_chromosomes : Bool
  heredity : Bool
  variation : Bool

/-- Given that meiosis and fertilization are important for maintaining constant
    chromosome numbers in species and crucial for the biology of organisms,
    prove that they are crucial for maintaining constant chromosome numbers in
    somatic cells of successive generations and are important for heredity and variation. -/
theorem meiosis_fertilization_importance
  (process : ReproductiveProcess)
  (importance : BiologicalImportance)
  (h1 : process.meiosis ∧ process.fertilization)
  (h2 : importance.chromosome_maintenance)
  (h3 : importance.organism_biology) :
  ∃ (effect : GenerationalEffect),
    effect.somatic_cell_chromosomes ∧
    effect.heredity ∧
    effect.variation :=
sorry

end NUMINAMATH_CALUDE_meiosis_fertilization_importance_l3921_392194


namespace NUMINAMATH_CALUDE_photos_per_album_l3921_392189

/-- Given 180 total photos divided equally among 9 albums, prove that each album contains 20 photos. -/
theorem photos_per_album (total_photos : ℕ) (num_albums : ℕ) (photos_per_album : ℕ) : 
  total_photos = 180 → num_albums = 9 → total_photos = num_albums * photos_per_album → photos_per_album = 20 := by
  sorry

end NUMINAMATH_CALUDE_photos_per_album_l3921_392189


namespace NUMINAMATH_CALUDE_cubic_factorization_l3921_392166

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x-2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l3921_392166


namespace NUMINAMATH_CALUDE_min_tangent_length_l3921_392182

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define the symmetry line
def symmetry_line (a b x y : ℝ) : Prop := 2*a*x + b*y + 6 = 0

-- Define the tangent point
def tangent_point (a b : ℝ) : Prop := ∃ x y : ℝ, circle_C x y ∧ symmetry_line a b x y

-- Theorem statement
theorem min_tangent_length (a b : ℝ) : 
  tangent_point a b → 
  (∃ t : ℝ, t ≥ 0 ∧ 
    (∀ s : ℝ, s ≥ 0 → 
      (∃ x y : ℝ, circle_C x y ∧ (x - a)^2 + (y - b)^2 = s^2) → 
      t ≤ s) ∧ 
    t = 4) := 
sorry

end NUMINAMATH_CALUDE_min_tangent_length_l3921_392182


namespace NUMINAMATH_CALUDE_midpoint_property_l3921_392121

/-- Given two points A and B in a 2D plane, prove that if C is the midpoint of AB,
    then 2x - 4y = -15, where (x, y) are the coordinates of C. -/
theorem midpoint_property (A B C : ℝ × ℝ) : 
  A = (17, 10) → B = (-2, 5) → C = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) → 
  2 * C.1 - 4 * C.2 = -15 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_property_l3921_392121


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l3921_392123

/-- Represents the number of people wearing each color of clothing -/
structure ClothingCounts where
  blue : Nat
  yellow : Nat
  red : Nat

/-- Calculates the number of valid arrangements for a given set of clothing counts -/
def validArrangements (counts : ClothingCounts) : Nat :=
  sorry

/-- The specific problem instance -/
def problemInstance : ClothingCounts :=
  { blue := 2, yellow := 2, red := 1 }

/-- The main theorem stating that the number of valid arrangements for the problem instance is 48 -/
theorem valid_arrangements_count :
  validArrangements problemInstance = 48 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l3921_392123


namespace NUMINAMATH_CALUDE_unique_intersection_points_l3921_392119

/-- The first curve -/
def curve1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- The second curve -/
def curve2 (x : ℝ) : ℝ := -x^3 + 9 * x^2 - 4 * x + 2

/-- The intersection points of the two curves -/
def intersection_points : Set (ℝ × ℝ) := {(0, 2), (6, 86)}

/-- Theorem stating that the intersection_points are the only intersection points of curve1 and curve2 -/
theorem unique_intersection_points :
  ∀ x y : ℝ, curve1 x = curve2 x ∧ y = curve1 x ↔ (x, y) ∈ intersection_points :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_points_l3921_392119


namespace NUMINAMATH_CALUDE_no_real_roots_l3921_392145

theorem no_real_roots : ∀ x : ℝ, x ≠ 2 → 
  (3 * x^2) / (x - 2) - (3 * x + 8) / 2 + (5 - 9 * x) / (x - 2) + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3921_392145


namespace NUMINAMATH_CALUDE_number_of_cows_l3921_392144

/-- Represents the number of legs for each animal type -/
def legs_per_animal : (Fin 2) → ℕ
| 0 => 2  -- chickens
| 1 => 4  -- cows

/-- Represents the total number of animals -/
def total_animals : ℕ := 160

/-- Represents the total number of legs -/
def total_legs : ℕ := 400

/-- Proves that the number of cows is 40 given the conditions -/
theorem number_of_cows : 
  ∃ (chickens cows : ℕ), 
    chickens + cows = total_animals ∧ 
    chickens * legs_per_animal 0 + cows * legs_per_animal 1 = total_legs ∧
    cows = 40 := by
  sorry

end NUMINAMATH_CALUDE_number_of_cows_l3921_392144


namespace NUMINAMATH_CALUDE_conditions_sufficient_not_necessary_l3921_392160

theorem conditions_sufficient_not_necessary (m : ℝ) (h : m > 0) :
  (∀ x y a : ℝ, |x - a| < m ∧ |y - a| < m → |x - y| < 2*m) ∧
  (∃ x y a : ℝ, |x - y| < 2*m ∧ (|x - a| ≥ m ∨ |y - a| ≥ m)) :=
by sorry

end NUMINAMATH_CALUDE_conditions_sufficient_not_necessary_l3921_392160


namespace NUMINAMATH_CALUDE_flag_design_count_l3921_392142

/-- The number of school colors -/
def num_colors : ℕ := 3

/-- The number of horizontal stripes on the flag -/
def num_horizontal_stripes : ℕ := 3

/-- The number of options for the vertical stripe (3 colors + no stripe) -/
def vertical_stripe_options : ℕ := num_colors + 1

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_horizontal_stripes * vertical_stripe_options

theorem flag_design_count :
  total_flag_designs = 108 :=
sorry

end NUMINAMATH_CALUDE_flag_design_count_l3921_392142


namespace NUMINAMATH_CALUDE_pen_price_proof_l3921_392181

/-- Represents the regular price of a pen in dollars -/
def regular_price : ℝ := 2

/-- Represents the total number of pens bought -/
def total_pens : ℕ := 20

/-- Represents the total cost paid by the customer in dollars -/
def total_cost : ℝ := 30

/-- Represents the number of pens at regular price -/
def regular_price_pens : ℕ := 10

/-- Represents the number of pens at half price -/
def half_price_pens : ℕ := 10

theorem pen_price_proof :
  regular_price * regular_price_pens + 
  (regular_price / 2) * half_price_pens = total_cost ∧
  regular_price_pens + half_price_pens = total_pens := by
  sorry

end NUMINAMATH_CALUDE_pen_price_proof_l3921_392181


namespace NUMINAMATH_CALUDE_roots_polynomial_d_values_l3921_392139

theorem roots_polynomial_d_values (u v c d : ℝ) : 
  (∃ w : ℝ, {u, v, w} = {x | x^3 + c*x + d = 0}) ∧
  (∃ w : ℝ, {u+3, v-2, w} = {x | x^3 + c*x + (d+120) = 0}) →
  d = 84 ∨ d = -25 := by
sorry

end NUMINAMATH_CALUDE_roots_polynomial_d_values_l3921_392139


namespace NUMINAMATH_CALUDE_a_range_l3921_392141

def sequence_a (a : ℝ) : ℕ+ → ℝ
  | ⟨1, _⟩ => a
  | ⟨n+1, _⟩ => 4*(n+1) + (-1)^(n+1) * (8 - 2*a)

theorem a_range (a : ℝ) :
  (∀ n : ℕ+, sequence_a a n < sequence_a a (n + 1)) →
  (3 < a ∧ a < 5) :=
by sorry

end NUMINAMATH_CALUDE_a_range_l3921_392141


namespace NUMINAMATH_CALUDE_root_count_relationship_l3921_392155

-- Define the number of real roots for each equation
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- State the theorem
theorem root_count_relationship : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_root_count_relationship_l3921_392155


namespace NUMINAMATH_CALUDE_swap_result_l3921_392168

def swap_values (x y : ℕ) : ℕ × ℕ :=
  let t := x
  let x := y
  let y := t
  (x, y)

theorem swap_result : swap_values 5 6 = (6, 5) := by
  sorry

end NUMINAMATH_CALUDE_swap_result_l3921_392168


namespace NUMINAMATH_CALUDE_horatio_sonnets_l3921_392136

/-- Represents the number of lines in a sonnet -/
def lines_per_sonnet : ℕ := 14

/-- Represents the number of sonnets the lady heard before telling Horatio to leave -/
def sonnets_heard : ℕ := 7

/-- Represents the number of romantic lines Horatio wrote that were never heard -/
def unheard_lines : ℕ := 70

/-- Calculates the total number of sonnets Horatio wrote -/
def total_sonnets : ℕ := sonnets_heard + (unheard_lines / lines_per_sonnet)

theorem horatio_sonnets : total_sonnets = 12 := by sorry

end NUMINAMATH_CALUDE_horatio_sonnets_l3921_392136


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3921_392193

theorem inequality_system_solution (x : ℤ) : 
  (2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1) ↔ x ∈ ({3, 4, 5} : Set ℤ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3921_392193


namespace NUMINAMATH_CALUDE_largest_number_l3921_392161

def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

def number_85_9 : Nat := to_decimal [8, 5] 9
def number_210_6 : Nat := to_decimal [2, 1, 0] 6
def number_1000_4 : Nat := to_decimal [1, 0, 0, 0] 4
def number_11111_2 : Nat := to_decimal [1, 1, 1, 1, 1] 2

theorem largest_number :
  number_210_6 > number_85_9 ∧
  number_210_6 > number_1000_4 ∧
  number_210_6 > number_11111_2 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l3921_392161


namespace NUMINAMATH_CALUDE_flight_duration_sum_l3921_392132

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDifferenceInMinutes (t1 t2 : Time) : ℕ :=
  (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)

/-- Theorem: Flight duration calculation -/
theorem flight_duration_sum (departureTime : Time) (arrivalTime : Time) 
  (h m : ℕ) (hm : 0 < m ∧ m < 60) :
  departureTime.hours = 9 ∧ departureTime.minutes = 17 →
  arrivalTime.hours = 13 ∧ arrivalTime.minutes = 53 →
  timeDifferenceInMinutes departureTime arrivalTime = h * 60 + m →
  h + m = 41 := by
  sorry

#check flight_duration_sum

end NUMINAMATH_CALUDE_flight_duration_sum_l3921_392132


namespace NUMINAMATH_CALUDE_parabolas_intersection_circle_l3921_392190

/-- The parabolas y = (x - 2)^2 and x + 1 = (y + 2)^2 intersect at four points that lie on a circle with radius squared equal to 3/2 -/
theorem parabolas_intersection_circle (x y : ℝ) : 
  (y = (x - 2)^2 ∧ x + 1 = (y + 2)^2) → 
  (x - 5/2)^2 + (y + 3/2)^2 = 3/2 := by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_circle_l3921_392190


namespace NUMINAMATH_CALUDE_train_length_proof_l3921_392146

-- Define the given parameters
def train_speed : Real := 45 -- km/hr
def platform_length : Real := 180 -- meters
def time_to_pass : Real := 43.2 -- seconds

-- Define the theorem
theorem train_length_proof :
  let speed_ms : Real := train_speed * 1000 / 3600 -- Convert km/hr to m/s
  let total_distance : Real := speed_ms * time_to_pass
  let train_length : Real := total_distance - platform_length
  train_length = 360 := by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l3921_392146


namespace NUMINAMATH_CALUDE_problem_statement_l3921_392184

theorem problem_statement (x y : ℝ) : 
  (x - 1)^2 + |y + 1| = 0 → 2*(x^2 - y^2 + 1) - 2*(x^2 + y^2) + x*y = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3921_392184


namespace NUMINAMATH_CALUDE_inequality_solution_l3921_392169

theorem inequality_solution : 
  ∀ x : ℝ, (|x - 1| + |x + 2| + |x| < 7) ↔ (-2 < x ∧ x < 2) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3921_392169


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l3921_392185

theorem nested_fraction_equality : 
  1 + 1 / (1 - 1 / (2 + 1 / 3)) = 11 / 4 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l3921_392185


namespace NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3921_392135

theorem quadratic_root_implies_coefficient (b : ℝ) : 
  (2^2 + b*2 - 10 = 0) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_coefficient_l3921_392135


namespace NUMINAMATH_CALUDE_lcm_210_396_l3921_392115

theorem lcm_210_396 : Nat.lcm 210 396 = 13860 := by
  sorry

end NUMINAMATH_CALUDE_lcm_210_396_l3921_392115


namespace NUMINAMATH_CALUDE_xy_range_l3921_392138

theorem xy_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2/x + 8/y = 1) :
  ∃ (m : ℝ), m = 64 ∧ xy ≥ m ∧ ∀ (z : ℝ), z > m → ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 2/a + 8/b = 1 ∧ a * b = z :=
sorry

end NUMINAMATH_CALUDE_xy_range_l3921_392138


namespace NUMINAMATH_CALUDE_clothing_price_theorem_l3921_392151

/-- The price per item of clothing, in yuan. -/
def price : ℝ := 110

/-- The number of items sold with the percentage discount. -/
def items_with_percent_discount : ℕ := 10

/-- The number of items sold with the fixed discount. -/
def items_with_fixed_discount : ℕ := 11

/-- The percentage discount applied to the first set of items. -/
def percent_discount : ℝ := 0.08

/-- The fixed discount applied to the second set of items, in yuan. -/
def fixed_discount : ℝ := 30

theorem clothing_price_theorem : 
  (items_with_percent_discount : ℝ) * price * (1 - percent_discount) = 
  (items_with_fixed_discount : ℝ) * (price - fixed_discount) := by sorry

end NUMINAMATH_CALUDE_clothing_price_theorem_l3921_392151


namespace NUMINAMATH_CALUDE_complex_sum_equals_two_l3921_392192

def z : ℂ := 1 - Complex.I

theorem complex_sum_equals_two : (2 / z) + z = 2 := by sorry

end NUMINAMATH_CALUDE_complex_sum_equals_two_l3921_392192


namespace NUMINAMATH_CALUDE_initial_number_of_people_l3921_392112

/-- Given a group of people where replacing one person increases the average weight,
    this theorem proves the initial number of people in the group. -/
theorem initial_number_of_people
  (n : ℕ) -- Initial number of people
  (weight_increase_per_person : ℝ) -- Average weight increase per person
  (weight_difference : ℝ) -- Weight difference between new and replaced person
  (h1 : weight_increase_per_person = 2.5)
  (h2 : weight_difference = 20)
  (h3 : n * weight_increase_per_person = weight_difference) :
  n = 8 :=
by sorry

end NUMINAMATH_CALUDE_initial_number_of_people_l3921_392112


namespace NUMINAMATH_CALUDE_equation_solution_l3921_392124

theorem equation_solution : 
  ∀ x : ℝ, (x^2 + x)^2 - 4*(x^2 + x) - 12 = 0 ↔ x = -3 ∨ x = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l3921_392124


namespace NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_l3921_392177

/-- A cubic polynomial function -/
def f (b c d : ℝ) (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

/-- Monotonicity of a function on ℝ -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y ∨ (∀ x y, x < y → f x > f y)

/-- Intersection with x-axis at exactly one point -/
def IntersectsOnce (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

/-- Theorem stating that monotonicity is sufficient but not necessary for intersecting x-axis once -/
theorem monotonic_sufficient_not_necessary (b c d : ℝ) :
  (Monotonic (f b c d) → IntersectsOnce (f b c d)) ∧
  ¬(IntersectsOnce (f b c d) → Monotonic (f b c d)) :=
sorry

end NUMINAMATH_CALUDE_monotonic_sufficient_not_necessary_l3921_392177


namespace NUMINAMATH_CALUDE_compound_composition_l3921_392171

/-- The atomic weight of Aluminium in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Sulphur in g/mol -/
def atomic_weight_S : ℝ := 32.06

/-- The number of Sulphur atoms in the compound -/
def num_S_atoms : ℕ := 3

/-- The molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := 150

/-- The number of Aluminium atoms in the compound -/
def num_Al_atoms : ℕ := 2

theorem compound_composition :
  num_Al_atoms * atomic_weight_Al + num_S_atoms * atomic_weight_S = molecular_weight := by
  sorry

end NUMINAMATH_CALUDE_compound_composition_l3921_392171


namespace NUMINAMATH_CALUDE_bird_nests_calculation_l3921_392126

/-- Calculates the total number of nests required for birds in a park --/
theorem bird_nests_calculation (total_birds : Nat) 
  (sparrows pigeons starlings : Nat)
  (sparrow_nests pigeon_nests starling_nests : Nat)
  (h1 : total_birds = sparrows + pigeons + starlings)
  (h2 : total_birds = 10)
  (h3 : sparrows = 4)
  (h4 : pigeons = 3)
  (h5 : starlings = 3)
  (h6 : sparrow_nests = 1)
  (h7 : pigeon_nests = 2)
  (h8 : starling_nests = 3) :
  sparrows * sparrow_nests + pigeons * pigeon_nests + starlings * starling_nests = 19 := by
  sorry

end NUMINAMATH_CALUDE_bird_nests_calculation_l3921_392126


namespace NUMINAMATH_CALUDE_sequence_properties_l3921_392157

def a (n : ℕ) : ℕ := 3 * (n^2 + n) + 7

theorem sequence_properties :
  (∀ k : ℕ, 
    5 ∣ a (5*k + 2) ∧ 
    ¬(5 ∣ a (5*k)) ∧ 
    ¬(5 ∣ a (5*k + 1)) ∧ 
    ¬(5 ∣ a (5*k + 3)) ∧ 
    ¬(5 ∣ a (5*k + 4))) ∧
  (∀ n t : ℕ, a n ≠ t^3) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3921_392157


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l3921_392170

theorem smallest_absolute_value : ∀ (a b c : ℤ), 
  a = -3 → b = -2 → c = 1 → 
  |0| < |a| ∧ |0| < |b| ∧ |0| < |c| :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l3921_392170


namespace NUMINAMATH_CALUDE_system_solution_l3921_392137

theorem system_solution (x y : ℝ) : 
  x^5 + y^5 = 1 ∧ x^6 + y^6 = 1 ↔ (x = 1 ∧ y = 0) ∨ (x = 0 ∧ y = 1) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3921_392137


namespace NUMINAMATH_CALUDE_butterflies_in_garden_l3921_392154

theorem butterflies_in_garden (initial : ℕ) (flew_away : ℕ) (remaining : ℕ) : 
  initial = 9 → 
  flew_away = initial / 3 → 
  remaining = initial - flew_away → 
  remaining = 6 := by
sorry

end NUMINAMATH_CALUDE_butterflies_in_garden_l3921_392154


namespace NUMINAMATH_CALUDE_b_sequence_max_at_4_l3921_392176

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sequence_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := n * (2 * a₁ + (n - 1 : ℚ) * d) / 2

def b_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := (1 + arithmetic_sequence a₁ d n) / arithmetic_sequence a₁ d n

theorem b_sequence_max_at_4 (a₁ d : ℚ) (h₁ : a₁ = -5/2) (h₂ : sequence_sum a₁ d 4 = 2 * sequence_sum a₁ d 2 + 4) :
  ∀ n : ℕ, n ≥ 1 → b_sequence a₁ d 4 ≥ b_sequence a₁ d n :=
sorry

end NUMINAMATH_CALUDE_b_sequence_max_at_4_l3921_392176


namespace NUMINAMATH_CALUDE_existence_of_sum_greater_than_one_l3921_392188

theorem existence_of_sum_greater_than_one : 
  ¬(∀ (x y : ℝ), x + y ≤ 1) := by sorry

end NUMINAMATH_CALUDE_existence_of_sum_greater_than_one_l3921_392188


namespace NUMINAMATH_CALUDE_sum_of_digits_next_l3921_392158

/-- S(n) is the sum of the digits of a positive integer n -/
def S (n : ℕ+) : ℕ :=
  sorry

theorem sum_of_digits_next (n : ℕ+) (h : S n = 1274) : S (n + 1) = 1239 :=
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_next_l3921_392158


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l3921_392110

theorem quadratic_rewrite (x : ℝ) : ∃ (a b c : ℤ), 
  16 * x^2 - 40 * x - 72 = (a * x + b)^2 + c ∧ a * b = -20 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l3921_392110


namespace NUMINAMATH_CALUDE_fraction_square_equals_49_l3921_392120

theorem fraction_square_equals_49 : (3072 - 2993)^2 / 121 = 49 := by sorry

end NUMINAMATH_CALUDE_fraction_square_equals_49_l3921_392120


namespace NUMINAMATH_CALUDE_min_value_a_l3921_392178

theorem min_value_a (h : ∀ x y : ℝ, x > 0 → y > 0 → x + y ≥ 9) :
  ∃ a : ℝ, a > 0 ∧ (∀ x : ℝ, x > 0 → x + a ≥ 9) ∧
  (∀ b : ℝ, b > 0 → (∀ x : ℝ, x > 0 → x + b ≥ 9) → b ≥ a) ∧
  a = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l3921_392178


namespace NUMINAMATH_CALUDE_football_game_cost_l3921_392197

def total_spent : ℚ := 35.52
def strategy_game_cost : ℚ := 9.46
def batman_game_cost : ℚ := 12.04

theorem football_game_cost :
  total_spent - strategy_game_cost - batman_game_cost = 13.02 := by
  sorry

end NUMINAMATH_CALUDE_football_game_cost_l3921_392197


namespace NUMINAMATH_CALUDE_dividend_calculation_l3921_392172

theorem dividend_calculation (divisor quotient remainder : Int) 
  (h1 : divisor = 800)
  (h2 : quotient = 594)
  (h3 : remainder = -968) :
  divisor * quotient + remainder = 474232 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3921_392172


namespace NUMINAMATH_CALUDE_three_numbers_problem_l3921_392107

theorem three_numbers_problem (a b c : ℝ) 
  (sum_eq : a + b + c = 15)
  (sum_minus_third : a + b - c = 10)
  (sum_minus_second : a - b + c = 8) :
  a = 9 ∧ b = 3.5 ∧ c = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_three_numbers_problem_l3921_392107


namespace NUMINAMATH_CALUDE_x_plus_y_values_l3921_392196

theorem x_plus_y_values (x y : ℝ) (h1 : -x = 3) (h2 : |y| = 5) :
  x + y = -8 ∨ x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_values_l3921_392196


namespace NUMINAMATH_CALUDE_correct_operation_l3921_392162

theorem correct_operation (a : ℝ) : (-2 * a^2)^3 = -8 * a^6 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l3921_392162


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3921_392198

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 3 > 0) ↔ a > 1/3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3921_392198


namespace NUMINAMATH_CALUDE_inverse_proportion_l3921_392179

/-- Given that the product of x and y is constant, and x = 30 when y = 10,
    prove that x = 60 when y = 5 and the relationship doesn't hold for x = 48 and y = 15 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 30 * 10 = k) :
  (5 * 60 = k) ∧ ¬(48 * 15 = k) := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l3921_392179


namespace NUMINAMATH_CALUDE_equation_has_four_solutions_l3921_392143

-- Define the equation
def equation (x : ℝ) : Prop := (2*x^2 - 10*x + 3)^2 = 4

-- State the theorem
theorem equation_has_four_solutions :
  ∃ (a b c d : ℝ), a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
  equation a ∧ equation b ∧ equation c ∧ equation d ∧
  (∀ x : ℝ, equation x → (x = a ∨ x = b ∨ x = c ∨ x = d)) :=
sorry

end NUMINAMATH_CALUDE_equation_has_four_solutions_l3921_392143


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l3921_392128

theorem quadratic_root_implies_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m^2 - 4 = 0 ∧ x = 1) → (m = 2 ∨ m = -2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l3921_392128


namespace NUMINAMATH_CALUDE_count_valid_numbers_l3921_392183

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def valid_number (n : ℕ) : Prop :=
  n ≥ 10 ∧ n ≤ 99 ∧
  n > 80 ∧
  is_prime (n / 10) ∧
  is_even (n % 10)

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, valid_number n) ∧ S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_count_valid_numbers_l3921_392183


namespace NUMINAMATH_CALUDE_factorization_proof_l3921_392148

theorem factorization_proof (a b c : ℝ) : 
  (a^2 + 2*b^2 - 2*c^2 + 3*a*b + a*c = (a + b - c)*(a + 2*b + 2*c)) ∧
  (a^2 - 2*b^2 - 2*c^2 - a*b + 5*b*c - a*c = (a - 2*b + c)*(a + b - 2*c)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3921_392148


namespace NUMINAMATH_CALUDE_culture_and_messengers_l3921_392159

-- Define the types
structure Performance :=
  (troupe : String)
  (location : String)
  (impression : String)

-- Define the conditions
def legend_show : Performance :=
  { troupe := "Chinese Acrobatic Troupe",
    location := "United States",
    impression := "favorable" }

-- Define the properties we want to prove
def is_national_and_global (p : Performance) : Prop :=
  p.troupe ≠ p.location ∧ p.impression = "favorable"

def are_cultural_messengers (p : Performance) : Prop :=
  p.troupe = "Chinese Acrobatic Troupe" ∧ p.impression = "favorable"

-- The theorem to prove
theorem culture_and_messengers :
  is_national_and_global legend_show ∧ are_cultural_messengers legend_show :=
by sorry

end NUMINAMATH_CALUDE_culture_and_messengers_l3921_392159


namespace NUMINAMATH_CALUDE_constant_term_expansion_l3921_392122

theorem constant_term_expansion (x : ℝ) : 
  (x^4 + x + 5) * (x^5 + x^3 + 15) = x^9 + x^7 + 15*x^4 + x^6 + x^4 + 15*x + 5*x^5 + 5*x^3 + 75 := by
  sorry

#check constant_term_expansion

end NUMINAMATH_CALUDE_constant_term_expansion_l3921_392122


namespace NUMINAMATH_CALUDE_game_collection_proof_l3921_392175

theorem game_collection_proof (games_from_friend games_from_garage_sale total_good_games : ℕ) :
  let total_games := games_from_friend + games_from_garage_sale
  let non_working_games := total_games - total_good_games
  total_good_games = total_games - non_working_games :=
by
  sorry

end NUMINAMATH_CALUDE_game_collection_proof_l3921_392175


namespace NUMINAMATH_CALUDE_workers_savings_l3921_392134

theorem workers_savings (monthly_pay : ℝ) (savings_fraction : ℝ) 
  (h1 : savings_fraction = 1 / 7)
  (h2 : savings_fraction > 0)
  (h3 : savings_fraction < 1) : 
  12 * (savings_fraction * monthly_pay) = 2 * ((1 - savings_fraction) * monthly_pay) := by
  sorry

end NUMINAMATH_CALUDE_workers_savings_l3921_392134


namespace NUMINAMATH_CALUDE_magic_square_property_l3921_392125

def magic_square : Matrix (Fin 3) (Fin 3) ℚ :=
  ![![7.5, 5, 2.5],
    ![0, 5, 10],
    ![7.5, 5, 2.5]]

def row_sum (m : Matrix (Fin 3) (Fin 3) ℚ) (i : Fin 3) : ℚ :=
  m i 0 + m i 1 + m i 2

def col_sum (m : Matrix (Fin 3) (Fin 3) ℚ) (j : Fin 3) : ℚ :=
  m 0 j + m 1 j + m 2 j

def diag_sum (m : Matrix (Fin 3) (Fin 3) ℚ) : ℚ :=
  m 0 0 + m 1 1 + m 2 2

def anti_diag_sum (m : Matrix (Fin 3) (Fin 3) ℚ) : ℚ :=
  m 0 2 + m 1 1 + m 2 0

theorem magic_square_property :
  (∀ i : Fin 3, row_sum magic_square i = 15) ∧
  (∀ j : Fin 3, col_sum magic_square j = 15) ∧
  diag_sum magic_square = 15 ∧
  anti_diag_sum magic_square = 15 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_property_l3921_392125


namespace NUMINAMATH_CALUDE_min_value_M_l3921_392114

theorem min_value_M (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let M := (((a / (b + c)) ^ (1/4) : ℝ) + ((b / (c + a)) ^ (1/4) : ℝ) + ((c / (b + a)) ^ (1/4) : ℝ) +
            ((b + c) / a) ^ (1/2) + ((a + c) / b) ^ (1/2) + ((a + b) / c) ^ (1/2))
  M ≥ 3 * Real.sqrt 2 + (3 / 2) * (8 ^ (1/4) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_min_value_M_l3921_392114


namespace NUMINAMATH_CALUDE_similarity_criteria_l3921_392111

/-- A structure representing a triangle -/
structure Triangle where
  -- We'll assume triangles are defined by their side lengths and angles
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Two triangles are similar if they have the same shape but not necessarily the same size -/
def similar (t1 t2 : Triangle) : Prop :=
  sorry

/-- SSS (Side-Side-Side) Similarity: Two triangles are similar if their corresponding sides are proportional -/
def SSS_similarity (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.side1 / t2.side1 = k ∧
    t1.side2 / t2.side2 = k ∧
    t1.side3 / t2.side3 = k

/-- SAS (Side-Angle-Side) Similarity: Two triangles are similar if two pairs of corresponding sides are proportional and the included angles are equal -/
def SAS_similarity (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t1.side1 / t2.side1 = k ∧
    t1.side2 / t2.side2 = k ∧
    t1.angle3 = t2.angle3

/-- Theorem: Two triangles are similar if and only if they satisfy either SSS or SAS similarity criteria -/
theorem similarity_criteria (t1 t2 : Triangle) :
  similar t1 t2 ↔ SSS_similarity t1 t2 ∨ SAS_similarity t1 t2 :=
sorry

end NUMINAMATH_CALUDE_similarity_criteria_l3921_392111


namespace NUMINAMATH_CALUDE_minimum_value_of_f_max_a_for_decreasing_f_properties_l3921_392140

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (4-a)*x^2 - 15*x + a

-- Theorem 1
theorem minimum_value_of_f (a : ℝ) :
  f a 0 = -2 → a = -2 ∧ ∃ x₀, ∀ x, f (-2) x ≥ f (-2) x₀ ∧ f (-2) x₀ = -10 :=
sorry

-- Theorem 2
theorem max_a_for_decreasing (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1) 1, ∀ y ∈ Set.Ioo (-1) 1, x < y → f a x > f a y) →
  a ≤ 10 :=
sorry

-- Theorem combining both results
theorem f_properties :
  (∃ a, f a 0 = -2 ∧ a = -2 ∧ ∃ x₀, ∀ x, f a x ≥ f a x₀ ∧ f a x₀ = -10) ∧
  (∃ a_max, a_max = 10 ∧ ∀ a > a_max, ¬(∀ x ∈ Set.Ioo (-1) 1, ∀ y ∈ Set.Ioo (-1) 1, x < y → f a x > f a y)) :=
sorry

end NUMINAMATH_CALUDE_minimum_value_of_f_max_a_for_decreasing_f_properties_l3921_392140


namespace NUMINAMATH_CALUDE_triangle_properties_l3921_392105

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the main theorem
theorem triangle_properties (t : Triangle)
  (h : t.a / (Real.cos t.C * Real.sin t.B) = t.b / Real.sin t.B + t.c / Real.cos t.C) :
  t.B = π / 4 ∧
  (t.b = Real.sqrt 2 → 
    ∀ (area : ℝ), area = 1 / 2 * t.a * t.c * Real.sin t.B → area ≤ (Real.sqrt 2 + 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l3921_392105


namespace NUMINAMATH_CALUDE_haley_trees_count_l3921_392133

/-- The number of trees that died after the typhoon -/
def dead_trees : ℕ := 5

/-- The number of trees left after the typhoon -/
def remaining_trees : ℕ := 12

/-- The total number of trees Haley grew -/
def total_trees : ℕ := dead_trees + remaining_trees

theorem haley_trees_count : total_trees = 17 := by
  sorry

end NUMINAMATH_CALUDE_haley_trees_count_l3921_392133


namespace NUMINAMATH_CALUDE_factorial_simplification_l3921_392101

theorem factorial_simplification :
  (13 : ℕ).factorial / ((11 : ℕ).factorial + 3 * (9 : ℕ).factorial) = 17160 / 113 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l3921_392101


namespace NUMINAMATH_CALUDE_isosceles_triangle_circle_properties_l3921_392186

/-- An isosceles triangle with base 48 and side length 30 -/
structure IsoscelesTriangle where
  base : ℝ
  side : ℝ
  isIsosceles : base = 48 ∧ side = 30

/-- Properties of the inscribed and circumscribed circles of the isosceles triangle -/
def CircleProperties (t : IsoscelesTriangle) : Prop :=
  ∃ (r R d : ℝ),
    r = 8 ∧  -- radius of inscribed circle
    R = 25 ∧  -- radius of circumscribed circle
    d = 15 ∧  -- distance between centers
    r > 0 ∧ R > 0 ∧ d > 0

/-- Theorem stating the properties of the inscribed and circumscribed circles -/
theorem isosceles_triangle_circle_properties (t : IsoscelesTriangle) :
  CircleProperties t :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_circle_properties_l3921_392186


namespace NUMINAMATH_CALUDE_population_percentage_l3921_392163

theorem population_percentage (men women : ℝ) (h : women = 0.9 * men) :
  (men / women) * 100 = (1 / 0.9) * 100 := by
  sorry

end NUMINAMATH_CALUDE_population_percentage_l3921_392163
