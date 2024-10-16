import Mathlib

namespace NUMINAMATH_CALUDE_expand_expression_l2983_298333

theorem expand_expression (x y : ℝ) : 5 * (4 * x^3 - 3 * x * y + 7) = 20 * x^3 - 15 * x * y + 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2983_298333


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l2983_298329

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with given dimensions and rate -/
theorem paving_cost_calculation :
  paving_cost 5.5 3.75 800 = 16500 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l2983_298329


namespace NUMINAMATH_CALUDE_unique_root_condition_l2983_298307

theorem unique_root_condition (a : ℝ) : 
  (∃! x, Real.log (x - 2*a) - 3*(x - 2*a)^2 + 2*a = 0) ↔ 
  (a = (Real.log 6 + 1) / 4) := by sorry

end NUMINAMATH_CALUDE_unique_root_condition_l2983_298307


namespace NUMINAMATH_CALUDE_balboa_earned_180_l2983_298309

/-- Represents the earnings of students from three middle schools --/
structure SchoolEarnings where
  allen_students : Nat
  allen_days : Nat
  balboa_students : Nat
  balboa_days : Nat
  carver_students : Nat
  carver_days : Nat
  total_paid : Nat

/-- Calculates the total earnings for Balboa school students --/
def balboa_earnings (e : SchoolEarnings) : Nat :=
  let total_student_days := e.allen_students * e.allen_days + 
                            e.balboa_students * e.balboa_days + 
                            e.carver_students * e.carver_days
  let daily_wage := e.total_paid / total_student_days
  daily_wage * e.balboa_students * e.balboa_days

/-- Theorem stating that Balboa school students earned 180 dollars --/
theorem balboa_earned_180 (e : SchoolEarnings) 
  (h1 : e.allen_students = 7)
  (h2 : e.allen_days = 3)
  (h3 : e.balboa_students = 4)
  (h4 : e.balboa_days = 5)
  (h5 : e.carver_students = 5)
  (h6 : e.carver_days = 9)
  (h7 : e.total_paid = 744) :
  balboa_earnings e = 180 := by
  sorry

end NUMINAMATH_CALUDE_balboa_earned_180_l2983_298309


namespace NUMINAMATH_CALUDE_greendale_final_score_l2983_298341

/-- Roosevelt High School's basketball tournament scoring --/
def roosevelt_tournament (first_game : ℕ) (bonus : ℕ) : ℕ :=
  let second_game := first_game / 2
  let third_game := second_game * 3
  first_game + second_game + third_game + bonus

/-- Greendale High School's total points --/
def greendale_points (roosevelt_total : ℕ) : ℕ :=
  roosevelt_total - 10

/-- Theorem stating Greendale's final score --/
theorem greendale_final_score :
  greendale_points (roosevelt_tournament 30 50) = 130 := by
  sorry

end NUMINAMATH_CALUDE_greendale_final_score_l2983_298341


namespace NUMINAMATH_CALUDE_hyperbola_relation_l2983_298374

/-- Two hyperbolas M and N with the given properties -/
structure HyperbolaPair where
  /-- Eccentricity of hyperbola M -/
  e₁ : ℝ
  /-- Eccentricity of hyperbola N -/
  e₂ : ℝ
  /-- Half the length of the transverse axis of hyperbola N -/
  a : ℝ
  /-- Half the length of the conjugate axis of both hyperbolas -/
  b : ℝ
  /-- M and N are centered at the origin -/
  center_origin : True
  /-- Symmetric axes are coordinate axes -/
  symmetric_axes : True
  /-- Length of transverse axis of M is twice that of N -/
  transverse_axis_relation : True
  /-- Conjugate axes of M and N are equal -/
  conjugate_axis_equal : True
  /-- e₁ and e₂ are positive -/
  e₁_pos : e₁ > 0
  e₂_pos : e₂ > 0
  /-- a and b are positive -/
  a_pos : a > 0
  b_pos : b > 0
  /-- Definition of e₂ for hyperbola N -/
  e₂_def : e₂^2 = 1 + b^2 / a^2
  /-- Definition of e₁ for hyperbola M -/
  e₁_def : e₁^2 = 1 + b^2 / (4*a^2)

/-- The point (e₁, e₂) satisfies the equation of the hyperbola 4x²-y²=3 -/
theorem hyperbola_relation (h : HyperbolaPair) : 4 * h.e₁^2 - h.e₂^2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_relation_l2983_298374


namespace NUMINAMATH_CALUDE_wrong_value_correction_l2983_298380

theorem wrong_value_correction (n : ℕ) (initial_mean correct_mean wrong_value : ℝ) 
  (h1 : n = 20)
  (h2 : initial_mean = 150)
  (h3 : wrong_value = 135)
  (h4 : correct_mean = 151.25) :
  (n : ℝ) * correct_mean - ((n : ℝ) * initial_mean - wrong_value) = 160 := by
  sorry

end NUMINAMATH_CALUDE_wrong_value_correction_l2983_298380


namespace NUMINAMATH_CALUDE_stating_max_squares_correct_max_squares_1000_l2983_298349

/-- 
Represents the maximum number of squares that can be chosen on an m × n chessboard 
such that no three chosen squares have two in the same row and two in the same column.
-/
def max_squares (m n : ℕ) : ℕ :=
  if m = 1 then n else m + n - 2

/-- 
Theorem stating that max_squares gives the correct maximum number of squares
that can be chosen on an m × n chessboard under the given constraints.
-/
theorem max_squares_correct (m n : ℕ) (h : m ≤ n) :
  max_squares m n = 
    if m = 1 
    then n
    else m + n - 2 :=
by sorry

/-- 
Corollary for the specific case of a 1000 × 1000 chessboard.
-/
theorem max_squares_1000 : max_squares 1000 1000 = 1998 :=
by sorry

end NUMINAMATH_CALUDE_stating_max_squares_correct_max_squares_1000_l2983_298349


namespace NUMINAMATH_CALUDE_area_of_triangle_BDE_l2983_298377

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Angle between three points in 3D space -/
def angle (p q r : Point3D) : ℝ := sorry

/-- Check if two lines are parallel in 3D space -/
def parallel_lines (p1 q1 p2 q2 : Point3D) : Prop := sorry

/-- Check if a plane is parallel to a line in 3D space -/
def plane_parallel_to_line (p1 p2 p3 l1 l2 : Point3D) : Prop := sorry

/-- Calculate the area of a triangle given its three vertices -/
def triangle_area (p q r : Point3D) : ℝ := sorry

theorem area_of_triangle_BDE (A B C D E : Point3D)
  (h1 : distance A B = 3)
  (h2 : distance B C = 3)
  (h3 : distance C D = 3)
  (h4 : distance D E = 3)
  (h5 : distance E A = 3)
  (h6 : angle A B C = Real.pi / 2)
  (h7 : angle C D E = Real.pi / 2)
  (h8 : angle D E A = Real.pi / 2)
  (h9 : plane_parallel_to_line A C D B E) :
  triangle_area B D E = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_BDE_l2983_298377


namespace NUMINAMATH_CALUDE_least_of_four_consecutive_integers_with_sum_two_l2983_298369

theorem least_of_four_consecutive_integers_with_sum_two :
  ∀ n : ℤ, (n + (n + 1) + (n + 2) + (n + 3) = 2) → n = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_least_of_four_consecutive_integers_with_sum_two_l2983_298369


namespace NUMINAMATH_CALUDE_journey_distance_l2983_298390

/-- Proves that the total distance of a journey is 35 miles given specific conditions -/
theorem journey_distance (speed : ℝ) (time : ℝ) (total_portions : ℕ) (covered_portions : ℕ) :
  speed = 40 →
  time = 0.7 →
  total_portions = 5 →
  covered_portions = 4 →
  (speed * time) / covered_portions * total_portions = 35 :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l2983_298390


namespace NUMINAMATH_CALUDE_sqrt_115_between_consecutive_integers_product_l2983_298323

theorem sqrt_115_between_consecutive_integers_product :
  ∃ (n : ℕ), n > 0 ∧ (n : ℝ) < Real.sqrt 115 ∧ Real.sqrt 115 < (n + 1) ∧ n * (n + 1) = 110 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_115_between_consecutive_integers_product_l2983_298323


namespace NUMINAMATH_CALUDE_polygon_with_540_degree_sum_is_pentagon_l2983_298305

theorem polygon_with_540_degree_sum_is_pentagon (n : ℕ) (h : n ≥ 3) :
  (n - 2) * 180 = 540 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_polygon_with_540_degree_sum_is_pentagon_l2983_298305


namespace NUMINAMATH_CALUDE_distinct_tetrahedrons_count_l2983_298327

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of vertices required to form a tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of non-tetrahedral configurations -/
def non_tetrahedral_configurations : ℕ := 12

/-- The number of distinct tetrahedrons that can be formed using the vertices of a cube -/
def distinct_tetrahedrons : ℕ :=
  Nat.choose cube_vertices tetrahedron_vertices - non_tetrahedral_configurations

theorem distinct_tetrahedrons_count : distinct_tetrahedrons = 58 := by
  sorry

end NUMINAMATH_CALUDE_distinct_tetrahedrons_count_l2983_298327


namespace NUMINAMATH_CALUDE_range_of_3a_minus_b_l2983_298376

theorem range_of_3a_minus_b (a b : ℝ) 
  (h1 : 1 ≤ a + b) (h2 : a + b ≤ 4) 
  (h3 : -1 ≤ a - b) (h4 : a - b ≤ 2) : 
  (∃ (x y : ℝ), (1 ≤ x + y ∧ x + y ≤ 4 ∧ -1 ≤ x - y ∧ x - y ≤ 2 ∧ 3*x - y = -1)) ∧
  (∃ (x y : ℝ), (1 ≤ x + y ∧ x + y ≤ 4 ∧ -1 ≤ x - y ∧ x - y ≤ 2 ∧ 3*x - y = 8)) ∧
  (∀ (x y : ℝ), 1 ≤ x + y → x + y ≤ 4 → -1 ≤ x - y → x - y ≤ 2 → -1 ≤ 3*x - y ∧ 3*x - y ≤ 8) :=
by sorry

end NUMINAMATH_CALUDE_range_of_3a_minus_b_l2983_298376


namespace NUMINAMATH_CALUDE_problem_statement_l2983_298342

theorem problem_statement (x : ℝ) (h : x + 2/x = 4) :
  -5*x / (x^2 + 2) = -5/4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2983_298342


namespace NUMINAMATH_CALUDE_polynomial_equality_l2983_298361

-- Define the polynomial h(x)
def h : Polynomial ℝ := sorry

-- State the theorem
theorem polynomial_equality :
  4 * X^5 + 5 * X^3 - 3 * X + h = 2 * X^3 - 4 * X^2 + 9 * X + 2 →
  h = -4 * X^5 - 3 * X^3 - 4 * X^2 + 12 * X + 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2983_298361


namespace NUMINAMATH_CALUDE_problem_solution_l2983_298371

theorem problem_solution (A B : ℝ) 
  (h1 : A + 2 * B = 814.8)
  (h2 : A = 10 * B) : 
  A - B = 611.1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2983_298371


namespace NUMINAMATH_CALUDE_sixtieth_point_coordinates_l2983_298312

/-- Represents a point with integer coordinates -/
structure Point where
  x : ℕ
  y : ℕ

/-- The sequence of points -/
def pointSequence : ℕ → Point := sorry

/-- The sum of x and y coordinates for the nth point -/
def coordinateSum (n : ℕ) : ℕ := (pointSequence n).x + (pointSequence n).y

/-- The row number for a given point in the sequence -/
def rowNumber (n : ℕ) : ℕ := sorry

/-- The property that the coordinate sum increases by 1 for every n points -/
axiom coordinate_sum_property (n : ℕ) :
  ∀ k, k > n → coordinateSum k = coordinateSum n + (rowNumber k - rowNumber n)

/-- The main theorem: The 60th point has coordinates (5,7) -/
theorem sixtieth_point_coordinates :
  pointSequence 60 = Point.mk 5 7 := by sorry

end NUMINAMATH_CALUDE_sixtieth_point_coordinates_l2983_298312


namespace NUMINAMATH_CALUDE_oil_price_reduction_l2983_298351

/-- Calculates the additional amount of oil a housewife can obtain after a price reduction --/
theorem oil_price_reduction (original_price reduced_price budget : ℝ) : 
  original_price > 0 →
  reduced_price > 0 →
  budget > 0 →
  reduced_price = original_price * (1 - 0.35) →
  reduced_price = 56 →
  budget = 800 →
  let additional_amount := budget / reduced_price - budget / original_price
  ∃ ε > 0, |additional_amount - 5.01| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_oil_price_reduction_l2983_298351


namespace NUMINAMATH_CALUDE_final_parity_after_odd_changes_not_even_after_33_changes_l2983_298345

/-- Represents the parity of a number -/
inductive Parity
  | Even
  | Odd

/-- Function to change the parity -/
def changeParity (p : Parity) : Parity :=
  match p with
  | Parity.Even => Parity.Odd
  | Parity.Odd => Parity.Even

/-- Function to apply n changes to initial parity -/
def applyNChanges (initial : Parity) (n : Nat) : Parity :=
  match n with
  | 0 => initial
  | k + 1 => changeParity (applyNChanges initial k)

theorem final_parity_after_odd_changes 
  (initial : Parity) (n : Nat) (h : Odd n) :
  applyNChanges initial n ≠ initial := by
  sorry

/-- Main theorem: After 33 changes, an initially even number cannot be even -/
theorem not_even_after_33_changes :
  applyNChanges Parity.Even 33 ≠ Parity.Even := by
  sorry

end NUMINAMATH_CALUDE_final_parity_after_odd_changes_not_even_after_33_changes_l2983_298345


namespace NUMINAMATH_CALUDE_distance_to_line_l2983_298317

/-- The distance from a point in polar coordinates to a line in polar form -/
def distance_polar_to_line (m : ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  |m - 2|

/-- The theorem stating the distance from the point (m, π/3) to the line ρcos(θ - π/3) = 2 -/
theorem distance_to_line (m : ℝ) (h : m > 0) :
  distance_polar_to_line m (fun ρ θ ↦ ρ * Real.cos (θ - Real.pi / 3) = 2) = |m - 2| := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l2983_298317


namespace NUMINAMATH_CALUDE_concentric_circles_chord_theorem_l2983_298370

/-- Represents two concentric circles with chords of the outer circle tangent to the inner circle -/
structure ConcentricCircles where
  outer : ℝ → ℝ → Prop
  inner : ℝ → ℝ → Prop
  is_concentric : Prop
  tangent_chords : Prop

/-- The angle between two adjacent chords -/
def chord_angle (c : ConcentricCircles) : ℝ := 60

/-- The number of chords needed to complete a full circle -/
def num_chords (c : ConcentricCircles) : ℕ := 3

theorem concentric_circles_chord_theorem (c : ConcentricCircles) :
  chord_angle c = 60 → num_chords c = 3 := by sorry

end NUMINAMATH_CALUDE_concentric_circles_chord_theorem_l2983_298370


namespace NUMINAMATH_CALUDE_function_symmetry_l2983_298311

def is_symmetric_about_one (g : ℝ → ℝ) : Prop :=
  ∀ x, g (1 - x) = g (1 + x)

theorem function_symmetry 
  (f : ℝ → ℝ) 
  (h1 : f 0 = 0)
  (h2 : ∀ x, f (-x) = f x)
  (h3 : ∀ t, f (1 - t) - f (1 + t) + 4 * t = 0) :
  is_symmetric_about_one (λ x => f x - 2 * x) := by
sorry

end NUMINAMATH_CALUDE_function_symmetry_l2983_298311


namespace NUMINAMATH_CALUDE_basketball_games_count_l2983_298365

theorem basketball_games_count : ∃ (x : ℕ), 
  x > 0 ∧ 
  x ∣ 60 ∧ 
  (3 * x / 5 : ℚ) = ⌊(3 * x / 5 : ℚ)⌋ ∧
  (7 * (x + 10) / 12 : ℚ) = ⌊(7 * (x + 10) / 12 : ℚ)⌋ ∧
  (7 * (x + 10) / 12 : ℕ) = (3 * x / 5 : ℕ) + 5 ∧
  x = 60 := by
  sorry

end NUMINAMATH_CALUDE_basketball_games_count_l2983_298365


namespace NUMINAMATH_CALUDE_equal_cost_sharing_l2983_298306

theorem equal_cost_sharing (A B : ℝ) (h : A < B) :
  (B - A) / 2 = (A + B) / 2 - A := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_sharing_l2983_298306


namespace NUMINAMATH_CALUDE_most_accurate_value_for_given_K_l2983_298324

/-- Given a scientific constant K and its error margin, 
    returns the most accurate value with all digits significant -/
def most_accurate_value (K : ℝ) (error : ℝ) : ℝ :=
  sorry

theorem most_accurate_value_for_given_K :
  let K : ℝ := 3.68547
  let error : ℝ := 0.00256
  most_accurate_value K error = 3.7 := by sorry

end NUMINAMATH_CALUDE_most_accurate_value_for_given_K_l2983_298324


namespace NUMINAMATH_CALUDE_joan_balloon_count_l2983_298397

/-- Given an initial count of balloons and a number of lost balloons, 
    calculate the final count of balloons. -/
def final_balloon_count (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that with 8 initial balloons and 2 lost balloons, 
    the final count is 6. -/
theorem joan_balloon_count : final_balloon_count 8 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_balloon_count_l2983_298397


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2983_298385

-- Define a type for lines
variable (Line : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem parallel_transitivity (l1 l2 l3 : Line) :
  parallel l1 l3 → parallel l2 l3 → parallel l1 l2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l2983_298385


namespace NUMINAMATH_CALUDE_max_lateral_area_triangular_prism_l2983_298330

/-- The maximum lateral area of a triangular prism inscribed in a sphere -/
theorem max_lateral_area_triangular_prism (r : ℝ) (h : r = 2) :
  ∃ (a h : ℝ),
    -- Condition: prism inscribed in sphere
    4 * a^2 + 3 * h^2 = 48 ∧
    -- Condition: lateral area
    (3 : ℝ) * a * h ≤ 12 * Real.sqrt 3 ∧
    -- Condition: maximum value
    ∀ (a' h' : ℝ), 4 * a'^2 + 3 * h'^2 = 48 → (3 : ℝ) * a' * h' ≤ 12 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_max_lateral_area_triangular_prism_l2983_298330


namespace NUMINAMATH_CALUDE_total_marbles_count_l2983_298344

/-- Represents the colors of marbles in the bag -/
inductive MarbleColor
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents the bag of marbles -/
structure MarbleBag where
  red : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- The ratio of marbles in the bag -/
def marbleRatio : MarbleBag := {
  red := 2,
  blue := 4,
  green := 3,
  yellow := 1
}

/-- The number of green marbles in the bag -/
def greenMarbleCount : ℕ := 24

/-- Theorem stating the total number of marbles in the bag -/
theorem total_marbles_count (bag : MarbleBag) 
  (h1 : bag.red = 2 * bag.green / 3)
  (h2 : bag.blue = 4 * bag.green / 3)
  (h3 : bag.yellow = bag.green / 3)
  (h4 : bag.green = greenMarbleCount) :
  bag.red + bag.blue + bag.green + bag.yellow = 80 := by
  sorry

#check total_marbles_count

end NUMINAMATH_CALUDE_total_marbles_count_l2983_298344


namespace NUMINAMATH_CALUDE_seven_twentyfour_twentyfive_pythagorean_triple_l2983_298358

/-- A Pythagorean triple consists of three positive integers a, b, and c that satisfy a² + b² = c² -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- Prove that 7, 24, and 25 form a Pythagorean triple -/
theorem seven_twentyfour_twentyfive_pythagorean_triple :
  is_pythagorean_triple 7 24 25 := by
sorry

end NUMINAMATH_CALUDE_seven_twentyfour_twentyfive_pythagorean_triple_l2983_298358


namespace NUMINAMATH_CALUDE_square_area_after_cut_l2983_298315

theorem square_area_after_cut (x : ℝ) : 
  x > 0 →  -- ensure x is positive
  x^2 = 2*x + 80 → -- equation from the problem
  x^2 = 100 := by sorry

end NUMINAMATH_CALUDE_square_area_after_cut_l2983_298315


namespace NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l2983_298367

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  daughtersWithChildren : ℕ
  totalDescendants : ℕ

/-- The actual Bertha family configuration -/
def berthaActual : BerthaFamily :=
  { daughters := 8,
    daughtersWithChildren := 7,  -- This is derived, not given directly
    totalDescendants := 36 }

/-- Theorem stating the number of daughters and granddaughters without children -/
theorem daughters_and_granddaughters_without_children
  (b : BerthaFamily)
  (h1 : b.daughters = berthaActual.daughters)
  (h2 : b.totalDescendants = berthaActual.totalDescendants)
  (h3 : ∀ d, d ≤ b.daughters → (d = b.daughtersWithChildren ∨ d = b.daughters - b.daughtersWithChildren))
  (h4 : b.totalDescendants = b.daughters + 4 * b.daughtersWithChildren) :
  b.daughters - b.daughtersWithChildren + (b.totalDescendants - b.daughters) = 29 := by
  sorry

end NUMINAMATH_CALUDE_daughters_and_granddaughters_without_children_l2983_298367


namespace NUMINAMATH_CALUDE_direct_proportion_relationship_l2983_298325

theorem direct_proportion_relationship (x y : ℝ) :
  (∃ k : ℝ, ∀ x, y - 2 = k * x) →  -- y-2 is directly proportional to x
  (1 = 1 ∧ y = -6) →              -- when x=1, y=-6
  y = -8 * x + 2 :=                -- relationship between y and x
by
  sorry

end NUMINAMATH_CALUDE_direct_proportion_relationship_l2983_298325


namespace NUMINAMATH_CALUDE_student_distribution_l2983_298336

/-- The number of ways to distribute n students to k universities --/
def distribute (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The condition that each university admits at least one student --/
def at_least_one (n : ℕ) (k : ℕ) : Prop :=
  sorry

theorem student_distribution :
  ∃ (d : ℕ → ℕ → ℕ), ∃ (c : ℕ → ℕ → Prop),
    d 5 3 = 150 ∧ c 5 3 ∧
    ∀ (n k : ℕ), c n k → d n k = distribute n k :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_l2983_298336


namespace NUMINAMATH_CALUDE_solution_difference_l2983_298320

theorem solution_difference : ∃ (x₁ x₂ : ℝ),
  (x₁^(1/3 : ℝ) = -3 ∧ 9 - x₁^2 / 4 = (-3)^3) ∧
  (x₂^(1/3 : ℝ) = -3 ∧ 9 - x₂^2 / 4 = (-3)^3) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 24 :=
by sorry

end NUMINAMATH_CALUDE_solution_difference_l2983_298320


namespace NUMINAMATH_CALUDE_narration_per_disc_l2983_298304

/-- Represents the duration of the narration in minutes -/
def narration_duration : ℕ := 6 * 60 + 45

/-- Represents the capacity of each disc in minutes -/
def disc_capacity : ℕ := 75

/-- Calculates the minimum number of discs needed -/
def min_discs : ℕ := (narration_duration + disc_capacity - 1) / disc_capacity

/-- Theorem stating the duration of narration on each disc -/
theorem narration_per_disc :
  (narration_duration : ℚ) / min_discs = 67.5 := by sorry

end NUMINAMATH_CALUDE_narration_per_disc_l2983_298304


namespace NUMINAMATH_CALUDE_exactly_two_cubic_polynomials_satisfy_l2983_298337

/-- A polynomial function of degree 3 or less -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a*x^3 + b*x^2 + c*x + d

/-- The condition that f(x)f(-x) = f(x^3) for all x -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The main theorem stating that exactly two cubic polynomials satisfy the condition -/
theorem exactly_two_cubic_polynomials_satisfy :
  ∃! (s : Finset (ℝ → ℝ)),
    (∀ f ∈ s, ∃ a b c d, f = CubicPolynomial a b c d) ∧
    (∀ f ∈ s, SatisfiesCondition f) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_cubic_polynomials_satisfy_l2983_298337


namespace NUMINAMATH_CALUDE_brother_cousin_age_difference_l2983_298362

/-- Represents the ages of family members -/
structure FamilyAges where
  lexie : ℕ
  brother : ℕ
  sister : ℕ
  grandma : ℕ
  uncle : ℕ
  cousin : ℕ

/-- Defines the relationships between family members' ages -/
def valid_family_ages (ages : FamilyAges) : Prop :=
  ages.lexie = 8 ∧
  ages.grandma = 68 ∧
  ages.lexie = ages.brother + 6 ∧
  ages.sister = 2 * ages.lexie ∧
  ages.uncle = ages.grandma - 12 ∧
  ages.uncle = 3 * ages.sister ∧
  ages.cousin = ages.brother + 5 ∧
  ages.uncle = ages.cousin + 2

/-- Theorem stating the age difference between Lexie's brother and cousin -/
theorem brother_cousin_age_difference (ages : FamilyAges) 
  (h : valid_family_ages ages) : ages.cousin - ages.brother = 5 := by
  sorry

end NUMINAMATH_CALUDE_brother_cousin_age_difference_l2983_298362


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2983_298321

theorem quadratic_factorization (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2983_298321


namespace NUMINAMATH_CALUDE_music_festival_group_formation_l2983_298373

def total_friends : ℕ := 10
def musicians : ℕ := 4
def non_musicians : ℕ := 6
def group_size : ℕ := 4

theorem music_festival_group_formation :
  (Nat.choose total_friends group_size) - (Nat.choose non_musicians group_size) = 195 :=
sorry

end NUMINAMATH_CALUDE_music_festival_group_formation_l2983_298373


namespace NUMINAMATH_CALUDE_friendly_point_sum_l2983_298387

/-- Friendly point transformation in 2D plane -/
def friendly_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2 - 1, -p.1 - 1)

/-- Sequence of friendly points -/
def friendly_sequence (start : ℝ × ℝ) : ℕ → ℝ × ℝ
| 0 => start
| n + 1 => friendly_point (friendly_sequence start n)

theorem friendly_point_sum (x y : ℝ) :
  friendly_sequence (x, y) 2022 = (-3, -2) →
  x + y = 3 :=
by sorry

end NUMINAMATH_CALUDE_friendly_point_sum_l2983_298387


namespace NUMINAMATH_CALUDE_fish_population_calculation_l2983_298392

/-- Calculates the number of fish in a lake on May 1 based on sampling data --/
theorem fish_population_calculation (tagged_may : ℕ) (caught_sept : ℕ) (tagged_sept : ℕ) 
  (death_rate : ℚ) (new_fish_rate : ℚ) :
  tagged_may = 60 →
  caught_sept = 70 →
  tagged_sept = 3 →
  death_rate = 1/4 →
  new_fish_rate = 2/5 →
  (1 - death_rate) * tagged_may * caught_sept / tagged_sept * (1 - new_fish_rate) = 630 := by
  sorry

end NUMINAMATH_CALUDE_fish_population_calculation_l2983_298392


namespace NUMINAMATH_CALUDE_white_marbles_count_l2983_298366

theorem white_marbles_count (total : ℕ) (black red green : ℕ) 
  (h_total : total = 60)
  (h_black : black = 32)
  (h_red : red = 10)
  (h_green : green = 5)
  (h_sum : total = black + red + green + (total - (black + red + green))) :
  total - (black + red + green) = 13 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_count_l2983_298366


namespace NUMINAMATH_CALUDE_green_blue_difference_l2983_298335

/-- Represents the colors of disks in the bag -/
inductive DiskColor
  | Blue
  | Yellow
  | Green

/-- Represents the bag of disks -/
structure DiskBag where
  total : ℕ
  ratio : Fin 3 → ℕ
  color_sum : ratio 0 + ratio 1 + ratio 2 = 18

theorem green_blue_difference (bag : DiskBag) 
  (h1 : bag.total = 108)
  (h2 : bag.ratio 0 = 3)  -- Blue
  (h3 : bag.ratio 1 = 7)  -- Yellow
  (h4 : bag.ratio 2 = 8)  -- Green
  : (bag.total / 18 * bag.ratio 2) - (bag.total / 18 * bag.ratio 0) = 30 := by
  sorry

end NUMINAMATH_CALUDE_green_blue_difference_l2983_298335


namespace NUMINAMATH_CALUDE_unique_solutions_for_exponential_equation_l2983_298395

theorem unique_solutions_for_exponential_equation :
  ∀ x n : ℕ+, 3 * 2^(x : ℕ) + 4 = (n : ℕ)^2 ↔ (x = 2 ∧ n = 4) ∨ (x = 5 ∧ n = 10) ∨ (x = 6 ∧ n = 14) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_for_exponential_equation_l2983_298395


namespace NUMINAMATH_CALUDE_batsman_average_l2983_298352

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  runsLastInning : ℕ
  averageIncrease : ℕ

/-- Calculates the new average of a batsman after their last inning -/
def newAverage (b : Batsman) : ℕ :=
  b.averageIncrease + (b.innings - 1) * b.averageIncrease + b.runsLastInning

/-- Theorem stating that given the conditions, the batsman's new average is 140 -/
theorem batsman_average (b : Batsman) 
  (h1 : b.innings = 17) 
  (h2 : b.runsLastInning = 300) 
  (h3 : b.averageIncrease = 10) : 
  newAverage b = 140 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_l2983_298352


namespace NUMINAMATH_CALUDE_f_of_9_eq_836_l2983_298398

/-- The function f(n) = n^3 + n^2 + n + 17 -/
def f (n : ℕ) : ℕ := n^3 + n^2 + n + 17

/-- Theorem: The value of f(9) is 836 -/
theorem f_of_9_eq_836 : f 9 = 836 := by sorry

end NUMINAMATH_CALUDE_f_of_9_eq_836_l2983_298398


namespace NUMINAMATH_CALUDE_james_payment_l2983_298393

/-- Calculates James's payment at a restaurant given meal prices and tip percentage. -/
theorem james_payment (james_meal : ℝ) (friend_meal : ℝ) (tip_percentage : ℝ) : 
  james_meal = 16 →
  friend_meal = 14 →
  tip_percentage = 0.2 →
  james_meal + 0.5 * friend_meal + 0.5 * tip_percentage * (james_meal + friend_meal) = 19 :=
by sorry

end NUMINAMATH_CALUDE_james_payment_l2983_298393


namespace NUMINAMATH_CALUDE_floor_sqrt_12_squared_l2983_298378

theorem floor_sqrt_12_squared : ⌊Real.sqrt 12⌋^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_12_squared_l2983_298378


namespace NUMINAMATH_CALUDE_congruence_solution_l2983_298360

theorem congruence_solution :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -437 [ZMOD 10] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l2983_298360


namespace NUMINAMATH_CALUDE_horner_v4_equals_3_l2983_298314

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^7 - 2x^6 + 3x^3 - 4x^2 + 1 -/
def f (x : ℝ) : ℝ :=
  x^7 - 2*x^6 + 3*x^3 - 4*x^2 + 1

/-- v_4 in Horner's method for f(x) -/
def v_4 (x : ℝ) : ℝ :=
  (((x - 2) * x + 0) * x + 0) * x + 3

theorem horner_v4_equals_3 :
  v_4 2 = 3 := by sorry

#check horner_v4_equals_3

end NUMINAMATH_CALUDE_horner_v4_equals_3_l2983_298314


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l2983_298348

theorem product_from_lcm_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 45) 
  (h2 : Nat.gcd a b = 9) : 
  a * b = 405 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l2983_298348


namespace NUMINAMATH_CALUDE_peter_is_18_l2983_298328

-- Define Peter's current age
def peter_current_age : ℕ := sorry

-- Define Ivan's current age
def ivan_current_age : ℕ := sorry

-- Define Peter's past age when Ivan was Peter's current age
def peter_past_age : ℕ := sorry

-- Condition 1: Ivan's current age is twice Peter's past age
axiom ivan_age_relation : ivan_current_age = 2 * peter_past_age

-- Condition 2: Sum of their ages will be 54 when Peter reaches Ivan's current age
axiom future_age_sum : ivan_current_age + ivan_current_age = 54

-- Condition 3: The time difference between Peter's current age and past age
-- is equal to the time difference between Ivan's current age and Peter's current age
axiom age_difference_relation : ivan_current_age - peter_current_age = peter_current_age - peter_past_age

theorem peter_is_18 : peter_current_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_peter_is_18_l2983_298328


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l2983_298353

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem consecutive_odd_numbers_equation (N₁ N₂ N₃ : ℤ) : 
  is_odd N₁ ∧ is_odd N₂ ∧ is_odd N₃ ∧ 
  N₂ = N₁ + 2 ∧ N₃ = N₂ + 2 ∧
  N₁ = 9 →
  N₁ ≠ 3 * N₃ + 16 + 4 * N₂ :=
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_equation_l2983_298353


namespace NUMINAMATH_CALUDE_northwest_molded_break_even_price_l2983_298383

/-- Calculate the break-even price per handle for Northwest Molded -/
theorem northwest_molded_break_even_price 
  (variable_cost : ℝ) 
  (fixed_cost : ℝ) 
  (break_even_quantity : ℝ) :
  variable_cost = 0.60 →
  fixed_cost = 7640 →
  break_even_quantity = 1910 →
  (fixed_cost + variable_cost * break_even_quantity) / break_even_quantity = 4.60 :=
by sorry

end NUMINAMATH_CALUDE_northwest_molded_break_even_price_l2983_298383


namespace NUMINAMATH_CALUDE_count_less_than_ten_l2983_298303

def travel_times : List Nat := [10, 12, 15, 6, 3, 8, 9]

def less_than_ten (n : Nat) : Bool := n < 10

theorem count_less_than_ten :
  (travel_times.filter less_than_ten).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_count_less_than_ten_l2983_298303


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l2983_298394

theorem quadratic_real_roots (a : ℝ) : 
  (∃ x : ℝ, x^2 + a*x + (a+1)^2 = 0) ↔ (a ∈ Set.Icc (-2) (-2/3) ∧ a ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l2983_298394


namespace NUMINAMATH_CALUDE_distance_to_soccer_is_12_l2983_298363

-- Define the distances and costs
def distance_to_grocery : ℝ := 8
def distance_to_school : ℝ := 6
def miles_per_gallon : ℝ := 25
def cost_per_gallon : ℝ := 2.5
def total_gas_cost : ℝ := 5

-- Define the unknown distance to soccer practice
def distance_to_soccer : ℝ → ℝ := λ x => x

-- Define the total distance driven
def total_distance (x : ℝ) : ℝ :=
  distance_to_grocery + distance_to_school + distance_to_soccer x + 2 * distance_to_soccer x

-- Theorem stating that the distance to soccer practice is 12 miles
theorem distance_to_soccer_is_12 :
  ∃ x : ℝ, distance_to_soccer x = 12 ∧ 
    total_distance x = (total_gas_cost / cost_per_gallon) * miles_per_gallon := by
  sorry

end NUMINAMATH_CALUDE_distance_to_soccer_is_12_l2983_298363


namespace NUMINAMATH_CALUDE_geese_count_l2983_298372

/-- The number of ducks in the marsh -/
def ducks : ℕ := 37

/-- The total number of birds in the marsh -/
def total_birds : ℕ := 95

/-- The number of geese in the marsh -/
def geese : ℕ := total_birds - ducks

theorem geese_count : geese = 58 := by sorry

end NUMINAMATH_CALUDE_geese_count_l2983_298372


namespace NUMINAMATH_CALUDE_system_solution_l2983_298368

-- Define the two equations
def equation1 (x y : ℝ) : Prop :=
  8 * x^2 - 26 * x * y + 15 * y^2 + 116 * x - 150 * y + 360 = 0

def equation2 (x y : ℝ) : Prop :=
  8 * x^2 + 18 * x * y - 18 * y^2 + 60 * x + 45 * y + 108 = 0

-- Define the solution set
def solutions : Set (ℝ × ℝ) :=
  {(0, 4), (-7.5, 1), (-4.5, 0)}

-- Theorem statement
theorem system_solution :
  ∀ x y : ℝ, (equation1 x y ∧ equation2 x y) ↔ (x, y) ∈ solutions := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2983_298368


namespace NUMINAMATH_CALUDE_a_minus_b_and_c_linearly_dependent_l2983_298364

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (e₁ e₂ : V)

/-- e₁ and e₂ are not collinear -/
axiom not_collinear : ¬ ∃ (r : ℝ), e₁ = r • e₂

/-- Definition of vector a -/
def a : V := 2 • e₁ - e₂

/-- Definition of vector b -/
def b : V := e₁ + 2 • e₂

/-- Definition of vector c -/
def c : V := (1/2) • e₁ - (3/2) • e₂

/-- Theorem stating that (a - b) and c are linearly dependent -/
theorem a_minus_b_and_c_linearly_dependent :
  ∃ (r s : ℝ) (hs : s ≠ 0), r • (a e₁ e₂ - b e₁ e₂) + s • c e₁ e₂ = 0 :=
sorry

end NUMINAMATH_CALUDE_a_minus_b_and_c_linearly_dependent_l2983_298364


namespace NUMINAMATH_CALUDE_divisibility_by_1956_l2983_298319

theorem divisibility_by_1956 (n : ℕ) (h : Odd n) :
  ∃ k : ℤ, 24 * 80^n + 1992 * 83^(n-1) = 1956 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_1956_l2983_298319


namespace NUMINAMATH_CALUDE_largest_n_for_perfect_square_l2983_298381

theorem largest_n_for_perfect_square (n : ℕ) : 
  (∃ k : ℕ, 4^27 + 4^500 + 4^n = k^2) → n ≤ 972 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_for_perfect_square_l2983_298381


namespace NUMINAMATH_CALUDE_stream_current_is_six_l2983_298391

/-- Represents the man's rowing scenario -/
structure RowingScenario where
  r : ℝ  -- man's usual rowing speed in still water (miles per hour)
  w : ℝ  -- speed of the stream's current (miles per hour)

/-- The conditions of the rowing problem -/
def rowing_conditions (s : RowingScenario) : Prop :=
  -- Downstream time is 6 hours less than upstream time
  18 / (s.r + s.w) + 6 = 18 / (s.r - s.w) ∧
  -- When rowing speed is tripled, downstream time is 2 hours less than upstream time
  18 / (3 * s.r + s.w) + 2 = 18 / (3 * s.r - s.w)

/-- The theorem stating that the stream's current is 6 miles per hour -/
theorem stream_current_is_six (s : RowingScenario) :
  rowing_conditions s → s.w = 6 := by
  sorry

end NUMINAMATH_CALUDE_stream_current_is_six_l2983_298391


namespace NUMINAMATH_CALUDE_problem_solution_l2983_298310

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

def sum_probability (n : ℕ) : ℚ := (3 : ℚ) / choose n 2

def binomial_coefficient (n k : ℕ) : ℤ := (choose n k : ℤ)

def a (n k : ℕ) : ℤ := binomial_coefficient n k * (-2)^k

theorem problem_solution :
  ∃ (n : ℕ),
    (sum_probability n = 3/28) ∧
    (a 8 3 = -448) ∧
    (((choose 5 2 * choose 4 1 + choose 4 3) : ℚ) / choose 9 3 = 11/21) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2983_298310


namespace NUMINAMATH_CALUDE_coffee_shop_sales_l2983_298389

theorem coffee_shop_sales (teas lattes extra : ℕ) : 
  teas = 6 → lattes = 32 → 4 * teas + extra = lattes → extra = 8 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_sales_l2983_298389


namespace NUMINAMATH_CALUDE_daily_sales_volume_selling_price_for_profit_daily_sales_profit_and_max_l2983_298318

-- Define the variables and constants
variable (x : ℝ) -- Selling price in yuan
variable (y : ℝ) -- Daily sales volume in items
variable (w : ℝ) -- Daily sales profit in yuan

-- Define the given conditions
def cost_price : ℝ := 6
def min_price : ℝ := 6
def max_price : ℝ := 12
def base_price : ℝ := 8
def base_volume : ℝ := 200
def volume_change_rate : ℝ := 10

-- Theorem 1: Daily sales volume function
theorem daily_sales_volume : 
  ∀ x, min_price ≤ x ∧ x ≤ max_price → y = -volume_change_rate * x + (base_volume + volume_change_rate * base_price) :=
sorry

-- Theorem 2: Selling price for specific profit
theorem selling_price_for_profit (target_profit : ℝ) : 
  ∃ x, min_price ≤ x ∧ x ≤ max_price ∧ 
  (x - cost_price) * (-volume_change_rate * x + (base_volume + volume_change_rate * base_price)) = target_profit :=
sorry

-- Theorem 3: Daily sales profit function and maximum profit
theorem daily_sales_profit_and_max : 
  ∃ w_max : ℝ,
  (∀ x, min_price ≤ x ∧ x ≤ max_price → 
    w = -volume_change_rate * (x - 11)^2 + 1210) ∧
  (w_max = -volume_change_rate * (max_price - 11)^2 + 1210) ∧
  (∀ x, min_price ≤ x ∧ x ≤ max_price → w ≤ w_max) :=
sorry

end NUMINAMATH_CALUDE_daily_sales_volume_selling_price_for_profit_daily_sales_profit_and_max_l2983_298318


namespace NUMINAMATH_CALUDE_travel_equation_correct_l2983_298334

/-- Represents the scenario of Confucius and his students traveling to a school -/
structure TravelScenario where
  distance : ℝ
  student_speed : ℝ
  cart_speed_multiplier : ℝ
  head_start : ℝ

/-- The equation representing the travel times is correct for the given scenario -/
theorem travel_equation_correct (scenario : TravelScenario) 
  (h_distance : scenario.distance = 30)
  (h_cart_speed : scenario.cart_speed_multiplier = 1.5)
  (h_head_start : scenario.head_start = 1)
  (h_student_speed_pos : scenario.student_speed > 0) :
  scenario.distance / scenario.student_speed = 
    scenario.distance / (scenario.cart_speed_multiplier * scenario.student_speed) + scenario.head_start :=
sorry

end NUMINAMATH_CALUDE_travel_equation_correct_l2983_298334


namespace NUMINAMATH_CALUDE_inradius_exradius_inequality_l2983_298313

/-- Given a triangle ABC with inradius r, exradius r' touching side AB, and length c of side AB,
    prove that 4rr' ≤ c^2 -/
theorem inradius_exradius_inequality (r r' c : ℝ) (hr : r > 0) (hr' : r' > 0) (hc : c > 0) :
  4 * r * r' ≤ c^2 := by
  sorry

end NUMINAMATH_CALUDE_inradius_exradius_inequality_l2983_298313


namespace NUMINAMATH_CALUDE_hockey_league_teams_l2983_298301

/-- The number of games played in the hockey season -/
def total_games : ℕ := 1710

/-- The number of times each team faces every other team -/
def games_per_pair : ℕ := 10

/-- Calculates the total number of games in a season based on the number of teams -/
def calculate_games (n : ℕ) : ℕ :=
  (n * (n - 1) * games_per_pair) / 2

theorem hockey_league_teams :
  ∃ (n : ℕ), n > 0 ∧ calculate_games n = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l2983_298301


namespace NUMINAMATH_CALUDE_summer_mowing_count_l2983_298308

/-- The number of times Ned mowed his lawn in the spring -/
def spring_mows : ℕ := 6

/-- The total number of times Ned mowed his lawn -/
def total_mows : ℕ := 11

/-- The number of times Ned mowed his lawn in the summer -/
def summer_mows : ℕ := total_mows - spring_mows

theorem summer_mowing_count : summer_mows = 5 := by
  sorry

end NUMINAMATH_CALUDE_summer_mowing_count_l2983_298308


namespace NUMINAMATH_CALUDE_slope_at_negative_five_l2983_298331

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem slope_at_negative_five
  (f : ℝ → ℝ)
  (h_even : is_even f)
  (h_diff : Differentiable ℝ f)
  (h_der_one : deriv f 1 = 1)
  (h_period : has_period f 4) :
  deriv f (-5) = -1 := by
  sorry

end NUMINAMATH_CALUDE_slope_at_negative_five_l2983_298331


namespace NUMINAMATH_CALUDE_triangle_value_l2983_298347

theorem triangle_value (triangle p : ℤ) 
  (h1 : triangle + p = 75)
  (h2 : 3 * (triangle + p) - p = 198) : 
  triangle = 48 := by
sorry

end NUMINAMATH_CALUDE_triangle_value_l2983_298347


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l2983_298326

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequalities :
  ∀ a b c : ℝ,
  -- Part 1
  (∀ x : ℝ, -3 < x → x < 4 → f a b c x > 0) →
  (∀ x : ℝ, -3 < x → x < 5 → b * x^2 + 2 * a * x - (c + 3 * b) < 0) ∧
  -- Part 2
  (b = 2 → a > c → (∀ x : ℝ, f a b c x ≥ 0) → (∃ x₀ : ℝ, f a b c x₀ = 0) →
    ∃ min : ℝ, min = 2 * Real.sqrt 2 ∧ ∀ x : ℝ, (a^2 + c^2) / (a - c) ≥ min) ∧
  -- Part 3
  (a < b → (∀ x : ℝ, f a b c x ≥ 0) →
    ∃ min : ℝ, min = 8 ∧ ∀ x : ℝ, (a + 2 * b + 4 * c) / (b - a) ≥ min) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l2983_298326


namespace NUMINAMATH_CALUDE_det_of_specific_matrix_l2983_298316

theorem det_of_specific_matrix : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![1, 2; 4, 7]
  Matrix.det A = -1 := by
sorry

end NUMINAMATH_CALUDE_det_of_specific_matrix_l2983_298316


namespace NUMINAMATH_CALUDE_nineteen_in_base_three_l2983_298396

theorem nineteen_in_base_three : 
  (2 * 3^2 + 0 * 3^1 + 1 * 3^0) = 19 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_in_base_three_l2983_298396


namespace NUMINAMATH_CALUDE_net_change_in_cards_l2983_298359

def sold_cards : ℤ := 27
def received_cards : ℤ := 41
def bought_cards : ℤ := 20

theorem net_change_in_cards : -sold_cards + received_cards + bought_cards = 34 := by
  sorry

end NUMINAMATH_CALUDE_net_change_in_cards_l2983_298359


namespace NUMINAMATH_CALUDE_trip_time_calculation_l2983_298300

theorem trip_time_calculation (normal_distance : ℝ) (normal_time : ℝ) (additional_distance : ℝ) :
  normal_distance = 150 →
  normal_time = 3 →
  additional_distance = 100 →
  let speed := normal_distance / normal_time
  let total_distance := normal_distance + additional_distance
  let total_time := total_distance / speed
  total_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_trip_time_calculation_l2983_298300


namespace NUMINAMATH_CALUDE_fox_jeans_purchased_l2983_298384

/-- Represents the problem of determining the number of Fox jeans purchased during a sale. -/
theorem fox_jeans_purchased (fox_price pony_price total_savings total_jeans pony_jeans sum_discount_rates pony_discount : ℝ) 
  (h1 : fox_price = 15)
  (h2 : pony_price = 20)
  (h3 : total_savings = 9)
  (h4 : total_jeans = 5)
  (h5 : pony_jeans = 2)
  (h6 : sum_discount_rates = 0.22)
  (h7 : pony_discount = 0.18000000000000014) :
  ∃ fox_jeans : ℝ, fox_jeans = 3 ∧ 
  fox_jeans + pony_jeans = total_jeans ∧
  fox_jeans * (fox_price * (sum_discount_rates - pony_discount)) + 
  pony_jeans * (pony_price * pony_discount) = total_savings :=
sorry

end NUMINAMATH_CALUDE_fox_jeans_purchased_l2983_298384


namespace NUMINAMATH_CALUDE_negative_fraction_comparison_l2983_298322

theorem negative_fraction_comparison : -5/6 > -6/7 := by
  sorry

end NUMINAMATH_CALUDE_negative_fraction_comparison_l2983_298322


namespace NUMINAMATH_CALUDE_transformation_possible_l2983_298332

/-- Represents a move that can be applied to a sequence of numbers. -/
inductive Move
  | RotateThree (x y z : ℕ) : Move
  | SwapTwo (x y : ℕ) : Move

/-- Checks if a move is valid according to the rules. -/
def isValidMove (move : Move) : Prop :=
  match move with
  | Move.RotateThree x y z => (x + y + z) % 3 = 0
  | Move.SwapTwo x y => (x - y) % 3 = 0 ∨ (y - x) % 3 = 0

/-- Represents a sequence of numbers. -/
def Sequence := List ℕ

/-- Applies a move to a sequence. -/
def applyMove (seq : Sequence) (move : Move) : Sequence :=
  sorry

/-- Checks if a sequence can be transformed into another sequence using valid moves. -/
def canTransform (initial final : Sequence) : Prop :=
  ∃ (moves : List Move), (∀ move ∈ moves, isValidMove move) ∧
    (moves.foldl applyMove initial = final)

/-- The main theorem to be proved. -/
theorem transformation_possible (n : ℕ) :
  n > 1 →
  (canTransform (List.range n) ((n :: List.range (n-1)))) ↔ (n % 3 = 0 ∨ n % 3 = 1) :=
  sorry

end NUMINAMATH_CALUDE_transformation_possible_l2983_298332


namespace NUMINAMATH_CALUDE_parking_spot_difference_l2983_298338

/-- Represents the number of open parking spots on each level of a 4-story parking area -/
structure ParkingArea where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating the difference in open spots between second and first levels -/
theorem parking_spot_difference (p : ParkingArea) : 
  p.first = 4 → 
  p.third = p.second + 6 → 
  p.fourth = 14 → 
  p.first + p.second + p.third + p.fourth = 46 → 
  p.second - p.first = 7 := by
  sorry

#check parking_spot_difference

end NUMINAMATH_CALUDE_parking_spot_difference_l2983_298338


namespace NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l2983_298356

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 4*x^2 + 5*x - 4

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 3*x^2 - 8*x + 5

theorem extreme_values_and_tangent_line :
  -- Local minimum at x = 5/3
  (∃ δ₁ > 0, ∀ x ∈ Set.Ioo ((5:ℝ)/3 - δ₁) ((5:ℝ)/3 + δ₁), f x ≥ f ((5:ℝ)/3)) ∧
  f ((5:ℝ)/3) = -(58:ℝ)/27 ∧
  -- Local maximum at x = 1
  (∃ δ₂ > 0, ∀ x ∈ Set.Ioo (1 - δ₂) (1 + δ₂), f x ≤ f 1) ∧
  f 1 = -2 ∧
  -- Tangent line equation at (2, f(2))
  (∀ x : ℝ, f 2 + f' 2 * (x - 2) = x - 4) :=
sorry

end NUMINAMATH_CALUDE_extreme_values_and_tangent_line_l2983_298356


namespace NUMINAMATH_CALUDE_equation_solutions_l2983_298346

theorem equation_solutions (x : ℝ) : 
  (7.331 * (Real.log x / Real.log 3 - 1) / (Real.log (x/3) / Real.log 3) - 
   2 * Real.log (Real.sqrt x) / Real.log 3 + 
   (Real.log x / Real.log 3)^2 = 3) ↔ 
  (x = 1/3 ∨ x = 9) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2983_298346


namespace NUMINAMATH_CALUDE_construction_cost_is_212900_l2983_298354

-- Define the cost components
def land_cost_per_sqm : ℚ := 60
def land_area : ℚ := 2500
def brick_cost_per_1000 : ℚ := 120
def brick_quantity : ℚ := 15000
def roof_tile_cost : ℚ := 12
def roof_tile_quantity : ℚ := 800
def cement_bag_cost : ℚ := 8
def cement_bag_quantity : ℚ := 250
def wooden_beam_cost_per_m : ℚ := 25
def wooden_beam_length : ℚ := 1000
def steel_bar_cost_per_m : ℚ := 15
def steel_bar_length : ℚ := 500
def electrical_wiring_cost_per_m : ℚ := 2
def electrical_wiring_length : ℚ := 2000
def plumbing_pipe_cost_per_m : ℚ := 4
def plumbing_pipe_length : ℚ := 3000

-- Define the total construction cost function
def total_construction_cost : ℚ :=
  land_cost_per_sqm * land_area +
  brick_cost_per_1000 * brick_quantity / 1000 +
  roof_tile_cost * roof_tile_quantity +
  cement_bag_cost * cement_bag_quantity +
  wooden_beam_cost_per_m * wooden_beam_length +
  steel_bar_cost_per_m * steel_bar_length +
  electrical_wiring_cost_per_m * electrical_wiring_length +
  plumbing_pipe_cost_per_m * plumbing_pipe_length

-- Theorem statement
theorem construction_cost_is_212900 :
  total_construction_cost = 212900 := by
  sorry

end NUMINAMATH_CALUDE_construction_cost_is_212900_l2983_298354


namespace NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2983_298350

theorem smallest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 17 ∣ n → 102 ≤ n := by
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_multiple_of_17_l2983_298350


namespace NUMINAMATH_CALUDE_proposition_implication_l2983_298379

theorem proposition_implication (P : ℕ → Prop) 
  (h1 : ∀ k : ℕ, k > 0 → (P k → P (k + 1)))
  (h2 : ¬ P 5) : 
  ¬ P 4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_implication_l2983_298379


namespace NUMINAMATH_CALUDE_product_inequality_l2983_298382

theorem product_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 ∧ 
  ((a + 1) * (b + 1) * (c + 1) = 27 ↔ a = 2 ∧ b = 2 ∧ c = 2) := by
  sorry

#check product_inequality

end NUMINAMATH_CALUDE_product_inequality_l2983_298382


namespace NUMINAMATH_CALUDE_easter_egg_distribution_l2983_298355

theorem easter_egg_distribution (baskets : Nat) (eggs_per_basket : Nat) (people : Nat) :
  baskets = 15 → eggs_per_basket = 12 → people = 20 →
  (baskets * eggs_per_basket) / people = 9 := by
  sorry

end NUMINAMATH_CALUDE_easter_egg_distribution_l2983_298355


namespace NUMINAMATH_CALUDE_log_sum_equality_l2983_298302

theorem log_sum_equality : 
  Real.log 8 / Real.log 2 + 3 * (Real.log 4 / Real.log 2) + 
  2 * (Real.log 16 / Real.log 8) + (Real.log 64 / Real.log 4) = 44 / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_sum_equality_l2983_298302


namespace NUMINAMATH_CALUDE_power_five_sum_minus_two_l2983_298399

theorem power_five_sum_minus_two (n : ℕ) : n^5 + n^5 + n^5 + n^5 - 2 * n^5 = 2 * n^5 :=
by
  sorry

end NUMINAMATH_CALUDE_power_five_sum_minus_two_l2983_298399


namespace NUMINAMATH_CALUDE_nathan_daily_hours_l2983_298388

/-- Proves that Nathan played 3 hours per day given the conditions of the problem -/
theorem nathan_daily_hours : ∃ x : ℕ, 
  (14 * x + 5 * 7 = 77) ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_nathan_daily_hours_l2983_298388


namespace NUMINAMATH_CALUDE_f_1988_11_equals_169_l2983_298340

/-- Sum of digits of a positive integer -/
def sumOfDigits (k : ℕ+) : ℕ := sorry

/-- Square of sum of digits -/
def f₁ (k : ℕ+) : ℕ := (sumOfDigits k) ^ 2

/-- Recursive definition of fₙ -/
def f (n : ℕ) (k : ℕ+) : ℕ :=
  match n with
  | 0 => k.val
  | 1 => f₁ k
  | n + 1 => f₁ ⟨f n k, sorry⟩

/-- The main theorem to prove -/
theorem f_1988_11_equals_169 : f 1988 11 = 169 := by sorry

end NUMINAMATH_CALUDE_f_1988_11_equals_169_l2983_298340


namespace NUMINAMATH_CALUDE_train_length_l2983_298339

/-- The length of a train given its speed and time to cross an electric pole -/
theorem train_length (speed_kmh : ℝ) (time_sec : ℝ) : 
  speed_kmh = 75.6 → time_sec = 21 → speed_kmh * (1000 / 3600) * time_sec = 441 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2983_298339


namespace NUMINAMATH_CALUDE_routes_from_p_to_q_l2983_298386

/-- Represents a directed graph with vertices P, R, S, T, Q -/
structure Network where
  vertices : Finset Char
  edges : Finset (Char × Char)

/-- Counts the number of paths between two vertices in the network -/
def count_paths (n : Network) (start finish : Char) : ℕ :=
  sorry

/-- The specific network described in the problem -/
def problem_network : Network :=
  { vertices := {'P', 'R', 'S', 'T', 'Q'},
    edges := {('P', 'R'), ('P', 'S'), ('P', 'T'), ('R', 'T'), ('R', 'Q'), ('S', 'R'), ('S', 'T'), ('S', 'Q'), ('T', 'R'), ('T', 'S'), ('T', 'Q')} }

theorem routes_from_p_to_q (n : Network := problem_network) :
  count_paths n 'P' 'Q' = 16 :=
sorry

end NUMINAMATH_CALUDE_routes_from_p_to_q_l2983_298386


namespace NUMINAMATH_CALUDE_line_equation_proof_l2983_298357

/-- Given a line defined by the equation (2, -1) · ((x, y) - (1, 5)) = 0,
    prove that its slope is 2 and y-intercept is 3. -/
theorem line_equation_proof :
  ∀ (x y : ℝ),
  (2 * (x - 1) + (-1) * (y - 5) = 0) →
  ∃ (m b : ℝ), m = 2 ∧ b = 3 ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_line_equation_proof_l2983_298357


namespace NUMINAMATH_CALUDE_paul_initial_strawberries_l2983_298343

/-- The number of strawberries Paul initially had -/
def initial_strawberries : ℕ := sorry

/-- The number of strawberries Paul picked -/
def picked_strawberries : ℕ := 35

/-- The total number of strawberries Paul had after picking more -/
def total_strawberries : ℕ := 63

theorem paul_initial_strawberries : 
  initial_strawberries = 28 :=
by
  have h : initial_strawberries + picked_strawberries = total_strawberries := sorry
  sorry

end NUMINAMATH_CALUDE_paul_initial_strawberries_l2983_298343


namespace NUMINAMATH_CALUDE_boat_length_l2983_298375

/-- The length of a boat given specific conditions -/
theorem boat_length (breadth : Real) (sinking_depth : Real) (man_mass : Real) (water_density : Real) :
  breadth = 3 ∧ 
  sinking_depth = 0.01 ∧ 
  man_mass = 210 ∧ 
  water_density = 1000 →
  ∃ (length : Real), length = 7 ∧ 
    man_mass = water_density * (length * breadth * sinking_depth) :=
by sorry

end NUMINAMATH_CALUDE_boat_length_l2983_298375
