import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1584_158488

theorem quadratic_roots_sum_product (x₁ x₂ k d : ℝ) : 
  x₁ ≠ x₂ →
  4 * x₁^2 - k * x₁ = d →
  4 * x₂^2 - k * x₂ = d →
  x₁ + x₂ = 2 →
  d = -12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1584_158488


namespace NUMINAMATH_CALUDE_jake_weight_loss_l1584_158484

theorem jake_weight_loss (total_weight sister_weight jake_weight : ℕ) 
  (h1 : total_weight = 212)
  (h2 : jake_weight = 152)
  (h3 : total_weight = jake_weight + sister_weight) :
  jake_weight - (2 * sister_weight) = 32 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l1584_158484


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l1584_158450

theorem min_value_x_plus_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x * y = 2 * x + y) :
  x + y ≥ 3 + 2 * Real.sqrt 2 ∧
  (x + y = 3 + 2 * Real.sqrt 2 ↔ x = Real.sqrt 2 + 1) :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l1584_158450


namespace NUMINAMATH_CALUDE_prob_rolling_six_is_five_thirty_sixths_l1584_158408

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numFaces * numFaces

/-- The number of ways to roll a sum of 6 with two dice -/
def waysToRollSix : ℕ := 5

/-- The probability of rolling a sum of 6 with two fair dice -/
def probRollingSix : ℚ := waysToRollSix / totalOutcomes

theorem prob_rolling_six_is_five_thirty_sixths :
  probRollingSix = 5 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_rolling_six_is_five_thirty_sixths_l1584_158408


namespace NUMINAMATH_CALUDE_find_A_l1584_158414

theorem find_A : ∃ (A B : ℕ), A < 10 ∧ B < 10 ∧ A * 100 + 30 + B - 41 = 591 ∧ A = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l1584_158414


namespace NUMINAMATH_CALUDE_initial_plant_ratio_l1584_158411

/-- Represents the number and types of plants in Roxy's garden -/
structure Garden where
  flowering : ℕ
  fruiting : ℕ

/-- Represents the transactions of buying and giving away plants -/
structure Transactions where
  bought_flowering : ℕ
  bought_fruiting : ℕ
  given_flowering : ℕ
  given_fruiting : ℕ

/-- Calculates the final number of plants after transactions -/
def final_plants (initial : Garden) (trans : Transactions) : ℕ :=
  initial.flowering + initial.fruiting + trans.bought_flowering + trans.bought_fruiting - 
  trans.given_flowering - trans.given_fruiting

/-- Theorem stating the initial ratio of fruiting to flowering plants -/
theorem initial_plant_ratio (initial : Garden) (trans : Transactions) :
  initial.flowering = 7 ∧ 
  trans.bought_flowering = 3 ∧ 
  trans.bought_fruiting = 2 ∧
  trans.given_flowering = 1 ∧
  trans.given_fruiting = 4 ∧
  final_plants initial trans = 21 →
  initial.fruiting = 2 * initial.flowering :=
by
  sorry


end NUMINAMATH_CALUDE_initial_plant_ratio_l1584_158411


namespace NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1584_158473

/-- A right trapezoid with an inscribed circle -/
structure RightTrapezoidWithInscribedCircle where
  /-- The length of the smaller segment of the larger lateral side -/
  smaller_segment : ℝ
  /-- The length of the larger segment of the larger lateral side -/
  larger_segment : ℝ
  /-- The smaller segment is positive -/
  smaller_segment_pos : 0 < smaller_segment
  /-- The larger segment is positive -/
  larger_segment_pos : 0 < larger_segment

/-- The area of a right trapezoid with an inscribed circle -/
def area (t : RightTrapezoidWithInscribedCircle) : ℝ :=
  18 -- Definition without proof

/-- Theorem stating that the area of the specific right trapezoid is 18 -/
theorem area_of_specific_trapezoid :
  ∀ t : RightTrapezoidWithInscribedCircle,
  t.smaller_segment = 1 ∧ t.larger_segment = 4 →
  area t = 18 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_trapezoid_l1584_158473


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1584_158464

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | |x| ≥ 1}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

-- State the theorem
theorem complement_intersection_theorem :
  (Set.compl A) ∩ (Set.compl B) = {x : ℝ | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1584_158464


namespace NUMINAMATH_CALUDE_inequality_proof_l1584_158459

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1584_158459


namespace NUMINAMATH_CALUDE_area_of_extended_quadrilateral_l1584_158415

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (EF FG GH HE : ℝ)
  (area : ℝ)

-- Define the extended quadrilateral E'F'G'H'
structure ExtendedQuadrilateral :=
  (base : Quadrilateral)
  (EE' FF' GG' HH' : ℝ)

-- Define our specific quadrilateral
def EFGH : Quadrilateral :=
  { EF := 5
  , FG := 10
  , GH := 9
  , HE := 7
  , area := 12 }

-- Define our specific extended quadrilateral
def EFGH_extended : ExtendedQuadrilateral :=
  { base := EFGH
  , EE' := 7
  , FF' := 5
  , GG' := 10
  , HH' := 9 }

-- State the theorem
theorem area_of_extended_quadrilateral :
  (EFGH_extended.base.area + 
   EFGH_extended.base.EF * EFGH_extended.FF' +
   EFGH_extended.base.FG * EFGH_extended.GG' +
   EFGH_extended.base.GH * EFGH_extended.HH' +
   EFGH_extended.base.HE * EFGH_extended.EE') = 36 := by
  sorry

end NUMINAMATH_CALUDE_area_of_extended_quadrilateral_l1584_158415


namespace NUMINAMATH_CALUDE_eccentric_annulus_area_l1584_158439

/-- Eccentric annulus area theorem -/
theorem eccentric_annulus_area 
  (R r d : ℝ) 
  (h1 : R > r) 
  (h2 : d < R) : 
  Real.pi * (R - r - d^2 / (R - r)) = 
    Real.pi * R^2 - Real.pi * r^2 :=
sorry

end NUMINAMATH_CALUDE_eccentric_annulus_area_l1584_158439


namespace NUMINAMATH_CALUDE_inequality_multiplication_l1584_158466

theorem inequality_multiplication (x y : ℝ) (h : x > y) : 3 * x > 3 * y := by
  sorry

end NUMINAMATH_CALUDE_inequality_multiplication_l1584_158466


namespace NUMINAMATH_CALUDE_solution_set_l1584_158440

def A : Set ℝ := {x : ℝ | x^2 - 5*x + 6 = 0}

def B (m : ℝ) : Set ℝ := {x : ℝ | m*x + 1 = 0}

theorem solution_set (m : ℝ) : (A ∪ B m = A) ↔ m ∈ ({0, -1/2, -1/3} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l1584_158440


namespace NUMINAMATH_CALUDE_min_sum_of_digits_for_odd_primes_l1584_158471

def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def n (p : ℕ) : ℕ := p^4 - 5*p^2 + 13

theorem min_sum_of_digits_for_odd_primes :
  ∀ p : ℕ, is_odd_prime p → sum_of_digits (n p) ≥ sum_of_digits (n 5) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_digits_for_odd_primes_l1584_158471


namespace NUMINAMATH_CALUDE_cylinder_cut_surface_area_l1584_158404

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ

/-- Represents the area of a flat surface created by cutting the cylinder -/
def cutSurfaceArea (c : RightCircularCylinder) (arcAngle : ℝ) : ℝ :=
  sorry

theorem cylinder_cut_surface_area :
  let c : RightCircularCylinder := { radius := 8, height := 10 }
  let arcAngle : ℝ := π / 2  -- 90 degrees in radians
  cutSurfaceArea c arcAngle = 40 * π - 40 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_cut_surface_area_l1584_158404


namespace NUMINAMATH_CALUDE_cow_spots_count_l1584_158443

/-- The number of spots on a cow with given left and right side spot counts. -/
def total_spots (left : ℕ) (right : ℕ) : ℕ := left + right

/-- The number of spots on the right side of the cow, given the number on the left side. -/
def right_spots (left : ℕ) : ℕ := 3 * left + 7

theorem cow_spots_count :
  let left := 16
  let right := right_spots left
  total_spots left right = 71 := by
  sorry

end NUMINAMATH_CALUDE_cow_spots_count_l1584_158443


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1584_158461

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / (Real.sin A) = b / (Real.sin B) →
  a / (Real.sin A) = c / (Real.sin C) →
  -- Given condition
  a * Real.cos B + b * Real.cos A = 2 * c * Real.cos B →
  -- Conclusions
  B = π / 3 ∧
  (∀ x, x ∈ Set.Ioo (-3/2) (1/2) ↔
    ∃ A', 0 < A' ∧ A' < 2*π/3 ∧
    x = Real.sin A' * (Real.sqrt 3 * Real.cos A' - Real.sin A')) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1584_158461


namespace NUMINAMATH_CALUDE_wheat_flour_amount_l1584_158431

/-- The amount of wheat flour used by the bakery -/
def wheat_flour : ℝ := sorry

/-- The amount of white flour used by the bakery -/
def white_flour : ℝ := 0.1

/-- The total amount of flour used by the bakery -/
def total_flour : ℝ := 0.3

/-- Theorem stating that the amount of wheat flour used is 0.2 bags -/
theorem wheat_flour_amount : wheat_flour = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_wheat_flour_amount_l1584_158431


namespace NUMINAMATH_CALUDE_product_mod_eight_l1584_158451

theorem product_mod_eight : (71 * 73) % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_eight_l1584_158451


namespace NUMINAMATH_CALUDE_stating_inspection_probability_theorem_l1584_158403

/-- Represents the total number of items -/
def total_items : ℕ := 5

/-- Represents the number of defective items -/
def defective_items : ℕ := 2

/-- Represents the number of good items -/
def good_items : ℕ := 3

/-- Represents the number of inspections after which we want to calculate the probability -/
def target_inspections : ℕ := 4

/-- Represents the probability of the inspection stopping after exactly the target number of inspections -/
noncomputable def inspection_probability : ℚ := 3/5

/-- 
Theorem stating that the probability of the inspection stopping after exactly 
the target number of inspections is equal to the calculated probability
-/
theorem inspection_probability_theorem : 
  let p := inspection_probability
  p = (1 : ℚ) - (defective_items.choose 2 / total_items.choose 2) - 
      ((good_items.choose 3 + defective_items.choose 1 * good_items.choose 1 * (total_items - 3).choose 1) / total_items.choose 3) :=
by sorry

end NUMINAMATH_CALUDE_stating_inspection_probability_theorem_l1584_158403


namespace NUMINAMATH_CALUDE_smallest_gcd_yz_l1584_158409

theorem smallest_gcd_yz (x y z : ℕ+) (h1 : Nat.gcd x.val y.val = 224) (h2 : Nat.gcd x.val z.val = 546) :
  ∃ (y' z' : ℕ+), Nat.gcd y'.val z'.val = 14 ∧ 
  (∀ (a b : ℕ+), Nat.gcd x.val a.val = 224 → Nat.gcd x.val b.val = 546 → 
    Nat.gcd a.val b.val ≥ 14) :=
sorry

end NUMINAMATH_CALUDE_smallest_gcd_yz_l1584_158409


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l1584_158401

/-- The eccentricity of an ellipse with equation x^2 + y^2/4 = 1 is √3/2 -/
theorem ellipse_eccentricity : 
  let e : ℝ := (Real.sqrt 3) / 2
  ∀ x y : ℝ, x^2 + y^2/4 = 1 → e = (Real.sqrt (4 - 1)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l1584_158401


namespace NUMINAMATH_CALUDE_tank_volume_proof_l1584_158457

def inletRate : ℝ := 5
def outletRate1 : ℝ := 9
def outletRate2 : ℝ := 8
def emptyTime : ℝ := 2880
def inchesPerFoot : ℝ := 12

def tankVolume : ℝ := 20

theorem tank_volume_proof :
  let netEmptyRate := outletRate1 + outletRate2 - inletRate
  let volumeInCubicInches := netEmptyRate * emptyTime
  let cubicInchesPerCubicFoot := inchesPerFoot ^ 3
  volumeInCubicInches / cubicInchesPerCubicFoot = tankVolume := by
  sorry

#check tank_volume_proof

end NUMINAMATH_CALUDE_tank_volume_proof_l1584_158457


namespace NUMINAMATH_CALUDE_equation_is_parabola_l1584_158452

/-- Represents a conic section --/
inductive ConicSection
  | Circle
  | Parabola
  | Ellipse
  | Hyperbola
  | None

/-- Determines the type of conic section for the given equation --/
def determine_conic_section (equation : ℝ → ℝ → Prop) : ConicSection := sorry

/-- The equation |x-3| = √((y+4)² + x²) --/
def equation (x y : ℝ) : Prop :=
  |x - 3| = Real.sqrt ((y + 4)^2 + x^2)

theorem equation_is_parabola :
  determine_conic_section equation = ConicSection.Parabola := by sorry

end NUMINAMATH_CALUDE_equation_is_parabola_l1584_158452


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1584_158405

/-- A line passing through (2, 3) with equal x and y intercepts -/
def EqualInterceptLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (a : ℝ), (p.1 / a + p.2 / a = 1 ∧ a ≠ 0) ∨ (p.1 = 2 ∧ p.2 = 3) ∨ (p.1 = 0 ∧ p.2 = 0)}

theorem equal_intercept_line_equation :
  EqualInterceptLine = {p : ℝ × ℝ | p.1 + p.2 - 5 = 0} ∪ {p : ℝ × ℝ | 3 * p.1 - 2 * p.2 = 0} :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1584_158405


namespace NUMINAMATH_CALUDE_R_and_T_largest_area_l1584_158433

/-- Represents a polygon constructed from unit squares and right triangles with legs of length 1 -/
structure Polygon where
  squares : ℕ
  triangles : ℕ

/-- Calculates the area of a polygon -/
def area (p : Polygon) : ℚ :=
  p.squares + p.triangles / 2

/-- The five polygons P, Q, R, S, T -/
def P : Polygon := ⟨3, 2⟩
def Q : Polygon := ⟨4, 1⟩
def R : Polygon := ⟨6, 0⟩
def S : Polygon := ⟨2, 4⟩
def T : Polygon := ⟨5, 2⟩

/-- Theorem stating that R and T have the largest area among the five polygons -/
theorem R_and_T_largest_area :
  area R = area T ∧
  area R ≥ area P ∧
  area R ≥ area Q ∧
  area R ≥ area S :=
sorry

end NUMINAMATH_CALUDE_R_and_T_largest_area_l1584_158433


namespace NUMINAMATH_CALUDE_black_ball_count_l1584_158494

/-- Given a bag with white and black balls, prove the number of black balls when the probability of drawing a white ball is known -/
theorem black_ball_count 
  (white_balls : ℕ) 
  (black_balls : ℕ) 
  (total_balls : ℕ) 
  (prob_white : ℚ) 
  (h1 : white_balls = 20)
  (h2 : total_balls = white_balls + black_balls)
  (h3 : prob_white = 2/5)
  (h4 : prob_white = white_balls / total_balls) :
  black_balls = 30 := by
sorry

end NUMINAMATH_CALUDE_black_ball_count_l1584_158494


namespace NUMINAMATH_CALUDE_infinite_division_sum_equal_l1584_158499

/-- Represents a shape with an area -/
class HasArea (α : Type*) where
  area : α → ℝ

/-- Represents a shape that can be divided -/
class Divisible (α : Type*) where
  divide : α → ℝ → α

variable (T : Type*) [HasArea T] [Divisible T]
variable (Q : Type*) [HasArea Q] [Divisible Q]

/-- The sum of areas after infinite divisions -/
noncomputable def infiniteDivisionSum (shape : T) (ratio : ℝ) : ℝ := sorry

/-- Theorem stating the equality of infinite division sums -/
theorem infinite_division_sum_equal
  (triangle : T)
  (quad : Q)
  (ratio : ℝ)
  (h : HasArea.area triangle = 1.5 * HasArea.area quad) :
  infiniteDivisionSum T triangle ratio = infiniteDivisionSum Q quad ratio := by
  sorry

end NUMINAMATH_CALUDE_infinite_division_sum_equal_l1584_158499


namespace NUMINAMATH_CALUDE_parabola_equation_l1584_158453

/-- A parabola with vertex at the origin and directrix y = 4 has the standard equation x^2 = -16y -/
theorem parabola_equation (p : ℝ → ℝ → Prop) :
  (∀ x y, p x y ↔ y = -x^2 / 16) →  -- Standard equation of the parabola
  (∀ x, p x 0) →  -- Vertex at the origin
  (∀ x, p x 4 ↔ x = 0) →  -- Directrix equation
  ∀ x y, p x y ↔ x^2 = -16 * y := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l1584_158453


namespace NUMINAMATH_CALUDE_problem_solution_l1584_158492

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xyz : x * y * z = 1)
  (h_x_z : x + 1 / z = 4)
  (h_y_x : y + 1 / x = 20) :
  z + 1 / y = 26 / 79 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1584_158492


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1584_158407

/-- A geometric sequence with the given first four terms has a common ratio of -3/2 -/
theorem geometric_sequence_common_ratio :
  ∀ (a : ℕ → ℚ), 
    a 0 = 32 ∧ a 1 = -48 ∧ a 2 = 72 ∧ a 3 = -108 →
    ∃ (r : ℚ), r = -3/2 ∧ ∀ n, a (n + 1) = r * a n :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1584_158407


namespace NUMINAMATH_CALUDE_fred_has_five_balloons_l1584_158442

/-- The number of yellow balloons Fred has -/
def fred_balloons (total sam mary : ℕ) : ℕ := total - (sam + mary)

/-- Theorem: Fred has 5 yellow balloons -/
theorem fred_has_five_balloons (total sam mary : ℕ) 
  (h_total : total = 18) 
  (h_sam : sam = 6) 
  (h_mary : mary = 7) : 
  fred_balloons total sam mary = 5 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_five_balloons_l1584_158442


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1584_158468

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A point on an ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The equation of a line -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ
  h_not_zero : A ≠ 0 ∨ B ≠ 0

/-- Theorem: The equation of the tangent line at a point on an ellipse -/
theorem tangent_line_equation (e : Ellipse) (p : PointOnEllipse e) :
  ∃ (l : Line), l.A = p.x / e.a^2 ∧ l.B = p.y / e.b^2 ∧ l.C = -1 ∧
  (∀ (x y : ℝ), x^2 / e.a^2 + y^2 / e.b^2 = 1 → l.A * x + l.B * y + l.C = 0 → x = p.x ∧ y = p.y) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1584_158468


namespace NUMINAMATH_CALUDE_point_coordinates_l1584_158449

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant of the 2D plane -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

theorem point_coordinates (M : Point) 
  (h1 : secondQuadrant M)
  (h2 : distanceToXAxis M = 5)
  (h3 : distanceToYAxis M = 3) :
  M.x = -3 ∧ M.y = 5 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1584_158449


namespace NUMINAMATH_CALUDE_equation_represents_line_l1584_158437

-- Define the equation
def equation (x y : ℝ) : Prop :=
  ((x^2 + y^2 - 2*x) * Real.sqrt (x + y - 3) = 0)

-- Define what it means for the equation to represent a line
def represents_line (f : ℝ → ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∨ b ≠ 0 ∧ ∀ x y : ℝ, f x y ↔ a*x + b*y + c = 0

-- Theorem statement
theorem equation_represents_line :
  represents_line equation :=
sorry

end NUMINAMATH_CALUDE_equation_represents_line_l1584_158437


namespace NUMINAMATH_CALUDE_least_possible_area_of_square_l1584_158496

/-- The least possible side length of a square when measured as 7 cm to the nearest centimeter -/
def least_possible_side : ℝ := 6.5

/-- The measured side length of the square to the nearest centimeter -/
def measured_side : ℕ := 7

/-- The least possible area of the square -/
def least_possible_area : ℝ := least_possible_side ^ 2

theorem least_possible_area_of_square :
  least_possible_side ≥ (measured_side : ℝ) - 0.5 ∧
  least_possible_side < (measured_side : ℝ) ∧
  least_possible_area = 42.25 := by
  sorry

end NUMINAMATH_CALUDE_least_possible_area_of_square_l1584_158496


namespace NUMINAMATH_CALUDE_students_without_portraits_l1584_158432

theorem students_without_portraits (total_students : ℕ) 
  (before_break : ℕ) (during_break : ℕ) (after_lunch : ℕ) : 
  total_students = 60 →
  before_break = total_students / 4 →
  during_break = (total_students - before_break) / 3 →
  after_lunch = 10 →
  total_students - (before_break + during_break + after_lunch) = 20 := by
sorry

end NUMINAMATH_CALUDE_students_without_portraits_l1584_158432


namespace NUMINAMATH_CALUDE_rational_cosine_terms_l1584_158475

theorem rational_cosine_terms (x : ℝ) 
  (hS : ∃ q : ℚ, (Real.sin (64 * x) + Real.sin (65 * x)) = ↑q)
  (hC : ∃ q : ℚ, (Real.cos (64 * x) + Real.cos (65 * x)) = ↑q) :
  ∃ (q1 q2 : ℚ), Real.cos (64 * x) = ↑q1 ∧ Real.cos (65 * x) = ↑q2 :=
sorry

end NUMINAMATH_CALUDE_rational_cosine_terms_l1584_158475


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l1584_158465

def M : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def N : Set ℝ := {x | x - 2 ≤ x ∧ x < 3}

theorem complement_M_intersect_N :
  ∀ x : ℝ, x ∈ (M ∩ N)ᶜ ↔ x < -2 ∨ x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l1584_158465


namespace NUMINAMATH_CALUDE_polynomial_sum_at_zero_and_four_l1584_158406

/-- Given a polynomial f(x) = x⁴ + ax³ + bx² + cx + d with zeros 1, 2, and 3,
    prove that f(0) + f(4) = 24 -/
theorem polynomial_sum_at_zero_and_four 
  (f : ℝ → ℝ) 
  (a b c d : ℝ) 
  (h1 : ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d) 
  (h2 : f 1 = 0) 
  (h3 : f 2 = 0) 
  (h4 : f 3 = 0) : 
  f 0 + f 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_sum_at_zero_and_four_l1584_158406


namespace NUMINAMATH_CALUDE_restaurant_group_kids_l1584_158419

/-- Proves that in a group of 12 people, where adult meals cost $3 each and kids eat free,
    if the total cost is $15, then the number of kids in the group is 7. -/
theorem restaurant_group_kids (total_people : ℕ) (adult_meal_cost : ℕ) (total_cost : ℕ) 
  (h1 : total_people = 12)
  (h2 : adult_meal_cost = 3)
  (h3 : total_cost = 15) :
  total_people - (total_cost / adult_meal_cost) = 7 :=
by sorry

end NUMINAMATH_CALUDE_restaurant_group_kids_l1584_158419


namespace NUMINAMATH_CALUDE_blue_rows_count_l1584_158482

/-- Given a grid with the following properties:
  * 10 rows and 15 squares per row
  * 4 rows of 6 squares in the middle are red
  * 66 squares are green
  * All remaining squares are blue
  * Blue squares cover entire rows
Prove that the number of rows colored blue at the beginning and end of the grid is 4 -/
theorem blue_rows_count (total_rows : Nat) (squares_per_row : Nat) 
  (red_rows : Nat) (red_squares_per_row : Nat) (green_squares : Nat) : 
  total_rows = 10 → 
  squares_per_row = 15 → 
  red_rows = 4 → 
  red_squares_per_row = 6 → 
  green_squares = 66 → 
  (total_rows * squares_per_row - red_rows * red_squares_per_row - green_squares) / squares_per_row = 4 := by
  sorry

end NUMINAMATH_CALUDE_blue_rows_count_l1584_158482


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1584_158448

/-- A system of linear equations with integer coefficients -/
structure LinearSystem where
  eq1 : ℤ → ℤ → ℤ → ℤ
  eq2 : ℤ → ℤ → ℤ → ℤ

/-- The specific system of equations from the problem -/
def problemSystem : LinearSystem :=
  { eq1 := λ x y z => 3*x - 4*y + 5*z - 10
    eq2 := λ x y z => 8*x + 7*y - 3*z - 13 }

/-- A solution to the system is valid if both equations equal zero -/
def isValidSolution (s : LinearSystem) (x y z : ℤ) : Prop :=
  s.eq1 x y z = 0 ∧ s.eq2 x y z = 0

/-- A positive integer solution -/
def isPositiveIntegerSolution (x y z : ℤ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0

theorem unique_positive_solution :
  ∀ x y z : ℤ,
    isValidSolution problemSystem x y z ∧ isPositiveIntegerSolution x y z
    ↔ x = 1 ∧ y = 2 ∧ z = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1584_158448


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1584_158420

/-- The line in 3D space --/
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + 2*t, -2 - 5*t, 3 - 2*t)

/-- The plane in 3D space --/
def plane (x y z : ℝ) : Prop :=
  x + 2*y - 5*z + 16 = 0

/-- The intersection point --/
def intersection_point : ℝ × ℝ × ℝ :=
  (3, -7, 1)

theorem intersection_point_is_unique :
  (∃! p : ℝ × ℝ × ℝ, ∃ t : ℝ, line t = p ∧ plane p.1 p.2.1 p.2.2) ∧
  (∃ t : ℝ, line t = intersection_point ∧ plane intersection_point.1 intersection_point.2.1 intersection_point.2.2) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1584_158420


namespace NUMINAMATH_CALUDE_k_range_theorem_l1584_158435

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x - 1

-- Define the even function g and odd function h
def g (x : ℝ) : ℝ := x^2 - 1
def h (x : ℝ) : ℝ := x

-- State the theorem
theorem k_range_theorem (k : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → g (k*x + k/x) < g (x^2 + 1/x^2 + 1)) ↔ 
  (-3/2 < k ∧ k < 3/2) :=
sorry

end NUMINAMATH_CALUDE_k_range_theorem_l1584_158435


namespace NUMINAMATH_CALUDE_pool_draining_and_filling_time_l1584_158469

/-- The time it takes for a pool to reach a certain water level when being simultaneously drained and filled -/
theorem pool_draining_and_filling_time 
  (pool_capacity : ℝ) 
  (drain_rate : ℝ) 
  (fill_rate : ℝ) 
  (final_volume : ℝ) 
  (h1 : pool_capacity = 120)
  (h2 : drain_rate = 1 / 4)
  (h3 : fill_rate = 1 / 6)
  (h4 : final_volume = 90) :
  ∃ t : ℝ, t = 3 ∧ 
  pool_capacity - (drain_rate * pool_capacity - fill_rate * pool_capacity) * t = final_volume :=
sorry

end NUMINAMATH_CALUDE_pool_draining_and_filling_time_l1584_158469


namespace NUMINAMATH_CALUDE_chord_equation_l1584_158481

theorem chord_equation (m n s t : ℝ) (hm : 0 < m) (hn : 0 < n) (hs : 0 < s) (ht : 0 < t)
  (h1 : m + n = 2) (h2 : m / s + n / t = 9) (h3 : s + t = 4 / 9)
  (h4 : ∃ (x1 y1 x2 y2 : ℝ), 
    x1^2 / 4 + y1^2 / 2 = 1 ∧ 
    x2^2 / 4 + y2^2 / 2 = 1 ∧ 
    (x1 + x2) / 2 = m ∧ 
    (y1 + y2) / 2 = n) :
  ∃ (a b c : ℝ), a * m + b * n + c = 0 ∧ a = 1 ∧ b = 2 ∧ c = -3 := by
sorry

end NUMINAMATH_CALUDE_chord_equation_l1584_158481


namespace NUMINAMATH_CALUDE_salt_fraction_in_solution_l1584_158421

theorem salt_fraction_in_solution (salt_weight water_weight : ℚ) :
  salt_weight = 6 → water_weight = 30 →
  salt_weight / (salt_weight + water_weight) = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_salt_fraction_in_solution_l1584_158421


namespace NUMINAMATH_CALUDE_climbing_five_floors_l1584_158444

/-- The number of ways to climb a building with a given number of floors and staircases per floor -/
def climbingWays (floors : ℕ) (staircasesPerFloor : ℕ) : ℕ :=
  staircasesPerFloor ^ (floors - 1)

/-- Theorem: In a 5-floor building with 2 staircases per floor, there are 16 ways to go from the first to the fifth floor -/
theorem climbing_five_floors :
  climbingWays 5 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_climbing_five_floors_l1584_158444


namespace NUMINAMATH_CALUDE_no_solution_iff_m_eq_four_l1584_158427

-- Define the equation
def equation (x m : ℝ) : Prop := 2 / x = m / (2 * x + 1)

-- Theorem stating the condition for no solution
theorem no_solution_iff_m_eq_four :
  (∀ x : ℝ, ¬ equation x m) ↔ m = 4 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_m_eq_four_l1584_158427


namespace NUMINAMATH_CALUDE_five_students_three_locations_l1584_158476

/-- The number of ways to assign students to locations -/
def assignment_count (n : ℕ) (k : ℕ) : ℕ :=
  -- n: number of students
  -- k: number of locations
  sorry

/-- Theorem stating the number of assignment plans for 5 students and 3 locations -/
theorem five_students_three_locations :
  assignment_count 5 3 = 150 := by sorry

end NUMINAMATH_CALUDE_five_students_three_locations_l1584_158476


namespace NUMINAMATH_CALUDE_right_triangles_AF_length_l1584_158400

theorem right_triangles_AF_length 
  (AB DE CD EF BC : ℝ)
  (h1 : AB = 12)
  (h2 : DE = 12)
  (h3 : CD = 8)
  (h4 : EF = 8)
  (h5 : BC = 5)
  (h6 : AB^2 + BC^2 = AC^2)  -- ABC is a right triangle
  (h7 : AC^2 + CD^2 = AD^2)  -- ACD is a right triangle
  (h8 : AD^2 + DE^2 = AE^2)  -- ADE is a right triangle
  (h9 : AE^2 + EF^2 = AF^2)  -- AEF is a right triangle
  : AF = 21 := by
    sorry

end NUMINAMATH_CALUDE_right_triangles_AF_length_l1584_158400


namespace NUMINAMATH_CALUDE_madhav_rank_l1584_158424

theorem madhav_rank (total_students : ℕ) (rank_from_last : ℕ) (rank_from_start : ℕ) : 
  total_students = 31 →
  rank_from_last = 15 →
  rank_from_start = total_students - (rank_from_last - 1) →
  rank_from_start = 17 := by
sorry

end NUMINAMATH_CALUDE_madhav_rank_l1584_158424


namespace NUMINAMATH_CALUDE_symmetric_point_in_third_quadrant_l1584_158477

/-- A point in 2D Cartesian coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of symmetry with respect to x-axis -/
def symmetricXAxis (p : Point) : Point :=
  ⟨p.x, -p.y⟩

/-- Definition of third quadrant -/
def isThirdQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- The main theorem -/
theorem symmetric_point_in_third_quadrant :
  let P : Point := ⟨-2, 1⟩
  let P' := symmetricXAxis P
  isThirdQuadrant P' :=
by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_in_third_quadrant_l1584_158477


namespace NUMINAMATH_CALUDE_km2_to_hectares_conversion_m2_to_km2_conversion_l1584_158462

-- Define the conversion factors
def km2_to_hectares : ℝ := 100
def m2_to_km2 : ℝ := 1000000

-- Theorem 1: 3.4 km² = 340 hectares
theorem km2_to_hectares_conversion :
  3.4 * km2_to_hectares = 340 := by sorry

-- Theorem 2: 690000 m² = 0.69 km²
theorem m2_to_km2_conversion :
  690000 / m2_to_km2 = 0.69 := by sorry

end NUMINAMATH_CALUDE_km2_to_hectares_conversion_m2_to_km2_conversion_l1584_158462


namespace NUMINAMATH_CALUDE_max_basketballs_l1584_158402

/-- Represents the prices and quantities of soccer balls and basketballs --/
structure BallPurchase where
  soccer_price : ℝ
  basketball_price : ℝ
  soccer_quantity : ℕ
  basketball_quantity : ℕ

/-- The conditions of the ball purchase problem --/
def valid_purchase (p : BallPurchase) : Prop :=
  p.basketball_price = 2 * p.soccer_price - 30 ∧
  3 * p.soccer_price * p.soccer_quantity = 2 * p.basketball_price * p.basketball_quantity ∧
  p.soccer_quantity + p.basketball_quantity = 200 ∧
  p.soccer_price * p.soccer_quantity + p.basketball_price * p.basketball_quantity ≤ 15500

/-- The theorem stating the maximum number of basketballs that can be purchased --/
theorem max_basketballs (p : BallPurchase) :
  valid_purchase p → p.basketball_quantity ≤ 116 := by
  sorry

end NUMINAMATH_CALUDE_max_basketballs_l1584_158402


namespace NUMINAMATH_CALUDE_candidate_a_vote_percentage_l1584_158478

/-- Represents the percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.6

/-- Represents the percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 1 - democrat_percentage

/-- Represents the percentage of Republican voters expected to vote for candidate A -/
def republican_vote_percentage : ℝ := 0.2

/-- Represents the total percentage of registered voters expected to vote for candidate A -/
def total_vote_percentage : ℝ := 0.5

/-- Represents the percentage of Democratic voters expected to vote for candidate A -/
def democrat_vote_percentage : ℝ := 0.7

theorem candidate_a_vote_percentage :
  democrat_percentage * democrat_vote_percentage +
  republican_percentage * republican_vote_percentage =
  total_vote_percentage :=
sorry

end NUMINAMATH_CALUDE_candidate_a_vote_percentage_l1584_158478


namespace NUMINAMATH_CALUDE_expression_simplification_l1584_158495

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  (x / (x^2 - 2*x + 1)) / ((x + 1) / (x^2 - 1) + 1) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1584_158495


namespace NUMINAMATH_CALUDE_count_numbers_with_digits_eq_six_l1584_158463

/-- The count of integers between 600 and 2000 that contain the digits 3, 5, and 7 -/
def count_numbers_with_digits : ℕ :=
  -- Definition goes here
  sorry

/-- The range of integers to consider -/
def lower_bound : ℕ := 600
def upper_bound : ℕ := 2000

/-- The required digits -/
def required_digits : List ℕ := [3, 5, 7]

theorem count_numbers_with_digits_eq_six :
  count_numbers_with_digits = 6 :=
sorry

end NUMINAMATH_CALUDE_count_numbers_with_digits_eq_six_l1584_158463


namespace NUMINAMATH_CALUDE_bob_winning_strategy_l1584_158454

/-- A polynomial with natural number coefficients -/
def NatPoly := ℕ → ℕ

/-- The evaluation of a polynomial at a given point -/
def eval (P : NatPoly) (x : ℤ) : ℕ :=
  sorry

/-- The degree of a polynomial -/
def degree (P : NatPoly) : ℕ :=
  sorry

/-- Bob's strategy: choose two integers and receive their polynomial values -/
def bob_strategy (P : NatPoly) : (ℤ × ℤ × ℕ × ℕ) :=
  sorry

theorem bob_winning_strategy :
  ∀ (P Q : NatPoly),
    (∀ (x : ℤ), eval P x = eval Q x) →
    let (a, b, Pa, Pb) := bob_strategy P
    eval P a = Pa ∧ eval P b = Pb ∧ eval Q a = Pa ∧ eval Q b = Pb →
    P = Q :=
  sorry

end NUMINAMATH_CALUDE_bob_winning_strategy_l1584_158454


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l1584_158486

theorem consecutive_numbers_sum (n : ℕ) : 
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 105) → 
  (n + (n + 5) = 35) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l1584_158486


namespace NUMINAMATH_CALUDE_area_of_triangle_perimeter_of_triangle_l1584_158474

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a = 3 ∧ t.b = 2 * t.c

-- Part 1
theorem area_of_triangle (t : Triangle) (h : triangle_conditions t) (h_A : t.A = 2 * Real.pi / 3) :
  (1/2) * t.b * t.c * Real.sin t.A = 9 * Real.sqrt 3 / 14 := by sorry

-- Part 2
theorem perimeter_of_triangle (t : Triangle) (h : triangle_conditions t) (h_BC : 2 * Real.sin t.B - Real.sin t.C = 1) :
  t.a + t.b + t.c = 4 * Real.sqrt 2 - Real.sqrt 5 + 3 ∨ 
  t.a + t.b + t.c = 4 * Real.sqrt 2 + Real.sqrt 5 + 3 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_perimeter_of_triangle_l1584_158474


namespace NUMINAMATH_CALUDE_polynomial_square_prime_values_l1584_158491

def P (n : ℤ) : ℤ := n^3 - n^2 - 5*n + 2

theorem polynomial_square_prime_values :
  {n : ℤ | ∃ (p : ℕ), Prime p ∧ (P n)^2 = p^2} = {-3, -1, 0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_polynomial_square_prime_values_l1584_158491


namespace NUMINAMATH_CALUDE_nested_root_equality_l1584_158472

theorem nested_root_equality (a : ℝ) (ha : a > 0) : 
  Real.sqrt (a * Real.sqrt (a * Real.sqrt a)) = a ^ (7/8) :=
by sorry

end NUMINAMATH_CALUDE_nested_root_equality_l1584_158472


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1584_158480

theorem quadratic_distinct_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 - 2*x₁ + m = 0 ∧ x₂^2 - 2*x₂ + m = 0) →
  m < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_range_l1584_158480


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l1584_158410

theorem cubic_equation_solution (y : ℝ) :
  (((30 * y + (30 * y + 27) ^ (1/3 : ℝ)) ^ (1/3 : ℝ)) = 15) → y = 1674/15 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l1584_158410


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1584_158418

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a)
  (h_a5 : a 5 = -9)
  (h_a8 : a 8 = 6) :
  a 11 = -4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1584_158418


namespace NUMINAMATH_CALUDE_polynomial_characterization_l1584_158460

theorem polynomial_characterization (p : ℕ) (hp : Nat.Prime p) :
  ∀ (P : ℕ → ℕ),
    (∀ x : ℕ, x > 0 → P x > x) →
    (∀ m : ℕ, m > 0 → ∃ l : ℕ, m ∣ (Nat.iterate P l p)) →
    (∃ a b : ℕ → ℕ, (∀ x, P x = x + 1 ∨ P x = x + p) ∧
                    (∀ x, a x = x + 1) ∧
                    (∀ x, b x = x + p)) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_characterization_l1584_158460


namespace NUMINAMATH_CALUDE_factorial_calculation_l1584_158497

theorem factorial_calculation : (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_calculation_l1584_158497


namespace NUMINAMATH_CALUDE_magicians_number_l1584_158498

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  h1 : a ≤ 9
  h2 : b ≤ 9
  h3 : c ≤ 9
  h4 : 0 < a

/-- Calculates the value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.a + 10 * n.b + n.c

/-- Calculates the sum of all permutations of a three-digit number -/
def sumOfPermutations (n : ThreeDigitNumber) : Nat :=
  (value n) + 
  (100 * n.a + 10 * n.c + n.b) +
  (100 * n.b + 10 * n.c + n.a) +
  (100 * n.b + 10 * n.a + n.c) +
  (100 * n.c + 10 * n.a + n.b) +
  (100 * n.c + 10 * n.b + n.a)

/-- The main theorem to prove -/
theorem magicians_number (n : ThreeDigitNumber) 
  (h : sumOfPermutations n = 4332) : value n = 118 := by
  sorry

end NUMINAMATH_CALUDE_magicians_number_l1584_158498


namespace NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1584_158445

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 15 → ¬(Nat.Prime p ∧ p ∣ n)

theorem smallest_composite_no_small_factors : 
  (is_composite 323) ∧ 
  (has_no_small_prime_factors 323) ∧ 
  (∀ m : ℕ, m < 323 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_composite_no_small_factors_l1584_158445


namespace NUMINAMATH_CALUDE_expression_evaluation_l1584_158446

theorem expression_evaluation :
  let x : ℚ := 2/3
  let y : ℚ := 4/5
  (6*x + 8*y + x^2*y) / (60*x*y^2) = 21/50 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1584_158446


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1584_158470

def line1 (x y : ℝ) : Prop := 3 * y - 2 * x + 4 = 0
def line2 (x y : ℝ) (b : ℝ) : Prop := 5 * y + b * x - 1 = 0

def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

theorem perpendicular_lines (b : ℝ) : 
  (∀ x y : ℝ, line1 x y → ∃ m1 : ℝ, y = m1 * x + (-4/3)) →
  (∀ x y : ℝ, line2 x y b → ∃ m2 : ℝ, y = m2 * x + (1/5)) →
  perpendicular (2/3) (-b/5) →
  b = 15/2 := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1584_158470


namespace NUMINAMATH_CALUDE_pool_filling_time_l1584_158458

/-- Proves that filling a 30,000-gallon pool with 5 hoses, each supplying 2.5 gallons per minute, takes 40 hours. -/
theorem pool_filling_time :
  let pool_capacity : ℝ := 30000
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℝ := 2.5
  let minutes_per_hour : ℕ := 60
  let total_flow_rate : ℝ := num_hoses * flow_rate_per_hose * minutes_per_hour
  pool_capacity / total_flow_rate = 40 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l1584_158458


namespace NUMINAMATH_CALUDE_a_greater_than_b_l1584_158423

theorem a_greater_than_b (n : ℕ) (a b : ℝ) 
  (h_n : n ≥ 2) 
  (h_a_pos : a > 0) 
  (h_b_pos : b > 0) 
  (h_a : a^n = a + 1) 
  (h_b : b^(2*n) = b + 3*a) : 
  a > b := by sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l1584_158423


namespace NUMINAMATH_CALUDE_number_equation_l1584_158430

theorem number_equation (x n : ℝ) : 
  x = 596.95 → 3639 + n - x = 3054 → n = 11.95 := by sorry

end NUMINAMATH_CALUDE_number_equation_l1584_158430


namespace NUMINAMATH_CALUDE_irrational_equality_l1584_158412

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- State the theorem
theorem irrational_equality (α β : ℝ) (h_irrational_α : Irrational α) (h_irrational_β : Irrational β) 
  (h_equality : ∀ x : ℝ, x > 0 → floor (α * floor (β * x)) = floor (β * floor (α * x))) :
  α = β :=
sorry

end NUMINAMATH_CALUDE_irrational_equality_l1584_158412


namespace NUMINAMATH_CALUDE_intersection_point_l1584_158428

/-- The linear function y = 2x + 4 -/
def f (x : ℝ) : ℝ := 2 * x + 4

/-- The y-axis is the vertical line with x-coordinate 0 -/
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- The graph of the linear function f -/
def graph_f : Set (ℝ × ℝ) := {p | p.2 = f p.1}

/-- The intersection point of the graph of f with the y-axis -/
def intersection : ℝ × ℝ := (0, f 0)

theorem intersection_point :
  intersection ∈ y_axis ∧ intersection ∈ graph_f ∧ intersection = (0, 4) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l1584_158428


namespace NUMINAMATH_CALUDE_second_candidate_percentage_l1584_158490

/-- Represents an exam with total marks and passing marks. -/
structure Exam where
  totalMarks : ℝ
  passingMarks : ℝ

/-- Represents a candidate's performance in the exam. -/
structure Candidate where
  marksObtained : ℝ

def Exam.firstCandidateCondition (e : Exam) : Prop :=
  0.3 * e.totalMarks = e.passingMarks - 50

def Exam.secondCandidateCondition (e : Exam) (c : Candidate) : Prop :=
  c.marksObtained = e.passingMarks + 25

/-- The theorem stating the percentage of marks obtained by the second candidate. -/
theorem second_candidate_percentage (e : Exam) (c : Candidate) :
  e.passingMarks = 199.99999999999997 →
  e.firstCandidateCondition →
  e.secondCandidateCondition c →
  c.marksObtained / e.totalMarks = 0.45 := by
  sorry


end NUMINAMATH_CALUDE_second_candidate_percentage_l1584_158490


namespace NUMINAMATH_CALUDE_total_cards_traded_is_128_l1584_158483

/-- Represents the number of cards of each type --/
structure CardCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents a trade of cards --/
structure Trade where
  fromA : ℕ
  fromB : ℕ
  fromC : ℕ
  toA : ℕ
  toB : ℕ
  toC : ℕ

/-- Calculates the total number of cards traded in a single trade --/
def cardsTraded (trade : Trade) : ℕ :=
  trade.fromA + trade.fromB + trade.fromC + trade.toA + trade.toB + trade.toC

/-- Represents the initial card counts and trades for each round --/
structure RoundData where
  initialPadma : CardCounts
  initialRobert : CardCounts
  padmaTrade : Trade
  robertTrade : Trade

theorem total_cards_traded_is_128 
  (round1 : RoundData)
  (round2 : RoundData)
  (round3 : RoundData)
  (h1 : round1.initialPadma = ⟨50, 45, 30⟩)
  (h2 : round1.padmaTrade = ⟨5, 12, 0, 0, 0, 20⟩)
  (h3 : round2.initialRobert = ⟨60, 50, 40⟩)
  (h4 : round2.robertTrade = ⟨10, 3, 15, 8, 18, 0⟩)
  (h5 : round3.padmaTrade = ⟨0, 15, 10, 12, 0, 0⟩) :
  cardsTraded round1.padmaTrade + cardsTraded round2.robertTrade + cardsTraded round3.padmaTrade = 128 := by
  sorry


end NUMINAMATH_CALUDE_total_cards_traded_is_128_l1584_158483


namespace NUMINAMATH_CALUDE_husband_catches_up_l1584_158441

/-- Yolanda's bike speed in miles per hour -/
def yolanda_speed : ℝ := 20

/-- Yolanda's husband's car speed in miles per hour -/
def husband_speed : ℝ := 40

/-- Time difference between Yolanda and her husband's departure in minutes -/
def time_difference : ℝ := 15

/-- The time it takes for Yolanda's husband to catch up to her in minutes -/
def catch_up_time : ℝ := 15

theorem husband_catches_up :
  yolanda_speed * (catch_up_time + time_difference) / 60 = husband_speed * catch_up_time / 60 :=
sorry

end NUMINAMATH_CALUDE_husband_catches_up_l1584_158441


namespace NUMINAMATH_CALUDE_original_number_is_two_thirds_l1584_158436

theorem original_number_is_two_thirds :
  ∃ x : ℚ, (1 + 1 / x = 5 / 2) ∧ (x = 2 / 3) := by sorry

end NUMINAMATH_CALUDE_original_number_is_two_thirds_l1584_158436


namespace NUMINAMATH_CALUDE_certain_amount_proof_l1584_158489

theorem certain_amount_proof : 
  ∀ (amount : ℝ), 
    (0.25 * 680 = 0.20 * 1000 - amount) → 
    amount = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l1584_158489


namespace NUMINAMATH_CALUDE_lumber_price_increase_l1584_158434

theorem lumber_price_increase 
  (original_lumber_cost : ℝ)
  (original_nails_cost : ℝ)
  (original_fabric_cost : ℝ)
  (total_cost_increase : ℝ)
  (h1 : original_lumber_cost = 450)
  (h2 : original_nails_cost = 30)
  (h3 : original_fabric_cost = 80)
  (h4 : total_cost_increase = 97) :
  let original_total_cost := original_lumber_cost + original_nails_cost + original_fabric_cost
  let new_total_cost := original_total_cost + total_cost_increase
  let new_lumber_cost := new_total_cost - (original_nails_cost + original_fabric_cost)
  let lumber_cost_increase := new_lumber_cost - original_lumber_cost
  let percentage_increase := (lumber_cost_increase / original_lumber_cost) * 100
  percentage_increase = 21.56 := by
  sorry

end NUMINAMATH_CALUDE_lumber_price_increase_l1584_158434


namespace NUMINAMATH_CALUDE_joshua_finish_time_difference_l1584_158479

/-- Race parameters -/
def race_length : ℕ := 15
def uphill_length : ℕ := 5
def flat_length : ℕ := race_length - uphill_length

/-- Runner speeds (in minutes per mile) -/
def malcolm_flat_speed : ℕ := 4
def joshua_flat_speed : ℕ := 6
def malcolm_uphill_additional : ℕ := 2
def joshua_uphill_additional : ℕ := 3

/-- Calculate total race time for a runner -/
def total_race_time (flat_speed uphill_additional : ℕ) : ℕ :=
  flat_speed * flat_length + (flat_speed + uphill_additional) * uphill_length

/-- Theorem: Joshua finishes 35 minutes after Malcolm -/
theorem joshua_finish_time_difference :
  total_race_time joshua_flat_speed joshua_uphill_additional -
  total_race_time malcolm_flat_speed malcolm_uphill_additional = 35 := by
  sorry


end NUMINAMATH_CALUDE_joshua_finish_time_difference_l1584_158479


namespace NUMINAMATH_CALUDE_carla_water_consumption_l1584_158485

/-- Given the conditions of Carla's liquid consumption, prove that she drank 15 ounces of water. -/
theorem carla_water_consumption (water soda : ℝ) 
  (h1 : soda = 3 * water - 6)
  (h2 : water + soda = 54) :
  water = 15 := by
  sorry

end NUMINAMATH_CALUDE_carla_water_consumption_l1584_158485


namespace NUMINAMATH_CALUDE_third_number_value_l1584_158416

-- Define the proportion
def proportion (a b c d : ℚ) : Prop := a * d = b * c

-- State the theorem
theorem third_number_value : 
  ∃ (third_number : ℚ), 
    proportion (75/100) (6/5) third_number 8 ∧ third_number = 5 := by
  sorry

end NUMINAMATH_CALUDE_third_number_value_l1584_158416


namespace NUMINAMATH_CALUDE_james_change_calculation_l1584_158487

/-- Calculates the change James receives after purchasing items with discounts. -/
theorem james_change_calculation (candy_packs : ℕ) (chocolate_bars : ℕ) (chip_bags : ℕ)
  (candy_price : ℚ) (chocolate_price : ℚ) (chip_price : ℚ)
  (candy_discount : ℚ) (chip_discount : ℚ) (payment : ℚ) :
  candy_packs = 3 →
  chocolate_bars = 2 →
  chip_bags = 4 →
  candy_price = 12 →
  chocolate_price = 3 →
  chip_price = 2 →
  candy_discount = 15 / 100 →
  chip_discount = 10 / 100 →
  payment = 50 →
  let candy_total := candy_packs * candy_price * (1 - candy_discount)
  let chocolate_total := chocolate_price -- Due to buy-one-get-one-free offer
  let chip_total := chip_bags * chip_price * (1 - chip_discount)
  let total_cost := candy_total + chocolate_total + chip_total
  payment - total_cost = 9.2 := by sorry

end NUMINAMATH_CALUDE_james_change_calculation_l1584_158487


namespace NUMINAMATH_CALUDE_total_books_l1584_158493

theorem total_books (tim_books sam_books alex_books : ℕ) 
  (h1 : tim_books = 44)
  (h2 : sam_books = 52)
  (h3 : alex_books = 65) :
  tim_books + sam_books + alex_books = 161 := by
  sorry

end NUMINAMATH_CALUDE_total_books_l1584_158493


namespace NUMINAMATH_CALUDE_opposite_of_2023_l1584_158429

theorem opposite_of_2023 : -(2023 : ℤ) = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l1584_158429


namespace NUMINAMATH_CALUDE_two_a_minus_three_b_value_l1584_158422

theorem two_a_minus_three_b_value (a b : ℝ) (h1 : |a| = 2) (h2 : |b| = 3) (h3 : b < a) :
  2 * a - 3 * b = 13 ∨ 2 * a - 3 * b = 5 :=
by sorry

end NUMINAMATH_CALUDE_two_a_minus_three_b_value_l1584_158422


namespace NUMINAMATH_CALUDE_marble_selection_l1584_158426

theorem marble_selection (n m k b : ℕ) (h1 : n = 10) (h2 : m = 2) (h3 : k = 4) (h4 : b = 2) :
  (Nat.choose n k) - (Nat.choose (n - m) k) = 140 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_l1584_158426


namespace NUMINAMATH_CALUDE_min_value_expression_l1584_158413

theorem min_value_expression (x : ℝ) : 
  (∀ y : ℝ, (15 - y) * (8 - y) * (15 + y) * (8 + y) - 200 ≥ (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200) → 
  (15 - x) * (8 - x) * (15 + x) * (8 + x) - 200 = -6680.25 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1584_158413


namespace NUMINAMATH_CALUDE_brendan_rounds_won_all_l1584_158417

/-- The number of rounds where Brendan won all matches in a kickboxing competition -/
def rounds_won_all (total_matches_won : ℕ) (matches_per_full_round : ℕ) (last_round_matches : ℕ) : ℕ :=
  ((total_matches_won - (last_round_matches / 2)) / matches_per_full_round)

/-- Theorem stating that Brendan won all matches in 2 rounds -/
theorem brendan_rounds_won_all :
  rounds_won_all 14 6 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_brendan_rounds_won_all_l1584_158417


namespace NUMINAMATH_CALUDE_base_number_proof_l1584_158467

theorem base_number_proof (k : ℕ) (x : ℤ) 
  (h1 : 21^k ∣ 435961)
  (h2 : x^k - k^7 = 1) :
  x = 2 :=
sorry

end NUMINAMATH_CALUDE_base_number_proof_l1584_158467


namespace NUMINAMATH_CALUDE_eight_divides_Q_largest_divisor_eight_largest_divisor_l1584_158425

/-- The product of three consecutive positive even integers -/
def Q (n : ℕ) : ℕ := (2*n) * (2*n + 2) * (2*n + 4)

/-- 8 divides Q for all positive n -/
theorem eight_divides_Q (n : ℕ) : (8 : ℕ) ∣ Q n := by sorry

/-- For any d > 8, there exists an n such that d does not divide Q n -/
theorem largest_divisor (d : ℕ) (h : d > 8) : ∃ n : ℕ, ¬(d ∣ Q n) := by sorry

/-- 8 is the largest integer that divides Q for all positive n -/
theorem eight_largest_divisor : ∀ d : ℕ, (∀ n : ℕ, d ∣ Q n) → d ≤ 8 := by sorry

end NUMINAMATH_CALUDE_eight_divides_Q_largest_divisor_eight_largest_divisor_l1584_158425


namespace NUMINAMATH_CALUDE_m_fourth_plus_twice_m_cubed_minus_m_plus_2007_l1584_158447

theorem m_fourth_plus_twice_m_cubed_minus_m_plus_2007 (m : ℝ) 
  (h : m^2 + m - 1 = 0) : 
  m^4 + 2*m^3 - m + 2007 = 2007 := by
  sorry

end NUMINAMATH_CALUDE_m_fourth_plus_twice_m_cubed_minus_m_plus_2007_l1584_158447


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1584_158438

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n + 2) →  -- arithmetic sequence with common difference 2
  (a 3)^2 = a 1 * a 4 →         -- a₁, a₃, a₄ form a geometric sequence
  a 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l1584_158438


namespace NUMINAMATH_CALUDE_sum_division_l1584_158455

theorem sum_division (x y z : ℝ) (total : ℝ) (y_share : ℝ) : 
  total = 245 →
  y_share = 63 →
  y = 0.45 * x →
  total = x + y + z →
  z / x = 0.30 := by
  sorry

end NUMINAMATH_CALUDE_sum_division_l1584_158455


namespace NUMINAMATH_CALUDE_junior_score_junior_score_is_89_l1584_158456

theorem junior_score (n : ℝ) (junior_ratio : ℝ) (senior_ratio : ℝ) 
  (overall_avg : ℝ) (senior_avg : ℝ) : ℝ :=
  let junior_count := junior_ratio * n
  let senior_count := senior_ratio * n
  let total_score := overall_avg * n
  let senior_total := senior_avg * senior_count
  let junior_total := total_score - senior_total
  junior_total / junior_count

theorem junior_score_is_89 :
  junior_score 100 0.2 0.8 85 84 = 89 := by
  sorry

end NUMINAMATH_CALUDE_junior_score_junior_score_is_89_l1584_158456
