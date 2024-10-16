import Mathlib

namespace NUMINAMATH_CALUDE_zero_subset_A_l4152_415254

def A : Set ℝ := {x | x > -3}

theorem zero_subset_A : {0} ⊆ A := by
  sorry

end NUMINAMATH_CALUDE_zero_subset_A_l4152_415254


namespace NUMINAMATH_CALUDE_circumcenter_minimizes_max_distance_l4152_415293

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle in a 2D plane. -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- The distance between two points. -/
def distance (p1 p2 : Point2D) : ℝ := sorry

/-- Checks if a point is inside or on the boundary of a triangle. -/
def isInsideOrOnBoundary (p : Point2D) (t : Triangle) : Prop := sorry

/-- Checks if a triangle is acute or right. -/
def isAcuteOrRight (t : Triangle) : Prop := sorry

/-- The circumcenter of a triangle. -/
def circumcenter (t : Triangle) : Point2D := sorry

/-- Theorem: The point that minimizes the maximum distance to the vertices of an acute or right triangle is its circumcenter. -/
theorem circumcenter_minimizes_max_distance (t : Triangle) (h : isAcuteOrRight t) :
  ∀ p, isInsideOrOnBoundary p t →
    distance p t.A ≤ distance (circumcenter t) t.A ∧
    distance p t.B ≤ distance (circumcenter t) t.B ∧
    distance p t.C ≤ distance (circumcenter t) t.C :=
  sorry

end NUMINAMATH_CALUDE_circumcenter_minimizes_max_distance_l4152_415293


namespace NUMINAMATH_CALUDE_alok_chapati_order_l4152_415279

-- Define the variables
def rice_plates : ℕ := 5
def vegetable_plates : ℕ := 7
def ice_cream_cups : ℕ := 6
def chapati_cost : ℕ := 6
def rice_cost : ℕ := 45
def vegetable_cost : ℕ := 70
def total_paid : ℕ := 1111

-- Define the theorem
theorem alok_chapati_order :
  ∃ (chapatis : ℕ), 
    chapatis * chapati_cost + 
    rice_plates * rice_cost + 
    vegetable_plates * vegetable_cost + 
    ice_cream_cups * (total_paid - (chapatis * chapati_cost + rice_plates * rice_cost + vegetable_plates * vegetable_cost)) / ice_cream_cups = 
    total_paid ∧ 
    chapatis = 66 := by
  sorry

end NUMINAMATH_CALUDE_alok_chapati_order_l4152_415279


namespace NUMINAMATH_CALUDE_equation_solution_l4152_415214

theorem equation_solution : ∃ x : ℝ, (64 : ℝ)^(3*x) = (16 : ℝ)^(4*x - 5) ↔ x = -10 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l4152_415214


namespace NUMINAMATH_CALUDE_candy_shipment_proof_l4152_415282

/-- Represents the number of cases of each candy type in a shipment -/
structure CandyShipment where
  chocolate : ℕ
  lollipops : ℕ
  gummy_bears : ℕ

/-- The ratio of chocolate bars to lollipops to gummy bears -/
def candy_ratio : CandyShipment := ⟨3, 2, 1⟩

/-- The actual shipment received -/
def actual_shipment : CandyShipment := ⟨36, 48, 24⟩

theorem candy_shipment_proof :
  (actual_shipment.chocolate / candy_ratio.chocolate = 
   actual_shipment.lollipops / candy_ratio.lollipops) ∧
  (actual_shipment.gummy_bears = 
   actual_shipment.chocolate / candy_ratio.chocolate * candy_ratio.gummy_bears) ∧
  (actual_shipment.chocolate + actual_shipment.lollipops + actual_shipment.gummy_bears = 108) :=
by sorry

#check candy_shipment_proof

end NUMINAMATH_CALUDE_candy_shipment_proof_l4152_415282


namespace NUMINAMATH_CALUDE_max_value_of_trig_function_l4152_415269

theorem max_value_of_trig_function :
  ∃ (M : ℝ), M = Real.sqrt 5 / 2 ∧ 
  (∀ x : ℝ, 2 * (Real.cos x)^2 + Real.sin x * Real.cos x - 1 ≤ M) ∧
  (∃ x : ℝ, 2 * (Real.cos x)^2 + Real.sin x * Real.cos x - 1 = M) := by
sorry

end NUMINAMATH_CALUDE_max_value_of_trig_function_l4152_415269


namespace NUMINAMATH_CALUDE_event_organization_ways_l4152_415276

def number_of_friends : ℕ := 5
def number_of_organizers : ℕ := 3

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem event_organization_ways :
  choose number_of_friends number_of_organizers = 10 := by
  sorry

end NUMINAMATH_CALUDE_event_organization_ways_l4152_415276


namespace NUMINAMATH_CALUDE_quadratic_binomial_square_l4152_415202

theorem quadratic_binomial_square (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 - 50*x + c = (x - a)^2) → c = 625 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_binomial_square_l4152_415202


namespace NUMINAMATH_CALUDE_conference_left_handed_fraction_l4152_415221

theorem conference_left_handed_fraction 
  (total : ℕ) 
  (red : ℕ) 
  (blue : ℕ) 
  (h1 : red + blue = total) 
  (h2 : red = 2 * blue) 
  (h3 : red > 0) 
  (h4 : blue > 0) : 
  (red * (1/3) + blue * (2/3)) / total = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_conference_left_handed_fraction_l4152_415221


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l4152_415287

theorem quadratic_expression_value (x : ℝ) (h : x^2 + 2*x - 2 = 0) : x*(x+2) + 3 = 5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l4152_415287


namespace NUMINAMATH_CALUDE_percentage_of_indian_men_l4152_415200

theorem percentage_of_indian_men (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percentage_indian_women : ℚ) (percentage_indian_children : ℚ)
  (percentage_not_indian : ℚ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  percentage_indian_women = 60 / 100 →
  percentage_indian_children = 70 / 100 →
  percentage_not_indian = 55.38461538461539 / 100 →
  (total_men * (10 / 100) + total_women * percentage_indian_women + total_children * percentage_indian_children : ℚ) =
  (total_men + total_women + total_children : ℕ) * (1 - percentage_not_indian) :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_indian_men_l4152_415200


namespace NUMINAMATH_CALUDE_trapezoid_segment_equality_l4152_415236

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a trapezoid -/
structure Trapezoid where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

/-- Checks if two line segments are parallel -/
def areParallel (p1 p2 p3 p4 : Point2D) : Prop :=
  (p2.y - p1.y) * (p4.x - p3.x) = (p2.x - p1.x) * (p4.y - p3.y)

/-- Checks if a point is on a line segment -/
def isOnSegment (p q r : Point2D) : Prop :=
  q.x <= max p.x r.x ∧ q.x >= min p.x r.x ∧
  q.y <= max p.y r.y ∧ q.y >= min p.y r.y

/-- Represents the intersection of two line segments -/
def intersect (p1 p2 p3 p4 : Point2D) : Option Point2D :=
  sorry -- Implementation omitted for brevity

theorem trapezoid_segment_equality (ABCD : Trapezoid) (M N K L : Point2D) :
  areParallel ABCD.B ABCD.C M N →
  isOnSegment ABCD.A ABCD.B M →
  isOnSegment ABCD.C ABCD.D N →
  intersect M N ABCD.A ABCD.C = some K →
  intersect M N ABCD.B ABCD.D = some L →
  (K.x - M.x)^2 + (K.y - M.y)^2 = (L.x - N.x)^2 + (L.y - N.y)^2 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_segment_equality_l4152_415236


namespace NUMINAMATH_CALUDE_no_integer_solutions_l4152_415275

theorem no_integer_solutions : ¬∃ (x y : ℤ), x = x^2 + y^2 + 1 ∧ y = 3*x*y := by sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l4152_415275


namespace NUMINAMATH_CALUDE_problem_solution_l4152_415299

theorem problem_solution (a b c : ℝ) (h1 : a < 0) (h2 : 0 < b) (h3 : b < c) :
  (a^2 * b < a^2 * c) ∧ (a^3 < a^2 * b) ∧ (a + b < b + c) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4152_415299


namespace NUMINAMATH_CALUDE_domino_tiling_triomino_tiling_l_tetromino_tiling_l4152_415278

/-- Represents a chessboard --/
structure Chessboard :=
  (size : ℕ)

/-- Represents a tile --/
structure Tile :=
  (size : ℕ)

/-- Defines a 9x9 chessboard --/
def chessboard_9x9 : Chessboard :=
  ⟨9 * 9⟩

/-- Defines a 2x1 domino --/
def domino : Tile :=
  ⟨2⟩

/-- Defines a 3x1 triomino --/
def triomino : Tile :=
  ⟨3⟩

/-- Defines an L-shaped tetromino --/
def l_tetromino : Tile :=
  ⟨4⟩

/-- Determines if a chessboard can be tiled with a given tile --/
def can_tile (c : Chessboard) (t : Tile) : Prop :=
  c.size % t.size = 0

theorem domino_tiling :
  ¬ can_tile chessboard_9x9 domino :=
sorry

theorem triomino_tiling :
  can_tile chessboard_9x9 triomino :=
sorry

theorem l_tetromino_tiling :
  ¬ can_tile chessboard_9x9 l_tetromino :=
sorry

end NUMINAMATH_CALUDE_domino_tiling_triomino_tiling_l_tetromino_tiling_l4152_415278


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l4152_415259

theorem circle_diameter_ratio (D C : Real) (h1 : D = 20) 
  (h2 : C > 0) (h3 : C < D) 
  (h4 : (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 4) : 
  C = 4 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l4152_415259


namespace NUMINAMATH_CALUDE_first_reduction_percentage_l4152_415260

theorem first_reduction_percentage (x : ℝ) : 
  (1 - 0.7) * (1 - x / 100) = 1 - 0.775 → x = 25 := by sorry

end NUMINAMATH_CALUDE_first_reduction_percentage_l4152_415260


namespace NUMINAMATH_CALUDE_unique_equal_sum_existence_l4152_415235

/-- The sum of the first n terms of an arithmetic sequence with first term a and common difference d -/
def arithmetic_sum (a d n : ℤ) : ℤ := n * (2 * a + (n - 1) * d) / 2

/-- The statement that there exists exactly one positive integer n such that
    the sum of the first n terms of the arithmetic sequence (8, 12, ...)
    equals the sum of the first n terms of the arithmetic sequence (17, 19, ...) -/
theorem unique_equal_sum_existence : ∃! (n : ℕ), n > 0 ∧ 
  arithmetic_sum 8 4 n = arithmetic_sum 17 2 n := by sorry

end NUMINAMATH_CALUDE_unique_equal_sum_existence_l4152_415235


namespace NUMINAMATH_CALUDE_smallest_perimeter_square_sides_l4152_415247

theorem smallest_perimeter_square_sides : ∃ (a b c : ℕ), 
  (0 < a ∧ 0 < b ∧ 0 < c) ∧  -- positive integers
  (a < b ∧ b < c) ∧  -- distinct
  (a^2 + b^2 > c^2) ∧  -- triangle inequality
  (a^2 + c^2 > b^2) ∧
  (b^2 + c^2 > a^2) ∧
  (a^2 + b^2 + c^2 = 77) ∧  -- perimeter is 77
  (∀ (x y z : ℕ), (0 < x ∧ 0 < y ∧ 0 < z) →
    (x < y ∧ y < z) →
    (x^2 + y^2 > z^2) →
    (x^2 + z^2 > y^2) →
    (y^2 + z^2 > x^2) →
    (x^2 + y^2 + z^2 ≥ 77)) :=
by
  sorry

#check smallest_perimeter_square_sides

end NUMINAMATH_CALUDE_smallest_perimeter_square_sides_l4152_415247


namespace NUMINAMATH_CALUDE_total_cookie_sales_l4152_415288

/-- Represents the sales data for Robyn and Lucy's cookie selling adventure -/
structure CookieSales where
  /-- Sales in the first neighborhood -/
  neighborhood1 : Nat × Nat
  /-- Sales in the second neighborhood -/
  neighborhood2 : Nat × Nat
  /-- Sales in the third neighborhood -/
  neighborhood3 : Nat × Nat
  /-- Total sales in the first park -/
  park1_total : Nat
  /-- Total sales in the second park -/
  park2_total : Nat

/-- Theorem stating the total number of packs sold by Robyn and Lucy -/
theorem total_cookie_sales (sales : CookieSales)
  (h1 : sales.neighborhood1 = (15, 12))
  (h2 : sales.neighborhood2 = (23, 15))
  (h3 : sales.neighborhood3 = (17, 16))
  (h4 : sales.park1_total = 25)
  (h5 : ∃ x y : Nat, x = 2 * y ∧ x + y = sales.park1_total)
  (h6 : sales.park2_total = 35)
  (h7 : ∃ x y : Nat, y = x + 5 ∧ x + y = sales.park2_total) :
  (sales.neighborhood1.1 + sales.neighborhood1.2 +
   sales.neighborhood2.1 + sales.neighborhood2.2 +
   sales.neighborhood3.1 + sales.neighborhood3.2 +
   sales.park1_total + sales.park2_total) = 158 := by
  sorry

end NUMINAMATH_CALUDE_total_cookie_sales_l4152_415288


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l4152_415249

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- Sum function

/-- Theorem: For an arithmetic sequence, if S_p = q and S_q = p where p ≠ q, then S_{p+q} = -(p + q) -/
theorem arithmetic_sequence_sum_property (a : ArithmeticSequence) (p q : ℕ) 
    (h1 : a.S p = q)
    (h2 : a.S q = p)
    (h3 : p ≠ q) : 
  a.S (p + q) = -(p + q) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l4152_415249


namespace NUMINAMATH_CALUDE_root_square_plus_inverse_square_l4152_415270

theorem root_square_plus_inverse_square (m : ℝ) : 
  m^2 - 2*m - 1 = 0 → m^2 + 1/m^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_root_square_plus_inverse_square_l4152_415270


namespace NUMINAMATH_CALUDE_collinear_points_values_coplanar_points_value_l4152_415255

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop :=
  ∃ t : ℝ, (p3.x - p1.x, p3.y - p1.y, p3.z - p1.z) = 
    t • (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z)

/-- Check if four points are coplanar -/
def coplanar (p1 p2 p3 p4 : Point3D) : Prop :=
  ∃ a b c : ℝ, 
    (p4.x - p1.x, p4.y - p1.y, p4.z - p1.z) = 
    a • (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z) +
    b • (p3.x - p1.x, p3.y - p1.y, p3.z - p1.z)

theorem collinear_points_values (a b : ℝ) :
  let A : Point3D := ⟨2, a, -1⟩
  let B : Point3D := ⟨-2, 3, b⟩
  let C : Point3D := ⟨1, 2, -2⟩
  collinear A B C → a = 5/3 ∧ b = -5 := by
  sorry

theorem coplanar_points_value (a : ℝ) :
  let A : Point3D := ⟨2, a, -1⟩
  let B : Point3D := ⟨-2, 3, -3⟩
  let C : Point3D := ⟨1, 2, -2⟩
  let D : Point3D := ⟨-1, 3, -3⟩
  coplanar A B C D → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_values_coplanar_points_value_l4152_415255


namespace NUMINAMATH_CALUDE_vinnie_tips_l4152_415292

theorem vinnie_tips (paul_tips : ℕ) (vinnie_more : ℕ) : 
  paul_tips = 14 → vinnie_more = 16 → paul_tips + vinnie_more = 30 := by
  sorry

end NUMINAMATH_CALUDE_vinnie_tips_l4152_415292


namespace NUMINAMATH_CALUDE_bee_count_l4152_415284

theorem bee_count (initial_bees : ℕ) (incoming_bees : ℕ) : 
  initial_bees = 16 → incoming_bees = 9 → initial_bees + incoming_bees = 25 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l4152_415284


namespace NUMINAMATH_CALUDE_roots_reciprocal_sum_l4152_415233

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 5 = 0 → x₂^2 - 2*x₂ - 5 = 0 → 1/x₁ + 1/x₂ = -2/5 := by
  sorry

end NUMINAMATH_CALUDE_roots_reciprocal_sum_l4152_415233


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l4152_415203

theorem arithmetic_square_root_of_one_fourth :
  let x : ℚ := 1/2
  (x * x = 1/4) ∧ (∀ y : ℚ, y * y = 1/4 → y = x ∨ y = -x) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_one_fourth_l4152_415203


namespace NUMINAMATH_CALUDE_apple_pie_cost_per_serving_l4152_415237

/-- Calculates the cost per serving of an apple pie given the ingredients and their costs. -/
def cost_per_serving (granny_smith_weight : Float) (granny_smith_price : Float)
                     (gala_weight : Float) (gala_price : Float)
                     (honeycrisp_weight : Float) (honeycrisp_price : Float)
                     (pie_crust_price : Float) (lemon_price : Float) (butter_price : Float)
                     (servings : Nat) : Float :=
  let total_cost := granny_smith_weight * granny_smith_price +
                    gala_weight * gala_price +
                    honeycrisp_weight * honeycrisp_price +
                    pie_crust_price + lemon_price + butter_price
  total_cost / servings.toFloat

/-- The cost per serving of the apple pie is $1.16375. -/
theorem apple_pie_cost_per_serving :
  cost_per_serving 0.5 1.80 0.8 2.20 0.7 2.50 2.50 0.60 1.80 8 = 1.16375 := by
  sorry

end NUMINAMATH_CALUDE_apple_pie_cost_per_serving_l4152_415237


namespace NUMINAMATH_CALUDE_sqrt_100_equals_10_l4152_415212

theorem sqrt_100_equals_10 : Real.sqrt 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_100_equals_10_l4152_415212


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_l4152_415216

theorem quadratic_equation_solutions :
  let f : ℝ → ℝ := λ x ↦ x^2 - x
  ∃ (x₁ x₂ : ℝ), (f x₁ = 0 ∧ f x₂ = 0) ∧ x₁ = 0 ∧ x₂ = 1 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_l4152_415216


namespace NUMINAMATH_CALUDE_sum_squares_interior_8th_row_l4152_415289

/-- Pascal's Triangle row function -/
def pascal_row (n : ℕ) : List ℕ := sorry

/-- Function to get interior numbers of a row -/
def interior_numbers (row : List ℕ) : List ℕ := sorry

/-- Sum of squares function -/
def sum_of_squares (list : List ℕ) : ℕ := sorry

/-- Theorem: Sum of squares of interior numbers in 8th row of Pascal's Triangle is 3430 -/
theorem sum_squares_interior_8th_row : 
  sum_of_squares (interior_numbers (pascal_row 8)) = 3430 := by sorry

end NUMINAMATH_CALUDE_sum_squares_interior_8th_row_l4152_415289


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l4152_415213

theorem company_picnic_attendance (men_attendance : Real) (women_attendance : Real) 
  (total_attendance : Real) (men_percentage : Real) :
  men_attendance = 0.2 →
  women_attendance = 0.4 →
  total_attendance = 0.31000000000000007 →
  men_attendance * men_percentage + women_attendance * (1 - men_percentage) = total_attendance →
  men_percentage = 0.45 := by
sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l4152_415213


namespace NUMINAMATH_CALUDE_ellipse_equation_l4152_415277

/-- Represents an ellipse with its properties -/
structure Ellipse where
  a : ℝ  -- Length of semi-major axis
  b : ℝ  -- Length of semi-minor axis
  c : ℝ  -- Distance from center to focus

/-- The standard equation of an ellipse -/
def standardEquation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Conditions for the ellipse -/
def ellipseConditions (e : Ellipse) : Prop :=
  e.a = 2 * Real.sqrt 3 ∧
  e.c = Real.sqrt 3 ∧
  e.b^2 = e.a^2 - e.c^2 ∧
  e.a = 3 * e.b ∧
  standardEquation e 3 0

theorem ellipse_equation (e : Ellipse) (h : ellipseConditions e) :
  (∀ x y, standardEquation e x y ↔ x^2 / 12 + y^2 / 9 = 1) ∨
  (∀ x y, standardEquation e x y ↔ x^2 / 9 + y^2 / 12 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l4152_415277


namespace NUMINAMATH_CALUDE_sum_of_squares_l4152_415272

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 21) (h2 : x * y = 43) : x^2 + y^2 = 355 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l4152_415272


namespace NUMINAMATH_CALUDE_systematic_sampling_result_l4152_415225

/-- Represents a systematic sampling result -/
structure SystematicSample where
  first : Nat
  interval : Nat
  size : Nat

/-- Generates a sequence of numbers using systematic sampling -/
def generateSequence (sample : SystematicSample) : List Nat :=
  List.range sample.size |>.map (fun i => sample.first + i * sample.interval)

/-- Checks if a sequence is within the given range -/
def isWithinRange (seq : List Nat) (maxVal : Nat) : Prop :=
  seq.all (· ≤ maxVal)

theorem systematic_sampling_result :
  let classSize : Nat := 50
  let sampleSize : Nat := 5
  let result : List Nat := [5, 15, 25, 35, 45]
  ∃ (sample : SystematicSample),
    sample.size = sampleSize ∧
    sample.interval = classSize / sampleSize ∧
    generateSequence sample = result ∧
    isWithinRange result classSize :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_result_l4152_415225


namespace NUMINAMATH_CALUDE_marble_remainder_l4152_415220

theorem marble_remainder (r p : ℤ) 
  (hr : r % 8 = 5) 
  (hp : p % 8 = 6) : 
  (r + p) % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l4152_415220


namespace NUMINAMATH_CALUDE_necessary_condition_for_greater_than_l4152_415263

theorem necessary_condition_for_greater_than (a b : ℝ) : a > b → a > b - 1 := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_greater_than_l4152_415263


namespace NUMINAMATH_CALUDE_min_sum_of_product_72_l4152_415283

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) : 
  (∀ x y : ℤ, x * y = 72 → a + b ≤ x + y) ∧ (∃ x y : ℤ, x * y = 72 ∧ x + y = -17) := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_product_72_l4152_415283


namespace NUMINAMATH_CALUDE_arithmetic_mean_fraction_l4152_415206

theorem arithmetic_mean_fraction (x b : ℝ) (h : x ≠ 0) :
  ((x + b) / x + (x - 2 * b) / x) / 2 = 1 - b / (2 * x) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_fraction_l4152_415206


namespace NUMINAMATH_CALUDE_line_vector_proof_l4152_415211

def line_vector (t : ℝ) : ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 1 = (2, 5) ∧ line_vector 4 = (5, -7)) →
  line_vector (-3) = (-2, 21) := by sorry

end NUMINAMATH_CALUDE_line_vector_proof_l4152_415211


namespace NUMINAMATH_CALUDE_base4_multiplication_division_l4152_415281

-- Define a function to convert from base 4 to base 10
def base4ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 4
def base10ToBase4 (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem base4_multiplication_division :
  base10ToBase4 (base4ToBase10 132 * base4ToBase10 22 / base4ToBase10 3) = 154 := by sorry

end NUMINAMATH_CALUDE_base4_multiplication_division_l4152_415281


namespace NUMINAMATH_CALUDE_complete_square_with_integer_l4152_415273

theorem complete_square_with_integer (y : ℝ) :
  ∃ (k : ℤ) (b : ℝ), y^2 + 12*y + 44 = (y + b)^2 + k := by
  sorry

end NUMINAMATH_CALUDE_complete_square_with_integer_l4152_415273


namespace NUMINAMATH_CALUDE_stratified_sampling_second_year_selection_l4152_415215

theorem stratified_sampling_second_year_selection
  (total_students : ℕ)
  (first_year_students : ℕ)
  (second_year_students : ℕ)
  (first_year_selected : ℕ)
  (h1 : total_students = 70)
  (h2 : first_year_students = 30)
  (h3 : second_year_students = 40)
  (h4 : first_year_selected = 6)
  (h5 : total_students = first_year_students + second_year_students) :
  (first_year_selected : ℚ) / first_year_students * second_year_students = 8 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_year_selection_l4152_415215


namespace NUMINAMATH_CALUDE_parabola_point_distance_to_focus_l4152_415222

/-- Given a parabola y = 4x² and a point M(x, y) on the parabola,
    if the distance from M to the focus (0, 1/16) is 1,
    then the y-coordinate of M is 15/16 -/
theorem parabola_point_distance_to_focus (x y : ℝ) :
  y = 4 * x^2 →
  (x - 0)^2 + (y - 1/16)^2 = 1 →
  y = 15/16 := by
sorry

end NUMINAMATH_CALUDE_parabola_point_distance_to_focus_l4152_415222


namespace NUMINAMATH_CALUDE_finite_valid_combinations_l4152_415280

/-- Represents the number of banknotes of each denomination --/
structure Banknotes :=
  (hun : Nat)
  (fif : Nat)
  (twe : Nat)
  (ten : Nat)

/-- The total value of a set of banknotes in yuan --/
def totalValue (b : Banknotes) : Nat :=
  100 * b.hun + 50 * b.fif + 20 * b.twe + 10 * b.ten

/-- The available banknotes --/
def availableBanknotes : Banknotes :=
  ⟨1, 2, 5, 10⟩

/-- A valid combination of banknotes is one that sums to 200 yuan and doesn't exceed the available banknotes --/
def isValidCombination (b : Banknotes) : Prop :=
  totalValue b = 200 ∧
  b.hun ≤ availableBanknotes.hun ∧
  b.fif ≤ availableBanknotes.fif ∧
  b.twe ≤ availableBanknotes.twe ∧
  b.ten ≤ availableBanknotes.ten

theorem finite_valid_combinations :
  ∃ (n : Nat), ∃ (combinations : Finset Banknotes),
    combinations.card = n ∧
    (∀ b ∈ combinations, isValidCombination b) ∧
    (∀ b, isValidCombination b → b ∈ combinations) :=
by sorry

end NUMINAMATH_CALUDE_finite_valid_combinations_l4152_415280


namespace NUMINAMATH_CALUDE_max_l_value_l4152_415229

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 8 * x + 3

-- Define the condition for l(a)
def is_valid_l (a : ℝ) (l : ℝ) : Prop :=
  a < 0 ∧ l > 0 ∧ ∀ x ∈ Set.Icc 0 l, |f a x| ≤ 5

-- Define l(a) as the supremum of valid l values
noncomputable def l (a : ℝ) : ℝ :=
  ⨆ (l : ℝ) (h : is_valid_l a l), l

-- State the theorem
theorem max_l_value :
  ∃ (a : ℝ), a < 0 ∧ l a = (Real.sqrt 5 + 1) / 2 ∧
  ∀ (b : ℝ), b < 0 → l b ≤ l a :=
sorry

end NUMINAMATH_CALUDE_max_l_value_l4152_415229


namespace NUMINAMATH_CALUDE_power_division_simplification_l4152_415243

theorem power_division_simplification : 8^15 / 64^3 = 8^9 := by
  sorry

end NUMINAMATH_CALUDE_power_division_simplification_l4152_415243


namespace NUMINAMATH_CALUDE_simplify_x_expression_simplify_a_expression_l4152_415226

-- First equation
theorem simplify_x_expression (x : ℝ) : 3 * x^4 * x^2 + (2 * x^2)^3 = 11 * x^6 := by
  sorry

-- Second equation
theorem simplify_a_expression (a : ℝ) : 3 * a * (9 * a + 3) - 4 * a * (2 * a - 1) = 19 * a^2 + 13 * a := by
  sorry

end NUMINAMATH_CALUDE_simplify_x_expression_simplify_a_expression_l4152_415226


namespace NUMINAMATH_CALUDE_range_of_m_l4152_415274

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : 4 * x + y - x * y = 0) (h2 : x * y ≥ m^2 - 6*m) : 
  -2 ≤ m ∧ m ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l4152_415274


namespace NUMINAMATH_CALUDE_problem_solution_l4152_415223

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x : ℝ | a + 1 ≤ x ∧ x ≤ 2 * a - 1}

theorem problem_solution :
  (∀ x : ℝ, x ∈ (M ∪ N (7/2)) ↔ -2 ≤ x ∧ x ≤ 6) ∧
  (∀ x : ℝ, x ∈ ((Set.univ \ M) ∩ N (7/2)) ↔ 5 < x ∧ x ≤ 6) ∧
  (∀ a : ℝ, M ⊇ N a ↔ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l4152_415223


namespace NUMINAMATH_CALUDE_mirror_area_l4152_415286

/-- The area of a rectangular mirror with a frame -/
theorem mirror_area (overall_length overall_width frame_width : ℝ) 
  (h1 : overall_length = 100)
  (h2 : overall_width = 50)
  (h3 : frame_width = 8) : 
  (overall_length - 2 * frame_width) * (overall_width - 2 * frame_width) = 2856 := by
  sorry

end NUMINAMATH_CALUDE_mirror_area_l4152_415286


namespace NUMINAMATH_CALUDE_absent_men_count_l4152_415256

/-- Proves the number of absent men in a work group --/
theorem absent_men_count (total_men : ℕ) (original_days : ℕ) (actual_days : ℕ) 
  (h1 : total_men = 20)
  (h2 : original_days = 20)
  (h3 : actual_days = 40)
  (h4 : total_men * original_days = (total_men - absent_men) * actual_days) :
  absent_men = 10 := by
  sorry

#check absent_men_count

end NUMINAMATH_CALUDE_absent_men_count_l4152_415256


namespace NUMINAMATH_CALUDE_isosceles_triangle_apex_angle_l4152_415290

-- Define an isosceles triangle
structure IsoscelesTriangle where
  base_angle : ℝ
  apex_angle : ℝ
  is_isosceles : base_angle ≥ 0 ∧ apex_angle ≥ 0
  angle_sum : 2 * base_angle + apex_angle = 180

-- Theorem statement
theorem isosceles_triangle_apex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.base_angle = 42) : 
  triangle.apex_angle = 96 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_apex_angle_l4152_415290


namespace NUMINAMATH_CALUDE_bus_passenger_count_l4152_415295

/-- The number of passengers who got on at the first stop -/
def passengers_first_stop : ℕ := 16

theorem bus_passenger_count : passengers_first_stop = 16 :=
  let initial_passengers : ℕ := 50
  let final_passengers : ℕ := 49
  let passengers_off : ℕ := 22
  let passengers_on_other_stops : ℕ := 5
  have h : initial_passengers + passengers_first_stop - (passengers_off - passengers_on_other_stops) = final_passengers :=
    by sorry
  by sorry

end NUMINAMATH_CALUDE_bus_passenger_count_l4152_415295


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l4152_415219

-- Define the quadratic function
def f (a c x : ℝ) := a * x^2 + x + c

-- Define the solution set of the original inequality
def S := {x : ℝ | 1 < x ∧ x < 3}

-- Define the solution set A
def A (a c : ℝ) := {x : ℝ | a * x^2 + 2*x + 4*c > 0}

-- Define the solution set B
def B (a c m : ℝ) := {x : ℝ | 3*a*x + c*m < 0}

-- State the theorem
theorem quadratic_inequality_problem 
  (h1 : ∀ x, x ∈ S ↔ f a c x > 0)
  (h2 : A a c ⊆ B a c m) :
  a = -1/4 ∧ c = -3/4 ∧ m ≥ -2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l4152_415219


namespace NUMINAMATH_CALUDE_average_k_for_quadratic_roots_l4152_415205

theorem average_k_for_quadratic_roots (k : ℤ) : 
  let factors := [(1, 24), (2, 12), (3, 8), (4, 6)]
  let k_values := factors.map (λ (a, b) => a + b)
  let distinct_k_values := k_values.eraseDups
  (distinct_k_values.sum / distinct_k_values.length : ℚ) = 15 := by
  sorry

end NUMINAMATH_CALUDE_average_k_for_quadratic_roots_l4152_415205


namespace NUMINAMATH_CALUDE_pipe_filling_time_l4152_415266

/-- Given two pipes A and B that fill a tank, where:
    - Pipe A fills the tank in t minutes
    - Pipe B fills the tank 3 times as fast as Pipe A
    - Both pipes together fill the tank in 3 minutes
    Then, Pipe A takes 12 minutes to fill the tank alone. -/
theorem pipe_filling_time (t : ℝ) 
  (hA : t > 0)  -- Pipe A's filling time is positive
  (hB : t / 3 > 0)  -- Pipe B's filling time is positive
  (h_both : 1 / t + 1 / (t / 3) = 1 / 3)  -- Combined filling rate equals 1/3
  : t = 12 := by
  sorry


end NUMINAMATH_CALUDE_pipe_filling_time_l4152_415266


namespace NUMINAMATH_CALUDE_marbles_cost_l4152_415228

/-- The amount spent on marbles, given the total spent on toys and the cost of a football -/
def amount_spent_on_marbles (total_spent : ℝ) (football_cost : ℝ) : ℝ :=
  total_spent - football_cost

/-- Theorem stating that the amount spent on marbles is $6.59 -/
theorem marbles_cost (total_spent : ℝ) (football_cost : ℝ)
  (h1 : total_spent = 12.30)
  (h2 : football_cost = 5.71) :
  amount_spent_on_marbles total_spent football_cost = 6.59 := by
  sorry

end NUMINAMATH_CALUDE_marbles_cost_l4152_415228


namespace NUMINAMATH_CALUDE_fourth_day_income_l4152_415218

def cab_driver_income (day1 day2 day3 day4 day5 : ℝ) : Prop :=
  day1 = 200 ∧ day2 = 150 ∧ day3 = 750 ∧ day5 = 500 ∧
  (day1 + day2 + day3 + day4 + day5) / 5 = 400

theorem fourth_day_income (day1 day2 day3 day4 day5 : ℝ) :
  cab_driver_income day1 day2 day3 day4 day5 → day4 = 400 := by
  sorry

end NUMINAMATH_CALUDE_fourth_day_income_l4152_415218


namespace NUMINAMATH_CALUDE_no_maximum_on_interval_l4152_415242

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 2 * m * x + 3

-- Define the property of being an even function
def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

-- Theorem statement
theorem no_maximum_on_interval (m : ℝ) :
  is_even_function (f m) →
  ¬∃ (y : ℝ), ∀ x ∈ Set.Ioo (-2 : ℝ) (-1), f m x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_no_maximum_on_interval_l4152_415242


namespace NUMINAMATH_CALUDE_function_equality_implies_sum_l4152_415265

theorem function_equality_implies_sum (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 4) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 14 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_implies_sum_l4152_415265


namespace NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l4152_415230

theorem tan_theta_minus_pi_fourth (θ : Real) :
  (-π/2 < θ) → (θ < 0) → -- θ is in the fourth quadrant
  (Real.sin (θ + π/4) = 3/5) →
  (Real.tan (θ - π/4) = -4/3) := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_minus_pi_fourth_l4152_415230


namespace NUMINAMATH_CALUDE_gifted_books_count_l4152_415252

def books_per_month : ℕ := 2
def months_per_year : ℕ := 12
def bought_books : ℕ := 8
def reread_old_books : ℕ := 4

def borrowed_books : ℕ := bought_books - 2

def total_books_needed : ℕ := books_per_month * months_per_year
def new_books_needed : ℕ := total_books_needed - reread_old_books
def new_books_acquired : ℕ := bought_books + borrowed_books

theorem gifted_books_count : new_books_needed - new_books_acquired = 6 := by
  sorry

end NUMINAMATH_CALUDE_gifted_books_count_l4152_415252


namespace NUMINAMATH_CALUDE_binary_conversion_and_sum_l4152_415291

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary1 : List Bool := [true, true, false, true, false, true, true]
def binary2 : List Bool := [false, true, true, false, true, false, true]

theorem binary_conversion_and_sum :
  (binary_to_decimal binary1 = 107) ∧
  (binary_to_decimal binary2 = 86) ∧
  (binary_to_decimal binary1 + binary_to_decimal binary2 = 193) := by
  sorry

end NUMINAMATH_CALUDE_binary_conversion_and_sum_l4152_415291


namespace NUMINAMATH_CALUDE_range_of_a_l4152_415271

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*a*x + 4 > 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → -(5-2*a)^x > -(5-2*a)^y

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) → a ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l4152_415271


namespace NUMINAMATH_CALUDE_count_four_digit_distinct_prime_last_l4152_415245

/-- A function that returns true if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- The set of single-digit prime numbers -/
def singleDigitPrimes : Finset ℕ := sorry

/-- A function that returns the set of digits of a natural number -/
def digits (n : ℕ) : Finset ℕ := sorry

/-- The count of four-digit numbers with distinct digits and a prime last digit -/
def countValidNumbers : ℕ := sorry

theorem count_four_digit_distinct_prime_last :
  countValidNumbers = 1344 := by
  sorry

end NUMINAMATH_CALUDE_count_four_digit_distinct_prime_last_l4152_415245


namespace NUMINAMATH_CALUDE_extremum_point_and_monotonicity_l4152_415251

noncomputable section

variables (x : ℝ) (m : ℝ)

def f (x : ℝ) : ℝ := Real.exp x - Real.log (x + m)

def f_derivative (x : ℝ) : ℝ := Real.exp x - 1 / (x + m)

theorem extremum_point_and_monotonicity :
  (f_derivative 0 = 0) →
  (m = 1) ∧
  (∀ x > 0, f_derivative x > 0) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f_derivative x < 0) :=
by sorry

end

end NUMINAMATH_CALUDE_extremum_point_and_monotonicity_l4152_415251


namespace NUMINAMATH_CALUDE_q_equals_six_l4152_415253

/-- Represents a digit from 4 to 9 -/
def Digit := {n : ℕ // 4 ≤ n ∧ n ≤ 9}

/-- The theorem stating that Q must be 6 given the conditions -/
theorem q_equals_six 
  (P Q R S T U : Digit) 
  (unique : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ P ≠ U ∧
            Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ Q ≠ U ∧
            R ≠ S ∧ R ≠ T ∧ R ≠ U ∧
            S ≠ T ∧ S ≠ U ∧
            T ≠ U)
  (sum_constraint : P.val + Q.val + S.val + 
                    T.val + U.val + R.val + 
                    P.val + T.val + S.val + 
                    R.val + Q.val + S.val + 
                    P.val + U.val = 100) : 
  Q.val = 6 := by
  sorry

end NUMINAMATH_CALUDE_q_equals_six_l4152_415253


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l4152_415244

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (avg_children_all : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 12)
  (h2 : avg_children_all = 3)
  (h3 : childless_families = 3)
  : (total_families * avg_children_all) / (total_families - childless_families) = 4 := by
  sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l4152_415244


namespace NUMINAMATH_CALUDE_min_ones_is_one_l4152_415296

/-- Represents the count of squares of each size --/
structure SquareCounts where
  threes : Nat
  twos : Nat
  ones : Nat

/-- Checks if the given square counts fit within a 7x7 square --/
def fitsIn7x7 (counts : SquareCounts) : Prop :=
  9 * counts.threes + 4 * counts.twos + counts.ones = 49

/-- Defines a valid square division --/
def isValidDivision (counts : SquareCounts) : Prop :=
  fitsIn7x7 counts ∧ counts.threes ≥ 0 ∧ counts.twos ≥ 0 ∧ counts.ones ≥ 0

/-- The main theorem stating that the minimum number of 1x1 squares is 1 --/
theorem min_ones_is_one :
  ∃ (counts : SquareCounts), isValidDivision counts ∧ counts.ones = 1 ∧
  (∀ (other : SquareCounts), isValidDivision other → other.ones ≥ counts.ones) :=
sorry

end NUMINAMATH_CALUDE_min_ones_is_one_l4152_415296


namespace NUMINAMATH_CALUDE_train_speed_problem_l4152_415210

/-- Proves that given two trains traveling towards each other from cities 100 miles apart,
    with one train traveling at 45 mph, if they meet after 1.33333333333 hours,
    then the speed of the other train must be 30 mph. -/
theorem train_speed_problem (distance : ℝ) (speed_train1 : ℝ) (time : ℝ) (speed_train2 : ℝ) : 
  distance = 100 →
  speed_train1 = 45 →
  time = 1.33333333333 →
  distance = speed_train1 * time + speed_train2 * time →
  speed_train2 = 30 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l4152_415210


namespace NUMINAMATH_CALUDE_cubes_in_box_l4152_415294

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit in each dimension -/
def cubesPerDimension (boxDim : ℕ) (cubeDim : ℕ) : ℕ :=
  boxDim / cubeDim

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (box : BoxDimensions) (cubeDim : ℕ) : ℕ :=
  (cubesPerDimension box.length cubeDim) *
  (cubesPerDimension box.width cubeDim) *
  (cubesPerDimension box.height cubeDim)

/-- Calculates the volume of a cube -/
def cubeVolume (cubeDim : ℕ) : ℕ :=
  cubeDim ^ 3

/-- Calculates the total volume of all cubes in the box -/
def totalCubesVolume (box : BoxDimensions) (cubeDim : ℕ) : ℕ :=
  (totalCubes box cubeDim) * (cubeVolume cubeDim)

/-- Theorem: The number of 4-inch cubes that can fit in the box is 6,
    and they occupy 100% of the box's volume -/
theorem cubes_in_box (box : BoxDimensions)
    (h1 : box.length = 8)
    (h2 : box.width = 4)
    (h3 : box.height = 12)
    (cubeDim : ℕ)
    (h4 : cubeDim = 4) :
    totalCubes box cubeDim = 6 ∧
    totalCubesVolume box cubeDim = boxVolume box := by
  sorry


end NUMINAMATH_CALUDE_cubes_in_box_l4152_415294


namespace NUMINAMATH_CALUDE_circle_C_radius_range_l4152_415217

-- Define the triangle vertices
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (3, 2)

-- Define the circumcircle H
def H : ℝ × ℝ := (0, 3)

-- Define the line BH
def lineBH (x y : ℝ) : Prop := 3 * x + y - 3 = 0

-- Define a point P on line segment BH
def P (m : ℝ) : ℝ × ℝ := (m, 3 - 3 * m)

-- Define the circle with center C
def circleC (r : ℝ) (x y : ℝ) : Prop := (x - 3)^2 + (y - 2)^2 = r^2

-- Define the theorem
theorem circle_C_radius_range :
  ∀ (r : ℝ), 
  (∀ (m : ℝ), 0 ≤ m ∧ m ≤ 1 →
    ∃ (x y : ℝ), 
      circleC r x y ∧
      circleC r ((x + m) / 2) ((y + (3 - 3 * m)) / 2) ∧
      x ≠ m ∧ y ≠ (3 - 3 * m)) →
  (∀ (m : ℝ), 0 ≤ m ∧ m ≤ 1 →
    (m - 3)^2 + (1 - 3 * m)^2 > r^2) →
  Real.sqrt 10 / 3 ≤ r ∧ r < 4 * Real.sqrt 10 / 5 :=
sorry


end NUMINAMATH_CALUDE_circle_C_radius_range_l4152_415217


namespace NUMINAMATH_CALUDE_jills_age_l4152_415268

/-- Given that the sum of Henry and Jill's present ages is 41, and 7 years ago Henry was twice the age of Jill, prove that Jill's present age is 16 years. -/
theorem jills_age (henry_age jill_age : ℕ) 
  (sum_of_ages : henry_age + jill_age = 41)
  (past_relation : henry_age - 7 = 2 * (jill_age - 7)) : 
  jill_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_jills_age_l4152_415268


namespace NUMINAMATH_CALUDE_rhombus_diagonals_not_always_equal_l4152_415261

/-- A rhombus is a quadrilateral with all four sides of equal length -/
structure Rhombus where
  sides : Fin 4 → ℝ
  all_sides_equal : ∀ i j : Fin 4, sides i = sides j

/-- The diagonals of a rhombus -/
def diagonals (r : Rhombus) : ℝ × ℝ :=
  sorry

/-- Theorem: The diagonals of a rhombus are not always equal -/
theorem rhombus_diagonals_not_always_equal :
  ¬ (∀ r : Rhombus, (diagonals r).1 = (diagonals r).2) :=
sorry

end NUMINAMATH_CALUDE_rhombus_diagonals_not_always_equal_l4152_415261


namespace NUMINAMATH_CALUDE_logarithm_and_exponent_calculation_l4152_415285

theorem logarithm_and_exponent_calculation :
  (2 * Real.log 10 / Real.log 5 + Real.log 0.25 / Real.log 5 = 2) ∧
  ((0.027 ^ (-1/3 : ℝ)) - ((-1/7 : ℝ)⁻¹) + ((2 + 7/9 : ℝ) ^ (1/2 : ℝ)) - ((Real.sqrt 2 - 1) ^ (0 : ℝ)) = 11) :=
by sorry

end NUMINAMATH_CALUDE_logarithm_and_exponent_calculation_l4152_415285


namespace NUMINAMATH_CALUDE_tripod_height_is_2_sqrt_5_l4152_415209

/-- A tripod with two legs of length 6 and one leg of length 4 -/
structure Tripod :=
  (leg1 : ℝ)
  (leg2 : ℝ)
  (leg3 : ℝ)
  (h : leg1 = 6)
  (i : leg2 = 6)
  (j : leg3 = 4)

/-- The height of the tripod when fully extended -/
def tripod_height (t : Tripod) : ℝ := sorry

/-- Theorem stating that the height of the tripod is 2√5 -/
theorem tripod_height_is_2_sqrt_5 (t : Tripod) : tripod_height t = 2 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_tripod_height_is_2_sqrt_5_l4152_415209


namespace NUMINAMATH_CALUDE_range_of_m_value_of_m_l4152_415241

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + m - 2

-- Define the condition for real roots
def has_real_roots (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, quadratic_equation m x₁ = 0 ∧ quadratic_equation m x₂ = 0

-- Define the additional condition
def additional_condition (x₁ x₂ : ℝ) : Prop :=
  (x₁ + 2) * (x₂ + 2) - 2 * x₁ * x₂ = 17

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  has_real_roots m → (m ≥ 2/3 ∧ m ≠ 1) :=
sorry

-- Theorem for the value of m given the additional condition
theorem value_of_m (m : ℝ) (x₁ x₂ : ℝ) :
  has_real_roots m →
  quadratic_equation m x₁ = 0 →
  quadratic_equation m x₂ = 0 →
  additional_condition x₁ x₂ →
  m = 3/2 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_value_of_m_l4152_415241


namespace NUMINAMATH_CALUDE_product_local_abs_value_l4152_415257

-- Define the complex number
def z : ℂ := 564823 + 3*Complex.I

-- Define the digit of interest
def digit : ℕ := 4

-- Define the local value of the digit in the complex number
def local_value : ℕ := 4000

-- Define the absolute value of the digit
def abs_digit : ℕ := 4

-- Theorem to prove
theorem product_local_abs_value : 
  local_value * abs_digit = 16000 := by sorry

end NUMINAMATH_CALUDE_product_local_abs_value_l4152_415257


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_l4152_415267

theorem sugar_recipe_reduction :
  let original_recipe : ℚ := 27/4  -- 6 3/4 cups
  let reduced_recipe : ℚ := (1/3) * original_recipe
  reduced_recipe = 9/4  -- 2 1/4 cups
  := by sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_l4152_415267


namespace NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l4152_415207

theorem sqrt_18_times_sqrt_32 : Real.sqrt 18 * Real.sqrt 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_18_times_sqrt_32_l4152_415207


namespace NUMINAMATH_CALUDE_cube_structure_surface_area_total_surface_area_is_1266_l4152_415250

/-- Represents a cube with a given side length -/
structure Cube where
  sideLength : ℕ

/-- Calculates the volume of a cube -/
def Cube.volume (c : Cube) : ℕ := c.sideLength ^ 3

/-- Calculates the surface area of a cube -/
def Cube.surfaceArea (c : Cube) : ℕ := 6 * c.sideLength ^ 2

/-- Represents the structure formed by the cubes -/
structure CubeStructure where
  cubes : List Cube
  stackedCubes : List Cube
  adjacentCube : Cube
  topCube : Cube

/-- Theorem stating the total surface area of the cube structure -/
theorem cube_structure_surface_area (cs : CubeStructure) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem total_surface_area_is_1266 (cs : CubeStructure) 
  (h1 : cs.cubes.length = 8)
  (h2 : (cs.cubes.map Cube.volume) = [1, 8, 27, 64, 125, 216, 512, 729])
  (h3 : cs.stackedCubes.length = 6)
  (h4 : cs.stackedCubes = (cs.cubes.take 6).reverse)
  (h5 : cs.adjacentCube = cs.cubes[6])
  (h6 : cs.adjacentCube.sideLength = 6)
  (h7 : cs.stackedCubes[4].sideLength = 5)
  (h8 : cs.topCube = cs.cubes[7])
  (h9 : cs.topCube.sideLength = 8) :
  cube_structure_surface_area cs = 1266 :=
sorry

end NUMINAMATH_CALUDE_cube_structure_surface_area_total_surface_area_is_1266_l4152_415250


namespace NUMINAMATH_CALUDE_ball_placement_count_l4152_415208

theorem ball_placement_count :
  let n_balls : ℕ := 4
  let n_boxes : ℕ := 3
  let placement_options := n_boxes ^ n_balls
  placement_options = 81 :=
by sorry

end NUMINAMATH_CALUDE_ball_placement_count_l4152_415208


namespace NUMINAMATH_CALUDE_right_triangle_with_median_condition_l4152_415258

theorem right_triangle_with_median_condition (c : ℝ) (h : c > 0) :
  ∃ (a b : ℝ),
    a > 0 ∧ b > 0 ∧
    a^2 + b^2 = c^2 ∧
    (c / 2)^2 = a * b ∧
    a = (c * (Real.sqrt 6 + Real.sqrt 2)) / 4 ∧
    b = (c * (Real.sqrt 6 - Real.sqrt 2)) / 4 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_median_condition_l4152_415258


namespace NUMINAMATH_CALUDE_tuna_weight_l4152_415224

/-- A fish market scenario where we need to determine the weight of each tuna. -/
theorem tuna_weight (total_customers : ℕ) (num_tuna : ℕ) (pounds_per_customer : ℕ) (unserved_customers : ℕ) :
  total_customers = 100 →
  num_tuna = 10 →
  pounds_per_customer = 25 →
  unserved_customers = 20 →
  (total_customers - unserved_customers) * pounds_per_customer / num_tuna = 200 := by
sorry

end NUMINAMATH_CALUDE_tuna_weight_l4152_415224


namespace NUMINAMATH_CALUDE_salary_reduction_percentage_l4152_415297

theorem salary_reduction_percentage (x : ℝ) : 
  (100 - x + (100 - x) * (11.11111111111111 / 100) = 100) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_salary_reduction_percentage_l4152_415297


namespace NUMINAMATH_CALUDE_fuel_cost_solution_l4152_415248

/-- Represents the fuel cost calculation problem --/
def fuel_cost_problem (truck_capacity : ℝ) (car_capacity : ℝ) (hybrid_capacity : ℝ)
  (truck_fullness : ℝ) (car_fullness : ℝ) (hybrid_fullness : ℝ)
  (diesel_price : ℝ) (gas_price : ℝ)
  (diesel_discount : ℝ) (gas_discount : ℝ) : Prop :=
  let truck_to_fill := truck_capacity * (1 - truck_fullness)
  let car_to_fill := car_capacity * (1 - car_fullness)
  let hybrid_to_fill := hybrid_capacity * (1 - hybrid_fullness)
  let diesel_discounted := diesel_price - diesel_discount
  let gas_discounted := gas_price - gas_discount
  let total_cost := truck_to_fill * diesel_discounted +
                    car_to_fill * gas_discounted +
                    hybrid_to_fill * gas_discounted
  total_cost = 95.88

/-- The main theorem stating the solution to the fuel cost problem --/
theorem fuel_cost_solution :
  fuel_cost_problem 25 15 10 0.5 (1/3) 0.25 3.5 3.2 0.1 0.15 :=
by sorry

end NUMINAMATH_CALUDE_fuel_cost_solution_l4152_415248


namespace NUMINAMATH_CALUDE_expression_value_l4152_415234

theorem expression_value (a : ℝ) (h : a^2 + 2*a + 2 - Real.sqrt 3 = 0) :
  1 / (a + 1) - (a + 3) / (a^2 - 1) * (a^2 - 2*a + 1) / (a^2 + 4*a + 3) = Real.sqrt 3 + 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l4152_415234


namespace NUMINAMATH_CALUDE_ochos_friends_l4152_415240

theorem ochos_friends (total : ℕ) (boys girls : ℕ) (h1 : boys = girls) (h2 : boys + girls = total) (h3 : boys = 4) : total = 8 := by
  sorry

end NUMINAMATH_CALUDE_ochos_friends_l4152_415240


namespace NUMINAMATH_CALUDE_sum_of_special_numbers_l4152_415232

theorem sum_of_special_numbers :
  ∀ A B : ℤ,
  (A = -3 - (-5)) →
  (B = 2 + (-2)) →
  A + B = 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_special_numbers_l4152_415232


namespace NUMINAMATH_CALUDE_x_percent_of_2x_is_10_l4152_415231

theorem x_percent_of_2x_is_10 (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * (2 * x) = 10) : 
  x = 10 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_x_percent_of_2x_is_10_l4152_415231


namespace NUMINAMATH_CALUDE_dans_initial_money_l4152_415239

/-- Represents Dan's money transactions -/
def dans_money (initial : ℕ) (candy_cost : ℕ) (chocolate_cost : ℕ) (remaining : ℕ) : Prop :=
  initial = candy_cost + chocolate_cost + remaining

theorem dans_initial_money : 
  ∃ (initial : ℕ), dans_money initial 2 3 2 ∧ initial = 7 := by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l4152_415239


namespace NUMINAMATH_CALUDE_parallel_segment_length_l4152_415246

/-- A trapezoid with bases a and b -/
structure Trapezoid (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)

/-- A line segment within a trapezoid -/
def ParallelSegment (t : Trapezoid a b) := ℝ

/-- The property that a line divides a trapezoid into two similar trapezoids -/
def DividesSimilarly (t : Trapezoid a b) (s : ParallelSegment t) : Prop :=
  sorry

/-- Theorem: If a line parallel to the bases divides a trapezoid into two similar trapezoids,
    then the length of the segment is the square root of the product of the bases -/
theorem parallel_segment_length (a b : ℝ) (t : Trapezoid a b) (s : ParallelSegment t) :
  DividesSimilarly t s → s = Real.sqrt (a * b) :=
sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l4152_415246


namespace NUMINAMATH_CALUDE_inequality_solution_quadratic_solution_l4152_415201

-- Part 1: Integer solutions of the inequality
def integer_solutions : Set ℤ :=
  {x : ℤ | -2 ≤ (1 + 2*x) / 3 ∧ (1 + 2*x) / 3 ≤ 2}

theorem inequality_solution :
  integer_solutions = {-3, -2, -1, 0, 1, 2} := by sorry

-- Part 2: Quadratic equation
def quadratic_equation (a b : ℚ) (x : ℚ) : ℚ :=
  a * x^2 + b * x

theorem quadratic_solution (a b : ℚ) :
  (quadratic_equation a b 1 = 0 ∧ quadratic_equation a b 2 = 3) →
  quadratic_equation a b (-2) = 9 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_quadratic_solution_l4152_415201


namespace NUMINAMATH_CALUDE_range_of_f_l4152_415204

/-- The diamond operation -/
def diamond (x y : ℝ) : ℝ := (x + y)^2 - x * y

/-- The function f -/
def f (a x : ℝ) : ℝ := diamond a x

theorem range_of_f (a : ℝ) (h : diamond 1 a = 3) :
  ∀ y : ℝ, y > 1 → ∃ x : ℝ, x > 0 ∧ f a x = y ∧
  ∀ z : ℝ, z > 0 → f a z ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l4152_415204


namespace NUMINAMATH_CALUDE_franks_remaining_money_l4152_415238

/-- 
Given:
- Frank initially had $600.
- He spent 1/5 of his money on groceries.
- He then spent 1/4 of the remaining money on a magazine.

Prove that Frank has $360 left after buying groceries and the magazine.
-/
theorem franks_remaining_money (initial_amount : ℚ) 
  (h1 : initial_amount = 600)
  (grocery_fraction : ℚ) (h2 : grocery_fraction = 1/5)
  (magazine_fraction : ℚ) (h3 : magazine_fraction = 1/4) :
  let remaining_after_groceries := initial_amount - grocery_fraction * initial_amount
  let remaining_after_magazine := remaining_after_groceries - magazine_fraction * remaining_after_groceries
  remaining_after_magazine = 360 := by
sorry

end NUMINAMATH_CALUDE_franks_remaining_money_l4152_415238


namespace NUMINAMATH_CALUDE_extremum_implies_b_value_l4152_415227

/-- A function f with a real parameter a -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem extremum_implies_b_value (a b : ℝ) :
  (f' a b 1 = 0) →  -- Derivative is zero at x = 1
  (f a b 1 = 10) →  -- Function value is 10 at x = 1
  b = -11 := by
sorry

end NUMINAMATH_CALUDE_extremum_implies_b_value_l4152_415227


namespace NUMINAMATH_CALUDE_hexagonangulo_19_requires_59_l4152_415262

/-- A hexagonângulo is a shape formed by triangles -/
structure Hexagonangulo where
  triangles : ℕ
  perimeter : ℕ

/-- Calculates the number of unit triangles needed to form a triangle of given side length -/
def trianglesInLargerTriangle (side : ℕ) : ℕ := side^2

/-- Constructs a hexagonângulo with given perimeter using unit triangles -/
def constructHexagonangulo (p : ℕ) : Hexagonangulo :=
  { triangles := 
      4 * trianglesInLargerTriangle 2 + 
      3 * trianglesInLargerTriangle 3 + 
      1 * trianglesInLargerTriangle 4,
    perimeter := p }

/-- Theorem: A hexagonângulo with perimeter 19 requires 59 unit triangles -/
theorem hexagonangulo_19_requires_59 : 
  (constructHexagonangulo 19).triangles = 59 := by sorry

end NUMINAMATH_CALUDE_hexagonangulo_19_requires_59_l4152_415262


namespace NUMINAMATH_CALUDE_largest_value_l4152_415298

def expr_a : ℝ := 3 - 1 + 4 + 6
def expr_b : ℝ := 3 - 1 * 4 + 6
def expr_c : ℝ := 3 - (1 + 4) * 6
def expr_d : ℝ := 3 - 1 + 4 * 6
def expr_e : ℝ := 3 * (1 - 4) + 6

theorem largest_value :
  expr_d = 26 ∧
  expr_d > expr_a ∧
  expr_d > expr_b ∧
  expr_d > expr_c ∧
  expr_d > expr_e :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l4152_415298


namespace NUMINAMATH_CALUDE_expired_bottle_probability_l4152_415264

theorem expired_bottle_probability (total_bottles : ℕ) (expired_bottles : ℕ) 
  (prob_both_unexpired : ℚ) :
  total_bottles = 30 →
  expired_bottles = 3 →
  prob_both_unexpired = 351 / 435 →
  (1 - prob_both_unexpired : ℚ) = 28 / 145 :=
by sorry

end NUMINAMATH_CALUDE_expired_bottle_probability_l4152_415264
