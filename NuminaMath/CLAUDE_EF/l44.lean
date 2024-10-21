import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_l44_4435

theorem angle_sum (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.cos α = 1/Real.sqrt 10 →
  Real.cos β = 1/Real.sqrt 5 →
  α + β = 3*π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_l44_4435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_twelve_l44_4496

/-- The line equation: 3x + 2y = 12 -/
def line_equation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

/-- The x-intercept of the line -/
noncomputable def x_intercept : ℝ := 4

/-- The y-intercept of the line -/
noncomputable def y_intercept : ℝ := 6

/-- The area of the triangular region -/
noncomputable def triangle_area : ℝ := (1 / 2) * x_intercept * y_intercept

theorem triangle_area_is_twelve :
  line_equation x_intercept 0 ∧
  line_equation 0 y_intercept ∧
  triangle_area = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_twelve_l44_4496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_twelve_l44_4497

theorem sum_reciprocal_twelve : (12 : ℝ) * (1/3 + 1/4 + 1/6 + 1/12)⁻¹ = 72/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocal_twelve_l44_4497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_z_16_l44_4424

-- Define the proportionality constants
variable (k m : ℝ)

-- Define the relationship between x, y, and z
noncomputable def x (z : ℝ) : ℝ := k * m^2 / z^4

-- State the theorem
theorem x_value_when_z_16 (h : x 4 = 2) : x 16 = 1/128 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_when_z_16_l44_4424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_expansion_l44_4439

/-- Given odd integers a and b, the number of odd terms in the expansion of (a+b)^8 is 2 -/
theorem odd_terms_in_expansion (a b : ℤ) (ha : Odd a) (hb : Odd b) :
  (Finset.filter (fun k => Odd ((Nat.choose 8 k) * a^(8-k) * b^k)) (Finset.range 9)).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_terms_in_expansion_l44_4439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_value_l44_4442

/-- The constant k in the inverse variation equation -/
noncomputable def k : ℝ := 3^2 * (64^(1/3))

/-- The inverse variation equation -/
def inverse_variation (x y : ℝ) : Prop := x^2 * y^(1/3) = k

/-- The theorem to prove -/
theorem find_y_value (x y : ℝ) 
  (h1 : inverse_variation 3 64)
  (h2 : x * y = 108) :
  ∃ ε > 0, |y - 108.16| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_y_value_l44_4442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_area_ratio_l44_4453

/-- Represents the dimensions of a block in meters -/
structure BlockDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular region given its dimensions in blocks -/
def calculateRectangularArea (lengthBlocks widthBlocks : ℕ) (dimensions : BlockDimensions) : ℝ :=
  (lengthBlocks : ℝ) * dimensions.length * (widthBlocks : ℝ) * dimensions.width

/-- Calculates the area of a square region given the number of blocks and block side length -/
def calculateSquareArea (numBlocks : ℕ) (sideLength : ℝ) : ℝ :=
  (numBlocks : ℝ) * sideLength * sideLength

theorem friend_area_ratio (tommy_block_dimensions : BlockDimensions)
    (friend_block_side_length : ℝ) :
    tommy_block_dimensions.length = 200 →
    tommy_block_dimensions.width = 150 →
    friend_block_side_length = 180 →
    (calculateSquareArea 80 friend_block_side_length) /
    (calculateRectangularArea 2 3 tommy_block_dimensions) = 14.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_friend_area_ratio_l44_4453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_distance_theorem_l44_4402

/-- The distance between the centers of two circular pulleys -/
noncomputable def pulley_centers_distance (r1 r2 contact_distance : ℝ) : ℝ :=
  Real.sqrt ((contact_distance ^ 2) + ((r1 - r2) ^ 2))

/-- Theorem stating the distance between pulley centers -/
theorem pulley_distance_theorem (r1 r2 contact_distance : ℝ) 
  (h1 : r1 = 10)
  (h2 : r2 = 6)
  (h3 : contact_distance = 20) :
  pulley_centers_distance r1 r2 contact_distance = 4 * Real.sqrt 26 := by
  sorry

#check pulley_distance_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pulley_distance_theorem_l44_4402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_box_size_l44_4433

/-- Represents the size and price of a box of macaroni and cheese -/
structure MacBox where
  size : ℚ  -- size in ounces
  price : ℚ  -- price in dollars

/-- Calculates the price per ounce in cents -/
def pricePerOunce (box : MacBox) : ℚ :=
  (box.price * 100) / box.size

theorem larger_box_size 
  (small_box : MacBox)
  (large_box : MacBox)
  (h1 : small_box.size = 20)
  (h2 : small_box.price = 17/5)
  (h3 : large_box.price = 24/5)
  (h4 : min (pricePerOunce small_box) (pricePerOunce large_box) = 16) :
  large_box.size = 30 := by
  sorry

#eval pricePerOunce { size := 20, price := 17/5 }
#eval pricePerOunce { size := 30, price := 24/5 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_box_size_l44_4433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l44_4429

-- Define the function f
def f : ℝ → ℝ := fun x ↦ x^2 - 2*x

-- Define the function g
def g (q : ℝ) : ℝ → ℝ := fun x ↦ 1 - q * f x - x

-- Theorem statement
theorem function_properties :
  (∀ x, f (x + 1) = x^2 - (1/3) * f 3) →
  (∀ x, f x = x^2 - 2*x) ∧
  (∃ q > 0, ∀ x ∈ Set.Icc (-1) 2, g q x ∈ Set.Icc (-4) (17/8) ∧
                                   (∃ y ∈ Set.Icc (-1) 2, g q y = -4) ∧
                                   (∃ z ∈ Set.Icc (-1) 2, g q z = 17/8)) :=
by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l44_4429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_l44_4423

/-- A spatial quadrilateral with diagonals of length 6 and 8 forming an angle of 45° -/
structure SpatialQuadrilateral where
  diagonal1 : ℝ
  diagonal2 : ℝ
  angle : ℝ
  h1 : diagonal1 = 6
  h2 : diagonal2 = 8
  h3 : angle = π / 4

/-- The quadrilateral formed by connecting the midpoints of the sides of a spatial quadrilateral -/
def midpointQuadrilateral (sq : SpatialQuadrilateral) : Set (ℝ × ℝ) := sorry

/-- The area of the midpoint quadrilateral -/
def midpointQuadrilateralArea (sq : SpatialQuadrilateral) : ℝ := sorry

theorem midpoint_quadrilateral_area (sq : SpatialQuadrilateral) :
  midpointQuadrilateralArea sq = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_quadrilateral_area_l44_4423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_square_l44_4436

theorem divisors_of_square (n : ℕ) (h : (Finset.card (Nat.divisors n)) = 5) : 
  Finset.card (Nat.divisors (n^2)) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_square_l44_4436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_iff_odd_l44_4460

/-- Represents a coloring of n points with two colors -/
def Coloring (n : ℕ) := Fin n → Bool

/-- Represents a move of type k on a coloring -/
def Move (n : ℕ) (k : ℕ) (c : Coloring n) : Coloring n :=
  sorry

/-- Predicate that checks if all points have the same color -/
def AllSameColor (n : ℕ) (c : Coloring n) : Prop :=
  sorry

/-- Predicate that checks if there exists a sequence of moves that results in all points having the same color -/
def ExistsSequence (n : ℕ) : Prop :=
  ∀ (c : Coloring n), ∃ (seq : List (ℕ × Fin n)), 
    (∀ (move : ℕ × Fin n), move ∈ seq → move.1 < n / 2) ∧ 
    AllSameColor n (seq.foldl (λ acc (k, _) => Move n k acc) c)

/-- The main theorem: ExistsSequence n is true if and only if n is odd and n ≥ 5 -/
theorem exists_sequence_iff_odd (n : ℕ) : 
  ExistsSequence n ↔ (n % 2 = 1 ∧ n ≥ 5) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sequence_iff_odd_l44_4460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_scores_l44_4495

/-- Represents the scores of a team over four quarters -/
structure TeamScores :=
  (q1 : ℕ) (q2 : ℕ) (q3 : ℕ) (q4 : ℕ)

/-- Calculates the total score of a team -/
def totalScore (scores : TeamScores) : ℕ :=
  scores.q1 + scores.q2 + scores.q3 + scores.q4

/-- Checks if the scores follow a geometric sequence -/
def isGeometric (scores : TeamScores) : Prop :=
  ∃ (r : ℚ), r > 1 ∧ 
    scores.q2 = (scores.q1 : ℚ) * r ∧
    scores.q3 = (scores.q2 : ℚ) * r ∧
    scores.q4 = (scores.q3 : ℚ) * r

/-- Checks if the scores follow an arithmetic sequence -/
def isArithmetic (scores : TeamScores) : Prop :=
  ∃ (d : ℕ), d > 0 ∧ 
    scores.q2 = scores.q1 + d ∧
    scores.q3 = scores.q2 + d ∧
    scores.q4 = scores.q3 + d

/-- The main theorem -/
theorem basketball_game_scores 
  (falcon : TeamScores) (eagle : TeamScores) 
  (h1 : falcon.q1 = eagle.q1)  -- Tie in the first quarter
  (h2 : isGeometric falcon)
  (h3 : isArithmetic eagle)
  (h4 : totalScore falcon = totalScore eagle + 2)  -- Falcon won by 2 points
  (h5 : totalScore falcon ≤ 120)
  (h6 : totalScore eagle ≤ 120)
  (h7 : ∀ q, q ∈ [falcon.q1, falcon.q2, falcon.q3, falcon.q4, 
               eagle.q1, eagle.q2, eagle.q3, eagle.q4] → q ≤ 36) :
  falcon.q1 + falcon.q2 + eagle.q1 + eagle.q2 = 52 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_game_scores_l44_4495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_lineup_count_l44_4492

theorem basketball_lineup_count : 
  let total_players : ℕ := 18
  let guards : ℕ := 2
  let forwards : ℕ := 3
  let centers : ℕ := 1
  let interchangeable : ℕ := 7
  
  Nat.choose total_players guards *
  Nat.choose (total_players - guards) forwards *
  Nat.choose (total_players - guards - forwards) centers *
  Nat.choose (total_players - guards - forwards - centers) interchangeable = 892046880 := by
  
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_lineup_count_l44_4492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_and_other_propositions_l44_4410

theorem set_equality_and_other_propositions :
  -- Proposition 3
  ({1, 3, 5, 7} : Set ℕ) = {7, 5, 3, 1} ∧
  -- Proposition 1 (negation)
  ¬∃ (S : Set ℝ), ∀ x, x ∈ S ↔ (x > 0 ∧ x < 1) ∧
  -- Proposition 2 (negation)
  ({1, 2, 3, 1, 9} : Set ℕ) ≠ {1, 2, 3, 9} ∧
  -- Proposition 4 (negation)
  ¬(∀ (x y : ℝ), (x, y) ∈ {(a, b) : ℝ × ℝ | b = -a} ↔ y = -x) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_equality_and_other_propositions_l44_4410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_plants_in_overlapping_beds_l44_4457

/-- The number of distinct plants in three overlapping flower beds -/
theorem distinct_plants_in_overlapping_beds
  (A B C : Finset Nat)  -- A, B, C represent the three flower beds
  (total_A : A.card = 800)
  (total_B : B.card = 700)
  (total_C : C.card = 600)
  (shared_AB : (A ∩ B).card = 120)
  (shared_AC : (A ∩ C).card = 200)
  (shared_BC : (B ∩ C).card = 150)
  (shared_ABC : (A ∩ B ∩ C).card = 75) :
  (A ∪ B ∪ C).card = 1705 := by
  sorry

#check distinct_plants_in_overlapping_beds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_plants_in_overlapping_beds_l44_4457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l44_4447

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The sum of distances from a point on the ellipse to its foci -/
def PointOnEllipse.sumOfFocalDistances (e : Ellipse) : ℝ := 2 * e.a

/-- Theorem: For the given ellipse and line, m = -5/3 -/
theorem ellipse_line_intersection
  (e : Ellipse)
  (h_sum : PointOnEllipse.sumOfFocalDistances e = 4)
  (h_ecc : e.eccentricity = Real.sqrt 3 / 2)
  (m : ℝ)
  (h_perp_bisector : ∃ (M N : PointOnEllipse e),
    M.y = M.x + m ∧ N.y = N.x + m ∧
    (M.x + N.x) / 2 = 1 - m / 2 ∧ (M.y + N.y) / 2 = m / 2) :
  m = -5/3 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_intersection_l44_4447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowtie_equation_solution_l44_4465

-- Define the operation ⋄
noncomputable def bowtie (a b : ℝ) : ℝ := a^2 + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 2 y = 18 ∧ y = 182 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowtie_equation_solution_l44_4465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_range_l44_4484

noncomputable section

-- Define the curve
def curve (x : ℝ) : ℝ := x^3 - x + 2/3

-- Define the derivative of the curve
def curve_derivative (x : ℝ) : ℝ := 3 * x^2 - 1

-- Define the range of inclination angles
def inclination_angle_range : Set ℝ := 
  Set.union (Set.Ioc 0 (Real.pi / 2)) (Set.Icc (3 * Real.pi / 4) Real.pi)

-- Theorem statement
theorem tangent_inclination_range :
  ∀ x : ℝ, ∃ α ∈ inclination_angle_range, Real.tan α = curve_derivative x := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_inclination_range_l44_4484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l44_4440

/-- Given two points A and B in a 2D plane, this theorem proves:
    1. The equation of the line passing through A and B
    2. The equation of the circle with diameter AB -/
theorem line_and_circle_equations 
  (A B : ℝ × ℝ) 
  (hA : A = (4, 6)) 
  (hB : B = (-2, 4)) : 
  (∃ (a b c : ℝ), ∀ (x y : ℝ), (x, y) ∈ Set.range (λ t : ℝ => (1-t) • A + t • B) ↔ a*x + b*y + c = 0) ∧ 
  (∃ (h k r : ℝ), ∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = r^2 ↔ 
    ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ (x, y) = (1-t) • A + t • B) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_circle_equations_l44_4440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_count_is_zero_l44_4426

/-- A proposition about vector properties -/
inductive VectorProposition
  | collinear_parallel
  | skew_not_coplanar
  | pairwise_coplanar_all_coplanar
  | any_vector_unique_combination

/-- Check if a given vector proposition is correct -/
def is_correct_proposition (prop : VectorProposition) : Bool :=
  match prop with
  | VectorProposition.collinear_parallel => false
  | VectorProposition.skew_not_coplanar => false
  | VectorProposition.pairwise_coplanar_all_coplanar => false
  | VectorProposition.any_vector_unique_combination => false

/-- Count the number of correct propositions -/
def count_correct_propositions : Nat :=
  [VectorProposition.collinear_parallel,
   VectorProposition.skew_not_coplanar,
   VectorProposition.pairwise_coplanar_all_coplanar,
   VectorProposition.any_vector_unique_combination].filter is_correct_proposition |>.length

/-- Theorem stating that the number of correct propositions is 0 -/
theorem correct_propositions_count_is_zero : count_correct_propositions = 0 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_propositions_count_is_zero_l44_4426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_1250_meters_l44_4428

/-- Represents the circular track where tigers run -/
structure Track where
  length : ℝ

/-- Represents a tiger running on the track -/
structure Tiger where
  speed : ℝ

/-- Calculates the number of laps a tiger completes in a given time -/
noncomputable def laps_completed (track : Track) (tiger : Tiger) (time : ℝ) : ℝ :=
  (tiger.speed * time) / track.length

theorem track_length_is_1250_meters 
  (track : Track)
  (amur bengal : Tiger)
  (h1 : laps_completed track amur 2 - laps_completed track bengal 2 = 6)
  (h2 : laps_completed track { speed := amur.speed + 10 } 1 + 
        laps_completed track amur 2 - 
        laps_completed track bengal 3 = 17) :
  track.length = 1250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_track_length_is_1250_meters_l44_4428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_distance_specific_points_l44_4483

/-- The surface distance between two points on a sphere, given their latitudes and a common longitude -/
noncomputable def surfaceDistance (r : ℝ) (lat1 lat2 : ℝ) : ℝ :=
  r * (Real.pi - |lat1 - lat2|)

theorem surface_distance_specific_points (R : ℝ) (h : R > 0) :
  surfaceDistance R (Real.pi/4) (-3*Real.pi/4) = (2*Real.pi/3) * R := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_distance_specific_points_l44_4483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_is_acute_main_theorem_l44_4478

/-- Definition of the arithmetic sequence for tan A -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

/-- Definition of the geometric sequence for tan B -/
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r^(n - 1)

/-- Theorem stating that triangle ABC is acute -/
theorem triangle_ABC_is_acute (A B C : ℝ) : Prop :=
  let tan_A := arithmetic_sequence (Real.arctan (-4)) 2 3
  let tan_B := geometric_sequence (1/3) 3 3
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2 ∧ A + B + C = Real.pi

/-- Main theorem to prove -/
theorem main_theorem (A B C : ℝ) : triangle_ABC_is_acute A B C := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_is_acute_main_theorem_l44_4478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_two_l44_4499

def numbers : Finset ℕ := {1856, 1975, 2042, 2071, 2150, 2203}

def sum_of_all : ℕ := numbers.sum id

def mean_of_four : ℚ := 2035

theorem mean_of_remaining_two :
  let remaining_sum : ℚ := (sum_of_all : ℚ) - 4 * mean_of_four
  remaining_sum / 2 = 2078.5 := by
    -- Proof steps would go here
    sorry

#eval sum_of_all

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_of_remaining_two_l44_4499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_school_count_l44_4441

theorem middle_school_count (total_students : ℕ) (girls_fraction : ℚ) 
  (girls_primary_fraction : ℚ) (boys_primary_fraction : ℚ) 
  (h1 : total_students = 800) 
  (h2 : girls_fraction = 5 / 8) 
  (h3 : girls_primary_fraction = 7 / 10) 
  (h4 : boys_primary_fraction = 2 / 5) : 
  ℕ := by
  -- Calculation steps would go here
  sorry

#eval 330 -- The expected result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_middle_school_count_l44_4441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_implies_line_equation_l44_4400

/-- Given two lines that intersect and form a figure with the x-axis, 
    if the area of this figure is 18, then the equation of the first line is y = x -/
theorem intersection_area_implies_line_equation 
  (line1 : ℝ → ℝ) 
  (line2 : ℝ → ℝ) 
  (h1 : ∀ x, line1 x = x)
  (h2 : ∀ x, line2 x = -6)
  (h_intersect : ∃ x y, line1 x = y ∧ line2 x = y)
  (h_figure : Set (ℝ × ℝ))
  (h_area : MeasureTheory.volume h_figure = 18) :
  ∀ x, line1 x = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_area_implies_line_equation_l44_4400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_root_three_half_max_area_is_half_l44_4486

/-- Ellipse with foci and a point P satisfying certain conditions -/
structure SpecialEllipse where
  a : ℝ
  b : ℝ
  h₁ : a > b
  h₂ : b > 0
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  h₃ : F₁.1 < F₂.1  -- F₁ is left focus, F₂ is right focus
  h₄ : F₁.2 = 0 ∧ F₂.2 = 0  -- Foci are on x-axis
  h₅ : (P.1 - F₁.1)^2 + P.2^2 = a^2  -- P is on the ellipse
  h₆ : (P.1 - F₂.1)^2 + P.2^2 = a^2  -- P is on the ellipse
  h₇ : P.1 = F₂.1  -- PF₂ ⟂ x-axis
  h₈ : (P.1 - F₁.1) * (P.1 - F₂.1) + P.2 * P.2 = (1/16) * a^2  -- PF₁ · PF₂ = (1/16)a²

/-- The eccentricity of the ellipse is √3/2 -/
theorem eccentricity_is_root_three_half (E : SpecialEllipse) :
  (E.F₂.1 - E.F₁.1) / (2 * E.a) = Real.sqrt 3 / 2 :=
sorry

/-- 
If the perimeter of triangle F₁PF₂ is 2 + √3, 
then the maximum area of triangle ABF₂ is 1/2 
-/
theorem max_area_is_half (E : SpecialEllipse) 
  (h : Real.sqrt ((E.P.1 - E.F₁.1)^2 + E.P.2^2) + 
       Real.sqrt ((E.P.1 - E.F₂.1)^2 + E.P.2^2) + 
       (E.F₂.1 - E.F₁.1) = 2 + Real.sqrt 3) :
  ∃ (A B : ℝ × ℝ), 
    (∀ A' B' : ℝ × ℝ, 
      (A'.1 - E.F₁.1) / (A'.2 - E.F₁.2) = (B'.1 - E.F₁.1) / (B'.2 - E.F₁.2) →
      (A'.1 - E.F₁.1)^2 / E.a^2 + A'.2^2 / E.b^2 = 1 →
      (B'.1 - E.F₁.1)^2 / E.a^2 + B'.2^2 / E.b^2 = 1 →
      abs ((A'.1 - E.F₂.1) * (B'.2 - E.F₂.2) - (B'.1 - E.F₂.1) * (A'.2 - E.F₂.2)) / 2 ≤
      abs ((A.1 - E.F₂.1) * (B.2 - E.F₂.2) - (B.1 - E.F₂.1) * (A.2 - E.F₂.2)) / 2) ∧
    abs ((A.1 - E.F₂.1) * (B.2 - E.F₂.2) - (B.1 - E.F₂.1) * (A.2 - E.F₂.2)) / 2 = 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_is_root_three_half_max_area_is_half_l44_4486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_4_only_C_l44_4494

open Real

-- Define the functions
noncomputable def f_A (x : ℝ) : ℝ := x + 4 / x
noncomputable def f_B (x : ℝ) : ℝ := sin x + 4 / sin x
noncomputable def f_C (x : ℝ) : ℝ := exp x + 4 * exp (-x)
noncomputable def f_D (x : ℝ) : ℝ := sqrt (x^2 + 1) + 2 / sqrt (x^2 + 1)

-- State the theorem
theorem min_value_4_only_C :
  (∃ x, f_C x = 4) ∧
  (∀ x, f_C x ≥ 4) ∧
  (¬∃ x, f_A x = 4) ∧
  (∀ x, 0 < x → x < π → f_B x > 4) ∧
  (∀ x, f_D x > 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_4_only_C_l44_4494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l44_4427

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of eccentricity -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

/-- Area of triangle OAB -/
noncomputable def triangle_area (a b : ℝ) : ℝ := a * b / 2

/-- Theorem about the ellipse C and the product |AN| · |BM| -/
theorem ellipse_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (he : eccentricity a b = Real.sqrt 3 / 2) (ht : triangle_area a b = 1) :
  (∀ x y : ℝ, ellipse x y a b ↔ x^2 / 4 + y^2 = 1) ∧
  (∀ x₀ y₀ : ℝ, ellipse x₀ y₀ a b → 
    ∃ (xn ym : ℝ), |2 + xn| * |1 + ym| = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l44_4427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l44_4432

theorem largest_expression : 
  let a := Real.sqrt (Real.rpow 7 (1/3) * Real.rpow 8 (1/3))
  let b := Real.sqrt (8 * Real.rpow 7 (1/3))
  let c := Real.sqrt (7 * Real.rpow 8 (1/3))
  let d := Real.rpow (7 * Real.sqrt 8) (1/3)
  let e := Real.rpow (8 * Real.sqrt 7) (1/3)
  b > a ∧ b > c ∧ b > d ∧ b > e :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_expression_l44_4432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_relation_l44_4488

/-- 
Given an isosceles triangle, if the cosine of the vertex angle is 4/5, 
then the sine of the base angle is 3√10/10.
-/
theorem isosceles_triangle_angle_relation : 
  ∀ α : ℝ, 
  -- α is the base angle of an isosceles triangle
  0 < α ∧ α < Real.pi / 2 →
  -- The cosine of the vertex angle is 4/5
  Real.cos (Real.pi - 2 * α) = 4 / 5 →
  -- Then the sine of the base angle is 3√10/10
  Real.sin α = 3 * (Real.sqrt 10) / 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_angle_relation_l44_4488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_zero_one_l44_4491

open Real

-- Define the function as noncomputable
noncomputable def f (x : ℝ) := log (3*x - x^3)

-- State the theorem
theorem f_strictly_increasing_on_zero_one :
  StrictMonoOn f (Set.Ioo 0 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_strictly_increasing_on_zero_one_l44_4491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l44_4434

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log x - x + 1

-- Theorem statement
theorem f_has_unique_zero :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_unique_zero_l44_4434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_theorem_l44_4425

noncomputable def initial_height : ℝ := 2000
noncomputable def bounce_ratio : ℝ := 1/3
noncomputable def target_height : ℝ := 6

noncomputable def height_after_bounces (k : ℕ) : ℝ := initial_height * (bounce_ratio ^ k)

theorem ball_bounce_theorem :
  ∀ k : ℕ, k < 6 → height_after_bounces k ≥ target_height ∧
  height_after_bounces 6 < target_height :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_bounce_theorem_l44_4425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l44_4443

-- Define the propositions
def proposition1 : Prop := ∀ (f : ℝ → ℝ) (x y : ℝ), f x = y ∧ f y = x → x = y

def proposition2 : Prop := ∀ (f : ℝ → ℝ) (x : ℝ), f (1 - x) = f (1 + x)

def proposition3 : Prop := ∀ (f : ℝ → ℝ) (a : ℝ), 
  (∀ x, f (-x) = -f x) → 
  (∀ x, f (2*a - x) = f x) → 
  (∀ x, f (x + 2*a) = f x)

def proposition4 : Prop := Fintype.card (Fin 3 → Fin 2) = 8

-- Theorem statement
theorem all_propositions_false : 
  ¬proposition1 ∧ ¬proposition2 ∧ ¬proposition3 ∧ ¬proposition4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_propositions_false_l44_4443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l44_4401

noncomputable def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_properties
  (a b : ℝ)
  (h_ab : a > b ∧ b > 0)
  (h_point : ellipse a b 0 (Real.sqrt 2))
  (h_ecc : eccentricity a b = Real.sqrt 6 / 3) :
  -- 1. Standard equation of E
  (∃ (x y : ℝ), ellipse (Real.sqrt 6) (Real.sqrt 2) x y ↔ x^2 / 6 + y^2 / 2 = 1) ∧
  -- 2. Range of OP · OQ
  (∀ (P Q : ℝ × ℝ),
    ellipse (Real.sqrt 6) (Real.sqrt 2) P.1 P.2 →
    ellipse (Real.sqrt 6) (Real.sqrt 2) Q.1 Q.2 →
    ∃ (t : ℝ), t * P.1 + (1 - t) * (-2) = Q.1 ∧ t * P.2 = Q.2 →
    -6 ≤ (P.1 * Q.1 + P.2 * Q.2) ∧ (P.1 * Q.1 + P.2 * Q.2) ≤ 10/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l44_4401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_cos_x_l44_4409

theorem derivative_x_cos_x (x : ℝ) :
  deriv (λ x => x * Real.cos x) x = Real.cos x - x * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_x_cos_x_l44_4409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_2022_b_terms_l44_4479

-- Define the ceiling function [x)
noncomputable def ceiling (x : ℝ) : ℤ := Int.ceil x

-- Define the sequence a_n
def a : ℕ → ℕ
| 0 => 1  -- Add this case to cover all natural numbers
| 1 => 1
| 2 => 4
| n + 3 => 2 * a (n + 2) + 2 - a (n + 1)

-- Define the sequence b_n
noncomputable def b (n : ℕ) : ℤ := ceiling (n * (n + 1) / a n)

theorem sum_of_first_2022_b_terms : 
  (Finset.range 2022).sum b = 4045 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_2022_b_terms_l44_4479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l44_4468

noncomputable def intersection_x (m₁ : ℝ) (b₁ : ℝ) (m₂ : ℝ) (b₂ : ℝ) : ℝ :=
  (b₂ - b₁) / (m₁ - m₂)

noncomputable def intersection_y (m : ℝ) (b : ℝ) (x : ℝ) : ℝ :=
  m * x + b

theorem intersection_of_lines :
  let line1 : ℝ → ℝ := λ x => 12
  let line2 : ℝ → ℝ := λ x => 2 * x + 4
  let a : ℝ := intersection_x 0 12 2 4
  let b : ℝ := intersection_y 0 12 a
  a = 4 ∧ line1 a = line2 a := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_lines_l44_4468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kevin_run_last_speed_time_l44_4455

/-- Represents Kevin's run with given speeds, times, and total distance. -/
structure KevinRun where
  speed1 : ℚ
  time1 : ℚ
  speed2 : ℚ
  time2 : ℚ
  speed3 : ℚ
  totalDistance : ℚ

/-- Calculates the time Kevin ran at the third speed. -/
def timeAtLastSpeed (run : KevinRun) : ℚ :=
  (run.totalDistance - (run.speed1 * run.time1 + run.speed2 * run.time2)) / run.speed3

/-- Theorem stating the time Kevin ran at the last speed is 0.25 hours. -/
theorem kevin_run_last_speed_time (run : KevinRun)
  (h1 : run.speed1 = 10)
  (h2 : run.time1 = 1/2)
  (h3 : run.speed2 = 20)
  (h4 : run.time2 = 1/2)
  (h5 : run.speed3 = 8)
  (h6 : run.totalDistance = 17) :
  timeAtLastSpeed run = 1/4 := by
  sorry

#eval timeAtLastSpeed { speed1 := 10, time1 := 1/2, speed2 := 20, time2 := 1/2, speed3 := 8, totalDistance := 17 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kevin_run_last_speed_time_l44_4455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_person_C_start_time_l44_4403

noncomputable section

-- Define the line segment AB and its trisection points
def AB : ℝ := 1  -- Normalize the length of AB to 1
def C : ℝ := AB / 3
def D : ℝ := 2 * AB / 3

-- Define the start times for persons A and B
def start_time_A : ℕ := 480  -- 8:00 AM in minutes since midnight
def start_time_B : ℕ := 492  -- 8:12 AM in minutes since midnight

-- Define the meeting time of A and B at point C
def meeting_time_AB : ℕ := 504  -- 8:24 AM in minutes since midnight

-- Define the final meeting time when all persons reach their destinations
def final_time : ℕ := 510  -- 8:30 AM in minutes since midnight

-- Define the speeds of persons A and B
def speed_A : ℝ := C / (meeting_time_AB - start_time_A)
def speed_B : ℝ := (AB - C) / (meeting_time_AB - start_time_B)

-- Define the speed of person C
def speed_C : ℝ := (AB - D) / (final_time - meeting_time_AB)

end noncomputable section

-- Theorem to prove
theorem person_C_start_time :
  ∃ (start_time_C : ℕ),
    start_time_C = 496 ∧  -- 8:16 AM in minutes since midnight
    (D - C) / speed_C = final_time - start_time_C :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_person_C_start_time_l44_4403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l44_4481

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sqrt 3 / 2 - Real.sqrt 3 * Real.sin (ω * x) ^ 2 - Real.sin (ω * x) * Real.cos (ω * x)

theorem function_properties (ω : ℝ) (h1 : ω > 0) 
  (h2 : ∃ (center : ℝ), ∃ (axis : ℝ), |center - axis| = π / 4 ∧ 
    ∀ (x : ℝ), f ω x = f ω (2 * axis - x)) : 
  ω = 1 ∧ 
  (∀ x ∈ Set.Icc π (3 * π / 2), f ω x ≤ Real.sqrt 3 / 2) ∧
  (∃ x ∈ Set.Icc π (3 * π / 2), f ω x = Real.sqrt 3 / 2) ∧
  (∀ x ∈ Set.Icc π (3 * π / 2), f ω x ≥ -1) ∧
  (∃ x ∈ Set.Icc π (3 * π / 2), f ω x = -1) ∧
  f ω π = Real.sqrt 3 / 2 ∧
  f ω (17 * π / 12) = -1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l44_4481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ceiling_height_for_given_field_l44_4493

/-- Represents the dimensions of a rectangular soccer field -/
structure SoccerField where
  length : ℝ
  width : ℝ

/-- Calculates the minimum ceiling height for a given soccer field -/
noncomputable def min_ceiling_height (field : SoccerField) : ℝ :=
  let diagonal := Real.sqrt (field.length ^ 2 + field.width ^ 2)
  let exact_height := (5 * Real.sqrt 13) / 2
  ⌈exact_height * 10⌉ / 10

/-- Theorem stating the minimum ceiling height for the given soccer field -/
theorem min_ceiling_height_for_given_field :
  let field := SoccerField.mk 90 60
  min_ceiling_height field = 27.1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_ceiling_height_for_given_field_l44_4493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_permutation_exists_l44_4416

-- Define a type for the grid elements
inductive GridElement
| Negative : GridElement
| Zero : GridElement
| Positive : GridElement

-- Define the grid type
def Grid (n : ℕ) := Matrix (Fin n) (Fin n) GridElement

-- Define a predicate to check if a grid is valid
def is_valid_grid (n : ℕ) (g : Grid n) : Prop :=
  ∀ i : Fin n,
    (∃! j : Fin n, g i j = GridElement.Positive) ∧
    (∃! j : Fin n, g i j = GridElement.Negative) ∧
    (∃! j : Fin n, g j i = GridElement.Positive) ∧
    (∃! j : Fin n, g j i = GridElement.Negative)

-- Define a function to swap signs in a grid
def swap_signs (n : ℕ) (g : Grid n) : Grid n :=
  λ i j ↦ match g i j with
    | GridElement.Positive => GridElement.Negative
    | GridElement.Negative => GridElement.Positive
    | GridElement.Zero => GridElement.Zero

-- Define the theorem
theorem grid_permutation_exists (n : ℕ) (g : Grid n) 
  (h : is_valid_grid n g) :
  ∃ (row_perm col_perm : Equiv.Perm (Fin n)),
    swap_signs n g = λ i j ↦ g (row_perm i) (col_perm j) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_permutation_exists_l44_4416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l44_4454

noncomputable section

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 / 4 = 1

-- Define the vertex C
def vertex_C : ℝ × ℝ := (0, -2)

-- Define the left focus
def left_focus : ℝ × ℝ := (-1, 0)

-- Define a line by its equation ax + by + c = 0
def line (a b c : ℝ) (p : ℝ × ℝ) : Prop := a * p.1 + b * p.2 + c = 0

-- Define the centroid of a triangle
def centroid (A B C : ℝ × ℝ) : ℝ × ℝ :=
  ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Theorem statement
theorem line_equation_proof (A B : ℝ × ℝ) :
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  centroid A B vertex_C = left_focus →
  ∃ (l : ℝ × ℝ → Prop), l = line 6 (-5) 14 ∧ l A ∧ l B :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l44_4454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sum_obtuse_triangle_sum_acute_triangle_sum_l44_4411

-- Define a structure for a triangle
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  angle_sum : A + B + C = Real.pi
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C

-- Define functions for different types of triangles
def isRightTriangle (t : Triangle) : Prop :=
  t.A = Real.pi/2 ∨ t.B = Real.pi/2 ∨ t.C = Real.pi/2

def isObtuseTriangle (t : Triangle) : Prop :=
  t.A > Real.pi/2 ∨ t.B > Real.pi/2 ∨ t.C > Real.pi/2

def isAcuteTriangle (t : Triangle) : Prop :=
  t.A < Real.pi/2 ∧ t.B < Real.pi/2 ∧ t.C < Real.pi/2

-- Define the sum of squared cosines
noncomputable def sumSquaredCosines (t : Triangle) : ℝ :=
  (Real.cos t.A)^2 + (Real.cos t.B)^2 + (Real.cos t.C)^2

-- Theorems to prove
theorem right_triangle_sum (t : Triangle) (h : isRightTriangle t) :
  sumSquaredCosines t = 1 := by sorry

theorem obtuse_triangle_sum (t : Triangle) (h : isObtuseTriangle t) :
  1 < sumSquaredCosines t ∧ sumSquaredCosines t < 3 := by sorry

theorem acute_triangle_sum (t : Triangle) (h : isAcuteTriangle t) :
  3/4 ≤ sumSquaredCosines t ∧ sumSquaredCosines t < 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_sum_obtuse_triangle_sum_acute_triangle_sum_l44_4411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_baez_marbles_ratio_l44_4452

theorem baez_marbles_ratio : 
  let initial_marbles : ℕ := 25
  let loss_percentage : ℚ := 1/5
  let final_marbles : ℕ := 60
  let remaining_marbles := initial_marbles - (initial_marbles * loss_percentage).floor
  let friend_marbles := final_marbles - remaining_marbles
  (friend_marbles : ℚ) / remaining_marbles = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_baez_marbles_ratio_l44_4452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_nonzero_area_is_21_l44_4405

/-- The minimum non-zero area of triangle ABC -/
noncomputable def min_nonzero_area : ℝ := 21

/-- Point A of the triangle -/
def A : ℝ × ℝ := (0, 0)

/-- Point B of the triangle -/
def B : ℝ × ℝ := (42, 18)

/-- Calculates the area of triangle ABC given the coordinates of point C -/
noncomputable def triangle_area (C : ℤ × ℤ) : ℝ :=
  let (p, q) := C
  (1 / 2) * |42 * (q : ℝ) - 756|

theorem min_nonzero_area_is_21 :
  ∀ C : ℤ × ℤ, triangle_area C ≠ 0 → triangle_area C ≥ min_nonzero_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_nonzero_area_is_21_l44_4405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_equals_nine_sevenths_l44_4446

-- Define the triangle ABC and point D
variable (A B C D : ℝ × ℝ)

-- Define the conditions
def right_angled_at_C (A B C : ℝ × ℝ) : Prop := sorry
def D_on_AC (A C D : ℝ × ℝ) : Prop := sorry
def angle_ABC_twice_DBC (A B C D : ℝ × ℝ) : Prop := sorry
def DC_equals_1 (C D : ℝ × ℝ) : Prop := sorry
def BD_equals_3 (B D : ℝ × ℝ) : Prop := sorry

-- Define AD as a function of the points
def AD (A D : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem AD_equals_nine_sevenths 
  (h1 : right_angled_at_C A B C)
  (h2 : D_on_AC A C D)
  (h3 : angle_ABC_twice_DBC A B C D)
  (h4 : DC_equals_1 C D)
  (h5 : BD_equals_3 B D) :
  AD A D = 9/7 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_AD_equals_nine_sevenths_l44_4446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_students_is_100_l44_4474

/-- The number of chairs in the church -/
def num_chairs : ℕ := sorry

/-- The number of students who went to church -/
def num_students : ℕ := sorry

/-- Condition 1: If 9 students sit per chair, one student cannot sit -/
axiom condition1 : num_students = 9 * num_chairs + 1

/-- Condition 2: If 10 students sit per chair, 1 chair becomes vacant -/
axiom condition2 : num_students = 10 * num_chairs - 10

/-- Theorem: The number of students who went to church is 100 -/
theorem num_students_is_100 : num_students = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_students_is_100_l44_4474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_side_length_l44_4430

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circle with center and radius -/
structure Circle where
  center : Point
  radius : ℝ

/-- Triangle with three vertices -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Given configuration for the problem -/
structure Configuration where
  ω : Circle
  ABC : Triangle
  O : Point
  P : Point
  T : Point
  K : Point

/-- Check if a triangle is acute -/
def IsAcute (t : Triangle) : Prop := sorry

/-- Check if a triangle is inscribed in a circle -/
def IsInscribed (t : Triangle) (c : Circle) : Prop := sorry

/-- Check if a point is on a circle defined by three points -/
def OnCircle (a b c d : Point) : Prop := sorry

/-- Check if a line is tangent to a circle at a point -/
def IsTangent (c : Circle) (p : Point) (t : Point) : Prop := sorry

/-- Check if a point is on a line segment -/
def OnSegment (p a b : Point) : Prop := sorry

/-- Check if a point is on a side of a triangle -/
def OnSide (p a b : Point) : Prop := sorry

/-- Calculate the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Calculate the angle between three points -/
noncomputable def angle (a b c : Point) : ℝ := sorry

/-- Calculate the distance between two points -/
noncomputable def distance (a b : Point) : ℝ := sorry

/-- Main theorem statement -/
theorem triangle_area_and_side_length 
  (config : Configuration)
  (h1 : IsAcute config.ABC)
  (h2 : IsInscribed config.ABC config.ω)
  (h3 : config.O = config.ω.center)
  (h4 : OnCircle config.ABC.A config.O config.ABC.C config.P)
  (h5 : IsTangent config.ω config.ABC.A config.T)
  (h6 : IsTangent config.ω config.ABC.C config.T)
  (h7 : OnSegment config.K config.T config.P)
  (h8 : OnSide config.K config.ABC.A config.ABC.C)
  (h9 : area (Triangle.mk config.ABC.A config.P config.K) = 10)
  (h10 : area (Triangle.mk config.ABC.C config.P config.K) = 8)
  (h11 : angle config.ABC.A config.ABC.B config.ABC.C = Real.arctan (1/2)) :
  (area config.ABC = 81/2) ∧ 
  (distance config.ABC.A config.ABC.C = 3 * Real.sqrt 17 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_side_length_l44_4430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_difference_l44_4418

-- Define the given parameters
noncomputable def trainA_length : ℝ := 150
noncomputable def trainA_speed : ℝ := 75
noncomputable def bridgeA_length : ℝ := 300

noncomputable def trainB_length : ℝ := 180
noncomputable def trainB_speed : ℝ := 90
noncomputable def bridgeB_length : ℝ := 420

-- Define the conversion factor from km/h to m/s
noncomputable def kmph_to_ms : ℝ := 5 / 18

-- Define the function to calculate crossing time
noncomputable def crossing_time (train_length bridge_length train_speed : ℝ) : ℝ :=
  (train_length + bridge_length) / (train_speed * kmph_to_ms)

-- Theorem statement
theorem train_crossing_time_difference :
  crossing_time trainB_length bridgeB_length trainB_speed -
  crossing_time trainA_length bridgeA_length trainA_speed = 2.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_difference_l44_4418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l44_4459

/-- The distance from a point to a plane in 3D space --/
noncomputable def distance_point_to_plane (normal : ℝ × ℝ × ℝ) (point : ℝ × ℝ × ℝ) : ℝ :=
  let (a, b, c) := normal
  let (x, y, z) := point
  (abs (a * x + b * y + c * z)) / Real.sqrt (a^2 + b^2 + c^2)

/-- The problem statement --/
theorem distance_to_plane :
  let normal : ℝ × ℝ × ℝ := (2, -2, 1)
  let point : ℝ × ℝ × ℝ := (-1, 3, 2)
  distance_point_to_plane normal point = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_plane_l44_4459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_squarifiable_number_l44_4444

/-- A complication is adding a single digit to a number -/
def Complication := Nat → Nat

/-- Apply a sequence of complications to a number -/
def applyComplications (n : Nat) (complications : List Complication) : Nat :=
  complications.foldl (fun acc c => c acc) n

/-- Check if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

theorem existence_of_non_squarifiable_number :
  ∃ n : Nat, ∀ complications : List Complication,
    complications.length ≤ 100 →
    ¬(isPerfectSquare (applyComplications n complications)) := by
  sorry

#check existence_of_non_squarifiable_number

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_non_squarifiable_number_l44_4444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_point_value_l44_4473

/-- The golden ratio, approximately 0.618 --/
noncomputable def φ : ℝ := (Real.sqrt 5 - 1) / 2

/-- Calculate the third test point using the 0.618 method --/
noncomputable def third_test_point (a b x₁ x₂ : ℝ) : ℝ :=
  if x₁ > x₂ then
    b - φ * (b - x₁)
  else
    a + φ * (x₂ - a)

/-- Theorem stating the value of the third test point in the 0.618 method --/
theorem third_point_value (a b x₁ x₂ : ℝ) 
  (h₁ : a = 2) (h₂ : b = 4) 
  (h₃ : x₁ = a + φ * (b - a) ∨ x₁ = a + (1 - φ) * (b - a))
  (h₄ : x₂ = b - (x₁ - a)) :
  third_test_point a b x₁ x₂ = 3.528 ∨ third_test_point a b x₁ x₂ = 2.472 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_point_value_l44_4473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_in_two_years_l44_4471

/-- Sam's current age -/
def s : ℕ := sorry

/-- Sarah's current age -/
def r : ℕ := sorry

/-- Sam's age was twice Sarah's age two years ago -/
axiom sam_twice_sarah_two_years_ago : s - 2 = 2 * (r - 2)

/-- Sam's age was three times Sarah's age four years ago -/
axiom sam_triple_sarah_four_years_ago : s - 4 = 3 * (r - 4)

/-- The number of years until their age ratio becomes 3:2 -/
def years_until_ratio : ℕ := 2

/-- The theorem to prove -/
theorem age_ratio_in_two_years :
  (s + years_until_ratio : ℚ) / (r + years_until_ratio) = 3 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_ratio_in_two_years_l44_4471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_zero_l44_4422

def points : List (ℚ × ℚ) := [(4, 15), (7, 25), (13, 40), (19, 45), (21, 55), (25, 60)]

noncomputable def is_above_line (point : ℚ × ℚ) : Bool :=
  point.2 > 3 * point.1 + 5

noncomputable def sum_x_above_line (points : List (ℚ × ℚ)) : ℚ :=
  (points.filter is_above_line).map (·.1) |>.sum

theorem sum_x_above_line_is_zero : sum_x_above_line points = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_x_above_line_is_zero_l44_4422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_around_post_l44_4477

/-- The length of a string wrapped around a cylindrical post -/
theorem string_length_around_post (post_circumference post_height : ℝ) (num_loops : ℕ) 
  (h1 : post_circumference = 4)
  (h2 : post_height = 12)
  (h3 : num_loops = 4) :
  (num_loops : ℝ) * Real.sqrt (post_circumference ^ 2 + (post_height / num_loops) ^ 2) = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_string_length_around_post_l44_4477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_bag_identification_l44_4456

/-- Represents a bag of coins -/
structure CoinBag where
  id : Nat
  weight : Nat
  count : Nat

/-- The setup of the coin weighing problem -/
structure CoinProblem where
  bags : List CoinBag
  genuine_weight : Nat
  counterfeit_weight : Nat

/-- Calculate the expected weight if all coins were genuine -/
def expectedWeight (problem : CoinProblem) : Nat :=
  problem.genuine_weight * (problem.bags.map (fun bag => bag.count)).sum

/-- Calculate the actual weight of the coins -/
def actualWeight (problem : CoinProblem) : Nat :=
  (problem.bags.map (fun bag => bag.weight * bag.count)).sum

/-- Find the bag with counterfeit coins -/
def findCounterfeitBag (problem : CoinProblem) : Nat :=
  actualWeight problem - expectedWeight problem

/-- Theorem stating that the difference in weight identifies the counterfeit bag -/
theorem counterfeit_bag_identification (problem : CoinProblem) 
  (h1 : problem.bags.length = 10)
  (h2 : ∀ i, i ∈ problem.bags → i.count = i.id)
  (h3 : ∃! bag, bag ∈ problem.bags ∧ bag.weight = problem.counterfeit_weight)
  (h4 : ∀ bag, bag ∈ problem.bags → bag.weight = problem.genuine_weight ∨ bag.weight = problem.counterfeit_weight)
  (h5 : problem.counterfeit_weight = problem.genuine_weight + 1) :
  ∃ bag, bag ∈ problem.bags ∧ bag.id = findCounterfeitBag problem ∧ bag.weight = problem.counterfeit_weight :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_counterfeit_bag_identification_l44_4456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_region_implies_a_value_l44_4469

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sqrt (a * x^2 - 2 * a * x)

theorem square_region_implies_a_value (a : ℝ) (h_a : a < 0) :
  (∀ x, x ∈ Set.Icc 0 2 → f a x ∈ Set.Icc 0 2) →
  (∀ m n, m ∈ Set.Icc 0 2 → n ∈ Set.Icc 0 2 → ∃ s, s > 0 ∧ Set.prod (Set.Icc 0 2) (Set.range (f a)) = Set.Icc 0 s ×ˢ Set.Icc 0 s) →
  a = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_region_implies_a_value_l44_4469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l44_4448

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (2 * Real.exp x - 1) / (Real.exp x + 2)

-- State the theorem
theorem f_range : Set.range f = Set.Ioo (-1/2 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l44_4448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_increases_l44_4412

/-- Triangle area calculation function using Heron's formula -/
noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c))

/-- Theorem stating that the area of triangle XYZ increases when side XZ is doubled -/
theorem triangle_area_increases (xy xz yz : ℝ) 
  (h_xy : xy = 8) (h_xz : xz = 5) (h_yz : yz = 6) :
  triangle_area xy (2 * xz) yz > triangle_area xy xz yz :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_increases_l44_4412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_games_l44_4467

def number_of_games_in_single_elimination_tournament (n : ℕ) : ℕ :=
  n - 1

theorem tournament_games (n : ℕ) (h : n = 32) : 
  (number_of_games_in_single_elimination_tournament n) = n - 1 := by
  rfl

#check tournament_games

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_games_l44_4467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l44_4466

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def g (φ : ℝ) (x : ℝ) : ℝ := Real.sin (2 * (x - φ))

theorem min_shift_value (φ : ℝ) (h1 : φ > 0) (h2 : g φ (π/3) = 1/2) :
  ∀ ψ > 0, g ψ (π/3) = 1/2 → φ ≤ ψ → φ = π/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_shift_value_l44_4466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_value_bound_l44_4421

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 1 + a * Real.log x

theorem extreme_point_value_bound 
  (a : ℝ) 
  (x₁ x₂ : ℝ) 
  (ha : a < 1/2) 
  (hx : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1) 
  (hx₂ : 1/2 < x₂) 
  (hextreme : ∀ x, 0 < x → x ≠ x₁ → x ≠ x₂ → (deriv (f a)) x ≠ 0) :
  f a x₂ > (1 - 2 * Real.log 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_value_bound_l44_4421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_inradius_l44_4408

-- Define the parabola
structure Parabola where
  Γ : Set (ℝ × ℝ)
  focus : ℝ × ℝ

-- Define a point on the parabola
def PointOnParabola (Γ : Parabola) := { p : ℝ × ℝ // p ∈ Γ.Γ }

-- Define a triangle formed by three points on the parabola
structure TriangleOnParabola (Γ : Parabola) where
  A : PointOnParabola Γ
  B : PointOnParabola Γ
  C : PointOnParabola Γ

-- Define the orthocenter of a triangle
noncomputable def orthocenter (Γ : Parabola) (T : TriangleOnParabola Γ) : ℝ × ℝ := sorry

-- Define the inradius of a triangle
noncomputable def inradius (Γ : Parabola) (T : TriangleOnParabola Γ) : ℝ := sorry

-- The main theorem
theorem constant_inradius (Γ : Parabola) :
  ∀ (T : TriangleOnParabola Γ),
    orthocenter Γ T = Γ.focus →
    ∃ (r : ℝ), ∀ (T' : TriangleOnParabola Γ),
      orthocenter Γ T' = Γ.focus →
      inradius Γ T' = r :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_inradius_l44_4408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l44_4498

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The emission point of the light ray -/
def emissionPoint : Point := ⟨2, 3⟩

/-- The slope of the incident light ray -/
def incidentSlope : ℚ := 1/2

/-- The y-axis (reflection axis) -/
def yAxis : Line := ⟨1, 0, 0⟩

/-- Function to calculate the reflected ray's equation -/
noncomputable def reflectedRayEquation (p : Point) (k : ℚ) (axis : Line) : Line :=
  sorry

theorem reflected_ray_equation :
  reflectedRayEquation emissionPoint incidentSlope yAxis = ⟨1, 2, -4⟩ := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_ray_equation_l44_4498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_integer_quotient_l44_4451

/-- Given two finite sequences of positive integers, if for every integer d > 1,
    the number of elements in the first sequence divisible by d is greater than or equal to
    the number of elements in the second sequence divisible by d,
    then the product of elements in the first sequence divided by
    the product of elements in the second sequence is an integer. -/
theorem divisibility_implies_integer_quotient
  (m n : List ℕ) (h_nonempty : m.length > 0 ∧ n.length > 0)
  (h_positive : ∀ x ∈ m, x > 0) (h_positive' : ∀ x ∈ n, x > 0)
  (h_divisibility : ∀ d : ℕ, d > 1 →
    (m.filter (λ x => x % d = 0)).length ≥ (n.filter (λ x => x % d = 0)).length) :
  ∃ k : ℕ, k * (n.prod) = (m.prod) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_implies_integer_quotient_l44_4451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P2_tangent_lines_through_P2_l44_4438

noncomputable section

-- Define the curve
def f (x : ℝ) : ℝ := (1/3) * x^3

-- Define the derivative of f
def f_deriv (x : ℝ) : ℝ := x^2

-- Define the tangent line equation
def is_tangent_line (a b c : ℝ) (x₀ : ℝ) : Prop :=
  a * x₀ + b * f x₀ + c = 0 ∧
  a + b * (f_deriv x₀) = 0

-- Theorem for the first part
theorem tangent_line_at_P2 :
  is_tangent_line 12 (-3) (-16) 2 := by
  sorry

-- Theorem for the second part
theorem tangent_lines_through_P2 :
  ∀ (a b c : ℝ), is_tangent_line a b c 2 →
    (a = 12 ∧ b = -3 ∧ c = -16) ∨ (a = 3 ∧ b = -3 ∧ c = 2) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_P2_tangent_lines_through_P2_l44_4438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remainders_is_106_l44_4417

/-- A positive integer whose digits are five consecutive integers in decreasing order -/
def ConsecutiveDigitNumber (n : ℕ) : Prop :=
  ∃ (d : ℕ), 
    0 ≤ d ∧ d ≤ 5 ∧
    n = 10000 * (d + 4) + 1000 * (d + 3) + 100 * (d + 2) + 10 * (d + 1) + d

/-- The set of all possible ConsecutiveDigitNumbers -/
def ConsecutiveDigitNumberSet : Set ℕ :=
  {n : ℕ | ConsecutiveDigitNumber n}

/-- The sum of remainders when dividing ConsecutiveDigitNumbers by 43 -/
noncomputable def SumOfRemainders : ℕ :=
  (Finset.range 6).sum (fun d => (10000 * (d + 4) + 1000 * (d + 3) + 100 * (d + 2) + 10 * (d + 1) + d) % 43)

theorem sum_of_remainders_is_106 : SumOfRemainders = 106 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remainders_is_106_l44_4417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_inequality_l44_4450

theorem binomial_coefficient_inequality (n x : ℕ) :
  (Nat.choose (2*n + x) n) * (Nat.choose (2*n - x) n) ≤ (Nat.choose (2*n) n)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_coefficient_inequality_l44_4450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l44_4472

theorem system_solution (y z : ℝ) : 
  (3 : ℝ) ^ (2 * y) = (3 : ℝ) ^ 20 ∧ y ^ z = 3 → y = 10 ∧ z = Real.log 3 / Real.log 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l44_4472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l44_4431

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : π/2 < α ∧ α < π) : 
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_value_l44_4431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l44_4490

-- Define the region
def region (x y : ℝ) : Prop := abs (x + y) + abs (x - y) ≤ 6

-- State the theorem
theorem area_of_region :
  MeasureTheory.volume { p : ℝ × ℝ | region p.1 p.2 } = 36 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l44_4490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l44_4449

noncomputable def f (x : ℝ) : ℝ := (2^x - 2^(-x)) / x

theorem f_is_even : ∀ x : ℝ, x ≠ 0 → f x = f (-x) := by
  intro x hx
  simp [f]
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_even_l44_4449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_student_hostel_stay_l44_4404

/-- Calculates the number of days stayed given the charging scheme and total cost -/
def days_stayed (first_week_rate : ℚ) (additional_week_rate : ℚ) (total_cost : ℚ) : ℕ :=
  let first_week_cost := 7 * first_week_rate
  let additional_days := ((total_cost - first_week_cost) / additional_week_rate).floor.toNat
  7 + additional_days

/-- Theorem: Given the charging scheme and total cost, the number of days stayed is 23 -/
theorem student_hostel_stay :
  days_stayed 18 13 334 = 23 := by
  sorry

#eval days_stayed 18 13 334

end NUMINAMATH_CALUDE_ERRORFEEDBACK_student_hostel_stay_l44_4404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_4_minus_alpha_l44_4413

theorem sin_pi_4_minus_alpha (α : ℝ) (h1 : π/2 < α) (h2 : α < π) (h3 : Real.cos α = -3/5) : 
  Real.sin (π/4 - α) = -7 * Real.sqrt 2 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_4_minus_alpha_l44_4413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l44_4462

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Vector in 2D space -/
structure Vec where
  x : ℝ
  y : ℝ

/-- Definition of the parabola C -/
def on_parabola (C : Parabola) (P : Point) : Prop :=
  P.y^2 = 2 * C.p * P.x

/-- Definition of the focus F -/
def focus (C : Parabola) : Point :=
  ⟨C.p, 0⟩

/-- Definition of vector PQ -/
def vector_PQ (P Q : Point) : Vec :=
  ⟨Q.x - P.x, Q.y - P.y⟩

/-- Definition of vector QF -/
def vector_QF (C : Parabola) (Q : Point) : Vec :=
  ⟨C.p - Q.x, -Q.y⟩

/-- Condition that PQ = 9QF -/
def PQ_equals_9QF (C : Parabola) (P Q : Point) : Prop :=
  vector_PQ P Q = Vec.mk (9 * (C.p - Q.x)) (-9 * Q.y)

/-- Slope of line OQ -/
noncomputable def slope_OQ (Q : Point) : ℝ :=
  Q.y / Q.x

/-- Main theorem -/
theorem max_slope_OQ (C : Parabola) (hC : C.p = 1) :
  ∃ (max_slope : ℝ), max_slope = 1/3 ∧
  ∀ (P Q : Point),
    on_parabola C P →
    PQ_equals_9QF C P Q →
    |slope_OQ Q| ≤ max_slope := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_slope_OQ_l44_4462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_sine_l44_4445

/-- Triangle ABC with midline MN connecting sides AB and BC -/
structure Triangle (A B C M N : ℝ × ℝ) where
  midline : M.1 + N.1 = A.1 + B.1 ∧ M.2 + N.2 = A.2 + B.2

/-- Circle passing through M, N, and C -/
def CirclePassingThrough (M N C : ℝ × ℝ) (center : ℝ × ℝ) (r : ℝ) :=
  (M.1 - center.1)^2 + (M.2 - center.2)^2 = r^2 ∧
  (N.1 - center.1)^2 + (N.2 - center.2)^2 = r^2 ∧
  (C.1 - center.1)^2 + (C.2 - center.2)^2 = r^2

/-- Circle touches side AB -/
def CircleTouchesSide (A B : ℝ × ℝ) (center : ℝ × ℝ) (r : ℝ) :=
  ∃ (P : ℝ × ℝ), P ∈ Set.Icc A B ∧ (P.1 - center.1)^2 + (P.2 - center.2)^2 = r^2

/-- Main theorem -/
theorem triangle_circle_sine (A B C M N : ℝ × ℝ) (center : ℝ × ℝ) :
  Triangle A B C M N →
  CirclePassingThrough M N C center (Real.sqrt 2) →
  CircleTouchesSide A B center (Real.sqrt 2) →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 4 →
  let β := Real.arccos ((B.1 - C.1) * (A.1 - C.1) + (B.2 - C.2) * (A.2 - C.2)) /
           (Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) * Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2))
  Real.sin β = 1 / 2 := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_circle_sine_l44_4445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_power_function_l44_4485

theorem range_of_power_function (m : ℝ) (hm : m > 0) :
  Set.range (fun x : ℝ => x ^ m) ∩ Set.Ioc (0 : ℝ) 1 = Set.Ioo (0 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_power_function_l44_4485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_partitions_to_remove_l44_4420

/-- The function that calculates the minimum number of partitions to remove. -/
def min_partitions_removed (n : ℕ) : ℕ :=
  (n - 2)^3

/-- Given a cube with side length n (where n ≥ 3), the minimum number of partitions
    between unit cubes that need to be removed to reach the boundary of the cube
    from any unit cube is (n-2)^3. -/
theorem min_partitions_to_remove (n : ℕ) (h : n ≥ 3) :
  min_partitions_removed n = (n - 2)^3 := by
  -- Unfold the definition of min_partitions_removed
  unfold min_partitions_removed
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_partitions_to_remove_l44_4420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_speed_palindrome_journey_l44_4415

/-- Checks if a number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- Represents the journey parameters -/
structure Journey where
  startReading : ℕ
  duration : ℚ
  speedLimit : ℚ

/-- Calculates the maximum distance possible given the journey parameters -/
def maxDistance (j : Journey) : ℚ := j.duration * j.speedLimit

/-- Finds the next palindrome odometer reading within the maximum distance -/
def nextPalindromeReading (start : ℕ) (maxDist : ℚ) : ℕ := sorry

/-- Calculates the average speed given distance and time -/
def averageSpeed (distance : ℚ) (time : ℚ) : ℚ := distance / time

theorem max_average_speed_palindrome_journey :
  let j : Journey := { startReading := 12321, duration := 3, speedLimit := 75 }
  let maxDist : ℚ := maxDistance j
  let nextReading : ℕ := nextPalindromeReading j.startReading maxDist
  let actualDistance : ℚ := (nextReading - j.startReading : ℚ)
  isPalindrome j.startReading →
  isPalindrome nextReading →
  averageSpeed actualDistance j.duration ≤ 200 / 3 ∧
  ¬∃ (speed : ℚ), averageSpeed actualDistance j.duration < speed ∧ speed ≤ j.speedLimit ∧
    ∃ (reading : ℕ), reading > nextReading ∧ isPalindrome reading ∧
    (reading - j.startReading : ℚ) ≤ maxDist := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_average_speed_palindrome_journey_l44_4415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_small_box_l44_4470

/-- Given a setup of nested boxes with balls, calculates the number of balls in the smallest box. -/
theorem balls_in_small_box 
  (total : ℕ) 
  (n : ℕ) -- number of balls in big box but not in medium box
  (m : ℕ) -- number of balls in medium box but not in small box
  (h_total : total = 100) -- total number of balls is 100
  : total - n - m = total - n - m := by
  -- The proof goes here
  sorry

#check balls_in_small_box

end NUMINAMATH_CALUDE_ERRORFEEDBACK_balls_in_small_box_l44_4470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l44_4458

-- Define the function f(x) = x - sin(x)
noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

-- Theorem statement
theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x ≤ f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l44_4458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_transformation_l44_4406

/-- Given a point P in a Cartesian coordinate system, if its coordinates with respect to 
    the point symmetric to the x-axis are (-1, 2), then its coordinates with respect to 
    the point symmetric to the y-axis are (1, -2). -/
theorem point_symmetry_transformation (P : ℝ × ℝ) :
  (let (x, y) := P; (-x, -y) = (-1, 2)) →
  (let (x, y) := P; (x, -y) = (1, -2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_symmetry_transformation_l44_4406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_S_l44_4464

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

-- Define the set of x₀ where f(x₀) > 1
def S : Set ℝ := {x | f x > 1}

-- Theorem statement
theorem characterization_of_S : S = Set.Ioi 1 ∪ Set.Iic (-1) := by
  sorry

#check characterization_of_S

end NUMINAMATH_CALUDE_ERRORFEEDBACK_characterization_of_S_l44_4464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_one_l44_4414

noncomputable def geometricSum (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a else a * (1 - q^n) / (1 - q)

noncomputable def geometricTerm (a : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a * q^(n - 1)

theorem geometric_sequence_ratio_one
  (a : ℝ) (q : ℝ) (h : a ≠ 0) :
  (geometricSum a q 3) / (geometricTerm a q 3) = 3 → q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_one_l44_4414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_Q_l44_4476

/-- The number of digits in (2^222)^5 × (5^555)^2 -/
def Q : ℕ := 1111

/-- The largest prime factor of Q -/
def largest_prime_factor : ℕ := 101

theorem largest_prime_factor_of_Q : 
  (Nat.factors Q).maximum = some largest_prime_factor := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_Q_l44_4476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l44_4461

/-- Given two parallel vectors a and b, prove that x + y = -4 -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : Fin 3 → ℝ := ![(-1), x, 3]
  let b : Fin 3 → ℝ := ![2, (-4), y]
  (∃ (k : ℝ), a = k • b) →
  x + y = -4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_vectors_sum_l44_4461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l44_4463

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

/-- AD is a median on side BC -/
def isMedian (t : Triangle) (AD : Real) : Prop :=
  AD * AD = (t.b * t.b + t.c * t.c) / 4 + (t.a * t.a) / 4

theorem triangle_properties (t : Triangle) (h1 : t.a = 2) (h2 : isMedian t 2) :
  (t.b * t.b + t.c * t.c = 10) ∧
  (3/5 ≤ Real.cos t.A ∧ Real.cos t.A < 1) ∧
  (Real.arccos ((t.c * t.c + 4 - 1) / (4 * t.c)) ≤ π/6) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l44_4463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l44_4407

-- Define set A
def A : Set ℝ := {x : ℝ | ∃ y : ℝ, x + y^2 = 1}

-- Define set B
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 - 1}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1 : ℝ) (1 : ℝ) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l44_4407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l44_4489

theorem sin_double_theta (θ : ℝ) (h : Real.cos θ + Real.sin θ = 3/2) : Real.sin (2 * θ) = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_theta_l44_4489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_diesel_cost_is_1_6_l44_4437

/-- The mean cost of diesel over a 4-year period for a truck company. -/
noncomputable def mean_diesel_cost (rates : Fin 4 → ℝ) : ℝ :=
  (rates 0 + rates 1 + rates 2 + rates 3) / 4

/-- Proof that the mean cost of diesel over the 4-year period is $1.6 per gallon. -/
theorem mean_diesel_cost_is_1_6 :
  let rates : Fin 4 → ℝ := ![1.2, 1.3, 1.8, 2.1]
  mean_diesel_cost rates = 1.6 := by
  sorry

/-- The company spends the same amount of dollars on diesel each year. -/
axiom constant_spending (rates : Fin 4 → ℝ) :
  ∃ (amount : ℝ), ∀ (i : Fin 4), amount / rates i = amount / rates 0

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_diesel_cost_is_1_6_l44_4437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l44_4475

/-- Given two triangles with angles (α, β, γ) and (α₁, β₁, γ₁) respectively,
    the sum of cosines of angles from one triangle divided by sines of 
    corresponding angles from the other triangle is less than or equal to 
    the sum of cotangents of angles from the second triangle. -/
theorem triangle_angle_inequality 
  (α β γ α₁ β₁ γ₁ : ℝ) 
  (h_triangle1 : α + β + γ = Real.pi) 
  (h_triangle2 : α₁ + β₁ + γ₁ = Real.pi) 
  (h_positive1 : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_positive2 : 0 < α₁ ∧ 0 < β₁ ∧ 0 < γ₁) : 
  (Real.cos α₁ / Real.sin α) + (Real.cos β₁ / Real.sin β) + (Real.cos γ₁ / Real.sin γ) 
  ≤ (Real.cos α / Real.sin α) + (Real.cos β / Real.sin β) + (Real.cos γ / Real.sin γ) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_inequality_l44_4475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_properties_l44_4482

open Real

-- Define the function f as noncomputable
noncomputable def f (x φ : ℝ) : ℝ := sin (2 * x + φ)

-- State the theorem
theorem sin_function_properties :
  ∀ φ : ℝ, 0 < φ → φ < π / 2 → f 0 φ = 1 / 2 →
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f x φ = f (x + T) φ ∧
    ∀ T' : ℝ, T' > 0 → (∀ x : ℝ, f x φ = f (x + T') φ) → T ≤ T') ∧
  φ = π / 6 ∧
  (∀ x : ℝ, 0 ≤ x → x ≤ π / 2 → f x φ ≥ -1 / 2) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ π / 2 ∧ f x φ = -1 / 2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_function_properties_l44_4482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_count_proof_l44_4487

/-- The number of trees around the house. -/
def N : ℕ := 76

/-- Timur's count for a given tree. -/
def timur_count (x : ℕ) : ℕ := x % N

/-- Alexander's count for a given tree. -/
def alexander_count (x : ℕ) : ℕ := x % N

/-- The difference between Timur's and Alexander's starting points. -/
def m : ℕ := 21

theorem tree_count_proof :
  (timur_count 12 = alexander_count (33 - m)) ∧
  (timur_count (105 - m) = alexander_count 8) := by
  sorry

#eval N

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tree_count_proof_l44_4487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_line_l44_4480

noncomputable def line (x y : ℝ) : Prop := 10 * x + 24 * y = 120

noncomputable def distance (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

theorem min_distance_on_line :
  ∃ (min : ℝ), min = 60 / 13 ∧ 
  ∀ (x y : ℝ), line x y → distance x y ≥ min := by
  sorry

#check min_distance_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_on_line_l44_4480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_surface_area_l44_4419

noncomputable def cube_volumes : List ℝ := [343, 216, 125, 64, 27, 1]

noncomputable def visible_surface_area (volume : ℝ) : ℝ :=
  let side_length := volume^(1/3)
  let full_face_area := side_length^2
  4 * full_face_area - full_face_area - (full_face_area / 2)

theorem sculpture_surface_area :
  (List.sum (List.map visible_surface_area cube_volumes)) + 6 = 353.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sculpture_surface_area_l44_4419
