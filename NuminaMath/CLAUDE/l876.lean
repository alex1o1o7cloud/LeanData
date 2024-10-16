import Mathlib

namespace NUMINAMATH_CALUDE_stratified_sampling_female_count_l876_87614

theorem stratified_sampling_female_count
  (total_students : ℕ)
  (male_students : ℕ)
  (female_students : ℕ)
  (sample_size : ℕ)
  (h1 : total_students = male_students + female_students)
  (h2 : total_students = 49)
  (h3 : male_students = 28)
  (h4 : female_students = 21)
  (h5 : sample_size = 14) :
  (sample_size : ℚ) / total_students * female_students = 6 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_count_l876_87614


namespace NUMINAMATH_CALUDE_land_area_calculation_l876_87691

theorem land_area_calculation (average_yield total_area first_area first_yield second_yield : ℝ) : 
  average_yield = 675 →
  first_area = 5 →
  first_yield = 705 →
  second_yield = 650 →
  total_area * average_yield = first_area * first_yield + (total_area - first_area) * second_yield →
  total_area = 11 :=
by sorry

end NUMINAMATH_CALUDE_land_area_calculation_l876_87691


namespace NUMINAMATH_CALUDE_hermia_election_probability_l876_87670

theorem hermia_election_probability (n : ℕ) (hodd : Odd n) (hpos : 0 < n) :
  let p := (2^n - 1) / (n * 2^(n-1) : ℝ)
  ∃ (probability_hermia_elected : ℝ),
    probability_hermia_elected = p ∧
    0 ≤ probability_hermia_elected ∧
    probability_hermia_elected ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_hermia_election_probability_l876_87670


namespace NUMINAMATH_CALUDE_addition_of_decimals_l876_87643

theorem addition_of_decimals : (0.3 : ℝ) + 0.03 = 0.33 := by
  sorry

end NUMINAMATH_CALUDE_addition_of_decimals_l876_87643


namespace NUMINAMATH_CALUDE_arithmetic_number_difference_l876_87630

/-- A 3-digit number with distinct digits forming an arithmetic sequence -/
def ArithmeticNumber : Type :=
  { n : ℕ // n ≥ 100 ∧ n < 1000 ∧ 
    ∃ (a b c : ℕ), n = 100 * a + 10 * b + c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b - a = c - b }

/-- The largest arithmetic number -/
def largest_arithmetic : ArithmeticNumber :=
  ⟨759, sorry⟩

/-- The smallest arithmetic number -/
def smallest_arithmetic : ArithmeticNumber :=
  ⟨123, sorry⟩

theorem arithmetic_number_difference :
  largest_arithmetic.val - smallest_arithmetic.val = 636 := by sorry

end NUMINAMATH_CALUDE_arithmetic_number_difference_l876_87630


namespace NUMINAMATH_CALUDE_triangle_area_l876_87699

theorem triangle_area (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin_sum : Real.sin (A + B) = 3/5)
  (h_sin_diff : Real.sin (A - B) = 1/5)
  (h_AB : 3 = 3) :
  (1/2) * 3 * (2 * Real.sqrt 6 - 2) = (6 + 3 * Real.sqrt 6) / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l876_87699


namespace NUMINAMATH_CALUDE_series_relationship_l876_87679

-- Define the sequence of exponents
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n + 2 => a n + a (n + 1)

-- Define the series
def series (n : ℕ) : ℕ := 2^(a n)

-- Theorem statement
theorem series_relationship (n : ℕ) :
  series n * series (n + 1) = series (n + 2) := by
  sorry


end NUMINAMATH_CALUDE_series_relationship_l876_87679


namespace NUMINAMATH_CALUDE_cuboids_intersecting_diagonal_l876_87617

/-- Represents a cuboid with integer side lengths -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents a cube with integer side length -/
structure Cube where
  sideLength : ℕ

/-- Counts the number of cuboids intersecting the diagonal of a cube -/
def countIntersectingCuboids (cuboid : Cuboid) (cube : Cube) : ℕ :=
  sorry

/-- The main theorem to be proved -/
theorem cuboids_intersecting_diagonal
  (smallCuboid : Cuboid)
  (largeCube : Cube)
  (h1 : smallCuboid.length = 2)
  (h2 : smallCuboid.width = 3)
  (h3 : smallCuboid.height = 5)
  (h4 : largeCube.sideLength = 90)
  (h5 : largeCube.sideLength % smallCuboid.length = 0)
  (h6 : largeCube.sideLength % smallCuboid.width = 0)
  (h7 : largeCube.sideLength % smallCuboid.height = 0) :
  countIntersectingCuboids smallCuboid largeCube = 65 := by
  sorry


end NUMINAMATH_CALUDE_cuboids_intersecting_diagonal_l876_87617


namespace NUMINAMATH_CALUDE_chemistry_physics_score_difference_l876_87638

theorem chemistry_physics_score_difference
  (math_score physics_score chemistry_score : ℕ)
  (total_math_physics : math_score + physics_score = 60)
  (avg_math_chemistry : (math_score + chemistry_score) / 2 = 40)
  (chemistry_higher : chemistry_score > physics_score) :
  chemistry_score - physics_score = 20 :=
by sorry

end NUMINAMATH_CALUDE_chemistry_physics_score_difference_l876_87638


namespace NUMINAMATH_CALUDE_corresponding_angles_equal_l876_87654

theorem corresponding_angles_equal (α β γ : ℝ) :
  α + β + γ = 180 ∧ (180 - α) + β + γ = 180 →
  α = 180 - α ∧ β = β ∧ γ = γ := by
  sorry

end NUMINAMATH_CALUDE_corresponding_angles_equal_l876_87654


namespace NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l876_87636

/-- A point in the grid --/
structure Point where
  x : Nat
  y : Nat

/-- The grid dimensions --/
def gridWidth : Nat := 4
def gridHeight : Nat := 5

/-- The initially colored squares --/
def initialColoredSquares : List Point := [
  { x := 1, y := 4 },
  { x := 2, y := 1 },
  { x := 4, y := 2 }
]

/-- A function to check if a point is within the grid --/
def isInGrid (p : Point) : Prop :=
  1 ≤ p.x ∧ p.x ≤ gridWidth ∧ 1 ≤ p.y ∧ p.y ≤ gridHeight

/-- A function to check if two points are symmetrical about the vertical line --/
def isVerticallySymmetric (p1 p2 : Point) : Prop :=
  p1.x + p2.x = gridWidth + 1 ∧ p1.y = p2.y

/-- A function to check if two points are symmetrical about the horizontal line --/
def isHorizontallySymmetric (p1 p2 : Point) : Prop :=
  p1.x = p2.x ∧ p1.y + p2.y = gridHeight + 1

/-- A function to check if two points are rotationally symmetric --/
def isRotationallySymmetric (p1 p2 : Point) : Prop :=
  p1.x + p2.x = gridWidth + 1 ∧ p1.y + p2.y = gridHeight + 1

/-- The main theorem --/
theorem min_additional_squares_for_symmetry :
  ∃ (additionalSquares : List Point),
    (∀ p ∈ additionalSquares, isInGrid p) ∧
    (∀ p ∈ initialColoredSquares ++ additionalSquares,
      (∃ q ∈ initialColoredSquares ++ additionalSquares, isVerticallySymmetric p q) ∧
      (∃ q ∈ initialColoredSquares ++ additionalSquares, isHorizontallySymmetric p q) ∧
      (∃ q ∈ initialColoredSquares ++ additionalSquares, isRotationallySymmetric p q)) ∧
    additionalSquares.length = 9 ∧
    (∀ (otherSquares : List Point),
      (∀ p ∈ otherSquares, isInGrid p) →
      (∀ p ∈ initialColoredSquares ++ otherSquares,
        (∃ q ∈ initialColoredSquares ++ otherSquares, isVerticallySymmetric p q) ∧
        (∃ q ∈ initialColoredSquares ++ otherSquares, isHorizontallySymmetric p q) ∧
        (∃ q ∈ initialColoredSquares ++ otherSquares, isRotationallySymmetric p q)) →
      otherSquares.length ≥ 9) :=
sorry

end NUMINAMATH_CALUDE_min_additional_squares_for_symmetry_l876_87636


namespace NUMINAMATH_CALUDE_sea_hidden_by_cloud_l876_87667

theorem sea_hidden_by_cloud (total_landscape visible_island cloud_cover : ℚ) :
  cloud_cover = 1/2 ∧ 
  visible_island = 1/4 ∧ 
  visible_island = 3/4 * (visible_island + (cloud_cover - 1/2)) →
  cloud_cover - (cloud_cover - 1/2) - visible_island = 5/12 :=
by sorry

end NUMINAMATH_CALUDE_sea_hidden_by_cloud_l876_87667


namespace NUMINAMATH_CALUDE_rain_probability_l876_87669

theorem rain_probability (p_saturday p_sunday : ℝ) 
  (h1 : p_saturday = 0.35)
  (h2 : p_sunday = 0.45)
  (h3 : 0 ≤ p_saturday ∧ p_saturday ≤ 1)
  (h4 : 0 ≤ p_sunday ∧ p_sunday ≤ 1) :
  1 - (1 - p_saturday) * (1 - p_sunday) = 0.6425 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l876_87669


namespace NUMINAMATH_CALUDE_quadratic_inequality_l876_87622

theorem quadratic_inequality (x : ℝ) :
  x^2 - 50*x + 575 ≤ 25 ↔ 25 - 5*Real.sqrt 3 ≤ x ∧ x ≤ 25 + 5*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l876_87622


namespace NUMINAMATH_CALUDE_local_call_cost_is_five_cents_l876_87690

/-- Represents the cost structure and duration of Freddy's phone calls -/
structure CallData where
  local_duration : ℕ
  international_duration : ℕ
  international_cost_per_minute : ℕ
  total_cost_cents : ℕ

/-- Calculates the cost of a local call per minute -/
def local_call_cost_per_minute (data : CallData) : ℚ :=
  (data.total_cost_cents - data.international_duration * data.international_cost_per_minute) / data.local_duration

/-- Theorem stating that the local call cost per minute is 5 cents -/
theorem local_call_cost_is_five_cents (data : CallData) 
    (h1 : data.local_duration = 45)
    (h2 : data.international_duration = 31)
    (h3 : data.international_cost_per_minute = 25)
    (h4 : data.total_cost_cents = 1000) :
    local_call_cost_per_minute data = 5 := by
  sorry

#eval local_call_cost_per_minute {
  local_duration := 45,
  international_duration := 31,
  international_cost_per_minute := 25,
  total_cost_cents := 1000
}

end NUMINAMATH_CALUDE_local_call_cost_is_five_cents_l876_87690


namespace NUMINAMATH_CALUDE_difference_ones_zeros_199_l876_87629

/-- Counts the number of ones in a binary representation --/
def countOnes (n : ℕ) : ℕ := sorry

/-- Counts the number of zeros in a binary representation --/
def countZeros (n : ℕ) : ℕ := sorry

/-- Converts a natural number to its binary representation --/
def toBinary (n : ℕ) : ℕ := sorry

theorem difference_ones_zeros_199 :
  let binary := toBinary 199
  let y := countOnes binary
  let x := countZeros binary
  y - x = 2 := by sorry

end NUMINAMATH_CALUDE_difference_ones_zeros_199_l876_87629


namespace NUMINAMATH_CALUDE_fixed_point_theorem_proof_l876_87696

def fixed_point_theorem (f : ℝ → ℝ) (h_inverse : Function.Bijective f) : Prop :=
  let f_inv := Function.invFun f
  (f_inv (-(-1) + 2) = 2) → (f ((-3) - 1) = -3)

theorem fixed_point_theorem_proof (f : ℝ → ℝ) (h_inverse : Function.Bijective f) :
  fixed_point_theorem f h_inverse := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_theorem_proof_l876_87696


namespace NUMINAMATH_CALUDE_distance_from_origin_l876_87672

theorem distance_from_origin (x y : ℝ) (n : ℝ) : 
  y = 15 →
  (x - 5)^2 + (y - 8)^2 = 13^2 →
  x > 5 →
  n^2 = x^2 + y^2 →
  n = Real.sqrt (370 + 20 * Real.sqrt 30) :=
by sorry

end NUMINAMATH_CALUDE_distance_from_origin_l876_87672


namespace NUMINAMATH_CALUDE_slope_determines_y_coordinate_l876_87628

/-- Given two points P and Q, if the slope of the line passing through them is 1/4,
    then the y-coordinate of Q is -3. -/
theorem slope_determines_y_coordinate 
  (x_P y_P x_Q : ℝ) (slope : ℝ) :
  x_P = -3 →
  y_P = -5 →
  x_Q = 5 →
  slope = 1/4 →
  (y_Q - y_P) / (x_Q - x_P) = slope →
  y_Q = -3 :=
by sorry

end NUMINAMATH_CALUDE_slope_determines_y_coordinate_l876_87628


namespace NUMINAMATH_CALUDE_exists_number_with_properties_l876_87604

/-- A function that counts the occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- A function that checks if a natural number contains only 7s and 5s -/
def containsOnly7sAnd5s (n : ℕ) : Prop := sorry

/-- Theorem stating the existence of a number with the required properties -/
theorem exists_number_with_properties : ∃ n : ℕ, 
  containsOnly7sAnd5s n ∧ 
  countDigit n 7 = countDigit n 5 ∧ 
  n % 7 = 0 ∧ 
  n % 5 = 0 := by sorry

end NUMINAMATH_CALUDE_exists_number_with_properties_l876_87604


namespace NUMINAMATH_CALUDE_bobby_adult_jumps_per_second_l876_87688

/-- Bobby's jumping ability as a child and adult -/
def bobby_jumping (child_jumps_per_minute : ℕ) (additional_jumps_per_minute : ℕ) : Prop :=
  let adult_jumps_per_minute := child_jumps_per_minute + additional_jumps_per_minute
  let adult_jumps_per_second := adult_jumps_per_minute / 60
  adult_jumps_per_second = 1

/-- Theorem: Bobby can jump 1 time per second as an adult -/
theorem bobby_adult_jumps_per_second :
  bobby_jumping 30 30 := by
  sorry

end NUMINAMATH_CALUDE_bobby_adult_jumps_per_second_l876_87688


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l876_87621

/-- Two points are symmetric about the y-axis if their y-coordinates are the same and their x-coordinates are opposites -/
def symmetric_about_y_axis (p1 p2 : ℝ × ℝ) : Prop :=
  p1.2 = p2.2 ∧ p1.1 = -p2.1

theorem symmetric_points_sum_power (a b : ℝ) :
  symmetric_about_y_axis (a, 3) (2, b) → (a + b)^2015 = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l876_87621


namespace NUMINAMATH_CALUDE_coins_per_roll_is_25_l876_87659

/-- Represents the number of coins in a single roll -/
def coins_per_roll : ℕ := sorry

/-- The number of rolls each bank teller has -/
def rolls_per_teller : ℕ := 10

/-- The number of bank tellers -/
def number_of_tellers : ℕ := 4

/-- The total number of coins among all tellers -/
def total_coins : ℕ := 1000

theorem coins_per_roll_is_25 : 
  coins_per_roll * rolls_per_teller * number_of_tellers = total_coins →
  coins_per_roll = 25 := by
  sorry

end NUMINAMATH_CALUDE_coins_per_roll_is_25_l876_87659


namespace NUMINAMATH_CALUDE_range_of_f_inequality_l876_87652

open Real

noncomputable def f (x : ℝ) : ℝ := 2*x + sin x

theorem range_of_f_inequality (h1 : ∀ x ∈ Set.Ioo (-2) 2, HasDerivAt f (2 + cos x) x)
                              (h2 : f 0 = 0) :
  {x : ℝ | f (1 + x) + f (x - x^2) > 0} = Set.Ioo (1 - Real.sqrt 2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_f_inequality_l876_87652


namespace NUMINAMATH_CALUDE_square_graph_triangles_l876_87610

/-- Represents a planar graph formed by connecting points in a square --/
structure SquareGraph where
  /-- The number of internal points marked in the square --/
  internalPoints : ℕ
  /-- The total number of vertices (internal points + 4 square vertices) --/
  totalVertices : ℕ
  /-- The number of edges in the graph --/
  edges : ℕ
  /-- The number of faces (regions) formed, including the external face --/
  faces : ℕ
  /-- Condition: The total vertices is the sum of internal points and square vertices --/
  vertexCount : totalVertices = internalPoints + 4
  /-- Condition: Euler's formula for planar graphs --/
  eulerFormula : totalVertices - edges + faces = 2
  /-- Condition: Relationship between edges and faces --/
  edgeFaceRelation : 2 * edges = 3 * (faces - 1) + 4

/-- Theorem: In a square with 20 internal points connected as described, 42 triangles are formed --/
theorem square_graph_triangles (g : SquareGraph) (h : g.internalPoints = 20) : g.faces - 1 = 42 := by
  sorry


end NUMINAMATH_CALUDE_square_graph_triangles_l876_87610


namespace NUMINAMATH_CALUDE_proposition_not_hold_for_2_l876_87640

theorem proposition_not_hold_for_2 (P : ℕ → Prop)
  (h1 : ¬ P 3)
  (h2 : ∀ k : ℕ, k > 0 → P k → P (k + 1)) :
  ¬ P 2 := by
  sorry

end NUMINAMATH_CALUDE_proposition_not_hold_for_2_l876_87640


namespace NUMINAMATH_CALUDE_max_value_2x_minus_y_l876_87605

theorem max_value_2x_minus_y (x y : ℝ) 
  (h1 : x + y - 1 < 0) 
  (h2 : x - y ≤ 0) 
  (h3 : x ≥ 0) : 
  ∀ z, z = 2*x - y → z ≤ (1/2 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_max_value_2x_minus_y_l876_87605


namespace NUMINAMATH_CALUDE_start_page_second_day_l876_87608

/-- Given a book with 200 pages, and 20% read on the first day,
    prove that the page number to start reading on the second day is 41. -/
theorem start_page_second_day (total_pages : ℕ) (percent_read : ℚ) : 
  total_pages = 200 → percent_read = 1/5 → 
  (total_pages : ℚ) * percent_read + 1 = 41 := by
  sorry

end NUMINAMATH_CALUDE_start_page_second_day_l876_87608


namespace NUMINAMATH_CALUDE_exists_30digit_root_l876_87603

/-- A function that checks if a number is a three-digit natural number -/
def isThreeDigitNatural (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- The main theorem -/
theorem exists_30digit_root (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℕ) 
  (h₀ : isThreeDigitNatural a₀)
  (h₁ : isThreeDigitNatural a₁)
  (h₂ : isThreeDigitNatural a₂)
  (h₃ : isThreeDigitNatural a₃)
  (h₄ : isThreeDigitNatural a₄)
  (h₅ : isThreeDigitNatural a₅)
  (h₆ : isThreeDigitNatural a₆)
  (h₇ : isThreeDigitNatural a₇)
  (h₈ : isThreeDigitNatural a₈)
  (h₉ : isThreeDigitNatural a₉) :
  ∃ (N : ℕ) (x : ℤ), 
    (N ≥ 10^29 ∧ N < 10^30) ∧ 
    (a₉ * x^9 + a₈ * x^8 + a₇ * x^7 + a₆ * x^6 + a₅ * x^5 + 
     a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀ = N) := by
  sorry

end NUMINAMATH_CALUDE_exists_30digit_root_l876_87603


namespace NUMINAMATH_CALUDE_line_equation_through_point_l876_87658

/-- The equation of a line with slope 2 passing through the point (2, 3) is 2x - y - 1 = 0 -/
theorem line_equation_through_point (x y : ℝ) :
  let slope : ℝ := 2
  let point : ℝ × ℝ := (2, 3)
  (y - point.2 = slope * (x - point.1)) ↔ (2 * x - y - 1 = 0) := by
sorry

end NUMINAMATH_CALUDE_line_equation_through_point_l876_87658


namespace NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l876_87634

theorem integer_solutions_quadratic_equation :
  ∀ x y : ℤ, x^2 - x*y - 6*y^2 + 2*x + 19*y = 18 ↔ (x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_quadratic_equation_l876_87634


namespace NUMINAMATH_CALUDE_length_of_segment_AB_l876_87641

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y + 3 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 6 = 0

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  line_l A.1 A.2 ∧ circle_C A.1 A.2 ∧
  line_l B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_segment_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_length_of_segment_AB_l876_87641


namespace NUMINAMATH_CALUDE_half_area_of_rectangle_l876_87609

/-- Half the area of a rectangle with width 25 cm and height 16 cm is 200 cm². -/
theorem half_area_of_rectangle (width height : ℝ) (h1 : width = 25) (h2 : height = 16) :
  (width * height) / 2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_half_area_of_rectangle_l876_87609


namespace NUMINAMATH_CALUDE_workers_completion_time_l876_87685

/-- Given two workers who can each complete a task in 32 days, 
    prove they can complete the task together in 16 days -/
theorem workers_completion_time (work_rate_A work_rate_B : ℝ) : 
  work_rate_A = 1 / 32 →
  work_rate_B = 1 / 32 →
  1 / (work_rate_A + work_rate_B) = 16 := by
sorry

end NUMINAMATH_CALUDE_workers_completion_time_l876_87685


namespace NUMINAMATH_CALUDE_seven_balls_four_boxes_l876_87660

/-- The number of ways to partition n into at most k parts, where the order doesn't matter -/
def partition_count (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 11 ways to partition 7 into at most 4 parts -/
theorem seven_balls_four_boxes : partition_count 7 4 = 11 := by sorry

end NUMINAMATH_CALUDE_seven_balls_four_boxes_l876_87660


namespace NUMINAMATH_CALUDE_max_segment_product_l876_87662

-- Define the segment AB of unit length
def unitSegment : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 1}

-- Define a function to calculate the product of segment lengths
def segmentProduct (a b c : ℝ) : ℝ :=
  a * (a + b) * 1 * b * (b + c) * c

-- Theorem statement
theorem max_segment_product :
  ∃ (max : ℝ), max = Real.sqrt 5 / 125 ∧
  ∀ (a b c : ℝ), a ∈ unitSegment → b ∈ unitSegment → c ∈ unitSegment →
  a + b + c = 1 → segmentProduct a b c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_segment_product_l876_87662


namespace NUMINAMATH_CALUDE_isosceles_triangle_most_stable_l876_87665

-- Define the shapes
inductive Shape
  | RegularHexagon
  | Square
  | Pentagon
  | IsoscelesTriangle

-- Define a function to get the number of sides for each shape
def numSides (s : Shape) : Nat :=
  match s with
  | .RegularHexagon => 6
  | .Square => 4
  | .Pentagon => 5
  | .IsoscelesTriangle => 3

-- Define stability as inversely proportional to the number of sides
def stability (s : Shape) : Nat := 7 - numSides s

-- Theorem: Isosceles triangle is the most stable shape
theorem isosceles_triangle_most_stable :
  ∀ s : Shape, s ≠ Shape.IsoscelesTriangle → 
    stability Shape.IsoscelesTriangle > stability s :=
by sorry

#check isosceles_triangle_most_stable

end NUMINAMATH_CALUDE_isosceles_triangle_most_stable_l876_87665


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l876_87616

/-- The total cost of circus tickets for a group of kids and adults -/
def total_ticket_cost (num_kids : ℕ) (num_adults : ℕ) (kid_ticket_price : ℚ) : ℚ :=
  let adult_ticket_price := 2 * kid_ticket_price
  num_kids * kid_ticket_price + num_adults * adult_ticket_price

/-- Theorem stating the total cost of circus tickets for a specific group -/
theorem circus_ticket_cost :
  total_ticket_cost 6 2 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l876_87616


namespace NUMINAMATH_CALUDE_exposed_sides_is_21_l876_87635

/-- Represents a polygon with a specific number of sides -/
structure Polygon where
  sides : ℕ
  sides_positive : sides > 0

/-- Represents the configuration of polygons -/
structure PolygonConfiguration where
  triangle : Polygon
  square : Polygon
  pentagon : Polygon
  hexagon : Polygon
  heptagon : Polygon
  triangle_is_equilateral : triangle.sides = 3
  square_is_square : square.sides = 4
  pentagon_is_pentagon : pentagon.sides = 5
  hexagon_is_hexagon : hexagon.sides = 6
  heptagon_is_heptagon : heptagon.sides = 7

/-- The number of shared sides in the configuration -/
def shared_sides : ℕ := 4

/-- Theorem stating that the number of exposed sides in the configuration is 21 -/
theorem exposed_sides_is_21 (config : PolygonConfiguration) : 
  config.triangle.sides + config.square.sides + config.pentagon.sides + 
  config.hexagon.sides + config.heptagon.sides - shared_sides = 21 := by
  sorry

end NUMINAMATH_CALUDE_exposed_sides_is_21_l876_87635


namespace NUMINAMATH_CALUDE_prob_two_red_cards_standard_deck_l876_87650

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (red_suits : Nat)
  (black_suits : Nat)

/-- Definition of a standard deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    suits := 4,
    cards_per_suit := 13,
    red_suits := 2,
    black_suits := 2 }

/-- Probability of drawing two red cards in succession -/
def prob_two_red_cards (d : Deck) : Rat :=
  let red_cards := d.red_suits * d.cards_per_suit
  let first_draw := red_cards / d.total_cards
  let second_draw := (red_cards - 1) / (d.total_cards - 1)
  first_draw * second_draw

/-- Theorem: The probability of drawing two red cards in succession from a standard deck is 25/102 -/
theorem prob_two_red_cards_standard_deck :
  prob_two_red_cards standard_deck = 25 / 102 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_red_cards_standard_deck_l876_87650


namespace NUMINAMATH_CALUDE_pencil_cost_l876_87618

theorem pencil_cost (total_money : ℕ) (num_pencils : ℕ) (cost_per_pencil : ℕ) : 
  total_money = 50 → 
  num_pencils = 10 → 
  total_money = num_pencils * cost_per_pencil → 
  cost_per_pencil = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l876_87618


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l876_87647

theorem similar_triangles_leg_length (x : ℝ) : x > 0 →
  (12 : ℝ) / x = 9 / 7 → x = 84 / 9 := by sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l876_87647


namespace NUMINAMATH_CALUDE_collinear_vector_combinations_l876_87682

/-- Given two non-zero vectors in a real vector space that are not collinear,
    if a linear combination of these vectors with scalar k is collinear with
    another linear combination of the same vectors where k's role is swapped,
    then k must be either 1 or -1. -/
theorem collinear_vector_combinations (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (e₁ e₂ : V) (k : ℝ) 
  (h_nonzero₁ : e₁ ≠ 0)
  (h_nonzero₂ : e₂ ≠ 0)
  (h_not_collinear : ¬ ∃ (c : ℝ), e₁ = c • e₂)
  (h_collinear : ∃ (t : ℝ), k • e₁ + e₂ = t • (e₁ + k • e₂)) :
  k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_CALUDE_collinear_vector_combinations_l876_87682


namespace NUMINAMATH_CALUDE_real_axis_length_l876_87639

/-- Hyperbola C with center at origin and foci on x-axis -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  center_origin : equation 0 0
  foci_on_x_axis : ∀ y, ¬(∃ x ≠ 0, equation x y ∧ equation (-x) y)

/-- Parabola with equation y² = 16x -/
def Parabola : ℝ → ℝ → Prop :=
  λ x y => y^2 = 16 * x

/-- Directrix of the parabola y² = 16x -/
def Directrix : ℝ → Prop :=
  λ x => x = -4

/-- Points A and B where hyperbola C intersects the directrix -/
structure IntersectionPoints (C : Hyperbola) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  on_directrix : Directrix A.1 ∧ Directrix B.1
  on_hyperbola : C.equation A.1 A.2 ∧ C.equation B.1 B.2
  distance : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 3

/-- The theorem to be proved -/
theorem real_axis_length (C : Hyperbola) (AB : IntersectionPoints C) :
  ∃ a : ℝ, a = 4 ∧ ∀ x y, C.equation x y ↔ x^2 / a^2 - y^2 / (a^2 - 4) = 1 :=
sorry

end NUMINAMATH_CALUDE_real_axis_length_l876_87639


namespace NUMINAMATH_CALUDE_teacher_assignment_ways_l876_87637

/-- The number of ways to assign teachers to classes -/
def assignmentWays (n m : ℕ) : ℕ :=
  Nat.choose n 2 * Nat.choose (n - 2) 2 * Nat.choose (n - 4) 2 * Nat.choose (n - 6) 2

/-- Theorem stating the number of ways to assign 4 teachers to 8 classes -/
theorem teacher_assignment_ways :
  assignmentWays 8 4 = 2520 :=
by sorry

#eval assignmentWays 8 4

end NUMINAMATH_CALUDE_teacher_assignment_ways_l876_87637


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l876_87645

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0) ↔ m ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l876_87645


namespace NUMINAMATH_CALUDE_fruit_drink_volume_l876_87600

theorem fruit_drink_volume (grapefruit_percent : ℝ) (lemon_percent : ℝ) (orange_volume : ℝ) :
  grapefruit_percent = 0.25 →
  lemon_percent = 0.35 →
  orange_volume = 20 →
  ∃ total_volume : ℝ,
    total_volume = 50 ∧
    grapefruit_percent * total_volume + lemon_percent * total_volume + orange_volume = total_volume :=
by sorry

end NUMINAMATH_CALUDE_fruit_drink_volume_l876_87600


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l876_87693

/-- Computes the annual interest rate given the principal, time, compounding frequency, and final amount -/
def calculate_interest_rate (principal : ℝ) (time : ℝ) (compounding_frequency : ℕ) (final_amount : ℝ) : ℝ :=
  sorry

theorem interest_rate_calculation (principal : ℝ) (time : ℝ) (compounding_frequency : ℕ) (final_amount : ℝ) 
  (h1 : principal = 6000)
  (h2 : time = 1.5)
  (h3 : compounding_frequency = 2)
  (h4 : final_amount = 6000 + 945.75) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ 
  |calculate_interest_rate principal time compounding_frequency final_amount - 0.099| < ε :=
sorry

end NUMINAMATH_CALUDE_interest_rate_calculation_l876_87693


namespace NUMINAMATH_CALUDE_cookie_problem_l876_87680

/-- Cookie problem statement -/
theorem cookie_problem (alyssa_cookies aiyanna_cookies brady_cookies : ℕ) 
  (h1 : alyssa_cookies = 1523)
  (h2 : aiyanna_cookies = 3720)
  (h3 : brady_cookies = 2265) :
  (aiyanna_cookies - alyssa_cookies = 2197) ∧ 
  (aiyanna_cookies - brady_cookies = 1455) ∧ 
  (brady_cookies - alyssa_cookies = 742) := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l876_87680


namespace NUMINAMATH_CALUDE_square_root_of_sixteen_l876_87674

theorem square_root_of_sixteen : 
  {x : ℝ | x^2 = 16} = {4, -4} := by sorry

end NUMINAMATH_CALUDE_square_root_of_sixteen_l876_87674


namespace NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_tan_x_given_sin_x_l876_87657

-- Part I
theorem cos_negative_nineteen_pi_sixths :
  Real.cos (-19 * π / 6) = -Real.sqrt 3 / 2 := by sorry

-- Part II
theorem tan_x_given_sin_x (x : Real) :
  x ∈ Set.Icc (π / 2) (3 * π / 2) →
  Real.sin x = -3 / 5 →
  Real.tan x = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_cos_negative_nineteen_pi_sixths_tan_x_given_sin_x_l876_87657


namespace NUMINAMATH_CALUDE_power_less_than_threshold_l876_87612

theorem power_less_than_threshold : ∃ (n1 n2 n3 : ℕ+),
  (0.99 : ℝ) ^ (n1 : ℝ) < 0.000001 ∧
  (0.999 : ℝ) ^ (n2 : ℝ) < 0.000001 ∧
  (0.999999 : ℝ) ^ (n3 : ℝ) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_power_less_than_threshold_l876_87612


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l876_87611

theorem complex_fraction_simplification : (Complex.I + 2) / (1 - 2 * Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l876_87611


namespace NUMINAMATH_CALUDE_a_capital_is_15000_l876_87655

/-- The amount of money partner a put into the business -/
def a_capital : ℝ := sorry

/-- The amount of money partner b put into the business -/
def b_capital : ℝ := 25000

/-- The total profit of the business -/
def total_profit : ℝ := 9600

/-- The percentage of profit a receives for managing the business -/
def management_fee_percentage : ℝ := 0.1

/-- The total amount a receives -/
def a_total_received : ℝ := 4200

theorem a_capital_is_15000 :
  a_capital = 15000 :=
by
  sorry

#check a_capital_is_15000

end NUMINAMATH_CALUDE_a_capital_is_15000_l876_87655


namespace NUMINAMATH_CALUDE_incircle_radius_of_specific_triangle_l876_87624

theorem incircle_radius_of_specific_triangle : 
  ∀ (a b c h : ℝ) (r : ℝ),
  a = 5 ∧ b = 12 ∧ c = 13 ∧ h = 10 →
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem to ensure right-angled triangle
  r = (b * h / 2) / ((a + b + c) / 2) →
  r = 4 := by sorry

end NUMINAMATH_CALUDE_incircle_radius_of_specific_triangle_l876_87624


namespace NUMINAMATH_CALUDE_simplify_fraction_l876_87694

theorem simplify_fraction : 
  (5^1004)^2 - (5^1002)^2 / (5^1003)^2 - (5^1001)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_simplify_fraction_l876_87694


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l876_87649

theorem geometric_series_first_term 
  (sum : ℝ) 
  (sum_squares : ℝ) 
  (h1 : sum = 20) 
  (h2 : sum_squares = 80) : 
  ∃ (a r : ℝ), 
    a / (1 - r) = sum ∧ 
    a^2 / (1 - r^2) = sum_squares ∧ 
    a = 20 / 3 := by 
sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l876_87649


namespace NUMINAMATH_CALUDE_situp_competition_result_l876_87692

/-- Adam's sit-up performance -/
def adam_situps (round : ℕ) : ℕ :=
  40 - 8 * (round - 1)

/-- Barney's sit-up performance -/
def barney_situps : ℕ := 45

/-- Carrie's sit-up performance -/
def carrie_situps : ℕ := 2 * barney_situps

/-- Jerrie's sit-up performance -/
def jerrie_situps : ℕ := carrie_situps + 5

/-- Total sit-ups for Adam -/
def adam_total : ℕ :=
  (adam_situps 1) + (adam_situps 2) + (adam_situps 3)

/-- Total sit-ups for Barney -/
def barney_total : ℕ := barney_situps * 5

/-- Total sit-ups for Carrie -/
def carrie_total : ℕ := carrie_situps * 4

/-- Total sit-ups for Jerrie -/
def jerrie_total : ℕ := jerrie_situps * 6

/-- The combined total of sit-ups -/
def combined_total : ℕ :=
  adam_total + barney_total + carrie_total + jerrie_total

theorem situp_competition_result :
  combined_total = 1251 := by
  sorry

end NUMINAMATH_CALUDE_situp_competition_result_l876_87692


namespace NUMINAMATH_CALUDE_trailing_zeros_500_factorial_l876_87683

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ := (Finset.range n).prod (λ i => i + 1)

/-- Definition to count trailing zeros -/
def trailingZeros (n : ℕ) : ℕ :=
  Nat.log 10 (Nat.gcd n (10^(Nat.log 2 n + 1)))

/-- Theorem: The number of trailing zeros in 500! is 124 -/
theorem trailing_zeros_500_factorial :
  trailingZeros (factorial 500) = 124 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_500_factorial_l876_87683


namespace NUMINAMATH_CALUDE_oldest_child_age_oldest_child_age_proof_l876_87656

/-- Proves that given four children with an average age of 8 years, 
    and three of them being 5, 7, and 10 years old, 
    the age of the fourth child is 10 years. -/
theorem oldest_child_age 
  (total_children : Nat)
  (average_age : ℚ)
  (younger_children_ages : List Nat)
  (h1 : total_children = 4)
  (h2 : average_age = 8)
  (h3 : younger_children_ages = [5, 7, 10])
  : Nat :=
10

theorem oldest_child_age_proof : oldest_child_age 4 8 [5, 7, 10] rfl rfl rfl = 10 := by
  sorry

end NUMINAMATH_CALUDE_oldest_child_age_oldest_child_age_proof_l876_87656


namespace NUMINAMATH_CALUDE_base5_123_equals_38_l876_87646

/-- Converts a base-5 number to decimal --/
def base5ToDecimal (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 5^2 + tens * 5^1 + ones * 5^0

/-- Theorem: The base-5 number 123₍₅₎ is equal to the decimal number 38 --/
theorem base5_123_equals_38 : base5ToDecimal 1 2 3 = 38 := by
  sorry

end NUMINAMATH_CALUDE_base5_123_equals_38_l876_87646


namespace NUMINAMATH_CALUDE_imohkprelim_combinations_l876_87602

def letter_list : List Char := ['I', 'M', 'O', 'H', 'K', 'P', 'R', 'E', 'L', 'I', 'M']

def count_combinations (letters : List Char) : Nat :=
  let unique_letters := letters.eraseDups
  let combinations_distinct := Nat.choose unique_letters.length 3
  let combinations_with_repeat := 
    (letters.filter (λ c => letters.count c > 1)).eraseDups.length * (unique_letters.length - 1)
  combinations_distinct + combinations_with_repeat

theorem imohkprelim_combinations :
  count_combinations letter_list = 100 := by
  sorry

end NUMINAMATH_CALUDE_imohkprelim_combinations_l876_87602


namespace NUMINAMATH_CALUDE_ratio_HD_HA_is_5_11_l876_87686

/-- A triangle with sides of lengths 13, 14, and 15 -/
structure Triangle :=
  (a b c : ℝ)
  (side_a : a = 13)
  (side_b : b = 14)
  (side_c : c = 15)

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The altitude from vertex A to the side of length 14 -/
def altitude_AD (t : Triangle) : ℝ := sorry

/-- The ratio of HD to HA -/
def ratio_HD_HA (t : Triangle) : ℚ := sorry

/-- Theorem: The ratio HD:HA is 5:11 -/
theorem ratio_HD_HA_is_5_11 (t : Triangle) : 
  ratio_HD_HA t = 5 / 11 := by sorry

end NUMINAMATH_CALUDE_ratio_HD_HA_is_5_11_l876_87686


namespace NUMINAMATH_CALUDE_cube_root_equation_solutions_l876_87673

theorem cube_root_equation_solutions :
  ∀ x : ℝ, (x^(1/3) = 15 / (8 - x^(1/3))) ↔ (x = 27 ∨ x = 125) := by sorry

end NUMINAMATH_CALUDE_cube_root_equation_solutions_l876_87673


namespace NUMINAMATH_CALUDE_new_average_weight_with_D_l876_87644

/-- Given the weights of individuals A, B, C, D, and E, prove that the average weight
    of the group when D joins is 82 kg. -/
theorem new_average_weight_with_D (w_A w_B w_C w_D : ℝ) : 
  w_A = 95 →
  (w_A + w_B + w_C) / 3 = 80 →
  ∃ w_E : ℝ, w_E = w_D + 3 ∧ (w_B + w_C + w_D + w_E) / 4 = 81 →
  (w_A + w_B + w_C + w_D) / 4 = 82 := by
  sorry


end NUMINAMATH_CALUDE_new_average_weight_with_D_l876_87644


namespace NUMINAMATH_CALUDE_abc_equality_l876_87663

theorem abc_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : 4 * a * b * c * (a + b + c) = (a + b)^2 * (a + c)^2) :
  a * (a + b + c) = b * c := by
sorry

end NUMINAMATH_CALUDE_abc_equality_l876_87663


namespace NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l876_87689

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  depth : ℕ

/-- Calculates the smallest number of identical cubes that can fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : ℕ :=
  let sideLengthOfCube := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / sideLengthOfCube) * (box.width / sideLengthOfCube) * (box.depth / sideLengthOfCube)

/-- The theorem stating that for a box with given dimensions, 
    the smallest number of identical cubes that can fill it completely is 84 -/
theorem smallest_number_of_cubes_for_given_box : 
  smallestNumberOfCubes ⟨49, 42, 14⟩ = 84 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_of_cubes_for_given_box_l876_87689


namespace NUMINAMATH_CALUDE_land_to_cabin_ratio_example_l876_87631

/-- Given a total cost and cabin cost, calculate the ratio of land cost to cabin cost -/
def land_to_cabin_ratio (total_cost cabin_cost : ℕ) : ℚ :=
  (total_cost - cabin_cost) / cabin_cost

/-- Theorem: The ratio of land cost to cabin cost is 4 when the total cost is $30,000 and the cabin cost is $6,000 -/
theorem land_to_cabin_ratio_example : land_to_cabin_ratio 30000 6000 = 4 := by
  sorry

end NUMINAMATH_CALUDE_land_to_cabin_ratio_example_l876_87631


namespace NUMINAMATH_CALUDE_triangle_properties_l876_87661

theorem triangle_properties (a b c A B C : ℝ) (h1 : a * Real.sin A - c * Real.sin C = (a - b) * Real.sin B)
  (h2 : c = 2 * Real.sqrt 3) (h3 : (1/2) * a * b * Real.sin C = 2 * Real.sqrt 3) :
  C = π / 3 ∧ a + b + c = 6 + 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l876_87661


namespace NUMINAMATH_CALUDE_exactly_two_statements_true_l876_87684

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define symmetry about y-axis
def symmetricAboutYAxis (a b : Point2D) : Prop :=
  a.x = -b.x ∧ a.y = b.y

-- Define symmetry about x-axis
def symmetricAboutXAxis (a b : Point2D) : Prop :=
  a.x = b.x ∧ a.y = -b.y

-- Define the four statements
def statement1 (a b : Point2D) : Prop :=
  symmetricAboutYAxis a b → a.y = b.y

def statement2 (a b : Point2D) : Prop :=
  a.y = b.y → symmetricAboutYAxis a b

def statement3 (a b : Point2D) : Prop :=
  a.x = b.x → symmetricAboutXAxis a b

def statement4 (a b : Point2D) : Prop :=
  symmetricAboutXAxis a b → a.x = b.x

-- Theorem stating that exactly two of the statements are true
theorem exactly_two_statements_true :
  ∃ (a b : Point2D),
    (statement1 a b ∧ ¬statement2 a b ∧ ¬statement3 a b ∧ statement4 a b) :=
  sorry

end NUMINAMATH_CALUDE_exactly_two_statements_true_l876_87684


namespace NUMINAMATH_CALUDE_last_remaining_100_l876_87677

def last_remaining (n : ℕ) : ℕ :=
  if n ≤ 1 then n else
  let m := n / 2
  2 * (if m % 2 = 0 then last_remaining m else m + 1 - last_remaining m)

theorem last_remaining_100 : last_remaining 100 = 64 := by
  sorry

end NUMINAMATH_CALUDE_last_remaining_100_l876_87677


namespace NUMINAMATH_CALUDE_function_composition_equality_l876_87671

/-- Given real numbers a, b, c, d, and functions f and h, 
    prove that f(h(x)) = h(f(x)) for all x if and only if a = c or b = d -/
theorem function_composition_equality 
  (a b c d : ℝ) 
  (f : ℝ → ℝ) 
  (h : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x + b) 
  (hh : ∀ x, h x = c * x + d) : 
  (∀ x, f (h x) = h (f x)) ↔ (a = c ∨ b = d) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l876_87671


namespace NUMINAMATH_CALUDE_logarithmic_function_value_l876_87695

noncomputable def f (a : ℝ) (x : ℝ) := (a^2 + a - 5) * Real.log x / Real.log a

theorem logarithmic_function_value (a : ℝ) :
  (a > 0) →
  (a ≠ 1) →
  (a^2 + a - 5 = 1) →
  f a (1/8) = -3 :=
by sorry

end NUMINAMATH_CALUDE_logarithmic_function_value_l876_87695


namespace NUMINAMATH_CALUDE_bucket_capacity_proof_l876_87676

/-- The capacity of a bucket that satisfies the given conditions -/
def bucket_capacity : ℝ :=
  let tank_capacity : ℝ := 48
  let small_bucket_capacity : ℝ := 3
  3

theorem bucket_capacity_proof :
  let tank_capacity : ℝ := 48
  let small_bucket_capacity : ℝ := 3
  (tank_capacity / bucket_capacity : ℝ) = (tank_capacity / small_bucket_capacity) - 4 := by
  sorry

#check bucket_capacity_proof

end NUMINAMATH_CALUDE_bucket_capacity_proof_l876_87676


namespace NUMINAMATH_CALUDE_strawberry_preference_percentage_l876_87642

def total_responses : ℕ := 80 + 70 + 90 + 60 + 50
def strawberry_responses : ℕ := 90

def strawberry_percentage : ℚ :=
  (strawberry_responses : ℚ) / (total_responses : ℚ) * 100

theorem strawberry_preference_percentage :
  (strawberry_percentage : ℚ) = 25.71 := by sorry

end NUMINAMATH_CALUDE_strawberry_preference_percentage_l876_87642


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_seven_l876_87681

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

def is_multiple_of_seven (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 7 * k

theorem least_non_lucky_multiple_of_seven :
  (14 % sum_of_digits 14 ≠ 0) ∧
  is_multiple_of_seven 14 ∧
  ∀ n : ℕ, 0 < n ∧ n < 14 ∧ is_multiple_of_seven n → is_lucky n :=
by sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_seven_l876_87681


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l876_87607

-- Define the parabola
def parabola (x y : ℝ) : Prop := y = x^2 - x - 6

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - x + 5*y - 6 = 0

-- Define the point P
def point_P : ℝ × ℝ := (-2, 5)

-- Define the line l (vertical case)
def line_l_vertical (x : ℝ) : Prop := x = -2

-- Define the line l (non-vertical case)
def line_l_nonvertical (x y : ℝ) : Prop := 4*x + 3*y - 7 = 0

-- Theorem statement
theorem circle_and_line_problem :
  ∃ (A B : ℝ × ℝ),
    -- Circle C passes through intersection points of parabola and axes
    (∀ (x y : ℝ), (x = 0 ∨ y = 0) ∧ parabola x y → circle_C x y) ∧
    -- Line l passes through P and intersects C at A and B
    (line_l_vertical A.1 ∨ line_l_nonvertical A.1 A.2) ∧
    (line_l_vertical B.1 ∨ line_l_nonvertical B.1 B.2) ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    -- Tangents at A and B are perpendicular
    (∃ (tA tB : ℝ × ℝ → ℝ × ℝ),
      (tA A = B ∨ tB B = A) →
      (tA A • tB B = 0)) →
    -- Conclusion: Equations of circle C and line l
    (∀ (x y : ℝ), circle_C x y ↔ x^2 + y^2 - x + 5*y - 6 = 0) ∧
    (∀ (x y : ℝ), (x = -2 ∨ 4*x + 3*y - 7 = 0) ↔ (line_l_vertical x ∨ line_l_nonvertical x y))
  := by sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l876_87607


namespace NUMINAMATH_CALUDE_absolute_difference_m_n_l876_87606

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem absolute_difference_m_n (m n : ℝ) 
  (h : (m + 2 * i) / i = n + i) : 
  |m - n| = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_difference_m_n_l876_87606


namespace NUMINAMATH_CALUDE_solution_replacement_fraction_l876_87664

theorem solution_replacement_fraction (initial_conc : ℚ) (replacement_conc : ℚ) (final_conc : ℚ)
  (h_initial : initial_conc = 60 / 100)
  (h_replacement : replacement_conc = 25 / 100)
  (h_final : final_conc = 35 / 100) :
  let replaced_fraction := (initial_conc - final_conc) / (initial_conc - replacement_conc)
  replaced_fraction = 5 / 7 := by
sorry

end NUMINAMATH_CALUDE_solution_replacement_fraction_l876_87664


namespace NUMINAMATH_CALUDE_two_numbers_difference_l876_87633

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 32) :
  |x - y| = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l876_87633


namespace NUMINAMATH_CALUDE_max_m_value_l876_87625

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_eq : 2/a + 1/b = 1/4) :
  (∀ m : ℝ, 2*a + b ≥ 9*m) → (∃ m_max : ℝ, m_max = 4 ∧ ∀ m : ℝ, (2*a + b ≥ 9*m → m ≤ m_max)) :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l876_87625


namespace NUMINAMATH_CALUDE_meet_once_l876_87666

/-- Represents the movement of Michael and the garbage truck --/
structure Movement where
  michael_speed : ℝ
  truck_speed : ℝ
  pail_distance : ℝ
  truck_stop_time : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def count_meetings (m : Movement) : ℕ :=
  sorry

/-- The specific scenario described in the problem --/
def problem_scenario : Movement where
  michael_speed := 4
  truck_speed := 12
  pail_distance := 200
  truck_stop_time := 20

/-- Theorem stating that Michael and the truck meet exactly once --/
theorem meet_once :
  count_meetings problem_scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l876_87666


namespace NUMINAMATH_CALUDE_abs_neg_two_equals_two_l876_87619

theorem abs_neg_two_equals_two : abs (-2 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_two_equals_two_l876_87619


namespace NUMINAMATH_CALUDE_woojin_harvest_l876_87627

/-- The amount of potatoes harvested by Woojin's family -/
def potato_harvest (younger_brother older_sister woojin : ℝ) : Prop :=
  -- Younger brother's harvest
  younger_brother = 3.8 ∧
  -- Older sister's harvest
  older_sister = younger_brother + 8.4 ∧
  -- Woojin's harvest in grams
  woojin * 1000 = (older_sister / 10) * 1000 + 3720

theorem woojin_harvest :
  ∀ younger_brother older_sister woojin : ℝ,
  potato_harvest younger_brother older_sister woojin →
  woojin = 4.94 := by
sorry

end NUMINAMATH_CALUDE_woojin_harvest_l876_87627


namespace NUMINAMATH_CALUDE_area_of_closed_figure_l876_87615

/-- The area of the closed figure bounded by y = 1/2, y = 2, y = 1/x, and the y-axis is 2ln(2) -/
theorem area_of_closed_figure : 
  let lower_bound : ℝ := 1/2
  let upper_bound : ℝ := 2
  let curve (x : ℝ) : ℝ := 1/x
  ∫ y in lower_bound..upper_bound, (1/y) = 2 * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_closed_figure_l876_87615


namespace NUMINAMATH_CALUDE_common_root_pairs_l876_87668

theorem common_root_pairs (n : ℕ) (hn : n > 1) :
  ∀ s t : ℤ, (∃ x : ℝ, x^n + s*x = 2007 ∧ x^n + t*x = 2008) ↔ 
  ((s = 2006 ∧ t = 2007) ∨ 
   (s = -2008 ∧ t = -2009 ∧ Even n) ∨ 
   (s = -2006 ∧ t = -2007 ∧ Odd n)) :=
by sorry

end NUMINAMATH_CALUDE_common_root_pairs_l876_87668


namespace NUMINAMATH_CALUDE_simplify_expression_l876_87601

theorem simplify_expression : (2^10 + 7^5) * (2^3 - (-2)^3)^8 = 76600653103936 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l876_87601


namespace NUMINAMATH_CALUDE_sum_common_terms_example_l876_87620

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℕ
  diff : ℕ
  last : ℕ

/-- Calculates the sum of common terms between two arithmetic sequences -/
def sumCommonTerms (seq1 seq2 : ArithmeticSequence) : ℕ :=
  sorry

theorem sum_common_terms_example :
  let seq1 : ArithmeticSequence := ⟨2, 4, 210⟩
  let seq2 : ArithmeticSequence := ⟨2, 6, 212⟩
  sumCommonTerms seq1 seq2 = 1872 :=
by sorry

end NUMINAMATH_CALUDE_sum_common_terms_example_l876_87620


namespace NUMINAMATH_CALUDE_binomial_8_choose_3_l876_87675

theorem binomial_8_choose_3 : Nat.choose 8 3 = 56 := by
  sorry

end NUMINAMATH_CALUDE_binomial_8_choose_3_l876_87675


namespace NUMINAMATH_CALUDE_combined_miles_per_gallon_l876_87651

/-- The combined miles per gallon of two cars given their individual efficiencies and distance ratio -/
theorem combined_miles_per_gallon
  (sam_mpg : ℝ)
  (alex_mpg : ℝ)
  (distance_ratio : ℚ)
  (h_sam_mpg : sam_mpg = 50)
  (h_alex_mpg : alex_mpg = 20)
  (h_distance_ratio : distance_ratio = 2 / 3) :
  (2 * distance_ratio + 3) / (2 * distance_ratio / sam_mpg + 3 / alex_mpg) = 500 / 19 := by
  sorry

end NUMINAMATH_CALUDE_combined_miles_per_gallon_l876_87651


namespace NUMINAMATH_CALUDE_douglas_vote_percentage_l876_87623

theorem douglas_vote_percentage (total_percentage : ℝ) (x_percentage : ℝ) (x_ratio : ℝ) (y_ratio : ℝ) :
  total_percentage = 0.54 →
  x_percentage = 0.62 →
  x_ratio = 3 →
  y_ratio = 2 →
  let total_ratio := x_ratio + y_ratio
  let y_percentage := (total_percentage * total_ratio - x_percentage * x_ratio) / y_ratio
  y_percentage = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_douglas_vote_percentage_l876_87623


namespace NUMINAMATH_CALUDE_coin_collection_problem_l876_87687

/-- Represents the types of coins in the collection --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- The value of each coin in cents --/
def coinValue : Coin → ℕ
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- A collection of coins --/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- The total number of coins in the collection --/
def CoinCollection.totalCoins (c : CoinCollection) : ℕ :=
  c.pennies + c.nickels + c.dimes + c.quarters

/-- The total value of the collection in cents --/
def CoinCollection.totalValue (c : CoinCollection) : ℕ :=
  c.pennies * coinValue Coin.Penny +
  c.nickels * coinValue Coin.Nickel +
  c.dimes * coinValue Coin.Dime +
  c.quarters * coinValue Coin.Quarter

theorem coin_collection_problem :
  ∀ c : CoinCollection,
    c.totalCoins = 10 ∧
    c.totalValue = 110 ∧
    c.pennies ≥ 1 ∧
    c.nickels ≥ 1 ∧
    c.dimes ≥ 1 ∧
    c.quarters ≥ 2
    →
    c.dimes = 5 :=
by sorry

end NUMINAMATH_CALUDE_coin_collection_problem_l876_87687


namespace NUMINAMATH_CALUDE_circumscribed_circle_radius_of_specific_trapezoid_l876_87613

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  a : ℝ  -- Length of the longer base
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumscribedCircleRadius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem: The radius of the circumscribed circle of the given isosceles trapezoid is 85/8 -/
theorem circumscribed_circle_radius_of_specific_trapezoid :
  let t : IsoscelesTrapezoid := { a := 21, b := 9, h := 8 }
  circumscribedCircleRadius t = 85 / 8 := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_circle_radius_of_specific_trapezoid_l876_87613


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_l876_87626

theorem simplify_sqrt_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_l876_87626


namespace NUMINAMATH_CALUDE_existence_of_rationals_l876_87678

theorem existence_of_rationals (a b c d m n : ℤ) (ε : ℝ) 
  (h : a * d - b * c ≠ 0) (hε : ε > 0) :
  ∃ x y : ℚ, 0 < |a * x + b * y - m| ∧ |a * x + b * y - m| < ε ∧
           0 < |c * x + d * y - n| ∧ |c * x + d * y - n| < ε :=
by sorry


end NUMINAMATH_CALUDE_existence_of_rationals_l876_87678


namespace NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l876_87653

theorem smallest_fourth_lucky_number :
  let first_three : List Nat := [68, 24, 85]
  let sum_first_three := first_three.sum
  let sum_digits_first_three := (first_three.map (fun n => n / 10 + n % 10)).sum
  ∀ x : Nat,
    x ≥ 10 ∧ x < 100 →
    (sum_first_three + x) * 1/4 = sum_digits_first_three + x / 10 + x % 10 →
    x ≥ 93 :=
by sorry

end NUMINAMATH_CALUDE_smallest_fourth_lucky_number_l876_87653


namespace NUMINAMATH_CALUDE_part_I_part_II_l876_87632

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine law for triangle ABC -/
def cosineLaw (t : Triangle) : Prop :=
  2 * t.b - t.c = 2 * t.a * Real.cos t.C

/-- Additional condition for part II -/
def additionalCondition (t : Triangle) : Prop :=
  4 * (t.b + t.c) = 3 * t.b * t.c

/-- Theorem for part I -/
theorem part_I (t : Triangle) (h : cosineLaw t) : t.A = 2 * Real.pi / 3 := by sorry

/-- Theorem for part II -/
theorem part_II (t : Triangle) (h1 : cosineLaw t) (h2 : additionalCondition t) (h3 : t.a = 2 * Real.sqrt 3) :
  (1/2) * t.b * t.c * Real.sin t.A = 4 * Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_part_I_part_II_l876_87632


namespace NUMINAMATH_CALUDE_normal_distribution_std_dev_l876_87648

/-- For a normal distribution with mean 10.5, if a value 2 standard deviations
    below the mean is 8.5, then the standard deviation is 1. -/
theorem normal_distribution_std_dev (μ σ : ℝ) (x : ℝ) : 
  μ = 10.5 → x = μ - 2 * σ → x = 8.5 → σ = 1 := by sorry

end NUMINAMATH_CALUDE_normal_distribution_std_dev_l876_87648


namespace NUMINAMATH_CALUDE_inequality_proof_l876_87698

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_one : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l876_87698


namespace NUMINAMATH_CALUDE_school_children_count_prove_school_children_count_l876_87697

theorem school_children_count : ℕ → Prop :=
  fun total_children =>
    let total_bananas := 2 * total_children
    total_bananas = 4 * (total_children - 350) →
    total_children = 700

-- Proof
theorem prove_school_children_count :
  ∃ (n : ℕ), school_children_count n :=
by
  sorry

end NUMINAMATH_CALUDE_school_children_count_prove_school_children_count_l876_87697
