import Mathlib

namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l244_24430

theorem imaginary_part_of_z (i : ℂ) (h : i * i = -1) :
  let z : ℂ := (3 - 4 * i) / i
  Complex.im z = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l244_24430


namespace NUMINAMATH_CALUDE_mn_minus_n_value_l244_24464

theorem mn_minus_n_value (m n : ℝ) (h1 : |m| = 4) (h2 : |n| = 5/2) (h3 : m * n < 0) : 
  m * n - n = -7.5 ∨ m * n - n = -12.5 := by
sorry

end NUMINAMATH_CALUDE_mn_minus_n_value_l244_24464


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l244_24419

def arithmetic_sequence (a : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → a (i + 1) - a i = a (j + 1) - a j

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) :
  arithmetic_sequence a n →
  a 0 = 3 →
  a 1 = 8 →
  a 2 = 13 →
  a (n - 1) = 38 →
  a (n - 2) + a (n - 3) = 61 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l244_24419


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l244_24423

/-- The sum of the coordinates of the midpoint of a segment with endpoints (8, -3) and (-4, 15) is 8 -/
theorem midpoint_coordinate_sum : 
  let x₁ : ℝ := 8
  let y₁ : ℝ := -3
  let x₂ : ℝ := -4
  let y₂ : ℝ := 15
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y = 8 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l244_24423


namespace NUMINAMATH_CALUDE_playground_area_is_22500_l244_24465

/-- Represents a rectangular playground --/
structure Playground where
  width : ℝ
  length : ℝ

/-- Properties of the playground --/
def PlaygroundProperties (p : Playground) : Prop :=
  p.length = 2 * p.width + 25 ∧
  2 * (p.length + p.width) = 650

/-- The area of the playground --/
def playgroundArea (p : Playground) : ℝ :=
  p.length * p.width

/-- Theorem: The area of the playground with given properties is 22,500 square feet --/
theorem playground_area_is_22500 :
  ∀ p : Playground, PlaygroundProperties p → playgroundArea p = 22500 := by
  sorry

end NUMINAMATH_CALUDE_playground_area_is_22500_l244_24465


namespace NUMINAMATH_CALUDE_puzzle_solving_time_l244_24405

/-- The total time spent solving puzzles given a warm-up puzzle and two longer puzzles -/
theorem puzzle_solving_time (warm_up_time : ℕ) (num_long_puzzles : ℕ) (long_puzzle_factor : ℕ) : 
  warm_up_time = 10 → 
  num_long_puzzles = 2 → 
  long_puzzle_factor = 3 → 
  warm_up_time + num_long_puzzles * (long_puzzle_factor * warm_up_time) = 70 :=
by sorry

end NUMINAMATH_CALUDE_puzzle_solving_time_l244_24405


namespace NUMINAMATH_CALUDE_towels_given_to_mother_l244_24487

theorem towels_given_to_mother (green_towels white_towels remaining_towels : ℕ) : 
  green_towels = 35 → 
  white_towels = 21 → 
  remaining_towels = 22 → 
  green_towels + white_towels - remaining_towels = 34 :=
by
  sorry

end NUMINAMATH_CALUDE_towels_given_to_mother_l244_24487


namespace NUMINAMATH_CALUDE_direction_vector_b_value_l244_24477

/-- Given a line passing through points (-1, 3) and (2, 7) with direction vector (2, b), prove that b = 8/3 -/
theorem direction_vector_b_value (b : ℚ) : 
  let p1 : ℚ × ℚ := (-1, 3)
  let p2 : ℚ × ℚ := (2, 7)
  let direction_vector : ℚ × ℚ := (2, b)
  (∃ (k : ℚ), k • (p2.1 - p1.1, p2.2 - p1.2) = direction_vector) →
  b = 8/3 := by
sorry

end NUMINAMATH_CALUDE_direction_vector_b_value_l244_24477


namespace NUMINAMATH_CALUDE_f_monotone_and_inequality_l244_24460

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * Real.log x - a * x + a

theorem f_monotone_and_inequality (a : ℝ) :
  (a > 0 ∧ a ≤ 2) ↔
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f a x < f a y) ∧
  (∀ x : ℝ, x > 0 → (x - 1) * f a x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_monotone_and_inequality_l244_24460


namespace NUMINAMATH_CALUDE_last_two_digits_of_1032_power_1032_l244_24442

theorem last_two_digits_of_1032_power_1032 : ∃ k : ℕ, 1032^1032 ≡ 76 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_1032_power_1032_l244_24442


namespace NUMINAMATH_CALUDE_sum_of_products_of_roots_l244_24470

theorem sum_of_products_of_roots (p q r : ℂ) : 
  (2 * p^3 + p^2 - 7*p + 2 = 0) → 
  (2 * q^3 + q^2 - 7*q + 2 = 0) → 
  (2 * r^3 + r^2 - 7*r + 2 = 0) → 
  p * q + q * r + r * p = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_products_of_roots_l244_24470


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l244_24449

def A : Set ℝ := {Real.sin (90 * Real.pi / 180), Real.cos (180 * Real.pi / 180)}
def B : Set ℝ := {x : ℝ | x^2 + x = 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l244_24449


namespace NUMINAMATH_CALUDE_circle_area_ratio_l244_24433

/-- Given two circles r and s, if the diameter of r is 50% of the diameter of s,
    then the area of r is 25% of the area of s. -/
theorem circle_area_ratio (r s : Real) (hr : r > 0) (hs : s > 0) 
  (h_diameter : 2 * r = 0.5 * (2 * s)) : 
  π * r^2 = 0.25 * (π * s^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l244_24433


namespace NUMINAMATH_CALUDE_car_speed_second_hour_l244_24480

/-- Given a car's speed in the first hour and its average speed over two hours,
    calculate the speed in the second hour. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 98)
  (h2 : average_speed = 79) :
  let speed_second_hour := 2 * average_speed - speed_first_hour
  speed_second_hour = 60 := by
  sorry

end NUMINAMATH_CALUDE_car_speed_second_hour_l244_24480


namespace NUMINAMATH_CALUDE_max_sum_hexagonal_prism_with_pyramid_l244_24407

/-- Represents a three-dimensional geometric shape -/
structure Shape3D where
  faces : ℕ
  vertices : ℕ
  edges : ℕ

/-- A hexagonal prism -/
def hexagonal_prism : Shape3D :=
  { faces := 8, vertices := 12, edges := 18 }

/-- Adds a pyramid to one face of a given shape -/
def add_pyramid (s : Shape3D) : Shape3D :=
  { faces := s.faces + 5,  -- Loses 1 face, gains 6
    vertices := s.vertices + 1,
    edges := s.edges + 6 }

/-- Calculates the sum of faces, vertices, and edges -/
def shape_sum (s : Shape3D) : ℕ :=
  s.faces + s.vertices + s.edges

/-- Theorem: The maximum sum of faces, vertices, and edges after adding a pyramid to a hexagonal prism is 44 -/
theorem max_sum_hexagonal_prism_with_pyramid :
  shape_sum (add_pyramid hexagonal_prism) = 44 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_hexagonal_prism_with_pyramid_l244_24407


namespace NUMINAMATH_CALUDE_number_operations_l244_24496

theorem number_operations (x : ℚ) : (x - 5) / 7 = 7 → (x - 2) / 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_number_operations_l244_24496


namespace NUMINAMATH_CALUDE_infinite_sum_equals_9_320_l244_24469

/-- The sum of the infinite series n / (n^4 + 16) from n=1 to infinity equals 9/320 -/
theorem infinite_sum_equals_9_320 :
  (∑' n : ℕ, n / (n^4 + 16 : ℝ)) = 9 / 320 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_equals_9_320_l244_24469


namespace NUMINAMATH_CALUDE_inscribed_semicircle_radius_in_reflected_triangle_l244_24425

/-- Represents a rectangle formed by reflecting an isosceles triangle over its base -/
structure ReflectedTriangleRectangle where
  base : ℝ
  height : ℝ
  inscribed_semicircle_radius : ℝ

/-- The theorem stating the radius of the inscribed semicircle in the specific rectangle -/
theorem inscribed_semicircle_radius_in_reflected_triangle
  (rect : ReflectedTriangleRectangle)
  (h_base : rect.base = 24)
  (h_height : rect.height = 10) :
  rect.inscribed_semicircle_radius = 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_semicircle_radius_in_reflected_triangle_l244_24425


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_eight_pi_thirds_l244_24488

noncomputable def f (x : Real) : Real := Real.sqrt 3 * Real.sin x + Real.cos x

theorem sum_of_roots_equals_eight_pi_thirds (a : Real) :
  0 < a → a < 1 → ∃ x₁ x₂ : Real, 
    x₁ ∈ Set.Icc 0 (2 * Real.pi) ∧ 
    x₂ ∈ Set.Icc 0 (2 * Real.pi) ∧ 
    f x₁ = a ∧ 
    f x₂ = a ∧ 
    x₁ + x₂ = 8 * Real.pi / 3 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_eight_pi_thirds_l244_24488


namespace NUMINAMATH_CALUDE_minimize_theta_l244_24413

def angle : ℝ := -495

theorem minimize_theta : 
  ∃ (K : ℤ) (θ : ℝ), 
    angle = K * 360 + θ ∧ 
    ∀ (K' : ℤ) (θ' : ℝ), angle = K' * 360 + θ' → |θ| ≤ |θ'| ∧
    θ = -135 := by
  sorry

end NUMINAMATH_CALUDE_minimize_theta_l244_24413


namespace NUMINAMATH_CALUDE_paul_failed_by_10_marks_l244_24471

/-- Calculates the number of marks a student failed by in an exam -/
def marksFailed (maxMarks passingPercentage gotMarks : ℕ) : ℕ :=
  let passingMarks := (passingPercentage * maxMarks) / 100
  if gotMarks ≥ passingMarks then 0 else passingMarks - gotMarks

/-- Theorem stating that Paul failed by 10 marks -/
theorem paul_failed_by_10_marks :
  marksFailed 120 50 50 = 10 := by
  sorry

end NUMINAMATH_CALUDE_paul_failed_by_10_marks_l244_24471


namespace NUMINAMATH_CALUDE_triangle_equilateral_iff_area_condition_l244_24474

/-- Triangle with vertices A₁, A₂, A₃ -/
structure Triangle where
  A₁ : ℝ × ℝ
  A₂ : ℝ × ℝ
  A₃ : ℝ × ℝ

/-- Altitude of a triangle from a vertex to the opposite side -/
def altitude (T : Triangle) (i : Fin 3) : ℝ := sorry

/-- Area of a triangle -/
def area (T : Triangle) : ℝ := sorry

/-- Length of a side of a triangle -/
def sideLength (T : Triangle) (i j : Fin 3) : ℝ := sorry

/-- A triangle is equilateral if all its sides have equal length -/
def isEquilateral (T : Triangle) : Prop :=
  sideLength T 0 1 = sideLength T 1 2 ∧ sideLength T 1 2 = sideLength T 2 0

/-- Main theorem: A triangle is equilateral iff its area satisfies the given condition -/
theorem triangle_equilateral_iff_area_condition (T : Triangle) :
  isEquilateral T ↔ 
    area T = (1/6) * (sideLength T 0 1 * altitude T 0 + 
                      sideLength T 1 2 * altitude T 1 + 
                      sideLength T 2 0 * altitude T 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_equilateral_iff_area_condition_l244_24474


namespace NUMINAMATH_CALUDE_sector_area_l244_24463

theorem sector_area (θ : Real) (L : Real) (A : Real) : 
  θ = π / 6 → 
  L = 2 * π / 3 → 
  A = 4 * π / 3 → 
  ∃ (r : Real), 
    L = r * θ ∧ 
    A = 1 / 2 * r^2 * θ := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l244_24463


namespace NUMINAMATH_CALUDE_evaluate_expression_l244_24476

theorem evaluate_expression : 
  Real.sqrt ((16^6 + 8^8) / (16^3 + 8^9)) = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l244_24476


namespace NUMINAMATH_CALUDE_remaining_apples_l244_24468

def initial_apples : ℕ := 127
def given_apples : ℕ := 88

theorem remaining_apples : initial_apples - given_apples = 39 := by
  sorry

end NUMINAMATH_CALUDE_remaining_apples_l244_24468


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l244_24428

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 3) = 7 → x = 46 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l244_24428


namespace NUMINAMATH_CALUDE_attendance_theorem_l244_24418

/-- Represents the admission prices and attendance for a play -/
structure PlayAttendance where
  adult_price : ℕ
  child_price : ℕ
  total_receipts : ℕ
  num_children : ℕ

/-- Calculates the total number of attendees given the play attendance data -/
def total_attendees (p : PlayAttendance) : ℕ :=
  p.num_children + (p.total_receipts - p.num_children * p.child_price) / p.adult_price

/-- Theorem stating that given the specific conditions, the total number of attendees is 610 -/
theorem attendance_theorem (p : PlayAttendance) 
    (h1 : p.adult_price = 2)
    (h2 : p.child_price = 1)
    (h3 : p.total_receipts = 960)
    (h4 : p.num_children = 260) : 
  total_attendees p = 610 := by
  sorry

#eval total_attendees ⟨2, 1, 960, 260⟩

end NUMINAMATH_CALUDE_attendance_theorem_l244_24418


namespace NUMINAMATH_CALUDE_set_inclusion_iff_a_range_l244_24454

/-- The set A -/
def A : Set ℝ := {x | -2 < x ∧ x < 3}

/-- The set B -/
def B : Set ℝ := {x | x^2 + 2*x - 8 > 0}

/-- The set C parameterized by a -/
def C (a : ℝ) : Set ℝ := {x | x^2 - 4*a*x + 3*a^2 < 0}

/-- The main theorem stating the equivalence between C being a subset of (A ∩ ℝ\B) and the range of a -/
theorem set_inclusion_iff_a_range :
  ∀ a : ℝ, (C a ⊆ (A ∩ (Set.univ \ B))) ↔ (0 < a ∧ a ≤ 2/3) :=
sorry

end NUMINAMATH_CALUDE_set_inclusion_iff_a_range_l244_24454


namespace NUMINAMATH_CALUDE_common_ratio_is_two_l244_24479

/-- An increasing geometric sequence with specific conditions -/
def IncreasingGeometricSequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, q > 1 ∧ ∀ n : ℕ, a (n + 1) = a n * q) ∧
  a 2 = 2 ∧
  a 4 - a 3 = 4

/-- The common ratio of the sequence is 2 -/
theorem common_ratio_is_two (a : ℕ → ℝ) (h : IncreasingGeometricSequence a) :
    ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = a n * q) ∧ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_common_ratio_is_two_l244_24479


namespace NUMINAMATH_CALUDE_min_jumps_to_cover_race_l244_24453

/-- Represents the possible jump distances of the cricket -/
inductive JumpDistance where
  | short : JumpDistance -- 8 meters
  | long : JumpDistance  -- 9 meters

/-- The race distance in meters -/
def raceDistance : ℕ := 100

/-- Calculates the total distance covered by a sequence of jumps -/
def totalDistance (jumps : List JumpDistance) : ℕ :=
  jumps.foldl (fun acc jump => acc + match jump with
    | JumpDistance.short => 8
    | JumpDistance.long => 9) 0

/-- Checks if a sequence of jumps exactly covers the race distance -/
def isValidJumpSequence (jumps : List JumpDistance) : Prop :=
  totalDistance jumps = raceDistance

/-- The main theorem to be proved -/
theorem min_jumps_to_cover_race :
  ∃ (jumps : List JumpDistance),
    isValidJumpSequence jumps ∧
    jumps.length = 12 ∧
    ∀ (other_jumps : List JumpDistance),
      isValidJumpSequence other_jumps →
      other_jumps.length ≥ 12 :=
by sorry

end NUMINAMATH_CALUDE_min_jumps_to_cover_race_l244_24453


namespace NUMINAMATH_CALUDE_min_value_expression_l244_24414

theorem min_value_expression (a b c : ℝ) (h1 : c > 0) (h2 : a ≠ 0) (h3 : b ≠ 0)
  (h4 : 4 * a^2 - 2 * a * b + 4 * b^2 - c = 0)
  (h5 : ∀ (x y : ℝ), x ≠ 0 → y ≠ 0 → 4 * x^2 - 2 * x * y + 4 * y^2 - c = 0 →
    |2 * a + b| ≥ |2 * x + y|) :
  ∃ (m : ℝ), m = -2 ∧ ∀ (x y z : ℝ), x ≠ 0 → y ≠ 0 → z > 0 →
    4 * x^2 - 2 * x * y + 4 * y^2 - z = 0 →
    3 / x - 4 / y + 5 / z ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l244_24414


namespace NUMINAMATH_CALUDE_sin_cos_sum_21_39_l244_24455

theorem sin_cos_sum_21_39 : 
  Real.sin (21 * π / 180) * Real.cos (39 * π / 180) + 
  Real.cos (21 * π / 180) * Real.sin (39 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_21_39_l244_24455


namespace NUMINAMATH_CALUDE_vector_magnitude_l244_24473

theorem vector_magnitude (a b : ℝ × ℝ) (m : ℝ) :
  a = (2, 1) →
  b = (3, m) →
  (∃ k : ℝ, (2 • a - b) = k • b) →
  ‖b‖ = (3 * Real.sqrt 5) / 2 :=
by sorry

end NUMINAMATH_CALUDE_vector_magnitude_l244_24473


namespace NUMINAMATH_CALUDE_complete_square_equivalence_l244_24422

/-- Given a quadratic equation x^2 - 4x = 5, prove that it is equivalent to (x-2)^2 = 9 when completed square. -/
theorem complete_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x = 5 ↔ (x-2)^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_equivalence_l244_24422


namespace NUMINAMATH_CALUDE_blood_expiry_time_l244_24467

/-- Represents time as hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Represents a date -/
structure Date where
  month : ℕ
  day : ℕ
  year : ℕ

/-- Represents a datetime -/
structure DateTime where
  date : Date
  time : Time

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def addSeconds (dt : DateTime) (seconds : ℕ) : DateTime :=
  sorry -- Implementation not required for the statement

theorem blood_expiry_time 
  (donation_time : DateTime)
  (expiry_seconds : ℕ)
  (h_donation_time : donation_time = ⟨⟨1, 1, 2023⟩, ⟨8, 0, sorry, sorry⟩⟩)
  (h_expiry_seconds : expiry_seconds = factorial 8) :
  addSeconds donation_time expiry_seconds = ⟨⟨1, 1, 2023⟩, ⟨19, 12, sorry, sorry⟩⟩ :=
sorry

end NUMINAMATH_CALUDE_blood_expiry_time_l244_24467


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_six_l244_24486

theorem largest_three_digit_multiple_of_six :
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 6 = 0 → n ≤ 996 :=
sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_six_l244_24486


namespace NUMINAMATH_CALUDE_ellipse_max_distance_sum_l244_24434

/-- Given an ellipse with equation x^2/4 + y^2/3 = 1 and foci F₁ and F₂,
    where a line l passing through F₁ intersects the ellipse at points A and B,
    the maximum value of |BF₂| + |AF₂| is 5. -/
theorem ellipse_max_distance_sum (F₁ F₂ A B : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  (∀ x y, x^2/4 + y^2/3 = 1 → (x, y) ∈ l → (x, y) = A ∨ (x, y) = B) →
  F₁ ∈ l →
  F₁.1 < F₂.1 →
  (∀ x y, x^2/4 + y^2/3 = 1 → dist (x, y) F₁ + dist (x, y) F₂ = 4) →
  dist B F₂ + dist A F₂ ≤ 5 :=
sorry


end NUMINAMATH_CALUDE_ellipse_max_distance_sum_l244_24434


namespace NUMINAMATH_CALUDE_rectangle_length_l244_24481

/-- Proves that a rectangle with given perimeter-to-breadth ratio and area has a specific length -/
theorem rectangle_length (P b l A : ℝ) : 
  P / b = 5 → 
  P = 2 * (l + b) → 
  A = l * b → 
  A = 216 → 
  l = 18 := by sorry

end NUMINAMATH_CALUDE_rectangle_length_l244_24481


namespace NUMINAMATH_CALUDE_stairs_ratio_l244_24441

theorem stairs_ratio (samir veronica : ℕ) (total : ℕ) (h1 : samir = 318) (h2 : total = 495) (h3 : samir + veronica = total) :
  (veronica : ℚ) / (samir / 2 : ℚ) = (total - samir : ℚ) / (samir / 2 : ℚ) := by
sorry

end NUMINAMATH_CALUDE_stairs_ratio_l244_24441


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l244_24409

theorem cos_squared_minus_sin_squared_15_deg :
  Real.cos (15 * π / 180) ^ 2 - Real.sin (15 * π / 180) ^ 2 = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_15_deg_l244_24409


namespace NUMINAMATH_CALUDE_triangle_areas_sum_l244_24448

/-- Given a rectangle and two triangles with specific properties, prove that the combined area of the triangles is 108 cm² -/
theorem triangle_areas_sum (rectangle_length rectangle_width : ℝ)
  (triangle1_area_factor : ℝ)
  (triangle2_base triangle2_base_height_sum : ℝ)
  (h_rectangle_length : rectangle_length = 6)
  (h_rectangle_width : rectangle_width = 4)
  (h_rectangle_triangle1_ratio : (rectangle_length * rectangle_width) / (5 * triangle1_area_factor) = 2 / 5)
  (h_triangle2_base : triangle2_base = 8)
  (h_triangle2_sum : triangle2_base + (triangle2_base_height_sum - triangle2_base) = 20)
  (h_triangle_ratio : (triangle2_base * (triangle2_base_height_sum - triangle2_base)) / (10 * triangle1_area_factor) = 3 / 5) :
  5 * triangle1_area_factor + (triangle2_base * (triangle2_base_height_sum - triangle2_base)) / 2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_triangle_areas_sum_l244_24448


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l244_24421

-- Define the quadratic function
def f (x : ℝ) := x^2 + 3*x - 4

-- Define the solution set
def solution_set : Set ℝ := {x | -4 < x ∧ x < 1}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | f x < 0} = solution_set := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l244_24421


namespace NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l244_24478

theorem surface_area_ratio_of_cubes (a b : ℝ) (h : a / b = 7) :
  (6 * a^2) / (6 * b^2) = 49 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_ratio_of_cubes_l244_24478


namespace NUMINAMATH_CALUDE_intersection_point_l244_24456

theorem intersection_point (a : ℝ) :
  (∃! p : ℝ × ℝ, (p.2 = a * p.1 + a ∧ p.2 = p.1 ∧ p.2 = 2 - 2 * a * p.1)) ↔ (a = 1/2 ∨ a = -2) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l244_24456


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l244_24427

theorem quadratic_equation_solution_sum : ∃ (c d : ℝ), 
  (c^2 - 6*c + 15 = 25) ∧ 
  (d^2 - 6*d + 15 = 25) ∧ 
  (c ≥ d) ∧ 
  (3*c + 2*d = 15 + Real.sqrt 19) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_sum_l244_24427


namespace NUMINAMATH_CALUDE_simplify_algebraic_expression_l244_24491

theorem simplify_algebraic_expression (a b : ℝ) : -3 * a^2 * b + 2 * a^2 * b = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_simplify_algebraic_expression_l244_24491


namespace NUMINAMATH_CALUDE_neighboring_cells_difference_l244_24452

/-- A type representing a cell in an n × n grid --/
structure Cell (n : ℕ) where
  row : Fin n
  col : Fin n

/-- A function that assigns values to cells in the grid --/
def GridAssignment (n : ℕ) := Cell n → Fin (n^2)

/-- Two cells are neighbors if they share at least one point --/
def IsNeighbor {n : ℕ} (c1 c2 : Cell n) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col.val = c2.col.val + 1)

/-- The main theorem to be proved --/
theorem neighboring_cells_difference {n : ℕ} (h : n > 1) (g : GridAssignment n) :
  ∃ (c1 c2 : Cell n), IsNeighbor c1 c2 ∧ 
    (g c1).val ≥ (g c2).val + n + 1 ∨ (g c2).val ≥ (g c1).val + n + 1 :=
sorry

end NUMINAMATH_CALUDE_neighboring_cells_difference_l244_24452


namespace NUMINAMATH_CALUDE_largest_common_divisor_l244_24439

theorem largest_common_divisor : 
  ∃ (n : ℕ), n = 35 ∧ 
  n ∣ 420 ∧ n ∣ 385 ∧ 
  ∀ (m : ℕ), m ∣ 420 ∧ m ∣ 385 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l244_24439


namespace NUMINAMATH_CALUDE_distance_between_five_and_six_l244_24451

/-- The distance to the nearest town in miles -/
def d : ℝ := sorry

/-- Alice's statement is false -/
axiom alice_false : ¬(d ≥ 6)

/-- Bob's statement is false -/
axiom bob_false : ¬(d ≤ 5)

/-- Charlie's statement is false -/
axiom charlie_false : ¬(d ≤ 4)

/-- Theorem: The distance to the nearest town is between 5 and 6 miles -/
theorem distance_between_five_and_six : 5 < d ∧ d < 6 := by sorry

end NUMINAMATH_CALUDE_distance_between_five_and_six_l244_24451


namespace NUMINAMATH_CALUDE_function_properties_l244_24497

/-- Given a function f and a real number a, prove properties of f --/
theorem function_properties (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f (2 * x + 1) = 3 * a^(x + 1) - 4) 
  (h2 : a > 0) 
  (h3 : a ≠ 1) :
  (∀ x, f x = 3 * a^((x + 1) / 2) - 4) ∧ 
  (f (-1) = -1) ∧
  (a > 1 → ∀ x, f (x - 3/4) ≥ 3 / a^(x^2 / 2) - 4) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l244_24497


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l244_24483

theorem chicken_rabbit_problem :
  ∀ (chickens rabbits : ℕ),
    chickens + rabbits = 100 →
    2 * chickens = 4 * rabbits + 26 →
    chickens = 71 :=
by
  sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l244_24483


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l244_24432

/-- The maximum marks for an exam where:
  1. The passing mark is 33% of the maximum marks
  2. A student who got 175 marks failed by 56 marks
-/
theorem exam_maximum_marks : ∃ (M : ℕ), 
  (M * 33 / 100 = 175 + 56) ∧ 
  M = 700 := by
  sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l244_24432


namespace NUMINAMATH_CALUDE_product_of_real_parts_is_two_l244_24436

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the quadratic equation
def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 3*z = -7 + 2*i

-- Theorem statement
theorem product_of_real_parts_is_two :
  ∃ (z₁ z₂ : ℂ), quadratic_equation z₁ ∧ quadratic_equation z₂ ∧
  z₁ ≠ z₂ ∧ (z₁.re * z₂.re = 2) :=
sorry

end NUMINAMATH_CALUDE_product_of_real_parts_is_two_l244_24436


namespace NUMINAMATH_CALUDE_f_inequality_range_l244_24494

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality_range (x : ℝ) : 
  f (2 * x) > f (x - 1) ↔ x < -1 ∨ x > 1/3 :=
sorry

end NUMINAMATH_CALUDE_f_inequality_range_l244_24494


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l244_24424

theorem right_triangle_shorter_leg (a b c m : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ m > 0 →  -- Positive lengths
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  b = 2 * a →  -- One leg is twice the other
  m = 15 →  -- Median to hypotenuse is 15
  m^2 = (c^2) / 4 + (a^2 + b^2) / 4 →  -- Median formula
  a = 6 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l244_24424


namespace NUMINAMATH_CALUDE_sequence_equality_l244_24475

/-- Given a sequence a₀, a₁, a₂, ..., prove that aₙ = 10ⁿ for all natural numbers n,
    if the following equation holds for all real t:
    ∑_{n=0}^∞ aₙ * t^n / n! = (∑_{n=0}^∞ 2^n * t^n / n!)² * (∑_{n=0}^∞ 3^n * t^n / n!)² -/
theorem sequence_equality (a : ℕ → ℝ) :
  (∀ t : ℝ, ∑' n, a n * t^n / n.factorial = (∑' n, 2^n * t^n / n.factorial)^2 * (∑' n, 3^n * t^n / n.factorial)^2) →
  ∀ n : ℕ, a n = 10^n :=
by sorry

end NUMINAMATH_CALUDE_sequence_equality_l244_24475


namespace NUMINAMATH_CALUDE_complex_equation_real_part_l244_24489

-- Define complex number z as a + bi
def z (a b : ℝ) : ℂ := Complex.mk a b

-- State the theorem
theorem complex_equation_real_part 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : z a b ^ 3 + 2 * z a b ^ 2 * Complex.I - 2 * z a b * Complex.I - 8 = 1624 * Complex.I) : 
  a ^ 3 - 3 * a * b ^ 2 - 8 = 0 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_real_part_l244_24489


namespace NUMINAMATH_CALUDE_rectangular_to_polar_l244_24458

theorem rectangular_to_polar :
  let x : ℝ := 3
  let y : ℝ := 3 * Real.sqrt 3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := Real.arctan (y / x)
  r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧ r = 6 ∧ θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_polar_l244_24458


namespace NUMINAMATH_CALUDE_james_candy_packs_l244_24435

/-- Given the initial amount, change received, and cost per pack of candy,
    calculate the number of packs of candy bought. -/
def candyPacks (initialAmount change costPerPack : ℕ) : ℕ :=
  (initialAmount - change) / costPerPack

/-- Theorem stating that James bought 3 packs of candy -/
theorem james_candy_packs :
  candyPacks 20 11 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_james_candy_packs_l244_24435


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l244_24472

theorem algebraic_expression_value (a b : ℝ) (h : a - b + 3 = 0) :
  2 - 3*a + 3*b = 11 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l244_24472


namespace NUMINAMATH_CALUDE_three_digit_number_operation_l244_24461

theorem three_digit_number_operation (a b c : ℕ) : 
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9 ∧ c ≥ 0 ∧ c ≤ 9 →  -- Ensures it's a three-digit number
  a = 2*c - 3 →  -- Hundreds digit is 3 less than twice the units digit
  ((100*a + 10*b + c) - ((100*c + 10*b + a) + 50)) % 10 = 3 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_number_operation_l244_24461


namespace NUMINAMATH_CALUDE_problem_solution_l244_24431

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < c then c * x + 1
  else if c ≤ x ∧ x < 1 then 2^(x / c^2) + 1
  else 0

theorem problem_solution (c : ℝ) :
  (0 < c ∧ c < 1) →
  (f c (c^2) = 9/8) →
  (c = 1/2) ∧
  (∀ x : ℝ, f (1/2) x > Real.sqrt 2 / 8 + 1 ↔ Real.sqrt 2 / 4 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l244_24431


namespace NUMINAMATH_CALUDE_remaining_money_after_ticket_l244_24410

def octal_to_decimal (n : ℕ) : ℕ := sorry

theorem remaining_money_after_ticket : 
  let savings := octal_to_decimal 5376
  let ticket_cost := 1200
  savings - ticket_cost = 1614 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_after_ticket_l244_24410


namespace NUMINAMATH_CALUDE_batsman_new_average_is_35_l244_24411

/-- Represents a batsman's score history -/
structure Batsman where
  previousInnings : Nat
  previousTotalScore : Nat
  newInningScore : Nat
  averageIncrease : Nat

/-- Calculates the new average after the latest inning -/
def newAverage (b : Batsman) : Nat :=
  (b.previousTotalScore + b.newInningScore) / (b.previousInnings + 1)

/-- Theorem: Given the conditions, prove that the new average is 35 -/
theorem batsman_new_average_is_35 (b : Batsman)
  (h1 : b.previousInnings = 10)
  (h2 : b.newInningScore = 85)
  (h3 : b.averageIncrease = 5)
  (h4 : newAverage b = (b.previousTotalScore / b.previousInnings) + b.averageIncrease) :
  newAverage b = 35 := by
  sorry

#eval newAverage { previousInnings := 10, previousTotalScore := 300, newInningScore := 85, averageIncrease := 5 }

end NUMINAMATH_CALUDE_batsman_new_average_is_35_l244_24411


namespace NUMINAMATH_CALUDE_percentage_fraction_difference_l244_24446

theorem percentage_fraction_difference : (75 / 100 * 40) - (4 / 5 * 25) = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_fraction_difference_l244_24446


namespace NUMINAMATH_CALUDE_tea_leaves_problem_l244_24447

theorem tea_leaves_problem (num_plants : ℕ) (initial_leaves : ℕ) (fall_fraction : ℚ) : 
  num_plants = 3 → 
  initial_leaves = 18 → 
  fall_fraction = 1/3 → 
  (num_plants * initial_leaves * (1 - fall_fraction) : ℚ) = 36 := by
  sorry

end NUMINAMATH_CALUDE_tea_leaves_problem_l244_24447


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l244_24438

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) : 
  (1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + x * y * z ≥ 2 :=
by sorry

theorem min_value_achieved (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) : 
  (1 / ((1 - x) * (1 - y) * (1 - z))) + (1 / ((1 + x) * (1 + y) * (1 + z))) + x * y * z = 2 ↔ 
  x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achieved_l244_24438


namespace NUMINAMATH_CALUDE_juggling_contest_winner_l244_24403

/-- Represents the number of rotations for an object over 4 minutes -/
structure Rotations :=
  (minute1 : ℕ) (minute2 : ℕ) (minute3 : ℕ) (minute4 : ℕ)

/-- Calculates the total rotations for a contestant -/
def totalRotations (obj1Count : ℕ) (obj1Rotations : Rotations) 
                   (obj2Count : ℕ) (obj2Rotations : Rotations) : ℕ :=
  obj1Count * (obj1Rotations.minute1 + obj1Rotations.minute2 + obj1Rotations.minute3 + obj1Rotations.minute4) +
  obj2Count * (obj2Rotations.minute1 + obj2Rotations.minute2 + obj2Rotations.minute3 + obj2Rotations.minute4)

theorem juggling_contest_winner (tobyBaseballs : Rotations) (tobyFrisbees : Rotations)
                                (annaApples : Rotations) (annaOranges : Rotations)
                                (jackTennisBalls : Rotations) (jackWaterBalloons : Rotations) :
  tobyBaseballs = ⟨80, 85, 75, 90⟩ →
  tobyFrisbees = ⟨60, 70, 65, 80⟩ →
  annaApples = ⟨101, 99, 98, 102⟩ →
  annaOranges = ⟨95, 90, 92, 93⟩ →
  jackTennisBalls = ⟨82, 81, 85, 87⟩ →
  jackWaterBalloons = ⟨100, 96, 101, 97⟩ →
  (max (totalRotations 5 tobyBaseballs 3 tobyFrisbees)
       (max (totalRotations 4 annaApples 5 annaOranges)
            (totalRotations 6 jackTennisBalls 4 jackWaterBalloons))) = 3586 := by
  sorry

end NUMINAMATH_CALUDE_juggling_contest_winner_l244_24403


namespace NUMINAMATH_CALUDE_last_digit_periodic_l244_24440

theorem last_digit_periodic (n : ℕ) : n^n % 10 = (n + 20)^(n + 20) % 10 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_periodic_l244_24440


namespace NUMINAMATH_CALUDE_power_of_power_l244_24444

theorem power_of_power (a : ℝ) : (a^3)^2 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l244_24444


namespace NUMINAMATH_CALUDE_function_f_theorem_l244_24462

/-- A function f: ℝ → ℝ satisfying the given conditions -/
def FunctionF (f : ℝ → ℝ) : Prop :=
  (∃ (S : Finset ℝ), ∀ x ≠ 0, ∃ c ∈ S, f x = c * x) ∧
  (∀ x, f (x - 1 - f x) = f x - 1 - x)

/-- The theorem stating that f(x) = x or f(x) = -x -/
theorem function_f_theorem (f : ℝ → ℝ) (h : FunctionF f) :
  (∀ x, f x = x) ∨ (∀ x, f x = -x) := by
  sorry

end NUMINAMATH_CALUDE_function_f_theorem_l244_24462


namespace NUMINAMATH_CALUDE_square_root_of_four_l244_24499

theorem square_root_of_four (x : ℝ) : x^2 = 4 ↔ x = 2 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_four_l244_24499


namespace NUMINAMATH_CALUDE_series_sum_equals_399002_l244_24426

/-- The sum of the series 1-2-3+4+5-6-7+8+9-10-11+12+13-...-1994-1995+1996+1997 -/
def seriesSum : ℕ → ℤ
  | 0 => 0
  | n + 1 => seriesSum n + term (n + 1)
where
  term : ℕ → ℤ
  | n => if n % 5 ≤ 2 then -(n : ℤ) else (n : ℤ)

theorem series_sum_equals_399002 : seriesSum 1997 = 399002 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_399002_l244_24426


namespace NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l244_24482

/-- Represents the number of flies eaten per day in a swamp ecosystem -/
def flies_eaten_per_day (
  frog_flies : ℕ)  -- flies eaten by one frog per day
  (fish_frogs : ℕ)  -- frogs eaten by one fish per day
  (gharial_fish : ℕ)  -- fish eaten by one gharial per day
  (heron_frogs : ℕ)  -- frogs eaten by one heron per day
  (heron_fish : ℕ)  -- fish eaten by one heron per day
  (caiman_gharials : ℕ)  -- gharials eaten by one caiman per day
  (caiman_herons : ℕ)  -- herons eaten by one caiman per day
  (num_gharials : ℕ)  -- number of gharials in the swamp
  (num_herons : ℕ)  -- number of herons in the swamp
  (num_caimans : ℕ)  -- number of caimans in the swamp
  : ℕ :=
  sorry

/-- Theorem stating the number of flies eaten per day in the given swamp ecosystem -/
theorem swamp_ecosystem_flies_eaten :
  flies_eaten_per_day 30 8 15 5 3 2 2 9 12 7 = 42840 :=
by sorry

end NUMINAMATH_CALUDE_swamp_ecosystem_flies_eaten_l244_24482


namespace NUMINAMATH_CALUDE_expression_factorization_l244_24404

theorem expression_factorization (x : ℝ) : 
  (16 * x^6 - 36 * x^4) - (4 * x^6 - 9 * x^4 + 12) = 3 * x^4 * (2 * x + 3) * (2 * x - 3) - 12 := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l244_24404


namespace NUMINAMATH_CALUDE_unique_solution_system_l244_24457

/-- The system of equations has only one real solution (0, 0, 0, 0) -/
theorem unique_solution_system :
  ∃! (x y z w : ℝ),
    x = z + w + Real.sqrt (z * w * x) ∧
    y = w + x + Real.sqrt (w * x * y) ∧
    z = x + y + Real.sqrt (x * y * z) ∧
    w = y + z + Real.sqrt (y * z * w) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l244_24457


namespace NUMINAMATH_CALUDE_inequality_never_satisfied_l244_24493

theorem inequality_never_satisfied (m : ℝ) :
  (∀ x : ℝ, ¬(|x - 4| + |3 - x| < m)) → m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_never_satisfied_l244_24493


namespace NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l244_24450

theorem units_digit_of_k_squared_plus_two_to_k (k : ℕ) : 
  k = 2008^2 + 2^2008 → (k^2 + 2^k) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_k_squared_plus_two_to_k_l244_24450


namespace NUMINAMATH_CALUDE_fraction_of_fraction_two_ninths_of_three_fourths_l244_24437

theorem fraction_of_fraction (a b c d : ℚ) (h1 : b ≠ 0) (h2 : d ≠ 0) :
  (a / b) / (c / d) = (a * d) / (b * c) := by sorry

theorem two_ninths_of_three_fourths :
  (2 : ℚ) / 9 / ((3 : ℚ) / 4) = 8 / 27 := by sorry

end NUMINAMATH_CALUDE_fraction_of_fraction_two_ninths_of_three_fourths_l244_24437


namespace NUMINAMATH_CALUDE_parallel_lines_m_value_l244_24445

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {a b c d : ℝ} : 
  (∀ x y : ℝ, a * x + b * y = 0 ↔ y = c * x + d) → b ≠ 0 → a / b = -c

/-- The value of m for which the lines 2x + my = 0 and y = 3x - 1 are parallel -/
theorem parallel_lines_m_value : 
  ∃ m : ℝ, (∀ x y : ℝ, 2 * x + m * y = 0 ↔ y = 3 * x - 1) ∧ m = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_m_value_l244_24445


namespace NUMINAMATH_CALUDE_remainder_6n_mod_4_l244_24495

theorem remainder_6n_mod_4 (n : ℤ) (h : n ≡ 3 [ZMOD 4]) : 6 * n ≡ 2 [ZMOD 4] := by
  sorry

end NUMINAMATH_CALUDE_remainder_6n_mod_4_l244_24495


namespace NUMINAMATH_CALUDE_scaled_prism_volume_scaled_54_cubic_feet_prism_l244_24402

/-- Theorem: Scaling a rectangular prism's volume -/
theorem scaled_prism_volume 
  (V : ℝ) 
  (a b c : ℝ) 
  (hV : V > 0) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hc : c > 0) : 
  a * b * c * V = (a * b * c) * V := by sorry

/-- Corollary: Specific case of scaling a 54 cubic feet prism -/
theorem scaled_54_cubic_feet_prism :
  let V : ℝ := 54
  let a : ℝ := 2
  let b : ℝ := 3
  let c : ℝ := 1.5
  a * b * c * V = 486 := by sorry

end NUMINAMATH_CALUDE_scaled_prism_volume_scaled_54_cubic_feet_prism_l244_24402


namespace NUMINAMATH_CALUDE_average_speed_calculation_l244_24484

/-- Given a journey of 234 miles that takes 27/4 hours, prove that the average speed is 936/27 miles per hour. -/
theorem average_speed_calculation (distance : ℝ) (time : ℝ) (h1 : distance = 234) (h2 : time = 27/4) :
  distance / time = 936 / 27 := by
  sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l244_24484


namespace NUMINAMATH_CALUDE_closure_property_implies_divisibility_characterization_l244_24443

theorem closure_property_implies_divisibility_characterization 
  (S : Set ℤ) 
  (closure : ∀ a b : ℤ, a ∈ S → b ∈ S → (a + b) ∈ S) 
  (has_negative : ∃ n : ℤ, n < 0 ∧ n ∈ S) 
  (has_positive : ∃ p : ℤ, p > 0 ∧ p ∈ S) : 
  ∃ d : ℤ, ∀ x : ℤ, x ∈ S ↔ d ∣ x := by
sorry

end NUMINAMATH_CALUDE_closure_property_implies_divisibility_characterization_l244_24443


namespace NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_eq_zero_l244_24417

theorem cos_alpha_minus_pi_sixth_eq_zero (α : Real)
  (h1 : 2 * Real.tan α * Real.sin α = 3)
  (h2 : -Real.pi/2 < α)
  (h3 : α < 0) :
  Real.cos (α - Real.pi/6) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_minus_pi_sixth_eq_zero_l244_24417


namespace NUMINAMATH_CALUDE_two_out_of_three_win_probability_l244_24401

/-- The probability that exactly two out of three players win a game, given their individual probabilities of success. -/
theorem two_out_of_three_win_probability
  (p_alice : ℚ) (p_benjamin : ℚ) (p_carol : ℚ)
  (h_alice : p_alice = 1/5)
  (h_benjamin : p_benjamin = 3/8)
  (h_carol : p_carol = 2/7) :
  (p_alice * p_benjamin * (1 - p_carol)) +
  (p_alice * p_carol * (1 - p_benjamin)) +
  (p_benjamin * p_carol * (1 - p_alice)) = 49/280 := by
  sorry

end NUMINAMATH_CALUDE_two_out_of_three_win_probability_l244_24401


namespace NUMINAMATH_CALUDE_ratio_q_p_l244_24400

def total_slips : ℕ := 60
def num_range : Set ℕ := Finset.range 10
def slips_per_num : ℕ := 6
def drawn_slips : ℕ := 4

def p : ℚ := (10 : ℚ) / Nat.choose total_slips drawn_slips
def q : ℚ := (5400 : ℚ) / Nat.choose total_slips drawn_slips

theorem ratio_q_p : q / p = 540 := by sorry

end NUMINAMATH_CALUDE_ratio_q_p_l244_24400


namespace NUMINAMATH_CALUDE_min_value_quadratic_form_l244_24485

theorem min_value_quadratic_form (x y : ℝ) :
  x^2 + 2*x*y + 2*y^2 ≥ 0 ∧ (x^2 + 2*x*y + 2*y^2 = 0 ↔ x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_form_l244_24485


namespace NUMINAMATH_CALUDE_millionthDigitOf1Over41_l244_24498

-- Define the fraction
def fraction : ℚ := 1 / 41

-- Define the function to get the nth digit after the decimal point
noncomputable def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

-- State the theorem
theorem millionthDigitOf1Over41 : 
  nthDigitAfterDecimal fraction 1000000 = 9 := by sorry

end NUMINAMATH_CALUDE_millionthDigitOf1Over41_l244_24498


namespace NUMINAMATH_CALUDE_savings_calculation_l244_24406

/-- Calculates the total savings of Thomas and Joseph after 6 years -/
def total_savings (thomas_monthly_savings : ℚ) (years : ℕ) : ℚ :=
  let months : ℕ := years * 12
  let thomas_total : ℚ := thomas_monthly_savings * months
  let joseph_monthly_savings : ℚ := thomas_monthly_savings - (2 / 5) * thomas_monthly_savings
  let joseph_total : ℚ := joseph_monthly_savings * months
  thomas_total + joseph_total

/-- Proves that Thomas and Joseph's combined savings after 6 years equals $4608 -/
theorem savings_calculation : total_savings 40 6 = 4608 := by
  sorry

end NUMINAMATH_CALUDE_savings_calculation_l244_24406


namespace NUMINAMATH_CALUDE_largest_n_value_l244_24416

/-- Represents a digit in base 5 -/
def Base5Digit := Fin 5

/-- Represents a digit in base 9 -/
def Base9Digit := Fin 9

/-- Converts a three-digit number in base 5 to base 10 -/
def base5ToBase10 (x y z : Base5Digit) : ℕ :=
  25 * x.val + 5 * y.val + z.val

/-- Converts a three-digit number in base 9 to base 10 -/
def base9ToBase10 (z y x : Base9Digit) : ℕ :=
  81 * z.val + 9 * y.val + x.val

theorem largest_n_value (n : ℕ) 
  (h1 : ∃ (x y z : Base5Digit), n = base5ToBase10 x y z)
  (h2 : ∃ (x y z : Base9Digit), n = base9ToBase10 z y x) :
  n ≤ 121 ∧ ∃ (x y z : Base5Digit), 121 = base5ToBase10 x y z ∧ 
    ∃ (x y z : Base9Digit), 121 = base9ToBase10 z y x :=
by sorry

end NUMINAMATH_CALUDE_largest_n_value_l244_24416


namespace NUMINAMATH_CALUDE_min_x_squared_isosceles_trapezoid_l244_24412

/-- Represents a trapezoid ABCD with specific properties -/
structure IsoscelesTrapezoid where
  -- Length of base AB
  ab : ℝ
  -- Length of base CD
  cd : ℝ
  -- Length of side AD (equal to BC)
  x : ℝ
  -- Ensures the trapezoid is isosceles
  isIsosceles : ad = bc
  -- Ensures a circle with center on AB is tangent to AD and BC
  hasTangentCircle : ∃ (center : ℝ), 0 ≤ center ∧ center ≤ ab ∧
    ∃ (radius : ℝ), radius > 0 ∧
    (center - radius)^2 + x^2 = (ab/2)^2 ∧
    (center + radius)^2 + x^2 = (ab/2)^2

/-- The theorem stating the minimum value of x^2 for the given trapezoid -/
theorem min_x_squared_isosceles_trapezoid (t : IsoscelesTrapezoid)
  (h1 : t.ab = 50)
  (h2 : t.cd = 14) :
  ∃ (m : ℝ), m^2 = 800 ∧ ∀ (y : ℝ), t.x = y → y^2 ≥ m^2 := by
  sorry

end NUMINAMATH_CALUDE_min_x_squared_isosceles_trapezoid_l244_24412


namespace NUMINAMATH_CALUDE_largest_n_with_unique_k_l244_24459

theorem largest_n_with_unique_k : ∃ (k : ℤ), 
  (5 : ℚ)/11 < (359 : ℚ)/(359 + k) ∧ (359 : ℚ)/(359 + k) < (6 : ℚ)/11 ∧
  (∀ (n : ℕ) (k₁ k₂ : ℤ), n > 359 →
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₁) ∧ (n : ℚ)/(n + k₁) < (6 : ℚ)/11) ∧
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₂) ∧ (n : ℚ)/(n + k₂) < (6 : ℚ)/11) →
    k₁ = k₂) →
  (∃ (k₁ k₂ : ℤ), k₁ ≠ k₂ ∧
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₁) ∧ (n : ℚ)/(n + k₁) < (6 : ℚ)/11) ∧
    ((5 : ℚ)/11 < (n : ℚ)/(n + k₂) ∧ (n : ℚ)/(n + k₂) < (6 : ℚ)/11)) :=
by sorry

end NUMINAMATH_CALUDE_largest_n_with_unique_k_l244_24459


namespace NUMINAMATH_CALUDE_ordering_of_exponential_and_logarithm_l244_24492

/-- Given a = e^0.1 - 1, b = 0.1, and c = ln 1.1, prove that a > b > c -/
theorem ordering_of_exponential_and_logarithm :
  let a := Real.exp 0.1 - 1
  let b := 0.1
  let c := Real.log 1.1
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_ordering_of_exponential_and_logarithm_l244_24492


namespace NUMINAMATH_CALUDE_parabola_directrix_coefficient_l244_24429

/-- For a parabola with equation y = ax² and directrix y = 2, prove that a = -1/8 -/
theorem parabola_directrix_coefficient (a : ℝ) : 
  (∀ x y : ℝ, y = a * x^2) →  -- Parabola equation
  (∃ y : ℝ, y = 2 ∧ ∀ x : ℝ, y = -1 / (4 * a)) →  -- Directrix equation
  a = -1/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_coefficient_l244_24429


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l244_24466

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 120) :
  a 9 - (1/3) * a 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l244_24466


namespace NUMINAMATH_CALUDE_female_democrats_count_l244_24408

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) :
  total = 750 →
  female + male = total →
  (female / 2 : ℚ) + (male / 4 : ℚ) = (total / 3 : ℚ) →
  female / 2 = 125 :=
by sorry

end NUMINAMATH_CALUDE_female_democrats_count_l244_24408


namespace NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_one_fifth_l244_24415

theorem sqrt_meaningful_iff_x_geq_one_fifth (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 5 * x - 1) ↔ x ≥ 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_iff_x_geq_one_fifth_l244_24415


namespace NUMINAMATH_CALUDE_robins_gum_pieces_l244_24490

/-- 
Given that Robin had an initial number of gum pieces, her brother gave her 26 more,
and now she has 44 pieces in total, prove that she initially had 18 pieces.
-/
theorem robins_gum_pieces (x : ℕ) : x + 26 = 44 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_robins_gum_pieces_l244_24490


namespace NUMINAMATH_CALUDE_symmetry_implies_a_value_l244_24420

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are the same. -/
def symmetric_y_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = B.2

theorem symmetry_implies_a_value :
  ∀ a : ℝ, symmetric_y_axis (a, 1) (-3, 1) → a = 3 :=
by sorry

end NUMINAMATH_CALUDE_symmetry_implies_a_value_l244_24420
