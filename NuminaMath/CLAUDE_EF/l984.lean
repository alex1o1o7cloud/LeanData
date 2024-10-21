import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l984_98429

/-- The cubic equation -/
def cubic_equation (x : ℝ) : Prop := 27 * x^3 - 12 * x^2 - 4 * x - 1 = 0

/-- The form of the root -/
def root_form (p q d : ℕ+) (x : ℝ) : Prop :=
  x = ((p : ℝ)^(1/3) + (q : ℝ)^(1/3) + 1) / (d : ℝ)

/-- The theorem statement -/
theorem cubic_root_sum (p q d : ℕ+) :
  (∃ x : ℝ, cubic_equation x ∧ root_form p q d x) → p + q + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l984_98429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_eight_twelfth_power_l984_98402

theorem fourth_root_eight_twelfth_power : (8 : ℝ) ^ ((1/4 : ℝ) * 12) = 512 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_root_eight_twelfth_power_l984_98402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l984_98420

def monotonically_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem interval_of_increase (a : ℝ) :
  (monotonically_decreasing (λ x ↦ a * x + 1)) →
  (Set.Iic 2 : Set ℝ) = {x | ∀ y, x < y → a * (x^2 - 4*x + 3) < a * (y^2 - 4*y + 3)} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interval_of_increase_l984_98420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l984_98418

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -Real.sqrt 3; Real.sqrt 3, 1]

theorem matrix_power_four : 
  A ^ 4 = !![(-8 : ℝ), 8 * Real.sqrt 3; -8 * Real.sqrt 3, (-8 : ℝ)] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_four_l984_98418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_fraction_is_two_thirds_l984_98439

/-- The fraction of cookies in the blue and green tins -/
noncomputable def cookies_in_blue_and_green (total : ℝ) : ℝ :=
  let blue := (1 / 4 : ℝ) * total
  let not_blue := total - blue
  let green := (5 / 9 : ℝ) * not_blue
  (blue + green) / total

/-- Theorem stating that the fraction of cookies in the blue and green tins is 2/3 -/
theorem cookies_fraction_is_two_thirds (total : ℝ) (h : total > 0) :
  cookies_in_blue_and_green total = 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_fraction_is_two_thirds_l984_98439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_61_l984_98462

/-- Represents the age group of a person -/
inductive AgeGroup
  | Under18
  | Over18

/-- Represents the time of day -/
inductive TimeOfDay
  | Morning
  | Afternoon

/-- Represents the type of ride -/
inductive RideType
  | BumperCar
  | SpaceShuttle
  | FerrisWheel

/-- Returns the price of a ride based on age group, ride type, and time of day -/
def ridePrice (age : AgeGroup) (ride : RideType) (time : TimeOfDay) : ℕ :=
  match age, ride, time with
  | AgeGroup.Under18, RideType.BumperCar, TimeOfDay.Morning => 2
  | AgeGroup.Over18, RideType.BumperCar, TimeOfDay.Morning => 3
  | AgeGroup.Under18, RideType.SpaceShuttle, TimeOfDay.Morning => 4
  | AgeGroup.Over18, RideType.SpaceShuttle, TimeOfDay.Morning => 5
  | AgeGroup.Under18, RideType.FerrisWheel, TimeOfDay.Morning => 5
  | AgeGroup.Over18, RideType.FerrisWheel, TimeOfDay.Morning => 6
  | _, _, TimeOfDay.Afternoon => 
    match age, ride with
    | AgeGroup.Under18, RideType.BumperCar => 3
    | AgeGroup.Over18, RideType.BumperCar => 4
    | AgeGroup.Under18, RideType.SpaceShuttle => 5
    | AgeGroup.Over18, RideType.SpaceShuttle => 6
    | AgeGroup.Under18, RideType.FerrisWheel => 6
    | AgeGroup.Over18, RideType.FerrisWheel => 7

/-- Calculates the total cost for a person's rides -/
def personCost (age : AgeGroup) (bumperCarMorning bumperCarAfternoon spaceShuttleMorning spaceShuttleAfternoon ferrisWheelMorning ferrisWheelAfternoon : ℕ) : ℕ :=
  bumperCarMorning * ridePrice age RideType.BumperCar TimeOfDay.Morning +
  bumperCarAfternoon * ridePrice age RideType.BumperCar TimeOfDay.Afternoon +
  spaceShuttleMorning * ridePrice age RideType.SpaceShuttle TimeOfDay.Morning +
  spaceShuttleAfternoon * ridePrice age RideType.SpaceShuttle TimeOfDay.Afternoon +
  ferrisWheelMorning * ridePrice age RideType.FerrisWheel TimeOfDay.Morning +
  ferrisWheelAfternoon * ridePrice age RideType.FerrisWheel TimeOfDay.Afternoon

theorem total_cost_is_61 :
  let maraCost := personCost AgeGroup.Under18 1 1 0 0 3 0
  let rileyCost := personCost AgeGroup.Over18 0 0 2 2 2 1
  maraCost + rileyCost = 61 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_cost_is_61_l984_98462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_from_matrix_product_l984_98438

noncomputable def dilation_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![k, 0; 0, k]

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := 
  !![Real.cos θ, -Real.sin θ; Real.sin θ, Real.cos θ]

def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![1, 0; 0, -1]

theorem tan_theta_from_matrix_product (k θ : ℝ) (hk : k > 0) :
  reflection_matrix * rotation_matrix θ * dilation_matrix k = !![5, -12; -12, -5] →
  Real.tan θ = 12 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_from_matrix_product_l984_98438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l984_98430

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin (x^2)

-- State the theorem
theorem f_derivative (x : ℝ) : 
  deriv f x = 2 * x * Real.cos (x^2) := by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_l984_98430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_and_x_values_l984_98461

def A : Set ℝ := {1, 3, 5}
def B (x : ℝ) : Set ℝ := {1, 2, x^2 - 1}

theorem set_intersection_and_x_values :
  ∀ x : ℝ, (A ∪ B x) = {1, 2, 3, 5} →
  ((x = 2 ∨ x = -2 ∨ x = Real.sqrt 6 ∨ x = -Real.sqrt 6) ∧
   ((A ∩ B x) = {1, 3} ∨ (A ∩ B x) = {1, 5})) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_intersection_and_x_values_l984_98461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_square_l984_98467

/-- A function that checks if a number is a perfect square. -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- A function that returns the digits of a natural number. -/
def digits (n : ℕ) : List ℕ :=
  sorry

/-- A function that checks if two lists have no common elements. -/
def disjoint (l1 l2 : List ℕ) : Prop :=
  sorry

theorem smallest_four_digit_square :
  ∃ (a b : ℕ),
    isPerfectSquare 1369 ∧
    isPerfectSquare a ∧
    isPerfectSquare b ∧
    a ≥ 10 ∧ a < 100 ∧
    b ≥ 100 ∧ b < 1000 ∧
    disjoint (digits 1369) (digits a) ∧
    disjoint (digits 1369) (digits b) ∧
    disjoint (digits a) (digits b) ∧
    (digits 1369 ++ digits a ++ digits b).toFinset = {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
    (∀ (n : ℕ), 1000 ≤ n ∧ n < 1369 →
      ¬(∃ (x y : ℕ),
        isPerfectSquare n ∧
        isPerfectSquare x ∧
        isPerfectSquare y ∧
        x ≥ 10 ∧ x < 100 ∧
        y ≥ 100 ∧ y < 1000 ∧
        disjoint (digits n) (digits x) ∧
        disjoint (digits n) (digits y) ∧
        disjoint (digits x) (digits y) ∧
        (digits n ++ digits x ++ digits y).toFinset = {1, 2, 3, 4, 5, 6, 7, 8, 9})) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_four_digit_square_l984_98467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_glasses_cost_l984_98419

/-- Calculates the total cost of James's glasses purchase --/
theorem james_glasses_cost (frames_cost lenses_cost coatings_cost insurance_coverage
                            frame_coupon loyalty_discount sales_tax : ℝ) : 
  frames_cost = 200 ∧ 
  lenses_cost = 500 ∧ 
  coatings_cost = 150 ∧ 
  insurance_coverage = 0.8 ∧ 
  frame_coupon = 50 ∧ 
  loyalty_discount = 0.1 ∧ 
  sales_tax = 0.07 →
  ((frames_cost - frame_coupon) * (1 - loyalty_discount) + 
   lenses_cost * (1 - insurance_coverage) + coatings_cost) * 
  (1 + sales_tax) = 411.95 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_glasses_cost_l984_98419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_condition_equivalent_to_m_range_l984_98427

/-- The function f(x) = (m-1)ln(x) + mx² + 1 -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * Real.log x + m * x^2 + 1

/-- The theorem stating the equivalence of the condition and the range of m -/
theorem f_condition_equivalent_to_m_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ > 0 → f m x₁ - f m x₂ > 2 * (x₁ - x₂)) ↔
  m ≥ (1 + Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_condition_equivalent_to_m_range_l984_98427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l984_98484

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (4, 0)

-- Define point A
def point_A : ℝ × ℝ := (6, 0)

-- Define a point on the right branch of the hyperbola
def on_right_branch (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ P.1 > 0

-- Distance function
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem min_distance_sum :
  ∃ (min_val : ℝ), ∀ (P : ℝ × ℝ), 
    on_right_branch P → 
    distance P point_A + distance P right_focus ≥ min_val ∧
    min_val = 2 * Real.sqrt 11.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l984_98484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_condition_l984_98478

/-- A function f: ℝ → ℝ with derivative f'(x) = a(x-a)(x-1) has a local maximum at x=1 
    if and only if a is in the set (-∞, 0) ∪ (1, +∞) -/
theorem local_max_condition (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, deriv f x = a * (x - a) * (x - 1)) 
  (h2 : IsLocalMax f 1) : 
  a < 0 ∨ a > 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_max_condition_l984_98478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l984_98482

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x

-- Part 1
theorem part_one :
  ∀ m : ℝ, (∀ x : ℝ, x ≥ 0 → (x - 1) * f x ≥ m * x^2 - 1) ↔ m ≤ (1/2) :=
by sorry

-- Part 2
theorem part_two :
  ∀ x : ℝ, x > 0 → f x > 4 * Real.log x + 8 - 8 * Real.log 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l984_98482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_equality_implies_zero_sum_l984_98486

theorem tangent_ratio_equality_implies_zero_sum (θ α β γ a b c : ℝ) 
  (h : Real.tan (θ + α) / a = Real.tan (θ + β) / b ∧ Real.tan (θ + β) / b = Real.tan (θ + γ) / c) :
  (a + b) / (a - b) * Real.sin (α - β) ^ 2 + 
  (b + c) / (b - c) * Real.sin (β - γ) ^ 2 + 
  (c + a) / (c - a) * Real.sin (γ - α) ^ 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_ratio_equality_implies_zero_sum_l984_98486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_on_line_l984_98415

/-- Given three real numbers forming an arithmetic sequence and defining a line,
    prove the range of distances from a point on the line to a fixed point. -/
theorem distance_range_on_line (a b c : ℝ) (P N : ℝ × ℝ) : 
  b = (a + c) / 2 →  -- arithmetic sequence condition
  P = (-3, 2) →  -- given point P
  N = (2, 3) →  -- given point N
  ∃ M : ℝ × ℝ, 
    (a * M.fst + b * M.snd + c = 0) ∧  -- M is on the line
    ((M.fst - P.fst) * a + (M.snd - P.snd) * b = 0) →  -- PM is perpendicular to the line
    Real.sqrt 2 ≤ Real.sqrt ((M.fst - N.fst)^2 + (M.snd - N.snd)^2) ∧ 
    Real.sqrt ((M.fst - N.fst)^2 + (M.snd - N.snd)^2) ≤ 5 * Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_range_on_line_l984_98415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bc_length_l984_98414

/-- Given two right triangles ABC and ABD with specific measurements, 
    prove that the length of BC is 2√379 -/
theorem bc_length (A B C D : ℝ × ℝ) : 
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  AD = 50 →
  CD = 24 →
  AC = 25 →
  (A.2 = B.2 ∧ A.2 = C.2) →  -- A, B, C form a right triangle
  (A.2 = B.2 ∧ A.2 = D.2) →  -- A, B, D form a right triangle
  BC = 2 * Real.sqrt 379 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bc_length_l984_98414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_product_plus_vector_l984_98444

open Matrix

theorem matrix_vector_product_plus_vector :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![4, -2; 6, 5]
  let v : Matrix (Fin 2) (Fin 1) ℝ := !![-2; 3]
  let w : Matrix (Fin 2) (Fin 1) ℝ := !![1; -1]
  A * v + w = !![-13; 2] := by
  -- Expand the definitions
  simp
  -- Perform the matrix multiplication and addition
  ring
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_product_plus_vector_l984_98444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_picking_problem_l984_98463

/-- The total number of oranges picked after a 10% increase -/
def total_oranges_after_increase (mary jason tom sarah : ℕ) : ℕ :=
  let increase (n : ℕ) := (n : ℚ) + (n : ℚ) * (1 : ℚ) / (10 : ℚ)
  ⌊increase mary + increase jason + increase tom + increase sarah⌋₊

/-- Theorem stating the total number of oranges picked after the increase -/
theorem orange_picking_problem :
  total_oranges_after_increase 122 105 85 134 = 491 := by
  sorry

#eval total_oranges_after_increase 122 105 85 134

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_picking_problem_l984_98463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_distances_constant_diff_inverse_distances_constant_l984_98428

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a chord
structure Chord where
  start : Point
  finish : Point  -- Changed 'end' to 'finish' to avoid reserved keyword

-- Function to check if a point is inside a circle
def isInside (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.1)^2 + (p.y - c.center.2)^2 < c.radius^2

-- Function to calculate distance from a point to a tangent
noncomputable def distToTangent (c : Circle) (p : Point) (tangentPoint : Point) : ℝ :=
  sorry

-- Theorem for the first part
theorem sum_inverse_distances_constant (c : Circle) (m : Point) (chord : Chord) 
  (h_inside : isInside c m) :
  ∃ (k : ℝ), ∀ (a b : Point), 
    chord.start = a → chord.finish = b →
    (1 / distToTangent c m a + 1 / distToTangent c m b) = k := by
  sorry

-- Theorem for the second part
theorem diff_inverse_distances_constant (c : Circle) (m : Point) (chord : Chord) 
  (h_outside : ¬isInside c m) :
  ∃ (k : ℝ), ∀ (a b : Point), 
    chord.start = a → chord.finish = b →
    |1 / distToTangent c m a - 1 / distToTangent c m b| = k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_inverse_distances_constant_diff_inverse_distances_constant_l984_98428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probabilities_l984_98459

def total_cells : ℕ := 64
def marked_cells : ℕ := 8
def drawn_cells : ℕ := 8

def probability_k_correct (k : ℕ) : ℚ :=
  (Nat.choose marked_cells k * Nat.choose (total_cells - marked_cells) (drawn_cells - k)) / Nat.choose total_cells drawn_cells

theorem lottery_probabilities :
  (probability_k_correct 4 = (Nat.choose marked_cells 4 * Nat.choose (total_cells - marked_cells) (drawn_cells - 4)) / Nat.choose total_cells drawn_cells) ∧
  (probability_k_correct 5 = (Nat.choose marked_cells 5 * Nat.choose (total_cells - marked_cells) (drawn_cells - 5)) / Nat.choose total_cells drawn_cells) ∧
  (probability_k_correct 8 = 1 / Nat.choose total_cells drawn_cells) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lottery_probabilities_l984_98459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_puzzle_l984_98448

theorem absolute_value_puzzle (x : ℤ) (h : x = -2023) : 
  (abs (abs (abs (abs x - x) - abs x) - x)) = 4046 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_puzzle_l984_98448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_angle_l984_98400

theorem cone_lateral_surface_angle (r s : ℝ) (hr : r = 3) (hs : s = 10) :
  (2 * Real.pi * r) / (2 * Real.pi * s) * 360 = 108 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_lateral_surface_angle_l984_98400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l984_98441

-- Define a cuboid
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : length > 0
  width_pos : width > 0
  height_pos : height > 0

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the distance function between two points
noncomputable def distance (p1 p2 : Point3D) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2)

-- Define the vertices of a cuboid
def vertices (c : Cuboid) : List Point3D :=
  [
    ⟨0, 0, 0⟩,
    ⟨c.length, 0, 0⟩,
    ⟨c.length, c.width, 0⟩,
    ⟨0, c.width, 0⟩,
    ⟨0, 0, c.height⟩,
    ⟨c.length, 0, c.height⟩,
    ⟨c.length, c.width, c.height⟩,
    ⟨0, c.width, c.height⟩
  ]

-- Theorem: There exists a point equidistant from all vertices in a cuboid
theorem equidistant_point_exists (c : Cuboid) :
  ∃ p : Point3D, ∀ v1 v2 : Point3D, v1 ∈ vertices c → v2 ∈ vertices c →
    distance p v1 = distance p v2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_exists_l984_98441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_square_rectangles_l984_98470

/-- The number of dots in each row and column of the square array -/
def gridSize : Nat := 5

/-- The total number of rectangles (including squares) that can be formed -/
def totalRectangles : Nat := (gridSize.choose 2) * (gridSize.choose 2)

/-- The number of squares that can be formed -/
def totalSquares : Nat := Finset.sum (Finset.range (gridSize - 1)) (fun i => (gridSize - i) ^ 2)

/-- The number of non-square rectangles that can be formed -/
def nonSquareRectangles : Nat := totalRectangles - totalSquares

theorem count_non_square_rectangles :
  nonSquareRectangles = 70 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_non_square_rectangles_l984_98470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_area_ratio_l984_98442

/-- Given a square and a circle where each side of the square contains a chord 
    of the circle equal in length to the diameter of the circle, 
    the ratio of the area of the square to the area of the circle is 4/π -/
theorem square_circle_area_ratio : 
  ∀ (r : ℝ), r > 0 → 
  (let square_side := 2 * r
   let circle_diameter := 2 * r
   let square_area := square_side ^ 2
   let circle_area := π * r ^ 2
   square_area / circle_area) = 4 / π := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_area_ratio_l984_98442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_distance_ratio_l984_98469

/-- Proves that the ratio of Sarah's swimming distance to Julien's is 2:1 given the problem conditions -/
theorem swimming_distance_ratio :
  ∀ (sarah_distance julien_distance jamir_distance : ℕ),
  julien_distance = 50 →
  jamir_distance = sarah_distance + 20 →
  7 * (sarah_distance + jamir_distance + julien_distance) = 1890 →
  sarah_distance = 2 * julien_distance := by
  intro sarah_distance julien_distance jamir_distance h_julien h_jamir h_total
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_swimming_distance_ratio_l984_98469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_routes_count_l984_98495

/-- Represents a bus route in the city -/
structure BusRoute where
  stops : Finset Nat
  stop_count : stops.card = 3

/-- Represents the configuration of bus routes in the city -/
structure BusSystem where
  routes : Finset BusRoute
  route_between_stops : ∀ s₁ s₂, ∃ r ∈ routes, s₁ ∈ r.stops ∧ s₂ ∈ r.stops
  common_stop : ∀ r₁ r₂, r₁ ∈ routes → r₂ ∈ routes → r₁ ≠ r₂ → (r₁.stops ∩ r₂.stops).card = 1

/-- The number of bus routes in a valid bus system is either 1 or 7 -/
theorem bus_routes_count (bs : BusSystem) : bs.routes.card = 1 ∨ bs.routes.card = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_routes_count_l984_98495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_l984_98488

-- Define the curve f(x) = √x
noncomputable def f (x : ℝ) := Real.sqrt x

-- Define the derivative of f(x)
noncomputable def f_derivative (x : ℝ) : ℝ := 1 / (2 * Real.sqrt x)

-- Define the tangent line y = mx + n
structure TangentLine where
  m : ℝ
  n : ℝ

-- State the theorem
theorem tangent_line_properties (x : ℝ) (line : TangentLine) 
  (h_pos : x > 0)
  (h_tangent : line.m = f_derivative x ∧ f x = line.m * x + line.n) :
  line.m > 0 ∧ line.n * Real.log line.m ≤ 1 / (4 * Real.exp 1) := by
  sorry

#check tangent_line_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_properties_l984_98488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_min_value_expression_l984_98490

noncomputable def semicircle (x : ℝ) : ℝ := Real.sqrt (1 - x^2)

theorem min_distance_to_line : 
  ∃ (min_dist : ℝ), (∀ x : ℝ, |x - 2 + semicircle x| / Real.sqrt 2 ≥ min_dist) ∧ min_dist = Real.sqrt 2 - 1 := by sorry

theorem min_value_expression : 
  ∃ (min_val : ℝ), (∀ x : ℝ, |x - 2 + semicircle x| ≥ min_val) ∧ min_val = 2 - Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_line_min_value_expression_l984_98490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_parenthesizations_l984_98492

-- Define the two parenthesizations
def parenthesization1 : ℕ := (3^(3^3))^3
def parenthesization2 : ℕ := 3^((3^3)^3)

-- Theorem statement
theorem distinct_parenthesizations :
  Finset.card {parenthesization1, parenthesization2} = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_parenthesizations_l984_98492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_same_domain_other_pairs_not_equal_l984_98468

-- Define the functions for option B
noncomputable def f (x : ℝ) : ℝ := x^2 / x
noncomputable def g (x : ℝ) : ℝ := x

-- Theorem stating that f and g are the same function for x ≠ 0
theorem f_equals_g : ∀ x : ℝ, x ≠ 0 → f x = g x := by sorry

-- Define the domain of f and g
def domain_f_g : Set ℝ := {x : ℝ | x ≠ 0}

-- Theorem stating that f and g have the same domain
theorem same_domain : domain_f_g = {x : ℝ | x ≠ 0} := by sorry

-- Theorem stating that other function pairs are not the same
theorem other_pairs_not_equal :
  (∃ x : ℝ, (1 : ℝ) ≠ x^0) ∧
  (∃ x : ℝ, |x| ≠ (if x > 0 then x else -x)) ∧
  (∃ x : ℝ, (1 : ℝ) ≠ x/x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_same_domain_other_pairs_not_equal_l984_98468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_shapes_l984_98449

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define possible shapes
inductive ReflectedShape
  | ConvexDeltoid
  | ConcaveDeltoid
  | IsoscelesTriangle
  | Rhombus
  | Square

-- Define a function that reflects a triangle across one of its sides
def reflectTriangle (t : Triangle) (side : Fin 3) : Triangle :=
  sorry

-- Define a function that determines the resulting shape after reflection
def resultingShape (original : Triangle) (reflected : Triangle) : ReflectedShape :=
  sorry

-- Theorem stating that reflecting a triangle can result in the specified shapes
theorem triangle_reflection_shapes (t : Triangle) :
  ∃ (side : Fin 3), 
    let reflected := reflectTriangle t side
    (resultingShape t reflected = ReflectedShape.ConvexDeltoid) ∨
    (resultingShape t reflected = ReflectedShape.ConcaveDeltoid) ∨
    (resultingShape t reflected = ReflectedShape.IsoscelesTriangle) ∨
    (resultingShape t reflected = ReflectedShape.Rhombus) ∨
    (resultingShape t reflected = ReflectedShape.Square) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_reflection_shapes_l984_98449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l984_98477

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x - 1 / x

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 (3/2), f (a * x - 1) > f 2) ↔ 
  a ∈ Set.union (Set.Ioo (1/2) (2/3)) (Set.Ioi 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l984_98477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l984_98466

theorem tan_alpha_plus_pi_fourth (α : Real) : 
  (∃ m : Real, m = 2 ∧ Real.tan α = m) → 
  Real.tan (α + π/4) = -3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l984_98466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_positive_period_of_f_l984_98446

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + 2

-- State the theorem
theorem minimal_positive_period_of_f :
  ∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
  (∀ (T' : ℝ), T' > 0 ∧ (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧
  T = Real.pi := by
  sorry

#check minimal_positive_period_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimal_positive_period_of_f_l984_98446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expressions_l984_98422

theorem trigonometric_expressions :
  (∀ x : ℝ,
    Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) -
    Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1 / 2) ∧
  (∀ α : ℝ,
    (Real.sin (2 * π - α) * Real.cos (π + α) * Real.cos (π / 2 + α) * Real.cos (11 * π / 2 - α)) /
    (Real.cos (π - α) * Real.sin (3 * π - α) * Real.sin (-π - α) * Real.sin (9 * π / 2 + α)) = -Real.tan α) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expressions_l984_98422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l984_98409

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt 3 * (((a + b) * (b + c) * (c + a)) ^ (1/3 : ℝ)) ≥ 2 * Real.sqrt (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_proof_l984_98409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_values_for_sec_minus_tan_l984_98494

theorem cos_values_for_sec_minus_tan (x : ℝ) :
  (1 / Real.cos x) - (Real.sin x / Real.cos x) = 3/4 → (Real.cos x = 24/25 ∨ Real.cos x = -24/25) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_values_for_sec_minus_tan_l984_98494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_average_rate_of_change_comparison_l984_98413

theorem sine_average_rate_of_change_comparison (Δx : ℝ) (h : Δx ≠ 0) :
  let k₁ := (Real.sin Δx - Real.sin 0) / Δx
  let k₂ := (Real.sin (π/2 + Δx) - Real.sin (π/2)) / Δx
  k₁ > k₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_average_rate_of_change_comparison_l984_98413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_sequence_l984_98416

/-- If a, b, c form an arithmetic sequence, then 2^a, 2^b, 2^c form a geometric sequence. -/
theorem arithmetic_to_geometric_sequence (a b c : ℝ) : 
  (b - a = c - b) → ((2:ℝ)^b)^2 = ((2:ℝ)^a) * ((2:ℝ)^c) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_to_geometric_sequence_l984_98416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_inequality_l984_98456

open Real

/-- The function f(x) = (x^2 + m) / x - 2 ln(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + m) / x - 2 * log x

/-- Theorem: If f(x) has two extreme points x₁ and x₂ with x₁ < x₂, then f(x₂) < x₂ - 1 -/
theorem extreme_point_inequality (m : ℝ) (x₁ x₂ : ℝ) :
  x₁ > 0 → x₂ > 0 → x₁ < x₂ →
  (∀ x, x > 0 → (deriv (f m)) x = 0 → x = x₁ ∨ x = x₂) →
  f m x₂ < x₂ - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_point_inequality_l984_98456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l984_98453

theorem trigonometric_identities (α : ℝ) 
  (h1 : 0 < α) (h2 : α < Real.pi) (h3 : Real.tan α = -2) : 
  (Real.sin (α + Real.pi/6) = (2*Real.sqrt 15 - Real.sqrt 5) / 10) ∧ 
  ((2*Real.cos (Real.pi/2 + α) - Real.cos (Real.pi - α)) / (Real.sin (Real.pi/2 - α) - 3*Real.sin (Real.pi + α)) = 5/7) ∧ 
  (2*Real.sin α^2 - Real.sin α * Real.cos α + Real.cos α^2 = 11/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l984_98453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_extremum_l984_98491

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- State the theorem
theorem not_necessarily_extremum :
  (∀ x, HasDerivAt f (3 * x^2) x) ∧  -- f is differentiable
  (HasDerivAt f 0 0) ∧               -- f'(0) = 0
  ¬ (∃ ε > 0, ∀ x, ‖x‖ < ε → f x ≤ f 0 ∨ f x ≥ f 0) -- x = 0 is not an extremum
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_necessarily_extremum_l984_98491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_other_side_red_given_red_side_l984_98498

/-- Represents a card with two sides -/
inductive Card
| BlackBlack
| BlackRed
| RedRed
deriving DecidableEq

/-- The box containing the cards -/
def box : Multiset Card :=
  4 • {Card.BlackBlack} +
  3 • {Card.BlackRed} +
  3 • {Card.RedRed}

/-- The total number of cards in the box -/
def totalCards : ℕ := 10

/-- Predicate for a card having at least one red side -/
def hasRedSide (c : Card) : Prop :=
  c = Card.BlackRed ∨ c = Card.RedRed

/-- Predicate for a card being red on both sides -/
def isRedRed (c : Card) : Prop :=
  c = Card.RedRed

/-- The probability of drawing a card with a red side -/
noncomputable def probRedSide : ℝ :=
  (box.count Card.BlackRed + 2 * box.count Card.RedRed) / totalCards

theorem prob_other_side_red_given_red_side :
  (box.count Card.RedRed * 2) / (box.count Card.BlackRed + 2 * box.count Card.RedRed) = 2/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_other_side_red_given_red_side_l984_98498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_common_ratios_l984_98480

theorem sum_of_common_ratios (k p r : ℝ) (h_k : k ≠ 0) (h_p_neq_r : p ≠ r) :
  (k * p^2 + 2 * k * p = k * r^2 + 3 * k * r) →
  p + r = 1 := by
  intro h
  -- The proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_common_ratios_l984_98480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_l984_98497

/-- Calculates the simple interest given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

theorem total_interest_after_trebling (P : ℝ) (R : ℝ) :
  simpleInterest P R 10 = 1000 →
  simpleInterest P R 5 + simpleInterest (3 * P) R 5 = 2500 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_trebling_l984_98497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_and_tangent_l984_98489

/-- Given a line l₂ and a circle C, if line l₁ is perpendicular to l₂ and tangent to C,
    then l₁ has one of two specific equations. -/
theorem line_perpendicular_and_tangent
  (l₂ : Set (ℝ × ℝ)) -- Line l₂
  (C : Set (ℝ × ℝ)) -- Circle C
  (l₁ : Set (ℝ × ℝ)) -- Line l₁
  (h_l₂_eq : l₂ = {p : ℝ × ℝ | 4 * p.1 - 3 * p.2 + 1 = 0}) -- Equation of l₂
  (h_C_eq : C = {p : ℝ × ℝ | p.1^2 + p.2^2 = -2 * p.2 + 3}) -- Equation of C
  (h_perp : ∀ p ∈ l₁, ∀ q ∈ l₂, (p.1 - q.1) * 4 + (p.2 - q.2) * (-3) = 0) -- l₁ is perpendicular to l₂
  (h_tangent : ∃ p, p ∈ l₁ ∧ p ∈ C ∧ ∀ q ∈ l₁, q ∈ C → q = p) -- l₁ is tangent to C
  : l₁ = {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 + 6 = 0} ∨
    l₁ = {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 14 = 0} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_perpendicular_and_tangent_l984_98489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_triangle_l984_98435

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) * Real.cos (2 * x) - Real.cos (2 * x) ^ 2

theorem f_range_in_triangle (a b c x : ℝ) (h1 : b ^ 2 = a * c) (h2 : 0 < x) (h3 : x < π) :
  ∃ (y : ℝ), f x = y ∧ -2 ≤ y ∧ y ≤ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_in_triangle_l984_98435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_parabola_and_x_axis_l984_98433

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 1

-- State the theorem
theorem area_enclosed_by_parabola_and_x_axis :
  ∫ x in Set.Icc (-1) 1, (0 - f x) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_parabola_and_x_axis_l984_98433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lab_techs_to_uniforms_ratio_l984_98440

theorem lab_techs_to_uniforms_ratio :
  ∀ (num_uniforms num_lab_coats num_lab_techs : ℕ),
    num_uniforms = 12 →
    num_lab_coats = 6 * num_uniforms →
    (num_lab_coats + num_uniforms) / num_lab_techs = 14 →
    num_lab_techs * 2 = num_uniforms :=
λ num_uniforms num_lab_coats num_lab_techs
  h_uniforms h_lab_coats h_distribution ↦
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lab_techs_to_uniforms_ratio_l984_98440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_sales_ratio_l984_98424

/-- Proves the ratio of books sold on Wednesday to Tuesday is 3:1 --/
theorem bookstore_sales_ratio : 
  ∀ (tuesday wednesday thursday : ℕ),
  tuesday = 7 →
  thursday = 3 * wednesday →
  tuesday + wednesday + thursday = 91 →
  wednesday = 3 * tuesday :=
λ tuesday wednesday thursday h1 h2 h3 ↦
by
  sorry

#check bookstore_sales_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bookstore_sales_ratio_l984_98424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_distribution_equation_l984_98426

-- Define the total amount of silver
variable (x : ℝ)

-- Define the number of people in the group
noncomputable def num_people : ℝ := (x - 4) / 7

-- Theorem stating that the equation correctly represents the problem
theorem silver_distribution_equation (x : ℝ) :
  num_people x = (x + 8) / 9 :=
by
  -- The proof is omitted for now
  sorry

#check silver_distribution_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_silver_distribution_equation_l984_98426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_continuation_iff_equilateral_l984_98483

-- Define a triangle type
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a < b + c ∧ b < a + c ∧ c < a + b

-- Define the semi-perimeter
noncomputable def semiPerimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

-- Define the next triangle in the sequence
noncomputable def nextTriangle (t : Triangle) : Triangle where
  a := semiPerimeter t - t.a
  b := semiPerimeter t - t.b
  c := semiPerimeter t - t.c
  pos_a := by sorry
  pos_b := by sorry
  pos_c := by sorry
  triangle_ineq := by sorry

-- Define the property of indefinite continuation
def indefiniteContinuation (t : Triangle) : Prop :=
  ∀ n : ℕ, ∃ t' : Triangle, t' = (nextTriangle^[n]) t

-- The main theorem
theorem indefinite_continuation_iff_equilateral (t : Triangle) :
  indefiniteContinuation t ↔ t.a = t.b ∧ t.b = t.c := by
  sorry

#check indefinite_continuation_iff_equilateral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_continuation_iff_equilateral_l984_98483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_impossibility_l984_98445

theorem vector_inequality_impossibility
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (a b c : V) :
  ¬(Real.sqrt 3 * ‖a‖ < ‖b - c‖ ∧
    Real.sqrt 3 * ‖b‖ < ‖c - a‖ ∧
    Real.sqrt 3 * ‖c‖ < ‖a - b‖) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_inequality_impossibility_l984_98445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l984_98434

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  area : ℝ

/-- The given condition in the problem -/
def given_condition (t : Triangle) : Prop :=
  2 * t.a * (Real.cos (t.C / 2))^2 + 2 * t.c * (Real.cos (t.A / 2))^2 = 5/2 * t.b

/-- The first part of the theorem to prove -/
theorem part_one (t : Triangle) (h : given_condition t) : 2 * (t.a + t.c) = 3 * t.b := by
  sorry

/-- The second part of the theorem to prove -/
theorem part_two (t : Triangle) (h1 : given_condition t) (h2 : Real.cos t.B = 1/4) (h3 : t.area = Real.sqrt 15) : t.b = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l984_98434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l984_98479

open Real Set

/-- The function f defined on (0, π/2) -/
noncomputable def f (x : ℝ) : ℝ := 9 / (sin x)^2 + 4 / (cos x)^2

/-- The theorem stating the maximum value of t and the solution set -/
theorem inequality_problem :
  (∃ (t : ℝ), ∀ (x : ℝ), x ∈ Ioo 0 (π/2) → f x ≥ t ∧ 
    (∀ (t' : ℝ), (∀ (x : ℝ), x ∈ Ioo 0 (π/2) → f x ≥ t') → t' ≤ t) ∧ 
    t = 25) ∧
  {x : ℝ | |x + 5| + |2*x - 1| ≤ 6} = Icc 0 (2/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_problem_l984_98479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l984_98460

noncomputable def f (x : Real) : Real := Real.sin (Real.pi / 2 - x) * Real.sin x - Real.sqrt 3 * Real.sin x ^ 2

theorem f_properties :
  (∃ T : Real, T > 0 ∧ T = Real.pi ∧ ∀ x, f (x + T) = f x) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≥ (1 - Real.sqrt 3) / 2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x = (1 - Real.sqrt 3) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l984_98460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l984_98475

noncomputable def f (a x : ℝ) : ℝ := a * abs (2 - x) + (x^2 - x - 6) / (3 - x)

def A : Set ℝ := Set.Icc (-1) 1 ∪ {5}

theorem unique_solution (a : ℝ) :
  (∃! x : ℝ, x ≠ 3 ∧ f a x = 0) ↔ a ∈ A := by
  sorry

#check unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_l984_98475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l984_98455

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n, b (n + 1) = r * b n

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a d →
  a 1 = 2 →
  d ≠ 0 →
  geometric_sequence (λ n ↦ a (1 + 2 * (n - 1))) →
  sum_of_arithmetic_sequence a 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l984_98455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_l984_98425

theorem sum_of_factors (n m : ℕ) (h : m ≤ Nat.factorial n) : 
  ∃ (i : ℕ) (k : Fin i → ℕ), i ≤ n ∧ 
    (∀ j : Fin i, k j ∣ Nat.factorial n) ∧ 
    (∀ j₁ j₂ : Fin i, j₁ ≠ j₂ → k j₁ ≠ k j₂) ∧
    m = Finset.sum (Finset.univ : Finset (Fin i)) k :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_factors_l984_98425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_calculation_l984_98493

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 inches and 20 inches is 100√3 square inches. -/
theorem parallelogram_area_calculation (α : Real) (a b : Real) : 
  α = 150 * π / 180 → a = 10 → b = 20 → 
  a * b * Real.sin α = 100 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_area_calculation_l984_98493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_F_l984_98481

-- Define the original function F
noncomputable def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

-- Define the substitution function
noncomputable def subst (x : ℝ) : ℝ := (2 * x - x^3) / (1 + 2 * x^2)

-- Define the new function G
noncomputable def G (x : ℝ) : ℝ := F (subst x)

-- Theorem statement
theorem G_equals_F : ∀ x, G x = F x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_equals_F_l984_98481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_perfect_square_condition_converse_l984_98421

/-- A quadratic expression is a perfect square if and only if its discriminant is zero -/
def is_perfect_square (a b c : ℝ) : Prop :=
  b^2 - 4*a*c = 0

/-- The main theorem: if 9y^2 + my + 1/4 is a perfect square, then m = 3 or m = -3 -/
theorem perfect_square_condition (m : ℝ) :
  (∀ y : ℝ, is_perfect_square 9 m (1/4)) → m = 3 ∨ m = -3 :=
by
  sorry

/-- The converse: if m = 3 or m = -3, then 9y^2 + my + 1/4 is a perfect square -/
theorem perfect_square_condition_converse (m : ℝ) :
  m = 3 ∨ m = -3 → (∀ y : ℝ, is_perfect_square 9 m (1/4)) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perfect_square_condition_perfect_square_condition_converse_l984_98421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l984_98401

-- Define the power function as noncomputable
noncomputable def powerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ x^a

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) (a : ℝ) :
  f = powerFunction a →  -- f is a power function
  f 3 = 1/9 →            -- f passes through (3, 1/9)
  f 2 = 1/4 :=           -- prove that f(2) = 1/4
by
  -- The proof steps would go here, but for now we'll use sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_value_l984_98401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l984_98496

/-- Given a line y = mx + b, this function returns the distance between it and a parallel line y = mx + c -/
noncomputable def distance_between_parallel_lines (m b c : ℝ) : ℝ :=
  |c - b| / Real.sqrt (m^2 + 1)

/-- Represents a line in the form y = mx + b -/
structure Line where
  m : ℝ
  b : ℝ

theorem parallel_lines_at_distance (given : Line) (l1 l2 : Line) :
  given.m = 3/4 ∧ given.b = 6 ∧
  l1.m = 3/4 ∧ l1.b = 12.25 ∧
  l2.m = 3/4 ∧ l2.b = -0.25 →
  distance_between_parallel_lines given.m given.b l1.b = 5 ∧
  distance_between_parallel_lines given.m given.b l2.b = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_at_distance_l984_98496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_in_interval_l984_98471

-- Define the recursive function f
noncomputable def f : ℕ → ℝ
| 0 => 0  -- Base case for 0
| 1 => 0  -- Base case for 1
| 2 => 0  -- Base case for 2
| 3 => Real.log 3
| (n+1) => Real.log (n + 1 + f n)

-- Define A as f(2022)
noncomputable def A : ℝ := f 2022

-- Theorem statement
theorem A_in_interval : A > Real.log 2025 ∧ A < Real.log 2026 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_in_interval_l984_98471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_most_suitable_l984_98473

/-- Represents a sampling method -/
inductive SamplingMethod
| SimpleRandom
| Systematic
| Stratified

/-- Represents a population with subgroups having varying characteristics -/
structure Population where
  subgroups : List (Set ℕ)
  varying_characteristics : ∀ s, s ∈ subgroups → ∃ c : ℝ, True

/-- Represents the suitability of a sampling method for a given population -/
def suitability (method : SamplingMethod) (pop : Population) : ℕ :=
  match method with
  | SamplingMethod.SimpleRandom => 1
  | SamplingMethod.Systematic => 2
  | SamplingMethod.Stratified => 3

/-- The theorem stating that stratified sampling is the most suitable for a population with varying characteristics across subgroups -/
theorem stratified_most_suitable (pop : Population) :
  suitability SamplingMethod.Stratified pop = 
    (List.map (fun m => suitability m pop) [SamplingMethod.SimpleRandom, SamplingMethod.Systematic, SamplingMethod.Stratified]).maximum := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_most_suitable_l984_98473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_l984_98457

/-- The volume of a cone inscribed in a triangular pyramid -/
theorem inscribed_cone_volume (α : Real) (V : Real) :
  let V_k := V * Real.pi * (Real.tan (α / 2))^2 * (1 / Real.tan α)
  (0 < α) → (α < Real.pi) → (0 < V) →
  (∃ (pyramid : Real → Real → Real),
    ∃ (cone : Real → Real → Real),
    ∃ (base_triangle : Real → Real → Real),
    (pyramid α V = V) ∧
    (∃ (h : Real), base_triangle α V = h) ∧
    (∃ (r : Real), cone α V = r * base_triangle α V) ∧
    (cone α V = V_k)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cone_volume_l984_98457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_never_equals_dice_sum_l984_98411

-- Define the sum of two dice rolls
def diceSum : Set ℕ := {n : ℕ | 2 ≤ n ∧ n ≤ 12}

-- Define the area of a circle given its diameter
noncomputable def circleArea (d : ℕ) : ℝ := Real.pi * (d : ℝ)^2 / 4

-- Theorem statement
theorem circle_area_never_equals_dice_sum :
  ∀ s ∈ diceSum, circleArea s ≠ s := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_area_never_equals_dice_sum_l984_98411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_term_value_l984_98408

def my_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 1/4
  | n+1 => 1 - 1 / my_sequence n

theorem ninth_term_value : my_sequence 8 = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninth_term_value_l984_98408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_quadrilateral_area_l984_98417

/-- A convex quadrilateral -/
structure ConvexQuadrilateral where
  area : ℝ
  area_pos : area > 0

/-- A point inside a convex quadrilateral -/
structure InnerPoint (Q : ConvexQuadrilateral) where
  -- No specific properties needed for this problem

/-- The new quadrilateral formed by reflecting an inner point over the midpoints of the sides -/
def ReflectedQuadrilateral (Q : ConvexQuadrilateral) (P : InnerPoint Q) : ConvexQuadrilateral :=
  { area := 2 * Q.area
    area_pos := by
      apply mul_pos
      · exact two_pos
      · exact Q.area_pos }

/-- Theorem stating that the area of the reflected quadrilateral is twice the original area -/
theorem reflected_quadrilateral_area (Q : ConvexQuadrilateral) (P : InnerPoint Q) :
  (ReflectedQuadrilateral Q P).area = 2 * Q.area := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_reflected_quadrilateral_area_l984_98417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l984_98451

theorem system_solution :
  ∃ (k : ℤ), ∀ (x y : ℝ),
    (x^2 + 4 * Real.sin y^2 - 4 = 0 ∧
     Real.cos x - 2 * Real.cos y^2 - 1 = 0) →
    (x = 0 ∧ y = Real.pi/2 + Real.pi * ↑k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l984_98451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l984_98452

/-- The golden ratio -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- Given a line segment AB of length 10 with C as its golden section point where AC > BC,
    the length of AC is equal to 5√5 - 5 -/
theorem golden_section_length (A B C : ℝ) : 
  (B - A = 10) →                  -- AB = 10
  (C - A) / (B - C) = φ →         -- C is the golden section point
  (C - A > B - C) →               -- AC > BC
  (C - A = 5 * Real.sqrt 5 - 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_golden_section_length_l984_98452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_travel_time_l984_98437

/-- Represents the time it takes for a ship to travel between two ports under different conditions -/
structure ShipTravel where
  downstream_time : ℚ
  upstream_time : ℚ

/-- Calculates the time it takes for a ship to travel in still water conditions -/
def still_water_time (travel : ShipTravel) : ℚ :=
  (2 * travel.downstream_time * travel.upstream_time) / (travel.upstream_time - travel.downstream_time)

/-- Theorem stating that for a ship traveling downstream in 6 hours and upstream in 8 hours,
    the still water travel time is 48/7 hours -/
theorem ship_travel_time (travel : ShipTravel) 
    (h_downstream : travel.downstream_time = 6)
    (h_upstream : travel.upstream_time = 8) :
    still_water_time travel = 48 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ship_travel_time_l984_98437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l984_98464

-- Define the given values
noncomputable def train_speed_kmh : ℝ := 72
noncomputable def platform_length : ℝ := 250
noncomputable def train_length : ℝ := 50

-- Define the conversion factor from km/h to m/s
noncomputable def kmh_to_ms : ℝ := 5 / 18

-- Theorem statement
theorem train_crossing_time :
  let total_distance := platform_length + train_length
  let train_speed_ms := train_speed_kmh * kmh_to_ms
  total_distance / train_speed_ms = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_crossing_time_l984_98464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_trajectory_l984_98436

/-- The distance between two points in a plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The distance from a point to a vertical line -/
def distanceToVerticalLine (x _ lineX : ℝ) : ℝ :=
  |x - lineX|

/-- Theorem: If a point P(x,y) in the plane satisfies the condition that its distance
    to (1,0) is 2 units less than its distance to the line x = -3, then P lies on
    the parabola y² = 4x -/
theorem point_trajectory (x y : ℝ) :
  distance x y 1 0 = distanceToVerticalLine x y (-3) - 2 →
  y^2 = 4*x := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_trajectory_l984_98436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l984_98410

/-- The function f(x) = 1/x + a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 1/x + a * Real.log x

/-- Theorem stating the condition for the minimum value of f(x) on [1/2, 1] -/
theorem min_value_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≥ 0) ∧
  (∃ x ∈ Set.Icc (1/2 : ℝ) 1, f a x = 0) ↔
  a = 2 / Real.log 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_condition_l984_98410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_of_trees_l984_98406

def tree_heights (h : Fin 6 → ℝ) : Prop :=
  (h 1 = 18) ∧ 
  (h 3 = 54) ∧ 
  (∀ i : Fin 5, 
    (h i = 3 * h (i.succ) ∨ h i = h (i.succ) / 3))

theorem average_height_of_trees (h : Fin 6 → ℝ) 
  (hh : tree_heights h) : 
  (h 0 + h 1 + h 2 + h 3 + h 4 + h 5) / 6 = 26.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_height_of_trees_l984_98406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_motion_l984_98447

/-- A point moving along a straight line with constant acceleration -/
structure ConstantAccelerationMotion where
  a : ℝ  -- Constant acceleration
  v₀ : ℝ  -- Initial velocity
  s₀ : ℝ  -- Initial position

/-- The velocity of the point at time t -/
noncomputable def velocity (m : ConstantAccelerationMotion) (t : ℝ) : ℝ :=
  m.a * t + m.v₀

/-- The position of the point at time t -/
noncomputable def position (m : ConstantAccelerationMotion) (t : ℝ) : ℝ :=
  (1/2) * m.a * t^2 + m.v₀ * t + m.s₀

/-- The law of motion for a point with constant acceleration -/
theorem law_of_motion (m : ConstantAccelerationMotion) :
  ∀ t : ℝ,
    (velocity m t = m.a * t + m.v₀) ∧
    (position m t = (1/2) * m.a * t^2 + m.v₀ * t + m.s₀) := by
  intro t
  constructor
  . -- Prove velocity equation
    rfl
  . -- Prove position equation
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_law_of_motion_l984_98447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_l984_98454

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem floor_inequality (a b : ℝ) (ha : a = 3.5) (hb : b = 2.5) :
  floor (a + b) ≠ floor a + floor b + floor (a * b) - floor a * floor b := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_inequality_l984_98454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_bounds_l984_98474

open Real

-- Define the cyclic sum function
def cyclic_sum (f : ℝ → ℝ → ℝ → ℝ) : ℝ → ℝ → ℝ → ℝ :=
  λ x y z => f x y z + f y z x + f z x y

-- Define M as in the problem
noncomputable def M (x y z : ℝ) : ℝ :=
  cyclic_sum (λ a b c => (a^2 + a*b + b^2).sqrt * (b^2 + b*c + c^2).sqrt) x y z

-- State the theorem
theorem alpha_beta_bounds :
  ∃ (α_max β_min : ℝ),
    (∀ α β x y z : ℝ,
      α * (x*y + y*z + z*x) ≤ M x y z ∧
      M x y z ≤ β * (x^2 + y^2 + z^2) →
      α ≤ α_max ∧ β_min ≤ β) ∧
    α_max = 3 ∧ β_min = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_beta_bounds_l984_98474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_Al2CO33_approx_l984_98405

/-- The molar mass of aluminum in g/mol -/
noncomputable def molar_mass_Al : ℝ := 26.98

/-- The molar mass of carbon in g/mol -/
noncomputable def molar_mass_C : ℝ := 12.01

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of the carbonate group (CO3) in g/mol -/
noncomputable def molar_mass_CO3 : ℝ := molar_mass_C + 3 * molar_mass_O

/-- The molar mass of aluminum carbonate (Al2(CO3)3) in g/mol -/
noncomputable def molar_mass_Al2CO33 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_CO3

/-- The mass of aluminum in aluminum carbonate (Al2(CO3)3) in g/mol -/
noncomputable def mass_Al_in_Al2CO33 : ℝ := 2 * molar_mass_Al

/-- The mass percentage of aluminum in aluminum carbonate (Al2(CO3)3) -/
noncomputable def mass_percentage_Al_in_Al2CO33 : ℝ := (mass_Al_in_Al2CO33 / molar_mass_Al2CO33) * 100

theorem mass_percentage_Al_in_Al2CO33_approx :
  |mass_percentage_Al_in_Al2CO33 - 23.05| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_Al2CO33_approx_l984_98405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_l984_98403

theorem factorial_equation : 2^7 * 3^3 * 1050 = Nat.factorial 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_l984_98403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_divides_power_plus_one_l984_98404

theorem odd_integer_divides_power_plus_one (n : ℕ) :
  n ≠ 0 → Odd n ∧ (n ∣ 3^n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integer_divides_power_plus_one_l984_98404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consumption_income_ratio_l984_98431

/-- Represents the regression line equation y = 0.66x + 1.562 -/
def regression_line (x : ℝ) : ℝ := 0.66 * x + 1.562

/-- The given per capita consumption level -/
def consumption_level : ℝ := 7.675

/-- Theorem stating that the ratio of consumption to income is approximately 0.829 -/
theorem consumption_income_ratio :
  let x := (consumption_level - 1.562) / 0.66
  ∃ ε > 0, |consumption_level / x - 0.829| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_consumption_income_ratio_l984_98431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l984_98432

/-- Calculates the length of a platform given the length of a train, the time it takes to cross the platform, and the time it takes to cross a signal pole. -/
noncomputable def platform_length (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) : ℝ :=
  let train_speed := train_length / time_pole
  train_speed * time_platform - train_length

/-- Theorem stating that for a 300-meter train taking 51 seconds to cross a platform and 18 seconds to cross a signal pole, the platform length is approximately 550.17 meters. -/
theorem platform_length_calculation : 
  let train_length : ℝ := 300
  let time_platform : ℝ := 51
  let time_pole : ℝ := 18
  abs (platform_length train_length time_platform time_pole - 550.17) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_calculation_l984_98432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distribution_l984_98458

def book_distribution : List Nat := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

theorem valid_distribution :
  -- The list has 10 elements
  book_distribution.length = 10 ∧
  -- The sum of all elements is 100
  book_distribution.sum = 100 ∧
  -- All elements are distinct
  book_distribution.Nodup ∧
  -- Dividing any element into two positive integers results in a duplicate
  ∀ (i : Nat) (x y : Nat),
    i < book_distribution.length →
    x > 0 → y > 0 →
    x + y = book_distribution[i]! →
    ∃ (j : Nat), j < book_distribution.length + 1 ∧
      (x ∈ book_distribution ∨ y ∈ book_distribution ∨
       ∃ (k : Nat), k < book_distribution.length ∧ k ≠ i ∧
         (x = book_distribution[k]! ∨ y = book_distribution[k]!)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_distribution_l984_98458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_untouchable_area_of_tetrahedron_l984_98443

/-- The surface area of a regular tetrahedron with edge length a -/
noncomputable def tetrahedronSurfaceArea (a : ℝ) : ℝ := Real.sqrt 3 * a^2

/-- The edge length of the smaller tetrahedron that touches the sphere -/
noncomputable def smallerTetrahedronEdge : ℝ := 2 * Real.sqrt 6

/-- The edge length of the larger tetrahedron container -/
noncomputable def largerTetrahedronEdge : ℝ := 6 * Real.sqrt 6

/-- The theorem stating the area of the inner wall that the ball can never touch -/
theorem untouchable_area_of_tetrahedron : 
  tetrahedronSurfaceArea largerTetrahedronEdge - tetrahedronSurfaceArea smallerTetrahedronEdge = 192 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_untouchable_area_of_tetrahedron_l984_98443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_movie_tickets_l984_98499

theorem sara_movie_tickets : ℕ := by
  -- Define the cost of each movie theater ticket
  let ticket_cost : ℚ := 10.62

  -- Define the cost of renting a movie
  let rental_cost : ℚ := 1.59

  -- Define the cost of buying a movie
  let purchase_cost : ℚ := 13.95

  -- Define the total amount Sara spent on movies
  let total_spent : ℚ := 36.78

  -- Define the number of tickets Sara bought
  let num_tickets : ℕ := 2

  -- State and prove the theorem
  have : ticket_cost * num_tickets + rental_cost + purchase_cost = total_spent := by
    sorry

  exact num_tickets

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sara_movie_tickets_l984_98499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relation_l984_98476

/-- Two right circular cylinders with identical volumes -/
def CylinderVolumes (r₁ h₁ r₂ h₂ : ℝ) : Prop :=
  Real.pi * r₁^2 * h₁ = Real.pi * r₂^2 * h₂

/-- The radius of the second cylinder is 20% more than the radius of the first -/
def RadiusRelation (r₁ r₂ : ℝ) : Prop :=
  r₂ = 1.2 * r₁

theorem cylinder_height_relation
  (r₁ h₁ r₂ h₂ : ℝ)
  (hv : CylinderVolumes r₁ h₁ r₂ h₂)
  (hr : RadiusRelation r₁ r₂) :
  h₁ = 1.44 * h₂ :=
by
  sorry

#check cylinder_height_relation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_height_relation_l984_98476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_equals_four_l984_98412

theorem tan_plus_cot_equals_four (θ : ℝ) (h : Real.sin (2 * θ) = 1/2) :
  Real.tan θ + 1 / Real.tan θ = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_plus_cot_equals_four_l984_98412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_conditions_set_A_contains_neg_two_l984_98485

def A (x : ℝ) : Set ℝ := {3, x, x^2 - 2*x}

theorem set_A_conditions (x : ℝ) : 
  (∀ a b : ℝ, a ∈ A x → b ∈ A x → a ≠ b) → (x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 3) :=
by sorry

theorem set_A_contains_neg_two (x : ℝ) :
  -2 ∈ A x → x = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_set_A_conditions_set_A_contains_neg_two_l984_98485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_12_plus_a_l984_98423

theorem sin_pi_12_plus_a (a : ℝ) (h1 : Real.sin (π/3 + a) = 5/13) (h2 : π/6 < a ∧ a < 2*π/3) :
  Real.sin (π/12 + a) = 17 * Real.sqrt 2 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_12_plus_a_l984_98423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rain_duration_proof_l984_98487

/-- Calculates the duration of heavy rain given initial and final water volumes and flow rate. -/
noncomputable def rain_duration (initial_volume : ℝ) (final_volume : ℝ) (flow_rate : ℝ) : ℝ :=
  (final_volume - initial_volume) / flow_rate

/-- Proves that the duration of heavy rain is 90 minutes given the specified conditions. -/
theorem rain_duration_proof (initial_volume final_volume flow_rate : ℝ) 
    (h1 : initial_volume = 100)
    (h2 : final_volume = 280)
    (h3 : flow_rate = 2) : 
  rain_duration initial_volume final_volume flow_rate = 90 := by
  unfold rain_duration
  rw [h1, h2, h3]
  norm_num
  
-- Remove the #eval statement as it's not computable
-- #eval rain_duration 100 280 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rain_duration_proof_l984_98487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_l984_98450

noncomputable def move (z : ℂ) : ℂ := z * Complex.exp (Complex.I * (Real.pi / 6)) + Complex.I * 8

noncomputable def final_position (n : ℕ) : ℂ := 
  (move^[n]) (3 : ℂ)

theorem particle_position : 
  final_position 90 = Complex.mk (-3) (48 + 24 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_particle_position_l984_98450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_ratio_characterization_l984_98472

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- The statement to prove -/
theorem divisor_ratio_characterization :
  ∀ K : ℕ, K > 0 → (∃ n : ℕ, n > 0 ∧ (num_divisors (n^2) : ℚ) / (num_divisors n) = K) ↔ K % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_ratio_characterization_l984_98472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_planes_count_l984_98465

-- Define a type for points in 3D space
def Point3D := ℝ × ℝ × ℝ

-- Define a function to check if four points are coplanar
def are_coplanar (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define a function to check if a plane is equidistant from four points
def is_equidistant_plane (plane : Set Point3D) (p1 p2 p3 p4 : Point3D) : Prop := sorry

-- Define the main theorem
theorem equidistant_planes_count 
  (p1 p2 p3 p4 : Point3D) 
  (h : ¬ are_coplanar p1 p2 p3 p4) : 
  ∃ (planes : Finset (Set Point3D)), 
    (∀ plane ∈ planes, is_equidistant_plane plane p1 p2 p3 p4) ∧ 
    (Finset.card planes = 4) := by
  sorry

#check equidistant_planes_count

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_planes_count_l984_98465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l984_98407

noncomputable def complex_number : ℂ := (2 * Complex.I) / (1 + Complex.I) * Complex.I

theorem complex_number_in_first_quadrant :
  Complex.re complex_number > 0 ∧ Complex.im complex_number > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_in_first_quadrant_l984_98407
