import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_S_equals_nine_l484_48459

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![Real.cos θ, -Real.sin θ],
    ![Real.sin θ,  Real.cos θ]]

def scaling_matrix (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![k, 0],
    ![0, k]]

noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ :=
  rotation_matrix (Real.pi/4) * scaling_matrix 3

theorem det_S_equals_nine : 
  det S = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_S_equals_nine_l484_48459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_month_is_june_l484_48411

/-- Represents months of the year --/
inductive Month
| January
| February
| March
| April
| May
| June
| July
| August
| September
| October
| November
| December

/-- Returns the month that is n months after the given month --/
def monthsLater (m : Month) (n : Nat) : Month :=
  match n with
  | 0 => m
  | n + 1 => monthsLater (match m with
    | Month.January => Month.February
    | Month.February => Month.March
    | Month.March => Month.April
    | Month.April => Month.May
    | Month.May => Month.June
    | Month.June => Month.July
    | Month.July => Month.August
    | Month.August => Month.September
    | Month.September => Month.October
    | Month.October => Month.November
    | Month.November => Month.December
    | Month.December => Month.January
  ) n

theorem exam_month_is_june 
  (start_month : Month)
  (preparation_duration : Nat)
  (h1 : start_month = Month.January)
  (h2 : preparation_duration = 5) :
  monthsLater start_month preparation_duration = Month.June := by
  sorry

-- Remove the #eval line as it's causing issues

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exam_month_is_june_l484_48411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_count_l484_48417

/-- The number of ways to arrange 7 people (3 athletes A, B, C, and 4 volunteers) in a line,
    where A and B must be adjacent and C cannot be at either end. -/
def arrangement_count : ℕ := 960

/-- The number of people in the arrangement -/
def total_people : ℕ := 7

/-- The number of volunteers -/
def volunteer_count : ℕ := 4

/-- The number of athletes -/
def athlete_count : ℕ := 3

theorem correct_arrangement_count :
  arrangement_count = 
    (Nat.factorial (total_people - 1)) *    -- Permutations of 6 elements (AB treated as one)
    (total_people - 3) *     -- Number of positions for C (not at ends)
    2                        -- Ways to arrange A and B within their pair
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_arrangement_count_l484_48417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_average_l484_48464

/-- The average of the two real solutions of a quadratic equation ax^2 - 2ax + b = 0 is 1 -/
theorem quadratic_solution_average (a b : ℝ) (ha : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 - 2*a*x + b
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 → (x₁ + x₂) / 2 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_solution_average_l484_48464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l484_48462

theorem max_third_side_length (A B C : Real) (a b c : Real) :
  A + B + C = π →
  C = 5*π/6 →
  a = 8 →
  b = 15 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = Real.sqrt (289 + 120 * Real.sqrt 3) :=
by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_third_side_length_l484_48462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_chase_l484_48493

/-- Tom's chase function: given initial position (x, y) and time t, 
    returns Tom's position at time t -/
noncomputable def tom_position (x y : ℝ) (t : ℕ) : ℝ × ℝ := sorry

/-- Jerry's position at time t -/
def jerry_position (t : ℕ) : ℝ × ℝ := (t, 0)

/-- Predicate that Tom catches Jerry at time n -/
def catches_at (x y : ℝ) (n : ℕ) : Prop := 
  tom_position x y n = jerry_position n

/-- Theorem stating the conditions and conclusion of the chase problem -/
theorem tom_chase (x y : ℝ) :
  (∃ n : ℕ, catches_at x y n) → 
  (x ≥ 0 ∧ ∃ a : ℝ, a = Real.sqrt 3 / 3 ∧ 
    ∀ x' y' : ℝ, (∃ n' : ℕ, catches_at x' y' n') → y' / (x' + 1) ≤ a) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tom_chase_l484_48493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_discount_approx_l484_48404

noncomputable def bag1_marked : ℝ := 240
noncomputable def bag1_sold : ℝ := 120
noncomputable def bag2_marked : ℝ := 360
noncomputable def bag2_sold : ℝ := 270
noncomputable def bag3_marked : ℝ := 480
noncomputable def bag3_sold : ℝ := 384

noncomputable def discount_rate (marked : ℝ) (sold : ℝ) : ℝ :=
  (marked - sold) / marked * 100

noncomputable def average_discount : ℝ :=
  (discount_rate bag1_marked bag1_sold +
   discount_rate bag2_marked bag2_sold +
   discount_rate bag3_marked bag3_sold) / 3

theorem average_discount_approx :
  ∃ ε > 0, |average_discount - 31.67| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_discount_approx_l484_48404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixing_ratio_theorem_l484_48482

/-- Represents the ratio of water, oil, and vinegar in a bottle -/
structure LiquidRatio where
  water : ℚ
  oil : ℚ
  vinegar : ℚ

/-- Calculates the final ratio of liquids after mixing two bottles -/
def mixBottles (bottle1 : LiquidRatio) (bottle2 : LiquidRatio) : LiquidRatio :=
  { water := bottle1.water + bottle2.water,
    oil := bottle1.oil + bottle2.oil,
    vinegar := bottle1.vinegar + bottle2.vinegar }

theorem mixing_ratio_theorem (M : ℚ) :
  let bottle1 : LiquidRatio := { water := M / 6, oil := M / 3, vinegar := M / 2 }
  let bottle2 : LiquidRatio := { water := M / 4, oil := M / 3, vinegar := 5 * M / 12 }
  let mixed := mixBottles bottle1 bottle2
  (mixed.water : ℚ) / (mixed.water + mixed.oil + mixed.vinegar) = 5 / 24 ∧
  (mixed.oil : ℚ) / (mixed.water + mixed.oil + mixed.vinegar) = 8 / 24 ∧
  (mixed.vinegar : ℚ) / (mixed.water + mixed.oil + mixed.vinegar) = 11 / 24 := by
  sorry

#check mixing_ratio_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixing_ratio_theorem_l484_48482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_letter_probability_l484_48495

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

noncomputable def derangement (n : ℕ) : ℚ := 
  ↑(factorial n) * (List.range (n+1)).foldl (fun acc i => acc + ((-1)^i : ℚ) / (factorial i : ℚ)) 0

def choose (n k : ℕ) : ℕ := 
  factorial n / (factorial k * factorial (n - k))

theorem exactly_one_correct_letter_probability : 
  let n : ℕ := 6
  let total_distributions := factorial n
  let ways_one_correct := ↑(choose n 1) * derangement (n - 1)
  ways_one_correct / total_distributions = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_correct_letter_probability_l484_48495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sum_squares_l484_48445

theorem quadratic_root_sum_squares (h : ℝ) : 
  (∃ x y : ℝ, x^2 + 4*h*x = 5 ∧ y^2 + 4*h*y = 5 ∧ x^2 + y^2 = 13) → 
  |h| = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_root_sum_squares_l484_48445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l484_48442

theorem remainder_theorem (x : ℕ) (h : (7 * x) % 31 = 1) : (14 + x) % 31 = 23 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_theorem_l484_48442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l484_48425

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x

-- State the theorem
theorem tangent_line_implies_a_value (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ deriv (f a) x = 0) → a = -Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l484_48425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roots_of_unity_sum_l484_48400

noncomputable def x : ℂ := (-1 + Complex.I * Real.sqrt 3) / 2
noncomputable def y : ℂ := (-1 - Complex.I * Real.sqrt 3) / 2

theorem cube_roots_of_unity_sum : x^12 + y^12 ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_roots_of_unity_sum_l484_48400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l484_48485

def z : ℂ := 1 - Complex.I

theorem complex_fraction_equality : (z^2 - 2*z) / (z - 1) = 2*Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l484_48485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_area_is_175_l484_48431

/-- Represents the properties of tiles on a rectangular wall -/
structure TileWall where
  regularTileArea : ℝ
  jumboTileProportion : ℝ
  jumboTileLengthRatio : ℝ
  regularTileCoverage : ℝ

/-- Calculates the total area of a wall covered with regular and jumbo tiles -/
noncomputable def totalWallArea (wall : TileWall) : ℝ :=
  wall.regularTileCoverage / (1 - wall.jumboTileProportion)

/-- Theorem stating that the total area of the wall is 175 square feet -/
theorem wall_area_is_175 (wall : TileWall)
  (h1 : wall.regularTileArea > 0)
  (h2 : wall.jumboTileProportion = 1/3)
  (h3 : wall.jumboTileLengthRatio = 3)
  (h4 : wall.regularTileCoverage = 70) :
  totalWallArea wall = 175 := by
  sorry

#check wall_area_is_175

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wall_area_is_175_l484_48431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_function_and_point_l484_48432

/-- Given that y is directly proportional to x+2 and y=-18 when x=4, prove the functional relationship and point location. -/
theorem proportional_function_and_point : 
  ∃ k : ℝ, 
    (∀ x : ℝ, ∃ y : ℝ, y = k * (x + 2)) ∧  -- y is directly proportional to x+2
    (∃ y : ℝ, y = k * (4 + 2) ∧ y = -18) ∧ -- when x=4, y=-18
    (∀ x : ℝ, k * (x + 2) = -3 * x - 6) ∧  -- functional relationship
    k * (7 + 2) ≠ -25                      -- P(7,-25) not on the graph
  := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proportional_function_and_point_l484_48432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_likes_two_digits_l484_48466

def is_even_digit (d : ℕ) : Prop := d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 6 ∨ d = 8

def can_be_divisible_by_three (d : ℕ) : Prop :=
  ∃ n : ℕ, (10 * n + d) % 3 = 0

def julia_likes_digit (d : ℕ) : Prop :=
  is_even_digit d ∧ can_be_divisible_by_three d

theorem julia_likes_two_digits :
  ∃! (s : Finset ℕ), (∀ d ∈ s, julia_likes_digit d) ∧ s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_julia_likes_two_digits_l484_48466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_in_equilateral_triangle_l484_48412

/-- Represents a point in barycentric coordinates -/
structure BarycentricPoint where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the area of a triangle given three points in barycentric coordinates -/
noncomputable def triangleArea (p1 p2 p3 : BarycentricPoint) : ℝ :=
  (1/2) * abs (p1.x * (p2.y * p3.z - p3.y * p2.z) +
               p2.x * (p3.y * p1.z - p1.y * p3.z) +
               p3.x * (p1.y * p2.z - p2.y * p1.z))

/-- Theorem: Area of trapezoid BB''C''C in equilateral triangle ABC -/
theorem area_of_trapezoid_in_equilateral_triangle (A B C : BarycentricPoint)
  (A' B' C' : BarycentricPoint) (A'' B'' C'' : BarycentricPoint) :
  -- Given conditions
  (A.x = 0 ∧ A.y = 0 ∧ A.z = 1) →  -- A is (0, 0, 1)
  (B.x = 1 ∧ B.y = 0 ∧ B.z = 0) →  -- B is (1, 0, 0)
  (C.x = 0 ∧ C.y = 1 ∧ C.z = 0) →  -- C is (0, 1, 0)
  (triangleArea A B C = 1) →  -- Area of ABC is 1
  -- Midpoints
  (A'.x = 0 ∧ A'.y = 1/2 ∧ A'.z = 1/2) →
  (B'.x = 1/2 ∧ B'.y = 0 ∧ B'.z = 1/2) →
  (C'.x = 1/2 ∧ C'.y = 1/2 ∧ C'.z = 0) →
  (A''.x = 1/4 ∧ A''.y = 1/4 ∧ A''.z = 1/2) →
  (B''.x = 1/4 ∧ B''.y = 1/2 ∧ B''.z = 1/4) →
  (C''.x = 1/2 ∧ C''.y = 1/4 ∧ C''.z = 1/4) →
  -- Conclusion
  triangleArea B B'' C + triangleArea B'' C'' C = 9/32 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_trapezoid_in_equilateral_triangle_l484_48412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nm_length_constant_l484_48415

/-- A circle with a chord AB and points on it -/
structure CircleWithChord where
  circle : Set (EuclideanSpace ℝ (Fin 2))
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  W : EuclideanSpace ℝ (Fin 2)
  is_chord : A ∈ circle ∧ B ∈ circle
  W_is_midpoint : W ∈ circle

/-- Configuration of points for the problem -/
structure PointConfiguration (cw : CircleWithChord) where
  C : EuclideanSpace ℝ (Fin 2)
  X : EuclideanSpace ℝ (Fin 2)
  Y : EuclideanSpace ℝ (Fin 2)
  N : EuclideanSpace ℝ (Fin 2)
  M : EuclideanSpace ℝ (Fin 2)
  C_on_major_arc : C ∈ cw.circle
  X_on_tangents : sorry
  Y_on_tangents : sorry
  N_on_intersection : sorry
  M_on_intersection : sorry

/-- The main theorem -/
theorem nm_length_constant (cw : CircleWithChord) (pc : PointConfiguration cw) :
  ‖pc.N - pc.M‖ = ‖cw.A - cw.B‖ / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nm_length_constant_l484_48415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_1600_approx_l484_48426

-- Define the sine values for the given angles
def sin_10 : ℝ := 0.1736
def sin_20 : ℝ := 0.3420
def sin_30 : ℝ := 0.5000
def sin_40 : ℝ := 0.6427
def sin_50 : ℝ := 0.7660
def sin_60 : ℝ := 0.8660
def sin_70 : ℝ := 0.9397
def sin_80 : ℝ := 0.9848

-- Define a function to represent the tangent of an angle
noncomputable def tan (θ : ℝ) : ℝ := Real.tan θ

-- State the theorem
theorem tan_1600_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.005 ∧ |tan (1600 * π / 180) + 0.36| < ε :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_1600_approx_l484_48426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_of_set_X_l484_48402

def set_X (n : ℕ) := Finset.range n
def set_Y := Finset.range 21

theorem ending_number_of_set_X (n : ℕ) :
  (∃ (S : Finset ℕ), S ⊆ set_X n ∧ S ⊆ set_Y ∧ S.card = 12) →
  n = 12 :=
by
  intro h
  sorry

#check ending_number_of_set_X

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ending_number_of_set_X_l484_48402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l484_48461

/-- The line l is defined by the equation x - y + 1 = 0 -/
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The shortest path from (0,0) to (1,1) via a point on line l -/
theorem shortest_path : ∃ (x y : ℝ), 
  line_l x y ∧ 
  distance 0 0 x y + distance x y 1 1 = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_path_l484_48461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_revolution_difference_l484_48492

/-- A cart with two wheels of different circumferences -/
structure Cart where
  front_circumference : ℚ
  back_circumference : ℚ
  total_distance : ℚ

/-- Calculate the number of revolutions for a wheel -/
def revolutions (circumference : ℚ) (distance : ℚ) : ℚ :=
  distance / circumference

/-- The difference in revolutions between front and back wheels -/
def revolution_difference (cart : Cart) : ℚ :=
  revolutions cart.front_circumference cart.total_distance - 
  revolutions cart.back_circumference cart.total_distance

/-- Theorem stating the difference in revolutions for the given cart -/
theorem cart_revolution_difference :
  ∃ (cart : Cart),
    cart.front_circumference = 30 ∧
    cart.back_circumference = 32 ∧
    cart.total_distance = 2400 ∧
    revolution_difference cart = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_revolution_difference_l484_48492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_diagonal_and_circle_chord_l484_48456

-- Part 1: Cube diagonal
def cube_edge_length : ℕ := 90
def cuboid_lengths : List ℕ := [2, 3, 5]

-- Part 2: Circle and chord
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 24*x - 28*y - 36 = 0
def point_Q : ℝ × ℝ := (4, 2)

theorem cube_diagonal_and_circle_chord :
  -- Part 1: Cube diagonal
  (∃ n : ℕ, n = 66 ∧ n * (Nat.lcm (List.get! cuboid_lengths 0) (List.get! cuboid_lengths 1)) = cube_edge_length) ∧
  -- Part 2: Circle and chord
  (∀ x y : ℝ, (∃ a b : ℝ, 
    circle_equation a b ∧
    circle_equation x y ∧
    (a - point_Q.1) * (x - point_Q.1) + (b - point_Q.2) * (y - point_Q.2) = 0) →
    (x - 8)^2 + (y - 8)^2 = 136) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_diagonal_and_circle_chord_l484_48456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l484_48473

/-- Given an ellipse with foci F₁ and F₂, and point A on the ellipse, prove that
    under certain conditions, the equation of the ellipse is x²/3 + y²/2 = 1 -/
theorem ellipse_equation (a b c : ℝ) (A B F₁ F₂ : ℝ × ℝ) :
  a > 0 → b > 0 → a > b →
  (∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ (x, y) ∈ Set.range (fun t ↦ (a * Real.cos t, b * Real.sin t))) →
  F₁ = (-c, 0) →
  F₂ = (c, 0) →
  A = (0, b) →
  B.1^2/a^2 + B.2^2/b^2 = 1 →
  F₂ = 2 • F₁ →
  (F₁.1 - A.1, F₁.2 - A.2) • (B.1 - A.1, B.2 - A.2) = 3/2 →
  ∃ x y : ℝ, x^2/3 + y^2/2 = 1 ↔ (x, y) ∈ Set.range (fun t ↦ (a * Real.cos t, b * Real.sin t)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l484_48473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l484_48475

-- Define the total number of votes
variable (total_votes : ℕ)

-- Define the winning percentage
def winning_percentage : ℚ := 70 / 100

-- Define the vote majority
def vote_majority : ℕ := 360

-- Theorem statement
theorem election_votes :
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = vote_majority →
  total_votes = 900 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_election_votes_l484_48475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_simultaneous_meeting_l484_48414

/-- Represents a car on the circular track -/
structure Car where
  speed : ℚ  -- Speed in laps per minute
  clockwise : Bool  -- Direction of movement

/-- The circular race track -/
structure Track where
  cars : Finset Car
  carCount : cars.card = 4

/-- Time when two cars meet -/
noncomputable def meetTime (c1 c2 : Car) : ℚ :=
  if c1.clockwise = c2.clockwise
  then 1 / |c1.speed - c2.speed|
  else 1 / (c1.speed + c2.speed)

/-- Theorem: First simultaneous meeting of all cars -/
theorem first_simultaneous_meeting
  (track : Track)
  (a b c d : Car)
  (h1 : a ∈ track.cars)
  (h2 : b ∈ track.cars)
  (h3 : c ∈ track.cars)
  (h4 : d ∈ track.cars)
  (h5 : a.clockwise ∧ b.clockwise ∧ ¬c.clockwise ∧ ¬d.clockwise)
  (h6 : meetTime a c = 7)
  (h7 : meetTime b d = 7)
  (h8 : meetTime a b = 53)
  (h9 : a.speed ≠ b.speed ∧ a.speed ≠ c.speed ∧ a.speed ≠ d.speed ∧
        b.speed ≠ c.speed ∧ b.speed ≠ d.speed ∧ c.speed ≠ d.speed) :
  ∃ t : ℕ, t = 371 ∧ t % (meetTime a c).num = 0 ∧
             t % (meetTime b d).num = 0 ∧
             t % (meetTime a b).num = 0 ∧
             t % (meetTime c d).num = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_simultaneous_meeting_l484_48414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_distance_l484_48470

/-- Calculates the remaining distance between two people walking towards each other. -/
noncomputable def remaining_distance (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  total_distance - (speed1 + speed2 * (1 / 60)) * time

theorem meeting_distance :
  remaining_distance 2.5 0.08 2.4 15 = 0.7 := by
  -- Unfold the definition of remaining_distance
  unfold remaining_distance
  -- Simplify the expression
  simp
  -- The proof is completed numerically
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_meeting_distance_l484_48470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_closest_value_l484_48427

theorem sqrt_difference_closest_value :
  let diff := Real.sqrt 126 - Real.sqrt 121
  ∀ x ∈ ({0.19, 0.21, 0.25, 0.27} : Set ℝ), |diff - 0.23| < |diff - x| :=
by
  -- Introduce the local definition of diff
  intro diff
  -- Introduce the arbitrary element x from the set
  intro x hx
  -- The actual proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_difference_closest_value_l484_48427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_C₃_l484_48449

noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (-4 + Real.cos t, -3 + Real.sin t)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := (8 * Real.cos θ, -3 * Real.sin θ)

noncomputable def C₃ (t : ℝ) : ℝ × ℝ := (3 + 2*t, -2 + t)

noncomputable def P : ℝ × ℝ := C₁ (Real.pi / 2)

noncomputable def M (θ : ℝ) : ℝ × ℝ := (
  (P.1 + (C₂ θ).1) / 2,
  (P.2 + (C₂ θ).2) / 2
)

noncomputable def distance_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - 2*p.2 - 7| / Real.sqrt 5

theorem min_distance_to_C₃ :
  ∃ θ : ℝ, ∀ φ : ℝ, distance_to_line (M θ) ≤ distance_to_line (M φ) ∧
  distance_to_line (M θ) = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_C₃_l484_48449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_deg_sum_l484_48401

theorem tan_22_5_deg_sum (a b c d : ℕ) :
  (a ≥ b) → (b ≥ c) → (c ≥ d) → (a > 0) → (b > 0) → (c > 0) → (d > 0) →
  (Real.tan (22.5 * π / 180) = Real.sqrt (a : ℝ) - Real.sqrt (b : ℝ) + Real.sqrt (c : ℝ) - (d : ℝ)) →
  a + b + c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_22_5_deg_sum_l484_48401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dimes_for_sneakers_l484_48440

/-- The minimum number of dimes needed to afford sneakers -/
noncomputable def min_dimes_needed (sneaker_cost : ℚ) (bills_10 : ℕ) (quarters : ℕ) : ℕ :=
  (((sneaker_cost - (bills_10 * 10 + quarters * (1/4))) / (1/10)).ceil).toNat

/-- Theorem stating the minimum number of dimes needed in the given scenario -/
theorem min_dimes_for_sneakers :
  min_dimes_needed 45.35 3 4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dimes_for_sneakers_l484_48440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l484_48472

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem angle_between_vectors (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_norm : ‖a‖ = ‖b‖ ∧ ‖a‖ = ‖a - b‖) : 
  Real.arccos ((inner a (a + b)) / (‖a‖ * ‖a + b‖)) = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_vectors_l484_48472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survival_equation_l484_48458

/-- The probability of survival for one month -/
noncomputable def survival_prob : ℝ := 9/10

/-- The initial number of animals -/
def initial_count : ℝ := 200

/-- The expected number of survivors -/
def expected_survivors : ℝ := 145.8

/-- The number of months -/
noncomputable def months : ℝ := Real.log (expected_survivors / initial_count) / Real.log survival_prob

theorem survival_equation :
  initial_count * survival_prob ^ months = expected_survivors := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survival_equation_l484_48458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_point_and_line_l484_48423

/-- The line described by the parametric equations -/
def line_eq (x y z : ℝ) : Prop :=
  (x - 2) / 4 = (y - 4) / (-3) ∧ (y - 4) / (-3) = (z + 3) / 2

/-- The point through which the plane passes -/
def point : ℝ × ℝ × ℝ := (1, 9, -8)

/-- The equation of the plane -/
def plane_eq (x y z : ℝ) : Prop :=
  75 * x - 29 * y + 86 * z + 274 = 0

/-- Theorem stating that the plane containing the given line and point
    has the specified equation -/
theorem plane_through_point_and_line :
  ∀ x y z : ℝ, line_eq x y z →
  plane_eq x y z ∧
  plane_eq point.1 point.2.1 point.2.2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_through_point_and_line_l484_48423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l484_48443

theorem train_speed_calculation (goods_train_length : ℝ) 
                                (goods_train_speed_kmh : ℝ) 
                                (passing_time : ℝ) : ℝ := by
  have h1 : goods_train_length = 280 := by sorry
  have h2 : goods_train_speed_kmh = 62 := by sorry
  have h3 : passing_time = 9 := by sorry
  
  let goods_train_speed_ms := goods_train_speed_kmh * 1000 / 3600
  let relative_speed := goods_train_length / passing_time
  let mans_train_speed_ms := relative_speed - goods_train_speed_ms
  let mans_train_speed_kmh := mans_train_speed_ms * 3600 / 1000
  
  have h4 : mans_train_speed_kmh = 50 := by sorry
  
  exact mans_train_speed_kmh


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l484_48443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trigonometry_l484_48490

theorem angle_trigonometry (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : Real.sin α = 3/5) 
  (h4 : Real.tan (α - β) = 1/3) : 
  (Real.sin (α - β) = -Real.sqrt 10 / 10) ∧ 
  (Real.cos β = 9 * Real.sqrt 10 / 50) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_trigonometry_l484_48490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_15_to_3157_l484_48463

theorem closest_multiple_of_15_to_3157 :
  ∀ n : ℤ, n % 15 = 0 → |n - 3157| ≥ |3150 - 3157| :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_multiple_of_15_to_3157_l484_48463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l484_48437

theorem no_such_function_exists :
  ∀ f : ℝ → ℝ, ∃ x y : ℝ, y > x ∧ f y ≤ (y - x) * (f x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l484_48437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_implies_a_value_l484_48403

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(-x)

noncomputable def f_inverse (a : ℝ) (y : ℝ) : ℝ := -Real.log y / Real.log a

theorem inverse_function_point_implies_a_value (a : ℝ) (h : a > 0) :
  (f_inverse a (1/2) = 1) → a = 2 := by
  intro h_inverse
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_point_implies_a_value_l484_48403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_constellation_divisors_l484_48447

/-- Represents a stellar classification --/
inductive StellarClass
| class1 | class2 | class3 | class4 | class5 | class6 | class7

/-- Represents a star in the constellation --/
structure ConstellationStar where
  classification : StellarClass

/-- Represents a connection between two stars --/
structure Connection where
  star1 : ConstellationStar
  star2 : ConstellationStar

/-- Represents the constellation Leo --/
structure Constellation where
  stars : Finset ConstellationStar
  connections : Finset Connection

/-- Checks if a constellation is valid according to the problem's conditions --/
def is_valid_constellation (c : Constellation) : Prop :=
  c.stars.card = 9 ∧
  c.connections.card = 10 ∧
  ∀ conn ∈ c.connections, conn.star1.classification ≠ conn.star2.classification

/-- Counts the number of valid stellar classifications for the constellation --/
noncomputable def count_valid_classifications (c : Constellation) : ℕ :=
  sorry -- Implementation details omitted

/-- Counts the number of positive integer divisors of a natural number --/
def count_divisors (n : ℕ) : ℕ :=
  sorry -- Implementation details omitted

theorem leo_constellation_divisors :
  ∀ c : Constellation, is_valid_constellation c →
    count_divisors (count_valid_classifications c) = 160 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_leo_constellation_divisors_l484_48447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l484_48416

/-- A hyperbola with center O and foci F₁ and F₂ -/
structure Hyperbola where
  O : ℝ × ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- Point on a hyperbola -/
def PointOnHyperbola (h : Hyperbola) (P : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), |P.1 - h.F₁.1| - |P.1 - h.F₂.1| = k ∧ 
             |P.2 - h.F₁.2| - |P.2 - h.F₂.2| = k

/-- Distance between two points -/
noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem hyperbola_property (h : Hyperbola) (P : ℝ × ℝ) 
  (on_hyperbola : PointOnHyperbola h P)
  (focal_product : distance P h.F₁ * distance P h.F₂ = 6) :
  distance P h.O = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l484_48416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sphere_volume_is_8pi_over_3_l484_48413

/-- A right triangular pyramid with specific properties -/
structure RightTriangularPyramid where
  height : ℝ
  base_area : ℝ
  lateral_face_area : ℝ
  height_is_10 : height = 10
  base_area_relation : base_area = lateral_face_area / 3

/-- A sphere in the sequence of spheres inside the pyramid -/
structure Sphere (p : RightTriangularPyramid) where
  radius : ℝ
  touches_lateral_faces : Bool

/-- The sequence of spheres inside the pyramid -/
noncomputable def sphere_sequence (p : RightTriangularPyramid) : ℕ → Sphere p
  | 0 => { radius := 1, touches_lateral_faces := true }  -- S₁
  | n + 1 => { radius := 0.5 * (sphere_sequence p n).radius, touches_lateral_faces := true }  -- Sₙ₊₁

/-- The volume of a single sphere -/
noncomputable def sphere_volume (p : RightTriangularPyramid) (s : Sphere p) : ℝ := 
  (4 / 3) * Real.pi * s.radius ^ 3

/-- The total volume of the infinite sequence of spheres -/
noncomputable def total_sphere_volume (p : RightTriangularPyramid) : ℝ :=
  ∑' n, sphere_volume p (sphere_sequence p n)

/-- The main theorem stating the total volume of spheres -/
theorem total_sphere_volume_is_8pi_over_3 (p : RightTriangularPyramid) :
  total_sphere_volume p = (8 / 3) * Real.pi := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sphere_volume_is_8pi_over_3_l484_48413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cos_A_l484_48408

theorem right_triangle_cos_A (AB BC : ℝ) (h_right : AB^2 + BC^2 = (AB^2 + BC^2)) 
  (h_AB : AB = 8) (h_BC : BC = 12) : 
  Real.cos (Real.arccos (BC / Real.sqrt (AB^2 + BC^2))) = (3 * Real.sqrt 13) / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_cos_A_l484_48408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l484_48457

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in a 2D plane represented by ax + by = c --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between a point and a line --/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  |l.a * p.1 + l.b * p.2 - l.c| / Real.sqrt (l.a^2 + l.b^2)

/-- Predicate to check if a line is tangent to a circle --/
def isTangent (l : Line) (c : Circle) : Prop :=
  distancePointToLine c.center l = c.radius

/-- The main theorem --/
theorem tangent_line_to_circle (b : ℝ) :
  let l : Line := { a := 6, b := 8, c := b }
  let c : Circle := { center := (1, 1), radius := 1 }
  isTangent l c → b = 4 ∨ b = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l484_48457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l484_48476

/-- A cubic polynomial Q with coefficients in ℝ -/
def CubicPolynomial (m : ℝ) : Type := {Q : ℝ → ℝ // ∃ (a b c : ℝ), ∀ x, Q x = a * x^3 + b * x^2 + c * x + 2 * m}

theorem cubic_polynomial_sum (m : ℝ) (Q : CubicPolynomial m) :
  (Q.val 1 = 5 * m) → (Q.val (-1) = 7 * m) → (Q.val 2 + Q.val (-2) = 36 * m) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_sum_l484_48476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_pure_imaginary_for_negative_k_l484_48436

-- Define the complex quadratic equation
def complex_quadratic (z : ℂ) (k : ℝ) : ℂ := 10 * z^2 - 3 * Complex.I * z - k

-- State the theorem
theorem roots_pure_imaginary_for_negative_k :
  ∀ (k : ℝ), k < 0 →
  ∃ (z₁ z₂ : ℂ), complex_quadratic z₁ k = 0 ∧ complex_quadratic z₂ k = 0 ∧
                 z₁.re = 0 ∧ z₂.re = 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_pure_imaginary_for_negative_k_l484_48436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_of_hyperbola_l484_48486

-- Define the parameters and equations
axiom a : ℝ
axiom b : ℝ
axiom a_gt_b : a > b
axiom b_gt_0 : b > 0

-- Define the eccentricities
noncomputable def e1 : ℝ := Real.sqrt (1 - b^2 / a^2)
noncomputable def e2 : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Define the product of eccentricities
axiom eccentricity_product : e1 * e2 = Real.sqrt 3 / 2

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := x = Real.sqrt 2 * y ∨ x = -Real.sqrt 2 * y

-- Theorem statement
theorem asymptotes_of_hyperbola :
  ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → asymptote x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptotes_of_hyperbola_l484_48486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l484_48497

theorem min_abs_difference (a b : ℕ) (h : a * b - 4 * a + 3 * b = 221) :
  ∃ (a' b' : ℕ), a' * b' - 4 * a' + 3 * b' = 221 ∧
  ∀ (x y : ℕ), x * y - 4 * x + 3 * y = 221 →
  |Int.ofNat a' - Int.ofNat b'| ≤ |Int.ofNat x - Int.ofNat y| ∧ |Int.ofNat a' - Int.ofNat b'| = 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l484_48497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_odd_l484_48444

def sequence_a : ℕ → ℤ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => 7
  | (n + 3) => 3 * sequence_a (n + 2) + 2 * sequence_a (n + 1)  -- Defined recursively based on the problem

theorem sequence_a_odd (n : ℕ) (h : n > 1) : 
  ∃ k : ℤ, sequence_a n = 2 * k + 1 :=
by
  sorry

axiom sequence_a_property (n : ℕ) (h : n ≥ 2) : 
  -1/2 < (sequence_a (n + 1) : ℚ) - (sequence_a n)^2 / (sequence_a (n - 1)) ∧ 
  (sequence_a (n + 1) : ℚ) - (sequence_a n)^2 / (sequence_a (n - 1)) ≤ 1/2

#check sequence_a_odd

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_odd_l484_48444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_routes_A_to_B_l484_48454

-- Define the cities
inductive City : Type
| A | B | C | D | E | F
deriving DecidableEq

-- Define the roads
inductive Road : Type
| AB | AD | AE | BC | BD | CD | DE | EF
deriving DecidableEq

-- Define a route as a list of roads
def Route := List Road

-- Function to check if a route is valid (uses each road exactly once)
def isValidRoute (r : Route) : Prop :=
  r.length = 8 ∧ r.toFinset.card = 8

-- Function to check if a route starts at A and ends at B
def isAtoB (r : Route) : Prop :=
  match r with
  | [] => False
  | hd :: _ => hd = Road.AB ∨ hd = Road.AD ∨ hd = Road.AE

-- Theorem stating the number of valid routes from A to B
theorem number_of_routes_A_to_B :
  (∃ routes : Finset Route, routes.card = 16 ∧
    ∀ r ∈ routes, isValidRoute r ∧ isAtoB r) := by
  sorry

#check number_of_routes_A_to_B

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_routes_A_to_B_l484_48454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l484_48446

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x - 3*a else a^x - 2

-- State the theorem
theorem range_of_a_for_decreasing_f :
  ∀ a : ℝ, 
    (a > 0) → 
    (a ≠ 1) → 
    (∀ x y : ℝ, x < y → f a x > f a y) → 
    (a ∈ Set.Ioc 0 (1/3)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_decreasing_f_l484_48446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l484_48481

-- Define the function f
noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x < c then c * x + 1
  else if c ≤ x ∧ x < 1 then 2^(-x/c^2) + 1
  else 0  -- undefined for other x values

-- State the theorem
theorem function_properties :
  ∃ (c : ℝ), 
    (f c c = 5/4) ∧ 
    (c = 1/2) ∧ 
    (∀ x : ℝ, f c x > Real.sqrt 2 / 8 + 1 ↔ Real.sqrt 2 / 4 < x ∧ x < 5/8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l484_48481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_C_l484_48451

/-- Pentagon with specific properties -/
structure SymmetricPentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ
  verticalSymmetry : True  -- Placeholder for vertical symmetry condition
  squareArea : (B.2 - A.2) * (D.1 - A.1) = 10
  pointA : A = (0, 0)
  pointB : B = (0, 4)
  pointD : D = (4, 4)
  pointE : E = (4, 0)

/-- Theorem stating the y-coordinate of point C -/
theorem y_coordinate_of_C (p : SymmetricPentagon) 
    (triangleArea : (1/2 : ℝ) * (p.D.1 - p.B.1) * (p.C.2 - p.B.2) = 30) :
    p.C.2 = 6 * Real.sqrt 10 + 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_coordinate_of_C_l484_48451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wendy_received_amount_l484_48468

/-- Represents the types of chocolate bars --/
inductive ChocolateType
| Dark
| Milk
| White

/-- Represents the price of each chocolate bar type --/
def price (t : ChocolateType) : ℚ :=
  match t with
  | .Dark => 3
  | .Milk => 4
  | .White => 5

/-- Represents the number of bars of each type in a full box --/
def barsPerType : ℕ := 4

/-- Represents the discount rate for buying a full box --/
def discountRate : ℚ := 1 / 10

/-- Represents the number of bars returned for each type --/
def returned (t : ChocolateType) : ℕ :=
  match t with
  | .Dark => 0
  | .Milk => 2
  | .White => 1

/-- Calculates the amount Wendy received after the transaction --/
def amountReceived : ℚ :=
  let fullBoxPrice := (price .Dark + price .Milk + price .White) * barsPerType
  let discountedPrice := fullBoxPrice * (1 - discountRate)
  let returnedAmount := (price .Dark * returned .Dark) + (price .Milk * returned .Milk) + (price .White * returned .White)
  discountedPrice - returnedAmount

/-- Theorem stating that the amount Wendy received is $30.20 --/
theorem wendy_received_amount :
  amountReceived = 151 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wendy_received_amount_l484_48468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_david_emily_frank_distance_l484_48477

/-- The vertical distance between the midpoint of two points and a third point -/
noncomputable def vertical_distance_to_midpoint (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  y3 - (y1 + y2) / 2

theorem david_emily_frank_distance :
  let david_x := (10 : ℝ)
  let david_y := (-10 : ℝ)
  let emily_x := (-4 : ℝ)
  let emily_y := (22 : ℝ)
  let frank_x := (3 : ℝ)
  let frank_y := (10 : ℝ)
  vertical_distance_to_midpoint david_x david_y emily_x emily_y frank_x frank_y = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_david_emily_frank_distance_l484_48477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_root_three_l484_48448

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is √3 under specific conditions. -/
theorem triangle_area_is_root_three (a b c : ℝ) (A B C : ℝ) : 
  -- Conditions
  (Real.sin A = Real.sqrt 3 * Real.sin C) →
  (B = π / 6) →  -- 30° in radians
  (b = 2) →
  -- Conclusion
  (1/2 * a * c * Real.sin B = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_root_three_l484_48448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triples_count_l484_48422

def count_valid_triples : ℕ :=
  let b := 2000
  (Finset.filter (fun p : ℕ × ℕ × ℕ =>
    let (a, c, x) := p
    a ≤ c ∧ c ≤ b ∧
    0 < x ∧ x < a ∧
    x * b = a * a ∧ a * b = c * x ∧
    a * c = b * b
  ) (Finset.product (Finset.range (b + 1)) (Finset.product (Finset.range (b + 1)) (Finset.range (b + 1))))).card

theorem valid_triples_count : count_valid_triples = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_triples_count_l484_48422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_of_geometric_sequence_l484_48438

noncomputable def geometric_sequence (a : ℝ) : Fin 3 → ℝ
  | 0 => a + Real.log 3 / Real.log 2
  | 1 => a + Real.log 3 / Real.log 4
  | 2 => a + Real.log 3 / Real.log 8

theorem common_ratio_of_geometric_sequence (a : ℝ) :
  let seq := geometric_sequence a
  (seq 1 - seq 0) / (seq 0 - a) = 2/3 ∧
  (seq 2 - seq 1) / (seq 1 - a) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_ratio_of_geometric_sequence_l484_48438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_angle_l484_48496

/-- Given a triangle ABC with centroid G, prove that if 
    a * GA + (3/5) * b * GB + (3/7) * c * GC = 0, then angle C = 2π/3 -/
theorem triangle_centroid_angle (A B C G : EuclideanSpace ℝ (Fin 2)) (a b c : ℝ) :
  let GA := A - G
  let GB := B - G
  let GC := C - G
  (G = (1/3 : ℝ) • (A + B + C)) →  -- G is the centroid
  (a • GA + (3/5 : ℝ) • b • GB + (3/7 : ℝ) • c • GC = 0) →
  (EuclideanGeometry.angle A B C = 2 * Real.pi / 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_angle_l484_48496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l484_48406

/-- The initial amount of water in the bowl, in ounces -/
def initial_water : ℝ := 10

/-- The daily evaporation rate, in ounces per day -/
def daily_evaporation : ℝ := 0.007

/-- The number of days over which evaporation occurred -/
def days : ℕ := 50

/-- The percentage of water that evaporated, as a decimal -/
def evaporation_percentage : ℝ := 0.035000000000000004

theorem initial_water_amount :
  initial_water * evaporation_percentage = daily_evaporation * (days : ℝ) ∧
  initial_water = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_water_amount_l484_48406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_in_bisected_right_triangle_l484_48494

/-- Given a right-angled triangle PQR with ∠PQR = 90°, and a line PS that bisects ∠PQR,
    if ∠PQS = y° and ∠SQR = 2x°, then x + y = 45. -/
theorem angle_sum_in_bisected_right_triangle (x y : ℝ) : 
  (90 : ℝ) = 90 ∧ 
  y = y ∧ 
  2 * x = 2 * x ∧
  y = 2 * y → 
  x + y = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_in_bisected_right_triangle_l484_48494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_values_of_a_and_c_value_of_sin_A_minus_B_l484_48419

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Add conditions to ensure it's a valid triangle
  valid : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = Real.pi

-- Define the problem conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.c = 6 ∧ t.b = 2 ∧ Real.cos t.B = 7/9

-- Theorem 1: Values of a and c
theorem values_of_a_and_c (t : Triangle) (h : triangle_conditions t) : 
  t.a = 3 ∧ t.c = 3 := by sorry

-- Theorem 2: Value of sin(A-B)
theorem value_of_sin_A_minus_B (t : Triangle) (h : triangle_conditions t) : 
  Real.sin (t.A - t.B) = 10 * Real.sqrt 2 / 27 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_values_of_a_and_c_value_of_sin_A_minus_B_l484_48419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_evaluation_l484_48434

theorem nested_sqrt_evaluation : 
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt (45 * Real.sqrt 9))) = 15 * (Real.sqrt (Real.sqrt 15)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_sqrt_evaluation_l484_48434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l484_48499

/-- Given function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x + a) / x

/-- Function g -/
noncomputable def g (a k : ℝ) (x : ℝ) : ℝ := x * f a x + |x^2 - 1| + (k - a)*x - a

/-- Main theorem -/
theorem main_theorem (a : ℝ) (k : ℝ) (x₁ x₂ : ℝ) 
  (ha : a < 1)
  (hx₁ : 0 < x₁) (hx₂ : x₂ < 2) (hx₁₂ : x₁ < x₂)
  (hg₁ : g a k x₁ = 0) (hg₂ : g a k x₂ = 0) :
  (∀ x y, 1 ≤ x → x < y → f a x < f a y) ∧ 
  (-7/2 < k ∧ k < -1) ∧
  (1/x₁ + 1/x₂ < 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_main_theorem_l484_48499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_form_open_segment_l484_48409

/-- Represents a triangle in a 2D plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a parallelogram in a 2D plane -/
structure Parallelogram where
  M : ℝ × ℝ
  N : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ

/-- Represents a direction in a 2D plane -/
structure Direction where
  dx : ℝ
  dy : ℝ

/-- Defines a line segment between two points -/
def lineSegment (a b : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ p = (1 - t) • a + t • b}

/-- Checks if two line segments are parallel -/
def parallel (a b c d : ℝ × ℝ) : Prop :=
  (b.1 - a.1) * (d.2 - c.2) = (b.2 - a.2) * (d.1 - c.1)

/-- Defines a family of parallelograms inscribed in a triangle -/
def InscribedParallelogramFamily (T : Triangle) (d1 d2 : Direction) : Set Parallelogram :=
  {P : Parallelogram | 
    (P.M ∈ lineSegment T.A T.B ∧ P.N ∈ lineSegment T.A T.B) ∧
    P.K ∈ lineSegment T.B T.C ∧
    P.L ∈ lineSegment T.C T.A ∧
    parallel P.M P.N P.K P.L ∧
    parallel P.M P.L P.N P.K}

/-- The center of a parallelogram -/
noncomputable def center (P : Parallelogram) : ℝ × ℝ :=
  ((P.M.1 + P.K.1) / 2, (P.M.2 + P.K.2) / 2)

/-- The set of centers of a family of parallelograms -/
def CenterSet (T : Triangle) (d1 d2 : Direction) : Set (ℝ × ℝ) :=
  {c | ∃ P ∈ InscribedParallelogramFamily T d1 d2, c = center P}

/-- Predicate to check if a set is an open line segment -/
def IsOpenSegment (S : Set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ × ℝ, a ≠ b ∧ S = {p | ∃ t : ℝ, 0 < t ∧ t < 1 ∧ p = (1 - t) • a + t • b}

/-- Theorem: The set of centers forms an open line segment -/
theorem centers_form_open_segment (T : Triangle) (d1 d2 : Direction) :
  IsOpenSegment (CenterSet T d1 d2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_centers_form_open_segment_l484_48409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l484_48450

theorem product_remainder (a b c : ℕ) :
  a % 7 = 3 → b % 7 = 4 → c % 7 = 5 → (a * b * c) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_l484_48450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solutions_l484_48407

def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 + 2 * x + 1 = 0}

theorem quadratic_equation_solutions (a : ℝ) :
  (1 ∈ A a → A a = {-1/3, 1}) ∧
  ((∃! x, x ∈ A a) → a = 0 ∨ a = 1) ∧
  ((∀ x y, x ∈ A a → y ∈ A a → x = y) → a ≥ 1 ∨ a = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solutions_l484_48407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_test_average_score_l484_48491

/-- Calculates the average score of students given the score distribution -/
def averageScore (totalQuestions : ℕ) (scoreDistribution : List (ℕ × ℚ)) : ℚ :=
  scoreDistribution.foldr (fun (score, proportion) acc => acc + score * proportion) 0

/-- Proves that the average score is 1.9 given the specified conditions -/
theorem test_average_score :
  let totalQuestions : ℕ := 3
  let scoreDistribution : List (ℕ × ℚ) := [(3, 3/10), (2, 4/10), (1, 2/10), (0, 1/10)]
  let numStudents : ℕ := 30
  averageScore totalQuestions scoreDistribution = 19/10 := by
  -- Unfold the definition of averageScore
  unfold averageScore
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

#eval averageScore 3 [(3, 3/10), (2, 4/10), (1, 2/10), (0, 1/10)]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_test_average_score_l484_48491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_swaps_to_divisible_by_11_l484_48410

def num_digits : ℕ := 4026
def num_ones : ℕ := 2013
def num_twos : ℕ := 2013

def is_valid_number (n : ℕ) : Prop :=
  n.digits 10 = List.replicate num_ones 1 ++ List.replicate num_twos 2

def swap_digits (n : ℕ) (i j : ℕ) : ℕ := sorry

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

theorem max_swaps_to_divisible_by_11 :
  ∀ n : ℕ, is_valid_number n →
  ∃ k : ℕ, k ≤ 5 ∧
  ∃ (swaps : List (ℕ × ℕ)),
    swaps.length = k ∧
    is_divisible_by_11 (swaps.foldl (λ m (i, j) => swap_digits m i j) n) :=
by sorry

#check max_swaps_to_divisible_by_11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_swaps_to_divisible_by_11_l484_48410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_divisibility_impossible_l484_48429

/-- A heptagon is represented as a vector of 7 integers -/
def Heptagon := Fin 7 → Int

/-- Check if a divides b -/
def divides (a b : Int) : Prop := ∃ k : Int, b = a * k

/-- Check if either a divides b or b divides a -/
def sideDivisibility (a b : Int) : Prop := divides a b ∨ divides b a

/-- Check if neither a divides b nor b divides a -/
def diagonalNonDivisibility (a b : Int) : Prop := ¬(divides a b) ∧ ¬(divides b a)

/-- Check if the heptagon satisfies the side divisibility condition -/
def satisfiesSideCondition (h : Heptagon) : Prop :=
  ∀ i : Fin 7, sideDivisibility (h i) (h ((i + 1) : Fin 7))

/-- Check if the heptagon satisfies the diagonal non-divisibility condition -/
def satisfiesDiagonalCondition (h : Heptagon) : Prop :=
  ∀ i j : Fin 7, (j ≠ (i + 1 : Fin 7) ∧ j ≠ (i - 1 : Fin 7)) → diagonalNonDivisibility (h i) (h j)

/-- The main theorem: It is impossible to construct a heptagon satisfying both conditions -/
theorem heptagon_divisibility_impossible :
  ¬∃ h : Heptagon, satisfiesSideCondition h ∧ satisfiesDiagonalCondition h :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heptagon_divisibility_impossible_l484_48429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mirror_point_theorem_l484_48474

/-- Given a point in 3D space with spherical coordinates (ρ, θ, φ), 
    this function returns the spherical coordinates of the point (x, -y, -z) --/
noncomputable def mirror_point (ρ θ φ : Real) : Real × Real × Real :=
  (ρ, (2 * Real.pi - θ) % (2 * Real.pi), Real.pi - φ)

theorem mirror_point_theorem :
  let original : Real × Real × Real := (3, 9 * Real.pi / 7, Real.pi / 3)
  mirror_point original.1 original.2.1 original.2.2 = (3, 5 * Real.pi / 7, 2 * Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mirror_point_theorem_l484_48474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l484_48418

noncomputable def rational_function (p q : ℝ → ℝ) : ℝ → ℝ :=
  fun x => p x / q x

/-- The degree of a polynomial -/
def polynomial_degree (p : ℝ → ℝ) : ℕ := sorry

theorem rational_function_value :
  ∀ (p q : ℝ → ℝ),
    (polynomial_degree p = 2) →
    (polynomial_degree q = 2) →
    (∀ x, x ≠ -4 → rational_function p q x = -3 * x / (x - 3)) →
    (∃ k, ∀ x, x ≠ -4 → p x = k * (x + 4) * x) →
    (∀ x, x ≠ -4 → q x = (x + 4) * (x - 3)) →
    rational_function p q 0 = 0 →
    rational_function p q 2 = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_value_l484_48418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_subtraction_l484_48479

/-- Represents a number in base 7 --/
structure Base7 where
  value : Nat

/-- Converts a base 7 number to a natural number --/
def Base7.toNat (n : Base7) : Nat :=
  sorry

/-- Converts a natural number to a base 7 number --/
def Base7.fromNat (n : Nat) : Base7 :=
  sorry

/-- Subtracts two base 7 numbers --/
def Base7.sub (a b : Base7) : Base7 :=
  sorry

theorem base7_subtraction :
  (Base7.fromNat 5123).sub (Base7.fromNat 2345) = Base7.fromNat 2065 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_subtraction_l484_48479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_josiah_yards_per_game_l484_48424

/-- Proof that Josiah ran 22 yards in each game given the conditions of the problem -/
theorem josiah_yards_per_game 
  (malik_yards_per_game : ℕ) 
  (darnell_avg_yards : ℕ) 
  (total_games : ℕ) 
  (total_yards : ℕ) 
  (josiah_yards_per_game : ℕ)
  (h1 : malik_yards_per_game = 18)
  (h2 : darnell_avg_yards = 11)
  (h3 : total_games = 4)
  (h4 : total_yards = 204)
  (h5 : josiah_yards_per_game * total_games + 
        malik_yards_per_game * total_games + 
        darnell_avg_yards * total_games = total_yards) :
  josiah_yards_per_game = 22 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_josiah_yards_per_game_l484_48424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l484_48405

/-- The line x cos θ + y sin θ + 2 = 0 is tangent to the circle x² + y² = 4 -/
theorem line_tangent_to_circle :
  ∀ θ : ℝ,
  let l : ℝ × ℝ → Prop := λ p ↦ p.1 * Real.cos θ + p.2 * Real.sin θ + 2 = 0
  let c : ℝ × ℝ → Prop := λ p ↦ p.1^2 + p.2^2 = 4
  ∃! p : ℝ × ℝ, l p ∧ c p :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l484_48405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l484_48439

/-- Represents an ellipse with center at origin and focus on x-axis -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_bc : b > 0
  h_ac : a^2 = b^2 + c^2

/-- Represents a line -/
structure Line where
  k : ℝ
  m : ℝ

noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ := e.c / e.a

def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.k * x + l.m

def Circle.equation (r : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 = r^2

theorem ellipse_properties (e : Ellipse) :
  e.eccentricity = 1/2 →
  ∃ (l : Line), l.equation (-e.c) 0 ∧ 
    (∀ x y, l.equation x y → x > -e.c → y / (x + e.c) = Real.sqrt 3) ∧
    (∃ x y, l.equation x y ∧ Circle.equation (e.b/e.a) x y) →
  e.equation = λ x y ↦ x^2/4 + y^2/3 = 1 ∧
  ∃ (l : Line), 
    (∃ x₁ y₁ x₂ y₂, x₁ ≠ x₂ ∧ e.equation x₁ y₁ ∧ e.equation x₂ y₂ ∧ 
      l.equation x₁ y₁ ∧ l.equation x₂ y₂ ∧
      (x₁ - 2) * (x₂ - 2) + y₁ * y₂ = 0) →
    l.equation (2/7) 0 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l484_48439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_parabola_vertex_count_l484_48452

/-- The number of values of a for which the line y = 2x + a intersects 
    the vertex of the parabola y = x^2 + 3a^2 -/
theorem line_intersects_parabola_vertex_count : 
  let line := λ (x a : ℝ) => 2 * x + a
  let parabola := λ (x a : ℝ) => x^2 + 3 * a^2
  let vertex_x := 0
  let vertex_y := λ (a : ℝ) => parabola vertex_x a
  let intersection_condition := λ (a : ℝ) => line vertex_x a = vertex_y a
  ∃ s : Finset ℝ, s.card = 2 ∧ ∀ a : ℝ, a ∈ s ↔ intersection_condition a := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_parabola_vertex_count_l484_48452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l484_48455

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.cos (ω * x) * Real.sin (ω * x - Real.pi / 3) + Real.sqrt 3 * (Real.cos (ω * x))^2 - Real.sqrt 3 / 4

theorem function_properties
  (ω : ℝ)
  (h_ω : ω > 0)
  (h_symmetry : ∃ (c : ℝ), ∀ (x : ℝ), |x - c| = Real.pi / 4 → f ω x = f ω (2 * c - x))
  (A B C : ℝ)
  (h_fA : f ω A = Real.sqrt 3 / 4)
  (h_sinC : Real.sin C = 1 / 3)
  (h_a : Real.sin C / Real.sin A = Real.sqrt 3) :
  ω = 1 ∧
  (∃ (k : ℤ), A = Real.pi / 6) ∧
  (∃ (k : ℤ), ∀ (x : ℝ), f ω x = f ω (Real.pi / 6 + k * Real.pi / 2 - x)) ∧
  Real.sin B / Real.sin C = (3 + 2 * Real.sqrt 6) / 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l484_48455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_five_l484_48488

-- Define the function f as noncomputable
noncomputable def f (a b x : ℝ) : ℝ := a * x - 5 * b / x + 2

-- State the theorem
theorem f_negative_five (a b : ℝ) (h : f a b 5 = 5) : f a b (-5) = -1 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_negative_five_l484_48488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_orientation_after_711_transformations_l484_48430

/-- Represents the possible orientations of the square --/
inductive SquareOrientation
  | ABCD
  | ADCB
  | BADC
  | CDAB
  | DBCA

/-- Represents a transformation of the square --/
def transform (o : SquareOrientation) : SquareOrientation :=
  match o with
  | SquareOrientation.ABCD => SquareOrientation.ADCB
  | SquareOrientation.ADCB => SquareOrientation.BADC
  | SquareOrientation.BADC => SquareOrientation.CDAB
  | SquareOrientation.CDAB => SquareOrientation.DBCA
  | SquareOrientation.DBCA => SquareOrientation.ABCD

/-- Applies n transformations to the initial orientation --/
def applyTransformations (n : Nat) : SquareOrientation :=
  match n with
  | 0 => SquareOrientation.ABCD
  | n + 1 => transform (applyTransformations n)

theorem square_orientation_after_711_transformations :
  applyTransformations 711 = SquareOrientation.ADCB := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_orientation_after_711_transformations_l484_48430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_inverse_correct_l484_48483

-- Define the functions f, g, and k
noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := 3 * x - 4
noncomputable def k (x : ℝ) : ℝ := f (g x)

-- Define the inverse function
noncomputable def k_inv (x : ℝ) : ℝ := (x + 11) / 12

-- Theorem statement
theorem k_inverse_correct : 
  ∀ x : ℝ, k (k_inv x) = x ∧ k_inv (k x) = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_inverse_correct_l484_48483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_25_even_is_650_l484_48471

/-- The sum of the first n even numbers -/
def sum_first_n_even (n : ℕ) : ℕ := 
  n * (n + 1)

/-- Theorem: The sum of the first 25 even numbers is 650 -/
theorem sum_first_25_even_is_650 : sum_first_n_even 25 = 650 := by
  -- Unfold the definition of sum_first_n_even
  unfold sum_first_n_even
  -- Evaluate the expression
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_25_even_is_650_l484_48471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_infinite_l484_48420

noncomputable def sequenceT (t : ℝ) : ℕ → ℝ
  | 0 => t
  | n + 1 => 2 * (sequenceT t n)^2 - 1

def S : Set ℝ :=
  {t ∈ Set.Icc (-1 : ℝ) 1 | ∃ N : ℕ, ∀ n ≥ N, sequenceT t n = 1}

theorem S_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_infinite_l484_48420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_developers_total_time_l484_48428

/-- The time Katherine takes to complete one website -/
noncomputable def katherine_time : ℝ := 20

/-- The number of websites each junior developer needs to complete -/
def websites_per_developer : ℕ := 10

/-- The time Naomi takes to complete one website -/
noncomputable def naomi_time : ℝ := katherine_time * (1 + 1/4)

/-- The time Lucas takes to complete one website -/
noncomputable def lucas_time : ℝ := katherine_time * (1 + 1/3)

/-- The time Isabella takes to complete one website -/
noncomputable def isabella_time : ℝ := katherine_time * (1 + 1/2)

/-- The total time for all three junior developers to complete their assigned websites -/
noncomputable def total_time : ℝ := 
  naomi_time * (websites_per_developer : ℝ) + 
  lucas_time * (websites_per_developer : ℝ) + 
  isabella_time * (websites_per_developer : ℝ)

theorem junior_developers_total_time : 
  total_time = 816.7 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_junior_developers_total_time_l484_48428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_segment_length_l484_48498

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A point on the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point is on an ellipse -/
def Ellipse.contains (e : Ellipse) (p : Point) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- The length of a line segment intercepted by an ellipse and a line -/
noncomputable def intercepted_segment_length (e : Ellipse) (p : Point) (m : ℝ) : ℝ :=
  sorry -- Definition of the length calculation

/-- Main theorem -/
theorem ellipse_and_segment_length 
  (e : Ellipse) 
  (h_contains : e.contains ⟨0, 4⟩) 
  (h_eccentricity : e.eccentricity = 3/5) :
  (e.a = 5 ∧ e.b = 4) ∧ 
  intercepted_segment_length e ⟨3, 0⟩ (4/5) = 41/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_segment_length_l484_48498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EF_length_l484_48487

/-- Represents a trapezoid ABCD with bases AB and CD, and perpendiculars from C and D to AB meeting at E and F respectively. -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  AD_eq_BC : Bool
  EF : ℝ

/-- The length of EF in a trapezoid with given properties -/
noncomputable def EF_length (t : Trapezoid) : ℝ := 24 / 13

/-- Theorem stating that for a trapezoid with AB = 8, CD = 5, and AD = BC, the length of EF is 24/13 -/
theorem trapezoid_EF_length (t : Trapezoid) 
  (h1 : t.AB = 8) 
  (h2 : t.CD = 5) 
  (h3 : t.AD_eq_BC = true) : 
  t.EF = EF_length t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_EF_length_l484_48487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_one_inequality_holds_iff_m_in_range_l484_48465

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * x^2 - 3 * x

-- Part 1: Prove that a = 1 when the tangent line at (1, f(1)) is y = -2
theorem tangent_line_implies_a_equals_one (a : ℝ) :
  (deriv (f a) 1 = -2) → a = 1 := by sorry

-- Part 2: Prove the inequality holds iff m ≤ -1710
theorem inequality_holds_iff_m_in_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 10 →
    f 1 x₁ - f 1 x₂ > m * (x₂ - x₁) / (x₁ * x₂)) ↔
  m ≤ -1710 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_equals_one_inequality_holds_iff_m_in_range_l484_48465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_ten_km_cheaper_than_two_fifteen_km_l484_48460

/-- Calculates the taxi fare for a given distance -/
noncomputable def taxiFare (distance : ℝ) : ℝ :=
  if distance < 3 then 5
  else if distance ≤ 10 then 1.2 * distance + 1.4
  else 1.8 * distance - 4.6

/-- Calculates the total fare for a journey split into segments -/
noncomputable def segmentedJourneyFare (segmentDistance : ℝ) (numberOfSegments : ℕ) : ℝ :=
  (taxiFare segmentDistance) * (numberOfSegments : ℝ)

/-- Theorem stating that three 10 km segments are cheaper than two 15 km segments -/
theorem three_ten_km_cheaper_than_two_fifteen_km :
  segmentedJourneyFare 10 3 < segmentedJourneyFare 15 2 := by
  -- Unfold the definitions
  unfold segmentedJourneyFare taxiFare
  -- Simplify the expressions
  simp
  -- Prove the inequality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_ten_km_cheaper_than_two_fifteen_km_l484_48460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_sight_not_blocked_l484_48480

-- Define the circle C
noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 3

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B (a : ℝ) : ℝ × ℝ := (2, a)

-- Define the line AB
noncomputable def line_AB (a x : ℝ) : ℝ := (a / 4) * x

-- Define the non-intersection condition
def no_intersection (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬(circle_C x (line_AB a x))

-- Theorem statement
theorem line_of_sight_not_blocked (a : ℝ) :
  no_intersection a ↔ a < -4 * Real.sqrt 3 ∨ a > 4 * Real.sqrt 3 := by
  sorry

#check line_of_sight_not_blocked

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_of_sight_not_blocked_l484_48480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_18_7_l484_48453

-- Define the quadrilateral PQRS
structure Quadrilateral :=
  (PQ QR RS SP PR : ℝ)

-- Define the area calculation function
noncomputable def area (q : Quadrilateral) : ℝ :=
  let s1 := (q.PQ + q.QR + q.PR) / 2
  let s2 := (q.PR + q.RS + q.SP) / 2
  let area1 := Real.sqrt (s1 * (s1 - q.PQ) * (s1 - q.QR) * (s1 - q.PR))
  let area2 := Real.sqrt (s2 * (s2 - q.PR) * (s2 - q.RS) * (s2 - q.SP))
  area1 + area2

-- State the theorem
theorem quadrilateral_area_is_18_7 (q : Quadrilateral) 
  (h1 : q.PQ = 4) (h2 : q.QR = 5) (h3 : q.RS = 3) (h4 : q.SP = 6) (h5 : q.PR = 7) : 
  ∃ ε > 0, |area q - 18.7| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_area_is_18_7_l484_48453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l484_48489

theorem sin_2alpha_value (α : ℝ) 
  (h1 : Real.cos (α + π/4) = 3/5)
  (h2 : π/2 ≤ α)
  (h3 : α ≤ 3*π/2) : 
  Real.sin (2*α) = 7/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l484_48489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_ball_probability_l484_48421

/-- Represents the probability of drawing a ball of a specific color from a bag -/
def probability (color : ℚ) (total : ℚ) : ℚ := color / total

theorem black_ball_probability
  (total_balls : ℕ)
  (red_balls : ℕ)
  (white_prob : ℚ)
  (h_total : total_balls = 100)
  (h_red : red_balls = 45)
  (h_white_prob : white_prob = 23 / 100) :
  probability ((total_balls : ℚ) - (red_balls : ℚ) - (white_prob * total_balls)) (total_balls : ℚ) = 32 / 100 := by
  sorry

#check black_ball_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_ball_probability_l484_48421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_scores_l484_48478

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Minimum distance from a point to either of two other points -/
noncomputable def minDistance (p x y : Point) : ℝ :=
  min (distance p x) (distance p y)

/-- The four vertices of a unit square -/
def unitSquareVertices : List Point :=
  [{ x := 0, y := 0 }, { x := 1, y := 0 }, { x := 1, y := 1 }, { x := 0, y := 1 }]

/-- Sum of scores for all vertices -/
noncomputable def sumOfScores (x y : Point) : ℝ :=
  (unitSquareVertices.map (fun v => minDistance v x y)).sum

/-- A point is inside the unit square if its coordinates are between 0 and 1 -/
def isInsideUnitSquare (p : Point) : Prop :=
  0 ≤ p.x ∧ p.x ≤ 1 ∧ 0 ≤ p.y ∧ p.y ≤ 1

theorem min_sum_of_scores :
  ∀ x y : Point, isInsideUnitSquare x → isInsideUnitSquare y →
  sumOfScores x y ≥ Real.sqrt ((Real.sqrt 6 + Real.sqrt 2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_scores_l484_48478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_approx_seven_point_three_three_l484_48435

/-- Represents Elaine's financial situation over three years --/
structure ElaineFinances where
  last_year_earnings : ℝ
  last_year_rent_percent : ℝ
  last_year_utilities_percent : ℝ
  this_year_earnings_increase : ℝ
  this_year_rent_percent : ℝ
  this_year_utilities_percent : ℝ
  next_year_earnings_increase : ℝ
  next_year_rent_percent : ℝ
  next_year_utilities_percent : ℝ

/-- Calculates the percentage increase in rent spending from this year to next year --/
noncomputable def rent_increase_percent (e : ElaineFinances) : ℝ :=
  let this_year_earnings := e.last_year_earnings * (1 + e.this_year_earnings_increase)
  let next_year_earnings := this_year_earnings * (1 + e.next_year_earnings_increase)
  let this_year_rent := this_year_earnings * e.this_year_rent_percent
  let next_year_rent := next_year_earnings * e.next_year_rent_percent
  (next_year_rent - this_year_rent) / this_year_rent * 100

/-- Theorem stating that the rent increase percentage is approximately 7.33% --/
theorem rent_increase_approx_seven_point_three_three
  (e : ElaineFinances)
  (h1 : e.last_year_rent_percent = 0.20)
  (h2 : e.last_year_utilities_percent = 0.15)
  (h3 : e.this_year_earnings_increase = 0.25)
  (h4 : e.this_year_rent_percent = 0.30)
  (h5 : e.this_year_utilities_percent = 0.20)
  (h6 : e.next_year_earnings_increase = 0.15)
  (h7 : e.next_year_rent_percent = 0.28)
  (h8 : e.next_year_utilities_percent = 0.22) :
  ∃ (ε : ℝ), ε > 0 ∧ abs (rent_increase_percent e - 7.33) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rent_increase_approx_seven_point_three_three_l484_48435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l484_48441

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on the ellipse -/
structure PointOnEllipse (e : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / e.a^2 + y^2 / e.b^2 = 1

theorem eccentricity_range (e : Ellipse) (p : PointOnEllipse e) 
  (h_perp : p.x = Real.sqrt (e.a^2 - e.b^2)) -- PF perpendicular to x-axis
  (h_angle : Real.sin (Real.arctan (p.y / (p.x + e.a))) < Real.sqrt 10 / 10) :
  2/3 < eccentricity e ∧ eccentricity e < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eccentricity_range_l484_48441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l484_48484

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them -/
noncomputable def trapezium_area (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: The area of a trapezium with parallel sides of lengths 26 cm and 18 cm, 
    and a distance of 15 cm between them, is equal to 330 cm² -/
theorem trapezium_area_example : trapezium_area 26 18 15 = 330 := by
  -- Unfold the definition of trapezium_area
  unfold trapezium_area
  -- Perform the calculation
  simp [mul_add, mul_div_assoc]
  -- The rest of the proof (which would involve numeric calculations)
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_area_example_l484_48484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_inspection_l484_48433

/-- Represents a box of beverage cans -/
structure BeverageBox where
  total_cans : ℕ
  qualified_cans : ℕ
  unqualified_cans : ℕ
  total_eq_sum : total_cans = qualified_cans + unqualified_cans

/-- Represents the selection process -/
structure Selection where
  box : BeverageBox
  selected_cans : ℕ

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem beverage_inspection (b : BeverageBox) (s : Selection)
  (h1 : b.total_cans = 6)
  (h2 : b.qualified_cans = 4)
  (h3 : b.unqualified_cans = 2)
  (h4 : s.box = b)
  (h5 : s.selected_cans = 2) :
  (choose b.total_cans s.selected_cans = 15) ∧
  (Nat.cast (choose b.total_cans s.selected_cans - choose b.qualified_cans s.selected_cans) / Nat.cast (choose b.total_cans s.selected_cans) : ℚ) = 3/5 := by
  sorry

#check beverage_inspection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_beverage_inspection_l484_48433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_Na_and_unbounded_l484_48467

-- Define the type for base a representation
def BaseRepresentation (a : ℕ) := List ℕ

-- Function to calculate the sum of squares of digits
def sumOfSquaresOfDigits (a : ℕ) (k : ℕ) : ℕ :=
  let digits : BaseRepresentation a := sorry
  (digits.map (λ d => d * d)).sum

-- Define Na as the number of k satisfying the property
def Na (a : ℕ) : ℕ :=
  (Finset.filter (λ k => sumOfSquaresOfDigits a k = k) (Finset.range (a^a))).card

-- Main theorem
theorem odd_Na_and_unbounded :
  ∀ a : ℕ, a ≥ 2 → Odd (Na a) ∧ ∀ M : ℕ, ∃ a' : ℕ, a' ≥ 2 ∧ Na a' ≥ M :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_Na_and_unbounded_l484_48467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_not_determined_by_A_P_l484_48469

-- Define a polynomial type
def MyPolynomial (α : Type) [Semiring α] := ℕ → α

-- Define a characteristic function for polynomials
noncomputable def A_P {α : Type} [Semiring α] (P : MyPolynomial α) : α := sorry

-- Define a degree function for polynomials
noncomputable def degree {α : Type} [Semiring α] (P : MyPolynomial α) : ℕ := sorry

-- Theorem statement
theorem degree_not_determined_by_A_P :
  ∃ (P₁ P₂ : MyPolynomial ℝ),
    (degree P₁ ≠ degree P₂) ∧ (A_P P₁ = A_P P₂) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_not_determined_by_A_P_l484_48469
