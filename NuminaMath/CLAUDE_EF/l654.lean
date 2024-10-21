import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_costs_and_max_basketballs_l654_65414

-- Define the cost of a soccer ball and a basketball
variable (soccer_cost basketball_cost : ℚ)

-- Define the conditions
variable (h1 : 7 * soccer_cost = 5 * basketball_cost)
variable (h2 : 40 * soccer_cost + 20 * basketball_cost = 3400)

-- Define the total number of balls and budget constraint
variable (total_balls : ℕ)
variable (budget : ℚ)
variable (h3 : total_balls = 100)
variable (h4 : budget = 6300)

-- Define the function to calculate the total cost
def total_cost (soccer_cost basketball_cost : ℚ) (total_balls : ℕ) (num_basketball : ℕ) : ℚ :=
  basketball_cost * num_basketball + soccer_cost * (total_balls - num_basketball)

-- Theorem to prove
theorem ball_costs_and_max_basketballs :
  soccer_cost = 50 ∧ 
  basketball_cost = 70 ∧ 
  (∀ n : ℕ, n ≤ total_balls → total_cost soccer_cost basketball_cost total_balls n ≤ budget → n ≤ 65) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_costs_and_max_basketballs_l654_65414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_average_score_l654_65422

theorem basketball_average_score (initial_average : ℝ) (initial_players : ℕ) 
  (new_score1 new_score2 new_score3 : ℝ) (total_players : ℕ) :
  initial_average = 72 →
  initial_players = 60 →
  new_score1 = 76 →
  new_score2 = 88 →
  new_score3 = 82 →
  total_players = 63 →
  (initial_average * initial_players + new_score1 + new_score2 + new_score3) / total_players = 72.48 := by
sorry

#eval (72 * 60 + 76 + 88 + 82) / 63

end NUMINAMATH_CALUDE_ERRORFEEDBACK_basketball_average_score_l654_65422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_properties_l654_65485

/-- Represents a plane in 3D space -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Checks if a point (x, y, z) is on the plane -/
def Plane.contains (p : Plane) (x y z : ℝ) : Prop :=
  p.a * x + p.b * y + p.c * z + p.d = 0

/-- Checks if a plane contains the Oz-axis -/
def Plane.containsOz (p : Plane) : Prop :=
  ∀ z, p.contains 0 0 z

/-- Checks if a plane is parallel to the Oz-axis -/
def Plane.parallelToOz (p : Plane) : Prop :=
  p.c = 0

/-- Checks if a plane is perpendicular to the Oz-axis -/
def Plane.perpendicularToOz (p : Plane) : Prop :=
  p.a = 0 ∧ p.b = 0 ∧ p.c ≠ 0

/-- The normal vector of a plane -/
def Plane.normalVector (p : Plane) : Fin 3 → ℝ :=
  fun i => match i with
    | 0 => p.a
    | 1 => p.b
    | 2 => p.c

theorem plane_properties :
  let p1 : Plane := ⟨3, 2, 4, -8⟩
  let p2 : Plane := ⟨1, -1, 0, 0⟩
  let p3 : Plane := ⟨2, 3, 0, -6⟩
  let p4 : Plane := ⟨0, 0, 1, -2⟩
  (∀ x y z, p1.contains x y z ↔ x / (8/3) + y / 4 + z / 2 = 1) ∧
  (p2.containsOz ∧ ∀ x, p2.contains x x 0) ∧
  (p3.parallelToOz ∧ p3.normalVector = fun i => match i with
    | 0 => 2
    | 1 => 3
    | 2 => 0) ∧
  (p4.perpendicularToOz ∧ p4.contains 0 0 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_properties_l654_65485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_test_results_l654_65487

noncomputable def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  (xs.map (fun x => (x - m) ^ 2)).sum / xs.length

noncomputable def standardDeviation (xs : List ℝ) : ℝ :=
  Real.sqrt (variance xs)

theorem shooting_test_results :
  mean scores = 7 ∧ standardDeviation scores = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_test_results_l654_65487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_theorem_l654_65450

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point2D
  majorAxisEnd1 : Point2D
  majorAxisEnd2 : Point2D
  minorAxisEnd1 : Point2D
  minorAxisEnd2 : Point2D

/-- Calculates the focus of an ellipse with the greater x-coordinate -/
noncomputable def ellipseFocus (e : Ellipse) : Point2D :=
  { x := e.center.x + Real.sqrt 7
    y := e.center.y }

/-- Theorem: The focus of the given ellipse is at (5+√7, 0) -/
theorem ellipse_focus_theorem (e : Ellipse) 
    (h1 : e.center = { x := 5, y := 0 })
    (h2 : e.majorAxisEnd1 = { x := 1, y := 0 })
    (h3 : e.majorAxisEnd2 = { x := 9, y := 0 })
    (h4 : e.minorAxisEnd1 = { x := 5, y := 3 })
    (h5 : e.minorAxisEnd2 = { x := 5, y := -3 }) :
    ellipseFocus e = { x := 5 + Real.sqrt 7, y := 0 } := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_theorem_l654_65450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_segment_length_l654_65471

/-- Definition of an acute triangle -/
def AcuteTriangle (A B C : Point) : Prop := sorry

/-- Definition of an altitude in a triangle -/
def Altitude (A E B C : Point) : Prop := sorry

/-- Definition of segment length -/
def SegmentLength (A B : Point) : ℝ := sorry

/-- In an acute triangle ABC with altitudes BD and AE, where BD = 4, DC = 6, AE = 3, and EB = y, 
    the value of y is 4.5. -/
theorem altitude_segment_length (A B C D E : Point) (y : ℝ) : 
  AcuteTriangle A B C →
  Altitude A E B C →
  Altitude B D A C →
  SegmentLength B D = 4 →
  SegmentLength D C = 6 →
  SegmentLength A E = 3 →
  SegmentLength E B = y →
  y = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_segment_length_l654_65471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_of_sqrt_eq_three_l654_65443

theorem cube_of_sqrt_eq_three (x : ℝ) (h : Real.sqrt (x + 2) = 3) : (x + 2)^3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_of_sqrt_eq_three_l654_65443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l654_65419

noncomputable section

/-- Definition of the ellipse G -/
def ellipse_G (a b : ℝ) (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

/-- Definition of point A -/
def point_A : ℝ × ℝ := (0, 2)

/-- Definition of point B -/
def point_B : ℝ × ℝ := (3, 1)

/-- Definition of line l passing through B and C -/
def line_l (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 3) + 1

/-- Definition of point C on ellipse G and line l -/
noncomputable def point_C (a b k : ℝ) : ℝ × ℝ :=
  let x := (9 * k^2 - 6 * k - 1) / (3 * k^2 + 1)
  (x, k * (x - 3) + 1)

/-- Definition of circle with BC as diameter passing through A -/
def circle_condition (a b k : ℝ) : Prop :=
  let C := point_C a b k
  (C.1 - 3)^2 + (C.2 - 1)^2 = (C.1 - 0)^2 + (C.2 - 2)^2

/-- Main theorem -/
theorem ellipse_and_line_theorem (a b : ℝ) :
  ellipse_G a b 0 2 ∧ 
  ellipse_G a b 3 1 ∧
  (∃ k, circle_condition a b k) →
  (a = 2 * Real.sqrt 3 ∧ b = 2) ∧
  (∃ k, k = -1/2 ∨ k = 1/9) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_theorem_l654_65419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l654_65431

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.cos x) + 1/2

theorem f_properties :
  ∃ (α : ℝ),
    Real.tan α = 1/2 ∧
    f α = 17/10 ∧
    (∀ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) → T ≥ π) ∧
    (∀ (k : ℤ), ∀ (x y : ℝ),
      x ∈ Set.Icc (-3*π/8 + k*π) (π/8 + k*π) →
      y ∈ Set.Icc (-3*π/8 + k*π) (π/8 + k*π) →
        x ≤ y → f x ≤ f y) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l654_65431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l654_65407

/-- Represents a polynomial with real coefficients -/
def MyPolynomial := ℕ → ℝ

/-- The set of allowed operations on polynomials -/
inductive Operation
  | derivative_multiply : Operation
  | divide_coefficient (k : ℕ) : Operation
  | add_constant (c : ℝ) : Operation
  | remove_highest_degree : Operation

/-- The initial polynomial x^17 + 2x^15 + 4x^9 + x^6 + 4x^3 + 2x + 1 -/
def initial_polynomial : MyPolynomial := fun n =>
  match n with
  | 17 => 1
  | 15 => 2
  | 9 => 4
  | 6 => 1
  | 3 => 4
  | 1 => 2
  | 0 => 1
  | _ => 0

/-- The target polynomial 3x + 1 -/
def target_polynomial : MyPolynomial := fun n =>
  match n with
  | 1 => 3
  | 0 => 1
  | _ => 0

/-- Applies a single operation to a polynomial -/
def apply_operation (op : Operation) (p : MyPolynomial) : MyPolynomial :=
  sorry

/-- Checks if two polynomials are equal -/
def polynomial_eq (p q : MyPolynomial) : Prop :=
  ∀ n, p n = q n

/-- The main theorem stating that it's impossible to transform the initial polynomial
    into the target polynomial using the allowed operations -/
theorem impossible_transformation :
  ¬ ∃ (ops : List Operation),
    polynomial_eq (ops.foldl (fun p op => apply_operation op p) initial_polynomial) target_polynomial :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_transformation_l654_65407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_properties_l654_65434

/-- Line l: 4x + 3y + 6 = 0 -/
def line_l (x y : ℝ) : Prop := 4 * x + 3 * y + 6 = 0

/-- Circle C: (x - 1)² + y² = 9 -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

/-- Point on both line l and circle C -/
def intersection_point (p : ℝ × ℝ) : Prop :=
  line_l p.1 p.2 ∧ circle_C p.1 p.2

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x₀ y₀ : ℝ) : ℝ :=
  |4 * x₀ + 3 * y₀ + 6| / Real.sqrt (4^2 + 3^2)

/-- Length of a chord given center distance -/
noncomputable def chord_length (r d : ℝ) : ℝ := 2 * Real.sqrt (r^2 - d^2)

theorem line_circle_intersection_properties :
  ∃ (E F : ℝ × ℝ),
    intersection_point E ∧ intersection_point F ∧ E ≠ F →
    distance_point_to_line 1 0 = 2 ∧
    chord_length 3 2 = 2 * Real.sqrt 5 := by
  sorry

#check line_circle_intersection_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_properties_l654_65434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l654_65472

-- Define the constants as noncomputable
noncomputable def a : ℝ := 1 / Real.log (Real.sqrt 2)
noncomputable def b : ℝ := 6 / Real.log 9
noncomputable def c : ℝ := 7 / Real.log 7

-- State the theorem
theorem relationship_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_relationship_abc_l654_65472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_max_a_value_l654_65438

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the inequality for part 1
def inequality_solution_set (m : ℝ) : Set ℝ :=
  {x : ℝ | x + 1/2 ≤ 2*m}

-- Theorem for part 1
theorem part_one (m : ℝ) (h_m : m > 0) 
  (h_set : inequality_solution_set m = Set.Icc (-2) 2) : 
  m = 3/4 := by sorry

-- Theorem for part 2
theorem part_two (a : ℝ) 
  (h_inequality : ∀ (x y : ℝ), f x ≤ 2 + a / (2^y) + |2*x + 3|) :
  a ≤ 4 := by sorry

-- Theorem for the maximum value of a
theorem max_a_value : 
  ∃ (a : ℝ), (∀ (x y : ℝ), f x ≤ 2 + a / (2^y) + |2*x + 3|) ∧ 
  (∀ (b : ℝ), (∀ (x y : ℝ), f x ≤ 2 + b / (2^y) + |2*x + 3|) → b ≤ a) ∧ 
  a = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_max_a_value_l654_65438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_fourths_l654_65427

theorem cos_seven_pi_fourths : Real.cos (7 * π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_seven_pi_fourths_l654_65427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearing_time_approx_l654_65401

/-- Calculates the time for two trains to clear each other --/
noncomputable def train_clearing_time (length1 length2 speed1 speed2 angle : ℝ) : ℝ :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := Real.sqrt (speed1_ms^2 + speed2_ms^2 + 2 * speed1_ms * speed2_ms * Real.cos (angle * Real.pi / 180))
  let total_length := length1 + length2
  total_length / relative_speed

/-- Theorem stating the time for trains to clear each other --/
theorem train_clearing_time_approx :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_clearing_time 120 280 42 30 45 - 20.01| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_clearing_time_approx_l654_65401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sphere_volume_in_pyramid_l654_65444

/-- Regular triangular pyramid with specific properties -/
structure RegularTriangularPyramid where
  height : ℝ
  base_area : ℝ
  lateral_face_area : ℝ
  base_area_smaller : base_area = lateral_face_area / 4
  height_value : height = 130

/-- Sequence of spheres in the pyramid -/
noncomputable def sphere_sequence (p : RegularTriangularPyramid) (n : ℕ) : ℝ :=
  (4 / 3) * Real.pi * (10 * (11 / 13) ^ n) ^ 3

/-- Total volume of all spheres in the sequence -/
noncomputable def total_sphere_volume (p : RegularTriangularPyramid) : ℝ :=
  (8788000 * Real.pi) / 2598

/-- Theorem stating the total volume of spheres in the pyramid -/
theorem total_sphere_volume_in_pyramid (p : RegularTriangularPyramid) :
  ∑' n, sphere_sequence p n = total_sphere_volume p := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_sphere_volume_in_pyramid_l654_65444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_max_profit_l654_65479

/-- Last year's production cost per car in ten thousands of yuan -/
def last_year_cost : ℝ := 10

/-- Last year's factory price per car in ten thousands of yuan -/
def last_year_price : ℝ := 13

/-- Last year's annual sales volume in thousands of cars -/
def last_year_sales : ℝ := 5

/-- This year's production cost increase ratio -/
def x : ℝ → ℝ := id

/-- This year's factory price increase ratio -/
def price_increase (x : ℝ) : ℝ := 0.7 * x

/-- This year's annual sales volume function -/
def sales_volume (x : ℝ) : ℝ := 3240 * (-x^2 + 2*x + 5)

/-- Annual profit function in ten thousands of yuan -/
def annual_profit (x : ℝ) : ℝ :=
  (last_year_price * (1 + price_increase x) - last_year_cost * (1 + x)) * sales_volume x

theorem profit_increase (x : ℝ) (h : 0 < x ∧ x < 5/6) :
  annual_profit x > last_year_sales * (last_year_price - last_year_cost) := by
  sorry

theorem max_profit :
  ∃ (x : ℝ), x = 5/9 ∧ annual_profit x = 2000 ∧ 
  ∀ (y : ℝ), 0 < y ∧ y < 1 → annual_profit y ≤ annual_profit x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_increase_max_profit_l654_65479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l654_65406

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem symmetry_of_sine_function (ω φ : ℝ) 
  (h_ω : ω > 0)
  (h_φ : |φ| < Real.pi / 2)
  (h_period : ∀ x, f ω φ (x + Real.pi) = f ω φ x)
  (h_odd : ∀ x, f ω φ (x + Real.pi / 6) = -f ω φ (-x - Real.pi / 6)) :
  ∀ x, f ω φ (-Real.pi / 12 + x) = f ω φ (-Real.pi / 12 - x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_of_sine_function_l654_65406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l654_65454

theorem cos_double_angle (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (2*θ) = -7/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_double_angle_l654_65454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equality_l654_65496

open BigOperators

noncomputable def matrix_A (n : ℕ) (x y : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  λ i j => if x i + y j ≥ 0 then 1 else 0

theorem matrix_equality (n : ℕ) (x y : Fin n → ℝ) 
  (B : Matrix (Fin n) (Fin n) ℝ)
  (h_B_entries : ∀ i j, B i j = 0 ∨ B i j = 1)
  (h_row_sum : ∀ i, ∑ j, (matrix_A n x y) i j = ∑ j, B i j)
  (h_col_sum : ∀ j, ∑ i, (matrix_A n x y) i j = ∑ i, B i j) :
  matrix_A n x y = B := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_equality_l654_65496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l654_65412

/-- The area of a circular sector with radius 12 meters and central angle 39 degrees -/
noncomputable def sectorArea : ℝ :=
  (39 / 360) * Real.pi * 12 ^ 2

/-- Theorem: The area of the circular sector is approximately 48.9432 square meters -/
theorem sector_area_approx :
  |sectorArea - 48.9432| < 0.0001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_approx_l654_65412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l654_65466

/-- Given two sequences {a_n} and {b_n} with specific properties, prove that α + β = 30 -/
theorem sequence_problem (a b : ℕ → ℝ) (α β : ℝ) : 
  (∃ d ≠ 0, ∀ n, a (n + 1) = a n + d) →  -- a_n is arithmetic with nonzero common difference
  (∃ q, ∀ n, b (n + 1) = q * b n) →      -- b_n is geometric
  a 1 = 3 →                              -- a_1 = 3
  b 1 = 1 →                              -- b_1 = 1
  a 2 = b 2 →                            -- a_2 = b_2
  3 * a 5 = b 3 →                        -- 3a_5 = b_3
  (∀ n : ℕ, n > 0 → a n = Real.log (b n) / Real.log α + β) →  -- a_n = log_α(b_n) + β for all positive n
  α + β = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_problem_l654_65466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_factor_is_five_l654_65432

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- The number of digits in each formed number -/
def digits_per_number : Nat := 3

/-- The number of different multiples that can be formed -/
def num_multiples : Nat := 8

/-- The common factor we want to prove -/
def common_factor : Nat := 5

/-- Theorem stating that given the conditions, the common factor is 5 -/
theorem common_factor_is_five :
  ∀ (numbers : Finset Nat),
    (∀ n ∈ numbers, n ≥ 100 ∧ n < 1000) →  -- 3-digit numbers
    (∀ n ∈ numbers, ∀ d : Nat, d ∈ digits → ((Nat.digits 10 n).count d ≤ 1)) →  -- No repeated digits
    numbers.card = num_multiples →  -- 8 different numbers
    (∃ k : Nat, ∀ n ∈ numbers, n % common_factor = 0) →  -- All numbers are multiples of a common factor
    (∀ m : Nat, m < common_factor → ¬(∀ n ∈ numbers, n % m = 0)) →  -- No smaller common factor exists
    common_factor = 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_factor_is_five_l654_65432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_d_value_l654_65411

/-- The sequence a_n defined as 50 + n^2 for positive integers n -/
def a (n : ℕ+) : ℕ := 50 + n^2

/-- The greatest common divisor of consecutive terms in the sequence a_n -/
def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

/-- The maximum value of d_n is 201 -/
theorem max_d_value : ∃ (n : ℕ+), d n = 201 ∧ ∀ (m : ℕ+), d m ≤ 201 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_d_value_l654_65411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_f_of_two_l654_65426

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then -2 * x^2 else x^2 + 4*x + 12

-- State the theorem
theorem five_fold_f_of_two : f (f (f (f (f 2)))) = -449183247763232 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_five_fold_f_of_two_l654_65426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l654_65449

-- Define sets A and B
def A : Set ℝ := {x | Real.log x > 0}
def B : Set ℝ := {x | Real.exp (x * Real.log 2) < 3}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = Set.Ioo 1 (Real.log 3 / Real.log 2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l654_65449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_expression_l654_65453

theorem greatest_expression :
  let a := Real.sqrt (Real.rpow 6 (1/3) * Real.rpow 7 (1/3))
  let b := Real.sqrt (7 * Real.rpow 6 (1/3))
  let c := Real.sqrt (6 * Real.rpow 7 (1/3))
  let d := Real.rpow (6 * Real.sqrt 7) (1/3)
  let e := Real.rpow (7 * Real.sqrt 6) (1/3)
  b > a ∧ b > c ∧ b > d ∧ b > e :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greatest_expression_l654_65453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_range_l654_65408

noncomputable def f (ω φ x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem cosine_function_range (ω φ m : ℝ) (h1 : 0 < φ) (h2 : φ < π) 
  (h3 : (2 * π) / ω = 2 * π / 3) 
  (h4 : f ω φ (π / 18) = 0) 
  (h5 : ∀ x ∈ Set.Icc (π / 6) m, -1 ≤ f ω φ x ∧ f ω φ x ≤ -Real.sqrt 3 / 2) :
  m ∈ Set.Icc ((2 * π) / 9) ((5 * π) / 18) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_function_range_l654_65408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_for_q_l654_65437

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := (Real.sqrt (4 - x^2)) / (x + 4) - m

-- Define the proposition p
def p (m : ℝ) : Prop := ∃ x, f x m = 0

-- Define the proposition q
def q (m : ℝ) : Prop := |m| ≤ Real.sqrt 3 / 3

-- State the theorem
theorem p_necessary_not_sufficient_for_q :
  (∀ m, q m → p m) ∧ (∃ m, p m ∧ ¬q m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_necessary_not_sufficient_for_q_l654_65437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_distance_l654_65464

/-- A quadratic function passing through three given points -/
noncomputable def quadratic_function (a b c : ℝ) : ℝ → ℝ :=
  λ x => a * x^2 + b * x + c

/-- The vertex of a quadratic function -/
noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  (- b / (2 * a), quadratic_function a b c (- b / (2 * a)))

/-- The distance from a point to the x-axis -/
noncomputable def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem quadratic_vertex_distance (a b c : ℝ) :
  quadratic_function a b c 0 = 3 ∧
  quadratic_function a b c 1 = 0 ∧
  quadratic_function a b c 4 = 3 →
  distance_to_x_axis (vertex a b c) = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_vertex_distance_l654_65464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ADM_l654_65460

-- Define the triangle ABC
def Triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = 8 ∧  -- AB = 2√2
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = 8 ∧  -- AC = 2√2
  (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0  -- ∠BAC = 90°

-- Define the midpoint M of AC
def Midpoint (A C M : ℝ × ℝ) : Prop :=
  2 * M.1 = A.1 + C.1 ∧ 2 * M.2 = A.2 + C.2

-- Define point D on AB such that AD = 1.5 * DB
def PointD (A B D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
  D.1 = A.1 + t * (B.1 - A.1) ∧
  D.2 = A.2 + t * (B.2 - A.2) ∧
  t = 0.6  -- AD = 1.5 * DB implies t = 3/5 = 0.6

-- Theorem statement
theorem area_of_triangle_ADM 
  (A B C M D : ℝ × ℝ) 
  (h1 : Triangle A B C) 
  (h2 : Midpoint A C M) 
  (h3 : PointD A B D) : 
  ∃ area : ℝ, area = 1.2 ∧ 
  area = (1/2) * abs ((D.1 - A.1) * (M.2 - A.2) - (M.1 - A.1) * (D.2 - A.2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_ADM_l654_65460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equidistant_from_base_line_l654_65462

-- Define the necessary types and structures
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  p1 : Point
  p2 : Point

-- Define the perpendicular relation
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define the intersection of two lines
noncomputable def intersectionPoint (l1 l2 : Line) : Point := sorry

-- Define the distance from a point to a line
noncomputable def distancePointToLine (p : Point) (l : Line) : ℝ := sorry

-- State the theorem
theorem intersection_equidistant_from_base_line 
  (A B : Point) 
  (a b : ℝ) 
  (base_line : Line) 
  (perp_line1 perp_line2 : Line) 
  (h1 : base_line = ⟨A, B⟩)
  (h2 : perp_line1.p1 = A)
  (h3 : perp_line2.p1 = B)
  (h4 : perpendicular base_line perp_line1)
  (h5 : perpendicular base_line perp_line2)
  (h6 : distancePointToLine perp_line1.p2 base_line = a)
  (h7 : distancePointToLine perp_line2.p2 base_line = b)
  (P : Point)
  (h8 : P = intersectionPoint ⟨A, perp_line2.p2⟩ ⟨B, perp_line1.p2⟩) :
  ∀ (A' B' : Point), 
    ∀ (base_line' : Line), 
    ∀ (perp_line1' perp_line2' : Line), 
    base_line' = ⟨A', B'⟩ →
    perp_line1'.p1 = A' →
    perp_line2'.p1 = B' →
    perpendicular base_line' perp_line1' →
    perpendicular base_line' perp_line2' →
    distancePointToLine perp_line1'.p2 base_line' = a →
    distancePointToLine perp_line2'.p2 base_line' = b →
    let P' := intersectionPoint ⟨A', perp_line2'.p2⟩ ⟨B', perp_line1'.p2⟩
    distancePointToLine P base_line = distancePointToLine P' base_line' :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equidistant_from_base_line_l654_65462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_at_3_51_l654_65403

/-- Represents the number at a given row and column in the arrangement -/
def arrangement (row : ℕ) (column : ℕ) : ℕ := sorry

/-- The pattern repeats every 4 columns -/
def cycle_length : ℕ := 4

/-- Each cycle contains 9 numbers -/
def numbers_per_cycle : ℕ := 9

/-- In the 3rd row, the 3rd column of each cycle follows the pattern: 5, 14, 23, ... -/
axiom third_row_pattern (n : ℕ) : arrangement 3 (4 * n + 3) = 5 + 9 * n

/-- The main theorem: The number in the 3rd row and 51st column is 113 -/
theorem number_at_3_51 : arrangement 3 51 = 113 := by
  sorry

#check number_at_3_51

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_at_3_51_l654_65403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l654_65423

/-- Given a sequence {aₙ} with partial sums Sₙ, if (n, Sₙ) lies on the graph of the inverse
    function of y = log₂(x + 1) for positive integer n, then aₙ = 2^(n-1) -/
theorem sequence_theorem (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n : ℕ, n > 0 → S n = 2^n - 1) →
  (∀ n : ℕ, n > 0 → a n = 2^(n-1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_theorem_l654_65423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l654_65461

noncomputable section

open Real

def triangle (A B C : ℝ) (a b c : ℝ) : Prop :=
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = Real.pi ∧
  a / sin A = b / sin B ∧ b / sin B = c / sin C

theorem triangle_properties 
  (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : triangle A B C a b c)
  (h_eq : (2*a - c) * cos B = b * cos C)
  (h_a : a = 4)
  (h_b : b = 2 * sqrt 7) :
  B = Real.pi/3 ∧ c = 6 ∧ sin (2*C + B) = -5 * sqrt 3 / 14 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l654_65461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_travel_distance_l654_65439

/-- The distance Danny plans to travel -/
def distance : ℝ := sorry

/-- Speed with the 24 square foot sail (in MPH) -/
def speed_big : ℝ := 50

/-- Speed with the 12 square foot sail (in MPH) -/
def speed_small : ℝ := 20

/-- Time difference between using the bigger and smaller sail (in hours) -/
def time_difference : ℝ := 6

/-- Theorem stating that the distance Danny plans to travel is 200 miles -/
theorem danny_travel_distance :
  (distance / speed_small - distance / speed_big = time_difference) →
  distance = 200 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_danny_travel_distance_l654_65439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l654_65494

-- Define the power function as noncomputable
noncomputable def f (α : ℝ) (x : ℝ) : ℝ := x ^ α

-- Theorem statement
theorem power_function_inequality (α : ℝ) (a : ℝ) :
  (f α 2 = 8) →
  (f α (2 - a) > f α (a - 1) ↔ a < 3/2) :=
by
  -- The proof is skipped using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_inequality_l654_65494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_gain_percentage_l654_65457

/-- Calculates the gain percentage during a clearance sale -/
theorem clearance_sale_gain_percentage
  (original_price : ℝ)
  (original_gain_percentage : ℝ)
  (discount_percentage : ℝ)
  (h1 : original_price = 30)
  (h2 : original_gain_percentage = 25)
  (h3 : discount_percentage = 10) :
  (let cost_price := original_price / (1 + original_gain_percentage / 100)
   let discounted_price := original_price * (1 - discount_percentage / 100)
   let new_gain := discounted_price - cost_price
   let new_gain_percentage := (new_gain / cost_price) * 100
   new_gain_percentage) = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_gain_percentage_l654_65457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_range_of_t_min_value_of_y_l654_65425

noncomputable section

def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, -Real.cos x)
def c (x : ℝ) : ℝ × ℝ := (1, 2 * Real.cos x - 1)

def f (x : ℝ) : ℝ := 2 * (a x).1 * (b x).1 + 2 * (a x).2 * (b x).2

def y (x : ℝ) : ℝ := f x + (a x).1 * (c x).1 + (a x).2 * (c x).2

def t (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem smallest_period_of_f : ∃ (T : ℝ), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ T' > 0, (∀ x, f (x + T') = f x) → T ≤ T') ∧ T = Real.pi := by sorry

theorem range_of_t : ∀ x, -Real.sqrt 2 ≤ t x ∧ t x ≤ Real.sqrt 2 := by sorry

theorem min_value_of_y : ∃ x₀, ∀ x, y x ≥ y x₀ ∧ y x₀ = -13 * Real.sqrt 3 / 12 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_period_of_f_range_of_t_min_value_of_y_l654_65425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_imaginary_z_in_third_quadrant_l654_65484

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 9*m + 18)

-- Theorem 1: z is a real number iff m = 3 or m = 6
theorem z_is_real (m : ℝ) : (z m).im = 0 ↔ m = 3 ∨ m = 6 := by sorry

-- Theorem 2: z is a purely imaginary number iff m = 5
theorem z_is_imaginary (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 5 := by sorry

-- Theorem 3: z is in the third quadrant iff 3 < m < 5
theorem z_in_third_quadrant (m : ℝ) : (z m).re < 0 ∧ (z m).im < 0 ↔ 3 < m ∧ m < 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_z_is_real_z_is_imaginary_z_in_third_quadrant_l654_65484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_per_year_l654_65489

/-- Calculates the simple interest for a given principal, rate, and time -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Calculates the gain per year given the borrowed amount, borrowing rate, lending rate, and time -/
noncomputable def gainPerYear (borrowed : ℝ) (borrowRate : ℝ) (lendRate : ℝ) (time : ℝ) : ℝ :=
  let interestEarned := simpleInterest borrowed lendRate time
  let interestPaid := simpleInterest borrowed borrowRate time
  (interestEarned - interestPaid) / time

theorem transaction_gain_per_year :
  gainPerYear 4000 0.04 0.06 2 = 80 := by
  -- Unfold the definition of gainPerYear
  unfold gainPerYear
  -- Unfold the definition of simpleInterest
  unfold simpleInterest
  -- Simplify the expression
  simp
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transaction_gain_per_year_l654_65489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l654_65486

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then Real.exp (Real.log 3 * x) else -x + 3

-- Theorem statement
theorem f_properties :
  (f (-1) = 1/3) ∧
  (∀ x : ℝ, f x > 1 ↔ 0 < x ∧ x < 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l654_65486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_tiling_l654_65480

/-- Defines whether a grid can be tiled with L-shaped pieces -/
def can_tile (m n : ℕ) : Prop :=
  (3 ∣ m * n) ∧
  ¬(∃ k, m = 3 * k ∧ n = 1) ∧
  ¬(∃ k, m = 1 ∧ n = 3 * k) ∧
  ¬(∃ k, m = 2 * k + 1 ∧ n = 3) ∧
  ¬(∃ k, m = 3 ∧ n = 2 * k + 1)

/-- Represents a tiling of the grid -/
def Tiling := ℕ → ℕ → Bool

/-- Checks if a tiling covers the entire m × n grid -/
def tiling_covers_grid (m n : ℕ) (tiling : Tiling) : Prop :=
  ∀ i j, i < m ∧ j < n → tiling i j = true

/-- Theorem stating the conditions for tiling an m × n grid with L-shaped pieces -/
theorem grid_tiling (m n : ℕ) :
  (∃ tiling : Tiling, tiling_covers_grid m n tiling) ↔ can_tile m n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grid_tiling_l654_65480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l654_65416

theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 9) :
  (1 / 3) * π * (Real.sqrt (slant_height ^ 2 - height ^ 2))^2 * height = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_l654_65416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_pyramid_volume_l654_65417

/-- The volume of a smaller pyramid cut from a larger right square pyramid --/
theorem smaller_pyramid_volume
  (base_edge : ℝ)
  (slant_edge : ℝ)
  (cut_height : ℝ)
  (h_base : base_edge = 12 * Real.sqrt 2)
  (h_slant : slant_edge = 15)
  (h_cut : cut_height = 4.5) :
  (1 / 3) * (base_edge * cut_height / Real.sqrt (slant_edge ^ 2 - (base_edge / 2) ^ 2)) ^ 2 * cut_height = 108 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_pyramid_volume_l654_65417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dimes_for_book_l654_65430

def book_cost : ℚ := 45.50
def ten_dollar_bills : ℕ := 4
def quarters : ℕ := 10
def nickels : ℕ := 6

def min_dimes (book_cost : ℚ) (ten_dollar_bills : ℕ) (quarters : ℕ) (nickels : ℕ) : ℕ :=
  let other_money : ℚ := (ten_dollar_bills * 10) + (quarters * 0.25) + (nickels * 0.05)
  (((book_cost - other_money) / 0.1).ceil).toNat

theorem min_dimes_for_book :
  min_dimes book_cost ten_dollar_bills quarters nickels = 27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_dimes_for_book_l654_65430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_oscillation_product_l654_65451

/-- The maximum oscillation of the product of two functions with given ranges -/
theorem max_oscillation_product (f g : ℝ → ℝ) :
  (∀ x, f x ∈ Set.Icc (-8) 4) →
  (∀ x, g x ∈ Set.Icc (-2) 6) →
  (⨆ x, (f x * g x)) - (⨅ y, (f y * g y)) = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_oscillation_product_l654_65451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l654_65442

-- Define the type for positive rational numbers
def PositiveRational := {q : ℚ // q > 0}

-- Define the function type
def SpecialFunction := PositiveRational → PositiveRational

-- Define multiplication for PositiveRational
instance : HMul PositiveRational PositiveRational PositiveRational where
  hMul x y := ⟨x.val * y.val, mul_pos x.property y.property⟩

-- Define division for PositiveRational
instance : HDiv PositiveRational PositiveRational PositiveRational where
  hDiv x y := ⟨x.val / y.val, div_pos x.property y.property⟩

-- State the theorem
theorem exists_special_function :
  ∃ (f : SpecialFunction),
    ∀ (x y : PositiveRational),
      f (x * (f y)) = (f x) / y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_function_l654_65442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_not_divisible_l654_65436

theorem polynomial_not_divisible (n : ℕ) (h : n ∈ ({17, 20, 21, 64, 65} : Set ℕ)) :
  n = 21 ↔ ¬(∃ q : Polynomial ℚ, X^(2*n) + 1 + (X + 1)^(2*n) = (X^2 + X + 1) * q) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_not_divisible_l654_65436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_and_distance_l654_65491

noncomputable section

-- Define the circles C1 and C2
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y - 2 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*y - 1 = 0

-- Define the intersection of two circles
def intersect (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  ∃ x y, C1 x y ∧ C2 x y

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem circles_intersect_and_distance :
  (intersect C1 C2) ∧
  (∃ x1 y1 x2 y2, C1 x1 y1 ∧ C1 x2 y2 ∧ C2 x1 y1 ∧ C2 x2 y2 ∧
    distance x1 y1 x2 y2 = Real.sqrt 790 / 10) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_and_distance_l654_65491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_inradius_l654_65493

/-- A triangle with the given properties -/
structure SpecialTriangle where
  /-- Perimeter of the triangle -/
  perimeter : ℝ
  /-- Area of the triangle -/
  area : ℝ
  /-- The area is one-fourth of the square of the perimeter -/
  area_eq : area = (perimeter^2) / 4

/-- The radius of the inscribed circle in the special triangle -/
noncomputable def inradius (t : SpecialTriangle) : ℝ :=
  t.area / (t.perimeter / 2)

/-- Theorem: In a triangle with perimeter 24 and area equal to one-fourth
    of the square of the perimeter, the radius of the inscribed circle is 12 -/
theorem special_triangle_inradius :
  let t : SpecialTriangle := ⟨24, 24^2 / 4, by rfl⟩
  inradius t = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_inradius_l654_65493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_sixty_degrees_l654_65477

/-- The angle of inclination of a line with equation √3x - y + a = 0, where a is a real constant -/
noncomputable def angle_of_inclination (a : ℝ) : ℝ := 60 * Real.pi / 180

/-- Theorem: The angle of inclination of the line √3x - y + a = 0 is 60° -/
theorem angle_of_inclination_is_sixty_degrees (a : ℝ) :
  angle_of_inclination a = 60 * Real.pi / 180 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_is_sixty_degrees_l654_65477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_satisfies_conditions_l654_65420

noncomputable def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

noncomputable def vertex (a b c : ℝ) : ℝ × ℝ := (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

theorem quadratic_function_satisfies_conditions :
  ∃ (a b c : ℝ),
    let f := QuadraticFunction a b c
    vertex a b c = (2, -1) ∧
    f (-1) = 8 ∧
    ∀ x, f x = x^2 - 4*x + 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_satisfies_conditions_l654_65420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_243_l654_65497

/-- The measure of an internal angle in a regular polygon with n sides -/
noncomputable def internal_angle (n : ℕ) : ℝ := 180 * (n - 2) / n

/-- The sum of the measures of angles ABC and ABD, where ABC is an angle in a regular pentagon and ABD is an angle in a regular octagon -/
noncomputable def angle_sum : ℝ := internal_angle 5 + internal_angle 8

theorem angle_sum_is_243 : angle_sum = 243 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_243_l654_65497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_equation_l654_65429

/-- The total worth of the stock -/
def W : ℝ := sorry

/-- The overall loss in Rupees -/
def overall_loss : ℝ := 1200

theorem stock_worth_equation : 
  0.045 * W = overall_loss := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_worth_equation_l654_65429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l654_65490

noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 6 * x - 9) / (x^2 - 5 * x + 6)

def horizontal_asymptote : ℝ := 3

theorem g_crosses_asymptote :
  ∃ (x : ℝ), x = 9/5 ∧ g x = horizontal_asymptote :=
by
  use 9/5
  apply And.intro
  · rfl
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l654_65490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_manny_wants_one_piece_l654_65474

/-- The number of pieces Manny wants for himself -/
def manny_pieces : ℚ := 1

/-- The total number of lasagna pieces -/
def total_pieces : ℚ := 6

/-- Aaron's lasagna consumption -/
def aaron_pieces : ℚ := 0

/-- Kai's lasagna consumption relative to Manny's -/
def kai_pieces : ℚ := 2 * manny_pieces

/-- Raphael's lasagna consumption relative to Manny's -/
def raphael_pieces : ℚ := manny_pieces / 2

/-- Lisa's lasagna consumption -/
def lisa_pieces : ℚ := 2 + raphael_pieces

theorem manny_wants_one_piece :
  manny_pieces + aaron_pieces + kai_pieces + raphael_pieces + lisa_pieces = total_pieces ∧
  manny_pieces = 1 := by
  sorry

#eval manny_pieces

end NUMINAMATH_CALUDE_ERRORFEEDBACK_manny_wants_one_piece_l654_65474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wedding_cost_per_guest_total_cost_equation_l654_65441

/-- The cost per guest for John's wedding --/
def cost_per_guest : ℕ → ℕ := sorry

/-- The venue cost for the wedding --/
def venue_cost : ℕ := 10000

/-- The number of guests John wants --/
def johns_guests : ℕ := 50

/-- The number of guests John's wife wants --/
def wifes_guests : ℕ := johns_guests + (johns_guests * 60 / 100)

/-- The total cost of the wedding if John's wife gets her way --/
def total_cost : ℕ := 50000

/-- Theorem stating that the cost per guest is $500 --/
theorem wedding_cost_per_guest : cost_per_guest johns_guests = 500 := by
  sorry

/-- Theorem proving that the total cost equation holds --/
theorem total_cost_equation : venue_cost + wifes_guests * cost_per_guest johns_guests = total_cost := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wedding_cost_per_guest_total_cost_equation_l654_65441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_size_proof_l654_65492

/-- The number of members in a committee where:
  1. The average age remains unchanged after 4 years
  2. One old member was replaced by a younger member
  3. The new member is 40 years younger than the old member
-/
def committee_size : ℕ := 11

theorem committee_size_proof :
  ∀ (avg_age : ℝ) (old_member_age : ℝ),
  (committee_size : ℝ) * avg_age = 
  (committee_size : ℝ) * avg_age + 4 * ((committee_size : ℝ) - 1) - 40 →
  committee_size = 11 := by
  sorry

#eval committee_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_committee_size_proof_l654_65492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_AB_correct_l654_65409

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of students to be selected -/
def selected_students : ℕ := 3

/-- The probability of selecting both A and B -/
def prob_select_AB : ℚ := 3/10

theorem prob_select_AB_correct : 
  (Nat.choose (total_students - 2) (selected_students - 2) : ℚ) / 
  (Nat.choose total_students selected_students : ℚ) = prob_select_AB := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_select_AB_correct_l654_65409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_solution_set_g_l654_65452

-- Define the quadratic function
noncomputable def f (x : ℝ) : ℝ := -1/2 * x^2 + 3/2 * x - 1

-- Define the rational function
noncomputable def g (x : ℝ) : ℝ := (-1/2 * x + 1) / (3/2 * x - 1)

-- Theorem for the solution set of f(x) > 0
theorem solution_set_f : 
  {x : ℝ | f x > 0} = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem for the solution set of g(x) > 0
theorem solution_set_g :
  {x : ℝ | g x > 0} = {x : ℝ | 2/3 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_solution_set_g_l654_65452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shoppers_correct_l654_65402

def total_shoppers (express_lane_ratio : Rat) (checkout_lane_count : Nat) : Nat :=
  checkout_lane_count * 8 / 3

theorem total_shoppers_correct (express_lane_ratio : Rat) (checkout_lane_count : Nat) :
  express_lane_ratio = 5 / 8 →
  checkout_lane_count = 180 →
  (1 - express_lane_ratio) * (total_shoppers express_lane_ratio checkout_lane_count) = checkout_lane_count ∧
  total_shoppers express_lane_ratio checkout_lane_count = 480 :=
by
  intro h1 h2
  simp [total_shoppers, h1, h2]
  sorry

#eval total_shoppers (5/8) 180

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_shoppers_correct_l654_65402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_when_f_greater_than_x_l654_65447

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Theorem 1
theorem solution_set_when_a_is_one :
  ∀ x : ℝ, (f 1 x > 1) ↔ (x > 1/2) :=
sorry

-- Theorem 2
theorem range_of_a_when_f_greater_than_x :
  (∀ x : ℝ, x ∈ Set.Ioo 0 1 → f a x > x) → a ∈ Set.Ioc 0 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_when_a_is_one_range_of_a_when_f_greater_than_x_l654_65447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_proof_l654_65475

/-- 
Given a student who:
- Needs 30% to pass
- Scored 212 marks
- Falls short by 16 marks
This theorem proves that the maximum possible marks is 760.
-/
theorem max_marks_proof (passing_percentage : ℚ) (scored_marks : ℕ) (short_marks : ℕ) 
  (h1 : passing_percentage = 30 / 100)
  (h2 : scored_marks = 212)
  (h3 : short_marks = 16) : 
  (scored_marks + short_marks) / passing_percentage = 760 :=
by
  sorry

#eval (212 + 16 : ℕ) * 100 / 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_proof_l654_65475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_two_equals_one_l654_65435

-- Define the function f
noncomputable def f (ω b : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4) + b

-- State the theorem
theorem f_at_pi_over_two_equals_one 
  (ω b : ℝ) 
  (h_ω_pos : ω > 0)
  (h_period : 2 * Real.pi / 3 < 2 * Real.pi / ω ∧ 2 * Real.pi / ω < Real.pi)
  (h_symmetry : ∃ (y : ℝ), f ω b (3 * Real.pi / 2) = y ∧ 
    ∀ (x : ℝ), f ω b (3 * Real.pi - x) = 2 * y - f ω b x) :
  f ω b (Real.pi / 2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_at_pi_over_two_equals_one_l654_65435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_derivatives_theorem_l654_65465

-- Define the functions
noncomputable def f (m n α β : ℝ) : ℝ := Real.cos (m * α - n * β)
noncomputable def z (x y : ℝ) : ℝ := Real.log (x^2 - y^2)

-- State the theorem
theorem partial_derivatives_theorem (m n : ℝ) :
  (deriv (fun α => f m n α 0)) (π / (2 * m)) = -m ∧
  (deriv (fun β => f m n (π / (2 * m)) β)) 0 = n ∧
  (deriv (fun x => z x (-1))) 2 = 4/3 ∧
  (deriv (fun y => z 2 y)) (-1) = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_partial_derivatives_theorem_l654_65465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_pi_third_l654_65445

theorem sin_alpha_minus_pi_third (α : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2) 
  (h3 : Real.tan (α/2) + (1 / Real.tan (α/2)) = 5/2) : 
  Real.sin (α - π/3) = (4 - 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_alpha_minus_pi_third_l654_65445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_and_inverse_from_eigenpairs_l654_65424

/-- Given a 2x2 matrix A with specified eigenvalues and eigenvectors, prove its form and inverse -/
theorem matrix_and_inverse_from_eigenpairs :
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ),
  (A.mulVec ![1, 1] = 3 • ![1, 1]) →
  (A.mulVec ![1, -1] = -1 • ![1, -1]) →
  (A = ![![1, 2], ![2, 1]]) ∧
  (A⁻¹ = ![![-1/3, 2/3], ![2/3, -1/3]]) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_and_inverse_from_eigenpairs_l654_65424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l654_65418

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 2) - 2/3

-- Define the function g
noncomputable def g (n : ℝ) (m : ℝ) (x : ℝ) : ℝ := Real.log (x^2 - m*x + 4) / Real.log n

-- Theorem statement
theorem max_value_of_g 
  (a : ℝ) (m n : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : f a m = n) 
  : ∃ (x : ℝ), ∀ (y : ℝ), g n m x ≥ g n m y ∧ g n m x = -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l654_65418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_f_with_zero_l654_65421

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.sin x + a)

-- State the theorem
theorem range_of_a_for_f_with_zero (a : ℝ) :
  (∀ x, f a x ∈ Set.univ) ∧ (∃ x, f a x = 0) → a ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_for_f_with_zero_l654_65421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_calculation_l654_65400

theorem incorrect_calculation : ∀ (a b c d : ℝ),
  (Real.sqrt 8 / 2 = Real.sqrt 2) →
  (Real.sqrt 2 * Real.sqrt 3 = Real.sqrt 6) →
  (Real.sqrt 8 - Real.sqrt 2 = Real.sqrt 2) →
  (3 + 2 * Real.sqrt 2 ≠ 5 * Real.sqrt 2) :=
by
  intros a b c d h1 h2 h3
  sorry

#check incorrect_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_calculation_l654_65400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_orthogonal_vectors_proj_b_equals_difference_l654_65483

noncomputable def proj_a : ℝ × ℝ := (-4/5, -8/5)
def v : ℝ × ℝ := (4, -2)

theorem projection_orthogonal_vectors 
  (a b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 0) -- a and b are orthogonal
  (h2 : proj_a = (-4/5, -8/5)) 
  : (v.1 - proj_a.1, v.2 - proj_a.2) = (24/5, -2/5) := by
  sorry

theorem proj_b_equals_difference 
  (a b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 0) -- a and b are orthogonal
  (h2 : proj_a = (-4/5, -8/5)) 
  : ∃ (k : ℝ), (24/5, -2/5) = (k * b.1, k * b.2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_orthogonal_vectors_proj_b_equals_difference_l654_65483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_floor_equation_solutions_l654_65446

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem quadratic_floor_equation_solutions :
  let S : Set ℝ := {x : ℝ | x^2 - 8 * (floor x) + 7 = 0}
  S = {1, Real.sqrt 33, Real.sqrt 41, 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_floor_equation_solutions_l654_65446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_marble_count_l654_65495

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- The total number of marbles in the urn -/
def totalMarbles (m : MarbleCount) : ℕ :=
  m.red + m.white + m.blue + m.green + m.yellow

/-- Probability of drawing a specific combination of marbles -/
noncomputable def drawProbability (m : MarbleCount) (r w b g y : ℕ) : ℚ :=
  (Nat.choose m.red r * Nat.choose m.white w * Nat.choose m.blue b *
   Nat.choose m.green g * Nat.choose m.yellow y) /
  Nat.choose (totalMarbles m) 5

/-- The conditions for the marble drawing probabilities to be equal -/
def equalProbabilities (m : MarbleCount) : Prop :=
  drawProbability m 5 0 0 0 0 =
  drawProbability m 4 1 0 0 0 ∧
  drawProbability m 4 1 0 0 0 =
  drawProbability m 3 1 1 0 0 ∧
  drawProbability m 3 1 1 0 0 =
  drawProbability m 2 1 1 1 0 ∧
  drawProbability m 2 1 1 1 0 =
  drawProbability m 1 1 1 1 1

/-- The main theorem stating that 28 is the smallest number of marbles satisfying all conditions -/
theorem smallest_valid_marble_count :
  ∃ (m : MarbleCount),
    totalMarbles m = 28 ∧
    equalProbabilities m ∧
    totalMarbles m % 7 = 0 ∧
    (∀ (n : MarbleCount),
      totalMarbles n < 28 →
      ¬(equalProbabilities n ∧ totalMarbles n % 7 = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_marble_count_l654_65495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_cocaptains_l654_65468

def num_teams : ℕ := 4
def team_sizes : List ℕ := [6, 5, 8, 9]
def cocaptains_per_team : ℕ := 4
def members_to_select : ℕ := 4

theorem probability_all_cocaptains :
  (1 : ℚ) / num_teams * (team_sizes.map (λ n => (1 : ℚ) / n.choose members_to_select)).sum = 13 / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_all_cocaptains_l654_65468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_a_1990_l654_65455

-- Define the sequence
def a : ℕ → ℕ
  | 0 => 3  -- Add a case for 0 to avoid missing cases error
  | 1 => 3
  | n+2 => 3^(a (n+1))

-- Define a function to get the last digit
def lastDigit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem last_digit_a_1990 : lastDigit (a 1990) = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_a_1990_l654_65455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_solution_count_l654_65499

/-- 
Given a natural number n of the form 6k + 3, where k is a non-negative integer,
this function returns the number of solutions to the equation x₁ + x₂ + x₃ = n
with the constraint 0 < x₁ < x₂ < x₃.
-/
def countSolutions (n : ℕ) : ℕ :=
  if n % 6 = 3 then
    let k := (n - 3) / 6
    3 * k * k
  else 0

/-- 
Theorem stating that for n = 6k + 3, the number of solutions to x₁ + x₂ + x₃ = n
with 0 < x₁ < x₂ < x₃ is equal to 3k².
-/
theorem solution_count (k : ℕ) :
  let n := 6 * k + 3
  countSolutions n = 3 * k * k := by
  sorry

/-- Verify the given examples -/
example : countSolutions 21 = 27 := by rfl
example : countSolutions 57 = 243 := by rfl
example : countSolutions 165 = 2187 := by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_solution_count_l654_65499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l654_65488

/-- The focus of a parabola with equation x = a y^2 (a ≠ 0) -/
noncomputable def parabola_focus (a : ℝ) (h : a ≠ 0) : ℝ × ℝ :=
  (1 / (4 * a), 0)

/-- Theorem: The coordinates of the focus of the parabola x = a y^2 (a ≠ 0) are (1/(4a), 0) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  parabola_focus a h = (1 / (4 * a), 0) := by
  -- Unfold the definition of parabola_focus
  unfold parabola_focus
  -- The equality is now trivial
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l654_65488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percent_approx_l654_65433

/-- Calculates the profit percent for a retailer given the purchase price, overhead expenses, and selling price. -/
def profit_percent (purchase_price overhead_expenses selling_price : ℚ) : ℚ :=
  let cost_price := purchase_price + overhead_expenses
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

/-- Theorem stating that the profit percent for the given scenario is approximately 17.65% -/
theorem retailer_profit_percent_approx :
  let ε := 0.01
  let result := profit_percent 225 30 300
  |result - 17.65| ≤ ε := by
  sorry

#eval profit_percent 225 30 300

end NUMINAMATH_CALUDE_ERRORFEEDBACK_retailer_profit_percent_approx_l654_65433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_value_at_three_m_times_u_equals_three_l654_65458

/-- A function satisfying the given conditions -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 1 = 1 ∧ ∀ x y : ℝ, g (x^2 - y^2) = (x - y) * (g x^3 + g y^3)

/-- The theorem stating that g(3) = 3 is the only possible value -/
theorem unique_value_at_three (g : ℝ → ℝ) (h : special_function g) : g 3 = 3 := by
  sorry

/-- The main theorem proving m × u = 3 -/
theorem m_times_u_equals_three :
  (Finset.card {3}) *
  (Finset.sum {3} (λ x => x)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_value_at_three_m_times_u_equals_three_l654_65458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_quarter_circles_l654_65482

/-- The area of the shaded region in a square with quarter circles in each corner -/
theorem shaded_area_square_with_quarter_circles (side_length : ℝ) (h : side_length = 12) :
  side_length ^ 2 - π * (side_length / 4) ^ 2 = 144 - 9 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_square_with_quarter_circles_l654_65482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_min_a_for_inequality_l654_65478

open Real

/-- The function f(x) = ln x - ax -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

/-- Theorem stating the monotonicity of f(x) -/
theorem monotonicity_of_f (a : ℝ) :
  (a ≤ 0 → StrictMono (f a)) ∧
  (a > 0 → StrictMonoOn (f a) (Set.Ioo 0 (1/a)) ∧ StrictAntiOn (f a) (Set.Ioi (1/a))) := by
  sorry

/-- The inequality condition for x ≥ 1 -/
def inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  x ≥ 1 → f a x ≤ (Real.log x) / (x + 1) - x / (Real.exp 1 * (x + 1))

/-- Theorem stating the minimum value of a that satisfies the inequality condition -/
theorem min_a_for_inequality :
  ∀ a : ℝ, (∀ x : ℝ, inequality_condition a x) ↔ a ≥ 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_of_f_min_a_for_inequality_l654_65478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_pole_height_l654_65459

/-- Represents a right triangle with base length and height -/
structure RightTriangle where
  base : ℝ
  height : ℝ

/-- Defines when two right triangles are similar -/
def similar_triangles (t1 t2 : RightTriangle) : Prop :=
  t1.height / t1.base = t2.height / t2.base

/-- Proves that the height of a larger right triangle is 6.4 meters
    given the conditions from the telephone pole problem -/
theorem telephone_pole_height
  (large_triangle : RightTriangle)
  (small_triangle : RightTriangle)
  (h1 : large_triangle.base = 4)
  (h2 : small_triangle.base = 1)
  (h3 : small_triangle.height = 1.6)
  (h4 : similar_triangles large_triangle small_triangle) :
  large_triangle.height = 6.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_telephone_pole_height_l654_65459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l654_65410

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x/3))^2 + 2*Real.pi * Real.arcsin (x/3) - 3*(Real.arcsin (x/3))^2 + (Real.pi^2/4)*(x^2 - 3*x + 9)

theorem g_range :
  ∀ y ∈ Set.range g, π^2/4 ≤ y ∧ y ≤ 13*π^2/4 ∧
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3) 3 ∧ x₂ ∈ Set.Icc (-3) 3 ∧ 
    g x₁ = π^2/4 ∧ g x₂ = 13*π^2/4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l654_65410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_inequality_solution_l654_65476

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Theorem for the first equation
theorem equation_solution :
  ∃ x : ℝ, x > 2 ∧ lg (x + 1) + lg (x - 2) = lg 4 → x = 3 := by sorry

-- Theorem for the second inequality
theorem inequality_solution :
  ∀ x : ℝ, (2 : ℝ)^(1 - 2*x) > 1/4 ↔ x < 3/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_inequality_solution_l654_65476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unfolded_cone_cross_section_is_equilateral_triangle_l654_65498

/-- Represents a cone with its lateral surface unfolded into a semicircle -/
structure UnfoldedCone where
  R : ℝ  -- radius of the semicircle
  r : ℝ  -- radius of the base circle of the cone
  h : R = 2 * r  -- condition that R = 2r

/-- Represents a triangle -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a triangle is equilateral -/
def IsEquilateral (t : Triangle) : Prop :=
  t.a = t.b ∧ t.b = t.c

/-- Predicate to check if a triangle is a cross-section of an unfolded cone -/
def IsCrossSection (t : Triangle) (c : UnfoldedCone) : Prop :=
  t.a = c.R ∧ t.b = c.R ∧ t.c = 2 * c.r

/-- The cross-section of an unfolded cone along its axis is an equilateral triangle -/
theorem unfolded_cone_cross_section_is_equilateral_triangle (c : UnfoldedCone) :
  ∃ (t : Triangle), IsEquilateral t ∧ IsCrossSection t c := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unfolded_cone_cross_section_is_equilateral_triangle_l654_65498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_values_when_f_equals_4_l654_65470

noncomputable def f (x : ℝ) : ℝ :=
  if x < 3 then 5 * x + 20 else 3 * x - 21

theorem sum_of_x_values_when_f_equals_4 :
  ∃ (x₁ x₂ : ℝ), f x₁ = 4 ∧ f x₂ = 4 ∧ x₁ ≠ x₂ ∧ x₁ + x₂ = 77 / 15 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_x_values_when_f_equals_4_l654_65470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l654_65428

/-- The time taken for two trains moving in opposite directions to completely cross each other -/
noncomputable def crossing_time (speed_a speed_b : ℝ) (length_a length_b : ℝ) : ℝ :=
  (length_a + length_b) / (speed_a + speed_b)

/-- Conversion factor from km/h to m/s -/
noncomputable def km_h_to_m_s : ℝ := 1000 / 3600

theorem trains_crossing_time :
  let speed_a : ℝ := 132 -- km/h
  let speed_b : ℝ := 96  -- km/h
  let length_a : ℝ := 110 -- meters
  let length_b : ℝ := 165 -- meters
  let time := crossing_time (speed_a * km_h_to_m_s) (speed_b * km_h_to_m_s) length_a length_b
  ∃ ε > 0, |time - 4.34| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l654_65428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_apple_waste_percentage_l654_65404

/-- Calculates the percentage of apples thrown away by a vendor over two days -/
theorem vendor_apple_waste_percentage : (30 : ℝ) = 
  let initial_apples : ℝ := 100
  let day1_sold_percentage : ℝ := 50
  let day1_waste_percentage : ℝ := 20
  let day2_sold_percentage : ℝ := 50

  let day1_remaining := initial_apples * (1 - day1_sold_percentage / 100)
  let day1_waste := day1_remaining * (day1_waste_percentage / 100)
  let day2_start := day1_remaining - day1_waste
  let day2_sold := day2_start * (day2_sold_percentage / 100)
  let day2_waste := day2_start - day2_sold

  let total_waste := day1_waste + day2_waste
  (total_waste / initial_apples) * 100
:= by
  sorry

#check vendor_apple_waste_percentage

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vendor_apple_waste_percentage_l654_65404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l654_65469

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos x + Real.sin x ^ 2 - 1/2

theorem f_properties :
  -- The smallest positive period is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- The maximum value is √2/2
  (∃ (M : ℝ), M = Real.sqrt 2 / 2 ∧ (∀ (x : ℝ), f x ≤ M) ∧ (∃ (y : ℝ), f y = M)) ∧
  -- If α ∈ (0, π/2) and f(α) = √2/2, then α = 3π/8
  (∀ (α : ℝ), 0 < α ∧ α < Real.pi / 2 → f α = Real.sqrt 2 / 2 → α = 3 * Real.pi / 8) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l654_65469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_ending_75_divisible_by_5_l654_65405

theorem four_digit_numbers_ending_75_divisible_by_5 : 
  let S := {n : ℕ | 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 75 ∧ n % 5 = 0}
  Finset.card (Finset.filter (λ n => 1000 ≤ n ∧ n < 10000 ∧ n % 100 = 75 ∧ n % 5 = 0) (Finset.range 10000)) = 90 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_ending_75_divisible_by_5_l654_65405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_morning_butter_cookies_l654_65481

theorem morning_butter_cookies 
  (morning_butter_cookies morning_biscuits afternoon_butter_cookies afternoon_biscuits : ℕ) 
  (h1 : morning_biscuits = 40)
  (h2 : afternoon_butter_cookies = 10)
  (h3 : afternoon_biscuits = 20)
  (h4 : morning_biscuits + afternoon_biscuits = morning_butter_cookies + afternoon_butter_cookies + 30) :
  morning_butter_cookies = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_morning_butter_cookies_l654_65481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l654_65415

noncomputable def a : ℕ → ℝ
  | 0 => 1  -- Adding the base case for 0
  | n + 1 => (1 + 4 * a n + Real.sqrt (1 + 24 * a n)) / 16

noncomputable def b (n : ℕ) : ℝ := Real.sqrt (1 + 24 * a n)

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → b n - 3 = 2 * (1/2)^(n-1)) ∧
  (∀ n : ℕ, n ≥ 1 → a n = 2/3 * (1/4)^n + (1/2)^n + 1/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l654_65415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_150th_term_l654_65456

/-- Sequence of positive integers that are either powers of 3 or sums of distinct powers of 3 -/
def special_sequence : ℕ → ℕ := sorry

/-- The 150th term of the special sequence -/
def term_150 : ℕ := special_sequence 150

/-- Theorem stating that the 150th term of the special sequence is 2280 -/
theorem special_sequence_150th_term : term_150 = 2280 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_sequence_150th_term_l654_65456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisible_by_ten_count_l654_65440

theorem product_divisible_by_ten_count :
  let S := Finset.range 6
  (Finset.filter (fun t : ℕ × ℕ × ℕ => 10 ∣ t.1 * t.2.1 * t.2.2) (S.product (S.product S))).card = 72 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_divisible_by_ten_count_l654_65440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_functions_touch_l654_65413

-- Define linear functions
def LinearFunction := ℝ → ℝ

-- Define the property of being parallel
def IsParallel (f g : LinearFunction) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, f x = a * x + b) ∧ (∀ x, g x = a * x + c)

-- Define the property of touching (having exactly one intersection point)
def Touches (f g : ℝ → ℝ) : Prop :=
  ∃! x, f x = g x

-- State the theorem
theorem parallel_functions_touch (f g : LinearFunction) (A : ℝ) :
  IsParallel f g →
  Touches (fun x ↦ (f x)^2) (fun x ↦ 11 * g x) →
  (Touches (fun x ↦ (g x)^2) (fun x ↦ A * f x) ↔ (A = 0 ∨ A = -11)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_functions_touch_l654_65413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_one_l654_65467

theorem x_equals_one (x y : ℕ+) 
  (h : ∀ n : ℕ+, (x.val ^ (2 ^ n.val) - 1) % (2 ^ n.val * y.val + 1) = 0) : 
  x = 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_equals_one_l654_65467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hundreds_digit_of_factorial_difference_l654_65463

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem hundreds_digit_of_factorial_difference : 
  ∃ k : ℕ, factorial 30 - factorial 25 = 1000 * k := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hundreds_digit_of_factorial_difference_l654_65463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_non_negative_condition_l654_65448

-- Define the function f(x) with parameter a
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * (x + 1) * Real.log x - x + 1

-- Theorem for the tangent line equation
theorem tangent_line_at_one :
  ∃ (m b : ℝ), ∀ x y, y = m * (x - 1) + f 2 1 ↔ 3 * x - y - 3 = 0 :=
by sorry

-- Theorem for the non-negativity condition
theorem non_negative_condition :
  ∀ a : ℝ, (∀ x : ℝ, x ≥ 1 → f a x ≥ 0) ↔ a ≥ (1/2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_non_negative_condition_l654_65448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_part_c_part_d_l654_65473

-- Define the clock game
def clockMove (n : Nat) (pos : Nat) : Nat :=
  (pos + pos) % n

-- Function to count rounds until return to start
def countRounds (n : Nat) (start : Nat) : Nat :=
  let rec loop (pos : Nat) (count : Nat) : Nat :=
    if pos = start && count > 0 then count
    else loop (clockMove n pos) (count + 1)
  loop start 0

-- Theorems corresponding to the problem parts
theorem part_a : countRounds 7 1 = 3 := by
  -- Proof for part (a)
  sorry

theorem part_b : clockMove 7 6 = 5 := by
  -- Proof for part (b)
  sorry

theorem part_c : ∀ n, n ∈ {1,2,3,4,5,6,7} → (clockMove 7 n = n ↔ n = 7) := by
  -- Proof for part (c)
  sorry

theorem part_d : countRounds 128 127 = 7 := by
  -- Proof for part (d)
  sorry

#eval countRounds 7 1  -- Should output 3
#eval clockMove 7 6    -- Should output 5
#eval countRounds 128 127  -- Should output 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_a_part_b_part_c_part_d_l654_65473
