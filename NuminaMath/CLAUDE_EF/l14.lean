import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_value_of_u_on_unit_interval_l14_1401

-- Define the function u(x)
noncomputable def u (x : ℝ) : ℝ := 1 / Real.sqrt x

-- Define the average value of a function on an interval
noncomputable def average_value (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  (∫ x in a..b, f x) / (b - a)

-- State the theorem
theorem average_value_of_u_on_unit_interval :
  average_value u 0 1 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_value_of_u_on_unit_interval_l14_1401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l14_1478

/-- The time (in days) it takes for two workers to complete a job together, 
    given their individual work rates. -/
noncomputable def time_to_complete (rate_a : ℝ) (rate_b : ℝ) : ℝ :=
  1 / (rate_a + rate_b)

/-- The rate at which a worker completes a job, given the time it takes them to finish. -/
noncomputable def rate_from_time (days : ℝ) : ℝ :=
  1 / days

theorem job_completion_time 
  (rate_b : ℝ) 
  (h1 : rate_b = rate_from_time 10) 
  (h2 : rate_a = 2 * rate_b) : 
  time_to_complete rate_a rate_b = 10 / 3 := by
  sorry

#check job_completion_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l14_1478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l14_1444

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * a * x^2 + (1/2) * a * x + 1

def has_extremum (f : ℝ → ℝ) : Prop := ∃ x : ℝ, ∀ y : ℝ, f y ≤ f x ∨ f y ≥ f x

def original_proposition (a : ℝ) : Prop := a = 1 → ¬(has_extremum (f a))

def converse_proposition (a : ℝ) : Prop := ¬(has_extremum (f a)) → a = 1

def inverse_proposition (a : ℝ) : Prop := a ≠ 1 → has_extremum (f a)

def contrapositive_proposition (a : ℝ) : Prop := has_extremum (f a) → a ≠ 1

theorem two_true_propositions :
  (original_proposition 1 ∧ contrapositive_proposition 1) ∧
  ¬(converse_proposition 1 ∨ inverse_proposition 1) :=
by
  sorry

#check two_true_propositions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_true_propositions_l14_1444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_range_l14_1452

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then x + 1/x + a else Real.exp (x * Real.log 2) + a

theorem unique_solution_range (a : ℝ) :
  (∃! x, f a x = -x) ↔ (a ≥ -1 ∨ a = -2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_range_l14_1452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l14_1436

/-- The number of students who solved only problem A -/
def x : ℕ := sorry

/-- The number of students who solved A and at least one other problem -/
def y : ℕ := sorry

/-- The number of students who solved only problem B -/
def b : ℕ := sorry

/-- The number of students who solved only problem C -/
def c : ℕ := sorry

/-- The number of students who solved both problems B and C -/
def d : ℕ := sorry

theorem problem_solution :
  -- Total number of students
  x + y + b + c + d = 25 →
  -- Each student solved at least one problem
  x > 0 ∧ y > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →
  -- Among students who didn't solve A, twice as many solved B as those who solved C
  b + d = 2 * (c + d) →
  -- One more student solved only A than the number of other students who also solved A
  x = y + 1 →
  -- Half of the students who solved only one problem did not solve A
  b + c = (x + b + c) / 2 →
  -- The number of students who solved only problem B is 6
  b = 6 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l14_1436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_finds_bus_probability_l14_1474

/-- The probability of Joe finding the bus at the station -/
noncomputable def probability_joe_finds_bus : ℝ := 5 / 18

/-- The time range in minutes from 1:00 PM to 2:30 PM -/
def time_range : ℝ := 90

/-- The waiting time of the bus in minutes -/
def bus_wait_time : ℝ := 30

/-- Theorem stating the probability of Joe finding the bus at the station -/
theorem joe_finds_bus_probability :
  probability_joe_finds_bus =
    (bus_wait_time * (time_range - bus_wait_time) + (1/2) * bus_wait_time^2) /
    time_range^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_joe_finds_bus_probability_l14_1474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fifteen_l14_1499

/-- Given an arithmetic sequence {a_n}, S_n is the sum of the first n terms -/
def S (n : ℕ) : ℝ := sorry

/-- The property that S_5, S_10 - S_5, and S_15 - S_10 form an arithmetic sequence -/
axiom arithmetic_property : ∃ d : ℝ, (S 10 - S 5) - (S 5) = d ∧ (S 15 - S 10) - (S 10 - S 5) = d

theorem sum_of_fifteen (h1 : S 5 = 8) (h2 : S 10 = 20) : S 15 = 36 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fifteen_l14_1499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l14_1459

-- Define the circle
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def my_line (x y : ℝ) : Prop := x + y - 2 * Real.sqrt 2 = 0

-- Define the distance function from a point to the line
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  abs (x + y - 2 * Real.sqrt 2) / Real.sqrt 2

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ min_dist : ℝ, min_dist = 1 ∧
    ∀ x y : ℝ, my_circle x y →
      ∀ px py : ℝ, my_circle px py →
        distance_to_line px py ≥ min_dist :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_circle_to_line_l14_1459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digits_make_divisible_by_four_l14_1400

/-- A function that checks if a four-digit number is divisible by 4 -/
def isDivisibleByFour (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧ n % 4 = 0

/-- The set of all single-digit numbers -/
def singleDigits : Set ℕ := {n : ℕ | n ≥ 0 ∧ n ≤ 9}

/-- The function that appends a digit to 456 -/
def appendTo456 (n : ℕ) : ℕ := 456 * 10 + n

/-- The theorem stating that there are exactly 3 digits that satisfy the condition -/
theorem three_digits_make_divisible_by_four :
  ∃! (s : Finset ℕ), s.toSet ⊆ singleDigits ∧ 
    (∀ n, n ∈ s ↔ isDivisibleByFour (appendTo456 n)) ∧
    s.card = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digits_make_divisible_by_four_l14_1400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_is_37_l14_1407

noncomputable def star (a b : ℝ) : ℝ := Real.sqrt (a + b + 36) / Real.sqrt (a - b)

theorem x_value_is_37 : ∃ x : ℝ, (star x 36 = 9) ∧ (x = 37) := by
  use 37
  apply And.intro
  · -- Proof that star 37 36 = 9
    sorry
  · -- Proof that x = 37
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_value_is_37_l14_1407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_perfect_squares_l14_1461

theorem three_digit_perfect_squares : ∃ n : ℕ, 
  n = (Finset.filter (λ x : ℕ ↦ 100 ≤ x * x ∧ x * x ≤ 999) (Finset.range 1000)).card ∧ 
  n = 22 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_perfect_squares_l14_1461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_inequality_range_minimum_value_m_l14_1457

-- Define the function f
noncomputable def f (k a x : ℝ) : ℝ := k * (a^x) - a^(-x)

-- Define the function g
noncomputable def g (a m x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m*(f 1 a x)

theorem odd_function_constant (k a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x, f k a x = -f k a (-x)) → k = 1 := by sorry

theorem inequality_range (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x, f 1 a (x+2) + f 1 a (3-2*x) > 0) → (∀ x, x > 5) := by sorry

theorem minimum_value_m (a m : ℝ) (h1 : f 1 a 1 = 8/3) :
  (∀ x ≥ 1, g a m x ≥ -2) ∧ (∃ x ≥ 1, g a m x = -2) → m = 25/12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_constant_inequality_range_minimum_value_m_l14_1457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_rolls_l14_1481

/-- Represents the number of rolls of wrapping paper sold in a fundraiser. -/
structure RollsSold where
  value : ℕ

/-- The total goal of rolls to be sold. -/
def total_goal : RollsSold := ⟨12⟩

/-- The number of rolls sold to the grandmother. -/
def sold_to_grandmother : RollsSold := ⟨3⟩

/-- The number of rolls sold to the neighbor. -/
def sold_to_neighbor : RollsSold := ⟨3⟩

/-- The number of rolls still needed to be sold. -/
def rolls_needed : RollsSold := ⟨2⟩

/-- The number of rolls sold to the uncle. -/
def sold_to_uncle : RollsSold :=
  ⟨total_goal.value - rolls_needed.value - (sold_to_grandmother.value + sold_to_neighbor.value)⟩

/-- Theorem stating that the number of rolls sold to the uncle is 4. -/
theorem uncle_rolls : sold_to_uncle.value = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_uncle_rolls_l14_1481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l14_1417

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The distance between two parallel lines -/
noncomputable def distance (l₁ l₂ : Line) : ℝ :=
  abs (l₁.c - l₂.c) / Real.sqrt (l₁.a^2 + l₁.b^2)

theorem parallel_lines_distance (a : ℝ) :
  let l₁ : Line := ⟨1, -2, 1⟩
  let l₂ : Line := ⟨2, a, -2⟩
  (l₁.a / l₁.b = -l₂.a / l₂.b) →
  distance l₁ l₂ = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_distance_l14_1417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_x_product_exists_point_p_point_p_coordinate_l14_1421

-- Define the parabola and line
def parabola (x y : ℝ) : Prop := x^2 = 6*y
def line (k x y : ℝ) : Prop := y = k*x + 3

-- Define the intersection points
def intersection_points (k : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ parabola x y ∧ line k x y}

-- Theorem 1: Product of x-coordinates is -18
theorem intersection_x_product (k : ℝ) :
  ∃ (x₁ x₂ y₁ y₂ : ℝ), (x₁, y₁) ∈ intersection_points k ∧ 
                       (x₂, y₂) ∈ intersection_points k ∧ 
                       x₁ * x₂ = -18 := by
  sorry

-- Theorem 2: Existence of point P
theorem exists_point_p :
  ∃ (b : ℝ), ∀ (k : ℝ), 
    ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (x₁, y₁) ∈ intersection_points k ∧
      (x₂, y₂) ∈ intersection_points k ∧
      (y₁ - b) / x₁ + (y₂ - b) / x₂ = 0 := by
  sorry

-- The point P has y-coordinate -3
theorem point_p_coordinate : 
  ∃ (b : ℝ), b = -3 ∧
    ∀ (k : ℝ), 
      ∃ (x₁ y₁ x₂ y₂ : ℝ), 
        (x₁, y₁) ∈ intersection_points k ∧
        (x₂, y₂) ∈ intersection_points k ∧
        (y₁ - b) / x₁ + (y₂ - b) / x₂ = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_x_product_exists_point_p_point_p_coordinate_l14_1421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_necessary_not_sufficient_l14_1466

-- Define the types for planes and lines
structure Plane where

structure Line where

-- Define the perpendicular relation between planes and between a line and a plane
axiom perpendicular_planes : Plane → Plane → Prop

axiom perpendicular_line_plane : Line → Plane → Prop

-- Define the relation of a line being in a plane
axiom line_in_plane : Line → Plane → Prop

-- Theorem statement
theorem perpendicular_planes_necessary_not_sufficient 
  (α β : Plane) (m : Line) 
  (h_diff : α ≠ β) 
  (h_m_in_α : line_in_plane m α) :
  (perpendicular_planes α β → perpendicular_line_plane m β) ∧
  ¬(perpendicular_planes α β ↔ perpendicular_line_plane m β) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_planes_necessary_not_sufficient_l14_1466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_engineering_department_men_count_l14_1440

theorem engineering_department_men_count 
  (total : ℕ) 
  (men_percentage : ℚ) 
  (women_count : ℕ) 
  (h1 : men_percentage = 70 / 100) 
  (h2 : women_count = 180) 
  (h3 : ↑women_count = (1 - men_percentage) * ↑total) :
  (men_percentage * ↑total).floor = 420 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_engineering_department_men_count_l14_1440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_arrival_time_l14_1429

/-- Represents time in 24-hour format as minutes since midnight -/
def TimeInMinutes := Nat

/-- Converts hours and minutes to TimeInMinutes -/
def toMinutes (hours minutes : Nat) : TimeInMinutes :=
  hours * 60 + minutes

/-- Adds minutes to a given time -/
def addMinutes (time delay : Nat) : TimeInMinutes :=
  (time + delay) % 1440  -- 1440 minutes in a day

/-- Theorem: A train scheduled to arrive at 11:40 and delayed by 25 minutes will arrive at 12:05 -/
theorem train_arrival_time :
  let scheduled_time := toMinutes 11 40
  let delay := 25
  addMinutes scheduled_time delay = toMinutes 12 5 := by
  -- Proof goes here
  sorry

#eval addMinutes (toMinutes 11 40) 25  -- Should output 725 (12:05 in minutes)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_arrival_time_l14_1429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l14_1403

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := (1/5)^x + 1

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := Real.log (x - 1) / Real.log (1/5)

-- Theorem statement
theorem inverse_function_proof :
  ∀ x : ℝ, x > 1 → g (f x) = x ∧ f (g x) = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l14_1403


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_6_l14_1441

/-- The function to be minimized -/
noncomputable def f (c : ℝ) : ℝ := 3/4 * c^2 - 9*c + 13

/-- The theorem stating that f is minimized when c = 6 -/
theorem f_min_at_6 : 
  ∀ x : ℝ, f 6 ≤ f x := by
  sorry

#check f_min_at_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_at_6_l14_1441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_in_rectangle_R_l14_1472

-- Define the tiles and rectangles
inductive Tile | X | Y | Z | W
inductive Rectangle | P | Q | R | S

-- Define the sides of a tile
inductive Side | Top | Right | Bottom | Left

-- Define the function that gives the number on each side of a tile
def tileNumber (t : Tile) (s : Side) : ℕ :=
  match t, s with
  | Tile.X, Side.Top => 5
  | Tile.X, Side.Right => 3
  | Tile.X, Side.Bottom => 6
  | Tile.X, Side.Left => 2
  | Tile.Y, Side.Top => 3
  | Tile.Y, Side.Right => 6
  | Tile.Y, Side.Bottom => 2
  | Tile.Y, Side.Left => 5
  | Tile.Z, Side.Top => 6
  | Tile.Z, Side.Right => 0
  | Tile.Z, Side.Bottom => 1
  | Tile.Z, Side.Left => 5
  | Tile.W, Side.Top => 2
  | Tile.W, Side.Right => 5
  | Tile.W, Side.Bottom => 3
  | Tile.W, Side.Left => 0

-- Define the placement of tiles in rectangles
def placement : Rectangle → Tile := sorry

-- Define the adjacency of rectangles
def adjacent : Rectangle → Rectangle → Prop := sorry

-- Define the condition that adjacent tiles must have matching numbers on common sides
def matchingAdjacent : Prop :=
  ∀ r1 r2 : Rectangle, adjacent r1 r2 →
    (tileNumber (placement r1) Side.Right = tileNumber (placement r2) Side.Left) ∨
    (tileNumber (placement r1) Side.Left = tileNumber (placement r2) Side.Right) ∨
    (tileNumber (placement r1) Side.Top = tileNumber (placement r2) Side.Bottom) ∨
    (tileNumber (placement r1) Side.Bottom = tileNumber (placement r2) Side.Top)

-- Theorem statement
theorem tile_in_rectangle_R :
  matchingAdjacent →
  placement Rectangle.R = Tile.W := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tile_in_rectangle_R_l14_1472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l14_1418

-- Define the fixed points F₁ and F₂
def F₁ : ℝ × ℝ := (5, 0)
def F₂ : ℝ × ℝ := (-5, 0)

-- Define the distance function between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the condition for point M
def satisfies_condition (M : ℝ × ℝ) : Prop :=
  distance M F₁ + distance M F₂ = 10

-- Define what it means for a point to be on a line segment
def on_line_segment (p q r : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = (t * p.1 + (1 - t) * q.1, t * p.2 + (1 - t) * q.2)

-- The main theorem
theorem trajectory_is_line_segment :
  ∀ M : ℝ × ℝ, satisfies_condition M → on_line_segment F₁ F₂ M :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_is_line_segment_l14_1418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_count_l14_1422

/-- A polynomial of degree 500 with real coefficients -/
structure Polynomial500 where
  coeff : Fin 501 → ℝ
  leading_nonzero : coeff 500 ≠ 0

/-- The roots of a polynomial -/
def roots (p : Polynomial500) : Multiset ℂ := sorry

/-- The number of distinct magnitudes among the roots -/
noncomputable def distinct_magnitudes (p : Polynomial500) : ℕ :=
  (roots p).map Complex.abs |>.toFinset.card

/-- The number of real roots of a polynomial -/
def real_root_count (p : Polynomial500) : ℕ := sorry

theorem min_real_roots_count (p : Polynomial500) 
  (h : distinct_magnitudes p = 250) :
  ∃ q : Polynomial500, real_root_count q = 0 ∧ distinct_magnitudes q = 250 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_real_roots_count_l14_1422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l14_1494

/-- Represents a hyperbola with equation x²/4 - y²/m² = 1 where m > 0 -/
structure Hyperbola where
  m : ℝ
  h_pos : m > 0

/-- The slope of the asymptote of the hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := Real.sqrt 2

/-- The eccentricity of the hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt 3

/-- 
  Theorem: If the slope of the asymptote of the hyperbola is √2, 
  then its eccentricity is √3
-/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  asymptote_slope h = Real.sqrt 2 → eccentricity h = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l14_1494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hours_worked_per_day_l14_1487

/-- Given a total of 7.5 hours worked over 3 days, prove that the number of hours worked each day is 2.5 hours. -/
theorem hours_worked_per_day (total_hours : ℝ) (total_days : ℕ) 
  (h1 : total_hours = 7.5) 
  (h2 : total_days = 3) : 
  total_hours / (total_days : ℝ) = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hours_worked_per_day_l14_1487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_intersections_l14_1443

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * Real.sqrt 3 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the line l
def l (k : ℝ) (x : ℝ) : ℝ := k * x

-- Theorem statement
theorem max_distance_between_intersections :
  ∃ (k : ℝ), k ≠ 0 ∧
  (∀ (k' : ℝ), k' ≠ 0 →
    let A := C₁ (Real.arcsin (l k' (Real.sqrt (1 + k'*k')) / Real.sqrt (1 + k'*k')))
    let B := C₂ (Real.arctan k')
    (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (C₁ (2*π/3)).1^2 + (C₂ (2*π/3)).2^2) ∧
  (C₁ (2*π/3)).1^2 + (C₂ (2*π/3)).2^2 = 16 ∧
  k = -Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_between_intersections_l14_1443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_sum_l14_1467

noncomputable def f (x : ℝ) : ℝ := Real.log 9 * (Real.log x / Real.log 3)

theorem f_derivative_sum : 
  (deriv f 2) + (deriv f) 2 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_sum_l14_1467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_constraint_l14_1473

theorem intersection_constraint (m : ℝ) : 
  let M : Set ℂ := {1, 2, Complex.mk (m^2 - 2*m - 5) (m^2 + 5*m + 6)}
  let N : Set ℂ := {3, 10}
  (M ∩ N).Nonempty → m = -2 ∨ m = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_constraint_l14_1473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_l14_1446

/-- The set of all real solutions to the equation ∛(15x - 1) + ∛(13x + 1) = 4∛x -/
noncomputable def solution_set : Set ℝ :=
  {x : ℝ | (15 * x - 1) ^ (1/3 : ℝ) + (13 * x + 1) ^ (1/3 : ℝ) = 4 * x ^ (1/3 : ℝ)}

/-- Theorem stating that the solution set contains exactly 0, 1/14, and -1/12 -/
theorem solution_set_eq : solution_set = {0, 1/14, -1/12} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_eq_l14_1446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_parabola_l14_1439

-- Define the function f(x) = √x
noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

-- Define the tangent line equation
def tangent_line (x y : ℝ) : Prop := x - 2*y + 1 = 0

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := x^2 = 2*y

theorem tangent_and_parabola :
  -- The tangent line at x=1 is x - 2y + 1 = 0
  (∀ x y : ℝ, (x = 1 ∧ y = f 1) → tangent_line x y) ∧
  -- The parabola with focus at the intersection of the tangent line and y-axis is x² = 2y
  (∃ p : ℝ × ℝ, p.1 = 0 ∧ tangent_line p.1 p.2 ∧ 
    ∀ x y : ℝ, (x - p.1)^2 = 4*p.2*(y - p.2) ↔ parabola x y) :=
by
  sorry

#check tangent_and_parabola

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_parabola_l14_1439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_geq_nine_l14_1415

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 else -x^2 + 2*x

theorem f_composition_geq_nine (x : ℝ) :
  f (f x) ≥ 9 ↔ x ≤ -Real.sqrt 3 ∨ x ≥ Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_geq_nine_l14_1415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_70_percent_acute_l14_1435

/-- A set of points in a plane -/
structure PointSet where
  points : Finset (ℝ × ℝ)
  card_eq : points.card = 100

/-- Defines a property that no three points in the set are collinear -/
def NonCollinear (ps : PointSet) : Prop :=
  ∀ p q r : ℝ × ℝ, p ∈ ps.points → q ∈ ps.points → r ∈ ps.points →
    p ≠ q → q ≠ r → p ≠ r →
    ¬ ∃ (m b : ℝ), (p.2 = m * p.1 + b) ∧ (q.2 = m * q.1 + b) ∧ (r.2 = m * r.1 + b)

/-- Counts the number of acute-angled triangles in a point set -/
noncomputable def CountAcuteTriangles (ps : PointSet) : ℕ :=
  sorry

/-- The main theorem -/
theorem at_most_70_percent_acute (ps : PointSet) (h_non_collinear : NonCollinear ps) :
  (CountAcuteTriangles ps : ℚ) / (Nat.choose ps.points.card 3 : ℚ) ≤ 7/10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_70_percent_acute_l14_1435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_rainfall_l14_1471

/-- Rainfall problem -/
theorem monday_rainfall (tuesday_rainfall : ℝ) (difference : ℝ) (monday_rainfall : ℝ)
  (h1 : tuesday_rainfall = 0.2)
  (h2 : difference = 0.7)
  (h3 : tuesday_rainfall + difference = monday_rainfall) :
  monday_rainfall = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monday_rainfall_l14_1471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_area_l14_1468

/-- The area of a circle circumscribed about an equilateral triangle -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  π * (s / Real.sqrt 3)^2 = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_circle_area_l14_1468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l14_1493

theorem square_root_problem (a b c : ℝ) : 
  (Real.sqrt (a - 1) = 2) → 
  (∃ (x y : ℝ), x^2 = b ∧ y^2 = b ∧ x = 2*c - 1 ∧ y = -c + 2) → 
  (Real.sqrt (2*a + b + c) = 3*Real.sqrt 2 ∨ Real.sqrt (2*a + b + c) = -3*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_problem_l14_1493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_duration_l14_1442

/-- The total amount Joan needs to pay back -/
noncomputable def total_amount : ℝ := 914800

/-- Joan's first mortgage payment -/
noncomputable def first_payment : ℝ := 100

/-- The factor by which each payment increases -/
noncomputable def increase_factor : ℝ := 3

/-- The function representing the total amount paid after n months -/
noncomputable def total_paid (n : ℕ) : ℝ :=
  first_payment * (1 - increase_factor ^ n) / (1 - increase_factor)

/-- The theorem stating that 9 is the smallest positive integer satisfying the equation -/
theorem mortgage_duration :
  (∀ k : ℕ, k < 9 → total_paid k < total_amount) ∧
  total_paid 9 ≥ total_amount := by
  sorry

#check mortgage_duration

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mortgage_duration_l14_1442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l14_1480

/-- Triangle ABC with side lengths a, b, c and angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_sum : A + B + C = Real.pi
  cos_law : b^2 + c^2 - 2*b*c*(Real.cos C) = a^2

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 - (Real.sqrt 6 / 2) * t.b * t.c = t.a^2) :
  (Real.cos t.A = Real.sqrt 6 / 4) ∧ 
  (t.B = 2 * t.A → t.b = Real.sqrt 6 → t.a = 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l14_1480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_pyramid_volume_l14_1447

/-- The volume of a pyramid with a square base -/
noncomputable def pyramid_volume (s : ℝ) (h : ℝ) : ℝ := (1/3) * s^2 * h

theorem new_pyramid_volume (s₀ h₀ : ℝ) (h₀_pos : 0 < h₀) (s₀_pos : 0 < s₀) :
  pyramid_volume s₀ h₀ = 60 →
  pyramid_volume (3*s₀) (2*h₀) = 1080 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_pyramid_volume_l14_1447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_difference_l14_1402

/-- An acute triangle with consecutive integer side lengths -/
structure AcuteTriangle where
  n : ℕ
  is_acute : n > 1

/-- The height from the vertex opposite the middle side to that side -/
noncomputable def AcuteTriangle.height (t : AcuteTriangle) : ℝ := sorry

/-- The segments created by the height on the middle side -/
noncomputable def AcuteTriangle.segments (t : AcuteTriangle) : ℝ × ℝ := sorry

/-- Theorem: The difference of the segments is always 4 -/
theorem segment_difference (t : AcuteTriangle) : 
  let (x, y) := t.segments
  |y - x| = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_difference_l14_1402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l14_1476

/-- The repeating decimal 0.3666... is equal to 11/30 -/
theorem repeating_decimal_to_fraction : 
  (∑' n, 3 * (1/10)^(n+1) + 6 * (1/10)^(n+2)) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_to_fraction_l14_1476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_in_cos_2x_l14_1449

theorem quadratic_in_cos_2x (a b c x : ℝ) :
  a * (Real.cos x)^2 + b * Real.cos x + c = 0 →
  a^2 * (Real.cos (2*x))^2 + 2*(a^2 + a*c - b^2) * Real.cos (2*x) + (a^2 + 2*a*c - 2*b^2 + 4*c^2) = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_in_cos_2x_l14_1449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_expansion_l14_1477

/-- Given a natural number n, such that the sum of binomial coefficients
    of (2x^(-2) - x^3)^n is 256, the coefficient of x^4 in its expansion is 1120 -/
theorem coefficient_x4_expansion (n : ℕ) 
  (h : (Finset.sum (Finset.range (n + 1)) (fun k => Nat.choose n k)) = 256) :
  (Finset.sum (Finset.range (n + 1)) 
    (fun k => Nat.choose n k * (2^(n - k) * ((-1 : ℤ)^k).toNat) * 
      (if 5 * k - 16 = 4 then 1 else 0))) = 1120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x4_expansion_l14_1477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solutions_l14_1427

noncomputable def g (x : ℝ) : ℝ :=
  if x < 0 then 4 * x + 8 else 3 * x - 15

theorem g_solutions :
  {x : ℝ | g x = 5} = {-3/4, 20/3} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_solutions_l14_1427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_twelve_l14_1470

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents the dimensions of a rectangular opening (door or window) -/
structure Opening where
  height : ℝ
  width : ℝ

/-- Calculates the area of walls to be whitewashed in a room -/
def areaToWhitewash (room : RoomDimensions) (door : Opening) (window : Opening) (numWindows : ℕ) : ℝ :=
  2 * (room.length + room.width) * room.height - 
  (door.height * door.width + (numWindows : ℝ) * window.height * window.width)

/-- Theorem stating the height of the room given the conditions -/
theorem room_height_is_twelve 
  (room : RoomDimensions)
  (door : Opening)
  (window : Opening)
  (whitewashCost : ℝ)
  (totalCost : ℝ) :
  room.length = 25 →
  room.width = 15 →
  door.height = 6 →
  door.width = 3 →
  window.height = 4 →
  window.width = 3 →
  whitewashCost = 10 →
  totalCost = 9060 →
  areaToWhitewash room door window 3 = totalCost / whitewashCost →
  room.height = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_room_height_is_twelve_l14_1470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_abs_even_and_increasing_l14_1412

-- Define the function f(x) = |sin(x)|
noncomputable def f (x : ℝ) : ℝ := |Real.sin x|

-- State the theorem
theorem sin_abs_even_and_increasing :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_abs_even_and_increasing_l14_1412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_four_divisors_l14_1428

/-- A function that returns the number of divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- A function that checks if a sequence of consecutive natural numbers all have exactly four divisors -/
def all_have_four_divisors (start : ℕ) (length : ℕ) : Prop :=
  ∀ i ∈ Finset.range length, divisor_count (start + i) = 4

/-- The theorem stating that the maximum number of consecutive natural numbers,
    each having exactly four divisors, is 3 -/
theorem max_consecutive_four_divisors :
  (∃ start : ℕ, all_have_four_divisors start 3) ∧
  (∀ start length : ℕ, length > 3 → ¬ all_have_four_divisors start length) := by
  sorry

#check max_consecutive_four_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_consecutive_four_divisors_l14_1428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_five_points_l14_1465

/-- A type representing a point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A predicate that checks if three points are collinear -/
def collinear (p q r : Point3D) : Prop := sorry

/-- A predicate that checks if four points form a triangular pyramid -/
def forms_triangular_pyramid (p q r s : Point3D) : Prop := sorry

/-- A function that counts the number of planes determined by a set of points -/
def count_planes (points : Finset Point3D) : ℕ := sorry

theorem max_planes_from_five_points (points : Finset Point3D) :
  points.card = 5 →
  (∀ p q r, p ∈ points → q ∈ points → r ∈ points → p ≠ q → q ≠ r → r ≠ p → ¬collinear p q r) →
  (∃ pyramids : Finset (Finset Point3D),
    pyramids.card = 4 ∧
    ∀ pyr ∈ pyramids, pyr.card = 4 ∧ ∃ p q r s, p ∈ points ∧ q ∈ points ∧ r ∈ points ∧ s ∈ points ∧ forms_triangular_pyramid p q r s) →
  count_planes points ≤ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_planes_from_five_points_l14_1465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2010_l14_1482

def sequence_property (x : ℕ → ℝ) (a : ℝ) : Prop :=
  (∀ n : ℕ, x (n + 3) = x n) ∧
  (∀ n : ℕ, x (n + 2) = |x (n + 1) - x n|) ∧
  (x 1 = 1) ∧
  (x 2 = a) ∧
  (a ≤ 1) ∧
  (a ≠ 0)

def sum_2010 (x : ℕ → ℝ) : ℝ :=
  Finset.sum (Finset.range 2010) (λ i => x (i + 1))

theorem sequence_sum_2010 (x : ℕ → ℝ) (a : ℝ) :
  sequence_property x a → sum_2010 x = 1340 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_sum_2010_l14_1482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_house_sales_revenue_l14_1406

/-- Calculates the net revenue per person for two house sales --/
theorem house_sales_revenue
  (house_a_market_value : ℝ)
  (house_a_over_market : ℝ)
  (house_b_market_value : ℝ)
  (house_b_below_market : ℝ)
  (house_a_tax_rate : ℝ)
  (house_b_tax_rate : ℝ)
  (house_a_people : ℕ)
  (house_b_people : ℕ)
  (h1 : house_a_market_value = 500000)
  (h2 : house_a_over_market = 0.2)
  (h3 : house_b_market_value = 700000)
  (h4 : house_b_below_market = 0.1)
  (h5 : house_a_tax_rate = 0.1)
  (h6 : house_b_tax_rate = 0.12)
  (h7 : house_a_people = 4)
  (h8 : house_b_people = 5) :
  (house_a_market_value * (1 + house_a_over_market) * (1 - house_a_tax_rate) / house_a_people = 135000) ∧
  (house_b_market_value * (1 - house_b_below_market) * (1 - house_b_tax_rate) / house_b_people = 110880) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_house_sales_revenue_l14_1406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l14_1420

/-- Represents an ellipse in the xy-plane -/
structure Ellipse where
  a : ℝ
  b : ℝ

/-- Defines the equation of an ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- Calculates the eccentricity of an ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (min e.a e.b / max e.a e.b)^2)

/-- Represents a point in the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

theorem ellipse_problem (C₁ C₂ : Ellipse) (O A B : Point) :
  C₁.a = 2 ∧ C₁.b = 1 ∧
  C₂.b = C₁.a ∧
  C₂.eccentricity = C₁.eccentricity ∧
  O.x = 0 ∧ O.y = 0 ∧
  C₁.equation A.x A.y ∧
  C₂.equation B.x B.y ∧
  B.x = 2 * A.x ∧ B.y = 2 * A.y →
  (C₂.a = 4 ∧ C₂.b = 2) ∧
  (A.y = A.x ∨ A.y = -A.x) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_problem_l14_1420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_theta_l14_1425

theorem even_function_theta (θ : ℝ) : 
  (∀ x : ℝ, Real.sin (x + θ) + Real.cos (x + θ) = Real.sin (-x + θ) + Real.cos (-x + θ)) → 
  ∃ k : ℤ, θ = π / 4 + k * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_theta_l14_1425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_buses_equal_stoppage_time_l14_1434

/-- Represents a bus with its speeds with and without stoppages -/
structure Bus where
  speed_without_stoppage : ℚ
  speed_with_stoppage : ℚ

/-- Calculates the stoppage time per hour for a given bus -/
def stoppage_time (bus : Bus) : ℚ :=
  1 - (bus.speed_with_stoppage / bus.speed_without_stoppage)

/-- The three buses with their respective speeds -/
def bus1 : Bus := ⟨50, 40⟩
def bus2 : Bus := ⟨60, 48⟩
def bus3 : Bus := ⟨70, 56⟩

theorem all_buses_equal_stoppage_time :
  stoppage_time bus1 = stoppage_time bus2 ∧
  stoppage_time bus2 = stoppage_time bus3 ∧
  stoppage_time bus1 = 1/5 := by
  sorry

#eval stoppage_time bus1
#eval stoppage_time bus2
#eval stoppage_time bus3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_buses_equal_stoppage_time_l14_1434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_tangent_line_at_2_l14_1430

-- Part 1
noncomputable def f (x : ℝ) : ℝ := Real.exp (-x) * Real.sin (2 * x)

theorem f_derivative : 
  deriv f = λ x ↦ Real.exp (-x) * (2 * Real.cos (2 * x) - Real.sin (2 * x)) := by sorry

-- Part 2
def g (x : ℝ) : ℝ := x^3 - 3*x

theorem tangent_line_at_2 :
  let m := deriv g 2
  let b := g 2 - m * 2
  (λ x ↦ m * x + b) = (λ x ↦ 9 * x - 24) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_derivative_tangent_line_at_2_l14_1430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_over_4_f_period_f_monotonic_increase_l14_1432

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + Real.cos (2*x) - 1

-- Theorem for part (I)
theorem f_pi_over_4 : f (Real.pi/4) = 1 := by sorry

-- Theorem for part (II) - smallest positive period
theorem f_period : ∃ T : ℝ, T > 0 ∧ T = Real.pi ∧ ∀ x : ℝ, f (x + T) = f x := by sorry

-- Theorem for part (II) - interval of monotonic increase
theorem f_monotonic_increase :
  ∀ k : ℤ, ∀ x y : ℝ,
    k * Real.pi - 3 * Real.pi / 8 ≤ x ∧
    x < y ∧
    y ≤ Real.pi / 8 + k * Real.pi →
    f x < f y := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_pi_over_4_f_period_f_monotonic_increase_l14_1432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integral_solution_is_three_l14_1411

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then -2 * x + 1 else x^2 - 2 * x

-- Define the property for the maximum integral solution
def is_max_integral_solution (d : ℤ) : Prop :=
  (f (↑d) = 3) ∧ ∀ k : ℤ, f (↑k) = 3 → k ≤ d

-- Theorem statement
theorem max_integral_solution_is_three :
  is_max_integral_solution 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integral_solution_is_three_l14_1411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l14_1424

/-- The speed of a man rowing in still water, given his upstream and downstream speeds -/
noncomputable def speed_in_still_water (upstream_speed downstream_speed : ℝ) : ℝ :=
  (upstream_speed + downstream_speed) / 2

/-- Theorem: Given a man who can row upstream at 27 kmph and downstream at 35 kmph,
    his speed in still water is 31 kmph -/
theorem man_rowing_speed :
  speed_in_still_water 27 35 = 31 := by
  -- Unfold the definition of speed_in_still_water
  unfold speed_in_still_water
  -- Simplify the arithmetic
  simp [add_div]
  -- Check that 62 / 2 = 31
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rowing_speed_l14_1424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_weight_is_five_fourths_swallow_l14_1490

/-- The weight of a swallow in grams -/
noncomputable def swallow_weight : ℝ := Real.pi  -- Using pi as a placeholder for an arbitrary real number

/-- The weight of a sparrow in grams -/
noncomputable def sparrow_weight : ℝ := (5/4) * swallow_weight

/-- Six sparrows are heavier than seven swallows -/
axiom six_sparrows_heavier : 6 * sparrow_weight > 7 * swallow_weight

/-- If one sparrow and one swallow are exchanged, the weights are equal -/
axiom exchange_equal : 5 * sparrow_weight + swallow_weight = 6 * swallow_weight + sparrow_weight

theorem sparrow_weight_is_five_fourths_swallow :
  sparrow_weight = (5/4) * swallow_weight := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sparrow_weight_is_five_fourths_swallow_l14_1490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l14_1413

/-- A triangular pyramid with mutually perpendicular lateral edges -/
structure TriangularPyramid where
  /-- The area of the first lateral face -/
  area1 : ℝ
  /-- The area of the second lateral face -/
  area2 : ℝ
  /-- The area of the third lateral face -/
  area3 : ℝ
  /-- The lateral edges are mutually perpendicular -/
  perpendicular_edges : True

/-- The volume of a triangular pyramid -/
noncomputable def volume (p : TriangularPyramid) : ℝ :=
  Real.sqrt (p.area1 * p.area2 * p.area3) / 6

/-- Theorem: The volume of the specific triangular pyramid is 2 cm³ -/
theorem specific_pyramid_volume :
  ∃ (p : TriangularPyramid), p.area1 = 1.5 ∧ p.area2 = 2 ∧ p.area3 = 6 ∧ volume p = 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_pyramid_volume_l14_1413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peters_walk_l14_1475

/-- Peter's walk to the grocery store -/
theorem peters_walk (total_distance walking_speed remaining_time : ℝ) : 
  total_distance = 2.5 ∧ 
  walking_speed = 1 / 20 ∧ 
  remaining_time = 30 → 
  total_distance - (walking_speed * remaining_time) = 1 := by
  sorry

#check peters_walk

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peters_walk_l14_1475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l14_1431

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : (2 * t.a - t.c) * Real.cos t.B = t.b * Real.cos t.C)
  (h2 : 0 < t.A) (h3 : t.A < Real.pi)
  (h4 : 0 < t.B) (h5 : t.B < Real.pi)
  (h6 : 0 < t.C) (h7 : t.C < Real.pi) :
  t.B = Real.pi / 3 ∧ 
  (t.a = 2 → t.c = 3 → Real.sin t.C = 3 * Real.sqrt 21 / 14) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l14_1431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_squared_l14_1488

-- Define the circle C
def circle_C (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 + 2)^2 = r^2}

-- Define the line l₁
def line_l1 : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 1}

-- Define the line that contains the center
def center_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | 2 * p.1 + p.2 = 0}

-- Define IsTangentAt
def IsTangentAt (S T : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ S ∩ T ∧ ∀ q ∈ S, q ≠ p → q ∉ T

-- Define IsCenter
def IsCenter (S : Set (ℝ × ℝ)) (c : ℝ × ℝ) : Prop :=
  ∀ p ∈ S, (p.1 - c.1)^2 + (p.2 - c.2)^2 = (2 - c.1)^2 + (-1 - c.2)^2

-- Theorem statement
theorem circle_radius_squared (r : ℝ) :
  (2, -1) ∈ circle_C r ∧
  (∃ p, p ∈ circle_C r ∧ p ∈ line_l1 ∧ IsTangentAt (circle_C r) line_l1 p) ∧
  (∃ c, c ∈ center_line ∧ IsCenter (circle_C r) c) →
  r^2 = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_squared_l14_1488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_ones_twos_threes_l14_1445

theorem four_digit_numbers_with_ones_twos_threes :
  let ones := 3
  let twos := 2
  let threes := 5
  let digit_count := 4
  let number_of_combinations := 71
  ↑number_of_combinations = 71
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_digit_numbers_with_ones_twos_threes_l14_1445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_departure_time_l14_1462

/-- Represents the time in hours since midnight -/
def Time := ℝ

/-- Represents the speed in km/hr -/
def Speed := ℝ

/-- Represents the distance in km -/
def Distance := ℝ

theorem train_departure_time 
  (speed_A : Speed) 
  (speed_B : Speed)
  (start_time_B : Time)
  (meet_time : Time)
  (total_distance : Distance)
  (h1 : speed_A = (60 : ℝ))
  (h2 : speed_B = (75 : ℝ))
  (h3 : start_time_B = (9 : ℝ))
  (h4 : meet_time = (11 : ℝ))
  (h5 : total_distance = (330 : ℝ)) :
  ∃ (start_time_A : Time), start_time_A = (8 : ℝ) :=
by
  sorry

#eval "Theorem defined successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_departure_time_l14_1462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l14_1489

noncomputable section

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 8*x + 20

-- Define the line
def line (x : ℝ) : ℝ := x - 7

-- Define the distance function between a point on the parabola and the line
noncomputable def distance (a : ℝ) : ℝ := |parabola a - line a| / Real.sqrt 2

-- Theorem statement
theorem min_distance_parabola_to_line :
  ∃ (min_dist : ℝ), min_dist = 3 * Real.sqrt 2 / 4 ∧
  ∀ (a : ℝ), distance a ≥ min_dist := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_parabola_to_line_l14_1489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_sum_l14_1438

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (t.a - t.b) * Real.sin t.B = t.a * Real.sin t.A - t.c * Real.sin t.C ∧
  t.a^2 + t.b^2 - 6*(t.a + t.b) + 18 = 0

-- Define the dot product sum
noncomputable def dot_product_sum (t : Triangle) : Real :=
  t.a * t.b * Real.cos t.C + t.b * t.c * Real.cos t.A + t.c * t.a * Real.cos t.B

-- State the theorem
theorem triangle_dot_product_sum (t : Triangle) :
  satisfies_conditions t → dot_product_sum t = -27/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_dot_product_sum_l14_1438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_sum_polynomials_l14_1483

-- Define the polynomials f and g
def f (z : ℂ) (a₃ a₂ a₁ a₀ : ℂ) : ℂ := a₃ * z^3 + a₂ * z^2 + a₁ * z + a₀
def g (z : ℂ) (b₁ b₀ : ℂ) : ℂ := b₁ * z + b₀

-- Define the degree of a polynomial
noncomputable def degree (p : ℂ → ℂ) : ℕ := sorry

-- Theorem statement
theorem degree_of_sum_polynomials
  (a₃ a₂ a₁ a₀ b₁ b₀ : ℂ)
  (h₁ : a₃ ≠ 0)
  (h₂ : degree (f · a₃ a₂ a₁ a₀) = 3)
  (h₃ : degree (g · b₁ b₀) = 1) :
  degree (λ z ↦ f z a₃ a₂ a₁ a₀ + g z b₁ b₀) = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_of_sum_polynomials_l14_1483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_18n4_l14_1437

theorem divisors_of_18n4 (n : ℕ) (h_odd : Odd n) (h_divisors : (Nat.divisors n).card = 13) :
  (Nat.divisors (18 * n^4)).card = 294 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_of_18n4_l14_1437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_c_range_l14_1416

theorem cubic_function_c_range (a b c : ℝ) :
  (∀ x : ℝ, 0 ≤ x^3 + a*x^2 + b*x + c ∧ x^3 + a*x^2 + b*x + c ≤ 3) →
  ((-1)^3 + a*(-1)^2 + b*(-1) + c = (-2)^3 + a*(-2)^2 + b*(-2) + c) →
  ((-2)^3 + a*(-2)^2 + b*(-2) + c = (-3)^3 + a*(-3)^2 + b*(-3) + c) →
  (6 ≤ c ∧ c ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_c_range_l14_1416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_integral_l14_1498

theorem binomial_integral (a : ℝ) :
  (∃ k : ℝ, k * a^3 = 280 ∧ k ≠ 0) →
  ∫ x in a..2*Real.exp 1, 1/x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_integral_l14_1498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l14_1456

-- Define the ellipse E
noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the focus point
noncomputable def focus : ℝ × ℝ := (Real.sqrt 3, 0)

-- Define the point P on the ellipse
noncomputable def point_P : ℝ × ℝ := (-Real.sqrt 3, 1/2)

-- Define the point M through which line l passes
noncomputable def point_M : ℝ × ℝ := (0, Real.sqrt 2)

-- Theorem statement
theorem max_chord_length :
  ∀ A B : ℝ × ℝ,
  ellipse A.1 A.2 →
  ellipse B.1 B.2 →
  ∃ k : ℝ, (A.2 - point_M.2 = k * (A.1 - point_M.1) ∧
            B.2 - point_M.2 = k * (B.1 - point_M.1)) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ 5 * Real.sqrt 6 / 6 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_chord_length_l14_1456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_AO_range_l14_1460

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the circumcenter O
variable (O : ℝ × ℝ)

-- Define side lengths
variable (a b c : ℝ)

-- Define BC and AO as functions
def BC (B C : ℝ × ℝ) : ℝ × ℝ := (C.1 - B.1, C.2 - B.2)
def AO (A O : ℝ × ℝ) : ℝ × ℝ := (O.1 - A.1, O.2 - A.2)

-- Define the dot product of BC and AO
def BC_dot_AO (A B C O : ℝ × ℝ) : ℝ :=
  let bc := BC B C
  let ao := AO A O
  bc.1 * ao.1 + bc.2 * ao.2

-- State the theorem
theorem BC_AO_range (h1 : b^2 - 2*b + c^2 = 0) :
  -1/4 ≤ BC_dot_AO A B C O ∧ BC_dot_AO A B C O < 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_BC_AO_range_l14_1460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_animals_count_l14_1455

theorem zoo_animals_count (zebras camels monkeys parrots adult_giraffes baby_giraffes : ℕ) : 
  zebras = 12 →
  camels = zebras / 2 →
  monkeys = 4 * camels →
  parrots = monkeys - 5 + (monkeys - 5) / 2 →
  adult_giraffes = 3 * parrots + 1 →
  baby_giraffes = adult_giraffes / 4 →
  monkeys < adult_giraffes + baby_giraffes :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zoo_animals_count_l14_1455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l14_1448

/-- The function f(x) = 3sin(ωx + φ) -/
noncomputable def f (ω φ x : ℝ) : ℝ := 3 * Real.sin (ω * x + φ)

/-- The theorem stating the maximum value of ω given the conditions -/
theorem max_omega_value (ω φ : ℝ) : 
  ω > 0 →
  0 < φ ∧ φ < Real.pi →
  f ω φ (-Real.pi/3) = 0 →
  (∀ x : ℝ, f ω φ x ≤ |f ω φ (Real.pi/3)|) →
  (∃! x₁ : ℝ, Real.pi/15 < x₁ ∧ x₁ < Real.pi/5 ∧ f ω φ x₁ = 3) →
  (∀ ω' : ℝ, ω' > ω → ¬(∃! x₁ : ℝ, Real.pi/15 < x₁ ∧ x₁ < Real.pi/5 ∧ f ω' φ x₁ = 3)) →
  ω = 105/4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_omega_value_l14_1448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prom_night_cost_l14_1492

/-- Calculate the total cost of James and Susan's prom night -/
theorem prom_night_cost : 
  (2 : ℕ) * 100 + 120 + (120 : ℚ) * (30 / 100) + 6 * 80 = 836 := by
  -- Convert the fractional part to a natural number
  have h1 : (120 : ℚ) * (30 / 100) = 36 := by norm_num
  
  -- Simplify the left-hand side
  calc
    (2 : ℕ) * 100 + 120 + (120 : ℚ) * (30 / 100) + 6 * 80
    = 200 + 120 + 36 + 480 := by 
      rw [h1]
      norm_num
    _ = 836 := by norm_num

  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prom_night_cost_l14_1492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertices_distance_l14_1404

/-- The equation of the parabolas -/
noncomputable def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 3) = 5

/-- The vertices of the parabolas -/
def vertices : Set (ℝ × ℝ) :=
  {(0, 4), (0, -1)}

/-- The distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_vertices_distance :
  ∀ p q, p ∈ vertices → q ∈ vertices → p ≠ q → distance p q = 5 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertices_distance_l14_1404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l14_1496

noncomputable def distance_from_origin (x y : ℝ) : ℝ := Real.sqrt (x^2 + y^2)

def point_set : List (ℝ × ℝ) := [(2, 5), (3, 1), (4, -3), (7, 0), (0, -6)]

theorem farthest_point :
  ∀ (p : ℝ × ℝ), p ∈ point_set → distance_from_origin p.1 p.2 ≤ distance_from_origin 7 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farthest_point_l14_1496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l14_1408

/-- A power function that passes through (4, 2) and has f(m) = 3 -/
noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem power_function_m_value (α : ℝ) (m : ℝ) :
  power_function α 4 = 2 →
  power_function α m = 3 →
  m = 9 := by
  intro h1 h2
  -- The proof steps would go here
  sorry

#check power_function_m_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_m_value_l14_1408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_with_restriction_arrangements_not_adjacent_l14_1495

-- Define the number of students
def n : ℕ := 6

-- Define the number of specific students that cannot be adjacent
def k : ℕ := 3

-- Theorem 1: Arrangements with one student not in first or last position
theorem arrangements_with_restriction (n : ℕ) : 
  (n - 2) * Nat.factorial (n - 1) = 480 :=
sorry

-- Theorem 2: Arrangements with three students not adjacent
theorem arrangements_not_adjacent (n k : ℕ) : 
  Nat.factorial (n - k) * Nat.choose (n - k + 1) k = 144 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arrangements_with_restriction_arrangements_not_adjacent_l14_1495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_18800_l14_1486

/-- Represents the number of type D plates produced from one type A plate -/
def l : ℕ := 1  -- Assuming l = 1 for this problem

/-- Total number of type A and B plates -/
def total_plates : ℕ := 50

/-- Maximum number of type C plates allowed -/
def max_type_c : ℕ := 86

/-- Maximum number of type D plates allowed -/
def max_type_d : ℕ := 90

/-- Profit per type C plate in yuan -/
def profit_type_c : ℕ := 100

/-- Profit per type D plate in yuan -/
def profit_type_d : ℕ := 120

/-- Conversion function for type A plates to type C plates -/
def type_a_to_c (a : ℕ) : ℕ := 2 * a

/-- Conversion function for type A plates to type D plates -/
def type_a_to_d (a : ℕ) : ℕ := l * a

/-- Conversion function for type B plates to type C plates -/
def type_b_to_c (b : ℕ) : ℕ := b

/-- Conversion function for type B plates to type D plates -/
def type_b_to_d (b : ℕ) : ℕ := 3 * b

/-- Calculate total profit given the number of type A plates purchased -/
def total_profit (a : ℕ) : ℕ :=
  let b := total_plates - a
  let c := min (type_a_to_c a + type_b_to_c b) max_type_c
  let d := min (type_a_to_d a + type_b_to_d b) max_type_d
  c * profit_type_c + d * profit_type_d

/-- Theorem stating that the maximum profit is 18800 yuan -/
theorem max_profit_is_18800 :
  ∃ a : ℕ, a ≤ total_plates ∧ total_profit a = 18800 ∧
  ∀ x : ℕ, x ≤ total_plates → total_profit x ≤ 18800 :=
by
  -- The proof goes here
  sorry

#eval total_profit 30  -- This should output 18800

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_is_18800_l14_1486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculation_cubic_equation_solution_l14_1458

-- Part 1
theorem sqrt_calculation : Real.sqrt 16 - (125 : ℝ)^(1/3) + |Real.sqrt 3 - 2| = 1 - Real.sqrt 3 := by
  sorry

-- Part 2
theorem cubic_equation_solution : ∃ x : ℝ, (x - 1)^3 + 27 = 0 ∧ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_calculation_cubic_equation_solution_l14_1458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_liquid_depth_l14_1479

/-- Represents a horizontal cylindrical container with liquid --/
structure LiquidContainer where
  length : ℝ
  diameter : ℝ
  liquidSurfaceArea : ℝ

/-- Calculates the depth of liquid in the container --/
noncomputable def liquidDepth (container : LiquidContainer) : ℝ :=
  4 - 2 * Real.sqrt 3

/-- Theorem stating the correct depth of liquid for the given container --/
theorem correct_liquid_depth (container : LiquidContainer) 
  (h1 : container.length = 12)
  (h2 : container.diameter = 8)
  (h3 : container.liquidSurfaceArea = 48) :
  liquidDepth container = 4 - 2 * Real.sqrt 3 := by
  sorry

/-- Evaluates the liquid depth for the given container --/
def evaluate_liquid_depth : ℚ :=
  4 - 2 * (3 : ℚ).sqrt

#eval evaluate_liquid_depth

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_liquid_depth_l14_1479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_parameters_l14_1410

/-- A cubic function with parameters a, b, and c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- The derivative of f with respect to x -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem cubic_function_parameters :
  ∀ a b c : ℝ,
  (f' a b (-2) = 0) ∧ 
  (f' a b 1 = -3) ∧ 
  (f a b c 1 = 0) →
  a = 1 ∧ b = -8 ∧ c = 6 := by
  intros a b c h
  sorry

#check cubic_function_parameters

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_parameters_l14_1410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l14_1497

/-- The area of a triangle given its vertices' coordinates -/
noncomputable def triangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

/-- Theorem: The area of triangle PQR with given coordinates is 21 -/
theorem area_of_triangle_PQR : 
  triangleArea (-3) 2 1 7 3 (-1) = 21 := by
  -- Unfold the definition of triangleArea
  unfold triangleArea
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_PQR_l14_1497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l14_1453

/-- Given a point P(-4, 3) on the terminal side of angle α, prove that cos α = -4/5 -/
theorem cos_alpha_for_point (α : ℝ) (P : ℝ × ℝ) : 
  P = (-4, 3) → Real.cos α = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_for_point_l14_1453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_10_40_l14_1405

/-- The angle moved by the minute hand in one minute -/
noncomputable def minute_hand_angle_per_minute : ℝ := 360 / 60

/-- The angle moved by the hour hand in one hour -/
noncomputable def hour_hand_angle_per_hour : ℝ := 360 / 12

/-- The angle moved by the hour hand in one minute -/
noncomputable def hour_hand_angle_per_minute : ℝ := hour_hand_angle_per_hour / 60

/-- The position of the minute hand at 40 minutes past the hour -/
noncomputable def minute_hand_position (minutes : ℕ) : ℝ :=
  minutes * minute_hand_angle_per_minute

/-- The position of the hour hand at 10:40 -/
noncomputable def hour_hand_position (hours minutes : ℕ) : ℝ :=
  hours * hour_hand_angle_per_hour + minutes * hour_hand_angle_per_minute

/-- The smaller angle between the clock hands -/
noncomputable def smaller_angle (angle : ℝ) : ℝ :=
  min angle (360 - angle)

theorem clock_angle_at_10_40 :
  smaller_angle (abs (hour_hand_position 10 40 - minute_hand_position 40)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_10_40_l14_1405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_fraction_coeff_integer_valued_l14_1485

theorem polynomial_with_fraction_coeff_integer_valued : 
  ∃ (P : Polynomial ℚ) (n : ℕ), 
    (∃ (i : ℕ) (c : ℚ), P.coeff i = c ∧ c = 1 / 2016) ∧ 
    (∀ (i : ℕ), i ≠ n → ∃ (q : ℚ), P.coeff i = q) ∧
    (∀ (x : ℤ), ∃ (y : ℤ), (P.eval x : ℚ) = y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_with_fraction_coeff_integer_valued_l14_1485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_record_storage_cost_l14_1451

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℚ :=
  d.length * d.width * d.height

/-- Represents the record storage problem -/
structure RecordStorageProblem where
  boxDim : BoxDimensions
  totalSpace : ℚ
  costPerBox : ℚ

/-- Calculates the total monthly cost for record storage -/
def totalMonthlyCost (p : RecordStorageProblem) : ℚ :=
  (p.totalSpace / boxVolume p.boxDim) * p.costPerBox

/-- Theorem: The total monthly cost for record storage is $240 -/
theorem record_storage_cost (p : RecordStorageProblem) 
  (h1 : p.boxDim = ⟨15, 12, 10⟩) 
  (h2 : p.totalSpace = 1080000)
  (h3 : p.costPerBox = 2/5) : 
  totalMonthlyCost p = 240 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_record_storage_cost_l14_1451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cranberries_left_l14_1484

/-- The number of cranberries left in a bog after harvesting and elk consumption --/
theorem cranberries_left (total : ℕ) (percent_harvested : ℚ) (eaten_by_elk : ℕ) : 
  total = 60000 → 
  percent_harvested = 40 / 100 → 
  eaten_by_elk = 20000 → 
  total - (percent_harvested * ↑total).floor - eaten_by_elk = 16000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cranberries_left_l14_1484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pairs_mutually_generative_l14_1409

-- Define the concept of mutually generative equation pairs
def mutually_generative (f g : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), ∀ x, f (a * x + b) = g (c * x + d) ∨ f (a * x + b) = g (c * (-x) + d)

-- Define the equations
noncomputable def f1 (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def g1 (x : ℝ) : ℝ := Real.sqrt 2 * Real.sin x + 1

def f2 (x y : ℝ) : Prop := y^2 - x^2 = 2
def g2 (x y : ℝ) : Prop := x^2 - y^2 = 2

def f3 (x y : ℝ) : Prop := y^2 = 4*x
def g3 (x y : ℝ) : Prop := x^2 = 4*y

noncomputable def f4 (x : ℝ) : ℝ := Real.log (x - 1)
noncomputable def g4 (x : ℝ) : ℝ := Real.exp x + 1

-- Theorem statement
theorem all_pairs_mutually_generative :
  (mutually_generative f1 g1) ∧
  (∀ x y, f2 x y ↔ g2 y x) ∧
  (∀ x y, f3 x y ↔ g3 y x) ∧
  (mutually_generative f4 g4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_pairs_mutually_generative_l14_1409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_l14_1450

theorem triangle_sine_sum (A B C : ℝ) (a b c : ℝ) :
  A = 60 * Real.pi / 180 → 
  a = Real.sqrt 3 → 
  (Real.sin A + Real.sin B + Real.sin C) / (a + b + c) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sine_sum_l14_1450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cindys_running_speed_l14_1414

/-- Cindy's running speed in miles per hour -/
noncomputable def R : ℝ := sorry

/-- Time taken for the entire journey in hours -/
noncomputable def total_time : ℝ := 2/3

/-- Distance run in miles -/
noncomputable def run_distance : ℝ := 1/2

/-- Distance walked in miles -/
noncomputable def walk_distance : ℝ := 1/2

/-- Walking speed in miles per hour -/
noncomputable def walk_speed : ℝ := 1

theorem cindys_running_speed :
  (run_distance / R) + (walk_distance / walk_speed) = total_time → R = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cindys_running_speed_l14_1414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_gather_all_herrings_l14_1454

/-- Represents the color of a sector -/
inductive Color
| White
| Black
deriving Inhabited, Repr

/-- Represents a sector in the circle -/
structure Sector where
  color : Color
  herringCount : Nat
deriving Inhabited, Repr

/-- Represents the state of the circle -/
structure CircleState where
  sectors : List Sector
deriving Inhabited, Repr

/-- Defines a valid initial state of the circle -/
def initialState : CircleState :=
  { sectors := List.replicate 6 { color := Color.White, herringCount := 1 } }

/-- Defines a valid move operation -/
def validMove (state : CircleState) : CircleState :=
  sorry

/-- Theorem stating that it's impossible to gather all herrings in one sector -/
theorem impossible_to_gather_all_herrings (state : CircleState) :
  ¬∃ (finalState : CircleState), 
    (∃ (moves : Nat), Nat.iterate validMove moves state = finalState) ∧ 
    (∃ (i : Nat), i < 6 ∧ (finalState.sectors.get! i).herringCount = 6) :=
by
  sorry

#eval initialState

end NUMINAMATH_CALUDE_ERRORFEEDBACK_impossible_to_gather_all_herrings_l14_1454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l14_1433

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 + (3 : ℝ)^(-x)

-- State the theorem
theorem inverse_function_value :
  ∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ Function.RightInverse g f ∧ g 10 = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_value_l14_1433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_two_digit_multiples_of_eight_l14_1491

theorem arithmetic_mean_two_digit_multiples_of_eight : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, 10 ≤ n ∧ n < 100 ∧ n % 8 = 0) ∧ 
  (S.sum id / S.card : ℚ) = 56 := by
  -- Define the set of positive two-digit multiples of 8
  let S : Finset ℕ := Finset.filter (fun n => 10 ≤ n ∧ n < 100 ∧ n % 8 = 0) (Finset.range 100)
  
  -- Use S as the witness for the existential quantifier
  use S

  -- Prove the two parts of the conjunction
  apply And.intro

  -- Prove that all elements in S satisfy the condition
  · intro n hn
    simp [S, Finset.mem_filter] at hn
    exact hn.2

  -- Prove that the arithmetic mean of S is 56
  · -- This part requires actual computation and proof, which is beyond the scope of this example
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_two_digit_multiples_of_eight_l14_1491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_model_tower_height_l14_1463

/-- Represents a cone-shaped tower -/
structure ConeTower where
  height : ℝ
  volume : ℝ

/-- Calculates the scale factor for linear dimensions based on volume ratio -/
noncomputable def scaleFactor (original : ConeTower) (model : ConeTower) : ℝ :=
  (original.volume / model.volume) ^ (1/3)

/-- Theorem: Given the original tower's dimensions and the model's volume, 
    prove that the model's height is 0.6 meters -/
theorem model_tower_height 
  (original : ConeTower) 
  (model_volume : ℝ) 
  (h1 : original.height = 60) 
  (h2 : original.volume = 200000) 
  (h3 : model_volume = 0.2) : 
  original.height / scaleFactor { height := original.height, volume := original.volume } 
                   { height := 0, volume := model_volume } = 0.6 := by
  sorry

#eval "Proof completed"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_model_tower_height_l14_1463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l14_1426

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 1 / (x^2 - 5*x + 6)

-- Define the domain of f
def domain_f : Set ℝ := {x : ℝ | x ≠ 2 ∧ x ≠ 3}

-- Theorem stating that the domain of f is (-∞, 2) ∪ (2, 3) ∪ (3, ∞)
theorem domain_of_f : domain_f = Set.Iio 2 ∪ Set.Ioo 2 3 ∪ Set.Ioi 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l14_1426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_s_value_l14_1423

/-- The largest possible value of s for regular polygons P₁ and P₂ with r and s sides respectively,
    where the interior angle of P₁ is 61/60 times that of P₂ -/
theorem largest_s_value : ℕ := by
  -- Define the functions and variables
  let r : ℕ → ℕ := fun s => (120 * s) / (122 - s)
  let interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

  -- State the conditions
  have h1 : ∀ s, r s ≥ s := sorry
  have h2 : ∀ s, s ≥ 3 := sorry
  have h3 : ∀ s, interior_angle (r s) = 61 / 60 * interior_angle s := sorry

  -- State that 121 is the largest possible value
  have h4 : ∀ t > 121, ¬(∃ u, u ≥ t ∧ t ≥ 3 ∧ interior_angle u = 61 / 60 * interior_angle t) := sorry

  -- Prove that 121 satisfies all conditions
  have h5 : r 121 ≥ 121 := sorry
  have h6 : 121 ≥ 3 := sorry
  have h7 : interior_angle (r 121) = 61 / 60 * interior_angle 121 := sorry

  -- Conclude that 121 is the answer
  exact 121


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_s_value_l14_1423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_weight_in_jugs_l14_1469

/-- The weight of sand in two partially filled jugs -/
theorem sand_weight_in_jugs 
  (jug_capacity : ℝ) 
  (num_jugs : ℕ) 
  (fill_percentage : ℝ) 
  (sand_density : ℝ) : 
  jug_capacity = 2 → 
  num_jugs = 2 → 
  fill_percentage = 0.7 → 
  sand_density = 5 → 
  jug_capacity * (num_jugs : ℝ) * fill_percentage * sand_density = 14 := by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

#check sand_weight_in_jugs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sand_weight_in_jugs_l14_1469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_area_ratio_when_diameter_tripled_l14_1464

theorem circular_area_ratio_when_diameter_tripled :
  ∀ (r : ℝ), r > 0 →
  (π * r^2) / (π * (3*r)^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circular_area_ratio_when_diameter_tripled_l14_1464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l14_1419

noncomputable def data : List ℝ := [2, 2, 3, 3, 5]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

theorem variance_of_data : variance data = 1.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_variance_of_data_l14_1419
