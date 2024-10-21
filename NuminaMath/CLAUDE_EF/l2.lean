import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_real_root_of_equation_l2_214

theorem smallest_real_root_of_equation (x : ℝ) :
  (Real.sqrt (x * (x + 1) * (x + 2) * (x + 3) + 1) = 71) →
  (∀ y : ℝ, Real.sqrt (y * (y + 1) * (y + 2) * (y + 3) + 1) = 71 → y ≥ x) →
  x = -10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_real_root_of_equation_l2_214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_non_defective_l2_236

/-- Represents the grade of a product -/
inductive Grade
  | A
  | B
  | C

/-- Determines if a grade is defective -/
def is_defective (g : Grade) : Prop :=
  g = Grade.B ∨ g = Grade.C

/-- The probability of producing a Grade B product -/
def prob_B : ℝ := 0.03

/-- The probability of producing a Grade C product -/
def prob_C : ℝ := 0.01

/-- The probability of obtaining a non-defective product in a random inspection -/
theorem prob_non_defective : 1 - (prob_B + prob_C) = 0.96 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_non_defective_l2_236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_interesting_with_triangle_l2_254

/-- Represents a meeting of people and their handshakes. -/
structure Meeting (m : ℕ) where
  people : Fin (3 * m)
  handshake : Fin (3 * m) → Fin (3 * m) → Bool

/-- Defines an n-interesting meeting. -/
def is_n_interesting (n : ℕ) (meeting : Meeting m) : Prop :=
  ∃ (subset : Fin n → Fin (3 * m)),
    ∀ (i : Fin n), (Finset.sum Finset.univ (λ j ↦ if meeting.handshake (subset i) j then 1 else 0) : ℕ) = i.val + 1

/-- Defines a triangle in the meeting (3 people shaking hands with each other). -/
def has_triangle (meeting : Meeting m) : Prop :=
  ∃ (a b c : Fin (3 * m)), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    meeting.handshake a b ∧ meeting.handshake b c ∧ meeting.handshake a c

/-- The main theorem to be proved. -/
theorem min_n_interesting_with_triangle (m : ℕ) (h : m ≥ 2) :
  (∀ n : ℕ, n ≤ 3 * m - 1 → 
    (∀ meeting : Meeting m, is_n_interesting n meeting → has_triangle meeting)) ↔ 
  (∀ n : ℕ, n ≥ 2 * m + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_interesting_with_triangle_l2_254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_function_exp_range_f_function_polynomial_f_function_quartic_range_l2_240

-- Define F-function
def is_f_function (φ : ℝ → ℝ) (m n : ℝ) : Prop :=
  m < n ∧ ∃ r, m < r ∧ r < n ∧
    (∀ x y, m ≤ x ∧ x ≤ y ∧ y ≤ r → φ x ≤ φ y) ∧
    (∀ x y, r ≤ x ∧ x ≤ y ∧ y ≤ n → φ y ≤ φ x)

-- Question 1
theorem f_function_exp_range (a : ℝ) :
  is_f_function (fun x => (x + a) / Real.exp x) 1 2 → a ∈ Set.Ioo (-1) 0 :=
sorry

-- Question 2
theorem f_function_polynomial (p : ℝ) (hp : p > 0) :
  is_f_function (fun x => p * x - (x^2 / 2 + x^3 / 3 + x^4 / 4 + p * x^5 / 5)) 0 p :=
sorry

-- Question 3
theorem f_function_quartic_range (t m n : ℝ) :
  is_f_function (fun x => (x^2 - x) * (x^2 - x + t)) m n → t < 1/2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_function_exp_range_f_function_polynomial_f_function_quartic_range_l2_240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_breadth_is_two_l2_260

/-- The breadth of a boat given its length, sinking depth, and the mass of a man --/
noncomputable def boat_breadth (boat_length : ℝ) (sinking_depth : ℝ) (man_mass : ℝ) : ℝ :=
  let water_density : ℝ := 1000
  let gravity : ℝ := 9.81
  let displaced_volume : ℝ := man_mass * gravity / (water_density * gravity)
  displaced_volume / (boat_length * sinking_depth)

/-- Theorem stating that the breadth of the boat is 2 meters --/
theorem boat_breadth_is_two :
  boat_breadth 7 0.01 140 = 2 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_breadth_is_two_l2_260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_wrt_B_l2_291

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | Real.exp ((x + 1) * Real.log 2) > 1}

-- Define the complement of A with respect to B
def complement_B (A B : Set ℝ) : Set ℝ := B \ A

-- Theorem statement
theorem complement_of_A_wrt_B :
  complement_B A B = {x : ℝ | x ≥ 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_wrt_B_l2_291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l2_227

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then x else -1

def S : Set ℝ := {x | f (x + 2) ≤ 3}

theorem solution_set : S = Set.Iic 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_l2_227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_right_triangles_2016_eq_12_l2_243

/-- A right triangle with one leg equal to √2016 -/
structure RightTriangle2016 where
  b : ℕ
  c : ℕ
  h1 : c^2 = 2016 + b^2

/-- The count of distinct right triangles with one leg equal to √2016 -/
def count_right_triangles_2016 : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 < p.2 ∧ p.1 * p.2 = 2016 ∧ 
    (p.2 + p.1) % 2 = 0 ∧ (p.2 - p.1) % 2 = 0
  ) (Finset.range 1009 ×ˢ Finset.range 1009)).card

/-- Theorem stating that there are exactly 12 distinct right triangles with one leg equal to √2016 -/
theorem count_right_triangles_2016_eq_12 : 
  count_right_triangles_2016 = 12 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_right_triangles_2016_eq_12_l2_243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l2_203

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * Real.sin (x + Real.pi/6) - 1/2

-- Define the interval
def interval : Set ℝ := Set.Icc (-7*Real.pi/12) (-Real.pi/4)

-- Theorem statement
theorem f_properties :
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧  -- Periodic
  (∀ k : ℤ, ∀ x, k*Real.pi + Real.pi/6 ≤ x → x ≤ k*Real.pi + 2*Real.pi/3 → 
    ∀ y, x < y → f y < f x) ∧  -- Decreasing intervals
  (∀ x ∈ interval, -1 ≤ f x ∧ f x ≤ 0) ∧  -- Bounds on the interval
  (∃ x ∈ interval, f x = -1) ∧  -- Minimum value
  (∃ x ∈ interval, f x = 0)  -- Maximum value
  := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l2_203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_max_value_l2_210

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then x
  else if x ≥ 1 then 1 / x
  else 0  -- This case is not specified in the original problem, but we need to handle it

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * f x - |x - 1|

theorem inequality_and_max_value :
  (∀ b : ℝ, (∀ x : ℝ, x > 0 → g 0 x ≤ |x - 2| + b) ↔ b ≥ -1) ∧
  (∃ M : ℝ, M = 1 ∧ ∀ x : ℝ, x > 0 → g 1 x ≤ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_and_max_value_l2_210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_arc_sum_l2_228

/-- The sum of the length of segment AB and the shorter arc AB on a circle --/
theorem circle_segment_arc_sum (r m b : ℝ) : 
  r = 4 → m = 2 - Real.sqrt 3 → b = 4 →
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}
  let line := {p : ℝ × ℝ | p.2 = -m * p.1 + b}
  let intersections := circle ∩ line
  let A := (0, 4)
  let B := (2, 2 * Real.sqrt 3)
  let segment_length := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let angle := Real.arccos ((2 * r^2 - segment_length^2) / (2 * r^2))
  let arc_length := angle * r
  segment_length + arc_length = 4 * Real.sqrt (2 - Real.sqrt 3) + (2 * Real.pi / 3) := by
  sorry

#eval "Theorem statement compiled successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_arc_sum_l2_228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matches_needed_equals_players_minus_one_l2_257

/-- Represents a single-elimination tennis tournament. -/
structure TennisTournament where
  n : ℕ  -- Number of players
  n_pos : 0 < n  -- Ensure there's at least one player

/-- Function to calculate the number of matches needed in a tournament. -/
def number_of_matches_needed (t : TennisTournament) : ℕ :=
  t.n - 1

/-- 
The number of matches needed to determine the winner in a single-elimination 
tennis tournament is equal to the number of players minus one.
-/
theorem matches_needed_equals_players_minus_one (t : TennisTournament) :
  number_of_matches_needed t = t.n - 1 :=
by
  -- Unfold the definition of number_of_matches_needed
  unfold number_of_matches_needed
  -- The equality now holds by definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matches_needed_equals_players_minus_one_l2_257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_rate_calculation_l2_279

/-- Calculates the rate of return on a stock investment -/
noncomputable def stock_rate_of_return (investment : ℝ) (stock_percentage : ℝ) (annual_income : ℝ) : ℝ :=
  (annual_income / (investment * stock_percentage)) * 100

/-- Theorem: Given the investment conditions, the stock rate of return is approximately 29.41% -/
theorem stock_rate_calculation (investment : ℝ) (stock_percentage : ℝ) (annual_income : ℝ)
  (h1 : investment = 6800)
  (h2 : stock_percentage = 0.4)
  (h3 : annual_income = 2000) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |stock_rate_of_return investment stock_percentage annual_income - 29.41| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_rate_calculation_l2_279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l2_218

-- Define structures for Hyperbola and Ellipse
structure Hyperbola where
  equation : (ℝ → ℝ → Prop)

structure Ellipse where
  equation : (ℝ → ℝ → Prop)
  foci : Set (ℝ × ℝ)
  eccentricity : ℝ

theorem ellipse_equation (h : Hyperbola) (e : Ellipse) :
  h.equation = (fun x y => x^2 / 3 - y^2 / 2 = 1) →
  e.foci = {(-Real.sqrt 5, 0), (Real.sqrt 5, 0)} →
  e.eccentricity = Real.sqrt 5 / 5 →
  e.equation = (fun x y => x^2 / 25 + y^2 / 20 = 1) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l2_218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_theorem_l2_242

/-- Represents a square pyramid --/
structure SquarePyramid where
  baseEdge : ℝ
  altitude : ℝ

/-- Volume of a square pyramid --/
noncomputable def pyramidVolume (p : SquarePyramid) : ℝ :=
  (1 / 3) * p.baseEdge^2 * p.altitude

/-- Represents a frustum formed by cutting a square pyramid --/
structure Frustum where
  originalPyramid : SquarePyramid
  smallerPyramid : SquarePyramid

/-- Volume of a frustum --/
noncomputable def frustumVolume (f : Frustum) : ℝ :=
  pyramidVolume f.originalPyramid - pyramidVolume f.smallerPyramid

theorem frustum_volume_theorem (f : Frustum) 
  (h1 : f.originalPyramid.baseEdge = 16)
  (h2 : f.originalPyramid.altitude = 10)
  (h3 : f.smallerPyramid.baseEdge = 8)
  (h4 : f.smallerPyramid.altitude = 5) :
  frustumVolume f = 2240 / 3 := by
  sorry

#check frustum_volume_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_volume_theorem_l2_242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_circle_l2_268

-- Define the fixed circle N
def circle_N (x y : ℝ) : Prop := x^2 + (y+2)^2 = 4

-- Define the fixed point M
def point_M : ℝ × ℝ := (0, 2)

-- Define the property of internal tangency
def is_internally_tangent (center_P : ℝ × ℝ) (radius_P : ℝ) : Prop :=
  let (x, y) := center_P
  (x^2 + (y+2)^2) = (radius_P - 2)^2

-- Define the property of passing through point M
def passes_through_M (center_P : ℝ × ℝ) (radius_P : ℝ) : Prop :=
  let (x, y) := center_P
  (x - 0)^2 + (y - 2)^2 = radius_P^2

-- Statement of the theorem
theorem trajectory_of_moving_circle :
  ∀ (x y : ℝ),
  (∃ (radius_P : ℝ),
    is_internally_tangent (x, y) radius_P ∧
    passes_through_M (x, y) radius_P) →
  y < 0 →
  y^2 - x^2/3 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_moving_circle_l2_268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_upper_bound_l2_272

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - x) + Real.sqrt (1 + x)

-- Theorem for the range of f
theorem f_range : 
  ∀ x ∈ Set.Icc (-1 : ℝ) 1, f x ∈ Set.Icc (Real.sqrt 2) 2 := by sorry

-- Theorem for the upper bound of f on [0, 1]
theorem f_upper_bound : 
  ∀ x ∈ Set.Icc (0 : ℝ) 1, f x ≤ 2 - (1/4) * x^2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_f_upper_bound_l2_272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_50_digits_eq_270_l2_264

/-- The decimal representation of 1/1234 -/
def decimal_rep : ℚ := 1 / 1234

/-- The sequence of digits after the decimal point in the decimal representation of 1/1234 -/
def digit_sequence : ℕ → ℕ := sorry

/-- The sum of the first 50 digits after the decimal point in the decimal representation of 1/1234 -/
def sum_first_50_digits : ℕ := (Finset.range 50).sum (λ i ↦ digit_sequence i)

/-- Theorem: The sum of the first 50 digits after the decimal point in the decimal representation of 1/1234 is 270 -/
theorem sum_first_50_digits_eq_270 : sum_first_50_digits = 270 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_first_50_digits_eq_270_l2_264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_both_in_photo_probability_l2_245

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℝ  -- Time to complete one lap in seconds
  direction : Bool  -- true for counterclockwise, false for clockwise

/-- Represents the race scenario -/
structure RaceScenario where
  rachel : Runner
  robert : Runner
  photoStartTime : ℝ   -- in seconds
  photoEndTime : ℝ     -- in seconds
  photoCoverage : ℝ    -- fraction of track covered by photo

/-- Calculates the probability of both runners being in the photo simultaneously -/
noncomputable def probabilityBothInPhoto (scenario : RaceScenario) : ℝ :=
  sorry  -- Proof to be implemented

/-- The main theorem to be proved -/
theorem both_in_photo_probability (scenario : RaceScenario) : 
  scenario.rachel = Runner.mk 120 true ∧ 
  scenario.robert = Runner.mk 100 false ∧
  scenario.photoStartTime = 900 ∧
  scenario.photoEndTime = 1200 ∧
  scenario.photoCoverage = 1/3 →
  probabilityBothInPhoto scenario = 1/9 := by
  sorry  -- Proof to be implemented


end NUMINAMATH_CALUDE_ERRORFEEDBACK_both_in_photo_probability_l2_245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_drums_oil_transfer_l2_253

-- Define the capacity of drum X
variable (C : ℝ)

-- Define the oil content and capacity of each drum
noncomputable def drum_X_oil (C : ℝ) : ℝ := C / 2
noncomputable def drum_Y_capacity (C : ℝ) : ℝ := 2 * C
noncomputable def drum_Y_initial_oil (C : ℝ) : ℝ := (drum_Y_capacity C) / 4
noncomputable def drum_Z_capacity (C : ℝ) : ℝ := 3 * C
noncomputable def drum_Z_oil (C : ℝ) : ℝ := (drum_Z_capacity C) / 3

-- Theorem statement
theorem drums_oil_transfer (C : ℝ) (h : C > 0) :
  drum_Y_initial_oil C + drum_X_oil C + drum_Z_oil C = drum_Y_capacity C :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_drums_oil_transfer_l2_253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_periodic_l2_235

def prime_factor_sum (n : ℕ) : ℕ :=
  (Nat.factors n).sum + 1

def sequence_operation (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => prime_factor_sum (sequence_operation start n)

theorem sequence_eventually_periodic (start : ℕ) :
  ∃ (period : ℕ) (start_index : ℕ), period > 0 ∧
    ∀ (n : ℕ), n ≥ start_index →
      sequence_operation start n = sequence_operation start (n + period) :=
by
  sorry

#check sequence_eventually_periodic

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_eventually_periodic_l2_235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_l2_290

theorem cos_equation_solution (x : ℝ) :
  8 * (Real.cos x)^5 - 5 * Real.cos x - 2 * Real.cos (3 * x) = 1 →
  ∃ n : ℤ, x = 2 * Real.pi * ↑n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_equation_solution_l2_290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_neg_three_f_two_thirds_l2_217

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 3) + 1 / (x + 2)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ -3 ∧ x ≠ -2}

-- Theorem for the domain of f
theorem domain_of_f : 
  ∀ x : ℝ, x ∈ domain_f ↔ (x + 3 ≥ 0 ∧ x + 2 ≠ 0) := by
  sorry

-- Theorem for f(-3)
theorem f_neg_three : f (-3) = -1 := by
  sorry

-- Theorem for f(2/3)
theorem f_two_thirds : f (2/3) = (8 * Real.sqrt 33 + 9) / 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_f_neg_three_f_two_thirds_l2_217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_distance_l2_215

/-- The point where the mouse starts getting farther from the cheese -/
noncomputable def critical_point : ℝ × ℝ := (55/6, 83/6)

/-- The location of the cheese -/
def cheese_location : ℝ × ℝ := (15, 15)

/-- The equation of the line the mouse is running on: y = -5x + 23 -/
def mouse_path (x : ℝ) : ℝ := -5*x + 23

/-- Theorem stating that the critical point is where the mouse starts getting farther from the cheese -/
theorem mouse_cheese_distance :
  let (a, b) := critical_point
  let (cx, cy) := cheese_location
  let perpendicular_slope := (1 : ℝ)/5
  (b - cy = perpendicular_slope * (a - cx)) ∧ 
  (b = mouse_path a) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mouse_cheese_distance_l2_215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_l2_278

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle ABCD -/
structure Rectangle where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Calculates the angle between three points -/
noncomputable def angle (p1 p2 p3 : Point) : ℝ := sorry

theorem rectangle_diagonal (ABCD : Rectangle) (E : Point) :
  distance ABCD.A ABCD.B = 30 →
  distance ABCD.B ABCD.C = 15 →
  E.x = ABCD.C.x ∧ E.y ≥ ABCD.C.y ∧ E.y ≤ ABCD.D.y →
  angle ABCD.C ABCD.B E = π/6 →
  distance ABCD.A E = 15 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_diagonal_l2_278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l2_238

/-- Represents the time in hours it takes to fill a cistern when both filling and emptying taps are open simultaneously -/
noncomputable def simultaneous_fill_time (fill_rate : ℝ) (empty_rate : ℝ) : ℝ :=
  1 / (fill_rate - empty_rate)

/-- Theorem stating that it takes 6 hours to fill the cistern when both taps are open -/
theorem cistern_fill_time :
  let fill_rate : ℝ := 1 / 3  -- Cistern filled in 3 hours
  let empty_rate : ℝ := 1 / 6 -- Cistern emptied in 6 hours
  simultaneous_fill_time fill_rate empty_rate = 6 := by
  sorry

#check cistern_fill_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cistern_fill_time_l2_238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l2_287

/-- Represents an ellipse with center at the origin -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- The equation of the ellipse -/
def Ellipse.equation (e : Ellipse) (x y : ℝ) : Prop :=
  x^2 / e.a^2 + y^2 / e.b^2 = 1

/-- The eccentricity of the ellipse -/
noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - e.b^2 / e.a^2)

/-- A line passing through a point -/
structure Line where
  k : ℝ
  b : ℝ

/-- The equation of the line -/
def Line.equation (l : Line) (x y : ℝ) : Prop :=
  y = l.k * x + l.b

/-- Theorem stating the properties of the ellipse and the intersecting line -/
theorem ellipse_and_line_properties
  (e : Ellipse)
  (l : Line)
  (h_ecc : e.eccentricity = Real.sqrt 6 / 3)
  (h_focus_dist : Real.sqrt 3 = Real.sqrt (e.a^2 - e.b^2) + e.b)
  (h_line_point : l.equation 0 (3/2))
  (h_equal_distances : ∃ (M N : ℝ × ℝ),
    e.equation M.1 M.2 ∧
    e.equation N.1 N.2 ∧
    l.equation M.1 M.2 ∧
    l.equation N.1 N.2 ∧
    (M.1 - 0)^2 + (M.2 + e.b)^2 = (N.1 - 0)^2 + (N.2 + e.b)^2) :
  e.equation (Real.sqrt 3) 0 ∧ e.equation 0 1 ∧
  (l.k = Real.sqrt 6 / 3 ∨ l.k = -Real.sqrt 6 / 3) ∧ l.b = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_and_line_properties_l2_287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distance_bounds_l2_276

noncomputable section

/-- The ellipse x²/16 + y²/9 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The line 3x - 4y = 24 -/
def Line (x y : ℝ) : Prop := 3*x - 4*y = 24

/-- Distance from a point (x, y) to the line 3x - 4y = 24 -/
noncomputable def Distance (x y : ℝ) : ℝ := |3*x - 4*y - 24| / 5

theorem ellipse_line_distance_bounds :
  ∀ x y : ℝ, Ellipse x y →
  (∀ d : ℝ, Distance x y ≤ d → d ≤ 12*(2 + Real.sqrt 2)/5) ∧
  (∀ d : ℝ, 12*(2 - Real.sqrt 2)/5 ≤ d → d ≤ Distance x y) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_line_distance_bounds_l2_276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_is_zero_l2_258

noncomputable def f (n : ℝ) : ℝ := ((3 - n)^4 - (2 - n)^4) / ((1 - n)^5 - (1 + n)^3)

theorem limit_is_zero : 
  Filter.Tendsto f Filter.atTop (nhds 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_is_zero_l2_258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_opposite_shared_vertex_l2_299

-- Define the circle
structure Circle where
  center : EuclideanSpace ℝ (Fin 2)
  radius : ℝ

-- Define the square
structure Square where
  vertices : Fin 4 → EuclideanSpace ℝ (Fin 2)

-- Define the isosceles right triangle
structure IsoscelesRightTriangle where
  vertices : Fin 3 → EuclideanSpace ℝ (Fin 2)

-- Define predicates for inscribed shapes and isosceles right property
def InscribedIn (shape : Type) (circle : Circle) : Prop := sorry
def IsIsoscelesRight (triangle : IsoscelesRightTriangle) : Prop := sorry

-- Define the shared vertex and coinciding side
def SharedVertex (s : Square) (t : IsoscelesRightTriangle) : EuclideanSpace ℝ (Fin 2) := sorry
def CoincidingSide (s : Square) (t : IsoscelesRightTriangle) : Set (EuclideanSpace ℝ (Fin 2)) := sorry

-- Define the hypotenuse and angle opposite shared vertex
def Hypotenuse (t : IsoscelesRightTriangle) : Set (EuclideanSpace ℝ (Fin 2)) := sorry
def AngleOppositeSharedVertex (t : IsoscelesRightTriangle) : ℝ := sorry

-- Theorem statement
theorem angle_opposite_shared_vertex
  (c : Circle)
  (s : Square)
  (t : IsoscelesRightTriangle)
  (h1 : InscribedIn Square c)
  (h2 : InscribedIn IsoscelesRightTriangle c)
  (h3 : ∃ v, v = SharedVertex s t)
  (h4 : CoincidingSide s t = Hypotenuse t)
  (h5 : IsIsoscelesRight t) :
  AngleOppositeSharedVertex t = 90 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_opposite_shared_vertex_l2_299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_excluding_stops_l2_200

/-- Represents the speed of a bus in different scenarios -/
structure BusSpeed where
  /-- Speed including stoppages (km/h) -/
  withStops : ℚ
  /-- Duration of stoppages per hour (minutes) -/
  stopDuration : ℚ

/-- Calculates the speed of the bus excluding stoppages -/
def speedExcludingStops (bs : BusSpeed) : ℚ :=
  bs.withStops * 60 / (60 - bs.stopDuration)

/-- Theorem: Given a bus that travels at 45 km/h including stoppages
    and stops for 10 minutes each hour, its speed excluding stoppages is 54 km/h -/
theorem bus_speed_excluding_stops :
  let bs : BusSpeed := { withStops := 45, stopDuration := 10 }
  speedExcludingStops bs = 54 := by
  sorry

#eval speedExcludingStops { withStops := 45, stopDuration := 10 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_speed_excluding_stops_l2_200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l2_225

/-- Given two vectors a and b in ℝ², prove that if the angle between them is 120°,
    |a| = 3, and |a + b| = √13, then |b| = 4. -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  (Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * Real.pi / 3) →
  Real.sqrt (a.1^2 + a.2^2) = 3 →
  Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = Real.sqrt 13 →
  Real.sqrt (b.1^2 + b.2^2) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_magnitude_l2_225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_irrational_rotation_dense_l2_288

/-- A sequence is dense in [0,1] if for any ε > 0 and y ∈ [0,1], 
    there exists an element of the sequence within ε of y. -/
def is_dense_in_unit_interval (s : ℕ → ℝ) : Prop :=
  ∀ ε > 0, ∀ y ∈ Set.Icc 0 1, ∃ n : ℕ, |s n - y| < ε

/-- The fractional part of a real number -/
noncomputable def frac (x : ℝ) : ℝ := x - ⌊x⌋

theorem periodic_function_irrational_rotation_dense
  (f : ℝ → ℝ) (hf_cont : Continuous f) (hf_per : ∃ T > 0, ∀ x, f (x + T) = f x)
  (α : ℝ) (hα_irrat : Irrational α) :
  is_dense_in_unit_interval (λ n => frac (n * α + f (n * α))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_periodic_function_irrational_rotation_dense_l2_288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_exists_l2_239

/-- The depth of the well -/
noncomputable def depth : ℝ := sorry

/-- The time it takes for the stone to fall to the bottom of the well -/
noncomputable def fall_time (d : ℝ) : ℝ := Real.sqrt (d / 20)

/-- The time it takes for the sound to travel from the bottom to the top of the well -/
noncomputable def sound_time (d : ℝ) : ℝ := d / 1050

/-- The theorem stating the existence of a depth satisfying the problem conditions -/
theorem well_depth_exists : ∃ d : ℝ, fall_time d + sound_time d = 10.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_well_depth_exists_l2_239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_approx_l2_256

/-- Calculates the selling price of a bond given its face value, interest rate on face value, and interest as a percentage of selling price. -/
noncomputable def bond_selling_price (face_value : ℝ) (interest_rate_face : ℝ) (interest_rate_selling : ℝ) : ℝ :=
  (face_value * interest_rate_face) / interest_rate_selling

/-- Theorem stating that the selling price of a bond with given parameters is approximately 3846.15 -/
theorem bond_selling_price_approx :
  let face_value : ℝ := 5000
  let interest_rate_face : ℝ := 0.05
  let interest_rate_selling : ℝ := 0.065
  let selling_price := bond_selling_price face_value interest_rate_face interest_rate_selling
  abs (selling_price - 3846.15) < 0.01 := by
  sorry

/-- Evaluation of the bond selling price -/
def evaluate_bond_selling_price : ℚ :=
  (5000 : ℚ) * (5 : ℚ) / (100 : ℚ) / ((65 : ℚ) / (1000 : ℚ))

#eval evaluate_bond_selling_price

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bond_selling_price_approx_l2_256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cross_bridge_time_l2_283

-- Define the train length in meters
noncomputable def train_length : ℝ := 110

-- Define the bridge length in meters
noncomputable def bridge_length : ℝ := 340

-- Define the train speed in kilometers per hour
noncomputable def train_speed_kmph : ℝ := 60

-- Convert km/h to m/s
noncomputable def train_speed_ms : ℝ := train_speed_kmph * (1000 / 3600)

-- Calculate the total distance
noncomputable def total_distance : ℝ := train_length + bridge_length

-- Calculate the time taken
noncomputable def time_taken : ℝ := total_distance / train_speed_ms

-- Theorem to prove
theorem train_cross_bridge_time :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ abs (time_taken - 27) < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_cross_bridge_time_l2_283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_XZ_length_is_26_div_3_l2_269

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)

-- Define the properties of the triangle
def is_right_angled (t : Triangle) : Prop :=
  let (x1, y1) := t.X
  let (x2, y2) := t.Y
  let (x3, y3) := t.Z
  (x2 - x1) * (x3 - x1) + (y2 - y1) * (y3 - y1) = 0

def angle_X_is_90_degrees (t : Triangle) : Prop :=
  is_right_angled t

noncomputable def YZ_length (t : Triangle) : ℝ :=
  let (x2, y2) := t.Y
  let (x3, y3) := t.Z
  Real.sqrt ((x3 - x2)^2 + (y3 - y2)^2)

noncomputable def tan_Z (t : Triangle) : ℝ :=
  let (x1, y1) := t.X
  let (x2, y2) := t.Y
  let (x3, y3) := t.Z
  (y2 - y1) / (x2 - x1)

noncomputable def sin_Z (t : Triangle) : ℝ :=
  let (x1, y1) := t.X
  let (x2, y2) := t.Y
  let (x3, y3) := t.Z
  (y2 - y1) / YZ_length t

noncomputable def XZ_length (t : Triangle) : ℝ :=
  let (x1, y1) := t.X
  let (x3, y3) := t.Z
  Real.sqrt ((x3 - x1)^2 + (y3 - y1)^2)

-- State the theorem
theorem XZ_length_is_26_div_3 (t : Triangle) :
  angle_X_is_90_degrees t →
  YZ_length t = 26 →
  tan_Z t = 3 * sin_Z t →
  XZ_length t = 26 / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_XZ_length_is_26_div_3_l2_269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocals_l2_286

theorem tan_sum_reciprocals (a b : ℝ) 
  (h1 : Real.sin a / Real.cos b + Real.sin b / Real.cos a = 2)
  (h2 : Real.cos a / Real.sin b + Real.cos b / Real.sin a = 4) :
  Real.tan a / Real.tan b + Real.tan b / Real.tan a = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_sum_reciprocals_l2_286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_sequence_l2_209

/-- Represents the allowed operations on the matrix -/
inductive Operation
  | DoubleRow (row : Fin 8)
  | SubtractColumn (col : Fin 5)

/-- The type of our matrix -/
def MyMatrix := Fin 8 → Fin 5 → ℕ

/-- Apply an operation to a matrix -/
def applyOperation (m : MyMatrix) (op : Operation) : MyMatrix :=
  match op with
  | Operation.DoubleRow row =>
      fun i j => if i = row then 2 * m i j else m i j
  | Operation.SubtractColumn col =>
      fun i j => if j = col then m i j - 1 else m i j

/-- A sequence of operations -/
def OperationSequence := List Operation

/-- Apply a sequence of operations to a matrix -/
def applySequence (m : MyMatrix) (seq : OperationSequence) : MyMatrix :=
  seq.foldl applyOperation m

/-- Check if a matrix is zero -/
def isZeroMatrix (m : MyMatrix) : Prop :=
  ∀ i j, m i j = 0

/-- The main theorem -/
theorem exists_zero_sequence (m : MyMatrix) :
  ∃ (seq : OperationSequence), isZeroMatrix (applySequence m seq) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_sequence_l2_209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_always_negative_iff_k_in_range_l2_259

/-- The function f(x) = kx^2 - 4x + k - 3 is negative for all real x if and only if k is in the open interval (-∞, -1) -/
theorem function_always_negative_iff_k_in_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 4*x + k - 3 < 0) ↔ k ∈ Set.Ioi (-1 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_always_negative_iff_k_in_range_l2_259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l2_261

noncomputable section

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

/-- Definition of the point P -/
def P : ℝ × ℝ := (4, 0)

/-- Definition of the fixed point Q -/
def Q : ℝ × ℝ := (2/5, 0)

/-- Definition of the dot product condition -/
def satisfies_dot_product_condition (M N : ℝ × ℝ) : Prop :=
  let (mx, my) := M
  let (nx, ny) := N
  let (px, py) := P
  (mx - px) * (nx - px) + (my - py) * (ny - py) = 12

/-- Main theorem statement -/
theorem ellipse_fixed_point :
  is_on_ellipse Q.1 Q.2 ∧
  ∀ (M N : ℝ × ℝ),
    is_on_ellipse M.1 M.2 →
    is_on_ellipse N.1 N.2 →
    ∃ (k m : ℝ), (M.2 = k * M.1 + m ∧ N.2 = k * N.1 + m ∧ Q.2 = k * Q.1 + m) →
    satisfies_dot_product_condition M N :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l2_261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l2_241

def set_A : Set ℝ := {x | x ≤ 3}
def set_B : Set ℕ := {x | x - 1 ≥ 0}

def set_B_real : Set ℝ := {x | ∃ n : ℕ, n ∈ set_B ∧ x = n}

theorem intersection_of_A_and_B :
  (set_A ∩ set_B_real) = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l2_241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l2_282

-- Define an isosceles triangle with given side lengths
noncomputable def IsoscelesTriangle (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b / 2

-- Define the area of the triangle
noncomputable def TriangleArea (a b : ℝ) : ℝ :=
  let h := Real.sqrt (a^2 - (b/2)^2)
  (b * h) / 2

-- Theorem statement
theorem isosceles_triangle_area :
  ∀ a b : ℝ, IsoscelesTriangle a b →
  a = 17 ∧ b = 16 →
  TriangleArea a b = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_area_l2_282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l2_212

noncomputable def f (x : ℝ) := Real.sin (2 * x) - Real.cos (2 * x + Real.pi / 6)

theorem f_properties :
  (∀ y ∈ Set.range f, -Real.sqrt 3 ≤ y ∧ y ≤ Real.sqrt 3) ∧
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (Real.pi / 3 + k * Real.pi) (5 * Real.pi / 6 + k * Real.pi),
    ∀ y ∈ Set.Icc (Real.pi / 3 + k * Real.pi) (5 * Real.pi / 6 + k * Real.pi),
    x < y → f y < f x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l2_212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_6_or_8_not_both_l2_262

/-- The set of positive integers less than 201 that are multiples of either 6 or 8, but not both -/
def S : Finset ℕ := (Finset.range 201).filter (fun n => (n % 6 = 0 ∨ n % 8 = 0) ∧ ¬(n % 6 = 0 ∧ n % 8 = 0))

/-- The cardinality of set S is 42 -/
theorem count_multiples_6_or_8_not_both : S.card = 42 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_multiples_6_or_8_not_both_l2_262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_sold_day1_l2_266

/-- Represents the sales and earnings of a flower shop over three days -/
structure FlowerShopSales where
  tulips_day1 : ℕ
  roses_day1 : ℕ
  tulip_price : ℚ
  rose_price : ℚ
  total_earnings : ℚ
  -- Invariant: total_earnings equals the sum of earnings over three days

/-- The theorem stating the number of roses sold on the first day -/
theorem roses_sold_day1 (sale : FlowerShopSales) : 
  sale.tulips_day1 = 30 ∧ 
  sale.tulip_price = 2 ∧ 
  sale.rose_price = 3 ∧
  sale.total_earnings = 420 →
  sale.roses_day1 = 20 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roses_sold_day1_l2_266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_of_8_equals_11_l2_226

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

-- Define f(n)
def f (n : ℕ+) : ℕ := sum_of_digits ((n : ℕ)^2 + 1)

-- Define f_k(n) recursively
def f_k : ℕ → ℕ+ → ℕ
  | 0, n => (n : ℕ)
  | k+1, n => f ⟨f_k k n, by sorry⟩

theorem f_2005_of_8_equals_11 : f_k 2005 8 = 11 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2005_of_8_equals_11_l2_226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_value_difference_l2_265

def total_coins : ℕ := 2300
def penny_value : ℕ := 1
def nickel_value : ℕ := 5

theorem coin_value_difference :
  ∃ (max_value min_value : ℕ),
    (∀ (pennies : ℕ),
      1 ≤ pennies → pennies ≤ total_coins - 1 →
      let nickels := total_coins - pennies
      let value := penny_value * pennies + nickel_value * nickels
      value ≤ max_value ∧ min_value ≤ value) ∧
    max_value - min_value = 9192 := by
  -- Proof goes here
  sorry

#check coin_value_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_value_difference_l2_265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_condition_for_cube_root_sum_l2_205

theorem integer_condition_for_cube_root_sum (a : ℝ) (h_a : 0 < a) :
  let b := (a + Real.sqrt (a^2 + 1))^(1/3) + (a - Real.sqrt (a^2 + 1))^(1/3)
  (∃ n : ℕ, (b : ℝ) = n) ↔ ∃ n : ℕ, a = (n * (n^2 + 3) : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_condition_for_cube_root_sum_l2_205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_power_towers_l2_222

-- Define a function to represent the power tower
def powerTower (base : ℕ) : ℕ → ℕ
  | 0 => 1
  | n + 1 => base ^ (powerTower base n)

-- State the theorem
theorem no_equal_power_towers :
  ¬ ∃ (a b m n : ℕ), a ≠ b ∧ a > 1 ∧ b > 1 ∧ m ≥ 2 ∧ n ≥ 2 ∧ powerTower a m = powerTower b n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_equal_power_towers_l2_222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_point_l2_230

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

noncomputable def distance_PM (x y a : ℝ) : ℝ := Real.sqrt ((x - a)^2 + y^2)

theorem min_distance_ellipse_point (a : ℝ) 
  (h1 : 0 < a) (h2 : a < 3) :
  (∃ x y : ℝ, ellipse x y ∧ 
    (∀ x' y' : ℝ, ellipse x' y' → distance_PM x y a ≤ distance_PM x' y' a) ∧
    distance_PM x y a = 1) →
  a = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_point_l2_230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_pyramid_volume_l2_284

/-- A pyramid with a right triangular base -/
structure RightTrianglePyramid where
  /-- Height of the pyramid -/
  H : ℝ
  /-- Acute angle of the right triangle base -/
  α : ℝ
  /-- Angle between lateral edges and the base -/
  β : ℝ
  /-- H is positive -/
  H_pos : 0 < H
  /-- α is between 0 and π/2 -/
  α_range : 0 < α ∧ α < π/2
  /-- β is between 0 and π/2 -/
  β_range : 0 < β ∧ β < π/2

/-- The volume of a right triangle pyramid -/
noncomputable def volume (p : RightTrianglePyramid) : ℝ :=
  (p.H^3 * Real.sin (2 * p.α)) / (3 * (Real.tan p.β)^2)

/-- Theorem stating the volume of a right triangle pyramid -/
theorem right_triangle_pyramid_volume (p : RightTrianglePyramid) :
  volume p = (p.H^3 * Real.sin (2 * p.α)) / (3 * (Real.tan p.β)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_pyramid_volume_l2_284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l2_280

def a (n : ℕ) : ℝ := 2^(5 - n)

def b (n : ℕ) (k : ℝ) : ℝ := n + k

noncomputable def c (n : ℕ) (k : ℝ) : ℝ := 
  if a n ≤ b n k then b n k else a n

theorem range_of_k : 
  (∀ k : ℝ, (∀ n : ℕ, n ≥ 1 → c 5 k ≤ c n k) ↔ k ∈ Set.Icc (-4) (-3)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_k_l2_280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_inequalities_l2_294

theorem rectangular_prism_inequalities
  (a b c : ℝ)
  (h_abc : a > b ∧ b > c)
  (p : ℝ)
  (h_p : p = 4 * (a + b + c))
  (S : ℝ)
  (h_S : S = 2 * (a * b + b * c + c * a))
  (d : ℝ)
  (h_d : d^2 = a^2 + b^2 + c^2) :
  a > (1/3) * (p/4 + Real.sqrt (d^2 - S/2)) ∧
  c < (1/3) * (p/4 - Real.sqrt (d^2 - S/2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_prism_inequalities_l2_294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_circle_equation_conversion_l2_248

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (Real.sqrt 3 / 2) * t, 1 + (1 / 2) * t)

-- Define the circle in polar coordinates
noncomputable def circle_polar (θ : ℝ) : ℝ := 10 * Real.cos (Real.pi / 3 - θ)

-- Define the circle in Cartesian coordinates
def circle_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 5*x - 5*Real.sqrt 3*y = 0

-- Theorem for the line equation
theorem line_equation_correct : 
  ∀ t : ℝ, line_l t = (1 + (Real.sqrt 3 / 2) * t, 1 + (1 / 2) * t) := by
  sorry

-- Theorem for the circle equation conversion
theorem circle_equation_conversion : 
  ∀ x y : ℝ, (∃ θ : ℝ, x = circle_polar θ * Real.cos θ ∧ y = circle_polar θ * Real.sin θ) ↔ circle_cartesian x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_correct_circle_equation_conversion_l2_248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l2_219

noncomputable section

/-- A geometric sequence with first term 120, positive second term a, and third term 45/28 -/
def GeometricSequence (a : ℝ) : Prop :=
  a > 0 ∧ ∃ r : ℝ, r ≠ 0 ∧ a = 120 * r ∧ 45/28 = a * r

/-- The sum of the first three terms of the geometric sequence -/
def SumFirstThreeTerms (a : ℝ) : ℝ :=
  120 + a + 45/28

theorem geometric_sequence_properties (a : ℝ) (h : GeometricSequence a) :
  a = 5 * Real.sqrt (135/7) ∧
  SumFirstThreeTerms a = 120 + 5 * Real.sqrt (135/7) + 45/28 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l2_219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_for_all_secrets_l2_252

/-- Represents a person with their known secrets -/
structure Person where
  id : Nat
  knownSecrets : Finset Nat

/-- Represents the state of knowledge distribution among people -/
structure KnowledgeState where
  people : Finset Person
  totalSecrets : Nat

/-- Simulates a phone call where caller shares all known secrets with recipient -/
def makeCall (state : KnowledgeState) (caller : Person) (recipient : Person) : KnowledgeState :=
  sorry

/-- Checks if all people know all secrets -/
def allKnowAllSecrets (state : KnowledgeState) : Prop :=
  sorry

/-- The main theorem stating the minimum number of calls required -/
theorem min_calls_for_all_secrets (n : Nat) :
  ∃ (initialState : KnowledgeState) (callSequence : List (Person × Person)),
    initialState.totalSecrets = n ∧
    initialState.people.card = n ∧
    (∀ p ∈ initialState.people, p.knownSecrets.card = 1) ∧
    allKnowAllSecrets (callSequence.foldl (fun state (caller, recipient) => makeCall state caller recipient) initialState) ∧
    callSequence.length = 2 * n - 2 ∧
    (∀ shorterSequence : List (Person × Person), shorterSequence.length < 2 * n - 2 →
      ¬allKnowAllSecrets (shorterSequence.foldl (fun state (caller, recipient) => makeCall state caller recipient) initialState)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_calls_for_all_secrets_l2_252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_time_l2_263

/-- The time taken by worker a to complete the work alone -/
def time_a : ℝ := sorry

/-- The time taken by worker b to complete the work alone -/
def time_b : ℝ := 20

/-- The time taken by worker c to complete the work alone -/
def time_c : ℝ := 45

/-- The time taken by workers a, b, and c to complete the work together -/
def time_together : ℝ := 7.2

/-- Theorem stating that under given conditions, worker a completes the work in 15 days -/
theorem worker_a_time : 
  (1 / time_a + 1 / time_b + 1 / time_c = 1 / time_together) → 
  time_a = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_worker_a_time_l2_263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_expression_value_at_half_l2_234

theorem expression_evaluation (x : ℝ) (h : x > 0) :
  (x^(2/3 : ℝ) * x^(-1/2 : ℝ)) / ((x^(1/3 : ℝ))^2 * x^(1/2 : ℝ)) = 1 / x :=
by sorry

theorem expression_value_at_half :
  (((1/2:ℝ)^(2/3 : ℝ) * (1/2:ℝ)^(-1/2 : ℝ)) / (((1/2:ℝ)^(1/3 : ℝ))^2 * (1/2:ℝ)^(1/2 : ℝ))) = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_expression_value_at_half_l2_234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fundamental_terms_div_four_l2_251

/-- Represents a cell in the grid -/
inductive Cell
| Pos : Cell  -- Represents +1
| Neg : Cell  -- Represents -1

/-- Represents an n × n grid filled with +1 or -1 -/
def Grid (n : ℕ) := Fin n → Fin n → Cell

/-- A fundamental term is a product of n cells, with no two in the same row or column -/
def FundamentalTerm (n : ℕ) (grid : Grid n) (perm : Equiv.Perm (Fin n)) : Int :=
  (Finset.univ.prod fun i => match grid i (perm i) with
    | Cell.Pos => 1
    | Cell.Neg => -1)

/-- The sum of all fundamental terms -/
def SumOfFundamentalTerms (n : ℕ) (grid : Grid n) : Int :=
  Finset.sum (Finset.univ : Finset (Equiv.Perm (Fin n))) fun perm => FundamentalTerm n grid perm

/-- Theorem: The sum of all fundamental terms is always divisible by 4 for n ≥ 4 -/
theorem sum_of_fundamental_terms_div_four (n : ℕ) (h : n ≥ 4) (grid : Grid n) :
  4 ∣ SumOfFundamentalTerms n grid := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_fundamental_terms_div_four_l2_251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_and_double_angle_l2_250

theorem cos_angle_and_double_angle (α : ℝ) :
  (∃ (x y : ℝ), x = -3 ∧ y = 4 ∧ x^2 + y^2 = 25 ∧ Real.cos α = x / Real.sqrt (x^2 + y^2)) →
  Real.cos α = -3/5 ∧ Real.cos (2*α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_and_double_angle_l2_250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_exchange_impossibility_l2_281

/-- Represents a carpet with two dimensions -/
structure Carpet where
  length : ℝ
  width : ℝ
deriving Inhabited

/-- Represents the state of all carpets after a series of exchanges -/
structure CarpetState where
  carpets : List Carpet

/-- Defines the initial state with one large carpet -/
noncomputable def initial_state (a b : ℝ) : CarpetState :=
  { carpets := [{ length := a, width := b }] }

/-- Defines the first type of exchange -/
noncomputable def exchange_type1 (c : Carpet) : Carpet :=
  { length := 1 / c.length, width := 1 / c.width }

/-- Defines the second type of exchange -/
noncomputable def exchange_type2 (c : Carpet) (x : ℝ) : (Carpet × Carpet) :=
  ({ length := x, width := c.width }, { length := c.length / x, width := c.width })

/-- Defines a single exchange operation -/
noncomputable def exchange (state : CarpetState) (index : Nat) (type : Nat) (x : ℝ) : CarpetState :=
  match type with
  | 1 => { carpets := state.carpets.set index (exchange_type1 (state.carpets.get! index)) }
  | 2 => 
    let (c1, c2) := exchange_type2 (state.carpets.get! index) x
    { carpets := state.carpets.removeNth index ++ [c1, c2] }
  | _ => state

/-- Theorem: It's impossible to reach a state where all carpets have one side > 1 and one side < 1 -/
theorem carpet_exchange_impossibility (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  ∀ (final_state : CarpetState),
  (∃ (exchange_sequence : List (Nat × Nat × ℝ)), 
   final_state = exchange_sequence.foldl (λ s (idx, t, x) => exchange s idx t x) (initial_state a b)) →
  ∃ (c : Carpet), c ∈ final_state.carpets ∧ (c.length > 1 ∧ c.width > 1 ∨ c.length < 1 ∧ c.width < 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carpet_exchange_impossibility_l2_281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_affine_preserves_ratio_l2_273

/-- An affine transformation in a vector space -/
def AffineTransformation (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  {L : V → V // ∀ (x y : V) (t : ℝ), L ((1 - t) • x + t • y) = (1 - t) • L x + t • L y}

/-- Theorem: If C divides AB in ratio p:q, then C' divides A'B' in the same ratio -/
theorem affine_preserves_ratio
  {V : Type*} [AddCommGroup V] [Module ℝ V]
  (L : AffineTransformation V)
  (A B C A' B' C' : V)
  (p q : ℝ)
  (hA : A' = L.val A)
  (hB : B' = L.val B)
  (hC : C' = L.val C)
  (hABC : q • (C - A) = p • (B - C)) :
  q • (C' - A') = p • (B' - C') :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_affine_preserves_ratio_l2_273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l2_285

theorem trig_identity (α : ℝ) : 
  (Real.sin α + (Real.sin α)⁻¹ + Real.tan α)^2 + (Real.cos α + (Real.cos α)⁻¹ + (Real.tan α)⁻¹)^2 
  = 9 + 2 * (Real.tan α^2 + (Real.tan α)⁻¹^2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l2_285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_single_value_point_l2_201

/-- A polynomial that has an infinite set of values, each taken at least at two integer points -/
structure SpecialPolynomial (R : Type) [CommRing R] where
  P : Polynomial R
  infiniteDoubleValues : Set R
  infiniteDoubleValuesProperty : 
    Set.Infinite infiniteDoubleValues ∧ 
    ∀ y ∈ infiniteDoubleValues, ∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ 
      (Polynomial.eval (↑x₁ : R) P = y) ∧ (Polynomial.eval (↑x₂ : R) P = y)

/-- The main theorem stating that there is at most one value taken at exactly one integer point -/
theorem at_most_one_single_value_point (R : Type) [CommRing R] [CharZero R] (P : SpecialPolynomial R) :
  ∃! y : R, ∃! x : ℤ, Polynomial.eval (↑x : R) P.P = y :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_at_most_one_single_value_point_l2_201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l2_271

theorem cube_root_equation_solution :
  ∃ (x : ℚ), (5 - x) = (-5/3)^3 ∧ x = 260/27 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l2_271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_min_value_l2_275

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 1 ≤ x ∧ x ≤ 2 then 1
  else if 2 < x ∧ x ≤ 3 then x - 1
  else 0  -- This case is added to make f total, but it's not used in the problem

-- Define v(a) as specified in the problem
noncomputable def v (a : ℝ) : ℝ :=
  let s := Set.Icc 1 3  -- The closed interval [1, 3]
  (⨆ (x : ℝ) (_ : x ∈ s), f x - a * x) - (⨅ (x : ℝ) (_ : x ∈ s), f x - a * x)

-- State the theorem
theorem v_min_value :
  ∃ (a_min : ℝ), ∀ (a : ℝ), v a_min ≤ v a ∧ v a_min = (1 : ℝ) / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_min_value_l2_275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_supplies_cost_l2_244

/-- Calculates the total cost of school supplies for a class --/
def totalCost (
  numStudents : ℕ) 
  (penQuantity notebookQuantity binderQuantity highlighterQuantity folderQuantity : ℕ)
  (penPrice notebookPrice binderPrice highlighterPrice folderPrice : ℚ)
  (teacherDiscount bulkDiscount : ℚ)
  (taxRate : ℚ)
  : ℚ :=
  let subtotal := 
    numStudents * (
      penQuantity * penPrice +
      notebookQuantity * notebookPrice +
      binderQuantity * binderPrice +
      highlighterQuantity * highlighterPrice +
      folderQuantity * folderPrice
    )
  let afterDiscounts := subtotal - teacherDiscount - bulkDiscount
  let tax := afterDiscounts * taxRate
  afterDiscounts + tax

/-- The total cost of school supplies is $1622.25 --/
theorem school_supplies_cost : 
  totalCost 
    50 -- numStudents
    7 5 3 4 2 -- item quantities
    (70/100) (160/100) (510/100) (90/100) (115/100) -- item prices
    135 25 -- discounts
    (5/100) -- tax rate
  = 1622.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_supplies_cost_l2_244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_b_value_l2_224

open Real

-- Define the function f(A)
noncomputable def f (A : ℝ) : ℝ :=
  2 * (cos (A / 2)) * (sin (π - A / 2)) + (sin (A / 2))^2 - (cos (A / 2))^2

-- Theorem for the maximum value of f(A)
theorem f_max_value : ∃ (A : ℝ), 0 < A ∧ A < π ∧ f A = sqrt 2 ∧ ∀ (B : ℝ), 0 < B → B < π → f B ≤ f A := by
  sorry

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  angle_sum : A + B + C = π
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  sine_law : a / sin A = b / sin B

-- Theorem for the value of b
theorem b_value (t : Triangle) (h1 : f t.A = 0) (h2 : t.C = 5 * π / 12) (h3 : t.a = sqrt 6) : t.b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_b_value_l2_224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_2030th_position_l2_216

/-- Represents the possible orientations of the square -/
inductive SquareOrientation
  | EFGH
  | GHEF
  | EFHG
  | HGEF

/-- Applies the transformation (180-degree rotation followed by horizontal reflection) to the given orientation -/
def transform (o : SquareOrientation) : SquareOrientation :=
  match o with
  | SquareOrientation.EFGH => SquareOrientation.EFHG
  | SquareOrientation.GHEF => SquareOrientation.HGEF
  | SquareOrientation.EFHG => SquareOrientation.EFGH
  | SquareOrientation.HGEF => SquareOrientation.GHEF

/-- Returns the orientation after n transformations -/
def nthOrientation (n : Nat) : SquareOrientation :=
  match n % 4 with
  | 0 => SquareOrientation.EFGH
  | 1 => SquareOrientation.GHEF
  | 2 => SquareOrientation.EFHG
  | _ => SquareOrientation.HGEF

theorem square_2030th_position :
  nthOrientation 2030 = SquareOrientation.GHEF := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_2030th_position_l2_216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_not_odd_function_inequality_when_a_2_l2_220

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x + 1) / (x - 1)

-- Theorem 1: Center of symmetry
theorem center_of_symmetry (a : ℝ) :
  ∀ x₁ x₂ : ℝ, x₁ ≠ 1 ∧ x₂ ≠ 1 → 
  (x₁ - 1 = -(x₂ - 1) → f a x₁ - a = -(f a x₂ - a)) :=
by
  sorry

-- Theorem 2: Not an odd function
theorem not_odd_function :
  ∀ a : ℝ, ∃ x : ℝ, x ≠ 1 ∧ x ≠ 0 ∧ f a (-x) ≠ -f a x :=
by
  sorry

-- Theorem 3: Inequality when a = 2
theorem inequality_when_a_2 :
  ∀ x₁ x₂ : ℝ, 2 < x₁ ∧ x₁ < x₂ → f 2 x₁ - f 2 x₂ < 3 * (x₂ - x₁) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_center_of_symmetry_not_odd_function_inequality_when_a_2_l2_220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l2_249

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 / Real.exp x - 2 * x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (f (a - 3) + f (2 * a^2) ≤ 0) ↔ (-3/2 ≤ a ∧ a ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l2_249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_integers_in_list_l2_296

def consecutive_integers (start : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => start + i)

theorem range_of_positive_integers_in_list (k : List ℤ) :
  k = consecutive_integers (-5) 12 →
  (k.filter (λ x => x > 0)).maximum? = some 6 ∧
  (k.filter (λ x => x > 0)).minimum? = some 1 := by
  sorry

#eval consecutive_integers (-5) 12
#eval (consecutive_integers (-5) 12).filter (λ x => x > 0)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positive_integers_in_list_l2_296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l2_233

/-- The area of a triangle with vertices at (2, 3), (8, 3), and (5, 10) is 21 square units. -/
theorem triangle_area 
  (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ)
  (h₁ : x₁ = 2 ∧ y₁ = 3)
  (h₂ : x₂ = 8 ∧ y₂ = 3)
  (h₃ : x₃ = 5 ∧ y₃ = 10) :
  (1/2 : ℝ) * abs ((x₂ - x₁) * (y₃ - y₁) - (x₃ - x₁) * (y₂ - y₁)) = 21 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l2_233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_sequence_properties_l2_229

-- Define the sequence α_n
noncomputable def alpha_seq (α : ℝ) : ℕ → ℝ
  | 0 => α
  | n + 1 => min (2 * alpha_seq α n) (1 - 2 * alpha_seq α n)

-- Statement of the theorem
theorem alpha_sequence_properties (α : ℝ) (h_irr : Irrational α) (h_bound : 0 < α ∧ α < 1/2) :
  (∃ n : ℕ, alpha_seq α n < 3/16) ∧
  (∃ α : ℝ, ∀ n : ℕ, alpha_seq α n > 7/40) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alpha_sequence_properties_l2_229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_bar_cost_for_scout_camp_l2_246

/-- Calculates the cost of chocolate bars for a scout camp out -/
theorem chocolate_bar_cost_for_scout_camp (bar_cost : ℚ) (smores_per_bar : ℕ) 
  (num_scouts : ℕ) (smores_per_scout : ℕ) (discount_rate : ℚ) 
  (discount_threshold : ℕ) : bar_cost = 3/2 → smores_per_bar = 3 → 
  num_scouts = 15 → smores_per_scout = 2 → discount_rate = 15/100 → 
  discount_threshold = 10 → 
  (num_scouts * smores_per_scout + smores_per_bar - 1) / smores_per_bar * bar_cost * 
  (1 - discount_rate) = 51/4 := by
  sorry

#eval (15 * 2 + 3 - 1) / 3 * (3/2 : ℚ) * (1 - 15/100) -- This should evaluate to 51/4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chocolate_bar_cost_for_scout_camp_l2_246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_l2_232

-- Define the circle
def on_unit_circle (m n : ℝ) : Prop := m^2 + n^2 = 1

-- Define the point Q
def point_Q (m n : ℝ) : ℝ × ℝ := (m - n, 2 * m * n)

-- Theorem statement
theorem trajectory_of_Q {m n : ℝ} (h : on_unit_circle m n) :
  let (x, y) := point_Q m n
  x^2 + y = 1 ∧ x^2 ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_Q_l2_232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_k_value_l2_267

-- Define the points A, B, and C
def A : ℝ × ℝ := (5, 5)
def B : ℝ × ℝ := (2, 1)
def C (k : ℝ) : ℝ × ℝ := (0, k)

-- Define the distance function between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the total distance function
noncomputable def total_distance (k : ℝ) : ℝ :=
  distance A (C k) + distance B (C k)

-- State the theorem
theorem optimal_k_value :
  ∃ k : ℝ, k = 15/7 ∧ ∀ x : ℝ, total_distance k ≤ total_distance x :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_k_value_l2_267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_zero_solution_l2_255

-- Define the function f as noncomputable
noncomputable def f (a b x : ℝ) : ℝ := 2 / (a * x + b)

-- State the theorem
theorem inverse_f_zero_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ∃ x : ℝ, f a b (2 / b) = x ∧ f a b x = 0 ↔ x = 2 / b :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_f_zero_solution_l2_255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_arrangement_l2_237

theorem digit_arrangement (a : Finset ℕ) (h1 : a.card = 5) (h2 : ∀ x ∈ a, x ≤ 5) :
  (∃ n : ℕ, (Nat.factorial (5 * n)) / ((Nat.factorial n) ^ 5) = 120) →
  n = 1 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_arrangement_l2_237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moderate_intensity_sets_l2_277

/-- Physical health coefficient -/
noncomputable def k : ℝ := 7

/-- Normal heart rate in beats per minute -/
noncomputable def normal_heart_rate : ℝ := 80

/-- Number of push-ups in one set -/
noncomputable def pushups_per_set : ℝ := 12

/-- Heart rate after push-ups as a function of the number of push-ups -/
noncomputable def heart_rate (x : ℝ) : ℝ := normal_heart_rate * (Real.log (Real.sqrt (x / pushups_per_set)) + 1)

/-- Ratio of heart rate after exercise to normal heart rate -/
noncomputable def heart_rate_ratio (a : ℝ) : ℝ := heart_rate (pushups_per_set * a) / normal_heart_rate

/-- Function relating heart rate ratio to exercise intensity -/
noncomputable def f (t : ℝ) : ℝ := k * Real.exp t

/-- Theorem stating that 3 is the only positive integer number of sets
    that satisfies the moderate intensity exercise condition -/
theorem moderate_intensity_sets :
  ∃! (a : ℕ), a > 0 ∧ 28 ≤ f (heart_rate_ratio (a : ℝ)) ∧ f (heart_rate_ratio (a : ℝ)) ≤ 34 ∧ a = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moderate_intensity_sets_l2_277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_l2_221

/-- A sphere with an inscribed cone and a circumscribed cone -/
structure SphereConeSystem where
  r : ℝ
  h : r > 0

/-- Surface area of the inscribed cone -/
noncomputable def inscribed_cone_surface_area (s : SphereConeSystem) : ℝ :=
  (3 * Real.pi * s.r^2) / 4

/-- Surface area of the sphere -/
noncomputable def sphere_surface_area (s : SphereConeSystem) : ℝ :=
  4 * Real.pi * s.r^2

/-- Surface area of the circumscribed cone -/
noncomputable def circumscribed_cone_surface_area (s : SphereConeSystem) : ℝ :=
  3 * Real.pi * s.r^2

/-- Theorem: The ratio of surface areas is 9 : 16 : 36 -/
theorem surface_area_ratio (s : SphereConeSystem) :
  ∃ (k : ℝ), k > 0 ∧
    inscribed_cone_surface_area s = 9 * k ∧
    sphere_surface_area s = 16 * k ∧
    circumscribed_cone_surface_area s = 36 * k := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_ratio_l2_221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_specific_circles_l2_247

noncomputable def circle_overlap_area (r₁ r₂ d : ℝ) : ℝ :=
  r₁^2 * Real.arccos ((d^2 + r₁^2 - r₂^2) / (2 * d * r₁)) +
  r₂^2 * Real.arccos ((d^2 + r₂^2 - r₁^2) / (2 * d * r₂)) -
  (1/2) * Real.sqrt ((-d + r₁ + r₂) * (d + r₁ - r₂) * (d - r₁ + r₂) * (d + r₁ + r₂))

theorem overlap_area_specific_circles :
  circle_overlap_area 3 5 3 = 9 * Real.arccos (-7/18) + 25 * Real.arccos (5/6) - (1/2) * Real.sqrt 275 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overlap_area_specific_circles_l2_247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_cable_payment_l2_292

/-- The cost of cable program for James --/
noncomputable def cable_cost (first_100_cost : ℝ) (total_channels : ℕ) : ℝ :=
  if total_channels ≤ 100 then
    first_100_cost
  else
    first_100_cost + (first_100_cost / 2)

/-- James' share of the cable cost --/
noncomputable def james_payment (total_cost : ℝ) : ℝ :=
  total_cost / 2

/-- Theorem: James pays $75 for his share of the cable program --/
theorem james_cable_payment :
  james_payment (cable_cost 100 200) = 75 := by
  -- Unfold the definitions
  unfold james_payment cable_cost
  -- Simplify the if-then-else expression
  simp
  -- Perform the arithmetic
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_cable_payment_l2_292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2010_equals_4_l2_270

-- Define the sequence a_n
def a : ℕ → ℕ
  | 0 => 2  -- We define a(0) to be 2 to match a(1) in the problem
  | 1 => 3  -- This matches a(2) in the problem
  | n + 2 => (a (n + 1) * a n) % 10

-- Theorem statement
theorem a_2010_equals_4 : a 2009 = 4 := by
  sorry

-- Note: We use 2009 instead of 2010 because our sequence is 0-indexed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_2010_equals_4_l2_270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_height_weight_correlation_only_height_weight_correlation_l2_207

-- Define the concept of a pair of variables
structure VariablePair where
  var1 : String
  var2 : String

-- Define the types of relationships between variables
inductive Relationship
  | Functional
  | Correlation
  | NoRelationship

-- Function to determine the relationship between variables
def determineRelationship (pair : VariablePair) : Relationship :=
  match pair with
  | ⟨"square area", "side length"⟩ => Relationship.Functional
  | ⟨"distance", "time"⟩ => Relationship.Functional
  | ⟨"height", "weight"⟩ => Relationship.Correlation
  | ⟨"height", "eyesight"⟩ => Relationship.NoRelationship
  | _ => Relationship.NoRelationship

-- Theorem stating that only height and weight have a correlation
theorem height_weight_correlation (pairs : List VariablePair) :
  (∃ pair ∈ pairs, determineRelationship pair = Relationship.Correlation) ↔
  (∃ pair ∈ pairs, pair.var1 = "height" ∧ pair.var2 = "weight") :=
sorry

-- Define the given pairs
def givenPairs : List VariablePair := [
  ⟨"square area", "side length"⟩,
  ⟨"distance", "time"⟩,
  ⟨"height", "weight"⟩,
  ⟨"height", "eyesight"⟩
]

-- Main theorem to prove
theorem only_height_weight_correlation :
  ∃! pair : VariablePair, pair ∈ givenPairs ∧ determineRelationship pair = Relationship.Correlation :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_height_weight_correlation_only_height_weight_correlation_l2_207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_l2_274

theorem cube_root_fraction :
  Real.rpow (9 / 18.75) (1/3) = (Real.rpow 2 (2/3) * Real.rpow 3 (1/3)) / Real.rpow 5 (2/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_l2_274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l2_213

theorem line_segment_length : ∃ d : ℝ, d = 17 := by
  -- Define the coordinates of the endpoints
  let x₁ : ℝ := 1
  let y₁ : ℝ := 1
  let x₂ : ℝ := 9
  let y₂ : ℝ := 16

  -- Define the distance function
  let distance := λ (x₁ y₁ x₂ y₂ : ℝ) => Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

  -- Calculate the distance
  let d := distance x₁ y₁ x₂ y₂

  -- State and prove the theorem
  use d
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_length_l2_213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_width_calculation_l2_297

/-- The width of a rectangular sheet of paper with length 11 inches, 
    whose combined front and back area is 100 square inches less than 
    that of a 11 by 19 inch sheet, is approximately 14.45 inches. -/
theorem paper_width_calculation : 
  ∃ (w : ℝ), 
    (2 * 11 * 19 = 2 * 11 * w + 100) ∧ 
    (abs (w - 14.45) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_width_calculation_l2_297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_iff_even_l2_223

theorem function_existence_iff_even (n : ℕ) (hn : 0 < n) :
  (∃ (f g : Fin n → Fin n), ∀ i : Fin n,
    (f (g i) = i ∧ g (f i) ≠ i) ∨ (g (f i) = i ∧ f (g i) ≠ i)) ↔
  Even n := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_existence_iff_even_l2_223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_exists_l2_206

/-- A set of positive integers satisfying the given condition -/
def SpecialSet : Finset ℕ+ := sorry

/-- The cardinality of the SpecialSet is 2009 -/
axiom special_set_card : Finset.card SpecialSet = 2009

/-- The condition holds for any two elements in the SpecialSet -/
axiom special_set_condition (x y : ℕ+) (hx : x ∈ SpecialSet) (hy : y ∈ SpecialSet) :
  Int.natAbs (x.val - y.val) = Nat.gcd x.val y.val

/-- Theorem: There exists a set of 2009 different positive integers satisfying the condition -/
theorem special_set_exists : ∃ (S : Finset ℕ+), 
  Finset.card S = 2009 ∧ 
  ∀ (x y : ℕ+), x ∈ S → y ∈ S → Int.natAbs (x.val - y.val) = Nat.gcd x.val y.val :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_set_exists_l2_206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_perpendicular_to_given_line_l2_289

/-- The circle in the problem -/
def problem_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*y + 5 = 0

/-- The line perpendicular to the target line -/
def perpendicular_line (x y : ℝ) : Prop :=
  x + y + 1 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ :=
  (0, 3)

/-- The target line passing through the circle center and perpendicular to the given line -/
def target_line (x y : ℝ) : Prop :=
  x - y + 3 = 0

theorem line_through_circle_center_perpendicular_to_given_line :
  ∀ x y : ℝ, problem_circle x y ∧ perpendicular_line x y →
  ∃ a b : ℝ, (a, b) = circle_center ∧ target_line a b ∧
  (x - a) * (x - 0) + (y - b) * (y - 3) = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_circle_center_perpendicular_to_given_line_l2_289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_theorem_l2_211

/-- Given two lines in the 2D plane, this function returns true if they are symmetric with respect to a third line. -/
def are_symmetric_lines (line1 line2 symmetry_line : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), line1 x y → 
    ∃ (x' y' : ℝ), line2 x' y' ∧
      -- The midpoint of (x, y) and (x', y') lies on the symmetry line
      symmetry_line ((x + x') / 2) ((y + y') / 2) ∧
      -- The line connecting (x, y) and (x', y') is perpendicular to the symmetry line
      (y - y') + (x - x') = 0

/-- The main theorem stating that 3x + 2y = 0 is symmetric to 2x + 3y + 1 = 0 with respect to x - y - 1 = 0 -/
theorem symmetric_line_theorem :
  are_symmetric_lines 
    (λ x y => 2 * x + 3 * y + 1 = 0) 
    (λ x y => 3 * x + 2 * y = 0) 
    (λ x y => x - y - 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_line_theorem_l2_211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_bound_l2_231

/-- A type representing a line segment with a length -/
structure Segment where
  length : ℝ
  length_nonneg : length ≥ 1

/-- A function that checks if a list of segments can form a polygon -/
def can_form_polygon (segments : List Segment) : Prop :=
  ∀ (i : Fin segments.length), 
    segments[i].length < (segments.map (λ s => s.length)).sum - segments[i].length

theorem total_length_bound (segments : List Segment) 
  (h1 : segments.length = 2000)
  (h2 : ∀ s ∈ segments, s.length ≥ 1)
  (h3 : ¬ ∃ sub : List Segment, sub ⊆ segments ∧ can_form_polygon sub) :
  (segments.map (λ s => s.length)).sum ≥ 2^1999 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_length_bound_l2_231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l2_208

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

theorem equation_solutions : 
  {x : ℝ | x^2 - 8 * (floor x) + 7 = 0} = {1, Real.sqrt 33, Real.sqrt 41, 7} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solutions_l2_208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APB_is_60_degrees_l2_293

-- Define the line l: y = 2x
def line_l (x y : ℝ) : Prop := y = 2 * x

-- Define the circle C: (x - 6)^2 + (y - 2)^2 = 5
def circle_C (x y : ℝ) : Prop := (x - 6)^2 + (y - 2)^2 = 5

-- Define a point P on line l
structure Point_P where
  x : ℝ
  y : ℝ
  on_line_l : line_l x y

-- Define tangent points A and B on circle C
structure Tangent_Point where
  x : ℝ
  y : ℝ
  on_circle_C : circle_C x y

-- Define the property of symmetric tangents
def symmetric_tangents (P : Point_P) (A B : Tangent_Point) : Prop :=
  ∃ (l₁ l₂ : ℝ → ℝ), 
    (l₁ A.x = A.y ∧ l₂ B.x = B.y) ∧  -- l₁ and l₂ pass through A and B respectively
    (l₁ P.x = P.y ∧ l₂ P.x = P.y) ∧  -- l₁ and l₂ pass through P
    (∀ x y, line_l x y → (l₁ x - y) = (y - l₂ x))  -- l₁ and l₂ are symmetric w.r.t. line l

-- Define angle measure (this is a placeholder definition)
def angle_measure (A B : Tangent_Point) (P : Point_P) : ℝ := sorry

-- Theorem statement
theorem angle_APB_is_60_degrees 
  (P : Point_P) (A B : Tangent_Point) 
  (h : symmetric_tangents P A B) : 
  angle_measure A B P = 60 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_APB_is_60_degrees_l2_293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l2_202

-- Define the circle
def myCircle (R : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = R^2}

-- Define the line
def myLine (a b R : ℝ) : Set (ℝ × ℝ) := {p | a * p.1 + b * p.2 = R^2}

-- Theorem statement
theorem point_outside_circle (a b R : ℝ) (h_R : R > 0) 
  (h_intersect : Set.Nonempty (myLine a b R ∩ myCircle R)) :
  (a^2 + b^2 : ℝ) > R^2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_outside_circle_l2_202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_12_terms_l2_295

def a (n : ℕ) : ℕ :=
  if n % 2 = 1 then 2^(n-1) else 2*n - 1

def S (n : ℕ) : ℕ :=
  (List.range n).map (fun i => a (i+1)) |>.sum

theorem sum_of_first_12_terms : S 12 = 1443 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_12_terms_l2_295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l2_204

theorem pizza_toppings (total_slices : ℕ) (ham_slices : ℕ) (pineapple_slices : ℕ)
  (h_total : total_slices = 15)
  (h_ham : ham_slices = 8)
  (h_pineapple : pineapple_slices = 12)
  (h_coverage : ∀ slice, slice < total_slices → 
    (slice ∈ Finset.range ham_slices ∨ slice ∈ Finset.range pineapple_slices)) :
  (Finset.range ham_slices ∩ Finset.range pineapple_slices).card = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_toppings_l2_204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_fixed_point_line_l2_298

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define a point inside the parabola
def point_inside_parabola (p m n : ℝ) : Prop := 
  parabola p m (n^2/(2*p)) ∧ m > 0

-- Define the slopes of the lines
def slopes (k1 k2 : ℝ) : Prop := k1 ≠ 0 ∧ k2 ≠ 0

-- Define the theorem for the minimum area
theorem min_area_triangle (p m : ℝ) (k1 k2 : ℝ) :
  parabola p m 0 →
  point_inside_parabola p m 0 →
  slopes k1 k2 →
  k1 * k2 = -1 →
  ∃ (area : ℝ), area = p^2 ∧ 
    ∀ (other_area : ℝ), other_area ≥ area := by sorry

-- Define the theorem for the fixed point
theorem fixed_point_line (p m n lambda : ℝ) (k1 k2 : ℝ) :
  parabola p m (n^2/(2*p)) →
  point_inside_parabola p m n →
  slopes k1 k2 →
  k1 + k2 = lambda →
  lambda ≠ 0 →
  ∃ (x y : ℝ), x = m - n/lambda ∧ y = p/lambda ∧
    (∀ (x' y' : ℝ), x' = m - n/lambda ∧ y' = p/lambda → 
      ∃ (t : ℝ), x' = t*x ∧ y' = t*y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_triangle_fixed_point_line_l2_298
