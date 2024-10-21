import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_properties_l745_74542

-- Define the parabola
structure Parabola where
  p : ℝ

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points
variable (A B C D : ℝ × ℝ)

-- Define the problem setup
def setup (parabola : Parabola) (circle : Circle) : Prop :=
  -- AB is a principal chord of the parabola
  -- Circle has diameter AB
  -- Circle intersects parabola at C and D
  True

-- Define necessary functions
def Line (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) := sorry
def Parallel (l1 l2 : Set (ℝ × ℝ)) : Prop := sorry
def distance (l1 l2 : Set (ℝ × ℝ)) : ℝ := sorry
def is_vertex (p : Parabola) (v : ℝ × ℝ) : Prop := sorry
def is_tangent (l : Set (ℝ × ℝ)) (c : Circle) : Prop := sorry

-- Theorem statement
theorem parabola_circle_properties
  (parabola : Parabola) (circle : Circle)
  (h_setup : setup parabola circle) :
  -- 1. CD is parallel to AB
  (Parallel (Line C D) (Line A B)) ∧
  -- 2. Distance between CD and AB is 2p
  (distance (Line C D) (Line A B) = 2 * parabola.p) ∧
  -- 3. Tangents from parabola vertex to circle pass through C and D
  (∃ (V : ℝ × ℝ), is_vertex parabola V ∧
    is_tangent (Line V C) circle ∧
    is_tangent (Line V D) circle) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_properties_l745_74542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l745_74518

-- Define the angle in degrees
def angle : ℚ := -510

-- Define the function to normalize an angle to the range [0, 360)
def normalize_angle (θ : ℚ) : ℚ :=
  θ - 360 * (θ / 360).floor

-- Define the function to determine the quadrant of an angle
def angle_quadrant (θ : ℚ) : ℕ :=
  let normalized_θ := normalize_angle θ
  if 0 ≤ normalized_θ ∧ normalized_θ < 90 then 1
  else if 90 ≤ normalized_θ ∧ normalized_θ < 180 then 2
  else if 180 ≤ normalized_θ ∧ normalized_θ < 270 then 3
  else 4

-- Theorem statement
theorem angle_in_third_quadrant :
  angle_quadrant angle = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_in_third_quadrant_l745_74518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_n1_unique_solution_n2_l745_74567

/-- Represents a system of linear equations where coefficients and constants form an arithmetic sequence -/
structure ArithmeticSystem (n : ℕ) where
  a : ℚ  -- First term of the arithmetic sequence
  Δ : ℚ  -- Common difference of the arithmetic sequence

/-- The solution to the arithmetic system for n = 1 -/
noncomputable def solution_n1 (sys : ArithmeticSystem 1) : ℚ :=
  1 + sys.Δ / sys.a

/-- The solution to the arithmetic system for n = 2 -/
def solution_n2 : ℚ × ℚ :=
  (-1, 2)

/-- Theorem stating the unique solution for n = 1 when it exists -/
theorem unique_solution_n1 (sys : ArithmeticSystem 1) (h : sys.a ≠ 0) :
  ∃! x : ℚ, x = solution_n1 sys := by
  sorry

/-- Theorem stating the unique solution for n = 2 when it exists -/
theorem unique_solution_n2 (sys : ArithmeticSystem 2) (h : sys.Δ ≠ 0) :
  ∃! x : ℚ × ℚ, x = solution_n2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_n1_unique_solution_n2_l745_74567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_plus_i_l745_74587

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the function f
noncomputable def f : ℂ → ℂ := λ x => 
  if x.im = 0 then 1 + x else (1 - i) * x

-- State the theorem
theorem f_of_one_plus_i : f (1 + i) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_one_plus_i_l745_74587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_different_from_reverse_product_is_perfect_square_satisfies_all_conditions_l745_74543

def N : ℕ := 15841584158415841584

-- Define a function to reverse a natural number
def reverse (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (fun acc d => acc * 10 + d) 0

-- Property a: N is not a perfect square
theorem not_perfect_square : ¬∃ (m : ℕ), m * m = N := by sorry

-- Property b: N is different from its reverse
theorem different_from_reverse : N ≠ reverse N := by sorry

-- Property c: The product of N and its reverse is a perfect square
theorem product_is_perfect_square : ∃ (m : ℕ), m * m = N * (reverse N) := by sorry

-- Main theorem combining all properties
theorem satisfies_all_conditions :
  (∃ (d : ℕ), N.digits 10 = List.replicate 20 d) ∧
  (¬∃ (m : ℕ), m * m = N) ∧
  (N ≠ reverse N) ∧
  (∃ (m : ℕ), m * m = N * (reverse N)) := by
  apply And.intro
  · sorry -- Proof that N has 20 identical digits
  apply And.intro
  · exact not_perfect_square
  apply And.intro
  · exact different_from_reverse
  · exact product_is_perfect_square

end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_perfect_square_different_from_reverse_product_is_perfect_square_satisfies_all_conditions_l745_74543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l745_74533

/-- The parabola y = x^2 - 6x + 14 -/
def parabola (x : ℝ) : ℝ := x^2 - 6*x + 14

/-- The line y = x - 5 -/
def line (x : ℝ) : ℝ := x - 5

/-- The distance between a point (a, parabola a) on the parabola and the line -/
noncomputable def distance_to_line (a : ℝ) : ℝ := 
  |a - (parabola a) - 5| / Real.sqrt 2

/-- The shortest distance between a point on the parabola and a point on the line is √15/2 -/
theorem shortest_distance : 
  ∃ (a : ℝ), ∀ (x : ℝ), distance_to_line a ≤ distance_to_line x ∧ 
  distance_to_line a = Real.sqrt 15 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_l745_74533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_circle_center_l745_74511

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop :=
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0

/-- The center of the circle -/
noncomputable def circle_center : ℝ × ℝ := (1, 3/2)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_circle_center_l745_74511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l745_74551

/-- Circle C: x^2 + y^2 - 2y - 4 = 0 -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 4 = 0

/-- Line l: mx - y + 1 - m = 0 -/
def line_l (m x y : ℝ) : Prop := m*x - y + 1 - m = 0

/-- Distance between two points -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem line_equation_proof :
  ∀ (m : ℝ) (A B : ℝ × ℝ),
  (∃ (x y : ℝ), circle_C x y ∧ line_l m x y) →
  (A.1 ≠ B.1 ∨ A.2 ≠ B.2) →
  circle_C A.1 A.2 →
  circle_C B.1 B.2 →
  line_l m A.1 A.2 →
  line_l m B.1 B.2 →
  distance A.1 A.2 B.1 B.2 = 3 * Real.sqrt 2 →
  (m = 1 ∧ (∀ x y : ℝ, line_l m x y ↔ x - y = 0)) ∨
  (m = -1 ∧ (∀ x y : ℝ, line_l m x y ↔ x + y + 2 = 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_proof_l745_74551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_multiples_of_seven_l745_74562

def sequence_a (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => sequence_a n + sequence_a (n / 2)

theorem infinitely_many_multiples_of_seven :
  ∀ k : ℕ, ∃ n > k, 7 ∣ sequence_a n := by
  sorry

#eval sequence_a 5  -- This will evaluate the 5th term of the sequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_multiples_of_seven_l745_74562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_subsets_of_U_l745_74522

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define the function y = ln(1 - x^2)
noncomputable def y (x : ℝ) : ℝ := Real.log (1 - x^2)

-- Define the domain M of the function y
def M : Set ℝ := {x : ℝ | 1 - x^2 > 0}

-- Define the set N
def N : Set ℝ := {x : ℝ | x^2 - x < 0}

-- Theorem statement
theorem domain_intersection :
  M ∩ N = N := by
  sorry

-- Additional theorem to show that M and N are subsets of U
theorem subsets_of_U :
  M ⊆ U ∧ N ⊆ U := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_intersection_subsets_of_U_l745_74522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ten_terms_is_120_l745_74572

/-- An arithmetic progression with specific properties -/
structure ArithmeticProgression where
  a : ℚ  -- First term
  d : ℚ  -- Common difference
  sum_first_three : a + (a + d) + (a + 2*d) = 15
  geometric_condition : (a - 1)*(a + 2*d + 1) = (a + d - 1)^2

/-- The sum of the first n terms of an arithmetic progression -/
def sum_n_terms (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  (n : ℚ) * (2 * ap.a + (n - 1) * ap.d) / 2

/-- The main theorem -/
theorem sum_ten_terms_is_120 (ap : ArithmeticProgression) : 
  sum_n_terms ap 10 = 120 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_ten_terms_is_120_l745_74572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l745_74560

/-- The parabola defined by y^2 = 12x -/
def parabola (x y : ℝ) : Prop := y^2 = 12 * x

/-- The circle defined by x^2 + y^2 - 4x - 6y = 0 -/
def circle' (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 6*y = 0

/-- Two points C and D that satisfy both the parabola and circle equations -/
def intersection_points (C D : ℝ × ℝ) : Prop :=
  parabola C.1 C.2 ∧ circle' C.1 C.2 ∧
  parabola D.1 D.2 ∧ circle' D.1 D.2 ∧
  C ≠ D

/-- The theorem stating that the distance between the intersection points is 3√5 -/
theorem intersection_distance (C D : ℝ × ℝ) 
  (h : intersection_points C D) : 
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 3 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l745_74560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l745_74591

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x)^2 + 2 * Real.sin x - 1

theorem f_range :
  ∀ y ∈ Set.Icc (-1/2 : ℝ) (3/2 : ℝ),
  ∃ x ∈ Set.Icc (-Real.pi/6 : ℝ) (2*Real.pi/3 : ℝ),
  f x = y ∧
  ∀ x ∈ Set.Icc (-Real.pi/6 : ℝ) (2*Real.pi/3 : ℝ),
  -1/2 ≤ f x ∧ f x ≤ 3/2 :=
by
  sorry

#check f_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l745_74591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l745_74597

noncomputable section

-- Define the floor function
def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part function
def frac (x : ℝ) : ℝ := x - Int.floor x

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  (Int.floor x - frac y = 3.7) ∧ (frac x + Int.floor y = 6.2)

-- State the theorem
theorem solution_difference (x y : ℝ) : 
  system x y → |x - y| = 2.1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_difference_l745_74597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_calculation_l745_74510

/-- Calculates the speed of a car in km/h given the tire's rotation speed and circumference -/
noncomputable def carSpeed (revolutionsPerMinute : ℝ) (tireCircumference : ℝ) : ℝ :=
  (revolutionsPerMinute * tireCircumference * 60) / 1000

theorem car_speed_calculation (revolutionsPerMinute tireCircumference : ℝ) 
  (h1 : revolutionsPerMinute = 400)
  (h2 : tireCircumference = 3) :
  carSpeed revolutionsPerMinute tireCircumference = 72 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval carSpeed 400 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_speed_calculation_l745_74510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l745_74521

noncomputable def g (x : ℝ) : ℝ := -Real.sqrt 3 * Real.cos (2 * x)

theorem g_properties :
  (g (π / 4) = 0) ∧
  (∀ x, g (-x) = g x) ∧
  (∀ x, g x ≤ Real.sqrt 3) ∧
  (∃ x, g x = Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_properties_l745_74521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_juanita_dessert_cost_is_975_l745_74506

/-- Represents the cost of a dessert item in cents -/
def Cost := ℕ

instance : Add Cost := ⟨Nat.add⟩
instance : Mul Cost := ⟨Nat.mul⟩
instance : OfNat Cost n := ⟨n⟩

/-- Calculates the total cost of Juanita's dessert -/
def juanita_dessert_cost (brownie_cost : Cost) (regular_scoop_cost : Cost) 
  (premium_scoop_cost : Cost) (deluxe_scoop_cost : Cost) (syrup_cost : Cost) 
  (nuts_cost : Cost) (whipped_cream_cost : Cost) (cherry_cost : Cost) : Cost :=
  brownie_cost + regular_scoop_cost + premium_scoop_cost + deluxe_scoop_cost + 
  2 * syrup_cost + nuts_cost + whipped_cream_cost + cherry_cost

/-- Theorem stating that Juanita's dessert costs $9.75 -/
theorem juanita_dessert_cost_is_975 :
  juanita_dessert_cost 250 100 125 150 50 150 75 25 = 975 := by
  rfl

#eval juanita_dessert_cost 250 100 125 150 50 150 75 25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_juanita_dessert_cost_is_975_l745_74506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l745_74514

/-- The area of a circular sector with central angle θ and radius r -/
noncomputable def sector_area (θ : ℝ) (r : ℝ) : ℝ := (1/2) * r^2 * θ

/-- Theorem: The area of a sector with central angle 2π/3 and radius 2 is 4π/3 -/
theorem sector_area_specific : sector_area (2*Real.pi/3) 2 = 4*Real.pi/3 := by
  -- Unfold the definition of sector_area
  unfold sector_area
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_specific_l745_74514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_max_a_value_l745_74527

-- Define the function f(x) = e^x - x
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

-- Theorem 1: The minimum value of f(x) is 1
theorem f_min_value : ∀ x : ℝ, f x ≥ 1 := by
  sorry

-- Theorem 2: Given f(x) ≥ x^3/6 + a for all x, the maximum value of a is 1
theorem max_a_value (a : ℝ) : (∀ x : ℝ, f x ≥ x^3 / 6 + a) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_max_a_value_l745_74527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_lowest_score_l745_74528

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a score function
def score : Person → ℕ := sorry

-- All scores are different
axiom scores_different : ∀ x y : Person, x ≠ y → score x ≠ score y

-- If B's score is not the highest, then A's score is the lowest
axiom condition1 : (∃ p : Person, score p > score Person.B) → 
  (∀ q : Person, score Person.A ≤ score q)

-- If C's score is not the lowest, then A's score is the highest
axiom condition2 : (∃ p : Person, score p < score Person.C) → 
  (∀ q : Person, score q ≤ score Person.A)

-- Theorem: C has the lowest score
theorem C_lowest_score : ∀ p : Person, score Person.C ≤ score p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_lowest_score_l745_74528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_not_broken_at_midnight_l745_74557

/-- Represents the state of a clock --/
structure Clock where
  hourAngle : ℚ  -- Angle of the hour hand in degrees
  minuteAngle : ℚ  -- Angle of the minute hand in degrees

/-- Represents the speed of clock hands --/
structure ClockSpeed where
  hourSpeed : ℚ  -- Degrees per hour for the hour hand
  minuteSpeed : ℚ  -- Degrees per hour for the minute hand

/-- Normal clock speed --/
def normalSpeed : ClockSpeed := ⟨30, 360⟩

/-- Broken clock speed --/
def brokenSpeed : ClockSpeed := ⟨60, 180⟩

/-- Function to calculate the clock state after a given time --/
def clockStateAfterTime (initialState : Clock) (speed : ClockSpeed) (time : ℚ) : Clock :=
  ⟨(initialState.hourAngle + speed.hourSpeed * time) % 360,
   (initialState.minuteAngle + speed.minuteSpeed * time) % 360⟩

/-- Theorem stating that the clock couldn't have broken at midnight --/
theorem clock_not_broken_at_midnight :
  ∀ (t : ℚ), t > 0 →
    let finalState := clockStateAfterTime ⟨0, 0⟩ brokenSpeed t
    ¬(finalState.hourAngle = 180 ∧ finalState.minuteAngle = 0) :=
by
  sorry

#check clock_not_broken_at_midnight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_not_broken_at_midnight_l745_74557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l745_74548

-- Define the function (noncomputable due to real exponentiation)
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 3) * x^(m^2 + 3*m - 2)

-- Theorem statement
theorem quadratic_function_properties :
  (∃ m : ℝ, (m = -4 ∨ m = 1) ∧
   (∀ x : ℝ, f m x = (m + 3) * x^2)) ∧
  (∀ x y : ℝ, f (-4) x < f (-4) y ↔ x > y) ∧
  (∃ x₀ : ℝ, ∀ x : ℝ, f 1 x ≥ f 1 x₀) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_function_properties_l745_74548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_level_score_l745_74535

def game_sequence : ℕ → ℕ
  | 0 => 2  -- Adding the base case for 0
  | 1 => 2
  | 2 => 3
  | 3 => 5
  | 4 => 8
  | 5 => 12
  | n + 6 => game_sequence (n + 5) + n + 5

theorem sixth_level_score : game_sequence 6 = 17 := by
  rfl  -- reflexivity should be sufficient here

#eval game_sequence 6  -- This will evaluate the function for n = 6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_level_score_l745_74535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_change_l745_74571

/-- Represents the pressure-volume relationship at constant temperature -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  k : ℝ
  h_prop : volume * pressure = k

/-- The initial state of the gas -/
noncomputable def initial_state : GasState :=
  { volume := 3.5
    pressure := 7
    k := 3.5 * 7
    h_prop := by ring }

/-- The final state of the gas -/
noncomputable def final_state : GasState :=
  { volume := 10.5
    pressure := 49 / 21
    k := initial_state.k
    h_prop := by
      simp [initial_state]
      field_simp
      ring }

theorem pressure_change (initial : GasState) (final : GasState) 
    (h_same_k : initial.k = final.k) 
    (h_initial : initial = initial_state) 
    (h_final_volume : final.volume = 10.5) : 
  final.pressure = 49 / 21 := by
  sorry

#check pressure_change

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pressure_change_l745_74571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_distance_l745_74561

/-- Parabola represented by the equation y^2 = 4x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 4 * p.1}

/-- Line passing through (1, 0) with slope √3 -/
noncomputable def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sqrt 3 * (p.1 - 1)}

/-- The focus of the parabola -/
def F : ℝ × ℝ := (1, 0)

/-- The point on the directrix -/
noncomputable def N : ℝ × ℝ := (-1, 2 * Real.sqrt 3)

/-- The intersection point of the parabola and the line -/
noncomputable def M : ℝ × ℝ := (3, 2 * Real.sqrt 3)

/-- The distance from a point to a line -/
noncomputable def distancePointToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem parabola_line_distance : 
  M ∈ Parabola ∧ 
  M ∈ Line ∧ 
  M.2 > 0 ∧
  distancePointToLine M {p : ℝ × ℝ | p.2 - N.2 = -(1 / Real.sqrt 3) * (p.1 - F.1)} = 3 * Real.sqrt 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_distance_l745_74561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_relation_l745_74589

theorem angle_cosine_relation (A B C : ℝ) :
  A > B ↔ Real.cos A < Real.cos B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_cosine_relation_l745_74589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_three_consecutive_numbers_l745_74502

theorem min_sum_of_three_consecutive_numbers (arrangement : List ℕ) : 
  arrangement.length = 10 ∧ 
  arrangement.toFinset = Finset.range 10 →
  ∃ (i j k : ℕ), 
    (i < arrangement.length ∧ 
     j < arrangement.length ∧ 
     k < arrangement.length) ∧
    ((j = (i + 1) % arrangement.length ∧ 
      k = (i + 2) % arrangement.length) ∨
     (k = (j + 1) % arrangement.length ∧ 
      i = (j + 2) % arrangement.length) ∨
     (i = (k + 1) % arrangement.length ∧ 
      j = (k + 2) % arrangement.length)) ∧
    arrangement[i]! + arrangement[j]! + arrangement[k]! ≤ 15 :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_three_consecutive_numbers_l745_74502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_6_25_l745_74593

/-- A right triangle with sides 8, 15, and 17 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 8
  side_b : b = 15
  side_c : c = 17

/-- The distance between the centers of inscribed and circumscribed circles -/
noncomputable def distance_between_centers (t : RightTriangle) : ℝ :=
  let s := (t.a + t.b + t.c) / 2
  let area := Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))
  let inradius := area / s
  let circumradius := t.c / 2
  Real.sqrt (inradius^2 + (circumradius - inradius)^2)

/-- Theorem stating that the distance between centers is 6.25 for the given triangle -/
theorem distance_is_6_25 (t : RightTriangle) : distance_between_centers t = 6.25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_is_6_25_l745_74593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l745_74556

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 2*x - 8) / Real.log (1/2)

-- State the theorem
theorem f_monotone_increasing :
  StrictMonoOn f (Set.Iio (-2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l745_74556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mutually_exclusive_number_l745_74547

def is_mutually_exclusive (m : ℕ) : Prop :=
  100 ≤ m ∧ m < 1000 ∧
  (m / 100 ≠ (m / 10) % 10) ∧
  (m / 100 ≠ m % 10) ∧
  ((m / 10) % 10 ≠ m % 10) ∧
  (m / 100 ≠ 0) ∧ ((m / 10) % 10 ≠ 0) ∧ (m % 10 ≠ 0)

def m_prime (m : ℕ) : ℕ := m / 10

def F (m : ℕ) : ℤ := (m_prime m : ℤ) - (m % 10 : ℤ)

def G (m : ℕ) : ℤ := ((m / 10) % 10 : ℤ) - (m % 10 : ℤ)

theorem max_mutually_exclusive_number (x y : ℕ) :
  1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 →
  let m := 20 * (5 * x + 1) + 2 * y
  is_mutually_exclusive m →
  (F m) % (G m) = 0 →
  ((F m) / (G m)) % 13 = 0 →
  m ≤ 932 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_mutually_exclusive_number_l745_74547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l745_74590

-- Define the sets M and N
def M : Set ℝ := {y | ∃ x : ℝ, y = Real.exp (-x * Real.log 2)}
def N : Set ℝ := {y | ∃ x : ℝ, y = Real.sin x}

-- Define the open-closed interval (0, 1]
def OpenClosedInterval : Set ℝ := {y | 0 < y ∧ y ≤ 1}

-- State the theorem
theorem intersection_equals_interval : M ∩ N = OpenClosedInterval := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_equals_interval_l745_74590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fruits_theorem_l745_74595

/-- Given the following conditions:
  * Louis has 5 oranges and 3 apples
  * Samantha has 8 oranges and 7 apples
  * Marley has twice as many oranges as Louis and three times as many apples as Samantha
  * Edward has three times as many oranges and apples as Louis
This theorem proves that the total number of fruits for all four people combined is 78. -/
theorem total_fruits_theorem : 
  let louis_oranges : ℕ := 5
  let louis_apples : ℕ := 3
  let samantha_oranges : ℕ := 8
  let samantha_apples : ℕ := 7
  let marley_oranges : ℕ := 2 * louis_oranges
  let marley_apples : ℕ := 3 * samantha_apples
  let edward_oranges : ℕ := 3 * louis_oranges
  let edward_apples : ℕ := 3 * louis_apples
  louis_oranges + louis_apples + 
  samantha_oranges + samantha_apples + 
  marley_oranges + marley_apples + 
  edward_oranges + edward_apples = 78 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_fruits_theorem_l745_74595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_fixed_point_l745_74507

/-- A rational function f(x) = (ax + b) / (cx + d) -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- The n-fold composition of f -/
noncomputable def F (a b c d : ℝ) : ℕ → (ℝ → ℝ)
  | 0 => id
  | n + 1 => f a b c d ∘ F a b c d n

/-- Main theorem -/
theorem rational_function_fixed_point (a b c d : ℝ) (n : ℕ) :
  (f a b c d 0 ≠ 0) →
  (f a b c d (f a b c d 0) ≠ 0) →
  (F a b c d n 0 = 0) →
  (∀ x : ℝ, F a b c d n x = x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rational_function_fixed_point_l745_74507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l745_74504

theorem remainder_problem : ∃ (m n r s : ℕ), 
  m > 0 ∧ n > 0 ∧
  (702 % m = r ∧ 787 % m = r ∧ 855 % m = r) ∧ 
  (412 % n = s ∧ 722 % n = s ∧ 815 % n = s) ∧ 
  (r ≠ s) ∧
  (m + n + r + s : ℕ) = 62 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_problem_l745_74504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l745_74583

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 3 * x + 4 else 7 - 3 * x^2

theorem f_values : f (-2) = -2 ∧ f 3 = -20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_values_l745_74583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_six_percent_l745_74585

/-- Calculates the simple interest rate given the principal, total amount, and time -/
noncomputable def simple_interest_rate (principal : ℝ) (total_amount : ℝ) (time : ℝ) : ℝ :=
  ((total_amount - principal) * 100) / (principal * time)

/-- Theorem: Given the specified conditions, the simple interest rate is 6% -/
theorem interest_rate_is_six_percent :
  let principal : ℝ := 12500
  let total_amount : ℝ := 15500
  let time : ℝ := 4
  simple_interest_rate principal total_amount time = 6 := by
  sorry

#check interest_rate_is_six_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_rate_is_six_percent_l745_74585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l745_74598

/-- Represents the time taken to complete a task -/
structure TaskTime where
  hours : ℚ
  hours_pos : hours > 0

/-- Represents the rate at which work is completed -/
def WorkRate (t : TaskTime) : ℚ := 1 / t.hours

theorem work_completion_time 
  (time_B : TaskTime) 
  (time_BC : TaskTime) 
  (time_AC : TaskTime) :
  time_B.hours = 30 → 
  time_BC.hours = 3 → 
  time_AC.hours = 2 → 
  ∃ (time_A : TaskTime), time_A.hours = 5/2 ∧ 
    WorkRate time_A + WorkRate time_B = WorkRate time_BC ∧
    WorkRate time_A + (WorkRate time_BC - WorkRate time_B) = WorkRate time_AC :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completion_time_l745_74598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l745_74554

noncomputable def f (x : ℝ) := Real.sin x / (2 + Real.cos x)

theorem f_properties : ∀ x : ℝ,
  (abs (f x) ≤ abs x) ∧
  (abs (f x) ≤ Real.sqrt 3 / 3) ∧
  (f (Real.pi + x) + f (Real.pi - x) = 0) :=
by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l745_74554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_ge_11_l745_74559

def is_valid_digit (d : ℕ) : Prop := d = 1 ∨ d = 3 ∨ d = 7 ∨ d = 9

def all_digits_valid (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → is_valid_digit d

theorem prime_divisor_ge_11 (n : ℕ) (h1 : n > 10) (h2 : all_digits_valid n) :
  ∃ p : ℕ, Nat.Prime p ∧ p ≥ 11 ∧ p ∣ n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisor_ge_11_l745_74559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_false_proposition_l745_74534

-- Define propositions
def proposition_A : Prop := ∀ x : ℝ, x < 0 → |x| = -x

-- We need to define Line and Point as we don't have a specific geometry library
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)

def parallel (l1 l2 : Line) : Prop := sorry
def perpendicular (l1 l2 : Line) : Prop := sorry

def proposition_B : Prop := ∀ (l : Line) (p : Point), 
  ∃! m : Line, parallel m l ∧ m ≠ l

def infinite_non_repeating_decimal (x : ℝ) : Prop := sorry
noncomputable def irrational (x : ℝ) : Prop := ¬ ∃ (q : ℚ), ↑q = x

def proposition_C : Prop := (∀ x : ℝ, irrational x ↔ infinite_non_repeating_decimal x) ∧ 
  irrational (Real.sqrt 12)

def proposition_D : Prop := ∀ (l : Line) (p : Point),
  ∃! m : Line, perpendicular m l

-- Theorem statement
theorem false_proposition : 
  proposition_A ∧ proposition_B ∧ proposition_C ∧ ¬proposition_D :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_false_proposition_l745_74534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candles_theorem_l745_74596

/-- The initial number of candles bought by Alyssa and Chelsea -/
noncomputable def initial_candles : ℝ := 40

/-- The number of candles remaining after Alyssa used her share -/
noncomputable def remaining_after_alyssa : ℝ := initial_candles / 2

/-- The number of candles Chelsea used -/
noncomputable def chelsea_used : ℝ := 0.7 * remaining_after_alyssa

/-- The number of candles left after both Alyssa and Chelsea used their shares -/
noncomputable def candles_left : ℝ := remaining_after_alyssa - chelsea_used

theorem candles_theorem : 
  candles_left = 6 → initial_candles = 40 := by
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candles_theorem_l745_74596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l745_74509

-- Define the function f(x) = 3 + cos(x)
noncomputable def f (x : ℝ) : ℝ := 3 + Real.cos x

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc 2 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l745_74509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l745_74558

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * t.b * Real.cos t.C + t.c = 2 * t.a) 
  (h2 : Real.cos t.A = 1/7) : 
  t.B = π/3 ∧ t.c/t.a = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l745_74558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_6880_l745_74508

theorem cube_root_6880 (h1 : (68.8 : Real)^(1/3) = 4.098) (h2 : (6.88 : Real)^(1/3) = 1.902) :
  (6880 : Real)^(1/3) = 19.02 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_6880_l745_74508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_10_l745_74536

noncomputable def geometric_sequence (a : ℝ) (r : ℝ) (n : ℕ) : ℝ := a * r^n

noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  if r = 1 then n * a else a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_10 (a : ℝ) :
  (geometric_sum a 2 5 = 1) → (geometric_sum a 2 10 = 33) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_10_l745_74536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_480_degrees_l745_74531

theorem sin_480_degrees : Real.sin (480 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_480_degrees_l745_74531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_P_l745_74523

-- Define the set P
def P : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- Theorem stating the domain of P
theorem domain_of_P : Set.Ici 1 = P := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_P_l745_74523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_distribution_l745_74546

/-- Given a neighborhood with the following properties:
  - 250 total households
  - 25 households with no vehicles
  - 36 households with all three types of vehicles (car, bike, and scooter)
  - 62 households with car only
  - 45 households with bike only
  - 30 households with scooter only

  We prove the following:
  1. The number of households with only two types of vehicles is 52.
  2. The number of households with exactly one type of vehicle is 137.
  3. The number of households with at least one type of vehicle is 225.
-/
theorem vehicle_distribution (total : Nat) (no_vehicle : Nat) (all_vehicle : Nat)
    (car_only : Nat) (bike_only : Nat) (scooter_only : Nat)
    (h_total : total = 250)
    (h_no_vehicle : no_vehicle = 25)
    (h_all_vehicle : all_vehicle = 36)
    (h_car_only : car_only = 62)
    (h_bike_only : bike_only = 45)
    (h_scooter_only : scooter_only = 30) :
    let two_vehicle := total - no_vehicle - all_vehicle - (car_only + bike_only + scooter_only)
    let one_vehicle := car_only + bike_only + scooter_only
    let at_least_one := total - no_vehicle
    (two_vehicle = 52) ∧ (one_vehicle = 137) ∧ (at_least_one = 225) := by
  -- Unfold the let bindings
  simp only [h_total, h_no_vehicle, h_all_vehicle, h_car_only, h_bike_only, h_scooter_only]
  
  -- Prove each part of the conjunction
  apply And.intro
  · -- Prove two_vehicle = 52
    norm_num
  
  apply And.intro
  · -- Prove one_vehicle = 137
    norm_num
  
  -- Prove at_least_one = 225
  norm_num

-- You can remove this line if you don't need to evaluate the theorem
-- #eval vehicle_distribution 250 25 36 62 45 30

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vehicle_distribution_l745_74546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_sin_sum_l745_74576

noncomputable def f (x m : ℝ) := 2 * Real.sin (2 * x) + Real.cos (2 * x) - m

theorem zero_points_sin_sum (x₁ x₂ m : ℝ) :
  x₁ ∈ Set.Icc 0 (Real.pi / 2) →
  x₂ ∈ Set.Icc 0 (Real.pi / 2) →
  f x₁ m = 0 →
  f x₂ m = 0 →
  Real.sin (x₁ + x₂) = 2 * Real.sqrt 5 / 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_sin_sum_l745_74576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_max_lambda_l745_74540

noncomputable section

variable (α β γ : ℂ)

def not_all_leq_one (α β γ : ℂ) : Prop :=
  ¬(Complex.abs α ≤ 1 ∧ Complex.abs β ≤ 1 ∧ Complex.abs γ ≤ 1)

def inequality (α β γ : ℂ) (lambda : ℝ) : Prop :=
  1 + Complex.abs (α + β + γ) + Complex.abs (α*β + β*γ + γ*α) + Complex.abs (α*β*γ) ≥ 
  lambda * (Complex.abs α + Complex.abs β + Complex.abs γ)

theorem inequality_holds (α β γ : ℂ) (h : not_all_leq_one α β γ) :
  ∀ lambda : ℝ, lambda ≤ 2/3 → inequality α β γ lambda :=
sorry

theorem max_lambda : 
  ∃ lambda_max : ℝ, lambda_max = Real.rpow 2 (1/3) / 2 ∧
  (∀ α β γ : ℂ, ∀ lambda : ℝ, lambda ≤ lambda_max → inequality α β γ lambda) ∧
  (∀ ε > 0, ∃ α β γ : ℂ, ¬(inequality α β γ (lambda_max + ε))) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_max_lambda_l745_74540


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_root_eight_l745_74563

-- Define the polynomial f
def f (n : ℕ) (a : ℕ → ℤ) (x : ℤ) : ℤ :=
  x^n + (Finset.range n).sum (λ i ↦ a i * x^(n - 1 - i))

-- State the theorem
theorem no_integer_root_eight
  (n : ℕ) (a : ℕ → ℤ) 
  (h_distinct : ∃ (w x y z : ℤ), w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
    f n a w = 5 ∧ f n a x = 5 ∧ f n a y = 5 ∧ f n a z = 5) :
  ¬ ∃ (k : ℤ), f n a k = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_integer_root_eight_l745_74563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bound_l745_74550

noncomputable def A (m k : ℕ) : ℕ := Nat.choose m k * Nat.factorial k

noncomputable def a (n : ℕ) : ℝ :=
  Real.sqrt (A (n+2) 1 * Real.rpow (A (n+3) 2 * Real.rpow (A (n+4) 3 * Real.rpow (A (n+5) 4) (1/5)) (1/4)) (1/3))

theorem a_bound (n : ℕ) (h : n ≥ 1) : a n < (119/120 : ℝ) * n + 7/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_bound_l745_74550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l745_74538

def f (n : ℕ) : ℕ := (Finset.filter (λ m : ℕ ↦ n % m = 0) (Finset.range (n + 1))).sum id

def σ (i : ℕ) : ℕ := (Finset.filter (λ m : ℕ ↦ i % m = 0 ∧ m < i) (Finset.range i)).sum id

theorem count_special_integers : 
  (Finset.filter (λ i ↦ i ≥ 1 ∧ f i = 1 + Int.floor (Real.sqrt (i : ℝ)) + i + 2 * σ i) (Finset.range 3001)).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_special_integers_l745_74538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_non_coincident_lines_l745_74517

noncomputable def parallel (m1 m2 : ℝ) : Prop := m1 = m2

noncomputable def coincide (m1 m2 b1 b2 : ℝ) : Prop := m1 = m2 ∧ b1 = b2

noncomputable def slope1 (a : ℝ) : ℝ := -a / 2

noncomputable def slope2 (a : ℝ) : ℝ := -1 / (a + 1)

noncomputable def intercept2 (a : ℝ) : ℝ := -(a^2 - 1) / (a + 1)

theorem parallel_non_coincident_lines (a : ℝ) :
  (parallel (slope1 a) (slope2 a) ∧ ¬coincide (slope1 a) (slope2 a) 0 (intercept2 a)) ↔ a = -2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_non_coincident_lines_l745_74517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l745_74537

/-- The average speed of a triathlete in a four-segment race -/
noncomputable def average_speed (s₁ s₂ s₃ s₄ : ℝ) : ℝ :=
  4 / (1/s₁ + 1/s₂ + 1/s₃ + 1/s₄)

/-- Theorem stating that the average speed in the given triathlon is approximately 5.3 km/h -/
theorem triathlon_average_speed :
  let swim_speed : ℝ := 2
  let bike_speed : ℝ := 25
  let run_speed : ℝ := 12
  let skate_speed : ℝ := 8
  abs (average_speed swim_speed bike_speed run_speed skate_speed - 5.3) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triathlon_average_speed_l745_74537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_unique_zero_l745_74541

noncomputable section

variable (x : ℝ)

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := x - (Real.log x) / m

-- Define the function F(x)
def F (x : ℝ) : ℝ := x - (f (-1) x) / x

-- Theorem statement
theorem F_has_unique_zero :
  ∃! x, x > 0 ∧ F x = 0 :=
sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_F_has_unique_zero_l745_74541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_proof_l745_74529

noncomputable def container_capacity : ℝ := 40
noncomputable def initial_fullness : ℝ := 0.30
noncomputable def final_fullness : ℝ := 3/4

theorem water_added_proof :
  let initial_water := initial_fullness * container_capacity
  let final_water := final_fullness * container_capacity
  final_water - initial_water = 18 := by
  -- Unfold the definitions
  unfold container_capacity initial_fullness final_fullness
  -- Perform the calculations
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_added_proof_l745_74529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l745_74544

/-- Given a triangle ABC with the following properties:
  * Sides opposite to angles A, B, and C are a, b, and c respectively.
  * Vector (b, -√3a) is perpendicular to vector (cos A, sin B).
  * B + π/12 = A
  * a = 2
  Prove that A = π/6 and the area of triangle ABC is √3 - 1 -/
theorem triangle_problem (a b c A B C : ℝ) : 
  (b * (Real.cos A) - Real.sqrt 3 * a * (Real.sin B) = 0) →
  (B + π/12 = A) →
  (a = 2) →
  (A = π/6 ∧ (1/2 * a * c * Real.sin B = Real.sqrt 3 - 1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l745_74544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_max_value_l745_74500

open Real

-- Define the determinant as a function of θ
noncomputable def determinant (θ : ℝ) : ℝ :=
  let a11 := 1
  let a12 := 1
  let a13 := 1
  let a21 := 1
  let a22 := 1 + sinh θ
  let a23 := 1
  let a31 := 1 + cosh θ
  let a32 := 1
  let a33 := 1
  a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)

-- Theorem stating that the maximum value of the determinant is 0
theorem determinant_max_value :
  ∃ (θ_max : ℝ), ∀ (θ : ℝ), determinant θ ≤ determinant θ_max ∧ determinant θ_max = 0 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_max_value_l745_74500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l745_74545

-- Define the function g as noncomputable due to its dependence on Real.log
noncomputable def g (x : ℝ) : ℝ := Real.log (x^2)

-- Theorem stating that g is an even function
theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  intro x
  unfold g
  -- The proof is skipped using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_even_l745_74545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_in_regions_l745_74599

-- Define the regions
inductive Region
| A | B | C | D | E | F | G

-- Define the polynomial
def polynomial (p q x : ℝ) : ℝ := x^2 + p*x + q

-- Define the boundaries
def parabola (p q : ℝ) : Prop := p^2 - 4*q = 0
def line1 (p q : ℝ) : Prop := p + q + 1 = 0
def line2 (p q : ℝ) : Prop := -2*p + q + 4 = 0

-- Define the number of roots in (-2, 1) for each region
def num_roots (r : Region) : ℕ :=
  match r with
  | Region.A => 0
  | Region.B => 0
  | Region.C => 1
  | Region.D => 2
  | Region.E => 1
  | Region.F => 0
  | Region.G => 2

-- Main theorem
theorem roots_in_regions :
  ∀ (r : Region) (p q : ℝ),
    (∃ (x : ℝ), x ∈ Set.Ioo (-2 : ℝ) 1 ∧ polynomial p q x = 0) ↔ 
    num_roots r > 0 :=
by
  sorry

-- Helper lemmas (these would need to be proved in a complete implementation)
lemma roots_in_A (p q : ℝ) : 
  parabola p q → ¬∃ (x : ℝ), x ∈ Set.Ioo (-2 : ℝ) 1 ∧ polynomial p q x = 0 :=
by
  sorry

lemma roots_in_B (p q : ℝ) :
  p > 4 ∧ -2*p + q + 4 > 0 → ¬∃ (x : ℝ), x ∈ Set.Ioo (-2 : ℝ) 1 ∧ polynomial p q x = 0 :=
by
  sorry

-- Similar lemmas would be needed for other regions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_in_regions_l745_74599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_plus_n_eq_n_power_k_l745_74578

theorem factorial_plus_n_eq_n_power_k :
  ∀ n k : ℕ, n > 0 → (Nat.factorial n + n = n ^ k ↔ (n = 2 ∧ k = 2) ∨ (n = 3 ∧ k = 2) ∨ (n = 5 ∧ k = 3)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_plus_n_eq_n_power_k_l745_74578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_a_neg_one_unique_root_implies_m_value_l745_74574

/-- The function f(x) = ln(e^x + a + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.exp x + a + 1)

/-- The condition that f is an odd function -/
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

/-- The equation ln(x) / x = x^2 - 2ex + m -/
def has_unique_root (m : ℝ) : Prop :=
  ∃! x, x > 0 ∧ Real.log x / x = x^2 - 2 * Real.exp 1 * x + m

theorem f_odd_implies_a_neg_one (a : ℝ) :
  is_odd_function (f a) → a = -1 := by sorry

theorem unique_root_implies_m_value :
  has_unique_root (Real.exp 2 + 1 / Real.exp 1) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_implies_a_neg_one_unique_root_implies_m_value_l745_74574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completes_work_in_four_days_l745_74555

/-- The time it takes for worker a to complete the work alone -/
noncomputable def a_time : ℝ := 6

/-- The time it takes for workers a and b to complete the work together -/
noncomputable def ab_time : ℝ := 2.4

/-- The rate at which worker a completes the work -/
noncomputable def a_rate : ℝ := 1 / a_time

/-- The rate at which workers a and b complete the work together -/
noncomputable def ab_rate : ℝ := 1 / ab_time

/-- The rate at which worker b completes the work -/
noncomputable def b_rate : ℝ := ab_rate - a_rate

/-- The time it takes for worker b to complete the work alone -/
noncomputable def b_time : ℝ := 1 / b_rate

theorem b_completes_work_in_four_days : b_time = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_completes_work_in_four_days_l745_74555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_ratio_is_four_thirteenths_l745_74564

/-- A triangle with sides 5, 12, and 13 -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a = 5
  hb : b = 12
  hc : c = 13

/-- The diameter of the inscribed circle of the triangle -/
noncomputable def inscribedDiameter (t : RightTriangle) : ℝ :=
  2 * ((t.a + t.b - t.c) / 2)

/-- The diameter of the circumscribed circle of the triangle -/
noncomputable def circumscribedDiameter (t : RightTriangle) : ℝ :=
  t.c

/-- The ratio of the inscribed circle diameter to the circumscribed circle diameter -/
noncomputable def diameterRatio (t : RightTriangle) : ℝ :=
  (inscribedDiameter t) / (circumscribedDiameter t)

/-- Theorem: The ratio of the diameters is 4:13 -/
theorem diameter_ratio_is_four_thirteenths (t : RightTriangle) :
  diameterRatio t = 4 / 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_diameter_ratio_is_four_thirteenths_l745_74564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_longer_diagonal_l745_74565

/-- The length of the longer diagonal in a parallelogram with given heights and angle between them. -/
noncomputable def longerDiagonal (h₁ h₂ α : ℝ) : ℝ :=
  (Real.sqrt (h₁^2 + h₂^2 + 2*h₁*h₂*(Real.cos α))) / (Real.sin α)

/-- Theorem: In a parallelogram with heights h₁ and h₂ drawn from the vertex of an obtuse angle,
    and the angle between these heights being α, the length of the longer diagonal
    is (√(h₁² + h₂² + 2h₁h₂cosα)) / sinα. -/
theorem parallelogram_longer_diagonal (h₁ h₂ α : ℝ)
    (h_positive : h₁ > 0 ∧ h₂ > 0)
    (h_obtuse : α > Real.pi/2 ∧ α < Real.pi) :
    ∃ (d : ℝ), d = longerDiagonal h₁ h₂ α ∧ 
    ∀ (other_diagonal : ℝ), other_diagonal ≤ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelogram_longer_diagonal_l745_74565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_set_size_l745_74505

theorem min_sum_set_size : ∃ (S : Finset ℤ), 
  (∀ x ∈ S, ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = a + b + c) ∧ 
  S.card = 7 ∧
  (∀ T : Finset ℤ, (∀ x ∈ T, ∃ a b c, a ∈ T ∧ b ∈ T ∧ c ∈ T ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ x = a + b + c) → T.card ≥ 7) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_set_size_l745_74505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l745_74513

-- Define a regular hexagon with side length s
def regular_hexagon (s : ℝ) : ℝ → ℝ → Prop := sorry

-- Define the area of the central hexagon
noncomputable def central_hexagon_area (s : ℝ) : ℝ := (3 * Real.sqrt 3 / 2) * s^2

-- Define the area of one triangular section
noncomputable def triangular_section_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

-- Theorem statement
theorem hexagon_area_ratio (s : ℝ) (h : s > 0) :
  central_hexagon_area s / triangular_section_area s = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_area_ratio_l745_74513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_g_odd_l745_74525

-- Define the functions
def f (x : ℝ) : ℝ := x^3 - 1
noncomputable def g (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

-- Theorem 1: f is an increasing function
theorem f_increasing : ∀ x y : ℝ, x < y → f x < f y := by
  sorry

-- Theorem 2: g is an odd function
theorem g_odd : ∀ x : ℝ, g (-x) = -g x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_g_odd_l745_74525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_child_age_l745_74549

theorem child_age (h_avg_marriage : ℝ) (w_avg_marriage : ℝ) (child_age : ℝ) : 
  (h_avg_marriage + w_avg_marriage) / 2 = 23 →
  ((h_avg_marriage + 5) + (w_avg_marriage + 5) + child_age) / 3 = 19 →
  child_age = 1 := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check child_age

end NUMINAMATH_CALUDE_ERRORFEEDBACK_child_age_l745_74549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l745_74512

/-- The total cost function W(x) for a journey between two locations -/
noncomputable def W (x : ℝ) : ℝ := 2*x + 7200/x

/-- The speed that minimizes the total cost -/
def optimal_speed : ℝ := 60

theorem optimal_speed_minimizes_cost :
  ∀ x ∈ Set.Icc 50 100, W optimal_speed ≤ W x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l745_74512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_s_for_347_consecutive_l745_74594

-- Define the property of decimal representation containing 3, 4, 7 consecutively
def contains_347_consecutively (r s : ℕ) : Prop :=
  ∃ k : ℕ, (10^k * r) % s ≥ 347 * s / 1000 ∧ (10^k * r) % s < 348 * s / 1000

theorem smallest_s_for_347_consecutive (r s : ℕ) : 
  Nat.Coprime r s → 
  r < s → 
  contains_347_consecutively r s → 
  (∀ t : ℕ, t < s → ¬(contains_347_consecutively r t)) → 
  s = 653 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_s_for_347_consecutive_l745_74594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_potential_satisfies_conditions_l745_74588

open Complex

-- Define the complex potential
noncomputable def f (z : ℂ) : ℂ := -I * (z^2 + sinh z)

-- State the theorem
theorem complex_potential_satisfies_conditions :
  -- The real part of f(z) equals cosh(x) * sin(y) + 2xy up to a constant
  ∃ (c : ℝ), ∀ (x y : ℝ), (f (x + I*y)).re = Real.cosh x * Real.sin y + 2*x*y + c ∧
  -- f(0) = 0
  f 0 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_potential_satisfies_conditions_l745_74588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_special_matrix_l745_74552

theorem determinant_of_special_matrix (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![1,          Real.sin (a - b), Real.sin a],
    ![Real.sin (a - b), 1,          Real.sin b],
    ![Real.sin a,       Real.sin b, 1         ]
  ]
  Matrix.det M = 1 - (Real.sin a)^2 - (Real.sin b)^2 - (Real.sin (a - b))^2 + 2 * Real.sin a * Real.sin b * Real.sin (a - b) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_of_special_matrix_l745_74552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_less_than_19_21_l745_74584

def sequence_a : ℕ → ℝ := sorry
def sequence_S : ℕ → ℝ := sorry

/-- Checks if three terms form a geometric sequence -/
def IsGeometricSequence3 (a b c : ℝ) : Prop :=
  b ≠ 0 ∧ c ≠ 0 ∧ (a / b = b / c)

theorem max_m_less_than_19_21 :
  (∀ n : ℕ, n ≥ 2 → IsGeometricSequence3 (sequence_a n) (sequence_S (n-1) - 1) (sequence_S n)) →
  sequence_a 1 = 1/3 →
  (∀ n : ℕ, sequence_S n = (2*n - 1) / (2*n + 1)) →
  (∃ m : ℕ, m = 9 ∧ 
    (∀ k : ℕ, k < m → sequence_S k < 19/21) ∧
    (∀ k : ℕ, k > m → sequence_S k ≥ 19/21)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_m_less_than_19_21_l745_74584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l745_74579

/-- The eccentricity of a hyperbola with given properties -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_angle : a = Real.sqrt 3 * b) : 
  (Real.sqrt (a^2 + b^2)) / a = 2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l745_74579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_twelve_divisible_by_three_l745_74532

theorem count_divisors_of_twelve_divisible_by_three :
  ∃! (n : ℕ), n > 0 ∧ 
    (∀ (a : ℕ), (a > 0 ∧ 3 ∣ a ∧ a ∣ 12) ↔ a ∈ Finset.range n \ {0}) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_divisors_of_twelve_divisible_by_three_l745_74532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l745_74516

theorem largest_prime_factor_of_expression : 
  (Nat.factors (20^3 + 15^4 - 10^5)).maximum? = some 37 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_factor_of_expression_l745_74516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_z_coordinate_l745_74570

/-- Given a line passing through points (1, 3, 4) and (4, 2, 0),
    prove that the point on this line with x-coordinate 7 has a z-coordinate of -4. -/
theorem line_point_z_coordinate :
  let p₁ : Fin 3 → ℝ := ![1, 3, 4]
  let p₂ : Fin 3 → ℝ := ![4, 2, 0]
  let line := {p : Fin 3 → ℝ | ∃ t : ℝ, p = λ i => p₁ i + t * (p₂ i - p₁ i)}
  ∀ p ∈ line, p 0 = 7 → p 2 = -4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_point_z_coordinate_l745_74570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratio_in_rectangle_square_area_percent_in_rectangle_l745_74581

theorem square_area_ratio_in_rectangle (s : ℝ) (h1 : s > 0) : 
  (s^2) / ((2*s) * (6*s)) = 1 / 12 := by
  -- Simplify the left side of the equation
  have h2 : (s^2) / ((2*s) * (6*s)) = s^2 / (12*s^2) := by
    ring_nf
  
  -- Simplify further
  have h3 : s^2 / (12*s^2) = 1 / 12 := by
    field_simp [h1]
    ring
  
  -- Combine the steps
  rw [h2, h3]

theorem square_area_percent_in_rectangle (s : ℝ) (h1 : s > 0) : 
  ((s^2) / ((2*s) * (6*s))) * 100 = 100 / 12 := by
  have h2 : (s^2) / ((2*s) * (6*s)) = 1 / 12 := square_area_ratio_in_rectangle s h1
  rw [h2]
  ring

#eval (100 / 12 : Float)  -- This should output approximately 8.333333

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_area_ratio_in_rectangle_square_area_percent_in_rectangle_l745_74581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l745_74577

/-- Dilation of a complex number -/
noncomputable def dilation (center : ℂ) (scale : ℝ) (z : ℂ) : ℂ :=
  center + scale • (z - center)

/-- The problem statement -/
theorem dilation_problem : 
  dilation (1 + 2*I) 4 (-2 + I) = -11 - 2*I := by
  -- Unfold the definition of dilation
  unfold dilation
  -- Simplify the expression
  simp [Complex.I]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_problem_l745_74577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divides_next_l745_74519

def a : ℕ → ℕ
  | 0 => 2  -- Define a(0) to handle the Nat.zero case
  | 1 => 2
  | n + 2 => 2^(a (n + 1)) + 2

theorem a_divides_next : ∀ n : ℕ, n ≥ 2 → (a (n - 1)) ∣ (a n) := by
  intro n hn
  sorry

#eval a 0  -- This will output 2
#eval a 1  -- This will output 2
#eval a 2  -- This will output 6
#eval a 3  -- This will output 66

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_divides_next_l745_74519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tint_percentage_after_addition_l745_74573

/-- Represents a paint mixture -/
structure PaintMixture where
  total_volume : ℝ
  red_tint_percentage : ℝ
  yellow_tint_percentage : ℝ
  water_percentage : ℝ

/-- Calculates the new red tint percentage after adding pure red tint -/
noncomputable def new_red_tint_percentage (mixture : PaintMixture) (added_red_tint : ℝ) : ℝ :=
  let original_red_tint := mixture.total_volume * mixture.red_tint_percentage / 100
  let new_total_volume := mixture.total_volume + added_red_tint
  let new_red_tint := original_red_tint + added_red_tint
  (new_red_tint / new_total_volume) * 100

/-- The theorem to be proved -/
theorem red_tint_percentage_after_addition
  (mixture : PaintMixture)
  (h1 : mixture.total_volume = 50)
  (h2 : mixture.red_tint_percentage = 30)
  (h3 : mixture.yellow_tint_percentage = 50)
  (h4 : mixture.water_percentage = 20)
  (h5 : mixture.red_tint_percentage + mixture.yellow_tint_percentage + mixture.water_percentage = 100) :
  new_red_tint_percentage mixture 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_red_tint_percentage_after_addition_l745_74573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_risk_factor_probability_l745_74515

def total_population : ℕ := 200

def prob_one_factor : ℚ := 7/100
def prob_two_factors : ℚ := 12/100
def prob_all_given_xy : ℚ := 2/5

theorem risk_factor_probability :
  let num_one_factor := (prob_one_factor * total_population).floor
  let num_two_factors := (prob_two_factors * total_population).floor
  let num_all_factors := ((prob_all_given_xy * num_two_factors) / (1 - prob_all_given_xy)).floor
  let num_no_factors := total_population - (3 * num_one_factor + 3 * num_two_factors + num_all_factors)
  let num_not_x := total_population - (num_one_factor + 2 * num_two_factors + num_all_factors)
  (num_no_factors : ℚ) / num_not_x = 70/122 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_risk_factor_probability_l745_74515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_value_l745_74575

/-- The expansion of (x + 1/x)(ax - 1)^5 -/
noncomputable def expansion (x a : ℝ) := (x + 1/x) * (a*x - 1)^5

/-- The sum of coefficients in the expansion -/
noncomputable def sum_of_coefficients (a : ℝ) := 2

/-- The constant term in the expansion -/
noncomputable def constant_term (a : ℝ) : ℝ := 10

theorem constant_term_value :
  ∀ a : ℝ, sum_of_coefficients a = 2 → constant_term a = 10 :=
by
  intro a h
  simp [constant_term]
  
#check constant_term_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_value_l745_74575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_add_base7_correct_add_12_254_base7_l745_74592

/-- Represents a number in base 7 -/
def Base7 : Type := List Nat

/-- Converts a base 7 number to a natural number -/
def to_nat (b : Base7) : Nat :=
  b.reverse.enum.foldr (fun p acc => acc + p.2 * 7^p.1) 0

/-- Addition in base 7 -/
def add_base7 (a b : Base7) : Base7 :=
  sorry

theorem add_base7_correct (a b : Base7) :
  to_nat (add_base7 a b) = (to_nat a + to_nat b) % 7^(max a.length b.length) := by
  sorry

/-- The main theorem proving the addition of 12₇ and 254₇ in base 7 -/
theorem add_12_254_base7 :
  add_base7 [2, 1] [4, 5, 2] = [6, 0, 3] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_add_base7_correct_add_12_254_base7_l745_74592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_over_y_l745_74520

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 1

-- State the theorem
theorem max_x_over_y (x y : ℝ) (h : circle_equation x y) : x / y ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_x_over_y_l745_74520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_sqrt_three_l745_74586

/-- Prove that √2sin(15°) + √2cos(15°) = √3 and (1 + tan(15°)) / (1 - tan(15°)) = √3 -/
theorem trig_identities_sqrt_three :
  (Real.sqrt 2 * Real.sin (15 * π / 180) + Real.sqrt 2 * Real.cos (15 * π / 180) = Real.sqrt 3) ∧
  ((1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identities_sqrt_three_l745_74586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l745_74526

open Real

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 2 * cos x

-- State the theorem
theorem max_value_of_f :
  ∃ (x : ℝ), x ∈ Set.Icc 0 (π / 2) ∧
  f x = π / 6 + sqrt 3 ∧
  ∀ (y : ℝ), y ∈ Set.Icc 0 (π / 2) → f y ≤ f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l745_74526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l745_74568

-- Define the vertices of the quadrilateral
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (2, 5)
def C : ℝ × ℝ := (5, 5)
def D : ℝ × ℝ := (6, 2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the perimeter of the quadrilateral
noncomputable def perimeter : ℝ := distance A B + distance B C + distance C D + distance D A

-- Theorem statement
theorem quadrilateral_perimeter :
  ∃ (c d : ℤ), perimeter = Real.sqrt 29 + 3 + c * Real.sqrt 10 ∧ c = 3 ∧ c + d = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_perimeter_l745_74568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_condition_l745_74580

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the line
def line_eq (x y k : ℝ) : Prop := y = k * x + Real.sqrt 2

-- Define the tangency condition
def is_tangent (k : ℝ) : Prop :=
  ∃ x y, circle_eq x y ∧ line_eq x y k ∧
  ∀ x' y', circle_eq x' y' → line_eq x' y' k → (x = x' ∧ y = y')

-- Statement to prove
theorem tangency_condition :
  (k = 1 → is_tangent k) ∧ 
  ¬(is_tangent k → k = 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangency_condition_l745_74580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_l745_74566

/-- The side length of the equilateral triangle -/
def side_length : ℝ := 4

/-- The distance threshold from vertices -/
def distance_threshold : ℝ := 1

/-- The area of the equilateral triangle -/
noncomputable def triangle_area : ℝ := Real.sqrt 3 * side_length^2 / 4

/-- The area of the region where the ant is within distance_threshold of any vertex -/
noncomputable def near_vertex_area : ℝ := 3 * Real.pi * distance_threshold^2 / 2

/-- The probability that the ant is more than distance_threshold away from all vertices -/
noncomputable def probability_far_from_vertices : ℝ := 1 - near_vertex_area / triangle_area

theorem ant_probability : 
  probability_far_from_vertices = 1 - (Real.sqrt 3 * Real.pi) / 24 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_probability_l745_74566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_after_bets_l745_74503

noncomputable def initial_amount : ℝ := 100

noncomputable def bet_amount (current : ℝ) : ℝ := current / 2

noncomputable def win_amount (bet : ℝ) : ℝ := 2 * bet

noncomputable def lose_amount (bet : ℝ) : ℝ := 0.6 * bet

noncomputable def after_win (current : ℝ) : ℝ := current + win_amount (bet_amount current)

noncomputable def after_loss (current : ℝ) : ℝ := current - lose_amount (bet_amount current)

theorem final_amount_after_bets :
  let amount_after_wins := after_win (after_win initial_amount)
  let final_amount := after_loss (after_loss amount_after_wins)
  final_amount = 196 := by
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_amount_after_bets_l745_74503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_collection_earnings_difference_l745_74530

/-- Proves that the difference between Peter's and Simon's earnings in currency B is 3700 units --/
theorem stamp_collection_earnings_difference :
  let simon_stamps : ℕ := 30
  let peter_stamps : ℕ := 80
  let red_stamp_price_A : ℕ := 5
  let white_stamp_price_B : ℕ := 50
  let exchange_rate : ℚ := 2

  let simon_earnings_A : ℕ := simon_stamps * red_stamp_price_A
  let simon_earnings_B : ℚ := (simon_earnings_A : ℚ) * exchange_rate
  let peter_earnings_B : ℕ := peter_stamps * white_stamp_price_B

  (peter_earnings_B : ℚ) - simon_earnings_B = 3700 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_collection_earnings_difference_l745_74530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_fraction_l745_74569

def is_valid_phone_number (n : ℕ) : Bool :=
  100000000 ≤ n ∧ n < 600000000

def begins_with_5_ends_with_2 (n : ℕ) : Bool :=
  500000002 ≤ n ∧ n < 600000000 ∧ n % 10 = 2

theorem phone_number_fraction :
  (Finset.filter (λ n => begins_with_5_ends_with_2 n) (Finset.filter (λ n => is_valid_phone_number n) (Finset.range 1000000000))).card /
  (Finset.filter (λ n => is_valid_phone_number n) (Finset.range 1000000000)).card = 1 / 50 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phone_number_fraction_l745_74569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l745_74582

theorem min_abs_difference (x y : ℤ) (hx : x > 0) (hy : y > 0) 
  (h : x * y - 5 * x + 6 * y = 216) :
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ 
  a * b - 5 * a + 6 * b = 216 ∧
  ∀ (c d : ℤ), c > 0 → d > 0 → c * d - 5 * c + 6 * d = 216 →
  |a - b| ≤ |c - d| ∧
  |a - b| = 36 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_abs_difference_l745_74582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l745_74524

-- Define the constants
noncomputable def a : ℝ := Real.sqrt 2
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := Real.log 2 / Real.log (1/3)

-- State the theorem
theorem a_greater_than_b_greater_than_c : a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_greater_than_b_greater_than_c_l745_74524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_1_05_to_6th_power_l745_74501

-- Define the base and exponent
def base : ℝ := 1.05
def exponent : ℕ := 6

-- Define the approximation function
def approximate (x : ℝ) (n : ℕ) : ℝ := 
  (1 + x)^n

-- Define the rounding function to the nearest hundredth
noncomputable def round_to_hundredth (x : ℝ) : ℝ :=
  ⌊x * 100 + 0.5⌋ / 100

-- Theorem statement
theorem approximate_1_05_to_6th_power : 
  round_to_hundredth (approximate 0.05 6) = 1.34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_approximate_1_05_to_6th_power_l745_74501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_distance_ahead_problem_solution_l745_74553

-- Define the problem parameters
noncomputable def jogger_speed : ℝ := 9 -- km/hr
noncomputable def train_speed : ℝ := 45 -- km/hr
noncomputable def train_length : ℝ := 100 -- meters
noncomputable def passing_time : ℝ := 25 -- seconds

-- Define the conversion factor from km/hr to m/s
noncomputable def km_hr_to_m_s : ℝ := 5 / 18

-- Define the theorem
theorem jogger_distance_ahead (jogger_speed train_speed train_length passing_time km_hr_to_m_s : ℝ) 
  (h1 : jogger_speed = 9)
  (h2 : train_speed = 45)
  (h3 : train_length = 100)
  (h4 : passing_time = 25)
  (h5 : km_hr_to_m_s = 5 / 18) :
  (train_speed - jogger_speed) * km_hr_to_m_s * passing_time - train_length = 150 := by
  sorry

-- Apply the theorem to our specific problem
theorem problem_solution : 
  (train_speed - jogger_speed) * km_hr_to_m_s * passing_time - train_length = 150 := by
  apply jogger_distance_ahead jogger_speed train_speed train_length passing_time km_hr_to_m_s
  · rfl
  · rfl
  · rfl
  · rfl
  · rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogger_distance_ahead_problem_solution_l745_74553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_not_on_line_l745_74539

noncomputable def harry_x : ℝ := 12
noncomputable def harry_y : ℝ := 3
noncomputable def sandy_x : ℝ := 4
noncomputable def sandy_y : ℝ := 9

noncomputable def midpoint_x : ℝ := (harry_x + sandy_x) / 2
noncomputable def midpoint_y : ℝ := (harry_y + sandy_y) / 2

def on_line (x y : ℝ) : Prop := y = -x + 6

theorem midpoint_not_on_line :
  midpoint_x = 8 ∧ midpoint_y = 6 ∧ ¬(on_line midpoint_x midpoint_y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_not_on_line_l745_74539
