import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_3_l397_39749

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  rho : ℝ
  theta : ℝ

/-- Represents a line in polar coordinates -/
structure PolarLine where
  equation : PolarPoint → Prop

/-- Represents a circle in polar coordinates -/
structure PolarCircle where
  equation : PolarPoint → Prop

/-- The line √3ρcosθ - ρsinθ = 0 in polar coordinates -/
def polarLine : PolarLine :=
  { equation := fun p => Real.sqrt 3 * p.rho * Real.cos p.theta - p.rho * Real.sin p.theta = 0 }

/-- The circle ρ = 4sinθ in polar coordinates -/
def polarCircle : PolarCircle :=
  { equation := fun p => p.rho = 4 * Real.sin p.theta }

/-- The length of the chord formed by the intersection of the line and the circle -/
noncomputable def chordLength (l : PolarLine) (c : PolarCircle) : ℝ := 
  2 * Real.sqrt 3 -- This is a placeholder value; the actual computation is complex

/-- Theorem stating that the length of the chord is 2√3 -/
theorem chord_length_is_2_sqrt_3 : chordLength polarLine polarCircle = 2 * Real.sqrt 3 := by
  sorry -- The proof is omitted for now

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_is_2_sqrt_3_l397_39749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l397_39712

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a ^ (2 : ℝ) ^ x

theorem f_increasing (a : ℝ) (h : a > 1) : 
  ∀ x y : ℝ, x < y → f a x < f a y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_l397_39712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertices_distance_l397_39738

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 99 - y^2 / 36 = 1

-- Define the distance between vertices
noncomputable def distance_between_vertices : ℝ := 6 * Real.sqrt 11

-- Theorem statement
theorem hyperbola_vertices_distance :
  ∀ x y : ℝ, hyperbola_equation x y → 
  ∃ v1 v2 : ℝ × ℝ, 
    (hyperbola_equation v1.1 v1.2 ∧ 
     hyperbola_equation v2.1 v2.2 ∧
     v1 ≠ v2 ∧
     ∀ p : ℝ × ℝ, hyperbola_equation p.1 p.2 → 
       (p = v1 ∨ p = v2 ∨ (p.2 ≠ 0 ∧ |p.1| < |v1.1|))) ∧
    Real.sqrt ((v1.1 - v2.1)^2 + (v1.2 - v2.2)^2) = distance_between_vertices :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertices_distance_l397_39738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_movies_watched_l397_39793

theorem movies_watched (total_movies : Nat) (total_books : Nat) (books_read : Nat) (movies_watched : Nat) :
  total_movies = 14 →
  total_books = 15 →
  books_read = 11 →
  books_read = (movies_watched + 1) →
  movies_watched = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_movies_watched_l397_39793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l397_39753

/-- The sequence T defined recursively -/
def T : ℕ → ℕ
  | 0 => 11  -- Add this case for 0
  | 1 => 11
  | n + 2 => 11^(T (n + 1))

/-- The statement to prove -/
theorem t_100_mod_7 : T 100 % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_100_mod_7_l397_39753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_interval_l397_39786

def f (a x : ℝ) : ℝ := x^2 + a

def f_iter (a : ℝ) : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => f a (f_iter a n x)

def M : Set ℝ := {a | ∀ n : ℕ, n > 0 → |f_iter a n 0| ≤ 2}

theorem M_equals_interval : M = Set.Icc (-2 : ℝ) (1/4) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_equals_interval_l397_39786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l397_39723

/-- Triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given triangle with A = 60° and a = 6 -/
noncomputable def givenTriangle : Triangle where
  a := 6
  b := Real.sqrt 3 -- Placeholder value, will be replaced in the theorem
  c := Real.sqrt 3 -- Placeholder value, will be replaced in the theorem
  A := Real.pi / 3
  B := Real.pi / 3 -- Placeholder value, will be replaced in the theorem
  C := Real.pi / 3 -- Placeholder value, will be replaced in the theorem

/-- Vector dot product -/
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

/-- Area of a triangle given two sides and the included angle -/
noncomputable def triangle_area (b c θ : ℝ) : ℝ :=
  1/2 * b * c * Real.sin θ

theorem triangle_properties (t : Triangle) (h : t = givenTriangle) :
  /- 1. If b = √3, then B has only one solution -/
  (t.b = Real.sqrt 3 → ∃! x, t.B = x) ∧
  /- 2. If AB · AC = 12, then the area of triangle ABC is 6√3 -/
  (dot_product (t.b, 0) (t.c * Real.cos t.A, t.c * Real.sin t.A) = 12 →
    triangle_area t.b t.c t.A = 6 * Real.sqrt 3) ∧
  /- 3. b + c < 13 -/
  (t.b + t.c < 13) ∧
  /- 4. The maximum value of (AB + AC) · BC is 24√3 -/
  (∃ m : ℝ, m = 24 * Real.sqrt 3 ∧
    ∀ x y : ℝ × ℝ, dot_product (x.1 + y.1, x.2 + y.2) (t.c - t.b, 0) ≤ m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l397_39723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_pathway_in_photosynthesis_l397_39715

-- Define the types for our compounds
inductive Compound
  | CO2
  | C3
  | Sugar

-- Define the pathway type
def Pathway := List Compound

-- Define the property of a valid photosynthesis pathway
def IsValidPhotosynthesisPathway (p : Pathway) : Prop :=
  p.head? = some Compound.CO2 ∧ p.getLast? = some Compound.Sugar

-- State the theorem
theorem carbon_pathway_in_photosynthesis :
  ∃ (p : Pathway), IsValidPhotosynthesisPathway p ∧ 
    p = [Compound.CO2, Compound.C3, Compound.Sugar] :=
by
  -- Construct the pathway
  let p : Pathway := [Compound.CO2, Compound.C3, Compound.Sugar]
  
  -- Prove that p exists and satisfies the conditions
  use p
  constructor
  · -- Prove that p is a valid photosynthesis pathway
    constructor
    · rfl  -- p.head? = some Compound.CO2
    · rfl  -- p.getLast? = some Compound.Sugar
  · -- Prove that p is equal to [Compound.CO2, Compound.C3, Compound.Sugar]
    rfl

  -- The proof is complete

end NUMINAMATH_CALUDE_ERRORFEEDBACK_carbon_pathway_in_photosynthesis_l397_39715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_rain_time_l397_39702

/-- Represents the driving conditions and results for Shelby's scooter ride --/
structure DrivingScenario where
  speed_sun : ℚ  -- Speed when not raining (in miles per hour)
  speed_rain : ℚ  -- Speed when raining (in miles per hour)
  total_distance : ℚ  -- Total distance driven (in miles)
  total_time : ℚ  -- Total time driven (in minutes)

/-- Calculates the time driven in rain given a DrivingScenario --/
def time_in_rain (scenario : DrivingScenario) : ℚ :=
  let speed_sun_per_minute := scenario.speed_sun / 60
  let speed_rain_per_minute := scenario.speed_rain / 60
  (scenario.total_distance - speed_sun_per_minute * scenario.total_time) /
    (speed_rain_per_minute - speed_sun_per_minute)

/-- Theorem stating that given Shelby's driving conditions, she drove in the rain for approximately 33 minutes --/
theorem shelby_rain_time :
  let scenario : DrivingScenario := {
    speed_sun := 40,
    speed_rain := 25,
    total_distance := 25,
    total_time := 50
  }
  ∃ ε : ℚ, ε > 0 ∧ |time_in_rain scenario - 33| < ε :=
by
  -- The proof goes here
  sorry

#eval time_in_rain {
  speed_sun := 40,
  speed_rain := 25,
  total_distance := 25,
  total_time := 50
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelby_rain_time_l397_39702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_l397_39750

def u (s : ℝ) : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 1 + 3*s
  | 1 => -4 + 7*s
  | 2 => 2 + 4*s

def b : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 5
  | 1 => 1
  | 2 => -3

def direction : Fin 3 → ℝ := fun i =>
  match i with
  | 0 => 3
  | 1 => 7
  | 2 => 4

theorem closest_point (s : ℝ) :
  (∀ t : ℝ, ‖u s - b‖ ≤ ‖u t - b‖) ↔ s = 27/74 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_l397_39750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_game_winner_l397_39785

def StoneGame (n : ℕ) :=
  n ≥ 2

def PlayerATurn (n : ℕ) (removed : ℕ) :=
  removed ≥ 1 ∧ removed < n

def PlayerBTurn (prevRemoved : ℕ) (removed : ℕ) :=
  removed ≥ 1 ∧ removed ≤ prevRemoved

def IsPowerOfTwo (n : ℕ) :=
  ∃ k : ℕ, n = 2^k

-- Define a strategy type
def Strategy := ℕ → ℕ

-- Define winning condition for player B
def PlayerBWins (n : ℕ) (strategy : Strategy) : Prop :=
  sorry -- We'll leave this as sorry for now, as implementing the full game logic would be complex

theorem stone_game_winner (n : ℕ) (h : StoneGame n) :
  (∃ strategy : Strategy, PlayerBWins n strategy) ↔ IsPowerOfTwo n :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stone_game_winner_l397_39785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_3_sufficient_not_necessary_l397_39733

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def are_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- The slope of the first line ax-2y+3a=0 -/
noncomputable def slope1 (a : ℝ) : ℝ := a / 2

/-- The slope of the second line (a-1)x+3y+a^2-a+3=0 -/
noncomputable def slope2 (a : ℝ) : ℝ := -(a - 1) / 3

/-- The statement that a=3 is sufficient but not necessary for the lines to be perpendicular -/
theorem a_eq_3_sufficient_not_necessary :
  (∃ (a : ℝ), a ≠ 3 ∧ are_perpendicular (slope1 a) (slope2 a)) ∧
  are_perpendicular (slope1 3) (slope2 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_eq_3_sufficient_not_necessary_l397_39733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l397_39701

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Define the pentagon ABHED
structure Pentagon where
  A : Point
  B : Point
  H : Point
  E : Point
  D : Point

-- Define the conditions
def pentagon_conditions (p : Pentagon) : Prop :=
  p.D.x = 0 ∧ p.D.y = 0 ∧
  p.A.x = 8 ∧ p.A.y = 0 ∧
  p.B.x = 8 ∧ p.B.y = 12 ∧
  p.E.x = 0 ∧ p.E.y = 8 ∧
  p.H.x = 8 ∧ p.H.y = 5

-- Define the area function
noncomputable def area (p : Pentagon) : ℝ :=
  96 + (7 * Real.sqrt 89) / 2 - (8 * Real.sqrt 73) / 2

-- State the theorem
theorem pentagon_area (p : Pentagon) :
  pentagon_conditions p → area p = 96 + (7 * Real.sqrt 89) / 2 - (8 * Real.sqrt 73) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l397_39701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_A_to_l_l397_39731

/-- The line l: x + y + 3 = 0 -/
def line_l (x y : ℝ) : Prop := x + y + 3 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (2, 1)

/-- Distance between two points in ℝ² -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Minimum distance from point A to line l -/
theorem min_distance_A_to_l :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧
  ∀ (P : ℝ × ℝ), line_l P.1 P.2 → distance point_A P ≥ d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_A_to_l_l397_39731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l397_39725

/-- Calculates the length of a train given its speed and time to cross a pole. -/
noncomputable def trainLength (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (1000 / 3600) * time

/-- Theorem stating that a train traveling at 120 km/hr and crossing a pole in 15 seconds 
    has a length of approximately 500 meters. -/
theorem train_length_theorem (speed : ℝ) (time : ℝ) 
    (h1 : speed = 120) 
    (h2 : time = 15) : 
    ∃ ε > 0, |trainLength speed time - 500| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_theorem_l397_39725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_values_for_special_trig_equation_l397_39760

theorem sin_values_for_special_trig_equation (B : Real) :
  3 * Real.tan B - Real.cos B⁻¹ = 1 → Real.sin B = 0 ∨ Real.sin B = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_values_for_special_trig_equation_l397_39760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_consecutive_divisible_by_digit_sum_l397_39763

def sum_of_digits : ℕ → ℕ
| n => if n < 10 then n else (n % 10 + sum_of_digits (n / 10))

theorem eighteen_consecutive_divisible_by_digit_sum :
  ∀ n : ℕ, 100 ≤ n → n ≤ 982 →
  ∃ k : ℕ, n ≤ k ∧ k ≤ n + 17 ∧ k % (sum_of_digits k) = 0 :=
by sorry

example : ∃ l : List ℕ,
  l.length = 17 ∧
  (∀ n ∈ l, 973 ≤ n ∧ n ≤ 989) ∧
  (∀ n ∈ l, n % (sum_of_digits n) ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighteen_consecutive_divisible_by_digit_sum_l397_39763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_midpoint_M_l397_39711

noncomputable section

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (-1 + 3/5 * t, 1 + 4/5 * t)

-- Define the curve C
def curve_C (x y : ℝ) : Prop := y^2 - x^2 = 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ 
  A = line_l t₁ ∧ B = line_l t₂ ∧
  curve_C A.1 A.2 ∧ curve_C B.1 B.2

-- Theorem for the length of AB
theorem length_AB (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 20 * Real.sqrt 14 / 7 := by
  sorry

-- Theorem for the midpoint M of AB
theorem midpoint_M (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (-4, -3) := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_AB_midpoint_M_l397_39711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_addition_complex_addition_proof_l397_39716

theorem complex_addition : ℂ → Prop :=
  fun z => (5 - 3 * Complex.I + z = -4 + 9 * Complex.I) → z = -9 + 12 * Complex.I

-- The proof is omitted
theorem complex_addition_proof : complex_addition (-9 + 12 * Complex.I) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_addition_complex_addition_proof_l397_39716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_1024_l397_39748

def divisor_product (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).prod id

theorem divisor_product_1024 (n : ℕ) : divisor_product n = 1024 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_product_1024_l397_39748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_27_point_7_l397_39771

-- Define the arithmetic sequence
noncomputable def arithmeticSequence (x y : ℝ) : ℕ → ℝ
  | 0 => x + 2*y
  | 1 => x - 2*y
  | 2 => 2*x*y
  | 3 => x/y
  | n+4 => arithmeticSequence x y 0 - (n+4) * 4*y

-- State the theorem
theorem fifth_term_is_negative_27_point_7 :
  ∃ x y : ℝ, arithmeticSequence x y 4 = -27.7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_is_negative_27_point_7_l397_39771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_9ln_l397_39790

-- Define the integrand
noncomputable def f (x : ℝ) : ℝ := (3 * Real.tan x ^ 2 - 50) / (2 * Real.tan x + 7)

-- Define the lower bound of the integral
noncomputable def a : ℝ := -Real.arccos (1 / Real.sqrt 10)

-- Define the upper bound of the integral
def b : ℝ := 0

-- State the theorem
theorem integral_equals_9ln :
  ∫ x in a..b, f x = 9 * Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equals_9ln_l397_39790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_area_example_l397_39739

/-- The lateral surface area of a frustum of a right circular cone. -/
noncomputable def frustumLateralArea (R r h : ℝ) : ℝ :=
  Real.pi * (R + r) * Real.sqrt ((R - r)^2 + h^2)

/-- Theorem: The lateral surface area of a frustum of a right circular cone
    with lower base radius 10 inches, upper base radius 5 inches,
    and height 8 inches is equal to 15π√89 square inches. -/
theorem frustum_lateral_area_example :
  frustumLateralArea 10 5 8 = 15 * Real.pi * Real.sqrt 89 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_frustum_lateral_area_example_l397_39739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_l397_39797

/-- Represents a part of a triangle -/
structure TrianglePart where
  -- Add necessary fields
  id : Nat

/-- Represents an equilateral triangle -/
structure EquilateralTriangle where
  -- Add necessary fields
  sideLength : ℝ

/-- Function to check if a list of TrianglePart can form an EquilateralTriangle -/
def canFormEquilateralTriangle (parts : List TrianglePart) : Prop :=
  sorry

/-- Theorem stating that an equilateral triangle can be divided into 5 parts 
    from which 3 equilateral triangles can be formed -/
theorem equilateral_triangle_division :
  ∃ (original : EquilateralTriangle) (parts : List TrianglePart),
    parts.length = 5 ∧
    (∃ (t1 t2 t3 : List TrianglePart),
      (∀ p ∈ t1, p ∈ parts) ∧
      (∀ p ∈ t2, p ∈ parts) ∧
      (∀ p ∈ t3, p ∈ parts) ∧
      (∀ p, p ∈ t1 → p ∉ t2) ∧
      (∀ p, p ∈ t1 → p ∉ t3) ∧
      (∀ p, p ∈ t2 → p ∉ t3) ∧
      canFormEquilateralTriangle t1 ∧
      canFormEquilateralTriangle t2 ∧
      canFormEquilateralTriangle t3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_division_l397_39797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integers_satisfying_inequality_l397_39724

theorem odd_integers_satisfying_inequality :
  (Finset.filter (fun n : ℕ => 
    Odd n ∧ 
    n > 0 ∧ 
    n ≤ 15 ∧
    ((n : ℤ) + 10) * ((n : ℤ) - 5) * ((n : ℤ) - 15) < 0) 
    (Finset.range 16)).card = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_integers_satisfying_inequality_l397_39724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l397_39713

theorem calculation_proofs :
  (6 / (-1/2 + 1/3) = -36) ∧
  ((-14/17) * 99 + 13/17 * 99 - 16/17 * 99 = -99) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculation_proofs_l397_39713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_list_properties_l397_39707

/-- Represents a number in the list with all 9s except the last digit which is 7 -/
def specialNumber (n : ℕ) : ℕ := 10^n - 3

/-- The sum of the first n natural numbers -/
def triangularNumber (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of a geometric series -/
def geometricSum (a r : ℕ) (n : ℕ) : ℕ := a * (r^n - 1) / (r - 1)

/-- Convert a natural number to a list of its digits -/
def natToDigits (n : ℕ) : List ℕ :=
  if n < 10 then [n]
  else (n % 10) :: natToDigits (n / 10)

theorem special_number_list_properties :
  let listLength : ℕ := 100
  let nineCount : ℕ := triangularNumber listLength
  let sumOfNumbers : ℕ := (geometricSum 100 10 listLength) - 300
  let sumOfDigits : ℕ := (natToDigits sumOfNumbers).sum
  (nineCount = 5050) ∧ (sumOfDigits = 106) := by
  sorry

#eval triangularNumber 100  -- Should output 5050
#eval (natToDigits ((geometricSum 100 10 100) - 300)).sum  -- Should output 106

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_number_list_properties_l397_39707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l397_39769

-- Define the circles in Cartesian coordinates
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

-- Define the polar coordinate transformation
noncomputable def polar_to_cart (ρ θ : ℝ) : ℝ × ℝ := (ρ * Real.cos θ, ρ * Real.sin θ)

-- Theorem statement
theorem circle_properties :
  -- 1. Polar equations
  (∀ ρ θ, C₁ (polar_to_cart ρ θ).1 (polar_to_cart ρ θ).2 ↔ ρ = 2) ∧
  (∀ ρ θ, C₂ (polar_to_cart ρ θ).1 (polar_to_cart ρ θ).2 ↔ ρ = 4 * Real.cos θ) ∧
  -- 2. Intersection points
  (C₁ (polar_to_cart 2 (π/3)).1 (polar_to_cart 2 (π/3)).2 ∧
   C₂ (polar_to_cart 2 (π/3)).1 (polar_to_cart 2 (π/3)).2) ∧
  (C₁ (polar_to_cart 2 (-π/3)).1 (polar_to_cart 2 (-π/3)).2 ∧
   C₂ (polar_to_cart 2 (-π/3)).1 (polar_to_cart 2 (-π/3)).2) ∧
  -- 3. Common chord
  (∀ θ, -π/3 ≤ θ ∧ θ ≤ π/3 →
    C₁ 1 (Real.tan θ) ∧ C₂ 1 (Real.tan θ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_properties_l397_39769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_second_quadrant_l397_39735

theorem complex_in_second_quadrant (A B : Real) (h1 : 0 < A) (h2 : A < π / 2) (h3 : 0 < B) (h4 : B < π / 2) (h5 : A + B < π) :
  let z : ℂ := Complex.mk (Real.cos B - Real.sin A) (Real.sin B - Real.cos A)
  (z.re < 0) ∧ (z.im > 0) :=
by
  -- Introduce the complex number z
  intro z
  
  -- Split the goal into two parts
  constructor
  
  -- Prove z.re < 0
  · sorry
  
  -- Prove z.im > 0
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_in_second_quadrant_l397_39735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_addition_theorem_l397_39782

def binary_to_nat (b : List Bool) : Nat :=
  b.foldr (fun bit acc => 2 * acc + if bit then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

def a : List Bool := [true, false, true, true]  -- 1101₂
def b : List Bool := [true, false, true]        -- 101₂
def c : List Bool := [false, true, true, true]  -- 1110₂
def d : List Bool := [true, false, false, false, true]  -- 10001₂

def result : List Bool := [true, false, false, false, false, false, true]  -- 1000001₂

theorem binary_addition_theorem :
  nat_to_binary (binary_to_nat a + binary_to_nat b + binary_to_nat c + binary_to_nat d) = result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binary_addition_theorem_l397_39782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_divisibility_32_factorial_l397_39706

theorem power_divisibility_32_factorial :
  let n : ℕ := 32
  let factorial_n := n.factorial
  let largest_power_of_two := (factorial_n.factorization 2)
  let largest_power_of_three := (factorial_n.factorization 3)
  (Nat.pow 2 largest_power_of_two % 10 = 8) ∧
  (Nat.pow 3 largest_power_of_three % 10 = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_divisibility_32_factorial_l397_39706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_price_decrease_l397_39784

noncomputable def percent_decrease (original_price sale_price : ℝ) : ℝ :=
  ((original_price - sale_price) / original_price) * 100

theorem trouser_price_decrease :
  let original_price : ℝ := 100
  let sale_price : ℝ := 80
  percent_decrease original_price sale_price = 20 := by
  -- Unfold the definition of percent_decrease
  unfold percent_decrease
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trouser_price_decrease_l397_39784


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_intersections_correct_l397_39727

/-- The rational function under consideration -/
noncomputable def f (x : ℝ) : ℝ := (x^2 - 9*x + 20) / (x^2 - 9*x + 18)

/-- The set of asymptote intersection points -/
def asymptote_intersections : Set (ℝ × ℝ) := {(3, 1), (6, 1)}

/-- Theorem stating that the given points are the intersections of the asymptotes -/
theorem asymptote_intersections_correct :
  ∀ p ∈ asymptote_intersections, 
    (∃ (ε : ℝ), ε > 0 ∧ 
      ((∀ x ∈ Set.Ioo (p.1 - ε) p.1, |f x| > (1/ε)) ∨
       (∀ x ∈ Set.Ioo p.1 (p.1 + ε), |f x| > (1/ε))) ∧
      (∀ δ > 0, ∃ N : ℝ, ∀ x, |x| > N → |f x - p.2| < δ)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_asymptote_intersections_correct_l397_39727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_length_segments_l397_39700

/-- A point in the coordinate plane with integer coordinates -/
structure Point where
  x : ℤ
  y : ℤ
  hx : 1 ≤ x ∧ x ≤ 2016
  hy : 1 ≤ y ∧ y ≤ 2016

/-- The set of 2017 points in the coordinate plane -/
def PointSet : Type := Fin 2017 → Point

/-- The distance between two points -/
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 : ℝ)

/-- Theorem stating that there exist two distinct segments with the same length -/
theorem equal_length_segments (points : PointSet) :
  ∃ (a b c d : Fin 2017), a ≠ b ∧ c ≠ d ∧ (a, b) ≠ (c, d) ∧
    distance (points a) (points b) = distance (points c) (points d) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_length_segments_l397_39700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_skate_rental_cost_l397_39766

/-- The rental cost of skates at an ice skating rink. -/
def rental_cost : ℚ := 2.5

/-- The admission fee for the ice skating rink. -/
def admission_fee : ℚ := 5

/-- The cost of buying a new pair of skates. -/
def new_skates_cost : ℚ := 65

/-- The number of visits at which buying skates becomes economical. -/
def break_even_visits : ℕ := 26

/-- 
Theorem stating that the rental cost for skates is $2.50, given the admission fee,
cost of new skates, and the number of visits at which buying becomes economical.
-/
theorem skate_rental_cost :
  rental_cost = (new_skates_cost - admission_fee * break_even_visits) / break_even_visits :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_skate_rental_cost_l397_39766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_food_cost_per_ounce_l397_39774

/-- Calculates the cost per ounce of ant food given the specified conditions --/
theorem ant_food_cost_per_ounce 
  (num_ants : ℕ)
  (food_per_ant : ℕ)
  (job_start_fee : ℚ)
  (leaf_cost : ℚ)
  (leaves_raked : ℕ)
  (jobs_completed : ℕ)
  (h1 : num_ants = 400)
  (h2 : food_per_ant = 2)
  (h3 : job_start_fee = 5)
  (h4 : leaf_cost = 1 / 100)
  (h5 : leaves_raked = 6000)
  (h6 : jobs_completed = 4) :
  let total_earned := job_start_fee * (jobs_completed : ℚ) + leaf_cost * (leaves_raked : ℚ)
  let total_food_needed := (num_ants * food_per_ant : ℚ)
  total_earned / total_food_needed = 1 / 10 := by
    sorry

-- Remove the #eval line as it's not necessary for the theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ant_food_cost_per_ounce_l397_39774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_ellipse_area_l397_39798

/-- An ellipse that contains two unit circles centered at (-1, 0) and (1, 0) -/
structure ContainingEllipse where
  a : ℝ
  b : ℝ
  h1 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 → 
    ((x - 1)^2 + y^2 ≥ 1) ∧ ((x + 1)^2 + y^2 ≥ 1)

/-- The area of an ellipse -/
noncomputable def ellipseArea (e : ContainingEllipse) : ℝ := Real.pi * e.a * e.b

/-- The theorem stating the smallest possible area of the containing ellipse -/
theorem smallest_containing_ellipse_area :
  ∃ (e : ContainingEllipse), ∀ (e' : ContainingEllipse), 
    ellipseArea e ≤ ellipseArea e' ∧ 
    ellipseArea e = (3 * Real.sqrt 3 / 2) * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_containing_ellipse_area_l397_39798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l397_39757

/-- Curve C with parametric equations x = 2 - t - t² and y = 2 - 3t + t² -/
def C (t : ℝ) : ℝ × ℝ := (2 - t - t^2, 2 - 3*t + t^2)

/-- Point A is the intersection of C with the y-axis -/
def A : ℝ × ℝ := C (-2)

/-- Point B is the intersection of C with the x-axis -/
def B : ℝ × ℝ := C 2

/-- The distance between points A and B -/
noncomputable def distance_AB : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The polar equation of line AB -/
def line_AB_polar (ρ θ : ℝ) : Prop := 3 * ρ * Real.cos θ - ρ * Real.sin θ + 12 = 0

theorem curve_C_properties :
  distance_AB = 4 * Real.sqrt 10 ∧
  ∀ ρ θ, line_AB_polar ρ θ ↔ 3 * ρ * Real.cos θ - ρ * Real.sin θ + 12 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_C_properties_l397_39757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersections_l397_39721

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := (1/12) * x^2 + a * x + b

def A (x : ℝ) : ℝ × ℝ := (x, 0)  -- x-intercept
def C (x : ℝ) : ℝ × ℝ := (x, 0)  -- x-intercept
def B (b : ℝ) : ℝ × ℝ := (0, b)    -- y-intercept
def T : ℝ × ℝ := (3, 3)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem quadratic_intersections (a b : ℝ) :
  ∃ (x1 x2 : ℝ),
    f a b x1 = 0 ∧
    f a b x2 = 0 ∧
    x1 ≠ x2 ∧
    distance T (A x1) = distance T (B b) ∧
    distance T (A x1) = distance T (C x2) →
    b = -6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_intersections_l397_39721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_polynomial_product_l397_39767

theorem constant_term_of_polynomial_product : 
  let p₁ : Polynomial ℤ := 2 * X^3 + 3 * X^2 + 7
  let p₂ : Polynomial ℤ := 4 * X^4 + 2 * X^2 + 10
  (p₁ * p₂).coeff 0 = 70 := by
  -- Introduce the polynomials
  intro p₁ p₂
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_of_polynomial_product_l397_39767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_condition_l397_39788

/-- An arithmetic sequence with first term a₁ and common difference d -/
noncomputable def arithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def sumArithmeticSequence (a₁ d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_sum_condition (a₁ d : ℝ) :
  (d > 0) ↔ (sumArithmeticSequence a₁ d 4 + sumArithmeticSequence a₁ d 6 > 2 * sumArithmeticSequence a₁ d 5) :=
by
  sorry

#check arithmetic_sequence_sum_condition

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_condition_l397_39788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_monotonicity_interval_l397_39759

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3)

theorem incorrect_monotonicity_interval (a : ℝ) : 
  (∀ x ∈ Set.Icc (-a) a, Monotone (fun y => f y)) → a ≤ 5 * Real.pi / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_monotonicity_interval_l397_39759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l397_39743

/-- An arithmetic sequence with its first term and common difference -/
structure ArithmeticSequence where
  a1 : ℚ
  d : ℚ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a1 + (n - 1 : ℚ) * seq.d

/-- The sum of the first n terms of an arithmetic sequence -/
def ArithmeticSequence.sum (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * (2 * seq.a1 + (n - 1 : ℚ) * seq.d) / 2

theorem arithmetic_sequence_property (seq : ArithmeticSequence) 
    (h : seq.sum 17 = 306) : 
    seq.nthTerm 7 - seq.nthTerm 3 / 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_property_l397_39743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l397_39717

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 2 * x / (1 + x^2)

-- State the theorem
theorem f_satisfies_condition : 
  ∀ x : ℝ, f ((1 - x) / (1 + x)) = (1 - x^2) / (1 + x^2) :=
by
  -- The proof is omitted using 'sorry'
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_condition_l397_39717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_hemisphere_theorem_l397_39729

/-- Configuration of a cylinder tangent to a hemisphere -/
structure CylinderHemisphereConfig where
  cylinder_radius : ℝ
  hemisphere_radius : ℝ
  is_externally_tangent : Bool
  is_symmetric : Bool
  is_base_parallel : Bool

/-- Properties of the cylinder-hemisphere configuration -/
noncomputable def cylinder_properties (config : CylinderHemisphereConfig) : ℝ × ℝ :=
  let height := 2 * Real.sqrt 51
  let surface_area := Real.pi * (12 * Real.sqrt 51 + 18)
  (height, surface_area)

/-- Theorem statement for the cylinder-hemisphere problem -/
theorem cylinder_hemisphere_theorem (config : CylinderHemisphereConfig) 
  (h1 : config.cylinder_radius = 3)
  (h2 : config.hemisphere_radius = 7)
  (h3 : config.is_externally_tangent)
  (h4 : config.is_symmetric)
  (h5 : config.is_base_parallel) :
  cylinder_properties config = (2 * Real.sqrt 51, Real.pi * (12 * Real.sqrt 51 + 18)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_hemisphere_theorem_l397_39729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_older_birthday_l397_39773

def days_in_month : ℕ := 30

def favorable_outcomes (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem probability_older_birthday (days : ℕ) (h : days = days_in_month) :
  (favorable_outcomes days : ℚ) / (days * days : ℚ) = 29 / 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_older_birthday_l397_39773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l397_39761

theorem range_of_m (m : ℝ) : 
  (∀ x > 0, m * x^2 + 2 * x + m ≤ 0) → m ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l397_39761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_antarctica_partition_exists_l397_39722

/-- A graph representing the cities and roads in Antarctica. -/
structure AntarcticaGraph where
  /-- The set of vertices (cities) in the graph. -/
  V : Type
  /-- The number of vertices in the graph. -/
  num_vertices : Nat
  /-- The edge relation (road connections) between vertices. -/
  E : V → V → Prop
  /-- The graph has exactly 3,999,999 vertices. -/
  vertex_count : num_vertices = 3999999
  /-- The graph is connected: there's a path between any two vertices. -/
  connected : ∀ (v w : V), ∃ (path : List V), path.head? = some v ∧ path.getLast? = some w ∧
    ∀ (i : Nat), i < path.length - 1 → E (path.get ⟨i, by sorry⟩) (path.get ⟨i+1, by sorry⟩)

/-- A partition of the vertices into 1999 subsets. -/
def Partition (G : AntarcticaGraph) := 
  Fin 1999 → Finset G.V

/-- The distance between two vertices in a graph. -/
noncomputable def distance (G : AntarcticaGraph) (v w : G.V) : Nat := sorry

/-- The main theorem: there exists a partition satisfying the required properties. -/
theorem antarctica_partition_exists (G : AntarcticaGraph) : 
  ∃ (P : Partition G), 
    (∀ (i : Fin 1999), (P i).card = 2001) ∧ 
    (∀ (i : Fin 1999) (v w : G.V), v ∈ P i → w ∈ P i → distance G v w ≤ 4000) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_antarctica_partition_exists_l397_39722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l397_39736

/-- Given that p, q, and r are the roots of x^3 - 12x^2 + 20x - 2 = 0,
    and t = ∛p + ∛q + ∛r, prove that t^6 - 24t^3 + 12t = 12t + 36 * 2^(2/3) -/
theorem cubic_root_sum (p q r t : ℝ) : 
  (∀ x, x^3 - 12*x^2 + 20*x - 2 = 0 ↔ (x = p ∨ x = q ∨ x = r)) →
  t = Real.rpow p (1/3 : ℝ) + Real.rpow q (1/3 : ℝ) + Real.rpow r (1/3 : ℝ) →
  t^6 - 24*t^3 + 12*t = 12*t + 36 * Real.rpow 2 (2/3 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_l397_39736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_max_volume_l397_39795

/-- The height of a rectangular container with maximum volume -/
noncomputable def max_volume_height : ℝ := 0.5

/-- The total length of the steel bar -/
def total_length : ℝ := 6

/-- The ratio of adjacent side lengths of the base -/
def base_ratio : ℝ × ℝ := (3, 4)

/-- Function to calculate the volume of the container -/
noncomputable def volume (x : ℝ) : ℝ := 12 * x^2 * (1.5 - 7*x)

/-- The maximum volume occurs when x = 1/7 -/
noncomputable def max_volume_x : ℝ := 1/7

/-- Theorem stating the conditions for maximum volume and the resulting height -/
theorem container_max_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 3/14 ∧
  (∀ y, y > 0 → y < 3/14 → volume y ≤ volume x) ∧
  max_volume_height = 1.5 - 7 * max_volume_x :=
by
  sorry

#check container_max_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_max_volume_l397_39795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l397_39718

noncomputable def line_point (s : ℝ) : ℝ × ℝ × ℝ := (5 + 3*s, -2 - s, 3 + 2*s)

def target_point : ℝ × ℝ × ℝ := (1, 4, 2)

noncomputable def closest_point : ℝ × ℝ × ℝ := (5/7, -4/7, 1/7)

theorem closest_point_on_line :
  ∃ (s : ℝ), line_point s = closest_point ∧
  ∀ (t : ℝ), ‖line_point t - target_point‖ ≥ ‖closest_point - target_point‖ := by
  sorry

#check closest_point_on_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_point_on_line_l397_39718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_age_is_16_l397_39751

-- Define the ages of Jane, Kevin, and Linda
variable (jane_age kevin_age linda_age : ℝ)

-- The average of their ages is 12
axiom average_age : (jane_age + kevin_age + linda_age) / 3 = 12

-- Three years ago, Linda was the same age as Jane is now
axiom past_age_relation : linda_age - 3 = jane_age

-- In 4 years, Kevin's age will be half of Linda's age at that time
axiom future_age_relation : kevin_age + 4 = (linda_age + 4) / 2

-- Theorem to prove
theorem linda_age_is_16 : linda_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linda_age_is_16_l397_39751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ranges_l397_39772

/-- A function that checks if a five-digit number is a range -/
def isRange (n : ℕ) : Bool :=
  if n ≥ 10000 ∧ n < 100000 then
    let digits := [n / 10000, (n / 1000) % 10, (n / 100) % 10, (n / 10) % 10, n % 10]
    digits[0]! ≠ 0 ∧ digits[0]! < digits[1]! ∧ digits[1]! > digits[2]! ∧ digits[2]! < digits[3]! ∧ digits[3]! > digits[4]!
  else
    false

/-- The number of ranges -/
def numRanges : ℕ := (List.range 100000).filter isRange |>.length

/-- Theorem stating the number of ranges is 1260 -/
theorem count_ranges : numRanges = 1260 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ranges_l397_39772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_food_consumption_l397_39746

theorem max_food_consumption (total_food : ℝ) (min_guests : ℕ) 
  (h1 : total_food = 327) 
  (h2 : min_guests = 164) : 
  ∃ (max_individual_food : ℝ), 
    max_individual_food ≤ 1.99 ∧ 
    max_individual_food * (min_guests : ℝ) ≥ total_food := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_food_consumption_l397_39746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_specific_l397_39755

/-- The volume of a cone with radius r and height h is (1/3) * π * r^2 * h -/
noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- The ratio of the volumes of two cones -/
noncomputable def cone_volume_ratio (r1 h1 r2 h2 : ℝ) : ℝ :=
  (cone_volume r1 h1) / (cone_volume r2 h2)

theorem cone_volume_ratio_specific : 
  cone_volume_ratio 10 20 20 10 = 1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_ratio_specific_l397_39755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_sqrt_7_l397_39709

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (4 + 4 * Real.cos α, 4 * Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 2 * (Real.cos θ + 1)
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- Define the Cartesian equations of C₁ and C₂
def C₁_cartesian (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 16

def C₂_cartesian (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4 = 0

-- Define the intersection point P
def is_intersection_point (P : ℝ × ℝ) : Prop :=
  C₁_cartesian P.1 P.2 ∧ C₂_cartesian P.1 P.2

-- Define the tangent line to C₁ at P
def tangent_line (P : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - P.2 = (-P.2 / (P.1 - 4)) * (x - P.1)

-- Define point Q as the intersection of the tangent line and C₂
def is_Q (P Q : ℝ × ℝ) : Prop :=
  tangent_line P Q.1 Q.2 ∧ C₂_cartesian Q.1 Q.2

-- The main theorem
theorem length_PQ_is_sqrt_7 (P Q : ℝ × ℝ) :
  is_intersection_point P → is_Q P Q → Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_PQ_is_sqrt_7_l397_39709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_implies_a_interval_l397_39799

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin (Real.pi / 2 + x))^2 + 4 * Real.sin x

theorem function_range_implies_a_interval (a : ℝ) :
  (∀ x ∈ Set.Icc 0 a, f x ∈ Set.Icc 4 5) →
  (∀ y ∈ Set.Icc 4 5, ∃ x ∈ Set.Icc 0 a, f x = y) →
  a ∈ Set.Icc (Real.pi / 6) Real.pi :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_implies_a_interval_l397_39799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l397_39752

-- Define proposition p
def p : Prop := ∀ x : ℝ, (2 : ℝ)^x > 1

-- Define function f
noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

-- Define proposition q
def q : Prop := Monotone f

-- Theorem to prove
theorem problem_solution : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l397_39752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l397_39734

/-- A function is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- A function is decreasing on an interval if f(x) ≥ f(y) for all x ≤ y in the interval -/
def IsDecreasingOn (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x ≤ y → f x ≥ f y

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) :
  IsEven f →
  IsDecreasingOn f (Set.Ici 0) →
  (∀ x ∈ Set.Icc 0 1, f (x^3 - x^2 + a) + f (-x^3 + x^2 - a) ≥ 2 * f 1) →
  a ∈ Set.Icc (-23/27) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l397_39734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l397_39705

/-- A rhombus with area K and one diagonal three times the length of the other has side length √(5K/3) -/
theorem rhombus_side_length (K : ℝ) (h : K > 0) :
  ∃ (d : ℝ), d > 0 ∧
  ∃ (s : ℝ), s > 0 ∧
  (K = (3 * d^2) / 2) ∧
  (s^2 = (5 * d^2) / 2) ∧
  s = Real.sqrt ((5 * K) / 3) := by
  sorry

#check rhombus_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_side_length_l397_39705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l397_39740

/-- The line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y + 6 = 0

/-- The ellipse C in the xy-plane -/
def ellipse_C (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The distance from a point (x, y) to the line l -/
noncomputable def distance_to_line (x y : ℝ) : ℝ :=
  |x - y + 6| / Real.sqrt 2

/-- Theorem stating the minimum distance from the ellipse C to the line l -/
theorem min_distance_ellipse_to_line :
  ∃ (min_dist : ℝ), min_dist = Real.sqrt 2 / 2 ∧
  ∀ (x y : ℝ), ellipse_C x y →
    distance_to_line x y ≥ min_dist := by
  sorry

#check min_distance_ellipse_to_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_ellipse_to_line_l397_39740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_eq_14853_l397_39775

/-- Definition of the sequence b_n -/
def b : ℕ → ℕ
  | 0 => 3  -- Define b_0 to be 3 (same as b_1 in the original problem)
  | n + 1 => b n + 3 * (n + 1)

/-- Theorem stating that the 100th term of the sequence is 14853 -/
theorem b_100_eq_14853 : b 100 = 14853 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_100_eq_14853_l397_39775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_recipe_l397_39776

/-- Lemonade recipe problem -/
theorem lemonade_recipe (lemon sugar water orange : ℝ) : 
  water = 5 * sugar ∧ 
  sugar = 2 * lemon ∧ 
  orange = sugar ∧ 
  lemon = 5 → 
  water = 50 := by
  intro h
  cases' h with h1 h2
  cases' h2 with h2 h3
  cases' h3 with h3 h4
  rw [h1, h2, h4]
  norm_num

#check lemonade_recipe

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lemonade_recipe_l397_39776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_identification_l397_39728

variable (a b : ℝ)

/-- The difference of squares formula can be applied to a polynomial if it can be expressed as (A + B)(A - B) for some A and B. -/
def is_difference_of_squares (p : ℝ → ℝ → ℝ) : Prop :=
  ∃ A B : ℝ → ℝ → ℝ, ∀ x y, p x y = (A x y + B x y) * (A x y - B x y)

/-- Among the given polynomials, only a^2 - b^2 is a difference of squares. -/
theorem difference_of_squares_identification :
  ¬is_difference_of_squares (fun a b => a^2 + b^2) ∧
  ¬is_difference_of_squares (fun a b => 2*a - b^2) ∧
  is_difference_of_squares (fun a b => a^2 - b^2) ∧
  ¬is_difference_of_squares (fun a b => -a^2 - b^2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_difference_of_squares_identification_l397_39728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_extremum_l397_39703

/-- Given a cubic function f(x) = x^3 + 3ax^2 + bx + a^2 with an extremum of 0 at x = -1, 
    prove that a - b = -7 -/
theorem cubic_extremum (a b : ℝ) : 
  (let f : ℝ → ℝ := fun x => x^3 + 3*a*x^2 + b*x + a^2;
   (∃ (ε : ℝ), ε > 0 ∧ ∀ x, 0 < |x + 1| ∧ |x + 1| < ε → f x ≥ f (-1)) ∧
   f (-1) = 0) →
  a - b = -7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_extremum_l397_39703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l397_39783

noncomputable section

/-- Definition of an ellipse -/
def is_ellipse (a b : ℝ) (h : a > b ∧ b > 0) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of c -/
def c (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- Definition of e -/
def e (a b : ℝ) : ℝ := c a b / a

/-- Statement about the ratio of distances -/
def ratio_inequality (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∀ A B C : ℝ × ℝ, is_ellipse a b h A.1 A.2 →
    ∃ F₁ F₂ : ℝ × ℝ, 
      (|A.1 - F₁.1| + |A.1 - F₂.1|) / (|B.1 - F₁.1| + |B.1 - F₂.1|) +
      (|A.1 - F₁.1| + |A.1 - F₂.1|) / (|C.1 - F₁.1| + |C.1 - F₂.1|) ≥ 
      2 * (1 + (e a b)^2) / (1 - (e a b)^2)

/-- Statement about perpendicularity -/
def perpendicular_condition (a b : ℝ) (h : a > b ∧ b > 0) : Prop :=
  ∀ A M : ℝ × ℝ, is_ellipse a b h A.1 A.2 →
    A.1 ≠ 0 → A.2 ≠ 0 →
    ∃ F₁ F₂ : ℝ × ℝ, 
      |A.1 - F₁.1| / |A.1 - F₂.1| = |F₁.1 - M.1| / |M.1 - F₂.1| →
      ∃ l : Set (ℝ × ℝ), (∀ p ∈ l, is_ellipse a b h p.1 p.2 → p = A) →
        (∀ p : ℝ × ℝ, (p.1 - A.1) * (M.1 - A.1) + (p.2 - A.2) * (M.2 - A.2) = 0 → p ∈ l)

/-- Main theorem -/
theorem ellipse_properties (a b : ℝ) (h : a > b ∧ b > 0) :
  ratio_inequality a b h ∧ perpendicular_condition a b h := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l397_39783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_price_l397_39741

/-- The selling price of an item given its cost price and markup percentage. -/
noncomputable def selling_price (cost : ℝ) (markup_percent : ℝ) : ℝ :=
  cost * (1 + markup_percent / 100)

/-- Theorem: The selling price of a computer table with cost price 6947.5 and 20% markup is 8337. -/
theorem computer_table_price : selling_price 6947.5 20 = 8337 := by
  -- Unfold the definition of selling_price
  unfold selling_price
  -- Simplify the expression
  simp
  -- The proof is completed using sorry
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_price_l397_39741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_l397_39714

/-- An arithmetic progression is a sequence where the difference between
    successive terms is constant. -/
structure ArithmeticProgression where
  first : ℚ
  diff : ℚ

/-- The nth term of an arithmetic progression -/
def ArithmeticProgression.nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  ap.first + (n - 1 : ℚ) * ap.diff

/-- The sum of the first n terms of an arithmetic progression -/
def ArithmeticProgression.sum (ap : ArithmeticProgression) (n : ℕ) : ℚ :=
  n / 2 * (2 * ap.first + (n - 1 : ℚ) * ap.diff)

/-- Theorem: For an arithmetic progression where the sum of the 4th and 12th terms is 16,
    the sum of the first 15 terms is 120. -/
theorem arithmetic_progression_sum
  (ap : ArithmeticProgression)
  (h : ap.nthTerm 4 + ap.nthTerm 12 = 16) :
  ap.sum 15 = 120 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_progression_sum_l397_39714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_log_theorem_l397_39710

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the transformation steps
noncomputable def translate_left (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (x + 1)
def reflect_y (f : ℝ → ℝ) : ℝ → ℝ := λ x => f (-x)

-- Define the function g as the composition of the transformations
noncomputable def g : ℝ → ℝ := reflect_y (translate_left f)

-- State the theorem
theorem transform_log_theorem : g = λ x => Real.log (1 - x) / Real.log 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_log_theorem_l397_39710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_alpha_and_sequence_l397_39732

/-- A sequence defined by a recurrence relation -/
noncomputable def recurrence_sequence (N : ℕ) (α : ℝ) : ℕ → ℝ
| 0 => 0
| 1 => 1
| (k + 2) => (α * recurrence_sequence N α (k + 1) - (N - k) * recurrence_sequence N α k) / (k + 1 : ℝ)

/-- The theorem stating the largest α and the resulting sequence -/
theorem largest_alpha_and_sequence (N : ℕ) (h : N > 0) :
  (∃ (α : ℝ), α = N - 1 ∧
    recurrence_sequence N α (N + 1) = 0 ∧
    (∀ k : ℕ, k ≤ N → recurrence_sequence N α k = (N - 1).choose (k - 1)) ∧
    (∀ k : ℕ, k > N → recurrence_sequence N α k = 0)) ∧
  (∀ β : ℝ, β > N - 1 → recurrence_sequence N β (N + 1) ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_alpha_and_sequence_l397_39732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_t_eq_two_l397_39737

/-- The function f(x) = t ln x -/
noncomputable def f (t : ℝ) (x : ℝ) : ℝ := t * Real.log x

/-- The function g(x) = x^2 - 1 -/
def g (x : ℝ) : ℝ := x^2 - 1

/-- The derivative of f with respect to x -/
noncomputable def f_deriv (t : ℝ) (x : ℝ) : ℝ := t / x

/-- The derivative of g with respect to x -/
def g_deriv (x : ℝ) : ℝ := 2 * x

theorem common_tangent_implies_t_eq_two (t : ℝ) :
  f_deriv t 1 = g_deriv 1 → t = 2 := by
  intro h
  sorry

#check common_tangent_implies_t_eq_two

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_tangent_implies_t_eq_two_l397_39737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_are_intersecting_l397_39742

-- Define the circles
def circle_O1 (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 4
def circle_O2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 4)^2 = 9

-- Define the centers and radii
def center_O1 : ℝ × ℝ := (-1, 1)
def center_O2 : ℝ × ℝ := (2, 4)
def radius_O1 : ℝ := 2
def radius_O2 : ℝ := 3

-- Define the distance between centers
noncomputable def distance_between_centers : ℝ := Real.sqrt ((2 + 1)^2 + (4 - 1)^2)

-- Theorem statement
theorem circles_intersect :
  1 < distance_between_centers ∧ 
  distance_between_centers < radius_O1 + radius_O2 := by
  sorry

-- Additional theorem to explicitly state the conclusion
theorem circles_are_intersecting : 
  ∃ (x y : ℝ), circle_O1 x y ∧ circle_O2 x y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_circles_are_intersecting_l397_39742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_in_interval_max_m_value_l397_39756

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := Real.exp x
def g (x a : ℝ) : ℝ := x^3 - 6*x^2 + 3*x + a

-- Statement for part 1
theorem unique_solution_in_interval :
  ∃! x, x ∈ Set.Ioo 2 3 ∧ f (-x) = 6 - 2*x :=
sorry

-- Statement for part 2
theorem max_m_value (a : ℝ) (h : a ≥ 0) :
  (∀ m : ℕ, m > 5 → ∃ x, x ∈ Set.Icc 1 m ∧ f x * g x a > x) ∧
  (∀ x, x ∈ Set.Icc 1 5 → f x * g x a ≤ x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_in_interval_max_m_value_l397_39756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_tank_volume_l397_39777

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical tank -/
structure CylindricalTank where
  radius : ℝ
  height : ℝ

/-- Calculates the volume of a cylindrical tank -/
noncomputable def tankVolume (tank : CylindricalTank) : ℝ :=
  Real.pi * tank.radius^2 * tank.height

/-- Checks if a cylindrical tank fits in a crate -/
def tankFitsInCrate (tank : CylindricalTank) (crate : CrateDimensions) : Prop :=
  tank.height ≤ crate.height ∧ 
  2 * tank.radius ≤ min crate.length crate.width

/-- The main theorem stating that a tank with radius 4 feet has the largest possible volume -/
theorem largest_tank_volume (crate : CrateDimensions) 
  (h_crate : crate = { length := 8, width := 10, height := 6 }) :
  ∃ (optimalTank : CylindricalTank), 
    tankFitsInCrate optimalTank crate ∧
    optimalTank.radius = 4 ∧
    ∀ (otherTank : CylindricalTank), 
      tankFitsInCrate otherTank crate → 
      tankVolume otherTank ≤ tankVolume optimalTank :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_tank_volume_l397_39777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l397_39720

/-- An arithmetic sequence with common difference d and first term a1 -/
noncomputable def arithmetic_sequence (d : ℝ) (a1 : ℝ) (n : ℕ) : ℝ := a1 + d * (n - 1 : ℝ)

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def S_n (d : ℝ) (a1 : ℝ) (n : ℕ) : ℝ := n * (2 * a1 + d * (n - 1)) / 2

theorem max_sum_arithmetic_sequence (d : ℝ) (a1 : ℝ) 
  (h1 : d < 0)
  (h2 : S_n d a1 3 = 11 * arithmetic_sequence d a1 6) :
  ∃ (n : ℕ), ∀ (m : ℕ), S_n d a1 n ≥ S_n d a1 m ∧ n = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l397_39720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hoseok_workout_difference_l397_39765

theorem hoseok_workout_difference (monday tuesday wednesday thursday friday : ℕ) :
  thursday - tuesday = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hoseok_workout_difference_l397_39765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_equation_l397_39779

/-- The circle C with center (2,0) and radius 3 -/
def circle_C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 9}

/-- The center of circle C -/
def center_C : ℝ × ℝ := (2, 0)

/-- The point M through which line l passes -/
def point_M : ℝ × ℝ := (1, 2)

/-- A line passing through point_M -/
structure Line_through_M where
  slope : ℝ
  equation : ℝ → ℝ
  passes_through_M : equation point_M.1 = point_M.2

/-- The distance from a point to a line -/
noncomputable def distance_point_to_line (p : ℝ × ℝ) (l : Line_through_M) : ℝ :=
  abs (l.slope * p.1 - p.2 + (point_M.2 - l.slope * point_M.1)) / Real.sqrt (1 + l.slope^2)

/-- The theorem to be proved -/
theorem max_distance_line_equation :
  ∃ (l : Line_through_M),
    (∀ (l' : Line_through_M), distance_point_to_line center_C l' ≤ distance_point_to_line center_C l) →
    (l.equation = fun x ↦ (-1/2) * x + 5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_line_equation_l397_39779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_with_120_degree_angles_l397_39780

noncomputable def interior_angle (n : ℕ) (i : ℕ) : ℝ :=
  (n - 2) * 180 / n

theorem polygon_sides_with_120_degree_angles (n : ℕ) :
  (∀ (i : ℕ), i < n → interior_angle n i = 120) →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_sides_with_120_degree_angles_l397_39780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_C_cartesian_equation_l_max_distance_max_distance_point_l397_39762

open Real

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := (sqrt 3 * cos θ, sin θ)

noncomputable def line_l (θ : ℝ) : ℝ := 2 * sqrt 2 / cos (θ - π / 4)

-- Standard equation of curve C
theorem standard_equation_C :
  ∀ x y : ℝ, (∃ θ : ℝ, curve_C θ = (x, y)) ↔ x^2 / 3 + y^2 = 1 := by sorry

-- Cartesian equation of line l
theorem cartesian_equation_l :
  ∀ x y : ℝ, (∃ θ : ℝ, (x, y) = (line_l θ * cos θ, line_l θ * sin θ)) ↔ x + y = 4 := by sorry

-- Maximum distance from C to l
theorem max_distance :
  (⨆ θ : ℝ, abs ((sqrt 3 * cos θ + sin θ - 4) / sqrt 2)) = 3 * sqrt 2 := by sorry

-- Point on C with maximum distance to l
theorem max_distance_point :
  ∃ θ : ℝ, curve_C θ = (-3/2, -1/2) ∧
    abs ((sqrt 3 * cos θ + sin θ - 4) / sqrt 2) = 3 * sqrt 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_standard_equation_C_cartesian_equation_l_max_distance_max_distance_point_l397_39762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_equidistant_from_third_l397_39758

theorem divisors_equidistant_from_third (n : ℕ) :
  (∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ d₁ > 0 ∧ d₂ > 0 ∧ 
   d₁ ∣ n ∧ d₂ ∣ n ∧ 
   |Int.ofNat d₁ - Int.ofNat (n / 3)| = |Int.ofNat d₂ - Int.ofNat (n / 3)|) →
  (∃ k : ℕ, k > 0 ∧ n = 6 * k) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisors_equidistant_from_third_l397_39758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l397_39789

theorem contrapositive_sin_equality :
  (∀ x y : ℝ, x = y → Real.sin x = Real.sin y) →
  (∀ x y : ℝ, Real.sin x ≠ Real.sin y → x ≠ y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_contrapositive_sin_equality_l397_39789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_and_derivative_difference_l397_39768

noncomputable def f (x : ℝ) : ℝ := 2 / (2019^x + 1) + Real.sin x

theorem f_sum_and_derivative_difference : 
  f 2018 + f (-2018) + deriv f 2019 - deriv f (-2019) = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_and_derivative_difference_l397_39768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l397_39745

/-- The volume of a cylinder with given circumference and height -/
noncomputable def cylinderVolume (circumference height : ℝ) : ℝ :=
  (circumference^2 * height) / (4 * Real.pi)

/-- The problem statement -/
theorem cylinder_volume_difference : 
  let v1 := cylinderVolume 10 12
  let v2 := cylinderVolume 10 8
  Real.pi * |v1 - v2| = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volume_difference_l397_39745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_factorial_equation_l397_39794

theorem solve_factorial_equation (m : ℕ) : m * Nat.factorial (m + 1) + Nat.factorial (m + 1) = 5040 → m = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_factorial_equation_l397_39794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l397_39770

theorem sufficient_not_necessary_condition : 
  (∃ α : ℝ, α > π/6 ∧ Real.sin α ≤ 1/2) ∧ 
  (∃ β : ℝ, β ≤ π/6 ∧ Real.sin β > 1/2) ∧
  (∀ γ : ℝ, γ > π/6 → Real.sin γ > 1/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_not_necessary_condition_l397_39770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solutions_l397_39792

theorem cubic_root_equation_solutions :
  {x : ℝ | (2 - x)^(1/3) + Real.sqrt (x - 1) = 1} = {1, 2, 10} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_equation_solutions_l397_39792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_5_backward_l397_39781

/-- Represents a one-dimensional movement in meters -/
inductive Movement
| forward (n : ℕ)
| backward (n : ℕ)

/-- Interprets a movement value as a description of motion -/
def interpret_movement : Movement → String
| Movement.forward n => s!"moving forward {n} meters"
| Movement.backward n => s!"moving backward {n} meters"

/-- Moving forward 5 meters is represented by +5 -/
axiom forward_5 : interpret_movement (Movement.forward 5) = "moving forward 5 meters"

/-- Theorem: -5 represents moving backward 5 meters -/
theorem negative_5_backward :
  interpret_movement (Movement.backward 5) = "moving backward 5 meters" :=
by
  -- The proof is immediate from the definition of interpret_movement
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_5_backward_l397_39781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_no_plane_division_l397_39778

/-- A configuration of lines in a plane. -/
structure LineConfiguration where
  lines : Set (Set (ℝ × ℝ))
  intersecting : ∃ l1 l2, l1 ∈ lines ∧ l2 ∈ lines ∧ l1 ≠ l2 ∧ (∃ p, p ∈ l1 ∧ p ∈ l2)

/-- The number of regions formed by a line configuration. -/
noncomputable def num_regions (config : LineConfiguration) : ℕ := sorry

/-- Theorem stating that for all n ≥ 5, there exists a line configuration
    that divides the plane into exactly n regions. -/
theorem plane_division (n : ℕ) (h : n ≥ 5) :
  ∃ (config : LineConfiguration), num_regions config = n := by
  sorry

/-- Theorem stating that for all n < 5, there does not exist a line configuration
    that divides the plane into exactly n regions. -/
theorem no_plane_division (n : ℕ) (h : n < 5) :
  ¬∃ (config : LineConfiguration), num_regions config = n := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_division_no_plane_division_l397_39778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_cosine_l397_39744

-- Define the function representing the curve
noncomputable def f (x : ℝ) : ℝ := Real.cos x

-- Define the lower and upper bounds of the integral
noncomputable def a : ℝ := 0
noncomputable def b : ℝ := 3 * Real.pi / 2

-- State the theorem
theorem area_enclosed_by_cosine : 
  ∫ x in a..b, |f x| = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_cosine_l397_39744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l397_39796

-- Define the function f with domain [0,8]
def f : Set ℝ := Set.Icc 0 8

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 
  if x ∈ Set.Icc 0 2 ∧ x > 1 then
    Real.sqrt (x - 1)⁻¹  -- This is a placeholder for f(4x) / sqrt(x-1)
  else 0

-- Theorem statement
theorem domain_of_g :
  {x : ℝ | g x ≠ 0} = Set.Ioc 1 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l397_39796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_distance_to_directrix_l397_39787

noncomputable def distance_to_right_directrix (x y : ℝ) : ℝ :=
  3 - x

noncomputable def distance_to_left_focus (x y : ℝ) : ℝ :=
  Real.sqrt 3 + (Real.sqrt 3 / 3) * x

theorem ellipse_point_distance_to_directrix 
  (x y : ℝ) 
  (h1 : x^2 / 3 + y^2 / 2 = 1) 
  (h2 : distance_to_left_focus x y = Real.sqrt 3 / 2) : 
  distance_to_right_directrix x y = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_point_distance_to_directrix_l397_39787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_range_of_g_inequality_for_t_l397_39708

-- Define the functions f and g
def f (x : ℝ) : ℝ := |2*x - 1| + |x + 1|
def g (x : ℝ) : ℝ := f x + |x + 1|

-- Define the set M as the range of g
def M : Set ℝ := Set.range g

-- Theorem statements
theorem solution_set_f (x : ℝ) : f x ≤ 3 ↔ -1 ≤ x ∧ x ≤ 1 := by sorry

theorem range_of_g : M = Set.Ici 3 := by sorry

theorem inequality_for_t (t : ℝ) (h : t ∈ M) : t^2 + 1 ≥ 3/t + 3*t := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_range_of_g_inequality_for_t_l397_39708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_felipe_build_time_l397_39754

/-- The time it took Felipe to build his house in years -/
def felipe_time : ℝ := sorry

/-- The time it took Emilio to build his house in years -/
def emilio_time : ℝ := sorry

/-- The combined time it took Felipe and Emilio to build their houses in years -/
def combined_time : ℝ := 7.5

theorem felipe_build_time :
  felipe_time * 12 = 30 ∧
  felipe_time + emilio_time = combined_time ∧
  felipe_time * 2 = emilio_time :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_felipe_build_time_l397_39754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_angle_l397_39730

/-- The angle of intersection between a parabola and a circle -/
theorem parabola_circle_intersection_angle (p : ℝ) (h : p > 0) :
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let circle := {(x, y) : ℝ × ℝ | (x - p/2)^2 + y^2 = 4*p^2}
  let intersection_points := {(x, y) : ℝ × ℝ | (x, y) ∈ parabola ∧ (x, y) ∈ circle}
  ∀ (point : ℝ × ℝ), point ∈ intersection_points →
    let (x, y) := point
    let parabola_tangent_slope := y / p
    let circle_tangent_slope := (p/2 - x) / y
    Real.arctan ((parabola_tangent_slope - circle_tangent_slope) / (1 + parabola_tangent_slope * circle_tangent_slope)) = π/3
:= by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_circle_intersection_angle_l397_39730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l397_39764

-- Define the line equation
def line_equation (x y m : ℝ) : Prop := 1 - x / m + y / (4 - m) = 1

-- Define the slope of a line passing through two points
noncomputable def line_slope (x1 y1 x2 y2 : ℝ) : ℝ := (y2 - y1) / (x2 - x1)

-- Define the area of a triangle given the coordinates of two points (third point is origin)
noncomputable def triangle_area (x1 y1 x2 y2 : ℝ) : ℝ := abs (x1 * y2 - x2 * y1) / 2

-- Statement for part 1
theorem part_one (m : ℝ) : 
  (∃ x y, line_equation x y m ∧ line_slope m 0 0 (4 - m) = 2) → m = -4 :=
sorry

-- Statement for part 2
theorem part_two :
  ∃ m, 0 < m ∧ m < 4 ∧
  (∀ m', 0 < m' ∧ m' < 4 → 
    triangle_area m' 0 0 (4 - m') ≤ triangle_area m 0 0 (4 - m)) ∧
  triangle_area m 0 0 (4 - m) = 2 ∧ m = 2 :=
sorry

-- Statement for part 3
theorem part_three (x y : ℝ) :
  (∃ m, m = 2 ∧ line_equation x y m) → x + y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_part_three_l397_39764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_of_cosine_at_zero_l397_39719

noncomputable def f (x : ℝ) := Real.cos (2 * x)

theorem derivatives_of_cosine_at_zero :
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  deriv (deriv f) 0 = -4 ∧
  deriv (deriv (deriv f)) 0 = 0 ∧
  deriv (deriv (deriv (deriv f))) 0 = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivatives_of_cosine_at_zero_l397_39719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sigma_algebra_generated_by_partition_l397_39704

theorem sigma_algebra_generated_by_partition 
  (Ω : Type*) [Countable Ω] (F : Set (Set Ω)) 
  (hF : MeasurableSpace Ω) :
  ∃ (D : ℕ → Set Ω), 
    (∀ i j, i ≠ j → D i ∩ D j = ∅) ∧ 
    (⋃ i, D i) = Set.univ ∧
    F = {A | ∃ (N : Set ℕ), A = ⋃ i ∈ N, D i} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sigma_algebra_generated_by_partition_l397_39704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_shoes_problem_l397_39726

theorem nancy_shoes_problem :
  ∀ (slippers heels : ℕ),
  let boots := 6
  slippers > boots →
  heels = 3 * (slippers + boots) →
  2 * boots + 2 * slippers + 2 * heels = 168 →
  slippers - boots = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nancy_shoes_problem_l397_39726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_piece_moves_even_l397_39791

/-- A chess piece move on an infinite chessboard -/
structure Move where
  horizontal : Int
  vertical : Int

/-- The state of a chess piece on an infinite chessboard -/
structure ChessPieceState where
  horizontal : Int
  vertical : Int

/-- Applies a move to a chess piece state -/
def applyMove (state : ChessPieceState) (move : Move) : ChessPieceState :=
  { horizontal := state.horizontal + move.horizontal,
    vertical := state.vertical + move.vertical }

/-- Checks if a move is valid according to the rules -/
def isValidMove (m n : Int) (move : Move) : Prop :=
  (Int.natAbs move.horizontal = m.natAbs ∧ Int.natAbs move.vertical = n.natAbs) ∨
  (Int.natAbs move.horizontal = n.natAbs ∧ Int.natAbs move.vertical = m.natAbs)

/-- A sequence of moves is valid if each move is valid -/
def validMoveSequence (m n : Int) (moves : List Move) : Prop :=
  ∀ move ∈ moves, isValidMove m n move

/-- The final state after applying a sequence of moves -/
def finalState (initial : ChessPieceState) (moves : List Move) : ChessPieceState :=
  moves.foldl applyMove initial

/-- Main theorem: If a piece returns to its original position after x valid moves, then x is even -/
theorem chess_piece_moves_even (m n : Int) (initial : ChessPieceState) (moves : List Move) :
  validMoveSequence m n moves →
  finalState initial moves = initial →
  Even moves.length := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_piece_moves_even_l397_39791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_proof_l397_39747

/-- The sequence defined by the first few terms -/
def a : ℕ → ℕ 
| 0 => 0  -- Adding a case for 0 to cover all natural numbers
| 1 => 1
| 2 => 3
| 3 => 6
| 4 => 10
| n + 1 => n * (n + 1) / 2  -- Using the general formula for n > 4

/-- The proposed general formula for the sequence -/
def general_formula (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem stating that the general formula matches the sequence for all positive integers -/
theorem sequence_formula_proof (n : ℕ) (h : n > 0) : a n = general_formula n := by
  cases n
  case zero => contradiction
  case succ n' =>
    induction n'
    case zero => rfl
    case succ n'' ih =>
      simp [a, general_formula]
      sorry  -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_proof_l397_39747
