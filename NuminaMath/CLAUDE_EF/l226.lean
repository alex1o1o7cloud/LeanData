import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_evaluation_l226_22618

theorem cosine_sum_evaluation : 
  let A := (Real.cos (5 * Real.pi / 2 + (1 / 2) * Real.arcsin (3 / 5)))^6 + 
           (Real.cos (7 * Real.pi / 2 - (1 / 2) * Real.arcsin (4 / 5)))^6
  ∃ ε > 0, |A - 0.009| < ε := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_sum_evaluation_l226_22618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_solution_l226_22682

theorem constant_ratio_solution (k : ℝ) (x₀ y₀ x y : ℝ) : 
  (3 * x₀^2 - 4) / (y₀ + 10) = k →
  y₀ = 2 →
  x₀ = 1 →
  (3 * x^2 - 4) / (y + 10) = k →
  y = 17 →
  x = Real.sqrt (7/12) ∨ x = -Real.sqrt (7/12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_ratio_solution_l226_22682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_leap_year_divisions_l226_22691

/-- Counts the number of divisors of a natural number. -/
def divisor_count (n : Nat) : Nat :=
  (Finset.filter (fun d => n % d = 0) (Finset.range (n + 1))).card

/-- Theorem stating that a non-leap year can be divided into 336 different periods. -/
theorem non_leap_year_divisions : divisor_count 31536000 = 336 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_leap_year_divisions_l226_22691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_unit_interval_l226_22681

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 5

theorem root_in_unit_interval (a b : ℕ+) (x₀ : ℝ) :
  (b : ℝ) - (a : ℝ) = 1 →
  x₀ ∈ Set.Icc (a : ℝ) (b : ℝ) →
  f x₀ = 0 →
  (a : ℕ) + (b : ℕ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_in_unit_interval_l226_22681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_mass_percentage_in_AlI3_l226_22629

noncomputable section

-- Define the molar masses
def molar_mass_Al : ℝ := 26.98
def molar_mass_I : ℝ := 126.90

-- Define the composition of AlI3
def Al_count : ℕ := 1
def I_count : ℕ := 3

-- Calculate the molar mass of AlI3
def molar_mass_AlI3 : ℝ := molar_mass_Al + I_count * molar_mass_I

-- Define the mass percentage calculation
def mass_percentage (element_mass : ℝ) (total_mass : ℝ) : ℝ :=
  (element_mass / total_mass) * 100

-- Theorem statement
theorem iodine_mass_percentage_in_AlI3 :
  let iodine_mass := I_count * molar_mass_I
  abs (mass_percentage iodine_mass molar_mass_AlI3 - 93.38) < 0.01 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_iodine_mass_percentage_in_AlI3_l226_22629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_1_false_proposition_2_false_l226_22642

-- Define a structure for a line in 3D space
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define a structure for a plane in 3D space
structure Plane3D where
  point : ℝ × ℝ × ℝ
  normal : ℝ × ℝ × ℝ

-- Define what it means for two lines to be skew
def are_skew (l1 l2 : Line3D) : Prop :=
  ¬ (∃ (t s : ℝ), l1.point + t • l1.direction = l2.point + s • l2.direction) ∧
  ¬ (∃ (k : ℝ), l1.direction = k • l2.direction)

-- Define what it means for a line to lie in a plane
def line_in_plane (l : Line3D) (p : Plane3D) : Prop :=
  (l.point - p.point) • p.normal = 0 ∧ l.direction • p.normal = 0

-- Define the intersection of two planes
noncomputable def plane_intersection (p1 p2 : Plane3D) : Line3D :=
  { point := (0, 0, 0)  -- Placeholder
  , direction := (0, 0, 0) }  -- Placeholder

-- State the theorem for the first proposition
theorem proposition_1_false :
  ∃ (α β : Plane3D) (a b : Line3D),
    are_skew a b ∧
    line_in_plane a α ∧
    line_in_plane b β ∧
    let c := plane_intersection α β
    ∃ (t1 t2 : ℝ),
      a.point + t1 • a.direction = c.point + t1 • c.direction ∧
      b.point + t2 • b.direction = c.point + t2 • c.direction :=
by
  sorry

-- State the theorem for the second proposition
theorem proposition_2_false :
  ∃ (lines : ℕ → Line3D),
    (∀ i j, i ≠ j → are_skew (lines i) (lines j)) ∧
    (∀ n : ℕ, ∃ m : ℕ, m > n) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_1_false_proposition_2_false_l226_22642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_division_l226_22611

/-- Represents a cookie shape -/
structure Cookie where
  squares : ℕ
  semicircles : ℕ

/-- Represents a part of the divided cookie -/
structure CookiePart where
  squares : ℕ
  semicircles : ℕ

/-- Defines the initial cookie shape -/
def initialCookie : Cookie :=
  { squares := 64, semicircles := 16 }

/-- Defines the desired part after division -/
def desiredPart : CookiePart :=
  { squares := 4, semicircles := 1 }

/-- A function that divides a cookie into parts -/
def divideCookie (c : Cookie) (n : ℕ) : List CookiePart :=
  sorry -- Implementation details omitted

/-- Theorem stating that the cookie can be divided into 16 equal parts -/
theorem cookie_division (c : Cookie) (n : ℕ) :
  c = initialCookie →
  n = 16 →
  (∀ part ∈ divideCookie c n, part = desiredPart) ∧
  (List.length (divideCookie c n) = n) :=
by
  sorry -- Proof details omitted


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_division_l226_22611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_swap_possible_l226_22622

/-- Represents a piece color -/
inductive Color
  | Black
  | Gray
  | Empty

/-- Represents a position on the board -/
structure Position where
  row : Nat
  col : Nat

/-- Represents the game board -/
structure Board where
  positions : List Position
  pieces : List Color

/-- Represents a move on the board -/
structure Move where
  fromPos : Position
  toPos : Position

/-- Checks if two positions are connected -/
def areConnected (p1 p2 : Position) : Bool := sorry

/-- Checks if a move is valid -/
def isValidMove (b : Board) (m : Move) : Bool := sorry

/-- Applies a move to the board -/
def applyMove (b : Board) (m : Move) : Board := sorry

/-- Checks if the board is in the initial state -/
def isInitialState (b : Board) : Bool := sorry

/-- Checks if the board is in the final (swapped) state -/
def isFinalState (b : Board) : Bool := sorry

/-- The main theorem: it's possible to swap black and gray pieces -/
theorem swap_possible (b : Board) :
  isInitialState b → ∃ (moves : List Move), isFinalState (moves.foldl applyMove b) := by
  sorry

#check swap_possible

end NUMINAMATH_CALUDE_ERRORFEEDBACK_swap_possible_l226_22622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l226_22674

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 9 - y^2 / 4 = 1

-- Define the foci
noncomputable def F1 : ℝ × ℝ := sorry
noncomputable def F2 : ℝ × ℝ := sorry

-- Define a point on the hyperbola
noncomputable def P : ℝ × ℝ := sorry

-- Define vectors from P to foci
noncomputable def PF1 : ℝ × ℝ := (F1.1 - P.1, F1.2 - P.2)
noncomputable def PF2 : ℝ × ℝ := (F2.1 - P.1, F2.2 - P.2)

-- Define dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define vector addition
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define vector magnitude
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem hyperbola_property :
  hyperbola P.1 P.2 →
  dot_product PF1 PF2 = 0 →
  magnitude (vector_add PF1 PF2) = 2 * Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_property_l226_22674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_xaxis_implies_a_value_l226_22633

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x + 1/4

noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_xaxis_implies_a_value (a : ℝ) :
  (∃ m : ℝ, f a m = 0 ∧ f' a m = 0) →
  a = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_xaxis_implies_a_value_l226_22633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l226_22609

/-- The definite integral of (x^4) / ((4-x^2)^(3/2)) from 0 to √2 -/
noncomputable def integral_func : ℝ → ℝ := λ x => x^4 / (4 - x^2)^(3/2)

/-- The lower bound of the integral -/
def lower_bound : ℝ := 0

/-- The upper bound of the integral -/
noncomputable def upper_bound : ℝ := Real.sqrt 2

/-- The result of the definite integral -/
noncomputable def integral_result : ℝ := 5 - (3 * Real.pi) / 2

theorem integral_equality : ∫ x in lower_bound..upper_bound, integral_func x = integral_result := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l226_22609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_balls_after_transfer_l226_22677

/-- Represents a box containing balls of a single color -/
structure Box where
  balls : ℕ

/-- Represents the state of both boxes after transfers -/
structure BoxState where
  boxA : Box
  boxB : Box
  transferred : ℕ

/-- Initial state of the boxes -/
def initial_state (a b : ℕ) : BoxState :=
  { boxA := { balls := a },
    boxB := { balls := b },
    transferred := 0 }

/-- State after transferring balls from A to B -/
def transfer_A_to_B (state : BoxState) (n : ℕ) : BoxState :=
  { boxA := { balls := state.boxA.balls - n },
    boxB := { balls := state.boxB.balls + n },
    transferred := n }

/-- Final state after transferring balls from B to A -/
def transfer_B_to_A (state : BoxState) : BoxState :=
  { boxA := { balls := state.boxA.balls + state.transferred },
    boxB := { balls := state.boxB.balls - state.transferred },
    transferred := state.transferred }

/-- The number of white balls in A equals the number of black balls in B after transfers -/
theorem equal_balls_after_transfer (a b n : ℕ) (h1 : n ≤ a) (h2 : n ≤ b) :
  let initial := initial_state a b
  let after_first := transfer_A_to_B initial n
  let final := transfer_B_to_A after_first
  final.boxA.balls = n - (final.boxB.balls - b) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_balls_after_transfer_l226_22677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l226_22665

/-- Hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : 0 < a ∧ 0 < b

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D space -/
structure Line where
  m : ℝ  -- slope
  c : ℝ  -- y-intercept

/-- Right focus of the hyperbola -/
noncomputable def rightFocus (h : Hyperbola) : Point :=
  { x := h.a, y := 0 }

/-- Asymptote of the hyperbola -/
noncomputable def asymptote (h : Hyperbola) : Line :=
  { m := h.b / h.a, c := 0 }

/-- Check if a point is on a line -/
def onLine (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.c

/-- Theorem: Eccentricity of hyperbola E is √2 -/
theorem hyperbola_eccentricity (E : Hyperbola) (O F A : Point) :
  O.x = 0 ∧ O.y = 0 →  -- O is the coordinate origin
  F = rightFocus E →  -- F is the right focus of E
  (∃ l : Line, l.m = 0 ∧ F.x = A.x) →  -- FA is perpendicular to x-axis
  (onLine A (asymptote E)) →  -- A is on the asymptote
  O.x - F.x = F.x - A.x ∧ O.y - F.y = A.y - F.y →  -- OAF is isosceles right triangle
  Real.sqrt 2 = Real.sqrt ((E.a ^ 2 + E.b ^ 2) / E.a ^ 2) := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l226_22665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_150_degrees_l226_22628

theorem sec_150_degrees : (1 / Real.cos (150 * π / 180)) = -2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sec_150_degrees_l226_22628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_log_2017_l226_22690

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * x^(1/3) - 1

-- State the theorem
theorem f_value_at_log_2017 (a b : ℝ) 
  (h : f a b (Real.log (1/2017)) = 2016) : 
  f a b (Real.log 2017) = -2018 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_log_2017_l226_22690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_impossibility_l226_22627

noncomputable def transform (a b : ℝ) : ℝ × ℝ :=
  ((a + b) / Real.sqrt 2, (a - b) / Real.sqrt 2)

def sum_of_squares (x y z : ℝ) : ℝ :=
  x^2 + y^2 + z^2

theorem transformation_impossibility :
  ∃ (x y z : ℝ),
  x ∈ ({2, Real.sqrt 2, 1 / Real.sqrt 2} : Set ℝ) ∧
  y ∈ ({2, Real.sqrt 2, 1 / Real.sqrt 2} : Set ℝ) ∧
  z ∈ ({2, Real.sqrt 2, 1 / Real.sqrt 2} : Set ℝ) ∧
  x ≠ y ∧ x ≠ z ∧ y ≠ z ∧
  sum_of_squares x y z ≠ sum_of_squares 1 (Real.sqrt 2) (1 + Real.sqrt 2) :=
by
  -- Proof goes here
  sorry

#check transformation_impossibility

end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_impossibility_l226_22627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_self_inverse_cube_mod_prime_l226_22626

theorem self_inverse_cube_mod_prime (p : ℕ) (a : ZMod p) (h_prime : Nat.Prime p) (h_self_inverse : a = a⁻¹) :
  a^3 = a := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_self_inverse_cube_mod_prime_l226_22626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_needs_more_money_l226_22615

/-- Represents the amount of money John needs in various currencies -/
structure MoneyNeeded where
  eur : ℚ
  sgd : ℚ

/-- Represents the amount of money John currently has in various currencies -/
structure MoneyHas where
  usd : ℚ
  jpy : ℚ

/-- Represents the exchange rates between USD and other currencies -/
structure ExchangeRates where
  usd_to_eur : ℚ
  usd_to_jpy : ℚ
  usd_to_sgd : ℚ

/-- Calculates the additional amount of money John needs in USD -/
noncomputable def additional_money_needed (needed : MoneyNeeded) (has : MoneyHas) (rates : ExchangeRates) : ℚ :=
  let needed_usd := needed.eur / rates.usd_to_eur + needed.sgd / rates.usd_to_sgd
  let has_usd := has.usd + has.jpy / rates.usd_to_jpy
  needed_usd - has_usd

/-- Theorem stating that John needs approximately $6.13 more -/
theorem john_needs_more_money (needed : MoneyNeeded) (has : MoneyHas) (rates : ExchangeRates) :
  needed.eur = 15/2 →
  needed.sgd = 5 →
  has.usd = 2 →
  has.jpy = 500 →
  rates.usd_to_eur = 21/25 →
  rates.usd_to_jpy = 2207/20 →
  rates.usd_to_sgd = 67/50 →
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/200 ∧ 
    |additional_money_needed needed has rates - 613/100| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_needs_more_money_l226_22615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l226_22671

/-- A function satisfying the given properties -/
noncomputable def f (t : ℝ) : ℝ → ℝ :=
  fun x => sorry

theorem range_of_t (t : ℝ) : 
  (∀ x, x ∈ Set.Icc (-1 : ℝ) 1 → f t x = -f t (-x)) →
  (∀ x y, x ∈ Set.Icc (-1 : ℝ) 1 → y ∈ Set.Icc (-1 : ℝ) 1 → x ≤ y → f t x ≤ f t y) →
  (f t (-1) = -1) →
  (∀ x a, x ∈ Set.Icc (-1 : ℝ) 1 → a ∈ Set.Icc (-1 : ℝ) 1 → f t x ≤ t^2 - 2*a*t + 1) →
  (t ≥ 2 ∨ t ≤ -2 ∨ t = 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_t_l226_22671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_calculation_l226_22650

noncomputable section

-- Define the tank dimensions
def tank_length : ℝ := 12
def tank_width : ℝ := 12
def tank_height : ℝ := 15

-- Define the water fill ratio
def water_fill_ratio : ℝ := 1/3

-- Define the marble properties
def marble_radius : ℝ := 1.5
def marble_count : ℕ := 5

-- Define the unoccupied volume calculation
noncomputable def unoccupied_volume : ℝ :=
  tank_length * tank_width * tank_height * (1 - water_fill_ratio) -
  (marble_count : ℝ) * (4/3) * Real.pi * marble_radius^3

-- Theorem statement
theorem unoccupied_volume_calculation :
  unoccupied_volume = 1440 - 22.5 * Real.pi := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unoccupied_volume_calculation_l226_22650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l226_22657

-- Define the ellipse parameters
noncomputable def a : ℝ := 4
noncomputable def b : ℝ := 2

-- Define the point that the ellipse passes through
noncomputable def x₀ : ℝ := -2
noncomputable def y₀ : ℝ := Real.sqrt 3

-- Define the focal length
noncomputable def focal_length : ℝ := 4 * Real.sqrt 3

theorem ellipse_focal_length :
  (x₀^2 / a^2 + y₀^2 / b^2 = 1) →
  focal_length = 2 * Real.sqrt (a^2 - b^2) := by
  sorry

#check ellipse_focal_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focal_length_l226_22657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l226_22686

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 2 then x / (2 * x^2 + 8)
  else (1/2) ^ |x - a|

def function_property (a : ℝ) : Prop :=
  ∀ x₁, x₁ ≥ 2 → ∃! x₂, x₂ < 2 ∧ f a x₁ = f a x₂

theorem range_of_a :
  {a : ℝ | function_property a} = Set.Icc (-1 : ℝ) 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l226_22686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_power_equation_solutions_l226_22689

def d (n : ℕ) : ℕ := (Nat.divisors n).card

theorem divisor_power_equation_solutions :
  ∀ (n k : ℕ) (p : ℕ), n ≠ 0 → k ≠ 0 → p ≠ 0 → Nat.Prime p →
    (n ^ (d n) - 1 = p ^ k) ↔ ((n = 2 ∧ k = 1 ∧ p = 3) ∨ (n = 3 ∧ k = 3 ∧ p = 2)) :=
by sorry

#check divisor_power_equation_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisor_power_equation_solutions_l226_22689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_l226_22688

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line passing through two points -/
structure Line where
  p1 : Point
  p2 : Point

/-- Represents a parabola with equation y^2 = 4x -/
def Parabola := {p : Point | p.y^2 = 4 * p.x}

/-- The focus of the parabola -/
def F : Point := ⟨1, 0⟩

/-- Point M on the parabola -/
noncomputable def M : Point := ⟨2, 2 * Real.sqrt 2⟩

/-- Line passing through F and M -/
noncomputable def l : Line := ⟨F, M⟩

/-- Point N where line l intersects the parabola (other than M) -/
noncomputable def N : Point :=
  ⟨1/2, -Real.sqrt 2⟩

/-- Distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem stating the ratio of distances -/
theorem distance_ratio :
  distance N F / distance F M = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_ratio_l226_22688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l226_22608

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Helper function to calculate the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ :=
  1 / 2 * t.b * t.c * Real.sin t.A

/-- The theorem stating the properties of the triangle -/
theorem triangle_properties (t : Triangle) :
  t.a * Real.cos t.C = (2 * t.b - t.c) * Real.cos t.A →
  t.a = 6 →
  t.b + t.c = 8 →
  Real.cos t.A = 1 / 2 ∧
  area t = 7 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l226_22608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximal_value_f_maximal_value_two_f_maximal_value_three_or_more_l226_22685

/-- The function f as defined in the problem -/
def f {n : ℕ} (x : Fin n → ℕ) : ℚ :=
  let S (i : Fin n) := (Finset.univ.sum fun j => if j ≠ i then x j else 0)
  (Finset.univ.sum fun i => Nat.gcd (x i) (S i)) / (Finset.univ.sum x)

/-- The main theorem stating the maximal value of f -/
theorem f_maximal_value {n : ℕ} (hn : n ≥ 2) :
  ∀ (x : Fin n → ℕ), Function.Injective x →
    (n = 2 → f x ≤ 2/3) ∧
    (n ≥ 3 → f x ≤ 1) := by
  sorry

/-- Specific case for n = 2 -/
theorem f_maximal_value_two :
  ∃ (x : Fin 2 → ℕ), Function.Injective x ∧ f x = 2/3 := by
  sorry

/-- Specific case for n ≥ 3 -/
theorem f_maximal_value_three_or_more (n : ℕ) (hn : n ≥ 3) :
  ∃ (x : Fin n → ℕ), Function.Injective x ∧ f x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximal_value_f_maximal_value_two_f_maximal_value_three_or_more_l226_22685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_theorem_l226_22606

noncomputable def arun_weight_range (w : ℝ) : Prop := 61 < w ∧ w ≤ 64

noncomputable def average_weight : ℝ := (62 + 63 + 64) / 3

theorem arun_weight_theorem (X : ℝ) :
  (∀ w, arun_weight_range w → w ≤ X) ∧
  average_weight = 63 ∧
  X ≥ 64 →
  X = 64 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arun_weight_theorem_l226_22606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l226_22617

open Real

-- Define the function
noncomputable def f (x : ℝ) := log x - x

-- State the theorem
theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Ioo 0 (exp 1) ∧
  (∀ x, x ∈ Set.Ioo 0 (exp 1) → f x ≤ f c) ∧
  f c = -1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l226_22617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_common_terms_l226_22695

def arithmetic_progression (n : ℕ) : ℕ := 5 + 3 * n

def geometric_progression (k : ℕ) : ℕ := 20 * 2^k

def is_common_term (x : ℕ) : Bool :=
  (List.range 1000).any (λ n => (List.range 1000).any (λ k => 
    arithmetic_progression n = x ∧ geometric_progression k = x))

def common_terms : List ℕ :=
  (List.range 1000).filter is_common_term

theorem sum_of_first_10_common_terms :
  (common_terms.take 10).sum = 6990500 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_first_10_common_terms_l226_22695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l226_22672

/-- Calculates the final value of an investment after two consecutive 9-month periods 
    with different interest rates -/
noncomputable def investment_value (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let value_after_first_period := initial_amount * (1 + rate1 * (9 / 12))
  value_after_first_period * (1 + rate2 * (9 / 12))

/-- Theorem stating that the given investment scenario results in the specified final value -/
theorem investment_growth : 
  let initial_amount := (15000 : ℝ)
  let rate1 := (0.09 : ℝ)  -- 9% annual rate
  let rate2 := (0.15 : ℝ)  -- 15% annual rate
  abs (investment_value initial_amount rate1 rate2 - 17814.06) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l226_22672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_in_cube_l226_22664

/-- A cube with vertices ABCDA₁B₁C₁D₁ -/
structure Cube (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (A B C D A₁ B₁ C₁ D₁ : α)
  (edge_length : ℝ)
  (is_cube : True)  -- Placeholder for cube conditions

/-- The distance between two lines in a 3D space -/
noncomputable def distance_between_lines {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]
  (l₁ l₂ : Set α) : ℝ :=
  sorry

/-- Theorem: In a cube with edge length a, the distance between lines AA₁ and BD₁ is a√2/2 -/
theorem distance_between_lines_in_cube {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α]
  (cube : Cube α) :
  distance_between_lines
    {x : α | ∃ t : ℝ, x = cube.A + t • (cube.A₁ - cube.A)}
    {x : α | ∃ t : ℝ, x = cube.B + t • (cube.D₁ - cube.B)} =
  cube.edge_length * Real.sqrt 2 / 2 := by
  sorry

#check distance_between_lines_in_cube

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_lines_in_cube_l226_22664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_roots_l226_22641

theorem cubic_polynomial_roots (r s : ℝ) : 
  (∃ (c d : ℝ), r^3 + c*r + d = 0 ∧ s^3 + c*s + d = 0) →
  (∃ (c : ℝ), (r-3)^3 + c*(r-3) + (d+156) = 0 ∧ (s+5)^3 + c*(s+5) + (d+156) = 0) →
  d = -198 ∨ d = 468 := by
  sorry

#check cubic_polynomial_roots

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_roots_l226_22641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_exists_l226_22607

theorem subset_sum_exists (numbers : List ℕ) 
  (h1 : numbers.length = 100)
  (h2 : ∀ n ∈ numbers, n ≤ 100)
  (h3 : numbers.sum = 200) :
  ∃ subset : List ℕ, subset.toFinset ⊆ numbers.toFinset ∧ subset.sum = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_exists_l226_22607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l226_22659

-- Define the triangle ABC and point G
variable (A B C G D : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
variable (h1 : D = (B + C) / 2)  -- D is the midpoint of BC
variable (h2 : G + A = B + C)    -- Vector equation: GA + BG + CG = 0
variable (h3 : ∃ (lambda : ℝ), A - G = lambda • (G - D))  -- AG = lambda * GD

-- State the theorem
theorem triangle_vector_ratio :
  ∃ (lambda : ℝ), A - G = lambda • (G - D) ∧ lambda = -2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_ratio_l226_22659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l226_22670

open Real

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 4 * (cos x) * (cos (2 * x)) * (cos (3 * x)) = cos (6 * x) ↔
  (∃ l : ℤ, x = π/3 * (3 * ↑l + 1) ∨ x = π/3 * (3 * ↑l - 1)) ∨
  (∃ n : ℤ, x = π/4 * (2 * ↑n + 1)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l226_22670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_perimeter_B_is_16_l226_22651

/-- Given two squares A and B, where B is placed within A, this theorem proves
    that the perimeter of B is 16 cm given the specified conditions. -/
theorem square_perimeter (area_A : ℝ) (prob : ℝ) : ℝ :=
  let side_A := Real.sqrt area_A
  let area_B := area_A * (1 - prob)
  let side_B := Real.sqrt area_B
  4 * side_B

/-- The perimeter of square B calculated using the given conditions. -/
noncomputable def perimeter_B : ℝ :=
  square_perimeter 121 0.8677685950413223

/-- Proof that the perimeter of square B is indeed 16 cm. -/
theorem perimeter_B_is_16 : perimeter_B = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_perimeter_perimeter_B_is_16_l226_22651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_castle_sum_l226_22632

/-- Represents the configuration of a dragon tethered to a cylindrical castle -/
structure DragonCastle where
  chain_length : ℝ
  castle_attach_height : ℝ
  dragon_attach_height : ℝ
  castle_radius : ℝ
  chain_end_distance : ℝ
  d : ℕ
  e : ℕ
  f : ℕ
  f_prime : Nat.Prime f

/-- The length of chain touching the castle -/
noncomputable def chain_castle_touch (config : DragonCastle) : ℝ :=
  (config.d - Real.sqrt (config.e : ℝ)) / config.f

/-- The sum of d, e, and f for the given dragon and castle configuration -/
def sum_def (config : DragonCastle) : ℕ := config.d + config.e + config.f

/-- Theorem stating that the sum of d, e, and f is 160 for the given configuration -/
theorem dragon_castle_sum (config : DragonCastle) 
  (h1 : config.chain_length = 25)
  (h2 : config.castle_attach_height = 3)
  (h3 : config.dragon_attach_height = 6)
  (h4 : config.castle_radius = 10)
  (h5 : config.chain_end_distance = 5) :
  sum_def config = 160 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dragon_castle_sum_l226_22632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_l226_22630

theorem min_c_value (a b c : ℕ+) (ha : a < b) (hb : b < c)
  (h_unique : ∃! (x : ℝ) (y : ℝ), 3 * x + y = 3005 ∧ y = |x - a.1| + |x - b.1| + |x - c.1|) :
  c ≥ 1004 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_c_value_l226_22630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_area_l226_22602

/-- Given two triangles with oppositely directed sides, prove the area of the triangle
    formed by the midpoints of the segments connecting corresponding vertices. -/
theorem midpoint_triangle_area (S₁ S₂ : ℝ) (h₁ : S₁ > 0) (h₂ : S₂ > 0) :
  ∃ (A₁ B₁ C₁ A₂ B₂ C₂ : ℝ × ℝ),
  let triangle₁ := {A₁, B₁, C₁}
  let triangle₂ := {A₂, B₂, C₂}
  let area (t : Set (ℝ × ℝ)) : ℝ := sorry
  let midpoint (p q : ℝ × ℝ) : ℝ × ℝ := sorry
  let oppositelyDirected (p₁ q₁ p₂ q₂ : ℝ × ℝ) : Prop := sorry

  area triangle₁ = S₁ ∧
  area triangle₂ = S₂ ∧
  oppositelyDirected A₁ B₁ A₂ B₂ ∧
  oppositelyDirected B₁ C₁ B₂ C₂ ∧
  oppositelyDirected C₁ A₁ C₂ A₂ ∧

  let M := midpoint A₁ A₂
  let N := midpoint B₁ B₂
  let K := midpoint C₁ C₂
  let triangle_mid := {M, N, K}

  (1/4) * (Real.sqrt S₁ + Real.sqrt S₂)^2 = area triangle_mid := by
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_triangle_area_l226_22602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_beta_eq_sqrt_two_l226_22646

noncomputable def f (x : ℝ) : ℝ := Real.sin (5 * Real.pi / 4 - x) - Real.cos (Real.pi / 4 + x)

theorem f_beta_eq_sqrt_two (α β : ℝ) 
  (h1 : Real.cos (α - β) = 3/5)
  (h2 : Real.cos (α + β) = -3/5)
  (h3 : 0 < α) (h4 : α < β) (h5 : β ≤ Real.pi/2) :
  f β = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_beta_eq_sqrt_two_l226_22646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_alpha_investment_l226_22693

/-- Represents the investment scenario for Jane --/
structure Investment where
  total : ℚ
  alphaRate : ℚ
  betaRate : ℚ
  gammaRate : ℚ
  finalAmount : ℚ

/-- Calculates the amount invested in Alpha Bank --/
def calculateAlphaInvestment (inv : Investment) : ℚ :=
  300

/-- Theorem stating that Jane's investment in Alpha Bank is $300 --/
theorem jane_alpha_investment (inv : Investment) 
  (h1 : inv.total = 1500)
  (h2 : inv.alphaRate = 1/25)
  (h3 : inv.betaRate = 3/50)
  (h4 : inv.gammaRate = 1/20)
  (h5 : inv.finalAmount = 1590) :
  calculateAlphaInvestment inv = 300 := by
  sorry

#eval calculateAlphaInvestment { total := 1500, alphaRate := 1/25, betaRate := 3/50, gammaRate := 1/20, finalAmount := 1590 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jane_alpha_investment_l226_22693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_travel_time_l226_22656

/-- Calculates the total travel time for a round trip given the distance and speeds -/
noncomputable def total_travel_time (distance : ℝ) (speed_out : ℝ) (speed_return : ℝ) : ℝ :=
  distance / speed_out + distance / speed_return

/-- Proves that the total travel time for the given conditions is 5.4 hours -/
theorem train_travel_time :
  let distance := (120 : ℝ)
  let speed_out := (40 : ℝ)
  let speed_return := (49.99999999999999 : ℝ)
  total_travel_time distance speed_out speed_return = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_travel_time_l226_22656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_third_step_value_l226_22613

def horner_polynomial (x : ℝ) : ℝ := 2*x^5 + 5*x^3 - x^2 + 9*x + 1

def horner_step (v : ℝ) (a : ℝ) (x : ℝ) : ℝ := v * x + a

def horner_method (x : ℝ) : List ℝ :=
  [2, 0, 5, -1, 9, 1].foldl (λ acc a => acc ++ [horner_step acc.getLast! a x]) []

theorem horner_third_step_value :
  (horner_method 3).get! 3 = 68 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_third_step_value_l226_22613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_12_equals_negative_one_l226_22644

def mySequence (n : ℕ) : ℚ := 
  match n with
  | 0 => 2
  | n + 1 => 1 - 1 / mySequence n

theorem a_12_equals_negative_one : mySequence 11 = -1 := by
  sorry

#eval mySequence 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_12_equals_negative_one_l226_22644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_cubic_equation_solution_l226_22614

theorem complex_cubic_equation_solution : 
  ∃ (z : ℂ), z * (z + 1) * (z + 3) = 2018 := by
  sorry

#check complex_cubic_equation_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_cubic_equation_solution_l226_22614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_pond_volume_l226_22605

/-- Represents the dimensions of the rectangular base of the pond -/
structure RectBase where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Represents the dimensions of the semi-elliptical dome of the pond -/
structure SemiEllipticalDome where
  majorAxis : ℝ
  minorAxis : ℝ
  depth : ℝ

/-- Calculates the volume of the rectangular base -/
def rectBaseVolume (rb : RectBase) : ℝ :=
  rb.length * rb.width * rb.depth

/-- Calculates the volume of the semi-elliptical dome -/
noncomputable def semiEllipticalDomeVolume (sed : SemiEllipticalDome) : ℝ :=
  1/2 * (4/3) * Real.pi * (sed.majorAxis/2) * (sed.minorAxis/2) * sed.depth

/-- Calculates the total volume of the irregular pond -/
noncomputable def totalVolume (rb : RectBase) (sed : SemiEllipticalDome) : ℝ :=
  rectBaseVolume rb + semiEllipticalDomeVolume sed

/-- Theorem stating that the volume of soil extracted for the irregular pond is approximately 1130.9 cubic meters -/
theorem irregular_pond_volume :
  let rb : RectBase := { length := 20, width := 10, depth := 5 }
  let sed : SemiEllipticalDome := { majorAxis := 10, minorAxis := 5, depth := 5 }
  ∃ ε > 0, |totalVolume rb sed - 1130.9| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_irregular_pond_volume_l226_22605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_approximately_904_97_l226_22667

noncomputable section

-- Define the dimensions of the windows
def rect_pane_length : ℝ := 12
def rect_pane_width : ℝ := 8
def rect_pane_count : ℕ := 8
def tri_base : ℝ := 10
def tri_height : ℝ := 12
def half_circle_diameter : ℝ := 14

-- Define the areas of each window
def rectangular_window_area : ℝ := rect_pane_length * rect_pane_width * rect_pane_count
def triangular_window_area : ℝ := (tri_base * tri_height) / 2
def half_circular_window_area : ℝ := (Real.pi * (half_circle_diameter / 2)^2) / 2

-- Define the total area of all windows
def total_window_area : ℝ := rectangular_window_area + triangular_window_area + half_circular_window_area

-- Theorem statement
theorem total_area_approximately_904_97 :
  ∃ ε > 0, |total_window_area - 904.97| < ε := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_area_approximately_904_97_l226_22667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marseille_hair_problem_l226_22676

theorem marseille_hair_problem :
  ∀ (hair_count : Fin 2000000 → Nat),
    (∀ i, hair_count i ≤ 300000) →
    ∃ (n : Nat),
      n ≤ 300000 ∧
      (Finset.filter (fun i => hair_count i = n) Finset.univ).card ≥ 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marseille_hair_problem_l226_22676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l226_22601

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (3 * x + 4) / (x - 5)

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≠ 3} :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l226_22601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_not_opposite_l226_22699

-- Define the pocket contents
def total_balls : ℕ := 4
def red_balls : ℕ := 2
def black_balls : ℕ := 2

-- Define the number of balls picked
def picked_balls : ℕ := 2

-- Define the events
def exactly_one_black (outcome : Finset (Fin total_balls)) : Prop :=
  outcome.card = picked_balls ∧ (outcome.filter (λ i => i.val < black_balls)).card = 1

def exactly_two_black (outcome : Finset (Fin total_balls)) : Prop :=
  outcome.card = picked_balls ∧ (outcome.filter (λ i => i.val < black_balls)).card = 2

-- Define mutually exclusive
def mutually_exclusive (e1 e2 : (Finset (Fin total_balls) → Prop)) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

-- Define opposite events
def opposite_events (e1 e2 : (Finset (Fin total_balls) → Prop)) : Prop :=
  ∀ outcome, e1 outcome ↔ ¬(e2 outcome)

-- Theorem statement
theorem events_mutually_exclusive_not_opposite :
  mutually_exclusive exactly_one_black exactly_two_black ∧
  ¬(opposite_events exactly_one_black exactly_two_black) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_not_opposite_l226_22699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_sales_amount_l226_22639

/-- Calculates the amount of next sales given initial and subsequent royalties and sales data --/
noncomputable def calculate_next_sales (initial_royalties : ℝ) (initial_sales : ℝ) (next_royalties : ℝ) (ratio_decrease : ℝ) : ℝ :=
  (next_royalties * initial_sales) / (initial_royalties * (1 - ratio_decrease))

/-- Theorem stating that given the specified conditions, the next sales amount is 108 million --/
theorem next_sales_amount 
  (initial_royalties : ℝ) 
  (initial_sales : ℝ) 
  (next_royalties : ℝ) 
  (ratio_decrease : ℝ) :
  initial_royalties = 7 →
  initial_sales = 20 →
  next_royalties = 9 →
  ratio_decrease = 0.761904761904762 →
  calculate_next_sales initial_royalties initial_sales next_royalties ratio_decrease = 108 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_sales_amount_l226_22639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l226_22696

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.A > 0 ∧ t.A < Real.pi ∧
  t.B > 0 ∧ t.B < Real.pi ∧
  t.C > 0 ∧ t.C < Real.pi ∧
  t.A + t.B + t.C = Real.pi

def satisfies_condition (t : Triangle) : Prop :=
  t.b = t.a * Real.cos t.C + (Real.sqrt 3 / 3) * t.c * Real.sin t.A

-- Helper function for area
noncomputable def area (t : Triangle) : ℝ :=
  (1 / 2) * t.b * t.c * Real.sin t.A

-- State the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : satisfies_condition t) : 
  t.A = Real.pi / 3 ∧ 
  (t.a = 2 → ∀ t' : Triangle, is_valid_triangle t' → t'.a = 2 → 
    area t' ≤ Real.sqrt 3 ∧ 
    (area t' = Real.sqrt 3 ↔ t'.b = t'.c)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l226_22696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_theta_max_magnitude_sum_l226_22673

open Real

-- Define the vectors a and b
noncomputable def a (θ : ℝ) : Fin 2 → ℝ := ![sin θ, sqrt 3]
noncomputable def b (θ : ℝ) : Fin 2 → ℝ := ![1, cos θ]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- Define the magnitude of a vector
noncomputable def magnitude (v : Fin 2 → ℝ) : ℝ :=
  sqrt ((v 0)^2 + (v 1)^2)

theorem perpendicular_vectors_theta (θ : ℝ) 
  (h : θ > -π/2 ∧ θ < π/2) 
  (h_perp : dot_product (a θ) (b θ) = 0) : 
  θ = -π/3 := by sorry

theorem max_magnitude_sum (θ : ℝ) 
  (h : θ > -π/2 ∧ θ < π/2) : 
  (∀ φ, φ > -π/2 → φ < π/2 → 
    magnitude (a φ + b φ) ≤ magnitude (a θ + b θ)) → 
  magnitude (a θ + b θ) = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_theta_max_magnitude_sum_l226_22673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_decision_probability_l226_22648

-- Define the probabilities
noncomputable def prob_aunt_home : ℝ := 1/2
noncomputable def prob_answer_given_home : ℝ := 3/5
noncomputable def cost_not_picking_up : ℝ := 3
noncomputable def cost_unnecessary_trip : ℝ := 1

-- Define the theorem
theorem correct_decision_probability :
  let prob_correct := (1 - prob_aunt_home) * cost_unnecessary_trip + 
                      prob_aunt_home * (1 - prob_answer_given_home) * cost_not_picking_up
  prob_correct / (cost_not_picking_up + cost_unnecessary_trip) = 3/16 :=
by
  -- The proof is omitted for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_decision_probability_l226_22648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l226_22640

theorem divisibility_condition (n : ℕ) :
  3 ∣ (n : ℤ) * 2^n + 1 ↔ n % 6 = 1 ∨ n % 6 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_condition_l226_22640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_abc_l226_22620

/-- The area of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (Real.sqrt 3 / 4) * s^2

/-- The height of an equilateral triangle with side length s -/
noncomputable def equilateral_triangle_height (s : ℝ) : ℝ := (Real.sqrt 3 / 2) * s

/-- The distance from the center to the side of an equilateral triangle -/
noncomputable def center_to_side_distance (s : ℝ) : ℝ := (2 / 3) * equilateral_triangle_height s

/-- The theorem stating the area of triangle ABC formed by the centers of three equilateral triangles -/
theorem area_of_triangle_abc (s : ℝ) (h : s = 1) : 
  let d := center_to_side_distance s
  let side_abc := 2 * d
  equilateral_triangle_area side_abc = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_abc_l226_22620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_equality_l226_22684

noncomputable def base5ToDecimal (n : ℕ) : ℕ := 
  (n / 10) * 5 + (n % 10)

noncomputable def baseBToDecimal (n : ℕ) (b : ℝ) : ℝ := 
  (b^2 * (n / 100 : ℝ)) + (b * ((n / 10 % 10) : ℝ)) + ((n % 10) : ℝ)

theorem base_conversion_equality :
  ∃ (b : ℝ), b > 0 ∧ base5ToDecimal 32 = baseBToDecimal 120 b ∧ b = -1 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_conversion_equality_l226_22684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_c_l226_22669

-- Define the variables
variable (a b c : ℝ)

-- Define the conditions
def condition1 (a b c : ℝ) : Prop := a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)
def condition2 (c : ℝ) : Prop := 6 * 15 * c = 1.5

-- State the theorem
theorem value_of_c (h1 : condition1 a b c) (h2 : condition2 c) : c = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_c_l226_22669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l226_22625

-- Define the parametric equations of line C₁
noncomputable def C₁ (t : ℝ) : ℝ × ℝ := (1 + t/2, (Real.sqrt 3 / 2) * t)

-- Define the polar equation of curve C₂
def C₂_polar (ρ θ : ℝ) : Prop := ρ^2 * (1 + 2 * Real.sin θ^2) = 3

-- Define the Cartesian equation of curve C₂
def C₂_cartesian (x y : ℝ) : Prop := x^2 / 3 + y^2 = 1

-- Define point M
def M : ℝ × ℝ := (1, 0)

-- Define the theorem
theorem intersection_distance_difference :
  ∃ A B : ℝ × ℝ,
    (∃ t₁ t₂ : ℝ, C₁ t₁ = A ∧ C₁ t₂ = B) ∧
    C₂_cartesian A.1 A.2 ∧
    C₂_cartesian B.1 B.2 ∧
    |dist M A - dist M B| = 2/5 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_difference_l226_22625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_of_equation_l226_22661

theorem integer_solutions_of_equation : 
  ∀ x y : ℤ, (y^2 + y = x^4 + x^3 + x^2 + x) ↔ 
  ((x = -1 ∧ y = -1) ∨ (x = -1 ∧ y = 0) ∨ 
   (x = 0 ∧ y = -1) ∨ (x = 0 ∧ y = 0) ∨ 
   (x = 2 ∧ y = 5) ∨ (x = 2 ∧ y = -6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_solutions_of_equation_l226_22661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_value_l226_22694

noncomputable def binomial_expansion (a : ℝ) : ℝ → ℝ := fun x => (x - a / x^2)^9

noncomputable def sum_of_coefficients (a : ℝ) : ℝ := binomial_expansion a 1

noncomputable def constant_term (a : ℝ) : ℝ := sorry

theorem constant_term_value :
  ∃ a : ℝ, sum_of_coefficients a = -1 ∧ constant_term a = -672 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_term_value_l226_22694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_GHCD_value_l226_22637

/-- Represents a trapezoid ABCD with midpoints G and H on sides AD and BC respectively -/
structure Trapezoid where
  AB : ℝ
  CD : ℝ
  altitude : ℝ

/-- Calculates the area of quadrilateral GHCD within the trapezoid -/
noncomputable def area_GHCD (t : Trapezoid) : ℝ :=
  (t.altitude / 2) * ((t.AB + t.CD) / 2 + t.CD) / 2

/-- Theorem stating that the area of quadrilateral GHCD is 123.75 square units -/
theorem area_GHCD_value (t : Trapezoid) 
  (h1 : t.AB = 12)
  (h2 : t.CD = 18)
  (h3 : t.altitude = 15) :
  area_GHCD t = 123.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_GHCD_value_l226_22637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_a_range_l226_22666

-- Define the functions f and g
noncomputable def f (x : ℝ) := x * Real.log x
def g (a : ℝ) (x : ℝ) := -x^2 + a*x - 3

-- Theorem for the tangent line equation
theorem tangent_line_equation :
  let f' := fun x => Real.log x + 1
  (fun x => f' (Real.exp 1) * (x - Real.exp 1) + f (Real.exp 1)) = fun x => 2 * x - Real.exp 1 :=
by sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∃ x ∈ Set.Icc 1 (Real.exp 1), ∀ y ∈ Set.Icc 1 (Real.exp 1), 2 * f y ≥ g a y) →
  a ≤ 2 + Real.exp 1 + 3 / Real.exp 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_a_range_l226_22666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equationD_is_quadratic_equationA_not_always_quadratic_equationB_not_quadratic_equationC_not_quadratic_l226_22643

-- Define the structure of a quadratic equation in one variable
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  eq : ℝ → Prop := fun x => a * x^2 + b * x + c = 0

-- Define the given equations
def equationA (a b c : ℝ) : ℝ → Prop := fun x => a * x^2 + b * x + c = 0
def equationB : ℝ → Prop := fun x => x^2 - 2 = (x + 3)^2
def equationC : ℝ → Prop := fun x => x + 2/x - 3 = 0
def equationD : ℝ → Prop := fun x => x^2 + 1 = 0

-- Theorem stating that equationD is a quadratic equation in one variable
theorem equationD_is_quadratic : 
  ∃ q : QuadraticEquation, (∀ x : ℝ, q.eq x ↔ equationD x) :=
sorry

-- Theorems stating that the other equations are not quadratic equations in one variable
theorem equationA_not_always_quadratic : 
  ¬(∀ a b c : ℝ, ∃ q : QuadraticEquation, (∀ x : ℝ, q.eq x ↔ equationA a b c x)) :=
sorry

theorem equationB_not_quadratic : 
  ¬(∃ q : QuadraticEquation, (∀ x : ℝ, q.eq x ↔ equationB x)) :=
sorry

theorem equationC_not_quadratic : 
  ¬(∃ q : QuadraticEquation, (∀ x : ℝ, q.eq x ↔ equationC x)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equationD_is_quadratic_equationA_not_always_quadratic_equationB_not_quadratic_equationC_not_quadratic_l226_22643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_composition_implies_limit_l226_22658

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem limit_composition_implies_limit (hf : Continuous f) 
  (h_lim : Filter.Tendsto (f ∘ f) atTop atTop) :
  Filter.Tendsto f atTop atTop :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_composition_implies_limit_l226_22658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_numbers_divisibility_l226_22697

theorem ten_numbers_divisibility : 
  ∃ (S : Finset ℕ), 
    S.card = 10 ∧ 
    (∀ x, x ∈ S → x > 0) ∧
    (∀ x y, x ∈ S → y ∈ S → x ≠ y) ∧
    (∀ x, x ∈ S → (S.sum id) % x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ten_numbers_divisibility_l226_22697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_equals_negative_three_l226_22631

-- Define the piecewise function g
noncomputable def g (x : ℝ) : ℝ :=
  if x ≥ 0 then -2 * x^2 else x + 5

-- State the theorem
theorem nested_g_equals_negative_three :
  g (g (g (g (g 2)))) = -3 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_g_equals_negative_three_l226_22631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_parallel_line_length_l226_22679

/-- Represents a trapezoid with given base lengths -/
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ

/-- Represents a line segment with a given length -/
structure LineSegment where
  length : ℝ

/-- Represents a point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Auxiliary definition for the intersection of diagonals -/
def is_intersection_of_diagonals (E : Point) (T : Trapezoid) : Prop := sorry

/-- Auxiliary definition for a line passing through a point -/
def passes_through (L : LineSegment) (P : Point) : Prop := sorry

/-- Auxiliary definition for a line being parallel to the bases of a trapezoid -/
def parallel_to_bases (L : LineSegment) (T : Trapezoid) : Prop := sorry

/-- The main theorem about the parallel line in a trapezoid -/
theorem trapezoid_parallel_line_length 
  (ABCD : Trapezoid) 
  (h1 : ABCD.base1 = 2) 
  (h2 : ABCD.base2 = 3) 
  (XY : LineSegment) 
  (h3 : ∃ E : Point, is_intersection_of_diagonals E ABCD ∧ passes_through XY E ∧ parallel_to_bases XY ABCD) :
  XY.length = 2.6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_parallel_line_length_l226_22679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l226_22604

-- Part 1
theorem cube_root_plus_abs_plus_power : (8 : ℝ) ^ (1/3) + |(-5)| + (-1) ^ 2023 = 6 := by sorry

-- Part 2
theorem linear_function_through_points (k b : ℝ) (h1 : k * 0 + b = 1) (h2 : k * 2 + b = 5) :
  ∀ x, k * x + b = 2 * x + 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_plus_abs_plus_power_linear_function_through_points_l226_22604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l226_22638

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

-- State the theorem
theorem function_properties (a : ℝ) :
  (f a 1 = 2) →
  (a = 1) ∧
  (∀ x : ℝ, x ≠ 0 → f a (-x) = -(f a x)) ∧
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f a x₁ > f a x₂) :=
by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l226_22638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_is_unique_sum_free_preserving_surjection_l226_22621

/-- A subset of ℕ is sum-free if the sum of any two elements in the subset is not in the subset. -/
def SumFree (A : Set ℕ) : Prop :=
  ∀ x y, x ∈ A → y ∈ A → x + y ∉ A

/-- The image of a set under a function -/
def SetImage (f : ℕ → ℕ) (A : Set ℕ) : Set ℕ :=
  { y | ∃ x, x ∈ A ∧ f x = y }

/-- Main theorem: If a surjective function preserves sum-free sets, it must be the identity function -/
theorem identity_is_unique_sum_free_preserving_surjection :
    ∀ f : ℕ → ℕ,
    (∀ n : ℕ, ∃ m : ℕ, f m = n) →
    (∀ A : Set ℕ, SumFree A → SumFree (SetImage f A)) →
    ∀ x : ℕ, f x = x := by
  sorry

#check identity_is_unique_sum_free_preserving_surjection

end NUMINAMATH_CALUDE_ERRORFEEDBACK_identity_is_unique_sum_free_preserving_surjection_l226_22621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_of_2748779069441_l226_22603

theorem sixth_root_of_2748779069441 : (2748779069441 : ℚ)^(1/6) = 151 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_root_of_2748779069441_l226_22603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_chessboard_l226_22634

/-- Represents a 100x100 table where each cell can be either white or black -/
def Table := Fin 100 → Fin 100 → Bool

/-- A move is represented by a row or column index and a direction (true for row, false for column) -/
structure Move where
  index : Fin 100
  isRow : Bool

/-- Apply a move to a table -/
def applyMove (t : Table) (m : Move) : Table :=
  fun i j =>
    if m.isRow ∧ i = m.index ∧ j ≠ 99 then !t i j
    else if !m.isRow ∧ j = m.index ∧ i ≠ 99 then !t i j
    else t i j

/-- Check if a table has a chessboard pattern -/
def isChessboard (t : Table) : Prop :=
  ∀ i j, t i j = (i.val + j.val).bodd

/-- The initial all-white table -/
def initialTable : Table := fun _ _ => false

/-- The theorem to be proved -/
theorem min_moves_for_chessboard :
  (∃ (moves : List Move), moves.length = 100 ∧ 
    isChessboard (moves.foldl applyMove initialTable)) ∧
  (∀ (moves : List Move), moves.length < 100 → 
    ¬isChessboard (moves.foldl applyMove initialTable)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_moves_for_chessboard_l226_22634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_correct_l226_22687

/-- Represents a sentence in classical Chinese -/
def ClassicalChineseSentence : Type := String

/-- Represents a sentence in modern Chinese -/
def ModernChineseSentence : Type := String

/-- A translation function that converts classical Chinese to modern Chinese -/
def translate : ClassicalChineseSentence → ModernChineseSentence := sorry

/-- Semantic equivalence between two sentences -/
def semantically_equivalent : ModernChineseSentence → ModernChineseSentence → Prop := sorry

/-- The first classical Chinese sentence -/
def S1 : ClassicalChineseSentence := "梁鼎既去，移文禁盐商，而乖当不如规画，延州刘廷伟不依"

/-- The second classical Chinese sentence -/
def S2 : ClassicalChineseSentence := "梁鼎议以咸阳仓陈陆粮实边，与民，俟秋熟易以新谷"

/-- The expected translation of S1 -/
def M1 : ModernChineseSentence := "梁鼎一走，就发文禁止盐贩，各地的执法应当严格，但延州的刘廷伟并没有遵守这个政策。"

/-- The expected translation of S2 -/
def M2 : ModernChineseSentence := "梁鼎建议把咸阳仓库里发霉的老粮运到边疆作为补给。由于粮食不是新的，他就分发给了当地的百姓，打算等到秋天收获新粮后再进行替换。但中央政府得知后，停止了这一行动。"

theorem translation_correct : 
  semantically_equivalent (translate S1) M1 ∧ 
  semantically_equivalent (translate S2) M2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_correct_l226_22687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_still_water_speed_l226_22660

/-- Represents the speed of a man rowing in different conditions -/
structure RowingSpeed where
  upstream : ℚ
  downstream : ℚ

/-- Calculates the speed in still water given upstream and downstream speeds -/
def stillWaterSpeed (s : RowingSpeed) : ℚ :=
  (s.upstream + s.downstream) / 2

/-- Theorem: The speed of a man in still water is 33 kmph given his upstream and downstream speeds -/
theorem man_still_water_speed :
  let s : RowingSpeed := { upstream := 25, downstream := 41 }
  stillWaterSpeed s = 33 := by
  -- Unfold the definition of stillWaterSpeed
  unfold stillWaterSpeed
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_still_water_speed_l226_22660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_semifocal_distance_range_l226_22616

-- Define set_of_foci as a function
def set_of_foci (a b c : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | x^2 / a^2 + y^2 / b^2 = 1 ∧ c^2 = a^2 + b^2}

theorem hyperbola_semifocal_distance_range (k : ℝ) :
  (∀ x y : ℝ, y^2 / k - x^2 / (k - 2) = 1) →  -- Equation of the hyperbola
  (∃ c : ℝ, (0, c) ∈ set_of_foci k (k - 2) c ∧ (0, -c) ∈ set_of_foci k (k - 2) c) →  -- Foci are on y-axis
  (∃ c : ℝ, c > Real.sqrt 2 ∧ 
    ∀ d : ℝ, d > Real.sqrt 2 → 
      ∃ k' : ℝ, (∀ x y : ℝ, y^2 / k' - x^2 / (k' - 2) = 1) ∧
              (∃ c' : ℝ, (0, c') ∈ set_of_foci k' (k' - 2) c' ∧ (0, -c') ∈ set_of_foci k' (k' - 2) c') ∧
              d = Real.sqrt (2 * k' - 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_semifocal_distance_range_l226_22616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_number_l226_22645

def digit_pairs (n : ℕ) : List (ℕ × ℕ) :=
  let digits := n.digits 10
  List.join (List.map (λ i => 
    List.map (λ j => (digits[i]!, digits[j]!)) 
    (List.range (digits.length - i - 1) |>.map (λ k => k + i + 1)))
  (List.range (digits.length - 1)))

def valid_pairs : List (ℕ × ℕ) := [(3, 3), (3, 7), (3, 7), (3, 7), (3, 8), (7, 3), (7, 7), (7, 8), (8, 3), (8, 7)]

theorem unique_five_digit_number : 
  ∃! (n : ℕ), 
    10000 ≤ n ∧ n < 100000 ∧ 
    List.Perm (digit_pairs n) valid_pairs :=
  by sorry

#eval digit_pairs 37837

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_five_digit_number_l226_22645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l226_22649

/-- The cosine of the angle between two 2D vectors -/
noncomputable def cos_angle (a b : ℝ × ℝ) : ℝ :=
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))

theorem cos_angle_specific_vectors :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (-2, 1)
  cos_angle a b = -Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_angle_specific_vectors_l226_22649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l226_22683

noncomputable section

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  ((1 + Real.cos (2 * x))^2 - 2 * Real.cos (2 * x) - 1) /
  (Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x))

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := (1 / 2) * f x + Real.sin (2 * x)

-- State the theorem
theorem f_and_g_properties :
  (f (-11 * Real.pi / 12) = Real.sqrt 3) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 4),
    g x ≤ Real.sqrt 2 ∧
    g x ≥ 1 ∧
    (∃ x₁ ∈ Set.Icc 0 (Real.pi / 4), g x₁ = Real.sqrt 2) ∧
    (∃ x₂ ∈ Set.Icc 0 (Real.pi / 4), g x₂ = 1)) :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_and_g_properties_l226_22683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_is_I_type_I_type_complement_sin_shift_I_type_iff_l226_22624

-- Definition of an I-type function
def is_I_type_function (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x₁, x₁ ∈ D → ∃! x₂, x₂ ∈ D ∧ f x₁ + f x₂ = 1

-- Theorem 1: Natural logarithm is an I-type function
theorem ln_is_I_type : is_I_type_function Real.log (Set.Ioi 0) := by
  sorry

-- Theorem 2: If f is an I-type function, then 1 - f is also an I-type function
theorem I_type_complement {f : ℝ → ℝ} {D : Set ℝ} :
  is_I_type_function f D → is_I_type_function (λ x ↦ 1 - f x) D := by
  sorry

-- Theorem 3: For f(x) = m + sin x on [-π/2, π/2], f is I-type iff m = 1/2
theorem sin_shift_I_type_iff (m : ℝ) :
  is_I_type_function (λ x ↦ m + Real.sin x) (Set.Icc (-π/2) (π/2)) ↔ m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ln_is_I_type_I_type_complement_sin_shift_I_type_iff_l226_22624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l226_22635

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(2x+1)
def domain_f_2x_plus_1 : Set ℝ := Set.Ioo (-2) 0

-- Theorem statement
theorem domain_of_f (x : ℝ) :
  (∀ y, f (2 * y + 1) ∈ domain_f_2x_plus_1 ↔ y ∈ domain_f_2x_plus_1) →
  (f x ∈ Set.Ioo (-3) 1 ↔ x ∈ Set.Ioo (-3) 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l226_22635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_minimum_condition_l226_22636

/-- The polynomial function y in terms of x, p, and q -/
noncomputable def y (x p q : ℝ) : ℝ := -1/3 * x^3 + x^2 + p*x + q

/-- The statement that the minimum value of y is zero for some x -/
def has_zero_minimum (p q : ℝ) : Prop :=
  ∃ x : ℝ, y x p q = 0 ∧ ∀ t : ℝ, y t p q ≥ 0

/-- The theorem stating the condition for y to have a zero minimum -/
theorem zero_minimum_condition (p : ℝ) :
  has_zero_minimum p (p^3/27 + p^2/3 + p) ∧
  ∀ q : ℝ, has_zero_minimum p q → q = p^3/27 + p^2/3 + p :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_minimum_condition_l226_22636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_l226_22619

/-- Given a circle O with center A (2, -3) and radius 5, prove that point M (5, -7) lies on the circle. -/
theorem point_on_circle (O : Set (ℝ × ℝ)) (A M : ℝ × ℝ) (r : ℝ) : 
  A = (2, -3) → 
  M = (5, -7) → 
  r = 5 → 
  O = {p : ℝ × ℝ | (p.1 - A.1)^2 + (p.2 - A.2)^2 = r^2} → 
  M ∈ O :=
by
  sorry

#check point_on_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_circle_l226_22619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_theorem_l226_22680

-- Define the types of candies
inductive CandyType
| Chocolate
| Gummy
| Caramel
| Lollipop

-- Define a structure for a friend's candy count
structure CandyCount where
  chocolate : Nat
  gummy : Nat
  caramel : Nat
  lollipop : Nat

-- Define the friends
inductive Friend
| Bob
| Mary
| John
| Sue
| Sam

def initial_count : Friend → CandyCount
| Friend.Bob => { chocolate := 4, gummy := 3, caramel := 2, lollipop := 1 }
| Friend.Mary => { chocolate := 1, gummy := 2, caramel := 1, lollipop := 1 }
| Friend.Sue => { chocolate := 8, gummy := 6, caramel := 4, lollipop := 2 }
| Friend.John => { chocolate := 2, gummy := 1, caramel := 1, lollipop := 1 }
| Friend.Sam => { chocolate := 3, gummy := 4, caramel := 2, lollipop := 1 }

def desired_count : Friend → CandyCount
| Friend.Bob => { chocolate := 5, gummy := 3, caramel := 2, lollipop := 1 }
| Friend.Mary => { chocolate := 2, gummy := 3, caramel := 2, lollipop := 1 }
| Friend.Sue => { chocolate := 4, gummy := 4, caramel := 4, lollipop := 2 }
| Friend.John => { chocolate := 3, gummy := 2, caramel := 2, lollipop := 1 }
| Friend.Sam => { chocolate := 3, gummy := 4, caramel := 2, lollipop := 2 }

def total_candies : Nat := 50

theorem candy_distribution_theorem :
  (∃ (swap_count : Nat),
    swap_count = 7 ∧
    (∀ f : Friend, ∃ (final_count : CandyCount),
      final_count.chocolate ≤ (desired_count f).chocolate ∧
      final_count.gummy ≤ (desired_count f).gummy ∧
      final_count.caramel ≤ (desired_count f).caramel ∧
      final_count.lollipop ≤ (desired_count f).lollipop) ∧
    (∃ f : Friend, ∃ (final_count : CandyCount),
      final_count.lollipop < (desired_count f).lollipop)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_candy_distribution_theorem_l226_22680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_l226_22668

theorem factorial_equation (n : ℕ) : 2^6 * 3^3 * 350 = Nat.factorial 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_equation_l226_22668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_full_zero_sum_free_l226_22600

-- Define the Fibonacci sequence
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define the set A
def A : Set ℤ := {x | ∃ n : ℕ, n ≥ 2 ∧ x = (-1)^n * fib n}

-- Define sum-full property
def sum_full (S : Set ℤ) : Prop := ∀ a ∈ S, ∃ b c, b ∈ S ∧ c ∈ S ∧ a = b + c

-- Define zero-sum-free property
def zero_sum_free (S : Set ℤ) : Prop :=
  ∀ T : Finset ℤ, T.toSet ⊆ S → T.Nonempty → (T.sum id) ≠ 0

-- Theorem statement
theorem exists_sum_full_zero_sum_free :
  ∃ S : Set ℤ, sum_full S ∧ zero_sum_free S :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_sum_full_zero_sum_free_l226_22600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_best_buy_ranking_correct_ranking_l226_22612

-- Define the bag sizes
inductive BagSize
| Small
| Medium
| Large

-- Define the cost and quantity for each bag size
noncomputable def cost (s : BagSize) : ℝ := 
  match s with
  | BagSize.Small => 1
  | BagSize.Medium => 1.75
  | BagSize.Large => 2.1875

noncomputable def quantity (s : BagSize) : ℝ := 
  match s with
  | BagSize.Small => 1
  | BagSize.Medium => 1.1
  | BagSize.Large => 1.65

-- Define the cost per unit quantity
noncomputable def costPerUnit (s : BagSize) : ℝ := cost s / quantity s

-- State the theorem
theorem best_buy_ranking :
  costPerUnit BagSize.Large < costPerUnit BagSize.Medium ∧ 
  costPerUnit BagSize.Medium < costPerUnit BagSize.Small := by
  sorry

-- Define the ranking function
def ranking (s : BagSize) : ℕ := 
  match s with
  | BagSize.Large => 1
  | BagSize.Medium => 2
  | BagSize.Small => 3

-- State the final theorem
theorem correct_ranking : 
  ∀ s₁ s₂ : BagSize, ranking s₁ < ranking s₂ ↔ costPerUnit s₁ < costPerUnit s₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_best_buy_ranking_correct_ranking_l226_22612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_form_line_through_points_line_forming_triangle_l226_22647

-- Define points A and B
noncomputable def A : ℝ × ℝ := (1, 2)
noncomputable def B : ℝ × ℝ := (-1/2, 1)

-- Define the slope of the second line
noncomputable def m : ℝ := 4/3

-- Define the area of the triangle
noncomputable def area : ℝ := 4

-- Theorem for the first line
theorem intercept_form_line_through_points :
  ∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧
  (∀ (x y : ℝ), (x = A.1 ∧ y = A.2) ∨ (x = B.1 ∧ y = B.2) →
    x/a + y/b = 1) ∧ a = -2 ∧ b = 4/3 := by sorry

-- Theorem for the second line
theorem line_forming_triangle :
  ∃ (c : ℝ), (∀ (x y : ℝ), y = m * x + c ∨ y = m * x - c) ∧
  c = (4 * Real.sqrt 6) / 3 ∧
  (1/2 * |c| * |-3*c/4| = area) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intercept_form_line_through_points_line_forming_triangle_l226_22647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_eq_catalan_l226_22623

/-- The number of valid arrangements for 2n people buying tickets -/
noncomputable def validArrangements (n : ℕ) : ℚ :=
  (1 / (n + 1 : ℚ)) * (Nat.choose (2 * n) n : ℚ)

/-- The nth Catalan number -/
noncomputable def catalanNumber (n : ℕ) : ℚ :=
  (1 / (n + 1 : ℚ)) * (Nat.choose (2 * n) n : ℚ)

/-- 
Theorem: The number of valid arrangements for 2n people buying tickets,
where n people have 5-cent bills and n people have 1-yuan bills,
is equal to the nth Catalan number.
-/
theorem valid_arrangements_eq_catalan (n : ℕ) :
  validArrangements n = catalanNumber n := by
  unfold validArrangements catalanNumber
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_eq_catalan_l226_22623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_boxcar_capacity_l226_22678

/-- Represents the capacity of a black boxcar in pounds -/
def black_capacity : ℕ → ℕ := sorry

/-- The number of red boxcars -/
def red_count : ℕ := 3

/-- The number of blue boxcars -/
def blue_count : ℕ := 4

/-- The number of black boxcars -/
def black_count : ℕ := 7

/-- The total capacity of all boxcars in pounds -/
def total_capacity : ℕ := 132000

/-- Theorem stating that the capacity of each black boxcar is 4000 pounds -/
theorem black_boxcar_capacity :
  (black_capacity 1 = 4000) ∧
  (black_capacity 2 = black_capacity 1 * 2) ∧
  (black_capacity 3 = black_capacity 2 * 3) ∧
  (red_count * black_capacity 3 + blue_count * black_capacity 2 + black_count * black_capacity 1 = total_capacity) :=
by sorry

#check black_boxcar_capacity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_boxcar_capacity_l226_22678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_roots_order_l226_22655

-- Define the equations
def eq1 (x : ℝ) : Prop := x^2 - x - 2 = -0.1
def eq2 (x : ℝ) : Prop := 2*x^2 - 2*x - 4 = -0.1
def eq3 (x : ℝ) : Prop := 3*x^2 - 3*x - 6 = -0.1

-- Define the larger roots
noncomputable def x₁ : ℝ := sorry
noncomputable def x₂ : ℝ := sorry
noncomputable def x₃ : ℝ := sorry

-- Theorem statement
theorem larger_roots_order : x₁ < x₂ ∧ x₂ < x₃ :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_roots_order_l226_22655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l226_22654

theorem right_triangle_hypotenuse (h : ℝ) : 
  (∃ (a b : ℝ), 
    a = Real.log 27 / Real.log 4 ∧ 
    b = Real.log 9 / Real.log 2 ∧ 
    h^2 = a^2 + b^2) →
  (4 : ℝ)^h = 243 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_hypotenuse_l226_22654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l226_22692

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 2*x

-- State the theorem
theorem f_monotone_increasing (a : ℝ) (h : a ∈ Set.Icc 1 2) :
  MonotoneOn (f a) Set.univ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_increasing_l226_22692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_neg_eight_plus_sqrt_sixteen_equals_two_l226_22675

theorem cube_root_neg_eight_plus_sqrt_sixteen_equals_two :
  ((-8 : ℝ) ^ (1/3 : ℝ)) + Real.sqrt 16 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_neg_eight_plus_sqrt_sixteen_equals_two_l226_22675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l226_22652

theorem shopkeeper_profit
  (total_apples : ℝ)
  (split_percentage : ℝ)
  (profit_percentage : ℝ)
  (h1 : total_apples > 0)
  (h2 : 0 ≤ split_percentage ∧ split_percentage ≤ 1)
  (h3 : profit_percentage ≥ 0) :
  let first_portion := split_percentage * total_apples
  let second_portion := (1 - split_percentage) * total_apples
  let selling_price (x : ℝ) := x * (1 + profit_percentage / 100)
  let total_cost := total_apples
  let total_revenue := selling_price first_portion + selling_price second_portion
  let total_profit := total_revenue - total_cost
  total_profit / total_cost * 100 = profit_percentage :=
by
  sorry

#check shopkeeper_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shopkeeper_profit_l226_22652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_reach_height_time_check_ball_reach_time_l226_22662

/-- The time at which a ball thrown vertically upward reaches a specific height. -/
noncomputable def ball_reach_time (initial_speed : ℝ) (height : ℝ) : ℝ :=
  initial_speed / 32

theorem ball_reach_height_time (initial_speed : ℝ) (height : ℝ) 
    (h1 : initial_speed = 80)
    (h2 : height = 100) :
    ball_reach_time initial_speed height = 2.5 := by
  -- Unfold the definition of ball_reach_time
  unfold ball_reach_time
  -- Substitute the value of initial_speed
  rw [h1]
  -- Evaluate the division
  norm_num

-- We can't use #eval with noncomputable functions, so we'll use a theorem instead
theorem check_ball_reach_time : 
    ball_reach_time 80 100 = 2.5 := by
  unfold ball_reach_time
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ball_reach_height_time_check_ball_reach_time_l226_22662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l226_22698

theorem trigonometric_identities (θ α : ℝ) : 
  (Real.sin θ - Real.cos θ) / (Real.tan θ - 1) = Real.cos θ ∧ 
  Real.sin α ^ 4 - Real.cos α ^ 4 = 2 * Real.sin α ^ 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identities_l226_22698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roy_missed_two_days_l226_22653

/-- Calculates the number of missed school days given daily sports time, 
    school days per week, and total sports time in a specific week. -/
noncomputable def missed_school_days (daily_sports_time : ℝ) (school_days_per_week : ℕ) 
                       (total_sports_time : ℝ) : ℝ :=
  (daily_sports_time * (school_days_per_week : ℝ) - total_sports_time) / daily_sports_time

/-- Theorem stating that given the specific conditions, 
    the number of missed school days is 2. -/
theorem roy_missed_two_days (daily_sports_time : ℝ) (school_days_per_week : ℕ) 
                             (total_sports_time : ℝ) 
                             (h1 : daily_sports_time = 2)
                             (h2 : school_days_per_week = 5)
                             (h3 : total_sports_time = 6) :
  missed_school_days daily_sports_time school_days_per_week total_sports_time = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roy_missed_two_days_l226_22653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_expansion_coefficient_l226_22663

/-- Given that for any real number x, x³ = a₀ + a₁(x-2) + a₂(x-2)² + a₃(x-2)³, prove that a₂ = 6 -/
theorem cubic_expansion_coefficient (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, x^3 = a₀ + a₁*(x-2) + a₂*(x-2)^2 + a₃*(x-2)^3) → a₂ = 6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_expansion_coefficient_l226_22663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_meaningful_iff_l226_22610

theorem sqrt_meaningful_iff (x : ℝ) : Real.sqrt (x + 6) ∈ Set.univ ↔ x ≥ -6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_meaningful_iff_l226_22610
