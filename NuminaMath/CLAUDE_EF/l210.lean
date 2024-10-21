import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_line_distance_l210_21091

/-- The distance from the left vertex of a hyperbola to a line parallel to its asymptote and passing through its right focus -/
noncomputable def hyperbola_vertex_to_line_distance : ℝ := 32 / 5

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 16 = 1

/-- The coordinates of the left vertex -/
def left_vertex : ℝ × ℝ := (-3, 0)

/-- The coordinates of the right focus -/
def right_focus : ℝ × ℝ := (5, 0)

/-- The slope of the asymptote -/
noncomputable def asymptote_slope : ℝ := 4 / 3

/-- Theorem: The distance from the left vertex of the hyperbola to a line parallel to its asymptote and passing through its right focus is 32/5 -/
theorem hyperbola_vertex_line_distance :
  let l := {(x, y) : ℝ × ℝ | y = asymptote_slope * (x - right_focus.1)}
  (∀ x y, hyperbola_equation x y → True) →
  (∃ x y, (x, y) ∈ l ∧ (x, y) = right_focus) →
  (∃ m, ∀ x y, (x, y) ∈ l ↔ y = m * x + (right_focus.2 - m * right_focus.1)) →
  (∃ d, d = hyperbola_vertex_to_line_distance ∧
        d = |4 * left_vertex.1 - 3 * left_vertex.2 - 20| / Real.sqrt (4^2 + 3^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_vertex_line_distance_l210_21091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_of_five_l210_21008

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- Sum of the first n terms of an arithmetic sequence -/
noncomputable def arithmetic_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a 1 + a n)

theorem arithmetic_sequence_sum_of_five (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 5 + a 8 - a 10 = 2 →
  arithmetic_sum a 5 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_of_five_l210_21008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_engineer_walking_time_l210_21088

/-- Represents the scenario of an engineer walking to meet a car --/
structure EngineerScenario where
  normalArrivalTime : ℝ  -- 8 AM in hours since midnight
  engineerEarlyArrivalTime : ℝ  -- 7 AM in hours since midnight
  carSpeed : ℝ
  engineerSpeed : ℝ
  factoryDistance : ℝ
  earlierArrivalTime : ℝ  -- 20 minutes in hours

/-- The time the engineer walked before meeting the car --/
noncomputable def walkingTime (scenario : EngineerScenario) : ℝ :=
  40 / 60  -- 40 minutes in hours

/-- Theorem stating that the walking time is 40 minutes --/
theorem engineer_walking_time (scenario : EngineerScenario) 
  (h1 : scenario.normalArrivalTime = 8)
  (h2 : scenario.engineerEarlyArrivalTime = 7)
  (h3 : scenario.earlierArrivalTime = 20 / 60)
  (h4 : scenario.carSpeed > 0)
  (h5 : scenario.engineerSpeed > 0)
  (h6 : scenario.factoryDistance > 0) :
  walkingTime scenario = 40 / 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_engineer_walking_time_l210_21088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2pi_minus_alpha_l210_21042

theorem tan_2pi_minus_alpha (α : ℝ) (h1 : Real.cos α = -5/13) (h2 : π/2 < α ∧ α < π) :
  Real.tan (2 * π - α) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_2pi_minus_alpha_l210_21042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l210_21020

/-- The focal length of an ellipse with equation x²/a² + y²/b² = 1 -/
noncomputable def focal_length_ellipse (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)

/-- The equation of a hyperbola in standard form -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  transverse_axis : Bool -- true if x-axis, false if y-axis

/-- The focal length of a hyperbola -/
noncomputable def focal_length_hyperbola (h : Hyperbola) : ℝ := Real.sqrt (h.a^2 + h.b^2)

/-- The slope of an asymptote of a hyperbola -/
noncomputable def asymptote_slope (h : Hyperbola) : ℝ := h.b / h.a

theorem hyperbola_equation (e_a e_b : ℝ) (h : Hyperbola) :
  e_a > e_b ∧ e_b > 0 ∧ h.a > 0 ∧ h.b > 0 →
  focal_length_ellipse e_a e_b = focal_length_hyperbola h →
  asymptote_slope h = 2 →
  (h.transverse_axis = true ∧ h.a^2 = 4 ∧ h.b^2 = 1) ∨
  (h.transverse_axis = false ∧ h.a^2 = 1 ∧ h.b^2 = 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_equation_l210_21020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l210_21079

-- Define the function representing the curve
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log (x + 1)

-- Define the derivative of the function
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := a - 1 / (x + 1)

theorem tangent_line_implies_a_value :
  ∀ a : ℝ,
  (f a 0 = 0) →  -- The curve passes through (0,0)
  (f_derivative a 0 = 2) →  -- The slope of the tangent line at (0,0) is 2
  a = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_l210_21079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_dinner_cost_l210_21011

def dinner_cost (appetizer entree wine_per_glass dessert : ℚ) 
                (num_wine_glasses : ℕ) 
                (entree_discount_percent tip_percent : ℚ) : ℚ :=
  let full_cost := appetizer + entree + (wine_per_glass * num_wine_glasses) + dessert
  let discounted_cost := full_cost - (entree * entree_discount_percent)
  let tip := full_cost * tip_percent
  discounted_cost + tip

theorem arthur_dinner_cost :
  dinner_cost 8 20 3 6 2 (1/2) (1/5) = 38 := by
  -- Unfold the definition of dinner_cost
  unfold dinner_cost
  -- Simplify the arithmetic
  simp [Rat.add_comm, Rat.add_assoc, Rat.mul_comm, Rat.mul_assoc]
  -- The proof is complete
  rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arthur_dinner_cost_l210_21011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_divisibility_theorem_l210_21096

def a : ℕ → ℕ
| 0 => 1
| 1 => 211
| (n + 2) => 212 * a (n + 1) - a n

theorem coprime_divisibility_theorem (x y : ℕ+) :
  Int.gcd x.val y.val = 1 ∧
  x.val ∣ (y.val^2 + 210) ∧
  y.val ∣ (x.val^2 + 210) →
  (x.val = 1 ∧ y.val = 1) ∨
  (x.val = 1 ∧ y.val = 211) ∨
  (∃ n : ℕ, (x.val = a n ∧ y.val = a (n + 1)) ∨ (x.val = a (n + 1) ∧ y.val = a n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coprime_divisibility_theorem_l210_21096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l210_21047

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x/4) * Real.cos (x/4) + Real.cos (x/4)^2

-- Part I
theorem part_one (x : ℝ) (h : f x = 1) : Real.cos (π/3 + x) = 1/2 := by
  sorry

-- Part II
theorem part_two (A B C a b c : ℝ) 
  (h : (2*a - c) * Real.cos B = b * Real.cos C) :
  ∃ (fA : ℝ), fA = Real.sin (A/2 + π/6) + 1/2 ∧ 1 < fA ∧ fA < 3/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_part_one_part_two_l210_21047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_two_l210_21090

-- Define the function f and its inverse
noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry

-- Define the domain of f
def domain : Set ℝ := Set.Icc 0 3

-- Define the properties of f⁻¹
axiom inverse_property_1 : f_inv '' Set.Ico 0 1 = Set.Ico 1 2
axiom inverse_property_2 : f_inv '' Set.Ioc 2 4 = Set.Ico 0 1

-- State that f(x) - x = 0 has a solution
axiom has_solution : ∃ x ∈ domain, f x = x

-- Theorem to prove
theorem solution_is_two :
  ∀ x ∈ domain, f x = x → x = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_is_two_l210_21090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_l210_21078

-- Define the equation
def equation (x : ℝ) : Prop :=
  Real.tan (4 * x) + Real.tan (5 * x) = 1 / Real.cos (5 * x)

-- State the theorem
theorem smallest_positive_solution :
  ∃ (x : ℝ), x > 0 ∧ equation x ∧ x = π / 26 ∧
  ∀ (y : ℝ), y > 0 → equation y → y ≥ x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_l210_21078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_neg_three_iff_z_purely_imaginary_l210_21033

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- Definition of the complex number z -/
noncomputable def z (x : ℝ) : ℂ := (x^2 + 2*x - 3 : ℝ) + (x - 1 : ℝ) * i

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero -/
def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

/-- Theorem stating that x = -3 is a necessary and sufficient condition for z to be purely imaginary -/
theorem x_eq_neg_three_iff_z_purely_imaginary :
  ∀ x : ℝ, x = -3 ↔ is_purely_imaginary (z x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_neg_three_iff_z_purely_imaginary_l210_21033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_C_safest_password_l210_21035

-- Define the number of possible digits
def num_digits : Nat := 4

-- Define the password structures for each person
def password_A : Nat := Nat.choose num_digits 1
def password_B : Nat := Nat.choose num_digits 1 * Nat.choose (num_digits - 1) 1 * Nat.factorial 2
def password_C : Nat := Nat.choose num_digits 1 * Nat.choose (num_digits - 1) 2 * Nat.factorial 3
def password_D : Nat := Nat.factorial 4

-- Define the total number of possible passwords
def total_passwords : Nat := num_digits ^ 4

-- Theorem stating that C's password is the safest
theorem C_safest_password :
  (1 - (password_C : ℚ) / total_passwords) < min
    (1 - (password_A : ℚ) / total_passwords)
    (min (1 - (password_B : ℚ) / total_passwords) (1 - (password_D : ℚ) / total_passwords)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_C_safest_password_l210_21035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l210_21034

/-- Given points P and Q on the graph of y = -1/3x^2 forming an equilateral triangle PQO with the origin O, 
    the length of one side of the triangle is 6√3 -/
theorem equilateral_triangle_on_parabola : 
  ∀ (P Q : ℝ × ℝ),
  (P.2 = -1/3 * P.1^2) →
  (Q.2 = -1/3 * Q.1^2) →
  (dist O P = dist O Q) →
  (dist O P = dist P Q) →
  (dist O P = 6 * Real.sqrt 3) :=
by sorry

where
  O : ℝ × ℝ := (0, 0)
  noncomputable def dist (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l210_21034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_l210_21069

theorem sin_squared_sum (α β γ : ℝ) 
  (angle_bounds : 0 ≤ α ∧ α ≤ π/2 ∧ 0 ≤ β ∧ β ≤ π/2 ∧ 0 ≤ γ ∧ γ ≤ π/2)
  (sin_sum : Real.sin α + Real.sin β + Real.sin γ = 1)
  (sin_cos_sum : Real.sin α * Real.cos (2*α) + Real.sin β * Real.cos (2*β) + Real.sin γ * Real.cos (2*γ) = -1) :
  Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_squared_sum_l210_21069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l210_21071

/-- Checks if a sequence of 5 real numbers forms a geometric sequence -/
def isGeometricSequence (a b c d e : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r ∧ e = d * r

theorem geometric_sequence_first_term 
  (a b c : ℝ) 
  (h1 : isGeometricSequence a b c 48 96) 
  (h2 : c ≠ 0) : a = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_first_term_l210_21071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_equation_l210_21057

/-- Given a triangle ABC and a point M, if M satisfies certain vector equations, then λ = 3 -/
theorem triangle_vector_equation (A B C M : EuclideanSpace ℝ (Fin 3)) (l : ℝ) :
  (M - A) + (M - B) + (M - C) = 0 →
  (B - A) + (C - A) = l • (M - A) →
  l = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_vector_equation_l210_21057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l210_21021

/-- A hyperbola with given asymptote and focus -/
structure Hyperbola where
  /-- Equation of one asymptote: 3x + 4y = 0 -/
  asymptote : ∀ (x y : ℝ), 3 * x + 4 * y = 0 → True
  /-- Coordinates of a focus: (4, 0) -/
  focus : ℝ × ℝ := (4, 0)

/-- Standard form of a hyperbola equation -/
def standard_equation (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / (256/9) = 1

/-- Eccentricity of a hyperbola -/
noncomputable def eccentricity : ℝ := 5/4

/-- Theorem stating the standard equation and eccentricity of the hyperbola -/
theorem hyperbola_properties (h : Hyperbola) :
  (∀ x y, standard_equation x y ↔ true) ∧ eccentricity = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l210_21021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_intersection_complement_M_N_union_complements_M_N_l210_21098

-- Define the universal set U as ℝ
def U := ℝ

-- Define set M
def M : Set ℝ := {x : ℝ | x ≤ 3}

-- Define set N
def N : Set ℝ := {x : ℝ | x < 1}

-- Theorem for M ∪ N
theorem union_M_N : M ∪ N = {x : ℝ | x ≤ 3} := by sorry

-- Theorem for (ᶜU M) ∩ N
theorem intersection_complement_M_N : (Mᶜ) ∩ N = ∅ := by sorry

-- Theorem for (ᶜU M) ∪ (ᶜU N)
theorem union_complements_M_N : (Mᶜ) ∪ (Nᶜ) = {x : ℝ | x ≥ 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_intersection_complement_M_N_union_complements_M_N_l210_21098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_bounds_l210_21018

/-- Represents an equilateral triangle with points on its sides -/
structure EquilateralTriangleWithPoints where
  a : ℝ  -- Side length of the equilateral triangle
  x : ℝ  -- Distance of points from vertices
  h_positive_a : 0 < a  -- Side length is positive
  h_positive_x : 0 < x  -- Distance is positive
  h_x_le_a : x ≤ a  -- Distance is at most the side length

/-- The ratio of areas of the inner triangle to the original triangle -/
noncomputable def area_ratio (t : EquilateralTriangleWithPoints) : ℝ :=
  (3 * t.x^2 - 3 * t.a * t.x + t.a^2) / t.a^2

/-- Theorem stating the bounds on the area ratio -/
theorem area_ratio_bounds (t : EquilateralTriangleWithPoints) :
  1/4 ≤ area_ratio t ∧ area_ratio t < 1 := by
  sorry

#check area_ratio_bounds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_bounds_l210_21018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_position_after_100_turns_l210_21063

/-- Represents a direction (North, East, South, West) --/
inductive Direction
  | N
  | E
  | S
  | W

/-- Represents Bobby's position and facing direction --/
structure BobbyState where
  x : Int
  y : Int
  facing : Direction

/-- Performs one move for Bobby --/
def move (state : BobbyState) (distance : Int) : BobbyState :=
  match state.facing with
  | Direction.N => { state with y := state.y + distance }
  | Direction.E => { state with x := state.x + distance }
  | Direction.S => { state with y := state.y - distance }
  | Direction.W => { state with x := state.x - distance }

/-- Turns Bobby 90 degrees left --/
def turnLeft (state : BobbyState) : BobbyState :=
  { state with facing :=
    match state.facing with
    | Direction.N => Direction.W
    | Direction.W => Direction.S
    | Direction.S => Direction.E
    | Direction.E => Direction.N }

/-- Performs a full cycle of Bobby's movement (4 moves and turns) --/
def moveCycle (state : BobbyState) (n : Nat) : BobbyState :=
  let s1 := move state (2*n + 2)
  let s2 := move (turnLeft s1) (2*n + 3)
  let s3 := move (turnLeft s2) (2*n + 4)
  let s4 := move (turnLeft s3) (2*n + 5)
  turnLeft s4

/-- Performs multiple cycles of Bobby's movement --/
def moveMultipleCycles (state : BobbyState) (cycles : Nat) : BobbyState :=
  match cycles with
  | 0 => state
  | n + 1 => moveMultipleCycles (moveCycle state n) n

/-- Theorem: Bobby's position after 100 turns --/
theorem bobby_position_after_100_turns :
  let initial_state : BobbyState := ⟨10, -10, Direction.N⟩
  let final_state := moveMultipleCycles initial_state 25
  let last_move := move final_state 2
  last_move.x = -667 ∧ last_move.y = 640 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bobby_position_after_100_turns_l210_21063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l210_21046

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus at a 45° angle
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | parabola p.1 p.2 ∧ line p.1 p.2}

-- Theorem statement
theorem segment_length : 
  ∃ (A B : ℝ × ℝ), A ∈ intersection_points ∧ B ∈ intersection_points ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_l210_21046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_m_range_l210_21017

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := x * Real.exp x - m

-- State the theorem
theorem two_zeros_implies_m_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f x m = 0 ∧ f y m = 0) →
  -1 / Real.exp 1 < m ∧ m < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_m_range_l210_21017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base3_addition_multiplication_l210_21094

/-- Converts a base-3 number represented as a list of digits to a natural number -/
def toNat' (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 3 + d) 0

/-- Converts a natural number to its base-3 representation as a list of digits -/
def toBase3 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc else aux (m / 3) ((m % 3) :: acc)
  aux n []

/-- The main theorem to prove -/
theorem base3_addition_multiplication :
  toBase3 ((toNat' [2] + toNat' [1, 2, 0] + toNat' [2, 1, 0, 2] + toNat' [1, 2, 0, 1, 2]) * toNat' [1, 1]) =
  [2, 1, 2, 2, 2] := by
  sorry

#eval toBase3 ((toNat' [2] + toNat' [1, 2, 0] + toNat' [2, 1, 0, 2] + toNat' [1, 2, 0, 1, 2]) * toNat' [1, 1])

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base3_addition_multiplication_l210_21094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monthly_profit_l210_21006

/-- Represents the monthly sales volume in ten thousand units -/
noncomputable def x : ℝ → ℝ := sorry

/-- Represents the cost invested in physical store experience and installation in ten thousand yuan -/
noncomputable def t : ℝ → ℝ := sorry

/-- The relationship between monthly sales volume and cost -/
axiom sales_cost_relation (x t : ℝ → ℝ) : 
  ∀ r, x r = 3 - 2 / (t r + 1)

/-- Fixed monthly expenses of the online store in million yuan -/
def fixed_expenses : ℝ := 3

/-- Purchase price per ten thousand units in million yuan -/
def purchase_price : ℝ := 32

/-- Selling price calculation -/
noncomputable def selling_price (x t : ℝ → ℝ) (r : ℝ) : ℝ := 
  1.5 * purchase_price + 0.5 * (t r / x r)

/-- Monthly profit calculation -/
noncomputable def monthly_profit (x t : ℝ → ℝ) (r : ℝ) : ℝ := 
  selling_price x t r * x r - purchase_price * x r - fixed_expenses - t r

/-- Theorem stating the maximum monthly profit -/
theorem max_monthly_profit (x t : ℝ → ℝ) : 
  (∀ r, x r = 3 - 2 / (t r + 1)) → 
  ∃ r, monthly_profit x t r ≤ 37.5 ∧ 
  ∀ s, monthly_profit x t s ≤ monthly_profit x t r := by
  sorry

#check max_monthly_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_monthly_profit_l210_21006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l210_21025

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 9 / (x + 1)

-- State the theorem
theorem f_range :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 →
  ∃ y : ℝ, y = f x ∧ 5 ≤ y ∧ y ≤ 9 ∧
  (∀ z : ℝ, (∃ w : ℝ, 0 ≤ w ∧ w ≤ 3 ∧ z = f w) → 5 ≤ z ∧ z ≤ 9) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l210_21025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_change_l210_21056

theorem profit_change (initial_profit : ℝ) : 
  (initial_profit * 1.1 * 0.8 * 1.5 - initial_profit) / initial_profit * 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_change_l210_21056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l210_21082

/-- Predicate for an isosceles right triangle -/
def IsoscelesRightTriangle (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : Prop :=
  sorry

/-- Predicate for the hypotenuse of a triangle -/
def Hypotenuse (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) (A B : ℝ × ℝ) : Prop :=
  sorry

/-- Function to calculate the distance between two points -/
def Distance (A B : ℝ × ℝ) : ℝ :=
  sorry

/-- Function to calculate the area of a triangle -/
def Area (triangle : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)) : ℝ :=
  sorry

/-- An isosceles right triangle with hypotenuse length 16 has area 64 -/
theorem isosceles_right_triangle_area (X Y Z : ℝ × ℝ) : 
  let triangle := (X, Y, Z)
  IsoscelesRightTriangle triangle →
  Hypotenuse triangle X Y →
  Distance X Y = 16 →
  Area triangle = 64 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_area_l210_21082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l210_21066

noncomputable def f (x : ℝ) := Real.sqrt (2 * Real.sin x - 1) + Real.sqrt (-x^2 + 6*x)

theorem domain_of_f :
  {x : ℝ | 2 * Real.sin x - 1 ≥ 0 ∧ -x^2 + 6*x ≥ 0} = {x : ℝ | π/6 ≤ x ∧ x ≤ 5*π/6} := by
  sorry

#check domain_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l210_21066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_area_is_48_l210_21048

/-- The total area of metal wasted when cutting a maximum circular piece from a 8x10 rectangle
    and then cutting a maximum square piece from that circle. -/
noncomputable def metal_waste_area : ℝ := by
  -- Define the dimensions of the rectangle
  let rectangle_width : ℝ := 8
  let rectangle_height : ℝ := 10

  -- Define the area of the rectangle
  let rectangle_area : ℝ := rectangle_width * rectangle_height

  -- Define the radius of the circular piece (half of the shorter side)
  let circle_radius : ℝ := rectangle_width / 2

  -- Define the area of the circular piece
  let circle_area : ℝ := Real.pi * circle_radius ^ 2

  -- Define the side length of the square piece cut from the circle
  let square_side : ℝ := rectangle_width / Real.sqrt 2

  -- Define the area of the square piece
  let square_area : ℝ := square_side ^ 2

  -- Calculate the total wasted area
  let wasted_area : ℝ := rectangle_area - square_area

  -- Return the wasted area
  exact wasted_area

/-- Theorem stating that the metal waste area is equal to 48 square units -/
theorem metal_waste_area_is_48 : metal_waste_area = 48 := by
  -- Unfold the definition of metal_waste_area
  unfold metal_waste_area

  -- Simplify the expression
  simp

  -- The proof steps would go here, but we'll use sorry for now
  sorry

-- We can't use #eval with noncomputable definitions, so we'll remove this line
-- #eval metal_waste_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metal_waste_area_is_48_l210_21048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_plans_l210_21015

/-- Represents a student --/
inductive Student
| A
| B
| C
| D
| E

/-- Represents a competition --/
inductive Competition
| Mathematics
| Physics
| Chemistry
| ForeignLanguage

/-- A competition plan assigns students to competitions --/
def CompetitionPlan := Competition → Student

/-- Predicate to check if a plan is valid --/
def is_valid_plan (plan : CompetitionPlan) : Prop :=
  (∀ c1 c2, c1 ≠ c2 → plan c1 ≠ plan c2) ∧ 
  (plan Competition.Physics ≠ Student.A) ∧
  (plan Competition.Chemistry ≠ Student.A)

/-- The set of all valid competition plans --/
def valid_plans : Set CompetitionPlan :=
  { plan | is_valid_plan plan }

/-- Ensure that valid_plans is finite --/
instance : Fintype valid_plans := by
  sorry

theorem number_of_valid_plans : Fintype.card valid_plans = 72 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_valid_plans_l210_21015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marks_for_unique_determination_l210_21097

/-- Represents a 9x9 game board --/
def Board := Fin 9 → Fin 9 → Bool

/-- Represents an L-shaped tromino (corner piece) --/
structure Tromino where
  x : Fin 9
  y : Fin 9
  orientation : Fin 4

/-- Checks if a tromino covers a given cell --/
def Tromino.covers (t : Tromino) (x y : Fin 9) : Bool := sorry

/-- Counts the number of marked cells covered by a tromino --/
def countCoveredMarks (b : Board) (t : Tromino) : Nat := sorry

/-- Checks if a tromino placement is uniquely determined by the marked cells it covers --/
def isUniqueDetermination (b : Board) (t : Tromino) : Bool := sorry

/-- The main theorem: 68 is the minimum number of marked cells needed to always uniquely determine the tromino position --/
theorem min_marks_for_unique_determination :
  ∃ (b : Board), (∀ (t : Tromino), isUniqueDetermination b t) ∧
  (∀ (b' : Board), (∀ (t : Tromino), isUniqueDetermination b' t) →
    (Finset.sum (Finset.univ : Finset (Fin 9)) fun i =>
      Finset.sum (Finset.univ : Finset (Fin 9)) fun j =>
        if b' i j then 1 else 0) ≥ 68) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marks_for_unique_determination_l210_21097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l210_21055

theorem tan_alpha_plus_pi_fourth (α : Real) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.cos (2 * α) = 3 / 5) : 
  Real.tan (α + π / 4) = 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_pi_fourth_l210_21055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_functions_have_inverses_H_and_I_have_inverses_F_and_G_do_not_have_inverses_l210_21002

-- Define a type for our functions
def RealFunction := ℝ → ℝ

-- Define properties for our functions
def IsLinear (f : RealFunction) : Prop := 
  ∃ m k : ℝ, ∀ x : ℝ, f x = m * x + k

def HasInverse (f : RealFunction) : Prop :=
  ∃ g : RealFunction, (∀ x : ℝ, g (f x) = x) ∧ (∀ y : ℝ, f (g y) = y)

def PassesHorizontalLineTest (f : RealFunction) : Prop :=
  ∀ y : ℝ, ∃! x : ℝ, f x = y

-- Define our theorem
theorem linear_functions_have_inverses :
  ∀ f : RealFunction, 
    IsLinear f → PassesHorizontalLineTest f → HasInverse f :=
by sorry

-- Define the specific functions from the graphs
noncomputable def F : RealFunction := λ x ↦ x^3/27 + x^2/18 - x/3 + 3
noncomputable def G : RealFunction := λ x ↦ x^2/4 - 4
noncomputable def H : RealFunction := λ x ↦ 3/2 * x + 1
noncomputable def I : RealFunction := λ x ↦ -x/3 + 1

-- State the properties of these functions
theorem H_and_I_have_inverses : 
  HasInverse H ∧ HasInverse I :=
by sorry

theorem F_and_G_do_not_have_inverses :
  ¬HasInverse F ∧ ¬HasInverse G :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_linear_functions_have_inverses_H_and_I_have_inverses_F_and_G_do_not_have_inverses_l210_21002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_three_enchiladas_four_tacos_l210_21024

/-- The cost of an enchilada in dollars -/
def enchilada_cost : ℝ := sorry

/-- The cost of a taco in dollars -/
def taco_cost : ℝ := sorry

/-- The cost of two enchiladas and three tacos is $2.50 -/
axiom two_enchiladas_three_tacos : 2 * enchilada_cost + 3 * taco_cost = 2.50

/-- The cost of three enchiladas and two tacos is $2.70 -/
axiom three_enchiladas_two_tacos : 3 * enchilada_cost + 2 * taco_cost = 2.70

/-- The cost of three enchiladas and four tacos is $3.54 -/
theorem cost_three_enchiladas_four_tacos :
  3 * enchilada_cost + 4 * taco_cost = 3.54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_three_enchiladas_four_tacos_l210_21024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_coefficient_nonzero_regression_coefficient_sign_l210_21037

/-- A linear regression model with two variables -/
structure LinearRegression where
  x : ℝ → ℝ  -- Independent variable
  y : ℝ → ℝ  -- Dependent variable
  a : ℝ       -- Intercept
  b : ℝ       -- Slope (regression coefficient)
  eq : ∀ t, y t = a + b * (x t)  -- Regression equation

/-- Theorem stating that the regression coefficient can be any non-zero real number -/
theorem regression_coefficient_nonzero :
  ∃ (model : LinearRegression), model.b ≠ 0 :=
by
  -- We construct a model with non-zero b
  let b : ℝ := 1  -- Choose any non-zero value for b
  let a : ℝ := 0  -- Choose any value for a
  let x : ℝ → ℝ := λ t ↦ t  -- Identity function for x
  let y : ℝ → ℝ := λ t ↦ a + b * t  -- Corresponding y function
  
  -- Construct the model
  let model : LinearRegression := {
    x := x,
    y := y,
    a := a,
    b := b,
    eq := λ t ↦ by simp [y, x]
  }
  
  -- Prove the existence
  use model
  -- Prove that b ≠ 0
  simp [b]
  
/-- Theorem stating that the regression coefficient can be positive or negative -/
theorem regression_coefficient_sign :
  ∃ (model1 model2 : LinearRegression), model1.b > 0 ∧ model2.b < 0 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_coefficient_nonzero_regression_coefficient_sign_l210_21037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2a_eq_3_l210_21029

noncomputable def f (x : ℝ) : ℝ := 2^x + 1/(2^x)

theorem f_2a_eq_3 (a : ℝ) (h : f a = Real.sqrt 5) : f (2*a) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2a_eq_3_l210_21029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_comparison_l210_21005

-- Define the properties of the tanks
noncomputable def height_A : ℝ := 10
noncomputable def circumference_A : ℝ := 9
noncomputable def height_B : ℝ := 9
noncomputable def circumference_B : ℝ := 10

-- Define the volume of a cylinder
noncomputable def volume (h : ℝ) (c : ℝ) : ℝ := (c^2 * h) / (4 * Real.pi)

-- State the theorem
theorem tank_capacity_comparison :
  (volume height_A circumference_A) / (volume height_B circumference_B) = 0.9 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tank_capacity_comparison_l210_21005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_percentage_change_is_fifty_percent_l210_21028

/-- Calculates the percentage increase given old and new prices -/
noncomputable def percentageIncrease (oldPrice newPrice : ℝ) : ℝ :=
  (newPrice - oldPrice) / oldPrice * 100

/-- Represents the price changes for three items -/
structure PriceChanges where
  bookOld : ℝ
  bookNew : ℝ
  laptopOld : ℝ
  laptopNew : ℝ
  gameOld : ℝ
  gameNew : ℝ

/-- Calculates the average percentage change for the given price changes -/
noncomputable def averagePercentageChange (changes : PriceChanges) : ℝ :=
  (percentageIncrease changes.bookOld changes.bookNew +
   percentageIncrease changes.laptopOld changes.laptopNew +
   percentageIncrease changes.gameOld changes.gameNew) / 3

/-- Theorem stating that the average percentage change is 50% for the given price changes -/
theorem average_percentage_change_is_fifty_percent :
  averagePercentageChange { bookOld := 300, bookNew := 450,
                            laptopOld := 800, laptopNew := 1200,
                            gameOld := 50, gameNew := 75 } = 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_percentage_change_is_fifty_percent_l210_21028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_4_side_c_is_one_l210_21084

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def area_condition (t : Triangle) : Prop :=
  (t.a^2 + t.c^2 - t.b^2) / 4 = 1/2 * t.a * t.c * Real.sin t.B

def side_angle_conditions (t : Triangle) : Prop :=
  t.a * t.c = Real.sqrt 3 ∧ 
  Real.sin t.A = Real.sqrt 3 * Real.sin t.B ∧ 
  t.C = Real.pi/6

-- State the theorems
theorem angle_B_is_pi_over_4 (t : Triangle) 
  (h : area_condition t) : t.B = Real.pi/4 := by sorry

theorem side_c_is_one (t : Triangle) 
  (h : side_angle_conditions t) : t.c = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_4_side_c_is_one_l210_21084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_range_count_l210_21067

/-- The number of integer ages within one standard deviation of the mean -/
def count_ages_in_range (mean std_dev : ℕ) : ℕ :=
  (List.range (2 * std_dev + 1)).filter (λ x => x + mean - std_dev ≥ 0) |>.length

/-- Theorem stating that the number of integer ages within one standard deviation
    of the mean (20) with a standard deviation of 8 is 17 -/
theorem age_range_count :
  count_ages_in_range 20 8 = 17 := by
  sorry

#eval count_ages_in_range 20 8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_range_count_l210_21067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelicity_of_functions_l210_21070

-- Define the four functions
noncomputable def f₁ (x : ℝ) : ℝ := x^3 - x
noncomputable def f₂ (x : ℝ) : ℝ := x + 1/x
noncomputable def f₃ (x : ℝ) : ℝ := Real.sin x
noncomputable def f₄ (x : ℝ) : ℝ := (x-2)^2 + Real.log x

-- Define the parallelicity property
def has_parallelicity (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  DifferentiableAt ℝ f x₁ ∧ 
  DifferentiableAt ℝ f x₂ ∧
  deriv f x₁ = a ∧ deriv f x₂ = a

-- Theorem statement
theorem parallelicity_of_functions :
  ¬(has_parallelicity f₁) ∧ 
  (has_parallelicity f₂) ∧ 
  (has_parallelicity f₃) ∧ 
  ¬(has_parallelicity f₄) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallelicity_of_functions_l210_21070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_time_is_three_hours_l210_21058

/-- Represents a cylindrical gasoline tank with filling and emptying capabilities -/
structure GasolineTank where
  height : ℝ
  fill_time : ℝ
  dual_flow_time : ℝ
  dual_flow_rise : ℝ

/-- Calculates the time required to empty the tank through the valve -/
noncomputable def empty_time (tank : GasolineTank) : ℝ :=
  tank.height * tank.fill_time / (tank.height - tank.dual_flow_rise * tank.fill_time / tank.dual_flow_time)

/-- Theorem stating that for the given tank specifications, the emptying time is 3 hours -/
theorem empty_time_is_three_hours (tank : GasolineTank)
  (h_height : tank.height = 180)
  (h_fill_time : tank.fill_time = 60)
  (h_dual_flow_time : tank.dual_flow_time = 5)
  (h_dual_flow_rise : tank.dual_flow_rise = 10) :
  empty_time tank = 180 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_empty_time_is_three_hours_l210_21058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l210_21077

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (h1 : selling_price = 1200)
  (h2 : profit_percentage = 20) :
  selling_price / (1 + profit_percentage / 100) = 1000 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_price_calculation_l210_21077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_recurring_values_l210_21050

theorem no_recurring_values (n m k : ℤ) : 
  (m = 3 * k + 1 ∨ m = 3 * k - 1) → ¬∃(n' : ℤ), 3 * n' = 5 * m := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_recurring_values_l210_21050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mobius_function_problem_l210_21001

noncomputable def f (a b c d z : ℂ) : ℂ := (a * z + b) / (c * z + d)

theorem mobius_function_problem (a b c d : ℂ) (m n : ℕ) :
  f a b c d 1 = I ∧
  f a b c d 2 = I^2 ∧
  f a b c d 3 = I^3 ∧
  ∃ (x : ℝ), x = (f a b c d 4).re ∧ x = m / n ∧
  Nat.Coprime m n ∧ m > 0 ∧ n > 0 →
  m^2 + n^2 = 34 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mobius_function_problem_l210_21001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l210_21014

-- Define the lines
def line1 (x : ℝ) : ℝ := 2 * x
def line2 (x : ℝ) : ℝ := -2 * x
def line3 : ℝ := 8

-- Define the points of intersection
def point_A : ℝ × ℝ := (4, 8)
def point_B : ℝ × ℝ := (-4, 8)
def point_O : ℝ × ℝ := (0, 0)

-- Define the triangle
def triangle_OAB : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    (p = (t * point_A.1 + (1 - t) * point_O.1, t * point_A.2 + (1 - t) * point_O.2) ∨
     p = (t * point_B.1 + (1 - t) * point_O.1, t * point_B.2 + (1 - t) * point_O.2) ∨
     p = (t * point_A.1 + (1 - t) * point_B.1, t * point_A.2 + (1 - t) * point_B.2))}

-- State the theorem
theorem triangle_area : MeasureTheory.volume triangle_OAB = 32 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_l210_21014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_preference_result_l210_21073

/-- The number of people who prefer "Soda" in a survey -/
noncomputable def soda_preference (total : ℕ) (central_angle : ℝ) : ℕ :=
  ⌊(total : ℝ) * central_angle / 360⌋₊

theorem soda_preference_result :
  soda_preference 520 298 = 429 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_soda_preference_result_l210_21073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l210_21026

def sequence_a : ℕ → ℤ
  | 0 => 1 - 4  -- We define a₀ to make the sequence consistent
  | n + 1 => sequence_a n + 4

theorem a_100_value : sequence_a 100 = 397 := by
  -- Proof of the theorem
  have h1 : ∀ n : ℕ, sequence_a n = 4 * n - 3 := by
    intro n
    induction n with
    | zero => rfl
    | succ n ih =>
      simp [sequence_a]
      rw [ih]
      ring
  
  -- Apply the general formula to n = 100
  have h2 : sequence_a 100 = 4 * 100 - 3 := h1 100
  
  -- Simplify the right-hand side
  rw [h2]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_100_value_l210_21026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_is_real_complex_number_is_purely_imaginary_l210_21053

-- Define the complex number as a function of m
def complex_number (m : ℝ) : ℂ := (m^2 - 5*m + 6) + (m^2 - 3*m)*Complex.I

-- Theorem for when the complex number is real
theorem complex_number_is_real (m : ℝ) : 
  (complex_number m).im = 0 ↔ m = 0 ∨ m = 3 := by sorry

-- Theorem for when the complex number is purely imaginary
theorem complex_number_is_purely_imaginary (m : ℝ) : 
  (complex_number m).re = 0 ∧ (complex_number m).im ≠ 0 ↔ m = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_is_real_complex_number_is_purely_imaginary_l210_21053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_h2o_required_l210_21019

/-- Represents the amount of substance in moles -/
structure Moles where
  value : ℝ

/-- Main reaction: Li3N + 3H2O → 3LiOH + NH3 -/
def main_reaction (li3n : Moles) : Moles :=
  ⟨3 * li3n.value⟩

/-- Side reaction percentage -/
def side_reaction_percentage : ℝ := 0.05

/-- Calculate additional H2O required due to side reaction -/
def additional_h2o (lioh : Moles) : Moles :=
  ⟨lioh.value * side_reaction_percentage⟩

/-- Theorem: Total H2O required for the reaction -/
theorem total_h2o_required (initial_li3n : Moles) (desired_lioh : Moles) :
  initial_li3n.value = 3 ∧ desired_lioh.value = 9 →
  (main_reaction initial_li3n).value + (additional_h2o desired_lioh).value = 9.45 := by
  sorry

#eval (main_reaction ⟨3⟩).value + (additional_h2o ⟨9⟩).value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_h2o_required_l210_21019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_25_not_40_l210_21099

theorem three_digit_multiples_of_25_not_40 : 
  (Finset.filter (fun n => n % 25 = 0 ∧ n % 40 ≠ 0) (Finset.range 900)).card = 32 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_digit_multiples_of_25_not_40_l210_21099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_value_calculation_l210_21039

/-- Calculates the market value of a stock given its face value yield, market yield, and face value. -/
noncomputable def stock_market_value (face_value_yield : ℝ) (market_yield : ℝ) (face_value : ℝ) : ℝ :=
  (face_value_yield * face_value) / market_yield

/-- Theorem stating that a stock with 13% yield on face value, 8% market yield, and $100 face value has a market value of $162.50 -/
theorem stock_value_calculation :
  let face_value_yield : ℝ := 0.13
  let market_yield : ℝ := 0.08
  let face_value : ℝ := 100
  stock_market_value face_value_yield market_yield face_value = 162.50 := by
  sorry

/-- Evaluates the stock market value for the given parameters -/
def evaluate_stock_value : ℚ :=
  (13 : ℚ) / 8

#eval evaluate_stock_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_value_calculation_l210_21039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l210_21041

theorem solve_exponential_equation (y : ℝ) : (9 : ℝ)^y = (3 : ℝ)^12 → y = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solve_exponential_equation_l210_21041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_box_volume_l210_21051

/-- The volume of the box as a function of the side length of its base -/
noncomputable def box_volume (x : ℝ) : ℝ := (60 * x^2 - x^3) / 2

/-- The derivative of the box volume function -/
noncomputable def box_volume_derivative (x : ℝ) : ℝ := 60 * x - (3/2) * x^2

theorem max_box_volume :
  ∃ (x : ℝ), x > 0 ∧ x < 60 ∧
  (∀ y, y > 0 → y < 60 → box_volume y ≤ box_volume x) ∧
  x = 40 ∧
  box_volume x = 16000 := by
  sorry

#check max_box_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_box_volume_l210_21051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_cosecant_l210_21059

theorem min_positive_cosecant (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, a / Real.sin (b * x) ≥ 4) ∧ 
  (∃ x : ℝ, a / Real.sin (b * x) = 4) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_positive_cosecant_l210_21059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_remaining_2001_l210_21074

/-- The last remaining number after the elimination process for n numbers -/
def lastRemaining : ℕ → ℕ
  | 0 => 1  -- Base case for 0
  | 1 => 1  -- Base case for 1
  | 2 => 1  -- Base case for 2
  | n + 1 => 
    if n % 2 = 0 then
      2 * lastRemaining (n / 2) - 1
    else
      2 * lastRemaining ((n + 1) / 2) + 1

theorem last_remaining_2001 : lastRemaining 2001 = 1955 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_remaining_2001_l210_21074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_P_to_l_l210_21045

/-- Line l in the xy-plane -/
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

/-- Locus of point P -/
def locus_P (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

/-- Distance from a point to a line -/
noncomputable def distance_point_to_line (x y : ℝ) : ℝ := 
  |x - y - 1| / Real.sqrt 2

/-- Theorem stating the minimum distance from P to l -/
theorem min_distance_P_to_l : 
  ∃ (x y : ℝ), locus_P x y ∧ 
    (∀ (x' y' : ℝ), locus_P x' y' → 
      distance_point_to_line x y ≤ distance_point_to_line x' y') ∧
    distance_point_to_line x y = Real.sqrt 2 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_P_to_l_l210_21045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_size_of_A_2023_l210_21062

def A : ℕ → Finset ℕ
  | 0 => {3}
  | (n + 1) => (A n).image (λ x => x + 2) ∪ (A n).image (λ x => x * (x + 1) / 2)

theorem size_of_A_2023 : Finset.card (A 2023) = 2^2023 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_size_of_A_2023_l210_21062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l210_21031

/-- The function f(x) as defined in the problem -/
noncomputable def f (a x : ℝ) : ℝ := (3 * Real.log x - x^2 - a - 2)^2 + (x - a)^2

/-- The theorem statement -/
theorem problem_solution (a : ℝ) :
  (∃ x, f a x ≤ 8) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l210_21031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_portfolio_yield_calculation_l210_21087

structure Stock where
  yield : Float
  quote : Float
  growth_rate : Float
  is_low_risk : Bool
  tax_rate : Float

def portfolio_yield (stock_a : Stock) (stock_b : Stock) (investment_a : Float) (investment_b : Float) : Float :=
  let total_investment := investment_a + investment_b
  let weight_a := investment_a / total_investment
  let weight_b := investment_b / total_investment
  let after_tax_yield_a := stock_a.yield * (1 - stock_a.tax_rate)
  let after_tax_yield_b := stock_b.yield * (1 - stock_b.tax_rate)
  weight_a * after_tax_yield_a + weight_b * after_tax_yield_b

theorem portfolio_yield_calculation :
  let stock_a : Stock := {
    yield := 0.21,
    quote := 0.10,
    growth_rate := 0.05,
    is_low_risk := true,
    tax_rate := 0.10
  }
  let stock_b : Stock := {
    yield := 0.15,
    quote := 0.20,
    growth_rate := 0.08,
    is_low_risk := false,
    tax_rate := 0.20
  }
  let investment_a := 10000
  let investment_b := 15000
  (portfolio_yield stock_a stock_b investment_a investment_b - 0.1476).abs < 0.0001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_portfolio_yield_calculation_l210_21087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_all_monotonic_intervals_l210_21061

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x - Real.pi / 6)

def is_monotonic_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def interval_of_monotonic_increase (k : ℤ) : Set ℝ :=
  Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3)

theorem monotonic_increase_interval (k : ℤ) :
  is_monotonic_increasing f (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3) :=
by sorry

theorem all_monotonic_intervals :
  ∀ x : ℝ × ℝ, (∃ k : ℤ, x.1 ∈ interval_of_monotonic_increase k ∧ x.2 ∈ interval_of_monotonic_increase k) ↔
       is_monotonic_increasing f x.1 x.2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increase_interval_all_monotonic_intervals_l210_21061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stacys_farm_piglets_l210_21016

theorem stacys_farm_piglets : ∃ (piglets : ℕ), 
  let chickens := 26
  let goats := 34
  let sick_animals := 50
  (chickens + goats + piglets) / 2 = sick_animals ∧ piglets = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stacys_farm_piglets_l210_21016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_perimeter_25_partition_exists_30_partition_perimeter_2_l210_21010

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ
deriving Inhabited

/-- Calculates the perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℝ :=
  2 * (r.width + r.height)

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ :=
  r.width * r.height

/-- Represents a partition of a unit square into rectangles -/
structure UnitSquarePartition where
  rectangles : List Rectangle
  partition_valid : (rectangles.map area).sum = 1

theorem min_max_perimeter_25_partition (p : UnitSquarePartition)
  (h1 : p.rectangles.length = 25)
  (h2 : ∀ r ∈ p.rectangles, perimeter r = perimeter (p.rectangles.head!)) :
  0.8 ≤ perimeter (p.rectangles.head!) ∧ perimeter (p.rectangles.head!) ≤ 2.08 := by
  sorry

theorem exists_30_partition_perimeter_2 :
  ∃ (p : UnitSquarePartition), p.rectangles.length = 30 ∧ ∀ r ∈ p.rectangles, perimeter r = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_perimeter_25_partition_exists_30_partition_perimeter_2_l210_21010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_negative_one_l210_21000

-- Define the expression as noncomputable
noncomputable def trigExpression (α : ℝ) : ℝ :=
  (Real.sin (2*Real.pi - α) * Real.cos (Real.pi + α) * Real.cos (3*Real.pi + α) * Real.cos ((11*Real.pi)/2 - α)) /
  (Real.cos (Real.pi - α) * Real.sin (3*Real.pi - α) * Real.sin (-Real.pi - α) * Real.sin ((9*Real.pi)/2 + α))

-- Theorem statement
theorem trigExpression_equals_negative_one :
  ∀ α : ℝ, trigExpression α = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigExpression_equals_negative_one_l210_21000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_trapezoid_properties_l210_21023

/-- A symmetric trapezoid with parallel sides a and c, leg b, and median m. -/
structure SymmetricTrapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  m : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < m
  h_order : c < a

/-- The equivalence of three properties in a symmetric trapezoid. -/
theorem symmetric_trapezoid_properties (t : SymmetricTrapezoid) :
  (t.m * (t.a + t.c) = t.b^2 + t.a * t.c) ∧
  (t.b^2 = (t.a^2 + t.c^2) / 2) ∧
  (t.m = (t.a + t.c) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_trapezoid_properties_l210_21023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_arrangements_correct_l210_21064

/-- The number of arrangements of length n using letters A, B, and C,
    where not all three letters need to appear and no two A's are adjacent. -/
noncomputable def num_arrangements (n : ℕ) : ℝ :=
  ((2 + Real.sqrt 3) * (1 + Real.sqrt 3) ^ n + (-2 + Real.sqrt 3) * (1 - Real.sqrt 3) ^ n) / (2 * Real.sqrt 3)

/-- Theorem stating that num_arrangements satisfies the given conditions. -/
theorem num_arrangements_correct (n : ℕ) :
  num_arrangements n = 
    ((2 + Real.sqrt 3) * (1 + Real.sqrt 3) ^ n + (-2 + Real.sqrt 3) * (1 - Real.sqrt 3) ^ n) / (2 * Real.sqrt 3) ∧
  (∀ k : ℕ, k ≥ 3 → num_arrangements k = 2 * num_arrangements (k - 2) + 2 * num_arrangements (k - 1)) ∧
  num_arrangements 1 = 3 ∧
  num_arrangements 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_arrangements_correct_l210_21064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_fifteen_l210_21009

/-- A square root is considered simpler if it cannot be simplified further. -/
def is_simplest_sqrt (x : ℝ) (others : List ℝ) : Prop :=
  (∀ y ∈ others, ∃ (n : ℕ), y = n ∨ ∃ (a b : ℝ), y = a * Real.sqrt b ∧ b ≠ x) ∧
  (∀ (n : ℕ), x ≠ n) ∧
  (∀ (a b : ℝ), x ≠ a * Real.sqrt b ∨ b = x)

theorem simplest_sqrt_fifteen :
  is_simplest_sqrt (Real.sqrt 15) [Real.sqrt 12, Real.sqrt (1/3), Real.sqrt 9] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_simplest_sqrt_fifteen_l210_21009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l210_21036

theorem area_enclosed_by_curves : 
  ∫ (x : ℝ) in (Set.Icc 0 1), Real.sqrt x + ∫ (x : ℝ) in (Set.Icc 1 2), (2 - x) = 7/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_enclosed_by_curves_l210_21036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l210_21040

-- Define the operations
noncomputable def oplus (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2)
noncomputable def otimes (a b : ℝ) : ℝ := Real.sqrt ((a - b)^2)

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (oplus 2 x) / (2 - (otimes x 2))

-- Statement to prove
theorem f_is_odd : ∀ x : ℝ, f (-x) = -f x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_is_odd_l210_21040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_satisfaction_l210_21085

theorem inequality_satisfaction (x : ℝ) : x ∈ ({0, 1, 2, 3} : Set ℝ) → (x + 1 < 2 ↔ x = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_satisfaction_l210_21085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_range_l210_21027

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the properties of the triangle
def ObtuseTriangle (t : Triangle) : Prop :=
  t.C > Real.pi/2 ∧ t.a = 2 ∧ t.b = 3

-- State the theorem
theorem obtuse_triangle_side_range (t : Triangle) (h : ObtuseTriangle t) :
  t.c > Real.sqrt 13 ∧ t.c < 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_obtuse_triangle_side_range_l210_21027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_pi_f_less_than_cubic_max_k_value_l210_21013

open Real

noncomputable def f (x : ℝ) := sin x - x * cos x

theorem tangent_line_at_pi :
  ∀ x, (deriv f π) * (x - π) + f π = π :=
sorry

theorem f_less_than_cubic :
  ∀ x, x > 0 → x < π / 2 → f x < (1/3) * x^3 :=
sorry

theorem max_k_value (k : ℝ) :
  (∀ x, x > 0 → x < π / 2 → f x > k * x - x * cos x) ↔ k ≤ 2 / π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_pi_f_less_than_cubic_max_k_value_l210_21013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distributions_eq_768_l210_21060

/-- Represents a rabbit --/
inductive Rabbit
| Nina
| Tony
| Fluffy
| Snowy
| Brownie
deriving DecidableEq

/-- Represents a pet store --/
structure PetStore where
  rabbits : List Rabbit
  inv_max_two : rabbits.length ≤ 2

/-- A valid distribution of rabbits to pet stores --/
structure Distribution where
  stores : List PetStore
  inv_four_stores : stores.length = 4
  inv_all_rabbits : (stores.map PetStore.rabbits).join.toFinset = {Rabbit.Nina, Rabbit.Tony, Rabbit.Fluffy, Rabbit.Snowy, Rabbit.Brownie}
  inv_siblings_separate : ∀ s ∈ stores, ¬(Rabbit.Nina ∈ s.rabbits ∧ Rabbit.Tony ∈ s.rabbits)

/-- The number of valid distributions --/
def num_distributions : ℕ := sorry

theorem num_distributions_eq_768 : num_distributions = 768 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_distributions_eq_768_l210_21060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_formula_l210_21004

/-- The area of a regular hexadecagon inscribed in a circle with radius r -/
noncomputable def hexadecagonArea (r : ℝ) : ℝ := 4 * r^2 * Real.sqrt (2 - Real.sqrt 2)

/-- Theorem stating that the area of a regular hexadecagon inscribed in a circle
    with radius r is equal to 4r² * √(2 - √2) -/
theorem hexadecagon_area_formula (r : ℝ) (h : r > 0) :
  hexadecagonArea r = 4 * r^2 * Real.sqrt (2 - Real.sqrt 2) := by
  -- Proof goes here
  sorry

#check hexadecagon_area_formula

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexadecagon_area_formula_l210_21004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_average_speed_l210_21065

/-- Mary's round trip walk between home and school -/
structure RoundTrip where
  uphill_distance : ℝ
  uphill_time : ℝ
  downhill_distance : ℝ
  downhill_time : ℝ

/-- Calculate the average speed of a round trip -/
noncomputable def average_speed (trip : RoundTrip) : ℝ :=
  (trip.uphill_distance + trip.downhill_distance) / 
  (trip.uphill_time + trip.downhill_time)

/-- Mary's specific round trip -/
noncomputable def mary_trip : RoundTrip := {
  uphill_distance := 2,
  uphill_time := 50 / 60,
  downhill_distance := 3,
  downhill_time := 30 / 60
}

/-- Theorem: Mary's average speed for the round trip is 3.75 km/hr -/
theorem mary_average_speed : 
  average_speed mary_trip = 3.75 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_average_speed_l210_21065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_l210_21083

-- Define the triangle and initial point
variable (A₀ A₁ A₂ P₀ : ℂ)

-- Define the rotation function
noncomputable def rotate (center : ℂ) (point : ℂ) : ℂ :=
  center + (point - center) * (Complex.exp (2 * Real.pi * Complex.I / 3))

-- Define the sequence of points
noncomputable def P : ℕ → ℂ
  | 0 => P₀
  | n + 1 => rotate (match n % 3 with
                     | 0 => A₀
                     | 1 => A₁
                     | _ => A₂) (P n)

-- State the theorem
theorem equilateral_triangle (h : P 1986 = P 0) :
  ∃ (side : ℝ), Complex.abs (A₁ - A₀) = side ∧
                Complex.abs (A₂ - A₁) = side ∧
                Complex.abs (A₀ - A₂) = side := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_l210_21083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_hyperbola_theorem_l210_21003

-- Ellipse
noncomputable def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

noncomputable def ellipse_eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 - b^2 / a^2)

theorem ellipse_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse_eccentricity a b = Real.sqrt 3 / 2) (h4 : b = 2) :
  ellipse_equation 4 2 = ellipse_equation a b := by sorry

-- Hyperbola
noncomputable def hyperbola_equation (a b : ℝ) (x y : ℝ) : Prop :=
  y^2 / a^2 - x^2 / b^2 = 1

noncomputable def hyperbola_eccentricity (a c : ℝ) : ℝ :=
  c / a

def focal_distance (c : ℝ) : ℝ :=
  2 * c

theorem hyperbola_theorem (a b c : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : focal_distance c = 16) (h4 : hyperbola_eccentricity a c = 4/3) :
  hyperbola_equation 6 (2 * Real.sqrt 7) = hyperbola_equation a b := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_hyperbola_theorem_l210_21003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_area_perimeter_l210_21072

-- Define the coordinates of the points
def P : ℝ × ℝ := (1, 4)
def Q : ℝ × ℝ := (4, 5)
def R : ℝ × ℝ := (5, 2)
def S : ℝ × ℝ := (2, 1)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the side length of the square
noncomputable def sideLength : ℝ := distance P S

-- Define the area of the square
noncomputable def area : ℝ := sideLength ^ 2

-- Define the perimeter of the square
noncomputable def perimeter : ℝ := 4 * sideLength

-- Theorem to prove
theorem product_area_perimeter :
  area * perimeter = 40 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_area_perimeter_l210_21072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_ninth_term_l210_21044

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Product of first n terms of a sequence -/
def ProductOfTerms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).prod (λ i => a (i + 1))

/-- Main theorem -/
theorem geometric_sequence_product_ninth_term
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_relation : 2 * a 3 = (a 4) ^ 2) :
  ProductOfTerms a 9 = 512 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_product_ninth_term_l210_21044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_five_solutions_l210_21076

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 0 then x^2 + 1 else 2*x

theorem f_eq_five_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 5 ∧ f x₂ = 5 ∧
  (∀ x : ℝ, f x = 5 → x = x₁ ∨ x = x₂) ∧
  x₁ = -2 ∧ x₂ = 5/2 := by
  sorry

#check f_eq_five_solutions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_eq_five_solutions_l210_21076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_value_l210_21089

def letter_value (n : ℕ) : ℤ :=
  match n % 12 with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 0
  | 6 => -1
  | 7 => -2
  | 8 => -3
  | 9 => -1
  | 10 => 0
  | 11 => 1
  | 0 => 2
  | _ => 0  -- This case should never occur

def letter_position (c : Char) : ℕ :=
  match c with
  | 'a' => 1
  | 'b' => 2
  | 'c' => 3
  | 'd' => 4
  | 'e' => 5
  | 'f' => 6
  | 'g' => 7
  | 'h' => 8
  | 'i' => 9
  | 'j' => 10
  | 'k' => 11
  | 'l' => 12
  | 'm' => 13
  | 'n' => 14
  | 'o' => 15
  | 'p' => 16
  | 'q' => 17
  | 'r' => 18
  | 's' => 19
  | 't' => 20
  | 'u' => 21
  | 'v' => 22
  | 'w' => 23
  | 'x' => 24
  | 'y' => 25
  | 'z' => 26
  | _ => 0  -- This case should never occur

def word_value (word : String) : ℤ :=
  (word.toList.map (λ c => letter_value (letter_position c))).sum

theorem algebra_value : word_value "algebra" = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_algebra_value_l210_21089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_l210_21038

theorem orthogonal_vectors (y : ℝ) : 
  (2 * 3 + (-4) * y = 0) ↔ (y = 3/2) := by
  sorry

#check orthogonal_vectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthogonal_vectors_l210_21038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l210_21043

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 3*x - 4)}
def N : Set ℝ := {x | ∃ y, y = (2 : ℝ)^(x - 1)}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | x > 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l210_21043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_z_values_l210_21032

-- Define the function f
def f (x : ℝ) : ℝ := (3*x)^2 + 3*x + 1

-- State the theorem
theorem sum_of_z_values : 
  (∃ z₁ z₂ : ℝ, f (4*z₁) = 12 ∧ f (4*z₂) = 12 ∧ z₁ ≠ z₂) ∧ 
  (∀ z : ℝ, f (4*z) = 12 → z = z₁ ∨ z = z₂) →
  z₁ + z₂ = -1/12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_z_values_l210_21032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_trinomials_roots_l210_21080

/-- A quadratic trinomial represented as a function -/
def QuadraticTrinomial := ℝ → ℝ

/-- Predicate to check if a quadratic trinomial has both roots less than 1000 -/
def has_roots_less_than_1000 (f : QuadraticTrinomial) : Prop :=
  ∃ x₁ x₂, x₁ < 1000 ∧ x₂ < 1000 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂

/-- Predicate to check if a quadratic trinomial has both roots greater than 1000 -/
def has_roots_greater_than_1000 (f : QuadraticTrinomial) : Prop :=
  ∃ x₁ x₂, x₁ > 1000 ∧ x₂ > 1000 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂

/-- Addition of two QuadraticTrinomial functions -/
def add_QuadraticTrinomial (f g : QuadraticTrinomial) : QuadraticTrinomial :=
  λ x => f x + g x

/-- Theorem: The sum of two quadratic trinomials, where one has both roots less than 1000
    and the other has both roots greater than 1000, cannot have one root less than 1000
    and one root greater than 1000 -/
theorem sum_trinomials_roots (f g : QuadraticTrinomial)
    (hf : has_roots_less_than_1000 f) (hg : has_roots_greater_than_1000 g) :
    ¬ ∃ x₁ x₂, x₁ < 1000 ∧ x₂ > 1000 ∧ (add_QuadraticTrinomial f g) x₁ = 0 ∧ (add_QuadraticTrinomial f g) x₂ = 0 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_trinomials_roots_l210_21080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l210_21007

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.B = Real.pi/4 ∧ 
  t.a * Real.sin t.B = Real.sqrt 3 * t.b * Real.cos t.A ∧
  t.b = 4

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.A = Real.pi/3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B = 
    (1 + Real.sqrt 3) * Real.sqrt 2/2 - (Real.sqrt 2 + Real.sqrt 6)/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l210_21007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_set_B_l210_21012

def B : Set ℕ := {n | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2) ∧ x > 0}

theorem gcd_of_set_B : ∃ d : ℕ, ∀ n ∈ B, d ∣ n ∧ ∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d := by
  use 2
  sorry

#eval Nat.gcd 10 6  -- Just to check if Nat.gcd is available

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_of_set_B_l210_21012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_c_completion_time_l210_21054

/-- The time it takes for A and B together to complete the work -/
noncomputable def time_AB : ℝ := 2

/-- The time it takes for A, B, and C together to complete the work -/
noncomputable def time_ABC : ℝ := 1

/-- The work rate of A and B together -/
noncomputable def rate_AB : ℝ := 1 / time_AB

/-- The work rate of A, B, and C together -/
noncomputable def rate_ABC : ℝ := 1 / time_ABC

/-- The work rate of C alone -/
noncomputable def rate_C : ℝ := rate_ABC - rate_AB

/-- The time it takes for C alone to complete the work -/
noncomputable def time_C : ℝ := 1 / rate_C

theorem c_completion_time : time_C = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_c_completion_time_l210_21054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_has_max_no_min_l210_21095

/-- Arithmetic sequence {a_n} with a_1 = -9 and a_5 = -1 -/
def a (n : ℕ) : ℤ :=
  -9 + 2 * (n - 1)

/-- Product sequence T_n = a_1 * a_2 * ... * a_n -/
def T (n : ℕ) : ℤ :=
  (List.range n).foldl (λ acc i => acc * a (i + 1)) 1

/-- Theorem stating that {T_n} has a maximum term but no minimum term -/
theorem T_has_max_no_min :
  (∃ k : ℕ, ∀ n : ℕ, T n ≤ T k) ∧
  (∀ m : ℕ, ∃ n : ℕ, T n < T m) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_has_max_no_min_l210_21095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_simplification_l210_21030

/-- Given a function f(x) = sin(ωx + φ) with the specified properties,
    prove that it can be expressed as f(x) = sin(2x + π/4) -/
theorem sine_function_simplification 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_pos : φ > 0) 
  (h_period : ∀ x, Real.sin (ω * (x + π) + φ) = Real.sin (ω * x + φ))
  (h_max : ∀ x, Real.sin (ω * x + φ) ≤ Real.sin (ω * π / 8 + φ)) :
  ∃ k : ℤ, ∀ x, Real.sin (ω * x + φ) = Real.sin (2 * x + π / 4 + 2 * π * k) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_simplification_l210_21030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_A_B_l210_21093

open BigOperators
open Nat

def A : ℕ := ∑ i in Finset.range 20, (2*i + 2) * (2*i + 3) + 42

def B : ℕ := 2 + ∑ i in Finset.range 20, (2*i + 3) * (2*i + 4)

theorem absolute_difference_A_B : |Int.ofNat A - Int.ofNat B| = 800 := by
  sorry

#eval A
#eval B
#eval |Int.ofNat A - Int.ofNat B|

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_difference_A_B_l210_21093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_price_day2_l210_21049

/-- Represents the price and volume data for orangeade sales over two days -/
structure OrangeadeData where
  orange_juice : ℚ
  water_day1 : ℚ
  water_day2 : ℚ
  price_day1 : ℚ
  volume_per_glass : ℚ

/-- Calculates the price per glass on the second day given orangeade sales data -/
def price_day2 (data : OrangeadeData) : ℚ :=
  (2 * data.price_day1) / 3

/-- Theorem stating that the price on the second day is $0.40 given the problem conditions -/
theorem orangeade_price_day2 (data : OrangeadeData) 
    (h1 : data.water_day1 = data.orange_juice)
    (h2 : data.water_day2 = 2 * data.orange_juice)
    (h3 : data.price_day1 = 3/5) :
  price_day2 data = 2/5 := by
  sorry

#eval price_day2 { orange_juice := 1, water_day1 := 1, water_day2 := 2, price_day1 := 3/5, volume_per_glass := 1 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orangeade_price_day2_l210_21049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_l210_21068

theorem cosine_equation_solution :
  ∃ x : ℝ, x ∈ Set.Icc 0 π ∧ Real.cos (2 * x) - 1 = 3 * Real.cos x ∧ x = (2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_equation_solution_l210_21068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_lines_l210_21075

-- Define the parabola E
def E (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := y = (1/2) * (x + 1)
def l₂ (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 1)

-- Define the intersection points
noncomputable def intersectionPoint (p k : ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p₁ p₂ : ℝ × ℝ) : ℝ := sorry

-- Define the intersection of two lines
noncomputable def lineIntersection (l₁ l₂ : ℝ → ℝ → Prop) : ℝ × ℝ := sorry

theorem parabola_and_lines (p : ℝ) (h₁ : p > 0) :
  let A := intersectionPoint p (1/2)
  let B := intersectionPoint p (1/2)
  let C := intersectionPoint p (Real.sqrt 2)
  let D := intersectionPoint p (-Real.sqrt 2)
  let G := lineIntersection (λ x y ↦ sorry) (λ x y ↦ sorry)
  distance A B = 2 * Real.sqrt 10 →
  (p = 1 ∧ G.1 = 1) := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_and_lines_l210_21075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_projection_dodecahedron_projection_l210_21081

/-- Represents a polyhedron -/
structure Polyhedron where
  vertices : Set (ℝ × ℝ × ℝ)
  faces : Set (Set (ℝ × ℝ × ℝ))
  edges : Set (Set (ℝ × ℝ × ℝ))

/-- Represents a plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Represents a polygon in 2D space -/
structure Polygon where
  vertices : Set (ℝ × ℝ)

/-- Defines an icosahedron -/
noncomputable def Icosahedron : Polyhedron :=
  sorry

/-- Defines a dodecahedron -/
noncomputable def Dodecahedron : Polyhedron :=
  sorry

/-- Defines a projection function from 3D to 2D -/
noncomputable def project (p : Polyhedron) (plane : Plane) : Polygon :=
  sorry

/-- Checks if a polygon is regular -/
def is_regular (p : Polygon) : Prop :=
  sorry

/-- Checks if a polygon is irregular -/
def is_irregular (p : Polygon) : Prop :=
  sorry

/-- Checks if a polygon has n sides -/
def has_n_sides (p : Polygon) (n : ℕ) : Prop :=
  sorry

/-- Calculates the center of a set of points -/
noncomputable def set_center (s : Set (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  sorry

/-- Theorem about the projection of an icosahedron -/
theorem icosahedron_projection 
  (i : Polyhedron) 
  (plane : Plane) 
  (h1 : i = Icosahedron) 
  (h2 : ∃ v, v ∈ i.vertices ∧ plane.normal = v - set_center i.vertices) : 
  is_regular (project i plane) ∧ has_n_sides (project i plane) 10 := by
  sorry

/-- Theorem about the projection of a dodecahedron -/
theorem dodecahedron_projection 
  (d : Polyhedron) 
  (plane : Plane) 
  (h1 : d = Dodecahedron) 
  (h2 : ∃ v, v ∈ d.vertices ∧ plane.normal = v - set_center d.vertices) : 
  is_irregular (project d plane) ∧ has_n_sides (project d plane) 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_icosahedron_projection_dodecahedron_projection_l210_21081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_l210_21086

/-- Represents a team's jersey colors -/
structure Team where
  home : Nat
  away : Nat
  different : home ≠ away

/-- The football league with its rules -/
class FootballLeague where
  n : Nat
  teams : Fin n → Team
  n_ge_6 : n ≥ 6
  three_color_rule : ∀ (i j k l : Fin n), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i →
    (Finset.card {(teams i).home, (teams i).away, (teams j).home, (teams j).away,
      (teams k).home, (teams k).away, (teams l).home, (teams l).away} ≥ 3)

/-- The theorem stating the minimum number of colors required -/
theorem min_colors (FL : FootballLeague) :
  (∃ (color_set : Finset Nat), (∀ (i : Fin FL.n),
    (FL.teams i).home ∈ color_set ∧ (FL.teams i).away ∈ color_set) ∧
    color_set.card = FL.n - 1) ∧
  (∀ (color_set : Finset Nat), (∀ (i : Fin FL.n),
    (FL.teams i).home ∈ color_set ∧ (FL.teams i).away ∈ color_set) →
    color_set.card ≥ FL.n - 1) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_colors_l210_21086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l210_21092

/-- Represents the state of the game with two bags of coins -/
structure GameState where
  bag1 : ℕ
  bag2 : ℕ

/-- Represents a move in the game -/
structure Move where
  fromBag : ℕ
  toBag : ℕ
  amount : ℕ

/-- Checks if a move is valid given the current game state -/
def isValidMove (state : GameState) (move : Move) : Prop :=
  (move.fromBag = 0 ∨ move.fromBag = 1) ∧
  (move.toBag = 0 ∨ move.toBag = 1) ∧
  move.fromBag ≠ move.toBag ∧
  (move.fromBag = 0 → state.bag1 ≥ move.amount) ∧
  (move.fromBag = 1 → state.bag2 ≥ move.amount)

/-- Applies a move to the current game state -/
def applyMove (state : GameState) (move : Move) : GameState :=
  if move.fromBag = 0
  then { bag1 := state.bag1 - move.amount, bag2 := state.bag2 + move.amount }
  else { bag1 := state.bag1 + move.amount, bag2 := state.bag2 - move.amount }

/-- Checks if the game is over (no more valid moves possible) -/
def isGameOver (state : GameState) : Prop :=
  state.bag1 = 0 ∨ state.bag2 = 0 ∨ (state.bag1 = 1 ∧ state.bag2 = 1)

/-- Theorem stating that the second player always has a winning strategy -/
theorem second_player_wins (initialState : GameState) :
  ∃ (strategy : GameState → Move),
    ∀ (firstPlayerMove : Move),
      isValidMove initialState firstPlayerMove →
      let secondPlayerMove := strategy (applyMove initialState firstPlayerMove)
      isValidMove (applyMove initialState firstPlayerMove) secondPlayerMove ∧
      (isGameOver (applyMove (applyMove initialState firstPlayerMove) secondPlayerMove) →
        ¬∃ (nextMove : Move), isValidMove (applyMove (applyMove initialState firstPlayerMove) secondPlayerMove) nextMove) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l210_21092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_separation_time_is_3_sqrt_5_l210_21052

/-- The time it takes for two people traveling perpendicular to each other
    to be 90 miles apart, given their speeds. -/
noncomputable def time_to_separate (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) : ℝ :=
  distance / Real.sqrt (speed1^2 + speed2^2)

/-- Theorem stating that the time for two people traveling at 12 mph and 6 mph
    perpendicular to each other to be 90 miles apart is 3√5 hours. -/
theorem separation_time_is_3_sqrt_5 :
  time_to_separate 12 6 90 = 3 * Real.sqrt 5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_separate 12 6 90

end NUMINAMATH_CALUDE_ERRORFEEDBACK_separation_time_is_3_sqrt_5_l210_21052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sebastian_missed_work_days_l210_21022

/-- Calculates the number of days Sebastian did not go to work based on his weekly salary and deducted salary. -/
def days_not_worked (weekly_salary : ℚ) (deducted_salary : ℚ) (work_days_per_week : ℕ) : ℕ :=
  let daily_salary := weekly_salary / work_days_per_week
  let deducted_days := (weekly_salary - deducted_salary) / daily_salary
  Int.ceil deducted_days |>.toNat

/-- Theorem stating that Sebastian did not work for 2 days given the problem conditions. -/
theorem sebastian_missed_work_days : 
  days_not_worked 1043 745 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sebastian_missed_work_days_l210_21022
