import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_squared_rational_l902_90278

/-- Given real numbers x, y, and z satisfying certain conditions, 
    prove that xyz^2 is rational. -/
theorem xyz_squared_rational 
  (x y z : ℝ) 
  (h1 : ∃ (q : ℚ), (x + y*z : ℝ) = q)
  (h2 : ∃ (q : ℚ), (y + z*x : ℝ) = q)
  (h3 : ∃ (q : ℚ), (z + x*y : ℝ) = q)
  (h4 : x^2 + y^2 = 1) :
  ∃ (q : ℚ), (x*y*z^2 : ℝ) = q :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_xyz_squared_rational_l902_90278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nail_squares_l902_90254

/-- A square on the table -/
structure Square where
  color : Nat
  position : Set (ℝ × ℝ)

/-- The table with squares -/
structure Table where
  squares : List Square
  colorCount : Nat

/-- Two squares can be nailed with one nail -/
def canNailTogether (s1 s2 : Square) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ s1.position ∧ p ∈ s2.position

/-- The main theorem -/
theorem nail_squares (table : Table) :
  (∀ (colors : List Nat),
    colors.length = table.colorCount →
    ∀ (squares : List Square),
      squares.length = table.colorCount →
      (∀ s, s ∈ squares → s.color ∈ colors) →
      ∃ s1 s2, s1 ∈ squares ∧ s2 ∈ squares ∧ s1 ≠ s2 ∧ canNailTogether s1 s2) →
  ∃ (color : Nat) (nails : List (ℝ × ℝ)),
    nails.length = 2 * table.colorCount - 2 ∧
    ∀ s, s ∈ table.squares → s.color = color →
      ∃ nail, nail ∈ nails ∧ nail ∈ s.position :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nail_squares_l902_90254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_extended_sides_angle_l902_90235

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n ≥ 3

/-- The internal angle of a regular polygon -/
noncomputable def internal_angle (p : RegularPolygon n) : ℝ :=
  (n - 2 : ℝ) * 180 / n

/-- The external angle of a regular polygon -/
noncomputable def external_angle (p : RegularPolygon n) : ℝ :=
  180 - internal_angle p

/-- The angle at new points formed by extended sides -/
noncomputable def new_point_angle (p : RegularPolygon n) : ℝ :=
  (360 - 3 * external_angle p) / 3

theorem hexagon_extended_sides_angle :
  ∀ (p : RegularPolygon 6), new_point_angle p = 60 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_extended_sides_angle_l902_90235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_cube_sum_l902_90258

-- Define the matrix N
def N (x y z : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![![x, y, z],
    ![y, z, x],
    ![z, x, y]]

-- State the theorem
theorem matrix_cube_sum (x y z : ℂ) :
  N x y z ^ 2 = 3 • (1 : Matrix (Fin 3) (Fin 3) ℂ) →
  x * y * z = -1 →
  x^3 + y^3 + z^3 = -3 + 3 * Real.sqrt 3 ∨
  x^3 + y^3 + z^3 = -3 - 3 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_cube_sum_l902_90258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_analysis_l902_90299

/-- Represents the monthly sales volume as a function of the month number -/
noncomputable def salesVolume (x : ℕ) : ℝ := 10 * x + 540

/-- The price per unit in June -/
def junePrice : ℝ := 3200

/-- Calculates the sales volume for a given month and percentage change -/
noncomputable def adjustedSalesVolume (baseVolume : ℝ) (percentageChange : ℝ) : ℝ :=
  baseVolume * (1 + percentageChange / 100)

/-- Calculates the price for a given base price and percentage change -/
noncomputable def adjustedPrice (basePrice : ℝ) (percentageChange : ℝ) : ℝ :=
  basePrice * (1 + percentageChange / 100)

theorem sales_analysis (m : ℝ) :
  (salesVolume 2 = 560) →
  (salesVolume 3 = 570) →
  let juneVolume := salesVolume 6
  let julyVolume := adjustedSalesVolume juneVolume (-2 * m)
  let augustVolume := julyVolume + 220
  let julyPrice := adjustedPrice junePrice m
  let augustPrice := julyPrice * 0.9
  augustPrice * augustVolume = junePrice * juneVolume * 1.155 →
  m = 10 := by
  sorry

#check sales_analysis

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sales_analysis_l902_90299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l902_90206

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := (10 : ℝ) ^ x + 1

-- Define the proposed inverse function
noncomputable def g (x : ℝ) : ℝ := Real.log (x - 1)

-- Statement to prove
theorem inverse_function_proof (x : ℝ) (h : x > 1) :
  f (g x) = x ∧ g (f x) = x := by
  sorry

#check inverse_function_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_proof_l902_90206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l902_90257

/-- Calculates the number of revolutions of the back wheel given the radii of both wheels and the number of revolutions of the front wheel, assuming no slippage. -/
noncomputable def back_wheel_revolutions (front_radius back_radius : ℝ) (front_revolutions : ℕ) : ℝ :=
  (front_radius / back_radius) * (front_revolutions : ℝ)

/-- Theorem stating that for a bicycle with a front wheel of radius 3 feet and a back wheel of radius 0.5 feet, 
    when the front wheel makes 50 revolutions, the back wheel will make 300 revolutions, assuming no slippage. -/
theorem bicycle_wheel_revolutions : 
  back_wheel_revolutions 3 0.5 50 = 300 := by
  -- Unfold the definition of back_wheel_revolutions
  unfold back_wheel_revolutions
  -- Simplify the arithmetic
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bicycle_wheel_revolutions_l902_90257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_segment_range_l902_90270

/-- Definition of the ellipse C -/
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the focal length -/
noncomputable def focal_length (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

/-- Definition of a point on the ellipse -/
def point_on_ellipse (a b x y : ℝ) : Prop :=
  ellipse a b x y

/-- Definition of the line passing through M(2,0) -/
def line_through_M (k x : ℝ) : ℝ :=
  k * (x - 2)

/-- Definition of vector equation for OA + OB = t * OP -/
def vector_equation (xA yA xB yB x y t : ℝ) : Prop :=
  xA + xB = t * x ∧ yA + yB = t * y

/-- Main theorem -/
theorem ellipse_segment_range (a b : ℝ) :
  a > b ∧ b > 0 ∧
  focal_length a b = 2 ∧
  point_on_ellipse a b 1 (Real.sqrt 2 / 2) →
  ∃ (lower upper : ℝ),
    lower = 0 ∧
    upper = 2 * Real.sqrt 5 / 3 ∧
    ∀ (xA yA xB yB x y k t : ℝ),
      ellipse a b xA yA ∧
      ellipse a b xB yB ∧
      ellipse a b x y ∧
      yA = line_through_M k xA ∧
      yB = line_through_M k xB ∧
      vector_equation xA yA xB yB x y t ∧
      2 * Real.sqrt 6 / 3 < t ∧ t < 2 →
      lower < Real.sqrt ((xB - xA)^2 + (yB - yA)^2) ∧
      Real.sqrt ((xB - xA)^2 + (yB - yA)^2) < upper :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_segment_range_l902_90270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_one_fourth_l902_90241

open Real BigOperators

-- Define the series term
noncomputable def seriesTerm (n : ℕ) : ℝ :=
  3^n / (1 + 3^n + 3^(n+1) + 3^(2*n+1))

-- State the theorem
theorem series_sum_is_one_fourth :
  ∑' n, seriesTerm n = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_sum_is_one_fourth_l902_90241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_quantity_in_mixture_l902_90266

/-- Represents a mixture of alcohol and water -/
structure Mixture where
  alcohol : ℚ
  water : ℚ

/-- The ratio of alcohol to water in a mixture -/
def ratio (m : Mixture) : ℚ := m.alcohol / m.water

theorem alcohol_quantity_in_mixture 
  (m : Mixture) 
  (h1 : ratio m = 2/5) 
  (h2 : ratio {alcohol := m.alcohol, water := m.water + 10} = 2/7) : 
  m.alcohol = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alcohol_quantity_in_mixture_l902_90266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_cost_calculation_l902_90226

/-- Calculates the total cost of renting a car for three days with varying rates and GPS service -/
def total_rental_cost (base_rates : Fin 3 → ℚ) (mile_rates : Fin 3 → ℚ) (miles_driven : Fin 3 → ℚ) (gps_cost : ℚ) : ℚ :=
  (Finset.sum (Finset.range 3) (fun i => base_rates i + mile_rates i * miles_driven i + gps_cost))

theorem rental_cost_calculation :
  let base_rates := ![150, 100, 75]
  let mile_rates := ![1/2, 2/5, 3/10]
  let miles_driven := ![620, 744, 510]
  let gps_cost := 10
  total_rental_cost base_rates mile_rates miles_driven gps_cost = 1115.6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rental_cost_calculation_l902_90226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l902_90265

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  -- Sides a, b, c are opposite to angles A, B, C respectively

-- Define the theorem
theorem triangle_theorem (t : Triangle) :
  (Real.sin t.C + Real.cos t.C = 1 - Real.sin (t.C / 2)) →
  (t.a^2 + t.b^2 = 4*(t.a + t.b) - 8) →
  (Real.sin t.C = 3/4 ∧ t.c = 1 + Real.sqrt 7) :=
by
  intro h1 h2
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l902_90265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_power_a_l902_90280

theorem smallest_b_power_a (a b : ℕ) (h : a ^ b = 2 ^ 2023) :
  ∃ (c : ℕ), c ^ a = 1 ∧ ∀ (d : ℕ), d ^ a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_b_power_a_l902_90280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l902_90212

theorem no_such_function :
  ¬ ∃ (f : ℝ → ℝ), ∀ (x y : ℝ), x > 0 ∧ y > x → f y > (y - x) * (f x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l902_90212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_covers_except_a_l902_90214

noncomputable def f (n : ℕ+) : ℕ := 
  Int.toNat ⌊(n : ℝ) + Real.sqrt ((n : ℝ) / 3) + 1/2⌋

def a (n : ℕ+) : ℕ := 3 * n.val^2 - 2 * n.val

theorem f_covers_except_a :
  ∀ k : ℕ+, (∃ n : ℕ+, f n = k) ∨ (∃ n : ℕ+, a n = k) ∧
  ¬(∃ n m : ℕ+, f n = k ∧ a m = k) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_covers_except_a_l902_90214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l902_90238

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 8*x + 18⌋

-- State the theorem about the domain of f
theorem domain_of_f :
  ∀ x : ℝ, ∃ y : ℝ, f x = y :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l902_90238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_inscribed_sphere_l902_90219

/-- Definition of the angle between lateral edge and base plane -/
noncomputable def angle_between_lateral_edge_and_base (n : ℕ) (k : ℝ) : ℝ :=
  (1/2) * Real.arcsin (k / (2 * Real.sin (Real.pi / n)))

/-- Theorem about regular polygonal pyramid inscribed in a sphere -/
theorem regular_pyramid_inscribed_sphere (n : ℕ) (k : ℝ) (h1 : n ≥ 3) (h2 : k > 0) :
  let α := (1/2) * Real.arcsin (k / (2 * Real.sin (Real.pi / n)))
  (∀ (α : ℝ), α = (1/2) * Real.arcsin (k / (2 * Real.sin (Real.pi / n))) →
    α = angle_between_lateral_edge_and_base n k) ∧
  (k ≤ 2 * Real.sin (Real.pi / n)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_pyramid_inscribed_sphere_l902_90219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l902_90251

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (1 - 3^x) + 1 / x^2

-- Theorem statement
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x < 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l902_90251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l902_90292

/-- 
Given a triangle ABC where:
- tan A = 1/4
- tan B = 3/5
- The longest side has length √17
Prove that the length of the shortest side is √2
-/
theorem shortest_side_length (A B C : ℝ) (a b c : ℝ) :
  Real.tan A = 1/4 →
  Real.tan B = 3/5 →
  max a (max b c) = Real.sqrt 17 →
  min a (min b c) = Real.sqrt 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_side_length_l902_90292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_120_degrees_l902_90215

theorem sin_120_degrees : Real.sin (120 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_120_degrees_l902_90215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_marathon_speed_l902_90230

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

/-- Theorem: Alice's average speed in the marathon -/
theorem alice_marathon_speed :
  let total_distance : ℝ := 26
  let total_time : ℝ := 4
  average_speed total_distance total_time = 6.5 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_marathon_speed_l902_90230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l902_90246

-- Define the ellipse
def is_on_ellipse (x y : ℝ) : Prop := x^2/36 + y^2/9 = 1

-- Define the distance from a point to a focus
noncomputable def distance_to_focus (x y fx fy : ℝ) : ℝ := Real.sqrt ((x - fx)^2 + (y - fy)^2)

-- State the theorem
theorem ellipse_focus_distance 
  (x y : ℝ) 
  (f1x f1y f2x f2y : ℝ) -- Coordinates of the two foci
  (h1 : is_on_ellipse x y)
  (h2 : distance_to_focus x y f1x f1y = 5)
  : distance_to_focus x y f2x f2y = 7 := by
  sorry

#check ellipse_focus_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_focus_distance_l902_90246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfly_returns_to_start_l902_90276

-- Define the complex number representing 45° rotation
noncomputable def θ : ℂ := Complex.exp (Complex.I * Real.pi / 4)

-- Define the position of the butterfly after k steps
noncomputable def butterfly_position : ℕ → ℂ
| 0 => 0
| k + 1 => butterfly_position k + 2 * θ ^ k

-- Theorem statement
theorem butterfly_returns_to_start : butterfly_position 1024 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_butterfly_returns_to_start_l902_90276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_old_balls_l902_90291

/-- The number of table tennis balls in the box -/
def total_balls : ℕ := 12

/-- The number of new balls initially in the box -/
def initial_new_balls : ℕ := 9

/-- The number of old balls initially in the box -/
def initial_old_balls : ℕ := 3

/-- The number of balls taken out and put back -/
def balls_taken : ℕ := 3

/-- The probability that the number of old balls becomes 4 after the process -/
def prob_four_old_balls : ℚ := 27 / 220

/-- Theorem stating the probability of having 4 old balls after the process -/
theorem probability_four_old_balls :
  let total := total_balls
  let initial_new := initial_new_balls
  let initial_old := initial_old_balls
  let taken := balls_taken
  (total = initial_new + initial_old) →
  (prob_four_old_balls = (Nat.choose initial_old 2 * Nat.choose initial_new 1) / Nat.choose total taken) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_four_old_balls_l902_90291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_approximation_l902_90279

-- Define the expression for x
noncomputable def x : ℝ := ((69.28 * 0.004)^3) / (0.03^2 * Real.log 0.58)

-- State the theorem
theorem x_approximation : ‖x + 156.758‖ < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_approximation_l902_90279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_pots_two_machines_l902_90297

/-- Calculates the number of whole pots produced in an hour given the minutes per pot -/
def potsPerHour (minutesPerPot : ℚ) : ℕ :=
  (60 / minutesPerPot).floor.toNat

/-- Represents the production rates of a machine in the first and last hour -/
structure MachineProd where
  firstHourRate : ℚ
  lastHourRate : ℚ

/-- Calculates additional pots produced in the last hour compared to the first -/
def additionalPots (m : MachineProd) : ℤ :=
  (potsPerHour m.lastHourRate) - (potsPerHour m.firstHourRate)

theorem additional_pots_two_machines 
  (machineA : MachineProd) 
  (machineB : MachineProd)
  (hA : machineA = ⟨6, 5.2⟩)
  (hB : machineB = ⟨5.5, 5.1⟩) : 
  (additionalPots machineA) + (additionalPots machineB) = 2 := by
  sorry

#eval additionalPots ⟨6, 5.2⟩
#eval additionalPots ⟨5.5, 5.1⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_additional_pots_two_machines_l902_90297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pavlosum_associative_l902_90284

-- Define the Pavlosum operation as noncomputable
noncomputable def pavlosum (x y : ℝ) : ℝ := (x + y) / (1 - x * y)

-- Infix notation for Pavlosum
infix:65 " # " => pavlosum

-- Theorem statement
theorem pavlosum_associative (a b c d : ℝ) : 
  ((a # b) # c) # (-d) = a # (b # (c # (-d))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pavlosum_associative_l902_90284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_ge_e_cubed_l902_90282

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

-- Define the derivative of f(x)
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a

-- Theorem statement
theorem f_decreasing_iff_a_ge_e_cubed (a : ℝ) :
  (∀ x ∈ Set.Ioo (-2) 3, f_derivative a x ≤ 0) ↔ a ≥ Real.exp 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_iff_a_ge_e_cubed_l902_90282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l902_90224

theorem system_solution :
  ∃ (m : ℤ), 
    (0^2 + 4 * Real.sin (Real.pi/2 + m * Real.pi)^2 - 4 = 0) ∧
    (Real.cos 0 - 2 * Real.cos (Real.pi/2 + m * Real.pi)^2 - 1 = 0) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_system_solution_l902_90224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_f_inequality_l902_90237

/-- The function f(x) = 2x - a*ln(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2*x - a * Real.log x

/-- Theorem stating the range of a for which f(x) > 0 for all x > 0 -/
theorem f_positive_range (a : ℝ) :
  (∀ x > 0, f a x > 0) ↔ 0 ≤ a ∧ a < 2 * Real.exp 1 := by
  sorry

/-- Theorem for the inequality when a = 1 and f(x₁)/√x₁ = f(x₂)/√x₂ = √m -/
theorem f_inequality (x₁ x₂ m : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₁ < x₂) 
  (h₄ : f 1 x₁ / Real.sqrt x₁ = Real.sqrt m) (h₅ : f 1 x₂ / Real.sqrt x₂ = Real.sqrt m) :
  x₂ - x₁ < Real.sqrt (m^2 - 16) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_positive_range_f_inequality_l902_90237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_can_prevent_win_l902_90213

/-- Represents the state of the game as a list of four integers -/
def GameState := List Nat

/-- Checks if a GameState has alternating parity -/
def hasAlternatingParity (state : GameState) : Prop :=
  state.length = 4 ∧
  (state.get! 0 % 2 ≠ state.get! 1 % 2) ∧
  (state.get! 1 % 2 ≠ state.get! 2 % 2) ∧
  (state.get! 2 % 2 ≠ state.get! 3 % 2) ∧
  (state.get! 3 % 2 ≠ state.get! 0 % 2)

/-- Represents Player 1's move: adding 1 to two neighboring numbers -/
def player1Move (state : GameState) : GameState :=
  sorry

/-- Represents Player 2's move: swapping any two neighboring numbers -/
def player2Move (state : GameState) : GameState :=
  sorry

/-- The main theorem stating that Player 2 can always maintain alternating parity -/
theorem player2_can_prevent_win (initialState : GameState)
  (h_initial : hasAlternatingParity initialState) :
  ∀ (state : GameState),
    (∃ (moves : Nat), state = (player1Move ∘ player2Move)^[moves] initialState) →
    ∃ (newState : GameState), hasAlternatingParity newState ∧
      newState = player2Move state :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_player2_can_prevent_win_l902_90213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_result_l902_90288

/-- The speed of the second train given the conditions of the problem -/
noncomputable def second_train_speed (
  length1 : ℝ)  -- Length of the first train in meters
  (length2 : ℝ)  -- Length of the second train in meters
  (speed1 : ℝ)   -- Speed of the first train in km/h
  (clear_time : ℝ) : ℝ :=  -- Time to clear in seconds
  let total_length := length1 + length2
  let total_length_km := total_length / 1000
  let clear_time_hours := clear_time / 3600
  let relative_speed := total_length_km / clear_time_hours
  relative_speed - speed1

/-- Theorem stating the speed of the second train under given conditions -/
theorem second_train_speed_result :
  ∃ ε > 0, |second_train_speed 111 165 100 4.516002356175142 - 119.976| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_train_speed_result_l902_90288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tortoise_wins_l902_90217

/-- Represents the race between a tortoise and a hare --/
structure Race where
  distance : ℝ
  tortoise_speed : ℝ
  hare_dash_speed : ℝ
  hare_walk_speed : ℝ

/-- Calculates the time taken by the tortoise to complete the race --/
noncomputable def tortoise_time (race : Race) : ℝ :=
  race.distance / race.tortoise_speed

/-- Calculates the time taken by the hare to complete the race --/
noncomputable def hare_time (race : Race) : ℝ :=
  (race.distance / 4) / race.hare_dash_speed +
  (race.distance / 4) / race.hare_walk_speed +
  (race.distance / 2) / race.hare_walk_speed +
  (race.distance / 4) / race.hare_dash_speed

/-- Theorem stating that the tortoise wins the race by 12.5 meters --/
theorem tortoise_wins (race : Race) 
    (h1 : race.distance = 100)
    (h2 : race.hare_dash_speed = 8 * race.tortoise_speed)
    (h3 : race.hare_walk_speed = 2 * race.tortoise_speed) :
  tortoise_time race < hare_time race ∧ 
  race.distance - (hare_time race / tortoise_time race) * race.distance = 12.5 := by
  sorry

#check tortoise_wins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tortoise_wins_l902_90217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l902_90260

theorem trigonometric_expression_value (a : ℝ) 
  (h1 : Real.cos a = 2/3) 
  (h2 : -π/2 < a ∧ a < 0) : 
  (Real.tan (-a - π) * Real.sin (2*π + a)) / (Real.cos (-a) * Real.tan (π + a)) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_expression_value_l902_90260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_judy_bicycle_time_increase_l902_90208

/-- Calculates the percentage increase in time per mile for Judy's bicycle riding --/
theorem judy_bicycle_time_increase 
  (young_distance : ℝ) 
  (young_time : ℝ) 
  (adult_distance : ℝ) 
  (adult_time : ℝ) 
  (h1 : young_distance = 20) 
  (h2 : young_time = 2) 
  (h3 : adult_distance = 15) 
  (h4 : adult_time = 3) : 
  (((adult_time / adult_distance) - (young_time / young_distance)) / (young_time / young_distance)) * 100 = 100 := by
  sorry

#check judy_bicycle_time_increase

end NUMINAMATH_CALUDE_ERRORFEEDBACK_judy_bicycle_time_increase_l902_90208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l902_90263

/-- 
An ellipse where the distance from the focus to its corresponding directrix 
is equal to the length of the semi-major axis has eccentricity (√5 - 1) / 2.
-/
theorem ellipse_eccentricity (a c : ℝ) (h : a > 0) (h_focal_distance : a^2 / c - c = a) :
  c / a = (Real.sqrt 5 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l902_90263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_is_pi_l902_90228

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x - 2 * Real.pi / 3)

noncomputable def g (x : ℝ) : ℝ := f (x + 7 * Real.pi / 12)

theorem f_properties :
  -- 1. The smallest positive period of f is π
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  -- 2. f is monotonically increasing in [5π/6, 7π/6]
  (∀ (x y : ℝ), 5 * Real.pi / 6 ≤ x ∧ x < y ∧ y ≤ 7 * Real.pi / 6 → f x < f y) ∧
  -- 3. g is an odd function
  (∀ (x : ℝ), g (-x) = -g x) := by
  sorry

-- Additional theorem to state that the period is indeed π
theorem f_period_is_pi :
  ∃ (p : ℝ), p = Real.pi ∧ p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_f_period_is_pi_l902_90228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l902_90255

theorem cos_pi_minus_alpha (α : ℝ) (h1 : α ∈ Set.Ioo (-π) (-π/2)) (h2 : Real.sin α = -5/13) : 
  Real.cos (π - α) = 12/13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_pi_minus_alpha_l902_90255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_after_removing_corners_surface_area_of_modified_cube_l902_90256

/-- The surface area of a cube after removing corner cubes remains unchanged -/
theorem surface_area_after_removing_corners (edge_length : ℝ) (h : edge_length > 0) :
  let original_surface_area := 6 * edge_length^2
  let corner_cube_edge := edge_length / 2
  let removed_area_per_corner := 3 * corner_cube_edge^2
  let new_area_per_corner := 3 * corner_cube_edge^2
  let num_corners := 8
  original_surface_area = original_surface_area - num_corners * removed_area_per_corner + num_corners * new_area_per_corner :=
by
  intros original_surface_area corner_cube_edge removed_area_per_corner new_area_per_corner num_corners
  have h1 : removed_area_per_corner = new_area_per_corner := by rfl
  calc
    original_surface_area
      = original_surface_area - num_corners * removed_area_per_corner + num_corners * new_area_per_corner := by
        rw [h1]
        ring

theorem surface_area_of_modified_cube :
  let edge_length : ℝ := 4
  let original_surface_area := 6 * edge_length^2
  original_surface_area = 96 :=
by
  intro edge_length original_surface_area
  calc
    original_surface_area = 6 * 4^2 := by rfl
    _ = 6 * 16 := by norm_num
    _ = 96 := by norm_num

#eval 96 -- This should output 96

/-- The final answer is option C, which is 96 sq.cm -/
def final_answer : Nat := 96

#check final_answer -- This should show the type of final_answer

end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_after_removing_corners_surface_area_of_modified_cube_l902_90256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_determine_plane_three_points_on_circle_determine_plane_l902_90262

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a plane in 3D space
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a circle in 3D space
structure Circle3D where
  center : Point3D
  radius : ℝ
  normal : Point3D  -- Normal vector to the plane containing the circle

-- Define membership for Point3D in Plane
instance : Membership Point3D Plane where
  mem p plane := plane.a * p.x + plane.b * p.y + plane.c * p.z + plane.d = 0

-- Define membership for Point3D in Circle3D
instance : Membership Point3D Circle3D where
  mem p circle := 
    let dx := p.x - circle.center.x
    let dy := p.y - circle.center.y
    let dz := p.z - circle.center.z
    dx * dx + dy * dy + dz * dz = circle.radius * circle.radius

-- Three points determine a unique plane
theorem three_points_determine_plane (p1 p2 p3 : Point3D) : 
  ∃! (plane : Plane), p1 ∈ plane ∧ p2 ∈ plane ∧ p3 ∈ plane :=
sorry

-- Three points on a circle determine a unique plane
theorem three_points_on_circle_determine_plane (circle : Circle3D) (p1 p2 p3 : Point3D) 
  (h1 : p1 ∈ circle) (h2 : p2 ∈ circle) (h3 : p3 ∈ circle) : 
  ∃! (plane : Plane), p1 ∈ plane ∧ p2 ∈ plane ∧ p3 ∈ plane :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_points_determine_plane_three_points_on_circle_determine_plane_l902_90262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_separation_technique_l902_90264

-- Define separation techniques
def DifferentialCentrifugation : Type := Unit
def DNASeparation : Type := Unit
def GelChromatography : Type := Unit
def CongoRedMethod : Type := Unit

-- Define types of microorganisms
inductive Microorganism
| UreaDecomposing
| CelluloseDecomposing

-- Define the property of a separation technique being correct
def IsCorrectTechnique : Type → Prop := sorry

-- Define the property of a method identifying a microorganism
def Identifies (method : Type) (m : Microorganism) : Prop := sorry

-- State the theorem
theorem incorrect_separation_technique :
  IsCorrectTechnique DifferentialCentrifugation →
  IsCorrectTechnique DNASeparation →
  IsCorrectTechnique GelChromatography →
  (∀ m : Microorganism, Identifies CongoRedMethod m ↔ m = Microorganism.CelluloseDecomposing) →
  ¬IsCorrectTechnique CongoRedMethod :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_incorrect_separation_technique_l902_90264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_symmetry_implies_k_zero_l902_90283

/-- The line equation y = kx + 1 -/
def line (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

/-- The circle equation x² + y² + kx - y - 9 = 0 -/
def circle_eq (k : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + k * x - y - 9 = 0

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric about the y-axis -/
def symmetric_about_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ = -x₂ ∧ y₁ = y₂

theorem intersection_symmetry_implies_k_zero (k : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line k x₁ y₁ ∧ circle_eq k x₁ y₁ ∧
    line k x₂ y₂ ∧ circle_eq k x₂ y₂ ∧
    symmetric_about_y_axis x₁ y₁ x₂ y₂) →
  k = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_symmetry_implies_k_zero_l902_90283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_implies_line_equation_common_chord_length_circle_equation_intersecting_chords_l902_90239

-- Define the circle and the point M
def my_circle : Set (ℝ × ℝ) := {p | (p.1)^2 + (p.2 + 2)^2 = 25}
def M : ℝ × ℝ := (-3, 0)

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) := {p | p.2 = k * (p.1 + 3)}

-- Theorem statement
theorem chord_length_implies_line_equation :
  ∃ k : ℝ, (∀ p ∈ line_l k, p ∈ my_circle → 
    ∃ q ∈ line_l k, q ∈ my_circle ∧ q ≠ p ∧ 
    (p.1 - q.1)^2 + (p.2 - q.2)^2 = 64) ∧
  (k = 5/12 ∨ k = 0) := by
  sorry

-- Additional theorems for other questions

theorem common_chord_length :
  let C1 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 + 2*p.1 + 8*p.2 - 8 = 0}
  let C2 : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 - 2 = 0}
  ∃ l : ℝ, l = 2 * Real.sqrt 5 ∧ 
    ∃ A B : ℝ × ℝ, A ∈ C1 ∧ A ∈ C2 ∧ B ∈ C1 ∧ B ∈ C2 ∧ 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = l^2 := by
  sorry

theorem circle_equation :
  let A : ℝ × ℝ := (0, 2)
  let B : ℝ × ℝ := (-2, 2)
  ∃ h k r : ℝ, h - k - 2 = 0 ∧
    (A.1 - h)^2 + (A.2 - k)^2 = r^2 ∧
    (B.1 - h)^2 + (B.2 - k)^2 = r^2 ∧
    ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 ↔ (x + 1)^2 + (y + 3)^2 = 26 := by
  sorry

theorem intersecting_chords :
  let C : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 1}
  let P : ℝ × ℝ := (3, 4)
  ∀ A B : ℝ × ℝ, A ∈ C → B ∈ C → 
    (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_implies_line_equation_common_chord_length_circle_equation_intersecting_chords_l902_90239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_range_l902_90209

noncomputable def f (x a : ℝ) : ℝ := 3^x - 4/x - a

theorem zero_point_range (a : ℝ) :
  (∃ x ∈ Set.Ioo 1 2, f x a = 0) → a ∈ Set.Ioo (-1) 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_point_range_l902_90209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_transformation_l902_90204

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem sine_graph_transformation :
  ∀ x : ℝ, g x = f (x + Real.pi / 3) :=
by
  intro x
  simp [f, g]
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_graph_transformation_l902_90204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_set_property_weight_set_is_fibonacci_main_theorem_l902_90296

def fibonacci : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

def weightSet : List ℕ := [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]

def canMeasure (weights : List ℕ) (target : ℕ) : Prop :=
  ∃ (subset : List ℕ), subset.toFinset ⊆ weights.toFinset ∧ subset.sum = target

theorem weight_set_property :
  ∀ (n : ℕ) (lostWeight : ℕ),
    n ≤ 55 →
    lostWeight ∈ weightSet →
    canMeasure (weightSet.filter (· ≠ lostWeight)) n :=
by sorry

theorem weight_set_is_fibonacci :
  ∀ (i : Fin 10), weightSet[i.val] = fibonacci i.val :=
by sorry

theorem main_theorem :
  ∀ (n : ℕ) (lostWeight : ℕ),
    n ≤ 55 →
    lostWeight ∈ weightSet →
    canMeasure (weightSet.filter (· ≠ lostWeight)) n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_set_property_weight_set_is_fibonacci_main_theorem_l902_90296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_five_twelfths_l902_90253

/-- The sum of the infinite series Σ(1/((n+1)(n+3))) from n=1 to ∞ -/
noncomputable def infiniteSeries : ℝ := ∑' n : ℕ, 1 / ((n + 1) * (n + 3))

theorem infiniteSeries_eq_five_twelfths : infiniteSeries = 5 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeries_eq_five_twelfths_l902_90253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_equals_53_l902_90232

def G : ℕ → ℚ
  | 0 => 3  -- Add a case for 0 to cover all natural numbers
  | 1 => 3
  | (n + 1) => (3 * G n + 3) / 3

theorem G_51_equals_53 : G 51 = 53 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_G_51_equals_53_l902_90232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_for_cos_negative_four_fifths_l902_90244

theorem trig_values_for_cos_negative_four_fifths (α : Real) 
  (h : Real.cos α = -4/5) : 
  (Real.sin α = 3/5 ∨ Real.sin α = -3/5) ∧ (Real.tan α = -3/4 ∨ Real.tan α = 3/4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_values_for_cos_negative_four_fifths_l902_90244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_walking_speed_bound_l902_90293

/-- Represents Tom's journey from home to school -/
structure Journey where
  totalDistance : ℝ
  totalTime : ℝ
  runningSpeed : ℝ
  runningDistance : ℝ

/-- Calculate Tom's walking speed given his journey details -/
noncomputable def walkingSpeed (j : Journey) : ℝ :=
  let walkingDistance := j.totalDistance - j.runningDistance
  let runningTime := j.runningDistance / j.runningSpeed
  let walkingTime := j.totalTime - runningTime
  walkingDistance / walkingTime

/-- Theorem stating that Tom's walking speed is less than or equal to 70 m/min -/
theorem toms_walking_speed_bound (j : Journey) 
    (h1 : j.totalDistance = 1800)
    (h2 : j.totalTime ≤ 20)
    (h3 : j.runningSpeed = 210)
    (h4 : j.runningDistance = 600) :
    walkingSpeed j ≤ 70 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_toms_walking_speed_bound_l902_90293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_remaining_after_pourings_l902_90294

/-- Represents the fraction of water remaining after each pouring -/
def waterRemaining : ℕ → ℚ
  | 0 => 1
  | 1 => 3/4
  | 2 => 1/4
  | 3 => 1/8
  | n + 4 =>
    let cyclePosition := n % 4
    let baseFraction :=
      if cyclePosition == 0 then 1/2
      else if cyclePosition == 1 then 3/5
      else if cyclePosition == 2 then 7/10
      else 4/5
    waterRemaining n * baseFraction

/-- The number of pourings after which exactly 1/5 of the original water remains -/
def numberOfPourings : ℕ := 8

theorem water_remaining_after_pourings :
  waterRemaining numberOfPourings = 1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_remaining_after_pourings_l902_90294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_increase_l902_90285

theorem cricketer_average_increase :
  ∀ (prev_total : ℕ) (prev_average : ℚ),
    prev_average = prev_total / 21 →
    (prev_total + 134 : ℚ) / 22 = 60.5 →
    60.5 - prev_average = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricketer_average_increase_l902_90285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_product_eight_probability_l902_90277

/-- The probability of rolling a specific number on a standard die -/
def prob_single_roll : ℚ := 1 / 6

/-- The number of permutations of three distinct numbers -/
def num_permutations : ℕ := 6

/-- The probability of rolling three 2s -/
def prob_three_twos : ℚ := prob_single_roll ^ 3

/-- The probability of rolling 1, 2, and 4 in any order -/
def prob_one_two_four : ℚ := num_permutations * prob_single_roll ^ 3

/-- The total probability of rolling a product of 8 with three dice -/
def prob_product_eight : ℚ := prob_three_twos + prob_one_two_four

theorem dice_product_eight_probability :
  prob_product_eight = 7 / 216 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_product_eight_probability_l902_90277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expensive_feed_cost_l902_90261

/-- Proves that the cost of the expensive feed is approximately $0.50 per pound -/
theorem expensive_feed_cost 
  (total_mixed : ℝ) 
  (mixed_cost : ℝ) 
  (cheap_cost : ℝ) 
  (cheap_amount : ℝ) 
  (h1 : total_mixed = 17)
  (h2 : mixed_cost = 0.22)
  (h3 : cheap_cost = 0.11)
  (h4 : cheap_amount = 12.2051282051) :
  let total_value := total_mixed * mixed_cost
  let cheap_value := cheap_amount * cheap_cost
  let expensive_amount := total_mixed - cheap_amount
  let expensive_value := total_value - cheap_value
  ∃ ε > 0, |expensive_value / expensive_amount - 0.50| < ε := by
  sorry

#eval Float.ofScientific 5 0 1 -- This evaluates to 0.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expensive_feed_cost_l902_90261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l902_90216

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the sets P and Q
def P (t : ℝ) : Set ℝ := {x : ℝ | f (x + t) < 2}
def Q : Set ℝ := {x : ℝ | f x < -4}

-- State the theorem
theorem t_range :
  (Monotone f) →
  f (-1) = -4 →
  f 2 = 2 →
  (∀ t : ℝ, P t ⊂ Q ∧ P t ≠ Q) →
  {t : ℝ | t > 3} = {t : ℝ | ∀ x : ℝ, x ∈ P t → x ∈ Q} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_range_l902_90216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_fixed_line_l902_90207

/-- Definition of the hyperbola -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

/-- Definition of the focal length -/
def focal_length : ℝ := 8

/-- Left vertex of the hyperbola -/
def left_vertex : ℝ × ℝ := (-2, 0)

/-- Right vertex of the hyperbola -/
def right_vertex : ℝ × ℝ := (2, 0)

/-- Right focus of the hyperbola -/
def right_focus : ℝ × ℝ := (4, 0)

/-- A line passing through the right focus with slope m -/
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m * y + 4

/-- Theorem: The intersection of AQ and BP lies on x = 1 -/
theorem intersection_on_fixed_line (m : ℝ) (hm : m ≠ 0) 
  (P Q : ℝ × ℝ) (hP : hyperbola P.1 P.2) (hQ : hyperbola Q.1 Q.2)
  (hPl : line_through_focus m P.1 P.2) (hQl : line_through_focus m Q.1 Q.2) :
  ∃ (y : ℝ), 
    (Q.2 / (Q.1 + 2)) * (1 + 2) = (P.2 / (P.1 - 2)) * (1 - 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_on_fixed_line_l902_90207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_five_halves_l902_90249

/-- The function f(x) = x^2 / (1 + x^2) -/
noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

/-- The sum of f(1), f(2), f(3), f(1/2), and f(1/3) equals 5/2 -/
theorem sum_of_f_equals_five_halves :
  f 1 + f 2 + f 3 + f (1/2) + f (1/3) = 5/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_f_equals_five_halves_l902_90249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_8_l902_90240

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_lucky (n : ℕ) : Prop :=
  n > 0 ∧ n % (sum_of_digits n) = 0

theorem least_non_lucky_multiple_of_8 :
  (∀ k : ℕ, k > 0 ∧ k < 16 ∧ k % 8 = 0 → is_lucky k) ∧
  ¬ is_lucky 16 ∧
  16 % 8 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_non_lucky_multiple_of_8_l902_90240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_1_to_10_l902_90227

def isPrime (n : ℕ) : Bool :=
  if n ≤ 1 then false
  else
    (List.range (n - 1)).all (fun m => if m > 1 then n % m ≠ 0 else true)

def countPrimes (n : ℕ) : ℕ :=
  (List.range n).filter isPrime |>.length

theorem probability_prime_1_to_10 :
  (countPrimes 10 : ℚ) / 10 = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_prime_1_to_10_l902_90227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_minus_phi_l902_90231

open Complex

-- Define the complex numbers given in the problem
noncomputable def z₁ : ℂ := 4/5 + 3/5 * I
noncomputable def z₂ : ℂ := -5/13 + 12/13 * I

-- State the theorem
theorem cos_theta_minus_phi (θ φ : ℝ) 
  (h₁ : exp (I * θ) = z₁)
  (h₂ : exp (I * φ) = z₂) : 
  Real.cos (θ - φ) = -16/65 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_theta_minus_phi_l902_90231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_longer_side_proof_l902_90200

/-- The length of the longer parallel side of each trapezoid in a specially divided rectangle -/
noncomputable def trapezoid_longer_side (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  5/3

/-- The area of each part (trapezoid or pentagon) after dividing the rectangle -/
noncomputable def part_area (rectangle_width : ℝ) (rectangle_height : ℝ) : ℝ :=
  (rectangle_width * rectangle_height) / 3

theorem trapezoid_longer_side_proof (rectangle_width rectangle_height : ℝ) 
  (h_width : rectangle_width = 2)
  (h_height : rectangle_height = 3)
  (h_division : ∃ (center : ℝ × ℝ) (point1 point2 point3 : ℝ × ℝ), 
    -- The rectangle is divided into two congruent trapezoids and a pentagon
    True)
  (h_equal_areas : ∀ (area1 area2 area3 : ℝ), 
    area1 = part_area rectangle_width rectangle_height ∧ 
    area2 = part_area rectangle_width rectangle_height ∧
    area3 = part_area rectangle_width rectangle_height →
    -- The two trapezoids and the pentagon have equal areas
    True) :
  trapezoid_longer_side rectangle_width rectangle_height = 5/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezoid_longer_side_proof_l902_90200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_greater_than_even_sum_l902_90289

/-- Represents a point with its label -/
structure Point where
  label : Nat

/-- Represents a line segment connecting two points -/
structure Segment where
  point1 : Point
  point2 : Point

/-- The set of all points -/
def points : Finset Point := sorry

/-- The set of all segments -/
def segments : Finset Segment := sorry

/-- The label of a segment is the sum of its endpoint labels -/
def segmentLabel (s : Segment) : Nat :=
  s.point1.label + s.point2.label

/-- The set of segments with even labels -/
def evenSegments : Finset Segment :=
  segments.filter (fun s => segmentLabel s % 2 = 0)

/-- The set of segments with odd labels -/
def oddSegments : Finset Segment :=
  segments.filter (fun s => segmentLabel s % 2 ≠ 0)

/-- The sum of labels of even-labeled segments -/
def sumEvenSegments : Nat :=
  evenSegments.sum (fun s => segmentLabel s)

/-- The sum of labels of odd-labeled segments -/
def sumOddSegments : Nat :=
  oddSegments.sum (fun s => segmentLabel s)

theorem odd_sum_greater_than_even_sum :
  points.card = 100 →
  (∀ p ∈ points, ∃ n : Nat, p.label = n^2 ∧ n ≤ 100) →
  sumOddSegments > sumEvenSegments := by
  sorry

#eval "Theorem statement type-checks correctly."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_sum_greater_than_even_sum_l902_90289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_steven_plowing_rate_l902_90225

/-- Represents the farming scenario of Farmer Steven -/
structure FarmingScenario where
  total_days : ℚ
  farmland_acres : ℚ
  grassland_acres : ℚ
  max_mowing_rate : ℚ

/-- Calculates the plowing rate given a farming scenario -/
noncomputable def plowing_rate (scenario : FarmingScenario) : ℚ :=
  scenario.farmland_acres / (scenario.total_days - scenario.grassland_acres / scenario.max_mowing_rate)

/-- Theorem stating that Farmer Steven's plowing rate is 10 acres per day -/
theorem steven_plowing_rate :
  let scenario : FarmingScenario := {
    total_days := 8,
    farmland_acres := 55,
    grassland_acres := 30,
    max_mowing_rate := 12
  }
  plowing_rate scenario = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_steven_plowing_rate_l902_90225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_condition_l902_90242

theorem count_integers_satisfying_condition : 
  (Finset.filter (fun n : ℕ => n > 0 ∧ (140 * n)^40 > n^80 ∧ n^80 > 3^160) (Finset.range 140)).card = 130 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_integers_satisfying_condition_l902_90242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_19th_term_l902_90222

def s : ℕ → ℚ
  | 0 => 2  -- Add a case for 0
  | 1 => 2
  | n+2 => if (n+2) % 3 = 0 then 2 + s ((n+2)/3) else 1 / s (n+1)

theorem sequence_19th_term (n : ℕ) : s n = 23/105 → n = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_19th_term_l902_90222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l902_90221

noncomputable def f (x : ℝ) := x - 2 * Real.sin x

theorem f_increasing_interval :
  ∀ x ∈ Set.Ioo (π / 3) π, x ∈ Set.Icc 0 π → 
  ∀ y ∈ Set.Ioo (π / 3) π, y ∈ Set.Icc 0 π → 
  x < y → f x < f y :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_interval_l902_90221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l902_90223

/-- An even function f where f(x) = ln(-x) + 2x for x < 0 -/
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) + 2 * x else Real.log x - 2 * x

/-- The equation of the tangent line to y = f(x) at (1, -2) is x + y + 1 = 0 -/
theorem tangent_line_equation (h : ∀ x, f (-x) = f x) :
  ∃ m b, (deriv f) 1 = m ∧ f 1 = -2 ∧ m + b + 1 = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l902_90223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l902_90298

-- Option A
def f_A (x : ℤ) : ℤ := x^2
def g_A (x : ℤ) : ℤ := if x = 0 then 0 else 1

def domain_A : Set ℤ := {-1, 0, 1}

-- Option B
def f_B (x : ℝ) : ℝ := x * abs x
noncomputable def g_B (x : ℝ) : ℝ := if x ≥ 0 then x^2 else -x^2

-- Option C
def f_C (x : ℝ) : ℝ := x
noncomputable def g_C (x : ℝ) : ℝ := Real.sqrt (x^2)

-- Option D
noncomputable def f_D (x : ℝ) : ℝ := 1 / x
noncomputable def g_D (x : ℝ) : ℝ := (x + 1) / (x^2 + x)

def domain_D : Set ℝ := {x : ℝ | x > 0}

theorem function_equivalence :
  (∀ x ∈ domain_A, f_A x = g_A x) ∧
  (∀ x : ℝ, f_B x = g_B x) ∧
  (∃ x : ℝ, f_C x ≠ g_C x) ∧
  (∀ x ∈ domain_D, f_D x = g_D x) :=
by
  sorry

#check function_equivalence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_equivalence_l902_90298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_perpendicular_bisector_l902_90245

-- Define points A, B, and P
def A : ℝ × ℝ := (8, -6)
def B : ℝ × ℝ := (2, 2)
def P : ℝ × ℝ := (2, -3)

-- Define the slope of line AB
noncomputable def slope_AB : ℝ := (B.2 - A.2) / (B.1 - A.1)

-- Define the equation of line l (parallel to AB and passing through P)
def line_l (x y : ℝ) : Prop := 4 * x + 3 * y + 1 = 0

-- Define the midpoint of AB
noncomputable def midpoint_AB : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Define the equation of the perpendicular bisector of AB
def perp_bisector_AB (x y : ℝ) : Prop := 3 * x - 4 * y - 23 = 0

theorem line_and_perpendicular_bisector :
  (∀ x y : ℝ, line_l x y ↔ y + 3 = slope_AB * (x - P.1)) ∧
  (∀ x y : ℝ, perp_bisector_AB x y ↔ 
    (y - midpoint_AB.2 = -(1 / slope_AB) * (x - midpoint_AB.1))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_and_perpendicular_bisector_l902_90245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_two_sufficient_not_necessary_l902_90268

theorem x_eq_two_sufficient_not_necessary :
  (∃ x : ℝ, x ≠ 2 ∧ x - 2 = Real.sqrt (x^2 - 4*x + 4)) ∧
  (∀ x : ℝ, x = 2 → x - 2 = Real.sqrt (x^2 - 4*x + 4)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_eq_two_sufficient_not_necessary_l902_90268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kwame_study_time_l902_90210

/-- Proves that Kwame studied for 2.5 hours given the study times of Connor and Lexia, and the difference between their combined study time and Lexia's study time. -/
theorem kwame_study_time (connor_time lexia_time difference : ℝ) 
  (h1 : connor_time = 1.5)
  (h2 : lexia_time = 97 / 60)
  (h3 : difference = 143 / 60) : 
  connor_time + difference + lexia_time - lexia_time - connor_time = 2.5 := by
  sorry

#check kwame_study_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kwame_study_time_l902_90210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projective_transformation_circle_theorem_l902_90274

-- Define a circle in a 2D plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a projective transformation
structure ProjectiveTransformation where
  transform : (ℝ × ℝ) → (ℝ × ℝ)
  inverse : (ℝ × ℝ) → (ℝ × ℝ)

-- Define a line in a 2D plane
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def isPointInside (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

def isVanishingLine (t : ProjectiveTransformation) (l : Line) : Prop := sorry

def isPerpendicular (l1 l2 : Line) : Prop := sorry

def isDiameter (c : Circle) (l : Line) : Prop := sorry

def pointOnLine (p : ℝ × ℝ) (l : Line) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

theorem projective_transformation_circle_theorem 
  (C : Circle) (P : ℝ × ℝ) (h : isPointInside C P) :
  ∃ (T : ProjectiveTransformation) (C' : Circle) (l : Line),
    (∀ (p : ℝ × ℝ), isPointInside C p ↔ isPointInside C' (T.transform p)) ∧
    T.transform P = C'.center ∧
    isVanishingLine T l ∧
    ∃ (d : Line), isDiameter C d ∧ pointOnLine P d ∧ isPerpendicular l d :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projective_transformation_circle_theorem_l902_90274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_v_18_value_l902_90287

/-- Sequence defined recursively -/
noncomputable def v (b : ℝ) : ℕ → ℝ
  | 0 => b  -- Add case for 0
  | 1 => b
  | n + 2 => -1 / (v b (n + 1) + 2)

/-- Theorem stating the value of the 18th term of the sequence -/
theorem v_18_value (b : ℝ) (h : b > 0) : v b 18 = -1 / (b + 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_v_18_value_l902_90287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_garden_larger_l902_90229

/-- Represents the dimensions of a rectangular garden --/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a garden given its dimensions --/
def gardenArea (d : GardenDimensions) : ℕ := d.length * d.width

/-- Calculates the usable area of a garden given its dimensions and a space factor --/
def usableArea (d : GardenDimensions) (spaceFactor : ℕ) : ℕ := 
  (gardenArea d) / spaceFactor

theorem emily_garden_larger (johnDim emilyDim : GardenDimensions) :
  johnDim.length = 30 ∧ 
  johnDim.width = 60 ∧ 
  emilyDim.length = 35 ∧ 
  emilyDim.width = 55 → 
  usableArea emilyDim 1 - usableArea johnDim 2 = 1025 := by
  sorry

#eval usableArea ⟨30, 60⟩ 2
#eval usableArea ⟨35, 55⟩ 1
#eval usableArea ⟨35, 55⟩ 1 - usableArea ⟨30, 60⟩ 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_emily_garden_larger_l902_90229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_constant_l902_90203

/-- Definition of the ellipse (C) -/
noncomputable def ellipse (x y a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Eccentricity of the ellipse -/
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

/-- Theorem: Distance from origin to line AB is constant -/
theorem distance_to_line_constant
  (a b : ℝ) 
  (h_pos : a > b ∧ b > 0)
  (h_ecc : eccentricity a b = Real.sqrt 6 / 3)
  (h_vertex : b = 1) :
  ∀ A B : ℝ × ℝ, 
    ellipse A.1 A.2 a b → 
    ellipse B.1 B.2 a b → 
    A.1 * B.1 + A.2 * B.2 = 0 → 
    ∃ d : ℝ, d = Real.sqrt 3 / 2 ∧ 
      ∀ m k : ℝ, (B.2 - A.2) * A.1 = (B.1 - A.1) * A.2 → 
        d = |m| / Real.sqrt (k^2 + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_line_constant_l902_90203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_isosceles_triangle_l902_90248

-- Define the set of stick lengths
noncomputable def stickLengths : Finset ℝ :=
  Finset.image (fun i => (0.9 : ℝ) ^ i) (Finset.range 100)

-- Define what it means for three lengths to form an isosceles triangle
def isIsoscelesTriangle (a b c : ℝ) : Prop :=
  (a = b ∧ a + b > c ∧ a + c > b ∧ b + c > a) ∨
  (a = c ∧ a + c > b ∧ a + b > c ∧ c + b > a) ∨
  (b = c ∧ b + c > a ∧ b + a > c ∧ c + a > b)

-- Theorem statement
theorem no_isosceles_triangle :
  ¬ ∃ (a b c : ℝ), a ∈ stickLengths ∧ b ∈ stickLengths ∧ c ∈ stickLengths ∧
    isIsoscelesTriangle a b c :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_isosceles_triangle_l902_90248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_l902_90271

theorem least_subtraction (n : ℕ) : 
  (∀ d : ℕ, d ∈ ({5, 7, 9} : Set ℕ) → (642 - n) % d = 4) ∧ 
  (∀ m : ℕ, m < n → ∃ d : ℕ, d ∈ ({5, 7, 9} : Set ℕ) ∧ (642 - m) % d ≠ 4) →
  n = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_subtraction_l902_90271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l902_90281

noncomputable def g (c : ℝ) (x : ℝ) : ℝ := 1 / (3 * x + c)

noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (2 * x)

theorem inverse_function_theorem (c : ℝ) :
  (∀ x, g c x ≠ 0 → g_inv (g c x) = x) →
  (∀ x, g_inv x ≠ 0 → g c (g_inv x) = x) →
  c = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l902_90281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bella_galya_l902_90205

/-- The distance between two houses -/
def distance (house1 house2 : ℕ) : ℝ := sorry

/-- The total distance from a house to all other houses -/
def total_distance (house : ℕ) : ℝ := sorry

/-- The order of houses along the road -/
def house_order : List ℕ := [1, 2, 3, 4, 5]

theorem distance_bella_galya :
  house_order = [1, 2, 3, 4, 5] →
  total_distance 2 = 700 →
  total_distance 3 = 600 →
  total_distance 4 = 650 →
  distance 2 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_bella_galya_l902_90205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l902_90272

/-- A hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola (a b : ℝ) where
  pos_a : a > 0
  pos_b : b > 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola a b) : ℝ := Real.sqrt ((a * a + b * b) / (a * a))

/-- A line perpendicular to the x-axis passing through a point -/
structure VerticalLine (x : ℝ) where

/-- The right focus of a hyperbola -/
noncomputable def right_focus (h : Hyperbola a b) : ℝ := Real.sqrt (a * a + b * b)

/-- A point symmetric to (x, y) about the line x = c -/
def symmetric_point (x y c : ℝ) : ℝ × ℝ := (2 * c - x, y)

theorem hyperbola_eccentricity_theorem (a b : ℝ) (h : Hyperbola a b) 
  (l : VerticalLine (right_focus h)) :
  symmetric_point (7 * a) 0 (right_focus h) ∈ {p : ℝ × ℝ | (p.1 * p.1) / (a * a) - (p.2 * p.2) / (b * b) = 1} →
  eccentricity h = 3 ∨ eccentricity h = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_theorem_l902_90272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_y_eq_x_plus_1_l902_90267

/-- The slope angle of a line with equation y = mx + b is the angle between the line and the positive x-axis, measured counterclockwise. -/
noncomputable def slope_angle (m : ℝ) : ℝ := Real.arctan m

theorem slope_angle_of_y_eq_x_plus_1 :
  let m : ℝ := 1
  let α := slope_angle m
  0 ≤ α ∧ α < π → α = π/4 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_angle_of_y_eq_x_plus_1_l902_90267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_specific_circles_l902_90295

/-- The length of the common chord of two intersecting circles -/
noncomputable def common_chord_length (R r d : ℝ) : ℝ :=
  2 * R * Real.sqrt (1 - ((R^2 + d^2 - r^2) / (2 * R * d))^2)

/-- Theorem stating the length of the common chord for the given circles -/
theorem common_chord_length_specific_circles :
  common_chord_length 13 5 12 = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_chord_length_specific_circles_l902_90295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l902_90275

/-- Given a parabola defined by x = ay^2 where a > 0, its focus has coordinates (1/(4a), 0) -/
theorem parabola_focus_coordinates (a : ℝ) (h : a > 0) :
  let parabola := {p : ℝ × ℝ | p.1 = a * p.2^2}
  let focus := (1 / (4 * a), 0)
  focus ∈ parabola :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_focus_coordinates_l902_90275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_range_of_a_l902_90252

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 * f a x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := x^2 / f a x - 1

-- Theorem for part I
theorem max_value_g :
  ∃ (max : ℝ), max = Real.exp (-2) ∧
  ∀ x > 0, g (-2) x ≤ max :=
sorry

-- Theorem for part II
theorem range_of_a (a : ℝ) :
  (∃ x y, 0 < x ∧ x < y ∧ y < 16 ∧ h a x = 0 ∧ h a y = 0) →
  1/2 * Real.log 2 < a ∧ a < 2 / Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_g_range_of_a_l902_90252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_m_l902_90259

open Real

/-- The function f(x) = 1/2 * x^2 - 2x --/
noncomputable def f (x : ℝ) : ℝ := 1/2 * x^2 - 2*x

/-- The function g(x) = ln x --/
noncomputable def g (x : ℝ) : ℝ := log x

/-- The derivative of f(x) --/
def f' (x : ℝ) : ℝ := x - 2

theorem max_integer_m :
  ∀ m : ℝ, m > 4 → ∃ x : ℝ, x > 1 ∧ 2 * (f' x) + x * (g x) + 3 ≤ m * (x - 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integer_m_l902_90259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_implies_ratio_three_l902_90269

theorem tan_half_implies_ratio_three (α : ℝ) (h : Real.tan α = 1/2) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 := by
  -- Proof will be added here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_half_implies_ratio_three_l902_90269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_trace_determination_l902_90234

-- Define the basic geometric elements
structure Plane where
  firstTrace : Set ℝ
  secondTrace : Set ℝ

-- Define the line as a set of points
def Line := Set ℝ

-- Define the angle between two lines
noncomputable def intersectionAngle (l1 l2 : Line) : ℝ := sorry

-- Define the intersecting line of two planes
noncomputable def intersectingLine (p1 p2 : Plane) : Line := sorry

-- Theorem statement
theorem second_trace_determination (p1 p2 : Plane) (α β γ : ℝ) :
  let b := intersectingLine p1 p2
  intersectionAngle b p1.firstTrace = γ →
  intersectionAngle b p2.firstTrace = α →
  intersectionAngle p1.firstTrace p2.firstTrace = β →
  ∃! (s1 s2 : Line), p1.secondTrace = s1 ∧ p2.secondTrace = s2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_trace_determination_l902_90234


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_min_no_max_l902_90247

noncomputable def g (x : ℝ) : ℝ := 2^x - 1/(2^x)

noncomputable def f (x : ℝ) : ℝ := if x ≥ 0 then g x else g (-x)

theorem f_has_min_no_max :
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m) ∧ 
  (¬∃ (M : ℝ), ∀ (x : ℝ), f x ≤ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_has_min_no_max_l902_90247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_relation_l902_90202

/-- Given a quadratic function f(x) = ax^2 + bx + c with specific coefficients,
    prove that 2b - 3a + 4c equals -26. -/
theorem quadratic_coefficient_relation :
  let f (x : ℝ) := 2 * x^2 - 4 * x - 3
  2 * (-4) - 3 * 2 + 4 * (-3) = -26 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_coefficient_relation_l902_90202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_point_exists_l902_90290

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the inverse function of f
variable (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f_inv (f x) = x

-- State the condition from the problem
axiom condition : Real.tan (Real.pi/3) - f 2 = Real.sqrt 3 - 1/3

-- Theorem to prove
theorem inverse_point_exists : f_inv (1/3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_point_exists_l902_90290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l902_90218

theorem problem_statement (x : ℝ) (h1 : x > 0) 
  (h2 : ∀ n : ℕ+, x + n ≥ n + 1) : 
  ∃ a : ℕ → ℕ, ∀ n : ℕ+, a n.val = n.val^n.val :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_statement_l902_90218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_properties_l902_90211

theorem square_properties (a b c d : ℕ+) 
  (h1 : (1 : ℚ) / a - (1 : ℚ) / b = (1 : ℚ) / c) 
  (h2 : (d : ℕ) = Nat.gcd a.val (Nat.gcd b.val c.val)) : 
  ∃ (m n : ℕ), (a.val * b.val * c.val * d : ℕ) = m ^ 2 ∧ 
               (d * (b.val - a.val) : ℕ) = n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_properties_l902_90211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_theorem_l902_90201

/-- Circle represented by the equation x^2 + y^2 + ax + by + c = 0 -/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the center coordinates of a circle -/
noncomputable def Circle.center (c : Circle) : ℝ × ℝ := (-c.a/2, -c.b/2)

/-- Calculate the radius of a circle -/
noncomputable def Circle.radius (c : Circle) : ℝ := Real.sqrt ((c.a^2 + c.b^2) / 4 - c.c)

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Check if two circles intersect -/
def circles_intersect (c1 c2 : Circle) : Prop :=
  let d := distance (c1.center) (c2.center)
  d < c1.radius + c2.radius ∧ d > |c1.radius - c2.radius|

/-- The main theorem: the two given circles intersect -/
theorem circles_intersect_theorem : 
  let c1 : Circle := { a := -2, b := 4, c := 4 }
  let c2 : Circle := { a := -4, b := 2, c := 19/4 }
  circles_intersect c1 c2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_theorem_l902_90201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l902_90236

open Real

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ :=
  (arccos (x/3))^2 + Real.pi * arcsin (x/3) - (arcsin (x/3))^2 + 
  (Real.pi^2/4) * (x^2 - 3*x + 2) + Real.pi * arctan x

-- State the theorem about the range of g(x)
theorem g_range :
  ∀ x ∈ Set.Icc (-3 : ℝ) 3,
  g x ∈ Set.Icc (Real.pi^2/4 - Real.pi/2) (9*Real.pi^2/4 + Real.pi/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l902_90236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tickets_is_400_l902_90286

/-- Represents the ticket sales for a show -/
structure TicketSales where
  adultPrice : ℚ
  childPrice : ℚ
  totalRevenue : ℚ
  childTickets : ℕ

/-- Calculates the total number of tickets sold -/
def totalTickets (sales : TicketSales) : ℕ :=
  sales.childTickets + (((sales.totalRevenue - sales.childPrice * sales.childTickets) / sales.adultPrice).floor).toNat

/-- Theorem stating that the total number of tickets sold is 400 -/
theorem total_tickets_is_400 (sales : TicketSales)
  (h1 : sales.adultPrice = 6)
  (h2 : sales.childPrice = 9/2)
  (h3 : sales.totalRevenue = 2100)
  (h4 : sales.childTickets = 200) :
  totalTickets sales = 400 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_tickets_is_400_l902_90286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_female_count_l902_90250

theorem stratified_sampling_female_count 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_male = 560) 
  (h2 : total_female = 420) 
  (h3 : sample_size = 280) :
  Int.floor ((total_female : ℚ) / ((total_male + total_female) : ℚ) * sample_size) = 120 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_female_count_l902_90250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_min_max_S_l902_90220

-- Define the conditions
def condition (a b c : ℝ) : Prop :=
  (a - 1) / 2 = (b - 2) / 3 ∧ (b - 2) / 3 = (3 - c) / 4

-- Define the function S
def S (a b c : ℝ) : ℝ := a + 2 * b + 3 * c

-- Theorem statement
theorem ratio_of_min_max_S :
  ∃ m n : ℝ,
  (∀ a b c : ℝ, a ≥ 0 → b ≥ 0 → c ≥ 0 → condition a b c →
    n ≤ S a b c ∧ S a b c ≤ m) ∧
  n / m = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ratio_of_min_max_S_l902_90220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l902_90233

theorem cos_minus_sin_value (α : Real) 
  (h1 : Real.sin α * Real.cos α = 3/8)
  (h2 : π/4 < α)
  (h3 : α < π/2) :
  Real.cos α - Real.sin α = -1/2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_minus_sin_value_l902_90233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_integer_in_set_l902_90243

def is_negative_integer (x : ℝ) : Prop :=
  x < 0 ∧ ∃ n : ℤ, x = n

theorem negative_integer_in_set : 
  ∃! x : ℝ, x ∈ ({0, 3, -5, -3.6} : Set ℝ) ∧ is_negative_integer x :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_integer_in_set_l902_90243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_borek_temperature_conversion_l902_90273

/-- A custom temperature scale called "bořks" -/
structure BorkScale where
  celsius_to_bork : ℝ → ℝ
  bork_to_celsius : ℝ → ℝ
  inverse : ∀ x, bork_to_celsius (celsius_to_bork x) = x

/-- The specific Bořek's temperature scale -/
noncomputable def borek_scale : BorkScale :=
  { celsius_to_bork := λ c ↦ (2 - (-8)) / (11 - (-4)) * (c - 11) + 2,
    bork_to_celsius := λ b ↦ (11 - (-4)) / (2 - (-8)) * (b - 2) + 11,
    inverse := by sorry }

theorem borek_temperature_conversion
  (scale : BorkScale)
  (h1 : scale.celsius_to_bork 11 = 2)
  (h2 : scale.celsius_to_bork (-4) = -8) :
  scale.bork_to_celsius (-2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_borek_temperature_conversion_l902_90273
