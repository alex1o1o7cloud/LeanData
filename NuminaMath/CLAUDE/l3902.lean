import Mathlib

namespace NUMINAMATH_CALUDE_circle_point_inequality_l3902_390205

theorem circle_point_inequality (m n c : ℝ) : 
  (∀ m n, m^2 + (n - 2)^2 = 1 → m + n + c ≥ 1) → c ≥ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_point_inequality_l3902_390205


namespace NUMINAMATH_CALUDE_geometry_theorem_l3902_390262

-- Define the types for lines and planes
variable {Point : Type*}
variable {Line : Type*}
variable {Plane : Type*}

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the theorem
theorem geometry_theorem 
  (m n l : Line) (α β : Plane) 
  (h_different_lines : m ≠ n ∧ m ≠ l ∧ n ≠ l) 
  (h_different_planes : α ≠ β) :
  (¬(subset m α) ∧ (subset n α) ∧ (parallel_lines m n) → parallel_line_plane m α) ∧
  ((subset m α) ∧ (perpendicular_line_plane m β) → perpendicular_planes α β) :=
sorry

end NUMINAMATH_CALUDE_geometry_theorem_l3902_390262


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3902_390236

theorem linear_equation_solution (x y a : ℝ) : 
  x = 1 → y = 2 → x - a * y = 3 → a = -1 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3902_390236


namespace NUMINAMATH_CALUDE_max_value_theorem_l3902_390217

theorem max_value_theorem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : x^2 - x*y + 2*y^2 = 8) :
  x^2 + x*y + 2*y^2 ≤ (72 + 32*Real.sqrt 2) / 7 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 - x₀*y₀ + 2*y₀^2 = 8 ∧
  x₀^2 + x₀*y₀ + 2*y₀^2 = (72 + 32*Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3902_390217


namespace NUMINAMATH_CALUDE_lastTwoDigits_7_2012_l3902_390222

/-- The last two digits of 7^n, for any natural number n -/
def lastTwoDigits (n : ℕ) : ℕ :=
  (7^n) % 100

/-- The pattern of last two digits repeats every 4 exponents -/
axiom lastTwoDigitsPattern (k : ℕ) :
  (lastTwoDigits (4*k - 2) = 49) ∧
  (lastTwoDigits (4*k - 1) = 43) ∧
  (lastTwoDigits (4*k) = 1) ∧
  (lastTwoDigits (4*k + 1) = 7)

theorem lastTwoDigits_7_2012 :
  lastTwoDigits 2012 = 1 := by sorry

end NUMINAMATH_CALUDE_lastTwoDigits_7_2012_l3902_390222


namespace NUMINAMATH_CALUDE_congruence_from_equal_sides_equal_sides_from_congruence_l3902_390244

/-- Triangle type -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Congruence relation between triangles -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Length of a side of a triangle -/
def side_length (t : Triangle) (i : Fin 3) : ℝ := sorry

/-- Two triangles have equal corresponding sides -/
def equal_sides (t1 t2 : Triangle) : Prop :=
  ∀ i : Fin 3, side_length t1 i = side_length t2 i

theorem congruence_from_equal_sides (t1 t2 : Triangle) :
  equal_sides t1 t2 → congruent t1 t2 := by sorry

theorem equal_sides_from_congruence (t1 t2 : Triangle) :
  congruent t1 t2 → equal_sides t1 t2 := by sorry

end NUMINAMATH_CALUDE_congruence_from_equal_sides_equal_sides_from_congruence_l3902_390244


namespace NUMINAMATH_CALUDE_game_points_calculation_l3902_390251

/-- Calculates the total points scored in a game given points per round and number of rounds played. -/
def totalPoints (pointsPerRound : ℕ) (numRounds : ℕ) : ℕ :=
  pointsPerRound * numRounds

/-- Theorem stating that for a game with 146 points per round and 157 rounds, the total points scored is 22822. -/
theorem game_points_calculation :
  totalPoints 146 157 = 22822 := by
  sorry

end NUMINAMATH_CALUDE_game_points_calculation_l3902_390251


namespace NUMINAMATH_CALUDE_unique_function_solution_l3902_390278

/-- The functional equation f(x + f(y)) = x + y + k has exactly one solution. -/
theorem unique_function_solution :
  ∃! f : ℝ → ℝ, ∃ k : ℝ, ∀ x y : ℝ, f (x + f y) = x + y + k :=
by sorry

end NUMINAMATH_CALUDE_unique_function_solution_l3902_390278


namespace NUMINAMATH_CALUDE_cousin_future_age_l3902_390255

/-- Given the ages of Nick and his relatives, prove the cousin's future age. -/
theorem cousin_future_age (nick_age : ℕ) (sister_age_diff : ℕ) (cousin_age_diff : ℕ) :
  nick_age = 13 →
  sister_age_diff = 6 →
  cousin_age_diff = 3 →
  let sister_age := nick_age + sister_age_diff
  let brother_age := (nick_age + sister_age) / 2
  let cousin_age := brother_age - cousin_age_diff
  cousin_age + (2 * brother_age - cousin_age) = 32 := by
  sorry

end NUMINAMATH_CALUDE_cousin_future_age_l3902_390255


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3902_390229

-- Define an isosceles triangle with sides 4, 8, and 8
def isosceles_triangle (a b c : ℝ) : Prop :=
  a = 4 ∧ b = 8 ∧ c = 8 ∧ b = c

-- Define the perimeter of a triangle
def triangle_perimeter (a b c : ℝ) : ℝ := a + b + c

-- Theorem statement
theorem isosceles_triangle_perimeter :
  ∀ a b c : ℝ, isosceles_triangle a b c → triangle_perimeter a b c = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l3902_390229


namespace NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3902_390241

/-- Given a cylinder with base area S and lateral surface that unfolds into a square,
    prove that its lateral surface area is 4πS -/
theorem cylinder_lateral_surface_area (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / Real.pi)
  let circumference := 2 * Real.pi * r
  let height := circumference
  circumference * height = 4 * Real.pi * S := by
  sorry

end NUMINAMATH_CALUDE_cylinder_lateral_surface_area_l3902_390241


namespace NUMINAMATH_CALUDE_scarf_final_price_l3902_390253

def original_price : ℝ := 15
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25

theorem scarf_final_price :
  original_price * (1 - first_discount) * (1 - second_discount) = 9 := by
  sorry

end NUMINAMATH_CALUDE_scarf_final_price_l3902_390253


namespace NUMINAMATH_CALUDE_sin_from_tan_l3902_390257

theorem sin_from_tan (a b x : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 < x) (h4 : x < π / 2)
  (h5 : Real.tan x = 2 * a * b / (a^2 - b^2)) : 
  Real.sin x = 2 * a * b / (a^2 + b^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_from_tan_l3902_390257


namespace NUMINAMATH_CALUDE_perpendicular_implies_perpendicular_lines_l3902_390218

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)
variable (perpendicularLines : Line → Line → Prop)

-- State the theorem
theorem perpendicular_implies_perpendicular_lines 
  (l m : Line) (α β : Plane) 
  (h1 : perpendicular l α) 
  (h2 : subset m β) :
  parallel α β → perpendicularLines l m :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_implies_perpendicular_lines_l3902_390218


namespace NUMINAMATH_CALUDE_M_intersect_N_eq_N_l3902_390232

-- Define the sets M and N
def M : Set ℝ := {x | x > 0}
def N : Set ℝ := {x | Real.log x > 0}

-- State the theorem
theorem M_intersect_N_eq_N : M ∩ N = N := by sorry

end NUMINAMATH_CALUDE_M_intersect_N_eq_N_l3902_390232


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3902_390209

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3902_390209


namespace NUMINAMATH_CALUDE_max_piles_theorem_l3902_390214

/-- Represents the state of the stone piles -/
structure StonePiles where
  piles : List Nat
  sum_stones : Nat
  deriving Repr

/-- Check if the piles satisfy the size constraint -/
def valid_piles (sp : StonePiles) : Prop :=
  ∀ i j, i < sp.piles.length → j < sp.piles.length →
    2 * sp.piles[i]! > sp.piles[j]! ∧ 2 * sp.piles[j]! > sp.piles[i]!

/-- The initial state with 660 stones -/
def initial_state : StonePiles :=
  { piles := [660], sum_stones := 660 }

/-- A move splits one pile into two smaller piles -/
def move (sp : StonePiles) (index : Nat) (split : Nat) : Option StonePiles :=
  if index ≥ sp.piles.length ∨ split ≥ sp.piles[index]! then none
  else some {
    piles := sp.piles.set index (sp.piles[index]! - split) |>.insertNth index split,
    sum_stones := sp.sum_stones
  }

/-- The theorem to be proved -/
theorem max_piles_theorem (sp : StonePiles) :
  sp.sum_stones = 660 →
  valid_piles sp →
  sp.piles.length ≤ 30 :=
sorry

#eval initial_state

end NUMINAMATH_CALUDE_max_piles_theorem_l3902_390214


namespace NUMINAMATH_CALUDE_p_has_four_digits_l3902_390239

-- Define p as given in the problem
def p : ℚ := 125 * 243 * 16 / 405

-- Function to count the number of digits in a rational number
def count_digits (q : ℚ) : ℕ := sorry

-- Theorem stating that p has 4 digits
theorem p_has_four_digits : count_digits p = 4 := by sorry

end NUMINAMATH_CALUDE_p_has_four_digits_l3902_390239


namespace NUMINAMATH_CALUDE_gym_class_students_l3902_390293

theorem gym_class_students (n : ℕ) : 
  150 ≤ n ∧ n ≤ 300 ∧ 
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 2 → 
  n = 165 ∨ n = 237 := by
sorry

end NUMINAMATH_CALUDE_gym_class_students_l3902_390293


namespace NUMINAMATH_CALUDE_triangle_abc_proof_l3902_390203

open Real

theorem triangle_abc_proof (a b c A B C : ℝ) (m n : ℝ × ℝ) :
  -- Given conditions
  (0 < A) → (A < 2 * π / 3) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (m = (a / 2, c / 2)) →
  (n = (cos C, cos A)) →
  (n.1 * m.1 + n.2 * m.2 = b * cos B) →
  (cos ((A - C) / 2) = sqrt 3 * sin A) →
  (m.1 * m.1 + m.2 * m.2 = 5) →
  -- Conclusions
  (B = π / 3) ∧
  (1 / 2 * a * b * sin C = 2 * sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_abc_proof_l3902_390203


namespace NUMINAMATH_CALUDE_smallest_factor_of_36_l3902_390290

theorem smallest_factor_of_36 (a b c : ℤ) 
  (h1 : a * b * c = 36)
  (h2 : a + b + c = 4) :
  min a (min b c) = -4 :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_of_36_l3902_390290


namespace NUMINAMATH_CALUDE_pizza_combinations_l3902_390202

/-- The number of available toppings -/
def n : ℕ := 8

/-- The number of toppings on each pizza -/
def k : ℕ := 3

/-- The maximum number of unique pizzas that can be made -/
def max_pizzas : ℕ := Nat.choose n k

theorem pizza_combinations :
  max_pizzas = 56 := by sorry

end NUMINAMATH_CALUDE_pizza_combinations_l3902_390202


namespace NUMINAMATH_CALUDE_polar_to_cartesian_line_l3902_390246

/-- The curve defined by the polar equation r = 1 / (sin θ + cos θ) is a straight line -/
theorem polar_to_cartesian_line : 
  ∀ (x y : ℝ), 
  (∃ (r θ : ℝ), r > 0 ∧ 
    x = r * Real.cos θ ∧ 
    y = r * Real.sin θ ∧ 
    r = 1 / (Real.sin θ + Real.cos θ)) 
  ↔ (x + y)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_line_l3902_390246


namespace NUMINAMATH_CALUDE_coin_problem_solution_l3902_390259

/-- Represents the number of coins of each denomination -/
structure CoinCount where
  one_jiao : ℕ
  five_jiao : ℕ

/-- Verifies if the given coin count satisfies the problem conditions -/
def is_valid_solution (coins : CoinCount) : Prop :=
  coins.one_jiao + coins.five_jiao = 30 ∧
  coins.one_jiao + 5 * coins.five_jiao = 86

/-- Theorem stating the unique solution to the coin problem -/
theorem coin_problem_solution : 
  ∃! (coins : CoinCount), is_valid_solution coins ∧ 
    coins.one_jiao = 16 ∧ coins.five_jiao = 14 := by sorry

end NUMINAMATH_CALUDE_coin_problem_solution_l3902_390259


namespace NUMINAMATH_CALUDE_zigzag_angle_theorem_l3902_390242

theorem zigzag_angle_theorem (ACB FEG DCE DEC : Real) : 
  ACB = 10 →
  FEG = 26 →
  DCE + 14 + 80 = 180 →
  DEC + 33 + 64 = 180 →
  ∃ θ : Real, θ = 11 ∧ θ + DCE + DEC = 180 :=
by sorry

end NUMINAMATH_CALUDE_zigzag_angle_theorem_l3902_390242


namespace NUMINAMATH_CALUDE_second_half_speed_l3902_390266

def total_distance : ℝ := 336
def total_time : ℝ := 15
def first_half_speed : ℝ := 21

theorem second_half_speed : ℝ := by
  have h1 : total_distance / 2 = first_half_speed * (total_time / 2) := by sorry
  have h2 : total_distance / 2 = 24 * (total_time - total_time / 2) := by sorry
  exact 24

end NUMINAMATH_CALUDE_second_half_speed_l3902_390266


namespace NUMINAMATH_CALUDE_factorial_divisibility_implies_inequality_l3902_390294

theorem factorial_divisibility_implies_inequality (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) 
  (h : (a.factorial + (a + b).factorial) ∣ (a.factorial * (a + b).factorial)) : 
  a ≥ 2 * b + 1 := by
sorry

end NUMINAMATH_CALUDE_factorial_divisibility_implies_inequality_l3902_390294


namespace NUMINAMATH_CALUDE_max_value_a_plus_2b_l3902_390258

theorem max_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^2 + 2*a*b + 4*b^2 = 6) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x^2 + 2*x*y + 4*y^2 = 6 → a + 2*b ≤ x + 2*y → a + 2*b ≤ 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_a_plus_2b_l3902_390258


namespace NUMINAMATH_CALUDE_exists_unreachable_all_plus_configuration_l3902_390292

/-- Represents the sign in a cell: + or - -/
inductive Sign
| Plus
| Minus

/-- Represents an 8x8 grid of signs -/
def Grid := Fin 8 → Fin 8 → Sign

/-- Represents the allowed operations: flipping signs in 3x3 or 4x4 squares -/
def flip_square (g : Grid) (top_left : Fin 8 × Fin 8) (size : Fin 2) : Grid :=
  sorry

/-- Counts the number of minus signs in specific columns of the grid -/
def count_minus_outside_columns_3_6 (g : Grid) : Nat :=
  sorry

/-- Theorem stating that there exists a grid configuration that cannot be transformed to all plus signs -/
theorem exists_unreachable_all_plus_configuration :
  ∃ (initial : Grid), ¬∃ (final : Grid),
    (∀ i j, final i j = Sign.Plus) ∧
    (∃ (ops : List ((Fin 8 × Fin 8) × Fin 2)),
      final = ops.foldl (λ g (tl, s) => flip_square g tl s) initial) :=
  sorry

end NUMINAMATH_CALUDE_exists_unreachable_all_plus_configuration_l3902_390292


namespace NUMINAMATH_CALUDE_line_passes_through_center_l3902_390240

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3*x + 2*y = 0

/-- The center of the circle -/
def center : ℝ × ℝ := (2, -3)

/-- Theorem stating that the line passes through the center of the circle -/
theorem line_passes_through_center : 
  line_equation center.1 center.2 ∧ circle_equation center.1 center.2 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_center_l3902_390240


namespace NUMINAMATH_CALUDE_unique_positive_zero_implies_negative_a_l3902_390291

/-- The function f(x) = ax³ + 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem unique_positive_zero_implies_negative_a :
  ∀ a : ℝ, (∃! x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) → a < 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_positive_zero_implies_negative_a_l3902_390291


namespace NUMINAMATH_CALUDE_max_x_minus_y_l3902_390243

theorem max_x_minus_y (x y z : ℝ) (sum_eq : x + y + z = 2) (prod_eq : x*y + y*z + z*x = 1) :
  ∃ (max : ℝ), max = 2 * Real.sqrt 3 / 3 ∧ ∀ (a b c : ℝ), a + b + c = 2 → a*b + b*c + c*a = 1 → |a - b| ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l3902_390243


namespace NUMINAMATH_CALUDE_sine_of_angle_l3902_390272

/-- Given an angle α with vertex at the origin, initial side on the non-negative x-axis,
    and terminal side in the third quadrant intersecting the unit circle at (-√5/5, m),
    prove that sin α = -2√5/5 -/
theorem sine_of_angle (α : Real) (m : Real) : 
  ((-Real.sqrt 5 / 5) ^ 2 + m ^ 2 = 1) →  -- Point on unit circle
  (m < 0) →  -- In third quadrant
  (Real.sin α = m) →  -- Definition of sine
  (Real.sin α = -2 * Real.sqrt 5 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sine_of_angle_l3902_390272


namespace NUMINAMATH_CALUDE_object_ends_on_left_l3902_390211

/-- Represents the sides of a square --/
inductive SquareSide
  | Top
  | Right
  | Bottom
  | Left

/-- Represents the vertices of a regular octagon --/
inductive OctagonVertex
  | Bottom
  | BottomLeft
  | Left
  | TopLeft
  | Top
  | TopRight
  | Right
  | BottomRight

/-- The number of sides a square rolls to reach the leftmost position from the bottom --/
def numRolls : Nat := 4

/-- The angle of rotation for each roll of the square --/
def rotationPerRoll : Int := 135

/-- Function to calculate the final position of an object on a square
    after rolling around an octagon --/
def finalPosition (initialSide : SquareSide) (rolls : Nat) : SquareSide :=
  sorry

/-- Theorem stating that an object initially on the right side of the square
    will end up on the left side after rolling to the leftmost position --/
theorem object_ends_on_left :
  finalPosition SquareSide.Right numRolls = SquareSide.Left :=
  sorry

end NUMINAMATH_CALUDE_object_ends_on_left_l3902_390211


namespace NUMINAMATH_CALUDE_abs_value_of_z_l3902_390228

theorem abs_value_of_z (z : ℂ) (h : z = Complex.I * (1 - Complex.I)) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_z_l3902_390228


namespace NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_gt_3_l3902_390261

-- Define the sets A, B, and C
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | x^2 - 12*x + 20 < 0}
def C (a : ℝ) : Set ℝ := {x | x < a}

-- Theorem 1: A ∪ B = {x | 2 < x < 10}
theorem union_A_B : A ∪ B = {x : ℝ | 2 < x ∧ x < 10} := by sorry

-- Theorem 2: (ℝ \ A) ∩ B = {x | 2 < x < 3 or 7 ≤ x < 10}
theorem complement_A_intersect_B : (Set.univ \ A) ∩ B = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} := by sorry

-- Theorem 3: If A ∩ C ≠ ∅, then a > 3
theorem intersection_A_C_nonempty_implies_a_gt_3 (a : ℝ) : (A ∩ C a).Nonempty → a > 3 := by sorry

end NUMINAMATH_CALUDE_union_A_B_complement_A_intersect_B_intersection_A_C_nonempty_implies_a_gt_3_l3902_390261


namespace NUMINAMATH_CALUDE_armband_cost_is_fifteen_l3902_390249

/-- The cost of an individual ride ticket in dollars -/
def ticket_cost : ℚ := 0.75

/-- The number of rides equivalent to the armband -/
def equivalent_rides : ℕ := 20

/-- The cost of the armband in dollars -/
def armband_cost : ℚ := ticket_cost * equivalent_rides

/-- Theorem stating that the armband costs $15.00 -/
theorem armband_cost_is_fifteen : armband_cost = 15 := by
  sorry

end NUMINAMATH_CALUDE_armband_cost_is_fifteen_l3902_390249


namespace NUMINAMATH_CALUDE_intersection_M_P_l3902_390220

-- Define the sets M and P
def M (a : ℝ) : Set ℝ := {x | x > a ∧ a^2 - 12*a + 20 < 0}
def P : Set ℝ := {x | x ≤ 10}

-- Theorem statement
theorem intersection_M_P (a : ℝ) : M a ∩ P = {x | a < x ∧ x ≤ 10} :=
by sorry

end NUMINAMATH_CALUDE_intersection_M_P_l3902_390220


namespace NUMINAMATH_CALUDE_marcus_scored_half_l3902_390245

/-- Calculates the percentage of team points scored by Marcus -/
def marcus_percentage (three_point_goals : ℕ) (two_point_goals : ℕ) (team_total : ℕ) : ℚ :=
  let marcus_points := 3 * three_point_goals + 2 * two_point_goals
  (marcus_points : ℚ) / team_total * 100

/-- Proves that Marcus scored 50% of the team's total points -/
theorem marcus_scored_half (three_point_goals two_point_goals team_total : ℕ) 
  (h1 : three_point_goals = 5)
  (h2 : two_point_goals = 10)
  (h3 : team_total = 70) :
  marcus_percentage three_point_goals two_point_goals team_total = 50 := by
sorry

#eval marcus_percentage 5 10 70

end NUMINAMATH_CALUDE_marcus_scored_half_l3902_390245


namespace NUMINAMATH_CALUDE_f_10_equals_107_l3902_390270

/-- The function f defined as f(n) = n^2 - n + 17 for all n -/
def f (n : ℕ) : ℕ := n^2 - n + 17

/-- Theorem stating that f(10) = 107 -/
theorem f_10_equals_107 : f 10 = 107 := by
  sorry

end NUMINAMATH_CALUDE_f_10_equals_107_l3902_390270


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_l3902_390254

theorem triangle_inequality_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a + b > c) (hbc : b + c > a) (hca : c + a > b) :
  c / (a + b) + a / (b + c) + b / (c + a) > 1 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_l3902_390254


namespace NUMINAMATH_CALUDE_inequality_proof_l3902_390260

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  (2 * a * b) / (a + b) > Real.sqrt (a * b) ∧ Real.sqrt (a * b) > (a + b) / 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3902_390260


namespace NUMINAMATH_CALUDE_percentage_of_270_l3902_390233

theorem percentage_of_270 : (33 + 1/3 : ℚ) / 100 * 270 = 90 := by sorry

end NUMINAMATH_CALUDE_percentage_of_270_l3902_390233


namespace NUMINAMATH_CALUDE_determinant_implies_cosine_l3902_390252

theorem determinant_implies_cosine (α : Real) : 
  (Real.cos (75 * π / 180) * Real.cos α + Real.sin (75 * π / 180) * Real.sin α = 1/3) →
  (Real.cos ((30 * π / 180) + 2 * α) = 7/9) := by
sorry

end NUMINAMATH_CALUDE_determinant_implies_cosine_l3902_390252


namespace NUMINAMATH_CALUDE_prob_ace_jack_queen_is_8_over_16575_l3902_390267

/-- A standard deck of cards. -/
def StandardDeck : ℕ := 52

/-- The number of Aces in a standard deck. -/
def NumAces : ℕ := 4

/-- The number of Jacks in a standard deck. -/
def NumJacks : ℕ := 4

/-- The number of Queens in a standard deck. -/
def NumQueens : ℕ := 4

/-- The probability of drawing an Ace, then a Jack, then a Queen from a standard deck without replacement. -/
def probAceJackQueen : ℚ :=
  (NumAces : ℚ) / StandardDeck *
  (NumJacks : ℚ) / (StandardDeck - 1) *
  (NumQueens : ℚ) / (StandardDeck - 2)

theorem prob_ace_jack_queen_is_8_over_16575 :
  probAceJackQueen = 8 / 16575 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_jack_queen_is_8_over_16575_l3902_390267


namespace NUMINAMATH_CALUDE_max_profit_scheme_l3902_390216

-- Define the variables
def bean_sprout_price : ℚ := 60
def dried_tofu_price : ℚ := 40
def bean_sprout_sell : ℚ := 80
def dried_tofu_sell : ℚ := 55
def total_units : ℕ := 200
def max_cost : ℚ := 10440

-- Define the profit function
def profit (bean_sprouts : ℕ) : ℚ :=
  (bean_sprout_sell - bean_sprout_price) * bean_sprouts + 
  (dried_tofu_sell - dried_tofu_price) * (total_units - bean_sprouts)

-- Theorem statement
theorem max_profit_scheme :
  ∀ bean_sprouts : ℕ,
  (2 * bean_sprout_price + 3 * dried_tofu_price = 240) →
  (3 * bean_sprout_price + 4 * dried_tofu_price = 340) →
  (bean_sprouts + (total_units - bean_sprouts) = total_units) →
  (bean_sprout_price * bean_sprouts + dried_tofu_price * (total_units - bean_sprouts) ≤ max_cost) →
  (bean_sprouts ≥ (3/2) * (total_units - bean_sprouts)) →
  profit bean_sprouts ≤ profit 122 ∧ profit 122 = 3610 := by
sorry

end NUMINAMATH_CALUDE_max_profit_scheme_l3902_390216


namespace NUMINAMATH_CALUDE_multiplication_division_equality_l3902_390227

theorem multiplication_division_equality : 15 * (1 / 5) * 40 / 4 = 30 := by sorry

end NUMINAMATH_CALUDE_multiplication_division_equality_l3902_390227


namespace NUMINAMATH_CALUDE_average_pages_proof_l3902_390288

def book_pages : List Nat := [50, 75, 80, 120, 100, 90, 110, 130]

theorem average_pages_proof :
  (book_pages.sum : ℚ) / book_pages.length = 94.375 := by
  sorry

end NUMINAMATH_CALUDE_average_pages_proof_l3902_390288


namespace NUMINAMATH_CALUDE_f_is_even_and_increasing_l3902_390295

def f (x : ℝ) : ℝ := |x| + 1

theorem f_is_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) := by
sorry

end NUMINAMATH_CALUDE_f_is_even_and_increasing_l3902_390295


namespace NUMINAMATH_CALUDE_max_value_theorem_l3902_390284

theorem max_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 12) : 
  ∃ (z : ℝ), z = x^2 + 2*x*y + 3*y^2 ∧ z ≤ 132 + 48 * Real.sqrt 3 ∧
  ∃ (a b : ℝ), a^2 - 2*a*b + 3*b^2 = 12 ∧ a > 0 ∧ b > 0 ∧
  a^2 + 2*a*b + 3*b^2 = 132 + 48 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l3902_390284


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3902_390287

theorem arithmetic_expression_equality : 5 * 7 - (3 * 2 + 5 * 4) / 2 = 22 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3902_390287


namespace NUMINAMATH_CALUDE_circle_radius_d_value_l3902_390256

theorem circle_radius_d_value (d : ℝ) :
  (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + d = 0 → (x - 4)^2 + (y + 5)^2 = 36) →
  d = 5 := by
sorry

end NUMINAMATH_CALUDE_circle_radius_d_value_l3902_390256


namespace NUMINAMATH_CALUDE_student_sums_correct_l3902_390224

theorem student_sums_correct (total : ℕ) (correct : ℕ) (wrong : ℕ) : 
  total = 54 → 
  wrong = 2 * correct → 
  total = correct + wrong → 
  correct = 18 := by
sorry

end NUMINAMATH_CALUDE_student_sums_correct_l3902_390224


namespace NUMINAMATH_CALUDE_plant_structure_unique_solution_l3902_390273

/-- Represents a plant with branches and small branches -/
structure Plant where
  branches : ℕ
  smallBranchesPerBranch : ℕ

/-- The total number of parts (main stem, branches, and small branches) in a plant -/
def totalParts (p : Plant) : ℕ :=
  1 + p.branches + p.branches * p.smallBranchesPerBranch

/-- Theorem stating that a plant with 6 small branches per branch satisfies the given conditions -/
theorem plant_structure : ∃ (p : Plant), p.smallBranchesPerBranch = 6 ∧ totalParts p = 43 :=
  sorry

/-- Theorem proving that 6 is the unique solution for the number of small branches per branch -/
theorem unique_solution (p : Plant) (h : totalParts p = 43) : p.smallBranchesPerBranch = 6 :=
  sorry

end NUMINAMATH_CALUDE_plant_structure_unique_solution_l3902_390273


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3902_390213

def is_valid_number (n : ℕ) : Prop :=
  ∃ (k : ℕ) (m : ℕ),
    n = 5 * 10^k + m ∧
    m * 10 + 5 = (5 * 10^k + m) / 4

theorem smallest_valid_number :
  ∃ (n : ℕ),
    is_valid_number n ∧
    ∀ (m : ℕ), is_valid_number m → n ≤ m ∧
    n = 512820
  := by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3902_390213


namespace NUMINAMATH_CALUDE_chinese_vs_english_spanish_difference_l3902_390212

def hours_english : ℕ := 6
def hours_chinese : ℕ := 7
def hours_spanish : ℕ := 4
def hours_french : ℕ := 5

theorem chinese_vs_english_spanish_difference :
  Int.natAbs (hours_chinese - (hours_english + hours_spanish)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_chinese_vs_english_spanish_difference_l3902_390212


namespace NUMINAMATH_CALUDE_points_six_units_from_negative_one_l3902_390281

theorem points_six_units_from_negative_one :
  let a : ℝ := -1
  let distance : ℝ := 6
  let point_left : ℝ := a - distance
  let point_right : ℝ := a + distance
  point_left = -7 ∧ point_right = 5 := by
sorry

end NUMINAMATH_CALUDE_points_six_units_from_negative_one_l3902_390281


namespace NUMINAMATH_CALUDE_probability_theorem_l3902_390269

def is_valid (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  1 ≤ a ∧ a ≤ 12 ∧
  1 ≤ b ∧ b ≤ 12 ∧
  1 ≤ c ∧ c ≤ 12 ∧
  a = 2 * b ∧ b = 2 * c

def total_assignments : ℕ := 12 * 11 * 10

def valid_assignments : ℕ := 3

theorem probability_theorem :
  (valid_assignments : ℚ) / total_assignments = 1 / 440 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3902_390269


namespace NUMINAMATH_CALUDE_geometric_progression_arcsin_least_t_l3902_390230

theorem geometric_progression_arcsin_least_t : 
  ∃ (t : ℝ), t > 0 ∧ 
  (∀ (α : ℝ), 0 < α → α < π / 2 → 
    ∃ (r : ℝ), r > 0 ∧
    (Real.arcsin (Real.sin α) = α) ∧
    (Real.arcsin (Real.sin (3 * α)) = r * α) ∧
    (Real.arcsin (Real.sin (8 * α)) = r^2 * α) ∧
    (Real.arcsin (Real.sin (t * α)) = r^3 * α)) ∧
  (∀ (t' : ℝ), t' > 0 → 
    (∀ (α : ℝ), 0 < α → α < π / 2 → 
      ∃ (r : ℝ), r > 0 ∧
      (Real.arcsin (Real.sin α) = α) ∧
      (Real.arcsin (Real.sin (3 * α)) = r * α) ∧
      (Real.arcsin (Real.sin (8 * α)) = r^2 * α) ∧
      (Real.arcsin (Real.sin (t' * α)) = r^3 * α)) →
    t ≤ t') ∧
  t = 16 * Real.sqrt 6 / 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_arcsin_least_t_l3902_390230


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3902_390271

theorem quadratic_equation_properties (k : ℝ) :
  let f (x : ℝ) := x^2 + (2*k - 1)*x - k - 1
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  (x₁ + x₂ - 4*x₁*x₂ = 2 → k = -3/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3902_390271


namespace NUMINAMATH_CALUDE_quadratic_point_m_l3902_390247

/-- Given a quadratic function y = -ax² + 2ax + 3 where a > 0,
    if the point (m, 3) lies on the graph and m ≠ 0, then m = 2. -/
theorem quadratic_point_m (a m : ℝ) (ha : a > 0) (hm : m ≠ 0) :
  (3 = -a * m^2 + 2 * a * m + 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_point_m_l3902_390247


namespace NUMINAMATH_CALUDE_no_three_common_tangents_l3902_390277

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents the number of common tangents between two circles -/
def commonTangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- Theorem: It's impossible for two circles in the same plane to have exactly 3 common tangents -/
theorem no_three_common_tangents (c1 c2 : Circle) : 
  commonTangents c1 c2 ≠ 3 := by
  sorry

end NUMINAMATH_CALUDE_no_three_common_tangents_l3902_390277


namespace NUMINAMATH_CALUDE_divisibility_condition_l3902_390210

/-- Sum of divisors of n -/
def A (n : ℕ+) : ℕ := sorry

/-- Sum of products of pairs of divisors of n -/
def B (n : ℕ+) : ℕ := sorry

/-- A positive integer n is a perfect square -/
def is_perfect_square (n : ℕ+) : Prop := ∃ m : ℕ+, n = m ^ 2

theorem divisibility_condition (n : ℕ+) : 
  (A n ∣ B n) ↔ is_perfect_square n := by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3902_390210


namespace NUMINAMATH_CALUDE_problem_statement_l3902_390237

theorem problem_statement (n m : ℕ) : 
  2 * 8^n * 16^n = 2^15 →
  (∀ x y : ℝ, (m*x + y) * (2*x - y) = 2*m*x^2 - y^2) →
  n - m = 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3902_390237


namespace NUMINAMATH_CALUDE_subtracted_amount_l3902_390235

theorem subtracted_amount (N : ℝ) (A : ℝ) : 
  N = 300 → 0.30 * N - A = 20 → A = 70 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_amount_l3902_390235


namespace NUMINAMATH_CALUDE_centroid_of_concave_pentagon_l3902_390208

/-- A regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- A rhombus -/
structure Rhombus where
  vertices : Fin 4 → ℝ × ℝ
  is_rhombus : sorry

/-- The centroid of a planar figure -/
def centroid (figure : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Theorem: Centroid of concave pentagonal plate -/
theorem centroid_of_concave_pentagon
  (ABCDE : RegularPentagon)
  (ABFE : Rhombus)
  (hF : F = sorry) -- F is the intersection of diagonals EC and BD
  (hABFE : sorry) -- ABFE is cut out from ABCDE
  : centroid (sorry) = F := by sorry

end NUMINAMATH_CALUDE_centroid_of_concave_pentagon_l3902_390208


namespace NUMINAMATH_CALUDE_sqrt_two_minus_two_cos_four_equals_two_sin_two_l3902_390276

theorem sqrt_two_minus_two_cos_four_equals_two_sin_two :
  Real.sqrt (2 - 2 * Real.cos 4) = 2 * Real.sin 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_two_cos_four_equals_two_sin_two_l3902_390276


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l3902_390225

def a : ℝ × ℝ := (1, 3)
def b (x : ℝ) : ℝ × ℝ := (x, -1)

theorem perpendicular_vectors (x : ℝ) : 
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l3902_390225


namespace NUMINAMATH_CALUDE_school_pairing_fraction_l3902_390296

theorem school_pairing_fraction (t s : ℕ) (ht : t > 0) (hs : s > 0) : 
  (t / 4 : ℚ) = (s / 3 : ℚ) → 
  ((t / 4 + s / 3) : ℚ) / ((t + s) : ℚ) = 2 / 7 := by
sorry

end NUMINAMATH_CALUDE_school_pairing_fraction_l3902_390296


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3902_390206

theorem absolute_value_inequality (x : ℝ) : 2 ≤ |x - 5| ∧ |x - 5| ≤ 4 ↔ x ∈ Set.Icc 1 3 ∪ Set.Icc 7 9 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3902_390206


namespace NUMINAMATH_CALUDE_movie_channels_cost_12_l3902_390221

def basic_cable_cost : ℝ := 15
def total_cost : ℝ := 36

def movie_channel_cost : ℝ → Prop :=
  λ m => m > 0 ∧ 
         m + (m - 3) + basic_cable_cost = total_cost

theorem movie_channels_cost_12 : 
  movie_channel_cost 12 := by sorry

end NUMINAMATH_CALUDE_movie_channels_cost_12_l3902_390221


namespace NUMINAMATH_CALUDE_fourth_task_completion_time_l3902_390226

-- Define a custom time type
structure Time where
  hours : Nat
  minutes : Nat

-- Define the problem parameters
def start_time : Time := { hours := 8, minutes := 45 }
def third_task_completion : Time := { hours := 11, minutes := 25 }
def num_tasks : Nat := 4

-- Calculate the time difference in minutes
def time_diff (t1 t2 : Time) : Nat :=
  (t2.hours * 60 + t2.minutes) - (t1.hours * 60 + t1.minutes)

-- Calculate the duration of a single task
def single_task_duration : Nat :=
  (time_diff start_time third_task_completion) / (num_tasks - 1)

-- Function to add minutes to a given time
def add_minutes (t : Time) (m : Nat) : Time :=
  let total_minutes := t.hours * 60 + t.minutes + m
  { hours := total_minutes / 60, minutes := total_minutes % 60 }

-- Theorem to prove
theorem fourth_task_completion_time :
  add_minutes third_task_completion single_task_duration = { hours := 12, minutes := 18 } := by
  sorry

end NUMINAMATH_CALUDE_fourth_task_completion_time_l3902_390226


namespace NUMINAMATH_CALUDE_area_of_region_l3902_390283

-- Define the lower bound function
def lower_bound (x : ℝ) : ℝ := |x - 4|

-- Define the upper bound function
def upper_bound (x : ℝ) : ℝ := 5 - |x - 2|

-- Define the region
def region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | lower_bound p.1 ≤ p.2 ∧ p.2 ≤ upper_bound p.1}

-- Theorem statement
theorem area_of_region : MeasureTheory.volume region = 10 := by
  sorry

end NUMINAMATH_CALUDE_area_of_region_l3902_390283


namespace NUMINAMATH_CALUDE_solutions_are_correct_l3902_390268

def solutions : Set ℂ := {
  (16/15)^(1/4) + Complex.I * (16/15)^(1/4),
  -(16/15)^(1/4) - Complex.I * (16/15)^(1/4),
  -(16/15)^(1/4) + Complex.I * (16/15)^(1/4),
  (16/15)^(1/4) - Complex.I * (16/15)^(1/4),
  Complex.I * 2^(2/3),
  -Complex.I * 2^(2/3)
}

theorem solutions_are_correct : {z : ℂ | z^6 = -16} = solutions := by
  sorry

end NUMINAMATH_CALUDE_solutions_are_correct_l3902_390268


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l3902_390238

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

-- Define the property of Fibonacci sequence modulo 9 repeating every 24 terms
axiom fib_mod_9_period_24 : ∀ n : ℕ, fib n % 9 = fib (n % 24) % 9

-- Theorem: The remainder when the 150th Fibonacci number is divided by 9 is 8
theorem fib_150_mod_9 : fib 150 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l3902_390238


namespace NUMINAMATH_CALUDE_rent_utilities_percentage_after_raise_l3902_390286

theorem rent_utilities_percentage_after_raise (initial_income : ℝ) 
  (initial_percentage : ℝ) (salary_increase : ℝ) : 
  initial_income = 1000 →
  initial_percentage = 40 →
  salary_increase = 600 →
  let initial_amount := initial_income * (initial_percentage / 100)
  let new_income := initial_income + salary_increase
  (initial_amount / new_income) * 100 = 25 := by
sorry

end NUMINAMATH_CALUDE_rent_utilities_percentage_after_raise_l3902_390286


namespace NUMINAMATH_CALUDE_cistern_emptying_l3902_390223

/-- Represents the fraction of a cistern emptied in a given time -/
def fractionEmptied (time : ℚ) : ℚ :=
  if time = 8 then 1/3
  else if time = 16 then 2/3
  else 0

theorem cistern_emptying (t : ℚ) :
  fractionEmptied 8 = 1/3 →
  fractionEmptied 16 = 2 * fractionEmptied 8 :=
by sorry

end NUMINAMATH_CALUDE_cistern_emptying_l3902_390223


namespace NUMINAMATH_CALUDE_statement_d_is_incorrect_l3902_390231

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp : Plane → Plane → Prop)
variable (perpLine : Line → Plane → Prop)
variable (perpLines : Line → Line → Prop)

-- State the theorem
theorem statement_d_is_incorrect
  (α β : Plane) (l m n : Line)
  (h_diff_planes : α ≠ β)
  (h_diff_lines : l ≠ m ∧ m ≠ n ∧ l ≠ n)
  (h_perp_planes : perp α β)
  (h_perp_m_α : perpLine m α)
  (h_perp_n_β : perpLine n β) :
  ¬ (∀ m n, perpLines m n) :=
sorry

end NUMINAMATH_CALUDE_statement_d_is_incorrect_l3902_390231


namespace NUMINAMATH_CALUDE_average_sale_is_7000_l3902_390200

/-- Calculates the average sale for six months given the sales of five months and a required sale for the sixth month. -/
def average_sale (sales : List ℕ) (required_sale : ℕ) : ℚ :=
  (sales.sum + required_sale) / 6

/-- Theorem stating that the average sale for the given problem is 7000. -/
theorem average_sale_is_7000 :
  let sales : List ℕ := [4000, 6524, 5689, 7230, 6000]
  let required_sale : ℕ := 12557
  average_sale sales required_sale = 7000 := by
  sorry

#eval average_sale [4000, 6524, 5689, 7230, 6000] 12557

end NUMINAMATH_CALUDE_average_sale_is_7000_l3902_390200


namespace NUMINAMATH_CALUDE_article_cost_price_l3902_390207

theorem article_cost_price (profit_percent : ℝ) (discount_percent : ℝ) (price_reduction : ℝ) (new_profit_percent : ℝ) :
  profit_percent = 25 →
  discount_percent = 20 →
  price_reduction = 8.40 →
  new_profit_percent = 30 →
  ∃ (cost : ℝ), 
    cost > 0 ∧
    (cost + profit_percent / 100 * cost) - price_reduction = 
    (cost * (1 - discount_percent / 100)) * (1 + new_profit_percent / 100) ∧
    cost = 40 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_price_l3902_390207


namespace NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3902_390215

/-- Given a complex number z = a^2 - 1 + (a+1)i where a ∈ ℝ and z is a pure imaginary number,
    the imaginary part of 1/(z+a) is -2/5 -/
theorem imaginary_part_of_reciprocal (a : ℝ) (z : ℂ) : 
  z = a^2 - 1 + (a + 1) * I → 
  z.re = 0 → 
  z.im ≠ 0 → 
  Complex.im (1 / (z + a)) = -2/5 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_reciprocal_l3902_390215


namespace NUMINAMATH_CALUDE_sum_square_gt_four_times_adjacent_products_l3902_390279

theorem sum_square_gt_four_times_adjacent_products 
  (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 > 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end NUMINAMATH_CALUDE_sum_square_gt_four_times_adjacent_products_l3902_390279


namespace NUMINAMATH_CALUDE_min_value_theorem_l3902_390204

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) :
  y / x + 4 / y ≥ 8 ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ y / x + 4 / y = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3902_390204


namespace NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l3902_390264

-- Part 1: Non-existence of positive integer sequence
theorem no_positive_integer_sequence :
  ¬ ∃ (a : ℕ → ℕ+), ∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2)) :=
sorry

-- Part 2: Existence of positive irrational number sequence
theorem exists_positive_irrational_sequence :
  ∃ (a : ℕ → ℝ), (∀ n : ℕ, Irrational (a n) ∧ a n > 0) ∧
    (∀ n : ℕ, (a (n + 1))^2 ≥ 2 * (a n) * (a (n + 2))) :=
sorry

end NUMINAMATH_CALUDE_no_positive_integer_sequence_exists_positive_irrational_sequence_l3902_390264


namespace NUMINAMATH_CALUDE_adventure_distance_l3902_390289

/-- The square of the distance between two points, where one travels east and the other travels north. -/
def squareDistance (eastDistance : ℝ) (northDistance : ℝ) : ℝ :=
  eastDistance ^ 2 + northDistance ^ 2

/-- Theorem: The square of the distance between two points, where one travels 18 km east
    and the other travels 10 km north, is 424 km². -/
theorem adventure_distance : squareDistance 18 10 = 424 := by
  sorry

end NUMINAMATH_CALUDE_adventure_distance_l3902_390289


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l3902_390201

theorem weight_of_replaced_person
  (n : ℕ) -- number of people in the group
  (w_new : ℝ) -- weight of the new person
  (w_avg_increase : ℝ) -- increase in average weight
  (h1 : n = 12) -- there are 12 people initially
  (h2 : w_new = 106) -- the new person weighs 106 kg
  (h3 : w_avg_increase = 4) -- average weight increases by 4 kg
  : ∃ w_old : ℝ, w_old = 58 ∧ n * w_avg_increase = w_new - w_old :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l3902_390201


namespace NUMINAMATH_CALUDE_jelly_bean_ratio_l3902_390219

theorem jelly_bean_ratio : 
  ∀ (total_jelly_beans red_jelly_beans coconut_flavored_red_jelly_beans : ℕ),
    total_jelly_beans = 4000 →
    coconut_flavored_red_jelly_beans = 750 →
    4 * coconut_flavored_red_jelly_beans = red_jelly_beans →
    3 * total_jelly_beans = 4 * red_jelly_beans :=
by
  sorry

end NUMINAMATH_CALUDE_jelly_bean_ratio_l3902_390219


namespace NUMINAMATH_CALUDE_pizza_sales_l3902_390297

theorem pizza_sales (small_price large_price total_slices total_revenue : ℕ)
  (h1 : small_price = 150)
  (h2 : large_price = 250)
  (h3 : total_slices = 5000)
  (h4 : total_revenue = 1050000) :
  ∃ (small_slices large_slices : ℕ),
    small_slices + large_slices = total_slices ∧
    small_price * small_slices + large_price * large_slices = total_revenue ∧
    small_slices = 1500 :=
by sorry

end NUMINAMATH_CALUDE_pizza_sales_l3902_390297


namespace NUMINAMATH_CALUDE_tangent_x_axis_tangent_y_axis_unique_x_intercept_unique_y_intercept_is_parabola_l3902_390275

/-- A parabola represented by the equation (x + 1/2y - 1)² = 0 -/
def parabola (x y : ℝ) : Prop := (x + 1/2 * y - 1)^2 = 0

/-- The parabola is tangent to the x-axis at the point (1,0) -/
theorem tangent_x_axis : parabola 1 0 := by sorry

/-- The parabola is tangent to the y-axis at the point (0,2) -/
theorem tangent_y_axis : parabola 0 2 := by sorry

/-- The parabola touches the x-axis only at (1,0) -/
theorem unique_x_intercept (x : ℝ) : 
  parabola x 0 → x = 1 := by sorry

/-- The parabola touches the y-axis only at (0,2) -/
theorem unique_y_intercept (y : ℝ) : 
  parabola 0 y → y = 2 := by sorry

/-- The equation represents a parabola -/
theorem is_parabola : 
  ∃ (a b c : ℝ), ∀ (x y : ℝ), parabola x y ↔ y = a*x^2 + b*x + c := by sorry

end NUMINAMATH_CALUDE_tangent_x_axis_tangent_y_axis_unique_x_intercept_unique_y_intercept_is_parabola_l3902_390275


namespace NUMINAMATH_CALUDE_exponent_addition_l3902_390298

theorem exponent_addition (a : ℝ) : a^3 * a^6 = a^9 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l3902_390298


namespace NUMINAMATH_CALUDE_power_difference_sum_equals_six_l3902_390299

theorem power_difference_sum_equals_six : 3^2 - 2^2 + 1^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_sum_equals_six_l3902_390299


namespace NUMINAMATH_CALUDE_rainy_days_pigeonhole_l3902_390285

theorem rainy_days_pigeonhole (n : ℕ) (m : ℕ) (h : n > 2 * m) :
  ∃ (x : ℕ), x ≤ m ∧ (∃ (S : Finset ℕ), S.card ≥ 3 ∧ ∀ i ∈ S, i < n ∧ x = i % (m + 1)) :=
by
  sorry

#check rainy_days_pigeonhole 64 30

end NUMINAMATH_CALUDE_rainy_days_pigeonhole_l3902_390285


namespace NUMINAMATH_CALUDE_smallest_right_triangle_area_l3902_390265

/-- The smallest area of a right triangle with sides 6 and 8 -/
theorem smallest_right_triangle_area :
  let sides : Finset ℝ := {6, 8}
  ∃ (a b c : ℝ), a ∈ sides ∧ b ∈ sides ∧ c > 0 ∧
    a^2 + b^2 = c^2 ∧
    (∀ (x y z : ℝ), x ∈ sides → y ∈ sides → z > 0 → x^2 + y^2 = z^2 →
      (1/2) * a * b ≤ (1/2) * x * y) ∧
    (1/2) * a * b = 6 * Real.sqrt 7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_right_triangle_area_l3902_390265


namespace NUMINAMATH_CALUDE_number_of_combinations_prob_one_black_prob_at_least_one_blue_l3902_390263

/-- Represents the total number of pens -/
def total_pens : ℕ := 6

/-- Represents the number of black pens -/
def black_pens : ℕ := 3

/-- Represents the number of blue pens -/
def blue_pens : ℕ := 2

/-- Represents the number of red pens -/
def red_pens : ℕ := 1

/-- Represents the number of pens to be selected -/
def selected_pens : ℕ := 3

/-- Theorem stating the number of possible combinations when selecting 3 pens out of 6 -/
theorem number_of_combinations : Nat.choose total_pens selected_pens = 20 := by sorry

/-- Theorem stating the probability of selecting exactly one black pen -/
theorem prob_one_black : (Nat.choose black_pens 1 * Nat.choose (blue_pens + red_pens) 2) / Nat.choose total_pens selected_pens = 9 / 20 := by sorry

/-- Theorem stating the probability of selecting at least one blue pen -/
theorem prob_at_least_one_blue : 1 - (Nat.choose (black_pens + red_pens) selected_pens) / Nat.choose total_pens selected_pens = 4 / 5 := by sorry

end NUMINAMATH_CALUDE_number_of_combinations_prob_one_black_prob_at_least_one_blue_l3902_390263


namespace NUMINAMATH_CALUDE_sin_210_degrees_l3902_390280

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l3902_390280


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l3902_390282

/-- Given that α is inversely proportional to β, prove that α = -8/3 when β = -3,
    given that α = 4 when β = 2. -/
theorem inverse_proportion_problem (α β : ℝ) (h1 : ∃ k, ∀ x y, x * y = k → (α = x ↔ β = y))
    (h2 : α = 4 ∧ β = 2) : β = -3 → α = -8/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l3902_390282


namespace NUMINAMATH_CALUDE_gcd_4034_10085_base5_l3902_390274

/-- Converts a natural number to its base-5 representation as a list of digits -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

/-- Checks if a list of digits is a valid base-5 representation -/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

theorem gcd_4034_10085_base5 :
  let g := Nat.gcd 4034 10085
  isValidBase5 (toBase5 g) ∧ toBase5 g = [2, 3, 0, 1, 3] := by
  sorry

end NUMINAMATH_CALUDE_gcd_4034_10085_base5_l3902_390274


namespace NUMINAMATH_CALUDE_equation_system_solution_l3902_390234

/-- Represents a solution to the equation system -/
structure Solution :=
  (x : ℚ)
  (y : ℚ)

/-- Represents the equation system 2ax + y = 5 and 2x - by = 13 -/
def EquationSystem (a b : ℚ) (sol : Solution) : Prop :=
  2 * a * sol.x + sol.y = 5 ∧ 2 * sol.x - b * sol.y = 13

/-- Theorem stating the conditions and the correct solution -/
theorem equation_system_solution :
  let personA : Solution := ⟨7/2, -2⟩
  let personB : Solution := ⟨3, -7⟩
  let correctSol : Solution := ⟨2, -3⟩
  ∀ a b : ℚ,
    (EquationSystem 1 b personA) →  -- Person A misread a as 1
    (EquationSystem a 1 personB) →  -- Person B misread b as 1
    (a = 2 ∧ b = 3) ∧               -- Correct values of a and b
    (EquationSystem a b correctSol) -- Correct solution
  := by sorry

end NUMINAMATH_CALUDE_equation_system_solution_l3902_390234


namespace NUMINAMATH_CALUDE_coefficient_x10_is_179_l3902_390250

/-- The coefficient of x^10 in the expansion of (x+2)^10(x^2-1) -/
def coefficient_x10 : ℤ := 179

/-- The expansion of (x+2)^10(x^2-1) -/
def expansion (x : ℝ) : ℝ := (x + 2)^10 * (x^2 - 1)

/-- Theorem stating that the coefficient of x^10 in the expansion is equal to 179 -/
theorem coefficient_x10_is_179 : 
  (∃ f : ℝ → ℝ, ∀ x, expansion x = coefficient_x10 * x^10 + f x ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x| < δ → |f x| < ε * |x|^10)) :=
sorry

end NUMINAMATH_CALUDE_coefficient_x10_is_179_l3902_390250


namespace NUMINAMATH_CALUDE_original_triangle_area_l3902_390248

theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ (side : ℝ), new_area = (4 * side)^2 * (original_area / side^2)) →
  new_area = 64 →
  original_area = 4 := by
sorry

end NUMINAMATH_CALUDE_original_triangle_area_l3902_390248
