import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_count_two_digit_numbers_count_proof_l912_91297

theorem two_digit_numbers_count : Nat := 16

theorem two_digit_numbers_count_proof :
  let digits : Finset Nat := {0, 1, 2, 3, 4}
  let valid_first_digits := digits.filter (λ d => d ≠ 0)
  (valid_first_digits.card * (digits.card - 1)) = 16 := by
    sorry

#check two_digit_numbers_count
#check two_digit_numbers_count_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_digit_numbers_count_two_digit_numbers_count_proof_l912_91297


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_power_l912_91283

theorem consecutive_product_power (n m k : ℕ) 
  (hn : n > 0) (hm : m > 0) (hk : k > 0)
  (h : (n - 1) * n * (n + 1) = m ^ k) : k = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_product_power_l912_91283


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l912_91209

def a : ℕ → ℝ → ℝ
  | 0, _ => 1
  | 1, _ => 2
  | (n+2), q => (1+q) * a (n+1) q - q * a n q

def b (n : ℕ) (q : ℝ) : ℝ := a (n+1) q - a n q

theorem sequence_properties (q : ℝ) (h : q ≠ 0) :
  (∀ n : ℕ, n ≥ 1 → ∃ r : ℝ, b (n+1) q = r * b n q) ∧
  (∀ n : ℕ, n ≥ 1 → 
    a n q = if q ≠ 1 
      then 1 + (1 - q^(n-1)) / (1 - q)
      else n) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l912_91209


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_equals_perimeter_l912_91290

theorem rectangle_area_equals_perimeter (x : ℝ) : 
  (4 * x) * (x + 4) = 2 * ((4 * x) + (x + 4)) → x = (-3 + Real.sqrt 41) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_equals_perimeter_l912_91290


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_eq_one_l912_91250

open Real

/-- The function f defined as f(x) = e^(ax) - x - 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp (a * x) - x - 1

/-- Theorem stating that if f(x) ≥ 0 for all x ∈ ℝ and a ≠ 0, then a = 1 -/
theorem f_nonnegative_implies_a_eq_one (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x : ℝ, f a x ≥ 0) : a = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_nonnegative_implies_a_eq_one_l912_91250


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equation_unique_solution_l912_91288

/-- Represents a single digit (0-9) -/
def Digit := Fin 10

/-- Converts a two-digit number represented by two digits into an integer -/
def twoDigitToInt (tens ones : Digit) : ℕ :=
  10 * tens.val + ones.val

/-- Converts a three-digit number represented by three digits into an integer -/
def threeDigitToInt (hundreds tens ones : Digit) : ℕ :=
  100 * hundreds.val + 10 * tens.val + ones.val

/-- The equation AA + AB = BBB has a unique solution where A = 5 and B = 5 -/
theorem digit_equation_unique_solution :
  ∃! (A B : Digit), 
    (twoDigitToInt A A) + (twoDigitToInt A B) = (threeDigitToInt B B B) ∧
    A = ⟨5, by norm_num⟩ ∧ B = ⟨5, by norm_num⟩ := by
  sorry

#check digit_equation_unique_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_equation_unique_solution_l912_91288


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_order_l912_91276

-- Define the constants
noncomputable def a : ℝ := Real.tan 1
noncomputable def b : ℝ := Real.tan 2
noncomputable def c : ℝ := Real.tan 3

-- Theorem statement
theorem tan_order : b < c ∧ c < a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_order_l912_91276


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_handle_is_sixty_cents_l912_91244

/-- Represents the cost structure and break-even point for a handle molding operation -/
structure HandleMoldingOperation where
  fixedCost : ℚ
  sellingPrice : ℚ
  breakEvenQuantity : ℚ

/-- Calculates the cost per handle to mold -/
def costPerHandle (op : HandleMoldingOperation) : ℚ :=
  (op.sellingPrice * op.breakEvenQuantity - op.fixedCost) / op.breakEvenQuantity

/-- Theorem stating that given the specific conditions, the cost per handle to mold is $0.60 -/
theorem cost_per_handle_is_sixty_cents 
  (op : HandleMoldingOperation) 
  (h1 : op.fixedCost = 7640)
  (h2 : op.sellingPrice = 46/10)
  (h3 : op.breakEvenQuantity = 1910) : 
  costPerHandle op = 6/10 := by
  sorry

#eval costPerHandle { fixedCost := 7640, sellingPrice := 46/10, breakEvenQuantity := 1910 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cost_per_handle_is_sixty_cents_l912_91244


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_6789_l912_91287

noncomputable def c : ℝ := 5 + 4 * Real.sqrt 2
noncomputable def d : ℝ := 5 - 4 * Real.sqrt 2

noncomputable def S (n : ℕ) : ℝ := (1 / 2) * (c ^ n + d ^ n)

theorem units_digit_S_6789 : ∃ k : ℤ, S 6789 = 10 * k + 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_units_digit_S_6789_l912_91287


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l912_91279

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (Real.sqrt (Real.sin x)))

-- State the theorem
theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, 0 < x ∧ x < π/2 ∧ f x = y) ↔ y ≤ 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l912_91279


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_theorem_l912_91213

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : EuclideanSpace ℝ (Fin 2))

-- Define the conditions
def is_convex (q : Quadrilateral) : Prop := sorry

def angle_measure (p1 p2 p3 : EuclideanSpace ℝ (Fin 2)) : ℝ := sorry

theorem quadrilateral_angle_theorem (q : Quadrilateral) 
  (h_convex : is_convex q)
  (h_isosceles : dist q.A q.B = dist q.A q.D)
  (h_angle_A : angle_measure q.B q.A q.D = 40)
  (h_angle_C : angle_measure q.B q.C q.D = 130)
  (h_angle_diff : angle_measure q.A q.D q.C - angle_measure q.A q.B q.C = 20) :
  angle_measure q.C q.D q.B = 35 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_theorem_l912_91213


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l912_91215

-- Define the two circles
def circle1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_between_circles :
  ∃ (min_dist : ℝ), 
    (∀ (x1 y1 x2 y2 : ℝ), 
      circle1 x1 y1 → circle2 x2 y2 → 
      distance x1 y1 x2 y2 ≥ min_dist) ∧
    (∃ (x1 y1 x2 y2 : ℝ), 
      circle1 x1 y1 ∧ circle2 x2 y2 ∧ 
      distance x1 y1 x2 y2 = min_dist) ∧
    min_dist = 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l912_91215


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_of_M_l912_91228

def S1 : Finset ℕ := {0, 1, 2, 3, 4}
def S2 : Finset ℕ := {0, 2, 4, 8}

theorem max_cardinality_of_M (M : Finset ℕ) (h1 : M ⊆ S1) (h2 : M ⊆ S2) :
  M.card ≤ 3 ∧ ∃ N : Finset ℕ, N ⊆ S1 ∧ N ⊆ S2 ∧ N.card = 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_of_M_l912_91228


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_coloring_count_l912_91229

/-- Represents a triangle in the configuration --/
inductive Triangle
| S
| T
| U
| V

/-- Represents a coloring of the triangles --/
def Coloring := Triangle → Bool

/-- Checks if two triangles are adjacent --/
def are_adjacent (t1 t2 : Triangle) : Bool :=
  match t1, t2 with
  | Triangle.S, Triangle.V => true
  | Triangle.T, Triangle.V => true
  | Triangle.U, Triangle.V => true
  | Triangle.V, Triangle.S => true
  | Triangle.V, Triangle.T => true
  | Triangle.V, Triangle.U => true
  | _, _ => false

/-- Checks if a coloring is valid (two blue triangles share a side) --/
def is_valid_coloring (c : Coloring) : Bool :=
  let pairs := [(Triangle.S, Triangle.V), (Triangle.T, Triangle.V), (Triangle.U, Triangle.V)]
  pairs.any fun (t1, t2) => c t1 ∧ c t2

/-- Generates all possible colorings --/
def all_colorings : List Coloring :=
  let bools := [true, false]
  [Triangle.S, Triangle.T, Triangle.U, Triangle.V].mapM (fun _ => bools) |>.map fun l =>
    fun t => match t with
    | Triangle.S => l[0]!
    | Triangle.T => l[1]!
    | Triangle.U => l[2]!
    | Triangle.V => l[3]!

/-- Counts the number of valid colorings --/
def count_valid_colorings : Nat :=
  (all_colorings.filter is_valid_coloring).length

theorem valid_coloring_count : count_valid_colorings = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_coloring_count_l912_91229


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_is_square_l912_91236

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- A circle in the plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Represents the configuration of lines and circle as described in the problem -/
structure Configuration :=
  (O : Point)
  (a b c d : Line)
  (circle : Circle)

/-- Represents a tangent line to the circle -/
def Tangent := Line

/-- Represents the quadrilateral formed by the second tangents -/
structure Quadrilateral :=
  (v1 v2 v3 v4 : Point)

/-- States that two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop := sorry

/-- States that a line forms a 45° angle with another line -/
def angle_45 (l1 l2 : Line) : Prop := sorry

/-- States that a line passes through a point -/
def passes_through (l : Line) (p : Point) : Prop := sorry

/-- States that a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop := sorry

/-- States that a quadrilateral is a square -/
def is_square (q : Quadrilateral) : Prop := sorry

/-- Theorem stating that the quadrilateral formed by the second tangents is a square -/
theorem tangent_quadrilateral_is_square (config : Configuration) 
  (h1 : perpendicular config.c config.a)
  (h2 : angle_45 config.b config.a)
  (h3 : angle_45 config.d config.a)
  (h4 : passes_through config.a config.O)
  (h5 : passes_through config.b config.O)
  (h6 : passes_through config.c config.O)
  (h7 : passes_through config.d config.O)
  (t : Tangent)
  (h8 : is_tangent t config.circle) :
  ∃ (q : Quadrilateral), is_square q := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_quadrilateral_is_square_l912_91236


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_stick_length_l912_91265

def IsEven (n : ℝ) : Prop := ∃ k : ℤ, n = 2 * ↑k

theorem third_stick_length (stick1 : ℝ) (stick2 : ℝ) (stick3 : ℝ) : 
  stick1 = 2 → 
  stick2 = 4 → 
  IsEven stick3 →
  stick1 + stick2 > stick3 ∧ stick1 + stick3 > stick2 ∧ stick2 + stick3 > stick1 →
  stick3 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_stick_length_l912_91265


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_power_log_l912_91257

theorem inequalities_power_log (a b c : ℝ) 
  (h1 : a > b) (h2 : b > 1) (h3 : c < 0) : 
  (c / a > c / b) ∧ 
  (a^c < b^c) ∧ 
  (Real.log (a - c) / Real.log b > Real.log (b - c) / Real.log a) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequalities_power_log_l912_91257


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l912_91256

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := Real.sin x ^ 6 - 3 * Real.sin x * Real.cos x + Real.cos x ^ 6

-- State the theorem about the range of g(x)
theorem g_range :
  (∀ x : ℝ, 0 ≤ g x ∧ g x ≤ 7/4) ∧
  (∃ x : ℝ, g x = 0) ∧
  (∃ x : ℝ, g x = 7/4) := by
  sorry

#check g_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_range_l912_91256


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2011_l912_91291

/-- Arithmetic sequence sum -/
noncomputable def arithmetic_sum (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_sum_2011 
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (h1 : a 1 = -2010)  -- First term
  (h2 : S 2009 / 2009 - S 2007 / 2007 = 2)  -- Given condition
  (h3 : ∀ n, S n = arithmetic_sum (a 1) ((a 2) - (a 1)) n)  -- Sum formula
  : S 2011 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_2011_l912_91291


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l912_91294

noncomputable def w : ℝ × ℝ := ⟨-2, 1⟩
def v1 : ℝ × ℝ := ⟨1, 5⟩
def v2 : ℝ × ℝ := ⟨3, 2⟩
def proj_v1 : ℝ × ℝ := ⟨-2, 1⟩

theorem projection_theorem :
  let proj_v2 := (((v2.1 * w.1 + v2.2 * w.2) / (w.1 * w.1 + w.2 * w.2)) : ℝ) • w
  proj_v2 = ⟨8/5, -4/5⟩ := by
  sorry

#check projection_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_theorem_l912_91294


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_field_area_l912_91285

theorem rhombus_field_area :
  ∀ (scale : ℝ) (map_diagonal : ℝ) (angle : ℝ),
    scale = 300 →
    map_diagonal = 6 →
    angle = 60 →
    (1 / 2) * (scale * map_diagonal) * (scale * map_diagonal) * Real.sin (angle * π / 180) = 810000 * Real.sqrt 3 :=
by
  intros scale map_diagonal angle h_scale h_map_diagonal h_angle
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_field_area_l912_91285


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_theorem_l912_91266

/-- Represents a right circular cone water tank -/
structure WaterTank where
  baseRadius : ℝ
  height : ℝ

/-- Calculates the volume of a cone -/
noncomputable def coneVolume (r : ℝ) (h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h

/-- Represents the water level in the tank -/
noncomputable def waterLevel (tank : WaterTank) (fillRatio : ℝ) : ℝ :=
  tank.height * (fillRatio^(1/3))

theorem water_tank_theorem (tank : WaterTank) (h : tank.baseRadius = 20 ∧ tank.height = 60) :
  waterLevel tank 0.5 = 30 * Real.rpow 2 (1/3) := by
  sorry

#check water_tank_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_water_tank_theorem_l912_91266


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_of_m_l912_91252

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := 2^(-abs x) - m

-- State the theorem
theorem intersection_range_of_m :
  (∃ x, f x m = 0) ↔ (0 < m ∧ m ≤ 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_of_m_l912_91252


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_QEFGH_l912_91263

/-- Represents a pyramid with a rectangular base -/
structure RectangularBasePyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ
  perp_to_length : Bool
  perp_to_width : Bool

/-- Calculate the volume of a rectangular base pyramid -/
noncomputable def volume (p : RectangularBasePyramid) : ℝ :=
  (1 / 3) * p.base_length * p.base_width * p.height

/-- The specific pyramid QEFGH -/
def pyramid_QEFGH : RectangularBasePyramid :=
  { base_length := 10
    base_width := 3
    height := 9
    perp_to_length := true
    perp_to_width := true }

theorem volume_of_QEFGH :
  volume pyramid_QEFGH = 90 := by
  -- Unfold the definitions
  unfold volume
  unfold pyramid_QEFGH
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_left_comm]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_of_QEFGH_l912_91263


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_tangent_line_l912_91296

/-- The area of the triangle formed by the tangent line to y = x^3 at (1,1),
    the x-axis, and the line x = 2 is equal to 8/3. -/
theorem triangle_area_tangent_line : 
  let f (x : ℝ) := x^3
  let tangent_point := (1 : ℝ)
  let tangent_line (x : ℝ) := (3 : ℝ) * (x - 1) + 1
  let x_axis := 0
  let vertical_line := 2
  let triangle_area := (1 / 2 : ℝ) * (vertical_line - (tangent_line⁻¹ x_axis)) * (tangent_line vertical_line)
  triangle_area = 8/3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_tangent_line_l912_91296


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l912_91216

/-- The initial amount that, when increased by 1/8th of itself each year for two years, 
    results in a final value of 78468.75 --/
noncomputable def initial_amount : ℚ := 61952

/-- The rate of increase each year --/
def rate : ℚ := 1/8

/-- The final amount after two years of increase --/
noncomputable def final_amount : ℚ := 78468.75

/-- Theorem stating that the initial amount, when increased by the given rate for two years,
    results in the final amount --/
theorem initial_amount_proof : 
  initial_amount * (1 + rate) * (1 + rate) = final_amount := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_amount_proof_l912_91216


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_return_run_probability_first_run_mandatory_recurrence_relation_odd_more_likely_l912_91282

/-- The probability of needing n return runs in the dice rolling game -/
def P (n : ℕ) : ℚ :=
  3/5 * (1 - (-2/3)^n.succ)

/-- The probability that the sum of two fair dice is divisible by 3 -/
def prob_divisible_by_3 : ℚ := 1/3

/-- The probability that the sum of two fair dice is not divisible by 3 -/
def prob_not_divisible_by_3 : ℚ := 2/3

theorem return_run_probability (n : ℕ) :
  P n = 3/5 * (1 - (-2/3)^n.succ) := by sorry

theorem first_run_mandatory :
  P 0 = 1 := by sorry

theorem recurrence_relation (n : ℕ) (h : n ≥ 1) :
  P n = prob_divisible_by_3 * P (n-1) + prob_not_divisible_by_3 * P (n-2) := by sorry

theorem odd_more_likely (n : ℕ) (h : n > 0) :
  P (2*n - 1) > P (2*n) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_return_run_probability_first_run_mandatory_recurrence_relation_odd_more_likely_l912_91282


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_six_times_g_of_seven_l912_91238

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := -1 / x

-- State the theorem
theorem six_times_g_of_seven : g (g (g (g (g (g 7))))) = 7 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_six_times_g_of_seven_l912_91238


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_symmetry_l912_91201

open Real MeasureTheory

theorem sine_function_symmetry (φ : ℝ) :
  φ ∈ Set.Ioo 0 π →
  (∀ x : ℝ, 3 * sin (2 * |x| + π / 3 + φ) = 3 * sin (2 * x + π / 3 + φ)) →
  φ = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_symmetry_l912_91201


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_fifth_power_l912_91269

theorem number_of_divisors_of_fifth_power (n : ℕ) (x : ℕ) :
  n = 2 * 3 * 5 →
  x = n^5 →
  (Finset.card (Finset.filter (λ d => d ∣ x ∧ d > 0) (Finset.range (x + 1)))) = 216 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_divisors_of_fifth_power_l912_91269


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_win_strategy_l912_91200

/-- Represents the modified tic-tac-toe board -/
inductive Board
| board_3x4 : Board
| board_4x3 : Board

/-- Represents a cell on the board -/
structure Cell where
  row : Nat
  col : Nat

/-- Represents a player -/
inductive Player
| X
| O

/-- Represents the state of a cell -/
inductive CellState
| Empty
| Marked (p : Player)

/-- Represents the game state -/
structure GameState where
  board : Board
  cells : List (Cell × CellState)

/-- Defines what constitutes a win -/
def is_win (gs : GameState) (p : Player) : Prop := sorry

/-- Defines optimal play -/
def is_optimal_play (gs : GameState) : Prop := sorry

/-- Gets the center cell of the board -/
def center_cell (b : Board) : Cell :=
  match b with
  | Board.board_3x4 => ⟨2, 2⟩
  | Board.board_4x3 => ⟨2, 2⟩

/-- The main theorem -/
theorem first_player_win_strategy :
  ∀ (b : Board),
  ∃ (final_state : GameState),
    final_state.board = b ∧
    final_state.cells.length > 9 ∧
    (∃ (initial_state : GameState),
      initial_state.board = b ∧
      initial_state.cells.length > 9 ∧
      (∃ (moves : List (Cell × Player)),
        moves.head? = some (center_cell b, Player.X) ∧
        is_optimal_play final_state ∧
        is_win final_state Player.X)) :=
  by
    intro b
    sorry -- Proof skipped for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_player_win_strategy_l912_91200


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_schedule_count_l912_91261

/-- Represents a chess match between two schools -/
structure ChessMatch where
  /-- Number of players in each school -/
  num_players : ℕ
  /-- Number of games played simultaneously in each round -/
  games_per_round : ℕ

/-- Calculates the number of ways to schedule a chess match -/
def schedule_count (m : ChessMatch) : ℕ :=
  sorry

/-- Theorem stating the number of ways to schedule the specific chess match -/
theorem chess_match_schedule_count :
  ∃ (m : ChessMatch),
    m.num_players = 4 ∧
    m.games_per_round = 4 ∧
    schedule_count m = 576 :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chess_match_schedule_count_l912_91261


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_is_37_5_l912_91212

/-- Represents the radius of a circle in the design -/
def radius (n : ℕ) : ℝ := 3 * n.succ

/-- Calculates the area of a circle given its radius -/
noncomputable def area (r : ℝ) : ℝ := Real.pi * r^2

/-- Determines if a circle is black based on its position -/
def is_black (n : ℕ) : Bool := n % 2 = 0

/-- Calculates the total area of the design -/
noncomputable def total_area (n : ℕ) : ℝ := area (radius n)

/-- Calculates the area of black regions in the design -/
noncomputable def black_area (n : ℕ) : ℝ :=
  Finset.sum (Finset.range n.succ) (λ i => if is_black i then area (radius i) - area (radius (i-1)) else 0)

/-- The main theorem stating that the percentage of black area in the design is 37.5% -/
theorem black_percentage_is_37_5 (n : ℕ) : 
  black_area n / total_area n = 3/8 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_black_percentage_is_37_5_l912_91212


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_fee_is_two_dollars_l912_91240

/-- Represents the taxi service pricing structure and a specific trip -/
structure TaxiTrip where
  initialFee : ℚ
  chargePerSegment : ℚ
  segmentLength : ℚ
  tripDistance : ℚ
  totalCharge : ℚ

/-- Calculates the number of segments in a trip -/
def numSegments (trip : TaxiTrip) : ℚ :=
  trip.tripDistance / trip.segmentLength

/-- Calculates the charge for the distance traveled -/
def distanceCharge (trip : TaxiTrip) : ℚ :=
  trip.chargePerSegment * numSegments trip

/-- Theorem stating that the initial fee is $2.00 -/
theorem initial_fee_is_two_dollars
  (trip : TaxiTrip)
  (h1 : trip.chargePerSegment = 35/100)
  (h2 : trip.segmentLength = 2/5)
  (h3 : trip.tripDistance = 36/10)
  (h4 : trip.totalCharge = 515/100)
  (h5 : trip.totalCharge = trip.initialFee + distanceCharge trip) :
  trip.initialFee = 2 := by
  sorry

#eval toString (2 : ℚ)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_fee_is_two_dollars_l912_91240


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l912_91251

theorem triangle_side_range (a : ℝ) : 
  (∃ (x y z : ℝ), x = 3 ∧ y = 1 - 2*a ∧ z = 8 ∧ 
   x + y > z ∧ y + z > x ∧ z + x > y) ↔ -5 < a ∧ a < -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_range_l912_91251


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_sequence_area_l912_91210

/-- Represents a rectangle with a given width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the diagonal length of a rectangle using the Pythagorean theorem -/
noncomputable def Rectangle.diagonal (r : Rectangle) : ℝ := Real.sqrt (r.width^2 + r.height^2)

/-- Represents a sequence of rectangles with the given properties -/
structure RectangleSequence where
  r₀ : Rectangle
  r₁ : Rectangle
  r₂ : Rectangle
  r₃ : Rectangle

/-- The theorem to be proved -/
theorem rectangle_sequence_area (rs : RectangleSequence) : 
  rs.r₀.width = 3 ∧ rs.r₀.height = 4 →
  (∀ n : Fin 3, (rs.r₁.width = rs.r₀.diagonal ∨ rs.r₁.height = rs.r₀.diagonal) ∧
                (rs.r₂.width = rs.r₁.diagonal ∨ rs.r₂.height = rs.r₁.diagonal) ∧
                (rs.r₃.width = rs.r₂.diagonal ∨ rs.r₃.height = rs.r₂.diagonal)) →
  rs.r₀.area + rs.r₁.area + rs.r₂.area + rs.r₃.area = 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_sequence_area_l912_91210


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l912_91202

noncomputable def a (n : ℕ+) : ℝ := 6 * n - 5

noncomputable def T (n : ℕ+) : ℝ := (1 / 2) * (1 - 1 / (6 * n + 1))

theorem sequence_properties :
  (∀ n : ℕ+, ∃ k : ℝ, a (n + 1) - a n = k) ∧
  (∀ m : ℕ+, (m ≥ 10 ↔ ∀ n : ℕ+, T n < m / 20) ∧
             (m < 10 → ∃ n : ℕ+, T n ≥ m / 20)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l912_91202


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_theorem_l912_91243

open BigOperators

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def product (n : ℕ) : ℚ :=
  ∏ k in Finset.range (n - 2), ((fibonacci (k + 3) : ℚ) / fibonacci (k + 2) - (fibonacci (k + 3) : ℚ) / fibonacci (k + 4))

theorem fibonacci_product_theorem : product 50 = (fibonacci 50 : ℚ) / fibonacci 51 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fibonacci_product_theorem_l912_91243


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_decrease_rate_l912_91242

/-- Given an initial population and the population after two years,
    calculate the annual decrease rate. -/
noncomputable def annual_decrease_rate (initial_population : ℝ) (population_after_two_years : ℝ) : ℝ :=
  100 * (1 - (population_after_two_years / initial_population) ^ (1/2 : ℝ))

/-- Theorem stating that for the given population values, 
    the annual decrease rate is 10%. -/
theorem village_population_decrease_rate :
  let initial_population := (6000 : ℝ)
  let population_after_two_years := (4860 : ℝ)
  annual_decrease_rate initial_population population_after_two_years = 10 := by
  sorry

-- Remove the #eval statement as it's not compatible with noncomputable definitions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_population_decrease_rate_l912_91242


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l912_91221

noncomputable def reflection_matrix : Matrix (Fin 2) (Fin 2) ℝ := !![3/5, 4/5; 4/5, -3/5]

noncomputable def direction_vector : Fin 2 → ℝ := ![2, 1]

theorem direction_vector_proof :
  -- The vector is an eigenvector of the reflection matrix with eigenvalue 1
  reflection_matrix.mulVec direction_vector = direction_vector ∧
  -- The vector has positive first component
  direction_vector 0 > 0 ∧
  -- The components are integers
  ∀ i, ∃ n : ℤ, direction_vector i = n ∧
  -- The GCD of the absolute values of the components is 1
  Int.gcd (Int.natAbs (Int.floor (direction_vector 0))) 
          (Int.natAbs (Int.floor (direction_vector 1))) = 1 := by
  sorry

#check direction_vector_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_direction_vector_proof_l912_91221


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_union_N_eq_M_l912_91223

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | |p.1 * p.2| = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.arctan p.1 + Real.arctan (1 / p.2) = Real.pi}

-- State the theorem
theorem M_union_N_eq_M : M ∪ N = M := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_union_N_eq_M_l912_91223


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l912_91239

open Real

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  t.A + t.B + t.C = Real.pi ∧ t.A > 0 ∧ t.B > 0 ∧ t.C > 0

def perimeter_is_one (t : Triangle) : Prop :=
  Real.sin t.A + Real.sin t.B + Real.sin t.C = 1

def angle_condition (t : Triangle) : Prop :=
  Real.sin (2 * t.A) + Real.sin (2 * t.B) = 4 * Real.sin t.A * Real.sin t.B

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : is_valid_triangle t) 
  (h2 : perimeter_is_one t) 
  (h3 : angle_condition t) : 
  (t.C = Real.pi / 2) ∧ 
  (∀ (s : Triangle), is_valid_triangle s → perimeter_is_one s → 
    (1/2 * Real.sin s.A * Real.sin s.B ≤ (3 - 2 * Real.sqrt 2) / 4)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l912_91239


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_factorial_fraction_l912_91220

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 126) = 12 * Real.sqrt 30 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_of_factorial_fraction_l912_91220


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derangements_of_four_l912_91298

/-- The number of derangements of n elements -/
def myNumDerangements (n : ℕ) : ℕ := sorry

/-- Theorem: The number of derangements of 4 elements is 9 -/
theorem derangements_of_four : myNumDerangements 4 = 9 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derangements_of_four_l912_91298


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_G_function_h_is_G_function_iff_l912_91227

-- Define the G-function property
def is_G_function (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≥ 0 → x₂ ≥ 0 → x₁ + x₂ ≤ 1 → f (x₁ + x₂) ≥ f x₁ + f x₂)

-- Define the functions g and h
def g (x : ℝ) : ℝ := x^2

noncomputable def h (b : ℝ) (x : ℝ) : ℝ := Real.exp (x * Real.log 2) - b

-- Theorem statements
theorem g_is_G_function : is_G_function g := by sorry

theorem h_is_G_function_iff (b : ℝ) : is_G_function (h b) ↔ b = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_G_function_h_is_G_function_iff_l912_91227


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_of_factorial_l912_91249

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_sum_of_factors_of_factorial : 
  ∀ x y z : ℕ+, 
    (x : ℕ) * y * z = factorial 8 → 
    (∀ a b c : ℕ+, (a : ℕ) * b * c = factorial 8 → x + y + z ≤ a + b + c) → 
    x + y + z = 103 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_factors_of_factorial_l912_91249


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_candies_order_independent_l912_91264

-- Define the gender of a child
inductive Gender
| Boy
| Girl

-- Define a child with their gender
structure Child where
  gender : Gender

-- Define the state of the candy distribution
structure CandyState where
  totalCandies : ℕ
  remainingChildren : ℕ
  boysCandies : ℕ

def distributeCandy (state : CandyState) (child : Child) : CandyState :=
  match child.gender with
  | Gender.Boy =>
      let candies := (state.totalCandies + state.remainingChildren - 1) / state.remainingChildren
      { totalCandies := state.totalCandies - candies,
        remainingChildren := state.remainingChildren - 1,
        boysCandies := state.boysCandies + candies }
  | Gender.Girl =>
      let candies := state.totalCandies / state.remainingChildren
      { totalCandies := state.totalCandies - candies,
        remainingChildren := state.remainingChildren - 1,
        boysCandies := state.boysCandies }

-- Theorem statement
theorem boys_candies_order_independent
  (initialCandies : ℕ)
  (children : List Child) :
  ∀ (perm : List Child),
    List.Perm children perm →
    (List.foldl distributeCandy
      { totalCandies := initialCandies,
        remainingChildren := children.length,
        boysCandies := 0 }
      children).boysCandies =
    (List.foldl distributeCandy
      { totalCandies := initialCandies,
        remainingChildren := perm.length,
        boysCandies := 0 }
      perm).boysCandies :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_boys_candies_order_independent_l912_91264


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_calculation_l912_91211

/-- Calculates the time for a round trip in a stream given the current speed, boat speed, and distance. -/
noncomputable def roundTripTime (streamSpeed boatSpeed distance : ℝ) : ℝ :=
  let downstreamSpeed := boatSpeed + streamSpeed
  let upstreamSpeed := boatSpeed - streamSpeed
  (distance / downstreamSpeed) + (distance / upstreamSpeed)

theorem round_trip_time_calculation :
  let streamSpeed : ℝ := 4
  let boatSpeed : ℝ := 8
  let distance : ℝ := 6
  roundTripTime streamSpeed boatSpeed distance = 2 := by
  -- Unfold the definition of roundTripTime
  unfold roundTripTime
  -- Simplify the expression
  simp
  -- The proof is completed with numerical computation
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_round_trip_time_calculation_l912_91211


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l912_91246

noncomputable def m (α : ℝ) : ℝ × ℝ := (Real.cos α, 1 - Real.sin α)
noncomputable def n (α : ℝ) : ℝ × ℝ := (-Real.cos α, Real.sin α)

def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def vector_sub (v w : ℝ × ℝ) : ℝ × ℝ :=
  (v.1 - w.1, v.2 - w.2)

noncomputable def vector_norm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2)

theorem vector_problem (α : ℝ) :
  (perpendicular (m α) (n α) → ∃ k : ℤ, α = 2 * k * Real.pi + Real.pi / 2) ∧
  (vector_norm (vector_sub (m α) (n α)) = Real.sqrt 3 → Real.cos (2 * α) = 1 / 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_problem_l912_91246


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_and_max_distance_l912_91254

-- Define the real parameter p
variable (p : ℝ)

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x - p * y + p - 2 = 0
def l₂ (x y : ℝ) : Prop := p * x + y + 2 * p - 4 = 0

-- Define the fixed points A and B
def A : ℝ × ℝ := (2, 1)
def B : ℝ × ℝ := (-2, 4)

-- Define the intersection point M
noncomputable def M (p : ℝ) : ℝ × ℝ := sorry

-- State the theorem
theorem lines_perpendicular_and_max_distance (p : ℝ) :
  (∀ x y : ℝ, l₁ p x y → l₂ p x y → (x - 2) * (p * x + y + 2 * p - 4) + (y - 1) * (x - p * y + p - 2) = 0) ∧
  (∃ C : ℝ, ∀ θ : ℝ, C * (Real.cos θ + 2 * Real.sin θ) ≤ 5 * Real.sqrt 5 ∧
    ∃ θ₀ : ℝ, C * (Real.cos θ₀ + 2 * Real.sin θ₀) = 5 * Real.sqrt 5) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_perpendicular_and_max_distance_l912_91254


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_rotated_line_l912_91225

-- Define the original line ℓ
def line_ℓ (x y : ℝ) : Prop := 4 * x - 3 * y + 60 = 0

-- Define the rotation angle
noncomputable def rotation_angle : ℝ := 30 * (Real.pi / 180)

-- Define the rotation center
def rotation_center : ℝ × ℝ := (10, 10)

-- Define the rotated line k
def line_k (x y : ℝ) : Prop :=
  ∃ (x₀ y₀ : ℝ),
    line_ℓ x₀ y₀ ∧
    (x - rotation_center.1) = (x₀ - rotation_center.1) * Real.cos rotation_angle - (y₀ - rotation_center.2) * Real.sin rotation_angle ∧
    (y - rotation_center.2) = (x₀ - rotation_center.1) * Real.sin rotation_angle + (y₀ - rotation_center.2) * Real.cos rotation_angle

-- Define the x-intercept of line k
noncomputable def x_intercept (k : ℝ → ℝ → Prop) : ℝ :=
  Classical.choose (sorry : ∃ x, k x 0)

-- Theorem statement
theorem x_intercept_of_rotated_line :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.1 ∧ |x_intercept line_k - 8.5| < ε :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_of_rotated_line_l912_91225


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_EFGH_to_ABCD_l912_91277

-- Define the square ABCD
noncomputable def Square (s : ℝ) : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ s ∧ 0 ≤ p.2 ∧ p.2 ≤ s}

-- Define the midpoints E, F, G, H
noncomputable def E (s : ℝ) : ℝ × ℝ := (s/2, -s/2)
noncomputable def F (s : ℝ) : ℝ × ℝ := (3*s/2, s/2)
noncomputable def G (s : ℝ) : ℝ × ℝ := (s/2, 3*s/2)
noncomputable def H (s : ℝ) : ℝ × ℝ := (-s/2, s/2)

-- Define the square EFGH
noncomputable def SquareEFGH (s : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
    (p = ((1-t) * (E s).1 + t * (F s).1, (1-t) * (E s).2 + t * (F s).2) ∨
     p = ((1-t) * (F s).1 + t * (G s).1, (1-t) * (F s).2 + t * (G s).2) ∨
     p = ((1-t) * (G s).1 + t * (H s).1, (1-t) * (G s).2 + t * (H s).2) ∨
     p = ((1-t) * (H s).1 + t * (E s).1, (1-t) * (H s).2 + t * (E s).2))}

-- Define the areas of the squares
noncomputable def AreaABCD (s : ℝ) : ℝ := s^2
noncomputable def AreaEFGH (s : ℝ) : ℝ := ((F s).1 - (E s).1)^2 + ((F s).2 - (E s).2)^2

-- State the theorem
theorem area_ratio_EFGH_to_ABCD (s : ℝ) (h : s > 0) :
  AreaEFGH s / AreaABCD s = 5/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_EFGH_to_ABCD_l912_91277


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_distance_relation_l912_91207

open Real

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculate the squared distance between two points -/
noncomputable def squaredDistance (p q : Point) : ℝ :=
  (p.x - q.x)^2 + (p.y - q.y)^2

/-- Calculate the center of mass of a quadrilateral -/
noncomputable def centerOfMass (a b c d : Point) : Point :=
  { x := (a.x + b.x + c.x + d.x) / 4,
    y := (a.y + b.y + c.y + d.y) / 4 }

/-- Theorem: For any quadrilateral ABCD with center of mass M and any point Q,
    QA^2 + QB^2 + QC^2 + QD^2 = 4 * QM^2 + MA^2 + MB^2 + MC^2 + MD^2 -/
theorem quadrilateral_distance_relation (a b c d q : Point) :
  let m := centerOfMass a b c d
  squaredDistance q a + squaredDistance q b + squaredDistance q c + squaredDistance q d =
  4 * squaredDistance q m + squaredDistance m a + squaredDistance m b +
  squaredDistance m c + squaredDistance m d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_distance_relation_l912_91207


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_lines_l912_91241

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = x

-- Define the circle
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a line being tangent to the circle
def tangent_to_circle (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (2 + y₁ * y₂)^2 = (1 + (y₁ + y₂)^2) * ((x₁ - 2)^2 + y₁^2)

-- Main theorem
theorem parabola_tangent_lines (A₁ A₂ A₃ : ℝ × ℝ) :
  let (x₁, y₁) := A₁
  let (x₂, y₂) := A₂
  let (x₃, y₃) := A₃
  parabola x₁ y₁ ∧ parabola x₂ y₂ ∧ parabola x₃ y₃ →
  tangent_to_circle x₁ y₁ x₂ y₂ →
  tangent_to_circle x₁ y₁ x₃ y₃ →
  tangent_to_circle x₂ y₂ x₃ y₃ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_lines_l912_91241


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_y_equals_neg_five_l912_91270

noncomputable def x : ℂ := (2 - Complex.I) / (3 + Complex.I)

def matrix_op (a b c d : ℂ) : ℂ := a * d - b * c

noncomputable def y : ℂ := matrix_op (4 * Complex.I) (3 - x * Complex.I) (1 + Complex.I) (x + Complex.I)

theorem y_equals_neg_five : y = -5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_y_equals_neg_five_l912_91270


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logical_implication_l912_91274

theorem logical_implication
  (x y z w : ℝ)
  (u v : Type)
  (cond1 : x < y → z > w)
  (cond2 : z ≤ w → u ≠ v) :
  x ≥ y → u ≠ v :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logical_implication_l912_91274


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_concatenated_number_not_square_l912_91208

/-- The concatenation of all five-digit numbers from 11111 to 99999 in any order -/
def concatenated_number : ℕ := sorry

/-- The concatenated number has 444445 digits -/
axiom concatenated_number_digits : (Nat.digits 10 concatenated_number).length = 444445

/-- The concatenated number is congruent to 2 modulo 3 -/
axiom concatenated_number_mod_3 : concatenated_number % 3 = 2

/-- Theorem: The concatenation of all five-digit numbers from 11111 to 99999 in any order cannot be a square number -/
theorem concatenated_number_not_square : ¬ ∃ n : ℕ, n ^ 2 = concatenated_number := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_concatenated_number_not_square_l912_91208


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_sharing_l912_91203

/-- Represents the amount LeRoy must pay to achieve equal cost sharing -/
noncomputable def amountToPay (A B C : ℝ) : ℝ := (B + C - 2 * A) / 3

theorem equal_cost_sharing (A B C : ℝ) (h1 : A < B) (h2 : B < C) :
  let totalCost := A + B + C
  let equalShare := totalCost / 3
  equalShare - A = amountToPay A B C := by
  sorry

#check equal_cost_sharing

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_cost_sharing_l912_91203


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_theta_in_range_l912_91232

open Real

theorem inequality_holds_iff_theta_in_range :
  ∀ k : ℤ, ∀ θ : ℝ, 
    (∀ x : ℝ, x ∈ Set.Icc 0 1 → x^2 * cos θ - x*(1-x) + (1-x)^2 * sin θ > 0) ↔ 
    (2*π*k + π/12 < θ ∧ θ < 2*π*k + 5*π/12) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_iff_theta_in_range_l912_91232


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_perimeter_inequality_other_diagonal_not_one_other_diagonal_can_be_two_other_diagonal_can_be_1001_l912_91280

/-- A convex quadrilateral with perimeter 2004 and one diagonal of length 1001 -/
structure ConvexQuadrilateral where
  perimeter : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ
  perimeter_eq : perimeter = 2004
  diagonal1_eq : diagonal1 = 1001
  convex : True  -- We'll use True here as a placeholder for convexity

/-- Theorem about the relationship between diagonals and perimeter of a quadrilateral -/
theorem diagonal_perimeter_inequality (q : ConvexQuadrilateral) :
  2 * (q.diagonal1 + q.diagonal2) > q.perimeter := by sorry

/-- The other diagonal cannot be 1 -/
theorem other_diagonal_not_one (q : ConvexQuadrilateral) :
  q.diagonal2 ≠ 1 := by sorry

/-- The other diagonal can be 2 -/
theorem other_diagonal_can_be_two :
  ∃ q : ConvexQuadrilateral, q.diagonal2 = 2 := by sorry

/-- The other diagonal can be 1001 -/
theorem other_diagonal_can_be_1001 :
  ∃ q : ConvexQuadrilateral, q.diagonal2 = 1001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_perimeter_inequality_other_diagonal_not_one_other_diagonal_can_be_two_other_diagonal_can_be_1001_l912_91280


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_e_power_x_l912_91275

theorem integral_e_power_x : ∫ x in (0:ℝ)..(1:ℝ), Real.exp x = Real.exp 1 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_e_power_x_l912_91275


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l912_91237

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 5)

-- State the theorem
theorem domain_of_f :
  ∀ x : ℝ, f x ≠ 0 ↔ x ≠ 5 :=
by
  intro x
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l912_91237


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_orientation_l912_91289

noncomputable def parabola1 (x : ℝ) : ℝ := -x^2 - x + 3
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 + x + 1

noncomputable def vertex1 : ℝ × ℝ := (1/2, 13/4)
noncomputable def vertex2 : ℝ × ℝ := (-1/2, 3/4)

theorem parabola_orientation :
  vertex1.1 > vertex2.1 ∧ vertex1.2 > vertex2.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_orientation_l912_91289


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l912_91262

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -1/3 * x^3 + 2*a*x^2 - 3*a^2*x

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := -x^2 + 4*a*x - 3*a^2

theorem function_properties (a : ℝ) (h : a ≠ 0) :
  -- Part 1: Tangent line equation when a = -1
  (a = -1 → ∃ (m b : ℝ), ∀ x y, y = f (-1) x → (x = -2 ∧ y = f (-1) (-2)) → 3*x - 3*y + 8 = 0) ∧
  
  -- Part 2: Extreme values when a > 0
  (a > 0 → 
    (∃ x, f a x = 0) ∧ 
    (∀ x, f a x ≤ 0) ∧
    (∃ x, f a x = -4/3 * a^3) ∧
    (∀ x, f a x ≥ -4/3 * a^3)) ∧
  
  -- Part 3: Range of a when |f'(x)| ≤ 3a for x ∈ [2a, 2a+2]
  (∀ x, x ≥ 2*a ∧ x ≤ 2*a + 2 → |f_derivative a x| ≤ 3*a → 1 ≤ a ∧ a ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l912_91262


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_miles_to_km_conversion_l912_91226

/-- Conversion factor between miles and kilometers -/
noncomputable def conversion_factor (miles_per_minute : ℝ) (km_per_hour : ℝ) : ℝ :=
  (miles_per_minute * 60) / km_per_hour

/-- Theorem: Given a driving rate of 6 miles per minute and a speed of 600 kilometers per hour,
    the conversion factor between miles and kilometers is 0.6 miles per kilometer. -/
theorem miles_to_km_conversion (miles_per_minute km_per_hour : ℝ) 
  (h1 : miles_per_minute = 6)
  (h2 : km_per_hour = 600) :
  conversion_factor miles_per_minute km_per_hour = 0.6 := by
  -- Unfold the definition of conversion_factor
  unfold conversion_factor
  -- Substitute the given values
  rw [h1, h2]
  -- Perform the calculation
  norm_num
  -- The proof is complete
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_miles_to_km_conversion_l912_91226


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_term_at_8_l912_91217

noncomputable def a (n : ℕ+) : ℝ := (n - 7) / (n - 5 * Real.sqrt 2)

def is_maximum (m : ℕ+) : Prop :=
  ∀ n : ℕ+, a n ≤ a m

theorem max_term_at_8 : is_maximum 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_term_at_8_l912_91217


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l912_91205

/-- The parabola function -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * (x - 2)^2 - 1

/-- Theorem: The point (0, 1) lies on both the parabola and the y-axis -/
theorem intersection_point :
  f 0 = 1 ∧ 0 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_l912_91205


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_point_l912_91218

def f (x : ℝ) := x^3 - x

theorem tangent_line_point : 
  ∃ a b : ℝ,
  (∀ x, 2*x - f x = 2 → x = a) ∧
  f a = b ∧
  (3*a^2 - 1) = 2 ∧
  a = 1 ∧ b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_point_l912_91218


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l912_91259

/-- The function f(x) = e^x - x -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x - x

/-- The theorem stating that the maximum value of f(x) on [-1, 1] is e - 1 -/
theorem max_value_f_on_interval : 
  ∃ (c : ℝ), c ∈ Set.Icc (-1 : ℝ) 1 ∧ 
    ∀ (x : ℝ), x ∈ Set.Icc (-1 : ℝ) 1 → f x ≤ f c ∧ 
    f c = Real.exp 1 - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_f_on_interval_l912_91259


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_difference_sum_l912_91284

theorem product_difference_sum (a b : ℕ) : 
  a > 0 → b > 0 → a * b = 24 → |Int.ofNat a - Int.ofNat b| = 2 → a + b = 10 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_difference_sum_l912_91284


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_one_twelfth_l912_91258

-- Define the functions for the curves
def f (x : ℝ) := x^2
def g (x : ℝ) := x^3

-- Define the area of the enclosed shape
noncomputable def enclosed_area : ℝ := ∫ x in (0)..(1), (f x - g x)

-- Theorem statement
theorem enclosed_area_is_one_twelfth : enclosed_area = 1/12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_enclosed_area_is_one_twelfth_l912_91258


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_seven_l912_91281

def mySequence (n : ℕ) : ℤ :=
  if n % 2 = 0 then -2 - 10 * (n / 2) else 8 + 10 * (n / 2)

def myProduct : ℤ := (Finset.range 20).prod (λ i => mySequence i)

theorem product_remainder_mod_seven : myProduct % 7 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_remainder_mod_seven_l912_91281


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_to_linear_l912_91231

/-- For a quadratic equation (k-3)x^2 + 2x - 3 = 0 to be linear in x, k must equal 3 -/
theorem quadratic_to_linear (k : ℝ) : 
  (∀ x, (k - 3) * x^2 + 2 * x - 3 = 0 → (k - 3) * x^2 = 0) ↔ k = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_to_linear_l912_91231


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_proposition_l912_91293

theorem negation_of_cosine_proposition :
  (¬ ∀ x : ℝ, Real.cos x > 1) ↔ (∃ x : ℝ, Real.cos x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_cosine_proposition_l912_91293


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_range_minimum_value_l912_91273

-- Define the functions
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := Real.exp x + a * x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x * Real.exp (a * x - 1) - 2 * a * x + f a x

-- Define the theorem for the first part
theorem monotonicity_range (a : ℝ) :
  (∀ x, x > 0 → x < Real.log 3 → (∀ y, y > 0 → y < Real.log 3 → 
    (f a x < f a y ↔ F a x < F a y))) →
  a ≤ -3 := by
  sorry

-- Define the theorem for the second part
theorem minimum_value (a : ℝ) (φ : ℝ → ℝ) :
  a ≤ -(Real.exp 2)⁻¹ →
  (∃ m, ∀ x, x > 0 → g a x ≥ m) →
  (∃ x, x > 0 ∧ g a x = φ a) →
  φ a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_range_minimum_value_l912_91273


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_determines_magnitude_l912_91233

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable def min_value (a b : E) : ℝ :=
  Real.sqrt (‖a‖^2 - (inner a b)^2 / ‖b‖^2)

theorem vector_angle_determines_magnitude (a b : E) (θ : ℝ) :
  a ≠ 0 → b ≠ 0 → 
  θ = Real.arccos (inner a b / (‖a‖ * ‖b‖)) →
  min_value a b = 1 →
  ∃! x : ℝ, x = ‖a‖ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_determines_magnitude_l912_91233


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_max_value_when_a_is_two_nonnegative_condition_l912_91272

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 - a * Real.sin x

-- Define the interval [0, π]
def I : Set ℝ := Set.Icc 0 Real.pi

-- Statement 1
theorem tangent_line_implies_a_value (a : ℝ) :
  (∀ h : ℝ, h ≠ 0 → (f a h - f a 0) / h = -1) → a = 2 := by sorry

-- Statement 2
theorem max_value_when_a_is_two :
  ∃ x ∈ I, ∀ y ∈ I, f 2 y ≤ f 2 x ∧ f 2 x = Real.exp Real.pi - 1 := by sorry

-- Statement 3
theorem nonnegative_condition (a : ℝ) :
  (∀ x ∈ I, f a x ≥ 0) ↔ a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_value_max_value_when_a_is_two_nonnegative_condition_l912_91272


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_platform_length_is_250_l912_91235

/-- Given a train and two platforms, calculates the length of the second platform. -/
noncomputable def second_platform_length (train_length : ℝ) (first_platform_length : ℝ) 
  (time_first_platform : ℝ) (time_second_platform : ℝ) : ℝ :=
  let train_speed := (train_length + first_platform_length) / time_first_platform
  train_speed * time_second_platform - train_length

/-- Theorem stating that under the given conditions, the length of the second platform is 250 m. -/
theorem second_platform_length_is_250 :
  second_platform_length 70 170 15 20 = 250 := by
  unfold second_platform_length
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_platform_length_is_250_l912_91235


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_IJKL_max_area_IJKL_proof_l912_91206

theorem max_area_IJKL (area_ABCD area_EFGH : ℕ) (IJKL_exists : Prop) : Prop :=
  area_ABCD = 2016 →
  0 < area_EFGH →
  area_EFGH < area_ABCD →
  IJKL_exists →
  ∃ (area_IJKL : ℕ), (area_IJKL = 1008 ∧ 
    ∀ (other_area : ℕ), (other_area ≤ area_ABCD ∧ other_area ≥ area_EFGH) → other_area ≤ area_IJKL)

-- The proof would go here
theorem max_area_IJKL_proof (area_ABCD area_EFGH : ℕ) (IJKL_exists : Prop) :
  max_area_IJKL area_ABCD area_EFGH IJKL_exists := by
  sorry

#check max_area_IJKL
#check max_area_IJKL_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_IJKL_max_area_IJKL_proof_l912_91206


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bishops_on_8x8_chessboard_l912_91204

/-- Represents a chessboard --/
structure Chessboard :=
  (size : Nat)

/-- Represents a bishop placement on a chessboard --/
structure BishopPlacement :=
  (board : Chessboard)
  (count : Nat)

/-- Function to count bishops on a diagonal (definition omitted) --/
def count_bishops_on_diagonal (diagonal : Nat) (placement : BishopPlacement) : Nat :=
  sorry

/-- Predicate to check if a bishop placement is valid --/
def is_valid_placement (placement : BishopPlacement) : Prop :=
  ∀ d, count_bishops_on_diagonal d placement ≤ 3

/-- The maximum number of bishops that can be placed on a diagonal --/
def max_bishops_per_diagonal : Nat := 3

/-- Theorem stating the maximum number of bishops on an 8x8 chessboard --/
theorem max_bishops_on_8x8_chessboard :
  ∃ (placement : BishopPlacement),
    placement.board.size = 8 ∧
    is_valid_placement placement ∧
    placement.count = 38 ∧
    (∀ (other_placement : BishopPlacement),
      other_placement.board.size = 8 →
      is_valid_placement other_placement →
      other_placement.count ≤ placement.count) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bishops_on_8x8_chessboard_l912_91204


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nonSquareSequence_2003_l912_91224

/-- A function that returns true if a number is a perfect square, false otherwise -/
def isPerfectSquare (n : ℕ) : Bool :=
  match Nat.sqrt n with
  | m => m * m = n

/-- The sequence of positive integers with perfect squares removed -/
def nonSquareSequence : ℕ → ℕ
  | 0 => 0
  | n + 1 => if isPerfectSquare (n + 1) then nonSquareSequence n else n + 1

/-- The 2003rd term of the sequence is 2048 -/
theorem nonSquareSequence_2003 : nonSquareSequence 2003 = 2048 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nonSquareSequence_2003_l912_91224


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_and_area_l912_91255

/-- Cone properties -/
structure Cone where
  diameter : ℝ
  height : ℝ

/-- Volume of a cone -/
noncomputable def volume (c : Cone) : ℝ := (1/3) * Real.pi * (c.diameter/2)^2 * c.height

/-- Lateral surface area of a cone -/
noncomputable def lateralSurfaceArea (c : Cone) : ℝ :=
  Real.pi * (c.diameter/2) * (((c.diameter/2)^2 + c.height^2).sqrt)

/-- Theorem: Volume and lateral surface area of a specific cone -/
theorem cone_volume_and_area (c : Cone) 
  (h_diameter : c.diameter = 12)
  (h_height : c.height = 9) :
  volume c = 108 * Real.pi ∧ lateralSurfaceArea c = 6 * Real.pi * Real.sqrt 117 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_volume_and_area_l912_91255


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_sum_24hour_clock_l912_91286

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Fin 24
  minutes : Fin 60

/-- Calculates the sum of digits for a given number -/
def sumDigits (n : Nat) : Nat :=
  (n.repr.toList.map (fun c => c.toString.toNat!)).sum

/-- Calculates the sum of digits for a given time -/
def timeDigitSum (t : Time24) : Nat :=
  sumDigits t.hours.val + sumDigits t.minutes.val

/-- The maximum sum of digits on a 24-hour format digital clock -/
theorem max_digit_sum_24hour_clock :
  (⨆ t : Time24, timeDigitSum t) = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_digit_sum_24hour_clock_l912_91286


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bathtub_max_volume_l912_91247

/-- A half-cylindrical bathtub with surface area S has maximum volume when its radius r and length l satisfy specific relationships with S. -/
theorem bathtub_max_volume (S : ℝ) (h : S > 0) :
  let r := Real.sqrt (S / (3 * Real.pi))
  let l := 2 * r
  let v := fun r' l' ↦ (1/2) * Real.pi * r' ^ 2 * l'
  let surface_area := fun r' l' ↦ Real.pi * r' ^ 2 + Real.pi * r' * l'
  ∀ r' l', r' > 0 → l' > 0 → surface_area r' l' = S →
    v r' l' ≤ v r l := by
  sorry

#check bathtub_max_volume

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bathtub_max_volume_l912_91247


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circles_tangent_length_l912_91292

-- Define the rectangle ABCD
structure Rectangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  is_rectangle : 
    (A.1 = D.1 ∧ B.1 = C.1 ∧ A.2 = B.2 ∧ D.2 = C.2) ∧
    (B.1 - A.1)^2 + (B.2 - A.2)^2 = (C.1 - D.1)^2 + (C.2 - D.2)^2

-- Define circles ω₁ and ω₂
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the concept of a tangent line
def IsTangentLine (ℓ : Set (ℝ × ℝ)) (ω : Set (ℝ × ℝ)) : Prop :=
  ∃ (p : ℝ × ℝ), p ∈ ℓ ∧ p ∈ ω ∧ ∀ (q : ℝ × ℝ), q ∈ ℓ ∧ q ∈ ω → q = p

-- Define the common internal tangent
def CommonInternalTangent (ω₁ ω₂ : Set (ℝ × ℝ)) (E F : ℝ × ℝ) : Prop :=
  ∃ (ℓ : Set (ℝ × ℝ)), IsTangentLine ℓ ω₁ ∧ IsTangentLine ℓ ω₂ ∧ E ∈ ℓ ∧ F ∈ ℓ

-- The main theorem
theorem rectangle_circles_tangent_length
  (ABCD : Rectangle)
  (h_AB : Real.sqrt ((ABCD.B.1 - ABCD.A.1)^2 + (ABCD.B.2 - ABCD.A.2)^2) = 10)
  (h_BC : Real.sqrt ((ABCD.C.1 - ABCD.B.1)^2 + (ABCD.C.2 - ABCD.B.2)^2) = 26)
  (ω₁ : Set (ℝ × ℝ))
  (h_ω₁ : ω₁ = Circle ((ABCD.A.1 + ABCD.B.1) / 2, (ABCD.A.2 + ABCD.B.2) / 2) 5)
  (ω₂ : Set (ℝ × ℝ))
  (h_ω₂ : ω₂ = Circle ((ABCD.C.1 + ABCD.D.1) / 2, (ABCD.C.2 + ABCD.D.2) / 2) 5)
  (E : ℝ × ℝ)
  (F : ℝ × ℝ)
  (h_tangent : CommonInternalTangent ω₁ ω₂ E F)
  (h_E_on_AD : E.1 = ABCD.A.1 ∧ E.2 ∈ Set.Icc ABCD.A.2 ABCD.D.2)
  (h_F_on_BC : F.1 = ABCD.B.1 ∧ F.2 ∈ Set.Icc ABCD.B.2 ABCD.C.2) :
  Real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2) = 24 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_circles_tangent_length_l912_91292


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAB_is_sqrt_three_l912_91268

/-- Definition of the ellipse C -/
noncomputable def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Definition of a companion point -/
noncomputable def companion_point (x y : ℝ) : ℝ × ℝ := (x/2, y/(Real.sqrt 3))

/-- A line intersects the ellipse at two points -/
def intersects_ellipse (l : ℝ → ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, A ≠ B ∧ ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧
    (∀ t : ℝ, (t, l t) = A ∨ (t, l t) = B)

/-- Circle with diameter PQ passes through origin -/
def circle_through_origin (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

/-- Area of a triangle given three points -/
noncomputable def triangle_area (O A B : ℝ × ℝ) : ℝ :=
  abs ((A.1 - O.1) * (B.2 - O.2) - (B.1 - O.1) * (A.2 - O.2)) / 2

/-- Main theorem -/
theorem area_of_triangle_OAB_is_sqrt_three (l : ℝ → ℝ) :
  intersects_ellipse l →
  (∀ A B : ℝ × ℝ, A ≠ B → ellipse A.1 A.2 → ellipse B.1 B.2 →
    (∀ t : ℝ, (t, l t) = A ∨ (t, l t) = B) →
    circle_through_origin (companion_point A.1 A.2) (companion_point B.1 B.2) →
    triangle_area (0, 0) A B = Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_triangle_OAB_is_sqrt_three_l912_91268


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_is_12_5_l912_91222

/-- The percentage of taxable land in the village -/
noncomputable def taxable_percentage : ℝ := 60

/-- The total farm tax collected from the village in dollars -/
noncomputable def total_tax : ℝ := 4000

/-- The farm tax paid by Mr. Willam in dollars -/
noncomputable def willam_tax : ℝ := 500

/-- The percentage of Mr. Willam's taxable land over the total taxable land of the village -/
noncomputable def willam_land_percentage : ℝ := (willam_tax / total_tax) * 100

theorem willam_land_percentage_is_12_5 :
  willam_land_percentage = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_willam_land_percentage_is_12_5_l912_91222


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_riddle_solution_l912_91267

-- Define the days of the week
inductive Day
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  deriving BEq, Repr

-- Define the brothers
inductive Brother
  | First
  | Second
  deriving BEq, Repr

-- Define the character types
inductive Character
  | Lion
  | Unicorn
  deriving BEq, Repr

-- Define the properties of the problem
structure RiddleProblem where
  current_day : Day
  tweedledee : Brother
  tweedledee_character : Character
  lion_lie_days : List Day
  tweedledee_lie_days : List Day

-- Helper function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Helper function to get the previous day
def prevDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Saturday
  | Day.Monday => Day.Sunday
  | Day.Tuesday => Day.Monday
  | Day.Wednesday => Day.Tuesday
  | Day.Thursday => Day.Wednesday
  | Day.Friday => Day.Thursday
  | Day.Saturday => Day.Friday

-- Define the statements made by the brothers
def first_statement (day : Day) : Prop := day ≠ Day.Sunday
def second_statement (day : Day) : Prop := day = Day.Monday
def third_statement (problem : RiddleProblem) : Prop :=
  problem.tweedledee_lie_days.contains (nextDay problem.current_day)
def fourth_statement (problem : RiddleProblem) : Prop :=
  problem.lion_lie_days.contains (prevDay problem.current_day)

-- Define the main theorem
theorem riddle_solution (problem : RiddleProblem) :
  (∀ d : Day, d ≠ Day.Sunday → (first_statement d ↔ ¬ second_statement d)) →
  (third_statement problem ↔ ¬ fourth_statement problem) →
  (problem.current_day = Day.Friday ∧
   problem.tweedledee = Brother.Second ∧
   problem.tweedledee_character = Character.Unicorn) := by
  sorry

#check riddle_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_riddle_solution_l912_91267


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_donut_boxes_l912_91260

-- Define constants
def donuts_per_day : ℕ := 100
def days_per_year : ℕ := 365
def jeff_consumption_rate : ℚ := 4 / 100
def friend_chris_consumption : ℕ := 15
def friend_sam_consumption : ℕ := 20
def friend_emily_consumption : ℕ := 25
def charity_donation_rate : ℚ := 10 / 100
def neighbor_share_rate : ℚ := 6 / 100
def donuts_per_box : ℕ := 50

-- Define the theorem
theorem jeff_donut_boxes : 
  ∃ (remaining_donuts : ℕ), 
    remaining_donuts = 
      donuts_per_day * days_per_year - 
      (↑(donuts_per_day * days_per_year) * jeff_consumption_rate).floor - 
      ((friend_chris_consumption + friend_sam_consumption + friend_emily_consumption) * 12) - 
      (↑(donuts_per_day * 30 * 12) * charity_donation_rate).floor - 
      (↑(donuts_per_day * days_per_year) * neighbor_share_rate).floor ∧
    remaining_donuts / donuts_per_box = 570 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jeff_donut_boxes_l912_91260


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_l912_91253

/-- A man crosses a bridge of length 1250 meters in 15 minutes. -/
theorem mans_speed (bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : bridge_length = 1250) (h2 : crossing_time = 15) :
  bridge_length / (crossing_time / 60) = 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mans_speed_l912_91253


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l912_91271

/-- Calculates the speed in km/h given distance in meters and time in minutes -/
noncomputable def calculate_speed (distance_m : ℝ) (time_min : ℝ) : ℝ :=
  (distance_m / 1000) / (time_min / 60)

theorem speed_calculation :
  let street_length_m : ℝ := 1000
  let crossing_time_min : ℝ := 10
  calculate_speed street_length_m crossing_time_min = 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_speed_calculation_l912_91271


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ellipse_perimeter_l912_91219

/-- Given a rectangle ABCD and an ellipse, prove the perimeter of the rectangle -/
theorem rectangle_ellipse_perimeter (x y : ℝ) (h1 : x > 0) (h2 : y > 0) : 
  x * y = 2006 →
  (x + y) / 2 * (4012 / (x + y)) = 2006 →
  ((x + y) / 2)^2 = (4012 / (x + y))^2 + (Real.sqrt (x^2 + y^2) / 2)^2 →
  2 * (x + y) = 8 * Real.sqrt 1003 := by
  intro h3 h4 h5
  sorry

#check rectangle_ellipse_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_ellipse_perimeter_l912_91219


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l912_91248

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 1)

-- State the theorem
theorem power_function_decreasing (m : ℝ) :
  (∀ x > 0, f m x = x^(m^2 + m - 1)) →
  (∀ x > 0, ∀ y > 0, x < y → f m x > f m y) →
  m = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_decreasing_l912_91248


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_tetrahedron_volume_ratio_l912_91230

/-- The volume of a regular tetrahedron with side length a -/
noncomputable def tetrahedron_volume (a : ℝ) : ℝ := (a^3 * Real.sqrt 2) / 12

/-- The ratio of volumes of two regular tetrahedrons where the edge length of one is half the other -/
noncomputable def volume_ratio (a : ℝ) : ℝ :=
  tetrahedron_volume (a / 2) / tetrahedron_volume a

theorem midpoint_tetrahedron_volume_ratio :
  ∀ a : ℝ, a > 0 → volume_ratio a = 1 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_midpoint_tetrahedron_volume_ratio_l912_91230


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_last_year_l912_91245

/-- The price of apples this year in yuan per kilogram -/
def a : ℝ := 1  -- Assigning a default value for demonstration

/-- The discount percentage as a decimal -/
def discount : ℝ := 0.2

/-- The price of apples last year in yuan per kilogram -/
noncomputable def last_year_price : ℝ := a / (1 - discount)

/-- Theorem: The price of apples last year was a / (1 - 20%) yuan per kilogram -/
theorem apple_price_last_year :
  (1 - discount) * last_year_price = a := by
  -- Unfold the definitions
  unfold last_year_price discount
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_apple_price_last_year_l912_91245


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l912_91299

/-- Given a triangle DEF with side lengths d = 6, e = 8, and cos(D - E) = 15/17,
    the length of side f is approximately 6.5. -/
theorem triangle_side_length (D E F : ℝ) (d e f : ℝ) : 
  d = 6 →
  e = 8 →
  Real.cos (D - E) = 15 / 17 →
  0 < D ∧ D < π →
  0 < E ∧ E < π →
  0 < F ∧ F < π →
  D + E + F = π →
  f^2 = d^2 + e^2 - 2 * d * e * Real.cos F →
  abs (f - 6.5) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_length_l912_91299


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l912_91214

/-- Represents a tetrahedron ABCD with perpendicular edges AB, AC, and AD -/
structure Tetrahedron where
  a : ℝ  -- Length of edge AB
  b : ℝ  -- Length of edge AC
  c : ℝ  -- Length of edge AD
  x : ℝ  -- Area of triangle ABC
  y : ℝ  -- Area of triangle ACD
  z : ℝ  -- Area of triangle ADB
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_perp : True  -- Represents that AB, AC, and AD are mutually perpendicular
  h_area_x : x = (1/2) * a * b
  h_area_y : y = (1/2) * b * c
  h_area_z : z = (1/2) * c * a

/-- The volume of the tetrahedron ABCD -/
noncomputable def tetrahedron_volume (t : Tetrahedron) : ℝ := (8 * t.x * t.y * t.z) / (t.a * t.b * t.c)

/-- Theorem stating that the volume of the tetrahedron ABCD is (8xyz)/(abc) -/
theorem volume_formula (t : Tetrahedron) : 
  tetrahedron_volume t = (8 * t.x * t.y * t.z) / (t.a * t.b * t.c) := by
  -- Proof will be added later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_formula_l912_91214


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l912_91295

-- Define the curve E
def E (x y : ℝ) : Prop := x^2 / 2 + y^2 = 1

-- Define the distance ratio condition
noncomputable def distance_ratio (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) / abs (x - 2) = Real.sqrt 2 / 2

-- Define the line intersecting E
def intersecting_line (k m : ℝ) (x y : ℝ) : Prop := y = k * x + m

-- Define the condition for equal triangle areas
def equal_triangle_areas (k m : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  abs (x₁ + m / k) * y₁ / 2 = abs (x₂ * m) / 2

-- Theorem statement
theorem curve_and_line_properties :
  ∀ (x y k m : ℝ),
    E x y →
    distance_ratio x y →
    m ≠ 0 →
    m^2 < 2 * k^2 + 1 →
    abs m < Real.sqrt 2 →
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      E x₁ y₁ ∧ E x₂ y₂ ∧
      intersecting_line k m x₁ y₁ ∧
      intersecting_line k m x₂ y₂ ∧
      (k = Real.sqrt 2 / 2 ∨ k = -Real.sqrt 2 / 2) →
      equal_triangle_areas k m x₁ y₁ x₂ y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_and_line_properties_l912_91295


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l912_91278

-- Define the sets M and N
def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = Set.Ioo (-1) 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_M_and_N_l912_91278


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l912_91234

theorem sin_beta_value (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
  (h3 : Real.cos α = 4 / 5) (h4 : Real.cos (α + β) = 5 / 13) : Real.sin β = 33 / 65 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_beta_value_l912_91234
