import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_14th_row_5th_number_l41_4114

theorem pascal_triangle_14th_row_5th_number : 
  Nat.choose 14 4 = 1001 := by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_14th_row_5th_number_l41_4114


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l41_4119

noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + d * (n - 1 : ℝ)

noncomputable def sum_arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_problem (a₁ d : ℝ) (k : ℕ) :
  a₁ = 1 →
  sum_arithmetic_sequence a₁ d 9 = sum_arithmetic_sequence a₁ d 4 →
  arithmetic_sequence a₁ d k + arithmetic_sequence a₁ d 4 = 0 →
  k = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l41_4119


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_l41_4109

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (1/2)^x - 7 else x^2

theorem f_solution (a : ℝ) : f a = 1 ↔ a = -3 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_solution_l41_4109


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l41_4116

-- Define the ellipse C
noncomputable def ellipse (x y : ℝ) (a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the eccentricity
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 - b^2 / a^2)

-- Define the area of the incircle
noncomputable def incircle_area (a b c : ℝ) : ℝ := Real.pi * (b * c / (a + c))^2

-- Define the ratio sum
noncomputable def ratio_sum (pf1 f1a pf2 f2b : ℝ) : ℝ := pf1 / f1a + pf2 / f2b

-- Theorem statement
theorem ellipse_properties (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a b = 1/2) 
  (h4 : ∃ x y, ellipse x y a b ∧ incircle_area a b (Real.sqrt (a^2 - b^2)) = Real.pi / 3) :
  (a = 2 ∧ b = Real.sqrt 3) ∧
  (∀ x y pf1 f1a pf2 f2b, 
    ellipse x y a b → ratio_sum pf1 f1a pf2 f2b = 10/3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l41_4116


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_correct_l41_4123

noncomputable def p (x : ℝ) : ℝ := (3/2) * x^2 - 3*x - 9/2

theorem p_is_correct :
  (p 3 = 0) ∧
  (p (-1) = 0) ∧
  (∀ n : ℕ, n ≥ 3 → ∀ x : ℝ, p x ≠ x^n) ∧
  (p (-3) = 18) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_is_correct_l41_4123


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_fifty_l41_4189

noncomputable def purchase_price : ℚ := 11000
noncomputable def repair_cost : ℚ := 5000
noncomputable def transportation_charges : ℚ := 1000
noncomputable def selling_price : ℚ := 25500

noncomputable def total_cost : ℚ := purchase_price + repair_cost + transportation_charges
noncomputable def profit : ℚ := selling_price - total_cost
noncomputable def profit_percentage : ℚ := (profit / total_cost) * 100

theorem profit_percentage_is_fifty :
  profit_percentage = 50 := by
  -- Expand definitions
  unfold profit_percentage profit total_cost
  -- Simplify the expression
  simp [purchase_price, repair_cost, transportation_charges, selling_price]
  -- The proof is complete
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_profit_percentage_is_fifty_l41_4189


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_relationships_l41_4193

noncomputable def circle_radii : List Real := [2, 3, 4, 5, 6, 7]

noncomputable def circumference (r : Real) : Real := 2 * Real.pi * r
def diameter (r : Real) : Real := 2 * r
noncomputable def area (r : Real) : Real := Real.pi * r^2

def is_linear_relationship (xs ys : List Real) : Prop :=
  ∃ (m c : Real), ∀ (pair : Real × Real), pair ∈ (List.zip xs ys) → pair.2 = m * pair.1 + c

theorem circle_relationships :
  let c_values := circle_radii.map circumference
  let d_values := circle_radii.map diameter
  let a_values := circle_radii.map area
  (is_linear_relationship c_values d_values) ∧
  ¬(is_linear_relationship c_values a_values) ∧
  ¬(is_linear_relationship d_values a_values) := by
  sorry

#eval "Circle relationships theorem stated successfully."

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_relationships_l41_4193


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_range_of_m_l41_4165

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop :=
  y + Real.sqrt 3 * x = Real.sqrt 3 * m

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 3

-- Define the distance from a point to the line
noncomputable def distance_to_line (m : ℝ) (x y : ℝ) : ℝ :=
  abs (y + Real.sqrt 3 * x - Real.sqrt 3 * m) / 2

-- Theorem 1: Line l is tangent to curve C when m = 3
theorem line_tangent_to_curve : 
  ∃ (x y : ℝ), curve_C x y ∧ line_l 3 x y ∧
  ∀ (x' y' : ℝ), curve_C x' y' → distance_to_line 3 x' y' ≥ 0 := by
  sorry

-- Theorem 2: Range of m for which there exists a point on C at distance √3/2 from l
theorem range_of_m : 
  ∀ m : ℝ, (∃ (x y : ℝ), curve_C x y ∧ distance_to_line m x y = Real.sqrt 3 / 2) ↔ 
  -2 ≤ m ∧ m ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_curve_range_of_m_l41_4165


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_implies_m_value_l41_4161

/-- 
Given a line passing through points (1,m) and (2,3),
if the slope of the line is ± 5/12, then m = 2.
-/
theorem line_slope_implies_m_value 
  (m : ℝ) 
  (h : (m - 3) / (1 - 2) = 5 / 12 ∨ (m - 3) / (1 - 2) = -5 / 12) : 
  m = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_slope_implies_m_value_l41_4161


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l41_4127

noncomputable def f (x : ℝ) : ℝ := 3 - 4 * x

noncomputable def g (x : ℝ) : ℝ := (3 - x) / 4

theorem f_inverse_is_g : Function.LeftInverse g f ∧ Function.RightInverse g f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inverse_is_g_l41_4127


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l41_4162

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₁ - C₂| / Real.sqrt (A^2 + B^2)

/-- Line l₁: 3x + 4y - 5 = 0 -/
def l₁ (x y : ℝ) : Prop := 3 * x + 4 * y - 5 = 0

/-- Line l₂: 3x + 4y + 5 = 0 -/
def l₂ (x y : ℝ) : Prop := 3 * x + 4 * y + 5 = 0

theorem distance_between_l₁_and_l₂ :
  distance_between_parallel_lines 3 4 (-5) 5 = 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_l₁_and_l₂_l41_4162


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l41_4121

theorem expression_evaluation (x : ℝ) (h : x = 2) : 
  (3 / (x + 2) + x - 2) / ((x^2 - 2*x + 1) / (x + 2)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l41_4121


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_altitude_rectangle_area_l41_4142

-- Define the necessary structures and functions
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

def IsRightTriangle (t : Triangle) : Prop := sorry
def HypotenuseLength (t : Triangle) : ℝ := sorry
def LegLength (t : Triangle) (i : Fin 2) : ℝ := sorry
def Altitude : Type := ℝ × ℝ
def IsAltitudeToHypotenuse (t : Triangle) (a : Altitude) : Prop := sorry
def AreaOfRectangleWithDiagonal (a : Altitude) : ℝ := sorry

theorem right_triangle_altitude_rectangle_area 
  (t : Triangle) 
  (h_right : IsRightTriangle t)
  (h_hypotenuse : HypotenuseLength t = 2)
  (h_leg : ∃ i : Fin 2, LegLength t i = 1)
  (a : Altitude)
  (h_altitude : IsAltitudeToHypotenuse t a) :
  AreaOfRectangleWithDiagonal a = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_altitude_rectangle_area_l41_4142


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seashell_difference_l41_4100

/- Define the number of seashells for each child -/
variable (stefan : ℕ)
variable (vail : ℕ)
def aiguo : ℕ := 20

/- Define the conditions -/
axiom stefan_more_than_vail : stefan = vail + 16
axiom vail_less_than_aiguo : vail < aiguo
axiom total_seashells : stefan + vail + aiguo = 66

/- Theorem to prove -/
theorem seashell_difference : aiguo - vail = 5 := by
  sorry

#check seashell_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seashell_difference_l41_4100


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_not_contradictory_l41_4143

-- Define the set of students
inductive Student : Type
| A : Student
| B : Student
| C : Student
| D : Student
| E : Student

-- Define the set of colors
inductive Color : Type
| Red : Color
| Other1 : Color
| Other2 : Color
| Other3 : Color
| Other4 : Color

-- Define the distribution of balls
def distribution : Student → Color := sorry

-- Define the event "A receives the red ball"
def event_A : Prop := distribution Student.A = Color.Red

-- Define the event "B receives the red ball"
def event_B : Prop := distribution Student.B = Color.Red

-- State the theorem
theorem events_mutually_exclusive_not_contradictory :
  (∀ s : Student, ∃! c : Color, distribution s = c) →
  (∀ c : Color, ∃! s : Student, distribution s = c) →
  (¬(event_A ∧ event_B)) ∧ (¬(¬event_A ∧ ¬event_B)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_events_mutually_exclusive_not_contradictory_l41_4143


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heather_walking_distance_l41_4185

/-- The distance between Stacy and Heather's starting points in miles -/
noncomputable def initial_distance : ℝ := 5

/-- Heather's walking speed in miles per hour -/
noncomputable def heather_speed : ℝ := 5

/-- The difference between Stacy's and Heather's speeds in miles per hour -/
noncomputable def speed_difference : ℝ := 1

/-- The time difference between Stacy's and Heather's start times in hours -/
noncomputable def start_time_difference : ℝ := 24 / 60

/-- The distance Heather walks before meeting Stacy -/
noncomputable def heather_distance (t : ℝ) : ℝ := heather_speed * t

/-- The distance Stacy walks before meeting Heather -/
noncomputable def stacy_distance (t : ℝ) : ℝ := (heather_speed + speed_difference) * (t + start_time_difference)

/-- The theorem stating that Heather walks approximately 1.18 miles before meeting Stacy -/
theorem heather_walking_distance :
  ∃ t : ℝ, t > 0 ∧ heather_distance t + stacy_distance t = initial_distance ∧
  |heather_distance t - 1.18| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heather_walking_distance_l41_4185


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_5_l41_4132

-- Define the functions t and f
noncomputable def t (x : ℝ) : ℝ := Real.sqrt (4 * x + 2)
noncomputable def f (x : ℝ) : ℝ := 7 - 2 * t x

-- State the theorem
theorem t_of_f_5 : t (f 5) = Real.sqrt (30 - 8 * Real.sqrt 22) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_t_of_f_5_l41_4132


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_behavior_l41_4118

/-- Represents an ellipse with foci on the x-axis -/
structure Ellipse (a : ℝ) where
  equation : ∀ x y : ℝ, x^2 / (4*a) + y^2 / (a^2 + 1) = 1
  foci_on_x_axis : True
  a_range : 2 - Real.sqrt 3 < a ∧ a < 2 + Real.sqrt 3

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (a : ℝ) : ℝ := Real.sqrt ((4*a - a^2 - 1) / (4*a))

/-- Theorem stating the behavior of ellipse eccentricity as 'a' increases -/
theorem ellipse_eccentricity_behavior (a : ℝ) (e : Ellipse a) :
  ∃ a₁ a₂ : ℝ, 
    (2 - Real.sqrt 3 < a₁ ∧ a₁ < 2) ∧
    (2 < a₂ ∧ a₂ < 2 + Real.sqrt 3) ∧
    (∀ x y : ℝ, a₁ < x ∧ x < y ∧ y < 2 → eccentricity x < eccentricity y) ∧
    (∀ x y : ℝ, 2 < x ∧ x < y ∧ y < a₂ → eccentricity x > eccentricity y) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_behavior_l41_4118


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l41_4175

/-- The sum of the infinite series Σ(1 / (n(n+1)(n+2))) from n=1 to infinity -/
noncomputable def infiniteSeries : ℝ := ∑' n, 1 / (n * (n + 1) * (n + 2))

/-- Theorem stating that the sum of the infinite series equals 1/4 -/
theorem infiniteSeriesSum : infiniteSeries = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infiniteSeriesSum_l41_4175


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_eq_real_l41_4171

/-- The set M of real numbers x such that x^2 + 3x + 2 > 0 -/
def M : Set ℝ := {x | x^2 + 3*x + 2 > 0}

/-- The set N of real numbers x such that (1/2)^x ≤ 4 -/
def N : Set ℝ := {x | Real.rpow (1/2) x ≤ 4}

/-- Theorem stating that the union of sets M and N is equal to the set of all real numbers -/
theorem union_M_N_eq_real : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_eq_real_l41_4171


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_strength_increases_l41_4184

/-- The correlation coefficient between two variables -/
def correlation_coefficient : ℝ → ℝ := sorry

/-- The strength of linear correlation between two variables -/
def linear_correlation_strength : ℝ → ℝ := sorry

/-- As the absolute value of the correlation coefficient approaches 1,
    the linear correlation strength increases -/
theorem correlation_strength_increases (r : ℝ) :
  ∀ ε > 0, ∃ δ > 0, ∀ r', |r'| > 1 - δ → 
  linear_correlation_strength r' > linear_correlation_strength r - ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correlation_strength_increases_l41_4184


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_are_skew_iff_b_neq_9_l41_4167

/-- Two lines in 3D space defined by their parametric equations -/
structure Line3D where
  point : Fin 3 → ℝ
  direction : Fin 3 → ℝ

/-- Check if two lines in 3D space are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  ∀ s v : ℝ, ∃ i : Fin 3, 
    l1.point i + s * l1.direction i ≠ l2.point i + v * l2.direction i

/-- The main theorem stating the condition for the lines to be skew -/
theorem lines_are_skew_iff_b_neq_9 (b : ℝ) :
  are_skew 
    (Line3D.mk (![2, 3, b]) (![3, 4, 5]))
    (Line3D.mk (![5, 2, 1]) (![6, 3, 2]))
  ↔ b ≠ 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_lines_are_skew_iff_b_neq_9_l41_4167


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_and_variance_of_linear_transformation_l41_4115

-- Define the random variable X
def X : ℝ → ℝ := sorry

-- Define the probability mass function for X
noncomputable def pmf_X (x : ℝ) : ℝ :=
  if x = 0 then 0.3
  else if x = 1 then 0.4
  else if x = 2 then 0.3
  else 0

-- Define the expected value of X
noncomputable def E_X : ℝ := 0 * 0.3 + 1 * 0.4 + 2 * 0.3

-- Define the second moment of X
noncomputable def E_X_squared : ℝ := 0^2 * 0.3 + 1^2 * 0.4 + 2^2 * 0.3

-- Define the variance of X
noncomputable def Var_X : ℝ := E_X_squared - E_X^2

-- Theorem to prove
theorem expected_value_and_variance_of_linear_transformation :
  (2 * E_X - 1 = 1) ∧ (4 * Var_X = 2.4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_value_and_variance_of_linear_transformation_l41_4115


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l41_4144

/-- A journey with two parts, where the first part is traveled at a known speed -/
structure Journey where
  total_distance : ℝ
  total_time : ℝ
  first_part_speed : ℝ
  first_part_distance_ratio : ℝ
  first_part_time_ratio : ℝ

/-- Calculate the speed required for the second part of the journey -/
noncomputable def second_part_speed (j : Journey) : ℝ :=
  (j.total_distance * (1 - j.first_part_distance_ratio)) /
  (j.total_time * (1 - j.first_part_time_ratio))

/-- Theorem stating the conditions and the result to be proved -/
theorem journey_speed_calculation (j : Journey) 
  (h1 : j.first_part_speed = 80)
  (h2 : j.first_part_distance_ratio = 2/3)
  (h3 : j.first_part_time_ratio = 1/3) :
  second_part_speed j = 20 := by
  sorry

#check journey_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_speed_calculation_l41_4144


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bilion_always_wins_winning_amount_l41_4169

/-- Represents the denominations of bills available in the game -/
inductive Bill
  | one
  | two
  | five
  | ten

/-- Represents a player in the game -/
inductive Player
  | bilion
  | trilion

/-- The game state -/
structure GameState where
  pile : ℕ  -- Current total in the pile
  turn : Player  -- Player whose turn it is

/-- Function to get the value of a bill -/
def billValue : Bill → ℕ
  | Bill.one => 1
  | Bill.two => 2
  | Bill.five => 5
  | Bill.ten => 10

/-- Function to determine if a game state is winning for the current player -/
def isWinningState (state : GameState) : Prop :=
  ∃ (bill : Bill), state.pile + billValue bill = 1000000

/-- The main theorem stating that Bilion can always win -/
theorem bilion_always_wins :
  ∀ (state : GameState),
    state.turn = Player.bilion →
    state.pile < 1000000 →
    state.pile % 3 = 1 ∨ state.pile % 3 = 2 →
    isWinningState state ∨
    ∃ (bill : Bill),
      ¬isWinningState { pile := state.pile + billValue bill, turn := Player.trilion } :=
by sorry

/-- The final theorem stating the winning amount -/
theorem winning_amount :
  ∃ (finalState : GameState),
    finalState.pile = 1000000 ∧
    finalState.turn = Player.bilion ∧
    isWinningState finalState :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bilion_always_wins_winning_amount_l41_4169


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_subtraction_theorem_l41_4197

/-- Represents a number in base 7 --/
structure Base7 where
  value : Nat
  is_valid : value < 7^4 := by sorry

/-- Converts a base 7 number to its decimal representation --/
def to_decimal (n : Base7) : Nat :=
  sorry

/-- Subtracts two base 7 numbers and returns the result in base 7 --/
def base7_subtract (a b : Base7) : Base7 :=
  sorry

/-- Constructs a Base7 number from a Nat --/
def mk_base7 (n : Nat) : Base7 :=
  ⟨n % (7^4), by sorry⟩

theorem base7_subtraction_theorem :
  base7_subtract (mk_base7 1000) (mk_base7 666) = mk_base7 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base7_subtraction_theorem_l41_4197


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_game_probability_difference_l41_4130

theorem biased_coin_game_probability_difference :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let game_c_win (outcomes : List Bool) : Bool :=
    outcomes.length = 4 ∧ (outcomes.all id ∨ outcomes.all not)
  let game_d_win (outcomes : List Bool) : Bool :=
    outcomes.length = 5 ∧
    ((outcomes.take 3).all id ∨ (outcomes.take 3).all not) ∧
    (outcomes.get? 3).isSome ∧ (outcomes.get? 4).isSome ∧
    (outcomes.get? 3 ≠ outcomes.get? 4)
  let p_game_c : ℚ := 17/81
  let p_game_d : ℚ := 12/81
  p_game_c - p_game_d = 5/81 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_biased_coin_game_probability_difference_l41_4130


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l41_4180

noncomputable def a : ℕ → ℝ
  | 0 => 1
  | n + 1 => (5 / 3) * a n + (4 / 3) * Real.sqrt (4^n - (a n)^2)

theorem a_5_value : a 5 = 22.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_5_value_l41_4180


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_line_l41_4120

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parabola defined by its focus and directrix -/
structure Parabola where
  focus : Point2D
  directrix : ℝ → ℝ → Prop

/-- The intersection line of two parabolas -/
def intersectionLine (p1 p2 : Parabola) : ℝ → ℝ → Prop := sorry

/-- Given two parabolas with the same focus and specific directrices, 
    their intersection line has the equation √3x - y = 0 -/
theorem parabola_intersection_line :
  ∀ (c1 c2 : Parabola),
    c1.focus = Point2D.mk 2 (Real.sqrt 3) ∧
    c2.focus = Point2D.mk 2 (Real.sqrt 3) ∧
    c1.directrix = (fun x y => x = 0) ∧
    c2.directrix = (fun x y => x - Real.sqrt 3 * y = 0) →
    intersectionLine c1 c2 = (fun x y => Real.sqrt 3 * x - y = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_line_l41_4120


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_to_decimal_l41_4187

/-- The decimal form of 5.2 × 10^(-5) is 0.000052 -/
theorem scientific_to_decimal : (5.2 * (10 : ℝ)^(-5 : ℤ)) = 0.000052 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_scientific_to_decimal_l41_4187


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_relative_position_l41_4150

noncomputable def parabola1 (x : ℝ) : ℝ := x^2 - 2*x + 3
noncomputable def parabola2 (x : ℝ) : ℝ := x^2 + 2*x + 1

noncomputable def vertex (a b c : ℝ) : ℝ × ℝ :=
  let h := -b / (2*a)
  let k := c - (b^2) / (4*a)
  (h, k)

theorem parabola_relative_position :
  let v1 := vertex 1 (-2) 3
  let v2 := vertex 1 2 1
  v1.1 > v2.1 ∧ v1.2 > v2.2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_relative_position_l41_4150


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_most_frequent_l41_4134

/-- The set of integers from 1 to 9 -/
def IntSet : Finset ℕ := Finset.filter (λ n => 1 ≤ n ∧ n ≤ 9) (Finset.range 10)

/-- The sum of two integers from IntSet -/
def SumSet : Finset ℕ := Finset.biUnion IntSet (λ x => Finset.image (λ y => x + y) IntSet)

/-- The count of occurrences for each units digit in SumSet -/
def DigitCount (d : ℕ) : ℕ := (SumSet.filter (λ s => s % 10 = d)).card

theorem zero_most_frequent :
  ∀ d : ℕ, d ≠ 0 → DigitCount 0 > DigitCount d :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_most_frequent_l41_4134


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_trigonometric_inequality_holds_l41_4159

noncomputable section

open Real

theorem trigonometric_inequality : ℝ → Prop :=
  fun r : ℝ => 
    let a := (1/2 : ℝ) * cos (7 * π / 180) + (sqrt 3 / 2) * sin (7 * π / 180)
    let b := (2 * tan (19 * π / 180)) / (1 - tan (19 * π / 180) ^ 2)
    let c := sqrt ((1 - cos (72 * π / 180)) / 2)
    b > a ∧ a > c

theorem trigonometric_inequality_holds : trigonometric_inequality 1 := by
  -- The proof is omitted
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_inequality_trigonometric_inequality_holds_l41_4159


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_slice_area_l41_4147

noncomputable def cube_side_length : ℝ := 2

noncomputable def space_diagonal_length : ℝ := 2 * Real.sqrt 3

noncomputable def face_diagonal_length : ℝ := 2 * Real.sqrt 2

noncomputable def quadrilateral_side_length : ℝ := Real.sqrt 5

noncomputable def quadrilateral_area : ℝ := 2 * Real.sqrt 6

theorem cube_slice_area :
  let cube_side := cube_side_length
  let space_diag := space_diagonal_length
  let face_diag := face_diagonal_length
  let quad_side := quadrilateral_side_length
  quad_side = Real.sqrt (1^2 + cube_side^2) ∧
  space_diag = Real.sqrt (3 * cube_side^2) ∧
  face_diag = Real.sqrt (2 * cube_side^2) ∧
  quadrilateral_area = (1/2) * space_diag * face_diag := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_slice_area_l41_4147


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_common_roots_cubic_polynomials_l41_4188

theorem common_roots_cubic_polynomials :
  ∃! (c d : ℝ), 
    (∃ (u v : ℝ), u ≠ v ∧ 
      (u^3 + c*u^2 + 8*u + 5 = 0) ∧ 
      (u^3 + d*u^2 + 10*u + 7 = 0) ∧
      (v^3 + c*v^2 + 8*v + 5 = 0) ∧ 
      (v^3 + d*v^2 + 10*v + 7 = 0)) →
    c = 5 ∧ d = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_common_roots_cubic_polynomials_l41_4188


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cubes_sphere_l41_4136

theorem inscribed_cubes_sphere (outer_cube_surface_area : ℝ) 
  (h_outer_area : outer_cube_surface_area = 24) : 
  ∃ (inner_surface_area : ℝ), inner_surface_area = 8 := by
  let outer_side := Real.sqrt (outer_cube_surface_area / 6)
  let sphere_diameter := outer_side
  let inner_side := sphere_diameter / Real.sqrt 3
  let inner_surface_area := 6 * inner_side^2
  exists inner_surface_area
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cubes_sphere_l41_4136


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_icing_area_is_28_l41_4112

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the cube and its cuts -/
structure CutCube where
  edgeLength : ℝ
  topFace : List Point3D
  firstCut : Point3D × Point3D
  secondCut : Point3D × Point3D

/-- Calculates the volume of the quadrilateral piece -/
noncomputable def calculateVolume (cube : CutCube) : ℝ :=
  sorry

/-- Calculates the surface area of icing on the quadrilateral piece -/
noncomputable def calculateIcingArea (cube : CutCube) : ℝ :=
  sorry

/-- Theorem stating that the sum of volume and icing area is 28 -/
theorem volume_plus_icing_area_is_28 (cube : CutCube) 
  (h1 : cube.edgeLength = 4)
  (h2 : cube.topFace = [{ x := 0, y := 0, z := 4 }, { x := 4, y := 0, z := 4 }, { x := 4, y := 4, z := 4 }, { x := 0, y := 4, z := 4 }])
  (h3 : cube.firstCut = ({ x := 0, y := 0, z := 4 }, { x := 4, y := 2, z := 4 }))
  (h4 : cube.secondCut = ({ x := 2, y := 2, z := 2 }, { x := 2, y := 0, z := 4 })) :
  calculateVolume cube + calculateIcingArea cube = 28 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_plus_icing_area_is_28_l41_4112


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_area_is_444_l41_4102

/-- Represents a room in Johan's house -/
structure Room where
  walls : ℕ

/-- Johan's house with 10 bedrooms -/
def house : List Room := [
  ⟨6⟩, ⟨8⟩, ⟨10⟩, ⟨12⟩, ⟨14⟩, ⟨8⟩, ⟨5⟩, ⟨10⟩, ⟨7⟩, ⟨15⟩
]

/-- The wall area in square meters -/
def wallArea : ℝ := 12

/-- Predicate to determine if a room is painted purple -/
def isPurple (r : Room) : Bool :=
  r.walls ∉ [6, 8, 10, 12, 14] || (r.walls = 10 && (house.filter (λ x => x.walls = 10)).length > 1)

/-- The total surface area of walls painted purple -/
def purpleArea : ℝ :=
  (house.filter isPurple).map (λ r => (r.walls : ℝ) * wallArea) |>.sum

/-- Theorem: The surface area of walls painted purple is 444 square meters -/
theorem purple_area_is_444 : purpleArea = 444 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_purple_area_is_444_l41_4102


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l41_4168

-- Define the hyperbola
noncomputable def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the asymptote
def asymptote (x y : ℝ) : Prop :=
  3*x - y + 1 = 0

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2) / a

-- Theorem statement
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) :
  (∃ (k : ℝ), ∀ (x y : ℝ), hyperbola a b x y → asymptote (x + k) (y + k)) →
  eccentricity a b = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l41_4168


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l41_4164

theorem angle_sum_proof (α β : Real) : 
  0 < α ∧ α < π/2 →  -- α is acute
  0 < β ∧ β < π/2 →  -- β is acute
  Real.sin α = Real.sqrt 5 / 5 →
  Real.cos β = 3 * Real.sqrt 10 / 10 →
  α + β = π/4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_proof_l41_4164


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_sale_loss_l41_4126

theorem calculator_sale_loss (price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  price = 120 ∧ 
  profit_percent = 20 ∧ 
  loss_percent = 20 → 
  (price / (1 + profit_percent / 100) + price / (1 - loss_percent / 100)) - 2 * price = -10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculator_sale_loss_l41_4126


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_twelve_positive_integers_l41_4117

def first_twelve_positive_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

theorem median_of_first_twelve_positive_integers :
  let sorted_list := first_twelve_positive_integers
  let n := sorted_list.length
  let median := if n % 2 = 0
                then (sorted_list[n / 2 - 1]! + sorted_list[n / 2]!) / 2
                else sorted_list[n / 2]!
  median = (13 : ℚ) / 2 := by
  sorry

#eval (13 : ℚ) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_median_of_first_twelve_positive_integers_l41_4117


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_abc_measure_l41_4105

/-- An irregular pentagon is a polygon with five sides and five angles. -/
structure IrregularPentagon where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The angle between three points -/
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem angle_abc_measure (pentagon : IrregularPentagon) :
  angle pentagon.A pentagon.B pentagon.C = 2 * angle pentagon.D pentagon.B pentagon.E →
  angle pentagon.A pentagon.B pentagon.C = 60 := by
  sorry

#check angle_abc_measure

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_abc_measure_l41_4105


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l41_4174

-- Define the hyperbola equation
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 1

-- Define the bounding line equation
def boundingLine (x y : ℝ) : Prop := 3*x = 4*y

-- Define the region of interest
def region (x y : ℝ) : Prop :=
  hyperbola x y ∧ y ≥ 0 ∧ boundingLine x y

-- State the theorem
theorem area_of_region :
  ∃ (A : ℝ), A = (Real.log 7) / 4 ∧
  A = ∫ x in Set.Icc 0 ((4:ℝ)/3), ∫ y in Set.Icc 0 (Real.sqrt (x^2 - 1)), 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_region_l41_4174


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_trains_problem_l41_4192

/-- Calculates the distance traveled by a train given initial velocity, acceleration, and time -/
noncomputable def distance (v₀ : ℝ) (a : ℝ) (t : ℝ) : ℝ :=
  v₀ * t + (1/2) * a * t^2

/-- Represents the problem of two trains traveling on parallel tracks -/
theorem two_trains_problem (v₁₀ v₂₀ a₁ a₂ t : ℝ) 
  (h₁ : v₁₀ = 11) (h₂ : v₂₀ = 31) (h₃ : a₁ = 2.5) (h₄ : a₂ = 0.5) (h₅ : t = 8) :
  distance v₂₀ a₂ t - distance v₁₀ a₁ t = 96 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_trains_problem_l41_4192


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_l41_4131

-- Define the line l: y = x + 2
def line_l (x : ℝ) : ℝ := x + 2

-- Define the circle x^2 + y^2 = 5
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the intersection points M and N
def intersection_points (M N : ℝ × ℝ) : Prop :=
  circle_equation M.1 M.2 ∧ circle_equation N.1 N.2 ∧
  M.2 = line_l M.1 ∧ N.2 = line_l N.1 ∧
  M ≠ N

-- Theorem statement
theorem length_MN (M N : ℝ × ℝ) :
  intersection_points M N → dist M N = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_MN_l41_4131


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_approx_l41_4156

noncomputable def total_investment : ℝ := 25000
noncomputable def investment_per_venture : ℝ := 16250
noncomputable def profit_percentage : ℝ := 0.15
noncomputable def overall_return_percentage : ℝ := 0.08

noncomputable def loss_percentage : ℝ := 
  (profit_percentage * investment_per_venture - overall_return_percentage * total_investment) / investment_per_venture

theorem loss_percentage_approx :
  abs (loss_percentage - 0.0269) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_loss_percentage_approx_l41_4156


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_reciprocal_sum_l41_4177

theorem nested_reciprocal_sum : ((((2 : ℚ) + 1)⁻¹ + 1)⁻¹ + 1)⁻¹ + 1 = 11 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_reciprocal_sum_l41_4177


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_reciprocal_sum_l41_4125

theorem nested_reciprocal_sum : ((((3 : ℚ) + 2)⁻¹ + 2)⁻¹ + 2)⁻¹ + 2 = 65 / 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nested_reciprocal_sum_l41_4125


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_inscribed_circle_l41_4170

-- Define the side length of the equilateral triangle
def side_length : ℝ := 6

-- Define the altitude of the equilateral triangle
noncomputable def altitude (s : ℝ) : ℝ := (Real.sqrt 3 / 2) * s

-- Define the radius of the inscribed circle
noncomputable def inscribed_radius (h : ℝ) : ℝ := h / 2

-- Define x as the altitude minus 1 unit
def x (h : ℝ) : ℝ := h - 1

-- Theorem statement
theorem equilateral_triangle_inscribed_circle :
  x (altitude side_length) = 3 * Real.sqrt 3 - 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_inscribed_circle_l41_4170


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_animal_configurations_l41_4101

theorem three_animal_configurations : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    30 * p.1 + 35 * p.2 = 1400 ∧ p.2 ≥ p.1 ∧ p.1 > 0 ∧ p.2 > 0) 
    (Finset.product (Finset.range 47) (Finset.range 41))).card ∧ n = 3 := by
  sorry

#eval (Finset.filter (fun p : ℕ × ℕ => 
  30 * p.1 + 35 * p.2 = 1400 ∧ p.2 ≥ p.1 ∧ p.1 > 0 ∧ p.2 > 0) 
  (Finset.product (Finset.range 47) (Finset.range 41))).card

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_animal_configurations_l41_4101


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bending_resistance_l41_4183

/-- The diameter of the circular log --/
noncomputable def D : ℝ := 15 * Real.sqrt 3

/-- The bending resistance function --/
noncomputable def F (k : ℝ) (a : ℝ) : ℝ := k * a * (D^2 - a^2)

/-- Theorem: The width that maximizes bending resistance is 15 --/
theorem max_bending_resistance (k : ℝ) (h : k > 0) :
  ∃ (a : ℝ), a > 0 ∧ a < D ∧ 
  (∀ (x : ℝ), x > 0 → x < D → F k a ≥ F k x) ∧
  a = 15 := by
  sorry

#check max_bending_resistance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_bending_resistance_l41_4183


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_theta_l41_4108

theorem tan_pi_fourth_minus_theta (θ : Real) (h : Real.tan θ = 1/2) :
  Real.tan (π/4 - θ) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_fourth_minus_theta_l41_4108


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_series_l41_4124

/-- Triangle with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Set of ordered pairs for defining midpoints -/
def S (n : ℕ) : Set (ℕ × ℕ) :=
  {p | 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ n}

/-- Expected value of the square of the distance between a point and the centroid -/
noncomputable def expected_distance_squared (t : Triangle) (n : ℕ) : ℝ := sorry

/-- The infinite series in question -/
noncomputable def infinite_series (t : Triangle) : ℝ :=
  ∑' i, expected_distance_squared t (i + 4) * (3/4)^i

/-- The main theorem -/
theorem triangle_centroid_series (t : Triangle) 
  (h1 : t.side1 = 4) (h2 : t.side2 = 5) (h3 : t.side3 = 7) : 
  infinite_series t = 1859/84 - 1024/63 * Real.log 2 + 512/63 * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_centroid_series_l41_4124


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_a_range_l41_4148

/-- The cubic function f(x) = x^3 + ax^2 + 1 -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

/-- The derivative of f(x) -/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x

/-- The x-coordinate of the center of symmetry -/
noncomputable def x₀ (a : ℝ) : ℝ := -a/3

theorem cubic_function_a_range (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    f a x₁ = 0 ∧ f a x₂ = 0 ∧ f a x₃ = 0) →
  x₀ a > 0 →
  a < -3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_function_a_range_l41_4148


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l41_4195

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- f' is the derivative of f
axiom is_derivative : ∀ x, HasDerivAt f (f' x) x

-- f is an even function
axiom f_even : ∀ x, f x = f (-x)

-- f' is less than f for all x
axiom f'_less_f : ∀ x, f' x < f x

-- The theorem to prove
theorem inequality_holds : Real.exp (-1) * f 1 < f 0 ∧ f 0 < Real.exp 2 * f 2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_holds_l41_4195


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1991_of_2_1990_l41_4155

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- f₁(k) is the square of the sum of digits of k -/
def f₁ (k : ℕ) : ℕ := (sum_of_digits k) ^ 2

/-- Recursive definition of fₙ(k) -/
def f : ℕ → ℕ → ℕ
  | 0, k => k
  | 1, k => f₁ k
  | n + 1, k => f₁ (f n k)

/-- Main theorem to prove -/
theorem f_1991_of_2_1990 : f 1991 (2^1990) = 256 := by
  sorry

#eval f 1991 (2^1990)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_1991_of_2_1990_l41_4155


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l41_4152

structure Ellipse where
  a : ℝ
  b : ℝ
  h_positive : 0 < b ∧ b < a

noncomputable def slope_product (e : Ellipse) : ℝ := -1/2

noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

noncomputable def triangle_area (e : Ellipse) (m n : ℝ × ℝ) : ℝ :=
  (1/2) * abs ((m.1 * n.2) - (n.1 * m.2))

theorem ellipse_properties (e : Ellipse) :
  eccentricity e = Real.sqrt 2 / 2 ∧
  (e.b = 1 →
    ∃ (max_area : ℝ),
      max_area = Real.sqrt 2 / 2 ∧
      ∀ (m n : ℝ × ℝ),
        m.1^2 / (2 * e.a^2) + m.2^2 = 1 →
        n.1^2 / (2 * e.a^2) + n.2^2 = 1 →
        ∃ (k : ℝ), m.1 = k * m.2 - 1 ∧ n.1 = k * n.2 - 1 →
        triangle_area e m n ≤ max_area) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l41_4152


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l41_4196

-- Define the triangle vertices
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, 7)
def C : ℝ × ℝ := (4, 3)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the side lengths
noncomputable def AB : ℝ := distance A B
noncomputable def AC : ℝ := distance A C
noncomputable def BC : ℝ := distance B C

-- Define the perimeter
noncomputable def perimeter : ℝ := AB + AC + BC

-- Theorem statement
theorem triangle_properties :
  (max AB (max AC BC) = 5) ∧ (perimeter = 12) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l41_4196


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_red_coloring_l41_4198

theorem unique_red_coloring (n : ℕ) (h : n > 2) :
  ∃! S : Finset ℕ,
    (S ⊆ Finset.range (2 * n + 1)) ∧
    (S.card = n + 1) ∧
    (∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z → x + y ≠ z) ∧
    (∀ k, k ∈ Finset.range (n + 1) → k + n ∈ S) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_red_coloring_l41_4198


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l41_4194

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem symmetry_axis_of_f :
  ∀ x : ℝ, f (π / 6 + x) = f (π / 6 - x) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axis_of_f_l41_4194


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_specific_point_l41_4173

/-- Conversion from rectangular to cylindrical coordinates -/
noncomputable def rectangular_to_cylindrical (x y z : ℝ) : ℝ × ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := if x > 0 then Real.arctan (y / x)
           else if x < 0 then Real.arctan (y / x) + Real.pi
           else if y ≥ 0 then Real.pi / 2
           else 3 * Real.pi / 2
  (r, θ, z)

theorem rectangular_to_cylindrical_specific_point :
  let (r, θ, z) := rectangular_to_cylindrical 3 (-3 * Real.sqrt 3) 2
  r = 6 ∧ θ = 5 * Real.pi / 3 ∧ z = 2 ∧ r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangular_to_cylindrical_specific_point_l41_4173


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_covering_progressions_all_numbers_covered_l41_4128

-- Define the type for arithmetic progressions
def ArithmeticProgression (a₀ d : ℕ) : Set ℕ :=
  {n : ℕ | ∃ k : ℕ, n = a₀ + k * d}

-- Define a function that generates N-1 arithmetic progressions
noncomputable def generateProgressions (N : ℕ) : List (Set ℕ) :=
  (List.range (N - 1)).filter (· ≠ 0) |>.map (λ d => ArithmeticProgression d d)

-- The main theorem
theorem exists_covering_progressions :
  ∃ N : ℕ, N > 1 ∧ (∀ n : ℕ, ∃ AP ∈ generateProgressions N, n ∈ AP) := by
  -- We claim that N = 12 works
  use 12
  constructor
  · norm_num -- proves 12 > 1
  · intro n
    -- We'll prove this later
    sorry

-- Helper theorem: every natural number is covered by one of the progressions
theorem all_numbers_covered (n : ℕ) : 
  ∃ AP ∈ generateProgressions 12, n ∈ AP := by
  -- We'll prove this later
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_covering_progressions_all_numbers_covered_l41_4128


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_l41_4190

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  first_positive : a 1 > 0
  diff_negative : d < 0
  is_arithmetic : ∀ n : ℕ, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (seq.a 1 + seq.a n)

theorem max_positive_sum (seq : ArithmeticSequence) 
  (h : sum_n seq 4 = sum_n seq 8) :
  (∀ n : ℕ, n ≤ 11 → sum_n seq n > 0) ∧
  sum_n seq 12 = 0 ∧
  (∀ n : ℕ, n > 11 → sum_n seq n ≤ 0) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_positive_sum_l41_4190


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_correct_l41_4141

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The given set of points from the experiment -/
def experimentPoints : List Point := [
  { x := 1, y := 3 },
  { x := 2, y := 3.8 },
  { x := 3, y := 5.2 },
  { x := 4, y := 6 }
]

/-- The proposed regression line equation -/
def regressionLine (x : ℝ) : ℝ := 1.04 * x + 1.9

/-- Theorem stating that the given regression line is correct for the experiment points -/
theorem regression_line_correct :
  ∀ p ∈ experimentPoints, |regressionLine p.x - p.y| < 0.1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_line_correct_l41_4141


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_largest_plus_second_smallest_odd_l41_4110

def odd_numbers_1_to_15 : List Nat := [1, 3, 5, 7, 9, 11, 13, 15]

theorem second_largest_plus_second_smallest_odd : 
  (List.get! (List.reverse odd_numbers_1_to_15) 1 + List.get! odd_numbers_1_to_15 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_largest_plus_second_smallest_odd_l41_4110


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l41_4129

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * Real.log x

noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a / x

theorem f_properties (a : ℝ) :
  a > 0 →
  (∃! x : ℝ, x > 0 ∧ f_deriv a x = 0) ∧
  (∀ x : ℝ, x > 0 → f a x ≥ a * (2 - Real.log a)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l41_4129


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l41_4154

noncomputable def triangle_vertex_1 : ℝ × ℝ := (3, 1)
noncomputable def triangle_vertex_2 : ℝ × ℝ := (7, 5)
noncomputable def triangle_vertex_3 : ℝ × ℝ := (8, 1)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def side1 : ℝ := distance triangle_vertex_1 triangle_vertex_2
noncomputable def side2 : ℝ := distance triangle_vertex_1 triangle_vertex_3
noncomputable def side3 : ℝ := distance triangle_vertex_2 triangle_vertex_3

theorem longest_side_length :
  max side1 (max side2 side3) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_longest_side_length_l41_4154


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_two_zeros_condition_l41_4135

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 2 * Real.log x + 2

-- Theorem 1: f(x) has an extremum at x = 1 iff a = 2
theorem extremum_condition (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a 1 ≤ f a x) ↔ a = 2 :=
by sorry

-- Theorem 2: f(x) has exactly two zeros iff 0 < a < 2/Real.exp 2
theorem two_zeros_condition (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ 
   ∀ z : ℝ, f a z = 0 → z = x ∨ z = y) ↔
  0 < a ∧ a < 2 / Real.exp 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extremum_condition_two_zeros_condition_l41_4135


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_three_l41_4149

theorem largest_among_three : (∀ x ∈ ({10, 11, 12} : Set ℕ), x ≤ 12) ∧ 12 ∈ ({10, 11, 12} : Set ℕ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_among_three_l41_4149


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_27_29_26_l41_4181

/-- The area of a triangle given two sides and a median to the third side -/
noncomputable def triangleArea (a b m : ℝ) : ℝ :=
  let s := (a + b + 2 * m) / 2
  2 * Real.sqrt (s * (s - a) * (s - b) * (s - 2 * m))

/-- Theorem: The area of a triangle with sides 27 and 29, and median 26 to the third side, is 270 -/
theorem triangle_area_27_29_26 :
  triangleArea 27 29 26 = 270 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_27_29_26_l41_4181


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_equality_l41_4137

theorem cube_root_fraction_equality : 
  (8 : ℝ) / 12.75 = ((32 : ℝ) / 51) ^ (1/3 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_fraction_equality_l41_4137


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_integer_solution_l41_4176

-- Define the circle
def circle_contains (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 ≤ 36

-- Define the point (x, 2x)
def point (x : ℝ) : ℝ × ℝ := (x, 2*x)

-- Define the condition for an integer x to satisfy the circle equation
def satisfies_circle (x : ℤ) : Prop := circle_contains (↑x) (2 * ↑x)

-- Theorem statement
theorem exactly_one_integer_solution :
  ∃! (x : ℤ), satisfies_circle x :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_one_integer_solution_l41_4176


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_l41_4113

open Real

noncomputable def f (x : ℝ) : ℝ := 2 + log ((1 + x) / (1 - x))

theorem max_min_sum (M m : ℝ) :
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f x ≤ M ∧ m ≤ f x) →
  (∃ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f x = M) →
  (∃ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f x = m) →
  M + m = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_sum_l41_4113


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_y_l41_4182

-- Define the function y
noncomputable def y (k₁ k₂ x : ℝ) : ℝ := k₁ * x^2 + k₂ / x^2

-- Define the conditions
def condition1 (k₁ k₂ : ℝ) : Prop := y k₁ k₂ 1 = 5
def condition2 (k₁ k₂ : ℝ) : Prop := y k₁ k₂ (Real.sqrt 3) = 7

-- Theorem statement
theorem minimize_y (k₁ k₂ : ℝ) (h1 : condition1 k₁ k₂) (h2 : condition2 k₁ k₂) :
  ∃ (x : ℝ), x > 0 ∧ ∀ (t : ℝ), t > 0 → y k₁ k₂ x ≤ y k₁ k₂ t ∧ x = (3/2)^(1/4) := by
  sorry

#check minimize_y

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_y_l41_4182


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l41_4178

/-- A point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line segment connecting two points -/
structure Segment where
  start : Point
  finish : Point

/-- The theorem statement -/
theorem triangle_existence (N : ℕ) (points : Finset Point) (segments : Finset Segment) :
  (points.card = 2 * N) →
  (segments.card = N^2 + 1) →
  (∀ s ∈ segments, s.start ∈ points ∧ s.finish ∈ points) →
  ∃ p1 p2 p3 : Point,
    p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
    Segment.mk p1 p2 ∈ segments ∧
    Segment.mk p2 p3 ∈ segments ∧
    Segment.mk p3 p1 ∈ segments :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_l41_4178


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_catches_rabbit_at_quarter_circle_l41_4138

/-- A circular yard with a dog and a rabbit -/
structure YardSetup where
  /-- The radius of the circular yard -/
  radius : ℝ
  /-- The speed at which both the dog and rabbit run -/
  speed : ℝ
  /-- Assumption that the radius and speed are positive -/
  radius_pos : radius > 0
  speed_pos : speed > 0

/-- The position of the rabbit on the circle's circumference -/
noncomputable def rabbitPosition (t : ℝ) (setup : YardSetup) : ℝ := 
  (setup.speed * t) % (2 * Real.pi * setup.radius)

/-- The distance between the dog and the center of the yard -/
noncomputable def dogPosition (t : ℝ) (setup : YardSetup) : ℝ := 
  min (setup.speed * t) setup.radius

/-- The theorem stating that the dog catches the rabbit at quarter circle -/
theorem dog_catches_rabbit_at_quarter_circle (setup : YardSetup) : 
  ∃ t : ℝ, t > 0 ∧ dogPosition t setup = setup.radius ∧ 
    rabbitPosition t setup = Real.pi * setup.radius / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dog_catches_rabbit_at_quarter_circle_l41_4138


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_multiples_of_seven_l41_4163

def a : ℕ → ℕ
  | 0 => 1
  | n+1 => a n + a ((n+1) / 2)

theorem infinitely_many_multiples_of_seven :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ 7 ∣ a n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_multiples_of_seven_l41_4163


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l41_4151

/-- Curve C1 in Cartesian coordinates -/
def C1 (x y : ℝ) : Prop :=
  (x - 4)^2 + (y - 5)^2 = 25

/-- Curve C2 in polar coordinates -/
def C2 (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sin θ

/-- Conversion from polar to Cartesian coordinates -/
noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

/-- Theorem stating the intersection points of C1 and C2 -/
theorem intersection_points :
  ∃ (ρ1 θ1 ρ2 θ2 : ℝ),
    C2 ρ1 θ1 ∧ C2 ρ2 θ2 ∧
    C1 (polar_to_cartesian ρ1 θ1).1 (polar_to_cartesian ρ1 θ1).2 ∧
    C1 (polar_to_cartesian ρ2 θ2).1 (polar_to_cartesian ρ2 θ2).2 ∧
    ρ1 = Real.sqrt 2 ∧ θ1 = π / 4 ∧
    ρ2 = 2 ∧ θ2 = π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_l41_4151


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_profit_solution_l41_4179

/-- The problem of calculating the number of oranges to sell for a specific profit --/
def orange_profit_problem (buy_quantity : ℕ) (buy_price : ℚ) 
  (sell_quantity : ℕ) (sell_price : ℚ) (target_profit : ℚ) : ℕ :=
  let cost_per_orange : ℚ := buy_price / buy_quantity
  let revenue_per_orange : ℚ := sell_price / sell_quantity
  let profit_per_orange : ℚ := revenue_per_orange - cost_per_orange
  let oranges_needed : ℚ := target_profit / profit_per_orange
  (oranges_needed.ceil.toNat)

/-- The specific instance of the orange profit problem --/
theorem orange_profit_solution : 
  orange_profit_problem 4 15 6 25 200 = 477 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_orange_profit_solution_l41_4179


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_problem_l41_4172

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

noncomputable def g (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := f ω φ x + 1

theorem sine_function_problem (ω φ : ℝ) :
  ω > 0 →
  φ ∈ Set.Icc (-Real.pi / 2) 0 →
  (∀ x : ℝ, f ω φ (x + Real.pi) = f ω φ x) →
  (∀ x ∈ Set.Ioo (-Real.pi / 3) (-Real.pi / 12), g ω φ x < 1) →
  g ω (-Real.pi / 3) (Real.pi / 4) = 3 / 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_function_problem_l41_4172


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l41_4140

def sequence_a (n : ℕ) (lambda : ℝ) : ℝ := n^2 + lambda * n

theorem lambda_range (lambda : ℝ) :
  (∀ n : ℕ, sequence_a n lambda < sequence_a (n + 1) lambda) →
  lambda > -3 ∧ lambda ∈ Set.Ioi (-3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lambda_range_l41_4140


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_is_6_8_l41_4199

/-- Represents the flagpole scenario -/
structure FlagpoleScenario where
  wire_ground_distance : ℝ
  person_distance : ℝ
  person_height : ℝ

/-- Calculates the height of the flagpole given the scenario -/
noncomputable def calculate_flagpole_height (scenario : FlagpoleScenario) : ℝ :=
  scenario.wire_ground_distance * scenario.person_height /
    (scenario.wire_ground_distance - scenario.person_distance)

/-- Theorem stating that the flagpole height is 6.8 meters given the specific conditions -/
theorem flagpole_height_is_6_8 :
  let scenario : FlagpoleScenario := {
    wire_ground_distance := 4,
    person_distance := 3,
    person_height := 1.7
  }
  calculate_flagpole_height scenario = 6.8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flagpole_height_is_6_8_l41_4199


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_wins_by_20_meters_l41_4139

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents a race with two runners -/
structure Race where
  distance : ℝ
  sunny : Runner
  misty : Runner

/-- The first race where Sunny finishes 20 meters ahead of Misty -/
noncomputable def first_race : Race :=
  { distance := 400,
    sunny := { speed := 400 / 380 },
    misty := { speed := 1 } }

/-- The second race where Sunny starts 40 meters behind Misty -/
def second_race (r : Race) : Race := r

/-- Calculate the distance a runner covers in a given time -/
noncomputable def distance_covered (runner : Runner) (time : ℝ) : ℝ :=
  runner.speed * time

/-- Calculate the time it takes for Sunny to finish the second race -/
noncomputable def sunny_finish_time (r : Race) : ℝ :=
  (r.distance + 40) / r.sunny.speed

/-- Calculate Sunny's lead at the end of the second race -/
noncomputable def sunny_lead (r : Race) : ℝ :=
  r.distance + 40 - distance_covered r.misty (sunny_finish_time r)

/-- Theorem stating that Sunny finishes 20 meters ahead in the second race -/
theorem sunny_wins_by_20_meters (r : Race) :
  sunny_lead (second_race r) = 20 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_wins_by_20_meters_l41_4139


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_invalidTransmission_l41_4122

-- Define our own XOR operation to avoid naming conflict
def myXor (a b : Bool) : Bool :=
  (a && !b) || (!a && b)

-- Define the transmission information generation function
def generateTransmission (a a₁ a₂ : Bool) : Bool × Bool × Bool × Bool × Bool :=
  let h := myXor a a₁
  let h₁ := myXor h a₂
  (h, a, a₁, a₂, h₁)

-- Theorem statement
theorem invalidTransmission :
  ∀ (a a₁ a₂ : Bool), generateTransmission a a₁ a₂ ≠ (true, false, true, true, true) :=
by
  intro a a₁ a₂
  simp [generateTransmission, myXor]
  sorry -- Placeholder for the actual proof


end NUMINAMATH_CALUDE_ERRORFEEDBACK_invalidTransmission_l41_4122


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_perpendicular_l41_4133

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a triangle
structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define reflection across y=x
def reflect (p : Point) : Point :=
  ⟨p.y, p.x⟩

-- Define the slope of a line given two points
noncomputable def slopeLine (p1 p2 : Point) : ℝ :=
  (p2.y - p1.y) / (p2.x - p1.x)

-- Define perpendicularity of two lines given by two pairs of points
def perpendicular (p1 p2 q1 q2 : Point) : Prop :=
  slopeLine p1 p2 * slopeLine q1 q2 = -1

-- Theorem statement
theorem not_always_perpendicular :
  ¬ ∀ (ABC : Triangle), 
    (ABC.A.x ≥ 0 ∧ ABC.A.y ≥ 0) → 
    (ABC.B.x ≥ 0 ∧ ABC.B.y ≥ 0) → 
    (ABC.C.x ≥ 0 ∧ ABC.C.y ≥ 0) → 
    ABC.A.x ≠ ABC.A.y → 
    ABC.B.x ≠ ABC.B.y → 
    ABC.C.x ≠ ABC.C.y → 
    perpendicular ABC.A ABC.B (reflect ABC.A) (reflect ABC.B) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_always_perpendicular_l41_4133


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l41_4106

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2^x - 3 else Real.sqrt (x + 1)

-- Theorem statement
theorem range_of_a (a : ℝ) :
  f a > 1 ↔ a ∈ Set.Ioi 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l41_4106


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l41_4191

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the given conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * Real.sqrt 3 * (Real.cos (t.C / 2))^2 = Real.sin t.C + Real.sqrt 3 + 1 ∧
  t.a = 2 * Real.sqrt 3 ∧
  t.c = 2

-- Theorem statement
theorem triangle_theorem (t : Triangle) (h : triangle_conditions t) :
  t.C = Real.pi / 6 ∧ (t.b = 2 ∨ t.b = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l41_4191


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_g_min_value_g_max_value_l41_4104

noncomputable section

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 2/x

-- Define the function g (renamed from y for clarity)
noncomputable def g (x : ℝ) : ℝ := 2 * (x^2 + x) / (x - 1)

-- Theorem for the decreasing property of f
theorem f_decreasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → x₂ < Real.sqrt 2 → f x₁ > f x₂ := by sorry

-- Theorem for the minimum value of g
theorem g_min_value :
  ∃ x : ℝ, 2 ≤ x ∧ x < 4 ∧ g x = 2 * (3 + 2 * Real.sqrt 2) ∧
  ∀ y : ℝ, 2 ≤ y ∧ y < 4 → g y ≥ 2 * (3 + 2 * Real.sqrt 2) := by sorry

-- Theorem for the maximum value of g
theorem g_max_value :
  ∃ x : ℝ, 2 ≤ x ∧ x < 4 ∧ g x = 40/3 ∧
  ∀ y : ℝ, 2 ≤ y ∧ y < 4 → g y ≤ 40/3 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_g_min_value_g_max_value_l41_4104


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_positive_l41_4166

/-- Transformation function for the sequence -/
def transform (t : ℝ × ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ × ℝ :=
  let (a, b, c, d) := t
  (a * b, b * c, c * d, d * a)

/-- Predicate to check if all numbers in a tuple are positive -/
def all_positive (t : ℝ × ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c, d) := t
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

/-- Main theorem: Eventually, all numbers become positive -/
theorem eventually_positive (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  ∃ n : ℕ, all_positive (Nat.iterate transform n (a, b, c, d)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_positive_l41_4166


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l41_4153

theorem problem_solution : 
  (|Real.sqrt 3 - Real.sqrt 2| + Real.sqrt 2 = Real.sqrt 3) ∧
  (Real.sqrt 2 * (Real.sqrt 2 + 2) = 2 + 2 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l41_4153


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l41_4186

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (4 * x + Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

def is_axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

theorem axis_of_symmetry_g :
  is_axis_of_symmetry g (Real.pi / 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_g_l41_4186


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_lanes_l41_4103

/-- Given a road with lanes, trucks, and cars, prove the number of lanes. -/
theorem number_of_lanes (total_vehicles trucks_per_lane : ℕ) : ℕ :=
  let num_lanes := 12
  have h1 : total_vehicles = 2160 := by sorry
  have h2 : trucks_per_lane = 60 := by sorry
  have h3 : ∀ n : ℕ, 2 * (trucks_per_lane * n) = trucks_per_lane * n + 2 * trucks_per_lane * n := by sorry
  have h4 : total_vehicles = trucks_per_lane * num_lanes + 2 * (trucks_per_lane * num_lanes) := by
    rw [h1, h2]
    norm_num
  num_lanes

#check number_of_lanes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_lanes_l41_4103


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_product_fourth_root_l41_4157

theorem power_of_two_product_fourth_root : (2^8 * 2^12 : ℝ)^(1/4) = 32 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_of_two_product_fourth_root_l41_4157


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l41_4107

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 7 * x - 8) / (x^2 - 5 * x + 3)

-- Define the horizontal asymptote
def horizontal_asymptote : ℝ := 3

-- Theorem statement
theorem g_crosses_asymptote :
  ∃ x : ℝ, g x = horizontal_asymptote ∧ x = 17/8 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_crosses_asymptote_l41_4107


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_satisfying_conditions_l41_4146

noncomputable def a (n : ℕ) : ℕ := ⌊Real.sqrt (n^2 + (n+1)^2)⌋.toNat

theorem infinitely_many_n_satisfying_conditions :
  Set.Infinite {n : ℕ | n ≥ 1 ∧ a n - a (n-1) > 1 ∧ a (n+1) - a n = 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_n_satisfying_conditions_l41_4146


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l41_4160

noncomputable section

-- Define the parabola and hyperbola
def parabola (p : ℝ) (x : ℝ) : ℝ := p * x - x^2
def hyperbola (q : ℝ) (x : ℝ) : ℝ := q / x

-- Define the intersection points
structure IntersectionPoint where
  x : ℝ
  y : ℝ

-- Define the theorem
theorem intersection_product (p q : ℝ) (A B C : IntersectionPoint) : 
  (∀ (x : ℝ), x ≠ 0 → parabola p x = hyperbola q x) →  -- Intersection condition
  (A.x ≠ B.x ∧ B.x ≠ C.x ∧ C.x ≠ A.x) →  -- Distinct points condition
  ((A.x - B.x)^2 + (B.x - C.x)^2 + (C.x - A.x)^2 + 
   (A.y - B.y)^2 + (B.y - C.y)^2 + (C.y - A.y)^2 = 324) →  -- Sum of squares condition
  ((A.x + B.x + C.x)^2 / 9 + (A.y + B.y + C.y)^2 / 9 = 4) →  -- Median intersection distance condition
  p * q = 42 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_product_l41_4160


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_distance_l41_4158

-- Define a point in the plane with integer coordinates
structure Point where
  x : ℤ
  y : ℤ

-- Define the distance between two points
noncomputable def distance (p q : Point) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

-- Define the property of three points being noncollinear
def noncollinear (a b c : Point) : Prop :=
  (b.x - a.x) * (c.y - a.y) ≠ (c.x - a.x) * (b.y - a.y)

-- State the theorem
theorem smallest_integer_distance (a b c : Point) :
  noncollinear a b c →
  (∃ m : ℤ, distance a b = m) ∧ 
  (∃ n : ℤ, distance a c = n) ∧ 
  (∃ k : ℤ, distance b c = k) →
  3 ≤ distance a b ∧ ∃ (a' b' c' : Point), distance a' b' = 3 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_integer_distance_l41_4158


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_to_banana_cost_ratio_l41_4145

/-- The ratio of muffin cost to banana cost given Susie and Calvin's purchases -/
theorem muffin_to_banana_cost_ratio
  (muffin_cost banana_cost cookie_cost : ℚ)
  (susie_purchase : 5 * muffin_cost + 4 * banana_cost + 2 * cookie_cost > 0)
  (calvin_purchase : 3 * muffin_cost + 20 * banana_cost + 6 * cookie_cost = 3 * (5 * muffin_cost + 4 * banana_cost + 2 * cookie_cost))
  (cookie_banana_ratio : cookie_cost = 2 * banana_cost) :
  muffin_cost / banana_cost = 4 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_muffin_to_banana_cost_ratio_l41_4145


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_match_probabilities_l41_4111

/-- Represents a volleyball team -/
inductive Team : Type
| A
| B

/-- Represents the outcome of a set -/
inductive SetOutcome : Type
| Win
| Loss

/-- Represents the state of the match after three sets -/
structure MatchState :=
(team_a_sets : Nat)
(team_b_sets : Nat)

/-- Represents the state of the deciding set -/
structure DecidingSetState :=
(team_a_points : Nat)
(team_b_points : Nat)
(serving_team : Team)

/-- The probability of a team winning a set -/
def set_win_probability : ℚ := 1/2

/-- The probability of Team A scoring when serving -/
def team_a_serve_probability : ℚ := 2/5

/-- The probability of Team A scoring when receiving -/
def team_a_receive_probability : ℚ := 3/5

/-- The initial state of the match after three sets -/
def initial_match_state : MatchState :=
{ team_a_sets := 2, team_b_sets := 1 }

/-- The state of the deciding set when both teams have 14 points and Team A is serving -/
def initial_deciding_set_state : DecidingSetState :=
{ team_a_points := 14, team_b_points := 14, serving_team := Team.A }

/-- Placeholder function for calculating the probability of winning the match -/
def probability_of_winning_match (team : Team) (state : MatchState) : ℚ := 0

/-- Placeholder function for calculating the probability of winning in a specific number of rallies -/
def probability_of_winning_in_rallies (team : Team) (state : DecidingSetState) (rallies : Nat) : ℚ := 0

/-- The theorem to be proved -/
theorem volleyball_match_probabilities :
  let p_win_match := 3/4
  let p_win_2_rallies := 4/25
  let p_win_4_rallies := 72/625
  (∀ (state : MatchState), state = initial_match_state →
    probability_of_winning_match Team.A state = p_win_match) ∧
  (∀ (state : DecidingSetState), state = initial_deciding_set_state →
    probability_of_winning_in_rallies Team.A state 2 = p_win_2_rallies) ∧
  (∀ (state : DecidingSetState), state = initial_deciding_set_state →
    probability_of_winning_in_rallies Team.A state 4 = p_win_4_rallies) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_volleyball_match_probabilities_l41_4111
