import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l674_67425

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (-x^3 + x - 1) / (2*(1-x)*x)

-- State the theorem
theorem function_satisfies_equation (x : ℝ) (h : x ≠ 0 ∧ x ≠ 1) : 
  f x + f (1 / (1 - x)) = x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_satisfies_equation_l674_67425


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_calculation_l674_67450

theorem absolute_value_calculation : 
  |Real.sqrt 3 - 2| - Real.sqrt ((-2)^2) - (64 : ℝ) ^ (1/3) = -4 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_absolute_value_calculation_l674_67450


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l674_67495

/-- Represents a 10x10 game board -/
def GameBoard := Fin 10 → Fin 10 → Bool

/-- Represents a domino placement on the board -/
structure DominoPlacement where
  row : Fin 10
  col : Fin 10
  horizontal : Bool

/-- Checks if a domino placement is valid on the given board -/
def isValidPlacement (board : GameBoard) (placement : DominoPlacement) : Prop :=
  if placement.horizontal then
    ∃ (nextCol : Fin 10), nextCol.val = placement.col.val + 1 ∧
    ¬board placement.row placement.col ∧
    ¬board placement.row nextCol
  else
    ∃ (nextRow : Fin 10), nextRow.val = placement.row.val + 1 ∧
    ¬board placement.row placement.col ∧
    ¬board nextRow placement.col

/-- Applies a domino placement to the board -/
def applyPlacement (board : GameBoard) (placement : DominoPlacement) : GameBoard :=
  fun r c =>
    if r = placement.row ∧ c = placement.col then true
    else if placement.horizontal ∧ r = placement.row ∧ c.val = placement.col.val + 1 then true
    else if ¬placement.horizontal ∧ r.val = placement.row.val + 1 ∧ c = placement.col then true
    else board r c

/-- Represents the game state -/
structure GameState where
  board : GameBoard
  currentPlayer : Bool  -- true for first player, false for second player

/-- The winning strategy for the second player -/
noncomputable def secondPlayerStrategy (state : GameState) (firstPlayerMove : DominoPlacement) : DominoPlacement :=
  sorry  -- The actual strategy implementation would go here

/-- Theorem stating that the second player has a winning strategy -/
theorem second_player_wins :
  ∀ (initialState : GameState),
  initialState.currentPlayer = true →
  ∃ (strategy : GameState → DominoPlacement → DominoPlacement),
  ∀ (game : ℕ → GameState),
  game 0 = initialState →
  (∀ n : ℕ, 
    let currentState := game n
    let nextState := game (n + 1)
    if currentState.currentPlayer
    then 
      ∃ move, isValidPlacement currentState.board move ∧
      nextState.board = applyPlacement currentState.board move ∧
      nextState.currentPlayer = false
    else
      ∃ prevMove, 
      let responseMove := strategy currentState prevMove
      isValidPlacement currentState.board responseMove ∧
      nextState.board = applyPlacement currentState.board responseMove ∧
      nextState.currentPlayer = true) →
  ∃ n : ℕ, ¬∃ move, isValidPlacement (game n).board move ∧ (game n).currentPlayer = true :=
by
  sorry  -- The proof goes here


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_player_wins_l674_67495


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_height_l674_67456

/-- The height of a right circular cylinder inscribed in a hemisphere -/
noncomputable def cylinder_height (hemisphere_radius : ℝ) (cylinder_radius : ℝ) : ℝ :=
  Real.sqrt (hemisphere_radius^2 - cylinder_radius^2)

/-- Theorem: The height of the inscribed cylinder is 3√5 -/
theorem inscribed_cylinder_height :
  let hemisphere_radius : ℝ := 7
  let cylinder_radius : ℝ := 2
  cylinder_height hemisphere_radius cylinder_radius = 3 * Real.sqrt 5 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_height_l674_67456


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_superb_sum_theorem_l674_67421

/-- A positive integer is superb if it is the least common multiple of 1,2,...,n for some positive integer n. -/
def IsSuperb (m : ℕ) : Prop :=
  ∃ n : ℕ, m = Finset.lcm (Finset.range n.succ) 1

/-- The theorem stating the only superb numbers x, y, and z that satisfy x + y = z. -/
theorem superb_sum_theorem :
  ∀ x y z : ℕ,
  IsSuperb x ∧ IsSuperb y ∧ IsSuperb z ∧ x + y = z →
  ∃ n : ℕ,
    x = Finset.lcm (Finset.range ((2^n : ℕ) - 1)) 1 ∧
    y = Finset.lcm (Finset.range ((2^n : ℕ) - 1)) 1 ∧
    z = Finset.lcm (Finset.range (2^n : ℕ)) 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_superb_sum_theorem_l674_67421


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_330_degrees_l674_67452

theorem cos_330_degrees : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_330_degrees_l674_67452


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l674_67429

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- positive terms
  (∃ q > 0, ∀ k, a (k + 1) = q * a k) →  -- geometric sequence
  a 6 = a 5 + 2 * a 4 →  -- given condition
  Real.sqrt (a m * a n) = 2 * a 1 →  -- given condition
  (∀ p q : ℝ, p > 0 → q > 0 → 1 / p + 9 / q ≥ 4) ∧ 
  (∃ p q : ℝ, p > 0 ∧ q > 0 ∧ 1 / p + 9 / q = 4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_minimum_l674_67429


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_range_l674_67491

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℕ+) : ℝ := a * (x : ℝ) - 20
noncomputable def g (a : ℝ) (x : ℕ+) : ℝ := Real.log ((x : ℝ) / a) / Real.log 10

-- Define harmonious functions
def harmonious (f g : ℕ+ → ℝ) : Prop :=
  ∀ x : ℕ+, f x * g x ≥ 0

-- State the theorem
theorem harmonious_range (a : ℝ) :
  (a > 0) →
  (harmonious (f a) (g a)) ↔ 
  (4 ≤ a ∧ a ≤ 5) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harmonious_range_l674_67491


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_ratio_l674_67490

noncomputable def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

noncomputable def asymptote_angle (a b : ℝ) : ℝ := 2 * Real.arctan (b / a)

theorem hyperbola_asymptote_ratio (a b : ℝ) (h1 : a > b) (h2 : a > 0) (h3 : b > 0) :
  asymptote_angle a b = π / 2 → a / b = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptote_ratio_l674_67490


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_is_one_l674_67412

noncomputable def product_of_square_roots (a b : ℝ) : ℝ :=
  let P := Real.sqrt a + Real.sqrt b
  let Q := -Real.sqrt a - Real.sqrt b
  let R := Real.sqrt a - Real.sqrt b
  let S := Real.sqrt b - Real.sqrt a
  P * Q * R * S

theorem product_is_one : 
  product_of_square_roots 2011 2012 = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_is_one_l674_67412


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l674_67446

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp (2 * x) + Real.exp 2) - x

-- State the theorem
theorem f_inequality : f (1/3) > f (Real.exp (1/3)) ∧ f (Real.exp (1/3)) > f (4/3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_l674_67446


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statement_l674_67410

/-- Represents whether a student completed the extra credit project -/
def completed_extra_credit : Prop := sorry

/-- Represents whether a student completed all homework -/
def completed_all_homework : Prop := sorry

/-- Represents whether a student passed the course -/
def passed_course : Prop := sorry

/-- Mr. Hamilton's statement -/
axiom hamilton_statement : (completed_extra_credit ∧ completed_all_homework) → passed_course

/-- The statement we want to prove -/
theorem correct_statement : 
  ¬passed_course → (¬completed_all_homework ∨ ¬completed_extra_credit) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statement_l674_67410


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l674_67473

-- Define the quadratic function with parameters
noncomputable def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c

-- State the theorem
theorem quadratic_properties (b c : ℝ) :
  (f b c 1 = 0) → (f b c 3 = -4) → (f b c 5 = 0) →
  (∀ x, f b c x ≥ f b c 3) →
  (∃ h, ∀ x, f b c x = (x - 3)^2 + h) ∧
  (∀ x, f b c x > 0 ↔ (x < 1 ∨ x > 5)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_properties_l674_67473


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_common_multiple_first_ten_l674_67458

def first_ten_integers : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem least_common_multiple_first_ten : 
  ∃ n : Nat, (∀ i ∈ first_ten_integers, i ∣ n) ∧ 
  (∀ m : Nat, m > 0 → (∀ i ∈ first_ten_integers, i ∣ m) → n ≤ m) ∧ 
  n = 2520 := by
  sorry

#check least_common_multiple_first_ten

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_common_multiple_first_ten_l674_67458


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_time_is_seven_hours_l674_67408

/-- Represents the number of hours in a day -/
noncomputable def total_hours : ℝ := 24

/-- Represents the fraction of the day spent sleeping -/
noncomputable def sleeping_fraction : ℝ := 1/3

/-- Represents the fraction of the day spent studying -/
noncomputable def studying_fraction : ℝ := 1/4

/-- Represents the fraction of the day spent eating -/
noncomputable def eating_fraction : ℝ := 1/8

/-- Theorem stating that the remaining time in the day is 7 hours -/
theorem remaining_time_is_seven_hours :
  total_hours * (1 - (sleeping_fraction + studying_fraction + eating_fraction)) = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_time_is_seven_hours_l674_67408


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_of_angles_l674_67463

theorem tan_difference_of_angles (α β : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π)
  (h4 : Real.cos α * Real.cos β = 1/6)
  (h5 : Real.sin α * Real.sin β = 1/3) : 
  Real.tan (β - α) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_difference_of_angles_l674_67463


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_periodic_mod_l674_67460

/-- Definition of the sequence a_n -/
def a (n : ℕ+) : ℕ := (n : ℕ)^(n : ℕ) + (n - 1)^((n : ℕ) + 1)

/-- Theorem stating that the sequence is eventually periodic modulo any positive integer -/
theorem eventually_periodic_mod (m : ℕ+) :
  ∃ (K s : ℕ+), ∀ (k : ℕ+), k ≥ K → a k % m = a (k + s) % m :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_eventually_periodic_mod_l674_67460


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l674_67471

theorem cube_root_equation_solution :
  ∃ (d e f : ℕ+),
    4 * Real.sqrt (Real.rpow 7 (1/3) - Real.rpow 3 (1/3)) = 
    Real.rpow d.val (1/3) + Real.rpow e.val (1/3) - Real.rpow f.val (1/3) ∧
    d.val + e.val + f.val = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_solution_l674_67471


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l674_67470

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := (x^2 - 5*x + 6) / (x - 1)

-- State the theorem
theorem f_properties :
  -- Domain is all real numbers except 1
  (∀ x : ℝ, x ≠ 1 → f x ∈ Set.univ) ∧
  -- Zeros are 2 and 3
  (f 2 = 0 ∧ f 3 = 0) ∧
  -- Vertical asymptote at x = 1
  (∀ ε > (0 : ℝ), ∃ δ > (0 : ℝ), ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ → |f x| > 1/ε) ∧
  -- Function approaches ±∞ as x approaches ±∞
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x > M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → f x < -M) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l674_67470


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tod_trip_time_l674_67486

/-- Represents a segment of Tod's trip -/
structure TripSegment where
  distance : ℝ
  speed : ℝ

/-- Calculates the time taken for a trip segment -/
noncomputable def segmentTime (segment : TripSegment) : ℝ :=
  segment.distance / segment.speed

/-- Tod's trip segments -/
def todTrip : List TripSegment := [
  ⟨55, 40⟩,
  ⟨95, 50⟩,
  ⟨30, 20⟩,
  ⟨75, 60⟩
]

/-- Theorem: The total driving time of Tod's trip is 6.025 hours -/
theorem tod_trip_time : 
  (todTrip.map segmentTime).sum = 6.025 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tod_trip_time_l674_67486


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l674_67474

/-- The function we're minimizing -/
noncomputable def f (x : ℝ) : ℝ := (x^2 + 8) / Real.sqrt (x^2 + 4)

/-- Theorem stating that the minimum value of f is 4 -/
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 4) ∧ (∃ x : ℝ, f x = 4) := by
  sorry

#check f_min_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_min_value_l674_67474


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l674_67406

noncomputable def numbers : List ℝ := [-9.3, 3/100, -20, 0, 0.01, -1, -7/2, 3.14, 100]

def is_positive (x : ℝ) : Prop := x > 0
def is_integer (x : ℝ) : Prop := ∃ n : ℤ, x = n
def is_negative_fraction (x : ℝ) : Prop := x < 0 ∧ ¬(is_integer x)
def is_non_negative (x : ℝ) : Prop := x ≥ 0
def is_natural (x : ℝ) : Prop := ∃ n : ℕ, x = n ∧ n > 0

theorem number_categorization :
  (∀ x ∈ numbers, is_positive x ↔ x ∈ [3/100, 0.01, 3.14, 100]) ∧
  (∀ x ∈ numbers, is_integer x ↔ x ∈ [-20, 0, -1, 100]) ∧
  (∀ x ∈ numbers, is_negative_fraction x ↔ x ∈ [-9.3, -7/2]) ∧
  (∀ x ∈ numbers, is_non_negative x ↔ x ∈ [3/100, 0, 0.01, 3.14, 100]) ∧
  (∀ x ∈ numbers, is_natural x ↔ x ∈ [100]) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_categorization_l674_67406


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_intersection_height_l674_67409

/-- Given two poles with heights and distance between them, calculates the intersection height of lines joining the top of each pole to the foot of the opposite pole -/
noncomputable def intersection_height (h1 h2 d : ℝ) : ℝ :=
  let m1 := (0 - h1) / d
  let m2 := (0 - h2) / (-d)
  let x := (h1 - 0) / (m2 - m1)
  m2 * x

theorem pole_intersection_height :
  let h1 : ℝ := 30
  let h2 : ℝ := 90
  let d : ℝ := 150
  intersection_height h1 h2 d = 22.5 := by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pole_intersection_height_l674_67409


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_test_probability_approx_one_eleventh_l674_67438

/-- The probability of having the disease in the population -/
noncomputable def disease_probability : ℝ := 1 / 500

/-- The probability of not having the disease in the population -/
noncomputable def no_disease_probability : ℝ := 1 - disease_probability

/-- The probability of testing positive given that the person has the disease -/
def true_positive_rate : ℝ := 1

/-- The probability of testing positive given that the person does not have the disease (false positive rate) -/
def false_positive_rate : ℝ := 0.02

/-- The probability of testing positive -/
noncomputable def positive_test_probability : ℝ := 
  true_positive_rate * disease_probability + false_positive_rate * no_disease_probability

/-- The probability that a person who tests positive actually has the disease -/
noncomputable def probability_disease_given_positive : ℝ := 
  (true_positive_rate * disease_probability) / positive_test_probability

theorem disease_test_probability_approx_one_eleventh : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.001 ∧ |probability_disease_given_positive - 1/11| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_disease_test_probability_approx_one_eleventh_l674_67438


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l674_67448

theorem max_value_expression (x : ℝ) (hx : x > 0) :
  (x^2 + 1 - Real.sqrt (x^4 + x^2 + 4)) / x ≤ 0 ∧
  ∃ y : ℝ, y > 0 ∧ (y^2 + 1 - Real.sqrt (y^4 + y^2 + 4)) / y = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_expression_l674_67448


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_point_y_coordinate_l674_67417

/-- Given a line passing through points (-1, -4) and (5, y), with slope 0.8, 
    prove that the y-coordinate of the second point (y) is equal to 0.8 -/
theorem second_point_y_coordinate 
  (line : Set (ℝ × ℝ))
  (point1 : ℝ × ℝ)
  (point2 : ℝ × ℝ)
  (slope : ℝ)
  (h1 : point1 = (-1, -4))
  (h2 : point2.1 = 5)
  (h3 : slope = 0.8)
  (h4 : point1 ∈ line)
  (h5 : point2 ∈ line)
  (h6 : slope = (point2.2 - point1.2) / (point2.1 - point1.1)) :
  point2.2 = 0.8 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_point_y_coordinate_l674_67417


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l674_67444

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2*x - 3 * Real.log x

-- State the theorem
theorem f_decreasing_on_interval :
  StrictMonoOn f (Set.Ioo (0 : ℝ) 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_decreasing_on_interval_l674_67444


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_approx_l674_67434

/-- Calculates the increase in wheel radius given initial and final trip distances and initial wheel radius -/
noncomputable def wheel_radius_increase (initial_distance : ℝ) (final_distance : ℝ) (initial_radius : ℝ) : ℝ :=
  let initial_circumference := 2 * Real.pi * initial_radius
  let initial_distance_per_rotation := initial_circumference / 63360
  let num_rotations := initial_distance / initial_distance_per_rotation
  let final_distance_per_rotation := final_distance / num_rotations
  let final_circumference := final_distance_per_rotation * 63360
  let final_radius := final_circumference / (2 * Real.pi)
  final_radius - initial_radius

/-- Theorem stating that the wheel radius increase is approximately 0.34 inches -/
theorem wheel_radius_increase_approx (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ 
  |wheel_radius_increase 600 582 18 - 0.34| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wheel_radius_increase_approx_l674_67434


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_is_right_angle_l674_67439

theorem triangle_angle_c_is_right_angle 
  (A B C : ℝ) 
  (h1 : Real.sin A + Real.cos B = Real.sqrt 2) 
  (h2 : Real.cos A + Real.sin B = Real.sqrt 2) 
  (h3 : A + B + C = Real.pi) -- Sum of angles in a triangle
  (h4 : 0 < A ∧ A < Real.pi) -- Angle A is between 0 and π
  (h5 : 0 < B ∧ B < Real.pi) -- Angle B is between 0 and π
  (h6 : 0 < C ∧ C < Real.pi) -- Angle C is between 0 and π
  : C = Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_c_is_right_angle_l674_67439


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comparison_l674_67422

noncomputable def f (n : ℕ+) : ℝ := (Finset.range n).sum (λ i => 1 / Real.sqrt (i + 1 : ℝ))

theorem f_comparison (n : ℕ+) :
  (n = 1 ∨ n = 2 → f n < Real.sqrt (n + 1 : ℝ)) ∧
  (n ≥ 3 → f n > Real.sqrt (n + 1 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_comparison_l674_67422


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_apartment_size_l674_67423

/-- The rent rate in dollars per square foot -/
noncomputable def rent_rate : ℚ := 120 / 100

/-- The monthly budget in dollars -/
def budget : ℚ := 720

/-- The largest apartment size in square feet -/
noncomputable def largest_size : ℚ := budget / rent_rate

theorem largest_apartment_size :
  largest_size = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_apartment_size_l674_67423


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_max_value_l674_67461

open Real Matrix

-- Define the determinant as a function of θ and φ
noncomputable def det_function (θ φ : ℝ) : ℝ :=
  det !![1, 1, 1;
         1, 1 + Real.sin θ, 1 + Real.cos φ;
         1 + Real.cos θ, 1 + Real.sin φ, 1]

-- State the theorem
theorem det_max_value :
  ∃ (max_value : ℝ), max_value = 2 ∧ 
  ∀ θ φ, det_function θ φ ≤ max_value :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_max_value_l674_67461


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_17_l674_67413

/-- An arithmetic sequence with the given first three terms -/
def arithmetic_sequence (x : ℝ) : ℕ → ℝ
  | 0 => x - 1  -- Adding the case for 0
  | 1 => x - 1
  | 2 => x + 1
  | 3 => 2*x + 3
  | n + 4 => arithmetic_sequence x (n + 3) + (arithmetic_sequence x 2 - arithmetic_sequence x 1)

/-- Theorem: The 10th term of the arithmetic sequence is 17 -/
theorem tenth_term_is_17 : ∀ x : ℝ, arithmetic_sequence x 10 = 17 := by
  intro x
  -- The proof steps would go here
  sorry

#eval arithmetic_sequence 0 10  -- This line is added to check the result

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_17_l674_67413


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_entrance_problem_main_theorem_l674_67464

/-- The time required for all students to pass through when two main entrances and one side entrance are open -/
noncomputable def time_to_pass_through (main_rate : ℝ) (side_rate : ℝ) (total_students : ℕ) : ℝ :=
  (total_students : ℝ) / (2 * main_rate + side_rate)

/-- The problem statement -/
theorem school_entrance_problem (main_rate : ℝ) (side_rate : ℝ) :
  main_rate > 0 →
  side_rate > 0 →
  main_rate + 2 * side_rate = 560 / 2 →
  main_rate + side_rate = 800 / 4 →
  time_to_pass_through main_rate side_rate (32 * 54) = 5.4 := by
  sorry

/-- The main theorem to prove -/
theorem main_theorem : ∃ (main_rate side_rate : ℝ),
  main_rate > 0 ∧
  side_rate > 0 ∧
  main_rate + 2 * side_rate = 560 / 2 ∧
  main_rate + side_rate = 800 / 4 ∧
  time_to_pass_through main_rate side_rate (32 * 54) = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_entrance_problem_main_theorem_l674_67464


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_company_acquisition_ratio_l674_67447

/-- The ratio of a company's price to the sum of two other companies' assets -/
noncomputable def price_to_assets_ratio (price : ℝ) (assets_a : ℝ) (assets_b : ℝ) : ℝ :=
  price / (assets_a + assets_b)

/-- Theorem: Given the conditions from the problem, the ratio of the price to combined assets is approximately 0.8889 -/
theorem company_acquisition_ratio :
  ∀ (price assets_a assets_b : ℝ),
  price = 1.6 * assets_a →
  price = 2 * assets_b →
  abs (price_to_assets_ratio price assets_a assets_b - 0.8889) < 0.0001 :=
by
  sorry

#check company_acquisition_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_company_acquisition_ratio_l674_67447


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_and_min_chord_length_l674_67428

-- Define the line l: y = kx + 1
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

-- Define the circle C: (x-1)^2 + (y+1)^2 = 12
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 12

-- Theorem statement
theorem line_circle_intersection_and_min_chord_length (k : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
    y₁ = line k x₁ ∧ y₂ = line k x₂) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
    y₁ = line k x₁ ∧ y₂ = line k x₂ ∧
    ∀ x₃ y₃ x₄ y₄ : ℝ,
      circle_eq x₃ y₃ ∧ circle_eq x₄ y₄ ∧ 
      y₃ = line k x₃ ∧ y₄ = line k x₄ →
      (x₁ - x₂)^2 + (y₁ - y₂)^2 ≤ (x₃ - x₄)^2 + (y₃ - y₄)^2) ∧
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    circle_eq x₁ y₁ ∧ circle_eq x₂ y₂ ∧ 
    y₁ = line k x₁ ∧ y₂ = line k x₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 28) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_and_min_chord_length_l674_67428


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l674_67477

def mySequence (n : ℕ) : ℕ :=
  match n with
  | 0 => 3
  | n + 1 => (mySequence n) ^ 2

theorem sixth_term_value : mySequence 5 = 1853020188851841 := by
  rfl

#eval mySequence 5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sixth_term_value_l674_67477


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l674_67432

theorem infinite_geometric_series_sum : 
  let a : ℚ := 5/3
  let r : ℚ := -9/20
  let S := (a / (1 - r) : ℚ)
  S = 100/87 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_geometric_series_sum_l674_67432


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l674_67405

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  -- Given conditions
  Real.sin (2 * C) = Real.sqrt 3 * Real.sin C →
  b = 6 →
  (1 / 2) * a * b * Real.sin C = 6 * Real.sqrt 3 →
  -- Conclusions
  C = π / 6 ∧
  a + b + c = 6 + 6 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_abc_properties_l674_67405


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sequence_formula_l674_67476

noncomputable def sequence_a : ℕ → ℝ := sorry
noncomputable def sequence_b : ℕ → ℝ := sorry

axiom nonzero_terms : ∀ n, sequence_a n ≠ 0 ∧ sequence_b n ≠ 0

axiom monotonic_increasing_b : ∀ n, sequence_b n < sequence_b (n + 1)

axiom relation1 : ∀ n, 2 * sequence_a n = sequence_b n * sequence_b (n + 1)

axiom relation2 : ∀ n, sequence_a n + sequence_a (n + 1) = (sequence_b (n + 1))^2

axiom initial_condition1 : sequence_a 1 = sequence_b 2

axiom initial_condition2 : sequence_a 2 = sequence_b 6

theorem b_sequence_formula :
  sequence_b 1 = 2 ∧ ∀ n, sequence_b n = 2 * n :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_sequence_formula_l674_67476


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_theorem_l674_67498

/-- Represents a segment on a line --/
structure Segment where
  start : ℝ
  stop : ℝ
  h : start ≤ stop

/-- Definition of segments sharing a common point --/
def share_point (s₁ s₂ : Segment) : Prop :=
  (s₁.start ≤ s₂.start ∧ s₂.start ≤ s₁.stop) ∨
  (s₂.start ≤ s₁.start ∧ s₁.start ≤ s₂.stop)

/-- Definition of disjoint segments --/
def disjoint (s₁ s₂ : Segment) : Prop :=
  s₁.stop < s₂.start ∨ s₂.stop < s₁.start

/-- Main theorem --/
theorem segment_theorem (segments : Finset Segment) (h : segments.card = 50) :
  (∃ (common_segments : Finset Segment), common_segments ⊆ segments ∧ common_segments.card = 8 ∧
    ∀ s₁ s₂, s₁ ∈ common_segments → s₂ ∈ common_segments → share_point s₁ s₂) ∨
  (∃ (disjoint_segments : Finset Segment), disjoint_segments ⊆ segments ∧ disjoint_segments.card = 8 ∧
    ∀ s₁ s₂, s₁ ∈ disjoint_segments → s₂ ∈ disjoint_segments → s₁ ≠ s₂ → disjoint s₁ s₂) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_theorem_l674_67498


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l674_67455

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) / Real.log 2
noncomputable def g (x : ℝ) : ℝ := Real.log (1 - x) / Real.log 2

noncomputable def F (x : ℝ) : ℝ := f x + g x

theorem function_properties :
  -- (I) Domain of F
  (∀ x : ℝ, F x ≠ 0 → -1 < x ∧ x < 1) ∧
  -- (II) F is even
  (∀ x : ℝ, F (-x) = F x) ∧
  -- (III) F is decreasing on (0, 1)
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → F x₁ > F x₂) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l674_67455


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_23_6_pi_l674_67457

theorem tan_negative_23_6_pi : Real.tan (-23/6 * Real.pi) = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_negative_23_6_pi_l674_67457


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_remaining_dimes_l674_67497

/-- Represents the number of dimes Fred has -/
def fred_dimes : ℕ → ℕ := sorry

/-- Represents the number of dimes Fred's sister borrowed -/
def sister_borrowed : ℕ := sorry

/-- 
Given that Fred initially had 7 dimes and his sister borrowed 3 dimes,
prove that Fred now has 4 dimes.
-/
theorem fred_remaining_dimes :
  fred_dimes 0 = 7 ∧ sister_borrowed = 3 →
  fred_dimes 1 = 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fred_remaining_dimes_l674_67497


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_proof_l674_67433

/-- The function f(x) = x^2 - 7x + c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 - 7*x + c

/-- 2 is in the range of f -/
def in_range (c : ℝ) : Prop := ∃ x, f c x = 2

/-- The largest value of c such that 2 is in the range of f -/
noncomputable def largest_c : ℝ := 57/4

theorem largest_c_proof :
  (∀ c > largest_c, ¬(in_range c)) ∧
  (∀ c ≤ largest_c, in_range c) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_c_proof_l674_67433


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_m_when_min_is_six_l674_67430

-- Define the function f
noncomputable def f (x m : ℝ) : ℝ := (x^2 + 3) / (x - m)

-- Part 1
theorem range_of_m (m : ℝ) :
  (∀ x > m, f x m + m ≥ 0) →
  m ≥ -2 * Real.sqrt 15 / 5 :=
by sorry

-- Part 2
theorem m_when_min_is_six (m : ℝ) :
  (∃ min_val : ℝ, min_val = 6 ∧ ∀ x > m, f x m ≥ min_val) →
  m = 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_m_when_min_is_six_l674_67430


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l674_67493

theorem complex_fraction_equality : 
  (3 : ℂ) / (1 - Complex.I)^2 = (3 / 2 : ℂ) * Complex.I := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l674_67493


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_count_l674_67404

theorem winning_strategy_count :
  let N := Finset.range 2019
  let winning_strategy := N.filter (fun n => (n + 1) % 3 ≠ 0)
  winning_strategy.card = 1346 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_strategy_count_l674_67404


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flowers_on_porch_l674_67437

def total_plants : ℕ := 80
def flowering_percentage : ℚ := 40 / 100
def porch_fraction : ℚ := 1 / 4
def flowers_per_plant : ℕ := 5

theorem flowers_on_porch : 
  (Int.floor (↑total_plants * flowering_percentage * porch_fraction * ↑flowers_per_plant)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flowers_on_porch_l674_67437


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_equation_l674_67436

/-- Ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0

/-- Line l passing through a focus of the ellipse -/
structure Line where
  m : ℝ

/-- Point on the ellipse -/
structure Point where
  x : ℝ
  y : ℝ

noncomputable def eccentricity (e : Ellipse) : ℝ := Real.sqrt (1 - e.b^2 / e.a^2)

theorem ellipse_equation (e : Ellipse) (h_ecc : eccentricity e = Real.sqrt 3 / 2) :
  e.a = 2 ∧ e.b = 1 :=
sorry

theorem line_equation (e : Ellipse) (l : Line) (A B M : Point)
  (h_ecc : eccentricity e = Real.sqrt 3 / 2)
  (h_A : A.x^2 / e.a^2 + A.y^2 / e.b^2 = 1)
  (h_B : B.x^2 / e.a^2 + B.y^2 / e.b^2 = 1)
  (h_M : M.x^2 / e.a^2 + M.y^2 / e.b^2 = 1)
  (h_l_A : A.x + l.m * A.y + Real.sqrt 3 = 0)
  (h_l_B : B.x + l.m * B.y + Real.sqrt 3 = 0)
  (h_OM : 2 * M.x = A.x + Real.sqrt 3 * B.x ∧ 2 * M.y = A.y + Real.sqrt 3 * B.y) :
  l.m = Real.sqrt 2 ∨ l.m = -Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_line_equation_l674_67436


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_minimum_m_value_l674_67407

-- Define points A and B
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, -2)

-- Define the distance function
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Define the locus condition
def locus_condition (P : ℝ × ℝ) : Prop :=
  distance P B = 2 * distance P A

-- Define the curve C
def curve_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*y = 0

-- Define the line equation
def line_equation (m x y : ℝ) : Prop :=
  m*x + y - 3*m - 1 = 0

-- Define the point M condition
def point_M_condition (M : ℝ × ℝ) : Prop :=
  distance M B ≥ 2 * distance M A

-- Theorem statements
theorem locus_is_circle :
  ∀ (P : ℝ × ℝ), locus_condition P ↔ curve_C P.1 P.2 := by sorry

theorem minimum_m_value :
  ∃ (m_min : ℝ), m_min = (3 - 2 * Real.sqrt 6) / 5 ∧
  (∀ (m : ℝ), (∃ (M : ℝ × ℝ), line_equation m M.1 M.2 ∧ point_M_condition M) → m ≥ m_min) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_is_circle_minimum_m_value_l674_67407


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_upper_bound_f_range_eq_l674_67419

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1 / (x - 1) + 1) / Real.log 0.5

-- State the theorem
theorem f_range_upper_bound :
  ∀ x > 1, f x ≤ -2 ∧ ∃ x > 1, f x = -2 :=
by sorry

-- Define the range of f
def f_range : Set ℝ := {y | ∃ x > 1, f x = y}

-- State the theorem for the exact range of f
theorem f_range_eq : f_range = Set.Iic (-2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_upper_bound_f_range_eq_l674_67419


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l674_67459

/-- An arithmetic sequence is a sequence where the difference between consecutive terms is constant. -/
structure ArithmeticSequence where
  a : ℕ → ℚ  -- The sequence (using rationals instead of reals)
  d : ℚ      -- Common difference
  h : ∀ n, a (n + 1) = a n + d

/-- The sum of the first n terms of an arithmetic sequence. -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * seq.a 1 + (n - 1) * seq.d)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h3 : seq.a 3 = 7)
  (h4 : seq.a 4 = 11)
  (h5 : seq.a 5 = 15) :
  sum_n seq 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l674_67459


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_solutions_l674_67402

-- Define the equation
def equation (x : ℝ) : Prop :=
  x - ⌊x⌋ = 1 / (⌊x⌋^2)

-- Define a function to check if a solution is valid
def is_valid_solution (x : ℝ) : Prop :=
  x > 0 ∧ equation x

-- Define the three smallest positive solutions
def smallest_solutions : Prop :=
  ∃ (s₁ s₂ s₃ : ℝ),
    is_valid_solution s₁ ∧
    is_valid_solution s₂ ∧
    is_valid_solution s₃ ∧
    s₁ < s₂ ∧
    s₂ < s₃ ∧
    ∀ (x : ℝ), is_valid_solution x → x ≥ s₁

-- Theorem statement
theorem sum_of_smallest_solutions :
  smallest_solutions →
  ∃ (s₁ s₂ s₃ : ℝ),
    is_valid_solution s₁ ∧
    is_valid_solution s₂ ∧
    is_valid_solution s₃ ∧
    s₁ + s₂ + s₃ = 9 + 73 / 144 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_smallest_solutions_l674_67402


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_control_weights_l674_67441

def is_balanceable (weights : List ℝ) (target : ℕ) : Prop :=
  ∃ (subset : List ℝ), subset.all (λ w => w ∈ weights) ∧ 
    (∃ (signs : List Int), signs.length = subset.length ∧ 
      (List.sum (List.zipWith (· * ·) subset signs) : ℝ) = target)

theorem min_control_weights :
  ∃ (weights : List ℝ),
    weights.length = 4 ∧
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 40 → is_balanceable weights n) ∧
    (∀ (other_weights : List ℝ),
      other_weights.length < 4 →
      ¬(∀ n : ℕ, 1 ≤ n ∧ n ≤ 40 → is_balanceable other_weights n)) :=
by
  sorry

#check min_control_weights

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_control_weights_l674_67441


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_surprise_weight_theorem_l674_67416

-- Define constants for unit conversions
noncomputable def pounds_to_grams : ℝ := 453.592
noncomputable def ounces_to_grams : ℝ := 28.3495
noncomputable def fluid_ounces_to_grams : ℝ := 29.5735
noncomputable def tablespoons_to_grams : ℝ := 14.1748

-- Define the recipe quantities
def servings : ℕ := 12
noncomputable def chicken_pounds : ℝ := 4.5
def stuffing_ounces : ℕ := 24
def broth_fluid_ounces : ℕ := 8
def butter_tablespoons : ℕ := 12

-- Define the function to calculate the weight of a single serving
noncomputable def chicken_surprise_single_serving_weight : ℝ :=
  let total_weight := 
    chicken_pounds * pounds_to_grams +
    (stuffing_ounces : ℝ) * ounces_to_grams +
    (broth_fluid_ounces : ℝ) * fluid_ounces_to_grams +
    (butter_tablespoons : ℝ) * tablespoons_to_grams
  total_weight / (servings : ℝ)

-- Theorem statement
theorem chicken_surprise_weight_theorem :
  ∃ ε > 0, |chicken_surprise_single_serving_weight - 260.68647| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chicken_surprise_weight_theorem_l674_67416


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_f_third_quadrant_l674_67480

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi + α) * Real.cos (2 * Real.pi - α) * Real.tan (-α)) /
  (Real.tan (-Real.pi - α) * Real.cos ((3 * Real.pi) / 2 + α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_specific_value : f (-31 * Real.pi / 3) = -1 / 2 := by sorry

theorem f_third_quadrant (α : Real)
  (h1 : Real.pi < α ∧ α < 3 * Real.pi / 2)  -- α is in the third quadrant
  (h2 : Real.sin α = -1 / 5) :
  f α = 2 * Real.sqrt 5 / 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_f_third_quadrant_l674_67480


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l674_67418

noncomputable def curve (x : ℝ) : ℝ := Real.log x
noncomputable def line (x b : ℝ) : ℝ := (1/2) * x + b

theorem tangent_line_to_ln_curve (b : ℝ) : 
  (∃ x₀ : ℝ, x₀ > 0 ∧ 
    curve x₀ = line x₀ b ∧ 
    (deriv curve x₀) = (1/2)) → 
  b = Real.log 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_ln_curve_l674_67418


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l674_67466

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 1 / ⌊x^2 - 7*x + 13⌋

-- Define the domain of f(x)
def domain_f : Set ℝ := {x | x ≤ 3 ∨ x ≥ 4}

-- Theorem statement
theorem domain_of_f : 
  ∀ x : ℝ, f x ≠ 0 ↔ x ∈ domain_f :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l674_67466


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l674_67427

theorem calculate_expression : (12 : ℚ) * ((1/3 + 1/4 + 1/6)⁻¹) = 16 := by
  -- Convert fractions to rationals
  have h1 : (1/3 : ℚ) + (1/4 : ℚ) + (1/6 : ℚ) = (3/4 : ℚ) := by norm_num
  
  -- Calculate the inverse
  have h2 : ((3/4 : ℚ)⁻¹) = (4/3 : ℚ) := by norm_num
  
  -- Multiply by 12
  calc
    (12 : ℚ) * ((1/3 + 1/4 + 1/6)⁻¹) = (12 : ℚ) * ((3/4 : ℚ)⁻¹) := by rw [h1]
    _ = (12 : ℚ) * (4/3 : ℚ) := by rw [h2]
    _ = 16 := by norm_num

  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calculate_expression_l674_67427


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l674_67453

theorem expression_equality (a b : ℝ) (h : a^2 - b^2 + a * b ≠ 0) : 
  (a^3 + 2 * a * b^2 - b^3) / (a^2 - b^2 + a * b) = 
  (a * (a^2 + b^2) + b^2 * (a - b)) / (a^2 + b * (a - b)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_equality_l674_67453


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_range_l674_67469

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a * sin A * sin B + b * cos^2 A = 2a, then 0 < A ≤ π/6 -/
theorem angle_A_range (a b c : ℝ) (A B C : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A + B + C = π →
  a * Real.sin A = b * Real.sin B →
  a * Real.sin A = c * Real.sin C →
  a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = 2 * a →
  0 < A ∧ A ≤ π / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_A_range_l674_67469


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_attendance_percentage_l674_67494

/-- Given a crew of 156 laborers with 70 present on a certain day,
    the percentage of laborers who showed up for work, rounded to the nearest tenth, is 44.9%. -/
theorem labor_attendance_percentage :
  let total_laborers : ℕ := 156
  let present_laborers : ℕ := 70
  let attendance_percentage : ℚ := (present_laborers : ℚ) / (total_laborers : ℚ) * 100
  (round (attendance_percentage * 10) : ℚ) / 10 = 44.9 := by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_labor_attendance_percentage_l674_67494


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_perfect_square_l674_67467

theorem product_perfect_square (S : Finset ℕ) 
  (h1 : S.card = 33)
  (h2 : ∀ n ∈ S, ∃ a b c d e : ℕ, n = 2^a * 3^b * 5^c * 7^d * 11^e) :
  ∃ i j, i ∈ S ∧ j ∈ S ∧ i ≠ j ∧ ∃ k : ℕ, i * j = k^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_perfect_square_l674_67467


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2_3_5_7_less_than_500_l674_67488

theorem divisible_by_2_3_5_7_less_than_500 : 
  (Finset.filter (fun n : Fin 500 => n.val > 0 ∧ 2 ∣ n.val ∧ 3 ∣ n.val ∧ 5 ∣ n.val ∧ 7 ∣ n.val) Finset.univ).card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_2_3_5_7_less_than_500_l674_67488


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positives_in_K_l674_67468

def list_K : List ℤ := List.range 12 |>.map (· - 3)

theorem range_of_positives_in_K : 
  (list_K.filter (· > 0)).maximum?.isSome ∧
  (list_K.filter (· > 0)).minimum?.isSome ∧
  ((list_K.filter (· > 0)).maximum?.get! - (list_K.filter (· > 0)).minimum?.get! = 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_positives_in_K_l674_67468


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l674_67487

/-- A function f(x) = (ax + b) / (cx + d) with specific properties -/
noncomputable def f (a b c d : ℝ) (x : ℝ) : ℝ := (a * x + b) / (c * x + d)

/-- The theorem stating the properties of the function and its range -/
theorem function_range_theorem 
  (a b c d : ℝ) 
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) 
  (h_f19 : f a b c d 19 = 19) 
  (h_f97 : f a b c d 97 = 97)
  (h_inverse : ∀ x : ℝ, x ≠ -d/c → f a b c d (f a b c d x) = x) :
  ∃! y : ℝ, y = 58 ∧ ∀ x : ℝ, f a b c d x ≠ y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_range_theorem_l674_67487


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l674_67499

-- Define a triangle by its angles
def Triangle (α β γ : ℝ) : Prop := 0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = Real.pi

-- Define the function f
noncomputable def f (α β γ : ℝ) : ℝ := Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2

theorem triangle_cosine_inequality (α β γ : ℝ) (h : Triangle α β γ) :
  f α β γ ≥ 3/4 ∧
  (f α β γ = 3/4 ↔ α = Real.pi/3 ∧ β = Real.pi/3 ∧ γ = Real.pi/3) ∧
  ¬∃ (m : ℝ), ∀ (α' β' γ' : ℝ), Triangle α' β' γ' → f α' β' γ' ≤ m :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_cosine_inequality_l674_67499


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_mass_equivalence_l674_67481

/-- Represents the mass of an object -/
structure Mass where
  value : ℝ

/-- Represents a circle with a specific mass -/
structure Circle where
  mass : Mass

/-- Represents a square with a specific mass -/
structure Square where
  mass : Mass

/-- Represents an equal-arm scale -/
structure EqualArmScale where
  left_arm : List Mass
  right_arm : List Mass

/-- Checks if an equal-arm scale is balanced -/
def is_balanced (scale : EqualArmScale) : Prop :=
  scale.left_arm.foldl (fun acc m => acc + m.value) 0 = 
  scale.right_arm.foldl (fun acc m => acc + m.value) 0

theorem square_circle_mass_equivalence 
  (c : Circle) 
  (s : Square) 
  (scale : EqualArmScale) :
  is_balanced { 
    left_arm := [c.mass, c.mass, c.mass, c.mass, s.mass, s.mass], 
    right_arm := [c.mass, c.mass, c.mass, c.mass, c.mass, c.mass, c.mass, c.mass, c.mass, c.mass] 
  } →
  s.mass = Mass.mk (3 * c.mass.value) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_circle_mass_equivalence_l674_67481


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l674_67445

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- The area of a polygon -/
noncomputable def area (vertices : List Point) : ℝ := sorry

/-- Check if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Check if a point is inside a quadrilateral -/
def isInside (p : Point) (q : Quadrilateral) : Prop := sorry

/-- Find the intersection of two lines -/
noncomputable def lineIntersection (p1 p2 p3 p4 : Point) : Point := sorry

/-- The main theorem -/
theorem quadrilateral_inequality 
  (ABCD : Quadrilateral) 
  (O : Point) 
  (hConvex : isConvex ABCD) 
  (hInside : isInside O ABCD) : 
  let E := lineIntersection ABCD.A ABCD.B O (lineIntersection ABCD.B ABCD.C O ABCD.C)
  let F := lineIntersection ABCD.B ABCD.C O (lineIntersection ABCD.A ABCD.B O ABCD.A)
  let G := lineIntersection ABCD.C ABCD.D O (lineIntersection ABCD.D ABCD.A O ABCD.D)
  let H := lineIntersection ABCD.D ABCD.A O (lineIntersection ABCD.C ABCD.D O ABCD.C)
  Real.sqrt (area [ABCD.A, H, O, E]) + Real.sqrt (area [ABCD.C, F, O, G]) 
  ≤ Real.sqrt (area [ABCD.A, ABCD.B, ABCD.C, ABCD.D]) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_inequality_l674_67445


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l674_67440

theorem triangle_properties (A B C : ℝ) (AC AB : ℝ) :
  0 < A ∧ A < Real.pi →
  0 < B ∧ B < Real.pi →
  0 < C ∧ C < Real.pi →
  A + B + C = Real.pi →
  AC = 6 →
  Real.cos B = 4/5 →
  C = Real.pi/4 →
  AB = 5 * Real.sqrt 2 ∧
  Real.cos (A - Real.pi/6) = (7 * Real.sqrt 2 - Real.sqrt 6) / 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l674_67440


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_d_value_l674_67449

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  focus1 : Point
  focus2 : Point
  tangentPoint : Point

/-- Checks if a point is in the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Checks if a point is on the y-axis -/
def isOnYAxis (p : Point) : Prop :=
  p.x = 0

/-- The distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: For an ellipse in the second quadrant tangent to the y-axis 
    with foci at (3,7) and (3,d), the value of d is 7 -/
theorem ellipse_foci_d_value (e : Ellipse) (d : ℝ) : 
  e.focus1 = Point.mk 3 7 →
  e.focus2 = Point.mk 3 d →
  isOnYAxis e.tangentPoint →
  isInSecondQuadrant e.tangentPoint →
  distance e.focus1 e.tangentPoint = distance e.focus2 e.tangentPoint →
  distance e.focus1 e.tangentPoint + distance e.focus2 e.tangentPoint = 6 →
  d = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_foci_d_value_l674_67449


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_202_203_l674_67424

def f (x : ℤ) : ℤ := x^2 - x + 2023

theorem gcd_f_202_203 : Int.gcd (f 202) (f 203) = 17 := by
  -- Convert the result of f to Int.natAbs before applying gcd
  have h1 : Int.gcd (f 202) (f 203) = Int.gcd (Int.natAbs (f 202)) (Int.natAbs (f 203)) := by sorry
  -- The rest of the proof would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_f_202_203_l674_67424


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_l674_67479

def box_width : ℝ := 12
def box_length : ℝ := 16
def triangle_area : ℝ := 30

theorem box_dimensions (m n : ℕ) (h1 : Nat.Coprime m n) 
  (h2 : (m : ℝ) / n = 36 / 5) : m + n = 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_box_dimensions_l674_67479


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_continuity_points_l674_67431

-- Define the piecewise function f
noncomputable def f (n : ℝ) (x : ℝ) : ℝ :=
  if x < n then x^2 + 2*x + 3 else 3*x + 6

-- Define continuity condition
def is_continuous_at_n (n : ℝ) : Prop :=
  n^2 + 2*n + 3 = 3*n + 6

-- Theorem statement
theorem sum_of_continuity_points (n₁ n₂ : ℝ) :
  is_continuous_at_n n₁ ∧ is_continuous_at_n n₂ ∧ n₁ ≠ n₂ → n₁ + n₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_continuity_points_l674_67431


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l674_67472

-- Define the circle
def my_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the external point
def external_point : ℝ × ℝ := (3, 1)

-- Define a tangent line
def is_tangent (m b : ℝ) : Prop :=
  ∃ (x y : ℝ), my_circle x y ∧ y = m * x + b ∧
  ∀ (x' y' : ℝ), my_circle x' y' → (y' - (m * x' + b))^2 ≥ 0

-- Define the theorem
theorem tangent_line_equation :
  ∃ (A B : ℝ × ℝ),
    my_circle A.1 A.2 ∧
    my_circle B.1 B.2 ∧
    is_tangent ((external_point.2 - A.2) / (external_point.1 - A.1)) (A.2 - ((external_point.2 - A.2) / (external_point.1 - A.1)) * A.1) ∧
    is_tangent ((external_point.2 - B.2) / (external_point.1 - B.1)) (B.2 - ((external_point.2 - B.2) / (external_point.1 - B.1)) * B.1) ∧
    (B.2 - A.2) / (B.1 - A.1) = -2 ∧
    A.2 - (-2 * A.1) = 3 :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l674_67472


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vowel_initials_probability_l674_67492

/-- Represents the alphabet --/
inductive Letter : Type
| A | B | C | D | E | F | G | H | I | J | K | L | M
| N | O | P | Q | R | S | T | U | V | W | X | Y | Z

/-- Defines vowels (excluding Y) --/
def isVowel (l : Letter) : Bool :=
  match l with
  | Letter.A | Letter.E | Letter.I | Letter.O | Letter.U => true
  | _ => false

/-- Represents a student's initials --/
structure Initials :=
  (letter : Letter)

/-- Represents the class --/
structure ClassInfo :=
  (students : Finset Initials)
  (size : Nat)
  (distinct_initials : students.card = size)
  (size_is_26 : size = 26)

/-- The main theorem --/
theorem vowel_initials_probability (c : ClassInfo) :
  (c.students.filter (fun i => isVowel i.letter)).card / c.size = 5 / 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vowel_initials_probability_l674_67492


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dominic_average_speed_l674_67411

/-- Calculates the average speed given distance and time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

theorem dominic_average_speed :
  let distance : ℝ := 184
  let time : ℝ := 8
  average_speed distance time = 23 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dominic_average_speed_l674_67411


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l674_67435

theorem sin_2theta_value (θ : ℝ) (h : (Real.sqrt 2 * Real.cos (2 * θ)) / Real.cos (π / 4 + θ) = Real.sqrt 3 * Real.sin (2 * θ)) :
  Real.sin (2 * θ) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2theta_value_l674_67435


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_ln_function_l674_67454

theorem domain_of_ln_function (x : ℝ) :
  (∃ k : ℤ, x ∈ Set.Ioo ((2 * k : ℝ) * π + π / 4) ((2 * k : ℝ) * π + 3 * π / 4)) ↔
  (2 * Real.sin x - Real.sqrt 2 > 0) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_ln_function_l674_67454


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_triangles_exist_l674_67485

/-- A structure representing a triangle with altitude feet and midpoint -/
structure TriangleWithPoints where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  A1 : ℝ × ℝ
  B1 : ℝ × ℝ
  C1 : ℝ × ℝ

/-- Predicate to check if a triangle is acute-angled -/
def isAcuteAngled (t : TriangleWithPoints) : Prop :=
  sorry -- Definition of acute-angled triangle

/-- Predicate to check if A1 is the foot of altitude from A -/
def isAltitudeFootA (t : TriangleWithPoints) : Prop :=
  sorry -- Definition of altitude foot

/-- Predicate to check if B1 is the foot of altitude from B -/
def isAltitudeFootB (t : TriangleWithPoints) : Prop :=
  sorry -- Definition of altitude foot

/-- Predicate to check if C1 is the midpoint of AB -/
def isMidpointAB (t : TriangleWithPoints) : Prop :=
  sorry -- Definition of midpoint

/-- Two triangles are non-congruent -/
def areNonCongruent (t1 t2 : TriangleWithPoints) : Prop :=
  sorry -- Definition of non-congruent triangles

/-- Main theorem: There exist multiple non-congruent triangles with the same special points -/
theorem multiple_triangles_exist :
  ∃ (t1 t2 : TriangleWithPoints),
    isAcuteAngled t1 ∧
    isAcuteAngled t2 ∧
    isAltitudeFootA t1 ∧
    isAltitudeFootA t2 ∧
    isAltitudeFootB t1 ∧
    isAltitudeFootB t2 ∧
    isMidpointAB t1 ∧
    isMidpointAB t2 ∧
    t1.A1 = t2.A1 ∧
    t1.B1 = t2.B1 ∧
    t1.C1 = t2.C1 ∧
    areNonCongruent t1 t2 :=
  by
    sorry -- Proof of the theorem


end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiple_triangles_exist_l674_67485


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l674_67443

/-- Prove that the cost price of an article is 1600, given the following conditions:
    1. The article was sold at a 5% profit.
    2. If it had been bought for 5% less and sold for 8 less, it would have made a 10% profit. -/
theorem article_cost_price : ∃ C : ℝ, C = 1600 := by
  let original_selling_price (C : ℝ) := 1.05 * C
  let new_cost_price (C : ℝ) := 0.95 * C
  let new_selling_price (C : ℝ) := original_selling_price C - 8
  
  have h : ∃ C : ℝ, new_selling_price C = 1.1 * new_cost_price C := by
    sorry
  
  have h_solution : ∃ C : ℝ, C = 1600 := by
    sorry
  
  exact h_solution


end NUMINAMATH_CALUDE_ERRORFEEDBACK_article_cost_price_l674_67443


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_beta_f_increasing_intervals_l674_67442

noncomputable def f (x : ℝ) : ℝ := Real.cos x * (Real.sin x + Real.cos x) - 0.5

theorem f_value_at_beta (β : ℝ) (h1 : 0 < β) (h2 : β < Real.pi / 2) (h3 : Real.sin β = 0.6) :
  f β = 31 / 50 := by
  sorry

theorem f_increasing_intervals (x : ℝ) (k : ℤ) :
  (k : ℝ) * Real.pi - 3 * Real.pi / 8 ≤ x ∧ x ≤ (k : ℝ) * Real.pi + Real.pi / 8 →
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), y < x → f y < f x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_value_at_beta_f_increasing_intervals_l674_67442


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_specific_value_l674_67451

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (x - Real.pi / 3) * Real.cos x + 1

theorem f_monotone_and_specific_value :
  (∀ (k : ℤ), StrictMonoOn f (Set.Icc (k * Real.pi - Real.pi / 3) (k * Real.pi + Real.pi / 6))) ∧
  (∀ (α : ℝ), α ∈ Set.Ioo 0 (Real.pi / 2) → 
    f (α + Real.pi / 12) = 7 / 6 → 
    Real.sin (7 * Real.pi / 6 - 2 * α) = 2 * Real.sqrt 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_and_specific_value_l674_67451


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_open_interval_l674_67496

open Set

-- Define the function f as noncomputable
noncomputable def f (x y z : ℝ) : ℝ := y / (y + x) + z / (z + y) + x / (x + z)

-- State the theorem
theorem f_range_open_interval
  (x y z : ℝ)
  (hpos : x > 0 ∧ y > 0 ∧ z > 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (hrel : x^2 + y^3 = z^4) :
  f x y z ∈ Ioo 1 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_open_interval_l674_67496


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_area_l674_67478

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Calculates the area of a triangle -/
noncomputable def area (t : Triangle) : ℝ := sorry

/-- Theorem: Given a triangle ABC and four points P, Q on AC and R, S on BC 
    (where P is between A and Q, and R is between B and S), at least one of 
    the four triangles formed by these points has an area not greater than 
    one-quarter of the area of triangle ABC. -/
theorem smallest_triangle_area 
  (ABC : Triangle) 
  (P Q R S : Point) 
  (h1 : P.x > ABC.A.x ∧ P.x < Q.x ∧ Q.x < ABC.C.x)  -- P and Q are on AC
  (h2 : P.y > ABC.A.y ∧ P.y < Q.y ∧ Q.y < ABC.C.y)  -- P and Q are on AC
  (h3 : R.x > ABC.B.x ∧ R.x < S.x ∧ S.x < ABC.C.x)  -- R and S are on BC
  (h4 : R.y < ABC.B.y ∧ R.y > S.y ∧ S.y > ABC.C.y)  -- R and S are on BC
  : ∃ (T : Triangle), area T ≤ (1/4) * area ABC := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_triangle_area_l674_67478


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_field_maximum_l674_67400

/-- The electric field magnitude along the perpendicular axis of a charged ring -/
noncomputable def E (Q R x : ℝ) : ℝ := Q * x / (R^2 + x^2)^(3/2)

/-- The derivative of the electric field magnitude with respect to x -/
noncomputable def dE (Q R x : ℝ) : ℝ := Q * (R^2 - 2*x^2) / (R^2 + x^2)^(5/2)

theorem electric_field_maximum (Q R : ℝ) (hQ : Q > 0) (hR : R > 0) :
  ∃ x : ℝ, x = R * Real.sqrt 2 ∧ 
    ∀ y : ℝ, E Q R y ≤ E Q R x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_electric_field_maximum_l674_67400


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l674_67475

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi/6)

theorem f_increasing_on_interval :
  ∀ x y, 0 < x ∧ x < y ∧ y < Real.pi/2 → f x < f y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l674_67475


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_g_always_negative_l674_67426

noncomputable section

-- Define the function f
def f (a b x : ℝ) : ℝ := (a * Real.log x - b * Real.exp x) / x

-- Define the function g
def g (x : ℝ) : ℝ := f 1 1 x + 2

-- Part I theorem
theorem max_value_of_f (a : ℝ) (ha : a > 0) :
  (∃ b : ℝ, ∀ x : ℝ, x > 0 → (deriv (f a b)) e = 0) →
  ∃ M : ℝ, M = a / Real.exp 1 ∧ ∀ x : ℝ, x > 0 → f a b x ≤ M :=
sorry

-- Part II theorem
theorem g_always_negative :
  ∀ x : ℝ, x > 0 → g x < 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_g_always_negative_l674_67426


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gyroscope_speed_after_90_seconds_l674_67482

/-- The speed of a gyroscope after a given time, given its initial speed and doubling rate. -/
noncomputable def gyroscope_speed (initial_speed : ℝ) (doubling_interval : ℝ) (elapsed_time : ℝ) : ℝ :=
  initial_speed * (2 ^ (elapsed_time / doubling_interval))

/-- Theorem: The gyroscope's speed after 90 seconds is 400 m/s, given the initial conditions. -/
theorem gyroscope_speed_after_90_seconds :
  gyroscope_speed 6.25 15 90 = 400 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gyroscope_speed_after_90_seconds_l674_67482


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_cosine_theorem_l674_67489

/-- Hyperbola type -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point type -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the hyperbola C: x² - y² = 2 -/
noncomputable def C : Hyperbola := { a := Real.sqrt 2, b := Real.sqrt 2, c := 2 }

/-- Left focus of the hyperbola -/
noncomputable def F₁ : Point := { x := -C.c, y := 0 }

/-- Right focus of the hyperbola -/
noncomputable def F₂ : Point := { x := C.c, y := 0 }

/-- A point P on the hyperbola C -/
noncomputable def P : Point := { x := sorry, y := sorry }

/-- Distance between two points -/
noncomputable def distance (p q : Point) : ℝ := 
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2)

/-- Cosine of the angle between three points -/
noncomputable def cosAngle (p q r : Point) : ℝ := 
  ((distance p q)^2 + (distance p r)^2 - (distance q r)^2) / (2 * distance p q * distance p r)

/-- Theorem: If P is on hyperbola C and |PF₁| = 2|PF₂|, then cos∠F₁PF₂ = 3/4 -/
theorem hyperbola_cosine_theorem (h1 : P.x^2 - P.y^2 = 2) 
  (h2 : distance P F₁ = 2 * distance P F₂) : 
  cosAngle F₁ P F₂ = 3/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_cosine_theorem_l674_67489


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_l674_67420

/-- Given a drug with an original price and a price reduction percentage,
    calculate the new price after the reduction. -/
noncomputable def new_price (original_price : ℝ) (reduction_percentage : ℝ) : ℝ :=
  original_price * (1 - reduction_percentage / 100)

/-- Theorem: The new price of a drug after a 40% reduction is 60% of the original price. -/
theorem price_reduction (a : ℝ) :
  new_price a 40 = 0.6 * a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_reduction_l674_67420


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_sets_l674_67483

/-- A structure representing a set of consecutive positive integers -/
structure ConsecutiveSet where
  start : ℕ+  -- The first integer in the set
  length : ℕ  -- The number of integers in the set
  length_ge_two : length ≥ 2  -- Ensure at least two integers

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : ℕ :=
  s.length * (2 * s.start.val + s.length - 1) / 2

/-- A predicate that checks if a ConsecutiveSet sums to 225 -/
def sums_to_225 (s : ConsecutiveSet) : Prop :=
  sum_consecutive s = 225

/-- The main theorem stating that there are exactly 4 valid sets -/
theorem exactly_four_sets : 
  ∃! (sets : Finset ConsecutiveSet), 
    (∀ s ∈ sets, sums_to_225 s) ∧ 
    Finset.card sets = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exactly_four_sets_l674_67483


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_geq_five_l674_67414

noncomputable def f (x : ℝ) : ℝ := Real.log (x^2 - 6*x + 5) / Real.log (Real.sin 1)

theorem function_decreasing_implies_a_geq_five (a : ℝ) :
  (∀ x y : ℝ, a < x ∧ x < y → f y < f x) →
  a ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_decreasing_implies_a_geq_five_l674_67414


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l674_67401

theorem number_of_subsets {α : Type*} (S : Finset α) (h : S.card = 4) : 
  (Finset.powerset S).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_subsets_l674_67401


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_study_five_hours_score_l674_67415

/-- Represents the exam score calculation system -/
structure ExamScore where
  totalPoints : ℕ
  baseHours : ℕ
  baseScore : ℕ
  additionalHourBonus : ℚ

/-- Calculates the exam score based on hours studied -/
noncomputable def calculateScore (e : ExamScore) (hoursStudied : ℕ) : ℚ :=
  let baseScore := (e.baseScore : ℚ) / (e.baseHours : ℚ) * min hoursStudied e.baseHours
  let additionalHours := max (hoursStudied - e.baseHours) 0
  let additionalScore := (additionalHours : ℚ) * e.additionalHourBonus * baseScore
  baseScore + additionalScore

/-- Theorem stating that studying for 5 hours results in a score of 162 points -/
theorem study_five_hours_score (e : ExamScore) 
  (h1 : e.totalPoints = 150)
  (h2 : e.baseHours = 3)
  (h3 : e.baseScore = 90)
  (h4 : e.additionalHourBonus = 1/10)
  (h5 : calculateScore e 2 = 90) :
  calculateScore e 5 = 162 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_study_five_hours_score_l674_67415


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_six_l674_67462

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define a function to check if two vectors are parallel
def isParallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v1.1 * v2.2 = k * v1.2 * v2.1

-- Define the theorem
theorem angle_B_is_pi_over_six (t : Triangle) 
  (h1 : isParallel (t.A.cos, t.A.sin) (1, Real.sqrt 3)) 
  (h2 : t.a * t.B.cos + t.b * t.A.cos = t.c * t.C.sin) : t.B = π / 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_B_is_pi_over_six_l674_67462


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_three_five_l674_67484

/-- The diamond operation for positive real numbers -/
noncomputable def diamond (a b : ℝ) : ℝ := sorry

/-- Positivity of the diamond operation -/
axiom diamond_pos (a b : ℝ) : 0 < a → 0 < b → 0 < diamond a b

/-- First condition of the diamond operation -/
axiom diamond_cond1 (a b : ℝ) : 0 < a → 0 < b → diamond (a^2 * b) b = a * diamond b b

/-- Second condition of the diamond operation -/
axiom diamond_cond2 (a : ℝ) : 0 < a → diamond (diamond a 1) a = diamond (a^2) 1

/-- Given condition: 1 ◇ 1 = 1 -/
axiom diamond_one : diamond 1 1 = 1

/-- The main theorem to prove -/
theorem diamond_three_five : diamond 3 5 = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_three_five_l674_67484


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_properties_l674_67465

/-- Regular octagon formed by cutting off corners of a square --/
structure RegularOctagon where
  /-- Side length of the inscribed square --/
  square_side : ℝ
  /-- Side length of the octagon --/
  octagon_side : ℝ
  /-- The square side is 4 + 2√2 --/
  h_square_side : square_side = 4 + 2 * Real.sqrt 2
  /-- The octagon side is 2√2 --/
  h_octagon_side : octagon_side = 2 * Real.sqrt 2

/-- Properties of the regular octagon --/
theorem regular_octagon_properties (o : RegularOctagon) :
  (o.square_side ^ 2 - 4 * (o.octagon_side / 2) ^ 2 = 16 + 8 * Real.sqrt 2) ∧
  (8 * o.octagon_side = 16 * Real.sqrt 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_properties_l674_67465


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l674_67403

/-- Represents a right cylinder -/
structure RightCylinder where
  height : ℝ
  radius : ℝ

/-- Calculates the total surface area of a right cylinder -/
noncomputable def totalSurfaceArea (c : RightCylinder) : ℝ :=
  2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius ^ 2

/-- Theorem: The total surface area of a right cylinder with height 8 inches and radius 3 inches is 66π square inches -/
theorem cylinder_surface_area :
  let c : RightCylinder := { height := 8, radius := 3 }
  totalSurfaceArea c = 66 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_surface_area_l674_67403
