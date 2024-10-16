import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_interior_angles_is_360_or_540_l2701_270194

/-- A regular polygon with all diagonals equal -/
structure EqualDiagonalRegularPolygon where
  /-- Number of sides of the polygon -/
  n : ℕ
  /-- Condition that the polygon has at least 3 sides -/
  h_n : n ≥ 3
  /-- Condition that all diagonals are equal -/
  all_diagonals_equal : True

/-- The sum of interior angles of a regular polygon with all diagonals equal -/
def sum_of_interior_angles (p : EqualDiagonalRegularPolygon) : ℝ :=
  (p.n - 2) * 180

/-- Theorem stating that the sum of interior angles is either 360° or 540° -/
theorem sum_of_interior_angles_is_360_or_540 (p : EqualDiagonalRegularPolygon) :
  sum_of_interior_angles p = 360 ∨ sum_of_interior_angles p = 540 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_is_360_or_540_l2701_270194


namespace NUMINAMATH_CALUDE_circle_symmetry_l2701_270186

/-- Given a circle with equation x^2 + y^2 + 2x - 4y + 4 = 0 that is symmetric about the line y = 2x + b, prove that b = 4 -/
theorem circle_symmetry (b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 4 = 0 → 
    ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 4 = 0 ∧ 
    y' = 2*x' + b ∧ 
    (x - x')^2 + (y - y')^2 = (x - x')^2 + ((2*x + b) - (2*x' + b))^2) →
  b = 4 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2701_270186


namespace NUMINAMATH_CALUDE_line_point_value_l2701_270166

/-- Given a line with slope 2 passing through (3, 5) and (a, 7), prove a = 4 -/
theorem line_point_value (m : ℝ) (a : ℝ) : 
  m = 2 → -- The line has a slope of 2
  (5 : ℝ) - 5 = m * ((3 : ℝ) - 3) → -- The line passes through (3, 5)
  (7 : ℝ) - 5 = m * (a - 3) → -- The line passes through (a, 7)
  a = 4 := by sorry

end NUMINAMATH_CALUDE_line_point_value_l2701_270166


namespace NUMINAMATH_CALUDE_min_k_value_l2701_270137

theorem min_k_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ k : ℝ, (1 / a + 1 / b + k / (a + b) ≥ 0)) →
  (∃ k_min : ℝ, k_min = -4 ∧ ∀ k : ℝ, (1 / a + 1 / b + k / (a + b) ≥ 0) → k ≥ k_min) :=
by sorry

end NUMINAMATH_CALUDE_min_k_value_l2701_270137


namespace NUMINAMATH_CALUDE_cups_in_first_stack_l2701_270102

theorem cups_in_first_stack (s : Fin 5 → ℕ) 
  (h1 : s 1 = 21)
  (h2 : s 2 = 25)
  (h3 : s 3 = 29)
  (h4 : s 4 = 33)
  (h_arithmetic : ∃ d : ℕ, ∀ i : Fin 4, s (i + 1) = s i + d) :
  s 0 = 17 := by
sorry

end NUMINAMATH_CALUDE_cups_in_first_stack_l2701_270102


namespace NUMINAMATH_CALUDE_max_value_expression_l2701_270112

theorem max_value_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l2701_270112


namespace NUMINAMATH_CALUDE_vector_expression_equality_l2701_270171

variable (V : Type*) [AddCommGroup V] [Module ℝ V]

theorem vector_expression_equality (a b : V) :
  (1 / 3 : ℝ) • ((1 / 2 : ℝ) • (2 • a + 8 • b) - (4 • a - 2 • b)) = 2 • b - a :=
sorry

end NUMINAMATH_CALUDE_vector_expression_equality_l2701_270171


namespace NUMINAMATH_CALUDE_simplify_sqrt_difference_l2701_270192

theorem simplify_sqrt_difference : (Real.sqrt 882 / Real.sqrt 98) - (Real.sqrt 108 / Real.sqrt 12) = 0 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_difference_l2701_270192


namespace NUMINAMATH_CALUDE_red_and_green_peaches_count_l2701_270154

/-- Given a basket of peaches, prove that the total number of red and green peaches is 22. -/
theorem red_and_green_peaches_count (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 6)
  (h2 : green_peaches = 16) : 
  red_peaches + green_peaches = 22 := by
sorry

end NUMINAMATH_CALUDE_red_and_green_peaches_count_l2701_270154


namespace NUMINAMATH_CALUDE_correct_number_of_bills_l2701_270101

/-- The total amount of money in dollars -/
def total_amount : ℕ := 10000

/-- The denomination of each bill in dollars -/
def bill_denomination : ℕ := 50

/-- The number of bills -/
def number_of_bills : ℕ := total_amount / bill_denomination

theorem correct_number_of_bills : number_of_bills = 200 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_of_bills_l2701_270101


namespace NUMINAMATH_CALUDE_cathy_final_state_l2701_270144

/-- Represents a direction on the coordinate plane -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents a position on the coordinate plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents Cathy's state after each move -/
structure CathyState :=
  (position : Position)
  (direction : Direction)
  (moveNumber : Nat)
  (distanceTraveled : Nat)

/-- Calculates the next direction after turning right -/
def turnRight (d : Direction) : Direction :=
  match d with
  | Direction.North => Direction.East
  | Direction.East => Direction.South
  | Direction.South => Direction.West
  | Direction.West => Direction.North

/-- Calculates the distance for a given move number -/
def moveDistance (n : Nat) : Nat :=
  2 * n

/-- Updates the position based on the current direction and distance -/
def updatePosition (p : Position) (d : Direction) (dist : Nat) : Position :=
  match d with
  | Direction.North => ⟨p.x, p.y + dist⟩
  | Direction.East => ⟨p.x + dist, p.y⟩
  | Direction.South => ⟨p.x, p.y - dist⟩
  | Direction.West => ⟨p.x - dist, p.y⟩

/-- Performs a single move and updates Cathy's state -/
def move (state : CathyState) : CathyState :=
  let newMoveNumber := state.moveNumber + 1
  let distance := moveDistance newMoveNumber
  let newPosition := updatePosition state.position state.direction distance
  let newDirection := turnRight state.direction
  let newDistanceTraveled := state.distanceTraveled + distance
  ⟨newPosition, newDirection, newMoveNumber, newDistanceTraveled⟩

/-- Performs n moves starting from the given initial state -/
def performMoves (initialState : CathyState) (n : Nat) : CathyState :=
  match n with
  | 0 => initialState
  | m + 1 => move (performMoves initialState m)

/-- The main theorem to prove -/
theorem cathy_final_state :
  let initialState : CathyState := ⟨⟨2, -3⟩, Direction.North, 0, 0⟩
  let finalState := performMoves initialState 12
  finalState.position = ⟨-10, -15⟩ ∧ finalState.distanceTraveled = 146 := by
  sorry


end NUMINAMATH_CALUDE_cathy_final_state_l2701_270144


namespace NUMINAMATH_CALUDE_cow_count_l2701_270156

/-- Represents the number of animals in the group -/
structure AnimalCount where
  ducks : ℕ
  cows : ℕ

/-- The total number of legs in the group -/
def totalLegs (ac : AnimalCount) : ℕ :=
  2 * ac.ducks + 4 * ac.cows

/-- The total number of heads in the group -/
def totalHeads (ac : AnimalCount) : ℕ :=
  ac.ducks + ac.cows

/-- The problem statement -/
theorem cow_count (ac : AnimalCount) :
  totalLegs ac = 2 * totalHeads ac + 36 → ac.cows = 18 := by
  sorry

end NUMINAMATH_CALUDE_cow_count_l2701_270156


namespace NUMINAMATH_CALUDE_cistern_emptied_l2701_270147

/-- Represents the emptying rate of a pipe in terms of fraction of cistern per minute -/
structure PipeRate where
  fraction : ℚ
  time : ℚ

/-- Calculates the rate at which a pipe empties a cistern -/
def emptyingRate (p : PipeRate) : ℚ :=
  p.fraction / p.time

/-- Calculates the total emptying rate of multiple pipes -/
def totalRate (pipes : List PipeRate) : ℚ :=
  pipes.map emptyingRate |> List.sum

/-- Theorem: Given the specified pipes and time, the entire cistern will be emptied -/
theorem cistern_emptied (pipeA pipeB pipeC : PipeRate) 
    (h1 : pipeA = { fraction := 3/4, time := 12 })
    (h2 : pipeB = { fraction := 1/2, time := 15 })
    (h3 : pipeC = { fraction := 1/3, time := 10 })
    (time : ℚ)
    (h4 : time = 8) :
    totalRate [pipeA, pipeB, pipeC] * time ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_cistern_emptied_l2701_270147


namespace NUMINAMATH_CALUDE_longer_worm_length_l2701_270136

/-- Given two worms, where one is 0.1 inch long and the other is 0.7 inches longer,
    prove that the longer worm is 0.8 inches long. -/
theorem longer_worm_length (short_worm long_worm : ℝ) 
  (h1 : short_worm = 0.1)
  (h2 : long_worm = short_worm + 0.7) :
  long_worm = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_longer_worm_length_l2701_270136


namespace NUMINAMATH_CALUDE_unique_factorial_solution_l2701_270124

theorem unique_factorial_solution : ∃! n : ℕ, n * n.factorial + 2 * n.factorial = 5040 := by
  sorry

end NUMINAMATH_CALUDE_unique_factorial_solution_l2701_270124


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2701_270181

theorem sum_of_coefficients_zero 
  (a₀ a₁ a₂ a₃ a₄ : ℝ) : 
  (∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*(x + 1) + a₂*(x + 1)^2 + a₃*(x + 1)^3 + a₄*(x + 1)^4) → 
  a₁ + a₂ + a₃ + a₄ = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l2701_270181


namespace NUMINAMATH_CALUDE_work_completion_time_l2701_270109

/-- The time (in days) it takes for A to complete the work alone -/
def a_time : ℝ := 30

/-- The time (in days) it takes for A and B to complete the work together -/
def ab_time : ℝ := 19.411764705882355

/-- The time (in days) it takes for B to complete the work alone -/
def b_time : ℝ := 55

/-- Theorem stating that if A can do the work in 30 days, and A and B together can do the work in 19.411764705882355 days, then B can do the work alone in 55 days -/
theorem work_completion_time : 
  (1 / a_time + 1 / b_time = 1 / ab_time) ∧ 
  (a_time > 0) ∧ (b_time > 0) ∧ (ab_time > 0) := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2701_270109


namespace NUMINAMATH_CALUDE_number_operation_l2701_270160

theorem number_operation (x : ℚ) : x - 7/3 = 3/2 → x + 7/3 = 37/6 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_l2701_270160


namespace NUMINAMATH_CALUDE_p_on_x_axis_p_on_line_through_a_m_in_third_quadrant_l2701_270155

-- Define point P as a function of m
def P (m : ℝ) : ℝ × ℝ := (2*m + 5, 3*m + 3)

-- Define point A
def A : ℝ × ℝ := (-5, 1)

-- Define point M as a function of m
def M (m : ℝ) : ℝ × ℝ := (2*m + 7, 3*m + 6)

-- Theorem 1
theorem p_on_x_axis (m : ℝ) : 
  (P m).2 = 0 → m = -1 := by sorry

-- Theorem 2
theorem p_on_line_through_a (m : ℝ) :
  (P m).1 = A.1 → P m = (-5, -12) := by sorry

-- Theorem 3
theorem m_in_third_quadrant (m : ℝ) :
  (M m).1 < 0 ∧ (M m).2 < 0 ∧ |(M m).1| = 7 → M m = (-7, -15) := by sorry

end NUMINAMATH_CALUDE_p_on_x_axis_p_on_line_through_a_m_in_third_quadrant_l2701_270155


namespace NUMINAMATH_CALUDE_distance_to_midpoint_l2701_270118

-- Define the triangle PQR
structure RightTriangle where
  PQ : ℝ
  PR : ℝ
  QR : ℝ
  is_right : PQ^2 = PR^2 + QR^2

-- Define the specific triangle given in the problem
def triangle_PQR : RightTriangle :=
  { PQ := 15
    PR := 9
    QR := 12
    is_right := by norm_num }

-- Theorem statement
theorem distance_to_midpoint (t : RightTriangle) (h : t = triangle_PQR) :
  (t.PQ / 2 : ℝ) = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_midpoint_l2701_270118


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2701_270134

theorem inequality_solution_set :
  {x : ℝ | x^2 + 2*x - 3 ≥ 0} = {x : ℝ | x ≤ -3 ∨ x ≥ 1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2701_270134


namespace NUMINAMATH_CALUDE_xy_max_value_l2701_270120

theorem xy_max_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 * x + 2 * y = 12) :
  x * y ≤ 6 := by
sorry

end NUMINAMATH_CALUDE_xy_max_value_l2701_270120


namespace NUMINAMATH_CALUDE_expand_expression_l2701_270138

theorem expand_expression (x : ℝ) : 5 * (9 * x^3 - 4 * x^2 + 3 * x - 7) = 45 * x^3 - 20 * x^2 + 15 * x - 35 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l2701_270138


namespace NUMINAMATH_CALUDE_shaded_area_between_circles_l2701_270128

theorem shaded_area_between_circles (r₁ r₂ : ℝ) (h₁ : r₁ = 3) (h₂ : r₂ = 4) : 
  let R := r₁ + r₂
  let A_large := π * R^2
  let A_small₁ := π * r₁^2
  let A_small₂ := π * r₂^2
  let A_shaded := A_large - A_small₁ - A_small₂
  A_shaded = 24 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_between_circles_l2701_270128


namespace NUMINAMATH_CALUDE_ps_length_l2701_270187

/-- Represents a quadrilateral PQRS with specific properties -/
structure Quadrilateral where
  PQ : ℝ
  QR : ℝ
  RS : ℝ
  sinR : ℝ
  cosQ : ℝ
  R_obtuse : Bool

/-- The theorem stating the length of PS in the given quadrilateral -/
theorem ps_length (quad : Quadrilateral)
  (h1 : quad.PQ = 6)
  (h2 : quad.QR = 7)
  (h3 : quad.RS = 25)
  (h4 : quad.sinR = 4/5)
  (h5 : quad.cosQ = -4/5)
  (h6 : quad.R_obtuse = true) :
  ∃ (PS : ℝ), PS^2 = 794 :=
sorry

end NUMINAMATH_CALUDE_ps_length_l2701_270187


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2701_270129

/-- The imaginary unit i -/
noncomputable def i : ℂ := Complex.I

/-- Statement: i^11 + i^111 = -2i -/
theorem imaginary_power_sum : i^11 + i^111 = -2*i := by sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2701_270129


namespace NUMINAMATH_CALUDE_circular_fields_radius_l2701_270114

theorem circular_fields_radius (r₁ r₂ : ℝ) : 
  r₂ = 10 →
  π * r₁^2 = 0.09 * (π * r₂^2) →
  r₁ = 3 := by
sorry

end NUMINAMATH_CALUDE_circular_fields_radius_l2701_270114


namespace NUMINAMATH_CALUDE_cosine_value_from_sine_l2701_270168

theorem cosine_value_from_sine (θ : Real) (h1 : 0 < θ) (h2 : θ < π / 2) 
  (h3 : Real.sin (θ / 2 + π / 6) = 3 / 5) : 
  Real.cos (θ + 5 * π / 6) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_cosine_value_from_sine_l2701_270168


namespace NUMINAMATH_CALUDE_negation_equivalence_l2701_270142

theorem negation_equivalence : 
  ¬(∀ x : ℝ, (x ≠ 3 ∧ x ≠ 2) → (x^2 - 5*x + 6 ≠ 0)) ↔ 
  (∀ x : ℝ, (x = 3 ∨ x = 2) → (x^2 - 5*x + 6 = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2701_270142


namespace NUMINAMATH_CALUDE_original_price_calculation_l2701_270185

/-- Calculates the original price of an article given the profit percentage and profit amount. -/
def calculate_original_price (profit_percentage : ℚ) (profit_amount : ℚ) : ℚ :=
  profit_amount / (profit_percentage / 100)

/-- Theorem: Given an article sold at a 50% profit, where the profit is Rs. 750, 
    the original price of the article was Rs. 1500. -/
theorem original_price_calculation :
  calculate_original_price 50 750 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2701_270185


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l2701_270117

theorem intersection_implies_a_value (a : ℝ) : 
  let A : Set ℝ := {-1, 2, 3}
  let B : Set ℝ := {a + 2, a^2 + 2}
  A ∩ B = {3} → a = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l2701_270117


namespace NUMINAMATH_CALUDE_min_value_of_f_l2701_270131

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem min_value_of_f :
  ∃ (min_val : ℝ), min_val = -2 ∧
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → f x ≥ min_val :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2701_270131


namespace NUMINAMATH_CALUDE_zoo_recovery_time_l2701_270148

/-- The total time spent recovering escaped animals from a zoo -/
def total_recovery_time (num_lions num_rhinos num_giraffes num_gorillas : ℕ) 
  (time_per_lion time_per_rhino time_per_giraffe time_per_gorilla : ℝ) : ℝ :=
  (num_lions : ℝ) * time_per_lion + 
  (num_rhinos : ℝ) * time_per_rhino + 
  (num_giraffes : ℝ) * time_per_giraffe + 
  (num_gorillas : ℝ) * time_per_gorilla

/-- Theorem stating that the total recovery time for the given scenario is 33 hours -/
theorem zoo_recovery_time : 
  total_recovery_time 5 3 2 4 2 3 4 1.5 = 33 := by
  sorry

end NUMINAMATH_CALUDE_zoo_recovery_time_l2701_270148


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_2023_l2701_270173

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (fun e => 2^e)).sum ∧ exponents.Nodup

theorem least_exponent_sum_for_2023 :
  ∃ (exponents : List ℕ),
    is_sum_of_distinct_powers_of_two 2023 exponents ∧
    exponents.sum = 48 ∧
    ∀ (other_exponents : List ℕ),
      is_sum_of_distinct_powers_of_two 2023 other_exponents →
      other_exponents.sum ≥ 48 :=
sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_2023_l2701_270173


namespace NUMINAMATH_CALUDE_article_purchase_price_l2701_270149

/-- The purchase price of an article given specific markup conditions -/
theorem article_purchase_price : 
  ∀ (markup overhead_percentage net_profit purchase_price : ℝ),
  markup = 40 →
  overhead_percentage = 0.15 →
  net_profit = 12 →
  markup = overhead_percentage * purchase_price + net_profit →
  purchase_price = 186.67 := by
sorry

end NUMINAMATH_CALUDE_article_purchase_price_l2701_270149


namespace NUMINAMATH_CALUDE_probability_greater_than_four_l2701_270121

-- Define a standard six-sided die
def standardDie : Finset Nat := Finset.range 6

-- Define the probability of an event on the die
def probability (event : Finset Nat) : Rat :=
  event.card / standardDie.card

-- Define the event of rolling a number greater than 4
def greaterThanFour : Finset Nat := Finset.filter (λ x => x > 4) standardDie

-- Theorem statement
theorem probability_greater_than_four :
  probability greaterThanFour = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_greater_than_four_l2701_270121


namespace NUMINAMATH_CALUDE_irrational_count_l2701_270191

-- Define the set of numbers
def S : Set ℝ := {4 * Real.pi, 0, Real.sqrt 7, Real.sqrt 16 / 2, 0.1, 0.212212221}

-- Define a function to count irrational numbers in a set
def count_irrational (T : Set ℝ) : ℕ := sorry

-- Theorem statement
theorem irrational_count : count_irrational S = 3 := by sorry

end NUMINAMATH_CALUDE_irrational_count_l2701_270191


namespace NUMINAMATH_CALUDE_street_light_ratio_l2701_270188

theorem street_light_ratio (first_month : ℕ) (second_month : ℕ) (remaining : ℕ) :
  first_month = 1200 →
  second_month = 1300 →
  remaining = 500 →
  (first_month + second_month) / remaining = 5 := by
  sorry

end NUMINAMATH_CALUDE_street_light_ratio_l2701_270188


namespace NUMINAMATH_CALUDE_complex_subtraction_simplification_l2701_270123

theorem complex_subtraction_simplification :
  (4 - 3*Complex.I) - (7 - 5*Complex.I) = -3 + 2*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_simplification_l2701_270123


namespace NUMINAMATH_CALUDE_largest_integer_in_range_l2701_270135

theorem largest_integer_in_range : 
  ∃ (x : ℤ), (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 3/5 ∧ 
  ∀ (y : ℤ), (1/4 : ℚ) < (y : ℚ)/7 ∧ (y : ℚ)/7 < 3/5 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_range_l2701_270135


namespace NUMINAMATH_CALUDE_simplify_fraction_sum_l2701_270164

theorem simplify_fraction_sum (a b c d : ℕ) (h1 : a * d = b * c) (h2 : Nat.gcd a b = 1) :
  a + b = 11 → 75 * d = 200 * c :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_sum_l2701_270164


namespace NUMINAMATH_CALUDE_monotonic_intervals_range_of_a_l2701_270170

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (1/2) * a * x^2

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x - a * x

-- Theorem for Part I
theorem monotonic_intervals (a : ℝ) (h : a ≤ 1) :
  (∀ x < 0, a ≤ 0 → (f' a x < 0)) ∧
  (∀ x > 0, a ≤ 0 → (f' a x > 0)) ∧
  (∀ x < Real.log a, 0 < a → a < 1 → (f' a x > 0)) ∧
  (∀ x ∈ Set.Ioo (Real.log a) 0, 0 < a → a < 1 → (f' a x < 0)) ∧
  (∀ x > 0, 0 < a → a < 1 → (f' a x > 0)) ∧
  (∀ x : ℝ, a = 1 → (f' a x ≥ 0)) :=
sorry

-- Theorem for Part II
theorem range_of_a (a : ℝ) :
  (∀ x > 0, f' a x > a * x^3 + x^2 - (a - 1) * x) ↔ a ∈ Set.Iic (1/2) :=
sorry

end NUMINAMATH_CALUDE_monotonic_intervals_range_of_a_l2701_270170


namespace NUMINAMATH_CALUDE_sine_symmetry_axis_l2701_270113

/-- The symmetry axis of the graph of y = sin(x - π/3) is x = -π/6 -/
theorem sine_symmetry_axis :
  ∀ x : ℝ, (∀ y : ℝ, y = Real.sin (x - π/3)) →
  (∃ k : ℤ, x = -π/6 + k * π) :=
sorry

end NUMINAMATH_CALUDE_sine_symmetry_axis_l2701_270113


namespace NUMINAMATH_CALUDE_five_a_value_l2701_270163

theorem five_a_value (a : ℝ) (h : 5 * (a - 3) = 25) : 5 * a = 40 := by
  sorry

end NUMINAMATH_CALUDE_five_a_value_l2701_270163


namespace NUMINAMATH_CALUDE_joe_total_cars_l2701_270161

def initial_cars : ℕ := 500
def additional_cars : ℕ := 120

theorem joe_total_cars : initial_cars + additional_cars = 620 := by
  sorry

end NUMINAMATH_CALUDE_joe_total_cars_l2701_270161


namespace NUMINAMATH_CALUDE_hash_2_3_neg1_l2701_270193

def hash (a b c : ℝ) : ℝ := b^3 - 4*a*c + b

theorem hash_2_3_neg1 : hash 2 3 (-1) = 38 := by
  sorry

end NUMINAMATH_CALUDE_hash_2_3_neg1_l2701_270193


namespace NUMINAMATH_CALUDE_subtracted_value_l2701_270199

theorem subtracted_value (x y : ℤ) : x = 60 ∧ 4 * x - y = 102 → y = 138 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l2701_270199


namespace NUMINAMATH_CALUDE_line_through_points_l2701_270197

/-- Given a line with equation x = 5y + 5 passing through points (m, n) and (m + 2, n + p),
    prove that p = 2/5 -/
theorem line_through_points (m n : ℝ) : 
  let p : ℝ := 2/5
  m = 5*n + 5 ∧ (m + 2) = 5*(n + p) + 5 → p = 2/5 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_l2701_270197


namespace NUMINAMATH_CALUDE_roots_of_equation_l2701_270174

theorem roots_of_equation (x : ℝ) : 
  (x - 3)^2 = 3 - x ↔ x = 3 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2701_270174


namespace NUMINAMATH_CALUDE_animals_fiber_intake_l2701_270195

-- Define the absorption rates and absorbed amounts
def koala_absorption_rate : ℝ := 0.30
def koala_absorbed_amount : ℝ := 12
def kangaroo_absorption_rate : ℝ := 0.40
def kangaroo_absorbed_amount : ℝ := 16

-- Define the theorem
theorem animals_fiber_intake :
  ∃ (koala_intake kangaroo_intake : ℝ),
    koala_intake * koala_absorption_rate = koala_absorbed_amount ∧
    kangaroo_intake * kangaroo_absorption_rate = kangaroo_absorbed_amount ∧
    koala_intake = 40 ∧
    kangaroo_intake = 40 := by
  sorry

end NUMINAMATH_CALUDE_animals_fiber_intake_l2701_270195


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2701_270172

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A nine-sided polygon has 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l2701_270172


namespace NUMINAMATH_CALUDE_product_selection_problem_l2701_270132

def total_products : ℕ := 12
def genuine_products : ℕ := 10
def defective_products : ℕ := 2
def selected_products : ℕ := 3

theorem product_selection_problem :
  (Nat.choose total_products selected_products = 220) ∧
  (Nat.choose defective_products 1 * Nat.choose genuine_products 2 = 90) ∧
  (Nat.choose total_products selected_products - Nat.choose genuine_products selected_products = 100) :=
by sorry

end NUMINAMATH_CALUDE_product_selection_problem_l2701_270132


namespace NUMINAMATH_CALUDE_stickers_needed_for_prizes_l2701_270151

def christine_stickers : ℕ := 2500
def robert_stickers : ℕ := 1750
def small_prize_requirement : ℕ := 4000
def medium_prize_requirement : ℕ := 7000
def large_prize_requirement : ℕ := 10000

def total_stickers : ℕ := christine_stickers + robert_stickers

theorem stickers_needed_for_prizes :
  (max 0 (small_prize_requirement - total_stickers) = 0) ∧
  (max 0 (medium_prize_requirement - total_stickers) = 2750) ∧
  (max 0 (large_prize_requirement - total_stickers) = 5750) := by
  sorry

end NUMINAMATH_CALUDE_stickers_needed_for_prizes_l2701_270151


namespace NUMINAMATH_CALUDE_least_number_of_trees_least_number_of_trees_is_168_l2701_270139

theorem least_number_of_trees : ℕ → Prop :=
  fun n => (n % 4 = 0) ∧ 
           (n % 7 = 0) ∧ 
           (n % 6 = 0) ∧ 
           (n % 4 = 0) ∧ 
           (n ≥ 100) ∧ 
           (∀ m : ℕ, m < n → ¬(least_number_of_trees m))

theorem least_number_of_trees_is_168 : 
  least_number_of_trees 168 := by sorry

end NUMINAMATH_CALUDE_least_number_of_trees_least_number_of_trees_is_168_l2701_270139


namespace NUMINAMATH_CALUDE_perfect_cube_values_l2701_270198

theorem perfect_cube_values (Z K : ℤ) (h1 : 600 < Z) (h2 : Z < 2000) (h3 : K > 1) (h4 : Z = K^3) :
  K = 9 ∨ K = 10 ∨ K = 11 ∨ K = 12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_cube_values_l2701_270198


namespace NUMINAMATH_CALUDE_calculate_expression_l2701_270167

theorem calculate_expression : 4 + (-2)^2 * 2 + (-36) / 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2701_270167


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l2701_270122

theorem binomial_expansion_problem (m n : ℕ) (hm : m ≠ 0) (hn : n ≥ 2) :
  (∀ k, k ∈ Finset.range (n + 1) → k ≠ 5 → Nat.choose n k ≤ Nat.choose n 5) ∧
  Nat.choose n 2 * m^2 = 9 * Nat.choose n 1 * m →
  m = 2 ∧ n = 10 ∧ ((-17)^n) % 6 = 1 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l2701_270122


namespace NUMINAMATH_CALUDE_rulers_produced_l2701_270140

theorem rulers_produced (rulers_per_minute : ℕ) (minutes : ℕ) : 
  rulers_per_minute = 8 → minutes = 15 → rulers_per_minute * minutes = 120 := by
  sorry

end NUMINAMATH_CALUDE_rulers_produced_l2701_270140


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2701_270125

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2701_270125


namespace NUMINAMATH_CALUDE_red_balls_count_l2701_270176

theorem red_balls_count (yellow_balls : ℕ) (total_balls : ℕ) (prob_yellow : ℚ) :
  yellow_balls = 10 →
  prob_yellow = 5/8 →
  total_balls ≤ 32 →
  total_balls = yellow_balls + (total_balls - yellow_balls) →
  (yellow_balls : ℚ) / total_balls = prob_yellow →
  total_balls - yellow_balls = 6 :=
by sorry

end NUMINAMATH_CALUDE_red_balls_count_l2701_270176


namespace NUMINAMATH_CALUDE_squirrel_pine_cones_l2701_270130

/-- The number of pine cones the squirrel planned to eat per day -/
def planned_daily_cones : ℕ := 6

/-- The additional number of pine cones the squirrel actually ate per day -/
def additional_daily_cones : ℕ := 2

/-- The number of days earlier the pine cones were finished -/
def days_earlier : ℕ := 5

/-- The total number of pine cones stored by the squirrel -/
def total_cones : ℕ := 120

theorem squirrel_pine_cones :
  ∃ (planned_days : ℕ),
    planned_days * planned_daily_cones =
    (planned_days - days_earlier) * (planned_daily_cones + additional_daily_cones) ∧
    total_cones = planned_days * planned_daily_cones :=
by sorry

end NUMINAMATH_CALUDE_squirrel_pine_cones_l2701_270130


namespace NUMINAMATH_CALUDE_range_of_r_l2701_270107

theorem range_of_r (a b c r : ℝ) 
  (h1 : b + c ≤ 4 * a)
  (h2 : c - b ≥ 0)
  (h3 : b ≥ a)
  (h4 : a > 0)
  (h5 : r > 0)
  (h6 : (a + b)^2 + (a + c)^2 ≠ (a * r)^2) :
  r ∈ Set.Ioo 0 (2 * Real.sqrt 2) ∪ Set.Ioi (3 * Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_r_l2701_270107


namespace NUMINAMATH_CALUDE_two_solutions_exist_l2701_270105

/-- A structure representing a triangle with integer side lengths -/
structure IntegerTriangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  triangle_inequality : a.val < b.val + c.val ∧ b.val < a.val + c.val ∧ c.val < a.val + b.val

/-- The condition from the original problem -/
def satisfies_equation (t : IntegerTriangle) : Prop :=
  (t.a.val * t.b.val * t.c.val : ℕ) = 2 * (t.a.val - 1) * (t.b.val - 1) * (t.c.val - 1)

/-- The main theorem stating that there are exactly two solutions -/
theorem two_solutions_exist : 
  (∃ (t1 t2 : IntegerTriangle), 
    satisfies_equation t1 ∧ 
    satisfies_equation t2 ∧ 
    t1 ≠ t2 ∧ 
    (∀ (t : IntegerTriangle), satisfies_equation t → (t = t1 ∨ t = t2))) ∧
  (∃ (t1 : IntegerTriangle), t1.a = 8 ∧ t1.b = 7 ∧ t1.c = 3 ∧ satisfies_equation t1) ∧
  (∃ (t2 : IntegerTriangle), t2.a = 6 ∧ t2.b = 5 ∧ t2.c = 4 ∧ satisfies_equation t2) :=
by sorry


end NUMINAMATH_CALUDE_two_solutions_exist_l2701_270105


namespace NUMINAMATH_CALUDE_typhoon_tree_problem_l2701_270159

theorem typhoon_tree_problem (initial_trees : ℕ) 
  (h1 : initial_trees = 13) 
  (dead_trees : ℕ) 
  (surviving_trees : ℕ) 
  (h2 : surviving_trees = dead_trees + 1) 
  (h3 : dead_trees + surviving_trees = initial_trees) : 
  dead_trees = 6 := by
sorry

end NUMINAMATH_CALUDE_typhoon_tree_problem_l2701_270159


namespace NUMINAMATH_CALUDE_quadrilateral_area_at_least_30_l2701_270108

-- Define the quadrilateral EFGH
structure Quadrilateral :=
  (E F G H : ℝ × ℝ)

-- Define the conditions
def is_valid_quadrilateral (q : Quadrilateral) : Prop :=
  let (ex, ey) := q.E
  let (fx, fy) := q.F
  let (gx, gy) := q.G
  let (hx, hy) := q.H
  -- EF = 5
  (ex - fx)^2 + (ey - fy)^2 = 25 ∧
  -- FG = 12
  (fx - gx)^2 + (fy - gy)^2 = 144 ∧
  -- GH = 5
  (gx - hx)^2 + (gy - hy)^2 = 25 ∧
  -- HE = 13
  (hx - ex)^2 + (hy - ey)^2 = 169 ∧
  -- Angle EFG is a right angle
  (ex - fx) * (gx - fx) + (ey - fy) * (gy - fy) = 0

-- Define the area function
def area (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_area_at_least_30 (q : Quadrilateral) :
  is_valid_quadrilateral q → area q ≥ 30 := by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_at_least_30_l2701_270108


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2701_270146

theorem greatest_three_digit_multiple_of_17 : ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2701_270146


namespace NUMINAMATH_CALUDE_second_discount_percentage_l2701_270115

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ)
  (h1 : original_price = 400)
  (h2 : first_discount = 10)
  (h3 : final_price = 342) :
  let price_after_first_discount := original_price * (1 - first_discount / 100)
  let second_discount := (price_after_first_discount - final_price) / price_after_first_discount * 100
  second_discount = 5 := by
sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l2701_270115


namespace NUMINAMATH_CALUDE_divisibility_by_five_l2701_270189

theorem divisibility_by_five (n : ℤ) : 
  ∃ (m : ℤ), 3 * (n^2 + n) + 7 = 5 * m ↔ ∃ (k : ℤ), n = 5 * k + 2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_five_l2701_270189


namespace NUMINAMATH_CALUDE_equation_solutions_l2701_270145

theorem equation_solutions :
  (∃ x : ℝ, 9.9 + x = -18 ∧ x = -27.9) ∧
  (∃ x : ℝ, x - 8.8 = -8.8 ∧ x = 0) ∧
  (∃ x : ℚ, -3/4 + x = -1/4 ∧ x = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2701_270145


namespace NUMINAMATH_CALUDE_sugar_profit_percentage_l2701_270157

theorem sugar_profit_percentage (total_sugar : ℝ) (sugar_at_12_percent : ℝ) (overall_profit_percent : ℝ) :
  total_sugar = 1600 →
  sugar_at_12_percent = 1200 →
  overall_profit_percent = 11 →
  let remaining_sugar := total_sugar - sugar_at_12_percent
  let profit_12_percent := sugar_at_12_percent * 12 / 100
  let total_profit := total_sugar * overall_profit_percent / 100
  let remaining_profit := total_profit - profit_12_percent
  remaining_profit / remaining_sugar * 100 = 8 :=
by sorry

end NUMINAMATH_CALUDE_sugar_profit_percentage_l2701_270157


namespace NUMINAMATH_CALUDE_shoe_difference_l2701_270141

/-- Given information about shoe boxes and quantities, prove the difference in pairs of shoes. -/
theorem shoe_difference (pairs_per_box : ℕ) (boxes_of_A : ℕ) (B_to_A_ratio : ℕ) : 
  pairs_per_box = 20 →
  boxes_of_A = 8 →
  B_to_A_ratio = 5 →
  B_to_A_ratio * (pairs_per_box * boxes_of_A) - (pairs_per_box * boxes_of_A) = 640 := by
  sorry

end NUMINAMATH_CALUDE_shoe_difference_l2701_270141


namespace NUMINAMATH_CALUDE_circle_and_line_problem_l2701_270190

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 2)^2 + (p.2 - 3)^2 = 1}

-- Define points A, B, and D
def point_A : ℝ × ℝ := (1, 3)
def point_B : ℝ × ℝ := (2, 2)
def point_D : ℝ × ℝ := (0, 1)

-- Define line m
def line_m (x y : ℝ) : Prop := 3 * x - 2 * y = 0

-- Define line l with slope k
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * (x - 0)

-- Define the property that line m bisects circle C
def bisects (l : (ℝ → ℝ → Prop)) (c : Set (ℝ × ℝ)) : Prop := sorry

-- Define the property that a line intersects a circle at two distinct points
def intersects_at_two_points (l : (ℝ → ℝ → Prop)) (c : Set (ℝ × ℝ)) : Prop := sorry

-- Define the squared distance between two points on a line
def squared_distance (l : (ℝ → ℝ → Prop)) (c : Set (ℝ × ℝ)) : ℝ := sorry

theorem circle_and_line_problem (k : ℝ) :
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  bisects line_m circle_C ∧
  intersects_at_two_points (line_l k) circle_C ∧
  squared_distance (line_l k) circle_C = 12 →
  k = 1 := by sorry

end NUMINAMATH_CALUDE_circle_and_line_problem_l2701_270190


namespace NUMINAMATH_CALUDE_johns_remaining_money_l2701_270110

/-- Converts a number from base 8 to base 10 -/
def base8_to_base10 (n : ℕ) : ℕ := sorry

/-- Proves that John's remaining money after buying the ticket is 1725 dollars -/
theorem johns_remaining_money :
  let savings := base8_to_base10 5555
  let ticket_cost := 1200
  savings - ticket_cost = 1725 := by sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l2701_270110


namespace NUMINAMATH_CALUDE_man_son_age_difference_man_son_age_difference_proof_l2701_270126

theorem man_son_age_difference : ℕ → ℕ → Prop :=
  fun man_age son_age =>
    son_age = 22 →
    man_age + 2 = 2 * (son_age + 2) →
    man_age - son_age = 24

-- Proof
theorem man_son_age_difference_proof :
  ∃ (man_age son_age : ℕ), man_son_age_difference man_age son_age := by
  sorry

end NUMINAMATH_CALUDE_man_son_age_difference_man_son_age_difference_proof_l2701_270126


namespace NUMINAMATH_CALUDE_power_function_value_l2701_270178

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, x > 0 → f x = x^a

-- State the theorem
theorem power_function_value 
  (f : ℝ → ℝ) 
  (h1 : isPowerFunction f) 
  (h2 : f 4 = 1/2) : 
  f (1/16) = 4 := by
sorry

end NUMINAMATH_CALUDE_power_function_value_l2701_270178


namespace NUMINAMATH_CALUDE_symmetric_line_l2701_270180

/-- Given a line l and another line, find the equation of the line symmetric to the given line with respect to l -/
theorem symmetric_line (a b c d e f : ℝ) :
  let l : ℝ → ℝ := λ x => 3 * x + 3
  let given_line : ℝ → ℝ := λ x => x - 2
  let symmetric_line : ℝ → ℝ := λ x => -7 * x - 22
  (∀ x, given_line x = x - (l x)) →
  (∀ x, symmetric_line x = (l x) - (given_line x - (l x))) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_line_l2701_270180


namespace NUMINAMATH_CALUDE_correct_calculation_l2701_270150

theorem correct_calculation (x : ℚ) (h : 6 * x = 42) : 3 * x = 21 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2701_270150


namespace NUMINAMATH_CALUDE_ellipse_sum_range_l2701_270100

theorem ellipse_sum_range (x y : ℝ) (h : 9 * x^2 + 16 * y^2 = 144) :
  5 ≤ x + y + 10 ∧ x + y + 10 ≤ 15 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_sum_range_l2701_270100


namespace NUMINAMATH_CALUDE_credit_card_balance_calculation_l2701_270196

/-- Calculates the final balance on a credit card after two interest applications and an additional charge. -/
def finalBalance (initialBalance : ℝ) (interestRate : ℝ) (additionalCharge : ℝ) : ℝ :=
  let balanceAfterFirstInterest := initialBalance * (1 + interestRate)
  let balanceAfterCharge := balanceAfterFirstInterest + additionalCharge
  balanceAfterCharge * (1 + interestRate)

/-- Theorem stating that given the specific conditions, the final balance is $96.00 -/
theorem credit_card_balance_calculation :
  finalBalance 50 0.2 20 = 96 := by
  sorry

end NUMINAMATH_CALUDE_credit_card_balance_calculation_l2701_270196


namespace NUMINAMATH_CALUDE_traditionalist_fraction_l2701_270111

theorem traditionalist_fraction (num_provinces : ℕ) (num_progressives : ℕ) (num_traditionalists_per_province : ℕ) :
  num_provinces = 4 →
  num_traditionalists_per_province * 12 = num_progressives →
  (num_traditionalists_per_province * num_provinces) / (num_progressives + num_traditionalists_per_province * num_provinces) = 1 / 4 :=
by sorry

end NUMINAMATH_CALUDE_traditionalist_fraction_l2701_270111


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_l2701_270152

/-- The function f(x) = x - a --/
def f (a : ℝ) (x : ℝ) : ℝ := x - a

/-- The open interval (0, 1) --/
def open_unit_interval : Set ℝ := { x | 0 < x ∧ x < 1 }

/-- f has a zero in (0, 1) --/
def has_zero_in_unit_interval (a : ℝ) : Prop :=
  ∃ x ∈ open_unit_interval, f a x = 0

theorem necessary_not_sufficient :
  (∀ a : ℝ, has_zero_in_unit_interval a → a < 1) ∧
  (∃ a : ℝ, a < 1 ∧ ¬has_zero_in_unit_interval a) := by sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_l2701_270152


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l2701_270103

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 7 + 2 * Real.sqrt 7 ∧ x₂ = 7 - 2 * Real.sqrt 7 ∧
    x₁^2 - 14*x₁ + 21 = 0 ∧ x₂^2 - 14*x₂ + 21 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 2 ∧
    x₁^2 - 3*x₁ + 2 = 0 ∧ x₂^2 - 3*x₂ + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l2701_270103


namespace NUMINAMATH_CALUDE_racecar_repair_cost_l2701_270165

/-- Proves that the original cost of fixing a racecar was $20,000 given specific conditions --/
theorem racecar_repair_cost 
  (discount_rate : Real) 
  (prize : Real) 
  (prize_keep_rate : Real) 
  (net_profit : Real) :
  discount_rate = 0.2 →
  prize = 70000 →
  prize_keep_rate = 0.9 →
  net_profit = 47000 →
  ∃ (original_cost : Real),
    original_cost = 20000 ∧
    prize * prize_keep_rate - original_cost * (1 - discount_rate) = net_profit :=
by
  sorry

end NUMINAMATH_CALUDE_racecar_repair_cost_l2701_270165


namespace NUMINAMATH_CALUDE_logarithm_expression_equality_l2701_270133

theorem logarithm_expression_equality : 
  (Real.log 243 / Real.log 3) / (Real.log 3 / Real.log 81) - 
  (Real.log 729 / Real.log 3) / (Real.log 3 / Real.log 27) = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_expression_equality_l2701_270133


namespace NUMINAMATH_CALUDE_horner_method_evaluation_l2701_270175

def horner_polynomial (x : ℝ) : ℝ :=
  ((((4 * x + 3) * x + 4) * x + 2) * x + 5) * x - 7 * x + 9

theorem horner_method_evaluation :
  horner_polynomial 4 = 20669 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_evaluation_l2701_270175


namespace NUMINAMATH_CALUDE_T_is_far_right_l2701_270104

/-- Represents a rectangle with four integer-labeled sides --/
structure Rectangle where
  w : Int
  x : Int
  y : Int
  z : Int

/-- Checks if a rectangle is at the far-right end of the row --/
def is_far_right (r : Rectangle) (others : List Rectangle) : Prop :=
  ∀ other ∈ others, r.y ≥ other.y ∧ (r.y = other.y → r.w ≥ other.w)

/-- The given rectangles --/
def P : Rectangle := ⟨3, 0, 9, 5⟩
def Q : Rectangle := ⟨6, 1, 0, 8⟩
def R : Rectangle := ⟨0, 3, 2, 7⟩
def S : Rectangle := ⟨8, 5, 4, 1⟩
def T : Rectangle := ⟨5, 2, 6, 9⟩

theorem T_is_far_right :
  is_far_right T [P, Q, R, S] :=
sorry

end NUMINAMATH_CALUDE_T_is_far_right_l2701_270104


namespace NUMINAMATH_CALUDE_acid_mixture_proof_l2701_270169

-- Define the volumes and concentrations
def volume_60_percent : ℝ := 4
def concentration_60_percent : ℝ := 0.60
def volume_75_percent : ℝ := 16
def concentration_75_percent : ℝ := 0.75
def total_volume : ℝ := 20
def final_concentration : ℝ := 0.72

-- Theorem statement
theorem acid_mixture_proof :
  (volume_60_percent * concentration_60_percent + 
   volume_75_percent * concentration_75_percent) / total_volume = final_concentration :=
by sorry

end NUMINAMATH_CALUDE_acid_mixture_proof_l2701_270169


namespace NUMINAMATH_CALUDE_max_k_for_intersecting_circles_l2701_270158

/-- The maximum value of k for which a circle with radius 1 centered on the line y = kx - 2 
    intersects the circle x² + y² - 8x + 15 = 0 -/
theorem max_k_for_intersecting_circles : 
  ∃ (max_k : ℝ), max_k = 4/3 ∧ 
  (∀ k : ℝ, (∃ x y : ℝ, 
    y = k * x - 2 ∧ 
    (∃ cx cy : ℝ, (cx - x)^2 + (cy - y)^2 = 1 ∧ 
      cx^2 + cy^2 - 8*cx + 15 = 0)) → 
    k ≤ max_k) ∧
  (∃ x y : ℝ, 
    y = max_k * x - 2 ∧ 
    (∃ cx cy : ℝ, (cx - x)^2 + (cy - y)^2 = 1 ∧ 
      cx^2 + cy^2 - 8*cx + 15 = 0)) :=
sorry

end NUMINAMATH_CALUDE_max_k_for_intersecting_circles_l2701_270158


namespace NUMINAMATH_CALUDE_perpendicular_planes_from_line_l2701_270116

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- m is contained in α -/
def contained_in (m : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- m is perpendicular to β -/
def perpendicular_line_plane (m : Line3D) (β : Plane3D) : Prop :=
  sorry

/-- α is perpendicular to β -/
def perpendicular_planes (α β : Plane3D) : Prop :=
  sorry

/-- If a line m is contained in a plane α and is perpendicular to another plane β, 
    then α is perpendicular to β -/
theorem perpendicular_planes_from_line 
  (m : Line3D) (α β : Plane3D) : 
  contained_in m α → perpendicular_line_plane m β → perpendicular_planes α β :=
by
  sorry

end NUMINAMATH_CALUDE_perpendicular_planes_from_line_l2701_270116


namespace NUMINAMATH_CALUDE_solution_x_l2701_270184

theorem solution_x (x y : ℝ) 
  (h1 : (2010 + x)^2 = x^2) 
  (h2 : x = 5*y + 2) : 
  x = -1005 := by
sorry

end NUMINAMATH_CALUDE_solution_x_l2701_270184


namespace NUMINAMATH_CALUDE_train_crossing_platform_time_l2701_270162

/-- Calculates the time taken for a train to cross a platform -/
theorem train_crossing_platform_time 
  (train_length : ℝ) 
  (signal_pole_time : ℝ) 
  (platform_length : ℝ) 
  (h1 : train_length = 200) 
  (h2 : signal_pole_time = 42) 
  (h3 : platform_length = 38.0952380952381) : 
  (train_length + platform_length) / (train_length / signal_pole_time) = 50 := by
  sorry

#check train_crossing_platform_time

end NUMINAMATH_CALUDE_train_crossing_platform_time_l2701_270162


namespace NUMINAMATH_CALUDE_polynomial_simplification_l2701_270179

theorem polynomial_simplification (q : ℝ) : 
  (5 * q^4 + 3 * q^3 - 7 * q + 8) + (6 - 9 * q^3 + 4 * q - 3 * q^4) = 
  2 * q^4 - 6 * q^3 - 3 * q + 14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l2701_270179


namespace NUMINAMATH_CALUDE_b_22_mod_35_l2701_270143

/-- Concatenates integers from 1 to n --/
def b (n : ℕ) : ℕ :=
  sorry

/-- The main theorem --/
theorem b_22_mod_35 : b 22 % 35 = 17 := by
  sorry

end NUMINAMATH_CALUDE_b_22_mod_35_l2701_270143


namespace NUMINAMATH_CALUDE_max_product_with_constraints_l2701_270153

theorem max_product_with_constraints :
  ∀ a b : ℕ,
  a + b = 100 →
  a % 3 = 2 →
  b % 7 = 5 →
  ∀ x y : ℕ,
  x + y = 100 →
  x % 3 = 2 →
  y % 7 = 5 →
  a * b ≤ 2491 ∧ (∃ a b : ℕ, a + b = 100 ∧ a % 3 = 2 ∧ b % 7 = 5 ∧ a * b = 2491) :=
by
  sorry

end NUMINAMATH_CALUDE_max_product_with_constraints_l2701_270153


namespace NUMINAMATH_CALUDE_blackboard_area_difference_l2701_270127

/-- The difference between the area of a square with side length 8 cm
    and the area of a rectangle with sides 10 cm and 5 cm is 14 cm². -/
theorem blackboard_area_difference : 
  (8 : ℝ) * 8 - (10 : ℝ) * 5 = 14 := by
  sorry

end NUMINAMATH_CALUDE_blackboard_area_difference_l2701_270127


namespace NUMINAMATH_CALUDE_prism_diagonal_angle_l2701_270183

/-- Given a right prism with a right triangular base, where one acute angle of the base is α
    and the largest lateral face is a square, this theorem states that the angle β between
    the intersecting diagonals of the other two lateral faces is arccos(2 / √(8 + sin²(2α))) -/
theorem prism_diagonal_angle (α : ℝ) (h1 : 0 < α) (h2 : α < π / 2) :
  ∃ (β : ℝ),
    β = Real.arccos (2 / Real.sqrt (8 + Real.sin (2 * α) ^ 2)) ∧
    0 ≤ β ∧
    β ≤ π :=
sorry

end NUMINAMATH_CALUDE_prism_diagonal_angle_l2701_270183


namespace NUMINAMATH_CALUDE_vacant_seats_l2701_270182

theorem vacant_seats (total_seats : ℕ) (filled_percentage : ℚ) 
  (h1 : total_seats = 600) 
  (h2 : filled_percentage = 75 / 100) : 
  (1 - filled_percentage) * total_seats = 150 := by
sorry


end NUMINAMATH_CALUDE_vacant_seats_l2701_270182


namespace NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2701_270119

/-- Given a parabola y = -(x+2)^2 - 3, its axis of symmetry is the line x = -2 -/
theorem parabola_axis_of_symmetry :
  let f : ℝ → ℝ := λ x => -(x + 2)^2 - 3
  ∃! a : ℝ, ∀ x y : ℝ, f x = f y → (x + y) / 2 = a :=
by sorry

end NUMINAMATH_CALUDE_parabola_axis_of_symmetry_l2701_270119


namespace NUMINAMATH_CALUDE_major_premise_wrong_l2701_270177

theorem major_premise_wrong : ¬ ∀ a b : ℝ, a > b → a^2 > b^2 := by
  sorry

end NUMINAMATH_CALUDE_major_premise_wrong_l2701_270177


namespace NUMINAMATH_CALUDE_lunch_cakes_count_cakes_sum_equals_total_l2701_270106

/-- The number of cakes served during lunch today -/
def lunch_cakes : ℕ := sorry

/-- The number of cakes served during dinner today -/
def dinner_cakes : ℕ := 6

/-- The number of cakes served yesterday -/
def yesterday_cakes : ℕ := 3

/-- The total number of cakes served -/
def total_cakes : ℕ := 14

/-- Theorem stating that the number of cakes served during lunch today is 5 -/
theorem lunch_cakes_count : lunch_cakes = 5 := by
  sorry

/-- Theorem proving that the sum of cakes served equals the total -/
theorem cakes_sum_equals_total : lunch_cakes + dinner_cakes + yesterday_cakes = total_cakes := by
  sorry

end NUMINAMATH_CALUDE_lunch_cakes_count_cakes_sum_equals_total_l2701_270106
