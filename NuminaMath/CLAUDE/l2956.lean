import Mathlib

namespace NUMINAMATH_CALUDE_expression_equality_l2956_295610

theorem expression_equality : 2 * Real.sin (π / 3) + Real.sqrt 12 + abs (-5) - (π - Real.sqrt 2) ^ 0 = 3 * Real.sqrt 3 + 4 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2956_295610


namespace NUMINAMATH_CALUDE_average_weight_increase_l2956_295644

/-- Proves that the average weight increase is 200 grams when a 45 kg student leaves a group of 60 students, resulting in a new average of 57 kg for the remaining 59 students. -/
theorem average_weight_increase (initial_count : ℕ) (left_weight : ℝ) (remaining_count : ℕ) (new_average : ℝ) : 
  initial_count = 60 → 
  left_weight = 45 → 
  remaining_count = 59 → 
  new_average = 57 → 
  (new_average - (initial_count * new_average - left_weight) / initial_count) * 1000 = 200 := by
  sorry

#check average_weight_increase

end NUMINAMATH_CALUDE_average_weight_increase_l2956_295644


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l2956_295632

theorem sin_2alpha_value (α : Real) (h : Real.sin α + Real.cos (π - α) = 1/3) :
  Real.sin (2 * α) = 8/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l2956_295632


namespace NUMINAMATH_CALUDE_sum_of_m_values_is_correct_l2956_295670

/-- The sum of all possible values of m for which the polynomials x^2 - 6x + 8 and x^2 - 7x + m have a root in common -/
def sum_of_m_values : ℝ := 22

/-- First polynomial: x^2 - 6x + 8 -/
def p1 (x : ℝ) : ℝ := x^2 - 6*x + 8

/-- Second polynomial: x^2 - 7x + m -/
def p2 (x m : ℝ) : ℝ := x^2 - 7*x + m

/-- Theorem stating that the sum of all possible values of m for which p1 and p2 have a common root is equal to sum_of_m_values -/
theorem sum_of_m_values_is_correct : 
  (∃ m1 m2 : ℝ, m1 ≠ m2 ∧ 
    (∃ x1 : ℝ, p1 x1 = 0 ∧ p2 x1 m1 = 0) ∧
    (∃ x2 : ℝ, p1 x2 = 0 ∧ p2 x2 m2 = 0) ∧
    m1 + m2 = sum_of_m_values) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_m_values_is_correct_l2956_295670


namespace NUMINAMATH_CALUDE_problem_statement_l2956_295683

theorem problem_statement (x y : ℝ) (hx : x = 20) (hy : y = 8) :
  (x - y) * (x + y) = 336 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2956_295683


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2956_295694

def U : Set Nat := {2, 4, 5, 7, 8}
def A : Set Nat := {4, 8}

theorem complement_of_A_in_U :
  (U \ A) = {2, 5, 7} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2956_295694


namespace NUMINAMATH_CALUDE_circle_center_sum_l2956_295671

/-- Given a circle with equation x^2 + y^2 = 4x - 6y + 9, 
    the sum of the coordinates of its center is -1. -/
theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → (∃ h k : ℝ, (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 9) ∧ h + k = -1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2956_295671


namespace NUMINAMATH_CALUDE_quadratic_intersection_l2956_295672

def quadratic (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

theorem quadratic_intersection (a b c d : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic a b x₁ = quadratic c d x₁ ∧ 
                quadratic a b x₂ = quadratic c d x₂ ∧ 
                x₁ ≠ x₂) →
  (∀ x : ℝ, quadratic a b (-a/2) ≤ quadratic a b x) →
  (∀ x : ℝ, quadratic c d (-c/2) ≤ quadratic c d x) →
  quadratic a b (-a/2) = -200 →
  quadratic c d (-c/2) = -200 →
  (∃ x : ℝ, quadratic c d x = 0 ∧ (-a/2)^2 = x) →
  (∃ x : ℝ, quadratic a b x = 0 ∧ (-c/2)^2 = x) →
  quadratic a b 150 = -200 →
  quadratic c d 150 = -200 →
  a + c = 300 - 4 * Real.sqrt 350 ∨ a + c = 300 + 4 * Real.sqrt 350 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_l2956_295672


namespace NUMINAMATH_CALUDE_min_m_value_l2956_295613

theorem min_m_value (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (hab : a > b) (hbc : b > c) (hcd : c > d) :
  ∃ (m : ℝ), m = 9 ∧ 
  (∀ (k : ℝ), k < 9 → 
    ∃ (x y z w : ℝ), x > y ∧ y > z ∧ z > w ∧ w > 0 ∧
      Real.log (2004 : ℝ) / Real.log (y / x) + 
      Real.log (2004 : ℝ) / Real.log (z / y) + 
      Real.log (2004 : ℝ) / Real.log (w / z) < 
      k * (Real.log (2004 : ℝ) / Real.log (w / x))) ∧
  (∀ (a' b' c' d' : ℝ), a' > b' ∧ b' > c' ∧ c' > d' ∧ d' > 0 →
    Real.log (2004 : ℝ) / Real.log (b' / a') + 
    Real.log (2004 : ℝ) / Real.log (c' / b') + 
    Real.log (2004 : ℝ) / Real.log (d' / c') ≥ 
    9 * (Real.log (2004 : ℝ) / Real.log (d' / a'))) := by
  sorry

end NUMINAMATH_CALUDE_min_m_value_l2956_295613


namespace NUMINAMATH_CALUDE_function_satisfies_condition_l2956_295651

/-- The function f(x) = 1/x - x satisfies the given condition for all x₁, x₂ in (0, +∞) where x₁ ≠ x₂ -/
theorem function_satisfies_condition :
  ∀ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ →
  (x₁ - x₂) * ((1 / x₁ - x₁) - (1 / x₂ - x₂)) < 0 := by
  sorry


end NUMINAMATH_CALUDE_function_satisfies_condition_l2956_295651


namespace NUMINAMATH_CALUDE_quadratic_polynomial_symmetry_l2956_295665

theorem quadratic_polynomial_symmetry (P : ℝ → ℝ) (h : ∃ a b c : ℝ, P x = a * x^2 + b * x + c) :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    P (b + c) = P a ∧ P (c + a) = P b ∧ P (a + b) = P c :=
by sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_symmetry_l2956_295665


namespace NUMINAMATH_CALUDE_marble_boxes_theorem_l2956_295609

/-- Given a number of marbles per box and a total number of marbles,
    calculate the number of boxes. -/
def number_of_boxes (marbles_per_box : ℕ) (total_marbles : ℕ) : ℕ :=
  total_marbles / marbles_per_box

/-- Theorem stating that with 6 marbles per box and 18 total marbles,
    the number of boxes is 3. -/
theorem marble_boxes_theorem :
  number_of_boxes 6 18 = 3 := by
  sorry

end NUMINAMATH_CALUDE_marble_boxes_theorem_l2956_295609


namespace NUMINAMATH_CALUDE_inequality_proof_l2956_295653

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) : a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2956_295653


namespace NUMINAMATH_CALUDE_daily_sales_extrema_l2956_295692

-- Define the sales volume function
def g (t : ℝ) : ℝ := 80 - 2 * t

-- Define the price function
def f (t : ℝ) : ℝ := 20 - abs (t - 10)

-- Define the daily sales function
def y (t : ℝ) : ℝ := g t * f t

-- Theorem statement
theorem daily_sales_extrema :
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 20 → y t ≤ 1200) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 20 ∧ y t = 1200) ∧
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 20 → y t ≥ 400) ∧
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 20 ∧ y t = 400) :=
by sorry

end NUMINAMATH_CALUDE_daily_sales_extrema_l2956_295692


namespace NUMINAMATH_CALUDE_vault_code_thickness_l2956_295625

/-- Thickness of an Alpha card in millimeters -/
def alpha_thickness : ℚ := 1.65

/-- Thickness of a Beta card in millimeters -/
def beta_thickness : ℚ := 2.05

/-- Thickness of a Gamma card in millimeters -/
def gamma_thickness : ℚ := 1.25

/-- Thickness of a Delta card in millimeters -/
def delta_thickness : ℚ := 1.85

/-- Total thickness of the stack in millimeters -/
def total_thickness : ℚ := 15.6

/-- The number of cards in the stack -/
def num_cards : ℕ := 8

theorem vault_code_thickness :
  num_cards * delta_thickness = total_thickness ∧
  ∀ (a b c d : ℕ), 
    a * alpha_thickness + b * beta_thickness + c * gamma_thickness + d * delta_thickness = total_thickness →
    a = 0 ∧ b = 0 ∧ c = 0 ∧ d = num_cards :=
by sorry

end NUMINAMATH_CALUDE_vault_code_thickness_l2956_295625


namespace NUMINAMATH_CALUDE_football_cost_l2956_295675

/-- The cost of a football given the total amount paid, change received, and cost of a baseball. -/
theorem football_cost (total_paid : ℝ) (change : ℝ) (baseball_cost : ℝ) 
  (h1 : total_paid = 20)
  (h2 : change = 4.05)
  (h3 : baseball_cost = 6.81) : 
  total_paid - change - baseball_cost = 9.14 := by
  sorry

#check football_cost

end NUMINAMATH_CALUDE_football_cost_l2956_295675


namespace NUMINAMATH_CALUDE_bus_journey_distance_l2956_295699

/-- Given a bus journey with two different speeds, prove the distance covered at the lower speed. -/
theorem bus_journey_distance (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 5)
  (h5 : total_time = (distance1 / speed1) + ((total_distance - distance1) / speed2)) :
  distance1 = 100 := by sorry


end NUMINAMATH_CALUDE_bus_journey_distance_l2956_295699


namespace NUMINAMATH_CALUDE_min_value_quadratic_with_linear_constraint_l2956_295615

theorem min_value_quadratic_with_linear_constraint :
  ∃ (min_u : ℝ), min_u = -66/13 ∧
  ∀ (x y : ℝ), 3*x + 2*y - 1 ≥ 0 →
    x^2 + y^2 + 6*x - 2*y ≥ min_u :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_with_linear_constraint_l2956_295615


namespace NUMINAMATH_CALUDE_basketball_game_scores_l2956_295649

/-- Represents the scores of a team in a basketball game -/
structure TeamScores :=
  (q1 q2 q3 q4 : ℕ)

/-- Checks if a sequence of four numbers is an arithmetic progression -/
def isArithmeticSequence (s : TeamScores) : Prop :=
  s.q2 - s.q1 = s.q3 - s.q2 ∧ s.q3 - s.q2 = s.q4 - s.q3 ∧ s.q2 > s.q1

/-- Checks if a sequence of four numbers is a geometric progression -/
def isGeometricSequence (s : TeamScores) : Prop :=
  ∃ r : ℚ, r > 1 ∧ s.q2 = s.q1 * r ∧ s.q3 = s.q2 * r ∧ s.q4 = s.q3 * r

/-- The main theorem statement -/
theorem basketball_game_scores 
  (falcons tigers : TeamScores) 
  (h1 : falcons.q1 = tigers.q1)
  (h2 : isArithmeticSequence falcons)
  (h3 : isGeometricSequence tigers)
  (h4 : falcons.q1 + falcons.q2 + falcons.q3 + falcons.q4 = 
        tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 + 2)
  (h5 : falcons.q1 + falcons.q2 + falcons.q3 + falcons.q4 ≤ 100)
  (h6 : tigers.q1 + tigers.q2 + tigers.q3 + tigers.q4 ≤ 100) :
  falcons.q1 + falcons.q2 + tigers.q1 + tigers.q2 = 14 :=
by
  sorry


end NUMINAMATH_CALUDE_basketball_game_scores_l2956_295649


namespace NUMINAMATH_CALUDE_three_odd_factors_is_nine_l2956_295607

theorem three_odd_factors_is_nine :
  ∃! n : ℕ, n > 1 ∧ (∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧
    Odd a ∧ Odd b ∧ Odd c ∧
    {d : ℕ | d > 1 ∧ d ∣ n ∧ Odd d} = {a, b, c}) :=
by
  sorry

end NUMINAMATH_CALUDE_three_odd_factors_is_nine_l2956_295607


namespace NUMINAMATH_CALUDE_binomial_10_5_l2956_295622

theorem binomial_10_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_5_l2956_295622


namespace NUMINAMATH_CALUDE_total_area_parallelogram_triangle_l2956_295661

/-- The total area of a shape consisting of a parallelogram and an adjacent right triangle -/
theorem total_area_parallelogram_triangle (angle : Real) (side1 side2 : Real) (leg : Real) : 
  angle = 150 * π / 180 →
  side1 = 10 →
  side2 = 24 →
  leg = 10 →
  (side1 * side2 * Real.sin angle) / 2 + (side2 * leg) / 2 = 170 := by sorry

end NUMINAMATH_CALUDE_total_area_parallelogram_triangle_l2956_295661


namespace NUMINAMATH_CALUDE_geometric_sum_2_power_63_l2956_295662

theorem geometric_sum_2_power_63 : 
  (Finset.range 64).sum (fun i => 2^i) = 2^64 - 1 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sum_2_power_63_l2956_295662


namespace NUMINAMATH_CALUDE_smaller_solution_quadratic_l2956_295666

theorem smaller_solution_quadratic : ∃ (x y : ℝ), 
  x < y ∧ 
  x^2 - 12*x - 28 = 0 ∧ 
  y^2 - 12*y - 28 = 0 ∧
  x = -2 ∧
  ∀ z : ℝ, z^2 - 12*z - 28 = 0 → z = x ∨ z = y := by
  sorry

end NUMINAMATH_CALUDE_smaller_solution_quadratic_l2956_295666


namespace NUMINAMATH_CALUDE_shop_profit_days_l2956_295621

theorem shop_profit_days (mean_profit : ℝ) (first_15_mean : ℝ) (last_15_mean : ℝ)
  (h1 : mean_profit = 350)
  (h2 : first_15_mean = 255)
  (h3 : last_15_mean = 445) :
  ∃ (total_days : ℕ), 
    total_days = 30 ∧ 
    (first_15_mean * 15 + last_15_mean * 15 : ℝ) = mean_profit * total_days :=
by
  sorry

end NUMINAMATH_CALUDE_shop_profit_days_l2956_295621


namespace NUMINAMATH_CALUDE_jiuquan_location_accuracy_l2956_295690

-- Define the possible location descriptions
inductive LocationDescription
  | NorthwestOfBeijing
  | LatitudeOnly (lat : Float)
  | LongitudeOnly (long : Float)
  | LatitudeLongitude (lat : Float) (long : Float)

-- Define the accuracy of a location description
def isAccurateLocation (desc : LocationDescription) : Prop :=
  match desc with
  | LocationDescription.LatitudeLongitude _ _ => True
  | _ => False

-- Theorem statement
theorem jiuquan_location_accuracy :
  ∀ (desc : LocationDescription),
    isAccurateLocation desc ↔
      desc = LocationDescription.LatitudeLongitude 39.75 98.52 :=
by sorry

end NUMINAMATH_CALUDE_jiuquan_location_accuracy_l2956_295690


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l2956_295626

/-- Calculates the net rate of pay for a driver given travel conditions and expenses. -/
theorem driver_net_pay_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (fuel_efficiency : ℝ)
  (earnings_rate : ℝ)
  (gasoline_cost : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 60)
  (h3 : fuel_efficiency = 30)
  (h4 : earnings_rate = 0.75)
  (h5 : gasoline_cost = 3) :
  let distance := travel_time * speed
  let fuel_used := distance / fuel_efficiency
  let earnings := distance * earnings_rate
  let fuel_expense := fuel_used * gasoline_cost
  let net_earnings := earnings - fuel_expense
  net_earnings / travel_time = 39 := by
sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l2956_295626


namespace NUMINAMATH_CALUDE_work_completion_theorem_l2956_295641

theorem work_completion_theorem (work : ℕ) (days1 days2 men1 : ℕ) 
  (h1 : work = men1 * days1)
  (h2 : work = 24 * (work / (men1 * days1) * men1))
  (h3 : men1 = 16)
  (h4 : days1 = 30)
  : work / (men1 * days1) * men1 = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l2956_295641


namespace NUMINAMATH_CALUDE_expression_evaluation_l2956_295697

theorem expression_evaluation (x : ℝ) (h1 : x^6 ≠ -1) (h2 : x^6 ≠ 1) :
  ((((x^2 + 1)^2 * (x^4 - x^2 + 1)^2) / (x^6 + 1)^2)^2 *
   (((x^2 - 1)^2 * (x^4 + x^2 + 1)^2) / (x^6 - 1)^2)^2) = 1 := by
  sorry


end NUMINAMATH_CALUDE_expression_evaluation_l2956_295697


namespace NUMINAMATH_CALUDE_geometry_relations_l2956_295681

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (line_perpendicular : Line → Line → Prop)
variable (line_parallel : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem geometry_relations 
  (m l : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : contained_in l β) :
  (((parallel α β) → (line_perpendicular m l)) ∧
   ((line_parallel m l) → (plane_perpendicular α β))) ∧
  ¬(((plane_perpendicular α β) → (line_parallel m l)) ∧
    ((line_perpendicular m l) → (parallel α β))) :=
sorry

end NUMINAMATH_CALUDE_geometry_relations_l2956_295681


namespace NUMINAMATH_CALUDE_unique_solution_l2956_295635

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  x = 2 * (2 * (2 * (2 * (2 * x - 1) - 1) - 1) - 1) - 1

/-- Theorem stating that 1 is the unique solution to the equation -/
theorem unique_solution :
  ∃! x : ℝ, equation x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l2956_295635


namespace NUMINAMATH_CALUDE_custom_op_example_l2956_295693

/-- Custom operation $\$$ defined for two integers -/
def custom_op (a b : Int) : Int := a * (b - 1) + a * b

/-- Theorem stating that 5 $\$$ (-3) = -35 -/
theorem custom_op_example : custom_op 5 (-3) = -35 := by
  sorry

end NUMINAMATH_CALUDE_custom_op_example_l2956_295693


namespace NUMINAMATH_CALUDE_p_and_q_true_l2956_295682

theorem p_and_q_true :
  (∃ x₀ : ℝ, x₀^2 < x₀) ∧ (∀ x : ℝ, x^2 - x + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_p_and_q_true_l2956_295682


namespace NUMINAMATH_CALUDE_function_equality_l2956_295629

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 7
def g (k : ℝ) (x : ℝ) : ℝ := x^2 - k * x + 5

-- State the theorem
theorem function_equality (k : ℝ) : f 5 - g k 5 = 0 → k = -92 / 5 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2956_295629


namespace NUMINAMATH_CALUDE_tank_filling_time_l2956_295642

theorem tank_filling_time (a b c : ℝ) (h1 : c = 2 * b) (h2 : b = 2 * a) (h3 : a + b + c = 1 / 8) :
  1 / a = 56 := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_time_l2956_295642


namespace NUMINAMATH_CALUDE_prob_select_two_after_transfer_l2956_295668

/-- Represents the label on a ball -/
inductive Label
  | one
  | two
  | three

/-- Represents a bag of balls -/
structure Bag where
  ones : Nat
  twos : Nat
  threes : Nat

/-- Initial state of bag A -/
def bagA : Bag := ⟨3, 2, 1⟩

/-- Initial state of bag B -/
def bagB : Bag := ⟨2, 1, 1⟩

/-- Probability of selecting a ball with a specific label from a bag -/
def probSelect (bag : Bag) (label : Label) : Rat :=
  match label with
  | Label.one => bag.ones / (bag.ones + bag.twos + bag.threes)
  | Label.two => bag.twos / (bag.ones + bag.twos + bag.threes)
  | Label.three => bag.threes / (bag.ones + bag.twos + bag.threes)

/-- Probability of selecting a ball labeled 2 from bag B after transfer -/
def probSelectTwoAfterTransfer : Rat :=
  (probSelect bagA Label.one) * (probSelect ⟨bagB.ones + 1, bagB.twos, bagB.threes⟩ Label.two) +
  (probSelect bagA Label.two) * (probSelect ⟨bagB.ones, bagB.twos + 1, bagB.threes⟩ Label.two) +
  (probSelect bagA Label.three) * (probSelect ⟨bagB.ones, bagB.twos, bagB.threes + 1⟩ Label.two)

theorem prob_select_two_after_transfer :
  probSelectTwoAfterTransfer = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_prob_select_two_after_transfer_l2956_295668


namespace NUMINAMATH_CALUDE_system_solution_l2956_295695

theorem system_solution (x y : ℝ) (eq1 : x + 2*y = 8) (eq2 : 2*x + y = 1) : x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2956_295695


namespace NUMINAMATH_CALUDE_typing_orders_count_l2956_295631

/-- Represents the order of letters delivered by the boss -/
def letterOrder : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9]

/-- Represents that letter 8 has been typed -/
def letter8Typed : Nat := 8

/-- The number of letters that can be either typed or not typed after lunch -/
def remainingLetters : Nat := 8

/-- Theorem: The number of possible after-lunch typing orders is 2^8 = 256 -/
theorem typing_orders_count : 
  (2 : Nat) ^ remainingLetters = 256 := by
  sorry

#check typing_orders_count

end NUMINAMATH_CALUDE_typing_orders_count_l2956_295631


namespace NUMINAMATH_CALUDE_binomial_18_10_l2956_295654

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 8008) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 43758 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l2956_295654


namespace NUMINAMATH_CALUDE_point_coordinates_l2956_295606

/-- A point in the coordinate plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def isInSecondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distanceToXAxis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distanceToYAxis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates 
    (p : Point) 
    (h1 : isInSecondQuadrant p) 
    (h2 : distanceToXAxis p = 4) 
    (h3 : distanceToYAxis p = 5) : 
  p = Point.mk (-5) 4 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l2956_295606


namespace NUMINAMATH_CALUDE_distance_difference_l2956_295617

/-- The rate at which Bjorn bikes in miles per hour -/
def bjorn_rate : ℝ := 12

/-- The rate at which Alberto bikes in miles per hour -/
def alberto_rate : ℝ := 15

/-- The duration of the biking trip in hours -/
def trip_duration : ℝ := 6

/-- The theorem stating the difference in distance traveled between Alberto and Bjorn -/
theorem distance_difference : 
  alberto_rate * trip_duration - bjorn_rate * trip_duration = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_l2956_295617


namespace NUMINAMATH_CALUDE_rod_division_theorem_l2956_295619

/-- Represents a rod divided into equal parts -/
structure DividedRod where
  length : ℕ
  divisions : List ℕ

/-- Calculates the total number of segments in a divided rod -/
def totalSegments (rod : DividedRod) : ℕ := sorry

/-- Calculates the length of the shortest segment in a divided rod -/
def shortestSegment (rod : DividedRod) : ℚ := sorry

/-- Theorem about a specific rod division -/
theorem rod_division_theorem (k : ℕ) :
  let rod := DividedRod.mk (72 * k) [8, 12, 18]
  totalSegments rod = 28 ∧ shortestSegment rod = 1 / 72 := by sorry

end NUMINAMATH_CALUDE_rod_division_theorem_l2956_295619


namespace NUMINAMATH_CALUDE_maximize_sqrt_expression_l2956_295684

theorem maximize_sqrt_expression :
  let add := Real.sqrt 8 + Real.sqrt 2
  let mul := Real.sqrt 8 * Real.sqrt 2
  let div := Real.sqrt 8 / Real.sqrt 2
  let sub := Real.sqrt 8 - Real.sqrt 2
  add > mul ∧ add > div ∧ add > sub := by
  sorry

end NUMINAMATH_CALUDE_maximize_sqrt_expression_l2956_295684


namespace NUMINAMATH_CALUDE_opposite_of_three_l2956_295623

theorem opposite_of_three : ∃ x : ℤ, x + 3 = 0 ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_three_l2956_295623


namespace NUMINAMATH_CALUDE_largest_a_value_l2956_295614

theorem largest_a_value (a : ℝ) :
  (5 * Real.sqrt ((3 * a)^2 + 2^2) - 5 * a^2 - 2) / (Real.sqrt (2 + 3 * a^2) + 2) = 1 →
  ∃ y : ℝ, y^2 - (5 * Real.sqrt 3 - 1) * y + 5 = 0 ∧
           y ≥ (5 * Real.sqrt 3 - 1 + Real.sqrt ((5 * Real.sqrt 3 - 1)^2 - 20)) / 2 ∧
           a = Real.sqrt ((y^2 - 2) / 3) ∧
           ∀ a' : ℝ, (5 * Real.sqrt ((3 * a')^2 + 2^2) - 5 * a'^2 - 2) / (Real.sqrt (2 + 3 * a'^2) + 2) = 1 →
                     a' ≤ a :=
by sorry

end NUMINAMATH_CALUDE_largest_a_value_l2956_295614


namespace NUMINAMATH_CALUDE_volume_of_rotated_solid_l2956_295686

/-- The equation of the region -/
def region_equation (x y : ℝ) : Prop := |x/3| + |y/3| = 2

/-- The region enclosed by the equation -/
def enclosed_region : Set (ℝ × ℝ) := {p : ℝ × ℝ | region_equation p.1 p.2}

/-- The volume of the solid generated by rotating the region around the x-axis -/
noncomputable def rotation_volume : ℝ := sorry

/-- Theorem stating that the volume of the rotated solid is equal to some value V -/
theorem volume_of_rotated_solid :
  ∃ V, rotation_volume = V :=
sorry

end NUMINAMATH_CALUDE_volume_of_rotated_solid_l2956_295686


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2956_295667

theorem inscribed_squares_ratio (a b c x y : ℝ) : 
  a = 5 → b = 12 → c = 13 →
  a^2 + b^2 = c^2 →
  x * (a + b - x) = a * b →
  y * (c - y) = (a - y) * (b - y) →
  x / y = 5 / 13 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2956_295667


namespace NUMINAMATH_CALUDE_fraction_sign_l2956_295687

theorem fraction_sign (a b : ℝ) (ha : a > 0) (hb : b < 0) : a / b < 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sign_l2956_295687


namespace NUMINAMATH_CALUDE_square_and_circle_measurements_l2956_295659

/-- Given a square with side length 70√2 cm and a circle with diameter equal to the square's diagonal,
    prove the square's diagonal length and the circle's circumference. -/
theorem square_and_circle_measurements :
  let square_side : ℝ := 70 * Real.sqrt 2
  let square_diagonal : ℝ := square_side * Real.sqrt 2
  let circle_diameter : ℝ := square_diagonal
  let circle_circumference : ℝ := π * circle_diameter
  (square_diagonal = 140) ∧ (circle_circumference = 140 * π) := by sorry

end NUMINAMATH_CALUDE_square_and_circle_measurements_l2956_295659


namespace NUMINAMATH_CALUDE_student_count_l2956_295652

/-- The number of students in Elementary and Middle School -/
def total_students (elementary : ℕ) (middle : ℕ) : ℕ :=
  elementary + middle

/-- Theorem stating the total number of students given the conditions -/
theorem student_count : ∃ (elementary : ℕ) (middle : ℕ),
  middle = 50 ∧ 
  elementary = 4 * middle - 3 ∧
  total_students elementary middle = 247 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2956_295652


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l2956_295688

def is_divisible_by_2 (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def is_divisible_by_6 (n : ℕ) : Prop := is_divisible_by_2 n ∧ is_divisible_by_3 n

def sum_of_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.sum

def is_four_digit (n : ℕ) : Prop := n ≥ 1000 ∧ n ≤ 9999

theorem largest_four_digit_divisible_by_6 :
  ∀ n : ℕ, is_four_digit n → is_divisible_by_6 n → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_6_l2956_295688


namespace NUMINAMATH_CALUDE_investment_scientific_notation_l2956_295656

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  h1 : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem investment_scientific_notation :
  toScientificNotation 909000000000 = ScientificNotation.mk 9.09 11 sorry := by
  sorry

end NUMINAMATH_CALUDE_investment_scientific_notation_l2956_295656


namespace NUMINAMATH_CALUDE_quadratic_root_range_l2956_295638

-- Define the quadratic function
def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - 1

theorem quadratic_root_range (a b : ℝ) :
  a > 0 →
  (∃ x y : ℝ, x ≠ y ∧ f a b x = 0 ∧ f a b y = 0) →
  (∃ z : ℝ, 1 < z ∧ z < 2 ∧ f a b z = 0) →
  ∀ k : ℝ, -1 < k ∧ k < 1 ↔ ∃ a b : ℝ, a - b = k :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_range_l2956_295638


namespace NUMINAMATH_CALUDE_range_of_f_l2956_295685

-- Define the function f
def f (x : ℝ) : ℝ := |x + 4| - |x - 5|

-- State the theorem about the range of f
theorem range_of_f : Set.range f = Set.Icc (-9 : ℝ) 9 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2956_295685


namespace NUMINAMATH_CALUDE_derivative_exp_cos_l2956_295658

open Real

theorem derivative_exp_cos (x : ℝ) : 
  deriv (λ x => exp x * cos x) x = exp x * (cos x - sin x) := by
sorry

end NUMINAMATH_CALUDE_derivative_exp_cos_l2956_295658


namespace NUMINAMATH_CALUDE_same_terminal_side_as_pi_sixth_l2956_295657

def coterminal (θ₁ θ₂ : Real) : Prop :=
  ∃ k : Int, θ₁ = θ₂ + 2 * k * Real.pi

theorem same_terminal_side_as_pi_sixth (θ : Real) : 
  coterminal θ (π/6) ↔ ∃ k : Int, θ = π/6 + 2 * k * π :=
sorry

end NUMINAMATH_CALUDE_same_terminal_side_as_pi_sixth_l2956_295657


namespace NUMINAMATH_CALUDE_lexie_age_difference_l2956_295676

/-- Given information about Lexie, her sister, and her brother's ages, prove the age difference between Lexie and her brother. -/
theorem lexie_age_difference (lexie_age : ℕ) (sister_age : ℕ) (brother_age : ℕ)
  (h1 : lexie_age = 8)
  (h2 : sister_age = 2 * lexie_age)
  (h3 : sister_age - brother_age = 14) :
  lexie_age - brother_age = 6 := by
  sorry

end NUMINAMATH_CALUDE_lexie_age_difference_l2956_295676


namespace NUMINAMATH_CALUDE_no_solution_l2956_295602

/-- Q(n) denotes the greatest prime factor of n -/
def Q (n : ℕ) : ℕ := sorry

/-- The theorem states that there are no positive integers n > 1 satisfying
    both Q(n) = √n and Q(3n + 16) = √(3n + 16) -/
theorem no_solution :
  ¬ ∃ (n : ℕ), n > 1 ∧ 
    Q n = Nat.sqrt n ∧ 
    Q (3 * n + 16) = Nat.sqrt (3 * n + 16) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l2956_295602


namespace NUMINAMATH_CALUDE_prime_divisibility_l2956_295630

theorem prime_divisibility (p q r : ℕ) : 
  Prime p → Prime q → Prime r → p ≠ q → p ≠ r → q ≠ r →
  (pqr : ℕ) = p * q * r →
  (pqr ∣ (p * q)^r + (q * r)^p + (r * p)^q - 1) →
  ((pqr)^3 ∣ 3 * ((p * q)^r + (q * r)^p + (r * p)^q - 1)) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2956_295630


namespace NUMINAMATH_CALUDE_equation_solution_l2956_295612

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), (x₁ = 4 + 3 * Real.sqrt 19) ∧ (x₂ = 4 - 3 * Real.sqrt 19) ∧
  (∀ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔ 
  (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2956_295612


namespace NUMINAMATH_CALUDE_perfect_square_fraction_l2956_295673

theorem perfect_square_fraction (m n : ℕ+) : 
  ∃ k : ℕ, (m + n : ℝ)^2 / (4 * (m : ℝ) * (m - n : ℝ)^2 + 4) = (k : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_fraction_l2956_295673


namespace NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l2956_295664

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem sixth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h_q : q = 2)
  (h_a2 : a 2 = 8) :
  a 6 = 128 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_geometric_sequence_l2956_295664


namespace NUMINAMATH_CALUDE_sphere_radii_ratio_l2956_295604

theorem sphere_radii_ratio (V1 V2 r1 r2 : ℝ) :
  V1 = 450 * Real.pi →
  V2 = 36 * Real.pi →
  V2 / V1 = (r2 / r1) ^ 3 →
  r2 / r1 = Real.rpow 2 (1/3) / 5 := by
sorry

end NUMINAMATH_CALUDE_sphere_radii_ratio_l2956_295604


namespace NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l2956_295678

theorem max_value_of_sum_of_square_roots (a b c : ℝ) 
  (nonneg_a : a ≥ 0) (nonneg_b : b ≥ 0) (nonneg_c : c ≥ 0) 
  (sum_constraint : a + b + c = 8) : 
  Real.sqrt (3 * a + 2) + Real.sqrt (3 * b + 2) + Real.sqrt (3 * c + 2) ≤ 3 * Real.sqrt 26 ∧ 
  ∃ a' b' c' : ℝ, a' ≥ 0 ∧ b' ≥ 0 ∧ c' ≥ 0 ∧ a' + b' + c' = 8 ∧
  Real.sqrt (3 * a' + 2) + Real.sqrt (3 * b' + 2) + Real.sqrt (3 * c' + 2) = 3 * Real.sqrt 26 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_of_square_roots_l2956_295678


namespace NUMINAMATH_CALUDE_divisor_problem_l2956_295601

theorem divisor_problem (n m : ℕ) (h1 : n = 987654) (h2 : m = 42) : 
  (n + m) % m = 0 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l2956_295601


namespace NUMINAMATH_CALUDE_complete_square_constant_l2956_295636

theorem complete_square_constant (a h k : ℚ) : 
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) → k = -49/4 := by
  sorry

end NUMINAMATH_CALUDE_complete_square_constant_l2956_295636


namespace NUMINAMATH_CALUDE_final_number_is_one_l2956_295634

def initialSum : ℕ := (1988 * 1989) / 2

def operationA (numbers : List ℕ) (d : ℕ) : List ℕ :=
  numbers.map (λ x => if x ≥ d then x - d else 0)

def operationB (numbers : List ℕ) : List ℕ :=
  match numbers with
  | x :: y :: rest => (x + y) :: rest
  | _ => numbers

def performOperations (numbers : List ℕ) (iterations : ℕ) : ℕ :=
  if iterations = 0 then
    match numbers with
    | [x] => x
    | _ => 0
  else
    let numbersAfterA := operationA numbers 1
    let numbersAfterB := operationB numbersAfterA
    performOperations numbersAfterB (iterations - 1)

theorem final_number_is_one :
  performOperations (List.range 1989) 1987 = 1 :=
sorry

end NUMINAMATH_CALUDE_final_number_is_one_l2956_295634


namespace NUMINAMATH_CALUDE_ellipse_and_line_property_l2956_295600

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a line -/
structure Line where
  k : ℝ
  b : ℝ
  h : k ≠ 0 ∧ b ≠ 0

/-- Given conditions of the problem -/
axiom ellipse_condition (C : Ellipse) : 
  C.a^2 - C.b^2 = 4 ∧ 2/C.a^2 + 3/C.b^2 = 1

/-- The theorem to be proved -/
theorem ellipse_and_line_property (C : Ellipse) (l : Line) :
  (∀ x y, x^2/C.a^2 + y^2/C.b^2 = 1 ↔ x^2/8 + y^2/4 = 1) ∧
  (∃ x₁ y₁ x₂ y₂, 
    x₁^2/8 + y₁^2/4 = 1 ∧
    x₂^2/8 + y₂^2/4 = 1 ∧
    y₁ = l.k * x₁ + l.b ∧
    y₂ = l.k * x₂ + l.b ∧
    x₁ ≠ x₂ ∧
    let xₘ := (x₁ + x₂)/2
    let yₘ := (y₁ + y₂)/2
    (yₘ / xₘ) * l.k = -1/2) := by sorry

end NUMINAMATH_CALUDE_ellipse_and_line_property_l2956_295600


namespace NUMINAMATH_CALUDE_milkman_profit_l2956_295624

/-- Calculates the profit of a milkman selling three mixtures of milk and water -/
theorem milkman_profit (total_milk : ℝ) (total_water : ℝ) 
  (milk1 : ℝ) (water1 : ℝ) (price1 : ℝ)
  (milk2 : ℝ) (water2 : ℝ) (price2 : ℝ)
  (water3 : ℝ) (price3 : ℝ)
  (milk_cost : ℝ) :
  total_milk = 80 ∧
  total_water = 20 ∧
  milk1 = 40 ∧
  water1 = 5 ∧
  price1 = 19 ∧
  milk2 = 25 ∧
  water2 = 10 ∧
  price2 = 18 ∧
  water3 = 5 ∧
  price3 = 21 ∧
  milk_cost = 22 →
  let milk3 := total_milk - milk1 - milk2
  let revenue1 := (milk1 + water1) * price1
  let revenue2 := (milk2 + water2) * price2
  let revenue3 := (milk3 + water3) * price3
  let total_revenue := revenue1 + revenue2 + revenue3
  let total_cost := total_milk * milk_cost
  let profit := total_revenue - total_cost
  profit = 50 := by
sorry

end NUMINAMATH_CALUDE_milkman_profit_l2956_295624


namespace NUMINAMATH_CALUDE_discount_calculation_l2956_295679

/-- The discount calculation problem --/
theorem discount_calculation (original_cost spent : ℝ) 
  (h1 : original_cost = 35)
  (h2 : spent = 18) : 
  original_cost - spent = 17 := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l2956_295679


namespace NUMINAMATH_CALUDE_lingonberry_price_theorem_l2956_295628

/-- The price per pound of lingonberries picked -/
def price_per_pound : ℚ := 2

/-- The total amount Steve wants to make -/
def total_amount : ℚ := 100

/-- The amount of lingonberries picked on Monday -/
def monday_picked : ℚ := 8

/-- The amount of lingonberries picked on Tuesday -/
def tuesday_picked : ℚ := 3 * monday_picked

/-- The amount of lingonberries picked on Wednesday -/
def wednesday_picked : ℚ := 0

/-- The amount of lingonberries picked on Thursday -/
def thursday_picked : ℚ := 18

/-- The total amount of lingonberries picked over four days -/
def total_picked : ℚ := monday_picked + tuesday_picked + wednesday_picked + thursday_picked

theorem lingonberry_price_theorem : 
  price_per_pound * total_picked = total_amount :=
by sorry

end NUMINAMATH_CALUDE_lingonberry_price_theorem_l2956_295628


namespace NUMINAMATH_CALUDE_cubic_function_increasing_iff_a_nonpositive_l2956_295647

/-- Theorem: For the function f(x) = x^3 - ax + 1 where a ∈ ℝ, 
    f(x) is increasing in its domain if and only if a ≤ 0 -/
theorem cubic_function_increasing_iff_a_nonpositive (a : ℝ) :
  (∀ x : ℝ, HasDerivAt (fun x => x^3 - a*x + 1) (3*x^2 - a) x) →
  (∀ x y : ℝ, x < y → (x^3 - a*x + 1) < (y^3 - a*y + 1)) ↔ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_increasing_iff_a_nonpositive_l2956_295647


namespace NUMINAMATH_CALUDE_rational_numbers_classification_l2956_295680

theorem rational_numbers_classification (x : ℚ) : 
  ¬(∀ x : ℚ, x > 0 ∨ x < 0) :=
by
  sorry

end NUMINAMATH_CALUDE_rational_numbers_classification_l2956_295680


namespace NUMINAMATH_CALUDE_hyperbola_focus_l2956_295603

/-- Given a hyperbola with equation ((x-5)^2)/7^2 - ((y-20)^2)/15^2 = 1,
    the focus with the larger x-coordinate has coordinates (5 + √274, 20) -/
theorem hyperbola_focus (x y : ℝ) :
  ((x - 5)^2 / 7^2) - ((y - 20)^2 / 15^2) = 1 →
  ∃ (f_x f_y : ℝ), f_x > 5 ∧ f_y = 20 ∧ f_x = 5 + Real.sqrt 274 ∧
  ∀ (x' y' : ℝ), ((x' - 5)^2 / 7^2) - ((y' - 20)^2 / 15^2) = 1 →
  (x' - 5)^2 / 7^2 + (y' - 20)^2 / 15^2 = (x' - f_x)^2 / 7^2 + (y' - f_y)^2 / 15^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l2956_295603


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2956_295650

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 8 → x ≤ y) ∧
    ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧
    x = 89/9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2956_295650


namespace NUMINAMATH_CALUDE_remainder_problem_l2956_295605

theorem remainder_problem (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2956_295605


namespace NUMINAMATH_CALUDE_no_arithmetic_progression_roots_l2956_295618

theorem no_arithmetic_progression_roots (a : ℝ) : 
  ¬ ∃ (x d : ℝ), 
    (∀ k : Fin 4, 16 * (x + k * d)^4 - a * (x + k * d)^3 + (2*a + 17) * (x + k * d)^2 - a * (x + k * d) + 16 = 0) ∧
    (d ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_arithmetic_progression_roots_l2956_295618


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2956_295620

theorem quadratic_equation_root (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 3 * x - 119 = 0) → 
  (a * 7^2 + 3 * 7 - 119 = 0) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2956_295620


namespace NUMINAMATH_CALUDE_subtraction_of_one_and_two_l2956_295611

theorem subtraction_of_one_and_two : 1 - 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_one_and_two_l2956_295611


namespace NUMINAMATH_CALUDE_promotional_price_calculation_l2956_295640

/-- The cost of one chocolate at the store with the promotion -/
def promotional_price : ℚ := 2

theorem promotional_price_calculation :
  let chocolates_per_week : ℕ := 2
  let weeks : ℕ := 3
  let local_price : ℚ := 3
  let total_savings : ℚ := 6
  promotional_price = (chocolates_per_week * weeks * local_price - total_savings) / (chocolates_per_week * weeks) :=
by sorry

end NUMINAMATH_CALUDE_promotional_price_calculation_l2956_295640


namespace NUMINAMATH_CALUDE_quadrilateral_angle_inequality_l2956_295677

variable (A B C D A₁ B₁ C₁ D₁ : Point)

-- Define the quadrilaterals
def is_convex_quadrilateral (P Q R S : Point) : Prop := sorry

-- Define the equality of corresponding sides
def equal_corresponding_sides (P Q R S P₁ Q₁ R₁ S₁ : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_angle_inequality
  (h_convex_ABCD : is_convex_quadrilateral A B C D)
  (h_convex_A₁B₁C₁D₁ : is_convex_quadrilateral A₁ B₁ C₁ D₁)
  (h_equal_sides : equal_corresponding_sides A B C D A₁ B₁ C₁ D₁)
  (h_angle_A : angle_measure B A D > angle_measure B₁ A₁ D₁) :
  angle_measure A B C < angle_measure A₁ B₁ C₁ ∧
  angle_measure B C D > angle_measure B₁ C₁ D₁ ∧
  angle_measure C D A < angle_measure C₁ D₁ A₁ :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_inequality_l2956_295677


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2956_295663

-- Problem 1
theorem problem_1 : (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) + Real.sqrt 6 / Real.sqrt 2 = 2 + Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h : x = Real.sqrt 2 - 2) : 
  ((1 / (x - 1) - 1 / (x + 1)) / ((x + 2) / (x^2 - 1))) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2956_295663


namespace NUMINAMATH_CALUDE_expression_value_l2956_295627

theorem expression_value : 
  let a := 2020
  (a^3 - 3*a^2*(a+1) + 4*a*(a+1)^2 - (a+1)^3 + 1) / (a*(a+1)) = 2021 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l2956_295627


namespace NUMINAMATH_CALUDE_total_shares_sold_l2956_295637

/-- Proves that the total number of shares sold is 300 given the specified conditions -/
theorem total_shares_sold (microtron_price dynaco_price avg_price : ℚ) (dynaco_shares : ℕ) : 
  microtron_price = 36 →
  dynaco_price = 44 →
  avg_price = 40 →
  dynaco_shares = 150 →
  ∃ (microtron_shares : ℕ), 
    (microtron_price * microtron_shares + dynaco_price * dynaco_shares) / (microtron_shares + dynaco_shares) = avg_price ∧
    microtron_shares + dynaco_shares = 300 := by
  sorry

end NUMINAMATH_CALUDE_total_shares_sold_l2956_295637


namespace NUMINAMATH_CALUDE_sqrt_15_bounds_l2956_295643

theorem sqrt_15_bounds : 3 < Real.sqrt 15 ∧ Real.sqrt 15 < 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_15_bounds_l2956_295643


namespace NUMINAMATH_CALUDE_root_sum_ratio_l2956_295669

theorem root_sum_ratio (a b c d : ℝ) (h1 : a ≠ 0) (h2 : d = 0)
  (h3 : a * (4 : ℝ)^3 + b * (4 : ℝ)^2 + c * (4 : ℝ) + d = 0)
  (h4 : a * (-3 : ℝ)^3 + b * (-3 : ℝ)^2 + c * (-3 : ℝ) + d = 0) :
  (b + c) / a = -13 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l2956_295669


namespace NUMINAMATH_CALUDE_range_of_m_l2956_295616

theorem range_of_m (m : ℝ) : 
  (∀ x, (1 ≤ x ∧ x ≤ 3) → (m + 1 ≤ x ∧ x ≤ 2*m + 7)) ∧ 
  (∃ x, m + 1 ≤ x ∧ x ≤ 2*m + 7 ∧ ¬(1 ≤ x ∧ x ≤ 3)) → 
  -2 ≤ m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2956_295616


namespace NUMINAMATH_CALUDE_unique_function_solution_l2956_295633

theorem unique_function_solution (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (1 + x * y) - f (x + y) = f x * f y) 
  (h2 : f (-1) ≠ 0) : 
  ∀ x : ℝ, f x = x - 1 := by
sorry

end NUMINAMATH_CALUDE_unique_function_solution_l2956_295633


namespace NUMINAMATH_CALUDE_purchase_system_of_equations_l2956_295698

/-- Represents the purchase of basketballs and soccer balls -/
structure PurchaseInfo where
  basketball_price : ℝ
  soccer_ball_price : ℝ
  basketball_count : ℕ
  soccer_ball_count : ℕ
  total_cost : ℝ
  price_difference : ℝ

/-- The system of equations for the purchase -/
def purchase_equations (p : PurchaseInfo) : Prop :=
  p.basketball_count * p.basketball_price + p.soccer_ball_count * p.soccer_ball_price = p.total_cost ∧
  p.basketball_price - p.soccer_ball_price = p.price_difference

theorem purchase_system_of_equations (p : PurchaseInfo) 
  (h1 : p.basketball_count = 3)
  (h2 : p.soccer_ball_count = 2)
  (h3 : p.total_cost = 474)
  (h4 : p.price_difference = 8) :
  purchase_equations p ↔ 
  (3 * p.basketball_price + 2 * p.soccer_ball_price = 474 ∧
   p.basketball_price - p.soccer_ball_price = 8) :=
by sorry

end NUMINAMATH_CALUDE_purchase_system_of_equations_l2956_295698


namespace NUMINAMATH_CALUDE_earth_sun_distance_scientific_notation_l2956_295691

/-- The distance from Earth to Sun in kilometers -/
def earth_sun_distance : ℕ := 150000000

/-- Represents a number in scientific notation as a pair (coefficient, exponent) -/
def scientific_notation := ℝ × ℤ

/-- Converts a natural number to scientific notation -/
def to_scientific_notation (n : ℕ) : scientific_notation :=
  sorry

theorem earth_sun_distance_scientific_notation :
  to_scientific_notation earth_sun_distance = (1.5, 8) :=
sorry

end NUMINAMATH_CALUDE_earth_sun_distance_scientific_notation_l2956_295691


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_sqrt_16_l2956_295660

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ := Real.sqrt (Real.sqrt x)

-- State the theorem
theorem arithmetic_sqrt_of_sqrt_16 : arithmetic_sqrt 16 = 2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_sqrt_16_l2956_295660


namespace NUMINAMATH_CALUDE_valid_choices_count_l2956_295646

/-- The number of elements in the list -/
def n : ℕ := 2016

/-- The number of elements to be shuffled -/
def m : ℕ := 2014

/-- Function to calculate the number of valid ways to choose a and b -/
def count_valid_choices : ℕ := sorry

/-- Theorem stating that the number of valid choices is equal to 508536 -/
theorem valid_choices_count : count_valid_choices = 508536 := by sorry

end NUMINAMATH_CALUDE_valid_choices_count_l2956_295646


namespace NUMINAMATH_CALUDE_effective_discount_l2956_295608

theorem effective_discount (initial_discount coupon_discount : ℝ) : 
  initial_discount = 0.6 →
  coupon_discount = 0.3 →
  let sale_price := 1 - initial_discount
  let final_price := sale_price * (1 - coupon_discount)
  1 - final_price = 0.72 :=
by sorry

end NUMINAMATH_CALUDE_effective_discount_l2956_295608


namespace NUMINAMATH_CALUDE_isoscelesTriangles29Count_l2956_295645

/-- An isosceles triangle with integer side lengths and perimeter 29 -/
structure IsoscelesTriangle29 where
  base : ℕ
  side : ℕ
  isIsosceles : side * 2 + base = 29
  isTriangle : base < side + side

/-- The count of valid isosceles triangles with perimeter 29 -/
def countIsoscelesTriangles29 : ℕ := sorry

/-- Theorem stating that there are exactly 5 isosceles triangles with integer side lengths and perimeter 29 -/
theorem isoscelesTriangles29Count : countIsoscelesTriangles29 = 5 := by sorry

end NUMINAMATH_CALUDE_isoscelesTriangles29Count_l2956_295645


namespace NUMINAMATH_CALUDE_clock_second_sale_price_l2956_295689

/-- Represents the clock sale scenario in the shop -/
structure ClockSale where
  originalCost : ℝ
  firstSalePrice : ℝ
  buyBackPrice : ℝ
  secondSalePrice : ℝ

/-- The conditions of the clock sale problem -/
def clockSaleProblem (sale : ClockSale) : Prop :=
  sale.firstSalePrice = 1.2 * sale.originalCost ∧
  sale.buyBackPrice = 0.5 * sale.firstSalePrice ∧
  sale.originalCost - sale.buyBackPrice = 100 ∧
  sale.secondSalePrice = sale.buyBackPrice * 1.8

/-- The theorem stating that under the given conditions, 
    the second sale price is 270 -/
theorem clock_second_sale_price (sale : ClockSale) :
  clockSaleProblem sale → sale.secondSalePrice = 270 := by
  sorry

end NUMINAMATH_CALUDE_clock_second_sale_price_l2956_295689


namespace NUMINAMATH_CALUDE_rectangular_plot_perimeter_l2956_295648

/-- Given a rectangular plot with length 10 meters more than width,
    and fencing cost of Rs. 910 at Rs. 6.5 per meter,
    prove that the perimeter is 140 meters. -/
theorem rectangular_plot_perimeter : 
  ∀ (width length : ℝ),
  length = width + 10 →
  910 = (2 * (length + width)) * 6.5 →
  2 * (length + width) = 140 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_plot_perimeter_l2956_295648


namespace NUMINAMATH_CALUDE_sum_of_digits_N_l2956_295655

def N : ℕ := 9 + 99 + 999 + 9999 + 99999 + 999999

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem sum_of_digits_N : sum_of_digits N = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_N_l2956_295655


namespace NUMINAMATH_CALUDE_parallel_perpendicular_implication_l2956_295674

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem parallel_perpendicular_implication 
  (m n : Line) (α : Plane) :
  parallel m n → perpendicular m α → perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_implication_l2956_295674


namespace NUMINAMATH_CALUDE_berry_cobbler_problem_l2956_295639

theorem berry_cobbler_problem (total_needed : ℕ) (blueberries : ℕ) (to_buy : ℕ) 
  (h1 : total_needed = 21)
  (h2 : blueberries = 8)
  (h3 : to_buy = 9) :
  total_needed - (blueberries + to_buy) = 4 := by
  sorry

end NUMINAMATH_CALUDE_berry_cobbler_problem_l2956_295639


namespace NUMINAMATH_CALUDE_smallest_number_l2956_295696

theorem smallest_number (s : Set ℚ) (h : s = {-1, 0, -3, -2}) : 
  ∃ x ∈ s, ∀ y ∈ s, x ≤ y ∧ x = -3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l2956_295696
