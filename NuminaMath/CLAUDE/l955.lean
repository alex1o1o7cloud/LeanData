import Mathlib

namespace NUMINAMATH_CALUDE_not_collinear_ABC_l955_95590

/-- Three points in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of collinearity for three points -/
def collinear (p q r : Point2D) : Prop :=
  (q.y - p.y) * (r.x - q.x) = (r.y - q.y) * (q.x - p.x)

/-- The theorem stating that points A(-1,4), B(-3,2), and C(0,6) are not collinear -/
theorem not_collinear_ABC : 
  let A : Point2D := ⟨-1, 4⟩
  let B : Point2D := ⟨-3, 2⟩
  let C : Point2D := ⟨0, 6⟩
  ¬(collinear A B C) := by
  sorry


end NUMINAMATH_CALUDE_not_collinear_ABC_l955_95590


namespace NUMINAMATH_CALUDE_sum_of_distances_is_ten_l955_95575

/-- Given a circle tangent to the sides of an angle at points A and B, with a point C on the circle,
    this structure represents the distances and conditions of the problem. -/
structure CircleTangentProblem where
  -- Distance from C to line AB
  h : ℝ
  -- Distance from C to the side of the angle passing through A
  h_A : ℝ
  -- Distance from C to the side of the angle passing through B
  h_B : ℝ
  -- Condition: h = 4
  h_eq_four : h = 4
  -- Condition: One distance is four times the other
  one_distance_four_times_other : h_B = 4 * h_A

/-- The theorem stating that the sum of distances from C to the sides of the angle is 10. -/
theorem sum_of_distances_is_ten (p : CircleTangentProblem) : p.h_A + p.h_B = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_distances_is_ten_l955_95575


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l955_95563

theorem smallest_n_for_candy_purchase : 
  (∃ n : ℕ, n > 0 ∧ 
    24 * n % 10 = 0 ∧ 
    24 * n % 16 = 0 ∧ 
    24 * n % 18 = 0 ∧
    (∀ m : ℕ, m > 0 → 
      (24 * m % 10 = 0 ∧ 24 * m % 16 = 0 ∧ 24 * m % 18 = 0) → 
      m ≥ n)) → 
  (∃ n : ℕ, n = 30 ∧
    24 * n % 10 = 0 ∧ 
    24 * n % 16 = 0 ∧ 
    24 * n % 18 = 0 ∧
    (∀ m : ℕ, m > 0 → 
      (24 * m % 10 = 0 ∧ 24 * m % 16 = 0 ∧ 24 * m % 18 = 0) → 
      m ≥ n)) :=
by sorry

#check smallest_n_for_candy_purchase

end NUMINAMATH_CALUDE_smallest_n_for_candy_purchase_l955_95563


namespace NUMINAMATH_CALUDE_computer_price_reduction_l955_95561

/-- Given a computer price reduction of 40% resulting in a final price of 'a' yuan,
    prove that the original price was (5/3)a yuan. -/
theorem computer_price_reduction (a : ℝ) : 
  (∃ (original_price : ℝ), 
    original_price * (1 - 0.4) = a ∧ 
    original_price = (5/3) * a) :=
by sorry

end NUMINAMATH_CALUDE_computer_price_reduction_l955_95561


namespace NUMINAMATH_CALUDE_trip_price_calculation_egypt_trip_price_l955_95587

theorem trip_price_calculation (num_people : ℕ) (discount_per_person : ℕ) (total_cost_after_discount : ℕ) : ℕ :=
  let total_discount := num_people * discount_per_person
  let total_cost_before_discount := total_cost_after_discount + total_discount
  let original_price_per_person := total_cost_before_discount / num_people
  original_price_per_person

theorem egypt_trip_price : 
  trip_price_calculation 2 14 266 = 147 := by
  sorry

end NUMINAMATH_CALUDE_trip_price_calculation_egypt_trip_price_l955_95587


namespace NUMINAMATH_CALUDE_remainder_problem_l955_95511

theorem remainder_problem (x : ℕ) :
  x < 100 →
  x % 3 = 2 →
  x % 4 = 2 →
  x % 5 = 2 →
  x = 2 ∨ x = 62 :=
by sorry

end NUMINAMATH_CALUDE_remainder_problem_l955_95511


namespace NUMINAMATH_CALUDE_square_sum_value_l955_95544

theorem square_sum_value (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 10) : x^2 + y^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_value_l955_95544


namespace NUMINAMATH_CALUDE_octagon_arc_length_l955_95595

/-- The length of the arc intercepted by one side of a regular octagon inscribed in a circle -/
theorem octagon_arc_length (r : ℝ) (h : r = 4) : 
  (2 * π * r) / 8 = π := by sorry

end NUMINAMATH_CALUDE_octagon_arc_length_l955_95595


namespace NUMINAMATH_CALUDE_reciprocal_problem_l955_95531

theorem reciprocal_problem (x : ℚ) (h : 8 * x = 3) : 150 * (1 / x) = 400 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_problem_l955_95531


namespace NUMINAMATH_CALUDE_sandys_change_l955_95598

/-- Represents the cost of a drink order -/
structure DrinkOrder where
  cappuccino : ℕ
  icedTea : ℕ
  cafeLatte : ℕ
  espresso : ℕ

/-- Calculates the total cost of a drink order -/
def totalCost (order : DrinkOrder) : ℚ :=
  2 * order.cappuccino + 3 * order.icedTea + 1.5 * order.cafeLatte + 1 * order.espresso

/-- Calculates the change received from a given payment -/
def changeReceived (payment : ℚ) (order : DrinkOrder) : ℚ :=
  payment - totalCost order

/-- Sandy's specific drink order -/
def sandysOrder : DrinkOrder :=
  { cappuccino := 3
  , icedTea := 2
  , cafeLatte := 2
  , espresso := 2 }

/-- Theorem stating that Sandy receives $3 in change -/
theorem sandys_change :
  changeReceived 20 sandysOrder = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandys_change_l955_95598


namespace NUMINAMATH_CALUDE_total_owls_on_fence_l955_95501

def initial_owls : ℕ := 3
def joining_owls : ℕ := 2

theorem total_owls_on_fence : initial_owls + joining_owls = 5 := by
  sorry

end NUMINAMATH_CALUDE_total_owls_on_fence_l955_95501


namespace NUMINAMATH_CALUDE_toluene_formation_l955_95503

-- Define the chemical species involved in the reaction
structure ChemicalSpecies where
  formula : String
  moles : ℝ

-- Define the chemical reaction
def reaction (reactant1 reactant2 product1 product2 : ChemicalSpecies) : Prop :=
  reactant1.formula = "C6H6" ∧ 
  reactant2.formula = "CH4" ∧ 
  product1.formula = "C6H5CH3" ∧ 
  product2.formula = "H2" ∧
  reactant1.moles = reactant2.moles ∧
  product1.moles = product2.moles ∧
  reactant1.moles = product1.moles

-- Theorem statement
theorem toluene_formation 
  (benzene : ChemicalSpecies)
  (methane : ChemicalSpecies)
  (toluene : ChemicalSpecies)
  (hydrogen : ChemicalSpecies)
  (h1 : reaction benzene methane toluene hydrogen)
  (h2 : methane.moles = 3)
  (h3 : hydrogen.moles = 3) :
  toluene.moles = 3 :=
sorry

end NUMINAMATH_CALUDE_toluene_formation_l955_95503


namespace NUMINAMATH_CALUDE_problem_solution_l955_95520

theorem problem_solution (a : ℚ) : a + a/3 - a/9 = 10/3 → a = 30/11 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l955_95520


namespace NUMINAMATH_CALUDE_slower_walking_speed_l955_95537

/-- Proves that the slower walking speed is 10 km/hr given the conditions of the problem -/
theorem slower_walking_speed 
  (actual_distance : ℝ) 
  (faster_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 13.333333333333332)
  (h2 : faster_speed = 25)
  (h3 : additional_distance = 20)
  : ∃ (v : ℝ), 
    actual_distance / v = (actual_distance + additional_distance) / faster_speed ∧ 
    v = 10 := by
  sorry

end NUMINAMATH_CALUDE_slower_walking_speed_l955_95537


namespace NUMINAMATH_CALUDE_alberto_bjorn_difference_l955_95502

/-- Represents a biker's travel distance over time --/
structure BikerTravel where
  miles : ℝ
  hours : ℝ

/-- Alberto's travel after 4 hours --/
def alberto : BikerTravel :=
  { miles := 60
  , hours := 4 }

/-- Bjorn's travel after 4 hours --/
def bjorn : BikerTravel :=
  { miles := 45
  , hours := 4 }

/-- The difference in miles traveled between two bikers --/
def mileDifference (a b : BikerTravel) : ℝ :=
  a.miles - b.miles

/-- Theorem stating the difference in miles traveled between Alberto and Bjorn --/
theorem alberto_bjorn_difference :
  mileDifference alberto bjorn = 15 := by
  sorry

end NUMINAMATH_CALUDE_alberto_bjorn_difference_l955_95502


namespace NUMINAMATH_CALUDE_last_interval_correct_l955_95519

/-- Represents a clock with specific ringing behavior -/
structure Clock where
  n : ℕ  -- number of rings per day
  x : ℝ  -- time between first two rings (in hours)
  y : ℝ  -- increase in time between subsequent rings (in hours)

/-- The time between the last two rings of the clock -/
def lastInterval (c : Clock) : ℝ :=
  c.x + (c.n - 3 : ℝ) * c.y

theorem last_interval_correct (c : Clock) (h : c.n ≥ 2) :
  lastInterval c = c.x + (c.n - 3 : ℝ) * c.y :=
sorry

end NUMINAMATH_CALUDE_last_interval_correct_l955_95519


namespace NUMINAMATH_CALUDE_real_pair_existence_l955_95532

theorem real_pair_existence :
  (∃ (u v : ℝ), (∃ (q : ℚ), u + v = q) ∧ 
    (∀ (n : ℕ), n ≥ 2 → ∀ (q : ℚ), u^n + v^n ≠ q)) ∧
  (¬ ∃ (u v : ℝ), (∀ (q : ℚ), u + v ≠ q) ∧ 
    (∀ (n : ℕ), n ≥ 2 → ∃ (q : ℚ), u^n + v^n = q)) :=
by sorry

end NUMINAMATH_CALUDE_real_pair_existence_l955_95532


namespace NUMINAMATH_CALUDE_gcd_of_35_91_840_l955_95539

theorem gcd_of_35_91_840 : Nat.gcd 35 (Nat.gcd 91 840) = 7 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_35_91_840_l955_95539


namespace NUMINAMATH_CALUDE_range_of_k_for_fractional_equation_l955_95597

theorem range_of_k_for_fractional_equation :
  ∀ k x : ℝ,
  (x > 0) →
  (x ≠ 2) →
  (1 / (x - 2) + 3 = (3 - k) / (2 - x)) →
  (k > -2 ∧ k ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_range_of_k_for_fractional_equation_l955_95597


namespace NUMINAMATH_CALUDE_edward_games_count_l955_95558

theorem edward_games_count :
  let sold_games : ℕ := 19
  let boxes_used : ℕ := 2
  let games_per_box : ℕ := 8
  let packed_games : ℕ := boxes_used * games_per_box
  let total_games : ℕ := sold_games + packed_games
  total_games = 35 := by sorry

end NUMINAMATH_CALUDE_edward_games_count_l955_95558


namespace NUMINAMATH_CALUDE_quadratic_always_positive_range_l955_95568

theorem quadratic_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_range_l955_95568


namespace NUMINAMATH_CALUDE_partner_a_share_l955_95512

/-- Calculates the share of a partner in a partnership based on investments and known share of another partner. -/
def calculate_share (investment_a investment_b investment_c : ℚ) (share_b : ℚ) : ℚ :=
  let total_investment := investment_a + investment_b + investment_c
  let ratio_a := investment_a / total_investment
  let ratio_b := investment_b / total_investment
  let total_profit := share_b / ratio_b
  ratio_a * total_profit

/-- Theorem stating that given the investments and b's share, a's share is approximately $560. -/
theorem partner_a_share (investment_a investment_b investment_c share_b : ℚ) 
  (h1 : investment_a = 7000)
  (h2 : investment_b = 11000)
  (h3 : investment_c = 18000)
  (h4 : share_b = 880) :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  |calculate_share investment_a investment_b investment_c share_b - 560| < ε :=
sorry

end NUMINAMATH_CALUDE_partner_a_share_l955_95512


namespace NUMINAMATH_CALUDE_building_floors_l955_95534

/-- Represents a staircase in the building -/
structure Staircase where
  steps : ℕ

/-- Represents the building with three staircases -/
structure Building where
  staircase_a : Staircase
  staircase_b : Staircase
  staircase_c : Staircase

/-- The number of floors in the building is equal to the GCD of the number of steps in each staircase -/
theorem building_floors (b : Building) 
  (h1 : b.staircase_a.steps = 104)
  (h2 : b.staircase_b.steps = 117)
  (h3 : b.staircase_c.steps = 156) : 
  ∃ (floors : ℕ), floors = Nat.gcd (Nat.gcd b.staircase_a.steps b.staircase_b.steps) b.staircase_c.steps ∧ 
    floors = 13 := by
  sorry

end NUMINAMATH_CALUDE_building_floors_l955_95534


namespace NUMINAMATH_CALUDE_janice_purchase_l955_95584

theorem janice_purchase (a b c : ℕ) : 
  a + b + c = 30 →
  30 * a + 200 * b + 300 * c = 3000 →
  a = 20 :=
by sorry

end NUMINAMATH_CALUDE_janice_purchase_l955_95584


namespace NUMINAMATH_CALUDE_skyscraper_anniversary_l955_95588

/-- Calculates the number of years in the future when it will be 5 years before the 200th anniversary of a skyscraper built 100 years ago. -/
theorem skyscraper_anniversary (years_since_built : ℕ) (years_to_anniversary : ℕ) (years_before_anniversary : ℕ) : 
  years_since_built = 100 →
  years_to_anniversary = 200 →
  years_before_anniversary = 5 →
  years_to_anniversary - years_before_anniversary - years_since_built = 95 :=
by sorry

end NUMINAMATH_CALUDE_skyscraper_anniversary_l955_95588


namespace NUMINAMATH_CALUDE_kaleb_final_amount_l955_95513

def kaleb_business (spring_earnings summer_earnings supplies_cost : ℕ) : ℕ :=
  spring_earnings + summer_earnings - supplies_cost

theorem kaleb_final_amount :
  kaleb_business 4 50 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_kaleb_final_amount_l955_95513


namespace NUMINAMATH_CALUDE_product_magnitude_l955_95562

open Complex

theorem product_magnitude (z₁ z₂ : ℂ) (h1 : abs z₁ = 3) (h2 : z₂ = 2 + I) : 
  abs (z₁ * z₂) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_product_magnitude_l955_95562


namespace NUMINAMATH_CALUDE_cory_fruit_eating_orders_l955_95514

/-- Represents the number of fruits Cory has of each type -/
structure FruitInventory where
  apples : Nat
  bananas : Nat
  mangoes : Nat

/-- Represents the constraints of Cory's fruit-eating schedule -/
structure EatingSchedule where
  days : Nat
  startsWithApple : Bool
  endsWithApple : Bool

/-- Calculates the number of ways Cory can eat his fruits given his inventory and schedule constraints -/
def countEatingOrders (inventory : FruitInventory) (schedule : EatingSchedule) : Nat :=
  sorry

/-- Theorem stating that given Cory's specific fruit inventory and eating schedule, 
    there are exactly 80 different orders in which he can eat his fruits -/
theorem cory_fruit_eating_orders :
  let inventory : FruitInventory := ⟨3, 3, 1⟩
  let schedule : EatingSchedule := ⟨7, true, true⟩
  countEatingOrders inventory schedule = 80 :=
by sorry

end NUMINAMATH_CALUDE_cory_fruit_eating_orders_l955_95514


namespace NUMINAMATH_CALUDE_square_area_in_circle_l955_95571

/-- Given a circle with radius 1 and a square with two vertices on the circle
    and one edge passing through the center, prove the area of the square is 4/5 -/
theorem square_area_in_circle (circle_radius : ℝ) (square_side : ℝ) : 
  circle_radius = 1 →
  ∃ (x : ℝ), square_side = 2 * x ∧ 
  x ^ 2 + (2 * x) ^ 2 = circle_radius ^ 2 →
  square_side ^ 2 = 4 / 5 := by
  sorry

#check square_area_in_circle

end NUMINAMATH_CALUDE_square_area_in_circle_l955_95571


namespace NUMINAMATH_CALUDE_length_PR_l955_95522

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 49}

structure PointsOnCircle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  h_P_on_circle : P ∈ Circle
  h_Q_on_circle : Q ∈ Circle
  h_PQ_distance : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 64
  h_R_midpoint : R = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Define the theorem
theorem length_PR (points : PointsOnCircle) : 
  ((points.P.1 - points.R.1)^2 + (points.P.2 - points.R.2)^2)^(1/2) = 4 * (2^(1/2)) := by
  sorry

end NUMINAMATH_CALUDE_length_PR_l955_95522


namespace NUMINAMATH_CALUDE_custom_op_theorem_l955_95580

-- Define the custom operation ⊗
def custom_op (P Q : Set ℝ) : Set ℝ :=
  {x | x ∈ P ∪ Q ∧ x ∉ P ∩ Q}

theorem custom_op_theorem :
  let P : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}
  let Q : Set ℝ := {x | x > 1}
  custom_op P Q = {x | (0 ≤ x ∧ x ≤ 1) ∨ (x > 2)} := by
sorry

end NUMINAMATH_CALUDE_custom_op_theorem_l955_95580


namespace NUMINAMATH_CALUDE_shooting_competition_probabilities_l955_95515

/-- Probability of A hitting the target in a single shot -/
def prob_A_hit : ℚ := 2/3

/-- Probability of B hitting the target in a single shot -/
def prob_B_hit : ℚ := 3/4

/-- Number of consecutive shots -/
def num_shots : ℕ := 3

theorem shooting_competition_probabilities :
  let prob_A_miss_at_least_once := 1 - prob_A_hit ^ num_shots
  let prob_A_hit_twice := (num_shots.choose 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit)
  let prob_B_hit_once := (num_shots.choose 1 : ℚ) * prob_B_hit * (1 - prob_B_hit)^2
  prob_A_miss_at_least_once = 19/27 ∧
  prob_A_hit_twice * prob_B_hit_once = 1/16 := by
  sorry


end NUMINAMATH_CALUDE_shooting_competition_probabilities_l955_95515


namespace NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_7_l955_95582

def ends_in_9 (n : ℕ) : Prop := n % 10 = 9

theorem smallest_positive_integer_ending_in_9_divisible_by_7 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_9 n ∧ n % 7 = 0 ∧
  ∀ (m : ℕ), m > 0 → ends_in_9 m → m % 7 = 0 → m ≥ n :=
by
  use 49
  sorry

end NUMINAMATH_CALUDE_smallest_positive_integer_ending_in_9_divisible_by_7_l955_95582


namespace NUMINAMATH_CALUDE_product_105_95_l955_95510

theorem product_105_95 : 105 * 95 = 9975 := by
  sorry

end NUMINAMATH_CALUDE_product_105_95_l955_95510


namespace NUMINAMATH_CALUDE_notebook_notepad_pen_cost_l955_95535

theorem notebook_notepad_pen_cost (x y z : ℤ) : 
  x + 3*y + 2*z = 98 →
  3*x + y = 5*z - 36 →
  Even x →
  x = 4 ∧ y = 22 ∧ z = 14 := by
sorry

end NUMINAMATH_CALUDE_notebook_notepad_pen_cost_l955_95535


namespace NUMINAMATH_CALUDE_range_of_a_l955_95530

theorem range_of_a (x a : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 1 → (x - a) * (x - a - 4) > 0) ∧ 
  (∃ x, (x - a) * (x - a - 4) > 0 ∧ (x < -2 ∨ x > 1)) →
  a < -6 ∨ a > 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l955_95530


namespace NUMINAMATH_CALUDE_fraction_simplification_l955_95573

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  (1 - 2 / (x + 1)) / (x / (x + 1)) = (x - 1) / x := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l955_95573


namespace NUMINAMATH_CALUDE_parallelogram_area_equals_rectangle_area_l955_95504

/-- Represents a rectangle with a given base and area -/
structure Rectangle where
  base : ℝ
  area : ℝ

/-- Represents a parallelogram with a given base and height -/
structure Parallelogram where
  base : ℝ
  height : ℝ

/-- Theorem: Given a rectangle with base 6 and area 24, and a parallelogram sharing the same base and height,
    the area of the parallelogram is 24 -/
theorem parallelogram_area_equals_rectangle_area 
  (rect : Rectangle) 
  (para : Parallelogram) 
  (h1 : rect.base = 6) 
  (h2 : rect.area = 24) 
  (h3 : para.base = rect.base) 
  (h4 : para.height = rect.area / rect.base) : 
  para.base * para.height = 24 := by
  sorry

#check parallelogram_area_equals_rectangle_area

end NUMINAMATH_CALUDE_parallelogram_area_equals_rectangle_area_l955_95504


namespace NUMINAMATH_CALUDE_squares_below_line_eq_660_l955_95500

/-- The number of squares below the line 7x + 221y = 1547 in the first quadrant -/
def squares_below_line : ℕ :=
  let x_intercept : ℕ := 221
  let y_intercept : ℕ := 7
  let total_squares : ℕ := x_intercept * y_intercept
  let diagonal_squares : ℕ := x_intercept + y_intercept - 1
  let non_diagonal_squares : ℕ := total_squares - diagonal_squares
  non_diagonal_squares / 2

/-- The number of squares below the line 7x + 221y = 1547 in the first quadrant is 660 -/
theorem squares_below_line_eq_660 : squares_below_line = 660 := by
  sorry

end NUMINAMATH_CALUDE_squares_below_line_eq_660_l955_95500


namespace NUMINAMATH_CALUDE_dartboard_angle_l955_95556

theorem dartboard_angle (probability : ℝ) (angle : ℝ) : 
  probability = 1/4 → angle = 90 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_angle_l955_95556


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l955_95554

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l955_95554


namespace NUMINAMATH_CALUDE_line_through_points_l955_95566

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Function to check if a point lies on a line
def pointOnLine (p : Point2D) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Theorem statement
theorem line_through_points :
  let p1 : Point2D := ⟨-2, 2⟩
  let p2 : Point2D := ⟨0, 6⟩
  let l : Line := ⟨2, -1, 6⟩
  pointOnLine p1 l ∧ pointOnLine p2 l := by sorry

end NUMINAMATH_CALUDE_line_through_points_l955_95566


namespace NUMINAMATH_CALUDE_locus_is_circle_l955_95594

/-- Given two fixed points A and B in a plane, the locus of points C 
    satisfying $\overrightarrow{AC} \cdot \overrightarrow{BC} = 1$ is a circle. -/
theorem locus_is_circle (A B : ℝ × ℝ) : 
  {C : ℝ × ℝ | (C.1 - A.1, C.2 - A.2) • (C.1 - B.1, C.2 - B.2) = 1} = 
  {C : ℝ × ℝ | ∃ (center : ℝ × ℝ) (radius : ℝ), 
    (C.1 - center.1)^2 + (C.2 - center.2)^2 = radius^2} :=
sorry

end NUMINAMATH_CALUDE_locus_is_circle_l955_95594


namespace NUMINAMATH_CALUDE_squirrel_acorns_l955_95577

/-- Represents the number of acorns each animal hides per hole -/
structure AcornsPerHole where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- Represents the number of holes each animal dug -/
structure Holes where
  chipmunk : ℕ
  squirrel : ℕ
  rabbit : ℕ

/-- The forest scenario with animals hiding acorns -/
def ForestScenario (a : AcornsPerHole) (h : Holes) : Prop :=
  -- Chipmunk and squirrel stash the same number of acorns
  a.chipmunk * h.chipmunk = a.squirrel * h.squirrel ∧
  -- Rabbit stashes the same number of acorns as the chipmunk
  a.rabbit * h.rabbit = a.chipmunk * h.chipmunk ∧
  -- Rabbit needs 3 more holes than the squirrel
  h.rabbit = h.squirrel + 3

/-- The theorem stating that the squirrel stashed 40 acorns -/
theorem squirrel_acorns (a : AcornsPerHole) (h : Holes)
  (ha : a.chipmunk = 4 ∧ a.squirrel = 5 ∧ a.rabbit = 3)
  (hf : ForestScenario a h) : 
  a.squirrel * h.squirrel = 40 := by
  sorry

end NUMINAMATH_CALUDE_squirrel_acorns_l955_95577


namespace NUMINAMATH_CALUDE_sin_equal_of_sum_pi_l955_95593

theorem sin_equal_of_sum_pi (α β : Real) (h : α + β = Real.pi) : Real.sin α = Real.sin β := by
  sorry

end NUMINAMATH_CALUDE_sin_equal_of_sum_pi_l955_95593


namespace NUMINAMATH_CALUDE_quiz_answer_key_l955_95536

theorem quiz_answer_key (n : ℕ) : 
  (2^5 - 2) * 4^n = 480 → n = 2 := by
  sorry

end NUMINAMATH_CALUDE_quiz_answer_key_l955_95536


namespace NUMINAMATH_CALUDE_average_disk_space_per_hour_l955_95541

/-- Proves that the average disk space per hour of music in a library
    containing 12 days of music and occupying 16,000 megabytes,
    rounded to the nearest whole number, is 56 megabytes. -/
theorem average_disk_space_per_hour (days : ℕ) (total_space : ℕ) 
  (h1 : days = 12) (h2 : total_space = 16000) : 
  round ((total_space : ℝ) / (days * 24)) = 56 := by
  sorry

#check average_disk_space_per_hour

end NUMINAMATH_CALUDE_average_disk_space_per_hour_l955_95541


namespace NUMINAMATH_CALUDE_right_triangle_sin_z_l955_95526

theorem right_triangle_sin_z (X Y Z : ℝ) : 
  X + Y + Z = π →
  X = π / 2 →
  Real.cos Y = 3 / 5 →
  Real.sin Z = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_z_l955_95526


namespace NUMINAMATH_CALUDE_five_boys_three_girls_arrangements_l955_95569

/-- The number of arrangements of boys and girls in a row -/
def arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  (Nat.factorial (num_boys + 1)) * (Nat.factorial num_girls)

/-- Theorem stating the number of arrangements for 5 boys and 3 girls -/
theorem five_boys_three_girls_arrangements :
  arrangements 5 3 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_five_boys_three_girls_arrangements_l955_95569


namespace NUMINAMATH_CALUDE_parallel_line_y_intercept_l955_95549

/-- A line parallel to y = -3x - 6 passing through (3, -1) has y-intercept 8 -/
theorem parallel_line_y_intercept :
  ∀ (b : ℝ → ℝ),
  (∀ x y, b x = y ↔ ∃ k, y = -3 * x + k) →  -- b is parallel to y = -3x - 6
  b 3 = -1 →                               -- b passes through (3, -1)
  ∃ k, b 0 = k ∧ k = 8 :=                  -- y-intercept of b is 8
by sorry

end NUMINAMATH_CALUDE_parallel_line_y_intercept_l955_95549


namespace NUMINAMATH_CALUDE_jakes_balloons_l955_95591

theorem jakes_balloons (total : ℕ) (allans_extra : ℕ) (h1 : total = 56) (h2 : allans_extra = 8) :
  ∃ (jake : ℕ), jake + (jake + allans_extra) = total ∧ jake = 24 := by
  sorry

end NUMINAMATH_CALUDE_jakes_balloons_l955_95591


namespace NUMINAMATH_CALUDE_two_digit_number_sum_l955_95586

theorem two_digit_number_sum (x : ℕ) : 
  x < 10 →                             -- units digit is less than 10
  (11 * x + 30) % (2 * x + 3) = 3 →    -- remainder is 3
  (11 * x + 30) / (2 * x + 3) = 7 →    -- quotient is 7
  2 * x + 3 = 7 :=                     -- sum of digits is 7
by sorry

end NUMINAMATH_CALUDE_two_digit_number_sum_l955_95586


namespace NUMINAMATH_CALUDE_max_sum_distances_l955_95547

/-- Given a real number k, two lines l₁ and l₂, and points P, Q, and M,
    prove that the maximum value of |MP| + |MQ| is 4. -/
theorem max_sum_distances (k : ℝ) :
  let P : ℝ × ℝ := (0, 0)
  let Q : ℝ × ℝ := (2, 2)
  let l₁ := {(x, y) : ℝ × ℝ | k * x + y = 0}
  let l₂ := {(x, y) : ℝ × ℝ | k * x - y - 2 * k + 2 = 0}
  let circle := {M : ℝ × ℝ | (M.1 - 1)^2 + (M.2 - 1)^2 = 2}
  ∀ M ∈ circle, (‖M - P‖ + ‖M - Q‖) ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_distances_l955_95547


namespace NUMINAMATH_CALUDE_fair_distribution_l955_95565

/-- Represents the number of books each player brings to the game -/
def books_per_player : ℕ := 4

/-- Represents the total number of books in the game -/
def total_books : ℕ := 2 * books_per_player

/-- Represents the number of points needed to win the game -/
def points_to_win : ℕ := 3

/-- Represents player A's current points -/
def a_points : ℕ := 2

/-- Represents player B's current points -/
def b_points : ℕ := 1

/-- Represents the probability of player A winning the game -/
def prob_a_wins : ℚ := 3/4

/-- Represents the probability of player B winning the game -/
def prob_b_wins : ℚ := 1/4

/-- Theorem stating the fair distribution of books -/
theorem fair_distribution :
  let a_books := (total_books : ℚ) * prob_a_wins
  let b_books := (total_books : ℚ) * prob_b_wins
  a_books = 6 ∧ b_books = 2 := by
  sorry

end NUMINAMATH_CALUDE_fair_distribution_l955_95565


namespace NUMINAMATH_CALUDE_symmetry_of_exponential_graphs_l955_95551

theorem symmetry_of_exponential_graphs :
  ∀ (a b : ℝ), b = 3^a ↔ -b = -(3^(-a)) := by sorry

end NUMINAMATH_CALUDE_symmetry_of_exponential_graphs_l955_95551


namespace NUMINAMATH_CALUDE_cannot_end_with_two_l955_95552

-- Define the initial set of numbers
def initial_numbers : List Nat := List.range 2017

-- Define the operation of taking the difference
def difference_operation (a b : Nat) : Nat := Int.natAbs (a - b)

-- Define the property of maintaining odd sum parity
def maintains_odd_sum_parity (numbers : List Nat) : Prop :=
  List.sum numbers % 2 = 1

-- Define the final state we want to disprove
def final_state (numbers : List Nat) : Prop :=
  numbers = [2]

-- Theorem statement
theorem cannot_end_with_two :
  ¬ ∃ (final_numbers : List Nat),
    (maintains_odd_sum_parity initial_numbers →
     maintains_odd_sum_parity final_numbers) ∧
    final_state final_numbers :=
by sorry

end NUMINAMATH_CALUDE_cannot_end_with_two_l955_95552


namespace NUMINAMATH_CALUDE_cubic_root_equation_solution_l955_95572

theorem cubic_root_equation_solution (y : ℝ) : 
  (y + 16) ^ (1/3) - (y - 4) ^ (1/3) = 2 → y = 12 ∨ y = -8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_equation_solution_l955_95572


namespace NUMINAMATH_CALUDE_incircle_radius_l955_95533

/-- The ellipse with semi-major axis 4 and semi-minor axis 1 -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2 = 1

/-- The incircle of the inscribed triangle ABC -/
def incircle (x y r : ℝ) : Prop := (x-2)^2 + y^2 = r^2

/-- A is the left vertex of the ellipse -/
def A : ℝ × ℝ := (-4, 0)

/-- Theorem: The radius of the incircle is 5 -/
theorem incircle_radius : ∃ (r : ℝ), 
  (∀ x y, incircle x y r → ellipse x y) ∧ 
  (incircle A.1 A.2 r) ∧ 
  r = 5 := by sorry

end NUMINAMATH_CALUDE_incircle_radius_l955_95533


namespace NUMINAMATH_CALUDE_elvin_first_month_bill_l955_95529

/-- Represents Elvin's monthly telephone bill structure -/
structure PhoneBill where
  fixed_charge : ℝ
  call_charge : ℝ

/-- Calculates the total bill given a PhoneBill -/
def total_bill (bill : PhoneBill) : ℝ :=
  bill.fixed_charge + bill.call_charge

theorem elvin_first_month_bill :
  ∀ (bill1 bill2 : PhoneBill),
    total_bill bill1 = 52 →
    total_bill bill2 = 76 →
    bill2.call_charge = 2 * bill1.call_charge →
    bill1.fixed_charge = bill2.fixed_charge →
    total_bill bill1 = 52 := by
  sorry

end NUMINAMATH_CALUDE_elvin_first_month_bill_l955_95529


namespace NUMINAMATH_CALUDE_box_area_l955_95545

theorem box_area (V : ℝ) (A2 A3 : ℝ) (hV : V = 720) (hA2 : A2 = 72) (hA3 : A3 = 60) :
  ∃ (L W H : ℝ), L > 0 ∧ W > 0 ∧ H > 0 ∧ 
    L * W * H = V ∧
    W * H = A2 ∧
    L * H = A3 ∧
    L * W = 120 :=
by sorry

end NUMINAMATH_CALUDE_box_area_l955_95545


namespace NUMINAMATH_CALUDE_extremum_derivative_zero_necessary_not_sufficient_l955_95579

/-- A function f: ℝ → ℝ has an extremum at x if it is either a local maximum or local minimum at x -/
def has_extremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  (∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x) ∨
  (∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≥ f x)

theorem extremum_derivative_zero_necessary_not_sufficient :
  (∀ f : ℝ → ℝ, Differentiable ℝ f → ∀ x : ℝ, has_extremum f x → deriv f x = 0) ∧
  (∃ f : ℝ → ℝ, Differentiable ℝ f ∧ ∃ x : ℝ, deriv f x = 0 ∧ ¬has_extremum f x) :=
sorry

end NUMINAMATH_CALUDE_extremum_derivative_zero_necessary_not_sufficient_l955_95579


namespace NUMINAMATH_CALUDE_jellybean_problem_l955_95542

theorem jellybean_problem (initial_bags : ℕ) (initial_average : ℕ) (average_increase : ℕ) :
  initial_bags = 34 →
  initial_average = 117 →
  average_increase = 7 →
  (initial_bags * initial_average + (initial_bags + 1) * average_increase) = 362 :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l955_95542


namespace NUMINAMATH_CALUDE_average_songs_in_remaining_sets_l955_95548

def bandRepertoire : ℕ := 30
def firstSetSongs : ℕ := 5
def secondSetSongs : ℕ := 7
def encoreSongs : ℕ := 2
def remainingSets : ℕ := 2

theorem average_songs_in_remaining_sets :
  (bandRepertoire - (firstSetSongs + secondSetSongs + encoreSongs)) / remainingSets = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_songs_in_remaining_sets_l955_95548


namespace NUMINAMATH_CALUDE_units_digit_of_n_l955_95567

theorem units_digit_of_n (m n : ℕ) : 
  m * n = 31^8 → 
  m % 10 = 7 → 
  n % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l955_95567


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l955_95509

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  (x - 1)^2 = 2 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2 := by
sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  x^2 - 6*x - 7 = 0 ↔ x = -1 ∨ x = 7 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l955_95509


namespace NUMINAMATH_CALUDE_even_function_sufficient_not_necessary_l955_95555

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def exists_symmetric_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = f (-x₀)

theorem even_function_sufficient_not_necessary :
  (∀ f : ℝ → ℝ, is_even_function f → exists_symmetric_point f) ∧
  ¬(∀ f : ℝ → ℝ, exists_symmetric_point f → is_even_function f) := by
  sorry

end NUMINAMATH_CALUDE_even_function_sufficient_not_necessary_l955_95555


namespace NUMINAMATH_CALUDE_jennys_score_is_14_total_questions_correct_l955_95592

/-- Represents a quiz with a specific scoring system -/
structure Quiz where
  totalQuestions : ℕ
  correctAnswers : ℕ
  incorrectAnswers : ℕ
  unansweredQuestions : ℕ
  correctPoints : ℚ
  incorrectPoints : ℚ

/-- Calculates the total score for a given quiz -/
def calculateScore (q : Quiz) : ℚ :=
  q.correctPoints * q.correctAnswers + q.incorrectPoints * q.incorrectAnswers

/-- Jenny's quiz results -/
def jennysQuiz : Quiz :=
  { totalQuestions := 25
    correctAnswers := 16
    incorrectAnswers := 4
    unansweredQuestions := 5
    correctPoints := 1
    incorrectPoints := -1/2 }

/-- Theorem stating that Jenny's quiz score is 14 -/
theorem jennys_score_is_14 : calculateScore jennysQuiz = 14 := by
  sorry

/-- Theorem verifying the total number of questions -/
theorem total_questions_correct :
  jennysQuiz.correctAnswers + jennysQuiz.incorrectAnswers + jennysQuiz.unansweredQuestions =
  jennysQuiz.totalQuestions := by
  sorry

end NUMINAMATH_CALUDE_jennys_score_is_14_total_questions_correct_l955_95592


namespace NUMINAMATH_CALUDE_sum_of_cubes_zero_l955_95525

theorem sum_of_cubes_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_sum_squares : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_zero_l955_95525


namespace NUMINAMATH_CALUDE_helens_oranges_l955_95589

/-- Helen's orange counting problem -/
theorem helens_oranges (initial : ℕ) (from_ann : ℕ) (to_sarah : ℕ) : 
  initial = 9 → from_ann = 29 → to_sarah = 14 → 
  initial + from_ann - to_sarah = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_helens_oranges_l955_95589


namespace NUMINAMATH_CALUDE_square_root_condition_l955_95550

theorem square_root_condition (x : ℝ) : 
  (∃ y : ℝ, y^2 = 3*x - 5) ↔ x ≥ 5/3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_condition_l955_95550


namespace NUMINAMATH_CALUDE_book_selection_theorem_l955_95516

theorem book_selection_theorem (chinese_books math_books sports_books : ℕ) 
  (h1 : chinese_books = 4) 
  (h2 : math_books = 5) 
  (h3 : sports_books = 6) : 
  (chinese_books + math_books + sports_books = 15) ∧ 
  (chinese_books * math_books * sports_books = 120) := by
  sorry

end NUMINAMATH_CALUDE_book_selection_theorem_l955_95516


namespace NUMINAMATH_CALUDE_line_through_points_l955_95540

-- Define the points
def P₁ : ℝ × ℝ := (3, -1)
def P₂ : ℝ × ℝ := (-2, 1)

-- Define the slope-intercept form
def slope_intercept (m b : ℝ) (x y : ℝ) : Prop :=
  y = m * x + b

-- Theorem statement
theorem line_through_points : 
  ∃ (m b : ℝ), m = -2/5 ∧ b = 1/5 ∧ 
  (slope_intercept m b P₁.1 P₁.2 ∧ slope_intercept m b P₂.1 P₂.2) :=
sorry

end NUMINAMATH_CALUDE_line_through_points_l955_95540


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l955_95527

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 12 → 
  ∀ x, x^2 - 16*x + 144 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l955_95527


namespace NUMINAMATH_CALUDE_lynn_in_fourth_car_l955_95553

-- Define the set of people
inductive Person : Type
| Trent : Person
| Jamie : Person
| Eden : Person
| Lynn : Person
| Mira : Person
| Cory : Person

-- Define the seating arrangement
def SeatingArrangement := Fin 6 → Person

-- Define the conditions of the seating arrangement
def ValidArrangement (s : SeatingArrangement) : Prop :=
  -- Trent is in the lead car
  s 0 = Person.Trent ∧
  -- Eden is directly behind Jamie
  (∃ i : Fin 5, s i = Person.Jamie ∧ s (i + 1) = Person.Eden) ∧
  -- Lynn sits ahead of Mira
  (∃ i j : Fin 6, i < j ∧ s i = Person.Lynn ∧ s j = Person.Mira) ∧
  -- Mira is not in the last car
  s 5 ≠ Person.Mira ∧
  -- At least two people sit between Cory and Lynn
  (∃ i j : Fin 6, |i - j| > 2 ∧ s i = Person.Cory ∧ s j = Person.Lynn)

-- The theorem to prove
theorem lynn_in_fourth_car (s : SeatingArrangement) :
  ValidArrangement s → s 3 = Person.Lynn :=
by sorry

end NUMINAMATH_CALUDE_lynn_in_fourth_car_l955_95553


namespace NUMINAMATH_CALUDE_unfolded_paper_has_eight_holes_l955_95524

/-- Represents a square piece of paper -/
structure SquarePaper where
  side : ℝ
  holes : Finset (ℝ × ℝ)

/-- Represents the state of the paper after folding and punching -/
structure FoldedPaper where
  original : SquarePaper
  center_hole : Bool
  upper_right_hole : Bool

/-- Counts the number of holes in the unfolded paper -/
def count_holes (folded : FoldedPaper) : ℕ :=
  let center_holes := if folded.center_hole then 4 else 0
  let corner_holes := if folded.upper_right_hole then 4 else 0
  center_holes + corner_holes

/-- Theorem stating that the unfolded paper has 8 holes -/
theorem unfolded_paper_has_eight_holes (paper : SquarePaper) :
  ∀ (folded : FoldedPaper),
    folded.original = paper →
    folded.center_hole = true →
    folded.upper_right_hole = true →
    count_holes folded = 8 :=
  sorry

end NUMINAMATH_CALUDE_unfolded_paper_has_eight_holes_l955_95524


namespace NUMINAMATH_CALUDE_solve_sqrt_equation_l955_95574

theorem solve_sqrt_equation (x : ℝ) :
  Real.sqrt ((2 / x) + 3) = 4 / 3 → x = -18 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_sqrt_equation_l955_95574


namespace NUMINAMATH_CALUDE_unique_rational_solution_l955_95585

theorem unique_rational_solution (x y z : ℚ) : 
  x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_rational_solution_l955_95585


namespace NUMINAMATH_CALUDE_road_trip_ratio_l955_95507

/-- Road trip problem -/
theorem road_trip_ratio : 
  ∀ (total michelle_dist katie_dist tracy_dist : ℕ),
  total = 1000 →
  michelle_dist = 294 →
  michelle_dist = 3 * katie_dist →
  tracy_dist = total - michelle_dist - katie_dist →
  (tracy_dist - 20) / michelle_dist = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_road_trip_ratio_l955_95507


namespace NUMINAMATH_CALUDE_data_transformation_theorem_l955_95528

variable {α : Type*} [LinearOrderedField α]

def average (data : Finset α) (f : α → α) : α :=
  (data.sum f) / data.card

def variance (data : Finset α) (f : α → α) (μ : α) : α :=
  (data.sum (fun x => (f x - μ) ^ 2)) / data.card

theorem data_transformation_theorem (data : Finset α) (f : α → α) :
  (average data (fun x => f x - 80) = 1.2) →
  (variance data (fun x => f x - 80) 1.2 = 4.4) →
  (average data f = 81.2) ∧ (variance data f 81.2 = 4.4) := by
  sorry

end NUMINAMATH_CALUDE_data_transformation_theorem_l955_95528


namespace NUMINAMATH_CALUDE_connor_date_cost_l955_95564

def movie_date_cost (ticket_price : ℚ) (combo_price : ℚ) (candy_price : ℚ) (cup_price : ℚ) : ℚ :=
  let discounted_ticket := ticket_price * (1/2)
  let tickets_total := ticket_price + discounted_ticket
  let candy_total := 2 * candy_price * (1 - 1/5)
  let cup_total := cup_price - 1
  tickets_total + combo_price + candy_total + cup_total

theorem connor_date_cost :
  movie_date_cost 14 11 2.5 5 = 40 := by
  sorry

end NUMINAMATH_CALUDE_connor_date_cost_l955_95564


namespace NUMINAMATH_CALUDE_second_discount_percentage_l955_95576

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) :
  original_price = 400 ∧
  first_discount = 30 ∧
  final_price = 224 →
  ∃ second_discount : ℝ,
    second_discount = 20 ∧
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) :=
by sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l955_95576


namespace NUMINAMATH_CALUDE_cheryl_material_usage_l955_95543

theorem cheryl_material_usage 
  (material1 : ℚ) 
  (material2 : ℚ) 
  (leftover : ℚ) 
  (h1 : material1 = 5 / 11)
  (h2 : material2 = 2 / 3)
  (h3 : leftover = 25 / 55) :
  material1 + material2 - leftover = 22 / 33 := by
  sorry

end NUMINAMATH_CALUDE_cheryl_material_usage_l955_95543


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l955_95517

theorem arithmetic_calculation : 8 / 2 - 3 - 12 + 3 * (5^2 - 4) = 52 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l955_95517


namespace NUMINAMATH_CALUDE_cost_price_calculation_l955_95505

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 1110 ∧ profit_percentage = 20 → 
  (selling_price / (1 + profit_percentage / 100) : ℝ) = 925 := by
sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l955_95505


namespace NUMINAMATH_CALUDE_task_completion_time_l955_95521

/-- The number of days needed for three people to complete the task -/
def three_people_days : ℕ := 3 * 7 + 3

/-- The number of people in the original scenario -/
def original_people : ℕ := 3

/-- The number of people in the new scenario -/
def new_people : ℕ := 4

/-- The time needed for four people to complete the task -/
def four_people_days : ℚ := 18

theorem task_completion_time :
  (three_people_days : ℚ) * original_people / new_people = four_people_days :=
sorry

end NUMINAMATH_CALUDE_task_completion_time_l955_95521


namespace NUMINAMATH_CALUDE_diamond_3_7_l955_95559

-- Define the star operation
def star (a b : ℕ) : ℕ := a^2 + 2*a*b + b^2

-- Define the diamond operation
def diamond (a b : ℕ) : ℕ := star a b - a*b

-- Theorem to prove
theorem diamond_3_7 : diamond 3 7 = 79 := by
  sorry

end NUMINAMATH_CALUDE_diamond_3_7_l955_95559


namespace NUMINAMATH_CALUDE_polynomial_division_quotient_l955_95596

theorem polynomial_division_quotient (x : ℝ) :
  (x^2 + 7*x + 17) * (x - 2) + 43 = x^3 + 5*x^2 + 3*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_quotient_l955_95596


namespace NUMINAMATH_CALUDE_remainder_seventeen_pow_sixtythree_mod_seven_l955_95581

theorem remainder_seventeen_pow_sixtythree_mod_seven :
  17^63 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_seventeen_pow_sixtythree_mod_seven_l955_95581


namespace NUMINAMATH_CALUDE_jill_minus_jake_equals_one_l955_95570

def peach_problem (jake steven jill : ℕ) : Prop :=
  (jake + 16 = steven) ∧ 
  (steven = jill + 15) ∧ 
  (jill = 12)

theorem jill_minus_jake_equals_one :
  ∀ jake steven jill : ℕ, peach_problem jake steven jill → jill - jake = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_jill_minus_jake_equals_one_l955_95570


namespace NUMINAMATH_CALUDE_candy_calculation_l955_95560

/-- Calculates the number of candy pieces Faye's sister gave her --/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

theorem candy_calculation (initial eaten final : ℕ) 
  (h1 : initial ≥ eaten) 
  (h2 : final ≥ initial - eaten) : 
  candy_from_sister initial eaten final = final - (initial - eaten) :=
by
  sorry

#eval candy_from_sister 47 25 62  -- Should output 40

end NUMINAMATH_CALUDE_candy_calculation_l955_95560


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l955_95546

theorem sqrt_difference_equality : Real.sqrt (49 + 121) - Real.sqrt (64 + 16) = Real.sqrt 170 - 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l955_95546


namespace NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l955_95506

theorem quadratic_always_positive_implies_a_greater_than_one (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) → a > 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_implies_a_greater_than_one_l955_95506


namespace NUMINAMATH_CALUDE_square_difference_l955_95523

theorem square_difference (a b : ℝ) :
  let A : ℝ := (5*a + 3*b)^2 - (5*a - 3*b)^2
  A = 60*a*b := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l955_95523


namespace NUMINAMATH_CALUDE_distance_focus_to_line_l955_95538

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point A (directrix intersection with x-axis)
def A : ℝ × ℝ := (-1, 0)

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * (x + 1)

-- State the theorem
theorem distance_focus_to_line :
  let d := Real.sqrt 3
  ∃ (x y : ℝ), parabola x y ∧ 
               line_l (A.1) (A.2) ∧
               (F.1 - x)^2 + (F.2 - y)^2 = d^2 :=
sorry

end NUMINAMATH_CALUDE_distance_focus_to_line_l955_95538


namespace NUMINAMATH_CALUDE_bible_length_l955_95518

/-- The number of pages in John's bible --/
def bible_pages : ℕ := sorry

/-- The number of hours John reads per day --/
def hours_per_day : ℕ := 2

/-- The number of pages John reads per hour --/
def pages_per_hour : ℕ := 50

/-- The number of weeks it takes John to read the entire bible --/
def weeks_to_read : ℕ := 4

/-- The number of days in a week --/
def days_per_week : ℕ := 7

theorem bible_length : bible_pages = 2800 := by sorry

end NUMINAMATH_CALUDE_bible_length_l955_95518


namespace NUMINAMATH_CALUDE_percentage_of_B_grades_l955_95599

def grading_scale : List (String × (Int × Int)) :=
  [("A", (94, 100)), ("B", (87, 93)), ("C", (78, 86)), ("D", (70, 77)), ("F", (0, 69))]

def scores : List Int := [93, 65, 88, 100, 72, 95, 82, 68, 79, 56, 87, 81, 74, 85, 91]

def is_grade (score : Int) (grade : String × (Int × Int)) : Bool :=
  let (_, (low, high)) := grade
  low ≤ score ∧ score ≤ high

def count_grade (scores : List Int) (grade : String × (Int × Int)) : Nat :=
  (scores.filter (fun score => is_grade score grade)).length

theorem percentage_of_B_grades :
  let b_grade := ("B", (87, 93))
  let total_students := scores.length
  let b_students := count_grade scores b_grade
  (b_students : Rat) / total_students * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_B_grades_l955_95599


namespace NUMINAMATH_CALUDE_smallest_n_with_gcd_conditions_l955_95583

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

theorem smallest_n_with_gcd_conditions :
  ∃ (n : ℕ), n > 200 ∧ 
  Nat.gcd 70 (n + 150) = 35 ∧ 
  Nat.gcd (n + 70) 150 = 75 ∧
  ∀ (m : ℕ), m > 200 → 
    Nat.gcd 70 (m + 150) = 35 → 
    Nat.gcd (m + 70) 150 = 75 → 
    n ≤ m ∧
  n = 305 ∧
  digit_sum n = 8 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_with_gcd_conditions_l955_95583


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l955_95578

universe u

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}

theorem complement_of_M_in_U :
  (U \ M) = {3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l955_95578


namespace NUMINAMATH_CALUDE_sqrt_identity_l955_95557

theorem sqrt_identity (t : ℝ) : 
  Real.sqrt (t^6 + t^4 + t^2) = |t| * Real.sqrt (t^4 + t^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_identity_l955_95557


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_l955_95508

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0) 
  (sum_squares_one : a^2 + b^2 + c^2 = 1) : 
  a^4 + b^4 + c^4 = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_l955_95508
