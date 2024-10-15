import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_roots_and_constant_term_l4105_410544

def polynomial (a b c d : ℤ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_roots_and_constant_term 
  (a b c d : ℤ) 
  (h1 : ∀ x : ℝ, polynomial a b c d x = 0 → (∃ n : ℕ, x = -↑n))
  (h2 : a + b + c + d = 2009) :
  d = 528 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_and_constant_term_l4105_410544


namespace NUMINAMATH_CALUDE_power_of_two_starts_with_1968_l4105_410539

-- Define the conditions
def m : ℕ := 3^2
def k : ℕ := 2^3

-- Define a function to check if a number starts with 1968
def starts_with_1968 (x : ℕ) : Prop :=
  ∃ y : ℕ, 1968 * 10^y ≤ x ∧ x < 1969 * 10^y

-- State the theorem
theorem power_of_two_starts_with_1968 :
  ∃ n : ℕ, n > 2^k ∧ starts_with_1968 (2^n) ∧
  ∀ m : ℕ, m < n → ¬starts_with_1968 (2^m) :=
sorry

end NUMINAMATH_CALUDE_power_of_two_starts_with_1968_l4105_410539


namespace NUMINAMATH_CALUDE_parking_solution_is_correct_l4105_410599

/-- Represents the parking lot problem. -/
structure ParkingLot where
  total_cars : ℕ
  total_fee : ℕ
  medium_fee : ℕ
  small_fee : ℕ

/-- Represents the solution to the parking lot problem. -/
structure ParkingSolution where
  medium_cars : ℕ
  small_cars : ℕ

/-- Checks if a given solution satisfies the parking lot conditions. -/
def is_valid_solution (p : ParkingLot) (s : ParkingSolution) : Prop :=
  s.medium_cars + s.small_cars = p.total_cars ∧
  s.medium_cars * p.medium_fee + s.small_cars * p.small_fee = p.total_fee

/-- The parking lot problem instance. -/
def parking_problem : ParkingLot :=
  { total_cars := 30
  , total_fee := 324
  , medium_fee := 15
  , small_fee := 8 }

/-- The proposed solution to the parking lot problem. -/
def parking_solution : ParkingSolution :=
  { medium_cars := 12
  , small_cars := 18 }

/-- Theorem stating that the proposed solution is correct for the given problem. -/
theorem parking_solution_is_correct :
  is_valid_solution parking_problem parking_solution := by
  sorry

end NUMINAMATH_CALUDE_parking_solution_is_correct_l4105_410599


namespace NUMINAMATH_CALUDE_area_of_intersection_is_point_eight_l4105_410568

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space with slope-intercept form y = mx + b -/
structure Line2D where
  m : ℝ
  b : ℝ

/-- Calculates the area of intersection between two triangles -/
noncomputable def areaOfIntersection (a b c d e : Point2D) (lineAC lineDE : Line2D) : ℝ :=
  sorry

/-- Theorem: The area of intersection between two specific triangles is 0.8 square units -/
theorem area_of_intersection_is_point_eight :
  let a : Point2D := ⟨1, 4⟩
  let b : Point2D := ⟨0, 0⟩
  let c : Point2D := ⟨2, 0⟩
  let d : Point2D := ⟨0, 1⟩
  let e : Point2D := ⟨4, 0⟩
  let lineAC : Line2D := ⟨-4, 8⟩
  let lineDE : Line2D := ⟨-1/4, 1⟩
  areaOfIntersection a b c d e lineAC lineDE = 0.8 :=
by
  sorry

end NUMINAMATH_CALUDE_area_of_intersection_is_point_eight_l4105_410568


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_57_l4105_410542

theorem smallest_n_divisible_by_57 :
  ∃ (n : ℕ), n > 0 ∧ 57 ∣ (7^n + 2*n) ∧ ∀ (m : ℕ), m > 0 ∧ 57 ∣ (7^m + 2*m) → n ≤ m :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_57_l4105_410542


namespace NUMINAMATH_CALUDE_six_digit_permutations_eq_60_l4105_410551

/-- The number of different positive, six-digit integers that can be formed using the digits 1, 1, 3, 3, 3, and 6 -/
def six_digit_permutations : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 3)

/-- Theorem stating that the number of different positive, six-digit integers
    that can be formed using the digits 1, 1, 3, 3, 3, and 6 is equal to 60 -/
theorem six_digit_permutations_eq_60 : six_digit_permutations = 60 := by
  sorry

end NUMINAMATH_CALUDE_six_digit_permutations_eq_60_l4105_410551


namespace NUMINAMATH_CALUDE_non_honda_red_percentage_is_51_25_l4105_410519

/-- Represents the car population in Chennai -/
structure CarPopulation where
  total : Nat
  honda : Nat
  toyota : Nat
  ford : Nat
  other : Nat
  honda_red_ratio : Rat
  toyota_red_ratio : Rat
  ford_red_ratio : Rat
  other_red_ratio : Rat

/-- Calculates the percentage of non-Honda cars that are red -/
def non_honda_red_percentage (pop : CarPopulation) : Rat :=
  let non_honda_total := pop.toyota + pop.ford + pop.other
  let non_honda_red := pop.toyota * pop.toyota_red_ratio + 
                       pop.ford * pop.ford_red_ratio + 
                       pop.other * pop.other_red_ratio
  (non_honda_red / non_honda_total) * 100

/-- The main theorem stating that the percentage of non-Honda cars that are red is 51.25% -/
theorem non_honda_red_percentage_is_51_25 (pop : CarPopulation) 
  (h1 : pop.total = 900)
  (h2 : pop.honda = 500)
  (h3 : pop.toyota = 200)
  (h4 : pop.ford = 150)
  (h5 : pop.other = 50)
  (h6 : pop.honda_red_ratio = 9/10)
  (h7 : pop.toyota_red_ratio = 3/4)
  (h8 : pop.ford_red_ratio = 3/10)
  (h9 : pop.other_red_ratio = 2/5) :
  non_honda_red_percentage pop = 51.25 := by
  sorry

#eval non_honda_red_percentage {
  total := 900,
  honda := 500,
  toyota := 200,
  ford := 150,
  other := 50,
  honda_red_ratio := 9/10,
  toyota_red_ratio := 3/4,
  ford_red_ratio := 3/10,
  other_red_ratio := 2/5
}

end NUMINAMATH_CALUDE_non_honda_red_percentage_is_51_25_l4105_410519


namespace NUMINAMATH_CALUDE_range_of_a_l4105_410529

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 + 3)^2 ≤ 4}
def B (a : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - a)^2 ≤ 1/4}

-- State the theorem
theorem range_of_a (a : ℝ) :
  (A ∩ B a = B a) →
  (-3 - Real.sqrt 5 / 2 ≤ a ∧ a ≤ -3 + Real.sqrt 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l4105_410529


namespace NUMINAMATH_CALUDE_rectangle_point_s_l4105_410508

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of a rectangle formed by four points -/
def isRectangle (p q r s : Point2D) : Prop :=
  (p.x = q.x ∧ r.x = s.x ∧ p.y = s.y ∧ q.y = r.y) ∨
  (p.x = s.x ∧ q.x = r.x ∧ p.y = q.y ∧ r.y = s.y)

/-- The theorem stating that given P, Q, and R, the point S forms a rectangle -/
theorem rectangle_point_s (p q r : Point2D)
  (h_p : p = ⟨3, -2⟩)
  (h_q : q = ⟨3, 1⟩)
  (h_r : r = ⟨7, 1⟩) :
  ∃ s : Point2D, s = ⟨7, -2⟩ ∧ isRectangle p q r s :=
sorry

end NUMINAMATH_CALUDE_rectangle_point_s_l4105_410508


namespace NUMINAMATH_CALUDE_population_growth_rate_l4105_410515

/-- Calculates the average percent increase per year given initial and final populations over a decade. -/
def average_percent_increase (initial_population final_population : ℕ) : ℚ :=
  let total_increase : ℕ := final_population - initial_population
  let average_annual_increase : ℚ := (total_increase : ℚ) / 10
  (average_annual_increase / initial_population) * 100

/-- Theorem stating that the average percent increase per year for the given population change is 7%. -/
theorem population_growth_rate : 
  average_percent_increase 175000 297500 = 7 := by
  sorry

#eval average_percent_increase 175000 297500

end NUMINAMATH_CALUDE_population_growth_rate_l4105_410515


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_l4105_410511

/-- 
Given an arithmetic sequence with first term a₁ = 1 and common difference d = 3,
prove that 2011 is the 671st term of this sequence.
-/
theorem arithmetic_sequence_2011 : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 → 
    (∀ n, a (n + 1) - a n = 3) → 
    a 671 = 2011 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_l4105_410511


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l4105_410545

theorem sum_of_roots_eq_fourteen : 
  let f : ℝ → ℝ := λ x => (x - 7)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_fourteen_l4105_410545


namespace NUMINAMATH_CALUDE_two_special_right_triangles_l4105_410516

/-- A right-angled triangle with integer sides where the area equals the perimeter -/
structure SpecialRightTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  h1 : a^2 + b^2 = c^2  -- Pythagorean theorem
  h2 : a * b = 2 * (a + b + c)  -- Area equals perimeter

/-- The set of all SpecialRightTriangles -/
def specialRightTriangles : Set SpecialRightTriangle :=
  {t : SpecialRightTriangle | True}

theorem two_special_right_triangles :
  ∃ (t1 t2 : SpecialRightTriangle),
    specialRightTriangles = {t1, t2} ∧
    ((t1.a = 5 ∧ t1.b = 12 ∧ t1.c = 13) ∨ (t1.a = 12 ∧ t1.b = 5 ∧ t1.c = 13)) ∧
    ((t2.a = 6 ∧ t2.b = 8 ∧ t2.c = 10) ∨ (t2.a = 8 ∧ t2.b = 6 ∧ t2.c = 10)) :=
  sorry

#check two_special_right_triangles

end NUMINAMATH_CALUDE_two_special_right_triangles_l4105_410516


namespace NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l4105_410583

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the carton -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 48, height := 60 }

/-- The dimensions of a soap box -/
def soapBoxDimensions : BoxDimensions :=
  { length := 8, width := 6, height := 5 }

/-- Theorem stating the maximum number of soap boxes that can fit in the carton -/
theorem max_soap_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume soapBoxDimensions) = 300 := by
  sorry

end NUMINAMATH_CALUDE_max_soap_boxes_in_carton_l4105_410583


namespace NUMINAMATH_CALUDE_leila_order_cost_l4105_410522

/-- Calculates the total cost of Leila's cake order --/
def total_cost (chocolate_quantity : ℕ) (chocolate_price : ℕ) 
                (strawberry_quantity : ℕ) (strawberry_price : ℕ) : ℕ :=
  chocolate_quantity * chocolate_price + strawberry_quantity * strawberry_price

/-- Proves that the total cost of Leila's order is $168 --/
theorem leila_order_cost : 
  total_cost 3 12 6 22 = 168 := by
  sorry

#eval total_cost 3 12 6 22

end NUMINAMATH_CALUDE_leila_order_cost_l4105_410522


namespace NUMINAMATH_CALUDE_zoo_count_l4105_410590

theorem zoo_count (zebras camels monkeys giraffes : ℕ) : 
  camels = zebras / 2 →
  monkeys = 4 * camels →
  giraffes = 2 →
  monkeys = giraffes + 22 →
  zebras = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_count_l4105_410590


namespace NUMINAMATH_CALUDE_decimal_arithmetic_l4105_410543

theorem decimal_arithmetic : 
  (∃ x : ℝ, x = 3.92 + 0.4 ∧ x = 3.96) ∧
  (∃ y : ℝ, y = 4.93 - 1.5 ∧ y = 3.43) := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_l4105_410543


namespace NUMINAMATH_CALUDE_smallest_multiple_of_9_and_6_l4105_410585

def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_multiple_of_9_and_6 : 
  (∀ n : ℕ, n > 0 ∧ is_multiple 9 n ∧ is_multiple 6 n → n ≥ 18) ∧
  (18 > 0 ∧ is_multiple 9 18 ∧ is_multiple 6 18) :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_9_and_6_l4105_410585


namespace NUMINAMATH_CALUDE_max_homework_time_l4105_410573

/-- Represents the time spent on each subject in minutes -/
structure HomeworkTime where
  biology : ℕ
  history : ℕ
  geography : ℕ

/-- Calculates the total time spent on homework given the conditions -/
def total_homework_time (t : HomeworkTime) : ℕ :=
  t.biology + t.history + t.geography

/-- Theorem stating that Max's total homework time is 180 minutes -/
theorem max_homework_time :
  ∀ t : HomeworkTime,
  t.biology = 20 ∧
  t.history = 2 * t.biology ∧
  t.geography = 3 * t.history →
  total_homework_time t = 180 :=
by
  sorry

#check max_homework_time

end NUMINAMATH_CALUDE_max_homework_time_l4105_410573


namespace NUMINAMATH_CALUDE_total_scissors_l4105_410567

/-- The total number of scissors after adding more is equal to the sum of the initial number of scissors and the number of scissors added. -/
theorem total_scissors (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

end NUMINAMATH_CALUDE_total_scissors_l4105_410567


namespace NUMINAMATH_CALUDE_sqrt_of_four_l4105_410560

theorem sqrt_of_four : Real.sqrt 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_four_l4105_410560


namespace NUMINAMATH_CALUDE_special_numbers_l4105_410586

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def satisfies_conditions (n : ℕ) : Prop :=
  is_four_digit n ∧ n % 11 = 0 ∧ digit_sum n = 11

theorem special_numbers :
  {n : ℕ | satisfies_conditions n} =
  {2090, 3080, 4070, 5060, 6050, 7040, 8030, 9020} := by sorry

end NUMINAMATH_CALUDE_special_numbers_l4105_410586


namespace NUMINAMATH_CALUDE_line_segment_translation_l4105_410584

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation vector -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Applies a translation to a point -/
def translatePoint (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem line_segment_translation (A B : Point) (A_new : Point) :
  A = { x := 1, y := 2 } →
  B = { x := 7, y := 5 } →
  A_new = { x := -6, y := -3 } →
  let t : Translation := { dx := A_new.x - A.x, dy := A_new.y - A.y }
  translatePoint B t = { x := 0, y := 0 } := by sorry

end NUMINAMATH_CALUDE_line_segment_translation_l4105_410584


namespace NUMINAMATH_CALUDE_totalDispatchPlansIs36_l4105_410535

/-- The number of people to choose from -/
def totalPeople : Nat := 5

/-- The number of tasks to be assigned -/
def totalTasks : Nat := 4

/-- The number of people who can only do certain tasks -/
def restrictedPeople : Nat := 2

/-- The number of tasks that restricted people can do -/
def restrictedTasks : Nat := 2

/-- The number of people who can do any task -/
def unrestrictedPeople : Nat := totalPeople - restrictedPeople

/-- Calculate the number of ways to select and arrange k items from n items -/
def arrangementNumber (n k : Nat) : Nat := sorry

/-- Calculate the number of ways to select k items from n items -/
def combinationNumber (n k : Nat) : Nat := sorry

/-- The total number of different dispatch plans -/
def totalDispatchPlans : Nat :=
  combinationNumber restrictedPeople 1 * combinationNumber restrictedTasks 1 * 
    arrangementNumber unrestrictedPeople 3 +
  arrangementNumber restrictedPeople 2 * arrangementNumber unrestrictedPeople 2

/-- Theorem stating that the total number of different dispatch plans is 36 -/
theorem totalDispatchPlansIs36 : totalDispatchPlans = 36 := by sorry

end NUMINAMATH_CALUDE_totalDispatchPlansIs36_l4105_410535


namespace NUMINAMATH_CALUDE_riverbend_prep_distance_l4105_410578

/-- Represents a relay race team -/
structure RelayTeam where
  name : String
  members : Nat
  raceLength : Nat

/-- Calculates the total distance covered by a relay team -/
def totalDistance (team : RelayTeam) : Nat :=
  team.members * team.raceLength

/-- Theorem stating that the total distance covered by Riverbend Prep is 1500 meters -/
theorem riverbend_prep_distance :
  let riverbendPrep : RelayTeam := ⟨"Riverbend Prep", 6, 250⟩
  totalDistance riverbendPrep = 1500 := by sorry

end NUMINAMATH_CALUDE_riverbend_prep_distance_l4105_410578


namespace NUMINAMATH_CALUDE_geometric_sequence_11th_term_l4105_410561

/-- Represents a geometric sequence -/
structure GeometricSequence where
  -- The sequence function
  a : ℕ → ℝ
  -- The common ratio
  r : ℝ
  -- The geometric sequence property
  geom_prop : ∀ n : ℕ, a (n + 1) = a n * r

/-- Theorem: In a geometric sequence where the 5th term is -2 and the 8th term is -54, the 11th term is -1458 -/
theorem geometric_sequence_11th_term
  (seq : GeometricSequence)
  (h5 : seq.a 5 = -2)
  (h8 : seq.a 8 = -54) :
  seq.a 11 = -1458 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_11th_term_l4105_410561


namespace NUMINAMATH_CALUDE_gcd_228_1995_l4105_410509

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l4105_410509


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l4105_410538

/-- Represents the outcome of drawing balls from a bag -/
structure BallDrawing where
  totalBalls : Nat
  redBalls : Nat
  blackBalls : Nat
  drawCount : Nat

/-- Calculates the expectation for drawing with replacement -/
def expectationWithReplacement (bd : BallDrawing) : Rat :=
  bd.drawCount * (bd.redBalls : Rat) / bd.totalBalls

/-- Calculates the variance for drawing with replacement -/
def varianceWithReplacement (bd : BallDrawing) : Rat :=
  bd.drawCount * (bd.redBalls : Rat) / bd.totalBalls * (1 - (bd.redBalls : Rat) / bd.totalBalls)

/-- Calculates the expectation for drawing without replacement -/
noncomputable def expectationWithoutReplacement (bd : BallDrawing) : Rat :=
  sorry

/-- Calculates the variance for drawing without replacement -/
noncomputable def varianceWithoutReplacement (bd : BallDrawing) : Rat :=
  sorry

theorem ball_drawing_theorem (bd : BallDrawing) 
    (h1 : bd.totalBalls = 10)
    (h2 : bd.redBalls = 4)
    (h3 : bd.blackBalls = 6)
    (h4 : bd.drawCount = 3) :
    expectationWithReplacement bd = expectationWithoutReplacement bd ∧
    varianceWithReplacement bd > varianceWithoutReplacement bd :=
  by sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l4105_410538


namespace NUMINAMATH_CALUDE_tangent_line_at_one_one_l4105_410562

/-- The equation of the tangent line to y = x^2 at (1, 1) is 2x - y - 1 = 0 -/
theorem tangent_line_at_one_one :
  let f : ℝ → ℝ := λ x ↦ x^2
  let point : ℝ × ℝ := (1, 1)
  let tangent_line : ℝ → ℝ → Prop := λ x y ↦ 2*x - y - 1 = 0
  (∀ x, HasDerivAt f (2*x) x) →
  tangent_line point.1 point.2 ∧
  ∀ x y, tangent_line x y ↔ y - point.2 = (2 * point.1) * (x - point.1) :=
by
  sorry


end NUMINAMATH_CALUDE_tangent_line_at_one_one_l4105_410562


namespace NUMINAMATH_CALUDE_remaining_money_calculation_l4105_410589

/-- Calculates the remaining money after expenses given a salary and expense ratios -/
def remaining_money (salary : ℚ) (food_ratio : ℚ) (rent_ratio : ℚ) (clothes_ratio : ℚ) : ℚ :=
  salary * (1 - (food_ratio + rent_ratio + clothes_ratio))

/-- Theorem stating that given the specific salary and expense ratios, the remaining money is 17000 -/
theorem remaining_money_calculation :
  remaining_money 170000 (1/5) (1/10) (3/5) = 17000 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_calculation_l4105_410589


namespace NUMINAMATH_CALUDE_chicken_cost_proof_l4105_410528

def total_cost : ℝ := 16
def beef_pounds : ℝ := 3
def beef_price_per_pound : ℝ := 4
def oil_price : ℝ := 1
def people_paying : ℕ := 3
def individual_payment : ℝ := 1

theorem chicken_cost_proof :
  total_cost - (beef_pounds * beef_price_per_pound + oil_price) = people_paying * individual_payment := by
  sorry

end NUMINAMATH_CALUDE_chicken_cost_proof_l4105_410528


namespace NUMINAMATH_CALUDE_squares_concurrency_l4105_410500

-- Define the vertices of the squares as complex numbers
variable (zA zB zC zD zA₁ zB₁ zC₁ zD₁ : ℂ)

-- Define the condition for equally oriented squares
def equally_oriented (zA zB zC zD zA₁ zB₁ zC₁ zD₁ : ℂ) : Prop :=
  ∃ (w t : ℂ), Complex.abs w = 1 ∧
    zA₁ = w * zA + t ∧
    zB₁ = w * zB + t ∧
    zC₁ = w * zC + t ∧
    zD₁ = w * zD + t

-- Define the concurrency condition
def concurrent (zA zB zC zD zA₁ zB₁ zC₁ zD₁ : ℂ) : Prop :=
  ∃ (P : ℂ),
    (zA₁ - zA) / (zB₁ - zB) = (zA₁ - zA) / (zC₁ - zC) ∧
    (zA₁ - zA) / (zB₁ - zB) = (zA₁ - zA) / (zD₁ - zD)

-- State the theorem
theorem squares_concurrency
  (h : equally_oriented zA zB zC zD zA₁ zB₁ zC₁ zD₁) :
  concurrent zA zB zC zD zA₁ zB₁ zC₁ zD₁ :=
by sorry

end NUMINAMATH_CALUDE_squares_concurrency_l4105_410500


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l4105_410532

theorem arithmetic_calculations :
  ((-1) + (-6) - (-4) + 0 = -3) ∧
  (24 * (-1/4) / (-3/2) = 4) := by
sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l4105_410532


namespace NUMINAMATH_CALUDE_total_pay_is_330_l4105_410527

/-- The total weekly pay for two employees, where one earns 120% of the other -/
def total_weekly_pay (y_pay : ℝ) : ℝ :=
  let x_pay := 1.2 * y_pay
  x_pay + y_pay

/-- Proof that the total weekly pay for two employees is 330 when one earns 150 and the other earns 120% of that -/
theorem total_pay_is_330 : total_weekly_pay 150 = 330 := by
  sorry

#eval total_weekly_pay 150

end NUMINAMATH_CALUDE_total_pay_is_330_l4105_410527


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_314_l4105_410598

/-- The perfect-square integer closest to 314 is 324. -/
theorem closest_perfect_square_to_314 : 
  ∀ n : ℕ, n ≠ 324 → n * n ≠ 0 → |314 - (324 : ℤ)| ≤ |314 - (n * n : ℤ)| := by
  sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_314_l4105_410598


namespace NUMINAMATH_CALUDE_dolls_ratio_l4105_410506

theorem dolls_ratio (R S G : ℕ) : 
  S = G + 2 →
  G = 50 →
  R + S + G = 258 →
  R / S = 3 := by
sorry

end NUMINAMATH_CALUDE_dolls_ratio_l4105_410506


namespace NUMINAMATH_CALUDE_total_earnings_proof_l4105_410597

def lauryn_earnings : ℝ := 2000
def aurelia_percentage : ℝ := 0.7

theorem total_earnings_proof :
  let aurelia_earnings := lauryn_earnings * aurelia_percentage
  lauryn_earnings + aurelia_earnings = 3400 :=
by sorry

end NUMINAMATH_CALUDE_total_earnings_proof_l4105_410597


namespace NUMINAMATH_CALUDE_cos_3theta_l4105_410537

theorem cos_3theta (θ : ℝ) : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 2) / 4 → Complex.cos (3 * θ) = 9 / 64 := by
  sorry

end NUMINAMATH_CALUDE_cos_3theta_l4105_410537


namespace NUMINAMATH_CALUDE_parabola_intersection_l4105_410517

theorem parabola_intersection (x₁ x₂ : ℝ) (m : ℝ) : 
  (∀ x y, x^2 = 4*y → ∃ k, y = k*x + m) →  -- Line passes through (0, m) and intersects parabola
  x₁ * x₂ = -4 →                           -- Product of x-coordinates is -4
  m = 1 :=                                 -- Conclusion: m must be 1
by sorry

end NUMINAMATH_CALUDE_parabola_intersection_l4105_410517


namespace NUMINAMATH_CALUDE_det_matrix1_det_matrix2_l4105_410570

-- Define the determinant function for 2x2 matrices
def det2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem for the first matrix
theorem det_matrix1 : det2 2 5 (-3) (-4) = 7 := by sorry

-- Theorem for the second matrix
theorem det_matrix2 (a b : ℝ) : det2 (a^2) (a*b) (a*b) (b^2) = 0 := by sorry

end NUMINAMATH_CALUDE_det_matrix1_det_matrix2_l4105_410570


namespace NUMINAMATH_CALUDE_A_intersect_B_is_empty_l4105_410512

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem A_intersect_B_is_empty : A ∩ B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_A_intersect_B_is_empty_l4105_410512


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l4105_410575

theorem right_triangle_shorter_leg : 
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 16 :=          -- The shorter leg is 16
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l4105_410575


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l4105_410592

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 42

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 70

theorem fgh_supermarkets_count :
  (us_supermarkets + canada_supermarkets = total_supermarkets) ∧
  (us_supermarkets = canada_supermarkets + 14) :=
by sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l4105_410592


namespace NUMINAMATH_CALUDE_factorization_equality_l4105_410518

theorem factorization_equality (x : ℝ) : 3 * x * (x - 5) + 4 * (x - 5) = (3 * x + 4) * (x - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l4105_410518


namespace NUMINAMATH_CALUDE_trig_identities_l4105_410524

/-- Given that sin(3π + α) = 2sin(3π/2 + α), prove two trigonometric identities. -/
theorem trig_identities (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin ((3 * Real.pi) / 2 + α)) : 
  (((2 * Real.sin α - 3 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α)) = 7 / 17) ∧ 
  ((Real.sin α)^2 + Real.sin (2 * α) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l4105_410524


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4105_410579

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  ((4 - 6*i) / (4 + 6*i) + (4 + 6*i) / (4 - 6*i)) = (-10 : ℚ) / 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4105_410579


namespace NUMINAMATH_CALUDE_polynomial_product_expansion_l4105_410504

theorem polynomial_product_expansion (x : ℝ) :
  (x^2 + 3*x - 4) * (x^2 - 5*x + 6) = x^4 - 2*x^3 - 13*x^2 + 38*x - 24 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_expansion_l4105_410504


namespace NUMINAMATH_CALUDE_cricket_bat_cost_price_l4105_410505

theorem cricket_bat_cost_price (profit_a_to_b : ℝ) (profit_b_to_c : ℝ) (price_c : ℝ) :
  profit_a_to_b = 0.20 →
  profit_b_to_c = 0.25 →
  price_c = 228 →
  ∃ (cost_price_a : ℝ),
    cost_price_a * (1 + profit_a_to_b) * (1 + profit_b_to_c) = price_c ∧
    cost_price_a = 152 :=
by sorry

end NUMINAMATH_CALUDE_cricket_bat_cost_price_l4105_410505


namespace NUMINAMATH_CALUDE_sequences_count_l4105_410593

/-- The number of students in the class -/
def num_students : ℕ := 15

/-- The number of class meetings per week -/
def meetings_per_week : ℕ := 3

/-- The number of different sequences of selecting 3 students from a group of 15 students,
    where each student can be selected at most once -/
def num_sequences : ℕ :=
  num_students * (num_students - 1) * (num_students - 2)

theorem sequences_count :
  num_sequences = 2730 := by
  sorry

end NUMINAMATH_CALUDE_sequences_count_l4105_410593


namespace NUMINAMATH_CALUDE_machine_value_calculation_l4105_410596

theorem machine_value_calculation (initial_value : ℝ) : 
  initial_value * (0.75 ^ 2) = 4000 → initial_value = 7111.11111111111 := by
  sorry

end NUMINAMATH_CALUDE_machine_value_calculation_l4105_410596


namespace NUMINAMATH_CALUDE_scientific_notation_34_million_l4105_410534

theorem scientific_notation_34_million :
  (34 : ℝ) * 1000000 = 3.4 * (10 : ℝ) ^ 7 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_34_million_l4105_410534


namespace NUMINAMATH_CALUDE_y_derivative_l4105_410533

noncomputable def y (x : ℝ) : ℝ := Real.sin x + Real.exp x * Real.cos x

theorem y_derivative (x : ℝ) : 
  deriv y x = (1 + Real.exp x) * Real.cos x - Real.exp x * Real.sin x :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l4105_410533


namespace NUMINAMATH_CALUDE_min_sum_of_squares_of_roots_l4105_410502

theorem min_sum_of_squares_of_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁^2 - 2*m*x₁ + (m^2 + 2*m + 3) = 0 →
  x₂^2 - 2*m*x₂ + (m^2 + 2*m + 3) = 0 →
  x₁ ≠ x₂ →
  ∃ (k : ℝ), k = x₁^2 + x₂^2 ∧ k ≥ 9/2 ∧ 
  (∀ (m' : ℝ) (y₁ y₂ : ℝ), 
    y₁^2 - 2*m'*y₁ + (m'^2 + 2*m' + 3) = 0 →
    y₂^2 - 2*m'*y₂ + (m'^2 + 2*m' + 3) = 0 →
    y₁ ≠ y₂ →
    y₁^2 + y₂^2 ≥ 9/2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_of_roots_l4105_410502


namespace NUMINAMATH_CALUDE_park_area_ratio_l4105_410577

theorem park_area_ratio (s : ℝ) (h : s > 0) : 
  (s^2) / ((3*s)^2) = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_park_area_ratio_l4105_410577


namespace NUMINAMATH_CALUDE_max_value_of_function_l4105_410591

theorem max_value_of_function (a : ℝ) (ha : 0 < a ∧ a < 1) :
  ∀ x y : ℝ, |x| + |y| ≤ 1 → (∀ x' y' : ℝ, |x'| + |y'| ≤ 1 → a * x + y ≤ a * x' + y') →
  a * x + y = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l4105_410591


namespace NUMINAMATH_CALUDE_girls_average_height_l4105_410580

/-- Calculates the average height of female students in a class -/
def average_height_girls (total_students : ℕ) (boys : ℕ) (avg_height_all : ℚ) (avg_height_boys : ℚ) : ℚ :=
  let girls := total_students - boys
  let total_height := (total_students : ℚ) * avg_height_all
  let boys_height := (boys : ℚ) * avg_height_boys
  let girls_height := total_height - boys_height
  girls_height / (girls : ℚ)

/-- Theorem stating the average height of girls in the class -/
theorem girls_average_height :
  average_height_girls 30 18 140 144 = 134 := by
  sorry

end NUMINAMATH_CALUDE_girls_average_height_l4105_410580


namespace NUMINAMATH_CALUDE_journey_distance_l4105_410510

theorem journey_distance (total_time : Real) (bike_speed : Real) (walk_speed : Real) 
  (h1 : total_time = 56 / 60) -- 56 minutes converted to hours
  (h2 : bike_speed = 20)
  (h3 : walk_speed = 4) :
  let total_distance := (total_time * bike_speed * walk_speed) / (1/3 * bike_speed + 2/3 * walk_speed)
  let walk_distance := 1/3 * total_distance
  walk_distance = 2.7 := by sorry

end NUMINAMATH_CALUDE_journey_distance_l4105_410510


namespace NUMINAMATH_CALUDE_subtraction_of_mixed_numbers_l4105_410520

theorem subtraction_of_mixed_numbers : (2 + 5/6) - (1 + 1/3) = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_of_mixed_numbers_l4105_410520


namespace NUMINAMATH_CALUDE_system_solution_l4105_410521

noncomputable def solve_system (x : Fin 12 → ℚ) : Prop :=
  x 0 + 12 * x 1 = 15 ∧
  x 0 - 12 * x 1 + 11 * x 2 = 2 ∧
  x 0 - 11 * x 2 + 10 * x 3 = 2 ∧
  x 0 - 10 * x 3 + 9 * x 4 = 2 ∧
  x 0 - 9 * x 4 + 8 * x 5 = 2 ∧
  x 0 - 8 * x 5 + 7 * x 6 = 2 ∧
  x 0 - 7 * x 6 + 6 * x 7 = 2 ∧
  x 0 - 6 * x 7 + 5 * x 8 = 2 ∧
  x 0 - 5 * x 8 + 4 * x 9 = 2 ∧
  x 0 - 4 * x 9 + 3 * x 10 = 2 ∧
  x 0 - 3 * x 10 + 2 * x 11 = 2 ∧
  x 0 - 2 * x 11 = 2

theorem system_solution :
  ∃! x : Fin 12 → ℚ, solve_system x ∧
    x 0 = 37/12 ∧ x 1 = 143/144 ∧ x 2 = 65/66 ∧ x 3 = 39/40 ∧
    x 4 = 26/27 ∧ x 5 = 91/96 ∧ x 6 = 13/14 ∧ x 7 = 65/72 ∧
    x 8 = 13/15 ∧ x 9 = 13/16 ∧ x 10 = 13/18 ∧ x 11 = 13/24 :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l4105_410521


namespace NUMINAMATH_CALUDE_december_sales_fraction_l4105_410587

theorem december_sales_fraction (average_sales : ℝ) (h : average_sales > 0) :
  let january_to_november_sales := 11 * average_sales
  let december_sales := 5 * average_sales
  let total_annual_sales := january_to_november_sales + december_sales
  december_sales / total_annual_sales = 5 / 16 := by
sorry

end NUMINAMATH_CALUDE_december_sales_fraction_l4105_410587


namespace NUMINAMATH_CALUDE_tino_jellybeans_l4105_410503

/-- Proves that Tino has 34 jellybeans given the conditions -/
theorem tino_jellybeans (arnold_jellybeans : ℕ) (lee_jellybeans : ℕ) (tino_jellybeans : ℕ)
  (h1 : arnold_jellybeans = 5)
  (h2 : arnold_jellybeans * 2 = lee_jellybeans)
  (h3 : tino_jellybeans = lee_jellybeans + 24) :
  tino_jellybeans = 34 := by
  sorry

end NUMINAMATH_CALUDE_tino_jellybeans_l4105_410503


namespace NUMINAMATH_CALUDE_circle_diameter_from_inscribed_triangles_l4105_410556

theorem circle_diameter_from_inscribed_triangles
  (triangle_a_side1 triangle_a_side2 triangle_a_hypotenuse : ℝ)
  (triangle_b_side1 triangle_b_side2 triangle_b_hypotenuse : ℝ)
  (h1 : triangle_a_side1 = 7)
  (h2 : triangle_a_side2 = 24)
  (h3 : triangle_a_hypotenuse = 39)
  (h4 : triangle_b_side1 = 15)
  (h5 : triangle_b_side2 = 36)
  (h6 : triangle_b_hypotenuse = 39)
  (h7 : triangle_a_side1^2 + triangle_a_side2^2 = triangle_a_hypotenuse^2)
  (h8 : triangle_b_side1^2 + triangle_b_side2^2 = triangle_b_hypotenuse^2)
  (h9 : triangle_a_hypotenuse = triangle_b_hypotenuse) :
  39 = triangle_a_hypotenuse ∧ 39 = triangle_b_hypotenuse := by
  sorry

#check circle_diameter_from_inscribed_triangles

end NUMINAMATH_CALUDE_circle_diameter_from_inscribed_triangles_l4105_410556


namespace NUMINAMATH_CALUDE_peaches_before_picking_l4105_410582

-- Define the variables
def peaches_picked : ℕ := 52
def total_peaches_now : ℕ := 86

-- Define the theorem
theorem peaches_before_picking (peaches_picked total_peaches_now : ℕ) :
  peaches_picked = 52 →
  total_peaches_now = 86 →
  total_peaches_now - peaches_picked = 34 := by
sorry

end NUMINAMATH_CALUDE_peaches_before_picking_l4105_410582


namespace NUMINAMATH_CALUDE_abhay_speed_l4105_410555

theorem abhay_speed (distance : ℝ) (abhay_speed : ℝ) (sameer_speed : ℝ) : 
  distance = 24 →
  distance / abhay_speed = distance / sameer_speed + 2 →
  distance / (2 * abhay_speed) = distance / sameer_speed - 1 →
  abhay_speed = 12 := by
sorry

end NUMINAMATH_CALUDE_abhay_speed_l4105_410555


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4105_410571

/-- Given a geometric sequence with first term a and common ratio r, 
    such that the sum of the first 1500 terms is 300 and 
    the sum of the first 3000 terms is 570,
    prove that the sum of the first 4500 terms is 813. -/
theorem geometric_sequence_sum (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) : 
  a * (1 - r^4500) / (1 - r) = 813 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4105_410571


namespace NUMINAMATH_CALUDE_book_reorganization_l4105_410549

theorem book_reorganization (initial_boxes : Nat) (initial_books_per_box : Nat) (new_books_per_box : Nat) :
  initial_boxes = 1278 →
  initial_books_per_box = 45 →
  new_books_per_box = 46 →
  (initial_boxes * initial_books_per_box) % new_books_per_box = 10 := by
  sorry

end NUMINAMATH_CALUDE_book_reorganization_l4105_410549


namespace NUMINAMATH_CALUDE_rectangle_ratio_is_two_l4105_410588

-- Define the side length of the inner square
def inner_square_side : ℝ := 1

-- Define the shorter side of the rectangle
def rectangle_short_side : ℝ := inner_square_side

-- Define the longer side of the rectangle
def rectangle_long_side : ℝ := 2 * inner_square_side

-- Define the side length of the outer square
def outer_square_side : ℝ := inner_square_side + 2 * rectangle_short_side

-- State the theorem
theorem rectangle_ratio_is_two :
  (outer_square_side ^ 2 = 9 * inner_square_side ^ 2) →
  (rectangle_long_side / rectangle_short_side = 2) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_ratio_is_two_l4105_410588


namespace NUMINAMATH_CALUDE_circle_equation_equivalence_l4105_410526

theorem circle_equation_equivalence :
  ∀ x y : ℝ, x^2 - 6*x + y^2 - 10*y + 18 = 0 ↔ (x-3)^2 + (y-5)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_equivalence_l4105_410526


namespace NUMINAMATH_CALUDE_jake_peaches_l4105_410569

/-- Given information about peaches owned by Steven, Jill, and Jake -/
theorem jake_peaches (steven_peaches : ℕ) (jill_peaches : ℕ) (jake_peaches : ℕ)
  (h1 : steven_peaches = 15)
  (h2 : steven_peaches = jill_peaches + 14)
  (h3 : jake_peaches = steven_peaches - 7) :
  jake_peaches = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_l4105_410569


namespace NUMINAMATH_CALUDE_externally_tangent_circles_equation_l4105_410530

-- Define the radii and angle
variable (r r' φ : ℝ)

-- Define the conditions
variable (hr : r > 0)
variable (hr' : r' > 0)
variable (hφ : 0 < φ ∧ φ < π)

-- Define the externally tangent circles condition
variable (h_tangent : r + r' > 0)

-- Theorem statement
theorem externally_tangent_circles_equation :
  (r + r')^2 * Real.sin φ = 4 * (r - r') * Real.sqrt (r * r') := by
  sorry

end NUMINAMATH_CALUDE_externally_tangent_circles_equation_l4105_410530


namespace NUMINAMATH_CALUDE_sqrt_simplification_algebraic_simplification_l4105_410552

-- Problem 1
theorem sqrt_simplification :
  Real.sqrt 6 * Real.sqrt (2/3) / Real.sqrt 2 = Real.sqrt 2 := by sorry

-- Problem 2
theorem algebraic_simplification :
  (Real.sqrt 2 + Real.sqrt 5)^2 - (Real.sqrt 2 + Real.sqrt 5)*(Real.sqrt 2 - Real.sqrt 5) = 10 + 2*Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_sqrt_simplification_algebraic_simplification_l4105_410552


namespace NUMINAMATH_CALUDE_perfect_square_condition_l4105_410523

theorem perfect_square_condition (n : ℕ) : 
  ∃ k : ℕ, n^2 + n + 1 = k^2 ↔ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l4105_410523


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4105_410541

/-- Given an arithmetic sequence {a_n} with positive terms, sum S_n, and common difference d,
    if {√S_n} is also arithmetic with the same difference d, then a_n = (2n - 1) / 4 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, S n = n * a 1 + n * (n - 1) / 2 * d) →
  (∀ n, a (n + 1) = a n + d) →
  (∀ n, Real.sqrt (S (n + 1)) = Real.sqrt (S n) + d) →
  ∀ n, a n = (2 * n - 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4105_410541


namespace NUMINAMATH_CALUDE_systematic_sampling_second_group_l4105_410563

/-- Systematic sampling function -/
def systematicSample (totalStudents : ℕ) (sampleSize : ℕ) (groupNumber : ℕ) : ℕ :=
  let interval := totalStudents / sampleSize
  (groupNumber - 1) * interval + 1

theorem systematic_sampling_second_group
  (totalStudents : ℕ)
  (sampleSize : ℕ)
  (h1 : totalStudents = 160)
  (h2 : sampleSize = 20)
  (h3 : systematicSample totalStudents sampleSize 16 = 123) :
  systematicSample totalStudents sampleSize 2 = 11 := by
sorry

end NUMINAMATH_CALUDE_systematic_sampling_second_group_l4105_410563


namespace NUMINAMATH_CALUDE_choose_five_items_eq_48_l4105_410546

/-- The number of ways to choose 5 items from 3 distinct types, 
    where no two consecutive items can be of the same type -/
def choose_five_items : ℕ :=
  let first_choice := 3  -- 3 choices for the first item
  let subsequent_choices := 2  -- 2 choices for each subsequent item
  first_choice * subsequent_choices^4

theorem choose_five_items_eq_48 : choose_five_items = 48 := by
  sorry

end NUMINAMATH_CALUDE_choose_five_items_eq_48_l4105_410546


namespace NUMINAMATH_CALUDE_sin_2α_plus_π_6_l4105_410565

theorem sin_2α_plus_π_6 (α : ℝ) (h : Real.sin (α + π / 3) = 3 / 5) :
  Real.sin (2 * α + π / 6) = -(7 / 25) := by
  sorry

end NUMINAMATH_CALUDE_sin_2α_plus_π_6_l4105_410565


namespace NUMINAMATH_CALUDE_cookie_arrangements_count_l4105_410576

/-- The number of distinct arrangements of letters in "COOKIE" -/
def cookieArrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)

/-- Theorem stating that the number of distinct arrangements of letters in "COOKIE" is 360 -/
theorem cookie_arrangements_count : cookieArrangements = 360 := by
  sorry

end NUMINAMATH_CALUDE_cookie_arrangements_count_l4105_410576


namespace NUMINAMATH_CALUDE_parabola_segment_length_l4105_410531

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  hp : p < 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point lies on the parabola -/
def onParabola (para : Parabola) (pt : Point) : Prop :=
  pt.y^2 = 2 * para.p * pt.x

/-- Represents a line passing through the focus of the parabola at an angle with the x-axis -/
structure FocusLine where
  para : Parabola
  angle : ℝ

/-- Calculates the length of the segment AB formed by the intersection of the focus line with the parabola -/
noncomputable def segmentLength (para : Parabola) (fl : FocusLine) : ℝ :=
  sorry -- Actual calculation would go here

theorem parabola_segment_length 
  (para : Parabola) 
  (ptA : Point) 
  (fl : FocusLine) :
  onParabola para ptA → 
  ptA.x = -2 → 
  ptA.y = -4 → 
  fl.para = para → 
  fl.angle = π/3 → 
  segmentLength para fl = 32/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_segment_length_l4105_410531


namespace NUMINAMATH_CALUDE_anne_weight_proof_l4105_410595

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference in weight between Anne and Douglas in pounds -/
def weight_difference : ℕ := 15

/-- Theorem: Anne's weight is 67 pounds, given Douglas's weight and the weight difference -/
theorem anne_weight_proof : anne_weight = douglas_weight + weight_difference := by
  sorry

end NUMINAMATH_CALUDE_anne_weight_proof_l4105_410595


namespace NUMINAMATH_CALUDE_team_selection_count_l4105_410507

/-- The number of ways to select a team of 5 people from a group of 16 people -/
def select_team (total : ℕ) (team_size : ℕ) : ℕ :=
  Nat.choose total team_size

/-- The total number of students in the math club -/
def total_students : ℕ := 16

/-- The number of boys in the math club -/
def num_boys : ℕ := 7

/-- The number of girls in the math club -/
def num_girls : ℕ := 9

/-- The size of the team to be selected -/
def team_size : ℕ := 5

theorem team_selection_count :
  select_team total_students team_size = 4368 ∧
  total_students = num_boys + num_girls :=
sorry

end NUMINAMATH_CALUDE_team_selection_count_l4105_410507


namespace NUMINAMATH_CALUDE_locus_of_p_l4105_410501

-- Define the hyperbola
def hyperbola (a b x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Define the point A on the hyperbola
def point_on_hyperbola (a b x y : ℝ) : Prop := hyperbola a b x y ∧ x ≠ 0 ∧ y ≠ 0

-- Define the reflection of a point about y-axis
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

-- Define the reflection of a point about x-axis
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)

-- Define the reflection of a point about origin
def reflect_origin (x y : ℝ) : ℝ × ℝ := (-x, -y)

-- Define perpendicularity of two lines
def perpendicular (x1 y1 x2 y2 : ℝ) : Prop := x1 * x2 + y1 * y2 = 0

-- Main theorem
theorem locus_of_p (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (x y : ℝ), y ≠ 0 →
  (∃ (x0 y0 x1 y1 : ℝ),
    point_on_hyperbola a b x0 y0 ∧
    point_on_hyperbola a b x1 y1 ∧
    perpendicular (x1 - x0) (y1 - y0) (-2*x0) (-2*y0) ∧
    x = ((a^2 + b^2) / (a^2 - b^2)) * x0 ∧
    y = -((a^2 + b^2) / (a^2 - b^2)) * y0) →
  x^2 / a^2 - y^2 / b^2 = (a^2 + b^2)^2 / (a^2 - b^2)^2 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_p_l4105_410501


namespace NUMINAMATH_CALUDE_stock_ratio_proof_l4105_410557

def stock_problem (expensive_shares : ℕ) (other_shares : ℕ) (total_value : ℕ) (expensive_price : ℕ) : Prop :=
  ∃ (other_price : ℕ),
    expensive_shares * expensive_price + other_shares * other_price = total_value ∧
    expensive_price / other_price = 2

theorem stock_ratio_proof :
  stock_problem 14 26 2106 78 := by
  sorry

end NUMINAMATH_CALUDE_stock_ratio_proof_l4105_410557


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l4105_410566

/-- Represents the number of triangles of each color in each half of the figure -/
structure HalfFigure where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs when the figure is folded -/
structure CoincidingPairs where
  redRed : ℕ
  blueBlue : ℕ
  redWhite : ℕ
  whiteWhite : ℕ

theorem white_pairs_coincide (half : HalfFigure) (pairs : CoincidingPairs) : 
  half.red = 4 ∧ 
  half.blue = 6 ∧ 
  half.white = 10 ∧ 
  pairs.redRed = 3 ∧ 
  pairs.blueBlue = 4 ∧ 
  pairs.redWhite = 3 → 
  pairs.whiteWhite = 5 := by
sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l4105_410566


namespace NUMINAMATH_CALUDE_base6_division_equality_l4105_410547

-- Define a function to convert from base 6 to base 10
def base6ToBase10 (n : ℕ) : ℕ := sorry

-- Define a function to convert from base 10 to base 6
def base10ToBase6 (n : ℕ) : ℕ := sorry

-- Define the division operation in base 6
def divBase6 (a b : ℕ) : ℕ := base10ToBase6 (base6ToBase10 a / base6ToBase10 b)

-- Theorem statement
theorem base6_division_equality :
  divBase6 2314 14 = 135 := by sorry

end NUMINAMATH_CALUDE_base6_division_equality_l4105_410547


namespace NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l4105_410558

/-- The surface area of a sphere circumscribing a rectangular solid with edge lengths 2, 3, and 4 is 29π. -/
theorem circumscribed_sphere_surface_area (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  let diagonal_squared := a^2 + b^2 + c^2
  let radius := Real.sqrt (diagonal_squared / 4)
  4 * Real.pi * radius^2 = 29 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_circumscribed_sphere_surface_area_l4105_410558


namespace NUMINAMATH_CALUDE_sin_alpha_fourth_quadrant_l4105_410574

theorem sin_alpha_fourth_quadrant (α : Real) : 
  (π/2 < α ∧ α < 2*π) →  -- α is in the fourth quadrant
  (Real.tan (π - α) = 5/12) → 
  (Real.sin α = -5/13) := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_fourth_quadrant_l4105_410574


namespace NUMINAMATH_CALUDE_divisibility_by_twelve_l4105_410536

theorem divisibility_by_twelve (a b c d : ℤ) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_twelve_l4105_410536


namespace NUMINAMATH_CALUDE_subtract_correction_l4105_410513

theorem subtract_correction (x : ℤ) (h : x - 42 = 50) : x - 24 = 68 := by
  sorry

end NUMINAMATH_CALUDE_subtract_correction_l4105_410513


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l4105_410564

theorem completing_square_equivalence (x : ℝ) : 
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l4105_410564


namespace NUMINAMATH_CALUDE_mikaela_personal_needs_fraction_l4105_410548

/-- Calculates the fraction of total earnings spent on personal needs --/
def fraction_spent_on_personal_needs (hourly_rate : ℚ) (first_month_hours : ℕ) (second_month_additional_hours : ℕ) (amount_saved : ℚ) : ℚ :=
  let first_month_earnings := hourly_rate * first_month_hours
  let second_month_hours := first_month_hours + second_month_additional_hours
  let second_month_earnings := hourly_rate * second_month_hours
  let total_earnings := first_month_earnings + second_month_earnings
  let amount_spent := total_earnings - amount_saved
  amount_spent / total_earnings

/-- Proves that Mikaela spent 4/5 of her total earnings on personal needs --/
theorem mikaela_personal_needs_fraction :
  fraction_spent_on_personal_needs 10 35 5 150 = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_mikaela_personal_needs_fraction_l4105_410548


namespace NUMINAMATH_CALUDE_total_cookies_kept_l4105_410553

def oatmeal_baked : ℕ := 40
def sugar_baked : ℕ := 28
def chocolate_baked : ℕ := 55

def oatmeal_given : ℕ := 26
def sugar_given : ℕ := 17
def chocolate_given : ℕ := 34

def cookies_kept (baked given : ℕ) : ℕ := baked - given

theorem total_cookies_kept :
  cookies_kept oatmeal_baked oatmeal_given +
  cookies_kept sugar_baked sugar_given +
  cookies_kept chocolate_baked chocolate_given = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_kept_l4105_410553


namespace NUMINAMATH_CALUDE_crackers_per_person_is_76_l4105_410525

/-- The number of crackers each person receives when Darren and Calvin's crackers are shared equally among themselves and 3 friends. -/
def crackers_per_person : ℕ :=
  let darren_type_a_boxes := 4
  let darren_type_b_boxes := 2
  let crackers_per_type_a_box := 24
  let crackers_per_type_b_box := 30
  let calvin_type_a_boxes := 2 * darren_type_a_boxes - 1
  let calvin_type_b_boxes := darren_type_b_boxes
  let total_crackers := 
    (darren_type_a_boxes + calvin_type_a_boxes) * crackers_per_type_a_box +
    (darren_type_b_boxes + calvin_type_b_boxes) * crackers_per_type_b_box
  let number_of_people := 5
  total_crackers / number_of_people

theorem crackers_per_person_is_76 : crackers_per_person = 76 := by
  sorry

end NUMINAMATH_CALUDE_crackers_per_person_is_76_l4105_410525


namespace NUMINAMATH_CALUDE_triangle_inequality_l4105_410581

theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  a = 1 ∧
  b * Real.cos A - Real.cos B = 1 →
  Real.sqrt 3 < Real.sin B + 2 * Real.sqrt 3 * Real.sin A * Real.sin A ∧
  Real.sin B + 2 * Real.sqrt 3 * Real.sin A * Real.sin A < 1 + Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l4105_410581


namespace NUMINAMATH_CALUDE_problem_statement_l4105_410594

theorem problem_statement (x y z : ℝ) 
  (h1 : x ≠ y)
  (h2 : x^2 * (y + z) = 2019)
  (h3 : y^2 * (z + x) = 2019) :
  z^2 * (x + y) - x * y * z = 4038 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4105_410594


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l4105_410554

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ := sorry

/-- The specific line passing through (2, -2) and (6, 10) -/
def specific_line : Line := { x₁ := 2, y₁ := -2, x₂ := 6, y₂ := 10 }

theorem x_intercept_of_specific_line :
  x_intercept specific_line = 8/3 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l4105_410554


namespace NUMINAMATH_CALUDE_part_to_whole_ratio_l4105_410572

theorem part_to_whole_ratio (N : ℚ) (x : ℚ) (h1 : N = 280) (h2 : x + 4 = N / 4 - 10) : x / N = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_part_to_whole_ratio_l4105_410572


namespace NUMINAMATH_CALUDE_chinese_in_group_l4105_410550

theorem chinese_in_group (total : ℕ) (americans : ℕ) (australians : ℕ) 
  (h1 : total = 49)
  (h2 : americans = 16)
  (h3 : australians = 11) :
  total - americans - australians = 22 :=
by sorry

end NUMINAMATH_CALUDE_chinese_in_group_l4105_410550


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l4105_410514

/-- The parabola y = 2x^2 intersects with the line y = kx + 2 at points A and B.
    M is the midpoint of AB, and N is the foot of the perpendicular from M to the x-axis.
    If the dot product of NA and NB is zero, then k = ±4√3. -/
theorem parabola_line_intersection (k : ℝ) : 
  let C : ℝ → ℝ := λ x => 2 * x^2
  let L : ℝ → ℝ := λ x => k * x + 2
  let A : ℝ × ℝ := (x₁, C x₁)
  let B : ℝ × ℝ := (x₂, C x₂)
  let M : ℝ × ℝ := ((x₁ + x₂)/2, (C x₁ + C x₂)/2)
  let N : ℝ × ℝ := (M.1, 0)
  C x₁ = L x₁ ∧ C x₂ = L x₂ ∧ x₁ ≠ x₂ →
  (A.1 - N.1) * (B.1 - N.1) + (A.2 - N.2) * (B.2 - N.2) = 0 →
  k = 4 * Real.sqrt 3 ∨ k = -4 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l4105_410514


namespace NUMINAMATH_CALUDE_A_3_2_equals_29_l4105_410559

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem A_3_2_equals_29 : A 3 2 = 29 := by sorry

end NUMINAMATH_CALUDE_A_3_2_equals_29_l4105_410559


namespace NUMINAMATH_CALUDE_solution_sets_l4105_410540

-- Define the solution sets
def S (c b : ℝ) : Set ℝ := {x | c * x^2 + x + b < 0}
def M (b c : ℝ) : Set ℝ := {x | b * x^2 + x + c > 0}
def N (a : ℝ) : Set ℝ := {x | x^2 + x < a^2 - a}

-- State the theorem
theorem solution_sets :
  ∃ (c b : ℝ),
    (S c b = {x | -1 < x ∧ x < 1/2}) ∧
    (∃ (a : ℝ), M b c ∪ (Set.univ \ N a) = Set.univ) →
    (M b c = {x | -1 < x ∧ x < 2}) ∧
    {a : ℝ | 0 ≤ a ∧ a ≤ 1} = {a : ℝ | ∃ (b c : ℝ), M b c ∪ (Set.univ \ N a) = Set.univ} :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_l4105_410540
