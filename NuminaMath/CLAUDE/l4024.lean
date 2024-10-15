import Mathlib

namespace NUMINAMATH_CALUDE_tangerines_highest_frequency_l4024_402452

/-- Represents the number of boxes for each fruit type -/
def num_boxes_tangerines : ℕ := 5
def num_boxes_apples : ℕ := 3
def num_boxes_pears : ℕ := 4

/-- Represents the number of fruits per box for each fruit type -/
def fruits_per_box_tangerines : ℕ := 30
def fruits_per_box_apples : ℕ := 20
def fruits_per_box_pears : ℕ := 15

/-- Represents the weight of each fruit in grams -/
def weight_tangerine : ℕ := 200
def weight_apple : ℕ := 450
def weight_pear : ℕ := 800

/-- Calculates the total number of fruits for each type -/
def total_tangerines : ℕ := num_boxes_tangerines * fruits_per_box_tangerines
def total_apples : ℕ := num_boxes_apples * fruits_per_box_apples
def total_pears : ℕ := num_boxes_pears * fruits_per_box_pears

/-- Theorem: Tangerines have the highest frequency -/
theorem tangerines_highest_frequency :
  total_tangerines > total_apples ∧ total_tangerines > total_pears :=
sorry

end NUMINAMATH_CALUDE_tangerines_highest_frequency_l4024_402452


namespace NUMINAMATH_CALUDE_distance_to_equidistant_line_in_unit_cube_l4024_402467

/-- Represents a 3D point -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a 3D line -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents a unit cube -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  B1 : Point3D
  C1 : Point3D
  D1 : Point3D

/-- Distance between a point and a line in 3D space -/
def distancePointToLine (p : Point3D) (l : Line3D) : ℝ :=
  sorry

/-- Check if two lines are parallel -/
def areLinesParallel (l1 l2 : Line3D) : Prop :=
  sorry

/-- Check if a line is equidistant from three other lines -/
def isLineEquidistantFromThreeLines (l l1 l2 l3 : Line3D) : Prop :=
  sorry

/-- The main theorem -/
theorem distance_to_equidistant_line_in_unit_cube 
  (cube : UnitCube) 
  (l : Line3D) 
  (hParallel : areLinesParallel l (Line3D.mk cube.A cube.C1))
  (hEquidistant : isLineEquidistantFromThreeLines l 
    (Line3D.mk cube.B cube.D) 
    (Line3D.mk cube.A1 cube.D1) 
    (Line3D.mk cube.C cube.B1)) :
  distancePointToLine cube.B (Line3D.mk cube.B cube.D) = Real.sqrt 2 / 6 ∧
  distancePointToLine cube.A1 (Line3D.mk cube.A1 cube.D1) = Real.sqrt 2 / 6 ∧
  distancePointToLine cube.C (Line3D.mk cube.C cube.B1) = Real.sqrt 2 / 6 :=
sorry

end NUMINAMATH_CALUDE_distance_to_equidistant_line_in_unit_cube_l4024_402467


namespace NUMINAMATH_CALUDE_value_of_B_l4024_402433

theorem value_of_B : ∃ B : ℚ, (3/4 : ℚ) * (1/8 : ℚ) * B = (1/4 : ℚ) * (1/6 : ℚ) * 48 ∧ B = 64/3 :=
by sorry

end NUMINAMATH_CALUDE_value_of_B_l4024_402433


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l4024_402474

theorem pie_crust_flour_calculation (initial_crusts : ℕ) (initial_flour_per_crust : ℚ) 
  (new_crusts : ℕ) (h1 : initial_crusts = 40) (h2 : initial_flour_per_crust = 1/8) 
  (h3 : new_crusts = 25) :
  let total_flour := initial_crusts * initial_flour_per_crust
  let new_flour_per_crust := total_flour / new_crusts
  new_flour_per_crust = 1/5 := by
sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l4024_402474


namespace NUMINAMATH_CALUDE_intersection_and_union_when_a_is_2_subset_condition_l4024_402416

-- Define sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a - 1}

-- Theorem for part (1)
theorem intersection_and_union_when_a_is_2 :
  (M ∩ N 2 = {3}) ∧ (M ∪ N 2 = M) := by sorry

-- Theorem for part (2)
theorem subset_condition :
  ∀ a : ℝ, (M ⊇ N a) ↔ a ≤ 3 := by sorry

end NUMINAMATH_CALUDE_intersection_and_union_when_a_is_2_subset_condition_l4024_402416


namespace NUMINAMATH_CALUDE_olympic_tournament_winners_l4024_402430

/-- Represents an Olympic system tournament -/
structure OlympicTournament where
  rounds : ℕ
  initialParticipants : ℕ
  winnersEachRound : List ℕ

/-- Checks if the tournament is valid -/
def isValidTournament (t : OlympicTournament) : Prop :=
  t.rounds > 0 ∧
  t.initialParticipants = 2^t.rounds ∧
  t.winnersEachRound.length = t.rounds ∧
  ∀ i, i ∈ t.winnersEachRound → i = t.initialParticipants / (2^(t.winnersEachRound.indexOf i + 1))

/-- Calculates the number of participants who won more games than they lost -/
def participantsWithMoreWins (t : OlympicTournament) : ℕ :=
  t.initialParticipants / 4

theorem olympic_tournament_winners (t : OlympicTournament) 
  (h1 : isValidTournament t) 
  (h2 : t.rounds = 6) : 
  participantsWithMoreWins t = 16 := by
  sorry

#check olympic_tournament_winners

end NUMINAMATH_CALUDE_olympic_tournament_winners_l4024_402430


namespace NUMINAMATH_CALUDE_a_range_l4024_402468

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 > -a * x - 1 ∧ a ≠ 0

def q (a : ℝ) : Prop := ∀ x y : ℝ, 
  x^2 + y^2 = a^2 → (x + 3)^2 + (y - 4)^2 > 4

-- Define the range of a
def range_a (a : ℝ) : Prop := (a > -3 ∧ a ≤ 0) ∨ (a ≥ 3 ∧ a < 4)

-- State the theorem
theorem a_range : 
  (∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a)) → 
  (∀ a : ℝ, range_a a ↔ (p a ∨ q a) ∧ ¬(p a ∧ q a)) :=
sorry

end NUMINAMATH_CALUDE_a_range_l4024_402468


namespace NUMINAMATH_CALUDE_product_odd_implies_sum_odd_l4024_402421

theorem product_odd_implies_sum_odd (a b c : ℤ) : 
  Odd (a * b * c) → Odd (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_product_odd_implies_sum_odd_l4024_402421


namespace NUMINAMATH_CALUDE_taxi_charge_theorem_l4024_402439

-- Define the parameters of the taxi service
def initial_fee : ℚ := 235 / 100
def charge_per_increment : ℚ := 35 / 100
def miles_per_increment : ℚ := 2 / 5
def trip_distance : ℚ := 36 / 10

-- Define the total charge function
def total_charge (initial_fee charge_per_increment miles_per_increment trip_distance : ℚ) : ℚ :=
  initial_fee + (trip_distance / miles_per_increment) * charge_per_increment

-- State the theorem
theorem taxi_charge_theorem :
  total_charge initial_fee charge_per_increment miles_per_increment trip_distance = 865 / 100 := by
  sorry

end NUMINAMATH_CALUDE_taxi_charge_theorem_l4024_402439


namespace NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l4024_402472

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 :=
sorry

end NUMINAMATH_CALUDE_two_solutions_iff_a_gt_neg_one_l4024_402472


namespace NUMINAMATH_CALUDE_positive_integer_division_l4024_402478

theorem positive_integer_division (a b : ℕ+) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔
    ((a = 11 ∧ b = 1) ∨
     (a = 49 ∧ b = 1) ∨
     (∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k)) := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_division_l4024_402478


namespace NUMINAMATH_CALUDE_johns_family_ages_l4024_402404

/-- Given information about John's family ages, prove John's and his sibling's ages -/
theorem johns_family_ages :
  ∀ (john_age dad_age sibling_age : ℕ),
  john_age + 30 = dad_age →
  john_age + dad_age = 90 →
  sibling_age = john_age + 5 →
  john_age = 30 ∧ sibling_age = 35 := by
  sorry

end NUMINAMATH_CALUDE_johns_family_ages_l4024_402404


namespace NUMINAMATH_CALUDE_max_revenue_l4024_402471

/-- Represents the production quantities of products A and B -/
structure Production where
  a : ℝ
  b : ℝ

/-- Calculates the revenue for a given production -/
def revenue (p : Production) : ℝ :=
  0.3 * p.a + 0.2 * p.b

/-- Checks if a production is feasible given the machine constraints -/
def is_feasible (p : Production) : Prop :=
  p.a ≥ 0 ∧ p.b ≥ 0 ∧
  1 * p.a + 2 * p.b ≤ 400 ∧
  2 * p.a + 1 * p.b ≤ 500

/-- Theorem stating the maximum monthly sales revenue -/
theorem max_revenue :
  ∃ (p : Production), is_feasible p ∧
    ∀ (q : Production), is_feasible q → revenue q ≤ revenue p ∧
    revenue p = 90 :=
sorry

end NUMINAMATH_CALUDE_max_revenue_l4024_402471


namespace NUMINAMATH_CALUDE_student_d_score_l4024_402451

/-- Represents a student's answers and score -/
structure StudentAnswers :=
  (answers : List Bool)
  (score : Nat)

/-- The problem setup -/
def mathTestProblem :=
  let numQuestions : Nat := 8
  let pointsPerQuestion : Nat := 5
  let totalPossibleScore : Nat := 40
  let studentA : StudentAnswers := ⟨[false, true, false, true, false, false, true, false], 30⟩
  let studentB : StudentAnswers := ⟨[false, false, true, true, true, false, false, true], 25⟩
  let studentC : StudentAnswers := ⟨[true, false, false, false, true, true, true, false], 25⟩
  let studentD : StudentAnswers := ⟨[false, true, false, true, true, false, true, true], 0⟩ -- score unknown
  (numQuestions, pointsPerQuestion, totalPossibleScore, studentA, studentB, studentC, studentD)

/-- The theorem to prove -/
theorem student_d_score :
  let (numQuestions, pointsPerQuestion, totalPossibleScore, studentA, studentB, studentC, studentD) := mathTestProblem
  studentD.score = 30 := by
  sorry


end NUMINAMATH_CALUDE_student_d_score_l4024_402451


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_polar_to_cartesian_line_l4024_402481

/-- Given a line and a circle, find the minimum distance from a point on the circle to the line -/
theorem min_distance_circle_to_line :
  let line := {(x, y) : ℝ × ℝ | x + y = 1}
  let circle := {(x, y) : ℝ × ℝ | ∃ θ : ℝ, x = 2 * Real.cos θ ∧ y = -2 + 2 * Real.sin θ}
  ∃ d : ℝ, d = (3 * Real.sqrt 2) / 2 - 2 ∧
    ∀ p ∈ circle, ∀ q ∈ line, Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d :=
by sorry

/-- The polar equation of the line can be converted to Cartesian form -/
theorem polar_to_cartesian_line :
  ∀ ρ θ : ℝ, ρ * Real.sin (θ + π/4) = Real.sqrt 2 / 2 →
  ∃ x y : ℝ, x + y = 1 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_polar_to_cartesian_line_l4024_402481


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4024_402448

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (4 - 5 * x) = 8 → x = -12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4024_402448


namespace NUMINAMATH_CALUDE_lesser_solution_quadratic_l4024_402413

theorem lesser_solution_quadratic (x : ℝ) : 
  x^2 + 10*x - 75 = 0 → (∃ y : ℝ, y^2 + 10*y - 75 = 0 ∧ y ≤ x) → x = -15 :=
sorry

end NUMINAMATH_CALUDE_lesser_solution_quadratic_l4024_402413


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l4024_402441

def p (x : ℝ) : ℝ := 3 * (x^2 - 3*x + 3) - 5 * (x^3 - 2*x^2 + 4*x - 1)

theorem sum_of_squares_of_coefficients :
  ∃ (a b c d : ℝ),
    (∀ x, p x = a * x^3 + b * x^2 + c * x + d) ∧
    a^2 + b^2 + c^2 + d^2 = 1231 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_l4024_402441


namespace NUMINAMATH_CALUDE_maci_school_supplies_cost_l4024_402461

/-- The cost of Maci's school supplies -/
def school_supplies_cost (blue_pen_price : ℚ) : ℚ :=
  let red_pen_price := 2 * blue_pen_price
  let pencil_price := red_pen_price / 2
  let notebook_price := 10 * blue_pen_price
  10 * blue_pen_price +  -- 10 blue pens
  15 * red_pen_price +   -- 15 red pens
  5 * pencil_price +     -- 5 pencils
  3 * notebook_price     -- 3 notebooks

/-- Theorem stating that the cost of Maci's school supplies is $7.50 -/
theorem maci_school_supplies_cost :
  school_supplies_cost (10 / 100) = 75 / 10 := by
  sorry

end NUMINAMATH_CALUDE_maci_school_supplies_cost_l4024_402461


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l4024_402458

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def B (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3*a) < 0}

-- Theorem for A ⊆ B
theorem subset_condition (a : ℝ) : 
  A ⊆ B a ↔ (4/3 ≤ a ∧ a ≤ 2 ∧ a ≠ 0) :=
sorry

-- Theorem for A ∩ B = ∅
theorem disjoint_condition (a : ℝ) :
  A ∩ B a = ∅ ↔ (a ≤ 2/3 ∨ a ≥ 4) :=
sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l4024_402458


namespace NUMINAMATH_CALUDE_double_factorial_sum_denominator_l4024_402470

/-- Double factorial for odd numbers -/
def odd_double_factorial (n : ℕ) : ℕ := sorry

/-- Double factorial for even numbers -/
def even_double_factorial (n : ℕ) : ℕ := sorry

/-- The sum of the ratios of double factorials -/
def double_factorial_sum : ℚ :=
  (Finset.range 2009).sum (fun i => (odd_double_factorial (2*i+1)) / (even_double_factorial (2*i+2)))

/-- The denominator of the sum when expressed in lowest terms -/
def denominator_of_sum : ℕ := sorry

/-- The power of 2 in the denominator -/
def a : ℕ := sorry

/-- The odd factor in the denominator -/
def b : ℕ := sorry

theorem double_factorial_sum_denominator :
  denominator_of_sum = 2^a * b ∧ Odd b ∧ a*b/10 = 401 := by sorry

end NUMINAMATH_CALUDE_double_factorial_sum_denominator_l4024_402470


namespace NUMINAMATH_CALUDE_emerson_first_part_distance_l4024_402401

/-- Emerson's rowing trip distances -/
structure RowingTrip where
  total : ℕ
  second : ℕ
  third : ℕ

/-- The distance covered in the first part of the rowing trip -/
def firstPartDistance (trip : RowingTrip) : ℕ :=
  trip.total - (trip.second + trip.third)

/-- Theorem: The first part distance of Emerson's specific trip is 6 miles -/
theorem emerson_first_part_distance :
  firstPartDistance ⟨39, 15, 18⟩ = 6 := by
  sorry

end NUMINAMATH_CALUDE_emerson_first_part_distance_l4024_402401


namespace NUMINAMATH_CALUDE_inequality_solution_l4024_402428

theorem inequality_solution (y : ℝ) : 
  (y^2 + 2*y^3 - 3*y^4) / (y + 2*y^2 - 3*y^3) ≥ -1 ↔ 
  (y ∈ Set.Icc (-1) (-1/3) ∪ Set.Ioo (-1/3) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1) ∧ 
  (y ≠ -1/3) ∧ (y ≠ 0) ∧ (y ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l4024_402428


namespace NUMINAMATH_CALUDE_sum_of_non_solutions_l4024_402454

/-- Given an equation with infinitely many solutions, prove the sum of non-solution x values -/
theorem sum_of_non_solutions (A B C : ℝ) : 
  (∀ x : ℝ, (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9)) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x + B) * (A * x + 36) ≠ 3 * (x + C) * (x + 9) ↔ (x = x₁ ∨ x = x₂)) →
  x₁ + x₂ = -21 :=
sorry

end NUMINAMATH_CALUDE_sum_of_non_solutions_l4024_402454


namespace NUMINAMATH_CALUDE_simplify_polynomial_l4024_402418

theorem simplify_polynomial (x : ℝ) : 
  2*x*(4*x^3 - 3*x + 1) - 4*(2*x^3 - x^2 + 3*x - 5) = 
  8*x^4 - 8*x^3 - 2*x^2 - 10*x + 20 := by sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l4024_402418


namespace NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_five_l4024_402405

theorem smallest_addition_for_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (726 + n) % 5 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ (726 + m) % 5 = 0 → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_multiple_of_five_l4024_402405


namespace NUMINAMATH_CALUDE_stamp_cost_correct_l4024_402477

/-- The cost of one stamp, given that three stamps cost $1.02 and the cost is constant -/
def stamp_cost : ℚ := 0.34

/-- The cost of three stamps -/
def three_stamps_cost : ℚ := 1.02

/-- Theorem stating that the cost of one stamp is correct -/
theorem stamp_cost_correct : 3 * stamp_cost = three_stamps_cost := by sorry

end NUMINAMATH_CALUDE_stamp_cost_correct_l4024_402477


namespace NUMINAMATH_CALUDE_some_employees_not_in_management_team_l4024_402465

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Employee : U → Prop)
variable (ManagementTeam : U → Prop)
variable (CompletesTraining : U → Prop)

-- State the theorem
theorem some_employees_not_in_management_team
  (h1 : ∃ x, Employee x ∧ ¬CompletesTraining x)
  (h2 : ∀ x, ManagementTeam x → CompletesTraining x) :
  ∃ x, Employee x ∧ ¬ManagementTeam x :=
by sorry

end NUMINAMATH_CALUDE_some_employees_not_in_management_team_l4024_402465


namespace NUMINAMATH_CALUDE_fraction_meaningful_l4024_402495

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (4 + x) / (4 - 2*x)) ↔ x ≠ 2 :=
sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l4024_402495


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l4024_402406

/-- Given a cone with base radius 6 and volume 30π, its lateral surface area is 39π -/
theorem cone_lateral_surface_area (r h l : ℝ) : 
  r = 6 → 
  (1/3) * π * r^2 * h = 30*π → 
  l^2 = r^2 + h^2 → 
  π * r * l = 39*π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l4024_402406


namespace NUMINAMATH_CALUDE_candle_height_after_80000_seconds_l4024_402435

/-- Represents the burning pattern of a candle -/
structure BurningPattern where
  oddCentimeterTime : ℕ → ℕ  -- Time to burn odd-numbered centimeters
  evenCentimeterTime : ℕ → ℕ -- Time to burn even-numbered centimeters

/-- Calculates the remaining height of a candle after a given time -/
def remainingHeight (initialHeight : ℕ) (pattern : BurningPattern) (elapsedTime : ℕ) : ℕ :=
  sorry

/-- The specific burning pattern for this problem -/
def candlePattern : BurningPattern :=
  { oddCentimeterTime := λ k => 10 * k,
    evenCentimeterTime := λ k => 15 * k }

/-- Theorem stating the remaining height of the candle after 80,000 seconds -/
theorem candle_height_after_80000_seconds :
  remainingHeight 150 candlePattern 80000 = 70 :=
sorry

end NUMINAMATH_CALUDE_candle_height_after_80000_seconds_l4024_402435


namespace NUMINAMATH_CALUDE_similar_polygons_ratio_l4024_402482

theorem similar_polygons_ratio (A₁ A₂ : ℝ) (s₁ s₂ : ℝ) :
  A₁ / A₂ = 9 / 4 →
  s₁ / s₂ = (A₁ / A₂).sqrt →
  s₁ / s₂ = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_similar_polygons_ratio_l4024_402482


namespace NUMINAMATH_CALUDE_journey_fraction_by_foot_l4024_402488

/-- Given a journey with a total distance of 24 km, where 1/4 of the distance
    is traveled by bus and 6 km is traveled by car, prove that the fraction
    of the distance traveled by foot is 1/2. -/
theorem journey_fraction_by_foot :
  ∀ (total_distance bus_fraction car_distance foot_distance : ℝ),
    total_distance = 24 →
    bus_fraction = 1/4 →
    car_distance = 6 →
    foot_distance = total_distance - (bus_fraction * total_distance + car_distance) →
    foot_distance / total_distance = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_journey_fraction_by_foot_l4024_402488


namespace NUMINAMATH_CALUDE_heart_club_probability_l4024_402463

/-- Represents a standard deck of 52 playing cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size : cards.card = 52)

/-- Represents the suit of a card -/
inductive Suit
| Hearts | Clubs | Diamonds | Spades

/-- The probability of drawing a heart first and a club second from a standard deck -/
def prob_heart_then_club (d : Deck) : ℚ :=
  (13 : ℚ) / 204

/-- Theorem stating the probability of drawing a heart first and a club second -/
theorem heart_club_probability (d : Deck) :
  prob_heart_then_club d = 13 / 204 := by
  sorry

end NUMINAMATH_CALUDE_heart_club_probability_l4024_402463


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l4024_402431

-- Define a binary number as a list of bits (0 or 1)
def BinaryNumber := List Nat

-- Define a function to convert a binary number to decimal
def binary_to_decimal (b : BinaryNumber) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + bit * 2^i) 0

-- State the theorem
theorem binary_101_equals_5 :
  binary_to_decimal [1, 0, 1] = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l4024_402431


namespace NUMINAMATH_CALUDE_line_equation_proof_l4024_402473

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem line_equation_proof (given_line : Line) (p : Point) :
  given_line.a = 1 ∧ given_line.b = -2 ∧ given_line.c = 3 →
  p.x = -1 ∧ p.y = 3 →
  ∃ (result_line : Line),
    result_line.a = 1 ∧ result_line.b = -2 ∧ result_line.c = 7 ∧
    pointOnLine p result_line ∧
    parallelLines given_line result_line := by
  sorry

end NUMINAMATH_CALUDE_line_equation_proof_l4024_402473


namespace NUMINAMATH_CALUDE_infinite_series_sum_l4024_402436

/-- The sum of the infinite series Σ(n=1 to ∞) [2^(2n) / (1 + 2^n + 2^(2n) + 2^(3n) + 2^(3n+1))] is equal to 1/25. -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (2^(2*n) : ℝ) / (1 + 2^n + 2^(2*n) + 2^(3*n) + 2^(3*n+1)) = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l4024_402436


namespace NUMINAMATH_CALUDE_last_number_is_one_seventh_l4024_402432

/-- A sequence of 100 non-zero real numbers where each number (except the first and last) 
    is the product of its neighbors, and the first number is 7 -/
def SpecialSequence (a : Fin 100 → ℝ) : Prop :=
  a 0 = 7 ∧ 
  (∀ i : Fin 98, a (i + 1) = a i * a (i + 2)) ∧
  (∀ i : Fin 100, a i ≠ 0)

/-- The last number in the sequence is 1/7 -/
theorem last_number_is_one_seventh (a : Fin 100 → ℝ) (h : SpecialSequence a) : 
  a 99 = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_last_number_is_one_seventh_l4024_402432


namespace NUMINAMATH_CALUDE_prism_volume_l4024_402425

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (side_area front_area bottom_area : ℝ) 
  (h_side : side_area = 20)
  (h_front : front_area = 12)
  (h_bottom : bottom_area = 8) :
  ∃ (a b c : ℝ), 
    a * b = side_area ∧ 
    b * c = front_area ∧ 
    a * c = bottom_area ∧ 
    a * b * c = 8 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l4024_402425


namespace NUMINAMATH_CALUDE_arrangements_a_middle_arrangements_a_b_not_adjacent_arrangements_a_b_not_ends_l4024_402485

-- Define the number of people
def n : ℕ := 5

-- Define the function for number of permutations
def permutations (n : ℕ) (r : ℕ) : ℕ := n.factorial / (n - r).factorial

-- Theorem 1: Person A in the middle
theorem arrangements_a_middle : permutations (n - 1) (n - 1) = 24 := by sorry

-- Theorem 2: Person A and B not adjacent
theorem arrangements_a_b_not_adjacent : 
  (permutations 3 3) * (permutations 4 2) = 72 := by sorry

-- Theorem 3: Person A and B not at ends
theorem arrangements_a_b_not_ends : 
  (permutations 3 2) * (permutations 3 3) = 36 := by sorry

end NUMINAMATH_CALUDE_arrangements_a_middle_arrangements_a_b_not_adjacent_arrangements_a_b_not_ends_l4024_402485


namespace NUMINAMATH_CALUDE_max_min_values_l4024_402455

theorem max_min_values (x y : ℝ) (h : x^2 + 4*y^2 = 4) :
  (∃ a b : ℝ, a^2 + 4*b^2 = 4 ∧ x^2 + 2*x*y + 4*y^2 ≤ a^2 + 2*a*b + 4*b^2) ∧
  (∃ c d : ℝ, c^2 + 4*d^2 = 4 ∧ x^2 + 2*x*y + 4*y^2 ≥ c^2 + 2*c*d + 4*d^2) ∧
  (∃ e f : ℝ, e^2 + 4*f^2 = 4 ∧ e^2 + 2*e*f + 4*f^2 = 6) ∧
  (∃ g h : ℝ, g^2 + 4*h^2 = 4 ∧ g^2 + 2*g*h + 4*h^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_l4024_402455


namespace NUMINAMATH_CALUDE_fifteen_percent_of_x_is_ninety_l4024_402483

theorem fifteen_percent_of_x_is_ninety (x : ℝ) : (15 / 100) * x = 90 → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_x_is_ninety_l4024_402483


namespace NUMINAMATH_CALUDE_penguin_sea_horse_difference_l4024_402408

/-- Given a ratio of sea horses to penguins and the number of sea horses,
    calculate the difference between the number of penguins and sea horses. -/
theorem penguin_sea_horse_difference 
  (ratio_sea_horses : ℕ) 
  (ratio_penguins : ℕ) 
  (num_sea_horses : ℕ) 
  (h1 : ratio_sea_horses = 5) 
  (h2 : ratio_penguins = 11) 
  (h3 : num_sea_horses = 70) :
  (ratio_penguins * (num_sea_horses / ratio_sea_horses)) - num_sea_horses = 84 :=
by
  sorry

#check penguin_sea_horse_difference

end NUMINAMATH_CALUDE_penguin_sea_horse_difference_l4024_402408


namespace NUMINAMATH_CALUDE_min_value_on_circle_l4024_402443

theorem min_value_on_circle (x y : ℝ) (h : x^2 + y^2 = 2) :
  (1 / (1 + x^2) + 1 / (1 + y^2)) ≥ 1 ∧
  (1 / (1 + x^2) + 1 / (1 + y^2) = 1 ↔ x^2 = 1 ∧ y^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_on_circle_l4024_402443


namespace NUMINAMATH_CALUDE_school_population_l4024_402444

theorem school_population (girls : ℕ) (boys : ℕ) (difference : ℕ) : 
  girls = 692 → difference = 458 → girls = boys + difference → girls + boys = 926 := by
  sorry

end NUMINAMATH_CALUDE_school_population_l4024_402444


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l4024_402498

theorem regular_polygon_interior_angle_sum :
  ∀ (n : ℕ) (exterior_angle : ℝ),
    n > 2 →
    exterior_angle = 24 →
    n * exterior_angle = 360 →
    (n - 2) * 180 = 2340 :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_sum_l4024_402498


namespace NUMINAMATH_CALUDE_exam_students_count_l4024_402427

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
  T = N * 80 →
  (T - 100) / (N - 5 : ℝ) = 95 →
  N = 25 :=
by
  sorry

end NUMINAMATH_CALUDE_exam_students_count_l4024_402427


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l4024_402420

theorem perfect_square_trinomial (a b : ℝ) : 
  (b - a = -7) → 
  (∃ k : ℝ, ∀ x : ℝ, 16 * x^2 + 144 * x + (a + b) = (k * x + (a + b) / (2 * k))^2) ↔ 
  (a = 165.5 ∧ b = 158.5) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l4024_402420


namespace NUMINAMATH_CALUDE_remainder_of_1493824_div_4_l4024_402476

theorem remainder_of_1493824_div_4 : 1493824 % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1493824_div_4_l4024_402476


namespace NUMINAMATH_CALUDE_hamburger_cost_calculation_l4024_402415

/-- Represents the cost calculation for hamburgers with higher quality meat -/
theorem hamburger_cost_calculation 
  (original_meat_pounds : ℝ) 
  (original_cost_per_pound : ℝ) 
  (original_hamburger_count : ℝ) 
  (new_hamburger_count : ℝ) 
  (cost_increase_percentage : ℝ) :
  original_meat_pounds = 5 →
  original_cost_per_pound = 4 →
  original_hamburger_count = 10 →
  new_hamburger_count = 30 →
  cost_increase_percentage = 0.25 →
  (original_meat_pounds / original_hamburger_count) * new_hamburger_count * 
  (original_cost_per_pound * (1 + cost_increase_percentage)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_hamburger_cost_calculation_l4024_402415


namespace NUMINAMATH_CALUDE_same_solution_implies_zero_power_l4024_402499

theorem same_solution_implies_zero_power (a b : ℝ) :
  (∃ x y : ℝ, 4*x + 3*y = 11 ∧ a*x + b*y = -2 ∧ 2*x - y = 3 ∧ b*x - a*y = 6) →
  (a + b)^2023 = 0 := by
sorry

end NUMINAMATH_CALUDE_same_solution_implies_zero_power_l4024_402499


namespace NUMINAMATH_CALUDE_chess_games_ratio_l4024_402410

theorem chess_games_ratio (total_games won_games : ℕ) 
  (h1 : total_games = 44)
  (h2 : won_games = 16) :
  let lost_games := total_games - won_games
  Nat.gcd lost_games won_games = 4 ∧ 
  lost_games / 4 = 7 ∧ 
  won_games / 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_chess_games_ratio_l4024_402410


namespace NUMINAMATH_CALUDE_linear_function_sum_l4024_402402

/-- A linear function f with specific properties -/
def f (x : ℝ) : ℝ := sorry

/-- The sum of f(2), f(4), ..., f(2n) -/
def sum_f (n : ℕ) : ℝ := sorry

theorem linear_function_sum :
  (f 0 = 1) →
  (∃ r : ℝ, f 1 * r = f 4 ∧ f 4 * r = f 13) →
  (∀ x y : ℝ, f (x + y) = f x + f y - f 0) →
  ∀ n : ℕ, sum_f n = n * (2 * n + 3) :=
sorry

end NUMINAMATH_CALUDE_linear_function_sum_l4024_402402


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_18_12_l4024_402419

/-- Perimeter of a parallelogram -/
def parallelogram_perimeter (side1 : ℝ) (side2 : ℝ) : ℝ :=
  2 * (side1 + side2)

/-- Theorem: The perimeter of a parallelogram with sides 18 cm and 12 cm is 60 cm -/
theorem parallelogram_perimeter_18_12 :
  parallelogram_perimeter 18 12 = 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_18_12_l4024_402419


namespace NUMINAMATH_CALUDE_circle_equation_proof_l4024_402464

/-- Given a circle with center (1, 1) intersecting the line x + y = 4 to form a chord of length 2√3,
    prove that the equation of the circle is (x-1)² + (y-1)² = 5. -/
theorem circle_equation_proof (x y : ℝ) : 
  let center : ℝ × ℝ := (1, 1)
  let line_equation := x + y = 4
  let chord_length : ℝ := 2 * Real.sqrt 3
  true → (x - 1)^2 + (y - 1)^2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l4024_402464


namespace NUMINAMATH_CALUDE_intersection_point_of_three_lines_l4024_402453

theorem intersection_point_of_three_lines (k b : ℝ) :
  (∀ x y : ℝ, (y = k * x + b) ∧ (y = 2 * k * x + 2 * b) ∧ (y = b * x + k)) →
  (k ≠ b) →
  (∃! p : ℝ × ℝ, 
    (p.2 = k * p.1 + b) ∧ 
    (p.2 = 2 * k * p.1 + 2 * b) ∧ 
    (p.2 = b * p.1 + k) ∧
    p = (1, 0)) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_of_three_lines_l4024_402453


namespace NUMINAMATH_CALUDE_parabola_directrix_l4024_402449

/-- Given a parabola with equation x = -2y^2, its directrix has equation x = 1/8 -/
theorem parabola_directrix (y : ℝ) : 
  (∃ x : ℝ, x = -2 * y^2) → 
  (∃ x : ℝ, x = 1/8 ∧ ∀ y : ℝ, (y, x) ∈ {p : ℝ × ℝ | p.1 = 1/8}) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l4024_402449


namespace NUMINAMATH_CALUDE_triangle_area_l4024_402490

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    where b = 1, c = √3, and ∠C = 2π/3, prove that its area is √3/4 -/
theorem triangle_area (a b c : ℝ) (A B C : ℝ) : 
  b = 1 → 
  c = Real.sqrt 3 → 
  C = 2 * Real.pi / 3 → 
  (1/2) * a * b * Real.sin C = Real.sqrt 3 / 4 := by
sorry


end NUMINAMATH_CALUDE_triangle_area_l4024_402490


namespace NUMINAMATH_CALUDE_ohms_law_application_l4024_402446

/-- Given a constant voltage U, current I inversely proportional to resistance R,
    prove that for I1 = 4A, R1 = 10Ω, and I2 = 5A, the value of R2 is 8Ω. -/
theorem ohms_law_application (U : ℝ) (I1 I2 R1 R2 : ℝ) : 
  U > 0 →  -- Voltage is positive
  I1 > 0 →  -- Current is positive
  I2 > 0 →  -- Current is positive
  R1 > 0 →  -- Resistance is positive
  R2 > 0 →  -- Resistance is positive
  (∀ I R, U = I * R) →  -- Ohm's law: U = IR
  I1 = 4 →
  R1 = 10 →
  I2 = 5 →
  R2 = 8 := by
sorry

end NUMINAMATH_CALUDE_ohms_law_application_l4024_402446


namespace NUMINAMATH_CALUDE_six_people_arrangement_l4024_402489

/-- The number of arrangements with A at the edge -/
def edge_arrangements : ℕ := 4 * 3 * 24

/-- The number of arrangements with A in the middle -/
def middle_arrangements : ℕ := 2 * 2 * 24

/-- The total number of valid arrangements -/
def total_arrangements : ℕ := edge_arrangements + middle_arrangements

theorem six_people_arrangement :
  total_arrangements = 384 :=
sorry

end NUMINAMATH_CALUDE_six_people_arrangement_l4024_402489


namespace NUMINAMATH_CALUDE_sunset_time_l4024_402440

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents the length of a time period in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

def sunrise : Time := { hours := 6, minutes := 12 }
def daylightLength : Duration := { hours := 12, minutes := 36 }

theorem sunset_time :
  addTime sunrise daylightLength = { hours := 18, minutes := 48 } := by
  sorry

end NUMINAMATH_CALUDE_sunset_time_l4024_402440


namespace NUMINAMATH_CALUDE_max_additional_plates_achievable_additional_plates_l4024_402469

/-- Represents the sets of symbols for car plates in Rivertown -/
structure CarPlateSymbols where
  firstLetters : Finset Char
  secondLetters : Finset Char
  digits : Finset Char

/-- Calculates the total number of possible car plates -/
def totalPlates (symbols : CarPlateSymbols) : ℕ :=
  symbols.firstLetters.card * symbols.secondLetters.card * symbols.digits.card

/-- The initial configuration of car plate symbols -/
def initialSymbols : CarPlateSymbols :=
  { firstLetters := {'A', 'B', 'G', 'H', 'T'},
    secondLetters := {'E', 'I', 'O', 'U'},
    digits := {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'} }

/-- Represents the addition of new symbols -/
structure NewSymbols where
  newLetters : ℕ
  newDigits : ℕ

/-- The new symbols to be added -/
def addedSymbols : NewSymbols :=
  { newLetters := 2,
    newDigits := 1 }

/-- Theorem: The maximum number of additional car plates after adding new symbols is 130 -/
theorem max_additional_plates :
  ∀ (newDistribution : CarPlateSymbols),
    (newDistribution.firstLetters.card + newDistribution.secondLetters.card = 
      initialSymbols.firstLetters.card + initialSymbols.secondLetters.card + addedSymbols.newLetters) →
    (newDistribution.digits.card = initialSymbols.digits.card + addedSymbols.newDigits) →
    totalPlates newDistribution - totalPlates initialSymbols ≤ 130 :=
by sorry

/-- Theorem: There exists a distribution that achieves 130 additional plates -/
theorem achievable_additional_plates :
  ∃ (newDistribution : CarPlateSymbols),
    (newDistribution.firstLetters.card + newDistribution.secondLetters.card = 
      initialSymbols.firstLetters.card + initialSymbols.secondLetters.card + addedSymbols.newLetters) ∧
    (newDistribution.digits.card = initialSymbols.digits.card + addedSymbols.newDigits) ∧
    totalPlates newDistribution - totalPlates initialSymbols = 130 :=
by sorry

end NUMINAMATH_CALUDE_max_additional_plates_achievable_additional_plates_l4024_402469


namespace NUMINAMATH_CALUDE_problem_solution_l4024_402447

theorem problem_solution (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47)
  (eq2 : 8 * a + 2 * b = 50) :
  3 * a + 3 * b = 73 / 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4024_402447


namespace NUMINAMATH_CALUDE_michaels_initial_money_proof_l4024_402491

/-- Michael's initial amount of money -/
def michaels_initial_money : ℕ := 152

/-- Amount Michael's brother had initially -/
def brothers_initial_money : ℕ := 17

/-- Amount spent on candy -/
def candy_cost : ℕ := 3

/-- Amount Michael's brother has left after buying candy -/
def brothers_remaining_money : ℕ := 35

theorem michaels_initial_money_proof :
  michaels_initial_money = 
    2 * (brothers_remaining_money + candy_cost + brothers_initial_money - brothers_initial_money) :=
by sorry

end NUMINAMATH_CALUDE_michaels_initial_money_proof_l4024_402491


namespace NUMINAMATH_CALUDE_hcf_problem_l4024_402484

theorem hcf_problem (a b : ℕ+) (h1 : a * b = 62216) (h2 : Nat.lcm a b = 2828) :
  Nat.gcd a b = 22 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l4024_402484


namespace NUMINAMATH_CALUDE_age_sum_l4024_402492

theorem age_sum (patrick michael monica : ℕ) 
  (h1 : 3 * michael = 5 * patrick)
  (h2 : 3 * monica = 5 * michael)
  (h3 : monica - patrick = 32) : 
  patrick + michael + monica = 98 := by
sorry

end NUMINAMATH_CALUDE_age_sum_l4024_402492


namespace NUMINAMATH_CALUDE_today_is_wednesday_l4024_402497

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents the number of days from Sunday -/
def daysFromSunday (d : DayOfWeek) : Nat :=
  match d with
  | .Sunday => 0
  | .Monday => 1
  | .Tuesday => 2
  | .Wednesday => 3
  | .Thursday => 4
  | .Friday => 5
  | .Saturday => 6

/-- Adds a number of days to a given day, wrapping around the week -/
def addDays (d : DayOfWeek) (n : Int) : DayOfWeek :=
  match (daysFromSunday d + n % 7 + 7) % 7 with
  | 0 => .Sunday
  | 1 => .Monday
  | 2 => .Tuesday
  | 3 => .Wednesday
  | 4 => .Thursday
  | 5 => .Friday
  | _ => .Saturday

/-- The condition given in the problem -/
def satisfiesCondition (today : DayOfWeek) : Prop :=
  let dayAfterTomorrow := addDays today 2
  let yesterday := addDays today (-1)
  let tomorrow := addDays today 1
  daysFromSunday (addDays dayAfterTomorrow 3) = daysFromSunday (addDays yesterday 2)

/-- The theorem to be proved -/
theorem today_is_wednesday : 
  ∃ (d : DayOfWeek), satisfiesCondition d ∧ d = DayOfWeek.Wednesday :=
sorry

end NUMINAMATH_CALUDE_today_is_wednesday_l4024_402497


namespace NUMINAMATH_CALUDE_sophies_spend_is_72_80_l4024_402466

/-- The total amount Sophie spends on her purchases -/
def sophies_total_spend : ℚ :=
  let cupcakes := 5 * 2
  let doughnuts := 6 * 1
  let apple_pie := 4 * 2
  let cookies := 15 * 0.6
  let chocolate_bars := 8 * 1.5
  let soda := 12 * 1.2
  let gum := 3 * 0.8
  let chips := 10 * 1.1
  cupcakes + doughnuts + apple_pie + cookies + chocolate_bars + soda + gum + chips

/-- Theorem stating that Sophie's total spend is $72.80 -/
theorem sophies_spend_is_72_80 : sophies_total_spend = 72.8 := by
  sorry

end NUMINAMATH_CALUDE_sophies_spend_is_72_80_l4024_402466


namespace NUMINAMATH_CALUDE_solution_set_f_max_value_f_range_of_m_l4024_402479

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 3| - |x + 5|

-- Theorem for the solution set of f(x) ≥ 2
theorem solution_set_f (x : ℝ) : f x ≥ 2 ↔ x ≤ -2 := by sorry

-- Theorem for the maximum value of f(x)
theorem max_value_f : ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, f x ≤ M := by sorry

-- Theorem for the range of m
theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + m ≤ 8) ↔ m ≤ 9 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_max_value_f_range_of_m_l4024_402479


namespace NUMINAMATH_CALUDE_tournament_handshakes_l4024_402429

/-- Represents a women's doubles tennis tournament --/
structure Tournament where
  numTeams : Nat
  playersPerTeam : Nat
  handshakesPerPlayer : Nat

/-- Calculates the total number of handshakes in the tournament --/
def totalHandshakes (t : Tournament) : Nat :=
  (t.numTeams * t.playersPerTeam * t.handshakesPerPlayer) / 2

/-- Theorem stating that the specific tournament configuration results in 24 handshakes --/
theorem tournament_handshakes :
  ∃ (t : Tournament),
    t.numTeams = 4 ∧
    t.playersPerTeam = 2 ∧
    t.handshakesPerPlayer = 6 ∧
    totalHandshakes t = 24 := by
  sorry


end NUMINAMATH_CALUDE_tournament_handshakes_l4024_402429


namespace NUMINAMATH_CALUDE_tv_weekly_cost_l4024_402493

/-- Calculate the weekly cost of running a TV -/
theorem tv_weekly_cost 
  (watt_per_hour : ℕ) 
  (hours_per_day : ℕ) 
  (cents_per_kwh : ℕ) 
  (h1 : watt_per_hour = 125)
  (h2 : hours_per_day = 4)
  (h3 : cents_per_kwh = 14) : 
  (watt_per_hour * hours_per_day * 7 * cents_per_kwh : ℚ) / 1000 = 49 := by
sorry

end NUMINAMATH_CALUDE_tv_weekly_cost_l4024_402493


namespace NUMINAMATH_CALUDE_max_candy_leftover_l4024_402460

theorem max_candy_leftover (x : ℕ+) : 
  ∃ (q r : ℕ), x = 12 * q + r ∧ r < 12 ∧ r ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_candy_leftover_l4024_402460


namespace NUMINAMATH_CALUDE_trig_simplification_l4024_402400

theorem trig_simplification :
  (Real.sin (30 * π / 180) + Real.sin (60 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (60 * π / 180)) =
  Real.tan (45 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_trig_simplification_l4024_402400


namespace NUMINAMATH_CALUDE_sum_of_combinations_l4024_402462

theorem sum_of_combinations : Nat.choose 10 3 + Nat.choose 10 4 = 330 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_combinations_l4024_402462


namespace NUMINAMATH_CALUDE_negation_of_proposition_negation_of_greater_than_sin_l4024_402445

theorem negation_of_proposition (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_greater_than_sin :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_negation_of_greater_than_sin_l4024_402445


namespace NUMINAMATH_CALUDE_age_problem_l4024_402486

theorem age_problem (p q : ℕ) 
  (h1 : p - 6 = (q - 6) / 2)  -- 6 years ago, p was half of q in age
  (h2 : p * 4 = q * 3)        -- The ratio of their present ages is 3:4
  : p + q = 21 := by           -- The total of their present ages is 21
sorry

end NUMINAMATH_CALUDE_age_problem_l4024_402486


namespace NUMINAMATH_CALUDE_tens_digit_of_6_pow_22_l4024_402480

/-- The tens digit of 6^n -/
def tens_digit_of_6_pow (n : ℕ) : ℕ :=
  match n % 5 with
  | 0 => 6
  | 1 => 3
  | 2 => 1
  | 3 => 9
  | 4 => 7
  | _ => 0  -- This case should never occur

theorem tens_digit_of_6_pow_22 : tens_digit_of_6_pow 22 = 3 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_6_pow_22_l4024_402480


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_specific_coefficients_l4024_402459

theorem infinite_solutions_imply_specific_coefficients :
  ∀ (a b : ℝ),
  (∀ x : ℝ, a * (x - 3) + b * (3 * x + 1) = 5 * (x + 1)) →
  (a = -1 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_specific_coefficients_l4024_402459


namespace NUMINAMATH_CALUDE_orchids_in_vase_orchids_count_is_two_l4024_402414

theorem orchids_in_vase (initial_roses : ℕ) (initial_orchids : ℕ) 
  (current_roses : ℕ) (rose_orchid_difference : ℕ) : ℕ :=
  let current_orchids := current_roses - rose_orchid_difference
  current_orchids

#check orchids_in_vase 5 3 12 10

theorem orchids_count_is_two :
  orchids_in_vase 5 3 12 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_orchids_in_vase_orchids_count_is_two_l4024_402414


namespace NUMINAMATH_CALUDE_remaining_money_l4024_402450

def gift_amount : ℕ := 200
def cassette_cost : ℕ := 15
def num_cassettes : ℕ := 3
def headphones_cost : ℕ := 55
def vinyl_cost : ℕ := 35
def poster_cost : ℕ := 45

def total_cost : ℕ := cassette_cost * num_cassettes + headphones_cost + vinyl_cost + poster_cost

theorem remaining_money :
  gift_amount - total_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l4024_402450


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l4024_402417

theorem quadratic_inequality_solution (x : ℝ) :
  -3 * x^2 + 8 * x + 3 > 0 ↔ x < -1/3 ∨ x > 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l4024_402417


namespace NUMINAMATH_CALUDE_quadratic_root_property_l4024_402457

theorem quadratic_root_property (n : ℝ) : 
  (∃ x₁ x₂ : ℝ, (x₁^2 - 3*x₁ + n = 0) ∧ (x₂^2 - 3*x₂ + n = 0) ∧ (x₁ + x₂ - 2 = x₁ * x₂)) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_property_l4024_402457


namespace NUMINAMATH_CALUDE_negation_of_universal_statement_l4024_402475

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, Real.exp x > Real.log x) ↔ (∃ x₀ : ℝ, Real.exp x₀ ≤ Real.log x₀) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_statement_l4024_402475


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l4024_402423

-- Define the hyperbola and its properties
def Hyperbola (a : ℝ) : Prop :=
  a > 0 ∧ ∃ (x y : ℝ), x^2 / a^2 - y^2 / 5 = 1

-- Define the asymptote
def Asymptote (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y = (Real.sqrt 5 / 2) * x

-- Define eccentricity
def Eccentricity (e : ℝ) (a : ℝ) : Prop :=
  e = Real.sqrt (a^2 + 5) / a

-- Theorem statement
theorem hyperbola_eccentricity (a : ℝ) :
  Hyperbola a → Asymptote a → Eccentricity (3/2) a := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l4024_402423


namespace NUMINAMATH_CALUDE_observation_count_l4024_402434

theorem observation_count (original_mean : ℝ) (incorrect_value : ℝ) (correct_value : ℝ) (new_mean : ℝ) : 
  original_mean = 36 →
  incorrect_value = 23 →
  correct_value = 30 →
  new_mean = 36.5 →
  ∃ n : ℕ, n > 0 ∧ (n : ℝ) * new_mean = n * original_mean + (correct_value - incorrect_value) ∧ n = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_observation_count_l4024_402434


namespace NUMINAMATH_CALUDE_cos_difference_value_l4024_402496

theorem cos_difference_value (A B : ℝ) 
  (h1 : Real.sin A + Real.sin B = 3/2) 
  (h2 : Real.cos A + Real.cos B = 1) : 
  Real.cos (A - B) = 5/8 := by sorry

end NUMINAMATH_CALUDE_cos_difference_value_l4024_402496


namespace NUMINAMATH_CALUDE_f_properties_l4024_402409

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 6 + Real.cos x ^ 6

theorem f_properties :
  (∀ x, f x ∈ Set.Icc (1/4 : ℝ) 1) ∧
  (∀ ε > 0, ∃ p ∈ Set.Ioo 0 ε, ∀ x, f (x + p) = f x) ∧
  (∀ k : ℤ, ∀ x, f (k * Real.pi / 4 - x) = f (k * Real.pi / 4 + x)) ∧
  (∀ k : ℤ, f (Real.pi / 8 + k * Real.pi / 4) = 5/8) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l4024_402409


namespace NUMINAMATH_CALUDE_composite_with_large_smallest_prime_divisor_l4024_402438

theorem composite_with_large_smallest_prime_divisor 
  (N : ℕ) 
  (h_composite : ¬ Prime N) 
  (h_smallest_divisor : ∀ p : ℕ, Prime p → p ∣ N → p > N^(1/3)) : 
  ∃ p q : ℕ, Prime p ∧ Prime q ∧ N = p * q :=
sorry

end NUMINAMATH_CALUDE_composite_with_large_smallest_prime_divisor_l4024_402438


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l4024_402407

open Set

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

theorem intersection_complement_equality :
  A ∩ (Set.univ \ B) = {x | -1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l4024_402407


namespace NUMINAMATH_CALUDE_trululu_nonexistence_l4024_402411

structure Individual where
  statement : Prop

def is_weekday (day : Nat) : Prop :=
  1 ≤ day ∧ day ≤ 5

def Barmaglot_lies (day : Nat) : Prop :=
  1 ≤ day ∧ day ≤ 3

theorem trululu_nonexistence (day : Nat) 
  (h1 : is_weekday day)
  (h2 : ∃ (i1 i2 : Individual), i1.statement = (∃ Trululu : Type, Nonempty Trululu) ∧ i2.statement = True)
  (h3 : ∀ (i : Individual), i.statement = True → i.statement)
  (h4 : Barmaglot_lies day → ¬(∃ Trululu : Type, Nonempty Trululu))
  (h5 : ¬(Barmaglot_lies day))
  : ¬(∃ Trululu : Type, Nonempty Trululu) := by
  sorry

#check trululu_nonexistence

end NUMINAMATH_CALUDE_trululu_nonexistence_l4024_402411


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l4024_402403

theorem complex_magnitude_problem : ∃ (T : ℂ), 
  T = (1 + Complex.I)^19 + (1 + Complex.I)^19 - (1 - Complex.I)^19 ∧ 
  Complex.abs T = Real.sqrt 5 * 2^(19/2) := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l4024_402403


namespace NUMINAMATH_CALUDE_K_on_circle_S₂_l4024_402424

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def S : Circle := { center := (0, 0), radius := 2 }
def S₁ : Circle := { center := (1, 0), radius := 1 }
def S₂ : Circle := { center := (3, 0), radius := 1 }

def B : ℝ × ℝ := (2, 0)
def A : ℝ × ℝ := (2, 0)

-- Define the intersection point K
def K : ℝ × ℝ := sorry

-- Define the properties of the circles
def S₁_tangent_to_S : Prop :=
  (S₁.center.1 - S.center.1)^2 + (S₁.center.2 - S.center.2)^2 = (S.radius - S₁.radius)^2

def S₂_tangent_to_S₁ : Prop :=
  (S₂.center.1 - S₁.center.1)^2 + (S₂.center.2 - S₁.center.2)^2 = (S₁.radius + S₂.radius)^2

def S₂_not_tangent_to_S : Prop :=
  (S₂.center.1 - S.center.1)^2 + (S₂.center.2 - S.center.2)^2 ≠ (S.radius - S₂.radius)^2

def K_on_line_AB : Prop :=
  (K.2 - A.2) * (B.1 - A.1) = (K.1 - A.1) * (B.2 - A.2)

def K_on_circle_S : Prop :=
  (K.1 - S.center.1)^2 + (K.2 - S.center.2)^2 = S.radius^2

-- Theorem to prove
theorem K_on_circle_S₂ (h1 : S₁_tangent_to_S) (h2 : S₂_tangent_to_S₁) 
    (h3 : S₂_not_tangent_to_S) (h4 : K_on_line_AB) (h5 : K_on_circle_S) :
  (K.1 - S₂.center.1)^2 + (K.2 - S₂.center.2)^2 = S₂.radius^2 := by
  sorry

end NUMINAMATH_CALUDE_K_on_circle_S₂_l4024_402424


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_theorem_l4024_402456

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Converts a three-digit number represented by individual digits to an integer -/
def threeDigitToInt (a b c : Digit) : ℕ := 100 * a.val + 10 * b.val + c.val

/-- Converts a repeating decimal of the form 0.abab... to a rational number -/
def abRepeatingToRational (a b : Digit) : ℚ := (10 * a.val + b.val : ℚ) / 99

/-- Converts a repeating decimal of the form 0.abcabc... to a rational number -/
def abcRepeatingToRational (a b c : Digit) : ℚ := (100 * a.val + 10 * b.val + c.val : ℚ) / 999

theorem repeating_decimal_sum_theorem (a b c : Digit) :
  abRepeatingToRational a b + abcRepeatingToRational a b c = 17 / 37 →
  threeDigitToInt a b c = 270 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_theorem_l4024_402456


namespace NUMINAMATH_CALUDE_point_on_line_l4024_402442

/-- Given two points (m, n) and (m + p, n + 9) on the line x = y/3 - 2/5, prove that p = 3 -/
theorem point_on_line (m n p : ℝ) : 
  (m = n / 3 - 2 / 5) →
  (m + p = (n + 9) / 3 - 2 / 5) →
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l4024_402442


namespace NUMINAMATH_CALUDE_line_through_points_with_45_degree_angle_l4024_402487

/-- A line passes through points A(m,2) and B(-m,2m-1) with an inclination angle of 45° -/
theorem line_through_points_with_45_degree_angle (m : ℝ) : 
  (∃ (line : Set (ℝ × ℝ)), 
    (m, 2) ∈ line ∧ 
    (-m, 2*m - 1) ∈ line ∧ 
    (∀ (x y : ℝ), (x, y) ∈ line → (y - 2) / (x - m) = 1)) → 
  m = 3/4 := by
sorry

end NUMINAMATH_CALUDE_line_through_points_with_45_degree_angle_l4024_402487


namespace NUMINAMATH_CALUDE_train_length_proof_l4024_402412

/-- Given a train with a speed of 40 km/hr that crosses a post in 18 seconds,
    prove that its length is approximately 200 meters. -/
theorem train_length_proof (speed : ℝ) (time : ℝ) (length : ℝ) : 
  speed = 40 → -- speed in km/hr
  time = 18 → -- time in seconds
  length = speed * (1000 / 3600) * time →
  ∃ ε > 0, |length - 200| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l4024_402412


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l4024_402422

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | (x^2 + a*x + b)*(x - 1) = 0}

-- Define the theorem
theorem sum_of_a_and_b_is_one 
  (B C : Set ℝ) 
  (a b : ℝ) 
  (h1 : A a b ∩ B = {1, 2})
  (h2 : A a b ∩ (C ∪ B) = {3}) :
  a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_is_one_l4024_402422


namespace NUMINAMATH_CALUDE_first_part_value_l4024_402437

theorem first_part_value (x y : ℝ) 
  (sum_constraint : x + y = 36)
  (weighted_sum_constraint : 8 * x + 3 * y = 203) :
  x = 19 := by
sorry

end NUMINAMATH_CALUDE_first_part_value_l4024_402437


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l4024_402426

theorem gcd_digits_bound (a b : ℕ) : 
  1000000 ≤ a ∧ a < 10000000 ∧ 
  1000000 ≤ b ∧ b < 10000000 ∧ 
  10000000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 100000000000 →
  Nat.gcd a b < 10000 := by
sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l4024_402426


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l4024_402494

theorem quadratic_equation_roots (p q : ℝ) (a b : ℝ) : 
  (a^2 + p*a + q = 0) → 
  (b^2 + p*b + q = 0) → 
  ∃ y₁ y₂ : ℝ, 
    (y₁ = (a+b)^2 ∧ y₂ = (a-b)^2) ∧ 
    (y₁^2 - 2*(p^2 - 2*q)*y₁ + (p^4 - 4*q*p^2) = 0) ∧
    (y₂^2 - 2*(p^2 - 2*q)*y₂ + (p^4 - 4*q*p^2) = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l4024_402494
