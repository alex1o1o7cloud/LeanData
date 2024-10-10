import Mathlib

namespace surface_area_of_six_cubes_l2864_286406

/-- Represents the configuration of 6 cubes fastened together -/
structure CubeConfiguration where
  numCubes : Nat
  edgeLength : ℝ
  numConnections : Nat

/-- Calculates the total surface area of the cube configuration -/
def totalSurfaceArea (config : CubeConfiguration) : ℝ :=
  (config.numCubes * 6 - 2 * config.numConnections) * config.edgeLength ^ 2

/-- Theorem stating that the total surface area of the given configuration is 26 square units -/
theorem surface_area_of_six_cubes :
  ∀ (config : CubeConfiguration),
    config.numCubes = 6 ∧
    config.edgeLength = 1 ∧
    config.numConnections = 10 →
    totalSurfaceArea config = 26 := by
  sorry

end surface_area_of_six_cubes_l2864_286406


namespace sugar_amount_l2864_286473

/-- The total amount of sugar the store owner started with, given the conditions. -/
theorem sugar_amount (num_packs : ℕ) (pack_weight : ℕ) (remaining_sugar : ℕ) 
  (h1 : num_packs = 12)
  (h2 : pack_weight = 250)
  (h3 : remaining_sugar = 20) :
  num_packs * pack_weight + remaining_sugar = 3020 :=
by sorry

end sugar_amount_l2864_286473


namespace coins_collected_in_hours_2_3_l2864_286427

/-- Represents the number of coins collected in each hour -/
structure CoinCollection where
  hour1 : ℕ
  hour2_3 : ℕ
  hour4 : ℕ
  given_away : ℕ
  total : ℕ

/-- The coin collection scenario for Joanne -/
def joannes_collection : CoinCollection where
  hour1 := 15
  hour2_3 := 0  -- This is what we need to prove
  hour4 := 50
  given_away := 15
  total := 120

/-- Theorem stating that Joanne collected 70 coins in hours 2 and 3 -/
theorem coins_collected_in_hours_2_3 :
  joannes_collection.hour2_3 = 70 :=
by sorry

end coins_collected_in_hours_2_3_l2864_286427


namespace smallest_number_with_gcd_six_l2864_286414

theorem smallest_number_with_gcd_six : ∃ (n : ℕ), 
  (70 ≤ n ∧ n ≤ 90) ∧ 
  Nat.gcd n 24 = 6 ∧ 
  (∀ m, (70 ≤ m ∧ m < n) → Nat.gcd m 24 ≠ 6) ∧
  n = 78 := by
sorry

end smallest_number_with_gcd_six_l2864_286414


namespace intersection_area_greater_than_half_l2864_286417

/-- Two identical rectangles with sides a and b -/
structure Rectangle where
  a : ℝ
  b : ℝ
  pos_a : 0 < a
  pos_b : 0 < b

/-- The configuration of two intersecting rectangles -/
structure IntersectingRectangles where
  rect : Rectangle
  intersection_points : ℕ
  eight_intersections : intersection_points = 8

/-- The area of intersection of two rectangles -/
def intersectionArea (ir : IntersectingRectangles) : ℝ := sorry

/-- The area of a single rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.a * r.b

/-- Theorem: The area of intersection is greater than half the area of each rectangle -/
theorem intersection_area_greater_than_half (ir : IntersectingRectangles) :
  intersectionArea ir > (1/2) * rectangleArea ir.rect :=
sorry

end intersection_area_greater_than_half_l2864_286417


namespace geometric_series_sum_l2864_286443

/-- The sum of the first n terms of a geometric series with first term a and common ratio r -/
def geometricSum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- Theorem: The sum of the first nine terms of a geometric series with first term 1/3 and common ratio 2/3 is 19171/19683 -/
theorem geometric_series_sum :
  geometricSum (1/3) (2/3) 9 = 19171/19683 := by
  sorry

end geometric_series_sum_l2864_286443


namespace square_number_divisible_by_5_between_20_and_110_l2864_286461

theorem square_number_divisible_by_5_between_20_and_110 (y : ℕ) :
  (∃ n : ℕ, y = n^2) →
  y % 5 = 0 →
  20 < y →
  y < 110 →
  (y = 25 ∨ y = 100) :=
by sorry

end square_number_divisible_by_5_between_20_and_110_l2864_286461


namespace marbles_per_friend_l2864_286497

theorem marbles_per_friend (total_marbles : ℕ) (num_friends : ℕ) 
  (h1 : total_marbles = 72) (h2 : num_friends = 9) :
  total_marbles / num_friends = 8 :=
by sorry

end marbles_per_friend_l2864_286497


namespace stan_boxes_count_l2864_286488

theorem stan_boxes_count (john jules joseph stan : ℕ) : 
  john = (120 * jules) / 100 →
  jules = joseph + 5 →
  joseph = (20 * stan) / 100 →
  john = 30 →
  stan = 100 := by
sorry

end stan_boxes_count_l2864_286488


namespace reflect_y_of_neg_five_two_l2864_286454

/-- Reflects a point across the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

theorem reflect_y_of_neg_five_two :
  reflect_y (-5, 2) = (5, 2) := by sorry

end reflect_y_of_neg_five_two_l2864_286454


namespace f_positive_iff_in_intervals_l2864_286423

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 3)

theorem f_positive_iff_in_intervals (x : ℝ) : 
  f x > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 3 :=
sorry

end f_positive_iff_in_intervals_l2864_286423


namespace smallest_n_satisfying_conditions_l2864_286419

theorem smallest_n_satisfying_conditions : ∃ (n : ℕ), 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧ 
  n % 8 = 5 ∧ 
  (∀ (m : ℕ), m > 20 ∧ m % 6 = 4 ∧ m % 7 = 3 ∧ m % 8 = 5 → m ≥ n) ∧
  n = 136 := by
sorry

end smallest_n_satisfying_conditions_l2864_286419


namespace division_problem_l2864_286462

theorem division_problem : 
  ∃ (q r : ℕ), 253 = (15 + 13 * 3 - 5) * q + r ∧ r < (15 + 13 * 3 - 5) ∧ q = 5 ∧ r = 8 := by
  sorry

end division_problem_l2864_286462


namespace notebook_cost_l2864_286438

theorem notebook_cost (total_students : ℕ) 
  (buyers : ℕ) 
  (notebooks_per_buyer : ℕ) 
  (notebook_cost : ℕ) 
  (total_cost : ℕ) :
  total_students = 40 →
  buyers > total_students / 2 →
  notebooks_per_buyer > 2 →
  notebook_cost > 2 * notebooks_per_buyer →
  buyers * notebooks_per_buyer * notebook_cost = total_cost →
  total_cost = 4515 →
  notebook_cost = 35 := by
sorry

end notebook_cost_l2864_286438


namespace equal_roots_quadratic_l2864_286453

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + m = 0 → y = x) → 
  m = 4 := by
  sorry

end equal_roots_quadratic_l2864_286453


namespace recipe_soap_amount_l2864_286426

/-- Given a container capacity, ounces per cup, and total soap amount, 
    calculate the amount of soap per cup of water. -/
def soapPerCup (containerCapacity : ℚ) (ouncesPerCup : ℚ) (totalSoap : ℚ) : ℚ :=
  totalSoap / (containerCapacity / ouncesPerCup)

/-- Prove that the recipe calls for 3 tablespoons of soap per cup of water. -/
theorem recipe_soap_amount :
  soapPerCup 40 8 15 = 3 := by
  sorry

end recipe_soap_amount_l2864_286426


namespace units_digit_of_product_l2864_286449

theorem units_digit_of_product (n : ℕ) : 
  (2^2021 * 5^2022 * 7^2023) % 10 = 0 :=
sorry

end units_digit_of_product_l2864_286449


namespace marks_initial_trees_l2864_286401

theorem marks_initial_trees (total_after_planting : ℕ) (trees_to_plant : ℕ) : 
  total_after_planting = 25 → trees_to_plant = 12 → total_after_planting - trees_to_plant = 13 := by
  sorry

end marks_initial_trees_l2864_286401


namespace negation_of_universal_proposition_l2864_286475

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^2 - x + 1 ≥ 0) ↔ (∃ x₀ : ℝ, 2 * x₀^2 - x₀ + 1 < 0) := by
  sorry

end negation_of_universal_proposition_l2864_286475


namespace binomial_arithmetic_sequence_l2864_286429

theorem binomial_arithmetic_sequence (n k : ℕ) :
  (∃ (u : ℕ), u > 2 ∧ n = u^2 - 2 ∧ (k = u.choose 2 - 1 ∨ k = (u + 1).choose 2 - 1)) ↔
  (Nat.choose n (k - 1) - 2 * Nat.choose n k + Nat.choose n (k + 1) = 0) := by
  sorry

end binomial_arithmetic_sequence_l2864_286429


namespace modulus_of_z_is_two_l2864_286468

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := by sorry

-- State the theorem
theorem modulus_of_z_is_two :
  z * (2 - 3 * i) = 6 + 4 * i → Complex.abs z = 2 := by sorry

end modulus_of_z_is_two_l2864_286468


namespace largest_common_divisor_m_squared_minus_n_squared_plus_two_l2864_286448

theorem largest_common_divisor_m_squared_minus_n_squared_plus_two
  (m n : ℤ) (h : n < m) :
  ∃ (k : ℤ), m^2 - n^2 + 2 = 2 * k ∧
  ∀ (d : ℤ), (∀ (a b : ℤ), b < a → ∃ (l : ℤ), a^2 - b^2 + 2 = d * l) → d ≤ 2 :=
sorry

end largest_common_divisor_m_squared_minus_n_squared_plus_two_l2864_286448


namespace pie_not_crust_percentage_l2864_286436

/-- Given a pie weighing 200 grams with a crust of 50 grams,
    prove that 75% of the pie is not crust. -/
theorem pie_not_crust_percentage :
  let total_weight : ℝ := 200
  let crust_weight : ℝ := 50
  let non_crust_weight : ℝ := total_weight - crust_weight
  let non_crust_percentage : ℝ := (non_crust_weight / total_weight) * 100
  non_crust_percentage = 75 := by
  sorry


end pie_not_crust_percentage_l2864_286436


namespace line_parallel_to_parallel_plane_l2864_286450

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Plane → Plane → Prop)
variable (line_parallel_plane : Line → Plane → Prop)
variable (line_intersect : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane
  (α β : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β)
  (h_planes_parallel : parallel α β)
  (h_m_parallel_α : line_parallel_plane m α)
  (h_n_intersect_m : line_intersect n m)
  (h_n_not_in_β : ¬ line_in_plane n β) :
  line_parallel_plane n β :=
sorry

end line_parallel_to_parallel_plane_l2864_286450


namespace percentage_equality_l2864_286476

theorem percentage_equality (x y : ℝ) (h1 : 2.5 * x = 0.75 * y) (h2 : x = 20) : y = 200 / 3 := by
  sorry

end percentage_equality_l2864_286476


namespace geometric_sequence_ratio_l2864_286460

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

/-- Two vectors are parallel -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v.1 * w.2 = k * v.2 * w.1

theorem geometric_sequence_ratio 
  (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : parallel (a 2, 2) (a 3, 3)) :
  (a 2 + a 4) / (a 3 + a 5) = 2/3 := by
sorry

end geometric_sequence_ratio_l2864_286460


namespace fraction_transformation_l2864_286492

theorem fraction_transformation (x y : ℝ) (h1 : x / y = 2 / 5) (h2 : x + y = 5.25) :
  (x + 3) / (2 * y) = 3 / 5 := by
sorry

end fraction_transformation_l2864_286492


namespace position_2025_l2864_286491

/-- Represents the possible positions of the square -/
inductive SquarePosition
  | ABCD
  | CDAB
  | BADC
  | DCBA

/-- Applies the transformation pattern to a given position -/
def transform (pos : SquarePosition) : SquarePosition :=
  match pos with
  | SquarePosition.ABCD => SquarePosition.CDAB
  | SquarePosition.CDAB => SquarePosition.BADC
  | SquarePosition.BADC => SquarePosition.DCBA
  | SquarePosition.DCBA => SquarePosition.ABCD

/-- Returns the position after n transformations -/
def nthPosition (n : Nat) : SquarePosition :=
  match n % 4 with
  | 0 => SquarePosition.ABCD
  | 1 => SquarePosition.CDAB
  | 2 => SquarePosition.BADC
  | _ => SquarePosition.DCBA

theorem position_2025 : nthPosition 2025 = SquarePosition.ABCD := by
  sorry


end position_2025_l2864_286491


namespace triangle_side_length_l2864_286432

theorem triangle_side_length (a b c : ℝ) (A B C : Real) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  2 * b = a + c →  -- Arithmetic sequence condition
  B = Real.pi / 3 →  -- 60 degrees in radians
  (1 / 2) * a * c * Real.sin B = 3 * Real.sqrt 3 →  -- Area condition
  b = 2 * Real.sqrt 3 :=
by sorry

end triangle_side_length_l2864_286432


namespace f_inequality_l2864_286420

/-- A function that is continuous and differentiable on ℝ -/
def ContinuousDifferentiableFunction (f : ℝ → ℝ) : Prop :=
  Continuous f ∧ Differentiable ℝ f

theorem f_inequality (f : ℝ → ℝ) 
  (h_f : ContinuousDifferentiableFunction f)
  (h_ineq : ∀ x, 2 * f x - deriv f x > 0) :
  f 1 > f 2 / Real.exp 2 := by
  sorry

end f_inequality_l2864_286420


namespace statement_II_always_true_l2864_286404

-- Define the possible digits
inductive Digit
| two : Digit
| three : Digit
| five : Digit
| six : Digit
| other : Digit

-- Define the statements
def statement_I (d : Digit) : Prop := d = Digit.two
def statement_II (d : Digit) : Prop := d ≠ Digit.three
def statement_III (d : Digit) : Prop := d = Digit.five
def statement_IV (d : Digit) : Prop := d ≠ Digit.six

-- Define the condition that exactly three statements are true
def three_true (d : Digit) : Prop :=
  (statement_I d ∧ statement_II d ∧ statement_III d) ∨
  (statement_I d ∧ statement_II d ∧ statement_IV d) ∨
  (statement_I d ∧ statement_III d ∧ statement_IV d) ∨
  (statement_II d ∧ statement_III d ∧ statement_IV d)

-- Theorem: Statement II is always true given the conditions
theorem statement_II_always_true :
  ∀ d : Digit, three_true d → statement_II d :=
by
  sorry


end statement_II_always_true_l2864_286404


namespace complex_multiplication_l2864_286478

theorem complex_multiplication (z₁ z₂ : ℂ) (h₁ : z₁ = 1 - I) (h₂ : z₂ = 2 + I) :
  z₁ * z₂ = 3 - I := by sorry

end complex_multiplication_l2864_286478


namespace candy_distribution_l2864_286418

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 858 →
  num_bags = 26 →
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 33 := by
sorry

end candy_distribution_l2864_286418


namespace intersection_point_of_problem_lines_l2864_286444

/-- Two lines in a 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → Prop

/-- The specific two lines from the problem -/
def problemLines : TwoLines where
  line1 := λ x y ↦ x + y + 3 = 0
  line2 := λ x y ↦ x - 2*y + 3 = 0

/-- Definition of an intersection point -/
def isIntersectionPoint (lines : TwoLines) (x y : ℝ) : Prop :=
  lines.line1 x y ∧ lines.line2 x y

/-- Theorem stating that (-3, 0) is the intersection point of the given lines -/
theorem intersection_point_of_problem_lines :
  isIntersectionPoint problemLines (-3) 0 := by
  sorry

end intersection_point_of_problem_lines_l2864_286444


namespace airplane_seats_l2864_286408

theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) (coach : ℕ) 
  (h1 : total_seats = 387)
  (h2 : first_class + coach = total_seats)
  (h3 : coach = 4 * first_class + 2) :
  first_class = 77 := by
  sorry

end airplane_seats_l2864_286408


namespace amanda_weekly_earnings_l2864_286457

def amanda_hourly_rate : ℝ := 20.00

def monday_appointments : ℕ := 5
def monday_appointment_duration : ℝ := 1.5

def tuesday_appointment_duration : ℝ := 3

def thursday_appointments : ℕ := 2
def thursday_appointment_duration : ℝ := 2

def saturday_appointment_duration : ℝ := 6

def total_hours : ℝ :=
  monday_appointments * monday_appointment_duration +
  tuesday_appointment_duration +
  thursday_appointments * thursday_appointment_duration +
  saturday_appointment_duration

theorem amanda_weekly_earnings :
  amanda_hourly_rate * total_hours = 410.00 := by
  sorry

end amanda_weekly_earnings_l2864_286457


namespace elevator_time_to_bottom_l2864_286472

/-- Proves that the elevator takes 2 hours to reach the bottom floor given the specified conditions. -/
theorem elevator_time_to_bottom (total_floors : ℕ) (first_half_time : ℕ) (mid_floors_time : ℕ) (last_floors_time : ℕ) :
  total_floors = 20 →
  first_half_time = 15 →
  mid_floors_time = 5 →
  last_floors_time = 16 →
  (first_half_time + 5 * mid_floors_time + 5 * last_floors_time) / 60 = 2 := by
  sorry

end elevator_time_to_bottom_l2864_286472


namespace pi_is_monomial_l2864_286484

-- Define what a monomial is
def is_monomial (e : ℝ → ℝ) : Prop :=
  ∃ (a : ℝ) (n : ℕ), ∀ x, e x = a * x^n

-- State the theorem
theorem pi_is_monomial : is_monomial (λ _ => Real.pi) := by
  sorry

end pi_is_monomial_l2864_286484


namespace cost_increase_when_b_doubled_l2864_286435

theorem cost_increase_when_b_doubled (t : ℝ) (b : ℝ) :
  let original_cost := t * b^4
  let new_cost := t * (2*b)^4
  new_cost = 16 * original_cost :=
by sorry

end cost_increase_when_b_doubled_l2864_286435


namespace sandy_initial_money_l2864_286470

/-- Sandy's initial amount of money before buying the pie -/
def initial_money : ℕ := sorry

/-- The cost of the pie -/
def pie_cost : ℕ := 6

/-- The amount of money Sandy has left after buying the pie -/
def remaining_money : ℕ := 57

/-- Theorem stating that Sandy's initial amount of money was 63 dollars -/
theorem sandy_initial_money : initial_money = 63 := by sorry

end sandy_initial_money_l2864_286470


namespace team_a_more_uniform_than_team_b_l2864_286433

/-- Represents a team in the gymnastics competition -/
structure Team where
  name : String
  variance : ℝ

/-- Determines if one team has more uniform heights than another -/
def hasMoreUniformHeights (team1 team2 : Team) : Prop :=
  team1.variance < team2.variance

/-- Theorem stating that Team A has more uniform heights than Team B -/
theorem team_a_more_uniform_than_team_b 
  (team_a team_b : Team)
  (h_team_a : team_a.name = "Team A" ∧ team_a.variance = 1.5)
  (h_team_b : team_b.name = "Team B" ∧ team_b.variance = 2.8) :
  hasMoreUniformHeights team_a team_b :=
by
  sorry

#check team_a_more_uniform_than_team_b

end team_a_more_uniform_than_team_b_l2864_286433


namespace exactly_one_and_two_red_mutually_exclusive_non_opposing_l2864_286428

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of drawing three balls -/
structure DrawOutcome :=
  (red_count : Nat)
  (white_count : Nat)
  (h_total : red_count + white_count = 3)

/-- The bag of balls -/
def bag : Multiset BallColor :=
  Multiset.replicate 5 BallColor.Red + Multiset.replicate 3 BallColor.White

/-- The event of drawing exactly one red ball -/
def exactly_one_red (outcome : DrawOutcome) : Prop :=
  outcome.red_count = 1

/-- The event of drawing exactly two red balls -/
def exactly_two_red (outcome : DrawOutcome) : Prop :=
  outcome.red_count = 2

/-- Two events are mutually exclusive -/
def mutually_exclusive (e1 e2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

/-- Two events are non-opposing -/
def non_opposing (e1 e2 : DrawOutcome → Prop) : Prop :=
  ∃ outcome, e1 outcome ∨ e2 outcome

theorem exactly_one_and_two_red_mutually_exclusive_non_opposing :
  mutually_exclusive exactly_one_red exactly_two_red ∧
  non_opposing exactly_one_red exactly_two_red :=
sorry

end exactly_one_and_two_red_mutually_exclusive_non_opposing_l2864_286428


namespace unique_correct_answers_l2864_286409

/-- Scoring rules for the Intermediate Maths Challenge -/
structure ScoringRules where
  totalQuestions : Nat
  easyQuestions : Nat
  hardQuestions : Nat
  easyMarks : Nat
  hardMarks : Nat
  easyPenalty : Nat
  hardPenalty : Nat

/-- Calculate the total score based on the number of correct answers -/
def calculateScore (rules : ScoringRules) (correctAnswers : Nat) : Int :=
  sorry

/-- Theorem stating that given the scoring rules and a total score of 80,
    the only possible number of correct answers is 16 -/
theorem unique_correct_answers (rules : ScoringRules) :
  rules.totalQuestions = 25 →
  rules.easyQuestions = 15 →
  rules.hardQuestions = 10 →
  rules.easyMarks = 5 →
  rules.hardMarks = 6 →
  rules.easyPenalty = 1 →
  rules.hardPenalty = 2 →
  ∃! (correctAnswers : Nat), calculateScore rules correctAnswers = 80 ∧ correctAnswers = 16 :=
sorry

end unique_correct_answers_l2864_286409


namespace remainder_problem_l2864_286463

theorem remainder_problem (n : ℕ) : 
  n % 68 = 0 ∧ n / 68 = 269 → n % 67 = 8 := by
  sorry

end remainder_problem_l2864_286463


namespace a_minus_b_equals_1790_l2864_286499

/-- Prove that A - B = 1790 given the definitions of A and B -/
theorem a_minus_b_equals_1790 :
  let A := 1 * 1000 + 16 * 100 + 28 * 10
  let B := 355 + 3 * 245
  A - B = 1790 := by
sorry

end a_minus_b_equals_1790_l2864_286499


namespace jamies_father_age_ratio_l2864_286486

/-- The year of Jamie's 10th birthday -/
def birth_year : ℕ := 2010

/-- Jamie's age on his 10th birthday -/
def jamie_initial_age : ℕ := 10

/-- The ratio of Jamie's father's age to Jamie's age on Jamie's 10th birthday -/
def initial_age_ratio : ℕ := 5

/-- The year when Jamie's father's age is twice Jamie's age -/
def target_year : ℕ := 2040

/-- The ratio of Jamie's father's age to Jamie's age in the target year -/
def target_age_ratio : ℕ := 2

theorem jamies_father_age_ratio :
  target_year = birth_year + (initial_age_ratio - target_age_ratio) * jamie_initial_age := by
  sorry

#check jamies_father_age_ratio

end jamies_father_age_ratio_l2864_286486


namespace sections_after_five_lines_l2864_286480

/-- The number of sections in a rectangle after drawing n line segments,
    where each line increases the number of sections by its sequence order. -/
def sections (n : ℕ) : ℕ :=
  1 + (List.range n).sum

/-- Theorem: After drawing 5 line segments in a rectangle that initially has 1 section,
    where each new line segment increases the number of sections by its sequence order,
    the final number of sections is 16. -/
theorem sections_after_five_lines :
  sections 5 = 16 := by
  sorry

end sections_after_five_lines_l2864_286480


namespace original_houses_count_l2864_286487

-- Define the given conditions
def houses_built_during_boom : ℕ := 97741
def current_total_houses : ℕ := 118558

-- Define the theorem to prove
theorem original_houses_count : 
  current_total_houses - houses_built_during_boom = 20817 := by
  sorry

end original_houses_count_l2864_286487


namespace complex_number_problem_l2864_286455

-- Define the complex numbers z₁ and z₂
variable (z₁ z₂ : ℂ)

-- Define the conditions
def condition1 : Prop := (z₁ - 2) * (1 + Complex.I) = 1 - Complex.I
def condition2 : Prop := z₂.im = 2
def condition3 : Prop := (z₁ * z₂).im = 0

-- State the theorem
theorem complex_number_problem :
  condition1 z₁ → condition2 z₂ → condition3 z₁ z₂ → z₂ = 4 + 2 * Complex.I :=
by sorry

end complex_number_problem_l2864_286455


namespace cloth_cost_price_l2864_286434

theorem cloth_cost_price (meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) :
  meters = 85 →
  selling_price = 8925 →
  profit_per_meter = 35 →
  (selling_price - meters * profit_per_meter) / meters = 70 := by
  sorry

end cloth_cost_price_l2864_286434


namespace f_properties_l2864_286485

noncomputable section

def f (x : ℝ) : ℝ := (Real.log x) / x - 1

def e : ℝ := Real.exp 1

theorem f_properties :
  (∀ x > 0, f x ≤ f e) ∧ 
  (∀ ε > 0, ∃ x > 0, f x < -1/ε) ∧
  (∀ m > 0, 
    (m ≤ e/2 → (∀ x ∈ Set.Icc m (2*m), f x ≤ f (2*m))) ∧
    (e/2 < m ∧ m < e → (∀ x ∈ Set.Icc m (2*m), f x ≤ f e)) ∧
    (m ≥ e → (∀ x ∈ Set.Icc m (2*m), f x ≤ f m))) :=
sorry

#check f_properties

end f_properties_l2864_286485


namespace lcm_28_72_l2864_286479

theorem lcm_28_72 : Nat.lcm 28 72 = 504 := by
  sorry

end lcm_28_72_l2864_286479


namespace min_value_sum_reciprocals_l2864_286445

theorem min_value_sum_reciprocals (p q r : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hsum : p + q + r = 3) :
  (1 / (p + 3*q) + 1 / (q + 3*r) + 1 / (r + 3*p)) ≥ 3/4 := by
  sorry

end min_value_sum_reciprocals_l2864_286445


namespace ellipse_hyperbola_tangency_l2864_286422

/-- An ellipse with equation x^2 + 9y^2 = 9 is tangent to a hyperbola with equation x^2 - m(y - 2)^2 = 4 -/
theorem ellipse_hyperbola_tangency (m : ℝ) : 
  (∃ x y : ℝ, x^2 + 9*y^2 = 9 ∧ x^2 - m*(y - 2)^2 = 4 ∧ 
   ∀ x' y' : ℝ, (x' ≠ x ∨ y' ≠ y) → 
   (x'^2 + 9*y'^2 - 9) * (x'^2 - m*(y' - 2)^2 - 4) > 0) → 
  m = 45/31 := by
sorry

end ellipse_hyperbola_tangency_l2864_286422


namespace rectangle_division_l2864_286464

theorem rectangle_division (a b c d e f : ℕ) : 
  (∀ a b, 39 ≠ 5 * a + 11 * b) ∧ 
  (∃ c d, 27 = 5 * c + 11 * d) ∧ 
  (∃ e f, 55 = 5 * e + 11 * f) := by
  sorry

end rectangle_division_l2864_286464


namespace set_intersection_complement_problem_l2864_286465

theorem set_intersection_complement_problem :
  let U : Type := ℝ
  let A : Set U := {x | x ≤ 3}
  let B : Set U := {x | x ≤ 6}
  (Aᶜ ∩ B) = {x : U | 3 < x ∧ x ≤ 6} := by
  sorry

end set_intersection_complement_problem_l2864_286465


namespace inclination_angle_60_degrees_l2864_286424

def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y + 1 = 0

theorem inclination_angle_60_degrees :
  line (Real.sqrt 3) 4 →
  ∃ θ : ℝ, θ = 60 * π / 180 ∧ Real.tan θ = Real.sqrt 3 :=
by sorry

end inclination_angle_60_degrees_l2864_286424


namespace cubic_sum_zero_l2864_286498

theorem cubic_sum_zero (a b c : ℝ) : 
  a + b + c = 0 → a^3 + a^2*c - a*b*c + b^2*c + b^3 = 0 := by sorry

end cubic_sum_zero_l2864_286498


namespace polynomial_division_remainder_l2864_286416

theorem polynomial_division_remainder :
  ∃ q : Polynomial ℤ, x^2021 + 1 = (x^12 - x^9 + x^6 - x^3 + 1) * q + (-x^4 + 1) := by
  sorry

end polynomial_division_remainder_l2864_286416


namespace function_characterization_l2864_286466

theorem function_characterization (f : ℝ → ℝ) (C : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≥ 0) →
  (∀ x : ℝ, x ≥ 0 → f (f x) = x^4) →
  (∀ x : ℝ, x ≥ 0 → f x ≤ C * x^2) →
  C ≥ 1 →
  (∀ x : ℝ, x ≥ 0 → f x = x^2) :=
by sorry

end function_characterization_l2864_286466


namespace triangle_problem_l2864_286431

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (S : ℝ) (CM : ℝ) :
  b * (3 * b - c) * Real.cos A = b * a * Real.cos C →
  S = 2 * Real.sqrt 2 →
  CM = Real.sqrt 17 / 2 →
  (Real.cos A = 1 / 3) ∧
  ((b = 2 ∧ c = 3) ∨ (b = 3 / 2 ∧ c = 4)) :=
by sorry

end triangle_problem_l2864_286431


namespace alternate_seating_l2864_286407

theorem alternate_seating (B : ℕ) :
  (∃ (G : ℕ), G = 1 ∧ B > 0 ∧ B - 1 = 24) → B = 25 := by
  sorry

end alternate_seating_l2864_286407


namespace infinitely_many_odd_n_composite_l2864_286405

theorem infinitely_many_odd_n_composite (n : ℕ) : 
  ∃ S : Set ℕ, (Set.Infinite S) ∧ 
  (∀ n ∈ S, Odd n ∧ ¬(Nat.Prime (2^n + n - 1))) :=
sorry

end infinitely_many_odd_n_composite_l2864_286405


namespace employee_payment_l2864_286425

theorem employee_payment (x y : ℝ) : 
  x + y = 770 →
  x = 1.2 * y →
  y = 350 := by
sorry

end employee_payment_l2864_286425


namespace digit_reversal_value_l2864_286421

theorem digit_reversal_value (x y : ℕ) : 
  x * 10 + y = 24 →  -- The original number is 24
  x * y = 8 →        -- The product of digits is 8
  x < 10 ∧ y < 10 →  -- The number is two-digit
  ∃ (a : ℕ), y * 10 + x = x * 10 + y + a ∧ a = 18 -- Value added to reverse digits is 18
  := by sorry

end digit_reversal_value_l2864_286421


namespace complement_M_intersect_N_l2864_286430

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (1 - x)}
def N : Set ℝ := {x | (2 : ℝ) ^ (x * (x - 2)) < 1}

-- State the theorem
theorem complement_M_intersect_N : 
  (Mᶜ ∩ N : Set ℝ) = {x | 1 ≤ x ∧ x < 2} := by sorry

end complement_M_intersect_N_l2864_286430


namespace expression_value_l2864_286469

theorem expression_value : 
  (0.02^2 + 0.52^2 + 0.035^2) / (0.002^2 + 0.052^2 + 0.0035^2) = 100 := by
  sorry

end expression_value_l2864_286469


namespace choose_starters_count_l2864_286483

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 7 starters from a team of 16 players,
    where exactly one player must be chosen from a set of 4 quadruplets -/
def choose_starters : ℕ :=
  4 * binomial 12 6

theorem choose_starters_count : choose_starters = 3696 := by sorry

end choose_starters_count_l2864_286483


namespace math_majors_consecutive_probability_l2864_286458

/-- The number of people sitting at the round table -/
def total_people : ℕ := 10

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of ways to choose seats for math majors -/
def total_ways : ℕ := Nat.choose total_people math_majors

/-- The number of ways math majors can sit consecutively -/
def consecutive_ways : ℕ := total_people

/-- The probability that all math majors sit in consecutive seats -/
def probability : ℚ := consecutive_ways / total_ways

theorem math_majors_consecutive_probability :
  probability = 1 / 21 := by
  sorry

end math_majors_consecutive_probability_l2864_286458


namespace square_perimeter_l2864_286451

theorem square_perimeter (s : ℝ) (h : s > 0) : 
  (7/3 * s = 42) → (4 * s = 72) := by
  sorry

end square_perimeter_l2864_286451


namespace girls_on_same_team_probability_l2864_286440

/-- The probability of all three girls being on the same team when five boys and three girls
    are randomly divided into two four-person teams is 1/7. -/
theorem girls_on_same_team_probability :
  let total_children : ℕ := 8
  let num_boys : ℕ := 5
  let num_girls : ℕ := 3
  let team_size : ℕ := 4
  let total_ways : ℕ := (Nat.choose total_children team_size) / 2
  let favorable_ways : ℕ := Nat.choose num_boys 1
  ↑favorable_ways / ↑total_ways = 1 / 7 :=
by sorry

end girls_on_same_team_probability_l2864_286440


namespace sum_nine_terms_is_99_l2864_286437

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  (a 1 + a 4 + a 7 = 35) ∧
  (a 3 + a 6 + a 9 = 27)

/-- The sum of the first n terms of an arithmetic sequence -/
def SumArithmeticSequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

/-- Theorem: The sum of the first 9 terms of the specified arithmetic sequence is 99 -/
theorem sum_nine_terms_is_99 (a : ℕ → ℚ) (h : ArithmeticSequence a) :
  SumArithmeticSequence a 9 = 99 := by
  sorry

end sum_nine_terms_is_99_l2864_286437


namespace tan_double_angle_problem_l2864_286459

open Real

theorem tan_double_angle_problem (θ : ℝ) 
  (h1 : tan (2 * θ) = -2 * sqrt 2) 
  (h2 : π < 2 * θ ∧ 2 * θ < 2 * π) : 
  tan θ = -sqrt 2 / 2 ∧ 
  (2 * (cos (θ / 2))^2 - sin θ - 1) / (sqrt 2 * sin (θ + π / 4)) = 3 + 2 * sqrt 2 := by
  sorry

end tan_double_angle_problem_l2864_286459


namespace max_cake_boxes_in_carton_l2864_286493

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the carton dimensions -/
def cartonDimensions : BoxDimensions :=
  { length := 25, width := 42, height := 60 }

/-- Represents the cake box dimensions -/
def cakeBoxDimensions : BoxDimensions :=
  { length := 8, width := 7, height := 5 }

/-- Theorem stating the maximum number of cake boxes that can fit in the carton -/
theorem max_cake_boxes_in_carton :
  (boxVolume cartonDimensions) / (boxVolume cakeBoxDimensions) = 225 := by
  sorry

end max_cake_boxes_in_carton_l2864_286493


namespace jerry_weller_votes_l2864_286467

theorem jerry_weller_votes 
  (total_votes : ℕ) 
  (vote_difference : ℕ) 
  (h1 : total_votes = 196554)
  (h2 : vote_difference = 20196) :
  ∃ (jerry_votes john_votes : ℕ),
    jerry_votes = 108375 ∧ 
    john_votes + vote_difference = jerry_votes ∧
    jerry_votes + john_votes = total_votes :=
by
  sorry

end jerry_weller_votes_l2864_286467


namespace min_value_expression_l2864_286482

theorem min_value_expression (x y : ℝ) : 3 * x^2 + 3 * x * y + y^2 - 6 * x + 4 * y + 5 ≥ 2 := by
  sorry

end min_value_expression_l2864_286482


namespace equation_solution_l2864_286494

theorem equation_solution : ∃ x : ℝ, 0.4 * x + (0.6 * 0.8) = 0.56 ∧ x = 0.2 := by
  sorry

end equation_solution_l2864_286494


namespace fruit_vendor_sales_l2864_286403

/-- Calculates the total sales for a fruit vendor given the prices and quantities sold --/
theorem fruit_vendor_sales
  (apple_price : ℚ)
  (orange_price : ℚ)
  (morning_apples : ℕ)
  (morning_oranges : ℕ)
  (afternoon_apples : ℕ)
  (afternoon_oranges : ℕ)
  (h1 : apple_price = 3/2)
  (h2 : orange_price = 1)
  (h3 : morning_apples = 40)
  (h4 : morning_oranges = 30)
  (h5 : afternoon_apples = 50)
  (h6 : afternoon_oranges = 40) :
  let morning_sales := apple_price * morning_apples + orange_price * morning_oranges
  let afternoon_sales := apple_price * afternoon_apples + orange_price * afternoon_oranges
  morning_sales + afternoon_sales = 205 :=
by sorry

end fruit_vendor_sales_l2864_286403


namespace smallest_among_four_l2864_286456

theorem smallest_among_four (a b c d : ℚ) (h1 : a = -2) (h2 : b = -1) (h3 : c = 0) (h4 : d = 1) :
  a ≤ b ∧ a ≤ c ∧ a ≤ d := by
  sorry

end smallest_among_four_l2864_286456


namespace smallest_positive_integer_ending_in_3_divisible_by_11_l2864_286410

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

theorem smallest_positive_integer_ending_in_3_divisible_by_11 :
  ∃ (n : ℕ), n > 0 ∧ ends_in_3 n ∧ n % 11 = 0 ∧
  ∀ (m : ℕ), m > 0 → ends_in_3 m → m % 11 = 0 → m ≥ n :=
by sorry

end smallest_positive_integer_ending_in_3_divisible_by_11_l2864_286410


namespace ellipse_triangle_perimeter_l2864_286474

/-- An ellipse with semi-major axis a, semi-minor axis b, and eccentricity e -/
structure Ellipse where
  a : ℝ
  b : ℝ
  e : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_a_gt_b : b < a
  h_e_eq : e = Real.sqrt (1 - b^2 / a^2)

/-- A point on the ellipse -/
structure EllipsePoint (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2 / E.a^2 + y^2 / E.b^2 = 1

/-- The upper vertex of the ellipse -/
def upperVertex (E : Ellipse) : EllipsePoint E where
  x := 0
  y := E.b
  h_on_ellipse := by sorry

/-- A focus of the ellipse -/
structure Focus (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_major_axis : y = 0
  h_distance_from_center : x^2 = E.a^2 * E.e^2

/-- A line perpendicular to the line connecting a focus and the upper vertex -/
structure PerpendicularLine (E : Ellipse) (F : Focus E) where
  slope : ℝ
  h_perpendicular : slope * (F.x / E.b) = -1

/-- The intersection points of the perpendicular line with the ellipse -/
structure IntersectionPoints (E : Ellipse) (F : Focus E) (L : PerpendicularLine E F) where
  D : EllipsePoint E
  E : EllipsePoint E
  h_on_line_D : D.y = L.slope * (D.x - F.x)
  h_on_line_E : E.y = L.slope * (E.x - F.x)
  h_distance : (D.x - E.x)^2 + (D.y - E.y)^2 = 36

/-- The main theorem -/
theorem ellipse_triangle_perimeter
  (E : Ellipse)
  (h_e : E.e = 1/2)
  (F₁ F₂ : Focus E)
  (L : PerpendicularLine E F₁)
  (I : IntersectionPoints E F₁ L) :
  let A := upperVertex E
  let D := I.D
  let E := I.E
  (Real.sqrt ((A.x - D.x)^2 + (A.y - D.y)^2) +
   Real.sqrt ((A.x - E.x)^2 + (A.y - E.y)^2) +
   Real.sqrt ((D.x - E.x)^2 + (D.y - E.y)^2)) = 13 := by sorry

end ellipse_triangle_perimeter_l2864_286474


namespace max_values_f_and_g_l2864_286412

noncomputable def f (θ : ℝ) := (1 + Real.cos θ) * (1 + Real.sin θ)
noncomputable def g (θ : ℝ) := (1/2 + Real.cos θ) * (Real.sqrt 3/2 + Real.sin θ)

theorem max_values_f_and_g :
  (∃ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) ∧ f θ = (3 + 2 * Real.sqrt 2)/2) ∧
  (∀ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) → f θ ≤ (3 + 2 * Real.sqrt 2)/2) ∧
  (∃ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) ∧ g θ = Real.sqrt 3/4 + 3/2 * Real.sin (5*Real.pi/9)) ∧
  (∀ (θ : ℝ), θ ∈ Set.Ioo 0 (Real.pi/2) → g θ ≤ Real.sqrt 3/4 + 3/2 * Real.sin (5*Real.pi/9)) :=
by sorry

end max_values_f_and_g_l2864_286412


namespace danny_initial_caps_l2864_286413

/-- The number of bottle caps Danny had initially -/
def initial_caps : ℕ := sorry

/-- The number of bottle caps Danny threw away -/
def thrown_away : ℕ := 60

/-- The number of new bottle caps Danny found -/
def found : ℕ := 58

/-- The number of bottle caps Danny traded away -/
def traded_away : ℕ := 15

/-- The number of bottle caps Danny received in trade -/
def received : ℕ := 25

/-- The number of bottle caps Danny has now -/
def final_caps : ℕ := 67

/-- Theorem stating that Danny initially had 59 bottle caps -/
theorem danny_initial_caps : 
  initial_caps = 59 ∧
  final_caps = initial_caps - thrown_away + found - traded_away + received :=
sorry

end danny_initial_caps_l2864_286413


namespace log_inequality_l2864_286477

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Main theorem
theorem log_inequality (a m n : ℝ) (ha : a > 1) (hm : 0 < m) (hmn : m < 1) (hn : 1 < n) :
  f a m < 0 ∧ 0 < f a n := by
  sorry

end log_inequality_l2864_286477


namespace total_wheels_on_floor_l2864_286415

theorem total_wheels_on_floor (num_people : ℕ) (wheels_per_skate : ℕ) (skates_per_person : ℕ) : 
  num_people = 40 → 
  wheels_per_skate = 2 → 
  skates_per_person = 2 → 
  num_people * wheels_per_skate * skates_per_person = 160 := by
  sorry

end total_wheels_on_floor_l2864_286415


namespace number_divisibility_l2864_286496

theorem number_divisibility (N : ℕ) (h1 : N % 68 = 0) (h2 : N % 67 = 1) : N = 68 := by
  sorry

end number_divisibility_l2864_286496


namespace unique_divisor_with_remainders_l2864_286481

theorem unique_divisor_with_remainders :
  ∃! b : ℕ, b > 1 ∧ 826 % b = 7 ∧ 4373 % b = 8 :=
by
  -- The proof goes here
  sorry

end unique_divisor_with_remainders_l2864_286481


namespace intersection_complement_theorem_l2864_286447

open Set

def U : Set ℝ := univ

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

def N : Set ℝ := {x | x ≥ 0}

theorem intersection_complement_theorem :
  M ∩ (U \ N) = {x : ℝ | -1 ≤ x ∧ x < 0} := by sorry

end intersection_complement_theorem_l2864_286447


namespace functional_equation_solution_l2864_286411

/-- A function satisfying the given functional equation for all integers -/
def SatisfiesFunctionalEq (f : ℤ → ℤ) : Prop :=
  ∀ m n : ℤ, f (f m + n) + f m = f n + f (3 * m) + 2014

/-- The theorem stating that any function satisfying the functional equation
    must be of the form f(n) = 2n + 1007 -/
theorem functional_equation_solution :
  ∀ f : ℤ → ℤ, SatisfiesFunctionalEq f → ∀ n : ℤ, f n = 2 * n + 1007 :=
by sorry

end functional_equation_solution_l2864_286411


namespace inequality_proof_l2864_286446

theorem inequality_proof (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  a^4 + b^4 + c^4 - 2*(a^2*b^2 + a^2*c^2 + b^2*c^2) + a^2*b*c + b^2*a*c + c^2*a*b ≥ 0 :=
by sorry

end inequality_proof_l2864_286446


namespace exist_similar_numbers_l2864_286442

/-- A function that generates a number by repeating a given 3-digit number n times -/
def repeatDigits (d : Nat) (n : Nat) : Nat :=
  (d * (Nat.pow 10 (3 * n) - 1)) / 999

/-- Theorem stating the existence of three similar 1995-digit numbers with the required property -/
theorem exist_similar_numbers : ∃ (A B C : Nat),
  (A = repeatDigits 459 665) ∧
  (B = repeatDigits 495 665) ∧
  (C = repeatDigits 954 665) ∧
  (A + B = C) ∧
  (A ≠ 0) ∧ (B ≠ 0) ∧ (C ≠ 0) :=
sorry

end exist_similar_numbers_l2864_286442


namespace max_sphere_in_cones_l2864_286402

/-- Right circular cone -/
structure Cone :=
  (base_radius : ℝ)
  (height : ℝ)

/-- Configuration of two intersecting cones -/
structure ConePair :=
  (cone : Cone)
  (intersection_distance : ℝ)

/-- The maximum squared radius of a sphere fitting in both cones -/
def max_sphere_radius_squared (cp : ConePair) : ℝ :=
  sorry

/-- Theorem statement -/
theorem max_sphere_in_cones :
  let cp := ConePair.mk (Cone.mk 5 12) 4
  max_sphere_radius_squared cp = 1600 / 169 := by
  sorry

end max_sphere_in_cones_l2864_286402


namespace teacher_student_ratio_l2864_286441

theorem teacher_student_ratio 
  (initial_student_teacher_ratio : ℚ) 
  (current_teachers : ℕ) 
  (student_increase : ℕ) 
  (teacher_increase : ℕ) 
  (new_student_teacher_ratio : ℚ) 
  (h1 : initial_student_teacher_ratio = 50 / current_teachers)
  (h2 : current_teachers = 3)
  (h3 : student_increase = 50)
  (h4 : teacher_increase = 5)
  (h5 : new_student_teacher_ratio = 25)
  (h6 : (initial_student_teacher_ratio * current_teachers + student_increase) / 
        (current_teachers + teacher_increase) = new_student_teacher_ratio) :
  (1 : ℚ) / initial_student_teacher_ratio = 1 / 50 :=
sorry

end teacher_student_ratio_l2864_286441


namespace decimal_multiplication_l2864_286471

theorem decimal_multiplication (a b c : ℚ) : 
  a = 8/10 → b = 25/100 → c = 2/10 → a * b * c = 4/100 := by
  sorry

end decimal_multiplication_l2864_286471


namespace max_value_of_a_l2864_286490

-- Define the condition
def inequality_holds (a : ℝ) : Prop :=
  ∀ x : ℝ, |x - 2| + |x - a| ≥ a

-- State the theorem
theorem max_value_of_a :
  ∃ a_max : ℝ, a_max = 1 ∧
  inequality_holds a_max ∧
  ∀ a : ℝ, inequality_holds a → a ≤ a_max :=
sorry

end max_value_of_a_l2864_286490


namespace solution_set_part_I_solution_part_II_l2864_286489

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + 3 * x

-- Part I
theorem solution_set_part_I :
  {x : ℝ | f 1 x ≥ 3 * x + 2} = {x : ℝ | x ≥ 3 ∨ x ≤ -1} := by sorry

-- Part II
theorem solution_part_II (a : ℝ) (h : a > 0) :
  ({x : ℝ | f a x ≤ 0} = {x : ℝ | x ≤ -3}) → a = 6 := by sorry

end solution_set_part_I_solution_part_II_l2864_286489


namespace only_zero_point_eight_greater_than_zero_point_seven_l2864_286452

theorem only_zero_point_eight_greater_than_zero_point_seven :
  let numbers : List ℝ := [0.07, -0.41, 0.8, 0.35, -0.9]
  ∀ x ∈ numbers, x > 0.7 ↔ x = 0.8 := by
  sorry

end only_zero_point_eight_greater_than_zero_point_seven_l2864_286452


namespace johnny_distance_l2864_286439

/-- The distance between Q and Y in kilometers -/
def total_distance : ℝ := 45

/-- Matthew's walking rate in kilometers per hour -/
def matthew_rate : ℝ := 3

/-- Johnny's walking rate in kilometers per hour -/
def johnny_rate : ℝ := 4

/-- The time difference in hours between when Matthew and Johnny start walking -/
def time_difference : ℝ := 1

/-- The theorem stating that Johnny walked 24 km when they met -/
theorem johnny_distance : ℝ := by
  sorry

end johnny_distance_l2864_286439


namespace expression_simplification_l2864_286400

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 7) :
  (2 / (x - 3) - 1 / (x + 3)) / ((x^2 + 9*x) / (x^2 - 9)) = Real.sqrt 7 / 7 := by
  sorry

end expression_simplification_l2864_286400


namespace max_students_distribution_l2864_286495

theorem max_students_distribution (pens pencils : ℕ) (h1 : pens = 2010) (h2 : pencils = 1050) : 
  (∃ (notebooks : ℕ), notebooks ≥ 30 ∧ 
    (∃ (distribution : ℕ → ℕ × ℕ × ℕ), 
      (∀ i j, i ≠ j → (distribution i).2.2 ≠ (distribution j).2.2) ∧
      (∀ i, i < 30 → (distribution i).1 = pens / 30 ∧ (distribution i).2.1 = pencils / 30))) ∧
  (∀ n : ℕ, n > 30 → 
    ¬(∃ (notebooks : ℕ), notebooks ≥ n ∧ 
      (∃ (distribution : ℕ → ℕ × ℕ × ℕ), 
        (∀ i j, i ≠ j → (distribution i).2.2 ≠ (distribution j).2.2) ∧
        (∀ i, i < n → (distribution i).1 = pens / n ∧ (distribution i).2.1 = pencils / n)))) :=
by sorry

end max_students_distribution_l2864_286495
