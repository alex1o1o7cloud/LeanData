import Mathlib

namespace NUMINAMATH_CALUDE_river_current_calculation_l2925_292505

/-- Represents the speed of a boat in still water -/
def boat_speed : ℝ := 20

/-- Represents the distance traveled up the river -/
def distance : ℝ := 91

/-- Represents the total time for the round trip -/
def total_time : ℝ := 10

/-- Calculates the speed of the river's current -/
def river_current_speed : ℝ := 6

theorem river_current_calculation :
  ∃ (c : ℝ), c = river_current_speed ∧
  distance / (boat_speed - c) + distance / (boat_speed + c) = total_time :=
by sorry

end NUMINAMATH_CALUDE_river_current_calculation_l2925_292505


namespace NUMINAMATH_CALUDE_only_rectangle_area_certain_l2925_292567

-- Define the events
inductive Event
  | WaterFreeze : Event
  | ExamScore : Event
  | CoinToss : Event
  | RectangleArea : Event

-- Define a function to check if an event is certain
def isCertainEvent : Event → Prop
  | Event.WaterFreeze => False
  | Event.ExamScore => False
  | Event.CoinToss => False
  | Event.RectangleArea => True

-- Theorem statement
theorem only_rectangle_area_certain :
  ∀ e : Event, isCertainEvent e ↔ e = Event.RectangleArea :=
by sorry

end NUMINAMATH_CALUDE_only_rectangle_area_certain_l2925_292567


namespace NUMINAMATH_CALUDE_teresas_class_size_l2925_292568

theorem teresas_class_size :
  ∃! n : ℕ, 50 < n ∧ n < 100 ∧ n % 3 = 2 ∧ n % 4 = 2 ∧ n % 5 = 2 ∧ n = 62 := by
  sorry

end NUMINAMATH_CALUDE_teresas_class_size_l2925_292568


namespace NUMINAMATH_CALUDE_system_solution_unique_l2925_292547

theorem system_solution_unique :
  ∃! (x y : ℚ), 3 * x + y = 2 ∧ 2 * x - y = 8 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_system_solution_unique_l2925_292547


namespace NUMINAMATH_CALUDE_min_y_value_l2925_292554

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 72*y) : 
  ∀ y' : ℝ, (∃ x' : ℝ, x'^2 + y'^2 = 20*x' + 72*y') → y ≥ 36 - Real.sqrt 1396 := by
sorry

end NUMINAMATH_CALUDE_min_y_value_l2925_292554


namespace NUMINAMATH_CALUDE_expression_simplification_l2925_292563

theorem expression_simplification (b : ℝ) : ((3 * b + 6) - 5 * b) / 3 = -2/3 * b + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2925_292563


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2925_292590

theorem sqrt_equation_solution : ∃ (x : ℝ), x = 1225 / 36 ∧ Real.sqrt x + Real.sqrt (x + 4) = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2925_292590


namespace NUMINAMATH_CALUDE_same_answers_l2925_292529

-- Define a type for questions
variable (Question : Type)

-- Define predicates for each witness's "yes" answers
variable (A B C : Question → Prop)

-- State the conditions
variable (h1 : ∀ q, B q ∧ C q → A q)
variable (h2 : ∀ q, A q → B q)
variable (h3 : ∀ q, B q → A q ∨ C q)

-- Theorem statement
theorem same_answers : ∀ q, A q ↔ B q := by sorry

end NUMINAMATH_CALUDE_same_answers_l2925_292529


namespace NUMINAMATH_CALUDE_parabola_distance_theorem_l2925_292545

/-- Parabola type -/
structure Parabola where
  /-- The equation of the parabola y^2 = 8x -/
  equation : ℝ → ℝ → Prop
  /-- The focus of the parabola -/
  focus : ℝ × ℝ
  /-- The directrix of the parabola -/
  directrix : ℝ → ℝ → Prop

/-- Point on the directrix -/
def PointOnDirectrix (p : Parabola) : Type := { point : ℝ × ℝ // p.directrix point.1 point.2 }

/-- Point on the parabola -/
def PointOnParabola (p : Parabola) : Type := { point : ℝ × ℝ // p.equation point.1 point.2 }

/-- Theorem: For a parabola y^2 = 8x, if FP = 4FQ, then |QF| = 3 -/
theorem parabola_distance_theorem (p : Parabola) 
  (hpeq : p.equation = fun x y ↦ y^2 = 8*x)
  (P : PointOnDirectrix p) 
  (Q : PointOnParabola p) 
  (hline : ∃ (t : ℝ), Q.val = p.focus + t • (P.val - p.focus))
  (hfp : ‖P.val - p.focus‖ = 4 * ‖Q.val - p.focus‖) :
  ‖Q.val - p.focus‖ = 3 := by sorry

end NUMINAMATH_CALUDE_parabola_distance_theorem_l2925_292545


namespace NUMINAMATH_CALUDE_custom_mult_square_identity_l2925_292573

-- Define the custom multiplication operation
def custom_mult (a b : ℝ) : ℝ := (a - b)^2

-- Theorem statement
theorem custom_mult_square_identity (x y : ℝ) :
  custom_mult (x^2) (y^2) = (x + y)^2 * (x - y)^2 := by
  sorry

end NUMINAMATH_CALUDE_custom_mult_square_identity_l2925_292573


namespace NUMINAMATH_CALUDE_unique_divisible_number_l2925_292552

theorem unique_divisible_number : ∃! n : ℕ, 
  45400 ≤ n ∧ n < 45500 ∧ 
  n % 2 = 0 ∧ 
  n % 7 = 0 ∧ 
  n % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_divisible_number_l2925_292552


namespace NUMINAMATH_CALUDE_correct_selection_methods_l2925_292527

def total_people : ℕ := 16
def people_per_class : ℕ := 4
def num_classes : ℕ := 4
def people_to_select : ℕ := 3

def selection_methods : ℕ := sorry

theorem correct_selection_methods :
  selection_methods = 472 := by sorry

end NUMINAMATH_CALUDE_correct_selection_methods_l2925_292527


namespace NUMINAMATH_CALUDE_train_speed_l2925_292581

theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (h1 : train_length = 500) (h2 : crossing_time = 50) :
  train_length / crossing_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l2925_292581


namespace NUMINAMATH_CALUDE_rectangle_dimensions_and_area_l2925_292548

theorem rectangle_dimensions_and_area (x : ℝ) : 
  (x - 3 > 0) →
  (3 * x + 4 > 0) →
  ((x - 3) * (3 * x + 4) = 12 * x - 7) →
  (x = (17 + Real.sqrt 349) / 6) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_and_area_l2925_292548


namespace NUMINAMATH_CALUDE_possible_days_l2925_292517

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def Anya_lies (d : Day) : Prop :=
  d = Day.Tuesday ∨ d = Day.Wednesday ∨ d = Day.Thursday

def Vanya_lies (d : Day) : Prop :=
  d = Day.Thursday ∨ d = Day.Friday ∨ d = Day.Saturday

def Anya_says_Friday (d : Day) : Prop :=
  (Anya_lies d ∧ d ≠ Day.Friday) ∨ (¬Anya_lies d ∧ d = Day.Friday)

def Vanya_says_Tuesday (d : Day) : Prop :=
  (Vanya_lies d ∧ d ≠ Day.Tuesday) ∨ (¬Vanya_lies d ∧ d = Day.Tuesday)

theorem possible_days :
  ∀ d : Day, (Anya_says_Friday d ∧ Vanya_says_Tuesday d) ↔ 
    (d = Day.Tuesday ∨ d = Day.Thursday ∨ d = Day.Friday) :=
by sorry

end NUMINAMATH_CALUDE_possible_days_l2925_292517


namespace NUMINAMATH_CALUDE_total_distance_is_20_l2925_292536

/-- Represents the walking scenario with given speeds and total time -/
structure WalkingScenario where
  flat_speed : ℝ
  uphill_speed : ℝ
  downhill_speed : ℝ
  total_time : ℝ

/-- Calculates the total distance walked given a WalkingScenario -/
def total_distance (s : WalkingScenario) : ℝ :=
  sorry

/-- Theorem stating that the total distance walked is 20 km -/
theorem total_distance_is_20 (s : WalkingScenario) 
  (h1 : s.flat_speed = 4)
  (h2 : s.uphill_speed = 3)
  (h3 : s.downhill_speed = 6)
  (h4 : s.total_time = 5) :
  total_distance s = 20 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_20_l2925_292536


namespace NUMINAMATH_CALUDE_min_value_theorem_l2925_292534

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum of first n terms
  second_term : a 2 = 4
  tenth_sum : S 10 = 110

/-- The theorem statement -/
theorem min_value_theorem (seq : ArithmeticSequence) :
  ∃ (n : ℕ), ∀ (m : ℕ), (seq.S m + 64) / seq.a m ≥ 17 / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2925_292534


namespace NUMINAMATH_CALUDE_five_twos_equal_twentyfour_l2925_292543

theorem five_twos_equal_twentyfour : ∃ (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ), 
  f (g 2 2 + 2) (g 2 2) = 24 :=
by sorry

end NUMINAMATH_CALUDE_five_twos_equal_twentyfour_l2925_292543


namespace NUMINAMATH_CALUDE_remaining_slices_eq_ten_l2925_292523

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 8

/-- The number of slices in an extra-large pizza -/
def extra_large_pizza_slices : ℕ := 12

/-- The number of slices Mary eats from the large pizza -/
def slices_eaten_from_large : ℕ := 7

/-- The number of slices Mary eats from the extra-large pizza -/
def slices_eaten_from_extra_large : ℕ := 3

/-- The total number of remaining slices after Mary eats from both pizzas -/
def total_remaining_slices : ℕ := 
  (large_pizza_slices - slices_eaten_from_large) + 
  (extra_large_pizza_slices - slices_eaten_from_extra_large)

theorem remaining_slices_eq_ten : total_remaining_slices = 10 := by
  sorry

end NUMINAMATH_CALUDE_remaining_slices_eq_ten_l2925_292523


namespace NUMINAMATH_CALUDE_relay_team_permutations_l2925_292550

theorem relay_team_permutations (n : ℕ) (h : n = 4) : Nat.factorial (n - 1) = 6 := by
  sorry

end NUMINAMATH_CALUDE_relay_team_permutations_l2925_292550


namespace NUMINAMATH_CALUDE_rationalize_denominator_sqrt_5_12_l2925_292591

theorem rationalize_denominator_sqrt_5_12 : 
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_sqrt_5_12_l2925_292591


namespace NUMINAMATH_CALUDE_fencing_cost_per_metre_l2925_292530

/-- Proof of fencing cost per metre for a rectangular field -/
theorem fencing_cost_per_metre
  (ratio_length_width : ℚ) -- Ratio of length to width
  (area : ℝ) -- Area of the field in square meters
  (total_cost : ℝ) -- Total cost of fencing
  (h_ratio : ratio_length_width = 3 / 4) -- The ratio of length to width is 3:4
  (h_area : area = 10092) -- The area is 10092 sq. m
  (h_cost : total_cost = 101.5) -- The total cost is 101.5
  : ∃ (length width : ℝ),
    length / width = ratio_length_width ∧
    length * width = area ∧
    (2 * (length + width)) * (total_cost / (2 * (length + width))) = total_cost ∧
    total_cost / (2 * (length + width)) = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_fencing_cost_per_metre_l2925_292530


namespace NUMINAMATH_CALUDE_company_blocks_l2925_292569

/-- Given a company with the following properties:
  - The total amount for gifts is $4000
  - Each gift costs $4
  - There are approximately 100 workers per block
  Prove that the number of blocks in the company is 10 -/
theorem company_blocks (total_amount : ℕ) (gift_cost : ℕ) (workers_per_block : ℕ) : 
  total_amount = 4000 →
  gift_cost = 4 →
  workers_per_block = 100 →
  (total_amount / gift_cost) / workers_per_block = 10 := by
  sorry

end NUMINAMATH_CALUDE_company_blocks_l2925_292569


namespace NUMINAMATH_CALUDE_expected_vote_for_a_l2925_292557

/-- Percentage of registered voters who are Democrats -/
def democrat_percentage : ℝ := 0.60

/-- Percentage of registered voters who are Republicans -/
def republican_percentage : ℝ := 1 - democrat_percentage

/-- Percentage of Democrats expected to vote for candidate A -/
def democrat_vote_a : ℝ := 0.70

/-- Percentage of Republicans expected to vote for candidate A -/
def republican_vote_a : ℝ := 0.20

/-- Theorem: The percentage of registered voters expected to vote for candidate A is 50% -/
theorem expected_vote_for_a :
  democrat_percentage * democrat_vote_a + republican_percentage * republican_vote_a = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_expected_vote_for_a_l2925_292557


namespace NUMINAMATH_CALUDE_max_point_difference_is_n_l2925_292535

/-- A soccer tournament with n teams -/
structure SoccerTournament where
  n : ℕ  -- number of teams
  n_pos : 0 < n  -- number of teams is positive

/-- The result of a match between two teams -/
inductive MatchResult
  | Win
  | Loss
  | Draw

/-- Points awarded for each match result -/
def pointsForResult (result : MatchResult) : ℕ :=
  match result with
  | MatchResult.Win => 2
  | MatchResult.Loss => 0
  | MatchResult.Draw => 1

/-- The maximum possible difference in points between adjacent teams -/
def maxPointDifference (tournament : SoccerTournament) : ℕ :=
  tournament.n

theorem max_point_difference_is_n (tournament : SoccerTournament) :
  ∃ (team1 team2 : ℕ),
    team1 < tournament.n ∧
    team2 < tournament.n ∧
    team1 + 1 = team2 ∧
    ∃ (points1 points2 : ℕ),
      points1 - points2 = maxPointDifference tournament :=
by
  sorry

#check max_point_difference_is_n

end NUMINAMATH_CALUDE_max_point_difference_is_n_l2925_292535


namespace NUMINAMATH_CALUDE_original_profit_percentage_l2925_292524

theorem original_profit_percentage 
  (original_selling_price : ℝ) 
  (additional_profit : ℝ) :
  original_selling_price = 1100 →
  additional_profit = 70 →
  ∃ (original_purchase_price : ℝ),
    (1.3 * (0.9 * original_purchase_price) = original_selling_price + additional_profit) ∧
    ((original_selling_price - original_purchase_price) / original_purchase_price * 100 = 10) :=
by sorry

end NUMINAMATH_CALUDE_original_profit_percentage_l2925_292524


namespace NUMINAMATH_CALUDE_fib_150_mod_9_l2925_292510

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib n + fib (n + 1)

-- Define the property that Fibonacci sequence mod 9 repeats every 24 terms
axiom fib_mod_9_period : ∀ n : ℕ, fib n % 9 = fib (n % 24) % 9

-- Theorem statement
theorem fib_150_mod_9 : fib 149 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_9_l2925_292510


namespace NUMINAMATH_CALUDE_equation_b_not_symmetric_l2925_292572

def is_symmetric_to_x_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f x y = f x (-y)

theorem equation_b_not_symmetric :
  ¬(is_symmetric_to_x_axis (fun x y => x^2*y + x*y^2 - 1)) ∧
  (is_symmetric_to_x_axis (fun x y => x^2 - x + y^2 - 1)) ∧
  (is_symmetric_to_x_axis (fun x y => 2*x^2 - y^2 - 1)) ∧
  (is_symmetric_to_x_axis (fun x y => x + y^2 + 1)) :=
by sorry

end NUMINAMATH_CALUDE_equation_b_not_symmetric_l2925_292572


namespace NUMINAMATH_CALUDE_maria_stationery_cost_l2925_292520

/-- The cost of Maria's stationery purchase -/
def stationery_cost (pencil_cost : ℝ) (pen_cost : ℝ) : Prop :=
  pencil_cost = 8 ∧ 
  pen_cost = pencil_cost / 2 ∧
  pencil_cost + pen_cost = 12

/-- Theorem: Maria paid $12 for both the pen and the pencil -/
theorem maria_stationery_cost : 
  ∃ (pencil_cost pen_cost : ℝ), stationery_cost pencil_cost pen_cost :=
by
  sorry

end NUMINAMATH_CALUDE_maria_stationery_cost_l2925_292520


namespace NUMINAMATH_CALUDE_product_last_two_digits_not_consecutive_l2925_292558

theorem product_last_two_digits_not_consecutive (a b c : ℕ) : 
  ¬ (∃ n : ℕ, 
    (10 ≤ n ∧ n < 100) ∧
    (ab % 100 = n ∧ ac % 100 = n + 1 ∧ bc % 100 = n + 2) ∨
    (ab % 100 = n ∧ bc % 100 = n + 1 ∧ ac % 100 = n + 2) ∨
    (ac % 100 = n ∧ ab % 100 = n + 1 ∧ bc % 100 = n + 2) ∨
    (ac % 100 = n ∧ bc % 100 = n + 1 ∧ ab % 100 = n + 2) ∨
    (bc % 100 = n ∧ ab % 100 = n + 1 ∧ ac % 100 = n + 2) ∨
    (bc % 100 = n ∧ ac % 100 = n + 1 ∧ ab % 100 = n + 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_product_last_two_digits_not_consecutive_l2925_292558


namespace NUMINAMATH_CALUDE_pairing_ways_eq_5040_l2925_292541

/-- Represents the number of students with each grade -/
structure GradeDistribution where
  grade5 : Nat
  grade4 : Nat
  grade3 : Nat

/-- Calculates the number of ways to form pairs of students with different grades -/
def pairingWays (dist : GradeDistribution) : Nat :=
  Nat.choose dist.grade4 dist.grade5 * Nat.factorial dist.grade5

/-- The given grade distribution in the problem -/
def problemDistribution : GradeDistribution :=
  { grade5 := 6, grade4 := 7, grade3 := 1 }

/-- Theorem stating that the number of pairing ways for the given distribution is 5040 -/
theorem pairing_ways_eq_5040 :
  pairingWays problemDistribution = 5040 := by
  sorry

end NUMINAMATH_CALUDE_pairing_ways_eq_5040_l2925_292541


namespace NUMINAMATH_CALUDE_complex_power_sum_l2925_292532

theorem complex_power_sum (z : ℂ) (h : z^2 + z + 1 = 0) : z^2010 + z^2009 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l2925_292532


namespace NUMINAMATH_CALUDE_min_square_value_l2925_292560

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ m : ℕ+, (15 * a + 16 * b : ℕ) = m^2)
  (h2 : ∃ n : ℕ+, (16 * a - 15 * b : ℕ) = n^2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 481 :=
sorry

end NUMINAMATH_CALUDE_min_square_value_l2925_292560


namespace NUMINAMATH_CALUDE_loan_years_is_eight_l2925_292582

/-- Given a loan scenario, calculate the number of years for the first part. -/
def calculate_years (total_sum interest_rate1 interest_rate2 second_part_sum second_part_years : ℚ) : ℚ :=
  let first_part_sum := total_sum - second_part_sum
  let second_part_interest := second_part_sum * interest_rate2 * second_part_years / 100
  second_part_interest * 100 / (first_part_sum * interest_rate1)

/-- Prove that the number of years for the first part of the loan is 8. -/
theorem loan_years_is_eight :
  let total_sum : ℚ := 2769
  let interest_rate1 : ℚ := 3
  let interest_rate2 : ℚ := 5
  let second_part_sum : ℚ := 1704
  let second_part_years : ℚ := 3
  calculate_years total_sum interest_rate1 interest_rate2 second_part_sum second_part_years = 8 := by
  sorry


end NUMINAMATH_CALUDE_loan_years_is_eight_l2925_292582


namespace NUMINAMATH_CALUDE_rhombus_side_length_l2925_292566

/-- A rhombus with one diagonal of length 20 and area 480 has sides of length 26 -/
theorem rhombus_side_length (d1 d2 area side : ℝ) : 
  d1 = 20 →
  area = 480 →
  area = d1 * d2 / 2 →
  side * side = (d1/2)^2 + (d2/2)^2 →
  side = 26 := by
sorry


end NUMINAMATH_CALUDE_rhombus_side_length_l2925_292566


namespace NUMINAMATH_CALUDE_no_triangle_with_special_sides_l2925_292539

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : a > 0 ∧ b > 0 ∧ c > 0
  h_inequality : a + b > c ∧ b + c > a ∧ c + a > b

-- Define functions for altitude, angle bisector, and median
def altitude (t : Triangle) : ℝ := sorry
def angleBisector (t : Triangle) : ℝ := sorry
def median (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem no_triangle_with_special_sides :
  ¬ ∃ (t : Triangle),
    (t.a = altitude t ∧ t.b = angleBisector t ∧ t.c = median t) ∨
    (t.a = altitude t ∧ t.b = median t ∧ t.c = angleBisector t) ∨
    (t.a = angleBisector t ∧ t.b = altitude t ∧ t.c = median t) ∨
    (t.a = angleBisector t ∧ t.b = median t ∧ t.c = altitude t) ∨
    (t.a = median t ∧ t.b = altitude t ∧ t.c = angleBisector t) ∨
    (t.a = median t ∧ t.b = angleBisector t ∧ t.c = altitude t) := by
  sorry

end NUMINAMATH_CALUDE_no_triangle_with_special_sides_l2925_292539


namespace NUMINAMATH_CALUDE_new_energy_vehicle_sales_growth_rate_l2925_292578

theorem new_energy_vehicle_sales_growth_rate 
  (january_sales : ℕ) 
  (march_sales : ℕ) 
  (growth_rate : ℝ) : 
  january_sales = 25 → 
  march_sales = 36 → 
  (1 + growth_rate)^2 = march_sales / january_sales → 
  growth_rate = 0.2 := by
sorry

end NUMINAMATH_CALUDE_new_energy_vehicle_sales_growth_rate_l2925_292578


namespace NUMINAMATH_CALUDE_unique_root_in_unit_interval_l2925_292575

theorem unique_root_in_unit_interval :
  ∃! α : ℝ, |α| < 1 ∧ α^3 - 2*α + 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_unique_root_in_unit_interval_l2925_292575


namespace NUMINAMATH_CALUDE_response_rate_is_sixty_percent_l2925_292577

/-- The response rate percentage for a questionnaire mailing --/
def response_rate_percentage (responses_needed : ℕ) (questionnaires_mailed : ℕ) : ℚ :=
  (responses_needed : ℚ) / (questionnaires_mailed : ℚ) * 100

/-- Theorem stating that the response rate percentage is 60% given the specified conditions --/
theorem response_rate_is_sixty_percent 
  (responses_needed : ℕ) 
  (questionnaires_mailed : ℕ) 
  (h1 : responses_needed = 300) 
  (h2 : questionnaires_mailed = 500) : 
  response_rate_percentage responses_needed questionnaires_mailed = 60 := by
  sorry

#eval response_rate_percentage 300 500

end NUMINAMATH_CALUDE_response_rate_is_sixty_percent_l2925_292577


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2925_292585

theorem inequality_solution_set (x : ℝ) : 
  (x - 1) * (x - 2) * (x - 3)^2 > 0 ↔ x < 1 ∨ 1 < x ∧ x < 2 ∨ 2 < x ∧ x < 3 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2925_292585


namespace NUMINAMATH_CALUDE_a_nonzero_sufficient_not_necessary_l2925_292514

/-- A cubic polynomial function -/
def cubic_polynomial (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The property that a cubic polynomial has a root -/
def has_root (a b c d : ℝ) : Prop := ∃ x : ℝ, cubic_polynomial a b c d x = 0

/-- The statement that "a≠0" is sufficient but not necessary for a cubic polynomial to have a root -/
theorem a_nonzero_sufficient_not_necessary :
  (∀ a b c d : ℝ, a ≠ 0 → has_root a b c d) ∧
  ¬(∀ a b c d : ℝ, has_root a b c d → a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_a_nonzero_sufficient_not_necessary_l2925_292514


namespace NUMINAMATH_CALUDE_m_range_l2925_292561

theorem m_range : ∃ m : ℝ, m = 3 * Real.sqrt 2 - 1 ∧ 3 < m ∧ m < 4 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l2925_292561


namespace NUMINAMATH_CALUDE_rectangle_puzzle_l2925_292556

-- Define the lengths of the segments
def top_segment1 : ℝ := 2
def top_segment2 : ℝ := 3
def top_segment4 : ℝ := 4
def bottom_segment1 : ℝ := 3
def bottom_segment2 : ℝ := 5

-- Define X as a real number
def X : ℝ := sorry

-- State the theorem
theorem rectangle_puzzle :
  top_segment1 + top_segment2 + X + top_segment4 = bottom_segment1 + bottom_segment2 + (X + 1) →
  X = 1 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_puzzle_l2925_292556


namespace NUMINAMATH_CALUDE_mo_hot_chocolate_consumption_l2925_292576

/-- The number of cups of hot chocolate Mo drinks on rainy mornings -/
def cups_of_hot_chocolate : ℚ := 1.75

/-- The number of cups of tea Mo drinks on non-rainy mornings -/
def cups_of_tea_non_rainy : ℕ := 5

/-- The total number of cups of tea and hot chocolate Mo drank last week -/
def total_cups_last_week : ℕ := 22

/-- The difference between tea cups and hot chocolate cups Mo drank last week -/
def tea_minus_chocolate : ℕ := 8

/-- The number of rainy days last week -/
def rainy_days : ℕ := 4

/-- The number of days in a week -/
def days_in_week : ℕ := 7

theorem mo_hot_chocolate_consumption :
  cups_of_hot_chocolate * rainy_days = 
    total_cups_last_week - (cups_of_tea_non_rainy * (days_in_week - rainy_days)) - tea_minus_chocolate := by
  sorry

end NUMINAMATH_CALUDE_mo_hot_chocolate_consumption_l2925_292576


namespace NUMINAMATH_CALUDE_abc_sum_l2925_292513

theorem abc_sum (a b c : ℕ+) 
  (h1 : a * b + c = 57)
  (h2 : b * c + a = 57)
  (h3 : a * c + b = 57) : 
  a + b + c = 9 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_l2925_292513


namespace NUMINAMATH_CALUDE_triangle_area_determines_p_l2925_292515

/-- Given a triangle ABC with vertices A(3, 15), B(15, 0), and C(0, p),
    prove that if the area of the triangle is 36, then p = 12.75 -/
theorem triangle_area_determines_p :
  let A : ℝ × ℝ := (3, 15)
  let B : ℝ × ℝ := (15, 0)
  let C : ℝ × ℝ := (0, p)
  let triangle_area (A B C : ℝ × ℝ) : ℝ :=
    (1/2) * abs ((A.1 - C.1) * (B.2 - C.2) - (B.1 - C.1) * (A.2 - C.2))
  ∀ p : ℝ, triangle_area A B C = 36 → p = 12.75 := by
  sorry

#check triangle_area_determines_p

end NUMINAMATH_CALUDE_triangle_area_determines_p_l2925_292515


namespace NUMINAMATH_CALUDE_absolute_difference_l2925_292598

theorem absolute_difference (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : 
  |x + 1| - |x - 3| = -4 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_l2925_292598


namespace NUMINAMATH_CALUDE_pyramid_volume_transformation_l2925_292592

theorem pyramid_volume_transformation (s h : ℝ) : 
  (1/3 : ℝ) * s^2 * h = 72 → 
  (1/3 : ℝ) * (3*s)^2 * (2*h) = 1296 := by
sorry

end NUMINAMATH_CALUDE_pyramid_volume_transformation_l2925_292592


namespace NUMINAMATH_CALUDE_solution_xyz_l2925_292549

theorem solution_xyz (x y z : ℝ) 
  (eq1 : 2*x + y = 4) 
  (eq2 : x + 2*y = 5) 
  (eq3 : 3*x - 1.5*y + z = 7) : 
  (x + y + z) / 3 = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_xyz_l2925_292549


namespace NUMINAMATH_CALUDE_emir_savings_correct_l2925_292580

/-- The amount Emir has saved from his allowance -/
def emirSavings (dictionaryCost cookbookCost dinosaurBookCost additionalNeeded : ℕ) : ℕ :=
  dictionaryCost + cookbookCost + dinosaurBookCost - additionalNeeded

theorem emir_savings_correct (dictionaryCost cookbookCost dinosaurBookCost additionalNeeded : ℕ) :
  emirSavings dictionaryCost cookbookCost dinosaurBookCost additionalNeeded =
  dictionaryCost + cookbookCost + dinosaurBookCost - additionalNeeded :=
by sorry

end NUMINAMATH_CALUDE_emir_savings_correct_l2925_292580


namespace NUMINAMATH_CALUDE_paper_stack_height_l2925_292509

/-- Given a package of paper with known thickness and number of sheets,
    calculate the height of a stack with a different number of sheets. -/
theorem paper_stack_height
  (package_sheets : ℕ)
  (package_thickness : ℝ)
  (stack_sheets : ℕ)
  (h_package_sheets : package_sheets = 400)
  (h_package_thickness : package_thickness = 4)
  (h_stack_sheets : stack_sheets = 1000) :
  (stack_sheets : ℝ) * package_thickness / package_sheets = 10 :=
sorry

end NUMINAMATH_CALUDE_paper_stack_height_l2925_292509


namespace NUMINAMATH_CALUDE_candy_bar_cost_l2925_292553

/-- The cost of the candy bar given the total spent and the cost of cookies -/
theorem candy_bar_cost (total_spent : ℕ) (cookie_cost : ℕ) (h1 : total_spent = 53) (h2 : cookie_cost = 39) :
  total_spent - cookie_cost = 14 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_cost_l2925_292553


namespace NUMINAMATH_CALUDE_palm_meadows_rooms_l2925_292599

theorem palm_meadows_rooms (two_bed_rooms three_bed_rooms : ℕ) : 
  two_bed_rooms = 8 →
  two_bed_rooms * 2 + three_bed_rooms * 3 = 31 →
  two_bed_rooms + three_bed_rooms = 13 := by
sorry

end NUMINAMATH_CALUDE_palm_meadows_rooms_l2925_292599


namespace NUMINAMATH_CALUDE_exam_score_below_mean_l2925_292502

/-- Given an exam with mean score and a known score above the mean,
    calculate the score that is a certain number of standard deviations below the mean. -/
theorem exam_score_below_mean 
  (mean : ℝ) 
  (score_above : ℝ) 
  (sd_above : ℝ) 
  (sd_below : ℝ) 
  (h1 : mean = 88.8)
  (h2 : score_above = 90)
  (h3 : sd_above = 3)
  (h4 : sd_below = 7)
  (h5 : score_above = mean + sd_above * ((score_above - mean) / sd_above)) :
  mean - sd_below * ((score_above - mean) / sd_above) = 86 := by
sorry


end NUMINAMATH_CALUDE_exam_score_below_mean_l2925_292502


namespace NUMINAMATH_CALUDE_sum_of_solutions_abs_eq_l2925_292500

theorem sum_of_solutions_abs_eq (x : ℝ) : (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |3 * x₁ - 12| = 6 ∧ |3 * x₂ - 12| = 6 ∧ x₁ + x₂ = 8) ∧ (∀ x : ℝ, |3 * x - 12| = 6 → x = 2 ∨ x = 6) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_abs_eq_l2925_292500


namespace NUMINAMATH_CALUDE_ana_bonita_age_difference_ana_bonita_age_difference_proof_l2925_292555

theorem ana_bonita_age_difference : ℕ → Prop := fun n =>
  ∀ (A B : ℕ),
    A = B + n →                    -- Ana is n years older than Bonita
    A - 1 = 3 * (B - 1) →          -- Last year Ana was 3 times as old as Bonita
    A = B * B →                    -- This year Ana's age is the square of Bonita's age
    n = 2                          -- The age difference is 2 years

-- The proof goes here
theorem ana_bonita_age_difference_proof : ana_bonita_age_difference 2 := by
  sorry

#check ana_bonita_age_difference_proof

end NUMINAMATH_CALUDE_ana_bonita_age_difference_ana_bonita_age_difference_proof_l2925_292555


namespace NUMINAMATH_CALUDE_scooter_price_proof_l2925_292503

theorem scooter_price_proof (initial_price : ℝ) : 
  (∃ (total_cost selling_price : ℝ),
    total_cost = initial_price + 300 ∧
    selling_price = 1260 ∧
    selling_price = total_cost * 1.05) →
  initial_price = 900 := by
sorry

end NUMINAMATH_CALUDE_scooter_price_proof_l2925_292503


namespace NUMINAMATH_CALUDE_perfect_square_property_l2925_292508

theorem perfect_square_property (n : ℤ) : 
  (∃ k : ℤ, 2 + 2 * Real.sqrt (1 + 12 * n^2) = k) → 
  ∃ m : ℤ, (2 + 2 * Real.sqrt (1 + 12 * n^2))^2 = m^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_property_l2925_292508


namespace NUMINAMATH_CALUDE_least_number_with_remainder_four_ninety_four_satisfies_conditions_ninety_four_is_least_number_l2925_292518

theorem least_number_with_remainder_four (n : ℕ) : 
  (n % 5 = 4 ∧ n % 6 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4) → n ≥ 94 :=
by sorry

theorem ninety_four_satisfies_conditions : 
  94 % 5 = 4 ∧ 94 % 6 = 4 ∧ 94 % 9 = 4 ∧ 94 % 18 = 4 :=
by sorry

theorem ninety_four_is_least_number : 
  ∀ n : ℕ, (n % 5 = 4 ∧ n % 6 = 4 ∧ n % 9 = 4 ∧ n % 18 = 4) → n ≥ 94 :=
by sorry

end NUMINAMATH_CALUDE_least_number_with_remainder_four_ninety_four_satisfies_conditions_ninety_four_is_least_number_l2925_292518


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2925_292511

/-- Sum of a geometric series with n terms -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/6
  let r : ℚ := -1/2
  let n : ℕ := 7
  geometric_sum a r n = 129/1152 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2925_292511


namespace NUMINAMATH_CALUDE_altitude_equation_tangent_lines_equal_intercepts_l2925_292571

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y = 0

-- Define the center of circle C
def center_C : ℝ × ℝ := (-1, 1)

-- Define points A and B
def point_A : ℝ × ℝ := (4, 0)
def point_B : ℝ × ℝ := (0, -2)

-- Theorem for the altitude equation
theorem altitude_equation :
  ∃ (x y : ℝ), 2*x + y + 1 = 0 ∧
  (∀ (p : ℝ × ℝ), p.1 = x ∧ p.2 = y →
    (p.2 - center_C.2) = -2 * (p.1 - center_C.1) ∧
    (p.2 - point_A.2) * (point_B.1 - point_A.1) = -(p.1 - point_A.1) * (point_B.2 - point_A.2)) :=
sorry

-- Theorem for tangent lines with equal intercepts
theorem tangent_lines_equal_intercepts :
  (∀ (x y : ℝ), (x - y = 0 ∨ x + y - 2 = 0 ∨ x + y + 2 = 0) →
    (∃ (t : ℝ), x = t ∧ y = t) ∨
    (∃ (t : ℝ), x = t ∧ y = 2 - t) ∨
    (∃ (t : ℝ), x = t ∧ y = -2 - t)) ∧
  (∀ (x y : ℝ), ((∃ (t : ℝ), x = t ∧ y = t) ∨
                 (∃ (t : ℝ), x = t ∧ y = 2 - t) ∨
                 (∃ (t : ℝ), x = t ∧ y = -2 - t)) →
    (x - center_C.1)^2 + (y - center_C.2)^2 = 2) :=
sorry

end NUMINAMATH_CALUDE_altitude_equation_tangent_lines_equal_intercepts_l2925_292571


namespace NUMINAMATH_CALUDE_rearrange_pegs_l2925_292537

/-- Represents a position on the board --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents the board state --/
def BoardState := List Position

/-- Checks if a given arrangement of pegs satisfies the condition of 5 rows with 4 pegs each --/
def isValidArrangement (arrangement : BoardState) : Bool :=
  sorry

/-- Counts the number of pegs that need to be moved to transform one arrangement into another --/
def pegsMoved (initial : BoardState) (final : BoardState) : Nat :=
  sorry

/-- The main theorem stating that it's possible to achieve the desired arrangement by moving exactly 3 pegs --/
theorem rearrange_pegs (initial : BoardState) :
  (initial.length = 10) →
  ∃ (final : BoardState), 
    isValidArrangement final ∧ 
    pegsMoved initial final = 3 :=
  sorry

end NUMINAMATH_CALUDE_rearrange_pegs_l2925_292537


namespace NUMINAMATH_CALUDE_inequality_solution_l2925_292516

theorem inequality_solution (x : Real) : 
  0 ≤ x ∧ x ≤ 2 * Real.pi →
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ∧
   |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔
  ((Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2) ∨ (3 * Real.pi / 2 ≤ x ∧ x ≤ 7 * Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2925_292516


namespace NUMINAMATH_CALUDE_min_value_quadratic_l2925_292512

theorem min_value_quadratic (k : ℝ) : 
  (∀ x y : ℝ, 3 * x^2 - 4 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 ≥ 0) ∧ 
  (∃ x y : ℝ, 3 * x^2 - 4 * k * x * y + (2 * k^2 + 3) * y^2 - 6 * x - 3 * y + 9 = 0) ↔ 
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l2925_292512


namespace NUMINAMATH_CALUDE_chantel_bracelets_l2925_292525

/-- The number of bracelets Chantel gave away at soccer practice -/
def bracelets_given_at_soccer : ℕ := sorry

/-- The number of days Chantel makes 2 bracelets per day -/
def days_making_two : ℕ := 5

/-- The number of bracelets Chantel makes per day in the first period -/
def bracelets_per_day_first : ℕ := 2

/-- The number of bracelets Chantel gives away at school -/
def bracelets_given_at_school : ℕ := 3

/-- The number of days Chantel makes 3 bracelets per day -/
def days_making_three : ℕ := 4

/-- The number of bracelets Chantel makes per day in the second period -/
def bracelets_per_day_second : ℕ := 3

/-- The number of bracelets Chantel has at the end -/
def bracelets_at_end : ℕ := 13

theorem chantel_bracelets : 
  bracelets_given_at_soccer = 
    days_making_two * bracelets_per_day_first + 
    days_making_three * bracelets_per_day_second - 
    bracelets_given_at_school - 
    bracelets_at_end := by sorry

end NUMINAMATH_CALUDE_chantel_bracelets_l2925_292525


namespace NUMINAMATH_CALUDE_mans_rate_l2925_292522

/-- The man's rate in still water given his speeds with and against the stream -/
theorem mans_rate (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 22)
  (h2 : speed_against_stream = 10) :
  (speed_with_stream + speed_against_stream) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_mans_rate_l2925_292522


namespace NUMINAMATH_CALUDE_brennan_pepper_amount_l2925_292533

def initial_pepper : ℝ := 0.25
def used_pepper : ℝ := 0.16
def remaining_pepper : ℝ := 0.09

theorem brennan_pepper_amount :
  initial_pepper = used_pepper + remaining_pepper :=
by sorry

end NUMINAMATH_CALUDE_brennan_pepper_amount_l2925_292533


namespace NUMINAMATH_CALUDE_nancy_carrots_l2925_292540

/-- The number of carrots Nancy threw out -/
def carrots_thrown_out : ℕ := 2

/-- The number of carrots Nancy initially picked -/
def initial_carrots : ℕ := 12

/-- The number of carrots Nancy picked the next day -/
def next_day_carrots : ℕ := 21

/-- The total number of carrots Nancy ended up with -/
def total_carrots : ℕ := 31

theorem nancy_carrots :
  initial_carrots - carrots_thrown_out + next_day_carrots = total_carrots :=
by sorry

end NUMINAMATH_CALUDE_nancy_carrots_l2925_292540


namespace NUMINAMATH_CALUDE_floor_negative_seven_fourths_l2925_292588

theorem floor_negative_seven_fourths : ⌊(-7 : ℚ) / 4⌋ = -2 := by
  sorry

end NUMINAMATH_CALUDE_floor_negative_seven_fourths_l2925_292588


namespace NUMINAMATH_CALUDE_janice_stairs_problem_l2925_292551

/-- The number of times Janice goes down the stairs in a day -/
def times_down (flights_per_floor : ℕ) (times_up : ℕ) (total_flights : ℕ) : ℕ :=
  (total_flights - flights_per_floor * times_up) / flights_per_floor

theorem janice_stairs_problem (flights_per_floor : ℕ) (times_up : ℕ) (total_flights : ℕ) 
    (h1 : flights_per_floor = 3)
    (h2 : times_up = 5)
    (h3 : total_flights = 24) :
  times_down flights_per_floor times_up total_flights = 3 := by
  sorry

#eval times_down 3 5 24

end NUMINAMATH_CALUDE_janice_stairs_problem_l2925_292551


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l2925_292538

/-- Given a polynomial x^4 - x^3 + x^2 + ax + b that is a perfect square,
    prove that b = 9/64 -/
theorem perfect_square_polynomial (a b : ℚ) : 
  (∃ p q r : ℚ, ∀ x, x^4 - x^3 + x^2 + a*x + b = (p*x^2 + q*x + r)^2) →
  b = 9/64 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l2925_292538


namespace NUMINAMATH_CALUDE_cube_edge_ratio_l2925_292519

theorem cube_edge_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_vol : a^3 / b^3 = 27 / 8) : 
  a / b = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_cube_edge_ratio_l2925_292519


namespace NUMINAMATH_CALUDE_xOzSymmetry_of_A_l2925_292565

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Performs symmetry transformation with respect to the xOz plane -/
def xOzSymmetry (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := p.z }

/-- The original point A -/
def A : Point3D :=
  { x := 2, y := -3, z := 1 }

/-- The expected result after symmetry -/
def expectedResult : Point3D :=
  { x := 2, y := 3, z := 1 }

theorem xOzSymmetry_of_A : xOzSymmetry A = expectedResult := by
  sorry

end NUMINAMATH_CALUDE_xOzSymmetry_of_A_l2925_292565


namespace NUMINAMATH_CALUDE_stock_price_fluctuation_l2925_292526

theorem stock_price_fluctuation (original_price : ℝ) (h : original_price > 0) :
  let increased_price := original_price * 1.4
  let decrease_factor := 1 - 0.2857
  decrease_factor * increased_price = original_price := by
  sorry

end NUMINAMATH_CALUDE_stock_price_fluctuation_l2925_292526


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l2925_292507

/-- The function g(x) as defined in the problem -/
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8|

/-- The theorem stating that the sum of the maximum and minimum values of g(x) over [1, 10] is 2 -/
theorem sum_of_max_min_g :
  (⨆ (x : ℝ) (h : x ∈ Set.Icc 1 10), g x) + (⨅ (x : ℝ) (h : x ∈ Set.Icc 1 10), g x) = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l2925_292507


namespace NUMINAMATH_CALUDE_f_property_l2925_292574

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then Real.sqrt x
  else if x ≥ 1 then 2 * (x - 1)
  else 0  -- We define f as 0 for x ≤ 0 to make it total

-- State the theorem
theorem f_property : ∃ a : ℝ, f a = f (a + 1) → f (1 / a) = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l2925_292574


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l2925_292570

theorem factorization_difference_of_squares (m n : ℝ) : (m + n)^2 - (m - n)^2 = 4 * m * n := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l2925_292570


namespace NUMINAMATH_CALUDE_student_count_l2925_292531

theorem student_count (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 16) 
  (h2 : rank_from_left = 6) : 
  rank_from_right + rank_from_left - 1 = 21 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2925_292531


namespace NUMINAMATH_CALUDE_participation_plans_specific_l2925_292562

/-- The number of ways to select three students from four, with one student always selected,
    for three different subjects. -/
def participation_plans (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  (n - 1).choose (k - 1) * m.factorial

theorem participation_plans_specific : participation_plans 4 3 3 = 18 := by
  sorry

#eval participation_plans 4 3 3

end NUMINAMATH_CALUDE_participation_plans_specific_l2925_292562


namespace NUMINAMATH_CALUDE_double_money_l2925_292506

theorem double_money (initial_amount : ℕ) : 
  initial_amount + initial_amount = 2 * initial_amount := by
  sorry

#check double_money

end NUMINAMATH_CALUDE_double_money_l2925_292506


namespace NUMINAMATH_CALUDE_solution_is_correct_l2925_292579

/-- The intersection point of two lines -/
def intersection_point (l1 l2 : ℝ → ℝ → ℝ) : ℝ × ℝ :=
  (1, 3)  -- We define this based on the given lines, without solving the system

/-- Checks if two lines are parallel -/
def are_parallel (l1 l2 : ℝ → ℝ → ℝ) : Prop :=
  ∃ (k : ℝ), ∀ x y, l1 x y = k * l2 x y

/-- The first given line -/
def line1 (x y : ℝ) : ℝ := 3*x - 2*y + 3

/-- The second given line -/
def line2 (x y : ℝ) : ℝ := x + y - 4

/-- The line parallel to which we need to find our solution -/
def parallel_line (x y : ℝ) : ℝ := 2*x + y - 1

/-- The proposed solution line -/
def solution_line (x y : ℝ) : ℝ := 2*x + y - 5

theorem solution_is_correct : 
  let (ix, iy) := intersection_point line1 line2
  solution_line ix iy = 0 ∧ 
  are_parallel solution_line parallel_line :=
by sorry

end NUMINAMATH_CALUDE_solution_is_correct_l2925_292579


namespace NUMINAMATH_CALUDE_community_size_after_five_years_l2925_292595

def community_growth (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | m + 1 => 4 * community_growth m - 15

theorem community_size_after_five_years :
  community_growth 5 = 15365 := by
  sorry

end NUMINAMATH_CALUDE_community_size_after_five_years_l2925_292595


namespace NUMINAMATH_CALUDE_range_of_a_l2925_292521

theorem range_of_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ x : ℝ, |x - 4| + |x - 3| < a) : a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2925_292521


namespace NUMINAMATH_CALUDE_total_weight_is_7000_l2925_292501

/-- The weight of the truck in pounds -/
def truck_weight : ℝ := 4800

/-- The weight of the trailer in pounds -/
def trailer_weight : ℝ := 0.5 * truck_weight - 200

/-- The total weight of the truck and trailer in pounds -/
def total_weight : ℝ := truck_weight + trailer_weight

/-- Theorem stating that the total weight of the truck and trailer is 7000 pounds -/
theorem total_weight_is_7000 : total_weight = 7000 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_is_7000_l2925_292501


namespace NUMINAMATH_CALUDE_cube_vector_sum_divisible_by_11_l2925_292593

/-- The size of the cube. -/
def cubeSize : ℕ := 1000

/-- The sum of squares of integers from 0 to n. -/
def sumOfSquares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The sum of squares of lengths of vectors from origin to all integer points in the cube. -/
def sumOfVectorLengthSquares : ℕ :=
  3 * (cubeSize + 1)^2 * sumOfSquares cubeSize

theorem cube_vector_sum_divisible_by_11 :
  sumOfVectorLengthSquares % 11 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_vector_sum_divisible_by_11_l2925_292593


namespace NUMINAMATH_CALUDE_circle_equation_l2925_292544

theorem circle_equation (x y : ℝ) : 
  (∃ c : ℝ × ℝ, (x - c.1)^2 + (y - c.2)^2 = 8^2) ↔ 
  x^2 + 14*x + y^2 + 8*y + 1 = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l2925_292544


namespace NUMINAMATH_CALUDE_koby_sparklers_count_l2925_292564

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of boxes Cherie has -/
def cherie_boxes : ℕ := 1

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 33

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

theorem koby_sparklers_count :
  koby_sparklers_per_box * koby_boxes +
  cherie_sparklers +
  koby_whistlers_per_box * koby_boxes +
  cherie_whistlers = total_fireworks :=
by sorry

end NUMINAMATH_CALUDE_koby_sparklers_count_l2925_292564


namespace NUMINAMATH_CALUDE_davids_biology_marks_l2925_292596

/-- Given David's marks in four subjects and his average marks, calculate his marks in Biology. -/
theorem davids_biology_marks
  (english_marks : ℕ)
  (math_marks : ℕ)
  (physics_marks : ℕ)
  (chemistry_marks : ℕ)
  (average_marks : ℚ)
  (h1 : english_marks = 96)
  (h2 : math_marks = 98)
  (h3 : physics_marks = 99)
  (h4 : chemistry_marks = 100)
  (h5 : average_marks = 98.2)
  (h6 : average_marks = (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks : ℚ) / 5) :
  biology_marks = 98 := by
  sorry

#check davids_biology_marks

end NUMINAMATH_CALUDE_davids_biology_marks_l2925_292596


namespace NUMINAMATH_CALUDE_exists_ten_points_five_kites_l2925_292597

/-- A point on a 4x4 grid --/
structure GridPoint where
  x : Fin 4
  y : Fin 4

/-- A kite formed by four points on the grid --/
structure Kite where
  p1 : GridPoint
  p2 : GridPoint
  p3 : GridPoint
  p4 : GridPoint

/-- Check if four points form a valid kite --/
def is_valid_kite (k : Kite) : Prop :=
  -- Two pairs of adjacent sides have equal length
  -- Diagonals intersect at a right angle
  -- One diagonal bisects the other
  sorry

/-- Count the number of kites formed by a set of points --/
def count_kites (points : Finset GridPoint) : Nat :=
  sorry

/-- Theorem stating that there exists an arrangement of 10 points forming exactly 5 kites --/
theorem exists_ten_points_five_kites :
  ∃ (points : Finset GridPoint),
    points.card = 10 ∧ count_kites points = 5 :=
  sorry

end NUMINAMATH_CALUDE_exists_ten_points_five_kites_l2925_292597


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l2925_292587

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- The theorem states that if point A(a, 1) and point B(-2, b) are symmetric with respect to the origin,
    then a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) :
  symmetric_wrt_origin a 1 (-2) b → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l2925_292587


namespace NUMINAMATH_CALUDE_phone_prob_theorem_l2925_292559

def phone_prob (p1 p2 p3 : ℝ) : Prop :=
  p1 = 0.5 ∧ p2 = 0.3 ∧ p3 = 0.2 →
  p1 + p2 = 0.8

theorem phone_prob_theorem :
  ∀ p1 p2 p3 : ℝ, phone_prob p1 p2 p3 :=
by
  sorry

end NUMINAMATH_CALUDE_phone_prob_theorem_l2925_292559


namespace NUMINAMATH_CALUDE_fred_final_cards_l2925_292546

/-- The number of baseball cards Fred has after various transactions -/
def fred_cards (initial : ℕ) (given_away : ℕ) (new_cards : ℕ) : ℕ :=
  initial - given_away + new_cards

/-- Theorem stating that Fred ends up with 48 cards given the specific numbers in the problem -/
theorem fred_final_cards : fred_cards 26 18 40 = 48 := by
  sorry

end NUMINAMATH_CALUDE_fred_final_cards_l2925_292546


namespace NUMINAMATH_CALUDE_c_range_l2925_292584

def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y

def q (c : ℝ) : Prop := ∀ x : ℝ, x ∈ Set.Icc (1/2) 2 → x + 1/x > 1/c

def range_c (c : ℝ) : Prop := (0 < c ∧ c ≤ 1/2) ∨ c ≥ 1

theorem c_range (c : ℝ) (h1 : c > 0) (h2 : (p c ∨ q c) ∧ ¬(p c ∧ q c)) : range_c c := by
  sorry

end NUMINAMATH_CALUDE_c_range_l2925_292584


namespace NUMINAMATH_CALUDE_product_equality_l2925_292586

theorem product_equality (a b : ℤ) : 
  (∃ C : ℤ, a * (a - 5) = C ∧ b * (b - 8) = C) → 
  (a * (a - 5) = 0 ∨ a * (a - 5) = 84) :=
by sorry

end NUMINAMATH_CALUDE_product_equality_l2925_292586


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2925_292542

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ (∃ x : ℝ, x^2 - x + 2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2925_292542


namespace NUMINAMATH_CALUDE_mildreds_father_oranges_l2925_292528

/-- The number of oranges Mildred's father ate -/
def oranges_eaten (initial : ℝ) (remaining : ℝ) : ℝ :=
  initial - remaining

/-- Proof that Mildred's father ate 2.0 oranges -/
theorem mildreds_father_oranges : oranges_eaten 77.0 75 = 2.0 := by
  sorry

end NUMINAMATH_CALUDE_mildreds_father_oranges_l2925_292528


namespace NUMINAMATH_CALUDE_overall_score_calculation_l2925_292594

/-- Calculate the overall score for a job applicant given their test scores and weights -/
theorem overall_score_calculation
  (written_score : ℝ)
  (interview_score : ℝ)
  (written_weight : ℝ)
  (interview_weight : ℝ)
  (h1 : written_score = 80)
  (h2 : interview_score = 60)
  (h3 : written_weight = 0.6)
  (h4 : interview_weight = 0.4)
  (h5 : written_weight + interview_weight = 1) :
  written_score * written_weight + interview_score * interview_weight = 72 :=
by sorry

end NUMINAMATH_CALUDE_overall_score_calculation_l2925_292594


namespace NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2925_292589

theorem largest_n_satisfying_inequality : 
  (∀ n : ℕ, n ≤ 7 → (1 : ℚ) / 4 + n / 6 < 3 / 2) ∧ 
  (∀ n : ℕ, n > 7 → (1 : ℚ) / 4 + n / 6 ≥ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_inequality_l2925_292589


namespace NUMINAMATH_CALUDE_codecracker_combinations_l2925_292504

/-- The number of available colors for the CodeCracker game -/
def num_colors : ℕ := 7

/-- The number of slots in the master code -/
def code_length : ℕ := 5

/-- The number of different master codes that can be formed in the CodeCracker game -/
def num_codes : ℕ := num_colors ^ code_length

theorem codecracker_combinations : num_codes = 16807 := by
  sorry

end NUMINAMATH_CALUDE_codecracker_combinations_l2925_292504


namespace NUMINAMATH_CALUDE_factor_polynomial_l2925_292583

theorem factor_polynomial (x : ℝ) : 90 * x^3 - 135 * x^9 = 45 * x^3 * (2 - 3 * x^6) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l2925_292583
