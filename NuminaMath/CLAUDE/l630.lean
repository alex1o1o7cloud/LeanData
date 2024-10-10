import Mathlib

namespace arrangement_existence_l630_63034

/-- Represents a group of kindergarten children -/
structure ChildrenGroup where
  total : ℕ  -- Total number of children

/-- Represents an arrangement of children in pairs -/
structure Arrangement where
  boy_pairs : ℕ  -- Number of pairs of two boys
  girl_pairs : ℕ  -- Number of pairs of two girls
  mixed_pairs : ℕ  -- Number of pairs with one boy and one girl

/-- Checks if an arrangement is valid for a given group -/
def is_valid_arrangement (group : ChildrenGroup) (arr : Arrangement) : Prop :=
  2 * (arr.boy_pairs + arr.girl_pairs) + arr.mixed_pairs = group.total

/-- Theorem stating the existence of a specific arrangement -/
theorem arrangement_existence (group : ChildrenGroup) 
  (arr1 arr2 : Arrangement) 
  (h1 : is_valid_arrangement group arr1)
  (h2 : is_valid_arrangement group arr2)
  (h3 : arr1.boy_pairs = 3 * arr1.girl_pairs)
  (h4 : arr2.boy_pairs = 4 * arr2.girl_pairs) :
  ∃ (arr3 : Arrangement), 
    is_valid_arrangement group arr3 ∧ 
    arr3.boy_pairs = 7 * arr3.girl_pairs := by
  sorry

end arrangement_existence_l630_63034


namespace new_apartment_rent_is_1400_l630_63023

/-- Calculates the monthly rent of John's new apartment -/
def new_apartment_rent (former_rent_per_sqft : ℚ) (former_sqft : ℕ) (annual_savings : ℚ) : ℚ :=
  let former_monthly_rent := former_rent_per_sqft * former_sqft
  let former_annual_rent := former_monthly_rent * 12
  let new_annual_rent := former_annual_rent - annual_savings
  new_annual_rent / 12

/-- Proves that the monthly rent of John's new apartment is $1400 -/
theorem new_apartment_rent_is_1400 :
  new_apartment_rent 2 750 1200 = 1400 := by sorry

end new_apartment_rent_is_1400_l630_63023


namespace bridgette_dogs_l630_63060

/-- Represents the number of baths given to an animal in a year. -/
def baths_per_year (frequency : ℕ) : ℕ := 12 / frequency

/-- Represents the total number of baths given to a group of animals in a year. -/
def total_baths (num_animals : ℕ) (frequency : ℕ) : ℕ :=
  num_animals * baths_per_year frequency

theorem bridgette_dogs :
  ∃ (num_dogs : ℕ),
    total_baths num_dogs 2 + -- Dogs bathed twice a month
    total_baths 3 1 + -- 3 cats bathed once a month
    total_baths 4 4 = 96 ∧ -- 4 birds bathed once every 4 months
    num_dogs = 2 := by
  sorry

end bridgette_dogs_l630_63060


namespace circles_intersect_l630_63018

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are intersecting -/
def are_intersecting (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  abs (c1.radius - c2.radius) < d ∧ d < c1.radius + c2.radius

theorem circles_intersect : 
  let circle1 : Circle := { center := (0, 0), radius := 2 }
  let circle2 : Circle := { center := (2, 0), radius := 3 }
  are_intersecting circle1 circle2 := by
  sorry

end circles_intersect_l630_63018


namespace zoo_animal_difference_l630_63056

theorem zoo_animal_difference (parrots : ℕ) (snakes : ℕ) (monkeys : ℕ) (elephants : ℕ) (zebras : ℕ) : 
  parrots = 8 → 
  snakes = 3 * parrots → 
  monkeys = 2 * snakes → 
  elephants = (parrots + snakes) / 2 → 
  zebras = elephants - 3 → 
  monkeys - zebras = 35 := by
sorry

end zoo_animal_difference_l630_63056


namespace complex_equation_solutions_l630_63025

theorem complex_equation_solutions :
  {z : ℂ | z^6 - 6*z^4 + 9*z^2 = 0} = {0, Complex.I * Real.sqrt 3, -Complex.I * Real.sqrt 3} := by
  sorry

end complex_equation_solutions_l630_63025


namespace planes_formed_by_three_lines_through_point_l630_63051

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in three-dimensional space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- Represents the number of planes formed by three lines -/
inductive NumPlanes
  | one
  | three

/-- Given a point and three lines through it, determines the number of planes formed -/
def planesFormedByThreeLines (p : Point3D) (l1 l2 l3 : Line3D) : NumPlanes :=
  sorry

theorem planes_formed_by_three_lines_through_point 
  (p : Point3D) (l1 l2 l3 : Line3D) 
  (h1 : l1.point = p) (h2 : l2.point = p) (h3 : l3.point = p) :
  planesFormedByThreeLines p l1 l2 l3 = NumPlanes.one ∨ 
  planesFormedByThreeLines p l1 l2 l3 = NumPlanes.three :=
sorry

end planes_formed_by_three_lines_through_point_l630_63051


namespace middle_digit_guess_probability_l630_63014

/-- Represents a three-digit lock --/
structure DigitLock :=
  (first : Nat)
  (second : Nat)
  (third : Nat)

/-- Condition: Each digit is between 0 and 9 --/
def isValidDigit (d : Nat) : Prop := d ≤ 9

/-- A lock is valid if all its digits are valid --/
def isValidLock (lock : DigitLock) : Prop :=
  isValidDigit lock.first ∧ isValidDigit lock.second ∧ isValidDigit lock.third

/-- The probability of guessing the middle digit correctly --/
def middleDigitGuessProbability (lock : DigitLock) : ℚ :=
  1 / 10

/-- Theorem: The probability of guessing the middle digit of a valid lock is 1/10 --/
theorem middle_digit_guess_probability 
  (lock : DigitLock) 
  (h : isValidLock lock) : 
  middleDigitGuessProbability lock = 1 / 10 := by
  sorry

end middle_digit_guess_probability_l630_63014


namespace white_squares_in_row_l630_63075

/-- Represents a modified stair-step figure where each row begins and ends with a black square
    and has alternating white and black squares. -/
structure ModifiedStairStep where
  /-- The number of squares in the nth row is 2n -/
  squares_in_row : ℕ → ℕ
  /-- Each row begins and ends with a black square -/
  begins_ends_black : ∀ n : ℕ, squares_in_row n ≥ 2
  /-- The number of squares in each row is even -/
  even_squares : ∀ n : ℕ, Even (squares_in_row n)

/-- The number of white squares in the nth row of a modified stair-step figure is equal to n -/
theorem white_squares_in_row (figure : ModifiedStairStep) (n : ℕ) :
  (figure.squares_in_row n) / 2 = n := by
  sorry

end white_squares_in_row_l630_63075


namespace blue_crayon_boxes_l630_63084

/-- Given information about crayon boxes and their contents, prove the number of blue crayon boxes -/
theorem blue_crayon_boxes (total_crayons : ℕ) (orange_boxes : ℕ) (orange_per_box : ℕ) 
  (red_boxes : ℕ) (red_per_box : ℕ) (blue_per_box : ℕ) :
  total_crayons = 94 →
  orange_boxes = 6 →
  orange_per_box = 8 →
  red_boxes = 1 →
  red_per_box = 11 →
  blue_per_box = 5 →
  ∃ (blue_boxes : ℕ), 
    total_crayons = orange_boxes * orange_per_box + red_boxes * red_per_box + blue_boxes * blue_per_box ∧
    blue_boxes = 7 := by
  sorry

end blue_crayon_boxes_l630_63084


namespace distribute_seven_to_twelve_l630_63080

/-- The number of ways to distribute distinct items to recipients -/
def distribute_items (num_items : ℕ) (num_recipients : ℕ) : ℕ :=
  num_recipients ^ num_items

/-- Theorem: Distributing 7 distinct items to 12 recipients results in 35,831,808 ways -/
theorem distribute_seven_to_twelve :
  distribute_items 7 12 = 35831808 := by
  sorry

end distribute_seven_to_twelve_l630_63080


namespace long_furred_brown_dogs_l630_63076

theorem long_furred_brown_dogs (total : ℕ) (long_furred : ℕ) (brown : ℕ) (neither : ℕ) :
  total = 45 →
  long_furred = 29 →
  brown = 17 →
  neither = 8 →
  long_furred + brown - (total - neither) = 9 :=
by sorry

end long_furred_brown_dogs_l630_63076


namespace function_ordering_l630_63005

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem function_ordering (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_decreasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0) :
  f 3 < f (-2) ∧ f (-2) < f 1 := by
  sorry

end function_ordering_l630_63005


namespace intersection_of_A_and_B_l630_63035

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 5 ≥ 0}
def B : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 5/2 ≤ x ∧ x < 3} := by sorry

end intersection_of_A_and_B_l630_63035


namespace sum_1_to_12_mod_9_l630_63013

theorem sum_1_to_12_mod_9 : (List.sum (List.range 12)).mod 9 = 6 := by
  sorry

end sum_1_to_12_mod_9_l630_63013


namespace linear_equation_property_l630_63006

theorem linear_equation_property (x y : ℝ) (h : x + 6 * y = 17) :
  7 * x + 42 * y = 119 := by
  sorry

end linear_equation_property_l630_63006


namespace lcm_of_1400_and_1050_l630_63074

theorem lcm_of_1400_and_1050 : Nat.lcm 1400 1050 = 4200 := by
  sorry

end lcm_of_1400_and_1050_l630_63074


namespace orchestra_price_is_12_l630_63070

/-- Represents the pricing and sales of theater tickets --/
structure TheaterSales where
  orchestra_price : ℝ
  balcony_price : ℝ
  orchestra_tickets : ℕ
  balcony_tickets : ℕ

/-- Theorem stating the price of orchestra seats given the conditions --/
theorem orchestra_price_is_12 (sales : TheaterSales) :
  sales.balcony_price = 8 ∧
  sales.orchestra_tickets + sales.balcony_tickets = 380 ∧
  sales.orchestra_price * sales.orchestra_tickets + sales.balcony_price * sales.balcony_tickets = 3320 ∧
  sales.balcony_tickets = sales.orchestra_tickets + 240
  → sales.orchestra_price = 12 := by
  sorry


end orchestra_price_is_12_l630_63070


namespace group_size_calculation_l630_63041

theorem group_size_calculation (n : ℕ) : 
  (n : ℝ) * 14 = n * 14 →                   -- Initial average age
  ((n : ℝ) * 14 + 32) / (n + 1) = 16 →      -- New average age
  n = 8 := by
sorry

end group_size_calculation_l630_63041


namespace minimum_trips_for_5000_rubles_l630_63082

theorem minimum_trips_for_5000_rubles :
  ∀ (x y : ℕ),
  31 * x + 32 * y = 5000 →
  x + y ≥ 157 :=
by
  sorry

end minimum_trips_for_5000_rubles_l630_63082


namespace impossible_to_swap_folds_l630_63040

/-- Represents the number of folds on one side of a rhinoceros -/
structure Folds :=
  (vertical : ℕ)
  (horizontal : ℕ)

/-- Represents the state of folds on both sides of a rhinoceros -/
structure RhinoState :=
  (left : Folds)
  (right : Folds)

/-- A scratch operation that can be performed on the rhinoceros -/
inductive ScratchOp
  | left_vertical
  | left_horizontal
  | right_vertical
  | right_horizontal

/-- Defines a valid initial state for the rhinoceros -/
def valid_initial_state (s : RhinoState) : Prop :=
  s.left.vertical + s.left.horizontal + s.right.vertical + s.right.horizontal = 17

/-- Defines the result of applying a scratch operation to a state -/
def apply_scratch (s : RhinoState) (op : ScratchOp) : RhinoState :=
  sorry

/-- Defines when a state is reachable from the initial state through scratching -/
def reachable (initial : RhinoState) (final : RhinoState) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to swap vertical and horizontal folds -/
theorem impossible_to_swap_folds (initial : RhinoState) :
  valid_initial_state initial →
  ¬∃ (final : RhinoState),
    reachable initial final ∧
    final.left.vertical = initial.left.horizontal ∧
    final.left.horizontal = initial.left.vertical ∧
    final.right.vertical = initial.right.horizontal ∧
    final.right.horizontal = initial.right.vertical :=
  sorry

end impossible_to_swap_folds_l630_63040


namespace lineup_constraint_ways_l630_63021

/-- The number of ways to arrange 5 people in a line with constraints -/
def lineupWays : ℕ :=
  let totalPeople : ℕ := 5
  let firstPositionOptions : ℕ := totalPeople - 1
  let lastPositionOptions : ℕ := totalPeople - 2
  let middlePositionsOptions : ℕ := 3 * 2 * 1
  firstPositionOptions * lastPositionOptions * middlePositionsOptions

theorem lineup_constraint_ways :
  lineupWays = 216 := by
  sorry

end lineup_constraint_ways_l630_63021


namespace unique_solution_l630_63081

theorem unique_solution (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.cos (π * x)^2 + 2 * Real.sin (π * y) = 1)
  (h2 : Real.sin (π * x) + Real.sin (π * y) = 0)
  (h3 : x^2 - y^2 = 12) :
  x = 4 ∧ y = 2 := by
sorry

end unique_solution_l630_63081


namespace fan_airflow_in_week_l630_63096

/-- Calculates the total airflow created by a fan in one week -/
theorem fan_airflow_in_week 
  (airflow_rate : ℝ) 
  (daily_operation_time : ℝ) 
  (days_in_week : ℕ) 
  (seconds_in_minute : ℕ) : 
  airflow_rate * daily_operation_time * (days_in_week : ℝ) * (seconds_in_minute : ℝ) = 42000 :=
by
  -- Assuming airflow_rate = 10, daily_operation_time = 10, days_in_week = 7, seconds_in_minute = 60
  sorry

#check fan_airflow_in_week

end fan_airflow_in_week_l630_63096


namespace initial_cards_l630_63048

theorem initial_cards (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 3 → total = 7 → initial + added = total → initial = 4 := by
  sorry

end initial_cards_l630_63048


namespace initial_average_mark_l630_63032

/-- Proves that the initial average mark of a class is 60, given the specified conditions. -/
theorem initial_average_mark (total_students : ℕ) (excluded_students : ℕ) (excluded_avg : ℚ) (remaining_avg : ℚ) :
  total_students = 9 →
  excluded_students = 5 →
  excluded_avg = 44 →
  remaining_avg = 80 →
  (total_students * (total_students * excluded_avg + (total_students - excluded_students) * remaining_avg)) / 
  (excluded_students * total_students + (total_students - excluded_students) * total_students) = 60 :=
by sorry

end initial_average_mark_l630_63032


namespace prop_1_prop_3_l630_63015

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Define non-coinciding lines and planes
variable (a b : Line)
variable (α β : Plane)
variable (h_lines_distinct : a ≠ b)
variable (h_planes_distinct : α ≠ β)

-- Proposition 1
theorem prop_1 : 
  parallel a b → perpendicular_line_plane a α → perpendicular_line_plane b α :=
sorry

-- Proposition 3
theorem prop_3 : 
  perpendicular_line_plane a α → perpendicular_line_plane a β → parallel_planes α β :=
sorry

end prop_1_prop_3_l630_63015


namespace james_waiting_period_l630_63042

/-- Represents the timeline of James' injury and recovery process -/
structure InjuryTimeline where
  pain_duration : ℕ
  healing_multiplier : ℕ
  additional_wait : ℕ
  total_time : ℕ

/-- Calculates the number of days James waited to start working out after full healing -/
def waiting_period (timeline : InjuryTimeline) : ℕ :=
  timeline.total_time - (timeline.pain_duration * timeline.healing_multiplier) - (timeline.additional_wait * 7)

/-- Theorem stating that James waited 3 days to start working out after full healing -/
theorem james_waiting_period :
  let timeline : InjuryTimeline := {
    pain_duration := 3,
    healing_multiplier := 5,
    additional_wait := 3,
    total_time := 39
  }
  waiting_period timeline = 3 := by sorry

end james_waiting_period_l630_63042


namespace math_contest_theorem_l630_63083

theorem math_contest_theorem (n m k : ℕ) (h_n : n = 200) (h_m : m = 6) (h_k : k = 120)
  (solved : Fin n → Fin m → Prop)
  (h_solved : ∀ j : Fin m, ∃ S : Finset (Fin n), S.card ≥ k ∧ ∀ i ∈ S, solved i j) :
  ∃ i₁ i₂ : Fin n, i₁ ≠ i₂ ∧ ∀ j : Fin m, solved i₁ j ∨ solved i₂ j := by
  sorry

end math_contest_theorem_l630_63083


namespace john_marathon_remainder_l630_63045

/-- The length of a marathon in miles -/
def marathon_miles : ℕ := 26

/-- The additional length of a marathon in yards -/
def marathon_extra_yards : ℕ := 385

/-- The number of yards in a mile -/
def yards_per_mile : ℕ := 1760

/-- The number of marathons John has run -/
def john_marathons : ℕ := 15

/-- Theorem stating that the remainder of yards after converting the total distance of John's marathons to miles is 495 -/
theorem john_marathon_remainder :
  (john_marathons * (marathon_miles * yards_per_mile + marathon_extra_yards)) % yards_per_mile = 495 := by
  sorry

end john_marathon_remainder_l630_63045


namespace high_school_baseball_games_l630_63097

/-- The number of baseball games Benny's high school played is equal to the sum of games he attended and missed -/
theorem high_school_baseball_games 
  (games_attended : ℕ) 
  (games_missed : ℕ) 
  (h1 : games_attended = 14) 
  (h2 : games_missed = 25) : 
  games_attended + games_missed = 39 := by
  sorry

end high_school_baseball_games_l630_63097


namespace vector_operations_l630_63012

theorem vector_operations (a b : ℝ × ℝ) :
  a = (1, 2) → b = (3, 1) →
  (a + b = (4, 3)) ∧ (a.1 * b.1 + a.2 * b.2 = 5) := by
  sorry

end vector_operations_l630_63012


namespace inequality_solution_set_l630_63092

theorem inequality_solution_set (a : ℝ) (h : (4 : ℝ)^a = 2^(a + 2)) :
  {x : ℝ | a^(2*x + 1) > a^(x - 1)} = {x : ℝ | x > -2} :=
sorry

end inequality_solution_set_l630_63092


namespace base_conversion_subtraction_l630_63026

-- Define a function to convert a number from base b to base 10
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

-- Define the number in base 7
def num_base_7 : List Nat := [1, 4, 3, 2, 5]

-- Define the number in base 8
def num_base_8 : List Nat := [1, 2, 3, 4]

-- Theorem statement
theorem base_conversion_subtraction :
  to_base_10 num_base_7 7 - to_base_10 num_base_8 8 = 10610 :=
by sorry

end base_conversion_subtraction_l630_63026


namespace hyperbola_focal_length_and_eccentricity_l630_63036

/-- Given a hyperbola with equation x² - y²/3 = 1, prove its focal length is 4 and eccentricity is 2 -/
theorem hyperbola_focal_length_and_eccentricity :
  let h : ℝ → ℝ → Prop := λ x y ↦ x^2 - y^2/3 = 1
  ∃ (a b c : ℝ),
    (a = 1 ∧ b^2 = 3) ∧
    (c^2 = a^2 + b^2) ∧
    (2 * c = 4) ∧
    (c / a = 2) :=
by sorry

end hyperbola_focal_length_and_eccentricity_l630_63036


namespace number_division_problem_l630_63002

theorem number_division_problem : ∃ N : ℕ, 
  (N / (555 + 445) = 2 * (555 - 445)) ∧ 
  (N % (555 + 445) = 30) ∧ 
  (N = 220030) := by
sorry

end number_division_problem_l630_63002


namespace probability_at_least_one_of_three_l630_63061

theorem probability_at_least_one_of_three (p : ℝ) (h : p = 1 / 3) :
  1 - (1 - p)^3 = 19 / 27 := by
  sorry

end probability_at_least_one_of_three_l630_63061


namespace books_sold_l630_63001

theorem books_sold (total_books : ℕ) (fraction_left : ℚ) (books_sold : ℕ) : 
  total_books = 15750 →
  fraction_left = 7 / 23 →
  books_sold = total_books - (total_books * fraction_left).floor →
  books_sold = 10957 := by
sorry

end books_sold_l630_63001


namespace tv_price_increase_l630_63017

theorem tv_price_increase (P : ℝ) (x : ℝ) : 
  (1.30 * P) * (1 + x / 100) = 1.82 * P ↔ x = 40 :=
sorry

end tv_price_increase_l630_63017


namespace units_digit_of_6541_pow_826_l630_63065

theorem units_digit_of_6541_pow_826 : (6541^826) % 10 = 1 := by
  sorry

end units_digit_of_6541_pow_826_l630_63065


namespace total_episodes_watched_l630_63044

def episode_length : ℕ := 44
def monday_minutes : ℕ := 138
def thursday_minutes : ℕ := 21
def friday_episodes : ℕ := 2
def weekend_minutes : ℕ := 105

theorem total_episodes_watched :
  (monday_minutes + thursday_minutes + friday_episodes * episode_length + weekend_minutes) / episode_length = 8 := by
  sorry

end total_episodes_watched_l630_63044


namespace solution_set_part_i_solution_set_part_ii_l630_63043

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 3| - 2 * |x + a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f x 3 > 2} = {x : ℝ | -7 < x ∧ x < -5/3} := by sorry

-- Part II
theorem solution_set_part_ii (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) (-1), f x a + x + 1 ≤ 0) →
  a ≥ 4 ∨ a ≤ -1 := by sorry

end solution_set_part_i_solution_set_part_ii_l630_63043


namespace cubic_equation_roots_l630_63063

theorem cubic_equation_roots :
  let f : ℝ → ℝ := λ x ↦ x^3 - 2*x
  f 0 = 0 ∧ f (Real.sqrt 2) = 0 ∧ f (-Real.sqrt 2) = 0 := by
  sorry

end cubic_equation_roots_l630_63063


namespace train_passing_pole_time_l630_63003

/-- Proves that a train with given speed and crossing time will take 10 seconds to pass a pole -/
theorem train_passing_pole_time 
  (train_speed_kmh : ℝ) 
  (stationary_train_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_speed_kmh = 72) 
  (h2 : stationary_train_length = 500) 
  (h3 : crossing_time = 35) :
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let train_length := train_speed_ms * crossing_time - stationary_train_length
  train_length / train_speed_ms = 10 := by
  sorry

end train_passing_pole_time_l630_63003


namespace composite_numbers_l630_63054

theorem composite_numbers (n : ℕ) (h : n > 0) : 
  (∃ k : ℕ, k > 1 ∧ k < 2 * 2^(2^(2*n)) + 1 ∧ (2 * 2^(2^(2*n)) + 1) % k = 0) ∧ 
  (∃ m : ℕ, m > 1 ∧ m < 3 * 2^(2*n) + 1 ∧ (3 * 2^(2*n) + 1) % m = 0) := by
sorry

end composite_numbers_l630_63054


namespace least_prime_factor_of_5_pow_5_minus_5_pow_4_l630_63029

theorem least_prime_factor_of_5_pow_5_minus_5_pow_4 :
  Nat.minFac (5^5 - 5^4) = 2 := by
sorry

end least_prime_factor_of_5_pow_5_minus_5_pow_4_l630_63029


namespace complex_fraction_simplification_l630_63058

theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (2 - i) / (1 + 4*i) = -2/17 - (9/17)*i := by sorry

end complex_fraction_simplification_l630_63058


namespace circle_center_sum_l630_63078

/-- The sum of the x and y coordinates of the center of a circle with equation x^2 + y^2 = 4x - 6y + 9 is -1 -/
theorem circle_center_sum (x y : ℝ) : x^2 + y^2 = 4*x - 6*y + 9 → x + y = -1 := by
  sorry

end circle_center_sum_l630_63078


namespace books_read_difference_l630_63086

def total_books : ℕ := 20
def peter_percentage : ℚ := 40 / 100
def brother_percentage : ℚ := 10 / 100

theorem books_read_difference : 
  (peter_percentage * total_books : ℚ).floor - (brother_percentage * total_books : ℚ).floor = 6 := by
  sorry

end books_read_difference_l630_63086


namespace pirate_treasure_probability_l630_63037

/-- The probability of finding treasure and no traps on a single island -/
def p_treasure : ℚ := 1/5

/-- The probability of finding traps and no treasure on a single island -/
def p_traps : ℚ := 1/10

/-- The probability of finding neither treasure nor traps on a single island -/
def p_neither : ℚ := 7/10

/-- The total number of islands -/
def total_islands : ℕ := 8

/-- The number of islands with treasure we want to find -/
def treasure_islands : ℕ := 4

theorem pirate_treasure_probability :
  (Nat.choose total_islands treasure_islands : ℚ) *
  p_treasure ^ treasure_islands *
  p_neither ^ (total_islands - treasure_islands) =
  33614 / 1250000 := by
  sorry


end pirate_treasure_probability_l630_63037


namespace distribution_ways_eq_1080_l630_63020

/-- The number of ways to distribute 6 distinct items among 4 groups,
    where two groups receive 2 items each and two groups receive 1 item each -/
def distribution_ways : ℕ :=
  (Nat.choose 6 2 * Nat.choose 4 2) / 2 * 24

/-- Theorem stating that the number of distribution ways is 1080 -/
theorem distribution_ways_eq_1080 : distribution_ways = 1080 := by
  sorry

end distribution_ways_eq_1080_l630_63020


namespace unique_m_solution_l630_63028

theorem unique_m_solution : 
  ∀ m : ℕ+, 
  (∃ a b c : ℕ+, (a.val * b.val * c.val * m.val : ℕ) = 1 + a.val^2 + b.val^2 + c.val^2) ↔ 
  m = 4 := by
sorry

end unique_m_solution_l630_63028


namespace line_slope_proportionality_l630_63024

/-- Given a line where an increase of 3 units in x corresponds to an increase of 7 units in y,
    prove that an increase of 9 units in x results in an increase of 21 units in y. -/
theorem line_slope_proportionality (f : ℝ → ℝ) (x : ℝ) :
  (f (x + 3) - f x = 7) → (f (x + 9) - f x = 21) :=
by sorry

end line_slope_proportionality_l630_63024


namespace initial_stock_proof_l630_63033

/-- The number of coloring books sold during the sale -/
def books_sold : ℕ := 6

/-- The number of shelves used for remaining books -/
def shelves_used : ℕ := 3

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 7

/-- The initial number of coloring books in stock -/
def initial_stock : ℕ := shelves_used * books_per_shelf + books_sold

theorem initial_stock_proof : initial_stock = 27 := by
  sorry

end initial_stock_proof_l630_63033


namespace vertical_shift_graph_l630_63059

-- Define a type for functions from real numbers to real numbers
def RealFunction := ℝ → ℝ

-- Define a vertical shift operation on functions
def verticalShift (f : RealFunction) (k : ℝ) : RealFunction :=
  λ x => f x + k

-- State the theorem
theorem vertical_shift_graph (f : RealFunction) (k : ℝ) :
  ∀ x y, y = f x + k ↔ y - k = f x :=
sorry

end vertical_shift_graph_l630_63059


namespace time_to_walk_five_miles_l630_63046

/-- Given that Tom walks 2 miles in 6 minutes, prove that it takes 15 minutes to walk 5 miles at the same rate. -/
theorem time_to_walk_five_miles (distance_to_jerry : ℝ) (time_to_jerry : ℝ) (distance_to_sam : ℝ) :
  distance_to_jerry = 2 →
  time_to_jerry = 6 →
  distance_to_sam = 5 →
  (distance_to_sam / (distance_to_jerry / time_to_jerry)) = 15 := by
sorry

end time_to_walk_five_miles_l630_63046


namespace circle_area_from_diameter_endpoints_l630_63057

/-- The area of a circle with diameter endpoints at (1, 3) and (8, 6) is 58π/4 square units. -/
theorem circle_area_from_diameter_endpoints :
  let A : ℝ × ℝ := (1, 3)
  let B : ℝ × ℝ := (8, 6)
  let diameter_squared := (B.1 - A.1)^2 + (B.2 - A.2)^2
  let radius_squared := diameter_squared / 4
  let circle_area := π * radius_squared
  circle_area = 58 * π / 4 := by
  sorry

end circle_area_from_diameter_endpoints_l630_63057


namespace min_purses_needed_l630_63095

/-- Represents a distribution of coins into purses -/
def CoinDistribution := List Nat

/-- Checks if a distribution is valid for a given number of sailors -/
def isValidDistribution (d : CoinDistribution) (n : Nat) : Prop :=
  (d.sum = 60) ∧ (∃ (x : Nat), d.sum = n * x)

/-- Checks if a distribution is valid for all required sailor counts -/
def isValidForAllSailors (d : CoinDistribution) : Prop :=
  isValidDistribution d 2 ∧
  isValidDistribution d 3 ∧
  isValidDistribution d 4 ∧
  isValidDistribution d 5

/-- The main theorem stating the minimum number of purses needed -/
theorem min_purses_needed :
  ∃ (d : CoinDistribution),
    d.length = 9 ∧
    isValidForAllSailors d ∧
    ∀ (d' : CoinDistribution),
      isValidForAllSailors d' →
      d'.length ≥ 9 := by
  sorry

end min_purses_needed_l630_63095


namespace prob_at_least_one_boy_and_girl_l630_63088

/-- The probability of having a boy or a girl -/
def p_boy_or_girl : ℚ := 1 / 2

/-- The number of children in the family -/
def num_children : ℕ := 4

/-- The probability of having at least one boy and one girl in a family of four children -/
theorem prob_at_least_one_boy_and_girl : 
  (1 : ℚ) - (p_boy_or_girl ^ num_children + p_boy_or_girl ^ num_children) = 7 / 8 := by
  sorry

end prob_at_least_one_boy_and_girl_l630_63088


namespace min_sum_of_square_roots_l630_63039

theorem min_sum_of_square_roots (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ x : ℝ, Real.sqrt ((x - a)^2 + b^2) + Real.sqrt ((x - b)^2 + a^2) ≥ Real.sqrt (2 * (a^2 + b^2))) ∧
  (∃ x : ℝ, Real.sqrt ((x - a)^2 + b^2) + Real.sqrt ((x - b)^2 + a^2) = Real.sqrt (2 * (a^2 + b^2))) :=
by
  sorry

#check min_sum_of_square_roots

end min_sum_of_square_roots_l630_63039


namespace recommendation_plans_count_l630_63064

/-- Represents the number of recommendation spots for each language -/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the gender distribution of selected candidates -/
structure SelectedCandidates :=
  (males : Nat)
  (females : Nat)

/-- Calculates the number of different recommendation plans -/
def countRecommendationPlans (spots : RecommendationSpots) (candidates : SelectedCandidates) : Nat :=
  sorry

theorem recommendation_plans_count :
  let spots : RecommendationSpots := ⟨2, 2, 1⟩
  let candidates : SelectedCandidates := ⟨3, 2⟩
  countRecommendationPlans spots candidates = 24 := by sorry

end recommendation_plans_count_l630_63064


namespace weed_spread_incomplete_weeds_cannot_fill_grid_l630_63050

/-- Represents a grid with weeds -/
structure WeedGrid :=
  (size : Nat)
  (initial_weeds : Nat)

/-- Calculates the maximum possible boundary length of a grid -/
def max_boundary (g : WeedGrid) : Nat :=
  4 * g.size

/-- Calculates the maximum initial boundary length of weed-filled cells -/
def initial_boundary (g : WeedGrid) : Nat :=
  4 * g.initial_weeds

/-- The weed spread theorem -/
theorem weed_spread_incomplete (g : WeedGrid) 
  (h_size : g.size = 10) 
  (h_initial : g.initial_weeds = 9) :
  initial_boundary g < max_boundary g := by
  sorry

/-- The main theorem: weeds cannot spread to all cells -/
theorem weeds_cannot_fill_grid (g : WeedGrid) 
  (h_size : g.size = 10) 
  (h_initial : g.initial_weeds = 9) :
  ¬ (∃ (final_weeds : Nat), final_weeds = g.size * g.size) := by
  sorry

end weed_spread_incomplete_weeds_cannot_fill_grid_l630_63050


namespace y_values_l630_63011

theorem y_values (x : ℝ) (h : x^2 + 6 * (x / (x - 3))^2 = 72) :
  let y := ((x - 3)^3 * (x + 4)) / (3 * x - 4)
  y = 135 / 7 ∨ y = 216 / 13 := by sorry

end y_values_l630_63011


namespace inequality_solution_set_l630_63010

theorem inequality_solution_set (a : ℝ) (h : 0 < a ∧ a < 1) :
  {x : ℝ | (a - x) * (x - 1/a) > 0} = Set.Ioo a (1/a) := by sorry

end inequality_solution_set_l630_63010


namespace vector_equation_solution_l630_63077

def a : ℝ × ℝ × ℝ := (1, 3, -2)
def b : ℝ × ℝ × ℝ := (2, 1, 0)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

def dot_product (v w : ℝ × ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2.1 * w.2.1 + v.2.2 * w.2.2

theorem vector_equation_solution :
  ∃ (p q r : ℝ),
    (5, 2, -3) = (p * a.1 + q * b.1 + r * (cross_product a b).1,
                  p * a.2.1 + q * b.2.1 + r * (cross_product a b).2.1,
                  p * a.2.2 + q * b.2.2 + r * (cross_product a b).2.2) →
    r = 17 / 45 := by
  sorry

end vector_equation_solution_l630_63077


namespace sum_of_fourth_powers_of_roots_l630_63094

theorem sum_of_fourth_powers_of_roots (p q r s : ℂ) : 
  (p^4 - p^3 + p^2 - 3*p + 3 = 0) →
  (q^4 - q^3 + q^2 - 3*q + 3 = 0) →
  (r^4 - r^3 + r^2 - 3*r + 3 = 0) →
  (s^4 - s^3 + s^2 - 3*s + 3 = 0) →
  p^4 + q^4 + r^4 + s^4 = 5 := by
sorry

end sum_of_fourth_powers_of_roots_l630_63094


namespace equation_solution_l630_63030

theorem equation_solution (a : ℝ) : (3 * 5 + 2 * a = 3) → a = -6 := by
  sorry

end equation_solution_l630_63030


namespace evaluate_dagger_l630_63008

-- Define the † operation
def dagger (m n p q : ℚ) : ℚ := m * p * (q / n)

-- Theorem statement
theorem evaluate_dagger : dagger (5/16) (12/5) = 75/4 := by
  sorry

end evaluate_dagger_l630_63008


namespace p_squared_plus_20_not_prime_l630_63098

theorem p_squared_plus_20_not_prime (p : ℕ) (h : Prime p) : ¬ Prime (p^2 + 20) := by
  sorry

end p_squared_plus_20_not_prime_l630_63098


namespace worst_player_is_father_l630_63038

-- Define the family members
inductive FamilyMember
  | Father
  | Sister
  | Daughter
  | Son

-- Define the sex of a family member
def sex : FamilyMember → Bool
  | FamilyMember.Father => true   -- true represents male
  | FamilyMember.Sister => false  -- false represents female
  | FamilyMember.Daughter => false
  | FamilyMember.Son => true

-- Define the twin relationship
def isTwin : FamilyMember → FamilyMember → Bool
  | FamilyMember.Father, FamilyMember.Sister => true
  | FamilyMember.Sister, FamilyMember.Father => true
  | FamilyMember.Daughter, FamilyMember.Son => true
  | FamilyMember.Son, FamilyMember.Daughter => true
  | _, _ => false

-- Define the theorem
theorem worst_player_is_father :
  ∀ (worst best : FamilyMember),
    (∃ twin : FamilyMember, isTwin worst twin ∧ sex twin ≠ sex best) →
    isTwin worst best →
    worst = FamilyMember.Father :=
by sorry

end worst_player_is_father_l630_63038


namespace polyhedron_property_l630_63004

/-- A convex polyhedron with the given properties -/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  t : ℕ  -- number of triangular faces
  h : ℕ  -- number of hexagonal faces
  T : ℕ  -- number of triangular faces meeting at each vertex
  H : ℕ  -- number of hexagonal faces meeting at each vertex
  euler_formula : V - E + F = 2
  face_count : F = 40
  face_composition : F = t + h
  vertex_property : 2 * T + H = 7
  edge_count : E = (3 * t + 6 * h) / 2

theorem polyhedron_property (P : ConvexPolyhedron) : 100 * P.H + 10 * P.T + P.V = 367 := by
  sorry

end polyhedron_property_l630_63004


namespace water_bottles_remaining_l630_63049

/-- Calculates the number of bottles remaining after two days of consumption --/
def bottlesRemaining (initialBottles : ℕ) : ℕ :=
  let firstDayRemaining := initialBottles - 
    (initialBottles / 4 + initialBottles / 6 + initialBottles / 8)
  let fatherSecondDay := firstDayRemaining / 5
  let motherSecondDay := (firstDayRemaining - fatherSecondDay) / 7
  let sonSecondDay := (firstDayRemaining - fatherSecondDay - motherSecondDay) / 9
  let daughterSecondDay := (firstDayRemaining - fatherSecondDay - motherSecondDay - sonSecondDay) / 9
  firstDayRemaining - (fatherSecondDay + motherSecondDay + sonSecondDay + daughterSecondDay)

theorem water_bottles_remaining (initialBottles : ℕ) :
  initialBottles = 48 → bottlesRemaining initialBottles = 14 := by
  sorry

end water_bottles_remaining_l630_63049


namespace scientific_notation_of_56_99_million_l630_63069

theorem scientific_notation_of_56_99_million :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    56990000 = a * (10 : ℝ) ^ n ∧
    a = 5.699 ∧ n = 7 := by
  sorry

end scientific_notation_of_56_99_million_l630_63069


namespace object_length_increase_l630_63067

/-- The number of days required for an object to reach 50 times its original length -/
def n : ℕ := 147

/-- The factor by which the object's length increases on day k -/
def increase_factor (k : ℕ) : ℚ := (k + 3 : ℚ) / (k + 2 : ℚ)

/-- The total increase factor after n days -/
def total_increase_factor (n : ℕ) : ℚ := (n + 3 : ℚ) / 3

theorem object_length_increase :
  total_increase_factor n = 50 := by sorry

end object_length_increase_l630_63067


namespace inequality_proof_l630_63007

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1/2) :
  1/(1-a) + 1/(1-b) ≥ 4 ∧ (1/(1-a) + 1/(1-b) = 4 ↔ a = 1/2 ∧ b = 1/2) := by
  sorry

#check inequality_proof

end inequality_proof_l630_63007


namespace not_p_and_q_implies_at_most_one_l630_63090

theorem not_p_and_q_implies_at_most_one (p q : Prop) : 
  ¬(p ∧ q) → (¬p ∨ ¬q) := by
  sorry

end not_p_and_q_implies_at_most_one_l630_63090


namespace well_depth_l630_63085

/-- Proves that a circular well with diameter 2 meters and volume 31.41592653589793 cubic meters has a depth of 10 meters -/
theorem well_depth (diameter : ℝ) (volume : ℝ) (depth : ℝ) : 
  diameter = 2 → 
  volume = 31.41592653589793 → 
  volume = Real.pi * (diameter / 2)^2 * depth → 
  depth = 10 := by sorry

end well_depth_l630_63085


namespace area_of_sixth_rectangle_l630_63087

/-- Given a rectangle divided into six smaller rectangles, prove that if five of these rectangles
    have areas 126, 63, 40, 20, and 161, then the area of the sixth rectangle is 101. -/
theorem area_of_sixth_rectangle (
  total_area : ℝ)
  (area1 area2 area3 area4 area5 : ℝ)
  (h1 : area1 = 126)
  (h2 : area2 = 63)
  (h3 : area3 = 40)
  (h4 : area4 = 20)
  (h5 : area5 = 161)
  (h_sum : total_area = area1 + area2 + area3 + area4 + area5 + (total_area - (area1 + area2 + area3 + area4 + area5))) :
  total_area - (area1 + area2 + area3 + area4 + area5) = 101 := by
  sorry


end area_of_sixth_rectangle_l630_63087


namespace bisection_method_structures_l630_63052

/-- Bisection method for finding the approximate root of x^2 - 5 = 0 -/
def bisection_method (f : ℝ → ℝ) (a b : ℝ) (ε : ℝ) : ℝ := sorry

/-- The equation to solve -/
def equation (x : ℝ) : ℝ := x^2 - 5

theorem bisection_method_structures :
  ∃ (sequential conditional loop : Bool),
    sequential ∧ conditional ∧ loop ∧
    (∀ (a b ε : ℝ), ε > 0 → 
      ∃ (result : ℝ), 
        bisection_method equation a b ε = result ∧ 
        |equation result| < ε) :=
sorry

end bisection_method_structures_l630_63052


namespace min_perimeter_isosceles_triangles_l630_63091

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  equal_side : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.equal_side + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * Real.sqrt (4 * (t.equal_side : ℝ)^2 - (t.base : ℝ)^2) / 4

/-- Theorem statement -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t1 t2 : IsoscelesTriangle),
    t1 ≠ t2 ∧
    perimeter t1 = perimeter t2 ∧
    area t1 = area t2 ∧
    5 * t1.base = 6 * t2.base ∧
    perimeter t1 = 399 ∧
    (∀ (s1 s2 : IsoscelesTriangle),
      s1 ≠ s2 →
      perimeter s1 = perimeter s2 →
      area s1 = area s2 →
      5 * s1.base = 6 * s2.base →
      perimeter s1 ≥ 399) :=
by sorry

end min_perimeter_isosceles_triangles_l630_63091


namespace car_profit_percent_l630_63055

/-- Calculate the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent (car_cost repair_cost taxes insurance selling_price : ℝ) :
  car_cost = 36400 →
  repair_cost = 8000 →
  taxes = 4500 →
  insurance = 2500 →
  selling_price = 68400 →
  let total_cost := car_cost + repair_cost + taxes + insurance
  let profit := selling_price - total_cost
  let profit_percent := (profit / total_cost) * 100
  abs (profit_percent - 33.07) < 0.01 := by
sorry

end car_profit_percent_l630_63055


namespace six_couples_handshakes_l630_63019

/-- The number of handshakes in a gathering of couples where each person
    shakes hands with everyone except their spouse -/
def handshakes (n : ℕ) : ℕ :=
  let total_people := 2 * n
  let total_potential_handshakes := total_people * (total_people - 1) / 2
  total_potential_handshakes - n

theorem six_couples_handshakes :
  handshakes 6 = 60 := by sorry

end six_couples_handshakes_l630_63019


namespace jaime_sum_with_square_l630_63047

theorem jaime_sum_with_square (n : ℕ) (k : ℕ) : 
  (∃ (i : ℕ), i < 100 ∧ n + i = k) →
  (50 * (2 * n + 99) - k + k^2 = 7500) →
  k = 26 := by
sorry

end jaime_sum_with_square_l630_63047


namespace phone_number_proof_l630_63071

def is_harmonic_mean (a b c : ℕ) : Prop :=
  2 * b * a * c = b * (a + c)

def is_six_digit (a b c d : ℕ) : Prop :=
  100000 ≤ a * 100000 + b * 10000 + c * 100 + d ∧
  a * 100000 + b * 10000 + c * 100 + d < 1000000

theorem phone_number_proof (a b c d : ℕ) : 
  a = 6 ∧ b = 8 ∧ c = 12 ∧ d = 24 →
  a < b ∧ b < c ∧ c < d ∧
  is_harmonic_mean a b c ∧
  is_harmonic_mean b c d ∧
  is_six_digit a b c d := by
  sorry

#eval [6, 8, 12, 24].map (λ x => x.toDigits 10)

end phone_number_proof_l630_63071


namespace max_value_inequality_l630_63053

theorem max_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + 2*y)^2 / (x^2 + y^2) ≤ 9/2 := by sorry

end max_value_inequality_l630_63053


namespace sum_inequality_l630_63089

theorem sum_inequality (a b c d : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  a * d + b * c < a * c + b * d ∧ a * c + b * d < a * b + c * d := by
  sorry

end sum_inequality_l630_63089


namespace basketball_game_free_throws_l630_63079

theorem basketball_game_free_throws :
  ∀ (three_pointers two_pointers free_throws : ℕ),
    three_pointers + two_pointers + free_throws = 32 →
    two_pointers = 4 * three_pointers + 3 →
    3 * three_pointers + 2 * two_pointers + free_throws = 65 →
    free_throws = 4 :=
by sorry

end basketball_game_free_throws_l630_63079


namespace circle_diameter_property_l630_63062

theorem circle_diameter_property (BC BD DA : ℝ) (h1 : BC = Real.sqrt 901) (h2 : BD = 1) (h3 : DA = 16) : ∃ EC : ℝ, EC = 1 ∧ BC * EC = BD * (BC - BD) := by
  sorry

end circle_diameter_property_l630_63062


namespace arithmetic_calculation_l630_63072

theorem arithmetic_calculation : 3 * 5 * 7 + 15 / 3 = 110 := by
  sorry

end arithmetic_calculation_l630_63072


namespace greatest_integer_x_l630_63068

theorem greatest_integer_x (x : ℕ) : x^4 / x^2 < 18 → x ≤ 4 :=
sorry

end greatest_integer_x_l630_63068


namespace proposition_p_sufficient_not_necessary_for_q_l630_63073

theorem proposition_p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, 0 < x ∧ x < 1 → x^2 < 2*x) ∧
  (∃ x : ℝ, 0 < x ∧ x < 2 ∧ x^2 < 2*x ∧ x ≥ 1) :=
by sorry

end proposition_p_sufficient_not_necessary_for_q_l630_63073


namespace jeffs_weekly_running_time_l630_63093

/-- Represents Jeff's weekly running schedule -/
structure RunningSchedule where
  normalDays : Nat  -- Number of days with normal running time
  normalTime : Nat  -- Normal running time in minutes
  thursdayReduction : Nat  -- Minutes reduced on Thursday
  fridayIncrease : Nat  -- Minutes increased on Friday

/-- Calculates the total running time for the week given a RunningSchedule -/
def totalRunningTime (schedule : RunningSchedule) : Nat :=
  schedule.normalDays * schedule.normalTime +
  (schedule.normalTime - schedule.thursdayReduction) +
  (schedule.normalTime + schedule.fridayIncrease)

/-- Theorem stating that Jeff's total running time for the week is 290 minutes -/
theorem jeffs_weekly_running_time :
  ∀ (schedule : RunningSchedule),
    schedule.normalDays = 3 ∧
    schedule.normalTime = 60 ∧
    schedule.thursdayReduction = 20 ∧
    schedule.fridayIncrease = 10 →
    totalRunningTime schedule = 290 := by
  sorry

end jeffs_weekly_running_time_l630_63093


namespace prob_ice_given_ski_l630_63022

/-- The probability that a high school student likes ice skating -/
def P_ice_skating : ℝ := 0.6

/-- The probability that a high school student likes skiing -/
def P_skiing : ℝ := 0.5

/-- The probability that a high school student likes either ice skating or skiing -/
def P_ice_or_ski : ℝ := 0.7

/-- The probability that a high school student likes both ice skating and skiing -/
def P_ice_and_ski : ℝ := P_ice_skating + P_skiing - P_ice_or_ski

theorem prob_ice_given_ski :
  P_ice_and_ski / P_skiing = 0.8 := by sorry

end prob_ice_given_ski_l630_63022


namespace child_tickets_sold_l630_63031

/-- Proves the number of child tickets sold given the ticket prices and total sales information -/
theorem child_tickets_sold 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (total_sales : ℕ) 
  (total_tickets : ℕ) 
  (h1 : adult_price = 5)
  (h2 : child_price = 3)
  (h3 : total_sales = 178)
  (h4 : total_tickets = 42) :
  ∃ (adult_tickets child_tickets : ℕ),
    adult_tickets + child_tickets = total_tickets ∧
    adult_tickets * adult_price + child_tickets * child_price = total_sales ∧
    child_tickets = 16 := by
  sorry


end child_tickets_sold_l630_63031


namespace solution_existence_condition_l630_63000

theorem solution_existence_condition (m : ℝ) : 
  (∃ x ∈ Set.Icc 0 2, x^3 - 3*x + m = 0) → m ≤ 2 ∧ 
  ¬(∀ m ≤ 2, ∃ x ∈ Set.Icc 0 2, x^3 - 3*x + m = 0) :=
by sorry

end solution_existence_condition_l630_63000


namespace last_four_matches_average_l630_63016

/-- Represents a cricket scoring scenario -/
structure CricketScoring where
  totalMatches : Nat
  firstMatchesCount : Nat
  totalAverage : ℚ
  firstMatchesAverage : ℚ

/-- Calculates the average score of the remaining matches -/
def remainingMatchesAverage (cs : CricketScoring) : ℚ :=
  let totalRuns := cs.totalAverage * cs.totalMatches
  let firstMatchesRuns := cs.firstMatchesAverage * cs.firstMatchesCount
  let remainingMatchesCount := cs.totalMatches - cs.firstMatchesCount
  (totalRuns - firstMatchesRuns) / remainingMatchesCount

/-- Theorem stating that under the given conditions, the average of the last 4 matches is 34.25 -/
theorem last_four_matches_average (cs : CricketScoring) 
  (h1 : cs.totalMatches = 10)
  (h2 : cs.firstMatchesCount = 6)
  (h3 : cs.totalAverage = 389/10)
  (h4 : cs.firstMatchesAverage = 42) :
  remainingMatchesAverage cs = 137/4 := by
  sorry

end last_four_matches_average_l630_63016


namespace discounted_cost_l630_63066

/-- The cost of a pencil without discount -/
def pencil_cost : ℚ := sorry

/-- The cost of a notebook -/
def notebook_cost : ℚ := sorry

/-- The discount per pencil when buying more than 10 pencils -/
def discount : ℚ := 0.05

/-- Condition: Cost of 8 pencils and 10 notebooks without discount -/
axiom condition1 : 8 * pencil_cost + 10 * notebook_cost = 5.36

/-- Condition: Cost of 12 pencils and 5 notebooks with discount -/
axiom condition2 : 12 * (pencil_cost - discount) + 5 * notebook_cost = 4.05

/-- The cost of 15 pencils and 12 notebooks with discount -/
def total_cost : ℚ := 15 * (pencil_cost - discount) + 12 * notebook_cost

theorem discounted_cost : total_cost = 7.01 := by sorry

end discounted_cost_l630_63066


namespace sqrt_fraction_simplification_l630_63027

theorem sqrt_fraction_simplification : 
  (Real.sqrt 3) / ((Real.sqrt 3) + (Real.sqrt 12)) = 1 / 3 := by
  sorry

end sqrt_fraction_simplification_l630_63027


namespace binomial_coefficient_17_16_l630_63099

theorem binomial_coefficient_17_16 : Nat.choose 17 16 = 17 := by sorry

end binomial_coefficient_17_16_l630_63099


namespace complex_number_properties_l630_63009

theorem complex_number_properties (z : ℂ) (h : Complex.I * (z + 1) = -2 + 2 * Complex.I) :
  (Complex.im z = 2) ∧ (let ω := z / (1 - 2 * Complex.I); Complex.abs ω ^ 2015 = 1) := by
  sorry

end complex_number_properties_l630_63009
