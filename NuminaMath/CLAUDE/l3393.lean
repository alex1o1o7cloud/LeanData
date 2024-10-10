import Mathlib

namespace group_bill_calculation_l3393_339320

/-- Calculates the total cost for a group at a restaurant where kids eat free. -/
def restaurant_bill (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

/-- Proves that the total cost for a group of 11 people, including 2 kids,
    at a restaurant where adult meals cost $8 and kids eat free, is $72. -/
theorem group_bill_calculation :
  restaurant_bill 11 2 8 = 72 := by
  sorry

end group_bill_calculation_l3393_339320


namespace car_driving_east_when_sun_setting_in_mirror_l3393_339316

-- Define the direction type
inductive Direction
| East
| West
| North
| South

-- Define the position of the sun
inductive SunPosition
| Setting
| Rising
| Overhead

-- Define the view of the sun
structure SunView where
  position : SunPosition
  throughMirror : Bool

-- Define the state of the car
structure CarState where
  direction : Direction
  sunView : SunView

-- Theorem statement
theorem car_driving_east_when_sun_setting_in_mirror 
  (car : CarState) : 
  car.sunView.position = SunPosition.Setting ∧ 
  car.sunView.throughMirror = true → 
  car.direction = Direction.East :=
sorry

end car_driving_east_when_sun_setting_in_mirror_l3393_339316


namespace total_interest_calculation_l3393_339345

theorem total_interest_calculation (stock1_rate stock2_rate stock3_rate : ℝ) 
  (face_value : ℝ) (h1 : stock1_rate = 0.16) (h2 : stock2_rate = 0.12) 
  (h3 : stock3_rate = 0.20) (h4 : face_value = 100) : 
  stock1_rate * face_value + stock2_rate * face_value + stock3_rate * face_value = 48 := by
  sorry

end total_interest_calculation_l3393_339345


namespace interest_group_members_l3393_339305

/-- Represents a math interest group -/
structure InterestGroup where
  members : ℕ
  average_age : ℝ

/-- The change in average age when members leave or join -/
def age_change (g : InterestGroup) : Prop :=
  (g.members * g.average_age - 5 * 9 = (g.average_age + 1) * (g.members - 5)) ∧
  (g.members * g.average_age + 17 * 5 = (g.average_age + 1) * (g.members + 5))

theorem interest_group_members :
  ∃ (g : InterestGroup), age_change g → g.members = 20 := by
  sorry

end interest_group_members_l3393_339305


namespace tangent_and_minimum_value_l3393_339347

open Real

-- Define the function f
noncomputable def f (a b x : ℝ) : ℝ := exp x * (a * x^2 + b * x + 1)

-- Define the derivative of f
noncomputable def f' (a b x : ℝ) : ℝ := exp x * (a * x^2 + (2 * a + b) * x + b + 1)

theorem tangent_and_minimum_value (a b : ℝ) :
  (f' a b (-1) = 0) →
  (
    -- Part I
    (b = 1 →
      ∃ (m c : ℝ), m = 2 ∧ c = 1 ∧
      ∀ x y, y = f a b x ∧ x = 0 → y = m * x + c
    ) ∧
    -- Part II
    (
      (∀ x, x ∈ Set.Icc (-1) 1 → f a b x ≥ 0) ∧
      (∃ x, x ∈ Set.Icc (-1) 1 ∧ f a b x = 0) →
      b = 2 ∨ b = -2
    )
  ) := by sorry

end tangent_and_minimum_value_l3393_339347


namespace rectangle_fold_theorem_l3393_339332

theorem rectangle_fold_theorem : ∃ (a b : ℕ+), 
  a ≤ b ∧ 
  (a.val : ℝ) / (b.val : ℝ) * Real.sqrt ((a.val : ℝ)^2 + (b.val : ℝ)^2) = 65 ∧
  2 * (a.val + b.val) = 408 := by
sorry

end rectangle_fold_theorem_l3393_339332


namespace outstanding_student_awards_l3393_339380

/-- The number of ways to distribute n identical awards among k classes,
    with each class receiving at least one award. -/
def distribution_schemes (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- Theorem: There are 36 ways to distribute 10 identical awards among 8 classes,
    with each class receiving at least one award. -/
theorem outstanding_student_awards : distribution_schemes 10 8 = 36 := by
  sorry

end outstanding_student_awards_l3393_339380


namespace integral_sin_cos_l3393_339308

theorem integral_sin_cos : 
  ∫ x in (0)..(2*Real.pi/3), (1 + Real.sin x) / (1 + Real.cos x + Real.sin x) = Real.pi/3 + Real.log 2 := by
  sorry

end integral_sin_cos_l3393_339308


namespace lg_sum_equals_lg_product_l3393_339389

-- Define logarithm base 10
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_sum_equals_lg_product : lg 2 + lg 5 = lg 10 := by sorry

end lg_sum_equals_lg_product_l3393_339389


namespace solve_dales_potatoes_l3393_339352

/-- The number of potatoes Dale bought -/
def dales_potatoes (marcel_corn dale_corn marcel_potatoes total_vegetables : ℕ) : ℕ :=
  total_vegetables - (marcel_corn + dale_corn + marcel_potatoes)

theorem solve_dales_potatoes :
  ∀ (marcel_corn dale_corn marcel_potatoes total_vegetables : ℕ),
    marcel_corn = 10 →
    dale_corn = marcel_corn / 2 →
    marcel_potatoes = 4 →
    total_vegetables = 27 →
    dales_potatoes marcel_corn dale_corn marcel_potatoes total_vegetables = 8 := by
  sorry

end solve_dales_potatoes_l3393_339352


namespace condition_2_not_implies_right_triangle_l3393_339397

/-- A triangle ABC --/
structure Triangle :=
  (A B C : ℝ)
  (sum_angles : A + B + C = 180)

/-- Definition of a right triangle --/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = 90 ∨ t.B = 90 ∨ t.C = 90

/-- The condition ∠A = ∠B - ∠C --/
def condition_2 (t : Triangle) : Prop :=
  t.A = t.B - t.C

/-- Theorem: The condition ∠A = ∠B - ∠C does not necessarily imply a right triangle --/
theorem condition_2_not_implies_right_triangle :
  ∃ t : Triangle, condition_2 t ∧ ¬is_right_triangle t :=
sorry

end condition_2_not_implies_right_triangle_l3393_339397


namespace line_slope_l3393_339321

/-- Given a line described by the equation 3y = 4x - 9 + 2z where z = 3,
    prove that the slope of this line is 4/3 -/
theorem line_slope (x y : ℝ) :
  3 * y = 4 * x - 9 + 2 * 3 →
  (∃ m b : ℝ, y = m * x + b ∧ m = 4 / 3) :=
by sorry

end line_slope_l3393_339321


namespace songs_storable_jeff_l3393_339349

/-- Calculates the number of songs that can be stored on a phone given the total storage, used storage, and size of each song. -/
def songs_storable (total_storage : ℕ) (used_storage : ℕ) (song_size : ℕ) : ℕ :=
  ((total_storage - used_storage) * 1000) / song_size

/-- Theorem stating that given the specific conditions, 400 songs can be stored. -/
theorem songs_storable_jeff : songs_storable 16 4 30 = 400 := by
  sorry

#eval songs_storable 16 4 30

end songs_storable_jeff_l3393_339349


namespace sixth_power_sum_of_roots_l3393_339334

theorem sixth_power_sum_of_roots (r s : ℝ) : 
  r^2 - 3*r*Real.sqrt 2 + 2 = 0 → 
  s^2 - 3*s*Real.sqrt 2 + 2 = 0 → 
  r^6 + s^6 = 2576 := by sorry

end sixth_power_sum_of_roots_l3393_339334


namespace probability_even_sum_and_same_number_l3393_339309

/-- A fair six-sided die -/
def Die : Type := Fin 6

/-- The outcome of rolling two dice -/
def RollOutcome : Type := Die × Die

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : Nat := 36

/-- Predicate for checking if a roll outcome has an even sum -/
def hasEvenSum (roll : RollOutcome) : Prop :=
  (roll.1.val + 1 + roll.2.val + 1) % 2 = 0

/-- Predicate for checking if both dice show the same number -/
def hasSameNumber (roll : RollOutcome) : Prop :=
  roll.1 = roll.2

/-- The set of favorable outcomes (even sum and same number) -/
def favorableOutcomes : Finset RollOutcome :=
  sorry

/-- The number of favorable outcomes -/
def numFavorableOutcomes : Nat :=
  favorableOutcomes.card

theorem probability_even_sum_and_same_number :
  (numFavorableOutcomes : ℚ) / totalOutcomes = 1 / 12 :=
sorry

end probability_even_sum_and_same_number_l3393_339309


namespace twentyByFifteenGridToothpicks_l3393_339364

/-- Represents a grid of toothpicks with alternating crossbars -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ

/-- Calculates the total number of toothpicks used in the grid -/
def totalToothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontalToothpicks := (grid.height + 1) * grid.width
  let verticalToothpicks := (grid.width + 1) * grid.height
  let totalSquares := grid.height * grid.width
  let crossbarToothpicks := (totalSquares / 2) * 2
  horizontalToothpicks + verticalToothpicks + crossbarToothpicks

/-- Theorem stating that a 20x15 grid uses 935 toothpicks -/
theorem twentyByFifteenGridToothpicks :
  totalToothpicks { height := 20, width := 15 } = 935 := by
  sorry


end twentyByFifteenGridToothpicks_l3393_339364


namespace sin_810_plus_cos_neg_60_l3393_339382

theorem sin_810_plus_cos_neg_60 : 
  Real.sin (810 * π / 180) + Real.cos (-60 * π / 180) = 3/2 := by
  sorry

end sin_810_plus_cos_neg_60_l3393_339382


namespace min_value_and_inequality_range_l3393_339376

theorem min_value_and_inequality_range (a b : ℝ) (ha : a ≠ 0) :
  (∀ x : ℝ, (|2*a + b| + |2*a - b|) / |a| ≥ 4) ∧
  (∀ x : ℝ, |2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|) ↔ -2 ≤ x ∧ x ≤ 2) := by
  sorry

end min_value_and_inequality_range_l3393_339376


namespace pieces_remaining_bound_l3393_339344

/-- Represents a 2n × 2n board with black and white pieces -/
structure Board (n : ℕ) where
  black_pieces : Finset (ℕ × ℕ)
  white_pieces : Finset (ℕ × ℕ)
  valid_board : ∀ (x y : ℕ), (x, y) ∈ black_pieces ∪ white_pieces → x < 2*n ∧ y < 2*n

/-- Removes black pieces on the same vertical line as white pieces -/
def remove_black (board : Board n) : Board n := sorry

/-- Removes white pieces on the same horizontal line as remaining black pieces -/
def remove_white (board : Board n) : Board n := sorry

/-- The final state of the board after removals -/
def final_board (board : Board n) : Board n := remove_white (remove_black board)

theorem pieces_remaining_bound (n : ℕ) (board : Board n) :
  (final_board board).black_pieces.card ≤ n^2 ∨ (final_board board).white_pieces.card ≤ n^2 := by
  sorry

end pieces_remaining_bound_l3393_339344


namespace difference_of_squares_l3393_339312

theorem difference_of_squares (x : ℝ) : x^2 - 121 = (x + 11) * (x - 11) := by
  sorry

end difference_of_squares_l3393_339312


namespace B_power_five_eq_scalar_multiple_l3393_339318

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 6]

theorem B_power_five_eq_scalar_multiple :
  B^5 = (4096 : ℝ) • B := by sorry

end B_power_five_eq_scalar_multiple_l3393_339318


namespace rectangle_width_25_l3393_339363

/-- A rectangle with given area and perimeter -/
structure Rectangle where
  area : ℝ
  perimeter : ℝ

/-- The width of a rectangle -/
def width (r : Rectangle) : ℝ :=
  sorry

/-- The length of a rectangle -/
def length (r : Rectangle) : ℝ :=
  sorry

theorem rectangle_width_25 (r : Rectangle) 
  (h_area : r.area = 750)
  (h_perimeter : r.perimeter = 110) :
  width r = 25 := by
  sorry

end rectangle_width_25_l3393_339363


namespace curve_transformation_l3393_339301

theorem curve_transformation (x : ℝ) : 
  Real.sin (4 * x + π / 3) = Real.cos (2 * (x - π / 24)) := by
  sorry

end curve_transformation_l3393_339301


namespace graces_pool_capacity_l3393_339368

/-- Represents the capacity of Grace's pool in gallons -/
def C : ℝ := sorry

/-- Represents the unknown initial drain rate in gallons per hour -/
def x : ℝ := sorry

/-- The rate of the first hose in gallons per hour -/
def hose1_rate : ℝ := 50

/-- The rate of the second hose in gallons per hour -/
def hose2_rate : ℝ := 70

/-- The duration of the first filling period in hours -/
def time1 : ℝ := 3

/-- The duration of the second filling period in hours -/
def time2 : ℝ := 2

/-- The increase in drain rate during the second period in gallons per hour -/
def drain_rate_increase : ℝ := 10

theorem graces_pool_capacity :
  C = (hose1_rate - x) * time1 + (hose1_rate + hose2_rate - (x + drain_rate_increase)) * time2 ∧
  C = 390 - 5 * x := by sorry

end graces_pool_capacity_l3393_339368


namespace middle_of_three_consecutive_sum_30_l3393_339319

theorem middle_of_three_consecutive_sum_30 (a b c : ℕ) :
  (a + 1 = b) ∧ (b + 1 = c) ∧ (a + b + c = 30) → b = 10 := by
  sorry

end middle_of_three_consecutive_sum_30_l3393_339319


namespace function_inequality_condition_l3393_339388

theorem function_inequality_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 3 * x + 2) →
  a > 0 →
  b > 0 →
  (∀ x, |x + 2| < b → |f x + 4| < a) ↔
  b ≤ a / 3 := by
  sorry

end function_inequality_condition_l3393_339388


namespace rectangle_length_fraction_l3393_339369

theorem rectangle_length_fraction (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ)
  (h1 : square_area = 4761)
  (h2 : rectangle_area = 598)
  (h3 : rectangle_breadth = 13) :
  let circle_radius := Real.sqrt square_area
  let rectangle_length := rectangle_area / rectangle_breadth
  rectangle_length / circle_radius = 2 / 3 := by sorry

end rectangle_length_fraction_l3393_339369


namespace elevator_problem_l3393_339379

def masses : List ℕ := [150, 60, 70, 71, 72, 100, 101, 102, 103]
def elevator_capacity : ℕ := 200

def is_valid_trip (trip : List ℕ) : Prop :=
  trip.sum ≤ elevator_capacity

def minimum_trips (m : List ℕ) (cap : ℕ) : ℕ :=
  sorry

theorem elevator_problem :
  minimum_trips masses elevator_capacity = 5 := by
  sorry

end elevator_problem_l3393_339379


namespace books_given_to_friend_l3393_339315

/-- Given that Paul initially had 134 books, sold 27 books, and was left with 68 books
    after giving some to his friend and selling in the garage sale,
    prove that the number of books Paul gave to his friend is 39. -/
theorem books_given_to_friend :
  ∀ (initial_books sold_books remaining_books books_to_friend : ℕ),
    initial_books = 134 →
    sold_books = 27 →
    remaining_books = 68 →
    initial_books - sold_books - books_to_friend = remaining_books →
    books_to_friend = 39 := by
  sorry

end books_given_to_friend_l3393_339315


namespace fraction_difference_equals_sqrt_five_l3393_339304

theorem fraction_difference_equals_sqrt_five (a b : ℝ) (h1 : a ≠ b) (h2 : 1/a + 1/b = Real.sqrt 5) :
  a / (b * (a - b)) - b / (a * (a - b)) = Real.sqrt 5 := by
  sorry

end fraction_difference_equals_sqrt_five_l3393_339304


namespace function_maximum_value_l3393_339351

theorem function_maximum_value (x : ℝ) (h : x < 0) : 
  ∃ (M : ℝ), M = -4 ∧ ∀ y, y < 0 → x + 4/x ≤ M :=
sorry

end function_maximum_value_l3393_339351


namespace circle_center_correct_l3393_339399

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 4*y - 4 = 0

/-- The center of a circle -/
def circle_center : ℝ × ℝ := (1, -2)

/-- Theorem: The center of the circle defined by the given equation is (1, -2) -/
theorem circle_center_correct :
  ∀ x y : ℝ, circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 9 :=
sorry

end circle_center_correct_l3393_339399


namespace middle_term_expansion_l3393_339306

theorem middle_term_expansion (n a : ℕ+) (h1 : n > a) (h2 : 1 + a ^ (n : ℕ) = 65) :
  let middle_term := Nat.choose n.val (n.val / 2) * a ^ (n.val / 2)
  middle_term = 160 := by
sorry

end middle_term_expansion_l3393_339306


namespace sum_of_two_numbers_l3393_339310

theorem sum_of_two_numbers (a b : ℕ) : a = 22 ∧ b = a - 10 → a + b = 34 := by
  sorry

end sum_of_two_numbers_l3393_339310


namespace unique_positive_solution_l3393_339357

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ 3 * x^2 + 13 * x - 10 = 0 :=
by
  -- The unique positive solution is x = 2/3
  use 2/3
  sorry

end unique_positive_solution_l3393_339357


namespace ages_product_l3393_339393

/-- Represents the ages of Roy, Julia, and Kelly -/
structure Ages where
  roy : ℕ
  julia : ℕ
  kelly : ℕ

/-- The conditions given in the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.roy = ages.julia + 8 ∧
  ages.roy = ages.kelly + (ages.roy - ages.julia) / 2 ∧
  ages.roy + 2 = 3 * (ages.julia + 2)

/-- The theorem to be proved -/
theorem ages_product (ages : Ages) :
  satisfiesConditions ages →
  (ages.roy + 2) * (ages.kelly + 2) = 96 := by
  sorry

end ages_product_l3393_339393


namespace juniper_bones_ratio_l3393_339361

theorem juniper_bones_ratio : 
  ∀ (initial_bones given_bones stolen_bones final_bones : ℕ),
    initial_bones = 4 →
    stolen_bones = 2 →
    final_bones = 6 →
    final_bones = initial_bones + given_bones - stolen_bones →
    (initial_bones + given_bones) / initial_bones = 2 := by
  sorry

end juniper_bones_ratio_l3393_339361


namespace cos_half_times_one_plus_sin_max_value_l3393_339395

theorem cos_half_times_one_plus_sin_max_value :
  ∀ θ : Real, 0 ≤ θ ∧ θ ≤ π / 2 →
    (∀ φ : Real, 0 ≤ φ ∧ φ ≤ π / 2 →
      Real.cos (θ / 2) * (1 + Real.sin θ) ≤ Real.cos (φ / 2) * (1 + Real.sin φ)) →
    Real.cos (θ / 2) * (1 + Real.sin θ) = 1 :=
by sorry

end cos_half_times_one_plus_sin_max_value_l3393_339395


namespace reinforcement_arrival_days_l3393_339343

/-- Calculates the number of days that passed before reinforcement arrived -/
def days_before_reinforcement (initial_garrison : ℕ) (initial_provision_days : ℕ) 
  (reinforcement_size : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_garrison := initial_garrison + reinforcement_size
  let x := (initial_garrison * initial_provision_days - total_garrison * remaining_days) / initial_garrison
  x

/-- Theorem stating that 15 days passed before reinforcement arrived -/
theorem reinforcement_arrival_days :
  days_before_reinforcement 2000 62 2700 20 = 15 := by
  sorry

#eval days_before_reinforcement 2000 62 2700 20

end reinforcement_arrival_days_l3393_339343


namespace bingo_last_column_permutations_l3393_339335

/-- The number of elements in the set to choose from -/
def n : ℕ := 10

/-- The number of elements to be chosen and arranged -/
def r : ℕ := 5

/-- The function to calculate the number of permutations -/
def permutations (n r : ℕ) : ℕ := (n - r + 1).factorial / (n - r).factorial

theorem bingo_last_column_permutations :
  permutations n r = 30240 := by sorry

end bingo_last_column_permutations_l3393_339335


namespace ball_drawing_probabilities_l3393_339350

/-- Represents the ball drawing process with 3 red and 2 white balls initially -/
structure BallDrawing where
  redBalls : ℕ := 3
  whiteBalls : ℕ := 2

/-- Probability of an event in the ball drawing process -/
def probability (event : Bool) : ℚ := sorry

/-- Event of drawing a red ball on the first draw -/
def A₁ : Bool := sorry

/-- Event of drawing a red ball on the second draw -/
def A₂ : Bool := sorry

/-- Event of drawing a white ball on the first draw -/
def B₁ : Bool := sorry

/-- Event of drawing a white ball on the second draw -/
def B₂ : Bool := sorry

/-- Event of drawing balls of the same color on both draws -/
def C : Bool := sorry

/-- Conditional probability of B₂ given A₁ -/
def conditionalProbability (B₂ A₁ : Bool) : ℚ := sorry

theorem ball_drawing_probabilities (bd : BallDrawing) :
  conditionalProbability B₂ A₁ = 3/5 ∧
  probability (B₁ ∧ A₂) = 8/25 ∧
  probability C = 8/25 := by sorry

end ball_drawing_probabilities_l3393_339350


namespace regular_icosahedron_faces_l3393_339373

/-- A regular icosahedron is a polyhedron with identical equilateral triangular faces. -/
structure RegularIcosahedron where
  is_polyhedron : Bool
  has_identical_equilateral_triangular_faces : Bool

/-- The number of faces of a regular icosahedron is 20. -/
theorem regular_icosahedron_faces (i : RegularIcosahedron) : Nat :=
  20

#check regular_icosahedron_faces

end regular_icosahedron_faces_l3393_339373


namespace sum_of_solutions_squared_equation_l3393_339338

theorem sum_of_solutions_squared_equation (x : ℝ) :
  (∀ x, (x - 4)^2 = 16 → x = 0 ∨ x = 8) →
  (∃ a b, (a - 4)^2 = 16 ∧ (b - 4)^2 = 16 ∧ a + b = 8) := by
  sorry

end sum_of_solutions_squared_equation_l3393_339338


namespace intersection_points_product_l3393_339360

theorem intersection_points_product (m : ℝ) (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) :
  (∃ y₁ y₂ : ℝ, (Real.log x₁ - 1 / x₁ = m * x₁ ∧ Real.log x₂ - 1 / x₂ = m * x₂) ∧ x₁ ≠ x₂) →
  x₁ * x₂ > 2 * Real.exp 2 :=
by sorry

end intersection_points_product_l3393_339360


namespace broken_seashells_l3393_339398

theorem broken_seashells (total : ℕ) (unbroken : ℕ) (h1 : total = 6) (h2 : unbroken = 2) :
  total - unbroken = 4 := by
  sorry

end broken_seashells_l3393_339398


namespace pete_and_raymond_spending_l3393_339390

theorem pete_and_raymond_spending :
  let initial_amount : ℕ := 250 -- $2.50 in cents
  let nickel_value : ℕ := 5
  let dime_value : ℕ := 10
  let pete_nickels_spent : ℕ := 4
  let raymond_dimes_left : ℕ := 7
  
  let pete_spent : ℕ := pete_nickels_spent * nickel_value
  let raymond_spent : ℕ := initial_amount - (raymond_dimes_left * dime_value)
  let total_spent : ℕ := pete_spent + raymond_spent

  total_spent = 200
  := by sorry

end pete_and_raymond_spending_l3393_339390


namespace total_cost_is_39_47_l3393_339367

def marbles_cost : Float := 9.05
def football_cost : Float := 4.95
def baseball_cost : Float := 6.52
def toy_car_original_cost : Float := 6.50
def toy_car_discount_percent : Float := 20
def puzzle_cost : Float := 3.25
def puzzle_quantity : Nat := 2
def action_figure_discounted_cost : Float := 10.50

def calculate_discounted_price (original_price : Float) (discount_percent : Float) : Float :=
  original_price * (1 - discount_percent / 100)

def calculate_total_cost : Float :=
  marbles_cost +
  football_cost +
  baseball_cost +
  calculate_discounted_price toy_car_original_cost toy_car_discount_percent +
  puzzle_cost +
  action_figure_discounted_cost

theorem total_cost_is_39_47 :
  calculate_total_cost = 39.47 := by sorry

end total_cost_is_39_47_l3393_339367


namespace part_one_part_two_l3393_339314

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Theorem for part (1)
theorem part_one : 
  (Set.univ \ A) ∪ (B 1) = {x : ℝ | x ≤ -2 ∨ x > 1} := by sorry

-- Theorem for part (2)
theorem part_two : 
  ∀ a : ℝ, A ⊆ B a ↔ a ≤ -2 := by sorry

end part_one_part_two_l3393_339314


namespace second_shop_expense_l3393_339333

theorem second_shop_expense (first_shop_books : ℕ) (second_shop_books : ℕ) 
  (first_shop_cost : ℕ) (average_price : ℕ) (total_books : ℕ)
  (h1 : first_shop_books = 65)
  (h2 : second_shop_books = 35)
  (h3 : first_shop_cost = 6500)
  (h4 : average_price = 85)
  (h5 : total_books = first_shop_books + second_shop_books) :
  (average_price * total_books) - first_shop_cost = 2000 := by
  sorry

end second_shop_expense_l3393_339333


namespace fraction_inequality_l3393_339394

theorem fraction_inequality (m n : ℝ) (h : m > n) : m / 4 > n / 4 := by
  sorry

end fraction_inequality_l3393_339394


namespace sum_of_naturals_equals_1035_l3393_339384

theorem sum_of_naturals_equals_1035 (n : ℕ) : (n * (n + 1)) / 2 = 1035 → n = 46 := by
  sorry

end sum_of_naturals_equals_1035_l3393_339384


namespace triangle_side_length_l3393_339359

theorem triangle_side_length (A B C : ℝ × ℝ) (tanB : ℝ) (AB : ℝ) :
  tanB = 4 / 3 →
  AB = 3 →
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = (AB * tanB)^2 →
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 25 :=
by sorry

end triangle_side_length_l3393_339359


namespace factors_of_x4_minus_4_l3393_339303

theorem factors_of_x4_minus_4 (x : ℝ) : 
  (x^4 - 4 = (x^2 + 2) * (x^2 - 2)) ∧ 
  (x^4 - 4 = (x^2 - 4) * (x^2 + 4)) ∧ 
  (x^4 - 4 ≠ (x + 1) * ((x^3 - x^2 - x + 5) / (x + 1))) ∧ 
  (x^4 - 4 ≠ (x^2 - 2*x + 2) * ((x^2 + 2*x + 2) / (x^2 - 2*x + 2))) :=
by sorry

end factors_of_x4_minus_4_l3393_339303


namespace orthogonal_vectors_l3393_339323

/-- Given non-zero plane vectors a and b satisfying |a + b| = |a - b|, prove that a ⋅ b = 0 -/
theorem orthogonal_vectors (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0)) 
  (h : ‖a + b‖ = ‖a - b‖) : a • b = 0 := by
  sorry

end orthogonal_vectors_l3393_339323


namespace intersection_with_complement_l3393_339322

def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {0, 1, 2}
def B : Set Int := {2, 3}

theorem intersection_with_complement :
  A ∩ (U \ B) = {0, 1} := by sorry

end intersection_with_complement_l3393_339322


namespace division_fraction_proof_l3393_339317

theorem division_fraction_proof : (5 : ℚ) / ((8 : ℚ) / 13) = 65 / 8 := by
  sorry

end division_fraction_proof_l3393_339317


namespace percentage_decrease_l3393_339371

theorem percentage_decrease (original : ℝ) (increase_percent : ℝ) (difference : ℝ) :
  original = 80 →
  increase_percent = 12.5 →
  difference = 30 →
  let increased_value := original * (1 + increase_percent / 100)
  let decrease_percent := (increased_value - original - difference) / original * 100
  decrease_percent = 25 := by sorry

end percentage_decrease_l3393_339371


namespace remaining_length_is_90_cm_l3393_339391

-- Define the initial length in meters
def initial_length : ℝ := 1

-- Define the erased length in centimeters
def erased_length : ℝ := 10

-- Theorem to prove
theorem remaining_length_is_90_cm :
  (initial_length * 100 - erased_length) = 90 := by
  sorry

end remaining_length_is_90_cm_l3393_339391


namespace inequality_two_integer_solutions_l3393_339307

theorem inequality_two_integer_solutions (k : ℝ) : 
  (∃ (x y : ℕ), x ≠ y ∧ 
    (k * (x : ℝ)^2 ≤ Real.log x + 1) ∧ 
    (k * (y : ℝ)^2 ≤ Real.log y + 1) ∧
    (∀ (z : ℕ), z ≠ x ∧ z ≠ y → k * (z : ℝ)^2 > Real.log z + 1)) →
  ((Real.log 3 + 1) / 9 < k ∧ k ≤ (Real.log 2 + 1) / 4) :=
by sorry

end inequality_two_integer_solutions_l3393_339307


namespace sum_of_roots_quadratic_l3393_339336

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → (∃ r₁ r₂ : ℝ, (r₁ + r₂ = 6) ∧ (x = r₁ ∨ x = r₂)) :=
by sorry

end sum_of_roots_quadratic_l3393_339336


namespace f_extremum_l3393_339331

/-- The function f(x, y) -/
def f (x y : ℝ) : ℝ := x^3 + 3*x*y^2 - 18*x^2 - 18*x*y - 18*y^2 + 57*x + 138*y + 290

/-- Theorem stating the extremum of f(x, y) -/
theorem f_extremum :
  (∃ (x y : ℝ), f x y = 10 ∧ ∀ (a b : ℝ), f a b ≥ 10) ∧
  (∃ (x y : ℝ), f x y = 570 ∧ ∀ (a b : ℝ), f a b ≤ 570) :=
sorry

end f_extremum_l3393_339331


namespace sum_equals_rounded_sum_l3393_339302

def round_to_nearest_five (n : ℕ) : ℕ :=
  5 * ((n + 2) / 5)

def sum_to_n (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_rounded_to_n (n : ℕ) : ℕ :=
  (List.range n).map round_to_nearest_five |>.sum

theorem sum_equals_rounded_sum (n : ℕ) (h : n = 200) : 
  sum_to_n n = sum_rounded_to_n n := by
  sorry

#eval sum_to_n 200
#eval sum_rounded_to_n 200

end sum_equals_rounded_sum_l3393_339302


namespace lillian_candy_count_l3393_339362

/-- The number of candies Lillian has after receiving candies from her father -/
def total_candies (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Lillian has 93 candies after receiving candies from her father -/
theorem lillian_candy_count :
  total_candies 88 5 = 93 := by
  sorry

end lillian_candy_count_l3393_339362


namespace field_trip_buses_l3393_339383

theorem field_trip_buses (total_classrooms : ℕ) (freshmen_classrooms : ℕ) (sophomore_classrooms : ℕ)
  (freshmen_per_room : ℕ) (sophomores_per_room : ℕ) (bus_capacity : ℕ) (teachers_per_room : ℕ)
  (bus_drivers : ℕ) :
  total_classrooms = 95 →
  freshmen_classrooms = 45 →
  sophomore_classrooms = 50 →
  freshmen_per_room = 58 →
  sophomores_per_room = 47 →
  bus_capacity = 40 →
  teachers_per_room = 2 →
  bus_drivers = 15 →
  ∃ (buses : ℕ), buses = 130 ∧ 
    buses * bus_capacity ≥ 
      freshmen_classrooms * freshmen_per_room + 
      sophomore_classrooms * sophomores_per_room + 
      total_classrooms * teachers_per_room + 
      bus_drivers ∧
    (buses - 1) * bus_capacity < 
      freshmen_classrooms * freshmen_per_room + 
      sophomore_classrooms * sophomores_per_room + 
      total_classrooms * teachers_per_room + 
      bus_drivers :=
by
  sorry


end field_trip_buses_l3393_339383


namespace moore_law_gpu_transistors_l3393_339340

def initial_year : Nat := 1992
def final_year : Nat := 2011
def initial_transistors : Nat := 500000
def doubling_period : Nat := 3

def moore_law_prediction (initial : Nat) (years : Nat) (period : Nat) : Nat :=
  initial * (2 ^ (years / period))

theorem moore_law_gpu_transistors :
  moore_law_prediction initial_transistors (final_year - initial_year) doubling_period = 32000000 := by
  sorry

end moore_law_gpu_transistors_l3393_339340


namespace car_rental_problem_l3393_339385

/-- Represents the characteristics of a car type -/
structure CarType where
  capacity : ℕ
  rentalFee : ℕ

/-- Represents a rental option -/
structure RentalOption where
  typeACars : ℕ
  typeBCars : ℕ

/-- Checks if a rental option is valid given the constraints -/
def isValidRental (opt : RentalOption) (typeA typeB : CarType) (totalCars maxCost totalPeople : ℕ) : Prop :=
  opt.typeACars + opt.typeBCars = totalCars ∧
  opt.typeACars > 0 ∧
  opt.typeBCars > 0 ∧
  opt.typeACars * typeA.rentalFee + opt.typeBCars * typeB.rentalFee ≤ maxCost ∧
  opt.typeACars * typeA.capacity + opt.typeBCars * typeB.capacity ≥ totalPeople

/-- Calculates the total cost of a rental option -/
def rentalCost (opt : RentalOption) (typeA typeB : CarType) : ℕ :=
  opt.typeACars * typeA.rentalFee + opt.typeBCars * typeB.rentalFee

theorem car_rental_problem (typeA typeB : CarType) 
    (h_typeA_capacity : typeA.capacity = 50)
    (h_typeA_fee : typeA.rentalFee = 400)
    (h_typeB_capacity : typeB.capacity = 30)
    (h_typeB_fee : typeB.rentalFee = 280)
    (totalCars : ℕ) (h_totalCars : totalCars = 10)
    (maxCost : ℕ) (h_maxCost : maxCost = 3500)
    (totalPeople : ℕ) (h_totalPeople : totalPeople = 360) :
  (∃ (opt : RentalOption), isValidRental opt typeA typeB totalCars maxCost totalPeople ∧ 
    opt.typeACars = 5 ∧ 
    (∀ (opt' : RentalOption), isValidRental opt' typeA typeB totalCars maxCost totalPeople → 
      opt'.typeACars ≤ opt.typeACars)) ∧
  (∃ (optCostEffective : RentalOption), 
    isValidRental optCostEffective typeA typeB totalCars maxCost totalPeople ∧
    optCostEffective.typeACars = 3 ∧ 
    optCostEffective.typeBCars = 7 ∧
    (∀ (opt' : RentalOption), isValidRental opt' typeA typeB totalCars maxCost totalPeople → 
      rentalCost optCostEffective typeA typeB ≤ rentalCost opt' typeA typeB)) := by
  sorry

end car_rental_problem_l3393_339385


namespace smallest_m_divisibility_l3393_339365

theorem smallest_m_divisibility (p : Nat) (h_prime : Nat.Prime p) (h_p_gt_3 : p > 3) :
  ∃ (m : Nat), m > 0 ∧ (∀ (q : Nat), Nat.Prime q → q > 3 → 105 ∣ 9^(q^2) - 29^q + m) ∧
  (∀ (k : Nat), k > 0 → k < m → ∃ (r : Nat), Nat.Prime r → r > 3 → ¬(105 ∣ 9^(r^2) - 29^r + k)) ∧
  m = 95 :=
sorry

end smallest_m_divisibility_l3393_339365


namespace complex_equation_modulus_l3393_339377

theorem complex_equation_modulus : ∀ (x y : ℝ), 
  (Complex.I : ℂ) * x + 2 * (Complex.I : ℂ) * x = (2 : ℂ) + (Complex.I : ℂ) * y → 
  Complex.abs (x + (Complex.I : ℂ) * y) = 2 * Real.sqrt 5 :=
by
  sorry

end complex_equation_modulus_l3393_339377


namespace one_third_comparison_l3393_339342

theorem one_third_comparison : (1 / 3 : ℚ) - (33333333 / 100000000 : ℚ) = 1 / (3 * 100000000) := by
  sorry

end one_third_comparison_l3393_339342


namespace quadratic_roots_to_coefficients_l3393_339337

theorem quadratic_roots_to_coefficients :
  ∀ (b c : ℝ),
  (∀ x : ℝ, 2 * x^2 + b * x + c = 0 ↔ x = -1 ∨ x = 3) →
  b = -4 ∧ c = -6 :=
by
  sorry

end quadratic_roots_to_coefficients_l3393_339337


namespace digit_sum_theorem_l3393_339353

/-- Given single-digit integers a and b satisfying the equation 3a * (10b + 4) = 146, 
    prove that a + b = 13 -/
theorem digit_sum_theorem (a b : ℕ) : 
  a < 10 → b < 10 → 3 * a * (10 * b + 4) = 146 → a + b = 13 := by
  sorry

end digit_sum_theorem_l3393_339353


namespace g_increasing_iff_a_in_range_l3393_339341

-- Define the piecewise function g(x)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ -1 then -a / (x - 1) else (3 - 3*a) * x + 1

-- State the theorem
theorem g_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → g a x < g a y) ↔ (4/5 ≤ a ∧ a < 1) :=
by sorry

end g_increasing_iff_a_in_range_l3393_339341


namespace pure_imaginary_solutions_l3393_339354

def polynomial (x : ℂ) : ℂ := x^4 - 4*x^3 + 10*x^2 - 64*x - 100

theorem pure_imaginary_solutions :
  ∀ x : ℂ, polynomial x = 0 ∧ ∃ k : ℝ, x = k * I ↔ x = 4 * I ∨ x = -4 * I :=
by sorry

end pure_imaginary_solutions_l3393_339354


namespace quadratic_function_properties_l3393_339329

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (h1 : f a b c (-1) = 0)
  (h2 : f a b c 0 = -3)
  (h3 : f a b c 2 = -3) :
  (∃ x y : ℝ, 
    (∀ z : ℝ, f a b c z = z^2 - 2*z - 3) ∧
    (x = 1 ∧ y = -4 ∧ ∀ z : ℝ, f a b c z ≥ f a b c x) ∧
    (∀ z : ℝ, z > 1 → ∀ w : ℝ, w > z → f a b c w > f a b c z) ∧
    (∀ z : ℝ, -1 < z ∧ z < 2 → -4 < f a b c z ∧ f a b c z < 0)) :=
by sorry

end quadratic_function_properties_l3393_339329


namespace y_divisibility_l3393_339324

def y : ℕ := 32 + 48 + 64 + 96 + 200 + 224 + 1600

theorem y_divisibility :
  (∃ k : ℕ, y = 4 * k) ∧
  (∃ k : ℕ, y = 8 * k) ∧
  (∃ k : ℕ, y = 16 * k) ∧
  ¬(∃ k : ℕ, y = 32 * k) :=
by sorry

end y_divisibility_l3393_339324


namespace phone_sale_problem_l3393_339374

theorem phone_sale_problem (total : ℕ) (defective : ℕ) (customer_a : ℕ) (customer_b : ℕ) 
  (h_total : total = 20)
  (h_defective : defective = 5)
  (h_customer_a : customer_a = 3)
  (h_customer_b : customer_b = 5)
  (h_all_sold : total - defective = customer_a + customer_b + (total - defective - customer_a - customer_b)) :
  total - defective - customer_a - customer_b = 7 := by
  sorry

end phone_sale_problem_l3393_339374


namespace f_properties_l3393_339330

noncomputable section

def f (x : ℝ) : ℝ :=
  if x ≤ -1 then x + 2
  else if x < 2 then x^2
  else 2*x

theorem f_properties :
  (f 2 = 4) ∧
  (f (1/2) = 1/4) ∧
  (f (f (-1)) = 1) ∧
  (∃ a : ℝ, f a = 3 ∧ (a = 1 ∨ a = Real.sqrt 3)) :=
by sorry

end

end f_properties_l3393_339330


namespace toy_store_order_l3393_339392

theorem toy_store_order (stored_toys : ℕ) (storage_percentage : ℚ) (total_toys : ℕ) :
  stored_toys = 140 →
  storage_percentage = 7/10 →
  (storage_percentage * total_toys : ℚ) = stored_toys →
  total_toys = 200 := by
sorry

end toy_store_order_l3393_339392


namespace f_4_has_eight_zeros_l3393_339327

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 1|

-- Define the recursive function f_n
def f_n : ℕ → (ℝ → ℝ)
  | 0 => id
  | 1 => f
  | (n + 1) => f ∘ f_n n

-- State the theorem
theorem f_4_has_eight_zeros :
  ∃! (zeros : Finset ℝ), zeros.card = 8 ∧ ∀ x ∈ zeros, f_n 4 x = 0 :=
sorry

end f_4_has_eight_zeros_l3393_339327


namespace f_properties_l3393_339346

-- Define the function f(x) = x^2 - 2x + 1
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Theorem stating the properties of f(x)
theorem f_properties :
  (∃ x : ℝ, f x = 0 ∧ x = 1) ∧
  (f 0 * f 2 > 0) ∧
  (¬ ∀ x y : ℝ, x < y → x < 0 → f x > f y) ∧
  (∀ x : ℝ, x < 0 → f x ≠ 0) :=
by sorry

end f_properties_l3393_339346


namespace min_value_expression_l3393_339328

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^3 + 8 * b^3 + 27 * c^3 + 1 / (3 * a * b * c) ≥ 6 ∧
  (4 * a^3 + 8 * b^3 + 27 * c^3 + 1 / (3 * a * b * c) = 6 ↔
    a = 1 / Real.rpow 6 (1/3) ∧ b = 1 / Real.rpow 12 (1/3) ∧ c = 1 / Real.rpow 54 (1/3)) :=
by sorry

end min_value_expression_l3393_339328


namespace uncle_james_height_difference_l3393_339396

theorem uncle_james_height_difference :
  ∀ (james_original_height james_new_height uncle_height : ℝ),
  uncle_height = 72 →
  james_original_height = (2/3) * uncle_height →
  james_new_height = james_original_height + 10 →
  uncle_height - james_new_height = 14 :=
by
  sorry

end uncle_james_height_difference_l3393_339396


namespace total_students_is_90_l3393_339356

/-- Represents a class with its exam statistics -/
structure ClassStats where
  totalStudents : ℕ
  averageMark : ℚ
  excludedStudents : ℕ
  excludedAverage : ℚ
  newAverage : ℚ

/-- Calculate the total number of students across all classes -/
def totalStudents (classA classB classC : ClassStats) : ℕ :=
  classA.totalStudents + classB.totalStudents + classC.totalStudents

/-- Theorem stating that the total number of students is 90 -/
theorem total_students_is_90 (classA classB classC : ClassStats)
  (hA : classA.averageMark = 80 ∧ classA.excludedStudents = 5 ∧
        classA.excludedAverage = 20 ∧ classA.newAverage = 92)
  (hB : classB.averageMark = 75 ∧ classB.excludedStudents = 6 ∧
        classB.excludedAverage = 25 ∧ classB.newAverage = 85)
  (hC : classC.averageMark = 70 ∧ classC.excludedStudents = 4 ∧
        classC.excludedAverage = 30 ∧ classC.newAverage = 78) :
  totalStudents classA classB classC = 90 := by
  sorry


end total_students_is_90_l3393_339356


namespace u_general_term_l3393_339300

def u : ℕ → ℚ
  | 0 => 1
  | 1 => 2
  | 2 => 0
  | (n + 3) => 2 * u (n + 2) + u (n + 1) - 2 * u n

theorem u_general_term : ∀ n : ℕ, u n = 2 - (2/3) * (-1)^n - (1/3) * 2^n := by
  sorry

end u_general_term_l3393_339300


namespace regular_polygon_tiling_l3393_339387

theorem regular_polygon_tiling (x y z : ℕ) (hx : x > 2) (hy : y > 2) (hz : z > 2) :
  (((x - 2 : ℝ) / x + (y - 2 : ℝ) / y + (z - 2 : ℝ) / z) = 2) →
  (1 / x + 1 / y + 1 / z : ℝ) = 1 / 2 := by
  sorry

end regular_polygon_tiling_l3393_339387


namespace special_arrangement_count_l3393_339370

/-- The number of permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of ways to arrange n people in a row -/
def linearArrangements (n : ℕ) : ℕ := factorial n

/-- The number of ways to arrange 5 people in a row, where 2 specific people
    must be adjacent and in a specific order -/
def specialArrangement : ℕ := linearArrangements 4

theorem special_arrangement_count :
  specialArrangement = 24 :=
sorry

end special_arrangement_count_l3393_339370


namespace garden_area_not_covered_by_flower_beds_l3393_339381

def garden_side_length : ℝ := 16
def flower_bed_radius : ℝ := 8

theorem garden_area_not_covered_by_flower_beds :
  let total_area := garden_side_length ^ 2
  let flower_bed_area := 4 * (π * flower_bed_radius ^ 2) / 4
  total_area - flower_bed_area = 256 - 64 * π := by sorry

end garden_area_not_covered_by_flower_beds_l3393_339381


namespace complex_equation_solution_l3393_339386

theorem complex_equation_solution (x : ℂ) : 
  Complex.abs x = 1 + 3 * Complex.I - x → x = -4 + 3 * Complex.I := by
  sorry

end complex_equation_solution_l3393_339386


namespace gcd_272_595_l3393_339339

theorem gcd_272_595 : Nat.gcd 272 595 = 17 := by
  sorry

end gcd_272_595_l3393_339339


namespace four_letter_words_with_a_l3393_339378

/-- The number of letters in the alphabet we're using -/
def alphabet_size : ℕ := 5

/-- The length of the words we're forming -/
def word_length : ℕ := 4

/-- The number of letters in the alphabet excluding 'A' -/
def alphabet_size_without_a : ℕ := 4

/-- The total number of possible 4-letter words using all 5 letters -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of 4-letter words not containing 'A' -/
def words_without_a : ℕ := alphabet_size_without_a ^ word_length

/-- The number of 4-letter words containing at least one 'A' -/
def words_with_a : ℕ := total_words - words_without_a

theorem four_letter_words_with_a : words_with_a = 369 := by
  sorry

end four_letter_words_with_a_l3393_339378


namespace smallest_number_divisibility_l3393_339313

theorem smallest_number_divisibility (n : ℕ) : n = 4722 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    m + 3 = 27 * k₁ ∧ 
    m + 3 = 35 * k₂ ∧ 
    m + 3 = 25 * k₃ ∧ 
    m + 3 = 21 * k₄)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    n + 3 = 27 * k₁ ∧ 
    n + 3 = 35 * k₂ ∧ 
    n + 3 = 25 * k₃ ∧ 
    n + 3 = 21 * k₄) := by
  sorry

#check smallest_number_divisibility

end smallest_number_divisibility_l3393_339313


namespace det_special_matrix_l3393_339358

def matrix (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![x + 2, x, x;
     x, x + 2, x;
     x, x, x + 2]

theorem det_special_matrix (x : ℝ) :
  Matrix.det (matrix x) = 8 * x + 4 := by
  sorry

end det_special_matrix_l3393_339358


namespace det_specific_matrix_l3393_339326

theorem det_specific_matrix :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 2, 0; 4, 5, -3; 7, 8, 6]
  Matrix.det A = -36 := by
  sorry

end det_specific_matrix_l3393_339326


namespace election_majority_l3393_339311

theorem election_majority (total_votes : ℕ) (winning_percentage : ℚ) : 
  total_votes = 500 →
  winning_percentage = 70/100 →
  (winning_percentage * total_votes : ℚ).num - ((1 - winning_percentage) * total_votes : ℚ).num = 200 := by
sorry

end election_majority_l3393_339311


namespace area_ratio_similar_triangles_l3393_339366

/-- Given two similar triangles with areas S and S₁, and similarity coefficient k, 
    prove that the ratio of their areas is equal to the square of the similarity coefficient. -/
theorem area_ratio_similar_triangles (S S₁ k : ℝ) (a b a₁ b₁ α : ℝ) :
  S = (1 / 2) * a * b * Real.sin α →
  S₁ = (1 / 2) * a₁ * b₁ * Real.sin α →
  a₁ = k * a →
  b₁ = k * b →
  k > 0 →
  S₁ / S = k^2 := by
  sorry

end area_ratio_similar_triangles_l3393_339366


namespace representative_selection_count_l3393_339375

def male_count : ℕ := 5
def female_count : ℕ := 4
def total_representatives : ℕ := 4
def min_male : ℕ := 2
def min_female : ℕ := 1

theorem representative_selection_count : 
  (Nat.choose male_count 2 * Nat.choose female_count 2) + 
  (Nat.choose male_count 3 * Nat.choose female_count 1) = 100 := by
  sorry

end representative_selection_count_l3393_339375


namespace abs_sum_complex_roots_l3393_339325

theorem abs_sum_complex_roots (a b c : ℂ) 
  (h1 : Complex.abs a = 1) 
  (h2 : Complex.abs b = 1) 
  (h3 : Complex.abs c = 1)
  (h4 : a^3 / (b^2 * c) + b^3 / (a^2 * c) + c^3 / (a^2 * b) = 1) :
  Complex.abs (a + b + c) = 1 ∨ Complex.abs (a + b + c) = 3 := by
  sorry

end abs_sum_complex_roots_l3393_339325


namespace short_sleeve_shirts_count_l3393_339372

/-- The number of short sleeve shirts washed -/
def short_sleeve_shirts : ℕ := 9 - 5

/-- The total number of shirts washed -/
def total_shirts : ℕ := 9

/-- The number of long sleeve shirts washed -/
def long_sleeve_shirts : ℕ := 5

theorem short_sleeve_shirts_count : short_sleeve_shirts = 4 := by
  sorry

end short_sleeve_shirts_count_l3393_339372


namespace selection_methods_eq_51_l3393_339348

/-- The number of ways to select k elements from n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 9

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of specific students (A, B, C) -/
def specific_students : ℕ := 3

/-- The number of ways to select 4 students from 9, where at least two of three specific students must be selected -/
def selection_methods : ℕ :=
  choose specific_students 2 * choose (total_students - specific_students) (selected_students - 2) +
  choose specific_students 3 * choose (total_students - specific_students) (selected_students - 3)

theorem selection_methods_eq_51 : selection_methods = 51 := by sorry

end selection_methods_eq_51_l3393_339348


namespace integral_power_x_l3393_339355

theorem integral_power_x (a : ℝ) (h : a > 0) : ∫ x in (0:ℝ)..1, x^a = 1 / (a + 1) := by sorry

end integral_power_x_l3393_339355
