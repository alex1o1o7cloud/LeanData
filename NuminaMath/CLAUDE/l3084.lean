import Mathlib

namespace NUMINAMATH_CALUDE_minesweeper_sum_invariant_l3084_308462

/-- Represents a cell in the Minesweeper grid -/
inductive Cell
| Mine : Cell
| Number (n : ℕ) : Cell

/-- A 10x10 Minesweeper grid -/
def MinesweeperGrid := Fin 10 → Fin 10 → Cell

/-- Calculates the sum of all numbers in a Minesweeper grid -/
def gridSum (grid : MinesweeperGrid) : ℕ := sorry

/-- Flips the state of all cells in a Minesweeper grid -/
def flipGrid (grid : MinesweeperGrid) : MinesweeperGrid := sorry

/-- Theorem stating that the sum of numbers remains constant after flipping the grid -/
theorem minesweeper_sum_invariant (grid : MinesweeperGrid) : 
  gridSum grid = gridSum (flipGrid grid) := by sorry

end NUMINAMATH_CALUDE_minesweeper_sum_invariant_l3084_308462


namespace NUMINAMATH_CALUDE_boat_trips_theorem_l3084_308483

/-- The number of boat trips in one day -/
def boat_trips_per_day (boat_capacity : ℕ) (people_per_two_days : ℕ) : ℕ :=
  (people_per_two_days / 2) / boat_capacity

/-- Theorem: The number of boat trips in one day is 4 -/
theorem boat_trips_theorem :
  boat_trips_per_day 12 96 = 4 := by
  sorry

end NUMINAMATH_CALUDE_boat_trips_theorem_l3084_308483


namespace NUMINAMATH_CALUDE_soccer_stars_points_l3084_308489

/-- Calculates the total points for a soccer team given their game results -/
def calculate_total_points (total_games wins losses : ℕ) : ℕ :=
  let draws := total_games - wins - losses
  let points_per_win := 3
  let points_per_draw := 1
  let points_per_loss := 0
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss

/-- Theorem stating that the Soccer Stars team's total points is 46 -/
theorem soccer_stars_points :
  calculate_total_points 20 14 2 = 46 := by
  sorry

end NUMINAMATH_CALUDE_soccer_stars_points_l3084_308489


namespace NUMINAMATH_CALUDE_cupcakes_left_over_l3084_308467

/-- The number of cupcakes Quinton brought to school -/
def total_cupcakes : ℕ := 80

/-- The number of students in Ms. Delmont's class -/
def ms_delmont_students : ℕ := 18

/-- The number of students in Mrs. Donnelly's class -/
def mrs_donnelly_students : ℕ := 16

/-- The number of school custodians -/
def custodians : ℕ := 3

/-- The number of Quinton's favorite teachers -/
def favorite_teachers : ℕ := 5

/-- The number of other classmates who received cupcakes -/
def other_classmates : ℕ := 10

/-- The number of cupcakes given to each favorite teacher -/
def cupcakes_per_favorite_teacher : ℕ := 2

/-- The total number of cupcakes given away -/
def cupcakes_given_away : ℕ :=
  ms_delmont_students + mrs_donnelly_students + 2 + 1 + 1 + custodians +
  (favorite_teachers * cupcakes_per_favorite_teacher) + other_classmates

/-- Theorem stating the number of cupcakes Quinton has left over -/
theorem cupcakes_left_over : total_cupcakes - cupcakes_given_away = 19 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_left_over_l3084_308467


namespace NUMINAMATH_CALUDE_cake_distribution_l3084_308412

theorem cake_distribution (total_cake : ℕ) (friends : ℕ) (pieces_per_friend : ℕ) 
  (h1 : total_cake = 150)
  (h2 : friends = 50)
  (h3 : pieces_per_friend * friends = total_cake) :
  pieces_per_friend = 3 := by
  sorry

end NUMINAMATH_CALUDE_cake_distribution_l3084_308412


namespace NUMINAMATH_CALUDE_badge_exchange_problem_l3084_308438

theorem badge_exchange_problem (vasya_initial tolya_initial : ℕ) : 
  vasya_initial = 50 ∧ tolya_initial = 45 →
  vasya_initial = tolya_initial + 5 ∧
  (vasya_initial - (vasya_initial * 24 / 100) + (tolya_initial * 20 / 100)) + 1 =
  (tolya_initial - (tolya_initial * 20 / 100) + (vasya_initial * 24 / 100)) :=
by sorry

end NUMINAMATH_CALUDE_badge_exchange_problem_l3084_308438


namespace NUMINAMATH_CALUDE_regression_line_not_necessarily_through_points_l3084_308444

/-- Sample data point -/
structure DataPoint where
  x : ℝ
  y : ℝ

/-- Linear regression model -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Predicts y value for a given x using the linear regression model -/
def predict (model : LinearRegression) (x : ℝ) : ℝ :=
  model.a + model.b * x

/-- Checks if a point lies on the regression line -/
def pointOnLine (model : LinearRegression) (point : DataPoint) : Prop :=
  predict model point.x = point.y

theorem regression_line_not_necessarily_through_points 
  (model : LinearRegression) (data : List DataPoint) : 
  ¬ (∀ point ∈ data, pointOnLine model point) := by
  sorry

#check regression_line_not_necessarily_through_points

end NUMINAMATH_CALUDE_regression_line_not_necessarily_through_points_l3084_308444


namespace NUMINAMATH_CALUDE_trick_deck_cost_l3084_308400

theorem trick_deck_cost (tom_decks : ℕ) (friend_decks : ℕ) (total_spent : ℕ) 
  (h1 : tom_decks = 3)
  (h2 : friend_decks = 5)
  (h3 : total_spent = 64) :
  (total_spent : ℚ) / (tom_decks + friend_decks : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_trick_deck_cost_l3084_308400


namespace NUMINAMATH_CALUDE_adult_ticket_cost_l3084_308428

theorem adult_ticket_cost (total_tickets : ℕ) (senior_price : ℕ) (total_receipts : ℕ) (senior_tickets : ℕ) :
  total_tickets = 510 →
  senior_price = 15 →
  total_receipts = 8748 →
  senior_tickets = 327 →
  (total_tickets - senior_tickets) * (total_receipts - senior_tickets * senior_price) / (total_tickets - senior_tickets) = 21 :=
by sorry

end NUMINAMATH_CALUDE_adult_ticket_cost_l3084_308428


namespace NUMINAMATH_CALUDE_factors_of_1728_l3084_308410

def number_of_factors (n : ℕ) : ℕ := (Nat.divisors n).card

theorem factors_of_1728 : number_of_factors 1728 = 28 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_1728_l3084_308410


namespace NUMINAMATH_CALUDE_nantucket_meeting_attendance_l3084_308464

theorem nantucket_meeting_attendance :
  let total_population : ℕ := 300
  let females_attending : ℕ := 50
  let males_attending : ℕ := 2 * females_attending
  let total_attending : ℕ := males_attending + females_attending
  (total_attending : ℚ) / total_population = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_nantucket_meeting_attendance_l3084_308464


namespace NUMINAMATH_CALUDE_manufacturing_degrees_l3084_308487

/-- Represents the number of degrees in a full circle. -/
def full_circle : ℝ := 360

/-- Represents the percentage of employees in manufacturing as a decimal. -/
def manufacturing_percentage : ℝ := 0.20

/-- Calculates the number of degrees in a circle graph for a given percentage. -/
def degrees_for_percentage (percentage : ℝ) : ℝ := full_circle * percentage

/-- Theorem: The manufacturing section in the circle graph takes up 72 degrees. -/
theorem manufacturing_degrees :
  degrees_for_percentage manufacturing_percentage = 72 := by
  sorry

end NUMINAMATH_CALUDE_manufacturing_degrees_l3084_308487


namespace NUMINAMATH_CALUDE_area_of_T_is_34_l3084_308434

/-- The area of a "T" shape formed within a rectangle -/
def area_of_T (rectangle_width rectangle_height removed_width removed_height : ℕ) : ℕ :=
  rectangle_width * rectangle_height - removed_width * removed_height

/-- Theorem stating that the area of the "T" shape is 34 square units -/
theorem area_of_T_is_34 :
  area_of_T 10 4 6 1 = 34 := by
  sorry

end NUMINAMATH_CALUDE_area_of_T_is_34_l3084_308434


namespace NUMINAMATH_CALUDE_factorization_equality_l3084_308409

theorem factorization_equality (x₁ x₂ : ℝ) :
  x₁^3 - 2*x₁^2*x₂ - x₁ + 2*x₂ = (x₁ - 1) * (x₁ + 1) * (x₁ - 2*x₂) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3084_308409


namespace NUMINAMATH_CALUDE_bowTie_seven_eq_nine_impl_g_eq_two_l3084_308496

/-- The bow-tie operation defined as a + √(b + √(b + √(b + ...))) -/
noncomputable def bowTie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

/-- Theorem stating that if 7 ⋈ g = 9, then g = 2 -/
theorem bowTie_seven_eq_nine_impl_g_eq_two :
  ∀ g : ℝ, bowTie 7 g = 9 → g = 2 := by
  sorry

end NUMINAMATH_CALUDE_bowTie_seven_eq_nine_impl_g_eq_two_l3084_308496


namespace NUMINAMATH_CALUDE_oliver_learning_time_l3084_308466

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The total number of days Oliver needs to learn all vowels -/
def total_days : ℕ := 25

/-- The number of days Oliver needs to learn one vowel -/
def days_per_vowel : ℕ := total_days / num_vowels

theorem oliver_learning_time : days_per_vowel = 5 := by
  sorry

end NUMINAMATH_CALUDE_oliver_learning_time_l3084_308466


namespace NUMINAMATH_CALUDE_george_coin_value_l3084_308425

/-- Calculates the total value of coins given the number of nickels and dimes -/
def totalCoinValue (totalCoins : ℕ) (nickels : ℕ) (nickelValue : ℚ) (dimeValue : ℚ) : ℚ :=
  let dimes := totalCoins - nickels
  nickels * nickelValue + dimes * dimeValue

theorem george_coin_value :
  totalCoinValue 28 4 (5 / 100) (10 / 100) = 260 / 100 := by
  sorry

end NUMINAMATH_CALUDE_george_coin_value_l3084_308425


namespace NUMINAMATH_CALUDE_new_oarsman_weight_l3084_308492

/-- Given a crew of 10 oarsmen, proves that replacing a 53 kg member with a new member
    that increases the average weight by 1.8 kg results in the new member weighing 71 kg. -/
theorem new_oarsman_weight (crew_size : Nat) (old_weight : ℝ) (avg_increase : ℝ) :
  crew_size = 10 →
  old_weight = 53 →
  avg_increase = 1.8 →
  (crew_size : ℝ) * avg_increase + old_weight = 71 :=
by sorry

end NUMINAMATH_CALUDE_new_oarsman_weight_l3084_308492


namespace NUMINAMATH_CALUDE_total_amount_is_70000_l3084_308499

/-- The total amount of money divided -/
def total_amount : ℕ := sorry

/-- The amount given at 10% interest -/
def amount_10_percent : ℕ := 60000

/-- The amount given at 20% interest -/
def amount_20_percent : ℕ := sorry

/-- The interest rate for the first part (10%) -/
def interest_rate_10 : ℚ := 1/10

/-- The interest rate for the second part (20%) -/
def interest_rate_20 : ℚ := 1/5

/-- The total profit after one year -/
def total_profit : ℕ := 8000

/-- Theorem stating that the total amount divided is 70,000 -/
theorem total_amount_is_70000 :
  total_amount = 70000 ∧
  amount_10_percent + amount_20_percent = total_amount ∧
  amount_10_percent * interest_rate_10 + amount_20_percent * interest_rate_20 = total_profit :=
sorry

end NUMINAMATH_CALUDE_total_amount_is_70000_l3084_308499


namespace NUMINAMATH_CALUDE_quadratic_root_exists_l3084_308424

/-- A quadratic function with specific values at certain points. -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  f_neg_two : f (-2) = -1
  f_neg_one : f (-1) = 2
  f_zero : f 0 = 3
  f_one : f 1 = 2

/-- Theorem stating that the quadratic function has a root between -2 and -1. -/
theorem quadratic_root_exists (qf : QuadraticFunction) :
  ∃ r : ℝ, -2 < r ∧ r < -1 ∧ qf.f r = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_exists_l3084_308424


namespace NUMINAMATH_CALUDE_parabola_properties_l3084_308431

-- Define the parabola
def parabola (a b c x : ℝ) := a * x^2 + b * x + c

-- State the theorem
theorem parabola_properties (a b c : ℝ) 
  (h1 : parabola a b c 2 = 0) :
  (b = -2*a → parabola a b c 0 = 0) ∧ 
  (c ≠ 4*a → (b^2 - 4*a*c > 0)) ∧
  (∀ x1 x2, x1 > x2 ∧ x2 > -1 ∧ parabola a b c x1 > parabola a b c x2 → 8*a + c ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l3084_308431


namespace NUMINAMATH_CALUDE_arithmetic_progression_divisibility_l3084_308440

/-- Given three integers a, b, c that form an arithmetic progression with a common difference of 7,
    and one of them is divisible by 7, their product abc is divisible by 294. -/
theorem arithmetic_progression_divisibility (a b c : ℤ) 
  (h1 : b - a = 7)
  (h2 : c - b = 7)
  (h3 : (∃ k : ℤ, a = 7 * k) ∨ (∃ k : ℤ, b = 7 * k) ∨ (∃ k : ℤ, c = 7 * k)) :
  ∃ m : ℤ, a * b * c = 294 * m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_divisibility_l3084_308440


namespace NUMINAMATH_CALUDE_trip_cost_per_person_l3084_308432

/-- Given a group of 11 people and a total cost of $12,100 for a trip,
    the cost per person is $1,100. -/
theorem trip_cost_per_person :
  let total_people : ℕ := 11
  let total_cost : ℕ := 12100
  total_cost / total_people = 1100 := by
  sorry

end NUMINAMATH_CALUDE_trip_cost_per_person_l3084_308432


namespace NUMINAMATH_CALUDE_base_number_proof_l3084_308472

theorem base_number_proof (n : ℕ) (x : ℝ) 
  (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = x^26) 
  (h2 : n = 25) : 
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l3084_308472


namespace NUMINAMATH_CALUDE_door_cost_ratio_l3084_308420

theorem door_cost_ratio (bedroom_doors : ℕ) (outside_doors : ℕ) 
  (outside_door_cost : ℚ) (total_cost : ℚ) :
  bedroom_doors = 3 →
  outside_doors = 2 →
  outside_door_cost = 20 →
  total_cost = 70 →
  ∃ (bedroom_door_cost : ℚ),
    bedroom_doors * bedroom_door_cost + outside_doors * outside_door_cost = total_cost ∧
    bedroom_door_cost / outside_door_cost = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_door_cost_ratio_l3084_308420


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_products_l3084_308411

theorem root_sum_reciprocal_products (p q r s : ℂ) : 
  (p^4 + 6*p^3 - 4*p^2 + 7*p + 3 = 0) →
  (q^4 + 6*q^3 - 4*q^2 + 7*q + 3 = 0) →
  (r^4 + 6*r^3 - 4*r^2 + 7*r + 3 = 0) →
  (s^4 + 6*s^3 - 4*s^2 + 7*s + 3 = 0) →
  1/(p*q) + 1/(p*r) + 1/(p*s) + 1/(q*r) + 1/(q*s) + 1/(r*s) = -4/3 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_products_l3084_308411


namespace NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3084_308474

/-- Represents a configuration of square tiles -/
structure TileConfiguration where
  tiles : ℕ
  perimeter : ℕ

/-- Adds tiles to a configuration -/
def add_tiles (config : TileConfiguration) (new_tiles : ℕ) : TileConfiguration :=
  { tiles := config.tiles + new_tiles, perimeter := config.perimeter }

theorem perimeter_after_adding_tiles 
  (initial : TileConfiguration) 
  (added_tiles : ℕ) 
  (final : TileConfiguration) :
  initial.tiles = 10 →
  initial.perimeter = 16 →
  added_tiles = 4 →
  final = add_tiles initial added_tiles →
  final.perimeter = 18 :=
by
  sorry

#check perimeter_after_adding_tiles

end NUMINAMATH_CALUDE_perimeter_after_adding_tiles_l3084_308474


namespace NUMINAMATH_CALUDE_expression_equals_one_l3084_308423

theorem expression_equals_one : 
  (120^2 - 13^2) / (90^2 - 19^2) * ((90 - 19) * (90 + 19)) / ((120 - 13) * (120 + 13)) = 1 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_one_l3084_308423


namespace NUMINAMATH_CALUDE_augmented_matrix_sum_l3084_308442

/-- Given a system of linear equations represented by an augmented matrix,
    prove that the sum of certain parameters equals 3. -/
theorem augmented_matrix_sum (m n : ℝ) : 
  (∃ (x y : ℝ), 2 * x = m ∧ n * x + y = 2 ∧ x = 1 ∧ y = 1) →
  m + n = 3 := by
  sorry

end NUMINAMATH_CALUDE_augmented_matrix_sum_l3084_308442


namespace NUMINAMATH_CALUDE_alice_average_speed_l3084_308422

/-- Alice's cycling journey --/
def alice_journey : Prop :=
  let first_distance : ℝ := 240
  let first_time : ℝ := 4.5
  let second_distance : ℝ := 300
  let second_time : ℝ := 5.25
  let total_distance : ℝ := first_distance + second_distance
  let total_time : ℝ := first_time + second_time
  let average_speed : ℝ := total_distance / total_time
  average_speed = 540 / 9.75

theorem alice_average_speed : alice_journey := by
  sorry

end NUMINAMATH_CALUDE_alice_average_speed_l3084_308422


namespace NUMINAMATH_CALUDE_cosine_is_periodic_l3084_308407

-- Define a type for functions from ℝ to ℝ
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be periodic
def IsPeriodic (f : RealFunction) : Prop :=
  ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x

-- Define what it means for a function to be trigonometric
def IsTrigonometric (f : RealFunction) : Prop :=
  -- This is a placeholder definition
  True

-- State the theorem
theorem cosine_is_periodic :
  (∀ f : RealFunction, IsTrigonometric f → IsPeriodic f) →
  IsTrigonometric (λ x : ℝ => Real.cos x) →
  IsPeriodic (λ x : ℝ => Real.cos x) :=
by
  sorry

end NUMINAMATH_CALUDE_cosine_is_periodic_l3084_308407


namespace NUMINAMATH_CALUDE_bonnie_sticker_count_l3084_308437

/-- Calculates Bonnie's initial sticker count given the problem conditions -/
def bonnies_initial_stickers (june_initial : ℕ) (grandparents_gift : ℕ) (final_total : ℕ) : ℕ :=
  final_total - (june_initial + 2 * grandparents_gift)

/-- Theorem stating that Bonnie's initial sticker count is 63 given the problem conditions -/
theorem bonnie_sticker_count :
  bonnies_initial_stickers 76 25 189 = 63 := by
  sorry

#eval bonnies_initial_stickers 76 25 189

end NUMINAMATH_CALUDE_bonnie_sticker_count_l3084_308437


namespace NUMINAMATH_CALUDE_intersection_implies_sum_zero_l3084_308465

theorem intersection_implies_sum_zero (α β : ℝ) :
  (∃ x₀ : ℝ, x₀ / (Real.sin α + Real.sin β) + (-x₀) / (Real.sin α + Real.cos β) = 1 ∧
              x₀ / (Real.cos α + Real.sin β) + (-x₀) / (Real.cos α + Real.cos β) = 1) →
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_zero_l3084_308465


namespace NUMINAMATH_CALUDE_imoHost2023_l3084_308433

theorem imoHost2023 (A B C : ℕ+) (hDistinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (hProduct : A * B * C = 2023) :
  A + B + C ≤ 297 ∧ ∃ (A' B' C' : ℕ+), A' ≠ B' ∧ B' ≠ C' ∧ A' ≠ C' ∧ A' * B' * C' = 2023 ∧ A' + B' + C' = 297 := by
  sorry

end NUMINAMATH_CALUDE_imoHost2023_l3084_308433


namespace NUMINAMATH_CALUDE_sum_of_exterior_angles_is_360_l3084_308479

/-- A polygon is a closed planar figure with straight sides -/
structure Polygon where
  sides : ℕ
  sides_positive : sides > 2

/-- An exterior angle of a polygon -/
def exterior_angle (p : Polygon) : ℝ := sorry

/-- The sum of exterior angles of a polygon -/
def sum_of_exterior_angles (p : Polygon) : ℝ := sorry

/-- Theorem: The sum of the exterior angles of any polygon is 360° -/
theorem sum_of_exterior_angles_is_360 (p : Polygon) : 
  sum_of_exterior_angles p = 360 := by sorry

end NUMINAMATH_CALUDE_sum_of_exterior_angles_is_360_l3084_308479


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3084_308456

theorem complex_fraction_equality (a b : ℝ) : 
  (Complex.I + 1) / (Complex.I - 1) = Complex.mk a b → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3084_308456


namespace NUMINAMATH_CALUDE_seats_filled_percentage_l3084_308404

/-- Given a hall with total seats and vacant seats, calculate the percentage of filled seats -/
def percentage_filled (total_seats vacant_seats : ℕ) : ℚ :=
  (total_seats - vacant_seats : ℚ) / total_seats * 100

/-- Theorem: In a hall with 600 seats where 300 are vacant, 50% of the seats are filled -/
theorem seats_filled_percentage :
  percentage_filled 600 300 = 50 := by
  sorry

#eval percentage_filled 600 300

end NUMINAMATH_CALUDE_seats_filled_percentage_l3084_308404


namespace NUMINAMATH_CALUDE_ellen_painted_17_lilies_l3084_308469

/-- Time in minutes to paint each type of flower or vine -/
def lily_time : ℕ := 5
def rose_time : ℕ := 7
def orchid_time : ℕ := 3
def vine_time : ℕ := 2

/-- Total time spent painting -/
def total_time : ℕ := 213

/-- Number of roses, orchids, and vines painted -/
def roses : ℕ := 10
def orchids : ℕ := 6
def vines : ℕ := 20

/-- Function to calculate the number of lilies painted -/
def lilies_painted : ℕ := 
  (total_time - (roses * rose_time + orchids * orchid_time + vines * vine_time)) / lily_time

theorem ellen_painted_17_lilies : lilies_painted = 17 := by
  sorry

end NUMINAMATH_CALUDE_ellen_painted_17_lilies_l3084_308469


namespace NUMINAMATH_CALUDE_max_a_value_l3084_308452

theorem max_a_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ a : ℝ, m^2 - a*m*n + 2*n^2 ≥ 0 → a ≤ 2*Real.sqrt 2) ∧
  ∃ a : ℝ, a = 2*Real.sqrt 2 ∧ m^2 - a*m*n + 2*n^2 ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_max_a_value_l3084_308452


namespace NUMINAMATH_CALUDE_find_y_value_l3084_308498

theorem find_y_value (x y : ℝ) 
  (h1 : (100 + 200 + 300 + x) / 4 = 250)
  (h2 : (300 + 150 + 100 + x + y) / 5 = 200) : 
  y = 50 := by
sorry

end NUMINAMATH_CALUDE_find_y_value_l3084_308498


namespace NUMINAMATH_CALUDE_segment_length_is_twenty_l3084_308402

/-- The volume of a geometric body formed by points whose distance to a line segment
    is no greater than r units -/
noncomputable def geometricBodyVolume (r : ℝ) (segmentLength : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3 + Real.pi * r^2 * segmentLength

/-- Theorem stating that if the volume of the geometric body with radius 3
    is 216π, then the segment length is 20 -/
theorem segment_length_is_twenty (segmentLength : ℝ) :
  geometricBodyVolume 3 segmentLength = 216 * Real.pi → segmentLength = 20 := by
  sorry

#check segment_length_is_twenty

end NUMINAMATH_CALUDE_segment_length_is_twenty_l3084_308402


namespace NUMINAMATH_CALUDE_consecutive_odd_sum_48_l3084_308486

theorem consecutive_odd_sum_48 (a b : ℤ) : 
  (∃ k : ℤ, a = 2*k + 1) →  -- a is odd
  (∃ m : ℤ, b = 2*m + 1) →  -- b is odd
  b = a + 2 →               -- b is the next consecutive odd after a
  a + b = 48 →              -- sum is 48
  b = 25 :=                 -- larger number is 25
by sorry

end NUMINAMATH_CALUDE_consecutive_odd_sum_48_l3084_308486


namespace NUMINAMATH_CALUDE_system_solution_l3084_308471

theorem system_solution (a b c x y z : ℝ) 
  (eq1 : x + y = a) 
  (eq2 : y + z = b) 
  (eq3 : z + x = c) : 
  x = (a + c - b) / 2 ∧ 
  y = (a + b - c) / 2 ∧ 
  z = (b + c - a) / 2 := by
sorry


end NUMINAMATH_CALUDE_system_solution_l3084_308471


namespace NUMINAMATH_CALUDE_f_properties_l3084_308461

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + x^2 - 3*x - x / Real.exp x

theorem f_properties (h : ∀ x, x > 0 → f x = x * Real.log x + x^2 - 3*x - x / Real.exp x) :
  (∃ x₀ > 0, ∀ x > 0, f x ≥ f x₀ ∧ f x₀ = -2 - 1 / Real.exp 1) ∧
  (∀ x, Real.exp x ≥ x + 1) ∧
  (∀ x y, 0 < x ∧ x < y → (deriv f x) < (deriv f y)) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l3084_308461


namespace NUMINAMATH_CALUDE_max_value_of_operation_l3084_308418

theorem max_value_of_operation : 
  ∃ (max : ℕ), max = 1200 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → 3 * (500 - n) ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_operation_l3084_308418


namespace NUMINAMATH_CALUDE_prob_at_least_one_female_is_four_fifths_l3084_308455

/-- Represents the number of students in each category -/
structure StudentCounts where
  total : ℕ
  maleHigh : ℕ
  femaleHigh : ℕ
  selected : ℕ
  final : ℕ

/-- Calculates the probability of selecting at least one female student -/
def probAtLeastOneFemale (counts : StudentCounts) : ℚ :=
  1 - (counts.maleHigh.choose counts.final) / (counts.maleHigh + counts.femaleHigh).choose counts.final

/-- The main theorem stating the probability of selecting at least one female student -/
theorem prob_at_least_one_female_is_four_fifths (counts : StudentCounts) 
  (h1 : counts.total = 200)
  (h2 : counts.maleHigh = 100)
  (h3 : counts.femaleHigh = 50)
  (h4 : counts.selected = 6)
  (h5 : counts.final = 3) :
  probAtLeastOneFemale counts = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_prob_at_least_one_female_is_four_fifths_l3084_308455


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3084_308435

theorem quadratic_solution_sum (a b : ℝ) : 
  (∀ x : ℂ, 3 * x^2 + 4 = 5 * x - 7 ↔ x = a + b * I ∨ x = a - b * I) →
  a + b^2 = 137 / 36 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3084_308435


namespace NUMINAMATH_CALUDE_always_quadratic_l3084_308454

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation (a² + 1)x² + bx + c = 0 is always quadratic -/
theorem always_quadratic (a b c : ℝ) :
  is_quadratic_equation (λ x => (a^2 + 1) * x^2 + b * x + c) := by
  sorry

#check always_quadratic

end NUMINAMATH_CALUDE_always_quadratic_l3084_308454


namespace NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3084_308408

/-- The ellipse defined by x²/4 + y²/m = 1 -/
def is_ellipse (x y m : ℝ) : Prop :=
  x^2 / 4 + y^2 / m = 1

/-- The line y = mx + 2 -/
def is_line (x y m : ℝ) : Prop :=
  y = m * x + 2

/-- The line is tangent to the ellipse -/
def is_tangent (m : ℝ) : Prop :=
  ∃ (x y : ℝ), is_ellipse x y m ∧ is_line x y m ∧
  ∀ (x' y' : ℝ), is_ellipse x' y' m → is_line x' y' m → (x = x' ∧ y = y')

theorem tangent_line_to_ellipse (m : ℝ) :
  is_tangent m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_ellipse_l3084_308408


namespace NUMINAMATH_CALUDE_inequality_solution_l3084_308490

theorem inequality_solution (x : ℝ) : (x^2 - 1) / ((x + 2)^2) ≥ 0 ↔ 
  x < -2 ∨ (-2 < x ∧ x ≤ -1) ∨ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l3084_308490


namespace NUMINAMATH_CALUDE_rational_function_value_l3084_308453

/-- A rational function with specific properties -/
def rational_function (p q : ℝ → ℝ) : Prop :=
  (∃ k : ℝ, ∀ x, p x = k * x) ∧  -- p is linear
  (∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c) ∧  -- q is quadratic
  p 0 / q 0 = 0 ∧  -- passes through (0,0)
  p 4 / q 4 = 2 ∧  -- passes through (4,2)
  q (-4) = 0 ∧  -- vertical asymptote at x = -4
  q 1 = 0  -- vertical asymptote at x = 1

theorem rational_function_value (p q : ℝ → ℝ) :
  rational_function p q → p (-1) / q (-1) = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_l3084_308453


namespace NUMINAMATH_CALUDE_fair_attendance_l3084_308488

/-- Given the number of people attending a fair over three years, prove the values of x, y, and z. -/
theorem fair_attendance (x y z : ℕ) 
  (h1 : z = 2 * y)
  (h2 : x = z - 200)
  (h3 : y = 600) :
  x = 1000 ∧ y = 600 ∧ z = 1200 := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_l3084_308488


namespace NUMINAMATH_CALUDE_equation_solution_l3084_308439

theorem equation_solution (x : ℝ) (h : x ≥ 0) :
  (2021 * x = 2022 * (x^(2021/2022)) - 1) ↔ (x = 1) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l3084_308439


namespace NUMINAMATH_CALUDE_students_in_davids_grade_l3084_308476

/-- Given that David is both the 75th best and 75th worst student in his grade,
    prove that there are 149 students in total. -/
theorem students_in_davids_grade (n : ℕ) 
  (h1 : n ≥ 75)  -- David's grade has at least 75 students
  (h2 : ∃ (better worse : ℕ), better = 74 ∧ worse = 74 ∧ n = better + worse + 1) 
  : n = 149 := by
  sorry

end NUMINAMATH_CALUDE_students_in_davids_grade_l3084_308476


namespace NUMINAMATH_CALUDE_area_of_enclosed_region_l3084_308450

/-- The curve defined by |x-1| + |y-1| = 1 -/
def curve (x y : ℝ) : Prop := |x - 1| + |y - 1| = 1

/-- The region enclosed by the curve -/
def enclosed_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | curve p.1 p.2}

/-- The area of the region enclosed by the curve is 2 -/
theorem area_of_enclosed_region : MeasureTheory.volume enclosed_region = 2 := by
  sorry

end NUMINAMATH_CALUDE_area_of_enclosed_region_l3084_308450


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3084_308406

theorem quadratic_equation_roots_range (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   k * x₁^2 - Real.sqrt (2 * k + 1) * x₁ + 1 = 0 ∧
   k * x₂^2 - Real.sqrt (2 * k + 1) * x₂ + 1 = 0) ∧
  k ≠ 0 →
  -1/2 ≤ k ∧ k < 1/2 ∧ k ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_range_l3084_308406


namespace NUMINAMATH_CALUDE_range_of_expression_l3084_308475

theorem range_of_expression (x y : ℝ) (h : x^2 - y^2 = 4) :
  ∃ (z : ℝ), z = (1/x^2) - (y/x) ∧ -1 ≤ z ∧ z ≤ 5/4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_expression_l3084_308475


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3084_308401

/-- The volume of a cylinder minus the volume of two congruent cones --/
theorem cylinder_minus_cones_volume 
  (r : ℝ) -- radius of cylinder and cones
  (h_cylinder : ℝ) -- height of cylinder
  (h_cone : ℝ) -- height of each cone
  (h_cylinder_eq : h_cylinder = 2 * h_cone) -- cylinder height is twice the cone height
  (r_eq : r = 10) -- radius is 10 cm
  (h_cone_eq : h_cone = 15) -- cone height is 15 cm
  : π * r^2 * h_cylinder - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3084_308401


namespace NUMINAMATH_CALUDE_better_hay_cost_is_18_l3084_308436

/-- The cost of better quality hay per bale -/
def better_hay_cost (initial_bales : ℕ) (price_increase : ℕ) (previous_cost : ℕ) : ℕ :=
  (initial_bales * previous_cost + price_increase) / (2 * initial_bales)

/-- Proof that the cost of better quality hay is $18 per bale -/
theorem better_hay_cost_is_18 :
  better_hay_cost 10 210 15 = 18 := by
  sorry

end NUMINAMATH_CALUDE_better_hay_cost_is_18_l3084_308436


namespace NUMINAMATH_CALUDE_hexagon_coloring_ways_l3084_308494

-- Define the colors
inductive Color
| Red
| Yellow
| Green

-- Define the hexagon grid
def HexagonGrid := List (List Color)

-- Define a function to check if two colors are different
def different_colors (c1 c2 : Color) : Prop :=
  c1 ≠ c2

-- Define a function to check if a coloring is valid
def valid_coloring (grid : HexagonGrid) : Prop :=
  -- Add conditions for valid coloring here
  sorry

-- Define the specific grid pattern with 8 hexagons
def specific_grid_pattern (grid : HexagonGrid) : Prop :=
  -- Add conditions for the specific grid pattern here
  sorry

-- Define the initial conditions (top-left and second-top hexagons are red)
def initial_conditions (grid : HexagonGrid) : Prop :=
  -- Add conditions for initial red hexagons here
  sorry

-- Theorem statement
theorem hexagon_coloring_ways :
  ∀ (grid : HexagonGrid),
    specific_grid_pattern grid →
    initial_conditions grid →
    valid_coloring grid →
    ∃! (n : Nat), n = 2 ∧ 
      ∃ (colorings : List HexagonGrid),
        colorings.length = n ∧
        ∀ c ∈ colorings, 
          specific_grid_pattern c ∧
          initial_conditions c ∧
          valid_coloring c :=
sorry

end NUMINAMATH_CALUDE_hexagon_coloring_ways_l3084_308494


namespace NUMINAMATH_CALUDE_elevator_trips_l3084_308429

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def capacity : ℕ := 190

def is_valid_trip (trip : List ℕ) : Bool :=
  trip.sum ≤ capacity

def min_trips (masses : List ℕ) (capacity : ℕ) : ℕ :=
  sorry

theorem elevator_trips :
  min_trips masses capacity = 6 := by
  sorry

end NUMINAMATH_CALUDE_elevator_trips_l3084_308429


namespace NUMINAMATH_CALUDE_product_is_2008th_power_l3084_308493

theorem product_is_2008th_power : ∃ (a b c n : ℕ), 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  b = (a + c) / 2 ∧
  a * b * c = n^2008 := by
sorry

end NUMINAMATH_CALUDE_product_is_2008th_power_l3084_308493


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3084_308446

/-- An arithmetic sequence with specific conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 2 = 5 ∧
  a 3 + a 5 = 2

/-- The common difference of an arithmetic sequence with given conditions is -2 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) (h : ArithmeticSequence a) : 
  ∃ d : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ d = -2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3084_308446


namespace NUMINAMATH_CALUDE_distinct_points_on_curve_l3084_308460

theorem distinct_points_on_curve (a b : ℝ) : 
  a ≠ b →
  (a^2 + Real.sqrt π^4 = 2 * (Real.sqrt π)^2 * a + 1) →
  (b^2 + Real.sqrt π^4 = 2 * (Real.sqrt π)^2 * b + 1) →
  |a - b| = 2 := by sorry

end NUMINAMATH_CALUDE_distinct_points_on_curve_l3084_308460


namespace NUMINAMATH_CALUDE_combined_tennis_percentage_l3084_308451

-- Define the given conditions
def north_students : ℕ := 1800
def north_tennis_percentage : ℚ := 25 / 100
def south_students : ℕ := 2700
def south_tennis_percentage : ℚ := 35 / 100

-- Define the theorem
theorem combined_tennis_percentage :
  let north_tennis := (north_students : ℚ) * north_tennis_percentage
  let south_tennis := (south_students : ℚ) * south_tennis_percentage
  let total_tennis := north_tennis + south_tennis
  let total_students := (north_students + south_students : ℚ)
  (total_tennis / total_students) * 100 = 31 := by
  sorry

end NUMINAMATH_CALUDE_combined_tennis_percentage_l3084_308451


namespace NUMINAMATH_CALUDE_expression_evaluation_l3084_308463

theorem expression_evaluation (x y : ℝ) (hx : x = 2) (hy : y = -1) :
  2*x*y - 1/2*(4*x*y - 8*x^2*y^2) + 2*(3*x*y - 5*x^2*y^2) = -36 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3084_308463


namespace NUMINAMATH_CALUDE_temperature_drop_l3084_308419

theorem temperature_drop (initial_temp final_temp drop : ℤ) :
  initial_temp = -6 ∧ drop = 5 → final_temp = initial_temp - drop → final_temp = -11 := by
  sorry

end NUMINAMATH_CALUDE_temperature_drop_l3084_308419


namespace NUMINAMATH_CALUDE_movie_theater_open_hours_l3084_308485

/-- A movie theater with multiple screens showing movies throughout the day. -/
structure MovieTheater where
  screens : ℕ
  total_movies : ℕ
  movie_duration : ℕ

/-- Calculate the number of hours a movie theater is open. -/
def theater_open_hours (theater : MovieTheater) : ℕ :=
  (theater.total_movies * theater.movie_duration) / theater.screens

/-- Theorem: A movie theater with 6 screens showing 24 movies, each lasting 2 hours, is open for 8 hours. -/
theorem movie_theater_open_hours :
  let theater := MovieTheater.mk 6 24 2
  theater_open_hours theater = 8 := by
  sorry

end NUMINAMATH_CALUDE_movie_theater_open_hours_l3084_308485


namespace NUMINAMATH_CALUDE_average_monthly_balance_l3084_308427

def monthly_balances : List ℝ := [120, 240, 180, 180, 160, 200]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 180 := by
  sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l3084_308427


namespace NUMINAMATH_CALUDE_rug_coverage_l3084_308468

theorem rug_coverage (rug_length : ℝ) (rug_width : ℝ) (floor_area : ℝ) 
  (h1 : rug_length = 2)
  (h2 : rug_width = 7)
  (h3 : floor_area = 64)
  (h4 : rug_length * rug_width ≤ floor_area) : 
  (floor_area - rug_length * rug_width) / floor_area = 25 / 32 := by
  sorry

end NUMINAMATH_CALUDE_rug_coverage_l3084_308468


namespace NUMINAMATH_CALUDE_crayon_count_l3084_308447

theorem crayon_count (blue : ℕ) (red : ℕ) (green : ℕ) : 
  blue = 3 → 
  red = 4 * blue → 
  green = 2 * red → 
  blue + red + green = 39 := by
sorry

end NUMINAMATH_CALUDE_crayon_count_l3084_308447


namespace NUMINAMATH_CALUDE_bracelets_given_to_school_is_three_l3084_308414

/-- The number of bracelets Chantel gave away to her friends at school -/
def bracelets_given_to_school : ℕ :=
  let days_first_period := 5
  let bracelets_per_day_first_period := 2
  let days_second_period := 4
  let bracelets_per_day_second_period := 3
  let bracelets_given_at_soccer := 6
  let bracelets_remaining := 13
  let total_bracelets_made := days_first_period * bracelets_per_day_first_period + 
                              days_second_period * bracelets_per_day_second_period
  let bracelets_after_soccer := total_bracelets_made - bracelets_given_at_soccer
  bracelets_after_soccer - bracelets_remaining

theorem bracelets_given_to_school_is_three : 
  bracelets_given_to_school = 3 := by sorry

end NUMINAMATH_CALUDE_bracelets_given_to_school_is_three_l3084_308414


namespace NUMINAMATH_CALUDE_ribbon_per_gift_l3084_308449

theorem ribbon_per_gift (total_gifts : ℕ) (total_ribbon : ℝ) (remaining_ribbon : ℝ) 
  (h1 : total_gifts = 8)
  (h2 : total_ribbon = 15)
  (h3 : remaining_ribbon = 3) :
  (total_ribbon - remaining_ribbon) / total_gifts = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ribbon_per_gift_l3084_308449


namespace NUMINAMATH_CALUDE_maximize_product_l3084_308415

theorem maximize_product (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 40) :
  x^3 * y^4 ≤ (160/7)^3 * (120/7)^4 ∧
  (x^3 * y^4 = (160/7)^3 * (120/7)^4 ↔ x = 160/7 ∧ y = 120/7) :=
by sorry

end NUMINAMATH_CALUDE_maximize_product_l3084_308415


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l3084_308441

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∃ x : ℝ, x = 1 → x^2 = 1) ∧
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l3084_308441


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3084_308459

/-- Two arithmetic sequences and their properties -/
structure ArithmeticSequences where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum of first n terms of sequence a
  T : ℕ → ℚ  -- Sum of first n terms of sequence b
  h_sum_ratio : ∀ n : ℕ, S n / T n = 3 * n / (2 * n + 1)

/-- The main theorem -/
theorem arithmetic_sequences_ratio 
  (seq : ArithmeticSequences) : 
  (seq.a 1 + seq.a 2 + seq.a 14 + seq.a 19) / 
  (seq.b 1 + seq.b 3 + seq.b 17 + seq.b 19) = 17 / 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l3084_308459


namespace NUMINAMATH_CALUDE_fundamental_theorem_of_algebra_l3084_308478

-- Define a polynomial with complex coefficients
def ComplexPolynomial := ℂ → ℂ

-- State the fundamental theorem of algebra
theorem fundamental_theorem_of_algebra :
  ∀ (P : ComplexPolynomial), ∃ (z : ℂ), P z = 0 :=
sorry

end NUMINAMATH_CALUDE_fundamental_theorem_of_algebra_l3084_308478


namespace NUMINAMATH_CALUDE_no_integer_roots_l3084_308403

theorem no_integer_roots (a b c : ℤ) (ha : a ≠ 0) (ha_even : Even a) (hb_even : Even b) (hc_odd : Odd c) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l3084_308403


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l3084_308443

/-- The proposition "If m > 0, then the equation x^2 + x - m = 0 has real roots" -/
def original_proposition (m : ℝ) : Prop :=
  m > 0 → ∃ x : ℝ, x^2 + x - m = 0

/-- The contrapositive of the original proposition -/
def contrapositive (m : ℝ) : Prop :=
  (¬∃ x : ℝ, x^2 + x - m = 0) → m ≤ 0

/-- Theorem stating that the contrapositive is equivalent to the expected form -/
theorem contrapositive_equivalence :
  ∀ m : ℝ, contrapositive m ↔ (¬∃ x : ℝ, x^2 + x - m = 0 → m ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l3084_308443


namespace NUMINAMATH_CALUDE_white_bread_count_l3084_308448

/-- The number of loaves of white bread bought each week -/
def white_bread_loaves : ℕ := sorry

/-- The cost of a loaf of white bread -/
def white_bread_cost : ℚ := 7/2

/-- The cost of a baguette -/
def baguette_cost : ℚ := 3/2

/-- The number of sourdough loaves bought each week -/
def sourdough_loaves : ℕ := 2

/-- The cost of a loaf of sourdough bread -/
def sourdough_cost : ℚ := 9/2

/-- The cost of an almond croissant -/
def croissant_cost : ℚ := 2

/-- The number of weeks -/
def weeks : ℕ := 4

/-- The total amount spent over all weeks -/
def total_spent : ℚ := 78

theorem white_bread_count :
  white_bread_loaves = 2 ∧
  (white_bread_loaves : ℚ) * white_bread_cost +
  baguette_cost +
  (sourdough_loaves : ℚ) * sourdough_cost +
  croissant_cost =
  total_spent / (weeks : ℚ) :=
sorry

end NUMINAMATH_CALUDE_white_bread_count_l3084_308448


namespace NUMINAMATH_CALUDE_marias_savings_l3084_308497

/-- Represents the cost of the bike in dollars -/
def bike_cost : ℕ := 600

/-- Represents the amount Maria's mother offered in dollars -/
def mother_offer : ℕ := 250

/-- Represents the amount Maria needs to earn in dollars -/
def amount_to_earn : ℕ := 230

/-- Represents Maria's initial savings in dollars -/
def initial_savings : ℕ := 120

theorem marias_savings :
  initial_savings + mother_offer + amount_to_earn = bike_cost :=
sorry

end NUMINAMATH_CALUDE_marias_savings_l3084_308497


namespace NUMINAMATH_CALUDE_variance_of_transformed_data_l3084_308413

-- Define a type for our dataset
def Dataset := List ℝ

-- Define the variance of a dataset
noncomputable def variance (X : Dataset) : ℝ := sorry

-- Define the transformation function
def transform (X : Dataset) : Dataset := X.map (λ x => 2 * x - 5)

-- Theorem statement
theorem variance_of_transformed_data (X : Dataset) :
  variance X = 1/2 → variance (transform X) = 2 := by sorry

end NUMINAMATH_CALUDE_variance_of_transformed_data_l3084_308413


namespace NUMINAMATH_CALUDE_buffy_whiskers_l3084_308473

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  juniper : ℕ

/-- Theorem stating the number of whiskers Buffy has -/
theorem buffy_whiskers (c : CatWhiskers) : 
  c.juniper = 12 →
  c.puffy = 3 * c.juniper →
  c.scruffy = 2 * c.puffy →
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3 →
  c.buffy = 40 := by
  sorry

#check buffy_whiskers

end NUMINAMATH_CALUDE_buffy_whiskers_l3084_308473


namespace NUMINAMATH_CALUDE_exam_average_theorem_l3084_308458

def average_percentage (group1_count : ℕ) (group1_avg : ℚ) (group2_count : ℕ) (group2_avg : ℚ) : ℚ :=
  let total_count : ℕ := group1_count + group2_count
  let total_points : ℚ := group1_count * group1_avg + group2_count * group2_avg
  total_points / total_count

theorem exam_average_theorem :
  average_percentage 15 (80/100) 10 (90/100) = 84/100 := by
  sorry

end NUMINAMATH_CALUDE_exam_average_theorem_l3084_308458


namespace NUMINAMATH_CALUDE_carnation_percentage_l3084_308477

/-- Represents the number of each type of flower in the shop -/
structure FlowerShop where
  carnations : ℝ
  violets : ℝ
  tulips : ℝ
  roses : ℝ

/-- Conditions for the flower shop inventory -/
def validFlowerShop (shop : FlowerShop) : Prop :=
  shop.violets = shop.carnations / 3 ∧
  shop.tulips = shop.violets / 3 ∧
  shop.roses = shop.tulips

/-- Theorem stating the percentage of carnations in the flower shop -/
theorem carnation_percentage (shop : FlowerShop) 
  (h : validFlowerShop shop) (h_pos : shop.carnations > 0) : 
  shop.carnations / (shop.carnations + shop.violets + shop.tulips + shop.roses) = 9 / 14 := by
  sorry

end NUMINAMATH_CALUDE_carnation_percentage_l3084_308477


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3084_308457

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), (4 * π * r^2 = 144 * π) → ((4/3) * π * r^3 = 288 * π) :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l3084_308457


namespace NUMINAMATH_CALUDE_toms_age_ratio_l3084_308495

theorem toms_age_ratio (T N : ℕ) : T > 0 → N > 0 → T = 7 * N := by
  sorry

#check toms_age_ratio

end NUMINAMATH_CALUDE_toms_age_ratio_l3084_308495


namespace NUMINAMATH_CALUDE_compound_weight_l3084_308482

/-- Given a compound with a molecular weight of 1050, 
    the total weight of 6 moles of this compound is 6300 grams. -/
theorem compound_weight (molecular_weight : ℝ) (moles : ℝ) : 
  molecular_weight = 1050 → moles = 6 → moles * molecular_weight = 6300 := by
  sorry

end NUMINAMATH_CALUDE_compound_weight_l3084_308482


namespace NUMINAMATH_CALUDE_rhombus_area_in_square_l3084_308405

/-- The area of a rhombus formed by connecting the midpoints of a square -/
theorem rhombus_area_in_square (side_length : ℝ) (h : side_length = 10) :
  let square_diagonal := side_length * Real.sqrt 2
  let rhombus_side := square_diagonal / 2
  let rhombus_area := (rhombus_side * rhombus_side) / 2
  rhombus_area = 25 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_in_square_l3084_308405


namespace NUMINAMATH_CALUDE_teds_age_l3084_308480

/-- Given that Ted's age is 10 years less than three times Sally's age,
    and the sum of their ages is 65, prove that Ted is 46 years old. -/
theorem teds_age (t s : ℕ) 
  (h1 : t = 3 * s - 10)
  (h2 : t + s = 65) : 
  t = 46 := by
  sorry

end NUMINAMATH_CALUDE_teds_age_l3084_308480


namespace NUMINAMATH_CALUDE_not_all_linear_functions_increasing_l3084_308484

/-- A linear function from ℝ to ℝ -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

/-- A function is increasing on ℝ -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- Theorem: Not all linear functions with non-zero slope are increasing on ℝ -/
theorem not_all_linear_functions_increasing :
  ¬(∀ k b : ℝ, k ≠ 0 → IsIncreasing (LinearFunction k b)) := by sorry

end NUMINAMATH_CALUDE_not_all_linear_functions_increasing_l3084_308484


namespace NUMINAMATH_CALUDE_max_square_plots_l3084_308470

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available fencing -/
def availableFencing : ℕ := 2500

/-- Calculates the number of square plots given the side length -/
def numPlots (fd : FieldDimensions) (sideLength : ℕ) : ℕ :=
  (fd.width / sideLength) * (fd.length / sideLength)

/-- Calculates the required internal fencing given the side length -/
def requiredFencing (fd : FieldDimensions) (sideLength : ℕ) : ℕ :=
  fd.width * ((fd.length / sideLength) - 1) + fd.length * ((fd.width / sideLength) - 1)

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (fd : FieldDimensions) 
    (h1 : fd.width = 30) 
    (h2 : fd.length = 60) : 
    ∃ (sideLength : ℕ), 
      sideLength > 0 ∧ 
      fd.width % sideLength = 0 ∧ 
      fd.length % sideLength = 0 ∧
      requiredFencing fd sideLength ≤ availableFencing ∧
      ∀ (s : ℕ), s > 0 → 
        fd.width % s = 0 → 
        fd.length % s = 0 → 
        requiredFencing fd s ≤ availableFencing → 
        numPlots fd s ≤ numPlots fd sideLength :=
  sorry

#eval numPlots ⟨30, 60⟩ 5  -- Should evaluate to 72

end NUMINAMATH_CALUDE_max_square_plots_l3084_308470


namespace NUMINAMATH_CALUDE_circle_line_problem_l3084_308426

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 15 = 0

-- Define the line l
def l (x y k : ℝ) : Prop := y = k*x - 2

-- Define tangency condition
def is_tangent (k : ℝ) : Prop := ∃ x y : ℝ, C x y ∧ l x y k

-- Define the condition for a point on l to be within distance 2 from the center of C
def point_within_distance (k : ℝ) : Prop := 
  ∃ x y : ℝ, l x y k ∧ (x - 4)^2 + y^2 ≤ 4

theorem circle_line_problem (k : ℝ) :
  (is_tangent k → k = (8 + Real.sqrt 19) / 15 ∨ k = (8 - Real.sqrt 19) / 15) ∧
  (point_within_distance k → 0 ≤ k ∧ k ≤ 4/3) :=
sorry

end NUMINAMATH_CALUDE_circle_line_problem_l3084_308426


namespace NUMINAMATH_CALUDE_blue_then_red_probability_l3084_308491

/-- The probability of drawing a blue ball first and a red ball second from a box 
    containing 15 balls (5 blue and 10 red) without replacement is 5/21. -/
theorem blue_then_red_probability (total : ℕ) (blue : ℕ) (red : ℕ) :
  total = 15 → blue = 5 → red = 10 →
  (blue : ℚ) / total * red / (total - 1) = 5 / 21 := by
  sorry

end NUMINAMATH_CALUDE_blue_then_red_probability_l3084_308491


namespace NUMINAMATH_CALUDE_chicken_rabbit_problem_l3084_308445

theorem chicken_rabbit_problem (c r : ℕ) : 
  c = r - 20 → 
  4 * r = 3 * (2 * c) + 10 → 
  c = 35 :=
by sorry

end NUMINAMATH_CALUDE_chicken_rabbit_problem_l3084_308445


namespace NUMINAMATH_CALUDE_total_time_conversion_l3084_308421

/-- Given 3450 minutes and 7523 seconds, prove that the total time is 59 hours, 35 minutes, and 23 seconds. -/
theorem total_time_conversion (minutes : ℕ) (seconds : ℕ) : 
  minutes = 3450 ∧ seconds = 7523 → 
  ∃ (hours : ℕ) (remaining_minutes : ℕ) (remaining_seconds : ℕ),
    hours = 59 ∧ 
    remaining_minutes = 35 ∧ 
    remaining_seconds = 23 ∧
    minutes * 60 + seconds = hours * 3600 + remaining_minutes * 60 + remaining_seconds :=
by sorry

end NUMINAMATH_CALUDE_total_time_conversion_l3084_308421


namespace NUMINAMATH_CALUDE_perimeter_of_PQRSU_l3084_308481

-- Define the points as 2D vectors
def P : ℝ × ℝ := (0, 8)
def Q : ℝ × ℝ := (4, 8)
def R : ℝ × ℝ := (4, 4)
def S : ℝ × ℝ := (9, 0)
def U : ℝ × ℝ := (0, 0)

-- Define the conditions
def PQ_length : ℝ := 4
def PU_length : ℝ := 8
def US_length : ℝ := 9

-- Define the right angles
def angle_PUQ_is_right : (P.1 - U.1) * (Q.1 - U.1) + (P.2 - U.2) * (Q.2 - U.2) = 0 := by sorry
def angle_UPQ_is_right : (U.1 - P.1) * (Q.1 - P.1) + (U.2 - P.2) * (Q.2 - P.2) = 0 := by sorry
def angle_PQR_is_right : (P.1 - Q.1) * (R.1 - Q.1) + (P.2 - Q.2) * (R.2 - Q.2) = 0 := by sorry

-- Define the theorem
theorem perimeter_of_PQRSU : 
  let perimeter := PQ_length + 
                   Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) + 
                   Real.sqrt ((S.1 - R.1)^2 + (S.2 - R.2)^2) + 
                   US_length + 
                   PU_length
  perimeter = 25 + Real.sqrt 41 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_PQRSU_l3084_308481


namespace NUMINAMATH_CALUDE_smallest_winning_number_l3084_308416

theorem smallest_winning_number :
  ∃ N : ℕ,
    N ≤ 499 ∧
    (∀ m : ℕ, m < N →
      (3 * m < 500 ∧
       3 * m + 25 < 500 ∧
       3 * (3 * m + 25) < 500 ∧
       3 * (3 * m + 25) + 25 ≥ 500 →
       False)) ∧
    3 * N < 500 ∧
    3 * N + 25 < 500 ∧
    3 * (3 * N + 25) < 500 ∧
    3 * (3 * N + 25) + 25 ≥ 500 ∧
    N = 45 :=
by sorry

#eval (45 / 10 + 45 % 10) -- Sum of digits of 45

end NUMINAMATH_CALUDE_smallest_winning_number_l3084_308416


namespace NUMINAMATH_CALUDE_carries_babysitting_earnings_l3084_308430

/-- Carrie's babysitting earnings problem -/
theorem carries_babysitting_earnings 
  (iphone_cost : ℕ) 
  (trade_in_value : ℕ) 
  (weeks_to_work : ℕ) 
  (h1 : iphone_cost = 800)
  (h2 : trade_in_value = 240)
  (h3 : weeks_to_work = 7) :
  (iphone_cost - trade_in_value) / weeks_to_work = 80 :=
by sorry

end NUMINAMATH_CALUDE_carries_babysitting_earnings_l3084_308430


namespace NUMINAMATH_CALUDE_min_value_n_plus_32_over_n_squared_l3084_308417

theorem min_value_n_plus_32_over_n_squared (n : ℝ) (h : n > 0) :
  n + 32 / n^2 ≥ 6 ∧ ∃ n₀ > 0, n₀ + 32 / n₀^2 = 6 := by sorry

end NUMINAMATH_CALUDE_min_value_n_plus_32_over_n_squared_l3084_308417
