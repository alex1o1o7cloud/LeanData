import Mathlib

namespace NUMINAMATH_CALUDE_apple_distribution_l3234_323456

/-- Represents the number of apples Karen has at the end -/
def karens_final_apples (initial_apples : ℕ) : ℕ :=
  let after_first_transfer := initial_apples - 12
  (after_first_transfer - after_first_transfer / 2)

/-- Represents the number of apples Alphonso has at the end -/
def alphonsos_final_apples (initial_apples : ℕ) : ℕ :=
  let after_first_transfer := initial_apples + 12
  let karens_remaining := initial_apples - 12
  (after_first_transfer + karens_remaining / 2)

theorem apple_distribution (initial_apples : ℕ) 
  (h1 : initial_apples ≥ 12)
  (h2 : alphonsos_final_apples initial_apples = 4 * karens_final_apples initial_apples) :
  karens_final_apples initial_apples = 24 := by
  sorry

end NUMINAMATH_CALUDE_apple_distribution_l3234_323456


namespace NUMINAMATH_CALUDE_time_to_destination_l3234_323442

-- Define the walking speeds and distances
def your_speed : ℝ := 2
def harris_speed : ℝ := 1
def harris_time : ℝ := 2
def distance_ratio : ℝ := 3

-- Theorem statement
theorem time_to_destination : 
  your_speed * harris_speed = 2 → 
  (your_speed * (distance_ratio * harris_time)) / your_speed = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_time_to_destination_l3234_323442


namespace NUMINAMATH_CALUDE_factories_unchecked_l3234_323454

/-- The number of unchecked factories given the total number of factories and the number checked by two groups -/
def unchecked_factories (total : ℕ) (group1 : ℕ) (group2 : ℕ) : ℕ :=
  total - (group1 + group2)

/-- Theorem stating that 67 factories remain unchecked -/
theorem factories_unchecked :
  unchecked_factories 259 105 87 = 67 := by
  sorry

end NUMINAMATH_CALUDE_factories_unchecked_l3234_323454


namespace NUMINAMATH_CALUDE_tan_sin_product_l3234_323451

theorem tan_sin_product (A B : Real) (hA : A = 10 * Real.pi / 180) (hB : B = 35 * Real.pi / 180) :
  (1 + Real.tan A) * (1 + Real.sin B) = 
    1 + Real.tan A + (Real.sqrt 2 / 2) * (Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) + 
    Real.tan A * (Real.sqrt 2 / 2) * (Real.cos (10 * Real.pi / 180) - Real.sin (10 * Real.pi / 180)) := by
  sorry

end NUMINAMATH_CALUDE_tan_sin_product_l3234_323451


namespace NUMINAMATH_CALUDE_f_at_negative_two_l3234_323402

-- Define the function f
def f (x : ℝ) : ℝ := 8 * x^3 - 6 * x^2 - 4 * x + 5

-- Theorem statement
theorem f_at_negative_two : f (-2) = -75 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_two_l3234_323402


namespace NUMINAMATH_CALUDE_total_pencils_l3234_323406

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of people who bought the color box -/
def num_people : ℕ := 3

/-- Theorem: The total number of pencils Serenity and her two friends have -/
theorem total_pencils : rainbow_colors * num_people = 21 := by
  sorry

end NUMINAMATH_CALUDE_total_pencils_l3234_323406


namespace NUMINAMATH_CALUDE_smallest_sticker_collection_l3234_323486

theorem smallest_sticker_collection (S : ℕ) : 
  S > 1 ∧ 
  S % 5 = 2 ∧ 
  S % 9 = 2 ∧ 
  S % 11 = 2 → 
  S ≥ 497 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sticker_collection_l3234_323486


namespace NUMINAMATH_CALUDE_present_worth_calculation_l3234_323492

/-- Calculates the present worth of a sum given the banker's gain, time period, and interest rate -/
def present_worth (bankers_gain : ℚ) (time : ℚ) (rate : ℚ) : ℚ :=
  let simple_interest := (bankers_gain * rate * (100 + rate * time)).sqrt
  (simple_interest * 100) / (rate * time)

/-- Theorem stating that the present worth is 7755 given the specified conditions -/
theorem present_worth_calculation :
  present_worth 24 2 (10/100) = 7755 := by
  sorry

end NUMINAMATH_CALUDE_present_worth_calculation_l3234_323492


namespace NUMINAMATH_CALUDE_max_switches_student_circle_l3234_323458

/-- 
Given n students with distinct heights arranged in a circle, 
where switches are allowed between a student and the one directly 
in front if the height difference is at least 2, the maximum number 
of possible switches before reaching a stable arrangement is ⁿC₃.
-/
theorem max_switches_student_circle (n : ℕ) : 
  ∃ (heights : Fin n → ℕ) (is_switch : Fin n → Fin n → Bool),
  (∀ i j, i ≠ j → heights i ≠ heights j) →
  (∀ i j, is_switch i j = true ↔ heights i > heights j + 1) →
  (∃ (switches : List (Fin n × Fin n)),
    (∀ (s : Fin n × Fin n), s ∈ switches → is_switch s.1 s.2 = true) ∧
    (∀ i j, is_switch i j = false) ∧
    switches.length = Nat.choose n 3) :=
by sorry

end NUMINAMATH_CALUDE_max_switches_student_circle_l3234_323458


namespace NUMINAMATH_CALUDE_inverse_contrapositive_equivalence_l3234_323468

theorem inverse_contrapositive_equivalence (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔ (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_inverse_contrapositive_equivalence_l3234_323468


namespace NUMINAMATH_CALUDE_boys_camp_total_l3234_323496

theorem boys_camp_total (total : ℕ) 
  (h1 : (total : ℚ) * (1/5) = (total : ℚ) * (20/100))
  (h2 : (total : ℚ) * (1/5) * (3/10) = (total : ℚ) * (1/5) * (30/100))
  (h3 : (total : ℚ) * (1/5) * (7/10) = 28) : 
  total = 200 := by
sorry

end NUMINAMATH_CALUDE_boys_camp_total_l3234_323496


namespace NUMINAMATH_CALUDE_min_value_inequality_l3234_323418

theorem min_value_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1/x + 4/y) ≥ 9 ∧
  ((x + y) * (1/x + 4/y) = 9 ↔ y/x = 4*x/y) :=
sorry

end NUMINAMATH_CALUDE_min_value_inequality_l3234_323418


namespace NUMINAMATH_CALUDE_last_boat_occupancy_l3234_323487

/-- The number of tourists in the travel group -/
def total_tourists (x : ℕ) : ℕ := 8 * x + 6

/-- The number of people that can be seated in (x-2) fully occupied 12-seat boats -/
def seated_tourists (x : ℕ) : ℕ := 12 * (x - 2)

theorem last_boat_occupancy (x : ℕ) (h : x > 2) :
  total_tourists x - seated_tourists x = 30 - 4 * x :=
by sorry

end NUMINAMATH_CALUDE_last_boat_occupancy_l3234_323487


namespace NUMINAMATH_CALUDE_inequality_implication_l3234_323460

theorem inequality_implication (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l3234_323460


namespace NUMINAMATH_CALUDE_blue_marble_difference_is_twenty_l3234_323450

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := 44

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := 24

/-- The difference in blue marbles between Jason and Tom -/
def blue_marble_difference : ℕ := jason_blue_marbles - tom_blue_marbles

theorem blue_marble_difference_is_twenty : blue_marble_difference = 20 := by
  sorry

end NUMINAMATH_CALUDE_blue_marble_difference_is_twenty_l3234_323450


namespace NUMINAMATH_CALUDE_homework_problems_left_l3234_323441

theorem homework_problems_left (math_problems science_problems finished_problems : ℕ) 
  (h1 : math_problems = 46)
  (h2 : science_problems = 9)
  (h3 : finished_problems = 40) : 
  math_problems + science_problems - finished_problems = 15 := by
sorry

end NUMINAMATH_CALUDE_homework_problems_left_l3234_323441


namespace NUMINAMATH_CALUDE_ratio_inequality_not_always_true_l3234_323469

theorem ratio_inequality_not_always_true :
  ¬ (∀ (a b c d : ℝ), (a / b = c / d) → (a > b → c > d)) := by
  sorry

end NUMINAMATH_CALUDE_ratio_inequality_not_always_true_l3234_323469


namespace NUMINAMATH_CALUDE_solve_for_a_l3234_323445

theorem solve_for_a (a b c d : ℝ) 
  (eq1 : a + b = d) 
  (eq2 : b + c = 6) 
  (eq3 : c + d = 7) : 
  a = 1 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l3234_323445


namespace NUMINAMATH_CALUDE_lake_distance_difference_l3234_323415

/-- The difference between the circumference of a circle with diameter 2 miles
    and its diameter, given π = 3.14 -/
theorem lake_distance_difference : 
  let π : ℝ := 3.14
  let diameter : ℝ := 2
  let circumference := π * diameter
  circumference - diameter = 4.28 := by sorry

end NUMINAMATH_CALUDE_lake_distance_difference_l3234_323415


namespace NUMINAMATH_CALUDE_carol_ate_twelve_cakes_l3234_323435

/-- The number of cakes Sara bakes per day -/
def cakes_per_day : ℕ := 10

/-- The number of days Sara bakes cakes -/
def baking_days : ℕ := 5

/-- The number of cans of frosting needed to frost a single cake -/
def cans_per_cake : ℕ := 2

/-- The number of cans of frosting Sara needs for the remaining cakes -/
def cans_needed : ℕ := 76

/-- The number of cakes Carol ate -/
def cakes_eaten_by_carol : ℕ := cakes_per_day * baking_days - cans_needed / cans_per_cake

theorem carol_ate_twelve_cakes : cakes_eaten_by_carol = 12 := by
  sorry

end NUMINAMATH_CALUDE_carol_ate_twelve_cakes_l3234_323435


namespace NUMINAMATH_CALUDE_orange_cost_theorem_l3234_323427

/-- The cost of oranges given a specific rate and quantity -/
def orangeCost (ratePrice : ℚ) (rateQuantity : ℚ) (purchaseQuantity : ℚ) : ℚ :=
  (ratePrice / rateQuantity) * purchaseQuantity

/-- Theorem: The cost of 18 pounds of oranges is 15 dollars when the rate is 5 dollars for 6 pounds -/
theorem orange_cost_theorem :
  orangeCost 5 6 18 = 15 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_theorem_l3234_323427


namespace NUMINAMATH_CALUDE_no_such_hexagon_exists_l3234_323476

-- Define a hexagon as a collection of 6 points in 2D space
def Hexagon := Fin 6 → ℝ × ℝ

-- Define convexity for a hexagon
def is_convex (h : Hexagon) : Prop := sorry

-- Define the condition that all sides are greater than 1
def all_sides_greater_than_one (h : Hexagon) : Prop :=
  ∀ i : Fin 6, dist (h i) (h ((i + 1) % 6)) > 1

-- Define the condition that the distance from M to any vertex is less than 1
def all_vertices_less_than_one_from_point (h : Hexagon) (m : ℝ × ℝ) : Prop :=
  ∀ i : Fin 6, dist (h i) m < 1

-- The main theorem
theorem no_such_hexagon_exists :
  ¬ ∃ (h : Hexagon) (m : ℝ × ℝ),
    is_convex h ∧
    all_sides_greater_than_one h ∧
    all_vertices_less_than_one_from_point h m :=
sorry

end NUMINAMATH_CALUDE_no_such_hexagon_exists_l3234_323476


namespace NUMINAMATH_CALUDE_difference_of_squares_l3234_323470

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3234_323470


namespace NUMINAMATH_CALUDE_f_property_l3234_323457

/-- A cubic function with specific properties -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x + 7

/-- Theorem stating that if f(-7) = -17, then f(7) = 31 -/
theorem f_property (a b : ℝ) (h : f a b (-7) = -17) : f a b 7 = 31 := by
  sorry

end NUMINAMATH_CALUDE_f_property_l3234_323457


namespace NUMINAMATH_CALUDE_fraction_cube_sum_l3234_323405

theorem fraction_cube_sum : 
  (10 / 11) ^ 3 * (1 / 3) ^ 3 + (1 / 2) ^ 3 = 5492 / 35937 := by
  sorry

end NUMINAMATH_CALUDE_fraction_cube_sum_l3234_323405


namespace NUMINAMATH_CALUDE_f_difference_l3234_323449

/-- The function f(x) = 3x^2 + 5x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^2 + 5 * x + 4

/-- Theorem stating that f(x+h) - f(x) = h(6x + 3h + 5) for all real x and h -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h + 5) := by
  sorry

end NUMINAMATH_CALUDE_f_difference_l3234_323449


namespace NUMINAMATH_CALUDE_thermometer_to_bottle_ratio_l3234_323439

/-- Proves that the ratio of thermometers sold to hot-water bottles sold is 7:1 given the problem conditions -/
theorem thermometer_to_bottle_ratio :
  ∀ (T H : ℕ), 
    (2 * T + 6 * H = 1200) →  -- Total sales equation
    (H = 60) →                -- Number of hot-water bottles sold
    (T : ℚ) / H = 7 / 1 :=    -- Ratio of thermometers to hot-water bottles
by
  sorry

#check thermometer_to_bottle_ratio

end NUMINAMATH_CALUDE_thermometer_to_bottle_ratio_l3234_323439


namespace NUMINAMATH_CALUDE_leastSquaresSolution_l3234_323485

-- Define the data points
def x : List ℝ := [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
def y : List ℝ := [6.01, 5.07, 4.30, 3.56, 3.07, 2.87, 2.18, 2.00, 2.14]

-- Define the quadratic model
def model (a₁ a₂ a₃ x : ℝ) : ℝ := a₁ * x^2 + a₂ * x + a₃

-- Define the sum of squared residuals
def sumSquaredResiduals (a₁ a₂ a₃ : ℝ) : ℝ :=
  List.sum (List.zipWith (λ xᵢ yᵢ => (yᵢ - model a₁ a₂ a₃ xᵢ)^2) x y)

-- State the theorem
theorem leastSquaresSolution :
  let a₁ : ℝ := 0.95586
  let a₂ : ℝ := -1.9733
  let a₃ : ℝ := 3.0684
  ∀ b₁ b₂ b₃ : ℝ, sumSquaredResiduals a₁ a₂ a₃ ≤ sumSquaredResiduals b₁ b₂ b₃ := by
  sorry

end NUMINAMATH_CALUDE_leastSquaresSolution_l3234_323485


namespace NUMINAMATH_CALUDE_sum_of_operation_l3234_323424

def A : Finset ℕ := {1, 2, 3, 5}
def B : Finset ℕ := {1, 2}

def operation (A B : Finset ℕ) : Finset ℕ :=
  Finset.image (λ (x : ℕ × ℕ) => x.1 * x.2) (A.product B)

theorem sum_of_operation :
  (operation A B).sum id = 31 := by sorry

end NUMINAMATH_CALUDE_sum_of_operation_l3234_323424


namespace NUMINAMATH_CALUDE_room_area_square_inches_l3234_323473

-- Define the conversion rate from feet to inches
def inches_per_foot : ℕ := 12

-- Define the side length of the room in feet
def room_side_feet : ℕ := 10

-- Theorem to prove the area of the room in square inches
theorem room_area_square_inches : 
  (room_side_feet * inches_per_foot) ^ 2 = 14400 := by
  sorry

end NUMINAMATH_CALUDE_room_area_square_inches_l3234_323473


namespace NUMINAMATH_CALUDE_cannot_make_105_with_5_coins_l3234_323428

def coin_denominations : List ℕ := [1, 5, 10, 25, 50]

def is_valid_sum (sum : ℕ) (n : ℕ) : Prop :=
  ∃ (coins : List ℕ), 
    coins.all (λ c => c ∈ coin_denominations) ∧ 
    coins.length = n ∧
    coins.sum = sum

theorem cannot_make_105_with_5_coins : 
  ¬ (is_valid_sum 105 5) :=
sorry

end NUMINAMATH_CALUDE_cannot_make_105_with_5_coins_l3234_323428


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l3234_323421

theorem pure_imaginary_fraction (b : ℝ) : 
  (Complex.I * (((1 : ℂ) + b * Complex.I) / ((2 : ℂ) - Complex.I))).re = 0 → 
  (((1 : ℂ) + b * Complex.I) / ((2 : ℂ) - Complex.I)).im ≠ 0 → 
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l3234_323421


namespace NUMINAMATH_CALUDE_max_value_xyz_expression_l3234_323436

theorem max_value_xyz_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x * y * z * (x + y + z) / ((x + y)^2 * (y + z)^2) ≤ 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_xyz_expression_l3234_323436


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l3234_323444

theorem fixed_point_on_line (m : ℝ) : 
  2 * (1/2 : ℝ) + m * ((1/2 : ℝ) - (1/2 : ℝ)) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l3234_323444


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficient_l3234_323412

/-- 
Given a quadratic equation 2x^2 = 3x - 1, prove that when converted to the general form
ax^2 + bx + c = 0 with a = 2, the coefficient of the linear term (b) is -3.
-/
theorem quadratic_equation_coefficient : ∃ b c : ℝ, 2 * x^2 = 3 * x - 1 ↔ 2 * x^2 + b * x + c = 0 ∧ b = -3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficient_l3234_323412


namespace NUMINAMATH_CALUDE_quadratic_sequence_problem_l3234_323471

theorem quadratic_sequence_problem (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ) 
  (eq1 : x₁ + 3*x₂ + 5*x₃ + 7*x₄ + 9*x₅ + 11*x₆ = 2)
  (eq2 : 3*x₁ + 5*x₂ + 7*x₃ + 9*x₄ + 11*x₅ + 13*x₆ = 15)
  (eq3 : 5*x₁ + 7*x₂ + 9*x₃ + 11*x₄ + 13*x₅ + 15*x₆ = 52) :
  7*x₁ + 9*x₂ + 11*x₃ + 13*x₄ + 15*x₅ + 17*x₆ = 65 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sequence_problem_l3234_323471


namespace NUMINAMATH_CALUDE_trivia_team_absentees_l3234_323433

theorem trivia_team_absentees (total_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) : 
  total_members = 9 →
  points_per_member = 2 →
  total_points = 12 →
  total_members - (total_points / points_per_member) = 3 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_absentees_l3234_323433


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3234_323452

theorem complex_magnitude_product : Complex.abs (4 - 3*I) * Complex.abs (4 + 3*I) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3234_323452


namespace NUMINAMATH_CALUDE_line_relationship_l3234_323429

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define the relationships between lines
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (intersecting : Line → Line → Prop)

-- State the theorem
theorem line_relationship (a b c : Line) 
  (h1 : skew a b) 
  (h2 : parallel c a) : 
  skew c b ∨ intersecting c b :=
sorry

end NUMINAMATH_CALUDE_line_relationship_l3234_323429


namespace NUMINAMATH_CALUDE_inverse_inequality_l3234_323404

theorem inverse_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a < b) :
  1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l3234_323404


namespace NUMINAMATH_CALUDE_meaningful_sqrt_range_l3234_323455

theorem meaningful_sqrt_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_meaningful_sqrt_range_l3234_323455


namespace NUMINAMATH_CALUDE_fishmonger_sales_l3234_323401

/-- Given a first week's sales and a multiplier for the second week's sales,
    calculate the total sales over two weeks. -/
def totalSales (firstWeekSales secondWeekMultiplier : ℕ) : ℕ :=
  firstWeekSales + firstWeekSales * secondWeekMultiplier

/-- Theorem stating that given the specific conditions of the problem,
    the total sales over two weeks is 200 kg. -/
theorem fishmonger_sales : totalSales 50 3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_fishmonger_sales_l3234_323401


namespace NUMINAMATH_CALUDE_stratified_sampling_female_athletes_l3234_323465

theorem stratified_sampling_female_athletes 
  (total_population : ℕ) 
  (sample_size : ℕ) 
  (female_athletes : ℕ) 
  (h1 : total_population = 224) 
  (h2 : sample_size = 32) 
  (h3 : female_athletes = 84) : 
  ↑sample_size * female_athletes / total_population = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_female_athletes_l3234_323465


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_equals_negative_three_l3234_323446

theorem tan_alpha_two_implies_fraction_equals_negative_three (α : Real) 
  (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_equals_negative_three_l3234_323446


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l3234_323400

theorem quadratic_inequality_solution_condition (d : ℝ) :
  d > 0 →
  (∃ x : ℝ, x^2 - 8*x + d < 0) ↔ d < 16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_condition_l3234_323400


namespace NUMINAMATH_CALUDE_cost_of_field_trip_l3234_323479

def field_trip_cost (grandma_contribution : ℝ) (candy_bar_price : ℝ) (candy_bars_to_sell : ℕ) : ℝ :=
  grandma_contribution + candy_bar_price * (candy_bars_to_sell : ℝ)

theorem cost_of_field_trip :
  field_trip_cost 250 1.25 188 = 485 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_field_trip_l3234_323479


namespace NUMINAMATH_CALUDE_trajectory_of_midpoint_l3234_323410

/-- The trajectory of point P given a moving point M on a circle and a fixed point B -/
theorem trajectory_of_midpoint (x y : ℝ) : 
  (∃ m n : ℝ, m^2 + n^2 = 1 ∧ 
              x = (m + 3) / 2 ∧ 
              y = n / 2) → 
  (2*x - 3)^2 + 4*y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_midpoint_l3234_323410


namespace NUMINAMATH_CALUDE_min_value_of_quartic_plus_constant_l3234_323411

theorem min_value_of_quartic_plus_constant :
  (∀ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 ≥ 2022) ∧
  (∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2023 = 2022) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_quartic_plus_constant_l3234_323411


namespace NUMINAMATH_CALUDE_clerts_in_120_degrees_proof_l3234_323488

/-- Represents the number of clerts in a full circle for Martian angle measurement -/
def clerts_in_full_circle : ℕ := 800

/-- Converts degrees to clerts -/
def degrees_to_clerts (degrees : ℚ) : ℚ :=
  (degrees / 360) * clerts_in_full_circle

/-- The number of clerts in a 120° angle -/
def clerts_in_120_degrees : ℕ := 267

theorem clerts_in_120_degrees_proof : 
  ⌊degrees_to_clerts 120⌋ = clerts_in_120_degrees :=
sorry

end NUMINAMATH_CALUDE_clerts_in_120_degrees_proof_l3234_323488


namespace NUMINAMATH_CALUDE_range_m_for_always_negative_range_m_for_bounded_interval_l3234_323475

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Theorem 1
theorem range_m_for_always_negative (m : ℝ) :
  (∀ x : ℝ, f m x < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

-- Theorem 2
theorem range_m_for_bounded_interval (m : ℝ) :
  (∀ x ∈ Set.Icc 1 3, f m x < -m + 5) ↔ m < 6/7 :=
sorry

end NUMINAMATH_CALUDE_range_m_for_always_negative_range_m_for_bounded_interval_l3234_323475


namespace NUMINAMATH_CALUDE_product_equality_l3234_323495

theorem product_equality : 100 * 19.98 * 2.998 * 1000 = 5994004 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l3234_323495


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3234_323434

def A : Set ℤ := {-2, 0}
def B : Set ℤ := {-2, 3}

theorem union_of_A_and_B : A ∪ B = {-2, 0, 3} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3234_323434


namespace NUMINAMATH_CALUDE_average_customer_donation_l3234_323480

/-- Given a restaurant fundraiser where:
    1. The restaurant's donation is 1/5 of the total customer donation.
    2. There are 40 customers.
    3. The restaurant's total donation is $24.
    Prove that the average customer donation is $3. -/
theorem average_customer_donation (restaurant_ratio : ℚ) (num_customers : ℕ) (restaurant_donation : ℚ) :
  restaurant_ratio = 1 / 5 →
  num_customers = 40 →
  restaurant_donation = 24 →
  (restaurant_donation / restaurant_ratio) / num_customers = 3 := by
sorry

end NUMINAMATH_CALUDE_average_customer_donation_l3234_323480


namespace NUMINAMATH_CALUDE_inverse_proportion_k_value_l3234_323430

theorem inverse_proportion_k_value (k : ℝ) (y x : ℝ → ℝ) :
  (∀ x, y x = k / x) →  -- y is an inverse proportion function of x
  k < 0 →  -- k is negative
  (∀ x, 1 ≤ x → x ≤ 3 → y x ≤ y 1 ∧ y x ≥ y 3) →  -- y is decreasing on [1, 3]
  y 1 - y 3 = 4 →  -- difference between max and min values is 4
  k = -6 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_k_value_l3234_323430


namespace NUMINAMATH_CALUDE_glasses_in_five_hours_l3234_323481

/-- The number of glasses of water consumed in a given time period -/
def glasses_consumed (rate_minutes : ℕ) (time_hours : ℕ) : ℕ :=
  (time_hours * 60) / rate_minutes

/-- Theorem: Given a rate of 1 glass every 20 minutes, 
    the number of glasses consumed in 5 hours is 15 -/
theorem glasses_in_five_hours : glasses_consumed 20 5 = 15 := by
  sorry

end NUMINAMATH_CALUDE_glasses_in_five_hours_l3234_323481


namespace NUMINAMATH_CALUDE_polynomial_symmetry_l3234_323478

-- Define the polynomial function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem polynomial_symmetry (a b : ℝ) :
  f a b (-2) = 10 → f a b 2 = -26 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_symmetry_l3234_323478


namespace NUMINAMATH_CALUDE_point_A_coordinates_l3234_323420

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 5

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 4 = 0

-- Define a point on the circle
def point_on_circle (P : ℝ × ℝ) : Prop := circle_O P.1 P.2

-- Define a point on the line
def point_on_line (A : ℝ × ℝ) : Prop := line_l A.1 A.2

-- Define the angle PAQ
def angle_PAQ (A P Q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem point_A_coordinates :
  ∀ A : ℝ × ℝ,
  point_on_line A →
  (∀ P Q : ℝ × ℝ, point_on_circle P → point_on_circle Q → angle_PAQ A P Q ≤ 90) →
  (∃ P Q : ℝ × ℝ, point_on_circle P ∧ point_on_circle Q ∧ angle_PAQ A P Q = 90) →
  A = (1, 3) :=
sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l3234_323420


namespace NUMINAMATH_CALUDE_gcd_9247_4567_l3234_323482

theorem gcd_9247_4567 : Nat.gcd 9247 4567 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_9247_4567_l3234_323482


namespace NUMINAMATH_CALUDE_abs_function_domain_range_intersection_l3234_323463

def A : Set ℝ := {-1, 0, 1}

def f (x : ℝ) : ℝ := |x|

theorem abs_function_domain_range_intersection :
  (A ∩ (f '' A)) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_abs_function_domain_range_intersection_l3234_323463


namespace NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l3234_323494

theorem sqrt_eight_and_nine_sixteenths : 
  Real.sqrt (8 + 9/16) = Real.sqrt 137 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_eight_and_nine_sixteenths_l3234_323494


namespace NUMINAMATH_CALUDE_equidistant_from_axes_l3234_323464

/-- A point in the 2D plane is equidistant from both coordinate axes if and only if the square of its x-coordinate equals the square of its y-coordinate. -/
theorem equidistant_from_axes (x y : ℝ) : (|x| = |y|) ↔ (x^2 = y^2) := by sorry

end NUMINAMATH_CALUDE_equidistant_from_axes_l3234_323464


namespace NUMINAMATH_CALUDE_circle_properties_l3234_323409

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 1

-- Theorem statement
theorem circle_properties :
  -- The center is on the y-axis
  ∃ (y₀ : ℝ), circle_equation 0 y₀
  -- The radius is 1
  ∧ ∀ (x y : ℝ), circle_equation x y → (x^2 + (y - 2)^2 = 1)
  -- The circle passes through (1,2)
  ∧ circle_equation 1 2 := by
sorry

end NUMINAMATH_CALUDE_circle_properties_l3234_323409


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3234_323461

/-- Given a hyperbola with equation (x^2 / 9) - (y^2 / 16) = 1, 
    prove its eccentricity and asymptote equations -/
theorem hyperbola_properties :
  let hyperbola := fun (x y : ℝ) => (x^2 / 9) - (y^2 / 16) = 1
  let eccentricity := 5/3
  let asymptote := fun (x : ℝ) => (4/3) * x
  (∀ x y, hyperbola x y → 
    (∃ c, c^2 = 25 ∧ eccentricity = c / 3)) ∧
  (∀ x, hyperbola x (asymptote x) ∨ hyperbola x (-asymptote x)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3234_323461


namespace NUMINAMATH_CALUDE_cooking_participants_l3234_323438

/-- The number of people who practice yoga -/
def yoga : ℕ := 35

/-- The number of people who study weaving -/
def weaving : ℕ := 15

/-- The number of people who study cooking only -/
def cooking_only : ℕ := 7

/-- The number of people who study both cooking and yoga -/
def cooking_and_yoga : ℕ := 5

/-- The number of people who participate in all curriculums -/
def all_curriculums : ℕ := 3

/-- The number of people who study both cooking and weaving -/
def cooking_and_weaving : ℕ := 5

/-- The total number of people who study cooking -/
def total_cooking : ℕ := cooking_only + (cooking_and_yoga - all_curriculums) + (cooking_and_weaving - all_curriculums) + all_curriculums

theorem cooking_participants : total_cooking = 14 := by
  sorry

end NUMINAMATH_CALUDE_cooking_participants_l3234_323438


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l3234_323490

theorem sum_of_three_numbers (a b c : ℕ) : 
  a = 200 → 
  b = 2 * c → 
  c = 100 → 
  a + b + c = 500 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l3234_323490


namespace NUMINAMATH_CALUDE_product_of_fractions_equals_21_8_l3234_323497

theorem product_of_fractions_equals_21_8 : 
  let f (n : ℕ) := (n^3 - 1) * (n - 2) / (n^3 + 1)
  f 3 * f 5 * f 7 * f 9 * f 11 = 21 / 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_equals_21_8_l3234_323497


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3234_323498

theorem least_positive_integer_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  n % 3 = 2 ∧
  n % 4 = 3 ∧
  n % 5 = 4 ∧
  n % 6 = 5 ∧
  n % 7 = 6 ∧
  (∀ m : ℕ, m > 0 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 ∧ m % 6 = 5 ∧ m % 7 = 6 → m ≥ n) :=
by
  use 2519
  sorry

#eval 2519 % 3  -- Should output 2
#eval 2519 % 4  -- Should output 3
#eval 2519 % 5  -- Should output 4
#eval 2519 % 6  -- Should output 5
#eval 2519 % 7  -- Should output 6

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l3234_323498


namespace NUMINAMATH_CALUDE_distance_traveled_l3234_323431

/-- Given a speed of 20 km/hr and a travel time of 2.5 hours, prove that the distance traveled is 50 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h1 : speed = 20)
  (h2 : time = 2.5)
  (h3 : distance = speed * time) : 
  distance = 50 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l3234_323431


namespace NUMINAMATH_CALUDE_money_distribution_l3234_323484

/-- Theorem: Given the conditions of money distribution among A, B, and C,
    prove that B receives 2/7 of what A and C together get. -/
theorem money_distribution (a b c : ℚ) : 
  a = (1 : ℚ) / 3 * (b + c) →  -- A gets one-third of what B and C together get
  ∃ x : ℚ, b = x * (a + c) →   -- B gets a fraction of what A and C together get
  a = b + 35 →                 -- A receives $35 more than B
  a + b + c = 1260 →           -- Total amount shared is $1260
  b = (2 : ℚ) / 7 * (a + c) := by  -- B receives 2/7 of what A and C together get
sorry

end NUMINAMATH_CALUDE_money_distribution_l3234_323484


namespace NUMINAMATH_CALUDE_ratio_problem_l3234_323466

theorem ratio_problem (antecedent consequent : ℚ) : 
  antecedent / consequent = 4 / 6 → antecedent = 20 → consequent = 30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3234_323466


namespace NUMINAMATH_CALUDE_relationship_correctness_l3234_323408

theorem relationship_correctness :
  (∃ a b c : ℝ, (a > b ↔ a * c^2 > b * c^2) → False) ∧
  (∃ a b : ℝ, (a > b → 1 / a < 1 / b) → False) ∧
  (∃ a b c d : ℝ, (a > b ∧ b > 0 ∧ c > d → a / d > b / c) → False) ∧
  (∀ a b c : ℝ, a > b ∧ b > 0 → a^c < b^c) :=
by sorry


end NUMINAMATH_CALUDE_relationship_correctness_l3234_323408


namespace NUMINAMATH_CALUDE_prime_divisors_inequality_l3234_323422

-- Define the variables
variable (x y z : ℕ)
variable (p q : ℕ)

-- Define the conditions
variable (h1 : x > 2)
variable (h2 : y > 1)
variable (h3 : z > 0)
variable (h4 : x^y + 1 = z^2)

-- Define p and q
variable (hp : p = (Nat.factors x).card)
variable (hq : q = (Nat.factors y).card)

-- State the theorem
theorem prime_divisors_inequality : p ≥ q + 2 := by
  sorry

end NUMINAMATH_CALUDE_prime_divisors_inequality_l3234_323422


namespace NUMINAMATH_CALUDE_zoo_trip_vans_needed_l3234_323483

theorem zoo_trip_vans_needed (van_capacity : ℕ) (students : ℕ) (adults : ℕ) : 
  van_capacity = 5 → students = 12 → adults = 3 → 
  (students + adults + van_capacity - 1) / van_capacity = 3 := by
  sorry

end NUMINAMATH_CALUDE_zoo_trip_vans_needed_l3234_323483


namespace NUMINAMATH_CALUDE_negative_forty_divided_by_five_l3234_323443

theorem negative_forty_divided_by_five : (-40 : ℤ) / 5 = -8 := by
  sorry

end NUMINAMATH_CALUDE_negative_forty_divided_by_five_l3234_323443


namespace NUMINAMATH_CALUDE_area_SUVR_area_SUVR_is_141_44_l3234_323447

/-- Triangle PQR with given properties and points S, T, U, V as described -/
structure TrianglePQR where
  /-- Side length PR -/
  pr : ℝ
  /-- Side length PQ -/
  pq : ℝ
  /-- Area of triangle PQR -/
  area : ℝ
  /-- Point S on PR such that PS = 1/3 * PR -/
  s : ℝ
  /-- Point T on PQ such that PT = 1/3 * PQ -/
  t : ℝ
  /-- Point U on ST -/
  u : ℝ
  /-- Point V on QR -/
  v : ℝ
  /-- PR equals 60 -/
  h_pr : pr = 60
  /-- PQ equals 15 -/
  h_pq : pq = 15
  /-- Area of triangle PQR equals 180 -/
  h_area : area = 180
  /-- PS equals 1/3 of PR -/
  h_s : s = 1/3 * pr
  /-- PT equals 1/3 of PQ -/
  h_t : t = 1/3 * pq
  /-- U is on the angle bisector of angle PQR -/
  h_u_bisector : True  -- Placeholder for the angle bisector condition
  /-- V is on the angle bisector of angle PQR -/
  h_v_bisector : True  -- Placeholder for the angle bisector condition

/-- The area of quadrilateral SUVR in the given triangle PQR is 141.44 -/
theorem area_SUVR (tri : TrianglePQR) : ℝ := 141.44

/-- The main theorem: The area of quadrilateral SUVR is 141.44 -/
theorem area_SUVR_is_141_44 (tri : TrianglePQR) : area_SUVR tri = 141.44 := by
  sorry

end NUMINAMATH_CALUDE_area_SUVR_area_SUVR_is_141_44_l3234_323447


namespace NUMINAMATH_CALUDE_swimming_speed_is_15_l3234_323403

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  /-- The man's swimming speed in still water (km/h) -/
  v : ℝ
  /-- The speed of the stream (km/h) -/
  s : ℝ
  /-- The time it takes to swim downstream (hours) -/
  t : ℝ
  /-- Assertion that it takes twice as long to swim upstream -/
  upstream_time : (v - s) * (2 * t) = (v + s) * t
  /-- The speed of the stream is 5 km/h -/
  stream_speed : s = 5

/-- Theorem stating that the man's swimming speed in still water is 15 km/h -/
theorem swimming_speed_is_15 (scenario : SwimmingScenario) : scenario.v = 15 := by
  sorry

end NUMINAMATH_CALUDE_swimming_speed_is_15_l3234_323403


namespace NUMINAMATH_CALUDE_team_loss_percentage_l3234_323448

/-- Represents the ratio of games won to games lost -/
def winLossRatio : Rat := 7 / 3

/-- The total number of games played -/
def totalGames : ℕ := 50

/-- Calculates the percentage of games lost -/
def percentLost : ℚ :=
  let gamesLost := totalGames / (1 + winLossRatio)
  (gamesLost / totalGames) * 100

theorem team_loss_percentage :
  ⌊percentLost⌋ = 30 :=
sorry

end NUMINAMATH_CALUDE_team_loss_percentage_l3234_323448


namespace NUMINAMATH_CALUDE_circle_sum_inequality_l3234_323472

theorem circle_sum_inequality (nums : Fin 100 → ℝ) (h_distinct : Function.Injective nums) :
  ∃ (i : Fin 100), (nums i + nums ((i + 1) % 100)) < (nums ((i + 2) % 100) + nums ((i + 3) % 100)) :=
sorry

end NUMINAMATH_CALUDE_circle_sum_inequality_l3234_323472


namespace NUMINAMATH_CALUDE_cuboid_base_area_l3234_323416

/-- Theorem: For a cuboid with volume 28 cm³ and height 4 cm, the base area is 7 cm² -/
theorem cuboid_base_area (volume : ℝ) (height : ℝ) (base_area : ℝ) :
  volume = 28 →
  height = 4 →
  volume = base_area * height →
  base_area = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_cuboid_base_area_l3234_323416


namespace NUMINAMATH_CALUDE_bus_interval_theorem_l3234_323413

/-- The interval between buses on a circular route -/
def interval (num_buses : ℕ) (total_time : ℕ) : ℕ :=
  total_time / num_buses

/-- The theorem stating the relationship between intervals for 2 and 3 buses -/
theorem bus_interval_theorem (total_time : ℕ) :
  interval 2 total_time = 21 → interval 3 total_time = 14 :=
by
  sorry

#check bus_interval_theorem

end NUMINAMATH_CALUDE_bus_interval_theorem_l3234_323413


namespace NUMINAMATH_CALUDE_board_piece_difference_l3234_323414

def board_length : ℝ := 20
def shorter_piece : ℝ := 8

theorem board_piece_difference : 
  let longer_piece := board_length - shorter_piece
  2 * shorter_piece - longer_piece = 4 := by
  sorry

end NUMINAMATH_CALUDE_board_piece_difference_l3234_323414


namespace NUMINAMATH_CALUDE_man_speed_man_speed_proof_l3234_323440

/-- Calculates the speed of a man moving opposite to a bullet train -/
theorem man_speed (train_length : ℝ) (train_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_length / passing_time * 3.6
  relative_speed - train_speed

/-- Proves that the man's speed is 4 kmph given the specific conditions -/
theorem man_speed_proof :
  man_speed 120 50 8 = 4 := by
  sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_proof_l3234_323440


namespace NUMINAMATH_CALUDE_no_real_roots_l3234_323493

-- Define the operation ⊕
def oplus (m n : ℝ) : ℝ := n^2 - m*n + 1

-- Theorem statement
theorem no_real_roots :
  ∀ x : ℝ, oplus 1 x ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3234_323493


namespace NUMINAMATH_CALUDE_inequality_range_l3234_323437

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3234_323437


namespace NUMINAMATH_CALUDE_rectangular_plot_area_l3234_323425

theorem rectangular_plot_area 
  (L B : ℝ) 
  (h_ratio : L / B = 7 / 5) 
  (h_perimeter : 2 * (L + B) = 288) : 
  L * B = 5040 := by sorry

end NUMINAMATH_CALUDE_rectangular_plot_area_l3234_323425


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l3234_323407

theorem cubic_equation_solution (a : ℝ) (h : a^2 - a - 1 = 0) : 
  a^3 - a^2 - a + 2023 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l3234_323407


namespace NUMINAMATH_CALUDE_smallest_sum_l3234_323477

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Reverses the digits of a two-digit number -/
def reverseDigits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Represents the problem setup -/
def ProblemSetup (x y m : ℕ) : Prop :=
  TwoDigitInt x ∧
  TwoDigitInt y ∧
  y = reverseDigits x ∧
  x^2 + y^2 = m^2 ∧
  ∃ k, x + y = 9 * (2 * k + 1)

theorem smallest_sum (x y m : ℕ) (h : ProblemSetup x y m) :
  x + y + m ≥ 169 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_l3234_323477


namespace NUMINAMATH_CALUDE_product_of_consecutive_integers_l3234_323423

theorem product_of_consecutive_integers : ∃ (a b c d e : ℤ),
  b = a + 1 ∧
  d = c + 1 ∧
  e = d + 1 ∧
  a * b = 300 ∧
  c * d * e = 300 ∧
  a + b + c + d + e = 49 :=
by sorry

end NUMINAMATH_CALUDE_product_of_consecutive_integers_l3234_323423


namespace NUMINAMATH_CALUDE_product_ends_in_zero_theorem_l3234_323491

def is_valid_assignment (assignment : Char → ℕ) : Prop :=
  (∀ c₁ c₂, c₁ ≠ c₂ → assignment c₁ ≠ assignment c₂) ∧
  (∀ c, assignment c < 10)

def satisfies_equation (assignment : Char → ℕ) : Prop :=
  10 * (assignment 'Ж') + (assignment 'Ж') + (assignment 'Ж') =
  100 * (assignment 'М') + 10 * (assignment 'Ё') + (assignment 'Д')

def product_ends_in_zero (assignment : Char → ℕ) : Prop :=
  (assignment 'В' * assignment 'И' * assignment 'H' * assignment 'H' *
   assignment 'U' * assignment 'П' * assignment 'У' * assignment 'X') % 10 = 0

theorem product_ends_in_zero_theorem (assignment : Char → ℕ) :
  is_valid_assignment assignment → satisfies_equation assignment →
  product_ends_in_zero assignment :=
by
  sorry

#check product_ends_in_zero_theorem

end NUMINAMATH_CALUDE_product_ends_in_zero_theorem_l3234_323491


namespace NUMINAMATH_CALUDE_total_calories_burned_first_week_l3234_323462

def calories_per_hour_walking : ℕ := 300

def calories_per_hour_dancing : ℕ := 2 * calories_per_hour_walking

def calories_per_hour_swimming : ℕ := (3 * calories_per_hour_walking) / 2

def calories_per_hour_cycling : ℕ := calories_per_hour_walking

def dancing_hours_per_week : ℕ := 3 * (2 * 1/2) + 1

def swimming_hours_per_week : ℕ := 2 * 3/2

def cycling_hours_per_week : ℕ := 2

def total_calories_burned : ℕ := 
  calories_per_hour_dancing * dancing_hours_per_week +
  calories_per_hour_swimming * swimming_hours_per_week +
  calories_per_hour_cycling * cycling_hours_per_week

theorem total_calories_burned_first_week : 
  total_calories_burned = 4350 := by sorry

end NUMINAMATH_CALUDE_total_calories_burned_first_week_l3234_323462


namespace NUMINAMATH_CALUDE_area_triangle_abc_l3234_323459

/-- Given a point A(x, y) where x ≠ 0 and y ≠ 0, with B symmetric to A with respect to the x-axis,
    C symmetric to A with respect to the y-axis, and the area of triangle AOB equal to 4,
    prove that the area of triangle ABC is equal to 8. -/
theorem area_triangle_abc (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  let A : ℝ × ℝ := (x, y)
  let B : ℝ × ℝ := (x, -y)
  let C : ℝ × ℝ := (-x, y)
  let O : ℝ × ℝ := (0, 0)
  let area_AOB := abs (x * y)
  area_AOB = 4 → abs (2 * x * y) = 8 :=
by sorry

end NUMINAMATH_CALUDE_area_triangle_abc_l3234_323459


namespace NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_one_forty_satisfies_conditions_one_forty_is_greatest_l3234_323426

theorem greatest_integer_with_gcd_six (n : ℕ) : n < 150 ∧ n.gcd 18 = 6 → n ≤ 140 :=
by sorry

theorem one_forty_satisfies_conditions : 140 < 150 ∧ Nat.gcd 140 18 = 6 :=
by sorry

theorem one_forty_is_greatest : 
  ∀ m : ℕ, m < 150 ∧ m.gcd 18 = 6 → m ≤ 140 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_with_gcd_six_one_forty_satisfies_conditions_one_forty_is_greatest_l3234_323426


namespace NUMINAMATH_CALUDE_power_mod_eleven_l3234_323499

theorem power_mod_eleven : 7^79 ≡ 6 [ZMOD 11] := by sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l3234_323499


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3234_323467

/-- Given an arithmetic sequence {a_n} where a_4 + a_8 = 16, prove that a_2 + a_6 + a_10 = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) : 
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) → -- arithmetic sequence condition
  (a 4 + a 8 = 16) →                               -- given condition
  (a 2 + a 6 + a 10 = 24) :=                       -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3234_323467


namespace NUMINAMATH_CALUDE_min_C_over_D_l3234_323453

theorem min_C_over_D (x C D : ℝ) (hx : x ≠ 0) 
  (hC : x^3 + 1/x^3 = C) (hD : x - 1/x = D) :
  ∀ y : ℝ, y ≠ 0 → y^3 + 1/y^3 / (y - 1/y) ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_min_C_over_D_l3234_323453


namespace NUMINAMATH_CALUDE_power_mod_eleven_l3234_323489

theorem power_mod_eleven : (Nat.pow 3 101 + 5) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_mod_eleven_l3234_323489


namespace NUMINAMATH_CALUDE_monopoly_produces_durable_iff_lowquality_cost_gt_six_l3234_323432

/-- Represents a coffee machine producer -/
structure Producer where
  isDurable : Bool
  cost : ℝ

/-- Represents a consumer of coffee machines -/
structure Consumer where
  benefit : ℝ
  periods : ℕ

/-- Represents the market for coffee machines -/
inductive Market
  | Monopoly
  | PerfectlyCompetitive

/-- Define the conditions for the coffee machine problem -/
def coffeeMachineProblem (c : Consumer) (pd : Producer) (pl : Producer) (m : Market) : Prop :=
  c.periods = 2 ∧ 
  c.benefit = 20 ∧
  pd.isDurable = true ∧
  pd.cost = 12 ∧
  pl.isDurable = false

/-- Theorem: A monopoly will produce only durable coffee machines if and only if 
    the average cost of producing a low-quality coffee machine is greater than 6 monetary units -/
theorem monopoly_produces_durable_iff_lowquality_cost_gt_six 
  (c : Consumer) (pd : Producer) (pl : Producer) (m : Market) :
  coffeeMachineProblem c pd pl Market.Monopoly →
  (∀ S, pl.cost = S → (pd.cost < pl.cost ↔ S > 6)) :=
sorry

end NUMINAMATH_CALUDE_monopoly_produces_durable_iff_lowquality_cost_gt_six_l3234_323432


namespace NUMINAMATH_CALUDE_stock_value_order_l3234_323474

/-- Represents the value of a stock over time -/
structure Stock :=
  (initial : ℝ)
  (first_year_change : ℝ)
  (second_year_change : ℝ)

/-- Calculates the final value of a stock after two years -/
def final_value (s : Stock) : ℝ :=
  s.initial * (1 + s.first_year_change) * (1 + s.second_year_change)

/-- The three stocks: Alabama Almonds (AA), Boston Beans (BB), and California Cauliflower (CC) -/
def AA : Stock := ⟨100, 0.2, -0.2⟩
def BB : Stock := ⟨100, -0.25, 0.25⟩
def CC : Stock := ⟨100, 0, 0⟩

theorem stock_value_order :
  final_value BB < final_value AA ∧ final_value AA < final_value CC :=
sorry

end NUMINAMATH_CALUDE_stock_value_order_l3234_323474


namespace NUMINAMATH_CALUDE_sequence_fifth_b_l3234_323417

/-- Given a sequence {aₙ}, where 2aₙ and aₙ₊₁ are the roots of x² - 3x + bₙ = 0,
    and a₁ = 2, prove that b₅ = -1054 -/
theorem sequence_fifth_b (a : ℕ → ℝ) (b : ℕ → ℝ) : 
  (∀ n, (2 * a n) * (a (n + 1)) = b n) → 
  (∀ n, 2 * a n + a (n + 1) = 3) → 
  a 1 = 2 → 
  b 5 = -1054 :=
by sorry

end NUMINAMATH_CALUDE_sequence_fifth_b_l3234_323417


namespace NUMINAMATH_CALUDE_pentomino_tiling_l3234_323419

/-- A pentomino is a shape that covers exactly 5 squares. -/
def Pentomino : Type := Unit

/-- A rectangle of size 5 × m. -/
structure Rectangle (m : ℕ) :=
  (width : Fin 5)
  (height : Fin m)

/-- Predicate to determine if a rectangle can be tiled by a pentomino. -/
def IsTileable (m : ℕ) : Prop := sorry

theorem pentomino_tiling (m : ℕ) : 
  IsTileable m ↔ Even m := by sorry

end NUMINAMATH_CALUDE_pentomino_tiling_l3234_323419
