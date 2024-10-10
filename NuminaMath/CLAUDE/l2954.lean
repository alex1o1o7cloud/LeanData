import Mathlib

namespace martha_coffee_savings_l2954_295410

/-- Calculates the annual savings from reducing coffee spending --/
def coffee_savings (latte_price : ℚ) (latte_days : ℕ) (ice_coffee_price : ℚ) (ice_coffee_days : ℕ) (reduction_percent : ℚ) : ℚ :=
  let weekly_latte_cost := latte_price * latte_days
  let weekly_ice_coffee_cost := ice_coffee_price * ice_coffee_days
  let weekly_total_cost := weekly_latte_cost + weekly_ice_coffee_cost
  let annual_cost := weekly_total_cost * 52
  annual_cost * reduction_percent

theorem martha_coffee_savings :
  coffee_savings 4 5 2 3 (1/4) = 338 :=
sorry

end martha_coffee_savings_l2954_295410


namespace second_hour_distance_l2954_295422

/-- Represents a 3-hour bike ride with specific distance relationships --/
structure BikeRide where
  first_hour : ℝ
  second_hour : ℝ
  third_hour : ℝ
  second_hour_condition : second_hour = 1.2 * first_hour
  third_hour_condition : third_hour = 1.25 * second_hour
  total_distance : first_hour + second_hour + third_hour = 37

/-- Theorem stating that the distance traveled in the second hour is 12 miles --/
theorem second_hour_distance (ride : BikeRide) : ride.second_hour = 12 := by
  sorry

end second_hour_distance_l2954_295422


namespace no_right_triangle_with_sides_x_2x_3x_l2954_295424

theorem no_right_triangle_with_sides_x_2x_3x :
  ¬ ∃ (x : ℝ), x > 0 ∧ x^2 + (2*x)^2 = (3*x)^2 := by
  sorry

end no_right_triangle_with_sides_x_2x_3x_l2954_295424


namespace factor_condition_l2954_295484

theorem factor_condition (t : ℚ) :
  (∃ k : ℚ, ∀ x, 4*x^2 + 11*x - 3 = (x - t) * k) ↔ (t = 1/4 ∨ t = -3) := by
sorry

end factor_condition_l2954_295484


namespace cody_tickets_l2954_295469

def arcade_tickets (initial_tickets spent_tickets additional_tickets : ℕ) : ℕ :=
  initial_tickets - spent_tickets + additional_tickets

theorem cody_tickets : arcade_tickets 49 25 6 = 30 := by
  sorry

end cody_tickets_l2954_295469


namespace curve_tangent_perpendicular_l2954_295482

-- Define the curve
def curve (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

-- Define the tangent line
def tangent_slope (a : ℝ) : ℝ := -a

-- Define the perpendicular line
def perp_line (x y : ℝ) : Prop := 2 * x + y + 10 = 0

-- State the theorem
theorem curve_tangent_perpendicular (a : ℝ) (h : a ≠ 0) :
  (tangent_slope a * 2 = -1) → a = -1/2 := by
  sorry

end curve_tangent_perpendicular_l2954_295482


namespace rational_function_value_l2954_295471

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  linear_p : ∃ (a b : ℝ), ∀ x, p x = a * x + b
  quadratic_q : ∃ (a b c : ℝ), ∀ x, q x = a * x^2 + b * x + c
  asymptote_neg4 : q (-4) = 0
  asymptote_1 : q 1 = 0
  point_0 : p 0 / q 0 = -1
  point_1 : p 1 / q 1 = -2

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.p 0 / f.q 0 = 1/4 := by
  sorry

end rational_function_value_l2954_295471


namespace profit_percentage_example_l2954_295421

/-- Calculates the percentage of profit given the cost price and selling price -/
def percentage_profit (cost_price selling_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that for a cost price of $600 and a selling price of $648, the percentage profit is 8% -/
theorem profit_percentage_example :
  percentage_profit 600 648 = 8 := by
  sorry

end profit_percentage_example_l2954_295421


namespace sqrt_equation_solution_l2954_295401

theorem sqrt_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 4 * Real.sqrt (2 + x) + 4 * Real.sqrt (2 - x) = 6 * Real.sqrt 3 ∧ x = (3 * Real.sqrt 15) / 8 := by
  sorry

end sqrt_equation_solution_l2954_295401


namespace pizza_consumption_l2954_295415

theorem pizza_consumption (rachel_pizza : ℕ) (bella_pizza : ℕ)
  (h1 : rachel_pizza = 598)
  (h2 : bella_pizza = 354) :
  rachel_pizza + bella_pizza = 952 :=
by sorry

end pizza_consumption_l2954_295415


namespace fraction_sum_inequality_l2954_295437

theorem fraction_sum_inequality (a b : ℝ) (h : a * b ≠ 0) :
  (a * b > 0 → b / a + a / b ≥ 2) ∧
  (a * b < 0 → |b / a + a / b| ≥ 2) := by
  sorry

end fraction_sum_inequality_l2954_295437


namespace right_triangle_arithmetic_sequence_l2954_295417

theorem right_triangle_arithmetic_sequence (a b c : ℕ) : 
  a < b ∧ b < c →                        -- sides form an increasing sequence
  a + b + c = 840 →                      -- perimeter is 840
  b - a = c - b →                        -- sides form an arithmetic sequence
  a^2 + b^2 = c^2 →                      -- it's a right triangle (Pythagorean theorem)
  c = 350 := by sorry                    -- largest side is 350

end right_triangle_arithmetic_sequence_l2954_295417


namespace principal_amount_calculation_l2954_295474

def simple_interest : ℝ := 4016.25
def interest_rate : ℝ := 0.09
def time_period : ℝ := 5

theorem principal_amount_calculation :
  simple_interest / (interest_rate * time_period) = 8925 := by
  sorry

end principal_amount_calculation_l2954_295474


namespace apple_percentage_after_adding_oranges_l2954_295435

/-- Given a basket with apples and oranges, calculate the percentage of apples after adding more oranges. -/
theorem apple_percentage_after_adding_oranges 
  (initial_apples initial_oranges added_oranges : ℕ) : 
  initial_apples = 10 → 
  initial_oranges = 5 → 
  added_oranges = 5 → 
  (initial_apples : ℚ) / (initial_apples + initial_oranges + added_oranges) * 100 = 50 := by
  sorry

end apple_percentage_after_adding_oranges_l2954_295435


namespace sin_product_identity_l2954_295436

theorem sin_product_identity :
  Real.sin (12 * π / 180) * Real.sin (48 * π / 180) * Real.sin (72 * π / 180) * Real.sin (84 * π / 180) =
  (1 / 8) * (1 + Real.cos (24 * π / 180)) := by
sorry

end sin_product_identity_l2954_295436


namespace pascal_interior_sum_l2954_295408

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- Pascal's Triangle interior numbers start from the third row -/
def interior_start : ℕ := 3

theorem pascal_interior_sum :
  interior_sum 4 = 6 ∧
  interior_sum 5 = 14 →
  interior_sum 9 = 254 := by
  sorry

end pascal_interior_sum_l2954_295408


namespace distance_between_points_l2954_295454

def point1 : ℝ × ℝ := (2, -3)
def point2 : ℝ × ℝ := (8, 9)

theorem distance_between_points : 
  Real.sqrt ((point2.1 - point1.1)^2 + (point2.2 - point1.1)^2) = 6 * Real.sqrt 5 := by
  sorry

end distance_between_points_l2954_295454


namespace largest_difference_is_62_l2954_295405

/-- Given a list of four digits, returns the largest 2-digit number that can be formed --/
def largest_two_digit (digits : List Nat) : Nat :=
  sorry

/-- Given a list of four digits, returns the smallest 2-digit number that can be formed --/
def smallest_two_digit (digits : List Nat) : Nat :=
  sorry

/-- The set of digits to be used --/
def digit_set : List Nat := [2, 4, 6, 8]

theorem largest_difference_is_62 :
  largest_two_digit digit_set - smallest_two_digit digit_set = 62 :=
sorry

end largest_difference_is_62_l2954_295405


namespace rectangle_max_area_l2954_295466

theorem rectangle_max_area (l w : ℕ) : 
  (2 * l + 2 * w = 40) → 
  (∀ a b : ℕ, 2 * a + 2 * b = 40 → l * w ≥ a * b) →
  l * w = 100 :=
by sorry

end rectangle_max_area_l2954_295466


namespace fraction_sum_difference_l2954_295498

theorem fraction_sum_difference : 2/5 + 3/8 - 1/10 = 27/40 := by
  sorry

end fraction_sum_difference_l2954_295498


namespace water_bottles_cost_l2954_295489

/-- The total cost of water bottles given the number of bottles, liters per bottle, and price per liter. -/
def total_cost (num_bottles : ℕ) (liters_per_bottle : ℕ) (price_per_liter : ℕ) : ℕ :=
  num_bottles * liters_per_bottle * price_per_liter

/-- Theorem stating that the total cost of six 2-liter bottles of water is $12 when the price is $1 per liter. -/
theorem water_bottles_cost :
  total_cost 6 2 1 = 12 := by
  sorry

end water_bottles_cost_l2954_295489


namespace x_fourth_minus_reciprocal_l2954_295447

theorem x_fourth_minus_reciprocal (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 723 := by
  sorry

end x_fourth_minus_reciprocal_l2954_295447


namespace missing_number_proof_l2954_295440

theorem missing_number_proof (numbers : List ℕ) (h_count : numbers.length = 9) 
  (h_sum : numbers.sum = 744 + 745 + 747 + 749 + 752 + 752 + 753 + 755 + 755) 
  (h_avg : (numbers.sum + missing) / 10 = 750) : missing = 1748 := by
  sorry

#check missing_number_proof

end missing_number_proof_l2954_295440


namespace store_price_reduction_l2954_295439

theorem store_price_reduction (original_price : ℝ) (h1 : original_price > 0) :
  let first_reduction := 0.09
  let final_price_ratio := 0.819
  let price_after_first := original_price * (1 - first_reduction)
  let second_reduction := 1 - (final_price_ratio / (1 - first_reduction))
  second_reduction = 0.181 := by sorry

end store_price_reduction_l2954_295439


namespace remainder_problem_l2954_295443

theorem remainder_problem (n : ℕ) : n % 13 = 11 → n = 349 → n % 17 = 9 := by
  sorry

end remainder_problem_l2954_295443


namespace clay_target_sequences_l2954_295425

theorem clay_target_sequences (n : ℕ) (a b c : ℕ) 
  (h1 : n = 8) 
  (h2 : a = 3) 
  (h3 : b = 3) 
  (h4 : c = 2) 
  (h5 : a + b + c = n) : 
  (Nat.factorial n) / (Nat.factorial a * Nat.factorial b * Nat.factorial c) = 560 :=
by sorry

end clay_target_sequences_l2954_295425


namespace fraction_to_decimal_l2954_295464

theorem fraction_to_decimal : (13 : ℚ) / (2 * 5^8) = 0.00001664 := by
  sorry

end fraction_to_decimal_l2954_295464


namespace no_cracked_seashells_l2954_295450

theorem no_cracked_seashells 
  (tom_initial : ℕ) 
  (fred_initial : ℕ) 
  (fred_more_than_tom : ℕ) 
  (h1 : tom_initial = 15)
  (h2 : fred_initial = 43)
  (h3 : fred_more_than_tom = 28)
  : ∃ (tom_final fred_final : ℕ),
    tom_initial + fred_initial = tom_final + fred_final ∧
    fred_final = tom_final + fred_more_than_tom ∧
    tom_initial - tom_final = 0 :=
by
  sorry

end no_cracked_seashells_l2954_295450


namespace total_trees_cut_is_1021_l2954_295465

/-- Calculates the total number of trees cut down by James and his helpers. -/
def total_trees_cut (james_rate : ℕ) (brother_rate : ℕ) (cousin_rate : ℕ) (professional_rate : ℕ) : ℕ :=
  let james_alone := 2 * james_rate
  let with_brothers := 3 * (james_rate + 2 * brother_rate)
  let with_cousin := 4 * (james_rate + 2 * brother_rate + cousin_rate)
  let all_together := 5 * (james_rate + 2 * brother_rate + cousin_rate + professional_rate)
  james_alone + with_brothers + with_cousin + all_together

/-- The theorem states that the total number of trees cut down is 1021. -/
theorem total_trees_cut_is_1021 :
  total_trees_cut 20 16 23 30 = 1021 := by
  sorry

#eval total_trees_cut 20 16 23 30

end total_trees_cut_is_1021_l2954_295465


namespace sufficient_not_necessary_l2954_295478

theorem sufficient_not_necessary (a : ℝ) :
  (∀ a, a > 2 → a^2 > 2*a) ∧
  (∃ a, a^2 > 2*a ∧ a ≤ 2) :=
by sorry

end sufficient_not_necessary_l2954_295478


namespace distance_AB_is_five_halves_l2954_295403

/-- Line represented by parametric equations -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- Line represented by a linear equation ax + by = c -/
structure LinearLine where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

def l₁ : ParametricLine :=
  { x := λ t => 1 + 3 * t
    y := λ t => 2 - 4 * t }

def l₂ : LinearLine :=
  { a := 2
    b := -4
    c := 5 }

def A : Point :=
  { x := 1
    y := 2 }

/-- Function to find the intersection point of a parametric line and a linear line -/
def intersection (pl : ParametricLine) (ll : LinearLine) : Point :=
  sorry

/-- Function to calculate the distance between two points -/
def distance (p1 p2 : Point) : ℝ :=
  sorry

theorem distance_AB_is_five_halves :
  distance A (intersection l₁ l₂) = 5 / 2 := by
  sorry

end distance_AB_is_five_halves_l2954_295403


namespace segment_division_sum_l2954_295444

/-- Given a line segment AB with A = (1, 1) and B = (x, y), and a point C = (2, 4) that divides AB in the ratio 2:1, prove that x + y = 8 -/
theorem segment_division_sum (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 1)
  let B : ℝ × ℝ := (x, y)
  let C : ℝ × ℝ := (2, 4)
  (C.1 - A.1) / (B.1 - C.1) = 2 ∧ 
  (C.2 - A.2) / (B.2 - C.2) = 2 →
  x + y = 8 := by
sorry

end segment_division_sum_l2954_295444


namespace total_expenses_calculation_l2954_295495

-- Define the initial conditions
def initial_price : ℝ := 1.4
def daily_price_decrease : ℝ := 0.1
def first_purchase : ℝ := 10
def second_purchase : ℝ := 25
def total_trip_distance : ℝ := 320
def distance_before_friday : ℝ := 200
def fuel_efficiency : ℝ := 8

-- Define the theorem
theorem total_expenses_calculation :
  let friday_price := initial_price - 4 * daily_price_decrease
  let cost_monday := first_purchase * initial_price
  let cost_friday := second_purchase * friday_price
  let total_cost_35_liters := cost_monday + cost_friday
  let remaining_distance := total_trip_distance - distance_before_friday
  let additional_liters := remaining_distance / fuel_efficiency
  let cost_additional_liters := additional_liters * friday_price
  let total_expenses := total_cost_35_liters + cost_additional_liters
  total_expenses = 54 := by sorry

end total_expenses_calculation_l2954_295495


namespace angle_supplement_l2954_295402

theorem angle_supplement (x : ℝ) : 
  (90 - x = 150) → (180 - x = 60) := by
  sorry

end angle_supplement_l2954_295402


namespace exponent_equality_l2954_295432

theorem exponent_equality : 
  ((-2 : ℤ)^3 ≠ (-3 : ℤ)^2) ∧ 
  (-(3 : ℤ)^2 ≠ (-3 : ℤ)^2) ∧ 
  (-(3 : ℤ)^3 = (-3 : ℤ)^3) ∧ 
  (-(3 : ℤ) * (2 : ℤ)^3 ≠ (-3 * 2 : ℤ)^3) :=
by sorry

end exponent_equality_l2954_295432


namespace specific_frustum_smaller_cone_altitude_l2954_295420

/-- Represents a frustum of a right circular cone. -/
structure Frustum where
  altitude : ℝ
  largerBaseArea : ℝ
  smallerBaseArea : ℝ

/-- Calculates the altitude of the smaller cone removed from a frustum. -/
def smallerConeAltitude (f : Frustum) : ℝ :=
  sorry

/-- Theorem stating that for a specific frustum, the altitude of the smaller cone is 15. -/
theorem specific_frustum_smaller_cone_altitude :
  let f : Frustum := { altitude := 15, largerBaseArea := 64 * Real.pi, smallerBaseArea := 16 * Real.pi }
  smallerConeAltitude f = 15 := by
  sorry

end specific_frustum_smaller_cone_altitude_l2954_295420


namespace single_root_condition_l2954_295472

theorem single_root_condition (n : ℕ) (a : ℝ) (h : n > 1) :
  (∃! x : ℝ, (1 + x)^(1/n : ℝ) + (1 - x)^(1/n : ℝ) = a) ↔ a = 2 := by sorry

end single_root_condition_l2954_295472


namespace bike_ride_percentage_increase_l2954_295441

theorem bike_ride_percentage_increase (d1 d2 d3 : ℝ) : 
  d2 = 12 →                   -- Second hour distance is 12 miles
  d2 = 1.2 * d1 →             -- Second hour is 20% farther than first hour
  d1 + d2 + d3 = 37 →         -- Total distance is 37 miles
  (d3 - d2) / d2 * 100 = 25   -- Percentage increase from second to third hour is 25%
  := by sorry

end bike_ride_percentage_increase_l2954_295441


namespace function_through_points_l2954_295486

/-- Given a function f(x) = a^x - k passing through (1,3) and (0,2), prove f(x) = 2^x + 1 -/
theorem function_through_points (a k : ℝ) (f : ℝ → ℝ) 
    (h1 : ∀ x, f x = a^x - k) 
    (h2 : f 1 = 3) 
    (h3 : f 0 = 2) : 
    ∀ x, f x = 2^x + 1 := by
  sorry

end function_through_points_l2954_295486


namespace complex_number_quadrant_l2954_295404

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (-2 + Complex.I) ∧ z.re < 0 ∧ z.im < 0 := by
  sorry

end complex_number_quadrant_l2954_295404


namespace john_hotel_cost_l2954_295411

/-- Calculates the total cost of a hotel stay with a discount -/
def hotel_cost (nights : ℕ) (price_per_night : ℕ) (discount : ℕ) : ℕ :=
  nights * price_per_night - discount

/-- Proves that a 3-night stay at $250 per night with a $100 discount costs $650 -/
theorem john_hotel_cost : hotel_cost 3 250 100 = 650 := by
  sorry

end john_hotel_cost_l2954_295411


namespace binomial_square_constant_l2954_295429

theorem binomial_square_constant (c : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 84*x + c = (x + a)^2) → c = 1764 := by
  sorry

end binomial_square_constant_l2954_295429


namespace average_mark_is_76_l2954_295459

def marks : List ℝ := [80, 70, 60, 90, 80]

theorem average_mark_is_76 : (marks.sum / marks.length : ℝ) = 76 := by
  sorry

end average_mark_is_76_l2954_295459


namespace solution_implies_m_equals_three_l2954_295457

/-- Given that x = 2 and y = 1 is a solution to the equation x + my = 5, prove that m = 3 -/
theorem solution_implies_m_equals_three (x y m : ℝ) 
  (h1 : x = 2) 
  (h2 : y = 1) 
  (h3 : x + m * y = 5) : 
  m = 3 := by
  sorry

end solution_implies_m_equals_three_l2954_295457


namespace quadratic_properties_l2954_295473

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  pass_through_minus_one : a * (-1)^2 + b * (-1) + c = 0
  pass_through_zero : c = -1.5
  pass_through_one : a + b + c = -2
  pass_through_two : 4 * a + 2 * b + c = -1.5

theorem quadratic_properties (f : QuadraticFunction) :
  (∃ a' : ℝ, ∀ x, f.a * x^2 + f.b * x + f.c = a' * (x - 1)^2 - 2) ∧
  (f.a * 0^2 + f.b * 0 + f.c + 1.5 = 0 ∧ f.a * 2^2 + f.b * 2 + f.c + 1.5 = 0) :=
sorry

end quadratic_properties_l2954_295473


namespace no_unique_solution_l2954_295451

/-- The system of equations does not have a unique solution if and only if k = 3 -/
theorem no_unique_solution (k : ℝ) : 
  (∃ (x y : ℝ), 4 * (3 * x + 4 * y) = 48 ∧ k * x + 12 * y = 30) ∧ 
  ¬(∃! (x y : ℝ), 4 * (3 * x + 4 * y) = 48 ∧ k * x + 12 * y = 30) ↔ 
  k = 3 := by
sorry

end no_unique_solution_l2954_295451


namespace smallest_prime_perimeter_scalene_triangle_l2954_295463

/-- A function that checks if a number is prime --/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- A function that checks if three numbers form a scalene triangle --/
def isScaleneTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ b + c > a ∧ a + c > b

/-- The main theorem --/
theorem smallest_prime_perimeter_scalene_triangle :
  ∃ (a b c : ℕ),
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a > 3 ∧ b > 3 ∧ c > 3 ∧
    isScaleneTriangle a b c ∧
    isPrime (a + b + c) ∧
    (a + b + c = 23) ∧
    (∀ (x y z : ℕ),
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x > 3 ∧ y > 3 ∧ z > 3 ∧
      isScaleneTriangle x y z ∧
      isPrime (x + y + z) →
      x + y + z ≥ 23) :=
by sorry

end smallest_prime_perimeter_scalene_triangle_l2954_295463


namespace expression_equality_l2954_295418

theorem expression_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 2/y) :
  (x^2 - 1/x^2) * (y^2 + 1/y^2) = (x^4/4) - (4/x^4) + 3.75 := by
  sorry

end expression_equality_l2954_295418


namespace rectangular_box_with_spheres_l2954_295467

theorem rectangular_box_with_spheres (h : ℝ) : 
  let box_base : ℝ := 4
  let large_sphere_radius : ℝ := 2
  let small_sphere_radius : ℝ := 1
  let num_small_spheres : ℕ := 8
  h > 0 ∧ 
  box_base > 0 ∧
  large_sphere_radius > 0 ∧
  small_sphere_radius > 0 ∧
  num_small_spheres > 0 ∧
  (∃ (box : Set (ℝ × ℝ × ℝ)) (large_sphere : Set (ℝ × ℝ × ℝ)) (small_spheres : Finset (Set (ℝ × ℝ × ℝ))),
    -- Box properties
    (∀ (x y z : ℝ), (x, y, z) ∈ box ↔ 0 ≤ x ∧ x ≤ box_base ∧ 0 ≤ y ∧ y ≤ box_base ∧ 0 ≤ z ∧ z ≤ h) ∧
    -- Large sphere properties
    (∃ (cx cy cz : ℝ), large_sphere = {(x, y, z) | (x - cx)^2 + (y - cy)^2 + (z - cz)^2 ≤ large_sphere_radius^2}) ∧
    -- Small spheres properties
    (small_spheres.card = num_small_spheres) ∧
    (∀ s ∈ small_spheres, ∃ (cx cy cz : ℝ), s = {(x, y, z) | (x - cx)^2 + (y - cy)^2 + (z - cz)^2 ≤ small_sphere_radius^2}) ∧
    -- Tangency conditions
    (∀ s ∈ small_spheres, ∃ (face1 face2 face3 : Set (ℝ × ℝ × ℝ)), face1 ∪ face2 ∪ face3 ⊆ box ∧ s ∩ face1 ≠ ∅ ∧ s ∩ face2 ≠ ∅ ∧ s ∩ face3 ≠ ∅) ∧
    (∀ s ∈ small_spheres, large_sphere ∩ s ≠ ∅)) →
  h = 2 + 2 * Real.sqrt 7 := by
sorry

end rectangular_box_with_spheres_l2954_295467


namespace subtraction_absolute_value_l2954_295485

theorem subtraction_absolute_value : ∃ (x y : ℝ), 
  (|9 - 4| - |x - y| = 3) ∧ (|x - y| = 2) :=
by sorry

end subtraction_absolute_value_l2954_295485


namespace minimum_value_theorem_l2954_295458

theorem minimum_value_theorem (x y : ℝ) (h1 : 0 < x) (h2 : x < 1) (h3 : 0 < y) (h4 : y < 1) (h5 : x * y = 1/2) :
  (2 / (1 - x) + 1 / (1 - y)) ≥ 10 := by
sorry

end minimum_value_theorem_l2954_295458


namespace at_most_one_tiling_l2954_295461

/-- Represents a polyomino -/
structure Polyomino where
  squares : Set (ℕ × ℕ)
  nonempty : squares.Nonempty

/-- An L-shaped polyomino consisting of three squares -/
def l_shape : Polyomino := {
  squares := {(0,0), (0,1), (1,0)}
  nonempty := by simp
}

/-- Another polyomino with at least two squares -/
def other_polyomino : Polyomino := {
  squares := {(0,0), (0,1)}  -- Minimal example with two squares
  nonempty := by simp
}

/-- Represents a tiling of a board -/
def Tiling (n : ℕ) (p1 p2 : Polyomino) :=
  ∃ (t : Set (ℕ × ℕ × ℕ × ℕ)), 
    (∀ x y, x < n ∧ y < n → ∃ a b dx dy, (a, b, dx, dy) ∈ t ∧
      ((dx, dy) ∈ p1.squares ∨ (dx, dy) ∈ p2.squares) ∧
      (a + dx = x ∧ b + dy = y)) ∧
    (∀ (a b dx dy : ℕ), (a, b, dx, dy) ∈ t →
      (dx, dy) ∈ p1.squares ∨ (dx, dy) ∈ p2.squares)

/-- The main theorem -/
theorem at_most_one_tiling (n m : ℕ) (h : Nat.Coprime n m) :
  ¬(Tiling n l_shape other_polyomino ∧ Tiling m l_shape other_polyomino) := by
  sorry

end at_most_one_tiling_l2954_295461


namespace two_valid_m_values_l2954_295493

/-- A right triangle in the coordinate plane with legs parallel to the axes -/
structure RightTriangle where
  a : ℝ  -- x-coordinate of the point on the x-axis
  b : ℝ  -- y-coordinate of the point on the y-axis

/-- Check if the given m value satisfies the conditions for the right triangle -/
def satisfiesConditions (t : RightTriangle) (m : ℝ) : Prop :=
  3 * (t.a / 2) + 1 = 0 ∧  -- Condition for the line y = 3x + 1
  t.b / 2 = 2 ∧           -- Condition for the line y = mx + 2
  (t.b / 2) / (t.a / 2) = 4  -- Condition for the ratio of slopes

/-- The theorem stating that there are exactly two values of m that satisfy the conditions -/
theorem two_valid_m_values :
  ∃ m₁ m₂ : ℝ,
    m₁ ≠ m₂ ∧
    (∃ t : RightTriangle, satisfiesConditions t m₁) ∧
    (∃ t : RightTriangle, satisfiesConditions t m₂) ∧
    (∀ m : ℝ, (∃ t : RightTriangle, satisfiesConditions t m) → m = m₁ ∨ m = m₂) :=
  sorry

end two_valid_m_values_l2954_295493


namespace no_real_solutions_l2954_295428

theorem no_real_solutions : 
  ¬∃ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) :=
by sorry

end no_real_solutions_l2954_295428


namespace fifth_from_end_l2954_295413

-- Define the sequence
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + 11

-- Define the final term
def final_term (a : ℕ → ℕ) (final : ℕ) : Prop :=
  ∃ k : ℕ, a k = final ∧ ∀ n > k, a n > final

-- Theorem statement
theorem fifth_from_end (a : ℕ → ℕ) :
  arithmetic_sequence a →
  final_term a 89 →
  ∃ k : ℕ, a k = 45 ∧ a (k + 4) = 89 :=
by sorry

end fifth_from_end_l2954_295413


namespace fran_average_speed_l2954_295430

/-- Proves that given Joann's average speed and time, and Fran's riding time,
    Fran's required average speed to travel the same distance as Joann is 14 mph. -/
theorem fran_average_speed 
  (joann_speed : ℝ) 
  (joann_time : ℝ) 
  (fran_time : ℝ) 
  (h1 : joann_speed = 16) 
  (h2 : joann_time = 3.5) 
  (h3 : fran_time = 4) : 
  (joann_speed * joann_time) / fran_time = 14 := by
  sorry

end fran_average_speed_l2954_295430


namespace problem_statement_l2954_295479

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5 * a * b) :
  (∃ (x : ℝ), x ≥ a + b ∧ x ≥ 4/5) ∧
  (∀ (x : ℝ), x * a * b ≤ b^2 + 5*a → x ≤ 9) := by
sorry

end problem_statement_l2954_295479


namespace triangle_angle_sum_special_case_l2954_295412

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (angle_sum : A + B + C = π)
  (law_of_cosines : c^2 = a^2 + b^2 - 2*a*b*(Real.cos C))

-- State the theorem
theorem triangle_angle_sum_special_case (t : Triangle) 
  (h : (t.a + t.c - t.b) * (t.a + t.c + t.b) = 3 * t.a * t.c) : 
  t.A + t.C = 2 * π / 3 :=
sorry

end triangle_angle_sum_special_case_l2954_295412


namespace x_range_when_f_lg_x_gt_f_1_l2954_295445

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define the properties of f
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x y, 0 ≤ x → x < y → f y < f x

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem x_range_when_f_lg_x_gt_f_1 (heven : is_even f) (hdec : is_decreasing_on_nonneg f) :
  (∀ x, f (lg x) > f 1) → (∀ x, x > (1/10) ∧ x < 10) :=
sorry

end x_range_when_f_lg_x_gt_f_1_l2954_295445


namespace petty_cash_for_support_staff_bonus_l2954_295470

/-- Represents the number of staff members in each category -/
structure StaffCount where
  total : Nat
  administrative : Nat
  junior : Nat
  support : Nat

/-- Represents the daily bonus amounts for each staff category -/
structure DailyBonus where
  administrative : Nat
  junior : Nat
  support : Nat

/-- Represents the financial details of the bonus distribution -/
structure BonusDistribution where
  staff : StaffCount
  daily_bonus : DailyBonus
  bonus_days : Nat
  accountant_amount : Nat
  petty_cash_budget : Nat

/-- Calculates the amount needed from petty cash for support staff bonuses -/
def petty_cash_needed (bd : BonusDistribution) : Nat :=
  let total_bonus := bd.staff.administrative * bd.daily_bonus.administrative * bd.bonus_days +
                     bd.staff.junior * bd.daily_bonus.junior * bd.bonus_days +
                     bd.staff.support * bd.daily_bonus.support * bd.bonus_days
  total_bonus - bd.accountant_amount

/-- Theorem stating the amount needed from petty cash for support staff bonuses -/
theorem petty_cash_for_support_staff_bonus 
  (bd : BonusDistribution) 
  (h1 : bd.staff.total = 30)
  (h2 : bd.staff.administrative = 10)
  (h3 : bd.staff.junior = 10)
  (h4 : bd.staff.support = 10)
  (h5 : bd.daily_bonus.administrative = 100)
  (h6 : bd.daily_bonus.junior = 120)
  (h7 : bd.daily_bonus.support = 80)
  (h8 : bd.bonus_days = 30)
  (h9 : bd.accountant_amount = 85000)
  (h10 : bd.petty_cash_budget = 25000) :
  petty_cash_needed bd = 5000 := by
  sorry

end petty_cash_for_support_staff_bonus_l2954_295470


namespace min_speed_to_arrive_earlier_l2954_295488

/-- Proves the minimum speed required for the second person to arrive before the first person --/
theorem min_speed_to_arrive_earlier (distance : ℝ) (speed_A : ℝ) (delay : ℝ) :
  distance = 120 →
  speed_A = 30 →
  delay = 1.5 →
  ∀ speed_B : ℝ, speed_B > 48 → 
    distance / speed_B < distance / speed_A - delay := by
  sorry

end min_speed_to_arrive_earlier_l2954_295488


namespace line_perp_parallel_planes_l2954_295416

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perp : Line → Plane → Prop)

-- Define the parallel relation between two planes
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_parallel_planes 
  (l : Line) (α β : Plane) 
  (h1 : perp l β) (h2 : para α β) : 
  perp l α :=
sorry

end line_perp_parallel_planes_l2954_295416


namespace cube_sum_minus_triple_product_l2954_295456

theorem cube_sum_minus_triple_product (p : ℕ) : 
  Prime p → 
  ({(x, y) : ℕ × ℕ | x^3 + y^3 - 3*x*y = p - 1} = 
    if p = 2 then {(1, 0), (0, 1)} 
    else if p = 5 then {(2, 2)} 
    else ∅) := by
sorry

end cube_sum_minus_triple_product_l2954_295456


namespace smallest_with_20_divisors_l2954_295480

/-- The number of positive divisors of a positive integer -/
def num_divisors (n : ℕ+) : ℕ := sorry

/-- Check if a number has exactly 20 positive divisors -/
def has_20_divisors (n : ℕ+) : Prop := num_divisors n = 20

theorem smallest_with_20_divisors :
  ∃ (n : ℕ+), has_20_divisors n ∧ ∀ (m : ℕ+), has_20_divisors m → n ≤ m :=
  sorry

end smallest_with_20_divisors_l2954_295480


namespace problem_statement_l2954_295427

theorem problem_statement (t : ℝ) (x y : ℝ) 
  (h1 : x = 3 - 2*t) 
  (h2 : y = 5*t + 6) 
  (h3 : x = 1) : 
  y = 11 := by
  sorry

end problem_statement_l2954_295427


namespace vector_sum_inequality_l2954_295453

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]
variable (a b c d : V)

theorem vector_sum_inequality (h : a + b + c + d = 0) :
  ‖a‖ + ‖b‖ + ‖c‖ + ‖d‖ ≥ ‖a + d‖ + ‖b + d‖ + ‖c + d‖ := by
  sorry

end vector_sum_inequality_l2954_295453


namespace car_gasoline_usage_l2954_295497

/-- Calculates the amount of gasoline used by a car given its efficiency, speed, and travel time. -/
def gasoline_used (efficiency : Real) (speed : Real) (time : Real) : Real :=
  efficiency * speed * time

theorem car_gasoline_usage :
  let efficiency : Real := 0.14  -- liters per kilometer
  let speed : Real := 93.6       -- kilometers per hour
  let time : Real := 2.5         -- hours
  gasoline_used efficiency speed time = 32.76 := by
  sorry

end car_gasoline_usage_l2954_295497


namespace gcd_840_1764_l2954_295494

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end gcd_840_1764_l2954_295494


namespace selling_price_calculation_l2954_295487

/-- Given an article with a gain of $15 and a gain percentage of 20%,
    prove that the selling price is $90. -/
theorem selling_price_calculation (gain : ℝ) (gain_percentage : ℝ) :
  gain = 15 →
  gain_percentage = 20 →
  ∃ (cost_price selling_price : ℝ),
    gain = (gain_percentage / 100) * cost_price ∧
    selling_price = cost_price + gain ∧
    selling_price = 90 := by
  sorry

end selling_price_calculation_l2954_295487


namespace adjacent_supplementary_not_always_complementary_l2954_295434

-- Define supplementary angles
def supplementary (α β : Real) : Prop := α + β = 180

-- Define complementary angles
def complementary (α β : Real) : Prop := α + β = 90

-- Theorem statement
theorem adjacent_supplementary_not_always_complementary :
  ∃ α β : Real, supplementary α β ∧ ¬complementary α β :=
sorry

end adjacent_supplementary_not_always_complementary_l2954_295434


namespace triangle_angle_sum_l2954_295490

theorem triangle_angle_sum (A B C : ℝ) (h1 : A + B = 115) : 
  A + B + C = 180 → C = 65 := by
sorry

end triangle_angle_sum_l2954_295490


namespace paint_usage_l2954_295452

theorem paint_usage (total_paint : ℝ) (first_week_fraction : ℝ) (second_week_fraction : ℝ) 
  (h1 : total_paint = 360)
  (h2 : first_week_fraction = 1/4)
  (h3 : second_week_fraction = 1/6) :
  let first_week_usage := first_week_fraction * total_paint
  let remaining_paint := total_paint - first_week_usage
  let second_week_usage := second_week_fraction * remaining_paint
  first_week_usage + second_week_usage = 135 := by
sorry

end paint_usage_l2954_295452


namespace max_semicircle_intersections_l2954_295448

/-- Given n distinct points on a line, the maximum number of intersection points
    of semicircles drawn on one side of the line with these points as endpoints
    is equal to (n choose 4). -/
theorem max_semicircle_intersections (n : ℕ) : ℕ :=
  Nat.choose n 4

#check max_semicircle_intersections

end max_semicircle_intersections_l2954_295448


namespace inequality_proof_l2954_295481

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 1) (hy : y > 1) (hz : z > 1)
  (h : 1/x + 1/y + 1/z = 2) : 
  Real.sqrt (x + y + z) ≥ Real.sqrt (x - 1) + Real.sqrt (y - 1) + Real.sqrt (z - 1) := by
sorry

end inequality_proof_l2954_295481


namespace blackjack_payout_40_dollars_l2954_295455

/-- Calculates the total amount received for a blackjack bet -/
def blackjack_payout (bet : ℚ) (payout_ratio : ℚ × ℚ) : ℚ :=
  bet + bet * (payout_ratio.1 / payout_ratio.2)

/-- Theorem: The total amount received for a $40 blackjack bet with 3:2 payout is $100 -/
theorem blackjack_payout_40_dollars :
  blackjack_payout 40 (3, 2) = 100 := by
  sorry

end blackjack_payout_40_dollars_l2954_295455


namespace cube_root_equation_solution_l2954_295477

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x)^(1/3 : ℝ) = 4 :=
by
  -- The proof goes here
  sorry

end cube_root_equation_solution_l2954_295477


namespace congruence_solutions_count_l2954_295496

theorem congruence_solutions_count : 
  (Finset.filter (fun x : ℕ => 
    x > 0 ∧ x < 150 ∧ (x + 20) % 45 = 75 % 45) 
    (Finset.range 150)).card = 4 := by sorry

end congruence_solutions_count_l2954_295496


namespace solution_sets_l2954_295409

def f (a x : ℝ) : ℝ := a * x^2 + x - a

theorem solution_sets (a : ℝ) :
  (a = 1 → {x : ℝ | f 1 x > 1} = {x : ℝ | x > 1 ∨ x < -2}) ∧
  (a < 0 →
    (a < -1/2 → {x : ℝ | f a x > 1} = {x : ℝ | -(a + 1)/a < x ∧ x < 1}) ∧
    (a = -1/2 → {x : ℝ | f a x > 1} = {x : ℝ | x ≠ 1}) ∧
    (0 > a ∧ a > -1/2 → {x : ℝ | f a x > 1} = {x : ℝ | 1 < x ∧ x < -(a + 1)/a})) :=
by sorry

end solution_sets_l2954_295409


namespace max_rods_in_box_l2954_295426

/-- A rod with dimensions 1×1×4 -/
structure Rod :=
  (length : ℕ := 4)
  (width : ℕ := 1)
  (height : ℕ := 1)

/-- A cube-shaped box with dimensions 6×6×6 -/
structure Box :=
  (length : ℕ := 6)
  (width : ℕ := 6)
  (height : ℕ := 6)

/-- Predicate to check if a rod can be placed parallel to the box faces -/
def isParallel (r : Rod) (b : Box) : Prop :=
  (r.length ≤ b.length ∧ r.width ≤ b.width ∧ r.height ≤ b.height) ∨
  (r.length ≤ b.width ∧ r.width ≤ b.height ∧ r.height ≤ b.length) ∨
  (r.length ≤ b.height ∧ r.width ≤ b.length ∧ r.height ≤ b.width)

/-- The maximum number of rods that can fit in the box -/
def maxRods (r : Rod) (b : Box) : ℕ := 52

/-- Theorem stating that 52 is the maximum number of 1×1×4 rods that can fit in a 6×6×6 box -/
theorem max_rods_in_box (r : Rod) (b : Box) :
  isParallel r b → maxRods r b = 52 ∧ ¬∃ n : ℕ, n > 52 ∧ n * r.length * r.width * r.height ≤ b.length * b.width * b.height :=
sorry


end max_rods_in_box_l2954_295426


namespace dave_book_cost_l2954_295483

/-- The cost per book given the total number of books and total amount spent -/
def cost_per_book (total_books : ℕ) (total_spent : ℚ) : ℚ :=
  total_spent / total_books

theorem dave_book_cost :
  let total_books : ℕ := 8 + 6 + 3
  let total_spent : ℚ := 102
  cost_per_book total_books total_spent = 6 := by
  sorry

end dave_book_cost_l2954_295483


namespace sharp_composition_10_l2954_295460

def sharp (N : ℕ) : ℕ := N^2 - N + 2

theorem sharp_composition_10 : sharp (sharp (sharp 10)) = 70123304 := by
  sorry

end sharp_composition_10_l2954_295460


namespace throne_occupant_identity_l2954_295400

-- Define the possible species
inductive Species
| Human
| Monkey

-- Define the possible truth-telling nature
inductive Nature
| Knight
| Liar

-- Define the statement made by A
def statement (s : Species) (n : Nature) : Prop :=
  ¬(s = Species.Monkey ∧ n = Nature.Knight)

-- Theorem to prove
theorem throne_occupant_identity :
  ∃ (s : Species) (n : Nature),
    statement s n ∧
    (n = Nature.Knight → statement s n = True) ∧
    (n = Nature.Liar → statement s n = False) ∧
    s = Species.Human ∧
    n = Nature.Knight := by
  sorry

end throne_occupant_identity_l2954_295400


namespace picture_frame_length_l2954_295475

/-- Given a rectangular picture frame with height 12 inches and perimeter 44 inches, 
    prove that its length is 10 inches. -/
theorem picture_frame_length (height : ℝ) (perimeter : ℝ) (length : ℝ) : 
  height = 12 → perimeter = 44 → perimeter = 2 * (length + height) → length = 10 := by
  sorry

end picture_frame_length_l2954_295475


namespace total_people_in_tribes_l2954_295433

/-- Proves that the total number of people in two tribes is 378, given specific conditions about the number of women, men, and cannoneers. -/
theorem total_people_in_tribes (cannoneers : ℕ) (women : ℕ) (men : ℕ) : 
  cannoneers = 63 →
  women = 2 * cannoneers →
  men = 2 * women →
  cannoneers + women + men = 378 :=
by sorry

end total_people_in_tribes_l2954_295433


namespace largest_prime_divisor_of_factorial_sum_l2954_295492

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem largest_prime_divisor_of_factorial_sum : 
  (Nat.factors (factorial 13 + factorial 14)).maximum? = some 13 := by
  sorry

end largest_prime_divisor_of_factorial_sum_l2954_295492


namespace unique_amount_theorem_l2954_295406

def is_multiple_of_50 (n : ℕ) : Prop := ∃ k : ℕ, n = 50 * k

def min_banknotes (amount : ℕ) (max_denom : ℕ) : ℕ :=
  sorry

theorem unique_amount_theorem (amount : ℕ) : 
  is_multiple_of_50 amount →
  min_banknotes amount 5000 ≥ 15 →
  min_banknotes amount 1000 ≥ 35 →
  amount = 29950 :=
sorry

end unique_amount_theorem_l2954_295406


namespace lawrence_county_camp_attendance_l2954_295499

/-- The number of kids from Lawrence county who go to camp -/
def kids_at_camp (total : ℕ) (stay_home : ℕ) : ℕ :=
  total - stay_home

/-- Proof that 610769 kids from Lawrence county go to camp -/
theorem lawrence_county_camp_attendance :
  kids_at_camp 1201565 590796 = 610769 := by
  sorry

end lawrence_county_camp_attendance_l2954_295499


namespace fourth_root_equation_l2954_295491

theorem fourth_root_equation (m : ℝ) : (m^4)^(1/4) = 2 → m = 2 ∨ m = -2 := by
  sorry

end fourth_root_equation_l2954_295491


namespace right_triangle_area_l2954_295414

theorem right_triangle_area (a b c : ℝ) (h : a > 0) : 
  a * a = 2 * b * b →  -- 45-45-90 triangle condition
  b = 4 →              -- altitude to hypotenuse is 4
  c = a / 2 →          -- c is half of hypotenuse
  (1/2) * a * b = 8 * Real.sqrt 2 := by
  sorry

end right_triangle_area_l2954_295414


namespace max_x_minus_y_on_circle_l2954_295431

theorem max_x_minus_y_on_circle (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end max_x_minus_y_on_circle_l2954_295431


namespace geometric_sequence_ratio_sum_l2954_295423

theorem geometric_sequence_ratio_sum (k p r : ℝ) (h1 : p ≠ r) (h2 : k ≠ 0) :
  k * p^2 - k * r^2 = 4 * (k * p - k * r) → p + r = 4 := by
  sorry

end geometric_sequence_ratio_sum_l2954_295423


namespace compound_interest_rate_l2954_295419

theorem compound_interest_rate (P : ℝ) (r : ℝ) : 
  P * (1 + r/100)^2 = 3650 ∧ 
  P * (1 + r/100)^3 = 4015 → 
  r = 10 := by
  sorry

end compound_interest_rate_l2954_295419


namespace min_value_T_l2954_295407

/-- Given a quadratic inequality (1/a)x² + bx + c ≤ 0 with solution set ℝ and b > 0,
    the minimum value of T = (5 + 2ab + 4ac) / (ab + 1) is 4 -/
theorem min_value_T (a b c : ℝ) (h1 : b > 0) 
  (h2 : ∀ x, (1/a) * x^2 + b * x + c ≤ 0) : 
  (∀ t, t = (5 + 2*a*b + 4*a*c) / (a*b + 1) → t ≥ 4) ∧ 
  (∃ t, t = (5 + 2*a*b + 4*a*c) / (a*b + 1) ∧ t = 4) :=
by sorry

end min_value_T_l2954_295407


namespace pure_imaginary_square_l2954_295462

theorem pure_imaginary_square (a : ℝ) : 
  (Complex.I * ((1 : ℂ) + a * Complex.I)^2).re = 0 → a = 1 ∨ a = -1 := by
  sorry

end pure_imaginary_square_l2954_295462


namespace line_through_points_with_slope_l2954_295476

theorem line_through_points_with_slope (k : ℝ) : 
  (∃ (m : ℝ), m = (3 * k - (-9)) / (7 - k) ∧ m = 2 * k) → 
  k = 9 / 2 ∨ k = 1 := by
  sorry

end line_through_points_with_slope_l2954_295476


namespace Q_zeros_count_l2954_295446

noncomputable def Q (x : ℝ) : ℂ :=
  2 + Complex.exp (Complex.I * x) - 2 * Complex.exp (2 * Complex.I * x) + Complex.exp (3 * Complex.I * x)

theorem Q_zeros_count : ∃! (s : Finset ℝ), s.card = 2 ∧ (∀ x ∈ s, 0 ≤ x ∧ x < 4 * Real.pi ∧ Q x = 0) ∧ (∀ x, 0 ≤ x → x < 4 * Real.pi → Q x = 0 → x ∈ s) := by
  sorry

end Q_zeros_count_l2954_295446


namespace twitter_to_insta_fb_ratio_l2954_295438

/-- Represents the number of followers on each social media platform -/
structure Followers where
  instagram : ℕ
  facebook : ℕ
  twitter : ℕ
  tiktok : ℕ
  youtube : ℕ

/-- Conditions for Malcolm's social media followers -/
def malcolm_followers (f : Followers) : Prop :=
  f.instagram = 240 ∧
  f.facebook = 500 ∧
  f.tiktok = 3 * f.twitter ∧
  f.youtube = f.tiktok + 510 ∧
  f.instagram + f.facebook + f.twitter + f.tiktok + f.youtube = 3840

/-- Theorem stating the ratio of Twitter followers to Instagram and Facebook followers -/
theorem twitter_to_insta_fb_ratio (f : Followers) 
  (h : malcolm_followers f) : 
  f.twitter * 2 = f.instagram + f.facebook := by
  sorry

end twitter_to_insta_fb_ratio_l2954_295438


namespace cube_with_holes_volume_l2954_295468

/-- The volume of a cube with holes drilled through it -/
theorem cube_with_holes_volume :
  let cube_edge : ℝ := 3
  let hole_side : ℝ := 1
  let cube_volume := cube_edge ^ 3
  let hole_volume := hole_side ^ 2 * cube_edge
  let num_hole_pairs := 3
  cube_volume - (num_hole_pairs * hole_volume) = 18 := by
  sorry

end cube_with_holes_volume_l2954_295468


namespace min_balls_theorem_l2954_295442

/-- Represents the number of balls of each color in the box -/
structure BallCounts where
  red : Nat
  green : Nat
  yellow : Nat
  blue : Nat
  white : Nat
  black : Nat

/-- The total number of balls in the box -/
def total_balls (counts : BallCounts) : Nat :=
  counts.red + counts.green + counts.yellow + counts.blue + counts.white + counts.black

/-- The minimum number of balls to draw to ensure at least n are of the same color -/
def min_balls_to_draw (counts : BallCounts) (n : Nat) : Nat :=
  sorry

theorem min_balls_theorem (counts : BallCounts) (n : Nat) :
  counts.red = 28 →
  counts.green = 20 →
  counts.yellow = 12 →
  counts.blue = 20 →
  counts.white = 10 →
  counts.black = 10 →
  total_balls counts = 100 →
  min_balls_to_draw counts 15 = 75 :=
by sorry

end min_balls_theorem_l2954_295442


namespace grandmothers_gift_amount_l2954_295449

theorem grandmothers_gift_amount (num_grandchildren : ℕ) (cards_per_year : ℕ) (money_per_card : ℕ) : 
  num_grandchildren = 3 → cards_per_year = 2 → money_per_card = 80 →
  num_grandchildren * cards_per_year * money_per_card = 480 := by
  sorry

end grandmothers_gift_amount_l2954_295449
