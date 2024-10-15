import Mathlib

namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3894_389499

theorem trigonometric_equation_solution (t : ℝ) : 
  (2 * (Real.sin (2 * t))^5 - (Real.sin (2 * t))^3 - 6 * (Real.sin (2 * t))^2 + 3 = 0) ↔ 
  (∃ k : ℤ, t = (π / 8) * (2 * ↑k + 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3894_389499


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3894_389414

theorem absolute_value_inequality (x : ℝ) : 
  ‖‖x - 2‖ - 1‖ ≤ 1 ↔ 0 ≤ x ∧ x ≤ 4 := by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3894_389414


namespace NUMINAMATH_CALUDE_equal_distance_to_axes_l3894_389467

theorem equal_distance_to_axes (m : ℝ) : 
  let M : ℝ × ℝ := (-3*m - 1, -2*m)
  (|M.1| = |M.2|) ↔ (m = -1/5 ∨ m = -1) := by
sorry

end NUMINAMATH_CALUDE_equal_distance_to_axes_l3894_389467


namespace NUMINAMATH_CALUDE_divisibility_of_sixth_power_difference_l3894_389430

theorem divisibility_of_sixth_power_difference (a b : ℤ) 
  (ha : ¬ 3 ∣ a) (hb : ¬ 3 ∣ b) : 
  9 ∣ (a^6 - b^6) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_of_sixth_power_difference_l3894_389430


namespace NUMINAMATH_CALUDE_unique_solution_for_star_equation_l3894_389421

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := 5*x - 4*y + 2*x*y

-- State the theorem
theorem unique_solution_for_star_equation :
  ∃! y : ℝ, star 4 y = 16 := by sorry

end NUMINAMATH_CALUDE_unique_solution_for_star_equation_l3894_389421


namespace NUMINAMATH_CALUDE_rectangular_playground_vertical_length_l3894_389400

/-- The vertical length of a rectangular playground given specific conditions -/
theorem rectangular_playground_vertical_length :
  ∀ (square_side : ℝ) (rect_horizontal : ℝ) (rect_vertical : ℝ),
    square_side = 12 →
    rect_horizontal = 9 →
    4 * square_side = 2 * (rect_horizontal + rect_vertical) →
    rect_vertical = 15 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_playground_vertical_length_l3894_389400


namespace NUMINAMATH_CALUDE_intersection_x_coordinate_l3894_389431

-- Define the equations of the two lines
def line1 (x y : ℝ) : Prop := y = 3 * x - 17
def line2 (x y : ℝ) : Prop := 3 * x + y = 103

-- Theorem stating that the x-coordinate of the intersection is 20
theorem intersection_x_coordinate :
  ∃ (x y : ℝ), line1 x y ∧ line2 x y ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_intersection_x_coordinate_l3894_389431


namespace NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_prism_l3894_389496

theorem sphere_volume_circumscribing_rectangular_prism :
  let edge1 : ℝ := 1
  let edge2 : ℝ := Real.sqrt 10
  let edge3 : ℝ := 5
  let space_diagonal : ℝ := Real.sqrt (edge1^2 + edge2^2 + edge3^2)
  let sphere_radius : ℝ := space_diagonal / 2
  let sphere_volume : ℝ := (4 / 3) * Real.pi * sphere_radius^3
  sphere_volume = 36 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_circumscribing_rectangular_prism_l3894_389496


namespace NUMINAMATH_CALUDE_greatest_lower_bound_reciprocal_sum_l3894_389454

theorem greatest_lower_bound_reciprocal_sum (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (1 / a + 1 / b ≥ 4) ∧ ∀ m > 4, ∃ a b, 0 < a ∧ 0 < b ∧ a + b = 1 ∧ 1 / a + 1 / b < m :=
sorry

end NUMINAMATH_CALUDE_greatest_lower_bound_reciprocal_sum_l3894_389454


namespace NUMINAMATH_CALUDE_reciprocal_of_repeating_three_l3894_389428

def repeating_decimal_to_fraction (d : ℚ) : ℚ :=
  if d ≥ 0 ∧ d < 1 then d / (1 - (10 * d - ⌊10 * d⌋)) else d

theorem reciprocal_of_repeating_three : 
  (repeating_decimal_to_fraction (1/3 : ℚ))⁻¹ = 3 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_repeating_three_l3894_389428


namespace NUMINAMATH_CALUDE_frequency_of_a_l3894_389490

def sentence : String := "Happy Teachers'Day!"

theorem frequency_of_a (s : String) (h : s = sentence) : 
  (s.toList.filter (· = 'a')).length = 3 := by sorry

end NUMINAMATH_CALUDE_frequency_of_a_l3894_389490


namespace NUMINAMATH_CALUDE_sum_after_decrease_l3894_389469

theorem sum_after_decrease (a b : ℤ) : 
  a + b = 100 → (a - 48) + b = 52 := by
  sorry

end NUMINAMATH_CALUDE_sum_after_decrease_l3894_389469


namespace NUMINAMATH_CALUDE_sams_income_l3894_389483

/-- Represents the income tax calculation for Sam's region -/
noncomputable def income_tax (q : ℝ) (income : ℝ) : ℝ :=
  0.01 * q * 30000 +
  0.01 * (q + 3) * (min 45000 (max 30000 income) - 30000) +
  0.01 * (q + 5) * (max 0 (income - 45000))

/-- Theorem stating Sam's annual income given the tax structure -/
theorem sams_income (q : ℝ) :
  ∃ (income : ℝ),
    income_tax q income = 0.01 * (q + 0.35) * income ∧
    income = 48376 :=
by sorry

end NUMINAMATH_CALUDE_sams_income_l3894_389483


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3894_389470

theorem trigonometric_identity (x : ℝ) :
  (1 / Real.cos (2022 * x) + Real.tan (2022 * x) = 1 / 2022) →
  (1 / Real.cos (2022 * x) - Real.tan (2022 * x) = 2022) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3894_389470


namespace NUMINAMATH_CALUDE_samantha_route_count_l3894_389478

/-- Represents the number of ways to arrange k items out of n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of routes from Samantha's house to the southwest corner of City Park -/
def routes_to_park : ℕ := binomial 4 1

/-- The number of routes through City Park -/
def routes_through_park : ℕ := 1

/-- The number of routes from the northeast corner of City Park to school -/
def routes_to_school : ℕ := binomial 6 3

/-- The total number of possible routes Samantha can take -/
def total_routes : ℕ := routes_to_park * routes_through_park * routes_to_school

theorem samantha_route_count : total_routes = 80 := by sorry

end NUMINAMATH_CALUDE_samantha_route_count_l3894_389478


namespace NUMINAMATH_CALUDE_barycentric_coords_proportional_to_areas_l3894_389412

-- Define a triangle ABC
variable (A B C : ℝ × ℝ)

-- Define a point P inside the triangle
variable (P : ℝ × ℝ)

-- Define the area function
noncomputable def area (X Y Z : ℝ × ℝ) : ℝ := sorry

-- Define the barycentric coordinates
def barycentric_coords (P A B C : ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

-- State the theorem
theorem barycentric_coords_proportional_to_areas :
  ∃ (k : ℝ), k ≠ 0 ∧ 
    barycentric_coords P A B C = 
      (k * area P B C, k * area P C A, k * area P A B) := by sorry

end NUMINAMATH_CALUDE_barycentric_coords_proportional_to_areas_l3894_389412


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3894_389436

theorem fraction_equivalence 
  (a b d k : ℝ) 
  (h1 : d ≠ 0) 
  (h2 : k ≠ 0) : 
  (∀ x, (a * (k * x) + b) / (a * (k * x) + d) = (b * (k * x)) / (d * (k * x))) ↔ b = d :=
sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3894_389436


namespace NUMINAMATH_CALUDE_min_ab_value_l3894_389474

-- Define the triangle ABC
def triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

-- Define the condition 2c * cos B = 2a + b
def condition1 (a b c : ℝ) (B : ℝ) : Prop :=
  2 * c * Real.cos B = 2 * a + b

-- Define the area condition S = (√3/2) * c
def condition2 (c : ℝ) (S : ℝ) : Prop :=
  S = (Real.sqrt 3 / 2) * c

-- Theorem statement
theorem min_ab_value (a b c : ℝ) (B : ℝ) (S : ℝ) :
  triangle a b c →
  condition1 a b c B →
  condition2 c S →
  a * b ≥ 12 :=
sorry

end NUMINAMATH_CALUDE_min_ab_value_l3894_389474


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3894_389485

theorem square_plus_reciprocal_square (a : ℝ) (h : (a + 1/a)^4 = 5) :
  a^2 + 1/a^2 = Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3894_389485


namespace NUMINAMATH_CALUDE_females_wearing_glasses_l3894_389425

/-- In a town with a given population, number of males, and percentage of females wearing glasses,
    calculate the number of females wearing glasses. -/
theorem females_wearing_glasses
  (total_population : ℕ)
  (males : ℕ)
  (female_glasses_percentage : ℚ)
  (h1 : total_population = 5000)
  (h2 : males = 2000)
  (h3 : female_glasses_percentage = 30 / 100) :
  (total_population - males) * female_glasses_percentage = 900 := by
sorry

end NUMINAMATH_CALUDE_females_wearing_glasses_l3894_389425


namespace NUMINAMATH_CALUDE_arithmetic_sequence_a7_l3894_389489

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a7 (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 + a 13 = 12 → a 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_a7_l3894_389489


namespace NUMINAMATH_CALUDE_new_manufacturing_cost_l3894_389473

/-- Given a constant selling price, an initial manufacturing cost, and profit percentages,
    calculate the new manufacturing cost after a change in profit percentage. -/
theorem new_manufacturing_cost
  (P : ℝ)  -- Selling price
  (initial_cost : ℝ)  -- Initial manufacturing cost
  (initial_profit_percent : ℝ)  -- Initial profit as a percentage of selling price
  (new_profit_percent : ℝ)  -- New profit as a percentage of selling price
  (h1 : initial_cost = 80)  -- Initial cost is $80
  (h2 : initial_profit_percent = 0.20)  -- Initial profit is 20%
  (h3 : new_profit_percent = 0.50)  -- New profit is 50%
  (h4 : P - initial_cost = initial_profit_percent * P)  -- Initial profit equation
  : P - new_profit_percent * P = 50 := by
  sorry


end NUMINAMATH_CALUDE_new_manufacturing_cost_l3894_389473


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3894_389466

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- The number 188 million -/
def number : ℝ := 188000000

/-- The scientific notation representation of 188 million -/
def scientificForm : ScientificNotation :=
  { coefficient := 1.88
    exponent := 8
    coeff_range := by sorry }

theorem scientific_notation_correct :
  number = scientificForm.coefficient * (10 : ℝ) ^ scientificForm.exponent := by sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3894_389466


namespace NUMINAMATH_CALUDE_laura_weekly_driving_distance_l3894_389450

/-- Calculates the total miles driven by Laura per week -/
def total_miles_per_week (
  house_to_school_round_trip : ℕ)
  (supermarket_extra_distance : ℕ)
  (school_trips_per_week : ℕ)
  (supermarket_trips_per_week : ℕ) : ℕ :=
  let school_miles := house_to_school_round_trip * school_trips_per_week
  let supermarket_round_trip := house_to_school_round_trip + 2 * supermarket_extra_distance
  let supermarket_miles := supermarket_round_trip * supermarket_trips_per_week
  school_miles + supermarket_miles

/-- Laura's weekly driving distance theorem -/
theorem laura_weekly_driving_distance :
  total_miles_per_week 20 10 5 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_laura_weekly_driving_distance_l3894_389450


namespace NUMINAMATH_CALUDE_min_lcm_ac_l3894_389465

theorem min_lcm_ac (a b c : ℕ+) (h1 : Nat.lcm a b = 18) (h2 : Nat.lcm b c = 28) :
  ∃ (a' c' : ℕ+), Nat.lcm a' c' = 126 ∧ 
    (∀ (x y : ℕ+), Nat.lcm x b = 18 → Nat.lcm b y = 28 → Nat.lcm a' c' ≤ Nat.lcm x y) :=
by sorry

end NUMINAMATH_CALUDE_min_lcm_ac_l3894_389465


namespace NUMINAMATH_CALUDE_average_daily_sales_l3894_389401

/-- Given the sales of pens over a 13-day period, calculate the average daily sales. -/
theorem average_daily_sales (day1_sales : ℕ) (other_days_sales : ℕ) (num_other_days : ℕ) : 
  day1_sales = 96 →
  other_days_sales = 44 →
  num_other_days = 12 →
  (day1_sales + num_other_days * other_days_sales) / (num_other_days + 1) = 48 :=
by
  sorry

#check average_daily_sales

end NUMINAMATH_CALUDE_average_daily_sales_l3894_389401


namespace NUMINAMATH_CALUDE_max_books_borrowed_l3894_389405

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 25)
  (h2 : zero_books = 3)
  (h3 : one_book = 11)
  (h4 : two_books = 6)
  (h5 : (total_students : ℚ) * 2 = (zero_books * 0 + one_book * 1 + two_books * 2 + 
    (total_students - zero_books - one_book - two_books) * 3 + 
    (total_students * 2 - zero_books * 0 - one_book * 1 - two_books * 2 - 
    (total_students - zero_books - one_book - two_books) * 3))) :
  ∃ (max_books : ℕ), max_books = 15 ∧ 
    max_books ≤ total_students * 2 - zero_books * 0 - one_book * 1 - two_books * 2 - 
    (total_students - zero_books - one_book - two_books - 1) * 3 :=
by sorry

end NUMINAMATH_CALUDE_max_books_borrowed_l3894_389405


namespace NUMINAMATH_CALUDE_circle_family_properties_l3894_389487

-- Define the family of circles
def circle_family (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*a*x + 2*a*y + 20*a - 20 = 0

-- Define the fixed circle
def fixed_circle (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

theorem circle_family_properties :
  (∀ a : ℝ, circle_family a 4 (-2)) ∧ 
  (circle_family (1 + Real.sqrt 5 / 5) = fixed_circle) ∧
  (circle_family (1 - Real.sqrt 5 / 5) = fixed_circle) :=
sorry

end NUMINAMATH_CALUDE_circle_family_properties_l3894_389487


namespace NUMINAMATH_CALUDE_paper_fold_distance_l3894_389460

theorem paper_fold_distance (area : ℝ) (h_area : area = 18) : ∃ (distance : ℝ), distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_paper_fold_distance_l3894_389460


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_eight_satisfies_conditions_ninety_eight_is_largest_l3894_389448

theorem largest_integer_with_remainder : 
  ∀ n : ℕ, n < 100 ∧ n % 6 = 2 → n ≤ 98 :=
by
  sorry

theorem ninety_eight_satisfies_conditions : 
  98 < 100 ∧ 98 % 6 = 2 :=
by
  sorry

theorem ninety_eight_is_largest :
  ∀ n : ℕ, n < 100 ∧ n % 6 = 2 → n ≤ 98 ∧ 
  ∃ m : ℕ, m < 100 ∧ m % 6 = 2 ∧ m = 98 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_ninety_eight_satisfies_conditions_ninety_eight_is_largest_l3894_389448


namespace NUMINAMATH_CALUDE_probability_less_than_20_l3894_389491

theorem probability_less_than_20 (total : ℕ) (more_than_30 : ℕ) (h1 : total = 160) (h2 : more_than_30 = 90) :
  let less_than_20 := total - more_than_30
  (less_than_20 : ℚ) / total = 7 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_20_l3894_389491


namespace NUMINAMATH_CALUDE_emily_candy_consumption_l3894_389408

/-- Emily's Halloween candy problem -/
theorem emily_candy_consumption (neighbor_candy : ℕ) (sister_candy : ℕ) (days : ℕ) 
  (h1 : neighbor_candy = 5)
  (h2 : sister_candy = 13)
  (h3 : days = 2) :
  (neighbor_candy + sister_candy) / days = 9 := by
  sorry

end NUMINAMATH_CALUDE_emily_candy_consumption_l3894_389408


namespace NUMINAMATH_CALUDE_intersection_point_sum_l3894_389406

theorem intersection_point_sum (a b : ℝ) : 
  (∃ x y : ℝ, x = (1/3) * y + a ∧ y = (1/3) * x + b ∧ x = 3 ∧ y = 3) → 
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_sum_l3894_389406


namespace NUMINAMATH_CALUDE_purchase_cost_l3894_389452

/-- The cost of a hamburger in dollars -/
def hamburger_cost : ℕ := 4

/-- The cost of a milkshake in dollars -/
def milkshake_cost : ℕ := 3

/-- The number of hamburgers purchased -/
def num_hamburgers : ℕ := 7

/-- The number of milkshakes purchased -/
def num_milkshakes : ℕ := 6

/-- The total cost of the purchase -/
def total_cost : ℕ := hamburger_cost * num_hamburgers + milkshake_cost * num_milkshakes

theorem purchase_cost : total_cost = 46 := by
  sorry

end NUMINAMATH_CALUDE_purchase_cost_l3894_389452


namespace NUMINAMATH_CALUDE_range_of_a_l3894_389435

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Ioo 0 2, (a - a^2) * (x^2 + 1) + x ≤ 0) ↔ 
  a ∈ Set.Iic ((1 - Real.sqrt 3) / 2) ∪ Set.Ici ((1 + Real.sqrt 3) / 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3894_389435


namespace NUMINAMATH_CALUDE_jeans_pricing_l3894_389495

theorem jeans_pricing (manufacturing_cost : ℝ) (manufacturing_cost_pos : manufacturing_cost > 0) :
  let retail_price := manufacturing_cost * (1 + 0.4)
  let customer_price := retail_price * (1 + 0.1)
  (customer_price - manufacturing_cost) / manufacturing_cost = 0.54 := by
sorry

end NUMINAMATH_CALUDE_jeans_pricing_l3894_389495


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3894_389411

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0 ∧ a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 4 + a^(x - 1)
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3894_389411


namespace NUMINAMATH_CALUDE_intersection_A_B_l3894_389423

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x | |x - 1| ≤ 1}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3894_389423


namespace NUMINAMATH_CALUDE_school_ratio_problem_l3894_389422

theorem school_ratio_problem (total_students : ℕ) (boys_percentage : ℚ) 
  (represented_students : ℕ) (h1 : total_students = 140) 
  (h2 : boys_percentage = 1/2) (h3 : represented_students = 98) : 
  (represented_students : ℚ) / (boys_percentage * total_students) = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_school_ratio_problem_l3894_389422


namespace NUMINAMATH_CALUDE_allocation_ways_l3894_389475

theorem allocation_ways (n : ℕ) (k : ℕ) (h : n = 8 ∧ k = 4) : k^n = 65536 := by
  sorry

end NUMINAMATH_CALUDE_allocation_ways_l3894_389475


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3894_389443

/-- 
A quadratic equation x^2 + 3x - k = 0 has two equal real roots if and only if k = -9/4.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 3*x - k = 0 ∧ (∀ y : ℝ, y^2 + 3*y - k = 0 → y = x)) ↔ k = -9/4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3894_389443


namespace NUMINAMATH_CALUDE_train_journey_time_l3894_389416

theorem train_journey_time (S : ℝ) (x : ℝ) (h1 : x > 0) (h2 : S > 0) :
  (S / (2 * x) + S / (2 * 0.75 * x)) - S / x = 0.5 →
  S / x + 0.5 = 3.5 := by
sorry

end NUMINAMATH_CALUDE_train_journey_time_l3894_389416


namespace NUMINAMATH_CALUDE_function_properties_l3894_389404

def f (x : ℝ) : ℝ := x^3 + x^2 + x + 1

theorem function_properties : 
  f 0 = 1 ∧ 
  f (-1) = 0 ∧ 
  ∃ ε > 0, |f 1 - 4| < ε := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3894_389404


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3894_389476

/-- A rectangular field with given area and width has a specific perimeter -/
theorem rectangle_perimeter (area width : ℝ) (h_area : area = 750) (h_width : width = 25) :
  2 * (area / width + width) = 110 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3894_389476


namespace NUMINAMATH_CALUDE_uniform_random_transformation_l3894_389488

theorem uniform_random_transformation (b₁ : ℝ) (b : ℝ) :
  (∀ x ∈ Set.Icc 0 1, b₁ ∈ Set.Icc 0 1) →
  b = (b₁ - 2) * 3 →
  (∀ y ∈ Set.Icc (-6) (-3), b ∈ Set.Icc (-6) (-3)) :=
by sorry

end NUMINAMATH_CALUDE_uniform_random_transformation_l3894_389488


namespace NUMINAMATH_CALUDE_distance_to_pole_l3894_389432

def polar_distance (ρ : ℝ) (θ : ℝ) : ℝ := ρ

theorem distance_to_pole (A : ℝ × ℝ) (h : A = (3, -4)) :
  polar_distance A.1 A.2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_pole_l3894_389432


namespace NUMINAMATH_CALUDE_sqrt_3_plus_2_times_sqrt_3_minus_2_l3894_389456

theorem sqrt_3_plus_2_times_sqrt_3_minus_2 : (Real.sqrt 3 + 2) * (Real.sqrt 3 - 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_3_plus_2_times_sqrt_3_minus_2_l3894_389456


namespace NUMINAMATH_CALUDE_max_puzzles_in_club_l3894_389444

/-- Represents a math club with members solving puzzles -/
structure MathClub where
  members : ℕ
  average_puzzles : ℕ
  min_puzzles : ℕ

/-- Calculates the maximum number of puzzles one member can solve -/
def max_puzzles_by_one (club : MathClub) : ℕ :=
  club.members * club.average_puzzles - (club.members - 1) * club.min_puzzles

/-- Theorem stating the maximum number of puzzles solved by one member in the given conditions -/
theorem max_puzzles_in_club (club : MathClub) 
  (h_members : club.members = 40)
  (h_average : club.average_puzzles = 6)
  (h_min : club.min_puzzles = 2) :
  max_puzzles_by_one club = 162 := by
  sorry

#eval max_puzzles_by_one ⟨40, 6, 2⟩

end NUMINAMATH_CALUDE_max_puzzles_in_club_l3894_389444


namespace NUMINAMATH_CALUDE_horner_method_for_f_l3894_389482

def horner_polynomial (a : List ℝ) (x : ℝ) : ℝ :=
  a.foldl (fun acc coeff => acc * x + coeff) 0

def f (x : ℝ) : ℝ := 5 * x^6 + 3 * x^4 + 2 * x + 1

theorem horner_method_for_f :
  f 2 = horner_polynomial [1, 2, 0, 3, 0, 0, 5] 2 ∧ 
  horner_polynomial [1, 2, 0, 3, 0, 0, 5] 2 = 373 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_l3894_389482


namespace NUMINAMATH_CALUDE_arc_length_150_degrees_l3894_389407

/-- The arc length of a circle with radius 1 cm and central angle 150° is (5π/6) cm. -/
theorem arc_length_150_degrees : 
  let radius : ℝ := 1
  let central_angle_degrees : ℝ := 150
  let central_angle_radians : ℝ := central_angle_degrees * (π / 180)
  let arc_length : ℝ := radius * central_angle_radians
  arc_length = (5 * π) / 6 := by sorry

end NUMINAMATH_CALUDE_arc_length_150_degrees_l3894_389407


namespace NUMINAMATH_CALUDE_tetrahedron_triangles_l3894_389484

/-- The number of vertices in a regular tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a regular tetrahedron -/
def distinct_triangles : ℕ := Nat.choose tetrahedron_vertices triangle_vertices

theorem tetrahedron_triangles :
  distinct_triangles = 4 :=
sorry

end NUMINAMATH_CALUDE_tetrahedron_triangles_l3894_389484


namespace NUMINAMATH_CALUDE_evaluate_expression_l3894_389471

theorem evaluate_expression (x y z : ℚ) 
  (hx : x = 1/4) 
  (hy : y = 3/4) 
  (hz : z = -2) : 
  x^3 * y^2 * z^2 = 9/16 := by
sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3894_389471


namespace NUMINAMATH_CALUDE_stating_club_truncator_probability_l3894_389493

/-- The number of matches played by Club Truncator -/
def num_matches : ℕ := 8

/-- The probability of winning, losing, or tying a single match -/
def single_match_prob : ℚ := 1/3

/-- The probability of finishing with more wins than losses -/
def more_wins_prob : ℚ := 2741/6561

/-- 
Theorem stating that given 8 matches where the probability of winning, 
losing, or tying each match is 1/3, the probability of finishing with 
more wins than losses is 2741/6561.
-/
theorem club_truncator_probability : 
  (num_matches = 8) → 
  (single_match_prob = 1/3) → 
  (more_wins_prob = 2741/6561) :=
by sorry

end NUMINAMATH_CALUDE_stating_club_truncator_probability_l3894_389493


namespace NUMINAMATH_CALUDE_starting_number_proof_l3894_389458

def has_two_fives (n : ℕ) : Prop :=
  (n / 10 = 5 ∧ n % 10 = 5) ∨ (n / 100 = 5 ∧ n % 100 / 10 = 5)

theorem starting_number_proof :
  ∀ (start : ℕ),
    start ≤ 54 →
    (∃! n : ℕ, start ≤ n ∧ n ≤ 50 ∧ has_two_fives n) →
    start = 54 :=
by sorry

end NUMINAMATH_CALUDE_starting_number_proof_l3894_389458


namespace NUMINAMATH_CALUDE_sqrt_difference_sum_l3894_389420

theorem sqrt_difference_sum (x : ℝ) : 
  Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4 →
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_sum_l3894_389420


namespace NUMINAMATH_CALUDE_king_arthur_table_seats_l3894_389418

/-- Represents a circular seating arrangement -/
structure CircularArrangement where
  size : ℕ
  opposite : ℕ → ℕ
  opposite_symmetric : ∀ n, n ≤ size → opposite (opposite n) = n

/-- The specific circular arrangement described in the problem -/
def kingArthurTable : CircularArrangement where
  size := 38
  opposite := fun n => (n + 19) % 38
  opposite_symmetric := sorry

theorem king_arthur_table_seats :
  ∃ (t : CircularArrangement), t.size = 38 ∧ t.opposite 10 = 29 := by
  use kingArthurTable
  constructor
  · rfl
  · rfl

#check king_arthur_table_seats

end NUMINAMATH_CALUDE_king_arthur_table_seats_l3894_389418


namespace NUMINAMATH_CALUDE_interior_triangle_area_l3894_389457

theorem interior_triangle_area (a b c : ℝ) (ha : a^2 = 49) (hb : b^2 = 64) (hc : c^2 = 225) :
  (1/2 : ℝ) * a * b = 28 :=
sorry

end NUMINAMATH_CALUDE_interior_triangle_area_l3894_389457


namespace NUMINAMATH_CALUDE_triangle_side_expression_simplification_l3894_389451

theorem triangle_side_expression_simplification
  (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  |a - b - c| + |b - c + a| + |c - a - b| = a + 3*b - c :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_expression_simplification_l3894_389451


namespace NUMINAMATH_CALUDE_square_side_length_l3894_389434

theorem square_side_length (P A : ℝ) (h1 : P = 12) (h2 : A = 9) : ∃ s : ℝ, s > 0 ∧ P = 4 * s ∧ A = s ^ 2 ∧ s = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3894_389434


namespace NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l3894_389455

theorem tan_product_from_cos_sum_diff (α β : ℝ) 
  (h1 : Real.cos (α + β) = 2/3) 
  (h2 : Real.cos (α - β) = 1/3) : 
  Real.tan α * Real.tan β = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_from_cos_sum_diff_l3894_389455


namespace NUMINAMATH_CALUDE_smallest_base_for_perfect_fourth_power_l3894_389447

theorem smallest_base_for_perfect_fourth_power : 
  (∃ (b : ℕ), b > 0 ∧ ∃ (x : ℕ), 7 * b^2 + 7 * b + 7 = x^4) ∧ 
  (∀ (b : ℕ), b > 0 → (∃ (x : ℕ), 7 * b^2 + 7 * b + 7 = x^4) → b ≥ 18) :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_for_perfect_fourth_power_l3894_389447


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3894_389413

theorem smallest_integer_with_remainders (n : ℕ) : 
  (n > 1) → 
  (∀ m : ℕ, m > 1 ∧ m < n → ¬(m % 6 = 1 ∧ m % 7 = 1 ∧ m % 9 = 1)) → 
  (n % 6 = 1 ∧ n % 7 = 1 ∧ n % 9 = 1) → 
  (n = 127 ∧ 120 < n ∧ n < 199) :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l3894_389413


namespace NUMINAMATH_CALUDE_candy_distribution_l3894_389427

theorem candy_distribution (num_students : ℕ) (pieces_per_student : ℕ) 
  (h1 : num_students = 43)
  (h2 : pieces_per_student = 8) :
  num_students * pieces_per_student = 344 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l3894_389427


namespace NUMINAMATH_CALUDE_club_size_l3894_389437

/-- A club with committees satisfying specific conditions -/
structure Club where
  /-- The number of committees in the club -/
  num_committees : Nat
  /-- The number of members in the club -/
  num_members : Nat
  /-- Each member belongs to exactly two committees -/
  member_in_two_committees : True
  /-- Each pair of committees has exactly one member in common -/
  one_common_member : True

/-- Theorem stating that a club with 4 committees satisfying the given conditions has 6 members -/
theorem club_size (c : Club) : c.num_committees = 4 → c.num_members = 6 := by
  sorry

end NUMINAMATH_CALUDE_club_size_l3894_389437


namespace NUMINAMATH_CALUDE_parabola_m_value_l3894_389461

/-- Theorem: For a parabola with equation x² = my, where m is a positive real number,
    if the distance from the vertex to the directrix is 1/2, then m = 2. -/
theorem parabola_m_value (m : ℝ) (h1 : m > 0) : 
  (∀ x y : ℝ, x^2 = m*y) →  -- Parabola equation
  (1/2 : ℝ) = (1/4 : ℝ) * m →  -- Distance from vertex to directrix is 1/2
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_parabola_m_value_l3894_389461


namespace NUMINAMATH_CALUDE_no_twin_prime_legs_in_right_triangle_l3894_389441

theorem no_twin_prime_legs_in_right_triangle :
  ∀ (p k : ℕ), 
    Prime p → 
    Prime (p + 2) → 
    (∃ (h : ℕ), h * h = p * p + (p + 2) * (p + 2)) → 
    False :=
by
  sorry

end NUMINAMATH_CALUDE_no_twin_prime_legs_in_right_triangle_l3894_389441


namespace NUMINAMATH_CALUDE_geometric_series_ratio_l3894_389463

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a * r^4 / (1 - r)) = (1 / 64) * (a / (1 - r)) →
  r = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_ratio_l3894_389463


namespace NUMINAMATH_CALUDE_exists_number_with_special_quotient_l3894_389403

-- Define a function to check if a number contains all digits from 1 to 8 exactly once
def containsAllDigitsOnce (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ Finset.range 8 → (∃! i : ℕ, i < (Nat.digits 10 n).length ∧ (Nat.digits 10 n).get ⟨i, by sorry⟩ = d + 1)

-- Theorem statement
theorem exists_number_with_special_quotient :
  ∃ N d : ℕ, N > 0 ∧ d > 0 ∧ containsAllDigitsOnce (N / d) :=
sorry

end NUMINAMATH_CALUDE_exists_number_with_special_quotient_l3894_389403


namespace NUMINAMATH_CALUDE_equation_solution_l3894_389439

theorem equation_solution (x : ℝ) : (2*x - 1)^2 = 81 → x = 5 ∨ x = -4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3894_389439


namespace NUMINAMATH_CALUDE_abies_chips_l3894_389438

theorem abies_chips (initial_bags : ℕ) (bought_bags : ℕ) (final_bags : ℕ) 
  (h1 : initial_bags = 20)
  (h2 : bought_bags = 6)
  (h3 : final_bags = 22) :
  initial_bags - (initial_bags - final_bags + bought_bags) = 4 :=
by sorry

end NUMINAMATH_CALUDE_abies_chips_l3894_389438


namespace NUMINAMATH_CALUDE_infinite_series_sum_l3894_389402

/-- The sum of the infinite series ∑(n=1 to ∞) (n+1) / (n^2(n+2)) is equal to 3/8 + π^2/24 -/
theorem infinite_series_sum : 
  ∑' n : ℕ, (n + 1 : ℝ) / (n^2 * (n + 2)) = 3/8 + π^2/24 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l3894_389402


namespace NUMINAMATH_CALUDE_escalator_steps_count_l3894_389498

/-- Represents the number of steps a person climbs on the escalator -/
structure ClimbingSteps where
  steps : ℕ

/-- Represents the speed at which a person climbs the escalator -/
structure ClimbingSpeed where
  speed : ℕ

/-- Represents a person climbing the escalator -/
structure Person where
  climbingSteps : ClimbingSteps
  climbingSpeed : ClimbingSpeed

/-- Calculates the total number of steps in the escalator -/
def escalatorSteps (personA personB : Person) : ℕ :=
  sorry

theorem escalator_steps_count
  (personA personB : Person)
  (hA : personA.climbingSteps.steps = 55)
  (hB : personB.climbingSteps.steps = 60)
  (hSpeed : personB.climbingSpeed.speed = 2 * personA.climbingSpeed.speed) :
  escalatorSteps personA personB = 66 :=
sorry

end NUMINAMATH_CALUDE_escalator_steps_count_l3894_389498


namespace NUMINAMATH_CALUDE_starting_lineup_count_l3894_389419

def team_size : Nat := 16
def lineup_size : Nat := 5
def twin_count : Nat := 2
def triplet_count : Nat := 3

theorem starting_lineup_count : 
  (triplet_count * Nat.choose (team_size - twin_count - triplet_count + 2) (lineup_size - twin_count - 1)) = 198 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_count_l3894_389419


namespace NUMINAMATH_CALUDE_company_managers_count_l3894_389464

/-- Proves that the number of managers is 15 given the conditions in the problem --/
theorem company_managers_count :
  ∀ (num_managers : ℕ) (num_associates : ℕ) (avg_salary_managers : ℚ) (avg_salary_associates : ℚ) (avg_salary_company : ℚ),
  num_associates = 75 →
  avg_salary_managers = 90000 →
  avg_salary_associates = 30000 →
  avg_salary_company = 40000 →
  (num_managers * avg_salary_managers + num_associates * avg_salary_associates) / (num_managers + num_associates : ℚ) = avg_salary_company →
  num_managers = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_company_managers_count_l3894_389464


namespace NUMINAMATH_CALUDE_orangeade_price_day2_l3894_389445

/-- Represents the price and volume of orangeade on a given day -/
structure OrangeadeDay where
  juice : ℝ
  water : ℝ
  price : ℝ

/-- Proves that the price per glass on the second day is $0.40 given the conditions -/
theorem orangeade_price_day2 (day1 day2 : OrangeadeDay) :
  day1.juice = day2.juice ∧ 
  day1.water = day1.juice ∧ 
  day2.water = 2 * day1.water ∧ 
  day1.price = 0.60 ∧ 
  (day1.juice + day1.water) * day1.price = (day2.juice + day2.water) * day2.price
  → day2.price = 0.40 := by
  sorry

#check orangeade_price_day2

end NUMINAMATH_CALUDE_orangeade_price_day2_l3894_389445


namespace NUMINAMATH_CALUDE_steve_gum_pieces_l3894_389440

theorem steve_gum_pieces (initial_gum : ℕ) (total_gum : ℕ) (h1 : initial_gum = 38) (h2 : total_gum = 54) :
  total_gum - initial_gum = 16 := by
  sorry

end NUMINAMATH_CALUDE_steve_gum_pieces_l3894_389440


namespace NUMINAMATH_CALUDE_number_of_skirts_l3894_389468

theorem number_of_skirts (total_ways : ℕ) (num_pants : ℕ) : 
  total_ways = 7 → num_pants = 4 → total_ways - num_pants = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_skirts_l3894_389468


namespace NUMINAMATH_CALUDE_terms_before_ten_l3894_389424

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + d * (n - 1)

theorem terms_before_ten (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  a₁ = 105 ∧ d = -5 →
  arithmetic_sequence a₁ d 20 = 10 ∧
  ∀ k : ℕ, k < 20 → arithmetic_sequence a₁ d k > 10 :=
by sorry

end NUMINAMATH_CALUDE_terms_before_ten_l3894_389424


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_145_l3894_389409

/-- The nth positive odd multiple of 5 -/
def nthOddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

theorem fifteenth_odd_multiple_of_5 : nthOddMultipleOf5 15 = 145 := by
  sorry

/-- The 15th positive integer that is both odd and a multiple of 5 -/
def fifteenthOddMultipleOf5 : ℕ := nthOddMultipleOf5 15

theorem fifteenth_odd_multiple_of_5_is_145 : fifteenthOddMultipleOf5 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_fifteenth_odd_multiple_of_5_is_145_l3894_389409


namespace NUMINAMATH_CALUDE_x_is_perfect_square_l3894_389494

theorem x_is_perfect_square (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ n : ℕ+, x = n^2 := by
sorry

end NUMINAMATH_CALUDE_x_is_perfect_square_l3894_389494


namespace NUMINAMATH_CALUDE_combined_pencil_length_l3894_389426

-- Define the length of a pencil in cubes
def pencil_length : ℕ := 12

-- Define the number of pencils
def num_pencils : ℕ := 2

-- Theorem: The combined length of two pencils is 24 cubes
theorem combined_pencil_length :
  num_pencils * pencil_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_combined_pencil_length_l3894_389426


namespace NUMINAMATH_CALUDE_max_vector_difference_value_l3894_389446

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the theorem
theorem max_vector_difference_value (a b : V) (ha : ‖a‖ = 2) (hb : ‖b‖ = 1) :
  ∃ (c : V), ∀ (x : V), ‖a - b‖ ≤ ‖x‖ ∧ ‖x‖ = 3 :=
sorry

end NUMINAMATH_CALUDE_max_vector_difference_value_l3894_389446


namespace NUMINAMATH_CALUDE_balls_sold_l3894_389410

theorem balls_sold (selling_price : ℕ) (cost_price : ℕ) (loss : ℕ) : 
  selling_price = 720 → 
  cost_price = 60 → 
  loss = 5 * cost_price → 
  ∃ n : ℕ, n * cost_price - selling_price = loss ∧ n = 17 :=
by sorry

end NUMINAMATH_CALUDE_balls_sold_l3894_389410


namespace NUMINAMATH_CALUDE_birds_on_fence_l3894_389477

theorem birds_on_fence (initial_birds : ℕ) (initial_storks : ℕ) (additional_birds : ℕ) : 
  initial_birds = 2 →
  initial_storks = 6 →
  initial_storks = (initial_birds + additional_birds + 1) →
  additional_birds = 3 := by
sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3894_389477


namespace NUMINAMATH_CALUDE_original_sales_tax_percentage_l3894_389480

/-- Proves that the original sales tax percentage was 5% given the conditions of the problem -/
theorem original_sales_tax_percentage
  (item_price : ℝ)
  (reduced_tax_rate : ℝ)
  (tax_difference : ℝ)
  (h1 : item_price = 1000)
  (h2 : reduced_tax_rate = 0.04)
  (h3 : tax_difference = 10)
  (h4 : item_price * reduced_tax_rate + tax_difference = item_price * (original_tax_rate / 100)) :
  original_tax_rate = 5 := by
  sorry

end NUMINAMATH_CALUDE_original_sales_tax_percentage_l3894_389480


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3894_389492

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_diff : a 7 - 2 * a 4 = 6) 
  (h_third : a 3 = 2) : 
  ∃ d : ℝ, d = 4 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3894_389492


namespace NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_equals_13_l3894_389417

theorem number_divided_by_6_multiplied_by_12_equals_13 : ∃ x : ℚ, (x / 6) * 12 = 13 ∧ x = 13/2 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_equals_13_l3894_389417


namespace NUMINAMATH_CALUDE_lemonade_water_requirement_l3894_389433

/-- The amount of water required for lemonade recipe -/
def water_required (water_parts : ℚ) (lemon_juice_parts : ℚ) (total_gallons : ℚ) (quarts_per_gallon : ℚ) (cups_per_quart : ℚ) : ℚ :=
  (water_parts / (water_parts + lemon_juice_parts)) * total_gallons * quarts_per_gallon * cups_per_quart

/-- Theorem stating the required amount of water for the lemonade recipe -/
theorem lemonade_water_requirement : 
  water_required 5 2 (3/2) 4 4 = 120/7 := by
  sorry

end NUMINAMATH_CALUDE_lemonade_water_requirement_l3894_389433


namespace NUMINAMATH_CALUDE_zero_vector_magnitude_is_zero_l3894_389462

/-- The magnitude of the zero vector in a 2D plane is 0. -/
theorem zero_vector_magnitude_is_zero :
  ∀ (v : ℝ × ℝ), v = (0, 0) → ‖v‖ = 0 := by
  sorry

end NUMINAMATH_CALUDE_zero_vector_magnitude_is_zero_l3894_389462


namespace NUMINAMATH_CALUDE_simplify_square_roots_l3894_389459

theorem simplify_square_roots : Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l3894_389459


namespace NUMINAMATH_CALUDE_f_not_in_second_quadrant_l3894_389479

/-- The function f(x) = x - 2 -/
def f (x : ℝ) : ℝ := x - 2

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem f_not_in_second_quadrant :
  ∀ x : ℝ, ¬(in_second_quadrant x (f x)) :=
by
  sorry

end NUMINAMATH_CALUDE_f_not_in_second_quadrant_l3894_389479


namespace NUMINAMATH_CALUDE_players_both_games_l3894_389442

/-- Given a group of players with the following properties:
  * There are 400 players in total
  * 350 players play outdoor games
  * 110 players play indoor games
  This theorem proves that the number of players who play both indoor and outdoor games is 60. -/
theorem players_both_games (total : ℕ) (outdoor : ℕ) (indoor : ℕ) 
  (h_total : total = 400)
  (h_outdoor : outdoor = 350)
  (h_indoor : indoor = 110) :
  ∃ (both : ℕ), both = outdoor + indoor - total ∧ both = 60 := by
  sorry

end NUMINAMATH_CALUDE_players_both_games_l3894_389442


namespace NUMINAMATH_CALUDE_closest_to_70_l3894_389415

def A : ℚ := 254 / 5
def B : ℚ := 400 / 6
def C : ℚ := 492 / 7

def target : ℚ := 70

theorem closest_to_70 :
  |C - target| ≤ |A - target| ∧ |C - target| ≤ |B - target| :=
sorry

end NUMINAMATH_CALUDE_closest_to_70_l3894_389415


namespace NUMINAMATH_CALUDE_variety_promotion_criterion_variety_B_more_suitable_l3894_389497

/-- Represents a rice variety with its yield statistics -/
structure RiceVariety where
  mean_yield : ℝ
  variance : ℝ

/-- Determines if a rice variety is more suitable for promotion based on yield stability -/
def more_suitable_for_promotion (a b : RiceVariety) : Prop :=
  a.mean_yield = b.mean_yield ∧ a.variance < b.variance

/-- Theorem stating that given two varieties with equal mean yields, 
    the one with lower variance is more suitable for promotion -/
theorem variety_promotion_criterion 
  (a b : RiceVariety) 
  (h_equal_means : a.mean_yield = b.mean_yield) 
  (h_lower_variance : a.variance < b.variance) : 
  more_suitable_for_promotion b a := by
  sorry

/-- The specific rice varieties from the problem -/
def variety_A : RiceVariety := ⟨1042, 6.5⟩
def variety_B : RiceVariety := ⟨1042, 1.2⟩

/-- Theorem applying the general criterion to the specific varieties -/
theorem variety_B_more_suitable : 
  more_suitable_for_promotion variety_B variety_A := by
  sorry

end NUMINAMATH_CALUDE_variety_promotion_criterion_variety_B_more_suitable_l3894_389497


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3894_389481

-- Define the coefficients of the lines
def l1_coeff (a : ℝ) : ℝ × ℝ := (a + 2, 1 - a)
def l2_coeff (a : ℝ) : ℝ × ℝ := (a - 1, 2*a + 3)

-- Define the perpendicularity condition
def perpendicular (a : ℝ) : Prop :=
  (l1_coeff a).1 * (l2_coeff a).1 + (l1_coeff a).2 * (l2_coeff a).2 = 0

-- Theorem statement
theorem perpendicular_lines_a_value (a : ℝ) :
  perpendicular a → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3894_389481


namespace NUMINAMATH_CALUDE_female_officers_count_l3894_389453

theorem female_officers_count (total_on_duty : ℕ) (female_duty_percent : ℚ) 
  (h1 : total_on_duty = 160)
  (h2 : female_duty_percent = 16 / 100) : 
  ∃ (total_female : ℕ), total_female = 1000 ∧ 
    (female_duty_percent * total_female : ℚ) = total_on_duty := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l3894_389453


namespace NUMINAMATH_CALUDE_box_office_growth_l3894_389449

theorem box_office_growth (x : ℝ) : 
  (∃ (initial final : ℝ), 
    initial = 2 ∧ 
    final = 4 ∧ 
    final = initial * (1 + x)^2) ↔ 
  2 * (1 + x)^2 = 4 := by sorry

end NUMINAMATH_CALUDE_box_office_growth_l3894_389449


namespace NUMINAMATH_CALUDE_sample_size_representation_l3894_389472

/-- Represents a population of students -/
structure Population where
  size : ℕ

/-- Represents a sample of students -/
structure Sample where
  size : ℕ
  population : Population
  h : size ≤ population.size

/-- Theorem: In a statistical analysis context, when 30 students are selected from a population of 500,
    the number 30 represents the sample size -/
theorem sample_size_representation (pop : Population) (s : Sample) :
  pop.size = 500 →
  s.size = 30 →
  s.population = pop →
  s.size = Sample.size s :=
by sorry

end NUMINAMATH_CALUDE_sample_size_representation_l3894_389472


namespace NUMINAMATH_CALUDE_max_value_fraction_sum_l3894_389486

theorem max_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a / (a + 1) + b / (b + 1)) ≤ 2/3 ∧
  (a / (a + 1) + b / (b + 1) = 2/3 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_sum_l3894_389486


namespace NUMINAMATH_CALUDE_symmetric_point_and_line_l3894_389429

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := 2 * x - 3 * y + 1 = 0

-- Define point A
def A : ℝ × ℝ := (-1, -2)

-- Define line m
def m (x y : ℝ) : Prop := 3 * x - 2 * y - 6 = 0

-- Define the symmetric point of a given point with respect to l₁
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the symmetric line of a given line with respect to l₁
def symmetric_line (l : (ℝ → ℝ → Prop)) : (ℝ → ℝ → Prop) := sorry

theorem symmetric_point_and_line :
  (symmetric_point A = (-33/13, 4/13)) ∧
  (∀ x y, symmetric_line m x y ↔ 3 * x - 11 * y + 34 = 0) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_and_line_l3894_389429
