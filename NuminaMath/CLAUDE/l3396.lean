import Mathlib

namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_tangent_point_property_l3396_339672

/-- A quadrilateral inscribed in a circle with an inscribed circle inside it. -/
structure InscribedQuadrilateral where
  /-- The lengths of the four sides of the quadrilateral -/
  sides : Fin 4 → ℝ
  /-- The quadrilateral is inscribed in a circle -/
  inscribed_in_circle : Bool
  /-- There is a circle inscribed in the quadrilateral -/
  has_inscribed_circle : Bool

/-- The point of tangency divides a side into two segments -/
def tangent_point_division (q : InscribedQuadrilateral) : ℝ × ℝ := sorry

/-- The theorem stating the property of the inscribed quadrilateral -/
theorem inscribed_quadrilateral_tangent_point_property 
  (q : InscribedQuadrilateral)
  (h1 : q.sides 0 = 65)
  (h2 : q.sides 1 = 95)
  (h3 : q.sides 2 = 125)
  (h4 : q.sides 3 = 105)
  (h5 : q.inscribed_in_circle = true)
  (h6 : q.has_inscribed_circle = true) :
  let (x, y) := tangent_point_division q
  |x - y| = 14 := by sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_tangent_point_property_l3396_339672


namespace NUMINAMATH_CALUDE_rivertown_marching_band_max_members_l3396_339696

theorem rivertown_marching_band_max_members :
  ∀ n : ℕ, 
    (20 * n ≡ 11 [MOD 31]) → 
    (20 * n < 1200) → 
    (∀ m : ℕ, (20 * m ≡ 11 [MOD 31]) → (20 * m < 1200) → (20 * m ≤ 20 * n)) →
    20 * n = 1100 :=
by sorry

end NUMINAMATH_CALUDE_rivertown_marching_band_max_members_l3396_339696


namespace NUMINAMATH_CALUDE_engine_capacity_l3396_339628

/-- The engine capacity (in cc) for which 85 litres of diesel is required to travel 600 km -/
def C : ℝ := 595

/-- The volume of diesel (in litres) required for the reference engine -/
def V₁ : ℝ := 170

/-- The capacity (in cc) of the reference engine -/
def C₁ : ℝ := 1200

/-- The volume of diesel (in litres) required for the engine capacity C -/
def V₂ : ℝ := 85

/-- The ratio of volume to capacity is constant -/
axiom volume_capacity_ratio : V₁ / C₁ = V₂ / C

theorem engine_capacity : C = 595 := by sorry

end NUMINAMATH_CALUDE_engine_capacity_l3396_339628


namespace NUMINAMATH_CALUDE_root_square_condition_l3396_339606

theorem root_square_condition (a : ℚ) : 
  (∃ x y : ℚ, x^2 - (15/4)*x + a^3 = 0 ∧ y^2 - (15/4)*y + a^3 = 0 ∧ x = y^2) ↔ 
  (a = 3/2 ∨ a = -5/2) := by
sorry

end NUMINAMATH_CALUDE_root_square_condition_l3396_339606


namespace NUMINAMATH_CALUDE_slower_speed_calculation_l3396_339680

theorem slower_speed_calculation (distance : ℝ) (time_saved : ℝ) (faster_speed : ℝ) :
  distance = 1200 →
  time_saved = 4 →
  faster_speed = 60 →
  ∃ slower_speed : ℝ,
    (distance / slower_speed) - (distance / faster_speed) = time_saved ∧
    slower_speed = 50 := by
  sorry

end NUMINAMATH_CALUDE_slower_speed_calculation_l3396_339680


namespace NUMINAMATH_CALUDE_sixth_sample_number_l3396_339627

/-- Represents a random number table --/
def RandomNumberTable := List (List Nat)

/-- Checks if a number is valid (between 000 and 799) --/
def isValidNumber (n : Nat) : Bool :=
  n ≤ 799

/-- Finds the nth valid number in a list --/
def findNthValidNumber (numbers : List Nat) (n : Nat) : Option Nat :=
  let validNumbers := numbers.filter isValidNumber
  validNumbers.get? (n - 1)

/-- The main theorem --/
theorem sixth_sample_number
  (table : RandomNumberTable)
  (startRow : Nat)
  (startCol : Nat) :
  findNthValidNumber (table.join.drop (startRow * table.head!.length + startCol)) 6 = some 245 :=
sorry

end NUMINAMATH_CALUDE_sixth_sample_number_l3396_339627


namespace NUMINAMATH_CALUDE_salary_change_l3396_339633

theorem salary_change (initial_salary : ℝ) (h : initial_salary > 0) :
  let increased_salary := initial_salary * (1 + 0.15)
  let final_salary := increased_salary * (1 - 0.15)
  (final_salary - initial_salary) / initial_salary = -0.0225 := by
sorry

end NUMINAMATH_CALUDE_salary_change_l3396_339633


namespace NUMINAMATH_CALUDE_tons_approximation_l3396_339693

/-- Two real numbers are approximately equal if their absolute difference is less than 0.5 -/
def approximately_equal (x y : ℝ) : Prop := |x - y| < 0.5

/-- 1 ton is defined as 1000 kilograms -/
def ton : ℝ := 1000

theorem tons_approximation : approximately_equal (29.6 * ton) (30 * ton) := by sorry

end NUMINAMATH_CALUDE_tons_approximation_l3396_339693


namespace NUMINAMATH_CALUDE_simplify_expression_l3396_339687

theorem simplify_expression : 
  3 * Real.sqrt 48 - 9 * Real.sqrt (1/3) - Real.sqrt 3 * (2 - Real.sqrt 27) = 7 * Real.sqrt 3 + 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3396_339687


namespace NUMINAMATH_CALUDE_parallel_lines_k_value_l3396_339671

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The value of k for which the lines y = 5x - 3 and y = (3k)x + 7 are parallel -/
theorem parallel_lines_k_value :
  (∀ x y : ℝ, y = 5 * x - 3 ↔ y = (3 * k) * x + 7) → k = 5 / 3 :=
by sorry

end NUMINAMATH_CALUDE_parallel_lines_k_value_l3396_339671


namespace NUMINAMATH_CALUDE_pigeonhole_divisibility_l3396_339667

theorem pigeonhole_divisibility (x : Fin 2020 → ℤ) :
  ∃ i j : Fin 2020, i ≠ j ∧ (x j - x i) % 2019 = 0 := by
  sorry

end NUMINAMATH_CALUDE_pigeonhole_divisibility_l3396_339667


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3396_339674

theorem simplify_sqrt_expression :
  Real.sqrt (37 - 20 * Real.sqrt 3) = 5 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3396_339674


namespace NUMINAMATH_CALUDE_tangent_point_at_negative_one_slope_l3396_339634

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x + 2

theorem tangent_point_at_negative_one_slope :
  ∃ (x : ℝ), f' x = -1 ∧ (x, f x) = (1, 0) := by
  sorry

end NUMINAMATH_CALUDE_tangent_point_at_negative_one_slope_l3396_339634


namespace NUMINAMATH_CALUDE_diagonals_from_vertex_is_six_l3396_339631

/-- A polygon with internal angles of 140 degrees -/
structure Polygon140 where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The internal angle of the polygon is 140 degrees -/
  internal_angle : sides * 140 = (sides - 2) * 180

/-- The number of diagonals from a single vertex in a Polygon140 -/
def diagonals_from_vertex (p : Polygon140) : ℕ :=
  p.sides - 3

/-- Theorem: The number of diagonals from a vertex in a Polygon140 is 6 -/
theorem diagonals_from_vertex_is_six (p : Polygon140) :
  diagonals_from_vertex p = 6 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_from_vertex_is_six_l3396_339631


namespace NUMINAMATH_CALUDE_hemisphere_volume_l3396_339632

/-- The volume of a hemisphere with radius 21.002817118114375 cm is 96993.17249452507 cubic centimeters. -/
theorem hemisphere_volume : 
  let r : Real := 21.002817118114375
  let V : Real := (2/3) * Real.pi * r^3
  V = 96993.17249452507 := by sorry

end NUMINAMATH_CALUDE_hemisphere_volume_l3396_339632


namespace NUMINAMATH_CALUDE_divisibility_by_thirteen_l3396_339662

theorem divisibility_by_thirteen (n : ℕ) : (4 * 3^(2^n) + 3 * 4^(2^n)) % 13 = 0 ↔ n % 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_thirteen_l3396_339662


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l3396_339615

/-- Given information about oranges and apples, calculates the total cost of a specific purchase. -/
theorem fruit_purchase_cost 
  (orange_bags : ℕ) (orange_weight : ℝ) (apple_bags : ℕ) (apple_weight : ℝ)
  (orange_price : ℝ) (apple_price : ℝ)
  (h_orange : orange_bags * (orange_weight / orange_bags) = 24)
  (h_apple : apple_bags * (apple_weight / apple_bags) = 30)
  (h_orange_price : orange_price = 1.5)
  (h_apple_price : apple_price = 2) :
  5 * (orange_weight / orange_bags) * orange_price + 
  4 * (apple_weight / apple_bags) * apple_price = 45 := by
  sorry


end NUMINAMATH_CALUDE_fruit_purchase_cost_l3396_339615


namespace NUMINAMATH_CALUDE_muffin_milk_calculation_l3396_339636

/-- Given that 24 muffins require 3 liters of milk and 1 liter equals 4 cups,
    prove that 6 muffins require 3 cups of milk. -/
theorem muffin_milk_calculation (muffins_large : ℕ) (milk_liters : ℕ) (cups_per_liter : ℕ) 
  (muffins_small : ℕ) :
  muffins_large = 24 →
  milk_liters = 3 →
  cups_per_liter = 4 →
  muffins_small = 6 →
  (milk_liters * cups_per_liter * muffins_small) / muffins_large = 3 :=
by
  sorry

#check muffin_milk_calculation

end NUMINAMATH_CALUDE_muffin_milk_calculation_l3396_339636


namespace NUMINAMATH_CALUDE_book_arrangement_count_l3396_339655

theorem book_arrangement_count : 
  let total_books : ℕ := 7
  let identical_math_books : ℕ := 3
  let identical_physics_books : ℕ := 2
  let distinct_books : ℕ := total_books - identical_math_books - identical_physics_books
  ↑(Nat.factorial total_books) / (↑(Nat.factorial identical_math_books) * ↑(Nat.factorial identical_physics_books)) = 420 :=
by sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l3396_339655


namespace NUMINAMATH_CALUDE_steve_initial_berries_l3396_339681

theorem steve_initial_berries (stacy_initial : ℕ) (steve_takes : ℕ) (difference : ℕ) : 
  stacy_initial = 32 →
  steve_takes = 4 →
  stacy_initial - (steve_takes + difference) = stacy_initial - 7 →
  stacy_initial - difference - steve_takes = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_steve_initial_berries_l3396_339681


namespace NUMINAMATH_CALUDE_new_tax_rate_is_32_percent_l3396_339641

/-- Calculates the new tax rate given the original rate, income, and differential savings -/
def calculate_new_tax_rate (original_rate : ℚ) (income : ℚ) (differential_savings : ℚ) : ℚ :=
  (original_rate * income - differential_savings) / income

/-- Theorem stating that the new tax rate is 32% given the problem conditions -/
theorem new_tax_rate_is_32_percent :
  let original_rate : ℚ := 42 / 100
  let income : ℚ := 42400
  let differential_savings : ℚ := 4240
  calculate_new_tax_rate original_rate income differential_savings = 32 / 100 := by
  sorry

#eval calculate_new_tax_rate (42 / 100) 42400 4240

end NUMINAMATH_CALUDE_new_tax_rate_is_32_percent_l3396_339641


namespace NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_squares_divisible_by_two_l3396_339677

theorem sum_of_three_consecutive_odd_squares_divisible_by_two (n : ℤ) (h : Odd n) :
  ∃ k : ℤ, 3 * n^2 + 8 = 2 * k := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_consecutive_odd_squares_divisible_by_two_l3396_339677


namespace NUMINAMATH_CALUDE_special_rhombus_sum_l3396_339686

/-- A rhombus with specific vertex coordinates and area -/
structure SpecialRhombus where
  a : ℤ
  b : ℤ
  a_pos : 0 < a
  b_pos : 0 < b
  a_neq_b : a ≠ b
  area_eq : 2 * (a - b)^2 = 32

/-- The sum of a and b in a SpecialRhombus is 8 -/
theorem special_rhombus_sum (r : SpecialRhombus) : r.a + r.b = 8 := by
  sorry

end NUMINAMATH_CALUDE_special_rhombus_sum_l3396_339686


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_conditional_l3396_339651

-- Define p and q as propositions
variable (p q : Prop)

-- Define what it means for p to be a sufficient condition for q
def is_sufficient_condition (p q : Prop) : Prop :=
  p → q

-- Theorem statement
theorem sufficient_condition_implies_conditional 
  (h : is_sufficient_condition p q) : (p → q) = True :=
sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_conditional_l3396_339651


namespace NUMINAMATH_CALUDE_vector_problem_l3396_339603

/-- Given planar vectors a and b with angle π/3 between them, |a| = 2, and |b| = 1,
    prove that a · b = 1 and |a + 2b| = 2√3 -/
theorem vector_problem (a b : ℝ × ℝ) 
    (angle : Real.cos (π / 3) = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
    (mag_a : a.1^2 + a.2^2 = 4)
    (mag_b : b.1^2 + b.2^2 = 1) :
  (a.1 * b.1 + a.2 * b.2 = 1) ∧ 
  ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2 = 12) :=
by sorry

end NUMINAMATH_CALUDE_vector_problem_l3396_339603


namespace NUMINAMATH_CALUDE_intersection_point_l3396_339689

/-- The linear function f(x) = 5x + 1 -/
def f (x : ℝ) : ℝ := 5 * x + 1

/-- The y-axis is the set of points with x-coordinate 0 -/
def y_axis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- The graph of f is the set of points (x, f(x)) -/
def graph_f : Set (ℝ × ℝ) := {p | p.2 = f p.1}

theorem intersection_point : 
  (Set.inter graph_f y_axis) = {(0, 1)} := by sorry

end NUMINAMATH_CALUDE_intersection_point_l3396_339689


namespace NUMINAMATH_CALUDE_cyclists_distance_l3396_339682

/-- Calculates the distance between two cyclists traveling in opposite directions -/
def distance_between_cyclists (speed1 : ℝ) (speed2 : ℝ) (time : ℝ) : ℝ :=
  (speed1 + speed2) * time

theorem cyclists_distance :
  let speed1 : ℝ := 10
  let speed2 : ℝ := 25
  let time : ℝ := 1.4285714285714286
  distance_between_cyclists speed1 speed2 time = 50 := by
  sorry

end NUMINAMATH_CALUDE_cyclists_distance_l3396_339682


namespace NUMINAMATH_CALUDE_fraction_ordering_l3396_339698

theorem fraction_ordering : (12 : ℚ) / 35 < 10 / 29 ∧ 10 / 29 < 6 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l3396_339698


namespace NUMINAMATH_CALUDE_simplify_expressions_l3396_339673

theorem simplify_expressions :
  (∃ x, x = Real.sqrt 20 - Real.sqrt 5 + Real.sqrt (1/5) ∧ x = (6 * Real.sqrt 5) / 5) ∧
  (∃ y, y = (Real.sqrt 12 + Real.sqrt 18) / Real.sqrt 3 - 2 * Real.sqrt (1/2) * Real.sqrt 3 ∧ y = 2) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l3396_339673


namespace NUMINAMATH_CALUDE_juans_number_puzzle_l3396_339600

theorem juans_number_puzzle (n : ℝ) : ((2 * (n + 2) - 2) / 2 = 7) → n = 6 := by
  sorry

end NUMINAMATH_CALUDE_juans_number_puzzle_l3396_339600


namespace NUMINAMATH_CALUDE_population_change_l3396_339601

theorem population_change (k m : ℝ) : 
  let decrease_factor : ℝ := 1 - k / 100
  let increase_factor : ℝ := 1 + m / 100
  let total_factor : ℝ := decrease_factor * increase_factor
  total_factor = 1 + (m - k - k * m / 100) / 100 := by sorry

end NUMINAMATH_CALUDE_population_change_l3396_339601


namespace NUMINAMATH_CALUDE_movie_theater_ticket_sales_l3396_339685

/-- Represents the type of ticket --/
inductive TicketType
  | Adult
  | Child
  | SeniorOrStudent

/-- Represents the showtime --/
inductive Showtime
  | Matinee
  | Evening

/-- Returns the price of a ticket based on its type and showtime --/
def ticketPrice (t : TicketType) (s : Showtime) : ℕ :=
  match s, t with
  | Showtime.Matinee, TicketType.Adult => 5
  | Showtime.Matinee, TicketType.Child => 3
  | Showtime.Matinee, TicketType.SeniorOrStudent => 4
  | Showtime.Evening, TicketType.Adult => 9
  | Showtime.Evening, TicketType.Child => 5
  | Showtime.Evening, TicketType.SeniorOrStudent => 6

theorem movie_theater_ticket_sales
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (adult_tickets : ℕ)
  (child_tickets : ℕ)
  (senior_student_tickets : ℕ)
  (h1 : total_tickets = 1500)
  (h2 : total_revenue = 10500)
  (h3 : child_tickets = adult_tickets + 300)
  (h4 : 2 * (adult_tickets + child_tickets) = senior_student_tickets)
  (h5 : total_tickets = adult_tickets + child_tickets + senior_student_tickets) :
  adult_tickets = 100 := by
  sorry

#check movie_theater_ticket_sales

end NUMINAMATH_CALUDE_movie_theater_ticket_sales_l3396_339685


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3396_339676

-- Define the parallelogram ABCD
def A : ℝ × ℝ := (2, 3)
def B : ℝ × ℝ := (6, 0)
def C : ℝ × ℝ := (10, 5)

-- Define the property that A and C are diagonally opposite
def diagonally_opposite (A C : ℝ × ℝ) : Prop :=
  ∃ (M : ℝ × ℝ), M = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)

-- Define the property of a parallelogram
def is_parallelogram (A B C D : ℝ × ℝ) : Prop :=
  (B.1 - A.1 = D.1 - C.1) ∧ (B.2 - A.2 = D.2 - C.2)

-- Theorem statement
theorem parallelogram_vertex_sum :
  ∀ D : ℝ × ℝ,
  is_parallelogram A B C D →
  diagonally_opposite A C →
  D.1 + D.2 = 14 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l3396_339676


namespace NUMINAMATH_CALUDE_library_books_distribution_l3396_339666

theorem library_books_distribution (a b c d : ℕ) 
  (h1 : b + c + d = 110)
  (h2 : a + c + d = 108)
  (h3 : a + b + d = 104)
  (h4 : a + b + c = 119) :
  (a = 37 ∧ b = 39 ∧ c = 43 ∧ d = 28) := by
  sorry

end NUMINAMATH_CALUDE_library_books_distribution_l3396_339666


namespace NUMINAMATH_CALUDE_fast_food_fries_l3396_339617

theorem fast_food_fries (total : ℕ) (ratio : ℚ) (small : ℕ) : 
  total = 24 → ratio = 5 → small * (1 + ratio) = total → small = 4 := by
  sorry

end NUMINAMATH_CALUDE_fast_food_fries_l3396_339617


namespace NUMINAMATH_CALUDE_complex_equation_result_l3396_339663

theorem complex_equation_result (x y : ℝ) (h : x * Complex.I - y = -1 + Complex.I) :
  (1 - Complex.I) * (x - y * Complex.I) = -2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l3396_339663


namespace NUMINAMATH_CALUDE_calculation_result_l3396_339657

theorem calculation_result : 2009 * 20082008 - 2008 * 20092009 = 0 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l3396_339657


namespace NUMINAMATH_CALUDE_reflection_sum_coordinates_l3396_339640

/-- Given a point C with coordinates (3, y), when reflected over the x-axis to point D,
    the sum of all coordinate values of C and D is equal to 6. -/
theorem reflection_sum_coordinates (y : ℝ) : 
  let C : ℝ × ℝ := (3, y)
  let D : ℝ × ℝ := (3, -y)
  (C.1 + C.2 + D.1 + D.2) = 6 := by sorry

end NUMINAMATH_CALUDE_reflection_sum_coordinates_l3396_339640


namespace NUMINAMATH_CALUDE_choir_members_count_l3396_339618

theorem choir_members_count : ∃! n : ℕ, 300 ≤ n ∧ n ≤ 400 ∧ n % 12 = 10 ∧ n % 14 = 12 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_count_l3396_339618


namespace NUMINAMATH_CALUDE_evaluate_32_to_5_over_2_l3396_339621

theorem evaluate_32_to_5_over_2 : 32^(5/2) = 4096 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_32_to_5_over_2_l3396_339621


namespace NUMINAMATH_CALUDE_angela_january_sleep_l3396_339605

/-- The number of hours Angela slept per night in December -/
def december_sleep_hours : ℝ := 6.5

/-- The number of days in December -/
def december_days : ℕ := 31

/-- The additional hours Angela slept in January compared to December -/
def january_additional_sleep : ℝ := 62

/-- The number of days in January -/
def january_days : ℕ := 31

/-- Calculates the total hours Angela slept in December -/
def december_total_sleep : ℝ := december_sleep_hours * december_days

/-- Calculates the total hours Angela slept in January -/
def january_total_sleep : ℝ := december_total_sleep + january_additional_sleep

/-- Theorem stating that Angela slept 8.5 hours per night in January -/
theorem angela_january_sleep :
  january_total_sleep / january_days = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_angela_january_sleep_l3396_339605


namespace NUMINAMATH_CALUDE_circus_ticket_price_l3396_339688

theorem circus_ticket_price :
  ∀ (adult_price kid_price : ℝ),
    kid_price = (1/2) * adult_price →
    6 * kid_price + 2 * adult_price = 50 →
    kid_price = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_price_l3396_339688


namespace NUMINAMATH_CALUDE_water_remaining_in_cylinder_l3396_339652

/-- The volume of water remaining in a cylinder after pouring some into a cone -/
theorem water_remaining_in_cylinder (cylinder_volume cone_volume : ℝ) : 
  cylinder_volume = 18 →
  cylinder_volume = 3 * cone_volume →
  cylinder_volume - cone_volume = 12 :=
by sorry

end NUMINAMATH_CALUDE_water_remaining_in_cylinder_l3396_339652


namespace NUMINAMATH_CALUDE_quadratic_to_linear_inequality_l3396_339646

theorem quadratic_to_linear_inequality (a b : ℝ) :
  (∀ x, x^2 + a*x + b > 0 ↔ x < 3 ∨ x > 1) →
  (∀ x, a*x + b < 0 ↔ x > 3/4) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_to_linear_inequality_l3396_339646


namespace NUMINAMATH_CALUDE_bus_problem_l3396_339647

/-- Calculates the number of children who got on the bus -/
def children_got_on (initial : ℕ) (got_off : ℕ) (final : ℕ) : ℕ :=
  final - (initial - got_off)

/-- Proves that 5 children got on the bus given the initial, final, and number of children who got off -/
theorem bus_problem : children_got_on 21 10 16 = 5 := by
  sorry

end NUMINAMATH_CALUDE_bus_problem_l3396_339647


namespace NUMINAMATH_CALUDE_election_winner_percentage_l3396_339602

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 1944 →
  margin = 288 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) * 100 = 54 := by
sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l3396_339602


namespace NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l3396_339684

theorem quadratic_inequality_no_solution : 
  {x : ℝ | x^2 + 4*x + 4 < 0} = ∅ := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_no_solution_l3396_339684


namespace NUMINAMATH_CALUDE_area_of_specific_rectangle_l3396_339683

/-- Represents a rectangle with given properties -/
structure Rectangle where
  breadth : ℝ
  length : ℝ
  perimeter : ℝ
  area : ℝ

/-- Theorem: Area of a specific rectangle -/
theorem area_of_specific_rectangle :
  ∀ (rect : Rectangle),
  rect.length = 3 * rect.breadth →
  rect.perimeter = 104 →
  rect.area = rect.length * rect.breadth →
  rect.area = 507 := by
sorry

end NUMINAMATH_CALUDE_area_of_specific_rectangle_l3396_339683


namespace NUMINAMATH_CALUDE_paint_calculation_l3396_339637

theorem paint_calculation (initial_paint : ℚ) : 
  (initial_paint / 9 + (initial_paint - initial_paint / 9) / 5 = 104) →
  initial_paint = 360 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l3396_339637


namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l3396_339644

/-- A right triangle with an inscribed rectangle -/
structure RightTriangleWithRectangle where
  -- The lengths of the legs of the right triangle
  ab : ℝ
  ac : ℝ
  -- The sides of the inscribed rectangle
  ad : ℝ
  am : ℝ
  -- Conditions
  ab_positive : 0 < ab
  ac_positive : 0 < ac
  ad_positive : 0 < ad
  am_positive : 0 < am
  ad_le_ab : ad ≤ ab
  am_le_ac : am ≤ ac

/-- The theorem statement -/
theorem inscribed_rectangle_sides
  (triangle : RightTriangleWithRectangle)
  (h_ab : triangle.ab = 5)
  (h_ac : triangle.ac = 12)
  (h_area : triangle.ad * triangle.am = 40 / 3)
  (h_diagonal : triangle.ad ^ 2 + triangle.am ^ 2 < 8 ^ 2) :
  triangle.ad = 4 ∧ triangle.am = 10 / 3 :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l3396_339644


namespace NUMINAMATH_CALUDE_grover_boxes_l3396_339624

/-- Represents the number of face masks in each box -/
def masks_per_box : ℕ := 20

/-- Represents the cost of each box in dollars -/
def cost_per_box : ℚ := 15

/-- Represents the selling price of each face mask in dollars -/
def price_per_mask : ℚ := 5/4  -- $1.25

/-- Represents the total profit in dollars -/
def total_profit : ℚ := 15

/-- Calculates the revenue from selling one box of face masks -/
def revenue_per_box : ℚ := masks_per_box * price_per_mask

/-- Calculates the profit from selling one box of face masks -/
def profit_per_box : ℚ := revenue_per_box - cost_per_box

/-- Theorem: Given the conditions, Grover bought 3 boxes of face masks -/
theorem grover_boxes : 
  ∃ (n : ℕ), n * profit_per_box = total_profit ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_grover_boxes_l3396_339624


namespace NUMINAMATH_CALUDE_valid_documents_l3396_339658

theorem valid_documents (total_papers : ℕ) (invalid_percentage : ℚ) 
  (h1 : total_papers = 400)
  (h2 : invalid_percentage = 40 / 100) :
  (total_papers : ℚ) * (1 - invalid_percentage) = 240 := by
  sorry

end NUMINAMATH_CALUDE_valid_documents_l3396_339658


namespace NUMINAMATH_CALUDE_cube_preserves_order_l3396_339613

theorem cube_preserves_order (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_preserves_order_l3396_339613


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3396_339611

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (Finset.range n.val).sum (λ i => seq.a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_ratio (a b : ArithmeticSequence) :
  (∀ n : ℕ+, (sum_n a n) / (sum_n b n) = (38 * n + 14) / (2 * n + 1)) →
  (a.a 6) / (b.a 7) = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3396_339611


namespace NUMINAMATH_CALUDE_restaurant_at_park_office_l3396_339661

/-- Represents the time in minutes for various parts of Dante's journey -/
structure JourneyTimes where
  toHiddenLake : ℕ
  fromHiddenLake : ℕ
  toRestaurant : ℕ

/-- The actual journey times given in the problem -/
def actualJourney : JourneyTimes where
  toHiddenLake := 15
  fromHiddenLake := 7
  toRestaurant := 0

/-- Calculates the total time for a journey without visiting the restaurant -/
def totalTimeWithoutRestaurant (j : JourneyTimes) : ℕ :=
  j.toHiddenLake + j.fromHiddenLake

/-- Calculates the total time for a journey with visiting the restaurant -/
def totalTimeWithRestaurant (j : JourneyTimes) : ℕ :=
  j.toRestaurant + j.toRestaurant + j.toHiddenLake + j.fromHiddenLake

/-- Theorem stating that the time to the restaurant is 0 given the journey times are equal -/
theorem restaurant_at_park_office (j : JourneyTimes) 
  (h : totalTimeWithoutRestaurant j = totalTimeWithRestaurant j) : 
  j.toRestaurant = 0 := by
  sorry

#check restaurant_at_park_office

end NUMINAMATH_CALUDE_restaurant_at_park_office_l3396_339661


namespace NUMINAMATH_CALUDE_ninety_percent_of_nine_thousand_l3396_339609

theorem ninety_percent_of_nine_thousand (total_population : ℕ) (percentage : ℚ) : 
  total_population = 9000 → percentage = 90 / 100 → 
  (percentage * total_population : ℚ) = 8100 := by
  sorry

end NUMINAMATH_CALUDE_ninety_percent_of_nine_thousand_l3396_339609


namespace NUMINAMATH_CALUDE_cube_sum_equality_l3396_339642

theorem cube_sum_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^3 + b^3 + c^3 - 3*a*b*c = 0) : a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l3396_339642


namespace NUMINAMATH_CALUDE_hypotenuse_length_l3396_339623

/-- A rectangle with an inscribed circle -/
structure RectangleWithInscribedCircle where
  -- The length of side AB
  ab : ℝ
  -- The length of side BC
  bc : ℝ
  -- The point where the circle touches AB
  p : ℝ × ℝ
  -- The point where the circle touches BC
  q : ℝ × ℝ
  -- The point where the circle touches CD
  r : ℝ × ℝ
  -- The point where the circle touches DA
  s : ℝ × ℝ

/-- The theorem stating the length of the hypotenuse of triangle APD -/
theorem hypotenuse_length (rect : RectangleWithInscribedCircle)
  (h_ab : rect.ab = 20)
  (h_bc : rect.bc = 10) :
  Real.sqrt ((rect.ab - 2 * (rect.ab * rect.bc) / (2 * (rect.ab + rect.bc)))^2 + rect.bc^2) = 50 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_length_l3396_339623


namespace NUMINAMATH_CALUDE_two_group_subcommittee_count_l3396_339614

theorem two_group_subcommittee_count :
  let total_people : ℕ := 8
  let group_a_size : ℕ := 5
  let group_b_size : ℕ := 3
  let subcommittee_size : ℕ := 2
  group_a_size + group_b_size = total_people →
  group_a_size * group_b_size = 15
  := by sorry

end NUMINAMATH_CALUDE_two_group_subcommittee_count_l3396_339614


namespace NUMINAMATH_CALUDE_min_value_sum_product_l3396_339625

theorem min_value_sum_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * ((x + y)⁻¹ + (x + z)⁻¹ + (y + z)⁻¹) ≥ (9 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_product_l3396_339625


namespace NUMINAMATH_CALUDE_at_least_two_equal_l3396_339639

theorem at_least_two_equal (x y z : ℝ) (h : x/y + y/z + z/x = z/y + y/x + x/z) :
  (x = y) ∨ (y = z) ∨ (z = x) := by
  sorry

end NUMINAMATH_CALUDE_at_least_two_equal_l3396_339639


namespace NUMINAMATH_CALUDE_x_axis_condition_l3396_339669

/-- A line in the 2D plane represented by the equation Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- The x-axis is a line where y = 0 for all x -/
def is_x_axis (l : Line) : Prop :=
  ∀ x y : ℝ, l.A * x + l.B * y + l.C = 0 ↔ y = 0

/-- If a line is the x-axis, then B ≠ 0 and A = C = 0 -/
theorem x_axis_condition (l : Line) :
  is_x_axis l → l.B ≠ 0 ∧ l.A = 0 ∧ l.C = 0 := by
  sorry

end NUMINAMATH_CALUDE_x_axis_condition_l3396_339669


namespace NUMINAMATH_CALUDE_matrix_equation_properties_l3396_339660

open Matrix ComplexConjugate

variable {n : ℕ}

def conjugate_transpose (A : Matrix (Fin n) (Fin n) ℂ) : Matrix (Fin n) (Fin n) ℂ :=
  star A

theorem matrix_equation_properties
  (α : ℂ)
  (A : Matrix (Fin n) (Fin n) ℂ)
  (h_alpha : α ≠ 0)
  (h_A : A ≠ 0)
  (h_eq : A ^ 2 + (conjugate_transpose A) ^ 2 = α • (A * conjugate_transpose A)) :
  α.im = 0 ∧ Complex.abs α ≤ 2 ∧ A * conjugate_transpose A = conjugate_transpose A * A := by
  sorry

end NUMINAMATH_CALUDE_matrix_equation_properties_l3396_339660


namespace NUMINAMATH_CALUDE_factors_comparison_infinite_equal_factors_infinite_more_4k1_factors_l3396_339620

-- Define f(n) as the number of prime factors of n of the form 4k+1
def f (n : ℕ+) : ℕ := sorry

-- Define g(n) as the number of prime factors of n of the form 4k+3
def g (n : ℕ+) : ℕ := sorry

-- Statement 1
theorem factors_comparison (n : ℕ+) : f n ≥ g n := by sorry

-- Statement 2
theorem infinite_equal_factors : Set.Infinite {n : ℕ+ | f n = g n} := by sorry

-- Statement 3
theorem infinite_more_4k1_factors : Set.Infinite {n : ℕ+ | f n > g n} := by sorry

end NUMINAMATH_CALUDE_factors_comparison_infinite_equal_factors_infinite_more_4k1_factors_l3396_339620


namespace NUMINAMATH_CALUDE_F_equality_implies_a_half_l3396_339608

/-- Definition of function F -/
def F (a b c : ℝ) : ℝ := a * (b^2 + c^2) + b * c

/-- Theorem: If F(a, 3, 4) = F(a, 2, 5), then a = 1/2 -/
theorem F_equality_implies_a_half :
  ∀ a : ℝ, F a 3 4 = F a 2 5 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_F_equality_implies_a_half_l3396_339608


namespace NUMINAMATH_CALUDE_square_difference_division_l3396_339694

theorem square_difference_division : (196^2 - 169^2) / 27 = 365 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_division_l3396_339694


namespace NUMINAMATH_CALUDE_papers_printed_proof_l3396_339607

theorem papers_printed_proof :
  let presses1 : ℕ := 40
  let presses2 : ℕ := 30
  let time1 : ℝ := 12
  let time2 : ℝ := 15.999999999999998
  let rate : ℝ := (presses2 * time2) / (presses1 * time1)
  presses1 * rate * time1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_papers_printed_proof_l3396_339607


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3396_339692

theorem arithmetic_geometric_mean_inequality (a b : ℝ) : 
  (a^2 + b^2) / 2 ≥ ((a + b) / 2)^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l3396_339692


namespace NUMINAMATH_CALUDE_power_of_two_greater_than_square_l3396_339649

theorem power_of_two_greater_than_square (n : ℕ) (h : n ≥ 1) :
  2^n > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_greater_than_square_l3396_339649


namespace NUMINAMATH_CALUDE_derivative_problems_l3396_339630

open Real

theorem derivative_problems :
  (∀ x : ℝ, deriv (λ x => (2*x^2 + 3)*(3*x - 1)) x = 18*x^2 - 4*x + 9) ∧
  (∀ x : ℝ, deriv (λ x => x * exp x + 2*x + 1) x = exp x + x * exp x + 2) := by
  sorry

end NUMINAMATH_CALUDE_derivative_problems_l3396_339630


namespace NUMINAMATH_CALUDE_max_balloon_surface_area_l3396_339645

/-- The maximum surface area of a spherical balloon inscribed in a cube --/
theorem max_balloon_surface_area (a : ℝ) (h : a > 0) :
  ∃ (A : ℝ), A = 2 * Real.pi * a^2 ∧ 
  ∀ (r : ℝ), r > 0 → r ≤ a * Real.sqrt 2 / 2 → 
  4 * Real.pi * r^2 ≤ A := by
  sorry

end NUMINAMATH_CALUDE_max_balloon_surface_area_l3396_339645


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l3396_339650

/-- Calculates the total cost of car repair given the following parameters:
  * rate1: hourly rate of the first mechanic
  * hours1: hours worked per day by the first mechanic
  * days1: number of days worked by the first mechanic
  * rate2: hourly rate of the second mechanic
  * hours2: hours worked per day by the second mechanic
  * days2: number of days worked by the second mechanic
  * parts_cost: cost of parts used in the repair
-/
def total_repair_cost (rate1 hours1 days1 rate2 hours2 days2 parts_cost : ℕ) : ℕ :=
  rate1 * hours1 * days1 + rate2 * hours2 * days2 + parts_cost

/-- Theorem stating that the total repair cost for the given scenario is $14,420 -/
theorem repair_cost_calculation :
  total_repair_cost 60 8 14 75 6 10 3200 = 14420 := by
  sorry

#eval total_repair_cost 60 8 14 75 6 10 3200

end NUMINAMATH_CALUDE_repair_cost_calculation_l3396_339650


namespace NUMINAMATH_CALUDE_zhang_daily_distance_l3396_339665

/-- Given a one-way distance and number of round trips, calculates the total distance driven. -/
def total_distance (one_way_distance : ℕ) (num_round_trips : ℕ) : ℕ :=
  2 * one_way_distance * num_round_trips

/-- Proves that given a one-way distance of 33 kilometers and 5 round trips per day, 
    the total distance driven is 330 kilometers. -/
theorem zhang_daily_distance : total_distance 33 5 = 330 := by
  sorry

end NUMINAMATH_CALUDE_zhang_daily_distance_l3396_339665


namespace NUMINAMATH_CALUDE_triangle_sum_equality_l3396_339695

theorem triangle_sum_equality 
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + x*y + y^2 = a^2)
  (h2 : y^2 + y*z + z^2 = b^2)
  (h3 : x^2 + x*z + z^2 = c^2) :
  let p := (a + b + c) / 2
  x*y + y*z + x*z = 4 * Real.sqrt ((p*(p-a)*(p-b)*(p-c))/3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_equality_l3396_339695


namespace NUMINAMATH_CALUDE_seventh_root_unity_product_l3396_339643

theorem seventh_root_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_unity_product_l3396_339643


namespace NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3396_339629

theorem log_equality_implies_golden_ratio (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (Real.log p / Real.log 8 = Real.log q / Real.log 18) ∧
  (Real.log q / Real.log 18 = Real.log (p + q) / Real.log 32) →
  q / p = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_log_equality_implies_golden_ratio_l3396_339629


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3396_339678

theorem quadratic_inequality_always_true (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 2 * a * x - 4 < 0) ↔ -4 < a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3396_339678


namespace NUMINAMATH_CALUDE_min_surface_area_3x3x3_minus_5_l3396_339679

/-- Represents a 3D cube composed of unit cubes -/
structure Cube3D where
  size : Nat
  total_units : Nat

/-- Represents the remaining solid after removing some unit cubes -/
structure RemainingCube where
  original : Cube3D
  removed : Nat

/-- Calculates the minimum surface area of the remaining solid -/
def min_surface_area (rc : RemainingCube) : Nat :=
  sorry

/-- Theorem stating the minimum surface area after removing 5 unit cubes from a 3x3x3 cube -/
theorem min_surface_area_3x3x3_minus_5 :
  let original_cube : Cube3D := { size := 3, total_units := 27 }
  let remaining_cube : RemainingCube := { original := original_cube, removed := 5 }
  min_surface_area remaining_cube = 50 := by
  sorry

end NUMINAMATH_CALUDE_min_surface_area_3x3x3_minus_5_l3396_339679


namespace NUMINAMATH_CALUDE_circle_area_ratio_l3396_339638

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) : 
  (30 / 360 : ℝ) * (2 * Real.pi * r₁) = (45 / 360 : ℝ) * (2 * Real.pi * r₂) →
  (Real.pi * r₁^2) / (Real.pi * r₂^2) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l3396_339638


namespace NUMINAMATH_CALUDE_product_simplification_l3396_339670

theorem product_simplification :
  (240 : ℚ) / 18 * (9 : ℚ) / 160 * (10 : ℚ) / 3 = 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l3396_339670


namespace NUMINAMATH_CALUDE_fraction_addition_l3396_339622

theorem fraction_addition (d : ℝ) : (5 + 4 * d) / 8 + 3 = (29 + 4 * d) / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l3396_339622


namespace NUMINAMATH_CALUDE_distinct_z_values_l3396_339656

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def z (x : ℕ) : ℕ := Int.natAbs (x - reverse_digits x)

theorem distinct_z_values :
  ∃ (S : Finset ℕ), (∀ x, is_two_digit x → z x ∈ S) ∧ S.card = 10 :=
sorry

end NUMINAMATH_CALUDE_distinct_z_values_l3396_339656


namespace NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3396_339653

/-- The minimum number of occupied seats required to ensure the next person must sit next to someone, given a total number of seats. -/
def min_occupied_seats (total_seats : ℕ) : ℕ :=
  2 * (total_seats / 4)

theorem min_occupied_seats_for_150 :
  min_occupied_seats 150 = 74 := by
  sorry

#eval min_occupied_seats 150

end NUMINAMATH_CALUDE_min_occupied_seats_for_150_l3396_339653


namespace NUMINAMATH_CALUDE_number_of_polynomials_l3396_339697

/-- A function to determine if an expression is a polynomial -/
def isPolynomial (expr : String) : Bool :=
  match expr with
  | "x^2+2" => true
  | "1/a+4" => false
  | "3ab^2/7" => true
  | "ab/c" => false
  | "-5x" => true
  | _ => false

/-- The list of expressions to check -/
def expressions : List String :=
  ["x^2+2", "1/a+4", "3ab^2/7", "ab/c", "-5x"]

/-- Theorem stating that the number of polynomials in the given list is 3 -/
theorem number_of_polynomials :
  (expressions.filter isPolynomial).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_polynomials_l3396_339697


namespace NUMINAMATH_CALUDE_polynomial_equality_l3396_339654

theorem polynomial_equality (h : ℝ → ℝ) : 
  (∀ x : ℝ, 7 * x^4 - 4 * x^3 + x + h x = 5 * x^3 - 7 * x + 6) →
  (∀ x : ℝ, h x = -7 * x^4 + 9 * x^3 - 8 * x + 6) :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3396_339654


namespace NUMINAMATH_CALUDE_bus_truck_meeting_time_l3396_339612

theorem bus_truck_meeting_time 
  (initial_distance : ℝ) 
  (truck_speed : ℝ) 
  (bus_speed : ℝ) 
  (final_distance : ℝ) 
  (h1 : initial_distance = 8)
  (h2 : truck_speed = 60)
  (h3 : bus_speed = 40)
  (h4 : final_distance = 78) :
  (final_distance - initial_distance) / (truck_speed + bus_speed) = 0.7 := by
sorry

end NUMINAMATH_CALUDE_bus_truck_meeting_time_l3396_339612


namespace NUMINAMATH_CALUDE_china_internet_users_scientific_notation_l3396_339699

/-- Represents the number of internet users in China in billions -/
def china_internet_users : ℝ := 1.067

/-- The scientific notation representation of the number of internet users -/
def scientific_notation : ℝ := 1.067 * (10 ^ 9)

theorem china_internet_users_scientific_notation :
  china_internet_users * (10 ^ 9) = scientific_notation := by sorry

end NUMINAMATH_CALUDE_china_internet_users_scientific_notation_l3396_339699


namespace NUMINAMATH_CALUDE_game_x_vs_game_y_l3396_339626

def coin_prob_heads : ℚ := 3/4
def coin_prob_tails : ℚ := 1/4

def game_x_win_prob : ℚ :=
  4 * (coin_prob_heads^4 * coin_prob_tails + coin_prob_tails^4 * coin_prob_heads)

def game_y_win_prob : ℚ :=
  coin_prob_heads^6 + coin_prob_tails^6

theorem game_x_vs_game_y :
  game_x_win_prob - game_y_win_prob = 298/2048 := by sorry

end NUMINAMATH_CALUDE_game_x_vs_game_y_l3396_339626


namespace NUMINAMATH_CALUDE_schools_count_proof_l3396_339610

def number_of_schools : ℕ := 24

theorem schools_count_proof :
  ∀ (total_students : ℕ) (andrew_rank : ℕ),
    total_students = 4 * number_of_schools →
    andrew_rank = (total_students + 1) / 2 →
    andrew_rank < 50 →
    andrew_rank > 48 →
    number_of_schools = 24 := by
  sorry

end NUMINAMATH_CALUDE_schools_count_proof_l3396_339610


namespace NUMINAMATH_CALUDE_dog_play_area_l3396_339616

/-- The area outside of a square doghouse that a dog can reach with a leash -/
theorem dog_play_area (leash_length : ℝ) (doghouse_side : ℝ) : 
  leash_length = 4 →
  doghouse_side = 2 →
  (3 / 4 * π * leash_length^2 + 2 * (1 / 4 * π * doghouse_side^2)) = 14 * π :=
by sorry

end NUMINAMATH_CALUDE_dog_play_area_l3396_339616


namespace NUMINAMATH_CALUDE_binary_101101_equals_45_l3396_339659

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_101101_equals_45 :
  binary_to_decimal [true, false, true, true, false, true] = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_45_l3396_339659


namespace NUMINAMATH_CALUDE_min_area_rectangle_l3396_339619

/-- A rectangle with integer dimensions and perimeter 200 has a minimum area of 99 square units. -/
theorem min_area_rectangle (l w : ℕ) : 
  (2 * l + 2 * w = 200) →  -- perimeter condition
  (l * w ≥ 99) :=          -- minimum area
by sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l3396_339619


namespace NUMINAMATH_CALUDE_smallest_valid_seating_l3396_339675

/-- Represents a circular table with chairs and seated people. -/
structure CircularTable where
  total_chairs : Nat
  seated_people : Nat

/-- Checks if a seating arrangement is valid (no isolated seats). -/
def is_valid_seating (table : CircularTable) : Prop :=
  ∀ n : Nat, n < table.total_chairs → 
    ∃ m : Nat, m < table.seated_people ∧ 
      (n = m ∨ n = (m + 1) % table.total_chairs ∨ n = (m - 1 + table.total_chairs) % table.total_chairs)

/-- The main theorem to be proved. -/
theorem smallest_valid_seating :
  ∀ table : CircularTable, 
    table.total_chairs = 60 →
    (is_valid_seating table ↔ table.seated_people ≥ 15) :=
by sorry

end NUMINAMATH_CALUDE_smallest_valid_seating_l3396_339675


namespace NUMINAMATH_CALUDE_pencils_per_box_l3396_339664

theorem pencils_per_box (total_pencils : ℕ) (num_boxes : ℚ) 
  (h1 : total_pencils = 2592) 
  (h2 : num_boxes = 4) : 
  (total_pencils : ℚ) / num_boxes = 648 := by
sorry

end NUMINAMATH_CALUDE_pencils_per_box_l3396_339664


namespace NUMINAMATH_CALUDE_trees_in_yard_l3396_339635

/-- Calculates the number of trees in a yard given the yard length and tree spacing -/
def num_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

/-- Theorem stating that the number of trees in a 375-meter yard with 15-meter spacing is 26 -/
theorem trees_in_yard : num_trees 375 15 = 26 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_yard_l3396_339635


namespace NUMINAMATH_CALUDE_class_receives_reward_l3396_339648

def standard_jumps : ℕ := 160

def performance_records : List ℤ := [16, -1, 20, -2, -5, 11, -7, 6, 9, 13]

def score (x : ℤ) : ℚ :=
  if x ≥ 0 then x
  else -0.5 * x.natAbs

def total_score (records : List ℤ) : ℚ :=
  (records.map score).sum

theorem class_receives_reward (records : List ℤ) :
  records = performance_records →
  total_score records > 65 := by
  sorry

end NUMINAMATH_CALUDE_class_receives_reward_l3396_339648


namespace NUMINAMATH_CALUDE_scientific_notation_4040000_l3396_339690

theorem scientific_notation_4040000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 4040000 = a * (10 : ℝ) ^ n ∧ a = 4.04 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_4040000_l3396_339690


namespace NUMINAMATH_CALUDE_cost_shop1_calculation_l3396_339604

-- Define the problem parameters
def books_shop1 : ℕ := 65
def books_shop2 : ℕ := 35
def cost_shop2 : ℕ := 2000
def avg_price : ℕ := 85

-- Theorem to prove
theorem cost_shop1_calculation :
  let total_books : ℕ := books_shop1 + books_shop2
  let total_cost : ℕ := total_books * avg_price
  let cost_shop1 : ℕ := total_cost - cost_shop2
  cost_shop1 = 6500 := by sorry

end NUMINAMATH_CALUDE_cost_shop1_calculation_l3396_339604


namespace NUMINAMATH_CALUDE_competition_outcomes_l3396_339668

/-- The number of participants in the competition -/
def n : ℕ := 6

/-- The number of places to be filled (1st, 2nd, 3rd) -/
def k : ℕ := 3

/-- The number of different ways to arrange k distinct items from a set of n distinct items -/
def arrangement_count (n k : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem competition_outcomes :
  arrangement_count n k = 120 :=
sorry

end NUMINAMATH_CALUDE_competition_outcomes_l3396_339668


namespace NUMINAMATH_CALUDE_line_through_two_points_l3396_339691

/-- Given a line with equation x = 4y + 5 passing through points (m, n) and (m + 2, n + p),
    prove that p = 1/2 -/
theorem line_through_two_points (m n p : ℝ) : 
  (m = 4 * n + 5) ∧ (m + 2 = 4 * (n + p) + 5) → p = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_line_through_two_points_l3396_339691
