import Mathlib

namespace NUMINAMATH_CALUDE_xy_plus_reciprocal_minimum_l3433_343333

theorem xy_plus_reciprocal_minimum (x y : ℝ) (hx : x < 0) (hy : y < 0) (hsum : x + y = -1) :
  ∀ z, z = x * y + 1 / (x * y) → z ≥ 17/4 :=
by sorry

end NUMINAMATH_CALUDE_xy_plus_reciprocal_minimum_l3433_343333


namespace NUMINAMATH_CALUDE_half_dollar_difference_l3433_343330

/-- Represents the number of coins of each type -/
structure CoinCount where
  nickels : ℕ
  quarters : ℕ
  half_dollars : ℕ

/-- The problem constraints -/
def valid_coin_count (c : CoinCount) : Prop :=
  c.nickels + c.quarters + c.half_dollars = 60 ∧
  5 * c.nickels + 25 * c.quarters + 50 * c.half_dollars = 1000

/-- The set of all valid coin counts -/
def valid_coin_counts : Set CoinCount :=
  {c | valid_coin_count c}

/-- The maximum number of half-dollars in any valid coin count -/
noncomputable def max_half_dollars : ℕ :=
  ⨆ (c : CoinCount) (h : c ∈ valid_coin_counts), c.half_dollars

/-- The minimum number of half-dollars in any valid coin count -/
noncomputable def min_half_dollars : ℕ :=
  ⨅ (c : CoinCount) (h : c ∈ valid_coin_counts), c.half_dollars

/-- The main theorem -/
theorem half_dollar_difference :
  max_half_dollars - min_half_dollars = 15 := by
  sorry

end NUMINAMATH_CALUDE_half_dollar_difference_l3433_343330


namespace NUMINAMATH_CALUDE_unique_positive_zero_implies_a_less_than_negative_two_l3433_343334

/-- Given a cubic function f(x) = ax^3 - 3x^2 + 1 with a unique positive zero,
    prove that the coefficient a must be less than -2. -/
theorem unique_positive_zero_implies_a_less_than_negative_two 
  (a : ℝ) (x₀ : ℝ) (h_unique : ∀ x : ℝ, a * x^3 - 3 * x^2 + 1 = 0 ↔ x = x₀) 
  (h_positive : x₀ > 0) : 
  a < -2 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_zero_implies_a_less_than_negative_two_l3433_343334


namespace NUMINAMATH_CALUDE_only_traffic_light_random_l3433_343372

-- Define the type for events
inductive Event
  | SunRise
  | TrafficLight
  | PeanutOil
  | NegativeSum

-- Define a predicate for random events
def isRandom (e : Event) : Prop :=
  match e with
  | Event.TrafficLight => True
  | _ => False

-- Theorem statement
theorem only_traffic_light_random :
  ∀ (e : Event), isRandom e ↔ e = Event.TrafficLight :=
by sorry

end NUMINAMATH_CALUDE_only_traffic_light_random_l3433_343372


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3433_343393

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | Real.sqrt x ≤ 3}
def B : Set ℝ := {x : ℝ | x^2 ≤ 9}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | 0 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3433_343393


namespace NUMINAMATH_CALUDE_smaller_number_problem_l3433_343390

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) :
  min x y = 15 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l3433_343390


namespace NUMINAMATH_CALUDE_student_guinea_pig_difference_l3433_343366

/-- The number of fifth-grade classrooms -/
def num_classrooms : ℕ := 5

/-- The number of students in each classroom -/
def students_per_classroom : ℕ := 20

/-- The number of guinea pigs in each classroom -/
def guinea_pigs_per_classroom : ℕ := 3

/-- The total number of students in all classrooms -/
def total_students : ℕ := num_classrooms * students_per_classroom

/-- The total number of guinea pigs in all classrooms -/
def total_guinea_pigs : ℕ := num_classrooms * guinea_pigs_per_classroom

theorem student_guinea_pig_difference :
  total_students - total_guinea_pigs = 85 := by
  sorry

end NUMINAMATH_CALUDE_student_guinea_pig_difference_l3433_343366


namespace NUMINAMATH_CALUDE_function_uniqueness_l3433_343367

theorem function_uniqueness (f : ℝ → ℝ) (a : ℝ) : 
  ∃! y, f a = y :=
sorry

end NUMINAMATH_CALUDE_function_uniqueness_l3433_343367


namespace NUMINAMATH_CALUDE_work_completion_time_l3433_343338

/-- The number of laborers originally employed by the contractor -/
def original_laborers : ℚ := 17.5

/-- The number of absent laborers -/
def absent_laborers : ℕ := 7

/-- The number of days it took the remaining laborers to complete the work -/
def actual_days : ℕ := 10

/-- The original number of days the work was supposed to be completed in -/
def original_days : ℚ := (original_laborers - absent_laborers : ℚ) * actual_days / original_laborers

theorem work_completion_time : original_days = 6 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3433_343338


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l3433_343383

/-- Given a line with equation y + 5 = -3(x + 6), 
    the sum of its x-intercept and y-intercept is -92/3 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 5 = -3 * (x + 6)) → 
  (∃ x_int y_int : ℝ, 
    (y_int + 5 = -3 * (x_int + 6)) ∧ 
    (0 + 5 = -3 * (x_int + 6)) ∧ 
    (y_int + 5 = -3 * (0 + 6)) ∧ 
    (x_int + y_int = -92/3)) := by
  sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l3433_343383


namespace NUMINAMATH_CALUDE_derivative_lg_l3433_343335

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem derivative_lg (x : ℝ) (h : x > 0) :
  deriv lg x = 1 / (x * Real.log 10) :=
sorry

end NUMINAMATH_CALUDE_derivative_lg_l3433_343335


namespace NUMINAMATH_CALUDE_investment_rate_problem_l3433_343303

/-- Given a sum of money invested for a certain period, this function calculates the simple interest. -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem investment_rate_problem (sum : ℝ) (time : ℝ) (base_rate : ℝ) (interest_difference : ℝ) 
  (higher_rate : ℝ) :
  sum = 14000 →
  time = 2 →
  base_rate = 0.12 →
  interest_difference = 840 →
  simpleInterest sum higher_rate time = simpleInterest sum base_rate time + interest_difference →
  higher_rate = 0.15 := by
sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l3433_343303


namespace NUMINAMATH_CALUDE_symmetry_implies_sum_l3433_343346

/-- Two points are symmetric about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are equal. -/
def symmetric_about_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_implies_sum (a b : ℝ) :
  symmetric_about_y_axis (a - 2, 3) (1, b + 1) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_sum_l3433_343346


namespace NUMINAMATH_CALUDE_sqrt_comparison_l3433_343354

theorem sqrt_comparison : Real.sqrt 11 - 3 < Real.sqrt 7 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_comparison_l3433_343354


namespace NUMINAMATH_CALUDE_cos_alpha_value_l3433_343326

theorem cos_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) 
  (h2 : Real.sin (α - π / 6) = 1 / 3) : 
  Real.cos α = (2 * Real.sqrt 6 - 1) / 6 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l3433_343326


namespace NUMINAMATH_CALUDE_axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l3433_343344

/-- The axis of symmetry of a parabola y = ax^2 + bx + c is the line x = -b/(2a) -/
theorem axis_of_symmetry_parabola (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  ∀ x, f (x + (-b / (2 * a))) = f (-b / (2 * a) - x) :=
sorry

/-- The axis of symmetry of the parabola y = -x^2 + 2022 is the line x = 0 -/
theorem axis_of_symmetry_specific_parabola :
  let f : ℝ → ℝ := λ x ↦ -x^2 + 2022
  ∀ x, f (x + 0) = f (0 - x) :=
sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_parabola_axis_of_symmetry_specific_parabola_l3433_343344


namespace NUMINAMATH_CALUDE_students_history_or_geography_not_both_l3433_343318

/-- The number of students taking both history and geography -/
def both : ℕ := 15

/-- The total number of students taking history -/
def history : ℕ := 30

/-- The number of students taking only geography -/
def only_geography : ℕ := 18

/-- Theorem: The number of students taking history or geography but not both is 33 -/
theorem students_history_or_geography_not_both : 
  (history - both) + only_geography = 33 := by sorry

end NUMINAMATH_CALUDE_students_history_or_geography_not_both_l3433_343318


namespace NUMINAMATH_CALUDE_tim_weekly_earnings_l3433_343384

/-- Tim's daily tasks -/
def daily_tasks : ℕ := 100

/-- Pay per task in dollars -/
def pay_per_task : ℚ := 6/5

/-- Number of working days per week -/
def working_days_per_week : ℕ := 6

/-- Tim's weekly earnings in dollars -/
def weekly_earnings : ℚ := daily_tasks * pay_per_task * working_days_per_week

theorem tim_weekly_earnings : weekly_earnings = 720 := by sorry

end NUMINAMATH_CALUDE_tim_weekly_earnings_l3433_343384


namespace NUMINAMATH_CALUDE_flip_invariant_numbers_l3433_343365

/-- A digit that remains unchanged when flipped upside down -/
inductive FlipInvariantDigit : Nat → Prop
  | zero : FlipInvariantDigit 0
  | eight : FlipInvariantDigit 8

/-- A three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≥ 0 ∧ tens ≤ 9 ∧ ones ≥ 0 ∧ ones ≤ 9

/-- A three-digit number that remains unchanged when flipped upside down -/
def FlipInvariantNumber (n : ThreeDigitNumber) : Prop :=
  FlipInvariantDigit n.hundreds ∧ FlipInvariantDigit n.tens ∧ FlipInvariantDigit n.ones

theorem flip_invariant_numbers :
  ∀ n : ThreeDigitNumber, FlipInvariantNumber n →
    (n.hundreds = 8 ∧ n.tens = 0 ∧ n.ones = 8) ∨ (n.hundreds = 8 ∧ n.tens = 8 ∧ n.ones = 8) :=
by sorry

end NUMINAMATH_CALUDE_flip_invariant_numbers_l3433_343365


namespace NUMINAMATH_CALUDE_min_value_sum_squared_fractions_l3433_343304

theorem min_value_sum_squared_fractions (x y z : ℕ+) (h : x + y + z = 9) :
  (x^2 + y^2) / (x + y : ℝ) + (x^2 + z^2) / (x + z : ℝ) + (y^2 + z^2) / (y + z : ℝ) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_squared_fractions_l3433_343304


namespace NUMINAMATH_CALUDE_circular_arc_length_l3433_343349

/-- The length of a circular arc with radius 10 meters and central angle 120° is 20π/3 meters. -/
theorem circular_arc_length : 
  ∀ (r : ℝ) (θ : ℝ), 
  r = 10 → 
  θ = 2 * π / 3 → 
  r * θ = 20 * π / 3 := by
sorry

end NUMINAMATH_CALUDE_circular_arc_length_l3433_343349


namespace NUMINAMATH_CALUDE_nineteen_vectors_sum_zero_l3433_343350

theorem nineteen_vectors_sum_zero (v : Fin 19 → (Fin 3 → ZMod 3)) :
  ∃ i j k : Fin 19, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ v i + v j + v k = 0 := by
  sorry

end NUMINAMATH_CALUDE_nineteen_vectors_sum_zero_l3433_343350


namespace NUMINAMATH_CALUDE_solution_set_equivalence_l3433_343389

theorem solution_set_equivalence (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equivalence_l3433_343389


namespace NUMINAMATH_CALUDE_xy_reciprocal_and_ratio_l3433_343302

theorem xy_reciprocal_and_ratio (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x * y = 1) (h4 : x / y = 36) : y = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_xy_reciprocal_and_ratio_l3433_343302


namespace NUMINAMATH_CALUDE_lateral_surface_area_of_specific_prism_l3433_343345

/-- A triangular prism with a regular triangular base and lateral edges perpendicular to the base -/
structure TriangularPrism where
  baseArea : ℝ
  lateralEdgeLength : ℝ

/-- The lateral surface area of a triangular prism -/
def lateralSurfaceArea (prism : TriangularPrism) : ℝ :=
  sorry

theorem lateral_surface_area_of_specific_prism :
  let prism : TriangularPrism := { baseArea := 4 * Real.sqrt 3, lateralEdgeLength := 3 }
  lateralSurfaceArea prism = 36 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_of_specific_prism_l3433_343345


namespace NUMINAMATH_CALUDE_complex_power_100_l3433_343391

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_100 : ((1 + i) / (1 - i)) ^ 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_100_l3433_343391


namespace NUMINAMATH_CALUDE_prime_product_square_l3433_343328

theorem prime_product_square (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p ≠ q → p ≠ r → q ≠ r →
  (p * q * r) % (p + q + r) = 0 →
  ∃ n : ℕ, (p - 1) * (q - 1) * (r - 1) + 1 = n ^ 2 :=
by sorry

end NUMINAMATH_CALUDE_prime_product_square_l3433_343328


namespace NUMINAMATH_CALUDE_cubic_polynomial_distinct_roots_condition_l3433_343305

theorem cubic_polynomial_distinct_roots_condition (p q : ℝ) : 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (∀ x : ℝ, x^3 + p*x + q = (x - a) * (x - b) * (x - c))) →
  p < 0 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_distinct_roots_condition_l3433_343305


namespace NUMINAMATH_CALUDE_swimmer_speed_l3433_343369

theorem swimmer_speed (downstream_distance upstream_distance : ℝ) 
  (downstream_time upstream_time : ℝ) (h1 : downstream_distance = 35) 
  (h2 : upstream_distance = 20) (h3 : downstream_time = 5) (h4 : upstream_time = 5) :
  ∃ (speed_still_water : ℝ), speed_still_water = 5.5 ∧
  ∃ (stream_speed : ℝ),
    (speed_still_water + stream_speed) * downstream_time = downstream_distance ∧
    (speed_still_water - stream_speed) * upstream_time = upstream_distance :=
by sorry

end NUMINAMATH_CALUDE_swimmer_speed_l3433_343369


namespace NUMINAMATH_CALUDE_isosceles_diagonal_probability_l3433_343388

/-- The probability of selecting two diagonals from a regular pentagon 
    such that they form the two legs of an isosceles triangle -/
theorem isosceles_diagonal_probability (n m : ℕ) : 
  n = 10 → m = 5 → (m : ℚ) / n = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_isosceles_diagonal_probability_l3433_343388


namespace NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3433_343339

theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), ∀ x, 4 * x^2 + 1 = 6 * x ↔ a * x^2 + b * x + c = 0 ∧ a = 4 ∧ b = -6 ∧ c = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_coefficients_l3433_343339


namespace NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l3433_343331

/-- Represents an alloy with its chromium percentage and weight -/
structure Alloy where
  chromium_percentage : Float
  weight : Float

/-- Calculates the total chromium weight in an alloy -/
def chromium_weight (a : Alloy) : Float :=
  a.chromium_percentage / 100 * a.weight

/-- Calculates the percentage of chromium in a new alloy formed by combining multiple alloys -/
def new_alloy_chromium_percentage (alloys : List Alloy) : Float :=
  let total_chromium : Float := (alloys.map chromium_weight).sum
  let total_weight : Float := (alloys.map (·.weight)).sum
  total_chromium / total_weight * 100

theorem chromium_percentage_in_new_alloy : 
  let a1 : Alloy := { chromium_percentage := 12, weight := 15 }
  let a2 : Alloy := { chromium_percentage := 10, weight := 35 }
  let a3 : Alloy := { chromium_percentage := 8, weight := 25 }
  let a4 : Alloy := { chromium_percentage := 15, weight := 10 }
  let alloys : List Alloy := [a1, a2, a3, a4]
  new_alloy_chromium_percentage alloys = 10.35 := by
  sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l3433_343331


namespace NUMINAMATH_CALUDE_quadratic_always_real_roots_triangle_perimeter_l3433_343325

-- Define the quadratic equation
def quadratic_equation (x k : ℝ) : Prop :=
  x^2 - (k + 2) * x + 2 * k = 0

-- Theorem 1: The equation always has real roots
theorem quadratic_always_real_roots (k : ℝ) :
  ∃ x : ℝ, quadratic_equation x k := by sorry

-- Define a right triangle with hypotenuse 3 and other sides as roots of the equation
def right_triangle_from_equation (k : ℝ) : Prop :=
  ∃ b c : ℝ,
    quadratic_equation b k ∧
    quadratic_equation c k ∧
    b^2 + c^2 = 3^2

-- Theorem 2: The perimeter of the triangle is 5 + √5
theorem triangle_perimeter (k : ℝ) :
  right_triangle_from_equation k →
  ∃ b c : ℝ, b + c + 3 = 5 + Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_quadratic_always_real_roots_triangle_perimeter_l3433_343325


namespace NUMINAMATH_CALUDE_cos_300_degrees_l3433_343374

theorem cos_300_degrees : Real.cos (300 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_300_degrees_l3433_343374


namespace NUMINAMATH_CALUDE_parabola_shift_left_l3433_343324

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := -2 * p.a * h + p.b,
    c := p.a * h^2 - p.b * h + p.c }

theorem parabola_shift_left :
  let original := Parabola.mk 1 0 2
  let shifted := shift_horizontal original 1
  shifted = Parabola.mk 1 2 3 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_left_l3433_343324


namespace NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l3433_343343

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular : Line → Plane → Prop)

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_planes
  (l : Line) (α β : Plane)
  (h1 : perpendicular l β)
  (h2 : parallel α β) :
  perpendicular l α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_parallel_planes_l3433_343343


namespace NUMINAMATH_CALUDE_bob_question_creation_l3433_343313

theorem bob_question_creation (x : ℕ) : 
  x + 2*x + 4*x = 91 → x = 13 := by
  sorry

end NUMINAMATH_CALUDE_bob_question_creation_l3433_343313


namespace NUMINAMATH_CALUDE_inequality_relation_l3433_343376

theorem inequality_relation (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_relation_l3433_343376


namespace NUMINAMATH_CALUDE_calculate_death_rate_city_death_rate_l3433_343323

/-- Given a birth rate and net population increase, calculate the death rate. -/
theorem calculate_death_rate (birth_rate : ℚ) (net_increase : ℕ) : ℚ :=
  let seconds_per_day : ℕ := 24 * 60 * 60
  let net_increase_per_second : ℚ := net_increase / seconds_per_day
  let death_rate : ℚ := birth_rate - net_increase_per_second
  death_rate

/-- Prove that the death rate is 2 people every 2 seconds given the conditions. -/
theorem city_death_rate :
  calculate_death_rate (6 / 2) 172800 = 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_death_rate_city_death_rate_l3433_343323


namespace NUMINAMATH_CALUDE_holly_weekly_pill_count_l3433_343314

/-- Calculates the total number of pills Holly takes in a week -/
def total_weekly_pills : ℕ :=
  let insulin_daily := 2
  let bp_daily := 3
  let anticonvulsant_daily := 2 * bp_daily
  let calcium_every_other_day := 3 * insulin_daily
  let vitamin_d_twice_weekly := 4
  let multivitamin_thrice_weekly := 1
  let anxiety_sunday := 3 * bp_daily

  7 * insulin_daily + 
  7 * bp_daily + 
  7 * anticonvulsant_daily + 
  (7 / 2) * calcium_every_other_day +
  2 * vitamin_d_twice_weekly + 
  3 * multivitamin_thrice_weekly + 
  anxiety_sunday

theorem holly_weekly_pill_count : total_weekly_pills = 118 := by
  sorry

end NUMINAMATH_CALUDE_holly_weekly_pill_count_l3433_343314


namespace NUMINAMATH_CALUDE_paige_homework_problems_l3433_343398

/-- The initial number of homework problems Paige had -/
def initial_problems (finished : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) : ℕ :=
  finished + remaining_pages * problems_per_page

/-- Theorem stating that Paige initially had 110 homework problems -/
theorem paige_homework_problems :
  initial_problems 47 7 9 = 110 := by
  sorry

end NUMINAMATH_CALUDE_paige_homework_problems_l3433_343398


namespace NUMINAMATH_CALUDE_candidate_votes_l3433_343373

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15/100 →
  candidate_percent = 65/100 →
  (1 - invalid_percent) * candidate_percent * total_votes = 309400 := by
  sorry

end NUMINAMATH_CALUDE_candidate_votes_l3433_343373


namespace NUMINAMATH_CALUDE_half_percent_of_160_l3433_343368

theorem half_percent_of_160 : (1 / 2 * 1 / 100) * 160 = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_half_percent_of_160_l3433_343368


namespace NUMINAMATH_CALUDE_angle_position_l3433_343306

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- An angle in the 2D plane -/
structure Angle where
  -- We don't need to define the internal structure of an angle for this problem

/-- The terminal side of an angle -/
def terminal_side (α : Angle) : Set Point :=
  sorry -- Definition not needed for the statement

/-- Predicate to check if a point is on the non-negative side of the y-axis -/
def on_nonnegative_y_side (p : Point) : Prop :=
  p.x = 0 ∧ p.y ≥ 0

theorem angle_position (α : Angle) (P : Point) :
  P ∈ terminal_side α →
  P = ⟨0, 3⟩ →
  ∃ (p : Point), p ∈ terminal_side α ∧ on_nonnegative_y_side p :=
sorry

end NUMINAMATH_CALUDE_angle_position_l3433_343306


namespace NUMINAMATH_CALUDE_half_power_inequality_l3433_343382

theorem half_power_inequality (a : ℝ) : 
  (1/2 : ℝ)^(2*a + 1) < (1/2 : ℝ)^(3 - 2*a) → a > 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_half_power_inequality_l3433_343382


namespace NUMINAMATH_CALUDE_sale_price_calculation_l3433_343310

theorem sale_price_calculation (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := 0.8 * original_price
  let final_price := 0.9 * first_sale_price
  final_price / original_price = 0.72 :=
by sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l3433_343310


namespace NUMINAMATH_CALUDE_alteration_cost_per_shoe_l3433_343386

-- Define the number of pairs of shoes
def num_pairs : ℕ := 14

-- Define the total cost of alteration
def total_cost : ℕ := 1036

-- Define the cost per shoe
def cost_per_shoe : ℕ := 37

-- Theorem statement
theorem alteration_cost_per_shoe :
  (total_cost : ℚ) / (2 * num_pairs) = cost_per_shoe := by
  sorry

end NUMINAMATH_CALUDE_alteration_cost_per_shoe_l3433_343386


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3433_343392

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {5, 6, 7}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {4, 8} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3433_343392


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3433_343355

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define the exterior angle for this problem
  exteriorAngle : ℝ

-- Theorem statement
theorem isosceles_triangle_vertex_angle 
  (triangle : IsoscelesTriangle) 
  (h : triangle.exteriorAngle = 40) : 
  -- The vertex angle is 140°
  (180 - triangle.exteriorAngle) = 140 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l3433_343355


namespace NUMINAMATH_CALUDE_quadratic_sum_l3433_343301

def quadratic_function (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℤ) : 
  (quadratic_function a b c 0 = 2) → 
  (∀ x, quadratic_function a b c x ≥ quadratic_function a b c 1) →
  (quadratic_function a b c 1 = -1) →
  a - b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3433_343301


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3433_343336

/-- The length of the real axis of a hyperbola with equation x²/9 - y² = 1 is 6. -/
theorem hyperbola_real_axis_length :
  ∃ (f : ℝ → ℝ → Prop),
    (∀ x y, f x y ↔ x^2/9 - y^2 = 1) →
    (∃ a : ℝ, a > 0 ∧ ∀ x y, f x y ↔ x^2/a^2 - y^2 = 1) →
    2 * Real.sqrt 9 = 6 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3433_343336


namespace NUMINAMATH_CALUDE_billys_age_l3433_343379

theorem billys_age (B J S : ℕ) 
  (h1 : B = 2 * J) 
  (h2 : B + J = 3 * S) 
  (h3 : S = 27) : 
  B = 54 := by
  sorry

end NUMINAMATH_CALUDE_billys_age_l3433_343379


namespace NUMINAMATH_CALUDE_circle_center_l3433_343316

/-- The center of a circle given by the equation x^2 - 8x + y^2 - 4y = 5 is (4, 2) -/
theorem circle_center (x y : ℝ) : x^2 - 8*x + y^2 - 4*y = 5 → (4, 2) = (x, y) := by
  sorry

end NUMINAMATH_CALUDE_circle_center_l3433_343316


namespace NUMINAMATH_CALUDE_soft_drink_added_sugar_percentage_l3433_343322

theorem soft_drink_added_sugar_percentage (
  soft_drink_calories : ℕ)
  (candy_bar_sugar_calories : ℕ)
  (candy_bars_taken : ℕ)
  (recommended_sugar_intake : ℕ)
  (exceeded_percentage : ℕ)
  (h1 : soft_drink_calories = 2500)
  (h2 : candy_bar_sugar_calories = 25)
  (h3 : candy_bars_taken = 7)
  (h4 : recommended_sugar_intake = 150)
  (h5 : exceeded_percentage = 100) :
  (((recommended_sugar_intake * (100 + exceeded_percentage) / 100) -
    (candy_bar_sugar_calories * candy_bars_taken)) * 100) /
    soft_drink_calories = 5 := by
  sorry

end NUMINAMATH_CALUDE_soft_drink_added_sugar_percentage_l3433_343322


namespace NUMINAMATH_CALUDE_contractor_problem_solution_correctness_l3433_343377

/-- Represents the number of days required to complete the work -/
def original_days : ℕ := 9

/-- Represents the number of absent laborers -/
def absent_laborers : ℕ := 6

/-- Represents the number of days required to complete the work with absent laborers -/
def new_days : ℕ := 15

/-- Represents the original number of laborers -/
def original_laborers : ℕ := 15

theorem contractor_problem :
  original_laborers * original_days = (original_laborers - absent_laborers) * new_days :=
by sorry

theorem solution_correctness :
  original_laborers = 15 :=
by sorry

end NUMINAMATH_CALUDE_contractor_problem_solution_correctness_l3433_343377


namespace NUMINAMATH_CALUDE_part_one_part_two_l3433_343364

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x - a| + a
def g (x : ℝ) : ℝ := |2 * x - 1|

-- Part I
theorem part_one : 
  {x : ℝ | f 2 x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

-- Part II
theorem part_two :
  (∀ x : ℝ, f a x + g x ≥ 3) → a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3433_343364


namespace NUMINAMATH_CALUDE_milk_problem_l3433_343319

theorem milk_problem (initial_milk : ℚ) (rachel_fraction : ℚ) (sam_fraction : ℚ) : 
  initial_milk = 3/4 →
  rachel_fraction = 1/2 →
  sam_fraction = 1/3 →
  sam_fraction * (initial_milk - rachel_fraction * initial_milk) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_milk_problem_l3433_343319


namespace NUMINAMATH_CALUDE_metallic_sheet_length_l3433_343395

/-- Proves that a rectangular sheet with one side 36 m, when cut to form a box of volume 3780 m³, has an original length of 48 m -/
theorem metallic_sheet_length (L : ℝ) : 
  L > 0 → 
  (L - 6) * (36 - 6) * 3 = 3780 → 
  L = 48 :=
by sorry

end NUMINAMATH_CALUDE_metallic_sheet_length_l3433_343395


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3433_343360

theorem complex_fraction_equality : Complex.I ^ 2 = -1 → (3 + Complex.I) / (1 - Complex.I) = 1 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3433_343360


namespace NUMINAMATH_CALUDE_integral_special_function_l3433_343311

theorem integral_special_function : 
  ∫ x in (0 : ℝ)..(Real.pi / 2), (1 - 5 * x^2) * Real.sin x = 11 - 5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_integral_special_function_l3433_343311


namespace NUMINAMATH_CALUDE_expression_evaluation_l3433_343362

theorem expression_evaluation (a : ℝ) (h : a^2 + 2*a - 1 = 0) :
  (((a^2 - 1) / (a^2 - 2*a + 1) - 1 / (1 - a)) / (1 / (a^2 - a))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3433_343362


namespace NUMINAMATH_CALUDE_min_value_expression_l3433_343378

theorem min_value_expression (a b c k : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hk : 0 < k) :
  (a^2 / (k*b)) + (b^2 / (k*c)) + (c^2 / (k*a)) ≥ 3/k ∧
  ((a^2 / (k*b)) + (b^2 / (k*c)) + (c^2 / (k*a)) = 3/k ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l3433_343378


namespace NUMINAMATH_CALUDE_rocket_components_most_suitable_for_comprehensive_survey_l3433_343348

/-- Represents the characteristics of a scenario that can be surveyed -/
structure SurveyScenario where
  population : Type
  countable : Bool
  criticalImportance : Bool
  requiresCompleteExamination : Bool

/-- Defines what makes a scenario suitable for a comprehensive survey -/
def isSuitableForComprehensiveSurvey (scenario : SurveyScenario) : Prop :=
  scenario.countable ∧ scenario.criticalImportance ∧ scenario.requiresCompleteExamination

/-- Represents the Long March II-F Y17 rocket components scenario -/
def rocketComponentsScenario : SurveyScenario :=
  { population := Unit,  -- The type doesn't matter for this example
    countable := true,
    criticalImportance := true,
    requiresCompleteExamination := true }

/-- Represents all other given scenarios -/
def otherScenarios : List SurveyScenario :=
  [ { population := Unit, countable := false, criticalImportance := false, requiresCompleteExamination := false },
    { population := Unit, countable := false, criticalImportance := true, requiresCompleteExamination := false },
    { population := Unit, countable := false, criticalImportance := false, requiresCompleteExamination := false } ]

theorem rocket_components_most_suitable_for_comprehensive_survey :
  isSuitableForComprehensiveSurvey rocketComponentsScenario ∧
  (∀ scenario ∈ otherScenarios, ¬(isSuitableForComprehensiveSurvey scenario)) :=
sorry

end NUMINAMATH_CALUDE_rocket_components_most_suitable_for_comprehensive_survey_l3433_343348


namespace NUMINAMATH_CALUDE_total_students_count_l3433_343381

/-- Represents the arrangement of students in two rows -/
structure StudentArrangement where
  boys_count : ℕ
  girls_count : ℕ
  rajan_left_position : ℕ
  vinay_right_position : ℕ
  boys_between_rajan_vinay : ℕ
  deepa_left_position : ℕ

/-- The total number of students in both rows -/
def total_students (arrangement : StudentArrangement) : ℕ :=
  arrangement.boys_count + arrangement.girls_count

/-- The theorem stating the total number of students given the conditions -/
theorem total_students_count (arrangement : StudentArrangement) 
  (h1 : arrangement.boys_count = arrangement.girls_count)
  (h2 : arrangement.rajan_left_position = 6)
  (h3 : arrangement.vinay_right_position = 10)
  (h4 : arrangement.boys_between_rajan_vinay = 8)
  (h5 : arrangement.deepa_left_position = 5)
  : total_students arrangement = 48 := by
  sorry

end NUMINAMATH_CALUDE_total_students_count_l3433_343381


namespace NUMINAMATH_CALUDE_race_coin_problem_l3433_343396

theorem race_coin_problem (x y : ℕ) (h1 : x > y) (h2 : y > 0) : 
  (∃ n : ℕ, n > 2 ∧ 
   (n - 2) * x + 2 * y = 42 ∧ 
   2 * x + (n - 2) * y = 35) → 
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_race_coin_problem_l3433_343396


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l3433_343312

def number : ℕ := 32767

/-- The greatest prime divisor of a natural number -/
def greatest_prime_divisor (n : ℕ) : ℕ := sorry

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor number) = 14 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l3433_343312


namespace NUMINAMATH_CALUDE_doctors_lawyers_ratio_l3433_343356

theorem doctors_lawyers_ratio (d l : ℕ) (h1 : d + l > 0) :
  (40 * d + 55 * l : ℚ) / (d + l : ℚ) = 45 →
  (d : ℚ) / (l : ℚ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_doctors_lawyers_ratio_l3433_343356


namespace NUMINAMATH_CALUDE_carlos_welfare_fund_contribution_l3433_343397

/-- The amount in cents dedicated to the welfare fund per hour -/
def welfare_fund_cents (hourly_wage : ℝ) (deduction_rate : ℝ) : ℝ :=
  hourly_wage * 100 * deduction_rate

/-- Proof that Carlos' welfare fund contribution is 40 cents per hour -/
theorem carlos_welfare_fund_contribution :
  welfare_fund_cents 25 0.016 = 40 := by
  sorry

end NUMINAMATH_CALUDE_carlos_welfare_fund_contribution_l3433_343397


namespace NUMINAMATH_CALUDE_project_completion_time_l3433_343332

/-- The number of days it takes for person A to complete the project alone -/
def days_A : ℝ := 20

/-- The number of days it takes for person B to complete the project alone -/
def days_B : ℝ := 40

/-- The total duration of the project when A and B work together, and A quits 10 days before completion -/
def total_days : ℝ := 20

/-- The number of days A works before quitting -/
def days_A_works : ℝ := total_days - 10

theorem project_completion_time :
  (days_A_works * (1 / days_A + 1 / days_B)) + (10 * (1 / days_B)) = 1 :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l3433_343332


namespace NUMINAMATH_CALUDE_divisibility_of_Z_l3433_343307

/-- Represents a 7-digit positive integer in the form abcabca -/
def Z (a b c : ℕ) : ℕ :=
  1000000 * a + 100000 * b + 10000 * c + 1000 * a + 100 * b + 10 * c + a

/-- Theorem stating that 1001 divides Z for any valid a, b, c -/
theorem divisibility_of_Z (a b c : ℕ) (ha : 0 < a) (ha' : a < 10) (hb : b < 10) (hc : c < 10) :
  1001 ∣ Z a b c := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_Z_l3433_343307


namespace NUMINAMATH_CALUDE_gcd_of_324_243_135_l3433_343387

theorem gcd_of_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_324_243_135_l3433_343387


namespace NUMINAMATH_CALUDE_smallest_congruent_integer_l3433_343340

theorem smallest_congruent_integer (n : ℕ) : 
  (0 ≤ n ∧ n ≤ 15) ∧ n ≡ 5673 [MOD 16] → n = 9 :=
by sorry

end NUMINAMATH_CALUDE_smallest_congruent_integer_l3433_343340


namespace NUMINAMATH_CALUDE_dart_board_probability_l3433_343352

theorem dart_board_probability (r : ℝ) (h : r = 10) :
  let circle_area := π * r^2
  let square_side := r * Real.sqrt 2
  let square_area := square_side^2
  square_area / circle_area = 2 / π := by sorry

end NUMINAMATH_CALUDE_dart_board_probability_l3433_343352


namespace NUMINAMATH_CALUDE_olympiad_scores_l3433_343359

theorem olympiad_scores (scores : Fin 20 → ℕ) 
  (distinct : ∀ i j, i ≠ j → scores i ≠ scores j)
  (sum_condition : ∀ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k → scores i < scores j + scores k) :
  ∀ i, scores i > 18 := by
  sorry

end NUMINAMATH_CALUDE_olympiad_scores_l3433_343359


namespace NUMINAMATH_CALUDE_x_sum_greater_than_two_over_a_l3433_343347

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp (a * x) * Real.log x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 
  (deriv (f a) x) / Real.exp (a * x)

theorem x_sum_greater_than_two_over_a 
  (a : ℝ) (x₁ x₂ : ℝ) (ha : a > 0) (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hx_dist : x₁ ≠ x₂) (hg_eq_f : g a x₁ = f a x₂) : 
  x₁ + x₂ > 2 / a := by
  sorry

end NUMINAMATH_CALUDE_x_sum_greater_than_two_over_a_l3433_343347


namespace NUMINAMATH_CALUDE_lakers_win_in_seven_l3433_343308

/-- The probability of the Celtics winning a single game -/
def p_celtics : ℚ := 2/3

/-- The probability of the Lakers winning a single game -/
def p_lakers : ℚ := 1 - p_celtics

/-- The number of ways to choose 3 games out of 6 -/
def ways_to_choose_3_of_6 : ℕ := 20

/-- The probability that the Lakers win the NBA finals in exactly 7 games -/
theorem lakers_win_in_seven (p_celtics : ℚ) (p_lakers : ℚ) (ways_to_choose_3_of_6 : ℕ) :
  p_celtics = 2/3 →
  p_lakers = 1 - p_celtics →
  ways_to_choose_3_of_6 = 20 →
  (ways_to_choose_3_of_6 : ℚ) * p_lakers^3 * p_celtics^3 * p_lakers = 160/2187 :=
by sorry

end NUMINAMATH_CALUDE_lakers_win_in_seven_l3433_343308


namespace NUMINAMATH_CALUDE_simplify_expression_l3433_343375

theorem simplify_expression : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) * (2^16 + 1) = 2^32 - 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3433_343375


namespace NUMINAMATH_CALUDE_root_in_interval_l3433_343353

-- Define the function g(x) = lg x + x - 2
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 10 + x - 2

-- State the theorem
theorem root_in_interval :
  ∃ x₀ : ℝ, g x₀ = 0 ∧ 1 < x₀ ∧ x₀ < 2 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l3433_343353


namespace NUMINAMATH_CALUDE_smallest_percent_increase_l3433_343309

def question_value : Fin 15 → ℕ
  | 0 => 100
  | 1 => 200
  | 2 => 300
  | 3 => 500
  | 4 => 1000
  | 5 => 2000
  | 6 => 4000
  | 7 => 8000
  | 8 => 16000
  | 9 => 32000
  | 10 => 64000
  | 11 => 125000
  | 12 => 250000
  | 13 => 500000
  | 14 => 1000000

def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

def options : List (Fin 15 × Fin 15) :=
  [(0, 1), (1, 2), (2, 3), (10, 11), (13, 14)]

theorem smallest_percent_increase :
  ∀ (pair : Fin 15 × Fin 15), pair ∈ options →
    percent_increase (question_value pair.1) (question_value pair.2) ≥
    percent_increase (question_value 1) (question_value 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_percent_increase_l3433_343309


namespace NUMINAMATH_CALUDE_lucy_shooting_problem_l3433_343342

/-- Lucy's basketball shooting problem -/
theorem lucy_shooting_problem
  (initial_shots : ℕ)
  (initial_percentage : ℚ)
  (additional_shots : ℕ)
  (new_percentage : ℚ)
  (h1 : initial_shots = 30)
  (h2 : initial_percentage = 3/5)
  (h3 : additional_shots = 10)
  (h4 : new_percentage = 31/50)
  : ⌊(new_percentage * (initial_shots + additional_shots) : ℚ)⌋ - 
    ⌊(initial_percentage * initial_shots : ℚ)⌋ = 7 := by
  sorry

#check lucy_shooting_problem

end NUMINAMATH_CALUDE_lucy_shooting_problem_l3433_343342


namespace NUMINAMATH_CALUDE_line_intersects_circle_l3433_343327

/-- The line l in polar coordinates -/
def line_l (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + Real.pi / 4) = Real.sqrt 2

/-- The circle C in Cartesian coordinates -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

/-- The line l in Cartesian coordinates -/
def line_l_cartesian (x y : ℝ) : Prop := x + y = 2

/-- Theorem stating that the line l intersects the circle C -/
theorem line_intersects_circle :
  ∃ (x y : ℝ), line_l_cartesian x y ∧ circle_C x y :=
sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l3433_343327


namespace NUMINAMATH_CALUDE_expected_value_is_thirteen_eighths_l3433_343315

/-- Represents the outcome of rolling an 8-sided die -/
inductive DieRoll
  | one
  | two
  | three
  | four
  | five
  | six
  | seven
  | eight

/-- Determines if a DieRoll is prime -/
def isPrime (roll : DieRoll) : Bool :=
  match roll with
  | DieRoll.two | DieRoll.three | DieRoll.five | DieRoll.seven => true
  | _ => false

/-- Calculates the winnings for a given DieRoll -/
def winnings (roll : DieRoll) : Int :=
  match roll with
  | DieRoll.two => 2
  | DieRoll.three => 3
  | DieRoll.five => 5
  | DieRoll.seven => 7
  | DieRoll.eight => -4
  | _ => 0

/-- The probability of each DieRoll -/
def probability : DieRoll → Rat
  | _ => 1/8

/-- The expected value of the winnings -/
def expectedValue : Rat :=
  (winnings DieRoll.one   * probability DieRoll.one)   +
  (winnings DieRoll.two   * probability DieRoll.two)   +
  (winnings DieRoll.three * probability DieRoll.three) +
  (winnings DieRoll.four  * probability DieRoll.four)  +
  (winnings DieRoll.five  * probability DieRoll.five)  +
  (winnings DieRoll.six   * probability DieRoll.six)   +
  (winnings DieRoll.seven * probability DieRoll.seven) +
  (winnings DieRoll.eight * probability DieRoll.eight)

theorem expected_value_is_thirteen_eighths :
  expectedValue = 13/8 := by sorry

end NUMINAMATH_CALUDE_expected_value_is_thirteen_eighths_l3433_343315


namespace NUMINAMATH_CALUDE_degree_of_polynomial_power_l3433_343358

/-- The degree of the polynomial (5x^3 + 7)^10 is 30. -/
theorem degree_of_polynomial_power : 
  Polynomial.degree ((5 * X ^ 3 + 7 : Polynomial ℝ) ^ 10) = 30 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_polynomial_power_l3433_343358


namespace NUMINAMATH_CALUDE_intersection_implies_m_equals_three_subset_complement_implies_m_range_l3433_343357

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*m*x - 4 ≤ 0}

-- Theorem 1
theorem intersection_implies_m_equals_three (m : ℝ) :
  A ∩ B m = {x | 1 ≤ x ∧ x ≤ 3} → m = 3 := by sorry

-- Theorem 2
theorem subset_complement_implies_m_range (m : ℝ) :
  A ⊆ (B m)ᶜ → m < -3 ∨ m > 5 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_m_equals_three_subset_complement_implies_m_range_l3433_343357


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_4_hours_l3433_343399

-- Define the motion equation
def s (t : ℝ) : ℝ := t^3 + t^2

-- Define the instantaneous velocity function
def v (t : ℝ) : ℝ := 3 * t^2 + 2 * t

-- Theorem statement
theorem instantaneous_velocity_at_4_hours (h : 4 > 0) :
  v 4 = 56 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_4_hours_l3433_343399


namespace NUMINAMATH_CALUDE_three_not_in_range_l3433_343341

def g (c : ℝ) (x : ℝ) : ℝ := x^2 + c*x + 5

theorem three_not_in_range (c : ℝ) :
  (∀ x, g c x ≠ 3) ↔ c ∈ Set.Ioo (-2 * Real.sqrt 2) (2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_three_not_in_range_l3433_343341


namespace NUMINAMATH_CALUDE_school_play_tickets_l3433_343317

theorem school_play_tickets (total_money : ℕ) (adult_price child_price : ℕ) 
  (child_tickets : ℕ) :
  total_money = 104 →
  adult_price = 6 →
  child_price = 4 →
  child_tickets = 11 →
  ∃ (adult_tickets : ℕ), 
    adult_price * adult_tickets + child_price * child_tickets = total_money ∧
    adult_tickets + child_tickets = 21 := by
  sorry

end NUMINAMATH_CALUDE_school_play_tickets_l3433_343317


namespace NUMINAMATH_CALUDE_circle_max_area_l3433_343361

/-- Given a circle equation with parameter m, prove that when the area is maximum, 
    the standard equation of the circle is (x-1)^2 + (y+3)^2 = 1 -/
theorem circle_max_area (x y m : ℝ) : 
  (∃ r, x^2 + y^2 - 2*x + 2*m*y + 2*m^2 - 6*m + 9 = 0 ↔ (x-1)^2 + (y+m)^2 = r^2) →
  (∀ m', ∃ r', x^2 + y^2 - 2*x + 2*m'*y + 2*m'^2 - 6*m' + 9 = 0 → 
    (x-1)^2 + (y+m')^2 = r'^2 ∧ r'^2 ≤ 1) →
  (x-1)^2 + (y+3)^2 = 1 := by sorry

end NUMINAMATH_CALUDE_circle_max_area_l3433_343361


namespace NUMINAMATH_CALUDE_fliers_remaining_l3433_343351

theorem fliers_remaining (initial_fliers : ℕ) 
  (morning_fraction : ℚ) (afternoon_fraction : ℚ) : 
  initial_fliers = 3000 →
  morning_fraction = 1/5 →
  afternoon_fraction = 1/4 →
  let remaining_after_morning := initial_fliers - (morning_fraction * initial_fliers).floor
  let final_remaining := remaining_after_morning - (afternoon_fraction * remaining_after_morning).floor
  final_remaining = 1800 := by
  sorry

end NUMINAMATH_CALUDE_fliers_remaining_l3433_343351


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l3433_343337

theorem modular_congruence_solution : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -4376 [ZMOD 10] ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l3433_343337


namespace NUMINAMATH_CALUDE_abc_remainder_l3433_343300

theorem abc_remainder (a b c : ℕ) : 
  a < 9 → b < 9 → c < 9 →
  (a + 3*b + 2*c) % 9 = 3 →
  (2*a + 2*b + 3*c) % 9 = 6 →
  (3*a + b + 2*c) % 9 = 1 →
  (a*b*c) % 9 = 4 := by
sorry

end NUMINAMATH_CALUDE_abc_remainder_l3433_343300


namespace NUMINAMATH_CALUDE_clothing_distribution_l3433_343380

theorem clothing_distribution (total : ℕ) (first_load : ℕ) (num_small_loads : ℕ) 
  (h1 : total = 59)
  (h2 : first_load = 32)
  (h3 : num_small_loads = 9)
  (h4 : first_load < total) :
  (total - first_load) / num_small_loads = 3 := by
  sorry

end NUMINAMATH_CALUDE_clothing_distribution_l3433_343380


namespace NUMINAMATH_CALUDE_tangent_circles_expression_l3433_343385

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- The distance between the centers of two tangent circles is the sum of their radii -/
def distance (c1 c2 : Circle) : ℝ := c1.radius + c2.radius

theorem tangent_circles_expression (a b c : ℝ) (A B C : Circle)
  (ha : A.radius = a)
  (hb : B.radius = b)
  (hc : C.radius = c)
  (hab : a > b)
  (hbc : b > c)
  (htangent : A.radius + B.radius = distance A B ∧ 
              B.radius + C.radius = distance B C ∧ 
              C.radius + A.radius = distance C A) :
  distance A B + distance B C - distance C A = b ∧ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circles_expression_l3433_343385


namespace NUMINAMATH_CALUDE_shortest_distance_principle_applies_l3433_343321

-- Define the phenomena
inductive Phenomenon
  | woodenBarFixing
  | treePlanting
  | electricWireLaying
  | roadStraightening

-- Define the principle
def shortestDistancePrinciple (p : Phenomenon) : Prop :=
  match p with
  | Phenomenon.electricWireLaying => true
  | Phenomenon.roadStraightening => true
  | _ => false

-- Theorem statement
theorem shortest_distance_principle_applies :
  (∀ p : Phenomenon, shortestDistancePrinciple p ↔ 
    (p = Phenomenon.electricWireLaying ∨ p = Phenomenon.roadStraightening)) := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_principle_applies_l3433_343321


namespace NUMINAMATH_CALUDE_consecutive_interior_angles_indeterminate_l3433_343371

-- Define consecutive interior angles
def consecutive_interior_angles (α β : ℝ) : Prop := sorry

-- Theorem statement
theorem consecutive_interior_angles_indeterminate (α β : ℝ) :
  consecutive_interior_angles α β → α = 55 → ¬∃!β, consecutive_interior_angles α β :=
sorry

end NUMINAMATH_CALUDE_consecutive_interior_angles_indeterminate_l3433_343371


namespace NUMINAMATH_CALUDE_one_sixth_star_neg_one_l3433_343320

-- Define the ※ operation for rational numbers
def star_op (m n : ℚ) : ℚ := (3*m + n) * (3*m - n) + n

-- State the theorem
theorem one_sixth_star_neg_one :
  star_op (1/6) (-1) = -7/4 := by sorry

end NUMINAMATH_CALUDE_one_sixth_star_neg_one_l3433_343320


namespace NUMINAMATH_CALUDE_sum_of_f_values_l3433_343394

noncomputable def f (x : ℝ) : ℝ := 2 / (2^x + 1) + Real.sin x

theorem sum_of_f_values : 
  f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 = 7 := by sorry

end NUMINAMATH_CALUDE_sum_of_f_values_l3433_343394


namespace NUMINAMATH_CALUDE_no_gcd_inverting_function_l3433_343329

theorem no_gcd_inverting_function :
  ¬ (∃ f : ℕ+ → ℕ+, ∀ a b : ℕ+, Nat.gcd a.val b.val = 1 ↔ Nat.gcd (f a).val (f b).val > 1) :=
sorry

end NUMINAMATH_CALUDE_no_gcd_inverting_function_l3433_343329


namespace NUMINAMATH_CALUDE_sandy_change_l3433_343370

def pants_cost : Float := 13.58
def shirt_cost : Float := 10.29
def sweater_cost : Float := 24.97
def shoes_cost : Float := 39.99
def paid_amount : Float := 100.00

def total_cost : Float := pants_cost + shirt_cost + sweater_cost + shoes_cost

theorem sandy_change : paid_amount - total_cost = 11.17 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_l3433_343370


namespace NUMINAMATH_CALUDE_opposite_lime_is_black_l3433_343363

-- Define the colors
inductive Color
  | Purple
  | Cyan
  | Magenta
  | Silver
  | Black
  | Lime

-- Define a square with a color
structure Square where
  color : Color

-- Define a cube made of squares
structure Cube where
  squares : List Square
  hinged : squares.length = 6

-- Define the opposite face relation
def oppositeFace (c : Cube) (f1 f2 : Square) : Prop :=
  f1 ∈ c.squares ∧ f2 ∈ c.squares ∧ f1 ≠ f2

-- Theorem statement
theorem opposite_lime_is_black (c : Cube) :
  ∃ (lime_face black_face : Square),
    lime_face.color = Color.Lime ∧
    black_face.color = Color.Black ∧
    oppositeFace c lime_face black_face :=
  sorry


end NUMINAMATH_CALUDE_opposite_lime_is_black_l3433_343363
