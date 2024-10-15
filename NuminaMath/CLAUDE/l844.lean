import Mathlib

namespace NUMINAMATH_CALUDE_investment_interest_calculation_l844_84401

theorem investment_interest_calculation
  (total_investment : ℝ)
  (investment_at_6_percent : ℝ)
  (interest_rate_6_percent : ℝ)
  (interest_rate_9_percent : ℝ)
  (h1 : total_investment = 10000)
  (h2 : investment_at_6_percent = 7200)
  (h3 : interest_rate_6_percent = 0.06)
  (h4 : interest_rate_9_percent = 0.09) :
  let investment_at_9_percent := total_investment - investment_at_6_percent
  let interest_from_6_percent := investment_at_6_percent * interest_rate_6_percent
  let interest_from_9_percent := investment_at_9_percent * interest_rate_9_percent
  let total_interest := interest_from_6_percent + interest_from_9_percent
  total_interest = 684 := by sorry

end NUMINAMATH_CALUDE_investment_interest_calculation_l844_84401


namespace NUMINAMATH_CALUDE_divisibility_condition_l844_84467

theorem divisibility_condition (n : ℕ+) : (n^2 + 1) ∣ (n + 1) ↔ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_condition_l844_84467


namespace NUMINAMATH_CALUDE_set_equality_implies_values_l844_84403

theorem set_equality_implies_values (A B : Set ℝ) (x y : ℝ) :
  A = {3, 4, x} → B = {2, 3, y} → A = B → x = 2 ∧ y = 4 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_values_l844_84403


namespace NUMINAMATH_CALUDE_wand_price_l844_84441

theorem wand_price (price : ℝ) (original_price : ℝ) : 
  price = 12 → price = (1/8) * original_price → original_price = 96 := by
sorry

end NUMINAMATH_CALUDE_wand_price_l844_84441


namespace NUMINAMATH_CALUDE_extreme_point_implies_a_zero_l844_84432

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x^2 - (a + 2) * x + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * x - (a + 2)

theorem extreme_point_implies_a_zero :
  ∀ a : ℝ, (f_derivative a 1 = 0) → a = 0 :=
by sorry

#check extreme_point_implies_a_zero

end NUMINAMATH_CALUDE_extreme_point_implies_a_zero_l844_84432


namespace NUMINAMATH_CALUDE_number_of_brown_dogs_l844_84420

/-- Given a group of dogs with white, black, and brown colors, 
    prove that the number of brown dogs is 20. -/
theorem number_of_brown_dogs 
  (total : ℕ) 
  (white : ℕ) 
  (black : ℕ) 
  (h1 : total = 45) 
  (h2 : white = 10) 
  (h3 : black = 15) : 
  total - (white + black) = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_of_brown_dogs_l844_84420


namespace NUMINAMATH_CALUDE_product_of_primes_summing_to_91_l844_84471

theorem product_of_primes_summing_to_91 (p q : ℕ) : 
  Prime p → Prime q → p + q = 91 → p * q = 178 := by sorry

end NUMINAMATH_CALUDE_product_of_primes_summing_to_91_l844_84471


namespace NUMINAMATH_CALUDE_bucket_capacity_l844_84431

/-- Calculates the capacity of a bucket used to fill a pool -/
theorem bucket_capacity
  (fill_time : ℕ)           -- Time to fill and empty one bucket (in seconds)
  (pool_capacity : ℕ)       -- Capacity of the pool (in gallons)
  (total_time : ℕ)          -- Total time to fill the pool (in minutes)
  (h1 : fill_time = 20)     -- Given: Time to fill and empty one bucket is 20 seconds
  (h2 : pool_capacity = 84) -- Given: Pool capacity is 84 gallons
  (h3 : total_time = 14)    -- Given: Total time to fill the pool is 14 minutes
  : ℕ := by
  sorry

#check bucket_capacity

end NUMINAMATH_CALUDE_bucket_capacity_l844_84431


namespace NUMINAMATH_CALUDE_tens_digit_of_power_five_l844_84458

theorem tens_digit_of_power_five : ∃ (n : ℕ), 5^(5^5) ≡ 25 [MOD 100] ∧ n = 2 := by sorry

end NUMINAMATH_CALUDE_tens_digit_of_power_five_l844_84458


namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l844_84429

theorem fraction_sum_theorem (a b c d x y z w : ℝ) 
  (h1 : x / a + y / b + z / c + w / d = 4)
  (h2 : a / x + b / y + c / z + d / w = 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 + w^2 / d^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l844_84429


namespace NUMINAMATH_CALUDE_train_speed_Q_l844_84435

/-- The distance between stations P and Q in kilometers -/
def distance_PQ : ℝ := 65

/-- The speed of the train starting from station P in kilometers per hour -/
def speed_P : ℝ := 20

/-- The time difference between the start of the two trains in hours -/
def time_difference : ℝ := 1

/-- The total time until the trains meet in hours -/
def total_time : ℝ := 2

/-- The speed of the train starting from station Q in kilometers per hour -/
def speed_Q : ℝ := 25

theorem train_speed_Q : speed_Q = (distance_PQ - speed_P * total_time) / (total_time - time_difference) :=
sorry

end NUMINAMATH_CALUDE_train_speed_Q_l844_84435


namespace NUMINAMATH_CALUDE_polynomial_transformation_l844_84473

/-- Given y = x + 1/x, prove that x^6 + x^5 - 5x^4 + x^3 + 3x^2 + x + 1 = 0 is equivalent to x^4*y^2 - 4*x^2*y^2 + 3*x^2 = 0 -/
theorem polynomial_transformation (x y : ℝ) (h : y = x + 1/x) :
  x^6 + x^5 - 5*x^4 + x^3 + 3*x^2 + x + 1 = 0 ↔ x^4*y^2 - 4*x^2*y^2 + 3*x^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_transformation_l844_84473


namespace NUMINAMATH_CALUDE_recurrence_sequence_property_l844_84462

/-- A sequence of integers satisfying the recurrence relation a_{n+2} = a_{n+1} - m * a_n -/
def RecurrenceSequence (m : ℤ) (a : ℕ → ℤ) : Prop :=
  (a 1 ≠ 0 ∨ a 2 ≠ 0) ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) - m * a n

/-- The main theorem -/
theorem recurrence_sequence_property (m : ℤ) (a : ℕ → ℤ) 
    (hm : |m| ≥ 2) 
    (ha : RecurrenceSequence m a) 
    (r s : ℕ) 
    (hrs : r > s ∧ s ≥ 2) 
    (heq : a r = a s ∧ a s = a 1) : 
  r - s ≥ |m| := by
  sorry

end NUMINAMATH_CALUDE_recurrence_sequence_property_l844_84462


namespace NUMINAMATH_CALUDE_initial_students_count_l844_84438

/-- The number of students who got off the bus at the first stop -/
def students_off : ℕ := 3

/-- The number of students remaining on the bus after the first stop -/
def students_remaining : ℕ := 7

/-- The initial number of students on the bus -/
def initial_students : ℕ := students_remaining + students_off

theorem initial_students_count : initial_students = 10 := by sorry

end NUMINAMATH_CALUDE_initial_students_count_l844_84438


namespace NUMINAMATH_CALUDE_average_of_first_50_even_numbers_l844_84470

def first_even_number : ℕ := 2

def last_even_number (n : ℕ) : ℕ := first_even_number + 2 * (n - 1)

def average_of_arithmetic_sequence (a₁ a_n n : ℕ) : ℚ :=
  (a₁ + a_n : ℚ) / 2

theorem average_of_first_50_even_numbers :
  average_of_arithmetic_sequence first_even_number (last_even_number 50) 50 = 51 := by
  sorry

end NUMINAMATH_CALUDE_average_of_first_50_even_numbers_l844_84470


namespace NUMINAMATH_CALUDE_average_weight_problem_l844_84489

/-- Given the average weights of pairs and the weight of one individual, 
    prove the average weight of all three. -/
theorem average_weight_problem (a b c : ℝ) 
    (h1 : (a + b) / 2 = 25) 
    (h2 : (b + c) / 2 = 28) 
    (h3 : b = 16) : 
    (a + b + c) / 3 = 30 := by
  sorry

end NUMINAMATH_CALUDE_average_weight_problem_l844_84489


namespace NUMINAMATH_CALUDE_age_difference_is_27_l844_84424

/-- Represents a person's age as a two-digit number -/
structure Age :=
  (tens : Nat)
  (ones : Nat)
  (h : tens < 10 ∧ ones < 10)

def Age.value (a : Age) : Nat := 10 * a.tens + a.ones

def Age.reverse (a : Age) : Age :=
  ⟨a.ones, a.tens, a.h.symm⟩

theorem age_difference_is_27 (alan_age bob_age : Age) : 
  (alan_age.reverse = bob_age) →
  (bob_age.value = alan_age.value / 2 + 6) →
  (alan_age.value + 2 = 5 * (bob_age.value - 4)) →
  (alan_age.value - bob_age.value = 27) :=
sorry

end NUMINAMATH_CALUDE_age_difference_is_27_l844_84424


namespace NUMINAMATH_CALUDE_sara_apples_l844_84460

theorem sara_apples (total : ℕ) (ali_ratio : ℕ) (sara_apples : ℕ) : 
  total = 80 →
  ali_ratio = 4 →
  total = sara_apples * (ali_ratio + 1) →
  sara_apples = 16 := by
sorry

end NUMINAMATH_CALUDE_sara_apples_l844_84460


namespace NUMINAMATH_CALUDE_lagaan_collection_l844_84465

/-- The total amount of lagaan collected from a village, given the payment of one farmer and their land proportion. -/
theorem lagaan_collection (farmer_payment : ℝ) (farmer_land_proportion : ℝ) 
  (h1 : farmer_payment = 480) 
  (h2 : farmer_land_proportion = 0.23255813953488372 / 100) : 
  (farmer_payment / farmer_land_proportion) = 206400000 := by
  sorry

end NUMINAMATH_CALUDE_lagaan_collection_l844_84465


namespace NUMINAMATH_CALUDE_rectangle_area_l844_84474

/-- The area of a rectangle with length thrice its breadth and perimeter 104 meters is 507 square meters. -/
theorem rectangle_area (breadth length perimeter area : ℝ) : 
  length = 3 * breadth →
  perimeter = 2 * (length + breadth) →
  perimeter = 104 →
  area = length * breadth →
  area = 507 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l844_84474


namespace NUMINAMATH_CALUDE_eighth_box_books_l844_84439

theorem eighth_box_books (total_books : ℕ) (num_boxes : ℕ) (books_per_box : ℕ) 
  (h1 : total_books = 800)
  (h2 : num_boxes = 8)
  (h3 : books_per_box = 105) :
  total_books - (num_boxes - 1) * books_per_box = 65 := by
  sorry

end NUMINAMATH_CALUDE_eighth_box_books_l844_84439


namespace NUMINAMATH_CALUDE_dima_walking_speed_l844_84453

/-- Represents the time in hours and minutes -/
structure Time where
  hour : Nat
  minute : Nat

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t2.hour * 60 + t2.minute) - (t1.hour * 60 + t1.minute)

/-- Represents the problem setup -/
structure ProblemSetup where
  scheduledArrival : Time
  actualArrival : Time
  carSpeed : Nat
  earlyArrivalTime : Nat

/-- Calculates Dima's walking speed -/
def calculateWalkingSpeed (setup : ProblemSetup) : Rat :=
  sorry

theorem dima_walking_speed (setup : ProblemSetup) 
  (h1 : setup.scheduledArrival = ⟨18, 0⟩)
  (h2 : setup.actualArrival = ⟨17, 5⟩)
  (h3 : setup.carSpeed = 60)
  (h4 : setup.earlyArrivalTime = 10) :
  calculateWalkingSpeed setup = 6 := by
  sorry

end NUMINAMATH_CALUDE_dima_walking_speed_l844_84453


namespace NUMINAMATH_CALUDE_cafe_tables_l844_84481

theorem cafe_tables (outdoor_tables : ℕ) (indoor_chairs : ℕ) (outdoor_chairs : ℕ) (total_chairs : ℕ) :
  outdoor_tables = 11 →
  indoor_chairs = 10 →
  outdoor_chairs = 3 →
  total_chairs = 123 →
  ∃ indoor_tables : ℕ, indoor_tables * indoor_chairs + outdoor_tables * outdoor_chairs = total_chairs ∧ indoor_tables = 9 :=
by sorry

end NUMINAMATH_CALUDE_cafe_tables_l844_84481


namespace NUMINAMATH_CALUDE_tangent_line_x_squared_at_one_l844_84434

/-- The equation of the tangent line to y = x^2 at x = 1 is y = 2x - 1 -/
theorem tangent_line_x_squared_at_one :
  let f (x : ℝ) := x^2
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := (deriv f) x₀
  ∀ x y : ℝ, y - y₀ = m * (x - x₀) ↔ y = 2*x - 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_x_squared_at_one_l844_84434


namespace NUMINAMATH_CALUDE_rectangle_to_triangle_altitude_l844_84409

/-- A rectangle with width 7 and length 21 can be rearranged into a triangle with altitude 14 -/
theorem rectangle_to_triangle_altitude (w h b : ℝ) : 
  w = 7 → h = 21 → b = 21 → 
  ∃ (altitude : ℝ), 
    w * h = (1/2) * b * altitude ∧ 
    altitude = 14 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_to_triangle_altitude_l844_84409


namespace NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_15_l844_84459

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 5
  sum_2_5 : a 2 + a 5 = 12
  nth_term : ∃ n, a n = 29

/-- The theorem stating that n = 15 for the given arithmetic sequence -/
theorem arithmetic_sequence_n_equals_15 (seq : ArithmeticSequence) : 
  ∃ n, seq.a n = 29 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_n_equals_15_l844_84459


namespace NUMINAMATH_CALUDE_total_cost_before_markup_is_47_l844_84436

/-- The markup percentage as a decimal -/
def markup : ℚ := 0.10

/-- The selling prices of the three books -/
def sellingPrices : List ℚ := [11.00, 16.50, 24.20]

/-- Calculate the original price before markup -/
def originalPrice (sellingPrice : ℚ) : ℚ := sellingPrice / (1 + markup)

/-- Calculate the total cost before markup -/
def totalCostBeforeMarkup : ℚ := (sellingPrices.map originalPrice).sum

/-- Theorem stating that the total cost before markup is $47.00 -/
theorem total_cost_before_markup_is_47 : totalCostBeforeMarkup = 47 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_before_markup_is_47_l844_84436


namespace NUMINAMATH_CALUDE_t_range_l844_84456

/-- The quadratic function -/
def f (x : ℝ) : ℝ := -x^2 + 6*x - 7

/-- The maximum value function -/
def y_max (t : ℝ) : ℝ := -(t-3)^2 + 2

/-- Theorem stating the range of t -/
theorem t_range (t : ℝ) :
  (∀ x, t ≤ x ∧ x ≤ t+2 → f x ≤ y_max t) →
  (∃ x, t ≤ x ∧ x ≤ t+2 ∧ f x = y_max t) →
  t ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_t_range_l844_84456


namespace NUMINAMATH_CALUDE_max_value_expression_l844_84405

theorem max_value_expression (x y : ℝ) :
  (2 * x + 3 * y + 4) / Real.sqrt (x^4 + y^2 + 1) ≤ Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l844_84405


namespace NUMINAMATH_CALUDE_kevins_toads_l844_84448

/-- The number of toads in Kevin's shoebox -/
def num_toads : ℕ := 8

/-- The number of worms each toad is fed daily -/
def worms_per_toad : ℕ := 3

/-- The time (in minutes) it takes Kevin to find each worm -/
def minutes_per_worm : ℕ := 15

/-- The time (in hours) it takes Kevin to find enough worms for all toads -/
def total_hours : ℕ := 6

/-- Theorem stating that the number of toads is 8 given the conditions -/
theorem kevins_toads : 
  num_toads = (total_hours * 60) / minutes_per_worm / worms_per_toad :=
by sorry

end NUMINAMATH_CALUDE_kevins_toads_l844_84448


namespace NUMINAMATH_CALUDE_smallest_divisor_and_quadratic_form_l844_84477

theorem smallest_divisor_and_quadratic_form : ∃ k : ℕ,
  (∃ n : ℕ, (2^n + 15) % k = 0) ∧
  (∃ x y : ℤ, k = 3*x^2 - 4*x*y + 3*y^2) ∧
  (∀ m : ℕ, m < k →
    (∃ n : ℕ, (2^n + 15) % m = 0) ∧
    (∃ x y : ℤ, m = 3*x^2 - 4*x*y + 3*y^2) →
    False) ∧
  k = 23 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisor_and_quadratic_form_l844_84477


namespace NUMINAMATH_CALUDE_product_PQRS_l844_84499

theorem product_PQRS : 
  let P := 2 * (Real.sqrt 2010 + Real.sqrt 2011)
  let Q := 3 * (-Real.sqrt 2010 - Real.sqrt 2011)
  let R := 2 * (Real.sqrt 2010 - Real.sqrt 2011)
  let S := 3 * (Real.sqrt 2011 - Real.sqrt 2010)
  P * Q * R * S = -36 := by
  sorry

end NUMINAMATH_CALUDE_product_PQRS_l844_84499


namespace NUMINAMATH_CALUDE_lost_shoes_count_l844_84421

/-- Given a number of initial shoe pairs and remaining matching pairs,
    calculates the number of individual shoes lost. -/
def shoes_lost (initial_pairs : ℕ) (remaining_pairs : ℕ) : ℕ :=
  2 * initial_pairs - 2 * remaining_pairs

/-- Theorem stating that with 24 initial pairs and 19 remaining pairs,
    10 individual shoes were lost. -/
theorem lost_shoes_count :
  shoes_lost 24 19 = 10 := by
  sorry


end NUMINAMATH_CALUDE_lost_shoes_count_l844_84421


namespace NUMINAMATH_CALUDE_a_8_value_l844_84410

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem a_8_value (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 6 + a 10 = -6) →
  (a 6 * a 10 = 2) →
  (a 6 < 0) →
  (a 10 < 0) →
  a 8 = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_a_8_value_l844_84410


namespace NUMINAMATH_CALUDE_alternating_color_probability_l844_84404

/-- The probability of drawing 8 balls from a box containing 5 white and 3 black balls,
    such that the draws alternate in color starting with a white ball. -/
theorem alternating_color_probability :
  let total_balls : ℕ := 8
  let white_balls : ℕ := 5
  let black_balls : ℕ := 3
  let total_arrangements : ℕ := Nat.choose total_balls black_balls
  let favorable_arrangements : ℕ := 1
  (favorable_arrangements : ℚ) / total_arrangements = 1 / 56 := by
  sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l844_84404


namespace NUMINAMATH_CALUDE_sum_denominator_divisible_by_prime_l844_84433

theorem sum_denominator_divisible_by_prime (p : ℕ) (n : ℕ) (b : Fin n → ℕ) :
  Prime p →
  (∃! i : Fin n, p ∣ b i) →
  (∀ i : Fin n, 0 < b i) →
  ∃ (num den : ℕ), 
    (0 < den) ∧
    (Nat.gcd num den = 1) ∧
    (p ∣ den) ∧
    (Finset.sum Finset.univ (λ i => 1 / (b i : ℚ)) = num / den) :=
by sorry

end NUMINAMATH_CALUDE_sum_denominator_divisible_by_prime_l844_84433


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l844_84423

/-- Represents the quantities of sugar types A, B, and C -/
structure SugarQuantities where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the given quantities satisfy the problem constraints -/
def satisfiesConstraints (q : SugarQuantities) : Prop :=
  q.a ≥ 0 ∧ q.b ≥ 0 ∧ q.c ≥ 0 ∧
  q.a + q.b + q.c = 1500 ∧
  (8 * q.a + 15 * q.b + 20 * q.c) / 1500 = 14

/-- There are infinitely many solutions to the sugar problem -/
theorem infinitely_many_solutions :
  ∀ ε > 0, ∃ q₁ q₂ : SugarQuantities,
    satisfiesConstraints q₁ ∧
    satisfiesConstraints q₂ ∧
    q₁ ≠ q₂ ∧
    ‖q₁.a - q₂.a‖ < ε ∧
    ‖q₁.b - q₂.b‖ < ε ∧
    ‖q₁.c - q₂.c‖ < ε :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l844_84423


namespace NUMINAMATH_CALUDE_missing_number_odd_l844_84428

def set_a : Finset Nat := {11, 44, 55}

def is_odd (n : Nat) : Prop := n % 2 = 1

def probability_even_sum (b : Nat) : Rat :=
  (set_a.filter (fun a => (a + b) % 2 = 0)).card / set_a.card

theorem missing_number_odd (b : Nat) :
  probability_even_sum b = 1/2 → is_odd b := by
  sorry

end NUMINAMATH_CALUDE_missing_number_odd_l844_84428


namespace NUMINAMATH_CALUDE_point_on_transformed_graph_l844_84482

/-- Given a function g where g(3) = 8, there exists a point (x, y) on the graph of 
    y = 4g(3x-1) + 6 such that x + y = 40 -/
theorem point_on_transformed_graph (g : ℝ → ℝ) (h : g 3 = 8) :
  ∃ x y : ℝ, 4 * g (3 * x - 1) + 6 = y ∧ x + y = 40 := by
  sorry

end NUMINAMATH_CALUDE_point_on_transformed_graph_l844_84482


namespace NUMINAMATH_CALUDE_hyperbola_equation_l844_84444

/-- Given a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1, where a > 0, b > 0,
    if one focus is at (2,0) and one asymptote has a slope of √3,
    then the equation of the hyperbola is x^2 - (y^2 / 3) = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (2 : ℝ)^2 = a^2 + b^2 →
  b / a = Real.sqrt 3 →
  ∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 - y^2 / 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l844_84444


namespace NUMINAMATH_CALUDE_some_number_solution_l844_84426

theorem some_number_solution : 
  ∃ x : ℝ, 45 - (28 - (x - (15 - 15))) = 54 ∧ x = 37 := by sorry

end NUMINAMATH_CALUDE_some_number_solution_l844_84426


namespace NUMINAMATH_CALUDE_tan_two_implies_specific_trig_ratio_l844_84400

theorem tan_two_implies_specific_trig_ratio (θ : Real) (h : Real.tan θ = 2) :
  (Real.sin θ * Real.sin (π/2 - θ)) / (Real.sin θ^2 + Real.cos (2*θ) + Real.cos θ^2) = 1/3 :=
by sorry

end NUMINAMATH_CALUDE_tan_two_implies_specific_trig_ratio_l844_84400


namespace NUMINAMATH_CALUDE_julie_weed_hours_l844_84418

/-- Represents Julie's landscaping business earnings --/
def julie_earnings (weed_hours : ℕ) : ℕ :=
  let mowing_rate : ℕ := 4
  let weed_rate : ℕ := 8
  let mowing_hours : ℕ := 25
  2 * (mowing_rate * mowing_hours + weed_rate * weed_hours)

/-- Proves that Julie spent 3 hours pulling weeds in September --/
theorem julie_weed_hours : 
  ∃ (weed_hours : ℕ), julie_earnings weed_hours = 248 ∧ weed_hours = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_julie_weed_hours_l844_84418


namespace NUMINAMATH_CALUDE_fermats_little_theorem_l844_84451

theorem fermats_little_theorem (p : ℕ) (a : ℕ) (hp : Prime p) : a^p ≡ a [MOD p] := by
  sorry

end NUMINAMATH_CALUDE_fermats_little_theorem_l844_84451


namespace NUMINAMATH_CALUDE_sin_cos_values_l844_84483

theorem sin_cos_values (α : Real) (h : Real.sin α + 3 * Real.cos α = 0) :
  (Real.sin α = 3 * (Real.sqrt 10) / 10 ∧ Real.cos α = -(Real.sqrt 10) / 10) ∨
  (Real.sin α = -(3 * (Real.sqrt 10) / 10) ∧ Real.cos α = (Real.sqrt 10) / 10) := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_values_l844_84483


namespace NUMINAMATH_CALUDE_jerrys_collection_cost_l844_84480

/-- The amount of money Jerry needs to finish his action figure collection -/
def jerrysMoney (currentFigures : ℕ) (totalRequired : ℕ) (costPerFigure : ℕ) : ℕ :=
  (totalRequired - currentFigures) * costPerFigure

/-- Proof that Jerry needs $72 to finish his collection -/
theorem jerrys_collection_cost : jerrysMoney 7 16 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_jerrys_collection_cost_l844_84480


namespace NUMINAMATH_CALUDE_fifth_sixth_sum_l844_84437

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n, a (n + 1) = a n * (a 2 / a 1)
  sum_1_2 : a 1 + a 2 = 20
  sum_3_4 : a 3 + a 4 = 40

/-- The theorem stating that a₅ + a₆ = 80 for the given geometric sequence -/
theorem fifth_sixth_sum (seq : GeometricSequence) : seq.a 5 + seq.a 6 = 80 := by
  sorry

end NUMINAMATH_CALUDE_fifth_sixth_sum_l844_84437


namespace NUMINAMATH_CALUDE_constant_function_proof_l844_84488

theorem constant_function_proof (f g h : ℕ → ℕ)
  (h_injective : Function.Injective h)
  (g_surjective : Function.Surjective g)
  (f_def : ∀ n, f n = g n - h n + 1) :
  ∀ n, f n = 1 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_proof_l844_84488


namespace NUMINAMATH_CALUDE_cricketer_average_score_l844_84450

theorem cricketer_average_score 
  (total_matches : ℕ) 
  (average_all : ℚ) 
  (average_last : ℚ) 
  (last_matches : ℕ) : 
  total_matches = 10 → 
  average_all = 389/10 → 
  average_last = 137/4 → 
  last_matches = 4 → 
  (total_matches * average_all - last_matches * average_last) / (total_matches - last_matches) = 42 := by
  sorry

end NUMINAMATH_CALUDE_cricketer_average_score_l844_84450


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l844_84490

theorem tan_value_from_trig_equation (θ : Real) 
  (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 2 * Real.sin θ * Real.sin (θ + π / 4) = 5 * Real.cos (2 * θ)) : 
  Real.tan θ = 5 / 6 := by
sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l844_84490


namespace NUMINAMATH_CALUDE_odd_power_eight_minus_one_mod_nine_l844_84485

theorem odd_power_eight_minus_one_mod_nine (n : ℕ) (h : Odd n) : (8^n - 1) % 9 = 7 := by
  sorry

end NUMINAMATH_CALUDE_odd_power_eight_minus_one_mod_nine_l844_84485


namespace NUMINAMATH_CALUDE_marking_implies_prime_f_1997_l844_84464

/-- Represents the marking procedure on a 2N-gon -/
def mark_procedure (N : ℕ) : Set ℕ := sorry

/-- The function f(N) that counts non-marked vertices -/
def f (N : ℕ) : ℕ := sorry

/-- Main theorem: If f(N) = 0, then 2N + 1 is prime -/
theorem marking_implies_prime (N : ℕ) (h1 : N > 2) (h2 : f N = 0) : Nat.Prime (2 * N + 1) := by
  sorry

/-- Computation of f(1997) -/
theorem f_1997 : f 1997 = 3810 := by
  sorry

end NUMINAMATH_CALUDE_marking_implies_prime_f_1997_l844_84464


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_four_l844_84469

theorem largest_four_digit_divisible_by_four :
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 4 = 0 → n ≤ 9996 :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_four_l844_84469


namespace NUMINAMATH_CALUDE_six_by_six_grid_squares_l844_84413

/-- The number of squares of size n×n in a grid of size m×m -/
def count_squares (n m : ℕ) : ℕ := (m - n + 1) * (m - n + 1)

/-- The total number of squares in a 6×6 grid -/
def total_squares : ℕ :=
  count_squares 1 6 + count_squares 2 6 + count_squares 3 6 + count_squares 4 6

theorem six_by_six_grid_squares :
  total_squares = 86 :=
sorry

end NUMINAMATH_CALUDE_six_by_six_grid_squares_l844_84413


namespace NUMINAMATH_CALUDE_shoe_cost_comparison_l844_84402

/-- Calculates the percentage increase in average cost per year of new shoes compared to repaired used shoes -/
theorem shoe_cost_comparison (used_repair_cost : ℝ) (used_lifespan : ℝ) (new_cost : ℝ) (new_lifespan : ℝ)
  (h1 : used_repair_cost = 11.50)
  (h2 : used_lifespan = 1)
  (h3 : new_cost = 28.00)
  (h4 : new_lifespan = 2)
  : (((new_cost / new_lifespan) - (used_repair_cost / used_lifespan)) / (used_repair_cost / used_lifespan)) * 100 = 21.74 := by
  sorry

end NUMINAMATH_CALUDE_shoe_cost_comparison_l844_84402


namespace NUMINAMATH_CALUDE_integral_tan_over_trig_expression_l844_84497

theorem integral_tan_over_trig_expression :
  let f := fun x : ℝ => (Real.tan x) / (Real.sin x ^ 2 - 5 * Real.cos x ^ 2 + 4)
  let a := Real.pi / 4
  let b := Real.arccos (1 / Real.sqrt 3)
  ∫ x in a..b, f x = (1 / 10) * Real.log (9 / 4) :=
by sorry

end NUMINAMATH_CALUDE_integral_tan_over_trig_expression_l844_84497


namespace NUMINAMATH_CALUDE_distinct_book_selections_l844_84454

theorem distinct_book_selections (n k : ℕ) (h1 : n = 15) (h2 : k = 3) :
  Nat.choose n k = 455 := by
  sorry

end NUMINAMATH_CALUDE_distinct_book_selections_l844_84454


namespace NUMINAMATH_CALUDE_volunteer_average_age_l844_84419

theorem volunteer_average_age (total_members : ℕ) (teens : ℕ) (parents : ℕ) (volunteers : ℕ)
  (teen_avg_age : ℝ) (parent_avg_age : ℝ) (overall_avg_age : ℝ) :
  total_members = 50 →
  teens = 30 →
  parents = 15 →
  volunteers = 5 →
  teen_avg_age = 16 →
  parent_avg_age = 35 →
  overall_avg_age = 23 →
  (total_members : ℝ) * overall_avg_age = 
    (teens : ℝ) * teen_avg_age + (parents : ℝ) * parent_avg_age + (volunteers : ℝ) * ((total_members : ℝ) * overall_avg_age - (teens : ℝ) * teen_avg_age - (parents : ℝ) * parent_avg_age) / (volunteers : ℝ) →
  ((total_members : ℝ) * overall_avg_age - (teens : ℝ) * teen_avg_age - (parents : ℝ) * parent_avg_age) / (volunteers : ℝ) = 29 :=
by
  sorry

#check volunteer_average_age

end NUMINAMATH_CALUDE_volunteer_average_age_l844_84419


namespace NUMINAMATH_CALUDE_min_value_theorem_l844_84475

theorem min_value_theorem (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) :
  2/x + 9/(1-2*x) ≥ 25 ∧ ∃ y ∈ Set.Ioo 0 (1/2), 2/y + 9/(1-2*y) = 25 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l844_84475


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l844_84446

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := a^2 - 4 + (a - 2) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → a = -2 := by
sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l844_84446


namespace NUMINAMATH_CALUDE_cookie_distribution_l844_84408

theorem cookie_distribution (x y z : ℚ) :
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x * (y / z) = 35 →
  (35 : ℚ) / 5 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l844_84408


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l844_84443

theorem sqrt_product_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 21 * q * Real.sqrt (2 * q) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l844_84443


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l844_84416

-- Problem 1
theorem problem_1 (a b : ℝ) : (a + 2*b)^2 - 4*b*(a + b) = a^2 := by sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  ((x^2 - 2*x) / (x^2 - 4*x + 4) + 1 / (2 - x)) / ((x - 1) / (x^2 - 4)) = x + 2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l844_84416


namespace NUMINAMATH_CALUDE_janous_inequality_l844_84472

theorem janous_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d ∧
  ∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ (a₀ + 2) * (b₀ + 2) = c₀ * d₀ := by
  sorry

#check janous_inequality

end NUMINAMATH_CALUDE_janous_inequality_l844_84472


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l844_84457

/-- A function that returns the sum of digits of a positive integer -/
def sum_of_digits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nth_number_with_digit_sum_13 (n : ℕ+) : ℕ+ := sorry

/-- The theorem stating that the 11th number with digit sum 13 is 175 -/
theorem eleventh_number_with_digit_sum_13 : 
  nth_number_with_digit_sum_13 11 = 175 := by sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l844_84457


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sin_value_l844_84455

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sin_value (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 2 + a 6 = 3 * Real.pi / 2 →
  Real.sin (2 * a 4 - Real.pi / 3) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sin_value_l844_84455


namespace NUMINAMATH_CALUDE_sequence_progression_l844_84494

-- Define the sequence type
def Sequence := ℕ → ℝ

-- Define arithmetic progression
def is_arithmetic_progression (a : Sequence) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define geometric progression
def is_geometric_progression (a : Sequence) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem sequence_progression (a : Sequence) 
  (h1 : a 4 + a 7 = 2) 
  (h2 : a 5 * a 6 = -8) :
  (is_arithmetic_progression a → a 1 * a 10 = -728) ∧
  (is_geometric_progression a → a 1 + a 10 = -7) := by
  sorry

end NUMINAMATH_CALUDE_sequence_progression_l844_84494


namespace NUMINAMATH_CALUDE_power_division_equality_l844_84445

theorem power_division_equality : 8^15 / 64^3 = 8^9 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l844_84445


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_sum_l844_84415

def Rectangle := (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ)

def vertices : Rectangle := ((1, 1), (1, 5), (6, 5), (6, 1))

def length (r : Rectangle) : ℝ := 
  let ((x1, _), (_, _), (x2, _), _) := r
  |x2 - x1|

def width (r : Rectangle) : ℝ := 
  let ((_, y1), (_, y2), _, _) := r
  |y2 - y1|

def perimeter (r : Rectangle) : ℝ := 2 * (length r + width r)

def area (r : Rectangle) : ℝ := length r * width r

theorem rectangle_perimeter_area_sum :
  perimeter vertices + area vertices = 38 := by sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_area_sum_l844_84415


namespace NUMINAMATH_CALUDE_divisibility_by_33_l844_84447

def five_digit_number (n : ℕ) : ℕ := 70000 + 1000 * n + 933

theorem divisibility_by_33 (n : ℕ) : 
  n < 10 → (five_digit_number n % 33 = 0 ↔ n = 5) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_33_l844_84447


namespace NUMINAMATH_CALUDE_sean_blocks_l844_84422

theorem sean_blocks (initial_blocks : ℕ) (eaten_blocks : ℕ) (remaining_blocks : ℕ) : 
  initial_blocks = 55 → eaten_blocks = 29 → remaining_blocks = initial_blocks - eaten_blocks → 
  remaining_blocks = 26 := by
sorry

end NUMINAMATH_CALUDE_sean_blocks_l844_84422


namespace NUMINAMATH_CALUDE_parabola_equation_l844_84495

/-- A parabola with vertex at the origin and focus at (2, 0) has the equation y^2 = 8x -/
theorem parabola_equation (p : ℝ × ℝ → Prop) : 
  (∀ x y, p (x, y) ↔ y^2 = 8*x) ↔ 
  (p (0, 0)  -- vertex at origin
   ∧ (∀ x y, p (x, y) → (x - 2)^2 + y^2 = 4)  -- focus at (2, 0)
   ∧ (∀ x y, p (x, y) → y^2 = 4 * 2 * x)) :=  -- standard form of parabola with p = 2
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l844_84495


namespace NUMINAMATH_CALUDE_first_reduction_percentage_l844_84461

theorem first_reduction_percentage (x : ℝ) : 
  (1 - x / 100) * (1 - 0.1) = 0.765 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_first_reduction_percentage_l844_84461


namespace NUMINAMATH_CALUDE_tangent_lines_at_k_zero_equal_angles_point_l844_84493

-- Define the curve C and the line
def C (x y : ℝ) : Prop := x^2 = 4*y
def L (k a x y : ℝ) : Prop := y = k*x + a

-- Define the intersection points M and N
def intersection_points (k a : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ (x y : ℝ), p = (x, y) ∧ C x y ∧ L k a x y}

-- Theorem for tangent lines when k = 0
theorem tangent_lines_at_k_zero (a : ℝ) (ha : a > 0) :
  ∃ (M N : ℝ × ℝ), M ∈ intersection_points 0 a ∧ N ∈ intersection_points 0 a ∧
  (∃ (x y : ℝ), M = (x, y) ∧ Real.sqrt a * x - y - a = 0) ∧
  (∃ (x y : ℝ), N = (x, y) ∧ Real.sqrt a * x + y + a = 0) :=
sorry

-- Theorem for the existence of point P
theorem equal_angles_point (a : ℝ) (ha : a > 0) :
  ∃ (P : ℝ × ℝ), P.1 = 0 ∧
  ∀ (k : ℝ) (M N : ℝ × ℝ), M ∈ intersection_points k a → N ∈ intersection_points k a →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), M = (x₁, y₁) ∧ N = (x₂, y₂) ∧
   (y₁ - P.2) / x₁ = -(y₂ - P.2) / x₂) :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_at_k_zero_equal_angles_point_l844_84493


namespace NUMINAMATH_CALUDE_luke_good_games_l844_84417

def budget : ℕ := 100
def price_a : ℕ := 15
def price_b : ℕ := 8
def price_c : ℕ := 6
def num_a : ℕ := 3
def num_b : ℕ := 5
def sold_games : ℕ := 2
def sold_price : ℕ := 12
def broken_a : ℕ := 3
def broken_b : ℕ := 2

def remaining_budget : ℕ := budget - (num_a * price_a + num_b * price_b) + (sold_games * sold_price)

def num_c : ℕ := remaining_budget / price_c

theorem luke_good_games : 
  (num_a - broken_a) + (num_b - broken_b) + num_c = 9 :=
sorry

end NUMINAMATH_CALUDE_luke_good_games_l844_84417


namespace NUMINAMATH_CALUDE_music_festival_children_avg_age_l844_84496

/-- Represents the demographics and age statistics of a music festival. -/
structure MusicFestival where
  total_participants : ℕ
  num_women : ℕ
  num_men : ℕ
  num_children : ℕ
  overall_avg_age : ℚ
  women_avg_age : ℚ
  men_avg_age : ℚ

/-- Calculates the average age of children in the music festival. -/
def children_avg_age (festival : MusicFestival) : ℚ :=
  (festival.total_participants * festival.overall_avg_age
   - festival.num_women * festival.women_avg_age
   - festival.num_men * festival.men_avg_age) / festival.num_children

/-- Theorem stating that for the given music festival data, the average age of children is 13. -/
theorem music_festival_children_avg_age :
  let festival : MusicFestival := {
    total_participants := 50,
    num_women := 22,
    num_men := 18,
    num_children := 10,
    overall_avg_age := 20,
    women_avg_age := 24,
    men_avg_age := 19
  }
  children_avg_age festival = 13 := by sorry

end NUMINAMATH_CALUDE_music_festival_children_avg_age_l844_84496


namespace NUMINAMATH_CALUDE_bus_ticket_impossibility_prove_bus_ticket_impossibility_l844_84427

theorem bus_ticket_impossibility 
  (num_passengers : ℕ) 
  (ticket_price : ℕ) 
  (coin_denominations : List ℕ) 
  (total_coins : ℕ) : Prop :=
  num_passengers = 40 →
  ticket_price = 5 →
  coin_denominations = [10, 15, 20] →
  total_coins = 49 →
  ¬∃ (payment : List ℕ),
    payment.sum = num_passengers * ticket_price ∧
    payment.length ≤ total_coins - num_passengers ∧
    ∀ c ∈ payment, c ∈ coin_denominations

theorem prove_bus_ticket_impossibility : 
  bus_ticket_impossibility 40 5 [10, 15, 20] 49 := by
  sorry

end NUMINAMATH_CALUDE_bus_ticket_impossibility_prove_bus_ticket_impossibility_l844_84427


namespace NUMINAMATH_CALUDE_alberts_current_funds_l844_84411

/-- The problem of calculating Albert's current funds --/
theorem alberts_current_funds
  (total_cost : ℝ)
  (additional_needed : ℝ)
  (h1 : total_cost = 18.50)
  (h2 : additional_needed = 12) :
  total_cost - additional_needed = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_alberts_current_funds_l844_84411


namespace NUMINAMATH_CALUDE_largest_common_value_of_aps_l844_84407

/-- The largest common value less than 300 between two arithmetic progressions -/
theorem largest_common_value_of_aps : ∃ (n m : ℕ),
  7 * (n + 1) = 5 + 10 * m ∧
  7 * (n + 1) < 300 ∧
  ∀ (k l : ℕ), 7 * (k + 1) = 5 + 10 * l → 7 * (k + 1) < 300 → 7 * (k + 1) ≤ 7 * (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_value_of_aps_l844_84407


namespace NUMINAMATH_CALUDE_number_wall_solution_l844_84466

/-- Represents a number wall with the given base numbers -/
structure NumberWall (m : ℤ) :=
  (base : Fin 4 → ℤ)
  (base_values : base 0 = m ∧ base 1 = 6 ∧ base 2 = -3 ∧ base 3 = 4)

/-- Calculates the value at the top of the number wall -/
def top_value (w : NumberWall m) : ℤ :=
  let level1_0 := w.base 0 + w.base 1
  let level1_1 := w.base 1 + w.base 2
  let level1_2 := w.base 2 + w.base 3
  let level2_0 := level1_0 + level1_1
  let level2_1 := level1_1 + level1_2
  level2_0 + level2_1

/-- The theorem to be proved -/
theorem number_wall_solution (m : ℤ) (w : NumberWall m) :
  top_value w = 20 → m = 7 := by sorry

end NUMINAMATH_CALUDE_number_wall_solution_l844_84466


namespace NUMINAMATH_CALUDE_a_minus_b_equals_negative_seven_l844_84452

theorem a_minus_b_equals_negative_seven
  (a b : ℝ)
  (h1 : |a| = 3)
  (h2 : Real.sqrt b = 2)
  (h3 : a * b < 0) :
  a - b = -7 := by
sorry

end NUMINAMATH_CALUDE_a_minus_b_equals_negative_seven_l844_84452


namespace NUMINAMATH_CALUDE_probability_different_rooms_l844_84406

theorem probability_different_rooms (n : ℕ) (h : n = 2) : 
  (n - 1 : ℚ) / n = 1 / 2 := by
  sorry

#check probability_different_rooms

end NUMINAMATH_CALUDE_probability_different_rooms_l844_84406


namespace NUMINAMATH_CALUDE_max_square_plots_l844_84498

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  length : ℕ
  width : ℕ

/-- Represents the available fence length for internal fencing -/
def available_fence : ℕ := 2200

/-- Calculates the number of square plots given the number of plots in a column -/
def num_plots (n : ℕ) : ℕ := n * (11 * n / 6)

/-- Calculates the required fence length for a given number of plots in a column -/
def required_fence (n : ℕ) : ℕ := 187 * n - 132

/-- The maximum number of square plots that can partition the field -/
def max_plots : ℕ := 264

/-- Theorem stating the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
  (h1 : field.length = 36) 
  (h2 : field.width = 66) : 
  (∀ n : ℕ, num_plots n ≤ max_plots ∧ required_fence n ≤ available_fence) ∧ 
  (∃ n : ℕ, num_plots n = max_plots ∧ required_fence n ≤ available_fence) :=
sorry

end NUMINAMATH_CALUDE_max_square_plots_l844_84498


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l844_84430

theorem min_value_reciprocal_sum (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  (1/a + 1/b + 1/c ≥ 9) ∧ 
  (1/a + 1/b + 1/c = 9 ↔ a = 1/3 ∧ b = 1/3 ∧ c = 1/3) :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l844_84430


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l844_84479

theorem min_value_of_sum_of_ratios (a d b c : ℝ) 
  (ha : a ≥ 0) (hd : d ≥ 0) (hb : b > 0) (hc : c > 0) (h_sum : b + c ≥ a + d) :
  ∀ x y, x > 0 ∧ y > 0 → (b / (c + d)) + (c / (a + b)) ≥ Real.sqrt 2 - 1/2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_ratios_l844_84479


namespace NUMINAMATH_CALUDE_calculate_expression_l844_84425

theorem calculate_expression : -1^4 + 16 / (-2)^3 * |(-3) - 1| = -9 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l844_84425


namespace NUMINAMATH_CALUDE_mother_three_times_daughter_age_l844_84484

/-- Proves that the number of years until the mother is three times as old as her daughter is 9,
    given that the mother is currently 42 years old and the daughter is currently 8 years old. -/
theorem mother_three_times_daughter_age (mother_age : ℕ) (daughter_age : ℕ) 
  (h1 : mother_age = 42) (h2 : daughter_age = 8) : 
  ∃ (years : ℕ), mother_age + years = 3 * (daughter_age + years) ∧ years = 9 := by
  sorry

end NUMINAMATH_CALUDE_mother_three_times_daughter_age_l844_84484


namespace NUMINAMATH_CALUDE_parabola_intersection_length_l844_84468

/-- Given a parabola y = x^2 - mx - 3 that intersects the x-axis at points A and B,
    where m is an integer, the length of AB is 4. -/
theorem parabola_intersection_length (m : ℤ) (A B : ℝ) : 
  (∃ x : ℝ, x^2 - m*x - 3 = 0) → 
  (A^2 - m*A - 3 = 0) → 
  (B^2 - m*B - 3 = 0) → 
  |A - B| = 4 := by
sorry

end NUMINAMATH_CALUDE_parabola_intersection_length_l844_84468


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l844_84476

theorem exponential_equation_solution :
  ∃! x : ℝ, (32 : ℝ) ^ (x - 2) / (16 : ℝ) ^ (x - 1) = (512 : ℝ) ^ (x + 1) ∧ x = -15/8 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l844_84476


namespace NUMINAMATH_CALUDE_A_infinite_B_infinite_unique_representation_l844_84478

/-- Two infinite sets of non-negative integers -/
def A : Set ℕ := sorry

/-- Two infinite sets of non-negative integers -/
def B : Set ℕ := sorry

/-- A is infinite -/
theorem A_infinite : Set.Infinite A := by sorry

/-- B is infinite -/
theorem B_infinite : Set.Infinite B := by sorry

/-- Every non-negative integer can be uniquely represented as a sum of elements from A and B -/
theorem unique_representation :
  ∀ n : ℕ, ∃! (a b : ℕ), a ∈ A ∧ b ∈ B ∧ n = a + b := by sorry

end NUMINAMATH_CALUDE_A_infinite_B_infinite_unique_representation_l844_84478


namespace NUMINAMATH_CALUDE_m_range_l844_84442

theorem m_range (m : ℝ) : 
  (∀ x : ℝ, |x - m| < 1 ↔ 1/3 < x ∧ x < 1/2) → 
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l844_84442


namespace NUMINAMATH_CALUDE_shane_garret_age_ratio_l844_84492

theorem shane_garret_age_ratio : 
  let shane_current_age : ℕ := 44
  let garret_current_age : ℕ := 12
  let years_ago : ℕ := 20
  let shane_past_age : ℕ := shane_current_age - years_ago
  shane_past_age / garret_current_age = 2 :=
by sorry

end NUMINAMATH_CALUDE_shane_garret_age_ratio_l844_84492


namespace NUMINAMATH_CALUDE_prime_not_divides_difference_l844_84412

theorem prime_not_divides_difference (a b c d p : ℕ) : 
  0 < a → 0 < b → 0 < c → 0 < d → 
  p = a + b + c + d → 
  Nat.Prime p → 
  ¬(p ∣ a * b - c * d) := by
sorry

end NUMINAMATH_CALUDE_prime_not_divides_difference_l844_84412


namespace NUMINAMATH_CALUDE_modified_prism_surface_area_difference_l844_84463

/-- Calculates the surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Calculates the surface area added by removing a cube from the center of a face -/
def added_surface_area (cube_side : ℝ) : ℝ := 5 * cube_side^2

theorem modified_prism_surface_area_difference :
  let original_sa := surface_area 2 4 5
  let modified_sa := original_sa + added_surface_area 1
  modified_sa - original_sa = 5 := by sorry

end NUMINAMATH_CALUDE_modified_prism_surface_area_difference_l844_84463


namespace NUMINAMATH_CALUDE_price_and_distance_proportions_l844_84414

-- Define the relationships
def inverse_proportion (x y : ℝ) (k : ℝ) : Prop := x * y = k
def direct_proportion (x y : ℝ) (k : ℝ) : Prop := x / y = k

-- State the theorem
theorem price_and_distance_proportions :
  -- For any positive real numbers representing unit price, quantity, and total price
  ∀ (unit_price quantity total_price : ℝ) (hp : unit_price > 0) (hq : quantity > 0) (ht : total_price > 0),
  -- When the total price is fixed
  (unit_price * quantity = total_price) →
  -- The unit price and quantity are in inverse proportion
  inverse_proportion unit_price quantity total_price ∧
  -- For any positive real numbers representing map distance, actual distance, and scale
  ∀ (map_distance actual_distance scale : ℝ) (hm : map_distance > 0) (ha : actual_distance > 0) (hs : scale > 0),
  -- When the scale is fixed
  (map_distance / actual_distance = scale) →
  -- The map distance and actual distance are in direct proportion
  direct_proportion map_distance actual_distance scale :=
by sorry

end NUMINAMATH_CALUDE_price_and_distance_proportions_l844_84414


namespace NUMINAMATH_CALUDE_cosine_identity_l844_84487

theorem cosine_identity (a : Real) (h : 3 * Real.pi / 2 < a ∧ a < 2 * Real.pi) :
  Real.sqrt (1/2 + 1/2 * Real.sqrt (1/2 + 1/2 * Real.cos (2 * a))) = -Real.cos (a / 2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l844_84487


namespace NUMINAMATH_CALUDE_sum_of_compositions_l844_84491

def p (x : ℝ) : ℝ := x^2 - 3

def q (x : ℝ) : ℝ := x - 2

def x_values : List ℝ := [-4, -3, -2, -1, 0, 1, 2, 3, 4]

theorem sum_of_compositions : 
  (x_values.map (λ x => q (p x))).sum = 15 := by sorry

end NUMINAMATH_CALUDE_sum_of_compositions_l844_84491


namespace NUMINAMATH_CALUDE_max_B_bins_l844_84440

/-- The cost of an A brand garbage bin in yuan -/
def cost_A : ℕ := 120

/-- The cost of a B brand garbage bin in yuan -/
def cost_B : ℕ := 150

/-- The total number of garbage bins to be purchased -/
def total_bins : ℕ := 30

/-- The maximum budget in yuan -/
def max_budget : ℕ := 4000

/-- Theorem stating the maximum number of B brand bins that can be purchased -/
theorem max_B_bins : 
  ∀ m : ℕ, 
  m ≤ total_bins ∧ 
  cost_B * m + cost_A * (total_bins - m) ≤ max_budget →
  m ≤ 13 :=
by sorry

end NUMINAMATH_CALUDE_max_B_bins_l844_84440


namespace NUMINAMATH_CALUDE_vector_magnitude_l844_84486

def a : ℝ × ℝ := (2, 1)

theorem vector_magnitude (b : ℝ × ℝ) 
  (h1 : a.1 * b.1 + a.2 * b.2 = 10)
  (h2 : Real.sqrt ((a.1 + b.1)^2 + (a.2 + b.2)^2) = 5 * Real.sqrt 2) :
  Real.sqrt (b.1^2 + b.2^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l844_84486


namespace NUMINAMATH_CALUDE_sin_30_sin_75_minus_sin_60_cos_105_l844_84449

theorem sin_30_sin_75_minus_sin_60_cos_105 :
  Real.sin (30 * π / 180) * Real.sin (75 * π / 180) -
  Real.sin (60 * π / 180) * Real.cos (105 * π / 180) =
  Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_30_sin_75_minus_sin_60_cos_105_l844_84449
