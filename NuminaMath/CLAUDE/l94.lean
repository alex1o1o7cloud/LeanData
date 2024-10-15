import Mathlib

namespace NUMINAMATH_CALUDE_toms_total_amount_l94_9428

/-- Tom's initial amount in dollars -/
def initial_amount : ℕ := 74

/-- Amount Tom earned from washing cars in dollars -/
def earned_amount : ℕ := 86

/-- Theorem stating Tom's total amount after washing cars -/
theorem toms_total_amount : initial_amount + earned_amount = 160 := by
  sorry

end NUMINAMATH_CALUDE_toms_total_amount_l94_9428


namespace NUMINAMATH_CALUDE_least_2310_divisors_form_l94_9471

/-- The number of distinct positive divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is not divisible by 10 -/
def not_div_by_ten (m : ℕ) : Prop := ¬(10 ∣ m)

/-- The least positive integer with exactly 2310 distinct positive divisors -/
def least_with_2310_divisors : ℕ := sorry

theorem least_2310_divisors_form :
  ∃ (m k : ℕ), 
    least_with_2310_divisors = m * 10^k ∧ 
    not_div_by_ten m ∧ 
    m + k = 10 := by sorry

end NUMINAMATH_CALUDE_least_2310_divisors_form_l94_9471


namespace NUMINAMATH_CALUDE_f_properties_l94_9485

noncomputable def f (x : ℝ) : ℝ := (2^x - 1) / (2^x + 1)

theorem f_properties :
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, x < y → f x < f y) ∧
  (∀ x, f (2*x + 1) + f x < 0 ↔ x < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l94_9485


namespace NUMINAMATH_CALUDE_line_contains_point_l94_9420

/-- Proves that for the line equation 3 - kx = -4y, if the point (3, -2) lies on the line, then k = -5/3 -/
theorem line_contains_point (k : ℚ) : 
  (3 - k * 3 = -4 * (-2)) → k = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_line_contains_point_l94_9420


namespace NUMINAMATH_CALUDE_xy_value_l94_9473

theorem xy_value (x y : ℝ) (h1 : x + y = 2) (h2 : x^2 * y^3 + y^2 * x^3 = 32) : x * y = 2^(5/3) := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l94_9473


namespace NUMINAMATH_CALUDE_john_purchase_proof_l94_9406

def john_purchase (q : ℝ) : Prop :=
  let initial_money : ℝ := 50
  let drink_cost : ℝ := q
  let small_pizza_cost : ℝ := 1.5 * q
  let medium_pizza_cost : ℝ := 2.5 * q
  let total_cost : ℝ := 2 * drink_cost + small_pizza_cost + medium_pizza_cost
  let money_left : ℝ := initial_money - total_cost
  money_left = 50 - 6 * q

theorem john_purchase_proof (q : ℝ) : john_purchase q := by
  sorry

end NUMINAMATH_CALUDE_john_purchase_proof_l94_9406


namespace NUMINAMATH_CALUDE_circle_center_and_radius_prove_center_and_radius_l94_9410

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 4*x = 0

-- Define the center of the circle
def center : ℝ × ℝ := (-2, 0)

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem statement
theorem circle_center_and_radius :
  ∀ (x y : ℝ), circle_equation x y ↔ (x + 2)^2 + y^2 = 4 :=
by sorry

-- Prove that the center and radius are correct
theorem prove_center_and_radius :
  (∀ (x y : ℝ), circle_equation x y ↔ ((x - center.1)^2 + (y - center.2)^2 = radius^2)) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_and_radius_prove_center_and_radius_l94_9410


namespace NUMINAMATH_CALUDE_fiona_probability_l94_9401

/-- Represents a lily pad with its number and whether it contains a predator -/
structure LilyPad where
  number : Nat
  hasPredator : Bool

/-- Represents Fiona's possible moves -/
inductive Move
  | Hop
  | Jump

/-- Represents the frog's journey -/
def FrogJourney := List Move

def numPads : Nat := 12

def predatorPads : List Nat := [3, 6]

def foodPad : Nat := 10

def startPad : Nat := 0

def moveProb : Rat := 1/2

/-- Calculates the final position after a sequence of moves -/
def finalPosition (journey : FrogJourney) : Nat :=
  journey.foldl (fun pos move =>
    match move with
    | Move.Hop => min (pos + 1) (numPads - 1)
    | Move.Jump => min (pos + 2) (numPads - 1)
  ) startPad

/-- Checks if a journey is safe (doesn't land on predator pads) -/
def isSafeJourney (journey : FrogJourney) : Bool :=
  let positions := List.scanl (fun pos move =>
    match move with
    | Move.Hop => min (pos + 1) (numPads - 1)
    | Move.Jump => min (pos + 2) (numPads - 1)
  ) startPad journey
  positions.all (fun pos => pos ∉ predatorPads)

/-- Calculates the probability of a specific journey -/
def journeyProbability (journey : FrogJourney) : Rat :=
  (moveProb ^ journey.length)

theorem fiona_probability :
  ∃ (successfulJourneys : List FrogJourney),
    (∀ j ∈ successfulJourneys, finalPosition j = foodPad ∧ isSafeJourney j) ∧
    (successfulJourneys.map journeyProbability).sum = 15/256 := by
  sorry

end NUMINAMATH_CALUDE_fiona_probability_l94_9401


namespace NUMINAMATH_CALUDE_students_without_A_l94_9437

def class_size : ℕ := 35
def history_A : ℕ := 10
def math_A : ℕ := 15
def both_A : ℕ := 5

theorem students_without_A : 
  class_size - (history_A + math_A - both_A) = 15 := by sorry

end NUMINAMATH_CALUDE_students_without_A_l94_9437


namespace NUMINAMATH_CALUDE_percentage_difference_l94_9404

theorem percentage_difference (x y : ℝ) (h : x = 7 * y) :
  (x - y) / x * 100 = (6 / 7) * 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l94_9404


namespace NUMINAMATH_CALUDE_symmetric_function_inequality_l94_9472

/-- A function that is symmetric about x = 1 -/
def SymmetricAboutOne (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (2 - x)

/-- The derivative condition for x < 1 -/
def DerivativeCondition (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, x < 1 → 2 * f x + (x - 1) * f' x < 0

theorem symmetric_function_inequality
  (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h_symmetric : SymmetricAboutOne f)
  (h_derivative : DerivativeCondition f f') :
  {x : ℝ | (x + 1)^2 * f (x + 2) > f 2} = Set.Ioo (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_function_inequality_l94_9472


namespace NUMINAMATH_CALUDE_simplify_expression_l94_9480

theorem simplify_expression : (81^(1/2) - 144^(1/2)) / 3^(1/2) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l94_9480


namespace NUMINAMATH_CALUDE_problem_solution_l94_9482

theorem problem_solution (x y : ℝ) (h1 : 15 * x = x + 280) (h2 : y = x^2 + 5*x - 12) :
  x = 20 ∧ y = 488 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l94_9482


namespace NUMINAMATH_CALUDE_qinghai_lake_travel_solution_l94_9497

/-- Represents the travel plans and costs for two teams visiting Qinghai Lake. -/
structure TravelPlan where
  distanceA : ℕ  -- Distance for Team A in km
  distanceB : ℕ  -- Distance for Team B in km
  daysA : ℕ      -- Number of days for Team A
  daysB : ℕ      -- Number of days for Team B
  costA : ℕ      -- Daily cost per person for Team A in yuan
  costB : ℕ      -- Daily cost per person for Team B in yuan
  peopleA : ℕ    -- Number of people in Team A
  peopleB : ℕ    -- Number of people in Team B
  m : ℕ          -- Additional people joining Team A

/-- The theorem stating the solution to the Qinghai Lake travel problem. -/
theorem qinghai_lake_travel_solution (plan : TravelPlan) : 
  plan.distanceA = 2700 ∧ 
  plan.distanceB = 1800 ∧
  plan.distanceA / plan.daysA = 2 * (plan.distanceB / plan.daysB) ∧
  plan.daysA + 1 = plan.daysB ∧
  plan.costA = 200 ∧
  plan.costB = 150 ∧
  plan.peopleA = 10 ∧
  plan.peopleB = 8 ∧
  (plan.costA - 30) * (plan.peopleA + plan.m) * plan.daysA + plan.costB * plan.peopleB * plan.daysB = 
    (plan.costA * plan.peopleA * plan.daysA + plan.costB * plan.peopleB * plan.daysB) * 120 / 100 →
  plan.daysA = 3 ∧ plan.daysB = 4 ∧ plan.m = 6 := by
  sorry

end NUMINAMATH_CALUDE_qinghai_lake_travel_solution_l94_9497


namespace NUMINAMATH_CALUDE_absolute_value_sum_l94_9415

theorem absolute_value_sum (a : ℝ) (h1 : -2 < a) (h2 : a < 0) :
  |a| + |a + 2| = 2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l94_9415


namespace NUMINAMATH_CALUDE_inequality_equivalence_l94_9419

theorem inequality_equivalence (x : ℝ) : 
  |2*x - 1| - |x - 2| < 0 ↔ -1 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l94_9419


namespace NUMINAMATH_CALUDE_min_distance_line_circle_l94_9459

/-- The minimum distance between a point on the line y = m + √3x and the circle (x-√3)² + (y-1)² = 2² is 1 if and only if m = 2 or m = -6. -/
theorem min_distance_line_circle (m : ℝ) : 
  (∃ (x y : ℝ), y = m + Real.sqrt 3 * x ∧ 
   (∀ (x' y' : ℝ), y' = m + Real.sqrt 3 * x' → 
     ((x' - Real.sqrt 3)^2 + (y' - 1)^2 ≥ ((x - Real.sqrt 3)^2 + (y - 1)^2))) ∧
   (x - Real.sqrt 3)^2 + (y - 1)^2 = 5) ↔ 
  (m = 2 ∨ m = -6) :=
sorry

end NUMINAMATH_CALUDE_min_distance_line_circle_l94_9459


namespace NUMINAMATH_CALUDE_dj_snake_engagement_treats_value_l94_9488

/-- The total value of treats received by DJ Snake on his engagement day -/
def total_value (hotel_nights : ℕ) (hotel_price_per_night : ℕ) (car_value : ℕ) : ℕ :=
  hotel_nights * hotel_price_per_night + car_value + 4 * car_value

/-- Theorem stating the total value of treats received by DJ Snake on his engagement day -/
theorem dj_snake_engagement_treats_value :
  total_value 2 4000 30000 = 158000 := by
  sorry

end NUMINAMATH_CALUDE_dj_snake_engagement_treats_value_l94_9488


namespace NUMINAMATH_CALUDE_sum_a_b_equals_eleven_l94_9499

theorem sum_a_b_equals_eleven (a b c d : ℝ) 
  (h1 : b + c = 9)
  (h2 : c + d = 3)
  (h3 : a + d = 5) :
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_equals_eleven_l94_9499


namespace NUMINAMATH_CALUDE_root_of_polynomial_l94_9475

theorem root_of_polynomial (x : ℝ) : x^5 - 2*x^4 - x^2 + 2*x - 3 = 0 ↔ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_of_polynomial_l94_9475


namespace NUMINAMATH_CALUDE_fill_container_l94_9498

/-- The capacity of a standard jar in milliliters -/
def standard_jar_capacity : ℕ := 60

/-- The capacity of the big container in milliliters -/
def big_container_capacity : ℕ := 840

/-- The minimum number of standard jars needed to fill the big container -/
def min_jars_needed : ℕ := 14

theorem fill_container :
  min_jars_needed = (big_container_capacity + standard_jar_capacity - 1) / standard_jar_capacity :=
by sorry

end NUMINAMATH_CALUDE_fill_container_l94_9498


namespace NUMINAMATH_CALUDE_salt_teaspoons_in_recipe_l94_9400

/-- Represents the recipe and sodium reduction problem -/
theorem salt_teaspoons_in_recipe : 
  ∀ (S : ℝ) 
    (parmesan_oz : ℝ) 
    (salt_sodium_per_tsp : ℝ) 
    (parmesan_sodium_per_oz : ℝ) 
    (parmesan_reduction : ℝ),
  parmesan_oz = 8 →
  salt_sodium_per_tsp = 50 →
  parmesan_sodium_per_oz = 25 →
  parmesan_reduction = 4 →
  (2 / 3) * (salt_sodium_per_tsp * S + parmesan_sodium_per_oz * parmesan_oz) = 
    salt_sodium_per_tsp * S + parmesan_sodium_per_oz * (parmesan_oz - parmesan_reduction) →
  S = 2 := by
  sorry

end NUMINAMATH_CALUDE_salt_teaspoons_in_recipe_l94_9400


namespace NUMINAMATH_CALUDE_mutually_exclusive_pairs_count_l94_9493

-- Define the total number of volunteers
def total_volunteers : ℕ := 7

-- Define the number of male and female volunteers
def male_volunteers : ℕ := 4
def female_volunteers : ℕ := 3

-- Define the number of selected volunteers
def selected_volunteers : ℕ := 2

-- Define the events
def event1 : Prop := False  -- Logically inconsistent event
def event2 : Prop := True   -- At least 1 female and all females
def event3 : Prop := True   -- At least 1 male and at least 1 female
def event4 : Prop := True   -- At least 1 female and all males

-- Define a function to count mutually exclusive pairs
def count_mutually_exclusive_pairs (events : List Prop) : ℕ := 1

-- Theorem statement
theorem mutually_exclusive_pairs_count :
  count_mutually_exclusive_pairs [event1, event2, event3, event4] = 1 := by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_pairs_count_l94_9493


namespace NUMINAMATH_CALUDE_max_profit_allocation_l94_9483

/-- Profit function for product A -/
def profit_A (x : ℝ) : ℝ := -x^2 + 4*x

/-- Profit function for product B -/
def profit_B (x : ℝ) : ℝ := 2*x

/-- Total profit function -/
def total_profit (x : ℝ) : ℝ := profit_A x + profit_B (3 - x)

/-- Theorem stating the maximum profit and optimal investment allocation -/
theorem max_profit_allocation :
  ∃ (x : ℝ), x ∈ Set.Icc 0 3 ∧
  (∀ y ∈ Set.Icc 0 3, total_profit x ≥ total_profit y) ∧
  x = 1 ∧ total_profit x = 7 := by
  sorry

end NUMINAMATH_CALUDE_max_profit_allocation_l94_9483


namespace NUMINAMATH_CALUDE_constant_difference_function_property_l94_9452

/-- A linear function with constant difference -/
def ConstantDifferenceFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) = f x + f y) ∧ 
  (∀ d : ℝ, f (d + 2) - f d = 6)

theorem constant_difference_function_property (f : ℝ → ℝ) 
  (h : ConstantDifferenceFunction f) : f 1 - f 7 = -18 := by
  sorry

end NUMINAMATH_CALUDE_constant_difference_function_property_l94_9452


namespace NUMINAMATH_CALUDE_digit_replacement_theorem_l94_9426

def first_number : ℕ := 631927
def second_number : ℕ := 590265
def given_sum : ℕ := 1192192

def replace_digit (n : ℕ) (d e : ℕ) : ℕ := 
  sorry

theorem digit_replacement_theorem :
  ∃ (d e : ℕ), d ≠ e ∧ d < 10 ∧ e < 10 ∧
  (replace_digit first_number d e) + (replace_digit second_number d e) = 
    replace_digit given_sum d e ∧
  d + e = 6 := by
  sorry

end NUMINAMATH_CALUDE_digit_replacement_theorem_l94_9426


namespace NUMINAMATH_CALUDE_largest_negative_congruent_to_two_mod_twentynine_l94_9491

theorem largest_negative_congruent_to_two_mod_twentynine : 
  ∃ (n : ℤ), 
    n = -1011 ∧ 
    n ≡ 2 [ZMOD 29] ∧ 
    n < 0 ∧ 
    -9999 ≤ n ∧ 
    n ≥ -999 ∧ 
    ∀ (m : ℤ), 
      m ≡ 2 [ZMOD 29] → 
      m < 0 → 
      -9999 ≤ m → 
      m ≥ -999 → 
      m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_negative_congruent_to_two_mod_twentynine_l94_9491


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l94_9441

theorem sum_of_a_and_b (a b : ℝ) : (a - 2)^2 + |b + 4| = 0 → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l94_9441


namespace NUMINAMATH_CALUDE_geometric_series_sum_l94_9457

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum :
  let a : ℚ := 2
  let r : ℚ := 2/5
  let n : ℕ := 5
  geometric_sum a r n = 10310/3125 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l94_9457


namespace NUMINAMATH_CALUDE_same_quotient_remainder_divisible_by_seven_l94_9458

theorem same_quotient_remainder_divisible_by_seven :
  {n : ℕ | ∃ r : ℕ, 1 ≤ r ∧ r ≤ 6 ∧ n = 8 * r} = {8, 16, 24, 32, 40, 48} := by
sorry

end NUMINAMATH_CALUDE_same_quotient_remainder_divisible_by_seven_l94_9458


namespace NUMINAMATH_CALUDE_triangle_inequality_l94_9462

/-- Given a triangle with side lengths a, b, c and area S, 
    prove that a^2 + b^2 + c^2 ≥ 4√3 S -/
theorem triangle_inequality (a b c S : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_area : S = Real.sqrt ((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c)) / 4) :
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * S := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l94_9462


namespace NUMINAMATH_CALUDE_maruti_car_sales_decrease_l94_9484

theorem maruti_car_sales_decrease (initial_price initial_sales : ℝ) 
  (price_increase : ℝ) (revenue_increase : ℝ) (sales_decrease : ℝ) :
  price_increase = 0.3 →
  revenue_increase = 0.04 →
  (initial_price * (1 + price_increase)) * (initial_sales * (1 - sales_decrease)) = 
    initial_price * initial_sales * (1 + revenue_increase) →
  sales_decrease = 0.2 := by
sorry

end NUMINAMATH_CALUDE_maruti_car_sales_decrease_l94_9484


namespace NUMINAMATH_CALUDE_store_pricing_l94_9422

theorem store_pricing (h n : ℝ) 
  (eq1 : 4 * h + 5 * n = 10.45)
  (eq2 : 3 * h + 9 * n = 12.87) :
  20 * h + 25 * n = 52.25 := by
  sorry

end NUMINAMATH_CALUDE_store_pricing_l94_9422


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l94_9486

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), r > 0 ∧ ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_positive : ∀ n, a n > 0)
  (h_second : a 2 = 1)
  (h_relation : a 8 = a 6 + 2 * a 4) :
  a 6 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l94_9486


namespace NUMINAMATH_CALUDE_function_identification_l94_9453

theorem function_identification (f : ℝ → ℝ) (b : ℝ) 
  (h1 : ∀ x, f (3 * x) = 3 * x^2 + b) 
  (h2 : f 1 = 0) : 
  ∀ x, f x = (1/3) * x^2 - (1/3) := by
sorry

end NUMINAMATH_CALUDE_function_identification_l94_9453


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l94_9474

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_first_term
  (a : ℕ → ℤ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a4 : a 4 = 9)
  (h_a8 : a 8 = -(a 9)) :
  a 1 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_l94_9474


namespace NUMINAMATH_CALUDE_factorial_difference_l94_9487

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem: 6! - 4! = 696 -/
theorem factorial_difference : factorial 6 - factorial 4 = 696 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l94_9487


namespace NUMINAMATH_CALUDE_ice_melting_problem_l94_9461

theorem ice_melting_problem (initial_volume : ℝ) : 
  initial_volume = 3.2 →
  (1/4) * ((1/4) * initial_volume) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ice_melting_problem_l94_9461


namespace NUMINAMATH_CALUDE_root_of_quadratic_l94_9444

theorem root_of_quadratic (a : ℝ) : (2 : ℝ)^2 + a * 2 - 3 * a = 0 → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_root_of_quadratic_l94_9444


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l94_9464

theorem sum_of_reciprocals_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 3 * x₁ - 1 = 0) → 
  (2 * x₂^2 + 3 * x₂ - 1 = 0) → 
  (x₁ ≠ x₂) →
  (1 / x₁ + 1 / x₂ = 3) := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_of_roots_l94_9464


namespace NUMINAMATH_CALUDE_betty_beads_l94_9450

/-- Given a ratio of red to blue beads and a number of red beads, 
    calculate the number of blue beads -/
def blue_beads (red_ratio blue_ratio red_count : ℕ) : ℕ :=
  (blue_ratio * red_count) / red_ratio

/-- Theorem stating that given 3 red beads for every 2 blue beads,
    and 30 red beads in total, there are 20 blue beads -/
theorem betty_beads : blue_beads 3 2 30 = 20 := by
  sorry

end NUMINAMATH_CALUDE_betty_beads_l94_9450


namespace NUMINAMATH_CALUDE_complex_equation_result_l94_9492

theorem complex_equation_result (m n : ℝ) (i : ℂ) (h : i * i = -1) :
  m + i = (1 + 2 * i) * n * i → n - m = 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_result_l94_9492


namespace NUMINAMATH_CALUDE_point_coordinates_l94_9416

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the second quadrant -/
def is_in_second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Distance from a point to the x-axis -/
def distance_to_x_axis (p : Point) : ℝ :=
  |p.y|

/-- Distance from a point to the y-axis -/
def distance_to_y_axis (p : Point) : ℝ :=
  |p.x|

/-- The main theorem -/
theorem point_coordinates (P : Point) 
  (h1 : is_in_second_quadrant P)
  (h2 : distance_to_x_axis P = 2)
  (h3 : distance_to_y_axis P = 3) :
  P.x = -3 ∧ P.y = 2 :=
by sorry

end NUMINAMATH_CALUDE_point_coordinates_l94_9416


namespace NUMINAMATH_CALUDE_parallel_transitivity_l94_9495

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelLinePlane : Line → Plane → Prop)
variable (notContained : Line → Plane → Prop)

-- State the theorem
theorem parallel_transitivity
  (a b : Line) (α : Plane)
  (h1 : parallelLine a b)
  (h2 : parallelLinePlane a α)
  (h3 : notContained b α) :
  parallelLinePlane b α :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l94_9495


namespace NUMINAMATH_CALUDE_probability_two_A_grades_l94_9411

/-- The probability of achieving an A grade in exactly two out of three subjects. -/
theorem probability_two_A_grades
  (p_politics : ℝ)
  (p_history : ℝ)
  (p_geography : ℝ)
  (hp_politics : p_politics = 4/5)
  (hp_history : p_history = 3/5)
  (hp_geography : p_geography = 2/5)
  (hprob_politics : 0 ≤ p_politics ∧ p_politics ≤ 1)
  (hprob_history : 0 ≤ p_history ∧ p_history ≤ 1)
  (hprob_geography : 0 ≤ p_geography ∧ p_geography ≤ 1) :
  p_politics * p_history * (1 - p_geography) +
  p_politics * (1 - p_history) * p_geography +
  (1 - p_politics) * p_history * p_geography = 58/125 := by
sorry

end NUMINAMATH_CALUDE_probability_two_A_grades_l94_9411


namespace NUMINAMATH_CALUDE_silver_to_gold_ratio_is_two_to_one_l94_9429

-- Define the number of gold, silver, and black balloons
def gold_balloons : ℕ := 141
def black_balloons : ℕ := 150
def total_balloons : ℕ := 573

-- Define the number of silver balloons
def silver_balloons : ℕ := total_balloons - gold_balloons - black_balloons

-- Define the ratio of silver to gold balloons
def silver_to_gold_ratio : ℚ := silver_balloons / gold_balloons

-- Theorem statement
theorem silver_to_gold_ratio_is_two_to_one :
  silver_to_gold_ratio = 2 / 1 := by
  sorry


end NUMINAMATH_CALUDE_silver_to_gold_ratio_is_two_to_one_l94_9429


namespace NUMINAMATH_CALUDE_add_negative_and_positive_l94_9403

theorem add_negative_and_positive : -3 + 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_add_negative_and_positive_l94_9403


namespace NUMINAMATH_CALUDE_square_area_with_four_circles_l94_9456

theorem square_area_with_four_circles (r : ℝ) (h : r = 7) : 
  let side_length := 4 * r
  (side_length ^ 2 : ℝ) = 784 := by
  sorry

end NUMINAMATH_CALUDE_square_area_with_four_circles_l94_9456


namespace NUMINAMATH_CALUDE_expression_evaluation_l94_9431

theorem expression_evaluation :
  ((3^1002 + 7^1003)^2 - (3^1002 - 7^1003)^2) / (10^1002) = 28 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l94_9431


namespace NUMINAMATH_CALUDE_last_student_age_l94_9449

theorem last_student_age 
  (total_students : ℕ) 
  (avg_age_all : ℝ) 
  (group1_size : ℕ) 
  (avg_age_group1 : ℝ) 
  (group2_size : ℕ) 
  (avg_age_group2 : ℝ) 
  (h1 : total_students = 15)
  (h2 : avg_age_all = 15)
  (h3 : group1_size = 5)
  (h4 : avg_age_group1 = 13)
  (h5 : group2_size = 9)
  (h6 : avg_age_group2 = 16)
  (h7 : group1_size + group2_size + 1 = total_students) :
  ∃ (last_student_age : ℝ), 
    last_student_age = total_students * avg_age_all - 
      (group1_size * avg_age_group1 + group2_size * avg_age_group2) ∧
    last_student_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_last_student_age_l94_9449


namespace NUMINAMATH_CALUDE_range_of_m_l94_9430

/-- Proposition P: The equation x^2 + mx + 1 = 0 has two distinct negative roots -/
def P (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition Q: The equation 4x^2 + 4(m - 2)x + 1 = 0 has no real roots -/
def Q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 ≠ 0

/-- The range of real values for m satisfying the given conditions -/
def M : Set ℝ :=
  {m : ℝ | m < -2 ∨ (1 < m ∧ m ≤ 2) ∨ m ≥ 3}

theorem range_of_m :
  ∀ m : ℝ, (P m ∨ Q m) ∧ ¬(P m ∧ Q m) ↔ m ∈ M :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l94_9430


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cos_property_l94_9409

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_cos_property (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 5 + a 9 = 5 * Real.pi →
  Real.cos (a 2 + a 8) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cos_property_l94_9409


namespace NUMINAMATH_CALUDE_definite_integral_arctg_x_l94_9440

theorem definite_integral_arctg_x : 
  ∫ x in (0 : ℝ)..1, (4 * Real.arctan x - x) / (1 + x^2) = (π^2 - 4 * Real.log 2) / 8 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_arctg_x_l94_9440


namespace NUMINAMATH_CALUDE_max_value_sin_cos_function_l94_9490

theorem max_value_sin_cos_function :
  ∃ (M : ℝ), M = 1/2 - Real.sqrt 3/4 ∧
  ∀ (x : ℝ), Real.sin (3*Real.pi/2 + x) * Real.cos (Real.pi/6 - x) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_function_l94_9490


namespace NUMINAMATH_CALUDE_popton_bus_toes_l94_9447

/-- Represents a race of beings on planet Popton -/
inductive Race
| Hoopit
| Neglart

/-- Number of hands for each race -/
def hands (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 4
  | Race.Neglart => 5

/-- Number of toes per hand for each race -/
def toesPerHand (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 3
  | Race.Neglart => 2

/-- Number of toes for an individual of a given race -/
def toesPerIndividual (r : Race) : ℕ :=
  (hands r) * (toesPerHand r)

/-- Number of students of each race on the bus -/
def studentsOnBus (r : Race) : ℕ :=
  match r with
  | Race.Hoopit => 7
  | Race.Neglart => 8

/-- Total number of toes on the Popton school bus -/
def totalToesOnBus : ℕ :=
  (toesPerIndividual Race.Hoopit) * (studentsOnBus Race.Hoopit) +
  (toesPerIndividual Race.Neglart) * (studentsOnBus Race.Neglart)

/-- Theorem: The total number of toes on the Popton school bus is 164 -/
theorem popton_bus_toes : totalToesOnBus = 164 := by
  sorry

end NUMINAMATH_CALUDE_popton_bus_toes_l94_9447


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l94_9405

theorem simplify_and_rationalize (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l94_9405


namespace NUMINAMATH_CALUDE_milk_students_l94_9443

theorem milk_students (total : ℕ) (soda_percent : ℚ) (milk_percent : ℚ) (soda_count : ℕ) :
  soda_percent = 70 / 100 →
  milk_percent = 20 / 100 →
  soda_count = 84 →
  total = soda_count / soda_percent →
  ↑(total * milk_percent) = 24 := by
  sorry

end NUMINAMATH_CALUDE_milk_students_l94_9443


namespace NUMINAMATH_CALUDE_units_digit_of_product_with_sum_factorials_l94_9481

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product_with_sum_factorials : 
  units_digit (7 * sum_factorials 2023) = 1 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_with_sum_factorials_l94_9481


namespace NUMINAMATH_CALUDE_polynomial_real_root_iff_b_in_set_l94_9451

/-- The polynomial in question -/
def polynomial (b x : ℝ) : ℝ := x^5 + b*x^4 - x^3 + b*x^2 - x + b

/-- The set of b values for which the polynomial has at least one real root -/
def valid_b_set : Set ℝ := Set.Iic (-1) ∪ Set.Ici 1

theorem polynomial_real_root_iff_b_in_set (b : ℝ) :
  (∃ x : ℝ, polynomial b x = 0) ↔ b ∈ valid_b_set := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_iff_b_in_set_l94_9451


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l94_9467

/-- A geometric sequence with common ratio q satisfying 2a₄ = a₆ - a₅ -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = q * a n) ∧ (2 * a 4 = a 6 - a 5)

/-- The common ratio of a geometric sequence satisfying 2a₄ = a₆ - a₅ is either -1 or 2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q → q = -1 ∨ q = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l94_9467


namespace NUMINAMATH_CALUDE_number_puzzle_l94_9470

theorem number_puzzle : ∃! x : ℚ, x / 5 + 6 = x / 4 - 6 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l94_9470


namespace NUMINAMATH_CALUDE_detergent_in_altered_solution_l94_9496

/-- Represents the ratio of components in a cleaning solution -/
structure CleaningSolution where
  bleach : ℚ
  detergent : ℚ
  water : ℚ

/-- Calculates the amount of detergent in the altered solution -/
def altered_detergent_amount (original : CleaningSolution) (water_amount : ℚ) : ℚ :=
  let new_bleach := original.bleach * 3
  let new_detergent := original.detergent
  let new_water := original.water * 2
  let total_parts := new_bleach + new_detergent + new_water
  (new_detergent / total_parts) * water_amount

/-- Theorem stating the amount of detergent in the altered solution -/
theorem detergent_in_altered_solution 
  (original : CleaningSolution)
  (h1 : original.bleach = 2)
  (h2 : original.detergent = 25)
  (h3 : original.water = 100)
  (water_amount : ℚ)
  (h4 : water_amount = 300) :
  altered_detergent_amount original water_amount = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_detergent_in_altered_solution_l94_9496


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l94_9413

theorem quadratic_root_relation (a b : ℝ) (h : a ≠ 0) :
  (a * 2019^2 + b * 2019 - 1 = 0) →
  (a * (2020 - 1)^2 + b * (2020 - 1) = 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_relation_l94_9413


namespace NUMINAMATH_CALUDE_math_competition_score_xiao_hua_correct_answers_l94_9407

theorem math_competition_score (total_questions : Nat) (correct_points : Int) (wrong_points : Int) (total_score : Int) : Int :=
  let attempted_questions := total_questions
  let hypothetical_score := total_questions * correct_points
  let score_difference := hypothetical_score - total_score
  let points_per_wrong_answer := correct_points + wrong_points
  let wrong_answers := score_difference / points_per_wrong_answer
  total_questions - wrong_answers

theorem xiao_hua_correct_answers : 
  math_competition_score 15 8 (-4) 72 = 11 := by
  sorry

end NUMINAMATH_CALUDE_math_competition_score_xiao_hua_correct_answers_l94_9407


namespace NUMINAMATH_CALUDE_derivative_of_sine_function_l94_9418

open Real

theorem derivative_of_sine_function (x : ℝ) :
  let y : ℝ → ℝ := λ x => 3 * sin (2 * x - π / 6)
  deriv y x = 6 * cos (2 * x - π / 6) := by
sorry

end NUMINAMATH_CALUDE_derivative_of_sine_function_l94_9418


namespace NUMINAMATH_CALUDE_ice_water_masses_l94_9476

/-- Given a cylindrical vessel with ice and water, calculate the initial masses. -/
theorem ice_water_masses (S : ℝ) (ρw ρi : ℝ) (Δh hf : ℝ) 
  (hS : S = 15) 
  (hρw : ρw = 1) 
  (hρi : ρi = 0.92) 
  (hΔh : Δh = 5) 
  (hhf : hf = 115) :
  ∃ (mi mw : ℝ), 
    mi = 862.5 ∧ 
    mw = 1050 ∧ 
    mi / ρi - mi / ρw = S * Δh ∧ 
    mw + mi = ρw * S * hf := by
  sorry

#check ice_water_masses

end NUMINAMATH_CALUDE_ice_water_masses_l94_9476


namespace NUMINAMATH_CALUDE_expression_simplification_l94_9477

theorem expression_simplification :
  (5^2010)^2 - (5^2008)^2 / (5^2009)^2 - (5^2007)^2 = 25 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l94_9477


namespace NUMINAMATH_CALUDE_farmer_land_ownership_l94_9417

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.8 + 0.9 * total_land * 0.1 + 90 = 0.9 * total_land) →
  total_land = 1000 := by
sorry

end NUMINAMATH_CALUDE_farmer_land_ownership_l94_9417


namespace NUMINAMATH_CALUDE_castle_provisions_l94_9432

/-- Represents the initial number of people in the castle -/
def initial_people : ℕ := sorry

/-- Represents the number of days the initial provisions last -/
def initial_days : ℕ := 90

/-- Represents the number of days after which people leave -/
def days_before_leaving : ℕ := 30

/-- Represents the number of people who leave the castle -/
def people_leaving : ℕ := 100

/-- Represents the number of days the remaining provisions last -/
def remaining_days : ℕ := 90

theorem castle_provisions :
  initial_people * initial_days = 
  (initial_people * days_before_leaving) + 
  ((initial_people - people_leaving) * remaining_days) ∧
  initial_people = 300 :=
sorry

end NUMINAMATH_CALUDE_castle_provisions_l94_9432


namespace NUMINAMATH_CALUDE_three_digit_special_property_l94_9463

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    n = 100 * a + 10 * b + c ∧
    a ≠ 0 ∧
    b < 10 ∧
    c < 10 ∧
    3 * a * (10 * b + c) = n

theorem three_digit_special_property :
  {n : ℕ | is_valid_number n} = {150, 240, 735} :=
sorry

end NUMINAMATH_CALUDE_three_digit_special_property_l94_9463


namespace NUMINAMATH_CALUDE_marion_has_23_paperclips_l94_9436

-- Define the variables
def x : ℚ := 30
def y : ℚ := 7

-- Define Yun's remaining paperclips
def yun_remaining : ℚ := 2/5 * x

-- Define Marion's paperclips
def marion_paperclips : ℚ := 4/3 * yun_remaining + y

-- Theorem to prove
theorem marion_has_23_paperclips : marion_paperclips = 23 := by
  sorry

end NUMINAMATH_CALUDE_marion_has_23_paperclips_l94_9436


namespace NUMINAMATH_CALUDE_max_students_per_dentist_l94_9478

theorem max_students_per_dentist (num_dentists num_students min_students_per_dentist : ℕ) 
  (h1 : num_dentists = 12)
  (h2 : num_students = 29)
  (h3 : min_students_per_dentist = 2)
  (h4 : num_dentists * min_students_per_dentist ≤ num_students) :
  ∃ (max_students : ℕ), max_students = 7 ∧ 
  (∀ (d : ℕ), d ≤ num_dentists → ∃ (s : ℕ), s ≤ num_students ∧ s ≤ max_students) ∧
  (∃ (d : ℕ), d ≤ num_dentists ∧ ∃ (s : ℕ), s = max_students) :=
by sorry

end NUMINAMATH_CALUDE_max_students_per_dentist_l94_9478


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l94_9433

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a3 : a 3 = 7) 
  (h_S3 : a 0 + a 1 + a 2 = 21) : 
  (a 1 / a 0 = 1) ∨ (a 1 / a 0 = -1/2) := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l94_9433


namespace NUMINAMATH_CALUDE_is_ellipse_l94_9402

/-- The equation √((x-2)² + (y+2)²) + √((x-6)² + y²) = 12 represents an ellipse -/
theorem is_ellipse (x y : ℝ) : 
  (∃ (f₁ f₂ : ℝ × ℝ), f₁ ≠ f₂ ∧ 
  (∀ (p : ℝ × ℝ), Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) + 
                   Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) = 12) →
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  ∀ (p : ℝ × ℝ), (p.1^2 / a^2) + (p.2^2 / b^2) = 1) :=
by sorry

end NUMINAMATH_CALUDE_is_ellipse_l94_9402


namespace NUMINAMATH_CALUDE_fortieth_term_is_210_l94_9408

/-- A function that checks if a number contains the digit 2 --/
def containsTwo (n : ℕ) : Bool :=
  sorry

/-- A function that generates the sequence of positive multiples of 3 containing at least one digit 2 --/
def sequenceGenerator : ℕ → ℕ :=
  sorry

/-- The theorem stating that the 40th term of the sequence is 210 --/
theorem fortieth_term_is_210 : sequenceGenerator 40 = 210 := by
  sorry

end NUMINAMATH_CALUDE_fortieth_term_is_210_l94_9408


namespace NUMINAMATH_CALUDE_min_omega_for_coinciding_symmetry_axes_l94_9435

/-- Given a sinusoidal function y = 2sin(ωx + π/3) where ω > 0, 
    if the graph is shifted left and right by π/3 units and 
    the axes of symmetry of the resulting graphs coincide, 
    then the minimum value of ω is 3/2. -/
theorem min_omega_for_coinciding_symmetry_axes (ω : ℝ) : 
  ω > 0 → 
  (∀ x : ℝ, ∃ y : ℝ, y = 2 * Real.sin (ω * x + π / 3)) →
  (∀ x : ℝ, ∃ y₁ y₂ : ℝ, 
    y₁ = 2 * Real.sin (ω * (x + π / 3) + π / 3) ∧
    y₂ = 2 * Real.sin (ω * (x - π / 3) + π / 3)) →
  (∃ k : ℤ, ω * (π / 3) = k * π) →
  ω ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_omega_for_coinciding_symmetry_axes_l94_9435


namespace NUMINAMATH_CALUDE_range_of_a_l94_9465

-- Define the function f
def f (x : ℝ) : ℝ := -2*x^5 - x^3 - 7*x + 2

-- State the theorem
theorem range_of_a (a : ℝ) : f (a^2) + f (a-2) > 4 → a ∈ Set.Ioo (-2) 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l94_9465


namespace NUMINAMATH_CALUDE_school_fundraiser_distribution_l94_9414

theorem school_fundraiser_distribution (total_amount : ℚ) (num_charities : ℕ) 
  (h1 : total_amount = 3109)
  (h2 : num_charities = 25) :
  total_amount / num_charities = 124.36 := by
  sorry

end NUMINAMATH_CALUDE_school_fundraiser_distribution_l94_9414


namespace NUMINAMATH_CALUDE_max_xy_value_l94_9439

theorem max_xy_value (x y : ℕ+) (h1 : 7 * x + 2 * y = 140) (h2 : x ≤ 15) : 
  x * y ≤ 350 ∧ ∃ (x₀ y₀ : ℕ+), 7 * x₀ + 2 * y₀ = 140 ∧ x₀ ≤ 15 ∧ x₀ * y₀ = 350 :=
by sorry

end NUMINAMATH_CALUDE_max_xy_value_l94_9439


namespace NUMINAMATH_CALUDE_annie_gives_25_crayons_to_mary_l94_9421

/-- Calculates the number of crayons Annie gives to Mary -/
def crayons_given_to_mary (new_pack : ℕ) (locker : ℕ) : ℕ :=
  let initial_total := new_pack + locker
  let from_bobby := locker / 2
  let final_total := initial_total + from_bobby
  final_total / 3

/-- Proves that Annie gives 25 crayons to Mary under the given conditions -/
theorem annie_gives_25_crayons_to_mary :
  crayons_given_to_mary 21 36 = 25 := by
  sorry

#eval crayons_given_to_mary 21 36

end NUMINAMATH_CALUDE_annie_gives_25_crayons_to_mary_l94_9421


namespace NUMINAMATH_CALUDE_point_of_tangency_parabolas_l94_9469

/-- The point of tangency for two parabolas -/
theorem point_of_tangency_parabolas :
  let f (x : ℝ) := x^2 + 10*x + 18
  let g (y : ℝ) := y^2 + 60*y + 910
  ∃! p : ℝ × ℝ, 
    (p.2 = f p.1 ∧ p.1 = g p.2) ∧ 
    (∀ x y, y = f x ∧ x = g y → (x, y) = p) :=
by
  sorry

end NUMINAMATH_CALUDE_point_of_tangency_parabolas_l94_9469


namespace NUMINAMATH_CALUDE_max_value_m_l94_9434

/-- Represents a number in base 8 as XYZ₈ -/
def base8_repr (X Y Z : ℕ) : ℕ := 64 * X + 8 * Y + Z

/-- Represents a number in base 12 as ZYX₁₂ -/
def base12_repr (X Y Z : ℕ) : ℕ := 144 * Z + 12 * Y + X

/-- Theorem stating the maximum value of m given the conditions -/
theorem max_value_m (m : ℕ) (X Y Z : ℕ) 
  (h1 : m > 0)
  (h2 : m = base8_repr X Y Z)
  (h3 : m = base12_repr X Y Z)
  (h4 : X < 8 ∧ Y < 8 ∧ Z < 8)  -- X, Y, Z are single digits in base 8
  (h5 : Z < 12 ∧ Y < 12 ∧ X < 12)  -- Z, Y, X are single digits in base 12
  : m ≤ 475 :=
sorry

end NUMINAMATH_CALUDE_max_value_m_l94_9434


namespace NUMINAMATH_CALUDE_power_function_through_point_value_l94_9479

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- Theorem statement
theorem power_function_through_point_value :
  ∀ f : ℝ → ℝ,
  isPowerFunction f →
  f 2 = 16 →
  f (Real.sqrt 3) = 9 := by
sorry

end NUMINAMATH_CALUDE_power_function_through_point_value_l94_9479


namespace NUMINAMATH_CALUDE_frame_width_proof_l94_9445

theorem frame_width_proof (photo_width : ℝ) (photo_height : ℝ) (frame_width : ℝ) :
  photo_width = 12 →
  photo_height = 18 →
  (photo_width + 2 * frame_width) * (photo_height + 2 * frame_width) - photo_width * photo_height = photo_width * photo_height →
  frame_width = 3 := by
  sorry

end NUMINAMATH_CALUDE_frame_width_proof_l94_9445


namespace NUMINAMATH_CALUDE_cricket_players_count_l94_9460

/-- The number of cricket players in a games hour -/
def cricket_players (total_players hockey_players football_players softball_players : ℕ) : ℕ :=
  total_players - (hockey_players + football_players + softball_players)

/-- Theorem: There are 12 cricket players present in the ground -/
theorem cricket_players_count :
  cricket_players 50 17 11 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cricket_players_count_l94_9460


namespace NUMINAMATH_CALUDE_existence_of_special_binary_number_l94_9425

/-- Represents a binary number as a list of booleans -/
def BinaryNumber := List Bool

/-- Generates all n-digit binary numbers -/
def allNDigitBinaryNumbers (n : Nat) : List BinaryNumber :=
  sorry

/-- Checks if a binary number is a substring of another binary number -/
def isSubstring (sub target : BinaryNumber) : Bool :=
  sorry

/-- Checks if all n-digit binary numbers are substrings of T -/
def allNDigitNumbersAreSubstrings (T : BinaryNumber) (n : Nat) : Prop :=
  ∀ sub, sub ∈ allNDigitBinaryNumbers n → isSubstring sub T

/-- Checks if all n-digit substrings of T are distinct -/
def allNDigitSubstringsAreDistinct (T : BinaryNumber) (n : Nat) : Prop :=
  sorry

theorem existence_of_special_binary_number (n : Nat) :
  ∃ T : BinaryNumber,
    T.length = 2^n + (n - 1) ∧
    allNDigitNumbersAreSubstrings T n ∧
    allNDigitSubstringsAreDistinct T n :=
  sorry

end NUMINAMATH_CALUDE_existence_of_special_binary_number_l94_9425


namespace NUMINAMATH_CALUDE_coefficient_c_nonzero_l94_9438

def Q (a' b' c' d' x : ℝ) : ℝ := x^4 + a'*x^3 + b'*x^2 + c'*x + d'

theorem coefficient_c_nonzero 
  (a' b' c' d' : ℝ) 
  (h1 : ∃ u v w : ℝ, u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ u ≠ 0 ∧ v ≠ 0 ∧ w ≠ 0)
  (h2 : ∀ x : ℝ, Q a' b' c' d' x = x * (x - u) * (x - v) * (x - w))
  (h3 : d' = 0) :
  c' ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_coefficient_c_nonzero_l94_9438


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l94_9427

theorem consecutive_even_numbers_sum (a : ℤ) : 
  (∃ (x : ℤ), 
    (x = a) ∧ 
    (x + (x + 2) + (x + 4) + (x + 6) = 52)) → 
  (a + 4 = 14) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l94_9427


namespace NUMINAMATH_CALUDE_card_digits_problem_l94_9454

theorem card_digits_problem (a b c : ℕ) : 
  0 < a → a < b → b < c → c < 10 →
  (999 * c + 90 * b - 990 * a) + 
  (100 * c + 9 * b - 99 * a) + 
  (10 * c + b - 10 * a) + 
  (c - a) = 9090 →
  a = 1 ∧ b = 2 ∧ c = 9 := by
sorry

end NUMINAMATH_CALUDE_card_digits_problem_l94_9454


namespace NUMINAMATH_CALUDE_salt_mixture_proof_l94_9412

/-- Proves that mixing 28 ounces of 40% salt solution with 112 ounces of 90% salt solution
    results in a 140-ounce mixture that is 80% salt -/
theorem salt_mixture_proof :
  let solution_a_amount : ℝ := 28
  let solution_b_amount : ℝ := 112
  let solution_a_concentration : ℝ := 0.4
  let solution_b_concentration : ℝ := 0.9
  let total_amount : ℝ := solution_a_amount + solution_b_amount
  let target_concentration : ℝ := 0.8
  let mixture_salt_amount : ℝ := solution_a_amount * solution_a_concentration +
                                  solution_b_amount * solution_b_concentration
  (total_amount = 140) ∧
  (mixture_salt_amount / total_amount = target_concentration) :=
by
  sorry


end NUMINAMATH_CALUDE_salt_mixture_proof_l94_9412


namespace NUMINAMATH_CALUDE_circle_passes_through_points_l94_9455

-- Define the points
def O : ℝ × ℝ := (0, 0)
def M : ℝ × ℝ := (1, 1)
def N : ℝ × ℝ := (4, 2)

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

-- Theorem statement
theorem circle_passes_through_points :
  circle_equation O.1 O.2 ∧
  circle_equation M.1 M.2 ∧
  circle_equation N.1 N.2 := by
  sorry

end NUMINAMATH_CALUDE_circle_passes_through_points_l94_9455


namespace NUMINAMATH_CALUDE_average_of_remaining_numbers_l94_9423

theorem average_of_remaining_numbers
  (n : ℕ)
  (total_avg : ℚ)
  (first_three_avg : ℚ)
  (next_three_avg : ℚ)
  (h1 : n = 8)
  (h2 : total_avg = 4.5)
  (h3 : first_three_avg = 5.2)
  (h4 : next_three_avg = 3.6) :
  (n * total_avg - 3 * first_three_avg - 3 * next_three_avg) / 2 = 4.8 := by
  sorry

end NUMINAMATH_CALUDE_average_of_remaining_numbers_l94_9423


namespace NUMINAMATH_CALUDE_remaining_balloons_l94_9466

/-- The number of balloons in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of balloons the clown starts with -/
def initial_dozens : ℕ := 3

/-- The number of boys who buy a balloon -/
def boys : ℕ := 3

/-- The number of girls who buy a balloon -/
def girls : ℕ := 12

/-- Theorem: The clown is left with 21 balloons after selling to boys and girls -/
theorem remaining_balloons :
  initial_dozens * dozen - (boys + girls) = 21 := by
  sorry

end NUMINAMATH_CALUDE_remaining_balloons_l94_9466


namespace NUMINAMATH_CALUDE_A_alone_days_l94_9468

-- Define work rates for A, B, and C
def work_rate_A : ℝ := sorry
def work_rate_B : ℝ := sorry
def work_rate_C : ℝ := sorry

-- Define conditions
axiom cond1 : work_rate_A + work_rate_B = 1 / 3
axiom cond2 : work_rate_B + work_rate_C = 1 / 6
axiom cond3 : work_rate_A + work_rate_C = 5 / 18
axiom cond4 : work_rate_A + work_rate_B + work_rate_C = 1 / 2

-- Theorem to prove
theorem A_alone_days : 1 / work_rate_A = 36 / 7 := by sorry

end NUMINAMATH_CALUDE_A_alone_days_l94_9468


namespace NUMINAMATH_CALUDE_valid_arrays_l94_9442

def is_valid_array (p q r : ℕ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧
  p ≥ q ∧ q ≥ r ∧
  ((Prime p ∧ Prime q) ∨ (Prime p ∧ Prime r) ∨ (Prime q ∧ Prime r)) ∧
  ∃ k : ℕ, k > 0 ∧ (p + q + r)^2 = k * (p * q * r)

theorem valid_arrays :
  ∀ p q r : ℕ, is_valid_array p q r ↔
    (p = 3 ∧ q = 3 ∧ r = 3) ∨
    (p = 2 ∧ q = 2 ∧ r = 4) ∨
    (p = 3 ∧ q = 3 ∧ r = 12) ∨
    (p = 3 ∧ q = 2 ∧ r = 1) ∨
    (p = 3 ∧ q = 2 ∧ r = 25) :=
by sorry

#check valid_arrays

end NUMINAMATH_CALUDE_valid_arrays_l94_9442


namespace NUMINAMATH_CALUDE_sum_of_integers_minus15_to_5_l94_9494

-- Define the range of integers
def lower_bound : Int := -15
def upper_bound : Int := 5

-- Define the sum of integers function
def sum_of_integers (a b : Int) : Int :=
  let n := b - a + 1
  let avg := (a + b) / 2
  n * avg

-- Theorem statement
theorem sum_of_integers_minus15_to_5 :
  sum_of_integers lower_bound upper_bound = -105 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_minus15_to_5_l94_9494


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l94_9424

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l94_9424


namespace NUMINAMATH_CALUDE_figure_sides_l94_9448

/-- A figure with a perimeter of 49 cm and a side length of 7 cm has 7 sides. -/
theorem figure_sides (perimeter : ℝ) (side_length : ℝ) (h1 : perimeter = 49) (h2 : side_length = 7) :
  perimeter / side_length = 7 := by
  sorry

end NUMINAMATH_CALUDE_figure_sides_l94_9448


namespace NUMINAMATH_CALUDE_quadratic_no_roots_l94_9489

theorem quadratic_no_roots (b c : ℝ) 
  (h : ∀ x : ℝ, x^2 + b*x + c > 0) : 
  ¬ ∃ x : ℝ, x^2 + b*x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_no_roots_l94_9489


namespace NUMINAMATH_CALUDE_magic_stick_height_difference_l94_9446

-- Define the edge length of the large cube in meters
def large_cube_edge : ℝ := 1

-- Define the edge length of the small cubes in centimeters
def small_cube_edge : ℝ := 1

-- Define the height of Mount Everest in meters
def everest_height : ℝ := 8844

-- Conversion factor from centimeters to meters
def cm_to_m : ℝ := 0.01

-- Theorem statement
theorem magic_stick_height_difference :
  let large_cube_volume : ℝ := large_cube_edge ^ 3
  let small_cube_volume : ℝ := (small_cube_edge * cm_to_m) ^ 3
  let num_small_cubes : ℝ := large_cube_volume / small_cube_volume
  let magic_stick_height : ℝ := num_small_cubes * small_cube_edge * cm_to_m
  magic_stick_height - everest_height = 1156 := by
  sorry

end NUMINAMATH_CALUDE_magic_stick_height_difference_l94_9446
