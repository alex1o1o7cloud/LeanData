import Mathlib

namespace NUMINAMATH_CALUDE_no_solution_to_equation_l475_47505

theorem no_solution_to_equation :
  ¬∃ x : ℝ, x ≠ 0 ∧ x ≠ 5 ∧ (3 * x^2 - 15 * x) / (x^2 - 5 * x) = x - 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_to_equation_l475_47505


namespace NUMINAMATH_CALUDE_erasers_lost_l475_47579

def initial_erasers : ℕ := 95
def final_erasers : ℕ := 53

theorem erasers_lost : initial_erasers - final_erasers = 42 := by
  sorry

end NUMINAMATH_CALUDE_erasers_lost_l475_47579


namespace NUMINAMATH_CALUDE_min_sum_squares_l475_47519

theorem min_sum_squares (p q r s t u v w : Int) : 
  p ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  q ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  r ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  s ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  t ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  u ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  v ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  w ∈ ({-8, -6, -4, -1, 1, 3, 5, 14} : Set Int) →
  p ≠ q → p ≠ r → p ≠ s → p ≠ t → p ≠ u → p ≠ v → p ≠ w →
  q ≠ r → q ≠ s → q ≠ t → q ≠ u → q ≠ v → q ≠ w →
  r ≠ s → r ≠ t → r ≠ u → r ≠ v → r ≠ w →
  s ≠ t → s ≠ u → s ≠ v → s ≠ w →
  t ≠ u → t ≠ v → t ≠ w →
  u ≠ v → u ≠ w →
  v ≠ w →
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 10 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l475_47519


namespace NUMINAMATH_CALUDE_existence_of_multiple_representations_l475_47537

def V_n (n : ℕ) : Set ℕ := {m | ∃ k : ℕ, m = 1 + k * n}

def indecomposable (n : ℕ) (m : ℕ) : Prop :=
  m ∈ V_n n ∧ ∀ p q : ℕ, p ∈ V_n n → q ∈ V_n n → p * q ≠ m

theorem existence_of_multiple_representations (n : ℕ) (h : n > 2) :
  ∃ r : ℕ, r ∈ V_n n ∧
    ∃ a b c d : ℕ,
      indecomposable n a ∧
      indecomposable n b ∧
      indecomposable n c ∧
      indecomposable n d ∧
      r = a * b ∧
      r = c * d ∧
      (a ≠ c ∨ b ≠ d) ∧
      (a ≠ d ∨ b ≠ c) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_multiple_representations_l475_47537


namespace NUMINAMATH_CALUDE_equilateral_triangle_product_l475_47583

/-- Given that (0, 0), (a, 8), and (b, 20) form an equilateral triangle,
    prove that ab = 320/3 -/
theorem equilateral_triangle_product (a b : ℝ) : 
  (∃ (θ : ℝ), θ = π/3 ∨ θ = -π/3) →
  (Complex.abs (Complex.I * 8 - 0) = Complex.abs (b + Complex.I * 20 - 0)) →
  (Complex.abs (b + Complex.I * 20 - (a + Complex.I * 8)) = Complex.abs (Complex.I * 8 - 0)) →
  (b + Complex.I * 20 = (a + Complex.I * 8) * Complex.exp (Complex.I * θ)) →
  a * b = 320 / 3 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_product_l475_47583


namespace NUMINAMATH_CALUDE_smallest_stable_triangle_side_l475_47578

/-- A stable triangle is a scalene triangle with positive integer side lengths that are multiples of 5, 80, and 112 respectively. -/
def StableTriangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧  -- scalene
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive
  ∃ (x y z : ℕ), a = 5 * x ∧ b = 80 * y ∧ c = 112 * z  -- multiples of 5, 80, 112

/-- The smallest possible side length in any stable triangle is 20. -/
theorem smallest_stable_triangle_side : 
  (∃ (a b c : ℕ), StableTriangle a b c) → 
  (∀ (a b c : ℕ), StableTriangle a b c → min a (min b c) ≥ 20) ∧
  (∃ (a b c : ℕ), StableTriangle a b c ∧ min a (min b c) = 20) :=
sorry

end NUMINAMATH_CALUDE_smallest_stable_triangle_side_l475_47578


namespace NUMINAMATH_CALUDE_cricket_innings_count_l475_47554

-- Define the problem parameters
def current_average : ℝ := 32
def runs_next_innings : ℝ := 116
def average_increase : ℝ := 4

-- Theorem statement
theorem cricket_innings_count :
  ∀ n : ℝ,
  (n > 0) →
  (current_average * n + runs_next_innings) / (n + 1) = current_average + average_increase →
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_cricket_innings_count_l475_47554


namespace NUMINAMATH_CALUDE_rhombus_properties_l475_47597

/-- Given a rhombus with diagonals of 18 inches and 24 inches, this theorem proves:
    1. The perimeter of the rhombus is 60 inches.
    2. The area of a triangle formed by one side of the rhombus and half of each diagonal is 67.5 square inches. -/
theorem rhombus_properties (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 24) :
  let s := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  (4 * s = 60) ∧ ((s * (d1 / 2)) / 2 = 67.5) :=
by sorry

end NUMINAMATH_CALUDE_rhombus_properties_l475_47597


namespace NUMINAMATH_CALUDE_cubic_function_root_sum_squares_l475_47572

/-- Given f(x) = x³ - 2x² - 3x + 4, if there exist distinct a, b, c such that f(a) = f(b) = f(c),
    then a² + b² + c² = 10 -/
theorem cubic_function_root_sum_squares (f : ℝ → ℝ) (a b c : ℝ) :
  f = (λ x => x^3 - 2*x^2 - 3*x + 4) →
  a < b →
  b < c →
  f a = f b →
  f b = f c →
  a^2 + b^2 + c^2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_root_sum_squares_l475_47572


namespace NUMINAMATH_CALUDE_farm_has_55_cows_l475_47591

/-- Given information about husk consumption by cows on a dairy farm -/
structure DairyFarm where
  totalBags : ℕ -- Total bags of husk consumed by the group
  totalDays : ℕ -- Total days for group consumption
  singleCowDays : ℕ -- Days for one cow to consume one bag

/-- Calculate the number of cows on the farm -/
def numberOfCows (farm : DairyFarm) : ℕ :=
  farm.totalBags * farm.singleCowDays / farm.totalDays

/-- Theorem stating that the number of cows is 55 under given conditions -/
theorem farm_has_55_cows (farm : DairyFarm)
  (h1 : farm.totalBags = 55)
  (h2 : farm.totalDays = 55)
  (h3 : farm.singleCowDays = 55) :
  numberOfCows farm = 55 := by
  sorry

end NUMINAMATH_CALUDE_farm_has_55_cows_l475_47591


namespace NUMINAMATH_CALUDE_unicorn_journey_flowers_l475_47541

/-- Calculates the number of flowers that bloom when unicorns walk across a forest -/
def unicorn_flowers (num_unicorns : ℕ) (journey_km : ℕ) (step_meters : ℕ) (flowers_per_step : ℕ) : ℕ :=
  let journey_meters := journey_km * 1000
  let num_steps := journey_meters / step_meters
  let flowers_per_unicorn := num_steps * flowers_per_step
  num_unicorns * flowers_per_unicorn

/-- Theorem stating that 6 unicorns walking 9 km with 3-meter steps, each causing 4 flowers to bloom, results in 72000 flowers -/
theorem unicorn_journey_flowers :
  unicorn_flowers 6 9 3 4 = 72000 := by
  sorry

end NUMINAMATH_CALUDE_unicorn_journey_flowers_l475_47541


namespace NUMINAMATH_CALUDE_product_sale_loss_l475_47585

/-- Represents the pricing and sale of a product -/
def ProductSale (cost_price : ℝ) : Prop :=
  let initial_markup := 1.20
  let price_reduction := 0.80
  let sale_price := 96
  initial_markup * cost_price * price_reduction = sale_price ∧
  cost_price > sale_price ∧
  cost_price - sale_price = 4

/-- Theorem stating the loss in the product sale -/
theorem product_sale_loss :
  ∃ (cost_price : ℝ), ProductSale cost_price :=
sorry

end NUMINAMATH_CALUDE_product_sale_loss_l475_47585


namespace NUMINAMATH_CALUDE_factorization_condition_l475_47571

def is_factorizable (m : ℤ) : Prop :=
  ∃ (a b c d e f : ℤ),
    ∀ (x y : ℤ),
      x^2 + 3*x*y + x + m*y - m = (a*x + b*y + c) * (d*x + e*y + f)

theorem factorization_condition (m : ℤ) :
  is_factorizable m ↔ (m = 0 ∨ m = 12) :=
sorry

end NUMINAMATH_CALUDE_factorization_condition_l475_47571


namespace NUMINAMATH_CALUDE_range_of_s_is_composite_positive_integers_l475_47520

-- Define the set of composite positive integers
def CompositePositiveIntegers : Set ℕ := {n : ℕ | n > 1 ∧ ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b}

-- Define the function s
def s (n : ℕ) : ℕ := n

-- State the theorem
theorem range_of_s_is_composite_positive_integers :
  {s n | n ∈ CompositePositiveIntegers} = CompositePositiveIntegers := by
  sorry


end NUMINAMATH_CALUDE_range_of_s_is_composite_positive_integers_l475_47520


namespace NUMINAMATH_CALUDE_outfit_combinations_l475_47506

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) : 
  shirts = 3 → pants = 4 → shirts * pants = 12 :=
by sorry

end NUMINAMATH_CALUDE_outfit_combinations_l475_47506


namespace NUMINAMATH_CALUDE_binomial_20_19_l475_47504

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_19_l475_47504


namespace NUMINAMATH_CALUDE_function_positivity_implies_ab_bound_l475_47522

/-- Given a function f(x) = (x - 1/x - a)(x - b), if f(x) > 0 for all x > 0, then ab > -1 -/
theorem function_positivity_implies_ab_bound (a b : ℝ) : 
  (∀ x > 0, (x - 1/x - a) * (x - b) > 0) → a * b > -1 := by
  sorry

end NUMINAMATH_CALUDE_function_positivity_implies_ab_bound_l475_47522


namespace NUMINAMATH_CALUDE_ice_cube_freeze_time_l475_47515

/-- The time in minutes to turn frozen ice cubes into one smoothie -/
def time_per_smoothie : ℕ := 3

/-- The total number of smoothies made -/
def num_smoothies : ℕ := 5

/-- The total time in minutes to make all smoothies, including freezing ice cubes -/
def total_time : ℕ := 55

/-- The time in minutes to freeze ice cubes -/
def freeze_time : ℕ := total_time - (time_per_smoothie * num_smoothies)

theorem ice_cube_freeze_time :
  freeze_time = 40 := by sorry

end NUMINAMATH_CALUDE_ice_cube_freeze_time_l475_47515


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l475_47553

theorem average_of_three_numbers (A B C : ℝ) 
  (sum_AB : A + B = 147)
  (sum_BC : B + C = 123)
  (sum_AC : A + C = 132) :
  (A + B + C) / 3 = 67 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l475_47553


namespace NUMINAMATH_CALUDE_sugar_left_l475_47516

theorem sugar_left (bought spilled : ℝ) (h1 : bought = 9.8) (h2 : spilled = 5.2) :
  bought - spilled = 4.6 := by
  sorry

end NUMINAMATH_CALUDE_sugar_left_l475_47516


namespace NUMINAMATH_CALUDE_jordan_running_time_l475_47566

/-- Given that Jordan ran 4 miles in one-third the time it took Steve to run 6 miles,
    and Steve took 36 minutes to run 6 miles, prove that Jordan would take 21 minutes
    to run 7 miles. -/
theorem jordan_running_time
  (steve_time : ℝ)
  (steve_distance : ℝ)
  (jordan_distance : ℝ)
  (jordan_time_fraction : ℝ)
  (jordan_new_distance : ℝ)
  (h1 : steve_time = 36)
  (h2 : steve_distance = 6)
  (h3 : jordan_distance = 4)
  (h4 : jordan_time_fraction = 1 / 3)
  (h5 : jordan_new_distance = 7)
  : (jordan_new_distance * jordan_time_fraction * steve_time) / jordan_distance = 21 := by
  sorry

#check jordan_running_time

end NUMINAMATH_CALUDE_jordan_running_time_l475_47566


namespace NUMINAMATH_CALUDE_jeff_phone_storage_capacity_l475_47540

theorem jeff_phone_storage_capacity :
  let storage_used : ℕ := 4
  let song_size : ℕ := 30
  let max_songs : ℕ := 400
  let mb_per_gb : ℕ := 1000
  let total_storage : ℕ := 
    storage_used + (song_size * max_songs) / mb_per_gb
  total_storage = 16 := by
  sorry

end NUMINAMATH_CALUDE_jeff_phone_storage_capacity_l475_47540


namespace NUMINAMATH_CALUDE_tire_usage_proof_l475_47590

/-- Represents the number of miles each tire is used when seven tires are used equally over a total distance --/
def miles_per_tire (total_miles : ℕ) : ℚ :=
  (4 * total_miles : ℚ) / 7

/-- Proves that given the conditions of the problem, each tire is used for 25,714 miles --/
theorem tire_usage_proof (total_miles : ℕ) (h1 : total_miles = 45000) :
  ⌊miles_per_tire total_miles⌋ = 25714 := by
  sorry

#eval ⌊miles_per_tire 45000⌋

end NUMINAMATH_CALUDE_tire_usage_proof_l475_47590


namespace NUMINAMATH_CALUDE_product_evaluation_l475_47544

theorem product_evaluation (n : ℕ) (h : n = 3) : 
  (n - 1) * n * (n + 1) * (n + 2) * (n + 3) * (n^2 + 1) = 7200 := by
  sorry

end NUMINAMATH_CALUDE_product_evaluation_l475_47544


namespace NUMINAMATH_CALUDE_percentage_less_problem_l475_47518

theorem percentage_less_problem (C A B : ℝ) : 
  B = 0.58 * C →
  B = 0.8923076923076923 * A →
  ∃ (ε : ℝ), abs (A - 0.65 * C) < ε ∧ ε > 0 := by
sorry

end NUMINAMATH_CALUDE_percentage_less_problem_l475_47518


namespace NUMINAMATH_CALUDE_algebraic_simplification_l475_47524

theorem algebraic_simplification (a b : ℝ) : 2*a - 3*(a-b) = -a + 3*b := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l475_47524


namespace NUMINAMATH_CALUDE_calculator_sale_result_l475_47570

def calculator_transaction (price : ℝ) (profit_rate : ℝ) (loss_rate : ℝ) : Prop :=
  let profit_calculator_cost : ℝ := price / (1 + profit_rate)
  let loss_calculator_cost : ℝ := price / (1 - loss_rate)
  let total_cost : ℝ := profit_calculator_cost + loss_calculator_cost
  let total_revenue : ℝ := 2 * price
  total_revenue - total_cost = -7.5

theorem calculator_sale_result :
  calculator_transaction 90 0.2 0.2 := by
  sorry

end NUMINAMATH_CALUDE_calculator_sale_result_l475_47570


namespace NUMINAMATH_CALUDE_jackie_pushups_l475_47556

/-- Calculates the number of push-ups Jackie can do in one minute given her initial rate,
    rate of decrease, break times, and rate recovery during breaks. -/
def pushups_in_one_minute (initial_rate : ℕ) (decrease_rate : ℚ) 
                          (break_times : List ℕ) (recovery_rate : ℚ) : ℕ :=
  sorry

/-- Theorem stating that Jackie can do 15 push-ups in one minute under the given conditions. -/
theorem jackie_pushups : 
  pushups_in_one_minute 5 (1/5) [22, 38] (1/10) = 15 := by sorry

end NUMINAMATH_CALUDE_jackie_pushups_l475_47556


namespace NUMINAMATH_CALUDE_decagon_triangle_count_l475_47525

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  (n_ge_3 : n ≥ 3)

/-- A triangle formed by three vertices of a regular polygon -/
structure PolygonTriangle (n : ℕ) where
  (polygon : RegularPolygon n)
  (v1 v2 v3 : Fin n)
  (distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v1)

/-- Two triangles in a regular polygon are congruent if they have the same shape -/
def CongruentTriangles (n : ℕ) (t1 t2 : PolygonTriangle n) : Prop :=
  sorry

/-- The number of non-congruent triangles in a regular decagon -/
def NumNonCongruentTriangles (p : RegularPolygon 10) : ℕ :=
  sorry

theorem decagon_triangle_count :
  ∀ (p : RegularPolygon 10), NumNonCongruentTriangles p = 8 :=
sorry

end NUMINAMATH_CALUDE_decagon_triangle_count_l475_47525


namespace NUMINAMATH_CALUDE_batsman_average_l475_47548

/-- Calculates the new average score after an additional inning -/
def new_average (prev_avg : ℚ) (prev_innings : ℕ) (new_score : ℕ) : ℚ :=
  (prev_avg * prev_innings + new_score) / (prev_innings + 1)

/-- Theorem: Given the conditions, the batsman's new average is 18 -/
theorem batsman_average : new_average 19 17 1 = 18 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_l475_47548


namespace NUMINAMATH_CALUDE_power_equation_solution_l475_47589

theorem power_equation_solution : ∃ x : ℕ, 27^3 + 27^3 + 27^3 + 27^3 = 3^x :=
by
  use 11
  have h1 : 27 = 3^3 := by sorry
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l475_47589


namespace NUMINAMATH_CALUDE_notebook_count_l475_47509

theorem notebook_count : ∃ (N : ℕ), 
  (∃ (S : ℕ), N = 4 * S + 3) ∧ 
  (∃ (S : ℕ), N + 6 = 5 * S) ∧ 
  N = 39 := by
  sorry

end NUMINAMATH_CALUDE_notebook_count_l475_47509


namespace NUMINAMATH_CALUDE_prime_factorization_sum_l475_47513

theorem prime_factorization_sum (w x y z : ℕ) : 
  2^w * 3^x * 5^y * 7^z = 1260 → 2*w + 3*x + 5*y + 7*z = 22 := by
  sorry

end NUMINAMATH_CALUDE_prime_factorization_sum_l475_47513


namespace NUMINAMATH_CALUDE_sixth_term_of_sequence_l475_47595

/-- Given a sequence {a_n} where a_1 = 1 and a_{n+1} = a_n + 2 for n ≥ 1, prove that a_6 = 11 -/
theorem sixth_term_of_sequence (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2) : 
  a 6 = 11 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_sequence_l475_47595


namespace NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l475_47577

/-- The sum of the digits of 10^91 + 100 is 2 -/
theorem sum_of_digits_of_large_number : ∃ (n : ℕ), n = 10^91 + 100 ∧ (n.digits 10).sum = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_large_number_l475_47577


namespace NUMINAMATH_CALUDE_parabola_y_intercepts_l475_47588

/-- The number of y-intercepts of the parabola x = 3y^2 - 2y + 1 -/
theorem parabola_y_intercepts : 
  let f : ℝ → ℝ := fun y => 3 * y^2 - 2 * y + 1
  (∃ y, f y = 0) = False :=
by sorry

end NUMINAMATH_CALUDE_parabola_y_intercepts_l475_47588


namespace NUMINAMATH_CALUDE_octal_135_equals_binary_1011101_l475_47514

-- Define a function to convert octal to binary
def octal_to_binary (octal : ℕ) : ℕ := sorry

-- State the theorem
theorem octal_135_equals_binary_1011101 :
  octal_to_binary 135 = 1011101 := by sorry

end NUMINAMATH_CALUDE_octal_135_equals_binary_1011101_l475_47514


namespace NUMINAMATH_CALUDE_difference_of_squares_312_308_l475_47534

theorem difference_of_squares_312_308 : 312^2 - 308^2 = 2480 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_312_308_l475_47534


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l475_47560

theorem complex_fraction_simplification :
  (2 * Complex.I) / (1 + 2 * Complex.I) = 4/5 + (2/5) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l475_47560


namespace NUMINAMATH_CALUDE_rows_per_wall_is_fifty_l475_47559

/-- The number of bricks in a single row of each wall -/
def bricks_per_row : ℕ := 30

/-- The total number of bricks used for both walls -/
def total_bricks : ℕ := 3000

/-- The number of rows in each wall -/
def rows_per_wall : ℕ := total_bricks / (2 * bricks_per_row)

theorem rows_per_wall_is_fifty : rows_per_wall = 50 := by
  sorry

end NUMINAMATH_CALUDE_rows_per_wall_is_fifty_l475_47559


namespace NUMINAMATH_CALUDE_scientific_notation_of_twelve_million_l475_47587

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a positive real number to scientific notation -/
def to_scientific_notation (n : ℝ) : ScientificNotation :=
  sorry

/-- The number we want to convert to scientific notation -/
def target_number : ℝ := 12000000

/-- Theorem stating that the scientific notation of 12,000,000 is 1.2 × 10^7 -/
theorem scientific_notation_of_twelve_million :
  (to_scientific_notation target_number).coefficient = 1.2 ∧
  (to_scientific_notation target_number).exponent = 7 :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_twelve_million_l475_47587


namespace NUMINAMATH_CALUDE_hospital_bill_ambulance_cost_l475_47555

/-- Given a hospital bill with specified percentages for various services and fixed costs,
    calculate the cost of the ambulance ride. -/
theorem hospital_bill_ambulance_cost 
  (total_bill : ℝ) 
  (medication_percent : ℝ) 
  (imaging_percent : ℝ) 
  (surgical_percent : ℝ) 
  (overnight_percent : ℝ) 
  (food_cost : ℝ) 
  (consultation_cost : ℝ) 
  (h1 : total_bill = 12000)
  (h2 : medication_percent = 0.40)
  (h3 : imaging_percent = 0.15)
  (h4 : surgical_percent = 0.20)
  (h5 : overnight_percent = 0.25)
  (h6 : food_cost = 300)
  (h7 : consultation_cost = 80)
  (h8 : medication_percent + imaging_percent + surgical_percent + overnight_percent = 1) :
  total_bill - (food_cost + consultation_cost) = 11620 := by
  sorry


end NUMINAMATH_CALUDE_hospital_bill_ambulance_cost_l475_47555


namespace NUMINAMATH_CALUDE_product_mod_nineteen_l475_47531

theorem product_mod_nineteen : (2001 * 2002 * 2003 * 2004 * 2005) % 19 = 11 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_nineteen_l475_47531


namespace NUMINAMATH_CALUDE_correct_num_cups_l475_47581

/-- The number of cups of coffee on the tray -/
def num_cups : ℕ := 5

/-- The initial volume of coffee in each cup (in ounces) -/
def initial_volume : ℝ := 8

/-- The shrink factor of the ray -/
def shrink_factor : ℝ := 0.5

/-- The total volume of coffee after shrinking (in ounces) -/
def final_total_volume : ℝ := 20

/-- Theorem stating that the number of cups is correct given the conditions -/
theorem correct_num_cups :
  initial_volume * shrink_factor * num_cups = final_total_volume :=
by sorry

end NUMINAMATH_CALUDE_correct_num_cups_l475_47581


namespace NUMINAMATH_CALUDE_sqrt_720_simplification_l475_47564

theorem sqrt_720_simplification : Real.sqrt 720 = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_720_simplification_l475_47564


namespace NUMINAMATH_CALUDE_total_people_in_program_l475_47562

theorem total_people_in_program (parents pupils : ℕ) 
  (h1 : parents = 105) 
  (h2 : pupils = 698) : 
  parents + pupils = 803 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l475_47562


namespace NUMINAMATH_CALUDE_boat_speed_current_l475_47538

/-- Proves that given a boat with a constant speed of 16 mph relative to water,
    making an upstream trip in 20 minutes and a downstream trip in 15 minutes,
    the speed of the current is 16/7 mph. -/
theorem boat_speed_current (boat_speed : ℝ) (upstream_time : ℝ) (downstream_time : ℝ) :
  boat_speed = 16 ∧ upstream_time = 20 / 60 ∧ downstream_time = 15 / 60 →
  ∃ current_speed : ℝ,
    (boat_speed - current_speed) * upstream_time = (boat_speed + current_speed) * downstream_time ∧
    current_speed = 16 / 7 :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_current_l475_47538


namespace NUMINAMATH_CALUDE_infinite_series_sum_l475_47563

theorem infinite_series_sum : 
  (∑' n : ℕ, (n : ℝ) / (5 ^ n)) = 5 / 16 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l475_47563


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_inequality_system_l475_47557

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := 2 * x + 3 ≤ 5 * x
def inequality2 (x : ℝ) : Prop := 5 * x - 1 ≤ 3 * (x + 1)
def inequality3 (x : ℝ) : Prop := (2 * x - 1) / 2 - (5 * x - 1) / 4 < 1

-- Theorem for the first inequality
theorem solution_inequality1 :
  {x : ℝ | inequality1 x} = {x : ℝ | x ≥ 1} :=
sorry

-- Theorem for the system of inequalities
theorem solution_inequality_system :
  {x : ℝ | inequality2 x ∧ inequality3 x} = {x : ℝ | -5 < x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_inequality_system_l475_47557


namespace NUMINAMATH_CALUDE_camping_matches_l475_47575

def matches_left (initial : ℕ) (dropped : ℕ) : ℕ :=
  initial - dropped - 2 * dropped

theorem camping_matches (initial : ℕ) (dropped : ℕ) 
  (h1 : initial ≥ dropped) 
  (h2 : initial ≥ dropped + 2 * dropped) :
  matches_left initial dropped = initial - dropped - 2 * dropped :=
by sorry

end NUMINAMATH_CALUDE_camping_matches_l475_47575


namespace NUMINAMATH_CALUDE_complex_equation_ratio_l475_47599

theorem complex_equation_ratio (a b : ℝ) : 
  (a - 2*Complex.I)*Complex.I = b + a*Complex.I → a/b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_ratio_l475_47599


namespace NUMINAMATH_CALUDE_parabola_parameter_l475_47596

/-- Theorem: For a parabola y^2 = 2px (p > 0) with focus F, if a line through F makes an angle of π/3
    with the x-axis and intersects the parabola at points A and B with |AB| = 8, then p = 3. -/
theorem parabola_parameter (p : ℝ) (A B : ℝ × ℝ) :
  p > 0 →
  (∀ x y, y^2 = 2*p*x) →
  (∃ m b, ∀ x y, y = m*x + b ∧ m = Real.sqrt 3) →
  (∀ x y, y^2 = 2*p*x → (∃ t, x = t ∧ y = Real.sqrt 3 * (t - p/2))) →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 →
  p = 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_parameter_l475_47596


namespace NUMINAMATH_CALUDE_rainfall_problem_l475_47507

/-- Rainfall problem -/
theorem rainfall_problem (total_time : ℕ) (total_rainfall : ℕ) 
  (storm1_rate : ℕ) (storm1_duration : ℕ) :
  total_time = 45 →
  total_rainfall = 975 →
  storm1_rate = 30 →
  storm1_duration = 20 →
  ∃ storm2_rate : ℕ, 
    storm2_rate * (total_time - storm1_duration) = 
      total_rainfall - (storm1_rate * storm1_duration) ∧
    storm2_rate = 15 := by
  sorry


end NUMINAMATH_CALUDE_rainfall_problem_l475_47507


namespace NUMINAMATH_CALUDE_reflect_P_across_y_axis_l475_47552

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that reflecting P(1, -2) across the y-axis results in (-1, -2) -/
theorem reflect_P_across_y_axis :
  let P : Point := { x := 1, y := -2 }
  reflectAcrossYAxis P = { x := -1, y := -2 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_across_y_axis_l475_47552


namespace NUMINAMATH_CALUDE_curve_crosses_at_point_l475_47503

/-- A curve in the xy-plane defined by parametric equations -/
def curve (t : ℝ) : ℝ × ℝ :=
  (3 * t^2 + 1, t^3 - 6 * t^2 + 4)

/-- The point where the curve crosses itself -/
def crossing_point : ℝ × ℝ := (109, -428)

/-- Theorem stating that the curve crosses itself at the specified point -/
theorem curve_crosses_at_point :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ curve t₁ = curve t₂ ∧ curve t₁ = crossing_point :=
sorry

end NUMINAMATH_CALUDE_curve_crosses_at_point_l475_47503


namespace NUMINAMATH_CALUDE_bookstore_shipment_count_l475_47526

theorem bookstore_shipment_count :
  ∀ (total : ℕ) (displayed : ℕ) (stockroom : ℕ),
    displayed = (30 : ℕ) * total / 100 →
    stockroom = (70 : ℕ) * total / 100 →
    stockroom = 140 →
    total = 200 := by
  sorry

end NUMINAMATH_CALUDE_bookstore_shipment_count_l475_47526


namespace NUMINAMATH_CALUDE_area_of_special_quadrilateral_in_cube_l475_47543

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube in 3D space -/
structure Cube where
  sideLength : ℝ

/-- Represents a quadrilateral in 3D space -/
structure Quadrilateral where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D

/-- Calculate the area of a quadrilateral given its vertices -/
def areaOfQuadrilateral (q : Quadrilateral) : ℝ := sorry

/-- Check if a point is a vertex of a cube -/
def isVertexOfCube (p : Point3D) (cube : Cube) : Prop := sorry

/-- Check if a point is a midpoint of an edge of a cube -/
def isMidpointOfCubeEdge (p : Point3D) (cube : Cube) : Prop := sorry

/-- Check if two points are diagonally opposite vertices of a cube -/
def areDiagonallyOppositeVertices (p1 p2 : Point3D) (cube : Cube) : Prop := sorry

/-- Main theorem -/
theorem area_of_special_quadrilateral_in_cube (cube : Cube) (a b c d : Point3D) :
  cube.sideLength = 2 →
  isVertexOfCube a cube →
  isVertexOfCube c cube →
  isMidpointOfCubeEdge b cube →
  isMidpointOfCubeEdge d cube →
  areDiagonallyOppositeVertices a c cube →
  areaOfQuadrilateral ⟨a, b, c, d⟩ = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_quadrilateral_in_cube_l475_47543


namespace NUMINAMATH_CALUDE_seven_story_pagoda_top_lanterns_l475_47542

/-- Represents a pagoda with a given number of stories and lanterns -/
structure Pagoda where
  stories : ℕ
  total_lanterns : ℕ
  lanterns_ratio : ℕ -- ratio of lanterns between adjacent stories

/-- Calculates the number of lanterns on the top story of a pagoda -/
def top_story_lanterns (p : Pagoda) : ℕ :=
  sorry

/-- Theorem: For a 7-story pagoda with a lantern ratio of 2 and 381 total lanterns,
    the number of lanterns on the top story is 3 -/
theorem seven_story_pagoda_top_lanterns :
  let p : Pagoda := { stories := 7, total_lanterns := 381, lanterns_ratio := 2 }
  top_story_lanterns p = 3 := by
  sorry

end NUMINAMATH_CALUDE_seven_story_pagoda_top_lanterns_l475_47542


namespace NUMINAMATH_CALUDE_circle_chord_length_l475_47558

theorem circle_chord_length (AB CD : ℝ) (h1 : AB = 13) (h2 : CD = 6) :
  let AD := (x : ℝ)
  (x = 4 ∨ x = 9) ↔ x^2 - AB*x + CD^2 = 0 := by
sorry

end NUMINAMATH_CALUDE_circle_chord_length_l475_47558


namespace NUMINAMATH_CALUDE_min_value_theorem_l475_47592

theorem min_value_theorem (x y : ℝ) (h1 : x * y + 3 * x = 3) (h2 : 0 < x) (h3 : x < 1/2) :
  (3 / x) + (1 / (y - 3)) ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ * y₀ + 3 * x₀ = 3 ∧ 0 < x₀ ∧ x₀ < 1/2 ∧ (3 / x₀) + (1 / (y₀ - 3)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l475_47592


namespace NUMINAMATH_CALUDE_max_value_of_f_l475_47546

noncomputable def f (x : ℝ) : ℝ := (-x^2 + x - 4) / x

theorem max_value_of_f :
  ∃ (x_max : ℝ), x_max > 0 ∧
  (∀ (x : ℝ), x > 0 → f x ≤ f x_max) ∧
  f x_max = -3 ∧
  x_max = 2 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l475_47546


namespace NUMINAMATH_CALUDE_fibonacci_fifth_is_s_plus_one_l475_47593

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def s : ℕ := 4

theorem fibonacci_fifth_is_s_plus_one :
  fibonacci 5 = s + 1 ∧ ∀ k < 5, fibonacci k ≠ s + 1 := by sorry

end NUMINAMATH_CALUDE_fibonacci_fifth_is_s_plus_one_l475_47593


namespace NUMINAMATH_CALUDE_rectangle_area_unchanged_l475_47584

theorem rectangle_area_unchanged (A l w : ℝ) (h1 : A = l * w) (h2 : A > 0) :
  let l' := 0.8 * l
  let w' := 1.25 * w
  l' * w' = A := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_unchanged_l475_47584


namespace NUMINAMATH_CALUDE_solve_for_y_l475_47551

theorem solve_for_y (x : ℝ) (y : ℝ) 
  (h1 : x = 101) 
  (h2 : x^3 * y - 2 * x^2 * y + x * y = 101000) : 
  y = 1/10 := by
sorry

end NUMINAMATH_CALUDE_solve_for_y_l475_47551


namespace NUMINAMATH_CALUDE_sqrt_expressions_equality_l475_47547

theorem sqrt_expressions_equality :
  (Real.sqrt 75 - Real.sqrt 54 + Real.sqrt 96 - Real.sqrt 108 = -Real.sqrt 3 + Real.sqrt 6) ∧
  (Real.sqrt 24 / Real.sqrt 3 + Real.sqrt (1/2) * Real.sqrt 18 - Real.sqrt 50 = 3 - 3 * Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_expressions_equality_l475_47547


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l475_47532

theorem simplify_complex_fraction (x : ℝ) 
  (h1 : x ≠ 3) (h2 : x ≠ 4) (h3 : x ≠ 5) : 
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  ((x - 1) * (x - 5)) / ((x - 3) * (x - 4) * (x - 2)) := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l475_47532


namespace NUMINAMATH_CALUDE_stratified_sampling_car_inspection_l475_47567

theorem stratified_sampling_car_inspection
  (total_sample : ℕ)
  (type_a_production type_b_production type_c_production : ℕ)
  (h_total_sample : total_sample = 47)
  (h_type_a : type_a_production = 1400)
  (h_type_b : type_b_production = 6000)
  (h_type_c : type_c_production = 2000) :
  ∃ (sample_a sample_b sample_c : ℕ),
    sample_a + sample_b + sample_c = total_sample ∧
    sample_a = 7 ∧
    sample_b = 30 ∧
    sample_c = 10 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_car_inspection_l475_47567


namespace NUMINAMATH_CALUDE_solve_equation_l475_47550

-- Define the function F
def F (a b c : ℚ) : ℚ := a * b^3 + c

-- Theorem statement
theorem solve_equation :
  ∃ a : ℚ, F a 3 8 = F a 5 12 ∧ a = -2/49 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l475_47550


namespace NUMINAMATH_CALUDE_initial_birds_count_l475_47521

theorem initial_birds_count (initial_birds additional_birds total_birds : ℕ) 
  (h1 : additional_birds = 13)
  (h2 : total_birds = 42)
  (h3 : initial_birds + additional_birds = total_birds) : 
  initial_birds = 29 := by
  sorry

end NUMINAMATH_CALUDE_initial_birds_count_l475_47521


namespace NUMINAMATH_CALUDE_goldfish_equality_l475_47586

/-- The number of months after which Alice and Bob have the same number of goldfish -/
def same_goldfish_month : ℕ := 7

/-- Alice's initial number of goldfish -/
def alice_initial : ℕ := 3

/-- Bob's initial number of goldfish -/
def bob_initial : ℕ := 256

/-- Alice's goldfish growth rate per month -/
def alice_growth_rate : ℕ := 3

/-- Bob's goldfish growth rate per month -/
def bob_growth_rate : ℕ := 4

/-- Alice's number of goldfish after n months -/
def alice_goldfish (n : ℕ) : ℕ := alice_initial * (alice_growth_rate ^ n)

/-- Bob's number of goldfish after n months -/
def bob_goldfish (n : ℕ) : ℕ := bob_initial * (bob_growth_rate ^ n)

theorem goldfish_equality :
  alice_goldfish same_goldfish_month = bob_goldfish same_goldfish_month ∧
  ∀ m : ℕ, m < same_goldfish_month → alice_goldfish m ≠ bob_goldfish m :=
by sorry

end NUMINAMATH_CALUDE_goldfish_equality_l475_47586


namespace NUMINAMATH_CALUDE_intersection_point_product_range_l475_47561

theorem intersection_point_product_range (k x₀ y₀ : ℝ) :
  x₀ + y₀ = 2 * k - 1 →
  x₀^2 + y₀^2 = k^2 + 2 * k - 3 →
  (11 - 6 * Real.sqrt 2) / 4 ≤ x₀ * y₀ ∧ x₀ * y₀ ≤ (11 + 6 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_product_range_l475_47561


namespace NUMINAMATH_CALUDE_Egypt_India_traditional_l475_47508

-- Define the types of countries and population growth patterns
inductive CountryType
| Developed
| Developing

inductive GrowthPattern
| Traditional
| Modern

-- Define a function to determine the growth pattern based on country type
def typicalGrowthPattern (ct : CountryType) : GrowthPattern :=
  match ct with
  | CountryType.Developed => GrowthPattern.Modern
  | CountryType.Developing => GrowthPattern.Traditional

-- Define specific countries
def Egypt : CountryType := CountryType.Developing
def India : CountryType := CountryType.Developing

-- China is an exception
def China : CountryType := CountryType.Developing
axiom China_exception : typicalGrowthPattern China = GrowthPattern.Modern

-- Theorem to prove
theorem Egypt_India_traditional :
  typicalGrowthPattern Egypt = GrowthPattern.Traditional ∧
  typicalGrowthPattern India = GrowthPattern.Traditional :=
sorry

end NUMINAMATH_CALUDE_Egypt_India_traditional_l475_47508


namespace NUMINAMATH_CALUDE_consecutive_pages_sum_l475_47569

theorem consecutive_pages_sum (n : ℕ) : 
  n * (n + 1) * (n + 2) = 479160 → n + (n + 1) + (n + 2) = 234 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_pages_sum_l475_47569


namespace NUMINAMATH_CALUDE_power_sum_sequence_l475_47527

/-- Given real numbers a and b satisfying certain conditions, prove that a^10 + b^10 = 123 -/
theorem power_sum_sequence (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry


end NUMINAMATH_CALUDE_power_sum_sequence_l475_47527


namespace NUMINAMATH_CALUDE_score_distribution_theorem_l475_47530

/-- Represents the frequency distribution of student scores -/
structure FrequencyDistribution :=
  (f65_70 f70_75 f75_80 f80_85 f85_90 f90_95 f95_100 : ℚ)

/-- Represents the problem setup -/
structure ProblemSetup :=
  (fd : FrequencyDistribution)
  (total_students : ℕ)
  (students_80_90 : ℕ)
  (prob_male_95_100 : ℚ)
  (female_65_70 : ℕ)

/-- The main theorem to prove -/
theorem score_distribution_theorem (setup : ProblemSetup) :
  (setup.total_students * setup.fd.f95_100 = 6) ∧
  (∃ (m : ℕ), m = 2 ∧ m ≤ 6 ∧ 
    (m * (m - 1) / 30 + m * (6 - m) / 15 : ℚ) = 3/5) ∧
  (∃ (p0 p1 p2 : ℚ), p0 + p1 + p2 = 1 ∧
    p0 * 0 + p1 * 1 + p2 * 2 = 1) :=
by sorry

/-- Assumptions about the problem setup -/
axiom setup_valid (setup : ProblemSetup) :
  setup.fd.f65_70 = 1/10 ∧
  setup.fd.f70_75 = 3/20 ∧
  setup.fd.f75_80 = 1/5 ∧
  setup.fd.f80_85 = 1/5 ∧
  setup.fd.f85_90 = 3/20 ∧
  setup.fd.f90_95 = 1/10 ∧
  setup.fd.f95_100 + setup.fd.f65_70 + setup.fd.f70_75 + setup.fd.f75_80 +
    setup.fd.f80_85 + setup.fd.f85_90 + setup.fd.f90_95 = 1 ∧
  setup.students_80_90 = 21 ∧
  setup.prob_male_95_100 = 3/5 ∧
  setup.female_65_70 = 4 ∧
  setup.total_students * (setup.fd.f80_85 + setup.fd.f85_90) = setup.students_80_90

end NUMINAMATH_CALUDE_score_distribution_theorem_l475_47530


namespace NUMINAMATH_CALUDE_max_passable_levels_l475_47594

/-- Represents the maximum number of points obtainable from a single dice throw -/
def max_dice_points : ℕ := 6

/-- Represents the pass condition for a level in the "pass-through game" -/
def pass_condition (n : ℕ) : ℕ := 2^n

/-- Represents the maximum sum of points obtainable from n dice throws -/
def max_sum_points (n : ℕ) : ℕ := n * max_dice_points

/-- Theorem stating the maximum number of levels that can be passed in the "pass-through game" -/
theorem max_passable_levels : 
  ∃ (max_level : ℕ), 
    (∀ n : ℕ, n ≤ max_level → max_sum_points n > pass_condition n) ∧ 
    (∀ n : ℕ, n > max_level → max_sum_points n ≤ pass_condition n) :=
sorry

end NUMINAMATH_CALUDE_max_passable_levels_l475_47594


namespace NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l475_47580

-- Problem 1
theorem simplify_expression (x y : ℝ) : 
  x - (2 * x - y) + (3 * x - 2 * y) = 2 * x - y := by sorry

-- Problem 2
theorem evaluate_expression : 
  -(1^4) + |3 - 5| - 8 + (-2) * (1/2) = -8 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_evaluate_expression_l475_47580


namespace NUMINAMATH_CALUDE_sector_angle_measure_l475_47565

/-- Given a circular sector with radius 10 and area 50π/3, 
    prove that its central angle measures π/3 radians. -/
theorem sector_angle_measure (r : ℝ) (S : ℝ) (α : ℝ) 
  (h_radius : r = 10)
  (h_area : S = 50 * Real.pi / 3)
  (h_sector_area : S = 1/2 * r^2 * α) :
  α = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_angle_measure_l475_47565


namespace NUMINAMATH_CALUDE_principal_amount_proof_l475_47501

/-- Proves that for a principal amount P, with an interest rate of 5% per annum over 2 years,
    if the difference between compound interest and simple interest is 17, then P equals 6800. -/
theorem principal_amount_proof (P : ℝ) : 
  P * (1 + 0.05)^2 - P - (P * 0.05 * 2) = 17 → P = 6800 := by
  sorry

end NUMINAMATH_CALUDE_principal_amount_proof_l475_47501


namespace NUMINAMATH_CALUDE_hot_dog_stand_sales_l475_47576

/-- A hot dog stand problem -/
theorem hot_dog_stand_sales 
  (price : ℝ) 
  (hours : ℝ) 
  (total_sales : ℝ) 
  (h1 : price = 2)
  (h2 : hours = 10)
  (h3 : total_sales = 200) :
  total_sales / (hours * price) = 10 :=
sorry

end NUMINAMATH_CALUDE_hot_dog_stand_sales_l475_47576


namespace NUMINAMATH_CALUDE_students_in_all_activities_l475_47533

theorem students_in_all_activities (total : ℕ) (chess : ℕ) (music : ℕ) (art : ℕ) (at_least_two : ℕ) :
  total = 25 →
  chess = 12 →
  music = 15 →
  art = 11 →
  at_least_two = 11 →
  ∃ (only_chess only_music only_art chess_music chess_art music_art all_three : ℕ),
    only_chess + only_music + only_art + chess_music + chess_art + music_art + all_three = total ∧
    only_chess + chess_music + chess_art + all_three = chess ∧
    only_music + chess_music + music_art + all_three = music ∧
    only_art + chess_art + music_art + all_three = art ∧
    chess_music + chess_art + music_art + all_three = at_least_two ∧
    all_three = 4 :=
by sorry

end NUMINAMATH_CALUDE_students_in_all_activities_l475_47533


namespace NUMINAMATH_CALUDE_prism_with_12_edges_has_quadrilateral_base_l475_47536

/-- A prism with n sides in its base has 3n edges. -/
def prism_edges (n : ℕ) : ℕ := 3 * n

/-- The number of sides in the base of a prism with 12 edges. -/
def base_sides : ℕ := 4

/-- Theorem: A prism with 12 edges has a quadrilateral base. -/
theorem prism_with_12_edges_has_quadrilateral_base :
  prism_edges base_sides = 12 :=
sorry

end NUMINAMATH_CALUDE_prism_with_12_edges_has_quadrilateral_base_l475_47536


namespace NUMINAMATH_CALUDE_a_14_mod_7_l475_47510

/-- Sequence defined recursively -/
def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1  -- We assume a₁ = 1 based on the solution
  | 2 => 2
  | (n + 3) => a (n + 1) + (a (n + 2))^2

/-- The 14th term of the sequence is congruent to 5 modulo 7 -/
theorem a_14_mod_7 : a 14 ≡ 5 [ZMOD 7] := by
  sorry

end NUMINAMATH_CALUDE_a_14_mod_7_l475_47510


namespace NUMINAMATH_CALUDE_new_rope_length_l475_47502

/-- Proves that given an initial rope length of 12 m and an additional grazing area of 565.7142857142857 m², 
    the new rope length that allows this additional grazing area is 18 m. -/
theorem new_rope_length 
  (initial_length : ℝ) 
  (additional_area : ℝ) 
  (h1 : initial_length = 12)
  (h2 : additional_area = 565.7142857142857) : 
  ∃ (new_length : ℝ), 
    new_length = 18 ∧ 
    π * new_length ^ 2 = π * initial_length ^ 2 + additional_area :=
by sorry

end NUMINAMATH_CALUDE_new_rope_length_l475_47502


namespace NUMINAMATH_CALUDE_prank_combinations_l475_47500

/-- The number of choices for each day of the week --/
def monday_choices : ℕ := 1
def tuesday_choices : ℕ := 4
def wednesday_choices : ℕ := 7
def thursday_choices : ℕ := 5
def friday_choices : ℕ := 1

/-- The total number of combinations --/
def total_combinations : ℕ := monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices

/-- Theorem stating that the total number of combinations is 140 --/
theorem prank_combinations : total_combinations = 140 := by
  sorry

end NUMINAMATH_CALUDE_prank_combinations_l475_47500


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_a_greater_than_neg_five_l475_47512

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 6}
def B (a : ℝ) : Set ℝ := {x | 1 - a < x}

-- State the theorem
theorem intersection_nonempty_iff_a_greater_than_neg_five (a : ℝ) :
  (A ∩ B a).Nonempty ↔ a > -5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_a_greater_than_neg_five_l475_47512


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l475_47582

theorem price_reduction_percentage (last_year_price : ℝ) : 
  let this_year_price := last_year_price * (1 + 0.25)
  let next_year_target := last_year_price * (1 + 0.10)
  ∃ (reduction_percentage : ℝ), 
    this_year_price * (1 - reduction_percentage) = next_year_target ∧ 
    reduction_percentage = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l475_47582


namespace NUMINAMATH_CALUDE_function_extrema_l475_47598

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + a*x - (a+1) * Real.log x

theorem function_extrema (a : ℝ) (h1 : a < -1) :
  (∀ x : ℝ, x > 0 → (deriv (f a)) 2 = 0) →
  (a = -3 ∧ 
   (∀ x : ℝ, x > 0 → f a x ≤ f a 1) ∧
   (∀ x : ℝ, x > 0 → f a x ≥ f a 2) ∧
   f a 1 = -5/2 ∧
   f a 2 = -4 + 2 * Real.log 2) := by
  sorry

end

end NUMINAMATH_CALUDE_function_extrema_l475_47598


namespace NUMINAMATH_CALUDE_complex_equation_solution_l475_47529

theorem complex_equation_solution :
  ∀ z : ℂ, z / (1 + Complex.I) = Complex.I ^ 2015 + Complex.I ^ 2016 → z = -2 * Complex.I :=
by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l475_47529


namespace NUMINAMATH_CALUDE_cookie_distribution_l475_47511

theorem cookie_distribution (total_cookies : ℕ) (num_adults : ℕ) (num_children : ℕ) 
  (h1 : total_cookies = 120)
  (h2 : num_adults = 2)
  (h3 : num_children = 4) :
  (total_cookies - (total_cookies / 3)) / num_children = 20 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_l475_47511


namespace NUMINAMATH_CALUDE_accidental_calculation_l475_47539

theorem accidental_calculation (x : ℝ) : (x + 12) / 8 = 8 → (x - 12) * 9 = 360 := by
  sorry

end NUMINAMATH_CALUDE_accidental_calculation_l475_47539


namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l475_47545

theorem larger_number_in_ratio (a b : ℕ+) : 
  (a : ℚ) / b = 2 / 3 →
  Nat.lcm a b = 120 →
  b = 72 := by
sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l475_47545


namespace NUMINAMATH_CALUDE_nested_cube_root_l475_47574

theorem nested_cube_root (M : ℝ) (h : M > 1) :
  (M * (M * (M * M^(1/3))^(1/3))^(1/3))^(1/3) = M^(40/81) := by
  sorry

end NUMINAMATH_CALUDE_nested_cube_root_l475_47574


namespace NUMINAMATH_CALUDE_num_positive_divisors_1386_l475_47517

/-- The number of positive divisors of a natural number -/
def numPositiveDivisors (n : ℕ) : ℕ := sorry

/-- Theorem: The number of positive divisors of 1386 is 24 -/
theorem num_positive_divisors_1386 : numPositiveDivisors 1386 = 24 := by sorry

end NUMINAMATH_CALUDE_num_positive_divisors_1386_l475_47517


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l475_47535

/-- Proves that a train 100 meters long, traveling at 72 km/hr, takes 5 seconds to pass a pole -/
theorem train_passing_pole_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 100 ∧ train_speed_kmh = 72 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 5 := by
sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l475_47535


namespace NUMINAMATH_CALUDE_smallest_positive_angle_l475_47573

theorem smallest_positive_angle (y : Real) : 
  (4 * Real.sin y * (Real.cos y)^3 - 4 * (Real.sin y)^3 * Real.cos y = Real.cos y) →
  (y > 0) →
  (∀ z, z > 0 ∧ 4 * Real.sin z * (Real.cos z)^3 - 4 * (Real.sin z)^3 * Real.cos z = Real.cos z → y ≤ z) →
  y = 18 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_l475_47573


namespace NUMINAMATH_CALUDE_train_speed_l475_47528

/-- The speed of a train given its length, time to cross a bridge, and total length of train and bridge -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (total_length : ℝ) :
  train_length = 130 →
  crossing_time = 30 →
  total_length = 245 →
  (total_length / crossing_time) * 3.6 = 29.4 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l475_47528


namespace NUMINAMATH_CALUDE_area_intersection_approx_l475_47549

/-- The elliptical region D₁ -/
def D₁ (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 ≤ 1

/-- The circular region D₂ -/
def D₂ (x y : ℝ) : Prop := x^2 + y^2 ≤ 2

/-- The intersection of D₁ and D₂ -/
def D_intersection (x y : ℝ) : Prop := D₁ x y ∧ D₂ x y

/-- The area of the intersection of D₁ and D₂ -/
noncomputable def area_intersection : ℝ := sorry

theorem area_intersection_approx :
  abs (area_intersection - 5.88) < 0.01 := by sorry

end NUMINAMATH_CALUDE_area_intersection_approx_l475_47549


namespace NUMINAMATH_CALUDE_triangle_ABC_point_C_l475_47568

-- Define the points A and B
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (-1, 5)

-- Define the line on which C lies
def line_C (x : ℝ) : ℝ := 3 * x + 3

-- Define the area of the triangle
def triangle_area : ℝ := 10

-- Theorem statement
theorem triangle_ABC_point_C :
  ∀ (C : ℝ × ℝ),
  (C.2 = line_C C.1) →  -- C lies on the line y = 3x + 3
  (abs ((C.1 - A.1) * (B.2 - A.2) - (C.2 - A.2) * (B.1 - A.1)) / 2 = triangle_area) →  -- Area of triangle ABC is 10
  (C = (-1, 0) ∨ C = (5/3, 14)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_point_C_l475_47568


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l475_47523

/-- Given two points are symmetric about a line, prove the equation of the line -/
theorem symmetric_points_line_equation (O A : ℝ × ℝ) (l : Set (ℝ × ℝ)) : 
  O = (0, 0) → 
  A = (-4, 2) → 
  (∀ p : ℝ × ℝ, p ∈ l ↔ (2 : ℝ) * p.1 - p.2 + 5 = 0) →
  (∀ p : ℝ × ℝ, p ∈ l → dist O p = dist A p) →
  True :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_line_equation_l475_47523
