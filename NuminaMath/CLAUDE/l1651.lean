import Mathlib

namespace NUMINAMATH_CALUDE_laundry_items_not_done_l1651_165155

theorem laundry_items_not_done (short_sleeve : ℕ) (long_sleeve : ℕ) (socks : ℕ) (handkerchiefs : ℕ)
  (shirts_washed : ℕ) (socks_folded : ℕ) (handkerchiefs_sorted : ℕ)
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 27)
  (h3 : socks = 50)
  (h4 : handkerchiefs = 34)
  (h5 : shirts_washed = 20)
  (h6 : socks_folded = 30)
  (h7 : handkerchiefs_sorted = 16) :
  (short_sleeve + long_sleeve - shirts_washed) + (socks - socks_folded) + (handkerchiefs - handkerchiefs_sorted) = 54 :=
by sorry

end NUMINAMATH_CALUDE_laundry_items_not_done_l1651_165155


namespace NUMINAMATH_CALUDE_no_natural_solutions_l1651_165198

theorem no_natural_solutions : ∀ x y z : ℕ, x^2 + y^2 + z^2 ≠ 2*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l1651_165198


namespace NUMINAMATH_CALUDE_book_arrangement_count_l1651_165175

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def arrangeBooksCount (mathBooks : ℕ) (englishBooks : ℕ) : ℕ :=
  2 * factorial mathBooks * factorial englishBooks

theorem book_arrangement_count :
  arrangeBooksCount 3 5 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l1651_165175


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1651_165194

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def satisfies_condition (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 1 → a n ^ 2 = a (n - 1) * a (n + 1)

theorem geometric_sequence_property :
  (∀ a : ℕ → ℝ, is_geometric_sequence a → satisfies_condition a) ∧
  (∃ a : ℕ → ℝ, satisfies_condition a ∧ ¬is_geometric_sequence a) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1651_165194


namespace NUMINAMATH_CALUDE_largest_factorable_m_l1651_165154

/-- A quadratic expression of the form 3x^2 + mx - 60 -/
def quadratic (m : ℤ) (x : ℤ) : ℤ := 3 * x^2 + m * x - 60

/-- Checks if a quadratic expression can be factored into two linear factors with integer coefficients -/
def is_factorable (m : ℤ) : Prop :=
  ∃ (a b c d : ℤ), ∀ x, quadratic m x = (a * x + b) * (c * x + d)

/-- The largest value of m for which the quadratic is factorable -/
def largest_m : ℤ := 57

theorem largest_factorable_m :
  (is_factorable largest_m) ∧
  (∀ m : ℤ, m > largest_m → ¬(is_factorable m)) := by sorry

end NUMINAMATH_CALUDE_largest_factorable_m_l1651_165154


namespace NUMINAMATH_CALUDE_calculate_principal_l1651_165148

/-- Given simple interest, rate, and time, calculate the principal amount --/
theorem calculate_principal
  (simple_interest : ℝ)
  (rate : ℝ)
  (time : ℝ)
  (h1 : simple_interest = 4016.25)
  (h2 : rate = 3)
  (h3 : time = 5)
  : (simple_interest * 100) / (rate * time) = 26775 := by
  sorry

end NUMINAMATH_CALUDE_calculate_principal_l1651_165148


namespace NUMINAMATH_CALUDE_sqrt_8_times_sqrt_50_l1651_165145

theorem sqrt_8_times_sqrt_50 : Real.sqrt 8 * Real.sqrt 50 = 20 := by sorry

end NUMINAMATH_CALUDE_sqrt_8_times_sqrt_50_l1651_165145


namespace NUMINAMATH_CALUDE_additional_oil_purchased_l1651_165162

/-- Proves that a 30% price reduction allows purchasing 9 more kgs of oil with a budget of 900 Rs. --/
theorem additional_oil_purchased (budget : ℝ) (reduced_price : ℝ) (reduction_percentage : ℝ) : 
  budget = 900 →
  reduced_price = 30 →
  reduction_percentage = 0.3 →
  ⌊budget / reduced_price - budget / (reduced_price / (1 - reduction_percentage))⌋ = 9 := by
  sorry

end NUMINAMATH_CALUDE_additional_oil_purchased_l1651_165162


namespace NUMINAMATH_CALUDE_problem_1_l1651_165199

theorem problem_1 : -3 + 8 - 15 - 6 = -16 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1651_165199


namespace NUMINAMATH_CALUDE_pascal_triangle_20th_number_in_25_number_row_l1651_165139

theorem pascal_triangle_20th_number_in_25_number_row : 
  let n : ℕ := 24  -- The row number (0-indexed) for a row with 25 numbers
  let k : ℕ := 19  -- The 0-indexed position of the 20th number
  Nat.choose n k = 4252 := by sorry

end NUMINAMATH_CALUDE_pascal_triangle_20th_number_in_25_number_row_l1651_165139


namespace NUMINAMATH_CALUDE_log_100_base_10_l1651_165159

theorem log_100_base_10 : Real.log 100 / Real.log 10 = 2 := by sorry

end NUMINAMATH_CALUDE_log_100_base_10_l1651_165159


namespace NUMINAMATH_CALUDE_xy_value_l1651_165140

theorem xy_value (x y : ℝ) (h : |x + 2| + (y - 3)^2 = 0) : x^y = -8 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l1651_165140


namespace NUMINAMATH_CALUDE_min_operations_2_to_400_l1651_165141

/-- Represents the possible operations on the calculator --/
inductive Operation
  | AddOne
  | MultiplyTwo

/-- Applies an operation to a number --/
def applyOperation (n : ℕ) (op : Operation) : ℕ :=
  match op with
  | Operation.AddOne => n + 1
  | Operation.MultiplyTwo => n * 2

/-- Checks if a sequence of operations transforms start into target --/
def transformsTo (start target : ℕ) (ops : List Operation) : Prop :=
  ops.foldl applyOperation start = target

/-- The minimum number of operations to transform 2 into 400 is 9 --/
theorem min_operations_2_to_400 :
  ∃ (ops : List Operation),
    transformsTo 2 400 ops ∧
    ops.length = 9 ∧
    (∀ (other_ops : List Operation),
      transformsTo 2 400 other_ops → other_ops.length ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_min_operations_2_to_400_l1651_165141


namespace NUMINAMATH_CALUDE_valid_topping_combinations_l1651_165186

/-- Represents the number of cheese options --/
def cheese_options : ℕ := 3

/-- Represents the number of meat options --/
def meat_options : ℕ := 4

/-- Represents the number of vegetable options --/
def vegetable_options : ℕ := 5

/-- Represents that peppers is one of the vegetable options --/
axiom peppers_is_vegetable : vegetable_options > 0

/-- Represents that pepperoni is one of the meat options --/
axiom pepperoni_is_meat : meat_options > 0

/-- Calculates the total number of combinations without restrictions --/
def total_combinations : ℕ := cheese_options * meat_options * vegetable_options

/-- Represents the number of invalid combinations (pepperoni with peppers) --/
def invalid_combinations : ℕ := 1

/-- Theorem stating the total number of valid topping combinations --/
theorem valid_topping_combinations : 
  total_combinations - invalid_combinations = 59 := by sorry

end NUMINAMATH_CALUDE_valid_topping_combinations_l1651_165186


namespace NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l1651_165138

/-- 
Given a parallelogram with area 98 square meters and base length 7 meters,
prove that the ratio of its altitude to its base is 2.
-/
theorem parallelogram_altitude_base_ratio :
  ∀ (area base altitude : ℝ),
  area = 98 →
  base = 7 →
  area = base * altitude →
  altitude / base = 2 := by
sorry

end NUMINAMATH_CALUDE_parallelogram_altitude_base_ratio_l1651_165138


namespace NUMINAMATH_CALUDE_jennis_age_l1651_165164

theorem jennis_age (sum diff : ℕ) (h_sum : sum = 70) (h_diff : diff = 32) :
  ∃ (age_jenni age_bai : ℕ), age_jenni + age_bai = sum ∧ age_bai - age_jenni = diff ∧ age_jenni = 19 :=
by sorry

end NUMINAMATH_CALUDE_jennis_age_l1651_165164


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l1651_165129

theorem cube_roots_of_unity_sum (ω ω_bar : ℂ) : 
  ω = (-1 + Complex.I * Real.sqrt 3) / 2 →
  ω_bar = (-1 - Complex.I * Real.sqrt 3) / 2 →
  ω^3 = 1 →
  ω_bar^3 = 1 →
  ω^9 + ω_bar^9 = 2 := by sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_sum_l1651_165129


namespace NUMINAMATH_CALUDE_minimum_buses_l1651_165169

theorem minimum_buses (max_capacity : ℕ) (total_students : ℕ) (h1 : max_capacity = 45) (h2 : total_students = 495) :
  ∃ n : ℕ, n * max_capacity ≥ total_students ∧ ∀ m : ℕ, m * max_capacity ≥ total_students → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_minimum_buses_l1651_165169


namespace NUMINAMATH_CALUDE_sum_of_three_numbers_l1651_165133

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 179) 
  (h2 : a*b + b*c + c*a = 131) : 
  a + b + c = 21 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_numbers_l1651_165133


namespace NUMINAMATH_CALUDE_unique_sum_preceding_numbers_l1651_165150

theorem unique_sum_preceding_numbers : 
  ∃! n : ℕ, n > 0 ∧ n = (n * (n - 1)) / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_sum_preceding_numbers_l1651_165150


namespace NUMINAMATH_CALUDE_log_identity_l1651_165100

theorem log_identity (x y : ℝ) 
  (hx : Real.log 5 / Real.log 4 = x)
  (hy : Real.log 7 / Real.log 5 = y) : 
  Real.log 7 / Real.log 10 = (2 * x * y) / (2 * x + 1) := by
  sorry

end NUMINAMATH_CALUDE_log_identity_l1651_165100


namespace NUMINAMATH_CALUDE_triangle_angle_C_l1651_165187

theorem triangle_angle_C (A B C : ℝ) (h : (Real.cos A + Real.sin A) * (Real.cos B + Real.sin B) = 2) :
  A + B + C = Real.pi → C = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l1651_165187


namespace NUMINAMATH_CALUDE_movie_day_points_l1651_165103

theorem movie_day_points (num_students : ℕ) (num_weeks : ℕ) (veg_per_week : ℕ) (points_per_veg : ℕ)
  (h1 : num_students = 25)
  (h2 : num_weeks = 2)
  (h3 : veg_per_week = 2)
  (h4 : points_per_veg = 2) :
  num_students * num_weeks * veg_per_week * points_per_veg = 200 := by
  sorry

#check movie_day_points

end NUMINAMATH_CALUDE_movie_day_points_l1651_165103


namespace NUMINAMATH_CALUDE_soda_price_calculation_l1651_165116

theorem soda_price_calculation (remy_morning : ℕ) (nick_diff : ℕ) (evening_sales : ℚ) (evening_increase : ℚ) :
  remy_morning = 55 →
  nick_diff = 6 →
  evening_sales = 55 →
  evening_increase = 3 →
  ∃ (price : ℚ), price = 1/2 ∧ 
    (remy_morning + (remy_morning - nick_diff)) * price + evening_increase = evening_sales :=
by
  sorry

end NUMINAMATH_CALUDE_soda_price_calculation_l1651_165116


namespace NUMINAMATH_CALUDE_smallest_marble_count_l1651_165110

/-- Represents the number of marbles of each color --/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability of drawing two marbles of one color and two of another --/
def prob_two_two (m : MarbleCount) (c1 c2 : ℕ) : ℚ :=
  (c1.choose 2 * c2.choose 2 : ℚ) / (m.red + m.white + m.blue + m.green).choose 4

/-- Calculates the probability of drawing one marble of each color --/
def prob_one_each (m : MarbleCount) : ℚ :=
  (m.red * m.white * m.blue * m.green : ℚ) / (m.red + m.white + m.blue + m.green).choose 4

/-- Checks if the probabilities of the three events are equal --/
def probabilities_equal (m : MarbleCount) : Prop :=
  prob_two_two m m.red m.blue = prob_two_two m m.white m.green ∧
  prob_two_two m m.red m.blue = prob_one_each m

/-- The theorem stating that 10 is the smallest number of marbles satisfying the conditions --/
theorem smallest_marble_count : 
  ∃ (m : MarbleCount), 
    (m.red + m.white + m.blue + m.green = 10) ∧ 
    probabilities_equal m ∧
    (∀ (n : MarbleCount), 
      (n.red + n.white + n.blue + n.green < 10) → ¬probabilities_equal n) :=
  sorry

end NUMINAMATH_CALUDE_smallest_marble_count_l1651_165110


namespace NUMINAMATH_CALUDE_alyssa_bought_224_cards_l1651_165118

/-- The number of Pokemon cards Jason initially had -/
def initial_cards : ℕ := 676

/-- The number of Pokemon cards Jason has after Alyssa bought some -/
def remaining_cards : ℕ := 452

/-- The number of Pokemon cards Alyssa bought -/
def cards_bought : ℕ := initial_cards - remaining_cards

theorem alyssa_bought_224_cards : cards_bought = 224 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_bought_224_cards_l1651_165118


namespace NUMINAMATH_CALUDE_multiples_of_five_average_l1651_165176

theorem multiples_of_five_average (n : ℕ) : 
  (((n : ℝ) / 2) * (5 + 5 * n)) / n = 55 → n = 21 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_five_average_l1651_165176


namespace NUMINAMATH_CALUDE_school_students_l1651_165111

/-- Prove that the total number of students in a school is 1000, given the conditions described. -/
theorem school_students (S : ℕ) : 
  (S / 2) / 2 = 250 → S = 1000 := by
  sorry

end NUMINAMATH_CALUDE_school_students_l1651_165111


namespace NUMINAMATH_CALUDE_find_p_l1651_165158

def fibonacci_like_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n, n ≥ 2 → a n = a (n - 1) + a (n - 2)

theorem find_p (a : ℕ → ℕ) (h : fibonacci_like_sequence a) 
  (h5 : a 4 = 5) (h8 : a 5 = 8) (h13 : a 6 = 13) : a 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_p_l1651_165158


namespace NUMINAMATH_CALUDE_soldiers_per_tower_l1651_165161

/-- Proves that the number of soldiers in each tower is 2 -/
theorem soldiers_per_tower (wall_length : ℕ) (tower_interval : ℕ) (total_soldiers : ℕ)
  (h1 : wall_length = 7300)
  (h2 : tower_interval = 5)
  (h3 : total_soldiers = 2920) :
  total_soldiers / (wall_length / tower_interval) = 2 := by
  sorry

end NUMINAMATH_CALUDE_soldiers_per_tower_l1651_165161


namespace NUMINAMATH_CALUDE_equation_with_prime_solutions_l1651_165134

theorem equation_with_prime_solutions (m : ℕ) : 
  (∃ x y : ℕ, Prime x ∧ Prime y ∧ x ≠ y ∧ x^2 - 1999*x + m = 0 ∧ y^2 - 1999*y + m = 0) → 
  m = 3994 := by
sorry

end NUMINAMATH_CALUDE_equation_with_prime_solutions_l1651_165134


namespace NUMINAMATH_CALUDE_age_problem_l1651_165124

theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2) 
  (h2 : b = 2 * c) 
  (h3 : a + b + c = 27) : 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1651_165124


namespace NUMINAMATH_CALUDE_triangle_side_calculation_l1651_165166

/-- Given a triangle ABC with side a = 4, angle B = π/3, and area S = 6√3,
    prove that side b = 2√7 -/
theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  -- Conditions
  a = 4 →
  B = π / 3 →
  (1 / 2) * a * c * Real.sin B = 6 * Real.sqrt 3 →
  -- Definition of cosine law
  b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B →
  -- Conclusion
  b = 2 * Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_calculation_l1651_165166


namespace NUMINAMATH_CALUDE_holiday_rain_probability_l1651_165184

/-- Probability of rain on Monday -/
def prob_rain_monday : ℝ := 0.3

/-- Probability of rain on Tuesday -/
def prob_rain_tuesday : ℝ := 0.6

/-- Probability of rain continuing to the next day -/
def prob_rain_continue : ℝ := 0.8

/-- Probability of rain on at least one day during the two-day holiday period -/
def prob_rain_at_least_one_day : ℝ :=
  1 - (1 - prob_rain_monday) * (1 - prob_rain_tuesday)

theorem holiday_rain_probability :
  prob_rain_at_least_one_day = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_holiday_rain_probability_l1651_165184


namespace NUMINAMATH_CALUDE_max_min_values_on_interval_l1651_165193

-- Define the function
def f (x : ℝ) : ℝ := 3 * x^4 + 4 * x^3 + 34

-- Define the interval
def interval : Set ℝ := { x | -2 ≤ x ∧ x ≤ 1 }

-- State the theorem
theorem max_min_values_on_interval :
  (∃ x ∈ interval, f x = 50 ∧ ∀ y ∈ interval, f y ≤ 50) ∧
  (∃ x ∈ interval, f x = 33 ∧ ∀ y ∈ interval, f y ≥ 33) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_on_interval_l1651_165193


namespace NUMINAMATH_CALUDE_fraction_simplification_l1651_165102

theorem fraction_simplification (a : ℝ) (h : a ≠ 1) : a / (a - 1) + 1 / (1 - a) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1651_165102


namespace NUMINAMATH_CALUDE_conversation_on_weekday_l1651_165160

-- Define the days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday

-- Define a function to check if a day is a weekday
def isWeekday (d : Day) : Prop :=
  d ≠ Day.Saturday ∧ d ≠ Day.Sunday

-- Define the brothers
structure Brother :=
  (liesOnSaturday : Bool)
  (liesOnSunday : Bool)
  (willLieTomorrow : Bool)

-- Define the conversation
def conversation (day : Day) (brother1 brother2 : Brother) : Prop :=
  brother1.liesOnSaturday = true
  ∧ brother1.liesOnSunday = true
  ∧ brother2.willLieTomorrow = true
  ∧ (day = Day.Saturday → ¬brother1.liesOnSaturday)
  ∧ (day = Day.Sunday → ¬brother1.liesOnSunday)
  ∧ (isWeekday day → ¬brother2.willLieTomorrow)

-- Theorem: The conversation occurs on a weekday
theorem conversation_on_weekday (day : Day) (brother1 brother2 : Brother) :
  conversation day brother1 brother2 → isWeekday day :=
by sorry

end NUMINAMATH_CALUDE_conversation_on_weekday_l1651_165160


namespace NUMINAMATH_CALUDE_reinforcement_theorem_l1651_165197

/-- Calculates the size of reinforcement given initial garrison size, initial provision duration,
    days passed before reinforcement, and remaining provision duration after reinforcement. -/
def reinforcement_size (initial_garrison : ℕ) (initial_duration : ℕ) 
    (days_before_reinforcement : ℕ) (remaining_duration : ℕ) : ℕ :=
  (initial_garrison * initial_duration - initial_garrison * days_before_reinforcement) / remaining_duration - initial_garrison

/-- Theorem stating that given the problem conditions, the reinforcement size is 2000. -/
theorem reinforcement_theorem : 
  reinforcement_size 2000 40 20 10 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_theorem_l1651_165197


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1651_165173

theorem quadratic_equation_solution (p : ℝ) (α β : ℝ) : 
  (∀ x, x^2 + p*x + p = 0 ↔ x = α ∨ x = β) →
  (α^2 + β^2 = 3) →
  p = -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1651_165173


namespace NUMINAMATH_CALUDE_average_price_is_52_cents_l1651_165163

/-- Represents the fruit selection problem -/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  oranges_removed : ℕ

/-- Calculates the average price of fruits kept -/
def average_price_kept (fs : FruitSelection) : ℚ :=
  sorry

/-- Theorem stating the average price of fruits kept is 52 cents -/
theorem average_price_is_52_cents (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 20)
  (h4 : fs.initial_avg_price = 56/100)
  (h5 : fs.oranges_removed = 10) :
  average_price_kept fs = 52/100 :=
by sorry

end NUMINAMATH_CALUDE_average_price_is_52_cents_l1651_165163


namespace NUMINAMATH_CALUDE_extended_midpoint_theorem_l1651_165112

/-- Given two points in 2D space, find the coordinates of a point that is twice as far from their midpoint towards the second point. -/
theorem extended_midpoint_theorem (x₁ y₁ x₂ y₂ : ℚ) :
  let a := (x₁, y₁)
  let b := (x₂, y₂)
  let m := ((x₁ + x₂) / 2, (y₁ + y₂) / 2)
  let p := ((2 * x₂ + x₁) / 3, (2 * y₂ + y₁) / 3)
  (x₁ = 2 ∧ y₁ = 6 ∧ x₂ = 8 ∧ y₂ = 2) →
  p = (7, 8/3) :=
by sorry

end NUMINAMATH_CALUDE_extended_midpoint_theorem_l1651_165112


namespace NUMINAMATH_CALUDE_chord_equation_through_midpoint_l1651_165101

/-- The equation of a chord passing through a point on an ellipse --/
theorem chord_equation_through_midpoint (x y : ℝ) :
  (4 * x^2 + 9 * y^2 = 144) →  -- Ellipse equation
  (3 : ℝ)^2 * 4 + 1^2 * 9 < 144 →  -- P(3,1) is inside the ellipse
  (∃ (x₁ y₁ x₂ y₂ : ℝ),  -- Existence of chord endpoints
    (4 * x₁^2 + 9 * y₁^2 = 144) ∧  -- A is on the ellipse
    (4 * x₂^2 + 9 * y₂^2 = 144) ∧  -- B is on the ellipse
    (x₁ + x₂ = 6) ∧  -- P is midpoint (x-coordinate)
    (y₁ + y₂ = 2)) →  -- P is midpoint (y-coordinate)
  (4 * x + 3 * y - 15 = 0)  -- Equation of the chord
  := by sorry

end NUMINAMATH_CALUDE_chord_equation_through_midpoint_l1651_165101


namespace NUMINAMATH_CALUDE_unattainable_value_l1651_165132

/-- The function f(x) = (1-2x) / (3x+4) cannot attain the value -2/3 for any real x ≠ -4/3. -/
theorem unattainable_value (x : ℝ) (hx : x ≠ -4/3) :
  (1 - 2*x) / (3*x + 4) ≠ -2/3 := by
  sorry


end NUMINAMATH_CALUDE_unattainable_value_l1651_165132


namespace NUMINAMATH_CALUDE_oatmeal_cookies_count_l1651_165156

/-- The number of batches of chocolate chip cookies -/
def chocolate_chip_batches : ℕ := 2

/-- The number of cookies in each batch of chocolate chip cookies -/
def cookies_per_batch : ℕ := 3

/-- The total number of cookies baked -/
def total_cookies : ℕ := 10

/-- The number of oatmeal cookies -/
def oatmeal_cookies : ℕ := total_cookies - (chocolate_chip_batches * cookies_per_batch)

theorem oatmeal_cookies_count : oatmeal_cookies = 4 := by
  sorry

end NUMINAMATH_CALUDE_oatmeal_cookies_count_l1651_165156


namespace NUMINAMATH_CALUDE_sample_definition_l1651_165121

/-- Represents a student's math score -/
def MathScore : Type := ℝ

/-- Represents a sample of math scores -/
def Sample : Type := List MathScore

structure SurveyData where
  totalStudents : ℕ
  sampleSize : ℕ
  scores : Sample
  h_sampleSize : sampleSize ≤ totalStudents

/-- Definition of a valid sample for the survey -/
def isValidSample (data : SurveyData) : Prop :=
  data.scores.length = data.sampleSize

theorem sample_definition (data : SurveyData) 
  (h_total : data.totalStudents = 960)
  (h_sample : data.sampleSize = 120)
  (h_valid : isValidSample data) :
  ∃ (sample : Sample), sample = data.scores ∧ sample.length = 120 :=
sorry

end NUMINAMATH_CALUDE_sample_definition_l1651_165121


namespace NUMINAMATH_CALUDE_smallest_twice_cube_thrice_square_l1651_165115

theorem smallest_twice_cube_thrice_square :
  (∃ k : ℕ, k > 0 ∧
    (∃ n : ℕ, k = 2 * n^3) ∧
    (∃ m : ℕ, k = 3 * m^2) ∧
    (∀ j : ℕ, j > 0 →
      (∃ p : ℕ, j = 2 * p^3) →
      (∃ q : ℕ, j = 3 * q^2) →
      j ≥ k)) →
  (∃ k : ℕ, k = 432 ∧
    (∃ n : ℕ, k = 2 * n^3) ∧
    (∃ m : ℕ, k = 3 * m^2) ∧
    (∀ j : ℕ, j > 0 →
      (∃ p : ℕ, j = 2 * p^3) →
      (∃ q : ℕ, j = 3 * q^2) →
      j ≥ k)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_twice_cube_thrice_square_l1651_165115


namespace NUMINAMATH_CALUDE_inequalities_always_true_l1651_165126

theorem inequalities_always_true (x y a b : ℝ) (h1 : x > y) (h2 : a > b) :
  (a + x > b + y) ∧ (x - b > y - a) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_always_true_l1651_165126


namespace NUMINAMATH_CALUDE_permutation_problem_l1651_165127

theorem permutation_problem (n : ℕ) : n * (n - 1) = 132 → n = 12 := by sorry

end NUMINAMATH_CALUDE_permutation_problem_l1651_165127


namespace NUMINAMATH_CALUDE_evaluate_expression_max_value_function_max_value_function_achievable_l1651_165137

-- Part 1
theorem evaluate_expression : 
  Real.sqrt 3 * Real.cos (π / 12) - Real.sin (π / 12) = Real.sqrt 2 := by sorry

-- Part 2
theorem max_value_function : 
  ∀ θ : ℝ, Real.sqrt 3 * Real.cos θ - Real.sin θ ≤ 2 := by sorry

theorem max_value_function_achievable : 
  ∃ θ : ℝ, Real.sqrt 3 * Real.cos θ - Real.sin θ = 2 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_max_value_function_max_value_function_achievable_l1651_165137


namespace NUMINAMATH_CALUDE_library_to_post_office_l1651_165191

def total_distance : ℝ := 0.8
def house_to_library : ℝ := 0.3
def post_office_to_house : ℝ := 0.4

theorem library_to_post_office :
  total_distance - house_to_library - post_office_to_house = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_library_to_post_office_l1651_165191


namespace NUMINAMATH_CALUDE_magnitude_of_z_is_one_l1651_165196

-- Define the complex number z
variable (z : ℂ)

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem magnitude_of_z_is_one (h : (1 - z) / (1 + z) = 2 * i) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_is_one_l1651_165196


namespace NUMINAMATH_CALUDE_distributive_property_subtraction_l1651_165185

theorem distributive_property_subtraction (a b c : ℝ) : a - (b + c) = a - b - c := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_subtraction_l1651_165185


namespace NUMINAMATH_CALUDE_xyz_product_l1651_165179

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * (y + z) = 180)
  (h2 : y * (z + x) = 192)
  (h3 : z * (x + y) = 204) :
  x * y * z = 168 * Real.sqrt 6 := by
sorry

end NUMINAMATH_CALUDE_xyz_product_l1651_165179


namespace NUMINAMATH_CALUDE_laptop_discount_l1651_165105

theorem laptop_discount (initial_discount additional_discount : ℝ) 
  (h1 : initial_discount = 0.3)
  (h2 : additional_discount = 0.5) : 
  1 - (1 - initial_discount) * (1 - additional_discount) = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_laptop_discount_l1651_165105


namespace NUMINAMATH_CALUDE_area_bounded_by_curves_l1651_165180

-- Define the function f(x) = x^3 - 4x
def f (x : ℝ) : ℝ := x^3 - 4*x

-- State the theorem
theorem area_bounded_by_curves : 
  ∃ (a b : ℝ), a ≥ 0 ∧ b > a ∧ f a = 0 ∧ f b = 0 ∧ 
  (∫ (x : ℝ) in a..b, |f x|) = 4 := by
  sorry

end NUMINAMATH_CALUDE_area_bounded_by_curves_l1651_165180


namespace NUMINAMATH_CALUDE_mabels_daisies_l1651_165183

/-- Given initial daisies, petals per daisy, and daisies given away, 
    calculate the number of petals on remaining daisies -/
def remaining_petals (initial_daisies : ℕ) (petals_per_daisy : ℕ) (daisies_given : ℕ) : ℕ :=
  (initial_daisies - daisies_given) * petals_per_daisy

/-- Theorem: Given 5 initial daisies with 8 petals each, 
    after giving away 2 daisies, 24 petals remain -/
theorem mabels_daisies : remaining_petals 5 8 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_mabels_daisies_l1651_165183


namespace NUMINAMATH_CALUDE_thirteen_fourth_mod_eight_l1651_165168

theorem thirteen_fourth_mod_eight (m : ℕ) : 
  13^4 % 8 = m ∧ 0 ≤ m ∧ m < 8 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_fourth_mod_eight_l1651_165168


namespace NUMINAMATH_CALUDE_total_groom_time_is_210_l1651_165146

/-- The time it takes to groom a poodle, in minutes. -/
def poodle_groom_time : ℕ := 30

/-- The time it takes to groom a terrier, in minutes. -/
def terrier_groom_time : ℕ := poodle_groom_time / 2

/-- The number of poodles to be groomed. -/
def num_poodles : ℕ := 3

/-- The number of terriers to be groomed. -/
def num_terriers : ℕ := 8

/-- The total grooming time for all dogs. -/
def total_groom_time : ℕ := num_poodles * poodle_groom_time + num_terriers * terrier_groom_time

theorem total_groom_time_is_210 : total_groom_time = 210 := by
  sorry

end NUMINAMATH_CALUDE_total_groom_time_is_210_l1651_165146


namespace NUMINAMATH_CALUDE_negative_square_and_subtraction_l1651_165170

theorem negative_square_and_subtraction :
  (-4^2 = -16) ∧ ((-3) - (-6) = 3) := by sorry

end NUMINAMATH_CALUDE_negative_square_and_subtraction_l1651_165170


namespace NUMINAMATH_CALUDE_green_shirt_percentage_l1651_165167

-- Define the total number of students
def total_students : ℕ := 800

-- Define the percentage of students wearing blue shirts
def blue_percentage : ℚ := 45 / 100

-- Define the percentage of students wearing red shirts
def red_percentage : ℚ := 23 / 100

-- Define the number of students wearing other colors
def other_colors : ℕ := 136

-- Theorem to prove
theorem green_shirt_percentage :
  (total_students - (blue_percentage * total_students).floor - 
   (red_percentage * total_students).floor - other_colors) / total_students = 15 / 100 := by
sorry

end NUMINAMATH_CALUDE_green_shirt_percentage_l1651_165167


namespace NUMINAMATH_CALUDE_fraction_equality_l1651_165143

theorem fraction_equality (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_neq_xy : x ≠ y) (h_neq_yz : y ≠ z) (h_neq_xz : x ≠ z)
  (h_eq1 : (y + 1) / (x + z) = (x + y + 2) / (z + 1))
  (h_eq2 : (y + 1) / (x + z) = (x + 1) / y) :
  (x + 1) / y = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1651_165143


namespace NUMINAMATH_CALUDE_orange_calorie_distribution_l1651_165117

theorem orange_calorie_distribution :
  ∀ (num_oranges : ℕ) 
    (pieces_per_orange : ℕ) 
    (num_people : ℕ) 
    (calories_per_orange : ℕ),
  num_oranges = 5 →
  pieces_per_orange = 8 →
  num_people = 4 →
  calories_per_orange = 80 →
  (num_oranges * pieces_per_orange / num_people) * (calories_per_orange / pieces_per_orange) = 100 :=
by
  sorry

end NUMINAMATH_CALUDE_orange_calorie_distribution_l1651_165117


namespace NUMINAMATH_CALUDE_smallest_n_with_equal_digits_sum_l1651_165114

/-- The sum of the first n positive integers -/
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Check if all digits of a number are equal -/
def all_digits_equal (n : ℕ) : Prop := 
  ∃ (d : ℕ) (k : ℕ), d ∈ Finset.range 10 ∧ n = d * (10^k - 1) / 9

/-- Sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

theorem smallest_n_with_equal_digits_sum : 
  ∃ (N : ℕ), 
    (∀ m : ℕ, m < N → ¬(all_digits_equal (m * sum_first_n 9))) ∧
    (all_digits_equal (N * sum_first_n 9)) ∧
    (digit_sum N = 37) := by sorry

end NUMINAMATH_CALUDE_smallest_n_with_equal_digits_sum_l1651_165114


namespace NUMINAMATH_CALUDE_expected_worth_unfair_coin_expected_worth_is_zero_l1651_165178

/-- The expected worth of an unfair coin flip -/
theorem expected_worth_unfair_coin : ℝ :=
  let p_heads : ℝ := 2/3
  let p_tails : ℝ := 1/3
  let gain_heads : ℝ := 5
  let loss_tails : ℝ := 10
  p_heads * gain_heads + p_tails * (-loss_tails)

/-- Proof that the expected worth of the unfair coin flip is 0 -/
theorem expected_worth_is_zero : expected_worth_unfair_coin = 0 := by
  sorry

end NUMINAMATH_CALUDE_expected_worth_unfair_coin_expected_worth_is_zero_l1651_165178


namespace NUMINAMATH_CALUDE_angle_set_impossibility_l1651_165119

/-- Represents a set of angles formed by lines through a single point -/
structure AngleSet where
  odd : ℕ  -- number of angles with odd integer measures
  even : ℕ -- number of angles with even integer measures

/-- The property that the number of odd-measure angles is 15 more than even-measure angles -/
def has_15_more_odd (as : AngleSet) : Prop :=
  as.odd = as.even + 15

/-- The property that both odd and even counts are even numbers due to vertical angles -/
def vertical_angle_property (as : AngleSet) : Prop :=
  Even as.odd ∧ Even as.even

theorem angle_set_impossibility : 
  ¬∃ (as : AngleSet), has_15_more_odd as ∧ vertical_angle_property as :=
sorry

end NUMINAMATH_CALUDE_angle_set_impossibility_l1651_165119


namespace NUMINAMATH_CALUDE_opposite_of_two_minus_sqrt_five_l1651_165108

theorem opposite_of_two_minus_sqrt_five :
  -(2 - Real.sqrt 5) = Real.sqrt 5 - 2 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_two_minus_sqrt_five_l1651_165108


namespace NUMINAMATH_CALUDE_non_right_triangles_count_l1651_165189

-- Define the points on the grid
def Point := Fin 6

-- Define the grid
def Grid := Point → ℝ × ℝ

-- Define the specific grid layout
def grid_layout : Grid := sorry

-- Define a function to check if a triangle is right-angled
def is_right_angled (p q r : Point) (g : Grid) : Prop := sorry

-- Define a function to count non-right-angled triangles
def count_non_right_triangles (g : Grid) : ℕ := sorry

-- Theorem statement
theorem non_right_triangles_count :
  count_non_right_triangles grid_layout = 4 := by sorry

end NUMINAMATH_CALUDE_non_right_triangles_count_l1651_165189


namespace NUMINAMATH_CALUDE_gcd_of_polynomials_l1651_165113

theorem gcd_of_polynomials (x : ℤ) (h : ∃ k : ℤ, x = 2 * k * 2027) :
  Int.gcd (3 * x^2 + 47 * x + 101) (x + 23) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomials_l1651_165113


namespace NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2011_l1651_165182

-- Define the function to get the last four digits
def lastFourDigits (n : ℕ) : ℕ := n % 10000

-- Define the cycle of last four digits
def lastFourDigitsCycle : List ℕ := [3125, 5625, 8125, 0625]

theorem last_four_digits_of_5_pow_2011 :
  lastFourDigits (5^2011) = 8125 := by
  sorry

end NUMINAMATH_CALUDE_last_four_digits_of_5_pow_2011_l1651_165182


namespace NUMINAMATH_CALUDE_function_characterization_l1651_165123

theorem function_characterization (f : ℝ → ℤ) 
  (h1 : ∀ x y : ℝ, f (x + y) < f x + f y)
  (h2 : ∀ x : ℝ, f (f x) = ⌊x⌋ + 2) :
  ∀ x : ℤ, f x = x + 1 := by sorry

end NUMINAMATH_CALUDE_function_characterization_l1651_165123


namespace NUMINAMATH_CALUDE_inequality_proof_l1651_165172

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  1 / Real.sqrt (b + 1 / a + 1 / 2) + 1 / Real.sqrt (c + 1 / b + 1 / 2) + 1 / Real.sqrt (a + 1 / c + 1 / 2) ≥ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1651_165172


namespace NUMINAMATH_CALUDE_claire_photos_l1651_165107

theorem claire_photos (c : ℕ) 
  (h1 : 3 * c = c + 10) : c = 5 := by
  sorry

#check claire_photos

end NUMINAMATH_CALUDE_claire_photos_l1651_165107


namespace NUMINAMATH_CALUDE_sum_of_coordinates_D_l1651_165195

-- Define the points
def C : ℝ × ℝ := (10, 6)
def N : ℝ × ℝ := (4, 8)

-- Define D as a variable point
variable (D : ℝ × ℝ)

-- Define the midpoint condition
def is_midpoint (M A B : ℝ × ℝ) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Theorem statement
theorem sum_of_coordinates_D :
  is_midpoint N C D → D.1 + D.2 = 16 := by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_D_l1651_165195


namespace NUMINAMATH_CALUDE_f_expression_on_negative_interval_l1651_165151

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_expression_on_negative_interval
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_known : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| :=
sorry

end NUMINAMATH_CALUDE_f_expression_on_negative_interval_l1651_165151


namespace NUMINAMATH_CALUDE_polynomial_identity_l1651_165122

theorem polynomial_identity (x y : ℝ) : (x - y) * (x^2 + x*y + y^2) = x^3 - y^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l1651_165122


namespace NUMINAMATH_CALUDE_limit_at_zero_l1651_165181

-- Define the function f
def f (x : ℝ) := x^2

-- State the theorem
theorem limit_at_zero (ε : ℝ) (hε : ε > 0) : 
  ∃ δ > 0, ∀ Δx : ℝ, 0 < |Δx| ∧ |Δx| < δ → 
    |(f Δx - f 0) / Δx - 0| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_at_zero_l1651_165181


namespace NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l1651_165171

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_x_value
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2 * x - 3)
  (h_a3 : a 3 = 5 * x + 4)
  : x = -11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_x_value_l1651_165171


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l1651_165142

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h1 : jogger_speed = 9 * (1000 / 3600))  -- 9 km/hr in m/s
  (h2 : train_speed = 45 * (1000 / 3600))  -- 45 km/hr in m/s
  (h3 : train_length = 120)                -- 120 m
  (h4 : initial_distance = 180)            -- 180 m
  : (initial_distance + train_length) / (train_speed - jogger_speed) = 30 := by
  sorry


end NUMINAMATH_CALUDE_train_passing_jogger_time_l1651_165142


namespace NUMINAMATH_CALUDE_parabola_shift_l1651_165157

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := p.b - 2 * p.a * h
  , c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_shift :
  let original := Parabola.mk 1 (-2) 3
  let shifted := shift_parabola original 1 (-3)
  shifted = Parabola.mk 1 0 (-1) := by sorry

end NUMINAMATH_CALUDE_parabola_shift_l1651_165157


namespace NUMINAMATH_CALUDE_three_X_four_equals_31_l1651_165147

-- Define the operation X
def X (a b : ℤ) : ℤ := b + 12 * a - a^2

-- Theorem statement
theorem three_X_four_equals_31 : X 3 4 = 31 := by
  sorry

end NUMINAMATH_CALUDE_three_X_four_equals_31_l1651_165147


namespace NUMINAMATH_CALUDE_johns_and_brothers_age_sum_l1651_165125

/-- Given that John's age is four less than six times his brother's age,
    and his brother is 8 years old, prove that the sum of their ages is 52. -/
theorem johns_and_brothers_age_sum :
  ∀ (john_age brother_age : ℕ),
    brother_age = 8 →
    john_age = 6 * brother_age - 4 →
    john_age + brother_age = 52 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_and_brothers_age_sum_l1651_165125


namespace NUMINAMATH_CALUDE_unrestricted_arrangements_count_restricted_arrangements_count_l1651_165177

/-- Represents the number of singers in the chorus -/
def total_singers : ℕ := 8

/-- Represents the number of female singers -/
def female_singers : ℕ := 6

/-- Represents the number of male singers -/
def male_singers : ℕ := 2

/-- Represents the number of people per row -/
def people_per_row : ℕ := 4

/-- Represents the number of rows -/
def num_rows : ℕ := 2

/-- Calculates the number of arrangements with no restrictions -/
def unrestricted_arrangements : ℕ := Nat.factorial total_singers

/-- Calculates the number of arrangements with lead singer in front and male singers in back -/
def restricted_arrangements : ℕ :=
  (Nat.choose (female_singers - 1) (people_per_row - 1)) *
  (Nat.factorial people_per_row) *
  (Nat.factorial people_per_row)

/-- Theorem stating the number of unrestricted arrangements -/
theorem unrestricted_arrangements_count :
  unrestricted_arrangements = 40320 := by sorry

/-- Theorem stating the number of restricted arrangements -/
theorem restricted_arrangements_count :
  restricted_arrangements = 5760 := by sorry

end NUMINAMATH_CALUDE_unrestricted_arrangements_count_restricted_arrangements_count_l1651_165177


namespace NUMINAMATH_CALUDE_shopkeeper_pricing_l1651_165192

theorem shopkeeper_pricing (CP : ℝ) 
  (h1 : 0.65 * CP = 416) : 1.25 * CP = 800 := by
  sorry

end NUMINAMATH_CALUDE_shopkeeper_pricing_l1651_165192


namespace NUMINAMATH_CALUDE_cells_with_three_neighbors_count_l1651_165106

/-- Represents a rectangular grid --/
structure RectangularGrid where
  a : ℕ
  b : ℕ
  h_a : a ≥ 3
  h_b : b ≥ 3

/-- Two cells are neighboring if they share a common side --/
def neighboring (grid : RectangularGrid) : Prop := sorry

/-- The number of cells with exactly four neighboring cells --/
def cells_with_four_neighbors (grid : RectangularGrid) : ℕ :=
  (grid.a - 2) * (grid.b - 2)

/-- The number of cells with exactly three neighboring cells --/
def cells_with_three_neighbors (grid : RectangularGrid) : ℕ :=
  2 * (grid.a - 2) + 2 * (grid.b - 2)

/-- Main theorem: In a rectangular grid where 23 cells have exactly four neighboring cells,
    the number of cells with exactly three neighboring cells is 48 --/
theorem cells_with_three_neighbors_count
  (grid : RectangularGrid)
  (h : cells_with_four_neighbors grid = 23) :
  cells_with_three_neighbors grid = 48 := by
  sorry

end NUMINAMATH_CALUDE_cells_with_three_neighbors_count_l1651_165106


namespace NUMINAMATH_CALUDE_ruth_gave_53_stickers_l1651_165174

/-- The number of stickers Janet initially had -/
def initial_stickers : ℕ := 3

/-- The total number of stickers Janet has after receiving more from Ruth -/
def final_stickers : ℕ := 56

/-- The number of stickers Ruth gave to Janet -/
def stickers_from_ruth : ℕ := final_stickers - initial_stickers

theorem ruth_gave_53_stickers : stickers_from_ruth = 53 := by
  sorry

end NUMINAMATH_CALUDE_ruth_gave_53_stickers_l1651_165174


namespace NUMINAMATH_CALUDE_min_area_over_sqrt_t_l1651_165128

/-- The area bounded by the tangent lines and the parabola -/
noncomputable def S (t : ℝ) : ℝ := (2 / 3) * (1 + t^2)^(3/2)

/-- The main theorem statement -/
theorem min_area_over_sqrt_t (t : ℝ) (ht : t > 0) :
  ∃ (min_value : ℝ), min_value = (2 * 6^(3/2)) / (3 * 5^(5/4)) ∧
  ∀ (t : ℝ), t > 0 → S t / Real.sqrt t ≥ min_value :=
sorry

end NUMINAMATH_CALUDE_min_area_over_sqrt_t_l1651_165128


namespace NUMINAMATH_CALUDE_three_digit_product_sum_l1651_165109

theorem three_digit_product_sum (P A U : ℕ) : 
  P ≠ A → P ≠ U → A ≠ U →
  P ≥ 1 → P ≤ 9 →
  A ≥ 0 → A ≤ 9 →
  U ≥ 0 → U ≤ 9 →
  100 * P + 10 * A + U ≥ 100 →
  100 * P + 10 * A + U ≤ 999 →
  (P + A + U) * P * A * U = 300 →
  ∃ (PAU : ℕ), PAU = 100 * P + 10 * A + U ∧ 
               (PAU.div 100 + (PAU.mod 100).div 10 + PAU.mod 10) * 
               PAU.div 100 * (PAU.mod 100).div 10 * PAU.mod 10 = 300 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_product_sum_l1651_165109


namespace NUMINAMATH_CALUDE_equation_solution_exists_l1651_165136

theorem equation_solution_exists (a : Real) : 
  a ∈ Set.Icc 0.5 1.5 →
  ∃ t ∈ Set.Icc 0 (Real.pi / 2), 
    (abs (Real.cos t - 0.5) + abs (Real.sin t) - a) / (Real.sqrt 3 * Real.sin t - Real.cos t) = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l1651_165136


namespace NUMINAMATH_CALUDE_sum_of_roots_l1651_165144

/-- The function f(x) = x^3 + 3x^2 + 6x + 14 -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x + 14

/-- Theorem: If f(a) = 1 and f(b) = 19, then a + b = -2 -/
theorem sum_of_roots (a b : ℝ) (ha : f a = 1) (hb : f b = 19) : a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1651_165144


namespace NUMINAMATH_CALUDE_population_growth_l1651_165131

theorem population_growth (p₀ : ℝ) : 
  let p₁ := p₀ * 1.1
  let p₂ := p₁ * 1.2
  let p₃ := p₂ * 1.3
  (p₃ - p₀) / p₀ * 100 = 71.6 := by
sorry

end NUMINAMATH_CALUDE_population_growth_l1651_165131


namespace NUMINAMATH_CALUDE_correct_conclusions_l1651_165152

-- Define the vector type
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the parallel relation
def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b

-- Define the dot product
variable (dot : V → V → ℝ)

-- Statement of the theorem
theorem correct_conclusions :
  (∀ (a b c : V), a = b → b = c → a = c) ∧ 
  (∃ (a b c : V), parallel a b → parallel b c → ¬ parallel a c) ∧
  (∃ (a b : V), |dot a b| ≠ |dot a (1 • b)|) ∧
  (∀ (a b c : V), b = c → dot a b = dot a c) :=
sorry

end NUMINAMATH_CALUDE_correct_conclusions_l1651_165152


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1651_165190

theorem inequality_solution_set (d : ℝ) : 
  (d / 4 ≤ 3 - d ∧ 3 - d < 1 - 2*d) ↔ (-2 < d ∧ d ≤ 12/5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1651_165190


namespace NUMINAMATH_CALUDE_alloy_mixture_theorem_l1651_165153

/-- The amount of the first alloy used to create the third alloy -/
def first_alloy_amount : ℝ := 15

/-- The percentage of chromium in the first alloy -/
def first_alloy_chromium_percent : ℝ := 0.10

/-- The percentage of chromium in the second alloy -/
def second_alloy_chromium_percent : ℝ := 0.06

/-- The amount of the second alloy used to create the third alloy -/
def second_alloy_amount : ℝ := 35

/-- The percentage of chromium in the resulting third alloy -/
def third_alloy_chromium_percent : ℝ := 0.072

theorem alloy_mixture_theorem :
  first_alloy_amount * first_alloy_chromium_percent +
  second_alloy_amount * second_alloy_chromium_percent =
  (first_alloy_amount + second_alloy_amount) * third_alloy_chromium_percent :=
by sorry

end NUMINAMATH_CALUDE_alloy_mixture_theorem_l1651_165153


namespace NUMINAMATH_CALUDE_puppy_cost_l1651_165135

/-- Calculates the cost of a puppy given the total cost, food requirements, and food prices. -/
theorem puppy_cost (total_cost : ℚ) (weeks : ℕ) (daily_food : ℚ) (bag_size : ℚ) (bag_cost : ℚ) : 
  total_cost = 14 →
  weeks = 3 →
  daily_food = 1/3 →
  bag_size = 7/2 →
  bag_cost = 2 →
  total_cost - (((weeks * 7 * daily_food) / bag_size).ceil * bag_cost) = 10 := by
  sorry

end NUMINAMATH_CALUDE_puppy_cost_l1651_165135


namespace NUMINAMATH_CALUDE_sum_of_xyz_l1651_165120

theorem sum_of_xyz (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = 4 * y + x) : x + y + z = 17 * x := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l1651_165120


namespace NUMINAMATH_CALUDE_worlds_largest_dough_ball_profit_l1651_165130

/-- Calculate the profit from making the world's largest dough ball -/
theorem worlds_largest_dough_ball_profit :
  let flour_needed : ℕ := 500
  let salt_needed : ℕ := 10
  let sugar_needed : ℕ := 20
  let butter_needed : ℕ := 50
  let flour_bag_size : ℕ := 50
  let flour_bag_price : ℚ := 20
  let salt_price_per_pound : ℚ := 0.2
  let sugar_price_per_pound : ℚ := 0.5
  let butter_price_per_pound : ℚ := 2
  let butter_discount : ℚ := 0.1
  let chef_a_payment : ℚ := 200
  let chef_b_payment : ℚ := 250
  let chef_c_payment : ℚ := 300
  let chef_tax_rate : ℚ := 0.05
  let promotion_cost : ℚ := 1000
  let ticket_price : ℚ := 20
  let tickets_sold : ℕ := 1200

  let flour_cost := (flour_needed / flour_bag_size : ℚ) * flour_bag_price
  let salt_cost := salt_needed * salt_price_per_pound
  let sugar_cost := sugar_needed * sugar_price_per_pound
  let butter_cost := butter_needed * butter_price_per_pound * (1 - butter_discount)
  let ingredient_cost := flour_cost + salt_cost + sugar_cost + butter_cost

  let chefs_payment := chef_a_payment + chef_b_payment + chef_c_payment
  let chefs_tax := chefs_payment * chef_tax_rate
  let total_chef_cost := chefs_payment + chefs_tax

  let total_cost := ingredient_cost + total_chef_cost + promotion_cost
  let revenue := tickets_sold * ticket_price
  let profit := revenue - total_cost

  profit = 21910.50 := by sorry

end NUMINAMATH_CALUDE_worlds_largest_dough_ball_profit_l1651_165130


namespace NUMINAMATH_CALUDE_quotient_invariance_l1651_165149

theorem quotient_invariance (a b k : ℝ) (hb : b ≠ 0) (hk : k ≠ 0) :
  (a * k) / (b * k) = a / b := by
  sorry

end NUMINAMATH_CALUDE_quotient_invariance_l1651_165149


namespace NUMINAMATH_CALUDE_facebook_group_removal_l1651_165104

/-- Proves the number of removed members from a Facebook group --/
theorem facebook_group_removal (initial_members : ℕ) (messages_per_day : ℕ) (total_messages_week : ℕ) : 
  initial_members = 150 →
  messages_per_day = 50 →
  total_messages_week = 45500 →
  (initial_members - (initial_members - 20)) * messages_per_day * 7 = total_messages_week :=
by
  sorry

end NUMINAMATH_CALUDE_facebook_group_removal_l1651_165104


namespace NUMINAMATH_CALUDE_determine_counterfeit_weight_l1651_165188

/-- Represents the result of a weighing -/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a coin -/
structure Coin :=
  (id : Nat)
  (isCounterfeit : Bool)

/-- Represents a weighing on a two-pan balance scale -/
def weighing (leftPan : List Coin) (rightPan : List Coin) : WeighingResult :=
  sorry

/-- The main theorem stating that it's possible to determine if counterfeit coins are heavier or lighter -/
theorem determine_counterfeit_weight
  (coins : List Coin)
  (h1 : coins.length = 61)
  (h2 : (coins.filter (fun c => c.isCounterfeit)).length = 2)
  (h3 : ∀ c1 c2 : Coin, ¬c1.isCounterfeit ∧ ¬c2.isCounterfeit → c1.id ≠ c2.id → weighing [c1] [c2] = WeighingResult.Equal)
  (h4 : ∃ w : WeighingResult, w ≠ WeighingResult.Equal ∧ 
    ∀ c1 c2 : Coin, c1.isCounterfeit ∧ ¬c2.isCounterfeit → weighing [c1] [c2] = w) :
  ∃ (f : List (List Coin × List Coin)), 
    f.length ≤ 3 ∧ 
    (∃ (result : Bool), 
      result = true → (∀ c1 c2 : Coin, c1.isCounterfeit ∧ ¬c2.isCounterfeit → weighing [c1] [c2] = WeighingResult.LeftHeavier) ∧
      result = false → (∀ c1 c2 : Coin, c1.isCounterfeit ∧ ¬c2.isCounterfeit → weighing [c1] [c2] = WeighingResult.RightHeavier)) :=
  sorry

end NUMINAMATH_CALUDE_determine_counterfeit_weight_l1651_165188


namespace NUMINAMATH_CALUDE_meghan_money_l1651_165165

/-- The total amount of money Meghan has, given the number of bills of each denomination -/
def total_money (hundred_bills : ℕ) (fifty_bills : ℕ) (ten_bills : ℕ) : ℕ :=
  100 * hundred_bills + 50 * fifty_bills + 10 * ten_bills

/-- Theorem stating that Meghan's total money is $550 -/
theorem meghan_money : total_money 2 5 10 = 550 := by
  sorry

end NUMINAMATH_CALUDE_meghan_money_l1651_165165
