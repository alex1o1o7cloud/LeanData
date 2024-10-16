import Mathlib

namespace NUMINAMATH_CALUDE_smallest_n_for_inequality_l280_28008

theorem smallest_n_for_inequality : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → 3^(3^(m+1)) ≥ 1007 → n ≤ m) ∧
  3^(3^(n+1)) ≥ 1007 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_inequality_l280_28008


namespace NUMINAMATH_CALUDE_quadratic_factorization_l280_28052

theorem quadratic_factorization (p q : ℤ) :
  (∀ x : ℝ, 25 * x^2 - 135 * x - 150 = (5 * x + p) * (5 * x + q)) →
  p - q = 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l280_28052


namespace NUMINAMATH_CALUDE_jellybean_problem_l280_28074

theorem jellybean_problem :
  ∃ (n : ℕ), n ≥ 100 ∧ n % 13 = 11 ∧ ∀ (m : ℕ), m ≥ 100 ∧ m % 13 = 11 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_jellybean_problem_l280_28074


namespace NUMINAMATH_CALUDE_yi_rong_ferry_distance_l280_28066

/-- The Yi Rong ferry problem -/
theorem yi_rong_ferry_distance :
  let ferry_speed : ℝ := 40
  let water_speed : ℝ := 24
  let downstream_speed : ℝ := ferry_speed + water_speed
  let upstream_speed : ℝ := ferry_speed - water_speed
  let distance : ℝ := 192  -- The distance we want to prove

  -- Odd day condition
  (distance / downstream_speed * (43 / 18) = 
   distance / 2 / downstream_speed + distance / 2 / water_speed) ∧ 
  
  -- Even day condition
  (distance / upstream_speed = 
   distance / 2 / water_speed + 1 + distance / 2 / (2 * upstream_speed)) →
  
  distance = 192 := by sorry

end NUMINAMATH_CALUDE_yi_rong_ferry_distance_l280_28066


namespace NUMINAMATH_CALUDE_european_stamps_cost_l280_28095

/-- Represents a country with its stamp counts and price --/
structure Country where
  name : String
  price : ℚ
  count_80s : ℕ
  count_90s : ℕ

/-- Calculates the total cost of stamps for a country in both decades --/
def totalCost (c : Country) : ℚ :=
  c.price * (c.count_80s + c.count_90s)

/-- The set of European countries in Laura's collection --/
def europeanCountries : List Country :=
  [{ name := "France", price := 9/100, count_80s := 10, count_90s := 12 },
   { name := "Spain", price := 7/100, count_80s := 18, count_90s := 16 }]

theorem european_stamps_cost :
  List.sum (europeanCountries.map totalCost) = 436/100 := by
  sorry

end NUMINAMATH_CALUDE_european_stamps_cost_l280_28095


namespace NUMINAMATH_CALUDE_car_speed_problem_l280_28097

/-- Proves that the speed of the first car is 60 mph given the problem conditions -/
theorem car_speed_problem (v : ℝ) : 
  v > 0 →  -- Assuming positive speed
  2.5 * v + 2.5 * 64 = 310 → 
  v = 60 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l280_28097


namespace NUMINAMATH_CALUDE_add_3333_minutes_to_leap_day_noon_l280_28032

/-- Represents a date and time -/
structure DateTime where
  year : ℕ
  month : ℕ
  day : ℕ
  hour : ℕ
  minute : ℕ

/-- Represents the starting date and time -/
def startDateTime : DateTime :=
  { year := 2020, month := 2, day := 29, hour := 12, minute := 0 }

/-- The number of minutes to add -/
def minutesToAdd : ℕ := 3333

/-- The expected result date and time -/
def expectedDateTime : DateTime :=
  { year := 2020, month := 3, day := 2, hour := 19, minute := 33 }

/-- Function to add minutes to a DateTime -/
def addMinutes (dt : DateTime) (minutes : ℕ) : DateTime :=
  sorry

theorem add_3333_minutes_to_leap_day_noon :
  addMinutes startDateTime minutesToAdd = expectedDateTime := by sorry

end NUMINAMATH_CALUDE_add_3333_minutes_to_leap_day_noon_l280_28032


namespace NUMINAMATH_CALUDE_water_tank_capacity_water_tank_capacity_proof_l280_28048

/-- Proves that a cylindrical water tank holds 75 liters when full -/
theorem water_tank_capacity : ℝ → Prop :=
  fun c => 
    (∃ w : ℝ, w / c = 1 / 3 ∧ (w + 5) / c = 2 / 5) → c = 75

/-- The proof of the water tank capacity theorem -/
theorem water_tank_capacity_proof : water_tank_capacity 75 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_water_tank_capacity_proof_l280_28048


namespace NUMINAMATH_CALUDE_c_investment_is_2000_l280_28049

/-- Represents the investment and profit distribution in a business partnership --/
structure BusinessPartnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  b_profit_share : ℕ
  a_c_profit_diff : ℕ

/-- Calculates the profit share for a given investment --/
def profit_share (investment total_investment total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

/-- Theorem stating that C's investment is 2000 given the problem conditions --/
theorem c_investment_is_2000 (bp : BusinessPartnership)
  (h1 : bp.a_investment = 6000)
  (h2 : bp.b_investment = 8000)
  (h3 : bp.b_profit_share = 1000)
  (h4 : bp.a_c_profit_diff = 500)
  (h5 : bp.b_profit_share = profit_share bp.b_investment (bp.a_investment + bp.b_investment + bp.c_investment) bp.total_profit)
  (h6 : bp.a_c_profit_diff = profit_share bp.a_investment (bp.a_investment + bp.b_investment + bp.c_investment) bp.total_profit -
                             profit_share bp.c_investment (bp.a_investment + bp.b_investment + bp.c_investment) bp.total_profit) :
  bp.c_investment = 2000 := by
  sorry


end NUMINAMATH_CALUDE_c_investment_is_2000_l280_28049


namespace NUMINAMATH_CALUDE_weight_of_B_l280_28099

theorem weight_of_B (A B C : ℝ) :
  (A + B + C) / 3 = 45 →
  (A + B) / 2 = 41 →
  (B + C) / 2 = 43 →
  B = 33 := by
sorry

end NUMINAMATH_CALUDE_weight_of_B_l280_28099


namespace NUMINAMATH_CALUDE_board_cut_theorem_l280_28064

theorem board_cut_theorem (total_length : ℝ) (shorter_length : ℝ) :
  total_length = 20 ∧
  shorter_length > 0 ∧
  shorter_length < total_length ∧
  2 * shorter_length = (total_length - shorter_length) + 4 →
  shorter_length = 8 := by
  sorry

end NUMINAMATH_CALUDE_board_cut_theorem_l280_28064


namespace NUMINAMATH_CALUDE_quadratic_one_root_l280_28039

theorem quadratic_one_root (b c : ℝ) 
  (h1 : ∃! x : ℝ, x^2 + b*x + c = 0)
  (h2 : b = 2*c - 1) : 
  c = 1/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_one_root_l280_28039


namespace NUMINAMATH_CALUDE_beth_candy_counts_l280_28080

def possible_candy_counts (total : ℕ) (anne_min : ℕ) (beth_min : ℕ) (chris_min : ℕ) (chris_max : ℕ) : Set ℕ :=
  {b | ∃ (a c : ℕ), 
    a + b + c = total ∧ 
    a ≥ anne_min ∧ 
    b ≥ beth_min ∧ 
    c ≥ chris_min ∧ 
    c ≤ chris_max}

theorem beth_candy_counts : 
  possible_candy_counts 10 3 2 2 3 = {2, 3, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_beth_candy_counts_l280_28080


namespace NUMINAMATH_CALUDE_sqrt_nested_expression_l280_28088

theorem sqrt_nested_expression : Real.sqrt (16 * Real.sqrt (8 * Real.sqrt 4)) = 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_nested_expression_l280_28088


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l280_28091

theorem consecutive_integers_sum (n : ℕ) (x : ℤ) : 
  (n > 0) → 
  (x + n - 1 = 9) → 
  (n * (2 * x + n - 1) / 2 = 24) → 
  n = 3 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l280_28091


namespace NUMINAMATH_CALUDE_sum_of_series_l280_28025

theorem sum_of_series : 
  (∑' n : ℕ, (4 * n + 1 : ℝ) / (3 : ℝ) ^ n) = 7 / 2 := by sorry

end NUMINAMATH_CALUDE_sum_of_series_l280_28025


namespace NUMINAMATH_CALUDE_cubic_sum_greater_than_mixed_product_l280_28056

theorem cubic_sum_greater_than_mixed_product (a b : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : 
  a^3 + b^3 > a^2*b + a*b^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_greater_than_mixed_product_l280_28056


namespace NUMINAMATH_CALUDE_ali_monday_flowers_l280_28012

/-- The number of flowers Ali sold on Monday -/
def monday_flowers : ℕ := sorry

/-- The number of flowers Ali sold on Tuesday -/
def tuesday_flowers : ℕ := 8

/-- The number of flowers Ali sold on Friday -/
def friday_flowers : ℕ := 2 * monday_flowers

/-- The total number of flowers Ali sold -/
def total_flowers : ℕ := 20

theorem ali_monday_flowers : 
  monday_flowers + tuesday_flowers + friday_flowers = total_flowers → monday_flowers = 4 := by
sorry

end NUMINAMATH_CALUDE_ali_monday_flowers_l280_28012


namespace NUMINAMATH_CALUDE_lab_budget_remaining_l280_28057

/-- Given a laboratory budget and expenses, calculate the remaining budget. -/
theorem lab_budget_remaining (budget : ℚ) (flask_cost : ℚ) : 
  budget = 325 →
  flask_cost = 150 →
  let test_tube_cost := (2 / 3) * flask_cost
  let safety_gear_cost := (1 / 2) * test_tube_cost
  let total_expense := flask_cost + test_tube_cost + safety_gear_cost
  budget - total_expense = 25 := by sorry

end NUMINAMATH_CALUDE_lab_budget_remaining_l280_28057


namespace NUMINAMATH_CALUDE_orange_boxes_total_l280_28051

theorem orange_boxes_total (box1_capacity box2_capacity box3_capacity : ℕ)
  (box1_fill box2_fill box3_fill : ℚ) :
  box1_capacity = 80 →
  box2_capacity = 50 →
  box3_capacity = 60 →
  box1_fill = 3/4 →
  box2_fill = 3/5 →
  box3_fill = 2/3 →
  (↑box1_capacity * box1_fill + ↑box2_capacity * box2_fill + ↑box3_capacity * box3_fill : ℚ) = 130 := by
  sorry

end NUMINAMATH_CALUDE_orange_boxes_total_l280_28051


namespace NUMINAMATH_CALUDE_gcd_binomial_integrality_l280_28026

theorem gcd_binomial_integrality (m n : ℕ) (h1 : 1 ≤ m) (h2 : m ≤ n) :
  ∃ (a b : ℤ), (Nat.gcd m n : ℚ) / n * Nat.choose n m = a * Nat.choose (n-1) (m-1) + b * Nat.choose n m := by
  sorry

end NUMINAMATH_CALUDE_gcd_binomial_integrality_l280_28026


namespace NUMINAMATH_CALUDE_max_value_of_a_plus_2b_l280_28085

/-- Two circles in a 2D plane -/
structure TwoCircles where
  a : ℝ
  b : ℝ
  circle1 : (x y : ℝ) → x^2 + y^2 + 2*a*x + a^2 - 4 = 0
  circle2 : (x y : ℝ) → x^2 + y^2 - 4*b*y - 1 + 4*b^2 = 0

/-- The property that two circles have exactly three common tangents -/
def have_three_common_tangents (c : TwoCircles) : Prop := sorry

/-- The theorem stating the maximum value of a+2b -/
theorem max_value_of_a_plus_2b (c : TwoCircles) 
  (h : have_three_common_tangents c) : 
  ∃ (max : ℝ), max = 3 * Real.sqrt 2 ∧ 
  ∀ (a b : ℝ), c.a = a → c.b = b → a + 2*b ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_a_plus_2b_l280_28085


namespace NUMINAMATH_CALUDE_min_distance_between_ellipses_l280_28073

theorem min_distance_between_ellipses (a b c d : ℝ) 
  (eq1 : 4 * a^2 + b^2 - 8*b + 12 = 0)
  (eq2 : c^2 - 8*c + 4*d^2 + 12 = 0) :
  ∃ (min_val : ℝ), 
    (∀ (x y : ℝ), (x - y)^2 + (b - d)^2 ≥ min_val) ∧ 
    min_val = 37 - 16 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_between_ellipses_l280_28073


namespace NUMINAMATH_CALUDE_red_cards_taken_out_l280_28063

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (h_total : total_cards = 52)
  (h_red : red_cards = total_cards / 2)

/-- Represents the state after some red cards were taken out -/
structure RemainingCards :=
  (remaining_red : ℕ)
  (h_remaining : remaining_red = 16)

theorem red_cards_taken_out (d : Deck) (r : RemainingCards) :
  d.red_cards - r.remaining_red = 10 := by
  sorry

end NUMINAMATH_CALUDE_red_cards_taken_out_l280_28063


namespace NUMINAMATH_CALUDE_max_min_values_l280_28000

-- Define the conditions
def positive_xy (x y : ℝ) : Prop := x > 0 ∧ y > 0
def constraint (x y : ℝ) : Prop := 3 * x + 2 * y = 10

-- Define the theorem
theorem max_min_values (x y : ℝ) 
  (h1 : positive_xy x y) (h2 : constraint x y) : 
  (∃ (m : ℝ), m = Real.sqrt (3 * x) + Real.sqrt (2 * y) ∧ 
    m ≤ 2 * Real.sqrt 5 ∧ 
    ∀ (x' y' : ℝ), positive_xy x' y' → constraint x' y' → 
      Real.sqrt (3 * x') + Real.sqrt (2 * y') ≤ m) ∧
  (∃ (n : ℝ), n = 3 / x + 2 / y ∧ 
    n ≥ 5 / 2 ∧ 
    ∀ (x' y' : ℝ), positive_xy x' y' → constraint x' y' → 
      3 / x' + 2 / y' ≥ n) :=
by sorry

end NUMINAMATH_CALUDE_max_min_values_l280_28000


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l280_28083

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℤ) + (⌈x⌉ : ℤ) = 7 ↔ 3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l280_28083


namespace NUMINAMATH_CALUDE_range_of_a_l280_28069

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 2^x - 2 > a^2 - 3*a) → a ∈ Set.Icc 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l280_28069


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l280_28087

theorem decimal_to_fraction : (2.24 : ℚ) = 56 / 25 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l280_28087


namespace NUMINAMATH_CALUDE_polynomial_simplification_l280_28044

theorem polynomial_simplification (x : ℝ) :
  5 - 3*x - 7*x^2 + 11 - 5*x + 9*x^2 - 13 + 7*x - 4*x^3 + 7*x^2 + 2*x^3 =
  3 - x + 9*x^2 - 2*x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l280_28044


namespace NUMINAMATH_CALUDE_circus_ticket_cost_l280_28028

/-- The cost of tickets at a circus --/
theorem circus_ticket_cost (cost_per_ticket : ℕ) (num_tickets : ℕ) (total_cost : ℕ) : 
  cost_per_ticket = 44 → num_tickets = 7 → total_cost = cost_per_ticket * num_tickets → total_cost = 308 := by
  sorry

end NUMINAMATH_CALUDE_circus_ticket_cost_l280_28028


namespace NUMINAMATH_CALUDE_pyramid_height_l280_28002

theorem pyramid_height (p q : ℝ) : 
  p > 0 ∧ q > 0 →
  3^2 + p^2 = 5^2 →
  (1/3) * (1/2 * 3 * p) * q = 12 →
  q = 6 :=
by sorry

end NUMINAMATH_CALUDE_pyramid_height_l280_28002


namespace NUMINAMATH_CALUDE_bridge_length_bridge_length_problem_l280_28005

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let distance_covered := train_speed_ms * crossing_time
  distance_covered - train_length

/-- The length of the bridge is 215 meters -/
theorem bridge_length_problem : bridge_length 160 45 30 = 215 := by
  sorry

end NUMINAMATH_CALUDE_bridge_length_bridge_length_problem_l280_28005


namespace NUMINAMATH_CALUDE_sum_of_powers_of_i_l280_28015

theorem sum_of_powers_of_i : 
  let i : ℂ := Complex.I
  (i + 2 * i^2 + 3 * i^3 + 4 * i^4 + 5 * i^5 + 6 * i^6 + 7 * i^7 + 8 * i^8) = (4 : ℂ) - 4 * i :=
by sorry

end NUMINAMATH_CALUDE_sum_of_powers_of_i_l280_28015


namespace NUMINAMATH_CALUDE_unique_solution_count_l280_28038

/-- A system of equations has exactly one solution -/
def has_unique_solution (k : ℝ) : Prop :=
  ∃! x y : ℝ, x^2 + y^2 = 2*k^2 ∧ k*x - y = 2*k

/-- The number of real values of k for which the system has a unique solution -/
theorem unique_solution_count :
  ∃ S : Finset ℝ, (∀ k : ℝ, k ∈ S ↔ has_unique_solution k) ∧ S.card = 3 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_count_l280_28038


namespace NUMINAMATH_CALUDE_evaluate_expression_l280_28024

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  4 * x^y + 5 * y^x = 76 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l280_28024


namespace NUMINAMATH_CALUDE_product_of_sums_equals_3280_l280_28004

theorem product_of_sums_equals_3280 :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equals_3280_l280_28004


namespace NUMINAMATH_CALUDE_sqrt_200_range_l280_28018

theorem sqrt_200_range : 14 < Real.sqrt 200 ∧ Real.sqrt 200 < 15 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_range_l280_28018


namespace NUMINAMATH_CALUDE_f_properties_l280_28076

/-- The function f(x) = a*ln(x) + b/x + c/(x^2) -/
noncomputable def f (a b c x : ℝ) : ℝ := a * Real.log x + b / x + c / (x^2)

/-- The statement that f has both a maximum and a minimum value -/
def has_max_and_min (f : ℝ → ℝ) : Prop := ∃ (x_max x_min : ℝ), ∀ x, f x ≤ f x_max ∧ f x_min ≤ f x

theorem f_properties (a b c : ℝ) (ha : a ≠ 0) 
  (h_max_min : has_max_and_min (f a b c)) :
  ab > 0 ∧ b^2 + 8*a*c > 0 ∧ a*c < 0 := by sorry

end NUMINAMATH_CALUDE_f_properties_l280_28076


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l280_28081

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (1/3)^2 + (1/4)^2 = (37*a/100/b) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt a / Real.sqrt b = 50/19 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_l280_28081


namespace NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l280_28078

-- Define the concept of a line in 3D space
structure Line3D where
  -- You might represent a line using a point and a direction vector
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

-- Define perpendicularity for lines
def perpendicular (l1 l2 : Line3D) : Prop :=
  -- Two lines are perpendicular if their direction vectors are orthogonal
  let (_, _, _) := l1.direction
  let (_, _, _) := l2.direction
  sorry

-- Define parallelism for lines
def parallel (l1 l2 : Line3D) : Prop :=
  -- Two lines are parallel if their direction vectors are scalar multiples of each other
  let (_, _, _) := l1.direction
  let (_, _, _) := l2.direction
  sorry

-- Theorem statement
theorem perpendicular_parallel_transitive (l1 l2 l3 : Line3D) :
  perpendicular l1 l2 → parallel l2 l3 → perpendicular l1 l3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_transitive_l280_28078


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l280_28034

/-- Represents the population sizes for each age group -/
structure Population :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Represents the sample sizes for each age group -/
structure Sample :=
  (elderly : ℕ)
  (middleAged : ℕ)
  (young : ℕ)

/-- Checks if the sample is proportional to the population -/
def isProportionalSample (pop : Population) (sam : Sample) (totalSample : ℕ) : Prop :=
  sam.elderly * (pop.elderly + pop.middleAged + pop.young) = pop.elderly * totalSample ∧
  sam.middleAged * (pop.elderly + pop.middleAged + pop.young) = pop.middleAged * totalSample ∧
  sam.young * (pop.elderly + pop.middleAged + pop.young) = pop.young * totalSample

theorem stratified_sampling_theorem (pop : Population) (sam : Sample) :
  pop.elderly = 27 →
  pop.middleAged = 54 →
  pop.young = 81 →
  sam.elderly + sam.middleAged + sam.young = 36 →
  isProportionalSample pop sam 36 →
  sam.elderly = 6 ∧ sam.middleAged = 12 ∧ sam.young = 18 := by
  sorry

#check stratified_sampling_theorem

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l280_28034


namespace NUMINAMATH_CALUDE_min_value_bisecting_line_l280_28062

/-- The minimum value of 1/a + 1/b for a line ax + by - 1 = 0 bisecting a specific circle -/
theorem min_value_bisecting_line (a b : ℝ) : 
  a * b > 0 → 
  (∃ x y : ℝ, a * x + b * y = 1 ∧ x^2 + y^2 - 2*x - 4*y + 1 = 0) →
  (∀ x y : ℝ, x^2 + y^2 - 2*x - 4*y + 1 = 0 → (a * x + b * y - 1) * (a * x + b * y - 1) ≤ (a^2 + b^2) * ((x-1)^2 + (y-2)^2)) →
  (1/a + 1/b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_bisecting_line_l280_28062


namespace NUMINAMATH_CALUDE_double_scientific_notation_l280_28077

theorem double_scientific_notation : 
  let x : ℝ := 1.2 * (10 ^ 6)
  2 * x = 2.4 * (10 ^ 6) := by sorry

end NUMINAMATH_CALUDE_double_scientific_notation_l280_28077


namespace NUMINAMATH_CALUDE_orphanage_donation_l280_28086

theorem orphanage_donation (total donation1 donation3 : ℚ) 
  (h1 : total = 650)
  (h2 : donation1 = 175)
  (h3 : donation3 = 250) :
  total - donation1 - donation3 = 225 := by
  sorry

end NUMINAMATH_CALUDE_orphanage_donation_l280_28086


namespace NUMINAMATH_CALUDE_subsequence_appears_l280_28040

/-- Defines the sequence where each digit after the first four is the last digit of the sum of the previous four digits -/
def digit_sequence : ℕ → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| 3 => 4
| n + 4 => (digit_sequence n + digit_sequence (n + 1) + digit_sequence (n + 2) + digit_sequence (n + 3)) % 10

/-- Checks if the subsequence 8123 appears starting at position n in the sequence -/
def appears_at (n : ℕ) : Prop :=
  digit_sequence n = 8 ∧
  digit_sequence (n + 1) = 1 ∧
  digit_sequence (n + 2) = 2 ∧
  digit_sequence (n + 3) = 3

/-- Theorem stating that the subsequence 8123 appears in the sequence -/
theorem subsequence_appears : ∃ n : ℕ, appears_at n := by
  sorry

end NUMINAMATH_CALUDE_subsequence_appears_l280_28040


namespace NUMINAMATH_CALUDE_brandon_skittles_proof_l280_28093

def brandon_initial_skittles (skittles_lost : ℕ) (final_skittles : ℕ) : ℕ :=
  final_skittles + skittles_lost

theorem brandon_skittles_proof :
  brandon_initial_skittles 9 87 = 96 :=
by sorry

end NUMINAMATH_CALUDE_brandon_skittles_proof_l280_28093


namespace NUMINAMATH_CALUDE_max_radius_difference_l280_28068

/-- The ellipse Γ in a 2D coordinate system -/
def Γ : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The first quadrant -/
def firstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 > 0}

/-- Point P on the ellipse Γ in the first quadrant -/
def P : (ℝ × ℝ) :=
  sorry

/-- Left focus F₁ of the ellipse Γ -/
def F₁ : (ℝ × ℝ) :=
  sorry

/-- Right focus F₂ of the ellipse Γ -/
def F₂ : (ℝ × ℝ) :=
  sorry

/-- Point Q₁ where extended PF₁ intersects Γ -/
def Q₁ : (ℝ × ℝ) :=
  sorry

/-- Point Q₂ where extended PF₂ intersects Γ -/
def Q₂ : (ℝ × ℝ) :=
  sorry

/-- Radius r₁ of the inscribed circle in triangle PF₁Q₂ -/
def r₁ : ℝ :=
  sorry

/-- Radius r₂ of the inscribed circle in triangle PF₂Q₁ -/
def r₂ : ℝ :=
  sorry

/-- Theorem stating that the maximum value of r₁ - r₂ is 1/3 -/
theorem max_radius_difference :
  P ∈ Γ ∩ firstQuadrant →
  ∃ (max : ℝ), max = (1 : ℝ) / 3 ∧ ∀ (p : ℝ × ℝ), p ∈ Γ ∩ firstQuadrant → r₁ - r₂ ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_radius_difference_l280_28068


namespace NUMINAMATH_CALUDE_cats_count_pet_store_cats_l280_28071

/-- Given a ratio of cats to dogs and the number of dogs, calculate the number of cats -/
theorem cats_count (cat_ratio : ℕ) (dog_ratio : ℕ) (dog_count : ℕ) : ℕ :=
  (cat_ratio * dog_count) / dog_ratio

/-- Prove that with a cat to dog ratio of 3:4 and 20 dogs, there are 15 cats -/
theorem pet_store_cats : cats_count 3 4 20 = 15 := by
  sorry

end NUMINAMATH_CALUDE_cats_count_pet_store_cats_l280_28071


namespace NUMINAMATH_CALUDE_natural_number_equations_l280_28011

theorem natural_number_equations :
  (∃! (x : ℕ), 2^(x-5) = 2) ∧
  (∃! (x : ℕ), 2^x = 512) ∧
  (∃! (x : ℕ), x^5 = 243) ∧
  (∃! (x : ℕ), x^4 = 625) :=
by
  sorry

end NUMINAMATH_CALUDE_natural_number_equations_l280_28011


namespace NUMINAMATH_CALUDE_cut_rectangle_decreases_area_and_perimeter_l280_28058

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

-- Define the area and perimeter functions
def area (r : Rectangle) : ℝ := r.length * r.width
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

-- State the theorem
theorem cut_rectangle_decreases_area_and_perimeter 
  (R : Rectangle) 
  (S : Rectangle) 
  (h_cut : S.length ≤ R.length ∧ S.width ≤ R.width) 
  (h_proper_subset : S.length < R.length ∨ S.width < R.width) : 
  area S < area R ∧ perimeter S < perimeter R := by
  sorry

end NUMINAMATH_CALUDE_cut_rectangle_decreases_area_and_perimeter_l280_28058


namespace NUMINAMATH_CALUDE_equal_product_grouping_l280_28072

theorem equal_product_grouping (numbers : Finset ℕ) 
  (h_numbers : numbers = {12, 30, 42, 44, 57, 91, 95, 143}) :
  (12 * 42 * 95 * 143 : ℕ) = (30 * 44 * 57 * 91 : ℕ) := by
  sorry

end NUMINAMATH_CALUDE_equal_product_grouping_l280_28072


namespace NUMINAMATH_CALUDE_paper_remaining_l280_28054

theorem paper_remaining (total : ℕ) (used : ℕ) (h1 : total = 900) (h2 : used = 156) :
  total - used = 744 := by
  sorry

end NUMINAMATH_CALUDE_paper_remaining_l280_28054


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l280_28035

def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum :
  (interior_sum 4 = 6) →
  (interior_sum 5 = 14) →
  (∀ k ≥ 3, interior_sum k = 2^(k-1) - 2) →
  interior_sum 9 = 254 :=
by sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l280_28035


namespace NUMINAMATH_CALUDE_average_first_21_multiples_of_5_l280_28047

theorem average_first_21_multiples_of_5 : 
  let multiples := (fun i => 5 * i) 
  let sum := (List.range 21).map multiples |>.sum
  sum / 21 = 55 := by
sorry


end NUMINAMATH_CALUDE_average_first_21_multiples_of_5_l280_28047


namespace NUMINAMATH_CALUDE_power_difference_equality_l280_28045

theorem power_difference_equality (x : ℝ) (h : x - 1/x = Real.sqrt 2) :
  x^1023 - 1/x^1023 = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_equality_l280_28045


namespace NUMINAMATH_CALUDE_cylindrical_tank_volume_increase_l280_28096

theorem cylindrical_tank_volume_increase (R H : ℝ) (hR : R = 10) (hH : H = 5) :
  ∃ k : ℝ, k > 0 ∧
  (π * (k * R)^2 * H - π * R^2 * H = π * R^2 * (H + k) - π * R^2 * H) ∧
  k = (1 + Real.sqrt 101) / 10 := by
sorry

end NUMINAMATH_CALUDE_cylindrical_tank_volume_increase_l280_28096


namespace NUMINAMATH_CALUDE_max_store_visits_l280_28061

theorem max_store_visits (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (double_visitors : ℕ) (h1 : total_stores = 8) (h2 : total_visits = 23) 
  (h3 : total_shoppers = 12) (h4 : double_visitors = 8) 
  (h5 : double_visitors ≤ total_shoppers) 
  (h6 : double_visitors * 2 + (total_shoppers - double_visitors) ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visits :=
by sorry

end NUMINAMATH_CALUDE_max_store_visits_l280_28061


namespace NUMINAMATH_CALUDE_plan_y_more_cost_effective_l280_28013

/-- Cost of Plan X in cents for m megabytes -/
def cost_x (m : ℕ) : ℕ := 15 * m

/-- Cost of Plan Y in cents for m megabytes -/
def cost_y (m : ℕ) : ℕ := 2500 + 7 * m

/-- The minimum whole number of megabytes for Plan Y to be more cost-effective than Plan X -/
def min_megabytes : ℕ := 313

theorem plan_y_more_cost_effective :
  ∀ m : ℕ, m ≥ min_megabytes → cost_y m < cost_x m ∧
  ∀ n : ℕ, n < min_megabytes → cost_y n ≥ cost_x n :=
by sorry

end NUMINAMATH_CALUDE_plan_y_more_cost_effective_l280_28013


namespace NUMINAMATH_CALUDE_raja_income_proof_l280_28031

/-- Raja's monthly income in rupees -/
def monthly_income : ℝ := 25000

/-- The amount Raja saves in rupees -/
def savings : ℝ := 5000

/-- Percentage of income spent on household items -/
def household_percentage : ℝ := 0.60

/-- Percentage of income spent on clothes -/
def clothes_percentage : ℝ := 0.10

/-- Percentage of income spent on medicines -/
def medicine_percentage : ℝ := 0.10

theorem raja_income_proof :
  monthly_income * household_percentage +
  monthly_income * clothes_percentage +
  monthly_income * medicine_percentage +
  savings = monthly_income :=
by sorry

end NUMINAMATH_CALUDE_raja_income_proof_l280_28031


namespace NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l280_28050

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (is_arithmetic : arithmetic_sequence a)
  (first_term : a 1 = 2)
  (sum_of_three : a 1 + a 2 + a 3 = 12) :
  a 6 = 12 := by
sorry

end NUMINAMATH_CALUDE_sixth_term_of_arithmetic_sequence_l280_28050


namespace NUMINAMATH_CALUDE_min_value_trig_function_l280_28001

open Real

theorem min_value_trig_function (θ : Real) (h₁ : θ > 0) (h₂ : θ < π / 2) :
  ∀ y : Real, y = 1 / (sin θ)^2 + 9 / (cos θ)^2 → y ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trig_function_l280_28001


namespace NUMINAMATH_CALUDE_certain_number_proof_l280_28042

theorem certain_number_proof (n : ℕ) : 
  n % 10 = 6 ∧ 1442 % 10 = 12 → n = 1446 :=
by sorry

end NUMINAMATH_CALUDE_certain_number_proof_l280_28042


namespace NUMINAMATH_CALUDE_log_inequality_l280_28027

theorem log_inequality (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  let a : ℝ := (Real.sqrt 2 + 1) / 2
  let f : ℝ → ℝ := fun x ↦ Real.log x / Real.log a
  f m > f n → m > n := by sorry

end NUMINAMATH_CALUDE_log_inequality_l280_28027


namespace NUMINAMATH_CALUDE_square_area_ratio_l280_28043

theorem square_area_ratio (side_c side_d : ℝ) (h1 : side_c = 45) (h2 : side_d = 60) :
  (side_c^2) / (side_d^2) = 9 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l280_28043


namespace NUMINAMATH_CALUDE_vector_projection_and_collinearity_l280_28067

def a : Fin 3 → ℚ := ![2, 2, -1]
def b : Fin 3 → ℚ := ![-1, 4, 3]
def p : Fin 3 → ℚ := ![40/29, 64/29, 17/29]

theorem vector_projection_and_collinearity :
  (∀ i : Fin 3, (a i - p i) • (b - a) = 0) ∧
  (∀ i : Fin 3, (b i - p i) • (b - a) = 0) ∧
  ∃ t : ℚ, ∀ i : Fin 3, p i = a i + t * (b i - a i) := by
  sorry

end NUMINAMATH_CALUDE_vector_projection_and_collinearity_l280_28067


namespace NUMINAMATH_CALUDE_product_mod_fifteen_l280_28092

theorem product_mod_fifteen : 59 * 67 * 78 ≡ 9 [ZMOD 15] := by sorry

end NUMINAMATH_CALUDE_product_mod_fifteen_l280_28092


namespace NUMINAMATH_CALUDE_martin_ice_cream_cost_l280_28022

/-- Represents the cost of ice cream scoops in dollars -/
structure IceCreamPrices where
  kiddie : ℕ
  regular : ℕ
  double : ℕ

/-- Represents the Martin family's ice cream order -/
structure MartinOrder where
  regular : ℕ
  kiddie : ℕ
  double : ℕ

/-- Calculates the total cost of the Martin family's ice cream order -/
def calculateTotalCost (prices : IceCreamPrices) (order : MartinOrder) : ℕ :=
  prices.regular * order.regular +
  prices.kiddie * order.kiddie +
  prices.double * order.double

/-- Theorem stating that the total cost for the Martin family's ice cream order is $32 -/
theorem martin_ice_cream_cost :
  ∃ (prices : IceCreamPrices) (order : MartinOrder),
    prices.kiddie = 3 ∧
    prices.regular = 4 ∧
    prices.double = 6 ∧
    order.regular = 2 ∧
    order.kiddie = 2 ∧
    order.double = 3 ∧
    calculateTotalCost prices order = 32 :=
  sorry

end NUMINAMATH_CALUDE_martin_ice_cream_cost_l280_28022


namespace NUMINAMATH_CALUDE_square_floor_tiles_l280_28007

/-- Represents a square floor covered with tiles -/
structure TiledFloor where
  side_length : ℕ
  is_even : Even side_length
  diagonal_tiles : ℕ
  h_diagonal : diagonal_tiles = 2 * side_length

/-- The total number of tiles on a square floor -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.side_length ^ 2

theorem square_floor_tiles (floor : TiledFloor) 
  (h_diagonal_count : floor.diagonal_tiles = 88) : 
  total_tiles floor = 1936 := by
  sorry

end NUMINAMATH_CALUDE_square_floor_tiles_l280_28007


namespace NUMINAMATH_CALUDE_simplify_expression_l280_28029

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) : x^3 * (y^3 / x)^2 = x * y^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l280_28029


namespace NUMINAMATH_CALUDE_f_derivative_l280_28053

noncomputable def f (x : ℝ) : ℝ := Real.log (5 * x + Real.sqrt (25 * x^2 + 1)) - Real.sqrt (25 * x^2 + 1) * Real.arctan (5 * x)

theorem f_derivative (x : ℝ) : 
  deriv f x = -(25 * x * Real.arctan (5 * x)) / Real.sqrt (25 * x^2 + 1) :=
by sorry

end NUMINAMATH_CALUDE_f_derivative_l280_28053


namespace NUMINAMATH_CALUDE_monica_subjects_l280_28036

theorem monica_subjects (monica marius millie : ℕ) 
  (h1 : millie = marius + 3)
  (h2 : marius = monica + 4)
  (h3 : monica + marius + millie = 41) : 
  monica = 10 :=
sorry

end NUMINAMATH_CALUDE_monica_subjects_l280_28036


namespace NUMINAMATH_CALUDE_min_value_theorem_l280_28060

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y = 1 → a*(b - 1) ≤ x*(y - 1) ∧ 
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 2/y = 1 ∧ x*(y - 1) = 3 + 2*Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l280_28060


namespace NUMINAMATH_CALUDE_rectangular_field_with_pond_l280_28082

theorem rectangular_field_with_pond (l w : ℝ) : 
  l = 2 * w →                        -- length is double the width
  l * w = 2 * (8 * 8) →              -- pond area is half of field area
  l = 16 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_with_pond_l280_28082


namespace NUMINAMATH_CALUDE_trig_problem_l280_28046

theorem trig_problem (α β : Real) 
  (h1 : Real.cos (α - β/2) = -2 * Real.sqrt 7 / 7)
  (h2 : Real.sin (α/2 - β) = 1/2)
  (h3 : π/2 < α ∧ α < π)
  (h4 : 0 < β ∧ β < π/2) :
  Real.cos ((α + β)/2) = -Real.sqrt 21 / 14 ∧ 
  Real.tan (α + β) = 5 * Real.sqrt 3 / 11 := by
sorry

end NUMINAMATH_CALUDE_trig_problem_l280_28046


namespace NUMINAMATH_CALUDE_zachary_pushups_count_l280_28006

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 51

/-- The number of push-ups David did -/
def david_pushups : ℕ := zachary_pushups + 22

/-- The number of push-ups John did -/
def john_pushups : ℕ := 69

theorem zachary_pushups_count : zachary_pushups = 51 := by
  have h1 : david_pushups = zachary_pushups + 22 := rfl
  have h2 : john_pushups = david_pushups - 4 := by sorry
  have h3 : john_pushups = 69 := rfl
  sorry

end NUMINAMATH_CALUDE_zachary_pushups_count_l280_28006


namespace NUMINAMATH_CALUDE_lawrence_county_kids_count_l280_28030

/-- The number of kids staying home during summer break in Lawrence county -/
def kids_staying_home : ℕ := 907611

/-- The number of kids going to camp from Lawrence county -/
def kids_going_to_camp : ℕ := 455682

/-- The total number of kids in Lawrence county -/
def total_kids : ℕ := kids_staying_home + kids_going_to_camp

/-- Theorem stating that the total number of kids in Lawrence county
    is equal to the sum of kids staying home and kids going to camp -/
theorem lawrence_county_kids_count :
  total_kids = kids_staying_home + kids_going_to_camp := by
  sorry

end NUMINAMATH_CALUDE_lawrence_county_kids_count_l280_28030


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l280_28041

theorem least_integer_absolute_value (x : ℤ) : 
  (∀ y : ℤ, y < x → ¬(|2 * y + 3| ≤ 12)) ∧ (|2 * x + 3| ≤ 12) → x = -7 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l280_28041


namespace NUMINAMATH_CALUDE_expression_evaluation_l280_28037

theorem expression_evaluation : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l280_28037


namespace NUMINAMATH_CALUDE_remainder_three_power_twentyfour_mod_seven_l280_28033

theorem remainder_three_power_twentyfour_mod_seven :
  3^24 % 7 = 1 := by
sorry

end NUMINAMATH_CALUDE_remainder_three_power_twentyfour_mod_seven_l280_28033


namespace NUMINAMATH_CALUDE_max_portions_is_two_l280_28019

/-- Represents the number of bags for each ingredient -/
structure Ingredients :=
  (nuts : ℕ)
  (dried_fruit : ℕ)
  (chocolate : ℕ)
  (coconut : ℕ)

/-- Represents the ratio of ingredients in each portion -/
structure Ratio :=
  (nuts : ℕ)
  (dried_fruit : ℕ)
  (chocolate : ℕ)
  (coconut : ℕ)

/-- Calculates the maximum number of portions that can be made -/
def max_portions (ingredients : Ingredients) (ratio : Ratio) : ℕ :=
  min (ingredients.nuts / ratio.nuts)
      (min (ingredients.dried_fruit / ratio.dried_fruit)
           (min (ingredients.chocolate / ratio.chocolate)
                (ingredients.coconut / ratio.coconut)))

/-- Proves that the maximum number of portions is 2 -/
theorem max_portions_is_two :
  let ingredients := Ingredients.mk 16 6 8 4
  let ratio := Ratio.mk 4 3 2 1
  max_portions ingredients ratio = 2 :=
by
  sorry

#eval max_portions (Ingredients.mk 16 6 8 4) (Ratio.mk 4 3 2 1)

end NUMINAMATH_CALUDE_max_portions_is_two_l280_28019


namespace NUMINAMATH_CALUDE_bus_purchase_problem_l280_28065

/-- Represents the cost and capacity of a bus type -/
structure BusType where
  cost : ℕ
  capacity : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

def totalBuses : ℕ := 10

def scenario1Cost : ℕ := 380
def scenario2Cost : ℕ := 360

def maxTotalCost : ℕ := 880
def minTotalPassengers : ℕ := 5200000

theorem bus_purchase_problem 
  (typeA typeB : BusType)
  (plans : List PurchasePlan)
  (bestPlan : PurchasePlan)
  (minCost : ℕ) :
  (typeA.cost + 3 * typeB.cost = scenario1Cost) →
  (2 * typeA.cost + 2 * typeB.cost = scenario2Cost) →
  (typeA.capacity = 500000) →
  (typeB.capacity = 600000) →
  (∀ plan ∈ plans, 
    plan.typeA + plan.typeB = totalBuses ∧
    plan.typeA * typeA.cost + plan.typeB * typeB.cost ≤ maxTotalCost ∧
    plan.typeA * typeA.capacity + plan.typeB * typeB.capacity ≥ minTotalPassengers) →
  (bestPlan ∈ plans) →
  (∀ plan ∈ plans, 
    plan.typeA * typeA.cost + plan.typeB * typeB.cost ≥ 
    bestPlan.typeA * typeA.cost + bestPlan.typeB * typeB.cost) →
  (minCost = bestPlan.typeA * typeA.cost + bestPlan.typeB * typeB.cost) →
  typeA.cost = 80 ∧ 
  typeB.cost = 100 ∧
  plans = [⟨6, 4⟩, ⟨7, 3⟩, ⟨8, 2⟩] ∧
  bestPlan = ⟨8, 2⟩ ∧
  minCost = 840 := by
  sorry

end NUMINAMATH_CALUDE_bus_purchase_problem_l280_28065


namespace NUMINAMATH_CALUDE_trig_identity_l280_28023

theorem trig_identity (α β : ℝ) : 
  Real.sin (2 * α) ^ 2 + Real.sin β ^ 2 + Real.cos (2 * α + β) * Real.cos (2 * α - β) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l280_28023


namespace NUMINAMATH_CALUDE_intersection_complement_eq_set_l280_28003

def R : Set ℝ := Set.univ

def M : Set ℝ := {-1, 1, 2, 4}

def N : Set ℝ := {x : ℝ | x^2 - 2*x > 3}

theorem intersection_complement_eq_set : M ∩ (R \ N) = {-1, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_eq_set_l280_28003


namespace NUMINAMATH_CALUDE_odd_functions_properties_l280_28055

-- Define the functions f and g
def f (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 + x - k
def g (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem odd_functions_properties :
  (∀ x, g (-x) = -g x) ∧  -- g is an odd function
  (g 1 = -2) ∧  -- g achieves minimum -2 at x = 1
  (∀ x, g x ≤ 2) ∧  -- maximum value of g is 2
  (∀ k, (∀ x ∈ Set.Icc (-1) 3, f k x ≤ g x) → k ≥ 8) ∧  -- range of k for f ≤ g on [-1,3]
  (∀ k, (∀ x₁ ∈ Set.Icc (-1) 3, ∀ x₂ ∈ Set.Icc (-1) 3, f k x₁ ≤ g x₂) → k ≥ 23)  -- range of k for f(x₁) ≤ g(x₂)
  := by sorry

end NUMINAMATH_CALUDE_odd_functions_properties_l280_28055


namespace NUMINAMATH_CALUDE_bingley_bracelets_l280_28014

theorem bingley_bracelets (initial : ℕ) : 
  let kellys_bracelets : ℕ := 16
  let received : ℕ := kellys_bracelets / 4
  let total : ℕ := initial + received
  let given_away : ℕ := total / 3
  let remaining : ℕ := total - given_away
  remaining = 6 → initial = 5 := by sorry

end NUMINAMATH_CALUDE_bingley_bracelets_l280_28014


namespace NUMINAMATH_CALUDE_children_left_on_bus_l280_28094

theorem children_left_on_bus (initial_children : Nat) (children_off : Nat) : 
  initial_children = 43 → children_off = 22 → initial_children - children_off = 21 := by
  sorry

end NUMINAMATH_CALUDE_children_left_on_bus_l280_28094


namespace NUMINAMATH_CALUDE_total_soaking_time_l280_28059

/-- Calculates the total soaking time for clothes with grass and marinara stains. -/
theorem total_soaking_time
  (grass_stain_time : ℕ)
  (marinara_stain_time : ℕ)
  (grass_stains : ℕ)
  (marinara_stains : ℕ)
  (h1 : grass_stain_time = 4)
  (h2 : marinara_stain_time = 7)
  (h3 : grass_stains = 3)
  (h4 : marinara_stains = 1) :
  grass_stain_time * grass_stains + marinara_stain_time * marinara_stains = 19 :=
by sorry

#check total_soaking_time

end NUMINAMATH_CALUDE_total_soaking_time_l280_28059


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l280_28090

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l280_28090


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_circle_intersection_chord_length_l280_28098

/-- The length of the chord formed by the intersection of an asymptote of a hyperbola with a specific circle -/
theorem hyperbola_asymptote_circle_intersection_chord_length 
  (a b : ℝ) (h_a : a > 0) (h_b : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let e := Real.sqrt 5  -- eccentricity
  let circle := {(x, y) : ℝ × ℝ | (x - 2)^2 + (y - 3)^2 = 1}
  let asymptote := {(x, y) : ℝ × ℝ | y = (b / a) * x ∨ y = -(b / a) * x}
  ∀ (A B : ℝ × ℝ), A ∈ circle → B ∈ circle → A ∈ asymptote → B ∈ asymptote →
  e^2 = 1 + b^2 / a^2 →
  ‖A - B‖ = 4 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_circle_intersection_chord_length_l280_28098


namespace NUMINAMATH_CALUDE_line_slope_at_minimum_l280_28021

/-- Given a line ax - by + 2 = 0 (a > 0, b > 0) passing through (-1, 2),
    the slope is 2 when 2/a + 1/b is minimized. -/
theorem line_slope_at_minimum (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 2*b = 2) →
  (∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 2 → 2/x + 1/y ≥ 2/a + 1/b) →
  b/a = 2 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_at_minimum_l280_28021


namespace NUMINAMATH_CALUDE_triangle_ratio_l280_28010

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if 2b*sin(2A) = 3a*sin(B) and c = 2b, then a/b = √2 -/
theorem triangle_ratio (a b c : ℝ) (A B C : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  2 * b * Real.sin (2 * A) = 3 * a * Real.sin B →
  c = 2 * b →
  a / b = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_ratio_l280_28010


namespace NUMINAMATH_CALUDE_product_sale_result_l280_28070

def cost_price : ℝ := 100
def markup_percentage : ℝ := 0.2
def discount_percentage : ℝ := 0.2
def final_selling_price : ℝ := 96

theorem product_sale_result :
  let initial_price := cost_price * (1 + markup_percentage)
  let discounted_price := initial_price * (1 - discount_percentage)
  discounted_price = final_selling_price ∧ 
  cost_price - final_selling_price = 4 := by
sorry

end NUMINAMATH_CALUDE_product_sale_result_l280_28070


namespace NUMINAMATH_CALUDE_susie_earnings_l280_28020

def slice_price : ℕ := 3
def whole_pizza_price : ℕ := 15
def slices_sold : ℕ := 24
def whole_pizzas_sold : ℕ := 3

theorem susie_earnings : 
  slice_price * slices_sold + whole_pizza_price * whole_pizzas_sold = 117 := by
  sorry

end NUMINAMATH_CALUDE_susie_earnings_l280_28020


namespace NUMINAMATH_CALUDE_exists_q_no_zeros_in_decimal_l280_28016

theorem exists_q_no_zeros_in_decimal : ∃ q : ℚ, ∃ a : ℕ, (
  (q * 2^1000 = a) ∧
  (∀ d : ℕ, d < 10 → (a.digits 10).contains d → (d = 1 ∨ d = 2))
) := by sorry

end NUMINAMATH_CALUDE_exists_q_no_zeros_in_decimal_l280_28016


namespace NUMINAMATH_CALUDE_simplify_expression_l280_28084

theorem simplify_expression (x : ℝ) : (2*x + 20) + (150*x + 20) = 152*x + 40 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l280_28084


namespace NUMINAMATH_CALUDE_q_is_false_l280_28075

theorem q_is_false (h1 : ¬(p ∧ q)) (h2 : ¬¬p) : ¬q :=
sorry

end NUMINAMATH_CALUDE_q_is_false_l280_28075


namespace NUMINAMATH_CALUDE_total_amount_is_3200_l280_28079

/-- Proves that the total amount of money divided into two parts is 3200, given the problem conditions. -/
theorem total_amount_is_3200 
  (total : ℝ) -- Total amount of money
  (part1 : ℝ) -- First part of money (invested at 3%)
  (part2 : ℝ) -- Second part of money (invested at 5%)
  (h1 : part1 = 800) -- First part is Rs 800
  (h2 : part2 = total - part1) -- Second part is the remainder
  (h3 : 0.03 * part1 + 0.05 * part2 = 144) -- Total interest is Rs 144
  : total = 3200 :=
by sorry

end NUMINAMATH_CALUDE_total_amount_is_3200_l280_28079


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l280_28089

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (l : Line) (α β : Plane) (h1 : α ≠ β) :
  perpendicular l α → perpendicular l β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l280_28089


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l280_28017

def polynomial (x m : ℝ) : ℝ := 3 * x^2 - 9 * x + m

theorem polynomial_divisibility (m : ℝ) : 
  (∃ q : ℝ → ℝ, ∀ x, polynomial x m = (x - 2) * q x) ↔ m = 6 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l280_28017


namespace NUMINAMATH_CALUDE_circle_segment_area_l280_28009

theorem circle_segment_area (R : ℝ) (R_pos : R > 0) : 
  let circle_area := π * R^2
  let square_side := R * Real.sqrt 2
  let square_area := square_side^2
  let segment_area := (circle_area - square_area) / 4
  segment_area = R^2 * (π - 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_circle_segment_area_l280_28009
