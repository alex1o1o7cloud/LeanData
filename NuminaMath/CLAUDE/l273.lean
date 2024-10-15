import Mathlib

namespace NUMINAMATH_CALUDE_goldbach_2024_l273_27324

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem goldbach_2024 : ∃ p q : ℕ, 
  p + q = 2024 ∧ 
  is_prime p ∧ 
  is_prime q ∧ 
  (p > 1000 ∨ q > 1000) :=
sorry

end NUMINAMATH_CALUDE_goldbach_2024_l273_27324


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l273_27389

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence : arithmetic_sequence 3 4 15 = 59 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l273_27389


namespace NUMINAMATH_CALUDE_josh_marbles_difference_l273_27351

theorem josh_marbles_difference (initial_marbles : ℕ) (lost_marbles : ℕ) (found_marbles : ℕ) 
  (num_friends : ℕ) (marbles_per_friend : ℕ) : 
  initial_marbles = 85 → 
  lost_marbles = 46 → 
  found_marbles = 130 → 
  num_friends = 12 → 
  marbles_per_friend = 3 → 
  found_marbles - (lost_marbles + num_friends * marbles_per_friend) = 48 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_difference_l273_27351


namespace NUMINAMATH_CALUDE_min_sum_of_squares_l273_27396

theorem min_sum_of_squares (a b c d : ℝ) (h : a + 2*b + 3*c + 4*d = 12) :
  a^2 + b^2 + c^2 + d^2 ≥ 24/5 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_squares_l273_27396


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l273_27335

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l273_27335


namespace NUMINAMATH_CALUDE_repeating_decimal_subtraction_l273_27366

theorem repeating_decimal_subtraction :
  ∃ (a b c : ℚ),
    (1000 * a - a = 567) ∧
    (1000 * b - b = 234) ∧
    (1000 * c - c = 345) ∧
    (a - b - c = -4 / 333) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_subtraction_l273_27366


namespace NUMINAMATH_CALUDE_sales_price_calculation_l273_27355

theorem sales_price_calculation (C S : ℝ) : 
  S - C = 1.25 * C →  -- Gross profit is 125% of the cost
  S - C = 30 →        -- Gross profit is $30
  S = 54 :=           -- Sales price is $54
by sorry

end NUMINAMATH_CALUDE_sales_price_calculation_l273_27355


namespace NUMINAMATH_CALUDE_graduation_messages_l273_27323

/-- 
Proves that for a class with x students, where each student writes a message 
for every other student, and the total number of messages is 930, 
the equation x(x-1) = 930 holds true.
-/
theorem graduation_messages (x : ℕ) 
  (h1 : x > 0) 
  (h2 : ∀ (s : ℕ), s < x → s.succ ≤ x → x - 1 = (x - s.succ) + s) 
  (h3 : (x * (x - 1)) = 930) : 
  x * (x - 1) = 930 := by
  sorry

end NUMINAMATH_CALUDE_graduation_messages_l273_27323


namespace NUMINAMATH_CALUDE_units_digit_is_seven_l273_27343

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : ℕ
  tens : ℕ
  units : ℕ
  hundreds_lt_10 : hundreds < 10
  tens_lt_10 : tens < 10
  units_lt_10 : units < 10
  hundreds_gt_0 : hundreds > 0

/-- The value of a three-digit number -/
def ThreeDigitNumber.value (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- The reversed value of a three-digit number -/
def ThreeDigitNumber.reversed_value (n : ThreeDigitNumber) : ℕ :=
  100 * n.units + 10 * n.tens + n.hundreds

/-- Theorem stating the units digit of the result is 7 -/
theorem units_digit_is_seven (n : ThreeDigitNumber) 
    (h : n.hundreds = n.units + 3) : 
    (n.reversed_value - 2 * n.value) % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_is_seven_l273_27343


namespace NUMINAMATH_CALUDE_janet_lives_calculation_l273_27353

theorem janet_lives_calculation (initial_lives current_lives gained_lives : ℕ) :
  initial_lives = 47 →
  current_lives = initial_lives - 23 →
  gained_lives = 46 →
  current_lives + gained_lives = 70 := by
  sorry

end NUMINAMATH_CALUDE_janet_lives_calculation_l273_27353


namespace NUMINAMATH_CALUDE_distance_of_opposite_numbers_a_and_neg_a_are_opposite_l273_27333

-- Define the concept of opposite numbers
def are_opposite (a b : ℝ) : Prop := a = -b

-- Define the distance from origin to a point on the number line
def distance_from_origin (a : ℝ) : ℝ := |a|

-- Statement 1: The distance from the origin to the points corresponding to two opposite numbers on the number line is equal
theorem distance_of_opposite_numbers (a : ℝ) : 
  distance_from_origin a = distance_from_origin (-a) := by sorry

-- Statement 2: For any real number a, a and -a are opposite numbers to each other
theorem a_and_neg_a_are_opposite (a : ℝ) : 
  are_opposite a (-a) := by sorry

end NUMINAMATH_CALUDE_distance_of_opposite_numbers_a_and_neg_a_are_opposite_l273_27333


namespace NUMINAMATH_CALUDE_definite_integral_tangent_fraction_l273_27339

theorem definite_integral_tangent_fraction : 
  ∫ x in (0)..(π/4), (4 - 7 * Real.tan x) / (2 + 3 * Real.tan x) = Real.log (25/8) - π/4 := by
  sorry

end NUMINAMATH_CALUDE_definite_integral_tangent_fraction_l273_27339


namespace NUMINAMATH_CALUDE_james_potato_problem_l273_27303

/-- The problem of calculating the number of people James made potatoes for. -/
theorem james_potato_problem (pounds_per_person : ℝ) (bag_weight : ℝ) (bag_cost : ℝ) (total_spent : ℝ) :
  pounds_per_person = 1.5 →
  bag_weight = 20 →
  bag_cost = 5 →
  total_spent = 15 →
  (total_spent / bag_cost) * bag_weight / pounds_per_person = 40 := by
  sorry

#check james_potato_problem

end NUMINAMATH_CALUDE_james_potato_problem_l273_27303


namespace NUMINAMATH_CALUDE_angle_D_measure_l273_27387

-- Define the hexagon and its angles
structure ConvexHexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the properties of the hexagon
def is_valid_hexagon (h : ConvexHexagon) : Prop :=
  h.A > 0 ∧ h.B > 0 ∧ h.C > 0 ∧ h.D > 0 ∧ h.E > 0 ∧ h.F > 0 ∧
  h.A + h.B + h.C + h.D + h.E + h.F = 720

-- Define the conditions of the problem
def satisfies_conditions (h : ConvexHexagon) : Prop :=
  h.A = h.B ∧ h.B = h.C ∧
  h.D = h.E ∧ h.E = h.F ∧
  h.A + 30 = h.D

-- Theorem statement
theorem angle_D_measure (h : ConvexHexagon) 
  (h_valid : is_valid_hexagon h) 
  (h_cond : satisfies_conditions h) : 
  h.D = 135 :=
sorry

end NUMINAMATH_CALUDE_angle_D_measure_l273_27387


namespace NUMINAMATH_CALUDE_problem_solution_l273_27315

theorem problem_solution (x : ℕ+) (y : ℚ) 
  (h1 : x = 11 * y + 4)
  (h2 : 2 * x = 24 * y + 3) : 
  13 * y - x = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l273_27315


namespace NUMINAMATH_CALUDE_ellipse_equation_l273_27344

theorem ellipse_equation (a b : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : a > b) :
  ∃ (x y : ℝ), x^2 / 25 + y^2 / 36 = 1 ∧ 
  ∀ (u v : ℝ), (u^2 / b^2 + v^2 / a^2 = 1 ↔ x^2 / 25 + y^2 / 36 = 1) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l273_27344


namespace NUMINAMATH_CALUDE_weight_order_l273_27381

theorem weight_order (P Q R S T : ℝ) 
  (h1 : P < 1000) (h2 : Q < 1000) (h3 : R < 1000) (h4 : S < 1000) (h5 : T < 1000)
  (h6 : Q + S = 1200) (h7 : R + T = 2100) (h8 : Q + T = 800) (h9 : Q + R = 900) (h10 : P + T = 700) :
  S > R ∧ R > T ∧ T > Q ∧ Q > P :=
by sorry

end NUMINAMATH_CALUDE_weight_order_l273_27381


namespace NUMINAMATH_CALUDE_sprocket_production_problem_l273_27361

/-- Represents a machine that produces sprockets -/
structure Machine where
  productionRate : ℝ
  timeToProduce660 : ℝ

/-- Given the conditions of the problem -/
theorem sprocket_production_problem 
  (machineA machineP machineQ : Machine)
  (h1 : machineA.productionRate = 6)
  (h2 : machineQ.productionRate = 1.1 * machineA.productionRate)
  (h3 : machineQ.timeToProduce660 = 660 / machineQ.productionRate)
  (h4 : machineP.timeToProduce660 > machineQ.timeToProduce660)
  (h5 : machineP.timeToProduce660 = machineQ.timeToProduce660 + (machineP.timeToProduce660 - machineQ.timeToProduce660)) :
  ¬ ∃ (x : ℝ), machineP.timeToProduce660 - machineQ.timeToProduce660 = x :=
sorry

end NUMINAMATH_CALUDE_sprocket_production_problem_l273_27361


namespace NUMINAMATH_CALUDE_max_value_of_linear_combination_l273_27328

theorem max_value_of_linear_combination (x y z : ℝ) 
  (h : x^2 + y^2 + z^2 = 9) : 
  ∃ (M : ℝ), M = 3 * Real.sqrt 14 ∧ 
  (∀ (a b c : ℝ), a^2 + b^2 + c^2 = 9 → a + 2*b + 3*c ≤ M) ∧
  (∃ (u v w : ℝ), u^2 + v^2 + w^2 = 9 ∧ u + 2*v + 3*w = M) :=
sorry

end NUMINAMATH_CALUDE_max_value_of_linear_combination_l273_27328


namespace NUMINAMATH_CALUDE_paving_cost_calculation_l273_27365

/-- Calculates the cost of paving a rectangular floor given its dimensions and the paving rate. -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a 5.5m by 4m room at Rs. 950 per square meter is Rs. 20,900. -/
theorem paving_cost_calculation :
  paving_cost 5.5 4 950 = 20900 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_calculation_l273_27365


namespace NUMINAMATH_CALUDE_max_product_of_three_l273_27350

def S : Finset Int := {-10, -5, -3, 0, 2, 6, 8}

theorem max_product_of_three (a b c : Int) :
  a ∈ S → b ∈ S → c ∈ S →
  a ≠ b → b ≠ c → a ≠ c →
  a * b * c ≤ 400 ∧ 
  ∃ (x y z : Int), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ 
    x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    x * y * z = 400 :=
by sorry

end NUMINAMATH_CALUDE_max_product_of_three_l273_27350


namespace NUMINAMATH_CALUDE_max_profit_allocation_l273_27317

/-- Represents the allocation of raw materials to workshops --/
structure Allocation :=
  (workshop_a : ℕ)
  (workshop_b : ℕ)

/-- Calculates the profit for a given allocation --/
def profit (a : Allocation) : ℝ :=
  let total_boxes := 60
  let box_cost := 80
  let water_cost := 5
  let product_price := 30
  let workshop_a_production := 12
  let workshop_b_production := 10
  let workshop_a_water := 4
  let workshop_b_water := 2
  30 * (workshop_a_production * a.workshop_a + workshop_b_production * a.workshop_b) -
  box_cost * total_boxes -
  water_cost * (workshop_a_water * a.workshop_a + workshop_b_water * a.workshop_b)

/-- Checks if an allocation satisfies the water consumption constraint --/
def water_constraint (a : Allocation) : Prop :=
  4 * a.workshop_a + 2 * a.workshop_b ≤ 200

/-- Checks if an allocation uses exactly 60 boxes --/
def total_boxes_constraint (a : Allocation) : Prop :=
  a.workshop_a + a.workshop_b = 60

/-- The theorem stating that the given allocation maximizes profit --/
theorem max_profit_allocation :
  ∀ a : Allocation,
  water_constraint a →
  total_boxes_constraint a →
  profit a ≤ profit { workshop_a := 40, workshop_b := 20 } :=
sorry

end NUMINAMATH_CALUDE_max_profit_allocation_l273_27317


namespace NUMINAMATH_CALUDE_pink_highlighters_count_l273_27399

theorem pink_highlighters_count (total yellow blue : ℕ) (h1 : total = 15) (h2 : yellow = 7) (h3 : blue = 5) :
  ∃ pink : ℕ, pink + yellow + blue = total ∧ pink = 3 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_count_l273_27399


namespace NUMINAMATH_CALUDE_total_holidays_in_year_l273_27337

def holidays : List Nat := [4, 3, 5, 3, 4, 2, 5, 3, 4, 3, 5, 4]

theorem total_holidays_in_year : holidays.sum = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_holidays_in_year_l273_27337


namespace NUMINAMATH_CALUDE_photo_arrangements_l273_27301

/-- The number of arrangements for four students (two boys and two girls) in a row,
    where the two girls must stand next to each other. -/
def arrangements_count : ℕ := 12

/-- The number of ways to arrange two girls next to each other. -/
def girls_arrangement : ℕ := 2

/-- The number of ways to arrange three entities (two boys and the pair of girls). -/
def entities_arrangement : ℕ := 6

theorem photo_arrangements :
  arrangements_count = girls_arrangement * entities_arrangement :=
by sorry

end NUMINAMATH_CALUDE_photo_arrangements_l273_27301


namespace NUMINAMATH_CALUDE_max_sum_abcd_l273_27388

theorem max_sum_abcd (a c d : ℤ) (b : ℕ+) 
  (h1 : a + b = c) 
  (h2 : b + c = d) 
  (h3 : c + d = a) : 
  (∃ (a' c' d' : ℤ) (b' : ℕ+), 
    a' + b' = c' ∧ 
    b' + c' = d' ∧ 
    c' + d' = a' ∧ 
    a' + b' + c' + d' = -5) ∧ 
  (∀ (a' c' d' : ℤ) (b' : ℕ+), 
    a' + b' = c' → 
    b' + c' = d' → 
    c' + d' = a' → 
    a' + b' + c' + d' ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_max_sum_abcd_l273_27388


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l273_27376

theorem digit_sum_puzzle :
  ∀ (B E D H : ℕ),
    B < 10 → E < 10 → D < 10 → H < 10 →
    B ≠ E → B ≠ D → B ≠ H → E ≠ D → E ≠ H → D ≠ H →
    (10 * B + E) * (10 * D + E) = 111 * H →
    E + B + D + H = 17 := by
  sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l273_27376


namespace NUMINAMATH_CALUDE_exists_x_in_interval_equivalence_of_statements_l273_27378

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

-- Theorem 1
theorem exists_x_in_interval : ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ (1/2)^x > log_half x := by sorry

-- Theorem 2
theorem equivalence_of_statements :
  (∀ x : ℝ, x ∈ Set.Ioo 1 5 → x + 1/x ≥ 2) ↔
  (∀ x : ℝ, x ∈ Set.Iic 1 ∪ Set.Ici 5 → x + 1/x < 2) := by sorry

end NUMINAMATH_CALUDE_exists_x_in_interval_equivalence_of_statements_l273_27378


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l273_27309

/-- Arithmetic sequence with given first term, last term, and common difference has the specified number of terms -/
theorem arithmetic_sequence_length
  (a₁ : ℤ)    -- First term
  (aₙ : ℤ)    -- Last term
  (d : ℤ)     -- Common difference
  (n : ℕ)     -- Number of terms
  (h1 : a₁ = -4)
  (h2 : aₙ = 32)
  (h3 : d = 3)
  (h4 : aₙ = a₁ + (n - 1) * d)  -- Formula for the nth term of an arithmetic sequence
  : n = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l273_27309


namespace NUMINAMATH_CALUDE_prob_ace_king_heart_value_l273_27334

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Probability of drawing Ace of clubs, King of clubs, and any heart in that order -/
def prob_ace_king_heart : ℚ :=
  1 / StandardDeck *
  1 / (StandardDeck - 1) *
  NumHearts / (StandardDeck - 2)

/-- Theorem stating the probability of drawing Ace of clubs, King of clubs, and any heart -/
theorem prob_ace_king_heart_value : prob_ace_king_heart = 13 / 132600 := by
  sorry

end NUMINAMATH_CALUDE_prob_ace_king_heart_value_l273_27334


namespace NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l273_27352

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 is 48π -/
theorem circle_area_equilateral_triangle : 
  ∀ (s : ℝ) (area : ℝ),
  s = 12 →  -- Side length of the equilateral triangle
  area = π * (s / Real.sqrt 3)^2 →  -- Area formula for circumscribed circle
  area = 48 * π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equilateral_triangle_l273_27352


namespace NUMINAMATH_CALUDE_special_function_inequality_l273_27321

/-- A function satisfying the given conditions -/
structure SpecialFunction where
  f : ℝ → ℝ
  diff : Differentiable ℝ f
  f_prime_1 : deriv f 1 = 0
  condition : ∀ x : ℝ, x ≠ 1 → (x - 1) * (deriv f x) > 0

/-- Theorem stating that for any function satisfying the SpecialFunction conditions,
    f(0) + f(2) > 2f(1) -/
theorem special_function_inequality (sf : SpecialFunction) : sf.f 0 + sf.f 2 > 2 * sf.f 1 := by
  sorry

end NUMINAMATH_CALUDE_special_function_inequality_l273_27321


namespace NUMINAMATH_CALUDE_outfit_combinations_l273_27354

theorem outfit_combinations (shirts : ℕ) (pants : ℕ) (excluded_combinations : ℕ) : 
  shirts = 5 → pants = 6 → excluded_combinations = 1 →
  shirts * pants - excluded_combinations = 29 := by
sorry

end NUMINAMATH_CALUDE_outfit_combinations_l273_27354


namespace NUMINAMATH_CALUDE_no_four_consecutive_power_numbers_l273_27359

theorem no_four_consecutive_power_numbers : 
  ¬ ∃ (n : ℕ), 
    (∃ (a b : ℕ) (k : ℕ), k > 1 ∧ n = a^k) ∧
    (∃ (c d : ℕ) (l : ℕ), l > 1 ∧ n + 1 = c^l) ∧
    (∃ (e f : ℕ) (m : ℕ), m > 1 ∧ n + 2 = e^m) ∧
    (∃ (g h : ℕ) (p : ℕ), p > 1 ∧ n + 3 = g^p) :=
by
  sorry


end NUMINAMATH_CALUDE_no_four_consecutive_power_numbers_l273_27359


namespace NUMINAMATH_CALUDE_recommendation_plans_count_l273_27394

/-- Represents the number of recommendation spots for each language --/
structure RecommendationSpots :=
  (russian : Nat)
  (japanese : Nat)
  (spanish : Nat)

/-- Represents the number of male and female candidates --/
structure Candidates :=
  (male : Nat)
  (female : Nat)

/-- Calculates the number of different recommendation plans --/
def countRecommendationPlans (spots : RecommendationSpots) (candidates : Candidates) : Nat :=
  sorry

/-- Theorem stating that the number of recommendation plans is 36 --/
theorem recommendation_plans_count :
  let spots := RecommendationSpots.mk 2 2 1
  let candidates := Candidates.mk 3 2
  countRecommendationPlans spots candidates = 36 :=
sorry

end NUMINAMATH_CALUDE_recommendation_plans_count_l273_27394


namespace NUMINAMATH_CALUDE_min_absolute_value_at_20_l273_27368

/-- An arithmetic sequence with first term 14 and common difference -3/4 -/
def arithmeticSequence (n : ℕ) : ℚ :=
  14 + (n - 1 : ℚ) * (-3/4)

/-- The absolute value of the nth term of the arithmetic sequence -/
def absoluteValue (n : ℕ) : ℚ :=
  |arithmeticSequence n|

theorem min_absolute_value_at_20 :
  ∀ n : ℕ, n ≠ 0 → absoluteValue 20 ≤ absoluteValue n :=
sorry

end NUMINAMATH_CALUDE_min_absolute_value_at_20_l273_27368


namespace NUMINAMATH_CALUDE_strategy_D_is_best_l273_27313

/-- Represents an investment strategy --/
inductive Strategy
| A  -- Six 1-year terms
| B  -- Three 2-year terms
| C  -- Two 3-year terms
| D  -- One 5-year term followed by one 1-year term

/-- Calculates the final amount for a given strategy --/
def calculate_return (strategy : Strategy) : ℝ :=
  match strategy with
  | Strategy.A => 10000 * (1 + 0.0225)^6
  | Strategy.B => 10000 * (1 + 0.025 * 2)^3
  | Strategy.C => 10000 * (1 + 0.028 * 3)^2
  | Strategy.D => 10000 * (1 + 0.03 * 5) * (1 + 0.0225)

/-- Theorem stating that Strategy D yields the highest return --/
theorem strategy_D_is_best :
  ∀ s : Strategy, calculate_return Strategy.D ≥ calculate_return s :=
by sorry


end NUMINAMATH_CALUDE_strategy_D_is_best_l273_27313


namespace NUMINAMATH_CALUDE_two_intersections_iff_m_values_l273_27329

def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + 3 * m * x + m - 1

theorem two_intersections_iff_m_values (m : ℝ) : 
  (∃! (p q : ℝ × ℝ), (p.1 = 0 ∨ p.2 = 0) ∧ (q.1 = 0 ∨ q.2 = 0) ∧ 
    p ≠ q ∧ f m p.1 = p.2 ∧ f m q.1 = q.2) ↔ 
  (m = 1 ∨ m = -5/4) := by
sorry

end NUMINAMATH_CALUDE_two_intersections_iff_m_values_l273_27329


namespace NUMINAMATH_CALUDE_angle_a_measure_l273_27326

/-- An isosceles right triangle with side lengths and angles -/
structure IsoscelesRightTriangle where
  -- Side lengths
  ab : ℝ
  bc : ℝ
  ac : ℝ
  -- Angles in radians
  angle_a : ℝ
  angle_b : ℝ
  angle_c : ℝ
  -- Properties
  ab_eq_bc : ab = bc
  right_angle_b : angle_b = Real.pi / 2
  angle_sum : angle_a + angle_b + angle_c = Real.pi

/-- The measure of angle A in an isosceles right triangle is π/4 radians (45 degrees) -/
theorem angle_a_measure (t : IsoscelesRightTriangle) : t.angle_a = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_a_measure_l273_27326


namespace NUMINAMATH_CALUDE_elizabeth_study_time_l273_27384

/-- Calculates the study time for math test given total study time and science test study time -/
def math_study_time (total_time science_time : ℕ) : ℕ :=
  total_time - science_time

/-- Theorem stating that given the total study time of 60 minutes and science test study time of 25 minutes, 
    the math test study time is 35 minutes -/
theorem elizabeth_study_time : 
  math_study_time 60 25 = 35 := by
  sorry

end NUMINAMATH_CALUDE_elizabeth_study_time_l273_27384


namespace NUMINAMATH_CALUDE_soup_feeding_problem_l273_27362

theorem soup_feeding_problem (initial_cans : ℕ) (adults_per_can : ℕ) (children_per_can : ℕ) 
  (children_to_feed : ℕ) (adults_fed : ℕ) : 
  initial_cans = 8 → 
  adults_per_can = 4 → 
  children_per_can = 6 → 
  children_to_feed = 24 → 
  adults_fed = (initial_cans - (children_to_feed / children_per_can)) * adults_per_can → 
  adults_fed = 16 := by
sorry

end NUMINAMATH_CALUDE_soup_feeding_problem_l273_27362


namespace NUMINAMATH_CALUDE_work_completion_time_l273_27372

/-- Given:
  - A can finish the work in 6 days
  - B worked for 10 days and left the job
  - A alone can finish the remaining work in 2 days
  Prove that B can finish the work in 15 days -/
theorem work_completion_time
  (total_work : ℝ)
  (a_completion_time : ℝ)
  (b_work_days : ℝ)
  (a_remaining_time : ℝ)
  (h1 : a_completion_time = 6)
  (h2 : b_work_days = 10)
  (h3 : a_remaining_time = 2)
  (h4 : total_work > 0) :
  ∃ (b_completion_time : ℝ),
    b_completion_time = 15 ∧
    (total_work / a_completion_time) * a_remaining_time =
      total_work - (total_work / b_completion_time) * b_work_days :=
by
  sorry


end NUMINAMATH_CALUDE_work_completion_time_l273_27372


namespace NUMINAMATH_CALUDE_combined_shoe_size_l273_27369

theorem combined_shoe_size (jasmine_size : ℝ) (alexa_size : ℝ) (clara_size : ℝ) (molly_shoe_size : ℝ) (molly_sandal_size : ℝ) :
  jasmine_size = 7 →
  alexa_size = 2 * jasmine_size →
  clara_size = 3 * jasmine_size →
  molly_shoe_size = 1.5 * jasmine_size →
  molly_sandal_size = molly_shoe_size - 0.5 →
  jasmine_size + alexa_size + clara_size + molly_shoe_size + molly_sandal_size = 62.5 := by
  sorry

end NUMINAMATH_CALUDE_combined_shoe_size_l273_27369


namespace NUMINAMATH_CALUDE_lcm_factor_is_twelve_l273_27330

def problem (A B X : ℕ) : Prop :=
  A > 0 ∧ B > 0 ∧
  Nat.gcd A B = 42 ∧
  A = 504 ∧
  Nat.lcm A B = 42 * X

theorem lcm_factor_is_twelve :
  ∀ A B X, problem A B X → X = 12 := by sorry

end NUMINAMATH_CALUDE_lcm_factor_is_twelve_l273_27330


namespace NUMINAMATH_CALUDE_matrix_operation_example_l273_27357

-- Define the operation
def matrix_operation (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem matrix_operation_example :
  matrix_operation (-2) (0.5) 2 4 = -9 := by
  sorry

end NUMINAMATH_CALUDE_matrix_operation_example_l273_27357


namespace NUMINAMATH_CALUDE_solution_set_l273_27367

theorem solution_set (x : ℝ) :
  (1 / Real.pi) ^ (-x + 1) > (1 / Real.pi) ^ (x^2 - x) ↔ x < -1 ∨ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l273_27367


namespace NUMINAMATH_CALUDE_freddy_age_l273_27363

/-- Given the ages of several people and their relationships, prove Freddy's age. -/
theorem freddy_age (stephanie tim job oliver tina freddy : ℝ) 
  (h1 : freddy = stephanie - 2.5)
  (h2 : 3 * stephanie = job + tim)
  (h3 : tim = oliver / 2)
  (h4 : oliver / 3 = tina)
  (h5 : tina = freddy - 2)
  (h6 : job = 5)
  (h7 : oliver = job + 10) : 
  freddy = 7 := by
  sorry

end NUMINAMATH_CALUDE_freddy_age_l273_27363


namespace NUMINAMATH_CALUDE_ceiling_times_self_equals_210_l273_27364

theorem ceiling_times_self_equals_210 :
  ∃! (x : ℝ), ⌈x⌉ * x = 210 ∧ x = 14 := by sorry

end NUMINAMATH_CALUDE_ceiling_times_self_equals_210_l273_27364


namespace NUMINAMATH_CALUDE_parabola_directrix_l273_27383

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y = (x^2 - 4*x + 4) / 8

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop :=
  y = -1/4

/-- Theorem: The directrix of the given parabola is y = -1/4 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → ∃ y_d : ℝ, directrix_equation y_d :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l273_27383


namespace NUMINAMATH_CALUDE_b_work_days_l273_27304

/-- Proves that B worked for 10 days before leaving the job -/
theorem b_work_days (a_total : ℕ) (b_total : ℕ) (a_remaining : ℕ) : 
  a_total = 21 → b_total = 15 → a_remaining = 7 → 
  ∃ (b_days : ℕ), b_days = 10 ∧ 
    (b_days : ℚ) / b_total + a_remaining / a_total = 1 :=
by sorry

end NUMINAMATH_CALUDE_b_work_days_l273_27304


namespace NUMINAMATH_CALUDE_store_profit_calculation_l273_27348

-- Define the types of sweaters
inductive SweaterType
| Turtleneck
| Crewneck
| Vneck

-- Define the initial cost, quantity, and markup percentages for each sweater type
def initial_cost (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 30
  | SweaterType.Crewneck => 25
  | SweaterType.Vneck => 20

def quantity (s : SweaterType) : ℕ :=
  match s with
  | SweaterType.Turtleneck => 100
  | SweaterType.Crewneck => 150
  | SweaterType.Vneck => 200

def initial_markup (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 0.2
  | SweaterType.Crewneck => 0.35
  | SweaterType.Vneck => 0.25

def new_year_markup (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 0.25
  | SweaterType.Crewneck => 0.15
  | SweaterType.Vneck => 0.2

def february_discount (s : SweaterType) : ℚ :=
  match s with
  | SweaterType.Turtleneck => 0.09
  | SweaterType.Crewneck => 0.12
  | SweaterType.Vneck => 0.15

-- Calculate the final price for each sweater type
def final_price (s : SweaterType) : ℚ :=
  let base_price := initial_cost s * (1 + initial_markup s)
  let new_year_price := base_price + initial_cost s * new_year_markup s
  new_year_price * (1 - february_discount s)

-- Calculate the profit for each sweater type
def profit (s : SweaterType) : ℚ :=
  (final_price s - initial_cost s) * quantity s

-- Calculate the total profit
def total_profit : ℚ :=
  profit SweaterType.Turtleneck + profit SweaterType.Crewneck + profit SweaterType.Vneck

-- Theorem statement
theorem store_profit_calculation :
  total_profit = 3088.5 := by sorry

end NUMINAMATH_CALUDE_store_profit_calculation_l273_27348


namespace NUMINAMATH_CALUDE_complex_root_sum_l273_27316

theorem complex_root_sum (w : ℂ) (hw : w^4 + w^2 + 1 = 0) :
  w^120 + w^121 + w^122 + w^123 + w^124 = w - 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_root_sum_l273_27316


namespace NUMINAMATH_CALUDE_circle_equation_l273_27311

-- Define the line l: x - 2y - 1 = 0
def line_l (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the circle C
def circle_C (center_x center_y radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center_x)^2 + (p.2 - center_y)^2 = radius^2}

theorem circle_equation :
  ∃ (center_x center_y radius : ℝ),
    (line_l center_x center_y) ∧
    ((2 : ℝ), 1) ∈ circle_C center_x center_y radius ∧
    ((1 : ℝ), 2) ∈ circle_C center_x center_y radius ∧
    center_x = -1 ∧ center_y = -1 ∧ radius^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l273_27311


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l273_27375

def M : Set ℤ := {1, 2, 3, 4, 5, 6}

def N : Set ℤ := {x | -2 < x ∧ x < 5}

theorem intersection_of_M_and_N : M ∩ N = {1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l273_27375


namespace NUMINAMATH_CALUDE_philips_banana_groups_l273_27336

/-- Given Philip's fruit collection, prove the number of banana groups -/
theorem philips_banana_groups
  (total_oranges : ℕ) (total_bananas : ℕ)
  (orange_groups : ℕ) (oranges_per_group : ℕ)
  (h1 : total_oranges = 384)
  (h2 : total_bananas = 192)
  (h3 : orange_groups = 16)
  (h4 : oranges_per_group = 24)
  (h5 : total_oranges = orange_groups * oranges_per_group)
  : total_bananas / oranges_per_group = 8 := by
  sorry

end NUMINAMATH_CALUDE_philips_banana_groups_l273_27336


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l273_27377

-- Define the cone
def cone_base_diameter : ℝ := 16
def cone_vertex_angle : ℝ := 90

-- Define the sphere
def sphere_touches_lateral_surfaces : Prop := sorry
def sphere_rests_on_table : Prop := sorry

-- Calculate the volume of the sphere
noncomputable def sphere_volume : ℝ := 
  let base_radius := cone_base_diameter / 2
  let cone_height := base_radius * 2
  let sphere_radius := base_radius / Real.sqrt 2
  (4 / 3) * Real.pi * (sphere_radius ^ 3)

-- Theorem statement
theorem inscribed_sphere_volume 
  (h1 : cone_vertex_angle = 90)
  (h2 : sphere_touches_lateral_surfaces)
  (h3 : sphere_rests_on_table) :
  sphere_volume = (512 * Real.sqrt 2 * Real.pi) / 3 := by sorry

end NUMINAMATH_CALUDE_inscribed_sphere_volume_l273_27377


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l273_27320

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l273_27320


namespace NUMINAMATH_CALUDE_power_three_mod_seven_l273_27322

theorem power_three_mod_seven : 3^123 % 7 = 6 := by sorry

end NUMINAMATH_CALUDE_power_three_mod_seven_l273_27322


namespace NUMINAMATH_CALUDE_word_count_theorems_l273_27302

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of words we're considering -/
def word_length : ℕ := 5

/-- The number of 5-letter words -/
def num_words : ℕ := alphabet_size ^ word_length

/-- The number of 5-letter words with all different letters -/
def num_words_diff : ℕ := 
  (List.range word_length).foldl (fun acc i => acc * (alphabet_size - i)) alphabet_size

/-- The number of 5-letter words without any letter repeating consecutively -/
def num_words_no_consec : ℕ := alphabet_size * (alphabet_size - 1)^(word_length - 1)

theorem word_count_theorems : 
  (num_words = 26^5) ∧ 
  (num_words_diff = 26 * 25 * 24 * 23 * 22) ∧ 
  (num_words_no_consec = 26 * 25^4) := by
  sorry

end NUMINAMATH_CALUDE_word_count_theorems_l273_27302


namespace NUMINAMATH_CALUDE_sum_equality_exists_l273_27385

theorem sum_equality_exists (a : Fin 16 → ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_positive : ∀ i, a i > 0) 
  (h_bound : ∀ i, a i ≤ 100) : 
  ∃ i j k l : Fin 16, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ a i + a j = a k + a l :=
sorry

end NUMINAMATH_CALUDE_sum_equality_exists_l273_27385


namespace NUMINAMATH_CALUDE_alice_score_l273_27345

theorem alice_score : 
  let correct_answers : ℕ := 15
  let incorrect_answers : ℕ := 5
  let unattempted : ℕ := 10
  let correct_points : ℚ := 1
  let incorrect_penalty : ℚ := 1/4
  correct_answers * correct_points - incorrect_answers * incorrect_penalty = 13.75 := by
  sorry

end NUMINAMATH_CALUDE_alice_score_l273_27345


namespace NUMINAMATH_CALUDE_intersection_distance_prove_intersection_distance_l273_27358

/-- The distance between the intersection points of y² = x and x + 2y = 10 is 2√55 -/
theorem intersection_distance : ℝ → Prop := fun d =>
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    (y₁^2 = x₁) ∧ (x₁ + 2*y₁ = 10) ∧
    (y₂^2 = x₂) ∧ (x₂ + 2*y₂ = 10) ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    d = ((x₂ - x₁)^2 + (y₂ - y₁)^2).sqrt ∧
    d = 2 * Real.sqrt 55

theorem prove_intersection_distance : intersection_distance (2 * Real.sqrt 55) := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_prove_intersection_distance_l273_27358


namespace NUMINAMATH_CALUDE_continuity_at_two_l273_27382

def f (x : ℝ) : ℝ := -5 * x^2 - 8

theorem continuity_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |f x - f 2| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_continuity_at_two_l273_27382


namespace NUMINAMATH_CALUDE_complex_on_real_axis_l273_27346

theorem complex_on_real_axis (a : ℝ) : 
  let z : ℂ := (a - Complex.I) * (1 + Complex.I)
  z.im = 0 → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_on_real_axis_l273_27346


namespace NUMINAMATH_CALUDE_smallest_c_inequality_l273_27310

theorem smallest_c_inequality (c : ℝ) : 
  (∀ x y : ℝ, x ≥ 0 ∧ y ≥ 0 → Real.sqrt (x * y) + c * Real.sqrt (|x - y|) ≥ (x + y) / 2) ↔ 
  c ≥ (1 / 2 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_smallest_c_inequality_l273_27310


namespace NUMINAMATH_CALUDE_melanie_plums_l273_27331

def initial_plums : ℕ := 7
def plums_given : ℕ := 3

theorem melanie_plums : initial_plums - plums_given = 4 := by
  sorry

end NUMINAMATH_CALUDE_melanie_plums_l273_27331


namespace NUMINAMATH_CALUDE_total_hats_bought_l273_27392

theorem total_hats_bought (blue_cost green_cost total_price green_count : ℕ) 
  (h1 : blue_cost = 6)
  (h2 : green_cost = 7)
  (h3 : total_price = 530)
  (h4 : green_count = 20) :
  ∃ (blue_count : ℕ), blue_count * blue_cost + green_count * green_cost = total_price ∧
                      blue_count + green_count = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_hats_bought_l273_27392


namespace NUMINAMATH_CALUDE_circle_inequality_max_k_l273_27325

theorem circle_inequality_max_k : 
  (∃ k : ℝ, ∀ x y : ℝ, x^2 + y^2 = 1 → x + y - k ≥ 0) ∧ 
  (∀ k : ℝ, (∀ x y : ℝ, x^2 + y^2 = 1 → x + y - k ≥ 0) → k ≤ -Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_inequality_max_k_l273_27325


namespace NUMINAMATH_CALUDE_simplify_expression_l273_27332

variables (x y : ℝ)

def A (x y : ℝ) : ℝ := x^2 + 3*x*y + y^2
def B (x y : ℝ) : ℝ := x^2 - 3*x*y + y^2

theorem simplify_expression (x y : ℝ) : 
  A x y - (B x y + 2 * B x y - (A x y + B x y)) = 12 * x * y := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l273_27332


namespace NUMINAMATH_CALUDE_ryn_to_nikki_ratio_l273_27370

/-- The lengths of favorite movies for Joyce, Michael, Nikki, and Ryn -/
structure MovieLengths where
  michael : ℝ
  joyce : ℝ
  nikki : ℝ
  ryn : ℝ

/-- The conditions of the movie lengths problem -/
def movie_conditions (m : MovieLengths) : Prop :=
  m.joyce = m.michael + 2 ∧
  m.nikki = 3 * m.michael ∧
  m.nikki = 30 ∧
  m.michael + m.joyce + m.nikki + m.ryn = 76

/-- The theorem stating the ratio of Ryn's movie length to Nikki's movie length -/
theorem ryn_to_nikki_ratio (m : MovieLengths) :
  movie_conditions m → m.ryn / m.nikki = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ryn_to_nikki_ratio_l273_27370


namespace NUMINAMATH_CALUDE_gathering_attendance_l273_27360

/-- The number of people who took wine -/
def W : ℕ := 26

/-- The number of people who took soda -/
def S : ℕ := 22

/-- The number of people who took juice -/
def J : ℕ := 18

/-- The number of people who took both wine and soda -/
def WS : ℕ := 17

/-- The number of people who took both wine and juice -/
def WJ : ℕ := 12

/-- The number of people who took both soda and juice -/
def SJ : ℕ := 10

/-- The number of people who took all three drinks -/
def WSJ : ℕ := 8

/-- The total number of people at the gathering -/
def total_people : ℕ := W + S + J - WS - WJ - SJ + WSJ

theorem gathering_attendance : total_people = 35 := by
  sorry

end NUMINAMATH_CALUDE_gathering_attendance_l273_27360


namespace NUMINAMATH_CALUDE_product_of_binary_and_ternary_l273_27393

/-- Converts a binary number represented as a list of digits to its decimal equivalent -/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 2^i) 0

/-- Converts a ternary number represented as a list of digits to its decimal equivalent -/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.reverse.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

theorem product_of_binary_and_ternary :
  let binary_num := [1, 1, 1, 0]
  let ternary_num := [1, 0, 2]
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 154 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binary_and_ternary_l273_27393


namespace NUMINAMATH_CALUDE_dog_to_hamster_lifespan_ratio_l273_27395

/-- The average lifespan of a hamster in years -/
def hamster_lifespan : ℝ := 2.5

/-- The lifespan of a fish in years -/
def fish_lifespan : ℝ := 12

/-- The lifespan of a dog in years -/
def dog_lifespan : ℝ := fish_lifespan - 2

theorem dog_to_hamster_lifespan_ratio :
  dog_lifespan / hamster_lifespan = 4 := by sorry

end NUMINAMATH_CALUDE_dog_to_hamster_lifespan_ratio_l273_27395


namespace NUMINAMATH_CALUDE_nine_identical_digits_multiples_l273_27319

theorem nine_identical_digits_multiples (n : ℕ) : 
  n ≥ 1 ∧ n ≤ 9 → 
  ∃ (d : ℕ), d ≥ 1 ∧ d ≤ 9 ∧ 12345679 * (9 * n) = d * 111111111 ∧
  (∀ (m : ℕ), 12345679 * m = d * 111111111 → m = 9 * n) :=
by sorry

end NUMINAMATH_CALUDE_nine_identical_digits_multiples_l273_27319


namespace NUMINAMATH_CALUDE_inequality_proof_l273_27397

theorem inequality_proof (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (ha : a₁ ≥ a₂ ∧ a₂ ≥ a₃ ∧ a₃ > 0)
  (hb : b₁ ≥ b₂ ∧ b₂ ≥ b₃ ∧ b₃ > 0)
  (hab : a₁ * a₂ * a₃ = b₁ * b₂ * b₃)
  (hdiff : a₁ - a₃ ≤ b₁ - b₃) :
  a₁ + a₂ + a₃ ≤ 2 * (b₁ + b₂ + b₃) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l273_27397


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l273_27314

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 5

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

-- Theorem statement
theorem monotonic_decreasing_interval :
  ∀ x : ℝ, (x > -1 ∧ x < 3) ↔ (f' x < 0) :=
sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l273_27314


namespace NUMINAMATH_CALUDE_amp_neg_eight_five_l273_27307

-- Define the & operation
def amp (a b : ℝ) : ℝ := (a + b) * (a - b)

-- State the theorem
theorem amp_neg_eight_five : amp (-8 : ℝ) 5 = 39 := by
  sorry

end NUMINAMATH_CALUDE_amp_neg_eight_five_l273_27307


namespace NUMINAMATH_CALUDE_angle_adjustment_l273_27379

def are_complementary (a b : ℝ) : Prop := a + b = 90

theorem angle_adjustment (x y : ℝ) 
  (h1 : are_complementary x y)
  (h2 : x / y = 1 / 2)
  (h3 : x < y) :
  let new_x := x * 1.2
  let new_y := 90 - new_x
  (y - new_y) / y = 0.1 := by sorry

end NUMINAMATH_CALUDE_angle_adjustment_l273_27379


namespace NUMINAMATH_CALUDE_bernardo_receives_345_l273_27312

/-- The distribution pattern for Bernardo: 2, 5, 8, 11, ... -/
def bernardoSequence (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- The sum of the first n terms in Bernardo's sequence -/
def bernardoSum (n : ℕ) : ℕ := n * (2 * 2 + (n - 1) * 3) / 2

/-- The total amount distributed -/
def totalAmount : ℕ := 1000

theorem bernardo_receives_345 :
  ∃ n : ℕ, bernardoSum n ≤ totalAmount ∧ 
  bernardoSum (n + 1) > totalAmount ∧ 
  bernardoSum n = 345 := by sorry

end NUMINAMATH_CALUDE_bernardo_receives_345_l273_27312


namespace NUMINAMATH_CALUDE_distance_not_proportional_to_time_l273_27390

/-- Uniform motion equation -/
def uniform_motion (a v t : ℝ) : ℝ := a + v * t

/-- Proportionality definition -/
def proportional (f : ℝ → ℝ) : Prop := ∀ (k t : ℝ), f (k * t) = k * f t

/-- Theorem: In uniform motion, distance is not generally proportional to time -/
theorem distance_not_proportional_to_time (a v : ℝ) (h : a ≠ 0) :
  ¬ proportional (uniform_motion a v) := by
  sorry

end NUMINAMATH_CALUDE_distance_not_proportional_to_time_l273_27390


namespace NUMINAMATH_CALUDE_storks_equal_other_birds_l273_27305

/-- Represents the count of different bird species on a fence --/
structure BirdCounts where
  sparrows : ℕ
  crows : ℕ
  storks : ℕ
  egrets : ℕ

/-- Calculates the final bird counts after all arrivals and departures --/
def finalBirdCounts (initial : BirdCounts) 
  (firstArrival : BirdCounts) 
  (firstDeparture : BirdCounts)
  (secondArrival : BirdCounts) : BirdCounts :=
  { sparrows := initial.sparrows + firstArrival.sparrows - firstDeparture.sparrows,
    crows := initial.crows + firstArrival.crows + secondArrival.crows,
    storks := initial.storks + firstArrival.storks + secondArrival.storks,
    egrets := firstArrival.egrets - firstDeparture.egrets }

/-- The main theorem stating that the number of storks equals the sum of all other birds --/
theorem storks_equal_other_birds : 
  let initial := BirdCounts.mk 2 1 3 0
  let firstArrival := BirdCounts.mk 1 3 6 4
  let firstDeparture := BirdCounts.mk 2 0 0 1
  let secondArrival := BirdCounts.mk 0 4 3 0
  let final := finalBirdCounts initial firstArrival firstDeparture secondArrival
  final.storks = final.sparrows + final.crows + final.egrets := by
  sorry

end NUMINAMATH_CALUDE_storks_equal_other_birds_l273_27305


namespace NUMINAMATH_CALUDE_sum_positive_implies_one_positive_l273_27340

theorem sum_positive_implies_one_positive (a b : ℝ) : 
  a + b > 0 → a > 0 ∨ b > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_positive_implies_one_positive_l273_27340


namespace NUMINAMATH_CALUDE_intersection_equals_three_l273_27398

theorem intersection_equals_three :
  ∃ a : ℝ, ({1, 3, a^2 + 3*a - 4} : Set ℝ) ∩ ({0, 6, a^2 + 4*a - 2, a + 3} : Set ℝ) = {3} :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_equals_three_l273_27398


namespace NUMINAMATH_CALUDE_average_equation_l273_27373

theorem average_equation (a : ℝ) : ((2 * a + 16) + (3 * a - 8)) / 2 = 69 → a = 26 := by
  sorry

end NUMINAMATH_CALUDE_average_equation_l273_27373


namespace NUMINAMATH_CALUDE_equation_solution_l273_27349

theorem equation_solution (x : ℝ) : 
  x ≠ 8 → x ≠ 7 → 
  ((x + 7) / (x - 8) - 6 = (5 * x - 55) / (7 - x)) ↔ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l273_27349


namespace NUMINAMATH_CALUDE_a_equals_base_conversion_l273_27347

/-- Convert a natural number to a different base representation --/
def toBase (n : ℕ) (base : ℕ) : List ℕ :=
  sorry

/-- Interpret a list of digits in a given base as a natural number --/
def fromBase (digits : List ℕ) (base : ℕ) : ℕ :=
  sorry

/-- Check if a list of numbers forms an arithmetic sequence --/
def isArithmeticSequence (seq : List ℕ) : Bool :=
  sorry

/-- Define the sequence a_n as described in the problem --/
def a (p : ℕ) : ℕ → ℕ
  | n => if n < p - 1 then n else
    sorry -- Find the least positive integer not forming an arithmetic sequence

/-- Main theorem to prove --/
theorem a_equals_base_conversion {p : ℕ} (hp : Nat.Prime p) (hodd : Odd p) :
  ∀ n, a p n = fromBase (toBase n (p - 1)) p :=
  sorry

end NUMINAMATH_CALUDE_a_equals_base_conversion_l273_27347


namespace NUMINAMATH_CALUDE_number_problem_l273_27386

theorem number_problem (x : ℝ) : 
  (15 / 100 * 40 = 25 / 100 * x + 2) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l273_27386


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l273_27391

/-- An arithmetic sequence with sum S_n and common difference d -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  d : ℝ
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Main theorem about properties of an arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence)
  (h : seq.S 7 > seq.S 6 ∧ seq.S 6 > seq.S 8) :
  seq.d < 0 ∧ 
  seq.S 14 < 0 ∧
  (∀ n, seq.S n ≤ seq.S 7) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l273_27391


namespace NUMINAMATH_CALUDE_product_of_integers_l273_27341

theorem product_of_integers (x y : ℕ+) 
  (sum_eq : x + y = 26)
  (diff_sq_eq : x^2 - y^2 = 52) : 
  x * y = 168 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l273_27341


namespace NUMINAMATH_CALUDE_b_share_of_earnings_l273_27371

theorem b_share_of_earnings 
  (a_days b_days c_days : ℕ) 
  (total_earnings : ℚ) 
  (ha : a_days = 6)
  (hb : b_days = 8)
  (hc : c_days = 12)
  (htotal : total_earnings = 2340) :
  (1 / b_days) / ((1 / a_days) + (1 / b_days) + (1 / c_days)) * total_earnings = 780 := by
  sorry

end NUMINAMATH_CALUDE_b_share_of_earnings_l273_27371


namespace NUMINAMATH_CALUDE_lattice_point_in_diagonal_pentagon_l273_27318

/-- A point in the 2D plane -/
structure Point where
  x : ℤ
  y : ℤ

/-- A pentagon defined by five points -/
structure Pentagon where
  a : Point
  b : Point
  c : Point
  d : Point
  e : Point

/-- Check if a pentagon is convex -/
def is_convex (p : Pentagon) : Prop := sorry

/-- Check if a point is inside or on the boundary of a polygon defined by a list of points -/
def is_inside_or_on_boundary (point : Point) (polygon : List Point) : Prop := sorry

/-- The pentagon formed by the diagonals of the given pentagon -/
def diagonal_pentagon (p : Pentagon) : List Point := sorry

theorem lattice_point_in_diagonal_pentagon (p : Pentagon) 
  (h_convex : is_convex p) : 
  ∃ (point : Point), is_inside_or_on_boundary point (diagonal_pentagon p) := by
  sorry

end NUMINAMATH_CALUDE_lattice_point_in_diagonal_pentagon_l273_27318


namespace NUMINAMATH_CALUDE_inequality_region_range_l273_27306

-- Define the inequality function
def f (m x y : ℝ) : Prop := x - (m^2 - 2*m + 4)*y - 6 > 0

-- Define the theorem
theorem inequality_region_range :
  ∀ m : ℝ, (∀ x y : ℝ, f m x y → (x ≠ -1 ∨ y ≠ -1)) ↔ -1 < m ∧ m < 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_region_range_l273_27306


namespace NUMINAMATH_CALUDE_tmall_double_eleven_sales_scientific_notation_l273_27380

theorem tmall_double_eleven_sales_scientific_notation :
  let billion : ℕ := 10^9
  let sales : ℕ := 2684 * billion
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ (a * 10^n : ℝ) = sales ∧ a = 2.684 ∧ n = 11 :=
by sorry

end NUMINAMATH_CALUDE_tmall_double_eleven_sales_scientific_notation_l273_27380


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l273_27356

theorem sufficient_not_necessary (x : ℝ) :
  (∀ x, x > 2 → x^2 > 4) ∧ (∃ x, x^2 > 4 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l273_27356


namespace NUMINAMATH_CALUDE_remainder_problem_l273_27308

theorem remainder_problem (N : ℕ) (R : ℕ) : 
  (∃ Q : ℕ, N = 67 * Q + 1) → 
  (N = 68 * 269 + R) → 
  R = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l273_27308


namespace NUMINAMATH_CALUDE_lego_problem_l273_27327

theorem lego_problem (simon bruce kent : ℕ) : 
  simon = (bruce * 6) / 5 →  -- Simon has 20% more legos than Bruce
  bruce = kent + 20 →        -- Bruce has 20 more legos than Kent
  simon = 72 →               -- Simon has 72 legos
  kent = 40 :=               -- Kent has 40 legos
by sorry

end NUMINAMATH_CALUDE_lego_problem_l273_27327


namespace NUMINAMATH_CALUDE_meaningful_reciprocal_l273_27338

theorem meaningful_reciprocal (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_reciprocal_l273_27338


namespace NUMINAMATH_CALUDE_total_peppers_weight_l273_27374

/-- The weight of green peppers bought by Hannah's Vegetarian Restaurant -/
def green_peppers : ℝ := 0.33

/-- The weight of red peppers bought by Hannah's Vegetarian Restaurant -/
def red_peppers : ℝ := 0.33

/-- The total weight of peppers bought by Hannah's Vegetarian Restaurant -/
def total_peppers : ℝ := green_peppers + red_peppers

/-- Theorem stating that the total weight of peppers is 0.66 pounds -/
theorem total_peppers_weight : total_peppers = 0.66 := by sorry

end NUMINAMATH_CALUDE_total_peppers_weight_l273_27374


namespace NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l273_27300

theorem smallest_four_digit_multiple_of_18 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 18 ∣ n → 1008 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_four_digit_multiple_of_18_l273_27300


namespace NUMINAMATH_CALUDE_congruence_system_solution_l273_27342

theorem congruence_system_solution (n : ℤ) :
  (n % 5 = 3 ∧ n % 7 = 4 ∧ n % 3 = 2) ↔ ∃ k : ℤ, n = 105 * k + 53 :=
sorry

end NUMINAMATH_CALUDE_congruence_system_solution_l273_27342
