import Mathlib

namespace NUMINAMATH_CALUDE_min_value_theorem_l972_97280

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y / x = 1) :
  1 / x + x / y ≥ 4 ∧ (1 / x + x / y = 4 ↔ y = x^2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l972_97280


namespace NUMINAMATH_CALUDE_clothing_price_proof_l972_97242

/-- Given the following conditions:
    - Total spent on 7 pieces of clothing is $610
    - One piece costs $49
    - Another piece costs $81
    - The remaining pieces all cost the same
    - The price of the remaining pieces is a multiple of 5
    Prove that each of the remaining pieces costs $96 -/
theorem clothing_price_proof (total_spent : ℕ) (total_pieces : ℕ) (price1 : ℕ) (price2 : ℕ) (price_other : ℕ) :
  total_spent = 610 →
  total_pieces = 7 →
  price1 = 49 →
  price2 = 81 →
  (total_spent - price1 - price2) % (total_pieces - 2) = 0 →
  price_other % 5 = 0 →
  price_other * (total_pieces - 2) + price1 + price2 = total_spent →
  price_other = 96 := by
  sorry

#eval 96 * 5 + 49 + 81  -- Should output 610

end NUMINAMATH_CALUDE_clothing_price_proof_l972_97242


namespace NUMINAMATH_CALUDE_binary_51_l972_97216

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- Theorem: The binary representation of 51 is [true, true, false, false, true, true] -/
theorem binary_51 : toBinary 51 = [true, true, false, false, true, true] := by
  sorry

#eval toBinary 51

end NUMINAMATH_CALUDE_binary_51_l972_97216


namespace NUMINAMATH_CALUDE_factory_produces_160_crayons_in_4_hours_l972_97230

/-- Represents a crayon factory with given specifications -/
structure CrayonFactory where
  num_colors : ℕ
  crayons_per_color_per_box : ℕ
  boxes_per_hour : ℕ

/-- Calculates the total number of crayons produced in a given number of hours -/
def total_crayons_produced (factory : CrayonFactory) (hours : ℕ) : ℕ :=
  factory.num_colors * factory.crayons_per_color_per_box * factory.boxes_per_hour * hours

/-- Theorem stating that a factory with given specifications produces 160 crayons in 4 hours -/
theorem factory_produces_160_crayons_in_4_hours 
  (factory : CrayonFactory) 
  (h1 : factory.num_colors = 4) 
  (h2 : factory.crayons_per_color_per_box = 2) 
  (h3 : factory.boxes_per_hour = 5) : 
  total_crayons_produced factory 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_factory_produces_160_crayons_in_4_hours_l972_97230


namespace NUMINAMATH_CALUDE_union_of_M_and_P_l972_97267

-- Define the sets M and P
def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 1}
def P : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- State the theorem
theorem union_of_M_and_P : M ∪ P = {x : ℝ | -2 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_P_l972_97267


namespace NUMINAMATH_CALUDE_determine_absolute_b_l972_97254

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the polynomial g(x)
def g (a b c : ℤ) (x : ℂ) : ℂ :=
  a * x^5 + b * x^4 + c * x^3 + c * x^2 + b * x + a

-- State the theorem
theorem determine_absolute_b (a b c : ℤ) : 
  g a b c (3 + i) = 0 →
  Int.gcd a b = 1 ∧ Int.gcd a c = 1 ∧ Int.gcd b c = 1 →
  |b| = 66 := by
  sorry

end NUMINAMATH_CALUDE_determine_absolute_b_l972_97254


namespace NUMINAMATH_CALUDE_cubic_equation_no_negative_roots_l972_97257

theorem cubic_equation_no_negative_roots :
  ∀ x : ℝ, x < 0 → x^3 - 9*x^2 + 23*x - 15 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_no_negative_roots_l972_97257


namespace NUMINAMATH_CALUDE_garden_area_l972_97296

/-- A rectangular garden with specific walking measurements -/
structure Garden where
  length : ℝ
  width : ℝ
  length_walk : length * 30 = 1200
  perimeter_walk : (2 * length + 2 * width) * 12 = 1200

/-- The area of the garden is 400 square meters -/
theorem garden_area (g : Garden) : g.length * g.width = 400 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l972_97296


namespace NUMINAMATH_CALUDE_equal_cost_at_280_minutes_unique_equal_cost_point_l972_97201

/-- Represents a phone service plan with a monthly fee and per-minute rate. -/
structure ServicePlan where
  monthlyFee : ℝ
  perMinuteRate : ℝ

/-- Calculates the cost of a service plan for a given number of minutes. -/
def planCost (plan : ServicePlan) (minutes : ℝ) : ℝ :=
  plan.monthlyFee + plan.perMinuteRate * minutes

/-- Theorem stating that the costs of two specific phone service plans are equal at 280 minutes. -/
theorem equal_cost_at_280_minutes : 
  let plan1 : ServicePlan := { monthlyFee := 22, perMinuteRate := 0.13 }
  let plan2 : ServicePlan := { monthlyFee := 8, perMinuteRate := 0.18 }
  planCost plan1 280 = planCost plan2 280 := by
  sorry

/-- Theorem stating that 280 minutes is the unique point where the costs are equal. -/
theorem unique_equal_cost_point : 
  let plan1 : ServicePlan := { monthlyFee := 22, perMinuteRate := 0.13 }
  let plan2 : ServicePlan := { monthlyFee := 8, perMinuteRate := 0.18 }
  ∀ x : ℝ, planCost plan1 x = planCost plan2 x ↔ x = 280 := by
  sorry

end NUMINAMATH_CALUDE_equal_cost_at_280_minutes_unique_equal_cost_point_l972_97201


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l972_97291

open Real

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > -1 → Real.log (x + 1) < x) ↔ (∃ x : ℝ, x > -1 ∧ Real.log (x + 1) ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l972_97291


namespace NUMINAMATH_CALUDE_coin_flip_probability_l972_97259

theorem coin_flip_probability :
  let n : ℕ := 5  -- total number of coins
  let k : ℕ := 3  -- number of specific coins we want to be heads
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := 2^(n - k)
  favorable_outcomes / total_outcomes = (1 : ℚ) / 8 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l972_97259


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l972_97222

theorem greatest_divisor_with_remainders : 
  Nat.gcd (450 - 60) (Nat.gcd (330 - 15) (Nat.gcd (675 - 45) (725 - 25))) = 5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l972_97222


namespace NUMINAMATH_CALUDE_equal_expressions_l972_97258

theorem equal_expressions (x y z : ℤ) :
  x + 2 * y * z = (x + y) * (x + 2 * z) ↔ x + y + 2 * z = 1 ∨ x = 0 :=
by sorry

end NUMINAMATH_CALUDE_equal_expressions_l972_97258


namespace NUMINAMATH_CALUDE_tangent_lines_range_l972_97212

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 3

-- Define the function g(x) = 2x^3 - 3x^2
def g (x : ℝ) : ℝ := 2*x^3 - 3*x^2

-- Theorem statement
theorem tangent_lines_range (t : ℝ) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (∀ x : ℝ, f x - (f a + f' a * (x - a)) = 0 → x = a) ∧
    (∀ x : ℝ, f x - (f b + f' b * (x - b)) = 0 → x = b) ∧
    (∀ x : ℝ, f x - (f c + f' c * (x - c)) = 0 → x = c) ∧
    t = f a + f' a * (3 - a) ∧
    t = f b + f' b * (3 - b) ∧
    t = f c + f' c * (3 - c)) →
  -9 < t ∧ t < 8 :=
sorry

end NUMINAMATH_CALUDE_tangent_lines_range_l972_97212


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_greater_than_neg_four_l972_97268

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + (a+2)*x + 1 = 0}
def B : Set ℝ := {x : ℝ | x > 0}

-- State the theorem
theorem intersection_empty_iff_a_greater_than_neg_four (a : ℝ) :
  A a ∩ B = ∅ ↔ a > -4 := by sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_greater_than_neg_four_l972_97268


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l972_97285

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.cos x - Real.cos (3 * x)) / x else 0

theorem f_derivative_at_zero : 
  deriv f 0 = 4 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l972_97285


namespace NUMINAMATH_CALUDE_square_side_length_l972_97236

theorem square_side_length (area : ℝ) (side : ℝ) : 
  area = 361 → side * side = area → side = 19 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l972_97236


namespace NUMINAMATH_CALUDE_fate_region_is_correct_l972_97298

def f (x : ℝ) := x^2 + 3*x + 2
def g (x : ℝ) := 2*x + 3

def is_fate_function (f g : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, |f x - g x| ≤ 1

def fate_region (f g : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | |f x - g x| ≤ 1}

theorem fate_region_is_correct :
  fate_region f g = Set.union (Set.Icc (-2) (-1)) (Set.Icc 0 1) :=
by sorry

end NUMINAMATH_CALUDE_fate_region_is_correct_l972_97298


namespace NUMINAMATH_CALUDE_road_length_is_10km_l972_97224

/-- Represents the road construction project -/
structure RoadProject where
  totalDays : ℕ
  initialWorkers : ℕ
  daysElapsed : ℕ
  completedLength : ℝ
  extraWorkers : ℕ

/-- Calculates the total length of the road given the project parameters -/
def calculateRoadLength (project : RoadProject) : ℝ :=
  sorry

/-- Theorem stating that the road length is 10 km given the specific project conditions -/
theorem road_length_is_10km (project : RoadProject) 
  (h1 : project.totalDays = 300)
  (h2 : project.initialWorkers = 30)
  (h3 : project.daysElapsed = 100)
  (h4 : project.completedLength = 2)
  (h5 : project.extraWorkers = 30) :
  calculateRoadLength project = 10 := by
  sorry

end NUMINAMATH_CALUDE_road_length_is_10km_l972_97224


namespace NUMINAMATH_CALUDE_randy_piano_expertise_l972_97292

/-- Represents the number of days in a year --/
def daysPerYear : ℕ := 365

/-- Represents the number of weeks in a year --/
def weeksPerYear : ℕ := 52

/-- Represents Randy's current age --/
def currentAge : ℕ := 12

/-- Represents Randy's target age to become an expert --/
def targetAge : ℕ := 20

/-- Represents the number of practice days per week --/
def practiceDaysPerWeek : ℕ := 5

/-- Represents the number of practice hours per day --/
def practiceHoursPerDay : ℕ := 5

/-- Represents the total hours needed to become an expert --/
def expertiseHours : ℕ := 10000

/-- Theorem stating that Randy can take 10 days of vacation per year and still achieve expertise --/
theorem randy_piano_expertise :
  ∃ (vacationDaysPerYear : ℕ),
    vacationDaysPerYear = 10 ∧
    (targetAge - currentAge) * weeksPerYear * practiceDaysPerWeek * practiceHoursPerDay -
    (targetAge - currentAge) * vacationDaysPerYear * practiceHoursPerDay ≥ expertiseHours :=
by sorry

end NUMINAMATH_CALUDE_randy_piano_expertise_l972_97292


namespace NUMINAMATH_CALUDE_heart_ratio_equals_one_l972_97207

def heart (n m : ℕ) : ℕ := n^3 * m^3

theorem heart_ratio_equals_one : (heart 3 5) / (heart 5 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_heart_ratio_equals_one_l972_97207


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l972_97249

theorem least_positive_integer_with_remainders : ∃! a : ℕ,
  a > 0 ∧
  a % 2 = 1 ∧
  a % 3 = 2 ∧
  a % 4 = 3 ∧
  a % 5 = 4 ∧
  ∀ b : ℕ, b > 0 ∧ b % 2 = 1 ∧ b % 3 = 2 ∧ b % 4 = 3 ∧ b % 5 = 4 → a ≤ b :=
by
  use 59
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l972_97249


namespace NUMINAMATH_CALUDE_root_implies_m_value_l972_97203

theorem root_implies_m_value (m : ℝ) : 
  (∃ x : ℝ, x^2 - m*x - 3 = 0 ∧ x = 3) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_m_value_l972_97203


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l972_97299

/-- A geometric sequence with specified terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_2 : a 2 = 2)
  (h_3 : a 3 = -4) :
  a 5 = -16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l972_97299


namespace NUMINAMATH_CALUDE_space_diagonal_length_l972_97227

/-- The length of the space diagonal in a rectangular prism with edge lengths 2, 3, and 4 is √29. -/
theorem space_diagonal_length (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 4) :
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 29 := by
  sorry


end NUMINAMATH_CALUDE_space_diagonal_length_l972_97227


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l972_97221

theorem sum_of_coefficients (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 15625 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l972_97221


namespace NUMINAMATH_CALUDE_f_properties_l972_97276

noncomputable section

def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.log x + a) / x

def monotonicity_intervals (a : ℝ) : Prop :=
  (a > 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (1 - a) → f a x₁ < f a x₂) ∧
            (∀ x₁ x₂, Real.exp (1 - a) < x₁ ∧ x₁ < x₂ → f a x₁ > f a x₂)) ∧
  (a < 0 → (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.exp (1 - a) → f a x₁ > f a x₂) ∧
            (∀ x₁ x₂, Real.exp (1 - a) < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂))

def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x, Real.exp 1 < x ∧ f a x = 0

theorem f_properties (a : ℝ) (h : a ≠ 0) :
  monotonicity_intervals a ∧ (has_root_in_interval a ↔ a < -1) :=
sorry

end

end NUMINAMATH_CALUDE_f_properties_l972_97276


namespace NUMINAMATH_CALUDE_simplify_expression_l972_97219

theorem simplify_expression (b : ℝ) : 3*b*(3*b^2 + 2*b) - b^2 = 9*b^3 + 5*b^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l972_97219


namespace NUMINAMATH_CALUDE_ordered_numbers_count_l972_97231

/-- Counts the numbers from 000 to 999 with digits in non-decreasing or non-increasing order -/
def count_ordered_numbers : ℕ :=
  let non_decreasing := Nat.choose 12 9
  let non_increasing := Nat.choose 12 9
  let double_counted := 10  -- Numbers with all identical digits
  non_decreasing + non_increasing - double_counted

/-- The count of numbers from 000 to 999 with digits in non-decreasing or non-increasing order is 430 -/
theorem ordered_numbers_count : count_ordered_numbers = 430 := by
  sorry

end NUMINAMATH_CALUDE_ordered_numbers_count_l972_97231


namespace NUMINAMATH_CALUDE_second_polygon_sides_l972_97275

/-- Given two regular polygons with the same perimeter, where the first has 45 sides
    and a side length three times as long as the second, prove that the second polygon
    has 135 sides. -/
theorem second_polygon_sides (p1 p2 : ℕ) (s : ℝ) : 
  p1 = 45 →                          -- The first polygon has 45 sides
  p1 * (3 * s) = p2 * s →            -- Both polygons have the same perimeter
  p2 = 135 := by sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l972_97275


namespace NUMINAMATH_CALUDE_fifteen_buses_needed_l972_97235

/-- Given the number of students, bus capacity, and pre-reserved bus seats,
    calculate the total number of buses needed. -/
def total_buses_needed (total_students : ℕ) (bus_capacity : ℕ) (pre_reserved_seats : ℕ) : ℕ :=
  let remaining_students := total_students - pre_reserved_seats
  let new_buses := (remaining_students + bus_capacity - 1) / bus_capacity
  new_buses + 1

/-- Theorem stating that 15 buses are needed for the given conditions. -/
theorem fifteen_buses_needed :
  total_buses_needed 635 45 20 = 15 := by
  sorry

#eval total_buses_needed 635 45 20

end NUMINAMATH_CALUDE_fifteen_buses_needed_l972_97235


namespace NUMINAMATH_CALUDE_textbook_recycling_savings_scientific_notation_l972_97237

theorem textbook_recycling_savings_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 
    1 ≤ a ∧ a < 10 ∧ 
    (31680000000 : ℝ) = a * (10 : ℝ) ^ n ∧
    a = 3.168 ∧ n = 10 := by
  sorry

end NUMINAMATH_CALUDE_textbook_recycling_savings_scientific_notation_l972_97237


namespace NUMINAMATH_CALUDE_a_properties_l972_97284

def a (n : ℕ+) : ℚ := (n - 1) / n

theorem a_properties :
  (∀ n : ℕ+, a n < 1) ∧
  (∀ n : ℕ+, a (n + 1) > a n) :=
by sorry

end NUMINAMATH_CALUDE_a_properties_l972_97284


namespace NUMINAMATH_CALUDE_lemon_candy_count_l972_97261

theorem lemon_candy_count (total : ℕ) (caramel : ℕ) (p : ℚ) (lemon : ℕ) : 
  caramel = 3 →
  p = 3 / 7 →
  p = caramel / total →
  lemon = total - caramel →
  lemon = 4 := by
sorry

end NUMINAMATH_CALUDE_lemon_candy_count_l972_97261


namespace NUMINAMATH_CALUDE_max_principals_is_three_l972_97260

/-- Represents the duration of a principal's term in years -/
def term_length : ℕ := 4

/-- Represents the period of interest in years -/
def period_length : ℕ := 9

/-- Calculates the maximum number of principals that can serve during the period -/
def max_principals : ℕ := 
  (period_length + term_length - 1) / term_length

/-- Theorem stating that the maximum number of principals is 3 -/
theorem max_principals_is_three : max_principals = 3 := by
  sorry

end NUMINAMATH_CALUDE_max_principals_is_three_l972_97260


namespace NUMINAMATH_CALUDE_letter_lock_max_attempts_l972_97283

/-- A letter lock with a given number of rings and letters per ring. -/
structure LetterLock :=
  (num_rings : ℕ)
  (letters_per_ring : ℕ)

/-- The maximum number of distinct unsuccessful attempts for a letter lock. -/
def max_unsuccessful_attempts (lock : LetterLock) : ℕ :=
  lock.letters_per_ring ^ lock.num_rings - 1

/-- Theorem: For a letter lock with 3 rings and 6 letters per ring,
    the maximum number of distinct unsuccessful attempts is 215. -/
theorem letter_lock_max_attempts :
  let lock := LetterLock.mk 3 6
  max_unsuccessful_attempts lock = 215 := by
  sorry

end NUMINAMATH_CALUDE_letter_lock_max_attempts_l972_97283


namespace NUMINAMATH_CALUDE_point_in_fourth_quadrant_l972_97218

/-- A point P(x, y) is in the fourth quadrant if x > 0 and y < 0 -/
def in_fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- The x-coordinate of point P -/
def x_coord (x : ℝ) : ℝ := 2 * x + 6

/-- The y-coordinate of point P -/
def y_coord (x : ℝ) : ℝ := 5 * x

theorem point_in_fourth_quadrant (x : ℝ) :
  in_fourth_quadrant (x_coord x) (y_coord x) ↔ -3 < x ∧ x < 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_fourth_quadrant_l972_97218


namespace NUMINAMATH_CALUDE_min_throws_correct_l972_97262

/-- The probability of hitting the target on a single throw -/
def p : ℝ := 0.6

/-- The desired minimum probability of hitting the target at least once -/
def min_prob : ℝ := 0.9

/-- The function that calculates the probability of hitting the target at least once in n throws -/
def prob_hit_at_least_once (n : ℕ) : ℝ := 1 - (1 - p)^n

/-- The minimum number of throws needed to exceed the desired probability -/
def min_throws : ℕ := 3

theorem min_throws_correct :
  (∀ k < min_throws, prob_hit_at_least_once k ≤ min_prob) ∧
  prob_hit_at_least_once min_throws > min_prob :=
sorry

end NUMINAMATH_CALUDE_min_throws_correct_l972_97262


namespace NUMINAMATH_CALUDE_linear_equation_solution_l972_97246

theorem linear_equation_solution (x y a : ℝ) : 
  x = 1 → y = 2 → x + a = 3 * y - 2 → a = 3 := by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l972_97246


namespace NUMINAMATH_CALUDE_horner_first_step_for_f_l972_97251

def f (x : ℝ) : ℝ := 0.5 * x^6 + 4 * x^5 - x^4 + 3 * x^3 - 5 * x

def horner_first_step (a₆ a₅ : ℝ) (x : ℝ) : ℝ := a₆ * x + a₅

theorem horner_first_step_for_f :
  horner_first_step 0.5 4 3 = 5.5 :=
sorry

end NUMINAMATH_CALUDE_horner_first_step_for_f_l972_97251


namespace NUMINAMATH_CALUDE_integer_solutions_system_l972_97295

theorem integer_solutions_system : 
  ∀ x y z : ℤ, 
    x^2 - y^2 - z^2 = 1 ∧ y + z - x = 3 →
    ((x = 9 ∧ y = 8 ∧ z = 4) ∨
     (x = -3 ∧ y = -2 ∧ z = 2) ∨
     (x = 9 ∧ y = 4 ∧ z = 8) ∨
     (x = -3 ∧ y = 2 ∧ z = -2)) :=
by
  sorry

end NUMINAMATH_CALUDE_integer_solutions_system_l972_97295


namespace NUMINAMATH_CALUDE_pencils_used_l972_97232

theorem pencils_used (initial : ℕ) (current : ℕ) (h1 : initial = 94) (h2 : current = 91) :
  initial - current = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_used_l972_97232


namespace NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l972_97256

theorem mod_equivalence_unique_solution : 
  ∃! n : ℕ, n ≤ 6 ∧ n ≡ -4752 [ZMOD 7] := by sorry

end NUMINAMATH_CALUDE_mod_equivalence_unique_solution_l972_97256


namespace NUMINAMATH_CALUDE_f_values_f_inequality_range_l972_97239

noncomputable section

variable (f : ℝ → ℝ)

axiom domain : ∀ x, x > 0 → f x ≠ 0
axiom f_2 : f 2 = 1
axiom f_mult : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_increasing : ∀ x₁ x₂, 0 < x₁ → x₁ < x₂ → f x₁ < f x₂

theorem f_values :
  f 1 = 0 ∧ f 4 = 2 ∧ f 8 = 3 :=
sorry

theorem f_inequality_range :
  ∀ x, (f x + f (x - 2) ≤ 3) ↔ (2 < x ∧ x ≤ 4) :=
sorry

end NUMINAMATH_CALUDE_f_values_f_inequality_range_l972_97239


namespace NUMINAMATH_CALUDE_compound_molar_mass_l972_97225

/-- Given a compound where 6 moles weighs 612 grams, prove its molar mass is 102 grams per mole -/
theorem compound_molar_mass (mass : ℝ) (moles : ℝ) (h1 : mass = 612) (h2 : moles = 6) :
  mass / moles = 102 := by
  sorry

end NUMINAMATH_CALUDE_compound_molar_mass_l972_97225


namespace NUMINAMATH_CALUDE_jar_contents_l972_97206

-- Define the number of candy pieces
def candy_pieces : Float := 3409.0

-- Define the number of secret eggs
def secret_eggs : Float := 145.0

-- Define the total number of items
def total_items : Float := candy_pieces + secret_eggs

-- Theorem statement
theorem jar_contents : total_items = 3554.0 := by
  sorry

end NUMINAMATH_CALUDE_jar_contents_l972_97206


namespace NUMINAMATH_CALUDE_plane_perpendicular_parallel_transitive_l972_97233

/-- A structure representing a 3D space with planes and perpendicularity/parallelism relations -/
structure Space3D where
  Plane : Type
  perpendicular : Plane → Plane → Prop
  parallel : Plane → Plane → Prop

/-- The main theorem to be proved -/
theorem plane_perpendicular_parallel_transitive 
  (S : Space3D) (α β γ : S.Plane) : 
  S.perpendicular α β → S.parallel α γ → S.perpendicular β γ := by
  sorry

/-- Helper lemma: If two planes are parallel, they are not perpendicular -/
lemma parallel_not_perpendicular 
  (S : Space3D) (p q : S.Plane) :
  S.parallel p q → ¬S.perpendicular p q := by
  sorry

/-- Helper lemma: Perpendicularity is symmetric -/
lemma perpendicular_symmetric 
  (S : Space3D) (p q : S.Plane) :
  S.perpendicular p q → S.perpendicular q p := by
  sorry

/-- Helper lemma: Parallelism is symmetric -/
lemma parallel_symmetric 
  (S : Space3D) (p q : S.Plane) :
  S.parallel p q → S.parallel q p := by
  sorry

end NUMINAMATH_CALUDE_plane_perpendicular_parallel_transitive_l972_97233


namespace NUMINAMATH_CALUDE_like_terms_power_l972_97208

/-- 
Given two monomials x^(a+3)y and -5xy^b that are like terms,
prove that (a+b)^2023 = -1
-/
theorem like_terms_power (a b : ℤ) : 
  (a + 3 = 1 ∧ b = 1) → (a + b)^2023 = -1 := by sorry

end NUMINAMATH_CALUDE_like_terms_power_l972_97208


namespace NUMINAMATH_CALUDE_equation_solution_range_l972_97269

theorem equation_solution_range (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ x ≠ 2 ∧ (x + m) / (x - 2) - 3 = (x - 1) / (2 - x)) ↔ 
  (m ≥ -5 ∧ m ≠ -3) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l972_97269


namespace NUMINAMATH_CALUDE_parabola_vertex_coordinates_l972_97243

/-- The vertex coordinates of the parabola y = x^2 + 2x - 2 are (-1, -3) -/
theorem parabola_vertex_coordinates :
  let f : ℝ → ℝ := λ x => x^2 + 2*x - 2
  ∃ (h : ℝ → ℝ), (∀ x, f x = h (x + 1) - 3) ∧ (∀ x, h x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_coordinates_l972_97243


namespace NUMINAMATH_CALUDE_max_area_right_triangle_l972_97223

theorem max_area_right_triangle (c : ℝ) (h : c = 8) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = c^2 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → x^2 + y^2 = c^2 →
  (1/2) * x * y ≤ (1/2) * a * b ∧
  (1/2) * a * b = 16 := by
sorry

end NUMINAMATH_CALUDE_max_area_right_triangle_l972_97223


namespace NUMINAMATH_CALUDE_quadratic_equation_with_prime_coefficients_l972_97213

theorem quadratic_equation_with_prime_coefficients (p q : ℕ) (hp : Prime p) (hq : Prime q) :
  (∃ x : ℤ, x^2 - p*x + q = 0) → p = 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_with_prime_coefficients_l972_97213


namespace NUMINAMATH_CALUDE_south_cyclist_speed_l972_97277

/-- The speed of a cyclist going south, given two cyclists start from the same place
    in opposite directions, one going north at 10 kmph, and they are 50 km apart after 1 hour. -/
def speed_of_south_cyclist : ℝ :=
  let speed_north : ℝ := 10
  let time : ℝ := 1
  let distance_apart : ℝ := 50
  distance_apart - speed_north * time

theorem south_cyclist_speed : speed_of_south_cyclist = 40 := by
  sorry

end NUMINAMATH_CALUDE_south_cyclist_speed_l972_97277


namespace NUMINAMATH_CALUDE_least_product_of_three_primes_over_50_l972_97229

theorem least_product_of_three_primes_over_50 :
  ∃ (p q r : Nat),
    Prime p ∧ Prime q ∧ Prime r ∧
    p > 50 ∧ q > 50 ∧ r > 50 ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    p * q * r = 190847 ∧
    ∀ (a b c : Nat),
      Prime a → Prime b → Prime c →
      a > 50 → b > 50 → c > 50 →
      a ≠ b → a ≠ c → b ≠ c →
      a * b * c ≥ 190847 :=
by
  sorry

end NUMINAMATH_CALUDE_least_product_of_three_primes_over_50_l972_97229


namespace NUMINAMATH_CALUDE_pyramid_faces_l972_97263

/-- A polygonal pyramid with a regular polygon base -/
structure PolygonalPyramid where
  base_sides : ℕ
  vertices : ℕ
  edges : ℕ
  faces : ℕ

/-- Properties of the polygonal pyramid -/
def pyramid_properties (p : PolygonalPyramid) : Prop :=
  p.vertices = p.base_sides + 1 ∧
  p.edges = 2 * p.base_sides ∧
  p.faces = p.base_sides + 1 ∧
  p.edges + p.vertices = 1915

theorem pyramid_faces (p : PolygonalPyramid) (h : pyramid_properties p) : p.faces = 639 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_faces_l972_97263


namespace NUMINAMATH_CALUDE_hamburgers_served_l972_97270

theorem hamburgers_served (total : Nat) (leftover : Nat) (served : Nat) :
  total = 9 → leftover = 6 → served = total - leftover → served = 3 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_served_l972_97270


namespace NUMINAMATH_CALUDE_admission_fee_problem_l972_97252

/-- Admission fee problem -/
theorem admission_fee_problem (child_fee : ℚ) (total_people : ℕ) (total_amount : ℚ) 
  (num_children : ℕ) (num_adults : ℕ) :
  child_fee = 3/2 →
  total_people = 2200 →
  total_amount = 5050 →
  num_children = 700 →
  num_adults = 1500 →
  num_children + num_adults = total_people →
  ∃ adult_fee : ℚ, 
    adult_fee * num_adults + child_fee * num_children = total_amount ∧
    adult_fee = 8/3 :=
by sorry

end NUMINAMATH_CALUDE_admission_fee_problem_l972_97252


namespace NUMINAMATH_CALUDE_contrapositive_truth_l972_97272

/-- The function f(x) = x^2 - mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - m*x + 1

/-- The theorem statement -/
theorem contrapositive_truth (m : ℝ) 
  (h : ∀ x > 0, f m x ≥ 0) :
  ∀ a b, a > 0 → b > 0 → 
    (a + b ≤ 1 → 1/a + 2/b ≥ 3 + Real.sqrt 2 * m) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_truth_l972_97272


namespace NUMINAMATH_CALUDE_maria_cookies_distribution_l972_97288

/-- Calculates the number of cookies per bag given the total number of cookies and the number of bags. -/
def cookiesPerBag (totalCookies : ℕ) (numBags : ℕ) : ℕ :=
  totalCookies / numBags

theorem maria_cookies_distribution (chocolateChipCookies oatmealCookies numBags : ℕ) 
  (h1 : chocolateChipCookies = 33)
  (h2 : oatmealCookies = 2)
  (h3 : numBags = 7) :
  cookiesPerBag (chocolateChipCookies + oatmealCookies) numBags = 5 := by
  sorry

end NUMINAMATH_CALUDE_maria_cookies_distribution_l972_97288


namespace NUMINAMATH_CALUDE_solution_set_inequality_l972_97211

theorem solution_set_inequality (x : ℝ) :
  (-x^2 + 3*x - 2 ≥ 0) ↔ (1 ≤ x ∧ x ≤ 2) := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l972_97211


namespace NUMINAMATH_CALUDE_cucumber_weight_problem_l972_97214

theorem cucumber_weight_problem (initial_water_percentage : Real)
                                (final_water_percentage : Real)
                                (final_weight : Real) :
  initial_water_percentage = 0.99 →
  final_water_percentage = 0.95 →
  final_weight = 20 →
  ∃ initial_weight : Real,
    initial_weight = 100 ∧
    (1 - initial_water_percentage) * initial_weight =
    (1 - final_water_percentage) * final_weight :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_problem_l972_97214


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l972_97286

theorem complex_product_pure_imaginary (x : ℝ) : 
  let z₁ : ℂ := 1 - I
  let z₂ : ℂ := -1 - x * I
  (z₁ * z₂).re = 0 → x = -1 := by sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l972_97286


namespace NUMINAMATH_CALUDE_three_number_problem_l972_97255

theorem three_number_problem (x y z : ℝ) 
  (sum_eq : x + y + z = 500)
  (x_eq : x = 200)
  (y_eq : y = 2 * z)
  (diff_eq : x - z = 0.5 * y) :
  z = 100 := by
  sorry

end NUMINAMATH_CALUDE_three_number_problem_l972_97255


namespace NUMINAMATH_CALUDE_roots_product_zero_l972_97240

theorem roots_product_zero (a b c d : ℝ) : 
  (a^2 + 57*a + 1 = 0) →
  (b^2 + 57*b + 1 = 0) →
  (c^2 - 57*c + 1 = 0) →
  (d^2 - 57*d + 1 = 0) →
  (a + c) * (b + c) * (a - d) * (b - d) = 0 := by
sorry

end NUMINAMATH_CALUDE_roots_product_zero_l972_97240


namespace NUMINAMATH_CALUDE_f_2015_value_l972_97202

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def has_period_shift (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 4) = f x + f 2

theorem f_2015_value (f : ℝ → ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period_shift f)
  (h_f1 : f 1 = 2) :
  f 2015 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_2015_value_l972_97202


namespace NUMINAMATH_CALUDE_cubic_root_sum_l972_97279

theorem cubic_root_sum (p q r : ℝ) : 
  (3 * p^3 - 5 * p^2 + 50 * p - 7 = 0) →
  (3 * q^3 - 5 * q^2 + 50 * q - 7 = 0) →
  (3 * r^3 - 5 * r^2 + 50 * r - 7 = 0) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = 249/9 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l972_97279


namespace NUMINAMATH_CALUDE_race_start_distances_l972_97265

-- Define the start distances
def start_A_B : ℝ := 50
def start_B_C : ℝ := 157.89473684210532

-- Theorem statement
theorem race_start_distances :
  let start_A_C := start_A_B + start_B_C
  start_A_C = 207.89473684210532 := by sorry

end NUMINAMATH_CALUDE_race_start_distances_l972_97265


namespace NUMINAMATH_CALUDE_square_sum_equation_solutions_l972_97234

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The property that a number is the sum of two squares -/
def isSumOfTwoSquares (n : ℕ) : Prop :=
  ∃ x y : ℕ, x^2 + y^2 = n

/-- The main theorem -/
theorem square_sum_equation_solutions :
  (∃ k : ℕ+, 
    (∃ a b c : ℕ+, a^2 + b^2 + c^2 = k * a * b * c) ∧ 
    (∀ n : ℕ, ∃ a_n b_n c_n : ℕ+,
      a_n^2 + b_n^2 + c_n^2 = k * a_n * b_n * c_n ∧
      isSumOfTwoSquares (a_n * b_n) ∧
      isSumOfTwoSquares (b_n * c_n) ∧
      isSumOfTwoSquares (c_n * a_n))) ↔
  (k = 1 ∨ k = 3) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equation_solutions_l972_97234


namespace NUMINAMATH_CALUDE_polar_midpoint_specific_case_l972_97273

/-- The midpoint of a line segment in polar coordinates --/
def polar_midpoint (r₁ : ℝ) (θ₁ : ℝ) (r₂ : ℝ) (θ₂ : ℝ) : ℝ × ℝ :=
  sorry

theorem polar_midpoint_specific_case :
  let (r, θ) := polar_midpoint 10 (π/3) 10 (5*π/6)
  r = 5 * Real.sqrt 2 ∧ θ = 2*π/3 :=
sorry

end NUMINAMATH_CALUDE_polar_midpoint_specific_case_l972_97273


namespace NUMINAMATH_CALUDE_potatoes_for_mashed_l972_97215

theorem potatoes_for_mashed (initial : ℕ) (salad : ℕ) (remaining : ℕ) : 
  initial = 52 → salad = 15 → remaining = 13 → initial - salad - remaining = 24 := by
  sorry

end NUMINAMATH_CALUDE_potatoes_for_mashed_l972_97215


namespace NUMINAMATH_CALUDE_tickets_for_pesos_l972_97293

/-- Given that T tickets cost R dollars and 10 pesos is worth 40 dollars,
    this theorem proves that the number of tickets that can be purchased
    for P pesos is 4PT/R. -/
theorem tickets_for_pesos (T R P : ℝ) (h1 : T > 0) (h2 : R > 0) (h3 : P > 0) :
  let dollars_per_peso : ℝ := 40 / 10
  let pesos_in_dollars : ℝ := P * dollars_per_peso
  let tickets_per_dollar : ℝ := T / R
  tickets_per_dollar * pesos_in_dollars = 4 * P * T / R :=
by sorry

end NUMINAMATH_CALUDE_tickets_for_pesos_l972_97293


namespace NUMINAMATH_CALUDE_square_root_729_l972_97247

theorem square_root_729 (x : ℝ) (h1 : x > 0) (h2 : x^2 = 729) : x = 27 := by
  sorry

end NUMINAMATH_CALUDE_square_root_729_l972_97247


namespace NUMINAMATH_CALUDE_range_of_f_when_a_is_2_properties_of_M_l972_97281

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 - a*x + 4 - a^2

-- Theorem for the range of f when a = 2
theorem range_of_f_when_a_is_2 :
  ∃ (y : ℝ), y ∈ Set.Icc (-1) 8 ↔ ∃ (x : ℝ), x ∈ Set.Icc (-2) 3 ∧ f 2 x = y :=
sorry

-- Define the set M
def M : Set ℝ := {4}

-- Theorem for the properties of M
theorem properties_of_M :
  (4 ∈ M) ∧
  (∀ a ∈ M, ∀ x ∈ Set.Icc (-2) 2, f a x ≤ 0) ∧
  (∃ b ∉ M, ∀ x ∈ Set.Icc (-2) 2, f b x ≤ 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_f_when_a_is_2_properties_of_M_l972_97281


namespace NUMINAMATH_CALUDE_swimmers_arrangement_count_l972_97253

/-- The number of swimmers -/
def num_swimmers : ℕ := 6

/-- The number of arrangements when A is leftmost -/
def arrangements_A_leftmost : ℕ := (num_swimmers - 1) * (Nat.factorial (num_swimmers - 2))

/-- The number of arrangements when B is leftmost -/
def arrangements_B_leftmost : ℕ := (num_swimmers - 2) * (Nat.factorial (num_swimmers - 2))

/-- The total number of arrangements -/
def total_arrangements : ℕ := arrangements_A_leftmost + arrangements_B_leftmost

theorem swimmers_arrangement_count :
  total_arrangements = 216 :=
sorry

end NUMINAMATH_CALUDE_swimmers_arrangement_count_l972_97253


namespace NUMINAMATH_CALUDE_voting_change_l972_97250

theorem voting_change (total_members : ℕ) 
  (h_total : total_members = 400)
  (initial_for initial_against : ℕ) 
  (h_initial_sum : initial_for + initial_against = total_members)
  (h_initial_reject : initial_against > initial_for)
  (second_for second_against : ℕ) 
  (h_second_sum : second_for + second_against = total_members)
  (h_second_pass : second_for > second_against)
  (h_margin : second_for - second_against = 3 * (initial_against - initial_for))
  (h_proportion : second_for = (10 * initial_against) / 9) :
  second_for - initial_for = 48 := by
sorry

end NUMINAMATH_CALUDE_voting_change_l972_97250


namespace NUMINAMATH_CALUDE_dark_tile_fraction_l972_97264

theorem dark_tile_fraction (block_size : ℕ) (dark_tiles : ℕ) : 
  block_size = 8 → 
  dark_tiles = 18 → 
  (dark_tiles : ℚ) / (block_size * block_size : ℚ) = 9 / 32 :=
by sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_l972_97264


namespace NUMINAMATH_CALUDE_simplify_expression_l972_97274

theorem simplify_expression (n : ℕ) : 
  (3^(n+4) - 3*(3^n) + 3^(n+2)) / (3*(3^(n+3))) = 29 / 27 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l972_97274


namespace NUMINAMATH_CALUDE_initial_commission_rate_l972_97290

theorem initial_commission_rate 
  (income_unchanged : ℝ → ℝ → ℝ → ℝ → Bool)
  (new_rate : ℝ)
  (business_slump : ℝ)
  (initial_rate : ℝ) :
  income_unchanged initial_rate new_rate business_slump initial_rate →
  new_rate = 5 →
  business_slump = 20.000000000000007 →
  initial_rate = 4 := by
sorry

end NUMINAMATH_CALUDE_initial_commission_rate_l972_97290


namespace NUMINAMATH_CALUDE_negation_of_proposition_l972_97204

theorem negation_of_proposition (a b : ℝ) :
  ¬(a + b > 0 → a > 0 ∧ b > 0) ↔ (a + b ≤ 0 → a ≤ 0 ∨ b ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l972_97204


namespace NUMINAMATH_CALUDE_duty_schedules_count_l972_97205

/-- Represents the number of people on duty -/
def num_people : ℕ := 3

/-- Represents the number of days in the duty schedule -/
def num_days : ℕ := 6

/-- Represents the number of duty days per person -/
def duty_days_per_person : ℕ := 2

/-- Calculates the number of valid duty schedules -/
def count_duty_schedules : ℕ :=
  let total_arrangements := (num_days.choose duty_days_per_person) * ((num_days - duty_days_per_person).choose duty_days_per_person)
  let invalid_arrangements := 2 * ((num_days - 1).choose duty_days_per_person) * ((num_days - duty_days_per_person - 1).choose duty_days_per_person)
  let double_counted := ((num_days - 2).choose duty_days_per_person) * ((num_days - duty_days_per_person - 2).choose duty_days_per_person)
  total_arrangements - invalid_arrangements + double_counted

theorem duty_schedules_count :
  count_duty_schedules = 42 :=
sorry

end NUMINAMATH_CALUDE_duty_schedules_count_l972_97205


namespace NUMINAMATH_CALUDE_amithab_january_expenditure_l972_97245

def january_expenditure (avg_jan_jun avg_feb_jul july_expenditure : ℝ) : ℝ :=
  6 * avg_feb_jul - 6 * avg_jan_jun + july_expenditure

theorem amithab_january_expenditure :
  january_expenditure 4200 4250 1500 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_amithab_january_expenditure_l972_97245


namespace NUMINAMATH_CALUDE_min_volume_ratio_l972_97220

/-- A spherical cap -/
structure SphericalCap where
  volume : ℝ

/-- A cylinder -/
structure Cylinder where
  volume : ℝ

/-- Configuration of a spherical cap and cylinder sharing a common inscribed sphere -/
structure Configuration where
  cap : SphericalCap
  cylinder : Cylinder
  bottom_faces_on_same_plane : Prop
  share_common_inscribed_sphere : Prop

/-- The minimum volume ratio theorem -/
theorem min_volume_ratio (config : Configuration) :
  ∃ (min_ratio : ℝ), min_ratio = 4/3 ∧
  ∀ (ratio : ℝ), ratio = config.cap.volume / config.cylinder.volume → min_ratio ≤ ratio :=
sorry

end NUMINAMATH_CALUDE_min_volume_ratio_l972_97220


namespace NUMINAMATH_CALUDE_smallest_b_value_l972_97209

theorem smallest_b_value (a b : ℝ) (h1 : 2 < a) (h2 : a < b) (h3 : a + b = 7)
  (h4 : ¬ (2 + a > b ∧ 2 + b > a ∧ a + b > 2)) : 
  (∀ b' : ℝ, (∃ a' : ℝ, 2 < a' ∧ a' < b' ∧ a' + b' = 7 ∧
    ¬ (2 + a' > b' ∧ 2 + b' > a' ∧ a' + b' > 2)) → b' ≥ 9/2) ∧ b = 9/2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_b_value_l972_97209


namespace NUMINAMATH_CALUDE_cost_sharing_ratio_l972_97278

def monthly_cost : ℚ := 14
def your_payment : ℚ := 84
def total_months : ℕ := 12

theorem cost_sharing_ratio :
  let yearly_cost := monthly_cost * total_months
  let friend_payment := yearly_cost - your_payment
  your_payment = friend_payment :=
by sorry

end NUMINAMATH_CALUDE_cost_sharing_ratio_l972_97278


namespace NUMINAMATH_CALUDE_twice_square_sum_l972_97289

theorem twice_square_sum (x y : ℤ) : x^4 + y^4 + (x+y)^4 = 2 * (x^2 + x*y + y^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_twice_square_sum_l972_97289


namespace NUMINAMATH_CALUDE_solve_for_m_l972_97271

-- Define the functions f and g
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m
def g (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + 9*m

-- State the theorem
theorem solve_for_m : ∃ m : ℝ, f m 2 = 2 * g m 2 ∧ m = 0 := by sorry

end NUMINAMATH_CALUDE_solve_for_m_l972_97271


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l972_97244

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence condition
  a 3 + a 8 = 10 →              -- given condition
  3 * a 5 + a 7 = 20 :=         -- conclusion to prove
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l972_97244


namespace NUMINAMATH_CALUDE_candy_cost_proof_l972_97294

def candy_problem (num_packs : ℕ) (total_paid : ℕ) (change : ℕ) : Prop :=
  let total_cost : ℕ := total_paid - change
  let cost_per_pack : ℕ := total_cost / num_packs
  cost_per_pack = 3

theorem candy_cost_proof :
  candy_problem 3 20 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_cost_proof_l972_97294


namespace NUMINAMATH_CALUDE_xiaojun_original_money_l972_97248

/-- The amount of money Xiaojun originally had -/
def original_money : ℝ := 30

/-- The daily allowance Xiaojun receives from his dad -/
def daily_allowance : ℝ := 5

/-- The number of days Xiaojun can last when spending 10 yuan per day -/
def days_at_10 : ℝ := 6

/-- The number of days Xiaojun can last when spending 15 yuan per day -/
def days_at_15 : ℝ := 3

/-- The daily spending when Xiaojun lasts for 6 days -/
def spending_10 : ℝ := 10

/-- The daily spending when Xiaojun lasts for 3 days -/
def spending_15 : ℝ := 15

theorem xiaojun_original_money :
  (days_at_10 * spending_10 - days_at_10 * daily_allowance = original_money) ∧
  (days_at_15 * spending_15 - days_at_15 * daily_allowance = original_money) :=
by sorry

end NUMINAMATH_CALUDE_xiaojun_original_money_l972_97248


namespace NUMINAMATH_CALUDE_kendra_shirts_theorem_l972_97297

/-- The number of shirts Kendra needs for a three-week period -/
def shirts_needed : ℕ :=
  let school_shirts := 5  -- 5 weekdays
  let club_shirts := 3    -- 3 days a week
  let saturday_shirts := 3 -- 1 for workout, 1 for art class, 1 for rest of the day
  let sunday_shirts := 3   -- 1 for church, 1 for volunteer work, 1 for rest of the day
  let weekly_shirts := school_shirts + club_shirts + saturday_shirts + sunday_shirts
  let weeks := 3
  weekly_shirts * weeks

/-- Theorem stating that Kendra needs 42 shirts for a three-week period -/
theorem kendra_shirts_theorem : shirts_needed = 42 := by
  sorry

end NUMINAMATH_CALUDE_kendra_shirts_theorem_l972_97297


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l972_97200

theorem geometric_sequence_problem (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  q > 1 →  -- common ratio > 1
  4 * (a 2005)^2 - 8 * (a 2005) + 3 = 0 →  -- a₂₀₀₅ is a root
  4 * (a 2006)^2 - 8 * (a 2006) + 3 = 0 →  -- a₂₀₀₆ is a root
  a 2007 + a 2008 = 18 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l972_97200


namespace NUMINAMATH_CALUDE_carol_rectangle_width_l972_97217

/-- Given two rectangles with equal areas, where one has a length of 5 inches
    and the other has dimensions of 2 inches by 60 inches,
    prove that the width of the first rectangle is 24 inches. -/
theorem carol_rectangle_width (w : ℝ) :
  (5 * w = 2 * 60) → w = 24 :=
by sorry

end NUMINAMATH_CALUDE_carol_rectangle_width_l972_97217


namespace NUMINAMATH_CALUDE_function_equation_solution_l972_97282

theorem function_equation_solution (f : ℝ → ℝ) (a : ℝ) :
  (∀ x : ℝ, f (f x) = x * f x - a * x) →
  (∃ x y : ℝ, f x ≠ f y) →
  (∃ t : ℝ, f t = a) →
  (a = 0 ∨ a = -1) :=
by sorry

end NUMINAMATH_CALUDE_function_equation_solution_l972_97282


namespace NUMINAMATH_CALUDE_complex_number_magnitude_product_l972_97287

theorem complex_number_magnitude_product (z₁ z₂ : ℂ) : 
  Complex.abs (z₁ * z₂) = Complex.abs z₁ * Complex.abs z₂ := by sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_product_l972_97287


namespace NUMINAMATH_CALUDE_inequality_implication_l972_97226

theorem inequality_implication (x y z : ℝ) (h : x^2 + x*y + x*z < 0) : y^2 > 4*x*z := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l972_97226


namespace NUMINAMATH_CALUDE_total_dogs_in_kennel_l972_97238

-- Define the sets and their sizes
def T : ℕ := 45  -- Number of dogs with tags
def C : ℕ := 40  -- Number of dogs with collars
def B : ℕ := 6   -- Number of dogs with both tags and collars
def N : ℕ := 1   -- Number of dogs with neither tags nor collars

-- Theorem statement
theorem total_dogs_in_kennel : T + C - B + N = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_dogs_in_kennel_l972_97238


namespace NUMINAMATH_CALUDE_quadratic_roots_not_integers_l972_97266

/-- 
Given a quadratic polynomial p(x) = ax² + bx + c where a, b, and c are odd integers,
if the roots x₁ and x₂ exist, they cannot both be integers.
-/
theorem quadratic_roots_not_integers 
  (a b c : ℤ) 
  (ha : Odd a) (hb : Odd b) (hc : Odd c)
  (hroots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) :
  ¬∃ (y₁ y₂ : ℤ), (y₁ : ℝ) = x₁ ∧ (y₂ : ℝ) = x₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_not_integers_l972_97266


namespace NUMINAMATH_CALUDE_jacket_price_after_discounts_l972_97241

theorem jacket_price_after_discounts (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : 
  original_price = 20 →
  discount1 = 0.25 →
  discount2 = 0.40 →
  original_price * (1 - discount1) * (1 - discount2) = 9 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_after_discounts_l972_97241


namespace NUMINAMATH_CALUDE_simplify_fraction_l972_97210

theorem simplify_fraction : 
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l972_97210


namespace NUMINAMATH_CALUDE_simplify_fraction_l972_97228

theorem simplify_fraction : (210 : ℚ) / 7350 * 14 = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l972_97228
