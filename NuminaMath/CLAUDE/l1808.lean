import Mathlib

namespace NUMINAMATH_CALUDE_fish_total_weight_l1808_180891

/-- The weight of a fish with specific weight relationships between its parts -/
def fish_weight (head body tail : ℝ) : Prop :=
  tail = 1 ∧ 
  head = tail + body / 2 ∧ 
  body = head + tail

theorem fish_total_weight : 
  ∀ (head body tail : ℝ), 
  fish_weight head body tail → 
  head + body + tail = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_total_weight_l1808_180891


namespace NUMINAMATH_CALUDE_largest_valid_number_l1808_180856

def is_valid_number (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n ≤ 9999 ∧
  (n % 10 = (n / 10 % 10 + n / 100 % 10) % 10) ∧
  (n / 10 % 10 = (n / 100 % 10 + n / 1000 % 10) % 10)

theorem largest_valid_number : 
  is_valid_number 9099 ∧ ∀ m : ℕ, is_valid_number m → m ≤ 9099 :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l1808_180856


namespace NUMINAMATH_CALUDE_odd_symmetric_function_property_l1808_180849

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is symmetric about x = 3 if f(3+x) = f(3-x) for all x -/
def SymmetricAboutThree (f : ℝ → ℝ) : Prop :=
  ∀ x, f (3 + x) = f (3 - x)

theorem odd_symmetric_function_property (f : ℝ → ℝ) 
  (h_odd : OddFunction f)
  (h_sym : SymmetricAboutThree f)
  (h_def : ∀ x ∈ Set.Ioo 0 3, f x = 2^x) :
  ∀ x ∈ Set.Ioo (-6) (-3), f x = -(2^(x + 6)) := by
  sorry

end NUMINAMATH_CALUDE_odd_symmetric_function_property_l1808_180849


namespace NUMINAMATH_CALUDE_time_to_work_calculation_l1808_180809

-- Define the problem parameters
def speed_to_work : ℝ := 50
def speed_to_home : ℝ := 110
def total_time : ℝ := 2

-- Define the theorem
theorem time_to_work_calculation :
  ∃ (distance : ℝ) (time_to_work : ℝ),
    distance / speed_to_work + distance / speed_to_home = total_time ∧
    time_to_work = distance / speed_to_work ∧
    time_to_work * 60 = 82.5 := by
  sorry


end NUMINAMATH_CALUDE_time_to_work_calculation_l1808_180809


namespace NUMINAMATH_CALUDE_first_valid_year_is_2049_l1808_180844

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_year (year : ℕ) : Prop :=
  year > 2010 ∧ year < 3000 ∧ sum_of_digits year = 15

theorem first_valid_year_is_2049 :
  (∀ year : ℕ, year < 2049 → ¬(is_valid_year year)) ∧ 
  is_valid_year 2049 :=
sorry

end NUMINAMATH_CALUDE_first_valid_year_is_2049_l1808_180844


namespace NUMINAMATH_CALUDE_smallest_non_prime_a_l1808_180816

def a : ℕ → ℕ
  | 0 => 2
  | n + 1 => 2 * a n + 1

theorem smallest_non_prime_a (n : ℕ) : 
  (∀ k < n, Nat.Prime (a k)) ∧ ¬Nat.Prime (a n) → n = 5 ∧ a n = 95 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_prime_a_l1808_180816


namespace NUMINAMATH_CALUDE_abs_sum_reciprocals_ge_two_l1808_180858

theorem abs_sum_reciprocals_ge_two (a b : ℝ) (h : a * b ≠ 0) :
  |a / b + b / a| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_reciprocals_ge_two_l1808_180858


namespace NUMINAMATH_CALUDE_max_garden_area_l1808_180898

/-- The maximum area of a rectangular garden with one side along a wall and 400 feet of fencing for the other three sides. -/
theorem max_garden_area : 
  ∃ (l w : ℝ), l > 0 ∧ w > 0 ∧ l + 2*w = 400 ∧
  (∀ (l' w' : ℝ), l' > 0 → w' > 0 → l' + 2*w' = 400 → l'*w' ≤ l*w) ∧
  l*w = 20000 :=
by sorry

end NUMINAMATH_CALUDE_max_garden_area_l1808_180898


namespace NUMINAMATH_CALUDE_borrowed_amount_l1808_180846

/-- Proves that the amount borrowed is 5000 given the specified conditions --/
theorem borrowed_amount (loan_duration : ℕ) (borrow_rate lend_rate : ℚ) (gain_per_year : ℕ) : 
  loan_duration = 2 →
  borrow_rate = 4 / 100 →
  lend_rate = 8 / 100 →
  gain_per_year = 200 →
  ∃ (amount : ℕ), amount = 5000 ∧ 
    (amount * lend_rate * loan_duration) - (amount * borrow_rate * loan_duration) = gain_per_year * loan_duration :=
by sorry

end NUMINAMATH_CALUDE_borrowed_amount_l1808_180846


namespace NUMINAMATH_CALUDE_max_a_value_l1808_180884

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x

-- Define the theorem
theorem max_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, |f a (f a x)| ≤ 2) →
  a ≤ (3 + Real.sqrt 17) / 4 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l1808_180884


namespace NUMINAMATH_CALUDE_no_inverse_implies_x_equals_five_l1808_180819

def M (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![x, 5],
    ![6, 6]]

theorem no_inverse_implies_x_equals_five :
  ∀ x : ℝ, ¬(IsUnit (M x)) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_inverse_implies_x_equals_five_l1808_180819


namespace NUMINAMATH_CALUDE_range_of_a_l1808_180871

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 + 2*a*x - 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 < 0

-- Define the theorem
theorem range_of_a :
  ∀ a : ℝ, 
  (a > 0) →
  (∀ x : ℝ, p x a → q x) →
  (∃ x : ℝ, q x ∧ ¬(p x a)) →
  (0 < a ∧ a ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1808_180871


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1808_180807

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x - 4

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -3
  let k : ℝ := f' x₀
  (f x₀ = y₀) ∧ 
  (∀ x y : ℝ, y = k * (x - x₀) + y₀ ↔ 5*x + y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1808_180807


namespace NUMINAMATH_CALUDE_game_sequence_repeats_a_2009_equals_65_l1808_180820

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The sequence defined by the game rules -/
def game_sequence (i : ℕ) : ℕ := 
  match i with
  | 0 => 5  -- n₁ = 5
  | i + 1 => sum_of_digits ((game_sequence i)^2 + 1)

/-- The a_i values in the sequence -/
def a_sequence (i : ℕ) : ℕ := (game_sequence i)^2 + 1

theorem game_sequence_repeats : 
  ∀ k : ℕ, k ≥ 3 → game_sequence k = game_sequence (k % 3) := sorry

theorem a_2009_equals_65 : a_sequence 2009 = 65 := by sorry

end NUMINAMATH_CALUDE_game_sequence_repeats_a_2009_equals_65_l1808_180820


namespace NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l1808_180839

/-- A trapezoid with an inscribed and circumscribed circle, and one angle of 60 degrees -/
structure SpecialTrapezoid where
  /-- The trapezoid has an inscribed circle -/
  has_inscribed_circle : Bool
  /-- The trapezoid has a circumscribed circle -/
  has_circumscribed_circle : Bool
  /-- One angle of the trapezoid is 60 degrees -/
  has_60_degree_angle : Bool
  /-- The length of the longer base of the trapezoid -/
  longer_base : ℝ
  /-- The length of the shorter base of the trapezoid -/
  shorter_base : ℝ

/-- The ratio of the bases of a special trapezoid is 3:1 -/
theorem special_trapezoid_base_ratio (t : SpecialTrapezoid) :
  t.has_inscribed_circle ∧ t.has_circumscribed_circle ∧ t.has_60_degree_angle →
  t.longer_base / t.shorter_base = 3 := by
  sorry

end NUMINAMATH_CALUDE_special_trapezoid_base_ratio_l1808_180839


namespace NUMINAMATH_CALUDE_average_age_of_contestants_l1808_180881

/-- Represents an age in years and months -/
structure Age where
  years : ℕ
  months : ℕ
  valid : months < 12

/-- Converts an age to total months -/
def ageToMonths (a : Age) : ℕ := a.years * 12 + a.months

/-- Converts total months to an age -/
def monthsToAge (m : ℕ) : Age :=
  { years := m / 12
  , months := m % 12
  , valid := by exact Nat.mod_lt m (by norm_num) }

/-- Calculates the average age of three contestants -/
def averageAge (a1 a2 a3 : Age) : Age :=
  monthsToAge ((ageToMonths a1 + ageToMonths a2 + ageToMonths a3) / 3)

theorem average_age_of_contestants :
  let age1 : Age := { years := 15, months := 9, valid := by norm_num }
  let age2 : Age := { years := 16, months := 1, valid := by norm_num }
  let age3 : Age := { years := 15, months := 8, valid := by norm_num }
  let avgAge := averageAge age1 age2 age3
  avgAge.years = 15 ∧ avgAge.months = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_age_of_contestants_l1808_180881


namespace NUMINAMATH_CALUDE_equation_root_in_interval_l1808_180862

theorem equation_root_in_interval : ∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x = 2 - x := by sorry

end NUMINAMATH_CALUDE_equation_root_in_interval_l1808_180862


namespace NUMINAMATH_CALUDE_binary_addition_theorem_l1808_180897

/-- Represents a binary number as a list of bits (0 or 1) in little-endian order -/
def BinaryNumber := List Bool

/-- Converts a decimal number to its binary representation -/
def decimalToBinary (n : Int) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : Int :=
  sorry

/-- Adds two binary numbers -/
def addBinary (a b : BinaryNumber) : BinaryNumber :=
  sorry

/-- Negates a binary number (two's complement) -/
def negateBinary (b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_addition_theorem :
  let b1 := decimalToBinary 13  -- 1101₂
  let b2 := decimalToBinary 10  -- 1010₂
  let b3 := decimalToBinary 7   -- 111₂
  let b4 := negateBinary (decimalToBinary 11)  -- -1011₂
  let sum := addBinary b1 (addBinary b2 (addBinary b3 b4))
  binaryToDecimal sum = 35  -- 100011₂
  := by sorry

end NUMINAMATH_CALUDE_binary_addition_theorem_l1808_180897


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l1808_180895

theorem magical_red_knights_fraction (total : ℚ) (red : ℚ) (blue : ℚ) (magical : ℚ) 
  (h1 : red = 3 / 8 * total)
  (h2 : blue = total - red)
  (h3 : magical = 1 / 4 * total)
  (h4 : ∃ (x y : ℚ), x / y > 0 ∧ red * (x / y) = 3 * (blue * (x / (3 * y))) ∧ red * (x / y) + blue * (x / (3 * y)) = magical) :
  ∃ (x y : ℚ), x / y = 3 / 7 ∧ red * (x / y) = magical := by
  sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l1808_180895


namespace NUMINAMATH_CALUDE_tims_earnings_per_visit_l1808_180840

theorem tims_earnings_per_visit
  (visitors_per_day : ℕ)
  (regular_days : ℕ)
  (total_earnings : ℚ)
  (h1 : visitors_per_day = 100)
  (h2 : regular_days = 6)
  (h3 : total_earnings = 18)
  : 
  let total_visitors := visitors_per_day * regular_days + 2 * (visitors_per_day * regular_days)
  total_earnings / total_visitors = 1 / 100 := by
  sorry

end NUMINAMATH_CALUDE_tims_earnings_per_visit_l1808_180840


namespace NUMINAMATH_CALUDE_farm_horses_cows_l1808_180815

theorem farm_horses_cows (initial_horses : ℕ) (initial_cows : ℕ) : 
  (initial_horses = 4 * initial_cows) →
  ((initial_horses - 15) / (initial_cows + 15) = 7 / 3) →
  (initial_horses - 15) - (initial_cows + 15) = 60 := by
  sorry

end NUMINAMATH_CALUDE_farm_horses_cows_l1808_180815


namespace NUMINAMATH_CALUDE_exchange_indifference_l1808_180886

/-- Represents the number of rubles a tourist plans to exchange. -/
def rubles : ℕ := 140

/-- Represents the exchange rate (in tugriks) for the first office. -/
def rate1 : ℕ := 3000

/-- Represents the exchange rate (in tugriks) for the second office. -/
def rate2 : ℕ := 2950

/-- Represents the commission fee (in tugriks) for the first office. -/
def commission : ℕ := 7000

theorem exchange_indifference :
  rate1 * rubles - commission = rate2 * rubles :=
by sorry

end NUMINAMATH_CALUDE_exchange_indifference_l1808_180886


namespace NUMINAMATH_CALUDE_complement_union_equals_singleton_l1808_180873

def M : Set (ℝ × ℝ) := {p | p.2 ≠ p.1 + 2}

def N : Set (ℝ × ℝ) := {p | p.2 ≠ -p.1}

def I : Set (ℝ × ℝ) := Set.univ

theorem complement_union_equals_singleton : 
  (I \ (M ∪ N)) = {(-1, 1)} := by sorry

end NUMINAMATH_CALUDE_complement_union_equals_singleton_l1808_180873


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1808_180823

-- Define the constant k
def k : ℝ := 4^2 * (256 ^ (1/4))

-- State the theorem
theorem inverse_variation_problem (x y : ℝ) 
  (h1 : x^2 * y^(1/4) = k)  -- x² and ⁴√y are inversely proportional
  (h2 : x * y = 128)        -- xy = 128
  : y = 8 := by
  sorry

-- Note: The condition x = 4 when y = 256 is implicitly used in the definition of k

end NUMINAMATH_CALUDE_inverse_variation_problem_l1808_180823


namespace NUMINAMATH_CALUDE_expression_value_l1808_180811

theorem expression_value : 
  let x : ℝ := 4
  (x^2 - 2*x - 15) / (x - 5) = 7 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1808_180811


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1808_180857

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4*x > 0}
def B : Set ℝ := {x : ℝ | x > 1}

-- State the theorem
theorem complement_A_intersect_B :
  (Aᶜ ∩ B) = {x : ℝ | 1 < x ∧ x ≤ 4} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1808_180857


namespace NUMINAMATH_CALUDE_factorization_equality_l1808_180822

theorem factorization_equality (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1808_180822


namespace NUMINAMATH_CALUDE_perpendicular_lines_imply_a_eq_neg_three_l1808_180885

/-- Two lines are perpendicular if the sum of the products of their coefficients is zero -/
def are_perpendicular (a₁ b₁ a₂ b₂ : ℝ) : Prop :=
  a₁ * a₂ + b₁ * b₂ = 0

/-- The first line: ax + 3y + 1 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop :=
  a * x + 3 * y + 1 = 0

/-- The second line: 2x + (a+1)y + 1 = 0 -/
def line2 (a : ℝ) (x y : ℝ) : Prop :=
  2 * x + (a + 1) * y + 1 = 0

/-- Theorem: If the lines are perpendicular, then a = -3 -/
theorem perpendicular_lines_imply_a_eq_neg_three (a : ℝ) :
  are_perpendicular a 3 2 (a + 1) → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_imply_a_eq_neg_three_l1808_180885


namespace NUMINAMATH_CALUDE_weight_loss_problem_l1808_180882

theorem weight_loss_problem (total_loss weight_loss_2 weight_loss_3 weight_loss_4 : ℕ) 
  (h1 : total_loss = 103)
  (h2 : weight_loss_3 = 28)
  (h3 : weight_loss_4 = 28)
  (h4 : weight_loss_2 = weight_loss_3 + weight_loss_4 - 7) :
  ∃ (weight_loss_1 : ℕ), 
    weight_loss_1 + weight_loss_2 + weight_loss_3 + weight_loss_4 = total_loss ∧ 
    weight_loss_1 = 27 := by
  sorry

end NUMINAMATH_CALUDE_weight_loss_problem_l1808_180882


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l1808_180841

theorem subtraction_preserves_inequality (a b c : ℝ) : a > b → a - c > b - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l1808_180841


namespace NUMINAMATH_CALUDE_max_points_at_distance_l1808_180875

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in a 2D plane -/
def Point := ℝ × ℝ

/-- The distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Whether a point is outside a circle -/
def isOutside (p : Point) (c : Circle) : Prop := sorry

/-- The number of points on a circle that are at a given distance from a point -/
def numPointsAtDistance (c : Circle) (p : Point) (d : ℝ) : ℕ := sorry

theorem max_points_at_distance (C : Circle) (P : Point) :
  isOutside P C →
  (∃ (n : ℕ), numPointsAtDistance C P 5 = n ∧ 
    ∀ (m : ℕ), numPointsAtDistance C P 5 ≤ m → n ≤ m) →
  numPointsAtDistance C P 5 = 2 := by sorry

end NUMINAMATH_CALUDE_max_points_at_distance_l1808_180875


namespace NUMINAMATH_CALUDE_y_value_l1808_180817

/-- In an acute triangle, two altitudes divide the sides into segments of lengths 7, 3, 6, and y units. -/
structure AcuteTriangle where
  -- Define the segment lengths
  a : ℝ
  b : ℝ
  c : ℝ
  y : ℝ
  -- Conditions on the segment lengths
  ha : a = 7
  hb : b = 3
  hc : c = 6
  -- The triangle is acute (we don't use this directly, but it's part of the problem statement)
  acute : True

/-- The value of y in the acute triangle with given segment lengths is 7. -/
theorem y_value (t : AcuteTriangle) : t.y = 7 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1808_180817


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l1808_180802

theorem geometric_series_first_term (a r : ℝ) (h1 : a / (1 - r) = 20) (h2 : a^2 / (1 - r^2) = 80) : a = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l1808_180802


namespace NUMINAMATH_CALUDE_x_intercept_of_line_l1808_180828

/-- The x-intercept of the line 4x + 7y = 28 is (7, 0) -/
theorem x_intercept_of_line (x y : ℝ) :
  4 * x + 7 * y = 28 → y = 0 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_intercept_of_line_l1808_180828


namespace NUMINAMATH_CALUDE_correct_multiplication_l1808_180831

theorem correct_multiplication (x : ℚ) : 14 * x = 42 → 12 * x = 36 := by
  sorry

end NUMINAMATH_CALUDE_correct_multiplication_l1808_180831


namespace NUMINAMATH_CALUDE_train_speed_calculation_l1808_180879

/-- Calculates the speed of a train crossing a platform -/
theorem train_speed_calculation (train_length platform_length : Real) 
  (crossing_time : Real) (h1 : train_length = 240) 
  (h2 : platform_length = 240) (h3 : crossing_time = 27) : 
  ∃ (speed : Real), abs (speed - 64) < 0.01 := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_train_speed_calculation_l1808_180879


namespace NUMINAMATH_CALUDE_bill_amount_calculation_l1808_180889

/-- Calculates the face value of a bill given the true discount, interest rate, and time to maturity. -/
def faceBill (trueDiscount : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  trueDiscount * (1 + rate * time)

/-- Theorem: Given a true discount of 150 on a bill due in 9 months at 16% per annum, the amount of the bill is 168. -/
theorem bill_amount_calculation : 
  let trueDiscount : ℝ := 150
  let rate : ℝ := 0.16  -- 16% per annum
  let time : ℝ := 0.75  -- 9 months = 9/12 years = 0.75 years
  faceBill trueDiscount rate time = 168 := by
  sorry


end NUMINAMATH_CALUDE_bill_amount_calculation_l1808_180889


namespace NUMINAMATH_CALUDE_p_20_equals_neg_8_l1808_180832

/-- A quadratic function with specific properties -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The theorem stating that p(20) = -8 given the conditions -/
theorem p_20_equals_neg_8 (a b c : ℝ) :
  (∀ x, p a b c (19 - x) = p a b c x) →  -- Axis of symmetry at x = 9.5
  p a b c 0 = -8 →                       -- p(0) = -8
  p a b c 20 = -8 := by
  sorry

end NUMINAMATH_CALUDE_p_20_equals_neg_8_l1808_180832


namespace NUMINAMATH_CALUDE_lemonade_price_calculation_l1808_180853

theorem lemonade_price_calculation (glasses_per_gallon : ℕ) (cost_per_gallon : ℚ) 
  (gallons_made : ℕ) (glasses_drunk : ℕ) (glasses_unsold : ℕ) (net_profit : ℚ) :
  glasses_per_gallon = 16 →
  cost_per_gallon = 7/2 →
  gallons_made = 2 →
  glasses_drunk = 5 →
  glasses_unsold = 6 →
  net_profit = 14 →
  (gallons_made * cost_per_gallon + net_profit) / 
    (gallons_made * glasses_per_gallon - glasses_drunk - glasses_unsold) = 1 := by
  sorry

#eval (2 * (7/2 : ℚ) + 14) / (2 * 16 - 5 - 6)

end NUMINAMATH_CALUDE_lemonade_price_calculation_l1808_180853


namespace NUMINAMATH_CALUDE_time_after_1750_minutes_l1808_180808

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  sorry

/-- Converts a number of minutes to hours and minutes -/
def minutesToTime (m : Nat) : Time :=
  sorry

theorem time_after_1750_minutes :
  let start_time : Time := ⟨8, 0, by sorry, by sorry⟩
  let added_time : Time := minutesToTime 1750
  let end_time : Time := addMinutes start_time 1750
  end_time = ⟨13, 10, by sorry, by sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_time_after_1750_minutes_l1808_180808


namespace NUMINAMATH_CALUDE_monthly_income_p_l1808_180866

/-- Given the average monthly incomes of pairs of individuals, prove that the monthly income of p is 4000. -/
theorem monthly_income_p (p q r : ℕ) : 
  (p + q) / 2 = 5050 →
  (q + r) / 2 = 6250 →
  (p + r) / 2 = 5200 →
  p = 4000 := by
sorry

end NUMINAMATH_CALUDE_monthly_income_p_l1808_180866


namespace NUMINAMATH_CALUDE_english_test_percentage_l1808_180896

theorem english_test_percentage (math_questions : ℕ) (english_questions : ℕ) 
  (math_percentage : ℚ) (total_correct : ℕ) : 
  math_questions = 40 →
  english_questions = 50 →
  math_percentage = 3/4 →
  total_correct = 79 →
  (total_correct - (math_percentage * math_questions).num) / english_questions = 49/50 := by
sorry

end NUMINAMATH_CALUDE_english_test_percentage_l1808_180896


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_increasing_implies_a_nonnegative_l1808_180865

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x + 8

-- Part 1: Extreme value at x = 3 implies a = 3
theorem extreme_value_implies_a (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3 - ε) (3 + ε), f a x ≤ f a 3) →
  a = 3 :=
sorry

-- Part 2: Increasing on (-∞, 0) implies a ∈ [0, +∞)
theorem increasing_implies_a_nonnegative (a : ℝ) :
  (∀ x y : ℝ, x < y ∧ y < 0 → f a x < f a y) →
  a ≥ 0 :=
sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_increasing_implies_a_nonnegative_l1808_180865


namespace NUMINAMATH_CALUDE_sum_of_integers_between_2_and_15_l1808_180804

theorem sum_of_integers_between_2_and_15 : 
  (Finset.range 12).sum (fun i => i + 3) = 102 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_between_2_and_15_l1808_180804


namespace NUMINAMATH_CALUDE_median_sum_lower_bound_l1808_180855

/-- Given a triangle ABC with sides a, b, c and medians ma, mb, mc, 
    the sum of the lengths of the medians is at least three quarters of its perimeter. -/
theorem median_sum_lower_bound (a b c ma mb mc : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_pos_ma : ma > 0) (h_pos_mb : mb > 0) (h_pos_mc : mc > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_ma : ma^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_mb : mb^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_mc : mc^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  ma + mb + mc ≥ 3/4 * (a + b + c) := by
  sorry

end NUMINAMATH_CALUDE_median_sum_lower_bound_l1808_180855


namespace NUMINAMATH_CALUDE_quadratic_function_and_area_bisection_l1808_180863

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  equal_roots : ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r)
  derivative : ∀ x, HasDerivAt f (2 * x + 2) x

/-- The main theorem about the quadratic function and area bisection -/
theorem quadratic_function_and_area_bisection (qf : QuadraticFunction) :
  (∀ x, qf.f x = x^2 + 2*x + 1) ∧
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
    (∫ x in (-1)..(-t), qf.f x) = (∫ x in (-t)..0, qf.f x) ∧
    t = 1 - 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_and_area_bisection_l1808_180863


namespace NUMINAMATH_CALUDE_first_discount_is_twenty_percent_l1808_180835

def initial_price : ℝ := 12000
def final_price : ℝ := 7752
def second_discount : ℝ := 0.15
def third_discount : ℝ := 0.05

def first_discount_percentage (x : ℝ) : Prop :=
  final_price = initial_price * (1 - x / 100) * (1 - second_discount) * (1 - third_discount)

theorem first_discount_is_twenty_percent :
  first_discount_percentage 20 := by
  sorry

end NUMINAMATH_CALUDE_first_discount_is_twenty_percent_l1808_180835


namespace NUMINAMATH_CALUDE_sqrt_nested_roots_l1808_180888

theorem sqrt_nested_roots (N : ℝ) (h : N > 1) : 
  Real.sqrt (N * Real.sqrt (N * Real.sqrt N)) = N^(7/8) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nested_roots_l1808_180888


namespace NUMINAMATH_CALUDE_absolute_value_sum_inequality_l1808_180803

theorem absolute_value_sum_inequality (x : ℝ) :
  |x - 1| + |x - 2| > 5 ↔ x < -1 ∨ x > 4 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_inequality_l1808_180803


namespace NUMINAMATH_CALUDE_cos_alpha_value_l1808_180854

theorem cos_alpha_value (α : ℝ) (h : Real.sin (π / 2 + α) = 3 / 5) : 
  Real.cos α = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l1808_180854


namespace NUMINAMATH_CALUDE_power_difference_equals_one_l1808_180825

theorem power_difference_equals_one (x y : ℕ) : 
  (2^x ∣ 180) ∧ 
  (3^y ∣ 180) ∧ 
  (∀ z : ℕ, z > x → ¬(2^z ∣ 180)) ∧ 
  (∀ w : ℕ, w > y → ¬(3^w ∣ 180)) → 
  (1/3 : ℚ)^(y - x) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_difference_equals_one_l1808_180825


namespace NUMINAMATH_CALUDE_max_a_for_increasing_f_l1808_180872

def f (x : ℝ) : ℝ := -x^2 + 2*x - 2

theorem max_a_for_increasing_f :
  (∃ a : ℝ, ∀ x₁ x₂ : ℝ, x₁ ≤ x₂ ∧ x₂ ≤ a → f x₁ ≤ f x₂) ∧
  (∀ b : ℝ, b > 1 → ∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ b ∧ f x₁ > f x₂) :=
by sorry

end NUMINAMATH_CALUDE_max_a_for_increasing_f_l1808_180872


namespace NUMINAMATH_CALUDE_intersection_and_parallel_line_l1808_180870

-- Define the two lines
def l₁ (x y : ℝ) : Prop := x + 8*y + 7 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y - 1 = 0

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := x + y + 1 = 0

theorem intersection_and_parallel_line :
  -- Part 1: Prove that (1, -1) is the intersection point
  (l₁ 1 (-1) ∧ l₂ 1 (-1)) ∧
  -- Part 2: Prove that x + y = 0 is the equation of the line passing through
  -- the intersection point and parallel to x + y + 1 = 0
  (∃ c : ℝ, ∀ x y : ℝ, (l₁ x y ∧ l₂ x y) → x + y = c) ∧
  (∀ x y : ℝ, (x + y = 0) ↔ (∃ k : ℝ, x = 1 + k ∧ y = -1 - k ∧ parallel_line (1 + k) (-1 - k))) :=
sorry

end NUMINAMATH_CALUDE_intersection_and_parallel_line_l1808_180870


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_18_l1808_180842

theorem four_digit_divisible_by_18 : 
  ∀ n : ℕ, n < 10 → (4150 + n) % 18 = 0 ↔ n = 8 := by sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_18_l1808_180842


namespace NUMINAMATH_CALUDE_square_circle_equal_area_l1808_180861

theorem square_circle_equal_area (r : ℝ) (s : ℝ) : 
  r = 5 →
  s = 2 * r →
  s^2 = π * r^2 →
  s = 5 * Real.sqrt π :=
by sorry

end NUMINAMATH_CALUDE_square_circle_equal_area_l1808_180861


namespace NUMINAMATH_CALUDE_sum_of_squares_inequality_l1808_180887

theorem sum_of_squares_inequality (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (2 + a) * (2 + b) ≥ c * d := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_inequality_l1808_180887


namespace NUMINAMATH_CALUDE_day_of_week_in_consecutive_years_l1808_180814

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  number : ℕ
  is_leap : Bool

/-- Returns the day of the week for a given day number in a year -/
def day_of_week (y : Year) (day_number : ℕ) : DayOfWeek :=
  sorry

/-- Returns the next year -/
def next_year (y : Year) : Year :=
  sorry

/-- Returns the previous year -/
def prev_year (y : Year) : Year :=
  sorry

theorem day_of_week_in_consecutive_years 
  (y : Year)
  (h1 : day_of_week y 250 = DayOfWeek.Friday)
  (h2 : day_of_week (next_year y) 150 = DayOfWeek.Friday) :
  day_of_week (prev_year y) 50 = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_day_of_week_in_consecutive_years_l1808_180814


namespace NUMINAMATH_CALUDE_valid_divisions_count_l1808_180806

/-- Represents a rectangle on the grid -/
structure Rectangle where
  x : Nat
  y : Nat
  width : Nat
  height : Nat

/-- Represents a division of the grid into 5 rectangles -/
structure GridDivision where
  center : Rectangle
  top : Rectangle
  bottom : Rectangle
  left : Rectangle
  right : Rectangle

/-- Checks if a rectangle touches the perimeter of an 11x11 grid -/
def touchesPerimeter (r : Rectangle) : Bool :=
  r.x = 0 || r.y = 0 || r.x + r.width = 11 || r.y + r.height = 11

/-- Checks if a grid division is valid -/
def isValidDivision (d : GridDivision) : Bool :=
  d.center.x > 0 && d.center.y > 0 && 
  d.center.x + d.center.width < 11 && 
  d.center.y + d.center.height < 11 &&
  touchesPerimeter d.top &&
  touchesPerimeter d.bottom &&
  touchesPerimeter d.left &&
  touchesPerimeter d.right

/-- Counts the number of valid grid divisions -/
def countValidDivisions : Nat :=
  sorry

theorem valid_divisions_count : countValidDivisions = 81 := by
  sorry

end NUMINAMATH_CALUDE_valid_divisions_count_l1808_180806


namespace NUMINAMATH_CALUDE_sum_of_special_integers_l1808_180847

theorem sum_of_special_integers (n : ℕ) : 
  (∃ (S : Finset ℕ), 
    (∀ m ∈ S, m < 100 ∧ m > 0 ∧ ∃ k : ℤ, 5 * m^2 + 3 * m - 5 = 15 * k) ∧ 
    (∀ m : ℕ, m < 100 ∧ m > 0 ∧ (∃ k : ℤ, 5 * m^2 + 3 * m - 5 = 15 * k) → m ∈ S) ∧
    (Finset.sum S id = 635)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_special_integers_l1808_180847


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l1808_180867

/-- Given two points M(-3, y₁) and N(2, y₂) on the line y = -3x + 1, prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = -3 * (-3) + 1) → (y₂ = -3 * 2 + 1) → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l1808_180867


namespace NUMINAMATH_CALUDE_direction_cannot_determine_position_l1808_180812

-- Define a type for positions
structure Position where
  x : ℝ
  y : ℝ

-- Define a type for directions
structure Direction where
  angle : ℝ

-- Define a function to check if a piece of data can determine a position
def canDeterminePosition (data : Type) : Prop :=
  ∃ (f : data → Position), Function.Injective f

-- Theorem statement
theorem direction_cannot_determine_position :
  ¬ (canDeterminePosition Direction) :=
sorry

end NUMINAMATH_CALUDE_direction_cannot_determine_position_l1808_180812


namespace NUMINAMATH_CALUDE_cost_of_45_lilies_l1808_180860

/-- The cost of a bouquet of lilies at Lila's Lily Shop -/
def bouquet_cost (n : ℕ) : ℚ :=
  let base_price := 2 * n  -- $2 per lily
  if n ≤ 30 then base_price else base_price * (1 - 1/10)  -- 10% discount if > 30 lilies

/-- Theorem: The cost of a bouquet with 45 lilies is $81 -/
theorem cost_of_45_lilies :
  bouquet_cost 45 = 81 := by sorry

end NUMINAMATH_CALUDE_cost_of_45_lilies_l1808_180860


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l1808_180826

/-- The number of participants in the chess tournament. -/
def n : ℕ := 19

/-- The number of matches played in a round-robin tournament. -/
def matches_played (x : ℕ) : ℕ := x * (x - 1) / 2

/-- The number of matches played after three players dropped out. -/
def matches_after_dropout (x : ℕ) : ℕ := (x - 3) * (x - 4) / 2

/-- Theorem stating that the number of participants in the chess tournament is 19. -/
theorem chess_tournament_participants :
  matches_played n - matches_after_dropout n = 130 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l1808_180826


namespace NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l1808_180843

/-- Proves that the ratio of a man's age to his son's age in two years is 2:1,
    given that the man is 37 years older than his son and the son's current age is 35. -/
theorem mans_age_to_sons_age_ratio :
  let sons_current_age : ℕ := 35
  let mans_current_age : ℕ := sons_current_age + 37
  let sons_age_in_two_years : ℕ := sons_current_age + 2
  let mans_age_in_two_years : ℕ := mans_current_age + 2
  (mans_age_in_two_years : ℚ) / (sons_age_in_two_years : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_to_sons_age_ratio_l1808_180843


namespace NUMINAMATH_CALUDE_popsicle_melting_rate_l1808_180878

/-- Given a sequence of 6 terms where each term is twice the previous term and the first term is 1,
    prove that the last term is equal to 32. -/
theorem popsicle_melting_rate (seq : Fin 6 → ℕ) 
    (h1 : seq 0 = 1)
    (h2 : ∀ i : Fin 5, seq (i.succ) = 2 * seq i) : 
  seq 5 = 32 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_melting_rate_l1808_180878


namespace NUMINAMATH_CALUDE_earth_sun_max_distance_l1808_180845

/-- The semi-major axis of Earth's orbit in kilometers -/
def semi_major_axis : ℝ := 1.5e8

/-- The semi-minor axis of Earth's orbit in kilometers -/
def semi_minor_axis : ℝ := 3e6

/-- The maximum distance from Earth to Sun in kilometers -/
def max_distance : ℝ := semi_major_axis + semi_minor_axis

theorem earth_sun_max_distance :
  max_distance = 1.53e8 := by sorry

end NUMINAMATH_CALUDE_earth_sun_max_distance_l1808_180845


namespace NUMINAMATH_CALUDE_no_real_roots_of_quartic_equation_l1808_180899

theorem no_real_roots_of_quartic_equation :
  ∀ x : ℝ, 5 * x^4 - 28 * x^3 + 57 * x^2 - 28 * x + 5 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_real_roots_of_quartic_equation_l1808_180899


namespace NUMINAMATH_CALUDE_quadrilateral_angle_equality_l1808_180800

-- Define the points
variable (A B C D E F P : Point)

-- Define the quadrilateral
def is_quadrilateral (A B C D : Point) : Prop := sorry

-- Define that a point is on a line segment
def on_segment (X Y Z : Point) : Prop := sorry

-- Define that two lines intersect at a point
def lines_intersect_at (W X Y Z P : Point) : Prop := sorry

-- Define angle equality
def angle_eq (A B C D E F : Point) : Prop := sorry

-- State the theorem
theorem quadrilateral_angle_equality 
  (h1 : is_quadrilateral A B C D)
  (h2 : on_segment B E C)
  (h3 : on_segment C F D)
  (h4 : lines_intersect_at B F D E P)
  (h5 : angle_eq B A E F A D) :
  angle_eq B A P C A D := by sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_equality_l1808_180800


namespace NUMINAMATH_CALUDE_parallelogram_height_l1808_180864

/-- Theorem: Height of a parallelogram with given area and base -/
theorem parallelogram_height (area base height : ℝ) : 
  area = 448 ∧ base = 32 ∧ area = base * height → height = 14 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1808_180864


namespace NUMINAMATH_CALUDE_power_of_three_in_product_l1808_180813

theorem power_of_three_in_product (w : ℕ+) : 
  (∃ k : ℕ, 936 * w = 2^5 * 11^2 * k) → 
  (132 ≤ w) →
  (∃ m : ℕ, 936 * w = 3^3 * m ∧ ∀ n > 3, ¬(∃ l : ℕ, 936 * w = 3^n * l)) := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_in_product_l1808_180813


namespace NUMINAMATH_CALUDE_periodic_sequence_properties_l1808_180877

/-- A periodic sequence with period T -/
def is_periodic (a : ℕ → ℕ) (T : ℕ) : Prop :=
  ∀ n, a (n + T) = a n

/-- The smallest period of a sequence -/
def smallest_period (a : ℕ → ℕ) (t : ℕ) : Prop :=
  is_periodic a t ∧ ∀ s, is_periodic a s → t ≤ s

theorem periodic_sequence_properties {a : ℕ → ℕ} {T : ℕ} (h : is_periodic a T) :
  (∃ t, smallest_period a t) ∧ (∀ t, smallest_period a t → T % t = 0) := by
  sorry

end NUMINAMATH_CALUDE_periodic_sequence_properties_l1808_180877


namespace NUMINAMATH_CALUDE_negation_quadratic_inequality_l1808_180848

theorem negation_quadratic_inequality (x : ℝ) : 
  ¬(x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_quadratic_inequality_l1808_180848


namespace NUMINAMATH_CALUDE_sum_square_values_l1808_180838

theorem sum_square_values (K M : ℕ) : 
  K * (K + 1) / 2 = M^2 →
  M < 200 →
  K > M →
  (K = 8 ∨ K = 49) ∧ 
  (∀ n : ℕ, n * (n + 1) / 2 = M^2 ∧ M < 200 ∧ n > M → n = 8 ∨ n = 49) :=
by sorry

end NUMINAMATH_CALUDE_sum_square_values_l1808_180838


namespace NUMINAMATH_CALUDE_oil_bill_problem_l1808_180868

/-- The oil bill problem -/
theorem oil_bill_problem (jan_bill feb_bill : ℝ) 
  (h1 : feb_bill / jan_bill = 5 / 4)
  (h2 : (feb_bill + 45) / jan_bill = 3 / 2) :
  jan_bill = 180 := by
  sorry

end NUMINAMATH_CALUDE_oil_bill_problem_l1808_180868


namespace NUMINAMATH_CALUDE_max_cos_a_value_l1808_180821

theorem max_cos_a_value (a b c : Real) 
  (h1 : Real.sin a = Real.cos b)
  (h2 : Real.sin b = Real.cos c)
  (h3 : Real.sin c = Real.cos a) :
  ∃ (max_cos_a : Real), max_cos_a = Real.sqrt 2 / 2 ∧ 
    ∀ x, Real.cos a ≤ x → x ≤ max_cos_a :=
by sorry

end NUMINAMATH_CALUDE_max_cos_a_value_l1808_180821


namespace NUMINAMATH_CALUDE_distance_difference_after_six_hours_l1808_180852

/-- Represents a cyclist with a given travel rate in miles per hour. -/
structure Cyclist where
  name : String
  rate : ℝ

/-- Calculates the distance traveled by a cyclist in a given time. -/
def distance_traveled (c : Cyclist) (time : ℝ) : ℝ :=
  c.rate * time

/-- The time period in hours for which we calculate the travel distance. -/
def travel_time : ℝ := 6

/-- Carmen, a cyclist with a travel rate of 15 miles per hour. -/
def carmen : Cyclist :=
  { name := "Carmen", rate := 15 }

/-- Daniel, a cyclist with a travel rate of 12.5 miles per hour. -/
def daniel : Cyclist :=
  { name := "Daniel", rate := 12.5 }

/-- Theorem stating the difference in distance traveled between Carmen and Daniel after 6 hours. -/
theorem distance_difference_after_six_hours :
    distance_traveled carmen travel_time - distance_traveled daniel travel_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_after_six_hours_l1808_180852


namespace NUMINAMATH_CALUDE_first_number_proof_l1808_180829

theorem first_number_proof (x y : ℝ) : 
  x + y = 10 → 2 * x = 3 * y + 5 → x = 7 := by sorry

end NUMINAMATH_CALUDE_first_number_proof_l1808_180829


namespace NUMINAMATH_CALUDE_roots_sum_powers_l1808_180833

theorem roots_sum_powers (γ δ : ℝ) : 
  γ^2 - 5*γ + 6 = 0 → δ^2 - 5*δ + 6 = 0 → 8*γ^5 + 15*δ^4 = 8425 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_powers_l1808_180833


namespace NUMINAMATH_CALUDE_basil_cookie_boxes_l1808_180874

/-- The number of cookies Basil gets in the morning and before bed -/
def morning_night_cookies : ℚ := 1/2 + 1/2

/-- The number of whole cookies Basil gets during the day -/
def day_cookies : ℕ := 2

/-- The number of cookies per box -/
def cookies_per_box : ℕ := 45

/-- The number of days Basil needs cookies for -/
def days : ℕ := 30

/-- Theorem stating the number of boxes Basil needs for 30 days -/
theorem basil_cookie_boxes : 
  ⌈(days * (morning_night_cookies + day_cookies)) / cookies_per_box⌉ = 2 := by
  sorry

end NUMINAMATH_CALUDE_basil_cookie_boxes_l1808_180874


namespace NUMINAMATH_CALUDE_five_digit_with_four_or_five_l1808_180830

/-- The number of five-digit positive integers -/
def total_five_digit : ℕ := 90000

/-- The number of digits that are not 4 or 5 -/
def non_four_five_digits : ℕ := 8

/-- The number of options for the first digit (excluding 0, 4, and 5) -/
def first_digit_options : ℕ := 7

/-- The number of five-digit positive integers without 4 or 5 -/
def without_four_five : ℕ := first_digit_options * non_four_five_digits^4

theorem five_digit_with_four_or_five :
  total_five_digit - without_four_five = 61328 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_with_four_or_five_l1808_180830


namespace NUMINAMATH_CALUDE_two_std_dev_less_than_mean_example_l1808_180850

/-- For a normal distribution with given mean and standard deviation,
    calculate the value that is exactly 2 standard deviations less than the mean -/
def twoStdDevLessThanMean (mean : ℝ) (stdDev : ℝ) : ℝ :=
  mean - 2 * stdDev

/-- Theorem stating that for a normal distribution with mean 12 and standard deviation 1.2,
    the value exactly 2 standard deviations less than the mean is 9.6 -/
theorem two_std_dev_less_than_mean_example :
  twoStdDevLessThanMean 12 1.2 = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_less_than_mean_example_l1808_180850


namespace NUMINAMATH_CALUDE_imaginary_unit_sum_zero_l1808_180801

theorem imaginary_unit_sum_zero (i : ℂ) (hi : i^2 = -1) : i + i^2 + i^3 + i^4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_sum_zero_l1808_180801


namespace NUMINAMATH_CALUDE_equation_solution_l1808_180824

theorem equation_solution : 
  ∀ x : ℝ, (x + 1) * (x + 3) = x + 1 ↔ x = -1 ∨ x = -2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1808_180824


namespace NUMINAMATH_CALUDE_arithmetic_problem_l1808_180883

theorem arithmetic_problem : 
  ((2 * 4 * 6) / (1 + 3 + 5 + 7) - (1 * 3 * 5) / (2 + 4 + 6)) / (1/2) = 3.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_problem_l1808_180883


namespace NUMINAMATH_CALUDE_remainder_8_900_mod_29_l1808_180869

theorem remainder_8_900_mod_29 : (8 : Nat)^900 % 29 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8_900_mod_29_l1808_180869


namespace NUMINAMATH_CALUDE_solution_eq_200_div_253_l1808_180851

/-- A binary operation on nonzero real numbers satisfying certain properties -/
def diamond (a b : ℝ) : ℝ := sorry

/-- The binary operation satisfies a ◇ (b ◇ c) = (a ◇ b) · c -/
axiom diamond_assoc (a b c : ℝ) : a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = (diamond a b) * c

/-- The binary operation satisfies a ◇ a = 1 -/
axiom diamond_self (a : ℝ) : a ≠ 0 → diamond a a = 1

/-- The solution to the equation 2024 ◇ (8 ◇ x) = 200 is 200/253 -/
theorem solution_eq_200_div_253 : ∃ (x : ℝ), x ≠ 0 ∧ diamond 2024 (diamond 8 x) = 200 ∧ x = 200/253 := by sorry

end NUMINAMATH_CALUDE_solution_eq_200_div_253_l1808_180851


namespace NUMINAMATH_CALUDE_soccer_field_kids_l1808_180859

theorem soccer_field_kids (initial_kids joining_kids : ℕ) : 
  initial_kids = 14 → joining_kids = 22 → initial_kids + joining_kids = 36 := by
sorry

end NUMINAMATH_CALUDE_soccer_field_kids_l1808_180859


namespace NUMINAMATH_CALUDE_smallest_candy_number_l1808_180818

theorem smallest_candy_number : ∃ (n : ℕ), 
  100 ≤ n ∧ n < 1000 ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 9) % 7 = 0 ∧
  ∀ m, 100 ≤ m ∧ m < n → ¬((m + 7) % 9 = 0 ∧ (m - 9) % 7 = 0) :=
by
  use 110
  sorry

end NUMINAMATH_CALUDE_smallest_candy_number_l1808_180818


namespace NUMINAMATH_CALUDE_square_of_binomial_l1808_180836

theorem square_of_binomial (x : ℝ) : ∃ (a : ℝ), x^2 - 20*x + 100 = (x - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l1808_180836


namespace NUMINAMATH_CALUDE_cricketer_average_after_22nd_inning_l1808_180805

/-- Represents the average score of a cricketer before the 22nd inning -/
def initial_average : ℝ := sorry

/-- The score made by the cricketer in the 22nd inning -/
def score_22nd_inning : ℝ := 134

/-- The increase in average after the 22nd inning -/
def average_increase : ℝ := 3.5

/-- The number of innings played before the 22nd inning -/
def previous_innings : ℕ := 21

/-- The total number of innings including the 22nd inning -/
def total_innings : ℕ := 22

/-- Calculates the new average after the 22nd inning -/
def new_average : ℝ := initial_average + average_increase

/-- Theorem stating that the new average after the 22nd inning is 60.5 -/
theorem cricketer_average_after_22nd_inning : 
  (previous_innings : ℝ) * initial_average + score_22nd_inning = 
    new_average * (total_innings : ℝ) ∧ new_average = 60.5 := by sorry

end NUMINAMATH_CALUDE_cricketer_average_after_22nd_inning_l1808_180805


namespace NUMINAMATH_CALUDE_short_trees_planted_l1808_180892

theorem short_trees_planted (initial_short : ℕ) (final_short : ℕ) :
  initial_short = 31 →
  final_short = 95 →
  final_short - initial_short = 64 := by
sorry

end NUMINAMATH_CALUDE_short_trees_planted_l1808_180892


namespace NUMINAMATH_CALUDE_scorpion_daily_segments_l1808_180894

/-- The number of body segments a cave scorpion needs to eat daily -/
def daily_segments : ℕ :=
  let segments_first_millipede := 60
  let segments_long_millipede := 2 * segments_first_millipede
  let segments_eaten := segments_first_millipede + 2 * segments_long_millipede
  let segments_to_eat := 10 * 50
  segments_eaten + segments_to_eat

theorem scorpion_daily_segments : daily_segments = 800 := by
  sorry

end NUMINAMATH_CALUDE_scorpion_daily_segments_l1808_180894


namespace NUMINAMATH_CALUDE_slower_train_speed_l1808_180890

/-- Proves that the speed of the slower train is 36 km/hr given the specified conditions -/
theorem slower_train_speed 
  (train_length : ℝ) 
  (faster_train_speed : ℝ) 
  (passing_time : ℝ) 
  (h1 : train_length = 25) 
  (h2 : faster_train_speed = 46) 
  (h3 : passing_time = 18) : 
  ∃ (slower_train_speed : ℝ), 
    slower_train_speed = 36 ∧ 
    (faster_train_speed - slower_train_speed) * (5 / 18) * passing_time = 2 * train_length :=
by sorry

end NUMINAMATH_CALUDE_slower_train_speed_l1808_180890


namespace NUMINAMATH_CALUDE_expression_value_l1808_180827

theorem expression_value : 50 * (50 - 5) - (50 * 50 - 5) = -245 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1808_180827


namespace NUMINAMATH_CALUDE_smallest_integer_in_special_average_l1808_180837

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem smallest_integer_in_special_average (m n : ℕ) 
  (h1 : is_two_digit m) 
  (h2 : is_three_digit n) 
  (h3 : (m + n) / 2 = m + n / 1000) : 
  min m n = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_in_special_average_l1808_180837


namespace NUMINAMATH_CALUDE_infinite_triples_exist_l1808_180810

/-- An infinite, strictly increasing sequence of positive integers -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The condition satisfied by the sequence -/
def SequenceCondition (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a (a n) ≤ a n + a (n + 3)

/-- The existence of infinitely many triples satisfying the condition -/
def InfinitelyManyTriples (a : ℕ → ℕ) : Prop :=
  ∀ N : ℕ, ∃ k l m : ℕ, k > N ∧ l > k ∧ m > l ∧ a k + a m = 2 * a l

/-- The main theorem -/
theorem infinite_triples_exist (a : ℕ → ℕ) 
  (h1 : StrictlyIncreasingSeq a) 
  (h2 : SequenceCondition a) : 
  InfinitelyManyTriples a :=
sorry

end NUMINAMATH_CALUDE_infinite_triples_exist_l1808_180810


namespace NUMINAMATH_CALUDE_odd_power_divisibility_l1808_180880

theorem odd_power_divisibility (a b : ℕ) (ha : Odd a) (hb : Odd b) (ha_pos : 0 < a) (hb_pos : 0 < b) :
  ∀ n : ℕ, 0 < n → ∃ m : ℕ, (2^n ∣ a^m * b^2 - 1) ∨ (2^n ∣ b^m * a^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_power_divisibility_l1808_180880


namespace NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l1808_180893

theorem sin_40_tan_10_minus_sqrt_3 : 
  Real.sin (40 * π / 180) * (Real.tan (10 * π / 180) - Real.sqrt 3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sin_40_tan_10_minus_sqrt_3_l1808_180893


namespace NUMINAMATH_CALUDE_equation_solution_l1808_180834

theorem equation_solution (a b : ℝ) (h : a ≠ 0) :
  let x : ℝ := (a^2 - b^2) / a
  x^2 + 4 * b^2 = (2 * a - x)^2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1808_180834


namespace NUMINAMATH_CALUDE_same_grade_probability_l1808_180876

/-- Represents the grades in the school -/
inductive Grade
| A
| B
| C

/-- Represents a student volunteer -/
structure Student where
  grade : Grade

/-- The total number of student volunteers -/
def total_students : Nat := 560

/-- The number of students in each grade -/
def students_per_grade (g : Grade) : Nat :=
  match g with
  | Grade.A => 240
  | Grade.B => 160
  | Grade.C => 160

/-- The number of students selected from each grade for the charity event -/
def selected_per_grade (g : Grade) : Nat :=
  match g with
  | Grade.A => 3
  | Grade.B => 2
  | Grade.C => 2

/-- The total number of students selected for the charity event -/
def total_selected : Nat := 7

/-- The number of students to be selected for sanitation work -/
def sanitation_workers : Nat := 2

/-- Theorem: The probability of selecting 2 students from the same grade for sanitation work is 5/21 -/
theorem same_grade_probability :
  (Nat.choose total_selected sanitation_workers) = 21 ∧
  (Nat.choose (selected_per_grade Grade.A) sanitation_workers +
   Nat.choose (selected_per_grade Grade.B) sanitation_workers +
   Nat.choose (selected_per_grade Grade.C) sanitation_workers) = 5 :=
by sorry


end NUMINAMATH_CALUDE_same_grade_probability_l1808_180876
