import Mathlib

namespace NUMINAMATH_CALUDE_prove_initial_number_l2365_236580

def initial_number : ℕ := 7899665
def result : ℕ := 7899593
def factor1 : ℕ := 12
def factor2 : ℕ := 3
def factor3 : ℕ := 2

theorem prove_initial_number :
  initial_number - (factor1 * factor2 * factor3) = result :=
by sorry

end NUMINAMATH_CALUDE_prove_initial_number_l2365_236580


namespace NUMINAMATH_CALUDE_area_to_paint_is_132_l2365_236599

/-- The area to be painted on a wall, given its dimensions and the dimensions of an area that doesn't need painting. -/
def areaToPaint (wallHeight wallLength paintingWidth paintingHeight : ℕ) : ℕ :=
  wallHeight * wallLength - paintingWidth * paintingHeight

/-- Theorem stating that the area to be painted is 132 square feet for the given dimensions. -/
theorem area_to_paint_is_132 :
  areaToPaint 10 15 3 6 = 132 := by
  sorry

end NUMINAMATH_CALUDE_area_to_paint_is_132_l2365_236599


namespace NUMINAMATH_CALUDE_f_sum_negative_l2365_236514

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h1 : ∀ x, f (4 - x) = -f x)
variable (h2 : ∀ x, x < 2 → StrictMonoDecreasing f (Set.Iio 2))

-- Define the conditions on x₁ and x₂
variable (x₁ x₂ : ℝ)
variable (h3 : x₁ + x₂ > 4)
variable (h4 : (x₁ - 2) * (x₂ - 2) < 0)

-- State the theorem
theorem f_sum_negative : f x₁ + f x₂ < 0 :=
sorry

end NUMINAMATH_CALUDE_f_sum_negative_l2365_236514


namespace NUMINAMATH_CALUDE_remi_bottle_capacity_l2365_236571

/-- The capacity of Remi's water bottle in ounces -/
def bottle_capacity : ℕ := 20

/-- The number of times Remi refills his bottle per day -/
def refills_per_day : ℕ := 3

/-- The number of days Remi drinks from his bottle -/
def days : ℕ := 7

/-- The amount of water Remi spills in ounces -/
def spilled_water : ℕ := 5 + 8

/-- The total amount of water Remi drinks in ounces -/
def total_water_drunk : ℕ := 407

/-- Theorem stating that given the conditions, Remi's water bottle capacity is 20 ounces -/
theorem remi_bottle_capacity :
  bottle_capacity * refills_per_day * days - spilled_water = total_water_drunk :=
by sorry

end NUMINAMATH_CALUDE_remi_bottle_capacity_l2365_236571


namespace NUMINAMATH_CALUDE_bob_corn_harvest_l2365_236527

/-- Calculates the number of bushels of corn harvested given the number of rows, 
    corn stalks per row, and corn stalks per bushel. -/
def corn_harvest (rows : ℕ) (stalks_per_row : ℕ) (stalks_per_bushel : ℕ) : ℕ :=
  (rows * stalks_per_row) / stalks_per_bushel

/-- Proves that Bob's corn harvest yields 50 bushels. -/
theorem bob_corn_harvest : 
  corn_harvest 5 80 8 = 50 := by
  sorry

end NUMINAMATH_CALUDE_bob_corn_harvest_l2365_236527


namespace NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l2365_236595

theorem largest_k_for_distinct_roots : 
  ∃ (k : ℤ), k = 8 ∧ 
  (∀ (x : ℝ), x^2 - 6*x + k = 0 → (∃ (y : ℝ), x ≠ y ∧ y^2 - 6*y + k = 0)) ∧
  (∀ (m : ℤ), m > k → ¬(∀ (x : ℝ), x^2 - 6*x + m = 0 → (∃ (y : ℝ), x ≠ y ∧ y^2 - 6*y + m = 0))) :=
by sorry

end NUMINAMATH_CALUDE_largest_k_for_distinct_roots_l2365_236595


namespace NUMINAMATH_CALUDE_ratio_expression_equality_l2365_236528

theorem ratio_expression_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 1 / 3) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_equality_l2365_236528


namespace NUMINAMATH_CALUDE_sum_is_real_product_is_real_sum_and_product_are_real_complex_numbers_satisfying_conditions_l2365_236582

-- Define complex numbers
def complex_number (a b : ℝ) : ℂ := a + b * Complex.I

-- Theorem for condition (a)
theorem sum_is_real (a b c : ℝ) :
  ∃ (z₁ z₂ : ℂ), (z₁ + z₂).im = 0 :=
sorry

-- Theorem for condition (b)
theorem product_is_real (a b k : ℝ) :
  ∃ (z₁ z₂ : ℂ), (z₁ * z₂).im = 0 :=
sorry

-- Theorem for condition (c)
theorem sum_and_product_are_real (a b : ℝ) :
  ∃ (z₁ z₂ : ℂ), (z₁ + z₂).im = 0 ∧ (z₁ * z₂).im = 0 :=
sorry

-- Main theorem combining all conditions
theorem complex_numbers_satisfying_conditions :
  (∃ (z₁ z₂ : ℂ), (z₁ + z₂).im = 0) ∧
  (∃ (z₁ z₂ : ℂ), (z₁ * z₂).im = 0) ∧
  (∃ (z₁ z₂ : ℂ), (z₁ + z₂).im = 0 ∧ (z₁ * z₂).im = 0) :=
sorry

end NUMINAMATH_CALUDE_sum_is_real_product_is_real_sum_and_product_are_real_complex_numbers_satisfying_conditions_l2365_236582


namespace NUMINAMATH_CALUDE_hilt_remaining_money_l2365_236509

def remaining_money (initial_amount cost : ℕ) : ℕ :=
  initial_amount - cost

theorem hilt_remaining_money :
  remaining_money 15 11 = 4 := by sorry

end NUMINAMATH_CALUDE_hilt_remaining_money_l2365_236509


namespace NUMINAMATH_CALUDE_base_sum_theorem_l2365_236587

/-- Represents a fraction in a given base --/
structure FractionInBase where
  numerator : ℕ
  denominator : ℕ
  base : ℕ

/-- Converts a repeating decimal to a fraction in a given base --/
def repeatingDecimalToFraction (digits : List ℕ) (base : ℕ) : FractionInBase :=
  sorry

/-- The sum of the bases R₁ and R₂ --/
def sumOfBases : ℕ := 14

theorem base_sum_theorem (R₁ R₂ : ℕ) :
  let F₁_R₁ := repeatingDecimalToFraction [4, 5] R₁
  let F₂_R₁ := repeatingDecimalToFraction [5, 4] R₁
  let F₁_R₂ := repeatingDecimalToFraction [3, 2] R₂
  let F₂_R₂ := repeatingDecimalToFraction [2, 3] R₂
  F₁_R₁.numerator * F₁_R₂.denominator = F₁_R₂.numerator * F₁_R₁.denominator ∧
  F₂_R₁.numerator * F₂_R₂.denominator = F₂_R₂.numerator * F₂_R₁.denominator →
  R₁ + R₂ = sumOfBases := by
  sorry

end NUMINAMATH_CALUDE_base_sum_theorem_l2365_236587


namespace NUMINAMATH_CALUDE_rectangular_park_area_l2365_236588

theorem rectangular_park_area (width : ℝ) (length : ℝ) (perimeter : ℝ) :
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 72 →
  width * length = 243 := by
sorry

end NUMINAMATH_CALUDE_rectangular_park_area_l2365_236588


namespace NUMINAMATH_CALUDE_number_problem_l2365_236551

theorem number_problem (N : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N = 30 → (40/100 : ℝ) * N = 360 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2365_236551


namespace NUMINAMATH_CALUDE_derivative_zero_at_negative_one_l2365_236589

theorem derivative_zero_at_negative_one (t : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^2 - 4) * (x - t)
  (deriv f) (-1) = 0 → t = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_zero_at_negative_one_l2365_236589


namespace NUMINAMATH_CALUDE_house_deal_profit_l2365_236577

theorem house_deal_profit (house_worth : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) : 
  house_worth = 10000 ∧ profit_percent = 10 ∧ loss_percent = 10 →
  let first_sale := house_worth * (1 + profit_percent / 100)
  let second_sale := first_sale * (1 - loss_percent / 100)
  first_sale - second_sale = 1100 := by
  sorry

end NUMINAMATH_CALUDE_house_deal_profit_l2365_236577


namespace NUMINAMATH_CALUDE_correct_sunset_time_l2365_236501

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : ℕ
  minutes : ℕ

/-- Adds a duration to a time -/
def addDurationToTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + d.hours * 60 + d.minutes
  let newHours := totalMinutes / 60
  let newMinutes := totalMinutes % 60
  { hours := newHours % 24, minutes := newMinutes, valid := by sorry }

theorem correct_sunset_time : 
  let sunrise : Time := { hours := 5, minutes := 35, valid := by sorry }
  let daylight : Duration := { hours := 14, minutes := 42 }
  let sunset := addDurationToTime sunrise daylight
  sunset.hours = 20 ∧ sunset.minutes = 17 := by sorry

end NUMINAMATH_CALUDE_correct_sunset_time_l2365_236501


namespace NUMINAMATH_CALUDE_choose_three_with_min_diff_3_from_14_l2365_236591

/-- The number of ways to choose three integers from 1 to 14 with minimum difference 3 -/
def choose_three_with_min_diff (n : ℕ) (k : ℕ) (min_diff : ℕ) : ℕ :=
  Nat.choose (n - k * (min_diff - 1) + k - 1) (k - 1)

/-- Theorem: There are 120 ways to choose three integers from 1 to 14 
    such that the absolute difference between any two is at least 3 -/
theorem choose_three_with_min_diff_3_from_14 : 
  choose_three_with_min_diff 14 3 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_three_with_min_diff_3_from_14_l2365_236591


namespace NUMINAMATH_CALUDE_max_value_quadratic_l2365_236583

theorem max_value_quadratic (a : ℝ) : 
  8 * a^2 + 6 * a + 2 = 0 → (∃ (x : ℝ), 3 * a + 2 ≤ x ∧ x = 5/4) :=
by sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l2365_236583


namespace NUMINAMATH_CALUDE_chocolate_chip_difference_l2365_236594

/-- The number of chocolate chips Viviana has exceeds the number Susana has -/
def viviana_more_chocolate (viviana_chocolate susana_chocolate : ℕ) : Prop :=
  viviana_chocolate > susana_chocolate

/-- The problem statement -/
theorem chocolate_chip_difference 
  (viviana_vanilla susana_chocolate : ℕ) 
  (h1 : viviana_vanilla = 20)
  (h2 : susana_chocolate = 25)
  (h3 : ∃ (viviana_chocolate susana_vanilla : ℕ), 
    viviana_more_chocolate viviana_chocolate susana_chocolate ∧
    susana_vanilla = 3 * viviana_vanilla / 4 ∧
    viviana_chocolate + viviana_vanilla + susana_chocolate + susana_vanilla = 90) :
  ∃ (viviana_chocolate : ℕ), viviana_chocolate - susana_chocolate = 5 := by
sorry

end NUMINAMATH_CALUDE_chocolate_chip_difference_l2365_236594


namespace NUMINAMATH_CALUDE_clock_equal_angles_l2365_236592

/-- The time in minutes when the hour and minute hands form equal angles with their positions at 12 o'clock -/
def equal_angle_time : ℚ := 55 + 5/13

/-- The angular speed of the minute hand in degrees per minute -/
def minute_hand_speed : ℚ := 6

/-- The angular speed of the hour hand in degrees per hour -/
def hour_hand_speed : ℚ := 30

theorem clock_equal_angles :
  let t : ℚ := equal_angle_time / 60  -- Convert minutes to hours
  minute_hand_speed * 60 * t = 360 - hour_hand_speed * t := by sorry

#eval equal_angle_time

end NUMINAMATH_CALUDE_clock_equal_angles_l2365_236592


namespace NUMINAMATH_CALUDE_bus_fraction_proof_l2365_236544

def total_distance : ℝ := 129.9999999999999
def train_fraction : ℚ := 3/5
def walk_distance : ℝ := 6.5

theorem bus_fraction_proof :
  let bus_distance := total_distance - train_fraction * total_distance - walk_distance
  bus_distance / total_distance = 7/20 := by sorry

end NUMINAMATH_CALUDE_bus_fraction_proof_l2365_236544


namespace NUMINAMATH_CALUDE_sin_shift_l2365_236550

theorem sin_shift (x : ℝ) : 
  Real.sin (4 * x - π / 3) = Real.sin (4 * (x - π / 12)) := by sorry

end NUMINAMATH_CALUDE_sin_shift_l2365_236550


namespace NUMINAMATH_CALUDE_students_per_table_l2365_236552

theorem students_per_table (num_tables : ℕ) (total_students : ℕ) 
  (h1 : num_tables = 34) (h2 : total_students = 204) : 
  total_students / num_tables = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_per_table_l2365_236552


namespace NUMINAMATH_CALUDE_min_red_cells_correct_l2365_236573

/-- Minimum number of red cells needed in an n x n grid such that at least one red cell remains
    after erasing any 2 rows and 2 columns -/
def min_red_cells (n : ℕ) : ℕ :=
  if n = 4 then 7 else n + 3

theorem min_red_cells_correct (n : ℕ) (h : n ≥ 4) :
  ∀ (red_cells : Finset (Fin n × Fin n)),
    (∀ (rows cols : Finset (Fin n)), rows.card = 2 → cols.card = 2 →
      ∃ (i j : Fin n), i ∉ rows ∧ j ∉ cols ∧ (i, j) ∈ red_cells) →
    red_cells.card ≥ min_red_cells n :=
by sorry

end NUMINAMATH_CALUDE_min_red_cells_correct_l2365_236573


namespace NUMINAMATH_CALUDE_min_pairs_to_test_l2365_236558

/-- Represents the result of testing a pair of batteries -/
inductive TestResult
| Working
| NonWorking

/-- Represents a strategy for testing battery pairs -/
def TestStrategy := List (Nat × Nat)

/-- The total number of batteries -/
def totalBatteries : Nat := 12

/-- The number of working batteries -/
def workingBatteries : Nat := 3

/-- The number of non-working batteries -/
def nonWorkingBatteries : Nat := totalBatteries - workingBatteries

/-- A function that determines if a strategy guarantees finding a working pair -/
def guaranteesFindingWorkingPair (strategy : TestStrategy) : Prop := sorry

/-- The theorem stating the minimum number of pairs to test -/
theorem min_pairs_to_test :
  ∃ (strategy : TestStrategy),
    strategy.length = 6 ∧
    guaranteesFindingWorkingPair strategy ∧
    ∀ (otherStrategy : TestStrategy),
      guaranteesFindingWorkingPair otherStrategy →
      otherStrategy.length ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_pairs_to_test_l2365_236558


namespace NUMINAMATH_CALUDE_inscribed_triangle_tangent_theorem_l2365_236548

/-- A parabola in the xy-plane -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop

/-- A triangle in the xy-plane -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Checks if a line is tangent to a parabola -/
def is_tangent (line : ℝ × ℝ → ℝ × ℝ → Prop) (parabola : Parabola) : Prop :=
  sorry

/-- Main theorem: If two sides of an inscribed triangle are tangent to a parabola, the third side is also tangent -/
theorem inscribed_triangle_tangent_theorem
  (p q : ℝ)
  (parabola1 : Parabola)
  (parabola2 : Parabola)
  (triangle : Triangle)
  (h1 : p > 0)
  (h2 : q > 0)
  (h3 : parabola1.equation = fun x y ↦ y^2 = 2*p*x)
  (h4 : parabola2.equation = fun x y ↦ x^2 = 2*q*y)
  (h5 : ∀ (x y : ℝ), parabola1.equation x y → (x = triangle.A.1 ∧ y = triangle.A.2) ∨ 
                                               (x = triangle.B.1 ∧ y = triangle.B.2) ∨ 
                                               (x = triangle.C.1 ∧ y = triangle.C.2))
  (h6 : is_tangent (fun A B ↦ ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
         (triangle.A.1 + t * (triangle.B.1 - triangle.A.1) = A.1) ∧
         (triangle.A.2 + t * (triangle.B.2 - triangle.A.2) = A.2)) parabola2)
  (h7 : is_tangent (fun B C ↦ ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
         (triangle.B.1 + t * (triangle.C.1 - triangle.B.1) = B.1) ∧
         (triangle.B.2 + t * (triangle.C.2 - triangle.B.2) = B.2)) parabola2)
  : is_tangent (fun A C ↦ ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ 
         (triangle.A.1 + t * (triangle.C.1 - triangle.A.1) = A.1) ∧
         (triangle.A.2 + t * (triangle.C.2 - triangle.A.2) = A.2)) parabola2 :=
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_tangent_theorem_l2365_236548


namespace NUMINAMATH_CALUDE_vector_angle_cosine_l2365_236538

theorem vector_angle_cosine (a b : ℝ × ℝ) :
  a + b = (2, -8) →
  a - b = (-8, 16) →
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = -63/65 := by
  sorry

end NUMINAMATH_CALUDE_vector_angle_cosine_l2365_236538


namespace NUMINAMATH_CALUDE_school_children_count_l2365_236516

theorem school_children_count : 
  ∀ (B C : ℕ), 
    B = 2 * C → 
    B = 4 * (C - 370) → 
    C = 740 := by
  sorry

end NUMINAMATH_CALUDE_school_children_count_l2365_236516


namespace NUMINAMATH_CALUDE_calculate_expression_l2365_236585

theorem calculate_expression : (-2)^49 + 2^(4^4 + 3^2 - 5^2) = -2^49 + 2^240 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2365_236585


namespace NUMINAMATH_CALUDE_middle_pile_has_five_cards_l2365_236541

/-- Represents the number of cards in each pile -/
structure CardPiles :=
  (left : ℕ)
  (middle : ℕ)
  (right : ℕ)

/-- The initial state of the card piles -/
def initial_state (x : ℕ) : CardPiles :=
  { left := x, middle := x, right := x }

/-- Condition that each pile has at least 2 cards initially -/
def valid_initial_state (s : CardPiles) : Prop :=
  s.left ≥ 2 ∧ s.middle ≥ 2 ∧ s.right ≥ 2

/-- The state after performing the four steps -/
def final_state (s : CardPiles) : CardPiles :=
  let step1 := s
  let step2 := { step1 with left := step1.left - 2, middle := step1.middle + 2 }
  let step3 := { step2 with right := step2.right - 1, middle := step2.middle + 1 }
  let step4 := { step3 with left := step3.left + step3.left, middle := step3.middle - step3.left }
  step4

/-- The main theorem stating that the middle pile always has 5 cards after the steps -/
theorem middle_pile_has_five_cards (x : ℕ) :
  let initial := initial_state x
  valid_initial_state initial →
  (final_state initial).middle = 5 :=
by sorry

end NUMINAMATH_CALUDE_middle_pile_has_five_cards_l2365_236541


namespace NUMINAMATH_CALUDE_problem_solution_l2365_236507

theorem problem_solution (a x y : ℝ) 
  (h1 : x ∈ Set.Icc (-π/4) (π/4))
  (h2 : y ∈ Set.Icc (-π/4) (π/4))
  (h3 : x^3 + Real.sin x = 2*a)
  (h4 : 4*y^3 + Real.sin y * Real.cos y = -a) :
  3 * Real.sin ((π + x)/2 + y) = 3 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2365_236507


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_165_6_l2365_236570

theorem percentage_of_360_equals_165_6 : ∃ (p : ℚ), p * 360 = 165.6 ∧ p * 100 = 46 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_165_6_l2365_236570


namespace NUMINAMATH_CALUDE_girls_in_class_l2365_236562

theorem girls_in_class (total : ℕ) (boys : ℕ) (prob : ℚ) : 
  total = 25 →
  (boys.choose 2 : ℚ) / (total.choose 2 : ℚ) = prob →
  prob = 3/25 →
  total - boys = 16 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_class_l2365_236562


namespace NUMINAMATH_CALUDE_rationalize_denominator_l2365_236502

theorem rationalize_denominator : 
  (Real.sqrt 18 + Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 8) = 5 * Real.sqrt 6 - 20 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l2365_236502


namespace NUMINAMATH_CALUDE_trapezium_k_value_l2365_236559

/-- A trapezium PQRS with specific angle relationships -/
structure Trapezium where
  /-- Angle PQR -/
  pqr : ℝ
  /-- Angle SPQ = 2 * PQR -/
  spq : ℝ
  /-- Angle RSP = 2 * SPQ -/
  rsp : ℝ
  /-- Angle QRS = k * PQR -/
  qrs : ℝ
  /-- The value of k -/
  k : ℝ
  /-- SPQ is twice PQR -/
  h_spq : spq = 2 * pqr
  /-- RSP is twice SPQ -/
  h_rsp : rsp = 2 * spq
  /-- QRS is k times PQR -/
  h_qrs : qrs = k * pqr
  /-- Sum of angles in a quadrilateral is 360° -/
  h_sum : pqr + spq + rsp + qrs = 360

/-- The value of k in the trapezium PQRS is 5 -/
theorem trapezium_k_value (t : Trapezium) : t.k = 5 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_k_value_l2365_236559


namespace NUMINAMATH_CALUDE_simplification_and_evaluation_l2365_236567

theorem simplification_and_evaluation (a : ℤ) 
  (h1 : -2 ≤ a ∧ a ≤ 2) 
  (h2 : a + 1 ≠ 0) 
  (h3 : a - 2 ≠ 0) : 
  (1 - 3 / (a + 1)) / ((a^2 - 4*a + 4) / (a + 1)) = 1 / (a - 2) := by
  sorry

end NUMINAMATH_CALUDE_simplification_and_evaluation_l2365_236567


namespace NUMINAMATH_CALUDE_expression_equality_l2365_236513

theorem expression_equality : 
  (12^4 + 324) * (24^4 + 324) * (36^4 + 324) / 
  ((6^4 + 324) * (18^4 + 324) * (30^4 + 324)) = 84/35 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2365_236513


namespace NUMINAMATH_CALUDE_modulus_of_z_l2365_236510

-- Define the complex number z
variable (z : ℂ)

-- Define the condition z(i-1) = 4
def condition (z : ℂ) : Prop := z * (Complex.I - 1) = 4

-- Theorem statement
theorem modulus_of_z (h : condition z) : Complex.abs z = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l2365_236510


namespace NUMINAMATH_CALUDE_lemonade_water_amount_l2365_236574

/- Define the ratios and amounts -/
def water_sugar_ratio : ℚ := 3
def sugar_lemon_ratio : ℚ := 3
def lemon_juice_amount : ℚ := 4

/- Define the function to calculate water amount -/
def water_amount (water_sugar : ℚ) (sugar_lemon : ℚ) (lemon : ℚ) : ℚ :=
  water_sugar * sugar_lemon * lemon

/- Theorem statement -/
theorem lemonade_water_amount :
  water_amount water_sugar_ratio sugar_lemon_ratio lemon_juice_amount = 36 := by
  sorry


end NUMINAMATH_CALUDE_lemonade_water_amount_l2365_236574


namespace NUMINAMATH_CALUDE_cyclic_reciprocal_product_l2365_236581

theorem cyclic_reciprocal_product (x y z : ℝ) 
  (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
  (h_cyclic : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x) : 
  x^2 * y^2 * z^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_cyclic_reciprocal_product_l2365_236581


namespace NUMINAMATH_CALUDE_ellen_painting_time_l2365_236586

/-- The time it takes Ellen to paint all flowers and vines -/
def total_painting_time (lily_time rose_time orchid_time vine_time : ℕ) 
                        (lily_count rose_count orchid_count vine_count : ℕ) : ℕ :=
  lily_time * lily_count + rose_time * rose_count + 
  orchid_time * orchid_count + vine_time * vine_count

/-- Theorem stating that Ellen's total painting time is 213 minutes -/
theorem ellen_painting_time : 
  total_painting_time 5 7 3 2 17 10 6 20 = 213 := by
  sorry

end NUMINAMATH_CALUDE_ellen_painting_time_l2365_236586


namespace NUMINAMATH_CALUDE_number_difference_l2365_236557

theorem number_difference (n : ℕ) : 
  n / 12 = 25 ∧ n % 12 = 11 → n - 25 = 286 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2365_236557


namespace NUMINAMATH_CALUDE_three_solutions_iff_a_values_l2365_236597

-- Define the system of equations
def equation1 (x y : ℝ) : Prop :=
  (|y - 10| + |x + 3| - 2) * (x^2 + y^2 - 6) = 0

def equation2 (x y a : ℝ) : Prop :=
  (x + 3)^2 + (y - 5)^2 = a

-- Define the solution set
def solution_set (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | equation1 p.1 p.2 ∧ equation2 p.1 p.2 a}

-- Theorem statement
theorem three_solutions_iff_a_values (a : ℝ) :
  (solution_set a).ncard = 3 ↔ (a = 49 ∨ a = 40 - 4 * Real.sqrt 51) :=
sorry

end NUMINAMATH_CALUDE_three_solutions_iff_a_values_l2365_236597


namespace NUMINAMATH_CALUDE_pyramid_prism_sum_max_pyramid_prism_sum_l2365_236532

/-- A solid formed by attaching a square pyramid to one rectangular face of a rectangular prism -/
structure PyramidPrism where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_new_faces : ℕ
  pyramid_new_edges : ℕ
  pyramid_new_vertex : ℕ

/-- The total number of exterior faces, edges, and vertices of the combined solid -/
def total_elements (pp : PyramidPrism) : ℕ :=
  (pp.prism_faces - 1 + pp.pyramid_new_faces) +
  (pp.prism_edges + pp.pyramid_new_edges) +
  (pp.prism_vertices + pp.pyramid_new_vertex)

/-- Theorem stating that the sum of faces, edges, and vertices of the combined solid is 34 -/
theorem pyramid_prism_sum (pp : PyramidPrism) 
  (h1 : pp.prism_faces = 6)
  (h2 : pp.prism_edges = 12)
  (h3 : pp.prism_vertices = 8)
  (h4 : pp.pyramid_new_faces = 4)
  (h5 : pp.pyramid_new_edges = 4)
  (h6 : pp.pyramid_new_vertex = 1) :
  total_elements pp = 34 := by
  sorry

/-- The maximum value of the sum of faces, edges, and vertices is 34 -/
theorem max_pyramid_prism_sum :
  ∀ pp : PyramidPrism, total_elements pp ≤ 34 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_prism_sum_max_pyramid_prism_sum_l2365_236532


namespace NUMINAMATH_CALUDE_no_cubic_polynomial_satisfies_conditions_l2365_236530

theorem no_cubic_polynomial_satisfies_conditions :
  ¬∃ f : ℝ → ℝ, (∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^3 + b * x^2 + c * x + d) ∧
    (∀ x, f (x^2) = (f x)^2) ∧ (∀ x, f (x^2) = f (f x)) :=
sorry

end NUMINAMATH_CALUDE_no_cubic_polynomial_satisfies_conditions_l2365_236530


namespace NUMINAMATH_CALUDE_jace_driving_problem_l2365_236561

/-- Jace's driving problem -/
theorem jace_driving_problem (speed : ℝ) (break_time : ℝ) (second_drive_time : ℝ) (total_distance : ℝ) :
  speed = 60 ∧ 
  break_time = 0.5 ∧ 
  second_drive_time = 9 ∧ 
  total_distance = 780 →
  ∃ (first_drive_time : ℝ), 
    first_drive_time * speed + second_drive_time * speed = total_distance ∧ 
    first_drive_time = 4 :=
by sorry

end NUMINAMATH_CALUDE_jace_driving_problem_l2365_236561


namespace NUMINAMATH_CALUDE_sum_a_b_values_l2365_236549

theorem sum_a_b_values (a b : ℝ) (h1 : |a| = 3) (h2 : |b| = 2) (h3 : |a-b| = b-a) :
  a + b = -1 ∨ a + b = -5 := by
sorry

end NUMINAMATH_CALUDE_sum_a_b_values_l2365_236549


namespace NUMINAMATH_CALUDE_spider_journey_l2365_236505

theorem spider_journey (r : ℝ) (third_leg : ℝ) (h1 : r = 75) (h2 : third_leg = 110) : 
  let diameter := 2 * r
  let second_leg := Real.sqrt (diameter^2 - third_leg^2)
  diameter + second_leg + third_leg = 362 := by
sorry

end NUMINAMATH_CALUDE_spider_journey_l2365_236505


namespace NUMINAMATH_CALUDE_real_number_properties_l2365_236504

theorem real_number_properties :
  (∃ n : ℝ, ∀ m : ℝ, m * n = m) ∧
  (¬ ∀ n : ℝ, n^2 ≥ n) ∧
  (¬ ∀ n : ℝ, ∃ m : ℝ, m^2 < n) ∧
  (¬ ∀ n : ℝ, n^2 < n) :=
by sorry

end NUMINAMATH_CALUDE_real_number_properties_l2365_236504


namespace NUMINAMATH_CALUDE_trigonometric_propositions_l2365_236521

open Real

theorem trigonometric_propositions :
  (¬ ∃ x : ℝ, sin x + cos x = 2) ∧
  (∃ x : ℝ, sin (2 * x) = sin x) ∧
  (∀ x ∈ Set.Icc (-π/2) (π/2), Real.sqrt ((1 + cos (2 * x)) / 2) = cos x) ∧
  (¬ ∀ x ∈ Set.Ioo 0 π, sin x > cos x) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_propositions_l2365_236521


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l2365_236564

/-- Represents the loan conditions and proves the principal amount --/
theorem loan_principal_calculation (initial_fee_rate : ℝ) (weeks : ℕ) (total_fee : ℝ) (principal : ℝ) : 
  initial_fee_rate = 0.05 →
  weeks = 2 →
  total_fee = 15 →
  (initial_fee_rate * principal) + (2 * initial_fee_rate * principal) = total_fee →
  principal = 100 := by
  sorry

#check loan_principal_calculation

end NUMINAMATH_CALUDE_loan_principal_calculation_l2365_236564


namespace NUMINAMATH_CALUDE_fraction_difference_zero_l2365_236535

theorem fraction_difference_zero (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x - y) / (Real.sqrt x + Real.sqrt y) - (x - 2 * Real.sqrt (x * y) + y) / (Real.sqrt x - Real.sqrt y) = 0 :=
by sorry

end NUMINAMATH_CALUDE_fraction_difference_zero_l2365_236535


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l2365_236512

theorem unknown_blanket_rate (blanket_price_1 blanket_price_2 average_price : ℚ)
  (num_blankets_1 num_blankets_2 num_blankets_unknown : ℕ) :
  blanket_price_1 = 100 →
  blanket_price_2 = 150 →
  num_blankets_1 = 2 →
  num_blankets_2 = 5 →
  num_blankets_unknown = 2 →
  average_price = 150 →
  ∃ unknown_price : ℚ,
    (num_blankets_1 * blanket_price_1 + num_blankets_2 * blanket_price_2 + num_blankets_unknown * unknown_price) /
    (num_blankets_1 + num_blankets_2 + num_blankets_unknown) = average_price ∧
    unknown_price = 200 :=
by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l2365_236512


namespace NUMINAMATH_CALUDE_solve_equation_l2365_236519

theorem solve_equation : ∃ x : ℚ, 5 * (x - 10) = 3 * (3 - 3 * x) + 9 ∧ x = 34/7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2365_236519


namespace NUMINAMATH_CALUDE_min_value_expression_l2365_236503

theorem min_value_expression (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (m : ℝ), m = 1 - Real.sqrt 2 ∧ ∀ (z : ℝ), z = (2 * x * y) / (x + y - 1) → m ≤ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2365_236503


namespace NUMINAMATH_CALUDE_complex_distance_sum_constant_l2365_236543

theorem complex_distance_sum_constant (z : ℂ) (h : Complex.abs (z - (3 - 2*I)) = 4) :
  Complex.abs (z + (1 - I))^2 + Complex.abs (z - (7 - 3*I))^2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_sum_constant_l2365_236543


namespace NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2365_236553

/-- The radius of the Sun in meters -/
def sun_radius : ℝ := 696000000

/-- Scientific notation representation of the Sun's radius -/
def sun_radius_scientific : ℝ := 6.96 * (10 ^ 8)

/-- Theorem stating that the Sun's radius is correctly expressed in scientific notation -/
theorem sun_radius_scientific_notation : sun_radius = sun_radius_scientific :=
sorry

end NUMINAMATH_CALUDE_sun_radius_scientific_notation_l2365_236553


namespace NUMINAMATH_CALUDE_katherines_fruits_l2365_236525

theorem katherines_fruits (apples pears bananas : ℕ) : 
  apples = 4 →
  pears = 3 * apples →
  apples + pears + bananas = 21 →
  bananas = 5 := by
sorry

end NUMINAMATH_CALUDE_katherines_fruits_l2365_236525


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2365_236526

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 - 2 * t^2 + t - 4) * (4 * t^2 - 2 * t + 5) =
  12 * t^5 - 14 * t^4 + 23 * t^3 - 28 * t^2 + 13 * t - 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2365_236526


namespace NUMINAMATH_CALUDE_carol_invitation_packs_l2365_236508

def invitations_per_pack : ℕ := 3
def friends_to_invite : ℕ := 9
def extra_invitations : ℕ := 3

theorem carol_invitation_packs :
  (friends_to_invite + extra_invitations) / invitations_per_pack = 4 := by
  sorry

end NUMINAMATH_CALUDE_carol_invitation_packs_l2365_236508


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2365_236520

-- Problem 1
theorem problem_1 : (π - 3.14) ^ 0 + (1/2) ^ (-1) + (-1) ^ 2023 = 2 := by sorry

-- Problem 2
theorem problem_2 (b : ℝ) : (-b)^2 * b + 6*b^4 / (2*b) + (-2*b)^3 = -4*b^3 := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (x - 1)^2 - x*(x + 2) = -4*x + 1 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_l2365_236520


namespace NUMINAMATH_CALUDE_pet_store_cages_l2365_236584

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 7

/-- The total number of birds in the pet store -/
def total_birds : ℕ := 72

/-- The number of bird cages in the pet store -/
def num_cages : ℕ := total_birds / (parrots_per_cage + parakeets_per_cage)

theorem pet_store_cages : num_cages = 8 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_cages_l2365_236584


namespace NUMINAMATH_CALUDE_triangle_area_squared_l2365_236596

theorem triangle_area_squared (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + CA) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - CA))
  AB = 7 ∧ BC = 9 ∧ CA = 4 → area^2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_squared_l2365_236596


namespace NUMINAMATH_CALUDE_largest_package_size_l2365_236531

theorem largest_package_size (ming_pencils catherine_pencils : ℕ) 
  (h1 : ming_pencils = 40)
  (h2 : catherine_pencils = 24) :
  Nat.gcd ming_pencils catherine_pencils = 8 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l2365_236531


namespace NUMINAMATH_CALUDE_gcd_459_357_l2365_236575

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2365_236575


namespace NUMINAMATH_CALUDE_second_mixture_percentage_l2365_236593

/-- Represents the composition of an alcohol mixture -/
structure AlcoholMixture where
  volume : ℝ
  percentage : ℝ

/-- Proves that the second mixture has 50% alcohol content -/
theorem second_mixture_percentage
  (total_mixture : AlcoholMixture)
  (first_mixture : AlcoholMixture)
  (h_total_volume : total_mixture.volume = 10)
  (h_total_percentage : total_mixture.percentage = 45)
  (h_first_volume : first_mixture.volume = 2.5)
  (h_first_percentage : first_mixture.percentage = 30)
  : ∃ (second_mixture : AlcoholMixture),
    second_mixture.volume = total_mixture.volume - first_mixture.volume ∧
    second_mixture.percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_second_mixture_percentage_l2365_236593


namespace NUMINAMATH_CALUDE_divisors_of_square_l2365_236554

-- Define a function that counts the number of divisors
def count_divisors (n : ℕ) : ℕ := sorry

-- Theorem statement
theorem divisors_of_square (n : ℕ) :
  count_divisors n = 5 → count_divisors (n^2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_square_l2365_236554


namespace NUMINAMATH_CALUDE_keith_attended_four_games_l2365_236565

/-- The number of football games Keith attended -/
def games_attended (total_games missed_games : ℕ) : ℕ :=
  total_games - missed_games

/-- Theorem: Keith attended 4 football games -/
theorem keith_attended_four_games (total_games missed_games : ℕ) 
  (h1 : total_games = 8)
  (h2 : missed_games = 4) : 
  games_attended total_games missed_games = 4 := by
  sorry

end NUMINAMATH_CALUDE_keith_attended_four_games_l2365_236565


namespace NUMINAMATH_CALUDE_incorrect_expression_l2365_236524

def repeating_decimal (X Y Z : ℕ) (t u v : ℕ) : ℚ :=
  sorry

theorem incorrect_expression
  (E : ℚ) (X Y Z : ℕ) (t u v : ℕ)
  (h_E : E = repeating_decimal X Y Z t u v) :
  ¬(10^t * (10^(u+v) - 1) * E = Z * (Y - 1)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_expression_l2365_236524


namespace NUMINAMATH_CALUDE_negative_cube_squared_l2365_236566

theorem negative_cube_squared (x : ℝ) : (-x^3)^2 = x^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_cube_squared_l2365_236566


namespace NUMINAMATH_CALUDE_larger_number_of_pair_l2365_236560

theorem larger_number_of_pair (x y : ℝ) (h_diff : x - y = 7) (h_sum : x + y = 47) :
  max x y = 27 := by
sorry

end NUMINAMATH_CALUDE_larger_number_of_pair_l2365_236560


namespace NUMINAMATH_CALUDE_cubic_function_uniqueness_l2365_236590

/-- Given a cubic function f(x) = ax³ + bx² + cx, prove that if it has critical points at x = ±1
    and its derivative at x = 0 is -3, then f(x) = x³ - 3x. -/
theorem cubic_function_uniqueness (a b c : ℝ) :
  let f := fun (x : ℝ) ↦ a * x^3 + b * x^2 + c * x
  let f' := fun (x : ℝ) ↦ 3 * a * x^2 + 2 * b * x + c
  (f' 1 = 0 ∧ f' (-1) = 0 ∧ f' 0 = -3) →
  (∀ x, f x = x^3 - 3*x) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_uniqueness_l2365_236590


namespace NUMINAMATH_CALUDE_mike_remaining_cards_l2365_236546

/-- Given Mike's initial number of baseball cards and the number of cards sold to Sam and Alex,
    calculate the number of cards Mike has left. -/
def remaining_cards (initial : ℕ) (sold_to_sam : ℕ) (sold_to_alex : ℕ) : ℕ :=
  initial - (sold_to_sam + sold_to_alex)

/-- Theorem stating that Mike has 59 baseball cards left after selling to Sam and Alex. -/
theorem mike_remaining_cards :
  remaining_cards 87 13 15 = 59 := by
  sorry

#eval remaining_cards 87 13 15

end NUMINAMATH_CALUDE_mike_remaining_cards_l2365_236546


namespace NUMINAMATH_CALUDE_odd_quadratic_implies_zero_coefficient_l2365_236563

/-- A function f is odd if f(-x) = -f(x) for all x in its domain -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The quadratic function f(x) = ax^2 + 2x -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x

theorem odd_quadratic_implies_zero_coefficient (a : ℝ) :
  IsOdd (f a) → a = 0 := by
  sorry


end NUMINAMATH_CALUDE_odd_quadratic_implies_zero_coefficient_l2365_236563


namespace NUMINAMATH_CALUDE_smaller_number_proof_l2365_236518

theorem smaller_number_proof (x y : ℝ) (h1 : x > y) (h2 : x - y = 1860) (h3 : 0.075 * x = 0.125 * y) : y = 2790 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l2365_236518


namespace NUMINAMATH_CALUDE_cauchy_functional_equation_l2365_236568

-- Define the property of the function
def IsCauchyFunctional (f : ℚ → ℝ) : Prop :=
  ∀ x y : ℚ, f (x + y) = f x + f y

-- State the theorem
theorem cauchy_functional_equation :
  ∀ f : ℚ → ℝ, IsCauchyFunctional f →
  ∃ a : ℝ, ∀ q : ℚ, f q = a * q :=
sorry

end NUMINAMATH_CALUDE_cauchy_functional_equation_l2365_236568


namespace NUMINAMATH_CALUDE_student_ticket_price_l2365_236540

theorem student_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℕ) 
  (student_tickets : ℕ) 
  (non_student_price : ℕ) :
  total_tickets = 2000 →
  total_revenue = 20960 →
  student_tickets = 520 →
  non_student_price = 11 →
  ∃ (student_price : ℕ),
    student_price * student_tickets + 
    non_student_price * (total_tickets - student_tickets) = 
    total_revenue ∧ student_price = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_student_ticket_price_l2365_236540


namespace NUMINAMATH_CALUDE_largest_common_divisor_l2365_236515

theorem largest_common_divisor : ∃ (d : ℕ), 
  d = 98 ∧ 
  (∀ (k : ℕ), k ∈ {28, 49, 98} ∪ {n : ℕ | n > 49 ∧ n % 7 = 0 ∧ n % 2 = 1} ∪ {n : ℕ | n > 98 ∧ n % 7 = 0 ∧ n % 2 = 0} → 
    (13511 % d = 13903 % d ∧ 13511 % d = 14589 % d) → k ≤ d) :=
by sorry

end NUMINAMATH_CALUDE_largest_common_divisor_l2365_236515


namespace NUMINAMATH_CALUDE_local_minimum_condition_l2365_236536

/-- The function f(x) = x³ - 2ax + a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 2*a*x + a

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a

theorem local_minimum_condition (a : ℝ) :
  (∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ IsLocalMin (f a) x) ↔ 0 < a ∧ a < 3/2 := by sorry

end NUMINAMATH_CALUDE_local_minimum_condition_l2365_236536


namespace NUMINAMATH_CALUDE_constant_value_proof_l2365_236523

theorem constant_value_proof (x y a : ℝ) 
  (h1 : (a * x + 4 * y) / (x - 2 * y) = 25)
  (h2 : x / (2 * y) = 3 / 2) : 
  a = 7 := by
sorry

end NUMINAMATH_CALUDE_constant_value_proof_l2365_236523


namespace NUMINAMATH_CALUDE_gcd_property_l2365_236511

theorem gcd_property (n : ℤ) : 
  (∃ k : ℤ, n = 31 * k - 11) ↔ Int.gcd (5 * n - 7) (3 * n + 2) = 31 := by
  sorry

end NUMINAMATH_CALUDE_gcd_property_l2365_236511


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2365_236545

theorem smallest_integer_with_remainders : ∃! n : ℕ, 
  n > 0 ∧
  n % 5 = 1 ∧
  n % 7 = 2 ∧
  n % 9 = 3 ∧
  n % 11 = 4 ∧
  ∀ m : ℕ, m > 0 ∧ m % 5 = 1 ∧ m % 7 = 2 ∧ m % 9 = 3 ∧ m % 11 = 4 → n ≤ m :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l2365_236545


namespace NUMINAMATH_CALUDE_paint_jar_capacity_l2365_236539

theorem paint_jar_capacity (mary_dragon : ℝ) (mike_castle : ℝ) (sun : ℝ) 
  (h1 : mary_dragon = 3)
  (h2 : mike_castle = mary_dragon + 2)
  (h3 : sun = 5) :
  mary_dragon + mike_castle + sun = 13 := by
  sorry

end NUMINAMATH_CALUDE_paint_jar_capacity_l2365_236539


namespace NUMINAMATH_CALUDE_exponential_function_fixed_point_l2365_236579

/-- The function f(x) = a^(x-2) - 1 always passes through the point (2, 0) for any a > 0 and a ≠ 1 -/
theorem exponential_function_fixed_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2) - 1
  f 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_fixed_point_l2365_236579


namespace NUMINAMATH_CALUDE_decagon_diagonals_l2365_236522

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A decagon has 10 sides -/
def decagon_sides : ℕ := 10

theorem decagon_diagonals : num_diagonals decagon_sides = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l2365_236522


namespace NUMINAMATH_CALUDE_range_of_a_l2365_236506

def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc 0 (Real.pi / 2), Real.cos x ^ 2 + 2 * Real.cos x - a = 0

def q (a : ℝ) : Prop := ∀ x : ℝ, x ^ 2 + 2 * a * x - 8 + 6 * a ≥ 0

theorem range_of_a :
  ∀ a : ℝ, (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ Set.Ioo 0 2 ∪ Set.Ioo 3 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2365_236506


namespace NUMINAMATH_CALUDE_specific_parallelepiped_face_areas_l2365_236578

/-- Represents a parallelepiped with given properties -/
structure Parallelepiped where
  h₁ : ℝ  -- Distance from first vertex to opposite face
  h₂ : ℝ  -- Distance from second vertex to opposite face
  h₃ : ℝ  -- Distance from third vertex to opposite face
  total_surface_area : ℝ  -- Total surface area of the parallelepiped

/-- The areas of the faces of the parallelepiped -/
def face_areas (p : Parallelepiped) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the areas of the faces for a specific parallelepiped -/
theorem specific_parallelepiped_face_areas :
  let p : Parallelepiped := {
    h₁ := 2,
    h₂ := 3,
    h₃ := 4,
    total_surface_area := 36
  }
  face_areas p = (108/13, 72/13, 54/13) := by sorry

end NUMINAMATH_CALUDE_specific_parallelepiped_face_areas_l2365_236578


namespace NUMINAMATH_CALUDE_van_tire_usage_l2365_236598

/-- Represents the number of miles each tire is used in a van with a tire rotation system -/
def miles_per_tire (total_miles : ℕ) (total_tires : ℕ) (simultaneous_tires : ℕ) : ℕ :=
  (total_miles * simultaneous_tires) / total_tires

/-- Theorem stating that in a van with 6 tires, where 4 are used simultaneously,
    traveling 40,000 miles results in each tire being used for approximately 26,667 miles -/
theorem van_tire_usage :
  miles_per_tire 40000 6 4 = 26667 := by
  sorry

end NUMINAMATH_CALUDE_van_tire_usage_l2365_236598


namespace NUMINAMATH_CALUDE_banana_cost_l2365_236533

/-- The price of bananas in dollars per 8 pounds -/
def banana_price : ℝ := 6

/-- The quantity of bananas in a standard unit (in pounds) -/
def standard_quantity : ℝ := 8

/-- The discount rate for purchases above 20 pounds -/
def discount_rate : ℝ := 0.1

/-- The threshold quantity for applying the discount (in pounds) -/
def discount_threshold : ℝ := 20

/-- The quantity of bananas to be purchased (in pounds) -/
def purchase_quantity : ℝ := 24

/-- Theorem stating the total cost of bananas -/
theorem banana_cost : 
  let price_per_pound := banana_price / standard_quantity
  let total_cost_before_discount := price_per_pound * purchase_quantity
  let discount_amount := if purchase_quantity > discount_threshold
                         then total_cost_before_discount * discount_rate
                         else 0
  let final_cost := total_cost_before_discount - discount_amount
  final_cost = 16.2 := by sorry

end NUMINAMATH_CALUDE_banana_cost_l2365_236533


namespace NUMINAMATH_CALUDE_teresa_total_score_l2365_236534

def teresa_scores (science music social_studies : ℕ) : ℕ → Prop :=
  λ total => 
    let physics := music / 2
    total = science + music + social_studies + physics

theorem teresa_total_score : 
  teresa_scores 70 80 85 275 := by sorry

end NUMINAMATH_CALUDE_teresa_total_score_l2365_236534


namespace NUMINAMATH_CALUDE_inscribed_rectangle_sides_l2365_236529

theorem inscribed_rectangle_sides (R : ℝ) (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (a^2 + b^2 = 4 * R^2) ∧ (a * b = (1/2) * π * R^2) →
  ((a = (R * Real.sqrt (π + 4) + R * Real.sqrt (4 - π)) / 2 ∧
    b = (R * Real.sqrt (π + 4) - R * Real.sqrt (4 - π)) / 2) ∨
   (a = (R * Real.sqrt (π + 4) - R * Real.sqrt (4 - π)) / 2 ∧
    b = (R * Real.sqrt (π + 4) + R * Real.sqrt (4 - π)) / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_sides_l2365_236529


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l2365_236547

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line
def Line (a b c : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}

-- Define the problem
theorem circle_and_tangent_lines 
  (C : Circle) 
  (h1 : (0, -6) ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2})
  (h2 : (1, -5) ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2})
  (h3 : C.center ∈ Line 1 (-1) 1) :
  (∀ x y : ℝ, (x + 3)^2 + (y + 2)^2 = 25 ↔ (x, y) ∈ {p : ℝ × ℝ | (p.1 - C.center.1)^2 + (p.2 - C.center.2)^2 = C.radius^2}) ∧
  (∀ x y : ℝ, (x = 2 ∨ 3*x - 4*y + 26 = 0) ↔ 
    ((x, y) ∈ Line 1 (-1) (-2) ∧ 
     ((x - 2)^2 + (y - 8)^2) * C.radius^2 = ((x - C.center.1)^2 + (y - C.center.2)^2) * ((2 - C.center.1)^2 + (8 - C.center.2)^2))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l2365_236547


namespace NUMINAMATH_CALUDE_triangle_ABC_properties_l2365_236517

theorem triangle_ABC_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  b = 4 →
  (Real.cos B) / (Real.cos C) = 4 / (2 * a - c) →
  -- Conditions for a valid triangle
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  -- Prove the following
  B = π / 3 ∧
  (∀ S : Real, S = (1/2) * a * c * Real.sin B → S ≤ 4 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_ABC_properties_l2365_236517


namespace NUMINAMATH_CALUDE_range_of_a_l2365_236572

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- State the theorem
theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a = 1 ∨ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2365_236572


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2365_236537

theorem pure_imaginary_product (x : ℝ) : 
  (∃ y : ℝ, (x + Complex.I) * ((x + 1) + Complex.I) * ((x + 2) + Complex.I) = Complex.I * y) ↔ 
  x = -3 ∨ x = -1 ∨ x = 1 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2365_236537


namespace NUMINAMATH_CALUDE_circle_radius_sqrt29_l2365_236556

/-- A circle with center on the x-axis passing through two given points has radius √29 -/
theorem circle_radius_sqrt29 (x : ℝ) :
  (x - 1)^2 + 5^2 = (x - 2)^2 + 4^2 →
  Real.sqrt ((x - 1)^2 + 5^2) = Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_sqrt29_l2365_236556


namespace NUMINAMATH_CALUDE_ae_be_implies_p_or_q_l2365_236500

theorem ae_be_implies_p_or_q (a b : ℝ) (h1 : a ≠ b) (h2 : a * Real.exp a = b * Real.exp b) :
  (Real.log a + a = Real.log b + b) ∨ ((a + 1) * (b + 1) < 0) := by
  sorry

end NUMINAMATH_CALUDE_ae_be_implies_p_or_q_l2365_236500


namespace NUMINAMATH_CALUDE_assignment_ways_for_given_tasks_l2365_236555

/-- The number of ways to assign people to tasks --/
def assignment_ways (total_people : ℕ) (selected_people : ℕ) (task_a_people : ℕ) : ℕ :=
  Nat.choose total_people selected_people *
  Nat.choose selected_people task_a_people *
  Nat.factorial (selected_people - task_a_people)

/-- Theorem stating the number of ways to assign 4 people out of 10 to the given tasks --/
theorem assignment_ways_for_given_tasks :
  assignment_ways 10 4 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_assignment_ways_for_given_tasks_l2365_236555


namespace NUMINAMATH_CALUDE_probability_quarter_or_dime_l2365_236569

/-- Represents the types of coins in the jar -/
inductive Coin
  | Quarter
  | Nickel
  | Dime
  | Penny

/-- The value of each coin type in cents -/
def coin_value : Coin → ℕ
  | Coin.Quarter => 25
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Penny => 1

/-- The total value of each coin type in the jar in cents -/
def total_value : Coin → ℕ
  | Coin.Quarter => 1500
  | Coin.Nickel => 1500
  | Coin.Dime => 1000
  | Coin.Penny => 500

/-- The number of coins of each type in the jar -/
def coin_count (c : Coin) : ℕ := total_value c / coin_value c

/-- The total number of coins in the jar -/
def total_coins : ℕ := (coin_count Coin.Quarter) + (coin_count Coin.Nickel) + (coin_count Coin.Dime) + (coin_count Coin.Penny)

/-- The probability of randomly choosing either a quarter or a dime from the jar -/
theorem probability_quarter_or_dime : 
  (coin_count Coin.Quarter + coin_count Coin.Dime : ℚ) / total_coins = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_quarter_or_dime_l2365_236569


namespace NUMINAMATH_CALUDE_length_EM_is_sqrt_6_l2365_236542

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  focus : ℝ × ℝ

/-- Line structure -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Point structure -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the parabola y² = 4x -/
def parabola : Parabola :=
  { p := 1
  , focus := (1, 0) }

/-- Definition of the line l passing through F and intersecting the parabola -/
def line_l : Line :=
  sorry

/-- Definition of points A and B where line l intersects the parabola -/
def point_A : Point :=
  sorry

def point_B : Point :=
  sorry

/-- Definition of point E (foot of the perpendicular) -/
def point_E : Point :=
  sorry

/-- Definition of point M (intersection of perpendicular bisector with x-axis) -/
def point_M : Point :=
  sorry

/-- Statement: The length of EM is √6 -/
theorem length_EM_is_sqrt_6 (h : Real.sqrt ((point_E.x - point_M.x)^2 + (point_E.y - point_M.y)^2) = Real.sqrt 6) :
  Real.sqrt ((point_E.x - point_M.x)^2 + (point_E.y - point_M.y)^2) = Real.sqrt 6 :=
sorry

end NUMINAMATH_CALUDE_length_EM_is_sqrt_6_l2365_236542


namespace NUMINAMATH_CALUDE_systematic_sampling_l2365_236576

theorem systematic_sampling (total_students : Nat) (num_groups : Nat) (selected_in_first_group : Nat) (target_group : Nat) : 
  total_students = 480 →
  num_groups = 30 →
  selected_in_first_group = 5 →
  target_group = 8 →
  (total_students / num_groups) * (target_group - 1) + selected_in_first_group = 117 :=
by
  sorry

#check systematic_sampling

end NUMINAMATH_CALUDE_systematic_sampling_l2365_236576
