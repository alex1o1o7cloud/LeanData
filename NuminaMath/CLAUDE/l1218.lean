import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achievable_l1218_121823

theorem quadratic_minimum (x : ℝ) : 7 * x^2 - 28 * x + 1425 ≥ 1397 :=
sorry

theorem quadratic_minimum_achievable : ∃ x : ℝ, 7 * x^2 - 28 * x + 1425 = 1397 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_quadratic_minimum_achievable_l1218_121823


namespace NUMINAMATH_CALUDE_f_comparison_l1218_121872

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the properties of f
variable (h_even : ∀ x, f (-x) = f x)
variable (h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

-- State the theorem
theorem f_comparison (a : ℝ) : f (-3/4) ≥ f (a^2 - a + 1) := by
  sorry

end NUMINAMATH_CALUDE_f_comparison_l1218_121872


namespace NUMINAMATH_CALUDE_train_passing_jogger_time_l1218_121814

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger_time
  (jogger_speed : ℝ)
  (train_speed : ℝ)
  (train_length : ℝ)
  (initial_distance : ℝ)
  (h_jogger_speed : jogger_speed = 9 * (1000 / 3600))
  (h_train_speed : train_speed = 45 * (1000 / 3600))
  (h_train_length : train_length = 120)
  (h_initial_distance : initial_distance = 250) :
  (initial_distance + train_length) / (train_speed - jogger_speed) = 37 := by
sorry

end NUMINAMATH_CALUDE_train_passing_jogger_time_l1218_121814


namespace NUMINAMATH_CALUDE_product_of_primes_with_sum_85_l1218_121863

theorem product_of_primes_with_sum_85 (p q : ℕ) : 
  Nat.Prime p → Nat.Prime q → p + q = 85 → p * q = 166 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_with_sum_85_l1218_121863


namespace NUMINAMATH_CALUDE_no_solution_set_characterization_l1218_121839

/-- The quadratic function f(x) = x² - 2x + 2 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 2

/-- The set of values k for which f(x) = k has no real solutions -/
def no_solution_set : Set ℝ := {k | ∀ x, f x ≠ k}

/-- Theorem stating that the no_solution_set is equivalent to {k | k < 1} -/
theorem no_solution_set_characterization :
  no_solution_set = {k | k < 1} := by sorry

end NUMINAMATH_CALUDE_no_solution_set_characterization_l1218_121839


namespace NUMINAMATH_CALUDE_tuesday_greatest_diff_greatest_diff_day_is_tuesday_l1218_121847

-- Define the temperature difference for each day
def monday_diff : ℤ := 5 - 2
def tuesday_diff : ℤ := 4 - (-1)
def wednesday_diff : ℤ := 0 - (-4)

-- Theorem stating that Tuesday has the greatest temperature difference
theorem tuesday_greatest_diff : 
  tuesday_diff > monday_diff ∧ tuesday_diff > wednesday_diff :=
by
  sorry

-- Define a function to get the day with the greatest temperature difference
def day_with_greatest_diff : String :=
  if tuesday_diff > monday_diff ∧ tuesday_diff > wednesday_diff then
    "Tuesday"
  else if monday_diff > tuesday_diff ∧ monday_diff > wednesday_diff then
    "Monday"
  else
    "Wednesday"

-- Theorem stating that the day with the greatest temperature difference is Tuesday
theorem greatest_diff_day_is_tuesday : 
  day_with_greatest_diff = "Tuesday" :=
by
  sorry

end NUMINAMATH_CALUDE_tuesday_greatest_diff_greatest_diff_day_is_tuesday_l1218_121847


namespace NUMINAMATH_CALUDE_function_behavior_l1218_121897

def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) := 
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

theorem function_behavior (f : ℝ → ℝ) 
  (h_even : is_even f)
  (h_prop : ∀ x, f x = -f (2 - x))
  (h_decr : is_decreasing_on f 1 2) :
  is_increasing_on f (-2) (-1) ∧ is_increasing_on f 3 4 := by
  sorry

end NUMINAMATH_CALUDE_function_behavior_l1218_121897


namespace NUMINAMATH_CALUDE_point_on_graph_l1218_121871

theorem point_on_graph (x y : ℝ) : 
  (x = 1 ∧ y = 4) → (y = 4 * x) := by sorry

end NUMINAMATH_CALUDE_point_on_graph_l1218_121871


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_from_medians_l1218_121808

/-- A right triangle with specific median lengths has a hypotenuse of 3√51 -/
theorem right_triangle_hypotenuse_from_medians 
  (a b c : ℝ) 
  (right_triangle : a^2 + b^2 = c^2) 
  (median1 : (b^2 + (a/2)^2) = 7^2) 
  (median2 : (a^2 + (b/2)^2) = (3*Real.sqrt 13)^2) : 
  c = 3 * Real.sqrt 51 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_hypotenuse_from_medians_l1218_121808


namespace NUMINAMATH_CALUDE_expression_evaluation_l1218_121813

theorem expression_evaluation :
  let x : ℝ := 2
  let expr := (x^2 + x) / (x^2 - 2*x + 1) / ((2 / (x - 1)) - (1 / x))
  expr = 4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1218_121813


namespace NUMINAMATH_CALUDE_dividend_rate_calculation_l1218_121811

/-- Dividend calculation problem -/
theorem dividend_rate_calculation
  (preferred_shares : ℕ)
  (common_shares : ℕ)
  (par_value : ℚ)
  (common_dividend_rate : ℚ)
  (total_annual_dividend : ℚ)
  (h1 : preferred_shares = 1200)
  (h2 : common_shares = 3000)
  (h3 : par_value = 50)
  (h4 : common_dividend_rate = 7/200)  -- 3.5% converted to a fraction
  (h5 : total_annual_dividend = 16500) :
  let preferred_dividend_rate := (total_annual_dividend - 2 * common_shares * par_value * common_dividend_rate) / (preferred_shares * par_value)
  preferred_dividend_rate = 1/10 := by sorry

end NUMINAMATH_CALUDE_dividend_rate_calculation_l1218_121811


namespace NUMINAMATH_CALUDE_correct_borrowing_process_l1218_121867

/-- Represents the steps in the book borrowing process -/
inductive BorrowingStep
  | StorageEntry
  | LocatingBook
  | Reading
  | Borrowing
  | StorageExit
  | Returning

/-- Defines the correct order of the book borrowing process -/
def correctBorrowingOrder : List BorrowingStep :=
  [BorrowingStep.StorageEntry, BorrowingStep.LocatingBook, BorrowingStep.Reading, 
   BorrowingStep.Borrowing, BorrowingStep.StorageExit, BorrowingStep.Returning]

/-- Theorem stating that the defined order is correct -/
theorem correct_borrowing_process :
  correctBorrowingOrder = [BorrowingStep.StorageEntry, BorrowingStep.LocatingBook, 
    BorrowingStep.Reading, BorrowingStep.Borrowing, BorrowingStep.StorageExit, 
    BorrowingStep.Returning] :=
by
  sorry


end NUMINAMATH_CALUDE_correct_borrowing_process_l1218_121867


namespace NUMINAMATH_CALUDE_min_square_side_length_l1218_121869

theorem min_square_side_length (square_area_min : ℝ) (circle_area_min : ℝ) :
  square_area_min = 900 →
  circle_area_min = 100 →
  ∃ (s : ℝ),
    s^2 ≥ square_area_min ∧
    π * (s/2)^2 ≥ circle_area_min ∧
    ∀ (t : ℝ), (t^2 ≥ square_area_min ∧ π * (t/2)^2 ≥ circle_area_min) → s ≤ t :=
by
  sorry

#check min_square_side_length

end NUMINAMATH_CALUDE_min_square_side_length_l1218_121869


namespace NUMINAMATH_CALUDE_factorization_equality_l1218_121891

theorem factorization_equality (x : ℝ) : x * (x - 3) - x + 3 = (x - 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1218_121891


namespace NUMINAMATH_CALUDE_garden_usable_area_l1218_121855

/-- Calculate the usable area of a rectangular garden with a square pond in one corner -/
theorem garden_usable_area 
  (garden_length : ℝ) 
  (garden_width : ℝ) 
  (pond_side : ℝ) 
  (h1 : garden_length = 20) 
  (h2 : garden_width = 18) 
  (h3 : pond_side = 4) : 
  garden_length * garden_width - pond_side * pond_side = 344 := by
  sorry

#check garden_usable_area

end NUMINAMATH_CALUDE_garden_usable_area_l1218_121855


namespace NUMINAMATH_CALUDE_impossible_sum_and_reciprocal_sum_l1218_121821

theorem impossible_sum_and_reciprocal_sum (x y z : ℝ) :
  x + y + z = 0 ∧ 1/x + 1/y + 1/z = 0 →
  x^1988 + y^1988 + z^1988 = 1/x^1988 + 1/y^1988 + 1/z^1988 :=
by sorry

end NUMINAMATH_CALUDE_impossible_sum_and_reciprocal_sum_l1218_121821


namespace NUMINAMATH_CALUDE_min_value_a_l1218_121833

theorem min_value_a (a : ℝ) (h1 : a > 0) 
  (h2 : ∀ (x y : ℝ), x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) : 
  a ≥ 4 ∧ ∀ (ε : ℝ), ε > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (x + y) * (1/(x) + (4 - ε)/y) < 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_l1218_121833


namespace NUMINAMATH_CALUDE_ice_cream_flavor_ratio_l1218_121898

def total_flavors : ℕ := 100
def flavors_two_years_ago : ℕ := total_flavors / 4
def flavors_remaining : ℕ := 25
def flavors_tried_total : ℕ := total_flavors - flavors_remaining
def flavors_last_year : ℕ := flavors_tried_total - flavors_two_years_ago

theorem ice_cream_flavor_ratio :
  (flavors_last_year : ℚ) / flavors_two_years_ago = 2 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavor_ratio_l1218_121898


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l1218_121886

theorem quadratic_equation_roots_ratio (m : ℝ) : 
  (∃ r s : ℝ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r^2 - 4*r + m = 0 ∧ s^2 - 4*s + m = 0) → m = 3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_ratio_l1218_121886


namespace NUMINAMATH_CALUDE_dave_added_sixty_apps_l1218_121899

/-- Calculates the number of apps Dave added to his phone -/
def apps_added (initial : ℕ) (removed : ℕ) (final : ℕ) : ℕ :=
  final - (initial - removed)

/-- Proves that Dave added 60 apps to his phone -/
theorem dave_added_sixty_apps :
  apps_added 50 10 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_dave_added_sixty_apps_l1218_121899


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1218_121822

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x - 2) = 8 → x = 66 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1218_121822


namespace NUMINAMATH_CALUDE_snickers_cost_calculation_l1218_121875

/-- The cost of a single piece of Snickers -/
def snickers_cost : ℚ := 1.5

/-- The number of Snickers pieces Julia bought -/
def snickers_count : ℕ := 2

/-- The number of M&M's packs Julia bought -/
def mm_count : ℕ := 3

/-- The cost of a pack of M&M's in terms of Snickers pieces -/
def mm_cost_in_snickers : ℕ := 2

/-- The total amount Julia gave to the cashier -/
def total_paid : ℚ := 20

/-- The change Julia received -/
def change_received : ℚ := 8

theorem snickers_cost_calculation :
  snickers_cost * (snickers_count + mm_count * mm_cost_in_snickers) = total_paid - change_received :=
by sorry

end NUMINAMATH_CALUDE_snickers_cost_calculation_l1218_121875


namespace NUMINAMATH_CALUDE_corner_with_same_color_l1218_121817

/-- Definition of a "corner" figure -/
def Corner (square : Fin 2017 → Fin 2017 → Fin 120) : Prop :=
  ∃ (i j : Fin 2017) (dir : Bool),
    let horizontal := if dir then (fun k => square i (j + k)) else (fun k => square (i + k) j)
    let vertical := if dir then (fun k => square (i + k) j) else (fun k => square i (j + k))
    (∀ k : Fin 10, horizontal k ∈ Set.range horizontal) ∧
    (∀ k : Fin 10, vertical k ∈ Set.range vertical) ∧
    (square i j ∈ Set.range horizontal ∪ Set.range vertical)

/-- The main theorem -/
theorem corner_with_same_color (square : Fin 2017 → Fin 2017 → Fin 120) :
  ∃ (corner : Corner square), 
    ∃ (c1 c2 : Fin 2017 × Fin 2017), c1 ≠ c2 ∧ 
      square c1.1 c1.2 = square c2.1 c2.2 :=
sorry

end NUMINAMATH_CALUDE_corner_with_same_color_l1218_121817


namespace NUMINAMATH_CALUDE_cosine_sum_range_inverse_tangent_sum_l1218_121870

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C)
  (positive_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (sine_law : a / Real.sin A = b / Real.sin B)
  (cosine_law : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)

-- Part 1
theorem cosine_sum_range (t : Triangle) (h : t.B = π/3) :
  1/2 < Real.cos t.A + Real.cos t.C ∧ Real.cos t.A + Real.cos t.C ≤ 1 :=
sorry

-- Part 2
theorem inverse_tangent_sum (t : Triangle) 
  (h1 : t.b^2 = t.a * t.c) (h2 : Real.cos t.B = 4/5) :
  1 / Real.tan t.A + 1 / Real.tan t.C = 5/3 :=
sorry

end NUMINAMATH_CALUDE_cosine_sum_range_inverse_tangent_sum_l1218_121870


namespace NUMINAMATH_CALUDE_son_score_calculation_l1218_121843

def father_score : ℕ := 48
def son_score_difference : ℕ := 8

theorem son_score_calculation (father_score : ℕ) (son_score_difference : ℕ) :
  father_score = 48 →
  son_score_difference = 8 →
  father_score / 2 - son_score_difference = 16 :=
by sorry

end NUMINAMATH_CALUDE_son_score_calculation_l1218_121843


namespace NUMINAMATH_CALUDE_jemma_grasshoppers_l1218_121858

/-- The number of grasshoppers Jemma saw on her African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found under the plant -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshoppers : total_grasshoppers = 31 := by
  sorry

end NUMINAMATH_CALUDE_jemma_grasshoppers_l1218_121858


namespace NUMINAMATH_CALUDE_hundred_million_composition_l1218_121845

-- Define the decimal counting system progression rate
def decimal_progression_rate : ℕ := 10

-- Define the units
def one_million : ℕ := 1000000
def ten_million : ℕ := 10000000
def hundred_million : ℕ := 100000000

-- Theorem statement
theorem hundred_million_composition :
  hundred_million = decimal_progression_rate * ten_million ∧
  hundred_million = (decimal_progression_rate * decimal_progression_rate) * one_million :=
by sorry

end NUMINAMATH_CALUDE_hundred_million_composition_l1218_121845


namespace NUMINAMATH_CALUDE_sarahs_bowling_score_l1218_121838

theorem sarahs_bowling_score (greg_score sarah_score : ℕ) : 
  sarah_score = greg_score + 60 →
  (sarah_score + greg_score) / 2 = 108 →
  sarah_score = 138 := by
sorry

end NUMINAMATH_CALUDE_sarahs_bowling_score_l1218_121838


namespace NUMINAMATH_CALUDE_digit_222_of_55_div_777_l1218_121854

/-- The decimal representation of a rational number -/
def decimal_representation (n d : ℕ) : ℕ → ℕ :=
  sorry

/-- The length of the repeating block in the decimal representation of a rational number -/
def repeating_block_length (n d : ℕ) : ℕ :=
  sorry

theorem digit_222_of_55_div_777 :
  decimal_representation 55 777 222 = 7 :=
sorry

end NUMINAMATH_CALUDE_digit_222_of_55_div_777_l1218_121854


namespace NUMINAMATH_CALUDE_seashell_sum_total_seashells_l1218_121831

theorem seashell_sum : Int → Int → Int → Int
  | sam, joan, alex => sam + joan + alex

theorem total_seashells (sam joan alex : Int) 
  (h1 : sam = 35) (h2 : joan = 18) (h3 : alex = 27) : 
  seashell_sum sam joan alex = 80 := by
  sorry

end NUMINAMATH_CALUDE_seashell_sum_total_seashells_l1218_121831


namespace NUMINAMATH_CALUDE_nonnegative_integer_representation_l1218_121873

theorem nonnegative_integer_representation (n : ℕ) : 
  ∃ (a b c : ℕ+), n = a^2 + b^2 - c^2 ∧ a < b ∧ b < c := by
  sorry

end NUMINAMATH_CALUDE_nonnegative_integer_representation_l1218_121873


namespace NUMINAMATH_CALUDE_sinusoidal_symmetry_center_l1218_121894

/-- Given a sinusoidal function with specific properties, prove that one of its symmetry centers has coordinates (-2π/3, 0) -/
theorem sinusoidal_symmetry_center 
  (f : ℝ → ℝ) 
  (ω φ : ℝ) 
  (h1 : ∀ x, f x = Real.sin (ω * x + φ))
  (h2 : ω > 0)
  (h3 : |φ| < π / 2)
  (h4 : ∀ x, f (x + 4 * π) = f x)
  (h5 : ∀ t, t > 0 → (∀ x, f (x + t) = f x) → t ≥ 4 * π)
  (h6 : f (π / 3) = 1) :
  ∃ k : ℤ, f (x + (-2 * π / 3)) = -f (-x + (-2 * π / 3)) := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_symmetry_center_l1218_121894


namespace NUMINAMATH_CALUDE_div_value_problem_l1218_121806

theorem div_value_problem (a b d : ℚ) 
  (h1 : a / b = 3) 
  (h2 : b / d = 2 / 5) : 
  d / a = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_div_value_problem_l1218_121806


namespace NUMINAMATH_CALUDE_z_magnitude_l1218_121830

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := sorry

-- State the theorem
theorem z_magnitude : 
  ((z - 2) * i = 1 + i) → Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_z_magnitude_l1218_121830


namespace NUMINAMATH_CALUDE_ratio_NBQ_ABQ_l1218_121825

-- Define the points
variable (A B C P Q N : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- BP and BQ divide ∠ABC into three equal parts
axiom divide_three_equal : angle A B P = angle P B Q ∧ angle P B Q = angle Q B C

-- BN bisects ∠QBP
axiom bisect_QBP : angle Q B N = angle N B P

-- Theorem to prove
theorem ratio_NBQ_ABQ : 
  (angle N B Q) / (angle A B Q) = 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_ratio_NBQ_ABQ_l1218_121825


namespace NUMINAMATH_CALUDE_ten_dollar_bill_count_l1218_121819

/-- Represents the number of bills of a certain denomination in a wallet. -/
structure BillCount where
  fives : Nat
  tens : Nat
  twenties : Nat

/-- Calculates the total amount in the wallet given the bill counts. -/
def totalAmount (bills : BillCount) : Nat :=
  5 * bills.fives + 10 * bills.tens + 20 * bills.twenties

/-- Theorem stating that given the conditions, there are 2 $10 bills in the wallet. -/
theorem ten_dollar_bill_count : ∃ (bills : BillCount), 
  bills.fives = 4 ∧ 
  bills.twenties = 3 ∧ 
  totalAmount bills = 100 ∧ 
  bills.tens = 2 := by
  sorry

end NUMINAMATH_CALUDE_ten_dollar_bill_count_l1218_121819


namespace NUMINAMATH_CALUDE_cafeteria_apples_l1218_121844

/-- The number of apples initially in the cafeteria -/
def initial_apples : ℕ := 50

/-- The number of oranges initially in the cafeteria -/
def initial_oranges : ℕ := 40

/-- The cost of an apple in dollars -/
def apple_cost : ℚ := 4/5

/-- The cost of an orange in dollars -/
def orange_cost : ℚ := 1/2

/-- The number of apples left after selling -/
def remaining_apples : ℕ := 10

/-- The number of oranges left after selling -/
def remaining_oranges : ℕ := 6

/-- The total earnings from selling apples and oranges in dollars -/
def total_earnings : ℚ := 49

theorem cafeteria_apples :
  apple_cost * (initial_apples - remaining_apples : ℚ) +
  orange_cost * (initial_oranges - remaining_oranges : ℚ) = total_earnings :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l1218_121844


namespace NUMINAMATH_CALUDE_trigonometric_ratio_equals_one_l1218_121828

theorem trigonometric_ratio_equals_one :
  (Real.cos (70 * π / 180) * Real.cos (10 * π / 180) + Real.cos (80 * π / 180) * Real.cos (20 * π / 180)) /
  (Real.cos (69 * π / 180) * Real.cos (9 * π / 180) + Real.cos (81 * π / 180) * Real.cos (21 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_equals_one_l1218_121828


namespace NUMINAMATH_CALUDE_base6_arithmetic_equality_l1218_121801

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ := sorry

theorem base6_arithmetic_equality :
  base10ToBase6 ((base6ToBase10 45321 - base6ToBase10 23454) + base6ToBase10 14553) = 45550 := by
  sorry

end NUMINAMATH_CALUDE_base6_arithmetic_equality_l1218_121801


namespace NUMINAMATH_CALUDE_ratio_ac_to_bd_l1218_121862

/-- Given points A, B, C, D, and E on a line in that order, with given distances between consecutive points,
    prove that the ratio of AC to BD is 7/6. -/
theorem ratio_ac_to_bd (A B C D E : ℝ) 
  (h_order : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_ab : B - A = 3)
  (h_bc : C - B = 4)
  (h_cd : D - C = 2)
  (h_de : E - D = 3) :
  (C - A) / (D - B) = 7 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ratio_ac_to_bd_l1218_121862


namespace NUMINAMATH_CALUDE_product_of_four_numbers_l1218_121834

theorem product_of_four_numbers (a b c d : ℝ) : 
  ((a + b + c + d) / 4 = 7.1) →
  (2.5 * a = b - 1.2) →
  (b - 1.2 = c + 4.8) →
  (c + 4.8 = 0.25 * d) →
  (a * b * c * d = 49.6) := by
sorry

end NUMINAMATH_CALUDE_product_of_four_numbers_l1218_121834


namespace NUMINAMATH_CALUDE_base7_4513_equals_1627_l1218_121827

/-- Converts a base-7 digit to its base-10 equivalent --/
def base7ToBase10Digit (d : ℕ) : ℕ := d

/-- Converts a list of base-7 digits to a base-10 number --/
def base7ToBase10 (digits : List ℕ) : ℕ :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 7^i) 0

theorem base7_4513_equals_1627 :
  base7ToBase10 [3, 1, 5, 4] = 1627 := by sorry

end NUMINAMATH_CALUDE_base7_4513_equals_1627_l1218_121827


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1218_121842

theorem purely_imaginary_complex_number (a : ℝ) : 
  let z : ℂ := (1 + a * Complex.I) * (1 - 2 * Complex.I)
  (z.re = 0 ∧ z.im ≠ 0) → a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l1218_121842


namespace NUMINAMATH_CALUDE_existence_of_uv_l1218_121836

theorem existence_of_uv (m n X : ℕ) (hm : X ≥ m) (hn : X ≥ n) :
  ∃ u v : ℤ,
    (|u| + |v| > 0) ∧
    (|u| ≤ Real.sqrt X) ∧
    (|v| ≤ Real.sqrt X) ∧
    (0 ≤ m * u + n * v) ∧
    (m * u + n * v ≤ 2 * Real.sqrt X) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_uv_l1218_121836


namespace NUMINAMATH_CALUDE_decryption_works_l1218_121846

-- Define the Russian alphabet (excluding 'ё')
def russian_alphabet : List Char := ['а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']

-- Define the encryption steps
def swap_adjacent (s : String) : String := sorry

def shift_right (s : String) (n : Nat) : String := sorry

def reverse_string (s : String) : String := sorry

-- Define the decryption steps
def shift_left (s : String) (n : Nat) : String := sorry

-- Define the full encryption and decryption processes
def encrypt (s : String) : String :=
  reverse_string (shift_right (swap_adjacent s) 2)

def decrypt (s : String) : String :=
  swap_adjacent (shift_left (reverse_string s) 2)

-- Theorem to prove
theorem decryption_works (encrypted : String) (decrypted : String) :
  encrypted = "врпвл терпраиэ вйзгцфпз" ∧ 
  decrypted = "нефте базы южного района" →
  decrypt encrypted = decrypted := by sorry

end NUMINAMATH_CALUDE_decryption_works_l1218_121846


namespace NUMINAMATH_CALUDE_cycle_gain_percent_l1218_121876

theorem cycle_gain_percent (cost_price selling_price : ℝ) : 
  cost_price = 675 →
  selling_price = 1080 →
  ((selling_price - cost_price) / cost_price) * 100 = 60 := by
sorry

end NUMINAMATH_CALUDE_cycle_gain_percent_l1218_121876


namespace NUMINAMATH_CALUDE_cricket_team_handedness_l1218_121851

theorem cricket_team_handedness (total_players : ℕ) (throwers : ℕ) (right_handed : ℕ) 
  (h1 : total_players = 67)
  (h2 : throwers = 37)
  (h3 : right_handed = 57)
  (h4 : throwers ≤ right_handed) :
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_handedness_l1218_121851


namespace NUMINAMATH_CALUDE_part1_part2_l1218_121852

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition c - b = 2b cos A -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.c - t.b = 2 * t.b * Real.cos t.A

theorem part1 (t : Triangle) (h : satisfiesCondition t) 
    (ha : t.a = 2 * Real.sqrt 6) (hb : t.b = 3) : 
  t.c = 5 := by
  sorry

theorem part2 (t : Triangle) (h : satisfiesCondition t) 
    (hc : t.C = Real.pi / 2) : 
  t.B = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_part1_part2_l1218_121852


namespace NUMINAMATH_CALUDE_percentage_difference_l1218_121841

theorem percentage_difference (A B C y : ℝ) : 
  A = B + C → 
  B > C → 
  C > 0 → 
  B = C * (1 + y / 100) → 
  y = 100 * ((B - C) / C) :=
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l1218_121841


namespace NUMINAMATH_CALUDE_banana_arrangements_l1218_121857

theorem banana_arrangements :
  let total_letters : ℕ := 6
  let a_count : ℕ := 3
  let n_count : ℕ := 2
  let b_count : ℕ := 1
  (total_letters! / (a_count! * n_count! * b_count!)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_banana_arrangements_l1218_121857


namespace NUMINAMATH_CALUDE_base_ten_to_five_235_l1218_121816

/-- Converts a number from base 10 to base 5 -/
def toBaseFive (n : ℕ) : List ℕ :=
  sorry

theorem base_ten_to_five_235 :
  toBaseFive 235 = [1, 4, 2, 0] :=
sorry

end NUMINAMATH_CALUDE_base_ten_to_five_235_l1218_121816


namespace NUMINAMATH_CALUDE_bike_rental_problem_l1218_121880

/-- Calculates the number of hours a bike was rented given the total amount paid,
    the initial charge, and the hourly rate. -/
def rental_hours (total_paid : ℚ) (initial_charge : ℚ) (hourly_rate : ℚ) : ℚ :=
  (total_paid - initial_charge) / hourly_rate

/-- Proves that given the specific rental conditions and total payment,
    the number of hours rented is 9. -/
theorem bike_rental_problem :
  let total_paid : ℚ := 80
  let initial_charge : ℚ := 17
  let hourly_rate : ℚ := 7
  rental_hours total_paid initial_charge hourly_rate = 9 := by
  sorry


end NUMINAMATH_CALUDE_bike_rental_problem_l1218_121880


namespace NUMINAMATH_CALUDE_calum_disco_ball_spending_l1218_121802

/-- Represents the problem of calculating the maximum amount Calum can spend on each disco ball. -/
theorem calum_disco_ball_spending (
  disco_ball_count : ℕ)
  (food_box_count : ℕ)
  (decoration_set_count : ℕ)
  (food_box_cost : ℚ)
  (decoration_set_cost : ℚ)
  (total_budget : ℚ)
  (disco_ball_budget_percentage : ℚ)
  (h1 : disco_ball_count = 4)
  (h2 : food_box_count = 10)
  (h3 : decoration_set_count = 20)
  (h4 : food_box_cost = 25)
  (h5 : decoration_set_cost = 10)
  (h6 : total_budget = 600)
  (h7 : disco_ball_budget_percentage = 0.3)
  : (total_budget * disco_ball_budget_percentage) / disco_ball_count = 45 := by
  sorry

end NUMINAMATH_CALUDE_calum_disco_ball_spending_l1218_121802


namespace NUMINAMATH_CALUDE_billiard_path_to_top_left_l1218_121849

/-- Represents a point in a 2D lattice -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Represents a rectangular lattice -/
structure RectangularLattice where
  width : ℕ
  height : ℕ

def billiardTable : RectangularLattice := { width := 1965, height := 26 }

/-- Checks if a point is on the top edge of the lattice -/
def isTopEdge (l : RectangularLattice) (p : LatticePoint) : Prop :=
  p.x = 0 ∧ p.y = l.height

/-- Represents a line with slope 1 starting from (0, 0) -/
def slopeLine (x : ℤ) : LatticePoint :=
  { x := x, y := x }

theorem billiard_path_to_top_left :
  ∃ (n : ℕ), isTopEdge billiardTable (slopeLine (n * billiardTable.width)) := by
  sorry

end NUMINAMATH_CALUDE_billiard_path_to_top_left_l1218_121849


namespace NUMINAMATH_CALUDE_two_books_from_shelves_l1218_121892

/-- The number of ways to choose two books of different subjects -/
def choose_two_books (chinese : ℕ) (math : ℕ) (english : ℕ) : ℕ :=
  chinese * math + chinese * english + math * english

/-- Theorem stating that choosing two books of different subjects from the given shelves results in 242 ways -/
theorem two_books_from_shelves :
  choose_two_books 10 9 8 = 242 := by
  sorry

end NUMINAMATH_CALUDE_two_books_from_shelves_l1218_121892


namespace NUMINAMATH_CALUDE_rectangle_division_l1218_121850

theorem rectangle_division (a b c d : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0)
  (area1 : a * b = 18) (area2 : a * c = 27) (area3 : b * d = 12) :
  c * d = 93 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_division_l1218_121850


namespace NUMINAMATH_CALUDE_garlic_cloves_left_is_600_l1218_121820

/-- The number of garlic cloves Maria has left after using some for a feast -/
def garlic_cloves_left : ℕ :=
  let kitchen_initial := 750
  let pantry_initial := 450
  let basement_initial := 300
  let kitchen_used := 500
  let pantry_used := 230
  let basement_used := 170
  (kitchen_initial - kitchen_used) + (pantry_initial - pantry_used) + (basement_initial - basement_used)

theorem garlic_cloves_left_is_600 : garlic_cloves_left = 600 := by
  sorry

end NUMINAMATH_CALUDE_garlic_cloves_left_is_600_l1218_121820


namespace NUMINAMATH_CALUDE_johns_initial_money_l1218_121803

theorem johns_initial_money (M : ℝ) : 
  (M > 0) →
  ((1 - 1/5) * M * (1 - 3/4) = 4) →
  (M = 20) := by
sorry

end NUMINAMATH_CALUDE_johns_initial_money_l1218_121803


namespace NUMINAMATH_CALUDE_bookcase_length_inches_l1218_121882

/-- Conversion factor from feet to inches -/
def inches_per_foot : ℕ := 12

/-- Length of the bookcase in feet -/
def bookcase_length_feet : ℕ := 4

/-- Theorem stating that a 4-foot bookcase is 48 inches long -/
theorem bookcase_length_inches : 
  bookcase_length_feet * inches_per_foot = 48 := by
  sorry

end NUMINAMATH_CALUDE_bookcase_length_inches_l1218_121882


namespace NUMINAMATH_CALUDE_input_for_output_nine_l1218_121868

theorem input_for_output_nine (x : ℝ) (y : ℝ) : 
  (x < 0 → y = (x + 1)^2) ∧
  (x ≥ 0 → y = (x - 1)^2) ∧
  (y = 9) →
  (x = -4 ∨ x = 4) :=
by sorry

end NUMINAMATH_CALUDE_input_for_output_nine_l1218_121868


namespace NUMINAMATH_CALUDE_no_triple_perfect_squares_l1218_121805

theorem no_triple_perfect_squares (n : ℕ+) : 
  ¬(∃ a b c : ℕ, (2 * n.val^2 + 1 = a^2) ∧ (3 * n.val^2 + 1 = b^2) ∧ (6 * n.val^2 + 1 = c^2)) :=
by sorry

end NUMINAMATH_CALUDE_no_triple_perfect_squares_l1218_121805


namespace NUMINAMATH_CALUDE_sunday_calorie_intake_theorem_l1218_121895

/-- Calculates John's calorie intake for Sunday given his meal structure and calorie content --/
def sunday_calorie_intake (breakfast_calories : ℝ) (morning_snack_addition : ℝ) 
  (lunch_percentage : ℝ) (afternoon_snack_reduction : ℝ) (dinner_multiplier : ℝ) 
  (energy_drink_calories : ℝ) : ℝ :=
  let lunch_calories := breakfast_calories * (1 + lunch_percentage)
  let afternoon_snack_calories := lunch_calories * (1 - afternoon_snack_reduction)
  let dinner_calories := lunch_calories * dinner_multiplier
  let weekday_calories := breakfast_calories + (breakfast_calories + morning_snack_addition) + 
                          lunch_calories + afternoon_snack_calories + dinner_calories
  let energy_drinks_calories := 2 * energy_drink_calories
  weekday_calories + energy_drinks_calories

theorem sunday_calorie_intake_theorem :
  sunday_calorie_intake 500 150 0.25 0.30 2 220 = 3402.5 := by
  sorry

end NUMINAMATH_CALUDE_sunday_calorie_intake_theorem_l1218_121895


namespace NUMINAMATH_CALUDE_vector_difference_magnitude_l1218_121829

theorem vector_difference_magnitude : ∃ x : ℝ,
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![x, 4]
  (∃ k : ℝ, a = k • b) →
  Real.sqrt ((a 0 - b 0)^2 + (a 1 - b 1)^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_difference_magnitude_l1218_121829


namespace NUMINAMATH_CALUDE_female_leader_probability_l1218_121890

theorem female_leader_probability (female_count male_count : ℕ) 
  (h1 : female_count = 4) 
  (h2 : male_count = 6) : 
  (female_count : ℚ) / (female_count + male_count) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_female_leader_probability_l1218_121890


namespace NUMINAMATH_CALUDE_max_value_on_ellipse_l1218_121874

theorem max_value_on_ellipse :
  ∃ (max : ℝ), max = 2 * Real.sqrt 10 ∧
  (∀ x y : ℝ, x^2/9 + y^2/4 = 1 → 2*x - y ≤ max) ∧
  (∃ x y : ℝ, x^2/9 + y^2/4 = 1 ∧ 2*x - y = max) := by
sorry

end NUMINAMATH_CALUDE_max_value_on_ellipse_l1218_121874


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1218_121853

theorem min_value_of_expression (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : b < 0) 
  (h3 : a - b = 5) : 
  ∃ (m : ℝ), m = 1/2 ∧ ∀ x, x = 1/(a+1) + 1/(2-b) → x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1218_121853


namespace NUMINAMATH_CALUDE_trig_ratio_equality_l1218_121865

theorem trig_ratio_equality (α : Real) (h : Real.tan α = 2 * Real.tan (π / 5)) :
  Real.cos (α - 3 * π / 10) / Real.sin (α - π / 5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_ratio_equality_l1218_121865


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1218_121893

theorem expression_simplification_and_evaluation :
  let x : ℚ := 4
  ((1 / (x + 2) + 1) / ((x^2 + 6*x + 9) / (x^2 - 4))) = 2/7 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l1218_121893


namespace NUMINAMATH_CALUDE_min_value_expression_l1218_121861

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (5 * z) / (2 * x + y) + (5 * x) / (y + 2 * z) + (2 * y) / (x + z) + (x + y + z) / (x * y + y * z + z * x) ≥ 9 ∧
  ((5 * z) / (2 * x + y) + (5 * x) / (y + 2 * z) + (2 * y) / (x + z) + (x + y + z) / (x * y + y * z + z * x) = 9 ↔ x = y ∧ y = z) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1218_121861


namespace NUMINAMATH_CALUDE_andy_cookies_l1218_121840

/-- Represents the number of cookies taken by each basketball team member -/
def basketballTeamCookies (n : ℕ) : ℕ := 2 * n - 1

/-- The sum of cookies taken by all basketball team members -/
def totalTeamCookies (teamSize : ℕ) : ℕ :=
  (teamSize * (basketballTeamCookies 1 + basketballTeamCookies teamSize)) / 2

theorem andy_cookies (initialCookies brotherCookies teamSize : ℕ) 
  (h1 : initialCookies = 72)
  (h2 : brotherCookies = 5)
  (h3 : teamSize = 8)
  (h4 : totalTeamCookies teamSize + brotherCookies < initialCookies) :
  initialCookies - (totalTeamCookies teamSize + brotherCookies) = 3 := by
  sorry

end NUMINAMATH_CALUDE_andy_cookies_l1218_121840


namespace NUMINAMATH_CALUDE_lcm_of_9_12_15_l1218_121807

theorem lcm_of_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_9_12_15_l1218_121807


namespace NUMINAMATH_CALUDE_downstream_distance_is_16_l1218_121889

/-- Represents the swimming scenario with given conditions -/
structure SwimmingScenario where
  upstream_distance : ℝ
  swim_time : ℝ
  still_water_speed : ℝ

/-- Calculates the downstream distance given a swimming scenario -/
def downstream_distance (s : SwimmingScenario) : ℝ :=
  sorry

/-- Theorem stating that under the given conditions, the downstream distance is 16 km -/
theorem downstream_distance_is_16 (s : SwimmingScenario) 
  (h1 : s.upstream_distance = 10)
  (h2 : s.swim_time = 2)
  (h3 : s.still_water_speed = 6.5) :
  downstream_distance s = 16 := by
  sorry

end NUMINAMATH_CALUDE_downstream_distance_is_16_l1218_121889


namespace NUMINAMATH_CALUDE_chocolate_eggs_weight_l1218_121859

theorem chocolate_eggs_weight (total_eggs : ℕ) (egg_weight : ℕ) (num_boxes : ℕ) (discarded_boxes : ℕ) : 
  total_eggs = 12 →
  egg_weight = 10 →
  num_boxes = 4 →
  discarded_boxes = 1 →
  (total_eggs / num_boxes) * egg_weight * (num_boxes - discarded_boxes) = 90 := by
sorry

end NUMINAMATH_CALUDE_chocolate_eggs_weight_l1218_121859


namespace NUMINAMATH_CALUDE_sum_five_consecutive_integers_l1218_121804

/-- Given a sequence of five consecutive integers with middle number m,
    prove that their sum is equal to 5m. -/
theorem sum_five_consecutive_integers (m : ℤ) : 
  (m - 2) + (m - 1) + m + (m + 1) + (m + 2) = 5 * m := by
  sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_integers_l1218_121804


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_squares_formula_l1218_121888

/-- The arithmetic mean of the squares of the first n positive integers -/
def arithmetic_mean_of_squares (n : ℕ+) : ℚ :=
  (↑n.val * (↑n.val + 1) * (2 * ↑n.val + 1)) / (6 * ↑n.val)

theorem arithmetic_mean_of_squares_formula (n : ℕ+) :
  arithmetic_mean_of_squares n = ((↑n.val + 1) * (2 * ↑n.val + 1)) / 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_squares_formula_l1218_121888


namespace NUMINAMATH_CALUDE_parabola_directrix_l1218_121887

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the directrix
def directrix (x : ℝ) : Prop := x = -1

-- Theorem statement
theorem parabola_directrix :
  ∀ (x y : ℝ), parabola x y → directrix (x - 2) :=
sorry

end NUMINAMATH_CALUDE_parabola_directrix_l1218_121887


namespace NUMINAMATH_CALUDE_perpendicular_vectors_parallel_vectors_l1218_121885

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 0)
def b : ℝ × ℝ := (-2, 1)

-- Define dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define vector subtraction
def vec_sub (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 - w.1, v.2 - w.2)

-- Define scalar multiplication
def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)

-- Define vector addition
def vec_add (v w : ℝ × ℝ) : ℝ × ℝ := (v.1 + w.1, v.2 + w.2)

-- Theorem 1: k*a - b is perpendicular to a + 3*b when k = -13/5
theorem perpendicular_vectors (k : ℝ) : 
  dot_product (vec_sub (scalar_mul k a) b) (vec_add a (scalar_mul 3 b)) = 0 ↔ k = -13/5 := by
  sorry

-- Theorem 2: k*a - b is parallel to a + 3*b when k = -1/3
theorem parallel_vectors (k : ℝ) : 
  ∃ (t : ℝ), vec_sub (scalar_mul k a) b = scalar_mul t (vec_add a (scalar_mul 3 b)) ↔ k = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_parallel_vectors_l1218_121885


namespace NUMINAMATH_CALUDE_cube_ratio_equals_27_l1218_121832

theorem cube_ratio_equals_27 : (81000 : ℚ)^3 / (27000 : ℚ)^3 = 27 := by sorry

end NUMINAMATH_CALUDE_cube_ratio_equals_27_l1218_121832


namespace NUMINAMATH_CALUDE_complex_expression_equality_l1218_121809

theorem complex_expression_equality : 
  let a : ℂ := 3 + 2*I
  let b : ℂ := 2 - I
  3*a + 4*b = 17 + 2*I :=
by sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l1218_121809


namespace NUMINAMATH_CALUDE_base8_12345_to_decimal_l1218_121896

/-- Converts a base-8 number to its decimal (base-10) equivalent -/
def base8_to_decimal (d1 d2 d3 d4 d5 : ℕ) : ℕ :=
  d1 * 8^4 + d2 * 8^3 + d3 * 8^2 + d4 * 8^1 + d5 * 8^0

/-- The decimal representation of 12345 in base-8 is 5349 -/
theorem base8_12345_to_decimal :
  base8_to_decimal 1 2 3 4 5 = 5349 := by
  sorry

end NUMINAMATH_CALUDE_base8_12345_to_decimal_l1218_121896


namespace NUMINAMATH_CALUDE_black_ball_count_l1218_121837

theorem black_ball_count (total : ℕ) (red : ℕ) (white : ℕ) (black : ℕ)
  (h_total : total = red + white + black)
  (h_red_prob : (red : ℚ) / total = 42 / 100)
  (h_white_prob : (white : ℚ) / total = 28 / 100)
  (h_red_count : red = 21) :
  black = 15 := by
  sorry

end NUMINAMATH_CALUDE_black_ball_count_l1218_121837


namespace NUMINAMATH_CALUDE_product_sum_multiple_l1218_121818

theorem product_sum_multiple (a b m : ℤ) : 
  b = 7 → 
  b - a = 2 → 
  a * b = m * (a + b) + 11 → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_product_sum_multiple_l1218_121818


namespace NUMINAMATH_CALUDE_find_k_l1218_121848

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 - 5 * x + 6
def g (k : ℝ) (x : ℝ) : ℝ := 2 * x^2 - k * x + 1

-- State the theorem
theorem find_k : ∃ k : ℝ, f 5 - g k 5 = 30 ∧ k = -10 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l1218_121848


namespace NUMINAMATH_CALUDE_system_elimination_l1218_121815

theorem system_elimination (x y : ℝ) : 
  (x + y = 5 ∧ x - y = 2) → 
  (∃ k : ℝ, (x + y) + (x - y) = k ∧ y ≠ k / 2) ∧ 
  (∃ m : ℝ, (x + y) - (x - y) = m ∧ x ≠ m / 2) :=
by sorry

end NUMINAMATH_CALUDE_system_elimination_l1218_121815


namespace NUMINAMATH_CALUDE_min_distance_curve_line_l1218_121866

/-- The minimum distance between a curve and a line -/
theorem min_distance_curve_line (a m n : ℝ) (h1 : a > 0) :
  let b := -1/2 * a^2 + 3 * Real.log a
  let line := {p : ℝ × ℝ | p.2 = 2 * p.1 + 1/2}
  let Q := (m, n)
  Q ∈ line →
  ∃ (min_dist : ℝ), min_dist = 9/5 ∧
    ∀ (p : ℝ × ℝ), p ∈ line → (a - p.1)^2 + (b - p.2)^2 ≥ min_dist :=
by sorry

end NUMINAMATH_CALUDE_min_distance_curve_line_l1218_121866


namespace NUMINAMATH_CALUDE_ellipse_equation_l1218_121864

/-- Represents an ellipse with specific properties -/
structure Ellipse where
  /-- The sum of distances from any point on the ellipse to the two foci -/
  focal_distance_sum : ℝ
  /-- The eccentricity of the ellipse -/
  eccentricity : ℝ

/-- Theorem stating the equation of an ellipse with given properties -/
theorem ellipse_equation (e : Ellipse) 
  (h1 : e.focal_distance_sum = 6)
  (h2 : e.eccentricity = 1/3) :
  ∃ (x y : ℝ), x^2/9 + y^2/8 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1218_121864


namespace NUMINAMATH_CALUDE_min_value_theorem_l1218_121877

theorem min_value_theorem (a b c d : ℝ) :
  (|b + a^2 - 4 * Real.log a| + |2 * c - d + 2| = 0) →
  ∃ (min_value : ℝ), (∀ (a' b' c' d' : ℝ), 
    (|b' + a'^2 - 4 * Real.log a'| + |2 * c' - d' + 2| = 0) →
    ((a' - c')^2 + (b' - d')^2 ≥ min_value)) ∧
  min_value = 5 :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1218_121877


namespace NUMINAMATH_CALUDE_hyperbola_sum_l1218_121826

/-- Given a hyperbola with center (2, 0), one focus at (2, 8), and one vertex at (2, 5),
    prove that h + k + a + b = 7 + √39, where (h, k) is the center, a is the distance
    from the center to a vertex, and b is derived from b^2 = c^2 - a^2, with c being
    the distance from the center to a focus. -/
theorem hyperbola_sum (h k a b c : ℝ) : 
  h = 2 ∧ k = 0 ∧ a = 5 ∧ c = 8 ∧ b^2 = c^2 - a^2 →
  h + k + a + b = 7 + Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l1218_121826


namespace NUMINAMATH_CALUDE_cryptarithm_solution_l1218_121879

def is_valid_solution (A R K : Nat) : Prop :=
  A ≠ R ∧ A ≠ K ∧ R ≠ K ∧
  A < 10 ∧ R < 10 ∧ K < 10 ∧
  1000 * A + 100 * R + 10 * K + A +
  100 * R + 10 * K + A +
  10 * K + A +
  A = 2014

theorem cryptarithm_solution :
  ∀ A R K : Nat, is_valid_solution A R K → A = 1 ∧ R = 4 ∧ K = 7 := by
  sorry

end NUMINAMATH_CALUDE_cryptarithm_solution_l1218_121879


namespace NUMINAMATH_CALUDE_elena_operation_l1218_121883

theorem elena_operation (x : ℝ) : (((3 * x + 5) - 3) * 2) / 2 = 17 → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_elena_operation_l1218_121883


namespace NUMINAMATH_CALUDE_euler_family_mean_age_l1218_121835

def euler_family_ages : List ℕ := [8, 8, 12, 12, 10, 14]

theorem euler_family_mean_age :
  (euler_family_ages.sum : ℚ) / euler_family_ages.length = 32 / 3 := by
  sorry

end NUMINAMATH_CALUDE_euler_family_mean_age_l1218_121835


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l1218_121824

theorem system_of_equations_solution :
  (∀ p q : ℚ, p + q = 4 ∧ 2 * p - q = 5 → p = 3 ∧ q = 1) ∧
  (∀ v t : ℚ, 2 * v + t = 3 ∧ 3 * v - 2 * t = 3 → v = 9 / 7 ∧ t = 3 / 7) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l1218_121824


namespace NUMINAMATH_CALUDE_computers_produced_per_month_l1218_121810

/-- Represents the number of computers produced per 30-minute interval -/
def computers_per_interval : ℕ := 4

/-- Represents the number of days in a month -/
def days_per_month : ℕ := 28

/-- Represents the number of 30-minute intervals in a day -/
def intervals_per_day : ℕ := 48

/-- Calculates the total number of computers produced in a month -/
def computers_per_month : ℕ :=
  computers_per_interval * days_per_month * intervals_per_day

/-- Theorem stating that the number of computers produced per month is 5376 -/
theorem computers_produced_per_month :
  computers_per_month = 5376 := by sorry

end NUMINAMATH_CALUDE_computers_produced_per_month_l1218_121810


namespace NUMINAMATH_CALUDE_emily_walks_farther_l1218_121884

/-- The distance Troy walks to school (in meters) -/
def troy_distance : ℕ := 75

/-- The distance Emily walks to school (in meters) -/
def emily_distance : ℕ := 98

/-- The number of days -/
def days : ℕ := 5

/-- The additional distance Emily walks compared to Troy over the given number of days -/
def additional_distance : ℕ := 
  days * (2 * emily_distance - 2 * troy_distance)

theorem emily_walks_farther : additional_distance = 230 := by
  sorry

end NUMINAMATH_CALUDE_emily_walks_farther_l1218_121884


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1218_121800

/-- The diameter of a circle with area 64π cm² is 16 cm. -/
theorem circle_diameter_from_area :
  ∀ (r : ℝ), r > 0 → π * r^2 = 64 * π → 2 * r = 16 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1218_121800


namespace NUMINAMATH_CALUDE_root_sum_squared_plus_triple_plus_other_root_l1218_121856

theorem root_sum_squared_plus_triple_plus_other_root (α β : ℝ) : 
  α^2 + 2*α - 2024 = 0 → β^2 + 2*β - 2024 = 0 → α^2 + 3*α + β = 2022 :=
by sorry

end NUMINAMATH_CALUDE_root_sum_squared_plus_triple_plus_other_root_l1218_121856


namespace NUMINAMATH_CALUDE_x_squared_y_plus_xy_squared_l1218_121878

theorem x_squared_y_plus_xy_squared (x y : ℝ) :
  x = Real.sqrt 3 + Real.sqrt 2 →
  y = Real.sqrt 3 - Real.sqrt 2 →
  x^2 * y + x * y^2 = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_x_squared_y_plus_xy_squared_l1218_121878


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l1218_121881

theorem max_value_trig_expression :
  ∀ x : ℝ, 2 * Real.cos x + 3 * Real.sin x ≤ Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l1218_121881


namespace NUMINAMATH_CALUDE_log_xy_value_l1218_121812

-- Define a real-valued logarithm function
noncomputable def log : ℝ → ℝ := sorry

-- State the theorem
theorem log_xy_value (x y : ℝ) (h1 : log (x^2 * y^3) = 2) (h2 : log (x^3 * y^2) = 2) :
  log (x * y) = 4/5 := by sorry

end NUMINAMATH_CALUDE_log_xy_value_l1218_121812


namespace NUMINAMATH_CALUDE_root_square_condition_l1218_121860

theorem root_square_condition (q : ℝ) : 
  (∃ a b : ℝ, a^2 - 12*a + q = 0 ∧ b^2 - 12*b + q = 0 ∧ (a = b^2 ∨ b = a^2)) ↔ 
  (q = -64 ∨ q = 27) := by
sorry

end NUMINAMATH_CALUDE_root_square_condition_l1218_121860
