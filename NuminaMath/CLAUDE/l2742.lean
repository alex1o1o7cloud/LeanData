import Mathlib

namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l2742_274290

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The original parabola y = 2x^2 - 1 -/
def original_parabola : Parabola :=
  { a := 2, b := 0, c := -1 }

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h_shift : ℝ) (v_shift : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h_shift + p.b
    c := p.a * h_shift^2 - p.b * h_shift + p.c + v_shift }

theorem parabola_shift_theorem :
  let shifted := shift_parabola original_parabola 1 (-2)
  shifted.a = 2 ∧ shifted.b = 4 ∧ shifted.c = -3 := by sorry

end NUMINAMATH_CALUDE_parabola_shift_theorem_l2742_274290


namespace NUMINAMATH_CALUDE_cubic_derivative_problem_l2742_274246

/-- Given a cubic function f(x) = ax³ + bx² + 3 where b is a constant,
    if f'(1) = -5, then f'(2) = -4 -/
theorem cubic_derivative_problem (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x^2 + 3
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 2 * b * x
  f' 1 = -5 → f' 2 = -4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_derivative_problem_l2742_274246


namespace NUMINAMATH_CALUDE_rope_cutting_game_winner_l2742_274204

/-- Represents a player in the rope-cutting game -/
inductive Player : Type
| A : Player
| B : Player

/-- Determines if a number is a power of 3 -/
def isPowerOfThree (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 3^k

/-- Represents the rope-cutting game -/
def RopeCuttingGame (a b : ℕ) : Prop :=
  a > 1 ∧ b > 1

/-- Determines if a player has a winning strategy -/
def hasWinningStrategy (p : Player) (a b : ℕ) : Prop :=
  RopeCuttingGame a b →
    (p = Player.B ↔ (a = 2 ∧ b = 3) ∨ isPowerOfThree a)

/-- Main theorem: Player B has a winning strategy iff a = 2 and b = 3, or a is a power of 3 -/
theorem rope_cutting_game_winner (a b : ℕ) :
  RopeCuttingGame a b →
    hasWinningStrategy Player.B a b := by
  sorry

end NUMINAMATH_CALUDE_rope_cutting_game_winner_l2742_274204


namespace NUMINAMATH_CALUDE_heart_then_ten_probability_l2742_274239

-- Define the total number of cards in a standard deck
def total_cards : ℕ := 52

-- Define the number of hearts in a deck
def num_hearts : ℕ := 13

-- Define the number of 10s in a deck
def num_tens : ℕ := 4

-- Define the probability of the event
def prob_heart_then_ten : ℚ := 1 / total_cards

-- State the theorem
theorem heart_then_ten_probability :
  prob_heart_then_ten = (num_hearts * num_tens) / (total_cards * (total_cards - 1)) :=
sorry

end NUMINAMATH_CALUDE_heart_then_ten_probability_l2742_274239


namespace NUMINAMATH_CALUDE_blue_faces_cube_l2742_274215

theorem blue_faces_cube (n : ℕ) : n > 0 →
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 := by sorry

end NUMINAMATH_CALUDE_blue_faces_cube_l2742_274215


namespace NUMINAMATH_CALUDE_insufficient_info_to_determine_sum_l2742_274232

/-- Represents a class with boys, girls, and a teacher -/
structure Classroom where
  numBoys : ℕ
  numGirls : ℕ
  avgAgeBoys : ℝ
  avgAgeGirls : ℝ
  avgAgeAll : ℝ
  teacherAge : ℕ

/-- The conditions given in the problem -/
def classroomConditions (c : Classroom) : Prop :=
  c.numBoys > 0 ∧
  c.numGirls > 0 ∧
  c.avgAgeBoys = c.avgAgeGirls ∧
  c.avgAgeGirls = c.avgAgeBoys ∧
  c.avgAgeAll = c.avgAgeBoys + c.avgAgeGirls ∧
  c.teacherAge = 42

/-- Theorem stating that the given conditions are insufficient to determine b + g -/
theorem insufficient_info_to_determine_sum (c : Classroom) 
  (h : classroomConditions c) : 
  ∃ (c1 c2 : Classroom), classroomConditions c1 ∧ classroomConditions c2 ∧ 
  c1.avgAgeBoys + c1.avgAgeGirls ≠ c2.avgAgeBoys + c2.avgAgeGirls :=
sorry

end NUMINAMATH_CALUDE_insufficient_info_to_determine_sum_l2742_274232


namespace NUMINAMATH_CALUDE_only_statement5_true_l2742_274261

-- Define the statements as functions
def statement1 (a b : ℝ) : Prop := b * (a + b) = b * a + b * b
def statement2 (x y : ℝ) : Prop := Real.log (x + y) = Real.log x + Real.log y
def statement3 (x y : ℝ) : Prop := (x + y)^2 = x^2 + y^2
def statement4 (a b : ℝ) : Prop := b^(a + b) = b^a + b^b
def statement5 (x y : ℝ) : Prop := x^2 / y^2 = (x / y)^2

-- Theorem stating that only statement5 is true for all real numbers
theorem only_statement5_true :
  (∀ x y : ℝ, statement5 x y) ∧
  (∃ a b : ℝ, ¬statement1 a b) ∧
  (∃ x y : ℝ, ¬statement2 x y) ∧
  (∃ x y : ℝ, ¬statement3 x y) ∧
  (∃ a b : ℝ, ¬statement4 a b) :=
sorry

end NUMINAMATH_CALUDE_only_statement5_true_l2742_274261


namespace NUMINAMATH_CALUDE_power_calculation_l2742_274209

theorem power_calculation : 2^24 / 16^3 * 2^4 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_power_calculation_l2742_274209


namespace NUMINAMATH_CALUDE_park_legs_count_l2742_274212

/-- Calculate the total number of legs for given numbers of dogs, cats, birds, and spiders -/
def totalLegs (dogs cats birds spiders : ℕ) : ℕ :=
  4 * dogs + 4 * cats + 2 * birds + 8 * spiders

/-- Theorem stating that the total number of legs for 109 dogs, 37 cats, 52 birds, and 19 spiders is 840 -/
theorem park_legs_count : totalLegs 109 37 52 19 = 840 := by
  sorry

end NUMINAMATH_CALUDE_park_legs_count_l2742_274212


namespace NUMINAMATH_CALUDE_largest_C_inequality_l2742_274226

theorem largest_C_inequality : 
  ∃ (C : ℝ), C = 17/4 ∧ 
  (∀ (x y : ℝ), y ≥ 4*x ∧ x > 0 → x^2 + y^2 ≥ C*x*y) ∧
  (∀ (C' : ℝ), C' > C → 
    ∃ (x y : ℝ), y ≥ 4*x ∧ x > 0 ∧ x^2 + y^2 < C'*x*y) :=
sorry

end NUMINAMATH_CALUDE_largest_C_inequality_l2742_274226


namespace NUMINAMATH_CALUDE_lana_extra_flowers_l2742_274207

/-- The number of extra flowers Lana picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Theorem stating that Lana picked 280 extra flowers -/
theorem lana_extra_flowers :
  extra_flowers 860 920 1500 = 280 := by
  sorry

end NUMINAMATH_CALUDE_lana_extra_flowers_l2742_274207


namespace NUMINAMATH_CALUDE_geometric_sequence_property_geometric_sequence_sum_l2742_274248

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) / a n = a (m + 1) / a m

/-- The property that if m + n = p + q, then a_m * a_n = a_p * a_q for a geometric sequence -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q :=
sorry

theorem geometric_sequence_sum (a : ℕ → ℝ) (h : GeometricSequence a) 
  (h_sum : a 4 + a 8 = -3) : 
  a 6 * (a 2 + 2 * a 6 + a 10) = 9 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_geometric_sequence_sum_l2742_274248


namespace NUMINAMATH_CALUDE_binary_1010101_is_85_l2742_274216

def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

theorem binary_1010101_is_85 :
  binary_to_decimal [true, false, true, false, true, false, true] = 85 := by
  sorry

end NUMINAMATH_CALUDE_binary_1010101_is_85_l2742_274216


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2742_274200

theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
    (4 * Real.pi * r^2 = 144 * Real.pi) →
    ((4 / 3) * Real.pi * r^3 = 288 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l2742_274200


namespace NUMINAMATH_CALUDE_negative_abs_negative_five_l2742_274288

theorem negative_abs_negative_five : -|-5| = -5 := by
  sorry

end NUMINAMATH_CALUDE_negative_abs_negative_five_l2742_274288


namespace NUMINAMATH_CALUDE_smallest_multiple_l2742_274251

theorem smallest_multiple (n : ℕ) : n = 714 ↔ 
  n > 0 ∧ 
  n % 17 = 0 ∧ 
  (n - 7) % 101 = 0 ∧ 
  ∀ m : ℕ, m > 0 → m % 17 = 0 → (m - 7) % 101 = 0 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_multiple_l2742_274251


namespace NUMINAMATH_CALUDE_nine_bulb_configurations_l2742_274225

def f : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 4
  | 3 => 8
  | n + 4 => f (n + 3) + f (n + 2) + f (n + 1) + f n

def circularConfigurations (n : ℕ) : ℕ :=
  f n - 3 * f 3 - 2 * f 2 - f 1

theorem nine_bulb_configurations :
  circularConfigurations 9 = 367 := by sorry

end NUMINAMATH_CALUDE_nine_bulb_configurations_l2742_274225


namespace NUMINAMATH_CALUDE_symmetry_and_evenness_l2742_274268

def symmetric_wrt_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (|x|) = f (-|x|)

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

theorem symmetry_and_evenness (f : ℝ → ℝ) :
  (even_function f → symmetric_wrt_y_axis f) ∧
  ∃ g : ℝ → ℝ, symmetric_wrt_y_axis g ∧ ¬even_function g :=
sorry

end NUMINAMATH_CALUDE_symmetry_and_evenness_l2742_274268


namespace NUMINAMATH_CALUDE_total_money_l2742_274295

/-- The total amount of money A, B, and C have between them is 700, given:
  * A and C together have 300
  * B and C together have 600
  * C has 200 -/
theorem total_money (A B C : ℕ) : 
  A + C = 300 → B + C = 600 → C = 200 → A + B + C = 700 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l2742_274295


namespace NUMINAMATH_CALUDE_lomonosov_card_puzzle_l2742_274272

theorem lomonosov_card_puzzle :
  ∃ (L O M N C B : ℕ),
    L ≠ O ∧ L ≠ M ∧ L ≠ N ∧ L ≠ C ∧ L ≠ B ∧
    O ≠ M ∧ O ≠ N ∧ O ≠ C ∧ O ≠ B ∧
    M ≠ N ∧ M ≠ C ∧ M ≠ B ∧
    N ≠ C ∧ N ≠ B ∧
    C ≠ B ∧
    L < 10 ∧ O < 10 ∧ M < 10 ∧ N < 10 ∧ C < 10 ∧ B < 10 ∧
    O < M ∧ O < C ∧
    L + O / M + O + N + O / C = 10 * O + B :=
by sorry

end NUMINAMATH_CALUDE_lomonosov_card_puzzle_l2742_274272


namespace NUMINAMATH_CALUDE_ingrids_tax_rate_l2742_274267

/-- Calculates the tax rate of the second person given the tax rate of the first person,
    both incomes, and their combined tax rate. -/
def calculate_second_tax_rate (first_tax_rate first_income second_income combined_tax_rate : ℚ) : ℚ :=
  let combined_income := first_income + second_income
  let total_tax := combined_tax_rate * combined_income
  let first_tax := first_tax_rate * first_income
  let second_tax := total_tax - first_tax
  second_tax / second_income

/-- Proves that given the specified conditions, Ingrid's tax rate is 40.00% -/
theorem ingrids_tax_rate :
  let john_tax_rate : ℚ := 30 / 100
  let john_income : ℚ := 56000
  let ingrid_income : ℚ := 74000
  let combined_tax_rate : ℚ := 3569 / 10000
  calculate_second_tax_rate john_tax_rate john_income ingrid_income combined_tax_rate = 40 / 100 :=
by sorry

end NUMINAMATH_CALUDE_ingrids_tax_rate_l2742_274267


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2742_274235

theorem greatest_of_three_consecutive_integers (x y z : ℤ) : 
  (y = x + 1) → (z = y + 1) → (x + y + z = 39) → 
  (max x (max y z) = 14) :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l2742_274235


namespace NUMINAMATH_CALUDE_sin_double_angle_plus_5pi_6_l2742_274208

theorem sin_double_angle_plus_5pi_6 (α : Real) 
  (h : Real.sin (α + π / 6) = 1 / 3) :
  Real.sin (2 * α + 5 * π / 6) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sin_double_angle_plus_5pi_6_l2742_274208


namespace NUMINAMATH_CALUDE_orange_seller_gain_percentage_l2742_274201

theorem orange_seller_gain_percentage
  (loss_rate : ℝ)
  (loss_quantity : ℝ)
  (gain_quantity : ℝ)
  (h_loss_rate : loss_rate = 0.04)
  (h_loss_quantity : loss_quantity = 16)
  (h_gain_quantity : gain_quantity = 12) :
  let cost_price := 1 / (1 - loss_rate)
  let gain_percentage := ((cost_price * gain_quantity) / (1 - loss_rate * cost_price) - 1) * 100
  gain_percentage = 28 := by
sorry

end NUMINAMATH_CALUDE_orange_seller_gain_percentage_l2742_274201


namespace NUMINAMATH_CALUDE_cell_chain_length_is_million_l2742_274273

/-- The length of a cell chain in nanometers -/
def cell_chain_length (cell_diameter : ℕ) (num_cells : ℕ) : ℕ :=
  cell_diameter * num_cells

/-- Theorem: The length of a cell chain is 10⁶ nanometers -/
theorem cell_chain_length_is_million :
  cell_chain_length 500 2000 = 1000000 := by
  sorry

end NUMINAMATH_CALUDE_cell_chain_length_is_million_l2742_274273


namespace NUMINAMATH_CALUDE_sumata_family_vacation_miles_l2742_274284

/-- Proves that given a 5-day vacation with a total of 1250 miles driven, the average miles driven per day is 250 miles. -/
theorem sumata_family_vacation_miles (total_miles : ℕ) (num_days : ℕ) (miles_per_day : ℕ) :
  total_miles = 1250 ∧ num_days = 5 ∧ miles_per_day = total_miles / num_days →
  miles_per_day = 250 :=
by sorry

end NUMINAMATH_CALUDE_sumata_family_vacation_miles_l2742_274284


namespace NUMINAMATH_CALUDE_simplify_expression_l2742_274278

theorem simplify_expression (x : ℝ) : (5 - 2*x) - (7 + 3*x) = -2 - 5*x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2742_274278


namespace NUMINAMATH_CALUDE_remaining_oranges_l2742_274247

def initial_oranges : ℕ := 60
def oranges_taken : ℕ := 35

theorem remaining_oranges :
  initial_oranges - oranges_taken = 25 := by
  sorry

end NUMINAMATH_CALUDE_remaining_oranges_l2742_274247


namespace NUMINAMATH_CALUDE_min_sum_given_log_sum_l2742_274289

theorem min_sum_given_log_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : Real.log a / Real.log 5 + Real.log b / Real.log 5 = 2) : 
  a + b ≥ 10 ∧ ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
    Real.log x / Real.log 5 + Real.log y / Real.log 5 = 2 ∧ x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_given_log_sum_l2742_274289


namespace NUMINAMATH_CALUDE_alfred_storage_period_l2742_274293

/-- Calculates the number of years Alfred stores maize -/
def years_storing_maize (
  monthly_storage : ℕ             -- tonnes stored per month
  ) (stolen : ℕ)                  -- tonnes stolen
  (donated : ℕ)                   -- tonnes donated
  (final_amount : ℕ)              -- final amount of maize in tonnes
  : ℕ :=
  (final_amount + stolen - donated) / (monthly_storage * 12)

/-- Theorem stating that Alfred stores maize for 2 years -/
theorem alfred_storage_period :
  years_storing_maize 1 5 8 27 = 2 := by
  sorry

end NUMINAMATH_CALUDE_alfred_storage_period_l2742_274293


namespace NUMINAMATH_CALUDE_last_digit_of_large_exponentiation_l2742_274254

/-- The last digit of a number -/
def lastDigit (n : ℕ) : ℕ := n % 10

/-- Exponentiation modulo 10 -/
def powMod10 (base exponent : ℕ) : ℕ :=
  (base ^ (exponent % 4)) % 10

theorem last_digit_of_large_exponentiation :
  lastDigit (powMod10 954950230952380948328708 470128749397540235934750230) = 4 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_large_exponentiation_l2742_274254


namespace NUMINAMATH_CALUDE_indefinite_integral_equality_l2742_274237

/-- The derivative of -(8/9) · √((1 + ∜(x³)) / ∜(x³))³ with respect to x
    is equal to (√(1 + ∜(x³))) / (x² · ⁸√x) for x > 0 -/
theorem indefinite_integral_equality (x : ℝ) (h : x > 0) :
  deriv (fun x => -(8/9) * Real.sqrt ((1 + x^(1/4)) / x^(1/4))^3) x =
  (Real.sqrt (1 + x^(3/4))) / (x^2 * x^(1/8)) :=
sorry

end NUMINAMATH_CALUDE_indefinite_integral_equality_l2742_274237


namespace NUMINAMATH_CALUDE_function_composition_l2742_274221

theorem function_composition (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_l2742_274221


namespace NUMINAMATH_CALUDE_centroid_property_l2742_274264

/-- Definition of a point in 2D space -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Definition of a triangle in 2D space -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- Calculate the centroid of a triangle -/
def centroid (t : Triangle) : Point2D :=
  { x := (t.A.x + t.B.x + t.C.x) / 3,
    y := (t.A.y + t.B.y + t.C.y) / 3 }

/-- The main theorem -/
theorem centroid_property :
  let t := Triangle.mk
    (Point2D.mk (-1) 4)
    (Point2D.mk 5 2)
    (Point2D.mk 3 10)
  let c := centroid t
  10 * c.x + c.y = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_centroid_property_l2742_274264


namespace NUMINAMATH_CALUDE_committee_formation_count_l2742_274236

/-- The number of ways to form a committee of size k from n eligible members. -/
def committee_count (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of members in the club. -/
def total_members : ℕ := 12

/-- The size of the committee to be formed. -/
def committee_size : ℕ := 5

/-- The number of ineligible members (Casey). -/
def ineligible_members : ℕ := 1

/-- The number of eligible members for the committee. -/
def eligible_members : ℕ := total_members - ineligible_members

theorem committee_formation_count :
  committee_count eligible_members committee_size = 462 := by
  sorry

end NUMINAMATH_CALUDE_committee_formation_count_l2742_274236


namespace NUMINAMATH_CALUDE_book_arrangement_proof_l2742_274260

/-- The number of distinct arrangements of books on a shelf. -/
def distinct_arrangements (total : ℕ) (identical : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial identical)

/-- Theorem: The number of distinct arrangements of 7 books with 3 identical copies is 840. -/
theorem book_arrangement_proof :
  distinct_arrangements 7 3 = 840 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_proof_l2742_274260


namespace NUMINAMATH_CALUDE_function_property_l2742_274242

def is_periodic (f : ℕ → ℕ) (period : ℕ) : Prop :=
  ∀ n, f (n + period) = f n

theorem function_property (f : ℕ → ℕ) 
  (h1 : ∀ n, f n ≠ 1)
  (h2 : ∀ n, f (n + 1) + f (n + 3) = f (n + 5) * f (n + 7) - 1375) :
  (is_periodic f 4) ∧ 
  (∀ n k, (f (n + 4 * k + 1) - 1) * (f (n + 4 * k + 3) - 1) = 1376) :=
sorry

end NUMINAMATH_CALUDE_function_property_l2742_274242


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2742_274233

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x + 3) * (3 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 32 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l2742_274233


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l2742_274258

theorem quadratic_form_ratio (x : ℝ) :
  let f := x^2 + 2600*x + 2600
  ∃ d e : ℝ, (∀ x, f = (x + d)^2 + e) ∧ e / d = -1298 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l2742_274258


namespace NUMINAMATH_CALUDE_cubic_not_decreasing_param_range_l2742_274259

/-- Given a cubic function that is not strictly decreasing, prove the range of its parameter. -/
theorem cubic_not_decreasing_param_range (b : ℝ) : 
  (∃ x y : ℝ, x < y ∧ (-x^3 + b*x^2 - (2*b + 3)*x + 2 - b) ≤ (-y^3 + b*y^2 - (2*b + 3)*y + 2 - b)) →
  (b < -1 ∨ b > 3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_not_decreasing_param_range_l2742_274259


namespace NUMINAMATH_CALUDE_original_price_calculation_l2742_274230

theorem original_price_calculation (total_sale : ℝ) (profit_rate : ℝ) (loss_rate : ℝ)
  (h1 : total_sale = 660)
  (h2 : profit_rate = 0.1)
  (h3 : loss_rate = 0.1) :
  ∃ (original_price : ℝ),
    original_price = total_sale / (1 + profit_rate) + total_sale / (1 - loss_rate) :=
by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l2742_274230


namespace NUMINAMATH_CALUDE_square_side_length_average_l2742_274234

theorem square_side_length_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 16) (h₂ : a₂ = 49) (h₃ : a₃ = 169) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l2742_274234


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2742_274210

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

theorem perpendicular_vectors (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) → k = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2742_274210


namespace NUMINAMATH_CALUDE_book_arrangement_count_l2742_274275

def num_arrangements (n_pushkin n_tarle : ℕ) : ℕ :=
  3 * (Nat.factorial 2) * (Nat.factorial 4)

theorem book_arrangement_count :
  num_arrangements 2 4 = 144 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_l2742_274275


namespace NUMINAMATH_CALUDE_tank_emptying_rate_l2742_274228

/-- Proves that given a tank of 30 cubic feet, with an inlet pipe rate of 5 cubic inches/min,
    one outlet pipe rate of 9 cubic inches/min, and a total emptying time of 4320 minutes
    when all pipes are open, the rate of the second outlet pipe is 8 cubic inches/min. -/
theorem tank_emptying_rate (tank_volume : ℝ) (inlet_rate : ℝ) (outlet_rate1 : ℝ)
    (emptying_time : ℝ) (inches_per_foot : ℝ) :
  tank_volume = 30 →
  inlet_rate = 5 →
  outlet_rate1 = 9 →
  emptying_time = 4320 →
  inches_per_foot = 12 →
  ∃ (outlet_rate2 : ℝ),
    outlet_rate2 = 8 ∧
    tank_volume * inches_per_foot^3 = (outlet_rate1 + outlet_rate2 - inlet_rate) * emptying_time :=
by sorry

end NUMINAMATH_CALUDE_tank_emptying_rate_l2742_274228


namespace NUMINAMATH_CALUDE_number_equality_l2742_274256

theorem number_equality : ∃ x : ℝ, (30 / 100) * x = (15 / 100) * 40 ∧ x = 20 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2742_274256


namespace NUMINAMATH_CALUDE_seats_between_17_and_39_l2742_274281

/-- The number of seats in the row -/
def total_seats : ℕ := 50

/-- The seat number of the first person -/
def seat1 : ℕ := 17

/-- The seat number of the second person -/
def seat2 : ℕ := 39

/-- The number of seats between two given seat numbers (exclusive) -/
def seats_between (a b : ℕ) : ℕ := 
  if a < b then b - a - 1 else a - b - 1

theorem seats_between_17_and_39 : 
  seats_between seat1 seat2 = 21 := by sorry

end NUMINAMATH_CALUDE_seats_between_17_and_39_l2742_274281


namespace NUMINAMATH_CALUDE_min_sum_squares_l2742_274213

theorem min_sum_squares (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 1) :
  ∃ (m : ℝ), ∀ (x y : ℝ), 0 < x → 0 < y → 2 * x + y = 1 → x^2 + y^2 ≥ m ∧ (∃ (u v : ℝ), 0 < u ∧ 0 < v ∧ 2 * u + v = 1 ∧ u^2 + v^2 = m) ∧ m = 1/5 :=
sorry

end NUMINAMATH_CALUDE_min_sum_squares_l2742_274213


namespace NUMINAMATH_CALUDE_house_price_calculation_house_price_proof_l2742_274211

theorem house_price_calculation (selling_price : ℝ) 
  (profit_rate : ℝ) (commission_rate : ℝ) : ℝ :=
  let original_price := selling_price / (1 + profit_rate - commission_rate)
  original_price

theorem house_price_proof :
  house_price_calculation 100000 0.2 0.05 = 100000 / 1.15 := by
  sorry

end NUMINAMATH_CALUDE_house_price_calculation_house_price_proof_l2742_274211


namespace NUMINAMATH_CALUDE_smallest_truck_shipments_l2742_274255

theorem smallest_truck_shipments (B : ℕ) : 
  B ≥ 120 → 
  B % 5 = 0 → 
  ∃ (T : ℕ), T ≠ 5 ∧ T > 1 ∧ B % T = 0 ∧ 
  ∀ (S : ℕ), S ≠ 5 → S > 1 → B % S = 0 → T ≤ S :=
by sorry

end NUMINAMATH_CALUDE_smallest_truck_shipments_l2742_274255


namespace NUMINAMATH_CALUDE_ellipse_equation_specific_l2742_274287

/-- Represents an ellipse in the Cartesian coordinate plane -/
structure Ellipse where
  center : ℝ × ℝ
  foci_axis : ℝ × ℝ
  minor_axis_length : ℝ
  eccentricity : ℝ

/-- The equation of an ellipse given its parameters -/
def ellipse_equation (e : Ellipse) (x y : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (x - e.center.1)^2 / a^2 + (y - e.center.2)^2 / b^2 = 1 ∧
    e.minor_axis_length = 2 * b ∧
    e.eccentricity = Real.sqrt (1 - b^2 / a^2)

theorem ellipse_equation_specific (e : Ellipse) :
  e.center = (0, 0) →
  e.foci_axis = (1, 0) →
  e.minor_axis_length = 2 →
  e.eccentricity = Real.sqrt 2 / 2 →
  ∀ (x y : ℝ), ellipse_equation e x y ↔ x^2 / 2 + y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_specific_l2742_274287


namespace NUMINAMATH_CALUDE_three_integers_with_difference_and_quotient_l2742_274282

theorem three_integers_with_difference_and_quotient :
  ∃ (a b c : ℤ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a = b - c ∧ b = c / a := by
  sorry

end NUMINAMATH_CALUDE_three_integers_with_difference_and_quotient_l2742_274282


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2742_274283

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | ∃ y, y = Real.log x}

theorem intersection_of_A_and_B : A ∩ B = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2742_274283


namespace NUMINAMATH_CALUDE_power_of_two_representation_l2742_274223

theorem power_of_two_representation (n : ℕ) (h : n ≥ 3) :
  ∃ (x y : ℤ), 2^n = 7*x^2 + y^2 ∧ Odd x ∧ Odd y :=
by sorry

end NUMINAMATH_CALUDE_power_of_two_representation_l2742_274223


namespace NUMINAMATH_CALUDE_calculation_result_l2742_274224

theorem calculation_result : 10 * 1.8 - 2 * 1.5 / 0.3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_calculation_result_l2742_274224


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2742_274285

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_3 = -1 and a_7 = -9, then a_5 = -3. -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_3 : a 3 = -1)
  (h_7 : a 7 = -9) :
  a 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2742_274285


namespace NUMINAMATH_CALUDE_fallen_cakes_ratio_l2742_274269

theorem fallen_cakes_ratio (total_cakes : ℕ) (destroyed_cakes : ℕ) : 
  total_cakes = 12 → 
  destroyed_cakes = 3 → 
  (2 * destroyed_cakes : ℚ) / total_cakes = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fallen_cakes_ratio_l2742_274269


namespace NUMINAMATH_CALUDE_supermarket_profit_analysis_l2742_274299

/-- Represents a supermarket area with its operating income and net profit percentages -/
structure Area where
  name : String
  operatingIncomePercentage : Float
  netProfitPercentage : Float

/-- Calculates the operating profit rate for an area given the total operating profit rate -/
def calculateOperatingProfitRate (area : Area) (totalOperatingProfitRate : Float) : Float :=
  (area.netProfitPercentage / area.operatingIncomePercentage) * totalOperatingProfitRate

theorem supermarket_profit_analysis 
  (freshArea dailyNecessitiesArea deliArea dairyArea otherArea : Area)
  (totalOperatingProfitRate : Float) :
  freshArea.name = "Fresh Area" →
  freshArea.operatingIncomePercentage = 48.6 →
  freshArea.netProfitPercentage = 65.8 →
  dailyNecessitiesArea.name = "Daily Necessities Area" →
  dailyNecessitiesArea.operatingIncomePercentage = 10.8 →
  dailyNecessitiesArea.netProfitPercentage = 20.2 →
  deliArea.name = "Deli Area" →
  deliArea.operatingIncomePercentage = 15.8 →
  deliArea.netProfitPercentage = -4.3 →
  dairyArea.name = "Dairy Area" →
  dairyArea.operatingIncomePercentage = 20.1 →
  dairyArea.netProfitPercentage = 16.5 →
  otherArea.name = "Other Area" →
  otherArea.operatingIncomePercentage = 4.7 →
  otherArea.netProfitPercentage = 1.8 →
  totalOperatingProfitRate = 32.5 →
  (freshArea.netProfitPercentage > 50) ∧ 
  (calculateOperatingProfitRate dailyNecessitiesArea totalOperatingProfitRate > 
   max (calculateOperatingProfitRate freshArea totalOperatingProfitRate)
       (max (calculateOperatingProfitRate deliArea totalOperatingProfitRate)
            (max (calculateOperatingProfitRate dairyArea totalOperatingProfitRate)
                 (calculateOperatingProfitRate otherArea totalOperatingProfitRate)))) ∧
  (calculateOperatingProfitRate freshArea totalOperatingProfitRate > 40) := by
  sorry

end NUMINAMATH_CALUDE_supermarket_profit_analysis_l2742_274299


namespace NUMINAMATH_CALUDE_ratio_expression_l2742_274276

theorem ratio_expression (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end NUMINAMATH_CALUDE_ratio_expression_l2742_274276


namespace NUMINAMATH_CALUDE_whitewashing_cost_is_16820_l2742_274203

/-- Calculates the cost of whitewashing a room with given dimensions and openings. -/
def whitewashing_cost (room_length room_width room_height : ℝ)
                      (door1_length door1_width : ℝ)
                      (door2_length door2_width : ℝ)
                      (window1_length window1_width : ℝ)
                      (window2_length window2_width : ℝ)
                      (window3_length window3_width : ℝ)
                      (window4_length window4_width : ℝ)
                      (window5_length window5_width : ℝ)
                      (cost_per_sqft : ℝ) : ℝ :=
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let openings_area := door1_length * door1_width + door2_length * door2_width +
                       window1_length * window1_width + window2_length * window2_width +
                       window3_length * window3_width + window4_length * window4_width +
                       window5_length * window5_width
  let whitewash_area := wall_area - openings_area
  whitewash_area * cost_per_sqft

/-- The cost of whitewashing the room with given dimensions and openings is Rs. 16820. -/
theorem whitewashing_cost_is_16820 :
  whitewashing_cost 40 20 15 7 4 5 3 5 4 4 3 3 3 4 2.5 6 4 10 = 16820 := by
  sorry


end NUMINAMATH_CALUDE_whitewashing_cost_is_16820_l2742_274203


namespace NUMINAMATH_CALUDE_completing_square_sum_l2742_274296

theorem completing_square_sum (x : ℝ) : 
  (∃ m n : ℝ, (x^2 - 6*x = 1) ↔ ((x - m)^2 = n)) → 
  (∃ m n : ℝ, (x^2 - 6*x = 1) ↔ ((x - m)^2 = n) ∧ m + n = 13) :=
by sorry

end NUMINAMATH_CALUDE_completing_square_sum_l2742_274296


namespace NUMINAMATH_CALUDE_union_of_sets_l2742_274274

theorem union_of_sets : 
  let A : Set ℕ := {1, 3}
  let B : Set ℕ := {3, 4}
  A ∪ B = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_union_of_sets_l2742_274274


namespace NUMINAMATH_CALUDE_square_cutting_l2742_274222

theorem square_cutting (a b : ℕ+) : 
  4 * a ^ 2 + 3 * b ^ 2 + 10 * a * b = 144 ↔ a = 2 ∧ b = 4 :=
by sorry

end NUMINAMATH_CALUDE_square_cutting_l2742_274222


namespace NUMINAMATH_CALUDE_solution_set_implies_a_and_b_l2742_274286

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 3

-- Define the theorem
theorem solution_set_implies_a_and_b :
  ∀ a b : ℝ, 
  (∀ x : ℝ, f a x > 0 ↔ b < x ∧ x < 1) →
  (a = -7 ∧ b = -3/7) := by
sorry

-- Note: The second part of the problem is not included in the Lean statement
-- as it relies on the solution of the first part, which should not be assumed
-- in the theorem statement according to the given criteria.

end NUMINAMATH_CALUDE_solution_set_implies_a_and_b_l2742_274286


namespace NUMINAMATH_CALUDE_product_72_difference_sum_l2742_274292

theorem product_72_difference_sum (P Q R S : ℕ+) : 
  P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ Q ≠ R ∧ Q ≠ S ∧ R ≠ S →
  P * Q = 72 →
  R * S = 72 →
  (P : ℤ) - (Q : ℤ) = (R : ℤ) + (S : ℤ) →
  P = 18 :=
by sorry

end NUMINAMATH_CALUDE_product_72_difference_sum_l2742_274292


namespace NUMINAMATH_CALUDE_five_students_left_l2742_274297

/-- Calculates the number of students who left during the year. -/
def students_who_left (initial_students new_students final_students : ℕ) : ℕ :=
  initial_students + new_students - final_students

/-- Proves that 5 students left during the year given the problem conditions. -/
theorem five_students_left : students_who_left 31 11 37 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_students_left_l2742_274297


namespace NUMINAMATH_CALUDE_committee_selection_count_l2742_274227

/-- The number of committee members -/
def total_members : ℕ := 5

/-- The number of roles to be filled -/
def roles_to_fill : ℕ := 3

/-- The number of members ineligible for the entertainment officer role -/
def ineligible_members : ℕ := 2

/-- The number of ways to select members for the given roles under the specified conditions -/
def selection_count : ℕ := 36

theorem committee_selection_count : 
  (total_members - ineligible_members) * 
  (total_members - 1) * 
  (total_members - 2) = selection_count :=
by sorry

end NUMINAMATH_CALUDE_committee_selection_count_l2742_274227


namespace NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2742_274271

/-- A function f is monotonic on ℝ if it is either non-decreasing or non-increasing on ℝ. -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, x ≤ y → f x ≤ f y) ∨ (∀ x y : ℝ, x ≤ y → f y ≤ f x)

/-- The function f(x) = x^3 + ax^2 + (a+6)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + (a+6)*x + 1

theorem monotonic_f_implies_a_range (a : ℝ) :
  Monotonic (f a) → -3 < a ∧ a < 6 := by
  sorry

#check monotonic_f_implies_a_range

end NUMINAMATH_CALUDE_monotonic_f_implies_a_range_l2742_274271


namespace NUMINAMATH_CALUDE_annika_age_l2742_274229

theorem annika_age (hans_age : ℕ) (annika_age : ℕ) : 
  hans_age = 8 →
  annika_age + 4 = 3 * (hans_age + 4) →
  annika_age = 32 := by
  sorry

end NUMINAMATH_CALUDE_annika_age_l2742_274229


namespace NUMINAMATH_CALUDE_product_of_integers_with_given_lcm_and_gcd_l2742_274219

theorem product_of_integers_with_given_lcm_and_gcd :
  ∀ x y : ℕ+, 
  x.val > 0 ∧ y.val > 0 →
  Nat.lcm x.val y.val = 60 →
  Nat.gcd x.val y.val = 5 →
  x.val * y.val = 300 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_with_given_lcm_and_gcd_l2742_274219


namespace NUMINAMATH_CALUDE_cab_driver_average_income_l2742_274263

def daily_incomes : List ℝ := [600, 250, 450, 400, 800]

theorem cab_driver_average_income :
  (daily_incomes.sum / daily_incomes.length : ℝ) = 500 := by
  sorry

end NUMINAMATH_CALUDE_cab_driver_average_income_l2742_274263


namespace NUMINAMATH_CALUDE_wrong_observation_value_l2742_274262

theorem wrong_observation_value (n : ℕ) (initial_mean correct_value new_mean : ℝ) 
  (h1 : n = 50)
  (h2 : initial_mean = 36)
  (h3 : correct_value = 60)
  (h4 : new_mean = 36.5) :
  ∃ wrong_value : ℝ,
    n * initial_mean - wrong_value + correct_value = n * new_mean ∧
    wrong_value = 35 := by
  sorry

end NUMINAMATH_CALUDE_wrong_observation_value_l2742_274262


namespace NUMINAMATH_CALUDE_divisors_of_1800_power_l2742_274257

theorem divisors_of_1800_power (n : Nat) : 
  (∃ (a b c : Nat), (a + 1) * (b + 1) * (c + 1) = 180 ∧
   n = 2^a * 3^b * 5^c ∧ n ∣ 1800^1800) ↔ n ∈ Finset.range 109 :=
by sorry

#check divisors_of_1800_power

end NUMINAMATH_CALUDE_divisors_of_1800_power_l2742_274257


namespace NUMINAMATH_CALUDE_locus_of_point_P_l2742_274244

/-- Given two rays OA and OB, and a point P inside the angle AOx, prove the equation of the locus of P and its domain --/
theorem locus_of_point_P (k : ℝ) (h_k : k > 0) :
  ∃ (f : ℝ → ℝ) (domain : Set ℝ),
    (∀ x y, (y = k * x ∧ x > 0) → (y = f x → x ∈ domain)) ∧
    (∀ x y, (y = -k * x ∧ x > 0) → (y = f x → x ∈ domain)) ∧
    (∀ x y, x ∈ domain → 0 < y ∧ y < k * x ∧ y < (1/k) * x) ∧
    (∀ x y, x ∈ domain → y = f x → y = Real.sqrt (x^2 - (1 + k^2))) ∧
    (0 < k ∧ k < 1 →
      domain = {x | Real.sqrt (k^2 + 1) < x ∧ x < Real.sqrt ((k^2 + 1)/(1 - k^2))}) ∧
    (k = 1 →
      domain = {x | Real.sqrt 2 < x}) ∧
    (k > 1 →
      domain = {x | Real.sqrt (k^2 + 1) < x ∧ x < k * Real.sqrt ((k^2 + 1)/(k^2 - 1))}) :=
by sorry

end NUMINAMATH_CALUDE_locus_of_point_P_l2742_274244


namespace NUMINAMATH_CALUDE_greatest_possible_median_l2742_274205

theorem greatest_possible_median (k m r s t : ℕ) : 
  k > 0 → m > 0 → r > 0 → s > 0 → t > 0 →
  (k + m + r + s + t) / 5 = 18 →
  k < m → m < r → r < s → s < t →
  t = 40 →
  r ≤ 23 ∧ ∃ (k' m' r' s' : ℕ), 
    k' > 0 ∧ m' > 0 ∧ r' > 0 ∧ s' > 0 ∧
    (k' + m' + r' + s' + 40) / 5 = 18 ∧
    k' < m' ∧ m' < r' ∧ r' < s' ∧ s' < 40 ∧
    r' = 23 := by
  sorry

end NUMINAMATH_CALUDE_greatest_possible_median_l2742_274205


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2742_274220

-- Define the curve
def f (x : ℝ) : ℝ := x^2 + 3*x + 1

-- Define the point on the tangent line
def point : ℝ × ℝ := (2, 5)

-- Define the equation of the tangent line
def tangent_line (x y : ℝ) : Prop := 7*x - y - 9 = 0

-- Theorem statement
theorem tangent_line_equation :
  ∃ (k : ℝ), 
    (∀ x, (deriv f) x = 2*x + 3) ∧ 
    (deriv f) point.1 = k ∧
    ∀ x y, tangent_line x y ↔ y - point.2 = k * (x - point.1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2742_274220


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2742_274240

theorem expand_and_simplify (y : ℝ) : 5 * (6 * y^2 - 3 * y + 2 - 4 * y^3) = -20 * y^3 + 30 * y^2 - 15 * y + 10 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2742_274240


namespace NUMINAMATH_CALUDE_similar_triangles_shortest_side_l2742_274245

theorem similar_triangles_shortest_side 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 = c^2) 
  (h2 : a = 24) 
  (h3 : c = 37) 
  (h4 : ∃ k, k > 0 ∧ k * c = 74) : 
  ∃ x, x > 0 ∧ x^2 = 793 ∧ 2 * x = min (2 * a) (2 * b) := by
sorry

end NUMINAMATH_CALUDE_similar_triangles_shortest_side_l2742_274245


namespace NUMINAMATH_CALUDE_max_integers_greater_than_26_l2742_274231

theorem max_integers_greater_than_26 (a b c d e : ℤ) :
  a + b + c + d + e = 3 →
  ∃ (count : ℕ), count ≤ 4 ∧
    count = (if a > 26 then 1 else 0) +
            (if b > 26 then 1 else 0) +
            (if c > 26 then 1 else 0) +
            (if d > 26 then 1 else 0) +
            (if e > 26 then 1 else 0) ∧
    ∀ (other_count : ℕ),
      other_count > count →
      ¬(∃ (a' b' c' d' e' : ℤ),
        a' + b' + c' + d' + e' = 3 ∧
        other_count = (if a' > 26 then 1 else 0) +
                      (if b' > 26 then 1 else 0) +
                      (if c' > 26 then 1 else 0) +
                      (if d' > 26 then 1 else 0) +
                      (if e' > 26 then 1 else 0)) :=
by sorry

end NUMINAMATH_CALUDE_max_integers_greater_than_26_l2742_274231


namespace NUMINAMATH_CALUDE_anne_heavier_than_douglas_l2742_274265

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference in weight between Anne and Douglas -/
def weight_difference : ℕ := anne_weight - douglas_weight

theorem anne_heavier_than_douglas : weight_difference = 15 := by
  sorry

end NUMINAMATH_CALUDE_anne_heavier_than_douglas_l2742_274265


namespace NUMINAMATH_CALUDE_system_solution_l2742_274294

theorem system_solution :
  ∃! (x y : ℚ), 4 * x - 3 * y = 2 ∧ 6 * x + 5 * y = 1 ∧ x = 13/38 ∧ y = -4/19 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2742_274294


namespace NUMINAMATH_CALUDE_altitude_sum_diff_values_l2742_274252

/-- A right triangle with sides 7, 24, and 25 units -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  right_angle : a^2 + b^2 = c^2
  side_a : a = 7
  side_b : b = 24
  side_c : c = 25

/-- The two longest altitudes in the right triangle -/
def longest_altitudes (t : RightTriangle) : ℝ × ℝ := (t.a, t.b)

/-- The sum and difference of the two longest altitudes -/
def altitude_sum_diff (t : RightTriangle) : ℝ × ℝ :=
  let (alt1, alt2) := longest_altitudes t
  (alt1 + alt2, |alt1 - alt2|)

theorem altitude_sum_diff_values (t : RightTriangle) :
  altitude_sum_diff t = (31, 17) := by sorry

end NUMINAMATH_CALUDE_altitude_sum_diff_values_l2742_274252


namespace NUMINAMATH_CALUDE_dots_on_abc_l2742_274298

/-- Represents a die face with a number of dots -/
structure DieFace :=
  (dots : Nat)
  (h : dots ≥ 1 ∧ dots ≤ 6)

/-- Represents a die with six faces -/
structure Die :=
  (faces : Fin 6 → DieFace)
  (opposite_sum : ∀ i : Fin 3, (faces i).dots + (faces (i + 3)).dots = 7)
  (all_different : ∀ i j : Fin 6, i ≠ j → (faces i).dots ≠ (faces j).dots)

/-- Represents the configuration of four glued dice -/
structure GluedDice :=
  (dice : Fin 4 → Die)
  (glued_faces_same : ∀ i j : Fin 4, i ≠ j → ∃ fi fj : Fin 6, 
    (dice i).faces fi = (dice j).faces fj)

/-- The main theorem stating the number of dots on faces A, B, and C -/
theorem dots_on_abc (gd : GluedDice) : 
  ∃ (a b c : DieFace), 
    a.dots = 2 ∧ b.dots = 2 ∧ c.dots = 6 ∧
    (∃ (i j k : Fin 4) (fi fj fk : Fin 6), 
      i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
      a = (gd.dice i).faces fi ∧
      b = (gd.dice j).faces fj ∧
      c = (gd.dice k).faces fk) :=
sorry

end NUMINAMATH_CALUDE_dots_on_abc_l2742_274298


namespace NUMINAMATH_CALUDE_runner_speed_impossibility_l2742_274250

/-- Proves that a runner cannot achieve an average speed of 12 mph over 24 miles
    when two-thirds of the distance has been run at 8 mph -/
theorem runner_speed_impossibility (total_distance : ℝ) (initial_speed : ℝ) (target_speed : ℝ) :
  total_distance = 24 →
  initial_speed = 8 →
  target_speed = 12 →
  (2 / 3 : ℝ) * total_distance / initial_speed = total_distance / target_speed :=
by sorry

end NUMINAMATH_CALUDE_runner_speed_impossibility_l2742_274250


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2742_274241

/-- Given a geometric sequence {a_n} where a_4 + a_8 = π, 
    prove that a_6(a_2 + 2a_6 + a_10) = π² -/
theorem geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- Definition of geometric sequence
  a 4 + a 8 = Real.pi →            -- Given condition
  a 6 * (a 2 + 2 * a 6 + a 10) = Real.pi ^ 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2742_274241


namespace NUMINAMATH_CALUDE_equal_positive_integers_l2742_274253

theorem equal_positive_integers (a b c n : ℕ+) 
  (eq1 : a^2 + b^2 = n * Nat.lcm a b + n^2)
  (eq2 : b^2 + c^2 = n * Nat.lcm b c + n^2)
  (eq3 : c^2 + a^2 = n * Nat.lcm c a + n^2) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equal_positive_integers_l2742_274253


namespace NUMINAMATH_CALUDE_dhoni_leftover_earnings_l2742_274214

theorem dhoni_leftover_earnings (rent dishwasher bills car groceries leftover : ℚ) : 
  rent = 20/100 →
  dishwasher = 15/100 →
  bills = 10/100 →
  car = 8/100 →
  groceries = 12/100 →
  leftover = 1 - (rent + dishwasher + bills + car + groceries) →
  leftover = 35/100 := by
sorry

end NUMINAMATH_CALUDE_dhoni_leftover_earnings_l2742_274214


namespace NUMINAMATH_CALUDE_correct_age_difference_l2742_274243

/-- The difference between Priya's father's age and Priya's age -/
def ageDifference (priyaAge fatherAge : ℕ) : ℕ :=
  fatherAge - priyaAge

theorem correct_age_difference :
  let priyaAge : ℕ := 11
  let fatherAge : ℕ := 42
  let futureSum : ℕ := 69
  let yearsLater : ℕ := 8
  (priyaAge + yearsLater) + (fatherAge + yearsLater) = futureSum →
  ageDifference priyaAge fatherAge = 31 := by
  sorry

end NUMINAMATH_CALUDE_correct_age_difference_l2742_274243


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_min_eccentricity_l2742_274270

/-- Given an ellipse and a hyperbola with the same foci, prove the minimum value of 3e₁² + e₂² -/
theorem ellipse_hyperbola_min_eccentricity (c : ℝ) (e₁ e₂ : ℝ) : 
  c > 0 → -- Foci are distinct points
  e₁ > 0 → -- Eccentricity of ellipse is positive
  e₂ > 0 → -- Eccentricity of hyperbola is positive
  e₁ * e₂ = 1 → -- Relationship between eccentricities due to shared foci and asymptote condition
  3 * e₁^2 + e₂^2 ≥ 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_min_eccentricity_l2742_274270


namespace NUMINAMATH_CALUDE_impossible_coin_probabilities_l2742_274249

theorem impossible_coin_probabilities :
  ¬∃ (p₁ p₂ : ℝ), 0 ≤ p₁ ∧ p₁ ≤ 1 ∧ 0 ≤ p₂ ∧ p₂ ≤ 1 ∧
    (1 - p₁) * (1 - p₂) = p₁ * p₂ ∧
    p₁ * p₂ = p₁ * (1 - p₂) + p₂ * (1 - p₁) :=
by sorry

end NUMINAMATH_CALUDE_impossible_coin_probabilities_l2742_274249


namespace NUMINAMATH_CALUDE_range_of_average_l2742_274202

theorem range_of_average (α β : ℝ) (h1 : -π/2 < α) (h2 : α < β) (h3 : β < π/2) :
  -π/2 < (α + β) / 2 ∧ (α + β) / 2 < 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_average_l2742_274202


namespace NUMINAMATH_CALUDE_no_common_points_l2742_274206

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 9
def circle2 (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem stating that there are no common points
theorem no_common_points : ¬ ∃ (x y : ℝ), circle1 x y ∧ circle2 x y :=
sorry

end NUMINAMATH_CALUDE_no_common_points_l2742_274206


namespace NUMINAMATH_CALUDE_hcl_effects_l2742_274266

-- Define the initial state of distilled water
structure DistilledWater :=
  (temp : ℝ)
  (pH : ℝ)
  (c_H : ℝ)
  (c_OH : ℝ)
  (Kw : ℝ)

-- Define the state after adding HCl
structure WaterWithHCl :=
  (temp : ℝ)
  (pH : ℝ)
  (c_H : ℝ)
  (c_OH : ℝ)
  (Kw : ℝ)
  (c_HCl : ℝ)

-- Define the theorem
theorem hcl_effects 
  (initial : DistilledWater) 
  (final : WaterWithHCl) 
  (h_temp : final.temp = initial.temp) 
  (h_HCl : final.c_HCl > 0) :
  (final.Kw = initial.Kw) ∧ 
  (final.pH < initial.pH) ∧ 
  (final.c_OH < initial.c_OH) ∧ 
  (final.c_H - final.c_HCl < initial.c_H) :=
sorry

end NUMINAMATH_CALUDE_hcl_effects_l2742_274266


namespace NUMINAMATH_CALUDE_scientific_notation_of_1680000_l2742_274238

theorem scientific_notation_of_1680000 : 
  ∃ (a : ℝ) (n : ℤ), 1680000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 1.68 ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1680000_l2742_274238


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2742_274217

theorem min_value_of_expression (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a : a > 0) 
  (h_inequality : x₁^2 - 4*a*x₁ + 3*a^2 < 0 ∧ x₂^2 - 4*a*x₂ + 3*a^2 < 0) :
  ∃ (m : ℝ), m = (4 * Real.sqrt 3) / 3 ∧ 
  ∀ (y₁ y₂ : ℝ), (y₁^2 - 4*a*y₁ + 3*a^2 < 0 ∧ y₂^2 - 4*a*y₂ + 3*a^2 < 0) → 
  (y₁ + y₂ + a / (y₁ * y₂)) ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2742_274217


namespace NUMINAMATH_CALUDE_triangle_area_change_l2742_274279

theorem triangle_area_change (h : ℝ) (b₁ b₂ : ℝ) (a₁ a₂ : ℝ) :
  h = 8 ∧ b₁ = 16 ∧ b₂ = 5 ∧
  a₁ = 1/2 * b₁ * h ∧
  a₂ = 1/2 * b₂ * h →
  a₁ = 64 ∧ a₂ = 20 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_change_l2742_274279


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2742_274291

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ 1 < x ∧ x < 2) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2742_274291


namespace NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l2742_274218

/-- Given a hyperbola with equation y²/9 - x²/b² = 1 and eccentricity 2,
    the distance from its focus to its asymptote is 3√3 -/
theorem hyperbola_focus_to_asymptote_distance
  (b : ℝ) -- Parameter b of the hyperbola
  (h1 : ∀ x y, y^2/9 - x^2/b^2 = 1) -- Equation of the hyperbola
  (h2 : 2 = (Real.sqrt (9 + b^2)) / 3) -- Eccentricity is 2
  : ∃ (focus : ℝ × ℝ) (asymptote : ℝ → ℝ),
    (∀ x, asymptote x = (Real.sqrt 3 / 3) * x ∨ asymptote x = -(Real.sqrt 3 / 3) * x) ∧
    Real.sqrt ((asymptote (focus.1) - focus.2)^2 / (1 + (Real.sqrt 3 / 3)^2)) = 3 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_to_asymptote_distance_l2742_274218


namespace NUMINAMATH_CALUDE_extremum_at_three_l2742_274280

noncomputable def f (x : ℝ) : ℝ := (x - 2) / Real.exp x

theorem extremum_at_three :
  ∀ x₀ : ℝ, (∀ x : ℝ, f x ≤ f x₀) ∨ (∀ x : ℝ, f x ≥ f x₀) → x₀ = 3 :=
by sorry

end NUMINAMATH_CALUDE_extremum_at_three_l2742_274280


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_third_l2742_274277

theorem arctan_sum_equals_pi_third (n : ℕ+) : 
  Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/n) = π/3 → n = 84 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_third_l2742_274277
