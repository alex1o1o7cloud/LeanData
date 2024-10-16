import Mathlib

namespace NUMINAMATH_CALUDE_tea_mixture_theorem_l606_60627

/-- Calculates the price of a tea mixture per kg -/
def tea_mixture_price (price1 price2 price3 : ℚ) (ratio1 ratio2 ratio3 : ℚ) : ℚ :=
  let total_cost := price1 * ratio1 + price2 * ratio2 + price3 * ratio3
  let total_quantity := ratio1 + ratio2 + ratio3
  total_cost / total_quantity

/-- Theorem stating the price of the specific tea mixture -/
theorem tea_mixture_theorem : 
  tea_mixture_price 126 135 177.5 1 1 2 = 154 := by
  sorry

#eval tea_mixture_price 126 135 177.5 1 1 2

end NUMINAMATH_CALUDE_tea_mixture_theorem_l606_60627


namespace NUMINAMATH_CALUDE_largest_number_with_digit_constraints_l606_60606

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def product_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * product_of_digits (n / 10)

theorem largest_number_with_digit_constraints : 
  ∀ n : ℕ, sum_of_digits n = 13 ∧ product_of_digits n = 36 → n ≤ 3322111 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_digit_constraints_l606_60606


namespace NUMINAMATH_CALUDE_alarm_system_probability_l606_60636

theorem alarm_system_probability (p : ℝ) (h1 : p = 0.4) :
  let prob_at_least_one_alerts := 1 - (1 - p) * (1 - p)
  prob_at_least_one_alerts = 0.64 := by
sorry

end NUMINAMATH_CALUDE_alarm_system_probability_l606_60636


namespace NUMINAMATH_CALUDE_focal_chord_length_l606_60651

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  parabola_eq : y^2 = 4*x

/-- Represents a line passing through the focal point of the parabola -/
structure FocalLine where
  A : ParabolaPoint
  B : ParabolaPoint
  sum_x : A.x + B.x = 6

/-- Theorem: The length of AB is 8 for the given conditions -/
theorem focal_chord_length (line : FocalLine) : 
  Real.sqrt ((line.B.x - line.A.x)^2 + (line.B.y - line.A.y)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_focal_chord_length_l606_60651


namespace NUMINAMATH_CALUDE_mrs_thomson_savings_l606_60620

theorem mrs_thomson_savings (incentive : ℝ) (food_fraction : ℝ) (clothes_fraction : ℝ) (saved_amount : ℝ)
  (h1 : incentive = 240)
  (h2 : food_fraction = 1/3)
  (h3 : clothes_fraction = 1/5)
  (h4 : saved_amount = 84) :
  let remaining := incentive - (food_fraction * incentive) - (clothes_fraction * incentive)
  saved_amount / remaining = 3/4 := by
sorry

end NUMINAMATH_CALUDE_mrs_thomson_savings_l606_60620


namespace NUMINAMATH_CALUDE_candy_mixture_problem_l606_60607

/-- Given two types of candy mixed to produce a specific mixture, 
    prove the amount of the second type of candy. -/
theorem candy_mixture_problem (X Y : ℝ) : 
  X + Y = 10 →
  3.50 * X + 4.30 * Y = 40 →
  Y = 6.25 := by
  sorry

end NUMINAMATH_CALUDE_candy_mixture_problem_l606_60607


namespace NUMINAMATH_CALUDE_company_production_days_l606_60604

/-- Given a company's production data, prove the number of past days. -/
theorem company_production_days (n : ℕ) : 
  (∀ (P : ℕ), P = 80 * n) →  -- Average daily production for past n days
  (∀ (new_total : ℕ), new_total = 80 * n + 220) →  -- Total including today's production
  (∀ (new_avg : ℝ), new_avg = (80 * n + 220) / (n + 1)) →  -- New average
  (new_avg = 95) →  -- New average is 95
  n = 8 := by sorry

end NUMINAMATH_CALUDE_company_production_days_l606_60604


namespace NUMINAMATH_CALUDE_cos_to_sin_shift_l606_60642

theorem cos_to_sin_shift (x : ℝ) : 
  Real.cos (2 * x + π / 3) = Real.sin (2 * (x + 5 * π / 12)) := by
  sorry

end NUMINAMATH_CALUDE_cos_to_sin_shift_l606_60642


namespace NUMINAMATH_CALUDE_rhombus_area_l606_60671

/-- The area of a rhombus with side length √117 and diagonals differing by 10 units is 72 square units. -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) (h₁ : s = Real.sqrt 117) (h₂ : d₂ - d₁ = 10) 
  (h₃ : s^2 = (d₁/2)^2 + (d₂/2)^2) : d₁ * d₂ / 2 = 72 := by
  sorry

#check rhombus_area

end NUMINAMATH_CALUDE_rhombus_area_l606_60671


namespace NUMINAMATH_CALUDE_x_value_l606_60668

theorem x_value (x y : ℚ) (h1 : x / y = 12 / 5) (h2 : y = 25) : x = 60 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l606_60668


namespace NUMINAMATH_CALUDE_set_operations_l606_60658

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_operations :
  (A ∩ B = {4}) ∧
  (A ∪ B = {1, 2, 4, 5, 6, 7, 8, 9, 10}) ∧
  ((U \ (A ∪ B)) = {3}) ∧
  ((U \ A) ∩ (U \ B) = {3}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l606_60658


namespace NUMINAMATH_CALUDE_no_valid_box_dimensions_l606_60638

theorem no_valid_box_dimensions :
  ¬∃ (a b c : ℕ), 
    (Prime a) ∧ (Prime b) ∧ (Prime c) ∧
    (a ≤ b) ∧ (b ≤ c) ∧
    (a * b * c = 2 * (a * b + b * c + a * c)) ∧
    (Prime (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_box_dimensions_l606_60638


namespace NUMINAMATH_CALUDE_cube_arrangement_exists_l606_60645

/-- Represents an arrangement of numbers on the edges of a cube -/
def CubeArrangement := Fin 12 → Fin 12

/-- Checks if the given arrangement is valid (all numbers from 1 to 12 are used exactly once) -/
def isValidArrangement (arr : CubeArrangement) : Prop :=
  ∀ i j : Fin 12, arr i = arr j → i = j

/-- Calculates the product of numbers on a face given by four edge indices -/
def faceProduct (arr : CubeArrangement) (e1 e2 e3 e4 : Fin 12) : ℕ :=
  (arr e1 + 1) * (arr e2 + 1) * (arr e3 + 1) * (arr e4 + 1)

/-- Indices of the edges on the top face -/
def topFace : Fin 12 × Fin 12 × Fin 12 × Fin 12 := (0, 1, 2, 3)

/-- Indices of the edges on the bottom face -/
def bottomFace : Fin 12 × Fin 12 × Fin 12 × Fin 12 := (4, 5, 6, 7)

/-- The main theorem stating that there exists a valid arrangement satisfying the required condition -/
theorem cube_arrangement_exists : ∃ (arr : CubeArrangement), 
  isValidArrangement arr ∧ 
  faceProduct arr topFace.1 topFace.2.1 topFace.2.2.1 topFace.2.2.2 = 
  faceProduct arr bottomFace.1 bottomFace.2.1 bottomFace.2.2.1 bottomFace.2.2.2 := by
  sorry

end NUMINAMATH_CALUDE_cube_arrangement_exists_l606_60645


namespace NUMINAMATH_CALUDE_smallest_non_factor_product_l606_60603

theorem smallest_non_factor_product (m n : ℕ) : 
  m ≠ n → 
  m > 0 → 
  n > 0 → 
  m ∣ 48 → 
  n ∣ 48 → 
  ¬(m * n ∣ 48) → 
  (∀ a b : ℕ, a ≠ b → a > 0 → b > 0 → a ∣ 48 → b ∣ 48 → ¬(a * b ∣ 48) → m * n ≤ a * b) →
  m * n = 18 :=
by sorry

end NUMINAMATH_CALUDE_smallest_non_factor_product_l606_60603


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l606_60654

theorem simplify_and_evaluate (x y : ℝ) (hx : x = -2) (hy : y = -1) :
  x + (1/3) * y^2 - 2 * (x - (1/3) * y^2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l606_60654


namespace NUMINAMATH_CALUDE_billy_strategy_l606_60663

def FencePainting (n : ℕ) :=
  ∃ (strategy : ℕ → ℕ),
    (∀ k, k ≤ n → strategy k ≤ n) ∧
    (∀ k, k < n → strategy k ≠ k) ∧
    (∀ k, k < n - 1 → strategy k ≠ strategy (k + 1))

theorem billy_strategy (n : ℕ) (h : n > 10) :
  FencePainting n ∧ (n % 2 = 1 → ∃ (winning_strategy : ℕ → ℕ), FencePainting n) :=
sorry

#check billy_strategy

end NUMINAMATH_CALUDE_billy_strategy_l606_60663


namespace NUMINAMATH_CALUDE_greatest_ratio_three_digit_number_l606_60617

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ ones ≤ 9

/-- The value of a three-digit number -/
def value (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- The sum of digits of a three-digit number -/
def digit_sum (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.ones

/-- The ratio of a three-digit number to the sum of its digits -/
def ratio (n : ThreeDigitNumber) : Rat :=
  (value n : Rat) / (digit_sum n : Rat)

theorem greatest_ratio_three_digit_number :
  (∀ n : ThreeDigitNumber, ratio n ≤ 100) ∧
  (∃ n : ThreeDigitNumber, ratio n = 100) :=
sorry

end NUMINAMATH_CALUDE_greatest_ratio_three_digit_number_l606_60617


namespace NUMINAMATH_CALUDE_absolute_value_equality_l606_60628

theorem absolute_value_equality (a b : ℝ) : 
  |a| = |b| → (a = b ∨ a = -b) := by sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l606_60628


namespace NUMINAMATH_CALUDE_area_PQRSTU_l606_60648

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a polygon with 6 vertices -/
structure Hexagon :=
  (P Q R S T U : Point)

/-- The given hexagonal polygon PQRSTU -/
def PQRSTU : Hexagon := sorry

/-- Point V, the intersection of extended lines QT and PU -/
def V : Point := sorry

/-- Length of side PQ -/
def PQ_length : ℝ := 8

/-- Length of side QR -/
def QR_length : ℝ := 10

/-- Length of side UT -/
def UT_length : ℝ := 7

/-- Length of side TU -/
def TU_length : ℝ := 3

/-- Predicate stating that PQRV is a rectangle -/
def is_rectangle_PQRV (h : Hexagon) (v : Point) : Prop := sorry

/-- Predicate stating that VUT is a rectangle -/
def is_rectangle_VUT (h : Hexagon) (v : Point) : Prop := sorry

/-- Function to calculate the area of a polygon -/
def area (h : Hexagon) : ℝ := sorry

/-- Theorem stating that the area of PQRSTU is 65 square units -/
theorem area_PQRSTU :
  is_rectangle_PQRV PQRSTU V →
  is_rectangle_VUT PQRSTU V →
  area PQRSTU = 65 := by sorry

end NUMINAMATH_CALUDE_area_PQRSTU_l606_60648


namespace NUMINAMATH_CALUDE_polynomial_factorization_l606_60687

theorem polynomial_factorization (x y z : ℝ) : 
  x^3 * (y - z) + y^3 * (z - x) + z^3 * (x - y) = (x + y + z) * (x - y) * (y - z) * (z - x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l606_60687


namespace NUMINAMATH_CALUDE_problem_solution_l606_60647

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (1 - a * x) / (x - 1) / Real.log (1 / 2) + x

theorem problem_solution :
  (∀ x, f a (-x) = -f a x) →
  (a = -1) ∧
  (∀ x y, 1 < x → x < y → f (-1) x < f (-1) y) ∧
  (∀ m, (∀ x, x ∈ Set.Icc 3 4 → f (-1) x > (1/2)^x + m) → m < 15/8) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l606_60647


namespace NUMINAMATH_CALUDE_min_additional_coins_alex_coin_distribution_l606_60624

theorem min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let min_required := (num_friends * (num_friends + 1)) / 2
  if min_required > initial_coins then
    min_required - initial_coins
  else
    0

theorem alex_coin_distribution : min_additional_coins 15 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_min_additional_coins_alex_coin_distribution_l606_60624


namespace NUMINAMATH_CALUDE_thomas_savings_years_l606_60623

/-- Represents the savings scenario for Thomas --/
structure SavingsScenario where
  allowance : ℕ  -- Weekly allowance in the first year
  wage : ℕ       -- Hourly wage from the second year
  hours : ℕ      -- Weekly work hours from the second year
  carCost : ℕ    -- Cost of the car
  spending : ℕ   -- Weekly spending
  remaining : ℕ  -- Amount still needed to buy the car

/-- Calculates the number of years Thomas has been saving --/
def yearsOfSaving (s : SavingsScenario) : ℕ :=
  2  -- This is the value we want to prove

/-- Theorem stating that Thomas has been saving for 2 years --/
theorem thomas_savings_years (s : SavingsScenario) 
  (h1 : s.allowance = 50)
  (h2 : s.wage = 9)
  (h3 : s.hours = 30)
  (h4 : s.carCost = 15000)
  (h5 : s.spending = 35)
  (h6 : s.remaining = 2000) :
  yearsOfSaving s = 2 := by
  sorry

#check thomas_savings_years

end NUMINAMATH_CALUDE_thomas_savings_years_l606_60623


namespace NUMINAMATH_CALUDE_rowing_round_trip_time_l606_60622

/-- Proves that the total time to row to a place and back is 1 hour, given the specified conditions -/
theorem rowing_round_trip_time
  (rowing_speed : ℝ)
  (current_speed : ℝ)
  (distance : ℝ)
  (h1 : rowing_speed = 5)
  (h2 : current_speed = 1)
  (h3 : distance = 2.4)
  : (distance / (rowing_speed + current_speed)) + (distance / (rowing_speed - current_speed)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_rowing_round_trip_time_l606_60622


namespace NUMINAMATH_CALUDE_remainder_problem_l606_60696

theorem remainder_problem (y : ℤ) (h : y % 264 = 42) : y % 22 = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l606_60696


namespace NUMINAMATH_CALUDE_maximum_marks_proof_l606_60681

/-- Given that a student needs 33% of total marks to pass, got 59 marks, and failed by 40 marks,
    prove that the maximum marks are 300. -/
theorem maximum_marks_proof (pass_percentage : Real) (obtained_marks : ℕ) (failing_margin : ℕ) :
  pass_percentage = 0.33 →
  obtained_marks = 59 →
  failing_margin = 40 →
  ∃ (max_marks : ℕ), max_marks = 300 ∧ pass_percentage * max_marks = obtained_marks + failing_margin :=
by sorry

end NUMINAMATH_CALUDE_maximum_marks_proof_l606_60681


namespace NUMINAMATH_CALUDE_negation_of_existential_quantifier_negation_of_inequality_l606_60694

theorem negation_of_existential_quantifier (p : ℝ → Prop) :
  (¬ ∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_inequality :
  (¬ ∃ x : ℝ, x^2 - 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 - 2*x + 2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existential_quantifier_negation_of_inequality_l606_60694


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l606_60631

theorem polygon_sides_from_angle_sum (sum_of_angles : ℕ) (h : sum_of_angles = 1260) :
  ∃ n : ℕ, n ≥ 3 ∧ (n - 2) * 180 = sum_of_angles ∧ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l606_60631


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l606_60602

theorem solution_set_quadratic_inequality :
  {x : ℝ | x^2 + x - 2 ≤ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l606_60602


namespace NUMINAMATH_CALUDE_solution_in_quadrant_I_l606_60680

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, 2 * x - y = 5 ∧ k * x + 2 * y = 4 ∧ x > 0 ∧ y > 0) ↔ -4 < k ∧ k < 8/5 := by
  sorry

end NUMINAMATH_CALUDE_solution_in_quadrant_I_l606_60680


namespace NUMINAMATH_CALUDE_power_product_equality_l606_60618

theorem power_product_equality : 3^3 * 2^2 * 7^2 * 11 = 58212 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l606_60618


namespace NUMINAMATH_CALUDE_no_prime_valued_polynomial_l606_60670

theorem no_prime_valued_polynomial : ¬ ∃ (P : ℕ → ℤ), (∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, k > n → P k = 0)) ∧ (∀ k : ℕ, Nat.Prime (P k).natAbs) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_valued_polynomial_l606_60670


namespace NUMINAMATH_CALUDE_simplify_expression_l606_60615

theorem simplify_expression (r : ℝ) : 90 * r - 44 * r = 46 * r := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l606_60615


namespace NUMINAMATH_CALUDE_scoops_left_is_16_l606_60605

/-- Represents the number of scoops in a carton of ice cream -/
def scoops_per_carton : ℕ := 10

/-- Represents the number of cartons Mary has -/
def marys_cartons : ℕ := 3

/-- Represents the number of scoops Ethan wants -/
def ethans_scoops : ℕ := 2

/-- Represents the number of people (Lucas, Danny, Connor) who want 2 scoops of chocolate each -/
def chocolate_lovers : ℕ := 3

/-- Represents the number of scoops each chocolate lover wants -/
def scoops_per_chocolate_lover : ℕ := 2

/-- Represents the number of scoops Olivia wants -/
def olivias_scoops : ℕ := 2

/-- Represents how many times more scoops Shannon wants compared to Olivia -/
def shannons_multiplier : ℕ := 2

/-- Theorem stating that the number of scoops left is 16 -/
theorem scoops_left_is_16 : 
  marys_cartons * scoops_per_carton - 
  (ethans_scoops + 
   chocolate_lovers * scoops_per_chocolate_lover + 
   olivias_scoops + 
   shannons_multiplier * olivias_scoops) = 16 := by
  sorry

end NUMINAMATH_CALUDE_scoops_left_is_16_l606_60605


namespace NUMINAMATH_CALUDE_bank_account_final_amount_l606_60611

/-- Calculates the final amount in a bank account given initial savings, withdrawal, and deposit. -/
def final_amount (initial_savings withdrawal : ℕ) : ℕ :=
  initial_savings - withdrawal + 2 * withdrawal

/-- Theorem stating that given the specific conditions, the final amount is $290. -/
theorem bank_account_final_amount : 
  final_amount 230 60 = 290 := by
  sorry

end NUMINAMATH_CALUDE_bank_account_final_amount_l606_60611


namespace NUMINAMATH_CALUDE_shower_has_three_walls_l606_60619

/-- Represents the properties of a shower with tiled walls -/
structure Shower :=
  (width_tiles : ℕ)
  (height_tiles : ℕ)
  (total_tiles : ℕ)

/-- Calculates the number of walls in a shower -/
def number_of_walls (s : Shower) : ℚ :=
  s.total_tiles / (s.width_tiles * s.height_tiles)

/-- Theorem: The shower has 3 walls -/
theorem shower_has_three_walls (s : Shower) 
  (h1 : s.width_tiles = 8)
  (h2 : s.height_tiles = 20)
  (h3 : s.total_tiles = 480) : 
  number_of_walls s = 3 := by
  sorry

#eval number_of_walls { width_tiles := 8, height_tiles := 20, total_tiles := 480 }

end NUMINAMATH_CALUDE_shower_has_three_walls_l606_60619


namespace NUMINAMATH_CALUDE_even_function_m_value_l606_60626

/-- A function f is even if f(x) = f(-x) for all x in its domain --/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The given function f(x) = x^2 + (m+2)x + 3 --/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + (m+2)*x + 3

theorem even_function_m_value :
  ∀ m : ℝ, IsEven (f m) → m = -2 := by
sorry

end NUMINAMATH_CALUDE_even_function_m_value_l606_60626


namespace NUMINAMATH_CALUDE_fraction_sum_simplification_l606_60685

theorem fraction_sum_simplification :
  18 / 462 + 35 / 77 = 38 / 77 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_simplification_l606_60685


namespace NUMINAMATH_CALUDE_mrs_lovely_class_l606_60632

/-- The number of students in Mrs. Lovely's class -/
def total_students : ℕ := 23

/-- The number of girls in the class -/
def girls : ℕ := 10

/-- The number of boys in the class -/
def boys : ℕ := girls + 3

/-- The total number of chocolates brought -/
def total_chocolates : ℕ := 500

/-- The number of chocolates left after distribution -/
def leftover_chocolates : ℕ := 10

theorem mrs_lovely_class :
  (girls * girls + boys * boys = total_chocolates - leftover_chocolates) ∧
  (girls + boys = total_students) := by
  sorry

end NUMINAMATH_CALUDE_mrs_lovely_class_l606_60632


namespace NUMINAMATH_CALUDE_base_number_proof_l606_60635

theorem base_number_proof (x n : ℕ) (h1 : 4 * x^(2*n) = 4^22) (h2 : n = 21) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_base_number_proof_l606_60635


namespace NUMINAMATH_CALUDE_problem_statement_l606_60601

theorem problem_statement (a b c d : ℤ) (x : ℝ) : 
  x = (a + b * Real.sqrt c) / d →
  (7 * x / 8) + 2 = 4 / x →
  (a * c * d) / b = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l606_60601


namespace NUMINAMATH_CALUDE_positive_sum_greater_than_abs_diff_l606_60657

theorem positive_sum_greater_than_abs_diff (x y : ℝ) :
  x + y > |x - y| ↔ x > 0 ∧ y > 0 := by sorry

end NUMINAMATH_CALUDE_positive_sum_greater_than_abs_diff_l606_60657


namespace NUMINAMATH_CALUDE_infinite_solutions_imply_a_equals_five_l606_60672

/-- If the equation 3(5 + ay) = 15y + 15 has infinitely many solutions for y, then a = 5 -/
theorem infinite_solutions_imply_a_equals_five (a : ℝ) : 
  (∀ y : ℝ, 3 * (5 + a * y) = 15 * y + 15) → a = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinite_solutions_imply_a_equals_five_l606_60672


namespace NUMINAMATH_CALUDE_length_of_AE_l606_60616

/-- Given a coordinate grid where:
    - A is at (0,4)
    - B is at (7,0)
    - C is at (3,0)
    - D is at (5,3)
    - Line segment AB meets line segment CD at point E
    Prove that the length of segment AE is (7√65)/13 -/
theorem length_of_AE (A B C D E : ℝ × ℝ) : 
  A = (0, 4) →
  B = (7, 0) →
  C = (3, 0) →
  D = (5, 3) →
  E ∈ Set.Icc A B →
  E ∈ Set.Icc C D →
  Real.sqrt ((E.1 - A.1)^2 + (E.2 - A.2)^2) = (7 * Real.sqrt 65) / 13 :=
by sorry

end NUMINAMATH_CALUDE_length_of_AE_l606_60616


namespace NUMINAMATH_CALUDE_probability_calm_in_mathematics_l606_60678

def letters_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}
def letters_calm : Finset Char := {'C', 'A', 'L', 'M'}

def count_occurrences (c : Char) : ℕ :=
  if c = 'M' ∨ c = 'A' then 2
  else if c ∈ letters_mathematics then 1
  else 0

def favorable_outcomes : ℕ := (letters_calm ∩ letters_mathematics).sum count_occurrences

theorem probability_calm_in_mathematics :
  (favorable_outcomes : ℚ) / 12 = 5 / 12 := by sorry

end NUMINAMATH_CALUDE_probability_calm_in_mathematics_l606_60678


namespace NUMINAMATH_CALUDE_work_completion_time_l606_60679

theorem work_completion_time (a b c : ℝ) (h1 : b = 6) (h2 : c = 12) 
  (h3 : 1/a + 1/b + 1/c = 7/24) : a = 24 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l606_60679


namespace NUMINAMATH_CALUDE_quadratic_inequality_transformation_l606_60639

theorem quadratic_inequality_transformation (a b c : ℝ) :
  (∀ x, 1 < x ∧ x < 2 → a * x^2 + b * x + c > 0) →
  (∀ x, c * x^2 + b * x + a > 0 ↔ 1/2 < x ∧ x < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_transformation_l606_60639


namespace NUMINAMATH_CALUDE_third_term_is_five_l606_60692

-- Define the sequence a_n
def a (n : ℕ) : ℕ := sorry

-- Define the sum function S_n
def S (n : ℕ) : ℕ := n^2

-- State the theorem
theorem third_term_is_five :
  a 3 = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_third_term_is_five_l606_60692


namespace NUMINAMATH_CALUDE_common_points_on_line_l606_60664

-- Define the curves and line
def C1 (a : ℝ) : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 1)^2 = a^2}
def C2 : Set (ℝ × ℝ) := {p | (p.1 - 2)^2 + p.2^2 = 4}
def C3 : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1}

theorem common_points_on_line (a : ℝ) (h : a > 0) : 
  (∀ p, p ∈ C1 a ∩ C2 → p ∈ C3) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_common_points_on_line_l606_60664


namespace NUMINAMATH_CALUDE_vector_magnitude_l606_60625

theorem vector_magnitude (a b : ℝ × ℝ × ℝ) : 
  (‖a‖ = 2) → (‖b‖ = 3) → (‖a + b‖ = 3) → ‖a + 2 • b‖ = 4 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_l606_60625


namespace NUMINAMATH_CALUDE_marbles_given_correct_l606_60682

/-- The number of marbles Jack gave to Josh -/
def marbles_given (initial final : ℕ) : ℕ := final - initial

/-- Theorem stating that the number of marbles given is the difference between final and initial counts -/
theorem marbles_given_correct (initial final : ℕ) (h : final ≥ initial) :
  marbles_given initial final = final - initial :=
by sorry

end NUMINAMATH_CALUDE_marbles_given_correct_l606_60682


namespace NUMINAMATH_CALUDE_students_wearing_other_colors_l606_60686

theorem students_wearing_other_colors 
  (total_students : ℕ) 
  (blue_percent : ℚ) 
  (red_percent : ℚ) 
  (green_percent : ℚ) 
  (h1 : total_students = 800) 
  (h2 : blue_percent = 45/100) 
  (h3 : red_percent = 23/100) 
  (h4 : green_percent = 15/100) : 
  ℕ := by
  sorry

#check students_wearing_other_colors

end NUMINAMATH_CALUDE_students_wearing_other_colors_l606_60686


namespace NUMINAMATH_CALUDE_subtraction_result_l606_60669

-- Define the two numbers
def a : ℚ := 888.88
def b : ℚ := 555.55

-- Define the result
def result : ℚ := a - b

-- Theorem to prove
theorem subtraction_result : result = 333.33 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_result_l606_60669


namespace NUMINAMATH_CALUDE_weight_of_three_moles_CaI2_l606_60667

/-- The atomic weight of Calcium in g/mol -/
def atomic_weight_Ca : ℝ := 40.08

/-- The atomic weight of Iodine in g/mol -/
def atomic_weight_I : ℝ := 126.90

/-- The molecular weight of CaI2 in g/mol -/
def molecular_weight_CaI2 : ℝ := atomic_weight_Ca + 2 * atomic_weight_I

/-- The weight of n moles of CaI2 in grams -/
def weight_CaI2 (n : ℝ) : ℝ := n * molecular_weight_CaI2

theorem weight_of_three_moles_CaI2 : 
  weight_CaI2 3 = 881.64 := by sorry

end NUMINAMATH_CALUDE_weight_of_three_moles_CaI2_l606_60667


namespace NUMINAMATH_CALUDE_perpendicular_condition_l606_60656

def is_perpendicular (a : ℝ) : Prop :=
  (a ≠ -1 ∧ a ≠ 0 ∧ -(a + 1) / (3 * a) * ((1 - a) / (a + 1)) = -1) ∨
  (a = -1)

theorem perpendicular_condition (a : ℝ) :
  (a = 1/4 → is_perpendicular a) ∧
  ¬(is_perpendicular a → a = 1/4) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l606_60656


namespace NUMINAMATH_CALUDE_right_triangle_area_l606_60652

/-- The area of a right triangle with hypotenuse 10√2 and one 45° angle is 50 square inches. -/
theorem right_triangle_area (h : ℝ) (α : ℝ) (A : ℝ) : 
  h = 10 * Real.sqrt 2 →  -- hypotenuse length
  α = 45 * π / 180 →      -- one angle in radians
  A = 50 →                -- area
  ∃ (a b : ℝ), 
    a^2 + b^2 = h^2 ∧     -- Pythagorean theorem
    Real.cos α = a / h ∧  -- cosine of the angle
    A = (1/2) * a * b     -- area formula
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l606_60652


namespace NUMINAMATH_CALUDE_vat_percentage_calculation_l606_60683

theorem vat_percentage_calculation (original_price final_price : ℝ) : 
  original_price = 1700 → 
  final_price = 1955 → 
  (final_price - original_price) / original_price * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_vat_percentage_calculation_l606_60683


namespace NUMINAMATH_CALUDE_mathematics_magnet_problem_l606_60659

/-- The number of letters in 'MATHEMATICS' -/
def total_letters : ℕ := 11

/-- The number of vowels in 'MATHEMATICS' -/
def num_vowels : ℕ := 4

/-- The number of consonants in 'MATHEMATICS' -/
def num_consonants : ℕ := 7

/-- The number of vowels selected -/
def selected_vowels : ℕ := 3

/-- The number of consonants selected -/
def selected_consonants : ℕ := 4

/-- The number of distinct possible collections of letters -/
def distinct_collections : ℕ := 490

theorem mathematics_magnet_problem :
  (total_letters = num_vowels + num_consonants) →
  (distinct_collections = 490) :=
by sorry

end NUMINAMATH_CALUDE_mathematics_magnet_problem_l606_60659


namespace NUMINAMATH_CALUDE_triangle_with_consecutive_sides_and_obtuse_angle_l606_60693

-- Define a triangle with sides of consecutive natural numbers
def ConsecutiveSidedTriangle (a b c : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧ (a > 0)

-- Define the condition for the largest angle to be obtuse
def HasObtuseAngle (a b c : ℕ) : Prop :=
  let cosLargestAngle := (a^2 + b^2 - c^2) / (2 * a * b)
  cosLargestAngle < 0

-- Theorem statement
theorem triangle_with_consecutive_sides_and_obtuse_angle
  (a b c : ℕ) (h1 : ConsecutiveSidedTriangle a b c) (h2 : HasObtuseAngle a b c) :
  (a = 2 ∧ b = 3 ∧ c = 4) :=
sorry

end NUMINAMATH_CALUDE_triangle_with_consecutive_sides_and_obtuse_angle_l606_60693


namespace NUMINAMATH_CALUDE_tank_filling_l606_60621

theorem tank_filling (original_buckets : ℕ) (capacity_reduction : ℚ) : 
  original_buckets = 10 →
  capacity_reduction = 2/5 →
  ∃ (new_buckets : ℕ), new_buckets ≥ 25 ∧ 
    (new_buckets : ℚ) * capacity_reduction = original_buckets := by
  sorry

end NUMINAMATH_CALUDE_tank_filling_l606_60621


namespace NUMINAMATH_CALUDE_toothpick_grid_25_15_l606_60673

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ

/-- Calculates the total number of toothpicks in a grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontal := (grid.height + 1) * grid.width
  let vertical := (grid.width + 1) * grid.height
  let diagonal := grid.height * grid.width
  horizontal + vertical + diagonal

/-- Theorem stating the total number of toothpicks in a 25x15 grid -/
theorem toothpick_grid_25_15 :
  total_toothpicks ⟨25, 15⟩ = 1165 := by sorry

end NUMINAMATH_CALUDE_toothpick_grid_25_15_l606_60673


namespace NUMINAMATH_CALUDE_x_less_than_y_l606_60614

theorem x_less_than_y (x y : ℝ) (h : (2023 : ℝ)^x + (2024 : ℝ)^(-y) < (2023 : ℝ)^y + (2024 : ℝ)^(-x)) : x < y := by
  sorry

end NUMINAMATH_CALUDE_x_less_than_y_l606_60614


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l606_60684

/-- Given a function y = x^α where α < 0, and a point A that lies on both y = x^α and y = mx + n 
    where m > 0 and n > 0, the minimum value of 1/m + 1/n is 4. -/
theorem min_value_reciprocal_sum (α m n : ℝ) (hα : α < 0) (hm : m > 0) (hn : n > 0) :
  (∃ x y : ℝ, y = x^α ∧ y = m*x + n) → 
  (∀ m' n' : ℝ, m' > 0 → n' > 0 → (∃ x' y' : ℝ, y' = x'^α ∧ y' = m'*x' + n') → 1/m + 1/n ≤ 1/m' + 1/n') →
  1/m + 1/n = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l606_60684


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l606_60689

theorem quadratic_equation_solution :
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l606_60689


namespace NUMINAMATH_CALUDE_initial_population_is_10000_l606_60612

/-- Represents the annual population growth rate -/
def annual_growth_rate : ℝ := 0.1

/-- Represents the population after 2 years -/
def population_after_2_years : ℕ := 12100

/-- Calculates the population after n years given an initial population -/
def population_after_n_years (initial_population : ℝ) (n : ℕ) : ℝ :=
  initial_population * (1 + annual_growth_rate) ^ n

/-- Theorem stating that if a population grows by 10% annually and reaches 12100 after 2 years,
    the initial population was 10000 -/
theorem initial_population_is_10000 :
  ∃ (initial_population : ℕ),
    (population_after_n_years initial_population 2 = population_after_2_years) ∧
    (initial_population = 10000) := by
  sorry

end NUMINAMATH_CALUDE_initial_population_is_10000_l606_60612


namespace NUMINAMATH_CALUDE_vector_on_line_l606_60650

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- A line passing through two vectors p and q can be parameterized as p + t(q - p) for some real t -/
def line_through (p q : V) (t : ℝ) : V := p + t • (q - p)

/-- The theorem states that if m*p + 5/8*q lies on the line through p and q, then m = 3/8 -/
theorem vector_on_line (p q : V) (m : ℝ) 
  (h : ∃ t : ℝ, m • p + (5/8) • q = line_through p q t) : 
  m = 3/8 := by
sorry

end NUMINAMATH_CALUDE_vector_on_line_l606_60650


namespace NUMINAMATH_CALUDE_divisibility_by_x_minus_a_squared_l606_60699

theorem divisibility_by_x_minus_a_squared (n : ℕ) (a : ℝ) (h : a ≠ 0) :
  ∃ P : ℝ → ℝ, ∀ x : ℝ, x^n - n*a^(n-1)*x + (n-1)*a^n = (x - a)^2 * P x :=
sorry

end NUMINAMATH_CALUDE_divisibility_by_x_minus_a_squared_l606_60699


namespace NUMINAMATH_CALUDE_other_communities_count_l606_60662

theorem other_communities_count (total : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total = 300 →
  muslim_percent = 44/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  (total : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 54 := by
sorry

end NUMINAMATH_CALUDE_other_communities_count_l606_60662


namespace NUMINAMATH_CALUDE_prob_both_white_l606_60629

/-- Represents an urn with white and black balls -/
structure Urn :=
  (white : ℕ)
  (black : ℕ)

/-- Calculates the probability of drawing a white ball from an urn -/
def prob_white (u : Urn) : ℚ :=
  u.white / (u.white + u.black)

/-- The first urn -/
def urn1 : Urn := ⟨2, 10⟩

/-- The second urn -/
def urn2 : Urn := ⟨8, 4⟩

/-- Theorem: The probability of drawing white balls from both urns is 1/9 -/
theorem prob_both_white : prob_white urn1 * prob_white urn2 = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_white_l606_60629


namespace NUMINAMATH_CALUDE_at_least_one_good_product_l606_60674

theorem at_least_one_good_product (total : Nat) (defective : Nat) (selected : Nat) 
  (h1 : total = 12)
  (h2 : defective = 2)
  (h3 : selected = 3)
  (h4 : defective < total)
  (h5 : selected ≤ total) :
  ∀ (selection : Finset (Fin total)), selection.card = selected → 
    ∃ (x : Fin total), x ∈ selection ∧ x.val ∉ Finset.range defective :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_good_product_l606_60674


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l606_60690

def polynomial (x : ℝ) : ℝ := 5 * (2 * x^8 - 3 * x^5 + 9) + 6 * (x^6 + 4 * x^3 - 6)

theorem sum_of_coefficients : (polynomial 1) = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l606_60690


namespace NUMINAMATH_CALUDE_conditional_prob_specific_given_different_l606_60653

/-- The number of attractions available for tourists to choose from. -/
def num_attractions : ℕ := 5

/-- The probability that two tourists choose different attractions. -/
def prob_different_attractions : ℚ := 4 / 5

/-- The probability that one tourist chooses a specific attraction and the other chooses any of the remaining attractions. -/
def prob_one_specific_others_different : ℚ := 8 / 25

/-- Theorem stating the conditional probability of both tourists choosing a specific attraction given they choose different attractions. -/
theorem conditional_prob_specific_given_different :
  prob_one_specific_others_different / prob_different_attractions = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_conditional_prob_specific_given_different_l606_60653


namespace NUMINAMATH_CALUDE_son_age_l606_60644

/-- Represents the ages of a father and son -/
structure Ages where
  father : ℕ
  son : ℕ

/-- The conditions of the age problem -/
def AgeConditions (ages : Ages) : Prop :=
  (ages.father + ages.son = 75) ∧
  (∃ (x : ℕ), ages.father = 8 * (ages.son - x) ∧ ages.father - x = ages.son)

/-- The theorem stating that under the given conditions, the son's age is 27 -/
theorem son_age (ages : Ages) (h : AgeConditions ages) : ages.son = 27 := by
  sorry

end NUMINAMATH_CALUDE_son_age_l606_60644


namespace NUMINAMATH_CALUDE_katies_speed_l606_60637

/-- Given the running speeds of Eugene, Brianna, Marcus, and Katie, prove Katie's speed -/
theorem katies_speed (eugene_speed : ℝ) (brianna_ratio : ℝ) (marcus_ratio : ℝ) (katie_ratio : ℝ)
  (h1 : eugene_speed = 5)
  (h2 : brianna_ratio = 3/4)
  (h3 : marcus_ratio = 5/6)
  (h4 : katie_ratio = 4/5) :
  katie_ratio * marcus_ratio * brianna_ratio * eugene_speed = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_katies_speed_l606_60637


namespace NUMINAMATH_CALUDE_smallest_prime_is_two_l606_60649

theorem smallest_prime_is_two (p q r : ℕ) : 
  Nat.Prime p → Nat.Prime q → Nat.Prime r → 
  p ≠ q → p ≠ r → q ≠ r →
  p^3 + q^3 + 3*p*q*r = r^3 →
  min p (min q r) = 2 := by
sorry

end NUMINAMATH_CALUDE_smallest_prime_is_two_l606_60649


namespace NUMINAMATH_CALUDE_smallest_valid_m_l606_60695

def T : Set ℂ := {z | ∃ x y : ℝ, z = x + y * Complex.I ∧ 1/2 ≤ x ∧ x ≤ Real.sqrt 3 / 2 ∧ Real.sqrt 2 / 2 ≤ y ∧ y ≤ 1}

def is_valid_m (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z ∈ T, z^n = Complex.I

theorem smallest_valid_m : 
  (is_valid_m 6) ∧ (∀ m : ℕ, m < 6 → ¬(is_valid_m m)) :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_m_l606_60695


namespace NUMINAMATH_CALUDE_rectangle_width_decrease_l606_60643

theorem rectangle_width_decrease (L W : ℝ) (h1 : L > 0) (h2 : W > 0) : 
  let new_length := 1.5 * L
  let new_width := W * (L / new_length)
  let percent_decrease := (W - new_width) / W * 100
  percent_decrease = 100/3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_decrease_l606_60643


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l606_60691

theorem min_value_reciprocal_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq : a + b + c = 9) :
  2/a + 2/b + 2/c ≥ 2 ∧ 
  (2/a + 2/b + 2/c = 2 ↔ a = 3 ∧ b = 3 ∧ c = 3) := by
  sorry

#check min_value_reciprocal_sum

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l606_60691


namespace NUMINAMATH_CALUDE_largest_non_sum_36_composite_l606_60641

/-- A function that checks if a number is composite -/
def isComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a multiple of 36 and a composite number -/
def isSum36Composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ isComposite m ∧ n = 36 * k + m

/-- Theorem stating that 253 is the largest number that cannot be expressed as the sum of a multiple of 36 and a composite number -/
theorem largest_non_sum_36_composite :
  (¬ isSum36Composite 253) ∧ (∀ n > 253, isSum36Composite n) :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_36_composite_l606_60641


namespace NUMINAMATH_CALUDE_coin_toss_probability_l606_60661

def coin_toss_events : ℕ := 2^4

def favorable_events : ℕ := 11

theorem coin_toss_probability : 
  (favorable_events : ℚ) / coin_toss_events = 11 / 16 :=
sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l606_60661


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l606_60666

theorem sum_of_reciprocals (x y : ℝ) (h1 : x + y = 16) (h2 : x * y = 48) : 
  1 / x + 1 / y = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l606_60666


namespace NUMINAMATH_CALUDE_carla_smoothie_cream_l606_60665

/-- Given information about Carla's smoothie recipe, prove the amount of cream used. -/
theorem carla_smoothie_cream (watermelon_puree : ℕ) (num_servings : ℕ) (serving_size : ℕ) 
  (h1 : watermelon_puree = 500)
  (h2 : num_servings = 4)
  (h3 : serving_size = 150) :
  num_servings * serving_size - watermelon_puree = 100 := by
  sorry

#check carla_smoothie_cream

end NUMINAMATH_CALUDE_carla_smoothie_cream_l606_60665


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_point_seven_l606_60600

theorem at_least_one_greater_than_point_seven (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (max a (max (b^2) (1 / (a^2 + b))) : ℝ) > 0.7 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_point_seven_l606_60600


namespace NUMINAMATH_CALUDE_spongebob_daily_earnings_l606_60697

/-- Spongebob's earnings for a day of work at the burger shop -/
def spongebob_earnings (num_burgers : ℕ) (price_burger : ℚ) (num_fries : ℕ) (price_fries : ℚ) : ℚ :=
  num_burgers * price_burger + num_fries * price_fries

/-- Theorem: Spongebob's earnings for the day are $78 -/
theorem spongebob_daily_earnings : 
  spongebob_earnings 30 2 12 (3/2) = 78 := by
sorry

end NUMINAMATH_CALUDE_spongebob_daily_earnings_l606_60697


namespace NUMINAMATH_CALUDE_hyperbola_equation_l606_60609

/-- The standard equation of a hyperbola given specific conditions -/
theorem hyperbola_equation (F₁ F₂ M : ℝ × ℝ) : 
  F₁ = (0, Real.sqrt 10) →
  F₂ = (0, -Real.sqrt 10) →
  (M.1 - F₁.1) * (M.1 - F₂.1) + (M.2 - F₁.2) * (M.2 - F₂.2) = 0 →
  Real.sqrt ((M.1 - F₁.1)^2 + (M.2 - F₁.2)^2) * 
    Real.sqrt ((M.1 - F₂.1)^2 + (M.2 - F₂.2)^2) = 2 →
  M.2^2 / 9 - M.1^2 = 1 := by
  sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l606_60609


namespace NUMINAMATH_CALUDE_special_line_equation_l606_60676

/-- A line passing through point (3,-1) with x-intercept twice its y-intercept -/
structure SpecialLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through point (3,-1) -/
  passes_through_point : slope * 3 + y_intercept = -1
  /-- The x-intercept is twice the y-intercept -/
  intercept_condition : y_intercept ≠ 0 → -y_intercept / slope = 2 * y_intercept

theorem special_line_equation (l : SpecialLine) :
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + 2 * y - 1 = 0) ∨
  (∀ x y, y = l.slope * x + l.y_intercept ↔ x + 3 * y = 0) :=
sorry

end NUMINAMATH_CALUDE_special_line_equation_l606_60676


namespace NUMINAMATH_CALUDE_album_duration_calculation_l606_60640

/-- Calculates the total duration of an album in minutes -/
def albumDuration (initialSongs : ℕ) (additionalSongs : ℕ) (songDuration : ℕ) : ℕ :=
  (initialSongs + additionalSongs) * songDuration

theorem album_duration_calculation :
  albumDuration 25 10 3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_album_duration_calculation_l606_60640


namespace NUMINAMATH_CALUDE_fractional_to_polynomial_equivalence_l606_60634

theorem fractional_to_polynomial_equivalence (x : ℝ) (h : x ≠ 2) :
  (x / (x - 2) + 2 = 1 / (2 - x)) ↔ (x + 2 * (x - 2) = -1) :=
sorry

end NUMINAMATH_CALUDE_fractional_to_polynomial_equivalence_l606_60634


namespace NUMINAMATH_CALUDE_shaded_region_characterization_l606_60698

def shaded_region (z : ℂ) : Prop :=
  Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ)

theorem shaded_region_characterization :
  ∀ z : ℂ, z ∈ {z | shaded_region z} ↔ 
    (Complex.abs z ≤ 1 ∧ Complex.im z ≥ (1/2 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_shaded_region_characterization_l606_60698


namespace NUMINAMATH_CALUDE_smallest_middle_term_l606_60677

theorem smallest_middle_term (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c → 
  (∃ d : ℝ, a = b - d ∧ c = b + d) → 
  a * b * c = 216 → 
  b ≥ 6 := by
sorry

end NUMINAMATH_CALUDE_smallest_middle_term_l606_60677


namespace NUMINAMATH_CALUDE_quadratic_root_problem_l606_60630

theorem quadratic_root_problem (m : ℝ) : 
  ((0 : ℝ) = 0 → (m - 2) * 0^2 + 4 * 0 + 2 - |m| = 0) ∧ 
  (m - 2 ≠ 0) → 
  m = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_problem_l606_60630


namespace NUMINAMATH_CALUDE_parallelepipeds_crossed_diagonal_count_l606_60688

/-- The edge length of the cube -/
def cube_edge : ℕ := 90

/-- The dimensions of the rectangular parallelepiped -/
def parallelepiped_dims : Fin 3 → ℕ
| 0 => 2
| 1 => 3
| 2 => 5

/-- The number of parallelepipeds that fit along each dimension of the cube -/
def parallelepipeds_per_dim (i : Fin 3) : ℕ := cube_edge / parallelepiped_dims i

/-- The total number of parallelepipeds that fit in the cube -/
def total_parallelepipeds : ℕ := (parallelepipeds_per_dim 0) * (parallelepipeds_per_dim 1) * (parallelepipeds_per_dim 2)

/-- The number of parallelepipeds crossed by a space diagonal of the cube -/
def parallelepipeds_crossed_by_diagonal : ℕ := 65

theorem parallelepipeds_crossed_diagonal_count :
  parallelepipeds_crossed_by_diagonal = 65 :=
sorry

end NUMINAMATH_CALUDE_parallelepipeds_crossed_diagonal_count_l606_60688


namespace NUMINAMATH_CALUDE_reciprocal_inequality_l606_60646

theorem reciprocal_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_inequality_l606_60646


namespace NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_unique_index_298_l606_60610

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℕ := 1 + (n - 1) * 3

/-- The theorem stating that the 100th term of the arithmetic sequence is 298 -/
theorem arithmetic_sequence_100th_term :
  arithmetic_sequence 100 = 298 := by sorry

/-- The theorem stating that 100 is the unique index for which the term is 298 -/
theorem unique_index_298 :
  ∀ n : ℕ, arithmetic_sequence n = 298 ↔ n = 100 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_100th_term_unique_index_298_l606_60610


namespace NUMINAMATH_CALUDE_equation_solution_l606_60660

theorem equation_solution (x : ℝ) : (2*x - 3)^(x + 3) = 1 ↔ x = -3 ∨ x = 2 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l606_60660


namespace NUMINAMATH_CALUDE_parabola_vertex_on_line_l606_60675

/-- The value of d for which the vertex of the parabola y = x^2 - 10x + d lies on the line y = 2x --/
theorem parabola_vertex_on_line (d : ℝ) : 
  (∃ x y : ℝ, y = x^2 - 10*x + d ∧ 
              y = 2*x ∧ 
              ∀ t : ℝ, (t^2 - 10*t + d) ≥ (x^2 - 10*x + d)) → 
  d = 35 := by
sorry

end NUMINAMATH_CALUDE_parabola_vertex_on_line_l606_60675


namespace NUMINAMATH_CALUDE_simplify_expression_l606_60608

theorem simplify_expression : (4 + 2 + 6) / 3 - (2 + 1) / 3 = 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l606_60608


namespace NUMINAMATH_CALUDE_constant_product_equals_one_fourth_l606_60633

/-- Given a function f(x) = (bx + 1) / (2x + a), where a and b are constants,
    and ab ≠ 2, prove that if f(x) * f(1/x) is constant for all x ≠ 0,
    then this constant equals 1/4. -/
theorem constant_product_equals_one_fourth
  (a b : ℝ) (h : a * b ≠ 2)
  (f : ℝ → ℝ)
  (hf : ∀ x, x ≠ 0 → f x = (b * x + 1) / (2 * x + a))
  (h_constant : ∃ k, ∀ x, x ≠ 0 → f x * f (1/x) = k) :
  ∃ k, (∀ x, x ≠ 0 → f x * f (1/x) = k) ∧ k = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_constant_product_equals_one_fourth_l606_60633


namespace NUMINAMATH_CALUDE_gel_pen_price_l606_60655

theorem gel_pen_price (x y b g : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : b > 0) (h4 : g > 0) : 
  ((x + y) * g = 4 * (x * b + y * g)) → 
  ((x + y) * b = (1/2) * (x * b + y * g)) → 
  g = 8 * b :=
by sorry

end NUMINAMATH_CALUDE_gel_pen_price_l606_60655


namespace NUMINAMATH_CALUDE_max_d_value_l606_60613

def a (n : ℕ+) : ℕ := 150 + 3 * n.val ^ 2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ (n : ℕ+), d n = 147) ∧ (∀ (n : ℕ+), d n ≤ 147) :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l606_60613
