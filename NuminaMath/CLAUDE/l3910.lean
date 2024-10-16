import Mathlib

namespace NUMINAMATH_CALUDE_regression_analysis_l3910_391046

structure RegressionData where
  n : ℕ
  x : Fin n → ℝ
  y : Fin n → ℝ
  original_slope : ℝ
  original_intercept : ℝ
  x_mean : ℝ
  new_slope : ℝ

def positive_correlation (data : RegressionData) : Prop :=
  data.new_slope > 0

def new_regression_equation (data : RegressionData) : Prop :=
  ∃ new_intercept : ℝ, new_intercept = data.x_mean * (data.original_slope - data.new_slope) + data.original_intercept + 1

def decreased_rate_of_increase (data : RegressionData) : Prop :=
  data.new_slope < data.original_slope

theorem regression_analysis (data : RegressionData) 
  (h1 : data.original_slope = 2)
  (h2 : data.original_intercept = -1)
  (h3 : data.x_mean = 3)
  (h4 : data.new_slope = 1.2) :
  positive_correlation data ∧ 
  new_regression_equation data ∧ 
  decreased_rate_of_increase data := by
  sorry

end NUMINAMATH_CALUDE_regression_analysis_l3910_391046


namespace NUMINAMATH_CALUDE_trapezoid_area_is_400_l3910_391059

-- Define the trapezoid and square properties
def trapezoid_base1 : ℝ := 50
def trapezoid_base2 : ℝ := 30
def num_trapezoids : ℕ := 4
def outer_square_area : ℝ := 2500

-- Theorem statement
theorem trapezoid_area_is_400 :
  let outer_square_side : ℝ := trapezoid_base1
  let inner_square_side : ℝ := trapezoid_base2
  let inner_square_area : ℝ := inner_square_side ^ 2
  let total_trapezoid_area : ℝ := outer_square_area - inner_square_area
  let single_trapezoid_area : ℝ := total_trapezoid_area / num_trapezoids
  single_trapezoid_area = 400 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_is_400_l3910_391059


namespace NUMINAMATH_CALUDE_smaller_number_proof_l3910_391036

theorem smaller_number_proof (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a / b = 3 / 4) (h4 : a + b = 21) (h5 : max a b = 12) : 
  min a b = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_number_proof_l3910_391036


namespace NUMINAMATH_CALUDE_earlier_movie_savings_l3910_391025

/-- Calculates the savings when attending an earlier movie with discounts -/
def calculate_savings (evening_ticket_cost : ℝ) (food_combo_cost : ℝ) 
  (ticket_discount_percent : ℝ) (food_discount_percent : ℝ) : ℝ :=
  (evening_ticket_cost * ticket_discount_percent) + 
  (food_combo_cost * food_discount_percent)

/-- Proves that the savings for the earlier movie is $7 -/
theorem earlier_movie_savings :
  let evening_ticket_cost : ℝ := 10
  let food_combo_cost : ℝ := 10
  let ticket_discount_percent : ℝ := 0.2
  let food_discount_percent : ℝ := 0.5
  calculate_savings evening_ticket_cost food_combo_cost 
    ticket_discount_percent food_discount_percent = 7 := by
  sorry

end NUMINAMATH_CALUDE_earlier_movie_savings_l3910_391025


namespace NUMINAMATH_CALUDE_min_perimeter_sum_l3910_391001

/-- Represents a chessboard configuration -/
structure ChessboardConfig (m : ℕ) where
  size : Fin (2^m) → Fin (2^m) → Bool
  diagonal_unit : ∀ i : Fin (2^m), size i i = true

/-- Calculates the sum of perimeters for a given chessboard configuration -/
def sumPerimeters (m : ℕ) (config : ChessboardConfig m) : ℕ :=
  sorry

/-- Theorem: The minimum sum of perimeters for a 2^m × 2^m chessboard configuration -/
theorem min_perimeter_sum (m : ℕ) : 
  (∃ (config : ChessboardConfig m), 
    ∀ (other_config : ChessboardConfig m), 
      sumPerimeters m config ≤ sumPerimeters m other_config) ∧
  (∃ (config : ChessboardConfig m), sumPerimeters m config = 2^(m+2) * (m+1)) := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_sum_l3910_391001


namespace NUMINAMATH_CALUDE_lindas_bills_l3910_391076

/-- Represents the number of bills of each denomination -/
structure BillCount where
  fives : ℕ
  tens : ℕ

/-- Calculates the total value of bills -/
def totalValue (bc : BillCount) : ℕ :=
  5 * bc.fives + 10 * bc.tens

/-- Calculates the total number of bills -/
def totalBills (bc : BillCount) : ℕ :=
  bc.fives + bc.tens

theorem lindas_bills :
  ∃ (bc : BillCount), totalValue bc = 80 ∧ totalBills bc = 12 ∧ bc.fives = 8 := by
  sorry

end NUMINAMATH_CALUDE_lindas_bills_l3910_391076


namespace NUMINAMATH_CALUDE_odd_sided_polygon_indivisible_l3910_391033

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → ℝ × ℝ
  is_regular : sorry -- Additional conditions to ensure the polygon is regular

/-- The diameter of a polygon -/
def diameter (p : RegularPolygon n) : ℝ := sorry

/-- A division of a polygon into two parts -/
structure Division (p : RegularPolygon n) where
  part1 : Set (Fin n)
  part2 : Set (Fin n)
  is_partition : part1 ∪ part2 = univ ∧ part1 ∩ part2 = ∅

/-- The diameter of a part of a polygon -/
def part_diameter (p : RegularPolygon n) (part : Set (Fin n)) : ℝ := sorry

theorem odd_sided_polygon_indivisible (n : ℕ) (h : Odd n) (p : RegularPolygon n) :
  ∀ d : Division p, 
    part_diameter p d.part1 = diameter p ∨ part_diameter p d.part2 = diameter p := by
  sorry

end NUMINAMATH_CALUDE_odd_sided_polygon_indivisible_l3910_391033


namespace NUMINAMATH_CALUDE_original_price_calculation_l3910_391010

-- Define the discounts
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.05

-- Define the final price
def final_price : ℝ := 266

-- Theorem statement
theorem original_price_calculation :
  ∃ P : ℝ, P * (1 - discount1) * (1 - discount2) = final_price ∧ P = 350 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l3910_391010


namespace NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l3910_391065

/-- The sum of the infinite series Σ(n=1 to ∞) (n^5 + 5n^3 + 15n + 15) / (2^n * (n^5 + 5)) is equal to 1 -/
theorem infinite_series_sum_equals_one :
  let f : ℕ → ℝ := λ n => (n^5 + 5*n^3 + 15*n + 15) / (2^n * (n^5 + 5))
  ∑' n, f n = 1 := by sorry

end NUMINAMATH_CALUDE_infinite_series_sum_equals_one_l3910_391065


namespace NUMINAMATH_CALUDE_scale_division_theorem_l3910_391084

/-- Represents the length of an object in feet and inches -/
structure Length where
  feet : ℕ
  inches : ℕ
  h : inches < 12

/-- Converts a Length to total inches -/
def Length.toInches (l : Length) : ℕ := l.feet * 12 + l.inches

/-- The total length of the scale -/
def totalLength : Length := ⟨6, 8, by norm_num⟩

/-- Number of equal parts to divide the scale into -/
def numParts : ℕ := 2

/-- Represents the result of dividing the scale -/
def dividedLength : Length := ⟨3, 4, by norm_num⟩

theorem scale_division_theorem :
  (totalLength.toInches / numParts : ℕ) = dividedLength.toInches := by
  sorry

end NUMINAMATH_CALUDE_scale_division_theorem_l3910_391084


namespace NUMINAMATH_CALUDE_super_mindmaster_codes_l3910_391079

theorem super_mindmaster_codes (colors : ℕ) (slots : ℕ) : 
  colors = 9 → slots = 5 → colors ^ slots = 59049 := by
  sorry

end NUMINAMATH_CALUDE_super_mindmaster_codes_l3910_391079


namespace NUMINAMATH_CALUDE_additive_increasing_non_neg_implies_odd_and_increasing_l3910_391015

/-- A function satisfying f(x₁ + x₂) = f(x₁) + f(x₂) for all x₁, x₂ ∈ ℝ -/
def IsAdditive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂

/-- A function that is increasing on non-negative reals -/
def IsIncreasingNonNeg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≥ x₂ → x₂ ≥ 0 → f x₁ ≥ f x₂

/-- Main theorem: If f is additive and increasing on non-negative reals,
    then it is odd and increasing on all reals -/
theorem additive_increasing_non_neg_implies_odd_and_increasing
    (f : ℝ → ℝ) (h1 : IsAdditive f) (h2 : IsIncreasingNonNeg f) :
    (∀ x, f (-x) = -f x) ∧ (∀ x₁ x₂, x₁ ≥ x₂ → f x₁ ≥ f x₂) := by
  sorry

end NUMINAMATH_CALUDE_additive_increasing_non_neg_implies_odd_and_increasing_l3910_391015


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_OBEC_area_of_quadrilateral_OBEC_proof_l3910_391052

/-- A line with slope -3 passing through points A, B, and E -/
structure Line1 where
  slope : ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  E : ℝ × ℝ

/-- Another line passing through points C, D, and E -/
structure Line2 where
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The origin point O -/
def O : ℝ × ℝ := (0, 0)

/-- Definition of the problem setup -/
def ProblemSetup (l1 : Line1) (l2 : Line2) : Prop :=
  l1.slope = -3 ∧
  l1.A.1 > 0 ∧ l1.A.2 = 0 ∧
  l1.B.1 = 0 ∧ l1.B.2 > 0 ∧
  l1.E = (3, 3) ∧
  l2.C = (6, 0) ∧
  l2.D.1 = 0 ∧ l2.D.2 ≠ 0 ∧
  l2.E = (3, 3)

/-- The main theorem to prove -/
theorem area_of_quadrilateral_OBEC (l1 : Line1) (l2 : Line2) 
  (h : ProblemSetup l1 l2) : ℝ :=
  22.5

/-- Proof of the theorem -/
theorem area_of_quadrilateral_OBEC_proof (l1 : Line1) (l2 : Line2) 
  (h : ProblemSetup l1 l2) : 
  area_of_quadrilateral_OBEC l1 l2 h = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_OBEC_area_of_quadrilateral_OBEC_proof_l3910_391052


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l3910_391028

theorem fraction_zero_implies_x_negative_one (x : ℝ) : 
  (1 - |x|) / (1 - x) = 0 → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_negative_one_l3910_391028


namespace NUMINAMATH_CALUDE_calum_disco_ball_budget_l3910_391066

/-- Calculates the maximum amount Calum can spend on each disco ball given his budget and expenses. -/
theorem calum_disco_ball_budget (num_disco_balls : ℕ) (num_food_boxes : ℕ) (food_box_cost : ℚ) (total_budget : ℚ) :
  num_disco_balls = 4 →
  num_food_boxes = 10 →
  food_box_cost = 25 →
  total_budget = 330 →
  (total_budget - num_food_boxes * food_box_cost) / num_disco_balls = 20 :=
by sorry

end NUMINAMATH_CALUDE_calum_disco_ball_budget_l3910_391066


namespace NUMINAMATH_CALUDE_polygon_sides_l3910_391054

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_l3910_391054


namespace NUMINAMATH_CALUDE_not_divisible_by_5_and_9_l3910_391080

def count_not_divisible (n : ℕ) (a b : ℕ) : ℕ :=
  n - (n / a + n / b - n / (a * b))

theorem not_divisible_by_5_and_9 :
  count_not_divisible 1199 5 9 = 853 := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_5_and_9_l3910_391080


namespace NUMINAMATH_CALUDE_two_wizards_theorem_l3910_391057

/-- Represents a student in the wizardry school -/
structure Student where
  id : Fin 13
  hasDiploma : Bool

/-- The configuration of students around the table -/
def StudentConfiguration := Fin 13 → Student

/-- Check if a student's prediction is correct -/
def isPredictionCorrect (config : StudentConfiguration) (s : Student) : Bool :=
  let otherStudents := (List.range 13).filter (fun i => 
    i ≠ s.id.val ∧ 
    i ≠ (s.id.val + 1) % 13 ∧ 
    i ≠ (s.id.val + 12) % 13)
  otherStudents.all (fun i => ¬(config i).hasDiploma)

/-- The main theorem to prove -/
theorem two_wizards_theorem :
  ∃ (config : StudentConfiguration),
    (∀ s, (config s.id = s)) ∧
    (∃! (s1 s2 : Student), s1.hasDiploma ∧ s2.hasDiploma ∧ s1 ≠ s2) ∧
    (∀ s, s.hasDiploma ↔ isPredictionCorrect config s) := by
  sorry


end NUMINAMATH_CALUDE_two_wizards_theorem_l3910_391057


namespace NUMINAMATH_CALUDE_vodka_alcohol_percentage_l3910_391039

/-- Calculates the percentage of pure alcohol in vodka -/
theorem vodka_alcohol_percentage
  (total_shots : ℕ)
  (ounces_per_shot : ℚ)
  (pure_alcohol_consumed : ℚ) :
  total_shots = 8 →
  ounces_per_shot = 3/2 →
  pure_alcohol_consumed = 3 →
  (pure_alcohol_consumed / (total_shots * ounces_per_shot)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_vodka_alcohol_percentage_l3910_391039


namespace NUMINAMATH_CALUDE_track_length_is_480_l3910_391018

/-- Represents the circular track and the runners' movements --/
structure TrackSystem where
  trackLength : ℝ
  janetSpeed : ℝ
  leahSpeed : ℝ

/-- Conditions of the problem --/
def ProblemConditions (s : TrackSystem) : Prop :=
  s.janetSpeed > 0 ∧ 
  s.leahSpeed > 0 ∧ 
  120 / s.janetSpeed = (s.trackLength / 2 - 120) / s.leahSpeed ∧
  (s.trackLength / 2 - 40) / s.janetSpeed = 200 / s.leahSpeed

/-- The main theorem to prove --/
theorem track_length_is_480 (s : TrackSystem) : 
  ProblemConditions s → s.trackLength = 480 :=
sorry

end NUMINAMATH_CALUDE_track_length_is_480_l3910_391018


namespace NUMINAMATH_CALUDE_fuel_tank_capacity_l3910_391032

/-- Given a fuel tank partially filled with fuel A and then filled to capacity with fuel B,
    prove that the capacity of the tank is 162.5 gallons. -/
theorem fuel_tank_capacity
  (capacity : ℝ)
  (fuel_a_volume : ℝ)
  (fuel_a_ethanol_percent : ℝ)
  (fuel_b_ethanol_percent : ℝ)
  (total_ethanol : ℝ)
  (h1 : fuel_a_volume = 49.99999999999999)
  (h2 : fuel_a_ethanol_percent = 0.12)
  (h3 : fuel_b_ethanol_percent = 0.16)
  (h4 : total_ethanol = 30)
  (h5 : fuel_a_ethanol_percent * fuel_a_volume +
        fuel_b_ethanol_percent * (capacity - fuel_a_volume) = total_ethanol) :
  capacity = 162.5 := by
  sorry

end NUMINAMATH_CALUDE_fuel_tank_capacity_l3910_391032


namespace NUMINAMATH_CALUDE_inequality_theorem_l3910_391051

theorem inequality_theorem (a b c d : ℝ) (h1 : a > b) (h2 : c = d) : a + c > b + d := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l3910_391051


namespace NUMINAMATH_CALUDE_star_two_one_l3910_391050

-- Define the ∗ operation for real numbers
def star (x y : ℝ) : ℝ := x - y + x * y

-- State the theorem
theorem star_two_one : star 2 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_star_two_one_l3910_391050


namespace NUMINAMATH_CALUDE_sum_mod_seven_l3910_391062

theorem sum_mod_seven : (5000 + 5001 + 5002 + 5003 + 5004) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_seven_l3910_391062


namespace NUMINAMATH_CALUDE_multiply_polynomials_l3910_391026

theorem multiply_polynomials (x : ℝ) : (x^4 + 8*x^2 + 64) * (x^2 - 8) = x^4 + 16*x^2 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l3910_391026


namespace NUMINAMATH_CALUDE_equation_solution_l3910_391042

theorem equation_solution : ∃ x : ℝ, (17.28 / x) / (3.6 * 0.2) = 2 ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3910_391042


namespace NUMINAMATH_CALUDE_range_of_m_l3910_391009

/-- Proposition p: The equation x² + mx + 1 = 0 has two different negative real roots -/
def p (m : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ < 0 ∧ x₁ ≠ x₂ ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

/-- Proposition q: The equation 4x² + 4(m-2)x + 1 = 0 has no real roots -/
def q (m : ℝ) : Prop :=
  ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

/-- The main theorem stating the equivalence of the given conditions and the solution -/
theorem range_of_m :
  ∀ m : ℝ, ((p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ (m ≥ 3 ∨ (1 < m ∧ m ≤ 2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3910_391009


namespace NUMINAMATH_CALUDE_gumball_packages_l3910_391040

theorem gumball_packages (package_size : ℕ) (total_consumed : ℕ) 
  (h1 : package_size = 5)
  (h2 : total_consumed = 20) :
  (total_consumed / package_size = 4) ∧ (total_consumed % package_size = 0) := by
  sorry

end NUMINAMATH_CALUDE_gumball_packages_l3910_391040


namespace NUMINAMATH_CALUDE_area_enclosed_by_graph_l3910_391022

/-- The area enclosed by the graph of |x| + |3y| = 12 -/
def areaEnclosedByGraph : ℝ := 96

/-- The equation of the graph -/
def graphEquation (x y : ℝ) : Prop := (abs x) + (abs (3 * y)) = 12

theorem area_enclosed_by_graph :
  areaEnclosedByGraph = 96 := by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_graph_l3910_391022


namespace NUMINAMATH_CALUDE_stating_thirty_cents_combinations_l3910_391012

/-- The value of a penny in cents -/
def pennyValue : ℕ := 1

/-- The value of a nickel in cents -/
def nickelValue : ℕ := 5

/-- The value of a dime in cents -/
def dimeValue : ℕ := 10

/-- The total value we want to achieve in cents -/
def totalValue : ℕ := 30

/-- 
A function that calculates the number of ways to make a given amount of cents
using pennies, nickels, and dimes.
-/
def countCombinations (cents : ℕ) : ℕ := sorry

/-- 
Theorem stating that the number of combinations to make 30 cents
using pennies, nickels, and dimes is 20.
-/
theorem thirty_cents_combinations : 
  countCombinations totalValue = 20 := by sorry

end NUMINAMATH_CALUDE_stating_thirty_cents_combinations_l3910_391012


namespace NUMINAMATH_CALUDE_probability_ABABABBB_proof_l3910_391099

/-- The probability of arranging 5 A tiles and 3 B tiles in the specific order ABABABBB -/
def probability_ABABABBB : ℚ :=
  1 / 56

/-- The total number of ways to arrange 5 A tiles and 3 B tiles in a row -/
def total_arrangements : ℕ :=
  Nat.choose 8 5

theorem probability_ABABABBB_proof :
  probability_ABABABBB = (1 : ℚ) / total_arrangements := by
  sorry

#eval probability_ABABABBB
#eval total_arrangements

end NUMINAMATH_CALUDE_probability_ABABABBB_proof_l3910_391099


namespace NUMINAMATH_CALUDE_eldoria_license_plates_l3910_391002

/-- The number of vowels available for the first letter of a license plate. -/
def numVowels : ℕ := 5

/-- The number of letters in the alphabet. -/
def numLetters : ℕ := 26

/-- The number of digits (0-9). -/
def numDigits : ℕ := 10

/-- The number of characters in a valid license plate. -/
def licensePlateLength : ℕ := 5

/-- Calculates the number of valid license plates in Eldoria. -/
def numValidLicensePlates : ℕ :=
  numVowels * numLetters * numDigits * numDigits * numDigits

/-- Theorem stating the number of valid license plates in Eldoria. -/
theorem eldoria_license_plates :
  numValidLicensePlates = 130000 := by
  sorry

end NUMINAMATH_CALUDE_eldoria_license_plates_l3910_391002


namespace NUMINAMATH_CALUDE_recreation_spending_comparison_l3910_391011

theorem recreation_spending_comparison (last_week_wages : ℝ) : 
  let last_week_recreation := 0.20 * last_week_wages
  let this_week_wages := 0.80 * last_week_wages
  let this_week_recreation := 0.40 * this_week_wages
  (this_week_recreation / last_week_recreation) * 100 = 160 := by
sorry

end NUMINAMATH_CALUDE_recreation_spending_comparison_l3910_391011


namespace NUMINAMATH_CALUDE_xy_value_l3910_391029

theorem xy_value (x y : ℤ) (h : (30 : ℚ) / 2 * (x * y) = 21 * x + 20 * y - 13) : x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3910_391029


namespace NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3910_391095

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  is_valid : hundreds ≥ 1 ∧ hundreds ≤ 9 ∧ tens ≤ 9 ∧ units ≤ 9

/-- Converts a ThreeDigitNumber to its numerical value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

/-- Reverses the digits of a ThreeDigitNumber -/
def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.units
  tens := n.tens
  units := n.hundreds
  is_valid := by sorry

theorem unique_number_satisfying_conditions :
  ∃! (n : ThreeDigitNumber),
    n.hundreds + n.tens + n.units = 20 ∧
    ∃ (m : ThreeDigitNumber),
      n.toNat - 16 = m.toNat ∧
      m = n.reverse ∧
    n.hundreds = 9 ∧ n.tens = 7 ∧ n.units = 4 := by sorry

end NUMINAMATH_CALUDE_unique_number_satisfying_conditions_l3910_391095


namespace NUMINAMATH_CALUDE_student_number_problem_l3910_391082

theorem student_number_problem (x : ℝ) : 3 * x - 220 = 110 → x = 110 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l3910_391082


namespace NUMINAMATH_CALUDE_prob_exactly_two_correct_prob_at_least_two_correct_prob_all_incorrect_l3910_391030

/-- The number of students and backpacks -/
def n : ℕ := 4

/-- The total number of ways to pick up backpacks -/
def total_outcomes : ℕ := 24

/-- The number of outcomes where exactly two students pick up their correct backpacks -/
def exactly_two_correct : ℕ := 6

/-- The number of outcomes where at least two students pick up their correct backpacks -/
def at_least_two_correct : ℕ := 7

/-- The number of outcomes where all backpacks are picked up incorrectly -/
def all_incorrect : ℕ := 9

/-- The probability of exactly two students picking up the correct backpacks -/
theorem prob_exactly_two_correct : 
  exactly_two_correct / total_outcomes = 1 / 4 := by sorry

/-- The probability of at least two students picking up the correct backpacks -/
theorem prob_at_least_two_correct : 
  at_least_two_correct / total_outcomes = 7 / 24 := by sorry

/-- The probability of all backpacks being picked up incorrectly -/
theorem prob_all_incorrect : 
  all_incorrect / total_outcomes = 3 / 8 := by sorry

end NUMINAMATH_CALUDE_prob_exactly_two_correct_prob_at_least_two_correct_prob_all_incorrect_l3910_391030


namespace NUMINAMATH_CALUDE_simplify_sqrt_fraction_l3910_391068

theorem simplify_sqrt_fraction : 
  (Real.sqrt 462 / Real.sqrt 330) + (Real.sqrt 245 / Real.sqrt 175) = 12 * Real.sqrt 35 / 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_fraction_l3910_391068


namespace NUMINAMATH_CALUDE_product_from_hcf_lcm_l3910_391049

theorem product_from_hcf_lcm (A B : ℕ+) :
  Nat.gcd A B = 22 →
  Nat.lcm A B = 2828 →
  A * B = 62216 := by
  sorry

end NUMINAMATH_CALUDE_product_from_hcf_lcm_l3910_391049


namespace NUMINAMATH_CALUDE_power_multiplication_l3910_391094

theorem power_multiplication (a b : ℕ) : (10 : ℝ) ^ a * (10 : ℝ) ^ b = (10 : ℝ) ^ (a + b) := by
  sorry

end NUMINAMATH_CALUDE_power_multiplication_l3910_391094


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l3910_391024

theorem algebraic_expression_value (x : ℝ) (h : x * (x + 2) = 2023) :
  2 * (x + 3) * (x - 1) - 2018 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l3910_391024


namespace NUMINAMATH_CALUDE_johns_calculation_l3910_391098

theorem johns_calculation (x : ℝ) : 
  Real.sqrt x - 20 = 15 → x^2 + 20 = 1500645 := by
  sorry

end NUMINAMATH_CALUDE_johns_calculation_l3910_391098


namespace NUMINAMATH_CALUDE_probability_at_least_one_mistake_l3910_391014

-- Define the probability of making a mistake on a single question
def p_mistake : ℝ := 0.1

-- Define the number of questions
def n_questions : ℕ := 3

-- Theorem statement
theorem probability_at_least_one_mistake :
  1 - (1 - p_mistake) ^ n_questions = 1 - 0.9 ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_mistake_l3910_391014


namespace NUMINAMATH_CALUDE_min_value_trigonometric_expression_l3910_391044

theorem min_value_trigonometric_expression (α β : ℝ) :
  (3 * Real.cos α + 4 * Real.sin β - 10)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2 ≥ 215 := by
  sorry

end NUMINAMATH_CALUDE_min_value_trigonometric_expression_l3910_391044


namespace NUMINAMATH_CALUDE_sum_of_fractions_zero_l3910_391043

theorem sum_of_fractions_zero (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) (h_sum : a + b + c = d) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_zero_l3910_391043


namespace NUMINAMATH_CALUDE_max_annual_average_profit_l3910_391008

def profit_function (x : ℕ+) : ℚ := -x^2 + 18*x - 25

def annual_average_profit (x : ℕ+) : ℚ := (profit_function x) / x

theorem max_annual_average_profit :
  ∃ (x : ℕ+), (∀ (y : ℕ+), annual_average_profit y ≤ annual_average_profit x) ∧
              x = 5 ∧
              annual_average_profit x = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_annual_average_profit_l3910_391008


namespace NUMINAMATH_CALUDE_a_pow_b_gt_one_iff_a_minus_one_b_gt_zero_l3910_391088

theorem a_pow_b_gt_one_iff_a_minus_one_b_gt_zero 
  (a b : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) : 
  a^b > 1 ↔ (a - 1) * b > 0 := by sorry

end NUMINAMATH_CALUDE_a_pow_b_gt_one_iff_a_minus_one_b_gt_zero_l3910_391088


namespace NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l3910_391017

theorem floor_of_expression_equals_eight :
  ⌊(2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023)⌋ = 8 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_expression_equals_eight_l3910_391017


namespace NUMINAMATH_CALUDE_curve_properties_l3910_391045

-- Define the curve y = ax^3 + bx
def curve (a b x : ℝ) : ℝ := a * x^3 + b * x

-- Define the derivative of the curve
def curve_derivative (a b x : ℝ) : ℝ := 3 * a * x^2 + b

theorem curve_properties (a b : ℝ) :
  curve a b 2 = 2 ∧ 
  curve_derivative a b 2 = 9 →
  a * b = -3 ∧
  Set.Icc (-3/2 : ℝ) 3 ⊆ Set.Icc (-2 : ℝ) 18 ∧
  ∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc (-3/2 : ℝ) 3 ∧ 
             x₂ ∈ Set.Icc (-3/2 : ℝ) 3 ∧
             curve a b x₁ = -2 ∧
             curve a b x₂ = 18 := by
  sorry

end NUMINAMATH_CALUDE_curve_properties_l3910_391045


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3910_391072

/-- Represents a repeating decimal with a single repeating digit -/
def repeating_decimal_single (n : ℕ) : ℚ := n / 9

/-- Represents a repeating decimal with two repeating digits -/
def repeating_decimal_double (n : ℕ) : ℚ := n / 99

theorem sum_of_repeating_decimals : 
  repeating_decimal_single 6 + repeating_decimal_double 45 = 37 / 33 := by sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l3910_391072


namespace NUMINAMATH_CALUDE_chris_win_probability_l3910_391035

theorem chris_win_probability :
  let chris_head_prob : ℝ := 1/4
  let drew_head_prob : ℝ := 1/3
  let both_tail_prob : ℝ := (1 - chris_head_prob) * (1 - drew_head_prob)
  chris_head_prob / (1 - both_tail_prob) = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_chris_win_probability_l3910_391035


namespace NUMINAMATH_CALUDE_sand_overflow_l3910_391093

/-- Represents the capacity of a bucket -/
structure BucketCapacity where
  value : ℚ
  positive : 0 < value

/-- Represents the amount of sand in a bucket -/
structure SandAmount where
  amount : ℚ
  nonnegative : 0 ≤ amount

/-- Theorem stating the overflow amount when pouring sand between buckets -/
theorem sand_overflow
  (CA : BucketCapacity) -- Capacity of Bucket A
  (sand_A : SandAmount) -- Initial sand in Bucket A
  (sand_B : SandAmount) -- Initial sand in Bucket B
  (sand_C : SandAmount) -- Initial sand in Bucket C
  (h1 : sand_A.amount = (1 : ℚ) / 4 * CA.value) -- Bucket A is 1/4 full
  (h2 : sand_B.amount = (3 : ℚ) / 8 * (CA.value / 2)) -- Bucket B is 3/8 full
  (h3 : sand_C.amount = (1 : ℚ) / 3 * (2 * CA.value)) -- Bucket C is 1/3 full
  : ∃ (overflow : ℚ), overflow = (17 : ℚ) / 48 * CA.value :=
by sorry

end NUMINAMATH_CALUDE_sand_overflow_l3910_391093


namespace NUMINAMATH_CALUDE_min_cost_plan_l3910_391097

/-- Represents the production plan for student desks and chairs -/
structure ProductionPlan where
  typeA : ℕ  -- Number of type A sets
  typeB : ℕ  -- Number of type B sets

/-- Calculates the total cost of a production plan -/
def totalCost (plan : ProductionPlan) : ℕ :=
  102 * plan.typeA + 124 * plan.typeB

/-- Checks if a production plan is valid according to the given constraints -/
def isValidPlan (plan : ProductionPlan) : Prop :=
  plan.typeA + plan.typeB = 500 ∧
  2 * plan.typeA + 3 * plan.typeB ≥ 1250 ∧
  5 * plan.typeA + 7 * plan.typeB ≤ 3020

/-- Theorem stating that the minimum total cost is achieved with 250 sets of each type -/
theorem min_cost_plan :
  ∀ (plan : ProductionPlan),
    isValidPlan plan →
    totalCost plan ≥ 56500 ∧
    (totalCost plan = 56500 ↔ plan.typeA = 250 ∧ plan.typeB = 250) := by
  sorry


end NUMINAMATH_CALUDE_min_cost_plan_l3910_391097


namespace NUMINAMATH_CALUDE_union_of_A_and_B_range_of_a_l3910_391047

-- Define sets A and B
def A (a : ℝ) := {x : ℝ | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B := {x : ℝ | -1/2 < x ∧ x ≤ 2}

-- Part I
theorem union_of_A_and_B (a : ℝ) (h : a = 1) :
  A a ∪ B = {x : ℝ | -1/2 < x ∧ x ≤ 6} := by sorry

-- Part II
theorem range_of_a (a : ℝ) (h1 : A a ∩ B = ∅) (h2 : a > 0) :
  0 < a ∧ a ≤ 1/2 := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_range_of_a_l3910_391047


namespace NUMINAMATH_CALUDE_no_solution_for_part_a_unique_solution_for_part_b_l3910_391069

-- Define S(x) as the sum of digits of a natural number
def S (x : ℕ) : ℕ := sorry

-- Theorem for part (a)
theorem no_solution_for_part_a :
  ¬ ∃ x : ℕ, x + S x + S (S x) = 1993 := by sorry

-- Theorem for part (b)
theorem unique_solution_for_part_b :
  ∃! x : ℕ, x + S x + S (S x) + S (S (S x)) = 1993 ∧ x = 1963 := by sorry

end NUMINAMATH_CALUDE_no_solution_for_part_a_unique_solution_for_part_b_l3910_391069


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3910_391064

def point : ℝ × ℝ := (-2, 3)

def is_in_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem point_in_second_quadrant : is_in_second_quadrant point := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3910_391064


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3910_391003

theorem right_triangle_hypotenuse (x : ℚ) :
  let a := 9
  let b := 3 * x + 6
  let c := x + 15
  (a + b + c = 45) →
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) →
  (max a (max b c) = 75 / 4) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l3910_391003


namespace NUMINAMATH_CALUDE_count_seven_digit_phone_numbers_l3910_391041

/-- The number of different seven-digit phone numbers where the first digit cannot be zero -/
def sevenDigitPhoneNumbers : ℕ := 9 * (10 ^ 6)

/-- Theorem stating that the number of different seven-digit phone numbers
    where the first digit cannot be zero is equal to 9 * 10^6 -/
theorem count_seven_digit_phone_numbers :
  sevenDigitPhoneNumbers = 9 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_count_seven_digit_phone_numbers_l3910_391041


namespace NUMINAMATH_CALUDE_farm_area_calculation_l3910_391006

/-- The total area of a farm with given sections and section area -/
def farm_total_area (num_sections : ℕ) (section_area : ℕ) : ℕ :=
  num_sections * section_area

/-- Theorem: The total area of a farm with 5 sections of 60 acres each is 300 acres -/
theorem farm_area_calculation :
  farm_total_area 5 60 = 300 := by
  sorry

end NUMINAMATH_CALUDE_farm_area_calculation_l3910_391006


namespace NUMINAMATH_CALUDE_quarters_spent_at_arcade_l3910_391055

theorem quarters_spent_at_arcade (initial_quarters : ℕ) (remaining_quarters : ℕ) 
  (h1 : initial_quarters = 88) 
  (h2 : remaining_quarters = 79) : 
  initial_quarters - remaining_quarters = 9 := by
  sorry

end NUMINAMATH_CALUDE_quarters_spent_at_arcade_l3910_391055


namespace NUMINAMATH_CALUDE_vacation_cost_is_120_l3910_391083

/-- Calculates the total cost of a vacation for two people. -/
def vacationCost (planeTicketCost hotelCostPerDay : ℕ) (durationInDays : ℕ) : ℕ :=
  2 * planeTicketCost + 2 * hotelCostPerDay * durationInDays

/-- Proves that the total cost of the vacation is $120. -/
theorem vacation_cost_is_120 :
  vacationCost 24 12 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_vacation_cost_is_120_l3910_391083


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l3910_391075

/-- An arithmetic sequence where each term is not 0 and satisfies a specific condition -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n ≠ 0) ∧
  (∃ d, ∀ n, a (n + 1) = a n + d) ∧
  a 6 - (a 7)^2 + a 8 = 0

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r, ∀ n, b (n + 1) = r * b n

/-- The main theorem -/
theorem arithmetic_geometric_sequence_product
  (a : ℕ → ℝ) (b : ℕ → ℝ)
  (ha : ArithmeticSequence a)
  (hb : GeometricSequence b)
  (h_equal : b 7 = a 7) :
  b 4 * b 7 * b 10 = 8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_product_l3910_391075


namespace NUMINAMATH_CALUDE_teal_color_survey_l3910_391013

theorem teal_color_survey (total : ℕ) (more_blue : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 150)
  (h_more_blue : more_blue = 90)
  (h_both : both = 45)
  (h_neither : neither = 20) :
  ∃ (more_green : ℕ), more_green = 85 ∧ 
    total = more_blue + more_green - both + neither :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l3910_391013


namespace NUMINAMATH_CALUDE_white_squares_47th_row_l3910_391092

/-- Represents the number of squares in a row of the stair-step figure -/
def totalSquares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the number of white squares in a row of the stair-step figure -/
def whiteSquares (n : ℕ) : ℕ := (totalSquares n - 1) / 2

/-- Theorem stating that the 47th row of the stair-step figure contains 46 white squares -/
theorem white_squares_47th_row :
  whiteSquares 47 = 46 := by
  sorry


end NUMINAMATH_CALUDE_white_squares_47th_row_l3910_391092


namespace NUMINAMATH_CALUDE_range_of_a_l3910_391031

/-- The set A defined by the equation x^2 + 4x = 0 -/
def A : Set ℝ := {x | x^2 + 4*x = 0}

/-- The set B defined by the equation x^2 + ax + a = 0, where a is a parameter -/
def B (a : ℝ) : Set ℝ := {x | x^2 + a*x + a = 0}

/-- The theorem stating the range of a given the conditions -/
theorem range_of_a (a : ℝ) : A ∪ B a = A ↔ 0 ≤ a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3910_391031


namespace NUMINAMATH_CALUDE_last_day_arrangement_l3910_391067

/-- The number of vases Jane can arrange in a day -/
def vases_per_day : ℕ := 16

/-- The total number of vases to be arranged -/
def total_vases : ℕ := 248

/-- The number of vases arranged on the last day -/
def last_day_vases : ℕ := total_vases % vases_per_day

theorem last_day_arrangement :
  last_day_vases = 8 :=
sorry

end NUMINAMATH_CALUDE_last_day_arrangement_l3910_391067


namespace NUMINAMATH_CALUDE_invariant_parity_and_final_digit_l3910_391019

/-- Represents the count of each digit -/
structure DigitCounts where
  zeros : ℕ
  ones : ℕ
  twos : ℕ

/-- Represents the possible operations on the board -/
inductive Operation
  | replaceZeroOne
  | replaceOneTwo
  | replaceZeroTwo

/-- Applies an operation to the digit counts -/
def applyOperation (counts : DigitCounts) (op : Operation) : DigitCounts :=
  match op with
  | Operation.replaceZeroOne => ⟨counts.zeros - 1, counts.ones - 1, counts.twos + 1⟩
  | Operation.replaceOneTwo => ⟨counts.zeros + 1, counts.ones - 1, counts.twos - 1⟩
  | Operation.replaceZeroTwo => ⟨counts.zeros - 1, counts.ones + 1, counts.twos - 1⟩

/-- The parity of the sum of digit counts -/
def sumParity (counts : DigitCounts) : ℕ :=
  (counts.zeros + counts.ones + counts.twos) % 2

/-- The final remaining digit -/
def finalDigit (initialCounts : DigitCounts) : ℕ :=
  if initialCounts.zeros % 2 ≠ initialCounts.ones % 2 ∧ initialCounts.zeros % 2 ≠ initialCounts.twos % 2 then 0
  else if initialCounts.ones % 2 ≠ initialCounts.zeros % 2 ∧ initialCounts.ones % 2 ≠ initialCounts.twos % 2 then 1
  else 2

theorem invariant_parity_and_final_digit (initialCounts : DigitCounts) (ops : List Operation) :
  (sumParity initialCounts = sumParity (ops.foldl applyOperation initialCounts)) ∧
  (finalDigit initialCounts = finalDigit (ops.foldl applyOperation initialCounts)) :=
sorry

end NUMINAMATH_CALUDE_invariant_parity_and_final_digit_l3910_391019


namespace NUMINAMATH_CALUDE_smallest_n_satisfying_ratio_l3910_391053

def sum_odd_from_3 (n : ℕ) : ℕ := n^2 + 2*n

def sum_even (n : ℕ) : ℕ := n*(n+1)

theorem smallest_n_satisfying_ratio : 
  ∀ n : ℕ, n > 0 → (n < 51 → (sum_odd_from_3 n : ℚ) / sum_even n ≠ 49/50) ∧
  (sum_odd_from_3 51 : ℚ) / sum_even 51 = 49/50 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_satisfying_ratio_l3910_391053


namespace NUMINAMATH_CALUDE_sin_cos_sum_ratio_equals_tan_60_l3910_391077

theorem sin_cos_sum_ratio_equals_tan_60 :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) := by sorry

end NUMINAMATH_CALUDE_sin_cos_sum_ratio_equals_tan_60_l3910_391077


namespace NUMINAMATH_CALUDE_bacon_calories_per_strip_l3910_391034

theorem bacon_calories_per_strip 
  (total_calories : ℕ) 
  (bacon_percentage : ℚ) 
  (num_bacon_strips : ℕ) 
  (h1 : total_calories = 1250)
  (h2 : bacon_percentage = 1/5)
  (h3 : num_bacon_strips = 2) :
  (total_calories : ℚ) * bacon_percentage / num_bacon_strips = 125 := by
sorry

end NUMINAMATH_CALUDE_bacon_calories_per_strip_l3910_391034


namespace NUMINAMATH_CALUDE_record_storage_space_theorem_l3910_391020

/-- Represents the dimensions of a box in inches -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

theorem record_storage_space_theorem 
  (box_dims : BoxDimensions)
  (storage_cost_per_box : ℝ)
  (total_monthly_payment : ℝ)
  (h1 : box_dims.length = 15)
  (h2 : box_dims.width = 12)
  (h3 : box_dims.height = 10)
  (h4 : storage_cost_per_box = 0.2)
  (h5 : total_monthly_payment = 120) :
  (total_monthly_payment / storage_cost_per_box) * boxVolume box_dims = 1080000 := by
  sorry

#check record_storage_space_theorem

end NUMINAMATH_CALUDE_record_storage_space_theorem_l3910_391020


namespace NUMINAMATH_CALUDE_unique_solution_xyz_squared_l3910_391087

theorem unique_solution_xyz_squared (x y z : ℕ) :
  x^2 + y^2 + z^2 = 2*x*y*z → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_xyz_squared_l3910_391087


namespace NUMINAMATH_CALUDE_tan_150_degrees_l3910_391096

theorem tan_150_degrees :
  Real.tan (150 * π / 180) = -1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l3910_391096


namespace NUMINAMATH_CALUDE_remainder_theorem_l3910_391027

theorem remainder_theorem (n : ℤ) (h : n % 5 = 2) : (n + 2023) % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3910_391027


namespace NUMINAMATH_CALUDE_special_square_area_l3910_391086

/-- A square with one side on a line and two vertices on a parabola -/
structure SpecialSquare where
  /-- The y-coordinate of vertex C -/
  y1 : ℝ
  /-- The y-coordinate of vertex D -/
  y2 : ℝ
  /-- C and D lie on the parabola y^2 = x -/
  h1 : y1^2 = (y1 : ℝ)
  h2 : y2^2 = (y2 : ℝ)
  /-- Side AB lies on the line y = x + 4 -/
  h3 : y2^2 - y1^2 + y1 = y1^2 + y1 - y2 + 4
  /-- The slope condition -/
  h4 : y1 - y2 = y1^2 - y2^2

/-- The area of a SpecialSquare is either 18 or 50 -/
theorem special_square_area (s : SpecialSquare) : (s.y2 - s.y1)^2 = 18 ∨ (s.y2 - s.y1)^2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_special_square_area_l3910_391086


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3910_391016

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  4 * x^2 - 16 * x - 16 * y^2 + 32 * y + 144 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance (eq : (ℝ → ℝ → Prop)) : ℝ :=
  sorry

theorem hyperbola_vertex_distance :
  vertex_distance hyperbola_eq = 12 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l3910_391016


namespace NUMINAMATH_CALUDE_fraction_inequality_l3910_391085

theorem fraction_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  c / a < c / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_l3910_391085


namespace NUMINAMATH_CALUDE_max_silver_tokens_l3910_391078

/-- Represents the number of tokens of each color --/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents the exchange rules --/
inductive ExchangeRule
  | RedToSilver
  | BlueToSilver
  | BothToSilver

/-- Applies an exchange rule to a token count --/
def applyExchange (tc : TokenCount) (rule : ExchangeRule) : TokenCount :=
  match rule with
  | ExchangeRule.RedToSilver => 
      { red := tc.red - 4, blue := tc.blue + 1, silver := tc.silver + 2 }
  | ExchangeRule.BlueToSilver => 
      { red := tc.red + 1, blue := tc.blue - 5, silver := tc.silver + 2 }
  | ExchangeRule.BothToSilver => 
      { red := tc.red - 3, blue := tc.blue - 3, silver := tc.silver + 3 }

/-- Checks if an exchange is possible --/
def canExchange (tc : TokenCount) (rule : ExchangeRule) : Prop :=
  match rule with
  | ExchangeRule.RedToSilver => tc.red ≥ 4
  | ExchangeRule.BlueToSilver => tc.blue ≥ 5
  | ExchangeRule.BothToSilver => tc.red ≥ 3 ∧ tc.blue ≥ 3

/-- The main theorem --/
theorem max_silver_tokens : 
  ∃ (final : TokenCount), 
    (∃ (exchanges : List ExchangeRule), 
      final = exchanges.foldl applyExchange { red := 100, blue := 100, silver := 0 } ∧
      ∀ rule, ¬(canExchange final rule)) ∧
    final.silver = 85 :=
  sorry


end NUMINAMATH_CALUDE_max_silver_tokens_l3910_391078


namespace NUMINAMATH_CALUDE_probability_of_white_after_red_l3910_391061

/-- Represents the number of balls in the box -/
def total_balls : ℕ := 20

/-- Represents the initial number of red balls -/
def initial_red_balls : ℕ := 10

/-- Represents the initial number of white balls -/
def initial_white_balls : ℕ := 10

/-- Represents that the first person draws a red ball -/
def first_draw_red : Prop := true

/-- The probability of drawing a white ball after a red ball is drawn -/
def prob_white_after_red : ℚ := 10 / 19

theorem probability_of_white_after_red :
  first_draw_red →
  prob_white_after_red = initial_white_balls / (total_balls - 1) :=
by sorry

end NUMINAMATH_CALUDE_probability_of_white_after_red_l3910_391061


namespace NUMINAMATH_CALUDE_interest_calculation_l3910_391089

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_calculation :
  let principal : ℝ := 10000
  let rate : ℝ := 0.05
  let time : ℝ := 1
  simple_interest principal rate time = 500 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l3910_391089


namespace NUMINAMATH_CALUDE_max_profit_toy_sales_exists_max_profit_price_l3910_391081

/-- Represents the profit function for toy sales -/
def profit_function (x : ℝ) : ℝ := -10 * x^2 + 1300 * x - 30000

/-- Represents the sales volume function for toy sales -/
def sales_volume (x : ℝ) : ℝ := -10 * x + 1000

/-- The maximum profit theorem for toy sales -/
theorem max_profit_toy_sales :
  ∀ x : ℝ,
  (x ≥ 44) →
  (x ≤ 46) →
  (sales_volume x ≥ 540) →
  profit_function x ≤ 8640 :=
by
  sorry

/-- The existence of a selling price that achieves the maximum profit -/
theorem exists_max_profit_price :
  ∃ x : ℝ,
  (x ≥ 44) ∧
  (x ≤ 46) ∧
  (sales_volume x ≥ 540) ∧
  profit_function x = 8640 :=
by
  sorry

end NUMINAMATH_CALUDE_max_profit_toy_sales_exists_max_profit_price_l3910_391081


namespace NUMINAMATH_CALUDE_circle_intersection_range_l3910_391038

-- Define the circle C
def circle_C (a x y : ℝ) : Prop :=
  (x - a - 1)^2 + (y - Real.sqrt 3 * a)^2 = 1

-- Define the condition |MA| = 2|MO|
def condition_M (x y : ℝ) : Prop :=
  (x + 3)^2 + y^2 = 4 * (x^2 + y^2)

-- Define the range of a
def range_a (a : ℝ) : Prop :=
  (1/2 ≤ a ∧ a ≤ 3/2) ∨ (-3/2 ≤ a ∧ a ≤ -1/2)

theorem circle_intersection_range :
  ∀ a : ℝ, (∃ x y : ℝ, circle_C a x y ∧ condition_M x y) ↔ range_a a :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_range_l3910_391038


namespace NUMINAMATH_CALUDE_investment_scientific_notation_l3910_391005

/-- Represents the scientific notation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Converts a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

/-- The total industrial investment in yuan -/
def total_investment : ℝ := 314.86 * 10^9

theorem investment_scientific_notation :
  to_scientific_notation total_investment = ScientificNotation.mk 3.1486 10 sorry := by
  sorry

end NUMINAMATH_CALUDE_investment_scientific_notation_l3910_391005


namespace NUMINAMATH_CALUDE_mikes_marbles_l3910_391048

/-- Given that Mike has 8 orange marbles initially and gives away 4 marbles,
    prove that he will have 4 orange marbles remaining. -/
theorem mikes_marbles (initial_marbles : ℕ) (marbles_given : ℕ) (remaining_marbles : ℕ) :
  initial_marbles = 8 →
  marbles_given = 4 →
  remaining_marbles = initial_marbles - marbles_given →
  remaining_marbles = 4 := by
sorry

end NUMINAMATH_CALUDE_mikes_marbles_l3910_391048


namespace NUMINAMATH_CALUDE_pasture_rental_problem_l3910_391007

/-- The pasture rental problem -/
theorem pasture_rental_problem 
  (total_cost : ℕ) 
  (a_horses b_horses c_horses : ℕ) 
  (b_months c_months : ℕ) 
  (b_payment : ℕ) 
  (h_total_cost : total_cost = 870)
  (h_a_horses : a_horses = 12)
  (h_b_horses : b_horses = 16)
  (h_c_horses : c_horses = 18)
  (h_b_months : b_months = 9)
  (h_c_months : c_months = 6)
  (h_b_payment : b_payment = 360)
  : ∃ (a_months : ℕ), 
    a_horses * a_months * b_payment = b_horses * b_months * (total_cost - b_payment - c_horses * c_months * b_payment / (b_horses * b_months)) ∧ 
    a_months = 8 :=
by sorry

end NUMINAMATH_CALUDE_pasture_rental_problem_l3910_391007


namespace NUMINAMATH_CALUDE_latest_90_degrees_time_l3910_391037

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 14*t + 40

-- Define the theorem
theorem latest_90_degrees_time :
  ∃ t : ℝ, t ≤ 17 ∧ temperature t = 90 ∧
  ∀ s : ℝ, s > 17 → temperature s ≠ 90 :=
by sorry

end NUMINAMATH_CALUDE_latest_90_degrees_time_l3910_391037


namespace NUMINAMATH_CALUDE_gcd_24_36_54_l3910_391073

theorem gcd_24_36_54 : Nat.gcd 24 (Nat.gcd 36 54) = 6 := by sorry

end NUMINAMATH_CALUDE_gcd_24_36_54_l3910_391073


namespace NUMINAMATH_CALUDE_probability_not_red_is_three_fifths_l3910_391091

/-- Represents the duration of each traffic light color in seconds -/
structure TrafficLightDurations where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the probability of not seeing the red light -/
def probability_not_red (d : TrafficLightDurations) : ℚ :=
  (d.yellow + d.green : ℚ) / (d.red + d.yellow + d.green)

/-- Theorem stating the probability of not seeing the red light is 3/5 -/
theorem probability_not_red_is_three_fifths :
  let d : TrafficLightDurations := ⟨30, 5, 40⟩
  probability_not_red d = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_not_red_is_three_fifths_l3910_391091


namespace NUMINAMATH_CALUDE_success_arrangements_l3910_391021

-- Define the total number of letters
def total_letters : ℕ := 7

-- Define the repetitions of each letter
def s_count : ℕ := 3
def c_count : ℕ := 2
def u_count : ℕ := 1
def e_count : ℕ := 1

-- Define the function to calculate the number of arrangements
def arrangements : ℕ := total_letters.factorial / (s_count.factorial * c_count.factorial * u_count.factorial * e_count.factorial)

-- State the theorem
theorem success_arrangements : arrangements = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_arrangements_l3910_391021


namespace NUMINAMATH_CALUDE_bertha_family_without_children_l3910_391058

/-- Represents Bertha's family structure -/
structure BerthaFamily where
  daughters : ℕ
  total_descendants : ℕ
  daughters_with_children : ℕ

/-- The number of Bertha's daughters and granddaughters without daughters -/
def daughters_without_children (f : BerthaFamily) : ℕ :=
  f.total_descendants - f.daughters_with_children

/-- Theorem stating the number of Bertha's daughters and granddaughters without daughters -/
theorem bertha_family_without_children (f : BerthaFamily) 
  (h1 : f.daughters = 5)
  (h2 : f.total_descendants = 25)
  (h3 : f.daughters_with_children * 5 = f.total_descendants - f.daughters) :
  daughters_without_children f = 21 := by
  sorry

#check bertha_family_without_children

end NUMINAMATH_CALUDE_bertha_family_without_children_l3910_391058


namespace NUMINAMATH_CALUDE_sphere_circular_views_l3910_391004

-- Define a type for geometric bodies
inductive GeometricBody
  | Cone
  | Sphere
  | Cylinder
  | HollowCylinder

-- Define a function to check if a view is circular
def isCircularView (body : GeometricBody) (view : String) : Prop :=
  match body, view with
  | GeometricBody.Sphere, _ => True
  | _, _ => False

-- Main theorem
theorem sphere_circular_views :
  ∀ (body : GeometricBody),
    (isCircularView body "main" ∧
     isCircularView body "left" ∧
     isCircularView body "top") →
    body = GeometricBody.Sphere :=
by sorry

end NUMINAMATH_CALUDE_sphere_circular_views_l3910_391004


namespace NUMINAMATH_CALUDE_divisibility_by_nine_l3910_391071

theorem divisibility_by_nine (n : ℕ) (h : 900 ≤ n ∧ n ≤ 999) : 
  (n % 9 = 0) ↔ ((n / 100 + (n / 10) % 10 + n % 10) % 9 = 0) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_nine_l3910_391071


namespace NUMINAMATH_CALUDE_percentage_decrease_in_people_l3910_391090

/-- Calculates the percentage decrease in the number of people to be fed given initial and new can counts. -/
theorem percentage_decrease_in_people (initial_cans initial_people new_cans : ℕ) : 
  initial_cans = 600 →
  initial_people = 40 →
  new_cans = 420 →
  (1 - (new_cans * initial_people : ℚ) / (initial_cans * initial_people)) * 100 = 30 := by
sorry

end NUMINAMATH_CALUDE_percentage_decrease_in_people_l3910_391090


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3910_391060

/-- Given a quadratic inequality ax^2 + (ab+1)x + b > 0 with solution set {x | 1 < x < 3},
    prove that a + b = -4 or a + b = -4/3 -/
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + (a*b + 1)*x + b > 0 ↔ 1 < x ∧ x < 3) →
  (a + b = -4 ∨ a + b = -4/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3910_391060


namespace NUMINAMATH_CALUDE_new_average_weight_l3910_391070

def original_team_size : ℕ := 7
def original_average_weight : ℝ := 121
def new_player1_weight : ℝ := 110
def new_player2_weight : ℝ := 60

theorem new_average_weight :
  let total_original_weight : ℝ := original_team_size * original_average_weight
  let new_total_weight : ℝ := total_original_weight + new_player1_weight + new_player2_weight
  let new_team_size : ℕ := original_team_size + 2
  (new_total_weight / new_team_size : ℝ) = 113 := by sorry

end NUMINAMATH_CALUDE_new_average_weight_l3910_391070


namespace NUMINAMATH_CALUDE_sum_of_integers_l3910_391023

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 18) (h2 : x.val * y.val = 98) :
  x.val + y.val = 2 * Real.sqrt 179 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3910_391023


namespace NUMINAMATH_CALUDE_total_nailcutter_sounds_l3910_391000

/-- The number of nails per person (fingers and toes combined) -/
def nails_per_person : ℕ := 20

/-- The number of customers -/
def num_customers : ℕ := 3

/-- The number of sounds produced when trimming one nail -/
def sounds_per_nail : ℕ := 1

/-- Theorem: The total number of nailcutter sounds for 3 customers is 60 -/
theorem total_nailcutter_sounds :
  nails_per_person * num_customers * sounds_per_nail = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_nailcutter_sounds_l3910_391000


namespace NUMINAMATH_CALUDE_string_cutting_problem_l3910_391056

theorem string_cutting_problem (s l : ℝ) (h1 : s > 0) (h2 : l > 0) 
  (h3 : l - s = 48) (h4 : l + s = 64) : l / s = 7 := by
  sorry

end NUMINAMATH_CALUDE_string_cutting_problem_l3910_391056


namespace NUMINAMATH_CALUDE_equation_solution_l3910_391063

theorem equation_solution : ∃ y : ℝ, y = (18 : ℝ) / 4 ∧ (8 * y^2 + 50 * y + 3) / (4 * y + 21) = 2 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3910_391063


namespace NUMINAMATH_CALUDE_eraser_cost_l3910_391074

theorem eraser_cost (total_students : Nat) (buyers : Nat) (erasers_per_student : Nat) (total_cost : Nat) :
  total_students = 36 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  erasers_per_student > 2 →
  total_cost = 3978 →
  ∃ (cost : Nat), cost > erasers_per_student ∧
                  buyers * erasers_per_student * cost = total_cost ∧
                  cost = 17 :=
by sorry

end NUMINAMATH_CALUDE_eraser_cost_l3910_391074
