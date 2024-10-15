import Mathlib

namespace NUMINAMATH_CALUDE_xu_shou_achievements_l120_12076

/-- Represents a historical figure in Chinese science and technology -/
structure HistoricalFigure where
  name : String

/-- Represents a scientific achievement -/
inductive Achievement
  | SteamEngine
  | RiverSteamer
  | ChemicalTranslationPrinciples
  | ElementTranslations

/-- Predicate to check if a historical figure accomplished a given achievement in a specific year -/
def accomplished (person : HistoricalFigure) (achievement : Achievement) (year : ℕ) : Prop :=
  match achievement with
  | Achievement.SteamEngine => person.name = "Xu Shou" ∧ year = 1863
  | Achievement.RiverSteamer => person.name = "Xu Shou"
  | Achievement.ChemicalTranslationPrinciples => person.name = "Xu Shou"
  | Achievement.ElementTranslations => person.name = "Xu Shou" ∧ ∃ n : ℕ, n = 36

/-- Theorem stating that Xu Shou accomplished all the mentioned achievements -/
theorem xu_shou_achievements (xu_shou : HistoricalFigure) 
    (h_name : xu_shou.name = "Xu Shou") :
    accomplished xu_shou Achievement.SteamEngine 1863 ∧
    accomplished xu_shou Achievement.RiverSteamer 0 ∧
    accomplished xu_shou Achievement.ChemicalTranslationPrinciples 0 ∧
    accomplished xu_shou Achievement.ElementTranslations 0 :=
  sorry

end NUMINAMATH_CALUDE_xu_shou_achievements_l120_12076


namespace NUMINAMATH_CALUDE_sine_function_parameters_l120_12052

theorem sine_function_parameters (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c < 0) :
  (∀ x, a * Real.sin (b * x) + c ≤ 3) ∧
  (∃ x, a * Real.sin (b * x) + c = 3) ∧
  (∀ x, a * Real.sin (b * x) + c ≥ -5) ∧
  (∃ x, a * Real.sin (b * x) + c = -5) →
  a = 4 ∧ c = -1 := by
sorry

end NUMINAMATH_CALUDE_sine_function_parameters_l120_12052


namespace NUMINAMATH_CALUDE_product_equals_half_l120_12070

theorem product_equals_half : 8 * 0.25 * 2 * 0.125 = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_half_l120_12070


namespace NUMINAMATH_CALUDE_contest_end_time_l120_12073

def contest_start : Nat := 15 * 60  -- 3:00 p.m. in minutes since midnight
def contest_duration : Nat := 850   -- total duration in minutes
def break_duration : Nat := 30      -- break duration in minutes

def minutes_in_day : Nat := 24 * 60 -- number of minutes in a day

def contest_end : Nat :=
  (contest_start + contest_duration - break_duration) % minutes_in_day

theorem contest_end_time :
  contest_end = 4 * 60 + 40 := by sorry

end NUMINAMATH_CALUDE_contest_end_time_l120_12073


namespace NUMINAMATH_CALUDE_always_even_expression_l120_12003

theorem always_even_expression (x y : ℕ) : 
  x ∈ Finset.range 15 → 
  y ∈ Finset.range 15 → 
  x ≠ y → 
  Even (x * y - 2 * x - 2 * y) := by
  sorry

#check always_even_expression

end NUMINAMATH_CALUDE_always_even_expression_l120_12003


namespace NUMINAMATH_CALUDE_gcd_of_72_120_168_l120_12004

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_72_120_168_l120_12004


namespace NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l120_12088

/-- The set of digits used to form the numbers -/
def digits : Finset Nat := {1, 2, 3, 4, 5}

/-- A four-digit number formed from the given digits -/
structure FourDigitNumber where
  d₁ : Nat
  d₂ : Nat
  d₃ : Nat
  d₄ : Nat
  h₁ : d₁ ∈ digits
  h₂ : d₂ ∈ digits
  h₃ : d₃ ∈ digits
  h₄ : d₄ ∈ digits
  distinct : d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₁ ≠ d₄ ∧ d₂ ≠ d₃ ∧ d₂ ≠ d₄ ∧ d₃ ≠ d₄

/-- The value of a four-digit number -/
def value (n : FourDigitNumber) : Nat :=
  1000 * n.d₁ + 100 * n.d₂ + 10 * n.d₃ + n.d₄

/-- The set of all valid four-digit numbers -/
def allFourDigitNumbers : Finset FourDigitNumber :=
  sorry

theorem sum_of_four_digit_numbers :
  (allFourDigitNumbers.sum value) = 399960 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_digit_numbers_l120_12088


namespace NUMINAMATH_CALUDE_device_working_prob_correct_l120_12047

/-- A device with two components, each having a probability of failure --/
structure Device where
  /-- The probability of a single component being damaged --/
  component_failure_prob : ℝ
  /-- Assumption that the component failure probability is between 0 and 1 --/
  h_prob_range : 0 ≤ component_failure_prob ∧ component_failure_prob ≤ 1

/-- The probability of the device working --/
def device_working_prob (d : Device) : ℝ :=
  (1 - d.component_failure_prob) * (1 - d.component_failure_prob)

/-- Theorem stating that for a device with component failure probability of 0.1,
    the probability of the device working is 0.81 --/
theorem device_working_prob_correct (d : Device) 
    (h : d.component_failure_prob = 0.1) : 
    device_working_prob d = 0.81 := by
  sorry

end NUMINAMATH_CALUDE_device_working_prob_correct_l120_12047


namespace NUMINAMATH_CALUDE_basketball_game_price_l120_12098

/-- The cost of Joan's video game purchase -/
def total_cost : ℝ := 9.43

/-- The cost of the racing game -/
def racing_game_cost : ℝ := 4.23

/-- The cost of the basketball game -/
def basketball_game_cost : ℝ := total_cost - racing_game_cost

theorem basketball_game_price : basketball_game_cost = 5.20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_price_l120_12098


namespace NUMINAMATH_CALUDE_triangle_area_l120_12038

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) :
  (1/2) * a * b = 180 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l120_12038


namespace NUMINAMATH_CALUDE_area_KLMN_value_l120_12042

/-- Triangle ABC with points K, L, N, and M -/
structure TriangleABC where
  -- Define the sides of the triangle
  AB : ℝ
  BC : ℝ
  AC : ℝ
  -- Define the positions of points K, L, and N
  AK : ℝ
  AL : ℝ
  BN : ℝ
  -- Ensure the triangle satisfies the given conditions
  h_AB : AB = 14
  h_BC : BC = 13
  h_AC : AC = 15
  h_AK : AK = 15/14
  h_AL : AL = 1
  h_BN : BN = 9

/-- The area of quadrilateral KLMN in the given triangle -/
def areaKLMN (t : TriangleABC) : ℝ := sorry

/-- Theorem stating that the area of KLMN is 36503/1183 -/
theorem area_KLMN_value (t : TriangleABC) : areaKLMN t = 36503/1183 := by sorry

end NUMINAMATH_CALUDE_area_KLMN_value_l120_12042


namespace NUMINAMATH_CALUDE_triangle_side_range_l120_12036

theorem triangle_side_range (a : ℝ) : 
  let AB := (5 : ℝ)
  let BC := 2 * a + 1
  let AC := (12 : ℝ)
  (AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB) → (3 < a ∧ a < 8) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_range_l120_12036


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l120_12035

theorem necessary_but_not_sufficient (a b : ℝ) :
  (a > 1 ∧ b > 1) → (a + b > 2 ∧ a * b > 1) ∧
  ¬((a + b > 2 ∧ a * b > 1) → (a > 1 ∧ b > 1)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l120_12035


namespace NUMINAMATH_CALUDE_base5_500_l120_12095

/-- Converts a natural number to its base-5 representation --/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- Converts a list of digits in base 5 to a natural number --/
def fromBase5 (digits : List ℕ) : ℕ :=
  digits.foldl (fun acc d => 5 * acc + d) 0

theorem base5_500 : toBase5 500 = [4, 0, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_base5_500_l120_12095


namespace NUMINAMATH_CALUDE_fraction_equality_implies_power_equality_l120_12044

theorem fraction_equality_implies_power_equality
  (a b c : ℝ) (k : ℕ) 
  (h_odd : Odd k)
  (h_eq : 1/a + 1/b + 1/c = 1/(a+b+c)) :
  1/a^k + 1/b^k + 1/c^k = 1/(a^k + b^k + c^k) :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_power_equality_l120_12044


namespace NUMINAMATH_CALUDE_fayes_carrots_l120_12097

/-- Proof of the number of carrots Faye picked -/
theorem fayes_carrots (good_carrots bad_carrots moms_carrots : ℕ) 
  (h1 : good_carrots = 12)
  (h2 : bad_carrots = 16)
  (h3 : moms_carrots = 5) :
  good_carrots + bad_carrots - moms_carrots = 23 := by
  sorry

end NUMINAMATH_CALUDE_fayes_carrots_l120_12097


namespace NUMINAMATH_CALUDE_temperature_data_inconsistency_l120_12037

theorem temperature_data_inconsistency (x_bar m S_squared : ℝ) 
  (h1 : x_bar = 0)
  (h2 : m = 4)
  (h3 : S_squared = 15.917)
  : |x_bar - m| > Real.sqrt S_squared := by
  sorry

end NUMINAMATH_CALUDE_temperature_data_inconsistency_l120_12037


namespace NUMINAMATH_CALUDE_real_part_of_z_l120_12010

theorem real_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = -1) : 
  Complex.re z = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l120_12010


namespace NUMINAMATH_CALUDE_broccoli_sales_amount_l120_12049

def farmers_market_sales (broccoli_sales : ℝ) : Prop :=
  let carrot_sales := 2 * broccoli_sales
  let spinach_sales := carrot_sales / 2 + 16
  let cauliflower_sales := 136
  broccoli_sales + carrot_sales + spinach_sales + cauliflower_sales = 380

theorem broccoli_sales_amount : ∃ (x : ℝ), farmers_market_sales x ∧ x = 57 :=
  sorry

end NUMINAMATH_CALUDE_broccoli_sales_amount_l120_12049


namespace NUMINAMATH_CALUDE_subtract_negative_five_l120_12083

theorem subtract_negative_five : 2 - (-5) = 7 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_five_l120_12083


namespace NUMINAMATH_CALUDE_inequality_system_solution_l120_12050

theorem inequality_system_solution (b : ℝ) : 
  (∀ (x y : ℝ), 2*b * Real.cos (2*(x-y)) + 8*b^2 * Real.cos (x-y) + 8*b^2*(b+1) + 5*b < 0 ∧
                 x^2 + y^2 + 1 > 2*b*x + 2*y + b - b^2) ↔ 
  (b < -1 - Real.sqrt 2 / 4 ∨ (-1/2 < b ∧ b < 0)) := by sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l120_12050


namespace NUMINAMATH_CALUDE_two_numbers_difference_l120_12017

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 50)
  (triple_minus_quadruple : 3 * y - 4 * x = 10)
  (y_geq_x : y ≥ x) :
  |y - x| = 10 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l120_12017


namespace NUMINAMATH_CALUDE_hash_3_7_l120_12062

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * b - b + b^2

-- State the theorem
theorem hash_3_7 : hash 3 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_hash_3_7_l120_12062


namespace NUMINAMATH_CALUDE_symmetric_point_example_l120_12087

/-- Given a point (x, y) in the plane, the point symmetric to it with respect to the x-axis is (x, -y) -/
def symmetric_point_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The coordinates of the point symmetric to (3, 8) with respect to the x-axis are (3, -8) -/
theorem symmetric_point_example : symmetric_point_x_axis (3, 8) = (3, -8) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_point_example_l120_12087


namespace NUMINAMATH_CALUDE_perimeter_gt_four_times_circumradius_l120_12019

/-- Definition of an acute-angled triangle -/
def IsAcuteAngledTriangle (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  a + b > c ∧ b + c > a ∧ c + a > b ∧
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2

/-- Perimeter of a triangle -/
def Perimeter (a b c : ℝ) : ℝ := a + b + c

/-- Circumradius of a triangle using the formula R = abc / (4A) where A is the area -/
noncomputable def Circumradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  (a * b * c) / (4 * area)

/-- Theorem: For any acute-angled triangle, its perimeter is greater than 4 times its circumradius -/
theorem perimeter_gt_four_times_circumradius (a b c : ℝ) 
  (h : IsAcuteAngledTriangle a b c) : 
  Perimeter a b c > 4 * Circumradius a b c := by
  sorry


end NUMINAMATH_CALUDE_perimeter_gt_four_times_circumradius_l120_12019


namespace NUMINAMATH_CALUDE_num_divisors_36_eq_9_l120_12026

/-- The number of positive divisors of 36 -/
def num_divisors_36 : ℕ :=
  (Finset.filter (· ∣ 36) (Finset.range 37)).card

/-- Theorem stating that the number of positive divisors of 36 is 9 -/
theorem num_divisors_36_eq_9 : num_divisors_36 = 9 := by
  sorry

end NUMINAMATH_CALUDE_num_divisors_36_eq_9_l120_12026


namespace NUMINAMATH_CALUDE_solution_sum_l120_12023

-- Define the solution set
def SolutionSet : Set ℝ := Set.union (Set.Iio 1) (Set.Ioi 4)

-- Define the theorem
theorem solution_sum (a b : ℝ) 
  (h : ∀ x, x ∈ SolutionSet ↔ (x - a) / (x - b) > 0) : 
  a + b = 5 := by sorry

end NUMINAMATH_CALUDE_solution_sum_l120_12023


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l120_12011

/-- A circle with a given center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line tangent to two circles -/
structure TangentLine where
  circle1 : Circle
  circle2 : Circle
  tangentPoint1 : ℝ × ℝ
  tangentPoint2 : ℝ × ℝ

/-- The y-intercept of a line tangent to two specific circles -/
def yIntercept (line : TangentLine) : ℝ :=
  sorry

/-- The main theorem stating the y-intercept of the tangent line -/
theorem tangent_line_y_intercept :
  let c1 : Circle := { center := (2, 0), radius := 2 }
  let c2 : Circle := { center := (5, 0), radius := 1 }
  ∀ (line : TangentLine),
    line.circle1 = c1 →
    line.circle2 = c2 →
    line.tangentPoint1.1 > 2 →
    line.tangentPoint1.2 > 0 →
    line.tangentPoint2.1 > 5 →
    line.tangentPoint2.2 > 0 →
    yIntercept line = 2 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l120_12011


namespace NUMINAMATH_CALUDE_students_in_jungkooks_class_l120_12092

theorem students_in_jungkooks_class :
  let glasses_wearers : Nat := 9
  let non_glasses_wearers : Nat := 16
  glasses_wearers + non_glasses_wearers = 25 :=
by sorry

end NUMINAMATH_CALUDE_students_in_jungkooks_class_l120_12092


namespace NUMINAMATH_CALUDE_tree_height_ratio_l120_12080

/-- Given three trees with specific height relationships, prove that the height of the smallest tree
    is 1/4 of the height of the middle-sized tree. -/
theorem tree_height_ratio :
  ∀ (h₁ h₂ h₃ : ℝ),
  h₁ = 108 →
  h₂ = h₁ / 2 - 6 →
  h₃ = 12 →
  h₃ / h₂ = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_tree_height_ratio_l120_12080


namespace NUMINAMATH_CALUDE_z_value_proof_l120_12064

theorem z_value_proof : 
  ∃ z : ℝ, (12 / 20 = (z / 20) ^ (1/3)) ∧ z = 4.32 :=
by
  sorry

end NUMINAMATH_CALUDE_z_value_proof_l120_12064


namespace NUMINAMATH_CALUDE_nested_square_roots_equality_l120_12090

theorem nested_square_roots_equality : Real.sqrt (36 * Real.sqrt (27 * Real.sqrt 9)) = 18 := by
  sorry

end NUMINAMATH_CALUDE_nested_square_roots_equality_l120_12090


namespace NUMINAMATH_CALUDE_tangent_line_parallel_l120_12032

/-- The function f(x) = ax³ + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

/-- The derivative of f(x) -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

theorem tangent_line_parallel (a : ℝ) : 
  (f_derivative a 1 = 4) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_l120_12032


namespace NUMINAMATH_CALUDE_anna_coins_l120_12084

/-- Represents the number of different coin values that can be obtained -/
def different_values (five_cent : ℕ) (twenty_cent : ℕ) : ℕ :=
  59 - 3 * five_cent

theorem anna_coins :
  ∀ (five_cent twenty_cent : ℕ),
    five_cent + twenty_cent = 15 →
    different_values five_cent twenty_cent = 24 →
    twenty_cent = 4 := by
  sorry

end NUMINAMATH_CALUDE_anna_coins_l120_12084


namespace NUMINAMATH_CALUDE_scoring_ratio_is_two_to_one_l120_12020

/-- Represents the scoring system for a test -/
structure TestScoring where
  totalQuestions : ℕ
  correctAnswers : ℕ
  score : ℕ
  scoringRatio : ℚ

/-- Calculates the score based on correct answers, incorrect answers, and the scoring ratio -/
def calculateScore (correct : ℕ) (incorrect : ℕ) (ratio : ℚ) : ℚ :=
  correct - ratio * incorrect

/-- Theorem stating that the scoring ratio is 2:1 for the given test conditions -/
theorem scoring_ratio_is_two_to_one (t : TestScoring)
    (h1 : t.totalQuestions = 100)
    (h2 : t.correctAnswers = 91)
    (h3 : t.score = 73)
    (h4 : calculateScore t.correctAnswers (t.totalQuestions - t.correctAnswers) t.scoringRatio = t.score) :
    t.scoringRatio = 2 := by
  sorry


end NUMINAMATH_CALUDE_scoring_ratio_is_two_to_one_l120_12020


namespace NUMINAMATH_CALUDE_largest_difference_l120_12031

theorem largest_difference (U V W X Y Z : ℕ) 
  (hU : U = 2 * 1002^1003)
  (hV : V = 1002^1003)
  (hW : W = 1001 * 1002^1002)
  (hX : X = 2 * 1002^1002)
  (hY : Y = 1002^1002)
  (hZ : Z = 1002^1001) :
  (U - V > V - W) ∧ 
  (U - V > W - X) ∧ 
  (U - V > X - Y) ∧ 
  (U - V > Y - Z) :=
sorry

end NUMINAMATH_CALUDE_largest_difference_l120_12031


namespace NUMINAMATH_CALUDE_first_division_percentage_l120_12069

theorem first_division_percentage (total_students : ℕ) 
  (second_division_percentage : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  second_division_percentage = 54 / 100 →
  just_passed = 63 →
  (total_students : ℚ) * (25 / 100) = 
    total_students - (total_students : ℚ) * second_division_percentage - just_passed :=
by
  sorry

end NUMINAMATH_CALUDE_first_division_percentage_l120_12069


namespace NUMINAMATH_CALUDE_unique_prime_sum_10123_l120_12008

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem unique_prime_sum_10123 :
  ∃! (p q : ℕ), is_prime p ∧ is_prime q ∧ p + q = 10123 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_sum_10123_l120_12008


namespace NUMINAMATH_CALUDE_complex_division_l120_12005

/-- Given complex numbers z₁ and z₂ corresponding to points (2, -1) and (0, -1) in the complex plane,
    prove that z₁ / z₂ = 1 + 2i -/
theorem complex_division (z₁ z₂ : ℂ) (h₁ : z₁ = 2 - I) (h₂ : z₂ = -I) : z₁ / z₂ = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_division_l120_12005


namespace NUMINAMATH_CALUDE_range_of_m_l120_12033

theorem range_of_m (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x - 1 < m^2 - 3*m) → m < 1 ∨ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l120_12033


namespace NUMINAMATH_CALUDE_jason_pokemon_cards_l120_12079

theorem jason_pokemon_cards 
  (cards_given_away : ℕ) 
  (cards_remaining : ℕ) 
  (h1 : cards_given_away = 9) 
  (h2 : cards_remaining = 4) : 
  cards_given_away + cards_remaining = 13 := by
sorry

end NUMINAMATH_CALUDE_jason_pokemon_cards_l120_12079


namespace NUMINAMATH_CALUDE_no_special_triangle_exists_l120_12074

/-- A triangle with sides and angles in arithmetic progression, given area, and circumradius -/
structure SpecialTriangle where
  /-- The common difference of the arithmetic progression of sides -/
  d : ℝ
  /-- The middle term of the arithmetic progression of sides -/
  b : ℝ
  /-- The area of the triangle -/
  area : ℝ
  /-- The radius of the circumscribed circle -/
  circumradius : ℝ
  /-- The sides form an arithmetic progression -/
  sides_progression : d ≥ 0 ∧ b > d
  /-- The angles form an arithmetic progression -/
  angles_progression : ∃ (α β γ : ℝ), α + β + γ = 180 ∧ β = 60 ∧ α < β ∧ β < γ
  /-- The area is 50 cm² -/
  area_constraint : area = 50
  /-- The circumradius is 10 cm -/
  circumradius_constraint : circumradius = 10

/-- Theorem stating that no triangle satisfies all the given conditions -/
theorem no_special_triangle_exists : ¬∃ (t : SpecialTriangle), True := by
  sorry

end NUMINAMATH_CALUDE_no_special_triangle_exists_l120_12074


namespace NUMINAMATH_CALUDE_flag_distribution_l120_12067

theorem flag_distribution (total_flags : ℕ) (blue_percent red_percent : ℚ) :
  total_flags % 2 = 0 ∧
  blue_percent = 60 / 100 ∧
  red_percent = 45 / 100 ∧
  blue_percent + red_percent > 1 →
  blue_percent + red_percent - 1 = 5 / 100 :=
by sorry

end NUMINAMATH_CALUDE_flag_distribution_l120_12067


namespace NUMINAMATH_CALUDE_jenny_recycling_payment_jenny_gets_three_cents_per_can_l120_12081

/-- Calculates the amount Jenny gets paid per can given the recycling conditions -/
theorem jenny_recycling_payment (bottle_weight : ℕ) (can_weight : ℕ) (total_weight : ℕ) 
  (num_cans : ℕ) (bottle_payment : ℕ) (total_payment : ℕ) : ℕ :=
  let remaining_weight := total_weight - (num_cans * can_weight)
  let num_bottles := remaining_weight / bottle_weight
  let bottle_total_payment := num_bottles * bottle_payment
  let can_total_payment := total_payment - bottle_total_payment
  can_total_payment / num_cans

/-- Proves that Jenny gets paid 3 cents per can under the given conditions -/
theorem jenny_gets_three_cents_per_can : 
  jenny_recycling_payment 6 2 100 20 10 160 = 3 := by
  sorry

end NUMINAMATH_CALUDE_jenny_recycling_payment_jenny_gets_three_cents_per_can_l120_12081


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l120_12057

theorem hemisphere_surface_area :
  ∀ (r : ℝ), 
    r > 0 →
    π * r^2 = 3 →
    let sphere_area := 4 * π * r^2
    let hemisphere_area := sphere_area / 2 + π * r^2
    hemisphere_area = 9 := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l120_12057


namespace NUMINAMATH_CALUDE_last_digit_of_powers_l120_12000

theorem last_digit_of_powers (n : Nat) :
  (∃ k : Nat, n = 2^1000 ∧ n % 10 = 6) ∧
  (∃ k : Nat, n = 3^1000 ∧ n % 10 = 1) ∧
  (∃ k : Nat, n = 7^1000 ∧ n % 10 = 1) :=
by sorry

end NUMINAMATH_CALUDE_last_digit_of_powers_l120_12000


namespace NUMINAMATH_CALUDE_brittany_age_after_vacation_l120_12027

/-- Represents a person with an age --/
structure Person where
  age : ℕ

/-- Represents a vacation --/
structure Vacation where
  duration : ℕ
  birthdaysCelebrated : ℕ
  hasLeapYear : Bool

/-- Calculates the age of a person after a vacation --/
def ageAfterVacation (person : Person) (vacation : Vacation) : ℕ :=
  person.age + vacation.birthdaysCelebrated

theorem brittany_age_after_vacation (rebecca : Person) (brittany : Person) (vacation : Vacation) :
  rebecca.age = 25 →
  brittany.age = rebecca.age + 3 →
  vacation.duration = 4 →
  vacation.birthdaysCelebrated = 3 →
  vacation.hasLeapYear = true →
  ageAfterVacation brittany vacation = 31 := by
  sorry

#eval ageAfterVacation (Person.mk 28) (Vacation.mk 4 3 true)

end NUMINAMATH_CALUDE_brittany_age_after_vacation_l120_12027


namespace NUMINAMATH_CALUDE_unique_solution_l120_12093

/-- Represents a three-digit number formed by digits U, H, and A -/
def three_digit_number (U H A : Nat) : Nat := 100 * U + 10 * H + A

/-- Represents a two-digit number formed by digits U and H -/
def two_digit_number (U H : Nat) : Nat := 10 * U + H

/-- Checks if a number is a valid digit (0-9) -/
def is_digit (n : Nat) : Prop := n ≤ 9

/-- Checks if three numbers are distinct -/
def are_distinct (a b c : Nat) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

/-- The main theorem stating the unique solution to the puzzle -/
theorem unique_solution :
  ∃! (U H A : Nat),
    is_digit U ∧ is_digit H ∧ is_digit A ∧
    are_distinct U H A ∧
    U ≠ 0 ∧
    three_digit_number U H A = Nat.lcm (two_digit_number U H) (Nat.lcm (two_digit_number U A) (two_digit_number H A)) ∧
    U = 1 ∧ H = 5 ∧ A = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l120_12093


namespace NUMINAMATH_CALUDE_train_journey_solution_l120_12072

/-- Represents the number of passengers from Zhejiang and Shanghai at a given point in the journey -/
structure PassengerCount where
  zhejiang : Nat
  shanghai : Nat

/-- Represents the train journey with passenger counts at each stage -/
structure TrainJourney where
  initial : PassengerCount
  afterB : PassengerCount
  afterC : PassengerCount
  afterD : PassengerCount
  afterE : PassengerCount
  final : PassengerCount

def total_passengers (pc : PassengerCount) : Nat :=
  pc.zhejiang + pc.shanghai

/-- The conditions of the train journey -/
def journey_conditions (j : TrainJourney) : Prop :=
  total_passengers j.initial = 19 ∧
  total_passengers j.afterB = 12 ∧
  total_passengers j.afterD = 7 ∧
  total_passengers j.final = 0 ∧
  j.initial.zhejiang = (total_passengers j.initial - total_passengers j.afterB) ∧
  j.afterB.zhejiang = (total_passengers j.afterB - total_passengers j.afterC) ∧
  j.afterC.zhejiang = (total_passengers j.afterC - total_passengers j.afterD) ∧
  j.afterD.zhejiang = (total_passengers j.afterD - total_passengers j.afterE) ∧
  j.afterE.zhejiang = (total_passengers j.afterE - total_passengers j.final)

/-- The theorem stating that given the conditions, the journey matches the solution -/
theorem train_journey_solution (j : TrainJourney) :
  journey_conditions j →
  j.initial = ⟨7, 12⟩ ∧
  j.afterB = ⟨3, 9⟩ ∧
  j.afterC = ⟨2, 7⟩ ∧
  j.afterD = ⟨2, 5⟩ :=
by sorry

end NUMINAMATH_CALUDE_train_journey_solution_l120_12072


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l120_12055

theorem cubic_roots_sum_cubes (p q r : ℂ) : 
  p ≠ q ∧ q ≠ r ∧ p ≠ r →
  p^3 - p^2 + p - 2 = 0 →
  q^3 - q^2 + q - 2 = 0 →
  r^3 - r^2 + r - 2 = 0 →
  p^3 + q^3 + r^3 = -6 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l120_12055


namespace NUMINAMATH_CALUDE_highest_demand_week_sales_total_sales_check_l120_12059

-- Define the sales for each week
def first_week_sales : ℕ := 20
def second_week_sales : ℕ := 3 * first_week_sales
def third_week_sales : ℕ := 2 * first_week_sales
def fourth_week_sales : ℕ := first_week_sales

-- Define the total sales for the month
def total_sales : ℕ := 300

-- Theorem to prove the highest demand week
theorem highest_demand_week_sales :
  max first_week_sales (max second_week_sales (max third_week_sales fourth_week_sales)) = 60 :=
by sorry

-- Verify that the sum of all weeks' sales equals the total monthly sales
theorem total_sales_check :
  first_week_sales + second_week_sales + third_week_sales + fourth_week_sales = total_sales :=
by sorry

end NUMINAMATH_CALUDE_highest_demand_week_sales_total_sales_check_l120_12059


namespace NUMINAMATH_CALUDE_green_toads_per_acre_l120_12086

/-- Given information about toads in central Texas countryside -/
structure ToadPopulation where
  /-- The ratio of green toads to brown toads -/
  green_to_brown_ratio : ℚ
  /-- The percentage of brown toads that are spotted -/
  spotted_brown_percentage : ℚ
  /-- The number of spotted brown toads per acre -/
  spotted_brown_per_acre : ℕ

/-- Theorem stating the number of green toads per acre -/
theorem green_toads_per_acre (tp : ToadPopulation)
  (h1 : tp.green_to_brown_ratio = 1 / 25)
  (h2 : tp.spotted_brown_percentage = 1 / 4)
  (h3 : tp.spotted_brown_per_acre = 50) :
  (tp.spotted_brown_per_acre : ℚ) / (tp.spotted_brown_percentage * tp.green_to_brown_ratio) = 8 := by
  sorry

end NUMINAMATH_CALUDE_green_toads_per_acre_l120_12086


namespace NUMINAMATH_CALUDE_mans_upstream_rate_l120_12089

/-- Prove that given a man's downstream rate, his rate in still water, and the current rate, his upstream rate can be calculated. -/
theorem mans_upstream_rate (downstream_rate still_water_rate current_rate : ℝ) 
  (h1 : downstream_rate = 32)
  (h2 : still_water_rate = 24.5)
  (h3 : current_rate = 7.5) :
  still_water_rate - current_rate = 17 := by
  sorry

end NUMINAMATH_CALUDE_mans_upstream_rate_l120_12089


namespace NUMINAMATH_CALUDE_nth_equation_l120_12054

theorem nth_equation (n : ℕ) : Real.sqrt ((n + 1) * (n + 3) + 1) = n + 2 := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_l120_12054


namespace NUMINAMATH_CALUDE_income_difference_after_raise_l120_12078

-- Define the annual raise percentage
def annual_raise_percent : ℚ := 8 / 100

-- Define Don's raise amount
def don_raise : ℕ := 800

-- Define Don's wife's raise amount
def wife_raise : ℕ := 840

-- Define function to calculate original salary given the raise amount
def original_salary (raise : ℕ) : ℚ := (raise : ℚ) / annual_raise_percent

-- Define function to calculate new salary after raise
def new_salary (raise : ℕ) : ℚ := original_salary raise + raise

-- Theorem statement
theorem income_difference_after_raise :
  new_salary wife_raise - new_salary don_raise = 540 := by
  sorry

end NUMINAMATH_CALUDE_income_difference_after_raise_l120_12078


namespace NUMINAMATH_CALUDE_pentagon_reflection_rotation_l120_12006

/-- A regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : ∀ i j : Fin 5, dist (vertices i) (vertices ((i + 1) % 5)) = dist (vertices j) (vertices ((j + 1) % 5))
  is_pentagon : ∀ i : Fin 5, vertices i ≠ vertices ((i + 1) % 5)

/-- Reflection of a point over a line through the center of the pentagon -/
def reflect (p : RegularPentagon) (line : ℝ × ℝ → Prop) : RegularPentagon :=
  sorry

/-- Rotation of a pentagon by an angle about its center -/
def rotate (p : RegularPentagon) (angle : ℝ) : RegularPentagon :=
  sorry

/-- The center of a regular pentagon -/
def center (p : RegularPentagon) : ℝ × ℝ :=
  sorry

theorem pentagon_reflection_rotation (p : RegularPentagon) (line : ℝ × ℝ → Prop) :
  rotate (reflect p line) (144 * π / 180) = rotate p (144 * π / 180) :=
sorry

end NUMINAMATH_CALUDE_pentagon_reflection_rotation_l120_12006


namespace NUMINAMATH_CALUDE_diamond_operation_result_l120_12034

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element

-- Define the diamond operation
def diamond : Element → Element → Element
  | Element.one, Element.one => Element.four
  | Element.one, Element.two => Element.three
  | Element.one, Element.three => Element.two
  | Element.one, Element.four => Element.one
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.four
  | Element.two, Element.three => Element.three
  | Element.two, Element.four => Element.two
  | Element.three, Element.one => Element.two
  | Element.three, Element.two => Element.one
  | Element.three, Element.three => Element.four
  | Element.three, Element.four => Element.three
  | Element.four, Element.one => Element.three
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.four

theorem diamond_operation_result :
  diamond (diamond Element.three Element.four) (diamond Element.two Element.one) = Element.two := by
  sorry

end NUMINAMATH_CALUDE_diamond_operation_result_l120_12034


namespace NUMINAMATH_CALUDE_bicycle_trip_time_l120_12016

/-- Proves that the time taken to go forth is 1 hour given the conditions of the bicycle problem -/
theorem bicycle_trip_time (speed_forth speed_back : ℝ) (time_diff : ℝ) 
  (h1 : speed_forth = 15)
  (h2 : speed_back = 10)
  (h3 : time_diff = 0.5)
  : ∃ (time_forth : ℝ), 
    speed_forth * time_forth = speed_back * (time_forth + time_diff) ∧ 
    time_forth = 1 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_trip_time_l120_12016


namespace NUMINAMATH_CALUDE_jillian_apartment_size_l120_12048

/-- The cost per square foot of apartment rentals in Rivertown -/
def cost_per_sqft : ℚ := 1.20

/-- Jillian's maximum monthly budget for rent -/
def max_budget : ℚ := 720

/-- The largest apartment size Jillian should consider -/
def largest_apartment_size : ℚ := max_budget / cost_per_sqft

theorem jillian_apartment_size :
  largest_apartment_size = 600 :=
by sorry

end NUMINAMATH_CALUDE_jillian_apartment_size_l120_12048


namespace NUMINAMATH_CALUDE_vector_equality_implies_m_equals_two_l120_12099

def a (m : ℝ) : ℝ × ℝ := (m, -2)
def b : ℝ × ℝ := (1, 1)

theorem vector_equality_implies_m_equals_two (m : ℝ) :
  ‖a m - b‖ = ‖a m + b‖ → m = 2 := by sorry

end NUMINAMATH_CALUDE_vector_equality_implies_m_equals_two_l120_12099


namespace NUMINAMATH_CALUDE_distance_traveled_l120_12063

/-- Given a person traveling at 6 km/h for 10 minutes, prove that the distance traveled is 1000 meters. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) : 
  speed = 6 → time = 1/6 → speed * time * 1000 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_distance_traveled_l120_12063


namespace NUMINAMATH_CALUDE_parallel_lines_sum_l120_12028

/-- Two parallel lines with a given distance between them -/
structure ParallelLines where
  m : ℝ
  n : ℝ
  h_m_pos : m > 0
  h_parallel : 1 / 2 = -2 / n
  h_distance : |m + 3| / Real.sqrt 5 = Real.sqrt 5

/-- The sum of m and n for the parallel lines is -2 -/
theorem parallel_lines_sum (l : ParallelLines) : l.m + l.n = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_sum_l120_12028


namespace NUMINAMATH_CALUDE_power_of_81_l120_12066

theorem power_of_81 : (81 : ℝ) ^ (5/4 : ℝ) = 243 := by
  sorry

end NUMINAMATH_CALUDE_power_of_81_l120_12066


namespace NUMINAMATH_CALUDE_weekend_run_ratio_l120_12056

/-- Represents the miles run by Bill and Julia over a weekend --/
structure WeekendRun where
  billSaturday : ℝ
  billSunday : ℝ
  juliaSunday : ℝ
  m : ℝ

/-- Conditions for a valid WeekendRun --/
def ValidWeekendRun (run : WeekendRun) : Prop :=
  run.billSunday = run.billSaturday + 4 ∧
  run.juliaSunday = run.m * run.billSunday ∧
  run.billSaturday + run.billSunday + run.juliaSunday = 32

theorem weekend_run_ratio (run : WeekendRun) 
  (h : ValidWeekendRun run) :
  run.juliaSunday / run.billSunday = run.m :=
by
  sorry

#check weekend_run_ratio

end NUMINAMATH_CALUDE_weekend_run_ratio_l120_12056


namespace NUMINAMATH_CALUDE_gauss_1998_cycle_l120_12082

def word_length : Nat := 5
def number_length : Nat := 4

theorem gauss_1998_cycle : Nat.lcm word_length number_length = 20 := by
  sorry

end NUMINAMATH_CALUDE_gauss_1998_cycle_l120_12082


namespace NUMINAMATH_CALUDE_special_rectangle_area_length_width_ratio_width_is_diameter_l120_12015

/-- A rectangle with an inscribed circle of radius 10 and a circumscribed circle -/
structure Rectangle where
  width : ℝ
  length : ℝ
  inscribed_circle_radius : ℝ
  has_circumscribed_circle : Prop

/-- The properties of our specific rectangle -/
def special_rectangle : Rectangle where
  width := 20
  length := 60
  inscribed_circle_radius := 10
  has_circumscribed_circle := true

/-- The theorem stating that the area of the special rectangle is 1200 -/
theorem special_rectangle_area :
  special_rectangle.length * special_rectangle.width = 1200 := by
  sorry

/-- The ratio of length to width is 3:1 -/
theorem length_width_ratio :
  special_rectangle.length = 3 * special_rectangle.width := by
  sorry

/-- The width is twice the radius of the inscribed circle -/
theorem width_is_diameter :
  special_rectangle.width = 2 * special_rectangle.inscribed_circle_radius := by
  sorry

end NUMINAMATH_CALUDE_special_rectangle_area_length_width_ratio_width_is_diameter_l120_12015


namespace NUMINAMATH_CALUDE_specific_frustum_volume_l120_12051

/-- A frustum with given base areas and lateral surface area -/
structure Frustum where
  upper_base_area : ℝ
  lower_base_area : ℝ
  lateral_surface_area : ℝ

/-- The volume of a frustum -/
def volume (f : Frustum) : ℝ := sorry

/-- Theorem stating the volume of the specific frustum -/
theorem specific_frustum_volume :
  ∃ (f : Frustum),
    f.upper_base_area = π ∧
    f.lower_base_area = 4 * π ∧
    f.lateral_surface_area = 6 * π ∧
    volume f = (7 * Real.sqrt 3 / 3) * π := by sorry

end NUMINAMATH_CALUDE_specific_frustum_volume_l120_12051


namespace NUMINAMATH_CALUDE_jamies_liquid_limit_l120_12043

/-- Jamie's liquid consumption limit problem -/
theorem jamies_liquid_limit :
  let cup_oz : ℕ := 8  -- A cup is 8 ounces
  let pint_oz : ℕ := 16  -- A pint is 16 ounces
  let milk_consumed : ℕ := cup_oz  -- Jamie had a cup of milk
  let juice_consumed : ℕ := pint_oz  -- Jamie had a pint of grape juice
  let water_limit : ℕ := 8  -- Jamie can drink 8 more ounces before needing the bathroom
  milk_consumed + juice_consumed + water_limit = 32  -- Jamie's total liquid limit
  := by sorry

end NUMINAMATH_CALUDE_jamies_liquid_limit_l120_12043


namespace NUMINAMATH_CALUDE_race_time_difference_l120_12040

/-- Represents the race scenario with Malcolm and Joshua -/
structure RaceScenario where
  malcolm_speed : ℝ  -- Malcolm's speed in minutes per mile
  joshua_speed : ℝ   -- Joshua's speed in minutes per mile
  race_distance : ℝ  -- Race distance in miles

/-- Calculates the time difference between Malcolm and Joshua finishing the race -/
def time_difference (scenario : RaceScenario) : ℝ :=
  scenario.joshua_speed * scenario.race_distance - scenario.malcolm_speed * scenario.race_distance

/-- Theorem stating the time difference for the given race scenario -/
theorem race_time_difference (scenario : RaceScenario) 
  (h1 : scenario.malcolm_speed = 5)
  (h2 : scenario.joshua_speed = 7)
  (h3 : scenario.race_distance = 12) :
  time_difference scenario = 24 := by
  sorry

#eval time_difference { malcolm_speed := 5, joshua_speed := 7, race_distance := 12 }

end NUMINAMATH_CALUDE_race_time_difference_l120_12040


namespace NUMINAMATH_CALUDE_bc_cd_ratio_l120_12068

-- Define the points on the line
variable (a b c d e : ℝ)

-- Define the conditions
axiom consecutive_points : a < b ∧ b < c ∧ c < d ∧ d < e
axiom de_length : e - d = 8
axiom ab_length : b - a = 5
axiom ac_length : c - a = 11
axiom ae_length : e - a = 22

-- Define the theorem
theorem bc_cd_ratio :
  (c - b) / (d - c) = 2 / 1 :=
sorry

end NUMINAMATH_CALUDE_bc_cd_ratio_l120_12068


namespace NUMINAMATH_CALUDE_decimal_to_fraction_l120_12061

theorem decimal_to_fraction : (3.375 : ℚ) = 27 / 8 := by sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_l120_12061


namespace NUMINAMATH_CALUDE_problem_solution_l120_12009

theorem problem_solution (a b c : ℝ) 
  (h1 : (a + b + c)^2 = 3*(a^2 + b^2 + c^2)) 
  (h2 : a + b + c = 12) : 
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l120_12009


namespace NUMINAMATH_CALUDE_largest_prime_factor_is_101_l120_12091

/-- A sequence of four-digit integers with a cyclic digit property -/
def CyclicSequence := List Nat

/-- The sum of all terms in a cyclic sequence -/
def sequenceSum (seq : CyclicSequence) : Nat :=
  seq.sum

/-- Predicate to check if a sequence satisfies the cyclic digit property -/
def hasCyclicDigitProperty (seq : CyclicSequence) : Prop :=
  sorry -- Definition of the cyclic digit property

/-- The largest prime factor that always divides the sum of a cyclic sequence -/
def largestPrimeFactor (seq : CyclicSequence) : Nat :=
  sorry -- Definition to find the largest prime factor

theorem largest_prime_factor_is_101 (seq : CyclicSequence) 
    (h : hasCyclicDigitProperty seq) :
    largestPrimeFactor seq = 101 := by
  sorry

#check largest_prime_factor_is_101

end NUMINAMATH_CALUDE_largest_prime_factor_is_101_l120_12091


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l120_12021

theorem quadratic_expression_value (x : ℝ) (h : 2 * x^2 + 3 * x - 5 = 0) :
  4 * x^2 + 6 * x + 9 = 19 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l120_12021


namespace NUMINAMATH_CALUDE_orange_slices_theorem_l120_12013

/-- Represents the number of slices each animal received -/
structure OrangeSlices where
  siskin : ℕ
  hedgehog : ℕ
  beaver : ℕ

/-- Conditions for the orange slices distribution -/
def valid_distribution (slices : OrangeSlices) : Prop :=
  slices.hedgehog = 2 * slices.siskin ∧
  slices.beaver = 5 * slices.siskin ∧
  slices.beaver = slices.siskin + 8

/-- The total number of slices in the orange -/
def total_slices (slices : OrangeSlices) : ℕ :=
  slices.siskin + slices.hedgehog + slices.beaver

/-- Theorem stating that the total number of slices is 16 -/
theorem orange_slices_theorem :
  ∃ (slices : OrangeSlices), valid_distribution slices ∧ total_slices slices = 16 :=
sorry

end NUMINAMATH_CALUDE_orange_slices_theorem_l120_12013


namespace NUMINAMATH_CALUDE_inequality_proof_l120_12065

theorem inequality_proof (x : ℝ) : (x - 4) / 2 - (x - 1) / 4 < 1 → x < 11 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l120_12065


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l120_12012

theorem complex_fraction_sum : (1 - 2*Complex.I) / (1 + Complex.I) + (1 + 2*Complex.I) / (1 - Complex.I) = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l120_12012


namespace NUMINAMATH_CALUDE_cost_of_leftover_drinks_l120_12096

theorem cost_of_leftover_drinks : 
  let soda_bought := 30
  let soda_price := 2
  let energy_bought := 20
  let energy_price := 3
  let smoothie_bought := 15
  let smoothie_price := 4
  let soda_consumed := 10
  let energy_consumed := 14
  let smoothie_consumed := 5
  
  let soda_leftover := soda_bought - soda_consumed
  let energy_leftover := energy_bought - energy_consumed
  let smoothie_leftover := smoothie_bought - smoothie_consumed
  
  let leftover_cost := soda_leftover * soda_price + 
                       energy_leftover * energy_price + 
                       smoothie_leftover * smoothie_price
  
  leftover_cost = 98 := by sorry

end NUMINAMATH_CALUDE_cost_of_leftover_drinks_l120_12096


namespace NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l120_12085

theorem rationalize_denominator_cube_root :
  ∃ (A B C : ℕ), 
    (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
    (∀ p : ℕ, Prime p → ¬(p^3 ∣ B)) ∧
    (5 / (3 * Real.rpow 7 (1/3)) = (A * Real.rpow B (1/3)) / C) ∧
    (A + B + C = 75) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l120_12085


namespace NUMINAMATH_CALUDE_B_grazed_five_months_l120_12001

/-- Represents the number of months B grazed his cows -/
def B_months : ℕ := sorry

/-- Total rent of the field in rupees -/
def total_rent : ℕ := 3250

/-- A's share of rent in rupees -/
def A_rent : ℕ := 720

/-- Number of cows grazed by each milkman -/
def cows : Fin 4 → ℕ
| 0 => 24  -- A
| 1 => 10  -- B
| 2 => 35  -- C
| 3 => 21  -- D

/-- Number of months each milkman grazed their cows -/
def months : Fin 4 → ℕ
| 0 => 3         -- A
| 1 => B_months  -- B
| 2 => 4         -- C
| 3 => 3         -- D

/-- Total cow-months for all milkmen -/
def total_cow_months : ℕ := 
  (cows 0 * months 0) + (cows 1 * months 1) + (cows 2 * months 2) + (cows 3 * months 3)

theorem B_grazed_five_months : B_months = 5 := by
  sorry

end NUMINAMATH_CALUDE_B_grazed_five_months_l120_12001


namespace NUMINAMATH_CALUDE_circle_radius_is_13_main_result_l120_12041

/-- Represents a circle with tangents -/
structure CircleWithTangents where
  r : ℝ  -- radius of the circle
  ab : ℝ  -- length of tangent AB
  ac : ℝ  -- length of tangent AC
  de : ℝ  -- length of tangent DE perpendicular to BC

/-- Theorem: Given the conditions, the radius of the circle is 13 -/
theorem circle_radius_is_13 (c : CircleWithTangents) 
  (h1 : c.ab = 5) 
  (h2 : c.ac = 12) 
  (h3 : c.de = 13) : 
  c.r = 13 := by
  sorry

/-- The main result -/
theorem main_result : ∃ c : CircleWithTangents, 
  c.ab = 5 ∧ c.ac = 12 ∧ c.de = 13 ∧ c.r = 13 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_is_13_main_result_l120_12041


namespace NUMINAMATH_CALUDE_num_polygons_twelve_points_l120_12014

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The minimum number of sides for the polygons we're considering -/
def min_sides : ℕ := 4

/-- The number of distinct convex polygons with 4 or more sides 
    that can be drawn using some or all of n points marked on a circle -/
def num_polygons (n : ℕ) (min_sides : ℕ) : ℕ :=
  2^n - (n.choose 0 + n.choose 1 + n.choose 2 + n.choose 3)

theorem num_polygons_twelve_points : 
  num_polygons n min_sides = 3797 := by
  sorry

end NUMINAMATH_CALUDE_num_polygons_twelve_points_l120_12014


namespace NUMINAMATH_CALUDE_square_diagonal_l120_12053

theorem square_diagonal (p : ℝ) (h : p = 200 * Real.sqrt 2) :
  let s := p / 4
  s * Real.sqrt 2 = 100 := by sorry

end NUMINAMATH_CALUDE_square_diagonal_l120_12053


namespace NUMINAMATH_CALUDE_max_true_statements_l120_12071

theorem max_true_statements (x : ℝ) : 
  let statements := [
    (0 < x^3 ∧ x^3 < 1),
    (x^3 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - x^3 ∧ x - x^3 < 1),
    (x^3 - x > 1)
  ]
  ∃ (true_statements : List Bool), 
    (∀ i, true_statements.get! i = true → statements.get! i) ∧
    true_statements.count true ≤ 3 ∧
    ∀ (other_true_statements : List Bool),
      (∀ i, other_true_statements.get! i = true → statements.get! i) →
      other_true_statements.count true ≤ true_statements.count true :=
by sorry


end NUMINAMATH_CALUDE_max_true_statements_l120_12071


namespace NUMINAMATH_CALUDE_camp_recoloring_l120_12046

/-- A graph representing friendships in a summer camp -/
structure CampGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  degree_eleven : ∀ v ∈ vertices, (edges.filter (fun e => e.1 = v ∨ e.2 = v)).card = 11
  symmetric : ∀ ⦃a b⦄, (a, b) ∈ edges → (b, a) ∈ edges

/-- A valid coloring of the graph -/
def ValidColoring (G : CampGraph) (coloring : Nat → Fin 7) : Prop :=
  ∀ ⦃a b⦄, (a, b) ∈ G.edges → coloring a ≠ coloring b

theorem camp_recoloring (G : CampGraph) (initial_coloring : Nat → Fin 7)
    (h_valid : ValidColoring G initial_coloring)
    (fixed_vertices : Finset Nat)
    (h_fixed_size : fixed_vertices.card = 100)
    (h_fixed_subset : fixed_vertices ⊆ G.vertices) :
    ∃ (new_coloring : Nat → Fin 7),
      ValidColoring G new_coloring ∧
      (∃ v ∈ G.vertices \ fixed_vertices, new_coloring v ≠ initial_coloring v) ∧
      (∀ v ∈ fixed_vertices, new_coloring v = initial_coloring v) :=
  sorry

end NUMINAMATH_CALUDE_camp_recoloring_l120_12046


namespace NUMINAMATH_CALUDE_rational_function_identity_l120_12039

theorem rational_function_identity (x : ℝ) (h1 : x ≠ 2) (h2 : x^2 + x + 1 ≠ 0) :
  (x + 3)^2 / ((x - 2) * (x^2 + x + 1)) = 
  25 / (7 * (x - 2)) + (-18 * x - 19) / (7 * (x^2 + x + 1)) := by
  sorry

#check rational_function_identity

end NUMINAMATH_CALUDE_rational_function_identity_l120_12039


namespace NUMINAMATH_CALUDE_scaling_transformation_l120_12029

-- Define the original circle equation
def original_equation (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the scaling transformation
def scale_x (x : ℝ) : ℝ := 5 * x
def scale_y (y : ℝ) : ℝ := 3 * y

-- State the theorem
theorem scaling_transformation :
  ∀ x' y' : ℝ, (∃ x y : ℝ, original_equation x y ∧ x' = scale_x x ∧ y' = scale_y y) →
  (x'^2 / 25 + y'^2 / 9 = 1) :=
by sorry

end NUMINAMATH_CALUDE_scaling_transformation_l120_12029


namespace NUMINAMATH_CALUDE_area_not_covered_by_square_l120_12002

/-- Given a rectangle with dimensions 10 units by 8 units and an inscribed square
    with side length 5 units, the area of the region not covered by the square
    is 55 square units. -/
theorem area_not_covered_by_square (rectangle_length : ℝ) (rectangle_width : ℝ) 
    (square_side : ℝ) (h1 : rectangle_length = 10) (h2 : rectangle_width = 8) 
    (h3 : square_side = 5) : 
    rectangle_length * rectangle_width - square_side^2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_area_not_covered_by_square_l120_12002


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l120_12018

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence
  a 1 = 1 →                                         -- first term
  a 2 = 5 →                                         -- second term
  a 3 = 9 →                                         -- third term
  ∀ n, a n = 4 * n - 3 :=                           -- general formula
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l120_12018


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l120_12024

/-- Given an ellipse with semi-major axis a and semi-minor axis b, where a > b > 0,
    and foci F₁ and F₂, a line passing through F₁ intersects the ellipse at points A and B.
    If AB ⟂ AF₂ and |AB| = |AF₂|, then the eccentricity of the ellipse is √6 - √3. -/
theorem ellipse_eccentricity (a b : ℝ) (F₁ F₂ A B : ℝ × ℝ) :
  a > b ∧ b > 0 →
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ (x, y) ∈ ({F₁, F₂, A, B} : Set (ℝ × ℝ))) →
  (A.1 - B.1) * (A.1 - F₂.1) + (A.2 - B.2) * (A.2 - F₂.2) = 0 →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (A.1 - F₂.1)^2 + (A.2 - F₂.2)^2 →
  let e := Real.sqrt ((a^2 - b^2) / a^2)
  e = Real.sqrt 6 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l120_12024


namespace NUMINAMATH_CALUDE_lagoonIslandMales_l120_12045

/-- Represents the population of alligators on Lagoon Island -/
structure AlligatorPopulation where
  total : ℕ
  males : ℕ
  females : ℕ
  juvenileFemales : ℕ
  adultFemales : ℕ

/-- The conditions of the alligator population on Lagoon Island -/
def lagoonIslandConditions (pop : AlligatorPopulation) : Prop :=
  pop.males = pop.females ∧
  pop.females = pop.juvenileFemales + pop.adultFemales ∧
  pop.juvenileFemales = (2 * pop.females) / 5 ∧
  pop.adultFemales = 15

theorem lagoonIslandMales (pop : AlligatorPopulation) 
  (h : lagoonIslandConditions pop) : pop.males = 25 := by
  sorry

end NUMINAMATH_CALUDE_lagoonIslandMales_l120_12045


namespace NUMINAMATH_CALUDE_simplify_fraction_x_squared_minus_y_squared_l120_12030

-- Part 1
theorem simplify_fraction (a : ℝ) (h : a > 0) : 
  1 / (Real.sqrt a + 1) = (Real.sqrt a - 1) / 2 :=
sorry

-- Part 2
theorem x_squared_minus_y_squared (x y : ℝ) 
  (hx : x = 1 / (2 + Real.sqrt 3)) 
  (hy : y = 1 / (2 - Real.sqrt 3)) : 
  x^2 - y^2 = -8 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_simplify_fraction_x_squared_minus_y_squared_l120_12030


namespace NUMINAMATH_CALUDE_paul_school_supplies_l120_12025

theorem paul_school_supplies : 
  let initial_regular_erasers : ℕ := 307
  let initial_jumbo_erasers : ℕ := 150
  let initial_standard_crayons : ℕ := 317
  let initial_jumbo_crayons : ℕ := 300
  let lost_regular_erasers : ℕ := 52
  let used_standard_crayons : ℕ := 123
  let used_jumbo_crayons : ℕ := 198

  let remaining_regular_erasers : ℕ := initial_regular_erasers - lost_regular_erasers
  let remaining_jumbo_erasers : ℕ := initial_jumbo_erasers
  let remaining_standard_crayons : ℕ := initial_standard_crayons - used_standard_crayons
  let remaining_jumbo_crayons : ℕ := initial_jumbo_crayons - used_jumbo_crayons

  let total_remaining_erasers : ℕ := remaining_regular_erasers + remaining_jumbo_erasers
  let total_remaining_crayons : ℕ := remaining_standard_crayons + remaining_jumbo_crayons

  (total_remaining_crayons : ℤ) - (total_remaining_erasers : ℤ) = -109
  := by sorry

end NUMINAMATH_CALUDE_paul_school_supplies_l120_12025


namespace NUMINAMATH_CALUDE_probability_perfect_square_sum_l120_12075

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def dice_sum_outcomes : ℕ := 64

def perfect_square_sums : List ℕ := [4, 9, 16]

def ways_to_get_sum (sum : ℕ) : ℕ :=
  if sum = 4 then 3
  else if sum = 9 then 8
  else if sum = 16 then 1
  else 0

def total_favorable_outcomes : ℕ :=
  (perfect_square_sums.map ways_to_get_sum).sum

theorem probability_perfect_square_sum :
  (total_favorable_outcomes : ℚ) / dice_sum_outcomes = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_perfect_square_sum_l120_12075


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l120_12007

/-- An arithmetic sequence with common difference 1 -/
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 1

/-- Three terms form a geometric sequence -/
def geometric_seq (x y z : ℝ) : Prop :=
  y^2 = x * z

/-- The main theorem -/
theorem arithmetic_geometric_sequence (a : ℕ → ℝ) 
  (h_arith : arithmetic_seq a)
  (h_geom : geometric_seq (a 1) (a 3) (a 7)) :
  a 5 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l120_12007


namespace NUMINAMATH_CALUDE_smallest_x_value_l120_12058

theorem smallest_x_value (y : ℕ+) (x : ℕ+) (h : (3 : ℚ) / 4 = y / (210 + x)) : 
  2 ≤ x.val :=
sorry

end NUMINAMATH_CALUDE_smallest_x_value_l120_12058


namespace NUMINAMATH_CALUDE_complex_and_imaginary_solution_l120_12022

-- Define z as a complex number
variable (z : ℂ)

-- Define the conditions
def condition1 : Prop := (z + Complex.I).im = 0
def condition2 : Prop := (z / (1 - Complex.I)).im = 0

-- Define m as a purely imaginary number
def m : ℂ → ℂ := fun c => Complex.I * c

-- Define the equation with real roots
def has_real_roots (z m : ℂ) : Prop :=
  ∃ x : ℝ, x^2 + x * (1 + z) - (3 * m - 1) * Complex.I = 0

-- State the theorem
theorem complex_and_imaginary_solution :
  condition1 z → condition2 z → has_real_roots z (m 1) →
  z = 1 - Complex.I ∧ m 1 = -Complex.I :=
sorry

end NUMINAMATH_CALUDE_complex_and_imaginary_solution_l120_12022


namespace NUMINAMATH_CALUDE_rational_inequality_l120_12094

theorem rational_inequality (a b : ℚ) 
  (h1 : |a| < |b|) 
  (h2 : a > 0) 
  (h3 : b < 0) : 
  b < -a ∧ -a < a ∧ a < -b := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_l120_12094


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l120_12060

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence condition
  a 1 = 3 →
  a 100 = 36 →
  a 3 + a 98 = 39 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l120_12060


namespace NUMINAMATH_CALUDE_sum_of_floors_even_l120_12077

theorem sum_of_floors_even (a b c : ℕ+) (h : a^2 + b^2 + 1 = c^2) :
  Even (⌊(a : ℝ) / 2⌋ + ⌊(c : ℝ) / 2⌋) := by sorry

end NUMINAMATH_CALUDE_sum_of_floors_even_l120_12077
