import Mathlib

namespace NUMINAMATH_CALUDE_cistern_empty_in_eight_minutes_l1188_118848

/-- Given a pipe that can empty 2/3 of a cistern in 10 minutes,
    this function calculates the part of the cistern that will be empty in a given number of minutes. -/
def cisternEmptyPart (emptyRate : Rat) (totalTime : Nat) (elapsedTime : Nat) : Rat :=
  (emptyRate / totalTime) * elapsedTime

/-- Theorem stating that given a pipe that can empty 2/3 of a cistern in 10 minutes,
    the part of the cistern that will be empty in 8 minutes is 8/15. -/
theorem cistern_empty_in_eight_minutes :
  cisternEmptyPart (2/3) 10 8 = 8/15 := by
  sorry

end NUMINAMATH_CALUDE_cistern_empty_in_eight_minutes_l1188_118848


namespace NUMINAMATH_CALUDE_helium_cost_per_ounce_l1188_118803

-- Define the constants
def total_money : ℚ := 200
def sheet_cost : ℚ := 42
def rope_cost : ℚ := 18
def propane_cost : ℚ := 14
def height_per_ounce : ℚ := 113
def max_height : ℚ := 9492

-- Define the theorem
theorem helium_cost_per_ounce :
  let money_left := total_money - (sheet_cost + rope_cost + propane_cost)
  let ounces_needed := max_height / height_per_ounce
  let cost_per_ounce := money_left / ounces_needed
  cost_per_ounce = 3/2 := by sorry

end NUMINAMATH_CALUDE_helium_cost_per_ounce_l1188_118803


namespace NUMINAMATH_CALUDE_minutes_in_three_and_half_hours_l1188_118859

/-- The number of minutes in one hour -/
def minutes_per_hour : ℕ := 60

/-- The number of hours -/
def hours : ℚ := 3.5

/-- Theorem: The number of minutes in 3.5 hours is 210 -/
theorem minutes_in_three_and_half_hours : 
  (hours * minutes_per_hour : ℚ) = 210 := by sorry

end NUMINAMATH_CALUDE_minutes_in_three_and_half_hours_l1188_118859


namespace NUMINAMATH_CALUDE_fruit_profit_equation_l1188_118835

/-- Represents the profit equation for a fruit selling scenario -/
theorem fruit_profit_equation 
  (cost : ℝ) 
  (initial_price : ℝ) 
  (initial_volume : ℝ) 
  (price_increase : ℝ) 
  (volume_decrease : ℝ) 
  (profit : ℝ) :
  cost = 40 →
  initial_price = 50 →
  initial_volume = 500 →
  price_increase > 0 →
  volume_decrease = 10 * price_increase →
  profit = 8000 →
  ∃ x : ℝ, x > 50 ∧ (x - cost) * (initial_volume - volume_decrease) = profit :=
by sorry

end NUMINAMATH_CALUDE_fruit_profit_equation_l1188_118835


namespace NUMINAMATH_CALUDE_basketball_score_proof_l1188_118831

theorem basketball_score_proof (two_point_shots three_point_shots free_throws : ℕ) :
  (3 * three_point_shots = 2 * two_point_shots) →
  (free_throws = 2 * three_point_shots) →
  (2 * two_point_shots + 3 * three_point_shots + free_throws = 80) →
  free_throws = 20 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l1188_118831


namespace NUMINAMATH_CALUDE_line_intersection_range_l1188_118891

theorem line_intersection_range (a : ℝ) : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ 2 * x + (3 - a) = 0) ↔ 5 ≤ a ∧ a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_range_l1188_118891


namespace NUMINAMATH_CALUDE_tile_perimeter_theorem_l1188_118822

/-- Represents the shape of the tile configuration -/
inductive TileShape
  | L

/-- Represents the possible perimeters after adding tiles -/
def PossiblePerimeters : Set ℕ := {12, 14, 16}

/-- The initial tile configuration -/
structure InitialConfig where
  shape : TileShape
  tileCount : ℕ
  tileSize : ℕ
  perimeter : ℕ

/-- The configuration after adding tiles -/
structure FinalConfig where
  initial : InitialConfig
  addedTiles : ℕ

/-- Predicate to check if a perimeter is possible after adding tiles -/
def IsValidPerimeter (config : FinalConfig) (p : ℕ) : Prop :=
  p ∈ PossiblePerimeters

/-- Main theorem statement -/
theorem tile_perimeter_theorem (config : FinalConfig)
  (h1 : config.initial.shape = TileShape.L)
  (h2 : config.initial.tileCount = 8)
  (h3 : config.initial.tileSize = 1)
  (h4 : config.initial.perimeter = 12)
  (h5 : config.addedTiles = 2) :
  ∃ (p : ℕ), IsValidPerimeter config p :=
sorry

end NUMINAMATH_CALUDE_tile_perimeter_theorem_l1188_118822


namespace NUMINAMATH_CALUDE_liquid_X_percentage_l1188_118825

/-- The percentage of liquid X in solution A -/
def percentage_X_in_A : ℝ := 1.67

/-- The weight of solution A in grams -/
def weight_A : ℝ := 600

/-- The weight of solution B in grams -/
def weight_B : ℝ := 700

/-- The percentage of liquid X in solution B -/
def percentage_X_in_B : ℝ := 1.8

/-- The percentage of liquid X in the resulting mixture -/
def percentage_X_in_mixture : ℝ := 1.74

theorem liquid_X_percentage :
  (percentage_X_in_A * weight_A + percentage_X_in_B * weight_B) / (weight_A + weight_B) = percentage_X_in_mixture := by
  sorry

end NUMINAMATH_CALUDE_liquid_X_percentage_l1188_118825


namespace NUMINAMATH_CALUDE_direct_inverse_variation_l1188_118882

theorem direct_inverse_variation (k : ℝ) (R X Y : ℝ → ℝ) :
  (∀ t, R t = k * X t / Y t) →  -- R varies directly as X and inversely as Y
  R 0 = 10 ∧ X 0 = 2 ∧ Y 0 = 4 →  -- Initial condition
  R 1 = 8 ∧ Y 1 = 5 →  -- New condition
  X 1 = 2 :=  -- Conclusion
by sorry

end NUMINAMATH_CALUDE_direct_inverse_variation_l1188_118882


namespace NUMINAMATH_CALUDE_irregular_decagon_angle_l1188_118897

/-- Theorem: In a 10-sided polygon where the sum of all interior angles is 1470°,
    and 9 of the angles are equal, the measure of the non-equal angle is 174°. -/
theorem irregular_decagon_angle (n : ℕ) (sum : ℝ) (regular_angle : ℝ) (irregular_angle : ℝ) :
  n = 10 ∧ 
  sum = 1470 ∧
  (n - 1) * regular_angle + irregular_angle = sum ∧
  (n - 1) * regular_angle = (n - 2) * 180 →
  irregular_angle = 174 := by
  sorry

end NUMINAMATH_CALUDE_irregular_decagon_angle_l1188_118897


namespace NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l1188_118801

/-- A four-digit palindrome is a number between 1000 and 9999 of the form abba where a and b are digits and a ≠ 0 -/
def FourDigitPalindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem all_four_digit_palindromes_divisible_by_11 :
  ∀ n : ℕ, FourDigitPalindrome n → n % 11 = 0 := by
  sorry

#check all_four_digit_palindromes_divisible_by_11

end NUMINAMATH_CALUDE_all_four_digit_palindromes_divisible_by_11_l1188_118801


namespace NUMINAMATH_CALUDE_simple_interest_principal_l1188_118867

/-- Simple interest calculation -/
theorem simple_interest_principal (rate : ℚ) (time : ℚ) (interest : ℚ) (principal : ℚ) : 
  rate = 6/100 →
  time = 4 →
  interest = 192 →
  principal * rate * time = interest →
  principal = 800 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l1188_118867


namespace NUMINAMATH_CALUDE_cashback_discount_percentage_l1188_118880

theorem cashback_discount_percentage
  (iphone_price : ℝ)
  (iwatch_price : ℝ)
  (iphone_discount : ℝ)
  (iwatch_discount : ℝ)
  (total_after_cashback : ℝ)
  (h1 : iphone_price = 800)
  (h2 : iwatch_price = 300)
  (h3 : iphone_discount = 0.15)
  (h4 : iwatch_discount = 0.10)
  (h5 : total_after_cashback = 931) :
  let discounted_iphone := iphone_price * (1 - iphone_discount)
  let discounted_iwatch := iwatch_price * (1 - iwatch_discount)
  let total_after_discounts := discounted_iphone + discounted_iwatch
  let cashback_amount := total_after_discounts - total_after_cashback
  let cashback_percentage := cashback_amount / total_after_discounts * 100
  cashback_percentage = 2 := by sorry

end NUMINAMATH_CALUDE_cashback_discount_percentage_l1188_118880


namespace NUMINAMATH_CALUDE_perpendicular_AC_AD_l1188_118813

/-- The curve E in the xy-plane -/
def E : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + 3 * p.2^2 / 4 = 1 ∧ p.1 ≠ 2 ∧ p.1 ≠ -2}

/-- Point A -/
def A : ℝ × ℝ := (-2, 0)

/-- Point Q -/
def Q : ℝ × ℝ := (-1, 0)

/-- A line with non-zero slope passing through Q -/
def line_through_Q (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * (p.1 + 1) ∧ m ≠ 0}

/-- The intersection points of the line and curve E -/
def intersection (m : ℝ) : Set (ℝ × ℝ) :=
  E ∩ line_through_Q m

theorem perpendicular_AC_AD (m : ℝ) 
  (hm : m ≠ 0) 
  (h_intersect : ∃ C D, C ∈ intersection m ∧ D ∈ intersection m ∧ C ≠ D) :
  ∀ C D, C ∈ intersection m → D ∈ intersection m → C ≠ D →
  (C.1 + 2) * (D.1 + 2) + C.2 * D.2 = 0 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_AC_AD_l1188_118813


namespace NUMINAMATH_CALUDE_cookies_left_after_six_days_l1188_118839

/-- Represents the number of cookies baked and eaten over six days -/
structure CookieCount where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  parentEaten : ℕ
  neighborEaten : ℕ

/-- Calculates the total number of cookies left after six days -/
def totalCookiesLeft (c : CookieCount) : ℕ :=
  c.monday + c.tuesday + c.wednesday + c.thursday + c.friday + c.saturday - (c.parentEaten + c.neighborEaten)

/-- Theorem stating the number of cookies left after six days -/
theorem cookies_left_after_six_days :
  ∃ (c : CookieCount),
    c.monday = 32 ∧
    c.tuesday = c.monday / 2 ∧
    c.wednesday = (c.tuesday * 3) - 4 ∧
    c.thursday = (c.monday * 2) - 10 ∧
    c.friday = (c.tuesday * 3) - 6 ∧
    c.saturday = c.monday + c.friday ∧
    c.parentEaten = 2 * 6 ∧
    c.neighborEaten = 8 ∧
    totalCookiesLeft c = 242 := by
  sorry


end NUMINAMATH_CALUDE_cookies_left_after_six_days_l1188_118839


namespace NUMINAMATH_CALUDE_base_10_to_base_7_l1188_118865

theorem base_10_to_base_7 : 
  ∃ (a b c d : ℕ), 
    784 = a * 7^3 + b * 7^2 + c * 7^1 + d * 7^0 ∧ 
    a = 2 ∧ b = 2 ∧ c = 0 ∧ d = 0 := by
  sorry

end NUMINAMATH_CALUDE_base_10_to_base_7_l1188_118865


namespace NUMINAMATH_CALUDE_checkers_games_theorem_l1188_118824

theorem checkers_games_theorem (games_friend1 games_friend2 : ℕ) 
  (h1 : games_friend1 = 25) 
  (h2 : games_friend2 = 17) : 
  (∃ (x y z : ℕ), x + z = games_friend1 ∧ y + z = games_friend2 ∧ x + y = 34) ∧ 
  (¬ ∃ (x y z : ℕ), x + z = games_friend1 ∧ y + z = games_friend2 ∧ x + y = 35) ∧
  (¬ ∃ (x y z : ℕ), x + z = games_friend1 ∧ y + z = games_friend2 ∧ x + y = 56) :=
by sorry

#check checkers_games_theorem

end NUMINAMATH_CALUDE_checkers_games_theorem_l1188_118824


namespace NUMINAMATH_CALUDE_girls_count_in_class_l1188_118844

/-- Represents the number of people in each category (girls, boys, teachers) -/
structure ClassComposition where
  girls : ℕ
  boys : ℕ
  teachers : ℕ

/-- Proves that in a class of 60 people with a 3:2:1 ratio of girls:boys:teachers, there are 30 girls -/
theorem girls_count_in_class (c : ClassComposition) : 
  c.girls + c.boys + c.teachers = 60 →
  c.girls = 3 * c.teachers →
  c.boys = 2 * c.teachers →
  c.girls = 30 := by
  sorry

#check girls_count_in_class

end NUMINAMATH_CALUDE_girls_count_in_class_l1188_118844


namespace NUMINAMATH_CALUDE_divisor_totient_sum_bound_l1188_118877

/-- d(n) represents the number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- φ(n) represents Euler's totient function -/
def φ (n : ℕ+) : ℕ := sorry

/-- Theorem stating that c must be less than or equal to 1 -/
theorem divisor_totient_sum_bound (n : ℕ+) (c : ℕ) (h : d n + φ n = n + c) : c ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_divisor_totient_sum_bound_l1188_118877


namespace NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l1188_118834

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def in_fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The specific point we're considering -/
def point_B : Point2D :=
  { x := 3, y := -7 }

/-- Theorem stating that point_B is in the fourth quadrant -/
theorem point_B_in_fourth_quadrant : in_fourth_quadrant point_B := by
  sorry

end NUMINAMATH_CALUDE_point_B_in_fourth_quadrant_l1188_118834


namespace NUMINAMATH_CALUDE_smaller_circle_circumference_l1188_118802

theorem smaller_circle_circumference 
  (square_area : ℝ) 
  (larger_radius : ℝ) 
  (smaller_radius : ℝ) 
  (h1 : square_area = 784) 
  (h2 : square_area = (2 * larger_radius)^2) 
  (h3 : larger_radius = (7/3) * smaller_radius) : 
  2 * Real.pi * smaller_radius = 12 * Real.pi := by
sorry

end NUMINAMATH_CALUDE_smaller_circle_circumference_l1188_118802


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1188_118894

theorem compound_interest_problem :
  ∃ (P r : ℝ), P > 0 ∧ r > 0 ∧ 
  P * (1 + r)^2 = 8840 ∧
  P * (1 + r)^3 = 9261 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_problem_l1188_118894


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1188_118881

theorem fraction_subtraction : 7 - (2 / 5)^3 = 867 / 125 := by sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1188_118881


namespace NUMINAMATH_CALUDE_three_fourths_cubed_l1188_118846

theorem three_fourths_cubed : (3 / 4 : ℚ) ^ 3 = 27 / 64 := by sorry

end NUMINAMATH_CALUDE_three_fourths_cubed_l1188_118846


namespace NUMINAMATH_CALUDE_tims_bodyguard_payment_l1188_118870

/-- The amount Tim pays his bodyguards in a week -/
def weekly_payment (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

/-- Theorem stating the total amount Tim pays his bodyguards in a week -/
theorem tims_bodyguard_payment :
  weekly_payment 2 20 8 7 = 2240 := by
  sorry

#eval weekly_payment 2 20 8 7

end NUMINAMATH_CALUDE_tims_bodyguard_payment_l1188_118870


namespace NUMINAMATH_CALUDE_train_length_calculation_train2_length_l1188_118857

/-- Calculates the length of a train given the conditions of two trains passing each other. -/
theorem train_length_calculation (length_train1 : ℝ) (speed_train1 : ℝ) (speed_train2 : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := speed_train1 * 1000 / 3600 + speed_train2 * 1000 / 3600
  let total_distance := relative_speed * time_to_cross
  total_distance - length_train1

/-- The length of Train 2 is approximately 269.95 meters. -/
theorem train2_length : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_length_calculation 230 120 80 9 - 269.95| < ε :=
sorry

end NUMINAMATH_CALUDE_train_length_calculation_train2_length_l1188_118857


namespace NUMINAMATH_CALUDE_table_sum_theorem_l1188_118853

/-- A 3x3 table filled with numbers from 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- The sum of elements along a diagonal -/
def diagonalSum (t : Table) (main : Bool) : Nat :=
  if main then t 0 0 + t 1 1 + t 2 2 else t 0 2 + t 1 1 + t 2 0

/-- The sum of elements in the specified cells -/
def specifiedSum (t : Table) : Nat :=
  t 1 0 + t 1 1 + t 1 2 + t 2 1 + t 2 2

/-- All numbers from 1 to 9 appear exactly once in the table -/
def isValid (t : Table) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), t i j = n

theorem table_sum_theorem (t : Table) (h_valid : isValid t) 
  (h_diag1 : diagonalSum t true = 7) (h_diag2 : diagonalSum t false = 21) :
  specifiedSum t = 25 := by
  sorry

end NUMINAMATH_CALUDE_table_sum_theorem_l1188_118853


namespace NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l1188_118837

/-- The number of unique arrangements of n distinct beads on a bracelet, 
    considering rotational and reflectional symmetry -/
def bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem stating that the number of unique arrangements of 8 distinct beads 
    on a bracelet, considering rotational and reflectional symmetry, is 2520 -/
theorem eight_bead_bracelet_arrangements :
  bracelet_arrangements 8 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_eight_bead_bracelet_arrangements_l1188_118837


namespace NUMINAMATH_CALUDE_range_of_m_solution_set_l1188_118872

-- Define the function f
def f (x : ℝ) : ℝ := |x - 5| - |x - 2|

-- Theorem for the range of m
theorem range_of_m : 
  {m : ℝ | ∃ x, f x ≤ m} = {m : ℝ | m ≥ -3} := by sorry

-- Theorem for the solution set of the inequality
theorem solution_set : 
  {x : ℝ | x^2 - 8*x + 15 + f x ≤ 0} = {x : ℝ | 5 - Real.sqrt 3 ≤ x ∧ x ≤ 6} := by sorry

end NUMINAMATH_CALUDE_range_of_m_solution_set_l1188_118872


namespace NUMINAMATH_CALUDE_article_cost_l1188_118854

/-- Proves that the cost of an article is 80, given the specified conditions -/
theorem article_cost (original_profit_percent : Real) (reduced_cost_percent : Real)
  (price_reduction : Real) (new_profit_percent : Real)
  (h1 : original_profit_percent = 25)
  (h2 : reduced_cost_percent = 20)
  (h3 : price_reduction = 16.80)
  (h4 : new_profit_percent = 30) :
  ∃ (cost : Real), cost = 80 ∧
    (cost * (1 + original_profit_percent / 100) - price_reduction =
     (cost * (1 - reduced_cost_percent / 100)) * (1 + new_profit_percent / 100)) :=
by sorry

end NUMINAMATH_CALUDE_article_cost_l1188_118854


namespace NUMINAMATH_CALUDE_base8_palindrome_count_l1188_118883

/-- Represents a digit in base 8 -/
def Base8Digit := Fin 8

/-- Represents a six-digit palindrome in base 8 -/
structure Base8Palindrome where
  a : Base8Digit
  b : Base8Digit
  c : Base8Digit
  d : Base8Digit
  h : a.val ≠ 0

/-- The count of six-digit palindromes in base 8 -/
def count_base8_palindromes : Nat :=
  (Finset.range 7).card * (Finset.range 8).card * (Finset.range 8).card * (Finset.range 8).card

theorem base8_palindrome_count :
  count_base8_palindromes = 3584 :=
sorry

end NUMINAMATH_CALUDE_base8_palindrome_count_l1188_118883


namespace NUMINAMATH_CALUDE_x_equals_plus_minus_fifteen_l1188_118886

theorem x_equals_plus_minus_fifteen (x : ℝ) : 
  (x / 5) / 3 = 5 / (x / 3) → x = 15 ∨ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_plus_minus_fifteen_l1188_118886


namespace NUMINAMATH_CALUDE_shorts_weight_l1188_118817

/-- The maximum allowed weight for washing clothes -/
def max_weight : ℕ := 50

/-- The weight of a pair of socks in ounces -/
def sock_weight : ℕ := 2

/-- The weight of a pair of underwear in ounces -/
def underwear_weight : ℕ := 4

/-- The weight of a shirt in ounces -/
def shirt_weight : ℕ := 5

/-- The weight of a pair of pants in ounces -/
def pants_weight : ℕ := 10

/-- The number of pairs of pants Tony is washing -/
def num_pants : ℕ := 1

/-- The number of shirts Tony is washing -/
def num_shirts : ℕ := 2

/-- The number of pairs of socks Tony is washing -/
def num_socks : ℕ := 3

/-- The number of additional pairs of underwear Tony can add -/
def additional_underwear : ℕ := 4

/-- Theorem stating that the weight of a pair of shorts is 8 ounces -/
theorem shorts_weight :
  ∃ (shorts_weight : ℕ),
    shorts_weight = max_weight -
      (num_pants * pants_weight +
       num_shirts * shirt_weight +
       num_socks * sock_weight +
       additional_underwear * underwear_weight) :=
by sorry

end NUMINAMATH_CALUDE_shorts_weight_l1188_118817


namespace NUMINAMATH_CALUDE_visible_red_bus_length_l1188_118828

/-- Proves that the visible length of a red bus from a yellow bus is 6 feet, given specific length relationships between red, orange, and yellow buses. -/
theorem visible_red_bus_length 
  (red_bus_length : ℝ)
  (orange_car_length : ℝ)
  (yellow_bus_length : ℝ)
  (h1 : red_bus_length = 4 * orange_car_length)
  (h2 : yellow_bus_length = 3.5 * orange_car_length)
  (h3 : red_bus_length = 48) :
  red_bus_length - yellow_bus_length = 6 := by
  sorry

#check visible_red_bus_length

end NUMINAMATH_CALUDE_visible_red_bus_length_l1188_118828


namespace NUMINAMATH_CALUDE_complex_number_properties_l1188_118874

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := Complex.mk (m^2 - 8*m + 15) (m^2 - 5*m)

theorem complex_number_properties :
  (∃ m : ℝ, z m = Complex.I * (z m).im) ∧
  (∃ m : ℝ, z m = 3 + 6*Complex.I) ∧
  (∃ m : ℝ, 0 < m ∧ m < 3 ∧ (z m).re > 0 ∧ (z m).im < 0) :=
by sorry


end NUMINAMATH_CALUDE_complex_number_properties_l1188_118874


namespace NUMINAMATH_CALUDE_system_of_inequalities_solution_l1188_118842

theorem system_of_inequalities_solution (x : ℝ) :
  (4 * x^2 - 27 * x + 18 > 0 ∧ x^2 + 4 * x + 4 > 0) ↔ 
  ((x < 3/4 ∨ x > 6) ∧ x ≠ -2) :=
by sorry

end NUMINAMATH_CALUDE_system_of_inequalities_solution_l1188_118842


namespace NUMINAMATH_CALUDE_freshman_class_size_l1188_118804

theorem freshman_class_size :
  ∃! n : ℕ, n < 600 ∧ n % 19 = 15 ∧ n % 17 = 11 ∧ n = 53 := by
  sorry

end NUMINAMATH_CALUDE_freshman_class_size_l1188_118804


namespace NUMINAMATH_CALUDE_quadratic_function_theorem_l1188_118840

/-- A quadratic function with specific properties -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

/-- The derivative of a function -/
def HasDerivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, deriv f x = f' x

/-- A quadratic equation has two equal real roots -/
def HasEqualRoots (f : ℝ → ℝ) : Prop :=
  ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r) ∧ (∀ ε > 0, ∃ δ > 0, ∀ x, |x - r| < δ → |f x| < ε)

/-- The main theorem -/
theorem quadratic_function_theorem (f : ℝ → ℝ) 
  (h1 : QuadraticFunction f)
  (h2 : HasEqualRoots f)
  (h3 : HasDerivative f (λ x ↦ 2 * x + 2)) :
  ∀ x, f x = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_theorem_l1188_118840


namespace NUMINAMATH_CALUDE_car_impact_suitable_for_sampling_l1188_118888

/-- Characteristics of a suitable sampling survey scenario -/
structure SamplingSurveyCharacteristics where
  large_population : Bool
  impractical_full_survey : Bool
  representative_sample_possible : Bool

/-- Options for the survey scenario -/
inductive SurveyOption
  | A  -- Understanding the height of students in Class 7(1)
  | B  -- Companies recruiting and interviewing job applicants
  | C  -- Investigating the impact resistance of a batch of cars
  | D  -- Selecting the fastest runner in our school for competition

/-- Determine if an option is suitable for sampling survey -/
def is_suitable_for_sampling (option : SurveyOption) : Bool :=
  match option with
  | SurveyOption.C => true
  | _ => false

/-- Characteristics of the car impact resistance scenario -/
def car_impact_scenario : SamplingSurveyCharacteristics :=
  { large_population := true,
    impractical_full_survey := true,
    representative_sample_possible := true }

/-- Theorem stating that investigating car impact resistance is suitable for sampling survey -/
theorem car_impact_suitable_for_sampling :
  is_suitable_for_sampling SurveyOption.C ∧
  car_impact_scenario.large_population ∧
  car_impact_scenario.impractical_full_survey ∧
  car_impact_scenario.representative_sample_possible :=
sorry

end NUMINAMATH_CALUDE_car_impact_suitable_for_sampling_l1188_118888


namespace NUMINAMATH_CALUDE_fraction_order_l1188_118832

theorem fraction_order : 
  let f1 := 21 / 14
  let f2 := 25 / 18
  let f3 := 23 / 16
  let f4 := 27 / 19
  f2 < f4 ∧ f4 < f3 ∧ f3 < f1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_order_l1188_118832


namespace NUMINAMATH_CALUDE_infinite_series_equality_l1188_118827

theorem infinite_series_equality (a b : ℝ) 
  (h : ∑' n, a / b^n = 6) : 
  ∑' n, a / (a + b)^n = 6/7 := by
sorry

end NUMINAMATH_CALUDE_infinite_series_equality_l1188_118827


namespace NUMINAMATH_CALUDE_sqrt_square_eq_abs_l1188_118833

theorem sqrt_square_eq_abs (x : ℝ) : Real.sqrt (x^2) = |x| := by
  sorry

end NUMINAMATH_CALUDE_sqrt_square_eq_abs_l1188_118833


namespace NUMINAMATH_CALUDE_trapezium_longer_side_length_l1188_118899

/-- Given a trapezium with the following properties:
    - One parallel side is 10 cm long
    - The distance between parallel sides is 15 cm
    - The area is 210 square centimeters
    This theorem proves that the length of the other parallel side is 18 cm. -/
theorem trapezium_longer_side_length (a b h : ℝ) : 
  a = 10 → h = 15 → (a + b) * h / 2 = 210 → b = 18 :=
by sorry

end NUMINAMATH_CALUDE_trapezium_longer_side_length_l1188_118899


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1188_118810

theorem cone_base_circumference (r : ℝ) (angle : ℝ) :
  r = 6 →
  angle = 300 →
  let full_circumference := 2 * Real.pi * r
  let remaining_fraction := angle / 360
  let cone_base_circumference := remaining_fraction * full_circumference
  cone_base_circumference = 10 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l1188_118810


namespace NUMINAMATH_CALUDE_balloon_comparison_l1188_118858

/-- The number of balloons Allan initially brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- The number of additional balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The total number of balloons Allan had in the park -/
def allan_total : ℕ := allan_initial + allan_bought

/-- The difference between Jake's balloons and Allan's total balloons -/
def balloon_difference : ℕ := jake_balloons - allan_total

theorem balloon_comparison : balloon_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_balloon_comparison_l1188_118858


namespace NUMINAMATH_CALUDE_number_of_boys_in_class_l1188_118862

/-- Proves the number of boys in a class given certain height information -/
theorem number_of_boys_in_class 
  (initial_average : ℝ)
  (wrong_height : ℝ)
  (actual_height : ℝ)
  (actual_average : ℝ)
  (h1 : initial_average = 183)
  (h2 : wrong_height = 166)
  (h3 : actual_height = 106)
  (h4 : actual_average = 181) :
  ∃ n : ℕ, n * initial_average - (wrong_height - actual_height) = n * actual_average ∧ n = 30 :=
by sorry

end NUMINAMATH_CALUDE_number_of_boys_in_class_l1188_118862


namespace NUMINAMATH_CALUDE_tangent_slope_at_one_l1188_118876

-- Define a differentiable function f
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the limit condition
variable (h : ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((f 1 - f (1 + 2*x)) / (2*x)) - 1| < ε)

-- State the theorem
theorem tangent_slope_at_one (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ ε > 0, ∃ δ > 0, ∀ x ≠ 0, |x| < δ → |((f 1 - f (1 + 2*x)) / (2*x)) - 1| < ε) : 
  deriv f 1 = -1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_at_one_l1188_118876


namespace NUMINAMATH_CALUDE_quadratic_minimum_l1188_118843

theorem quadratic_minimum (k : ℝ) : 
  (∃ x₀ ∈ Set.Icc 0 2, ∀ x ∈ Set.Icc 0 2, 
    (x^2 - 4*k*x + 4*k^2 + 2*k - 1) ≥ (x₀^2 - 4*k*x₀ + 4*k^2 + 2*k - 1)) → 
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l1188_118843


namespace NUMINAMATH_CALUDE_intersection_of_M_and_P_l1188_118811

-- Define the sets M and P
def M : Set ℝ := {y | ∃ x, y = 3^x}
def P : Set ℝ := {y | y ≥ 1}

-- State the theorem
theorem intersection_of_M_and_P : M ∩ P = {y | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_P_l1188_118811


namespace NUMINAMATH_CALUDE_gear_q_revolutions_per_minute_l1188_118814

/-- The number of revolutions per minute for gear p -/
def p_rev_per_min : ℚ := 10

/-- The number of seconds in the given time interval -/
def time_interval : ℚ := 10

/-- The additional revolutions gear q makes compared to gear p in the given time interval -/
def additional_rev : ℚ := 5

/-- The number of seconds in a minute -/
def seconds_per_minute : ℚ := 60

theorem gear_q_revolutions_per_minute :
  let p_rev_in_interval := p_rev_per_min * time_interval / seconds_per_minute
  let q_rev_in_interval := p_rev_in_interval + additional_rev
  let q_rev_per_min := q_rev_in_interval * seconds_per_minute / time_interval
  q_rev_per_min = 40 := by
  sorry

end NUMINAMATH_CALUDE_gear_q_revolutions_per_minute_l1188_118814


namespace NUMINAMATH_CALUDE_children_tickets_sold_l1188_118845

theorem children_tickets_sold (child_price adult_price total_tickets total_amount : ℕ) 
  (h1 : child_price = 6)
  (h2 : adult_price = 9)
  (h3 : total_tickets = 225)
  (h4 : total_amount = 1875) :
  ∃ (children adults : ℕ),
    children + adults = total_tickets ∧
    child_price * children + adult_price * adults = total_amount ∧
    children = 50 := by
  sorry

end NUMINAMATH_CALUDE_children_tickets_sold_l1188_118845


namespace NUMINAMATH_CALUDE_m_eq_2_sufficient_not_necessary_l1188_118875

def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

theorem m_eq_2_sufficient_not_necessary :
  (∃ m : ℝ, (A m) ∩ B = {4} ∧ m ≠ 2) ∧
  (∀ m : ℝ, m = 2 → (A m) ∩ B = {4}) :=
sorry

end NUMINAMATH_CALUDE_m_eq_2_sufficient_not_necessary_l1188_118875


namespace NUMINAMATH_CALUDE_f_properties_l1188_118805

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem f_properties :
  ∃ (x₀ : ℝ),
    (∀ x > 0, HasDerivAt f (Real.log x + 1) x) ∧
    (HasDerivAt f 2 x₀ → x₀ = Real.exp 1) ∧
    (∀ x ≥ Real.exp (-1), StrictMono f) ∧
    (∃! p, f p = -Real.exp (-1)) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1188_118805


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1188_118819

theorem sum_of_fractions_equals_one (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) :
  (a^3*b^3 / ((a^3 - b*c)*(b^3 - a*c))) + 
  (a^3*c^3 / ((a^3 - b*c)*(c^3 - a*b))) + 
  (b^3*c^3 / ((b^3 - a*c)*(c^3 - a*b))) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_one_l1188_118819


namespace NUMINAMATH_CALUDE_square_side_length_l1188_118871

theorem square_side_length (total_width total_height : ℕ) 
  (h_width : total_width = 4040)
  (h_height : total_height = 2420)
  (h_rectangles_equal : ∃ (r : ℕ), r = r) -- R₁ and R₂ have identical dimensions
  (h_squares : ∃ (s r : ℕ), s + r = s + r) -- S₁ and S₃ side length = S₂ side length + R₁ side length
  : ∃ (s : ℕ), s = 810 ∧ 
    ∃ (r : ℕ), 2 * r + s = total_height ∧ 
                2 * r + 3 * s = total_width :=
by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1188_118871


namespace NUMINAMATH_CALUDE_sin_2theta_value_l1188_118851

theorem sin_2theta_value (θ : Real) 
  (h1 : 0 < θ) (h2 : θ < π/2) 
  (h3 : Real.cos (π/4 - θ) * Real.cos (π/4 + θ) = Real.sqrt 2 / 6) : 
  Real.sin (2 * θ) = Real.sqrt 7 / 3 := by
sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l1188_118851


namespace NUMINAMATH_CALUDE_angle_B_value_side_lengths_l1188_118847

noncomputable section

-- Define the triangle ABC
variable (A B C : ℝ) -- Angles
variable (a b c : ℝ) -- Side lengths

-- Define the conditions
axiom triangle_condition : 2 * a * c * Real.sin B = Real.sqrt 3 * (a^2 + b^2 - c^2)
axiom side_b : b = 3
axiom angle_relation : Real.sin C = 2 * Real.sin A

-- Theorem 1: Prove that B = π/3
theorem angle_B_value : 
  2 * a * c * Real.sin B = Real.sqrt 3 * (a^2 + b^2 - c^2) → B = π/3 := by sorry

-- Theorem 2: Prove that a = √3 and c = 2√3
theorem side_lengths : 
  b = 3 → 
  Real.sin C = 2 * Real.sin A → 
  2 * a * c * Real.sin B = Real.sqrt 3 * (a^2 + b^2 - c^2) → 
  a = Real.sqrt 3 ∧ c = 2 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_angle_B_value_side_lengths_l1188_118847


namespace NUMINAMATH_CALUDE_problem_8_l1188_118896

theorem problem_8 (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + b^2 + c^2 = 63)
  (h2 : 2*a + 3*b + 6*c = 21*Real.sqrt 7) :
  (a/c)^(a/b) = (1/3)^(2/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_8_l1188_118896


namespace NUMINAMATH_CALUDE_project_completion_time_l1188_118836

theorem project_completion_time 
  (workers_initial : ℕ) 
  (days_initial : ℕ) 
  (workers_new : ℕ) 
  (h1 : workers_initial = 60) 
  (h2 : days_initial = 3) 
  (h3 : workers_new = 30) :
  workers_initial * days_initial = workers_new * (2 * days_initial) :=
by sorry

end NUMINAMATH_CALUDE_project_completion_time_l1188_118836


namespace NUMINAMATH_CALUDE_cookie_distribution_ways_l1188_118812

/-- The number of ways to distribute cookies among students -/
def distribute_cookies (total_cookies : ℕ) (num_students : ℕ) (min_cookies : ℕ) : ℕ :=
  Nat.choose (total_cookies - num_students * min_cookies + num_students - 1) (num_students - 1)

/-- Theorem: The number of ways to distribute 30 cookies among 5 students, 
    with each student receiving at least 3 cookies, is 3876 -/
theorem cookie_distribution_ways : distribute_cookies 30 5 3 = 3876 := by
  sorry

end NUMINAMATH_CALUDE_cookie_distribution_ways_l1188_118812


namespace NUMINAMATH_CALUDE_shari_walk_distance_l1188_118868

/-- Calculates the distance walked given a constant walking speed, total time, and break time. -/
def distance_walked (speed : ℝ) (total_time : ℝ) (break_time : ℝ) : ℝ :=
  speed * (total_time - break_time)

/-- Proves that walking at 4 miles per hour for 2 hours with a 30-minute break results in 6 miles walked. -/
theorem shari_walk_distance :
  let speed : ℝ := 4
  let total_time : ℝ := 2
  let break_time : ℝ := 0.5
  distance_walked speed total_time break_time = 6 := by sorry

end NUMINAMATH_CALUDE_shari_walk_distance_l1188_118868


namespace NUMINAMATH_CALUDE_book_price_increase_l1188_118849

theorem book_price_increase (original_price : ℝ) (new_price : ℝ) (increase_percentage : ℝ) : 
  new_price = 330 ∧ 
  increase_percentage = 10 ∧ 
  new_price = original_price * (1 + increase_percentage / 100) →
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l1188_118849


namespace NUMINAMATH_CALUDE_exists_points_on_hyperbola_with_midpoint_l1188_118869

/-- The hyperbola equation --/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/9 = 1

/-- Definition of a midpoint --/
def is_midpoint (x₀ y₀ x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2

theorem exists_points_on_hyperbola_with_midpoint :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    hyperbola x₁ y₁ ∧ 
    hyperbola x₂ y₂ ∧ 
    is_midpoint (-1) (-4) x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_exists_points_on_hyperbola_with_midpoint_l1188_118869


namespace NUMINAMATH_CALUDE_exists_fib_div_1000_l1188_118830

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Theorem: There exists a Fibonacci number divisible by 1000 -/
theorem exists_fib_div_1000 : ∃ n : ℕ, 1000 ∣ fib n := by
  sorry

end NUMINAMATH_CALUDE_exists_fib_div_1000_l1188_118830


namespace NUMINAMATH_CALUDE_nine_possible_H_values_l1188_118850

/-- A function that represents the number formed by digits E, F, G, G, F --/
def EFGGF (E F G : Nat) : Nat := 10000 * E + 1000 * F + 100 * G + 10 * G + F

/-- A function that represents the number formed by digits F, G, E, E, H --/
def FGEEH (F G E H : Nat) : Nat := 10000 * F + 1000 * G + 100 * E + 10 * E + H

/-- A function that represents the number formed by digits H, F, H, H, H --/
def HFHHH (H F : Nat) : Nat := 10000 * H + 1000 * F + 100 * H + 10 * H + H

/-- The main theorem stating that there are exactly 9 possible values for H --/
theorem nine_possible_H_values (E F G H : Nat) :
  (E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10) →  -- E, F, G, H are digits
  (E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ F ≠ G ∧ F ≠ H ∧ G ≠ H) →  -- E, F, G, H are distinct
  (EFGGF E F G + FGEEH F G E H = HFHHH H F) →  -- The addition equation
  (∃! (s : Finset Nat), s.card = 9 ∧ ∀ h, h ∈ s ↔ ∃ E F G, EFGGF E F G + FGEEH F G E h = HFHHH h F) :=
by sorry


end NUMINAMATH_CALUDE_nine_possible_H_values_l1188_118850


namespace NUMINAMATH_CALUDE_rectangle_formation_ways_l1188_118818

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of horizontal lines -/
def num_horizontal_lines : ℕ := 4

/-- The number of vertical lines -/
def num_vertical_lines : ℕ := 4

/-- The number of lines needed to form a side of the rectangle -/
def lines_per_side : ℕ := 2

/-- Theorem stating that the number of ways to choose lines to form a rectangle is 36 -/
theorem rectangle_formation_ways :
  (choose num_horizontal_lines lines_per_side) * (choose num_vertical_lines lines_per_side) = 36 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_formation_ways_l1188_118818


namespace NUMINAMATH_CALUDE_sphere_area_ratio_l1188_118829

theorem sphere_area_ratio (r₁ r₂ A₁ : ℝ) (h₁ : r₁ > 0) (h₂ : r₂ > 0) (h₃ : A₁ > 0) :
  let A₂ := A₁ * (r₂ / r₁)^2
  r₁ = 4 ∧ r₂ = 6 ∧ A₁ = 37 → A₂ = 83.25 := by
  sorry

end NUMINAMATH_CALUDE_sphere_area_ratio_l1188_118829


namespace NUMINAMATH_CALUDE_two_heads_in_three_tosses_l1188_118893

/-- The probability of getting exactly k successes in n trials with probability p of success on each trial. -/
def binomialProbability (n k : ℕ) (p : ℝ) : ℝ :=
  Nat.choose n k * p^k * (1 - p)^(n - k)

/-- Theorem: The probability of getting exactly 2 heads when a fair coin is tossed 3 times is 0.375 -/
theorem two_heads_in_three_tosses :
  binomialProbability 3 2 (1/2) = 0.375 := by
  sorry

end NUMINAMATH_CALUDE_two_heads_in_three_tosses_l1188_118893


namespace NUMINAMATH_CALUDE_largest_odd_proper_divisor_ratio_l1188_118815

/-- The largest odd proper divisor of a positive integer -/
def f (n : ℕ) : ℕ :=
  sorry

/-- The main theorem -/
theorem largest_odd_proper_divisor_ratio :
  let N : ℕ := 20^23 * 23^20
  f N / f (f (f N)) = 25 :=
by sorry

end NUMINAMATH_CALUDE_largest_odd_proper_divisor_ratio_l1188_118815


namespace NUMINAMATH_CALUDE_arithmetic_evaluation_l1188_118816

theorem arithmetic_evaluation : (300 + 5 * 8) / (2^3 : ℝ) = 42.5 := by sorry

end NUMINAMATH_CALUDE_arithmetic_evaluation_l1188_118816


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1188_118873

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) :
  selling_price = 150 ∧ profit_percentage = 25 →
  ∃ (cost_price : ℝ), cost_price = 120 ∧
    selling_price = cost_price * (1 + profit_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1188_118873


namespace NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l1188_118860

theorem square_sum_from_product_and_sum (x y : ℝ) 
  (h1 : x * y = 12) 
  (h2 : x + y = 10) : 
  x^2 + y^2 = 76 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_product_and_sum_l1188_118860


namespace NUMINAMATH_CALUDE_door_blocked_time_l1188_118863

/-- Represents a clock with a door near its center -/
structure Clock :=
  (door_blocked_by_minute_hand : ℕ → Bool)
  (door_blocked_by_hour_hand : ℕ → Bool)

/-- The duration of a day in minutes -/
def day_minutes : ℕ := 24 * 60

/-- Checks if the door is blocked at a given minute -/
def is_door_blocked (clock : Clock) (minute : ℕ) : Bool :=
  clock.door_blocked_by_minute_hand minute ∨ clock.door_blocked_by_hour_hand minute

/-- Counts the number of minutes the door is blocked in a day -/
def blocked_minutes (clock : Clock) : ℕ :=
  (List.range day_minutes).filter (is_door_blocked clock) |>.length

/-- The theorem stating that the door is blocked for 498 minutes per day -/
theorem door_blocked_time (clock : Clock) 
  (h1 : ∀ (hour : ℕ) (minute : ℕ), hour < 24 → minute < 60 → 
    clock.door_blocked_by_minute_hand (hour * 60 + minute) = (9 ≤ minute ∧ minute < 21))
  (h2 : ∀ (minute : ℕ), minute < day_minutes → 
    clock.door_blocked_by_hour_hand minute = 
      ((108 ≤ minute % 720 ∧ minute % 720 < 252) ∨ 
       (828 ≤ minute % 720 ∧ minute % 720 < 972))) :
  blocked_minutes clock = 498 := by
  sorry


end NUMINAMATH_CALUDE_door_blocked_time_l1188_118863


namespace NUMINAMATH_CALUDE_square_root_probability_l1188_118889

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def count_valid_numbers : ℕ := 71

def total_two_digit_numbers : ℕ := 90

theorem square_root_probability : 
  (count_valid_numbers : ℚ) / total_two_digit_numbers = 71 / 90 := by sorry

end NUMINAMATH_CALUDE_square_root_probability_l1188_118889


namespace NUMINAMATH_CALUDE_max_an_over_n_is_half_l1188_118826

/-- The number of trailing zeroes in the base-n representation of n! -/
def a (n : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the maximum value of a_n/n is 1/2 -/
theorem max_an_over_n_is_half :
  (∀ n > 1, (a n : ℚ) / n ≤ 1/2) ∧ (∃ n > 1, (a n : ℚ) / n = 1/2) :=
sorry

end NUMINAMATH_CALUDE_max_an_over_n_is_half_l1188_118826


namespace NUMINAMATH_CALUDE_sock_ratio_l1188_118800

/-- The ratio of black socks to blue socks in an order satisfying certain conditions -/
theorem sock_ratio :
  ∀ (b : ℕ) (x : ℝ),
  x > 0 →
  (18 * x + b * x) * 1.6 = 3 * b * x + 6 * x →
  (6 : ℝ) / b = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_sock_ratio_l1188_118800


namespace NUMINAMATH_CALUDE_min_value_expression_l1188_118807

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x - 1)^2 / (y - 2) + (y - 1)^2 / (x - 2) ≥ 8 ∧
  (∃ x y : ℝ, x > 2 ∧ y > 2 ∧ (x - 1)^2 / (y - 2) + (y - 1)^2 / (x - 2) = 8) :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l1188_118807


namespace NUMINAMATH_CALUDE_hospital_bed_charge_l1188_118821

theorem hospital_bed_charge 
  (days_in_hospital : ℕ) 
  (specialist_hourly_rate : ℚ) 
  (specialist_time : ℚ) 
  (num_specialists : ℕ) 
  (ambulance_cost : ℚ) 
  (total_bill : ℚ) :
  days_in_hospital = 3 →
  specialist_hourly_rate = 250 →
  specialist_time = 1/4 →
  num_specialists = 2 →
  ambulance_cost = 1800 →
  total_bill = 4625 →
  let daily_bed_charge := (total_bill - num_specialists * specialist_hourly_rate * specialist_time - ambulance_cost) / days_in_hospital
  daily_bed_charge = 900 := by
sorry

end NUMINAMATH_CALUDE_hospital_bed_charge_l1188_118821


namespace NUMINAMATH_CALUDE_loss_percent_calculation_l1188_118884

def cost_price : ℝ := 600
def selling_price : ℝ := 550

theorem loss_percent_calculation :
  let loss := cost_price - selling_price
  let loss_percent := (loss / cost_price) * 100
  ∃ ε > 0, abs (loss_percent - 8.33) < ε :=
by sorry

end NUMINAMATH_CALUDE_loss_percent_calculation_l1188_118884


namespace NUMINAMATH_CALUDE_cyclic_sum_factorization_l1188_118879

theorem cyclic_sum_factorization (a b c : ℝ) :
  a * (b - c)^4 + b * (c - a)^4 + c * (a - b)^4 =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + a*b + b*c + c*a) :=
by sorry

end NUMINAMATH_CALUDE_cyclic_sum_factorization_l1188_118879


namespace NUMINAMATH_CALUDE_range_of_a_l1188_118808

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → |x^2 - a| + |x + a| = |x^2 + x|) → 
  a ∈ Set.Icc (-1) 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1188_118808


namespace NUMINAMATH_CALUDE_books_bought_equals_difference_l1188_118878

/-- Represents the number of books Melanie bought at the yard sale -/
def books_bought : ℕ := sorry

/-- Melanie's initial number of books -/
def initial_books : ℕ := 41

/-- Melanie's final number of books after the yard sale -/
def final_books : ℕ := 87

/-- Theorem stating that the number of books bought is the difference between final and initial books -/
theorem books_bought_equals_difference : 
  books_bought = final_books - initial_books :=
by sorry

end NUMINAMATH_CALUDE_books_bought_equals_difference_l1188_118878


namespace NUMINAMATH_CALUDE_square_to_rectangle_ratio_l1188_118841

theorem square_to_rectangle_ratio : 
  ∀ (square_side : ℝ) (rectangle_base rectangle_height : ℝ),
  square_side = 4 →
  rectangle_base = 2 * Real.sqrt 5 →
  rectangle_height * rectangle_base = square_side^2 →
  rectangle_height / rectangle_base = 0.8 := by
  sorry

end NUMINAMATH_CALUDE_square_to_rectangle_ratio_l1188_118841


namespace NUMINAMATH_CALUDE_sqrt_product_simplification_l1188_118887

theorem sqrt_product_simplification (y : ℝ) (h : y > 0) :
  Real.sqrt (48 * y) * Real.sqrt (3 * y) * Real.sqrt (50 * y) = 30 * y * Real.sqrt (2 * y) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_product_simplification_l1188_118887


namespace NUMINAMATH_CALUDE_largest_square_4digits_base7_l1188_118809

/-- The largest integer whose square has exactly 4 digits in base 7 -/
def M : ℕ := 48

/-- Conversion of a natural number to its base 7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem largest_square_4digits_base7 :
  (M^2 ≥ 7^3) ∧ (M^2 < 7^4) ∧ (∀ n : ℕ, n > M → n^2 ≥ 7^4) ∧ (toBase7 M = [6, 6]) := by
  sorry

end NUMINAMATH_CALUDE_largest_square_4digits_base7_l1188_118809


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1188_118885

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 7 * I
  let z₂ : ℂ := 4 - 7 * I
  (z₁ / z₂) + (z₂ / z₁) = -66 / 65 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1188_118885


namespace NUMINAMATH_CALUDE_square_roots_ratio_l1188_118890

-- Define the complex polynomial z^2 + az + b
def complex_polynomial (a b z : ℂ) : ℂ := z^2 + a*z + b

-- Define the theorem
theorem square_roots_ratio (a b z₁ : ℂ) :
  (complex_polynomial a b z₁ = 0) →
  (complex_polynomial a b (Complex.I * z₁) = 0) →
  a^2 / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_ratio_l1188_118890


namespace NUMINAMATH_CALUDE_print_shop_Y_charge_l1188_118820

/-- The charge per color copy at print shop X -/
def charge_X : ℚ := 1.25

/-- The number of copies being compared -/
def num_copies : ℕ := 80

/-- The additional charge at print shop Y for the given number of copies -/
def additional_charge : ℚ := 120

/-- The charge per color copy at print shop Y -/
def charge_Y : ℚ := (charge_X * num_copies + additional_charge) / num_copies

theorem print_shop_Y_charge : charge_Y = 2.75 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_Y_charge_l1188_118820


namespace NUMINAMATH_CALUDE_triangle_sides_proportion_l1188_118856

/-- Represents a triangle with sides a, b, c and incircle diameter 2r --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ

/-- 
  Theorem: If the lengths of the sides of a triangle and the diameter of its incircle 
  form four consecutive terms of an arithmetic progression, then the sides of the 
  triangle are proportional to 3, 4, and 5.
--/
theorem triangle_sides_proportion (t : Triangle) : 
  (∃ (d : ℝ), d > 0 ∧ t.a = t.r * 2 + d ∧ t.b = t.r * 2 + 2 * d ∧ t.c = t.r * 2 + 3 * d) →
  ∃ (k : ℝ), k > 0 ∧ t.a = 3 * k ∧ t.b = 4 * k ∧ t.c = 5 * k :=
by sorry

end NUMINAMATH_CALUDE_triangle_sides_proportion_l1188_118856


namespace NUMINAMATH_CALUDE_factorization_proof_l1188_118852

theorem factorization_proof (a b : ℝ) : a * b^2 - 25 * a = a * (b + 5) * (b - 5) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1188_118852


namespace NUMINAMATH_CALUDE_problem_statement_l1188_118898

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, x^2 + (a-1)*x + a^2 > 0

def q : Prop := ∀ x₁ x₂ : ℝ, x₁ < x₂ → (2*a^2 - a)^x₁ < (2*a^2 - a)^x₂

theorem problem_statement : (p a ∨ q a) → (a < -1/2 ∨ a > 1/3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1188_118898


namespace NUMINAMATH_CALUDE_eugene_shoes_count_l1188_118866

/-- The cost of a T-shirt before discount -/
def t_shirt_cost : ℚ := 20

/-- The cost of a pair of pants before discount -/
def pants_cost : ℚ := 80

/-- The cost of a pair of shoes before discount -/
def shoes_cost : ℚ := 150

/-- The discount rate applied to all items -/
def discount_rate : ℚ := 1/10

/-- The number of T-shirts Eugene buys -/
def num_tshirts : ℕ := 4

/-- The number of pairs of pants Eugene buys -/
def num_pants : ℕ := 3

/-- The total amount Eugene pays -/
def total_paid : ℚ := 558

/-- The function to calculate the discounted price -/
def discounted_price (price : ℚ) : ℚ := price * (1 - discount_rate)

/-- The theorem stating the number of pairs of shoes Eugene buys -/
theorem eugene_shoes_count :
  ∃ (n : ℕ), n * discounted_price shoes_cost = 
    total_paid - (num_tshirts * discounted_price t_shirt_cost + num_pants * discounted_price pants_cost) ∧
    n = 2 := by sorry

end NUMINAMATH_CALUDE_eugene_shoes_count_l1188_118866


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1188_118838

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (x + 12) = 10 → x = 88 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1188_118838


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_and_k_l1188_118892

/-- Given a quadratic equation x^2 + 3x + k = 0 where x = -3 is a root, 
    prove that the other root is 0 and k = 0. -/
theorem quadratic_equation_roots_and_k (k : ℝ) : 
  ((-3 : ℝ)^2 + 3*(-3) + k = 0) → 
  (∃ (r : ℝ), r ≠ -3 ∧ r^2 + 3*r + k = 0 ∧ r = 0) ∧ 
  (k = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_and_k_l1188_118892


namespace NUMINAMATH_CALUDE_amara_clothing_proof_l1188_118864

def initial_clothing (donated_first : ℕ) (donated_second : ℕ) (thrown_away : ℕ) (remaining : ℕ) : ℕ :=
  remaining + donated_first + donated_second + thrown_away

theorem amara_clothing_proof :
  let donated_first := 5
  let donated_second := 3 * donated_first
  let thrown_away := 15
  let remaining := 65
  initial_clothing donated_first donated_second thrown_away remaining = 100 := by
  sorry

end NUMINAMATH_CALUDE_amara_clothing_proof_l1188_118864


namespace NUMINAMATH_CALUDE_factorization_proof_l1188_118823

theorem factorization_proof (a b : ℝ) : 4 * a^2 * (a - b) - (a - b) = (a - b) * (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l1188_118823


namespace NUMINAMATH_CALUDE_hayley_initial_meatballs_hayley_initial_meatballs_proof_l1188_118895

theorem hayley_initial_meatballs : ℕ → ℕ → ℕ → Prop :=
  fun initial_meatballs stolen_meatballs remaining_meatballs =>
    (stolen_meatballs = 14) →
    (remaining_meatballs = 11) →
    (initial_meatballs = stolen_meatballs + remaining_meatballs) →
    (initial_meatballs = 25)

-- Proof
theorem hayley_initial_meatballs_proof :
  hayley_initial_meatballs 25 14 11 := by
  sorry

end NUMINAMATH_CALUDE_hayley_initial_meatballs_hayley_initial_meatballs_proof_l1188_118895


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1188_118855

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 6x^2 - 14x + 10 has discriminant -44 -/
theorem quadratic_discriminant :
  discriminant 6 (-14) 10 = -44 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1188_118855


namespace NUMINAMATH_CALUDE_intersection_sum_l1188_118806

theorem intersection_sum (m b : ℝ) : 
  (2 * m * 3 + 3 = 9) →  -- First line passes through (3, 9)
  (4 * 3 + b = 9) →      -- Second line passes through (3, 9)
  b + 2 * m = -1 := by
sorry

end NUMINAMATH_CALUDE_intersection_sum_l1188_118806


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1188_118861

theorem polynomial_simplification (x : ℝ) :
  (2 * x^4 + 3 * x^3 - 5 * x^2 + 6 * x - 8) + (-7 * x^4 - 4 * x^3 + 2 * x^2 - 6 * x + 15) =
  -5 * x^4 - x^3 - 3 * x^2 + 7 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1188_118861
