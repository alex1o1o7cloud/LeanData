import Mathlib

namespace NUMINAMATH_CALUDE_scores_relative_to_average_l1555_155586

def scores : List ℤ := [95, 86, 90, 87, 92]
def average : ℚ := 90

theorem scores_relative_to_average :
  let relative_scores := scores.map (λ s => s - average)
  relative_scores = [5, -4, 0, -3, 2] := by
  sorry

end NUMINAMATH_CALUDE_scores_relative_to_average_l1555_155586


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l1555_155517

theorem quadratic_equation_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 6)
  (h_geometric : Real.sqrt (a * b) = 5) :
  ∃ (x : ℝ), x^2 - 12*x + 25 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l1555_155517


namespace NUMINAMATH_CALUDE_palindrome_difference_unique_l1555_155521

/-- A four-digit palindromic integer -/
def FourDigitPalindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ (a d : ℕ), n = 1001 * a + 110 * d ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

/-- A three-digit palindromic integer -/
def ThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ (c f : ℕ), n = 101 * c + 10 * f ∧ 1 ≤ c ∧ c ≤ 9 ∧ 0 ≤ f ∧ f ≤ 9

theorem palindrome_difference_unique :
  ∀ A B C : ℕ,
  FourDigitPalindrome A →
  FourDigitPalindrome B →
  ThreeDigitPalindrome C →
  A - B = C →
  C = 121 := by
  sorry

end NUMINAMATH_CALUDE_palindrome_difference_unique_l1555_155521


namespace NUMINAMATH_CALUDE_absolute_value_problem_l1555_155537

theorem absolute_value_problem (a b c : ℝ) 
  (ha : |a| = 5)
  (hb : |b| = 3)
  (hc : |c| = 6)
  (hab : |a+b| = -(a+b))
  (hac : |a+c| = a+c) :
  a - b + c = 4 ∨ a - b + c = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_problem_l1555_155537


namespace NUMINAMATH_CALUDE_compare_exponentials_l1555_155548

theorem compare_exponentials (h : 0 < 0.5 ∧ 0.5 < 1) : 0.5^(-2) > 0.5^(-0.8) := by
  sorry

end NUMINAMATH_CALUDE_compare_exponentials_l1555_155548


namespace NUMINAMATH_CALUDE_water_volume_in_cylinder_l1555_155500

theorem water_volume_in_cylinder (r : ℝ) (h : r = 2) : 
  let cylinder_base_area := π * r^2
  let ball_volume := (4/3) * π * r^3
  let water_height_with_ball := 2 * r
  let total_volume_with_ball := cylinder_base_area * water_height_with_ball
  let original_water_volume := total_volume_with_ball - ball_volume
  original_water_volume = (16 * π) / 3 := by
sorry

end NUMINAMATH_CALUDE_water_volume_in_cylinder_l1555_155500


namespace NUMINAMATH_CALUDE_john_average_increase_l1555_155543

def john_scores : List ℝ := [92, 85, 91, 95]

theorem john_average_increase :
  let first_three_avg := (john_scores.take 3).sum / 3
  let all_four_avg := john_scores.sum / 4
  all_four_avg - first_three_avg = 1.42 := by sorry

end NUMINAMATH_CALUDE_john_average_increase_l1555_155543


namespace NUMINAMATH_CALUDE_fraction_subtraction_l1555_155555

theorem fraction_subtraction (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  1 / x - 1 / y = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l1555_155555


namespace NUMINAMATH_CALUDE_housing_relocation_problem_l1555_155505

/-- Represents the housing relocation problem -/
theorem housing_relocation_problem 
  (household_area : ℝ) 
  (initial_green_space_ratio : ℝ)
  (final_green_space_ratio : ℝ)
  (additional_households : ℕ)
  (min_green_space_ratio : ℝ)
  (h1 : household_area = 150)
  (h2 : initial_green_space_ratio = 0.4)
  (h3 : final_green_space_ratio = 0.15)
  (h4 : additional_households = 20)
  (h5 : min_green_space_ratio = 0.2) :
  ∃ (initial_households : ℕ) (total_area : ℝ) (withdraw_households : ℕ),
    initial_households = 48 ∧ 
    total_area = 12000 ∧ 
    withdraw_households ≥ 4 ∧
    total_area - household_area * initial_households = initial_green_space_ratio * total_area ∧
    total_area - household_area * (initial_households + additional_households) = final_green_space_ratio * total_area ∧
    total_area - household_area * (initial_households + additional_households - withdraw_households) ≥ min_green_space_ratio * total_area :=
by sorry

end NUMINAMATH_CALUDE_housing_relocation_problem_l1555_155505


namespace NUMINAMATH_CALUDE_largest_non_prime_sequence_l1555_155583

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem largest_non_prime_sequence :
  ∃ (start : ℕ),
    (∀ i ∈ Finset.range 7, 
      let n := start + i
      10 ≤ n ∧ n < 40 ∧ ¬(is_prime n)) ∧
    (∀ j ≥ start + 7, 
      ¬(∀ i ∈ Finset.range 7, 
        let n := j + i
        10 ≤ n ∧ n < 40 ∧ ¬(is_prime n))) →
  start + 6 = 32 :=
sorry

end NUMINAMATH_CALUDE_largest_non_prime_sequence_l1555_155583


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1555_155567

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 3| = |x + 5| :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l1555_155567


namespace NUMINAMATH_CALUDE_bicycle_helmet_cost_ratio_l1555_155531

theorem bicycle_helmet_cost_ratio :
  ∀ (bicycle_cost helmet_cost : ℕ),
    helmet_cost = 40 →
    bicycle_cost + helmet_cost = 240 →
    ∃ (m : ℕ), bicycle_cost = m * helmet_cost →
    m = 5 := by
  sorry

end NUMINAMATH_CALUDE_bicycle_helmet_cost_ratio_l1555_155531


namespace NUMINAMATH_CALUDE_finley_class_size_l1555_155541

/-- The number of students in Mrs. Finley's class -/
def finley_class : ℕ := sorry

/-- The number of students in Mr. Johnson's class -/
def johnson_class : ℕ := 22

/-- Mr. Johnson's class has 10 more than half the number in Mrs. Finley's class -/
axiom johnson_class_size : johnson_class = finley_class / 2 + 10

theorem finley_class_size : finley_class = 24 := by sorry

end NUMINAMATH_CALUDE_finley_class_size_l1555_155541


namespace NUMINAMATH_CALUDE_student_marks_calculation_l1555_155529

theorem student_marks_calculation (total_marks : ℕ) (passing_percentage : ℚ) (failing_margin : ℕ) (student_marks : ℕ) : 
  total_marks = 500 →
  passing_percentage = 33 / 100 →
  failing_margin = 40 →
  student_marks = total_marks * passing_percentage - failing_margin →
  student_marks = 125 := by
sorry

end NUMINAMATH_CALUDE_student_marks_calculation_l1555_155529


namespace NUMINAMATH_CALUDE_greatest_x_lcm_l1555_155568

def is_lcm (a b c m : ℕ) : Prop :=
  m % a = 0 ∧ m % b = 0 ∧ m % c = 0 ∧
  ∀ n : ℕ, (n % a = 0 ∧ n % b = 0 ∧ n % c = 0) → m ≤ n

theorem greatest_x_lcm :
  ∀ x : ℕ, is_lcm x 15 21 105 → x ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_lcm_l1555_155568


namespace NUMINAMATH_CALUDE_town_distance_l1555_155582

/-- Three towns A, B, and C are equidistant from each other and are 3, 5, and 8 miles 
    respectively from a common railway station D. -/
structure TownConfiguration where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  equidistant : dist A B = dist B C ∧ dist B C = dist C A
  dist_AD : dist A D = 3
  dist_BD : dist B D = 5
  dist_CD : dist C D = 8

/-- The distance between any two towns is 7 miles. -/
theorem town_distance (config : TownConfiguration) : 
  dist config.A config.B = 7 ∧ dist config.B config.C = 7 ∧ dist config.C config.A = 7 := by
  sorry


end NUMINAMATH_CALUDE_town_distance_l1555_155582


namespace NUMINAMATH_CALUDE_system_solution_inequality_solution_set_inequality_solution_transformation_l1555_155570

-- Problem 1
theorem system_solution :
  let system (x y : ℝ) := y = x + 1 ∧ x^2 + 4*y^2 = 4
  ∃ (x₁ y₁ x₂ y₂ : ℝ), system x₁ y₁ ∧ system x₂ y₂ ∧
    ((x₁ = 0 ∧ y₁ = 1) ∨ (x₁ = -8/5 ∧ y₁ = -3/5)) ∧
    ((x₂ = 0 ∧ y₂ = 1) ∨ (x₂ = -8/5 ∧ y₂ = -3/5)) ∧
    x₁ ≠ x₂ := by sorry

-- Problem 2
theorem inequality_solution_set (t : ℝ) :
  let solution_set := {x : ℝ | x^2 - 2*t*x + 1 > 0}
  (t < -1 ∨ t > 1 → ∃ (a b : ℝ), solution_set = {x | x < a ∨ x > b}) ∧
  (-1 < t ∧ t < 1 → solution_set = Set.univ) ∧
  (t = 1 → solution_set = {x | x ≠ 1}) ∧
  (t = -1 → solution_set = {x | x ≠ -1}) := by sorry

-- Problem 3
theorem inequality_solution_transformation (a b c : ℝ) :
  ({x : ℝ | a*x^2 + b*x + c > 0} = Set.Ioo 1 2) →
  {x : ℝ | c*x^2 - b*x + a < 0} = {x : ℝ | x < -1 ∨ x > -1/2} := by sorry

end NUMINAMATH_CALUDE_system_solution_inequality_solution_set_inequality_solution_transformation_l1555_155570


namespace NUMINAMATH_CALUDE_arithmetic_expression_equals_28_l1555_155518

theorem arithmetic_expression_equals_28 : 12 / 4 - 3 - 10 + 3 * 10 + 2^3 = 28 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equals_28_l1555_155518


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_l1555_155578

def arithmetic_sequence (a₁ d n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem arithmetic_sequence_2011 :
  arithmetic_sequence 1 3 671 = 2011 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_l1555_155578


namespace NUMINAMATH_CALUDE_min_sum_of_distances_l1555_155504

-- Define the curve
def curve (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the distance from a point to line y = 2
def dist_to_line1 (x y : ℝ) : ℝ := |y - 2|

-- Define the distance from a point to line x = -1
def dist_to_line2 (x y : ℝ) : ℝ := |x + 1|

-- Define the sum of distances
def sum_of_distances (x y : ℝ) : ℝ := dist_to_line1 x y + dist_to_line2 x y

-- Theorem statement
theorem min_sum_of_distances :
  ∃ (min : ℝ), min = 4 - Real.sqrt 2 ∧
  (∀ (x y : ℝ), curve x y → sum_of_distances x y ≥ min) ∧
  (∃ (x y : ℝ), curve x y ∧ sum_of_distances x y = min) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_distances_l1555_155504


namespace NUMINAMATH_CALUDE_pentagon_perimeter_division_l1555_155599

/-- Given a regular pentagon with perimeter 125 and side length 25,
    prove that the perimeter divided by the side length equals 5. -/
theorem pentagon_perimeter_division (perimeter : ℝ) (side_length : ℝ) :
  perimeter = 125 →
  side_length = 25 →
  perimeter / side_length = 5 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_perimeter_division_l1555_155599


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1555_155598

theorem sufficient_but_not_necessary
  (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1)
  (f : ℝ → ℝ) (hf : f = λ x => a^x)
  (g : ℝ → ℝ) (hg : g = λ x => (2-a)*x^3) :
  (∀ x y : ℝ, x < y → f x > f y) →
  (∀ x y : ℝ, x < y → g x < g y) ∧
  ¬(∀ x y : ℝ, x < y → g x < g y → ∀ x y : ℝ, x < y → f x > f y) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1555_155598


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l1555_155596

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l1555_155596


namespace NUMINAMATH_CALUDE_notebook_cost_l1555_155562

/-- Represents the problem of determining the cost of notebooks --/
theorem notebook_cost (total_students : ℕ) 
  (buyers : ℕ) 
  (notebooks_per_buyer : ℕ) 
  (cost_per_notebook : ℕ) 
  (total_cost : ℕ) :
  total_students = 36 →
  buyers > total_students / 2 →
  buyers ≤ total_students →
  cost_per_notebook > notebooks_per_buyer →
  buyers * notebooks_per_buyer * cost_per_notebook = total_cost →
  total_cost = 2275 →
  cost_per_notebook = 13 :=
by sorry

end NUMINAMATH_CALUDE_notebook_cost_l1555_155562


namespace NUMINAMATH_CALUDE_weight_after_one_year_l1555_155507

def initial_weight : ℕ := 250

def training_loss : List ℕ := [8, 5, 7, 6, 8, 7, 5, 7, 4, 6, 5, 7]

def diet_loss_per_month : ℕ := 3

def months_in_year : ℕ := 12

theorem weight_after_one_year :
  initial_weight - (training_loss.sum + diet_loss_per_month * months_in_year) = 139 := by
  sorry

end NUMINAMATH_CALUDE_weight_after_one_year_l1555_155507


namespace NUMINAMATH_CALUDE_sum_of_ages_in_five_years_l1555_155585

/-- Given that Linda's current age is 13 and she is 3 more than 2 times Jane's age,
    prove that the sum of their ages in five years will be 28. -/
theorem sum_of_ages_in_five_years (jane_age : ℕ) (linda_age : ℕ) : 
  linda_age = 13 → linda_age = 2 * jane_age + 3 → 
  linda_age + 5 + (jane_age + 5) = 28 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_five_years_l1555_155585


namespace NUMINAMATH_CALUDE_mrs_hilt_has_more_money_l1555_155514

-- Define the value of each coin type
def penny_value : ℚ := 0.01
def nickel_value : ℚ := 0.05
def dime_value : ℚ := 0.10

-- Define the number of coins each person has
def mrs_hilt_pennies : ℕ := 2
def mrs_hilt_nickels : ℕ := 2
def mrs_hilt_dimes : ℕ := 2

def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1

-- Calculate the total amount for each person
def mrs_hilt_total : ℚ :=
  mrs_hilt_pennies * penny_value +
  mrs_hilt_nickels * nickel_value +
  mrs_hilt_dimes * dime_value

def jacob_total : ℚ :=
  jacob_pennies * penny_value +
  jacob_nickels * nickel_value +
  jacob_dimes * dime_value

-- State the theorem
theorem mrs_hilt_has_more_money :
  mrs_hilt_total - jacob_total = 0.13 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_has_more_money_l1555_155514


namespace NUMINAMATH_CALUDE_absolute_value_comparison_l1555_155590

theorem absolute_value_comparison (a b : ℚ) : 
  |a| = 2/3 ∧ |b| = 3/5 → 
  ((a = 2/3 ∨ a = -2/3) ∧ 
   (b = 3/5 ∨ b = -3/5) ∧ 
   (a = 2/3 → a > b) ∧ 
   (a = -2/3 → a < b)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_comparison_l1555_155590


namespace NUMINAMATH_CALUDE_line_curve_intersection_implies_m_geq_3_l1555_155513

-- Define the line equation
def line (k x : ℝ) : ℝ := k * x - k + 1

-- Define the curve equation
def curve (x y m : ℝ) : Prop := x^2 + 2 * y^2 = m

-- Theorem statement
theorem line_curve_intersection_implies_m_geq_3 (k m : ℝ) :
  (∃ x y : ℝ, line k x = y ∧ curve x y m) → m ≥ 3 := by sorry

end NUMINAMATH_CALUDE_line_curve_intersection_implies_m_geq_3_l1555_155513


namespace NUMINAMATH_CALUDE_sum_over_subsets_equals_power_of_two_l1555_155564

def S : Finset ℕ := Finset.range 1999

def f (T : Finset ℕ) : ℕ := T.sum id

theorem sum_over_subsets_equals_power_of_two :
  (Finset.powerset S).sum (fun E => (f E : ℚ) / (f S : ℚ)) = 2^1998 := by sorry

end NUMINAMATH_CALUDE_sum_over_subsets_equals_power_of_two_l1555_155564


namespace NUMINAMATH_CALUDE_inequality_proof_l1555_155554

theorem inequality_proof (a b c d : ℝ) (h1 : c < d) (h2 : a > b) (h3 : b > 0) :
  a - c > b - d := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1555_155554


namespace NUMINAMATH_CALUDE_smallest_even_five_digit_number_has_eight_in_tens_place_l1555_155587

-- Define a type for digits
inductive Digit : Type
  | one : Digit
  | three : Digit
  | five : Digit
  | six : Digit
  | eight : Digit

-- Define a function to convert Digit to Nat
def digitToNat : Digit → Nat
  | Digit.one => 1
  | Digit.three => 3
  | Digit.five => 5
  | Digit.six => 6
  | Digit.eight => 8

-- Define a function to check if a number is even
def isEven (n : Nat) : Bool :=
  n % 2 == 0

-- Define a function to construct a five-digit number from Digits
def makeNumber (a b c d e : Digit) : Nat :=
  10000 * (digitToNat a) + 1000 * (digitToNat b) + 100 * (digitToNat c) + 10 * (digitToNat d) + (digitToNat e)

-- Define the theorem
theorem smallest_even_five_digit_number_has_eight_in_tens_place :
  ∀ (a b c d e : Digit),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e →
    isEven (makeNumber a b c d e) →
    (∀ (x y z w v : Digit),
      x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ x ≠ v ∧
      y ≠ z ∧ y ≠ w ∧ y ≠ v ∧
      z ≠ w ∧ z ≠ v ∧
      w ≠ v →
      isEven (makeNumber x y z w v) →
      makeNumber a b c d e ≤ makeNumber x y z w v) →
    d = Digit.eight :=
  sorry

end NUMINAMATH_CALUDE_smallest_even_five_digit_number_has_eight_in_tens_place_l1555_155587


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1555_155532

-- Define a geometric sequence
def isGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Theorem statement
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  isGeometricSequence a →
  (∀ n : ℕ, a n > 0) →
  a 4 * a 12 + a 3 * a 5 = 15 →
  a 4 * a 8 = 5 →
  a 4 + a 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1555_155532


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l1555_155522

theorem quadratic_function_max_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 0 1, -x^2 + 2*a*x + 1 - a ≤ 2) ∧ 
  (∃ x ∈ Set.Icc 0 1, -x^2 + 2*a*x + 1 - a = 2) → 
  a = -1 ∨ a = 2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l1555_155522


namespace NUMINAMATH_CALUDE_jason_games_last_month_l1555_155519

/-- Calculates the number of games Jason planned to attend last month -/
def games_planned_last_month (games_this_month games_missed games_attended : ℕ) : ℕ :=
  (games_attended + games_missed) - games_this_month

theorem jason_games_last_month :
  games_planned_last_month 11 16 12 = 17 := by
  sorry

end NUMINAMATH_CALUDE_jason_games_last_month_l1555_155519


namespace NUMINAMATH_CALUDE_will_money_left_l1555_155569

/-- The amount of money Will has left after shopping --/
def money_left (initial_amount : ℝ) (sweater_price : ℝ) (tshirt_price : ℝ) (shoes_price : ℝ) 
  (hat_price : ℝ) (socks_price : ℝ) (shoe_refund_rate : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_cost := sweater_price + tshirt_price + shoes_price + hat_price + socks_price
  let refund := shoes_price * shoe_refund_rate
  let new_total := total_cost - refund
  let remaining_items_cost := sweater_price + tshirt_price + hat_price + socks_price
  let discount := remaining_items_cost * discount_rate
  let discounted_total := new_total - discount
  let sales_tax := discounted_total * tax_rate
  let final_cost := discounted_total + sales_tax
  initial_amount - final_cost

/-- Theorem stating that Will has $41.87 left after shopping --/
theorem will_money_left : 
  money_left 74 9 11 30 5 4 0.85 0.1 0.05 = 41.87 := by
  sorry

end NUMINAMATH_CALUDE_will_money_left_l1555_155569


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l1555_155573

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (x + a)(x - 4) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * (x - 4)

/-- If f(x) = (x + a)(x - 4) is an even function, then a = 4 -/
theorem even_function_implies_a_equals_four :
  ∀ a : ℝ, IsEven (f a) → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_four_l1555_155573


namespace NUMINAMATH_CALUDE_fraction_equality_l1555_155542

theorem fraction_equality (a b : ℝ) (h : a / b = 6 / 5) :
  (5 * a + 4 * b) / (5 * a - 4 * b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1555_155542


namespace NUMINAMATH_CALUDE_shots_per_puppy_l1555_155560

/-- Calculates the number of shots each puppy needs given the specified conditions -/
theorem shots_per_puppy
  (num_dogs : ℕ)
  (puppies_per_dog : ℕ)
  (cost_per_shot : ℕ)
  (total_cost : ℕ)
  (h1 : num_dogs = 3)
  (h2 : puppies_per_dog = 4)
  (h3 : cost_per_shot = 5)
  (h4 : total_cost = 120) :
  (total_cost / cost_per_shot) / (num_dogs * puppies_per_dog) = 2 := by
  sorry

#check shots_per_puppy

end NUMINAMATH_CALUDE_shots_per_puppy_l1555_155560


namespace NUMINAMATH_CALUDE_senior_mean_score_l1555_155595

theorem senior_mean_score (total_students : ℕ) (overall_mean : ℝ) 
  (senior_total_score : ℝ) :
  total_students = 200 →
  overall_mean = 80 →
  senior_total_score = 7200 →
  ∃ (num_seniors num_non_seniors : ℕ) (senior_mean non_senior_mean : ℝ),
    num_non_seniors = (5 / 4 : ℝ) * num_seniors ∧
    senior_mean = (6 / 5 : ℝ) * non_senior_mean ∧
    num_seniors + num_non_seniors = total_students ∧
    (num_seniors * senior_mean + num_non_seniors * non_senior_mean) / total_students = overall_mean ∧
    num_seniors * senior_mean = senior_total_score ∧
    senior_mean = 80.9 := by
  sorry

end NUMINAMATH_CALUDE_senior_mean_score_l1555_155595


namespace NUMINAMATH_CALUDE_claire_crafting_time_l1555_155593

/-- Represents Claire's daily schedule --/
structure ClairesSchedule where
  clean : ℝ
  cook : ℝ
  errands : ℝ
  craft : ℝ
  tailor : ℝ

/-- Conditions for Claire's schedule --/
def validSchedule (s : ClairesSchedule) : Prop :=
  s.clean = 2 * s.cook ∧
  s.errands = s.cook - 1 ∧
  s.craft = s.tailor ∧
  s.clean + s.cook + s.errands + s.craft + s.tailor = 16 ∧
  s.craft + s.tailor = 9

/-- Theorem stating that in a valid schedule, Claire spends 4.5 hours crafting --/
theorem claire_crafting_time (s : ClairesSchedule) (h : validSchedule s) : s.craft = 4.5 := by
  sorry


end NUMINAMATH_CALUDE_claire_crafting_time_l1555_155593


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l1555_155510

theorem greatest_prime_factor_of_factorial_sum : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ (Nat.factorial 15 + Nat.factorial 17) ∧ 
  ∀ q : ℕ, Nat.Prime q → q ∣ (Nat.factorial 15 + Nat.factorial 17) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l1555_155510


namespace NUMINAMATH_CALUDE_labor_cost_per_hour_l1555_155502

theorem labor_cost_per_hour 
  (total_repair_cost : ℝ)
  (part_cost : ℝ)
  (labor_hours : ℝ)
  (h1 : total_repair_cost = 2400)
  (h2 : part_cost = 1200)
  (h3 : labor_hours = 16) :
  (total_repair_cost - part_cost) / labor_hours = 75 := by
sorry

end NUMINAMATH_CALUDE_labor_cost_per_hour_l1555_155502


namespace NUMINAMATH_CALUDE_equation_solution_l1555_155588

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => (20 / (x^2 - 9)) - (3 / (x + 3)) - 2
  ∃ x₁ x₂ : ℝ, x₁ = (-3 + Real.sqrt 385) / 4 ∧ 
              x₂ = (-3 - Real.sqrt 385) / 4 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1555_155588


namespace NUMINAMATH_CALUDE_cylinder_unique_non_trapezoid_cross_section_l1555_155524

-- Define the solids
inductive Solid
| Frustum
| Cylinder
| Cube
| TriangularPrism

-- Define a predicate for whether a solid can have an isosceles trapezoid cross-section
def can_have_isosceles_trapezoid_cross_section : Solid → Prop
| Solid.Frustum => True
| Solid.Cylinder => False
| Solid.Cube => True
| Solid.TriangularPrism => True

-- Theorem statement
theorem cylinder_unique_non_trapezoid_cross_section :
  ∀ s : Solid, ¬(can_have_isosceles_trapezoid_cross_section s) ↔ s = Solid.Cylinder :=
by sorry

end NUMINAMATH_CALUDE_cylinder_unique_non_trapezoid_cross_section_l1555_155524


namespace NUMINAMATH_CALUDE_arithmetic_less_than_geometric_l1555_155559

/-- A positive arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), d > 0 ∧ ∀ n, a n = a₁ + (n - 1) * d

/-- A positive geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ (b₁ r : ℝ), r > 1 ∧ ∀ n, b n = b₁ * r^(n - 1)

theorem arithmetic_less_than_geometric
  (a b : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hb : geometric_sequence b)
  (h_eq1 : a 1 = b 1)
  (h_eq2 : a 2 = b 2) :
  ∀ n ≥ 3, a n < b n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_less_than_geometric_l1555_155559


namespace NUMINAMATH_CALUDE_fishing_problem_l1555_155512

theorem fishing_problem (a b c d : ℕ) : 
  a + b + c + d = 25 →
  a > b ∧ b > c ∧ c > d →
  a = b + c →
  b = c + d →
  (a = 11 ∧ b = 7 ∧ c = 4 ∧ d = 3) := by
sorry

end NUMINAMATH_CALUDE_fishing_problem_l1555_155512


namespace NUMINAMATH_CALUDE_solution_to_equation_l1555_155575

theorem solution_to_equation : ∃! x : ℝ, (x - 3)^3 = (1/27)⁻¹ := by
  use 6
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l1555_155575


namespace NUMINAMATH_CALUDE_colored_paper_problem_l1555_155515

/-- The number of pieces of colored paper Yuna had initially -/
def yunas_initial_paper : ℕ := 150

/-- The number of pieces of colored paper Namjoon had initially -/
def namjoons_initial_paper : ℕ := 250

/-- The number of pieces of colored paper Namjoon gave to Yuna -/
def paper_given : ℕ := 60

/-- The difference in paper count between Yuna and Namjoon after the exchange -/
def paper_difference : ℕ := 20

theorem colored_paper_problem :
  yunas_initial_paper = 150 ∧
  namjoons_initial_paper = 250 ∧
  paper_given = 60 ∧
  paper_difference = 20 →
  yunas_initial_paper + paper_given = namjoons_initial_paper - paper_given + paper_difference :=
by sorry

end NUMINAMATH_CALUDE_colored_paper_problem_l1555_155515


namespace NUMINAMATH_CALUDE_harry_apples_l1555_155552

/-- Proves that Harry has 19 apples given the conditions of the problem -/
theorem harry_apples :
  ∀ (martha_apples tim_apples harry_apples jane_apples : ℕ),
  martha_apples = 68 →
  tim_apples = martha_apples - 30 →
  harry_apples = tim_apples / 2 →
  jane_apples = ((martha_apples + tim_apples) * 25) / 100 →
  harry_apples = 19 := by
  sorry

#check harry_apples

end NUMINAMATH_CALUDE_harry_apples_l1555_155552


namespace NUMINAMATH_CALUDE_inequality_proof_l1555_155544

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
  (x + y) / (2 + x + y) < x / (2 + x) + y / (2 + y) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1555_155544


namespace NUMINAMATH_CALUDE_crow_votes_l1555_155534

def singing_contest (total_judges reported_total : ℕ)
                    (rooster_crow reported_rooster_crow : ℕ)
                    (crow_cuckoo reported_crow_cuckoo : ℕ)
                    (cuckoo_rooster reported_cuckoo_rooster : ℕ)
                    (max_error : ℕ) : Prop :=
  ∃ (rooster crow cuckoo : ℕ),
    -- Actual total of judges
    rooster + crow + cuckoo = total_judges ∧
    -- Reported total within error range
    (reported_total : ℤ) - (total_judges : ℤ) ≤ max_error ∧
    (total_judges : ℤ) - (reported_total : ℤ) ≤ max_error ∧
    -- Reported sums within error range
    (reported_rooster_crow : ℤ) - ((rooster + crow) : ℤ) ≤ max_error ∧
    ((rooster + crow) : ℤ) - (reported_rooster_crow : ℤ) ≤ max_error ∧
    (reported_crow_cuckoo : ℤ) - ((crow + cuckoo) : ℤ) ≤ max_error ∧
    ((crow + cuckoo) : ℤ) - (reported_crow_cuckoo : ℤ) ≤ max_error ∧
    (reported_cuckoo_rooster : ℤ) - ((cuckoo + rooster) : ℤ) ≤ max_error ∧
    ((cuckoo + rooster) : ℤ) - (reported_cuckoo_rooster : ℤ) ≤ max_error ∧
    -- The number of votes for Crow is 13
    crow = 13

theorem crow_votes :
  singing_contest 46 59 15 15 18 18 20 20 13 :=
by sorry

end NUMINAMATH_CALUDE_crow_votes_l1555_155534


namespace NUMINAMATH_CALUDE_product_inequality_l1555_155581

theorem product_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1/a) * (b + 1/b) ≥ 25/4 := by sorry

end NUMINAMATH_CALUDE_product_inequality_l1555_155581


namespace NUMINAMATH_CALUDE_five_integer_chords_l1555_155550

/-- Represents a circle with a point P inside it -/
structure CircleWithPoint where
  radius : ℝ
  distanceFromCenter : ℝ

/-- Counts the number of integer-length chords through P -/
def countIntegerChords (c : CircleWithPoint) : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem five_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 10)
  (h2 : c.distanceFromCenter = 6) : 
  countIntegerChords c = 5 :=
sorry

end NUMINAMATH_CALUDE_five_integer_chords_l1555_155550


namespace NUMINAMATH_CALUDE_fgh_supermarket_difference_l1555_155580

/-- The number of FGH supermarkets in the US and Canada -/
structure FGHSupermarkets where
  total : ℕ
  us : ℕ
  canada : ℕ

/-- The conditions for FGH supermarkets -/
def validFGHSupermarkets (s : FGHSupermarkets) : Prop :=
  s.total = 60 ∧
  s.us + s.canada = s.total ∧
  s.us = 37 ∧
  s.us > s.canada

/-- Theorem: The difference between FGH supermarkets in the US and Canada is 14 -/
theorem fgh_supermarket_difference (s : FGHSupermarkets) 
  (h : validFGHSupermarkets s) : s.us - s.canada = 14 := by
  sorry

end NUMINAMATH_CALUDE_fgh_supermarket_difference_l1555_155580


namespace NUMINAMATH_CALUDE_max_ab_for_line_circle_intersection_l1555_155565

/-- Given a line ax + by - 6 = 0 (a > 0, b > 0) intercepted by the circle x^2 + y^2 - 2x - 4y = 0
    to form a chord of length 2√5, the maximum value of ab is 9/2 -/
theorem max_ab_for_line_circle_intersection (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, a * x + b * y = 6 ∧ x^2 + y^2 - 2*x - 4*y = 0) →
  (∃ x1 y1 x2 y2 : ℝ, 
    a * x1 + b * y1 = 6 ∧ x1^2 + y1^2 - 2*x1 - 4*y1 = 0 ∧
    a * x2 + b * y2 = 6 ∧ x2^2 + y2^2 - 2*x2 - 4*y2 = 0 ∧
    (x1 - x2)^2 + (y1 - y2)^2 = 20) →
  a * b ≤ 9/2 :=
by sorry

end NUMINAMATH_CALUDE_max_ab_for_line_circle_intersection_l1555_155565


namespace NUMINAMATH_CALUDE_shepherd_a_has_seven_sheep_l1555_155503

/-- Represents the number of sheep each shepherd has -/
structure ShepherdSheep where
  a : ℕ
  b : ℕ

/-- The conditions of the problem are satisfied -/
def satisfiesConditions (s : ShepherdSheep) : Prop :=
  (s.a + 1 = 2 * (s.b - 1)) ∧ (s.a - 1 = s.b + 1)

/-- Theorem stating that shepherd A has 7 sheep -/
theorem shepherd_a_has_seven_sheep :
  ∃ s : ShepherdSheep, satisfiesConditions s ∧ s.a = 7 :=
sorry

end NUMINAMATH_CALUDE_shepherd_a_has_seven_sheep_l1555_155503


namespace NUMINAMATH_CALUDE_miles_difference_l1555_155511

/-- The number of miles Gervais drove per day -/
def gervais_daily_miles : ℕ := 315

/-- The number of days Gervais drove -/
def gervais_days : ℕ := 3

/-- The total number of miles Henri drove -/
def henri_total_miles : ℕ := 1250

/-- Theorem stating the difference in miles driven between Henri and Gervais -/
theorem miles_difference : henri_total_miles - (gervais_daily_miles * gervais_days) = 305 := by
  sorry

end NUMINAMATH_CALUDE_miles_difference_l1555_155511


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1555_155526

open Set

def U : Set Nat := {1,2,3,4,5,6}
def P : Set Nat := {1,3,5}
def Q : Set Nat := {1,2,4}

theorem complement_union_theorem :
  (U \ P) ∪ Q = {1,2,4,6} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1555_155526


namespace NUMINAMATH_CALUDE_C_7_3_2_eq_10_l1555_155509

/-- A function that calculates the number of ways to select k elements from a set of n elements
    with a minimum distance of m between selected elements. -/
def C (n k m : ℕ) : ℕ := sorry

/-- The theorem stating that C_7^(3,2) = 10 -/
theorem C_7_3_2_eq_10 : C 7 3 2 = 10 := by sorry

end NUMINAMATH_CALUDE_C_7_3_2_eq_10_l1555_155509


namespace NUMINAMATH_CALUDE_gcd_of_638_522_406_l1555_155506

theorem gcd_of_638_522_406 : Nat.gcd 638 (Nat.gcd 522 406) = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_638_522_406_l1555_155506


namespace NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l1555_155523

/-- Represents a batsman's statistics -/
structure Batsman where
  innings : ℕ
  totalScore : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (score : ℕ) : ℚ :=
  (b.totalScore + score) / (b.innings + 1)

/-- Theorem stating the batsman's new average after the 17th innings -/
theorem batsman_average_after_17th_innings
  (b : Batsman)
  (h1 : b.innings = 16)
  (h2 : newAverage b 85 = b.average + 3) :
  newAverage b 85 = 37 := by
  sorry

#check batsman_average_after_17th_innings

end NUMINAMATH_CALUDE_batsman_average_after_17th_innings_l1555_155523


namespace NUMINAMATH_CALUDE_yanna_change_l1555_155535

/-- The change Yanna received after buying shirts and sandals -/
def change_received (shirt_price shirt_quantity sandal_price sandal_quantity payment : ℕ) : ℕ :=
  payment - (shirt_price * shirt_quantity + sandal_price * sandal_quantity)

/-- Theorem stating that Yanna received $41 in change -/
theorem yanna_change :
  change_received 5 10 3 3 100 = 41 := by
  sorry

end NUMINAMATH_CALUDE_yanna_change_l1555_155535


namespace NUMINAMATH_CALUDE_faces_after_fifth_step_l1555_155591

/-- Represents the number of vertices at step n -/
def V : ℕ → ℕ
| 0 => 8
| n + 1 => 3 * V n

/-- Represents the number of faces at step n -/
def F : ℕ → ℕ
| 0 => 6
| n + 1 => F n + V n

/-- Theorem stating that the number of faces after the fifth step is 974 -/
theorem faces_after_fifth_step : F 5 = 974 := by
  sorry

end NUMINAMATH_CALUDE_faces_after_fifth_step_l1555_155591


namespace NUMINAMATH_CALUDE_parallelogram_area_is_three_l1555_155508

/-- The area of a parallelogram formed by two 2D vectors -/
def parallelogramArea (a b : Fin 2 → ℝ) : ℝ :=
  |a 0 * b 1 - a 1 * b 0|

/-- Given vectors v, w, and u, prove that the area of the parallelogram
    formed by (v + u) and w is 3 -/
theorem parallelogram_area_is_three :
  let v : Fin 2 → ℝ := ![7, -4]
  let w : Fin 2 → ℝ := ![3, 1]
  let u : Fin 2 → ℝ := ![-1, 5]
  parallelogramArea (v + u) w = 3 := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_area_is_three_l1555_155508


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1555_155540

def A : Set Int := {-1, 0, 1}
def B : Set Int := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1555_155540


namespace NUMINAMATH_CALUDE_missed_both_equiv_l1555_155566

-- Define propositions
variable (p q : Prop)

-- Define the meaning of "missed the target on both shots"
def missed_both (p q : Prop) : Prop := (¬p) ∧ (¬q)

-- Theorem: "missed the target on both shots" is equivalent to ¬(p ∨ q)
theorem missed_both_equiv (p q : Prop) : missed_both p q ↔ ¬(p ∨ q) := by
  sorry

end NUMINAMATH_CALUDE_missed_both_equiv_l1555_155566


namespace NUMINAMATH_CALUDE_boat_speed_correct_l1555_155527

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 15

/-- The speed of the stream -/
def stream_speed : ℝ := 3

/-- The time taken to travel downstream -/
def downstream_time : ℝ := 1

/-- The time taken to travel upstream -/
def upstream_time : ℝ := 1.5

/-- Theorem stating that the boat speed is correct given the conditions -/
theorem boat_speed_correct :
  ∃ (distance : ℝ),
    distance = (boat_speed + stream_speed) * downstream_time ∧
    distance = (boat_speed - stream_speed) * upstream_time :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_correct_l1555_155527


namespace NUMINAMATH_CALUDE_second_divisor_existence_l1555_155556

theorem second_divisor_existence : ∃ (D : ℕ+), 
  (∃ (N : ℤ), N % 35 = 25 ∧ N % D.val = 4) ∧ D.val = 21 := by
  sorry

end NUMINAMATH_CALUDE_second_divisor_existence_l1555_155556


namespace NUMINAMATH_CALUDE_no_real_solution_cubic_equation_l1555_155528

theorem no_real_solution_cubic_equation :
  ∀ x : ℂ, (x^3 + 3*x^2 + 4*x + 6) / (x + 5) = x^2 + 10 →
  (x = (-3 + Complex.I * Real.sqrt 79) / 2 ∨ x = (-3 - Complex.I * Real.sqrt 79) / 2) :=
by sorry

end NUMINAMATH_CALUDE_no_real_solution_cubic_equation_l1555_155528


namespace NUMINAMATH_CALUDE_cricket_overs_l1555_155516

theorem cricket_overs (initial_rate : ℝ) (remaining_rate : ℝ) (remaining_overs : ℝ) (target : ℝ) :
  initial_rate = 4.2 →
  remaining_rate = 8 →
  remaining_overs = 30 →
  target = 324 →
  ∃ (initial_overs : ℝ), 
    initial_overs * initial_rate + remaining_overs * remaining_rate = target ∧ 
    initial_overs = 20 := by
  sorry

end NUMINAMATH_CALUDE_cricket_overs_l1555_155516


namespace NUMINAMATH_CALUDE_election_votes_theorem_l1555_155546

theorem election_votes_theorem (total_votes : ℕ) 
  (h1 : total_votes > 0)
  (h2 : ∃ (winner_votes loser_votes : ℕ), 
    winner_votes + loser_votes = total_votes ∧ 
    winner_votes = (70 * total_votes) / 100 ∧
    winner_votes - loser_votes = 192) :
  total_votes = 480 := by
sorry

end NUMINAMATH_CALUDE_election_votes_theorem_l1555_155546


namespace NUMINAMATH_CALUDE_student_count_l1555_155597

theorem student_count : 
  ∃ n₁ n₂ : ℕ, n₁ ≠ n₂ ∧ 
  (∀ n : ℕ, (70 < n ∧ n < 130 ∧ 
             n % 4 = 2 ∧ 
             n % 5 = 2 ∧ 
             n % 6 = 2) ↔ (n = n₁ ∨ n = n₂)) ∧
  n₁ = 92 ∧ n₂ = 122 :=
by sorry

end NUMINAMATH_CALUDE_student_count_l1555_155597


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l1555_155520

theorem existence_of_special_numbers :
  ∃ (a b c : ℕ), 
    a > 10^10 ∧ b > 10^10 ∧ c > 10^10 ∧
    (∃ (k₁ k₂ k₃ : ℕ), 
      a * b * c = k₁ * (a + 2012) ∧
      a * b * c = k₂ * (b + 2012) ∧
      a * b * c = k₃ * (c + 2012)) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l1555_155520


namespace NUMINAMATH_CALUDE_no_infinite_sequence_with_sqrt_difference_l1555_155549

theorem no_infinite_sequence_with_sqrt_difference :
  ¬∃ (x : ℕ → ℝ), 
    (∀ n, x n > 0) ∧ 
    (∀ n, x (n + 2) = Real.sqrt (x (n + 1)) - Real.sqrt (x n)) := by
  sorry

end NUMINAMATH_CALUDE_no_infinite_sequence_with_sqrt_difference_l1555_155549


namespace NUMINAMATH_CALUDE_cube_root_of_64_l1555_155572

theorem cube_root_of_64 : (64 : ℝ) ^ (1/3 : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_cube_root_of_64_l1555_155572


namespace NUMINAMATH_CALUDE_max_area_inscribed_quadrilateral_l1555_155594

/-- The maximum area of an inscribed quadrilateral within a circle -/
def max_area_circle (r : ℝ) : ℝ := 2 * r^2

/-- The equation of an ellipse -/
def is_ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

/-- The maximum area of an inscribed quadrilateral within an ellipse -/
def max_area_ellipse (a b : ℝ) : ℝ := 2 * a * b

theorem max_area_inscribed_quadrilateral 
  (r a b : ℝ) 
  (hr : r > 0) 
  (hab : a > b) 
  (hb : b > 0) : 
  max_area_ellipse a b = 2 * a * b :=
sorry

end NUMINAMATH_CALUDE_max_area_inscribed_quadrilateral_l1555_155594


namespace NUMINAMATH_CALUDE_lunch_theorem_l1555_155561

def lunch_problem (total_spent friend_spent : ℕ) : Prop :=
  friend_spent > total_spent - friend_spent ∧
  friend_spent - (total_spent - friend_spent) = 1

theorem lunch_theorem :
  lunch_problem 15 8 := by
  sorry

end NUMINAMATH_CALUDE_lunch_theorem_l1555_155561


namespace NUMINAMATH_CALUDE_gcd_18_30_l1555_155525

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l1555_155525


namespace NUMINAMATH_CALUDE_janelle_initial_green_marbles_l1555_155577

/-- The number of bags of blue marbles Janelle bought -/
def blue_bags : ℕ := 6

/-- The number of marbles in each bag -/
def marbles_per_bag : ℕ := 10

/-- The number of green marbles in the gift -/
def green_marbles_gift : ℕ := 6

/-- The number of blue marbles in the gift -/
def blue_marbles_gift : ℕ := 8

/-- The number of marbles Janelle has left after giving the gift -/
def marbles_left : ℕ := 72

/-- The initial number of green marbles Janelle had -/
def initial_green_marbles : ℕ := 26

theorem janelle_initial_green_marbles :
  initial_green_marbles = green_marbles_gift + (marbles_left - (blue_bags * marbles_per_bag - blue_marbles_gift)) :=
by sorry

end NUMINAMATH_CALUDE_janelle_initial_green_marbles_l1555_155577


namespace NUMINAMATH_CALUDE_bike_to_tractor_speed_ratio_l1555_155553

/-- Prove that the ratio of bike speed to tractor speed is 2:1 -/
theorem bike_to_tractor_speed_ratio :
  let tractor_speed := 575 / 23
  let car_speed := 360 / 4
  let bike_speed := car_speed / (9/5)
  bike_speed / tractor_speed = 2 := by
  sorry

end NUMINAMATH_CALUDE_bike_to_tractor_speed_ratio_l1555_155553


namespace NUMINAMATH_CALUDE_test_passing_requirement_l1555_155551

def total_questions : ℕ := 80
def arithmetic_questions : ℕ := 15
def algebra_questions : ℕ := 25
def geometry_questions : ℕ := 40

def arithmetic_correct_rate : ℚ := 60 / 100
def algebra_correct_rate : ℚ := 50 / 100
def geometry_correct_rate : ℚ := 70 / 100

def passing_rate : ℚ := 65 / 100

def additional_correct_answers_needed : ℕ := 3

theorem test_passing_requirement : 
  let current_correct := 
    (arithmetic_questions * arithmetic_correct_rate).floor +
    (algebra_questions * algebra_correct_rate).floor +
    (geometry_questions * geometry_correct_rate).floor
  (current_correct + additional_correct_answers_needed : ℚ) / total_questions ≥ passing_rate :=
by sorry

end NUMINAMATH_CALUDE_test_passing_requirement_l1555_155551


namespace NUMINAMATH_CALUDE_square_area_increase_l1555_155592

theorem square_area_increase (s : ℝ) (h : s > 0) :
  let new_side := 1.15 * s
  let original_area := s^2
  let new_area := new_side^2
  (new_area - original_area) / original_area = 0.3225 := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l1555_155592


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l1555_155547

theorem solve_exponential_equation :
  ∃ n : ℕ, (8 : ℝ)^n * (8 : ℝ)^n * (8 : ℝ)^n * (8 : ℝ)^n = (64 : ℝ)^4 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l1555_155547


namespace NUMINAMATH_CALUDE_second_project_depth_l1555_155574

/-- Represents the dimensions of a digging project -/
structure DiggingProject where
  depth : ℝ
  length : ℝ
  breadth : ℝ

/-- Calculates the volume of a digging project -/
def volume (p : DiggingProject) : ℝ :=
  p.depth * p.length * p.breadth

/-- The first digging project -/
def project1 : DiggingProject :=
  { depth := 100, length := 25, breadth := 30 }

/-- The second digging project with unknown depth -/
def project2 (depth : ℝ) : DiggingProject :=
  { depth := depth, length := 20, breadth := 50 }

theorem second_project_depth :
  ∃ d : ℝ, volume project1 = volume (project2 d) ∧ d = 75 := by
  sorry

end NUMINAMATH_CALUDE_second_project_depth_l1555_155574


namespace NUMINAMATH_CALUDE_min_value_zero_l1555_155571

/-- The quadratic function for which we want to find the minimum value -/
def f (m x y : ℝ) : ℝ := 3*x^2 - 4*m*x*y + (2*m^2 + 3)*y^2 - 6*x - 9*y + 8

/-- The theorem stating the value of m that makes the minimum of f equal to 0 -/
theorem min_value_zero (m : ℝ) : 
  (∀ x y : ℝ, f m x y ≥ 0) ∧ (∃ x y : ℝ, f m x y = 0) ↔ 
  m = (6 + Real.sqrt 67.5) / 9 ∨ m = (6 - Real.sqrt 67.5) / 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_zero_l1555_155571


namespace NUMINAMATH_CALUDE_china_students_reading_l1555_155530

/-- Represents how a number is read in words -/
def NumberInWords : Type := String

/-- The correct way to read a given number -/
def correctReading (n : Float) : NumberInWords := sorry

/-- The number of primary school students enrolled in China in 2004 (in millions) -/
def chinaStudents2004 : Float := 11246.23

theorem china_students_reading :
  correctReading chinaStudents2004 = "eleven thousand two hundred forty-six point two three" := by
  sorry

end NUMINAMATH_CALUDE_china_students_reading_l1555_155530


namespace NUMINAMATH_CALUDE_work_done_by_force_l1555_155589

/-- Work done by a force on a particle -/
theorem work_done_by_force (F S : ℝ × ℝ) : 
  F = (-1, -2) → S = (3, 4) → F.1 * S.1 + F.2 * S.2 = -11 := by sorry

end NUMINAMATH_CALUDE_work_done_by_force_l1555_155589


namespace NUMINAMATH_CALUDE_factorization_equality_l1555_155558

theorem factorization_equality (x y : ℝ) : 2*x^2 - 8*x*y + 8*y^2 = 2*(x - 2*y)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l1555_155558


namespace NUMINAMATH_CALUDE_cupcake_package_size_l1555_155501

/-- The number of cupcakes in the smaller package -/
def smaller_package : ℕ := 10

/-- The number of cupcakes in the larger package -/
def larger_package : ℕ := 15

/-- The number of packs of each size bought -/
def packs_bought : ℕ := 4

/-- The total number of children to receive cupcakes -/
def total_children : ℕ := 100

theorem cupcake_package_size :
  packs_bought * larger_package + packs_bought * smaller_package = total_children :=
sorry

end NUMINAMATH_CALUDE_cupcake_package_size_l1555_155501


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_three_l1555_155576

def is_divisible_by_three (n : ℕ) : Prop :=
  ∃ k : ℕ, 8*(n+2)^5 - n^2 + 14*n - 30 = 3*k

theorem largest_n_divisible_by_three :
  (∀ m : ℕ, m < 100000 → is_divisible_by_three m) ∧
  (∀ m : ℕ, m > 99999 → m < 100000 → ¬is_divisible_by_three m) :=
sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_three_l1555_155576


namespace NUMINAMATH_CALUDE_f_has_max_and_min_l1555_155557

/-- A cubic function with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 6

/-- The derivative of f with respect to x -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 - 2 * x + 1

/-- Theorem stating the condition for f to have both maximum and minimum -/
theorem f_has_max_and_min (a : ℝ) : 
  (∃ x y : ℝ, ∀ z : ℝ, f a z ≤ f a x ∧ f a y ≤ f a z) ↔ a < 1/3 ∧ a ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_f_has_max_and_min_l1555_155557


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1555_155584

theorem quadratic_equation_roots (c : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + 2*x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁^2 + x₂^2 = c^2 - 2*c →
  c = -2 ∧ x₁ = -1 + Real.sqrt 3 ∧ x₂ = -1 - Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1555_155584


namespace NUMINAMATH_CALUDE_emily_weight_l1555_155539

def heather_weight : ℕ := 87
def weight_difference : ℕ := 78

theorem emily_weight : 
  ∃ (emily_weight : ℕ), 
    emily_weight = heather_weight - weight_difference ∧ 
    emily_weight = 9 := by
  sorry

end NUMINAMATH_CALUDE_emily_weight_l1555_155539


namespace NUMINAMATH_CALUDE_equal_area_rectangles_length_l1555_155563

/-- Given two rectangles of equal area, where one rectangle has dimensions 4 inches by 30 inches,
    and the other has a width of 24 inches, prove that the length of the second rectangle is 5 inches. -/
theorem equal_area_rectangles_length (area : ℝ) (width : ℝ) :
  area = 4 * 30 →
  width = 24 →
  area = width * 5 :=
by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_length_l1555_155563


namespace NUMINAMATH_CALUDE_weight_of_compound_approx_l1555_155536

/-- The atomic mass of carbon in g/mol -/
def carbon_mass : ℝ := 12.01

/-- The atomic mass of hydrogen in g/mol -/
def hydrogen_mass : ℝ := 1.008

/-- The atomic mass of oxygen in g/mol -/
def oxygen_mass : ℝ := 16.00

/-- The chemical formula of the compound -/
structure ChemicalFormula where
  carbon : ℕ
  hydrogen : ℕ
  oxygen : ℕ

/-- The compound C6H8O7 -/
def compound : ChemicalFormula := ⟨6, 8, 7⟩

/-- Calculate the molar mass of a chemical formula -/
def molar_mass (formula : ChemicalFormula) : ℝ :=
  formula.carbon * carbon_mass + 
  formula.hydrogen * hydrogen_mass + 
  formula.oxygen * oxygen_mass

/-- The number of moles -/
def num_moles : ℝ := 3

/-- The total weight in grams -/
def total_weight : ℝ := 576

/-- Theorem stating that the weight of 3 moles of C6H8O7 is approximately 576 grams -/
theorem weight_of_compound_approx (ε : ℝ) (ε_pos : ε > 0) : 
  |num_moles * molar_mass compound - total_weight| < ε := by
  sorry

end NUMINAMATH_CALUDE_weight_of_compound_approx_l1555_155536


namespace NUMINAMATH_CALUDE_joshua_toy_cars_l1555_155545

theorem joshua_toy_cars (total_boxes : ℕ) (cars_box1 : ℕ) (cars_box2 : ℕ) (total_cars : ℕ)
  (h1 : total_boxes = 3)
  (h2 : cars_box1 = 21)
  (h3 : cars_box2 = 31)
  (h4 : total_cars = 71) :
  total_cars - (cars_box1 + cars_box2) = 19 := by
  sorry

end NUMINAMATH_CALUDE_joshua_toy_cars_l1555_155545


namespace NUMINAMATH_CALUDE_decimal_89_equals_base5_324_l1555_155579

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: toBase5 (n / 5)

theorem decimal_89_equals_base5_324 : toBase5 89 = [4, 2, 3] := by
  sorry

end NUMINAMATH_CALUDE_decimal_89_equals_base5_324_l1555_155579


namespace NUMINAMATH_CALUDE_smallest_number_with_remainders_l1555_155538

theorem smallest_number_with_remainders : ∃! x : ℕ, 
  x % 4 = 1 ∧ 
  x % 3 = 2 ∧ 
  x % 5 = 3 ∧ 
  ∀ y : ℕ, (y % 4 = 1 ∧ y % 3 = 2 ∧ y % 5 = 3) → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_with_remainders_l1555_155538


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l1555_155533

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  ((c^2 = a^2 + b^2) ∨ (b^2 = a^2 + c^2)) → 
  ((d^2 = b^2 - a^2) ∨ (b^2 = a^2 + d^2)) → 
  c * d = 20 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l1555_155533
