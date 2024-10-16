import Mathlib

namespace NUMINAMATH_CALUDE_triangle_problem_l2522_252221

open Real

theorem triangle_problem (A B C : ℝ) (a b c S : ℝ) :
  (2 * sin B - 2 * sin B ^ 2 - cos (2 * B) = sqrt 3 - 1) →
  (B = π / 3 ∨ B = 2 * π / 3) ∧
  (B = π / 3 ∧ a = 6 ∧ S = 6 * sqrt 3 → b = 2 * sqrt 7) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2522_252221


namespace NUMINAMATH_CALUDE_min_presses_to_exceed_200_l2522_252293

def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

def exceed_200 (x : ℕ) : ℕ :=
  match x with
  | 0 => 0
  | n + 1 => if repeated_square 3 n > 200 then n else exceed_200 n

theorem min_presses_to_exceed_200 : exceed_200 0 = 3 := by sorry

end NUMINAMATH_CALUDE_min_presses_to_exceed_200_l2522_252293


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2522_252292

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) :
  ∃ (k : ℤ), (n - 1) + n + (n + 1) + (n + 2) = 2 * k := by
  sorry

#check sum_of_four_consecutive_integers_divisible_by_two

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l2522_252292


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_log_sum_l2522_252201

theorem arithmetic_geometric_mean_log_sum (a b c x y z m : ℝ) 
  (hb : b = (a + c) / 2)
  (hy : y^2 = x * z)
  (hx : x > 0)
  (hy_pos : y > 0)
  (hz : z > 0)
  (hm : m > 0 ∧ m ≠ 1) :
  (b - c) * (Real.log x / Real.log m) + 
  (c - a) * (Real.log y / Real.log m) + 
  (a - b) * (Real.log z / Real.log m) = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_log_sum_l2522_252201


namespace NUMINAMATH_CALUDE_rectangular_frame_area_l2522_252215

theorem rectangular_frame_area : 
  let width : ℚ := 81 / 4
  let depth : ℚ := 148 / 9
  let area : ℚ := width * depth
  ⌊area⌋ = 333 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_frame_area_l2522_252215


namespace NUMINAMATH_CALUDE_range_of_x_when_a_is_zero_range_of_a_when_p_implies_q_l2522_252291

-- Define the conditions
def p (x : ℝ) : Prop := 4 * x^2 + 12 * x - 7 ≤ 0
def q (a x : ℝ) : Prop := a - 3 ≤ x ∧ x ≤ a + 3

-- Theorem for the first question
theorem range_of_x_when_a_is_zero :
  ∀ x : ℝ, (p x ∧ ¬(q 0 x)) → (-7/2 ≤ x ∧ x < -3) :=
sorry

-- Theorem for the second question
theorem range_of_a_when_p_implies_q :
  (∀ x : ℝ, p x → q a x) → (-5/2 ≤ a ∧ a ≤ -1/2) :=
sorry

end NUMINAMATH_CALUDE_range_of_x_when_a_is_zero_range_of_a_when_p_implies_q_l2522_252291


namespace NUMINAMATH_CALUDE_circle_radius_proof_l2522_252234

theorem circle_radius_proof (r : ℝ) : r > 0 → 3 * (2 * π * r) = 2 * (π * r^2) → r = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l2522_252234


namespace NUMINAMATH_CALUDE_sqrt_negative_product_equals_two_sqrt_two_l2522_252288

theorem sqrt_negative_product_equals_two_sqrt_two :
  Real.sqrt ((-4) * (-2)) = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_sqrt_negative_product_equals_two_sqrt_two_l2522_252288


namespace NUMINAMATH_CALUDE_quadratic_properties_l2522_252289

/-- A quadratic function f(x) = mx² + (m-2)x + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + (m - 2) * x + 2

/-- The function is symmetric about the y-axis -/
def is_symmetric (m : ℝ) : Prop := ∀ x, f m x = f m (-x)

theorem quadratic_properties (m : ℝ) (h : is_symmetric m) :
  m = 2 ∧ 
  (∀ x y, x < y → f m x < f m y) ∧ 
  (∀ x, x > 0 → f m x > f m 0) ∧
  (∀ x, f m x ≥ 2) ∧ 
  f m 0 = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l2522_252289


namespace NUMINAMATH_CALUDE_number_of_pickers_l2522_252265

theorem number_of_pickers (drums_per_day : ℕ) (total_days : ℕ) (total_drums : ℕ) :
  drums_per_day = 221 →
  total_days = 77 →
  total_drums = 17017 →
  drums_per_day * total_days = total_drums →
  drums_per_day = 221 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_pickers_l2522_252265


namespace NUMINAMATH_CALUDE_kate_bouncy_balls_difference_l2522_252298

/-- The number of bouncy balls in each pack -/
def balls_per_pack : ℕ := 18

/-- The number of packs of red bouncy balls Kate bought -/
def red_packs : ℕ := 7

/-- The number of packs of yellow bouncy balls Kate bought -/
def yellow_packs : ℕ := 6

/-- The total number of red bouncy balls Kate bought -/
def total_red_balls : ℕ := red_packs * balls_per_pack

/-- The total number of yellow bouncy balls Kate bought -/
def total_yellow_balls : ℕ := yellow_packs * balls_per_pack

/-- The difference between the number of red and yellow bouncy balls -/
def difference_in_balls : ℕ := total_red_balls - total_yellow_balls

theorem kate_bouncy_balls_difference :
  difference_in_balls = 18 := by sorry

end NUMINAMATH_CALUDE_kate_bouncy_balls_difference_l2522_252298


namespace NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2522_252231

theorem average_marks_chemistry_mathematics 
  (P C M : ℕ) -- P: Physics marks, C: Chemistry marks, M: Mathematics marks
  (h : P + C + M = P + 130) -- Total marks condition
  : (C + M) / 2 = 65 := by
  sorry

end NUMINAMATH_CALUDE_average_marks_chemistry_mathematics_l2522_252231


namespace NUMINAMATH_CALUDE_fabric_price_system_l2522_252297

/-- Represents the price per foot of damask fabric in wen -/
def damask_price : ℝ := sorry

/-- Represents the price per foot of gauze fabric in wen -/
def gauze_price : ℝ := sorry

/-- The length of the damask fabric in feet -/
def damask_length : ℝ := 7

/-- The length of the gauze fabric in feet -/
def gauze_length : ℝ := 9

/-- The price difference per foot between damask and gauze fabrics in wen -/
def price_difference : ℝ := 36

theorem fabric_price_system :
  (damask_length * damask_price = gauze_length * gauze_price) ∧
  (damask_price - gauze_price = price_difference) := by sorry

end NUMINAMATH_CALUDE_fabric_price_system_l2522_252297


namespace NUMINAMATH_CALUDE_commutative_property_demonstration_l2522_252259

theorem commutative_property_demonstration :
  (2 + 1 + 5 - 1 = 2 - 1 + 1 + 5) →
  ∃ (a b c d : ℤ), a + b + c + d = b + c + d + a :=
by sorry

end NUMINAMATH_CALUDE_commutative_property_demonstration_l2522_252259


namespace NUMINAMATH_CALUDE_dans_cards_count_l2522_252274

/-- The number of Pokemon cards Dan gave to Sally -/
def cards_from_dan (initial_cards : ℕ) (bought_cards : ℕ) (total_cards : ℕ) : ℕ :=
  total_cards - initial_cards - bought_cards

/-- Theorem stating that Dan gave Sally 41 Pokemon cards -/
theorem dans_cards_count : cards_from_dan 27 20 88 = 41 := by
  sorry

end NUMINAMATH_CALUDE_dans_cards_count_l2522_252274


namespace NUMINAMATH_CALUDE_max_ab_for_line_circle_intersection_l2522_252233

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

end NUMINAMATH_CALUDE_max_ab_for_line_circle_intersection_l2522_252233


namespace NUMINAMATH_CALUDE_jackson_souvenirs_l2522_252217

/-- Calculates the total number of souvenirs collected by Jackson -/
def total_souvenirs (hermit_crabs : ℕ) (shells_per_crab : ℕ) (starfish_per_shell : ℕ) : ℕ :=
  let spiral_shells := hermit_crabs * shells_per_crab
  let starfish := spiral_shells * starfish_per_shell
  hermit_crabs + spiral_shells + starfish

/-- Proves that Jackson collects 450 souvenirs in total -/
theorem jackson_souvenirs :
  total_souvenirs 45 3 2 = 450 := by
  sorry

#eval total_souvenirs 45 3 2

end NUMINAMATH_CALUDE_jackson_souvenirs_l2522_252217


namespace NUMINAMATH_CALUDE_parallel_transitivity_l2522_252248

-- Define the type for planes
variable {Plane : Type}

-- Define the parallel relation between planes
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_transitivity 
  (α β γ : Plane) 
  (h1 : parallel γ α) 
  (h2 : parallel γ β) : 
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_transitivity_l2522_252248


namespace NUMINAMATH_CALUDE_fraction_multiplication_l2522_252209

theorem fraction_multiplication : (2 : ℚ) / 3 * (3 : ℚ) / 8 = (1 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l2522_252209


namespace NUMINAMATH_CALUDE_gabby_needs_ten_more_dollars_l2522_252210

def makeup_set_cost : ℕ := 65
def gabby_initial_savings : ℕ := 35
def mom_additional_money : ℕ := 20

theorem gabby_needs_ten_more_dollars : 
  makeup_set_cost - (gabby_initial_savings + mom_additional_money) = 10 := by
sorry

end NUMINAMATH_CALUDE_gabby_needs_ten_more_dollars_l2522_252210


namespace NUMINAMATH_CALUDE_unique_two_digit_number_l2522_252261

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def tens_digit (n : ℕ) : ℕ := n / 10

def ones_digit (n : ℕ) : ℕ := n % 10

def swap_digits (n : ℕ) : ℕ := 10 * (ones_digit n) + tens_digit n

theorem unique_two_digit_number : 
  ∃! n : ℕ, is_two_digit n ∧ 
    (tens_digit n * ones_digit n = 2 * (tens_digit n + ones_digit n)) ∧
    (n + 9 = 2 * swap_digits n) ∧
    n = 63 := by
  sorry

end NUMINAMATH_CALUDE_unique_two_digit_number_l2522_252261


namespace NUMINAMATH_CALUDE_inequality_proof_l2522_252224

def f (x : ℝ) : ℝ := 2 * abs (x - 1) + x - 1

def g (x : ℝ) : ℝ := 16 * x^2 - 8 * x + 1

def M : Set ℝ := {x | f x ≤ 1}

def N : Set ℝ := {x | g x ≤ 4}

theorem inequality_proof (x : ℝ) (hx : x ∈ M ∩ N) : x^2 * f x + x * (f x)^2 ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2522_252224


namespace NUMINAMATH_CALUDE_gcd_1237_1957_l2522_252255

theorem gcd_1237_1957 : Nat.gcd 1237 1957 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1237_1957_l2522_252255


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2522_252208

/-- An isosceles triangle with side lengths 3 and 7 has a perimeter of 17 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 3 ∧ b = 7 ∧ c = 7 →  -- Two sides are 7, one side is 3
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 17 := by  -- Perimeter is 17
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2522_252208


namespace NUMINAMATH_CALUDE_intersection_M_N_l2522_252245

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≠ x}

theorem intersection_M_N : M ∩ N = {-1, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2522_252245


namespace NUMINAMATH_CALUDE_carries_mountain_dew_oz_per_can_l2522_252219

/-- Represents the punch recipe and serving information -/
structure PunchRecipe where
  mountain_dew_cans : ℕ
  ice_oz : ℕ
  fruit_juice_oz : ℕ
  total_servings : ℕ
  oz_per_serving : ℕ

/-- Calculates the ounces of Mountain Dew per can -/
def mountain_dew_oz_per_can (recipe : PunchRecipe) : ℕ :=
  let total_oz := recipe.total_servings * recipe.oz_per_serving
  let non_mountain_dew_oz := recipe.ice_oz + recipe.fruit_juice_oz
  let total_mountain_dew_oz := total_oz - non_mountain_dew_oz
  total_mountain_dew_oz / recipe.mountain_dew_cans

/-- Carrie's punch recipe -/
def carries_recipe : PunchRecipe := {
  mountain_dew_cans := 6
  ice_oz := 28
  fruit_juice_oz := 40
  total_servings := 14
  oz_per_serving := 10
}

/-- Theorem stating that each can of Mountain Dew in Carrie's recipe contains 12 oz -/
theorem carries_mountain_dew_oz_per_can :
  mountain_dew_oz_per_can carries_recipe = 12 := by
  sorry

end NUMINAMATH_CALUDE_carries_mountain_dew_oz_per_can_l2522_252219


namespace NUMINAMATH_CALUDE_prime_pairs_perfect_square_l2522_252264

theorem prime_pairs_perfect_square :
  ∀ a b : ℕ,
  Prime a → Prime b → a > 0 → b > 0 →
  (∃ k : ℕ, 3 * a^2 * b + 16 * a * b^2 = k^2) →
  ((a = 19 ∧ b = 19) ∨ (a = 2 ∧ b = 3)) :=
sorry

end NUMINAMATH_CALUDE_prime_pairs_perfect_square_l2522_252264


namespace NUMINAMATH_CALUDE_ab_geq_one_implies_conditions_l2522_252226

theorem ab_geq_one_implies_conditions (a b : ℝ) (h : a * b ≥ 1) :
  a^2 ≥ 1 / b^2 ∧ a^2 + b^2 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_ab_geq_one_implies_conditions_l2522_252226


namespace NUMINAMATH_CALUDE_tangent_problem_l2522_252211

theorem tangent_problem (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β + Real.pi/4) = 1/4) :
  Real.tan (α - Real.pi/4) = 3/22 := by
  sorry

end NUMINAMATH_CALUDE_tangent_problem_l2522_252211


namespace NUMINAMATH_CALUDE_expression_evaluation_l2522_252232

theorem expression_evaluation : 6 * 5 * ((-1) ^ (2 ^ (3 ^ 5))) + ((-1) ^ (5 ^ (3 ^ 2))) = 29 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2522_252232


namespace NUMINAMATH_CALUDE_binary_1011011_eq_91_l2522_252236

/-- Converts a list of binary digits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1011011₂ -/
def binary_1011011 : List Bool := [true, true, false, true, true, false, true]

/-- Theorem: The decimal equivalent of 1011011₂ is 91 -/
theorem binary_1011011_eq_91 : binary_to_decimal binary_1011011 = 91 := by
  sorry

end NUMINAMATH_CALUDE_binary_1011011_eq_91_l2522_252236


namespace NUMINAMATH_CALUDE_letters_identity_l2522_252212

-- Define the Letter type
inductive Letter
| A
| B

-- Define a function to represent whether a letter tells the truth
def tellsTruth (l : Letter) : Bool :=
  match l with
  | Letter.A => true
  | Letter.B => false

-- Define the statements made by each letter
def statement1 (l1 l2 l3 : Letter) : Prop :=
  (l1 = l2 ∧ l1 ≠ l3) ∨ (l1 = l3 ∧ l1 ≠ l2)

def statement2 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.A ∧ l2 = Letter.A ∧ l3 = Letter.B) ∨
  (l1 = Letter.A ∧ l2 = Letter.B ∧ l3 = Letter.B) ∨
  (l1 = Letter.B ∧ l2 = Letter.A ∧ l3 = Letter.B) ∨
  (l1 = Letter.B ∧ l2 = Letter.B ∧ l3 = Letter.B)

def statement3 (l1 l2 l3 : Letter) : Prop :=
  (l1 = Letter.B ∧ l2 ≠ Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 = Letter.B ∧ l3 ≠ Letter.B) ∨
  (l1 ≠ Letter.B ∧ l2 ≠ Letter.B ∧ l3 = Letter.B)

-- Define the main theorem
theorem letters_identity :
  ∃! (l1 l2 l3 : Letter),
    (tellsTruth l1 = statement1 l1 l2 l3) ∧
    (tellsTruth l2 = statement2 l1 l2 l3) ∧
    (tellsTruth l3 = statement3 l1 l2 l3) ∧
    l1 = Letter.B ∧ l2 = Letter.A ∧ l3 = Letter.A :=
by sorry

end NUMINAMATH_CALUDE_letters_identity_l2522_252212


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2522_252246

/-- Given two perpendicular vectors a and b in ℝ², prove that m = 2 -/
theorem perpendicular_vectors_m_value (a b : ℝ × ℝ) (h : a.1 * b.1 + a.2 * b.2 = 0) 
  (ha : a = (-2, 3)) (hb : b = (3, m)) : m = 2 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l2522_252246


namespace NUMINAMATH_CALUDE_joe_lifts_l2522_252256

theorem joe_lifts (total_weight first_lift : ℕ) 
  (h1 : total_weight = 900)
  (h2 : first_lift = 400) :
  total_weight - first_lift = first_lift + 100 := by
  sorry

end NUMINAMATH_CALUDE_joe_lifts_l2522_252256


namespace NUMINAMATH_CALUDE_peters_leaf_raking_l2522_252205

/-- Given that Peter rakes 3 bags of leaves in 15 minutes at a constant rate,
    prove that it will take him 40 minutes to rake 8 bags of leaves. -/
theorem peters_leaf_raking (rate : ℚ) : 
  (rate * 15 = 3) → (rate * 40 = 8) :=
by sorry

end NUMINAMATH_CALUDE_peters_leaf_raking_l2522_252205


namespace NUMINAMATH_CALUDE_probability_prime_or_odd_l2522_252241

/-- A function that determines if a natural number is prime -/
def isPrime (n : ℕ) : Prop := sorry

/-- A function that determines if a natural number is odd -/
def isOdd (n : ℕ) : Prop := sorry

/-- The set of balls numbered 1 through 8 -/
def ballSet : Finset ℕ := sorry

/-- The probability of selecting a ball with a number that is either prime or odd -/
def probabilityPrimeOrOdd : ℚ := sorry

/-- Theorem stating that the probability of selecting a ball with a number
    that is either prime or odd is 5/8 -/
theorem probability_prime_or_odd :
  probabilityPrimeOrOdd = 5 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_prime_or_odd_l2522_252241


namespace NUMINAMATH_CALUDE_jeremy_wednesday_oranges_l2522_252267

/-- The number of oranges Jeremy picked on different days and the total -/
structure OrangePicks where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  total : ℕ

/-- Given the conditions of Jeremy's orange picking, prove that he picked 70 oranges on Wednesday -/
theorem jeremy_wednesday_oranges (picks : OrangePicks) 
  (h1 : picks.monday = 100)
  (h2 : picks.tuesday = 3 * picks.monday)
  (h3 : picks.total = 470)
  (h4 : picks.total = picks.monday + picks.tuesday + picks.wednesday) :
  picks.wednesday = 70 := by
  sorry


end NUMINAMATH_CALUDE_jeremy_wednesday_oranges_l2522_252267


namespace NUMINAMATH_CALUDE_square_plus_inverse_square_l2522_252200

theorem square_plus_inverse_square (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  x^2 + 1/x^2 = 11 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_inverse_square_l2522_252200


namespace NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l2522_252202

/-- A regular polygon with interior angles of 150 degrees has 12 sides. -/
theorem regular_polygon_with_150_degree_angles_has_12_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 150 → n * angle = (n - 2) * 180) →
  n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_with_150_degree_angles_has_12_sides_l2522_252202


namespace NUMINAMATH_CALUDE_next_friday_birthday_l2522_252260

/-- Represents the day of the week --/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Checks if a given year is a leap year --/
def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)

/-- Calculates the day of the week for May 27 in a given year, 
    assuming May 27, 2013 was a Monday --/
def dayOfWeekMay27 (year : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The next year after 2013 when May 27 falls on a Friday is 2016 --/
theorem next_friday_birthday : 
  (dayOfWeekMay27 2013 = DayOfWeek.Monday) → 
  (∀ y : Nat, 2013 < y ∧ y < 2016 → dayOfWeekMay27 y ≠ DayOfWeek.Friday) ∧
  (dayOfWeekMay27 2016 = DayOfWeek.Friday) :=
sorry

end NUMINAMATH_CALUDE_next_friday_birthday_l2522_252260


namespace NUMINAMATH_CALUDE_binomial_10_2_l2522_252237

theorem binomial_10_2 : Nat.choose 10 2 = 45 := by sorry

end NUMINAMATH_CALUDE_binomial_10_2_l2522_252237


namespace NUMINAMATH_CALUDE_pool_tiles_l2522_252283

theorem pool_tiles (blue_tiles : ℕ) (red_tiles : ℕ) (additional_tiles : ℕ) : 
  blue_tiles = 48 → red_tiles = 32 → additional_tiles = 20 →
  blue_tiles + red_tiles + additional_tiles = 100 := by
  sorry

end NUMINAMATH_CALUDE_pool_tiles_l2522_252283


namespace NUMINAMATH_CALUDE_last_digit_of_expression_l2522_252249

theorem last_digit_of_expression : ∃ n : ℕ, (287 * 287 + 269 * 269 - 2 * 287 * 269) % 10 = 8 ∧ 10 * n + 8 = 287 * 287 + 269 * 269 - 2 * 287 * 269 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_expression_l2522_252249


namespace NUMINAMATH_CALUDE_joshua_bottle_caps_l2522_252266

/-- Given an initial number of bottle caps and a number of newly purchased bottle caps,
    calculate the total number of bottle caps. -/
def total_bottle_caps (initial : ℕ) (newly_bought : ℕ) : ℕ :=
  initial + newly_bought

/-- Theorem stating that given 40 initial bottle caps and 7 newly bought bottle caps,
    the total number of bottle caps is 47. -/
theorem joshua_bottle_caps :
  total_bottle_caps 40 7 = 47 := by
  sorry

end NUMINAMATH_CALUDE_joshua_bottle_caps_l2522_252266


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2522_252235

theorem trigonometric_identity (α : ℝ) :
  Real.sin (4 * α) - Real.sin (5 * α) - Real.sin (6 * α) + Real.sin (7 * α) =
  -4 * Real.sin (α / 2) * Real.sin α * Real.sin (11 * α / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2522_252235


namespace NUMINAMATH_CALUDE_cube_geometric_shapes_l2522_252272

structure Cube where
  vertices : Fin 8 → ℝ × ℝ × ℝ

def isRectangle (points : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  sorry

def isParallelogramNotRectangle (points : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  sorry

def isRightAngledTetrahedron (points : Fin 4 → ℝ × ℝ × ℝ) : Prop :=
  sorry

def fourVertices (c : Cube) : Type :=
  {v : Fin 4 → Fin 8 // ∀ i j, i ≠ j → v i ≠ v j}

theorem cube_geometric_shapes (c : Cube) :
  ∃ (v : fourVertices c),
    isRectangle (fun i => c.vertices (v.val i)) ∧
    isParallelogramNotRectangle (fun i => c.vertices (v.val i)) ∧
    isRightAngledTetrahedron (fun i => c.vertices (v.val i)) ∧
    ¬∃ (w : fourVertices c),
      (isRectangle (fun i => c.vertices (w.val i)) ∧
       isParallelogramNotRectangle (fun i => c.vertices (w.val i)) ∧
       isRightAngledTetrahedron (fun i => c.vertices (w.val i)) ∧
       (isRectangle (fun i => c.vertices (w.val i)) ≠
        isRectangle (fun i => c.vertices (v.val i)) ∨
        isParallelogramNotRectangle (fun i => c.vertices (w.val i)) ≠
        isParallelogramNotRectangle (fun i => c.vertices (v.val i)) ∨
        isRightAngledTetrahedron (fun i => c.vertices (w.val i)) ≠
        isRightAngledTetrahedron (fun i => c.vertices (v.val i)))) :=
  sorry

end NUMINAMATH_CALUDE_cube_geometric_shapes_l2522_252272


namespace NUMINAMATH_CALUDE_expense_reduction_equation_l2522_252250

/-- Represents the average monthly reduction rate as a real number between 0 and 1 -/
def reduction_rate : ℝ := sorry

/-- The initial monthly expenses in yuan -/
def initial_expenses : ℝ := 2500

/-- The final monthly expenses after two months in yuan -/
def final_expenses : ℝ := 1600

/-- The number of months over which the reduction occurred -/
def num_months : ℕ := 2

theorem expense_reduction_equation :
  initial_expenses * (1 - reduction_rate) ^ num_months = final_expenses :=
sorry

end NUMINAMATH_CALUDE_expense_reduction_equation_l2522_252250


namespace NUMINAMATH_CALUDE_sequential_no_conditional_l2522_252216

-- Define the structures
inductive FlowchartStructure
  | Sequential
  | Loop
  | If
  | Until

-- Define a predicate for structures that generally contain a conditional judgment box
def hasConditionalJudgment : FlowchartStructure → Prop
  | FlowchartStructure.Sequential => False
  | FlowchartStructure.Loop => True
  | FlowchartStructure.If => True
  | FlowchartStructure.Until => True

theorem sequential_no_conditional : 
  ∀ (s : FlowchartStructure), ¬hasConditionalJudgment s ↔ s = FlowchartStructure.Sequential :=
by sorry

end NUMINAMATH_CALUDE_sequential_no_conditional_l2522_252216


namespace NUMINAMATH_CALUDE_sphere_cross_section_area_l2522_252222

theorem sphere_cross_section_area (R d : ℝ) (h1 : R = 3) (h2 : d = 2) :
  let r := (R^2 - d^2).sqrt
  π * r^2 = 5 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_cross_section_area_l2522_252222


namespace NUMINAMATH_CALUDE_problem_solution_l2522_252230

theorem problem_solution (s P k : ℝ) (h : P = s / Real.sqrt ((1 + k) ^ n)) :
  n = (2 * Real.log (s / P)) / Real.log (1 + k) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2522_252230


namespace NUMINAMATH_CALUDE_probability_three_girls_in_six_children_l2522_252227

theorem probability_three_girls_in_six_children :
  let n : ℕ := 6  -- Total number of children
  let k : ℕ := 3  -- Number of girls we're interested in
  let p : ℚ := 1/2  -- Probability of having a girl
  Nat.choose n k * p^k * (1-p)^(n-k) = 5/16 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_girls_in_six_children_l2522_252227


namespace NUMINAMATH_CALUDE_willow_peach_tree_count_l2522_252270

/-- Represents the dimensions of a rectangular playground -/
structure Playground where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangular playground -/
def perimeter (p : Playground) : ℕ := 2 * (p.length + p.width)

/-- Represents the spacing between trees -/
def treeSpacing : ℕ := 10

/-- Calculates the total number of tree positions along the perimeter -/
def totalTreePositions (p : Playground) : ℕ := perimeter p / treeSpacing

/-- Theorem: The number of willow trees (or peach trees) is half of the total tree positions -/
theorem willow_peach_tree_count (p : Playground) (h1 : p.length = 150) (h2 : p.width = 60) :
  totalTreePositions p / 2 = 21 := by
  sorry

#check willow_peach_tree_count

end NUMINAMATH_CALUDE_willow_peach_tree_count_l2522_252270


namespace NUMINAMATH_CALUDE_fraction_comparison_l2522_252220

theorem fraction_comparison : (19 : ℚ) / 15 < 17 / 13 ∧ 17 / 13 < 15 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l2522_252220


namespace NUMINAMATH_CALUDE_equal_sets_implies_a_equals_one_l2522_252240

theorem equal_sets_implies_a_equals_one (a : ℝ) : 
  ({2, -1} : Set ℝ) = {2, a^2 - 2*a} → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_equal_sets_implies_a_equals_one_l2522_252240


namespace NUMINAMATH_CALUDE_floor_area_K_l2522_252242

/-- The number of circles in the ring -/
def n : ℕ := 7

/-- The radius of the larger circle C -/
def R : ℝ := 35

/-- The radius of each of the n congruent circles -/
noncomputable def r : ℝ := R * (Real.sqrt (2 - 2 * Real.cos (2 * Real.pi / n))) / 2

/-- The area K of the region inside circle C and outside all n circles -/
noncomputable def K : ℝ := Real.pi * (R^2 - n * r^2)

theorem floor_area_K : ⌊K⌋ = 1476 := by sorry

end NUMINAMATH_CALUDE_floor_area_K_l2522_252242


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_max_l2522_252295

theorem triangle_cosine_sum_max (A B C : ℝ) : 
  0 ≤ A ∧ 0 ≤ B ∧ 0 ≤ C ∧ A + B + C = π →
  Real.cos A + Real.cos B * Real.cos C ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_max_l2522_252295


namespace NUMINAMATH_CALUDE_peach_tree_count_l2522_252285

theorem peach_tree_count (almond_trees : ℕ) (peach_trees : ℕ) : 
  almond_trees = 300 →
  peach_trees = 2 * almond_trees - 30 →
  peach_trees = 570 := by
sorry

end NUMINAMATH_CALUDE_peach_tree_count_l2522_252285


namespace NUMINAMATH_CALUDE_classroom_gpa_l2522_252204

theorem classroom_gpa (N : ℝ) (h : N > 0) :
  let gpa_one_third := 54
  let gpa_whole := 48
  let gpa_rest := (3 * gpa_whole - gpa_one_third) / 2
  gpa_rest = 45 := by sorry

end NUMINAMATH_CALUDE_classroom_gpa_l2522_252204


namespace NUMINAMATH_CALUDE_min_value_expression_l2522_252206

theorem min_value_expression (x : ℝ) (h : x > 0) :
  9 * x^2 + 1 / x^6 ≥ 4 * 3^(3/4) ∧
  ∃ y > 0, 9 * y^2 + 1 / y^6 = 4 * 3^(3/4) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l2522_252206


namespace NUMINAMATH_CALUDE_odd_periodic_function_difference_l2522_252214

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- A function f has period 4 if f(x) = f(x + 4) for all x -/
def HasPeriod4 (f : ℝ → ℝ) : Prop := ∀ x, f x = f (x + 4)

theorem odd_periodic_function_difference (f : ℝ → ℝ) 
  (h_odd : IsOdd f) 
  (h_period : HasPeriod4 f) 
  (h_def : ∀ x ∈ Set.Ioo (-2) 0, f x = 2^x) : 
  f 2016 - f 2015 = -1/2 := by
sorry

end NUMINAMATH_CALUDE_odd_periodic_function_difference_l2522_252214


namespace NUMINAMATH_CALUDE_correct_seating_count_l2522_252203

/-- Number of Democrats in the Senate committee -/
def num_democrats : ℕ := 6

/-- Number of Republicans in the Senate committee -/
def num_republicans : ℕ := 6

/-- Number of Independents in the Senate committee -/
def num_independents : ℕ := 2

/-- Total number of committee members -/
def total_members : ℕ := num_democrats + num_republicans + num_independents

/-- Function to calculate the number of valid seating arrangements -/
def seating_arrangements : ℕ :=
  12 * (Nat.factorial 10) / 2

/-- Theorem stating the number of valid seating arrangements -/
theorem correct_seating_count :
  seating_arrangements = 21772800 := by sorry

end NUMINAMATH_CALUDE_correct_seating_count_l2522_252203


namespace NUMINAMATH_CALUDE_calculation_proof_l2522_252286

theorem calculation_proof :
  (0.001)^(-1/3) + 27^(2/3) + (1/4)^(-1/2) - (1/9)^(-3/2) = -6 ∧
  1/2 * Real.log 25 / Real.log 10 + Real.log 2 / Real.log 10 - Real.log (Real.sqrt 0.1) / Real.log 10 - 
    (Real.log 9 / Real.log 2) * (Real.log 2 / Real.log 3) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2522_252286


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l2522_252253

theorem angle_measure_in_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = π ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  b / (Real.sin B) = c / (Real.sin C) ∧
  (2 * b - c) * Real.cos A = a * Real.cos C →
  A = π / 3 := by
sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l2522_252253


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l2522_252262

/-- Given an arithmetic sequence starting at 200, ending at 0, with a common difference of -5,
    the number of terms in the sequence is 41. -/
theorem arithmetic_sequence_length : 
  let start : ℤ := 200
  let end_val : ℤ := 0
  let diff : ℤ := -5
  let n : ℤ := (start - end_val) / (-diff) + 1
  n = 41 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l2522_252262


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l2522_252275

/-- The minimum distance from a point on the circle x^2 + y^2 = 1 to the line x - y = 2 is √2 - 1 -/
theorem min_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let line := {(x, y) : ℝ × ℝ | x - y = 2}
  (∃ (p : ℝ × ℝ), p ∈ circle ∧
    (∀ (q : ℝ × ℝ), q ∈ circle →
      ∀ (r : ℝ × ℝ), r ∈ line →
        dist p r ≥ Real.sqrt 2 - 1)) ∧
  (∃ (p : ℝ × ℝ) (r : ℝ × ℝ), p ∈ circle ∧ r ∈ line ∧ dist p r = Real.sqrt 2 - 1) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l2522_252275


namespace NUMINAMATH_CALUDE_sum_of_z_values_l2522_252276

-- Define the function f
def f (x : ℝ) : ℝ := (2*x)^2 - 2*(2*x) + 2

-- State the theorem
theorem sum_of_z_values (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 4 ∧ f z₂ = 4 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = 1/2) := by sorry

end NUMINAMATH_CALUDE_sum_of_z_values_l2522_252276


namespace NUMINAMATH_CALUDE_max_abs_z_value_l2522_252238

theorem max_abs_z_value (a b c z : ℂ) 
  (ha : Complex.abs a = 1)
  (hb : Complex.abs b = 1)
  (hc : Complex.abs c = 1)
  (heq : a * z^2 + 2 * b * z + c = 0) :
  Complex.abs z ≤ 1 + Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_value_l2522_252238


namespace NUMINAMATH_CALUDE_negation_of_implication_for_all_negation_of_zero_product_l2522_252251

theorem negation_of_implication_for_all (P Q : ℝ → ℝ → Prop) :
  (¬ ∀ a b : ℝ, P a b → Q a b) ↔ (∃ a b : ℝ, P a b ∧ ¬ Q a b) :=
by sorry

theorem negation_of_zero_product :
  (¬ ∀ a b : ℝ, a = 0 → a * b = 0) ↔ (∃ a b : ℝ, a = 0 ∧ a * b ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_for_all_negation_of_zero_product_l2522_252251


namespace NUMINAMATH_CALUDE_distance_center_to_endpoint_l2522_252279

/-- Given two points representing the endpoints of a circle's diameter,
    calculate the distance from the center of the circle to one of the endpoints. -/
theorem distance_center_to_endpoint
  (p1 : ℝ × ℝ)
  (p2 : ℝ × ℝ)
  (h1 : p1 = (12, -8))
  (h2 : p2 = (-6, 4))
  : Real.sqrt ((12 - ((p1.1 + p2.1) / 2))^2 + (-8 - ((p1.2 + p2.2) / 2))^2) = Real.sqrt 117 :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_endpoint_l2522_252279


namespace NUMINAMATH_CALUDE_sqrt_two_expansion_l2522_252268

theorem sqrt_two_expansion (a b : ℚ) : 
  (1 + Real.sqrt 2)^5 = a + Real.sqrt 2 * b → a - b = 12 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_expansion_l2522_252268


namespace NUMINAMATH_CALUDE_total_price_theorem_l2522_252296

def refrigerator_price : ℝ := 4275
def washing_machine_price : ℝ := refrigerator_price - 1490
def sales_tax_rate : ℝ := 0.07

def total_price_with_tax : ℝ :=
  (refrigerator_price + washing_machine_price) * (1 + sales_tax_rate)

theorem total_price_theorem :
  total_price_with_tax = 7554.20 := by sorry

end NUMINAMATH_CALUDE_total_price_theorem_l2522_252296


namespace NUMINAMATH_CALUDE_bus_stops_count_l2522_252294

/-- Represents a bus route in the city -/
structure BusRoute where
  stops : ℕ
  stops_ge_three : stops ≥ 3

/-- Represents the city's bus system -/
structure BusSystem where
  routes : Finset BusRoute
  route_count : routes.card = 57
  all_connected : ∀ (r₁ r₂ : BusRoute), r₁ ∈ routes → r₂ ∈ routes → ∃! (s : ℕ), s ≤ r₁.stops ∧ s ≤ r₂.stops
  stops_equal : ∀ (r₁ r₂ : BusRoute), r₁ ∈ routes → r₂ ∈ routes → r₁.stops = r₂.stops

theorem bus_stops_count (bs : BusSystem) : ∀ (r : BusRoute), r ∈ bs.routes → r.stops = 8 := by
  sorry

end NUMINAMATH_CALUDE_bus_stops_count_l2522_252294


namespace NUMINAMATH_CALUDE_inverse_inequality_l2522_252282

theorem inverse_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : 1 / a > 1 / b := by
  sorry

end NUMINAMATH_CALUDE_inverse_inequality_l2522_252282


namespace NUMINAMATH_CALUDE_order_of_numbers_l2522_252269

theorem order_of_numbers : 
  0 < 0.89 → 0.89 < 1 → 90.8 > 1 → Real.log 0.89 < 0 → 
  Real.log 0.89 < 0.89 ∧ 0.89 < 90.8 := by
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l2522_252269


namespace NUMINAMATH_CALUDE_system_solution_l2522_252229

theorem system_solution :
  ∃! (x y : ℝ), 3 * x - 2 * y = 6 ∧ 2 * x + 3 * y = 17 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_system_solution_l2522_252229


namespace NUMINAMATH_CALUDE_equation_solution_l2522_252257

noncomputable def f (x : ℝ) : ℝ := x + Real.arctan x * Real.sqrt (x^2 + 1)

theorem equation_solution :
  ∃! x : ℝ, 2*x + 2 + f x + f (x + 2) = 0 ∧ x = -1 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2522_252257


namespace NUMINAMATH_CALUDE_ratio_equality_l2522_252273

theorem ratio_equality (a b c d : ℚ) 
  (h1 : a = 4 * b) 
  (h2 : b = 5 * c) 
  (h3 : c = 3 * d) : 
  a * d / (b * c) = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l2522_252273


namespace NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l2522_252284

theorem quadratic_solution_implies_sum (a b : ℝ) : 
  (a * 2^2 - b * 2 + 2 = 0) → (2024 + 2*a - b = 2023) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_implies_sum_l2522_252284


namespace NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2522_252263

theorem unique_solution_trigonometric_equation :
  ∃! x : Real,
    0 < x ∧ x < 180 ∧
    Real.tan (120 - x) = (Real.sin 120 - Real.sin x) / (Real.cos 120 - Real.cos x) ∧
    x = 100 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_trigonometric_equation_l2522_252263


namespace NUMINAMATH_CALUDE_sqrt_equation_sum_l2522_252271

theorem sqrt_equation_sum (y : ℝ) (d e f : ℕ+) : 
  y = Real.sqrt ((Real.sqrt 73 / 3) + (5 / 3)) →
  y^52 = 3*y^50 + 10*y^48 + 25*y^46 - y^26 + d*y^22 + e*y^20 + f*y^18 →
  d + e + f = 184 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_sum_l2522_252271


namespace NUMINAMATH_CALUDE_oranges_remaining_l2522_252299

/-- The number of oranges Michaela needs to get full -/
def michaela_oranges : ℕ := 20

/-- The number of oranges Cassandra needs to get full -/
def cassandra_oranges : ℕ := 2 * michaela_oranges

/-- The total number of oranges picked from the farm -/
def total_oranges : ℕ := 90

/-- The number of oranges remaining after Michaela and Cassandra have eaten until they are full -/
theorem oranges_remaining : total_oranges - (michaela_oranges + cassandra_oranges) = 30 := by
  sorry

end NUMINAMATH_CALUDE_oranges_remaining_l2522_252299


namespace NUMINAMATH_CALUDE_equation_solution_l2522_252281

theorem equation_solution : ∃ x : ℚ, (2/7) * (1/8) * x - 4 = 12 ∧ x = 448 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2522_252281


namespace NUMINAMATH_CALUDE_hyperbola_sum_l2522_252258

-- Define the hyperbola parameters
def center : ℝ × ℝ := (3, -2)
def focus : ℝ × ℝ := (3, 5)
def vertex : ℝ × ℝ := (3, 0)

-- Define h and k from the center
def h : ℝ := center.1
def k : ℝ := center.2

-- Define a as the distance from center to vertex
def a : ℝ := |center.2 - vertex.2|

-- Define c as the distance from center to focus
def c : ℝ := |center.2 - focus.2|

-- Define b using the relationship c^2 = a^2 + b^2
noncomputable def b : ℝ := Real.sqrt (c^2 - a^2)

-- Theorem statement
theorem hyperbola_sum : h + k + a + b = 3 + 3 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l2522_252258


namespace NUMINAMATH_CALUDE_smallest_a_for_equation_l2522_252225

theorem smallest_a_for_equation : ∃ (p : ℕ) (b : ℕ), 
  Nat.Prime p ∧ 
  b ≥ 2 ∧ 
  (9^p - 9) / p = b^2 ∧ 
  ∀ (a : ℕ) (q : ℕ) (c : ℕ), 
    a > 0 ∧ a < 9 → 
    Nat.Prime q → 
    c ≥ 2 → 
    (a^q - a) / q ≠ c^2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_for_equation_l2522_252225


namespace NUMINAMATH_CALUDE_expression_evaluation_l2522_252277

theorem expression_evaluation : 
  let mixed_number : ℚ := 20 + 94 / 95
  let expression := (mixed_number * 1.65 - mixed_number + 7 / 20 * mixed_number) * 47.5 * 0.8 * 2.5
  expression = 1994 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2522_252277


namespace NUMINAMATH_CALUDE_cos_330_degrees_l2522_252228

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_degrees_l2522_252228


namespace NUMINAMATH_CALUDE_max_true_statements_l2522_252287

theorem max_true_statements (c d : ℝ) : 
  let statements := [
    (1 / c > 1 / d),
    (c^2 < d^2),
    (c > d),
    (c > 0),
    (d > 0)
  ]
  ∃ (trueStatements : Finset (Fin 5)), 
    (∀ i ∈ trueStatements, statements[i] = true) ∧ 
    trueStatements.card ≤ 3 ∧
    ∀ (otherStatements : Finset (Fin 5)), 
      (∀ i ∈ otherStatements, statements[i] = true) →
      otherStatements.card ≤ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_true_statements_l2522_252287


namespace NUMINAMATH_CALUDE_geometric_progression_existence_l2522_252244

theorem geometric_progression_existence : ∃ (a : ℕ → ℚ), 
  (∀ n, a (n + 1) = a n * (3/2)) ∧ 
  (a 1 = 2^99) ∧
  (∀ n, a (n + 1) > a n) ∧
  (∀ n ≤ 100, ∃ m : ℕ, a n = m) ∧
  (∀ n > 100, ∀ m : ℕ, a n ≠ m) := by
  sorry

#check geometric_progression_existence

end NUMINAMATH_CALUDE_geometric_progression_existence_l2522_252244


namespace NUMINAMATH_CALUDE_adjacent_same_face_exists_l2522_252218

/-- Represents a coin, which can be either heads or tails -/
inductive Coin
| Heads
| Tails

/-- Represents a circular arrangement of 11 coins -/
def CoinArrangement := Fin 11 → Coin

/-- Two positions in the circle are adjacent if they differ by 1 modulo 11 -/
def adjacent (i j : Fin 11) : Prop :=
  (i.val + 1) % 11 = j.val ∨ (j.val + 1) % 11 = i.val

/-- Main theorem: In any arrangement of 11 coins, there exists a pair of adjacent coins showing the same face -/
theorem adjacent_same_face_exists (arrangement : CoinArrangement) :
  ∃ (i j : Fin 11), adjacent i j ∧ arrangement i = arrangement j := by
  sorry

end NUMINAMATH_CALUDE_adjacent_same_face_exists_l2522_252218


namespace NUMINAMATH_CALUDE_integer_solutions_count_l2522_252252

theorem integer_solutions_count : 
  ∃! (S : Finset ℤ), 
    (∀ a ∈ S, ∃ x : ℤ, x^2 + a*x - 6*a = 0) ∧ 
    (∀ a : ℤ, (∃ x : ℤ, x^2 + a*x - 6*a = 0) → a ∈ S) ∧
    S.card = 9 :=
by sorry

end NUMINAMATH_CALUDE_integer_solutions_count_l2522_252252


namespace NUMINAMATH_CALUDE_correct_operation_l2522_252280

theorem correct_operation (m : ℝ) : (-m + 2) * (-m - 2) = m^2 - 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l2522_252280


namespace NUMINAMATH_CALUDE_equation_solutions_l2522_252239

theorem equation_solutions :
  (∀ x, x * (x - 1) - 3 * (x - 1) = 0 ↔ x = 1 ∨ x = 3) ∧
  (∀ x, x^2 + 2*x - 1 = 0 ↔ x = -1 + Real.sqrt 2 ∨ x = -1 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2522_252239


namespace NUMINAMATH_CALUDE_product_and_sum_of_factors_l2522_252290

theorem product_and_sum_of_factors : ∃ (a b : ℕ), 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 1540 ∧ 
  a + b = 97 := by
  sorry

end NUMINAMATH_CALUDE_product_and_sum_of_factors_l2522_252290


namespace NUMINAMATH_CALUDE_max_modest_number_l2522_252213

def is_modest_number (a b c d : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
  a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9 ∧ d ≤ 9 ∧
  5 * a = b + c + d

def G (a b c d : ℕ) : ℤ :=
  10 * a + b - 10 * c - d

theorem max_modest_number :
  ∀ (a b c d : ℕ),
    is_modest_number a b c d →
    d % 2 = 0 →
    (G a b c d) % 11 = 0 →
    (a + b + c) % 3 = 0 →
    a * 1000 + b * 100 + c * 10 + d ≤ 3816 :=
by sorry

end NUMINAMATH_CALUDE_max_modest_number_l2522_252213


namespace NUMINAMATH_CALUDE_min_minutes_for_cheaper_plan_b_l2522_252278

/-- Represents the cost of a phone plan in cents -/
def PlanCost := ℕ → ℕ

/-- Cost function for Plan A: 10 cents per minute -/
def planA : PlanCost := λ minutes => 10 * minutes

/-- Cost function for Plan B: $20 flat fee (2000 cents) plus 5 cents per minute -/
def planB : PlanCost := λ minutes => 2000 + 5 * minutes

/-- Theorem stating that 401 is the minimum number of minutes for Plan B to be cheaper -/
theorem min_minutes_for_cheaper_plan_b : 
  (∀ m : ℕ, m < 401 → planA m ≤ planB m) ∧ 
  (∀ m : ℕ, m ≥ 401 → planB m < planA m) := by
  sorry

end NUMINAMATH_CALUDE_min_minutes_for_cheaper_plan_b_l2522_252278


namespace NUMINAMATH_CALUDE_circle_symmetry_l2522_252254

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the property of symmetry
def is_symmetric (circle1 circle2 : (ℝ → ℝ → Prop)) (line : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), 
    circle1 x1 y1 ∧ 
    circle2 x2 y2 ∧ 
    line ((x1 + x2) / 2) ((y1 + y2) / 2)

-- Define our target circle
def target_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

-- The main theorem
theorem circle_symmetry :
  is_symmetric given_circle target_circle symmetry_line :=
sorry

end NUMINAMATH_CALUDE_circle_symmetry_l2522_252254


namespace NUMINAMATH_CALUDE_tangent_line_sum_l2522_252223

/-- Given a function f: ℝ → ℝ with a tangent line at x=1 described by the equation 3x+y-4=0,
    prove that f(1) + f'(1) = -2 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h_tangent : ∀ x y : ℝ, y = f x → (3 * 1 + f 1 - 4 = 0 ∧ 3 * x + y - 4 = 0)) : 
    f 1 + (deriv f) 1 = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l2522_252223


namespace NUMINAMATH_CALUDE_max_ab_value_l2522_252243

theorem max_ab_value (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃! x, x^2 + Real.sqrt a * x - b + 1/4 = 0) → 
  ∀ c, a * b ≤ c → c ≤ 1/16 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l2522_252243


namespace NUMINAMATH_CALUDE_min_value_of_product_l2522_252247

theorem min_value_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 47 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_product_l2522_252247


namespace NUMINAMATH_CALUDE_average_age_calculation_l2522_252207

/-- The average age of a group of fifth-graders, their parents, and teachers -/
theorem average_age_calculation (n_students : ℕ) (n_parents : ℕ) (n_teachers : ℕ)
  (avg_age_students : ℚ) (avg_age_parents : ℚ) (avg_age_teachers : ℚ)
  (h_students : n_students = 30)
  (h_parents : n_parents = 50)
  (h_teachers : n_teachers = 10)
  (h_avg_students : avg_age_students = 10)
  (h_avg_parents : avg_age_parents = 40)
  (h_avg_teachers : avg_age_teachers = 35) :
  (n_students * avg_age_students + n_parents * avg_age_parents + n_teachers * avg_age_teachers) /
  (n_students + n_parents + n_teachers : ℚ) = 530 / 18 :=
by sorry

end NUMINAMATH_CALUDE_average_age_calculation_l2522_252207
