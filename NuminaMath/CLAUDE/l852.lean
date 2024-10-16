import Mathlib

namespace NUMINAMATH_CALUDE_amount_spent_on_sweets_l852_85200

def initial_amount : ℚ := 10.50
def amount_per_friend : ℚ := 3.40
def number_of_friends : ℕ := 2

theorem amount_spent_on_sweets :
  initial_amount - (amount_per_friend * number_of_friends) = 3.70 := by
  sorry

end NUMINAMATH_CALUDE_amount_spent_on_sweets_l852_85200


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l852_85294

/-- The coefficient of the third term in the binomial expansion of (a + √x)^5 -/
def third_term_coefficient (a : ℝ) (x : ℝ) : ℝ := 10 * a^3 * x

/-- Theorem: If the coefficient of the third term in (a + √x)^5 is 80, then a = 2 -/
theorem binomial_expansion_coefficient (a : ℝ) (x : ℝ) :
  third_term_coefficient a x = 80 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l852_85294


namespace NUMINAMATH_CALUDE_simplify_fraction_l852_85212

theorem simplify_fraction : (160 : ℚ) / 2880 * 40 = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l852_85212


namespace NUMINAMATH_CALUDE_internet_rate_proof_l852_85281

/-- The regular monthly internet rate without discount -/
def regular_rate : ℝ := 50

/-- The discounted rate as a fraction of the regular rate -/
def discount_rate : ℝ := 0.95

/-- The number of months -/
def num_months : ℕ := 4

/-- The total payment for the given number of months -/
def total_payment : ℝ := 190

theorem internet_rate_proof : 
  regular_rate * discount_rate * num_months = total_payment := by
  sorry

#check internet_rate_proof

end NUMINAMATH_CALUDE_internet_rate_proof_l852_85281


namespace NUMINAMATH_CALUDE_calculation_proof_l852_85291

theorem calculation_proof (a b c : ℝ) (h1 : a = 3) (h2 : b = 2) (h3 : c = 4) :
  (c * (a^3 + b^3)) / (a^2 - a*b + b^2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l852_85291


namespace NUMINAMATH_CALUDE_poem_distribution_theorem_l852_85297

def distribute_poems (n : ℕ) (k : ℕ) (min_poems : ℕ) : ℕ :=
  let case1 := (n.choose 2) * ((n - 2).choose 2) * 3
  let case2 := (n.choose 2) * ((n - 2).choose 3) * 3
  case1 + case2

theorem poem_distribution_theorem :
  distribute_poems 8 3 2 = 2940 := by
  sorry

end NUMINAMATH_CALUDE_poem_distribution_theorem_l852_85297


namespace NUMINAMATH_CALUDE_division_problem_l852_85208

theorem division_problem (A : ℕ) (h : 23 = A * 3 + 2) : A = 7 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l852_85208


namespace NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l852_85240

theorem consecutive_product_plus_one_is_square (n : ℤ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_plus_one_is_square_l852_85240


namespace NUMINAMATH_CALUDE_prob_all_odd_is_one_42_l852_85228

/-- The number of slips in the hat -/
def total_slips : ℕ := 10

/-- The number of odd-numbered slips in the hat -/
def odd_slips : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability of drawing all odd-numbered slips -/
def prob_all_odd : ℚ := (odd_slips : ℚ) / total_slips *
                        (odd_slips - 1) / (total_slips - 1) *
                        (odd_slips - 2) / (total_slips - 2) *
                        (odd_slips - 3) / (total_slips - 3)

theorem prob_all_odd_is_one_42 : prob_all_odd = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_prob_all_odd_is_one_42_l852_85228


namespace NUMINAMATH_CALUDE_unknown_number_proof_l852_85279

theorem unknown_number_proof (x : ℝ) : x - (1002 / 200.4) = 3029 → x = 3034 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l852_85279


namespace NUMINAMATH_CALUDE_equation_solutions_l852_85254

theorem equation_solutions :
  ∀ x y : ℕ, 1 + 3^x = 2^y ↔ (x = 0 ∧ y = 1) ∨ (x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l852_85254


namespace NUMINAMATH_CALUDE_limit_at_one_l852_85239

def f (x : ℝ) : ℝ := x^2

theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ ∀ Δx : ℝ, 0 < |Δx| ∧ |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 2| < ε :=
  sorry

end NUMINAMATH_CALUDE_limit_at_one_l852_85239


namespace NUMINAMATH_CALUDE_ann_keeps_36_cookies_l852_85288

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of oatmeal raisin cookies Ann bakes -/
def oatmeal_baked : ℕ := 3 * dozen

/-- The number of sugar cookies Ann bakes -/
def sugar_baked : ℕ := 2 * dozen

/-- The number of chocolate chip cookies Ann bakes -/
def chocolate_baked : ℕ := 4 * dozen

/-- The number of oatmeal raisin cookies Ann gives away -/
def oatmeal_given : ℕ := 2 * dozen

/-- The number of sugar cookies Ann gives away -/
def sugar_given : ℕ := (3 * dozen) / 2

/-- The number of chocolate chip cookies Ann gives away -/
def chocolate_given : ℕ := (5 * dozen) / 2

/-- The total number of cookies Ann keeps -/
def total_kept : ℕ := (oatmeal_baked - oatmeal_given) + (sugar_baked - sugar_given) + (chocolate_baked - chocolate_given)

theorem ann_keeps_36_cookies : total_kept = 36 := by
  sorry

end NUMINAMATH_CALUDE_ann_keeps_36_cookies_l852_85288


namespace NUMINAMATH_CALUDE_range_of_f_l852_85280

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x + 2)

-- State the theorem
theorem range_of_f :
  Set.range f = {y : ℝ | y < 21 ∨ y > 21} :=
by sorry

end NUMINAMATH_CALUDE_range_of_f_l852_85280


namespace NUMINAMATH_CALUDE_max_value_of_sum_product_l852_85237

theorem max_value_of_sum_product (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → 
  a + b + c + d = 200 → 
  a * b + a * c + a * d ≤ 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_sum_product_l852_85237


namespace NUMINAMATH_CALUDE_group_size_from_circular_arrangements_l852_85218

/-- The number of ways to arrange k people from a group of n people around a circular table. -/
def circularArrangements (n k : ℕ) : ℕ := Nat.factorial (n - 1)

/-- Theorem: If there are 144 ways to seat 5 people around a circular table from a group of n people, then n = 7. -/
theorem group_size_from_circular_arrangements (n : ℕ) 
  (h : circularArrangements n 5 = 144) : n = 7 := by
  sorry

end NUMINAMATH_CALUDE_group_size_from_circular_arrangements_l852_85218


namespace NUMINAMATH_CALUDE_regular_polygon_interior_angle_l852_85214

theorem regular_polygon_interior_angle (n : ℕ) (h : n > 2) :
  (n - 2) * 180 / n = 140 → n = 9 := by sorry

end NUMINAMATH_CALUDE_regular_polygon_interior_angle_l852_85214


namespace NUMINAMATH_CALUDE_field_trip_adults_l852_85290

/-- The number of adults going on a field trip --/
theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) : 
  van_capacity = 7 → num_students = 33 → num_vans = 6 → 
  (num_vans * van_capacity) - num_students = 9 := by
  sorry

end NUMINAMATH_CALUDE_field_trip_adults_l852_85290


namespace NUMINAMATH_CALUDE_investment_amount_l852_85238

/-- Represents the investment scenario with changing interest rates and inflation --/
structure Investment where
  principal : ℝ
  baseRate : ℝ
  years : ℕ
  rateChangeYear2 : ℝ
  rateChangeYear4 : ℝ
  inflationRate : ℝ
  interestDifference : ℝ

/-- Calculates the total interest earned with rate changes --/
def totalInterestWithChanges (inv : Investment) : ℝ :=
  inv.principal * (5 * inv.baseRate + inv.rateChangeYear2 + inv.rateChangeYear4)

/-- Calculates the total interest earned without rate changes --/
def totalInterestWithoutChanges (inv : Investment) : ℝ :=
  inv.principal * 5 * inv.baseRate

/-- Theorem stating that the original investment amount is $30,000 --/
theorem investment_amount (inv : Investment) 
  (h1 : inv.years = 5)
  (h2 : inv.rateChangeYear2 = 0.005)
  (h3 : inv.rateChangeYear4 = 0.01)
  (h4 : inv.inflationRate = 0.01)
  (h5 : totalInterestWithChanges inv - totalInterestWithoutChanges inv = inv.interestDifference)
  (h6 : inv.interestDifference = 450) :
  inv.principal = 30000 := by
  sorry

#check investment_amount

end NUMINAMATH_CALUDE_investment_amount_l852_85238


namespace NUMINAMATH_CALUDE_courtyard_length_l852_85256

/-- The length of a rectangular courtyard given its width and paving stone information. -/
theorem courtyard_length (width : ℚ) (num_stones : ℕ) (stone_length stone_width : ℚ) :
  width = 33 / 2 →
  num_stones = 132 →
  stone_length = 5 / 2 →
  stone_width = 2 →
  (num_stones * stone_length * stone_width) / width = 40 :=
by sorry

end NUMINAMATH_CALUDE_courtyard_length_l852_85256


namespace NUMINAMATH_CALUDE_daisy_count_l852_85230

def white_daisies : ℕ := 6

def pink_daisies : ℕ := 9 * white_daisies

def red_daisies : ℕ := 4 * pink_daisies - 3

def total_daisies : ℕ := white_daisies + pink_daisies + red_daisies

theorem daisy_count : total_daisies = 273 := by
  sorry

end NUMINAMATH_CALUDE_daisy_count_l852_85230


namespace NUMINAMATH_CALUDE_cubic_polynomial_third_root_l852_85216

theorem cubic_polynomial_third_root 
  (a b : ℚ) 
  (h1 : a * (-1)^3 + (a + 3*b) * (-1)^2 + (b - 2*a) * (-1) + (10 - a) = 0)
  (h2 : a * 4^3 + (a + 3*b) * 4^2 + (b - 2*a) * 4 + (10 - a) = 0) :
  ∃ (r : ℚ), a * r^3 + (a + 3*b) * r^2 + (b - 2*a) * r + (10 - a) = 0 ∧ 
              r = -67/88 :=
by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_third_root_l852_85216


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l852_85206

theorem least_subtrahend_for_divisibility (n m : ℕ) : 
  ∃ (x : ℕ), x = n % m ∧ 
  (∀ (y : ℕ), (n - y) % m = 0 → y ≥ x) ∧
  (n - x) % m = 0 :=
sorry

#check least_subtrahend_for_divisibility 13602 87

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_l852_85206


namespace NUMINAMATH_CALUDE_anna_coins_l852_85266

/-- Represents the number of different coin values that can be obtained -/
def different_values (five_cent : ℕ) (twenty_cent : ℕ) : ℕ :=
  59 - 3 * five_cent

theorem anna_coins :
  ∀ (five_cent twenty_cent : ℕ),
    five_cent + twenty_cent = 15 →
    different_values five_cent twenty_cent = 24 →
    twenty_cent = 4 := by
  sorry

end NUMINAMATH_CALUDE_anna_coins_l852_85266


namespace NUMINAMATH_CALUDE_smallest_three_digit_palindrome_not_six_digit_palindrome_product_l852_85233

/-- Checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- Checks if a number is a six-digit palindrome -/
def isSixDigitPalindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ (n / 100000 = n % 10) ∧ ((n / 10000) % 10 = (n / 10) % 10) ∧ ((n / 1000) % 10 = (n / 100) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_not_six_digit_palindrome_product :
  isThreeDigitPalindrome 404 ∧
  ¬(isSixDigitPalindrome (404 * 102)) ∧
  ∀ n : ℕ, isThreeDigitPalindrome n → n < 404 → isSixDigitPalindrome (n * 102) :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_palindrome_not_six_digit_palindrome_product_l852_85233


namespace NUMINAMATH_CALUDE_log_inequality_l852_85249

theorem log_inequality (x y : ℝ) (h : Real.log x < Real.log y ∧ Real.log y < 0) : 
  0 < x ∧ x < y ∧ y < 1 := by sorry

end NUMINAMATH_CALUDE_log_inequality_l852_85249


namespace NUMINAMATH_CALUDE_three_number_problem_l852_85277

theorem three_number_problem :
  ∃ (X Y Z : ℤ),
    (X = (35 * X) / 100 + 60) ∧
    (X = (70 * Y) / 200 + Y / 2) ∧
    (Y = 2 * Z^2) ∧
    (X = 92) ∧
    (Y = 108) ∧
    (Z = 7) := by
  sorry

end NUMINAMATH_CALUDE_three_number_problem_l852_85277


namespace NUMINAMATH_CALUDE_remainder_cube_l852_85243

theorem remainder_cube (n : ℤ) : n % 13 = 5 → n^3 % 17 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_cube_l852_85243


namespace NUMINAMATH_CALUDE_linda_car_rental_cost_l852_85262

/-- Calculates the total cost of renting a car given the daily rate, mileage rate, number of days, and miles driven. -/
def total_rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that the total cost for Linda's car rental is $165. -/
theorem linda_car_rental_cost :
  total_rental_cost 30 0.25 3 300 = 165 := by
  sorry

end NUMINAMATH_CALUDE_linda_car_rental_cost_l852_85262


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_one_range_of_m_l852_85278

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - |x - 1|

-- Theorem for the solution set of f(x) > 1
theorem solution_set_f_greater_than_one :
  {x : ℝ | f x > 1} = {x : ℝ | x > 0} := by sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∃ x, f x + 4 ≥ |1 - 2*m|} = {m : ℝ | -6 ≤ m ∧ m ≤ 8} := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_one_range_of_m_l852_85278


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l852_85276

theorem sufficient_not_necessary_condition (x y : ℝ) :
  (∀ x y, x > y ∧ y > 0 → x / y > 1) ∧
  (∃ x y, x / y > 1 ∧ ¬(x > y ∧ y > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l852_85276


namespace NUMINAMATH_CALUDE_binomial_sum_l852_85271

theorem binomial_sum : Nat.choose 12 4 + Nat.choose 10 3 = 615 := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_l852_85271


namespace NUMINAMATH_CALUDE_daily_savings_amount_l852_85217

def total_savings : ℝ := 8760
def days_in_year : ℕ := 365

theorem daily_savings_amount :
  total_savings / days_in_year = 24 := by
  sorry

end NUMINAMATH_CALUDE_daily_savings_amount_l852_85217


namespace NUMINAMATH_CALUDE_base_sum_equals_55_base_7_l852_85242

/-- Represents a number in a given base --/
def BaseNumber (base : ℕ) := ℕ

/-- Converts a base number to its decimal representation --/
def to_decimal (base : ℕ) (n : BaseNumber base) : ℕ := sorry

/-- Converts a decimal number to its representation in a given base --/
def from_decimal (base : ℕ) (n : ℕ) : BaseNumber base := sorry

/-- Multiplies two numbers in a given base --/
def base_mul (base : ℕ) (a b : BaseNumber base) : BaseNumber base := sorry

/-- Adds two numbers in a given base --/
def base_add (base : ℕ) (a b : BaseNumber base) : BaseNumber base := sorry

theorem base_sum_equals_55_base_7 (c : ℕ) 
  (h : base_mul c (base_mul c (from_decimal c 14) (from_decimal c 18)) (from_decimal c 17) = from_decimal c 4185) :
  base_add c (base_add c (from_decimal c 14) (from_decimal c 18)) (from_decimal c 17) = from_decimal 7 55 := 
sorry

end NUMINAMATH_CALUDE_base_sum_equals_55_base_7_l852_85242


namespace NUMINAMATH_CALUDE_white_balls_count_prob_after_addition_l852_85251

/-- The total number of balls in the box -/
def total_balls : ℕ := 40

/-- The probability of picking a white ball -/
def prob_white : ℚ := 1/10 * 6

/-- The number of white balls in the box -/
def white_balls : ℕ := 24

/-- The number of additional balls added -/
def additional_balls : ℕ := 10

/-- Theorem stating the relationship between the number of white balls and the probability -/
theorem white_balls_count : white_balls = total_balls * prob_white := by sorry

/-- Theorem proving that adding 10 balls with 1 white results in 50% probability -/
theorem prob_after_addition : 
  (white_balls + 1) / (total_balls + additional_balls) = 1/2 := by sorry

end NUMINAMATH_CALUDE_white_balls_count_prob_after_addition_l852_85251


namespace NUMINAMATH_CALUDE_distribution_five_balls_three_boxes_l852_85236

/-- Represents the number of ways to distribute indistinguishable balls into boxes -/
def distributionWays (totalBalls : ℕ) (totalBoxes : ℕ) (indistinguishableBoxes : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 12 ways to distribute 5 indistinguishable balls
    into 3 boxes where 2 boxes are indistinguishable and 1 box is distinguishable -/
theorem distribution_five_balls_three_boxes :
  distributionWays 5 3 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_distribution_five_balls_three_boxes_l852_85236


namespace NUMINAMATH_CALUDE_least_six_digit_congruent_to_seven_mod_seventeen_l852_85223

theorem least_six_digit_congruent_to_seven_mod_seventeen :
  ∃ (n : ℕ), 
    n = 100008 ∧ 
    n ≥ 100000 ∧ 
    n < 1000000 ∧
    n % 17 = 7 ∧
    ∀ (m : ℕ), m ≥ 100000 ∧ m < 1000000 ∧ m % 17 = 7 → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_least_six_digit_congruent_to_seven_mod_seventeen_l852_85223


namespace NUMINAMATH_CALUDE_alcohol_mixture_proof_l852_85247

/-- Proves that mixing 300 mL of 10% alcohol solution with 450 mL of 30% alcohol solution results in a 22% alcohol solution -/
theorem alcohol_mixture_proof :
  let x_volume : ℝ := 300
  let y_volume : ℝ := 450
  let x_concentration : ℝ := 0.10
  let y_concentration : ℝ := 0.30
  let target_concentration : ℝ := 0.22
  
  x_volume * x_concentration + y_volume * y_concentration = 
    (x_volume + y_volume) * target_concentration :=
by
  sorry

#check alcohol_mixture_proof

end NUMINAMATH_CALUDE_alcohol_mixture_proof_l852_85247


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l852_85260

/-- The maximum distance from a point on the circle ρ = 8sinθ to the line θ = π/3 is 6 -/
theorem max_distance_circle_to_line :
  let circle := {p : ℝ × ℝ | p.1^2 + (p.2 - 4)^2 = 16}
  let line := {p : ℝ × ℝ | p.2 = Real.sqrt 3 * p.1}
  ∃ (max_dist : ℝ), max_dist = 6 ∧
    ∀ (p : ℝ × ℝ), p ∈ circle →
      ∀ (q : ℝ × ℝ), q ∈ line →
        Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l852_85260


namespace NUMINAMATH_CALUDE_oak_grove_library_books_l852_85211

theorem oak_grove_library_books :
  let public_library : ℝ := 1986
  let school_libraries : ℝ := 5106
  let community_college_library : ℝ := 3294.5
  let medical_library : ℝ := 1342.25
  let law_library : ℝ := 2785.75
  public_library + school_libraries + community_college_library + medical_library + law_library = 15514.5 := by
  sorry

end NUMINAMATH_CALUDE_oak_grove_library_books_l852_85211


namespace NUMINAMATH_CALUDE_sum_of_alternate_angles_less_than_450_l852_85275

-- Define a heptagon
structure Heptagon where
  vertices : Fin 7 → ℝ × ℝ

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the property of a heptagon being inscribed in a circle
def is_inscribed (h : Heptagon) (c : Circle) : Prop :=
  ∀ i : Fin 7, dist c.center (h.vertices i) = c.radius

-- Define the property of a point being inside a polygon
def is_inside (p : ℝ × ℝ) (h : Heptagon) : Prop :=
  sorry -- Definition of a point being inside a polygon

-- Define the angle at a vertex of the heptagon
def angle_at_vertex (h : Heptagon) (i : Fin 7) : ℝ :=
  sorry -- Definition of angle at a vertex

-- Theorem statement
theorem sum_of_alternate_angles_less_than_450 (h : Heptagon) (c : Circle) :
  is_inscribed h c → is_inside c.center h →
  angle_at_vertex h 0 + angle_at_vertex h 2 + angle_at_vertex h 4 < 450 :=
sorry

end NUMINAMATH_CALUDE_sum_of_alternate_angles_less_than_450_l852_85275


namespace NUMINAMATH_CALUDE_green_toads_per_acre_l852_85268

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

end NUMINAMATH_CALUDE_green_toads_per_acre_l852_85268


namespace NUMINAMATH_CALUDE_range_of_m_l852_85219

theorem range_of_m (p : ℝ → Prop) (m : ℝ) 
  (h1 : ∀ x, p x ↔ x^2 + 2*x - m > 0)
  (h2 : ¬ p 1)
  (h3 : p 2) :
  3 ≤ m ∧ m < 8 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l852_85219


namespace NUMINAMATH_CALUDE_simplify_expression_l852_85202

theorem simplify_expression :
  ∃ (a b c : ℕ+),
    (((Real.sqrt 3 - 1) ^ (2 - Real.sqrt 5)) / ((Real.sqrt 3 + 1) ^ (2 + Real.sqrt 5)) =
     (1 - (1/2) * Real.sqrt 3) * (2 ^ (-Real.sqrt 5))) ∧
    (∀ (p : ℕ), Nat.Prime p → ¬(p^2 ∣ c.val)) ∧
    ((1 - (1/2) * Real.sqrt 3) * (2 ^ (-Real.sqrt 5)) = a - b * Real.sqrt c) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l852_85202


namespace NUMINAMATH_CALUDE_complex_multiplication_l852_85210

theorem complex_multiplication (z : ℂ) (h : z + 1 = 2 + I) : z * (1 - I) = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_multiplication_l852_85210


namespace NUMINAMATH_CALUDE_sum_of_reciprocal_divisors_eq_two_l852_85225

def sum_of_divisors (n : ℕ) : ℚ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => x)

def sum_of_reciprocal_divisors (n : ℕ) : ℚ :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum (λ x => 1 / x)

theorem sum_of_reciprocal_divisors_eq_two (n : ℕ) (h : sum_of_divisors n = 2 * n) :
  sum_of_reciprocal_divisors n = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocal_divisors_eq_two_l852_85225


namespace NUMINAMATH_CALUDE_largest_seventh_term_coefficient_l852_85244

/-- 
Given that in the expansion of (x + y)^n the coefficient of the seventh term is the largest,
this theorem states that n must be either 11, 12, or 13.
-/
theorem largest_seventh_term_coefficient (n : ℕ) : 
  (∀ k : ℕ, k ≠ 6 → (n.choose k) ≤ (n.choose 6)) → 
  n = 11 ∨ n = 12 ∨ n = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_seventh_term_coefficient_l852_85244


namespace NUMINAMATH_CALUDE_optimal_prevention_plan_l852_85272

/-- Represents the cost and effectiveness of a preventive measure -/
structure PreventiveMeasure where
  cost : ℝ
  effectiveness : ℝ

/-- Calculates the total cost given preventive measures and event parameters -/
def totalCost (measures : List PreventiveMeasure) (eventProbability : ℝ) (eventLoss : ℝ) : ℝ :=
  (measures.map (·.cost)).sum + eventLoss * (1 - (measures.map (·.effectiveness)).prod)

theorem optimal_prevention_plan (eventProbability : ℝ) (eventLoss : ℝ)
  (measureA : PreventiveMeasure) (measureB : PreventiveMeasure) :
  eventProbability = 0.3 →
  eventLoss = 4 →
  measureA.cost = 0.45 →
  measureB.cost = 0.3 →
  measureA.effectiveness = 0.9 →
  measureB.effectiveness = 0.85 →
  totalCost [measureA, measureB] eventProbability eventLoss <
    min (totalCost [] eventProbability eventLoss)
      (min (totalCost [measureA] eventProbability eventLoss)
        (totalCost [measureB] eventProbability eventLoss)) :=
by
  sorry

end NUMINAMATH_CALUDE_optimal_prevention_plan_l852_85272


namespace NUMINAMATH_CALUDE_shoe_pairs_in_box_l852_85215

theorem shoe_pairs_in_box (total_shoes : ℕ) (prob_matching : ℚ) : 
  total_shoes = 200 →
  prob_matching = 1 / 199 →
  (total_shoes / 2 : ℕ) = 100 :=
by sorry

end NUMINAMATH_CALUDE_shoe_pairs_in_box_l852_85215


namespace NUMINAMATH_CALUDE_students_not_enrolled_l852_85269

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ)
  (h1 : total = 94)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 40 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l852_85269


namespace NUMINAMATH_CALUDE_multiplication_division_sum_l852_85282

theorem multiplication_division_sum : 4 * 6 * 8 + 24 / 4 = 198 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_division_sum_l852_85282


namespace NUMINAMATH_CALUDE_triangle_side_and_area_l852_85205

/-- Given a triangle ABC with side lengths a and c, and angle B, 
    prove the length of side b and the area of the triangle. -/
theorem triangle_side_and_area 
  (a c : ℝ) 
  (B : ℝ) 
  (ha : a = 3 * Real.sqrt 3) 
  (hc : c = 2) 
  (hB : B = 150 * π / 180) : 
  ∃ (b S : ℝ), 
    b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) ∧ 
    b = 7 ∧
    S = (1/2) * a * c * Real.sin B ∧ 
    S = (3/2) * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_triangle_side_and_area_l852_85205


namespace NUMINAMATH_CALUDE_final_probability_l852_85295

/-- Represents the number of operations performed -/
def num_operations : ℕ := 5

/-- Represents the initial number of red balls -/
def initial_red : ℕ := 2

/-- Represents the initial number of blue balls -/
def initial_blue : ℕ := 1

/-- Represents the final number of red balls -/
def final_red : ℕ := 4

/-- Represents the final number of blue balls -/
def final_blue : ℕ := 4

/-- Calculates the probability of drawing a specific sequence of balls -/
def sequence_probability (red_draws blue_draws : ℕ) : ℚ := sorry

/-- Calculates the number of possible sequences -/
def num_sequences : ℕ := sorry

/-- The main theorem stating the probability of the final outcome -/
theorem final_probability : 
  sequence_probability (final_red - initial_red) (final_blue - initial_blue) * num_sequences = 2/7 := by sorry

end NUMINAMATH_CALUDE_final_probability_l852_85295


namespace NUMINAMATH_CALUDE_difference_of_squares_625_375_l852_85258

theorem difference_of_squares_625_375 : 625^2 - 375^2 = 250000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_625_375_l852_85258


namespace NUMINAMATH_CALUDE_tom_apple_problem_l852_85255

theorem tom_apple_problem (num_apples : ℕ) : 
  let total_slices := num_apples * 8
  let remaining_after_jerry := total_slices * (5/8 : ℚ)
  let remaining_after_eating := remaining_after_jerry * (1/2 : ℚ)
  remaining_after_eating = 5 →
  num_apples = 2 := by
sorry

end NUMINAMATH_CALUDE_tom_apple_problem_l852_85255


namespace NUMINAMATH_CALUDE_determinant_evaluation_l852_85241

-- Define the matrix
def matrix (x y z : ℝ) : Matrix (Fin 4) (Fin 4) ℝ :=
λ i j => match i, j with
  | 0, 0 => 1
  | 0, 1 => x
  | 0, 2 => y
  | 0, 3 => z
  | 1, 0 => 1
  | 1, 1 => x + y
  | 1, 2 => y
  | 1, 3 => z
  | 2, 0 => 1
  | 2, 1 => x
  | 2, 2 => x + y
  | 2, 3 => z
  | 3, 0 => 1
  | 3, 1 => x
  | 3, 2 => y
  | 3, 3 => x + y + z

theorem determinant_evaluation (x y z : ℝ) :
  Matrix.det (matrix x y z) = y * x^2 + y^2 * x := by
  sorry

end NUMINAMATH_CALUDE_determinant_evaluation_l852_85241


namespace NUMINAMATH_CALUDE_new_year_cards_profit_l852_85274

/-- The profit calculation for a store selling New Year cards -/
theorem new_year_cards_profit
  (purchase_price : ℕ)
  (total_sale : ℕ)
  (h1 : purchase_price = 21)
  (h2 : total_sale = 1457)
  (h3 : ∃ (n : ℕ) (selling_price : ℕ), n * selling_price = total_sale ∧ selling_price ≤ 2 * purchase_price) :
  ∃ (n : ℕ) (selling_price : ℕ), 
    n * selling_price = total_sale ∧ 
    selling_price ≤ 2 * purchase_price ∧
    n * (selling_price - purchase_price) = 470 :=
by sorry


end NUMINAMATH_CALUDE_new_year_cards_profit_l852_85274


namespace NUMINAMATH_CALUDE_tangent_perpendicular_points_l852_85270

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_perpendicular_points :
  ∀ x y : ℝ, f x = y →
    (3 * x^2 + 1 = 4 ∨ 3 * x^2 + 1 = -1/4) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_points_l852_85270


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l852_85292

theorem parallel_line_through_point (x y : ℝ) : 
  (3 * x - 2 * y - 11 = 0) ↔ 
  (∃ (m : ℝ), y = (3/2) * x + m ∧ -1 = (3/2) * 3 + m) := by
sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l852_85292


namespace NUMINAMATH_CALUDE_estimate_fish_population_l852_85263

/-- Estimates the total number of fish in a pond using the capture-recapture method. -/
theorem estimate_fish_population (initial_catch : ℕ) (second_catch : ℕ) (marked_recaught : ℕ) :
  initial_catch = 60 →
  second_catch = 80 →
  marked_recaught = 5 →
  (initial_catch * second_catch) / marked_recaught = 960 :=
by
  sorry

end NUMINAMATH_CALUDE_estimate_fish_population_l852_85263


namespace NUMINAMATH_CALUDE_job_completion_time_l852_85293

theorem job_completion_time (lisa_time tom_time combined_time sam_time : ℝ) : 
  lisa_time = 6 →
  tom_time = 2 →
  combined_time = 1.09090909091 →
  1 / sam_time + 1 / lisa_time + 1 / tom_time = 1 / combined_time →
  sam_time = 4 := by
sorry

end NUMINAMATH_CALUDE_job_completion_time_l852_85293


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l852_85246

/-- Theorem: If the length of a rectangle is increased by 50% and the area remains constant, 
    then the width of the rectangle must be decreased by 33.33%. -/
theorem rectangle_dimension_change (L W A : ℝ) (h1 : A = L * W) (h2 : A > 0) (h3 : L > 0) (h4 : W > 0) :
  let new_L := 1.5 * L
  let new_W := A / new_L
  (W - new_W) / W = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l852_85246


namespace NUMINAMATH_CALUDE_jillian_shells_l852_85257

theorem jillian_shells (savannah_shells clayton_shells : ℕ) 
  (h1 : savannah_shells = 17)
  (h2 : clayton_shells = 8)
  (h3 : ∃ (total_shells : ℕ), total_shells = 27 * 2)
  (h4 : ∃ (jillian_shells : ℕ), jillian_shells + savannah_shells + clayton_shells = 27 * 2) :
  ∃ (jillian_shells : ℕ), jillian_shells = 29 := by
sorry

end NUMINAMATH_CALUDE_jillian_shells_l852_85257


namespace NUMINAMATH_CALUDE_greatest_integer_of_2e_minus_5_l852_85287

theorem greatest_integer_of_2e_minus_5 :
  ⌊2 * Real.exp 1 - 5⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_of_2e_minus_5_l852_85287


namespace NUMINAMATH_CALUDE_inverse_of_B_squared_l852_85261

theorem inverse_of_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = ![![3, -2], ![0, 1]] →
  (B^2)⁻¹ = ![![9, -6], ![0, 1]] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_B_squared_l852_85261


namespace NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l852_85259

theorem x_gt_one_sufficient_not_necessary_for_abs_x_gt_one :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧ 
  ¬(∀ x : ℝ, |x| > 1 → x > 1) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_one_sufficient_not_necessary_for_abs_x_gt_one_l852_85259


namespace NUMINAMATH_CALUDE_square_diagonal_shorter_path_l852_85265

theorem square_diagonal_shorter_path (ε : Real) (h : ε > 0) : 
  ∃ (diff : Real), 
    abs (diff - 0.3) < ε ∧ 
    (2 - Real.sqrt 2) / 2 = diff :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_shorter_path_l852_85265


namespace NUMINAMATH_CALUDE_farmer_land_area_l852_85220

theorem farmer_land_area : ∃ (total : ℚ),
  total > 0 ∧
  total / 3 + total / 4 + total / 5 + 26 = total ∧
  total = 120 := by
  sorry

end NUMINAMATH_CALUDE_farmer_land_area_l852_85220


namespace NUMINAMATH_CALUDE_wages_theorem_l852_85204

/-- 
Given:
- A sum of money can pay A's wages for 20 days
- The same sum of money can pay B's wages for 30 days

Prove:
The same sum of money can pay both A and B's wages together for 12 days
-/
theorem wages_theorem (A B : ℝ) (h1 : 20 * A = 30 * B) : 
  12 * (A + B) = 20 * A := by sorry

end NUMINAMATH_CALUDE_wages_theorem_l852_85204


namespace NUMINAMATH_CALUDE_modulus_of_z_l852_85222

def i : ℂ := Complex.I

def z : ℂ := (1 + i) * (1 + 2*i)

theorem modulus_of_z : Complex.abs z = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l852_85222


namespace NUMINAMATH_CALUDE_only_two_random_events_l852_85253

-- Define the events
inductive Event
| SameChargesRepel
| SunnyTomorrow
| FreeFallStraightLine
| ExponentialIncreasing

-- Define a predicate for random events
def IsRandomEvent : Event → Prop :=
  fun e => match e with
  | Event.SunnyTomorrow => True
  | Event.ExponentialIncreasing => True
  | _ => False

-- Theorem statement
theorem only_two_random_events :
  (∀ e : Event, IsRandomEvent e ↔ (e = Event.SunnyTomorrow ∨ e = Event.ExponentialIncreasing)) :=
by sorry

end NUMINAMATH_CALUDE_only_two_random_events_l852_85253


namespace NUMINAMATH_CALUDE_smallest_angle_equation_l852_85229

/-- The smallest positive angle θ in degrees that satisfies the equation
    cos θ = sin 45° + cos 60° - sin 30° - cos 15° -/
theorem smallest_angle_equation : ∃ θ : ℝ,
  θ > 0 ∧
  θ < 360 ∧
  Real.cos (θ * π / 180) = Real.sin (45 * π / 180) + Real.cos (60 * π / 180) - 
                           Real.sin (30 * π / 180) - Real.cos (15 * π / 180) ∧
  ∀ φ, 0 < φ ∧ φ < θ → 
    Real.cos (φ * π / 180) ≠ Real.sin (45 * π / 180) + Real.cos (60 * π / 180) - 
                             Real.sin (30 * π / 180) - Real.cos (15 * π / 180) ∧
  θ = 30 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_equation_l852_85229


namespace NUMINAMATH_CALUDE_factorization_equality_l852_85231

theorem factorization_equality (m n : ℝ) : m^2 * n - m * n = m * n * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l852_85231


namespace NUMINAMATH_CALUDE_min_horse_pony_difference_l852_85221

/-- Represents a ranch with horses and ponies -/
structure Ranch where
  horses : ℕ
  ponies : ℕ
  horseshoed_ponies : ℕ
  icelandic_horseshoed_ponies : ℕ

/-- Conditions for the ranch -/
def valid_ranch (r : Ranch) : Prop :=
  r.horses > r.ponies ∧
  r.horses + r.ponies = 164 ∧
  r.horseshoed_ponies = (3 * r.ponies) / 10 ∧
  r.icelandic_horseshoed_ponies = (5 * r.horseshoed_ponies) / 8

theorem min_horse_pony_difference (r : Ranch) (h : valid_ranch r) :
  r.horses - r.ponies = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_horse_pony_difference_l852_85221


namespace NUMINAMATH_CALUDE_difference_of_squares_l852_85245

theorem difference_of_squares (x y : ℝ) : (-x + y) * (x + y) = y^2 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l852_85245


namespace NUMINAMATH_CALUDE_intersection_of_sets_l852_85286

theorem intersection_of_sets (A B : Set ℝ) : 
  A = {x : ℝ | x^2 + 2*x - 3 > 0} →
  B = {-1, 0, 1, 2} →
  A ∩ B = {2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l852_85286


namespace NUMINAMATH_CALUDE_adoption_time_l852_85207

def initial_puppies : ℕ := 3
def new_puppies : ℕ := 3
def adoption_rate : ℕ := 3

theorem adoption_time :
  (initial_puppies + new_puppies) / adoption_rate = 2 :=
sorry

end NUMINAMATH_CALUDE_adoption_time_l852_85207


namespace NUMINAMATH_CALUDE_product_of_place_values_l852_85203

def numeral : ℚ := 8712480.83

theorem product_of_place_values :
  let millions := 8000000
  let thousands := 8000
  let tenths := 0.8
  millions * thousands * tenths = 51200000000 :=
by sorry

end NUMINAMATH_CALUDE_product_of_place_values_l852_85203


namespace NUMINAMATH_CALUDE_potassium_dichromate_oxidizes_Br_and_I_l852_85227

/-- Standard reduction potential for I₂ + 2e⁻ → 2I⁻ -/
def E_I₂ : ℝ := 0.54

/-- Standard reduction potential for Cr₂O₇²⁻ + 14H⁺ + 6e⁻ → 2Cr³⁺ + 7H₂O -/
def E_Cr₂O₇ : ℝ := 1.33

/-- Standard oxidation potential for 2Br⁻ - 2e⁻ → Br₂ -/
def E_Br : ℝ := 1.07

/-- Standard oxidation potential for 2I⁻ - 2e⁻ → I₂ -/
def E_I : ℝ := 0.54

/-- A reaction is spontaneous if its cell potential is positive -/
def is_spontaneous (cell_potential : ℝ) : Prop := cell_potential > 0

/-- Theorem: Potassium dichromate can oxidize both Br⁻ and I⁻ -/
theorem potassium_dichromate_oxidizes_Br_and_I :
  is_spontaneous (E_Cr₂O₇ - E_Br) ∧ is_spontaneous (E_Cr₂O₇ - E_I) := by
  sorry


end NUMINAMATH_CALUDE_potassium_dichromate_oxidizes_Br_and_I_l852_85227


namespace NUMINAMATH_CALUDE_bicycle_trip_time_l852_85209

theorem bicycle_trip_time (mary_speed john_speed : ℝ) (distance : ℝ) : 
  mary_speed = 12 → 
  john_speed = 9 → 
  distance = 90 → 
  ∃ t : ℝ, t = 6 ∧ mary_speed * t ^ 2 + john_speed * t ^ 2 = distance ^ 2 :=
by
  sorry

end NUMINAMATH_CALUDE_bicycle_trip_time_l852_85209


namespace NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l852_85267

theorem rationalize_denominator_cube_root :
  ∃ (A B C : ℕ), 
    (A > 0) ∧ (B > 0) ∧ (C > 0) ∧
    (∀ p : ℕ, Prime p → ¬(p^3 ∣ B)) ∧
    (5 / (3 * Real.rpow 7 (1/3)) = (A * Real.rpow B (1/3)) / C) ∧
    (A + B + C = 75) := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_cube_root_l852_85267


namespace NUMINAMATH_CALUDE_girls_on_track_l852_85252

/-- Calculates the total number of girls on a track with given specifications -/
def total_girls (track_length : ℕ) (student_spacing : ℕ) : ℕ :=
  let students_per_side := track_length / student_spacing + 1
  let cycles_per_side := students_per_side / 3
  let girls_per_side := cycles_per_side * 2
  girls_per_side * 2

/-- The total number of girls on a 100-meter track with students every 2 meters,
    arranged in a pattern of two girls followed by one boy, is 68 -/
theorem girls_on_track : total_girls 100 2 = 68 := by
  sorry

end NUMINAMATH_CALUDE_girls_on_track_l852_85252


namespace NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l852_85299

theorem sum_of_angles_two_triangles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ) :
  angle1 + angle3 + angle5 = 180 →
  angle2 + angle4 + angle6 = 180 →
  angle1 + angle2 + angle3 + angle4 + angle5 + angle6 = 360 := by
sorry

end NUMINAMATH_CALUDE_sum_of_angles_two_triangles_l852_85299


namespace NUMINAMATH_CALUDE_factorize_expression1_factorize_expression2_l852_85224

-- First expression
theorem factorize_expression1 (y : ℝ) :
  y + (y - 4) * (y - 1) = (y - 2)^2 := by sorry

-- Second expression
theorem factorize_expression2 (x y a b : ℝ) :
  9 * a^2 * (x - y) + 4 * b^2 * (y - x) = (x - y) * (3 * a - 2 * b) * (3 * a + 2 * b) := by sorry

end NUMINAMATH_CALUDE_factorize_expression1_factorize_expression2_l852_85224


namespace NUMINAMATH_CALUDE_music_library_space_per_hour_l852_85264

theorem music_library_space_per_hour :
  let days : ℕ := 15
  let total_space : ℕ := 20000
  let hours_per_day : ℕ := 24
  let total_hours : ℕ := days * hours_per_day
  let space_per_hour : ℚ := total_space / total_hours
  round space_per_hour = 56 := by
  sorry

end NUMINAMATH_CALUDE_music_library_space_per_hour_l852_85264


namespace NUMINAMATH_CALUDE_pizza_delivery_time_per_stop_l852_85285

theorem pizza_delivery_time_per_stop 
  (total_pizzas : ℕ) 
  (double_order_stops : ℕ) 
  (total_delivery_time : ℕ) 
  (h1 : total_pizzas = 12) 
  (h2 : double_order_stops = 2) 
  (h3 : total_delivery_time = 40) : 
  (total_delivery_time : ℚ) / (total_pizzas - double_order_stops : ℚ) = 4 := by
  sorry

end NUMINAMATH_CALUDE_pizza_delivery_time_per_stop_l852_85285


namespace NUMINAMATH_CALUDE_square_plus_product_equals_zero_l852_85273

theorem square_plus_product_equals_zero : (-2)^2 + (-2) * 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_product_equals_zero_l852_85273


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l852_85298

theorem absolute_value_equation_solution :
  ∃! y : ℚ, |y - 3| = |y + 2| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l852_85298


namespace NUMINAMATH_CALUDE_solve_for_a_l852_85289

theorem solve_for_a : ∃ a : ℝ, (∀ x : ℝ, x = 2 → a * x - 2 = 4) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_a_l852_85289


namespace NUMINAMATH_CALUDE_unique_solution_equation_l852_85250

theorem unique_solution_equation :
  ∃! x : ℝ, x ≠ 2 ∧ x > 0 ∧ (3 * x^2 - 12 * x) / (x^2 - 4 * x) = x - 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l852_85250


namespace NUMINAMATH_CALUDE_coefficient_of_c_l852_85283

theorem coefficient_of_c (A : ℝ) (c d : ℝ) : 
  (∀ c', c' ≤ 47) → 
  (A * 47 + (d - 12)^2 = 235) → 
  A = 5 := by
sorry

end NUMINAMATH_CALUDE_coefficient_of_c_l852_85283


namespace NUMINAMATH_CALUDE_second_student_speed_l852_85201

/-- Given two students walking in opposite directions from the same starting point,
    this theorem proves that the second student's speed is 9 km/hr. -/
theorem second_student_speed
  (student1_speed : ℝ)
  (time : ℝ)
  (distance : ℝ)
  (h1 : student1_speed = 6)
  (h2 : time = 4)
  (h3 : distance = 60) :
  ∃ (student2_speed : ℝ),
    student2_speed = 9 ∧
    distance = (student1_speed + student2_speed) * time :=
by sorry

end NUMINAMATH_CALUDE_second_student_speed_l852_85201


namespace NUMINAMATH_CALUDE_students_behind_yoongi_l852_85284

theorem students_behind_yoongi (total_students : ℕ) (students_in_front : ℕ) : 
  total_students = 20 → students_in_front = 11 → total_students - (students_in_front + 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_students_behind_yoongi_l852_85284


namespace NUMINAMATH_CALUDE_r_daily_earning_l852_85296

/-- The daily earnings of p, q, and r satisfy the given conditions and r earns 70 per day -/
theorem r_daily_earning (p q r : ℚ) : 
  (9 * (p + q + r) = 1620) → 
  (5 * (p + r) = 600) → 
  (7 * (q + r) = 910) → 
  r = 70 := by
  sorry

end NUMINAMATH_CALUDE_r_daily_earning_l852_85296


namespace NUMINAMATH_CALUDE_arthurs_walk_l852_85235

/-- Arthur's walk problem -/
theorem arthurs_walk (blocks_west blocks_south : ℕ) (block_length : ℚ) :
  blocks_west = 8 →
  blocks_south = 10 →
  block_length = 1/4 →
  (blocks_west + blocks_south : ℚ) * block_length = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_walk_l852_85235


namespace NUMINAMATH_CALUDE_systematic_sampling_interval_example_l852_85213

/-- Calculates the interval for systematic sampling -/
def systematicSamplingInterval (totalPopulation : ℕ) (sampleSize : ℕ) : ℕ :=
  totalPopulation / sampleSize

/-- Theorem: The systematic sampling interval for 1000 students with a sample size of 50 is 20 -/
theorem systematic_sampling_interval_example :
  systematicSamplingInterval 1000 50 = 20 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_interval_example_l852_85213


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l852_85248

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1 : ℚ) * d

theorem arithmetic_sequence_properties :
  let a₁ : ℚ := 4
  let d : ℚ := 5
  let seq := arithmetic_sequence a₁ d
  (seq 3 * seq 6 = 406) ∧
  (∃ (q r : ℚ), seq 9 = seq 4 * q + r ∧ q = 2 ∧ r = 6) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l852_85248


namespace NUMINAMATH_CALUDE_polynomial_simplification_l852_85232

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 - 3 * q^3 + 7 * q - 8) + (5 - 2 * q^3 + 9 * q^2 - 4 * q) =
  4 * q^4 - 5 * q^3 + 9 * q^2 + 3 * q - 3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l852_85232


namespace NUMINAMATH_CALUDE_shortest_path_length_l852_85226

/-- A regular tetrahedron with unit edge length -/
structure UnitRegularTetrahedron where
  -- We don't need to define the structure explicitly for this problem

/-- The shortest path on the surface of a unit regular tetrahedron between midpoints of opposite edges -/
def shortest_path (t : UnitRegularTetrahedron) : ℝ :=
  sorry -- Definition of the shortest path

/-- Theorem: The shortest path on the surface of a unit regular tetrahedron 
    between the midpoints of its opposite edges has a length of 1 -/
theorem shortest_path_length (t : UnitRegularTetrahedron) : 
  shortest_path t = 1 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_length_l852_85226


namespace NUMINAMATH_CALUDE_pencil_pen_cost_l852_85234

theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.20)
  (h2 : 4 * p + 3 * q = 5.60) : 
  p + q = 1.40 := by
  sorry

end NUMINAMATH_CALUDE_pencil_pen_cost_l852_85234
