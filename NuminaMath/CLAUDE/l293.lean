import Mathlib

namespace NUMINAMATH_CALUDE_base_eight_perfect_square_c_is_one_l293_29391

/-- Represents a number in base 8 with the form 1b27c -/
def BaseEightNumber (b c : ℕ) : ℕ := 1024 + 64 * b + 16 + 7 + c

/-- A number is a perfect square if there exists an integer whose square is that number -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- The last digit of a perfect square in base 8 can only be 0, 1, or 4 -/
axiom perfect_square_mod_8 (n : ℕ) : IsPerfectSquare n → n % 8 ∈ ({0, 1, 4} : Set ℕ)

theorem base_eight_perfect_square_c_is_one (b : ℕ) :
  IsPerfectSquare (BaseEightNumber b 1) →
  ∀ c : ℕ, IsPerfectSquare (BaseEightNumber b c) → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_perfect_square_c_is_one_l293_29391


namespace NUMINAMATH_CALUDE_max_value_abc_l293_29375

theorem max_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  a^4 * b^3 * c^2 ≤ 1 / 6561 := by
sorry

end NUMINAMATH_CALUDE_max_value_abc_l293_29375


namespace NUMINAMATH_CALUDE_at_least_one_greater_than_one_l293_29340

theorem at_least_one_greater_than_one (a b : ℝ) :
  a + b > 2 → max a b > 1 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_greater_than_one_l293_29340


namespace NUMINAMATH_CALUDE_fraction_equality_l293_29394

theorem fraction_equality : ∃ x : ℚ, x * (7/8 * 1/3) = 0.12499999999999997 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l293_29394


namespace NUMINAMATH_CALUDE_toy_cars_in_first_box_l293_29313

theorem toy_cars_in_first_box 
  (total_boxes : Nat)
  (total_cars : Nat)
  (cars_in_second : Nat)
  (cars_in_third : Nat)
  (h1 : total_boxes = 3)
  (h2 : total_cars = 71)
  (h3 : cars_in_second = 31)
  (h4 : cars_in_third = 19) :
  total_cars - cars_in_second - cars_in_third = 21 :=
by sorry

end NUMINAMATH_CALUDE_toy_cars_in_first_box_l293_29313


namespace NUMINAMATH_CALUDE_parabola_properties_l293_29300

-- Define the parabola function
def f (x : ℝ) : ℝ := (x + 2)^2 - 1

-- State the theorem
theorem parabola_properties :
  (∀ x y : ℝ, f x ≤ f y → (x + 2)^2 ≤ (y + 2)^2) ∧ -- Opens upwards
  (∀ x : ℝ, f ((-2) + x) = f ((-2) - x)) ∧ -- Axis of symmetry is x = -2
  (∀ x₁ x₂ : ℝ, x₁ > -2 ∧ x₂ > -2 ∧ x₁ < x₂ → f x₁ < f x₂) ∧ -- y increases as x increases when x > -2
  (∀ x : ℝ, f x ≥ f (-2)) ∧ -- Minimum value at x = -2
  (f (-2) = -1) -- Minimum value is -1
  := by sorry

end NUMINAMATH_CALUDE_parabola_properties_l293_29300


namespace NUMINAMATH_CALUDE_vertical_asymptotes_sum_l293_29352

theorem vertical_asymptotes_sum (a b c : ℝ) (h : a ≠ 0) :
  let p := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let q := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a = 4 ∧ b = 6 ∧ c = 3 → p + q = -1.75 := by
  sorry

#check vertical_asymptotes_sum

end NUMINAMATH_CALUDE_vertical_asymptotes_sum_l293_29352


namespace NUMINAMATH_CALUDE_wilsonTotalIsCorrect_l293_29312

/-- Calculates the total amount Wilson pays at a fast-food restaurant -/
def wilsonTotal : ℝ :=
  let hamburgerPrice := 5
  let hamburgerCount := 2
  let colaPrice := 2
  let colaCount := 3
  let friesPrice := 3
  let sundaePrice := 4
  let nuggetPrice := 1.5
  let nuggetCount := 4
  let saladPrice := 6.25
  let couponDiscount := 4
  let loyaltyDiscount := 0.1
  let freeNuggetCount := 1

  let initialTotal := hamburgerPrice * hamburgerCount + colaPrice * colaCount + 
                      friesPrice + sundaePrice + nuggetPrice * nuggetCount + saladPrice
  let promotionDiscount := nuggetPrice * freeNuggetCount
  let afterPromotionTotal := initialTotal - promotionDiscount
  let afterCouponTotal := afterPromotionTotal - couponDiscount
  let finalTotal := afterCouponTotal * (1 - loyaltyDiscount)

  finalTotal

theorem wilsonTotalIsCorrect : wilsonTotal = 26.77 := by sorry

end NUMINAMATH_CALUDE_wilsonTotalIsCorrect_l293_29312


namespace NUMINAMATH_CALUDE_chinese_count_l293_29306

theorem chinese_count (total : ℕ) (americans : ℕ) (australians : ℕ) 
  (h1 : total = 49)
  (h2 : americans = 16)
  (h3 : australians = 11) :
  total - (americans + australians) = 22 := by
sorry

end NUMINAMATH_CALUDE_chinese_count_l293_29306


namespace NUMINAMATH_CALUDE_nell_initial_cards_l293_29311

/-- The number of baseball cards Nell had initially -/
def initial_cards : ℕ := sorry

/-- The number of cards Jeff gave to Nell -/
def cards_from_jeff : ℝ := 276.0

/-- The total number of cards Nell has now -/
def total_cards : ℕ := 580

/-- Theorem stating that Nell's initial number of cards was 304 -/
theorem nell_initial_cards : 
  initial_cards = 304 :=
by
  sorry

end NUMINAMATH_CALUDE_nell_initial_cards_l293_29311


namespace NUMINAMATH_CALUDE_even_function_inequality_l293_29372

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_nonpositive (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → x₂ ≤ 0 → f x₁ < f x₂

theorem even_function_inequality (f : ℝ → ℝ) (n : ℕ) 
  (h_even : is_even_function f)
  (h_incr : increasing_on_nonpositive f) :
  f (n + 1) < f (-n) ∧ f (-n) < f (n - 1) :=
sorry

end NUMINAMATH_CALUDE_even_function_inequality_l293_29372


namespace NUMINAMATH_CALUDE_pages_to_read_on_day_three_l293_29319

theorem pages_to_read_on_day_three 
  (total_pages : ℕ) 
  (pages_day_one : ℕ) 
  (pages_day_two : ℕ) 
  (h1 : total_pages = 100)
  (h2 : pages_day_one = 35)
  (h3 : pages_day_two = pages_day_one - 5) :
  total_pages - (pages_day_one + pages_day_two) = 35 := by
  sorry

end NUMINAMATH_CALUDE_pages_to_read_on_day_three_l293_29319


namespace NUMINAMATH_CALUDE_triangle_value_l293_29350

theorem triangle_value (q : ℤ) (h1 : ∃ triangle : ℤ, triangle + q = 59) 
  (h2 : ∃ triangle : ℤ, (triangle + q) + q = 106) : 
  ∃ triangle : ℤ, triangle = 12 := by
  sorry

end NUMINAMATH_CALUDE_triangle_value_l293_29350


namespace NUMINAMATH_CALUDE_polynomial_functional_equation_l293_29357

-- Define a polynomial with real coefficients
def RealPolynomial := ℝ → ℝ

-- Define the functional equation
def SatisfiesFunctionalEquation (P : RealPolynomial) : Prop :=
  ∀ x : ℝ, 1 + P x = (1 / 2) * (P (x - 1) + P (x + 1))

-- Define the quadratic form
def IsQuadraticForm (P : RealPolynomial) : Prop :=
  ∃ b c : ℝ, ∀ x : ℝ, P x = x^2 + b * x + c

-- Theorem statement
theorem polynomial_functional_equation :
  ∀ P : RealPolynomial, SatisfiesFunctionalEquation P → IsQuadraticForm P :=
by
  sorry


end NUMINAMATH_CALUDE_polynomial_functional_equation_l293_29357


namespace NUMINAMATH_CALUDE_infinite_k_sin_k_greater_than_C_l293_29321

theorem infinite_k_sin_k_greater_than_C :
  ∀ C : ℝ, ∃ S : Set ℤ, (Set.Infinite S) ∧ (∀ k ∈ S, (k : ℝ) * Real.sin k > C) := by
  sorry

end NUMINAMATH_CALUDE_infinite_k_sin_k_greater_than_C_l293_29321


namespace NUMINAMATH_CALUDE_camille_bird_counting_l293_29396

theorem camille_bird_counting (cardinals : ℕ) 
  (h1 : cardinals > 0)
  (h2 : cardinals + 4 * cardinals + 2 * cardinals + (3 * cardinals + 1) = 31) :
  cardinals = 3 := by
sorry

end NUMINAMATH_CALUDE_camille_bird_counting_l293_29396


namespace NUMINAMATH_CALUDE_rectangular_field_area_l293_29363

/-- A rectangular field with length double its width and perimeter 180 meters has an area of 1800 square meters. -/
theorem rectangular_field_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  width > 0 →
  length = 2 * width →
  perimeter = 2 * (length + width) →
  perimeter = 180 →
  area = length * width →
  area = 1800 := by
  sorry

#check rectangular_field_area

end NUMINAMATH_CALUDE_rectangular_field_area_l293_29363


namespace NUMINAMATH_CALUDE_total_team_combinations_l293_29364

/-- The number of ways to choose k items from n items without replacement and without regard to order. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of people in each group. -/
def group_size : ℕ := 6

/-- The number of people to be selected from each group. -/
def team_size : ℕ := 3

/-- The number of groups. -/
def num_groups : ℕ := 2

theorem total_team_combinations : 
  (choose group_size team_size) ^ num_groups = 400 := by sorry

end NUMINAMATH_CALUDE_total_team_combinations_l293_29364


namespace NUMINAMATH_CALUDE_same_group_probability_l293_29303

/-- The probability that two randomly selected people from a group of 16 divided into two equal subgroups are from the same subgroup is 7/15. -/
theorem same_group_probability (n : ℕ) (h1 : n = 16) (h2 : n % 2 = 0) : 
  (Nat.choose (n / 2) 2 * 2) / Nat.choose n 2 = 7 / 15 := by
  sorry

end NUMINAMATH_CALUDE_same_group_probability_l293_29303


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l293_29330

theorem geometric_sequence_product (a b : ℝ) : 
  2 < a ∧ a < b ∧ b < 16 ∧ 
  (∃ r : ℝ, r > 0 ∧ a = 2 * r ∧ b = 2 * r^2 ∧ 16 = 2 * r^3) →
  a * b = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l293_29330


namespace NUMINAMATH_CALUDE_power_of_five_preceded_by_coprimes_l293_29331

theorem power_of_five_preceded_by_coprimes (x : ℕ) : 
  (5^x - 1 - (5^x / 5 - 1) = 7812500) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_of_five_preceded_by_coprimes_l293_29331


namespace NUMINAMATH_CALUDE_symmetric_line_l293_29384

/-- Given a line with equation 2x - y + 3 = 0 and a fixed point M(-1, 2),
    the equation of the line symmetric to the given line with respect to M is 2x - y + 5 = 0 -/
theorem symmetric_line (x y : ℝ) : 
  (∀ x y, 2*x - y + 3 = 0 → 2*x - y + 5 = 0) := by
  sorry

end NUMINAMATH_CALUDE_symmetric_line_l293_29384


namespace NUMINAMATH_CALUDE_odd_numbers_sum_product_equality_l293_29325

/-- For a positive integer n, there exist n positive odd numbers whose sum equals 
    their product if and only if n is of the form 4k + 1, where k is a non-negative integer. -/
theorem odd_numbers_sum_product_equality (n : ℕ+) : 
  (∃ (S : Finset ℕ), S.card = n ∧ 
    (∀ x ∈ S, Odd x ∧ x > 0) ∧ 
    (S.sum id = S.prod id)) ↔ 
  ∃ k : ℕ, n = 4 * k + 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_numbers_sum_product_equality_l293_29325


namespace NUMINAMATH_CALUDE_calculator_sum_l293_29355

/-- The number of participants in the circle. -/
def n : ℕ := 44

/-- The operation performed on the first calculator (squaring). -/
def op1 (x : ℕ) : ℕ := x ^ 2

/-- The operation performed on the second calculator (squaring). -/
def op2 (x : ℕ) : ℕ := x ^ 2

/-- The operation performed on the third calculator (negation). -/
def op3 (x : ℤ) : ℤ := -x

/-- The final value of the first calculator after n iterations. -/
def final1 : ℕ := 2 ^ (2 ^ n)

/-- The final value of the second calculator after n iterations. -/
def final2 : ℕ := 0

/-- The final value of the third calculator after n iterations. -/
def final3 : ℤ := (-1) ^ n

/-- The theorem stating the final sum of the calculators. -/
theorem calculator_sum :
  (final1 : ℤ) + final2 + final3 = 2 ^ (2 ^ n) + 1 := by sorry

end NUMINAMATH_CALUDE_calculator_sum_l293_29355


namespace NUMINAMATH_CALUDE_square_perimeter_l293_29323

theorem square_perimeter (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 125)
  (h2 : rectangle_width = 64)
  (h3 : rectangle_length > 0)
  (h4 : rectangle_width > 0) :
  let rectangle_area := rectangle_length * rectangle_width
  let square_area := 5 * rectangle_area
  let square_side := Real.sqrt square_area
  square_side * 4 = 800 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_l293_29323


namespace NUMINAMATH_CALUDE_g_monotone_and_range_l293_29328

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := (2^x + b) / (2^x - b)

theorem g_monotone_and_range (b : ℝ) :
  (b < 0 → ∀ x y : ℝ, x < y → g b x < g b y) ∧
  (b = -1 → ∀ a : ℝ, (∀ x : ℝ, g (-1) (x^2 + 1) + g (-1) (3 - a*x) > 0) ↔ -4 < a ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_g_monotone_and_range_l293_29328


namespace NUMINAMATH_CALUDE_age_ratio_proof_l293_29326

theorem age_ratio_proof (b_age : ℕ) (a_age : ℕ) : 
  b_age = 39 →
  a_age = b_age + 9 →
  (a_age + 10) / (b_age - 10) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l293_29326


namespace NUMINAMATH_CALUDE_mia_sock_purchase_l293_29346

/-- Represents the number of pairs of socks at each price point --/
structure SockPurchase where
  twoDoller : ℕ
  threeDoller : ℕ
  fiveDoller : ℕ

/-- Checks if the given SockPurchase satisfies the problem conditions --/
def isValidPurchase (p : SockPurchase) : Prop :=
  p.twoDoller + p.threeDoller + p.fiveDoller = 15 ∧
  2 * p.twoDoller + 3 * p.threeDoller + 5 * p.fiveDoller = 35 ∧
  p.twoDoller ≥ 1 ∧ p.threeDoller ≥ 1 ∧ p.fiveDoller ≥ 1

theorem mia_sock_purchase :
  ∃ (p : SockPurchase), isValidPurchase p ∧ p.twoDoller = 12 := by
  sorry

end NUMINAMATH_CALUDE_mia_sock_purchase_l293_29346


namespace NUMINAMATH_CALUDE_system_always_solvable_l293_29397

/-- Given a system of linear equations:
    ax + by = c - 1
    (a+5)x + (b+3)y = c + 1
    This theorem states that for the system to always have a solution
    for any real a and b, c must equal (2a + 5) / 5. -/
theorem system_always_solvable (a b c : ℝ) :
  (∀ x y : ℝ, a * x + b * y = c - 1 ∧ (a + 5) * x + (b + 3) * y = c + 1) ↔
  c = (2 * a + 5) / 5 := by
  sorry


end NUMINAMATH_CALUDE_system_always_solvable_l293_29397


namespace NUMINAMATH_CALUDE_R_calculation_l293_29351

/-- R_k is the integer composed of k repeating digits of 1 in decimal form -/
def R (k : ℕ) : ℕ := (10^k - 1) / 9

/-- The result of the calculation R_36/R_6 - R_3 -/
def result : ℕ := 100000100000100000100000100000099989

/-- Theorem stating that R_36/R_6 - R_3 equals the specified result -/
theorem R_calculation : R 36 / R 6 - R 3 = result := by
  sorry

end NUMINAMATH_CALUDE_R_calculation_l293_29351


namespace NUMINAMATH_CALUDE_initial_selling_price_theorem_l293_29370

/-- The number of articles sold at a gain -/
def articles_sold_gain : ℝ := 20

/-- The gain percentage -/
def gain_percentage : ℝ := 0.20

/-- The number of articles that would be sold at a loss -/
def articles_sold_loss : ℝ := 29.99999625000047

/-- The loss percentage -/
def loss_percentage : ℝ := 0.20

/-- Theorem stating that the initial selling price for articles sold at a gain
    is 24 times the cost price of one article -/
theorem initial_selling_price_theorem (cost_price : ℝ) :
  let selling_price_gain := cost_price * (1 + gain_percentage)
  let selling_price_loss := cost_price * (1 - loss_percentage)
  articles_sold_gain * selling_price_gain = articles_sold_loss * selling_price_loss →
  articles_sold_gain * selling_price_gain = 24 * cost_price :=
by sorry

end NUMINAMATH_CALUDE_initial_selling_price_theorem_l293_29370


namespace NUMINAMATH_CALUDE_number_of_cats_l293_29366

/-- The number of cats on a farm, given the number of dogs, fish, and total pets -/
theorem number_of_cats (dogs : ℕ) (fish : ℕ) (total_pets : ℕ) (h1 : dogs = 43) (h2 : fish = 72) (h3 : total_pets = 149) :
  total_pets - dogs - fish = 34 := by
sorry

end NUMINAMATH_CALUDE_number_of_cats_l293_29366


namespace NUMINAMATH_CALUDE_range_of_a_l293_29371

def A (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a) ≥ 0}
def B (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem range_of_a (a : ℝ) (h : A a ∪ B a = Set.univ) : a ∈ Set.Iic 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l293_29371


namespace NUMINAMATH_CALUDE_error_percentage_l293_29381

theorem error_percentage (x : ℝ) (h : x > 0) : 
  (|4*x - x/4|) / (4*x) = 15/16 := by
sorry

end NUMINAMATH_CALUDE_error_percentage_l293_29381


namespace NUMINAMATH_CALUDE_orange_weight_problem_l293_29392

theorem orange_weight_problem (initial_water_concentration : Real)
                               (water_decrease : Real)
                               (new_weight : Real) :
  initial_water_concentration = 0.95 →
  water_decrease = 0.05 →
  new_weight = 25 →
  ∃ (initial_weight : Real),
    initial_weight = 50 ∧
    (1 - initial_water_concentration) * initial_weight =
    (1 - (initial_water_concentration - water_decrease)) * new_weight :=
by sorry

end NUMINAMATH_CALUDE_orange_weight_problem_l293_29392


namespace NUMINAMATH_CALUDE_return_trip_time_l293_29378

/-- The time taken for a return trip given the conditions of the original journey -/
theorem return_trip_time 
  (total_distance : ℝ) 
  (uphill_speed downhill_speed : ℝ)
  (forward_time : ℝ)
  (h1 : total_distance = 21)
  (h2 : uphill_speed = 4)
  (h3 : downhill_speed = 6)
  (h4 : forward_time = 4.25)
  (h5 : ∃ (uphill_distance downhill_distance : ℝ), 
    uphill_distance + downhill_distance = total_distance ∧
    uphill_distance / uphill_speed + downhill_distance / downhill_speed = forward_time) :
  ∃ (return_time : ℝ), return_time = 4.5 := by
sorry

end NUMINAMATH_CALUDE_return_trip_time_l293_29378


namespace NUMINAMATH_CALUDE_spice_jar_cost_is_six_l293_29362

/-- Represents the cost and point structure for Martha's grocery shopping -/
structure GroceryShopping where
  pointsPerTenDollars : ℕ
  bonusThreshold : ℕ
  bonusPoints : ℕ
  beefPounds : ℕ
  beefPricePerPound : ℕ
  fruitVegPounds : ℕ
  fruitVegPricePerPound : ℕ
  spiceJars : ℕ
  otherGroceriesCost : ℕ
  totalPoints : ℕ

/-- Calculates the cost of each jar of spices based on the given shopping information -/
def calculateSpiceJarCost (shopping : GroceryShopping) : ℕ :=
  sorry

/-- Theorem stating that the cost of each jar of spices is $6 -/
theorem spice_jar_cost_is_six (shopping : GroceryShopping) 
  (h1 : shopping.pointsPerTenDollars = 50)
  (h2 : shopping.bonusThreshold = 100)
  (h3 : shopping.bonusPoints = 250)
  (h4 : shopping.beefPounds = 3)
  (h5 : shopping.beefPricePerPound = 11)
  (h6 : shopping.fruitVegPounds = 8)
  (h7 : shopping.fruitVegPricePerPound = 4)
  (h8 : shopping.spiceJars = 3)
  (h9 : shopping.otherGroceriesCost = 37)
  (h10 : shopping.totalPoints = 850) :
  calculateSpiceJarCost shopping = 6 :=
  sorry


end NUMINAMATH_CALUDE_spice_jar_cost_is_six_l293_29362


namespace NUMINAMATH_CALUDE_min_value_theorem_l293_29332

theorem min_value_theorem (x : ℝ) (h : x > 0) : 3 * Real.sqrt x + 2 / x^2 ≥ 5 ∧
  (3 * Real.sqrt x + 2 / x^2 = 5 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l293_29332


namespace NUMINAMATH_CALUDE_greatest_two_digit_product_12_l293_29318

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_product_12 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 12 → n ≤ 62 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_product_12_l293_29318


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l293_29369

/-- Converts a list of digits in base 3 to a base 10 integer -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number -/
def base3Digits : List Nat := [1, 2, 0, 2, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 142 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l293_29369


namespace NUMINAMATH_CALUDE_power_of_power_at_three_l293_29386

theorem power_of_power_at_three : (3^3)^(3^3) = 27^27 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_at_three_l293_29386


namespace NUMINAMATH_CALUDE_max_product_xy_l293_29327

theorem max_product_xy (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (eq1 : x + 1/y = 3) (eq2 : y + 2/x = 3) :
  ∃ (C : ℝ), C = x*y ∧ C ≤ 3 + Real.sqrt 7 ∧
  ∃ (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + 1/y' = 3 ∧ y' + 2/x' = 3 ∧ x'*y' = 3 + Real.sqrt 7 :=
sorry

end NUMINAMATH_CALUDE_max_product_xy_l293_29327


namespace NUMINAMATH_CALUDE_equation_solution_l293_29324

theorem equation_solution : ∃! x : ℚ, (7 * x - 2) / (x + 4) - 4 / (x + 4) = 2 / (x + 4) ∧ x = 8 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l293_29324


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l293_29320

-- Define the propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x - 1| > m
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5 - 2*m)^x) > (-(5 - 2*m)^y)

-- State the theorem
theorem p_sufficient_not_necessary_for_q :
  (∃ m : ℝ, p m ∧ q m) ∧ (∃ m : ℝ, q m ∧ ¬(p m)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l293_29320


namespace NUMINAMATH_CALUDE_min_coach_handshakes_l293_29302

theorem min_coach_handshakes (total_handshakes : ℕ) (h : total_handshakes = 465) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_handshakes ∧
  (∀ (m₁ m₂ : ℕ), m₁ + m₂ = n → m₁ + m₂ = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_coach_handshakes_l293_29302


namespace NUMINAMATH_CALUDE_adeline_hourly_wage_l293_29380

def hours_per_day : ℕ := 9
def days_per_week : ℕ := 5
def weeks_worked : ℕ := 7
def total_earnings : ℕ := 3780

def hourly_wage : ℚ :=
  total_earnings / (hours_per_day * days_per_week * weeks_worked)

theorem adeline_hourly_wage : hourly_wage = 12 := by
  sorry

end NUMINAMATH_CALUDE_adeline_hourly_wage_l293_29380


namespace NUMINAMATH_CALUDE_meaningful_fraction_l293_29304

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (2*x + 1)/(x - 2)) ↔ x ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l293_29304


namespace NUMINAMATH_CALUDE_right_triangle_solution_l293_29329

theorem right_triangle_solution :
  ∃ (x : ℝ), x > 0 ∧
  (4 * x + 2) > 0 ∧
  ((x - 3)^2) > 0 ∧
  (5 * x + 1) > 0 ∧
  (4 * x + 2)^2 + (x - 3)^4 = (5 * x + 1)^2 ∧
  x = Real.sqrt (3/2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_solution_l293_29329


namespace NUMINAMATH_CALUDE_sets_properties_l293_29308

def M : Set ℤ := {x | ∃ k : ℤ, x = 6*k + 1}
def N : Set ℤ := {x | ∃ k : ℤ, x = 6*k + 4}
def P : Set ℤ := {x | ∃ k : ℤ, x = 3*k - 2}

theorem sets_properties : (M ∩ N = ∅) ∧ (P \ M = N) := by sorry

end NUMINAMATH_CALUDE_sets_properties_l293_29308


namespace NUMINAMATH_CALUDE_glass_bowls_percentage_gain_l293_29307

/-- Calculate the percentage gain from buying and selling glass bowls -/
theorem glass_bowls_percentage_gain 
  (total_bought : ℕ) 
  (cost_price : ℚ) 
  (total_sold : ℕ) 
  (selling_price : ℚ) 
  (broken : ℕ) 
  (h1 : total_bought = 250)
  (h2 : cost_price = 18)
  (h3 : total_sold = 200)
  (h4 : selling_price = 25)
  (h5 : broken = 30)
  (h6 : total_sold + broken ≤ total_bought) :
  (((total_sold : ℚ) * selling_price - (total_bought : ℚ) * cost_price) / 
   ((total_bought : ℚ) * cost_price)) * 100 = 100 / 9 := by
sorry

#eval (100 : ℚ) / 9  -- To show the approximate result

end NUMINAMATH_CALUDE_glass_bowls_percentage_gain_l293_29307


namespace NUMINAMATH_CALUDE_junior_girls_count_l293_29314

theorem junior_girls_count (total_players : ℕ) (boy_percentage : ℚ) : 
  total_players = 50 → 
  boy_percentage = 60 / 100 → 
  (total_players : ℚ) * (1 - boy_percentage) / 2 = 10 := by
  sorry

end NUMINAMATH_CALUDE_junior_girls_count_l293_29314


namespace NUMINAMATH_CALUDE_apples_per_box_l293_29393

theorem apples_per_box (total_apples : ℕ) (rotten_apples : ℕ) (num_boxes : ℕ) 
  (h1 : total_apples = 40)
  (h2 : rotten_apples = 4)
  (h3 : num_boxes = 4)
  (h4 : rotten_apples < total_apples) :
  (total_apples - rotten_apples) / num_boxes = 9 := by
sorry

end NUMINAMATH_CALUDE_apples_per_box_l293_29393


namespace NUMINAMATH_CALUDE_triangle_properties_l293_29383

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  (t.b - t.c)^2 = t.a^2 - t.b * t.c ∧
  t.a = 3 ∧
  Real.sin t.C = 2 * Real.sin t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) (h : satisfies_conditions t) :
  t.A = π/3 ∧ t.a * t.b * Real.sin t.C / 2 = 3 * Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l293_29383


namespace NUMINAMATH_CALUDE_solve_equation_l293_29336

theorem solve_equation : ∃ y : ℚ, y + 2/3 = 1/4 - 2/5 * 2 ∧ y = -511/420 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l293_29336


namespace NUMINAMATH_CALUDE_trip_duration_proof_l293_29341

/-- The battery life in standby mode (in hours) -/
def standby_life : ℝ := 210

/-- The rate at which the battery depletes while talking compared to standby mode -/
def talking_depletion_rate : ℝ := 35

/-- Calculates the total trip duration given the time spent talking -/
def total_trip_duration (talking_time : ℝ) : ℝ := 2 * talking_time

/-- Theorem stating that the total trip duration is 11 hours and 40 minutes -/
theorem trip_duration_proof :
  ∃ (talking_time : ℝ),
    talking_time > 0 ∧
    talking_time ≤ standby_life ∧
    talking_depletion_rate * (standby_life - talking_time) = talking_time ∧
    total_trip_duration talking_time = 11 + 40 / 60 :=
by sorry

end NUMINAMATH_CALUDE_trip_duration_proof_l293_29341


namespace NUMINAMATH_CALUDE_shopkeeper_net_loss_percent_l293_29339

/-- Calculates the net profit or loss percentage for a shopkeeper's transactions -/
theorem shopkeeper_net_loss_percent : 
  let cost_price : ℝ := 1000
  let num_articles : ℕ := 4
  let profit_percent1 : ℝ := 10
  let loss_percent2 : ℝ := 10
  let profit_percent3 : ℝ := 20
  let loss_percent4 : ℝ := 25
  
  let selling_price1 : ℝ := cost_price * (1 + profit_percent1 / 100)
  let selling_price2 : ℝ := cost_price * (1 - loss_percent2 / 100)
  let selling_price3 : ℝ := cost_price * (1 + profit_percent3 / 100)
  let selling_price4 : ℝ := cost_price * (1 - loss_percent4 / 100)
  
  let total_cost : ℝ := cost_price * num_articles
  let total_selling : ℝ := selling_price1 + selling_price2 + selling_price3 + selling_price4
  
  let net_loss : ℝ := total_cost - total_selling
  let net_loss_percent : ℝ := (net_loss / total_cost) * 100
  
  net_loss_percent = 1.25 := by sorry

end NUMINAMATH_CALUDE_shopkeeper_net_loss_percent_l293_29339


namespace NUMINAMATH_CALUDE_ln_b_over_a_range_l293_29317

theorem ln_b_over_a_range (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : (1 : ℝ) / Real.exp 1 ≤ c / a) (h2 : c / a ≤ 2)
  (h3 : c * Real.log b = a + c * Real.log c) :
  ∃ (x : ℝ), x ∈ Set.Icc 1 (Real.exp 1 - 1) ∧ Real.log (b / a) = x :=
sorry

end NUMINAMATH_CALUDE_ln_b_over_a_range_l293_29317


namespace NUMINAMATH_CALUDE_income_calculation_l293_29389

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 5 →
  income - expenditure = savings →
  savings = 3800 →
  income = 19000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l293_29389


namespace NUMINAMATH_CALUDE_Q_roots_l293_29379

def Q (x : ℝ) : ℝ := x^6 - 5*x^5 - 12*x^3 - x + 16

theorem Q_roots :
  (∀ x < 0, Q x > 0) ∧ 
  (∃ x > 0, Q x = 0) := by
sorry

end NUMINAMATH_CALUDE_Q_roots_l293_29379


namespace NUMINAMATH_CALUDE_price_decrease_percentage_l293_29301

theorem price_decrease_percentage (original_price : ℝ) (h : original_price > 0) :
  let first_sale_price := (4 / 5 : ℝ) * original_price
  let second_sale_price := (1 / 2 : ℝ) * original_price
  let price_difference := first_sale_price - second_sale_price
  let percentage_decrease := (price_difference / first_sale_price) * 100
  percentage_decrease = 37.5 := by
sorry

end NUMINAMATH_CALUDE_price_decrease_percentage_l293_29301


namespace NUMINAMATH_CALUDE_product_of_areas_is_perfect_square_l293_29382

/-- A convex quadrilateral divided by its diagonals -/
structure ConvexQuadrilateral where
  /-- The areas of the four triangles formed by the diagonals -/
  area₁ : ℤ
  area₂ : ℤ
  area₃ : ℤ
  area₄ : ℤ

/-- The product of the areas of the four triangles in a convex quadrilateral
    divided by its diagonals is a perfect square -/
theorem product_of_areas_is_perfect_square (q : ConvexQuadrilateral) :
  ∃ (n : ℤ), q.area₁ * q.area₂ * q.area₃ * q.area₄ = n ^ 2 := by
  sorry


end NUMINAMATH_CALUDE_product_of_areas_is_perfect_square_l293_29382


namespace NUMINAMATH_CALUDE_octagon_diagonals_l293_29310

/-- The number of diagonals in a polygon with n vertices -/
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

/-- The number of vertices in an octagon -/
def octagon_vertices : ℕ := 8

/-- Theorem: The number of diagonals in an octagon is 20 -/
theorem octagon_diagonals : num_diagonals octagon_vertices = 20 := by
  sorry

end NUMINAMATH_CALUDE_octagon_diagonals_l293_29310


namespace NUMINAMATH_CALUDE_natalie_shopping_money_left_l293_29349

def initial_amount : ℕ := 26
def jumper_cost : ℕ := 9
def tshirt_cost : ℕ := 4
def heels_cost : ℕ := 5

theorem natalie_shopping_money_left :
  initial_amount - (jumper_cost + tshirt_cost + heels_cost) = 8 := by
  sorry

end NUMINAMATH_CALUDE_natalie_shopping_money_left_l293_29349


namespace NUMINAMATH_CALUDE_nesbitt_inequality_l293_29387

theorem nesbitt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_nesbitt_inequality_l293_29387


namespace NUMINAMATH_CALUDE_jungkook_red_balls_l293_29356

/-- Given that each box contains 3 red balls and Jungkook has 2 boxes, 
    prove that Jungkook has 6 red balls in total. -/
theorem jungkook_red_balls (balls_per_box : ℕ) (num_boxes : ℕ) 
  (h1 : balls_per_box = 3)
  (h2 : num_boxes = 2) :
  balls_per_box * num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_jungkook_red_balls_l293_29356


namespace NUMINAMATH_CALUDE_inverse_of_3_mod_221_l293_29322

theorem inverse_of_3_mod_221 : ∃ x : ℕ, x < 221 ∧ (3 * x) % 221 = 1 :=
by
  use 74
  sorry

end NUMINAMATH_CALUDE_inverse_of_3_mod_221_l293_29322


namespace NUMINAMATH_CALUDE_maggies_total_earnings_l293_29360

/-- Calculates Maggie's earnings from selling magazine subscriptions -/
def maggies_earnings (price_per_subscription : ℕ) 
  (parents_subscriptions : ℕ) 
  (grandfather_subscriptions : ℕ) 
  (nextdoor_subscriptions : ℕ) : ℕ :=
  let total_subscriptions := 
    parents_subscriptions + 
    grandfather_subscriptions + 
    nextdoor_subscriptions + 
    (2 * nextdoor_subscriptions)
  price_per_subscription * total_subscriptions

theorem maggies_total_earnings : 
  maggies_earnings 5 4 1 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_maggies_total_earnings_l293_29360


namespace NUMINAMATH_CALUDE_triangle_tangent_ratio_l293_29337

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_tangent_ratio (t : Triangle) 
  (h : t.a * Real.cos t.B - t.b * Real.cos t.A = (3/5) * t.c) : 
  Real.tan t.A / Real.tan t.B = 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_tangent_ratio_l293_29337


namespace NUMINAMATH_CALUDE_reinforcement_arrival_time_l293_29347

/-- Calculates the number of days passed before reinforcement arrived -/
def days_before_reinforcement (initial_garrison : ℕ) (initial_provisions : ℕ) 
  (reinforcement : ℕ) (remaining_provisions : ℕ) : ℕ :=
  (initial_garrison * initial_provisions - (initial_garrison + reinforcement) * remaining_provisions) / initial_garrison

/-- Theorem stating that 15 days passed before reinforcement arrived -/
theorem reinforcement_arrival_time :
  days_before_reinforcement 2000 65 3000 20 = 15 := by sorry

end NUMINAMATH_CALUDE_reinforcement_arrival_time_l293_29347


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l293_29359

theorem quadratic_always_positive : ∀ x : ℝ, x^2 + x + 2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l293_29359


namespace NUMINAMATH_CALUDE_train_passing_time_l293_29367

/-- Time for a train to pass a trolley moving in the opposite direction -/
theorem train_passing_time (train_length : ℝ) (train_speed : ℝ) (trolley_speed : ℝ) :
  train_length = 110 →
  train_speed = 60 * (1000 / 3600) →
  trolley_speed = 12 * (1000 / 3600) →
  (train_length / (train_speed + trolley_speed)) = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l293_29367


namespace NUMINAMATH_CALUDE_happy_water_consumption_l293_29358

/-- Given Happy's current water consumption and recommended increase percentage,
    calculate the new recommended number of cups per week. -/
theorem happy_water_consumption (current : ℝ) (increase_percent : ℝ) :
  current = 25 → increase_percent = 75 →
  current + (increase_percent / 100) * current = 43.75 := by
  sorry

end NUMINAMATH_CALUDE_happy_water_consumption_l293_29358


namespace NUMINAMATH_CALUDE_hair_cut_length_l293_29373

def hair_problem (initial_length growth_length final_length : ℕ) : ℕ :=
  initial_length + growth_length - final_length

theorem hair_cut_length :
  hair_problem 14 8 2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_hair_cut_length_l293_29373


namespace NUMINAMATH_CALUDE_g_at_one_l293_29348

theorem g_at_one (a b c d : ℝ) (h₁ : 1 < a) (h₂ : a < b) (h₃ : b < c) (h₄ : c < d) :
  let f : ℝ → ℝ := λ x => x^4 + a*x^3 + b*x^2 + c*x + d
  ∃ g : ℝ → ℝ,
    (∀ x, g x = 0 → ∃ y, f y = 0 ∧ x * y = 1) ∧
    (g 0 = 1) ∧
    (g 1 = (1 + a + b + c + d) / d) :=
by sorry

end NUMINAMATH_CALUDE_g_at_one_l293_29348


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_800_by_110_percent_l293_29338

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by
  sorry

theorem increase_800_by_110_percent :
  800 * (1 + 110 / 100) = 1680 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_800_by_110_percent_l293_29338


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l293_29399

-- Define the arithmetic sequence
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Theorem statement
theorem arithmetic_sequence_terms :
  ∃ n : ℕ, n > 0 ∧ arithmetic_sequence 17 6 n = 101 ∧ n = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l293_29399


namespace NUMINAMATH_CALUDE_auto_credit_percentage_l293_29376

def auto_finance_credit : ℝ := 40
def total_consumer_credit : ℝ := 342.857

theorem auto_credit_percentage :
  let total_auto_credit := 3 * auto_finance_credit
  let percentage := (total_auto_credit / total_consumer_credit) * 100
  ∃ ε > 0, |percentage - 35| < ε :=
sorry

end NUMINAMATH_CALUDE_auto_credit_percentage_l293_29376


namespace NUMINAMATH_CALUDE_parabola_tangent_circle_l293_29343

/-- Given a parabola x^2 = 16y, prove that a circle centered at the focus of the parabola
    and tangent to its directrix has the equation x^2 + (y - 4)^2 = 64 -/
theorem parabola_tangent_circle (x y : ℝ) :
  (x^2 = 16*y) →  -- Parabola equation
  ∃ (h k r : ℝ),
    (h = 0 ∧ k = 4) →  -- Focus of the parabola (center of the circle)
    (∀ (x' y' : ℝ), y' = -4 → (x' - h)^2 + (y' - k)^2 ≥ r^2) →  -- Circle tangent to directrix
    (x - h)^2 + (y - k)^2 = r^2 →  -- General circle equation
    x^2 + (y - 4)^2 = 64  -- Specific circle equation to be proved
  := by sorry

end NUMINAMATH_CALUDE_parabola_tangent_circle_l293_29343


namespace NUMINAMATH_CALUDE_input_statement_separator_l293_29368

/-- Represents the possible separators in an input statement -/
inductive Separator
  | Comma
  | Space
  | Semicolon
  | Pause

/-- Represents the general format of an input statement -/
structure InputStatement where
  separator : Separator

/-- The correct separator for multiple variables in an input statement -/
def correctSeparator : Separator := Separator.Comma

/-- Theorem stating that the correct separator in the general format of an input statement is a comma -/
theorem input_statement_separator :
  ∀ (stmt : InputStatement), stmt.separator = correctSeparator :=
sorry


end NUMINAMATH_CALUDE_input_statement_separator_l293_29368


namespace NUMINAMATH_CALUDE_object_speed_l293_29374

/-- An object traveling 10800 feet in one hour has a speed of 3 feet per second. -/
theorem object_speed (distance : ℝ) (time_in_seconds : ℝ) (h1 : distance = 10800) (h2 : time_in_seconds = 3600) :
  distance / time_in_seconds = 3 := by
  sorry

end NUMINAMATH_CALUDE_object_speed_l293_29374


namespace NUMINAMATH_CALUDE_taxi_fare_theorem_l293_29361

/-- Taxi fare function for distances greater than 5 kilometers -/
def taxi_fare (x : ℝ) : ℝ :=
  10 + 2 * 1.3 + 2.4 * (x - 5)

/-- Theorem stating the taxi fare function and its value for 6 kilometers -/
theorem taxi_fare_theorem (x : ℝ) (h : x > 5) :
  taxi_fare x = 2.4 * x + 0.6 ∧ taxi_fare 6 = 15 := by
  sorry

#check taxi_fare_theorem

end NUMINAMATH_CALUDE_taxi_fare_theorem_l293_29361


namespace NUMINAMATH_CALUDE_line_conditions_vector_at_zero_l293_29315

-- Define the line parameterization
def line_param (t : ℝ) : ℝ × ℝ := sorry

-- Define the conditions
theorem line_conditions :
  line_param 1 = (2, 5) ∧ line_param 4 = (11, -7) := sorry

-- Theorem to prove
theorem vector_at_zero :
  line_param 0 = (-1, 9) := by sorry

end NUMINAMATH_CALUDE_line_conditions_vector_at_zero_l293_29315


namespace NUMINAMATH_CALUDE_remainder_8_pow_2012_mod_10_l293_29342

/-- Definition of exponentiation --/
def pow (a : ℕ) (n : ℕ) : ℕ := (a : ℕ) ^ n

/-- The remainder when 8^2012 is divided by 10 --/
theorem remainder_8_pow_2012_mod_10 : pow 8 2012 % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_8_pow_2012_mod_10_l293_29342


namespace NUMINAMATH_CALUDE_find_x_l293_29344

theorem find_x : ∃ x : ℝ, 
  (3 + 7 + 10 + 15) / 4 = 2 * ((x + 20 + 6) / 3) ∧ 
  x = -12.875 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l293_29344


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l293_29334

/-- A number is composite if it has a factor other than 1 and itself -/
def IsComposite (n : ℕ) : Prop :=
  ∃ m, 1 < m ∧ m < n ∧ n % m = 0

/-- The theorem stating that 6n^2 is the largest divisor of n^4 - n^2 for all composite n -/
theorem largest_divisor_of_n4_minus_n2 (n : ℕ) (h : IsComposite n) :
  (∃ (k : ℕ), (n^4 - n^2) % (6 * n^2) = 0 ∧
    ∀ (m : ℕ), (n^4 - n^2) % m = 0 → m ≤ 6 * n^2) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l293_29334


namespace NUMINAMATH_CALUDE_color_change_probability_l293_29377

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  green_duration : ℕ
  yellow_duration : ℕ
  red_duration : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycle_duration (cycle : TrafficLightCycle) : ℕ :=
  cycle.green_duration + cycle.yellow_duration + cycle.red_duration

/-- Calculates the number of seconds where a color change can be observed in a 4-second interval -/
def change_observation_duration (cycle : TrafficLightCycle) : ℕ := 12

/-- Theorem: The probability of observing a color change during a random 4-second interval
    in the given traffic light cycle is 0.12 -/
theorem color_change_probability (cycle : TrafficLightCycle) 
  (h1 : cycle.green_duration = 45)
  (h2 : cycle.yellow_duration = 5)
  (h3 : cycle.red_duration = 50)
  (h4 : change_observation_duration cycle = 12) :
  (change_observation_duration cycle : ℚ) / (cycle_duration cycle) = 12 / 100 := by
  sorry

end NUMINAMATH_CALUDE_color_change_probability_l293_29377


namespace NUMINAMATH_CALUDE_unique_positive_number_l293_29345

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l293_29345


namespace NUMINAMATH_CALUDE_lowest_score_proof_l293_29335

theorem lowest_score_proof (scores : List ℝ) (highest lowest : ℝ) : 
  scores.length = 12 →
  scores.sum / scores.length = 82 →
  highest ∈ scores →
  lowest ∈ scores →
  highest = 98 →
  (scores.filter (λ x => x ≠ highest ∧ x ≠ lowest)).sum / 10 = 84 →
  lowest = 46 := by
sorry

end NUMINAMATH_CALUDE_lowest_score_proof_l293_29335


namespace NUMINAMATH_CALUDE_bear_mass_before_hibernation_l293_29395

/-- The mass of a bear after hibernation, given as a fraction of its original mass -/
def mass_after_hibernation_fraction : ℚ := 80 / 100

/-- The mass of the bear after hibernation in kilograms -/
def mass_after_hibernation : ℚ := 220

/-- Theorem: If a bear loses 20% of its original mass during hibernation and 
    its mass after hibernation is 220 kg, then its mass before hibernation was 275 kg -/
theorem bear_mass_before_hibernation :
  mass_after_hibernation = mass_after_hibernation_fraction * (275 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_bear_mass_before_hibernation_l293_29395


namespace NUMINAMATH_CALUDE_mr_johnson_class_size_l293_29385

def mrs_finley_class : ℕ := 24

def mr_johnson_class : ℕ := (mrs_finley_class / 2) + 10

theorem mr_johnson_class_size : mr_johnson_class = 22 := by
  sorry

end NUMINAMATH_CALUDE_mr_johnson_class_size_l293_29385


namespace NUMINAMATH_CALUDE_min_distance_ellipse_line_l293_29398

/-- The minimum distance between a point on the ellipse x²/8 + y²/4 = 1 
    and the line 4x + 3y - 24 = 0 is (24 - 2√41) / 5 -/
theorem min_distance_ellipse_line : 
  let ellipse := {p : ℝ × ℝ | p.1^2 / 8 + p.2^2 / 4 = 1}
  let line := {p : ℝ × ℝ | 4 * p.1 + 3 * p.2 - 24 = 0}
  ∃ (d : ℝ), d = (24 - 2 * Real.sqrt 41) / 5 ∧ 
    (∀ p ∈ ellipse, ∀ q ∈ line, d ≤ Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) ∧
    (∃ p ∈ ellipse, ∃ q ∈ line, d = Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)) := by
  sorry

end NUMINAMATH_CALUDE_min_distance_ellipse_line_l293_29398


namespace NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_of_570_l293_29353

theorem gcd_of_polynomial_and_multiple_of_570 (b : ℤ) : 
  (∃ k : ℤ, b = 570 * k) → Int.gcd (4 * b^3 + 2 * b^2 + 5 * b + 95) b = 95 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_polynomial_and_multiple_of_570_l293_29353


namespace NUMINAMATH_CALUDE_boys_in_class_l293_29333

theorem boys_in_class (total : ℕ) (girls_fraction : ℚ) (boys : ℕ) : 
  total = 160 → 
  girls_fraction = 1/4 → 
  boys = total - (girls_fraction * total).num → 
  boys = 120 := by
sorry

end NUMINAMATH_CALUDE_boys_in_class_l293_29333


namespace NUMINAMATH_CALUDE_different_color_probability_l293_29390

def total_chips : ℕ := 15
def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4

theorem different_color_probability : 
  let prob_blue_not_blue := (blue_chips : ℚ) / total_chips * ((red_chips + yellow_chips) : ℚ) / total_chips
  let prob_red_not_red := (red_chips : ℚ) / total_chips * ((blue_chips + yellow_chips) : ℚ) / total_chips
  let prob_yellow_not_yellow := (yellow_chips : ℚ) / total_chips * ((blue_chips + red_chips) : ℚ) / total_chips
  prob_blue_not_blue + prob_red_not_red + prob_yellow_not_yellow = 148 / 225 :=
by sorry

end NUMINAMATH_CALUDE_different_color_probability_l293_29390


namespace NUMINAMATH_CALUDE_product_of_decimals_l293_29365

theorem product_of_decimals (h : 268 * 74 = 19832) :
  2.68 * 0.74 = 1.9832 := by
  sorry

end NUMINAMATH_CALUDE_product_of_decimals_l293_29365


namespace NUMINAMATH_CALUDE_absolute_value_equals_cosine_roots_l293_29305

theorem absolute_value_equals_cosine_roots :
  ∃! (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (∀ x : ℝ, |x| = Real.cos x ↔ (x = a ∨ x = b ∨ x = c)) :=
sorry

end NUMINAMATH_CALUDE_absolute_value_equals_cosine_roots_l293_29305


namespace NUMINAMATH_CALUDE_road_trip_distance_ratio_l293_29354

theorem road_trip_distance_ratio : 
  ∀ (total_distance first_day_distance second_day_distance third_day_distance : ℝ),
  total_distance = 525 →
  first_day_distance = 200 →
  second_day_distance = 3/4 * first_day_distance →
  third_day_distance = total_distance - (first_day_distance + second_day_distance) →
  third_day_distance / (first_day_distance + second_day_distance) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_road_trip_distance_ratio_l293_29354


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l293_29388

/-- Given 50 observations with an initial mean, if one observation is corrected
    from 23 to 34, and the new mean becomes 36.5, then the initial mean must be 36.28. -/
theorem initial_mean_calculation (n : ℕ) (initial_mean corrected_mean : ℝ) :
  n = 50 ∧
  corrected_mean = 36.5 ∧
  (n : ℝ) * initial_mean + (34 - 23) = n * corrected_mean →
  initial_mean = 36.28 := by
  sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l293_29388


namespace NUMINAMATH_CALUDE_third_circle_radius_l293_29309

/-- Given two externally tangent circles with radii 2 and 5, 
    prove that a third circle tangent to both circles and their 
    common external tangent has a radius of (3 + √51) / 2. -/
theorem third_circle_radius 
  (A B O : ℝ × ℝ) -- Centers of the circles
  (r : ℝ) -- Radius of the third circle
  (h1 : ‖A - B‖ = 7) -- Distance between centers of first two circles
  (h2 : ‖O - A‖ = 2 + r) -- Distance between centers of first and third circles
  (h3 : ‖O - B‖ = 5 + r) -- Distance between centers of second and third circles
  (h4 : (O.1 - A.1)^2 + r^2 = (O.2 - A.2)^2) -- Third circle is tangent to common external tangent
  : r = (3 + Real.sqrt 51) / 2 := by
  sorry

end NUMINAMATH_CALUDE_third_circle_radius_l293_29309


namespace NUMINAMATH_CALUDE_scorpion_millipedes_l293_29316

/-- Calculates the number of millipedes needed to reach a daily segment goal -/
def millipedes_needed (daily_requirement : ℕ) (eaten_segments : ℕ) (remaining_millipede_segments : ℕ) : ℕ :=
  (daily_requirement - eaten_segments) / remaining_millipede_segments

theorem scorpion_millipedes :
  let daily_requirement : ℕ := 800
  let first_millipede_segments : ℕ := 60
  let long_millipede_segments : ℕ := 2 * first_millipede_segments
  let eaten_segments : ℕ := first_millipede_segments + 2 * long_millipede_segments
  let remaining_millipede_segments : ℕ := 50
  millipedes_needed daily_requirement eaten_segments remaining_millipede_segments = 10 := by
  sorry

end NUMINAMATH_CALUDE_scorpion_millipedes_l293_29316
