import Mathlib

namespace largest_B_divisible_by_three_l2513_251337

def seven_digit_number (B : ℕ) : ℕ := 4000000 + B * 100000 + 68251

theorem largest_B_divisible_by_three :
  ∀ B : ℕ, B ≤ 9 →
    (seven_digit_number B % 3 = 0) →
    B ≤ 7 ∧
    seven_digit_number 7 % 3 = 0 ∧
    (∀ C : ℕ, C > 7 → C ≤ 9 → seven_digit_number C % 3 ≠ 0) :=
by sorry

end largest_B_divisible_by_three_l2513_251337


namespace simplify_expression_l2513_251379

theorem simplify_expression (x y : ℝ) : 2 - (3 - (2 + (5 - (3*y - x)))) = 6 - 3*y + x := by
  sorry

end simplify_expression_l2513_251379


namespace smallest_natural_with_remainder_one_l2513_251327

theorem smallest_natural_with_remainder_one : ∃ n : ℕ, 
  n > 1 ∧ 
  n % 3 = 1 ∧ 
  n % 5 = 1 ∧ 
  (∀ m : ℕ, m > 1 ∧ m % 3 = 1 ∧ m % 5 = 1 → n ≤ m) ∧
  n = 16 := by
  sorry

end smallest_natural_with_remainder_one_l2513_251327


namespace sum_of_squares_lower_bound_l2513_251398

theorem sum_of_squares_lower_bound (a b c : ℝ) (h : a + b + c = 1) :
  a^2 + b^2 + c^2 ≥ 1/3 := by
  sorry

end sum_of_squares_lower_bound_l2513_251398


namespace arithmetic_progression_x_value_l2513_251358

def is_arithmetic_progression (a b c : ℝ) : Prop :=
  b - a = c - b

theorem arithmetic_progression_x_value :
  ∀ x : ℝ, is_arithmetic_progression (2*x - 3) (3*x - 2) (5*x + 2) → x = -3 :=
by
  sorry

end arithmetic_progression_x_value_l2513_251358


namespace john_soap_cost_l2513_251329

/-- The amount of money John spent on soap -/
def soap_cost (num_bars : ℕ) (weight_per_bar : ℚ) (price_per_pound : ℚ) : ℚ :=
  num_bars * weight_per_bar * price_per_pound

/-- Proof that John spent $15 on soap -/
theorem john_soap_cost :
  soap_cost 20 (3/2) (1/2) = 15 := by
  sorry

end john_soap_cost_l2513_251329


namespace negative_comparison_l2513_251315

theorem negative_comparison : -2023 > -2024 := by
  sorry

end negative_comparison_l2513_251315


namespace function_form_l2513_251361

/-- Given a function g: ℝ → ℝ satisfying certain conditions, prove it has a specific form. -/
theorem function_form (g : ℝ → ℝ) 
  (h1 : g 2 = 2)
  (h2 : ∀ x y : ℝ, g (x + y) = 5^y * g x + 3^x * g y) :
  ∀ x : ℝ, g x = (5^x - 3^x) / 8 := by
  sorry

end function_form_l2513_251361


namespace power_sum_of_i_l2513_251335

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^66 + i^103 = -1 - i := by sorry

end power_sum_of_i_l2513_251335


namespace vessel_capacity_l2513_251305

/-- The capacity of the vessel in litres -/
def C : ℝ := 60.01

/-- The amount of liquid removed and replaced with water each time, in litres -/
def removed : ℝ := 9

/-- The amount of pure milk in the final solution, in litres -/
def final_milk : ℝ := 43.35

/-- Theorem stating that the capacity of the vessel is 60.01 litres -/
theorem vessel_capacity :
  (C - removed) * (C - removed) / C = final_milk :=
sorry

end vessel_capacity_l2513_251305


namespace symmetry_implies_f_3_equals_1_l2513_251351

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the symmetry condition
def symmetric_about_y_equals_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f (x - 1) = y ↔ g y = x

-- State the theorem
theorem symmetry_implies_f_3_equals_1
  (h_sym : symmetric_about_y_equals_x f g)
  (h_g : g 1 = 2) :
  f 3 = 1 := by
  sorry

end symmetry_implies_f_3_equals_1_l2513_251351


namespace service_center_location_l2513_251352

/-- Represents a highway with exits and a service center -/
structure Highway where
  third_exit : ℝ
  tenth_exit : ℝ
  service_center : ℝ

/-- Theorem: Given a highway with the third exit at milepost 50 and the tenth exit at milepost 170,
    a service center located two-thirds of the way from the third exit to the tenth exit
    is at milepost 130. -/
theorem service_center_location (h : Highway)
  (h_third : h.third_exit = 50)
  (h_tenth : h.tenth_exit = 170)
  (h_service : h.service_center = h.third_exit + 2 / 3 * (h.tenth_exit - h.third_exit)) :
  h.service_center = 130 := by
  sorry

end service_center_location_l2513_251352


namespace youtube_ad_time_l2513_251389

/-- Calculates the time spent watching ads on Youtube --/
def time_watching_ads (videos_per_day : ℕ) (video_duration : ℕ) (total_time : ℕ) : ℕ :=
  total_time - (videos_per_day * video_duration)

/-- Theorem: The time spent watching ads is 3 minutes --/
theorem youtube_ad_time :
  time_watching_ads 2 7 17 = 3 := by
  sorry

end youtube_ad_time_l2513_251389


namespace product_sum_equality_l2513_251348

theorem product_sum_equality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b * c = 1) (h2 : a + 1 / c = 7) (h3 : b + 1 / a = 31) :
  c + 1 / b = 5 / 27 := by
  sorry

end product_sum_equality_l2513_251348


namespace a_minus_b_eq_neg_seven_l2513_251394

theorem a_minus_b_eq_neg_seven
  (h1 : Real.sqrt (a ^ 2) = 3)
  (h2 : Real.sqrt b = 2)
  (h3 : a * b < 0)
  : a - b = -7 := by
  sorry

end a_minus_b_eq_neg_seven_l2513_251394


namespace astros_win_in_seven_l2513_251332

/-- The probability of the Dodgers winning a single game -/
def p_dodgers : ℚ := 3/4

/-- The probability of the Astros winning a single game -/
def p_astros : ℚ := 1 - p_dodgers

/-- The number of games needed to win the World Series -/
def games_to_win : ℕ := 4

/-- The total number of games in a full World Series -/
def total_games : ℕ := 2 * games_to_win - 1

/-- The probability of the Astros winning the World Series in exactly 7 games -/
def p_astros_win_in_seven : ℚ := 135/4096

theorem astros_win_in_seven :
  p_astros_win_in_seven = (Nat.choose 6 3 : ℚ) * p_astros^3 * p_dodgers^3 * p_astros := by sorry

end astros_win_in_seven_l2513_251332


namespace arithmetic_sequence_ratio_l2513_251339

/-- Two arithmetic sequences and their sum sequences -/
structure ArithmeticSequencePair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  S : ℕ → ℚ  -- Sum sequence for a
  T : ℕ → ℚ  -- Sum sequence for b

/-- The main theorem -/
theorem arithmetic_sequence_ratio 
  (seq : ArithmeticSequencePair)
  (h_sum_ratio : ∀ n : ℕ, seq.S n / seq.T n = 2 * n / (3 * n + 1)) :
  seq.a 10 / seq.b 10 = 19 / 29 := by
  sorry

end arithmetic_sequence_ratio_l2513_251339


namespace ratio_of_x_intercepts_l2513_251340

/-- Given two lines with the same non-zero y-intercept, where the first line has slope 12
    and x-intercept (u, 0), and the second line has slope 8 and x-intercept (v, 0),
    prove that the ratio of u to v is 2/3. -/
theorem ratio_of_x_intercepts (b : ℝ) (u v : ℝ) (h1 : b ≠ 0)
    (h2 : 12 * u + b = 0) (h3 : 8 * v + b = 0) : u / v = 2 / 3 := by
  sorry

end ratio_of_x_intercepts_l2513_251340


namespace no_four_digit_numbers_divisible_by_11_sum_10_l2513_251393

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  a_nonzero : a ≠ 0
  a_digit : a < 10
  b_digit : b < 10
  c_digit : c < 10
  d_digit : d < 10

/-- Checks if a four-digit number is divisible by 11 -/
def isDivisibleBy11 (n : FourDigitNumber) : Prop :=
  (1000 * n.a + 100 * n.b + 10 * n.c + n.d) % 11 = 0

/-- Checks if the sum of digits of a four-digit number is 10 -/
def sumOfDigitsIs10 (n : FourDigitNumber) : Prop :=
  n.a + n.b + n.c + n.d = 10

/-- Theorem: There are no four-digit numbers divisible by 11 with digits summing to 10 -/
theorem no_four_digit_numbers_divisible_by_11_sum_10 :
  ¬ ∃ (n : FourDigitNumber), isDivisibleBy11 n ∧ sumOfDigitsIs10 n := by
  sorry

end no_four_digit_numbers_divisible_by_11_sum_10_l2513_251393


namespace overlapping_circles_area_l2513_251306

/-- The area of a figure consisting of two overlapping circles -/
theorem overlapping_circles_area (r1 r2 : ℝ) (overlap_area : ℝ) :
  r1 = 4 →
  r2 = 6 →
  overlap_area = 2 * Real.pi →
  (Real.pi * r1^2) + (Real.pi * r2^2) - overlap_area = 50 * Real.pi :=
by sorry

end overlapping_circles_area_l2513_251306


namespace third_to_second_ratio_l2513_251349

/-- The heights of four buildings satisfy certain conditions -/
structure BuildingHeights where
  h1 : ℝ  -- Height of the tallest building
  h2 : ℝ  -- Height of the second tallest building
  h3 : ℝ  -- Height of the third tallest building
  h4 : ℝ  -- Height of the fourth tallest building
  tallest : h1 = 100
  second_tallest : h2 = h1 / 2
  fourth_tallest : h4 = h3 / 5
  total_height : h1 + h2 + h3 + h4 = 180

/-- The ratio of the third tallest to the second tallest building is 1:2 -/
theorem third_to_second_ratio (b : BuildingHeights) : b.h3 / b.h2 = 1 / 2 := by
  sorry

end third_to_second_ratio_l2513_251349


namespace f_pow_ten_l2513_251373

/-- f(n) is the number of ones that occur in the decimal representations of all the numbers from 1 to n -/
def f (n : ℕ) : ℕ := sorry

/-- Theorem: For any natural number k, f(10^k) = k * 10^(k-1) + 1 -/
theorem f_pow_ten (k : ℕ) : f (10^k) = k * 10^(k-1) + 1 := by sorry

end f_pow_ten_l2513_251373


namespace greatest_divisor_with_remainders_l2513_251331

theorem greatest_divisor_with_remainders : 
  let a := 1657 - 6
  let b := 2037 - 5
  Nat.gcd a b = 127 := by sorry

end greatest_divisor_with_remainders_l2513_251331


namespace ratio_10_20_percent_l2513_251382

/-- The percent value of a ratio a:b is defined as (a/b) * 100 -/
def percent_value (a b : ℚ) : ℚ := (a / b) * 100

/-- The ratio 10:20 expressed as a percent is 50% -/
theorem ratio_10_20_percent : percent_value 10 20 = 50 := by
  sorry

end ratio_10_20_percent_l2513_251382


namespace race_finish_time_difference_l2513_251324

/-- Calculates the time difference at the finish line between two runners in a race -/
theorem race_finish_time_difference 
  (race_distance : ℝ) 
  (alice_speed : ℝ) 
  (bob_speed : ℝ) 
  (h1 : race_distance = 15) 
  (h2 : alice_speed = 7) 
  (h3 : bob_speed = 9) : 
  bob_speed * race_distance - alice_speed * race_distance = 30 := by
  sorry

#check race_finish_time_difference

end race_finish_time_difference_l2513_251324


namespace ant_colony_problem_l2513_251383

theorem ant_colony_problem (x y : ℕ) :
  x + y = 40 →
  64 * x + 729 * y = 8748 →
  64 * x = 1984 :=
by
  sorry

end ant_colony_problem_l2513_251383


namespace antelopes_count_l2513_251326

/-- Represents the count of animals on a safari --/
structure SafariCount where
  antelopes : ℕ
  rabbits : ℕ
  hyenas : ℕ
  wild_dogs : ℕ
  leopards : ℕ

/-- Conditions for the safari animal count --/
def safari_conditions (count : SafariCount) : Prop :=
  count.rabbits = count.antelopes + 34 ∧
  count.hyenas = count.antelopes + count.rabbits - 42 ∧
  count.wild_dogs = count.hyenas + 50 ∧
  count.leopards * 2 = count.rabbits ∧
  count.antelopes + count.rabbits + count.hyenas + count.wild_dogs + count.leopards = 605

/-- The theorem stating that the number of antelopes is 80 --/
theorem antelopes_count (count : SafariCount) :
  safari_conditions count → count.antelopes = 80 := by
  sorry

end antelopes_count_l2513_251326


namespace crayons_difference_l2513_251356

/-- Given the initial number of crayons, the number of crayons given away, and the number of crayons lost,
    prove that the difference between crayons given away and crayons lost is 410. -/
theorem crayons_difference (initial : ℕ) (given_away : ℕ) (lost : ℕ)
  (h1 : initial = 589)
  (h2 : given_away = 571)
  (h3 : lost = 161) :
  given_away - lost = 410 := by
  sorry

end crayons_difference_l2513_251356


namespace gcd_of_324_243_135_l2513_251309

theorem gcd_of_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by
  sorry

end gcd_of_324_243_135_l2513_251309


namespace trees_in_column_l2513_251399

/-- Proves the number of trees in one column of Jack's grove --/
theorem trees_in_column (trees_per_row : ℕ) (cleaning_time_per_tree : ℕ) (total_cleaning_time : ℕ) 
  (h1 : trees_per_row = 4)
  (h2 : cleaning_time_per_tree = 3)
  (h3 : total_cleaning_time = 60)
  (h4 : total_cleaning_time / cleaning_time_per_tree = trees_per_row * (total_cleaning_time / cleaning_time_per_tree / trees_per_row)) :
  total_cleaning_time / cleaning_time_per_tree / trees_per_row = 5 := by
  sorry

#check trees_in_column

end trees_in_column_l2513_251399


namespace selina_sold_two_shirts_l2513_251300

/-- Represents the store credit and pricing system --/
structure StoreCredit where
  pants_credit : ℕ
  shorts_credit : ℕ
  shirt_credit : ℕ
  jacket_credit : ℕ

/-- Represents the items Selina sold --/
structure ItemsSold where
  pants : ℕ
  shorts : ℕ
  jackets : ℕ

/-- Represents the items Selina purchased --/
structure ItemsPurchased where
  shirt1_price : ℕ
  shirt2_price : ℕ
  pants_price : ℕ

/-- Calculates the total store credit for non-shirt items --/
def nonShirtCredit (sc : StoreCredit) (is : ItemsSold) : ℕ :=
  sc.pants_credit * is.pants + sc.shorts_credit * is.shorts + sc.jacket_credit * is.jackets

/-- Calculates the total price of purchased items --/
def totalPurchasePrice (ip : ItemsPurchased) : ℕ :=
  ip.shirt1_price + ip.shirt2_price + ip.pants_price

/-- Applies discount and tax to the purchase price --/
def finalPurchasePrice (price : ℕ) (discount : ℚ) (tax : ℚ) : ℚ :=
  (price : ℚ) * (1 - discount) * (1 + tax)

/-- Main theorem: Proves that Selina sold 2 shirts --/
theorem selina_sold_two_shirts 
  (sc : StoreCredit)
  (is : ItemsSold)
  (ip : ItemsPurchased)
  (discount : ℚ)
  (tax : ℚ)
  (remaining_credit : ℕ)
  (h1 : sc = ⟨5, 3, 4, 7⟩)
  (h2 : is = ⟨3, 5, 2⟩)
  (h3 : ip = ⟨10, 12, 15⟩)
  (h4 : discount = 1/10)
  (h5 : tax = 1/20)
  (h6 : remaining_credit = 25) :
  ∃ (shirts_sold : ℕ), shirts_sold = 2 ∧
    (nonShirtCredit sc is + sc.shirt_credit * shirts_sold : ℚ) =
    finalPurchasePrice (totalPurchasePrice ip) discount tax + remaining_credit :=
sorry

end selina_sold_two_shirts_l2513_251300


namespace inequality_proof_l2513_251330

theorem inequality_proof (p : ℝ) (x y z v : ℝ) 
  (hp : p ≥ 2) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hv : v ≥ 0) :
  (x + y)^p + (z + v)^p + (x + z)^p + (y + v)^p ≤ 
  x^p + y^p + z^p + v^p + (x + y + z + v)^p :=
by sorry

end inequality_proof_l2513_251330


namespace four_people_three_rooms_l2513_251371

/-- The number of ways to distribute n people into k non-empty rooms -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem four_people_three_rooms :
  distribute 4 3 = 36 :=
by
  sorry

end four_people_three_rooms_l2513_251371


namespace solution_set_implies_a_value_l2513_251343

def f (a x : ℝ) : ℝ := |x + 1| + |x - a|

theorem solution_set_implies_a_value (a : ℝ) (h1 : a > 0) :
  (∀ x : ℝ, f a x ≥ 5 ↔ x ≤ -2 ∨ x > 3) → a = 2 := by
  sorry

end solution_set_implies_a_value_l2513_251343


namespace magnitude_of_Z_l2513_251311

def Z : ℂ := Complex.mk 3 (-4)

theorem magnitude_of_Z : Complex.abs Z = 5 := by
  sorry

end magnitude_of_Z_l2513_251311


namespace min_value_of_function_l2513_251333

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  let f := fun x => 4 * x + 2 / x
  (∀ y > 0, f y ≥ 4 * Real.sqrt 2) ∧ (∃ y > 0, f y = 4 * Real.sqrt 2) := by
sorry

end min_value_of_function_l2513_251333


namespace permutation_inequality_l2513_251341

theorem permutation_inequality (a₁ a₂ a₃ a₄ a₅ a₆ : ℕ) : 
  a₁ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₂ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₃ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₄ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₅ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₆ ∈ ({1, 2, 3, 4, 5, 6} : Finset ℕ) →
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₁ ≠ a₅ ∧ a₁ ≠ a₆ ∧
  a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₂ ≠ a₅ ∧ a₂ ≠ a₆ ∧
  a₃ ≠ a₄ ∧ a₃ ≠ a₅ ∧ a₃ ≠ a₆ ∧
  a₄ ≠ a₅ ∧ a₄ ≠ a₆ ∧
  a₅ ≠ a₆ →
  (a₁ + 1) / 2 * (a₂ + 2) / 2 * (a₃ + 3) / 2 * (a₄ + 4) / 2 * (a₅ + 5) / 2 * (a₆ + 6) / 2 < 40320 := by
  sorry

end permutation_inequality_l2513_251341


namespace max_dimes_count_l2513_251365

/-- Represents the value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- Represents the value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- Represents the total amount of money Liam has in dollars -/
def total_money : ℚ := 4.80

/-- 
Given that Liam has $4.80 in U.S. coins and an equal number of dimes and nickels,
this theorem states that the maximum number of dimes he could have is 32.
-/
theorem max_dimes_count : 
  ∃ (d : ℕ), d * (dime_value + nickel_value) = total_money ∧ 
             ∀ (x : ℕ), x * (dime_value + nickel_value) ≤ total_money → x ≤ d :=
by sorry

end max_dimes_count_l2513_251365


namespace cave_depth_calculation_l2513_251328

theorem cave_depth_calculation (total_depth remaining_distance : ℕ) 
  (h1 : total_depth = 974)
  (h2 : remaining_distance = 386) :
  total_depth - remaining_distance = 588 := by
sorry

end cave_depth_calculation_l2513_251328


namespace inequality_solution_set_l2513_251359

theorem inequality_solution_set (x : ℝ) :
  (1 / x < 2 ∧ 1 / x > -3) ↔ (x > 1 / 2 ∨ x < -1 / 3) := by sorry

end inequality_solution_set_l2513_251359


namespace problem_statement_l2513_251357

-- Define the line x + y - 3 = 0
def line1 (x y : ℝ) : Prop := x + y - 3 = 0

-- Define the vector (1, -1)
def vector1 : ℝ × ℝ := (1, -1)

-- Define the lines x + 2y - 4 = 0 and 2x + 4y + 1 = 0
def line2 (x y : ℝ) : Prop := x + 2*y - 4 = 0
def line3 (x y : ℝ) : Prop := 2*x + 4*y + 1 = 0

-- Define the point (3, 4)
def point1 : ℝ × ℝ := (3, 4)

-- Define a function to check if a line has equal intercepts on both axes
def has_equal_intercepts (a b c : ℝ) : Prop :=
  ∃ (t : ℝ), a * t + b * t + c = 0 ∧ t ≠ 0

theorem problem_statement :
  -- 1. (1,-1) is a directional vector of the line x+y-3=0
  (∀ (t : ℝ), line1 (vector1.1 * t) (vector1.2 * t)) ∧
  -- 2. The distance between lines x+2y-4=0 and 2x+4y+1=0 is 9√5/10
  (let d := (9 * Real.sqrt 5) / 10;
   ∀ (x y : ℝ), line2 x y → ∀ (x' y' : ℝ), line3 x' y' →
   ((x - x')^2 + (y - y')^2).sqrt = d) ∧
  -- 3. There are exactly 2 lines passing through point (3,4) with equal intercepts on the two coordinate axes
  (∃! (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    a₁ * point1.1 + b₁ * point1.2 + c₁ = 0 ∧
    a₂ * point1.1 + b₂ * point1.2 + c₂ = 0 ∧
    has_equal_intercepts a₁ b₁ c₁ ∧
    has_equal_intercepts a₂ b₂ c₂ ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂)) :=
sorry

end problem_statement_l2513_251357


namespace choose_captains_l2513_251354

theorem choose_captains (n : ℕ) (k : ℕ) (h1 : n = 15) (h2 : k = 4) :
  Nat.choose n k = 1365 := by
  sorry

end choose_captains_l2513_251354


namespace angle_A_value_side_a_range_l2513_251362

/-- Represents an acute triangle with sides a, b, c opposite to angles A, B, C respectively. -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2
  sum_angles : A + B + C = π

theorem angle_A_value (t : AcuteTriangle) :
  Real.cos (2 * t.A) - Real.cos (2 * t.B) + 2 * Real.cos (π/6 - t.B) * Real.cos (π/6 + t.B) = 0 →
  t.A = π/3 := by sorry

theorem side_a_range (t : AcuteTriangle) :
  t.b = Real.sqrt 3 → t.b ≤ t.a → t.A = π/3 →
  t.a ≥ Real.sqrt 3 ∧ t.a < 3 := by sorry

end angle_A_value_side_a_range_l2513_251362


namespace geometric_series_common_ratio_l2513_251310

/-- The common ratio of the infinite geometric series (-4/7) + (14/3) + (-98/9) + ... -/
def common_ratio : ℚ := -49/6

/-- The first term of the geometric series -/
def a₁ : ℚ := -4/7

/-- The second term of the geometric series -/
def a₂ : ℚ := 14/3

/-- The third term of the geometric series -/
def a₃ : ℚ := -98/9

theorem geometric_series_common_ratio :
  (a₂ / a₁ = common_ratio) ∧ (a₃ / a₂ = common_ratio) :=
sorry

end geometric_series_common_ratio_l2513_251310


namespace half_abs_diff_squares_20_15_l2513_251395

theorem half_abs_diff_squares_20_15 : (1/2 : ℝ) * |20^2 - 15^2| = 87.5 := by
  sorry

end half_abs_diff_squares_20_15_l2513_251395


namespace min_value_of_expression_l2513_251378

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y - 3 = 0) :
  ∀ z, z = 2*x + y → z ≥ 3 ∧ ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 + 2*x₀*y₀ - 3 = 0 ∧ 2*x₀ + y₀ = 3 :=
sorry

end min_value_of_expression_l2513_251378


namespace invalid_prism_diagonals_l2513_251364

/-- Represents the lengths of the extended diagonals of a right regular prism -/
structure PrismDiagonals where
  d1 : ℝ
  d2 : ℝ
  d3 : ℝ

/-- Checks if the given lengths can be the extended diagonals of a right regular prism -/
def is_valid_prism_diagonals (d : PrismDiagonals) : Prop :=
  ∃ (a b c : ℝ),
    (a > 0 ∧ b > 0 ∧ c > 0) ∧
    (d.d1^2 = a^2 + b^2 ∧ d.d2^2 = b^2 + c^2 ∧ d.d3^2 = a^2 + c^2)

/-- The main theorem stating that {3, 4, 6} cannot be the lengths of extended diagonals -/
theorem invalid_prism_diagonals :
  ¬ is_valid_prism_diagonals ⟨3, 4, 6⟩ :=
sorry

end invalid_prism_diagonals_l2513_251364


namespace marbles_lost_l2513_251360

theorem marbles_lost (initial : ℕ) (current : ℕ) (lost : ℕ) 
  (h1 : initial = 19) 
  (h2 : current = 8) 
  (h3 : lost = initial - current) : lost = 11 := by
  sorry

end marbles_lost_l2513_251360


namespace five_digit_multiples_of_five_l2513_251307

theorem five_digit_multiples_of_five : 
  (Finset.filter (fun n => n % 5 = 0) (Finset.range 90000)).card = 18000 :=
by sorry

end five_digit_multiples_of_five_l2513_251307


namespace inheritance_tax_calculation_l2513_251312

theorem inheritance_tax_calculation (inheritance : ℝ) 
  (federal_tax_rate : ℝ) (state_tax_rate : ℝ) (total_tax : ℝ) : 
  inheritance = 38600 →
  federal_tax_rate = 0.25 →
  state_tax_rate = 0.15 →
  total_tax = 14000 →
  total_tax = inheritance * federal_tax_rate + 
    (inheritance - inheritance * federal_tax_rate) * state_tax_rate :=
by sorry

end inheritance_tax_calculation_l2513_251312


namespace sugar_percentage_in_kola_solution_l2513_251342

/-- Calculates the percentage of sugar in a kola solution after adding ingredients -/
theorem sugar_percentage_in_kola_solution
  (initial_volume : ℝ)
  (initial_water_percent : ℝ)
  (initial_kola_percent : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percent = 88)
  (h3 : initial_kola_percent = 5)
  (h4 : added_sugar = 3.2)
  (h5 : added_water = 10)
  (h6 : added_kola = 6.8) :
  let initial_sugar_percent := 100 - initial_water_percent - initial_kola_percent
  let initial_sugar_volume := initial_sugar_percent / 100 * initial_volume
  let final_sugar_volume := initial_sugar_volume + added_sugar
  let final_volume := initial_volume + added_sugar + added_water + added_kola
  let final_sugar_percent := final_sugar_volume / final_volume * 100
  final_sugar_percent = 7.5 := by
sorry

end sugar_percentage_in_kola_solution_l2513_251342


namespace reciprocal_problem_l2513_251317

theorem reciprocal_problem (x : ℝ) (h : 8 * x = 4) : 200 * (1 / x) = 400 := by
  sorry

end reciprocal_problem_l2513_251317


namespace quadratic_expression_value_l2513_251377

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 17) 
  (eq2 : x + 4 * y = 23) : 
  17 * x^2 + 34 * x * y + 17 * y^2 = 818 := by
sorry

end quadratic_expression_value_l2513_251377


namespace five_solutions_for_f_f_eq_seven_l2513_251367

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ -2 then x^2 - 6 else x + 5

theorem five_solutions_for_f_f_eq_seven :
  ∃! (s : Finset ℝ), s.card = 5 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 7 :=
sorry

end five_solutions_for_f_f_eq_seven_l2513_251367


namespace property_rent_calculation_l2513_251370

theorem property_rent_calculation (purchase_price : ℝ) (maintenance_rate : ℝ) 
  (annual_tax : ℝ) (target_return_rate : ℝ) (monthly_rent : ℝ) : 
  purchase_price = 12000 ∧ 
  maintenance_rate = 0.15 ∧ 
  annual_tax = 400 ∧ 
  target_return_rate = 0.06 ∧ 
  monthly_rent = 109.80 →
  monthly_rent * 12 * (1 - maintenance_rate) = 
    purchase_price * target_return_rate + annual_tax :=
by
  sorry

#check property_rent_calculation

end property_rent_calculation_l2513_251370


namespace quadratic_inequality_solution_set_l2513_251302

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - x - 2 < 0} = Set.Ioo (-1 : ℝ) 2 := by
  sorry

end quadratic_inequality_solution_set_l2513_251302


namespace part_one_part_two_l2513_251375

-- Define the function y
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part 1
theorem part_one :
  (∀ x : ℝ, y m x < 0) ↔ m ∈ Set.Ioc (-4) 0 :=
sorry

-- Part 2
theorem part_two :
  (∀ x ∈ Set.Icc 1 3, y m x < -m + 5) ↔ m ∈ Set.Iio (6/7) :=
sorry

end part_one_part_two_l2513_251375


namespace final_number_calculation_l2513_251314

theorem final_number_calculation : ∃ (n : ℕ), n = 5 ∧ (3 * ((2 * n) + 9) = 57) := by
  sorry

end final_number_calculation_l2513_251314


namespace mrs_copper_class_size_l2513_251319

theorem mrs_copper_class_size :
  ∀ (initial_jellybeans : ℕ) 
    (absent_children : ℕ) 
    (jellybeans_per_child : ℕ) 
    (remaining_jellybeans : ℕ),
  initial_jellybeans = 100 →
  absent_children = 2 →
  jellybeans_per_child = 3 →
  remaining_jellybeans = 34 →
  ∃ (total_children : ℕ),
    total_children = 
      (initial_jellybeans - remaining_jellybeans) / jellybeans_per_child + absent_children ∧
    total_children = 24 :=
by
  sorry

end mrs_copper_class_size_l2513_251319


namespace cube_root_three_equation_l2513_251318

theorem cube_root_three_equation (s : ℝ) : s = 1 / (2 - (3 : ℝ)^(1/3)) → s = 2 + (3 : ℝ)^(1/3) := by
  sorry

end cube_root_three_equation_l2513_251318


namespace increasing_function_inequality_l2513_251391

theorem increasing_function_inequality (f : ℝ → ℝ) 
  (h_cont : ContinuousOn f (Set.Icc 0 1))
  (h_deriv : ∀ x ∈ Set.Ioo 0 1, DifferentiableAt ℝ f x ∧ deriv f x > 0) :
  f 1 > f 0 := by
  sorry

end increasing_function_inequality_l2513_251391


namespace torn_sheets_count_l2513_251396

/-- Represents a book with consecutively numbered pages. -/
structure Book where
  first_torn_page : Nat
  last_torn_page : Nat

/-- Checks if two numbers have the same digits. -/
def same_digits (a b : Nat) : Prop :=
  sorry

/-- Calculates the number of torn sheets given a Book. -/
def torn_sheets (book : Book) : Nat :=
  (book.last_torn_page - book.first_torn_page + 1) / 2

/-- The main theorem stating the number of torn sheets. -/
theorem torn_sheets_count (book : Book) :
    book.first_torn_page = 185
  → same_digits book.first_torn_page book.last_torn_page
  → Even book.last_torn_page
  → book.last_torn_page > book.first_torn_page
  → torn_sheets book = 167 := by
  sorry

end torn_sheets_count_l2513_251396


namespace sin_tan_greater_than_square_l2513_251336

theorem sin_tan_greater_than_square (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) : 
  Real.sin x * Real.tan x > x^2 := by
  sorry

end sin_tan_greater_than_square_l2513_251336


namespace petes_diner_cost_theorem_l2513_251308

/-- Represents the cost calculation at Pete's Diner -/
def PetesDinerCost (burgerPrice juicePrice discountAmount : ℕ) 
                   (discountThreshold : ℕ) 
                   (burgerCount juiceCount : ℕ) : ℕ :=
  let totalItems := burgerCount + juiceCount
  let subtotal := burgerCount * burgerPrice + juiceCount * juicePrice
  if totalItems > discountThreshold then subtotal - discountAmount else subtotal

/-- Proves that the total cost of 7 burgers and 5 juices at Pete's Diner is 38 dollars -/
theorem petes_diner_cost_theorem : 
  PetesDinerCost 4 3 5 10 7 5 = 38 := by
  sorry

#eval PetesDinerCost 4 3 5 10 7 5

end petes_diner_cost_theorem_l2513_251308


namespace puppy_food_consumption_l2513_251303

/-- Represents the feeding schedule for a puppy over 4 weeks plus one day -/
structure PuppyFeeding where
  first_two_weeks : ℚ  -- Amount of food per day in first two weeks
  second_two_weeks : ℚ  -- Amount of food per day in second two weeks
  today : ℚ  -- Amount of food given today

/-- Calculates the total amount of food eaten by the puppy over 4 weeks plus one day -/
def total_food (feeding : PuppyFeeding) : ℚ :=
  feeding.first_two_weeks * 14 + feeding.second_two_weeks * 14 + feeding.today

/-- Theorem stating that the puppy will eat 25 cups of food over 4 weeks plus one day -/
theorem puppy_food_consumption :
  let feeding := PuppyFeeding.mk (3/4) 1 (1/2)
  total_food feeding = 25 := by sorry

end puppy_food_consumption_l2513_251303


namespace new_years_party_assignments_l2513_251374

/-- The number of ways to assign teachers to classes -/
def assignTeachers (totalTeachers : ℕ) (numClasses : ℕ) (maxPerClass : ℕ) : ℕ := sorry

/-- Theorem stating the correct number of assignments for the given conditions -/
theorem new_years_party_assignments :
  assignTeachers 6 2 4 = 50 := by sorry

end new_years_party_assignments_l2513_251374


namespace a_equals_2a_is_valid_assignment_l2513_251369

/-- Definition of a valid assignment statement -/
def is_valid_assignment (stmt : String) : Prop :=
  ∃ (var : String) (expr : String),
    stmt = var ++ " = " ++ expr ∧
    var.length > 0 ∧
    (∀ c, c ∈ var.data → c.isAlpha)

/-- The statement "a = 2*a" is a valid assignment -/
theorem a_equals_2a_is_valid_assignment :
  is_valid_assignment "a = 2*a" := by
  sorry

#check a_equals_2a_is_valid_assignment

end a_equals_2a_is_valid_assignment_l2513_251369


namespace quadratic_is_perfect_square_l2513_251347

theorem quadratic_is_perfect_square (x : ℝ) : 
  ∃ (a : ℝ), x^2 - 20*x + 100 = (x + a)^2 := by
  sorry

end quadratic_is_perfect_square_l2513_251347


namespace jennifer_dogs_count_l2513_251368

/-- The number of dogs Jennifer has -/
def number_of_dogs : ℕ := 2

/-- Time in minutes to groom each dog -/
def grooming_time_per_dog : ℕ := 20

/-- Number of days Jennifer grooms her dogs -/
def grooming_days : ℕ := 30

/-- Total time in hours Jennifer spends grooming in 30 days -/
def total_grooming_time_hours : ℕ := 20

theorem jennifer_dogs_count :
  number_of_dogs * grooming_time_per_dog * grooming_days = total_grooming_time_hours * 60 :=
by sorry

end jennifer_dogs_count_l2513_251368


namespace number_count_proof_l2513_251363

theorem number_count_proof (total_avg : ℝ) (group1_avg : ℝ) (group2_avg : ℝ) (group3_avg : ℝ) :
  total_avg = 2.5 →
  group1_avg = 1.1 →
  group2_avg = 1.4 →
  group3_avg = 5 →
  ∃ (n : ℕ), n = 6 ∧ 
    n * total_avg = 2 * group1_avg + 2 * group2_avg + 2 * group3_avg :=
by sorry

end number_count_proof_l2513_251363


namespace intersection_S_T_l2513_251380

def S : Set ℝ := {x | x + 1 ≥ 2}
def T : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_S_T : S ∩ T = {1, 2} := by
  sorry

end intersection_S_T_l2513_251380


namespace rectangular_field_width_l2513_251301

/-- The width of a rectangular field given its length-to-width ratio and perimeter -/
def field_width (length_width_ratio : ℚ) (perimeter : ℚ) : ℚ :=
  perimeter / (2 * (length_width_ratio + 1))

theorem rectangular_field_width :
  field_width (7/5) 288 = 60 := by
  sorry

end rectangular_field_width_l2513_251301


namespace flour_amount_second_combination_l2513_251346

/-- The cost per pound of sugar and flour -/
def cost_per_pound : ℝ := 0.45

/-- The total cost of both combinations -/
def total_cost : ℝ := 26

/-- The amount of sugar in the first combination -/
def sugar_amount_1 : ℝ := 40

/-- The amount of flour in the first combination -/
def flour_amount_1 : ℝ := 16

/-- The amount of sugar in the second combination -/
def sugar_amount_2 : ℝ := 30

/-- The amount of flour in the second combination -/
def flour_amount_2 : ℝ := 28

theorem flour_amount_second_combination :
  sugar_amount_1 * cost_per_pound + flour_amount_1 * cost_per_pound = total_cost ∧
  sugar_amount_2 * cost_per_pound + flour_amount_2 * cost_per_pound = total_cost :=
by sorry

end flour_amount_second_combination_l2513_251346


namespace prime_factors_of_x_l2513_251390

theorem prime_factors_of_x (x : ℕ) 
  (h1 : x % 44 = 0 ∧ x / 44 = 432)
  (h2 : x % 31 = 5)
  (h3 : ∃ (a b c : ℕ), Prime a ∧ Prime b ∧ Prime c ∧ x = a^3 * b^2 * c) :
  ∃ (a b c : ℕ), a = 3 ∧ b = 4 ∧ c = 7 ∧ Prime a ∧ Prime b ∧ Prime c ∧ x = a^3 * b^2 * c :=
by sorry

end prime_factors_of_x_l2513_251390


namespace diagonal_four_sides_squared_l2513_251385

/-- A regular nonagon -/
structure RegularNonagon where
  /-- The length of a side -/
  a : ℝ
  /-- The length of a diagonal that jumps over four sides -/
  d : ℝ
  /-- Ensure the side length is positive -/
  a_pos : a > 0

/-- In a regular nonagon, the square of the length of a diagonal that jumps over four sides
    is equal to five times the square of the side length -/
theorem diagonal_four_sides_squared (n : RegularNonagon) : n.d^2 = 5 * n.a^2 := by
  sorry

end diagonal_four_sides_squared_l2513_251385


namespace circle_c_equation_l2513_251376

/-- A circle C with center on y = x^2, passing through origin, and intercepting 8 units on y-axis -/
structure CircleC where
  a : ℝ
  center : ℝ × ℝ
  center_on_parabola : center.2 = center.1^2
  passes_through_origin : (0 - center.1)^2 + (0 - center.2)^2 = (4 + center.1)^2
  intercepts_8_on_yaxis : (0 - center.1)^2 + (4 - center.2)^2 = (4 + center.1)^2

/-- The equation of circle C is either (x-2)^2 + (y-4)^2 = 20 or (x+2)^2 + (y-4)^2 = 20 -/
theorem circle_c_equation (c : CircleC) :
  ((λ (x y : ℝ) => (x - 2)^2 + (y - 4)^2 = 20) = λ (x y : ℝ) => (x - c.center.1)^2 + (y - c.center.2)^2 = (4 + c.a)^2) ∨
  ((λ (x y : ℝ) => (x + 2)^2 + (y - 4)^2 = 20) = λ (x y : ℝ) => (x - c.center.1)^2 + (y - c.center.2)^2 = (4 + c.a)^2) :=
sorry

end circle_c_equation_l2513_251376


namespace distance_to_focus_l2513_251384

/-- Given a parabola y^2 = 2x and a point P(m, 2) on the parabola,
    the distance from P to the focus of the parabola is 5/2 -/
theorem distance_to_focus (m : ℝ) (h : 2^2 = 2*m) : 
  let P : ℝ × ℝ := (m, 2)
  let F : ℝ × ℝ := (1/2, 0)
  Real.sqrt ((P.1 - F.1)^2 + (P.2 - F.2)^2) = 5/2 := by
  sorry

end distance_to_focus_l2513_251384


namespace function_periodicity_l2513_251344

open Real

theorem function_periodicity 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h_a : a > 0) 
  (h_f : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  ∀ x : ℝ, f (x + 2 * a) = f x := by
sorry

end function_periodicity_l2513_251344


namespace ln_is_elite_elite_bound_exists_nonincreasing_elite_sufficient_condition_elite_l2513_251353

/-- Definition of an "elite" function -/
def IsElite (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > 0 → f (x₁ + x₂) < f x₁ + f x₂

/-- Statement 1: ln(1+x) is an "elite" function -/
theorem ln_is_elite : IsElite (fun x => Real.log (1 + x)) := sorry

/-- Statement 2: For "elite" functions, f(n) < nf(1) for n ≥ 2 -/
theorem elite_bound (f : ℝ → ℝ) (hf : IsElite f) :
  ∀ n : ℕ, n ≥ 2 → f n < n * f 1 := sorry

/-- Statement 3: Existence of an "elite" function that is not strictly increasing -/
theorem exists_nonincreasing_elite :
  ∃ f : ℝ → ℝ, IsElite f ∧ ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ ≠ x₂ ∧ (f x₁ - f x₂) / (x₁ - x₂) ≤ 0 := sorry

/-- Statement 4: A sufficient condition for a function to be "elite" -/
theorem sufficient_condition_elite (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ > x₂ → x₂ > 0 → x₂ * f x₁ < x₁ * f x₂) : 
  IsElite f := sorry

end ln_is_elite_elite_bound_exists_nonincreasing_elite_sufficient_condition_elite_l2513_251353


namespace system_solution_l2513_251387

def solution_set : Set (ℝ × ℝ) :=
  {(-3/Real.sqrt 5, 1/Real.sqrt 5), (-3/Real.sqrt 5, -1/Real.sqrt 5),
   (3/Real.sqrt 5, -1/Real.sqrt 5), (3/Real.sqrt 5, 1/Real.sqrt 5)}

theorem system_solution :
  ∀ x y : ℝ, (x^2 + y^2 ≤ 2 ∧
    81*x^4 - 18*x^2*y^2 + y^4 - 360*x^2 - 40*y^2 + 400 = 0) ↔
  (x, y) ∈ solution_set :=
by sorry

end system_solution_l2513_251387


namespace common_difference_is_negative_three_l2513_251372

def arithmetic_sequence (n : ℕ) : ℤ := 2 - 3 * n

theorem common_difference_is_negative_three :
  ∃ d : ℤ, ∀ n : ℕ, arithmetic_sequence (n + 1) - arithmetic_sequence n = d ∧ d = -3 :=
sorry

end common_difference_is_negative_three_l2513_251372


namespace moon_permutations_eq_twelve_l2513_251345

/-- The number of distinct permutations of the letters in "MOON" -/
def moon_permutations : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

theorem moon_permutations_eq_twelve :
  moon_permutations = 12 := by
  sorry

#eval moon_permutations

end moon_permutations_eq_twelve_l2513_251345


namespace g_difference_l2513_251334

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5

-- State the theorem
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x + 3 * h + 4) := by
  sorry

end g_difference_l2513_251334


namespace cone_lateral_area_l2513_251355

/-- The lateral area of a cone with base radius 3 and slant height 5 is 15π -/
theorem cone_lateral_area :
  let base_radius : ℝ := 3
  let slant_height : ℝ := 5
  let lateral_area := π * base_radius * slant_height
  lateral_area = 15 * π :=
by sorry

end cone_lateral_area_l2513_251355


namespace jesse_room_area_l2513_251388

/-- The area of a rectangular room -/
def room_area (length width : ℝ) : ℝ := length * width

/-- Theorem: The area of Jesse's room is 96 square feet -/
theorem jesse_room_area :
  room_area 12 8 = 96 := by
  sorry

end jesse_room_area_l2513_251388


namespace cyclic_sum_inequality_l2513_251381

-- Define the cyclic sum function
def cyclicSum (f : ℝ → ℝ → ℝ → ℝ) (a b c : ℝ) : ℝ :=
  f a b c + f b c a + f c a b

-- State the theorem
theorem cyclic_sum_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  cyclicSum (fun x y z => (y + z - x)^2 / (x^2 + (y + z)^2)) a b c ≥ 3/5 := by
  sorry

end cyclic_sum_inequality_l2513_251381


namespace first_question_percentage_l2513_251350

/-- The percentage of students who answered the first question correctly -/
def first_question_correct : ℝ := sorry

/-- The percentage of students who answered the second question correctly -/
def second_question_correct : ℝ := 35

/-- The percentage of students who answered neither question correctly -/
def neither_correct : ℝ := 20

/-- The percentage of students who answered both questions correctly -/
def both_correct : ℝ := 30

/-- Theorem stating that the percentage of students who answered the first question correctly is 75% -/
theorem first_question_percentage :
  first_question_correct = 75 :=
by sorry

end first_question_percentage_l2513_251350


namespace magic_8_ball_probability_l2513_251313

/-- The number of questions asked to the Magic 8 Ball -/
def num_questions : ℕ := 7

/-- The number of possible responses from the Magic 8 Ball -/
def num_responses : ℕ := 3

/-- The probability of each type of response -/
def response_probability : ℚ := 1 / 3

/-- The number of desired positive responses -/
def desired_positive : ℕ := 3

/-- The number of desired neutral responses -/
def desired_neutral : ℕ := 2

/-- Theorem stating the probability of getting exactly 3 positive answers and 2 neutral answers
    when asking a Magic 8 Ball 7 questions, where each type of response has an equal probability of 1/3 -/
theorem magic_8_ball_probability :
  (Nat.choose num_questions desired_positive *
   Nat.choose (num_questions - desired_positive) desired_neutral *
   response_probability ^ num_questions) = 70 / 243 := by
  sorry

end magic_8_ball_probability_l2513_251313


namespace joe_remaining_money_l2513_251322

def joe_pocket_money : ℚ := 450

def chocolate_fraction : ℚ := 1/9
def fruit_fraction : ℚ := 2/5

def remaining_money : ℚ := joe_pocket_money - (chocolate_fraction * joe_pocket_money) - (fruit_fraction * joe_pocket_money)

theorem joe_remaining_money :
  remaining_money = 220 :=
by sorry

end joe_remaining_money_l2513_251322


namespace jack_books_left_l2513_251321

/-- The number of books left in Jack's classics section -/
def books_left (authors : ℕ) (books_per_author : ℕ) (lent_books : ℕ) (misplaced_books : ℕ) : ℕ :=
  authors * books_per_author - (lent_books + misplaced_books)

theorem jack_books_left :
  books_left 10 45 17 8 = 425 := by
  sorry

end jack_books_left_l2513_251321


namespace unfolded_paper_has_four_crosses_l2513_251320

/-- Represents a square piece of paper -/
structure Paper :=
  (side : ℝ)
  (is_square : side > 0)

/-- Represents a fold on the paper -/
inductive Fold
  | LeftRight
  | TopBottom

/-- Represents a cross pattern of holes -/
structure Cross :=
  (center : ℝ × ℝ)
  (size : ℝ)

/-- Represents the state of the paper after folding and punching -/
structure FoldedPaper :=
  (paper : Paper)
  (folds : List Fold)
  (cross : Cross)

/-- Represents the unfolded paper with crosses -/
structure UnfoldedPaper :=
  (paper : Paper)
  (crosses : List Cross)

/-- Function to unfold the paper -/
def unfold (fp : FoldedPaper) : UnfoldedPaper :=
  sorry

/-- Main theorem: Unfolding results in four crosses, one in each quadrant -/
theorem unfolded_paper_has_four_crosses (fp : FoldedPaper) 
  (h1 : fp.folds = [Fold.LeftRight, Fold.TopBottom])
  (h2 : fp.cross.center.1 > fp.paper.side / 2 ∧ fp.cross.center.2 > fp.paper.side / 2) :
  let up := unfold fp
  (up.crosses.length = 4) ∧ 
  (∀ q : ℕ, q < 4 → ∃ c ∈ up.crosses, 
    (c.center.1 < up.paper.side / 2 ↔ q % 2 = 0) ∧
    (c.center.2 < up.paper.side / 2 ↔ q < 2)) :=
  sorry

end unfolded_paper_has_four_crosses_l2513_251320


namespace fraction_replacement_l2513_251366

theorem fraction_replacement (x : ℚ) :
  ((5 / 2 / x * 5 / 2) / (5 / 2 * x / (5 / 2))) = 25 → x = 1/2 := by
  sorry

end fraction_replacement_l2513_251366


namespace cos_a2_a12_equals_half_l2513_251325

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem cos_a2_a12_equals_half
  (a : ℕ → ℝ)
  (h_geometric : geometric_sequence a)
  (h_condition : a 1 * a 13 + 2 * (a 7)^2 = 5 * Real.pi) :
  Real.cos (a 2 * a 12) = 1/2 := by
sorry

end cos_a2_a12_equals_half_l2513_251325


namespace tangent_line_to_circle_l2513_251304

theorem tangent_line_to_circle (a : ℝ) : 
  (∀ x y : ℝ, ax + y + 1 = 0 → (x - 2)^2 + y^2 = 4) →
  (∃! x y : ℝ, ax + y + 1 = 0 ∧ (x - 2)^2 + y^2 = 4) →
  a = 3/4 := by
sorry

end tangent_line_to_circle_l2513_251304


namespace triangle_angle_measure_l2513_251323

theorem triangle_angle_measure (D E F : ℝ) : 
  D + E + F = 180 →  -- Sum of angles in a triangle is 180°
  E = F →            -- Angle E is congruent to Angle F
  F = 3 * D →        -- Angle F is three times Angle D
  E = 540 / 7 :=     -- Measure of Angle E
by sorry

end triangle_angle_measure_l2513_251323


namespace pumps_emptying_time_l2513_251316

/-- Represents the time (in hours) it takes for pumps A, B, and C to empty a pool when working together. -/
def combined_emptying_time (rate_A rate_B rate_C : ℚ) : ℚ :=
  1 / (rate_A + rate_B + rate_C)

/-- Theorem stating that pumps A, B, and C with given rates will empty the pool in 24/13 hours when working together. -/
theorem pumps_emptying_time :
  let rate_A : ℚ := 1/4
  let rate_B : ℚ := 1/6
  let rate_C : ℚ := 1/8
  combined_emptying_time rate_A rate_B rate_C = 24/13 := by
  sorry

#eval (24 : ℚ) / 13 * 60 -- Converts the result to minutes

end pumps_emptying_time_l2513_251316


namespace range_of_cubic_function_l2513_251392

def f (x : ℝ) := x^3

theorem range_of_cubic_function :
  Set.range (fun x => f x) = Set.Ici (-1) :=
sorry

end range_of_cubic_function_l2513_251392


namespace isosceles_if_root_is_one_roots_of_equilateral_l2513_251386

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : a > 0
  hb : b > 0
  hc : c > 0

-- Define the quadratic equation
def quadratic (t : Triangle) (x : ℝ) : ℝ :=
  (t.a + t.c) * x^2 - 2 * t.b * x + (t.a - t.c)

theorem isosceles_if_root_is_one (t : Triangle) :
  quadratic t 1 = 0 → t.a = t.b :=
by sorry

theorem roots_of_equilateral (t : Triangle) :
  t.a = t.b ∧ t.b = t.c →
  (∀ x : ℝ, quadratic t x = 0 ↔ x = 0 ∨ x = 1) :=
by sorry

end isosceles_if_root_is_one_roots_of_equilateral_l2513_251386


namespace pirate_treasure_ratio_l2513_251338

theorem pirate_treasure_ratio : 
  let total_gold : ℕ := 3500
  let num_chests : ℕ := 5
  let total_silver : ℕ := 500
  let coins_per_chest : ℕ := 1000
  let gold_per_chest : ℕ := total_gold / num_chests
  let silver_per_chest : ℕ := total_silver / num_chests
  let bronze_per_chest : ℕ := coins_per_chest - gold_per_chest - silver_per_chest
  bronze_per_chest = 2 * silver_per_chest :=
by sorry

end pirate_treasure_ratio_l2513_251338


namespace square_of_sum_l2513_251397

theorem square_of_sum (a b : ℝ) : (a + b)^2 = a^2 + 2*a*b + b^2 := by
  sorry

end square_of_sum_l2513_251397
