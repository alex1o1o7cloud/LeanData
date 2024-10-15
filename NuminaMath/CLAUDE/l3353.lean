import Mathlib

namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3353_335343

theorem polynomial_division_theorem (x : ℝ) : 
  (x^4 + 13) = (x - 1) * (x^3 + x^2 + x + 1) + 14 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3353_335343


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3353_335380

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem max_value_of_expression (x y z : ℕ) 
  (h_two_digit_x : is_two_digit x)
  (h_two_digit_y : is_two_digit y)
  (h_two_digit_z : is_two_digit z)
  (h_mean : (x + y + z) / 3 = 60) :
  (∀ a b c : ℕ, is_two_digit a → is_two_digit b → is_two_digit c → 
    (a + b + c) / 3 = 60 → (a + b) / c ≤ 17) ∧
  (∃ a b c : ℕ, is_two_digit a ∧ is_two_digit b ∧ is_two_digit c ∧ 
    (a + b + c) / 3 = 60 ∧ (a + b) / c = 17) := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3353_335380


namespace NUMINAMATH_CALUDE_f_inequality_and_range_l3353_335394

noncomputable def f (x : ℝ) := 1 - Real.exp (-x)

theorem f_inequality_and_range :
  (∀ x > -1, f x ≥ x / (x + 1)) ∧
  (Set.Icc (0 : ℝ) (1/2) = {a | ∀ x ≥ 0, f x ≤ x / (a * x + 1)}) :=
sorry

end NUMINAMATH_CALUDE_f_inequality_and_range_l3353_335394


namespace NUMINAMATH_CALUDE_three_digit_same_divisible_by_37_l3353_335323

theorem three_digit_same_divisible_by_37 (a : ℕ) (h : a ≤ 9) :
  ∃ k : ℕ, 111 * a = 37 * k := by
  sorry

end NUMINAMATH_CALUDE_three_digit_same_divisible_by_37_l3353_335323


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l3353_335362

theorem complex_fraction_calculation : 
  (6 + 3/5 - (17/2 - 1/3) / (7/2)) * (2 + 5/18 + 11/12) = 368/27 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l3353_335362


namespace NUMINAMATH_CALUDE_eight_entrepreneurs_not_attending_l3353_335372

/-- The number of entrepreneurs who did not attend either session -/
def entrepreneurs_not_attending (total : ℕ) (digital : ℕ) (ecommerce : ℕ) (both : ℕ) : ℕ :=
  total - (digital + ecommerce - both)

/-- Theorem: Given the specified numbers of entrepreneurs, prove that 8 did not attend either session -/
theorem eight_entrepreneurs_not_attending :
  entrepreneurs_not_attending 40 22 18 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_eight_entrepreneurs_not_attending_l3353_335372


namespace NUMINAMATH_CALUDE_light_distance_500_years_l3353_335319

theorem light_distance_500_years :
  let distance_one_year : ℝ := 5870000000000
  let years : ℕ := 500
  (distance_one_year * years : ℝ) = 2.935 * (10 ^ 15) :=
by sorry

end NUMINAMATH_CALUDE_light_distance_500_years_l3353_335319


namespace NUMINAMATH_CALUDE_adrianna_gum_theorem_l3353_335332

/-- Calculates the remaining pieces of gum after sharing with friends -/
def remaining_gum (initial : ℕ) (additional : ℕ) (friends : ℕ) : ℕ :=
  initial + additional - friends

/-- Theorem stating that Adrianna's remaining gum pieces is 2 -/
theorem adrianna_gum_theorem (initial : ℕ) (additional : ℕ) (friends : ℕ)
  (h1 : initial = 10)
  (h2 : additional = 3)
  (h3 : friends = 11) :
  remaining_gum initial additional friends = 2 := by
  sorry

#eval remaining_gum 10 3 11

end NUMINAMATH_CALUDE_adrianna_gum_theorem_l3353_335332


namespace NUMINAMATH_CALUDE_four_composition_odd_l3353_335306

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem four_composition_odd (f : ℝ → ℝ) (h : IsOdd f) : IsOdd (fun x ↦ f (f (f (f x)))) := by
  sorry

end NUMINAMATH_CALUDE_four_composition_odd_l3353_335306


namespace NUMINAMATH_CALUDE_gcd_problem_l3353_335369

/-- The operation * represents the greatest common divisor -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- Theorem: The GCD of ((12 * 16) * (18 * 12)) is 2 -/
theorem gcd_problem : gcd_op (gcd_op (gcd_op 12 16) (gcd_op 18 12)) 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l3353_335369


namespace NUMINAMATH_CALUDE_bills_problem_l3353_335347

/-- Represents the bill amount for a person -/
structure Bill where
  amount : ℝ
  tipPercentage : ℝ
  tipAmount : ℝ

/-- The problem statement -/
theorem bills_problem (mike : Bill) (joe : Bill) (bill : Bill)
  (h_mike : mike.tipPercentage = 0.10 ∧ mike.tipAmount = 3)
  (h_joe : joe.tipPercentage = 0.15 ∧ joe.tipAmount = 4.5)
  (h_bill : bill.tipPercentage = 0.25 ∧ bill.tipAmount = 5) :
  bill.amount = 20 := by
  sorry


end NUMINAMATH_CALUDE_bills_problem_l3353_335347


namespace NUMINAMATH_CALUDE_taxi_trip_length_l3353_335352

theorem taxi_trip_length 
  (initial_fee : ℝ) 
  (additional_charge : ℝ) 
  (segment_length : ℝ) 
  (total_charge : ℝ) : 
  initial_fee = 2.25 →
  additional_charge = 0.15 →
  segment_length = 2/5 →
  total_charge = 3.60 →
  ∃ (trip_length : ℝ), 
    trip_length = 3.6 ∧ 
    total_charge = initial_fee + (trip_length / segment_length) * additional_charge :=
by sorry

end NUMINAMATH_CALUDE_taxi_trip_length_l3353_335352


namespace NUMINAMATH_CALUDE_sum_of_digits_499849_l3353_335353

def number : Nat := 499849

def sumOfDigits (n : Nat) : Nat :=
  let digits := n.repr.toList.map (fun c => c.toNat - '0'.toNat)
  digits.sum

theorem sum_of_digits_499849 :
  sumOfDigits number = 43 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_499849_l3353_335353


namespace NUMINAMATH_CALUDE_all_transformations_pass_through_point_l3353_335325

def f (x : ℝ) := (x - 2)^2
def g (x : ℝ) := (x - 1)^2 - 1
def h (x : ℝ) := x^2 - 4
def k (x : ℝ) := -x^2 + 4

theorem all_transformations_pass_through_point :
  f 2 = 0 ∧ g 2 = 0 ∧ h 2 = 0 ∧ k 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_all_transformations_pass_through_point_l3353_335325


namespace NUMINAMATH_CALUDE_vector_sum_magnitude_l3353_335321

def a : ℝ × ℝ := (2, 0)

theorem vector_sum_magnitude (b : ℝ × ℝ) 
  (h1 : Real.cos (Real.pi / 3) = (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  (h2 : b.1^2 + b.2^2 = 1) : 
  Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = 2 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_vector_sum_magnitude_l3353_335321


namespace NUMINAMATH_CALUDE_A_infinite_l3353_335312

/-- τ(n) denotes the number of positive divisors of the positive integer n -/
def tau (n : ℕ+) : ℕ := sorry

/-- The set of positive integers a for which τ(an) = n has no positive integer solutions n -/
def A : Set ℕ+ := {a | ∀ n : ℕ+, tau (a * n) ≠ n}

/-- Theorem: The set A is infinite -/
theorem A_infinite : Set.Infinite A := by sorry

end NUMINAMATH_CALUDE_A_infinite_l3353_335312


namespace NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l3353_335351

/-- Represents the number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute (balls : ℕ) (boxes : ℕ) (min_per_box : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 3 ways to distribute 7 balls into 4 boxes -/
theorem distribute_seven_balls_four_boxes :
  distribute 7 4 1 = 3 := by sorry

end NUMINAMATH_CALUDE_distribute_seven_balls_four_boxes_l3353_335351


namespace NUMINAMATH_CALUDE_carrot_weight_problem_l3353_335301

/-- Given 30 carrots weighing 5.94 kg, and 27 of these carrots having an average weight of 200 grams,
    the average weight of the remaining 3 carrots is 180 grams. -/
theorem carrot_weight_problem (total_weight : ℝ) (avg_weight_27 : ℝ) :
  total_weight = 5.94 →
  avg_weight_27 = 0.2 →
  (total_weight * 1000 - 27 * avg_weight_27 * 1000) / 3 = 180 :=
by sorry

end NUMINAMATH_CALUDE_carrot_weight_problem_l3353_335301


namespace NUMINAMATH_CALUDE_monthly_income_calculation_l3353_335388

/-- Proves that if 22% of a person's monthly income is Rs. 3800, then their monthly income is Rs. 17272.73. -/
theorem monthly_income_calculation (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) 
  (h1 : deposit = 3800)
  (h2 : percentage = 22)
  (h3 : deposit = (percentage / 100) * monthly_income) :
  monthly_income = 17272.73 := by
  sorry

end NUMINAMATH_CALUDE_monthly_income_calculation_l3353_335388


namespace NUMINAMATH_CALUDE_special_sequence_a9_l3353_335357

/-- A sequence of positive integers satisfying the given property -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∀ p q : ℕ, a (p + q) = a p + a q)

theorem special_sequence_a9 (a : ℕ → ℕ) (h : SpecialSequence a) (h2 : a 2 = 4) : 
  a 9 = 18 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_a9_l3353_335357


namespace NUMINAMATH_CALUDE_num_factors_180_multiple_15_eq_6_l3353_335340

/-- The number of positive factors of 180 that are also multiples of 15 -/
def num_factors_180_multiple_15 : ℕ :=
  (Finset.filter (λ x => 180 % x = 0 ∧ x % 15 = 0) (Finset.range 181)).card

/-- Theorem stating that the number of positive factors of 180 that are also multiples of 15 is 6 -/
theorem num_factors_180_multiple_15_eq_6 : num_factors_180_multiple_15 = 6 := by
  sorry

end NUMINAMATH_CALUDE_num_factors_180_multiple_15_eq_6_l3353_335340


namespace NUMINAMATH_CALUDE_third_flip_probability_is_one_sixth_l3353_335361

/-- Represents the "Treasure Box" game in the "Lucky 52" program --/
structure TreasureBoxGame where
  total_logos : ℕ
  winning_logos : ℕ
  flips : ℕ
  flipped_winning_logos : ℕ

/-- The probability of winning on the third flip in the Treasure Box game --/
def third_flip_probability (game : TreasureBoxGame) : ℚ :=
  let remaining_logos := game.total_logos - game.flipped_winning_logos
  let remaining_winning_logos := game.winning_logos - game.flipped_winning_logos
  remaining_winning_logos / remaining_logos

/-- Theorem stating the probability of winning on the third flip --/
theorem third_flip_probability_is_one_sixth 
  (game : TreasureBoxGame) 
  (h1 : game.total_logos = 20)
  (h2 : game.winning_logos = 5)
  (h3 : game.flips = 3)
  (h4 : game.flipped_winning_logos = 2) :
  third_flip_probability game = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_third_flip_probability_is_one_sixth_l3353_335361


namespace NUMINAMATH_CALUDE_simplify_expression_l3353_335330

theorem simplify_expression (y : ℝ) : 3 * y + 4 * y + 5 * y + 7 = 12 * y + 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3353_335330


namespace NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3353_335365

theorem no_function_satisfies_inequality :
  ¬∃ (f : ℝ → ℝ), ∀ (x y : ℝ), f x + f y ≥ 2 * f ((x + y) / 2) + 2 * |x - y| := by
  sorry

end NUMINAMATH_CALUDE_no_function_satisfies_inequality_l3353_335365


namespace NUMINAMATH_CALUDE_bike_rental_cost_theorem_l3353_335397

/-- The fee structure and rental details for a bicycle rental service. -/
structure BikeRental where
  fee_per_30min : ℕ  -- Fee in won for 30 minutes
  num_bikes : ℕ      -- Number of bikes rented
  duration_hours : ℕ -- Duration of rental in hours
  num_people : ℕ     -- Number of people splitting the cost

/-- Calculate the cost per person for a bike rental. -/
def cost_per_person (rental : BikeRental) : ℕ :=
  let total_cost := rental.fee_per_30min * 2 * rental.duration_hours * rental.num_bikes
  total_cost / rental.num_people

/-- Theorem stating that under the given conditions, each person pays 16000 won. -/
theorem bike_rental_cost_theorem (rental : BikeRental) 
  (h1 : rental.fee_per_30min = 4000)
  (h2 : rental.num_bikes = 4)
  (h3 : rental.duration_hours = 3)
  (h4 : rental.num_people = 6) : 
  cost_per_person rental = 16000 := by
  sorry

end NUMINAMATH_CALUDE_bike_rental_cost_theorem_l3353_335397


namespace NUMINAMATH_CALUDE_connie_watch_purchase_l3353_335336

/-- The additional amount Connie needs to buy a watch -/
def additional_amount (savings : ℕ) (watch_cost : ℕ) : ℕ :=
  watch_cost - savings

/-- Theorem: Given Connie's savings and the watch cost, prove the additional amount needed -/
theorem connie_watch_purchase (connie_savings : ℕ) (watch_price : ℕ) 
  (h1 : connie_savings = 39)
  (h2 : watch_price = 55) :
  additional_amount connie_savings watch_price = 16 := by
  sorry

end NUMINAMATH_CALUDE_connie_watch_purchase_l3353_335336


namespace NUMINAMATH_CALUDE_weight_of_second_new_student_l3353_335305

theorem weight_of_second_new_student
  (initial_students : Nat)
  (initial_avg_weight : ℝ)
  (new_students : Nat)
  (new_avg_weight : ℝ)
  (weight_of_first_new_student : ℝ)
  (h1 : initial_students = 29)
  (h2 : initial_avg_weight = 28)
  (h3 : new_students = initial_students + 2)
  (h4 : new_avg_weight = 27.5)
  (h5 : weight_of_first_new_student = 25)
  : ∃ (weight_of_second_new_student : ℝ),
    weight_of_second_new_student = 20.5 ∧
    (initial_students : ℝ) * initial_avg_weight + weight_of_first_new_student + weight_of_second_new_student =
    (new_students : ℝ) * new_avg_weight :=
by sorry

end NUMINAMATH_CALUDE_weight_of_second_new_student_l3353_335305


namespace NUMINAMATH_CALUDE_poem_word_count_l3353_335344

/-- Given a poem with the specified structure, prove that the total number of words is 1600. -/
theorem poem_word_count (stanzas : ℕ) (lines_per_stanza : ℕ) (words_per_line : ℕ)
  (h1 : stanzas = 20)
  (h2 : lines_per_stanza = 10)
  (h3 : words_per_line = 8) :
  stanzas * lines_per_stanza * words_per_line = 1600 := by
  sorry


end NUMINAMATH_CALUDE_poem_word_count_l3353_335344


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3353_335310

theorem sum_of_two_numbers (x y : ℝ) : 
  (x + y) + (x - y) = 8 → x^2 - y^2 = 160 → x + y = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3353_335310


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l3353_335328

/-- The number of trailing zeros in a natural number -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The product of 125 and 960 -/
def product : ℕ := 125 * 960

theorem product_trailing_zeros :
  trailingZeros product = 4 := by sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l3353_335328


namespace NUMINAMATH_CALUDE_intersection_A_not_B_range_of_a_l3353_335396

-- Define the sets A and B
def A : Set ℝ := {x | x > 1}
def B : Set ℝ := {x | x ≥ 2}

-- Define the complement of B
def not_B : Set ℝ := {x | x < 2}

-- Define the set C
def C (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Theorem 1: A ∩ (¬ᵣB) = {x | 1 < x < 2}
theorem intersection_A_not_B : A ∩ not_B = {x : ℝ | 1 < x ∧ x < 2} := by sorry

-- Theorem 2: The range of a is [1, +∞) when A ∩ C = C
theorem range_of_a (a : ℝ) : A ∩ C a = C a ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_A_not_B_range_of_a_l3353_335396


namespace NUMINAMATH_CALUDE_symmetrical_points_product_l3353_335398

/-- 
Given two points P₁(a, 5) and P₂(-4, b) that are symmetrical about the x-axis,
prove that their x-coordinate product is -20.
-/
theorem symmetrical_points_product (a b : ℝ) : 
  (a = 4 ∧ b = -5) → a * b = -20 := by sorry

end NUMINAMATH_CALUDE_symmetrical_points_product_l3353_335398


namespace NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l3353_335307

-- Problem 1
theorem solve_equation_1 : ∃ x : ℝ, (3 * x^2 - 32 * x - 48 = 0) ↔ (x = 12 ∨ x = -4/3) := by sorry

-- Problem 2
theorem solve_equation_2 : ∃ x : ℝ, (4 * x^2 + x - 3 = 0) ↔ (x = 3/4 ∨ x = -1) := by sorry

-- Problem 3
theorem solve_equation_3 : ∃ x : ℝ, ((3 * x + 1)^2 - 4 = 0) ↔ (x = 1/3 ∨ x = -1) := by sorry

-- Problem 4
theorem solve_equation_4 : ∃ x : ℝ, (9 * (x - 2)^2 = 4 * (x + 1)^2) ↔ (x = 8 ∨ x = 4/5) := by sorry

end NUMINAMATH_CALUDE_solve_equation_1_solve_equation_2_solve_equation_3_solve_equation_4_l3353_335307


namespace NUMINAMATH_CALUDE_students_passed_l3353_335300

theorem students_passed (total : ℕ) (fail_freq : ℚ) (h1 : total = 1000) (h2 : fail_freq = 0.4) :
  total - (total * fail_freq).floor = 600 := by
  sorry

end NUMINAMATH_CALUDE_students_passed_l3353_335300


namespace NUMINAMATH_CALUDE_f_neg_two_eq_three_l3353_335391

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

-- State the theorem
theorem f_neg_two_eq_three 
  (a b c : ℝ) 
  (h : f a b c 2 = -1) : 
  f a b c (-2) = 3 := by
sorry

end NUMINAMATH_CALUDE_f_neg_two_eq_three_l3353_335391


namespace NUMINAMATH_CALUDE_journey_distance_ratio_l3353_335329

/-- Given a journey where:
  - The initial distance traveled is 20 hours at 30 kilometers per hour
  - After a setback, the traveler is one-third of the way to the destination
  Prove that the ratio of the initial distance to the total distance is 1/3 -/
theorem journey_distance_ratio :
  ∀ (initial_speed : ℝ) (initial_time : ℝ) (total_distance : ℝ),
    initial_speed = 30 →
    initial_time = 20 →
    initial_speed * initial_time = (1/3) * total_distance →
    (initial_speed * initial_time) / total_distance = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_journey_distance_ratio_l3353_335329


namespace NUMINAMATH_CALUDE_total_slices_eq_207_l3353_335367

/-- The total number of watermelon and fruit slices at a family picnic --/
def total_slices : ℕ :=
  let danny_watermelon := 3 * 10
  let sister_watermelon := 1 * 15
  let cousin_watermelon := 2 * 8
  let cousin_apples := 5 * 4
  let aunt_watermelon := 4 * 12
  let aunt_oranges := 7 * 6
  let grandfather_watermelon := 1 * 6
  let grandfather_pineapples := 3 * 10
  danny_watermelon + sister_watermelon + cousin_watermelon + cousin_apples +
  aunt_watermelon + aunt_oranges + grandfather_watermelon + grandfather_pineapples

theorem total_slices_eq_207 : total_slices = 207 := by
  sorry

end NUMINAMATH_CALUDE_total_slices_eq_207_l3353_335367


namespace NUMINAMATH_CALUDE_remaining_apple_pies_l3353_335335

/-- Proves the number of apple pies remaining with Cooper --/
theorem remaining_apple_pies (pies_per_day : ℕ) (days : ℕ) (pies_eaten : ℕ) : 
  pies_per_day = 7 → days = 12 → pies_eaten = 50 → 
  pies_per_day * days - pies_eaten = 34 := by
  sorry

#check remaining_apple_pies

end NUMINAMATH_CALUDE_remaining_apple_pies_l3353_335335


namespace NUMINAMATH_CALUDE_kylie_coins_left_l3353_335376

/-- Calculates the number of coins Kylie has left after giving half to Laura -/
def kyliesRemainingCoins (piggyBank : ℕ) (brotherCoins : ℕ) (sofaCoins : ℕ) : ℕ :=
  let fatherCoins := 2 * brotherCoins
  let totalCoins := piggyBank + brotherCoins + fatherCoins + sofaCoins
  totalCoins - (totalCoins / 2)

/-- Theorem stating that Kylie has 62 coins left -/
theorem kylie_coins_left :
  kyliesRemainingCoins 30 26 15 = 62 := by
  sorry

#eval kyliesRemainingCoins 30 26 15

end NUMINAMATH_CALUDE_kylie_coins_left_l3353_335376


namespace NUMINAMATH_CALUDE_successive_integers_product_l3353_335350

theorem successive_integers_product (n : ℤ) : n * (n + 1) = 9506 → n = 97 := by
  sorry

end NUMINAMATH_CALUDE_successive_integers_product_l3353_335350


namespace NUMINAMATH_CALUDE_bead_division_problem_l3353_335318

/-- The number of equal parts into which the beads were divided -/
def n : ℕ := sorry

/-- The total number of beads -/
def total_beads : ℕ := 23 + 16

/-- The number of beads in each part after division but before removal -/
def beads_per_part : ℚ := total_beads / n

/-- The number of beads in each part after removal but before doubling -/
def beads_after_removal : ℚ := beads_per_part - 10

/-- The final number of beads in each part after doubling -/
def final_beads : ℕ := 6

theorem bead_division_problem :
  2 * beads_after_removal = final_beads ∧ n = 3 := by sorry

end NUMINAMATH_CALUDE_bead_division_problem_l3353_335318


namespace NUMINAMATH_CALUDE_smallest_positive_shift_is_90_l3353_335309

/-- A function with a 30-unit shift property -/
def ShiftFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x - 30) = f x

/-- The smallest positive shift for the scaled function -/
def SmallestPositiveShift (f : ℝ → ℝ) (b : ℝ) : Prop :=
  b > 0 ∧
  (∀ x : ℝ, f ((x - b) / 3) = f (x / 3)) ∧
  (∀ c : ℝ, c > 0 → (∀ x : ℝ, f ((x - c) / 3) = f (x / 3)) → b ≤ c)

theorem smallest_positive_shift_is_90 (f : ℝ → ℝ) (h : ShiftFunction f) :
  SmallestPositiveShift f 90 :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_shift_is_90_l3353_335309


namespace NUMINAMATH_CALUDE_school_duration_in_minutes_l3353_335379

-- Define the start and end times
def start_time : ℕ := 7
def end_time : ℕ := 11

-- Define the duration in hours
def duration_hours : ℕ := end_time - start_time

-- Define the conversion factor from hours to minutes
def minutes_per_hour : ℕ := 60

-- Theorem to prove
theorem school_duration_in_minutes :
  duration_hours * minutes_per_hour = 240 :=
sorry

end NUMINAMATH_CALUDE_school_duration_in_minutes_l3353_335379


namespace NUMINAMATH_CALUDE_z_is_in_fourth_quadrant_l3353_335333

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the sets M and N
def M (z : ℂ) : Set ℂ := {1, z * (1 + i)}
def N : Set ℂ := {3, 4}

-- State the theorem
theorem z_is_in_fourth_quadrant (z : ℂ) :
  M z ∪ N = {1, 2, 3, 4} → z = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_z_is_in_fourth_quadrant_l3353_335333


namespace NUMINAMATH_CALUDE_ellipse_circle_tangent_property_l3353_335374

/-- Given an ellipse and a circle, prove a property of tangents from a point on the ellipse to the circle -/
theorem ellipse_circle_tangent_property
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (P : ℝ × ℝ) (hP : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (hP_not_vertex : P ≠ (a, 0) ∧ P ≠ (-a, 0) ∧ P ≠ (0, b) ∧ P ≠ (0, -b))
  (A B : ℝ × ℝ)
  (hA : A.1^2 + A.2^2 = b^2)
  (hB : B.1^2 + B.2^2 = b^2)
  (hPA : P.1 * A.1 + P.2 * A.2 = b^2)
  (hPB : P.1 * B.1 + P.2 * B.2 = b^2)
  (M : ℝ × ℝ) (hM : M.2 = 0 ∧ M.1 * P.1 = b^2)
  (N : ℝ × ℝ) (hN : N.1 = 0 ∧ N.2 * P.2 = b^2) :
  a^2 / (N.2^2) + b^2 / (M.1^2) = a^2 / b^2 :=
sorry

end NUMINAMATH_CALUDE_ellipse_circle_tangent_property_l3353_335374


namespace NUMINAMATH_CALUDE_distribute_6_3_l3353_335358

/-- The number of ways to distribute n identical objects into k distinct containers,
    with each container having at least one object. -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n - 1) (k - 1)

/-- The problem statement -/
theorem distribute_6_3 : distribute 6 3 = 10 := by sorry

end NUMINAMATH_CALUDE_distribute_6_3_l3353_335358


namespace NUMINAMATH_CALUDE_congruence_problem_l3353_335366

theorem congruence_problem (x : ℤ) :
  x ≡ 3 [ZMOD 7] →
  x^2 ≡ 44 [ZMOD (7^2)] →
  x^3 ≡ 111 [ZMOD (7^3)] →
  x ≡ 17 [ZMOD 343] := by
sorry

end NUMINAMATH_CALUDE_congruence_problem_l3353_335366


namespace NUMINAMATH_CALUDE_factor_implies_b_value_l3353_335317

/-- The polynomial Q(x) = x^3 + 3x^2 + bx + 20 -/
def Q (b : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + b*x + 20

/-- Theorem: If x - 4 is a factor of Q(x), then b = -33 -/
theorem factor_implies_b_value (b : ℝ) :
  (∀ x, Q b x = 0 ↔ x = 4) → b = -33 := by
  sorry

end NUMINAMATH_CALUDE_factor_implies_b_value_l3353_335317


namespace NUMINAMATH_CALUDE_product_units_digit_l3353_335348

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n = a * b

def units_digit (n : ℕ) : ℕ := n % 10

theorem product_units_digit :
  is_composite 4 ∧ is_composite 6 ∧ is_composite 9 →
  units_digit (4 * 6 * 9) = 6 := by sorry

end NUMINAMATH_CALUDE_product_units_digit_l3353_335348


namespace NUMINAMATH_CALUDE_evaluate_expression_l3353_335302

theorem evaluate_expression (x y : ℝ) (hx : x = 4) (hy : y = 5) :
  2 * y * (y - 2 * x) = -30 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3353_335302


namespace NUMINAMATH_CALUDE_circumcircle_area_of_triangle_ABP_l3353_335316

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/3 = 1

-- Define points
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)
def F₂ : ℝ × ℝ := (2, 0)

-- Define the condition |AP⃗|cos<AP⃗, AF₂⃗> = |AF₂⃗|
def condition_AP_AF₂ (P : ℝ × ℝ) : Prop :=
  let AP := (P.1 - A.1, P.2 - A.2)
  let AF₂ := (F₂.1 - A.1, F₂.2 - A.2)
  Real.sqrt (AP.1^2 + AP.2^2) * (AP.1 * AF₂.1 + AP.2 * AF₂.2) / 
    (Real.sqrt (AP.1^2 + AP.2^2) * Real.sqrt (AF₂.1^2 + AF₂.2^2)) = 
    Real.sqrt (AF₂.1^2 + AF₂.2^2)

-- Define the theorem
theorem circumcircle_area_of_triangle_ABP (P : ℝ × ℝ) 
  (h₁ : hyperbola P.1 P.2)
  (h₂ : P.1 > B.1)  -- P is on the right branch
  (h₃ : condition_AP_AF₂ P) :
  ∃ (R : ℝ), R > 0 ∧ π * R^2 = 5 * π := by sorry

end NUMINAMATH_CALUDE_circumcircle_area_of_triangle_ABP_l3353_335316


namespace NUMINAMATH_CALUDE_novel_distribution_count_l3353_335383

/-- The number of ways to distribute 4 novels among 5 students -/
def novel_distribution : ℕ :=
  -- Definition goes here
  sorry

/-- Theorem stating that the number of novel distributions is 240 -/
theorem novel_distribution_count : novel_distribution = 240 := by
  sorry

end NUMINAMATH_CALUDE_novel_distribution_count_l3353_335383


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l3353_335364

theorem quadratic_no_real_roots :
  ∀ (x : ℝ), 2 * x^2 - 5 * x + 6 ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l3353_335364


namespace NUMINAMATH_CALUDE_distribute_identical_items_l3353_335354

theorem distribute_identical_items (n : ℕ) (k : ℕ) :
  n = 10 → k = 3 → Nat.choose (n + k - 1) k = 220 := by
  sorry

end NUMINAMATH_CALUDE_distribute_identical_items_l3353_335354


namespace NUMINAMATH_CALUDE_cos_double_angle_special_l3353_335339

theorem cos_double_angle_special (α : Real) (h1 : α ∈ Set.Ioo 0 π) 
  (h2 : Real.sin α + Real.cos α = 1/5) : Real.cos (2 * α) = -7/25 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_l3353_335339


namespace NUMINAMATH_CALUDE_angle_A_is_pi_over_three_perimeter_range_l3353_335355

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  -- Triangle inequality
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  ineq_a : a < b + c
  ineq_b : b < a + c
  ineq_c : c < a + b
  -- Angle sum is π
  angle_sum : A + B + C = π
  -- Sine rule
  sine_rule_a : a / Real.sin A = b / Real.sin B
  sine_rule_b : b / Real.sin B = c / Real.sin C
  -- Cosine rule
  cosine_rule_a : a^2 = b^2 + c^2 - 2*b*c*Real.cos A
  cosine_rule_b : b^2 = a^2 + c^2 - 2*a*c*Real.cos B
  cosine_rule_c : c^2 = a^2 + b^2 - 2*a*b*Real.cos C

theorem angle_A_is_pi_over_three (t : Triangle) (h : t.a * Real.cos t.C + (1/2) * t.c = t.b) :
  t.A = π/3 := by sorry

theorem perimeter_range (t : Triangle) (h1 : t.a = 1) (h2 : t.a * Real.cos t.C + (1/2) * t.c = t.b) :
  2 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 3 := by sorry

end NUMINAMATH_CALUDE_angle_A_is_pi_over_three_perimeter_range_l3353_335355


namespace NUMINAMATH_CALUDE_wire_length_ratio_l3353_335349

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Calculates the total wire length needed for a cuboid frame -/
def wireLength (d : CuboidDimensions) : ℝ :=
  4 * (d.length + d.width + d.height)

theorem wire_length_ratio : 
  let bonnie := CuboidDimensions.mk 8 10 10
  let roark := CuboidDimensions.mk 1 2 2
  let bonnieVolume := cuboidVolume bonnie
  let roarkVolume := cuboidVolume roark
  let numRoarkCuboids := bonnieVolume / roarkVolume
  let bonnieWire := wireLength bonnie
  let roarkTotalWire := numRoarkCuboids * wireLength roark
  bonnieWire / roarkTotalWire = 9 / 250 := by
  sorry

end NUMINAMATH_CALUDE_wire_length_ratio_l3353_335349


namespace NUMINAMATH_CALUDE_feathers_per_crown_calculation_l3353_335378

/-- Given a total number of feathers and a number of crowns, 
    calculate the number of feathers per crown. -/
def feathers_per_crown (total_feathers : ℕ) (num_crowns : ℕ) : ℕ :=
  (total_feathers + num_crowns - 1) / num_crowns

/-- Theorem stating that given 6538 feathers and 934 crowns, 
    the number of feathers per crown is 7. -/
theorem feathers_per_crown_calculation :
  feathers_per_crown 6538 934 = 7 := by
  sorry


end NUMINAMATH_CALUDE_feathers_per_crown_calculation_l3353_335378


namespace NUMINAMATH_CALUDE_rohans_salary_l3353_335370

/-- Rohan's monthly salary in Rupees -/
def monthly_salary : ℝ := 7500

/-- Percentage of salary spent on food -/
def food_expense_percent : ℝ := 40

/-- Percentage of salary spent on house rent -/
def rent_expense_percent : ℝ := 20

/-- Percentage of salary spent on entertainment -/
def entertainment_expense_percent : ℝ := 10

/-- Percentage of salary spent on conveyance -/
def conveyance_expense_percent : ℝ := 10

/-- Rohan's savings at the end of the month in Rupees -/
def savings : ℝ := 1500

/-- Theorem stating that Rohan's monthly salary is 7500 Rupees -/
theorem rohans_salary :
  monthly_salary = 7500 ∧
  food_expense_percent + rent_expense_percent + entertainment_expense_percent + conveyance_expense_percent = 80 ∧
  savings = monthly_salary * (100 - (food_expense_percent + rent_expense_percent + entertainment_expense_percent + conveyance_expense_percent)) / 100 :=
by sorry

end NUMINAMATH_CALUDE_rohans_salary_l3353_335370


namespace NUMINAMATH_CALUDE_interest_calculation_l3353_335382

/-- Simple interest calculation -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_calculation (principal rate interest : ℝ) :
  principal = 26 →
  rate = 7 / 100 →
  interest = 10.92 →
  ∃ (time : ℝ), simple_interest principal rate time = interest ∧ time = 6 :=
by sorry

end NUMINAMATH_CALUDE_interest_calculation_l3353_335382


namespace NUMINAMATH_CALUDE_bobbit_worm_days_l3353_335389

/-- The number of days the Bobbit worm was in the aquarium before James added more fish -/
def days_before_adding : ℕ := sorry

/-- The initial number of fish in the aquarium -/
def initial_fish : ℕ := 60

/-- The number of fish the Bobbit worm eats per day -/
def fish_eaten_per_day : ℕ := 2

/-- The number of fish James adds to the aquarium -/
def fish_added : ℕ := 8

/-- The number of days between adding fish and discovering the Bobbit worm -/
def days_after_adding : ℕ := 7

/-- The final number of fish in the aquarium when James discovers the Bobbit worm -/
def final_fish : ℕ := 26

theorem bobbit_worm_days : 
  initial_fish - (fish_eaten_per_day * days_before_adding) + fish_added - (fish_eaten_per_day * days_after_adding) = final_fish ∧
  days_before_adding = 14 := by sorry

end NUMINAMATH_CALUDE_bobbit_worm_days_l3353_335389


namespace NUMINAMATH_CALUDE_rhombus_area_l3353_335399

/-- A rhombus with side length √113 and diagonals differing by 10 units has area (√201)² - 5√201 -/
theorem rhombus_area (s : ℝ) (d₁ d₂ : ℝ) : 
  s = Real.sqrt 113 →
  d₂ = d₁ + 10 →
  d₁ * d₂ = 4 * s^2 →
  (1/2) * d₁ * d₂ = (Real.sqrt 201)^2 - 5 * Real.sqrt 201 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l3353_335399


namespace NUMINAMATH_CALUDE_family_spending_proof_l3353_335308

def planned_spending (family_size : ℕ) (orange_cost : ℚ) (savings_percentage : ℚ) : ℚ :=
  (family_size : ℚ) * orange_cost / (savings_percentage / 100)

theorem family_spending_proof :
  let family_size : ℕ := 4
  let orange_cost : ℚ := 3/2
  let savings_percentage : ℚ := 40
  planned_spending family_size orange_cost savings_percentage = 15 := by
sorry

end NUMINAMATH_CALUDE_family_spending_proof_l3353_335308


namespace NUMINAMATH_CALUDE_boat_round_trip_time_l3353_335322

/-- Calculate the total time for a round trip by boat, given the boat's speed in standing water,
    the stream's speed, and the distance to the destination. -/
theorem boat_round_trip_time
  (boat_speed : ℝ)
  (stream_speed : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 16)
  (h2 : stream_speed = 2)
  (h3 : distance = 7200)
  : ∃ (total_time : ℝ), abs (total_time - 914.2857) < 0.0001 :=
by
  sorry


end NUMINAMATH_CALUDE_boat_round_trip_time_l3353_335322


namespace NUMINAMATH_CALUDE_prob_one_student_two_books_is_eight_ninths_l3353_335303

/-- The number of students --/
def num_students : ℕ := 3

/-- The number of books --/
def num_books : ℕ := 4

/-- The probability of exactly one student receiving two different books --/
def prob_one_student_two_books : ℚ := 8/9

/-- Theorem stating that the probability of exactly one student receiving two different books
    when four distinct books are randomly gifted to three students is equal to 8/9 --/
theorem prob_one_student_two_books_is_eight_ninths :
  prob_one_student_two_books = 8/9 := by
  sorry


end NUMINAMATH_CALUDE_prob_one_student_two_books_is_eight_ninths_l3353_335303


namespace NUMINAMATH_CALUDE_shaded_area_in_circumscribed_square_l3353_335341

/-- Given a square with side length 20 cm circumscribed around a circle,
    where two of its diagonals form an isosceles triangle with the circle's center,
    the sum of the areas of the two small shaded regions is 100π - 200 square centimeters. -/
theorem shaded_area_in_circumscribed_square (π : ℝ) :
  let square_side : ℝ := 20
  let circle_radius : ℝ := square_side * Real.sqrt 2 / 2
  let sector_area : ℝ := π * circle_radius^2 / 4
  let triangle_area : ℝ := circle_radius^2 / 2
  let shaded_area : ℝ := 2 * (sector_area - triangle_area)
  shaded_area = 100 * π - 200 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_in_circumscribed_square_l3353_335341


namespace NUMINAMATH_CALUDE_modulus_of_one_minus_i_times_one_plus_i_l3353_335368

theorem modulus_of_one_minus_i_times_one_plus_i : 
  Complex.abs ((1 - Complex.I) * (1 + Complex.I)) = 2 := by sorry

end NUMINAMATH_CALUDE_modulus_of_one_minus_i_times_one_plus_i_l3353_335368


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3353_335385

-- Define the polynomial
def polynomial (a b c d : ℝ) (x : ℂ) : ℂ :=
  x^4 + a*x^3 + b*x^2 + c*x + d

-- Define the root
def root : ℂ := 2 + Complex.I

-- Theorem statement
theorem sum_of_coefficients (a b c d : ℝ) : 
  polynomial a b c d root = 0 → a + b + c + d = 10 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3353_335385


namespace NUMINAMATH_CALUDE_seating_uncertainty_l3353_335375

-- Define the types for people and seats
inductive Person : Type
| Abby : Person
| Bret : Person
| Carl : Person
| Dana : Person

inductive Seat : Type
| One : Seat
| Two : Seat
| Three : Seat
| Four : Seat

-- Define the seating arrangement
def Seating := Person → Seat

-- Define the "next to" relation
def next_to (s : Seating) (p1 p2 : Person) : Prop :=
  (s p1 = Seat.One ∧ s p2 = Seat.Two) ∨
  (s p1 = Seat.Two ∧ s p2 = Seat.Three) ∨
  (s p1 = Seat.Three ∧ s p2 = Seat.Four) ∨
  (s p2 = Seat.One ∧ s p1 = Seat.Two) ∨
  (s p2 = Seat.Two ∧ s p1 = Seat.Three) ∨
  (s p2 = Seat.Three ∧ s p1 = Seat.Four)

-- Define the "between" relation
def between (s : Seating) (p1 p2 p3 : Person) : Prop :=
  (s p1 = Seat.One ∧ s p2 = Seat.Two ∧ s p3 = Seat.Three) ∨
  (s p1 = Seat.Two ∧ s p2 = Seat.Three ∧ s p3 = Seat.Four) ∨
  (s p3 = Seat.One ∧ s p2 = Seat.Two ∧ s p1 = Seat.Three) ∨
  (s p3 = Seat.Two ∧ s p2 = Seat.Three ∧ s p1 = Seat.Four)

theorem seating_uncertainty (s : Seating) :
  (next_to s Person.Dana Person.Bret) ∧
  (¬ between s Person.Abby Person.Bret Person.Carl) ∧
  (s Person.Bret = Seat.One) →
  ¬ (∀ p : Person, s p = Seat.Three → (p = Person.Abby ∨ p = Person.Carl)) :=
by sorry

end NUMINAMATH_CALUDE_seating_uncertainty_l3353_335375


namespace NUMINAMATH_CALUDE_sum_five_consecutive_integers_l3353_335314

theorem sum_five_consecutive_integers (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 5 * n := by
  sorry

end NUMINAMATH_CALUDE_sum_five_consecutive_integers_l3353_335314


namespace NUMINAMATH_CALUDE_cucumber_weight_problem_l3353_335334

/-- Given cucumbers that are initially 99% water by weight, then 96% water by weight after
    evaporation with a new weight of 25 pounds, prove that the initial weight was 100 pounds. -/
theorem cucumber_weight_problem (initial_water_percent : ℝ) (final_water_percent : ℝ) (final_weight : ℝ) :
  initial_water_percent = 0.99 →
  final_water_percent = 0.96 →
  final_weight = 25 →
  ∃ (initial_weight : ℝ), initial_weight = 100 ∧
    (1 - initial_water_percent) * initial_weight = (1 - final_water_percent) * final_weight :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_problem_l3353_335334


namespace NUMINAMATH_CALUDE_paper_clip_distribution_l3353_335381

theorem paper_clip_distribution (total_clips : ℕ) (clips_per_box : ℕ) (boxes : ℕ) : 
  total_clips = 81 → 
  clips_per_box = 9 → 
  total_clips = boxes * clips_per_box → 
  boxes = 9 := by
sorry

end NUMINAMATH_CALUDE_paper_clip_distribution_l3353_335381


namespace NUMINAMATH_CALUDE_inequality_proof_l3353_335387

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_ineq : a + b < 2 * c) : 
  c - Real.sqrt (c^2 - a*b) < a ∧ a < c + Real.sqrt (c^2 - a*b) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3353_335387


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l3353_335395

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set A
def A : Finset Nat := {1, 5}

-- Define set B
def B : Finset Nat := {2, 4}

-- Theorem statement
theorem complement_intersection_theorem :
  (U \ A) ∩ B = {2, 4} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l3353_335395


namespace NUMINAMATH_CALUDE_equal_roots_cubic_l3353_335360

theorem equal_roots_cubic (k : ℝ) :
  (∃ a b : ℝ, (3 * a^3 + 9 * a^2 - 150 * a + k = 0) ∧
              (3 * b^3 + 9 * b^2 - 150 * b + k = 0) ∧
              (a ≠ b)) ∧
  (∃ x : ℝ, (3 * x^3 + 9 * x^2 - 150 * x + k = 0) ∧
            (∃ y : ℝ, y ≠ x ∧ 3 * y^3 + 9 * y^2 - 150 * y + k = 0)) ∧
  (k > 0) →
  k = 950 / 27 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_cubic_l3353_335360


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3353_335315

open Real

noncomputable def f (x : ℝ) : ℝ := -log (x^2 - 3*x + 2)

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Iio 1) := by sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3353_335315


namespace NUMINAMATH_CALUDE_unique_k_satisfying_conditions_l3353_335363

/-- A sequence of binomial coefficients forms an arithmetic progression -/
def is_arithmetic_progression (n : ℕ) (j : ℕ) (k : ℕ) : Prop :=
  ∃ d : ℤ, ∀ i : ℕ, i < k → (n.choose (j + i + 1) : ℤ) - (n.choose (j + i) : ℤ) = d

/-- Condition for part a) -/
def condition_a (k : ℕ) : Prop :=
  ∀ n : ℕ, ¬∃ j : ℕ, j ≤ n - k + 1 ∧ is_arithmetic_progression n j k

/-- Condition for part b) -/
def condition_b (k : ℕ) : Prop :=
  ∃ n : ℕ, ∃ j : ℕ, j ≤ n - k + 2 ∧ is_arithmetic_progression n j (k - 1)

/-- The main theorem -/
theorem unique_k_satisfying_conditions :
  ∃! k : ℕ, k > 0 ∧ condition_a k ∧ condition_b k :=
sorry

end NUMINAMATH_CALUDE_unique_k_satisfying_conditions_l3353_335363


namespace NUMINAMATH_CALUDE_total_coins_l3353_335392

theorem total_coins (quarters_piles dimes_piles nickels_piles pennies_piles : ℕ)
  (quarters_per_pile dimes_per_pile nickels_per_pile pennies_per_pile : ℕ)
  (h1 : quarters_piles = 5)
  (h2 : dimes_piles = 5)
  (h3 : nickels_piles = 3)
  (h4 : pennies_piles = 4)
  (h5 : quarters_per_pile = 3)
  (h6 : dimes_per_pile = 3)
  (h7 : nickels_per_pile = 4)
  (h8 : pennies_per_pile = 5) :
  quarters_piles * quarters_per_pile +
  dimes_piles * dimes_per_pile +
  nickels_piles * nickels_per_pile +
  pennies_piles * pennies_per_pile = 62 :=
by sorry

end NUMINAMATH_CALUDE_total_coins_l3353_335392


namespace NUMINAMATH_CALUDE_inequality_proof_l3353_335311

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1 + a)^4 * (1 + b)^4 ≥ 64 * a * b * (a + b)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3353_335311


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3353_335304

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -2] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3353_335304


namespace NUMINAMATH_CALUDE_fraction_simplification_l3353_335371

theorem fraction_simplification : (3 : ℚ) / (2 - 2 / 5) = 15 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3353_335371


namespace NUMINAMATH_CALUDE_sea_lion_count_l3353_335337

/-- Given the ratio of sea lions to penguins and their difference, 
    calculate the number of sea lions -/
theorem sea_lion_count (s p : ℕ) : 
  s * 11 = p * 4 →  -- ratio of sea lions to penguins is 4:11
  p = s + 84 →      -- 84 more penguins than sea lions
  s = 48 :=         -- prove that there are 48 sea lions
by sorry

end NUMINAMATH_CALUDE_sea_lion_count_l3353_335337


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3353_335393

/-- Given a quadratic equation mx^2 + nx - (m+n) = 0, prove that:
    1. The equation has two real roots.
    2. If n = 1 and the product of the roots is greater than 1, then -1/2 < m < 0. -/
theorem quadratic_equation_properties (m n : ℝ) :
  let f : ℝ → ℝ := λ x => m * x^2 + n * x - (m + n)
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧
  (n = 1 → (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ * x₂ > 1 → -1/2 < m ∧ m < 0)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3353_335393


namespace NUMINAMATH_CALUDE_prob_green_face_is_half_l3353_335346

/-- A six-faced dice with colored faces -/
structure ColoredDice :=
  (total_faces : ℕ)
  (green_faces : ℕ)
  (yellow_faces : ℕ)
  (purple_faces : ℕ)
  (h_total : total_faces = 6)
  (h_green : green_faces = 3)
  (h_yellow : yellow_faces = 2)
  (h_purple : purple_faces = 1)
  (h_sum : green_faces + yellow_faces + purple_faces = total_faces)

/-- The probability of rolling a green face on the colored dice -/
def prob_green_face (d : ColoredDice) : ℚ :=
  d.green_faces / d.total_faces

/-- Theorem: The probability of rolling a green face is 1/2 -/
theorem prob_green_face_is_half (d : ColoredDice) :
  prob_green_face d = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_prob_green_face_is_half_l3353_335346


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l3353_335338

theorem fractional_equation_solution :
  ∃ (x : ℝ), x ≠ 1 ∧ (3 / (x - 1) = 5 + 3 * x / (1 - x)) ↔ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l3353_335338


namespace NUMINAMATH_CALUDE_positive_numbers_inequality_l3353_335326

theorem positive_numbers_inequality (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum_squares : a^2 + b^2 + 4*c^2 = 3) : 
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_inequality_l3353_335326


namespace NUMINAMATH_CALUDE_log_relation_l3353_335324

theorem log_relation (a b : ℝ) (ha : a = Real.log 225 / Real.log 8) (hb : b = Real.log 15 / Real.log 2) : 
  a = (2 * b) / 3 := by
  sorry

end NUMINAMATH_CALUDE_log_relation_l3353_335324


namespace NUMINAMATH_CALUDE_four_correct_propositions_l3353_335313

theorem four_correct_propositions (x y : ℝ) :
  (((x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0) ∧
   ((x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0)) ∧
   (¬((x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0)) ∧
   ((x^2 + y^2 = 0) → (x = 0 ∧ y = 0))) :=
by sorry

end NUMINAMATH_CALUDE_four_correct_propositions_l3353_335313


namespace NUMINAMATH_CALUDE_complete_residue_system_l3353_335331

theorem complete_residue_system (m : ℕ) (x : Fin m → ℤ) 
  (h : ∀ i j : Fin m, i ≠ j → x i % m ≠ x j % m) :
  ∀ k : Fin m, ∃ i : Fin m, x i % m = k.val :=
sorry

end NUMINAMATH_CALUDE_complete_residue_system_l3353_335331


namespace NUMINAMATH_CALUDE_correct_calculation_l3353_335384

theorem correct_calculation (a b : ℝ) : 3 * a * b - 2 * a * b = a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3353_335384


namespace NUMINAMATH_CALUDE_route_length_l3353_335386

/-- Given two trains traveling on a route, prove the length of the route. -/
theorem route_length : 
  ∀ (route_length : ℝ) (train_x_speed : ℝ) (train_y_speed : ℝ),
  train_x_speed > 0 →
  train_y_speed > 0 →
  route_length / train_x_speed = 5 →
  route_length / train_y_speed = 4 →
  80 / train_x_speed + (route_length - 80) / train_y_speed = route_length / train_y_speed →
  route_length = 180 := by
  sorry

end NUMINAMATH_CALUDE_route_length_l3353_335386


namespace NUMINAMATH_CALUDE_max_min_sum_zero_l3353_335377

def f (x : ℝ) := x^3 - 3*x

theorem max_min_sum_zero :
  ∃ (m n : ℝ), (∀ x, f x ≤ m) ∧ (∃ x₁, f x₁ = m) ∧
                (∀ x, n ≤ f x) ∧ (∃ x₂, f x₂ = n) ∧
                (m + n = 0) := by sorry

end NUMINAMATH_CALUDE_max_min_sum_zero_l3353_335377


namespace NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3353_335342

/-- The number of ways to distribute distinguishable balls into distinguishable boxes -/
def distribute_balls (num_balls : ℕ) (num_boxes : ℕ) : ℕ := num_boxes ^ num_balls

/-- Theorem: There are 729 ways to distribute 6 distinguishable balls into 3 distinguishable boxes -/
theorem distribute_six_balls_three_boxes : distribute_balls 6 3 = 729 := by
  sorry

end NUMINAMATH_CALUDE_distribute_six_balls_three_boxes_l3353_335342


namespace NUMINAMATH_CALUDE_total_seeds_equals_685_l3353_335320

/- Morning plantings -/
def mike_morning_tomato : ℕ := 50
def mike_morning_pepper : ℕ := 30
def ted_morning_tomato : ℕ := 2 * mike_morning_tomato
def ted_morning_pepper : ℕ := mike_morning_pepper / 2
def sarah_morning_tomato : ℕ := mike_morning_tomato + 30
def sarah_morning_pepper : ℕ := mike_morning_pepper + 30

/- Afternoon plantings -/
def mike_afternoon_tomato : ℕ := 60
def mike_afternoon_pepper : ℕ := 40
def ted_afternoon_tomato : ℕ := mike_afternoon_tomato - 20
def ted_afternoon_pepper : ℕ := mike_afternoon_pepper
def sarah_afternoon_tomato : ℕ := sarah_morning_tomato + 20
def sarah_afternoon_pepper : ℕ := sarah_morning_pepper + 10

/- Total seeds planted -/
def total_seeds : ℕ := 
  mike_morning_tomato + mike_morning_pepper + 
  ted_morning_tomato + ted_morning_pepper + 
  sarah_morning_tomato + sarah_morning_pepper + 
  mike_afternoon_tomato + mike_afternoon_pepper + 
  ted_afternoon_tomato + ted_afternoon_pepper + 
  sarah_afternoon_tomato + sarah_afternoon_pepper

theorem total_seeds_equals_685 : total_seeds = 685 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_equals_685_l3353_335320


namespace NUMINAMATH_CALUDE_total_cards_l3353_335390

theorem total_cards (hockey_cards : ℕ) 
  (h1 : hockey_cards = 200)
  (h2 : ∃ football_cards : ℕ, football_cards = 4 * hockey_cards)
  (h3 : ∃ baseball_cards : ℕ, baseball_cards = football_cards - 50) :
  ∃ total_cards : ℕ, total_cards = hockey_cards + football_cards + baseball_cards ∧ total_cards = 1750 :=
by
  sorry

end NUMINAMATH_CALUDE_total_cards_l3353_335390


namespace NUMINAMATH_CALUDE_dog_speed_is_400_l3353_335373

-- Define the constants from the problem
def football_fields : ℕ := 6
def yards_per_field : ℕ := 200
def fetch_time_minutes : ℕ := 9
def feet_per_yard : ℕ := 3

-- Define the dog's speed as a function
def dog_speed : ℚ :=
  (football_fields * yards_per_field * feet_per_yard) / fetch_time_minutes

-- Theorem statement
theorem dog_speed_is_400 : dog_speed = 400 := by
  sorry

end NUMINAMATH_CALUDE_dog_speed_is_400_l3353_335373


namespace NUMINAMATH_CALUDE_unique_four_digit_square_l3353_335356

def is_consecutive_digits (n : ℕ) : Prop :=
  ∃ (a : ℕ), a < 10 ∧ n = a * 1000 + (a + 1) * 100 + (a + 2) * 10 + (a + 3)

def swap_first_two_digits (n : ℕ) : ℕ :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let rest := n % 100
  d2 * 1000 + d1 * 100 + rest

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem unique_four_digit_square : 
  ∀ (n : ℕ), 1000 ≤ n ∧ n < 10000 →
    (is_consecutive_digits n ∧ 
     is_perfect_square (swap_first_two_digits n)) ↔ 
    n = 4356 := by sorry

end NUMINAMATH_CALUDE_unique_four_digit_square_l3353_335356


namespace NUMINAMATH_CALUDE_equation_solutions_l3353_335345

theorem equation_solutions (x : ℝ) : 
  (x - 1)^2 * (x - 5)^2 / (x - 5) = 4 ↔ x = 3 ∨ x = -1 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3353_335345


namespace NUMINAMATH_CALUDE_largest_common_value_l3353_335359

def first_progression (n : ℕ) : ℕ := 4 + 5 * n

def second_progression (m : ℕ) : ℕ := 5 + 10 * m

theorem largest_common_value :
  ∃ (n m : ℕ),
    first_progression n = second_progression m ∧
    first_progression n < 1000 ∧
    first_progression n ≡ 1 [MOD 4] ∧
    ∀ (k l : ℕ),
      first_progression k = second_progression l →
      first_progression k < 1000 →
      first_progression k ≡ 1 [MOD 4] →
      first_progression k ≤ first_progression n :=
by
  sorry

end NUMINAMATH_CALUDE_largest_common_value_l3353_335359


namespace NUMINAMATH_CALUDE_hcl_concentration_in_mixed_solution_l3353_335327

/-- Calculates the concentration of HCl in a mixed solution -/
theorem hcl_concentration_in_mixed_solution 
  (volume1 : ℝ) (concentration1 : ℝ) 
  (volume2 : ℝ) (concentration2 : ℝ) :
  volume1 = 60 →
  concentration1 = 0.4 →
  volume2 = 90 →
  concentration2 = 0.15 →
  (volume1 * concentration1 + volume2 * concentration2) / (volume1 + volume2) = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_hcl_concentration_in_mixed_solution_l3353_335327
