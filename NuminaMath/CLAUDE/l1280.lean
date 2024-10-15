import Mathlib

namespace NUMINAMATH_CALUDE_real_estate_transaction_result_l1280_128047

/-- Represents the result of a transaction -/
inductive TransactionResult
  | Loss (amount : ℚ)
  | Gain (amount : ℚ)
  | NoChange

/-- Calculates the result of a real estate transaction -/
def calculateTransactionResult (houseSalePrice storeSalePrice : ℚ) 
                               (houseLossPercentage storeGainPercentage : ℚ) : TransactionResult :=
  let houseCost := houseSalePrice / (1 - houseLossPercentage)
  let storeCost := storeSalePrice / (1 + storeGainPercentage)
  let totalCost := houseCost + storeCost
  let totalSale := houseSalePrice + storeSalePrice
  let difference := totalCost - totalSale
  if difference > 0 then TransactionResult.Loss difference
  else if difference < 0 then TransactionResult.Gain (-difference)
  else TransactionResult.NoChange

/-- Theorem stating the result of the specific real estate transaction -/
theorem real_estate_transaction_result :
  calculateTransactionResult 15000 15000 (30/100) (25/100) = TransactionResult.Loss (3428.57/100) := by
  sorry

end NUMINAMATH_CALUDE_real_estate_transaction_result_l1280_128047


namespace NUMINAMATH_CALUDE_contest_team_mistakes_l1280_128023

/-- The number of incorrect answers for a team in a math contest -/
def team_incorrect_answers (total_questions : ℕ) (riley_mistakes : ℕ) (ofelia_correct_offset : ℕ) : ℕ :=
  let riley_correct := total_questions - riley_mistakes
  let ofelia_correct := riley_correct / 2 + ofelia_correct_offset
  let ofelia_mistakes := total_questions - ofelia_correct
  riley_mistakes + ofelia_mistakes

/-- Theorem stating the number of incorrect answers for Riley and Ofelia's team -/
theorem contest_team_mistakes :
  team_incorrect_answers 35 3 5 = 17 := by
  sorry

end NUMINAMATH_CALUDE_contest_team_mistakes_l1280_128023


namespace NUMINAMATH_CALUDE_john_ray_difference_l1280_128084

/-- The number of chickens each person took -/
structure ChickenCount where
  mary : ℕ
  john : ℕ
  ray : ℕ

/-- The conditions of the chicken problem -/
def chicken_problem (c : ChickenCount) : Prop :=
  c.john = c.mary + 5 ∧
  c.mary = c.ray + 6 ∧
  c.ray = 10

/-- The theorem stating John took 11 more chickens than Ray -/
theorem john_ray_difference (c : ChickenCount) 
  (h : chicken_problem c) : c.john - c.ray = 11 := by
  sorry


end NUMINAMATH_CALUDE_john_ray_difference_l1280_128084


namespace NUMINAMATH_CALUDE_train_length_l1280_128054

/-- Calculates the length of a train given its speed, tunnel length, and time to pass through -/
theorem train_length (train_speed : ℝ) (tunnel_length : ℝ) (time_to_pass : ℝ) :
  train_speed = 72 →
  tunnel_length = 3.5 →
  time_to_pass = 3 / 60 →
  (train_speed * time_to_pass - tunnel_length) * 1000 = 100 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1280_128054


namespace NUMINAMATH_CALUDE_g_of_seven_l1280_128007

theorem g_of_seven (g : ℝ → ℝ) (h : ∀ x : ℝ, g (3 * x - 2) = 5 * x + 4) : g 7 = 19 := by
  sorry

end NUMINAMATH_CALUDE_g_of_seven_l1280_128007


namespace NUMINAMATH_CALUDE_sin_870_degrees_l1280_128062

theorem sin_870_degrees : Real.sin (870 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_870_degrees_l1280_128062


namespace NUMINAMATH_CALUDE_senior_discount_percentage_l1280_128083

def original_cost : ℚ := 7.5
def coupon_discount : ℚ := 2.5
def final_payment : ℚ := 4

def cost_after_coupon : ℚ := original_cost - coupon_discount
def senior_discount_amount : ℚ := cost_after_coupon - final_payment

theorem senior_discount_percentage :
  (senior_discount_amount / cost_after_coupon) * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_senior_discount_percentage_l1280_128083


namespace NUMINAMATH_CALUDE_inequality_transformations_l1280_128088

theorem inequality_transformations (a b : ℝ) (h : a < b) :
  (a + 2 < b + 2) ∧ 
  (3 * a < 3 * b) ∧ 
  ((1/2) * a < (1/2) * b) ∧ 
  (-2 * a > -2 * b) :=
by sorry

end NUMINAMATH_CALUDE_inequality_transformations_l1280_128088


namespace NUMINAMATH_CALUDE_intersection_M_N_l1280_128086

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1280_128086


namespace NUMINAMATH_CALUDE_wendy_recycling_points_l1280_128092

/-- Calculates the points earned from recycling bags -/
def points_earned (total_bags : ℕ) (unrecycled_bags : ℕ) (points_per_bag : ℕ) : ℕ :=
  (total_bags - unrecycled_bags) * points_per_bag

/-- Proves that Wendy earns 210 points from recycling bags -/
theorem wendy_recycling_points :
  let total_bags : ℕ := 25
  let unrecycled_bags : ℕ := 4
  let points_per_bag : ℕ := 10
  points_earned total_bags unrecycled_bags points_per_bag = 210 :=
by
  sorry

end NUMINAMATH_CALUDE_wendy_recycling_points_l1280_128092


namespace NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l1280_128089

/-- Returns true if the given natural number contains the digit 7 -/
def containsSeven (n : ℕ) : Prop :=
  ∃ k : ℕ, n / (10 ^ k) % 10 = 7

theorem smallest_n_with_seven_in_squares : 
  ∀ n : ℕ, n < 26 → ¬(containsSeven (n^2) ∧ containsSeven ((n+1)^2)) ∧
  (containsSeven (26^2) ∧ containsSeven (27^2)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_seven_in_squares_l1280_128089


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l1280_128072

theorem complex_ratio_theorem (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 3) 
  (h₂ : Complex.abs z₂ = 5) 
  (h₃ : Complex.abs (z₁ - z₂) = 7) : 
  z₁ / z₂ = (3 / 5 : ℂ) * (-1 / 2 + Complex.I * Real.sqrt 3 / 2) ∨
  z₁ / z₂ = (3 / 5 : ℂ) * (-1 / 2 - Complex.I * Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l1280_128072


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1280_128063

theorem sum_of_reciprocals (x y : ℝ) 
  (h1 : 1/x + 1/y = 5) 
  (h2 : 1/x - 1/y = -3) : 
  x + y = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1280_128063


namespace NUMINAMATH_CALUDE_range_of_y_minus_2x_l1280_128028

theorem range_of_y_minus_2x (x y : ℝ) 
  (hx : -2 ≤ x ∧ x ≤ 1) 
  (hy : 2 ≤ y ∧ y ≤ 4) : 
  0 ≤ y - 2*x ∧ y - 2*x ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_y_minus_2x_l1280_128028


namespace NUMINAMATH_CALUDE_power_sum_equality_l1280_128045

theorem power_sum_equality : 2^345 + 3^5 * 3^3 = 2^345 + 6561 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l1280_128045


namespace NUMINAMATH_CALUDE_smallest_k_value_l1280_128001

theorem smallest_k_value : ∃ (k : ℕ), k > 0 ∧
  (∀ (k' : ℕ), k' > 0 →
    (∃ (n : ℕ), n > 0 ∧ 2000 < n ∧ n < 3000 ∧
      (∀ (i : ℕ), 2 ≤ i ∧ i ≤ k' → n % i = i - 1)) →
    k ≤ k') ∧
  k = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_k_value_l1280_128001


namespace NUMINAMATH_CALUDE_triangle_area_l1280_128096

/-- The area of a triangle ABC with given side lengths and angle relationship -/
theorem triangle_area (a b : ℝ) (h1 : a = 5) (h2 : b = 4) (h3 : Real.cos (A - B) = 31/32) :
  (1/2) * a * b * Real.sin C = (15 * Real.sqrt 7) / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l1280_128096


namespace NUMINAMATH_CALUDE_gcd_2703_1113_l1280_128076

theorem gcd_2703_1113 : Nat.gcd 2703 1113 = 159 := by
  sorry

end NUMINAMATH_CALUDE_gcd_2703_1113_l1280_128076


namespace NUMINAMATH_CALUDE_expression_value_l1280_128057

theorem expression_value (x y : ℝ) (h : 2 * x + y = 6) :
  ((x - y)^2 - (x + y)^2 + y * (2 * x - y)) / (-2 * y) = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1280_128057


namespace NUMINAMATH_CALUDE_crayons_to_mary_l1280_128048

def crayons_given_to_mary (new_pack : ℕ) (locker : ℕ) : ℕ :=
  let initial_total := new_pack + locker
  let from_bobby := locker / 2
  let final_total := initial_total + from_bobby
  final_total / 3

theorem crayons_to_mary :
  crayons_given_to_mary 21 36 = 25 := by
  sorry

end NUMINAMATH_CALUDE_crayons_to_mary_l1280_128048


namespace NUMINAMATH_CALUDE_divisibility_by_three_l1280_128012

theorem divisibility_by_three (d : Nat) : 
  d ≤ 9 → (15780 + d) % 3 = 0 ↔ d = 0 ∨ d = 3 ∨ d = 6 ∨ d = 9 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_three_l1280_128012


namespace NUMINAMATH_CALUDE_knit_socks_together_l1280_128051

/-- The number of days it takes for two people to knit a certain number of socks together -/
def days_to_knit (a_rate b_rate : ℚ) (pairs : ℚ) : ℚ :=
  pairs / (a_rate + b_rate)

/-- Theorem: Given the rates at which A and B can knit socks individually, 
    prove that they can knit two pairs of socks in 4 days when working together -/
theorem knit_socks_together 
  (a_rate : ℚ) (b_rate : ℚ) 
  (ha : a_rate = 1 / 3) 
  (hb : b_rate = 1 / 6) : 
  days_to_knit a_rate b_rate 2 = 4 := by
  sorry

#eval days_to_knit (1/3) (1/6) 2

end NUMINAMATH_CALUDE_knit_socks_together_l1280_128051


namespace NUMINAMATH_CALUDE_tutor_reunion_proof_l1280_128000

/-- The number of school days until all tutors work together again -/
def tutor_reunion_days : ℕ := 360

/-- Elisa's work schedule (every 5th day) -/
def elisa_schedule : ℕ := 5

/-- Frank's work schedule (every 6th day) -/
def frank_schedule : ℕ := 6

/-- Giselle's work schedule (every 8th day) -/
def giselle_schedule : ℕ := 8

/-- Hector's work schedule (every 9th day) -/
def hector_schedule : ℕ := 9

theorem tutor_reunion_proof :
  Nat.lcm elisa_schedule (Nat.lcm frank_schedule (Nat.lcm giselle_schedule hector_schedule)) = tutor_reunion_days :=
by sorry

end NUMINAMATH_CALUDE_tutor_reunion_proof_l1280_128000


namespace NUMINAMATH_CALUDE_parallelogram_height_l1280_128031

theorem parallelogram_height (area base height : ℝ) : 
  area = 231 ∧ base = 21 ∧ area = base * height → height = 11 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l1280_128031


namespace NUMINAMATH_CALUDE_multiply_72519_and_9999_l1280_128017

theorem multiply_72519_and_9999 : 72519 * 9999 = 724817481 := by
  sorry

end NUMINAMATH_CALUDE_multiply_72519_and_9999_l1280_128017


namespace NUMINAMATH_CALUDE_complex_number_with_conditions_l1280_128075

theorem complex_number_with_conditions (z : ℂ) :
  Complex.abs z = 1 →
  ∃ (y : ℝ), (3 + 4*I) * z = y*I →
  z = 4/5 - 3/5*I ∨ z = -4/5 + 3/5*I :=
by sorry

end NUMINAMATH_CALUDE_complex_number_with_conditions_l1280_128075


namespace NUMINAMATH_CALUDE_at_least_one_positive_negation_l1280_128090

theorem at_least_one_positive_negation (a b c : ℝ) :
  (¬ (a > 0 ∨ b > 0 ∨ c > 0)) ↔ (a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_positive_negation_l1280_128090


namespace NUMINAMATH_CALUDE_red_marbles_count_l1280_128058

/-- The number of red marbles Mary gave to Dan -/
def red_marbles : ℕ := 78 - 64

theorem red_marbles_count : red_marbles = 14 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_count_l1280_128058


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l1280_128021

theorem regular_polygon_sides (D : ℕ) (n : ℕ) : D = 15 → n * (n - 3) / 2 = D → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l1280_128021


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1280_128035

/-- Given a > 0 and a ≠ 1, the function f(x) = a^(x+1) + 1 always passes through the point (-1, 2) -/
theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x + 1) + 1
  f (-1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l1280_128035


namespace NUMINAMATH_CALUDE_min_likes_mozart_and_beethoven_l1280_128049

/-- Given a survey of 150 people where 120 liked Mozart and 80 liked Beethoven,
    the minimum number of people who liked both Mozart and Beethoven is 50. -/
theorem min_likes_mozart_and_beethoven
  (total : ℕ) (likes_mozart : ℕ) (likes_beethoven : ℕ)
  (h_total : total = 150)
  (h_mozart : likes_mozart = 120)
  (h_beethoven : likes_beethoven = 80) :
  (likes_mozart + likes_beethoven - total : ℤ).natAbs ≥ 50 := by
  sorry


end NUMINAMATH_CALUDE_min_likes_mozart_and_beethoven_l1280_128049


namespace NUMINAMATH_CALUDE_deposit_percentage_l1280_128034

theorem deposit_percentage (deposit : ℝ) (remaining : ℝ) : 
  deposit = 55 → remaining = 495 → (deposit / (deposit + remaining)) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_deposit_percentage_l1280_128034


namespace NUMINAMATH_CALUDE_square_field_perimeter_l1280_128005

theorem square_field_perimeter (a p : ℝ) (h1 : a ≥ 0) (h2 : p > 0) 
  (h3 : 6 * a = 6 * (2 * p + 9)) (h4 : a = a^2) : p = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_field_perimeter_l1280_128005


namespace NUMINAMATH_CALUDE_no_divisible_with_small_digit_sum_l1280_128022

/-- Represents a number consisting of m ones -/
def ones (m : ℕ) : ℕ := 
  (10^m - 1) / 9

/-- Calculates the digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digitSum (n / 10)

/-- Theorem stating that no natural number divisible by ones(m) has a digit sum less than m -/
theorem no_divisible_with_small_digit_sum (m : ℕ) : 
  ¬ ∃ (n : ℕ), (n % ones m = 0) ∧ (digitSum n < m) := by
  sorry

end NUMINAMATH_CALUDE_no_divisible_with_small_digit_sum_l1280_128022


namespace NUMINAMATH_CALUDE_x_greater_than_y_l1280_128013

theorem x_greater_than_y (x y : ℝ) (h : y = (1 - 0.9444444444444444) * x) : 
  x = 18 * y := by sorry

end NUMINAMATH_CALUDE_x_greater_than_y_l1280_128013


namespace NUMINAMATH_CALUDE_exam_average_proof_l1280_128094

theorem exam_average_proof (n₁ n₂ : ℕ) (avg₁ avg_total : ℚ) (h₁ : n₁ = 15) (h₂ : n₂ = 10)
  (h₃ : avg₁ = 75/100) (h₄ : avg_total = 81/100) (h₅ : n₁ + n₂ = 25) :
  let avg₂ := (((n₁ + n₂ : ℚ) * avg_total) - (n₁ * avg₁)) / n₂
  avg₂ = 90/100 := by
sorry

end NUMINAMATH_CALUDE_exam_average_proof_l1280_128094


namespace NUMINAMATH_CALUDE_smallest_n_for_probability_condition_l1280_128082

theorem smallest_n_for_probability_condition : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → (((m : ℝ) - 4)^3 / m^3 > 1/2) → m ≥ n) ∧
  ((n : ℝ) - 4)^3 / n^3 > 1/2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_probability_condition_l1280_128082


namespace NUMINAMATH_CALUDE_odd_cube_plus_three_square_minus_linear_minus_three_divisible_by_48_l1280_128038

theorem odd_cube_plus_three_square_minus_linear_minus_three_divisible_by_48 (x : ℤ) (h : ∃ k : ℤ, x = 2*k + 1) :
  ∃ m : ℤ, x^3 + 3*x^2 - x - 3 = 48*m := by
sorry

end NUMINAMATH_CALUDE_odd_cube_plus_three_square_minus_linear_minus_three_divisible_by_48_l1280_128038


namespace NUMINAMATH_CALUDE_type_a_cubes_count_l1280_128078

/-- Represents the dimensions of the rectangular solid -/
def solid_dimensions : Fin 3 → ℕ
  | 0 => 120
  | 1 => 350
  | 2 => 400
  | _ => 0

/-- Calculates the number of cubes traversed by the diagonal -/
def total_cubes_traversed : ℕ := sorry

/-- The number of type A cubes traversed by the diagonal -/
def type_a_cubes : ℕ := total_cubes_traversed / 2

theorem type_a_cubes_count : type_a_cubes = 390 := by sorry

end NUMINAMATH_CALUDE_type_a_cubes_count_l1280_128078


namespace NUMINAMATH_CALUDE_correct_addition_after_digit_change_l1280_128003

theorem correct_addition_after_digit_change :
  let num1 : ℕ := 364765
  let num2 : ℕ := 951872
  let incorrect_sum : ℕ := 1496637
  let d : ℕ := 3
  let e : ℕ := 4
  let new_num1 : ℕ := num1 + 100000 * (e - d)
  let new_num2 : ℕ := num2
  let new_sum : ℕ := incorrect_sum + 100000 * (e - d)
  new_num1 + new_num2 = new_sum ∧ d + e = 7 :=
by sorry

end NUMINAMATH_CALUDE_correct_addition_after_digit_change_l1280_128003


namespace NUMINAMATH_CALUDE_percent_relation_l1280_128006

theorem percent_relation (a b : ℝ) (h : a = 1.25 * b) : 4 * b = 3.2 * a := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l1280_128006


namespace NUMINAMATH_CALUDE_hooligan_theorem_l1280_128018

-- Define the universe
variable (Person : Type)

-- Define predicates
variable (isHooligan : Person → Prop)
variable (hasBeatlesHaircut : Person → Prop)
variable (hasRudeDemeanor : Person → Prop)

-- State the theorem
theorem hooligan_theorem 
  (exists_beatles_hooligan : ∃ x, isHooligan x ∧ hasBeatlesHaircut x)
  (all_hooligans_rude : ∀ y, isHooligan y → hasRudeDemeanor y) :
  (∃ z, isHooligan z ∧ hasRudeDemeanor z ∧ hasBeatlesHaircut z) ∧
  ¬(∀ w, isHooligan w ∧ hasRudeDemeanor w → hasBeatlesHaircut w) :=
by sorry

end NUMINAMATH_CALUDE_hooligan_theorem_l1280_128018


namespace NUMINAMATH_CALUDE_cookie_remainder_percentage_l1280_128044

/-- Proves that given 600 initial cookies, if Nicole eats 2/5 of the total and Eduardo eats 3/5 of the remaining, then 24% of the original cookies remain. -/
theorem cookie_remainder_percentage (initial_cookies : ℕ) (nicole_fraction : ℚ) (eduardo_fraction : ℚ)
  (h_initial : initial_cookies = 600)
  (h_nicole : nicole_fraction = 2 / 5)
  (h_eduardo : eduardo_fraction = 3 / 5) :
  (initial_cookies - nicole_fraction * initial_cookies - eduardo_fraction * (initial_cookies - nicole_fraction * initial_cookies)) / initial_cookies = 24 / 100 := by
  sorry

#check cookie_remainder_percentage

end NUMINAMATH_CALUDE_cookie_remainder_percentage_l1280_128044


namespace NUMINAMATH_CALUDE_at_least_one_good_certain_l1280_128093

def total_products : ℕ := 12
def good_products : ℕ := 10
def defective_products : ℕ := 2
def picked_products : ℕ := 3

theorem at_least_one_good_certain :
  Fintype.card {s : Finset (Fin total_products) // s.card = picked_products ∧ ∃ x ∈ s, x.val < good_products} =
  Fintype.card {s : Finset (Fin total_products) // s.card = picked_products} :=
sorry

end NUMINAMATH_CALUDE_at_least_one_good_certain_l1280_128093


namespace NUMINAMATH_CALUDE_gcd_78_36_l1280_128085

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_36_l1280_128085


namespace NUMINAMATH_CALUDE_gcd_2146_1813_l1280_128019

theorem gcd_2146_1813 : Nat.gcd 2146 1813 = 37 := by sorry

end NUMINAMATH_CALUDE_gcd_2146_1813_l1280_128019


namespace NUMINAMATH_CALUDE_at_least_one_composite_l1280_128070

theorem at_least_one_composite (a b c : ℕ) 
  (h_odd_a : Odd a) (h_odd_b : Odd b) (h_odd_c : Odd c)
  (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_positive_c : 0 < c)
  (h_not_square : ¬∃k, a = k^2)
  (h_equation : a^2 + a + 1 = 3 * (b^2 + b + 1) * (c^2 + c + 1)) :
  (∃k > 1, k ∣ (b^2 + b + 1)) ∨ (∃k > 1, k ∣ (c^2 + c + 1)) :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_composite_l1280_128070


namespace NUMINAMATH_CALUDE_divisibility_by_nineteen_l1280_128027

theorem divisibility_by_nineteen (k : ℕ) : 19 ∣ (2^(26*k + 2) + 3) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_nineteen_l1280_128027


namespace NUMINAMATH_CALUDE_pentagon_largest_angle_l1280_128068

/-- 
Given a pentagon where:
- Angles increase sequentially by 10 degrees
- The sum of all angles is 540 degrees
Prove that the largest angle is 128 degrees
-/
theorem pentagon_largest_angle : 
  ∀ (a₁ a₂ a₃ a₄ a₅ : ℝ),
  a₂ = a₁ + 10 →
  a₃ = a₁ + 20 →
  a₄ = a₁ + 30 →
  a₅ = a₁ + 40 →
  a₁ + a₂ + a₃ + a₄ + a₅ = 540 →
  a₅ = 128 := by
sorry

end NUMINAMATH_CALUDE_pentagon_largest_angle_l1280_128068


namespace NUMINAMATH_CALUDE_additional_cars_during_play_l1280_128015

/-- Calculates the number of additional cars that parked during a play given the initial conditions. -/
theorem additional_cars_during_play
  (front_initial : ℕ)
  (back_initial : ℕ)
  (total_end : ℕ)
  (h1 : front_initial = 100)
  (h2 : back_initial = 2 * front_initial)
  (h3 : total_end = 700) :
  total_end - (front_initial + back_initial) = 300 :=
by sorry

end NUMINAMATH_CALUDE_additional_cars_during_play_l1280_128015


namespace NUMINAMATH_CALUDE_arithmetic_sequence_constant_l1280_128056

theorem arithmetic_sequence_constant (x y z k : ℝ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x ≠ y → y ≠ z → x ≠ z →
  k ≠ 1 →
  k * x = y →
  let u := y / x
  let v := z / y
  (u - 1/v) - (v - 1/u) = (v - 1/u) - (1/u - u) →
  ∃ (k' : ℝ), k' * x = z ∧ 2 * k / k' - 2 * k + k^2 / k' - 1 / k = 0 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_constant_l1280_128056


namespace NUMINAMATH_CALUDE_remainder_of_9876543210_div_101_l1280_128052

theorem remainder_of_9876543210_div_101 : 9876543210 % 101 = 100 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_9876543210_div_101_l1280_128052


namespace NUMINAMATH_CALUDE_fib_150_mod_5_l1280_128066

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- The 150th Fibonacci number modulo 5 is 0 -/
theorem fib_150_mod_5 : fib 149 % 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fib_150_mod_5_l1280_128066


namespace NUMINAMATH_CALUDE_equation_value_l1280_128060

theorem equation_value (x y : ℚ) 
  (eq1 : 5 * x + 6 * y = 7) 
  (eq2 : 3 * x + 5 * y = 6) : 
  x + 4 * y = 5 := by
sorry

end NUMINAMATH_CALUDE_equation_value_l1280_128060


namespace NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l1280_128055

/-- The equation of a circle with center (0, 1) tangent to the line y = 2 is x^2 + (y-1)^2 = 1 -/
theorem circle_equation_tangent_to_line (x y : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ x^2 + (y - 1)^2 = r^2 ∧ |2 - 1| = r) → 
  x^2 + (y - 1)^2 = 1 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_tangent_to_line_l1280_128055


namespace NUMINAMATH_CALUDE_a_101_value_l1280_128032

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  a 1 = 2 ∧ ∀ n : ℕ, a (n + 1) - a n = 1 / 2

theorem a_101_value (a : ℕ → ℚ) (h : arithmetic_sequence a) : a 101 = 52 := by
  sorry

end NUMINAMATH_CALUDE_a_101_value_l1280_128032


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1280_128030

/-- The function that constructs the number 5678N from a single digit N -/
def constructNumber (N : ℕ) : ℕ := 5678 * 10 + N

/-- Predicate to check if a natural number is a single digit -/
def isSingleDigit (n : ℕ) : Prop := n < 10

/-- Theorem stating that 4 is the largest single-digit number N such that 5678N is divisible by 6 -/
theorem largest_digit_divisible_by_six :
  ∀ N : ℕ, isSingleDigit N → N > 4 → ¬(constructNumber N % 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l1280_128030


namespace NUMINAMATH_CALUDE_line_A2A3_tangent_to_circle_M_l1280_128033

-- Define the parabola C
def parabola_C (x y : ℝ) : Prop := y^2 = x

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 1

-- Define a point on the parabola
def point_on_parabola (A : ℝ × ℝ) : Prop := parabola_C A.1 A.2

-- Define a line tangent to the circle
def line_tangent_to_circle (A B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), circle_M (A.1 + t * (B.1 - A.1)) (A.2 + t * (B.2 - A.2))

-- Main theorem
theorem line_A2A3_tangent_to_circle_M (A₁ A₂ A₃ : ℝ × ℝ) :
  point_on_parabola A₁ →
  point_on_parabola A₂ →
  point_on_parabola A₃ →
  line_tangent_to_circle A₁ A₂ →
  line_tangent_to_circle A₁ A₃ →
  line_tangent_to_circle A₂ A₃ :=
sorry

end NUMINAMATH_CALUDE_line_A2A3_tangent_to_circle_M_l1280_128033


namespace NUMINAMATH_CALUDE_existence_of_stabilization_l1280_128053

-- Define the function type
def PositiveIntegerFunction := ℕ+ → ℕ+

-- Define the conditions on the function
def SatisfiesConditions (f : PositiveIntegerFunction) : Prop :=
  (∀ m n : ℕ+, Nat.gcd (f m) (f n) ≤ (Nat.gcd m n) ^ 2014) ∧
  (∀ n : ℕ+, n ≤ f n ∧ f n ≤ n + 2014)

-- State the theorem
theorem existence_of_stabilization (f : PositiveIntegerFunction) 
  (h : SatisfiesConditions f) : 
  ∃ N : ℕ+, ∀ n : ℕ+, n ≥ N → f n = n := by
  sorry

end NUMINAMATH_CALUDE_existence_of_stabilization_l1280_128053


namespace NUMINAMATH_CALUDE_room_width_l1280_128002

/-- Given a rectangular room with length 21 m, surrounded by a 2 m wide veranda on all sides,
    and the veranda area is 148 m², prove that the width of the room is 12 m. -/
theorem room_width (room_length : ℝ) (veranda_width : ℝ) (veranda_area : ℝ) :
  room_length = 21 →
  veranda_width = 2 →
  veranda_area = 148 →
  ∃ (room_width : ℝ),
    (room_length + 2 * veranda_width) * (room_width + 2 * veranda_width) -
    room_length * room_width = veranda_area ∧
    room_width = 12 := by
  sorry

end NUMINAMATH_CALUDE_room_width_l1280_128002


namespace NUMINAMATH_CALUDE_combined_work_time_is_14_minutes_l1280_128043

/-- Represents the time taken to complete a job when working together, given individual work rates -/
def combined_work_time (george_rate : ℚ) (abe_rate : ℚ) (carla_rate : ℚ) : ℚ :=
  1 / (george_rate + abe_rate + carla_rate)

/-- Theorem stating that given the individual work rates, the combined work time is 14 minutes -/
theorem combined_work_time_is_14_minutes :
  combined_work_time (1/70) (1/30) (1/42) = 14 := by
  sorry

#eval combined_work_time (1/70) (1/30) (1/42)

end NUMINAMATH_CALUDE_combined_work_time_is_14_minutes_l1280_128043


namespace NUMINAMATH_CALUDE_lena_collage_glue_drops_l1280_128042

/-- The number of closest friends Lena has -/
def num_friends : ℕ := 7

/-- The number of clippings per friend -/
def clippings_per_friend : ℕ := 3

/-- The number of glue drops needed per clipping -/
def glue_drops_per_clipping : ℕ := 6

/-- The total number of glue drops needed for Lena's collage clippings -/
def total_glue_drops : ℕ := num_friends * clippings_per_friend * glue_drops_per_clipping

theorem lena_collage_glue_drops : total_glue_drops = 126 := by
  sorry

end NUMINAMATH_CALUDE_lena_collage_glue_drops_l1280_128042


namespace NUMINAMATH_CALUDE_equation_solution_l1280_128097

theorem equation_solution :
  ∃ x : ℝ, (5 + 3.4 * x = 2.8 * x - 35) ∧ (x = -200 / 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1280_128097


namespace NUMINAMATH_CALUDE_snail_problem_l1280_128039

/-- The number of snails originally in Centerville -/
def original_snails : ℕ := 11760

/-- The number of snails removed from Centerville -/
def removed_snails : ℕ := 3482

/-- The number of snails remaining in Centerville -/
def remaining_snails : ℕ := original_snails - removed_snails

theorem snail_problem : remaining_snails = 8278 := by
  sorry

end NUMINAMATH_CALUDE_snail_problem_l1280_128039


namespace NUMINAMATH_CALUDE_brown_hat_fraction_l1280_128025

theorem brown_hat_fraction (H : ℝ) (H_pos : H > 0) : ∃ B : ℝ,
  B > 0 ∧ B < 1 ∧
  (1/5 * B * H) / (1/3 * H) = 0.15 ∧
  B = 1/4 := by
sorry

end NUMINAMATH_CALUDE_brown_hat_fraction_l1280_128025


namespace NUMINAMATH_CALUDE_card_count_proof_l1280_128037

/-- The ratio of Xiao Ming's counting speed to Xiao Hua's -/
def speed_ratio : ℚ := 6 / 4

/-- The number of cards Xiao Hua counted before forgetting -/
def forgot_count : ℕ := 48

/-- The number of cards Xiao Hua counted after starting over -/
def final_count : ℕ := 112

/-- The number of cards left in the box after Xiao Hua's final count -/
def remaining_cards : ℕ := 1

/-- The original number of cards in the box -/
def original_cards : ℕ := 353

theorem card_count_proof :
  (speed_ratio * forgot_count).num.toNat + final_count + remaining_cards = original_cards :=
sorry

end NUMINAMATH_CALUDE_card_count_proof_l1280_128037


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l1280_128040

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_pos : 0 < b ∧ b < a

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of an ellipse -/
structure Foci (a b : ℝ) where
  left : Point
  right : Point
  h_ellipse : Ellipse a b

/-- Represents a triangle formed by three points -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Defines an equilateral triangle -/
def is_equilateral (t : Triangle) : Prop := sorry

/-- Defines a line perpendicular to the x-axis passing through a point -/
def perpendicular_to_x_axis (p : Point) (A B : Point) : Prop := sorry

/-- Defines points on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse a b) : Prop := sorry

/-- Defines the eccentricity of an ellipse -/
def eccentricity (e : Ellipse a b) : ℝ := sorry

/-- Main theorem -/
theorem ellipse_eccentricity_theorem 
  (a b : ℝ) 
  (e : Ellipse a b) 
  (f : Foci a b) 
  (A B : Point) 
  (t : Triangle) :
  perpendicular_to_x_axis f.right A B →
  on_ellipse A e →
  on_ellipse B e →
  t = Triangle.mk A B f.left →
  is_equilateral t →
  eccentricity e = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_theorem_l1280_128040


namespace NUMINAMATH_CALUDE_candy_bar_savings_l1280_128014

/-- Calculates the number of items saved given weekly receipt, consumption rate, and time period. -/
def items_saved (weekly_receipt : ℕ) (consumption_rate : ℕ) (weeks : ℕ) : ℕ :=
  weekly_receipt * weeks - (weeks / consumption_rate)

/-- Proves that under the given conditions, 28 items are saved after 16 weeks. -/
theorem candy_bar_savings : items_saved 2 4 16 = 28 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_savings_l1280_128014


namespace NUMINAMATH_CALUDE_area_of_ABCD_l1280_128061

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The composed rectangle ABCD -/
def ABCD : Rectangle := { width := 10, height := 15 }

/-- One of the smaller identical rectangles -/
def SmallRect : Rectangle := { width := 5, height := 10 }

theorem area_of_ABCD : ABCD.area = 150 := by sorry

end NUMINAMATH_CALUDE_area_of_ABCD_l1280_128061


namespace NUMINAMATH_CALUDE_reporters_covering_local_politics_l1280_128010

/-- The percentage of reporters who do not cover politics -/
def non_politics_reporters : ℝ := 85.71428571428572

/-- The percentage of reporters covering politics who do not cover local politics in country x -/
def non_local_politics_reporters : ℝ := 30

/-- The percentage of reporters covering local politics in country x -/
def local_politics_reporters : ℝ := 10

theorem reporters_covering_local_politics :
  local_politics_reporters = 
    (100 - non_politics_reporters) * (100 - non_local_politics_reporters) / 100 := by
  sorry

end NUMINAMATH_CALUDE_reporters_covering_local_politics_l1280_128010


namespace NUMINAMATH_CALUDE_work_earnings_equality_l1280_128095

/-- Proves that t = 5 given the conditions of the work problem --/
theorem work_earnings_equality (t : ℝ) : 
  (t - 4 > 0) →  -- My working hours are positive
  (t - 2 > 0) →  -- Sarah's working hours are positive
  (t - 4) * (3*t - 7) = (t - 2) * (t + 1) → 
  t = 5 := by
  sorry

end NUMINAMATH_CALUDE_work_earnings_equality_l1280_128095


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_unique_term_2011_l1280_128080

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmeticSequence (n : ℕ) : ℤ := 1 + 3 * (n - 1)

/-- Theorem stating that the 671st term of the sequence is 2011 -/
theorem arithmetic_sequence_2011 : arithmeticSequence 671 = 2011 := by sorry

/-- Theorem proving that 671 is the unique natural number n for which a_n = 2011 -/
theorem unique_term_2011 : ∀ n : ℕ, arithmeticSequence n = 2011 ↔ n = 671 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_unique_term_2011_l1280_128080


namespace NUMINAMATH_CALUDE_smallest_angle_of_quadrilateral_l1280_128004

theorem smallest_angle_of_quadrilateral (a b c d : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 →  -- All angles are positive
  a + b + c + d = 360 →           -- Sum of angles in a quadrilateral
  b = 4/3 * a →                   -- Ratio condition
  c = 5/3 * a →                   -- Ratio condition
  d = 2 * a →                     -- Ratio condition
  a = 60 ∧ a ≤ b ∧ a ≤ c ∧ a ≤ d  -- a is the smallest angle and equals 60°
  := by sorry

end NUMINAMATH_CALUDE_smallest_angle_of_quadrilateral_l1280_128004


namespace NUMINAMATH_CALUDE_expression_evaluation_l1280_128064

theorem expression_evaluation :
  let x : ℝ := 4
  let y : ℝ := -1/2
  2 * x^2 * y - (5 * x * y^2 + 2 * (x^2 * y - 3 * x * y^2 + 1)) = -1 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1280_128064


namespace NUMINAMATH_CALUDE_statue_weight_calculation_l1280_128011

/-- The weight of a statue after a series of cuts -/
def final_statue_weight (original_weight : ℝ) : ℝ :=
  let after_first_cut := original_weight * (1 - 0.3)
  let after_second_cut := after_first_cut * (1 - 0.2)
  let after_third_cut := after_second_cut * (1 - 0.25)
  after_third_cut

/-- Theorem stating the final weight of the statue -/
theorem statue_weight_calculation :
  final_statue_weight 250 = 105 := by
  sorry

end NUMINAMATH_CALUDE_statue_weight_calculation_l1280_128011


namespace NUMINAMATH_CALUDE_expression_value_l1280_128079

theorem expression_value : 
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 5000 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1280_128079


namespace NUMINAMATH_CALUDE_not_divisible_by_81_l1280_128069

theorem not_divisible_by_81 (n : ℤ) : ¬(81 ∣ (n^3 - 9*n + 27)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_81_l1280_128069


namespace NUMINAMATH_CALUDE_vessel_width_calculation_l1280_128059

/-- Proves that the width of a rectangular vessel's base is 5 cm when a cube of edge 5 cm is
    immersed, causing a 2.5 cm rise in water level, given that the vessel's base length is 10 cm. -/
theorem vessel_width_calculation (cube_edge : ℝ) (vessel_length : ℝ) (water_rise : ℝ) :
  cube_edge = 5 →
  vessel_length = 10 →
  water_rise = 2.5 →
  ∃ (vessel_width : ℝ),
    vessel_width = 5 ∧
    cube_edge ^ 3 = vessel_length * vessel_width * water_rise :=
by sorry

end NUMINAMATH_CALUDE_vessel_width_calculation_l1280_128059


namespace NUMINAMATH_CALUDE_house_construction_delay_l1280_128099

/-- Represents the construction of a house -/
structure HouseConstruction where
  totalDays : ℕ
  initialMen : ℕ
  additionalMen : ℕ
  daysBeforeAddition : ℕ

/-- Calculates the total man-days of work for the house construction -/
def totalManDays (h : HouseConstruction) : ℕ :=
  h.initialMen * h.totalDays

/-- Calculates the days behind schedule without additional men -/
def daysBehindSchedule (h : HouseConstruction) : ℕ :=
  let totalWork := h.initialMen * h.daysBeforeAddition + (h.initialMen + h.additionalMen) * (h.totalDays - h.daysBeforeAddition)
  totalWork / h.initialMen - h.totalDays

/-- Theorem stating that the construction would be 80 days behind schedule without additional men -/
theorem house_construction_delay (h : HouseConstruction) 
  (h_total_days : h.totalDays = 100)
  (h_initial_men : h.initialMen = 100)
  (h_additional_men : h.additionalMen = 100)
  (h_days_before_addition : h.daysBeforeAddition = 20) :
  daysBehindSchedule h = 80 := by
  sorry

#eval daysBehindSchedule { totalDays := 100, initialMen := 100, additionalMen := 100, daysBeforeAddition := 20 }

end NUMINAMATH_CALUDE_house_construction_delay_l1280_128099


namespace NUMINAMATH_CALUDE_table_runner_coverage_l1280_128071

theorem table_runner_coverage (runners : Nat) 
  (area_first_three : ℝ) (area_last_two : ℝ) (table_area : ℝ) 
  (coverage_percentage : ℝ) (two_layer_area : ℝ) (one_layer_area : ℝ) :
  runners = 5 →
  area_first_three = 324 →
  area_last_two = 216 →
  table_area = 320 →
  coverage_percentage = 0.75 →
  two_layer_area = 36 →
  one_layer_area = 48 →
  ∃ (three_layer_area : ℝ),
    three_layer_area = 156 ∧
    coverage_percentage * table_area = one_layer_area + two_layer_area + three_layer_area :=
by sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l1280_128071


namespace NUMINAMATH_CALUDE_air_conditioning_price_calculation_air_conditioning_price_proof_l1280_128020

/-- Calculates the final price of an air-conditioning unit after a discount and subsequent increase -/
theorem air_conditioning_price_calculation (initial_price : ℚ) 
  (discount_rate : ℚ) (increase_rate : ℚ) : ℚ :=
  let discounted_price := initial_price * (1 - discount_rate)
  let final_price := discounted_price * (1 + increase_rate)
  final_price

/-- Proves that the final price of the air-conditioning unit is approximately $442.18 -/
theorem air_conditioning_price_proof : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.005 ∧ 
  |air_conditioning_price_calculation 470 (16/100) (12/100) - 442.18| < ε :=
sorry

end NUMINAMATH_CALUDE_air_conditioning_price_calculation_air_conditioning_price_proof_l1280_128020


namespace NUMINAMATH_CALUDE_probability_five_or_joker_l1280_128041

/-- A deck of cards with jokers -/
structure DeckWithJokers where
  standardCards : ℕ
  jokers : ℕ
  totalCards : ℕ
  total_is_sum : totalCards = standardCards + jokers

/-- The probability of drawing a specific card or a joker -/
def drawProbability (d : DeckWithJokers) (specificCards : ℕ) : ℚ :=
  (specificCards + d.jokers : ℚ) / d.totalCards

/-- The deck described in the problem -/
def problemDeck : DeckWithJokers where
  standardCards := 52
  jokers := 2
  totalCards := 54
  total_is_sum := by rfl

theorem probability_five_or_joker :
  drawProbability problemDeck 4 = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_five_or_joker_l1280_128041


namespace NUMINAMATH_CALUDE_forty_knocks_to_knicks_l1280_128029

-- Define the units
def Knick : Type := ℚ
def Knack : Type := ℚ
def Knock : Type := ℚ

-- Define the conversion rates
def knicks_to_knacks : ℚ := 3 / 8
def knacks_to_knocks : ℚ := 5 / 4

-- Theorem statement
theorem forty_knocks_to_knicks :
  (40 : ℚ) * knacks_to_knocks⁻¹ * knicks_to_knacks⁻¹ = 128 / 3 := by
  sorry

end NUMINAMATH_CALUDE_forty_knocks_to_knicks_l1280_128029


namespace NUMINAMATH_CALUDE_pet_store_puppies_l1280_128009

theorem pet_store_puppies (initial_birds initial_puppies initial_cats initial_spiders : ℕ)
  (sold_birds adopted_puppies loose_spiders : ℕ) (final_total : ℕ) :
  initial_birds = 12 →
  initial_cats = 5 →
  initial_spiders = 15 →
  sold_birds = initial_birds / 2 →
  adopted_puppies = 3 →
  loose_spiders = 7 →
  final_total = 25 →
  final_total = initial_birds - sold_birds + initial_cats + 
                (initial_spiders - loose_spiders) + (initial_puppies - adopted_puppies) →
  initial_puppies = 9 :=
by sorry

end NUMINAMATH_CALUDE_pet_store_puppies_l1280_128009


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_p_for_q_necessary_not_sufficient_not_p_for_not_q_l1280_128016

-- Define the conditions
def p (x : ℝ) : Prop := -x^2 + 7*x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - 4*m^2 ≤ 0

-- Theorem 1
theorem necessary_not_sufficient_p_for_q (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) ∧ (∃ x, p x ∧ ¬q x m) →
  m ≥ 7/2 :=
sorry

-- Theorem 2
theorem necessary_not_sufficient_not_p_for_not_q (m : ℝ) :
  (m > 0) →
  (∀ x, ¬p x → ¬q x m) ∧ (∃ x, ¬q x m ∧ p x) →
  1 ≤ m ∧ m ≤ 7/2 :=
sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_p_for_q_necessary_not_sufficient_not_p_for_not_q_l1280_128016


namespace NUMINAMATH_CALUDE_least_five_digit_square_cube_l1280_128091

theorem least_five_digit_square_cube : ∃ n : ℕ,
  (n ≥ 10000 ∧ n ≤ 99999) ∧
  (∃ a : ℕ, n = a^2) ∧
  (∃ b : ℕ, n = b^3) ∧
  (∀ m : ℕ, m ≥ 10000 ∧ m ≤ 99999 ∧ (∃ x : ℕ, m = x^2) ∧ (∃ y : ℕ, m = y^3) → m ≥ n) ∧
  n = 15625 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_square_cube_l1280_128091


namespace NUMINAMATH_CALUDE_circle_area_l1280_128026

/-- The area of the circle defined by the equation 3x^2 + 3y^2 - 12x + 9y + 27 = 0 is equal to 61π/4 -/
theorem circle_area (x y : ℝ) : 
  (3 * x^2 + 3 * y^2 - 12 * x + 9 * y + 27 = 0) → 
  (∃ (center : ℝ × ℝ) (radius : ℝ), 
    ((x - center.1)^2 + (y - center.2)^2 = radius^2) ∧ 
    (π * radius^2 = 61 * π / 4)) :=
by sorry

end NUMINAMATH_CALUDE_circle_area_l1280_128026


namespace NUMINAMATH_CALUDE_g_of_3_l1280_128081

def g (x : ℝ) : ℝ := -3 * x^4 + 4 * x^3 - 7 * x^2 + 5 * x - 2

theorem g_of_3 : g 3 = -185 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l1280_128081


namespace NUMINAMATH_CALUDE_total_cookies_l1280_128050

/-- Given 272 bags of cookies with 45 cookies in each bag, 
    prove that the total number of cookies is 12240 -/
theorem total_cookies (bags : ℕ) (cookies_per_bag : ℕ) 
  (h1 : bags = 272) (h2 : cookies_per_bag = 45) : 
  bags * cookies_per_bag = 12240 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_l1280_128050


namespace NUMINAMATH_CALUDE_certain_number_theorem_l1280_128036

theorem certain_number_theorem (a x : ℕ) (h1 : a = 105) (h2 : a^3 = x * 25 * 45 * 49) : x = 21 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_theorem_l1280_128036


namespace NUMINAMATH_CALUDE_tangent_line_length_l1280_128024

/-- The length of a tangent line from a point to a circle --/
theorem tangent_line_length 
  (l : ℝ → ℝ → Prop) 
  (C : ℝ → ℝ → Prop) 
  (a : ℝ) :
  (∀ x y, l x y ↔ x + a * y - 1 = 0) →
  (∀ x y, C x y ↔ x^2 + y^2 - 4*x - 2*y + 1 = 0) →
  (∀ x y, l x y → C x y → x = 2 ∧ y = 1) →
  l (-4) a →
  ∃ B : ℝ × ℝ, C B.1 B.2 ∧ 
    (B.1 + 4)^2 + (B.2 + 1)^2 = 36 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_length_l1280_128024


namespace NUMINAMATH_CALUDE_triangle_less_than_answer_l1280_128098

theorem triangle_less_than_answer (triangle : ℝ) (answer : ℝ) 
  (h : 8.5 + triangle = 5.6 + answer) : triangle < answer := by
  sorry

end NUMINAMATH_CALUDE_triangle_less_than_answer_l1280_128098


namespace NUMINAMATH_CALUDE_function_inequality_solution_set_l1280_128067

open Set
open Function

theorem function_inequality_solution_set
  (f : ℝ → ℝ)
  (h_domain : ∀ x, x > 0 → DifferentiableAt ℝ f x)
  (h_ineq : ∀ x, x > 0 → x * deriv f x > f x)
  (h_f2 : f 2 = 0) :
  {x : ℝ | x > 0 ∧ f x < 0} = Ioo 0 2 := by
sorry

end NUMINAMATH_CALUDE_function_inequality_solution_set_l1280_128067


namespace NUMINAMATH_CALUDE_sin_period_l1280_128046

/-- The period of y = sin(4x + π) is π/2 -/
theorem sin_period (x : ℝ) : 
  (∀ y, y = Real.sin (4 * x + π)) → 
  (∃ p, p > 0 ∧ ∀ x, Real.sin (4 * x + π) = Real.sin (4 * (x + p) + π) ∧ p = π / 2) :=
by sorry

end NUMINAMATH_CALUDE_sin_period_l1280_128046


namespace NUMINAMATH_CALUDE_min_value_x_plus_2y_l1280_128073

theorem min_value_x_plus_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y + 2*x*y = 8) :
  ∀ z, z = x + 2*y → z ≥ 4 ∧ ∃ x₀ y₀, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ + 2*x₀*y₀ = 8 ∧ x₀ + 2*y₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_2y_l1280_128073


namespace NUMINAMATH_CALUDE_equation_has_two_solutions_l1280_128008

-- Define the equation
def equation (x : ℝ) : Prop := |x - 2| = |x - 4| + |x - 6|

-- Define the set of solutions
def solution_set : Set ℝ := {x : ℝ | equation x}

-- Theorem statement
theorem equation_has_two_solutions : 
  ∃ (a b : ℝ), a ≠ b ∧ solution_set = {a, b} :=
sorry

end NUMINAMATH_CALUDE_equation_has_two_solutions_l1280_128008


namespace NUMINAMATH_CALUDE_max_value_of_f_l1280_128074

open Real

noncomputable def f (θ : ℝ) : ℝ := tan (θ / 2) * (1 - sin θ)

theorem max_value_of_f :
  ∃ (θ_max : ℝ), 
    -π/2 < θ_max ∧ θ_max < π/2 ∧
    θ_max = 2 * arctan ((-2 + Real.sqrt 7) / 3) ∧
    ∀ (θ : ℝ), -π/2 < θ ∧ θ < π/2 → f θ ≤ f θ_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1280_128074


namespace NUMINAMATH_CALUDE_wood_amount_correct_l1280_128065

/-- The amount of wood (in cubic meters) that two workers need to saw and chop in one day -/
def wood_amount : ℚ := 40 / 13

/-- The amount of wood (in cubic meters) that two workers can saw in one day -/
def saw_capacity : ℚ := 5

/-- The amount of wood (in cubic meters) that two workers can chop in one day -/
def chop_capacity : ℚ := 8

/-- Theorem stating that the wood_amount is the correct amount of wood that two workers 
    need to saw in order to have enough time to chop it for the remainder of the day -/
theorem wood_amount_correct : 
  wood_amount / saw_capacity + wood_amount / chop_capacity = 1 := by
  sorry


end NUMINAMATH_CALUDE_wood_amount_correct_l1280_128065


namespace NUMINAMATH_CALUDE_fraction_simplification_l1280_128077

theorem fraction_simplification (m : ℝ) (hm : m ≠ 0 ∧ m ≠ 1) : 
  (m - 1) / m / ((m - 1) / (m^2)) = m := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1280_128077


namespace NUMINAMATH_CALUDE_profit_percent_calculation_l1280_128087

/-- 
Theorem: Given an article where selling at half of a certain price results in a 20% loss,
the profit percent when selling at the full price is 60%.
-/
theorem profit_percent_calculation (cost_price selling_price : ℝ) : 
  (selling_price / 2 = cost_price * 0.8) →  -- Half price results in 20% loss
  (selling_price - cost_price) / cost_price = 0.6  -- Profit percent is 60%
  := by sorry

end NUMINAMATH_CALUDE_profit_percent_calculation_l1280_128087
