import Mathlib

namespace NUMINAMATH_CALUDE_milk_distribution_l2082_208256

def milk_problem (total_milk myeongseok_milk minjae_milk : Real) (mingu_extra : Real) : Prop :=
  let mingu_milk := myeongseok_milk + mingu_extra
  let friends_total := myeongseok_milk + mingu_milk + minjae_milk
  let remaining_milk := total_milk - friends_total
  (total_milk = 1) ∧ 
  (myeongseok_milk = 0.1) ∧ 
  (mingu_extra = 0.2) ∧ 
  (minjae_milk = 0.3) ∧ 
  (remaining_milk = 0.3)

theorem milk_distribution : 
  ∃ (total_milk myeongseok_milk minjae_milk mingu_extra : Real),
    milk_problem total_milk myeongseok_milk minjae_milk mingu_extra :=
by
  sorry

end NUMINAMATH_CALUDE_milk_distribution_l2082_208256


namespace NUMINAMATH_CALUDE_probability_white_ball_l2082_208216

/-- The probability of drawing a white ball from a bag with black and white balls -/
theorem probability_white_ball (black_balls white_balls : ℕ) : 
  black_balls = 6 → white_balls = 5 → 
  (white_balls : ℚ) / (black_balls + white_balls : ℚ) = 5 / 11 :=
by
  sorry

#check probability_white_ball

end NUMINAMATH_CALUDE_probability_white_ball_l2082_208216


namespace NUMINAMATH_CALUDE_discount_difference_is_978_75_l2082_208212

/-- The initial invoice amount -/
def initial_amount : ℝ := 15000

/-- The single discount rate -/
def single_discount_rate : ℝ := 0.5

/-- The successive discount rates -/
def successive_discount_rates : List ℝ := [0.3, 0.15, 0.05]

/-- Calculate the amount after applying a single discount -/
def amount_after_single_discount (amount : ℝ) (rate : ℝ) : ℝ :=
  amount * (1 - rate)

/-- Calculate the amount after applying successive discounts -/
def amount_after_successive_discounts (amount : ℝ) (rates : List ℝ) : ℝ :=
  rates.foldl (fun acc rate => acc * (1 - rate)) amount

/-- The difference between single discount and successive discounts -/
def discount_difference : ℝ :=
  amount_after_successive_discounts initial_amount successive_discount_rates -
  amount_after_single_discount initial_amount single_discount_rate

theorem discount_difference_is_978_75 :
  discount_difference = 978.75 := by sorry

end NUMINAMATH_CALUDE_discount_difference_is_978_75_l2082_208212


namespace NUMINAMATH_CALUDE_cubic_inequality_implies_value_range_l2082_208244

theorem cubic_inequality_implies_value_range (y : ℝ) : 
  y^3 - 6*y^2 + 11*y - 6 < 0 → 
  24 < y^3 + 6*y^2 + 11*y + 6 ∧ y^3 + 6*y^2 + 11*y + 6 < 120 := by
sorry

end NUMINAMATH_CALUDE_cubic_inequality_implies_value_range_l2082_208244


namespace NUMINAMATH_CALUDE_initial_stamp_ratio_l2082_208232

theorem initial_stamp_ratio (p q : ℕ) : 
  (p - 8 : ℚ) / (q + 8 : ℚ) = 6 / 5 →
  p - 8 = q + 8 →
  (p : ℚ) / q = 6 / 5 := by
sorry

end NUMINAMATH_CALUDE_initial_stamp_ratio_l2082_208232


namespace NUMINAMATH_CALUDE_single_transmission_prob_triple_transmission_better_for_zero_l2082_208218

/-- Represents a binary communication channel with error probabilities α and β -/
structure BinaryChannel where
  α : ℝ
  β : ℝ
  α_pos : 0 < α
  α_lt_one : α < 1
  β_pos : 0 < β
  β_lt_one : β < 1

/-- Probability of receiving 1,0,1 when sending 1,0,1 in single transmission -/
def singleTransmissionProb (c : BinaryChannel) : ℝ :=
  (1 - c.α) * (1 - c.β)^2

/-- Probability of decoding 0 when sending 0 in single transmission -/
def singleTransmission0Prob (c : BinaryChannel) : ℝ :=
  1 - c.α

/-- Probability of decoding 0 when sending 0 in triple transmission -/
def tripleTransmission0Prob (c : BinaryChannel) : ℝ :=
  (1 - c.α)^3 + 3 * c.α * (1 - c.α)^2

theorem single_transmission_prob (c : BinaryChannel) :
  singleTransmissionProb c = (1 - c.α) * (1 - c.β)^2 := by sorry

theorem triple_transmission_better_for_zero (c : BinaryChannel) (h : c.α < 0.5) :
  singleTransmission0Prob c < tripleTransmission0Prob c := by sorry

end NUMINAMATH_CALUDE_single_transmission_prob_triple_transmission_better_for_zero_l2082_208218


namespace NUMINAMATH_CALUDE_problem_solution_l2082_208205

def f (a : ℝ) (x : ℝ) : ℝ := |2*x - 1| + |x + a|

theorem problem_solution :
  (∀ x : ℝ, f 1 x ≥ 3 ↔ x ≥ 1 ∨ x ≤ -1) ∧
  (∃ x : ℝ, f a x ≤ |a - 1| ↔ a ≤ 1/4) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l2082_208205


namespace NUMINAMATH_CALUDE_simple_interest_principal_l2082_208225

/-- Simple interest calculation -/
theorem simple_interest_principal
  (rate : ℝ) (interest : ℝ) (time : ℝ)
  (h_rate : rate = 15)
  (h_interest : interest = 120)
  (h_time : time = 2) :
  (interest * 100) / (rate * time) = 400 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_principal_l2082_208225


namespace NUMINAMATH_CALUDE_even_sum_condition_l2082_208263

theorem even_sum_condition (m n : ℤ) : 
  (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → (∃ p : ℤ, m + n = 2 * p) ∧
  ¬(∀ q : ℤ, m + n = 2 * q → ∃ r s : ℤ, m = 2 * r ∧ n = 2 * s) :=
by sorry

end NUMINAMATH_CALUDE_even_sum_condition_l2082_208263


namespace NUMINAMATH_CALUDE_product_comparison_l2082_208266

theorem product_comparison (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1.1 * a) * (1.13 * b) * (0.8 * c) < a * b * c :=
by sorry

end NUMINAMATH_CALUDE_product_comparison_l2082_208266


namespace NUMINAMATH_CALUDE_min_sum_p_q_l2082_208226

theorem min_sum_p_q (p q : ℕ) : 
  p > 1 → q > 1 → 17 * (p + 1) = 28 * (q + 1) → 
  ∀ (p' q' : ℕ), p' > 1 → q' > 1 → 17 * (p' + 1) = 28 * (q' + 1) → 
  p + q ≤ p' + q' → p + q = 135 := by
sorry

end NUMINAMATH_CALUDE_min_sum_p_q_l2082_208226


namespace NUMINAMATH_CALUDE_dancing_preference_theorem_l2082_208230

structure DancingPreference where
  like : Rat
  neutral : Rat
  dislike : Rat
  likeSayLike : Rat
  likeSayDislike : Rat
  dislikeSayLike : Rat
  dislikeSayDislike : Rat
  neutralSayLike : Rat
  neutralSayDislike : Rat

/-- The fraction of students who say they dislike dancing but actually like it -/
def fractionLikeSayDislike (pref : DancingPreference) : Rat :=
  (pref.like * pref.likeSayDislike) /
  (pref.like * pref.likeSayDislike + pref.dislike * pref.dislikeSayDislike + pref.neutral * pref.neutralSayDislike)

theorem dancing_preference_theorem (pref : DancingPreference) 
  (h1 : pref.like = 1/2)
  (h2 : pref.neutral = 3/10)
  (h3 : pref.dislike = 1/5)
  (h4 : pref.likeSayLike = 7/10)
  (h5 : pref.likeSayDislike = 3/10)
  (h6 : pref.dislikeSayLike = 1/5)
  (h7 : pref.dislikeSayDislike = 4/5)
  (h8 : pref.neutralSayLike = 2/5)
  (h9 : pref.neutralSayDislike = 3/5)
  : fractionLikeSayDislike pref = 15/49 := by
  sorry

end NUMINAMATH_CALUDE_dancing_preference_theorem_l2082_208230


namespace NUMINAMATH_CALUDE_product_cost_price_l2082_208217

theorem product_cost_price (original_price : ℝ) (cost_price : ℝ) : 
  (0.8 * original_price - cost_price = 120) →
  (0.6 * original_price - cost_price = -20) →
  cost_price = 440 := by
  sorry

end NUMINAMATH_CALUDE_product_cost_price_l2082_208217


namespace NUMINAMATH_CALUDE_factorization_1_l2082_208251

theorem factorization_1 (a b x y : ℝ) : a * (x - y) + b * (y - x) = (a - b) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_1_l2082_208251


namespace NUMINAMATH_CALUDE_beats_played_example_l2082_208220

/-- Given a person who plays music at a certain rate for a specific duration each day over multiple days, calculate the total number of beats played. -/
def totalBeatsPlayed (beatsPerMinute : ℕ) (hoursPerDay : ℕ) (numberOfDays : ℕ) : ℕ :=
  beatsPerMinute * (hoursPerDay * 60) * numberOfDays

/-- Theorem stating that playing 200 beats per minute for 2 hours a day for 3 days results in 72,000 beats total. -/
theorem beats_played_example : totalBeatsPlayed 200 2 3 = 72000 := by
  sorry

end NUMINAMATH_CALUDE_beats_played_example_l2082_208220


namespace NUMINAMATH_CALUDE_min_value_expression_l2082_208213

theorem min_value_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  4 ≤ a^2 + 2 * Real.sqrt (a * b) + Real.rpow (a^2 * b * c) (1/3) ∧
  ∃ a' b' c' : ℝ, 0 ≤ a' ∧ 0 ≤ b' ∧ 0 ≤ c' ∧ a' + b' + c' = 3 ∧
    a'^2 + 2 * Real.sqrt (a' * b') + Real.rpow (a'^2 * b' * c') (1/3) = 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2082_208213


namespace NUMINAMATH_CALUDE_annie_future_age_l2082_208265

theorem annie_future_age (anna_current_age : ℕ) (annie_current_age : ℕ) : 
  anna_current_age = 13 →
  annie_current_age = 3 * anna_current_age →
  (3 * anna_current_age + (annie_current_age - anna_current_age) = 65) :=
by
  sorry


end NUMINAMATH_CALUDE_annie_future_age_l2082_208265


namespace NUMINAMATH_CALUDE_unique_n_satisfying_equation_l2082_208284

theorem unique_n_satisfying_equation : ∃! (n : ℕ), 
  n + Int.floor (Real.sqrt n) + Int.floor (Real.sqrt (Real.sqrt n)) = 2017 :=
by sorry

end NUMINAMATH_CALUDE_unique_n_satisfying_equation_l2082_208284


namespace NUMINAMATH_CALUDE_lagrange_interpolation_polynomial_l2082_208281

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 5

theorem lagrange_interpolation_polynomial :
  P (-1) = -11 ∧ P 1 = -3 ∧ P 2 = 1 ∧ P 3 = 13 :=
by sorry

end NUMINAMATH_CALUDE_lagrange_interpolation_polynomial_l2082_208281


namespace NUMINAMATH_CALUDE_sum_x_y_z_l2082_208283

/-- Given that:
    - 0.5% of x equals 0.65 rupees
    - 1.25% of y equals 1.04 rupees
    - 2.5% of z equals 75% of x
    Prove that the sum of x, y, and z is 4113.2 rupees -/
theorem sum_x_y_z (x y z : ℝ) 
  (hx : 0.005 * x = 0.65)
  (hy : 0.0125 * y = 1.04)
  (hz : 0.025 * z = 0.75 * x) :
  x + y + z = 4113.2 := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_z_l2082_208283


namespace NUMINAMATH_CALUDE_harlys_dogs_l2082_208255

theorem harlys_dogs (x : ℝ) : 
  (0.6 * x + 5 = 53) → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_harlys_dogs_l2082_208255


namespace NUMINAMATH_CALUDE_triangle_side_length_l2082_208235

theorem triangle_side_length (a b : ℝ) (A B : Real) :
  a = 10 →
  B = Real.pi / 3 →
  A = Real.pi / 4 →
  b = 10 * (Real.sin (Real.pi / 3) / Real.sin (Real.pi / 4)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l2082_208235


namespace NUMINAMATH_CALUDE_oblong_perimeter_l2082_208299

theorem oblong_perimeter :
  ∀ l w : ℕ,
    l > w →
    l * 3 = w * 4 →
    l * w = 4624 →
    2 * l + 2 * w = 182 := by
  sorry

end NUMINAMATH_CALUDE_oblong_perimeter_l2082_208299


namespace NUMINAMATH_CALUDE_johns_bill_total_l2082_208234

/-- Calculates the total amount due on a bill after applying late charges and annual interest. -/
def totalAmountDue (originalBill : ℝ) (lateChargeRate : ℝ) (numLateCharges : ℕ) (annualInterestRate : ℝ) : ℝ :=
  let afterLateCharges := originalBill * (1 + lateChargeRate) ^ numLateCharges
  afterLateCharges * (1 + annualInterestRate)

/-- Proves that the total amount due on John's bill is $557.13 after one year. -/
theorem johns_bill_total : 
  let originalBill : ℝ := 500
  let lateChargeRate : ℝ := 0.02
  let numLateCharges : ℕ := 3
  let annualInterestRate : ℝ := 0.05
  totalAmountDue originalBill lateChargeRate numLateCharges annualInterestRate = 557.13 := by
  sorry


end NUMINAMATH_CALUDE_johns_bill_total_l2082_208234


namespace NUMINAMATH_CALUDE_johns_remaining_money_l2082_208207

/-- Calculates the amount of money John has left after walking his dog and spending money on books and his sister. -/
theorem johns_remaining_money (total_days : Nat) (sundays : Nat) (daily_pay : Nat) (book_cost : Nat) (sister_gift : Nat) :
  total_days = 30 →
  sundays = 4 →
  daily_pay = 10 →
  book_cost = 50 →
  sister_gift = 50 →
  (total_days - sundays) * daily_pay - (book_cost + sister_gift) = 160 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_l2082_208207


namespace NUMINAMATH_CALUDE_solution_and_rationality_l2082_208203

theorem solution_and_rationality 
  (x y : ℝ) 
  (h : Real.sqrt (8 * x - y^2) + |y^2 - 16| = 0) : 
  (x = 2 ∧ (y = 4 ∨ y = -4)) ∧ 
  ((y = 4 → ∃ (q : ℚ), Real.sqrt (y + 12) = ↑q) ∧ 
   (y = -4 → ∀ (q : ℚ), Real.sqrt (y + 12) ≠ ↑q)) := by
  sorry

end NUMINAMATH_CALUDE_solution_and_rationality_l2082_208203


namespace NUMINAMATH_CALUDE_shooter_probability_l2082_208200

theorem shooter_probability (p : ℝ) (n k : ℕ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  let prob_hit := p
  let num_shots := n
  let num_hits := k
  Nat.choose num_shots num_hits * prob_hit ^ num_hits * (1 - prob_hit) ^ (num_shots - num_hits) =
  Nat.choose 5 4 * (0.8 : ℝ) ^ 4 * (0.2 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_shooter_probability_l2082_208200


namespace NUMINAMATH_CALUDE_base7_subtraction_l2082_208273

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a decimal number to base 7 --/
def decimalToBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem base7_subtraction :
  let a := [4, 3, 2, 1]  -- 1234 in base 7
  let b := [2, 5, 6]     -- 652 in base 7
  let result := [2, 5, 2] -- 252 in base 7
  decimalToBase7 (base7ToDecimal a - base7ToDecimal b) = result := by
  sorry

end NUMINAMATH_CALUDE_base7_subtraction_l2082_208273


namespace NUMINAMATH_CALUDE_min_value_expression_l2082_208290

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : b ≠ 0) :
  ((a + 2*b)^2 + (b - 2*c)^2 + (c - 2*a)^2) / b^2 ≥ 25/3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2082_208290


namespace NUMINAMATH_CALUDE_some_zens_not_cens_l2082_208215

-- Define the sets
variable (U : Type) -- Universe set
variable (Zen : Set U) -- Set of Zens
variable (Ben : Set U) -- Set of Bens
variable (Cen : Set U) -- Set of Cens

-- Define the hypotheses
variable (h1 : Zen ⊆ Ben) -- All Zens are Bens
variable (h2 : ∃ x, x ∈ Ben ∧ x ∉ Cen) -- Some Bens are not Cens

-- Theorem to prove
theorem some_zens_not_cens : ∃ x, x ∈ Zen ∧ x ∉ Cen :=
sorry

end NUMINAMATH_CALUDE_some_zens_not_cens_l2082_208215


namespace NUMINAMATH_CALUDE_infinitely_many_n_with_large_prime_divisor_l2082_208229

theorem infinitely_many_n_with_large_prime_divisor :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ n ∈ S, ∃ p : ℕ, Nat.Prime p ∧ p > 2*n + Real.sqrt (2*n) ∧ p ∣ (n^2 + 1) :=
sorry

end NUMINAMATH_CALUDE_infinitely_many_n_with_large_prime_divisor_l2082_208229


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l2082_208257

theorem shaded_area_calculation (area_ABCD area_overlap : ℝ) 
  (h1 : area_ABCD = 196)
  (h2 : area_overlap = 1)
  (h3 : ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b^2 = 4*a^2 ∧ a + b = Real.sqrt area_ABCD - Real.sqrt area_overlap) :
  ∃ (shaded_area : ℝ), shaded_area = 72 ∧ 
    shaded_area = area_ABCD - (((Real.sqrt area_ABCD - Real.sqrt area_overlap)/3)^2 + 4*((Real.sqrt area_ABCD - Real.sqrt area_overlap)/3)^2 - area_overlap) :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l2082_208257


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l2082_208289

theorem cubic_sum_theorem (x y z a b c : ℝ)
  (h1 : x + y + z = a)
  (h2 : x^2 + y^2 + z^2 = b^2)
  (h3 : x⁻¹ + y⁻¹ + z⁻¹ = c⁻¹)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0) :
  x^3 + y^3 + z^3 = a^3 + (3/2) * (a^2 - b^2) * (c - a) := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l2082_208289


namespace NUMINAMATH_CALUDE_quadrilateral_diagonals_theorem_l2082_208209

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : Bool

def diagonals_bisect (q : Quadrilateral) : Prop :=
  let d1 := q.vertices 2 - q.vertices 0
  let d2 := q.vertices 3 - q.vertices 1
  (d1.1 / 2 = d2.1 / 2) ∧ (d1.2 / 2 = d2.2 / 2)

def is_parallelogram (q : Quadrilateral) : Prop :=
  (q.vertices 1 - q.vertices 0 = q.vertices 3 - q.vertices 2) ∧
  (q.vertices 2 - q.vertices 1 = q.vertices 0 - q.vertices 3)

def diagonals_equal (q : Quadrilateral) : Prop :=
  let d1 := q.vertices 2 - q.vertices 0
  let d2 := q.vertices 3 - q.vertices 1
  d1.1 * d1.1 + d1.2 * d1.2 = d2.1 * d2.1 + d2.2 * d2.2

def diagonals_perpendicular (q : Quadrilateral) : Prop :=
  let d1 := q.vertices 2 - q.vertices 0
  let d2 := q.vertices 3 - q.vertices 1
  d1.1 * d2.1 + d1.2 * d2.2 = 0

theorem quadrilateral_diagonals_theorem :
  (∀ q : Quadrilateral, diagonals_bisect q → is_parallelogram q) ∧
  (∃ q : Quadrilateral, diagonals_equal q ∧ ¬is_parallelogram q) ∧
  (∃ q : Quadrilateral, diagonals_perpendicular q ∧ ¬is_parallelogram q) ∧
  (∃ q : Quadrilateral, diagonals_equal q ∧ diagonals_perpendicular q ∧ ¬is_parallelogram q) :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_diagonals_theorem_l2082_208209


namespace NUMINAMATH_CALUDE_ing_catches_bo_l2082_208248

/-- The distance Bo jumps after n jumps -/
def bo_distance (n : ℕ) : ℕ := 6 * n

/-- The distance Ing jumps after n jumps -/
def ing_distance (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of jumps needed for Ing to catch Bo -/
def catch_up_jumps : ℕ := 11

theorem ing_catches_bo : 
  bo_distance catch_up_jumps = ing_distance catch_up_jumps :=
sorry

end NUMINAMATH_CALUDE_ing_catches_bo_l2082_208248


namespace NUMINAMATH_CALUDE_ratio_equality_l2082_208297

theorem ratio_equality (x : ℚ) : (1 : ℚ) / 3 = (5 : ℚ) / (3 * x) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l2082_208297


namespace NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l2082_208245

/-- Represents a right triangle with sides a, b, and c --/
structure RightTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_right : a^2 + b^2 = c^2

/-- Represents the areas of right isosceles triangles constructed on the sides of a right triangle --/
structure IsoscelesTriangleAreas (t : RightTriangle) where
  A : ℝ
  B : ℝ
  C : ℝ
  area_def_A : A = (1/2) * t.a^2
  area_def_B : B = (1/2) * t.b^2
  area_def_C : C = (1/2) * t.c^2

/-- Theorem: For a 5-12-13 right triangle with right isosceles triangles constructed on each side,
    the sum of the areas of the isosceles triangles on the two shorter sides
    equals the area of the isosceles triangle on the hypotenuse --/
theorem isosceles_triangle_areas_sum (t : RightTriangle)
  (h : t.a = 5 ∧ t.b = 12 ∧ t.c = 13)
  (areas : IsoscelesTriangleAreas t) :
  areas.A + areas.B = areas.C := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_areas_sum_l2082_208245


namespace NUMINAMATH_CALUDE_max_gcd_abb_aba_l2082_208231

def abb (a b : ℕ) : ℕ := 100 * a + 11 * b

def aba (a b : ℕ) : ℕ := 101 * a + 10 * b

theorem max_gcd_abb_aba : 
  ∀ a b : ℕ, a ≠ b → a < 10 → b < 10 → 
  (∀ c d : ℕ, c ≠ d → c < 10 → d < 10 → 
    Nat.gcd (abb a b) (aba a b) ≥ Nat.gcd (abb c d) (aba c d)) → 
  Nat.gcd (abb a b) (aba a b) = 18 :=
by sorry

end NUMINAMATH_CALUDE_max_gcd_abb_aba_l2082_208231


namespace NUMINAMATH_CALUDE_multiples_of_15_between_17_and_152_l2082_208279

theorem multiples_of_15_between_17_and_152 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ n > 17 ∧ n < 152) (Finset.range 152)).card = 9 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_15_between_17_and_152_l2082_208279


namespace NUMINAMATH_CALUDE_wheel_configuration_theorem_l2082_208293

/-- Represents a configuration of wheels with spokes -/
structure WheelConfiguration where
  total_spokes : Nat
  max_spokes_per_wheel : Nat

/-- Checks if a given number of wheels is possible for the configuration -/
def isPossible (config : WheelConfiguration) (num_wheels : Nat) : Prop :=
  num_wheels * config.max_spokes_per_wheel ≥ config.total_spokes

theorem wheel_configuration_theorem (config : WheelConfiguration) 
  (h1 : config.total_spokes = 7)
  (h2 : config.max_spokes_per_wheel = 3) :
  isPossible config 3 ∧ ¬isPossible config 2 := by
  sorry

#check wheel_configuration_theorem

end NUMINAMATH_CALUDE_wheel_configuration_theorem_l2082_208293


namespace NUMINAMATH_CALUDE_probability_non_red_face_l2082_208246

theorem probability_non_red_face (total_faces : ℕ) (red_faces : ℕ) (yellow_faces : ℕ) (blue_faces : ℕ) (green_faces : ℕ)
  (h1 : total_faces = 10)
  (h2 : red_faces = 5)
  (h3 : yellow_faces = 3)
  (h4 : blue_faces = 1)
  (h5 : green_faces = 1)
  (h6 : total_faces = red_faces + yellow_faces + blue_faces + green_faces) :
  (yellow_faces + blue_faces + green_faces : ℚ) / total_faces = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_red_face_l2082_208246


namespace NUMINAMATH_CALUDE_solve_for_x_l2082_208296

theorem solve_for_x (x y : ℝ) (h1 : x + 3 * y = 33) (h2 : y = 10) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l2082_208296


namespace NUMINAMATH_CALUDE_double_price_increase_l2082_208214

theorem double_price_increase (P : ℝ) (h : P > 0) : 
  P * (1 + 0.1) * (1 + 0.1) = P * (1 + 0.21) := by
sorry

end NUMINAMATH_CALUDE_double_price_increase_l2082_208214


namespace NUMINAMATH_CALUDE_reciprocal_equality_l2082_208241

theorem reciprocal_equality (a b : ℝ) : 
  (1 / a = -8) → (1 / (-b) = 8) → (a = b) := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_equality_l2082_208241


namespace NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l2082_208202

/-- Represents a hyperbola with equation x^2/5 - y^2/b^2 = 1 -/
structure Hyperbola where
  b : ℝ
  eq : ∀ x y : ℝ, x^2/5 - y^2/b^2 = 1

/-- The distance from the focus to the asymptote of the hyperbola -/
def focus_to_asymptote_distance (h : Hyperbola) : ℝ := 2

/-- The length of the conjugate axis of the hyperbola -/
def conjugate_axis_length (h : Hyperbola) : ℝ := 2 * h.b

theorem hyperbola_conjugate_axis_length (h : Hyperbola) :
  focus_to_asymptote_distance h = 2 →
  conjugate_axis_length h = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_conjugate_axis_length_l2082_208202


namespace NUMINAMATH_CALUDE_evaluate_expression_l2082_208208

theorem evaluate_expression : -(((16 / 4) * 6 - 50) + 5^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2082_208208


namespace NUMINAMATH_CALUDE_polynomial_product_sum_l2082_208285

theorem polynomial_product_sum (b₁ b₂ b₃ c₁ c₂ c₃ : ℝ) : 
  (∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + b₁*x + c₁) * (x^2 + b₂*x + c₂) * (x^2 + b₃*x + c₃)) →
  b₁*c₁ + b₂*c₂ + b₃*c₃ = -1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_sum_l2082_208285


namespace NUMINAMATH_CALUDE_end_behavior_of_g_l2082_208236

noncomputable def g (x : ℝ) : ℝ := -3 * x^4 + 5 * x^3 - 4

theorem end_behavior_of_g :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x < M) :=
sorry

end NUMINAMATH_CALUDE_end_behavior_of_g_l2082_208236


namespace NUMINAMATH_CALUDE_adjacent_pairs_difference_l2082_208278

/-- Given a circular arrangement of symbols, this theorem proves that the difference
    between the number of adjacent pairs of one symbol and the number of adjacent pairs
    of another symbol equals the difference in the total count of these symbols. -/
theorem adjacent_pairs_difference (p q a b : ℕ) : 
  (p + q > 0) →  -- Ensure the circle is not empty
  (a ≤ p) →      -- Number of X pairs cannot exceed total X's
  (b ≤ q) →      -- Number of 0 pairs cannot exceed total 0's
  (a = 0 → p ≤ 1) →  -- If no X pairs, at most one X
  (b = 0 → q ≤ 1) →  -- If no 0 pairs, at most one 0
  a - b = p - q :=
by sorry

end NUMINAMATH_CALUDE_adjacent_pairs_difference_l2082_208278


namespace NUMINAMATH_CALUDE_P_consecutive_coprime_l2082_208269

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define P(n) as given in the problem
def P : ℕ → ℕ
  | 0 => 0  -- Undefined in the original problem, added for completeness
  | 1 => 0  -- Undefined in the original problem, added for completeness
  | (n + 2) => 
    if n % 2 = 0 then
      (fib ((n / 2) + 1) + fib ((n / 2) - 1)) ^ 2
    else
      fib (n + 2) + fib ((n - 1) / 2)

-- State the theorem
theorem P_consecutive_coprime (k : ℕ) (h : k ≥ 3) : 
  Nat.gcd (P k) (P (k + 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_P_consecutive_coprime_l2082_208269


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2082_208211

def phone_cost : ℝ := 2
def service_plan_monthly_cost : ℝ := 7
def service_plan_duration : ℕ := 4
def insurance_fee : ℝ := 10
def first_phone_tax_rate : ℝ := 0.05
def second_phone_tax_rate : ℝ := 0.03
def service_plan_discount_rate : ℝ := 0.20
def num_phones : ℕ := 2

def total_cost : ℝ :=
  let phone_total := phone_cost * num_phones
  let service_plan_total := service_plan_monthly_cost * service_plan_duration * num_phones
  let service_plan_discount := service_plan_total * service_plan_discount_rate
  let discounted_service_plan := service_plan_total - service_plan_discount
  let tax_total := (first_phone_tax_rate * phone_cost) + (second_phone_tax_rate * phone_cost)
  phone_total + discounted_service_plan + tax_total + insurance_fee

theorem total_cost_is_correct : total_cost = 58.96 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2082_208211


namespace NUMINAMATH_CALUDE_regular_polygon_sides_l2082_208252

theorem regular_polygon_sides : ∃ n : ℕ, 
  n > 0 ∧ 
  (360 : ℝ) / n = n - 9 ∧
  n = 24 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_sides_l2082_208252


namespace NUMINAMATH_CALUDE_pgcd_and_divisibility_properties_l2082_208270

/-- For a ≥ 2 and m ≥ n ≥ 1, prove properties of PGCD and divisibility -/
theorem pgcd_and_divisibility_properties 
  (a m n : ℕ) 
  (ha : a ≥ 2) 
  (hmn : m ≥ n) 
  (hn : n ≥ 1) :
  (Nat.gcd (a^m - 1) (a^n - 1) = Nat.gcd (a^(m-n) - 1) (a^n - 1)) ∧
  (Nat.gcd (a^m - 1) (a^n - 1) = a^(Nat.gcd m n) - 1) ∧
  ((a^m - 1) ∣ (a^n - 1) ↔ m ∣ n) :=
by sorry

end NUMINAMATH_CALUDE_pgcd_and_divisibility_properties_l2082_208270


namespace NUMINAMATH_CALUDE_minimum_j_10_l2082_208298

/-- A function is stringent if it satisfies the given inequality for all positive integers x and y. -/
def Stringent (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y ≥ 2 * x.val ^ 2 - y.val

/-- The sum of j from 1 to 15 -/
def SumJ (j : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (λ i => j ⟨i + 1, Nat.succ_pos i⟩)

theorem minimum_j_10 :
  ∃ j : ℕ+ → ℤ,
    Stringent j ∧
    (∀ k : ℕ+ → ℤ, Stringent k → SumJ j ≤ SumJ k) ∧
    j ⟨10, by norm_num⟩ = 137 ∧
    (∀ k : ℕ+ → ℤ, Stringent k → (∀ i : ℕ+, SumJ j ≤ SumJ k) → j ⟨10, by norm_num⟩ ≤ k ⟨10, by norm_num⟩) :=
by sorry

end NUMINAMATH_CALUDE_minimum_j_10_l2082_208298


namespace NUMINAMATH_CALUDE_sum_equals_power_of_two_l2082_208274

theorem sum_equals_power_of_two : 29 + 12 + 23 = 2^6 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_power_of_two_l2082_208274


namespace NUMINAMATH_CALUDE_andrews_age_l2082_208247

theorem andrews_age (andrew_age grandfather_age : ℕ) : 
  grandfather_age = 16 * andrew_age →
  grandfather_age - andrew_age = 60 →
  andrew_age = 4 := by
sorry

end NUMINAMATH_CALUDE_andrews_age_l2082_208247


namespace NUMINAMATH_CALUDE_combinations_equal_200_l2082_208258

/-- The number of varieties of gift bags -/
def gift_bags : ℕ := 10

/-- The number of colors of tissue paper -/
def tissue_papers : ℕ := 4

/-- The number of types of tags -/
def tags : ℕ := 5

/-- The total number of possible combinations -/
def total_combinations : ℕ := gift_bags * tissue_papers * tags

/-- Theorem stating that the total number of combinations is 200 -/
theorem combinations_equal_200 : total_combinations = 200 := by
  sorry

end NUMINAMATH_CALUDE_combinations_equal_200_l2082_208258


namespace NUMINAMATH_CALUDE_factorization_theorem_l2082_208224

theorem factorization_theorem (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) = (a^2+b^2)*(b^2+c^2)*(c^2+a^2) :=
by sorry

end NUMINAMATH_CALUDE_factorization_theorem_l2082_208224


namespace NUMINAMATH_CALUDE_remainder_3249_div_82_l2082_208282

theorem remainder_3249_div_82 : 3249 % 82 = 51 := by sorry

end NUMINAMATH_CALUDE_remainder_3249_div_82_l2082_208282


namespace NUMINAMATH_CALUDE_original_price_with_loss_l2082_208253

/-- Proves that given an article sold for 300 with a 50% loss, the original price was 600 -/
theorem original_price_with_loss (selling_price : ℝ) (loss_percent : ℝ) : 
  selling_price = 300 → loss_percent = 50 → 
  ∃ original_price : ℝ, 
    original_price = 600 ∧ 
    selling_price = original_price * (1 - loss_percent / 100) := by
  sorry

end NUMINAMATH_CALUDE_original_price_with_loss_l2082_208253


namespace NUMINAMATH_CALUDE_sum_of_factors_48_l2082_208250

def sum_of_factors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

theorem sum_of_factors_48 : sum_of_factors 48 = 124 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_factors_48_l2082_208250


namespace NUMINAMATH_CALUDE_min_value_theorem_l2082_208288

theorem min_value_theorem (a m n : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) (hm : m > 0) (hn : n > 0) :
  let f := fun x => a^(x - 1) - 2
  let A := (1, -1)
  (m * A.1 - n * A.2 - 1 = 0) →
  (∀ x, f x = -1 → x = 1) →
  (∃ (min_val : ℝ), min_val = 3 + 2 * Real.sqrt 2 ∧
    ∀ (m' n' : ℝ), m' > 0 → n' > 0 → m' * A.1 - n' * A.2 - 1 = 0 →
      1 / m' + 2 / n' ≥ min_val) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2082_208288


namespace NUMINAMATH_CALUDE_train_length_l2082_208294

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) : 
  train_speed = 45 * 1000 / 3600 → 
  crossing_time = 30 → 
  bridge_length = 255 → 
  train_speed * crossing_time - bridge_length = 120 := by
  sorry

#check train_length

end NUMINAMATH_CALUDE_train_length_l2082_208294


namespace NUMINAMATH_CALUDE_regular_fish_price_l2082_208233

/-- The regular price of fish per pound, given a 50% discount and half-pound package price -/
theorem regular_fish_price (discount_percent : ℚ) (discounted_half_pound_price : ℚ) : 
  discount_percent = 50 →
  discounted_half_pound_price = 3 →
  12 = (2 * discounted_half_pound_price) / (1 - discount_percent / 100) :=
by sorry

end NUMINAMATH_CALUDE_regular_fish_price_l2082_208233


namespace NUMINAMATH_CALUDE_marble_remainder_l2082_208210

theorem marble_remainder (l j : ℕ) 
  (hl : l % 8 = 5) 
  (hj : j % 8 = 6) : 
  (l + j) % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l2082_208210


namespace NUMINAMATH_CALUDE_total_goats_is_320_l2082_208206

/-- The number of goats Washington has -/
def washington_goats : ℕ := 140

/-- The number of additional goats Paddington has compared to Washington -/
def additional_goats : ℕ := 40

/-- The total number of goats Paddington and Washington have together -/
def total_goats : ℕ := washington_goats + (washington_goats + additional_goats)

/-- Theorem stating the total number of goats is 320 -/
theorem total_goats_is_320 : total_goats = 320 := by
  sorry

end NUMINAMATH_CALUDE_total_goats_is_320_l2082_208206


namespace NUMINAMATH_CALUDE_peanuts_in_box_l2082_208242

/-- Given a box with an initial number of peanuts and a number of peanuts added,
    calculate the total number of peanuts in the box. -/
def total_peanuts (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem: If there are 4 peanuts in a box and 2 more are added,
    the total number of peanuts in the box is 6. -/
theorem peanuts_in_box : total_peanuts 4 2 = 6 := by sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l2082_208242


namespace NUMINAMATH_CALUDE_bus_driver_compensation_l2082_208237

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
theorem bus_driver_compensation
  (regular_rate : ℝ)
  (regular_hours : ℝ)
  (overtime_percentage : ℝ)
  (total_hours : ℝ)
  (h1 : regular_rate = 14)
  (h2 : regular_hours = 40)
  (h3 : overtime_percentage = 0.75)
  (h4 : total_hours = 57.88) :
  ∃ (total_compensation : ℝ), 
    abs (total_compensation - 998.06) < 0.01 ∧
    total_compensation = 
      regular_rate * regular_hours + 
      (regular_rate * (1 + overtime_percentage)) * (total_hours - regular_hours) :=
by sorry


end NUMINAMATH_CALUDE_bus_driver_compensation_l2082_208237


namespace NUMINAMATH_CALUDE_floor_equation_solution_l2082_208295

theorem floor_equation_solution (a b : ℕ+) :
  (⌊(a : ℝ)^2 / b⌋ + ⌊(b : ℝ)^2 / a⌋ = ⌊((a : ℝ)^2 + (b : ℝ)^2) / (a * b)⌋ + (a * b : ℝ)) ↔
  (∃ k : ℕ+, (a = k ∧ b = k^2 + 1) ∨ (a = k^2 + 1 ∧ b = k)) :=
by sorry

end NUMINAMATH_CALUDE_floor_equation_solution_l2082_208295


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_ratio_l2082_208228

/-- A geometric sequence with first term less than zero and increasing terms has a common ratio between 0 and 1. -/
theorem geometric_sequence_increasing_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (q : ℝ)      -- The common ratio
  (h1 : a 1 < 0)  -- First term is negative
  (h2 : ∀ n : ℕ, a n < a (n + 1))  -- Sequence is strictly increasing
  (h3 : ∀ n : ℕ, a (n + 1) = a n * q)  -- Definition of geometric sequence
  : 0 < q ∧ q < 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_ratio_l2082_208228


namespace NUMINAMATH_CALUDE_babysitting_earnings_l2082_208219

theorem babysitting_earnings (total : ℚ) 
  (h1 : total / 4 + total / 2 + 50 = total) : total = 200 := by
  sorry

end NUMINAMATH_CALUDE_babysitting_earnings_l2082_208219


namespace NUMINAMATH_CALUDE_sine_intersection_theorem_l2082_208291

theorem sine_intersection_theorem (a b : ℕ) (h : a ≠ b) :
  ∃ c : ℕ, c ≠ a ∧ c ≠ b ∧
  ∀ x : ℝ, Real.sin (a * x) = Real.sin (b * x) →
            Real.sin (c * x) = Real.sin (a * x) :=
by sorry

end NUMINAMATH_CALUDE_sine_intersection_theorem_l2082_208291


namespace NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2082_208272

theorem least_n_satisfying_inequality : 
  ∃ n : ℕ+, (∀ k : ℕ+, k < n → (1 : ℚ) / k - (1 : ℚ) / (k + 1) ≥ (1 : ℚ) / 15) ∧
             ((1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧
             n = 4 :=
by sorry

end NUMINAMATH_CALUDE_least_n_satisfying_inequality_l2082_208272


namespace NUMINAMATH_CALUDE_wrong_height_calculation_wrong_height_is_176_l2082_208201

/-- Given a class of boys with an incorrect average height and one boy's height recorded incorrectly,
    calculate the wrongly written height of that boy. -/
theorem wrong_height_calculation (n : ℕ) (initial_avg correct_avg actual_height : ℝ) : ℝ :=
  let wrong_height := actual_height + n * (initial_avg - correct_avg)
  wrong_height

/-- Prove that the wrongly written height of a boy is 176 cm given the specified conditions. -/
theorem wrong_height_is_176 :
  wrong_height_calculation 35 180 178 106 = 176 := by
  sorry

end NUMINAMATH_CALUDE_wrong_height_calculation_wrong_height_is_176_l2082_208201


namespace NUMINAMATH_CALUDE_multiples_6_10_not_5_8_empty_l2082_208275

theorem multiples_6_10_not_5_8_empty : 
  {n : ℤ | 1 ≤ n ∧ n ≤ 300 ∧ 6 ∣ n ∧ 10 ∣ n ∧ ¬(5 ∣ n) ∧ ¬(8 ∣ n)} = ∅ := by
  sorry

end NUMINAMATH_CALUDE_multiples_6_10_not_5_8_empty_l2082_208275


namespace NUMINAMATH_CALUDE_steve_exceeds_wayne_in_2004_l2082_208286

def money_at_year (initial : ℕ) (multiplier : ℚ) (year : ℕ) : ℚ :=
  initial * multiplier ^ year

def steve_money (year : ℕ) : ℚ := money_at_year 100 2 year

def wayne_money (year : ℕ) : ℚ := money_at_year 10000 (1/2) year

def first_year_steve_exceeds_wayne : ℕ := 2004

theorem steve_exceeds_wayne_in_2004 :
  (∀ y : ℕ, y < first_year_steve_exceeds_wayne → steve_money y ≤ wayne_money y) ∧
  steve_money first_year_steve_exceeds_wayne > wayne_money first_year_steve_exceeds_wayne :=
sorry

end NUMINAMATH_CALUDE_steve_exceeds_wayne_in_2004_l2082_208286


namespace NUMINAMATH_CALUDE_jeremy_stroll_time_l2082_208249

/-- Proves that Jeremy's strolling time is 10 hours given his distance and speed -/
theorem jeremy_stroll_time (distance : ℝ) (speed : ℝ) (h1 : distance = 20) (h2 : speed = 2) :
  distance / speed = 10 := by
  sorry

end NUMINAMATH_CALUDE_jeremy_stroll_time_l2082_208249


namespace NUMINAMATH_CALUDE_max_digits_product_5_4_l2082_208204

theorem max_digits_product_5_4 : 
  ∀ (a b : ℕ), 
    10000 ≤ a ∧ a ≤ 99999 →
    1000 ≤ b ∧ b ≤ 9999 →
    a * b < 1000000000 := by
  sorry

end NUMINAMATH_CALUDE_max_digits_product_5_4_l2082_208204


namespace NUMINAMATH_CALUDE_ellipse_equation_l2082_208243

/-- Given an ellipse with eccentricity √7/4 and distance 4 from one endpoint of the minor axis to the right focus, prove its standard equation is x²/16 + y²/9 = 1 -/
theorem ellipse_equation (e : ℝ) (d : ℝ) (x y : ℝ) :
  e = Real.sqrt 7 / 4 →
  d = 4 →
  x^2 / 16 + y^2 / 9 = 1 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2082_208243


namespace NUMINAMATH_CALUDE_described_method_is_analogical_thinking_l2082_208223

/-- A learning method in mathematics -/
structure LearningMethod where
  compare_objects : Bool
  find_similarities : Bool
  deduce_similar_properties : Bool

/-- Analogical thinking in mathematics -/
def analogical_thinking : LearningMethod :=
  { compare_objects := true,
    find_similarities := true,
    deduce_similar_properties := true }

/-- The described learning method -/
def described_method : LearningMethod :=
  { compare_objects := true,
    find_similarities := true,
    deduce_similar_properties := true }

/-- Theorem stating that the described learning method is equivalent to analogical thinking -/
theorem described_method_is_analogical_thinking : described_method = analogical_thinking :=
  sorry

end NUMINAMATH_CALUDE_described_method_is_analogical_thinking_l2082_208223


namespace NUMINAMATH_CALUDE_venus_meal_cost_calculation_l2082_208254

/-- The cost per meal at Venus Hall -/
def venus_meal_cost : ℝ := 35

/-- The room rental cost for Caesar's -/
def caesars_room_cost : ℝ := 800

/-- The meal cost for Caesar's -/
def caesars_meal_cost : ℝ := 30

/-- The room rental cost for Venus Hall -/
def venus_room_cost : ℝ := 500

/-- The number of guests at which the costs are equal -/
def num_guests : ℝ := 60

theorem venus_meal_cost_calculation :
  caesars_room_cost + caesars_meal_cost * num_guests =
  venus_room_cost + venus_meal_cost * num_guests :=
by sorry

end NUMINAMATH_CALUDE_venus_meal_cost_calculation_l2082_208254


namespace NUMINAMATH_CALUDE_prob_divisible_by_five_prob_divisible_by_five_is_one_l2082_208238

/-- A three-digit positive integer with a ones digit of 5 -/
def ThreeDigitEndingIn5 : Type := { n : ℕ // 100 ≤ n ∧ n < 1000 ∧ n % 10 = 5 }

/-- The probability that a number in ThreeDigitEndingIn5 is divisible by 5 -/
theorem prob_divisible_by_five (n : ThreeDigitEndingIn5) : ℚ :=
  1

/-- The probability that a number in ThreeDigitEndingIn5 is divisible by 5 is 1 -/
theorem prob_divisible_by_five_is_one : 
  ∀ n : ThreeDigitEndingIn5, prob_divisible_by_five n = 1 :=
sorry

end NUMINAMATH_CALUDE_prob_divisible_by_five_prob_divisible_by_five_is_one_l2082_208238


namespace NUMINAMATH_CALUDE_silly_bills_game_l2082_208221

theorem silly_bills_game (x : ℕ) : 
  x + (x + 11) + (x - 18) > 0 →  -- Ensure positive number of bills
  x + 2 * (x + 11) + 3 * (x - 18) = 100 →
  x = 22 := by
sorry

end NUMINAMATH_CALUDE_silly_bills_game_l2082_208221


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_read_l2082_208260

/-- Represents the 'crazy silly school' series -/
structure CrazySillySchool where
  total_books : Nat
  total_movies : Nat
  books_read : Nat
  movies_watched : Nat

/-- Theorem: In the 'crazy silly school' series, if all available books and movies are consumed,
    and the number of movies watched is 2 more than the number of books read,
    then the number of books read is 8. -/
theorem crazy_silly_school_books_read
  (css : CrazySillySchool)
  (h1 : css.total_books = 8)
  (h2 : css.total_movies = 10)
  (h3 : css.movies_watched = css.books_read + 2)
  (h4 : css.books_read = css.total_books)
  (h5 : css.movies_watched = css.total_movies) :
  css.books_read = 8 := by
  sorry

#check crazy_silly_school_books_read

end NUMINAMATH_CALUDE_crazy_silly_school_books_read_l2082_208260


namespace NUMINAMATH_CALUDE_adjacent_diff_at_least_five_l2082_208268

/-- Represents a cell in the 8x8 grid -/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the 8x8 grid filled with integers from 1 to 64 -/
def Grid := Cell → Fin 64

/-- Two cells are adjacent if they share a common edge -/
def adjacent (c1 c2 : Cell) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c1.col.val = c2.col.val + 1) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col = c2.col) ∨
  (c1.row.val = c2.row.val + 1 ∧ c1.col = c2.col)

/-- Main theorem: In any 8x8 grid filled with integers from 1 to 64,
    there exist two adjacent cells whose values differ by at least 5 -/
theorem adjacent_diff_at_least_five (g : Grid) : 
  ∃ c1 c2 : Cell, adjacent c1 c2 ∧ (g c1).val + 5 ≤ (g c2).val ∨ (g c2).val + 5 ≤ (g c1).val :=
sorry

end NUMINAMATH_CALUDE_adjacent_diff_at_least_five_l2082_208268


namespace NUMINAMATH_CALUDE_four_terms_after_substitution_l2082_208262

-- Define the expression with a variable for the asterisk
def expression (a : ℝ → ℝ) : ℝ → ℝ := λ x => (x^4 - 3)^2 + (x^3 + a x)^2

-- Define the proposed replacement for the asterisk
def replacement : ℝ → ℝ := λ x => x^3 + 3*x

-- The theorem to prove
theorem four_terms_after_substitution :
  ∃ (c₁ c₂ c₃ c₄ : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    (∀ x, expression replacement x = c₁ * x^n₁ + c₂ * x^n₂ + c₃ * x^n₃ + c₄ * x^n₄) ∧
    (n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₁ ≠ n₄ ∧ n₂ ≠ n₃ ∧ n₂ ≠ n₄ ∧ n₃ ≠ n₄) :=
by
  sorry

end NUMINAMATH_CALUDE_four_terms_after_substitution_l2082_208262


namespace NUMINAMATH_CALUDE_min_h_10_l2082_208261

/-- A function is stringent if f(x) + f(y) > 2y^2 for all positive integers x and y -/
def Stringent (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y > 2 * y.val ^ 2

/-- The sum of h from 1 to 15 -/
def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem min_h_10 (h : ℕ+ → ℤ) (stringent_h : Stringent h) 
    (min_sum : ∀ g : ℕ+ → ℤ, Stringent g → SumH g ≥ SumH h) : 
    h ⟨10, by norm_num⟩ ≥ 136 := by
  sorry

end NUMINAMATH_CALUDE_min_h_10_l2082_208261


namespace NUMINAMATH_CALUDE_judy_hits_percentage_l2082_208227

theorem judy_hits_percentage (total_hits : ℕ) (home_runs : ℕ) (triples : ℕ) (doubles : ℕ)
  (h1 : total_hits = 50)
  (h2 : home_runs = 2)
  (h3 : triples = 3)
  (h4 : doubles = 8)
  (h5 : total_hits ≥ home_runs + triples + doubles) :
  (((total_hits - (home_runs + triples + doubles)) : ℚ) / total_hits) * 100 = 74 := by
sorry

end NUMINAMATH_CALUDE_judy_hits_percentage_l2082_208227


namespace NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_for_prop_b_l2082_208292

-- Define propositions p and q
variable (p q : Prop)

-- Define Proposition A
def PropA : Prop := p → q

-- Define Proposition B
def PropB : Prop := p ↔ q

-- Theorem to prove
theorem prop_a_necessary_not_sufficient_for_prop_b :
  (PropA p q → PropB p q) ∧ ¬(PropB p q → PropA p q) :=
sorry

end NUMINAMATH_CALUDE_prop_a_necessary_not_sufficient_for_prop_b_l2082_208292


namespace NUMINAMATH_CALUDE_equal_real_roots_no_real_solutions_l2082_208264

-- Define the quadratic equation
def quadratic_equation (a b x : ℝ) : Prop := a * x^2 + b * x + (1/4 : ℝ) = 0

-- Part 1: Equal real roots condition
theorem equal_real_roots (a b : ℝ) (h : a ≠ 0) :
  a = 1 ∧ b = 1 → ∃! x : ℝ, quadratic_equation a b x :=
sorry

-- Part 2: No real solutions condition
theorem no_real_solutions (a b : ℝ) (h1 : a > 1) (h2 : 0 < b) (h3 : b < 1) :
  ¬∃ x : ℝ, quadratic_equation a b x :=
sorry

end NUMINAMATH_CALUDE_equal_real_roots_no_real_solutions_l2082_208264


namespace NUMINAMATH_CALUDE_valid_pairs_l2082_208239

def is_valid_pair (m n : ℕ) : Prop :=
  Nat.Prime m ∧ Nat.Prime n ∧ m < n ∧ n < 5 * m ∧ Nat.Prime (m + 3 * n)

theorem valid_pairs :
  ∀ m n : ℕ, is_valid_pair m n ↔ (m = 2 ∧ n = 3) ∨ (m = 2 ∧ n = 5) ∨ (m = 2 ∧ n = 7) :=
by sorry

end NUMINAMATH_CALUDE_valid_pairs_l2082_208239


namespace NUMINAMATH_CALUDE_system_solution_l2082_208280

theorem system_solution (x y : ℝ) : 
  x^2 + y^2 ≤ 2 ∧ 
  81 * x^4 - 18 * x^2 * y^2 + y^4 - 360 * x^2 - 40 * y^2 + 400 = 0 ↔ 
  ((x = -3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
   (x = -3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
   (x = 3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
   (x = 3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2082_208280


namespace NUMINAMATH_CALUDE_farm_animal_difference_l2082_208276

theorem farm_animal_difference (goats chickens ducks pigs : ℕ) : 
  goats = 66 →
  chickens = 2 * goats →
  ducks = (goats + chickens) / 2 →
  pigs = ducks / 3 →
  goats - pigs = 33 := by
sorry

end NUMINAMATH_CALUDE_farm_animal_difference_l2082_208276


namespace NUMINAMATH_CALUDE_p_less_q_less_r_l2082_208267

-- Define the logarithm function (base 2)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 2

-- State the theorem
theorem p_less_q_less_r (a b : ℝ) 
  (h1 : a > b) (h2 : b > 1) 
  (P : ℝ) (hP : P = lg a * lg b)
  (Q : ℝ) (hQ : Q = lg a + lg b)
  (R : ℝ) (hR : R = lg (a * b)) :
  P < Q ∧ Q < R := by
  sorry

end NUMINAMATH_CALUDE_p_less_q_less_r_l2082_208267


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2082_208277

theorem complex_magnitude_problem (s : ℝ) (w : ℂ) 
  (h1 : |s| < 3) 
  (h2 : w + 2 / w = s) : 
  Complex.abs w = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2082_208277


namespace NUMINAMATH_CALUDE_product_of_3_6_and_0_5_l2082_208287

theorem product_of_3_6_and_0_5 : 3.6 * 0.5 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_3_6_and_0_5_l2082_208287


namespace NUMINAMATH_CALUDE_fraction_subtraction_result_l2082_208259

theorem fraction_subtraction_result : (18 : ℚ) / 42 - 3 / 8 - 1 / 12 = -5 / 168 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_result_l2082_208259


namespace NUMINAMATH_CALUDE_cos_graph_transformation_l2082_208240

theorem cos_graph_transformation (x : ℝ) : 
  4 * Real.cos (2 * x) = 4 * Real.cos (2 * (x - π/8) + π/4) :=
by sorry

end NUMINAMATH_CALUDE_cos_graph_transformation_l2082_208240


namespace NUMINAMATH_CALUDE_julios_age_l2082_208222

/-- Proves that Julio's current age is 36 years old, given the conditions of the problem -/
theorem julios_age (james_age : ℕ) (future_years : ℕ) (julio_age : ℕ) : 
  james_age = 11 →
  future_years = 14 →
  julio_age + future_years = 2 * (james_age + future_years) →
  julio_age = 36 :=
by sorry

end NUMINAMATH_CALUDE_julios_age_l2082_208222


namespace NUMINAMATH_CALUDE_square_cutting_solution_l2082_208271

/-- Represents the cutting of an 8x8 square into smaller pieces -/
structure SquareCutting where
  /-- The number of 2x2 squares -/
  num_squares : ℕ
  /-- The number of 1x4 rectangles -/
  num_rectangles : ℕ
  /-- The total length of cuts -/
  total_cut_length : ℕ

/-- Theorem stating the solution to the square cutting problem -/
theorem square_cutting_solution :
  ∃ (cut : SquareCutting),
    cut.num_squares = 10 ∧
    cut.num_rectangles = 6 ∧
    cut.total_cut_length = 54 ∧
    cut.num_squares + cut.num_rectangles = 64 / 4 ∧
    8 * cut.num_squares + 10 * cut.num_rectangles = 32 + 2 * cut.total_cut_length :=
by sorry

end NUMINAMATH_CALUDE_square_cutting_solution_l2082_208271
