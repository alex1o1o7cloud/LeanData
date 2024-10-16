import Mathlib

namespace NUMINAMATH_CALUDE_inequality_solution_l3410_341045

def solution_set (m : ℝ) : Set ℝ :=
  if m = -3 then { x | x > 1 }
  else if -3 < m ∧ m < -1 then { x | x < m / (m + 3) ∨ x > 1 }
  else if m < -3 then { x | 1 < x ∧ x < m / (m + 3) }
  else ∅

theorem inequality_solution (m : ℝ) (h : m < -1) :
  { x : ℝ | (m + 3) * x^2 - (2 * m + 3) * x + m > 0 } = solution_set m :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l3410_341045


namespace NUMINAMATH_CALUDE_original_number_is_four_l3410_341082

def is_correct_number (x : ℕ) : Prop :=
  (x + 3) % 5 = 2 ∧ 
  ((x + 5) + 3) % 5 = 2

theorem original_number_is_four : 
  ∃ (x : ℕ), is_correct_number x ∧ x = 4 :=
sorry

end NUMINAMATH_CALUDE_original_number_is_four_l3410_341082


namespace NUMINAMATH_CALUDE_sequence_ratio_l3410_341006

/-- Given an arithmetic sequence and a geometric sequence with specific properties, 
    prove that (b-a)/d = 1/2 --/
theorem sequence_ratio (a b c d e : ℝ) : 
  ((-1 : ℝ) - a = a - b) ∧ (b - (-4 : ℝ) = a - b) ∧  -- arithmetic sequence condition
  (c = (-1 : ℝ) * d / c) ∧ (d = c * e / d) ∧ (e = d * (-4 : ℝ) / e) →  -- geometric sequence condition
  (b - a) / d = (1 : ℝ) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sequence_ratio_l3410_341006


namespace NUMINAMATH_CALUDE_max_pencils_is_13_l3410_341040

def john_money : ℚ := 10
def regular_price : ℚ := 0.75
def discount_price : ℚ := 0.65
def discount_threshold : ℕ := 10

def cost (n : ℕ) : ℚ :=
  if n ≤ discount_threshold then
    n * regular_price
  else
    discount_threshold * regular_price + (n - discount_threshold) * discount_price

def can_afford (n : ℕ) : Prop :=
  cost n ≤ john_money

theorem max_pencils_is_13 :
  ∀ n : ℕ, can_afford n → n ≤ 13 ∧
  ∃ m : ℕ, m = 13 ∧ can_afford m :=
by sorry

end NUMINAMATH_CALUDE_max_pencils_is_13_l3410_341040


namespace NUMINAMATH_CALUDE_multiplicand_difference_l3410_341059

theorem multiplicand_difference (a b : ℕ) : 
  a * b = 100100 → 
  a < b → 
  a % 10 = 2 → 
  b % 10 = 6 → 
  b - a = 564 := by
sorry

end NUMINAMATH_CALUDE_multiplicand_difference_l3410_341059


namespace NUMINAMATH_CALUDE_power_of_two_sum_l3410_341055

theorem power_of_two_sum (y : ℕ) : 8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_sum_l3410_341055


namespace NUMINAMATH_CALUDE_calculator_time_saved_l3410_341050

/-- The time saved by using a calculator for math homework -/
theorem calculator_time_saved 
  (time_with_calc : ℕ)      -- Time per problem with calculator
  (time_without_calc : ℕ)   -- Time per problem without calculator
  (num_problems : ℕ)        -- Number of problems in the assignment
  (h1 : time_with_calc = 2) -- It takes 2 minutes per problem with calculator
  (h2 : time_without_calc = 5) -- It takes 5 minutes per problem without calculator
  (h3 : num_problems = 20)  -- The assignment has 20 problems
  : (time_without_calc - time_with_calc) * num_problems = 60 := by
  sorry

end NUMINAMATH_CALUDE_calculator_time_saved_l3410_341050


namespace NUMINAMATH_CALUDE_angle_B_value_l3410_341076

theorem angle_B_value (A B C : Real) (h1 : 0 < A) (h2 : A < B) (h3 : B < C) 
  (h4 : A + B + C = π) 
  (h5 : (Real.sin A + Real.sin B + Real.sin C) / (Real.cos A + Real.cos B + Real.cos C) = Real.sqrt 3) : 
  B = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_value_l3410_341076


namespace NUMINAMATH_CALUDE_range_of_a_plus_3b_l3410_341000

theorem range_of_a_plus_3b (a b : ℝ) 
  (h1 : -1 ≤ a + b ∧ a + b ≤ 1) 
  (h2 : 1 ≤ a - 2*b ∧ a - 2*b ≤ 3) : 
  -11/3 ≤ a + 3*b ∧ a + 3*b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_plus_3b_l3410_341000


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l3410_341086

theorem right_triangle_leg_square (a c : ℝ) (h1 : c = a + 2) :
  ∃ b : ℝ, b^2 = 4*a + 4 ∧ a^2 + b^2 = c^2 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l3410_341086


namespace NUMINAMATH_CALUDE_exam_score_problem_l3410_341072

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_score = 4 →
  wrong_score = -1 →
  total_score = 120 →
  ∃ (correct_answers : ℕ) (wrong_answers : ℕ),
    correct_answers + wrong_answers = total_questions ∧
    correct_score * correct_answers + wrong_score * wrong_answers = total_score ∧
    correct_answers = 36 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_problem_l3410_341072


namespace NUMINAMATH_CALUDE_extreme_value_implies_a_l3410_341066

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 3*x - 9

theorem extreme_value_implies_a (a : ℝ) :
  (∃ (ε : ℝ), ∀ (x : ℝ), |x + 3| < ε → f a x ≤ f a (-3)) →
  a = 5 := by
  sorry

end NUMINAMATH_CALUDE_extreme_value_implies_a_l3410_341066


namespace NUMINAMATH_CALUDE_x₁_plus_x₂_pos_l3410_341037

noncomputable section

variables (a : ℝ) (x x₁ x₂ : ℝ)

def f (x : ℝ) : ℝ := Real.log (a * x + 1) - a * x - Real.log a

axiom a_pos : a > 0

axiom x_domain : x > -1/a

axiom x₁_domain : -1/a < x₁ ∧ x₁ < 0

axiom x₂_domain : x₂ > 0

axiom f_roots : f a x₁ = 0 ∧ f a x₂ = 0

theorem x₁_plus_x₂_pos : x₁ + x₂ > 0 := by sorry

end NUMINAMATH_CALUDE_x₁_plus_x₂_pos_l3410_341037


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3410_341070

theorem closest_integer_to_cube_root (x : ℝ) : 
  x = (7^3 + 9^3 + 10^3 : ℝ)^(1/3) → 
  ∃ (n : ℤ), n = 13 ∧ ∀ (m : ℤ), |x - n| ≤ |x - m| :=
by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l3410_341070


namespace NUMINAMATH_CALUDE_cubic_root_problem_l3410_341089

/-- A monic cubic polynomial -/
def MonicCubic (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem cubic_root_problem (r : ℝ) (f g : ℝ → ℝ) 
    (hf : ∃ a b c, f = MonicCubic a b c)
    (hg : ∃ a b c, g = MonicCubic a b c)
    (hf_roots : f (r + 2) = 0 ∧ f (r + 4) = 0)
    (hg_roots : g (r + 3) = 0 ∧ g (r + 5) = 0)
    (h_diff : ∀ x, f x - g x = 2*r + 1) :
  r = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_problem_l3410_341089


namespace NUMINAMATH_CALUDE_product_in_N_l3410_341053

def M : Set ℤ := {x | ∃ m : ℤ, x = 3 * m + 1}
def N : Set ℤ := {y | ∃ n : ℤ, y = 3 * n + 2}

theorem product_in_N (x y : ℤ) (hx : x ∈ M) (hy : y ∈ N) : x * y ∈ N := by
  sorry

end NUMINAMATH_CALUDE_product_in_N_l3410_341053


namespace NUMINAMATH_CALUDE_simplify_fraction_multiplication_l3410_341032

theorem simplify_fraction_multiplication :
  (180 : ℚ) / 1620 * 20 = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_multiplication_l3410_341032


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3410_341019

/-- A quadratic function with a positive leading coefficient and symmetry about x = 2 -/
def symmetric_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a > 0 ∧ (∀ x, f x = a * x^2 + b * x + c) ∧ (∀ x, f x = f (4 - x))

theorem quadratic_inequality (f : ℝ → ℝ) (a : ℝ) 
  (h1 : symmetric_quadratic f) 
  (h2 : f (2 - a^2) < f (1 + a - a^2)) : 
  a < 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3410_341019


namespace NUMINAMATH_CALUDE_arithmetic_subsequence_multiples_of_3_l3410_341043

/-- An arithmetic sequence with common difference d -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

/-- The subsequence of an arithmetic sequence with indices that are multiples of 3 -/
def SubsequenceMultiplesOf3 (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = a (3 * n)

theorem arithmetic_subsequence_multiples_of_3 (a : ℕ → ℝ) (b : ℕ → ℝ) (d : ℝ) :
  ArithmeticSequence a d → SubsequenceMultiplesOf3 a b →
  ArithmeticSequence b (3 * d) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_subsequence_multiples_of_3_l3410_341043


namespace NUMINAMATH_CALUDE_leftover_coins_value_l3410_341052

def quarters_per_roll : ℕ := 50
def dimes_per_roll : ℕ := 40
def michael_quarters : ℕ := 95
def michael_dimes : ℕ := 172
def anna_quarters : ℕ := 140
def anna_dimes : ℕ := 287
def quarter_value : ℚ := 0.25
def dime_value : ℚ := 0.10

theorem leftover_coins_value :
  let total_quarters := michael_quarters + anna_quarters
  let total_dimes := michael_dimes + anna_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  let leftover_value := (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value
  leftover_value = 10.65 := by sorry

end NUMINAMATH_CALUDE_leftover_coins_value_l3410_341052


namespace NUMINAMATH_CALUDE_cube_root_of_unity_in_finite_field_l3410_341087

theorem cube_root_of_unity_in_finite_field (p : ℕ) (hp : p.Prime) (hp3 : p > 3) :
  let F := ZMod p
  (∃ x : F, x^2 = -3) →
    (∃! (a b c : F), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^3 = 1 ∧ b^3 = 1 ∧ c^3 = 1) ∧
  (¬∃ x : F, x^2 = -3) →
    (∃! a : F, a^3 = 1) :=
sorry

end NUMINAMATH_CALUDE_cube_root_of_unity_in_finite_field_l3410_341087


namespace NUMINAMATH_CALUDE_f_is_even_l3410_341079

def F (f : ℝ → ℝ) (x : ℝ) : ℝ := (x^3 - 2*x) * f x

def OddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def NotIdenticallyZero (f : ℝ → ℝ) : Prop := ∃ x, f x ≠ 0

theorem f_is_even (f : ℝ → ℝ) (h1 : OddFunction (F f)) (h2 : NotIdenticallyZero f) :
  EvenFunction f := by sorry

end NUMINAMATH_CALUDE_f_is_even_l3410_341079


namespace NUMINAMATH_CALUDE_mixture_volume_is_four_liters_l3410_341081

/-- Represents the weight in grams of 1 liter of ghee for a specific brand. -/
structure GheeWeight where
  weight : ℝ
  weight_positive : weight > 0

/-- Represents the volume ratio between two brands of ghee. -/
structure MixingRatio where
  a : ℝ
  b : ℝ
  a_positive : a > 0
  b_positive : b > 0

/-- Calculates the total volume of a ghee mixture given the weights and mixing ratio. -/
def calculate_mixture_volume (weight_a weight_b : GheeWeight) (ratio : MixingRatio) (total_weight_kg : ℝ) : ℝ :=
  sorry

/-- Theorem stating that the mixture volume is 4 liters given the problem conditions. -/
theorem mixture_volume_is_four_liters 
  (weight_a : GheeWeight)
  (weight_b : GheeWeight)
  (ratio : MixingRatio)
  (total_weight_kg : ℝ)
  (ha : weight_a.weight = 900)
  (hb : weight_b.weight = 750)
  (hr : ratio.a / ratio.b = 3 / 2)
  (hw : total_weight_kg = 3.36) :
  calculate_mixture_volume weight_a weight_b ratio total_weight_kg = 4 :=
sorry

end NUMINAMATH_CALUDE_mixture_volume_is_four_liters_l3410_341081


namespace NUMINAMATH_CALUDE_juice_bar_problem_l3410_341049

theorem juice_bar_problem (total_spent : ℕ) (mango_juice_cost : ℕ) (other_juice_total : ℕ) (total_people : ℕ) :
  total_spent = 94 →
  mango_juice_cost = 5 →
  other_juice_total = 54 →
  total_people = 17 →
  ∃ (other_juice_cost : ℕ),
    other_juice_cost = 6 ∧
    other_juice_cost * (total_people - (total_spent - other_juice_total) / mango_juice_cost) = other_juice_total :=
by sorry

end NUMINAMATH_CALUDE_juice_bar_problem_l3410_341049


namespace NUMINAMATH_CALUDE_second_quadrant_trig_identity_l3410_341057

/-- For any angle α in the second quadrant, (sin α / cos α) * √(1 / sin²α - 1) = -1 -/
theorem second_quadrant_trig_identity (α : Real) (h : π / 2 < α ∧ α < π) :
  (Real.sin α / Real.cos α) * Real.sqrt (1 / Real.sin α ^ 2 - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_second_quadrant_trig_identity_l3410_341057


namespace NUMINAMATH_CALUDE_both_in_picture_probability_l3410_341044

/-- Alice's lap time in seconds -/
def alice_lap_time : ℕ := 120

/-- Bob's lap time in seconds -/
def bob_lap_time : ℕ := 75

/-- Bob's start delay in seconds -/
def bob_start_delay : ℕ := 15

/-- Duration of one-third of the track for Alice in seconds -/
def alice_third_track : ℕ := alice_lap_time / 3

/-- Duration of one-third of the track for Bob in seconds -/
def bob_third_track : ℕ := bob_lap_time / 3

/-- Least common multiple of Alice and Bob's lap times -/
def lcm_lap_times : ℕ := lcm alice_lap_time bob_lap_time

/-- Time window for taking the picture in seconds -/
def picture_window : ℕ := 60

/-- Probability of both Alice and Bob being in the picture -/
def probability_both_in_picture : ℚ := 11 / 1200

theorem both_in_picture_probability :
  probability_both_in_picture = 11 / 1200 := by
  sorry

end NUMINAMATH_CALUDE_both_in_picture_probability_l3410_341044


namespace NUMINAMATH_CALUDE_mari_buttons_l3410_341085

theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) : 
  sue_buttons = 6 →
  sue_buttons = kendra_buttons / 2 →
  mari_buttons = 4 + 5 * kendra_buttons →
  mari_buttons = 64 := by
sorry

end NUMINAMATH_CALUDE_mari_buttons_l3410_341085


namespace NUMINAMATH_CALUDE_total_faces_is_198_l3410_341048

/-- The total number of faces on all dice and geometrical shapes -/
def total_faces : ℕ := sorry

/-- Number of six-sided dice -/
def six_sided_dice : ℕ := 4

/-- Number of eight-sided dice -/
def eight_sided_dice : ℕ := 5

/-- Number of twelve-sided dice -/
def twelve_sided_dice : ℕ := 3

/-- Number of twenty-sided dice -/
def twenty_sided_dice : ℕ := 2

/-- Number of cubes -/
def cubes : ℕ := 1

/-- Number of tetrahedrons -/
def tetrahedrons : ℕ := 3

/-- Number of icosahedrons -/
def icosahedrons : ℕ := 2

/-- Theorem stating that the total number of faces is 198 -/
theorem total_faces_is_198 : total_faces = 198 := by sorry

end NUMINAMATH_CALUDE_total_faces_is_198_l3410_341048


namespace NUMINAMATH_CALUDE_van_speed_for_longer_time_l3410_341034

/-- Given a van that travels 450 km in 5 hours, this theorem proves the speed
    required to cover the same distance in 3/2 of the original time. -/
theorem van_speed_for_longer_time (distance : ℝ) (initial_time : ℝ) (time_factor : ℝ) :
  distance = 450 ∧ initial_time = 5 ∧ time_factor = 3/2 →
  distance / (initial_time * time_factor) = 60 := by
  sorry

end NUMINAMATH_CALUDE_van_speed_for_longer_time_l3410_341034


namespace NUMINAMATH_CALUDE_count_base7_with_456_l3410_341021

/-- Represents a positive integer in base 7 --/
def Base7Int : Type := ℕ+

/-- Checks if a Base7Int contains the digits 4, 5, or 6 --/
def containsDigit456 (n : Base7Int) : Prop := sorry

/-- The set of the smallest 2401 positive integers in base 7 --/
def smallestBase7Ints : Set Base7Int := {n | n.val ≤ 2401}

/-- The count of numbers in smallestBase7Ints that contain 4, 5, or 6 --/
def countWith456 : ℕ := sorry

theorem count_base7_with_456 : countWith456 = 2146 := by sorry

end NUMINAMATH_CALUDE_count_base7_with_456_l3410_341021


namespace NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l3410_341096

theorem smallest_x_for_perfect_cube : ∃ (x : ℕ+), 
  (∀ (y : ℕ+), 2520 * y.val = (M : ℕ)^3 → x ≤ y) ∧ 
  (∃ (M : ℕ), 2520 * x.val = M^3) ∧
  x.val = 3675 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_perfect_cube_l3410_341096


namespace NUMINAMATH_CALUDE_train_crossing_time_l3410_341071

/-- The time it takes for a train to cross a pole -/
theorem train_crossing_time (train_speed : ℝ) (train_length : ℝ) : 
  train_speed = 270 →
  train_length = 375.03 →
  (train_length / (train_speed * 1000 / 3600)) = 5.0004 := by
  sorry

end NUMINAMATH_CALUDE_train_crossing_time_l3410_341071


namespace NUMINAMATH_CALUDE_max_a_value_l3410_341061

theorem max_a_value (x a : ℤ) : 
  (∃ x : ℤ, x^2 + a*x = -30) → 
  (a > 0) → 
  a ≤ 31 :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l3410_341061


namespace NUMINAMATH_CALUDE_sin_alpha_value_l3410_341095

theorem sin_alpha_value (α : Real) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 1 / 5) : 
  Real.sin α = (6 * Real.sqrt 2 - 1) / 10 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_value_l3410_341095


namespace NUMINAMATH_CALUDE_marias_profit_is_75_l3410_341094

/-- Calculates Maria's profit from bread sales given the specified conditions. -/
def marias_profit (total_loaves : ℕ) (cost_per_loaf : ℚ) 
  (morning_price : ℚ) (afternoon_price_ratio : ℚ) (evening_price : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3 * morning_price
  let afternoon_loaves := total_loaves - total_loaves / 3
  let afternoon_sales := afternoon_loaves / 2 * (afternoon_price_ratio * morning_price)
  let evening_loaves := afternoon_loaves - afternoon_loaves / 2
  let evening_sales := evening_loaves * evening_price
  let total_revenue := morning_sales + afternoon_sales + evening_sales
  let total_cost := total_loaves * cost_per_loaf
  total_revenue - total_cost

/-- Theorem stating that Maria's profit is $75 given the specified conditions. -/
theorem marias_profit_is_75 : 
  marias_profit 60 1 3 (3/4) (3/2) = 75 := by
  sorry

end NUMINAMATH_CALUDE_marias_profit_is_75_l3410_341094


namespace NUMINAMATH_CALUDE_deepak_current_age_l3410_341069

/-- Proves Deepak's current age given the ratio of ages and Arun's future age -/
theorem deepak_current_age 
  (arun_age : ℕ) 
  (deepak_age : ℕ) 
  (h1 : arun_age + 5 = 25) 
  (h2 : arun_age * 3 = deepak_age * 2) : 
  deepak_age = 30 := by
sorry

end NUMINAMATH_CALUDE_deepak_current_age_l3410_341069


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3410_341099

theorem max_value_of_expression (x y z : Real) 
  (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  let A := (Real.sqrt (8 * x^4 + y) + Real.sqrt (8 * y^4 + z) + Real.sqrt (8 * z^4 + x) - 3) / (x + y + z)
  A ≤ 2 ∧ ∃ x y z, (0 < x ∧ x ≤ 1) ∧ (0 < y ∧ y ≤ 1) ∧ (0 < z ∧ z ≤ 1) ∧ A = 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3410_341099


namespace NUMINAMATH_CALUDE_base_7_to_10_conversion_l3410_341013

def to_base_10 (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

theorem base_7_to_10_conversion :
  to_base_10 [3, 1, 2, 5] 7 = 1823 := by
  sorry

end NUMINAMATH_CALUDE_base_7_to_10_conversion_l3410_341013


namespace NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l3410_341091

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi ∧ Real.sin x - Real.cos x = Real.sqrt 2 → x = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l3410_341091


namespace NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l3410_341039

theorem ellipse_foci_on_y_axis (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (2 - k) + y^2 / (2*k - 1) = 1 → 
    (∃ c : ℝ, c > 0 ∧ 
      ∀ p : ℝ × ℝ, 
        (p.1 = 0 → (p.2 = c ∨ p.2 = -c)) ∧ 
        (p.2 = c ∨ p.2 = -c → p.1 = 0))) → 
  1 < k ∧ k < 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_on_y_axis_l3410_341039


namespace NUMINAMATH_CALUDE_torn_sheets_count_l3410_341012

/-- Represents a book with numbered pages -/
structure Book where
  /-- The number of the last page in the book -/
  lastPage : ℕ

/-- Represents a range of torn out pages -/
structure TornPages where
  /-- The number of the first torn out page -/
  first : ℕ
  /-- The number of the last torn out page -/
  last : ℕ

/-- Check if a number consists of the same digits as another number in a different order -/
def sameDigitsDifferentOrder (a b : ℕ) : Prop :=
  sorry

/-- Calculate the number of sheets torn out given the first and last torn page numbers -/
def sheetsTornOut (torn : TornPages) : ℕ :=
  (torn.last - torn.first + 1) / 2

/-- The main theorem to be proved -/
theorem torn_sheets_count (book : Book) (torn : TornPages) :
  torn.first = 185 ∧
  sameDigitsDifferentOrder torn.first torn.last ∧
  Even torn.last ∧
  torn.last > torn.first →
  sheetsTornOut torn = 167 := by
  sorry

end NUMINAMATH_CALUDE_torn_sheets_count_l3410_341012


namespace NUMINAMATH_CALUDE_youngest_child_age_l3410_341063

/-- Given 5 children born at intervals of 3 years, if the sum of their ages is 70 years,
    then the age of the youngest child is 8 years. -/
theorem youngest_child_age (youngest_age : ℕ) : 
  (youngest_age + (youngest_age + 3) + (youngest_age + 6) + (youngest_age + 9) + (youngest_age + 12) = 70) →
  youngest_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l3410_341063


namespace NUMINAMATH_CALUDE_intersection_condition_l3410_341017

-- Define the quadratic function
def f (a x : ℝ) := a * x^2 - 4 * a * x - 2

-- Define the solution set of the inequality
def solution_set (a : ℝ) := {x : ℝ | f a x > 0}

-- Define the given set
def given_set := {x : ℝ | 3 < x ∧ x < 4}

-- Theorem statement
theorem intersection_condition (a : ℝ) : 
  (∃ x, x ∈ solution_set a ∧ x ∈ given_set) ↔ a < -2/3 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_l3410_341017


namespace NUMINAMATH_CALUDE_bayberry_sales_theorem_l3410_341078

/-- Represents the bayberry selling scenario -/
structure BayberrySales where
  initial_price : ℝ
  initial_volume : ℝ
  cost_price : ℝ
  volume_increase_rate : ℝ

/-- Calculates the daily revenue given a price decrease -/
def daily_revenue (s : BayberrySales) (price_decrease : ℝ) : ℝ :=
  (s.initial_price - price_decrease) * (s.initial_volume + s.volume_increase_rate * price_decrease)

/-- Calculates the daily profit given a selling price -/
def daily_profit (s : BayberrySales) (selling_price : ℝ) : ℝ :=
  (selling_price - s.cost_price) * (s.initial_volume + s.volume_increase_rate * (s.initial_price - selling_price))

/-- The main theorem about bayberry sales -/
theorem bayberry_sales_theorem (s : BayberrySales) 
  (h1 : s.initial_price = 20)
  (h2 : s.initial_volume = 100)
  (h3 : s.cost_price = 8)
  (h4 : s.volume_increase_rate = 20) :
  (∃ x y, x ≠ y ∧ daily_revenue s x = 3000 ∧ daily_revenue s y = 3000 ∧ x = 5 ∧ y = 10) ∧
  (∃ max_price max_profit, 
    (∀ p, daily_profit s p ≤ max_profit) ∧
    daily_profit s max_price = max_profit ∧
    max_price = 16.5 ∧ max_profit = 1445) := by
  sorry


end NUMINAMATH_CALUDE_bayberry_sales_theorem_l3410_341078


namespace NUMINAMATH_CALUDE_triangle_theorem_l3410_341062

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition -/
def condition (t : Triangle) : Prop :=
  t.a / (Real.sqrt 3 * Real.cos t.A) = t.c / Real.sin t.C

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) (h1 : condition t) (h2 : t.a = 6) :
  t.A = π / 3 ∧ 12 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 18 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3410_341062


namespace NUMINAMATH_CALUDE_min_value_problem_l3410_341046

theorem min_value_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_constraint : a + b + c = 12) (product_constraint : a * b * c = 27) :
  (a^2 + b^2) / (a + b) + (a^2 + c^2) / (a + c) + (b^2 + c^2) / (b + c) ≥ 12 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l3410_341046


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3410_341038

theorem quadratic_root_condition (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁^2 + k*x₁ + 4*k^2 - 3 = 0 ∧ 
   x₂^2 + k*x₂ + 4*k^2 - 3 = 0 ∧
   x₁ + x₂ = x₁ * x₂) → 
  k = 3/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l3410_341038


namespace NUMINAMATH_CALUDE_new_person_weight_l3410_341083

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 45 →
  ∃ (new_weight : ℝ),
    new_weight = initial_count * weight_increase + replaced_weight ∧
    new_weight = 65 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l3410_341083


namespace NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3410_341009

-- Define the polar equation
def polar_equation (ρ θ : ℝ) : Prop := ρ = 5 * Real.sin θ

-- Define the Cartesian equation of a circle
def circle_equation (x y : ℝ) (h k r : ℝ) : Prop := (x - h)^2 + (y - k)^2 = r^2

-- Theorem statement
theorem polar_to_cartesian_circle :
  ∀ x y : ℝ, (∃ ρ θ : ℝ, polar_equation ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  ∃ h k r : ℝ, circle_equation x y h k r := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_circle_l3410_341009


namespace NUMINAMATH_CALUDE_log_27_greater_than_0_53_l3410_341028

theorem log_27_greater_than_0_53 : Real.log 27 > 0.53 := by
  sorry

end NUMINAMATH_CALUDE_log_27_greater_than_0_53_l3410_341028


namespace NUMINAMATH_CALUDE_correct_definition_in_list_correct_definition_unique_l3410_341073

/-- Definition of Digital Earth -/
def DigitalEarth : Type := String

/-- The correct definition of Digital Earth -/
def correct_definition : DigitalEarth :=
  "a technical system that digitizes the entire Earth's information and manages it through computer networks"

/-- Possible definitions of Digital Earth -/
def possible_definitions : List DigitalEarth :=
  [ "representing the size of the Earth with numbers"
  , correct_definition
  , "using the data of the latitude and longitude grid to represent the location of geographical entities"
  , "using GPS data to represent the location of various geographical entities on Earth"
  ]

/-- Theorem stating that the correct definition is in the list of possible definitions -/
theorem correct_definition_in_list : correct_definition ∈ possible_definitions :=
  by sorry

/-- Theorem stating that the correct definition is unique in the list -/
theorem correct_definition_unique :
  ∀ d ∈ possible_definitions, d = correct_definition ↔ d = possible_definitions[1] :=
  by sorry

end NUMINAMATH_CALUDE_correct_definition_in_list_correct_definition_unique_l3410_341073


namespace NUMINAMATH_CALUDE_constant_volume_l3410_341001

/-- Represents a line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Represents the configuration of the tetrahedron with moving vertices -/
structure MovingTetrahedron where
  fixedEdge : Line3D
  movingVertex1 : Line3D
  movingVertex2 : Line3D
  initialTetrahedron : Tetrahedron

/-- Checks if three lines are parallel -/
def areLinesParallel (l1 l2 l3 : Line3D) : Prop := sorry

/-- Calculates the tetrahedron at a given time t -/
def tetrahedronAtTime (mt : MovingTetrahedron) (t : ℝ) : Tetrahedron := sorry

/-- Theorem stating that the volume remains constant -/
theorem constant_volume (mt : MovingTetrahedron) 
  (h : areLinesParallel mt.fixedEdge mt.movingVertex1 mt.movingVertex2) :
  ∀ t : ℝ, tetrahedronVolume (tetrahedronAtTime mt t) = tetrahedronVolume mt.initialTetrahedron :=
sorry

end NUMINAMATH_CALUDE_constant_volume_l3410_341001


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_arithmetic_sequence_has_671_terms_l3410_341002

/-- An arithmetic sequence starting at 2, with common difference 3, and last term 2014 -/
def ArithmeticSequence : ℕ → ℤ := fun n ↦ 2 + 3 * (n - 1)

theorem arithmetic_sequence_length :
  ∃ n : ℕ, n > 0 ∧ ArithmeticSequence n = 2014 ∧ ∀ m : ℕ, m > n → ArithmeticSequence m > 2014 :=
by
  sorry

theorem arithmetic_sequence_has_671_terms :
  ∃! n : ℕ, n > 0 ∧ ArithmeticSequence n = 2014 ∧ n = 671 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_arithmetic_sequence_has_671_terms_l3410_341002


namespace NUMINAMATH_CALUDE_derivative_of_f_at_1_l3410_341097

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem derivative_of_f_at_1 : 
  deriv f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_derivative_of_f_at_1_l3410_341097


namespace NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3410_341004

theorem square_difference_division (a b : ℕ) (h : a > b) :
  (a^2 - b^2) / (a - b) = a + b :=
sorry

theorem problem_solution : (112^2 - 97^2) / 15 = 209 := by
  have h : 112 > 97 := by sorry
  have key := square_difference_division 112 97 h
  sorry

end NUMINAMATH_CALUDE_square_difference_division_problem_solution_l3410_341004


namespace NUMINAMATH_CALUDE_savings_percentage_l3410_341074

def monthly_salary : ℝ := 1000
def savings_after_increase : ℝ := 175
def expense_increase_rate : ℝ := 0.10

theorem savings_percentage :
  ∃ (savings_rate : ℝ),
    savings_rate * monthly_salary = monthly_salary - (monthly_salary - savings_rate * monthly_salary) * (1 + expense_increase_rate) ∧
    savings_rate * monthly_salary = savings_after_increase ∧
    savings_rate = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_savings_percentage_l3410_341074


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_value_l3410_341067

/-- If the roots of the quadratic 10x^2 - 6x + m are (3 ± i√191)/10, then m = 227/40 -/
theorem quadratic_roots_imply_m_value (m : ℝ) : 
  (∃ x : ℂ, x^2 * 10 - x * 6 + m = 0 ∧ x = (3 + Complex.I * Real.sqrt 191) / 10) →
  m = 227 / 40 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_value_l3410_341067


namespace NUMINAMATH_CALUDE_sticker_distribution_count_l3410_341031

/-- The number of ways to partition n identical objects into k or fewer non-negative integer parts -/
def partition_count (n k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution_count : 
  partition_count num_stickers num_sheets = 30 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_count_l3410_341031


namespace NUMINAMATH_CALUDE_min_value_of_sum_l3410_341022

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b - a * b = 0) :
  a + b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 4 * a₀ + b₀ - a₀ * b₀ = 0 ∧ a₀ + b₀ = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l3410_341022


namespace NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l3410_341027

theorem sum_of_four_primes_divisible_by_60 
  (p q r s : ℕ) 
  (hp : Nat.Prime p) 
  (hq : Nat.Prime q) 
  (hr : Nat.Prime r) 
  (hs : Nat.Prime s) 
  (h_order : 5 < p ∧ p < q ∧ q < r ∧ r < s ∧ s < p + 10) : 
  60 ∣ (p + q + r + s) := by
sorry

end NUMINAMATH_CALUDE_sum_of_four_primes_divisible_by_60_l3410_341027


namespace NUMINAMATH_CALUDE_fourth_term_is_twenty_l3410_341051

def sequence_term (n : ℕ) : ℕ := n + 2^n

theorem fourth_term_is_twenty : sequence_term 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_is_twenty_l3410_341051


namespace NUMINAMATH_CALUDE_sharadek_word_guessing_l3410_341098

theorem sharadek_word_guessing (n : ℕ) (h : n ≤ 1000000) :
  ∃ (q : ℕ), q ≤ 20 ∧ 2^q ≥ n := by
  sorry

end NUMINAMATH_CALUDE_sharadek_word_guessing_l3410_341098


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3410_341090

open Set

def M : Set ℝ := {x | x^2 - 4*x < 0}
def N : Set ℝ := {x | |x| ≤ 2}

theorem union_of_M_and_N : M ∪ N = Icc (-2) 4 := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3410_341090


namespace NUMINAMATH_CALUDE_problem_solution_l3410_341014

theorem problem_solution (a b : ℕ) 
  (sum_eq : a + b = 31462)
  (b_div_20 : b % 20 = 0)
  (a_eq_b_div_10 : a = b / 10) : 
  b - a = 28462 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3410_341014


namespace NUMINAMATH_CALUDE_unique_value_2n_plus_m_l3410_341054

theorem unique_value_2n_plus_m : ∃! v : ℤ, ∀ n m : ℤ,
  (3 * n - m < 5) →
  (n + m > 26) →
  (3 * m - 2 * n < 46) →
  (2 * n + m = v) := by
  sorry

end NUMINAMATH_CALUDE_unique_value_2n_plus_m_l3410_341054


namespace NUMINAMATH_CALUDE_correct_probability_l3410_341080

def total_rolls : ℕ := 12
def rolls_per_type : ℕ := 3
def num_types : ℕ := 4
def rolls_per_guest : ℕ := 4

def probability_correct_selection : ℚ :=
  (rolls_per_type * (rolls_per_type - 1) * (rolls_per_type - 2)) /
  (total_rolls * (total_rolls - 1) * (total_rolls - 2) * (total_rolls - 3))

theorem correct_probability :
  probability_correct_selection = 9 / 55 := by sorry

end NUMINAMATH_CALUDE_correct_probability_l3410_341080


namespace NUMINAMATH_CALUDE_binomial_coefficient_1000_l3410_341003

theorem binomial_coefficient_1000 : Nat.choose 1000 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_1000_l3410_341003


namespace NUMINAMATH_CALUDE_new_class_mean_l3410_341018

theorem new_class_mean (total_students : ℕ) (first_group : ℕ) (second_group : ℕ) 
  (first_mean : ℚ) (second_mean : ℚ) :
  total_students = first_group + second_group →
  first_group = 45 →
  second_group = 5 →
  first_mean = 80 / 100 →
  second_mean = 90 / 100 →
  (first_group * first_mean + second_group * second_mean) / total_students = 81 / 100 := by
sorry

end NUMINAMATH_CALUDE_new_class_mean_l3410_341018


namespace NUMINAMATH_CALUDE_roots_of_quadratic_l3410_341020

/-- The quadratic function f(x) = ax^2 + bx -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The equation f(x) = 6 -/
def equation (a b : ℝ) (x : ℝ) : Prop := f a b x = 6

theorem roots_of_quadratic (a b : ℝ) :
  equation a b (-2) ∧ equation a b 3 →
  (equation a b (-2) ∧ equation a b 3 ∧
   ∀ x : ℝ, equation a b x → x = -2 ∨ x = 3) :=
by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_l3410_341020


namespace NUMINAMATH_CALUDE_equation_solution_l3410_341023

theorem equation_solution :
  ∃ x : ℝ, (3 / 4 + 1 / x = 7 / 8) ∧ (x = 8) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3410_341023


namespace NUMINAMATH_CALUDE_total_population_of_three_cities_l3410_341030

/-- Given the populations of three cities with specific relationships, 
    prove that their total population is 56000. -/
theorem total_population_of_three_cities 
  (pop_lake_view pop_seattle pop_boise : ℕ) : 
  pop_lake_view = 24000 →
  pop_lake_view = pop_seattle + 4000 →
  pop_boise = (3 * pop_seattle) / 5 →
  pop_lake_view + pop_seattle + pop_boise = 56000 := by
  sorry

end NUMINAMATH_CALUDE_total_population_of_three_cities_l3410_341030


namespace NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l3410_341088

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25)

/-- The factorial of n -/
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem thirty_factorial_trailing_zeros :
  trailingZeros 30 = 7 :=
by sorry

end NUMINAMATH_CALUDE_thirty_factorial_trailing_zeros_l3410_341088


namespace NUMINAMATH_CALUDE_square_border_pieces_l3410_341041

/-- The number of pieces on one side of the square arrangement -/
def side_length : ℕ := 12

/-- The total number of pieces in the border of a square arrangement -/
def border_pieces (n : ℕ) : ℕ := 2 * n + 2 * (n - 2)

/-- Theorem stating that in a 12x12 square arrangement, there are 44 pieces in the border -/
theorem square_border_pieces :
  border_pieces side_length = 44 := by
  sorry

#eval border_pieces side_length

end NUMINAMATH_CALUDE_square_border_pieces_l3410_341041


namespace NUMINAMATH_CALUDE_common_chord_length_l3410_341064

-- Define the circles
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 25
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 2*y - 20 = 0

-- Define the intersection points
def intersection_points (A B : ℝ × ℝ) : Prop :=
  circle_O A.1 A.2 ∧ circle_C A.1 A.2 ∧
  circle_O B.1 B.2 ∧ circle_C B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem common_chord_length (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 95 := by
  sorry

end NUMINAMATH_CALUDE_common_chord_length_l3410_341064


namespace NUMINAMATH_CALUDE_real_part_of_z_l3410_341033

theorem real_part_of_z (z : ℂ) (h : z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I) :
  z.re = (Real.sqrt 2 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_z_l3410_341033


namespace NUMINAMATH_CALUDE_product_of_solutions_abs_value_equation_l3410_341024

theorem product_of_solutions_abs_value_equation :
  ∃ (x₁ x₂ : ℝ), (|x₁| = 3 * (|x₁| - 2) ∧ |x₂| = 3 * (|x₂| - 2) ∧ x₁ ≠ x₂) ∧ x₁ * x₂ = -9 :=
by sorry

end NUMINAMATH_CALUDE_product_of_solutions_abs_value_equation_l3410_341024


namespace NUMINAMATH_CALUDE_lomonosov_digit_mapping_l3410_341042

theorem lomonosov_digit_mapping :
  ∃ (L O M N S V H C B : ℕ),
    (L < 10) ∧ (O < 10) ∧ (M < 10) ∧ (N < 10) ∧ (S < 10) ∧
    (V < 10) ∧ (H < 10) ∧ (C < 10) ∧ (B < 10) ∧
    (L ≠ O) ∧ (L ≠ M) ∧ (L ≠ N) ∧ (L ≠ S) ∧ (L ≠ V) ∧ (L ≠ H) ∧ (L ≠ C) ∧ (L ≠ B) ∧
    (O ≠ M) ∧ (O ≠ N) ∧ (O ≠ S) ∧ (O ≠ V) ∧ (O ≠ H) ∧ (O ≠ C) ∧ (O ≠ B) ∧
    (M ≠ N) ∧ (M ≠ S) ∧ (M ≠ V) ∧ (M ≠ H) ∧ (M ≠ C) ∧ (M ≠ B) ∧
    (N ≠ S) ∧ (N ≠ V) ∧ (N ≠ H) ∧ (N ≠ C) ∧ (N ≠ B) ∧
    (S ≠ V) ∧ (S ≠ H) ∧ (S ≠ C) ∧ (S ≠ B) ∧
    (V ≠ H) ∧ (V ≠ C) ∧ (V ≠ B) ∧
    (H ≠ C) ∧ (H ≠ B) ∧
    (C ≠ B) ∧
    (L + O / M + O + H + O / C = O * 10 + B) ∧
    (O < M) ∧ (O < C) := by
  sorry

end NUMINAMATH_CALUDE_lomonosov_digit_mapping_l3410_341042


namespace NUMINAMATH_CALUDE_prob_king_ace_standard_deck_l3410_341058

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (kings : ℕ)
  (aces : ℕ)

/-- Calculates the probability of drawing a King first and an Ace second without replacement -/
def prob_king_ace (d : Deck) : ℚ :=
  (d.kings : ℚ) / d.total_cards * d.aces / (d.total_cards - 1)

/-- Theorem: The probability of drawing a King first and an Ace second from a standard deck is 4/663 -/
theorem prob_king_ace_standard_deck :
  let standard_deck : Deck := ⟨52, 4, 4⟩
  prob_king_ace standard_deck = 4 / 663 := by
sorry

end NUMINAMATH_CALUDE_prob_king_ace_standard_deck_l3410_341058


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3410_341010

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ a, a > 1 → 1/a < 1) ∧ 
  (∃ a, 1/a < 1 ∧ ¬(a > 1)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3410_341010


namespace NUMINAMATH_CALUDE_problem_statement_l3410_341093

theorem problem_statement (x y : ℝ) (h : x - 2*y = -5) : 2 - x + 2*y = 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3410_341093


namespace NUMINAMATH_CALUDE_unique_solution_is_four_l3410_341008

-- Define the equation
def equation (s x : ℝ) : Prop :=
  1 / (3 * x) = (s - x) / 9

-- State the theorem
theorem unique_solution_is_four :
  ∃! s : ℝ, (∃! x : ℝ, equation s x) ∧ s = 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_is_four_l3410_341008


namespace NUMINAMATH_CALUDE_range_of_a_l3410_341011

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + x + (1/2 : ℝ) > 0) → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3410_341011


namespace NUMINAMATH_CALUDE_parabola_constant_l3410_341025

theorem parabola_constant (c : ℝ) : 
  (∃ (x y : ℝ), y = x^2 - c ∧ x = 3 ∧ y = 8) → c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_constant_l3410_341025


namespace NUMINAMATH_CALUDE_log_equality_l3410_341015

theorem log_equality (x : ℝ) (h : x > 0) :
  (Real.log (2 * x) / Real.log (5 * x) = Real.log (8 * x) / Real.log (625 * x)) →
  (Real.log x / Real.log 2 = Real.log 5 / (2 * Real.log 2 - 3 * Real.log 5)) :=
by sorry

end NUMINAMATH_CALUDE_log_equality_l3410_341015


namespace NUMINAMATH_CALUDE_factorization_equality_l3410_341084

theorem factorization_equality (x : ℝ) :
  (x^2 - x - 6) * (x^2 + 3*x - 4) + 24 =
  (x + 3) * (x - 2) * (x + (1 + Real.sqrt 33) / 2) * (x + (1 - Real.sqrt 33) / 2) := by
sorry

end NUMINAMATH_CALUDE_factorization_equality_l3410_341084


namespace NUMINAMATH_CALUDE_triangle_area_l3410_341036

def a : Fin 2 → ℝ := ![5, 1]
def b : Fin 2 → ℝ := ![2, 4]

theorem triangle_area : 
  (1/2 : ℝ) * |Matrix.det ![a, b]| = 9 := by sorry

end NUMINAMATH_CALUDE_triangle_area_l3410_341036


namespace NUMINAMATH_CALUDE_digit_sum_property_l3410_341029

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem digit_sum_property (M : ℕ) :
  (∀ k : ℕ, k > 0 → k ≤ M → S (M * k) = S M) ↔
  ∃ n : ℕ, n > 0 ∧ M = 10^n - 1 :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l3410_341029


namespace NUMINAMATH_CALUDE_smallest_cube_multiple_l3410_341056

theorem smallest_cube_multiple : 
  ∃ (x : ℕ+) (M : ℤ), 
    (1890 : ℤ) * (x : ℤ) = M^3 ∧ 
    (∀ (y : ℕ+) (N : ℤ), (1890 : ℤ) * (y : ℤ) = N^3 → x ≤ y) ∧
    x = 4900 := by
  sorry

end NUMINAMATH_CALUDE_smallest_cube_multiple_l3410_341056


namespace NUMINAMATH_CALUDE_river_boat_capacity_l3410_341007

theorem river_boat_capacity (river_width : ℕ) (boat_width : ℕ) (space_required : ℕ) : 
  river_width = 42 ∧ boat_width = 3 ∧ space_required = 2 →
  (river_width / (boat_width + 2 * space_required) : ℕ) = 6 :=
by sorry

end NUMINAMATH_CALUDE_river_boat_capacity_l3410_341007


namespace NUMINAMATH_CALUDE_money_division_l3410_341060

theorem money_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 406)
  (h_a : a = b / 2)
  (h_b : b = c / 2)
  (h_sum : a + b + c = total) : c = 232 := by
  sorry

end NUMINAMATH_CALUDE_money_division_l3410_341060


namespace NUMINAMATH_CALUDE_equation_solution_l3410_341016

theorem equation_solution (x : ℝ) : 1 / x + x / 80 = 7 / 30 → x = 12 ∨ x = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3410_341016


namespace NUMINAMATH_CALUDE_quadrilateral_area_bounds_sum_l3410_341077

/-- A convex quadrilateral with given side lengths -/
structure ConvexQuadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  convex : ab > 0 ∧ bc > 0 ∧ cd > 0 ∧ da > 0

/-- The area of a convex quadrilateral -/
def area (q : ConvexQuadrilateral) : ℝ := sorry

/-- The lower bound of the area of a convex quadrilateral -/
def lowerBound (q : ConvexQuadrilateral) : ℝ := sorry

/-- The upper bound of the area of a convex quadrilateral -/
def upperBound (q : ConvexQuadrilateral) : ℝ := sorry

theorem quadrilateral_area_bounds_sum :
  ∀ q : ConvexQuadrilateral,
  q.ab = 7 ∧ q.bc = 4 ∧ q.cd = 5 ∧ q.da = 6 →
  lowerBound q + upperBound q = 2 * Real.sqrt 210 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_bounds_sum_l3410_341077


namespace NUMINAMATH_CALUDE_xNotEqual1_is_valid_l3410_341092

/-- Valid conditional operators -/
inductive ConditionalOperator
  | gt  -- >
  | ge  -- >=
  | lt  -- <
  | ne  -- <>
  | le  -- <=
  | eq  -- =

/-- A conditional expression -/
structure ConditionalExpression where
  operator : ConditionalOperator
  value : ℝ

/-- Check if a conditional expression is valid -/
def isValidConditionalExpression (expr : ConditionalExpression) : Prop :=
  expr.operator ∈ [ConditionalOperator.gt, ConditionalOperator.ge, ConditionalOperator.lt, 
                   ConditionalOperator.ne, ConditionalOperator.le, ConditionalOperator.eq]

/-- The specific conditional expression "x <> 1" -/
def xNotEqual1 : ConditionalExpression :=
  { operator := ConditionalOperator.ne, value := 1 }

/-- Theorem: "x <> 1" is a valid conditional expression -/
theorem xNotEqual1_is_valid : isValidConditionalExpression xNotEqual1 := by
  sorry

end NUMINAMATH_CALUDE_xNotEqual1_is_valid_l3410_341092


namespace NUMINAMATH_CALUDE_sara_frosting_cans_l3410_341035

/-- The number of cans of frosting needed to frost the remaining cakes after Sara's baking and Carol's eating -/
def frosting_cans_needed (cakes_per_day : ℕ) (days : ℕ) (cakes_eaten : ℕ) (frosting_cans_per_cake : ℕ) : ℕ :=
  ((cakes_per_day * days - cakes_eaten) * frosting_cans_per_cake)

/-- Theorem stating the number of frosting cans needed in Sara's specific scenario -/
theorem sara_frosting_cans : frosting_cans_needed 10 5 12 2 = 76 := by
  sorry

end NUMINAMATH_CALUDE_sara_frosting_cans_l3410_341035


namespace NUMINAMATH_CALUDE_incorrect_statement_E_l3410_341047

theorem incorrect_statement_E (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  -- Statement A
  (∀ x y : ℝ, x > 0 → y > 0 → x > y → x^2 > y^2) ∧
  -- Statement B
  (2 * a * b / (a + b) < Real.sqrt (a * b)) ∧
  -- Statement C
  (∀ p : ℝ, p > 0 → ∀ x y : ℝ, x > 0 → y > 0 → x * y = p → 
    x + y ≥ 2 * Real.sqrt p ∧ (x + y = 2 * Real.sqrt p ↔ x = y)) ∧
  -- Statement D
  ((a + b)^3 > (a^3 + b^3) / 2) ∧
  -- Statement E (negation)
  ¬((a + b)^2 / 4 > (a^2 + b^2) / 2) := by
sorry

end NUMINAMATH_CALUDE_incorrect_statement_E_l3410_341047


namespace NUMINAMATH_CALUDE_triangle_properties_l3410_341065

-- Define an acute triangle ABC
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  acute : angle_A > 0 ∧ angle_A < Real.pi/2 ∧
          angle_B > 0 ∧ angle_B < Real.pi/2 ∧
          angle_C > 0 ∧ angle_C < Real.pi/2

-- State the theorem
theorem triangle_properties (t : AcuteTriangle) 
  (h1 : t.a^2 + t.b^2 - t.c^2 = t.a * t.b)
  (h2 : t.c = Real.sqrt 7)
  (h3 : (1/2) * t.a * t.b * Real.sin t.angle_C = (3 * Real.sqrt 3) / 2) :
  t.angle_C = Real.pi/3 ∧ t.a + t.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3410_341065


namespace NUMINAMATH_CALUDE_circle_area_tripled_l3410_341005

theorem circle_area_tripled (r n : ℝ) : 
  (r > 0) → (n > 0) → (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 + 1)) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l3410_341005


namespace NUMINAMATH_CALUDE_largest_difference_l3410_341026

def S : Set Int := {-20, -5, 1, 5, 7, 19}

theorem largest_difference (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x - y = 39 ∧ ∀ (c d : Int), c ∈ S → d ∈ S → c - d ≤ 39 := by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l3410_341026


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3410_341068

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^7 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = -2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3410_341068


namespace NUMINAMATH_CALUDE_triangle_identity_l3410_341075

/-- For any triangle with sides a, b, c, circumradius R, and altitude CH from vertex C to side AB,
    the identity (a² + b² - c²) / (ab) = CH / R holds. -/
theorem triangle_identity (a b c R CH : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hR : R > 0) (hCH : CH > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (a^2 + b^2 - c^2) / (a * b) = CH / R := by
  sorry

end NUMINAMATH_CALUDE_triangle_identity_l3410_341075
