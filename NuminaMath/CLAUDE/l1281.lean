import Mathlib

namespace NUMINAMATH_CALUDE_lisa_photos_l1281_128188

def photo_problem (animal_photos flower_photos scenery_photos this_weekend last_weekend : ℕ) : Prop :=
  animal_photos = 10 ∧
  flower_photos = 3 * animal_photos ∧
  scenery_photos = flower_photos - 10 ∧
  this_weekend = animal_photos + flower_photos + scenery_photos ∧
  last_weekend = this_weekend - 15

theorem lisa_photos :
  ∀ animal_photos flower_photos scenery_photos this_weekend last_weekend,
  photo_problem animal_photos flower_photos scenery_photos this_weekend last_weekend →
  last_weekend = 45 := by
sorry

end NUMINAMATH_CALUDE_lisa_photos_l1281_128188


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1281_128154

theorem quadratic_equation_roots (α β : ℝ) : 
  ((1 + β) / (2 + β) = -1 / α) ∧ 
  ((α * β^2 + 121) / (1 - α^2 * β) = 1) →
  (∃ a b c : ℝ, (a * α^2 + b * α + c = 0) ∧ 
               (a * β^2 + b * β + c = 0) ∧ 
               ((a = 1 ∧ b = 12 ∧ c = 10) ∨ 
                (a = 1 ∧ b = -10 ∧ c = -12))) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1281_128154


namespace NUMINAMATH_CALUDE_triple_digit_sum_of_2012_pow_2012_l1281_128168

/-- The sum of the digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- The function that applies digit_sum three times -/
def triple_digit_sum (n : ℕ) : ℕ := digit_sum (digit_sum (digit_sum n))

theorem triple_digit_sum_of_2012_pow_2012 :
  triple_digit_sum (2012^2012) = 7 := by sorry

end NUMINAMATH_CALUDE_triple_digit_sum_of_2012_pow_2012_l1281_128168


namespace NUMINAMATH_CALUDE_prob_same_foot_is_three_sevenths_l1281_128198

/-- The number of pairs of shoes in the cabinet -/
def num_pairs : ℕ := 4

/-- The total number of shoes in the cabinet -/
def total_shoes : ℕ := 2 * num_pairs

/-- The number of shoes selected -/
def selected_shoes : ℕ := 2

/-- The number of ways to select 2 shoes out of the total shoes -/
def total_selections : ℕ := Nat.choose total_shoes selected_shoes

/-- The number of ways to select 2 shoes from the same foot -/
def same_foot_selections : ℕ := 2 * Nat.choose num_pairs selected_shoes

/-- The probability of selecting two shoes from the same foot -/
def prob_same_foot : ℚ := same_foot_selections / total_selections

theorem prob_same_foot_is_three_sevenths :
  prob_same_foot = 3 / 7 := by sorry

end NUMINAMATH_CALUDE_prob_same_foot_is_three_sevenths_l1281_128198


namespace NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_6x_l1281_128171

theorem largest_x_sqrt_3x_eq_6x :
  ∃ (x_max : ℚ), x_max = 1/12 ∧
  (∀ x : ℚ, x ≥ 0 → Real.sqrt (3 * x) = 6 * x → x ≤ x_max) ∧
  Real.sqrt (3 * x_max) = 6 * x_max :=
sorry

end NUMINAMATH_CALUDE_largest_x_sqrt_3x_eq_6x_l1281_128171


namespace NUMINAMATH_CALUDE_vector_subtraction_magnitude_l1281_128174

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-2, 4)

theorem vector_subtraction_magnitude : ‖a - b‖ = 5 := by sorry

end NUMINAMATH_CALUDE_vector_subtraction_magnitude_l1281_128174


namespace NUMINAMATH_CALUDE_rounded_expression_smaller_l1281_128105

theorem rounded_expression_smaller (a b c : ℕ+) :
  let exact_value := (a.val^2 : ℚ) / b.val + c.val^3
  let rounded_a := (a.val + 1 : ℚ)
  let rounded_b := (b.val + 1 : ℚ)
  let rounded_c := (c.val - 1 : ℚ)
  let rounded_value := rounded_a^2 / rounded_b + rounded_c^3
  rounded_value < exact_value :=
by sorry

end NUMINAMATH_CALUDE_rounded_expression_smaller_l1281_128105


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l1281_128141

theorem interval_length_implies_difference (c d : ℝ) :
  (∀ x : ℝ, c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d) →
  (∀ x : ℝ, c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d ↔ (c + 2) / 3 ≤ x ∧ x ≤ (d + 2) / 3) →
  ((d + 2) / 3 - (c + 2) / 3 = 15) →
  d - c = 45 := by
  sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l1281_128141


namespace NUMINAMATH_CALUDE_three_integers_difference_l1281_128116

theorem three_integers_difference (x y z : ℕ+) 
  (sum_xy : x + y = 998)
  (sum_xz : x + z = 1050)
  (sum_yz : y + z = 1234) :
  max x (max y z) - min x (min y z) = 236 := by
sorry

end NUMINAMATH_CALUDE_three_integers_difference_l1281_128116


namespace NUMINAMATH_CALUDE_inequality_and_equality_l1281_128160

theorem inequality_and_equality (x : ℝ) (h : x > 0) : 
  (x + 1/x ≥ 2) ∧ (x + 1/x = 2 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_equality_l1281_128160


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1281_128109

theorem polynomial_factorization (x : ℤ) :
  x^12 + x^9 + 1 = (x^4 + x^3 + x^2 + x + 1) * (x^8 - x^7 + x^6 - x^5 + x^3 - x^2 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1281_128109


namespace NUMINAMATH_CALUDE_alice_burger_spending_l1281_128149

/-- The number of days in June -/
def june_days : ℕ := 30

/-- The number of burgers Alice purchases each day -/
def burgers_per_day : ℕ := 4

/-- The cost of each burger in dollars -/
def burger_cost : ℕ := 13

/-- The total amount Alice spent on burgers in June -/
def total_spent : ℕ := june_days * burgers_per_day * burger_cost

theorem alice_burger_spending :
  total_spent = 1560 := by
  sorry

end NUMINAMATH_CALUDE_alice_burger_spending_l1281_128149


namespace NUMINAMATH_CALUDE_only_one_divides_power_plus_one_l1281_128189

theorem only_one_divides_power_plus_one :
  ∀ n : ℕ+, n.val % 2 = 1 ∧ (n.val ∣ 3^n.val + 1) → n = 1 := by sorry

end NUMINAMATH_CALUDE_only_one_divides_power_plus_one_l1281_128189


namespace NUMINAMATH_CALUDE_gcd_lcm_identity_l1281_128185

theorem gcd_lcm_identity (a b c : ℕ+) :
  (Nat.lcm (Nat.lcm a b) c)^2 / (Nat.lcm a b * Nat.lcm b c * Nat.lcm c a) =
  (Nat.gcd (Nat.gcd a b) c)^2 / (Nat.gcd a b * Nat.gcd b c * Nat.gcd c a) :=
by sorry

end NUMINAMATH_CALUDE_gcd_lcm_identity_l1281_128185


namespace NUMINAMATH_CALUDE_brandon_sales_theorem_l1281_128183

def total_sales : ℝ := 80

theorem brandon_sales_theorem :
  let credit_sales_ratio : ℝ := 2/5
  let cash_sales_ratio : ℝ := 1 - credit_sales_ratio
  let cash_sales_amount : ℝ := 48
  cash_sales_ratio * total_sales = cash_sales_amount :=
by sorry

end NUMINAMATH_CALUDE_brandon_sales_theorem_l1281_128183


namespace NUMINAMATH_CALUDE_wood_weight_calculation_l1281_128181

/-- Given a square piece of wood with side length 4 inches weighing 20 ounces,
    calculate the weight of a second square piece with side length 7 inches. -/
theorem wood_weight_calculation (thickness : ℝ) (density : ℝ) :
  let side1 : ℝ := 4
  let weight1 : ℝ := 20
  let side2 : ℝ := 7
  let area1 : ℝ := side1 * side1
  let area2 : ℝ := side2 * side2
  let weight2 : ℝ := weight1 * (area2 / area1)
  weight2 = 61.25 := by sorry

end NUMINAMATH_CALUDE_wood_weight_calculation_l1281_128181


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1281_128190

/-- An arithmetic sequence is a sequence where the difference between
    each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℚ)
  (h_arith : is_arithmetic_sequence a)
  (h_2 : a 2 = 1)
  (h_8 : a 8 = 2 * a 6 + a 4) :
  a 5 = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l1281_128190


namespace NUMINAMATH_CALUDE_initial_blocks_l1281_128170

theorem initial_blocks (initial : ℕ) (added : ℕ) (total : ℕ) : 
  added = 9 → total = 95 → initial + added = total → initial = 86 := by
  sorry

end NUMINAMATH_CALUDE_initial_blocks_l1281_128170


namespace NUMINAMATH_CALUDE_sum_of_b_values_l1281_128138

theorem sum_of_b_values (b₁ b₂ : ℝ) : 
  (∃! x, 9 * x^2 + b₁ * x + 15 * x + 16 = 0) ∧
  (∃! x, 9 * x^2 + b₂ * x + 15 * x + 16 = 0) →
  b₁ + b₂ = -30 := by
sorry

end NUMINAMATH_CALUDE_sum_of_b_values_l1281_128138


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1281_128169

/-- Given a geometric sequence {a_n} with first term a₁ and common ratio q,
    if a₁ + a₃ = 10 and a₄ + a₆ = 5/4, then q = 1/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h1 : ∀ n, a (n + 1) = a n * q) 
  (h2 : a 1 + a 3 = 10) 
  (h3 : a 4 + a 6 = 5/4) : 
  q = 1/2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1281_128169


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_l1281_128184

/-- The line x + y - 1 = 0 does not pass through the third quadrant -/
theorem line_not_in_third_quadrant :
  ∀ x y : ℝ, x + y - 1 = 0 → ¬(x < 0 ∧ y < 0) := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_l1281_128184


namespace NUMINAMATH_CALUDE_sum_product_theorem_l1281_128162

theorem sum_product_theorem (a b c d : ℝ) 
  (eq1 : a + b + c = -4)
  (eq2 : a + b + d = 2)
  (eq3 : a + c + d = 15)
  (eq4 : b + c + d = 10) :
  a * b + c * d = 485 / 9 := by
sorry

end NUMINAMATH_CALUDE_sum_product_theorem_l1281_128162


namespace NUMINAMATH_CALUDE_infinite_series_sum_l1281_128178

/-- The sum of the infinite series ∑(n=1 to ∞) (2n + 1) / (n(n + 1)(n + 2)) is equal to 1 -/
theorem infinite_series_sum : 
  (∑' n : ℕ+, (2 * n.val + 1 : ℝ) / (n.val * (n.val + 1) * (n.val + 2))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_sum_l1281_128178


namespace NUMINAMATH_CALUDE_square_root_power_and_increasing_l1281_128112

-- Define the function f(x) = x^(1/2) on the interval (0, +∞)
def f : ℝ → ℝ := fun x ↦ x^(1/2)

-- Define the interval (0, +∞)
def openRightHalfLine : Set ℝ := {x : ℝ | x > 0}

theorem square_root_power_and_increasing :
  (∃ r : ℝ, ∀ x ∈ openRightHalfLine, f x = x^r) ∧
  StrictMonoOn f openRightHalfLine :=
sorry

end NUMINAMATH_CALUDE_square_root_power_and_increasing_l1281_128112


namespace NUMINAMATH_CALUDE_remainder_97_pow_45_mod_100_l1281_128101

theorem remainder_97_pow_45_mod_100 : 97^45 % 100 = 57 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_pow_45_mod_100_l1281_128101


namespace NUMINAMATH_CALUDE_complement_of_union_l1281_128151

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem complement_of_union : U \ (A ∪ B) = {4} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l1281_128151


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l1281_128124

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (hn : n > 0) 
  (h_divisible : 72 ∣ n^2) : 
  ∀ m : ℕ, m ∣ n → m ≤ 12 ∧ 12 ∣ n :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l1281_128124


namespace NUMINAMATH_CALUDE_sin_cos_difference_l1281_128126

theorem sin_cos_difference (x : Real) :
  (Real.sin x)^3 - (Real.cos x)^3 = -1 → Real.sin x - Real.cos x = -1 := by
sorry

end NUMINAMATH_CALUDE_sin_cos_difference_l1281_128126


namespace NUMINAMATH_CALUDE_colored_ball_probability_l1281_128115

/-- The probability of drawing a colored ball from an urn -/
theorem colored_ball_probability (total : ℕ) (blue green white : ℕ)
  (h_total : total = blue + green + white)
  (h_blue : blue = 15)
  (h_green : green = 5)
  (h_white : white = 20) :
  (blue + green : ℚ) / total = 1 / 2 :=
by sorry

end NUMINAMATH_CALUDE_colored_ball_probability_l1281_128115


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1281_128161

/-- A quadratic function satisfying specific conditions -/
def f (x : ℝ) : ℝ := 6 * x^2 - 4

/-- The theorem stating the properties of the quadratic function -/
theorem quadratic_function_properties :
  (f (-1) = 2) ∧
  (deriv f 0 = 0) ∧
  (∫ x in (0)..(1), f x = -2) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≥ -4) ∧
  (∀ x ∈ Set.Icc (-1) 1, f x ≤ 2) ∧
  (f 0 = -4) ∧
  (f 1 = 2) ∧
  (f (-1) = 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1281_128161


namespace NUMINAMATH_CALUDE_min_distance_point_coordinates_l1281_128180

/-- Given two fixed points C(0,4) and K(6,0) in a Cartesian coordinate system,
    with A being a moving point on the line segment OK,
    D being the midpoint of AC,
    and B obtained by rotating AD clockwise 90° around A,
    prove that when BK reaches its minimum value,
    the coordinates of point B are (26/5, 8/5). -/
theorem min_distance_point_coordinates :
  ∀ (A : ℝ × ℝ) (B : ℝ × ℝ),
  let C : ℝ × ℝ := (0, 4)
  let K : ℝ × ℝ := (6, 0)
  let O : ℝ × ℝ := (0, 0)
  -- A is on line segment OK
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = (t * K.1, 0)) →
  -- D is midpoint of AC
  let D : ℝ × ℝ := ((A.1 + C.1) / 2, (A.2 + C.2) / 2)
  -- B is obtained by rotating AD 90° clockwise around A
  (B.1 = A.1 + (D.2 - A.2) ∧ B.2 = A.2 - (D.1 - A.1)) →
  -- When BK reaches its minimum value
  (∀ (A' : ℝ × ℝ) (B' : ℝ × ℝ),
    (∃ t' : ℝ, 0 ≤ t' ∧ t' ≤ 1 ∧ A' = (t' * K.1, 0)) →
    let D' : ℝ × ℝ := ((A'.1 + C.1) / 2, (A'.2 + C.2) / 2)
    (B'.1 = A'.1 + (D'.2 - A'.2) ∧ B'.2 = A'.2 - (D'.1 - A'.1)) →
    (B.1 - K.1)^2 + (B.2 - K.2)^2 ≤ (B'.1 - K.1)^2 + (B'.2 - K.2)^2) →
  -- Then the coordinates of B are (26/5, 8/5)
  B = (26/5, 8/5) := by sorry

end NUMINAMATH_CALUDE_min_distance_point_coordinates_l1281_128180


namespace NUMINAMATH_CALUDE_cost_of_3200_pencils_l1281_128137

/-- The cost of a given number of pencils based on a known price for a box of pencils -/
def pencil_cost (box_size : ℕ) (box_cost : ℚ) (num_pencils : ℕ) : ℚ :=
  (box_cost * num_pencils) / box_size

/-- Theorem: Given a box of 160 personalized pencils costs $48, the cost of 3200 pencils is $960 -/
theorem cost_of_3200_pencils :
  pencil_cost 160 48 3200 = 960 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_3200_pencils_l1281_128137


namespace NUMINAMATH_CALUDE_tomato_price_per_pound_l1281_128157

/-- Calculates the price per pound of a tomato based on grocery shopping details. -/
theorem tomato_price_per_pound
  (meat_weight : Real)
  (meat_price_per_pound : Real)
  (buns_price : Real)
  (lettuce_price : Real)
  (tomato_weight : Real)
  (pickles_price : Real)
  (pickles_coupon : Real)
  (paid_amount : Real)
  (change_received : Real)
  (h1 : meat_weight = 2)
  (h2 : meat_price_per_pound = 3.5)
  (h3 : buns_price = 1.5)
  (h4 : lettuce_price = 1)
  (h5 : tomato_weight = 1.5)
  (h6 : pickles_price = 2.5)
  (h7 : pickles_coupon = 1)
  (h8 : paid_amount = 20)
  (h9 : change_received = 6) :
  (paid_amount - change_received - (meat_weight * meat_price_per_pound + buns_price + lettuce_price + (pickles_price - pickles_coupon))) / tomato_weight = 2 := by
  sorry


end NUMINAMATH_CALUDE_tomato_price_per_pound_l1281_128157


namespace NUMINAMATH_CALUDE_problems_finished_at_school_l1281_128134

def math_problems : ℕ := 18
def science_problems : ℕ := 11
def problems_left : ℕ := 5

theorem problems_finished_at_school :
  math_problems + science_problems - problems_left = 24 := by
  sorry

end NUMINAMATH_CALUDE_problems_finished_at_school_l1281_128134


namespace NUMINAMATH_CALUDE_age_ratio_proof_l1281_128120

/-- Given Ronaldo's current age and the future age ratio, prove the past age ratio -/
theorem age_ratio_proof (ronaldo_current_age : ℕ) (future_ratio : ℚ) : 
  ronaldo_current_age = 36 → 
  future_ratio = 7 / 8 → 
  ∃ (roonie_current_age : ℕ), 
    (roonie_current_age + 4 : ℚ) / (ronaldo_current_age + 4 : ℚ) = future_ratio ∧ 
    (roonie_current_age - 1 : ℚ) / (ronaldo_current_age - 1 : ℚ) = 6 / 7 :=
by sorry

end NUMINAMATH_CALUDE_age_ratio_proof_l1281_128120


namespace NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l1281_128125

theorem binomial_coeff_not_coprime (n k l : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : 1 ≤ l) (h4 : l < n) :
  Nat.gcd (Nat.choose n k) (Nat.choose n l) > 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coeff_not_coprime_l1281_128125


namespace NUMINAMATH_CALUDE_fourth_root_of_81_l1281_128153

theorem fourth_root_of_81 : Real.sqrt (Real.sqrt 81) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_of_81_l1281_128153


namespace NUMINAMATH_CALUDE_new_year_cards_cost_l1281_128175

def card_price_1 : ℚ := 10 / 100
def card_price_2 : ℚ := 15 / 100
def card_price_3 : ℚ := 25 / 100
def card_price_4 : ℚ := 40 / 100

def total_cards : ℕ := 30

theorem new_year_cards_cost (q1 q2 q3 q4 : ℕ) 
  (h1 : q1 + q2 + q3 + q4 = total_cards)
  (h2 : (q1 = 5 ∧ q2 = 5) ∨ (q1 = 5 ∧ q3 = 5) ∨ (q1 = 5 ∧ q4 = 5) ∨ 
        (q2 = 5 ∧ q3 = 5) ∨ (q2 = 5 ∧ q4 = 5) ∨ (q3 = 5 ∧ q4 = 5))
  (h3 : (q1 = 10 ∧ q2 = 10) ∨ (q1 = 10 ∧ q3 = 10) ∨ (q1 = 10 ∧ q4 = 10) ∨ 
        (q2 = 10 ∧ q3 = 10) ∨ (q2 = 10 ∧ q4 = 10) ∨ (q3 = 10 ∧ q4 = 10))
  (h4 : ∃ (n : ℕ), q1 * card_price_1 + q2 * card_price_2 + q3 * card_price_3 + q4 * card_price_4 = n) :
  q1 * card_price_1 + q2 * card_price_2 + q3 * card_price_3 + q4 * card_price_4 = 7 := by
sorry


end NUMINAMATH_CALUDE_new_year_cards_cost_l1281_128175


namespace NUMINAMATH_CALUDE_three_zeros_implies_a_equals_four_l1281_128179

-- Define the piecewise function f
noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x ≠ 3 then 2 / |x - 3| else a

-- Define the function y
noncomputable def y (x : ℝ) (a : ℝ) : ℝ := f x a - 4

-- Theorem statement
theorem three_zeros_implies_a_equals_four (a : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
    y x₁ a = 0 ∧ y x₂ a = 0 ∧ y x₃ a = 0) →
  (∀ x : ℝ, y x a = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) →
  a = 4 :=
by sorry


end NUMINAMATH_CALUDE_three_zeros_implies_a_equals_four_l1281_128179


namespace NUMINAMATH_CALUDE_bobby_paycheck_l1281_128155

/-- Calculates the final amount in Bobby's paycheck after deductions --/
def final_paycheck (gross_salary : ℚ) : ℚ :=
  let federal_tax := gross_salary * (1/3)
  let state_tax := gross_salary * (8/100)
  let local_tax := gross_salary * (5/100)
  let health_insurance := 50
  let life_insurance := 20
  let parking_fee := 10
  let retirement_contribution := gross_salary * (3/100)
  let total_deductions := federal_tax + state_tax + local_tax + health_insurance + life_insurance + parking_fee + retirement_contribution
  gross_salary - total_deductions

/-- Proves that Bobby's final paycheck amount is $148 --/
theorem bobby_paycheck : final_paycheck 450 = 148 := by
  sorry

#eval final_paycheck 450

end NUMINAMATH_CALUDE_bobby_paycheck_l1281_128155


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1281_128135

theorem circle_area_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) :
  (60 / 360 * (2 * π * r₁)) = (48 / 360 * (2 * π * r₂)) →
  (π * r₁^2) / (π * r₂^2) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1281_128135


namespace NUMINAMATH_CALUDE_special_function_properties_l1281_128163

/-- A function satisfying specific properties -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  (f 0 ≠ 0) ∧
  (∀ x > 0, f x > 1) ∧
  (∀ a b : ℝ, f (a + b) = f a * f b)

theorem special_function_properties (f : ℝ → ℝ) (hf : SpecialFunction f) :
  (f 0 = 1) ∧
  (∀ x : ℝ, f x > 0) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) := by
  sorry

end NUMINAMATH_CALUDE_special_function_properties_l1281_128163


namespace NUMINAMATH_CALUDE_chocolate_chip_calculation_l1281_128102

/-- The number of cups of chocolate chips needed for one recipe -/
def chips_per_recipe : ℝ := 3.5

/-- The number of recipes to be made -/
def num_recipes : ℕ := 37

/-- The total number of cups of chocolate chips needed -/
def total_chips : ℝ := chips_per_recipe * num_recipes

theorem chocolate_chip_calculation : total_chips = 129.5 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_chip_calculation_l1281_128102


namespace NUMINAMATH_CALUDE_roses_sold_l1281_128158

/-- Proves that the number of roses sold is 2, given the initial, picked, and final numbers of roses. -/
theorem roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) 
  (h1 : initial = 11) 
  (h2 : picked = 32) 
  (h3 : final = 41) : 
  initial - (final - picked) = 2 := by
  sorry

end NUMINAMATH_CALUDE_roses_sold_l1281_128158


namespace NUMINAMATH_CALUDE_division_problem_l1281_128139

theorem division_problem (divisor : ℕ) : 
  (171 / divisor = 8) ∧ (171 % divisor = 3) → divisor = 21 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1281_128139


namespace NUMINAMATH_CALUDE_speed_equivalence_l1281_128159

/-- Conversion factor from m/s to km/h -/
def meters_per_second_to_kmph : ℝ := 3.6

/-- The speed in meters per second -/
def speed_mps : ℝ := 30.0024

/-- The speed in kilometers per hour -/
def speed_kmph : ℝ := 108.00864

/-- Theorem stating that the given speed in km/h is equivalent to the given speed in m/s -/
theorem speed_equivalence : speed_kmph = speed_mps * meters_per_second_to_kmph := by
  sorry

end NUMINAMATH_CALUDE_speed_equivalence_l1281_128159


namespace NUMINAMATH_CALUDE_prob_not_pass_overall_prob_pass_technical_given_overall_l1281_128133

-- Define the probabilities of not passing each review aspect
def p_not_pass_norms : ℚ := 4/25
def p_not_pass_account : ℚ := 13/48
def p_not_pass_content : ℚ := 1/5

-- Define the probability of passing both overall review and technical skills test
def p_pass_both : ℚ := 35/100

-- Theorem for the probability of not passing overall review
theorem prob_not_pass_overall : 
  1 - (1 - p_not_pass_norms) * (1 - p_not_pass_account) * (1 - p_not_pass_content) = 51/100 := by sorry

-- Theorem for the probability of passing technical skills test given passing overall review
theorem prob_pass_technical_given_overall : 
  let p_pass_overall := 1 - (1 - (1 - p_not_pass_norms) * (1 - p_not_pass_account) * (1 - p_not_pass_content))
  p_pass_both / p_pass_overall = 5/7 := by sorry

end NUMINAMATH_CALUDE_prob_not_pass_overall_prob_pass_technical_given_overall_l1281_128133


namespace NUMINAMATH_CALUDE_b_completes_job_in_20_days_l1281_128106

/-- The number of days it takes A to complete the job -/
def days_A : ℝ := 15

/-- The number of days A and B work together -/
def days_together : ℝ := 5

/-- The fraction of the job left after A and B work together -/
def fraction_left : ℝ := 0.41666666666666663

/-- The number of days it takes B to complete the job -/
def days_B : ℝ := 20

theorem b_completes_job_in_20_days :
  (days_together * (1 / days_A + 1 / days_B) = 1 - fraction_left) ∧
  (days_B = 20) := by sorry

end NUMINAMATH_CALUDE_b_completes_job_in_20_days_l1281_128106


namespace NUMINAMATH_CALUDE_angle_value_l1281_128192

theorem angle_value (a : ℝ) : 3 * a + 150 = 360 → a = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l1281_128192


namespace NUMINAMATH_CALUDE_beach_probability_l1281_128193

def beach_scenario (total_sunglasses : ℕ) (total_caps : ℕ) (prob_cap_given_sunglasses : ℚ) : Prop :=
  ∃ (both : ℕ),
    total_sunglasses = 60 ∧
    total_caps = 40 ∧
    prob_cap_given_sunglasses = 1/3 ∧
    both ≤ total_sunglasses ∧
    both ≤ total_caps ∧
    (both : ℚ) / total_sunglasses = prob_cap_given_sunglasses

theorem beach_probability (total_sunglasses total_caps : ℕ) (prob_cap_given_sunglasses : ℚ) :
  beach_scenario total_sunglasses total_caps prob_cap_given_sunglasses →
  (∃ (both : ℕ), (both : ℚ) / total_caps = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_beach_probability_l1281_128193


namespace NUMINAMATH_CALUDE_johns_height_l1281_128122

/-- Given the heights of John, Lena, and Rebeca, prove John's height is 152 cm -/
theorem johns_height (john lena rebeca : ℕ) 
  (h1 : john = lena + 15)
  (h2 : john + 6 = rebeca)
  (h3 : lena + rebeca = 295) :
  john = 152 := by sorry

end NUMINAMATH_CALUDE_johns_height_l1281_128122


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_factorials_l1281_128131

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem greatest_prime_factor_of_sum_factorials :
  ∃ p : ℕ, is_prime p ∧ p ∣ (factorial 15 + factorial 17) ∧
    ∀ q : ℕ, is_prime q → q ∣ (factorial 15 + factorial 17) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_sum_factorials_l1281_128131


namespace NUMINAMATH_CALUDE_direct_proportion_conditions_l1281_128143

/-- A function representing a potential direct proportion -/
def f (k b x : ℝ) : ℝ := (k - 4) * x + b

/-- Definition of a direct proportion function -/
def is_direct_proportion (g : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, g x = m * x

/-- Theorem stating the necessary and sufficient conditions for f to be a direct proportion -/
theorem direct_proportion_conditions (k b : ℝ) :
  is_direct_proportion (f k b) ↔ k ≠ 4 ∧ b = 0 :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_conditions_l1281_128143


namespace NUMINAMATH_CALUDE_remainder_3045_div_32_l1281_128100

theorem remainder_3045_div_32 : 3045 % 32 = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_3045_div_32_l1281_128100


namespace NUMINAMATH_CALUDE_sum_equals_zero_l1281_128176

/-- The number of numbers satisfying: "there is no other number whose absolute value is equal to the absolute value of a" -/
def a : ℕ := sorry

/-- The number of numbers satisfying: "there is no other number whose square is equal to the square of b" -/
def b : ℕ := sorry

/-- The number of numbers satisfying: "there is no other number that, when multiplied by c, results in a product greater than 1" -/
def c : ℕ := sorry

theorem sum_equals_zero : a + b + c = 0 := by sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l1281_128176


namespace NUMINAMATH_CALUDE_total_teaching_years_l1281_128127

/-- The combined total of years taught by Virginia, Adrienne, and Dennis -/
def combinedYears (adrienne virginia dennis : ℕ) : ℕ := adrienne + virginia + dennis

theorem total_teaching_years :
  ∀ (adrienne virginia dennis : ℕ),
  virginia = adrienne + 9 →
  virginia = dennis - 9 →
  dennis = 43 →
  combinedYears adrienne virginia dennis = 102 :=
by
  sorry

end NUMINAMATH_CALUDE_total_teaching_years_l1281_128127


namespace NUMINAMATH_CALUDE_point_coordinates_l1281_128172

/-- Given a point P that is 2 units right and 4 units up from the origin (0,0),
    prove that the coordinates of P are (2,4). -/
theorem point_coordinates (P : ℝ × ℝ) 
  (h1 : P.1 = 2)  -- P is 2 units right from the origin
  (h2 : P.2 = 4)  -- P is 4 units up from the origin
  : P = (2, 4) := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_l1281_128172


namespace NUMINAMATH_CALUDE_equation_proof_l1281_128121

theorem equation_proof (a b : ℝ) (h : a - 2 * b = 4) : 3 - a + 2 * b = -1 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l1281_128121


namespace NUMINAMATH_CALUDE_center_value_is_31_l1281_128108

/-- An arithmetic sequence -/
def ArithmeticSequence (a : Fin 5 → ℕ) : Prop :=
  ∃ d, ∀ i : Fin 4, a (i + 1) = a i + d

/-- A 5x5 array where each row and column is an arithmetic sequence -/
def ArithmeticArray (A : Fin 5 → Fin 5 → ℕ) : Prop :=
  (∀ i, ArithmeticSequence (λ j => A i j)) ∧
  (∀ j, ArithmeticSequence (λ i => A i j))

theorem center_value_is_31 (A : Fin 5 → Fin 5 → ℕ) 
  (h_array : ArithmeticArray A)
  (h_first_row : A 0 0 = 1 ∧ A 0 4 = 25)
  (h_last_row : A 4 0 = 17 ∧ A 4 4 = 81) :
  A 2 2 = 31 := by
  sorry

end NUMINAMATH_CALUDE_center_value_is_31_l1281_128108


namespace NUMINAMATH_CALUDE_anya_hair_growth_l1281_128146

/-- The number of hairs Anya washes down the drain -/
def washed_hairs : ℕ := 32

/-- The number of hairs Anya brushes out -/
def brushed_hairs : ℕ := washed_hairs / 2

/-- The number of hairs Anya needs to grow back -/
def hairs_to_grow : ℕ := washed_hairs + brushed_hairs + 1

theorem anya_hair_growth :
  hairs_to_grow = 49 :=
by sorry

end NUMINAMATH_CALUDE_anya_hair_growth_l1281_128146


namespace NUMINAMATH_CALUDE_binomial_parameters_unique_l1281_128118

/-- A random variable following a binomial distribution -/
structure BinomialRandomVariable where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRandomVariable) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRandomVariable) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_parameters_unique 
  (ξ : BinomialRandomVariable) 
  (h_exp : expectation ξ = 2.4)
  (h_var : variance ξ = 1.44) : 
  ξ.n = 6 ∧ ξ.p = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_parameters_unique_l1281_128118


namespace NUMINAMATH_CALUDE_max_triangle_sum_l1281_128145

def triangle_numbers : Finset ℕ := {5, 6, 7, 8, 9, 10}

def is_valid_arrangement (a b c d e f : ℕ) : Prop :=
  a ∈ triangle_numbers ∧ b ∈ triangle_numbers ∧ c ∈ triangle_numbers ∧
  d ∈ triangle_numbers ∧ e ∈ triangle_numbers ∧ f ∈ triangle_numbers ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f

def side_sum (a b c : ℕ) : ℕ := a + b + c

def equal_sums (a b c d e f : ℕ) : Prop :=
  side_sum a b c = side_sum c d e ∧
  side_sum c d e = side_sum e f a

theorem max_triangle_sum :
  ∀ a b c d e f : ℕ,
    is_valid_arrangement a b c d e f →
    equal_sums a b c d e f →
    side_sum a b c ≤ 24 :=
sorry

end NUMINAMATH_CALUDE_max_triangle_sum_l1281_128145


namespace NUMINAMATH_CALUDE_heroes_on_large_sheets_l1281_128114

/-- Represents the number of pictures that can be drawn on a sheet of paper. -/
structure SheetCapacity where
  small : ℕ
  large : ℕ
  large_twice_small : large = 2 * small

/-- Represents the distribution of pictures drawn during the lunch break. -/
structure PictureDistribution where
  total : ℕ
  on_back : ℕ
  on_front : ℕ
  total_sum : total = on_back + on_front
  half_on_back : on_back = total / 2

/-- Represents the time spent drawing during the lunch break. -/
structure DrawingTime where
  break_duration : ℕ
  time_per_drawing : ℕ
  time_left : ℕ
  total_drawing_time : ℕ
  drawing_time_calc : total_drawing_time = break_duration - time_left

/-- The main theorem to prove. -/
theorem heroes_on_large_sheets
  (sheet_capacity : SheetCapacity)
  (picture_dist : PictureDistribution)
  (drawing_time : DrawingTime)
  (h1 : picture_dist.total = 20)
  (h2 : drawing_time.break_duration = 75)
  (h3 : drawing_time.time_per_drawing = 5)
  (h4 : drawing_time.time_left = 5)
  : ∃ (n : ℕ), n = 6 ∧ n * sheet_capacity.small = picture_dist.on_front / 2 :=
sorry

end NUMINAMATH_CALUDE_heroes_on_large_sheets_l1281_128114


namespace NUMINAMATH_CALUDE_absolute_value_difference_l1281_128191

theorem absolute_value_difference (a b : ℝ) : 
  (a < b → |a - b| = b - a) ∧ (a ≥ b → |a - b| = a - b) := by sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l1281_128191


namespace NUMINAMATH_CALUDE_min_side_length_with_integer_altitude_l1281_128113

theorem min_side_length_with_integer_altitude (a b c h x y : ℕ) :
  -- Triangle with integer side lengths
  (a > 0) ∧ (b > 0) ∧ (c > 0) →
  -- Altitude h divides side b into segments x and y
  (x + y = b) →
  -- Difference between segments is 7
  (y = x + 7) →
  -- Pythagorean theorem for altitude
  (a^2 - y^2 = c^2 - x^2) →
  -- Altitude is an integer
  (h^2 = a^2 - y^2) →
  -- b is the minimum side length
  (∀ b' : ℕ, b' < b → ¬∃ a' c' h' x' y' : ℕ,
    (a' > 0) ∧ (b' > 0) ∧ (c' > 0) ∧
    (x' + y' = b') ∧ (y' = x' + 7) ∧
    (a'^2 - y'^2 = c'^2 - x'^2) ∧
    (h'^2 = a'^2 - y'^2)) →
  -- Conclusion: minimum side length is 25
  b = 25 := by sorry

end NUMINAMATH_CALUDE_min_side_length_with_integer_altitude_l1281_128113


namespace NUMINAMATH_CALUDE_complex_modulus_l1281_128123

theorem complex_modulus (z : ℂ) : z + 2*I = (3 - I^3) / (1 + I) → Complex.abs z = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l1281_128123


namespace NUMINAMATH_CALUDE_student_marks_calculation_l1281_128104

/-- Calculates the marks obtained by a student who failed an exam. -/
theorem student_marks_calculation
  (total_marks : ℕ)
  (passing_percentage : ℚ)
  (failing_margin : ℕ)
  (h_total : total_marks = 500)
  (h_passing : passing_percentage = 40 / 100)
  (h_failing : failing_margin = 50) :
  (total_marks : ℚ) * passing_percentage - failing_margin = 150 :=
by sorry

end NUMINAMATH_CALUDE_student_marks_calculation_l1281_128104


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1281_128196

theorem trigonometric_identity (α β γ : ℝ) :
  3.400 * Real.cos (α + β) * Real.cos γ + Real.cos α + Real.cos β + Real.cos γ - Real.sin (α + β) * Real.sin γ =
  4 * Real.cos ((α + β) / 2) * Real.cos ((α + γ) / 2) * Real.cos ((β + γ) / 2) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1281_128196


namespace NUMINAMATH_CALUDE_f_3_equals_18_l1281_128194

def f : ℕ → ℕ
  | 0     => 3
  | (n+1) => (n+1) * f n

theorem f_3_equals_18 : f 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_18_l1281_128194


namespace NUMINAMATH_CALUDE_product_of_x_values_l1281_128110

theorem product_of_x_values (x : ℝ) : 
  (|12 / x + 3| = 2) → (∃ y : ℝ, (|12 / y + 3| = 2) ∧ x * y = 144 / 5) :=
by sorry

end NUMINAMATH_CALUDE_product_of_x_values_l1281_128110


namespace NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l1281_128182

/-- A set of four points in three-dimensional space. -/
structure FourPoints where
  p1 : ℝ × ℝ × ℝ
  p2 : ℝ × ℝ × ℝ
  p3 : ℝ × ℝ × ℝ
  p4 : ℝ × ℝ × ℝ

/-- Predicate to check if four points are non-coplanar. -/
def NonCoplanar (points : FourPoints) : Prop := sorry

/-- The number of planes determined by a set of four points. -/
def NumPlanesDetermined (points : FourPoints) : ℕ := sorry

/-- Theorem stating that four non-coplanar points determine exactly four planes. -/
theorem four_noncoplanar_points_determine_four_planes (points : FourPoints) :
  NonCoplanar points → NumPlanesDetermined points = 4 := by sorry

end NUMINAMATH_CALUDE_four_noncoplanar_points_determine_four_planes_l1281_128182


namespace NUMINAMATH_CALUDE_inequality_proof_l1281_128107

theorem inequality_proof :
  (∀ m n p : ℝ, m > n ∧ n > 0 ∧ p > 0 → n / m < (n + p) / (m + p)) ∧
  (∀ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b →
    c / (a + b) + a / (b + c) + b / (c + a) < 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1281_128107


namespace NUMINAMATH_CALUDE_koi_fish_problem_l1281_128140

theorem koi_fish_problem (num_koi : ℕ) (subtracted_num : ℕ) : 
  num_koi = 39 → 
  2 * num_koi - subtracted_num = 64 → 
  subtracted_num = 14 := by
  sorry

end NUMINAMATH_CALUDE_koi_fish_problem_l1281_128140


namespace NUMINAMATH_CALUDE_remainder_444_power_222_mod_13_l1281_128148

theorem remainder_444_power_222_mod_13 : 444^222 ≡ 1 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_444_power_222_mod_13_l1281_128148


namespace NUMINAMATH_CALUDE_unique_solution_condition_l1281_128130

/-- The equation has exactly one real solution if and only if a < 7/4 -/
theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x^3 - a*x^2 - 3*a*x + a^2 - 2 = 0) ↔ a < 7/4 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l1281_128130


namespace NUMINAMATH_CALUDE_fair_coin_five_tosses_l1281_128156

/-- The probability of a fair coin landing on the same side for all tosses -/
def same_side_probability (n : ℕ) : ℚ :=
  (1 / 2) ^ n

/-- Theorem: The probability of a fair coin landing on the same side for 5 tosses is 1/32 -/
theorem fair_coin_five_tosses :
  same_side_probability 5 = 1 / 32 := by
  sorry


end NUMINAMATH_CALUDE_fair_coin_five_tosses_l1281_128156


namespace NUMINAMATH_CALUDE_third_term_is_four_l1281_128177

/-- Given a sequence {a_n} where S_n is the sum of the first n terms -/
def S (n : ℕ) : ℕ := 2^n - 1

/-- The n-th term of the sequence -/
def a (n : ℕ) : ℕ := S n - S (n-1)

/-- Theorem: The third term of the sequence is 4 -/
theorem third_term_is_four : a 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_third_term_is_four_l1281_128177


namespace NUMINAMATH_CALUDE_angle_equality_l1281_128164

theorem angle_equality (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (20 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 25 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_equality_l1281_128164


namespace NUMINAMATH_CALUDE_intersection_uniqueness_l1281_128136

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := -2 * y = 7 * x - 3

/-- The intersection point -/
def intersection_point : ℚ × ℚ := (-3/17, 36/17)

theorem intersection_uniqueness :
  ∃! p : ℚ × ℚ, line1 p.1 p.2 ∧ line2 p.1 p.2 ∧ p = intersection_point := by
  sorry

end NUMINAMATH_CALUDE_intersection_uniqueness_l1281_128136


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1281_128129

def C : Finset Nat := {34, 35, 37, 41, 43}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ 
    (∀ (m : Nat), m ∈ C → (∃ (p : Nat), Nat.Prime p ∧ p ∣ n) → 
      (∃ (q : Nat), Nat.Prime q ∧ q ∣ m → p ≤ q)) ∧
    n = 34 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1281_128129


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l1281_128187

def miles_to_school : ℝ := 15
def miles_to_softball : ℝ := 6
def miles_to_restaurant : ℝ := 2
def miles_to_friend : ℝ := 4
def miles_to_home : ℝ := 11
def initial_gas : ℝ := 2

def total_miles : ℝ := miles_to_school + miles_to_softball + miles_to_restaurant + miles_to_friend + miles_to_home

theorem car_fuel_efficiency :
  total_miles / initial_gas = 19 := by sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l1281_128187


namespace NUMINAMATH_CALUDE_rect_to_cylindrical_l1281_128103

/-- Conversion from rectangular to cylindrical coordinates --/
theorem rect_to_cylindrical :
  let x : ℝ := -4
  let y : ℝ := -4 * Real.sqrt 3
  let z : ℝ := -3
  let r : ℝ := Real.sqrt (x^2 + y^2)
  let θ : ℝ := 4 * Real.pi / 3
  (r > 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) →
  (x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ z = z) :=
by sorry

end NUMINAMATH_CALUDE_rect_to_cylindrical_l1281_128103


namespace NUMINAMATH_CALUDE_bill_split_correct_l1281_128152

/-- The number of people splitting the bill -/
def num_people : ℕ := 9

/-- The total bill amount in cents -/
def total_bill : ℕ := 51416

/-- The amount each person should pay in cents, rounded to the nearest cent -/
def amount_per_person : ℕ := 5713

/-- Theorem stating that the calculated amount per person is correct -/
theorem bill_split_correct : 
  (total_bill + num_people - 1) / num_people = amount_per_person :=
sorry

end NUMINAMATH_CALUDE_bill_split_correct_l1281_128152


namespace NUMINAMATH_CALUDE_uniform_payment_proof_l1281_128132

theorem uniform_payment_proof :
  ∃ (x y : ℕ), 
    5 * x - 3 * y = 24 ∧ 
    x > 0 ∧ 
    y ≥ 0 ∧ 
    ∀ (x' y' : ℕ), 5 * x' - 3 * y' = 24 → x' > 0 → y' ≥ 0 → x ≤ x' ∧ y ≤ y' :=
by sorry

end NUMINAMATH_CALUDE_uniform_payment_proof_l1281_128132


namespace NUMINAMATH_CALUDE_equation_with_increasing_roots_l1281_128186

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x + 1) - (m + 1) / (x^2 + x) = (x + 1) / x

-- Define the concept of increasing roots
def has_increasing_roots (m : ℝ) : Prop :=
  ∃ x y : ℝ, x < y ∧ equation x m ∧ equation y m

-- Theorem statement
theorem equation_with_increasing_roots (m : ℝ) :
  has_increasing_roots m → m = -2 ∨ m = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_with_increasing_roots_l1281_128186


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l1281_128166

theorem prime_pairs_dividing_sum_of_powers (p q : ℕ) : 
  Prime p → Prime q → (p * q ∣ 2^p + 2^q) → 
  ((p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l1281_128166


namespace NUMINAMATH_CALUDE_double_average_l1281_128111

theorem double_average (n : ℕ) (original_avg : ℝ) (h1 : n = 10) (h2 : original_avg = 80) :
  let total_marks := n * original_avg
  let new_total_marks := 2 * total_marks
  let new_avg := new_total_marks / n
  new_avg = 160 := by
sorry

end NUMINAMATH_CALUDE_double_average_l1281_128111


namespace NUMINAMATH_CALUDE_problem_1_l1281_128165

theorem problem_1 (x : ℕ) : 2 * 8^x * 16^x = 2^22 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1281_128165


namespace NUMINAMATH_CALUDE_total_spent_is_520_l1281_128197

/-- Shopping expenses for Lisa and Carly -/
def shopping_expenses (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ) : Prop :=
  T_L = 40 ∧
  J_L = T_L / 2 ∧
  C_L = 2 * T_L ∧
  S_L = 3 * J_L ∧
  T_C = T_L / 4 ∧
  J_C = 3 * J_L ∧
  C_C = C_L / 2 ∧
  S_C = S_L ∧
  D_C = 2 * S_C ∧
  A_C = J_C / 2

/-- The total amount spent by Lisa and Carly -/
def total_spent (T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ) : ℝ :=
  T_L + J_L + C_L + S_L + T_C + J_C + C_C + S_C + D_C + A_C

/-- Theorem stating that the total amount spent is $520 -/
theorem total_spent_is_520 :
  ∀ T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C : ℝ,
  shopping_expenses T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C →
  total_spent T_L J_L C_L S_L T_C J_C C_C S_C D_C A_C = 520 :=
by sorry

end NUMINAMATH_CALUDE_total_spent_is_520_l1281_128197


namespace NUMINAMATH_CALUDE_main_divisors_equal_implies_equal_l1281_128199

/-- The two largest proper divisors of a composite natural number -/
def main_divisors (n : ℕ) : Set ℕ :=
  {d ∈ Nat.divisors n | d ≠ n ∧ d ≠ 1 ∧ ∀ k ∈ Nat.divisors n, k ≠ n → k ≠ 1 → d ≥ k}

/-- A natural number is composite if it has at least one proper divisor -/
def is_composite (n : ℕ) : Prop :=
  ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

theorem main_divisors_equal_implies_equal (a b : ℕ) 
  (ha : is_composite a) (hb : is_composite b) 
  (h : main_divisors a = main_divisors b) : 
  a = b :=
sorry

end NUMINAMATH_CALUDE_main_divisors_equal_implies_equal_l1281_128199


namespace NUMINAMATH_CALUDE_max_tickets_purchasable_l1281_128128

theorem max_tickets_purchasable (ticket_price : ℕ) (budget : ℕ) : 
  ticket_price = 15 → budget = 150 → 
  (∀ n : ℕ, n * ticket_price ≤ budget → n ≤ 10) ∧ 
  10 * ticket_price ≤ budget :=
by sorry

end NUMINAMATH_CALUDE_max_tickets_purchasable_l1281_128128


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l1281_128195

theorem line_segment_endpoint (y : ℝ) : 
  y > 0 → 
  ((1 - (-8))^2 + (y - 3)^2)^(1/2 : ℝ) = 15 → 
  y = 15 := by
sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l1281_128195


namespace NUMINAMATH_CALUDE_general_term_formula_min_value_S_min_value_n_l1281_128142

-- Define the sum of the first n terms
def S (n : ℕ) : ℤ := 2 * n^2 - 30 * n

-- Define the general term of the sequence
def a (n : ℕ) : ℤ := 4 * n - 32

-- Theorem for the general term
theorem general_term_formula : ∀ n : ℕ, a n = S n - S (n - 1) :=
sorry

-- Theorem for the minimum value of S_n
theorem min_value_S : ∃ n : ℕ, S n = -112 ∧ ∀ m : ℕ, S m ≥ -112 :=
sorry

-- Theorem for the values of n that give the minimum
theorem min_value_n : ∀ n : ℕ, S n = -112 ↔ (n = 7 ∨ n = 8) :=
sorry

end NUMINAMATH_CALUDE_general_term_formula_min_value_S_min_value_n_l1281_128142


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_difference_l1281_128150

theorem right_triangle_hypotenuse_difference (longer_side shorter_side hypotenuse : ℝ) : 
  hypotenuse = 17 →
  shorter_side = longer_side - 7 →
  longer_side^2 + shorter_side^2 = hypotenuse^2 →
  hypotenuse - longer_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_difference_l1281_128150


namespace NUMINAMATH_CALUDE_correct_average_l1281_128117

/-- Given 10 numbers with an initial average of 40.2, if one number is 17 greater than
    it should be and another number is 13 instead of 31, then the correct average is 40.3. -/
theorem correct_average (n : ℕ) (initial_avg : ℚ) (error1 error2 : ℚ) (correct2 : ℚ) :
  n = 10 →
  initial_avg = 40.2 →
  error1 = 17 →
  error2 = 13 →
  correct2 = 31 →
  (n : ℚ) * initial_avg - error1 - error2 + correct2 = n * 40.3 :=
by sorry

end NUMINAMATH_CALUDE_correct_average_l1281_128117


namespace NUMINAMATH_CALUDE_product_pricing_equation_l1281_128167

/-- 
Given a product with:
- Marked price of 1375 yuan
- Sold at 80% of the marked price
- Making a profit of 100 yuan
Prove that the equation relating the cost price x to these values is:
1375 * 80% = x + 100
-/
theorem product_pricing_equation (x : ℝ) : 
  1375 * (80 / 100) = x + 100 := by sorry

end NUMINAMATH_CALUDE_product_pricing_equation_l1281_128167


namespace NUMINAMATH_CALUDE_area_between_circles_l1281_128119

/-- The area of the region inside a large circle and outside eight congruent circles forming a ring --/
theorem area_between_circles (R : ℝ) (h : R = 40) : ∃ L : ℝ,
  (∃ (r : ℝ), 
    -- Eight congruent circles with radius r
    -- Each circle is externally tangent to its two adjacent circles
    -- All eight circles are internally tangent to a larger circle with radius R
    r > 0 ∧ r = R / 3 ∧
    -- L is the area of the region inside the large circle and outside all eight circles
    L = π * R^2 - 8 * π * r^2) ∧
  L = 1600 * π :=
sorry

end NUMINAMATH_CALUDE_area_between_circles_l1281_128119


namespace NUMINAMATH_CALUDE_jean_domino_friends_l1281_128173

theorem jean_domino_friends :
  ∀ (total_dominoes : ℕ) (dominoes_per_player : ℕ) (total_players : ℕ),
    total_dominoes = 28 →
    dominoes_per_player = 7 →
    total_players * dominoes_per_player = total_dominoes →
    total_players - 1 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_jean_domino_friends_l1281_128173


namespace NUMINAMATH_CALUDE_games_missed_l1281_128144

/-- Given a total number of soccer games and the number of games Jessica attended,
    calculate the number of games Jessica missed. -/
theorem games_missed (total_games attended_games : ℕ) : 
  total_games = 6 → attended_games = 2 → total_games - attended_games = 4 := by
  sorry

end NUMINAMATH_CALUDE_games_missed_l1281_128144


namespace NUMINAMATH_CALUDE_solve_for_x_l1281_128147

theorem solve_for_x (y : ℝ) (h1 : y = 1) (h2 : 4 * x - 2 * y + 3 = 3 * x + 3 * y) : x = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l1281_128147
